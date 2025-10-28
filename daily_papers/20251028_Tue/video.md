# 计算机视觉 cs.CV

- **最新发布 253 篇**

- **更新 150 篇**

## 最新发布

#### [new 001] Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation
- **分类: cs.CV; eess.SP**

- **简介: 该论文针对城市街景中移动激光扫描点云的高精度配准问题，提出融合自适应分段与基于平面体素的改进ICP方法。通过半球检查预处理减少轨迹漂移影响，利用平面体素优化配准效率与精度，显著提升大范围点云融合效果，适用于智能城市三维建模与动态监测。**

- **链接: [http://arxiv.org/pdf/2510.23416v1](http://arxiv.org/pdf/2510.23416v1)**

> **作者:** Marco Antonio Ortiz Rincon; Yihui Yang; Christoph Holst
>
> **备注:** 10 pages, 7 figures. This manuscript is currently under review at the International Journal of Applied Earth Observation and Geoinformation (Elsevier). A preprint version will also be available on SSRN (Elsevier Preprints) with a DOI once processed. This is the original preprint version submitted for peer review
>
> **摘要:** This study presents a novel workflow designed to efficiently and accurately register large-scale mobile laser scanning (MLS) point clouds to a target model point cloud in urban street scenarios. This workflow specifically targets the complexities inherent in urban environments and adeptly addresses the challenges of integrating point clouds that vary in density, noise characteristics, and occlusion scenarios, which are common in bustling city centers. Two methodological advancements are introduced. First, the proposed Semi-sphere Check (SSC) preprocessing technique optimally fragments MLS trajectory data by identifying mutually orthogonal planar surfaces. This step reduces the impact of MLS drift on the accuracy of the entire point cloud registration, while ensuring sufficient geometric features within each fragment to avoid local minima. Second, we propose Planar Voxel-based Generalized Iterative Closest Point (PV-GICP), a fine registration method that selectively utilizes planar surfaces within voxel partitions. This pre-process strategy not only improves registration accuracy but also reduces computation time by more than 50% compared to conventional point-to-plane ICP methods. Experiments on real-world datasets from Munich's inner city demonstrate that our workflow achieves sub-0.01 m average registration accuracy while significantly shortening processing times. The results underscore the potential of the proposed methods to advance automated 3D urban modeling and updating, with direct applications in urban planning, infrastructure management, and dynamic city monitoring.
>
---
#### [new 002] Implicit Modeling for Transferability Estimation of Vision Foundation Models
- **分类: cs.CV**

- **简介: 该论文针对视觉基础模型的迁移能力评估问题，提出隐式迁移建模（ITM）框架，通过隐式建模模型内在可迁移性与分而治之变分近似策略，高效估算不同模型在下游任务中的表现，显著提升评估的准确性、稳定性和效率。**

- **链接: [http://arxiv.org/pdf/2510.23145v1](http://arxiv.org/pdf/2510.23145v1)**

> **作者:** Yaoyan Zheng; Huiqun Wang; Nan Zhou; Di Huang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Transferability estimation identifies the best pre-trained models for downstream tasks without incurring the high computational cost of full fine-tuning. This capability facilitates deployment and advances the pre-training and fine-tuning paradigm. However, existing methods often struggle to accurately assess transferability for emerging pre-trained models with diverse architectures, training strategies, and task alignments. In this work, we propose Implicit Transferability Modeling (ITM), a novel framework that implicitly models each model's intrinsic transferability, coupled with a Divide-and-Conquer Variational Approximation (DVA) strategy to efficiently approximate embedding space evolution. This design enables generalization across a broader range of models and downstream tasks. Extensive experiments on a comprehensive benchmark--spanning extensive training regimes and a wider variety of model types--demonstrate that ITM consistently outperforms existing methods in terms of stability, effectiveness, and efficiency.
>
---
#### [new 003] Finding 3D Scene Analogies with Multimodal Foundation Models
- **分类: cs.CV**

- **简介: 该论文研究3D场景类比任务，旨在零样本、开放词汇下自动匹配两组3D场景的对应区域。针对现有方法需额外训练和固定物体词表的问题，提出融合视觉-语言模型与3D形状基础模型的混合表示，通过粗到精的对齐策略实现精准场景映射，支持轨迹与目标点的跨场景迁移。**

- **链接: [http://arxiv.org/pdf/2510.23184v1](http://arxiv.org/pdf/2510.23184v1)**

> **作者:** Junho Kim; Young Min Kim
>
> **备注:** Accepted to FM4RoboPlan workshop at RSS 2025
>
> **摘要:** Connecting current observations with prior experiences helps robots adapt and plan in new, unseen 3D environments. Recently, 3D scene analogies have been proposed to connect two 3D scenes, which are smooth maps that align scene regions with common spatial relationships. These maps enable detailed transfer of trajectories or waypoints, potentially supporting demonstration transfer for imitation learning or task plan transfer across scenes. However, existing methods for the task require additional training and fixed object vocabularies. In this work, we propose to use multimodal foundation models for finding 3D scene analogies in a zero-shot, open-vocabulary setting. Central to our approach is a hybrid neural representation of scenes that consists of a sparse graph based on vision-language model features and a feature field derived from 3D shape foundation models. 3D scene analogies are then found in a coarse-to-fine manner, by first aligning the graph and refining the correspondence with feature fields. Our method can establish accurate correspondences between complex scenes, and we showcase applications in trajectory and waypoint transfer.
>
---
#### [new 004] AesCrop: Aesthetic-driven Cropping Guided by Composition
- **分类: cs.CV**

- **简介: 该论文聚焦于美学驱动的图像裁剪任务，旨在提升裁剪结果的视觉吸引力。针对现有方法在全局性与多样性上的不足，提出AesCrop模型，通过引入Mamba Composition Attention Bias，显式融合构图指导，实现端到端多裁剪生成与评分，显著提升质量与美感。**

- **链接: [http://arxiv.org/pdf/2510.22528v1](http://arxiv.org/pdf/2510.22528v1)**

> **作者:** Yen-Hong Wong; Lai-Kuan Wong
>
> **备注:** Accepted at the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, 2025
>
> **摘要:** Aesthetic-driven image cropping is crucial for applications like view recommendation and thumbnail generation, where visual appeal significantly impacts user engagement. A key factor in visual appeal is composition--the deliberate arrangement of elements within an image. Some methods have successfully incorporated compositional knowledge through evaluation-based and regression-based paradigms. However, evaluation-based methods lack globality while regression-based methods lack diversity. Recently, hybrid approaches that integrate both paradigms have emerged, bridging the gap between these two to achieve better diversity and globality. Notably, existing hybrid methods do not incorporate photographic composition guidance, a key attribute that defines photographic aesthetics. In this work, we introduce AesCrop, a composition-aware hybrid image-cropping model that integrates a VMamba image encoder, augmented with a novel Mamba Composition Attention Bias (MCAB) and a transformer decoder to perform end-to-end rank-based image cropping, generating multiple crops along with the corresponding quality scores. By explicitly encoding compositional cues into the attention mechanism, MCAB directs AesCrop to focus on the most compositionally salient regions. Extensive experiments demonstrate that AesCrop outperforms current state-of-the-art methods, delivering superior quantitative metrics and qualitatively more pleasing crops.
>
---
#### [new 005] AG-Fusion: adaptive gated multimodal fusion for 3d object detection in complex scenes
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对复杂场景下3D目标检测中多模态融合鲁棒性差的问题，提出AG-Fusion方法。通过自适应门控机制融合相机与激光雷达特征，在统一BEV空间中增强可靠性，显著提升在恶劣环境下的检测性能，并构建Excavator3D数据集进行验证。**

- **链接: [http://arxiv.org/pdf/2510.23151v1](http://arxiv.org/pdf/2510.23151v1)**

> **作者:** Sixian Liu; Chen Xu; Qiang Wang; Donghai Shi; Yiwen Li
>
> **摘要:** Multimodal camera-LiDAR fusion technology has found extensive application in 3D object detection, demonstrating encouraging performance. However, existing methods exhibit significant performance degradation in challenging scenarios characterized by sensor degradation or environmental disturbances. We propose a novel Adaptive Gated Fusion (AG-Fusion) approach that selectively integrates cross-modal knowledge by identifying reliable patterns for robust detection in complex scenes. Specifically, we first project features from each modality into a unified BEV space and enhance them using a window-based attention mechanism. Subsequently, an adaptive gated fusion module based on cross-modal attention is designed to integrate these features into reliable BEV representations robust to challenging environments. Furthermore, we construct a new dataset named Excavator3D (E3D) focusing on challenging excavator operation scenarios to benchmark performance in complex conditions. Our method not only achieves competitive performance on the standard KITTI dataset with 93.92% accuracy, but also significantly outperforms the baseline by 24.88% on the challenging E3D dataset, demonstrating superior robustness to unreliable modal information in complex industrial scenes.
>
---
#### [new 006] A Fully Interpretable Statistical Approach for Roadside LiDAR Background Subtraction
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶中路边激光雷达背景消减任务，提出一种可解释的统计方法。通过构建高斯分布网格建模背景，并设计过滤算法实现点云分类，有效区分前景与背景点。方法灵活适配多种激光雷达，仅需少量背景数据即可高精度运行，且可在低资源设备上高效部署。**

- **链接: [http://arxiv.org/pdf/2510.22390v1](http://arxiv.org/pdf/2510.22390v1)**

> **作者:** Aitor Iglesias; Nerea Aranjuelo; Patricia Javierre; Ainhoa Menendez; Ignacio Arganda-Carreras; Marcos Nieto
>
> **摘要:** We present a fully interpretable and flexible statistical method for background subtraction in roadside LiDAR data, aimed at enhancing infrastructure-based perception in automated driving. Our approach introduces both a Gaussian distribution grid (GDG), which models the spatial statistics of the background using background-only scans, and a filtering algorithm that uses this representation to classify LiDAR points as foreground or background. The method supports diverse LiDAR types, including multiline 360 degree and micro-electro-mechanical systems (MEMS) sensors, and adapts to various configurations. Evaluated on the publicly available RCooper dataset, it outperforms state-of-the-art techniques in accuracy and flexibility, even with minimal background data. Its efficient implementation ensures reliable performance on low-resource hardware, enabling scalable real-world deployment.
>
---
#### [new 007] STATUS Bench: A Rigorous Benchmark for Evaluating Object State Understanding in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文针对视觉语言模型在物体状态理解上的能力不足问题，提出STATUS Bench基准与STATUS Train数据集。通过三任务联合评估（状态识别、图像检索、状态变化识别），揭示现有模型在细微状态区分上表现不佳，强调了高质量评测与训练数据对提升物体状态理解的重要性。**

- **链接: [http://arxiv.org/pdf/2510.22571v1](http://arxiv.org/pdf/2510.22571v1)**

> **作者:** Mahiro Ukai; Shuhei Kurita; Nakamasa Inoue
>
> **摘要:** Object state recognition aims to identify the specific condition of objects, such as their positional states (e.g., open or closed) and functional states (e.g., on or off). While recent Vision-Language Models (VLMs) are capable of performing a variety of multimodal tasks, it remains unclear how precisely they can identify object states. To alleviate this issue, we introduce the STAte and Transition UnderStanding Benchmark (STATUS Bench), the first benchmark for rigorously evaluating the ability of VLMs to understand subtle variations in object states in diverse situations. Specifically, STATUS Bench introduces a novel evaluation scheme that requires VLMs to perform three tasks simultaneously: object state identification (OSI), image retrieval (IR), and state change identification (SCI). These tasks are defined over our fully hand-crafted dataset involving image pairs, their corresponding object state descriptions and state change descriptions. Furthermore, we introduce a large-scale training dataset, namely STATUS Train, which consists of 13 million semi-automatically created descriptions. This dataset serves as the largest resource to facilitate further research in this area. In our experiments, we demonstrate that STATUS Bench enables rigorous consistency evaluation and reveal that current state-of-the-art VLMs still significantly struggle to capture subtle object state distinctions. Surprisingly, under the proposed rigorous evaluation scheme, most open-weight VLMs exhibited chance-level zero-shot performance. After fine-tuning on STATUS Train, Qwen2.5-VL achieved performance comparable to Gemini 2.0 Flash. These findings underscore the necessity of STATUS Bench and Train for advancing object state recognition in VLM research.
>
---
#### [new 008] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations
- **分类: cs.CV**

- **简介: 该论文提出Concerto，一种联合2D-3D自监督学习框架，用于空间认知建模。旨在解决单一模态表征不完整问题，通过跨模态联合嵌入与自蒸馏，提升3D场景理解的几何与语义一致性。在多个基准上超越现有方法，实现更优的空间表征学习。**

- **链接: [http://arxiv.org/pdf/2510.23607v1](http://arxiv.org/pdf/2510.23607v1)**

> **作者:** Yujia Zhang; Xiaoyang Wu; Yixing Lao; Chengyao Wang; Zhuotao Tian; Naiyan Wang; Hengshuang Zhao
>
> **备注:** NeurIPS 2025, produced by Pointcept, project page: https://pointcept.github.io/Concerto
>
> **摘要:** Humans learn abstract concepts through multisensory synergy, and once formed, such representations can often be recalled from a single modality. Inspired by this principle, we introduce Concerto, a minimalist simulation of human concept learning for spatial cognition, combining 3D intra-modal self-distillation with 2D-3D cross-modal joint embedding. Despite its simplicity, Concerto learns more coherent and informative spatial features, as demonstrated by zero-shot visualizations. It outperforms both standalone SOTA 2D and 3D self-supervised models by 14.2% and 4.8%, respectively, as well as their feature concatenation, in linear probing for 3D scene perception. With full fine-tuning, Concerto sets new SOTA results across multiple scene understanding benchmarks (e.g., 80.7% mIoU on ScanNet). We further present a variant of Concerto tailored for video-lifted point cloud spatial understanding, and a translator that linearly projects Concerto representations into CLIP's language space, enabling open-world perception. These results highlight that Concerto emerges spatial representations with superior fine-grained geometric and semantic consistency.
>
---
#### [new 009] On the Faithfulness of Visual Thinking: Measurement and Enhancement
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉思维的忠实性问题，针对大视觉语言模型在强化学习微调后生成的多模态链式思维中视觉信息不可靠、不充分的问题。提出自动化评估指标与无标注的SCCM学习策略，提升视觉成分的可靠性与充分性，显著增强视觉推理的忠实性。**

- **链接: [http://arxiv.org/pdf/2510.23482v1](http://arxiv.org/pdf/2510.23482v1)**

> **作者:** Zujing Liu; Junwen Pan; Qi She; Yuan Gao; Guisong Xia
>
> **摘要:** Recent large vision-language models (LVLMs) can generate vision-text multimodal chain-of-thought (MCoT) traces after reinforcement fine-tuning (RFT). However, we observe that the visual information incorporated in MCoT is often inaccurate, though still yield correct answers, indicating a lack of faithfulness in the MCoT reasoning process. We attribute this unfaithfulness to the RL reward in RFT, which solely incentivizes the format of interleaved vision-text cues, ie, it encourages the model to incorporate visual information into its text reasoning steps without considering the correctness of the visual information. In this paper, we first probe the faithfulness of MCoT by measuring how much the prediction changes when its visual and textual thoughts are intervened. Surprisingly, the model's predictions remain nearly unchanged under visual intervention but change significantly under textual intervention, indicating that the visual evidence is largely ignored. To further analyze visual information, we introduce an automated LVLM-based evaluation metric that quantifies the faithfulness of visual cues from two perspectives: reliability and sufficiency. Our evaluation reveals that the visual information in current MCoT traces is simultaneously unreliable and insufficient. To address this issue, we propose a novel MCoT learning strategy termed Sufficient-Component Cause Model (SCCM) learning. This approach encourages the MCoT to generate sufficient yet minimal visual components that are independently capable of leading to correct answers. We note that the proposed SCCM is annotation-free and compatible with various RFT for MCoT in a plug-and-play manner. Empirical results demonstrate that SCCM consistently improves the visual faithfulness across a suite of fine-grained perception and reasoning benchmarks. Code is available at https://github.com/EugeneLiu01/Faithful_Thinking_with_Image.
>
---
#### [new 010] VLM-SlideEval: Evaluating VLMs on Structured Comprehension and Perturbation Sensitivity in PPT
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VLM-SlideEval框架，用于评估视觉语言模型在幻灯片上的结构理解与抗干扰能力。针对现有VLM在幻灯片内容理解上缺乏系统评测的问题，研究构建了统一标注标准，从元素提取、扰动鲁棒性、叙事结构恢复三方面进行测试，发现当前VLM在像素级提取和叙事理解上表现不足，推动构建更可靠的智能评估系统。**

- **链接: [http://arxiv.org/pdf/2510.22045v1](http://arxiv.org/pdf/2510.22045v1)**

> **作者:** Hyeonsu Kang; Emily Bao; Anjan Goswami
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Evaluating the Evolving LLM Lifecycle - Benchmarks, Emergent Abilities, and Scaling
>
> **摘要:** Vision-language models (VLMs) are increasingly used to evaluate multimodal content, including presentation slides, yet their slide-specific understanding remains underexplored {despite their growing role as critics in agentic, model-forward pipelines}. We introduce VLM-SlideEval, an evaluation framework that probes VLMs along three axes: (1) element-level extraction from slide images aligned to ground truth; (2) robustness to controlled perturbations in geometry, style, and text; and (3) higher-level comprehension, such as recovering a deck's narrative order from shuffled slides. Using publicly available decks from Zenodo (https://huggingface.co/datasets/Forceless/Zenodo10K/viewer/default/pptx), we standardize ground-truth element metadata from PowerPoint XML and live renderings into a unified, verifiable schema. Empirically, VLMs underperform on pixel-accurate extraction and show non-trivial agreement, fidelity, and consistency under controlled perturbations, while performing better on single-slide content understanding; however, they do not reliably capture narrative structure across slides. These results highlight the limits of current VLMs for slide evaluation and motivate calibrated, critic-in-the-loop evaluators that drive iterative refinement and selection in agentic pipelines.
>
---
#### [new 011] Symmetria: A Synthetic Dataset for Learning in Point Clouds
- **分类: cs.CV**

- **简介: 该论文提出Symmetria，一个基于对称性生成的点云合成数据集，解决点云学习中高质量数据稀缺问题。通过公式化生成确保精确标注，支持自监督预训练与少样本学习，提升分类、分割等任务性能，并提供对称性检测基准，促进点云领域研究。**

- **链接: [http://arxiv.org/pdf/2510.23414v1](http://arxiv.org/pdf/2510.23414v1)**

> **作者:** Ivan Sipiran; Gustavo Santelices; Lucas Oyarzún; Andrea Ranieri; Chiara Romanengo; Silvia Biasotti; Bianca Falcidieno
>
> **备注:** 40 pages
>
> **摘要:** Unlike image or text domains that benefit from an abundance of large-scale datasets, point cloud learning techniques frequently encounter limitations due to the scarcity of extensive datasets. To overcome this limitation, we present Symmetria, a formula-driven dataset that can be generated at any arbitrary scale. By construction, it ensures the absolute availability of precise ground truth, promotes data-efficient experimentation by requiring fewer samples, enables broad generalization across diverse geometric settings, and offers easy extensibility to new tasks and modalities. Using the concept of symmetry, we create shapes with known structure and high variability, enabling neural networks to learn point cloud features effectively. Our results demonstrate that this dataset is highly effective for point cloud self-supervised pre-training, yielding models with strong performance in downstream tasks such as classification and segmentation, which also show good few-shot learning capabilities. Additionally, our dataset can support fine-tuning models to classify real-world objects, highlighting our approach's practical utility and application. We also introduce a challenging task for symmetry detection and provide a benchmark for baseline comparisons. A significant advantage of our approach is the public availability of the dataset, the accompanying code, and the ability to generate very large collections, promoting further research and innovation in point cloud learning.
>
---
#### [new 012] Semantic Relation-Enhanced CLIP Adapter for Domain Adaptive Zero-Shot Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对领域自适应零样本学习（DAZSL）任务，解决现有方法在跨类别知识迁移与跨模态对齐上的不足。提出SRE-CLIP框架，引入语义关系结构损失与跨模态对齐保持策略，增强CLIP模型在目标域的泛化能力，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.21808v1](http://arxiv.org/pdf/2510.21808v1)**

> **作者:** Jiaao Yu; Mingjie Han; Jinkun Jiang; Junyu Dong; Tao Gong; Man Lan
>
> **备注:** 5 pages
>
> **摘要:** The high cost of data annotation has spurred research on training deep learning models in data-limited scenarios. Existing paradigms, however, fail to balance cross-domain transfer and cross-category generalization, giving rise to the demand for Domain-Adaptive Zero-Shot Learning (DAZSL). Although vision-language models (e.g., CLIP) have inherent advantages in the DAZSL field, current studies do not fully exploit their potential. Applying CLIP to DAZSL faces two core challenges: inefficient cross-category knowledge transfer due to the lack of semantic relation guidance, and degraded cross-modal alignment during target domain fine-tuning. To address these issues, we propose a Semantic Relation-Enhanced CLIP (SRE-CLIP) Adapter framework, integrating a Semantic Relation Structure Loss and a Cross-Modal Alignment Retention Strategy. As the first CLIP-based DAZSL method, SRE-CLIP achieves state-of-the-art performance on the I2AwA and I2WebV benchmarks, significantly outperforming existing approaches.
>
---
#### [new 013] Modal Aphasia: Can Unified Multimodal Models Describe Images From Memory?
- **分类: cs.CV; cs.CR**

- **简介: 该论文提出“模态失语”现象：统一多模态模型能准确记忆图像但无法用文字描述。研究揭示当前模型在图文联合训练下存在视觉与语言能力的系统性分离，暴露安全漏洞——文本对齐模型仍可生成危险图像。工作包括实证分析与合成数据实验，证实该现象为模型本质属性。**

- **链接: [http://arxiv.org/pdf/2510.21842v1](http://arxiv.org/pdf/2510.21842v1)**

> **作者:** Michael Aerni; Joshua Swanson; Kristina Nikolić; Florian Tramèr
>
> **摘要:** We present modal aphasia, a systematic dissociation in which current unified multimodal models accurately memorize concepts visually but fail to articulate them in writing, despite being trained on images and text simultaneously. For one, we show that leading frontier models can generate near-perfect reproductions of iconic movie artwork, but confuse crucial details when asked for textual descriptions. We corroborate those findings through controlled experiments on synthetic datasets in multiple architectures. Our experiments confirm that modal aphasia reliably emerges as a fundamental property of current unified multimodal models, not just as a training artifact. In practice, modal aphasia can introduce vulnerabilities in AI safety frameworks, as safeguards applied to one modality may leave harmful concepts accessible in other modalities. We demonstrate this risk by showing how a model aligned solely on text remains capable of generating unsafe images.
>
---
#### [new 014] VoMP: Predicting Volumetric Mechanical Property Fields
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文提出VoMP，一种预测3D物体体积内杨氏模量、泊松比和密度的前馈方法。针对物理仿真中机械属性需手工设计的问题，通过多视角特征聚合与几何变换器，从可渲染体素化表示中预测物理合理材料属性，构建了基于真实数据的训练管道与新基准，显著提升精度与效率。**

- **链接: [http://arxiv.org/pdf/2510.22975v1](http://arxiv.org/pdf/2510.22975v1)**

> **作者:** Rishit Dagli; Donglai Xiang; Vismay Modi; Charles Loop; Clement Fuji Tsang; Anka He Chen; Anita Hu; Gavriel State; David I. W. Levin; Maria Shugrina
>
> **备注:** hi-res paper and other details at: https://research.nvidia.com/labs/sil/projects/vomp
>
> **摘要:** Physical simulation relies on spatially-varying mechanical properties, often laboriously hand-crafted. VoMP is a feed-forward method trained to predict Young's modulus ($E$), Poisson's ratio ($\nu$), and density ($\rho$) throughout the volume of 3D objects, in any representation that can be rendered and voxelized. VoMP aggregates per-voxel multi-view features and passes them to our trained Geometry Transformer to predict per-voxel material latent codes. These latents reside on a manifold of physically plausible materials, which we learn from a real-world dataset, guaranteeing the validity of decoded per-voxel materials. To obtain object-level training data, we propose an annotation pipeline combining knowledge from segmented 3D datasets, material databases, and a vision-language model, along with a new benchmark. Experiments show that VoMP estimates accurate volumetric properties, far outperforming prior art in accuracy and speed.
>
---
#### [new 015] VR-Drive: Viewpoint-Robust End-to-End Driving with Feed-Forward 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出VR-Drive，一种面向视角鲁棒的端到端自动驾驶框架。针对不同摄像头视角导致的性能下降问题，通过联合学习3D场景重建与规划，实现感知-规划一体化。采用前馈式3D高斯点云渲染、视角混合记忆库与一致性知识蒸馏，提升多视角下系统鲁棒性，并发布新基准数据集评估性能。**

- **链接: [http://arxiv.org/pdf/2510.23205v1](http://arxiv.org/pdf/2510.23205v1)**

> **作者:** Hoonhee Cho; Jae-Young Kang; Giwon Lee; Hyemin Yang; Heejun Park; Seokwoo Jung; Kuk-Jin Yoon
>
> **备注:** Accepted by NeurIPS2025
>
> **摘要:** End-to-end autonomous driving (E2E-AD) has emerged as a promising paradigm that unifies perception, prediction, and planning into a holistic, data-driven framework. However, achieving robustness to varying camera viewpoints, a common real-world challenge due to diverse vehicle configurations, remains an open problem. In this work, we propose VR-Drive, a novel E2E-AD framework that addresses viewpoint generalization by jointly learning 3D scene reconstruction as an auxiliary task to enable planning-aware view synthesis. Unlike prior scene-specific synthesis approaches, VR-Drive adopts a feed-forward inference strategy that supports online training-time augmentation from sparse views without additional annotations. To further improve viewpoint consistency, we introduce a viewpoint-mixed memory bank that facilitates temporal interaction across multiple viewpoints and a viewpoint-consistent distillation strategy that transfers knowledge from original to synthesized views. Trained in a fully end-to-end manner, VR-Drive effectively mitigates synthesis-induced noise and improves planning under viewpoint shifts. In addition, we release a new benchmark dataset to evaluate E2E-AD performance under novel camera viewpoints, enabling comprehensive analysis. Our results demonstrate that VR-Drive is a scalable and robust solution for the real-world deployment of end-to-end autonomous driving systems.
>
---
#### [new 016] GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对扩散模型中基于GRPO的强化学习出现的隐式过优化问题，提出GRPO-Guard方法。通过比率归一化与梯度重加权，恢复重要性比分布平衡，实现有效剪裁，稳定训练并提升生成质量，无需依赖重型KL正则。**

- **链接: [http://arxiv.org/pdf/2510.22319v1](http://arxiv.org/pdf/2510.22319v1)**

> **作者:** Jing Wang; Jiajun Liang; Jie Liu; Henglin Liu; Gongye Liu; Jun Zheng; Wanyuan Pang; Ao Ma; Zhenyu Xie; Xintao Wang; Meng Wang; Pengfei Wan; Xiaodan Liang
>
> **摘要:** Recently, GRPO-based reinforcement learning has shown remarkable progress in optimizing flow-matching models, effectively improving their alignment with task-specific rewards. Within these frameworks, the policy update relies on importance-ratio clipping to constrain overconfident positive and negative gradients. However, in practice, we observe a systematic shift in the importance-ratio distribution-its mean falls below 1 and its variance differs substantially across timesteps. This left-shifted and inconsistent distribution prevents positive-advantage samples from entering the clipped region, causing the mechanism to fail in constraining overconfident positive updates. As a result, the policy model inevitably enters an implicit over-optimization stage-while the proxy reward continues to increase, essential metrics such as image quality and text-prompt alignment deteriorate sharply, ultimately making the learned policy impractical for real-world use. To address this issue, we introduce GRPO-Guard, a simple yet effective enhancement to existing GRPO frameworks. Our method incorporates ratio normalization, which restores a balanced and step-consistent importance ratio, ensuring that PPO clipping properly constrains harmful updates across denoising timesteps. In addition, a gradient reweighting strategy equalizes policy gradients over noise conditions, preventing excessive updates from particular timestep regions. Together, these designs act as a regulated clipping mechanism, stabilizing optimization and substantially mitigating implicit over-optimization without relying on heavy KL regularization. Extensive experiments on multiple diffusion backbones (e.g., SD3.5M, Flux.1-dev) and diverse proxy tasks demonstrate that GRPO-Guard significantly reduces over-optimization while maintaining or even improving generation quality.
>
---
#### [new 017] A Multi-Stage Hybrid Framework for Automated Interpretation of Multi-View Engineering Drawings Using Vision Language Model
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文针对2D多视图工程图自动解读任务，解决布局复杂、符号文本混合导致的识别难题。提出三阶段混合框架：布局分割、定向细粒度检测、OCR-free语义解析，利用视觉语言模型实现文本与数值信息精准提取，输出统一JSON格式，支持CAD与制造系统集成。**

- **链接: [http://arxiv.org/pdf/2510.21862v1](http://arxiv.org/pdf/2510.21862v1)**

> **作者:** Muhammad Tayyab Khan; Zane Yong; Lequn Chen; Wenhe Feng; Nicholas Yew Jin Tan; Seung Ki Moon
>
> **备注:** This draft has been submitted to the 13th International Conference on Industrial Engineering and Applications (ICIEA 2026)
>
> **摘要:** Engineering drawings are fundamental to manufacturing communication, serving as the primary medium for conveying design intent, tolerances, and production details. However, interpreting complex multi-view drawings with dense annotations remains challenging using manual methods, generic optical character recognition (OCR) systems, or traditional deep learning approaches, due to varied layouts, orientations, and mixed symbolic-textual content. To address these challenges, this paper proposes a three-stage hybrid framework for the automated interpretation of 2D multi-view engineering drawings using modern detection and vision language models (VLMs). In the first stage, YOLOv11-det performs layout segmentation to localize key regions such as views, title blocks, and notes. The second stage uses YOLOv11-obb for orientation-aware, fine-grained detection of annotations, including measures, GD&T symbols, and surface roughness indicators. The third stage employs two Donut-based, OCR-free VLMs for semantic content parsing: the Alphabetical VLM extracts textual and categorical information from title blocks and notes, while the Numerical VLM interprets quantitative data such as measures, GD&T frames, and surface roughness. Two specialized datasets were developed to ensure robustness and generalization: 1,000 drawings for layout detection and 1,406 for annotation-level training. The Alphabetical VLM achieved an overall F1 score of 0.672, while the Numerical VLM reached 0.963, demonstrating strong performance in textual and quantitative interpretation, respectively. The unified JSON output enables seamless integration with CAD and manufacturing databases, providing a scalable solution for intelligent engineering drawing analysis.
>
---
#### [new 018] FairJudge: MLLM Judging for Social Attributes and Prompt Image Alignment
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出FairJudge，一种用于评估文本到图像生成模型在社会属性和提示对齐方面的公平性评价协议。针对现有方法依赖表面特征、缺乏可解释性和对弱可见属性敏感度低的问题，采用多模态大模型作为公正裁判，基于证据进行[-1,1]评分并支持拒答，提升评估的可信度与可复现性。**

- **链接: [http://arxiv.org/pdf/2510.22827v1](http://arxiv.org/pdf/2510.22827v1)**

> **作者:** Zahraa Al Sahili; Maryam Fetanat; Maimuna Nowaz; Ioannis Patras; Matthew Purver
>
> **摘要:** Text-to-image (T2I) systems lack simple, reproducible ways to evaluate how well images match prompts and how models treat social attributes. Common proxies -- face classifiers and contrastive similarity -- reward surface cues, lack calibrated abstention, and miss attributes only weakly visible (for example, religion, culture, disability). We present FairJudge, a lightweight protocol that treats instruction-following multimodal LLMs as fair judges. It scores alignment with an explanation-oriented rubric mapped to [-1, 1]; constrains judgments to a closed label set; requires evidence grounded in the visible content; and mandates abstention when cues are insufficient. Unlike CLIP-only pipelines, FairJudge yields accountable, evidence-aware decisions; unlike mitigation that alters generators, it targets evaluation fairness. We evaluate gender, race, and age on FairFace, PaTA, and FairCoT; extend to religion, culture, and disability; and assess profession correctness and alignment on IdenProf, FairCoT-Professions, and our new DIVERSIFY-Professions. We also release DIVERSIFY, a 469-image corpus of diverse, non-iconic scenes. Across datasets, judge models outperform contrastive and face-centric baselines on demographic prediction and improve mean alignment while maintaining high profession accuracy, enabling more reliable, reproducible fairness audits.
>
---
#### [new 019] CogStereo: Neural Stereo Matching with Implicit Spatial Cognition Embedding
- **分类: cs.CV**

- **简介: 该论文提出CogStereo，一种基于隐式空间认知嵌入的神经立体匹配框架。针对现有方法在弱纹理、遮挡等区域零样本泛化能力差的问题，利用单目深度特征作为先验，融合像素不确定性与认知引导特征，实现全局一致的视差估计，显著提升跨域泛化性能。**

- **链接: [http://arxiv.org/pdf/2510.22119v1](http://arxiv.org/pdf/2510.22119v1)**

> **作者:** Lihuang Fang; Xiao Hu; Yuchen Zou; Hong Zhang
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Deep stereo matching has advanced significantly on benchmark datasets through fine-tuning but falls short of the zero-shot generalization seen in foundation models in other vision tasks. We introduce CogStereo, a novel framework that addresses challenging regions, such as occlusions or weak textures, without relying on dataset-specific priors. CogStereo embeds implicit spatial cognition into the refinement process by using monocular depth features as priors, capturing holistic scene understanding beyond local correspondences. This approach ensures structurally coherent disparity estimation, even in areas where geometry alone is inadequate. CogStereo employs a dual-conditional refinement mechanism that combines pixel-wise uncertainty with cognition-guided features for consistent global correction of mismatches. Extensive experiments on Scene Flow, KITTI, Middlebury, ETH3D, EuRoc, and real-world demonstrate that CogStereo not only achieves state-of-the-art results but also excels in cross-domain generalization, shifting stereo vision towards a cognition-driven approach.
>
---
#### [new 020] Prompt fidelity of ChatGPT4o / Dall-E3 text-to-image visualisations
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究ChatGPT4o与DALL-E3生成图像的提示保真度，聚焦文本到图像任务中属性一致性问题。通过分析200+230张图像，评估年龄、着装、配饰等属性的准确呈现，发现DALL-E3在15.6%的属性上出现偏差，尤其在人物年龄描述上误差最高，揭示了模型在细节还原上的局限性。**

- **链接: [http://arxiv.org/pdf/2510.21821v1](http://arxiv.org/pdf/2510.21821v1)**

> **作者:** Dirk HR Spennemann
>
> **摘要:** This study examines the prompt fidelity of ChatGPT4o / DALL-E3 text-to-image visualisations by analysing whether attributes explicitly specified in autogenously generated prompts are correctly rendered in the resulting images. Using two public-domain datasets comprising 200 visualisations of women working in the cultural and creative industries and 230 visualisations of museum curators, the study assessed accuracy across personal attributes (age, hair), appearance (attire, glasses), and paraphernalia (name tags, clipboards). While correctly rendered in most cases, DALL-E3 deviated from prompt specifications in 15.6% of all attributes (n=710). Errors were lowest for paraphernalia, moderate for personal appearance, and highest for depictions of the person themselves, particularly age. These findings demonstrate measurable prompt-to-image fidelity gaps with implications for bias detection and model evaluation.
>
---
#### [new 021] LightBagel: A Light-weighted, Double Fusion Framework for Unified Multimodal Understanding and Generation
- **分类: cs.CV**

- **简介: 该论文提出LightBagel框架，解决统一多模态理解与生成中模型训练成本高的问题。通过融合专用理解和生成模型，采用双路融合机制，在保留原模型优势的同时实现高效多模态交互，仅用350亿词元训练即在多个基准上取得优异性能。**

- **链接: [http://arxiv.org/pdf/2510.22946v1](http://arxiv.org/pdf/2510.22946v1)**

> **作者:** Zeyu Wang; Zilong Chen; Chenhui Gou; Feng Li; Chaorui Deng; Deyao Zhu; Kunchang Li; Weihao Yu; Haoqin Tu; Haoqi Fan; Cihang Xie
>
> **备注:** Preprint. Project page: https://ucsc-vlaa.github.io/LightBagel/
>
> **摘要:** Unified multimodal models have recently shown remarkable gains in both capability and versatility, yet most leading systems are still trained from scratch and require substantial computational resources. In this paper, we show that competitive performance can be obtained far more efficiently by strategically fusing publicly available models specialized for either generation or understanding. Our key design is to retain the original blocks while additionally interleaving multimodal self-attention blocks throughout the networks. This double fusion mechanism (1) effectively enables rich multi-modal fusion while largely preserving the original strengths of the base models, and (2) catalyzes synergistic fusion of high-level semantic representations from the understanding encoder with low-level spatial signals from the generation encoder. By training with only ~ 35B tokens, this approach achieves strong results across multiple benchmarks: 0.91 on GenEval for compositional text-to-image generation, 82.16 on DPG-Bench for complex text-to-image generation, 6.06 on GEditBench, and 3.77 on ImgEdit-Bench for image editing. By fully releasing the entire suite of code, model weights, and datasets, we hope to support future research on unified multimodal modeling.
>
---
#### [new 022] FastJAM: a Fast Joint Alignment Model for Images
- **分类: cs.CV**

- **简介: 该论文提出FastJAM，一种快速图像联合对齐方法。针对现有方法训练慢、计算复杂度高、需调参的问题，利用预训练匹配器与非参数聚类构建图结构，通过图神经网络高效推断单图同源变换，结合逆复合损失避免正则化调参，实现秒级对齐且精度优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.22842v1](http://arxiv.org/pdf/2510.22842v1)**

> **作者:** Omri Hirsch; Ron Shapira Weber; Shira Ifergane; Oren Freifeld
>
> **备注:** Accepted to NeurIPS 2025. Pages 1-10 are the Main Paper. Pages 23-31 are Supplemental Material. FastJAM website - https://bgu-cs-vil.github.io/FastJAM/
>
> **摘要:** Joint Alignment (JA) of images aims to align a collection of images into a unified coordinate frame, such that semantically-similar features appear at corresponding spatial locations. Most existing approaches often require long training times, large-capacity models, and extensive hyperparameter tuning. We introduce FastJAM, a rapid, graph-based method that drastically reduces the computational complexity of joint alignment tasks. FastJAM leverages pairwise matches computed by an off-the-shelf image matcher, together with a rapid nonparametric clustering, to construct a graph representing intra- and inter-image keypoint relations. A graph neural network propagates and aggregates these correspondences, efficiently predicting per-image homography parameters via image-level pooling. Utilizing an inverse-compositional loss, that eliminates the need for a regularization term over the predicted transformations (and thus also obviates the hyperparameter tuning associated with such terms), FastJAM performs image JA quickly and effectively. Experimental results on several benchmarks demonstrate that FastJAM achieves results better than existing modern JA methods in terms of alignment quality, while reducing computation time from hours or minutes to mere seconds. Our code is available at our project webpage, https://bgu-cs-vil.github.io/FastJAM/
>
---
#### [new 023] Bi-Encoder Contrastive Learning for Fingerprint and Iris Biometrics
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究指纹与虹膜生物特征的关联性，属于跨模态生物识别任务。针对“生物特征间统计独立”的传统假设，通过构建双编码器对比学习模型，利用274人数据验证指纹与虹膜的内在相关性。实验表明同人虹膜具强相关性，指纹亦有内聚性，但跨模态匹配效果接近随机，需更优数据与方法。**

- **链接: [http://arxiv.org/pdf/2510.22937v1](http://arxiv.org/pdf/2510.22937v1)**

> **作者:** Matthew So; Judah Goldfeder; Mark Lis; Hod Lipson
>
> **摘要:** There has been a historic assumption that the biometrics of an individual are statistically uncorrelated. We test this assumption by training Bi-Encoder networks on three verification tasks, including fingerprint-to-fingerprint matching, iris-to-iris matching, and cross-modal fingerprint-to-iris matching using 274 subjects with $\sim$100k fingerprints and 7k iris images. We trained ResNet-50 and Vision Transformer backbones in Bi-Encoder architectures such that the contrastive loss between images sampled from the same individual is minimized. The iris ResNet architecture reaches 91 ROC AUC score for iris-to-iris matching, providing clear evidence that the left and right irises of an individual are correlated. Fingerprint models reproduce the positive intra-subject suggested by prior work in this space. This is the first work attempting to use Vision Transformers for this matching. Cross-modal matching rises only slightly above chance, which suggests that more data and a more sophisticated pipeline is needed to obtain compelling results. These findings continue challenge independence assumptions of biometrics and we plan to extend this work to other biometrics in the future. Code available: https://github.com/MatthewSo/bio_fingerprints_iris.
>
---
#### [new 024] Activating Visual Context and Commonsense Reasoning through Masked Prediction in VLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型在多模态推理中缺乏视觉上下文与常识推理能力的问题，提出基于掩码预测的微调任务，融合视觉上下文与常识知识，构建评估基准MPCC Eval，并引入先验采样强化微调方法，显著提升模型在分布外及跨任务场景下的泛化推理能力。**

- **链接: [http://arxiv.org/pdf/2510.21807v1](http://arxiv.org/pdf/2510.21807v1)**

> **作者:** Jiaao Yu; Shenwei Li; Mingjie Han; Yifei Yin; Wenzheng Song; Chenghao Jia; Man Lan
>
> **备注:** 9 pages
>
> **摘要:** Recent breakthroughs in reasoning models have markedly advanced the reasoning capabilities of large language models, particularly via training on tasks with verifiable rewards. Yet, a significant gap persists in their adaptation to real world multimodal scenarios, most notably, vision language tasks, due to a heavy focus on single modal language settings. While efforts to transplant reinforcement learning techniques from NLP to VLMs have emerged, these approaches often remain confined to perception centric tasks or reduce images to textual summaries, failing to fully exploit visual context and commonsense knowledge, ultimately constraining the generalization of reasoning capabilities across diverse multimodal environments. To address this limitation, we introduce a novel fine tuning task, Masked Prediction via Context and Commonsense, which forces models to integrate visual context and commonsense reasoning by reconstructing semantically meaningful content from occluded images, thereby laying the foundation for generalized reasoning. To systematically evaluate the model performance in generalized reasoning, we developed a specialized evaluation benchmark, MPCC Eval, and employed various fine tuning strategies to guide reasoning. Among these, we introduced an innovative training method, Reinforcement Fine tuning with Prior Sampling, which not only enhances model performance but also improves its generalized reasoning capabilities in OOD and cross task scenarios.
>
---
#### [new 025] M$^{3}$T2IBench: A Large-Scale Multi-Category, Multi-Instance, Multi-Relation Text-to-Image Benchmark
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对文本到图像生成中的图文对齐问题，提出M³T2IBench基准，涵盖多类别、多实例、多关系复杂场景，并设计与人类评估高度一致的AlignScore指标。研究发现现有模型表现不佳，并提出无需训练的Revise-Then-Enforce后处理方法提升对齐效果。**

- **链接: [http://arxiv.org/pdf/2510.23020v1](http://arxiv.org/pdf/2510.23020v1)**

> **作者:** Huixuan Zhang; Xiaojun Wan
>
> **摘要:** Text-to-image models are known to struggle with generating images that perfectly align with textual prompts. Several previous studies have focused on evaluating image-text alignment in text-to-image generation. However, these evaluations either address overly simple scenarios, especially overlooking the difficulty of prompts with multiple different instances belonging to the same category, or they introduce metrics that do not correlate well with human evaluation. In this study, we introduce M$^3$T2IBench, a large-scale, multi-category, multi-instance, multi-relation along with an object-detection-based evaluation metric, $AlignScore$, which aligns closely with human evaluation. Our findings reveal that current open-source text-to-image models perform poorly on this challenging benchmark. Additionally, we propose the Revise-Then-Enforce approach to enhance image-text alignment. This training-free post-editing method demonstrates improvements in image-text alignment across a broad range of diffusion models. \footnote{Our code and data has been released in supplementary material and will be made publicly available after the paper is accepted.}
>
---
#### [new 026] EndoSfM3D: Learning to 3D Reconstruct Any Endoscopic Surgery Scene using Self-supervised Foundation Model
- **分类: cs.CV**

- **简介: 该论文属于医学图像3D重建任务，针对内窥镜手术中因无标定导致的深度与位姿估计难题，提出自监督方法EndoSfM3D，联合预测深度、位姿与内参。通过改进DA2模型并引入注意力网络与DoRA微调策略，实现无需标定的精准3D重建，在公开数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.22359v1](http://arxiv.org/pdf/2510.22359v1)**

> **作者:** Changhao Zhang; Matthew J. Clarkson; Mobarak I. Hoque
>
> **备注:** 11 pages
>
> **摘要:** 3D reconstruction of endoscopic surgery scenes plays a vital role in enhancing scene perception, enabling AR visualization, and supporting context-aware decision-making in image-guided surgery. A critical yet challenging step in this process is the accurate estimation of the endoscope's intrinsic parameters. In real surgical settings, intrinsic calibration is hindered by sterility constraints and the use of specialized endoscopes with continuous zoom and telescope rotation. Most existing methods for endoscopic 3D reconstruction do not estimate intrinsic parameters, limiting their effectiveness for accurate and reliable reconstruction. In this paper, we integrate intrinsic parameter estimation into a self-supervised monocular depth estimation framework by adapting the Depth Anything V2 (DA2) model for joint depth, pose, and intrinsics prediction. We introduce an attention-based pose network and a Weight-Decomposed Low-Rank Adaptation (DoRA) strategy for efficient fine-tuning of DA2. Our method is validated on the SCARED and C3VD public datasets, demonstrating superior performance compared to recent state-of-the-art approaches in self-supervised monocular depth estimation and 3D reconstruction. Code and model weights can be found in project repository: https://github.com/MOYF-beta/EndoSfM3D.
>
---
#### [new 027] Poisson Flow Consistency Training
- **分类: cs.CV; cs.AI; 68T07 (Primary), 68T45 (Secondary)**

- **简介: 该论文针对PFCM仅能通过知识蒸馏训练的问题，提出独立训练方法PFCT。通过引入扰动核、正弦离散化调度和Beta噪声，实现无需预训练PFGM++的训练，提升模型灵活性与生成质量。在低剂量CT图像去噪任务中表现优异，效果接近一致性模型，验证了PFCT的有效性与潜力。**

- **链接: [http://arxiv.org/pdf/2510.21857v1](http://arxiv.org/pdf/2510.21857v1)**

> **作者:** Anthony Zhang; Mahmut Gokmen; Dennis Hein; Rongjun Ge; Wenjun Xia; Ge Wang; Jin Chen
>
> **备注:** 5 pages, 3 figures, 1 table
>
> **摘要:** The Poisson Flow Consistency Model (PFCM) is a consistency-style model based on the robust Poisson Flow Generative Model++ (PFGM++) which has achieved success in unconditional image generation and CT image denoising. Yet the PFCM can only be trained in distillation which limits the potential of the PFCM in many data modalities. The objective of this research was to create a method to train the PFCM in isolation called Poisson Flow Consistency Training (PFCT). The perturbation kernel was leveraged to remove the pretrained PFGM++, and the sinusoidal discretization schedule and Beta noise distribution were introduced in order to facilitate adaptability and improve sample quality. The model was applied to the task of low dose computed tomography image denoising and improved the low dose image in terms of LPIPS and SSIM. It also displayed similar denoising effectiveness as models like the Consistency Model. PFCT is established as a valid method of training the PFCM from its effectiveness in denoising CT images, showing potential with competitive results to other generative models. Further study is needed in the precise optimization of PFCT and in its applicability to other generative modeling tasks. The framework of PFCT creates more flexibility for the ways in which a PFCM can be created and can be applied to the field of generative modeling.
>
---
#### [new 028] Human-Centric Anomaly Detection in Surveillance Videos Using YOLO-World and Spatio-Temporal Deep Learning
- **分类: cs.CV; cs.AI; I.2.10; I.4.9; I.2.6**

- **简介: 该论文针对监控视频中的异常检测任务，解决异常事件多样、类别不平衡及场景干扰问题。提出融合YOLO-World与时空深度学习的框架：先通过开放词汇检测定位人体并跟踪，用高斯模糊抑制背景干扰；再用InceptionV3提取空间特征，BiLSTM捕捉时序动态，实现多类异常分类，显著提升检测精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.22056v1](http://arxiv.org/pdf/2510.22056v1)**

> **作者:** Mohammad Ali Etemadi Naeen; Hoda Mohammadzade; Saeed Bagheri Shouraki
>
> **摘要:** Anomaly detection in surveillance videos remains a challenging task due to the diversity of abnormal events, class imbalance, and scene-dependent visual clutter. To address these issues, we propose a robust deep learning framework that integrates human-centric preprocessing with spatio-temporal modeling for multi-class anomaly classification. Our pipeline begins by applying YOLO-World - an open-vocabulary vision-language detector - to identify human instances in raw video clips, followed by ByteTrack for consistent identity-aware tracking. Background regions outside detected bounding boxes are suppressed via Gaussian blurring, effectively reducing scene-specific distractions and focusing the model on behaviorally relevant foreground content. The refined frames are then processed by an ImageNet-pretrained InceptionV3 network for spatial feature extraction, and temporal dynamics are captured using a bidirectional LSTM (BiLSTM) for sequence-level classification. Evaluated on a five-class subset of the UCF-Crime dataset (Normal, Burglary, Fighting, Arson, Explosion), our method achieves a mean test accuracy of 92.41% across three independent trials, with per-class F1-scores consistently exceeding 0.85. Comprehensive evaluation metrics - including confusion matrices, ROC curves, and macro/weighted averages - demonstrate strong generalization and resilience to class imbalance. The results confirm that foreground-focused preprocessing significantly enhances anomaly discrimination in real-world surveillance scenarios.
>
---
#### [new 029] Gestura: A LVLM-Powered System Bridging Motion and Semantics for Real-Time Free-Form Gesture Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Gestura系统，解决自由手势理解中识别不准、响应慢的问题。利用预训练大视觉语言模型（LVLM）结合手部关键点处理模块与思维链推理，实现端到端的实时自由手势语义理解，并构建首个开源自由手势意图理解数据集。**

- **链接: [http://arxiv.org/pdf/2510.21814v1](http://arxiv.org/pdf/2510.21814v1)**

> **作者:** Zhuoming Li; Aitong Liu; Mengxi Jia; Tengxiang Zhang; Dell Zhang; Xuelong Li
>
> **备注:** IMWUT2025
>
> **摘要:** Free-form gesture understanding is highly appealing for human-computer interaction, as it liberates users from the constraints of predefined gesture categories. However, the sole existing solution GestureGPT suffers from limited recognition accuracy and slow response times. In this paper, we propose Gestura, an end-to-end system for free-form gesture understanding. Gestura harnesses a pre-trained Large Vision-Language Model (LVLM) to align the highly dynamic and diverse patterns of free-form gestures with high-level semantic concepts. To better capture subtle hand movements across different styles, we introduce a Landmark Processing Module that compensate for LVLMs' lack of fine-grained domain knowledge by embedding anatomical hand priors. Further, a Chain-of-Thought (CoT) reasoning strategy enables step-by-step semantic inference, transforming shallow knowledge into deep semantic understanding and significantly enhancing the model's ability to interpret ambiguous or unconventional gestures. Together, these components allow Gestura to achieve robust and adaptable free-form gesture comprehension. Additionally, we have developed the first open-source dataset for free-form gesture intention reasoning and understanding with over 300,000 annotated QA pairs.
>
---
#### [new 030] An Efficient Remote Sensing Super Resolution Method Exploring Diffusion Priors and Multi-Modal Constraints for Crop Type Mapping
- **分类: cs.CV**

- **简介: 该论文针对遥感图像超分辨率（RSSR）任务，提出高效框架LSSR。通过融合预训练扩散模型、多模态约束与自适应模块，在保持快速推理（0.39秒/图）的同时，显著提升作物边界识别与光谱保真度，实现更精准的作物类型分类。**

- **链接: [http://arxiv.org/pdf/2510.23382v1](http://arxiv.org/pdf/2510.23382v1)**

> **作者:** Songxi Yang; Tang Sui; Qunying Huang
>
> **备注:** 41 pages
>
> **摘要:** Super resolution offers a way to harness medium even lowresolution but historically valuable remote sensing image archives. Generative models, especially diffusion models, have recently been applied to remote sensing super resolution (RSSR), yet several challenges exist. First, diffusion models are effective but require expensive training from scratch resources and have slow inference speeds. Second, current methods have limited utilization of auxiliary information as real-world constraints to reconstruct scientifically realistic images. Finally, most current methods lack evaluation on downstream tasks. In this study, we present a efficient LSSR framework for RSSR, supported by a new multimodal dataset of paired 30 m Landsat 8 and 10 m Sentinel 2 imagery. Built on frozen pretrained Stable Diffusion, LSSR integrates crossmodal attention with auxiliary knowledge (Digital Elevation Model, land cover, month) and Synthetic Aperture Radar guidance, enhanced by adapters and a tailored Fourier NDVI loss to balance spatial details and spectral fidelity. Extensive experiments demonstrate that LSSR significantly improves crop boundary delineation and recovery, achieving state-of-the-art performance with Peak Signal-to-Noise Ratio/Structural Similarity Index Measure of 32.63/0.84 (RGB) and 23.99/0.78 (IR), and the lowest NDVI Mean Squared Error (0.042), while maintaining efficient inference (0.39 sec/image). Moreover, LSSR transfers effectively to NASA Harmonized Landsat and Sentinel (HLS) super resolution, yielding more reliable crop classification (F1: 0.86) than Sentinel-2 (F1: 0.85). These results highlight the potential of RSSR to advance precision agriculture.
>
---
#### [new 031] EdgeSync: Accelerating Edge-Model Updates for Data Drift through Adaptive Continuous Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对边缘视频分析中因数据漂移导致模型精度下降的问题，提出EdgeSync框架。通过引入时效性与推理结果的样本筛选机制及动态训练管理，提升模型更新效率与准确性，实现更及时、精准的边缘模型自适应。**

- **链接: [http://arxiv.org/pdf/2510.21781v1](http://arxiv.org/pdf/2510.21781v1)**

> **作者:** Runchu Donga; Peng Zhao; Guiqin Wang; Nan Qi; Jie Lin
>
> **摘要:** Real-time video analytics systems typically deploy lightweight models on edge devices to reduce latency. However, the distribution of data features may change over time due to various factors such as changing lighting and weather conditions, leading to decreased model accuracy. Recent frameworks try to address this issue by leveraging remote servers to continuously train and adapt lightweight edge models using more complex models in the cloud. Despite these advancements, existing methods face two key challenges: first, the retraining process is compute-intensive, causing significant delays in model updates; second, the new model may not align well with the evolving data distribution of the current video stream. To address these challenges, we introduce EdgeSync, an efficient edge-model updating approach that enhances sample filtering by incorporating timeliness and inference results, thus ensuring training samples are more relevant to the current video content while reducing update delays. Additionally, EdgeSync features a dynamic training management module that optimizes the timing and sequencing of model updates to improve their timeliness. Evaluations on diverse and complex real-world datasets demonstrate that EdgeSync improves accuracy by approximately 3.4% compared to existing methods and by about 10% compared to traditional approaches.
>
---
#### [new 032] Alias-Free ViT: Fractional Shift Invariance via Linear Attention
- **分类: cs.CV**

- **简介: 该论文针对视觉变换器（ViT）缺乏平移不变性的问题，提出Alias-Free ViT。通过无混叠下采样与非线性、线性交叉协方差注意力机制，实现对整数和分数平移的等变性，提升模型对图像微小平移的鲁棒性，同时保持优异的图像分类性能。**

- **链接: [http://arxiv.org/pdf/2510.22673v1](http://arxiv.org/pdf/2510.22673v1)**

> **作者:** Hagay Michaeli; Daniel Soudry
>
> **备注:** Accepted at NeurIPS 2025. Code is available at https://github.com/hmichaeli/alias_free_vit
>
> **摘要:** Transformers have emerged as a competitive alternative to convnets in vision tasks, yet they lack the architectural inductive bias of convnets, which may hinder their potential performance. Specifically, Vision Transformers (ViTs) are not translation-invariant and are more sensitive to minor image translations than standard convnets. Previous studies have shown, however, that convnets are also not perfectly shift-invariant, due to aliasing in downsampling and nonlinear layers. Consequently, anti-aliasing approaches have been proposed to certify convnets' translation robustness. Building on this line of work, we propose an Alias-Free ViT, which combines two main components. First, it uses alias-free downsampling and nonlinearities. Second, it uses linear cross-covariance attention that is shift-equivariant to both integer and fractional translations, enabling a shift-invariant global representation. Our model maintains competitive performance in image classification and outperforms similar-sized models in terms of robustness to adversarial translations.
>
---
#### [new 033] Evaluation of Vision-LLMs in Surveillance Video
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型（VLMs）在监控视频中零样本异常行为识别的能力，旨在解决动态3D场景下空间推理不足与隐私保护冲突的问题。通过文本描述与语义蕴含评分，评估小规模预训练VLMs在UCF-Crime和RWF-2000上的表现，提出结构化提示、轻量级空间记忆等增强路径，推动无需任务微调的通用视频理解。**

- **链接: [http://arxiv.org/pdf/2510.23190v1](http://arxiv.org/pdf/2510.23190v1)**

> **作者:** Pascal Benschop; Cristian Meo; Justin Dauwels; Jelte P. Mense
>
> **备注:** Accepted as poster in the NeurIPS 2025 Workshop on Space in Vision, Language, and Embodied AI
>
> **摘要:** The widespread use of cameras in our society has created an overwhelming amount of video data, far exceeding the capacity for human monitoring. This presents a critical challenge for public safety and security, as the timely detection of anomalous or criminal events is crucial for effective response and prevention. The ability for an embodied agent to recognize unexpected events is fundamentally tied to its capacity for spatial reasoning. This paper investigates the spatial reasoning of vision-language models (VLMs) by framing anomalous action recognition as a zero-shot, language-grounded task, addressing the embodied perception challenge of interpreting dynamic 3D scenes from sparse 2D video. Specifically, we investigate whether small, pre-trained vision--LLMs can act as spatially-grounded, zero-shot anomaly detectors by converting video into text descriptions and scoring labels via textual entailment. We evaluate four open models on UCF-Crime and RWF-2000 under prompting and privacy-preserving conditions. Few-shot exemplars can improve accuracy for some models, but may increase false positives, and privacy filters -- especially full-body GAN transforms -- introduce inconsistencies that degrade accuracy. These results chart where current vision--LLMs succeed (simple, spatially salient events) and where they falter (noisy spatial cues, identity obfuscation). Looking forward, we outline concrete paths to strengthen spatial grounding without task-specific training: structure-aware prompts, lightweight spatial memory across clips, scene-graph or 3D-pose priors during description, and privacy methods that preserve action-relevant geometry. This positions zero-shot, language-grounded pipelines as adaptable building blocks for embodied, real-world video understanding. Our implementation for evaluating VLMs is publicly available at: https://github.com/pascalbenschopTU/VLLM_AnomalyRecognition
>
---
#### [new 034] IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出IGGT模型，解决3D重建与语义理解割裂的问题。通过2D视觉输入，利用3D一致性对比学习，实现几何结构与实例级语义的统一表征，支持端到端的语义3D重建。构建了InsScene-15K数据集以支持该任务。**

- **链接: [http://arxiv.org/pdf/2510.22706v1](http://arxiv.org/pdf/2510.22706v1)**

> **作者:** Hao Li; Zhengyu Zou; Fangfu Liu; Xuanyang Zhang; Fangzhou Hong; Yukang Cao; Yushi Lan; Manyuan Zhang; Gang Yu; Dingwen Zhang; Ziwei Liu
>
> **备注:** https://github.com/lifuguan/IGGT_official
>
> **摘要:** Humans naturally perceive the geometric structure and semantic content of a 3D world as intertwined dimensions, enabling coherent and accurate understanding of complex scenes. However, most prior approaches prioritize training large geometry models for low-level 3D reconstruction and treat high-level spatial understanding in isolation, overlooking the crucial interplay between these two fundamental aspects of 3D-scene analysis, thereby limiting generalization and leading to poor performance in downstream 3D understanding tasks. Recent attempts have mitigated this issue by simply aligning 3D models with specific language models, thus restricting perception to the aligned model's capacity and limiting adaptability to downstream tasks. In this paper, we propose InstanceGrounded Geometry Transformer (IGGT), an end-to-end large unified transformer to unify the knowledge for both spatial reconstruction and instance-level contextual understanding. Specifically, we design a 3D-Consistent Contrastive Learning strategy that guides IGGT to encode a unified representation with geometric structures and instance-grounded clustering through only 2D visual inputs. This representation supports consistent lifting of 2D visual inputs into a coherent 3D scene with explicitly distinct object instances. To facilitate this task, we further construct InsScene-15K, a large-scale dataset with high-quality RGB images, poses, depth maps, and 3D-consistent instance-level mask annotations with a novel data curation pipeline.
>
---
#### [new 035] Cross-Species Transfer Learning in Agricultural AI: Evaluating ZebraPose Adaptation for Dairy Cattle Pose Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于农业人工智能中的姿态估计任务，旨在解决畜牧数据稀缺导致模型泛化能力差的问题。研究通过跨物种迁移学习，将基于斑马图像训练的ZebraPose模型应用于奶牛姿态识别，评估其在真实牧场环境下的表现，揭示了合成到真实场景的域差距问题，并呼吁构建更贴近实际农场需求的AI系统。**

- **链接: [http://arxiv.org/pdf/2510.22618v1](http://arxiv.org/pdf/2510.22618v1)**

> **作者:** Mackenzie Tapp; Sibi Chakravarthy Parivendan; Kashfia Sailunaz; Suresh Neethirajan
>
> **备注:** 20 pages, 11 figures, 6 Tables
>
> **摘要:** Pose estimation serves as a cornerstone of computer vision for understanding animal posture, behavior, and welfare. Yet, agricultural applications remain constrained by the scarcity of large, annotated datasets for livestock, especially dairy cattle. This study evaluates the potential and limitations of cross-species transfer learning by adapting ZebraPose - a vision transformer-based model trained on synthetic zebra imagery - for 27-keypoint detection in dairy cows under real barn conditions. Using three configurations - a custom on-farm dataset (375 images, Sussex, New Brunswick, Canada), a subset of the APT-36K benchmark dataset, and their combination, we systematically assessed model accuracy and generalization across environments. While the combined model achieved promising performance (AP = 0.86, AR = 0.87, PCK 0.5 = 0.869) on in-distribution data, substantial generalization failures occurred when applied to unseen barns and cow populations. These findings expose the synthetic-to-real domain gap as a major obstacle to agricultural AI deployment and emphasize that morphological similarity between species is insufficient for cross-domain transfer. The study provides practical insights into dataset diversity, environmental variability, and computational constraints that influence real-world deployment of livestock monitoring systems. We conclude with a call for agriculture-first AI design, prioritizing farm-level realism, cross-environment robustness, and open benchmark datasets to advance trustworthy and scalable animal-centric technologies.
>
---
#### [new 036] Self-Attention Decomposition For Training Free Diffusion Editing
- **分类: cs.CV**

- **简介: 该论文针对扩散模型图像编辑中控制精度低的问题，提出无需训练的自注意力分解方法。通过分析预训练模型的自注意力权重矩阵特征向量，直接获取语义可解释的编辑方向，避免数据采样与额外训练，显著提升编辑效率并保持高质量结果。**

- **链接: [http://arxiv.org/pdf/2510.22650v1](http://arxiv.org/pdf/2510.22650v1)**

> **作者:** Tharun Anand; Mohammad Hassan Vali; Arno Solin
>
> **备注:** 4 pages (ICASSP Format)
>
> **摘要:** Diffusion models achieve remarkable fidelity in image synthesis, yet precise control over their outputs for targeted editing remains challenging. A key step toward controllability is to identify interpretable directions in the model's latent representations that correspond to semantic attributes. Existing approaches for finding interpretable directions typically rely on sampling large sets of images or training auxiliary networks, which limits efficiency. We propose an analytical method that derives semantic editing directions directly from the pretrained parameters of diffusion models, requiring neither additional data nor fine-tuning. Our insight is that self-attention weight matrices encode rich structural information about the data distribution learned during training. By computing the eigenvectors of these weight matrices, we obtain robust and interpretable editing directions. Experiments demonstrate that our method produces high-quality edits across multiple datasets while reducing editing time significantly by 60% over current benchmarks.
>
---
#### [new 037] Enpowering Your Pansharpening Models with Generalizability: Unified Distribution is All You Need
- **分类: cs.CV**

- **简介: 该论文针对遥感图像全色锐化任务中模型泛化能力差的问题，提出统一分布策略UniPAN。通过构建分布变换函数，使不同传感器数据在训练与测试时服从相同分布，提升模型跨数据集适用性，实现“一次训练，永久部署”。**

- **链接: [http://arxiv.org/pdf/2510.22217v1](http://arxiv.org/pdf/2510.22217v1)**

> **作者:** Yongchuan Cui; Peng Liu; Hui Zhang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Existing deep learning-based models for remote sensing pansharpening exhibit exceptional performance on training datasets. However, due to sensor-specific characteristics and varying imaging conditions, these models suffer from substantial performance degradation when applied to unseen satellite data, lacking generalizability and thus limiting their applicability. We argue that the performance drops stem primarily from distributional discrepancies from different sources and the key to addressing this challenge lies in bridging the gap between training and testing distributions. To validate the idea and further achieve a "train once, deploy forever" capability, this paper introduces a novel and intuitive approach to enpower any pansharpening models with generalizability by employing a unified distribution strategy (UniPAN). Specifically, we construct a distribution transformation function that normalizes the pixels sampled from different sources to conform to an identical distribution. The deep models are trained on the transformed domain, and during testing on new datasets, the new data are also transformed to match the training distribution. UniPAN aims to train and test the model on a unified and consistent distribution, thereby enhancing its generalizability. Extensive experiments validate the efficacy of UniPAN, demonstrating its potential to significantly enhance the performance of deep pansharpening models across diverse satellite sensors. Codes: https://github.com/yc-cui/UniPAN.
>
---
#### [new 038] H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows
- **分类: cs.CV**

- **简介: 该论文提出H2OFlow，用于3D人-物交互（HOI） affordance理解，解决现有方法依赖人工标注、局限于接触分析的问题。通过合成数据与3D点云扩散流，联合建模接触、朝向与空间占据，实现无标注的3D affordance学习，显著提升真实场景泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.21769v1](http://arxiv.org/pdf/2510.21769v1)**

> **作者:** Harry Zhang; Luca Carlone
>
> **摘要:** Understanding how humans interact with the surrounding environment, and specifically reasoning about object interactions and affordances, is a critical challenge in computer vision, robotics, and AI. Current approaches often depend on labor-intensive, hand-labeled datasets capturing real-world or simulated human-object interaction (HOI) tasks, which are costly and time-consuming to produce. Furthermore, most existing methods for 3D affordance understanding are limited to contact-based analysis, neglecting other essential aspects of human-object interactions, such as orientation (\eg, humans might have a preferential orientation with respect certain objects, such as a TV) and spatial occupancy (\eg, humans are more likely to occupy certain regions around an object, like the front of a microwave rather than its back). To address these limitations, we introduce \emph{H2OFlow}, a novel framework that comprehensively learns 3D HOI affordances -- encompassing contact, orientation, and spatial occupancy -- using only synthetic data generated from 3D generative models. H2OFlow employs a dense 3D-flow-based representation, learned through a dense diffusion process operating on point clouds. This learned flow enables the discovery of rich 3D affordances without the need for human annotations. Through extensive quantitative and qualitative evaluations, we demonstrate that H2OFlow generalizes effectively to real-world objects and surpasses prior methods that rely on manual annotations or mesh-based representations in modeling 3D affordance.
>
---
#### [new 039] MAGIC-Talk: Motion-aware Audio-Driven Talking Face Generation with Customizable Identity Control
- **分类: cs.CV**

- **简介: 该论文聚焦于音频驱动的说话人脸生成任务，旨在解决长期视频中身份保持、时序一致性和可定制性差的问题。提出MAGIC-Talk框架，基于单张参考图实现身份保真与文本控制的精细编辑，并通过结构化运动先验和渐进潜空间融合提升运动连贯性与视频质量。**

- **链接: [http://arxiv.org/pdf/2510.22810v1](http://arxiv.org/pdf/2510.22810v1)**

> **作者:** Fatemeh Nazarieh; Zhenhua Feng; Diptesh Kanojia; Muhammad Awais; Josef Kittler
>
> **摘要:** Audio-driven talking face generation has gained significant attention for applications in digital media and virtual avatars. While recent methods improve audio-lip synchronization, they often struggle with temporal consistency, identity preservation, and customization, especially in long video generation. To address these issues, we propose MAGIC-Talk, a one-shot diffusion-based framework for customizable and temporally stable talking face generation. MAGIC-Talk consists of ReferenceNet, which preserves identity and enables fine-grained facial editing via text prompts, and AnimateNet, which enhances motion coherence using structured motion priors. Unlike previous methods requiring multiple reference images or fine-tuning, MAGIC-Talk maintains identity from a single image while ensuring smooth transitions across frames. Additionally, a progressive latent fusion strategy is introduced to improve long-form video quality by reducing motion inconsistencies and flickering. Extensive experiments demonstrate that MAGIC-Talk outperforms state-of-the-art methods in visual quality, identity preservation, and synchronization accuracy, offering a robust solution for talking face generation.
>
---
#### [new 040] T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对扩散模型图像水印中鲁棒性与生成多样性难以兼顾的问题，提出T2SMark方法。通过尾部截断采样（TTS）在可靠区域嵌入水印，并引入会话密钥保证多样性，实现鲁棒性与多样性的平衡。**

- **链接: [http://arxiv.org/pdf/2510.22366v1](http://arxiv.org/pdf/2510.22366v1)**

> **作者:** Jindong Yang; Han Fang; Weiming Zhang; Nenghai Yu; Kejiang Chen
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Diffusion models have advanced rapidly in recent years, producing high-fidelity images while raising concerns about intellectual property protection and the misuse of generative AI. Image watermarking for diffusion models, particularly Noise-as-Watermark (NaW) methods, encode watermark as specific standard Gaussian noise vector for image generation, embedding the infomation seamlessly while maintaining image quality. For detection, the generation process is inverted to recover the initial noise vector containing the watermark before extraction. However, existing NaW methods struggle to balance watermark robustness with generation diversity. Some methods achieve strong robustness by heavily constraining initial noise sampling, which degrades user experience, while others preserve diversity but prove too fragile for real-world deployment. To address this issue, we propose T2SMark, a two-stage watermarking scheme based on Tail-Truncated Sampling (TTS). Unlike prior methods that simply map bits to positive or negative values, TTS enhances robustness by embedding bits exclusively in the reliable tail regions while randomly sampling the central zone to preserve the latent distribution. Our two-stage framework then ensures sampling diversity by integrating a randomly generated session key into both encryption pipelines. We evaluate T2SMark on diffusion models with both U-Net and DiT backbones. Extensive experiments show that it achieves an optimal balance between robustness and diversity. Our code is available at \href{https://github.com/0xD009/T2SMark}{https://github.com/0xD009/T2SMark}.
>
---
#### [new 041] Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling
- **分类: cs.CV**

- **简介: 该论文针对扩散模型采样效率与质量的权衡问题，提出AdaSDE，一种带自适应随机系数的单步SDE求解器。通过轻量级蒸馏学习动态调节误差校正强度，融合ODE高效性与SDE鲁棒性，在低计算预算下显著提升生成质量。**

- **链接: [http://arxiv.org/pdf/2510.23285v1](http://arxiv.org/pdf/2510.23285v1)**

> **作者:** Ruoyu Wang; Beier Zhu; Junzhi Li; Liangyu Yuan; Chi Zhang
>
> **备注:** To appear in NeurIPS 2025
>
> **摘要:** Diffusion-based generative processes, formulated as differential equation solving, frequently balance computational speed with sample quality. Our theoretical investigation of ODE- and SDE-based solvers reveals complementary weaknesses: ODE solvers accumulate irreducible gradient error along deterministic trajectories, while SDE methods suffer from amplified discretization errors when the step budget is limited. Building upon this insight, we introduce AdaSDE, a novel single-step SDE solver that aims to unify the efficiency of ODEs with the error resilience of SDEs. Specifically, we introduce a single per-step learnable coefficient, estimated via lightweight distillation, which dynamically regulates the error correction strength to accelerate diffusion sampling. Notably, our framework can be integrated with existing solvers to enhance their capabilities. Extensive experiments demonstrate state-of-the-art performance: at 5 NFE, AdaSDE achieves FID scores of 4.18 on CIFAR-10, 8.05 on FFHQ and 6.96 on LSUN Bedroom. Codes are available in https://github.com/WLU-wry02/AdaSDE.
>
---
#### [new 042] RatioWaveNet: A Learnable RDWT Front-End for Robust and Interpretable EEG Motor-Imagery Classification
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 该论文针对非侵入式脑电（EEG）运动想象分类中信号非平稳、信噪比低和个体差异大的问题，提出RatioWaveNet模型。通过可学习的有理稀疏小波变换（RDWT）作为前端，增强时频特征提取能力，结合CNN-Transformer架构，在BCI-IV数据集上显著提升最差受试者准确率，实现更鲁棒、可解释的分类。**

- **链接: [http://arxiv.org/pdf/2510.21841v1](http://arxiv.org/pdf/2510.21841v1)**

> **作者:** Marco Siino; Giuseppe Bonomo; Rosario Sorbello; Ilenia Tinnirello
>
> **摘要:** Brain-computer interfaces (BCIs) based on motor imagery (MI) translate covert movement intentions into actionable commands, yet reliable decoding from non-invasive EEG remains challenging due to nonstationarity, low SNR, and subject variability. We present RatioWaveNet, which augments a strong temporal CNN-Transformer backbone (TCFormer) with a trainable, Rationally-Dilated Wavelet Transform (RDWT) front end. The RDWT performs an undecimated, multi-resolution subband decomposition that preserves temporal length and shift-invariance, enhancing sensorimotor rhythms while mitigating jitter and mild artifacts; subbands are fused via lightweight grouped 1-D convolutions and passed to a multi-kernel CNN for local temporal-spatial feature extraction, a grouped-query attention encoder for long-range context, and a compact TCN head for causal temporal integration. Our goal is to test whether this principled wavelet front end improves robustness precisely where BCIs typically fail - on the hardest subjects - and whether such gains persist on average across seeds under both intra- and inter-subject protocols. On BCI-IV-2a and BCI-IV-2b, across five seeds, RatioWaveNet improves worst-subject accuracy over the Transformer backbone by +0.17 / +0.42 percentage points (Sub-Dependent / LOSO) on 2a and by +1.07 / +2.54 percentage points on 2b, with consistent average-case gains and modest computational overhead. These results indicate that a simple, trainable wavelet front end is an effective plug-in to strengthen Transformer-based BCIs, improving worst-case reliability without sacrificing efficiency.
>
---
#### [new 043] DQ3D: Depth-guided Query for Transformer-Based 3D Object Detection in Traffic Scenarios
- **分类: cs.CV**

- **简介: 该论文针对交通场景下基于多视角图像的3D目标检测任务，解决因参考点远离目标导致的误检问题。提出深度引导查询生成器（DQ3D），利用深度信息与2D检测结果确保参考点位于物体表面或内部，并引入融合历史检测结果的混合注意力机制，提升对遮挡目标的检测性能。**

- **链接: [http://arxiv.org/pdf/2510.23144v1](http://arxiv.org/pdf/2510.23144v1)**

> **作者:** Ziyu Wang; Wenhao Li; Ji Wu
>
> **摘要:** 3D object detection from multi-view images in traffic scenarios has garnered significant attention in recent years. Many existing approaches rely on object queries that are generated from 3D reference points to localize objects. However, a limitation of these methods is that some reference points are often far from the target object, which can lead to false positive detections. In this paper, we propose a depth-guided query generator for 3D object detection (DQ3D) that leverages depth information and 2D detections to ensure that reference points are sampled from the surface or interior of the object. Furthermore, to address partially occluded objects in current frame, we introduce a hybrid attention mechanism that fuses historical detection results with depth-guided queries, thereby forming hybrid queries. Evaluation on the nuScenes dataset demonstrates that our method outperforms the baseline by 6.3\% in terms of mean Average Precision (mAP) and 4.3\% in the NuScenes Detection Score (NDS).
>
---
#### [new 044] TrajGATFormer: A Graph-Based Transformer Approach for Worker and Obstacle Trajectory Prediction in Off-site Construction Environments
- **分类: cs.CV**

- **简介: 该论文针对施工现场工人与障碍物轨迹预测任务，解决动态环境中碰撞风险评估难题。提出TrajGATFormer及扩展模型，结合YOLOv10n与DeepSORT实现精准检测跟踪，利用图注意力机制与Transformer捕捉时空交互，显著提升预测精度，相较传统方法降低ADE与FDE超35%。**

- **链接: [http://arxiv.org/pdf/2510.22205v1](http://arxiv.org/pdf/2510.22205v1)**

> **作者:** Mohammed Alduais; Xinming Li; Qipei Mei
>
> **摘要:** As the demand grows within the construction industry for processes that are not only faster but also safer and more efficient, offsite construction has emerged as a solution, though it brings new safety risks due to the close interaction between workers, machinery, and moving obstacles. Predicting the future trajectories of workers and taking into account social and environmental factors is a crucial step for developing collision-avoidance systems to mitigate such risks. Traditional methods often struggle to adapt to the dynamic and unpredictable nature of construction environments. Many rely on simplified assumptions or require hand-crafted features, limiting their ability to respond to complex, real-time interactions between workers and moving obstacles. While recent data-driven methods have improved the modeling of temporal patterns, they still face challenges in capturing long-term behavior and accounting for the spatial and social context crucial to collision risk assessment. To address these limitations, this paper proposes a framework integrating YOLOv10n and DeepSORT for precise detection and tracking, along with two novel trajectory prediction models: TrajGATFormer and TrajGATFormer-Obstacle. YOLOv10n serves as the backbone for object detection, accurately identifying workers and obstacles in diverse scenes, while DeepSORT efficiently tracks them over time with unique IDs for continuity. Both models employ a transformer encoder-decoder with Graph Attention Networks (GAT) to capture temporal and spatial interactions. TrajGATFormer predicts worker trajectories with an ADE of 1.25 m and FDE of 2.3 m over a 4.8 s horizon, while TrajGATFormer-Obstacle extends prediction to both workers and obstacles, achieving higher accuracy (ADE 1.15 m, FDE 2.2 m). Comparative analysis shows both models outperform traditional methods, reducing ADE and FDE by up to 35% and 38%, respectively.
>
---
#### [new 045] Improving the Physics of Video Generation with VJEPA-2 Reward Signal
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于视频生成任务，旨在提升生成视频的物理合理性。针对现有模型缺乏物理理解的问题，研究将VJEPA-2作为奖励信号引入MAGI-1模型，利用其自监督学习获得的物理常识引导生成过程，显著提升了视频的物理可实现性，效果提升约6%。**

- **链接: [http://arxiv.org/pdf/2510.21840v1](http://arxiv.org/pdf/2510.21840v1)**

> **作者:** Jianhao Yuan; Xiaofeng Zhang; Felix Friedrich; Nicolas Beltran-Velez; Melissa Hall; Reyhane Askari-Hemmat; Xiaochuang Han; Nicolas Ballas; Michal Drozdzal; Adriana Romero-Soriano
>
> **备注:** 2 pages
>
> **摘要:** This is a short technical report describing the winning entry of the PhysicsIQ Challenge, presented at the Perception Test Workshop at ICCV 2025. State-of-the-art video generative models exhibit severely limited physical understanding, and often produce implausible videos. The Physics IQ benchmark has shown that visual realism does not imply physics understanding. Yet, intuitive physics understanding has shown to emerge from SSL pretraining on natural videos. In this report, we investigate whether we can leverage SSL-based video world models to improve the physics plausibility of video generative models. In particular, we build ontop of the state-of-the-art video generative model MAGI-1 and couple it with the recently introduced Video Joint Embedding Predictive Architecture 2 (VJEPA-2) to guide the generation process. We show that by leveraging VJEPA-2 as reward signal, we can improve the physics plausibility of state-of-the-art video generative models by ~6%.
>
---
#### [new 046] A Video Is Not Worth a Thousand Words
- **分类: cs.CV**

- **简介: 该论文研究多模态视频问答任务，针对视觉语言模型中文本主导问题，提出基于Shapley值的联合特征归因与模态评分方法。通过分析视频帧与文本元素的贡献，揭示模型对文本的依赖性及多选题中忽略干扰项的能力。**

- **链接: [http://arxiv.org/pdf/2510.23253v1](http://arxiv.org/pdf/2510.23253v1)**

> **作者:** Sam Pollard; Michael Wray
>
> **摘要:** As we become increasingly dependent on vision language models (VLMs) to answer questions about the world around us, there is a significant amount of research devoted to increasing both the difficulty of video question answering (VQA) datasets, and the context lengths of the models that they evaluate. The reliance on large language models as backbones has lead to concerns about potential text dominance, and the exploration of interactions between modalities is underdeveloped. How do we measure whether we're heading in the right direction, with the complexity that multi-modal models introduce? We propose a joint method of computing both feature attributions and modality scores based on Shapley values, where both the features and modalities are arbitrarily definable. Using these metrics, we compare $6$ VLM models of varying context lengths on $4$ representative datasets, focusing on multiple-choice VQA. In particular, we consider video frames and whole textual elements as equal features in the hierarchy, and the multiple-choice VQA task as an interaction between three modalities: video, question and answer. Our results demonstrate a dependence on text and show that the multiple-choice VQA task devolves into a model's ability to ignore distractors. Code available at https://github.com/sjpollard/a-video-is-not-worth-a-thousand-words.
>
---
#### [new 047] 3D Roadway Scene Object Detection with LIDARs in Snowfall Conditions
- **分类: cs.CV**

- **简介: 该论文针对车载激光雷达在降雪条件下性能下降的问题，研究了雪量对激光信号衰减及反射的影响，构建物理模型并生成模拟雪景数据。通过对比真实与合成数据，评估预训练目标检测模型在不同雪况下的表现，旨在量化恶劣天气下感知系统失效机制。**

- **链接: [http://arxiv.org/pdf/2510.22436v1](http://arxiv.org/pdf/2510.22436v1)**

> **作者:** Ghazal Farhani; Taufiq Rahman; Syed Mostaquim Ali; Andrew Liu; Mohamed Zaki; Dominique Charlebois; Benoit Anctil
>
> **备注:** 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC), pp. 1441--1448, Sept. 2024
>
> **摘要:** Because 3D structure of a roadway environment can be characterized directly by a Light Detection and Ranging (LiDAR) sensors, they can be used to obtain exceptional situational awareness for assitive and autonomous driving systems. Although LiDARs demonstrate good performance in clean and clear weather conditions, their performance significantly deteriorates in adverse weather conditions such as those involving atmospheric precipitation. This may render perception capabilities of autonomous systems that use LiDAR data in learning based models to perform object detection and ranging ineffective. While efforts have been made to enhance the accuracy of these models, the extent of signal degradation under various weather conditions remains largely not quantified. In this study, we focus on the performance of an automotive grade LiDAR in snowy conditions in order to develop a physics-based model that examines failure modes of a LiDAR sensor. Specifically, we investigated how the LiDAR signal attenuates with different snowfall rates and how snow particles near the source serve as small but efficient reflectors. Utilizing our model, we transform data from clear conditions to simulate snowy scenarios, enabling a comparison of our synthetic data with actual snowy conditions. Furthermore, we employ this synthetic data, representative of different snowfall rates, to explore the impact on a pre-trained object detection model, assessing its performance under varying levels of snowfall
>
---
#### [new 048] Morphology-Aware KOA Classification: Integrating Graph Priors with Vision Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对膝骨关节炎（KOA）X光片分类难题，提出融合形态图结构与视觉模型的多模态框架。通过SAM分割构建解剖图谱，利用互信息最大化对齐图嵌入与影像特征，引入临床导向的形态先验，显著提升分类准确率与F1分数，验证了结构信息在医学影像分析中的关键作用。**

- **链接: [http://arxiv.org/pdf/2510.21801v1](http://arxiv.org/pdf/2510.21801v1)**

> **作者:** Marouane Tliba; Mohamed Amine Kerkouri; Yassine Nasser; Nour Aburaed; Aladine Chetouani; Ulas Bagci; Rachid Jennane
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Knee osteoarthritis (KOA) diagnosis from radiographs remains challenging due to the subtle morphological details that standard deep learning models struggle to capture effectively. We propose a novel multimodal framework that combines anatomical structure with radiographic features by integrating a morphological graph representation - derived from Segment Anything Model (SAM) segmentations - with a vision encoder. Our approach enforces alignment between geometry-informed graph embeddings and radiographic features through mutual information maximization, significantly improving KOA classification accuracy. By constructing graphs from anatomical features, we introduce explicit morphological priors that mirror clinical assessment criteria, enriching the feature space and enhancing the model's inductive bias. Experiments on the Osteoarthritis Initiative dataset demonstrate that our approach surpasses single-modality baselines by up to 10\% in accuracy (reaching nearly 80\%), while outperforming existing state-of-the-art methods by 8\% in accuracy and 11\% in F1 score. These results underscore the critical importance of incorporating anatomical structure into radiographic analysis for accurate KOA severity grading.
>
---
#### [new 049] Lookahead Anchoring: Preserving Character Identity in Audio-Driven Human Animation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对音频驱动人体动画中的身份漂移问题，提出Lookahead Anchoring方法。通过利用未来时间步的键帧作为导向锚点，实现身份一致性与自然运动的平衡。无需额外键帧生成，支持自参考关键帧，提升唇同步与视觉质量，在多个模型上验证有效。**

- **链接: [http://arxiv.org/pdf/2510.23581v1](http://arxiv.org/pdf/2510.23581v1)**

> **作者:** Junyoung Seo; Rodrigo Mira; Alexandros Haliassos; Stella Bounareli; Honglie Chen; Linh Tran; Seungryong Kim; Zoe Landgraf; Jie Shen
>
> **备注:** Project page: https://lookahead-anchoring.github.io
>
> **摘要:** Audio-driven human animation models often suffer from identity drift during temporal autoregressive generation, where characters gradually lose their identity over time. One solution is to generate keyframes as intermediate temporal anchors that prevent degradation, but this requires an additional keyframe generation stage and can restrict natural motion dynamics. To address this, we propose Lookahead Anchoring, which leverages keyframes from future timesteps ahead of the current generation window, rather than within it. This transforms keyframes from fixed boundaries into directional beacons: the model continuously pursues these future anchors while responding to immediate audio cues, maintaining consistent identity through persistent guidance. This also enables self-keyframing, where the reference image serves as the lookahead target, eliminating the need for keyframe generation entirely. We find that the temporal lookahead distance naturally controls the balance between expressivity and consistency: larger distances allow for greater motion freedom, while smaller ones strengthen identity adherence. When applied to three recent human animation models, Lookahead Anchoring achieves superior lip synchronization, identity preservation, and visual quality, demonstrating improved temporal conditioning across several different architectures. Video results are available at the following link: https://lookahead-anchoring.github.io.
>
---
#### [new 050] Estimation of Fireproof Structure Class and Construction Year for Disaster Risk Assessment
- **分类: cs.CV**

- **简介: 该论文提出一种多任务学习模型，通过建筑外立面图像预测建造年份、结构类型和物业类型，进而推断防火等级（H/T/M），以解决日本二手房产中关键建筑元数据缺失问题。模型在大规模图像数据集上训练，实现高精度回归与分类，支持保险定价与灾害风险评估。**

- **链接: [http://arxiv.org/pdf/2510.22683v1](http://arxiv.org/pdf/2510.22683v1)**

> **作者:** Hibiki Ayabe; Kazushi Okamoto; Koki Karube; Atsushi Shibata; Kei Harada
>
> **摘要:** Structural fireproof classification is vital for disaster risk assessment and insurance pricing in Japan. However, key building metadata such as construction year and structure type are often missing or outdated, particularly in the second-hand housing market. This study proposes a multi-task learning model that predicts these attributes from facade images. The model jointly estimates the construction year, building structure, and property type, from which the structural fireproof class - defined as H (non-fireproof), T (semi-fireproof), or M (fireproof) - is derived via a rule-based mapping based on official insurance criteria. We trained and evaluated the model using a large-scale dataset of Japanese residential images, applying rigorous filtering and deduplication. The model achieved high accuracy in construction-year regression and robust classification across imbalanced categories. Qualitative analyses show that it captures visual cues related to building age and materials. Our approach demonstrates the feasibility of scalable, interpretable, image-based risk-profiling systems, offering potential applications in insurance, urban planning, and disaster preparedness.
>
---
#### [new 051] Gen-LangSplat: Generalized Language Gaussian Splatting with Pre-Trained Feature Compression
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D开放词汇语言场构建中的场景特定训练瓶颈，提出Gen-LangSplat，通过引入预训练的通用自动编码器替代原有场景自适应编码器，实现无需微调的高效特征压缩。该方法显著提升部署可扩展性，同时保持甚至超越原方法的查询性能。**

- **链接: [http://arxiv.org/pdf/2510.22930v1](http://arxiv.org/pdf/2510.22930v1)**

> **作者:** Pranav Saxena
>
> **摘要:** Modeling open-vocabulary language fields in 3D is essential for intuitive human-AI interaction and querying within physical environments. State-of-the-art approaches, such as LangSplat, leverage 3D Gaussian Splatting to efficiently construct these language fields, encoding features distilled from high-dimensional models like CLIP. However, this efficiency is currently offset by the requirement to train a scene-specific language autoencoder for feature compression, introducing a costly, per-scene optimization bottleneck that hinders deployment scalability. In this work, we introduce Gen-LangSplat, that eliminates this requirement by replacing the scene-wise autoencoder with a generalized autoencoder, pre-trained extensively on the large-scale ScanNet dataset. This architectural shift enables the use of a fixed, compact latent space for language features across any new scene without any scene-specific training. By removing this dependency, our entire language field construction process achieves a efficiency boost while delivering querying performance comparable to, or exceeding, the original LangSplat method. To validate our design choice, we perform a thorough ablation study empirically determining the optimal latent embedding dimension and quantifying representational fidelity using Mean Squared Error and cosine similarity between the original and reprojected 512-dimensional CLIP embeddings. Our results demonstrate that generalized embeddings can efficiently and accurately support open-vocabulary querying in novel 3D scenes, paving the way for scalable, real-time interactive 3D AI applications.
>
---
#### [new 052] DeepSalt: Bridging Laboratory and Satellite Spectra through Domain Adaptation and Knowledge Distillation for Large-Scale Soil Salinity Estimation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对土壤盐渍化大范围监测难题，提出DeepSalt框架，通过域适应与知识蒸馏，实现实验室光谱向卫星遥感数据的精准迁移。解决了地面采样成本高、卫星数据解释性弱的问题，实现了无需大量实地采样的高精度大尺度土壤盐度估算。**

- **链接: [http://arxiv.org/pdf/2510.23124v1](http://arxiv.org/pdf/2510.23124v1)**

> **作者:** Rupasree Dey; Abdul Matin; Everett Lewark; Tanjim Bin Faruk; Andrei Bachinin; Sam Leuthold; M. Francesca Cotrufo; Shrideep Pallickara; Sangmi Lee Pallickara
>
> **摘要:** Soil salinization poses a significant threat to both ecosystems and agriculture because it limits plants' ability to absorb water and, in doing so, reduces crop productivity. This phenomenon alters the soil's spectral properties, creating a measurable relationship between salinity and light reflectance that enables remote monitoring. While laboratory spectroscopy provides precise measurements, its reliance on in-situ sampling limits scalability to regional or global levels. Conversely, hyperspectral satellite imagery enables wide-area observation but lacks the fine-grained interpretability of laboratory instruments. To bridge this gap, we introduce DeepSalt, a deep-learning-based spectral transfer framework that leverages knowledge distillation and a novel Spectral Adaptation Unit to transfer high-resolution spectral insights from laboratory-based spectroscopy to satellite-based hyperspectral sensing. Our approach eliminates the need for extensive ground sampling while enabling accurate, large-scale salinity estimation, as demonstrated through comprehensive empirical benchmarks. DeepSalt achieves significant performance gains over methods without explicit domain adaptation, underscoring the impact of the proposed Spectral Adaptation Unit and the knowledge distillation strategy. The model also effectively generalized to unseen geographic regions, explaining a substantial portion of the salinity variance.
>
---
#### [new 053] GALA: A GlobAl-LocAl Approach for Multi-Source Active Domain Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出GALA方法，解决多源主动域适应（MS-ADA）中如何高效选择目标域标注样本的问题。针对跨类别差异与多源域变异挑战，GALA结合全局聚类与局部选择策略，无需额外参数即可显著提升性能，仅用1%标注即逼近全监督效果。**

- **链接: [http://arxiv.org/pdf/2510.22214v1](http://arxiv.org/pdf/2510.22214v1)**

> **作者:** Juepeng Zheng; Peifeng Zhang; Yibin Wen; Qingmei Li; Yang Zhang; Haohuan Fu
>
> **摘要:** Domain Adaptation (DA) provides an effective way to tackle target-domain tasks by leveraging knowledge learned from source domains. Recent studies have extended this paradigm to Multi-Source Domain Adaptation (MSDA), which exploits multiple source domains carrying richer and more diverse transferable information. However, a substantial performance gap still remains between adaptation-based methods and fully supervised learning. In this paper, we explore a more practical and challenging setting, named Multi-Source Active Domain Adaptation (MS-ADA), to further enhance target-domain performance by selectively acquiring annotations from the target domain. The key difficulty of MS-ADA lies in designing selection criteria that can jointly handle inter-class diversity and multi-source domain variation. To address these challenges, we propose a simple yet effective GALA strategy (GALA), which combines a global k-means clustering step for target-domain samples with a cluster-wise local selection criterion, effectively tackling the above two issues in a complementary manner. Our proposed GALA is plug-and-play and can be seamlessly integrated into existing DA frameworks without introducing any additional trainable parameters. Extensive experiments on three standard DA benchmarks demonstrate that GALA consistently outperforms prior active learning and active DA methods, achieving performance comparable to the fully-supervised upperbound while using only 1% of the target annotations.
>
---
#### [new 054] Precise classification of low quality G-banded Chromosome Images by reliability metrics and data pruning classifier
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对低质量G显带染色体图像的精确分类问题，提出基于可靠性阈值和数据剪枝的分类方法。通过改进特征工程与多模型融合，显著提升低质量图像下的分类精度，尤其适用于资源匮乏地区的病理实验室。**

- **链接: [http://arxiv.org/pdf/2510.21827v1](http://arxiv.org/pdf/2510.21827v1)**

> **作者:** Mojtaba Moattari
>
> **摘要:** In the last decade, due to high resolution cameras and accurate meta-phase analyzes, the accuracy of chromosome classification has improved substantially. However, current Karyotyping systems demand large number of high quality train data to have an adequately plausible Precision per each chromosome. Such provision of high quality train data with accurate devices are not yet accomplished in some out-reached pathological laboratories. To prevent false positive detections in low-cost systems and low-quality images settings, this paper improves the classification Precision of chromosomes using proposed reliability thresholding metrics and deliberately engineered features. The proposed method has been evaluated using a variation of deep Alex-Net neural network, SVM, K Nearest-Neighbors, and their cascade pipelines to an automated filtering of semi-straight chromosome. The classification results have highly improved over 90% for the chromosomes with more common defections and translocations. Furthermore, a comparative analysis over the proposed thresholding metrics has been conducted and the best metric is bolded with its salient characteristics. The high Precision results provided for a very low-quality G-banding database verifies suitability of the proposed metrics and pruning method for Karyotyping facilities in poor countries and lowbudget pathological laboratories.
>
---
#### [new 055] Reconnaissance Automatique des Langues des Signes : Une Approche Hybridée CNN-LSTM Basée sur Mediapipe
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于手语识别任务，旨在解决聋哑人群与健听者间沟通障碍问题。提出基于Mediapipe提取关键点、CNN-LSTM混合模型进行实时手语识别，实现92%平均准确率，可应用于医疗、教育等领域。**

- **链接: [http://arxiv.org/pdf/2510.22011v1](http://arxiv.org/pdf/2510.22011v1)**

> **作者:** Fraisse Sacré Takouchouang; Ho Tuong Vinh
>
> **备注:** in French language
>
> **摘要:** Sign languages play a crucial role in the communication of deaf communities, but they are often marginalized, limiting access to essential services such as healthcare and education. This study proposes an automatic sign language recognition system based on a hybrid CNN-LSTM architecture, using Mediapipe for gesture keypoint extraction. Developed with Python, TensorFlow and Streamlit, the system provides real-time gesture translation. The results show an average accuracy of 92\%, with very good performance for distinct gestures such as ``Hello'' and ``Thank you''. However, some confusions remain for visually similar gestures, such as ``Call'' and ``Yes''. This work opens up interesting perspectives for applications in various fields such as healthcare, education and public services.
>
---
#### [new 056] SCoPE VLM: Selective Context Processing for Efficient Document Navigation in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉语言模型在长文档导航中的效率问题，提出SCoPE VLM框架。通过链式滚动机制与强化学习，实现对文档的精准、低内存的逐段聚焦阅读，解决现有方法内存占用高、缺乏人类阅读行为模拟的问题，首次建模了多页文档问答中的代理式阅读模式。**

- **链接: [http://arxiv.org/pdf/2510.21850v1](http://arxiv.org/pdf/2510.21850v1)**

> **作者:** Gyubeum Lim; Yemo Koo; Vijay Krishna Madisetti
>
> **摘要:** Understanding long-context visual information remains a fundamental challenge for vision-language models, particularly in agentic tasks such as GUI control and web navigation. While web pages and GUI environments are inherently structured documents, current VLMs typically neglect decision-oriented document understanding in their training objectives. Existing approaches primarily extend visual embeddings to process long, high-resolution inputs, but these methods are memory-intensive and impractical for locally deployable solutions. To address these issues, we propose SCoPE VLM, a document navigation expert that leverages a novel Chain of Scroll mechanism to selectively and recursively navigate documents, focusing exclusively on relevant segments. We introduce a dedicated data generation pipeline to construct informative Chain of Scroll trajectories and Episodic Group Relative Policy Optimization, a tailored reinforcement learning method to reduce the gap between training and inference. Our method substantially reduces memory usage and effectively models human-like reading behaviors. To the best of our knowledge, SCoPE VLM is the first framework to explicitly model agentic reading patterns in multi-page document question answering, advancing the capabilities of multimodal agents.
>
---
#### [new 057] Survey of Multimodal Geospatial Foundation Models: Techniques, Applications, and Challenges
- **分类: cs.CV**

- **简介: 该论文聚焦多模态地理空间基础模型（GFMs），针对遥感数据的多模态、多分辨率与时空异构性挑战，系统综述了视觉与视觉-语言模态的融合技术、对齐方法及应用。论文梳理了核心架构、训练范式与下游任务表现，涵盖土地覆盖、灾害响应等场景，指出了泛化、可解释性等关键问题，推动遥感智能分析发展。**

- **链接: [http://arxiv.org/pdf/2510.22964v1](http://arxiv.org/pdf/2510.22964v1)**

> **作者:** Liling Yang; Ning Chen; Jun Yue; Yidan Liu; Jiayi Ma; Pedram Ghamisi; Antonio Plaza; Leyuan Fang
>
> **摘要:** Foundation models have transformed natural language processing and computer vision, and their impact is now reshaping remote sensing image analysis. With powerful generalization and transfer learning capabilities, they align naturally with the multimodal, multi-resolution, and multi-temporal characteristics of remote sensing data. To address unique challenges in the field, multimodal geospatial foundation models (GFMs) have emerged as a dedicated research frontier. This survey delivers a comprehensive review of multimodal GFMs from a modality-driven perspective, covering five core visual and vision-language modalities. We examine how differences in imaging physics and data representation shape interaction design, and we analyze key techniques for alignment, integration, and knowledge transfer to tackle modality heterogeneity, distribution shifts, and semantic gaps. Advances in training paradigms, architectures, and task-specific adaptation strategies are systematically assessed alongside a wealth of emerging benchmarks. Representative multimodal visual and vision-language GFMs are evaluated across ten downstream tasks, with insights into their architectures, performance, and application scenarios. Real-world case studies, spanning land cover mapping, agricultural monitoring, disaster response, climate studies, and geospatial intelligence, demonstrate the practical potential of GFMs. Finally, we outline pressing challenges in domain generalization, interpretability, efficiency, and privacy, and chart promising avenues for future research.
>
---
#### [new 058] Efficient Large-Deformation Medical Image Registration via Recurrent Dynamic Correlation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医学图像大形变配准任务，解决深度学习方法在处理大变形时效率与精度难以兼顾的问题。提出基于循环相关性的动态匹配框架，通过迭代重定位匹配区域并引入轻量级记忆模块，高效捕捉长程对应关系，显著降低计算量且保持高精度。**

- **链接: [http://arxiv.org/pdf/2510.22380v1](http://arxiv.org/pdf/2510.22380v1)**

> **作者:** Tianran Li; Marius Staring; Yuchuan Qiao
>
> **摘要:** Deformable image registration estimates voxel-wise correspondences between images through spatial transformations, and plays a key role in medical imaging. While deep learning methods have significantly reduced runtime, efficiently handling large deformations remains a challenging task. Convolutional networks aggregate local features but lack direct modeling of voxel correspondences, promoting recent works to explore explicit feature matching. Among them, voxel-to-region matching is more efficient for direct correspondence modeling by computing local correlation features whithin neighbourhoods, while region-to-region matching incurs higher redundancy due to excessive correlation pairs across large regions. However, the inherent locality of voxel-to-region matching hinders the capture of long-range correspondences required for large deformations. To address this, we propose a Recurrent Correlation-based framework that dynamically relocates the matching region toward more promising positions. At each step, local matching is performed with low cost, and the estimated offset guides the next search region, supporting efficient convergence toward large deformations. In addition, we uses a lightweight recurrent update module with memory capacity and decouples motion-related and texture features to suppress semantic redundancy. We conduct extensive experiments on brain MRI and abdominal CT datasets under two settings: with and without affine pre-registration. Results show that our method exibits a strong accuracy-computation trade-off, surpassing or matching the state-of-the-art performance. For example, it achieves comparable performance on the non-affine OASIS dataset, while using only 9.5% of the FLOPs and running 96% faster than RDP, a representative high-performing method.
>
---
#### [new 059] Scaling Up Occupancy-centric Driving Scene Generation: Dataset and Method
- **分类: cs.CV**

- **简介: 该论文聚焦于自动驾驶中的场景生成任务，针对语义占用数据稀缺问题，构建了大规模Nuplan-Occ数据集，并提出统一框架联合生成高保真语义占用、多视角视频与激光点云。通过时空解耦架构与新型渲染、传感器建模策略，实现4D动态场景的高质量生成与跨模态一致性，显著提升生成效果与下游应用价值。**

- **链接: [http://arxiv.org/pdf/2510.22973v1](http://arxiv.org/pdf/2510.22973v1)**

> **作者:** Bohan Li; Xin Jin; Hu Zhu; Hongsi Liu; Ruikai Li; Jiazhe Guo; Kaiwen Cai; Chao Ma; Yueming Jin; Hao Zhao; Xiaokang Yang; Wenjun Zeng
>
> **备注:** https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/tree/v2
>
> **摘要:** Driving scene generation is a critical domain for autonomous driving, enabling downstream applications, including perception and planning evaluation. Occupancy-centric methods have recently achieved state-of-the-art results by offering consistent conditioning across frames and modalities; however, their performance heavily depends on annotated occupancy data, which still remains scarce. To overcome this limitation, we curate Nuplan-Occ, the largest semantic occupancy dataset to date, constructed from the widely used Nuplan benchmark. Its scale and diversity facilitate not only large-scale generative modeling but also autonomous driving downstream applications. Based on this dataset, we develop a unified framework that jointly synthesizes high-quality semantic occupancy, multi-view videos, and LiDAR point clouds. Our approach incorporates a spatio-temporal disentangled architecture to support high-fidelity spatial expansion and temporal forecasting of 4D dynamic occupancy. To bridge modal gaps, we further propose two novel techniques: a Gaussian splatting-based sparse point map rendering strategy that enhances multi-view video generation, and a sensor-aware embedding strategy that explicitly models LiDAR sensor properties for realistic multi-LiDAR simulation. Extensive experiments demonstrate that our method achieves superior generation fidelity and scalability compared to existing approaches, and validates its practical value in downstream tasks. Repo: https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/tree/v2
>
---
#### [new 060] DecoDINO: 3D Human-Scene Contact Prediction with Semantic Classification
- **分类: cs.CV**

- **简介: 该论文针对3D人体与场景接触预测任务，解决现有方法在软表面、遮挡、儿童等场景下精度不足及仅支持二值接触的问题。提出DecoDINO模型，采用双DINOv2编码器与跨注意力机制，结合轻量MLP实现语义级接触分类，在DAMON基准上显著提升F1分数与定位精度。**

- **链接: [http://arxiv.org/pdf/2510.23203v1](http://arxiv.org/pdf/2510.23203v1)**

> **作者:** Lukas Bierling; Davide Pasero; Fleur Dolmans; Helia Ghasemi; Angelo Broere
>
> **摘要:** Accurate vertex-level contact prediction between humans and surrounding objects is a prerequisite for high fidelity human object interaction models used in robotics, AR/VR, and behavioral simulation. DECO was the first in the wild estimator for this task but is limited to binary contact maps and struggles with soft surfaces, occlusions, children, and false-positive foot contacts. We address these issues and introduce DecoDINO, a three-branch network based on DECO's framework. It uses two DINOv2 ViT-g/14 encoders, class-balanced loss weighting to reduce bias, and patch-level cross-attention for improved local reasoning. Vertex features are finally passed through a lightweight MLP with a softmax to assign semantic contact labels. We also tested a vision-language model (VLM) to integrate text features, but the simpler architecture performed better and was used instead. On the DAMON benchmark, DecoDINO (i) raises the binary-contact F1 score by 7$\%$, (ii) halves the geodesic error, and (iii) augments predictions with object-level semantic labels. Ablation studies show that LoRA fine-tuning and the dual encoders are key to these improvements. DecoDINO outperformed the challenge baseline in both tasks of the DAMON Challenge. Our code is available at https://github.com/DavidePasero/deco/tree/main.
>
---
#### [new 061] Moving Beyond Diffusion: Hierarchy-to-Hierarchy Autoregression for fMRI-to-Image Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦于fMRI到图像的重建任务，旨在克服扩散模型因固定引导导致层次神经信息丢失的问题。提出MindHier框架，通过分层编码、层级对齐与尺度感知的粗到细自回归引导，实现更高效、符合认知规律的图像生成，显著提升语义保真度与推理速度。**

- **链接: [http://arxiv.org/pdf/2510.22335v1](http://arxiv.org/pdf/2510.22335v1)**

> **作者:** Xu Zhang; Ruijie Quan; Wenguan Wang; Yi Yang
>
> **摘要:** Reconstructing visual stimuli from fMRI signals is a central challenge bridging machine learning and neuroscience. Recent diffusion-based methods typically map fMRI activity to a single high-level embedding, using it as fixed guidance throughout the entire generation process. However, this fixed guidance collapses hierarchical neural information and is misaligned with the stage-dependent demands of image reconstruction. In response, we propose MindHier, a coarse-to-fine fMRI-to-image reconstruction framework built on scale-wise autoregressive modeling. MindHier introduces three components: a Hierarchical fMRI Encoder to extract multi-level neural embeddings, a Hierarchy-to-Hierarchy Alignment scheme to enforce layer-wise correspondence with CLIP features, and a Scale-Aware Coarse-to-Fine Neural Guidance strategy to inject these embeddings into autoregression at matching scales. These designs make MindHier an efficient and cognitively-aligned alternative to diffusion-based methods by enabling a hierarchical reconstruction process that synthesizes global semantics before refining local details, akin to human visual perception. Extensive experiments on the NSD dataset show that MindHier achieves superior semantic fidelity, 4.67x faster inference, and more deterministic results than the diffusion-based baselines.
>
---
#### [new 062] STG-Avatar: Animatable Human Avatars via Spacetime Gaussian
- **分类: cs.CV**

- **简介: 该论文聚焦于单目视频驱动的高保真可动画人体虚拟化身重建任务。针对现有方法在非刚性形变（如衣物）和动态区域（如快速运动肢体）建模上的不足，提出STG-Avatar框架，融合时空高斯与线性混合蒙皮，通过光学流引导自适应优化，实现高效精准的动态细节重建与实时渲染。**

- **链接: [http://arxiv.org/pdf/2510.22140v1](http://arxiv.org/pdf/2510.22140v1)**

> **作者:** Guangan Jiang; Tianzi Zhang; Dong Li; Zhenjun Zhao; Haoang Li; Mingrui Li; Hongyu Wang
>
> **备注:** Accepted by the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Realistic animatable human avatars from monocular videos are crucial for advancing human-robot interaction and enhancing immersive virtual experiences. While recent research on 3DGS-based human avatars has made progress, it still struggles with accurately representing detailed features of non-rigid objects (e.g., clothing deformations) and dynamic regions (e.g., rapidly moving limbs). To address these challenges, we present STG-Avatar, a 3DGS-based framework for high-fidelity animatable human avatar reconstruction. Specifically, our framework introduces a rigid-nonrigid coupled deformation framework that synergistically integrates Spacetime Gaussians (STG) with linear blend skinning (LBS). In this hybrid design, LBS enables real-time skeletal control by driving global pose transformations, while STG complements it through spacetime adaptive optimization of 3D Gaussians. Furthermore, we employ optical flow to identify high-dynamic regions and guide the adaptive densification of 3D Gaussians in these regions. Experimental results demonstrate that our method consistently outperforms state-of-the-art baselines in both reconstruction quality and operational efficiency, achieving superior quantitative metrics while retaining real-time rendering capabilities. Our code is available at https://github.com/jiangguangan/STG-Avatar
>
---
#### [new 063] Discovering Latent Graphs with GFlowNets for Diverse Conditional Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Rainbow框架，用于条件图像生成中的多样性问题。针对条件不确定性导致的输出单一性，通过GFlowNets构建潜空间图，分解条件为多样潜表示，生成多条路径对应不同图像，提升多样性与保真度，适用于自然与医学图像生成任务。**

- **链接: [http://arxiv.org/pdf/2510.22107v1](http://arxiv.org/pdf/2510.22107v1)**

> **作者:** Bailey Trang; Parham Saremi; Alan Q. Wang; Fangrui Huang; Zahra TehraniNasab; Amar Kumar; Tal Arbel; Li Fei-Fei; Ehsan Adeli
>
> **摘要:** Capturing diversity is crucial in conditional and prompt-based image generation, particularly when conditions contain uncertainty that can lead to multiple plausible outputs. To generate diverse images reflecting this diversity, traditional methods often modify random seeds, making it difficult to discern meaningful differences between samples, or diversify the input prompt, which is limited in verbally interpretable diversity. We propose Rainbow, a novel conditional image generation framework, applicable to any pretrained conditional generative model, that addresses inherent condition/prompt uncertainty and generates diverse plausible images. Rainbow is based on a simple yet effective idea: decomposing the input condition into diverse latent representations, each capturing an aspect of the uncertainty and generating a distinct image. First, we integrate a latent graph, parameterized by Generative Flow Networks (GFlowNets), into the prompt representation computation. Second, leveraging GFlowNets' advanced graph sampling capabilities to capture uncertainty and output diverse trajectories over the graph, we produce multiple trajectories that collectively represent the input condition, leading to diverse condition representations and corresponding output images. Evaluations on natural image and medical image datasets demonstrate Rainbow's improvement in both diversity and fidelity across image synthesis, image generation, and counterfactual generation tasks.
>
---
#### [new 064] Generative AI in Depth: A Survey of Recent Advances, Model Variants, and Real-World Applications
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于生成式人工智能综述任务，旨在梳理GAN、VAE和扩散模型的最新进展。针对技术繁杂、发展迅速导致的研究跟进困难问题，构建了系统性分类框架，总结关键技术突破与应用，分析伦理风险，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2510.21887v1](http://arxiv.org/pdf/2510.21887v1)**

> **作者:** Shamim Yazdani; Akansha Singh; Nripsuta Saxena; Zichong Wang; Avash Palikhe; Deng Pan; Umapada Pal; Jie Yang; Wenbin Zhang
>
> **备注:** Accepted by the Journal of Big Data
>
> **摘要:** In recent years, deep learning based generative models, particularly Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Diffusion Models (DMs), have been instrumental in in generating diverse, high-quality content across various domains, such as image and video synthesis. This capability has led to widespread adoption of these models and has captured strong public interest. As they continue to advance at a rapid pace, the growing volume of research, expanding application areas, and unresolved technical challenges make it increasingly difficult to stay current. To address this need, this survey introduces a comprehensive taxonomy that organizes the literature and provides a cohesive framework for understanding the development of GANs, VAEs, and DMs, including their many variants and combined approaches. We highlight key innovations that have improved the quality, diversity, and controllability of generated outputs, reflecting the expanding potential of generative artificial intelligence. In addition to summarizing technical progress, we examine rising ethical concerns, including the risks of misuse and the broader societal impact of synthetic media. Finally, we outline persistent challenges and propose future research directions, offering a structured and forward looking perspective for researchers in this fast evolving field.
>
---
#### [new 065] Evaluating ChatGPT's Performance in Classifying Pneumonia from Chest X-Ray Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在评估GPT-4o在零样本条件下识别胸部X光片肺炎的能力。研究使用400张平衡数据集，对比四种提示设计，发现简洁的特征导向提示效果最佳（准确率74%），表明模型尚需提升视觉推理与临床适配性。**

- **链接: [http://arxiv.org/pdf/2510.21839v1](http://arxiv.org/pdf/2510.21839v1)**

> **作者:** Pragna Prahallad; Pranathi Prahallad
>
> **摘要:** In this study, we evaluate the ability of OpenAI's gpt-4o model to classify chest X-ray images as either NORMAL or PNEUMONIA in a zero-shot setting, without any prior fine-tuning. A balanced test set of 400 images (200 from each class) was used to assess performance across four distinct prompt designs, ranging from minimal instructions to detailed, reasoning-based prompts. The results indicate that concise, feature-focused prompts achieved the highest classification accuracy of 74\%, whereas reasoning-oriented prompts resulted in lower performance. These findings highlight that while ChatGPT exhibits emerging potential for medical image interpretation, its diagnostic reliability remains limited. Continued advances in visual reasoning and domain-specific adaptation are required before such models can be safely applied in clinical practice.
>
---
#### [new 066] Comparative Analysis of Object Detection Algorithms for Surface Defect Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于工业表面缺陷检测任务，旨在比较六种目标检测算法在NEU-DET数据集上的性能。通过评估准确率、速度与鲁棒性，发现YOLOv11在检测精度和效率上显著优于其他模型，尤其擅长识别细微缺陷。**

- **链接: [http://arxiv.org/pdf/2510.21811v1](http://arxiv.org/pdf/2510.21811v1)**

> **作者:** Arpan Maity; Tamal Ghosh
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** This article compares the performance of six prominent object detection algorithms, YOLOv11, RetinaNet, Fast R-CNN, YOLOv8, RT-DETR, and DETR, on the NEU-DET surface defect detection dataset, comprising images representing various metal surface defects, a crucial application in industrial quality control. Each model's performance was assessed regarding detection accuracy, speed, and robustness across different defect types such as scratches, inclusions, and rolled-in scales. YOLOv11, a state-of-the-art real-time object detection algorithm, demonstrated superior performance compared to the other methods, achieving a remarkable 70% higher accuracy on average. This improvement can be attributed to YOLOv11s enhanced feature extraction capabilities and ability to process the entire image in a single forward pass, making it faster and more efficient in detecting minor surface defects. Additionally, YOLOv11's architecture optimizations, such as improved anchor box generation and deeper convolutional layers, contributed to more precise localization of defects. In conclusion, YOLOv11's outstanding performance in accuracy and speed solidifies its position as the most effective model for surface defect detection on the NEU dataset, surpassing competing algorithms by a substantial margin.
>
---
#### [new 067] RoboSVG: A Unified Framework for Interactive SVG Generation with Multi-modal Guidance
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出RoboSVG框架，解决多模态引导下交互式SVG生成问题。通过文本、图像、数值信号联合指导，实现高保真SVG合成。构建百万级数据集RoboDraw，支持四类任务，显著提升生成质量与查询一致性，推动交互式矢量图形生成发展。**

- **链接: [http://arxiv.org/pdf/2510.22684v1](http://arxiv.org/pdf/2510.22684v1)**

> **作者:** Jiuniu Wang; Gongjie Zhang; Quanhao Qian; Junlong Gao; Deli Zhao; Ran Xu
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Scalable Vector Graphics (SVGs) are fundamental to digital design and robot control, encoding not only visual structure but also motion paths in interactive drawings. In this work, we introduce RoboSVG, a unified multimodal framework for generating interactive SVGs guided by textual, visual, and numerical signals. Given an input query, the RoboSVG model first produces multimodal guidance, then synthesizes candidate SVGs through dedicated generation modules, and finally refines them under numerical guidance to yield high-quality outputs. To support this framework, we construct RoboDraw, a large-scale dataset of one million examples, each pairing an SVG generation condition (e.g., text, image, and partial SVG) with its corresponding ground-truth SVG code. RoboDraw dataset enables systematic study of four tasks, including basic generation (Text-to-SVG, Image-to-SVG) and interactive generation (PartialSVG-to-SVG, PartialImage-to-SVG). Extensive experiments demonstrate that RoboSVG achieves superior query compliance and visual fidelity across tasks, establishing a new state of the art in versatile SVG generation. The dataset and source code of this project will be publicly available soon.
>
---
#### [new 068] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity-Aware Tree
- **分类: cs.CV**

- **简介: 该论文提出VADTree，用于无训练视频异常检测。针对固定时长采样难以捕捉不同时间跨度异常的问题，利用预训练模型识别事件边界，构建分层粒度树结构，通过多维先验与大语言模型实现节点级异常感知与推理，最终融合多粒度得分，提升检测性能并减少采样片段。**

- **链接: [http://arxiv.org/pdf/2510.22693v1](http://arxiv.org/pdf/2510.22693v1)**

> **作者:** Wenlong Li; Yifei Xu; Yuan Rao; Zhenhua Wang; Shuiguang Deng
>
> **备注:** NeurIPS 2025 Camera Ready
>
> **摘要:** Video anomaly detection (VAD) focuses on identifying anomalies in videos. Supervised methods demand substantial in-domain training data and fail to deliver clear explanations for anomalies. In contrast, training-free methods leverage the knowledge reserves and language interactivity of large pre-trained models to detect anomalies. However, the current fixed-length temporal window sampling approaches struggle to accurately capture anomalies with varying temporal spans. Therefore, we propose VADTree that utilizes a Hierarchical Granularityaware Tree (HGTree) structure for flexible sampling in VAD. VADTree leverages the knowledge embedded in a pre-trained Generic Event Boundary Detection (GEBD) model to characterize potential anomaly event boundaries. Specifically, VADTree decomposes the video into generic event nodes based on boundary confidence, and performs adaptive coarse-fine hierarchical structuring and redundancy removal to construct the HGTree. Then, the multi-dimensional priors are injected into the visual language models (VLMs) to enhance the node-wise anomaly perception, and anomaly reasoning for generic event nodes is achieved via large language models (LLMs). Finally, an inter-cluster node correlation method is used to integrate the multi-granularity anomaly scores. Extensive experiments on three challenging datasets demonstrate that VADTree achieves state-of-the-art performance in training-free settings while drastically reducing the number of sampled video segments. The code will be available at https://github.com/wenlongli10/VADTree.
>
---
#### [new 069] Robust Atypical Mitosis Classification with DenseNet121: Stain-Aware Augmentation and Hybrid Loss for Domain Generalization
- **分类: cs.CV**

- **简介: 该论文针对病理图像中非典型有丝分裂的分类任务，解决染色和扫描仪差异导致的域偏移及类别不平衡问题。提出基于DenseNet-121的框架，结合染色感知增强、几何与强度变换及混合损失函数，实现强泛化能力，在多域测试中达85.0%平衡准确率。**

- **链接: [http://arxiv.org/pdf/2510.22630v1](http://arxiv.org/pdf/2510.22630v1)**

> **作者:** Adinath Dukre; Ankan Deria; Yutong Xie; Imran Razzak
>
> **备注:** MIDOG 2025 MICCAI Workshop accepted
>
> **摘要:** Atypical mitotic figures are important biomarkers of tumor aggressiveness in histopathology, yet reliable recognition remains challenging due to severe class imbalance and variability across imaging domains. We present a DenseNet-121-based framework tailored for atypical mitosis classification in the MIDOG 2025 (Track 2) setting. Our method integrates stain-aware augmentation (Macenko), geometric and intensity transformations, and imbalance-aware learning via weighted sampling with a hybrid objective combining class-weighted binary cross-entropy and focal loss. Trained end-to-end with AdamW and evaluated across multiple independent domains, the model demonstrates strong generalization under scanner and staining shifts, achieving balanced accuracy 85.0%, AUROC 0.927, sensitivity 89.2%, and specificity 80.9% on the official test set. These results indicate that combining DenseNet-121 with stain-aware augmentation and imbalance-adaptive objectives yields a robust, domain-generalizable framework for atypical mitosis classification suitable for real-world computational pathology workflows.
>
---
#### [new 070] Seeing the Unseen: Towards Zero-Shot Inspection for Wind Turbine Blades using Knowledge-Augmented Vision Language Models
- **分类: cs.CV**

- **简介: 该论文针对风力机叶片损伤检测中标签数据稀缺的问题，提出一种基于知识增强视觉语言模型的零样本检测框架。通过构建多模态知识库并结合检索增强生成技术，实现无需任务训练即可识别未知缺陷，显著提升检测的可解释性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.22868v1](http://arxiv.org/pdf/2510.22868v1)**

> **作者:** Yang Zhang; Qianyu Zhou; Farhad Imani; Jiong Tang
>
> **摘要:** Wind turbine blades operate in harsh environments, making timely damage detection essential for preventing failures and optimizing maintenance. Drone-based inspection and deep learning are promising, but typically depend on large, labeled datasets, which limit their ability to detect rare or evolving damage types. To address this, we propose a zero-shot-oriented inspection framework that integrates Retrieval-Augmented Generation (RAG) with Vision-Language Models (VLM). A multimodal knowledge base is constructed, comprising technical documentation, representative reference images, and domain-specific guidelines. A hybrid text-image retriever with keyword-aware reranking assembles the most relevant context to condition the VLM at inference, injecting domain knowledge without task-specific training. We evaluate the framework on 30 labeled blade images covering diverse damage categories. Although the dataset is small due to the difficulty of acquiring verified blade imagery, it covers multiple representative defect types. On this test set, the RAG-grounded VLM correctly classified all samples, whereas the same VLM without retrieval performed worse in both accuracy and precision. We further compare against open-vocabulary baselines and incorporate uncertainty Clopper-Pearson confidence intervals to account for the small-sample setting. Ablation studies indicate that the key advantage of the framework lies in explainability and generalizability: retrieved references ground the reasoning process and enable the detection of previously unseen defects by leveraging domain knowledge rather than relying solely on visual cues. This research contributes a data-efficient solution for industrial inspection that reduces dependence on extensive labeled datasets.
>
---
#### [new 071] Through the Lens: Benchmarking Deepfake Detectors Against Moiré-Induced Distortions
- **分类: cs.CV**

- **简介: 该论文聚焦于深伪检测任务，针对真实场景中手机拍摄屏幕产生的莫尔纹干扰问题。研究构建了DMF数据集，系统评估15种先进检测器在莫尔纹影响下的性能，发现其导致准确率下降达25.4%，且去莫尔方法反而加剧问题，呼吁开发更鲁棒的检测模型。**

- **链接: [http://arxiv.org/pdf/2510.23225v1](http://arxiv.org/pdf/2510.23225v1)**

> **作者:** Razaib Tariq; Minji Heo; Simon S. Woo; Shahroz Tariq
>
> **备注:** 48 Pages, 29 Figures, 15 Tables
>
> **摘要:** Deepfake detection remains a pressing challenge, particularly in real-world settings where smartphone-captured media from digital screens often introduces Moir\'e artifacts that can distort detection outcomes. This study systematically evaluates state-of-the-art (SOTA) deepfake detectors on Moir\'e-affected videos, an issue that has received little attention. We collected a dataset of 12,832 videos, spanning 35.64 hours, from the Celeb-DF, DFD, DFDC, UADFV, and FF++ datasets, capturing footage under diverse real-world conditions, including varying screens, smartphones, lighting setups, and camera angles. To further examine the influence of Moir\'e patterns on deepfake detection, we conducted additional experiments using our DeepMoir\'eFake, referred to as (DMF) dataset and two synthetic Moir\'e generation techniques. Across 15 top-performing detectors, our results show that Moir\'e artifacts degrade performance by as much as 25.4%, while synthetically generated Moir\'e patterns lead to a 21.4% drop in accuracy. Surprisingly, demoir\'eing methods, intended as a mitigation approach, instead worsened the problem, reducing accuracy by up to 17.2%. These findings underscore the urgent need for detection models that can robustly handle Moir\'e distortions alongside other realworld challenges, such as compression, sharpening, and blurring. By introducing the DMF dataset, we aim to drive future research toward closing the gap between controlled experiments and practical deepfake detection.
>
---
#### [new 072] Token-Level Inference-Time Alignment for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型（VLM）的幻觉问题，提出TITA框架，通过轻量级奖励模型在推理时实现逐标记对齐，无需微调主模型。利用逻辑概率比提供细粒度反馈，显著降低幻觉，提升多任务表现，且推理开销极低。**

- **链接: [http://arxiv.org/pdf/2510.21794v1](http://arxiv.org/pdf/2510.21794v1)**

> **作者:** Kejia Chen; Jiawen Zhang; Jiacong Hu; Kewei Gao; Jian Lou; Zunlei Feng; Mingli Song
>
> **摘要:** Vision-Language Models (VLMs) have become essential backbones of modern multimodal intelligence, yet their outputs remain prone to hallucination-plausible text misaligned with visual inputs. Existing alignment approaches often rely on expensive fine-tuning with annotated preference data or sequence-level inference strategies that provide only coarse, delayed feedback. To overcome these limitations, we present TITA (Token-level Inference-Time Alignment), a lightweight framework that freezes the base VLM and instead trains a reward model to approximate its distribution. During inference, implicit preference signals are extracted as log-probability ratios between the reward model and the target VLM, yielding dense autoregressive feedback. This formulation can be viewed as an inference-time variant of Direct Preference Optimization (DPO), providing token-level corrective signals without retraining the backbone. Extensive evaluations on LLaVA-1.5-7B and 13B show consistent gains across 12 benchmarks, with improvements of 8.6% on MMVet and 6.7% on POPE, indicating stronger general understanding and reduced hallucinations. Additional experiments on Qwen2.5-VL-7B and DeepSeek-VL2-27.5B show comparable gains, especially in hallucination reduction and VQA accuracy, while incurring negligible inference overhead.
>
---
#### [new 073] Self-Calibrated Consistency can Fight Back for Adversarial Robustness in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型（如CLIP）在零样本场景下对对抗攻击的脆弱性问题，提出自校准一致性（SCC）防御方法。通过语义一致性和空间一致性两个模块，在测试时提升模型鲁棒性，无需额外标注数据，有效增强跨模态对齐稳定性，且可通用至其他VLMs。**

- **链接: [http://arxiv.org/pdf/2510.22785v1](http://arxiv.org/pdf/2510.22785v1)**

> **作者:** Jiaxiang Liu; Jiawei Du; Xiao Liu; Prayag Tiwari; Mingkun Xu
>
> **摘要:** Pre-trained vision-language models (VLMs) such as CLIP have demonstrated strong zero-shot capabilities across diverse domains, yet remain highly vulnerable to adversarial perturbations that disrupt image-text alignment and compromise reliability. Existing defenses typically rely on adversarial fine-tuning with labeled data, limiting their applicability in zero-shot settings. In this work, we identify two key weaknesses of current CLIP adversarial attacks -- lack of semantic guidance and vulnerability to view variations -- collectively termed semantic and viewpoint fragility. To address these challenges, we propose Self-Calibrated Consistency (SCC), an effective test-time defense. SCC consists of two complementary modules: Semantic consistency, which leverages soft pseudo-labels from counterattack warm-up and multi-view predictions to regularize cross-modal alignment and separate the target embedding from confusable negatives; and Spatial consistency, aligning perturbed visual predictions via augmented views to stabilize inference under adversarial perturbations. Together, these modules form a plug-and-play inference strategy. Extensive experiments on 22 benchmarks under diverse attack settings show that SCC consistently improves the zero-shot robustness of CLIP while maintaining accuracy, and can be seamlessly integrated with other VLMs for further gains. These findings highlight the great potential of establishing an adversarially robust paradigm from CLIP, with implications extending to broader vision-language domains such as BioMedCLIP.
>
---
#### [new 074] Multitask Multimodal Self-Supervised Learning for Medical Images
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对医疗图像分析中依赖大量标注数据的问题，提出一种多任务多模态自监督学习方法。通过构建Medformer模型，实现跨模态、跨尺寸医学图像的预训练与领域自适应，设计新颖的预训练任务以挖掘无标签数据特征，显著降低对标注数据的依赖，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.23325v1](http://arxiv.org/pdf/2510.23325v1)**

> **作者:** Cristian Simionescu
>
> **摘要:** This thesis works to address a pivotal challenge in medical image analysis: the reliance on extensive labeled datasets, which are often limited due to the need for expert annotation and constrained by privacy and legal issues. By focusing on the development of self-supervised learning techniques and domain adaptation methods, this research aims to circumvent these limitations, presenting a novel approach to enhance the utility and efficacy of deep learning in medical imaging. Central to this thesis is the development of the Medformer, an innovative neural network architecture designed for multitask learning and deep domain adaptation. This model is adept at pre-training on diverse medical image datasets, handling varying sizes and modalities, and is equipped with a dynamic input-output adaptation mechanism. This enables efficient processing and integration of a wide range of medical image types, from 2D X-rays to complex 3D MRIs, thus mitigating the dependency on large labeled datasets. Further, the thesis explores the current state of self-supervised learning in medical imaging. It introduces novel pretext tasks that are capable of extracting meaningful information from unlabeled data, significantly advancing the model's interpretative abilities. This approach is validated through rigorous experimentation, including the use of the MedMNIST dataset, demonstrating the model's proficiency in learning generalized features applicable to various downstream tasks. In summary, this thesis contributes to the advancement of medical image analysis by offering a scalable, adaptable framework that reduces reliance on labeled data. It paves the way for more accurate, efficient diagnostic tools in healthcare, signifying a major step forward in the application of deep learning in medical imaging.
>
---
#### [new 075] WAON: Large-Scale and High-Quality Japanese Image-Text Pair Dataset for Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出WAON，一个包含约1.55亿条日语图文对的大规模高质量数据集，用于提升视觉语言模型性能。针对日语文化图像识别任务，构建了手动标注的WAON-Bench评估基准。实验表明，基于WAON训练的模型在多项日语文化相关任务上优于现有方法，达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2510.22276v1](http://arxiv.org/pdf/2510.22276v1)**

> **作者:** Issa Sugiura; Shuhei Kurita; Yusuke Oda; Daisuke Kawahara; Yasuo Okabe; Naoaki Okazaki
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Large-scale and high-quality image-text pair datasets play an important role in developing high-performing Vision-Language Models (VLMs). In this work, we introduce WAON, a large-scale and high-quality Japanese image-text pair dataset containing approximately 155 million examples, collected from Common Crawl. Our dataset construction pipeline employs various techniques, including filtering and deduplication, which have been shown to be effective in previous studies. To evaluate its effectiveness, we also construct WAON-Bench, a manually curated benchmark for Japanese cultural image classification, consisting of 374 classes. To assess the effectiveness of our dataset, we conduct experiments using both WAON and the Japanese subset of ReLAION, one of the most widely used vision-language datasets. We fine-tune SigLIP2, a strong multilingual model, on both datasets. The results demonstrate that WAON enhances model performance on WAON-Bench more efficiently than ReLAION and achieves higher accuracy across all evaluated benchmarks. Furthermore, the model fine-tuned on WAON achieves state-of-the-art performance on several Japanese cultural benchmarks. We release our dataset, model, and code at https://speed1313.github.io/WAON.
>
---
#### [new 076] 2D_3D Feature Fusion via Cross-Modal Latent Synthesis and Attention Guided Restoration for Industrial Anomaly Detection
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文针对工业异常检测任务，解决2D与3D数据融合困难问题。提出MAFR框架，通过共享编码器融合RGB图像与点云，利用注意力引导的解码器恢复特征，基于重建误差定位异常。在多个基准上达SOTA性能，验证了方法的有效性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.21793v1](http://arxiv.org/pdf/2510.21793v1)**

> **作者:** Usman Ali; Ali Zia; Abdul Rehman; Umer Ramzan; Zohaib Hassan; Talha Sattar; Jing Wang; Wei Xiang
>
> **备注:** Accepted at 26th International Conference on Digital Image Computing: Techniques and Applications (DICTA 2025)
>
> **摘要:** Industrial anomaly detection (IAD) increasingly benefits from integrating 2D and 3D data, but robust cross-modal fusion remains challenging. We propose a novel unsupervised framework, Multi-Modal Attention-Driven Fusion Restoration (MAFR), which synthesises a unified latent space from RGB images and point clouds using a shared fusion encoder, followed by attention-guided, modality-specific decoders. Anomalies are localised by measuring reconstruction errors between input features and their restored counterparts. Evaluations on the MVTec 3D-AD and Eyecandies benchmarks demonstrate that MAFR achieves state-of-the-art results, with a mean I-AUROC of 0.972 and 0.901, respectively. The framework also exhibits strong performance in few-shot learning settings, and ablation studies confirm the critical roles of the fusion architecture and composite loss. MAFR offers a principled approach for fusing visual and geometric information, advancing the robustness and accuracy of industrial anomaly detection. Code is available at https://github.com/adabrh/MAFR
>
---
#### [new 077] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像扩散模型中的概念擦除任务，提出无需训练的零样本框架Semantic Surgery。通过动态检测并校准文本嵌入中的目标概念，实现精准、无损的语义擦除，有效消除有害内容，同时保持图像质量与局部细节，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.22851v1](http://arxiv.org/pdf/2510.22851v1)**

> **作者:** Lexiang Xiong; Chengyu Liu; Jingwen Ye; Yan Liu; Yuecong Xu
>
> **备注:** Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Code is available at https://github.com/Lexiang-Xiong/Semantic-Surgery
>
> **摘要:** Concept erasure in text-to-image diffusion models is crucial for mitigating harmful content, yet existing methods often compromise generative quality. We introduce Semantic Surgery, a novel training-free, zero-shot framework for concept erasure that operates directly on text embeddings before the diffusion process. It dynamically estimates the presence of target concepts in a prompt and performs a calibrated vector subtraction to neutralize their influence at the source, enhancing both erasure completeness and locality. The framework includes a Co-Occurrence Encoding module for robust multi-concept erasure and a visual feedback loop to address latent concept persistence. As a training-free method, Semantic Surgery adapts dynamically to each prompt, ensuring precise interventions. Extensive experiments on object, explicit content, artistic style, and multi-celebrity erasure tasks show our method significantly outperforms state-of-the-art approaches. We achieve superior completeness and robustness while preserving locality and image quality (e.g., 93.58 H-score in object erasure, reducing explicit content to just 1 instance, and 8.09 H_a in style erasure with no quality degradation). This robustness also allows our framework to function as a built-in threat detection system, offering a practical solution for safer text-to-image generation.
>
---
#### [new 078] MedXplain-VQA: Multi-Component Explainable Medical Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文提出MedXplain-VQA，面向医疗视觉问答任务，解决AI诊断缺乏可解释性问题。通过五组件融合：医学查询重写、增强Grad-CAM、区域精提取、链式推理等，提升结果透明度与临床相关性。实验显示其在500张病理图像上显著优于基线，生成结构化、术语准确的解释，增强医生信任。**

- **链接: [http://arxiv.org/pdf/2510.22803v1](http://arxiv.org/pdf/2510.22803v1)**

> **作者:** Hai-Dang Nguyen; Minh-Anh Dang; Minh-Tan Le; Minh-Tuan Le
>
> **备注:** 10 pages, 4 figures, IEEE conference format
>
> **摘要:** Explainability is critical for the clinical adoption of medical visual question answering (VQA) systems, as physicians require transparent reasoning to trust AI-generated diagnoses. We present MedXplain-VQA, a comprehensive framework integrating five explainable AI components to deliver interpretable medical image analysis. The framework leverages a fine-tuned BLIP-2 backbone, medical query reformulation, enhanced Grad-CAM attention, precise region extraction, and structured chain-of-thought reasoning via multi-modal language models. To evaluate the system, we introduce a medical-domain-specific framework replacing traditional NLP metrics with clinically relevant assessments, including terminology coverage, clinical structure quality, and attention region relevance. Experiments on 500 PathVQA histopathology samples demonstrate substantial improvements, with the enhanced system achieving a composite score of 0.683 compared to 0.378 for baseline methods, while maintaining high reasoning confidence (0.890). Our system identifies 3-5 diagnostically relevant regions per sample and generates structured explanations averaging 57 words with appropriate clinical terminology. Ablation studies reveal that query reformulation provides the most significant initial improvement, while chain-of-thought reasoning enables systematic diagnostic processes. These findings underscore the potential of MedXplain-VQA as a robust, explainable medical VQA system. Future work will focus on validation with medical experts and large-scale clinical datasets to ensure clinical readiness.
>
---
#### [new 079] SRSR: Enhancing Semantic Accuracy in Real-World Image Super-Resolution with Spatially Re-Focused Text-Conditioning
- **分类: cs.CV**

- **简介: 该论文针对真实世界图像超分辨率任务中因文本条件不准确导致的语义模糊与幻觉问题，提出SRSR框架。通过空间聚焦交叉注意力（SRCA）和空间目标分类器自由引导（STCFG），精准控制文本对图像的影响，提升生成结果的语义准确性和视觉质量。**

- **链接: [http://arxiv.org/pdf/2510.22534v1](http://arxiv.org/pdf/2510.22534v1)**

> **作者:** Chen Chen; Majid Abdolshah; Violetta Shevchenko; Hongdong Li; Chang Xu; Pulak Purkait
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Existing diffusion-based super-resolution approaches often exhibit semantic ambiguities due to inaccuracies and incompleteness in their text conditioning, coupled with the inherent tendency for cross-attention to divert towards irrelevant pixels. These limitations can lead to semantic misalignment and hallucinated details in the generated high-resolution outputs. To address these, we propose a novel, plug-and-play spatially re-focused super-resolution (SRSR) framework that consists of two core components: first, we introduce Spatially Re-focused Cross-Attention (SRCA), which refines text conditioning at inference time by applying visually-grounded segmentation masks to guide cross-attention. Second, we introduce a Spatially Targeted Classifier-Free Guidance (STCFG) mechanism that selectively bypasses text influences on ungrounded pixels to prevent hallucinations. Extensive experiments on both synthetic and real-world datasets demonstrate that SRSR consistently outperforms seven state-of-the-art baselines in standard fidelity metrics (PSNR and SSIM) across all datasets, and in perceptual quality measures (LPIPS and DISTS) on two real-world benchmarks, underscoring its effectiveness in achieving both high semantic fidelity and perceptual quality in super-resolution.
>
---
#### [new 080] Top-Down Semantic Refinement for Image Captioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对图像描述任务中大视觉语言模型因单步生成导致叙事不连贯、细节缺失的问题，提出基于自顶向下语义精炼的框架TDSR。通过将生成建模为马尔可夫决策过程，结合高效蒙特卡洛树搜索与轻量价值网络，显著降低对昂贵模型的调用次数，提升描述质量与复杂场景适应性。**

- **链接: [http://arxiv.org/pdf/2510.22391v1](http://arxiv.org/pdf/2510.22391v1)**

> **作者:** Jusheng Zhang; Kaitong Cai; Jing Yang; Jian Wang; Chengpei Tang; Keze Wang
>
> **摘要:** Large Vision-Language Models (VLMs) face an inherent contradiction in image captioning: their powerful single-step generation capabilities often lead to a myopic decision-making process. This makes it difficult to maintain global narrative coherence while capturing rich details, a limitation that is particularly pronounced in tasks that require multi-step and complex scene description. To overcome this fundamental challenge, we redefine image captioning as a goal-oriented hierarchical refinement planning problem, and further propose a novel framework, named Top-Down Semantic Refinement (TDSR), which models the generation process as a Markov Decision Process (MDP). However, planning within the vast state space of a VLM presents a significant computational hurdle. Our core contribution, therefore, is the design of a highly efficient Monte Carlo Tree Search (MCTS) algorithm tailored for VLMs. By incorporating a visual-guided parallel expansion and a lightweight value network, our TDSR reduces the call frequency to the expensive VLM by an order of magnitude without sacrificing planning quality. Furthermore, an adaptive early stopping mechanism dynamically matches computational overhead to the image's complexity. Extensive experiments on multiple benchmarks, including DetailCaps, COMPOSITIONCAP, and POPE, demonstrate that our TDSR, as a plug-and-play module, can significantly enhance the performance of existing VLMs (e.g., LLaVA-1.5, Qwen2.5-VL) by achieving state-of-the-art or highly competitive results in fine-grained description, compositional generalization, and hallucination suppression.
>
---
#### [new 081] Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型（VLM）的幻觉问题，提出GIFT方法。通过追踪视觉注意力的“凝视转移”生成显著性图，增强关键视觉信息与用户查询的关注度，缓解注意力错位与跨模态融合失衡，有效减少幻觉，提升生成与分类任务性能，计算开销低。**

- **链接: [http://arxiv.org/pdf/2510.22067v1](http://arxiv.org/pdf/2510.22067v1)**

> **作者:** Zheng Qi; Chao Shang; Evangelia Spiliopoulou; Nikolaos Pappas
>
> **摘要:** Vision language models (VLMs) often generate hallucination, i.e., content that cannot be substantiated by either textual or visual inputs. Prior work primarily attributes this to over-reliance on linguistic prior knowledge rather than visual inputs. Some methods attempt to mitigate hallucination by amplifying visual token attention proportionally to their attention scores. However, these methods overlook the visual attention sink problem, where attention is frequently misallocated to task-irrelevant visual regions, and neglect cross-modal fusion balance by enhancing only visual attention without adjusting attention to the user query. This can result in amplifying incorrect areas while failing to properly interpret the user query. To address these challenges, we propose a simple yet effective method called Gaze Shift-Guided Cross-modal Fusion Enhancement (GIFT). GIFT pre-computes a holistic visual saliency map by tracking positive changes in visual attention, or "gaze shifts", during user query comprehension, and leverages this map to amplify attention to both salient visual information and the user query at each decoding step. This reduces the impact of visual attention sink, as irrelevant tokens exhibit minimal shifts, while ensuring balanced cross-modal fusion for well-integrated representation. Extensive experiments show that GIFT effectively mitigates hallucination in VLMs across both generative and classification tasks, achieving up to 20.7% improvement over greedy decoding, while maintaining general vision-language performance with low computational overhead.
>
---
#### [new 082] EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT
- **分类: cs.CV**

- **简介: 该论文聚焦于第一人称视频推理任务，针对多模态大模型缺乏具身理解的问题，提出EgoThinker框架。通过构建大规模标注数据集EgoRe-5M，并采用分阶段训练策略，提升模型对隐含意图与细粒度时空交互的推理能力，显著改善了第一人称场景下的时空定位性能。**

- **链接: [http://arxiv.org/pdf/2510.23569v1](http://arxiv.org/pdf/2510.23569v1)**

> **作者:** Baoqi Pei; Yifei Huang; Jilan Xu; Yuping He; Guo Chen; Fei Wu; Yu Qiao; Jiangmiao Pang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Egocentric video reasoning centers on an unobservable agent behind the camera who dynamically shapes the environment, requiring inference of hidden intentions and recognition of fine-grained interactions. This core challenge limits current multimodal large language models MLLMs, which excel at visible event reasoning but lack embodied, first-person understanding. To bridge this gap, we introduce EgoThinker, a novel framework that endows MLLMs with robust egocentric reasoning capabilities through spatio-temporal chain-of-thought supervision and a two-stage learning curriculum. First, we introduce EgoRe-5M, a large-scale egocentric QA dataset constructed from 13M diverse egocentric video clips. This dataset features multi-minute segments annotated with detailed CoT rationales and dense hand-object grounding. Second, we employ SFT on EgoRe-5M to instill reasoning skills, followed by reinforcement fine-tuning RFT to further enhance spatio-temporal localization. Experimental results show that EgoThinker outperforms existing methods across multiple egocentric benchmarks, while achieving substantial improvements in fine-grained spatio-temporal localization tasks. Full code and data are released at https://github.com/InternRobotics/EgoThinker.
>
---
#### [new 083] Face-MakeUpV2: Facial Consistency Learning for Controllable Text-to-Image Generation
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文针对文本生成人脸时的面部属性泄露与物理一致性不足问题，提出Face-MakeUpV2模型。通过构建大规模图文掩码数据集，引入3D渲染与全局特征注入通道，并设计嵌入对齐与感知损失优化目标，有效保持参考人脸的身份与物理特征一致性，实现可控、可靠的面部图像生成。**

- **链接: [http://arxiv.org/pdf/2510.21775v1](http://arxiv.org/pdf/2510.21775v1)**

> **作者:** Dawei Dai; Yinxiu Zhou; Chenghang Li; Guolai Jiang; Chengfang Zhang
>
> **摘要:** In facial image generation, current text-to-image models often suffer from facial attribute leakage and insufficient physical consistency when responding to local semantic instructions. In this study, we propose Face-MakeUpV2, a facial image generation model that aims to maintain the consistency of face ID and physical characteristics with the reference image. First, we constructed a large-scale dataset FaceCaptionMask-1M comprising approximately one million image-text-masks pairs that provide precise spatial supervision for the local semantic instructions. Second, we employed a general text-to-image pretrained model as the backbone and introduced two complementary facial information injection channels: a 3D facial rendering channel to incorporate the physical characteristics of the image and a global facial feature channel. Third, we formulated two optimization objectives for the supervised learning of our model: semantic alignment in the model's embedding space to mitigate the attribute leakage problem and perceptual loss on facial images to preserve ID consistency. Extensive experiments demonstrated that our Face-MakeUpV2 achieves best overall performance in terms of preserving face ID and maintaining physical consistency of the reference images. These results highlight the practical potential of Face-MakeUpV2 for reliable and controllable facial editing in diverse applications.
>
---
#### [new 084] SemiETPicker: Fast and Label-Efficient Particle Picking for CryoET Tomography Using Semi-Supervised Learning
- **分类: cs.CV**

- **简介: 该论文针对冷冻电镜断层成像（CryoET）中粒子定位的瓶颈问题，提出SemiETPicker框架。通过半监督学习，结合热图监督检测与师生协同训练，利用大量未标注数据提升效率。引入多视角伪标签与专用增强策略，在少量标注下显著提升F1分数，实现快速、低标注依赖的粒子自动识别。**

- **链接: [http://arxiv.org/pdf/2510.22454v1](http://arxiv.org/pdf/2510.22454v1)**

> **作者:** Linhan Wang; Jianwen Dou; Wang Li; Shengkun Wang; Zhiwu Xie; Chang-Tien Lu; Yinlin Chen
>
> **摘要:** Cryogenic Electron Tomography (CryoET) combined with sub-volume averaging (SVA) is the only imaging modality capable of resolving protein structures inside cells at molecular resolution. Particle picking, the task of localizing and classifying target proteins in 3D CryoET volumes, remains the main bottleneck. Due to the reliance on time-consuming manual labels, the vast reserve of unlabeled tomograms remains underutilized. In this work, we present a fast, label-efficient semi-supervised framework that exploits this untapped data. Our framework consists of two components: (i) an end-to-end heatmap-supervised detection model inspired by keypoint detection, and (ii) a teacher-student co-training mechanism that enhances performance under sparse labeling conditions. Furthermore, we introduce multi-view pseudo-labeling and a CryoET-specific DropBlock augmentation strategy to further boost performance. Extensive evaluations on the large-scale CZII dataset show that our approach improves F1 by 10% over supervised baselines, underscoring the promise of semi-supervised learning for leveraging unlabeled CryoET data.
>
---
#### [new 085] Real-Time Semantic Segmentation on FPGA for Autonomous Vehicles Using LMIINet with the CGRA4ML Framework
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶中的实时语义分割任务，提出基于FPGA的轻量级LMIINet模型实现。通过QAT量化与CGRA4ML框架优化，简化网络结构并适配硬件，实现在ZCU104上20FPS、50.1ms延迟下90%像素准确率与45% mIoU，显著提升能效比，优于传统GPU方案。**

- **链接: [http://arxiv.org/pdf/2510.22243v1](http://arxiv.org/pdf/2510.22243v1)**

> **作者:** Amir Mohammad Khadem Hosseini; Sattar Mirzakuchaki
>
> **摘要:** Semantic segmentation has emerged as a fundamental problem in computer vision, gaining particular importance in real-time applications such as autonomous driving. The main challenge is achieving high accuracy while operating under computational and hardware constraints. In this research, we present an FPGA-based implementation of real-time semantic segmentation leveraging the lightweight LMIINet architecture and the Coarse-Grained Reconfigurable Array for Machine Learning (CGRA4ML) hardware framework. The model was trained using Quantization-Aware Training (QAT) with 8-bit precision on the Cityscapes dataset, reducing memory footprint by a factor of four while enabling efficient fixed-point computations. Necessary modifications were applied to adapt the model to CGRA4ML constraints, including simplifying skip connections, employing hardware-friendly operations such as depthwise-separable and 1A-1 convolutions, and redesigning parts of the Flatten Transformer. Our implementation achieves approximately 90% pixel accuracy and 45% mean Intersection-over-Union (mIoU), operating in real-time at 20 frames per second (FPS) with 50.1 ms latency on the ZCU104 FPGA board. The results demonstrate the potential of CGRA4ML, with its flexibility in mapping modern layers and off-chip memory utilization for skip connections, provides a path for implementing advanced semantic segmentation networks on FPGA for real-time applications to outperform traditional GPU solutions in terms of power efficiency while maintaining competitive accuracy. The code for this project is publicly available at https://github.com/STAmirr/ cgra4ml_semantic_segmentation
>
---
#### [new 086] Bridging Accuracy and Interpretability: Deep Learning with XAI for Breast Cancer Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对乳腺癌早期检测任务，提出一种可解释的深度学习框架。通过分析细针穿刺图像特征，模型实现高精度分类（准确率0.992），并结合SHAP/LIME技术提升可解释性，揭示“细胞核凹点”为关键影响特征，兼顾性能与临床可信度。**

- **链接: [http://arxiv.org/pdf/2510.21780v1](http://arxiv.org/pdf/2510.21780v1)**

> **作者:** Bishal Chhetri; B. V. Rathish Kumar
>
> **备注:** 15 pages, 14 figures
>
> **摘要:** In this study, we present an interpretable deep learning framework for the early detection of breast cancer using quantitative features extracted from digitized fine needle aspirate (FNA) images of breast masses. Our deep neural network, using ReLU activations, the Adam optimizer, and a binary cross-entropy loss, delivers state-of-the-art classification performance, achieving an accuracy of 0.992, precision of 1.000, recall of 0.977, and an F1 score of 0.988. These results substantially exceed the benchmarks reported in the literature. We evaluated the model under identical protocols against a suite of well-established algorithms (logistic regression, decision trees, random forests, stochastic gradient descent, K-nearest neighbors, and XGBoost) and found the deep model consistently superior on the same metrics. Recognizing that high predictive accuracy alone is insufficient for clinical adoption due to the black-box nature of deep learning models, we incorporated model-agnostic Explainable AI techniques such as SHAP and LIME to produce feature-level attributions and human-readable visualizations. These explanations quantify the contribution of each feature to individual predictions, support error analysis, and increase clinician trust, thus bridging the gap between performance and interpretability for real-world clinical use. The concave points feature of the cell nuclei is found to be the most influential feature positively impacting the classification task. This insight can be very helpful in improving the diagnosis and treatment of breast cancer by highlighting the key characteristics of breast tumor.
>
---
#### [new 087] TernaryCLIP: Efficiently Compressing Vision-Language Models with Ternary Weights and Distilled Knowledge
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉-语言模型高效部署问题，提出TernaryCLIP，通过将CLIP模型权重二值化为三值（-1, 0, +1），结合量化感知训练与知识蒸馏，实现99%权重压缩、16.98倍压缩比与2.3倍推理加速，显著降低存储与内存开销，同时保持零样本图像分类与图文检索性能。**

- **链接: [http://arxiv.org/pdf/2510.21879v1](http://arxiv.org/pdf/2510.21879v1)**

> **作者:** Shu-Hao Zhang; Wei-Cheng Tang; Chen Wu; Peng Hu; Nan Li; Liang-Jie Zhang; Qi Zhang; Shao-Qun Zhang
>
> **摘要:** Recent years have witnessed an increasing interest in image-text contrastive modeling, exemplified by models such as Contrastive Language-Image Pretraining (CLIP). In this paper, we propose the TernaryCLIP, a lightweight computational framework that converts connection weights of both vision and text encoders of CLIP into the ternary format, instead of full-precision or floating ones. TernaryCLIP incorporates quantization-aware training and distillation modules, preventing precision degradation and enabling low-cost and high-efficiency computations. Comprehensive experiments demonstrate that TernaryCLIP can achieve up to 99\% ternarized weights with 1.58-bit representation, 16.98 $\times$ compression ratio, 2.3 $\times$ inference acceleration, 16 $\times$ storage reduction, 10 $\times$ memory optimization, and 60\% sparsity while maintaining promising performance on zero-shot image classification and image-text retrieval tasks across 41 commonly used datasets. Our work highlights the feasibility of extreme quantization for large multimodal models, supporting effective and efficient deployment on resource-constrained devices. The model and code can be accessed from Hugging Face and GitHub.
>
---
#### [new 088] DynamicTree: Interactive Real Tree Animation via Sparse Voxel Spectrum
- **分类: cs.CV**

- **简介: 该论文提出DynamicTree框架，解决复杂真实树木的长期、交互式4D动画生成难题。通过稀疏体素谱表示树体运动，实现快速前向动态生成与实时交互响应，并构建了首个大规模合成4D树数据集4DTree，显著提升动画真实感与效率。**

- **链接: [http://arxiv.org/pdf/2510.22213v1](http://arxiv.org/pdf/2510.22213v1)**

> **作者:** Yaokun Li; Lihe Ding; Xiao Chen; Guang Tan; Tianfan Xue
>
> **备注:** Project Page: https://dynamictree-dev.github.io/DynamicTree.github.io/
>
> **摘要:** Generating dynamic and interactive 3D objects, such as trees, has wide applications in virtual reality, games, and world simulation. Nevertheless, existing methods still face various challenges in generating realistic 4D motion for complex real trees. In this paper, we propose DynamicTree, the first framework that can generate long-term, interactive animation of 3D Gaussian Splatting trees. Unlike prior optimization-based methods, our approach generates dynamics in a fast feed-forward manner. The key success of our approach is the use of a compact sparse voxel spectrum to represent the tree movement. Given a 3D tree from Gaussian Splatting reconstruction, our pipeline first generates mesh motion using the sparse voxel spectrum and then binds Gaussians to deform the mesh. Additionally, the proposed sparse voxel spectrum can also serve as a basis for fast modal analysis under external forces, allowing real-time interactive responses. To train our model, we also introduce 4DTree, the first large-scale synthetic 4D tree dataset containing 8,786 animated tree meshes with semantic labels and 100-frame motion sequences. Extensive experiments demonstrate that our method achieves realistic and responsive tree animations, significantly outperforming existing approaches in both visual quality and computational efficiency.
>
---
#### [new 089] I2-NeRF: Learning Neural Radiance Fields Under Physically-Grounded Media Interactions
- **分类: cs.CV**

- **简介: 该论文提出I2-NeRF，一种基于物理介质交互的神经辐射场模型。针对现有方法在复杂介质中失真问题，通过逆分层采样实现均匀空间采样，并建立统一的辐射衰减模型，提升3D重建的几何保真度与物理合理性，支持水下、雾霾等场景的精确感知与介质参数估计。**

- **链接: [http://arxiv.org/pdf/2510.22161v1](http://arxiv.org/pdf/2510.22161v1)**

> **作者:** Shuhong Liu; Lin Gu; Ziteng Cui; Xuangeng Chu; Tatsuya Harada
>
> **摘要:** Participating in efforts to endow generative AI with the 3D physical world perception, we propose I2-NeRF, a novel neural radiance field framework that enhances isometric and isotropic metric perception under media degradation. While existing NeRF models predominantly rely on object-centric sampling, I2-NeRF introduces a reverse-stratified upsampling strategy to achieve near-uniform sampling across 3D space, thereby preserving isometry. We further present a general radiative formulation for media degradation that unifies emission, absorption, and scattering into a particle model governed by the Beer-Lambert attenuation law. By composing the direct and media-induced in-scatter radiance, this formulation extends naturally to complex media environments such as underwater, haze, and even low-light scenes. By treating light propagation uniformly in both vertical and horizontal directions, I2-NeRF enables isotropic metric perception and can even estimate medium properties such as water depth. Experiments on real-world datasets demonstrate that our method significantly improves both reconstruction fidelity and physical plausibility compared to existing approaches.
>
---
#### [new 090] Nested AutoRegressive Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Nested AutoRegressive（NestAR）模型，用于图像生成任务。针对传统AR模型计算复杂度高、样本多样性不足的问题，NestAR采用分层嵌套的AR架构，通过多尺度模块逐级生成图像，将复杂度从O(n)降至O(log n)，并引入流匹配损失提升多样性，实现高效高质量图像生成。**

- **链接: [http://arxiv.org/pdf/2510.23028v1](http://arxiv.org/pdf/2510.23028v1)**

> **作者:** Hongyu Wu; Xuhui Fan; Zhangkai Wu; Longbing Cao
>
> **摘要:** AutoRegressive (AR) models have demonstrated competitive performance in image generation, achieving results comparable to those of diffusion models. However, their token-by-token image generation mechanism remains computationally intensive and existing solutions such as VAR often lead to limited sample diversity. In this work, we propose a Nested AutoRegressive~(NestAR) model, which proposes nested AutoRegressive architectures in generating images. NestAR designs multi-scale modules in a hierarchical order. These different scaled modules are constructed in an AR architecture, where one larger-scale module is conditioned on outputs from its previous smaller-scale module. Within each module, NestAR uses another AR structure to generate ``patches'' of tokens. The proposed nested AR architecture reduces the overall complexity from $\mathcal{O}(n)$ to $\mathcal{O}(\log n)$ in generating $n$ image tokens, as well as increases image diversities. NestAR further incorporates flow matching loss to use continuous tokens, and develops objectives to coordinate these multi-scale modules in model training. NestAR achieves competitive image generation performance while significantly lowering computational cost.
>
---
#### [new 091] Scaling Non-Parametric Sampling with Representation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种无参数生成模型，基于自然图像的三重特性（空间非平稳性、低层规律性、高层语义），通过局部上下文定义像素分布。无需训练即可生成高质量图像，揭示了“部件-整体”泛化的简单机制，为理解生成模型提供了可解释的最小理论框架。**

- **链接: [http://arxiv.org/pdf/2510.22196v1](http://arxiv.org/pdf/2510.22196v1)**

> **作者:** Vincent Lu; Aaron Truong; Zeyu Yun; Yubei Chen
>
> **摘要:** Scaling and architectural advances have produced strikingly photorealistic image generative models, yet their mechanisms still remain opaque. Rather than advancing scaling, our goal is to strip away complicated engineering tricks and propose a simple, non-parametric generative model. Our design is grounded in three principles of natural images-(i) spatial non-stationarity, (ii) low-level regularities, and (iii) high-level semantics-and defines each pixel's distribution from its local context window. Despite its minimal architecture and no training, the model produces high-fidelity samples on MNIST and visually compelling CIFAR-10 images. This combination of simplicity and strong empirical performance points toward a minimal theory of natural-image structure. The model's white-box nature also allows us to have a mechanistic understanding of how the model generalizes and generates diverse images. We study it by tracing each generated pixel back to its source images. These analyses reveal a simple, compositional procedure for "part-whole generalization", suggesting a hypothesis for how large neural network generative models learn to generalize.
>
---
#### [new 092] DynaPose4D: High-Quality 4D Dynamic Content Generation via Pose Alignment Loss
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DynaPose4D，旨在从单张静态图像生成高质量4D动态内容。针对传统方法在时间依赖建模和视角变化下动态几何捕捉不足的问题，融合4DGS与无类别姿态估计，通过姿态对齐损失提升运动一致性。实验表明其生成结果具有高连贯性与流畅性。**

- **链接: [http://arxiv.org/pdf/2510.22473v1](http://arxiv.org/pdf/2510.22473v1)**

> **作者:** Jing Yang; Yufeng Yang
>
> **摘要:** Recent advancements in 2D and 3D generative models have expanded the capabilities of computer vision. However, generating high-quality 4D dynamic content from a single static image remains a significant challenge. Traditional methods have limitations in modeling temporal dependencies and accurately capturing dynamic geometry changes, especially when considering variations in camera perspective. To address this issue, we propose DynaPose4D, an innovative solution that integrates 4D Gaussian Splatting (4DGS) techniques with Category-Agnostic Pose Estimation (CAPE) technology. This framework uses 3D Gaussian Splatting to construct a 3D model from single images, then predicts multi-view pose keypoints based on one-shot support from a chosen view, leveraging supervisory signals to enhance motion consistency. Experimental results show that DynaPose4D achieves excellent coherence, consistency, and fluidity in dynamic motion generation. These findings not only validate the efficacy of the DynaPose4D framework but also indicate its potential applications in the domains of computer vision and animation production.
>
---
#### [new 093] Cross-View UAV Geo-Localization with Precision-Focused Efficient Design: A Hierarchical Distillation Approach with Multi-view Refinement
- **分类: cs.CV**

- **简介: 该论文针对无人机在无GNSS环境下的跨视角地理定位任务，提出高效精准的PFED框架。通过分层知识蒸馏与多视角优化，在保持97.15%高召回率的同时，显著降低计算开销，实现251.5 FPS的实时推理，适用于边缘设备部署。**

- **链接: [http://arxiv.org/pdf/2510.22582v1](http://arxiv.org/pdf/2510.22582v1)**

> **作者:** Jian Sun; Kangdao Liu; Chi Zhang; Chuangquan Chen; Junge Shen; Chi-Man Vong
>
> **摘要:** Cross-view geo-localization (CVGL) enables UAV localization by matching aerial images to geo-tagged satellite databases, which is critical for autonomous navigation in GNSS-denied environments. However, existing methods rely on resource-intensive fine-grained feature extraction and alignment, where multiple branches and modules significantly increase inference costs, limiting their deployment on edge devices. We propose Precision-Focused Efficient Design (PFED), a resource-efficient framework combining hierarchical knowledge transfer and multi-view representation refinement. This innovative method comprises two key components: 1) During training, Hierarchical Distillation paradigm for fast and accurate CVGL (HD-CVGL), coupled with Uncertainty-Aware Prediction Alignment (UAPA) to distill essential information and mitigate the data imbalance without incurring additional inference overhead. 2) During inference, an efficient Multi-view Refinement Module (MRM) leverages mutual information to filter redundant samples and effectively utilize the multi-view data. Extensive experiments show that PFED achieves state-of-the-art performance in both accuracy and efficiency, reaching 97.15\% Recall@1 on University-1652 while being over $5 \times$ more efficient in FLOPs and $3 \times$ faster than previous top methods. Furthermore, PFED runs at 251.5 FPS on the AGX Orin edge device, demonstrating its practical viability for real-time UAV applications. The project is available at https://github.com/SkyEyeLoc/PFED
>
---
#### [new 094] Revisiting Multimodal Positional Encoding in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文聚焦视觉语言模型中的多模态位置编码问题，针对现有方法缺乏系统研究的不足，分析了旋转位置编码（RoPE）的定位设计与频率分配。提出MHRoPE与MRoPE-I两种即插即用方案，提升布局清晰性、表征丰富性与文本先验保留，显著增强多模态理解能力。**

- **链接: [http://arxiv.org/pdf/2510.23095v1](http://arxiv.org/pdf/2510.23095v1)**

> **作者:** Jie Huang; Xuejing Liu; Sibo Song; Ruibing Hou; Hong Chang; Junyang Lin; Shuai Bai
>
> **备注:** 16 pages
>
> **摘要:** Multimodal position encoding is essential for vision-language models, yet there has been little systematic investigation into multimodal position encoding. We conduct a comprehensive analysis of multimodal Rotary Positional Embedding (RoPE) by examining its two core components: position design and frequency allocation. Through extensive experiments, we identify three key guidelines: positional coherence, full frequency utilization, and preservation of textual priors-ensuring unambiguous layout, rich representation, and faithful transfer from the pre-trained LLM. Based on these insights, we propose Multi-Head RoPE (MHRoPE) and MRoPE-Interleave (MRoPE-I), two simple and plug-and-play variants that require no architectural changes. Our methods consistently outperform existing approaches across diverse benchmarks, with significant improvements in both general and fine-grained multimodal understanding. Code will be avaliable at https://github.com/JJJYmmm/Multimodal-RoPEs.
>
---
#### [new 095] FreeFuse: Multi-Subject LoRA Fusion via Auto Masking at Test Time
- **分类: cs.CV**

- **简介: 该论文提出FreeFuse，用于多主体文本到图像生成。针对现有方法需训练、修改LoRA或依赖分割模型的问题，提出基于交叉注意力权重自动生成动态掩码，在推理时无须额外训练或修改，仅需提供激活词即可融合多个主体LoRA，实现高效高质量生成。**

- **链接: [http://arxiv.org/pdf/2510.23515v1](http://arxiv.org/pdf/2510.23515v1)**

> **作者:** Yaoli Liu; Yao-Xiang Ding; Kun Zhou
>
> **摘要:** This paper proposes FreeFuse, a novel training-free approach for multi-subject text-to-image generation through automatic fusion of multiple subject LoRAs. In contrast to existing methods that either focus on pre-inference LoRA weight merging or rely on segmentation models and complex techniques like noise blending to isolate LoRA outputs, our key insight is that context-aware dynamic subject masks can be automatically derived from cross-attention layer weights. Mathematical analysis shows that directly applying these masks to LoRA outputs during inference well approximates the case where the subject LoRA is integrated into the diffusion model and used individually for the masked region. FreeFuse demonstrates superior practicality and efficiency as it requires no additional training, no modification to LoRAs, no auxiliary models, and no user-defined prompt templates or region specifications. Alternatively, it only requires users to provide the LoRA activation words for seamless integration into standard workflows. Extensive experiments validate that FreeFuse outperforms existing approaches in both generation quality and usability under the multi-subject generation tasks. The project page is at https://future-item.github.io/FreeFuse/
>
---
#### [new 096] Audio Frequency-Time Dual Domain Evaluation on Depression Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于抑郁症智能诊断任务，旨在解决传统诊断流程复杂、标准模糊、就诊率低等问题。研究利用语音信号的频时域双重特征，结合深度学习模型，提出一种新型抑郁分类算法，显著提升了诊断准确率，为抑郁筛查与评估提供了新方法。**

- **链接: [http://arxiv.org/pdf/2510.22225v1](http://arxiv.org/pdf/2510.22225v1)**

> **作者:** Yu Luo; Nan Huang; Sophie Yu; Hendry Xu; Jerry Wang; Colin Wang; Zhichao Liu; Chen Zeng
>
> **摘要:** Depression, as a typical mental disorder, has become a prevalent issue significantly impacting public health. However, the prevention and treatment of depression still face multiple challenges, including complex diagnostic procedures, ambiguous criteria, and low consultation rates, which severely hinder timely assessment and intervention. To address these issues, this study adopts voice as a physiological signal and leverages its frequency-time dual domain multimodal characteristics along with deep learning models to develop an intelligent assessment and diagnostic algorithm for depression. Experimental results demonstrate that the proposed method achieves excellent performance in the classification task for depression diagnosis, offering new insights and approaches for the assessment, screening, and diagnosis of depression.
>
---
#### [new 097] AI-Boosted Video Annotation: Assessing the Process Enhancement
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 该论文研究视频标注中人机协同的AI增强方法，旨在提升标注效率与质量。通过引入零样本预标注，利用Label Studio框架对UCF-Crime数据集进行异常行为识别，显著减少35%标注时间，且保持高一致性与准确性，验证了AI辅助标注的有效性。**

- **链接: [http://arxiv.org/pdf/2510.21798v1](http://arxiv.org/pdf/2510.21798v1)**

> **作者:** Juan Gutiérrez; Ángel Mora; Pablo Regodón; Silvia Rodriguez; José Luis Blanco
>
> **摘要:** We explore the enhancement of Human-in-the-Loop video annotation by integrating automatic capabilities to ease the task for annotators and assess their performance. The research delves into the practical implications of the annotation processes, the integration of AI components, and the evaluation of its outcomes. We analyze their impact on efficiency, accuracy, and overall annotation quality. Focusing on the Human-in-the-Loop for video annotation tasks, we implemented a single-iteration scheme using Label Studio and AI-powered zero-shot pre-annotations. Using this framework, we designed a test based on the annotation of the UCF-Crime dataset to discriminate between normal and abnormal activities in video footage. Our results evidence how automatic AI-based pre-annotation can streamline the video annotation workflow, empowering human annotators and optimizing the overall pipeline. Using the pre-annotated data, we observed a 35% reduction in the annotation time for 70% of the annotators with similar quality annotations, compared to the traditional manual annotation task. Results are consistent with asset duration and complexity. We also observed that while annotators rapidly learned to use the tool, the produced annotations are more coherent among annotators and better match the natural clustering of the video frames.
>
---
#### [new 098] Yesnt: Are Diffusion Relighting Models Ready for Capture Stage Compositing? A Hybrid Alternative to Bridge the Gap
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对体积视频重光照任务，解决现有扩散模型在序列中稳定性差、内存受限的问题。提出混合框架：结合扩散模型生成的材质先验与时空正则化，通过光流引导实现时序一致的阴影和反射渲染，利用高斯透明场构建网格代理进行物理渲染，显著提升重光照稳定性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.23494v1](http://arxiv.org/pdf/2510.23494v1)**

> **作者:** Elisabeth Jüttner; Leona Krath; Stefan Korfhage; Hannah Dröge; Matthias B. Hullin; Markus Plack
>
> **摘要:** Volumetric video relighting is essential for bringing captured performances into virtual worlds, but current approaches struggle to deliver temporally stable, production-ready results. Diffusion-based intrinsic decomposition methods show promise for single frames, yet suffer from stochastic noise and instability when extended to sequences, while video diffusion models remain constrained by memory and scale. We propose a hybrid relighting framework that combines diffusion-derived material priors with temporal regularization and physically motivated rendering. Our method aggregates multiple stochastic estimates of per-frame material properties into temporally consistent shading components, using optical-flow-guided regularization. For indirect effects such as shadows and reflections, we extract a mesh proxy from Gaussian Opacity Fields and render it within a standard graphics pipeline. Experiments on real and synthetic captures show that this hybrid strategy achieves substantially more stable relighting across sequences than diffusion-only baselines, while scaling beyond the clip lengths feasible for video diffusion. These results indicate that hybrid approaches, which balance learned priors with physically grounded constraints, are a practical step toward production-ready volumetric video relighting.
>
---
#### [new 099] Positional Preservation Embedding for Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型中视觉令牌冗余导致的效率问题，提出位置保全嵌入（PPE）方法。PPE在压缩视觉令牌时保留时空结构，支持级联聚类，无需参数且可无缝集成。实验表明，PPE显著提升多个视觉语言任务性能，验证了位置信息对高效推理的关键作用。**

- **链接: [http://arxiv.org/pdf/2510.22936v1](http://arxiv.org/pdf/2510.22936v1)**

> **作者:** Mouxiao Huang; Borui Jiang; Dehua Zheng; Hailin Hu; Kai Han; Xinghao Chen
>
> **摘要:** Multimodal large language models (MLLMs) have achieved strong performance on vision-language tasks, yet often suffer from inefficiencies due to redundant visual tokens. Existing token merging methods reduce sequence length but frequently disrupt spatial layouts and temporal continuity by disregarding positional relationships. In this work, we propose a novel encoding operator dubbed as \textbf{P}ositional \textbf{P}reservation \textbf{E}mbedding (\textbf{PPE}), which has the main hallmark of preservation of spatiotemporal structure during visual token compression. PPE explicitly introduces the disentangled encoding of 3D positions in the token dimension, enabling each compressed token to encapsulate different positions from multiple original tokens. Furthermore, we show that PPE can effectively support cascade clustering -- a progressive token compression strategy that leads to better performance retention. PPE is a parameter-free and generic operator that can be seamlessly integrated into existing token merging methods without any adjustments. Applied to state-of-the-art token merging framework, PPE achieves consistent improvements of $2\%\sim5\%$ across multiple vision-language benchmarks, including MMBench (general vision understanding), TextVQA (layout understanding) and VideoMME (temporal understanding). These results demonstrate that preserving positional cues is critical for efficient and effective MLLM reasoning.
>
---
#### [new 100] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views
- **分类: cs.CV; cs.CL; cs.RO; I.2.10; I.2.9; I.2.7; H.5.2**

- **简介: 该论文提出Look and Tell数据集，用于研究第一人称与第三人称视角下的多模态语义对齐。针对跨视角指代理解难题，通过同步记录眼动、语音与视频，结合3D场景重建，提供2.7k条标注的指代表达，推动具身智能体在情境对话中的理解能力发展。**

- **链接: [http://arxiv.org/pdf/2510.22672v1](http://arxiv.org/pdf/2510.22672v1)**

> **作者:** Anna Deichler; Jonas Beskow
>
> **备注:** 10 pages, 6 figures, 2 tables. Accepted to the NeurIPS 2025 Workshop on SPACE in Vision, Language, and Embodied AI (SpaVLE)
>
> **摘要:** We introduce Look and Tell, a multimodal dataset for studying referential communication across egocentric and exocentric perspectives. Using Meta Project Aria smart glasses and stationary cameras, we recorded synchronized gaze, speech, and video as 25 participants instructed a partner to identify ingredients in a kitchen. Combined with 3D scene reconstructions, this setup provides a benchmark for evaluating how different spatial representations (2D vs. 3D; ego vs. exo) affect multimodal grounding. The dataset contains 3.67 hours of recordings, including 2,707 richly annotated referential expressions, and is designed to advance the development of embodied agents that can understand and engage in situated dialogue.
>
---
#### [new 101] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对低预算语义分割任务，提出一种两阶段主动学习方法。利用预训练扩散模型提取多尺度特征，在第一阶段通过MaxHerding选择代表性像素，第二阶段结合熵与分歧度筛选高信息量像素，有效平衡多样性与不确定性，显著提升小样本下的分割精度。**

- **链接: [http://arxiv.org/pdf/2510.22229v1](http://arxiv.org/pdf/2510.22229v1)**

> **作者:** Jeongin Kim; Wonho Bae; YouLee Han; Giyeong Oh; Youngjae Yu; Danica J. Sutherland; Junhyug Noh
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Semantic segmentation demands dense pixel-level annotations, which can be prohibitively expensive - especially under extremely constrained labeling budgets. In this paper, we address the problem of low-budget active learning for semantic segmentation by proposing a novel two-stage selection pipeline. Our approach leverages a pre-trained diffusion model to extract rich multi-scale features that capture both global structure and fine details. In the first stage, we perform a hierarchical, representation-based candidate selection by first choosing a small subset of representative pixels per image using MaxHerding, and then refining these into a diverse global pool. In the second stage, we compute an entropy-augmented disagreement score (eDALD) over noisy multi-scale diffusion features to capture both epistemic uncertainty and prediction confidence, selecting the most informative pixels for annotation. This decoupling of diversity and uncertainty lets us achieve high segmentation accuracy with only a tiny fraction of labeled pixels. Extensive experiments on four benchmarks (CamVid, ADE-Bed, Cityscapes, and Pascal-Context) demonstrate that our method significantly outperforms existing baselines under extreme pixel-budget regimes. Our code is available at https://github.com/jn-kim/two-stage-edald.
>
---
#### [new 102] Caption-Driven Explainability: Probing CNNs for Bias via CLIP
- **分类: cs.CV; eess.IV; I.2.6; I.2.8; I.2.10; I.4.8**

- **简介: 该论文提出一种基于图像描述的可解释人工智能方法，旨在解决卷积神经网络因误用显著但虚假特征导致的鲁棒性问题。通过将待解释模型融入CLIP框架，利用文本描述识别影响预测的关键概念，提升解释准确性，助力构建更稳健的机器学习模型。**

- **链接: [http://arxiv.org/pdf/2510.22035v1](http://arxiv.org/pdf/2510.22035v1)**

> **作者:** Patrick Koller; Amil V. Dravid; Guido M. Schuster; Aggelos K. Katsaggelos
>
> **备注:** Accepted and presented at the IEEE ICIP 2025 Satellite Workshop "Generative AI for World Simulations and Communications & Celebrating 40 Years of Excellence in Education: Honoring Professor Aggelos Katsaggelos", Anchorage, Alaska, United States, September 14, 2025. Camera-ready preprint. The official IEEE Xplore version will be available after proceedings processing
>
> **摘要:** Robustness has become one of the most critical problems in machine learning (ML). The science of interpreting ML models to understand their behavior and improve their robustness is referred to as explainable artificial intelligence (XAI). One of the state-of-the-art XAI methods for computer vision problems is to generate saliency maps. A saliency map highlights the pixel space of an image that excites the ML model the most. However, this property could be misleading if spurious and salient features are present in overlapping pixel spaces. In this paper, we propose a caption-based XAI method, which integrates a standalone model to be explained into the contrastive language-image pre-training (CLIP) model using a novel network surgery approach. The resulting caption-based XAI model identifies the dominant concept that contributes the most to the models prediction. This explanation minimizes the risk of the standalone model falling for a covariate shift and contributes significantly towards developing robust ML models.
>
---
#### [new 103] MMSD3.0: A Multi-Image Benchmark for Real-World Multimodal Sarcasm Detection
- **分类: cs.CV; cs.MM**

- **简介: 该论文聚焦多图讽刺检测任务，针对现有研究忽视多图语义关联的问题，提出MMSD3.0基准数据集与跨图推理模型CIRM，通过细粒度跨模态融合增强多图信息整合，显著提升真实场景下的讽刺识别性能。**

- **链接: [http://arxiv.org/pdf/2510.23299v1](http://arxiv.org/pdf/2510.23299v1)**

> **作者:** Haochen Zhao; Yuyao Kong; Yongxiu Xu; Gaopeng Gou; Hongbo Xu; Yubin Wang; Haoliang Zhang
>
> **摘要:** Despite progress in multimodal sarcasm detection, existing datasets and methods predominantly focus on single-image scenarios, overlooking potential semantic and affective relations across multiple images. This leaves a gap in modeling cases where sarcasm is triggered by multi-image cues in real-world settings. To bridge this gap, we introduce MMSD3.0, a new benchmark composed entirely of multi-image samples curated from tweets and Amazon reviews. We further propose the Cross-Image Reasoning Model (CIRM), which performs targeted cross-image sequence modeling to capture latent inter-image connections. In addition, we introduce a relevance-guided, fine-grained cross-modal fusion mechanism based on text-image correspondence to reduce information loss during integration. We establish a comprehensive suite of strong and representative baselines and conduct extensive experiments, showing that MMSD3.0 is an effective and reliable benchmark that better reflects real-world conditions. Moreover, CIRM demonstrates state-of-the-art performance across MMSD, MMSD2.0 and MMSD3.0, validating its effectiveness in both single-image and multi-image scenarios.
>
---
#### [new 104] EndoWave: Rational-Wavelet 4D Gaussian Splatting for Endoscopic Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对内窥镜视频三维重建任务，解决光照不一致、组织非刚性运动和视点相关高光等问题。提出EndoWave框架，通过4D高斯溅射结合光流几何约束与多分辨率有理小波监督，提升时空一致性与细节还原能力，显著改善重建质量。**

- **链接: [http://arxiv.org/pdf/2510.23087v1](http://arxiv.org/pdf/2510.23087v1)**

> **作者:** Taoyu Wu; Yiyi Miao; Jiaxin Guo; Ziyan Chen; Sihang Zhao; Zhuoxiao Li; Zhe Tang; Baoru Huang; Limin Yu
>
> **摘要:** In robot-assisted minimally invasive surgery, accurate 3D reconstruction from endoscopic video is vital for downstream tasks and improved outcomes. However, endoscopic scenarios present unique challenges, including photometric inconsistencies, non-rigid tissue motion, and view-dependent highlights. Most 3DGS-based methods that rely solely on appearance constraints for optimizing 3DGS are often insufficient in this context, as these dynamic visual artifacts can mislead the optimization process and lead to inaccurate reconstructions. To address these limitations, we present EndoWave, a unified spatio-temporal Gaussian Splatting framework by incorporating an optical flow-based geometric constraint and a multi-resolution rational wavelet supervision. First, we adopt a unified spatio-temporal Gaussian representation that directly optimizes primitives in a 4D domain. Second, we propose a geometric constraint derived from optical flow to enhance temporal coherence and effectively constrain the 3D structure of the scene. Third, we propose a multi-resolution rational orthogonal wavelet as a constraint, which can effectively separate the details of the endoscope and enhance the rendering performance. Extensive evaluations on two real surgical datasets, EndoNeRF and StereoMIS, demonstrate that our method EndoWave achieves state-of-the-art reconstruction quality and visual accuracy compared to the baseline method.
>
---
#### [new 105] Wavelet-based GAN Fingerprint Detection using ResNet50
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像伪造检测任务，旨在识别StyleGAN生成的虚假图像。通过引入离散小波变换（DWT）对图像进行多分辨率预处理，并结合ResNet50分类器，提取生成图像在小波域的独特“指纹”。实验表明，基于Daubechies小波的模型准确率达95.1%，显著优于空间域模型（81.5%），验证了小波域分析在检测深度伪造图像中的有效性。**

- **链接: [http://arxiv.org/pdf/2510.21822v1](http://arxiv.org/pdf/2510.21822v1)**

> **作者:** Sai Teja Erukude; Suhasnadh Reddy Veluru; Viswa Chaitanya Marella
>
> **备注:** 6 pages; Published in IEEE
>
> **摘要:** Identifying images generated by Generative Adversarial Networks (GANs) has become a significant challenge in digital image forensics. This research presents a wavelet-based detection method that uses discrete wavelet transform (DWT) preprocessing and a ResNet50 classification layer to differentiate the StyleGAN-generated images from real ones. Haar and Daubechies wavelet filters are applied to convert the input images into multi-resolution representations, which will then be fed to a ResNet50 network for classification, capitalizing on subtle artifacts left by the generative process. Moreover, the wavelet-based models are compared to an identical ResNet50 model trained on spatial data. The Haar and Daubechies preprocessed models achieved a greater accuracy of 93.8 percent and 95.1 percent, much higher than the model developed in the spatial domain (accuracy rate of 81.5 percent). The Daubechies-based model outperforms Haar, showing that adding layers of descriptive frequency patterns can lead to even greater distinguishing power. These results indicate that the GAN-generated images have unique wavelet-domain artifacts or "fingerprints." The method proposed illustrates the effectiveness of wavelet-domain analysis to detect GAN images and emphasizes the potential of further developing the capabilities of future deepfake detection systems.
>
---
#### [new 106] Promptable Fire Segmentation: Unleashing SAM2's Potential for Real-Time Mobile Deployment with Strategic Bounding Box Guidance
- **分类: cs.CV**

- **简介: 该论文聚焦火情分割任务，针对火焰边界不规则、强度多变等问题，系统评估SAM2及其轻量版在移动端的适用性。通过多种提示策略对比，发现边界框提示更优，尤其Box+MP组合表现最佳，同时轻量模型显著降低资源消耗，为实时火情监测提供高效解决方案。**

- **链接: [http://arxiv.org/pdf/2510.21782v1](http://arxiv.org/pdf/2510.21782v1)**

> **作者:** Emmanuel U. Ugwu; Zhang Xinming
>
> **备注:** Accepted for presentation at the 9th International Conference on Image and Graphics Processing (ICIGP 2026) will be held in Wuhan, China during January 16-18, 2026 (publication forthcoming). 6 pages, 3 figures, 3 tables
>
> **摘要:** Fire segmentation remains a critical challenge in computer vision due to flames' irregular boundaries, translucent edges, and highly variable intensities. While the Segment Anything Models (SAM and SAM2) have demonstrated impressive cross-domain generalization capabilities, their effectiveness in fire segmentation -- particularly under mobile deployment constraints -- remains largely unexplored. This paper presents the first comprehensive evaluation of SAM2 variants for fire segmentation, focusing on bounding box prompting strategies to enhance deployment feasibility. We systematically evaluate four SAM2.1 variants (tiny, small, base_plus, large) alongside mobile-oriented variants (TinySAM, MobileSAM) across three fire datasets using multiple prompting strategies: automatic, single positive point (SP), single positive point + single negative point (SP+SN), multiple positive points (MP), bounding box (Box), and hybrid variants (Box+SP and Box+MP). Our experimental results demonstrate that bounding box prompts consistently outperform automatic and single point-based approaches, with Box+MP achieving the highest mean IoU (0.64) and Dice coefficient (0.75) on the Khan dataset. Lightweight variants such as TinySAM and MobileSAM further reduce memory and computational costs, making them more suitable for latency-tolerant edge scenarios. Overall, this work provides critical insights for deploying promptable segmentation models in fire monitoring systems and establishes benchmarks for future research in domain-specific SAM applications. Code is available at: https://github.com/UEmmanuel5/ProFSAM
>
---
#### [new 107] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视觉语言模型在输入噪声下的性能下降问题，提出Mint方法。研究发现，图像嵌入的类间方差随噪声增强而塌缩，导致识别能力下降。为此，Mint通过在线最大化伪标签下的类间方差，实现简单有效的测试时自适应，显著提升模型在多种噪声场景下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.22127v1](http://arxiv.org/pdf/2510.22127v1)**

> **作者:** Wenxuan Bao; Ruxi Deng; Jingrui He
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Pretrained vision-language models such as CLIP achieve strong zero-shot generalization but remain vulnerable to distribution shifts caused by input corruptions. In this work, we investigate how corruptions affect CLIP's image embeddings and uncover a consistent phenomenon we term as embedding variance collapse, where both intra-class and inter-class variances shrink as corruption severity increases. We find that this collapse is closely tied to performance degradation, with inter-class variance strongly correlated with classification accuracy. To explain this phenomenon, we analyze how corruptions alter the structure of the embedding space. Our theoretical results suggest that the visual encoder tends to encode corruption-related signals, which dilute class-discriminative features and compress the representation geometry. We further show that maximizing inter-class variance, even when estimated from pseudo-labels, can provably enhance embedding quality. Based on this insight, we propose Mint, a simple test-time adaptation method that maximizes pseudo-label-based inter-class variance on the fly using a mean accumulator and a gradient accumulator. Mint operates effectively with small batch sizes and consistently improves performance across multiple corruption benchmarks and CLIP architectures. Our code is available at https://github.com/baowenxuan/Mint .
>
---
#### [new 108] It Takes Two to Tango: Two Parallel Samplers Improve Quality in Diffusion Models for Limited Steps
- **分类: cs.CV; cs.LG; stat.ML**

- **简介: 该论文针对扩散模型在有限采样步数下的生成质量问题，提出双并行采样器方法。通过两个采样器在不同时间点进行去噪并融合信息，显著提升图像质量。方法简单、无需微调，且可通用。实验表明，盲目融合会降低质量，增加采样器数量未必更好。**

- **链接: [http://arxiv.org/pdf/2510.21802v1](http://arxiv.org/pdf/2510.21802v1)**

> **作者:** Pedro Cisneros-Velarde
>
> **摘要:** We consider the situation where we have a limited number of denoising steps, i.e., of evaluations of a diffusion model. We show that two parallel processors or samplers under such limitation can improve the quality of the sampled image. Particularly, the two samplers make denoising steps at successive times, and their information is appropriately integrated in the latent image. Remarkably, our method is simple both conceptually and to implement: it is plug-&-play, model agnostic, and does not require any additional fine-tuning or external models. We test our method with both automated and human evaluations for different diffusion models. We also show that a naive integration of the information from the two samplers lowers sample quality. Finally, we find that adding more parallel samplers does not necessarily improve sample quality.
>
---
#### [new 109] LiteDiff
- **分类: cs.CV**

- **简介: 该论文提出LiteDiff，一种轻量级微调方法，用于在数据稀缺的医学影像领域高效适配扩散模型。通过冻结主模型、引入小规模适配层，并结合潜在形态自编码器与像素级判别器，显著降低计算开销并提升泛化能力。实验表明其在胸部X光图像上优于全模型微调。**

- **链接: [http://arxiv.org/pdf/2510.22004v1](http://arxiv.org/pdf/2510.22004v1)**

> **作者:** Ruchir Namjoshi; Nagasai Thadishetty; Vignesh Kumar; Hemanth Venkateshwara
>
> **摘要:** In recent years, diffusion models have demonstrated remarkable success in high-fidelity image synthesis. However, fine-tuning these models for specialized domains, such as medical imaging, remains challenging due to limited domain-specific data and the high computational cost of full model adaptation. In this paper, we introduce Lite-Diff (Lightweight Diffusion Model Adaptation), a novel finetuning approach that integrates lightweight adaptation layers into a frozen diffusion U-Net while enhancing training with a latent morphological autoencoder (for domain-specific latent consistency) and a pixel level discriminator(for adversarial alignment). By freezing weights of the base model and optimizing only small residual adapter modules, LiteDiff significantly reduces the computational overhead and mitigates overfitting, even in minimal-data settings. Additionally, we conduct ablation studies to analyze the effects of selectively integrating adaptation layers in different U-Net blocks, revealing an optimal balance between efficiency and performance. Experiments on three chest X-ray datasets - (1) Kaggle Chest X-Ray Pneumonia, (2) NIH Chest X-ray14 and (3) VinBigData Chest X_ray demonstrate that LiteDiff achieves superior adaptation efficiency compared to naive full fine-tuning. Our framework provides a promising direction for transfer learning in diffusion models, facilitating their deployment in diverse low data domains.
>
---
#### [new 110] CoMo: Compositional Motion Customization for Text-to-Video Generation
- **分类: cs.CV**

- **简介: 该论文聚焦文本到视频生成中的动作可控性问题，针对复杂多主体运动的组合控制难题，提出CoMo框架。通过解耦运动与外观的两阶段方法，实现无需额外训练的多动作融合，显著提升运动精确性与合成质量。**

- **链接: [http://arxiv.org/pdf/2510.23007v1](http://arxiv.org/pdf/2510.23007v1)**

> **作者:** Youcan Xu; Zhen Wang; Jiaxin Shi; Kexin Li; Feifei Shao; Jun Xiao; Yi Yang; Jun Yu; Long Chen
>
> **摘要:** While recent text-to-video models excel at generating diverse scenes, they struggle with precise motion control, particularly for complex, multi-subject motions. Although methods for single-motion customization have been developed to address this gap, they fail in compositional scenarios due to two primary challenges: motion-appearance entanglement and ineffective multi-motion blending. This paper introduces CoMo, a novel framework for $\textbf{compositional motion customization}$ in text-to-video generation, enabling the synthesis of multiple, distinct motions within a single video. CoMo addresses these issues through a two-phase approach. First, in the single-motion learning phase, a static-dynamic decoupled tuning paradigm disentangles motion from appearance to learn a motion-specific module. Second, in the multi-motion composition phase, a plug-and-play divide-and-merge strategy composes these learned motions without additional training by spatially isolating their influence during the denoising process. To facilitate research in this new domain, we also introduce a new benchmark and a novel evaluation metric designed to assess multi-motion fidelity and blending. Extensive experiments demonstrate that CoMo achieves state-of-the-art performance, significantly advancing the capabilities of controllable video generation. Our project page is at https://como6.github.io/.
>
---
#### [new 111] LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割任务，解决多尺度输出融合效率低的问题。提出LoMix模块，通过可学习加权混合多尺度logits，自动优化融合策略与损失权重，提升模型性能且无推理开销。**

- **链接: [http://arxiv.org/pdf/2510.22995v1](http://arxiv.org/pdf/2510.22995v1)**

> **作者:** Md Mostafijur Rahman; Radu Marculescu
>
> **备注:** 25 pages, 13 figures, NeurIPS 2025 accepted paper
>
> **摘要:** U-shaped networks output logits at multiple spatial scales, each capturing a different blend of coarse context and fine detail. Yet, training still treats these logits in isolation - either supervising only the final, highest-resolution logits or applying deep supervision with identical loss weights at every scale - without exploring mixed-scale combinations. Consequently, the decoder output misses the complementary cues that arise only when coarse and fine predictions are fused. To address this issue, we introduce LoMix (Logits Mixing), a NAS-inspired, differentiable plug-and-play module that generates new mixed-scale outputs and learns how exactly each of them should guide the training process. More precisely, LoMix mixes the multi-scale decoder logits with four lightweight fusion operators: addition, multiplication, concatenation, and attention-based weighted fusion, yielding a rich set of synthetic mutant maps. Every original or mutant map is given a softplus loss weight that is co-optimized with network parameters, mimicking a one-step architecture search that automatically discovers the most useful scales, mixtures, and operators. Plugging LoMix into recent U-shaped architectures (i.e., PVT-V2-B2 backbone with EMCAD decoder) on Synapse 8-organ dataset improves DICE by +4.2% over single-output supervision, +2.2% over deep supervision, and +1.5% over equally weighted additive fusion, all with zero inference overhead. When training data are scarce (e.g., one or two labeled scans), the advantage grows to +9.23%, underscoring LoMix's data efficiency. Across four benchmarks and diverse U-shaped networks, LoMiX improves DICE by up to +13.5% over single-output supervision, confirming that learnable weighted mixed-scale fusion generalizes broadly while remaining data efficient, fully interpretable, and overhead-free at inference. Our code is available at https://github.com/SLDGroup/LoMix.
>
---
#### [new 112] DAMap: Distance-aware MapNet for High Quality HD Map Construction
- **分类: cs.CV**

- **简介: 该论文针对高精地图构建中的高质量预测问题，指出现有方法因任务标签与特征共享导致的偏差。提出DAMap框架，通过距离感知焦点损失、混合损失策略和任务调制可变形注意力机制，提升分类与定位精度，在NuScenes和Argoverse2上实现显著性能提升。**

- **链接: [http://arxiv.org/pdf/2510.22675v1](http://arxiv.org/pdf/2510.22675v1)**

> **作者:** Jinpeng Dong; Chen Li; Yutong Lin; Jingwen Fu; Sanping Zhou; Nanning Zheng
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** Predicting High-definition (HD) map elements with high quality (high classification and localization scores) is crucial to the safety of autonomous driving vehicles. However, current methods perform poorly in high quality predictions due to inherent task misalignment. Two main factors are responsible for misalignment: 1) inappropriate task labels due to one-to-many matching queries sharing the same labels, and 2) sub-optimal task features due to task-shared sampling mechanism. In this paper, we reveal two inherent defects in current methods and develop a novel HD map construction method named DAMap to address these problems. Specifically, DAMap consists of three components: Distance-aware Focal Loss (DAFL), Hybrid Loss Scheme (HLS), and Task Modulated Deformable Attention (TMDA). The DAFL is introduced to assign appropriate classification labels for one-to-many matching samples. The TMDA is proposed to obtain discriminative task-specific features. Furthermore, the HLS is proposed to better utilize the advantages of the DAFL. We perform extensive experiments and consistently achieve performance improvement on the NuScenes and Argoverse2 benchmarks under different metrics, baselines, splits, backbones, and schedules. Code will be available at https://github.com/jpdong-xjtu/DAMap.
>
---
#### [new 113] Xihe: Scalable Zero-Shot Time Series Learner Via Hierarchical Interleaved Block Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Xihe模型，针对时间序列基础模型在零样本迁移中难以捕捉多尺度时序依赖的问题，设计了分层交错块注意力（HIBA）机制，实现局部与全局时序模式的有效建模。通过可扩展架构，在参数效率与性能上均取得突破，显著提升零样本任务表现。**

- **链接: [http://arxiv.org/pdf/2510.21795v1](http://arxiv.org/pdf/2510.21795v1)**

> **作者:** Yinbo Sun; Yuchen Fang; Zhibo Zhu; Jia Li; Yu Liu; Qiwen Deng; Jun Zhou; Hang Yu; Xingyu Lu; Lintao Ma
>
> **摘要:** The rapid advancement of time series foundation models (TSFMs) has been propelled by migrating architectures from language models. While existing TSFMs demonstrate impressive performance, their direct adoption of cross-domain architectures constrains effective capture of multiscale temporal dependencies inherent to time series data. This limitation becomes particularly pronounced during zero-shot transfer across datasets with divergent underlying patterns and sampling strategies. To address these challenges, we propose Hierarchical Interleaved Block Attention (HIBA) which employs hierarchical inter- and intra-block sparse attention to effectively capture multi-scale dependencies. Intra-block attention facilitates local information exchange, and inter-block attention operates across blocks to capture global temporal pattern interaction and dynamic evolution. Leveraging the HIBA architecture, we introduce Xihe, a scalable TSFM family spanning from an ultra-efficient 9.5M parameter configuration to high-capacity 1.5B variant. Evaluated on the comprehensive GIFT-Eval benchmark, our most compact Xihe-tiny model (9.5M) surpasses the majority of contemporary TSFMs, demonstrating remarkable parameter efficiency. More impressively, Xihe-max (1.5B) establishes new state-of-the-art zero-shot performance, surpassing previous best results by a substantial margin. This consistent performance excellence across the entire parameter spectrum provides compelling evidence for the exceptional generalization capabilities and architectural superiority of HIBA.
>
---
#### [new 114] UrbanIng-V2X: A Large-Scale Multi-Vehicle, Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception
- **分类: cs.CV**

- **简介: 该论文提出UrbanIng-V2X数据集，解决多交叉口、多车与基础设施协同感知数据缺失问题。旨在支持智能交通中复杂场景下的协同感知算法研究，提供大规模、多模态、高精度标注数据，涵盖3个交叉口、34个序列，含12辆车载相机、2个车载LiDAR、17个路侧热成像相机和12个路侧LiDAR，共71.2万3D标注实例。**

- **链接: [http://arxiv.org/pdf/2510.23478v1](http://arxiv.org/pdf/2510.23478v1)**

> **作者:** Karthikeyan Chandra Sekaran; Markus Geisler; Dominik Rößle; Adithya Mohan; Daniel Cremers; Wolfgang Utschick; Michael Botsch; Werner Huber; Torsten Schön
>
> **备注:** Accepted to NeurIPS 2025. Including supplemental material. For code and dataset, see https://github.com/thi-ad/UrbanIng-V2X
>
> **摘要:** Recent cooperative perception datasets have played a crucial role in advancing smart mobility applications by enabling information exchange between intelligent agents, helping to overcome challenges such as occlusions and improving overall scene understanding. While some existing real-world datasets incorporate both vehicle-to-vehicle and vehicle-to-infrastructure interactions, they are typically limited to a single intersection or a single vehicle. A comprehensive perception dataset featuring multiple connected vehicles and infrastructure sensors across several intersections remains unavailable, limiting the benchmarking of algorithms in diverse traffic environments. Consequently, overfitting can occur, and models may demonstrate misleadingly high performance due to similar intersection layouts and traffic participant behavior. To address this gap, we introduce UrbanIng-V2X, the first large-scale, multi-modal dataset supporting cooperative perception involving vehicles and infrastructure sensors deployed across three urban intersections in Ingolstadt, Germany. UrbanIng-V2X consists of 34 temporally aligned and spatially calibrated sensor sequences, each lasting 20 seconds. All sequences contain recordings from one of three intersections, involving two vehicles and up to three infrastructure-mounted sensor poles operating in coordinated scenarios. In total, UrbanIng-V2X provides data from 12 vehicle-mounted RGB cameras, 2 vehicle LiDARs, 17 infrastructure thermal cameras, and 12 infrastructure LiDARs. All sequences are annotated at a frequency of 10 Hz with 3D bounding boxes spanning 13 object classes, resulting in approximately 712k annotated instances across the dataset. We provide comprehensive evaluations using state-of-the-art cooperative perception methods and publicly release the codebase, dataset, HD map, and a digital twin of the complete data collection environment.
>
---
#### [new 115] DiffusionLane: Diffusion Model for Lane Detection
- **分类: cs.CV**

- **简介: 该论文提出DiffusionLane，一种基于扩散模型的车道线检测方法。将车道检测建模为参数空间的去噪过程，通过噪声车道锚点逐步优化，结合混合解码器与辅助监督提升特征表示，显著提升检测精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.22236v1](http://arxiv.org/pdf/2510.22236v1)**

> **作者:** Kunyang Zhou; Yeqin Shao
>
> **摘要:** In this paper, we present a novel diffusion-based model for lane detection, called DiffusionLane, which treats the lane detection task as a denoising diffusion process in the parameter space of the lane. Firstly, we add the Gaussian noise to the parameters (the starting point and the angle) of ground truth lanes to obtain noisy lane anchors, and the model learns to refine the noisy lane anchors in a progressive way to obtain the target lanes. Secondly, we propose a hybrid decoding strategy to address the poor feature representation of the encoder, resulting from the noisy lane anchors. Specifically, we design a hybrid diffusion decoder to combine global-level and local-level decoders for high-quality lane anchors. Then, to improve the feature representation of the encoder, we employ an auxiliary head in the training stage to adopt the learnable lane anchors for enriching the supervision on the encoder. Experimental results on four benchmarks, Carlane, Tusimple, CULane, and LLAMAS, show that DiffusionLane possesses a strong generalization ability and promising detection performance compared to the previous state-of-the-art methods. For example, DiffusionLane with ResNet18 surpasses the existing methods by at least 1\% accuracy on the domain adaptation dataset Carlane. Besides, DiffusionLane with MobileNetV4 gets 81.32\% F1 score on CULane, 96.89\% accuracy on Tusimple with ResNet34, and 97.59\% F1 score on LLAMAS with ResNet101. Code will be available at https://github.com/zkyntu/UnLanedet.
>
---
#### [new 116] MELDAE: A Framework for Micro-Expression Spotting, Detection, and Automatic Evaluation in In-the-Wild Conversational Scenes
- **分类: cs.CV**

- **简介: 该论文针对真实对话场景下微表情的定位与检测难题，提出首个面向对话场景的微表情数据集、端到端检测框架MELDAE及边界感知损失函数，显著提升时序精度与泛化能力，实现更准确的微表情自动分析。**

- **链接: [http://arxiv.org/pdf/2510.22575v1](http://arxiv.org/pdf/2510.22575v1)**

> **作者:** Yigui Feng; Qinglin Wang; Yang Liu; Ke Liu; Haotian Mo; Enhao Huang; Gencheng Liu; Mingzhe Liu; Jie Liu
>
> **摘要:** Accurately analyzing spontaneous, unconscious micro-expressions is crucial for revealing true human emotions, but this task remains challenging in wild scenarios, such as natural conversation. Existing research largely relies on datasets from controlled laboratory environments, and their performance degrades dramatically in the real world. To address this issue, we propose three contributions: the first micro-expression dataset focused on conversational-in-the-wild scenarios; an end-to-end localization and detection framework, MELDAE; and a novel boundary-aware loss function that improves temporal accuracy by penalizing onset and offset errors. Extensive experiments demonstrate that our framework achieves state-of-the-art results on the WDMD dataset, improving the key F1_{DR} localization metric by 17.72% over the strongest baseline, while also demonstrating excellent generalization capabilities on existing benchmarks.
>
---
#### [new 117] Scanner-Agnostic MRI Harmonization via SSIM-Guided Disentanglement
- **分类: cs.CV**

- **简介: 该论文针对多中心MRI图像因扫描仪、协议和站点差异导致的异质性问题，提出一种基于SSIM引导解耦的图像级配准方法。通过分离解剖结构与设备/站点特异性因素，实现跨中心图像一致性提升，显著改善脑龄预测与阿尔茨海默病分类性能。**

- **链接: [http://arxiv.org/pdf/2510.22073v1](http://arxiv.org/pdf/2510.22073v1)**

> **作者:** Luca Caldera; Lara Cavinato; Francesca Ieva
>
> **摘要:** The variability introduced by differences in MRI scanner models, acquisition protocols, and imaging sites hinders consistent analysis and generalizability across multicenter studies. We present a novel image-based harmonization framework for 3D T1-weighted brain MRI, which disentangles anatomical content from scanner- and site-specific variations. The model incorporates a differentiable loss based on the Structural Similarity Index (SSIM) to preserve biologically meaningful features while reducing inter-site variability. This loss enables separate evaluation of image luminance, contrast, and structural components. Training and validation were performed on multiple publicly available datasets spanning diverse scanners and sites, with testing on both healthy and clinical populations. Harmonization using multiple style targets, including style-agnostic references, produced consistent and high-quality outputs. Visual comparisons, voxel intensity distributions, and SSIM-based metrics demonstrated that harmonized images achieved strong alignment across acquisition settings while maintaining anatomical fidelity. Following harmonization, structural SSIM reached 0.97, luminance SSIM ranged from 0.98 to 0.99, and Wasserstein distances between mean voxel intensity distributions decreased substantially. Downstream tasks showed substantial improvements: mean absolute error for brain age prediction decreased from 5.36 to 3.30 years, and Alzheimer's disease classification AUC increased from 0.78 to 0.85. Overall, our framework enhances cross-site image consistency, preserves anatomical fidelity, and improves downstream model performance, providing a robust and generalizable solution for large-scale multicenter neuroimaging studies.
>
---
#### [new 118] A Critical Study on Tea Leaf Disease Detection using Deep Learning Techniques
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标检测与实例分割任务，旨在识别茶叶病害（红锈病、Helopeltis害虫、红蜘蛛螨），并定位病变区域。对比SSD MobileNet V2与Faster R-CNN ResNet50 V1模型，后者表现更优；进一步采用Mask R-CNN实现病斑精确分割，提出自定义方法量化病害面积。**

- **链接: [http://arxiv.org/pdf/2510.22647v1](http://arxiv.org/pdf/2510.22647v1)**

> **作者:** Nabajyoti Borah; Raju Moni Borah; Bandan Boruah; Purnendu Bikash Acharjee; Sajal Saha; Ripjyoti Hazarika
>
> **摘要:** The proposed solution is Deep Learning Technique that will be able classify three types of tea leaves diseases from which two diseases are caused by the pests and one due to pathogens (infectious organisms) and environmental conditions and also show the area damaged by a disease in leaves. Namely Red Rust, Helopeltis and Red spider mite respectively. In this paper we have evaluated two models namely SSD MobileNet V2 and Faster R-CNN ResNet50 V1 for the object detection. The SSD MobileNet V2 gave precision of 0.209 for IOU range of 0.50:0.95 with recall of 0.02 on IOU 0.50:0.95 and final mAP of 20.9%. While Faster R-CNN ResNet50 V1 has precision of 0.252 on IOU range of 0.50:0.95 and recall of 0.044 on IOU of 0.50:0.95 with a mAP of 25%, which is better than SSD. Also used Mask R-CNN for Object Instance Segmentation where we have implemented our custom method to calculate the damaged diseased portion of leaves. Keywords: Tea Leaf Disease, Deep Learning, Red Rust, Helopeltis and Red Spider Mite, SSD MobileNet V2, Faster R-CNN ResNet50 V1 and Mask RCNN.
>
---
#### [new 119] iPac: Incorporating Intra-image Patch Context into Graph Neural Networks for Medical Image Classification
- **分类: cs.CV**

- **简介: 该论文针对医学图像分类任务，解决图神经网络忽视图像内部结构与局部区域关系的问题。提出iPac方法，通过分块、特征提取、聚类构建含上下文信息的图结构，提升模型对图像语义的理解能力，实验显示平均准确率提升5%。**

- **链接: [http://arxiv.org/pdf/2510.23504v1](http://arxiv.org/pdf/2510.23504v1)**

> **作者:** Usama Zidan; Mohamed Gaber; Mohammed M. Abdelsamea
>
> **备注:** Accepted for publication in the proceedings of ICONIP 2025
>
> **摘要:** Graph neural networks have emerged as a promising paradigm for image processing, yet their performance in image classification tasks is hindered by a limited consideration of the underlying structure and relationships among visual entities. This work presents iPac, a novel approach to introduce a new graph representation of images to enhance graph neural network image classification by recognizing the importance of underlying structure and relationships in medical image classification. iPac integrates various stages, including patch partitioning, feature extraction, clustering, graph construction, and graph-based learning, into a unified network to advance graph neural network image classification. By capturing relevant features and organising them into clusters, we construct a meaningful graph representation that effectively encapsulates the semantics of the image. Experimental evaluation on diverse medical image datasets demonstrates the efficacy of iPac, exhibiting an average accuracy improvement of up to 5% over baseline methods. Our approach offers a versatile and generic solution for image classification, particularly in the realm of medical images, by leveraging the graph representation and accounting for the inherent structure and relationships among visual entities.
>
---
#### [new 120] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文针对多模态大模型生成中检索策略僵化、模态选择不灵活、信息利用低效的问题，提出Windsock框架实现查询相关的检索决策与模态选择，并引入动态抗噪训练与自评估数据构建方法，显著提升生成质量并降低检索开销。**

- **链接: [http://arxiv.org/pdf/2510.22694v1](http://arxiv.org/pdf/2510.22694v1)**

> **作者:** Shu Zhao; Tianyi Shen; Nilesh Ahuja; Omesh Tickoo; Vijaykrishnan Narayanan
>
> **备注:** Accepted at NeurIPS 2025 UniReps Workshop
>
> **摘要:** Multimodal Retrieval-Augmented Generation (MRAG) has emerged as a promising method to generate factual and up-to-date responses of Multimodal Large Language Models (MLLMs) by incorporating non-parametric knowledge from external knowledge bases. However, existing MRAG approaches suffer from static retrieval strategies, inflexible modality selection, and suboptimal utilization of retrieved information, leading to three critical challenges: determining when to retrieve, what modality to incorporate, and how to utilize retrieved information effectively. To address these challenges, we introduce Windsock, a query-dependent module making decisions on retrieval necessity and modality selection, effectively reducing computational overhead and improving response quality. Additionally, we propose Dynamic Noise-Resistance (DANCE) Instruction Tuning, an adaptive training strategy that enhances MLLMs' ability to utilize retrieved information while maintaining robustness against noise. Moreover, we adopt a self-assessment approach leveraging knowledge within MLLMs to convert question-answering datasets to MRAG training datasets. Extensive experiments demonstrate that our proposed method significantly improves the generation quality by 17.07% while reducing 8.95% retrieval times.
>
---
#### [new 121] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于知识蒸馏任务，旨在降低多教师带来的高计算成本。提出单教师视图增强方法，通过附加分支生成角度多样化的多视图，利用两种角度多样性损失提升知识多样性，从而在保持高效的同时增强学生模型性能。**

- **链接: [http://arxiv.org/pdf/2510.22480v1](http://arxiv.org/pdf/2510.22480v1)**

> **作者:** Seonghoon Yu; Dongjun Nam; Dina Katabi; Jeany Son
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Knowledge Distillation (KD) aims to train a lightweight student model by transferring knowledge from a large, high-capacity teacher. Recent studies have shown that leveraging diverse teacher perspectives can significantly improve distillation performance; however, achieving such diversity typically requires multiple teacher networks, leading to high computational costs. In this work, we propose a novel cost-efficient knowledge augmentation method for KD that generates diverse multi-views by attaching multiple branches to a single teacher. To ensure meaningful semantic variation across multi-views, we introduce two angular diversity objectives: 1) constrained inter-angle diversify loss, which maximizes angles between augmented views while preserving proximity to the original teacher output, and 2) intra-angle diversify loss, which encourages an even distribution of views around the original output. The ensembled knowledge from these angularly diverse views, along with the original teacher, is distilled into the student. We further theoretically demonstrate that our objectives increase the diversity among ensemble members and thereby reduce the upper bound of the ensemble's expected loss, leading to more effective distillation. Experimental results show that our method surpasses an existing knowledge augmentation method across diverse configurations. Moreover, the proposed method is compatible with other KD frameworks in a plug-and-play fashion, providing consistent improvements in generalization performance.
>
---
#### [new 122] Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture
- **分类: cs.CV**

- **简介: 该论文针对精准畜牧业中的牧草生物量估测问题，构建了1,162张带标注的俯视图像数据集，涵盖多季节、多物种牧场。数据融合视觉、光谱与结构信息，用于训练机器学习模型，推动智能放牧管理发展。**

- **链接: [http://arxiv.org/pdf/2510.22916v1](http://arxiv.org/pdf/2510.22916v1)**

> **作者:** Qiyu Liao; Dadong Wang; Rebecca Haling; Jiajun Liu; Xun Li; Martyna Plomecka; Andrew Robson; Matthew Pringle; Rhys Pirie; Megan Walker; Joshua Whelan
>
> **备注:** 9 pages, 2 figures, 2 tables, The dataset is available on the official Kaggle webpage: https://www.kaggle.com/competitions/csiro-biomass
>
> **摘要:** Accurate estimation of pasture biomass is important for decision-making in livestock production systems. Estimates of pasture biomass can be used to manage stocking rates to maximise pasture utilisation, while minimising the risk of overgrazing and promoting overall system health. We present a comprehensive dataset of 1,162 annotated top-view images of pastures collected across 19 locations in Australia. The images were taken across multiple seasons and include a range of temperate pasture species. Each image captures a 70cm * 30cm quadrat and is paired with on-ground measurements including biomass sorted by component (green, dead, and legume fraction), vegetation height, and Normalized Difference Vegetation Index (NDVI) from Active Optical Sensors (AOS). The multidimensional nature of the data, which combines visual, spectral, and structural information, opens up new possibilities for advancing the use of precision grazing management. The dataset is released and hosted in a Kaggle competition that challenges the international Machine Learning community with the task of pasture biomass estimation. The dataset is available on the official Kaggle webpage: https://www.kaggle.com/competitions/csiro-biomass
>
---
#### [new 123] Cross-view Localization and Synthesis - Datasets, Challenges and Opportunities
- **分类: cs.CV**

- **简介: 该论文聚焦跨视角定位与合成任务，解决卫星/航拍图像与地面图像之间的视图差异问题。通过综述主流数据集、技术方法及挑战，系统分析了基于CNN/ViT的定位与基于GAN/扩散模型的合成进展，指出当前局限并展望未来方向。**

- **链接: [http://arxiv.org/pdf/2510.22736v1](http://arxiv.org/pdf/2510.22736v1)**

> **作者:** Ningli Xu; Rongjun Qin
>
> **备注:** 15 Figures
>
> **摘要:** Cross-view localization and synthesis are two fundamental tasks in cross-view visual understanding, which deals with cross-view datasets: overhead (satellite or aerial) and ground-level imagery. These tasks have gained increasing attention due to their broad applications in autonomous navigation, urban planning, and augmented reality. Cross-view localization aims to estimate the geographic position of ground-level images based on information provided by overhead imagery while cross-view synthesis seeks to generate ground-level images based on information from the overhead imagery. Both tasks remain challenging due to significant differences in viewing perspective, resolution, and occlusion, which are widely embedded in cross-view datasets. Recent years have witnessed rapid progress driven by the availability of large-scale datasets and novel approaches. Typically, cross-view localization is formulated as an image retrieval problem where ground-level features are matched with tiled overhead images feature, extracted by convolutional neural networks (CNNs) or vision transformers (ViTs) for cross-view feature embedding. Cross-view synthesis, on the other hand, seeks to generate ground-level views based on information from overhead imagery, generally using generative adversarial networks (GANs) or diffusion models. This paper presents a comprehensive survey of advances in cross-view localization and synthesis, reviewing widely used datasets, highlighting key challenges, and providing an organized overview of state-of-the-art techniques. Furthermore, it discusses current limitations, offers comparative analyses, and outlines promising directions for future research. We also include the project page via https://github.com/GDAOSU/Awesome-Cross-View-Methods.
>
---
#### [new 124] HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling
- **分类: cs.CV**

- **简介: 该论文聚焦视频时间定位任务，旨在精准定位长视频中自然语言查询的起止时间。针对现有方法在长视频中牺牲时间精度的问题，提出HieraMamba模型，通过层级锚点-Mamba池化结构，融合多粒度语义与时间细节，并设计双损失优化策略，显著提升定位精度，在多个基准上达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2510.23043v1](http://arxiv.org/pdf/2510.23043v1)**

> **作者:** Joungbin An; Kristen Grauman
>
> **备注:** Project Page: https://vision.cs.utexas.edu/projects/hieramamba/
>
> **摘要:** Video temporal grounding, the task of localizing the start and end times of a natural language query in untrimmed video, requires capturing both global context and fine-grained temporal detail. This challenge is particularly pronounced in long videos, where existing methods often compromise temporal fidelity by over-downsampling or relying on fixed windows. We present HieraMamba, a hierarchical architecture that preserves temporal structure and semantic richness across scales. At its core are Anchor-MambaPooling (AMP) blocks, which utilize Mamba's selective scanning to produce compact anchor tokens that summarize video content at multiple granularities. Two complementary objectives, anchor-conditioned and segment-pooled contrastive losses, encourage anchors to retain local detail while remaining globally discriminative. HieraMamba sets a new state-of-the-art on Ego4D-NLQ, MAD, and TACoS, demonstrating precise, temporally faithful localization in long, untrimmed videos.
>
---
#### [new 125] UniAIDet: A Unified and Universal Benchmark for AI-Generated Image Content Detection and Localization
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出UniAIDet，一个统一的AI生成图像内容检测与定位基准。针对现有基准覆盖模型和图像类型有限的问题，构建涵盖多种生成模型与图像类别的全面数据集，并评估检测方法的泛化能力与检测-定位关系，推动AI生成内容检测研究发展。**

- **链接: [http://arxiv.org/pdf/2510.23023v1](http://arxiv.org/pdf/2510.23023v1)**

> **作者:** Huixuan Zhang; Xiaojun Wan
>
> **摘要:** With the rapid proliferation of image generative models, the authenticity of digital images has become a significant concern. While existing studies have proposed various methods for detecting AI-generated content, current benchmarks are limited in their coverage of diverse generative models and image categories, often overlooking end-to-end image editing and artistic images. To address these limitations, we introduce UniAIDet, a unified and comprehensive benchmark that includes both photographic and artistic images. UniAIDet covers a wide range of generative models, including text-to-image, image-to-image, image inpainting, image editing, and deepfake models. Using UniAIDet, we conduct a comprehensive evaluation of various detection methods and answer three key research questions regarding generalization capability and the relation between detection and localization. Our benchmark and analysis provide a robust foundation for future research.
>
---
#### [new 126] Residual Diffusion Bridge Model for Image Restoration
- **分类: cs.CV**

- **简介: 该论文提出残差扩散桥模型（RDBM），用于通用图像修复任务。针对现有方法将扩散桥视为简单插值、全局噪声处理导致未退化区域失真的问题，提出基于残差调制的自适应修复机制，并建立统一理论框架，实现更精准的图像恢复。**

- **链接: [http://arxiv.org/pdf/2510.23116v1](http://arxiv.org/pdf/2510.23116v1)**

> **作者:** Hebaixu Wang; Jing Zhang; Haoyang Chen; Haonan Guo; Di Wang; Jiayi Ma; Bo Du
>
> **摘要:** Diffusion bridge models establish probabilistic paths between arbitrary paired distributions and exhibit great potential for universal image restoration. Most existing methods merely treat them as simple variants of stochastic interpolants, lacking a unified analytical perspective. Besides, they indiscriminately reconstruct images through global noise injection and removal, inevitably distorting undegraded regions due to imperfect reconstruction. To address these challenges, we propose the Residual Diffusion Bridge Model (RDBM). Specifically, we theoretically reformulate the stochastic differential equations of generalized diffusion bridge and derive the analytical formulas of its forward and reverse processes. Crucially, we leverage the residuals from given distributions to modulate the noise injection and removal, enabling adaptive restoration of degraded regions while preserving intact others. Moreover, we unravel the fundamental mathematical essence of existing bridge models, all of which are special cases of RDBM and empirically demonstrate the optimality of our proposed models. Extensive experiments are conducted to demonstrate the state-of-the-art performance of our method both qualitatively and quantitatively across diverse image restoration tasks. Code is publicly available at https://github.com/MiliLab/RDBM.
>
---
#### [new 127] LOC: A General Language-Guided Framework for Open-Set 3D Occupancy Prediction
- **分类: cs.CV; cs.CL; cs.LG; cs.RO; eess.IV**

- **简介: 该论文提出LOC框架，解决3D场景理解中因数据稀缺导致的开放集占用预测难题。通过语言引导融合多帧激光雷达点云与语义信息，结合对比学习增强特征区分性，实现无需额外训练即可识别未知类别的高精度3D占用预测。**

- **链接: [http://arxiv.org/pdf/2510.22141v1](http://arxiv.org/pdf/2510.22141v1)**

> **作者:** Yuhang Gao; Xiang Xiang; Sheng Zhong; Guoyou Wang
>
> **摘要:** Vision-Language Models (VLMs) have shown significant progress in open-set challenges. However, the limited availability of 3D datasets hinders their effective application in 3D scene understanding. We propose LOC, a general language-guided framework adaptable to various occupancy networks, supporting both supervised and self-supervised learning paradigms. For self-supervised tasks, we employ a strategy that fuses multi-frame LiDAR points for dynamic/static scenes, using Poisson reconstruction to fill voids, and assigning semantics to voxels via K-Nearest Neighbor (KNN) to obtain comprehensive voxel representations. To mitigate feature over-homogenization caused by direct high-dimensional feature distillation, we introduce Densely Contrastive Learning (DCL). DCL leverages dense voxel semantic information and predefined textual prompts. This efficiently enhances open-set recognition without dense pixel-level supervision, and our framework can also leverage existing ground truth to further improve performance. Our model predicts dense voxel features embedded in the CLIP feature space, integrating textual and image pixel information, and classifies based on text and semantic similarity. Experiments on the nuScenes dataset demonstrate the method's superior performance, achieving high-precision predictions for known classes and distinguishing unknown classes without additional training data.
>
---
#### [new 128] ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦多视图3D物体重建任务，针对视图重叠不足导致的重建不完整问题，提出ReconViaGen框架。通过引入重建先验，强化跨视图特征关联与生成过程可控性，提升生成结果在全局结构和局部细节上的一致性与准确性。**

- **链接: [http://arxiv.org/pdf/2510.23306v1](http://arxiv.org/pdf/2510.23306v1)**

> **作者:** Jiahao Chang; Chongjie Ye; Yushuang Wu; Yuantao Chen; Yidan Zhang; Zhongjin Luo; Chenghong Li; Yihao Zhi; Xiaoguang Han
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Existing multi-view 3D object reconstruction methods heavily rely on sufficient overlap between input views, where occlusions and sparse coverage in practice frequently yield severe reconstruction incompleteness. Recent advancements in diffusion-based 3D generative techniques offer the potential to address these limitations by leveraging learned generative priors to hallucinate invisible parts of objects, thereby generating plausible 3D structures. However, the stochastic nature of the inference process limits the accuracy and reliability of generation results, preventing existing reconstruction frameworks from integrating such 3D generative priors. In this work, we comprehensively analyze the reasons why diffusion-based 3D generative methods fail to achieve high consistency, including (a) the insufficiency in constructing and leveraging cross-view connections when extracting multi-view image features as conditions, and (b) the poor controllability of iterative denoising during local detail generation, which easily leads to plausible but inconsistent fine geometric and texture details with inputs. Accordingly, we propose ReconViaGen to innovatively integrate reconstruction priors into the generative framework and devise several strategies that effectively address these issues. Extensive experiments demonstrate that our ReconViaGen can reconstruct complete and accurate 3D models consistent with input views in both global structure and local details.Project page: https://jiahao620.github.io/reconviagen.
>
---
#### [new 129] PixelRefer: A Unified Framework for Spatio-Temporal Object Referring with Arbitrary Granularity
- **分类: cs.CV**

- **简介: 该论文提出PixelRefer，一个统一的时空目标指代框架，解决多模态大模型在细粒度对象级理解上的不足。通过自适应对象分词器和轻量级设计，实现任意粒度区域的精准指代，提升效率与精度。**

- **链接: [http://arxiv.org/pdf/2510.23603v1](http://arxiv.org/pdf/2510.23603v1)**

> **作者:** Yuqian Yuan; Wenqiao Zhang; Xin Li; Shihao Wang; Kehan Li; Wentong Li; Jun Xiao; Lei Zhang; Beng Chin Ooi
>
> **备注:** 22 pages, 13 figures
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated strong general-purpose capabilities in open-world visual comprehension. However, most existing MLLMs primarily focus on holistic, scene-level understanding, often overlooking the need for fine-grained, object-centric reasoning. In this paper, we present PixelRefer, a unified region-level MLLM framework that enables advanced fine-grained understanding over user-specified regions across both images and videos. Motivated by the observation that LLM attention predominantly focuses on object-level tokens, we propose a Scale-Adaptive Object Tokenizer (SAOT) to generate compact and semantically rich object representations from free-form regions. Our analysis reveals that global visual tokens contribute mainly in early LLM layers, inspiring the design of PixelRefer-Lite, an efficient variant that employs an Object-Centric Infusion module to pre-fuse global context into object tokens. This yields a lightweight Object-Only Framework that substantially reduces computational cost while maintaining high semantic fidelity. To facilitate fine-grained instruction tuning, we curate PixelRefer-2.2M, a high-quality object-centric instruction dataset. Extensive experiments across a range of benchmarks validate that PixelRefer achieves leading performance with fewer training samples, while PixelRefer-Lite offers competitive accuracy with notable gains in efficiency.
>
---
#### [new 130] Note on the Construction of Structure Tensor
- **分类: cs.CV; math.SP**

- **简介: 该论文探讨结构张量的两种构建方法，指出其在总最小二乘线性拟合框架下可统一。通过重新解释，证明1995年修正项冗余，简化张量性质，并拓展至非正交滤波器与非角向分块应用，提升灵活性与适用性。**

- **链接: [http://arxiv.org/pdf/2510.23137v1](http://arxiv.org/pdf/2510.23137v1)**

> **作者:** Josef Bigun; Fernado Alonso-Fernandez
>
> **摘要:** This note presents a theoretical discussion of two structure tensor constructions: one proposed by Bigun and Granlund 1987, and the other by Granlund and Knutsson 1995. At first glance, these approaches may appear quite different--the former is implemented by averaging outer products of gradient filter responses, while the latter constructs the tensor from weighted outer products of tune-in frequency vectors of quadrature filters. We argue that when both constructions are viewed through the common lens of Total Least Squares (TLS) line fitting to the power spectrum, they can be reconciled to a large extent, and additional benefits emerge. From this perspective, the correction term introduced in Granlund and Knutsson 1995 becomes unnecessary. Omitting it ensures that the resulting tensor remains positive semi-definite, thereby simplifying the interpretation of its eigenvalues. Furthermore, this interpretation allows fitting more than a single 0rientation to the input by reinterpreting quadrature filter responses without relying on a structure tensor. It also removes the constraint that responses must originate strictly from quadrature filters, allowing the use of alternative filter types and non-angular tessellations. These alternatives include Gabor filters--which, although not strictly quadrature, are still suitable for structure tensor construction--even when they tessellate the spectrum in a Cartesian fashion, provided they are sufficiently concentrated.
>
---
#### [new 131] Towards Accurate and Efficient Waste Image Classification: A Hybrid Deep Learning and Machine Learning Approach
- **分类: cs.CV; I.2.10; I.4.8; I.5.4; J.2**

- **简介: 该论文针对垃圾图像分类任务，解决传统方法准确率低、计算成本高的问题。通过对比机器学习、深度学习及混合模型，提出基于深度特征提取与经典分类器结合的高效混合方法，在多个数据集上实现超99.8%准确率，并显著降低特征维度与推理成本。**

- **链接: [http://arxiv.org/pdf/2510.21833v1](http://arxiv.org/pdf/2510.21833v1)**

> **作者:** Ngoc-Bao-Quang Nguyen; Tuan-Minh Do; Cong-Tam Phan; Thi-Thu-Hong Phan
>
> **备注:** 31 pages; 7 figures; 16 tables
>
> **摘要:** Automated image-based garbage classification is a critical component of global waste management; however, systematic benchmarks that integrate Machine Learning (ML), Deep Learning (DL), and efficient hybrid solutions remain underdeveloped. This study provides a comprehensive comparison of three paradigms: (1) machine learning algorithms using handcrafted features, (2) deep learning architectures, including ResNet variants and EfficientNetV2S, and (3) a hybrid approach that utilizes deep models for feature extraction combined with classical classifiers such as Support Vector Machine and Logistic Regression to identify the most effective strategy. Experiments on three public datasets - TrashNet, Garbage Classification, and a refined Household Garbage Dataset (with 43 corrected mislabels)- demonstrate that the hybrid method consistently outperforms the others, achieving up to 100% accuracy on TrashNet and the refined Household set, and 99.87% on Garbage Classification, thereby surpassing state-of-the-art benchmarks. Furthermore, feature selection reduces feature dimensionality by over 95% without compromising accuracy, resulting in faster training and inference. This work establishes more reliable benchmarks for waste classification and introduces an efficient hybrid framework that achieves high accuracy while reducing inference cost, making it suitable for scalable deployment in resource-constrained environments.
>
---
#### [new 132] Noise Aggregation Analysis Driven by Small-Noise Injection: Efficient Membership Inference for Diffusion Models
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文针对扩散模型的隐私风险，提出一种高效成员推断攻击方法。通过注入微小噪声并分析预测噪声分布的聚集程度，区分训练集与非训练集样本。相比现有方法，减少模型调用次数，且在多数据集及大规模文本生成模型上表现更优，验证了方法的有效性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.21783v1](http://arxiv.org/pdf/2510.21783v1)**

> **作者:** Guo Li; Yuyang Yu; Xuemiao Xu
>
> **摘要:** Diffusion models have demonstrated powerful performance in generating high-quality images. A typical example is text-to-image generator like Stable Diffusion. However, their widespread use also poses potential privacy risks. A key concern is membership inference attacks, which attempt to determine whether a particular data sample was used in the model training process. We propose an efficient membership inference attack method against diffusion models. This method is based on the injection of slight noise and the evaluation of the aggregation degree of the noise distribution. The intuition is that the noise prediction patterns of diffusion models for training set samples and non-training set samples exhibit distinguishable differences.Specifically, we suppose that member images exhibit higher aggregation of predicted noise around a certain time step of the diffusion process. In contrast, the predicted noises of non-member images exhibit a more discrete characteristic around the certain time step. Compared with other existing methods, our proposed method requires fewer visits to the target diffusion model. We inject slight noise into the image under test and then determine its membership by analyzing the aggregation degree of the noise distribution predicted by the model. Empirical findings indicate that our method achieves superior performance across multiple datasets. At the same time, our method can also show better attack effects in ASR and AUC when facing large-scale text-to-image diffusion models, proving the scalability of our method.
>
---
#### [new 133] Towards Generalisable Foundation Models for 3D Brain MRI
- **分类: cs.CV**

- **简介: 该论文提出BrainFound，一种用于3D脑部MRI的自监督基础模型，旨在解决医学影像中标注数据稀缺与多模态兼容性差的问题。通过扩展DINO-v2以处理三维体积数据，支持多模态输入，提升疾病检测与分割性能，显著减少对专家标注的依赖，适用于多种临床场景。**

- **链接: [http://arxiv.org/pdf/2510.23415v1](http://arxiv.org/pdf/2510.23415v1)**

> **作者:** Moona Mazher; Geoff J. M. Parker; Daniel C. Alexander
>
> **摘要:** Foundation models in artificial intelligence (AI) are transforming medical imaging by enabling general-purpose feature learning from large-scale, unlabeled datasets. In this work, we introduce BrainFound, a self-supervised foundation model for brain MRI, built by extending DINO-v2, a vision transformer originally designed for 2D natural images. BrainFound adapts DINO-v2 to model full 3D brain anatomy by incorporating volumetric information from sequential MRI slices, moving beyond conventional single-slice paradigms. It supports both single- and multimodal inputs, enabling a broad range of downstream tasks, including disease detection and image segmentation, while generalising across varied imaging protocols and clinical scenarios. We show that BrainFound consistently outperforms existing self-supervised pretraining strategies and supervised baselines, particularly in label-scarce and multi-contrast settings. By integrating information from diverse 3D MRI modalities (e.g., T1, T2, FLAIR), it enhances diagnostic accuracy and reduces dependency on extensive expert annotations. This flexibility makes BrainFound a scalable and practical solution for 3D neuroimaging pipelines, with significant potential for clinical deployment and research innovation.
>
---
#### [new 134] Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦多模态抽象推理任务，针对当前模型在处理多模态关系知识（MMRK）时的不足，提出自动数据生成引擎与两阶段增强框架，构建了STAR-64K数据集。实验表明，小模型经训练后可超越GPT-4o，显著提升结构化抽象推理能力。**

- **链接: [http://arxiv.org/pdf/2510.21828v1](http://arxiv.org/pdf/2510.21828v1)**

> **作者:** Yichi Zhang; Zhuo Chen; Lingbing Guo; Lei Liang; Wen Zhang; Huajun Chen
>
> **备注:** Work in Progress. Code and data will be released at https://github.com/zjukg/STAR
>
> **摘要:** Understanding and reasoning with abstractive information from the visual modality presents significant challenges for current multi-modal large language models (MLLMs). Among the various forms of abstractive information, Multi-Modal Relational Knowledge (MMRK), which represents abstract relational structures between multi-modal entities using node-edge formats, remains largely under-explored. In particular, STructured and Abstractive Reasoning (STAR) on such data has received little attention from the research community. To bridge the dual gaps in large-scale high-quality data and capability enhancement methodologies, this paper makes the following key contributions: (i). An automatic STAR data engine capable of synthesizing images with MMRK to build multi-modal instruction data with reliable chain-of-thought thinking for various STAR tasks and (ii). A comprehsive two-stage capability enhancement training framework, accompanied by a suite of evaluation protocols tailored to different STAR tasks. Based upon these contributions, we introduce STAR-64K, a dataset comprising 64K high-quality multi-modal instruction samples, and conduct experiments across 5 open-source MLLMs. Experimental results show that our two-stage enhancement framework enables smaller 3B/7B models to significantly outperform GPT-4o in STAR. Additionally, we provide in-depth analysis regarding the effectiveness of various designs, data transferability, and scalability.
>
---
#### [new 135] VideoTG-R1: Boosting Video Temporal Grounding via Curriculum Reinforcement Learning on Reflected Boundary Annotations
- **分类: cs.CV**

- **简介: 该论文针对视频时间定位（VTG）任务，解决标注不全和样本难度差异导致的训练效率低问题。提出VideoTG-R1框架，通过边界反射代理识别不完整标注，利用难度估计代理设计课程强化学习策略，动态屏蔽难样本，实现高效训练。**

- **链接: [http://arxiv.org/pdf/2510.23397v1](http://arxiv.org/pdf/2510.23397v1)**

> **作者:** Lu Dong; Haiyu Zhang; Han Lin; Ziang Yan; Xiangyu Zeng; Hongjie Zhang; Yifei Huang; Yi Wang; Zhen-Hua Ling; Limin Wang; Yali Wang
>
> **摘要:** Video temporal grounding (VTG) aims to locate precise segments in videos based on language queries, which is a fundamental challenge in video understanding. While recent Multimodal Large Language Models (MLLMs) have shown promise in tackling VTG through reinforcement learning (RL), they overlook the challenges arising from both the quality and difficulty of training samples. (1) Partially annotated samples. Many samples contain relevant segments beyond the annotated interval, introducing ambiguous supervision. (2) Hard-to-ground samples. Samples with poor zero-shot performance produce consistently low and indistinguishable rewards during RL training, exhibiting no clear preference among multiple outputs and thus hindering learning efficiency. To address these challenges, we propose VideoTG-R1, a novel curriculum RL framework with reflected boundary annotations, enabling data-efficient training. Specifically, we propose a Boundary Reflection Agent that utilizes MLLMs to predict query-relevant timestamps outside the annotated intervals, allowing us to identify and filter out partially annotated samples, thereby reducing ambiguity. Furthermore, we introduce a Difficulty Estimation Agent to assess the training difficulty of each sample and design a curriculum RL strategy that dynamically masks the videos of hard-to-ground samples according to the training steps, easing the training difficulty and providing clearer preference. Experiments on the VTG and grounded VideoQA tasks demonstrate the effectiveness of our method. Remarkably, with only 10% of the training samples and 21% of the computational budget, VideoTG-R1 outperforms full-data counterparts under both group relative policy optimization (GRPO) and supervised fine-tuning (SFT). The code is available at https://github.com/ldong1111/VideoTG-R1.
>
---
#### [new 136] Mitigating Coordinate Prediction Bias from Positional Encoding Failures
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对多模态大模型在高分辨率输入下坐标预测偏差问题，发现位置编码失效导致定向误差。提出VPSG方法，通过扰动位置编码生成负向证据，在不训练的前提下修正坐标预测，提升空间推理准确性。**

- **链接: [http://arxiv.org/pdf/2510.22102v1](http://arxiv.org/pdf/2510.22102v1)**

> **作者:** Xingjian Tao; Yiwei Wang; Yujun Cai; Yihong Luo; Jing Tang
>
> **摘要:** Multimodal large language models (MLLMs) excel at vision-language tasks such as VQA and document understanding, yet precise coordinate prediction remains challenging. High-resolution inputs exacerbate this difficulty by producing long token sequences that weaken positional encodings and introduce directional biases in coordinate outputs. We investigate this phenomenon by analyzing how MLLMs behave when visual positional encodings (VPEs) are deliberately perturbed through shuffling. Our analysis reveals that such perturbations induce predictable, non-random coordinate biases rather than random errors, suggesting that models rely on internal positional priors when spatial grounding signals are degraded. Crucially, we observe similar directional error patterns in natural high-resolution datasets, indicating that positional encoding failures are a key bottleneck for accurate coordinate prediction at scale. To address this issue, we propose Vision-PE Shuffle Guidance (VPSG), a training-free test-time method that leverages the directional nature of these biases for correction. VPSG runs auxiliary decoding with shuffled VPEs to isolate position-unconditioned tendencies, then uses this as negative evidence to guide digit prediction while preserving coordinate format through a lightweight finite-state machine. Experiments on ScreenSpot-Pro demonstrate reliable improvements, highlighting positional encoding robustness as a critical factor for spatial reasoning in MLLMs.
>
---
#### [new 137] LRW-Persian: Lip-reading in the Wild Dataset for Persian Language
- **分类: cs.CV**

- **简介: 该论文提出LRW-Persian，首个大规模野生场景波斯语唇读数据集，解决低资源语言唇读数据稀缺问题。包含743词、41.4万视频样本，支持训练与测试分离，具备多地区方言覆盖与丰富元数据。构建自动化数据清洗流程，并在该数据集上验证了主流唇读模型性能，推动多模态语音识别在非英语语境下的研究。**

- **链接: [http://arxiv.org/pdf/2510.22716v1](http://arxiv.org/pdf/2510.22716v1)**

> **作者:** Zahra Taghizadeh; Mohammad Shahverdikondori; Arian Noori; Alireza Dadgarnia
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Lipreading has emerged as an increasingly important research area for developing robust speech recognition systems and assistive technologies for the hearing-impaired. However, non-English resources for visual speech recognition remain limited. We introduce LRW-Persian, the largest in-the-wild Persian word-level lipreading dataset, comprising $743$ target words and over $414{,}000$ video samples extracted from more than $1{,}900$ hours of footage across $67$ television programs. Designed as a benchmark-ready resource, LRW-Persian provides speaker-disjoint training and test splits, wide regional and dialectal coverage, and rich per-clip metadata including head pose, age, and gender. To ensure large-scale data quality, we establish a fully automated end-to-end curation pipeline encompassing transcription based on Automatic Speech Recognition(ASR), active-speaker localization, quality filtering, and pose/mask screening. We further fine-tune two widely used lipreading architectures on LRW-Persian, establishing reference performance and demonstrating the difficulty of Persian visual speech recognition. By filling a critical gap in low-resource languages, LRW-Persian enables rigorous benchmarking, supports cross-lingual transfer, and provides a foundation for advancing multimodal speech research in underrepresented linguistic contexts. The dataset is publicly available at: https://lrw-persian.vercel.app.
>
---
#### [new 138] ConMatFormer: A Multi-attention and Transformer Integrated ConvNext based Deep Learning Model for Enhanced Diabetic Foot Ulcer Classification
- **分类: cs.CV**

- **简介: 该论文针对糖尿病足溃疡（DFU）分类任务，解决数据稀缺与类别不平衡问题。提出ConMatFormer模型，融合ConvNeXt、多注意力机制与Transformer，有效提取局部特征与全局上下文，提升分类精度与鲁棒性。实验表明其性能优于SOTA方法，并通过XAI技术增强模型可解释性。**

- **链接: [http://arxiv.org/pdf/2510.22743v1](http://arxiv.org/pdf/2510.22743v1)**

> **作者:** Raihan Ahamed Rifat; Fuyad Hasan Bhoyan; Md Humaion Kabir Mehedi; Md Kaviul Hossain; Md. Jakir Hossen; M. F. Mridha
>
> **摘要:** Diabetic foot ulcer (DFU) detection is a clinically significant yet challenging task due to the scarcity and variability of publicly available datasets. To solve these problems, we propose ConMatFormer, a new hybrid deep learning architecture that combines ConvNeXt blocks, multiple attention mechanisms convolutional block attention module (CBAM) and dual attention network (DANet), and transformer modules in a way that works together. This design facilitates the extraction of better local features and understanding of the global context, which allows us to model small skin patterns across different types of DFU very accurately. To address the class imbalance, we used data augmentation methods. A ConvNeXt block was used to obtain detailed local features in the initial stages. Subsequently, we compiled the model by adding a transformer module to enhance long-range dependency. This enabled us to pinpoint the DFU classes that were underrepresented or constituted minorities. Tests on the DS1 (DFUC2021) and DS2 (diabetic foot ulcer (DFU)) datasets showed that ConMatFormer outperformed state-of-the-art (SOTA) convolutional neural network (CNN) and Vision Transformer (ViT) models in terms of accuracy, reliability, and flexibility. The proposed method achieved an accuracy of 0.8961 and a precision of 0.9160 in a single experiment, which is a significant improvement over the current standards for classifying DFUs. In addition, by 4-fold cross-validation, the proposed model achieved an accuracy of 0.9755 with a standard deviation of only 0.0031. We further applied explainable artificial intelligence (XAI) methods, such as Grad-CAM, Grad-CAM++, and LIME, to consistently monitor the transparency and trustworthiness of the decision-making process.. Our findings set a new benchmark for DFU classification and provide a hybrid attention transformer framework for medical image analysis.
>
---
#### [new 139] Addressing Corner Cases in Autonomous Driving: A World Model-based Approach with Mixture of Experts and LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶中罕见高风险场景（corner cases）的运动预测难题，提出基于世界模型的WM-MoE框架。通过融合感知、时序记忆与决策，结合轻量级时序编码器与大语言模型增强推理，并引入混合专家机制实现场景自适应建模，显著提升复杂场景下的预测准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.21867v1](http://arxiv.org/pdf/2510.21867v1)**

> **作者:** Haicheng Liao; Bonan Wang; Junxian Yang; Chengyue Wang; Zhengbin He; Guohui Zhang; Chengzhong Xu; Zhenning Li
>
> **摘要:** Accurate and reliable motion forecasting is essential for the safe deployment of autonomous vehicles (AVs), particularly in rare but safety-critical scenarios known as corner cases. Existing models often underperform in these situations due to an over-representation of common scenes in training data and limited generalization capabilities. To address this limitation, we present WM-MoE, the first world model-based motion forecasting framework that unifies perception, temporal memory, and decision making to address the challenges of high-risk corner-case scenarios. The model constructs a compact scene representation that explains current observations, anticipates future dynamics, and evaluates the outcomes of potential actions. To enhance long-horizon reasoning, we leverage large language models (LLMs) and introduce a lightweight temporal tokenizer that maps agent trajectories and contextual cues into the LLM's feature space without additional training, enriching temporal context and commonsense priors. Furthermore, a mixture-of-experts (MoE) is introduced to decompose complex corner cases into subproblems and allocate capacity across scenario types, and a router assigns scenes to specialized experts that infer agent intent and perform counterfactual rollouts. In addition, we introduce nuScenes-corner, a new benchmark that comprises four real-world corner-case scenarios for rigorous evaluation. Extensive experiments on four benchmark datasets (nuScenes, NGSIM, HighD, and MoCAD) showcase that WM-MoE consistently outperforms state-of-the-art (SOTA) baselines and remains robust under corner-case and data-missing conditions, indicating the promise of world model-based architectures for robust and generalizable motion forecasting in fully AVs.
>
---
#### [new 140] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出egoEMOTION数据集，融合第一人称视觉与生理信号，用于真实场景下情绪与人格识别。针对现有研究忽略情感与人格的问题，构建多模态数据集，定义连续情绪、离散情绪与人格特质三类任务，验证了视觉信号在情绪预测中的有效性，推动了情感驱动的行为建模发展。**

- **链接: [http://arxiv.org/pdf/2510.22129v1](http://arxiv.org/pdf/2510.22129v1)**

> **作者:** Matthias Jammot; Bjöern Braun; Paul Streli; Rafael Wampfler; Christian Holz
>
> **备注:** Accepted for publication at NeurIPS 2025
>
> **摘要:** Understanding affect is central to anticipating human behavior, yet current egocentric vision benchmarks largely ignore the person's emotional states that shape their decisions and actions. Existing tasks in egocentric perception focus on physical activities, hand-object interactions, and attention modeling - assuming neutral affect and uniform personality. This limits the ability of vision systems to capture key internal drivers of behavior. In this paper, we present egoEMOTION, the first dataset that couples egocentric visual and physiological signals with dense self-reports of emotion and personality across controlled and real-world scenarios. Our dataset includes over 50 hours of recordings from 43 participants, captured using Meta's Project Aria glasses. Each session provides synchronized eye-tracking video, headmounted photoplethysmography, inertial motion data, and physiological baselines for reference. Participants completed emotion-elicitation tasks and naturalistic activities while self-reporting their affective state using the Circumplex Model and Mikels' Wheel as well as their personality via the Big Five model. We define three benchmark tasks: (1) continuous affect classification (valence, arousal, dominance); (2) discrete emotion classification; and (3) trait-level personality inference. We show that a classical learning-based method, as a simple baseline in real-world affect prediction, produces better estimates from signals captured on egocentric vision systems than processing physiological signals. Our dataset establishes emotion and personality as core dimensions in egocentric perception and opens new directions in affect-driven modeling of behavior, intent, and interaction.
>
---
#### [new 141] Open Multimodal Retrieval-Augmented Factual Image Generation
- **分类: cs.CV; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出ORIG框架，解决大模型生成图像时事实不一致的问题。针对“事实性图像生成”（FIG）任务，通过迭代检索网络多模态证据并动态更新提示，实现知识精准融合。构建FIG-Eval基准评估多维事实一致性，实验表明其显著提升生成图像的事实准确性和质量。**

- **链接: [http://arxiv.org/pdf/2510.22521v1](http://arxiv.org/pdf/2510.22521v1)**

> **作者:** Yang Tian; Fan Liu; Jingyuan Zhang; Wei Bi; Yupeng Hu; Liqiang Nie
>
> **备注:** Preprint
>
> **摘要:** Large Multimodal Models (LMMs) have achieved remarkable progress in generating photorealistic and prompt-aligned images, but they often produce outputs that contradict verifiable knowledge, especially when prompts involve fine-grained attributes or time-sensitive events. Conventional retrieval-augmented approaches attempt to address this issue by introducing external information, yet they are fundamentally incapable of grounding generation in accurate and evolving knowledge due to their reliance on static sources and shallow evidence integration. To bridge this gap, we introduce ORIG, an agentic open multimodal retrieval-augmented framework for Factual Image Generation (FIG), a new task that requires both visual realism and factual grounding. ORIG iteratively retrieves and filters multimodal evidence from the web and incrementally integrates the refined knowledge into enriched prompts to guide generation. To support systematic evaluation, we build FIG-Eval, a benchmark spanning ten categories across perceptual, compositional, and temporal dimensions. Experiments demonstrate that ORIG substantially improves factual consistency and overall image quality over strong baselines, highlighting the potential of open multimodal retrieval for factual image generation.
>
---
#### [new 142] Color and Frequency Correction for Image Colorization
- **分类: cs.CV**

- **简介: 该论文属于图像着色任务，针对DDColor模型在高频带表现不佳及因输入维度不足导致的颜色偏移问题，提出两种优化方案并融合应用，有效提升了图像的PSNR与SSIM指标。**

- **链接: [http://arxiv.org/pdf/2510.23399v1](http://arxiv.org/pdf/2510.23399v1)**

> **作者:** Yun Kai Zhuang
>
> **备注:** 7 pages, 5 tables
>
> **摘要:** The project has carried out the re-optimization of image coloring in accordance with the existing Autocolorization direction model DDColor. For the experiments on the existing weights of DDColor, we found that it has limitations in some frequency bands and the color cast problem caused by insufficient input dimension. We construct two optimization schemes and combine them, which achieves the performance improvement of indicators such as PSNR and SSIM of the images after DDColor.
>
---
#### [new 143] MOGRAS: Human Motion with Grasping in 3D Scenes
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文聚焦于3D场景中人体与物体交互的运动生成任务。针对现有方法在场景感知与精细抓握间缺失的问题，提出MOGRAS数据集，包含带标注的完整人体抓握动作与场景信息，并设计方法提升现有模型在3D场景中的生成能力，显著改善了真实感与物理合理性。**

- **链接: [http://arxiv.org/pdf/2510.22199v1](http://arxiv.org/pdf/2510.22199v1)**

> **作者:** Kunal Bhosikar; Siddharth Katageri; Vivek Madhavaram; Kai Han; Charu Sharma
>
> **备注:** British Machine Vision Conference Workshop - From Scene Understanding to Human Modeling
>
> **摘要:** Generating realistic full-body motion interacting with objects is critical for applications in robotics, virtual reality, and human-computer interaction. While existing methods can generate full-body motion within 3D scenes, they often lack the fidelity for fine-grained tasks like object grasping. Conversely, methods that generate precise grasping motions typically ignore the surrounding 3D scene. This gap, generating full-body grasping motions that are physically plausible within a 3D scene, remains a significant challenge. To address this, we introduce MOGRAS (Human MOtion with GRAsping in 3D Scenes), a large-scale dataset that bridges this gap. MOGRAS provides pre-grasping full-body walking motions and final grasping poses within richly annotated 3D indoor scenes. We leverage MOGRAS to benchmark existing full-body grasping methods and demonstrate their limitations in scene-aware generation. Furthermore, we propose a simple yet effective method to adapt existing approaches to work seamlessly within 3D scenes. Through extensive quantitative and qualitative experiments, we validate the effectiveness of our dataset and highlight the significant improvements our proposed method achieves, paving the way for more realistic human-scene interactions.
>
---
#### [new 144] Task-Agnostic Fusion of Time Series and Imagery for Earth Observation
- **分类: cs.CV**

- **简介: 该论文提出一种任务无关的多模态融合框架，用于融合时间序列与单时相影像，解决跨模态生成与下游任务性能提升问题。通过量化时间序列并利用掩码相关性学习，实现图像与时间序列在统一空间对齐，显著优于现有方法，在地球观测中验证了其有效性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.23118v1](http://arxiv.org/pdf/2510.23118v1)**

> **作者:** Gianfranco Basile; Johannes Jakubik; Benedikt Blumenstiel; Thomas Brunschwiler; Juan Bernabe Moreno
>
> **摘要:** We propose a task-agnostic framework for multimodal fusion of time series and single timestamp images, enabling cross-modal generation and robust downstream performance. Our approach explores deterministic and learned strategies for time series quantization and then leverages a masked correlation learning objective, aligning discrete image and time series tokens in a unified representation space. Instantiated in the Earth observation domain, the pretrained model generates consistent global temperature profiles from satellite imagery and is validated through counterfactual experiments. Across downstream tasks, our task-agnostic pretraining outperforms task-specific fusion by 6\% in R$^2$ and 2\% in RMSE on average, and exceeds baseline methods by 50\% in R$^2$ and 12\% in RMSE. Finally, we analyze gradient sensitivity across modalities, providing insights into model robustness. Code, data, and weights will be released under a permissive license.
>
---
#### [new 145] Frame-Difference Guided Dynamic Region Perception for CLIP Adaptation in Text-Video Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本-视频检索任务，解决现有CLIP模型在视频适配中对动态特征捕捉不足、静态冗余抑制弱的问题。提出FDA-CLIP框架，利用帧间差异生成动态区域掩码，作为Alpha通道引导模型关注关键动态区域，提升跨模态对齐精度与检索效率。**

- **链接: [http://arxiv.org/pdf/2510.21806v1](http://arxiv.org/pdf/2510.21806v1)**

> **作者:** Jiaao Yu; Mingjie Han; Tao Gong; Jian Zhang; Man Lan
>
> **备注:** 5 pages
>
> **摘要:** With the rapid growth of video data, text-video retrieval technology has become increasingly important in numerous application scenarios such as recommendation and search. Early text-video retrieval methods suffer from two critical drawbacks: first, they heavily rely on large-scale annotated video-text pairs, leading to high data acquisition costs; second, there is a significant modal gap between video and text features, which limits cross-modal alignment accuracy. With the development of vision-language model, adapting CLIP to video tasks has attracted great attention. However, existing adaptation methods generally lack enhancement for dynamic video features and fail to effectively suppress static redundant features. To address this issue, this paper proposes FDA-CLIP (Frame Difference Alpha-CLIP), which is a concise CLIP-based training framework for text-video alignment. Specifically, the method uses frame differences to generate dynamic region masks, which are input into Alpha-CLIP as an additional Alpha channel. This proactively guides the model to focus on semantically critical dynamic regions while suppressing static background redundancy. Experiments demonstrate that frame difference-guided video semantic encoding can effectively balance retrieval efficiency and accuracy.
>
---
#### [new 146] GSAlign: Geometric and Semantic Alignment Network for Aerial-Ground Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对航拍与地面视角下行人重识别（AG-ReID）任务，解决因视角差异大、姿态变化剧烈和遮挡导致的匹配难题。提出GSAlign网络，通过可学习薄板样条模块补偿几何畸变，动态对齐模块生成可见性感知语义掩码，提升跨视角匹配精度，在CARGO数据集上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.22268v1](http://arxiv.org/pdf/2510.22268v1)**

> **作者:** Qiao Li; Jie Li; Yukang Zhang; Lei Tan; Jing Chen; Jiayi Ji
>
> **备注:** Accepted by Neurips 2025
>
> **摘要:** Aerial-Ground person re-identification (AG-ReID) is an emerging yet challenging task that aims to match pedestrian images captured from drastically different viewpoints, typically from unmanned aerial vehicles (UAVs) and ground-based surveillance cameras. The task poses significant challenges due to extreme viewpoint discrepancies, occlusions, and domain gaps between aerial and ground imagery. While prior works have made progress by learning cross-view representations, they remain limited in handling severe pose variations and spatial misalignment. To address these issues, we propose a Geometric and Semantic Alignment Network (GSAlign) tailored for AG-ReID. GSAlign introduces two key components to jointly tackle geometric distortion and semantic misalignment in aerial-ground matching: a Learnable Thin Plate Spline (LTPS) Module and a Dynamic Alignment Module (DAM). The LTPS module adaptively warps pedestrian features based on a set of learned keypoints, effectively compensating for geometric variations caused by extreme viewpoint changes. In parallel, the DAM estimates visibility-aware representation masks that highlight visible body regions at the semantic level, thereby alleviating the negative impact of occlusions and partial observations in cross-view correspondence. A comprehensive evaluation on CARGO with four matching protocols demonstrates the effectiveness of GSAlign, achieving significant improvements of +18.8\% in mAP and +16.8\% in Rank-1 accuracy over previous state-of-the-art methods on the aerial-ground setting. The code is available at: \textcolor{magenta}{https://github.com/stone96123/GSAlign}.
>
---
#### [new 147] Diagnosing Bottlenecks in Data Visualization Understanding by Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦数据可视化理解任务，旨在诊断视觉语言模型（VLMs）在处理数据图时失败的原因。通过构建FUGU基准，结合激活修补与线性探测，发现错误多源于视觉-语言模块间的信息传递，而非编码或语言处理本身。即使提供正确坐标，复杂统计任务性能仍下降，表明现有VLM架构存在根本局限。**

- **链接: [http://arxiv.org/pdf/2510.21740v1](http://arxiv.org/pdf/2510.21740v1)**

> **作者:** Alexa R. Tartaglini; Satchel Grant; Daniel Wurgaft; Christopher Potts; Judith E. Fan
>
> **摘要:** Data visualizations are vital components of many scientific articles and news stories. Current vision-language models (VLMs) still struggle on basic data visualization understanding tasks, but the causes of failure remain unclear. Are VLM failures attributable to limitations in how visual information in the data visualization is encoded, how information is transferred between the vision and language modules, or how information is processed within the language module? We developed FUGU, a suite of data visualization understanding tasks, to precisely characterize potential sources of difficulty (e.g., extracting the position of data points, distances between them, and other summary statistics). We used FUGU to investigate three widely used VLMs. To diagnose the sources of errors produced by these models, we used activation patching and linear probes to trace information flow through models across a variety of prompting strategies. We found that some models fail to generate the coordinates of individual data points correctly, and these initial errors often lead to erroneous final responses. When these models are provided with the correct coordinates, performance improves substantially. Moreover, even when the model generates an incorrect response, the correct coordinates can be successfully read out from the latent representations in the vision encoder, suggesting that the source of these errors lies in the vision-language handoff. We further found that while providing correct coordinates helps with tasks involving one or a small number of data points, it generally worsens performance for tasks that require extracting statistical relationships across many data points. Fine-tuning models on FUGU also fails to yield ceiling performance. These findings point to architectural constraints in current VLMs that might pose significant challenges for reliable data visualization understanding.
>
---
#### [new 148] PRISM-Bench: A Benchmark of Puzzle-Based Visual Tasks with CoT Error Detection
- **分类: cs.CV**

- **简介: 该论文提出PRISM-Bench，一个基于谜题的多模态推理基准，旨在评估模型在复杂视觉任务中的逻辑一致性与错误检测能力。针对现有方法仅关注答案正确性的问题，该工作引入“识别链式思维中首个错误步骤”的诊断任务，揭示大模型在生成流畅推理过程时存在的推理不忠实问题，推动更可信的多模态大模型发展。**

- **链接: [http://arxiv.org/pdf/2510.23594v1](http://arxiv.org/pdf/2510.23594v1)**

> **作者:** Yusu Qian; Cheng Wan; Chao Jia; Yinfei Yang; Qingyu Zhao; Zhe Gan
>
> **摘要:** We introduce \textbf{PRISM-Bench}, a benchmark of puzzle-based visual challenges designed to evaluate not only whether models can solve problems, but how their reasoning unfolds. Unlike prior evaluations that measure only final-answer accuracy, PRISM-Bench introduces a diagnostic task: given a visual puzzle and a step-by-step chain-of-thought (CoT) containing exactly one error, models must identify the first incorrect step. This setting enables fine-grained assessment of logical consistency, error detection, and visual reasoning. The puzzles in PRISM-Bench require multi-step symbolic, geometric, and analogical reasoning, resisting shortcuts based on superficial pattern matching. Evaluations across state-of-the-art MLLMs reveal a persistent gap between fluent generation and faithful reasoning: models that produce plausible CoTs often fail to locate simple logical faults. By disentangling answer generation from reasoning verification, PRISM-Bench offers a sharper lens on multimodal reasoning competence and underscores the need for diagnostic evaluation protocols in the development of trustworthy MLLMs.
>
---
#### [new 149] Progressive Growing of Patch Size: Curriculum Learning for Accelerated and Improved Medical Image Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对3D医学图像分割任务，提出渐进式扩大图像块尺寸的自动课程学习方法。通过逐步增加训练时的图像块大小，改善小块下的类别平衡，加速收敛。实验表明，该方法在保持或提升分割性能的同时，显著减少训练时间，尤其适用于类别极度不平衡的任务，并可适配多种网络架构。**

- **链接: [http://arxiv.org/pdf/2510.23241v1](http://arxiv.org/pdf/2510.23241v1)**

> **作者:** Stefan M. Fischer; Johannes Kiechle; Laura Daza; Lina Felsner; Richard Osuala; Daniel M. Lang; Karim Lekadir; Jan C. Peeken; Julia A. Schnabel
>
> **备注:** Journal Extension of "Progressive Growing of Patch Size: Resource-Efficient Curriculum Learning for Dense Prediction Tasks" (MICCAI2024) submitted to MedIA
>
> **摘要:** In this work, we introduce Progressive Growing of Patch Size, an automatic curriculum learning approach for 3D medical image segmentation. Our approach progressively increases the patch size during model training, resulting in an improved class balance for smaller patch sizes and accelerated convergence of the training process. We evaluate our curriculum approach in two settings: a resource-efficient mode and a performance mode, both regarding Dice score performance and computational costs across 15 diverse and popular 3D medical image segmentation tasks. The resource-efficient mode matches the Dice score performance of the conventional constant patch size sampling baseline with a notable reduction in training time to only 44%. The performance mode improves upon constant patch size segmentation results, achieving a statistically significant relative mean performance gain of 1.28% in Dice Score. Remarkably, across all 15 tasks, our proposed performance mode manages to surpass the constant patch size baseline in Dice Score performance, while simultaneously reducing training time to only 89%. The benefits are particularly pronounced for highly imbalanced tasks such as lesion segmentation tasks. Rigorous experiments demonstrate that our performance mode not only improves mean segmentation performance but also reduces performance variance, yielding more trustworthy model comparison. Furthermore, our findings reveal that the proposed curriculum sampling is not tied to a specific architecture but represents a broadly applicable strategy that consistently boosts performance across diverse segmentation models, including UNet, UNETR, and SwinUNETR. In summary, we show that this simple yet elegant transformation on input data substantially improves both Dice Score performance and training runtime, while being compatible across diverse segmentation backbones.
>
---
#### [new 150] WaveMAE: Wavelet decomposition Masked Auto-Encoder for Remote Sensing
- **分类: cs.CV; 68T07; I.2.6; I.4.10; J.2**

- **简介: 该论文提出WaveMAE，一种用于多光谱遥感影像的自监督学习框架。针对标注数据稀缺问题，利用小波分解与地理先验编码，提升模型对高频特征和空间结构的感知能力。在PANGAEA基准上验证，显著优于现有方法，且轻量版本亦达SOTA。**

- **链接: [http://arxiv.org/pdf/2510.22697v1](http://arxiv.org/pdf/2510.22697v1)**

> **作者:** Vittorio Bernuzzi; Leonardo Rossi; Tomaso Fontanini; Massimo Bertozzi; Andrea Prati
>
> **摘要:** Self-supervised learning (SSL) has recently emerged as a key strategy for building foundation models in remote sensing, where the scarcity of annotated data limits the applicability of fully supervised approaches. In this work, we introduce WaveMAE, a masked autoencoding framework tailored for multispectral satellite imagery. Unlike conventional pixel-based reconstruction, WaveMAE leverages a multi-level Discrete Wavelet Transform (DWT) to disentangle frequency components and guide the encoder toward learning scale-aware high-frequency representations. We further propose a Geo-conditioned Positional Encoding (GPE), which incorporates geographical priors via Spherical Harmonics, encouraging embeddings that respect both semantic and geospatial structure. To ensure fairness in evaluation, all methods are pretrained on the same dataset (fMoW-S2) and systematically evaluated on the diverse downstream tasks of the PANGAEA benchmark, spanning semantic segmentation, regression, change detection, and multilabel classification. Extensive experiments demonstrate that WaveMAE achieves consistent improvements over prior state-of-the-art approaches, with substantial gains on segmentation and regression benchmarks. The effectiveness of WaveMAE pretraining is further demonstrated by showing that even a lightweight variant, containing only 26.4% of the parameters, achieves state-of-the-art performance. Our results establish WaveMAE as a strong and geographically informed foundation model for multispectral remote sensing imagery.
>
---
#### [new 151] Projection Embedded Diffusion Bridge for CT Reconstruction from Incomplete Data
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文针对CT图像从不完整投影数据中重建的任务，解决传统方法在数据不完整时重建质量差的问题。提出投影嵌入扩散桥（PEDB）模型，通过在逆向随机微分方程中显式嵌入投影数据，实现数据一致性约束，提升重建精度与细节恢复能力。**

- **链接: [http://arxiv.org/pdf/2510.22605v1](http://arxiv.org/pdf/2510.22605v1)**

> **作者:** Yuang Wang; Pengfei Jin; Siyeop Yoon; Matthew Tivnan; Shaoyang Zhang; Li Zhang; Quanzheng Li; Zhiqiang Chen; Dufan Wu
>
> **备注:** 53 pages, 7 figures, submitted to Medical Image Analysis
>
> **摘要:** Reconstructing CT images from incomplete projection data remains challenging due to the ill-posed nature of the problem. Diffusion bridge models have recently shown promise in restoring clean images from their corresponding Filtered Back Projection (FBP) reconstructions, but incorporating data consistency into these models remains largely underexplored. Incorporating data consistency can improve reconstruction fidelity by aligning the reconstructed image with the observed projection data, and can enhance detail recovery by integrating structural information contained in the projections. In this work, we propose the Projection Embedded Diffusion Bridge (PEDB). PEDB introduces a novel reverse stochastic differential equation (SDE) to sample from the distribution of clean images conditioned on both the FBP reconstruction and the incomplete projection data. By explicitly conditioning on the projection data in sampling the clean images, PEDB naturally incorporates data consistency. We embed the projection data into the score function of the reverse SDE. Under certain assumptions, we derive a tractable expression for the posterior score. In addition, we introduce a free parameter to control the level of stochasticity in the reverse process. We also design a discretization scheme for the reverse SDE to mitigate discretization error. Extensive experiments demonstrate that PEDB achieves strong performance in CT reconstruction from three types of incomplete data, including sparse-view, limited-angle, and truncated projections. For each of these types, PEDB outperforms evaluated state-of-the-art diffusion bridge models across standard, noisy, and domain-shift evaluations.
>
---
#### [new 152] Hybrid Deep Learning Framework for Enhanced Diabetic Retinopathy Detection: Integrating Traditional Features with AI-driven Insights
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对糖尿病视网膜病变（DR）早期检测难题，提出一种融合传统特征与深度学习的混合框架。通过结合可解释的临床特征与自动提取的深层特征，提升检测准确率，减少漏诊，实现高效、精准的DR筛查，适用于高糖尿病负担地区。**

- **链接: [http://arxiv.org/pdf/2510.21810v1](http://arxiv.org/pdf/2510.21810v1)**

> **作者:** Arpan Maity; Aviroop Pal; MD. Samiul Islam; Tamal Ghosh
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Diabetic Retinopathy (DR), a vision-threatening complication of Dia-betes Mellitus (DM), is a major global concern, particularly in India, which has one of the highest diabetic populations. Prolonged hyperglycemia damages reti-nal microvasculature, leading to DR symptoms like microaneurysms, hemor-rhages, and fluid leakage, which, if undetected, cause irreversible vision loss. Therefore, early screening is crucial as DR is asymptomatic in its initial stages. Fundus imaging aids precise diagnosis by detecting subtle retinal lesions. This paper introduces a hybrid diagnostic framework combining traditional feature extraction and deep learning (DL) to enhance DR detection. While handcrafted features capture key clinical markers, DL automates hierarchical pattern recog-nition, improving early diagnosis. The model synergizes interpretable clinical data with learned features, surpassing standalone DL approaches that demon-strate superior classification and reduce false negatives. This multimodal AI-driven approach enables scalable, accurate DR screening, crucial for diabetes-burdened regions.
>
---
#### [new 153] CityRiSE: Reasoning Urban Socio-Economic Status in Vision-Language Models via Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出CityRiSE框架，利用强化学习引导视觉语言模型推理城市社会经济状况。针对LVLM在视觉数据下预测不准、不可解释的问题，通过设计可验证奖励机制，促使模型聚焦语义线索，实现结构化、目标导向的推理，显著提升跨城市、跨指标的预测准确率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.22282v1](http://arxiv.org/pdf/2510.22282v1)**

> **作者:** Tianhui Liu; Hetian Pang; Xin Zhang; Jie Feng; Yong Li; Pan Hui
>
> **摘要:** Harnessing publicly available, large-scale web data, such as street view and satellite imagery, urban socio-economic sensing is of paramount importance for achieving global sustainable development goals. With the emergence of Large Vision-Language Models (LVLMs), new opportunities have arisen to solve this task by treating it as a multi-modal perception and understanding problem. However, recent studies reveal that LVLMs still struggle with accurate and interpretable socio-economic predictions from visual data. To address these limitations and maximize the potential of LVLMs, we introduce \textbf{CityRiSE}, a novel framework for \textbf{R}eason\textbf{i}ng urban \textbf{S}ocio-\textbf{E}conomic status in LVLMs through pure reinforcement learning (RL). With carefully curated multi-modal data and verifiable reward design, our approach guides the LVLM to focus on semantically meaningful visual cues, enabling structured and goal-oriented reasoning for generalist socio-economic status prediction. Experiments demonstrate that CityRiSE with emergent reasoning process significantly outperforms existing baselines, improving both prediction accuracy and generalization across diverse urban contexts, particularly for prediction on unseen cities and unseen indicators. This work highlights the promise of combining RL and LVLMs for interpretable and generalist urban socio-economic sensing.
>
---
#### [new 154] Mismatch reconstruction theory for unknown measurement matrix in imaging through multimode fiber bending
- **分类: cs.CV; physics.optics**

- **简介: 该论文针对多模光纤成像中因光纤弯曲导致测量矩阵未知的问题，提出不匹配重建理论。通过构建新的测量矩阵，实现无需已知矩阵的图像重建，解决了传统算法在实际应用中因系统失配失效的问题。**

- **链接: [http://arxiv.org/pdf/2510.21787v1](http://arxiv.org/pdf/2510.21787v1)**

> **作者:** Le Yang
>
> **摘要:** Multimode fiber imaging requires strict matching between measurement value and measurement matrix to achieve image reconstruction. However, in practical applications, the measurement matrix often cannot be obtained due to unknown system configuration or difficulty in real-time alignment after arbitrary fiber bending, resulting in the failure of traditional reconstruction algorithms. This paper presents a novel mismatch reconstruction theory for solving the problem of image reconstruction when measurement matrix is unknown. We first propose mismatch equation and design matched and calibration solution algorithms to construct a new measurement matrix. In addition, we also provide a detailed proof of these equations and algorithms in the appendix. The experimental results show that under low noise levels, constructed matrix can be used for matched pair in traditional reconstruction algorithms, and reconstruct the original image successfully. Then, we analyze the impact of noise, computational precision and orthogonality on reconstruction performance. The results show that proposed algorithms have a certain degree of robustness. Finally, we discuss the limitations and potential applications of this theory. The code is available: https://github.com/yanglebupt/mismatch-solution.
>
---
#### [new 155] FRBNet: Revisiting Low-Light Vision through Frequency-Domain Radial Basis Network
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对低光照视觉任务，解决光照不足导致的特征退化问题。提出FRBNet，通过频域通道比与可学习滤波器实现光照不变特征提取，可无缝集成至现有网络，显著提升暗光目标检测与夜间分割性能。**

- **链接: [http://arxiv.org/pdf/2510.23444v1](http://arxiv.org/pdf/2510.23444v1)**

> **作者:** Fangtong Sun; Congyu Li; Ke Yang; Yuchen Pan; Hanwen Yu; Xichuan Zhang; Yiying Li
>
> **摘要:** Low-light vision remains a fundamental challenge in computer vision due to severe illumination degradation, which significantly affects the performance of downstream tasks such as detection and segmentation. While recent state-of-the-art methods have improved performance through invariant feature learning modules, they still fall short due to incomplete modeling of low-light conditions. Therefore, we revisit low-light image formation and extend the classical Lambertian model to better characterize low-light conditions. By shifting our analysis to the frequency domain, we theoretically prove that the frequency-domain channel ratio can be leveraged to extract illumination-invariant features via a structured filtering process. We then propose a novel and end-to-end trainable module named \textbf{F}requency-domain \textbf{R}adial \textbf{B}asis \textbf{Net}work (\textbf{FRBNet}), which integrates the frequency-domain channel ratio operation with a learnable frequency domain filter for the overall illumination-invariant feature enhancement. As a plug-and-play module, FRBNet can be integrated into existing networks for low-light downstream tasks without modifying loss functions. Extensive experiments across various downstream tasks demonstrate that FRBNet achieves superior performance, including +2.2 mAP for dark object detection and +2.9 mIoU for nighttime segmentation. Code is available at: https://github.com/Sing-Forevet/FRBNet.
>
---
#### [new 156] Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出TIRE方法，解决3D/4D生成中主体身份一致性差的问题。通过视频追踪定位需修改区域，利用主体驱动的2D修复模型渐进填充，并将结果重投影回3D，有效提升跨视角的身份保真度。**

- **链接: [http://arxiv.org/pdf/2510.23605v1](http://arxiv.org/pdf/2510.23605v1)**

> **作者:** Shuhong Zheng; Ashkan Mirzaei; Igor Gilitschenski
>
> **备注:** NeurIPS 2025, 38 pages, 22 figures
>
> **摘要:** Current 3D/4D generation methods are usually optimized for photorealism, efficiency, and aesthetics. However, they often fail to preserve the semantic identity of the subject across different viewpoints. Adapting generation methods with one or few images of a specific subject (also known as Personalization or Subject-driven generation) allows generating visual content that align with the identity of the subject. However, personalized 3D/4D generation is still largely underexplored. In this work, we introduce TIRE (Track, Inpaint, REsplat), a novel method for subject-driven 3D/4D generation. It takes an initial 3D asset produced by an existing 3D generative model as input and uses video tracking to identify the regions that need to be modified. Then, we adopt a subject-driven 2D inpainting model for progressively infilling the identified regions. Finally, we resplat the modified 2D multi-view observations back to 3D while still maintaining consistency. Extensive experiments demonstrate that our approach significantly improves identity preservation in 3D/4D generation compared to state-of-the-art methods. Our project website is available at https://zsh2000.github.io/track-inpaint-resplat.github.io/.
>
---
#### [new 157] MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型中视觉-语言对齐的训练难题，提出MergeMix统一增强范式。通过注意力感知的图像混合与偏好驱动训练，结合SimPO损失，提升对齐质量与训练效率，兼顾可扩展性与稳定性，适用于分类与多模态理解任务。**

- **链接: [http://arxiv.org/pdf/2510.23479v1](http://arxiv.org/pdf/2510.23479v1)**

> **作者:** Xin Jin; Siyuan Li; Siyong Jian; Kai Yu; Huan Wang
>
> **备注:** Code Link: https://github.com/JinXins/MergeMix
>
> **摘要:** Vision-language alignment in multi-modal large language models (MLLMs) typically relies on supervised fine-tuning (SFT) or reinforcement learning (RL). SFT is stable and efficient but requires large-scale human annotations and cannot capture subtle preferences, while RL brings in a reward signal for training, but suffers from overhead and instability. These limitations highlight a trade-off between scalability, robustness, and alignment quality. To address this, we propose MergeMix, a training-time augmentation paradigm that bridges SFT and RL. It first applies an attention-aware image mixing via token merge with more cluster representation and spatial context, and then presents a preference-driven training paradigm for MLLMs by building preference pairs with mixed images and raw images, and optimizing via SimPO loss. As a mixup augmentation, MergeMix enhances attention consistency and efficiency, surpassing other heuristic-based methods in classification. Extensive experiments demonstrate that MergeMix achieves competitive accuracy with improved efficiency, providing a scalable approach to preference alignment in classification and MLLMs.
>
---
#### [new 158] Explainable Deep Learning in Medical Imaging: Brain Tumor and Pneumonia Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医学影像诊断中深度学习模型缺乏可解释性的问题，提出一种可解释的深度学习框架，用于脑肿瘤和肺炎检测。基于ResNet50与DenseNet121模型，在公开数据集上训练并结合Grad-CAM生成热力图，提升决策透明度。结果表明DenseNet121在准确率与聚焦病理区域方面表现更优，推动了可信赖的AI辅助诊断应用。**

- **链接: [http://arxiv.org/pdf/2510.21823v1](http://arxiv.org/pdf/2510.21823v1)**

> **作者:** Sai Teja Erukude; Viswa Chaitanya Marella; Suhasnadh Reddy Veluru
>
> **备注:** Published in IEEE
>
> **摘要:** Deep Learning (DL) holds enormous potential for improving medical imaging diagnostics, yet the lack of interpretability in most models hampers clinical trust and adoption. This paper presents an explainable deep learning framework for detecting brain tumors in MRI scans and pneumonia in chest X-ray images using two leading Convolutional Neural Networks, ResNet50 and DenseNet121. These models were trained on publicly available Kaggle datasets comprising 7,023 brain MRI images and 5,863 chest X-ray images, achieving high classification performance. DenseNet121 consistently outperformed ResNet50 with 94.3 percent vs. 92.5 percent accuracy for brain tumors and 89.1 percent vs. 84.4 percent accuracy for pneumonia. For better explainability, Gradient-weighted Class Activation Mapping (Grad-CAM) was integrated to create heatmap visualizations superimposed on the test images, indicating the most influential image regions in the decision-making process. Interestingly, while both models produced accurate results, Grad-CAM showed that DenseNet121 consistently focused on core pathological regions, whereas ResNet50 sometimes scattered attention to peripheral or non-pathological areas. Combining deep learning and explainable AI offers a promising path toward reliable, interpretable, and clinically useful diagnostic tools.
>
---
#### [new 159] Bag-of-Word-Groups (BoWG): A Robust and Efficient Loop Closure Detection Method Under Perceptual Aliasing
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对SLAM中的回环检测任务，解决感知相似环境下因重复纹理导致的误检问题。提出BoWG方法，通过视觉词组捕捉空间共现关系，结合时序一致性与特征分布分析，提升精度与效率，在公开及自建数据集上均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.22529v1](http://arxiv.org/pdf/2510.22529v1)**

> **作者:** Xiang Fei; Tina Tian; Howie Choset; Lu Li
>
> **备注:** This paper has been accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Loop closure is critical in Simultaneous Localization and Mapping (SLAM) systems to reduce accumulative drift and ensure global mapping consistency. However, conventional methods struggle in perceptually aliased environments, such as narrow pipes, due to vector quantization, feature sparsity, and repetitive textures, while existing solutions often incur high computational costs. This paper presents Bag-of-Word-Groups (BoWG), a novel loop closure detection method that achieves superior precision-recall, robustness, and computational efficiency. The core innovation lies in the introduction of word groups, which captures the spatial co-occurrence and proximity of visual words to construct an online dictionary. Additionally, drawing inspiration from probabilistic transition models, we incorporate temporal consistency directly into similarity computation with an adaptive scheme, substantially improving precision-recall performance. The method is further strengthened by a feature distribution analysis module and dedicated post-verification mechanisms. To evaluate the effectiveness of our method, we conduct experiments on both public datasets and a confined-pipe dataset we constructed. Results demonstrate that BoWG surpasses state-of-the-art methods, including both traditional and learning-based approaches, in terms of precision-recall and computational efficiency. Our approach also exhibits excellent scalability, achieving an average processing time of 16 ms per image across 17,565 images in the Bicocca25b dataset.
>
---
#### [new 160] Switchable Token-Specific Codebook Quantization For Face Image Compression
- **分类: cs.CV**

- **简介: 该论文针对人脸图像压缩任务，解决传统全局码本忽略类别相关性和语义差异的问题。提出可切换的令牌特定码本量化方法，为不同类别学习独立码本组，并以少量比特记录归属，提升低比特率下的重建质量与表达能力。**

- **链接: [http://arxiv.org/pdf/2510.22943v1](http://arxiv.org/pdf/2510.22943v1)**

> **作者:** Yongbo Wang; Haonan Wang; Guodong Mu; Ruixin Zhang; Jiaqi Chen; Jingyun Zhang; Jun Wang; Yuan Xie; Zhizhong Zhang; Shouhong Ding
>
> **摘要:** With the ever-increasing volume of visual data, the efficient and lossless transmission, along with its subsequent interpretation and understanding, has become a critical bottleneck in modern information systems. The emerged codebook-based solution utilize a globally shared codebook to quantize and dequantize each token, controlling the bpp by adjusting the number of tokens or the codebook size. However, for facial images, which are rich in attributes, such global codebook strategies overlook both the category-specific correlations within images and the semantic differences among tokens, resulting in suboptimal performance, especially at low bpp. Motivated by these observations, we propose a Switchable Token-Specific Codebook Quantization for face image compression, which learns distinct codebook groups for different image categories and assigns an independent codebook to each token. By recording the codebook group to which each token belongs with a small number of bits, our method can reduce the loss incurred when decreasing the size of each codebook group. This enables a larger total number of codebooks under a lower overall bpp, thereby enhancing the expressive capability and improving reconstruction performance. Owing to its generalizable design, our method can be integrated into any existing codebook-based representation learning approach and has demonstrated its effectiveness on face recognition datasets, achieving an average accuracy of 93.51% for reconstructed images at 0.05 bpp.
>
---
#### [new 161] DPGLA: Bridging the Gap between Synthetic and Real Data for Unsupervised Domain Adaptation in 3D LiDAR Semantic Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D LiDAR语义分割中的无监督域自适应问题，解决合成数据与真实数据间的域偏移及未标注数据利用不足。提出动态伪标签过滤（DPLF）和先验引导的数据增强（PG-DAP），结合数据混合一致性损失，有效提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.23525v1](http://arxiv.org/pdf/2510.23525v1)**

> **作者:** Wanmeng Li; Simone Mosco; Daniel Fusaro; Alberto Pretto
>
> **备注:** This paper has been accepted for publication at the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Annotating real-world LiDAR point clouds for use in intelligent autonomous systems is costly. To overcome this limitation, self-training-based Unsupervised Domain Adaptation (UDA) has been widely used to improve point cloud semantic segmentation by leveraging synthetic point cloud data. However, we argue that existing methods do not effectively utilize unlabeled data, as they either rely on predefined or fixed confidence thresholds, resulting in suboptimal performance. In this paper, we propose a Dynamic Pseudo-Label Filtering (DPLF) scheme to enhance real data utilization in point cloud UDA semantic segmentation. Additionally, we design a simple and efficient Prior-Guided Data Augmentation Pipeline (PG-DAP) to mitigate domain shift between synthetic and real-world point clouds. Finally, we utilize data mixing consistency loss to push the model to learn context-free representations. We implement and thoroughly evaluate our approach through extensive comparisons with state-of-the-art methods. Experiments on two challenging synthetic-to-real point cloud semantic segmentation tasks demonstrate that our approach achieves superior performance. Ablation studies confirm the effectiveness of the DPLF and PG-DAP modules. We release the code of our method in this paper.
>
---
#### [new 162] SARCLIP: A Vision Language Foundation Model for Semantic Understanding and Target Recognition in SAR Imagery
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SARCLIP，首个面向合成孔径雷达（SAR）图像的视觉语言基础模型。针对SAR影像语义理解与零样本目标识别难题，构建了百万级图文数据集SARCLIP-1M，通过对比学习实现跨模态对齐，显著提升特征提取与解释能力，在检索与分类任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.22665v1](http://arxiv.org/pdf/2510.22665v1)**

> **作者:** Qiwei Ma; Zhiyu Wang; Wang Liu; Xukun Lu; Bin Deng; Puhong Duan; Xudong Kang; Shutao Li
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Synthetic Aperture Radar (SAR) has emerged as a crucial imaging modality due to its all-weather capabilities. While recent advancements in self-supervised learning and Masked Image Modeling (MIM) have paved the way for SAR foundation models, these approaches primarily focus on low-level visual features, often overlooking multimodal alignment and zero-shot target recognition within SAR imagery. To address this limitation, we construct SARCLIP-1M, a large-scale vision language dataset comprising over one million text-image pairs aggregated from existing datasets. We further introduce SARCLIP, the first vision language foundation model tailored for the SAR domain. Our SARCLIP model is trained using a contrastive vision language learning approach by domain transferring strategy, enabling it to bridge the gap between SAR imagery and textual descriptions. Extensive experiments on image-text retrieval and zero-shot classification tasks demonstrate the superior performance of SARCLIP in feature extraction and interpretation, significantly outperforming state-of-the-art foundation models and advancing the semantic understanding of SAR imagery. The code and datasets will be released soon.
>
---
#### [new 163] EventFormer: A Node-graph Hierarchical Attention Transformer for Action-centric Video Event Prediction
- **分类: cs.CV; cs.AI; cs.MM; I.2.10**

- **简介: 该论文提出面向动作中心视频事件预测（AVEP）的新任务，旨在基于视频上下文预测后续事件。针对现有视觉模型难以处理复杂事件结构的问题，构建了包含35K视频与178K片段的细粒度标注数据集，并提出EventFormer模型，通过节点图层次注意力机制捕捉事件间及论元间的复杂关系，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.21786v1](http://arxiv.org/pdf/2510.21786v1)**

> **作者:** Qile Su; Shoutai Zhu; Shuai Zhang; Baoyu Liang; Chao Tong
>
> **备注:** 15 pages, 7 figures, 6 tables
>
> **摘要:** Script event induction, which aims to predict the subsequent event based on the context, is a challenging task in NLP, achieving remarkable success in practical applications. However, human events are mostly recorded and presented in the form of videos rather than scripts, yet there is a lack of related research in the realm of vision. To address this problem, we introduce AVEP (Action-centric Video Event Prediction), a task that distinguishes itself from existing video prediction tasks through its incorporation of more complex logic and richer semantic information. We present a large structured dataset, which consists of about $35K$ annotated videos and more than $178K$ video clips of event, built upon existing video event datasets to support this task. The dataset offers more fine-grained annotations, where the atomic unit is represented as a multimodal event argument node, providing better structured representations of video events. Due to the complexity of event structures, traditional visual models that take patches or frames as input are not well-suited for AVEP. We propose EventFormer, a node-graph hierarchical attention based video event prediction model, which can capture both the relationships between events and their arguments and the coreferencial relationships between arguments. We conducted experiments using several SOTA video prediction models as well as LVLMs on AVEP, demonstrating both the complexity of the task and the value of the dataset. Our approach outperforms all these video prediction models. We will release the dataset and code for replicating the experiments and annotations.
>
---
#### [new 164] hYOLO Model: Enhancing Object Classification with Hierarchical Context in YOLOv8
- **分类: cs.CV**

- **简介: 该论文针对目标检测与分类任务，提出hYOLO模型，旨在解决传统平铺式分类忽略物体间层次结构的问题。通过构建层次化架构、改进损失函数与评估指标，模型能更好地捕捉物体间的层级关系，提升对真实世界复杂场景的理解能力。**

- **链接: [http://arxiv.org/pdf/2510.23278v1](http://arxiv.org/pdf/2510.23278v1)**

> **作者:** Veska Tsenkova; Peter Stanchev; Daniel Petrov; Deyan Lazarov
>
> **备注:** 39 pages, 12 figures, 4 tables, code available at https://github.com/ds2run/hyolo
>
> **摘要:** Current convolution neural network (CNN) classification methods are predominantly focused on flat classification which aims solely to identify a specified object within an image. However, real-world objects often possess a natural hierarchical organization that can significantly help classification tasks. Capturing the presence of relations between objects enables better contextual understanding as well as control over the severity of mistakes. Considering these aspects, this paper proposes an end-to-end hierarchical model for image detection and classification built upon the YOLO model family. A novel hierarchical architecture, a modified loss function, and a performance metric tailored to the hierarchical nature of the model are introduced. The proposed model is trained and evaluated on two different hierarchical categorizations of the same dataset: a systematic categorization that disregards visual similarities between objects and a categorization accounting for common visual characteristics across classes. The results illustrate how the suggested methodology addresses the inherent hierarchical structure present in real-world objects, which conventional flat classification algorithms often overlook.
>
---
#### [new 165] Proportion and Perspective Control for Flow-Based Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本生成图像中空间结构控制不足的问题，提出两种专用ControlNet：比例控制网（基于边界框）和视角控制网（基于消失线），实现对物体位置、尺度及3D场景几何的精确调控。通过视觉语言模型辅助标注与定制化训练流程，提升生成可控性，模型已开源。**

- **链接: [http://arxiv.org/pdf/2510.21763v1](http://arxiv.org/pdf/2510.21763v1)**

> **作者:** Julien Boudier; Hugo Caselles-Dupré
>
> **备注:** Technical report after open-source release
>
> **摘要:** While modern text-to-image diffusion models generate high-fidelity images, they offer limited control over the spatial and geometric structure of the output. To address this, we introduce and evaluate two ControlNets specialized for artistic control: (1) a proportion ControlNet that uses bounding boxes to dictate the position and scale of objects, and (2) a perspective ControlNet that employs vanishing lines to control the 3D geometry of the scene. We support the training of these modules with data pipelines that leverage vision-language models for annotation and specialized algorithms for conditioning image synthesis. Our experiments demonstrate that both modules provide effective control but exhibit limitations with complex constraints. Both models are released on HuggingFace: https://huggingface.co/obvious-research
>
---
#### [new 166] From Pixels to Views: Learning Angular-Aware and Physics-Consistent Representations for Light Field Microscopy
- **分类: cs.CV**

- **简介: 该论文针对光场显微镜3D重建任务，解决数据集缺失与物理一致性建模难题。构建了XLFM-Zebrafish基准数据集，提出自监督的MVN-LF方法学习视角先验，并设计可微分的光学渲染一致性损失，提升重建精度与数据效率。**

- **链接: [http://arxiv.org/pdf/2510.22577v1](http://arxiv.org/pdf/2510.22577v1)**

> **作者:** Feng He; Guodong Tan; Qiankun Li; Jun Yu; Quan Wen
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Light field microscopy (LFM) has become an emerging tool in neuroscience for large-scale neural imaging in vivo, notable for its single-exposure volumetric imaging, broad field of view, and high temporal resolution. However, learning-based 3D reconstruction in XLFM remains underdeveloped due to two core challenges: the absence of standardized datasets and the lack of methods that can efficiently model its angular-spatial structure while remaining physically grounded. We address these challenges by introducing three key contributions. First, we construct the XLFM-Zebrafish benchmark, a large-scale dataset and evaluation suite for XLFM reconstruction. Second, we propose Masked View Modeling for Light Fields (MVN-LF), a self-supervised task that learns angular priors by predicting occluded views, improving data efficiency. Third, we formulate the Optical Rendering Consistency Loss (ORC Loss), a differentiable rendering constraint that enforces alignment between predicted volumes and their PSF-based forward projections. On the XLFM-Zebrafish benchmark, our method improves PSNR by 7.7% over state-of-the-art baselines.
>
---
#### [new 167] GRAID: Enhancing Spatial Reasoning of VLMs Through High-Fidelity Data Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型（VLMs）在空间推理能力上的不足，提出GRAID框架，通过2D边界框生成高质量的视觉问答数据。解决了现有方法因3D重建误差和生成幻觉导致的数据质量低问题，显著提升空间推理性能，且模型在未见任务上表现更优。**

- **链接: [http://arxiv.org/pdf/2510.22118v1](http://arxiv.org/pdf/2510.22118v1)**

> **作者:** Karim Elmaaroufi; Liheng Lai; Justin Svegliato; Yutong Bai; Sanjit A. Seshia; Matei Zaharia
>
> **备注:** 22 pages, 3 figures, 3 tables, project page: https://ke7.github.io/graid/
>
> **摘要:** Vision Language Models (VLMs) achieve strong performance on many vision-language tasks but often struggle with spatial reasoning\textemdash{}a prerequisite for many applications. Empirically, we find that a dataset produced by a current training data generation pipeline has a 57.6\% human validation rate. These rates stem from current limitations: single-image 3D reconstruction introduces cascading modeling errors and requires wide answer tolerances, while caption-based methods require hyper-detailed annotations and suffer from generative hallucinations. We present GRAID, built on the key insight that qualitative spatial relationships can be reliably determined from 2D geometric primitives alone. By operating exclusively on 2D bounding boxes from standard object detectors, GRAID avoids both 3D reconstruction errors and generative hallucinations, resulting in datasets that are of higher quality than existing tools that produce similar datasets as validated by human evaluations. We apply our framework to the BDD100k, NuImages, and Waymo datasets, generating over 8.5 million high-quality VQA pairs creating questions spanning spatial relations, counting, ranking, and size comparisons. We evaluate one of the datasets and find it achieves 91.16\% human-validated accuracy\textemdash{}compared to 57.6\% on a dataset generated by recent work. % or recent work Critically, we demonstrate that when trained on GRAID data, models learn spatial reasoning concepts that generalize: models fine-tuned on 6 question types improve on over 10 held-out types, with accuracy gains of 47.5\% on BDD and 37.9\% on NuImages for Llama 3.2B 11B, and when trained on all questions types, achieve improvements on several existing benchmarks such as BLINK. The GRAID framework, datasets, and additional information can be found on our \href{https://ke7.github.io/graid/}{project page}.
>
---
#### [new 168] Fast Voxel-Wise Kinetic Modeling in Dynamic PET using a Physics-Informed CycleGAN
- **分类: cs.CV; q-bio.OT**

- **简介: 该论文针对动态PET中动脉输入函数（AIF）估计复杂且侵入性的问题，提出基于物理信息循环生成对抗网络（CycleGAN）的快速体素级动力学建模方法。通过引入物理约束，实现高精度AIF预测与参数图生成，显著简化流程并提升量化准确性。**

- **链接: [http://arxiv.org/pdf/2510.23140v1](http://arxiv.org/pdf/2510.23140v1)**

> **作者:** Christian Salomonsen; Samuel Kuttner; Michael Kampffmeyer; Robert Jenssen; Kristoffer Wickstrøm; Jong Chul Ye; Elisabeth Wetzer
>
> **备注:** 5 pages, 1 figure. Pre-review preprint. Submitted to MedEurIPS 2025 (EurIPS workshop)
>
> **摘要:** Tracer kinetic modeling serves a vital role in diagnosis, treatment planning, tracer development and oncology, but burdens practitioners with complex and invasive arterial input function estimation (AIF). We adopt a physics-informed CycleGAN showing promise in DCE-MRI quantification to dynamic PET quantification. Our experiments demonstrate sound AIF predictions and parameter maps closely resembling the reference.
>
---
#### [new 169] LongCat-Video Technical Report
- **分类: cs.CV**

- **简介: 该论文提出LongCat-Video，一个13.6B参数的视频生成模型，旨在解决长视频高效生成难题。通过统一架构支持文本/图像到视频及视频续写任务，结合粗到精生成策略与块稀疏注意力，实现分钟级高质量长视频生成，并采用多奖励强化学习优化性能，推动世界模型发展。**

- **链接: [http://arxiv.org/pdf/2510.22200v1](http://arxiv.org/pdf/2510.22200v1)**

> **作者:** Meituan LongCat Team; Xunliang Cai; Qilong Huang; Zhuoliang Kang; Hongyu Li; Shijun Liang; Liya Ma; Siyu Ren; Xiaoming Wei; Rixu Xie; Tong Zhang
>
> **摘要:** Video generation is a critical pathway toward world models, with efficient long video inference as a key capability. Toward this end, we introduce LongCat-Video, a foundational video generation model with 13.6B parameters, delivering strong performance across multiple video generation tasks. It particularly excels in efficient and high-quality long video generation, representing our first step toward world models. Key features include: Unified architecture for multiple tasks: Built on the Diffusion Transformer (DiT) framework, LongCat-Video supports Text-to-Video, Image-to-Video, and Video-Continuation tasks with a single model; Long video generation: Pretraining on Video-Continuation tasks enables LongCat-Video to maintain high quality and temporal coherence in the generation of minutes-long videos; Efficient inference: LongCat-Video generates 720p, 30fps videos within minutes by employing a coarse-to-fine generation strategy along both the temporal and spatial axes. Block Sparse Attention further enhances efficiency, particularly at high resolutions; Strong performance with multi-reward RLHF: Multi-reward RLHF training enables LongCat-Video to achieve performance on par with the latest closed-source and leading open-source models. Code and model weights are publicly available to accelerate progress in the field.
>
---
#### [new 170] A Flow Model with Low-Rank Transformers for Incomplete Multimodal Survival Analysis
- **分类: cs.CV**

- **简介: 该论文针对多模态医学数据中因缺失模态导致的生存分析难题，提出结合低秩Transformer与流模型的框架。通过类别感知流模型对齐跨模态分布，构建一致的潜在空间，提升缺失模态重建质量；并设计轻量级低秩Transformer，有效建模模态内依赖，缓解高维融合过拟合问题。**

- **链接: [http://arxiv.org/pdf/2510.21829v1](http://arxiv.org/pdf/2510.21829v1)**

> **作者:** Yi Yin; Yuntao Shou; Zao Dai; Yun Peng; Tao Meng; Wei Ai; Keqin Li
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** In recent years, multimodal medical data-based survival analysis has attracted much attention. However, real-world datasets often suffer from the problem of incomplete modality, where some patient modality information is missing due to acquisition limitations or system failures. Existing methods typically infer missing modalities directly from observed ones using deep neural networks, but they often ignore the distributional discrepancy across modalities, resulting in inconsistent and unreliable modality reconstruction. To address these challenges, we propose a novel framework that combines a low-rank Transformer with a flow-based generative model for robust and flexible multimodal survival prediction. Specifically, we first formulate the concerned problem as incomplete multimodal survival analysis using the multi-instance representation of whole slide images (WSIs) and genomic profiles. To realize incomplete multimodal survival analysis, we propose a class-specific flow for cross-modal distribution alignment. Under the condition of class labels, we model and transform the cross-modal distribution. By virtue of the reversible structure and accurate density modeling capabilities of the normalizing flow model, the model can effectively construct a distribution-consistent latent space of the missing modality, thereby improving the consistency between the reconstructed data and the true distribution. Finally, we design a lightweight Transformer architecture to model intra-modal dependencies while alleviating the overfitting problem in high-dimensional modality fusion by virtue of the low-rank Transformer. Extensive experiments have demonstrated that our method not only achieves state-of-the-art performance under complete modality settings, but also maintains robust and superior accuracy under the incomplete modalities scenario.
>
---
#### [new 171] Sprint: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文针对扩散模型中Transformer训练成本高的问题，提出SPRINT方法。通过稀疏-密集残差融合机制，在保持生成质量的前提下实现高达75%的令牌丢弃，显著降低训练开销。采用两阶段训练策略，结合路径丢弃引导推理优化，大幅减少计算量并提升效率。**

- **链接: [http://arxiv.org/pdf/2510.21986v1](http://arxiv.org/pdf/2510.21986v1)**

> **作者:** Dogyun Park; Moayed Haji-Ali; Yanyu Li; Willi Menapace; Sergey Tulyakov; Hyunwoo J. Kim; Aliaksandr Siarohin; Anil Kag
>
> **摘要:** Diffusion Transformers (DiTs) deliver state-of-the-art generative performance but their quadratic training cost with sequence length makes large-scale pretraining prohibitively expensive. Token dropping can reduce training cost, yet na\"ive strategies degrade representations, and existing methods are either parameter-heavy or fail at high drop ratios. We present SPRINT, Sparse--Dense Residual Fusion for Efficient Diffusion Transformers, a simple method that enables aggressive token dropping (up to 75%) while preserving quality. SPRINT leverages the complementary roles of shallow and deep layers: early layers process all tokens to capture local detail, deeper layers operate on a sparse subset to cut computation, and their outputs are fused through residual connections. Training follows a two-stage schedule: long masked pre-training for efficiency followed by short full-token fine-tuning to close the train--inference gap. On ImageNet-1K 256x256, SPRINT achieves 9.8x training savings with comparable FID/FDD, and at inference, its Path-Drop Guidance (PDG) nearly halves FLOPs while improving quality. These results establish SPRINT as a simple, effective, and general solution for efficient DiT training.
>
---
#### [new 172] Exploring the design space of diffusion and flow models for data fusion
- **分类: cs.CV; physics.ins-det**

- **简介: 该论文研究卫星遥感中夜间灯光数据融合任务，针对DMSP-OLS与VIIRS数据融合问题，探索扩散与流模型的设计空间。通过对比UNet、扩散与流模型，发现基于UNet的扩散模型在保留细节和生成高质量图像方面表现最优，并提出噪声调度与量化策略以平衡性能与效率。**

- **链接: [http://arxiv.org/pdf/2510.21791v1](http://arxiv.org/pdf/2510.21791v1)**

> **作者:** Niraj Chaudhari; Manmeet Singh; Naveen Sudharsan; Amit Kumar Srivastava; Harsh Kamath; Dushyant Mahajan; Ayan Paul
>
> **摘要:** Data fusion is an essential task in various domains, enabling the integration of multi-source information to enhance data quality and insights. One key application is in satellite remote sensing, where fusing multi-sensor observations can improve spatial and temporal resolution. In this study, we explore the design space of diffusion and flow models for data fusion, focusing on the integration of Defense Meteorological Satellite Program's Operational Linescan System (DMSP-OLS) and Visible Infrared Imaging Radiometer Suite (VIIRS) nighttime lights data. Our approach leverages a diverse set of 2D image-to-image generative models, including UNET, diffusion, and flow modeling architectures. We evaluate the effectiveness of these architectures in satellite remote sensing data fusion, identifying diffusion models based on UNet as particularly adept at preserving fine-grained spatial details and generating high-fidelity fused images. We also provide guidance on the selection of noise schedulers in diffusion-based models, highlighting the trade-offs between iterative solvers for faster inference and discrete schedulers for higher-quality reconstructions. Additionally, we explore quantization techniques to optimize memory efficiency and computational cost without compromising performance. Our findings offer practical insights into selecting the most effective diffusion and flow model architectures for data fusion tasks, particularly in remote sensing applications, and provide recommendations for leveraging noise scheduling strategies to enhance fusion quality.
>
---
#### [new 173] Agro-Consensus: Semantic Self-Consistency in Vision-Language Models for Crop Disease Management in Developing Countries
- **分类: cs.CV**

- **简介: 该论文针对发展中国家农作物病害管理中专家缺乏、网络差等问题，提出Agro-Consensus框架，通过轻量级语义聚类与自一致性机制提升视觉语言模型的诊断准确性。工作包括构建多候选生成与共识选择流程，并引入人机协同验证，显著提升图像描述准确率。**

- **链接: [http://arxiv.org/pdf/2510.21757v1](http://arxiv.org/pdf/2510.21757v1)**

> **作者:** Mihir Gupta; Pratik Desai; Ross Greer
>
> **摘要:** Agricultural disease management in developing countries such as India, Kenya, and Nigeria faces significant challenges due to limited access to expert plant pathologists, unreliable internet connectivity, and cost constraints that hinder the deployment of large-scale AI systems. This work introduces a cost-effective self-consistency framework to improve vision-language model (VLM) reliability for agricultural image captioning. The proposed method employs semantic clustering, using a lightweight (80MB) pre-trained embedding model to group multiple candidate responses. It then selects the most coherent caption -- containing a diagnosis, symptoms, analysis, treatment, and prevention recommendations -- through a cosine similarity-based consensus. A practical human-in-the-loop (HITL) component is incorporated, wherein user confirmation of the crop type filters erroneous generations, ensuring higher-quality input for the consensus mechanism. Applied to the publicly available PlantVillage dataset using a fine-tuned 3B-parameter PaliGemma model, our framework demonstrates improvements over standard decoding methods. Evaluated on 800 crop disease images with up to 21 generations per image, our single-cluster consensus method achieves a peak accuracy of 83.1% with 10 candidate generations, compared to the 77.5% baseline accuracy of greedy decoding. The framework's effectiveness is further demonstrated when considering multiple clusters; accuracy rises to 94.0% when a correct response is found within any of the top four candidate clusters, outperforming the 88.5% achieved by a top-4 selection from the baseline.
>
---
#### [new 174] Semantic-Preserving Cross-Style Visual Reasoning for Robust Multi-Modal Understanding in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对大视觉语言模型在跨风格理解中的“风格陷阱”问题，提出SP-CSVR框架，通过风格内容解耦、语义对齐的上下文解码和自适应语义一致性模块，实现鲁棒的跨风格视觉推理。任务为多模态理解中的视觉问答与图像描述，解决风格干扰导致的语义不一致问题。**

- **链接: [http://arxiv.org/pdf/2510.22838v1](http://arxiv.org/pdf/2510.22838v1)**

> **作者:** Aya Nakayama; Brian Wong; Yuji Nishimura; Kaito Tanaka
>
> **摘要:** The "style trap" poses a significant challenge for Large Vision-Language Models (LVLMs), hindering robust semantic understanding across diverse visual styles, especially in in-context learning (ICL). Existing methods often fail to effectively decouple style from content, hindering generalization. To address this, we propose the Semantic-Preserving Cross-Style Visual Reasoner (SP-CSVR), a novel framework for stable semantic understanding and adaptive cross-style visual reasoning. SP-CSVR integrates a Cross-Style Feature Encoder (CSFE) for style-content disentanglement, a Semantic-Aligned In-Context Decoder (SAICD) for efficient few-shot style adaptation, and an Adaptive Semantic Consistency Module (ASCM) employing multi-task contrastive learning to enforce cross-style semantic invariance. Extensive experiments on a challenging multi-style dataset demonstrate SP-CSVR's state-of-the-art performance across visual captioning, visual question answering, and in-context style adaptation. Comprehensive evaluations, including ablation studies and generalization analysis, confirm SP-CSVR's efficacy in enhancing robustness, generalization, and efficiency across diverse visual styles.
>
---
#### [new 175] Attention Residual Fusion Network with Contrast for Source-free Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文针对无源域自适应（SFDA）任务，解决域偏移与负迁移问题。提出注意力残差融合网络（ARFNet），结合全局-局部对比学习与动态中心点评估，增强特征判别性，提升模型在无源数据下的适应能力。**

- **链接: [http://arxiv.org/pdf/2510.22142v1](http://arxiv.org/pdf/2510.22142v1)**

> **作者:** Renrong Shao; Wei Zhang; Jun Wang
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** Source-free domain adaptation (SFDA) involves training a model on source domain and then applying it to a related target domain without access to the source data and labels during adaptation. The complexity of scene information and lack of the source domain make SFDA a difficult task. Recent studies have shown promising results, but many approaches to domain adaptation concentrate on domain shift and neglect the effects of negative transfer, which may impede enhancements of model performance during adaptation. n this paper, addressing this issue, we propose a novel framework of Attention Residual Fusion Network (ARFNet) based on contrast learning for SFDA to alleviate negative transfer and domain shift during the progress of adaptation, in which attention residual fusion, global-local attention contrast, and dynamic centroid evaluation are exploited. Concretely, the attention mechanism is first exploited to capture the discriminative region of the target object. Then, in each block, attention features are decomposed into spatial-wise and channel-wise attentions to achieve the cross-layer attention residual fusion progressively and self-distillation. During adaptation progress, we contrast global and local representations to improve the perceptual capabilities of different categories, which enables the model to discriminate variations between inner-class and intra-class. Finally, a dynamic centroid evaluation strategy is exploited to evaluate the trustworthy centroids and labels for self-supervised self-distillation, which aims to accurately approximate the center of the source domain and pseudo-labels to mitigate domain shift. To validate the efficacy, we execute comprehensive experiments on five benchmarks of varying scales. Experimental outcomes indicate that our method surpasses other techniques, attaining superior performance across SFDA benchmarks.
>
---
#### [new 176] FARMER: Flow AutoRegressive Transformer over Pixels
- **分类: cs.CV**

- **简介: 该论文提出FARMER，一种基于像素的生成模型，融合归一化流与自回归机制，实现高效图像生成与精确似然估计。针对连续像素建模的高维长序列问题，引入自监督维度压缩与快速推理优化，提升生成质量与速度，解决像素级生成中的效率与精度难题。**

- **链接: [http://arxiv.org/pdf/2510.23588v1](http://arxiv.org/pdf/2510.23588v1)**

> **作者:** Guangting Zheng; Qinyu Zhao; Tao Yang; Fei Xiao; Zhijie Lin; Jie Wu; Jiajun Deng; Yanyong Zhang; Rui Zhu
>
> **备注:** Bytedance Seed Technical Report
>
> **摘要:** Directly modeling the explicit likelihood of the raw data distribution is key topic in the machine learning area, which achieves the scaling successes in Large Language Models by autoregressive modeling. However, continuous AR modeling over visual pixel data suffer from extremely long sequences and high-dimensional spaces. In this paper, we present FARMER, a novel end-to-end generative framework that unifies Normalizing Flows (NF) and Autoregressive (AR) models for tractable likelihood estimation and high-quality image synthesis directly from raw pixels. FARMER employs an invertible autoregressive flow to transform images into latent sequences, whose distribution is modeled implicitly by an autoregressive model. To address the redundancy and complexity in pixel-level modeling, we propose a self-supervised dimension reduction scheme that partitions NF latent channels into informative and redundant groups, enabling more effective and efficient AR modeling. Furthermore, we design a one-step distillation scheme to significantly accelerate inference speed and introduce a resampling-based classifier-free guidance algorithm to boost image generation quality. Extensive experiments demonstrate that FARMER achieves competitive performance compared to existing pixel-based generative models while providing exact likelihoods and scalable training.
>
---
#### [new 177] AI Powered Urban Green Infrastructure Assessment Through Aerial Imagery of an Industrial Township
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于城市绿地覆盖评估任务，旨在解决传统方法在数据处理、可扩展性和精度上的不足。研究利用深度学习驱动的无人机影像分析，实现高分辨率图像中绿植冠层的精准分割与覆盖估算，并基于云平台高效处理大规模数据，为工业城镇的绿化规划与碳汇评估提供支持。**

- **链接: [http://arxiv.org/pdf/2510.21876v1](http://arxiv.org/pdf/2510.21876v1)**

> **作者:** Anisha Dutta
>
> **备注:** Presented at IIIE Conference 2024, Jamshedpur
>
> **摘要:** Accurate assessment of urban canopy coverage is crucial for informed urban planning, effective environmental monitoring, and mitigating the impacts of climate change. Traditional practices often face limitations due to inadequate technical requirements, difficulties in scaling and data processing, and the lack of specialized expertise. This study presents an efficient approach for estimating green canopy coverage using artificial intelligence, specifically computer vision techniques, applied to aerial imageries. Our proposed methodology utilizes object-based image analysis, based on deep learning algorithms to accurately identify and segment green canopies from high-resolution drone images. This approach allows the user for detailed analysis of urban vegetation, capturing variations in canopy density and understanding spatial distribution. To overcome the computational challenges associated with processing large datasets, it was implemented over a cloud platform utilizing high-performance processors. This infrastructure efficiently manages space complexity and ensures affordable latency, enabling the rapid analysis of vast amounts of drone imageries. Our results demonstrate the effectiveness of this approach in accurately estimating canopy coverage at the city scale, providing valuable insights for urban forestry management of an industrial township. The resultant data generated by this method can be used to optimize tree plantation and assess the carbon sequestration potential of urban forests. By integrating these insights into sustainable urban planning, we can foster more resilient urban environments, contributing to a greener and healthier future.
>
---
#### [new 178] Video-Thinker: Sparking "Thinking with Videos" via Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文提出Video-Thinker，解决视频推理中缺乏动态思维机制的问题。通过构建自动生成推理线索的训练数据集与强化学习策略，使多模态大模型能自主进行视频定位与描述，实现“用视频思考”。在多个基准上显著超越现有方法，成为7B级别最优模型。**

- **链接: [http://arxiv.org/pdf/2510.23473v1](http://arxiv.org/pdf/2510.23473v1)**

> **作者:** Shijian Wang; Jiarui Jin; Xingjian Wang; Linxin Song; Runhao Fu; Hecheng Wang; Zongyuan Ge; Yuan Lu; Xuelian Cheng
>
> **摘要:** Recent advances in image reasoning methods, particularly "Thinking with Images", have demonstrated remarkable success in Multimodal Large Language Models (MLLMs); however, this dynamic reasoning paradigm has not yet been extended to video reasoning tasks. In this paper, we propose Video-Thinker, which empowers MLLMs to think with videos by autonomously leveraging their intrinsic "grounding" and "captioning" capabilities to generate reasoning clues throughout the inference process. To spark this capability, we construct Video-Thinker-10K, a curated dataset featuring autonomous tool usage within chain-of-thought reasoning sequences. Our training strategy begins with Supervised Fine-Tuning (SFT) to learn the reasoning format, followed by Group Relative Policy Optimization (GRPO) to strengthen this reasoning capability. Through this approach, Video-Thinker enables MLLMs to autonomously navigate grounding and captioning tasks for video reasoning, eliminating the need for constructing and calling external tools. Extensive experiments demonstrate that Video-Thinker achieves significant performance gains on both in-domain tasks and challenging out-of-domain video reasoning benchmarks, including Video-Holmes, CG-Bench-Reasoning, and VRBench. Our Video-Thinker-7B substantially outperforms existing baselines such as Video-R1 and establishes state-of-the-art performance among 7B-sized MLLMs.
>
---
#### [new 179] PlanarTrack: A high-quality and challenging benchmark for large-scale planar object tracking
- **分类: cs.CV**

- **简介: 该论文提出PlanarTrack，一个大规模、高挑战性平面目标跟踪基准。针对现有数据集规模小、多样性不足的问题，构建包含1150序列的高质量数据集，涵盖短/长期跟踪，采用精细人工标注与独特目标设计，推动平面跟踪算法评估与研究。**

- **链接: [http://arxiv.org/pdf/2510.23368v1](http://arxiv.org/pdf/2510.23368v1)**

> **作者:** Yifan Jiao; Xinran Liu; Xiaoqiong Liu; Xiaohui Yuan; Heng Fan; Libo Zhang
>
> **摘要:** Planar tracking has drawn increasing interest owing to its key roles in robotics and augmented reality. Despite recent great advancement, further development of planar tracking, particularly in the deep learning era, is largely limited compared to generic tracking due to the lack of large-scale platforms. To mitigate this, we propose PlanarTrack, a large-scale high-quality and challenging benchmark for planar tracking. Specifically, PlanarTrack consists of 1,150 sequences with over 733K frames, including 1,000 short-term and 150 new long-term videos, which enables comprehensive evaluation of short- and long-term tracking performance. All videos in PlanarTrack are recorded in unconstrained conditions from the wild, which makes PlanarTrack challenging but more realistic for real-world applications. To ensure high-quality annotations, each video frame is manually annotated by four corner points with multi-round meticulous inspection and refinement. To enhance target diversity of PlanarTrack, we only capture a unique target in one sequence, which is different from existing benchmarks. To our best knowledge, PlanarTrack is by far the largest and most diverse and challenging dataset dedicated to planar tracking. To understand performance of existing methods on PlanarTrack and to provide a comparison for future research, we evaluate 10 representative planar trackers with extensive comparison and in-depth analysis. Our evaluation reveals that, unsurprisingly, the top planar trackers heavily degrade on the challenging PlanarTrack, which indicates more efforts are required for improving planar tracking. Our data and results will be released at https://github.com/HengLan/PlanarTrack
>
---
#### [new 180] Strategies for Robust Deep Learning Based Deformable Registration
- **分类: cs.CV**

- **简介: 该论文针对深度学习图像配准在跨模态、跨对比度场景下泛化能力差的问题，提出将图像先转换至MIND特征空间再输入模型，并采用特殊集成策略，显著提升模型鲁棒性，有效应对训练数据分布外的挑战。**

- **链接: [http://arxiv.org/pdf/2510.23079v1](http://arxiv.org/pdf/2510.23079v1)**

> **作者:** Joel Honkamaa; Pekka Marttinen
>
> **摘要:** Deep learning based deformable registration methods have become popular in recent years. However, their ability to generalize beyond training data distribution can be poor, significantly hindering their usability. LUMIR brain registration challenge for Learn2Reg 2025 aims to advance the field by evaluating the performance of the registration on contrasts and modalities different from those included in the training set. Here we describe our submission to the challenge, which proposes a very simple idea for significantly improving robustness by transforming the images into MIND feature space before feeding them into the model. In addition, a special ensembling strategy is proposed that shows a small but consistent improvement.
>
---
#### [new 181] Interpretable Tile-Based Classification of Paclitaxel Exposure
- **分类: cs.CV**

- **简介: 该论文针对药物暴露分类任务，解决帕利他赛处理下胶质瘤细胞图像的细微剂量差异识别问题。提出基于局部图像块的分块聚合方法，显著提升分类准确率，并通过可解释性分析揭示模型有效性，推动医学图像研究向可解释与鲁棒方向发展。**

- **链接: [http://arxiv.org/pdf/2510.23363v1](http://arxiv.org/pdf/2510.23363v1)**

> **作者:** Sean Fletcher; Gabby Scott; Douglas Currie; Xin Zhang; Yuqi Song; Bruce MacLeod
>
> **摘要:** Medical image analysis is central to drug discovery and preclinical evaluation, where scalable, objective readouts can accelerate decision-making. We address classification of paclitaxel (Taxol) exposure from phase-contrast microscopy of C6 glioma cells -- a task with subtle dose differences that challenges full-image models. We propose a simple tiling-and-aggregation pipeline that operates on local patches and combines tile outputs into an image label, achieving state-of-the-art accuracy on the benchmark dataset and improving over the published baseline by around 20 percentage points, with trends confirmed by cross-validation. To understand why tiling is effective, we further apply Grad-CAM and Score-CAM and attention analyses, which enhance model interpretability and point toward robustness-oriented directions for future medical image research. Code is released to facilitate reproduction and extension.
>
---
#### [new 182] FlowOpt: Fast Optimization Through Whole Flow Processes for Training-Free Editing
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文提出FlowOpt，一种无需梯度的快速优化框架，用于训练无关的图像编辑。针对扩散与流匹配模型采样过程迭代性导致梯度优化不实际的问题，将整个生成流程视为黑箱，实现端到端优化。方法高效且支持中途监控与停止，理论上保证收敛，并在图像反转与文本引导编辑任务中达SOTA效果。**

- **链接: [http://arxiv.org/pdf/2510.22010v1](http://arxiv.org/pdf/2510.22010v1)**

> **作者:** Or Ronai; Vladimir Kulikov; Tomer Michaeli
>
> **备注:** Project's webpage at https://orronai.github.io/FlowOpt/
>
> **摘要:** The remarkable success of diffusion and flow-matching models has ignited a surge of works on adapting them at test time for controlled generation tasks. Examples range from image editing to restoration, compression and personalization. However, due to the iterative nature of the sampling process in those models, it is computationally impractical to use gradient-based optimization to directly control the image generated at the end of the process. As a result, existing methods typically resort to manipulating each timestep separately. Here we introduce FlowOpt - a zero-order (gradient-free) optimization framework that treats the entire flow process as a black box, enabling optimization through the whole sampling path without backpropagation through the model. Our method is both highly efficient and allows users to monitor the intermediate optimization results and perform early stopping if desired. We prove a sufficient condition on FlowOpt's step-size, under which convergence to the global optimum is guaranteed. We further show how to empirically estimate this upper bound so as to choose an appropriate step-size. We demonstrate how FlowOpt can be used for image editing, showcasing two options: (i) inversion (determining the initial noise that generates a given image), and (ii) directly steering the edited image to be similar to the source image while conforming to a target text prompt. In both cases, FlowOpt achieves state-of-the-art results while using roughly the same number of neural function evaluations (NFEs) as existing methods. Code and examples are available on the project's webpage.
>
---
#### [new 183] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras
- **分类: cs.CV**

- **简介: 该论文提出InFlux基准，解决真实视频中相机内参动态变化缺乏标注的问题。针对3D视觉中内参恒定假设不成立的挑战，构建了包含143K+帧、386段高分辨率视频的动态内参数据集，提供逐帧真值标注，并改进校准工具以提升精度。**

- **链接: [http://arxiv.org/pdf/2510.23589v1](http://arxiv.org/pdf/2510.23589v1)**

> **作者:** Erich Liang; Roma Bhattacharjee; Sreemanti Dey; Rafael Moschopoulos; Caitlin Wang; Michel Liao; Grace Tan; Andrew Wang; Karhan Kayan; Stamatis Alexandropoulos; Jia Deng
>
> **备注:** Accepted at NeurIPS 2025 DB Track, Camera Ready Version. Supplementary material included
>
> **摘要:** Accurately tracking camera intrinsics is crucial for achieving 3D understanding from 2D video. However, most 3D algorithms assume that camera intrinsics stay constant throughout a video, which is often not true for many real-world in-the-wild videos. A major obstacle in this field is a lack of dynamic camera intrinsics benchmarks--existing benchmarks typically offer limited diversity in scene content and intrinsics variation, and none provide per-frame intrinsic changes for consecutive video frames. In this paper, we present Intrinsics in Flux (InFlux), a real-world benchmark that provides per-frame ground truth intrinsics annotations for videos with dynamic intrinsics. Compared to prior benchmarks, InFlux captures a wider range of intrinsic variations and scene diversity, featuring 143K+ annotated frames from 386 high-resolution indoor and outdoor videos with dynamic camera intrinsics. To ensure accurate per-frame intrinsics, we build a comprehensive lookup table of calibration experiments and extend the Kalibr toolbox to improve its accuracy and robustness. Using our benchmark, we evaluate existing baseline methods for predicting camera intrinsics and find that most struggle to achieve accurate predictions on videos with dynamic intrinsics. For the dataset, code, videos, and submission, please visit https://influx.cs.princeton.edu/.
>
---
#### [new 184] GateFuseNet: An Adaptive 3D Multimodal Neuroimaging Fusion Network for Parkinson's Disease Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对帕金森病（PD）的精准诊断任务，解决传统MRI方法对病理敏感性不足的问题。提出GateFuseNet，一种融合QSM与T1w图像的自适应3D多模态网络，通过门控融合模块实现特征选择与调制，提升关键区域识别能力，显著提高诊断准确率与AUC。**

- **链接: [http://arxiv.org/pdf/2510.22507v1](http://arxiv.org/pdf/2510.22507v1)**

> **作者:** Rui Jin; Chen Chen; Yin Liu; Hongfu Sun; Min Zeng; Min Li; Yang Gao
>
> **备注:** The first two authors contributed equally to this work. Correspondence to: Yang Gao, E-mail: yang.gao@csu.edu.cn
>
> **摘要:** Accurate diagnosis of Parkinson's disease (PD) from MRI remains challenging due to symptom variability and pathological heterogeneity. Most existing methods rely on conventional magnitude-based MRI modalities, such as T1-weighted images (T1w), which are less sensitive to PD pathology than Quantitative Susceptibility Mapping (QSM), a phase-based MRI technique that quantifies iron deposition in deep gray matter nuclei. In this study, we propose GateFuseNet, an adaptive 3D multimodal fusion network that integrates QSM and T1w images for PD diagnosis. The core innovation lies in a gated fusion module that learns modality-specific attention weights and channel-wise gating vectors for selective feature modulation. This hierarchical gating mechanism enhances ROI-aware features while suppressing irrelevant signals. Experimental results show that our method outperforms three existing state-of-the-art approaches, achieving 85.00% accuracy and 92.06% AUC. Ablation studies further validate the contributions of ROI guidance, multimodal integration, and fusion positioning. Grad-CAM visualizations confirm the model's focus on clinically relevant pathological regions. The source codes and pretrained models can be found at https://github.com/YangGaoUQ/GateFuseNet
>
---
#### [new 185] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans
- **分类: cs.CV**

- **简介: 该论文提出MiCADangelo，解决3D扫描到参数化CAD模型的逆向工程问题。针对现有方法难以保留精细几何与约束的问题，提出基于多平面截面的细粒度重建方法，首次将草图约束融入重建过程，实现更精确、可编辑的CAD模型生成。**

- **链接: [http://arxiv.org/pdf/2510.23429v1](http://arxiv.org/pdf/2510.23429v1)**

> **作者:** Ahmet Serdar Karadeniz; Dimitrios Mallis; Danila Rukhovich; Kseniya Cherenkova; Anis Kacem; Djamila Aouada
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Computer-Aided Design (CAD) plays a foundational role in modern manufacturing and product development, often requiring designers to modify or build upon existing models. Converting 3D scans into parametric CAD representations--a process known as CAD reverse engineering--remains a significant challenge due to the high precision and structural complexity of CAD models. Existing deep learning-based approaches typically fall into two categories: bottom-up, geometry-driven methods, which often fail to produce fully parametric outputs, and top-down strategies, which tend to overlook fine-grained geometric details. Moreover, current methods neglect an essential aspect of CAD modeling: sketch-level constraints. In this work, we introduce a novel approach to CAD reverse engineering inspired by how human designers manually perform the task. Our method leverages multi-plane cross-sections to extract 2D patterns and capture fine parametric details more effectively. It enables the reconstruction of detailed and editable CAD models, outperforming state-of-the-art methods and, for the first time, incorporating sketch constraints directly into the reconstruction process.
>
---
#### [new 186] Ageing Drift in Binary Face Templates: A Bits-per-Decade Analysis
- **分类: cs.CV; 68T45, 68T10, 62H35**

- **简介: 该论文研究人脸二值模板随时间的稳定性，量化年龄漂移为比特/十年。针对64位和128位编码，通过线性模型分析汉明距离与年龄差关系，发现短码更抗老化。结合EER、TPR评估性能，提出定期重注册等缓解策略，适用于智能卡部署。**

- **链接: [http://arxiv.org/pdf/2510.21778v1](http://arxiv.org/pdf/2510.21778v1)**

> **作者:** Abdelilah Ganmati; Karim Afdel; Lahcen Koutti
>
> **备注:** 9 pages, 3 figures, 2 tables
>
> **摘要:** We study the longitudinal stability of compact binary face templates and quantify ageing drift directly in bits per decade. Float embeddings from a modern face CNN are compressed with PCA-ITQ into 64- and 128-bit codes. For each identity in AgeDB with at least three distinct ages, we form all genuine pairs and fit a per-identity linear model of Hamming distance versus absolute age gap. Across 566 identities, the median slope is 1.357 bits per decade for 64-bit templates and 2.571 bits per decade for 128-bit templates, with tight non-parametric 95 percent bootstrap confidence intervals. The distributions are predominantly positive, indicating a small but systematic increase in intra-class distance over time. Because drift scales with code length, shorter codes are inherently more age-stable at a fixed decision threshold. We connect these slopes to operating characteristics by reporting EER and TPR at FAR = 1 percent in three age bins. We discuss implications for smart-card and match-on-card deployments, including simple mitigations such as periodic re-enrolment and targeted parity on empirically unstable bit positions. Code and CSV artifacts are provided to support reproducibility.
>
---
#### [new 187] CURVETE: Curriculum Learning and Progressive Self-supervised Training for Medical Image Classification
- **分类: cs.CV**

- **简介: 该论文提出CURVETE模型，针对医疗图像分类中标注样本少、类别分布不均的问题。通过课程学习与渐进自监督训练，提升模型泛化能力与分类性能，在多个医学图像数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.23442v1](http://arxiv.org/pdf/2510.23442v1)**

> **作者:** Asmaa Abbas; Mohamed Gaber; Mohammed M. Abdelsamea
>
> **备注:** Accepted for publication in the proceedings of ICONIP 2025
>
> **摘要:** Identifying high-quality and easily accessible annotated samples poses a notable challenge in medical image analysis. Transfer learning techniques, leveraging pre-training data, offer a flexible solution to this issue. However, the impact of fine-tuning diminishes when the dataset exhibits an irregular distribution between classes. This paper introduces a novel deep convolutional neural network, named Curriculum Learning and Progressive Self-supervised Training (CURVETE). CURVETE addresses challenges related to limited samples, enhances model generalisability, and improves overall classification performance. It achieves this by employing a curriculum learning strategy based on the granularity of sample decomposition during the training of generic unlabelled samples. Moreover, CURVETE address the challenge of irregular class distribution by incorporating a class decomposition approach in the downstream task. The proposed method undergoes evaluation on three distinct medical image datasets: brain tumour, digital knee x-ray, and Mini-DDSM datasets. We investigate the classification performance using a generic self-supervised sample decomposition approach with and without the curriculum learning component in training the pretext task. Experimental results demonstrate that the CURVETE model achieves superior performance on test sets with an accuracy of 96.60% on the brain tumour dataset, 75.60% on the digital knee x-ray dataset, and 93.35% on the Mini-DDSM dataset using the baseline ResNet-50. Furthermore, with the baseline DenseNet-121, it achieved accuracies of 95.77%, 80.36%, and 93.22% on the brain tumour, digital knee x-ray, and Mini-DDSM datasets, respectively, outperforming other training strategies.
>
---
#### [new 188] Accurate and Scalable Multimodal Pathology Retrieval via Attentive Vision-Language Alignment
- **分类: cs.CV; cs.IR**

- **简介: 该论文提出PathSearch框架，解决数字病理全切片图像（WSI）检索中因分辨率高、语义复杂导致的精准匹配难题。通过视觉-语言对比学习实现细粒度图块与全局图像嵌入的对齐，支持图像与文本双模态检索，显著提升诊断准确性与一致性。**

- **链接: [http://arxiv.org/pdf/2510.23224v1](http://arxiv.org/pdf/2510.23224v1)**

> **作者:** Hongyi Wang; Zhengjie Zhu; Jiabo Ma; Fang Wang; Yue Shi; Bo Luo; Jili Wang; Qiuyu Cai; Xiuming Zhang; Yen-Wei Chen; Lanfen Lin; Hao Chen
>
> **摘要:** The rapid digitization of histopathology slides has opened up new possibilities for computational tools in clinical and research workflows. Among these, content-based slide retrieval stands out, enabling pathologists to identify morphologically and semantically similar cases, thereby supporting precise diagnoses, enhancing consistency across observers, and assisting example-based education. However, effective retrieval of whole slide images (WSIs) remains challenging due to their gigapixel scale and the difficulty of capturing subtle semantic differences amid abundant irrelevant content. To overcome these challenges, we present PathSearch, a retrieval framework that unifies fine-grained attentive mosaic representations with global-wise slide embeddings aligned through vision-language contrastive learning. Trained on a corpus of 6,926 slide-report pairs, PathSearch captures both fine-grained morphological cues and high-level semantic patterns to enable accurate and flexible retrieval. The framework supports two key functionalities: (1) mosaic-based image-to-image retrieval, ensuring accurate and efficient slide research; and (2) multi-modal retrieval, where text queries can directly retrieve relevant slides. PathSearch was rigorously evaluated on four public pathology datasets and three in-house cohorts, covering tasks including anatomical site retrieval, tumor subtyping, tumor vs. non-tumor discrimination, and grading across diverse organs such as breast, lung, kidney, liver, and stomach. External results show that PathSearch outperforms traditional image-to-image retrieval frameworks. A multi-center reader study further demonstrates that PathSearch improves diagnostic accuracy, boosts confidence, and enhances inter-observer agreement among pathologists in real clinical scenarios. These results establish PathSearch as a scalable and generalizable retrieval solution for digital pathology.
>
---
#### [new 189] Beyond Augmentation: Leveraging Inter-Instance Relation in Self-Supervised Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于自监督表示学习任务，旨在解决传统方法忽略样本间关系的问题。提出基于图神经网络的KNN图建模，同时捕捉实例内与实例间关系，在预训练后通过多跳消息传递进行表示精炼，显著提升分类准确率。**

- **链接: [http://arxiv.org/pdf/2510.22322v1](http://arxiv.org/pdf/2510.22322v1)**

> **作者:** Ali Javidani; Babak Nadjar Araabi; Mohammad Amin Sadeghi
>
> **备注:** Accepted in IEEE Signal Processing Letters, 2025
>
> **摘要:** This paper introduces a novel approach that integrates graph theory into self-supervised representation learning. Traditional methods focus on intra-instance variations generated by applying augmentations. However, they often overlook important inter-instance relationships. While our method retains the intra-instance property, it further captures inter-instance relationships by constructing k-nearest neighbor (KNN) graphs for both teacher and student streams during pretraining. In these graphs, nodes represent samples along with their latent representations. Edges encode the similarity between instances. Following pretraining, a representation refinement phase is performed. In this phase, Graph Neural Networks (GNNs) propagate messages not only among immediate neighbors but also across multiple hops, thereby enabling broader contextual integration. Experimental results on CIFAR-10, ImageNet-100, and ImageNet-1K demonstrate accuracy improvements of 7.3%, 3.2%, and 1.0%, respectively, over state-of-the-art methods. These results highlight the effectiveness of the proposed graph based mechanism. The code is publicly available at https://github.com/alijavidani/SSL-GraphNNCLR.
>
---
#### [new 190] Embodied Navigation with Auxiliary Task of Action Description Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究多模态机器人室内导航任务，针对强化学习决策系统缺乏可解释性的问题，提出将动作语言描述作为辅助任务。通过知识蒸馏融合预训练视觉-语言模型，实现高精度导航与自然语言描述的协同优化，在语义视听导航任务中达到领先性能。**

- **链接: [http://arxiv.org/pdf/2510.21809v1](http://arxiv.org/pdf/2510.21809v1)**

> **作者:** Haru Kondoh; Asako Kanezaki
>
> **备注:** ICCV 2025 Poster
>
> **摘要:** The field of multimodal robot navigation in indoor environments has garnered significant attention in recent years. However, as tasks and methods become more advanced, the action decision systems tend to become more complex and operate as black-boxes. For a reliable system, the ability to explain or describe its decisions is crucial; however, there tends to be a trade-off in that explainable systems can not outperform non-explainable systems in terms of performance. In this paper, we propose incorporating the task of describing actions in language into the reinforcement learning of navigation as an auxiliary task. Existing studies have found it difficult to incorporate describing actions into reinforcement learning due to the absence of ground-truth data. We address this issue by leveraging knowledge distillation from pre-trained description generation models, such as vision-language models. We comprehensively evaluate our approach across various navigation tasks, demonstrating that it can describe actions while attaining high navigation performance. Furthermore, it achieves state-of-the-art performance in the particularly challenging multimodal navigation task of semantic audio-visual navigation.
>
---
#### [new 191] Accident Anticipation via Temporal Occurrence Prediction
- **分类: cs.CV**

- **简介: 该论文聚焦交通事故提前预警任务，针对现有方法依赖模糊二值标签导致误报的问题，提出基于未来多时步事故评分预测的新范式。通过片段编码器与Transformer解码器联合建模时空动态，并利用精确的事故时间标注进行监督，显著提升预警准确性和时效性。**

- **链接: [http://arxiv.org/pdf/2510.22260v1](http://arxiv.org/pdf/2510.22260v1)**

> **作者:** Tianhao Zhao; Yiyang Zou; Zihao Mao; Peilun Xiao; Yulin Huang; Hongda Yang; Yuxuan Li; Qun Li; Guobin Wu; Yutian Lin
>
> **备注:** Accepted by NIPS 2025
>
> **摘要:** Accident anticipation aims to predict potential collisions in an online manner, enabling timely alerts to enhance road safety. Existing methods typically predict frame-level risk scores as indicators of hazard. However, these approaches rely on ambiguous binary supervision (labeling all frames in accident videos as positive) despite the fact that risk varies continuously over time, leading to unreliable learning and false alarms. To address this, we propose a novel paradigm that shifts the prediction target from current-frame risk scoring to directly estimating accident scores at multiple future time steps (e.g., 0.1s-2.0s ahead), leveraging precisely annotated accident timestamps as supervision. Our method employs a snippet-level encoder to jointly model spatial and temporal dynamics, and a Transformer-based temporal decoder that predicts accident scores for all future horizons simultaneously using dedicated temporal queries. Furthermore, we introduce a refined evaluation protocol that reports Time-to-Accident (TTA) and recall (evaluated at multiple pre-accident intervals (0.5s, 1.0s, and 1.5s)) only when the false alarm rate (FAR) remains within an acceptable range, ensuring practical relevance. Experiments show that our method achieves superior performance in both recall and TTA under realistic FAR constraints.
>
---
#### [new 192] SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency
- **分类: cs.CV**

- **简介: 该论文提出SceneDecorator，面向场景导向的故事生成任务，解决场景规划与场景一致性难题。通过VLM引导的全局到局部场景规划和长期场景共享注意力机制，实现跨场景叙事连贯性与多样性，无需训练即可提升图像生成在艺术、影视、游戏等领域的创造力。**

- **链接: [http://arxiv.org/pdf/2510.22994v1](http://arxiv.org/pdf/2510.22994v1)**

> **作者:** Quanjian Song; Donghao Zhou; Jingyu Lin; Fei Shen; Jiaze Wang; Xiaowei Hu; Cunjian Chen; Pheng-Ann Heng
>
> **备注:** Accepted by NeurIPS 2025; Project Page: https://lulupig12138.github.io/SceneDecorator
>
> **摘要:** Recent text-to-image models have revolutionized image generation, but they still struggle with maintaining concept consistency across generated images. While existing works focus on character consistency, they often overlook the crucial role of scenes in storytelling, which restricts their creativity in practice. This paper introduces scene-oriented story generation, addressing two key challenges: (i) scene planning, where current methods fail to ensure scene-level narrative coherence by relying solely on text descriptions, and (ii) scene consistency, which remains largely unexplored in terms of maintaining scene consistency across multiple stories. We propose SceneDecorator, a training-free framework that employs VLM-Guided Scene Planning to ensure narrative coherence across different scenes in a ``global-to-local'' manner, and Long-Term Scene-Sharing Attention to maintain long-term scene consistency and subject diversity across generated stories. Extensive experiments demonstrate the superior performance of SceneDecorator, highlighting its potential to unleash creativity in the fields of arts, films, and games.
>
---
#### [new 193] Benchmarking Egocentric Multimodal Goal Inference for Assistive Wearable Agents
- **分类: cs.CV; cs.LG**

- **简介: 该论文聚焦于穿戴式助手机器人中的自我中心多模态目标推理任务，旨在通过视觉、音频等多源数据自动推断用户潜在目标，减少交互负担。作者构建了WAGIBench基准数据集，包含3477条记录，验证了人类优于模型的表现，并发现大模型虽性能更优但仍不实用，且多模态信息有效提升推理效果。**

- **链接: [http://arxiv.org/pdf/2510.22443v1](http://arxiv.org/pdf/2510.22443v1)**

> **作者:** Vijay Veerabadran; Fanyi Xiao; Nitin Kamra; Pedro Matias; Joy Chen; Caley Drooff; Brett D Roads; Riley Williams; Ethan Henderson; Xuanyi Zhao; Kevin Carlberg; Joseph Tighe; Karl Ridgeway
>
> **备注:** Accepted as a spotlight paper at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** There has been a surge of interest in assistive wearable agents: agents embodied in wearable form factors (e.g., smart glasses) who take assistive actions toward a user's goal/query (e.g. "Where did I leave my keys?"). In this work, we consider the important complementary problem of inferring that goal from multi-modal contextual observations. Solving this "goal inference" problem holds the promise of eliminating the effort needed to interact with such an agent. This work focuses on creating WAGIBench, a strong benchmark to measure progress in solving this problem using vision-language models (VLMs). Given the limited prior work in this area, we collected a novel dataset comprising 29 hours of multimodal data from 348 participants across 3,477 recordings, featuring ground-truth goals alongside accompanying visual, audio, digital, and longitudinal contextual observations. We validate that human performance exceeds model performance, achieving 93% multiple-choice accuracy compared with 84% for the best-performing VLM. Generative benchmark results that evaluate several families of modern vision-language models show that larger models perform significantly better on the task, yet remain far from practical usefulness, as they produce relevant goals only 55% of the time. Through a modality ablation, we show that models benefit from extra information in relevant modalities with minimal performance degradation from irrelevant modalities.
>
---
#### [new 194] VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型（VLM）复杂推理能力不足的问题，提出VOLD框架，通过强化学习与在策略蒸馏，将纯文本大模型的推理能力迁移至VLM。关键在于冷启动对齐，确保教师与学生分布一致，显著提升推理性能。**

- **链接: [http://arxiv.org/pdf/2510.23497v1](http://arxiv.org/pdf/2510.23497v1)**

> **作者:** Walid Bousselham; Hilde Kuehne; Cordelia Schmid
>
> **备注:** www.walidbousselham.com/VOLD/
>
> **摘要:** Training vision-language models (VLMs) for complex reasoning remains a challenging task, i.a. due to the scarcity of high-quality image-text reasoning data. Conversely, text-based reasoning resources are abundant and scalable, but it is still an open question how to leveraging them for VLM reasoning. To address this problem, we propose VOLD, a framework to transfer reasoning capabilities from text-only teacher models to VLM student models. To this end, VOLD combines reinforcement learning via Group Relative Policy Optimization (GRPO) with on-policy distillation, which allows the student reasoning traces to be guided by the teacher model, resulting in a significant gain over using GRPO alone. We further show that a cold-start alignment is essential for an effective transfer during the online training phase in this scenario and that without sufficient distributional alignment between teacher and student, on-policy distillation fails to provide meaningful guidance. We evaluate VOLD across diverse benchmarks including MMMU-Pro, MathVision, MathVista, and LogicVista, showing that VOLD outperforms the baseline model significantly and improves over the state of the art by a margin. Our ablation shows the importance of a cold-start alignment via SFT for on-policy distillation with a text-only teacher.
>
---
#### [new 195] GeoDiffusion: A Training-Free Framework for Accurate 3D Geometric Conditioning in Image Generation
- **分类: cs.CV**

- **简介: 该论文提出GeoDiffusion，一个无需训练的3D几何控制框架，用于图像生成中的精确几何调节。针对传统方法耗时且生成模型几何精度不足的问题，利用3D对象作为几何先验，结合视点一致性和风格迁移，实现高效精准的拖拽编辑。**

- **链接: [http://arxiv.org/pdf/2510.22337v1](http://arxiv.org/pdf/2510.22337v1)**

> **作者:** Phillip Mueller; Talip Uenlue; Sebastian Schmidt; Marcel Kollovieh; Jiajie Fan; Stephan Guennemann; Lars Mikelsons
>
> **摘要:** Precise geometric control in image generation is essential for engineering \& product design and creative industries to control 3D object features accurately in image space. Traditional 3D editing approaches are time-consuming and demand specialized skills, while current image-based generative methods lack accuracy in geometric conditioning. To address these challenges, we propose GeoDiffusion, a training-free framework for accurate and efficient geometric conditioning of 3D features in image generation. GeoDiffusion employs a class-specific 3D object as a geometric prior to define keypoints and parametric correlations in 3D space. We ensure viewpoint consistency through a rendered image of a reference 3D object, followed by style transfer to meet user-defined appearance specifications. At the core of our framework is GeoDrag, improving accuracy and speed of drag-based image editing on geometry guidance tasks and general instructions on DragBench. Our results demonstrate that GeoDiffusion enables precise geometric modifications across various iterative design workflows.
>
---
#### [new 196] PSScreen V2: Partially Supervised Multiple Retinal Disease Screening
- **分类: cs.CV**

- **简介: 该论文提出PSScreen V2，用于多疾病眼底图像筛查。针对标签缺失与域偏移问题，设计三分支自训练框架，通过伪标签生成与低频特征增强提升模型鲁棒性与泛化能力，实现部分标注数据下的高性能跨域诊断。**

- **链接: [http://arxiv.org/pdf/2510.22589v1](http://arxiv.org/pdf/2510.22589v1)**

> **作者:** Boyi Zheng; Yalin Zheng; Hrvoje Bogunović; Qing Liu
>
> **摘要:** In this work, we propose PSScreen V2, a partially supervised self-training framework for multiple retinal disease screening. Unlike previous methods that rely on fully labelled or single-domain datasets, PSScreen V2 is designed to learn from multiple partially labelled datasets with different distributions, addressing both label absence and domain shift challenges. To this end, PSScreen V2 adopts a three-branch architecture with one teacher and two student networks. The teacher branch generates pseudo labels from weakly augmented images to address missing labels, while the two student branches introduce novel feature augmentation strategies: Low-Frequency Dropout (LF-Dropout), which enhances domain robustness by randomly discarding domain-related low-frequency components, and Low-Frequency Uncertainty (LF-Uncert), which estimates uncertain domain variability via adversarially learned Gaussian perturbations of low-frequency statistics. Extensive experiments on multiple in-domain and out-of-domain fundus datasets demonstrate that PSScreen V2 achieves state-of-the-art performance and superior domain generalization ability. Furthermore, compatibility tests with diverse backbones, including the vision foundation model DINOv2, as well as evaluations on chest X-ray datasets, highlight the universality and adaptability of the proposed framework. The codes are available at https://github.com/boyiZheng99/PSScreen_V2.
>
---
#### [new 197] SWAN: Self-supervised Wavelet Neural Network for Hyperspectral Image Unmixing
- **分类: cs.CV**

- **简介: 该论文提出SWAN模型，用于高光谱图像解混任务，旨在无需真实标签的情况下联合估计端元和丰度。通过自监督波尔特神经网络，在小波域中提取多尺度特征，利用三阶段架构实现高效解混，显著提升性能并减少参数量。**

- **链接: [http://arxiv.org/pdf/2510.22607v1](http://arxiv.org/pdf/2510.22607v1)**

> **作者:** Yassh Ramchandani; Vijayashekhar S S; Jignesh S. Bhatt
>
> **摘要:** In this article, we present SWAN: a three-stage, self-supervised wavelet neural network for joint estimation of endmembers and abundances from hyperspectral imagery. The contiguous and overlapping hyperspectral band images are first expanded to Biorthogonal wavelet basis space that provides sparse, distributed, and multi-scale representations. The idea is to exploit latent symmetries from thus obtained invariant and covariant features using a self-supervised learning paradigm. The first stage, SWANencoder maps the input wavelet coefficients to a compact lower-dimensional latent space. The second stage, SWANdecoder uses the derived latent representation to reconstruct the input wavelet coefficients. Interestingly, the third stage SWANforward learns the underlying physics of the hyperspectral image. A three-stage combined loss function is formulated in the image acquisition domain that eliminates the need for ground truth and enables self-supervised training. Adam is employed for optimizing the proposed loss function, while Sigmoid with a dropout of 0.3 is incorporated to avoid possible overfitting. Kernel regularizers bound the magnitudes and preserve spatial variations in the estimated endmember coefficients. The output of SWANencoder represents estimated abundance maps during inference, while weights of SWANdecoder are retrieved to extract endmembers. Experiments are conducted on two benchmark synthetic data sets with different signal-to-noise ratios as well as on three real benchmark hyperspectral data sets while comparing the results with several state-of-the-art neural network-based unmixing methods. The qualitative, quantitative, and ablation results show performance enhancement by learning a resilient unmixing function as well as promoting self-supervision and compact network parameters for practical applications.
>
---
#### [new 198] SITS-DECO: A Generative Decoder Is All You Need For Multitask Satellite Image Time Series Modelling
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出SITS-DECO，一种基于生成式解码器的多任务卫星图像时序建模方法。针对现有模型结构僵化、需额外适配的问题，利用统一序列框架实现无需任务特定设计的多模态、多任务学习，仅通过符号提示即可完成分类等任务，验证了时间序列建模的重要性与数据驱动范式的有效性。**

- **链接: [http://arxiv.org/pdf/2510.21813v1](http://arxiv.org/pdf/2510.21813v1)**

> **作者:** Samuel J. Barrett; Docko Sow
>
> **备注:** 27 pages, 7 figures
>
> **摘要:** Earth Observation (EO) Foundation Modelling (FM) holds great promise for simplifying and improving the use of EO data for diverse real-world tasks. However, most existing models require additional adaptation before they can be used and are structured rigidly around particular data sources or training approaches. To address this, we take inspiration from large language models, where diverse tasks, both pre-training and downstream, are implicitly captured through next-token prediction over unified token sequences, leveraging the structure and diversity of the training data. We introduce SITS-DECO (Satellite Image Time Series-DECoder Only), a proof-of-concept generative model that applies this unified-sequence framing to EO data. Using a simple GPT-style decoder-only architecture, and demonstrate its ability to perform useful EO tasks (pixel-wise, multi-temporal, multi-modal crop-type classification) in a purely generative framework. Through symbolic prompting, we show that the model can perform multiple supervised and self-supervised tasks within a single unified architecture, without task- or modality-specific adaptation. Despite its simplicity and lack of spatial context, SITS-DECO outperforms much larger EO foundation models on crop-type classification (PASTIS-R) demonstrating that dense temporal sequence modelling is a critical missing ingredient in the current paradigm. This work exemplifies a data-centric modelling paradigm in which capability arises from the diversity and structure of the training data rather than from architectural complexity. SITS-DECO provides a lightweight, practical route to multi-modal, multi-task EO modelling, and a conceptual bridge toward future generative EO foundation models.
>
---
#### [new 199] HARMONY: Hidden Activation Representations and Model Output-Aware Uncertainty Estimation for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型（VLMs）在高风险应用中输出不可靠的问题，提出HARMONY框架，联合利用模型隐藏层表示与输出概率分布进行不确定性估计。通过融合多模态内部信念与生成概率，提升可靠性判断准确率，在多个基准上达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2510.22171v1](http://arxiv.org/pdf/2510.22171v1)**

> **作者:** Erum Mushtaq; Zalan Fabian; Yavuz Faruk Bakman; Anil Ramakrishna; Mahdi Soltanolkotabi; Salman Avestimehr
>
> **摘要:** The growing deployment of Vision-Language Models (VLMs) in high-stakes applications such as autonomous driving and assistive technologies for visually impaired individuals necessitates reliable mechanisms to assess the trustworthiness of their generation. Uncertainty Estimation (UE) plays a central role in quantifying the reliability of model outputs and reducing unsafe generations via selective prediction. In this regard, most existing probability-based UE approaches rely on output probability distributions, aggregating token probabilities into a single uncertainty score using predefined functions such as length-normalization. Another line of research leverages model hidden representations and trains MLP-based models to predict uncertainty. However, these methods often fail to capture the complex multimodal relationships between semantic and textual tokens and struggle to identify biased probabilities often influenced by language priors. Motivated by these observations, we propose a novel UE framework, HARMONY, that jointly leverages fused multimodal information in model activations and the output distribution of the VLM to determine the reliability of responses. The key hypothesis of our work is that both the model's internal belief in its visual understanding, captured by its hidden representations, and the produced token probabilities carry valuable reliability signals that can be jointly leveraged to improve UE performance, surpassing approaches that rely on only one of these components. Experimental results on three open-ended VQA benchmarks, A-OKVQA, VizWiz, and PathVQA, and three state-of-the-art VLMs, LLaVa-7b, LLaVA-13b and InstructBLIP demonstrate that our method consistently performs on par with or better than existing approaches, achieving up to 4\% improvement in AUROC, and 6\% in PRR, establishing new state of the art in uncertainty estimation for VLMs.
>
---
#### [new 200] OCR-Quality: A Human-Annotated Dataset for OCR Quality Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出OCR-Quality数据集，用于评估光学字符识别（OCR）质量。针对真实场景下OCR质量评估缺乏可靠基准的问题，研究者收集1000页多类型文档，经VLM处理并人工标注四级质量评分，提供详实注释与案例，为训练和评测OCR验证系统提供公开基准。**

- **链接: [http://arxiv.org/pdf/2510.21774v1](http://arxiv.org/pdf/2510.21774v1)**

> **作者:** Yulong Zhang
>
> **摘要:** We present OCR-Quality, a comprehensive human-annotated dataset designed for evaluating and developing OCR quality assessment methods. The dataset consists of 1,000 PDF pages converted to PNG images at 300 DPI, sampled from diverse real-world scenarios, including academic papers, textbooks, e-books, and multilingual documents. Each document has been processed using state-of-the-art Vision-Language Models (VLMs) and manually annotated with quality scores using a 4-level scoring system (1: Excellent, 2: Good, 3: Fair, 4: Poor). The dataset includes detailed source information, annotation guidelines, and representative cases across various difficulty levels. OCR-Quality addresses the critical need for reliable OCR quality assessment in real-world applications and provides a valuable benchmark for training and evaluating OCR verification systems. The dataset is publicly available at https://huggingface.co/datasets/Aslan-mingye/OCR-Quality .
>
---
#### [new 201] Autoregressive Styled Text Image Generation, but Make it Reliable
- **分类: cs.CV**

- **简介: 该论文聚焦于手写文本图像生成任务，针对现有自回归模型依赖额外输入、缺乏停止机制及易陷入重复等问题，提出基于多模态提示的可靠生成框架Eruku。通过引入特殊文本标记增强内容控制，并采用无分类器引导策略，显著提升生成内容与提示的一致性及对未见风格的泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.23240v1](http://arxiv.org/pdf/2510.23240v1)**

> **作者:** Carmine Zaccagnino; Fabio Quattrini; Vittorio Pippi; Silvia Cascianelli; Alessio Tonioni; Rita Cucchiara
>
> **摘要:** Generating faithful and readable styled text images (especially for Styled Handwritten Text generation - HTG) is an open problem with several possible applications across graphic design, document understanding, and image editing. A lot of research effort in this task is dedicated to developing strategies that reproduce the stylistic characteristics of a given writer, with promising results in terms of style fidelity and generalization achieved by the recently proposed Autoregressive Transformer paradigm for HTG. However, this method requires additional inputs, lacks a proper stop mechanism, and might end up in repetition loops, generating visual artifacts. In this work, we rethink the autoregressive formulation by framing HTG as a multimodal prompt-conditioned generation task, and tackle the content controllability issues by introducing special textual input tokens for better alignment with the visual ones. Moreover, we devise a Classifier-Free-Guidance-based strategy for our autoregressive model. Through extensive experimental validation, we demonstrate that our approach, dubbed Eruku, compared to previous solutions requires fewer inputs, generalizes better to unseen styles, and follows more faithfully the textual prompt, improving content adherence.
>
---
#### [new 202] FAME: Fairness-aware Attention-modulated Video Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对训练-free视频编辑中的职业性别偏见问题，提出FAME模型。通过引入公平性嵌入和注意力调制机制，增强视觉区域一致性与时间连贯性，有效缓解偏见并保持语义准确，显著提升公平性与生成质量。**

- **链接: [http://arxiv.org/pdf/2510.22960v1](http://arxiv.org/pdf/2510.22960v1)**

> **作者:** Zhangkai Wu; Xuhui Fan; Zhongyuan Xie; Kaize Shi; Zhidong Li; Longbing Cao
>
> **摘要:** Training-free video editing (VE) models tend to fall back on gender stereotypes when rendering profession-related prompts. We propose \textbf{FAME} for \textit{Fairness-aware Attention-modulated Video Editing} that mitigates profession-related gender biases while preserving prompt alignment and temporal consistency for coherent VE. We derive fairness embeddings from existing minority representations by softly injecting debiasing tokens into the text encoder. Simultaneously, FAME integrates fairness modulation into both temporal self attention and prompt-to-region cross attention to mitigate the motion corruption and temporal inconsistency caused by directly introducing fairness cues. For temporal self attention, FAME introduces a region constrained attention mask combined with time decay weighting, which enhances intra-region coherence while suppressing irrelevant inter-region interactions. For cross attention, it reweights tokens to region matching scores by incorporating fairness sensitive similarity masks derived from debiasing prompt embeddings. Together, these modulations keep fairness-sensitive semantics tied to the right visual regions and prevent temporal drift across frames. Extensive experiments on new VE fairness-oriented benchmark \textit{FairVE} demonstrate that FAME achieves stronger fairness alignment and semantic fidelity, surpassing existing VE baselines.
>
---
#### [new 203] LLM-based Fusion of Multi-modal Features for Commercial Memorability Prediction
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文参与MediaEval 2025“商业广告记忆度预测”任务，旨在提升广告记忆度预测性能。提出基于Gemma-3 LLM的多模态融合框架，结合视觉与文本特征，利用LLM生成的推理提示引导融合，并采用LoRA微调。实验表明该方法在测试集上优于梯度提升树基线，具备更强的鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.22829v1](http://arxiv.org/pdf/2510.22829v1)**

> **作者:** Aleksandar Pramov
>
> **摘要:** This paper addresses the prediction of commercial (brand) memorability as part of "Subtask 2: Commercial/Ad Memorability" within the "Memorability: Predicting movie and commercial memorability" task at the MediaEval 2025 workshop competition. We propose a multimodal fusion system with a Gemma-3 LLM backbone that integrates pre-computed visual (ViT) and textual (E5) features by multi-modal projections. The model is adapted using Low-Rank Adaptation (LoRA). A heavily-tuned ensemble of gradient boosted trees serves as a baseline. A key contribution is the use of LLM-generated rationale prompts, grounded in expert-derived aspects of memorability, to guide the fusion model. The results demonstrate that the LLM-based system exhibits greater robustness and generalization performance on the final test set, compared to the baseline. The paper's codebase can be found at https://github.com/dsgt-arc/mediaeval-2025-memorability
>
---
#### [new 204] Multi-Agent Pose Uncertainty: A Differentiable Rendering Cramér-Rao Bound
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文针对计算机视觉中的姿态估计任务，解决密集或学习模型下姿态不确定性量化难题。通过将可微渲染器视为测量函数，推导出基于流形扰动的可微分Cramér-Rao下界，实现姿态协方差的闭式下界计算，并自然扩展至多相机协同场景，无需关键点匹配即可用于协同感知与新视角合成。**

- **链接: [http://arxiv.org/pdf/2510.21785v1](http://arxiv.org/pdf/2510.21785v1)**

> **作者:** Arun Muthukkumar
>
> **备注:** 5 pages, 3 figures, 1 table. Presented at IEEE/CVF International Conference on Computer Vision (ICCV 2025) and IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Pose estimation is essential for many applications within computer vision and robotics. Despite its uses, few works provide rigorous uncertainty quantification for poses under dense or learned models. We derive a closed-form lower bound on the covariance of camera pose estimates by treating a differentiable renderer as a measurement function. Linearizing image formation with respect to a small pose perturbation on the manifold yields a render-aware Cram\'er-Rao bound. Our approach reduces to classical bundle-adjustment uncertainty, ensuring continuity with vision theory. It also naturally extends to multi-agent settings by fusing Fisher information across cameras. Our statistical formulation has downstream applications for tasks such as cooperative perception and novel view synthesis without requiring explicit keypoint correspondences.
>
---
#### [new 205] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对多模态行人重识别中模态不匹配问题，提出MDReID框架。通过解耦共享与特定模态特征，并引入模态感知度量学习，实现任意模态间的跨模态检索，在多种场景下显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.23301v1](http://arxiv.org/pdf/2510.23301v1)**

> **作者:** Yingying Feng; Jie Li; Jie Hu; Yukang Zhang; Lei Tan; Jiayi Ji
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Real-world object re-identification (ReID) systems often face modality inconsistencies, where query and gallery images come from different sensors (e.g., RGB, NIR, TIR). However, most existing methods assume modality-matched conditions, which limits their robustness and scalability in practical applications. To address this challenge, we propose MDReID, a flexible any-to-any image-level ReID framework designed to operate under both modality-matched and modality-mismatched scenarios. MDReID builds on the insight that modality information can be decomposed into two components: modality-shared features that are predictable and transferable, and modality-specific features that capture unique, modality-dependent characteristics. To effectively leverage this, MDReID introduces two key components: the Modality Decoupling Learning (MDL) and Modality-aware Metric Learning (MML). Specifically, MDL explicitly decomposes modality features into modality-shared and modality-specific representations, enabling effective retrieval in both modality-aligned and mismatched scenarios. MML, a tailored metric learning strategy, further enforces orthogonality and complementarity between the two components to enhance discriminative power across modalities. Extensive experiments conducted on three challenging multi-modality ReID benchmarks (RGBNT201, RGBNT100, MSVR310) consistently demonstrate the superiority of MDReID. Notably, MDReID achieves significant mAP improvements of 9.8\%, 3.0\%, and 11.5\% in general modality-matched scenarios, and average gains of 3.4\%, 11.8\%, and 10.9\% in modality-mismatched scenarios, respectively. The code is available at: \textcolor{magenta}{https://github.com/stone96123/MDReID}.
>
---
#### [new 206] LVD-GS: Gaussian Splatting SLAM for Dynamic Scenes via Hierarchical Explicit-Implicit Representation Collaboration Rendering
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LVD-GS，一种用于动态场景的激光-视觉3D高斯泼溅SLAM系统。针对大尺度动态室外场景中尺度漂移与重建不稳问题，引入分层显式-隐式协同表示与联合动态建模，通过多源信息融合提升精度与鲁棒性，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.22669v1](http://arxiv.org/pdf/2510.22669v1)**

> **作者:** Wenkai Zhu; Xu Li; Qimin Xu; Benwu Wang; Kun Wei; Yiming Peng; Zihang Wang
>
> **摘要:** 3D Gaussian Splatting SLAM has emerged as a widely used technique for high-fidelity mapping in spatial intelligence. However, existing methods often rely on a single representation scheme, which limits their performance in large-scale dynamic outdoor scenes and leads to cumulative pose errors and scale ambiguity. To address these challenges, we propose \textbf{LVD-GS}, a novel LiDAR-Visual 3D Gaussian Splatting SLAM system. Motivated by the human chain-of-thought process for information seeking, we introduce a hierarchical collaborative representation module that facilitates mutual reinforcement for mapping optimization, effectively mitigating scale drift and enhancing reconstruction robustness. Furthermore, to effectively eliminate the influence of dynamic objects, we propose a joint dynamic modeling module that generates fine-grained dynamic masks by fusing open-world segmentation with implicit residual constraints, guided by uncertainty estimates from DINO-Depth features. Extensive evaluations on KITTI, nuScenes, and self-collected datasets demonstrate that our approach achieves state-of-the-art performance compared to existing methods.
>
---
#### [new 207] UGAE: Unified Geometry and Attribute Enhancement for G-PCC Compressed Point Clouds
- **分类: cs.CV**

- **简介: 该论文针对点云压缩后的几何与属性失真问题，提出统一增强框架UGAE。通过后处理几何重建、预压缩细节保真的颜色恢复及解码端残差增强，显著提升压缩点云的精度与感知质量，优于现有G-PCC方法。**

- **链接: [http://arxiv.org/pdf/2510.23009v1](http://arxiv.org/pdf/2510.23009v1)**

> **作者:** Pan Zhao; Hui Yuan; Chongzhen Tian; Tian Guo; Raouf Hamzaoui; Zhigeng Pan
>
> **摘要:** Lossy compression of point clouds reduces storage and transmission costs; however, it inevitably leads to irreversible distortion in geometry structure and attribute information. To address these issues, we propose a unified geometry and attribute enhancement (UGAE) framework, which consists of three core components: post-geometry enhancement (PoGE), pre-attribute enhancement (PAE), and post-attribute enhancement (PoAE). In PoGE, a Transformer-based sparse convolutional U-Net is used to reconstruct the geometry structure with high precision by predicting voxel occupancy probabilities. Building on the refined geometry structure, PAE introduces an innovative enhanced geometry-guided recoloring strategy, which uses a detail-aware K-Nearest Neighbors (DA-KNN) method to achieve accurate recoloring and effectively preserve high-frequency details before attribute compression. Finally, at the decoder side, PoAE uses an attribute residual prediction network with a weighted mean squared error (W-MSE) loss to enhance the quality of high-frequency regions while maintaining the fidelity of low-frequency regions. UGAE significantly outperformed existing methods on three benchmark datasets: 8iVFB, Owlii, and MVUB. Compared to the latest G-PCC test model (TMC13v29), UGAE achieved an average BD-PSNR gain of 9.98 dB and 90.98% BD-bitrate savings for geometry under the D1 metric, as well as a 3.67 dB BD-PSNR improvement with 56.88% BD-bitrate savings for attributes on the Y component. Additionally, it improved perceptual quality significantly.
>
---
#### [new 208] LSF-Animation: Label-Free Speech-Driven Facial Animation via Implicit Feature Representation
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出LSF-Animation，用于无标签语音驱动的3D人脸动画。针对现有方法依赖显式身份与情感标签、泛化能力差的问题，提出隐式特征表示，从语音中提取情感信息，从中性面部网格捕捉身份特征，并引入层次交互融合块，提升动画真实感与跨说话人泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.21864v1](http://arxiv.org/pdf/2510.21864v1)**

> **作者:** Xin Lu; Chuanqing Zhuang; Chenxi Jin; Zhengda Lu; Yiqun Wang; Wu Liu; Jun Xiao
>
> **摘要:** Speech-driven 3D facial animation has attracted increasing interest since its potential to generate expressive and temporally synchronized digital humans. While recent works have begun to explore emotion-aware animation, they still depend on explicit one-hot encodings to represent identity and emotion with given emotion and identity labels, which limits their ability to generalize to unseen speakers. Moreover, the emotional cues inherently present in speech are often neglected, limiting the naturalness and adaptability of generated animations. In this work, we propose LSF-Animation, a novel framework that eliminates the reliance on explicit emotion and identity feature representations. Specifically, LSF-Animation implicitly extracts emotion information from speech and captures the identity features from a neutral facial mesh, enabling improved generalization to unseen speakers and emotional states without requiring manual labels. Furthermore, we introduce a Hierarchical Interaction Fusion Block (HIFB), which employs a fusion token to integrate dual transformer features and more effectively integrate emotional, motion-related and identity-related cues. Extensive experiments conducted on the 3DMEAD dataset demonstrate that our method surpasses recent state-of-the-art approaches in terms of emotional expressiveness, identity generalization, and animation realism. The source code will be released at: https://github.com/Dogter521/LSF-Animation.
>
---
#### [new 209] VALA: Learning Latent Anchors for Training-Free and Temporally Consistent
- **分类: cs.CV**

- **简介: 该论文针对训练-free视频编辑中的时序不一致问题，提出VALA方法。通过变分对齐机制自适应选择关键帧并生成语义锚点，利用对比学习优化特征压缩，实现跨帧内容一致性与高效编辑。**

- **链接: [http://arxiv.org/pdf/2510.22970v1](http://arxiv.org/pdf/2510.22970v1)**

> **作者:** Zhangkai Wu; Xuhui Fan; Zhongyuan Xie; Kaize Shi; Longbing Cao
>
> **摘要:** Recent advances in training-free video editing have enabled lightweight and precise cross-frame generation by leveraging pre-trained text-to-image diffusion models. However, existing methods often rely on heuristic frame selection to maintain temporal consistency during DDIM inversion, which introduces manual bias and reduces the scalability of end-to-end inference. In this paper, we propose~\textbf{VALA} (\textbf{V}ariational \textbf{A}lignment for \textbf{L}atent \textbf{A}nchors), a variational alignment module that adaptively selects key frames and compresses their latent features into semantic anchors for consistent video editing. To learn meaningful assignments, VALA propose a variational framework with a contrastive learning objective. Therefore, it can transform cross-frame latent representations into compressed latent anchors that preserve both content and temporal coherence. Our method can be fully integrated into training-free text-to-image based video editing models. Extensive experiments on real-world video editing benchmarks show that VALA achieves state-of-the-art performance in inversion fidelity, editing quality, and temporal consistency, while offering improved efficiency over prior methods.
>
---
#### [new 210] More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出MERGE模型，统一图像生成与深度估计任务。针对预训练扩散模型微调后生成能力退化的问题，提出可插拔转换器与分组复用机制，实现两种模式无缝切换，既保留原生成能力，又提升深度估计性能，达到当前最优效果。**

- **链接: [http://arxiv.org/pdf/2510.23574v1](http://arxiv.org/pdf/2510.23574v1)**

> **作者:** Hongkai Lin; Dingkang Liang; Mingyang Du; Xin Zhou; Xiang Bai
>
> **备注:** Accepted by NeurIPS 2025. The code will be made available at https://github.com/H-EmbodVis/MERGE
>
> **摘要:** Generative depth estimation methods leverage the rich visual priors stored in pre-trained text-to-image diffusion models, demonstrating astonishing zero-shot capability. However, parameter updates during training lead to catastrophic degra- dation in the image generation capability of the pre-trained model. We introduce MERGE, a unified model for image generation and depth estimation, starting from a fixed pre-trained text-to-image model. MERGE demonstrates that the pre-trained text-to-image model can do more than image generation, but also expand to depth estimation effortlessly. Specifically, MERGE introduces a play- and-plug framework that enables seamless switching between image generation and depth estimation modes through simple and pluggable converters. Meanwhile, we propose a Group Reuse Mechanism to encourage parameter reuse and im- prove the utilization of the additional learnable parameters. MERGE unleashes the powerful depth estimation capability of the pre-trained text-to-image model while preserving its original image generation ability. Compared to other unified models for image generation and depth estimation, MERGE achieves state-of- the-art performance across multiple depth estimation benchmarks. The code will be made available at https://github.com/H-EmbodVis/MERGE
>
---
#### [new 211] MAGIC-Flow: Multiscale Adaptive Conditional Flows for Generation and Interpretable Classification
- **分类: cs.LG; cs.CV; eess.IV; stat.ML**

- **简介: 该论文提出MAGIC-Flow，一种用于医学影像的生成与可解释分类的条件多尺度归一化流模型。针对生成模型在临床应用中缺乏任务对齐的问题，通过可逆变换实现精确似然计算与样本可视化，同时支持可控生成与分类，提升数据有限场景下的生成质量与分类性能。**

- **链接: [http://arxiv.org/pdf/2510.22070v1](http://arxiv.org/pdf/2510.22070v1)**

> **作者:** Luca Caldera; Giacomo Bottacini; Lara Cavinato
>
> **摘要:** Generative modeling has emerged as a powerful paradigm for representation learning, but its direct applicability to challenging fields like medical imaging remains limited: mere generation, without task alignment, fails to provide a robust foundation for clinical use. We propose MAGIC-Flow, a conditional multiscale normalizing flow architecture that performs generation and classification within a single modular framework. The model is built as a hierarchy of invertible and differentiable bijections, where the Jacobian determinant factorizes across sub-transformations. We show how this ensures exact likelihood computation and stable optimization, while invertibility enables explicit visualization of sample likelihoods, providing an interpretable lens into the model's reasoning. By conditioning on class labels, MAGIC-Flow supports controllable sample synthesis and principled class-probability estimation, effectively aiding both generative and discriminative objectives. We evaluate MAGIC-Flow against top baselines using metrics for similarity, fidelity, and diversity. Across multiple datasets, it addresses generation and classification under scanner noise, and modality-specific synthesis and identification. Results show MAGIC-Flow creates realistic, diverse samples and improves classification. MAGIC-Flow is an effective strategy for generation and classification in data-limited domains, with direct benefits for privacy-preserving augmentation, robust generalization, and trustworthy medical AI.
>
---
#### [new 212] Seeing Structural Failure Before it Happens: An Image-Based Physics-Informed Neural Network (PINN) for Spaghetti Bridge Load Prediction
- **分类: cs.LG; cs.CV; 65M70 (Primary), 68T07 (Secondary); I.2.6; I.4.8; G.1.8**

- **简介: 该论文针对小尺度意大利面桥的承重预测任务，旨在通过物理信息神经网络（PINN）结合计算机视觉与结构参数，提升有限数据下的预测精度。提出新型PIKAN架构，融合物理规律与函数逼近理论，实现高准确率预测（R²=0.9603），并开发在线预测界面，助力早期结构失效分析。**

- **链接: [http://arxiv.org/pdf/2510.23117v1](http://arxiv.org/pdf/2510.23117v1)**

> **作者:** Omer Jauhar Khan; Sudais Khan; Hafeez Anwar
>
> **备注:** 12 pages, 17 figures. Preprint
>
> **摘要:** Physics Informed Neural Networks (PINNs) are gaining attention for their ability to embed physical laws into deep learning models, which is particularly useful in structural engineering tasks with limited data. This paper aims to explore the use of PINNs to predict the weight of small scale spaghetti bridges, a task relevant to understanding load limits and potential failure modes in simplified structural models. Our proposed framework incorporates physics-based constraints to the prediction model for improved performance. In addition to standard PINNs, we introduce a novel architecture named Physics Informed Kolmogorov Arnold Network (PIKAN), which blends universal function approximation theory with physical insights. The structural parameters provided as input to the model are collected either manually or through computer vision methods. Our dataset includes 15 real bridges, augmented to 100 samples, and our best model achieves an $R^2$ score of 0.9603 and a mean absolute error (MAE) of 10.50 units. From applied perspective, we also provide a web based interface for parameter entry and prediction. These results show that PINNs can offer reliable estimates of structural weight, even with limited data, and may help inform early stage failure analysis in lightweight bridge designs. The complete data and code are available at https://github.com/OmerJauhar/PINNS-For-Spaghetti-Bridges.
>
---
#### [new 213] RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出RobotArena∞，一个基于真实到仿真转换的可扩展机器人评估框架。针对真实测试耗时、安全风险高及仿真基准局限性问题，利用视觉语言模型与3D生成技术将真实视频演示转为仿真环境，结合自动评分与众包偏好判断，实现对机器人政策的高效、可复现评估，并通过环境扰动测试其泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.23571v1](http://arxiv.org/pdf/2510.23571v1)**

> **作者:** Yash Jangir; Yidi Zhang; Kashu Yamazaki; Chenyu Zhang; Kuan-Hsun Tu; Tsung-Wei Ke; Lei Ke; Yonatan Bisk; Katerina Fragkiadaki
>
> **备注:** Website: https://robotarenainf.github.io
>
> **摘要:** The pursuit of robot generalists - instructable agents capable of performing diverse tasks across diverse environments - demands rigorous and scalable evaluation. Yet real-world testing of robot policies remains fundamentally constrained: it is labor-intensive, slow, unsafe at scale, and difficult to reproduce. Existing simulation benchmarks are similarly limited, as they train and test policies within the same synthetic domains and cannot assess models trained from real-world demonstrations or alternative simulation environments. As policies expand in scope and complexity, these barriers only intensify, since defining "success" in robotics often hinges on nuanced human judgments of execution quality. In this paper, we introduce a new benchmarking framework that overcomes these challenges by shifting VLA evaluation into large-scale simulated environments augmented with online human feedback. Leveraging advances in vision-language models, 2D-to-3D generative modeling, and differentiable rendering, our approach automatically converts video demonstrations from widely used robot datasets into simulated counterparts. Within these digital twins, we assess VLA policies using both automated VLM-guided scoring and scalable human preference judgments collected from crowdworkers, transforming human involvement from tedious scene setup, resetting, and safety supervision into lightweight preference comparisons. To measure robustness, we systematically perturb simulated environments along multiple axes, such as textures and object placements, stress-testing policy generalization under controlled variation. The result is a continuously evolving, reproducible, and scalable benchmark for real-world trained robot manipulation policies, addressing a critical missing capability in today's robotics landscape.
>
---
#### [new 214] Simplifying Knowledge Transfer in Pretrained Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对预训练模型间知识迁移效率低的问题，提出基于模型仓库的双向知识转移框架。通过动态分配师生角色，实现跨模型、跨架构的知识共享，在图像分类、语义分割和视频显著性预测任务中均显著提升性能，推动多模型协同优化。**

- **链接: [http://arxiv.org/pdf/2510.22208v1](http://arxiv.org/pdf/2510.22208v1)**

> **作者:** Siddharth Jain; Shyamgopal Karthik; Vineet Gandhi
>
> **备注:** 12 pages, 3 figures, 6 tables, Accepted at TMLR 2025
>
> **摘要:** Pretrained models are ubiquitous in the current deep learning landscape, offering strong results on a broad range of tasks. Recent works have shown that models differing in various design choices exhibit categorically diverse generalization behavior, resulting in one model grasping distinct data-specific insights unavailable to the other. In this paper, we propose to leverage large publicly available model repositories as an auxiliary source of model improvements. We introduce a data partitioning strategy where pretrained models autonomously adopt either the role of a student, seeking knowledge, or that of a teacher, imparting knowledge. Experiments across various tasks demonstrate the effectiveness of our proposed approach. In image classification, we improved the performance of ViT-B by approximately 1.4% through bidirectional knowledge transfer with ViT-T. For semantic segmentation, our method boosted all evaluation metrics by enabling knowledge transfer both within and across backbone architectures. In video saliency prediction, our approach achieved a new state-of-the-art. We further extend our approach to knowledge transfer between multiple models, leading to considerable performance improvements for all model participants.
>
---
#### [new 215] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对可视化美学与质量评估任务，提出首个系统性基准VisJudge-Bench，涵盖3090个真实场景样本。研究发现主流MLLMs在评估上显著落后于人类专家，据此提出专用模型VisJudge，有效提升评估精度与一致性。**

- **链接: [http://arxiv.org/pdf/2510.22373v1](http://arxiv.org/pdf/2510.22373v1)**

> **作者:** Yupeng Xie; Zhiyang Zhang; Yifan Wu; Sirong Lu; Jiayi Zhang; Zhaoyang Yu; Jinlin Wang; Sirui Hong; Bang Liu; Chenglin Wu; Yuyu Luo
>
> **备注:** 53 pages, 26 figures, 5 tables
>
> **摘要:** Visualization, a domain-specific yet widely used form of imagery, is an effective way to turn complex datasets into intuitive insights, and its value depends on whether data are faithfully represented, clearly communicated, and aesthetically designed. However, evaluating visualization quality is challenging: unlike natural images, it requires simultaneous judgment across data encoding accuracy, information expressiveness, and visual aesthetics. Although multimodal large language models (MLLMs) have shown promising performance in aesthetic assessment of natural images, no systematic benchmark exists for measuring their capabilities in evaluating visualizations. To address this, we propose VisJudge-Bench, the first comprehensive benchmark for evaluating MLLMs' performance in assessing visualization aesthetics and quality. It contains 3,090 expert-annotated samples from real-world scenarios, covering single visualizations, multiple visualizations, and dashboards across 32 chart types. Systematic testing on this benchmark reveals that even the most advanced MLLMs (such as GPT-5) still exhibit significant gaps compared to human experts in judgment, with a Mean Absolute Error (MAE) of 0.551 and a correlation with human ratings of only 0.429. To address this issue, we propose VisJudge, a model specifically designed for visualization aesthetics and quality assessment. Experimental results demonstrate that VisJudge significantly narrows the gap with human judgment, reducing the MAE to 0.442 (a 19.8% reduction) and increasing the consistency with human experts to 0.681 (a 58.7% improvement) compared to GPT-5. The benchmark is available at https://github.com/HKUSTDial/VisJudgeBench.
>
---
#### [new 216] Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对奖励模型（RM）在多模态支持与偏好灵活性上的不足，提出Omni-Reward框架。通过构建首个支持自由形式偏好的多模态基准Omni-RewardBench、248K多模态偏好数据集Omni-RewardData，以及兼具判别与生成能力的Omni-RewardModel，实现对文本、图像、视频、音频、3D等多模态的通用奖励建模，提升模型对多样化人类偏好的适应性。**

- **链接: [http://arxiv.org/pdf/2510.23451v1](http://arxiv.org/pdf/2510.23451v1)**

> **作者:** Zhuoran Jin; Hongbang Yuan; Kejian Zhu; Jiachun Li; Pengfei Cao; Yubo Chen; Kang Liu; Jun Zhao
>
> **备注:** 48 pages, 17 figures
>
> **摘要:** Reward models (RMs) play a critical role in aligning AI behaviors with human preferences, yet they face two fundamental challenges: (1) Modality Imbalance, where most RMs are mainly focused on text and image modalities, offering limited support for video, audio, and other modalities; and (2) Preference Rigidity, where training on fixed binary preference pairs fails to capture the complexity and diversity of personalized preferences. To address the above challenges, we propose Omni-Reward, a step toward generalist omni-modal reward modeling with support for free-form preferences, consisting of: (1) Evaluation: We introduce Omni-RewardBench, the first omni-modal RM benchmark with free-form preferences, covering nine tasks across five modalities including text, image, video, audio, and 3D; (2) Data: We construct Omni-RewardData, a multimodal preference dataset comprising 248K general preference pairs and 69K instruction-tuning pairs for training generalist omni-modal RMs; (3) Model: We propose Omni-RewardModel, which includes both discriminative and generative RMs, and achieves strong performance on Omni-RewardBench as well as other widely used reward modeling benchmarks.
>
---
#### [new 217] USF-MAE: Ultrasound Self-Supervised Foundation Model with Masked Autoencoding
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出USF-MAE，一种基于掩码自编码的超声自监督基础模型，旨在解决超声图像分析中标注数据稀缺与域差异问题。通过在37万张超声图像上预训练，学习到强泛化能力的特征表示，并在多个肿瘤分类任务中超越传统模型，实现高性能跨器官诊断。**

- **链接: [http://arxiv.org/pdf/2510.22990v1](http://arxiv.org/pdf/2510.22990v1)**

> **作者:** Youssef Megahed; Robin Ducharme; Mark Walker; Steven Hawken; Adrian D. C. Chan
>
> **摘要:** Ultrasound imaging is one of the most widely used diagnostic modalities, offering real-time, radiation-free assessment across diverse clinical domains. However, interpretation of ultrasound images remains challenging due to high noise levels, operator dependence, and limited field of view, resulting in substantial inter-observer variability. Current Deep Learning approaches are hindered by the scarcity of large labeled datasets and the domain gap between general and sonographic images, which limits the transferability of models pretrained on non-medical data. To address these challenges, we introduce the Ultrasound Self-Supervised Foundation Model with Masked Autoencoding (USF-MAE), the first large-scale self-supervised MAE framework pretrained exclusively on ultrasound data. The model was pre-trained on 370,000 2D and 3D ultrasound images curated from 46 open-source datasets, collectively termed OpenUS-46, spanning over twenty anatomical regions. This curated dataset has been made publicly available to facilitate further research and reproducibility. Using a Vision Transformer encoder-decoder architecture, USF-MAE reconstructs masked image patches, enabling it to learn rich, modality-specific representations directly from unlabeled data. The pretrained encoder was fine-tuned on three public downstream classification benchmarks: BUS-BRA (breast cancer), MMOTU-2D (ovarian tumors), and GIST514-DB (gastrointestinal stromal tumors). Across all tasks, USF-MAE consistently outperformed conventional CNN and ViT baselines, achieving F1-scores of 81.6%, 79.6%, and 82.4%, respectively. Despite not using labels during pretraining, USF-MAE approached the performance of the supervised foundation model UltraSam on breast cancer classification and surpassed it on the other tasks, demonstrating strong cross-anatomical generalization.
>
---
#### [new 218] DynaSolidGeo: A Dynamic Benchmark for Genuine Spatial Mathematical Reasoning of VLMs in Solid Geometry
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出DynaSolidGeo，首个动态基准，用于评估视觉语言模型在立体几何中的真实空间数学推理能力。针对现有基准多限于2D、静态数据及忽视推理过程的问题，构建可动态生成的多样化多模态题集，并引入专家标注的推理链进行过程评估，揭示模型在空间智能上的显著不足。**

- **链接: [http://arxiv.org/pdf/2510.22340v1](http://arxiv.org/pdf/2510.22340v1)**

> **作者:** Changti Wu; Shijie Lian; Zihao Liu; Lei Zhang; Laurence Tianruo Yang; Kai Chen
>
> **备注:** The code and dataset are available at \href{https://zgca-ai4edu.github.io/DynaSolidGeo/}{DynaSolidGeo}
>
> **摘要:** Solid geometry problem solving demands spatial mathematical reasoning that integrates spatial intelligence and symbolic reasoning. However, most existing multimodal mathematical reasoning benchmarks focus primarily on 2D plane geometry, rely on static datasets prone to data contamination and memorization, and evaluate models solely by final answers, overlooking the reasoning process. To address these limitations, we introduce DynaSolidGeo, the first dynamic benchmark for evaluating genuine spatial reasoning in Vision-Language Models (VLMs). Constructed through a semi-automatic annotation pipeline, DynaSolidGeo contains 503 expert-curated seed questions that can, in principle, dynamically generate an unbounded number of diverse multimodal text-visual instances. Beyond answer accuracy, we incorporate process evaluation based on expert-annotated reasoning chains to measure logical validity and causal coherence. Experiments across representative open-source and closed-source VLMs reveal large performance gaps, severe degradation in dynamic settings, and poor performance on tasks requiring high-level spatial intelligence, such as mental rotation and visualization. The code and dataset are available at \href{https://zgca-ai4edu.github.io/DynaSolidGeo/}{DynaSolidGeo}.
>
---
#### [new 219] Power to the Clients: Federated Learning in a Dictatorship Setting
- **分类: cs.LG; cs.AI; cs.CL; cs.CR; cs.CV; cs.DC**

- **简介: 该论文研究联邦学习中的安全问题，针对恶意客户端可能破坏模型训练的漏洞，提出“独裁客户端”概念。通过理论分析与实证验证，揭示单个或多个独裁客户端如何抹除其他客户端贡献并影响模型收敛，揭示了联邦学习在对抗性环境下的脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.22149v1](http://arxiv.org/pdf/2510.22149v1)**

> **作者:** Mohammadsajad Alipour; Mohammad Mohammadi Amiri
>
> **摘要:** Federated learning (FL) has emerged as a promising paradigm for decentralized model training, enabling multiple clients to collaboratively learn a shared model without exchanging their local data. However, the decentralized nature of FL also introduces vulnerabilities, as malicious clients can compromise or manipulate the training process. In this work, we introduce dictator clients, a novel, well-defined, and analytically tractable class of malicious participants capable of entirely erasing the contributions of all other clients from the server model, while preserving their own. We propose concrete attack strategies that empower such clients and systematically analyze their effects on the learning process. Furthermore, we explore complex scenarios involving multiple dictator clients, including cases where they collaborate, act independently, or form an alliance in order to ultimately betray one another. For each of these settings, we provide a theoretical analysis of their impact on the global model's convergence. Our theoretical algorithms and findings about the complex scenarios including multiple dictator clients are further supported by empirical evaluations on both computer vision and natural language processing benchmarks.
>
---
#### [new 220] A Multimodal, Multitask System for Generating E Commerce Text Listings from Images
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出一种多模态多任务系统，用于从图像生成电商文本列表。针对人工编写耗时及现有模型易产生事实幻觉的问题，设计了联合训练视觉编码器与分层生成机制，提升事实一致性与效率，显著降低幻觉率并加速生成。**

- **链接: [http://arxiv.org/pdf/2510.21835v1](http://arxiv.org/pdf/2510.21835v1)**

> **作者:** Nayan Kumar Singh
>
> **备注:** 24 pages, 10 figures, 11 tables. Code can be found at: https://github.com/SinghNayanKumar/multimodal-product-lister/
>
> **摘要:** Manually generating catchy descriptions and names is labor intensive and a slow process for retailers. Although generative AI provides an automation solution in form of Vision to Language Models (VLM), the current VLMs are prone to factual "hallucinations". Siloed, single task models are not only inefficient but also fail to capture interdependent relationships between features. To address these challenges, we propose an end to end, multi task system that generates factually grounded textual listings from a single image. The contributions of this study are two proposals for the model architecture. First, application of multi task learning approach for fine tuning a vision encoder where a single vision backbone is jointly trained on attribute prediction such as color, hemline and neck style and price regression. Second, introduction of a hierarchical generation process where the model's own predicted attributes are embedded in a prompt and fed to the text decoder to improve factual consistency. The experiments demonstrate the superiority of this architecture. The multi tasking approach outperforms both the independent price regression, with a 3.6% better R2 Value and attribute classification, with a 6.6% improvement F1 score. Critically, the hierarchical generation process proves highly effective, slashing the factual hallucination rate from 12.7% to 7.1%, a 44.5% relative reduction, compared to a non hierarchical ablation. The hierarchical approach also reduces the latency of the autoregressive text generation process by a factor of 3.5 when compared to direct vision to language model of similar size. One minor caveat is that the model does perform 3.5% worse than direct vision-to-language model on ROUGE-L score.
>
---
#### [new 221] Localising under the drape: proprioception in the era of distributed surgical robotic system
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出一种无标记的本体感知方法，解决分布式手术机器人在无视觉线索下的精确定位问题。基于轻量级双目相机与新型Transformer模型，利用大规模自标注数据实现对全机器人及术野的鲁棒跟踪，提升视野覆盖25%，支持多机器人协同与智能控制，推动模块化自主手术发展。**

- **链接: [http://arxiv.org/pdf/2510.23512v1](http://arxiv.org/pdf/2510.23512v1)**

> **作者:** Martin Huber; Nicola A. Cavalcanti; Ayoob Davoodi; Ruixuan Li; Christopher E. Mower; Fabio Carrillo; Christoph J. Laux; Francois Teyssere; Thibault Chandanson; Antoine Harlé; Elie Saghbiny; Mazda Farshad; Guillaume Morel; Emmanuel Vander Poorten; Philipp Fürnstahl; Sébastien Ourselin; Christos Bergeles; Tom Vercauteren
>
> **摘要:** Despite their mechanical sophistication, surgical robots remain blind to their surroundings. This lack of spatial awareness causes collisions, system recoveries, and workflow disruptions, issues that will intensify with the introduction of distributed robots with independent interacting arms. Existing tracking systems rely on bulky infrared cameras and reflective markers, providing only limited views of the surgical scene and adding hardware burden in crowded operating rooms. We present a marker-free proprioception method that enables precise localisation of surgical robots under their sterile draping despite associated obstruction of visual cues. Our method solely relies on lightweight stereo-RGB cameras and novel transformer-based deep learning models. It builds on the largest multi-centre spatial robotic surgery dataset to date (1.4M self-annotated images from human cadaveric and preclinical in vivo studies). By tracking the entire robot and surgical scene, rather than individual markers, our approach provides a holistic view robust to occlusions, supporting surgical scene understanding and context-aware control. We demonstrate an example of potential clinical benefits during in vivo breathing compensation with access to tissue dynamics, unobservable under state of the art tracking, and accurately locate in multi-robot systems for future intelligent interaction. In addition, and compared with existing systems, our method eliminates markers and improves tracking visibility by 25%. To our knowledge, this is the first demonstration of marker-free proprioception for fully draped surgical robots, reducing setup complexity, enhancing safety, and paving the way toward modular and autonomous robotic surgery.
>
---
#### [new 222] HDR Image Reconstruction using an Unsupervised Fusion Model
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于多曝光图像融合任务，旨在解决传统相机动态范围有限导致无法完整捕捉自然场景亮度的问题。提出一种无监督深度学习方法，利用CNN融合过曝与欠曝的LDR图像，重建高质量HDR图像，无需真实HDR标签，通过自定义损失函数提升重建精度。**

- **链接: [http://arxiv.org/pdf/2510.21815v1](http://arxiv.org/pdf/2510.21815v1)**

> **作者:** Kumbha Nagaswetha
>
> **摘要:** High Dynamic Range (HDR) imaging aims to reproduce the wide range of brightness levels present in natural scenes, which the human visual system can perceive but conventional digital cameras often fail to capture due to their limited dynamic range. To address this limitation, we propose a deep learning-based multi-exposure fusion approach for HDR image generation. The method takes a set of differently exposed Low Dynamic Range (LDR) images, typically an underexposed and an overexposed image, and learns to fuse their complementary information using a convolutional neural network (CNN). The underexposed image preserves details in bright regions, while the overexposed image retains information in dark regions; the network effectively combines these to reconstruct a high-quality HDR output. The model is trained in an unsupervised manner, without relying on ground-truth HDR images, making it practical for real-world applications where such data is unavailable. We evaluate our results using the Multi-Exposure Fusion Structural Similarity Index Measure (MEF-SSIM) and demonstrate that our approach achieves superior visual quality compared to existing fusion methods. A customized loss function is further introduced to improve reconstruction fidelity and optimize model performance.
>
---
#### [new 223] A Robotic Stirring Method with Trajectory Optimization and Adaptive Speed Control for Accurate Pest Counting in Water Traps
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对水 trap 中害虫计数因遮挡导致精度低的问题，提出基于机械臂的轨迹优化与自适应速度控制搅拌方法。通过对比多种搅拌轨迹，确定最优路径，并设计闭环系统依据计数置信度动态调节搅拌速度，提升计数准确性。属于精准农业中的害虫动态监测任务。**

- **链接: [http://arxiv.org/pdf/2510.21732v1](http://arxiv.org/pdf/2510.21732v1)**

> **作者:** Xumin Gao; Mark Stevens; Grzegorz Cielniak
>
> **备注:** This paper has been submitted to ICRA 2026 and is currently under review
>
> **摘要:** Accurate monitoring of pest population dynamics is crucial for informed decision-making in precision agriculture. Currently, mainstream image-based pest counting methods primarily rely on image processing combined with machine learning or deep learning for pest counting. However, these methods have limitations and struggle to handle situations involving pest occlusion. To address this issue, this paper proposed a robotic stirring method with trajectory optimization and adaptive speed control for accurate pest counting in water traps. First, we developed an automated stirring system for pest counting in yellow water traps based on a robotic arm. Stirring alters the distribution of pests in the yellow water trap, making some of the occluded individuals visible for detection and counting. Then, we investigated the impact of different stirring trajectories on pest counting performance and selected the optimal trajectory for pest counting. Specifically, we designed six representative stirring trajectories, including circle, square, triangle, spiral, four small circles, and random lines, for the robotic arm to stir. And by comparing the overall average counting error and counting confidence of different stirring trajectories across various pest density scenarios, we determined the optimal trajectory. Finally, we proposed a counting confidence-driven closed-loop control system to achieve adaptive-speed stirring. It uses changes in pest counting confidence between consecutive frames as feedback to adjust the stirring speed. To the best of our knowledge, this is the first study dedicated to investigating the effects of different stirring trajectories on object counting in the dynamic liquid environment and to implement adaptive-speed stirring for this type of task. Experimental results show ...
>
---
#### [new 224] Hollywood Town: Long-Video Generation via Cross-Modal Multi-Agent Orchestration
- **分类: cs.MA; cs.CV**

- **简介: 该论文聚焦长视频生成任务，针对多智能体协作中模块化、上下文不足与迭代优化难题，提出三层创新：基于影视制作的分层图架构、用于临时讨论的超图节点、带有限重试的循环图结构，提升协作效率与生成质量。**

- **链接: [http://arxiv.org/pdf/2510.22431v1](http://arxiv.org/pdf/2510.22431v1)**

> **作者:** Zheng Wei; Mingchen Li; Zeqian Zhang; Ruibin Yuan; Pan Hui; Huamin Qu; James Evans; Maneesh Agrawala; Anyi Rao
>
> **摘要:** Recent advancements in multi-agent systems have demonstrated significant potential for enhancing creative task performance, such as long video generation. This study introduces three innovations to improve multi-agent collaboration. First, we propose OmniAgent, a hierarchical, graph-based multi-agent framework for long video generation that leverages a film-production-inspired architecture to enable modular specialization and scalable inter-agent collaboration. Second, inspired by context engineering, we propose hypergraph nodes that enable temporary group discussions among agents lacking sufficient context, reducing individual memory requirements while ensuring adequate contextual information. Third, we transition from directed acyclic graphs (DAGs) to directed cyclic graphs with limited retries, allowing agents to reflect and refine outputs iteratively, thereby improving earlier stages through feedback from subsequent nodes. These contributions lay the groundwork for developing more robust multi-agent systems in creative tasks.
>
---
#### [new 225] TraceTrans: Translation and Spatial Tracing for Surgical Prediction
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出TraceTrans，一种用于术后预测的可变形图像翻译模型。针对现有方法忽视空间对应关系导致结构不一致的问题，引入双解码器生成形变场与目标图像，确保解剖一致性。在医学美容和脑MRI数据集上验证了其准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2510.22379v1](http://arxiv.org/pdf/2510.22379v1)**

> **作者:** Xiyu Luo; Haodong LI; Xinxing Cheng; He Zhao; Yang Hu; Xuan Song; Tianyang Zhang
>
> **摘要:** Image-to-image translation models have achieved notable success in converting images across visual domains and are increasingly used for medical tasks such as predicting post-operative outcomes and modeling disease progression. However, most existing methods primarily aim to match the target distribution and often neglect spatial correspondences between the source and translated images. This limitation can lead to structural inconsistencies and hallucinations, undermining the reliability and interpretability of the predictions. These challenges are accentuated in clinical applications by the stringent requirement for anatomical accuracy. In this work, we present TraceTrans, a novel deformable image translation model designed for post-operative prediction that generates images aligned with the target distribution while explicitly revealing spatial correspondences with the pre-operative input. The framework employs an encoder for feature extraction and dual decoders for predicting spatial deformations and synthesizing the translated image. The predicted deformation field imposes spatial constraints on the generated output, ensuring anatomical consistency with the source. Extensive experiments on medical cosmetology and brain MRI datasets demonstrate that TraceTrans delivers accurate and interpretable post-operative predictions, highlighting its potential for reliable clinical deployment.
>
---
#### [new 226] Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction
- **分类: cs.AI; cs.CV**

- **简介: 该论文针对语义约束对抗样本（SemanticAE）生成中因指令不确定性导致的攻击效果不佳问题，提出多维度不确定性降低框架InSUR。通过稳定攻击方向、补全指令缺失信息、优化评估边界，提升生成对抗样本的可迁移性与适应性，并首次实现无参考的3D语义约束对抗样本生成。**

- **链接: [http://arxiv.org/pdf/2510.22981v1](http://arxiv.org/pdf/2510.22981v1)**

> **作者:** Jin Hu; Jiakai Wang; Linna Jing; Haolin Li; Haodong Liu; Haotong Qin; Aishan Liu; Ke Xu; Xianglong Liu
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recently, semantically constrained adversarial examples (SemanticAE), which are directly generated from natural language instructions, have become a promising avenue for future research due to their flexible attacking forms. To generate SemanticAEs, current methods fall short of satisfactory attacking ability as the key underlying factors of semantic uncertainty in human instructions, such as referring diversity, descriptive incompleteness, and boundary ambiguity, have not been fully investigated. To tackle the issues, this paper develops a multi-dimensional instruction uncertainty reduction (InSUR) framework to generate more satisfactory SemanticAE, i.e., transferable, adaptive, and effective. Specifically, in the dimension of the sampling method, we propose the residual-driven attacking direction stabilization to alleviate the unstable adversarial optimization caused by the diversity of language references. By coarsely predicting the language-guided sampling process, the optimization process will be stabilized by the designed ResAdv-DDIM sampler, therefore releasing the transferable and robust adversarial capability of multi-step diffusion models. In task modeling, we propose the context-encoded attacking scenario constraint to supplement the missing knowledge from incomplete human instructions. Guidance masking and renderer integration are proposed to regulate the constraints of 2D/3D SemanticAE, activating stronger scenario-adapted attacks. Moreover, in the dimension of generator evaluation, we propose the semantic-abstracted attacking evaluation enhancement by clarifying the evaluation boundary, facilitating the development of more effective SemanticAE generators. Extensive experiments demonstrate the superiority of the transfer attack performance of InSUR. Moreover, we realize the reference-free generation of semantically constrained 3D adversarial examples for the first time.
>
---
#### [new 227] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy
- **分类: cs.IR; cs.CV**

- **简介: 该论文针对视觉丰富文档的检索任务，解决单向量高效但粗略、多向量准确但昂贵的矛盾。提出HEAVEN框架，分两阶段：先用单向量快速筛选候选页，再以多向量精排并过滤无关词。构建首个相关基准ViMDOC，实验证明其在保持近似精度的同时大幅降低计算开销。**

- **链接: [http://arxiv.org/pdf/2510.22215v1](http://arxiv.org/pdf/2510.22215v1)**

> **作者:** Juyeon Kim; Geon Lee; Dongwon Choi; Taeuk Kim; Kijung Shin
>
> **摘要:** Retrieval over visually rich documents is essential for tasks such as legal discovery, scientific search, and enterprise knowledge management. Existing approaches fall into two paradigms: single-vector retrieval, which is efficient but coarse, and multi-vector retrieval, which is accurate but computationally expensive. To address this trade-off, we propose HEAVEN, a two-stage hybrid-vector framework. In the first stage, HEAVEN efficiently retrieves candidate pages using a single-vector method over Visually-Summarized Pages (VS-Pages), which assemble representative visual layouts from multiple pages. In the second stage, it reranks candidates with a multi-vector method while filtering query tokens by linguistic importance to reduce redundant computations. To evaluate retrieval systems under realistic conditions, we also introduce ViMDOC, the first benchmark for visually rich, multi-document, and long-document retrieval. Across four benchmarks, HEAVEN attains 99.87% of the Recall@1 performance of multi-vector models on average while reducing per-query computation by 99.82%, achieving efficiency and accuracy. Our code and datasets are available at: https://github.com/juyeonnn/HEAVEN
>
---
#### [new 228] Neural-HAR: A Dimension-Gated CNN Accelerator for Real-Time Radar Human Activity Recognition
- **分类: eess.SP; cs.CV**

- **简介: 该论文针对雷达人体活动识别（HAR）在边缘设备上计算资源受限的问题，提出Neural-HAR加速器。设计轻量级门控卷积网络GateCNN，通过频谱演化建模与双路径门控机制，在仅2.7k参数下实现86.4%准确率。基于FPGA的原型实现107.5μs低延迟与15mW低功耗，支持实时、高效边缘推理。**

- **链接: [http://arxiv.org/pdf/2510.22772v1](http://arxiv.org/pdf/2510.22772v1)**

> **作者:** Yizhuo Wu; Francesco Fioranelli; Chang Gao
>
> **摘要:** Radar-based human activity recognition (HAR) is attractive for unobtrusive and privacy-preserving monitoring, yet many CNN/RNN solutions remain too heavy for edge deployment, and even lightweight ViT/SSM variants often exceed practical compute and memory budgets. We introduce Neural-HAR, a dimension-gated CNN accelerator tailored for real-time radar HAR on resource-constrained platforms. At its core is GateCNN, a parameter-efficient Doppler-temporal network that (i) embeds Doppler vectors to emphasize frequency evolution over time and (ii) applies dual-path gated convolutions that modulate Doppler-aware content features with temporal gates, complemented by a residual path for stable training. On the University of Glasgow UoG2020 continuous radar dataset, GateCNN attains 86.4% accuracy with only 2.7k parameters and 0.28M FLOPs per inference, comparable to CNN-BiGRU at a fraction of the complexity. Our FPGA prototype on Xilinx Zynq-7000 Z-7007S reaches 107.5 $\mu$s latency and 15 mW dynamic power using LUT-based ROM and distributed RAM only (zero DSP/BRAM), demonstrating real-time, energy-efficient edge inference. Code and HLS conversion scripts are available at https://github.com/lab-emi/AIRHAR.
>
---
#### [new 229] J-ORA: A Framework and Multimodal Dataset for Japanese Object Identification, Reference, Action Prediction in Robot Perception
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出J-ORA框架与多模态数据集，聚焦日语人机对话中的机器人感知任务，解决物体识别、指代消解与动作预测问题。通过丰富物体属性标注，提升视觉语言模型性能，揭示专有与开源模型间的差距，并强调上下文敏感属性对动态环境感知的重要性。**

- **链接: [http://arxiv.org/pdf/2510.21761v1](http://arxiv.org/pdf/2510.21761v1)**

> **作者:** Jesse Atuhurra; Hidetaka Kamigaito; Taro Watanabe; Koichiro Yoshino
>
> **备注:** Accepted to IROS2025
>
> **摘要:** We introduce J-ORA, a novel multimodal dataset that bridges the gap in robot perception by providing detailed object attribute annotations within Japanese human-robot dialogue scenarios. J-ORA is designed to support three critical perception tasks, object identification, reference resolution, and next-action prediction, by leveraging a comprehensive template of attributes (e.g., category, color, shape, size, material, and spatial relations). Extensive evaluations with both proprietary and open-source Vision Language Models (VLMs) reveal that incorporating detailed object attributes substantially improves multimodal perception performance compared to without object attributes. Despite the improvement, we find that there still exists a gap between proprietary and open-source VLMs. In addition, our analysis of object affordances demonstrates varying abilities in understanding object functionality and contextual relationships across different VLMs. These findings underscore the importance of rich, context-sensitive attribute annotations in advancing robot perception in dynamic environments. See project page at https://jatuhurrra.github.io/J-ORA/.
>
---
#### [new 230] Atlas Urban Index: A VLM-Based Approach for Spatially and Temporally Calibrated Urban Development Monitoring
- **分类: cs.AI; cs.CV; cs.ET; eess.IV**

- **简介: 该论文提出Atlas Urban Index（AUI），用于空间与时间校准的城市发展监测。针对传统指数如NDBI受云雾、季节变化影响的问题，利用视觉语言模型（VLM）结合多时相哨兵2号影像，通过参考图像与时间锚定实现稳定评分，提升城市化监测的准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.22702v1](http://arxiv.org/pdf/2510.22702v1)**

> **作者:** Mithul Chander; Sai Pragnya Ranga; Prathamesh Mayekar
>
> **备注:** An abridged version of this paper will be presented at and appear in the Proceedings of ACM IKDD CODS 2025
>
> **摘要:** We introduce the {\em Atlas Urban Index} (AUI), a metric for measuring urban development computed using Sentinel-2 \citep{spoto2012sentinel2} satellite imagery. Existing approaches, such as the {\em Normalized Difference Built-up Index} (NDBI), often struggle to accurately capture urban development due to factors like atmospheric noise, seasonal variation, and cloud cover. These limitations hinder large-scale monitoring of human development and urbanization. To address these challenges, we propose an approach that leverages {\em Vision-Language Models }(VLMs) to provide a development score for regions. Specifically, we collect a time series of Sentinel-2 images for each region. Then, we further process the images within fixed time windows to get an image with minimal cloud cover, which serves as the representative image for that time window. To ensure consistent scoring, we adopt two strategies: (i) providing the VLM with a curated set of reference images representing different levels of urbanization, and (ii) supplying the most recent past image to both anchor temporal consistency and mitigate cloud-related noise in the current image. Together, these components enable AUI to overcome the challenges of traditional urbanization indices and produce more reliable and stable development scores. Our qualitative experiments on Bangalore suggest that AUI outperforms standard indices such as NDBI.
>
---
#### [new 231] Frequency-Spatial Interaction Driven Network for Low-Light Image Enhancement
- **分类: eess.IV; cs.CV; cs.LG; cs.MM; eess.SP**

- **简介: 该论文针对低光照图像增强任务，解决现有方法忽视频域信息或信息传播效率低的问题。提出两阶段频率-空间交互网络FSIDNet，通过频域与空间域信息互补融合及信息交换模块，提升光照恢复与细节还原能力，显著改善视觉效果与量化指标。**

- **链接: [http://arxiv.org/pdf/2510.22154v1](http://arxiv.org/pdf/2510.22154v1)**

> **作者:** Yunhong Tao; Wenbing Tao; Xiang Xiang
>
> **摘要:** Low-light image enhancement (LLIE) aims at improving the perception or interpretability of an image captured in an environment with poor illumination. With the advent of deep learning, the LLIE technique has achieved significant breakthroughs. However, existing LLIE methods either ignore the important role of frequency domain information or fail to effectively promote the propagation and flow of information, limiting the LLIE performance. In this paper, we develop a novel frequency-spatial interaction-driven network (FSIDNet) for LLIE based on two-stage architecture. To be specific, the first stage is designed to restore the amplitude of low-light images to improve the lightness, and the second stage devotes to restore phase information to refine fine-grained structures. Considering that Frequency domain and spatial domain information are complementary and both favorable for LLIE, we further develop two frequency-spatial interaction blocks which mutually amalgamate the complementary spatial and frequency information to enhance the capability of the model. In addition, we construct the Information Exchange Module (IEM) to associate two stages by adequately incorporating cross-stage and cross-scale features to effectively promote the propagation and flow of information in the two-stage network structure. Finally, we conduct experiments on several widely used benchmark datasets (i.e., LOL-Real, LSRW-Huawei, etc.), which demonstrate that our method achieves the excellent performance in terms of visual results and quantitative metrics while preserving good model efficiency.
>
---
#### [new 232] UrbanVLA: A Vision-Language-Action Model for Urban Micromobility
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对城市微移动机器人在复杂城市环境中的长距离导航难题，提出UrbanVLA框架。通过视觉-语言-动作联合建模，实现路线与视觉的动态对齐，并结合两阶段训练提升模型在真实场景下的安全性和适应性，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.23576v1](http://arxiv.org/pdf/2510.23576v1)**

> **作者:** Anqi Li; Zhiyong Wang; Jiazhao Zhang; Minghan Li; Yunpeng Qi; Zhibo Chen; Zhizheng Zhang; He Wang
>
> **摘要:** Urban micromobility applications, such as delivery robots, demand reliable navigation across large-scale urban environments while following long-horizon route instructions. This task is particularly challenging due to the dynamic and unstructured nature of real-world city areas, yet most existing navigation methods remain tailored to short-scale and controllable scenarios. Effective urban micromobility requires two complementary levels of navigation skills: low-level capabilities such as point-goal reaching and obstacle avoidance, and high-level capabilities, such as route-visual alignment. To this end, we propose UrbanVLA, a route-conditioned Vision-Language-Action (VLA) framework designed for scalable urban navigation. Our method explicitly aligns noisy route waypoints with visual observations during execution, and subsequently plans trajectories to drive the robot. To enable UrbanVLA to master both levels of navigation, we employ a two-stage training pipeline. The process begins with Supervised Fine-Tuning (SFT) using simulated environments and trajectories parsed from web videos. This is followed by Reinforcement Fine-Tuning (RFT) on a mixture of simulation and real-world data, which enhances the model's safety and adaptability in real-world settings. Experiments demonstrate that UrbanVLA surpasses strong baselines by more than 55% in the SocialNav task on MetaUrban. Furthermore, UrbanVLA achieves reliable real-world navigation, showcasing both scalability to large-scale urban environments and robustness against real-world uncertainties.
>
---
#### [new 233] BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SE**

- **简介: 该论文提出BLIP-FusePPO框架，用于自动驾驶车道保持任务。通过将视觉-语言模型的语义嵌入直接融合到状态表示中，结合几何与控制信号，提升策略学习的鲁棒性与可解释性。相比仅用语义模型奖励的方法，本方案减少推理开销，增强实时性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.22370v1](http://arxiv.org/pdf/2510.22370v1)**

> **作者:** Seyed Ahmad Hosseini Miangoleh; Amin Jalal Aghdasian; Farzaneh Abdollahi
>
> **备注:** https://github.com/Amin-A96/BLIP-FusePPO-A-Vision-Language-Deep-Reinforcement-Learning-Framework-for-Lane-Keeping-in-Autonomous.git
>
> **摘要:** In this paper, we propose Bootstrapped Language-Image Pretraining-driven Fused State Representation in Proximal Policy Optimization (BLIP-FusePPO), a novel multimodal reinforcement learning (RL) framework for autonomous lane-keeping (LK), in which semantic embeddings generated by a vision-language model (VLM) are directly fused with geometric states, LiDAR observations, and Proportional-Integral-Derivative-based (PID) control feedback within the agent observation space. The proposed method lets the agent learn driving rules that are aware of their surroundings and easy to understand by combining high-level scene understanding from the VLM with low-level control and spatial signals. Our architecture brings together semantic, geometric, and control-aware representations to make policy learning more robust. A hybrid reward function that includes semantic alignment, LK accuracy, obstacle avoidance, and speed regulation helps learning to be more efficient and generalizable. Our method is different from the approaches that only use semantic models to shape rewards. Instead, it directly embeds semantic features into the state representation. This cuts down on expensive runtime inference and makes sure that semantic guidance is always available. The simulation results show that the proposed model is better at LK stability and adaptability than the best vision-based and multimodal RL baselines in a wide range of difficult driving situations. We make our code publicly available.
>
---
#### [new 234] Revising Second Order Terms in Deep Animation Video Coding
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对深度动画视频编码中的头旋转问题，改进第一阶运动模型（FOMM），用全局旋转替代雅可比变换，提升头旋转场景的生成质量并降低40%-80%的P帧码率。结合先进归一化技术稳定对抗训练，通过LPIPS和DISTS评估验证优化效果。属于低码率视频编码任务。**

- **链接: [http://arxiv.org/pdf/2510.23561v1](http://arxiv.org/pdf/2510.23561v1)**

> **作者:** Konstantin Schmidt; Thomas Richter
>
> **摘要:** First Order Motion Model is a generative model that animates human heads based on very little motion information derived from keypoints. It is a promising solution for video communication because first it operates at very low bitrate and second its computational complexity is moderate compared to other learning based video codecs. However, it has strong limitations by design. Since it generates facial animations by warping source-images, it fails to recreate videos with strong head movements. This works concentrates on one specific kind of head movements, namely head rotations. We show that replacing the Jacobian transformations in FOMM by a global rotation helps the system to perform better on items with head-rotations while saving 40% to 80% of bitrate on P-frames. Moreover, we apply state-of-the-art normalization techniques to the discriminator to stabilize the adversarial training which is essential for generating visually appealing videos. We evaluate the performance by the learned metics LPIPS and DISTS to show the success our optimizations.
>
---
#### [new 235] Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMS
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文研究多模态语音识别中大语言模型的注意力陷阱与过激活问题。针对音频-视觉语音识别（AVSR）任务，发现除起始符外，中间低语义标记也存在注意力集中和特征激活异常。提出去相关损失函数，降低起始符与其他标记的余弦相似度，有效缓解该问题，并提升在高特征下采样率下的识别准确率。**

- **链接: [http://arxiv.org/pdf/2510.22603v1](http://arxiv.org/pdf/2510.22603v1)**

> **作者:** Anand; Umberto Cappellazzo; Stavros Petridis; Maja Pantic
>
> **备注:** The code is available at https://github.com/umbertocappellazzo/Llama-AVSR
>
> **摘要:** Large language models (LLMs) have recently advanced auditory speech recognition (ASR), visual speech recognition (VSR), and audio-visual speech recognition (AVSR). However, understanding of their internal dynamics under fine-tuning remains limited. In natural language processing, recent work has revealed attention sinks, tokens that attract disproportionately high attention, and associated massive activations in which some features of sink tokens exhibit huge activation in LLMs. In this work, we are the first to study these phenomena in multimodal speech recognition. Through a detailed analysis of audio-visual LLMs, we identify attention sinks and massive activations not only at the BOS token but also at intermediate low-semantic tokens across ASR, VSR, and AVSR. We show that massive activations originate in the MLP layers and correspond to fixed feature indices across all sink tokens. We further show that intermediate sink tokens exhibit high cosine similarity to the BOS token, thereby amplifying attention and activation. Building on these insights, we introduce a simple decorrelation loss that reduces cosine similarity between BOS and other tokens, effectively mitigating intermediate sinks and massive activations. Furthermore, our method improves word error rate (WER) under high audio-visual feature downsampling while remaining stable at lower downsampling rates.
>
---
#### [new 236] S-Chain: Structured Visual Chain-of-Thought For Medicine
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出S-Chain，首个大规模多语言医学视觉-语言推理数据集，包含12,000张带标注框的专家级图像与结构化视觉思维链（SV-CoT）。旨在解决医疗VLM中推理不透明、视觉证据对齐差的问题。通过构建数据集并评估多种模型，验证了SV-CoT提升解释性与鲁棒性，并提出新机制增强视觉与推理对齐，推动可信赖医疗AI发展。**

- **链接: [http://arxiv.org/pdf/2510.22728v1](http://arxiv.org/pdf/2510.22728v1)**

> **作者:** Khai Le-Duc; Duy M. H. Nguyen; Phuong T. H. Trinh; Tien-Phat Nguyen; Nghiem T. Diep; An Ngo; Tung Vu; Trinh Vuong; Anh-Tien Nguyen; Mau Nguyen; Van Trung Hoang; Khai-Nguyen Nguyen; Hy Nguyen; Chris Ngo; Anji Liu; Nhat Ho; Anne-Christin Hauschild; Khanh Xuan Nguyen; Thanh Nguyen-Tang; Pengtao Xie; Daniel Sonntag; James Zou; Mathias Niepert; Anh Totti Nguyen
>
> **备注:** First version
>
> **摘要:** Faithful reasoning in medical vision-language models (VLMs) requires not only accurate predictions but also transparent alignment between textual rationales and visual evidence. While Chain-of-Thought (CoT) prompting has shown promise in medical visual question answering (VQA), no large-scale expert-level dataset has captured stepwise reasoning with precise visual grounding. We introduce S-Chain, the first large-scale dataset of 12,000 expert-annotated medical images with bounding boxes and structured visual CoT (SV-CoT), explicitly linking visual regions to reasoning steps. The dataset further supports 16 languages, totaling over 700k VQA pairs for broad multilingual applicability. Using S-Chain, we benchmark state-of-the-art medical VLMs (ExGra-Med, LLaVA-Med) and general-purpose VLMs (Qwen2.5-VL, InternVL2.5), showing that SV-CoT supervision significantly improves interpretability, grounding fidelity, and robustness. Beyond benchmarking, we study its synergy with retrieval-augmented generation, revealing how domain knowledge and visual grounding interact during autoregressive reasoning. Finally, we propose a new mechanism that strengthens the alignment between visual evidence and reasoning, improving both reliability and efficiency. S-Chain establishes a new benchmark for grounded medical reasoning and paves the way toward more trustworthy and explainable medical VLMs.
>
---
#### [new 237] LT-Exosense: A Vision-centric Multi-session Mapping System for Lifelong Safe Navigation of Exoskeletons
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出LT-Exosense，一种面向外骨骼长期安全导航的视觉中心多会话建图系统。针对动态环境中持久感知与路径规划难题，通过增量融合多时段空间知识，实现环境变化检测与全局地图更新，支持自适应路径规划。实验证明其定位精度优于5cm，具备在复杂室内环境中长期稳定运行的能力。**

- **链接: [http://arxiv.org/pdf/2510.22164v1](http://arxiv.org/pdf/2510.22164v1)**

> **作者:** Jianeng Wang; Matias Mattamala; Christina Kassab; Nived Chebrolu; Guillaume Burger; Fabio Elnecave; Marine Petriaux; Maurice Fallon
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Self-balancing exoskeletons offer a promising mobility solution for individuals with lower-limb disabilities. For reliable long-term operation, these exoskeletons require a perception system that is effective in changing environments. In this work, we introduce LT-Exosense, a vision-centric, multi-session mapping system designed to support long-term (semi)-autonomous navigation for exoskeleton users. LT-Exosense extends single-session mapping capabilities by incrementally fusing spatial knowledge across multiple sessions, detecting environmental changes, and updating a persistent global map. This representation enables intelligent path planning, which can adapt to newly observed obstacles and can recover previous routes when obstructions are removed. We validate LT-Exosense through several real-world experiments, demonstrating a scalable multi-session map that achieves an average point-to-point error below 5 cm when compared to ground-truth laser scans. We also illustrate the potential application of adaptive path planning in dynamically changing indoor environments.
>
---
#### [new 238] LAMP: Data-Efficient Linear Affine Weight-Space Models for Parameter-Controlled 3D Shape Generation and Extrapolation
- **分类: cs.LG; cs.CE; cs.CV**

- **简介: 该论文提出LAMP框架，解决3D形状生成中数据效率低、可控性差与外推能力弱的问题。通过线性仿射权重空间混合，在少量样本下实现参数可控的3D形状生成与安全外推，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.22491v1](http://arxiv.org/pdf/2510.22491v1)**

> **作者:** Ghadi Nehme; Yanxia Zhang; Dule Shu; Matt Klenk; Faez Ahmed
>
> **摘要:** Generating high-fidelity 3D geometries that satisfy specific parameter constraints has broad applications in design and engineering. However, current methods typically rely on large training datasets and struggle with controllability and generalization beyond the training distributions. To overcome these limitations, we introduce LAMP (Linear Affine Mixing of Parametric shapes), a data-efficient framework for controllable and interpretable 3D generation. LAMP first aligns signed distance function (SDF) decoders by overfitting each exemplar from a shared initialization, then synthesizes new geometries by solving a parameter-constrained mixing problem in the aligned weight space. To ensure robustness, we further propose a safety metric that detects geometry validity via linearity mismatch. We evaluate LAMP on two 3D parametric benchmarks: DrivAerNet++ and BlendedNet. We found that LAMP enables (i) controlled interpolation within bounds with as few as 100 samples, (ii) safe extrapolation by up to 100% parameter difference beyond training ranges, (iii) physics performance-guided optimization under fixed parameters. LAMP significantly outperforms conditional autoencoder and Deep Network Interpolation (DNI) baselines in both extrapolation and data efficiency. Our results demonstrate that LAMP advances controllable, data-efficient, and safe 3D generation for design exploration, dataset generation, and performance-driven optimization.
>
---
#### [new 239] Dynamic Dropout: Leveraging Conway's Game of Life for Neural Networks Regularization
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出将康威生命游戏（GoL）用于神经网络正则化，以动态替代传统静态dropout。通过将神经元视为GoL细胞，利用其演化规则实现单元动态失活，增强模型泛化能力。实验表明该方法在CIFAR-10上性能与dropout相当，且可可视化训练过程中的空间模式，适用于深层网络。**

- **链接: [http://arxiv.org/pdf/2510.22383v1](http://arxiv.org/pdf/2510.22383v1)**

> **作者:** David Freire-Obregón; José Salas-Cáceres; Modesto Castrillón-Santana
>
> **备注:** Accepted for presentation at the 5th International Conference on Computing and Machine Intelligence (ICMI 2026)
>
> **摘要:** Regularization techniques play a crucial role in preventing overfitting and improving the generalization performance of neural networks. Dropout, a widely used regularization technique, randomly deactivates units during training to introduce redundancy and prevent co-adaptation among neurons. Despite its effectiveness, dropout has limitations, such as its static nature and lack of interpretability. In this paper, we propose a novel approach to regularization by substituting dropout with Conway's Game of Life (GoL), a cellular automata with simple rules that govern the evolution of a grid of cells. We introduce dynamic unit deactivation during training by representing neural network units as cells in a GoL grid and applying the game's rules to deactivate units. This approach allows for the emergence of spatial patterns that adapt to the training data, potentially enhancing the network's ability to generalize. We demonstrate the effectiveness of our approach on the CIFAR-10 dataset, showing that dynamic unit deactivation using GoL achieves comparable performance to traditional dropout techniques while offering insights into the network's behavior through the visualization of evolving patterns. Furthermore, our discussion highlights the applicability of our proposal in deeper architectures, demonstrating how it enhances the performance of different dropout techniques.
>
---
#### [new 240] An Intelligent Water-Saving Irrigation System Based on Multi-Sensor Fusion and Visual Servoing Control
- **分类: cs.RO; cs.CV; cs.SY; eess.SY**

- **简介: 该论文针对精准农业中灌溉效率低、地形适应性差的问题，提出一种基于多传感器融合与视觉伺服控制的智能节水灌溉系统。通过轻量YOLO模型实现高精度植株检测，结合简化手眼标定与主动调平技术，提升机器人定位与平台稳定性，在多种环境下实现30-50%节水率，水利用效率超92%。**

- **链接: [http://arxiv.org/pdf/2510.23003v1](http://arxiv.org/pdf/2510.23003v1)**

> **作者:** ZhengKai Huang; YiKun Wang; ChenYu Hui; XiaoCheng
>
> **摘要:** This paper introduces an intelligent water-saving irrigation system designed to address critical challenges in precision agriculture, such as inefficient water use and poor terrain adaptability. The system integrates advanced computer vision, robotic control, and real-time stabilization technologies via a multi-sensor fusion approach. A lightweight YOLO model, deployed on an embedded vision processor (K210), enables real-time plant container detection with over 96% accuracy under varying lighting conditions. A simplified hand-eye calibration algorithm-designed for 'handheld camera' robot arm configurations-ensures that the end effector can be precisely positioned, with a success rate exceeding 90%. The active leveling system, driven by the STM32F103ZET6 main control chip and JY901S inertial measurement data, can stabilize the irrigation platform on slopes up to 10 degrees, with a response time of 1.8 seconds. Experimental results across three simulated agricultural environments (standard greenhouse, hilly terrain, complex lighting) demonstrate a 30-50% reduction in water consumption compared to conventional flood irrigation, with water use efficiency exceeding 92% in all test cases.
>
---
#### [new 241] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning
- **分类: cs.LG; cs.CG; cs.CV**

- **简介: 该论文提出T-REGS，一种基于最小生成树长度的自监督学习正则化方法，旨在解决表示学习中的维度坍缩和分布不均匀问题。通过理论分析与实验验证，证明其能有效提升特征质量，在多种数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2510.23484v1](http://arxiv.org/pdf/2510.23484v1)**

> **作者:** Julie Mordacq; David Loiseaux; Vicky Kalogeiton; Steve Oudot
>
> **备注:** NeurIPS 2025
>
> **摘要:** Self-supervised learning (SSL) has emerged as a powerful paradigm for learning representations without labeled data, often by enforcing invariance to input transformations such as rotations or blurring. Recent studies have highlighted two pivotal properties for effective representations: (i) avoiding dimensional collapse-where the learned features occupy only a low-dimensional subspace, and (ii) enhancing uniformity of the induced distribution. In this work, we introduce T-REGS, a simple regularization framework for SSL based on the length of the Minimum Spanning Tree (MST) over the learned representation. We provide theoretical analysis demonstrating that T-REGS simultaneously mitigates dimensional collapse and promotes distribution uniformity on arbitrary compact Riemannian manifolds. Several experiments on synthetic data and on classical SSL benchmarks validate the effectiveness of our approach at enhancing representation quality.
>
---
#### [new 242] JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.SE**

- **简介: 该论文提出JanusCoder，面向代码智能的视觉-程序化接口，解决多模态代码数据稀缺与模型专用化问题。构建大规模多模态代码数据集JanusCode-800K，训练统一模型实现文本、视觉或混合输入生成代码，显著提升编码任务性能。**

- **链接: [http://arxiv.org/pdf/2510.23538v1](http://arxiv.org/pdf/2510.23538v1)**

> **作者:** Qiushi Sun; Jingyang Gong; Yang Liu; Qiaosheng Chen; Lei Li; Kai Chen; Qipeng Guo; Ben Kao; Fei Yuan
>
> **备注:** Work in progress
>
> **摘要:** The scope of neural code intelligence is rapidly expanding beyond text-based source code to encompass the rich visual outputs that programs generate. This visual dimension is critical for advanced applications like flexible content generation and precise, program-driven editing of visualizations. However, progress has been impeded by the scarcity of high-quality multimodal code data, a bottleneck stemming from challenges in synthesis and quality assessment. To address these challenges, we make contributions from both a data and modeling perspective. We first introduce a complete synthesis toolkit that leverages reciprocal synergies between data modalities to efficiently produce a large-scale, high-quality corpus spanning from standard charts to complex interactive web UIs and code-driven animations. Leveraging this toolkit, we construct JanusCode-800K, the largest multimodal code corpus to date. This powers the training of our models, JanusCoder and JanusCoderV, which establish a visual-programmatic interface for generating code from textual instructions, visual inputs, or a combination of both. Our unified model is a departure from existing approaches that build specialized models for isolated tasks. Extensive experiments on both text-centric and vision-centric coding tasks demonstrate the superior performance of the JanusCoder series, with our 7B to 14B scale models approaching or even exceeding the performance of commercial models. Furthermore, extensive analysis provides key insights into harmonizing programmatic logic with its visual expression. Our code and checkpoints will are available at https://github.com/InternLM/JanusCoder.
>
---
#### [new 243] T2I-RiskyPrompt: A Benchmark for Safety Evaluation, Attack, and Defense on Text-to-Image Model
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文针对文本到图像模型的安全性评估难题，提出T2I-RiskyPrompt基准。构建了6大类14小类的细粒度风险分类体系，收集并标注6432条有效风险提示，设计基于原因驱动的检测方法，全面评估多类模型、防御与攻击策略，揭示安全机制关键洞见。**

- **链接: [http://arxiv.org/pdf/2510.22300v1](http://arxiv.org/pdf/2510.22300v1)**

> **作者:** Chenyu Zhang; Tairen Zhang; Lanjun Wang; Ruidong Chen; Wenhui Li; Anan Liu
>
> **备注:** AAAI under review
>
> **摘要:** Using risky text prompts, such as pornography and violent prompts, to test the safety of text-to-image (T2I) models is a critical task. However, existing risky prompt datasets are limited in three key areas: 1) limited risky categories, 2) coarse-grained annotation, and 3) low effectiveness. To address these limitations, we introduce T2I-RiskyPrompt, a comprehensive benchmark designed for evaluating safety-related tasks in T2I models. Specifically, we first develop a hierarchical risk taxonomy, which consists of 6 primary categories and 14 fine-grained subcategories. Building upon this taxonomy, we construct a pipeline to collect and annotate risky prompts. Finally, we obtain 6,432 effective risky prompts, where each prompt is annotated with both hierarchical category labels and detailed risk reasons. Moreover, to facilitate the evaluation, we propose a reason-driven risky image detection method that explicitly aligns the MLLM with safety annotations. Based on T2I-RiskyPrompt, we conduct a comprehensive evaluation of eight T2I models, nine defense methods, five safety filters, and five attack strategies, offering nine key insights into the strengths and limitations of T2I model safety. Finally, we discuss potential applications of T2I-RiskyPrompt across various research fields. The dataset and code are provided in https://github.com/datar001/T2I-RiskyPrompt.
>
---
#### [new 244] Understanding What Is Not Said:Referring Remote Sensing Image Segmentation with Scarce Expressions
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文针对遥感图像分割中稀疏表述标注难的问题，提出弱化表述学习（WREL）框架，利用类别名作为弱标注与少量精准表述联合训练。通过可学习参考库和师生优化机制，提升模型在低标注成本下的性能，理论与实验证明其可逼近全标注训练效果。**

- **链接: [http://arxiv.org/pdf/2510.22760v1](http://arxiv.org/pdf/2510.22760v1)**

> **作者:** Kai Ye; Bowen Liu; Jianghang Lin; Jiayi Ji; Pingyang Dai; Liujuan Cao
>
> **摘要:** Referring Remote Sensing Image Segmentation (RRSIS) aims to segment instances in remote sensing images according to referring expressions. Unlike Referring Image Segmentation on general images, acquiring high-quality referring expressions in the remote sensing domain is particularly challenging due to the prevalence of small, densely distributed objects and complex backgrounds. This paper introduces a new learning paradigm, Weakly Referring Expression Learning (WREL) for RRSIS, which leverages abundant class names as weakly referring expressions together with a small set of accurate ones to enable efficient training under limited annotation conditions. Furthermore, we provide a theoretical analysis showing that mixed-referring training yields a provable upper bound on the performance gap relative to training with fully annotated referring expressions, thereby establishing the validity of this new setting. We also propose LRB-WREL, which integrates a Learnable Reference Bank (LRB) to refine weakly referring expressions through sample-specific prompt embeddings that enrich coarse class-name inputs. Combined with a teacher-student optimization framework using dynamically scheduled EMA updates, LRB-WREL stabilizes training and enhances cross-modal generalization under noisy weakly referring supervision. Extensive experiments on our newly constructed benchmark with varying weakly referring data ratios validate both the theoretical insights and the practical effectiveness of WREL and LRB-WREL, demonstrating that they can approach or even surpass models trained with fully annotated referring expressions.
>
---
#### [new 245] Seq-DeepIPC: Sequential Sensing for End-to-End Control in Legged Robot Navigation
- **分类: cs.RO; cs.CV; cs.SY; eess.IV; eess.SY**

- **简介: 该论文提出Seq-DeepIPC，面向足式机器人在真实环境中的端到端导航任务。针对感知与控制耦合不足的问题，融合RGB-D与GNSS多模态数据，通过时序融合提升感知精度，并简化航向估计。采用轻量编码器实现高效部署，验证了序列输入对性能的提升。**

- **链接: [http://arxiv.org/pdf/2510.23057v1](http://arxiv.org/pdf/2510.23057v1)**

> **作者:** Oskar Natan; Jun Miura
>
> **备注:** Preprint notice, this manuscript has been submitted to IEEE sensors journal for possible publication
>
> **摘要:** We present Seq-DeepIPC, a sequential end-to-end perception-to-control model for legged robot navigation in realworld environments. Seq-DeepIPC advances intelligent sensing for autonomous legged navigation by tightly integrating multi-modal perception (RGB-D + GNSS) with temporal fusion and control. The model jointly predicts semantic segmentation and depth estimation, giving richer spatial features for planning and control. For efficient deployment on edge devices, we use EfficientNet-B0 as the encoder, reducing computation while maintaining accuracy. Heading estimation is simplified by removing the noisy IMU and instead computing the bearing angle directly from consecutive GNSS positions. We collected a larger and more diverse dataset that includes both road and grass terrains, and validated Seq-DeepIPC on a robot dog. Comparative and ablation studies show that sequential inputs improve perception and control in our models, while other baselines do not benefit. Seq-DeepIPC achieves competitive or better results with reasonable model size; although GNSS-only heading is less reliable near tall buildings, it is robust in open areas. Overall, Seq-DeepIPC extends end-to-end navigation beyond wheeled robots to more versatile and temporally-aware systems. To support future research, we will release the codes to our GitHub repository at https://github.com/oskarnatan/Seq-DeepIPC.
>
---
#### [new 246] A supervised discriminant data representation: application to pattern classification
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对监督多类分类任务，提出一种融合RSLDA与ICS_DLSR优点的新型线性特征提取方法。通过引入稀疏性约束，同时实现特征选择与类内样本行稀疏一致性保持，提升数据表示能力。实验表明其在人脸、物体和数字等数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.21898v1](http://arxiv.org/pdf/2510.21898v1)**

> **作者:** Fadi Dornaika; Ahmad Khoder; Abdelmalik Moujahid; Wassim Khoder
>
> **摘要:** The performance of machine learning and pattern recognition algorithms generally depends on data representation. That is why, much of the current effort in performing machine learning algorithms goes into the design of preprocessing frameworks and data transformations able to support effective machine learning. The method proposed in this work consists of a hybrid linear feature extraction scheme to be used in supervised multi-class classification problems. Inspired by two recent linear discriminant methods: robust sparse linear discriminant analysis (RSLDA) and inter-class sparsitybased discriminative least square regression (ICS_DLSR), we propose a unifying criterion that is able to retain the advantages of these two powerful methods. The resulting transformation relies on sparsity-promoting techniques both to select the features that most accurately represent the data and to preserve the row-sparsity consistency property of samples from the same class. The linear transformation and the orthogonal matrix are estimated using an iterative alternating minimization scheme based on steepest descent gradient method and different initialization schemes. The proposed framework is generic in the sense that it allows the combination and tuning of other linear discriminant embedding methods. According to the experiments conducted on several datasets including faces, objects, and digits, the proposed method was able to outperform competing methods in most cases.
>
---
#### [new 247] SentiMaithili: A Benchmark Dataset for Sentiment and Reason Generation for the Low-Resource Maithili Language
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对低资源语言Maithili缺乏高质量情感分析数据的问题，构建了首个可解释情感分析基准数据集SentiMaithili。包含3221条带情感标签与母语理由的句子，由专家标注确保质量。实验验证了其在提升模型可解释性方面的有效性，推动了多语言NLP与可解释AI的发展。**

- **链接: [http://arxiv.org/pdf/2510.22160v1](http://arxiv.org/pdf/2510.22160v1)**

> **作者:** Rahul Ranjan; Mahendra Kumar Gurve; Anuj; Nitin; Yamuna Prasad
>
> **摘要:** Developing benchmark datasets for low-resource languages poses significant challenges, primarily due to the limited availability of native linguistic experts and the substantial time and cost involved in annotation. Given these challenges, Maithili is still underrepresented in natural language processing research. It is an Indo-Aryan language spoken by more than 13 million people in the Purvanchal region of India, valued for its rich linguistic structure and cultural significance. While sentiment analysis has achieved remarkable progress in high-resource languages, resources for low-resource languages, such as Maithili, remain scarce, often restricted to coarse-grained annotations and lacking interpretability mechanisms. To address this limitation, we introduce a novel dataset comprising 3,221 Maithili sentences annotated for sentiment polarity and accompanied by natural language justifications. Moreover, the dataset is carefully curated and validated by linguistic experts to ensure both label reliability and contextual fidelity. Notably, the justifications are written in Maithili, thereby promoting culturally grounded interpretation and enhancing the explainability of sentiment models. Furthermore, extensive experiments using both classical machine learning and state-of-the-art transformer architectures demonstrate the dataset's effectiveness for interpretable sentiment analysis. Ultimately, this work establishes the first benchmark for explainable affective computing in Maithili, thus contributing a valuable resource to the broader advancement of multilingual NLP and explainable AI.
>
---
#### [new 248] A U-Net and Transformer Pipeline for Multilingual Image Translation
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出一种端到端多语言图像文本翻译流水线，解决图像中文本检测、识别与跨语言翻译问题。工作包括：基于合成数据训练的自定义U-Net检测文本区域，Tesseract提取文字，以及从零训练的多语言Transformer模型实现序列到序列翻译，验证了全定制系统的有效性。**

- **链接: [http://arxiv.org/pdf/2510.23554v1](http://arxiv.org/pdf/2510.23554v1)**

> **作者:** Siddharth Sahay; Radhika Agarwal
>
> **备注:** 6 pages, 3 figures, 5 tables, and 2 algorithms. Prepared in IEEE double-column format
>
> **摘要:** This paper presents an end-to-end multilingual translation pipeline that integrates a custom U-Net for text detection, the Tesseract engine for text recognition, and a from-scratch sequence-to-sequence (Seq2Seq) Transformer for Neural Machine Translation (NMT). Our approach first utilizes a U-Net model, trained on a synthetic dataset , to accurately segment and detect text regions from an image. These detected regions are then processed by Tesseract to extract the source text. This extracted text is fed into a custom Transformer model trained from scratch on a multilingual parallel corpus spanning 5 languages. Unlike systems reliant on monolithic pre-trained models, our architecture emphasizes full customization and adaptability. The system is evaluated on its text detection accuracy, text recognition quality, and translation performance via BLEU scores. The complete pipeline demonstrates promising results, validating the viability of a custom-built system for translating text directly from images.
>
---
#### [new 249] DeepfakeBench-MM: A Comprehensive Benchmark for Multimodal Deepfake Detection
- **分类: cs.CR; cs.CV; cs.MM**

- **简介: 该论文针对多模态深度伪造检测任务，解决缺乏统一基准与多样数据的问题。构建了大规模数据集Mega-MMDF，并提出首个统一基准DeepfakeBench-MM，支持多模型评估与新方法探索，推动该领域发展。**

- **链接: [http://arxiv.org/pdf/2510.22622v1](http://arxiv.org/pdf/2510.22622v1)**

> **作者:** Kangran Zhao; Yupeng Chen; Xiaoyu Zhang; Yize Chen; Weinan Guan; Baicheng Chen; Chengzhe Sun; Soumyya Kanti Datta; Qingshan Liu; Siwei Lyu; Baoyuan Wu
>
> **备注:** Preprint
>
> **摘要:** The misuse of advanced generative AI models has resulted in the widespread proliferation of falsified data, particularly forged human-centric audiovisual content, which poses substantial societal risks (e.g., financial fraud and social instability). In response to this growing threat, several works have preliminarily explored countermeasures. However, the lack of sufficient and diverse training data, along with the absence of a standardized benchmark, hinder deeper exploration. To address this challenge, we first build Mega-MMDF, a large-scale, diverse, and high-quality dataset for multimodal deepfake detection. Specifically, we employ 21 forgery pipelines through the combination of 10 audio forgery methods, 12 visual forgery methods, and 6 audio-driven face reenactment methods. Mega-MMDF currently contains 0.1 million real samples and 1.1 million forged samples, making it one of the largest and most diverse multimodal deepfake datasets, with plans for continuous expansion. Building on it, we present DeepfakeBench-MM, the first unified benchmark for multimodal deepfake detection. It establishes standardized protocols across the entire detection pipeline and serves as a versatile platform for evaluating existing methods as well as exploring novel approaches. DeepfakeBench-MM currently supports 5 datasets and 11 multimodal deepfake detectors. Furthermore, our comprehensive evaluations and in-depth analyses uncover several key findings from multiple perspectives (e.g., augmentation, stacked forgery). We believe that DeepfakeBench-MM, together with our large-scale Mega-MMDF, will serve as foundational infrastructures for advancing multimodal deepfake detection.
>
---
#### [new 250] Edge Collaborative Gaussian Splatting with Integrated Rendering and Communication
- **分类: cs.IT; cs.CV; math.IT**

- **简介: 该论文针对低功耗设备上高保真3D渲染难题，提出边缘协同高斯点阵（ECO-GS）框架。通过联合优化协作状态与边缘资源分配，实现本地快速渲染与远程高质渲染的动态切换。设计了IRAC机制与PMM、ILO算法，在保证质量的同时显著提升效率。**

- **链接: [http://arxiv.org/pdf/2510.22718v1](http://arxiv.org/pdf/2510.22718v1)**

> **作者:** Yujie Wan; Chenxuan Liu; Shuai Wang; Tong Zhang; James Jianqiao Yu; Kejiang Ye; Dusit Niyato; Chengzhong Xu
>
> **备注:** 5 pages and 7 figures, submitted for possible publication
>
> **摘要:** Gaussian splatting (GS) struggles with degraded rendering quality on low-cost devices. To address this issue, we present edge collaborative GS (ECO-GS), where each user can switch between a local small GS model to guarantee timeliness and a remote large GS model to guarantee fidelity. However, deciding how to engage the large GS model is nontrivial, due to the interdependency between rendering requirements and resource conditions. To this end, we propose integrated rendering and communication (IRAC), which jointly optimizes collaboration status (i.e., deciding whether to engage large GS) and edge power allocation (i.e., enabling remote rendering) under communication constraints across different users by minimizing a newly-derived GS switching function. Despite the nonconvexity of the problem, we propose an efficient penalty majorization minimization (PMM) algorithm to obtain the critical point solution. Furthermore, we develop an imitation learning optimization (ILO) algorithm, which reduces the computational time by over 100x compared to PMM. Experiments demonstrate the superiority of PMM and the real-time execution capability of ILO.
>
---
#### [new 251] Learning Event-guided Exposure-agnostic Video Frame Interpolation via Adaptive Feature Blending
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文针对暴露未知的低帧率模糊视频，提出一种事件引导的帧插值方法。通过目标自适应事件采样与重要性映射，实现时序对齐特征的自适应融合，提升在动态曝光条件下的插值质量。**

- **链接: [http://arxiv.org/pdf/2510.22565v1](http://arxiv.org/pdf/2510.22565v1)**

> **作者:** Junsik Jung; Yoonki Cho; Woo Jae Kim; Lin Wang; Sune-eui Yoon
>
> **备注:** Accepted for BMVC2025
>
> **摘要:** Exposure-agnostic video frame interpolation (VFI) is a challenging task that aims to recover sharp, high-frame-rate videos from blurry, low-frame-rate inputs captured under unknown and dynamic exposure conditions. Event cameras are sensors with high temporal resolution, making them especially advantageous for this task. However, existing event-guided methods struggle to produce satisfactory results on severely low-frame-rate blurry videos due to the lack of temporal constraints. In this paper, we introduce a novel event-guided framework for exposure-agnostic VFI, addressing this limitation through two key components: a Target-adaptive Event Sampling (TES) and a Target-adaptive Importance Mapping (TIM). Specifically, TES samples events around the target timestamp and the unknown exposure time to better align them with the corresponding blurry frames. TIM then generates an importance map that considers the temporal proximity and spatial relevance of consecutive features to the target. Guided by this map, our framework adaptively blends consecutive features, allowing temporally aligned features to serve as the primary cues while spatially relevant ones offer complementary support. Extensive experiments on both synthetic and real-world datasets demonstrate the effectiveness of our approach in exposure-agnostic VFI scenarios.
>
---
#### [new 252] Expert Validation of Synthetic Cervical Spine Radiographs Generated with a Denoising Diffusion Probabilistic Model
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文研究生成真实感颈椎侧位X光片的扩散模型任务。针对神经外科影像数据稀缺问题，使用DDPM模型生成20,063张合成图像，并通过专家盲评验证其真实性。结果表明合成图像在视觉质量上与真实图像无统计差异，可有效支持机器学习在定位、分割和分类中的应用。**

- **链接: [http://arxiv.org/pdf/2510.22166v1](http://arxiv.org/pdf/2510.22166v1)**

> **作者:** Austin A. Barr; Brij S. Karmur; Anthony J. Winder; Eddie Guo; John T. Lysack; James N. Scott; William F. Morrish; Muneer Eesa; Morgan Willson; David W. Cadotte; Michael M. H. Yang; Ian Y. M. Chan; Sanju Lama; Garnette R. Sutherland
>
> **备注:** 10 pages, 4 figures, 1 table
>
> **摘要:** Machine learning in neurosurgery is limited by challenges in assembling large, high-quality imaging datasets. Synthetic data offers a scalable, privacy-preserving solution. We evaluated the feasibility of generating realistic lateral cervical spine radiographs using a denoising diffusion probabilistic model (DDPM) trained on 4,963 images from the Cervical Spine X-ray Atlas. Model performance was monitored via training/validation loss and Frechet inception distance, and synthetic image quality was assessed in a blinded "clinical Turing test" with six neuroradiologists and two spine-fellowship trained neurosurgeons. Experts reviewed 50 quartets containing one real and three synthetic images, identifying the real image and rating realism on a 4-point Likert scale. Experts correctly identified the real image in 29% of trials (Fleiss' kappa=0.061). Mean realism scores were comparable between real (3.323) and synthetic images (3.228, 3.258, and 3.320; p=0.383, 0.471, 1.000). Nearest-neighbor analysis found no evidence of memorization. We also provide a dataset of 20,063 synthetic radiographs. These results demonstrate that DDPM-generated cervical spine X-rays are statistically indistinguishable in realism and quality from real clinical images, offering a novel approach to creating large-scale neuroimaging datasets for ML applications in landmarking, segmentation, and classification.
>
---
#### [new 253] Privacy-Aware Federated nnU-Net for ECG Page Digitization
- **分类: cs.CR; cs.CV; cs.LG**

- **简介: 该论文针对多机构心电图（ECG）图像数字化任务，提出隐私感知的联邦nnU-Net框架。解决跨机构数据隐私与模型协同训练的矛盾，通过全模型联邦学习、安全聚合与用户级差分隐私，在非独立同分布数据下实现高效、安全的端到端波形提取，兼顾性能与隐私可审计性。**

- **链接: [http://arxiv.org/pdf/2510.22387v1](http://arxiv.org/pdf/2510.22387v1)**

> **作者:** Nader Nemati
>
> **摘要:** Deep neural networks can convert ECG page images into analyzable waveforms, yet centralized training often conflicts with cross-institutional privacy and deployment constraints. A cross-silo federated digitization framework is presented that trains a full-model nnU-Net segmentation backbone without sharing images and aggregates updates across sites under realistic non-IID heterogeneity (layout, grid style, scanner profile, noise). The protocol integrates three standard server-side aggregators--FedAvg, FedProx, and FedAdam--and couples secure aggregation with central, user-level differential privacy to align utility with formal guarantees. Key features include: (i) end-to-end full-model training and synchronization across clients; (ii) secure aggregation so the server only observes a clipped, weighted sum once a participation threshold is met; (iii) central Gaussian DP with Renyi accounting applied post-aggregation for auditable user-level privacy; and (iv) a calibration-aware digitization pipeline comprising page normalization, trace segmentation, grid-leakage suppression, and vectorization to twelve-lead signals. Experiments on ECG pages rendered from PTB-XL show consistently faster convergence and higher late-round plateaus with adaptive server updates (FedAdam) relative to FedAvg and FedProx, while approaching centralized performance. The privacy mechanism maintains competitive accuracy while preventing exposure of raw images or per-client updates, yielding deployable, auditable guarantees suitable for multi-institution settings.
>
---
## 更新

#### [replaced 001] Self-supervised Representation Learning with Local Aggregation for Image-based Profiling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14265v2](http://arxiv.org/pdf/2506.14265v2)**

> **作者:** Siran Dai; Qianqian Xu; Peisong Wen; Yang Liu; Qingming Huang
>
> **备注:** CVPR 2025 Computer Vision for Drug Discovery
>
> **摘要:** Image-based cell profiling aims to create informative representations of cell images. This technique is critical in drug discovery and has greatly advanced with recent improvements in computer vision. Inspired by recent developments in non-contrastive Self-Supervised Learning (SSL), this paper provides an initial exploration into training a generalizable feature extractor for cell images using such methods. However, there are two major challenges: 1) Unlike typical scenarios where each representation is based on a single image, cell profiling often involves multiple input images, making it difficult to effectively fuse all available information; and 2) There is a large difference between the distributions of cell images and natural images, causing the view-generation process in existing SSL methods to fail. To address these issues, we propose a self-supervised framework with local aggregation to improve cross-site consistency of cell representations. We introduce specialized data augmentation and representation post-processing methods tailored to cell images, which effectively address the issues mentioned above and result in a robust feature extractor. With these improvements, the proposed framework won the Cell Line Transferability challenge at CVPR 2025.
>
---
#### [replaced 002] Training-Free In-Context Forensic Chain for Image Manipulation Detection and Localization
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2510.10111v2](http://arxiv.org/pdf/2510.10111v2)**

> **作者:** Rui Chen; Bin Liu; Changtao Miao; Xinghao Wang; Yi Li; Tao Gong; Qi Chu; Nenghai Yu
>
> **摘要:** Advances in image tampering pose serious security threats, underscoring the need for effective image manipulation localization (IML). While supervised IML achieves strong performance, it depends on costly pixel-level annotations. Existing weakly supervised or training-free alternatives often underperform and lack interpretability. We propose the In-Context Forensic Chain (ICFC), a training-free framework that leverages multi-modal large language models (MLLMs) for interpretable IML tasks. ICFC integrates an objectified rule construction with adaptive filtering to build a reliable knowledge base and a multi-step progressive reasoning pipeline that mirrors expert forensic workflows from coarse proposals to fine-grained forensics results. This design enables systematic exploitation of MLLM reasoning for image-level classification, pixel-level localization, and text-level interpretability. Across multiple benchmarks, ICFC not only surpasses state-of-the-art training-free methods but also achieves competitive or superior performance compared to weakly and fully supervised approaches.
>
---
#### [replaced 003] HoliSafe: Holistic Safety Benchmarking and Modeling for Vision-Language Model
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.04704v3](http://arxiv.org/pdf/2506.04704v3)**

> **作者:** Youngwan Lee; Kangsan Kim; Kwanyong Park; Ilcahe Jung; Soojin Jang; Seanie Lee; Yong-Ju Lee; Sung Ju Hwang
>
> **备注:** Project page: https://youngwanlee.github.io/holisafe
>
> **摘要:** Despite emerging efforts to enhance the safety of Vision-Language Models (VLMs), current approaches face two main shortcomings. 1) Existing safety-tuning datasets and benchmarks only partially consider how image-text interactions can yield harmful content, often overlooking contextually unsafe outcomes from seemingly benign pairs. This narrow coverage leaves VLMs vulnerable to jailbreak attacks in unseen configurations. 2) Prior methods rely primarily on data-centric tuning, with limited architectural innovations to intrinsically strengthen safety. We address these gaps by introducing a holistic safety dataset and benchmark, \textbf{HoliSafe}, that spans all five safe/unsafe image-text combinations, providing a more robust basis for both training and evaluation (HoliSafe-Bench). We further propose a novel modular framework for enhancing VLM safety with a visual guard module (VGM) designed to assess the harmfulness of input images for VLMs. This module endows VLMs with a dual functionality: they not only learn to generate safer responses but can also provide an interpretable harmfulness classification to justify their refusal decisions. A significant advantage of this approach is its modularity; the VGM is designed as a plug-in component, allowing for seamless integration with diverse pre-trained VLMs across various scales. Experiments show that Safe-VLM with VGM, trained on our HoliSafe, achieves state-of-the-art safety performance across multiple VLM benchmarks. Additionally, the HoliSafe-Bench itself reveals critical vulnerabilities in existing VLM models. We hope that HoliSafe and VGM will spur further research into robust and interpretable VLM safety, expanding future avenues for multimodal alignment.
>
---
#### [replaced 004] A Novel Multi-branch ConvNeXt Architecture for Identifying Subtle Pathological Features in CT Scans
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.09107v2](http://arxiv.org/pdf/2510.09107v2)**

> **作者:** Irash Perera; Uthayasanker Thayasivam
>
> **备注:** Source Code : https://github.com/Irash-Perera/MedNeXt-Branch
>
> **摘要:** Intelligent analysis of medical imaging plays a crucial role in assisting clinical diagnosis, especially for identifying subtle pathological features. This paper introduces a novel multi-branch ConvNeXt architecture designed specifically for the nuanced challenges of medical image analysis. While applied here to the specific problem of COVID-19 diagnosis, the methodology offers a generalizable framework for classifying a wide range of pathologies from CT scans. The proposed model incorporates a rigorous end-to-end pipeline, from meticulous data preprocessing and augmentation to a disciplined two-phase training strategy that leverages transfer learning effectively. The architecture uniquely integrates features extracted from three parallel branches: Global Average Pooling, Global Max Pooling, and a new Attention-weighted Pooling mechanism. The model was trained and validated on a combined dataset of 2,609 CT slices derived from two distinct datasets. Experimental results demonstrate a superior performance on the validation set, achieving a final ROC-AUC of 0.9937, a validation accuracy of 0.9757, and an F1-score of 0.9825 for COVID-19 cases, outperforming all previously reported models on this dataset. These findings indicate that a modern, multi-branch architecture, coupled with careful data handling, can achieve performance comparable to or exceeding contemporary state-of-the-art models, thereby proving the efficacy of advanced deep learning techniques for robust medical diagnostics.
>
---
#### [replaced 005] SpineBench: A Clinically Salient, Level-Aware Benchmark Powered by the SpineMed-450k Corpus
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.03160v2](http://arxiv.org/pdf/2510.03160v2)**

> **作者:** Ming Zhao; Wenhui Dong; Yang Zhang; Xiang Zheng; Zhonghao Zhang; Zian Zhou; Yunzhi Guan; Liukun Xu; Wei Peng; Zhaoyang Gong; Zhicheng Zhang; Dachuan Li; Xiaosheng Ma; Yuli Ma; Jianing Ni; Changjiang Jiang; Lixia Tian; Qixin Chen; Kaishun Xia; Pingping Liu; Tongshun Zhang; Zhiqiang Liu; Zhongyan Bi; Chenyang Si; Tiansheng Sun; Caifeng Shan
>
> **摘要:** Spine disorders affect 619 million people globally and are a leading cause of disability, yet AI-assisted diagnosis remains limited by the lack of level-aware, multimodal datasets. Clinical decision-making for spine disorders requires sophisticated reasoning across X-ray, CT, and MRI at specific vertebral levels. However, progress has been constrained by the absence of traceable, clinically-grounded instruction data and standardized, spine-specific benchmarks. To address this, we introduce SpineMed, an ecosystem co-designed with practicing spine surgeons. It features SpineMed-450k, the first large-scale dataset explicitly designed for vertebral-level reasoning across imaging modalities with over 450,000 instruction instances, and SpineBench, a clinically-grounded evaluation framework. SpineMed-450k is curated from diverse sources, including textbooks, guidelines, open datasets, and ~1,000 de-identified hospital cases, using a clinician-in-the-loop pipeline with a two-stage LLM generation method (draft and revision) to ensure high-quality, traceable data for question-answering, multi-turn consultations, and report generation. SpineBench evaluates models on clinically salient axes, including level identification, pathology assessment, and surgical planning. Our comprehensive evaluation of several recently advanced large vision-language models (LVLMs) on SpineBench reveals systematic weaknesses in fine-grained, level-specific reasoning. In contrast, our model fine-tuned on SpineMed-450k demonstrates consistent and significant improvements across all tasks. Clinician assessments confirm the diagnostic clarity and practical utility of our model's outputs.
>
---
#### [replaced 006] PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20759v3](http://arxiv.org/pdf/2505.20759v3)**

> **作者:** Ansel Blume; Jeonghwan Kim; Hyeonjeong Ha; Elen Chatikyan; Xiaomeng Jin; Khanh Duy Nguyen; Nanyun Peng; Kai-Wei Chang; Derek Hoiem; Heng Ji
>
> **备注:** NeurIPS 2025 Spotlight; project page: https://wjdghks950.github.io/partonomy.github.io/
>
> **摘要:** Real-world objects are composed of distinctive, object-specific parts. Identifying these parts is key to performing fine-grained, compositional reasoning-yet, large multimodal models (LMMs) struggle to perform this seemingly straightforward task. In this work, we introduce PARTONOMY, an LMM benchmark designed for pixel-level part grounding. We construct PARTONOMY from existing part datasets and our own rigorously annotated set of images, encompassing 862 part labels and 534 object labels for evaluation. Unlike existing datasets that simply ask models to identify generic parts, PARTONOMY uses specialized concepts (e.g., agricultural airplane), and challenges models to compare objects' parts, consider part-whole relationships, and justify textual predictions with visual segmentations. Our experiments demonstrate significant limitations in state-of-the-art LMMs (e.g., LISA-13B achieves only 5.9% gIoU), highlighting a critical gap in their part grounding abilities. We note that existing segmentation-enabled LMMs (segmenting LMMs) have two key architectural shortcomings: they use special [SEG] tokens not seen during pretraining which induce distribution shift, and they discard predicted segmentations instead of using past predictions to guide future ones. To address these deficiencies, we train several part-centric LMMs and propose PLUM, a novel segmenting LMM that uses span tagging instead of segmentation tokens and that conditions on prior predictions in a feedback loop. We find that pretrained PLUM outperforms existing segmenting LMMs on reasoning segmentation, VQA, and visual hallucination benchmarks. In addition, PLUM finetuned on our proposed Explanatory Part Segmentation task is competitive with segmenting LMMs trained on significantly more segmentation data. Our work opens up new avenues towards enabling fine-grained, grounded visual understanding in LMMs.
>
---
#### [replaced 007] CapRecover: A Cross-Modality Feature Inversion Attack Framework on Vision Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.22828v3](http://arxiv.org/pdf/2507.22828v3)**

> **作者:** Kedong Xiu; Sai Qian Zhang
>
> **备注:** 9 pages, accepted by the 2025 ACM Multimedia Conference. Code is available at https://jus1mple.github.io/Image2CaptionAttack
>
> **摘要:** As Vision-Language Models (VLMs) are increasingly deployed in split-DNN configurations--with visual encoders (e.g., ResNet, ViT) operating on user devices and sending intermediate features to the cloud--there is a growing privacy risk from semantic information leakage. Existing approaches to reconstructing images from these intermediate features often result in blurry, semantically ambiguous images. To directly address semantic leakage, we propose CapRecover, a cross-modality inversion framework that recovers high-level semantic content, such as labels or captions, directly from intermediate features without image reconstruction. We evaluate CapRecover on multiple datasets and victim models, demonstrating strong performance in semantic recovery. Specifically, CapRecover achieves up to 92.71% Top-1 label accuracy on CIFAR-10 and generates fluent captions from ResNet50 features on COCO2017 with ROUGE-L scores up to 0.52. Our analysis further reveals that deeper convolutional layers encode significantly more semantic information compared to shallow layers. To mitigate semantic leakage, we introduce a simple yet effective protection method: adding random noise to intermediate features at each layer and removing the noise in the next layer. Experimental results show that this approach prevents semantic leakage without additional training costs. Our code is available at https://jus1mple.github.io/Image2CaptionAttack.
>
---
#### [replaced 008] RestoreVAR: Visual Autoregressive Generation for All-in-One Image Restoration
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18047v2](http://arxiv.org/pdf/2505.18047v2)**

> **作者:** Sudarshan Rajagopalan; Kartik Narayan; Vishal M. Patel
>
> **备注:** Project page: https://sudraj2002.github.io/restorevarpage/
>
> **摘要:** The use of latent diffusion models (LDMs) such as Stable Diffusion has significantly improved the perceptual quality of All-in-One image Restoration (AiOR) methods, while also enhancing their generalization capabilities. However, these LDM-based frameworks suffer from slow inference due to their iterative denoising process, rendering them impractical for time-sensitive applications. Visual autoregressive modeling (VAR), a recently introduced approach for image generation, performs scale-space autoregression and achieves comparable performance to that of state-of-the-art diffusion transformers with drastically reduced computational costs. Moreover, our analysis reveals that coarse scales in VAR primarily capture degradations while finer scales encode scene detail, simplifying the restoration process. Motivated by this, we propose RestoreVAR, a novel VAR-based generative approach for AiOR that significantly outperforms LDM-based models in restoration performance while achieving over $10\times$ faster inference. To optimally exploit the advantages of VAR for AiOR, we propose architectural modifications and improvements, including intricately designed cross-attention mechanisms and a latent-space refinement module, tailored for the AiOR task. Extensive experiments show that RestoreVAR achieves state-of-the-art performance among generative AiOR methods, while also exhibiting strong generalization capabilities.
>
---
#### [replaced 009] A Fine-Grained Attention and Geometric Correspondence Model for Musculoskeletal Risk Classification in Athletes Using Multimodal Visual and Skeletal Features
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.05913v2](http://arxiv.org/pdf/2509.05913v2)**

> **作者:** Md. Abdur Rahman; Mohaimenul Azam Khan Raiaan; Tamanna Shermin; Md Rafiqul Islam; Mukhtar Hussain; Sami Azam
>
> **摘要:** Musculoskeletal disorders pose significant risks to athletes, and assessing risk early is important for prevention. However, most existing methods are designed for controlled settings and fail to reliably assess risk in complex environments due to their reliance on a single type of data. This research introduces ViSK-GAT (Visual-Skeletal Geometric Attention Transformer), a novel multimodal deep learning framework that classifies musculoskeletal risk using both visual and skeletal coordinate-based features. A custom multimodal dataset (MusDis-Sports) was created by combining images and skeletal coordinates, with each sample labeled into eight risk categories based on the Rapid Entire Body Assessment (REBA) system. ViSK-GAT integrates two innovative modules: the Fine-Grained Attention Module (FGAM), which refines inter-modal features via cross-attention between visual and skeletal inputs, and the Multimodal Geometric Correspondence Module (MGCM), which enhances cross-modal alignment between image features and coordinates. The model achieved robust performance, with all key metrics exceeding 93%. Regression results also indicated a low RMSE of 0.1205 and MAE of 0.0156. ViSK-GAT consistently outperformed nine popular transfer learning backbones and showed its potential to advance AI-driven musculoskeletal risk assessment and enable early, impactful interventions in sports.
>
---
#### [replaced 010] Invertible generative models for inverse problems: mitigating representation error and dataset bias
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1905.11672v5](http://arxiv.org/pdf/1905.11672v5)**

> **作者:** Muhammad Asim; Mara Daniels; Oscar Leong; Ali Ahmed; Paul Hand
>
> **备注:** Camera ready version for ICML 2020, paper 2655
>
> **摘要:** Trained generative models have shown remarkable performance as priors for inverse problems in imaging -- for example, Generative Adversarial Network priors permit recovery of test images from 5-10x fewer measurements than sparsity priors. Unfortunately, these models may be unable to represent any particular image because of architectural choices, mode collapse, and bias in the training dataset. In this paper, we demonstrate that invertible neural networks, which have zero representation error by design, can be effective natural signal priors at inverse problems such as denoising, compressive sensing, and inpainting. Given a trained generative model, we study the empirical risk formulation of the desired inverse problem under a regularization that promotes high likelihood images, either directly by penalization or algorithmically by initialization. For compressive sensing, invertible priors can yield higher accuracy than sparsity priors across almost all undersampling ratios, and due to their lack of representation error, invertible priors can yield better reconstructions than GAN priors for images that have rare features of variation within the biased training set, including out-of-distribution natural images. We additionally compare performance for compressive sensing to unlearned methods, such as the deep decoder, and we establish theoretical bounds on expected recovery error in the case of a linear invertible model.
>
---
#### [replaced 011] Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21656v2](http://arxiv.org/pdf/2506.21656v2)**

> **作者:** Yifan Shen; Yuanzhe Liu; Jingyuan Zhu; Xu Cao; Xiaofeng Zhang; Yixiao He; Wenming Ye; James Matthew Rehg; Ismini Lourentzou
>
> **摘要:** Current Vision-Language Models (VLMs) struggle with fine-grained spatial reasoning, particularly when multi-step logic and precise spatial alignment are required. In this work, we introduce SpatialReasoner-R1, a vision-language reasoning model designed to address these limitations. To construct high-quality supervision for spatial reasoning, we design a Multi-Model Monte Carlo Tree Search (M3CTS) method that generates diverse, logically consistent Long Chain-of-Thought (LongCoT) reasoning trajectories. In addition, we propose fine-grained Direct Preference Optimization (fDPO), which introduces segment-specific preference granularity for descriptive grounding and logical reasoning, guided by a spatial reward mechanism that evaluates candidate responses based on visual consistency, spatial grounding, and logical coherence. Experimental results demonstrate that fDPO achieves an average improvement of 4.1% over standard DPO across spatial quality tasks, and a 9.0% gain in spatial quantity tasks. SpatialReasoner-R1, trained with fDPO, sets a new SoTA on SPATIALRGPT-Bench, outperforming the strongest baseline by 9.8% in average accuracy, while maintaining competitive performance on general vision-language tasks.
>
---
#### [replaced 012] Improving Video Generation with Human Feedback
- **分类: cs.CV; cs.AI; cs.GR; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.13918v2](http://arxiv.org/pdf/2501.13918v2)**

> **作者:** Jie Liu; Gongye Liu; Jiajun Liang; Ziyang Yuan; Xiaokun Liu; Mingwu Zheng; Xiele Wu; Qiulin Wang; Menghan Xia; Xintao Wang; Xiaohong Liu; Fei Yang; Pengfei Wan; Di Zhang; Kun Gai; Yujiu Yang; Wanli Ouyang
>
> **备注:** https://github.com/KwaiVGI/VideoAlign
>
> **摘要:** Video generation has achieved significant advances through rectified flow techniques, but issues like unsmooth motion and misalignment between videos and prompts persist. In this work, we develop a systematic pipeline that harnesses human feedback to mitigate these problems and refine the video generation model. Specifically, we begin by constructing a large-scale human preference dataset focused on modern video generation models, incorporating pairwise annotations across multi-dimensions. We then introduce VideoReward, a multi-dimensional video reward model, and examine how annotations and various design choices impact its rewarding efficacy. From a unified reinforcement learning perspective aimed at maximizing reward with KL regularization, we introduce three alignment algorithms for flow-based models. These include two training-time strategies: direct preference optimization for flow (Flow-DPO) and reward weighted regression for flow (Flow-RWR), and an inference-time technique, Flow-NRG, which applies reward guidance directly to noisy videos. Experimental results indicate that VideoReward significantly outperforms existing reward models, and Flow-DPO demonstrates superior performance compared to both Flow-RWR and supervised fine-tuning methods. Additionally, Flow-NRG lets users assign custom weights to multiple objectives during inference, meeting personalized video quality needs.
>
---
#### [replaced 013] Navigating the Accuracy-Size Trade-Off with Flexible Model Merging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23209v2](http://arxiv.org/pdf/2505.23209v2)**

> **作者:** Akash Dhasade; Divyansh Jhunjhunwala; Milos Vujasinovic; Gauri Joshi; Anne-Marie Kermarrec
>
> **摘要:** Model merging has emerged as an efficient method to combine multiple single-task fine-tuned models. The merged model can enjoy multi-task capabilities without expensive training. While promising, merging into a single model often suffers from an accuracy gap with respect to the fine-tuned models. On the other hand, deploying all individual fine-tuned models incurs high storage costs. We propose FlexMerge, a novel data-free model merging framework that: (a) flexibly generates merged models of varying sizes, spanning the full spectrum from a single merged model to retaining all fine-tuned models; and (b) supports multiple merging algorithms in a unified framework. Using FlexMerge, we systematically characterize the accuracy-size trade-off of different algorithms. Our study reveals two key findings: first, even modestly larger merged models can yield steep accuracy gains (up to 13.5% when just doubling the size); second, algorithm rankings are not consistent as size increases, with some methods overtaking others beyond the one-model regime. These results uncover a new design dimension for model merging: developing and comparing algorithms across the full spectrum of sizes rather than only at the single-model limit. Extensive experiments on vision and NLP benchmarks, with up to 30 tasks, confirm the generality and practicality of FlexMerge.
>
---
#### [replaced 014] GOOD: Training-Free Guided Diffusion Sampling for Out-of-Distribution Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.17131v2](http://arxiv.org/pdf/2510.17131v2)**

> **作者:** Xin Gao; Jiyao Liu; Guanghao Li; Yueming Lyu; Jianxiong Gao; Weichen Yu; Ningsheng Xu; Liang Wang; Caifeng Shan; Ziwei Liu; Chenyang Si
>
> **备注:** 28 pages, 16 figures, conference
>
> **摘要:** Recent advancements have explored text-to-image diffusion models for synthesizing out-of-distribution (OOD) samples, substantially enhancing the performance of OOD detection. However, existing approaches typically rely on perturbing text-conditioned embeddings, resulting in semantic instability and insufficient shift diversity, which limit generalization to realistic OOD. To address these challenges, we propose GOOD, a novel and flexible framework that directly guides diffusion sampling trajectories towards OOD regions using off-the-shelf in-distribution (ID) classifiers. GOOD incorporates dual-level guidance: (1) Image-level guidance based on the gradient of log partition to reduce input likelihood, drives samples toward low-density regions in pixel space. (2) Feature-level guidance, derived from k-NN distance in the classifier's latent space, promotes sampling in feature-sparse regions. Hence, this dual-guidance design enables more controllable and diverse OOD sample generation. Additionally, we introduce a unified OOD score that adaptively combines image and feature discrepancies, enhancing detection robustness. We perform thorough quantitative and qualitative analyses to evaluate the effectiveness of GOOD, demonstrating that training with samples generated by GOOD can notably enhance OOD detection performance.
>
---
#### [replaced 015] SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11774v2](http://arxiv.org/pdf/2509.11774v2)**

> **作者:** Changlu Guo; Anders Nymark Christensen; Anders Bjorholm Dahl; Yugen Yi; Morten Rieger Hannemose
>
> **备注:** The code is available at github.com/clguo/SA-UNetv2
>
> **摘要:** Retinal vessel segmentation is essential for early diagnosis of diseases such as diabetic retinopathy, hypertension, and neurodegenerative disorders. Although SA-UNet introduces spatial attention in the bottleneck, it underuses attention in skip connections and does not address the severe foreground-background imbalance. We propose SA-UNetv2, a lightweight model that injects cross-scale spatial attention into all skip connections to strengthen multi-scale feature fusion and adopts a weighted Binary Cross-Entropy (BCE) plus Matthews Correlation Coefficient (MCC) loss to improve robustness to class imbalance. On the public DRIVE and STARE datasets, SA-UNetv2 achieves state-of-the-art performance with only 1.2MB memory and 0.26M parameters (less than 50% of SA-UNet), and 1 second CPU inference on 592 x 592 x 3 images, demonstrating strong efficiency and deployability in resource-constrained, CPU-only settings.
>
---
#### [replaced 016] KAN or MLP? Point Cloud Shows the Way Forward
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13593v4](http://arxiv.org/pdf/2504.13593v4)**

> **作者:** Yan Shi; Qingdong He; Yijun Liu; Xiaoyu Liu; Jingyong Su
>
> **摘要:** Multi-Layer Perceptrons (MLPs) have become one of the fundamental architectural component in point cloud analysis due to its effective feature learning mechanism. However, when processing complex geometric structures in point clouds, MLPs' fixed activation functions struggle to efficiently capture local geometric features, while suffering from poor parameter efficiency and high model redundancy. In this paper, we propose PointKAN, which applies Kolmogorov-Arnold Networks (KANs) to point cloud analysis tasks to investigate their efficacy in hierarchical feature representation. First, we introduce a Geometric Affine Module (GAM) to transform local features, improving the model's robustness to geometric variations. Next, in the Local Feature Processing (LFP), a parallel structure extracts both group-level features and global context, providing a rich representation of both fine details and overall structure. Finally, these features are combined and processed in the Global Feature Processing (GFP). By repeating these operations, the receptive field gradually expands, enabling the model to capture complete geometric information of the point cloud. To overcome the high parameter counts and computational inefficiency of standard KANs, we develop Efficient-KANs in the PointKAN-elite variant, which significantly reduces parameters while maintaining accuracy. Experimental results demonstrate that PointKAN outperforms PointMLP on benchmark datasets such as ModelNet40, ScanObjectNN, and ShapeNetPart, with particularly strong performance in Few-shot Learning task. Additionally, PointKAN achieves substantial reductions in parameter counts and computational complexity (FLOPs). This work highlights the potential of KANs-based architectures in 3D vision and opens new avenues for research in point cloud understanding.
>
---
#### [replaced 017] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16503v2](http://arxiv.org/pdf/2411.16503v2)**

> **作者:** Boming Miao; Chunxiao Li; Xiaoxiao Wang; Andi Zhang; Rui Sun; Zizhe Wang; Yao Zhu
>
> **备注:** Updated author formatting; no substantive changes
>
> **摘要:** Diffusion models have achieved impressive success in generating photorealistic images, but challenges remain in ensuring precise semantic alignment with input prompts. Optimizing the initial noisy latent offers a more efficient alternative to modifying model architectures or prompt engineering for improving semantic alignment. A latest approach, InitNo, refines the initial noisy latent by leveraging attention maps; however, these maps capture only limited information, and the effectiveness of InitNo is highly dependent on the initial starting point, as it tends to converge on a local optimum near this point. To this end, this paper proposes leveraging the language comprehension capabilities of large vision-language models (LVLMs) to guide the optimization of the initial noisy latent, and introduces the Noise Diffusion process, which updates the noisy latent to generate semantically faithful images while preserving distribution consistency. Furthermore, we provide a theoretical analysis of the condition under which the update improves semantic faithfulness. Experimental results demonstrate the effectiveness and adaptability of our framework, consistently enhancing semantic alignment across various diffusion models. The code is available at https://github.com/Bomingmiao/NoiseDiffusion.
>
---
#### [replaced 018] ControlText: Unlocking Controllable Fonts in Multilingual Text Rendering without Font Annotations
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.10999v2](http://arxiv.org/pdf/2502.10999v2)**

> **作者:** Bowen Jiang; Yuan Yuan; Xinyi Bai; Zhuoqun Hao; Alyson Yin; Yaojie Hu; Wenyu Liao; Lyle Ungar; Camillo J. Taylor
>
> **备注:** The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) Findings
>
> **摘要:** This work demonstrates that diffusion models can achieve font-controllable multilingual text rendering using just raw images without font label annotations.Visual text rendering remains a significant challenge. While recent methods condition diffusion on glyphs, it is impossible to retrieve exact font annotations from large-scale, real-world datasets, which prevents user-specified font control. To address this, we propose a data-driven solution that integrates the conditional diffusion model with a text segmentation model, utilizing segmentation masks to capture and represent fonts in pixel space in a self-supervised manner, thereby eliminating the need for any ground-truth labels and enabling users to customize text rendering with any multilingual font of their choice. The experiment provides a proof of concept of our algorithm in zero-shot text and font editing across diverse fonts and languages, providing valuable insights for the community and industry toward achieving generalized visual text rendering. Code is available at github.com/bowen-upenn/ControlText.
>
---
#### [replaced 019] Vision Foundation Models as Effective Visual Tokenizers for Autoregressive Image Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08441v2](http://arxiv.org/pdf/2507.08441v2)**

> **作者:** Anlin Zheng; Xin Wen; Xuanyang Zhang; Chuofan Ma; Tiancai Wang; Gang Yu; Xiangyu Zhang; Xiaojuan Qi
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** In this work, we present a novel direction to build an image tokenizer directly on top of a frozen vision foundation model, which is a largely underexplored area. Specifically, we employ a frozen vision foundation model as the encoder of our tokenizer. To enhance its effectiveness, we introduce two key components: (1) a region-adaptive quantization framework that reduces redundancy in the pre-trained features on regular 2D grids, and (2) a semantic reconstruction objective that aligns the tokenizer's outputs with the foundation model's representations to preserve semantic fidelity. Based on these designs, our proposed image tokenizer, VFMTok, achieves substantial improvements in image reconstruction and generation quality, while also enhancing token efficiency. It further boosts autoregressive (AR) generation -- achieving a gFID of 1.36 on ImageNet benchmarks, while accelerating model convergence by three times, and enabling high-fidelity class-conditional synthesis without the need for classifier-free guidance (CFG). The code is available at https://github.com/CVMI-Lab/VFMTok.
>
---
#### [replaced 020] First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22453v2](http://arxiv.org/pdf/2505.22453v2)**

> **作者:** Lai Wei; Yuting Li; Chen Wang; Yue Wang; Linghe Kong; Weiran Huang; Lichao Sun
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Improving Multi-modal Large Language Models (MLLMs) in the post-training stage typically relies on supervised fine-tuning (SFT) or reinforcement learning (RL), which require expensive and manually annotated multi-modal data--an ultimately unsustainable resource. This limitation has motivated a growing interest in unsupervised paradigms as a third stage of post-training after SFT and RL. While recent efforts have explored this direction, their methods are complex and difficult to iterate. To address this, we propose MM-UPT, a simple yet effective framework for unsupervised post-training of MLLMs, enabling continual self-improvement without any external supervision. The training method of MM-UPT builds upon GRPO, replacing traditional reward signals with a self-rewarding mechanism based on majority voting over multiple sampled responses. Our experiments demonstrate that such training method effectively improves the reasoning ability of Qwen2.5-VL-7B (e.g., 66.3\%$\rightarrow$72.9\% on MathVista, 62.9\%$\rightarrow$68.7\% on We-Math), using standard dataset without ground truth labels. To further explore scalability, we extend our framework to a data self-generation setting, designing two strategies that prompt the MLLM to synthesize new training samples on its own. Additional experiments show that combining these synthetic data with the unsupervised training method can also boost performance, highlighting a promising approach for scalable self-improvement. Overall, MM-UPT offers a new paradigm for autonomous enhancement of MLLMs, serving as a critical third step after initial SFT and RL in the absence of external supervision. Our code is available at https://github.com/waltonfuture/MM-UPT.
>
---
#### [replaced 021] Macro2Micro: A Rapid and Precise Cross-modal Magnetic Resonance Imaging Synthesis using Multi-scale Structural Brain Similarity
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.11277v2](http://arxiv.org/pdf/2412.11277v2)**

> **作者:** Sooyoung Kim; Joonwoo Kwon; Junbeom Kwon; Jungyoun Janice Min; Sangyoon Bae; Yuewei Lin; Shinjae Yoo; Jiook Cha
>
> **备注:** The code will be made available upon acceptance
>
> **摘要:** The human brain is a complex system requiring both macroscopic and microscopic components for comprehensive understanding. However, mapping nonlinear relationships between these scales remains challenging due to technical limitations and the high cost of multimodal Magnetic Resonance Imaging (MRI) acquisition. To address this, we introduce Macro2Micro, a deep learning framework that predicts brain microstructure from macrostructure using a Generative Adversarial Network (GAN). Based on the hypothesis that microscale structural information can be inferred from macroscale structures, Macro2Micro explicitly encodes multiscale brain information into distinct processing branches. To enhance artifact elimination and output quality, we propose a simple yet effective auxiliary discriminator and learning objective. Extensive experiments demonstrated that Macro2Micro faithfully translates T1-weighted MRIs into corresponding Fractional Anisotropy (FA) images, achieving a 6.8\% improvement in the Structural Similarity Index Measure (SSIM) compared to previous methods, while retaining the individual biological characteristics of the brain. With an inference time of less than 0.01 seconds per MR modality translation, Macro2Micro introduces the potential for real-time multimodal rendering in medical and research applications. The code will be made available upon acceptance.
>
---
#### [replaced 022] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10270v3](http://arxiv.org/pdf/2503.10270v3)**

> **作者:** Zexuan Yan; Yue Ma; Chang Zou; Wenteng Chen; Qifeng Chen; Linfeng Zhang
>
> **备注:** accepted by ICCV2025
>
> **摘要:** Inversion-based image editing is rapidly gaining momentum while suffering from significant computation overhead, hindering its application in real-time interactive scenarios. In this paper, we rethink that the redundancy in inversion-based image editing exists in both the spatial and temporal dimensions, such as the unnecessary computation in unedited regions and the redundancy in the inversion progress. To tackle these challenges, we propose a practical framework, named EEdit, to achieve efficient image editing. Specifically, we introduce three techniques to solve them one by one. For spatial redundancy, spatial locality caching is introduced to compute the edited region and its neighboring regions while skipping the unedited regions, and token indexing preprocessing is designed to further accelerate the caching. For temporal redundancy, inversion step skipping is proposed to reuse the latent for efficient editing. Our experiments demonstrate an average of 2.46 $\times$ acceleration without performance drop in a wide range of editing tasks including prompt-guided image editing, dragging and image composition. Our codes are available at https://github.com/yuriYanZeXuan/EEdit
>
---
#### [replaced 023] BCR-Net: Boundary-Category Refinement Network for Weakly Semi-Supervised X-Ray Prohibited Item Detection with Points
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.18918v2](http://arxiv.org/pdf/2412.18918v2)**

> **作者:** Sanjoeng Wong
>
> **备注:** The authors withdraw this preprint because an error was found in a mathematical expression and the manuscript lacks evaluation on the COCO dataset. We will correct the error, extend experiments to include COCO, and resubmit a revised version
>
> **摘要:** Automatic prohibited item detection in X-ray images is crucial for public safety. However, most existing detection methods either rely on expensive box annotations to achieve high performance or use weak annotations but suffer from limited accuracy. To balance annotation cost and detection performance, we study Weakly Semi-Supervised X-ray Prohibited Item Detection with Points (WSSPID-P) and propose a novel \textbf{B}oundary-\textbf{C}ategory \textbf{R}efinement \textbf{Net}work (\textbf{BCR-Net}) that requires only a few box annotations and a large number of point annotations. BCR-Net is built based on Group R-CNN and introduces a new Boundary Refinement (BR) module and a new Category Refinement (CR) module. The BR module develops a dual attention mechanism to focus on both the boundaries and salient features of prohibited items. Meanwhile, the CR module incorporates contrastive branches into the heads of RPN and ROI by introducing a scale- and rotation-aware contrastive loss, enhancing intra-class consistency and inter-class separability in the feature space. Based on the above designs, BCR-Net effectively addresses the closely related problems of imprecise localization and inaccurate classification. Experimental results on public X-ray datasets show the effectiveness of BCR-Net, achieving significant performance improvements to state-of-the-art methods under limited annotations.
>
---
#### [replaced 024] Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.02745v2](http://arxiv.org/pdf/2510.02745v2)**

> **作者:** Lanyun Zhu; Deyi Ji; Tianrun Chen; Haiyang Wu; Shiqi Wang
>
> **备注:** NeurIPS 2025
>
> **摘要:** The success of DeepSeek-R1 demonstrates the immense potential of using reinforcement learning (RL) to enhance LLMs' reasoning capabilities. This paper introduces Retrv-R1, the first R1-style MLLM specifically designed for multimodal universal retrieval, achieving higher performance by employing step-by-step reasoning to produce more accurate retrieval results. We find that directly applying the methods of DeepSeek-R1 to retrieval tasks is not feasible, mainly due to (1) the high computational cost caused by the large token consumption required for multiple candidates with reasoning processes, and (2) the instability and suboptimal results when directly applying RL to train for retrieval tasks. To address these issues, Retrv-R1 introduces an information compression module with a details inspection mechanism, which enhances computational efficiency by reducing the number of tokens while ensuring that critical information for challenging candidates is preserved. Furthermore, a new training paradigm is proposed, including an activation stage using a retrieval-tailored synthetic CoT dataset for more effective optimization, followed by RL with a novel curriculum reward to improve both performance and efficiency. Incorporating these novel designs, Retrv-R1 achieves SOTA performance, high efficiency, and strong generalization ability, as demonstrated by experiments across multiple benchmarks and tasks.
>
---
#### [replaced 025] FaceTracer: Unveiling Source Identities from Swapped Face Images and Videos for Fraud Prevention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.08082v2](http://arxiv.org/pdf/2412.08082v2)**

> **作者:** Zhongyi Zhang; Jie Zhang; Wenbo Zhou; Xinghui Zhou; Qing Guo; Weiming Zhang; Tianwei Zhang; Nenghai Yu
>
> **备注:** 17 pages, 16 figures, TPAMI version
>
> **摘要:** Face-swapping techniques have advanced rapidly with the evolution of deep learning, leading to widespread use and growing concerns about potential misuse, especially in cases of fraud. While many efforts have focused on detecting swapped face images or videos, these methods are insufficient for tracing the malicious users behind fraudulent activities. Intrusive watermark-based approaches also fail to trace unmarked identities, limiting their practical utility. To address these challenges, we introduce FaceTracer, the first non-intrusive framework specifically designed to trace the identity of the source person from swapped face images or videos. Specifically, FaceTracer leverages a disentanglement module that effectively suppresses identity information related to the target person while isolating the identity features of the source person. This allows us to extract robust identity information that can directly link the swapped face back to the original individual, aiding in uncovering the actors behind fraudulent activities. Extensive experiments demonstrate FaceTracer's effectiveness across various face-swapping techniques, successfully identifying the source person in swapped content and enabling the tracing of malicious actors involved in fraudulent activities. Additionally, FaceTracer shows strong transferability to unseen face-swapping methods including commercial applications and robustness against transmission distortions and adaptive attacks.Our code is available at: https://github.com/zzy224/FaceTracer.
>
---
#### [replaced 026] Topology Sculptor, Shape Refiner: Discrete Diffusion Model for High-Fidelity 3D Meshes Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.21264v2](http://arxiv.org/pdf/2510.21264v2)**

> **作者:** Kaiyu Song; Hanjiang Lai; Yaqing Zhang; Chuangjian Cai; Yan Pan Kun Yue; Jian Yin
>
> **摘要:** In this paper, we introduce Topology Sculptor, Shape Refiner (TSSR), a novel method for generating high-quality, artist-style 3D meshes based on Discrete Diffusion Models (DDMs). Our primary motivation for TSSR is to achieve highly accurate token prediction while enabling parallel generation, a significant advantage over sequential autoregressive methods. By allowing TSSR to "see" all mesh tokens concurrently, we unlock a new level of efficiency and control. We leverage this parallel generation capability through three key innovations: 1) Decoupled Training and Hybrid Inference, which distinctly separates the DDM-based generation into a topology sculpting stage and a subsequent shape refinement stage. This strategic decoupling enables TSSR to effectively capture both intricate local topology and overarching global shape. 2) An Improved Hourglass Architecture, featuring bidirectional attention enriched by face-vertex-sequence level Rotational Positional Embeddings (RoPE), thereby capturing richer contextual information across the mesh structure. 3) A novel Connection Loss, which acts as a topological constraint to further enhance the realism and fidelity of the generated meshes. Extensive experiments on complex datasets demonstrate that TSSR generates high-quality 3D artist-style meshes, capable of achieving up to 10,000 faces at a remarkable spatial resolution of $1024^3$. The code will be released at: https://github.com/psky1111/Tencent-TSSR.
>
---
#### [replaced 027] Vision Transformers Don't Need Trained Registers
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.08010v5](http://arxiv.org/pdf/2506.08010v5)**

> **作者:** Nick Jiang; Amil Dravid; Alexei Efros; Yossi Gandelsman
>
> **备注:** Project page and code: https://avdravid.github.io/test-time-registers. Accepted to NeurIPS '25 (spotlight)
>
> **摘要:** We investigate the mechanism underlying a previously identified phenomenon in Vision Transformers - the emergence of high-norm tokens that lead to noisy attention maps (Darcet et al., 2024). We observe that in multiple models (e.g., CLIP, DINOv2), a sparse set of neurons is responsible for concentrating high-norm activations on outlier tokens, leading to irregular attention patterns and degrading downstream visual processing. While the existing solution for removing these outliers involves retraining models from scratch with additional learned register tokens, we use our findings to create a training-free approach to mitigate these artifacts. By shifting the high-norm activations from our discovered register neurons into an additional untrained token, we can mimic the effect of register tokens on a model already trained without registers. We demonstrate that our method produces cleaner attention and feature maps, enhances performance over base models across multiple downstream visual tasks, and achieves results comparable to models explicitly trained with register tokens. We then extend test-time registers to off-the-shelf vision-language models, yielding cleaner attention-based, text-to-image attribution. Finally, we outline a simple mathematical model that reflects the observed behavior of register neurons and high norm tokens. Our results suggest that test-time registers effectively take on the role of register tokens at test-time, offering a training-free solution for any pre-trained model released without them.
>
---
#### [replaced 028] Continuous and complete liver vessel segmentation with graph-attention guided diffusion
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.00617v3](http://arxiv.org/pdf/2411.00617v3)**

> **作者:** Xiaotong Zhang; Alexander Broersen; Gonnie CM van Erp; Silvia L. Pintea; Jouke Dijkstra
>
> **备注:** Accepted by Knowledge-Based Systems
>
> **摘要:** Improving connectivity and completeness are the most challenging aspects of liver vessel segmentation, especially for small vessels. These challenges require both learning the continuous vessel geometry, and focusing on small vessel detection. However, current methods do not explicitly address these two aspects and cannot generalize well when constrained by inconsistent annotations. Here, we take advantage of the generalization of the diffusion model and explicitly integrate connectivity and completeness in our diffusion-based segmentation model. Specifically, we use a graph-attention module that adds knowledge about vessel geometry, and thus adds continuity. Additionally, we perform the graph-attention at multiple-scales, thus focusing on small liver vessels. Our method outperforms eight state-of-the-art medical segmentation methods on two public datasets: 3D-ircadb-01 and LiVS. Our code is available at https://github.com/ZhangXiaotong015/GATSegDiff.
>
---
#### [replaced 029] Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14677v3](http://arxiv.org/pdf/2505.14677v3)**

> **作者:** Jiaer Xia; Yuhang Zang; Peng Gao; Sharon Li; Kaiyang Zhou
>
> **摘要:** Learning general-purpose reasoning capabilities has long been a challenging problem in AI. Recent research in large language models (LLMs), such as DeepSeek-R1, has shown that reinforcement learning techniques like GRPO can enable pre-trained LLMs to develop reasoning capabilities using simple question-answer pairs. In this paper, we aim to train visual language models (VLMs) to perform reasoning on image data through reinforcement learning and visual question-answer pairs, without any explicit chain-of-thought (CoT) supervision. Our findings indicate that simply applying reinforcement learning to a VLM -- by prompting the model to produce a reasoning chain before providing an answer -- can lead the model to develop shortcuts from easy questions, thereby reducing its ability to generalize across unseen data distributions. We argue that the key to mitigating shortcut learning is to encourage the model to interpret images prior to reasoning. Therefore, we train the model to adhere to a caption-reason-answer output format: initially generating a detailed caption for an image, followed by constructing an extensive reasoning chain. When trained on 273K CoT-free visual question-answer pairs and using only reinforcement learning, our model, named Visionary-R1, outperforms strong multimodal models, such as GPT-4o, Claude3.5-Sonnet, and Gemini-1.5-Pro, on multiple visual reasoning benchmarks.
>
---
#### [replaced 030] Zero-Shot Multi-modal Large Language Model v.s. Supervised Deep Learning: A Comparative Study on CT-Based Intracranial Hemorrhage Subtyping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.09252v2](http://arxiv.org/pdf/2505.09252v2)**

> **作者:** Yinuo Wang; Yue Zeng; Kai Chen; Cai Meng; Chao Pan; Zhouping Tang
>
> **摘要:** Introduction: Timely identification of intracranial hemorrhage (ICH) subtypes on non-contrast computed tomography is critical for prognosis prediction and therapeutic decision-making, yet remains challenging due to low contrast and blurring boundaries. This study evaluates the performance of zero-shot multi-modal large language models (MLLMs) compared to traditional deep learning methods in ICH binary classification and subtyping. Methods: We utilized a dataset provided by RSNA, comprising 192 NCCT volumes. The study compares various MLLMs, including GPT-4o, Gemini 2.0 Flash, and Claude 3.5 Sonnet V2, with conventional deep learning models, including ResNet50 and Vision Transformer. Carefully crafted prompts were used to guide MLLMs in tasks such as ICH presence, subtype classification, localization, and volume estimation. Results: The results indicate that in the ICH binary classification task, traditional deep learning models outperform MLLMs comprehensively. For subtype classification, MLLMs also exhibit inferior performance compared to traditional deep learning models, with Gemini 2.0 Flash achieving an macro-averaged precision of 0.41 and a macro-averaged F1 score of 0.31. Conclusion: While MLLMs excel in interactive capabilities, their overall accuracy in ICH subtyping is inferior to deep networks. However, MLLMs enhance interpretability through language interactions, indicating potential in medical imaging analysis. Future efforts will focus on model refinement and developing more precise MLLMs to improve performance in three-dimensional medical image processing.
>
---
#### [replaced 031] Refusal as Silence: Gendered Disparities in Vision-Language Model Responses
- **分类: cs.CV; cs.AI; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2406.08222v3](http://arxiv.org/pdf/2406.08222v3)**

> **作者:** Sha Luo; Sang Jung Kim; Zening Duan; Kaiping Chen
>
> **摘要:** Refusal behavior by Large Language Models is increasingly visible in content moderation, yet little is known about how refusals vary by the identity of the user making the request. This study investigates refusal as a sociotechnical outcome through a counterfactual persona design that varies gender identity--including male, female, non-binary, and transgender personas--while keeping the classification task and visual input constant. Focusing on a vision-language model (GPT-4V), we examine how identity-based language cues influence refusal in binary gender classification tasks. We find that transgender and non-binary personas experience significantly higher refusal rates, even in non-harmful contexts. Our findings also provide methodological implications for equity audits and content analysis using LLMs. Our findings underscore the importance of modeling identity-driven disparities and caution against uncritical use of AI systems for content coding. This study advances algorithmic fairness by reframing refusal as a communicative act that may unevenly regulate epistemic access and participation.
>
---
#### [replaced 032] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.04772v4](http://arxiv.org/pdf/2406.04772v4)**

> **作者:** Sungho Jeon; Xinyue Ma; Kwang In Kim; Myeongjae Jeon
>
> **摘要:** Recent rehearsal-free continual learning (CL) methods guided by prompts achieve strong performance on vision tasks with non-stationary data but remain resource-intensive, hindering real-world edge deployment. We introduce resource-efficient prompting (REP), which improves the computational and memory efficiency of prompt-based rehearsal-free continual learning methods while minimizing accuracy trade-offs. Our approach employs swift prompt selection to refine input data using a carefully provisioned model and introduces adaptive token merging (AToM) and adaptive layer dropping (ALD) for efficient prompt updates. AToM and ALD selectively skip data and model layers while preserving task-specific features during the learning of new tasks. Extensive experiments on multiple image classification datasets demonstrate REP's superior resource efficiency over state-of-the-art rehearsal-free CL methods.
>
---
#### [replaced 033] CannyEdit: Selective Canny Control and Dual-Prompt Guidance for Training-Free Image Editing
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.06937v2](http://arxiv.org/pdf/2508.06937v2)**

> **作者:** Weiyan Xie; Han Gao; Didan Deng; Kaican Li; April Hua Liu; Yongxiang Huang; Nevin L. Zhang
>
> **备注:** Project Page: vaynexie.github.io/CannyEdit/; MindSpore Code: github.com/mindspore-lab/mindone/tree/master/examples/canny_edit; PyTorch Code: github.com/vaynexie/CannyEdit
>
> **摘要:** Recent advances in text-to-image (T2I) models have enabled training-free regional image editing by leveraging the generative priors of foundation models. However, existing methods struggle to balance text adherence in edited regions, context fidelity in unedited areas, and seamless integration of edits. We introduce CannyEdit, a novel training-free framework that addresses this trilemma through two key innovations. First, Selective Canny Control applies structural guidance from a Canny ControlNet only to the unedited regions, preserving the original image's details while allowing for precise, text-driven changes in the specified editable area. Second, Dual-Prompt Guidance utilizes both a local prompt for the specific edit and a global prompt for overall scene coherence. Through this synergistic approach, these components enable controllable local editing for object addition, replacement, and removal, achieving a superior trade-off among text adherence, context fidelity, and editing seamlessness compared to current region-based methods. Beyond this, CannyEdit offers exceptional flexibility: it operates effectively with rough masks or even single-point hints in addition tasks. Furthermore, the framework can seamlessly integrate with vision-language models in a training-free manner for complex instruction-based editing that requires planning and reasoning. Our extensive evaluations demonstrate CannyEdit's strong performance against leading instruction-based editors in complex object addition scenarios.
>
---
#### [replaced 034] Reconstruction Alignment Improves Unified Multimodal Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.07295v3](http://arxiv.org/pdf/2509.07295v3)**

> **作者:** Ji Xie; Trevor Darrell; Luke Zettlemoyer; XuDong Wang
>
> **备注:** 34 pages, 28 figures and 11 tables; Update ablation study
>
> **摘要:** Unified multimodal models (UMMs) unify visual understanding and generation within a single architecture. However, conventional training relies on image-text pairs (or sequences) whose captions are typically sparse and miss fine-grained visual details--even when they use hundreds of words to describe a simple image. We introduce Reconstruction Alignment (RecA), a resource-efficient post-training method that leverages visual understanding encoder embeddings as dense "text prompts," providing rich supervision without captions. Concretely, RecA conditions a UMM on its own visual understanding embeddings and optimizes it to reconstruct the input image with a self-supervised reconstruction loss, thereby realigning understanding and generation. Despite its simplicity, RecA is broadly applicable: across autoregressive, masked-autoregressive, and diffusion-based UMMs, it consistently improves generation and editing fidelity. With only 27 GPU-hours, post-training with RecA substantially improves image generation performance on GenEval (0.73$\rightarrow$0.90) and DPGBench (80.93$\rightarrow$88.15), while also boosting editing benchmarks (ImgEdit 3.38$\rightarrow$3.75, GEdit 6.94$\rightarrow$7.25). Notably, RecA surpasses much larger open-source models and applies broadly across diverse UMM architectures, establishing it as an efficient and general post-training alignment strategy for UMMs
>
---
#### [replaced 035] Slot-BERT: Self-supervised Object Discovery in Surgical Video
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.12477v3](http://arxiv.org/pdf/2501.12477v3)**

> **作者:** Guiqiu Liao; Matjaz Jogan; Marcel Hussing; Kenta Nakahashi; Kazuhiro Yasufuku; Amin Madani; Eric Eaton; Daniel A. Hashimoto
>
> **备注:** 28 pages, 10 figures
>
> **摘要:** Object-centric slot attention is a powerful framework for unsupervised learning of structured and explainable representations that can support reasoning about objects and actions, including in surgical videos. While conventional object-centric methods for videos leverage recurrent processing to achieve efficiency, they often struggle with maintaining long-range temporal coherence required for long videos in surgical applications. On the other hand, fully parallel processing of entire videos enhances temporal consistency but introduces significant computational overhead, making it impractical for implementation on hardware in medical facilities. We present Slot-BERT, a bidirectional long-range model that learns object-centric representations in a latent space while ensuring robust temporal coherence. Slot-BERT scales object discovery seamlessly to long videos of unconstrained lengths. A novel slot contrastive loss further reduces redundancy and improves the representation disentanglement by enhancing slot orthogonality. We evaluate Slot-BERT on real-world surgical video datasets from abdominal, cholecystectomy, and thoracic procedures. Our method surpasses state-of-the-art object-centric approaches under unsupervised training achieving superior performance across diverse domains. We also demonstrate efficient zero-shot domain adaptation to data from diverse surgical specialties and databases.
>
---
#### [replaced 036] PlantSegNeRF: A few-shot, cross-species method for plant 3D instance point cloud reconstruction via joint-channel NeRF with multi-view image instance matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00371v3](http://arxiv.org/pdf/2507.00371v3)**

> **作者:** Xin Yang; Ruiming Du; Hanyang Huang; Jiayang Xie; Pengyao Xie; Leisen Fang; Ziyue Guo; Nanjun Jiang; Yu Jiang; Haiyan Cen
>
> **摘要:** Organ segmentation of plant point clouds is a prerequisite for the high-resolution and accurate extraction of organ-level phenotypic traits. Although the fast development of deep learning has boosted much research on segmentation of plant point clouds, the existing techniques for organ segmentation still face limitations in resolution, segmentation accuracy, and generalizability across various plant species. In this study, we proposed a novel approach called plant segmentation neural radiance fields (PlantSegNeRF), aiming to directly generate high-precision instance point clouds from multi-view RGB image sequences for a wide range of plant species. PlantSegNeRF performed 2D instance segmentation on the multi-view images to generate instance masks for each organ with a corresponding ID. The multi-view instance IDs corresponding to the same plant organ were then matched and refined using a specially designed instance matching module. The instance NeRF was developed to render an implicit scene, containing color, density, semantic and instance information. The implicit scene was ultimately converted into high-precision plant instance point clouds based on the volume density. The results proved that in semantic segmentation of point clouds, PlantSegNeRF outperformed the commonly used methods, demonstrating an average improvement of 16.1%, 18.3%, 17.8%, and 24.2% in precision, recall, F1-score, and IoU compared to the second-best results on structurally complex species. More importantly, PlantSegNeRF exhibited significant advantages in plant point cloud instance segmentation tasks. Across all plant species, it achieved average improvements of 11.7%, 38.2%, 32.2% and 25.3% in mPrec, mRec, mCov, mWCov, respectively. This study extends the organ-level plant phenotyping and provides a high-throughput way to supply high-quality 3D data for the development of large-scale models in plant science.
>
---
#### [replaced 037] AdFair-CLIP: Adversarial Fair Contrastive Language-Image Pre-training for Chest X-rays
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.23467v2](http://arxiv.org/pdf/2506.23467v2)**

> **作者:** Chenlang Yi; Zizhan Xiong; Qi Qi; Xiyuan Wei; Girish Bathla; Ching-Long Lin; Bobak Jack Mortazavi; Tianbao Yang
>
> **备注:** This preprint has been accepted by MICCAI 2025
>
> **摘要:** Contrastive Language-Image Pre-training (CLIP) models have demonstrated superior performance across various visual tasks including medical image classification. However, fairness concerns, including demographic biases, have received limited attention for CLIP models. This oversight leads to critical issues, particularly those related to race and gender, resulting in disparities in diagnostic outcomes and reduced reliability for underrepresented groups. To address these challenges, we introduce AdFair-CLIP, a novel framework employing adversarial feature intervention to suppress sensitive attributes, thereby mitigating spurious correlations and improving prediction fairness. We conduct comprehensive experiments on chest X-ray (CXR) datasets, and show that AdFair-CLIP significantly enhances both fairness and diagnostic accuracy, while maintaining robust generalization in zero-shot and few-shot scenarios. These results establish new benchmarks for fairness-aware learning in CLIP-based medical diagnostic models, particularly for CXR analysis.
>
---
#### [replaced 038] UKANFormer: Noise-Robust Semantic Segmentation for Coral Reef Mapping via a Kolmogorov-Arnold Network-Transformer Hybrid
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.16730v2](http://arxiv.org/pdf/2510.16730v2)**

> **作者:** Tianyang Dou; Ming Li; Jiangying Qin; Xuan Liao; Jiageng Zhong; Armin Gruen; Mengyi Deng
>
> **摘要:** Coral reefs are vital yet fragile ecosystems that require accurate large-scale mapping for effective conservation. Although global products such as the Allen Coral Atlas provide unprecedented coverage of global coral reef distri-bution, their predictions are frequently limited in spatial precision and semantic consistency, especially in regions requiring fine-grained boundary delineation. To address these challenges, we propose UKANFormer, a novel se-mantic segmentation model designed to achieve high-precision mapping under noisy supervision derived from Allen Coral Atlas. Building upon the UKAN architecture, UKANFormer incorporates a Global-Local Transformer (GL-Trans) block in the decoder, enabling the extraction of both global semantic structures and local boundary details. In experiments, UKANFormer achieved a coral-class IoU of 67.00% and pixel accuracy of 83.98%, outperforming conventional baselines under the same noisy labels setting. Remarkably, the model produces predictions that are visually and structurally more accurate than the noisy labels used for training. These results challenge the notion that data quality directly limits model performance, showing that architectural design can mitigate label noise and sup-port scalable mapping under imperfect supervision. UKANFormer provides a foundation for ecological monitoring where reliable labels are scarce.
>
---
#### [replaced 039] Unbiased Scene Graph Generation from Biased Training
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2002.11949v4](http://arxiv.org/pdf/2002.11949v4)**

> **作者:** Kaihua Tang; Yulei Niu; Jianqiang Huang; Jiaxin Shi; Hanwang Zhang
>
> **备注:** This paper is accepted by CVPR 2020. The code is publicly available on GitHub: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
>
> **摘要:** Today's scene graph generation (SGG) task is still far from practical, mainly due to the severe training bias, e.g., collapsing diverse "human walk on / sit on / lay on beach" into "human on beach". Given such SGG, the down-stream tasks such as VQA can hardly infer better scene structures than merely a bag of objects. However, debiasing in SGG is not trivial because traditional debiasing methods cannot distinguish between the good and bad bias, e.g., good context prior (e.g., "person read book" rather than "eat") and bad long-tailed bias (e.g., "near" dominating "behind / in front of"). In this paper, we present a novel SGG framework based on causal inference but not the conventional likelihood. We first build a causal graph for SGG, and perform traditional biased training with the graph. Then, we propose to draw the counterfactual causality from the trained graph to infer the effect from the bad bias, which should be removed. In particular, we use Total Direct Effect (TDE) as the proposed final predicate score for unbiased SGG. Note that our framework is agnostic to any SGG model and thus can be widely applied in the community who seeks unbiased predictions. By using the proposed Scene Graph Diagnosis toolkit on the SGG benchmark Visual Genome and several prevailing models, we observed significant improvements over the previous state-of-the-art methods.
>
---
#### [replaced 040] A Deep Learning-Based CCTV System for Automatic Smoking Detection in Fire Exit Zones
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.11696v2](http://arxiv.org/pdf/2508.11696v2)**

> **作者:** Sami Sadat; Mohammad Irtiza Hossain; Junaid Ahmed Sifat; Suhail Haque Rafi; Md. Waseq Alauddin Alvi; Md. Khalilur Rhaman
>
> **摘要:** A deep learning real-time smoking detection system for CCTV surveillance of fire exit areas is proposed due to critical safety requirements. The dataset contains 8,124 images from 20 different scenarios along with 2,708 raw samples demonstrating low-light areas. We evaluated three advanced object detection models: YOLOv8, YOLOv11, and YOLOv12, followed by development of a custom model derived from YOLOv8 with added structures for challenging surveillance contexts. The proposed model outperformed the others, achieving a recall of 78.90 percent and mAP at 50 of 83.70 percent, delivering optimal object detection across varied environments. Performance evaluation on multiple edge devices using multithreaded operations showed the Jetson Xavier NX processed data at 52 to 97 milliseconds per inference, establishing its suitability for time-sensitive operations. This system offers a robust and adaptable platform for monitoring public safety and enabling automatic regulatory compliance.
>
---
#### [replaced 041] VideoHallu: Evaluating and Mitigating Multi-modal Hallucinations on Synthetic Video Understanding
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.01481v4](http://arxiv.org/pdf/2505.01481v4)**

> **作者:** Zongxia Li; Xiyang Wu; Guangyao Shi; Yubin Qin; Hongyang Du; Fuxiao Liu; Tianyi Zhou; Dinesh Manocha; Jordan Lee Boyd-Graber
>
> **摘要:** Vision-Language Models (VLMs) have achieved strong results in video understanding, yet a key question remains: do they truly comprehend visual content or only learn shallow correlations between vision and language? Real visual understanding, especially of physics and common sense, is essential for AI systems that interact with the physical world. Current evaluations mostly use real-world videos similar to training data, so high benchmark scores may not reflect real reasoning ability. To address this, we propose negative-control tests using videos that depict physically impossible or logically inconsistent events. We introduce VideoHallu, a synthetic dataset of physics- and commonsense-violating scenes generated with Veo2, Sora, and Kling. It includes expert-annotated question-answer pairs across four categories of violations. Tests of leading VLMs (Qwen-2.5-VL, Video-R1, VideoChat-R1) show that, despite strong results on benchmarks such as MVBench and MMVU, they often miss these violations, exposing gaps in visual reasoning. Reinforcement learning fine-tuning on VideoHallu improves recognition of such violations without reducing standard benchmark performance. Our data is available at https://github.com/zli12321/VideoHallu.git.
>
---
#### [replaced 042] MECD+: Unlocking Event-Level Causal Graph Discovery for Video Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.07227v4](http://arxiv.org/pdf/2501.07227v4)**

> **作者:** Tieyuan Chen; Huabin Liu; Yi Wang; Yihang Chen; Tianyao He; Chaofan Gan; Huanyu He; Weiyao Lin
>
> **备注:** Accepted by IEEE TPAMI (IEEE Transactions on Pattern Analysis and Machine Intelligence)
>
> **摘要:** Video causal reasoning aims to achieve a high-level understanding of videos from a causal perspective. However, it exhibits limitations in its scope, primarily executed in a question-answering paradigm and focusing on brief video segments containing isolated events and basic causal relations, lacking comprehensive and structured causality analysis for videos with multiple interconnected events. To fill this gap, we introduce a new task and dataset, Multi-Event Causal Discovery (MECD). It aims to uncover the causal relations between events distributed chronologically across long videos. Given visual segments and textual descriptions of events, MECD identifies the causal associations between these events to derive a comprehensive and structured event-level video causal graph explaining why and how the result event occurred. To address the challenges of MECD, we devise a novel framework inspired by the Granger Causality method, incorporating an efficient mask-based event prediction model to perform an Event Granger Test. It estimates causality by comparing the predicted result event when premise events are masked versus unmasked. Furthermore, we integrate causal inference techniques such as front-door adjustment and counterfactual inference to mitigate challenges in MECD like causality confounding and illusory causality. Additionally, context chain reasoning is introduced to conduct more robust and generalized reasoning. Experiments validate the effectiveness of our framework in reasoning complete causal relations, outperforming GPT-4o and VideoChat2 by 5.77% and 2.70%, respectively. Further experiments demonstrate that causal relation graphs can also contribute to downstream video understanding tasks such as video question answering and video event prediction.
>
---
#### [replaced 043] Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.09110v3](http://arxiv.org/pdf/2502.09110v3)**

> **作者:** Eylon Mizrahi; Raz Lapid; Moshe Sipper
>
> **备注:** Accepted for Oral Presentation at SafeMM-AI @ ICCV 2025 (Spotlight)
>
> **摘要:** Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.
>
---
#### [replaced 044] A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10635v2](http://arxiv.org/pdf/2503.10635v2)**

> **作者:** Zhaoyi Li; Xiaohan Zhao; Dong-Dong Wu; Jiacheng Cui; Zhiqiang Shen
>
> **备注:** NeurIPS 2025. Code at: https://github.com/VILA-Lab/M-Attack
>
> **摘要:** Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against closed-source commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial black-box LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we propose to refine semantic clarity by encoding explicit semantic details within local regions, thus ensuring the capture of finer-grained features and inter-model transferability, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective baseline: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. While the naive source-target matching method has been utilized before in the literature, we are the first to provide a tight analysis, which establishes a close connection between perturbation optimization and semantics. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5/3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods with lower $\ell_1/\ell_2$ perturbations.
>
---
#### [replaced 045] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.08273v5](http://arxiv.org/pdf/2510.08273v5)**

> **作者:** Haipeng Liu; Yang Wang; Meng Wang
>
> **备注:** 27 pages, 11 figures, to appear at NeurIPS 2025
>
> **摘要:** Text-guided image inpainting aims at reconstructing the masked regions as per text prompts, where the longstanding challenges lie in the preservation for unmasked regions, while achieving the semantics consistency between unmasked and inpainted masked regions. Previous arts failed to address both of them, always with either of them to be remedied. Such facts, as we observed, stem from the entanglement of the hybrid (e.g., mid-and-low) frequency bands that encode varied image properties, which exhibit different robustness to text prompts during the denoising process. In this paper, we propose a null-text-null frequency-aware diffusion models, dubbed \textbf{NTN-Diff}, for text-guided image inpainting, by decomposing the semantics consistency across masked and unmasked regions into the consistencies as per each frequency band, while preserving the unmasked regions, to circumvent two challenges in a row. Based on the diffusion process, we further divide the denoising process into early (high-level noise) and late (low-level noise) stages, where the mid-and-low frequency bands are disentangled during the denoising process. As observed, the stable mid-frequency band is progressively denoised to be semantically aligned during text-guided denoising process, which, meanwhile, serves as the guidance to the null-text denoising process to denoise low-frequency band for the masked regions, followed by a subsequent text-guided denoising process at late stage, to achieve the semantics consistency for mid-and-low frequency bands across masked and unmasked regions, while preserve the unmasked regions. Extensive experiments validate the superiority of NTN-Diff over the state-of-the-art diffusion models to text-guided diffusion models. Our code can be accessed from https://github.com/htyjers/NTN-Diff.
>
---
#### [replaced 046] C-DiffDet+: Fusing Global Scene Context with Generative Denoising for High-Fidelity Car Damage Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00578v4](http://arxiv.org/pdf/2509.00578v4)**

> **作者:** Abdellah Zakaria Sellam; Ilyes Benaissa; Salah Eddine Bekhouche; Abdenour Hadid; Vito Renó; Cosimo Distante
>
> **摘要:** Fine-grained object detection in challenging visual domains, such as vehicle damage assessment, presents a formidable challenge even for human experts to resolve reliably. While DiffusionDet has advanced the state-of-the-art through conditional denoising diffusion, its performance remains limited by local feature conditioning in context-dependent scenarios. We address this fundamental limitation by introducing Context-Aware Fusion (CAF), which leverages cross-attention mechanisms to integrate global scene context with local proposal features directly. The global context is generated using a separate dedicated encoder that captures comprehensive environmental information, enabling each object proposal to attend to scene-level understanding. Our framework significantly enhances the generative detection paradigm by enabling each object proposal to attend to comprehensive environmental information. Experimental results demonstrate an improvement over state-of-the-art models on the CarDD benchmark, establishing new performance benchmarks for context-aware object detection in fine-grained domains
>
---
#### [replaced 047] Identity-Preserving Text-to-Video Generation Guided by Simple yet Effective Spatial-Temporal Decoupled Representations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04705v3](http://arxiv.org/pdf/2507.04705v3)**

> **作者:** Yuji Wang; Moran Li; Xiaobin Hu; Ran Yi; Jiangning Zhang; Han Feng; Weijian Cao; Yabiao Wang; Chengjie Wang; Lizhuang Ma
>
> **备注:** ACM Multimedia 2025; code URL: https://github.com/rain152/IPVG
>
> **摘要:** Identity-preserving text-to-video (IPT2V) generation, which aims to create high-fidelity videos with consistent human identity, has become crucial for downstream applications. However, current end-to-end frameworks suffer a critical spatial-temporal trade-off: optimizing for spatially coherent layouts of key elements (e.g., character identity preservation) often compromises instruction-compliant temporal smoothness, while prioritizing dynamic realism risks disrupting the spatial coherence of visual structures. To tackle this issue, we propose a simple yet effective spatial-temporal decoupled framework that decomposes representations into spatial features for layouts and temporal features for motion dynamics. Specifically, our paper proposes a semantic prompt optimization mechanism and stage-wise decoupled generation paradigm. The former module decouples the prompt into spatial and temporal components. Aligned with the subsequent stage-wise decoupled approach, the spatial prompts guide the text-to-image (T2I) stage to generate coherent spatial features, while the temporal prompts direct the sequential image-to-video (I2V) stage to ensure motion consistency. Experimental results validate that our approach achieves excellent spatiotemporal consistency, demonstrating outstanding performance in identity preservation, text relevance, and video quality. By leveraging this simple yet robust mechanism, our algorithm secures the runner-up position in 2025 ACM MultiMedia Challenge. Our code is available at https://github.com/rain152/IPVG.
>
---
#### [replaced 048] Kuramoto Orientation Diffusion Models
- **分类: cs.LG; cs.CV; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2509.15328v2](http://arxiv.org/pdf/2509.15328v2)**

> **作者:** Yue Song; T. Anderson Keller; Sevan Brodjian; Takeru Miyato; Yisong Yue; Pietro Perona; Max Welling
>
> **备注:** NeurIPS 2025
>
> **摘要:** Orientation-rich images, such as fingerprints and textures, often exhibit coherent angular directional patterns that are challenging to model using standard generative approaches based on isotropic Euclidean diffusion. Motivated by the role of phase synchronization in biological systems, we propose a score-based generative model built on periodic domains by leveraging stochastic Kuramoto dynamics in the diffusion process. In neural and physical systems, Kuramoto models capture synchronization phenomena across coupled oscillators -- a behavior that we re-purpose here as an inductive bias for structured image generation. In our framework, the forward process performs \textit{synchronization} among phase variables through globally or locally coupled oscillator interactions and attraction to a global reference phase, gradually collapsing the data into a low-entropy von Mises distribution. The reverse process then performs \textit{desynchronization}, generating diverse patterns by reversing the dynamics with a learned score function. This approach enables structured destruction during forward diffusion and a hierarchical generation process that progressively refines global coherence into fine-scale details. We implement wrapped Gaussian transition kernels and periodicity-aware networks to account for the circular geometry. Our method achieves competitive results on general image benchmarks and significantly improves generation quality on orientation-dense datasets like fingerprints and textures. Ultimately, this work demonstrates the promise of biologically inspired synchronization dynamics as structured priors in generative modeling.
>
---
#### [replaced 049] DERD-Net: Learning Depth from Event-based Ray Densities
- **分类: cs.CV; cs.LG; cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2504.15863v2](http://arxiv.org/pdf/2504.15863v2)**

> **作者:** Diego Hitzges; Suman Ghosh; Guillermo Gallego
>
> **备注:** 17 pages, 3 figures, 15 tables. Project page: https://github.com/tub-rip/DERD-Net. 39th Conference on Neural Information Processing Systems (NeurIPS), San Diego, 2025
>
> **摘要:** Event cameras offer a promising avenue for multi-view stereo depth estimation and Simultaneous Localization And Mapping (SLAM) due to their ability to detect blur-free 3D edges at high-speed and over broad illumination conditions. However, traditional deep learning frameworks designed for conventional cameras struggle with the asynchronous, stream-like nature of event data, as their architectures are optimized for discrete, image-like inputs. We propose a scalable, flexible and adaptable framework for pixel-wise depth estimation with event cameras in both monocular and stereo setups. The 3D scene structure is encoded into disparity space images (DSIs), representing spatial densities of rays obtained by back-projecting events into space via known camera poses. Our neural network processes local subregions of the DSIs combining 3D convolutions and a recurrent structure to recognize valuable patterns for depth prediction. Local processing enables fast inference with full parallelization and ensures constant ultra-low model complexity and memory costs, regardless of camera resolution. Experiments on standard benchmarks (MVSEC and DSEC datasets) demonstrate unprecedented effectiveness: (i) using purely monocular data, our method achieves comparable results to existing stereo methods; (ii) when applied to stereo data, it strongly outperforms all state-of-the-art (SOTA) approaches, reducing the mean absolute error by at least 42%; (iii) our method also allows for increases in depth completeness by more than 3-fold while still yielding a reduction in median absolute error of at least 30%. Given its remarkable performance and effective processing of event-data, our framework holds strong potential to become a standard approach for using deep learning for event-based depth estimation and SLAM. Project page: https://github.com/tub-rip/DERD-Net
>
---
#### [replaced 050] Flow-GRPO: Training Flow Matching Models via Online RL
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.05470v5](http://arxiv.org/pdf/2505.05470v5)**

> **作者:** Jie Liu; Gongye Liu; Jiajun Liang; Yangguang Li; Jiaheng Liu; Xintao Wang; Pengfei Wan; Di Zhang; Wanli Ouyang
>
> **备注:** Code: https://github.com/yifan123/flow_grpo
>
> **摘要:** We propose Flow-GRPO, the first method to integrate online policy gradient reinforcement learning (RL) into flow matching models. Our approach uses two key strategies: (1) an ODE-to-SDE conversion that transforms a deterministic Ordinary Differential Equation (ODE) into an equivalent Stochastic Differential Equation (SDE) that matches the original model's marginal distribution at all timesteps, enabling statistical sampling for RL exploration; and (2) a Denoising Reduction strategy that reduces training denoising steps while retaining the original number of inference steps, significantly improving sampling efficiency without sacrificing performance. Empirically, Flow-GRPO is effective across multiple text-to-image tasks. For compositional generation, RL-tuned SD3.5-M generates nearly perfect object counts, spatial relations, and fine-grained attributes, increasing GenEval accuracy from $63\%$ to $95\%$. In visual text rendering, accuracy improves from $59\%$ to $92\%$, greatly enhancing text generation. Flow-GRPO also achieves substantial gains in human preference alignment. Notably, very little reward hacking occurred, meaning rewards did not increase at the cost of appreciable image quality or diversity degradation.
>
---
#### [replaced 051] Representational Difference Explanations
- **分类: cs.CV; cs.AI; cs.LG; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2505.23917v2](http://arxiv.org/pdf/2505.23917v2)**

> **作者:** Neehar Kondapaneni; Oisin Mac Aodha; Pietro Perona
>
> **备注:** 9 pages, 6 figures, 21 supplementary pages, 14 supp figs
>
> **摘要:** We propose a method for discovering and visualizing the differences between two learned representations, enabling more direct and interpretable model comparisons. We validate our method, which we call Representational Differences Explanations (RDX), by using it to compare models with known conceptual differences and demonstrate that it recovers meaningful distinctions where existing explainable AI (XAI) techniques fail. Applied to state-of-the-art models on challenging subsets of the ImageNet and iNaturalist datasets, RDX reveals both insightful representational differences and subtle patterns in the data. Although comparison is a cornerstone of scientific analysis, current tools in machine learning, namely post hoc XAI methods, struggle to support model comparison effectively. Our work addresses this gap by introducing an effective and explainable tool for contrasting model representations.
>
---
#### [replaced 052] THUNDER: Tile-level Histopathology image UNDERstanding benchmark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07860v2](http://arxiv.org/pdf/2507.07860v2)**

> **作者:** Pierre Marza; Leo Fillioux; Sofiène Boutaj; Kunal Mahatha; Christian Desrosiers; Pablo Piantanida; Jose Dolz; Stergios Christodoulidis; Maria Vakalopoulou
>
> **备注:** Accepted at NeurIPS 2025 Datasets and Benchmarks Track (Spotlight)
>
> **摘要:** Progress in a research field can be hard to assess, in particular when many concurrent methods are proposed in a short period of time. This is the case in digital pathology, where many foundation models have been released recently to serve as feature extractors for tile-level images, being used in a variety of downstream tasks, both for tile- and slide-level problems. Benchmarking available methods then becomes paramount to get a clearer view of the research landscape. In particular, in critical domains such as healthcare, a benchmark should not only focus on evaluating downstream performance, but also provide insights about the main differences between methods, and importantly, further consider uncertainty and robustness to ensure a reliable usage of proposed models. For these reasons, we introduce THUNDER, a tile-level benchmark for digital pathology foundation models, allowing for efficient comparison of many models on diverse datasets with a series of downstream tasks, studying their feature spaces and assessing the robustness and uncertainty of predictions informed by their embeddings. THUNDER is a fast, easy-to-use, dynamic benchmark that can already support a large variety of state-of-the-art foundation, as well as local user-defined models for direct tile-based comparison. In this paper, we provide a comprehensive comparison of 23 foundation models on 16 different datasets covering diverse tasks, feature analysis, and robustness. The code for THUNDER is publicly available at https://github.com/MICS-Lab/thunder.
>
---
#### [replaced 053] CMIE: Combining MLLM Insights with External Evidence for Explainable Out-of-Context Misinformation Detection
- **分类: cs.MM; cs.CV; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.23449v3](http://arxiv.org/pdf/2505.23449v3)**

> **作者:** Fanxiao Li; Jiaying Wu; Canyuan He; Wei Zhou
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated impressive capabilities in visual reasoning and text generation. While previous studies have explored the application of MLLM for detecting out-of-context (OOC) misinformation, our empirical analysis reveals two persisting challenges of this paradigm. Evaluating the representative GPT-4o model on direct reasoning and evidence augmented reasoning, results indicate that MLLM struggle to capture the deeper relationships-specifically, cases in which the image and text are not directly connected but are associated through underlying semantic links. Moreover, noise in the evidence further impairs detection accuracy. To address these challenges, we propose CMIE, a novel OOC misinformation detection framework that incorporates a Coexistence Relationship Generation (CRG) strategy and an Association Scoring (AS) mechanism. CMIE identifies the underlying coexistence relationships between images and text, and selectively utilizes relevant evidence to enhance misinformation detection. Experimental results demonstrate that our approach outperforms existing methods.
>
---
#### [replaced 054] Unsupervised Document and Template Clustering using Multimodal Embeddings
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12116v3](http://arxiv.org/pdf/2506.12116v3)**

> **作者:** Phillipe R. Sampaio; Helene Maxcici
>
> **备注:** 24 pages, 12 figures
>
> **摘要:** We study unsupervised clustering of documents at both the category and template levels using frozen multimodal encoders and classical clustering algorithms. We systematize a model-agnostic pipeline that (i) projects heterogeneous last-layer states from text-layout-vision encoders into token-type-aware document vectors and (ii) performs clustering with centroid- or density-based methods, including an HDBSCAN + $k$-NN assignment to eliminate unlabeled points. We evaluate eight encoders (text-only, layout-aware, vision-only, and vision-language) with $k$-Means, DBSCAN, HDBSCAN + $k$-NN, and BIRCH on five corpora spanning clean synthetic invoices, their heavily degraded print-and-scan counterparts, scanned receipts, and real identity and certificate documents. The study reveals modality-specific failure modes and a robustness-accuracy trade-off, with vision features nearly solving template discovery on clean pages while text dominates under covariate shift, and fused encoders offering the best balance. We detail a reproducible, oracle-free tuning protocol and the curated evaluation settings to guide future work on unsupervised document organization.
>
---
#### [replaced 055] A Cycle Ride to HDR: Semantics Aware Self-Supervised Framework for Unpaired LDR-to-HDR Image Reconstruction
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO; Artificial intelligence, Computer vision, Machine learning, Deep
  learning; I.3.3; I.4.5**

- **链接: [http://arxiv.org/pdf/2410.15068v4](http://arxiv.org/pdf/2410.15068v4)**

> **作者:** Hrishav Bakul Barua; Kalin Stefanov; Lemuel Lai En Che; Abhinav Dhall; KokSheik Wong; Ganesh Krishnasamy
>
> **摘要:** Reconstruction of High Dynamic Range (HDR) from Low Dynamic Range (LDR) images is an important computer vision task. There is a significant amount of research utilizing both conventional non-learning methods and modern data-driven approaches, focusing on using both single-exposed and multi-exposed LDR for HDR image reconstruction. However, most current state-of-the-art methods require high-quality paired {LDR;HDR} datasets with limited literature use of unpaired datasets, that is, methods that learn the LDR-HDR mapping between domains. This paper proposes CycleHDR, a method that integrates self-supervision into a modified semantic- and cycle-consistent adversarial architecture that utilizes unpaired LDR and HDR datasets for training. Our method introduces novel artifact- and exposure-aware generators to address visual artifact removal. It also puts forward an encoder and loss to address semantic consistency, another under-explored topic. CycleHDR is the first to use semantic and contextual awareness for the LDR-HDR reconstruction task in a self-supervised setup. The method achieves state-of-the-art performance across several benchmark datasets and reconstructs high-quality HDR images. The official website of this work is available at: https://github.com/HrishavBakulBarua/Cycle-HDR
>
---
#### [replaced 056] Can Less Precise Be More Reliable? A Systematic Evaluation of Quantization's Impact on CLIP Beyond Accuracy
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.21173v3](http://arxiv.org/pdf/2509.21173v3)**

> **作者:** Aymen Bouguerra; Daniel Montoya; Alexandra Gomez-Villa; Fabio Arnez; Chokri Mraidha
>
> **备注:** Preprint, under peer review
>
> **摘要:** The powerful zero-shot generalization capabilities of vision-language models (VLMs) like CLIP have enabled new paradigms for safety-related tasks such as out-of-distribution (OOD) detection. However, additional aspects crucial for the computationally efficient and reliable deployment of CLIP are still overlooked. In particular, the impact of quantization on CLIP's performance beyond accuracy remains underexplored. This work presents a large-scale evaluation of quantization on CLIP models, assessing not only in-distribution accuracy but a comprehensive suite of reliability metrics and revealing counterintuitive results driven by pre-training source. We demonstrate that quantization consistently improves calibration for typically underconfident pre-trained models, while often degrading it for overconfident variants. Intriguingly, this degradation in calibration does not preclude gains in other reliability metrics; we find that OOD detection can still improve for these same poorly calibrated models. Furthermore, we identify specific quantization-aware training (QAT) methods that yield simultaneous gains in zero-shot accuracy, calibration, and OOD robustness, challenging the view of a strict efficiency-performance trade-off. These findings offer critical insights for navigating the multi-objective problem of deploying efficient, reliable, and robust VLMs by utilizing quantization beyond its conventional role.
>
---
#### [replaced 057] Attention! Your Vision Language Model Could Be Maliciously Manipulated
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19911v2](http://arxiv.org/pdf/2505.19911v2)**

> **作者:** Xiaosen Wang; Shaokang Wang; Zhijin Ge; Yuyang Luo; Shudong Zhang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets. Code is available at https://github.com/Trustworthy-AI-Group/VMA.
>
---
#### [replaced 058] Beyond sparse denoising in frames: minimax estimation with a scattering transform
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.19612v2](http://arxiv.org/pdf/2510.19612v2)**

> **作者:** Nathanaël Cuvelle--Magar; Stéphane Mallat
>
> **摘要:** A considerable amount of research in harmonic analysis has been devoted to non-linear estimators of signals contaminated by additive Gaussian noise. They are implemented by thresholding coefficients in a frame, which provide a sparse signal representation, or by minimising their $\ell^1$ norm. However, sparse estimators in frames are not sufficiently rich to adapt to complex signal regularities. For cartoon images whose edges are piecewise $\bf C^\alpha$ curves, wavelet, curvelet and Xlet frames are suboptimal if the Lipschitz exponent $\alpha \leq 2$ is an unknown parameter. Deep convolutional neural networks have recently obtained much better numerical results, which reach the minimax asymptotic bounds for all $\alpha$. Wavelet scattering coefficients have been introduced as simplified convolutional neural network models. They are computed by transforming the modulus of wavelet coefficients with a second wavelet transform. We introduce a denoising estimator by jointly minimising and maximising the $\ell^1$ norms of different subsets of scattering coefficients. We prove that these $\ell^1$ norms capture different types of geometric image regularity. Numerical experiments show that this denoising estimator reaches the minimax asymptotic bound for cartoon images for all Lipschitz exponents $\alpha \leq 2$. We state this numerical result as a mathematical conjecture. It provides a different harmonic analysis approach to suppress noise from signals, and to specify the geometric regularity of functions. It also opens a mathematical bridge between harmonic analysis and denoising estimators with deep convolutional network.
>
---
#### [replaced 059] Holistic Order Prediction in Natural Scenes
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.01704v2](http://arxiv.org/pdf/2510.01704v2)**

> **作者:** Pierre Musacchio; Hyunmin Lee; Jaesik Park
>
> **备注:** 24 pages, 11 figures, 6 tables
>
> **摘要:** Even in controlled settings, understanding instance-wise geometries is a challenging task for a wide range of visual models. Although specialized systems exist, modern arts rely on expensive input formats (category labels, binary segmentation masks) and inference costs (a quadratic amount of forward passes). We mitigate these limitations by proposing InstaFormer, a network capable of holistic order prediction. That is, solely given an input RGB image, InstaFormer returns the full occlusion and depth orderings for all the instances in the scene in a single forward pass. At its core, InstaFormer relies on interactions between object queries and latent mask descriptors that semantically represent the same objects while carrying complementary information. We comprehensively benchmark and ablate our approach to highlight its effectiveness. Our code and models are open-source and available at this URL: https://github.com/SNU-VGILab/InstaOrder.
>
---
#### [replaced 060] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04308v3](http://arxiv.org/pdf/2506.04308v3)**

> **作者:** Enshen Zhou; Jingkun An; Cheng Chi; Yi Han; Shanyu Rong; Chi Zhang; Pengwei Wang; Zhongyuan Wang; Tiejun Huang; Lu Sheng; Shanghang Zhang
>
> **备注:** Accepted by NeurIPS 2025. Project page: https://zhoues.github.io/RoboRefer/
>
> **摘要:** Spatial referring is a fundamental capability of embodied robots to interact with the 3D physical world. However, even with the powerful pretrained vision language models (VLMs), recent approaches are still not qualified to accurately understand the complex 3D scenes and dynamically reason about the instruction-indicated locations for interaction. To this end, we propose RoboRefer, a 3D-aware VLM that can first achieve precise spatial understanding by integrating a disentangled but dedicated depth encoder via supervised fine-tuning (SFT). Moreover, RoboRefer advances generalized multi-step spatial reasoning via reinforcement fine-tuning (RFT), with metric-sensitive process reward functions tailored for spatial referring tasks. To support SFT and RFT training, we introduce RefSpatial, a large-scale dataset of 20M QA pairs (2x prior), covering 31 spatial relations (vs. 15 prior) and supporting complex reasoning processes (up to 5 steps). In addition, we introduce RefSpatial-Bench, a challenging benchmark filling the gap in evaluating spatial referring with multi-step reasoning. Experiments show that SFT-trained RoboRefer achieves state-of-the-art spatial understanding, with an average success rate of 89.6%. RFT-trained RoboRefer further outperforms all other baselines by a large margin, even surpassing Gemini-2.5-Pro by 17.4% in average accuracy on RefSpatial-Bench. Notably, RoboRefer can be integrated with various control policies to execute long-horizon, dynamic tasks across diverse robots (e,g., UR5, G1 humanoid) in cluttered real-world scenes. Please see the project page at https://zhoues.github.io/RoboRefer.
>
---
#### [replaced 061] Gesplat: Robust Pose-Free 3D Reconstruction via Geometry-Guided Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.10097v2](http://arxiv.org/pdf/2510.10097v2)**

> **作者:** Jiahui Lu; Haihong Xiao; Xueyan Zhao; Wenxiong Kang
>
> **摘要:** Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have advanced 3D reconstruction and novel view synthesis, but remain heavily dependent on accurate camera poses and dense viewpoint coverage. These requirements limit their applicability in sparse-view settings, where pose estimation becomes unreliable and supervision is insufficient. To overcome these challenges, we introduce Gesplat, a 3DGS-based framework that enables robust novel view synthesis and geometrically consistent reconstruction from unposed sparse images. Unlike prior works that rely on COLMAP for sparse point cloud initialization, we leverage the VGGT foundation model to obtain more reliable initial poses and dense point clouds. Our approach integrates several key innovations: 1) a hybrid Gaussian representation with dual position-shape optimization enhanced by inter-view matching consistency; 2) a graph-guided attribute refinement module to enhance scene details; and 3) flow-based depth regularization that improves depth estimation accuracy for more effective supervision. Comprehensive quantitative and qualitative experiments demonstrate that our approach achieves more robust performance on both forward-facing and large-scale complex datasets compared to other pose-free methods.
>
---
#### [replaced 062] GOPLA: Generalizable Object Placement Learning via Synthetic Augmentation of Human Arrangement
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.14627v2](http://arxiv.org/pdf/2510.14627v2)**

> **作者:** Yao Zhong; Hanzhi Chen; Simon Schaefer; Anran Zhang; Stefan Leutenegger
>
> **摘要:** Robots are expected to serve as intelligent assistants, helping humans with everyday household organization. A central challenge in this setting is the task of object placement, which requires reasoning about both semantic preferences (e.g., common-sense object relations) and geometric feasibility (e.g., collision avoidance). We present GOPLA, a hierarchical framework that learns generalizable object placement from augmented human demonstrations. A multi-modal large language model translates human instructions and visual inputs into structured plans that specify pairwise object relationships. These plans are then converted into 3D affordance maps with geometric common sense by a spatial mapper, while a diffusion-based planner generates placement poses guided by test-time costs, considering multi-plan distributions and collision avoidance. To overcome data scarcity, we introduce a scalable pipeline that expands human placement demonstrations into diverse synthetic training data. Extensive experiments show that our approach improves placement success rates by 30.04 percentage points over the runner-up, evaluated on positioning accuracy and physical plausibility, demonstrating strong generalization across a wide range of real-world robotic placement scenarios.
>
---
#### [replaced 063] Robust Modality-incomplete Anomaly Detection: A Modality-instructive Framework with Benchmark
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2410.01737v2](http://arxiv.org/pdf/2410.01737v2)**

> **作者:** Bingchen Miao; Wenqiao Zhang; Juncheng Li; Wangyu Wu; Siliang Tang; Zhaocheng Li; Haochen Shi; Jun Xiao; Yueting Zhuang
>
> **摘要:** Multimodal Industrial Anomaly Detection (MIAD), which utilizes 3D point clouds and 2D RGB images to identify abnormal regions in products, plays a crucial role in industrial quality inspection. However, traditional MIAD settings assume that all 2D and 3D modalities are paired, ignoring the fact that multimodal data collected from the real world is often imperfect due to missing modalities. Additionally, models trained on modality-incomplete data are prone to overfitting. Therefore, MIAD models that demonstrate robustness against modality-incomplete data are highly desirable in practice. To address this, we introduce a pioneering study that comprehensively investigates Modality-Incomplete Industrial Anomaly Detection (MIIAD), and under the guidance of experts, we construct the MIIAD Bench with rich modality-missing settings to account for imperfect learning environments with incomplete multimodal information. As expected, we find that most existing MIAD methods perform poorly on the MIIAD Bench, leading to significant performance degradation. To tackle this challenge, we propose a novel two-stage Robust modAlity-aware fusing and Detecting framewoRk, abbreviated as RADAR. Specifically: i) We propose Modality-incomplete Instruction to guide the multimodal Transformer to robustly adapt to various modality-incomplete scenarios, and implement adaptive parameter learning based on HyperNetwork. ii) Then, we construct a Double-Pseudo Hybrid Module to highlight the uniqueness of modality combinations, mitigating overfitting issues and further enhancing the robustness of the MIIAD model. Our experimental results demonstrate that the proposed RADAR significantly outperforms traditional MIAD methods on our newly created MIIAD dataset, proving its practical application value.
>
---
#### [replaced 064] Measuring the (Un)Faithfulness of Concept-Based Explanations
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10833v2](http://arxiv.org/pdf/2504.10833v2)**

> **作者:** Shubham Kumar; Narendra Ahuja
>
> **备注:** Pre-print
>
> **摘要:** Post-hoc, unsupervised concept-based explanation methods (U-CBEMs) translate a vision model's internal reasoning into human-understandable concepts, leading to interpretable explanations. However, we find that many state-of-the-art (SOTA) U-CBEMs are not faithful: their concepts seem interpretable but fail to reproduce the model's predictions. We argue that this deficiency has gone unnoticed due to fragmented evaluation - each paper proposes its own faithfulness measure, with no measure-over-measure comparison or broad benchmarking. We close this gap by (i) organizing prior metrics in a unified framework, discussing their limitations, and identifying desiderata for a faithfulness measure; (ii) introducing the Surrogate Faithfulness (SURF) measure, which quantifies faithfulness via the predictive loss of a surrogate that maps explanations to the model's outputs; and (iii) delivering the first comprehensive U-CBEM faithfulness benchmark across diverse tasks and architectures. In a controlled setting, SURF outperforms prior faithfulness measures in measure-over-measure comparisons, and applying SURF to SOTA U-CBEMs reveals that many visually appealing U-CBEMs are surprisingly unfaithful. We demonstrate SURF applicability in two downstream settings - (i) faithfulness versus the number of concepts used in the explanation and (ii) U-CBEM robustness to adversarial attacks - underscoring SURF's value as a reliable faithfulness measure. Code to be released.
>
---
#### [replaced 065] Net2Net: When Un-trained Meets Pre-trained Networks for Robust Real-World Denoising
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.02733v2](http://arxiv.org/pdf/2510.02733v2)**

> **作者:** Weimin Yuan; Cai Meng
>
> **摘要:** Traditional denoising methods for noise removal have largely relied on handcrafted priors, often perform well in controlled environments but struggle to address the complexity and variability of real noise. In contrast, deep learning-based approaches have gained prominence for learning noise characteristics from large datasets, but these methods frequently require extensive labeled data and may not generalize effectively across diverse noise types and imaging conditions. In this paper, we present an innovative method, termed as Net2Net, that combines the strengths of untrained and pre-trained networks to tackle the challenges of real-world noise removal. The innovation of Net2Net lies in its combination of unsupervised DIP and supervised pre-trained model DRUNet by regularization by denoising (RED). The untrained network adapts to the unique noise characteristics of each input image without requiring labeled data, while the pre-trained network leverages learned representations from large-scale datasets to deliver robust denoising performance. This hybrid framework enhances generalization across varying noise patterns and improves performance, particularly in scenarios with limited training data. Extensive experiments on benchmark datasets demonstrate the superiority of our method for real-world noise removal.
>
---
#### [replaced 066] T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20625v3](http://arxiv.org/pdf/2502.20625v3)**

> **作者:** Yifei Qian; Zhongliang Guo; Bowen Deng; Chun Tong Lei; Shuai Zhao; Chun Pong Lau; Xiaopeng Hong; Michael P. Pound
>
> **备注:** Accepted by CVPR2025
>
> **摘要:** Zero-shot object counting aims to count instances of arbitrary object categories specified by text descriptions. Existing methods typically rely on vision-language models like CLIP, but often exhibit limited sensitivity to text prompts. We present T2ICount, a diffusion-based framework that leverages rich prior knowledge and fine-grained visual understanding from pretrained diffusion models. While one-step denoising ensures efficiency, it leads to weakened text sensitivity. To address this challenge, we propose a Hierarchical Semantic Correction Module that progressively refines text-image feature alignment, and a Representational Regional Coherence Loss that provides reliable supervision signals by leveraging the cross-attention maps extracted from the denosing U-Net. Furthermore, we observe that current benchmarks mainly focus on majority objects in images, potentially masking models' text sensitivity. To address this, we contribute a challenging re-annotated subset of FSC147 for better evaluation of text-guided counting ability. Extensive experiments demonstrate that our method achieves superior performance across different benchmarks. Code is available at https://github.com/cha15yq/T2ICount.
>
---
#### [replaced 067] EVODiff: Entropy-aware Variance Optimized Diffusion Inference
- **分类: cs.CV; cs.IT; cs.LG; math.IT; math.OC; stat.ML**

- **链接: [http://arxiv.org/pdf/2509.26096v2](http://arxiv.org/pdf/2509.26096v2)**

> **作者:** Shigui Li; Wei Chen; Delu Zeng
>
> **备注:** NeurIPS 2025, 40 pages, 14 figures
>
> **摘要:** Diffusion models (DMs) excel in image generation, but suffer from slow inference and the training-inference discrepancies. Although gradient-based solvers like DPM-Solver accelerate the denoising inference, they lack theoretical foundations in information transmission efficiency. In this work, we introduce an information-theoretic perspective on the inference processes of DMs, revealing that successful denoising fundamentally reduces conditional entropy in reverse transitions. This principle leads to our key insights into the inference processes: (1) data prediction parameterization outperforms its noise counterpart, and (2) optimizing conditional variance offers a reference-free way to minimize both transition and reconstruction errors. Based on these insights, we propose an entropy-aware variance optimized method for the generative process of DMs, called EVODiff, which systematically reduces uncertainty by optimizing conditional entropy during denoising. Extensive experiments on DMs validate our insights and demonstrate that our method significantly and consistently outperforms state-of-the-art (SOTA) gradient-based solvers. For example, compared to the DPM-Solver++, EVODiff reduces the reconstruction error by up to 45.5\% (FID improves from 5.10 to 2.78) at 10 function evaluations (NFE) on CIFAR-10, cuts the NFE cost by 25\% (from 20 to 15 NFE) for high-quality samples on ImageNet-256, and improves text-to-image generation while reducing artifacts. Code is available at https://github.com/ShiguiLi/EVODiff.
>
---
#### [replaced 068] Synthesize Privacy-Preserving High-Resolution Images via Private Textual Intermediaries
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07555v3](http://arxiv.org/pdf/2506.07555v3)**

> **作者:** Haoxiang Wang; Zinan Lin; Da Yu; Huishuai Zhang
>
> **摘要:** Generating high fidelity, differentially private (DP) synthetic images offers a promising route to share and analyze sensitive visual data without compromising individual privacy. However, existing DP image synthesis methods struggle to produce high resolution outputs that faithfully capture the structure of the original data. In this paper, we introduce a novel method, referred to as Synthesis via Private Textual Intermediaries (SPTI), that can generate high resolution DP images with easy adoption. The key idea is to shift the challenge of DP image synthesis from the image domain to the text domain by leveraging state of the art DP text generation methods. SPTI first summarizes each private image into a concise textual description using image to text models, then applies a modified Private Evolution algorithm to generate DP text, and finally reconstructs images using text to image models. Notably, SPTI requires no model training, only inference with off the shelf models. Given a private dataset, SPTI produces synthetic images of substantially higher quality than prior DP approaches. On the LSUN Bedroom dataset, SPTI attains an FID equal to 26.71 under epsilon equal to 1.0, improving over Private Evolution FID of 40.36. Similarly, on MM CelebA HQ, SPTI achieves an FID equal to 33.27 at epsilon equal to 1.0, compared to 57.01 from DP fine tuning baselines. Overall, our results demonstrate that Synthesis via Private Textual Intermediaries provides a resource efficient and proprietary model compatible framework for generating high resolution DP synthetic images, greatly expanding access to private visual datasets.
>
---
#### [replaced 069] Unmasking Puppeteers: Leveraging Biometric Leakage to Disarm Impersonation in AI-based Videoconferencing
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.03548v2](http://arxiv.org/pdf/2510.03548v2)**

> **作者:** Danial Samadi Vahdati; Tai Duc Nguyen; Ekta Prashnani; Koki Nagano; David Luebke; Orazio Gallo; Matthew Stamm
>
> **摘要:** AI-based talking-head videoconferencing systems reduce bandwidth by sending a compact pose-expression latent and re-synthesizing RGB at the receiver, but this latent can be puppeteered, letting an attacker hijack a victim's likeness in real time. Because every frame is synthetic, deepfake and synthetic video detectors fail outright. To address this security problem, we exploit a key observation: the pose-expression latent inherently contains biometric information of the driving identity. Therefore, we introduce the first biometric leakage defense without ever looking at the reconstructed RGB video: a pose-conditioned, large-margin contrastive encoder that isolates persistent identity cues inside the transmitted latent while cancelling transient pose and expression. A simple cosine test on this disentangled embedding flags illicit identity swaps as the video is rendered. Our experiments on multiple talking-head generation models show that our method consistently outperforms existing puppeteering defenses, operates in real-time, and shows strong generalization to out-of-distribution scenarios.
>
---
#### [replaced 070] Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16394v2](http://arxiv.org/pdf/2505.16394v2)**

> **作者:** Zhenjie Yang; Xiaosong Jia; Qifeng Li; Xue Yang; Maoqing Yao; Junchi Yan
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Reinforcement Learning (RL) can mitigate the causal confusion and distribution shift inherent to imitation learning (IL). However, applying RL to end-to-end autonomous driving (E2E-AD) remains an open problem for its training difficulty, and IL is still the mainstream paradigm in both academia and industry. Recently Model-based Reinforcement Learning (MBRL) have demonstrated promising results in neural planning; however, these methods typically require privileged information as input rather than raw sensor data. We fill this gap by designing Raw2Drive, a dual-stream MBRL approach. Initially, we efficiently train an auxiliary privileged world model paired with a neural planner that uses privileged information as input. Subsequently, we introduce a raw sensor world model trained via our proposed Guidance Mechanism, which ensures consistency between the raw sensor world model and the privileged world model during rollouts. Finally, the raw sensor world model combines the prior knowledge embedded in the heads of the privileged world model to effectively guide the training of the raw sensor policy. Raw2Drive is so far the only RL based end-to-end method on CARLA Leaderboard 2.0, and Bench2Drive and it achieves state-of-the-art performance.
>
---
#### [replaced 071] RWKV-UNet: Improving UNet with Long-Range Cooperation for Effective Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08458v3](http://arxiv.org/pdf/2501.08458v3)**

> **作者:** Juntao Jiang; Jiangning Zhang; Weixuan Liu; Muxuan Gao; Xiaobin Hu; Zhucun Xue; Yong Liu; Shuicheng Yan
>
> **摘要:** In recent years, significant advancements have been made in deep learning for medical image segmentation, particularly with convolutional neural networks (CNNs) and transformer models. However, CNNs face limitations in capturing long-range dependencies, while transformers suffer from high computational complexity. To address this, we propose RWKV-UNet, a novel model that integrates the RWKV (Receptance Weighted Key Value) structure into the U-Net architecture. This integration enhances the model's ability to capture long-range dependencies and to improve contextual understanding, which is crucial for accurate medical image segmentation. We build a strong encoder with developed Global-Local Spatial Perception (GLSP) blocks combining CNNs and RWKVs. We also propose a Cross-Channel Mix (CCM) module to improve skip connections with multi-scale feature fusion, achieving global channel information integration. Experiments on 11 benchmark datasets show that the RWKV-UNet achieves state-of-the-art performance on various types of medical image segmentation tasks. Additionally, smaller variants, RWKV-UNet-S and RWKV-UNet-T, balance accuracy and computational efficiency, making them suitable for broader clinical applications.
>
---
#### [replaced 072] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15510v2](http://arxiv.org/pdf/2505.15510v2)**

> **作者:** Zihui Cheng; Qiguang Chen; Xiao Xu; Jiaqi Wang; Weiyun Wang; Hao Fei; Yidong Wang; Alex Jinpeng Wang; Zhi Chen; Wanxiang Che; Libo Qin
>
> **备注:** Accepted at NeurIPS 2025;
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved significant success in multimodal tasks, with multimodal chain-of-thought (MCoT) further enhancing performance and interpretability. Recent MCoT methods fall into two categories: (i) Textual-MCoT (T-MCoT), which takes multimodal input and produces textual output; and (ii) Interleaved-MCoT (I-MCoT), which generates interleaved image-text outputs. Despite advances in both approaches, the mechanisms driving these improvements are not fully understood. To fill this gap, we first reveal that MCoT boosts LVLMs by incorporating visual thoughts, which convey image information to the reasoning process regardless of the MCoT format, depending only on clarity and conciseness of expression. Furthermore, to explore visual thoughts systematically, we define four distinct forms of visual thought expressions and analyze them comprehensively. Our findings demonstrate that these forms differ in clarity and conciseness, yielding varying levels of MCoT improvement. Additionally, we explore the internal nature of visual thoughts, finding that visual thoughts serve as intermediaries between the input image and reasoning to deeper transformer layers, enabling more advanced visual information transmission. We hope that the visual thoughts can inspire further breakthroughs for future MCoT research.
>
---
#### [replaced 073] Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23566v4](http://arxiv.org/pdf/2505.23566v4)**

> **作者:** Yu Li; Jin Jiang; Jianhua Zhu; Shuai Peng; Baole Wei; Yuxuan Zhou; Liangcai Gao
>
> **备注:** Accepted by NeurIPS 2025 as a spotlight
>
> **摘要:** Handwritten Mathematical Expression Recognition (HMER) remains a persistent challenge in Optical Character Recognition (OCR) due to the inherent freedom of symbol layouts and variability in handwriting styles. Prior methods have faced performance bottlenecks by proposing isolated architectural modifications, making them difficult to integrate coherently into a unified framework. Meanwhile, recent advances in pretrained vision-language models (VLMs) have demonstrated strong cross-task generalization, offering a promising foundation for developing unified solutions. In this paper, we introduce Uni-MuMER, which fully fine-tunes a VLM for the HMER task without modifying its architecture, effectively injecting domain-specific knowledge into a generalist framework. Our method integrates three data-driven tasks: Tree-Aware Chain-of-Thought (Tree-CoT) for structured spatial reasoning, Error-Driven Learning (EDL) for reducing confusion among visually similar characters, and Symbol Counting (SC) for improving recognition consistency in long expressions. Experiments on the CROHME and HME100K datasets show that Uni-MuMER achieves super state-of-the-art performance, outperforming the best lightweight specialized model SSAN by 16.31\% and the top-performing VLM Gemini2.5-flash by 24.42\% under zero-shot setting. Our datasets, models, and code are open-sourced at: {https://github.com/BFlameSwift/Uni-MuMER
>
---
#### [replaced 074] Visual Autoregressive Models Beat Diffusion Models on Inference Time Scaling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.16751v2](http://arxiv.org/pdf/2510.16751v2)**

> **作者:** Erik Riise; Mehmet Onurcan Kaya; Dim P. Papadopoulos
>
> **摘要:** While inference-time scaling through search has revolutionized Large Language Models, translating these gains to image generation has proven difficult. Recent attempts to apply search strategies to continuous diffusion models show limited benefits, with simple random sampling often performing best. We demonstrate that the discrete, sequential nature of visual autoregressive models enables effective search for image generation. We show that beam search substantially improves text-to-image generation, enabling a 2B parameter autoregressive model to outperform a 12B parameter diffusion model across benchmarks. Systematic ablations show that this advantage comes from the discrete token space, which allows early pruning and computational reuse, and our verifier analysis highlights trade-offs between speed and reasoning capability. These findings suggest that model architecture, not just scale, is critical for inference-time optimization in visual generation.
>
---
#### [replaced 075] Principled Multimodal Representation Learning
- **分类: cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.17343v2](http://arxiv.org/pdf/2507.17343v2)**

> **作者:** Xiaohao Liu; Xiaobo Xia; See-Kiong Ng; Tat-Seng Chua
>
> **备注:** Corrected typos and updated experimental results. 32 pages, 9 figures, 10 tables
>
> **摘要:** Multimodal representation learning seeks to create a unified representation space by integrating diverse data modalities to improve multimodal understanding. Traditional methods often depend on pairwise contrastive learning, which relies on a predefined anchor modality, restricting alignment across all modalities. Recent advances have investigated the simultaneous alignment of multiple modalities, yet several challenges remain, such as limitations imposed by fixed anchor points and instability arising from optimizing the product of singular values. To address the challenges, in this paper, we propose Principled Multimodal Representation Learning (PMRL), a novel framework that achieves simultaneous alignment of multiple modalities without anchor dependency in a more stable manner. Specifically, grounded in the theoretical insight that full alignment corresponds to a rank-1 Gram matrix, PMRL optimizes the dominant singular value of the representation matrix to align modalities along a shared leading direction. We propose a softmax-based loss function that treats singular values as logits to prioritize the largest singular value. Besides, instance-wise contrastive regularization on the leading eigenvectors maintains inter-instance separability and prevents representation collapse. Extensive experiments across diverse tasks demonstrate PMRL's superiority compared to baseline methods. The source code will be publicly available.
>
---
#### [replaced 076] Enhancing Feature Fusion of U-like Networks with Dynamic Skip Connections
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.14610v4](http://arxiv.org/pdf/2509.14610v4)**

> **作者:** Yue Cao; Quansong He; Kaishen Wang; Jianlong Xiong; Zhang Yi; Tao He
>
> **摘要:** U-like networks have become fundamental frameworks in medical image segmentation through skip connections that bridge high-level semantics and low-level spatial details. Despite their success, conventional skip connections exhibit two key limitations: inter-feature constraints and intra-feature constraints. The inter-feature constraint refers to the static nature of feature fusion in traditional skip connections, where information is transmitted along fixed pathways regardless of feature content. The intra-feature constraint arises from the insufficient modeling of multi-scale feature interactions, thereby hindering the effective aggregation of global contextual information. To overcome these limitations, we propose a novel Dynamic Skip Connection (DSC) block that fundamentally enhances cross-layer connectivity through adaptive mechanisms. The DSC block integrates two complementary components. (1) Test-Time Training (TTT) module. This module addresses the inter-feature constraint by enabling dynamic adaptation of hidden representations during inference, facilitating content-aware feature refinement. (2) Dynamic Multi-Scale Kernel (DMSK) module. To mitigate the intra-feature constraint, this module adaptively selects kernel sizes based on global contextual cues, enhancing the network capacity for multi-scale feature integration. The DSC block is architecture-agnostic and can be seamlessly incorporated into existing U-like network structures. Extensive experiments demonstrate the plug-and-play effectiveness of the proposed DSC block across CNN-based, Transformer-based, hybrid CNN-Transformer, and Mamba-based U-like networks.
>
---
#### [replaced 077] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02093v2](http://arxiv.org/pdf/2506.02093v2)**

> **作者:** Tianyu Lin; Xinran Li; Chuntung Zhuang; Qi Chen; Yuanhao Cai; Kai Ding; Alan L. Yuille; Zongwei Zhou
>
> **备注:** NeurIPS 2025
>
> **摘要:** Widely adopted evaluation metrics for sparse-view CT reconstruction--such as Structural Similarity Index Measure and Peak Signal-to-Noise Ratio--prioritize pixel-wise fidelity but often fail to capture the completeness of critical anatomical structures, particularly small or thin regions that are easily missed. To address this limitation, we propose a suite of novel anatomy-aware evaluation metrics designed to assess structural completeness across anatomical structures, including large organs, small organs, intestines, and vessels. Building on these metrics, we introduce CARE, a Completeness-Aware Reconstruction Enhancement framework that incorporates structural penalties during training to encourage anatomical preservation of significant structures. CARE is model-agnostic and can be seamlessly integrated into analytical, implicit, and generative methods. When applied to these methods, CARE substantially improves structural completeness in CT reconstructions, achieving up to +32% improvement for large organs, +22% for small organs, +40% for intestines, and +36% for vessels.
>
---
#### [replaced 078] AlignCAT: Visual-Linguistic Alignment of Category and Attribute for Weakly Supervised Visual Grounding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03201v3](http://arxiv.org/pdf/2508.03201v3)**

> **作者:** Yidan Wang; Chenyi Zhuang; Wutao Liu; Pan Gao; Nicu Sebe
>
> **摘要:** Weakly supervised visual grounding (VG) aims to locate objects in images based on text descriptions. Despite significant progress, existing methods lack strong cross-modal reasoning to distinguish subtle semantic differences in text expressions due to category-based and attribute-based ambiguity. To address these challenges, we introduce AlignCAT, a novel query-based semantic matching framework for weakly supervised VG. To enhance visual-linguistic alignment, we propose a coarse-grained alignment module that utilizes category information and global context, effectively mitigating interference from category-inconsistent objects. Subsequently, a fine-grained alignment module leverages descriptive information and captures word-level text features to achieve attribute consistency. By exploiting linguistic cues to their fullest extent, our proposed AlignCAT progressively filters out misaligned visual queries and enhances contrastive learning efficiency. Extensive experiments on three VG benchmarks, namely RefCOCO, RefCOCO+, and RefCOCOg, verify the superiority of AlignCAT against existing weakly supervised methods on two VG tasks. Our code is available at: https://github.com/I2-Multimedia-Lab/AlignCAT.
>
---
#### [replaced 079] Now you see me! Attribution Distributions Reveal What is Truly Important for a Prediction
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.07346v2](http://arxiv.org/pdf/2503.07346v2)**

> **作者:** Nils Philipp Walter; Jilles Vreeken; Jonas Fischer
>
> **摘要:** Neural networks are regularly employed in high-stakes decision-making, where understanding and transparency is key. Attribution methods have been developed to gain understanding into which input features neural networks use for a specific prediction. Although widely used in computer vision, these methods often result in unspecific saliency maps that fail to identify the relevant information that led to a decision, supported by different benchmarks results. Here, we revisit the common attribution pipeline and identify one cause for the lack of specificity in attributions as the computation of attribution of isolated logits. Instead, we suggest to combine attributions of multiple class logits in analogy to how the softmax combines the information across logits. By computing probability distributions of attributions over classes for each spatial location in the image, we unleash the true capabilities of existing attribution methods, revealing better object- and instance-specificity and uncovering discriminative as well as shared features between classes. On common benchmarks, including the grid-pointing game and randomization-based sanity checks, we show that this reconsideration of how and where we compute attributions across the network improves established attribution methods while staying agnostic to model architectures. We make the code publicly available: https://github.com/nilspwalter/var.
>
---
#### [replaced 080] Contrastive Conditional-Unconditional Alignment for Long-tailed Diffusion Model
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.09052v2](http://arxiv.org/pdf/2507.09052v2)**

> **作者:** Fang Chen; Alex Villa; Gongbo Liang; Xiaoyi Lu; Meng Tang
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Training data for class-conditional image synthesis often exhibit a long-tailed distribution with limited images for tail classes. Such an imbalance causes mode collapse and reduces the diversity of synthesized images for tail classes. For class-conditional diffusion models trained on imbalanced data, we aim to improve the diversity and fidelity of tail class images without compromising the quality of head class images. We achieve this by introducing two simple but highly effective loss functions. Firstly, we employ an Unsupervised Contrastive Loss (UCL) utilizing negative samples to increase the distance/dissimilarity among synthetic images. Such regularization is coupled with a standard trick of batch resampling to further diversify tail-class images. Our second loss is an Alignment Loss (AL) that aligns class-conditional generation with unconditional generation at large timesteps. This second loss makes the denoising process insensitive to class conditions for the initial steps, which enriches tail classes through knowledge sharing from head classes. We successfully leverage contrastive learning and conditional-unconditional alignment for class-imbalanced diffusion models. Our framework is easy to implement as demonstrated on both U-Net based architecture and Diffusion Transformer. Our method outperforms vanilla denoising diffusion probabilistic models, score-based diffusion model, and alternative methods for class-imbalanced image generation across various datasets, in particular ImageNet-LT with 256x256 resolution.
>
---
#### [replaced 081] Revisiting Transformation Invariant Geometric Deep Learning: An Initial Representation Perspective
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2112.12345v2](http://arxiv.org/pdf/2112.12345v2)**

> **作者:** Ziwei Zhang; Xin Wang; Zeyang Zhang; Peng Cui; Wenwu Zhu
>
> **备注:** 13 pages; accepted by IEEE TPAMI
>
> **摘要:** Deep neural networks have achieved great success in the last decade. When designing neural networks to handle the ubiquitous geometric data such as point clouds and graphs, it is critical that the model can maintain invariance towards various transformations such as translation, rotation, and scaling. Most existing graph neural network (GNN) approaches can only maintain permutation-invariance, failing to guarantee invariance with respect to other transformations. Besides GNNs, other works design sophisticated transformation-invariant layers, which are computationally expensive and difficult to be extended. In this paper, we revisit why general neural networks cannot maintain transformation invariance. Our findings show that transformation-invariant and distance-preserving initial point representations are sufficient to achieve transformation invariance rather than needing sophisticated neural layer designs. Motivated by these findings, we propose Transformation Invariant Neural Networks (TinvNN), a straightforward and general plug-in for geometric data. Specifically, we realize transformation invariant and distance-preserving initial point representations by modifying multi-dimensional scaling and feed the representations into existing neural networks. We prove that TinvNN can strictly guarantee transformation invariance, being general and flexible enough to be combined with the existing neural networks. Extensive experimental results on point cloud analysis and combinatorial optimization demonstrate the effectiveness and general applicability of our method. We also extend our method into equivariance cases. Based on the results, we advocate that TinvNN should be considered as an essential baseline for further studies of transformation-invariant geometric deep learning.
>
---
#### [replaced 082] DynamicVL: Benchmarking Multimodal Large Language Models for Dynamic City Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21076v2](http://arxiv.org/pdf/2505.21076v2)**

> **作者:** Weihao Xuan; Junjue Wang; Heli Qi; Zihang Chen; Zhuo Zheng; Yanfei Zhong; Junshi Xia; Naoto Yokoya
>
> **备注:** NeurIPS 2025
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in visual understanding, but their application to long-term Earth observation analysis remains limited, primarily focusing on single-temporal or bi-temporal imagery. To address this gap, we introduce DVL-Suite, a comprehensive framework for analyzing long-term urban dynamics through remote sensing imagery. Our suite comprises 14,871 high-resolution (1.0m) multi-temporal images spanning 42 major cities in the U.S. from 2005 to 2023, organized into two components: DVL-Bench and DVL-Instruct. The DVL-Bench includes six urban understanding tasks, from fundamental change detection (pixel-level) to quantitative analyses (regional-level) and comprehensive urban narratives (scene-level), capturing diverse urban dynamics including expansion/transformation patterns, disaster assessment, and environmental challenges. We evaluate 18 state-of-the-art MLLMs and reveal their limitations in long-term temporal understanding and quantitative analysis. These challenges motivate the creation of DVL-Instruct, a specialized instruction-tuning dataset designed to enhance models' capabilities in multi-temporal Earth observation. Building upon this dataset, we develop DVLChat, a baseline model capable of both image-level question-answering and pixel-level segmentation, facilitating a comprehensive understanding of city dynamics through language interactions.
>
---
#### [replaced 083] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11147v2](http://arxiv.org/pdf/2506.11147v2)**

> **作者:** Xiaotang Gai; Jiaxiang Liu; Yichen Li; Zijie Meng; Jian Wu; Zuozhu Liu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Medical Visual Question Answering (Med-VQA) holds significant potential for clinical decision support, yet existing efforts primarily focus on 2D imaging with limited task diversity. This paper presents 3D-RAD, a large-scale dataset designed to advance 3D Med-VQA using radiology CT scans. The 3D-RAD dataset encompasses six diverse VQA tasks: anomaly detection, image observation, medical computation, existence detection, static temporal diagnosis, and longitudinal temporal diagnosis. It supports both open- and closed-ended questions while introducing complex reasoning challenges, including computational tasks and multi-stage temporal analysis, to enable comprehensive benchmarking. Extensive evaluations demonstrate that existing vision-language models (VLMs), especially medical VLMs exhibit limited generalization, particularly in multi-temporal tasks, underscoring the challenges of real-world 3D diagnostic reasoning. To drive future advancements, we release a high-quality training set 3D-RAD-T of 136,195 expert-aligned samples, showing that fine-tuning on this dataset could significantly enhance model performance. Our dataset and code, aiming to catalyze multimodal medical AI research and establish a robust foundation for 3D medical visual understanding, are publicly available at https://github.com/Tang-xiaoxiao/3D-RAD.
>
---
#### [replaced 084] Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02096v3](http://arxiv.org/pdf/2502.02096v3)**

> **作者:** Yixiao Chen; Shikun Sun; Jianshu Li; Ruoyu Li; Zhe Li; Junliang Xing
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Adversarial attacks are widely used to evaluate model robustness, and in black-box scenarios, the transferability of these attacks becomes crucial. Existing generator-based attacks have excellent generalization and transferability due to their instance-agnostic nature. However, when training generators for multi-target tasks, the success rate of transfer attacks is relatively low due to the limitations of the model's capacity. To address these challenges, we propose a novel Dual-Flow framework for multi-target instance-agnostic adversarial attacks, utilizing Cascading Distribution Shift Training to develop an adversarial velocity function. Extensive experiments demonstrate that Dual-Flow significantly improves transferability over previous multi-target generative attacks. For example, it increases the success rate from Inception-v3 to ResNet-152 by 34.58\%. Furthermore, our attack method shows substantially stronger robustness against defense mechanisms, such as adversarially trained models. The code of Dual-Flow is available at: $\href{https://github.com/Chyxx/Dual-Flow}{https://github.com/Chyxx/Dual-Flow}$.
>
---
#### [replaced 085] Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.23881v2](http://arxiv.org/pdf/2506.23881v2)**

> **作者:** Reihaneh Zohrabi; Hosein Hasani; Mahdieh Soleymani Baghshah; Anna Rohrbach; Marcus Rohrbach; Mohammad Hossein Rohban
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Out-of-distribution (OOD) detection is crucial for ensuring the reliability and safety of machine learning models in real-world applications, where they frequently face data distributions unseen during training. Despite progress, existing methods are often vulnerable to spurious correlations that mislead models and compromise robustness. To address this, we propose SPROD, a novel prototype-based OOD detection approach that explicitly addresses the challenge posed by unknown spurious correlations. Our post-hoc method refines class prototypes to mitigate bias from spurious features without additional data or hyperparameter tuning, and is broadly applicable across diverse backbones and OOD detection settings. We conduct a comprehensive spurious correlation OOD detection benchmarking, comparing our method against existing approaches and demonstrating its superior performance across challenging OOD datasets, such as CelebA, Waterbirds, UrbanCars, Spurious Imagenet, and the newly introduced Animals MetaCoCo. On average, SPROD improves AUROC by 4.8% and FPR@95 by 9.4% over the second best.
>
---
#### [replaced 086] Efficient Semi-Supervised Adversarial Training via Latent Clustering-Based Data Reduction
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.10466v2](http://arxiv.org/pdf/2501.10466v2)**

> **作者:** Somrita Ghosh; Yuelin Xu; Xiao Zhang
>
> **备注:** Shorter version of this work accepted by NextGenAISafety Workshop at ICML 2024
>
> **摘要:** Achieving high model robustness under adversarial settings is widely recognized as demanding considerable training samples. Recent works propose semi-supervised adversarial training (SSAT) methods with external unlabeled or synthetically generated data, which are the current state-of-the-art. However, SSAT requires substantial extra data to attain high robustness, resulting in prolonged training time and increased memory usage. In this paper, we propose unlabeled data reduction strategies to improve the efficiency of SSAT. Specifically, we design novel latent clustering-based techniques to select or generate a small critical subset of data samples near the model's decision boundary. While focusing on boundary-adjacent points, our methods maintain a balanced ratio between boundary and non-boundary data points to avoid overfitting. Comprehensive experiments on benchmark datasets demonstrate that our methods can significantly reduce SSAT's data requirement and computation costs while preserving its strong robustness advantages. In particular, our latent-space selection scheme based on k-means clustering and our guided DDPM fine-tuning approach with LCG-KM are the most effective, achieving nearly identical robust accuracies with 5x to 10x less unlabeled data and approximately 4x less total runtime.
>
---
#### [replaced 087] S$^2$Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04016v3](http://arxiv.org/pdf/2508.04016v3)**

> **作者:** Weilun Feng; Haotong Qin; Chuanguang Yang; Xiangqi Li; Han Yang; Yuqi Li; Zhulin An; Libo Huang; Michele Magno; Yongjun Xu
>
> **摘要:** Diffusion transformers have emerged as the mainstream paradigm for video generation models. However, the use of up to billions of parameters incurs significant computational costs. Quantization offers a promising solution by reducing memory usage and accelerating inference. Nonetheless, we observe that the joint modeling of spatial and temporal information in video diffusion models (V-DMs) leads to extremely long token sequences, which introduces high calibration variance and learning challenges. To address these issues, we propose S$^2$Q-VDiT, a post-training quantization framework for V-DMs that leverages Salient data and Sparse token distillation. During the calibration phase, we identify that quantization performance is highly sensitive to the choice of calibration data. To mitigate this, we introduce \textit{Hessian-aware Salient Data Selection}, which constructs high-quality calibration datasets by considering both diffusion and quantization characteristics unique to V-DMs. To tackle the learning challenges, we further analyze the sparse attention patterns inherent in V-DMs. Based on this observation, we propose \textit{Attention-guided Sparse Token Distillation}, which exploits token-wise attention distributions to emphasize tokens that are more influential to the model's output. Under W4A6 quantization, S$^2$Q-VDiT achieves lossless performance while delivering $3.9\times$ model compression and $1.3\times$ inference acceleration. Code will be available at https://github.com/wlfeng0509/s2q-vdit.
>
---
#### [replaced 088] MagicMotion: Controllable Video Generation with Dense-to-Sparse Trajectory Guidance
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.16421v3](http://arxiv.org/pdf/2503.16421v3)**

> **作者:** Quanhao Li; Zhen Xing; Rui Wang; Hui Zhang; Qi Dai; Zuxuan Wu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent advances in video generation have led to remarkable improvements in visual quality and temporal coherence. Upon this, trajectory-controllable video generation has emerged to enable precise object motion control through explicitly defined spatial paths. However, existing methods struggle with complex object movements and multi-object motion control, resulting in imprecise trajectory adherence, poor object consistency, and compromised visual quality. Furthermore, these methods only support trajectory control in a single format, limiting their applicability in diverse scenarios. Additionally, there is no publicly available dataset or benchmark specifically tailored for trajectory-controllable video generation, hindering robust training and systematic evaluation. To address these challenges, we introduce MagicMotion, a novel image-to-video generation framework that enables trajectory control through three levels of conditions from dense to sparse: masks, bounding boxes, and sparse boxes. Given an input image and trajectories, MagicMotion seamlessly animates objects along defined trajectories while maintaining object consistency and visual quality. Furthermore, we present MagicData, a large-scale trajectory-controlled video dataset, along with an automated pipeline for annotation and filtering. We also introduce MagicBench, a comprehensive benchmark that assesses both video quality and trajectory control accuracy across different numbers of objects. Extensive experiments demonstrate that MagicMotion outperforms previous methods across various metrics. Our project page are publicly available at https://quanhaol.github.io/magicmotion-site.
>
---
#### [replaced 089] Robust Multimodal Learning via Cross-Modal Proxy Tokens
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.17823v4](http://arxiv.org/pdf/2501.17823v4)**

> **作者:** Md Kaykobad Reza; Ameya Patil; Mashhour Solh; M. Salman Asif
>
> **备注:** 28 Pages, 13 Figures, 11 Tables. Accepted by Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Multimodal models often experience a significant performance drop when one or more modalities are missing during inference. To address this challenge, we propose a simple yet effective approach that enhances robustness to missing modalities while maintaining strong performance when all modalities are available. Our method introduces cross-modal proxy tokens (CMPTs), which approximate the class token of a missing modality by attending only to the tokens of the available modality without requiring explicit modality generation or auxiliary networks. To efficiently learn these approximations with minimal computational overhead, we employ low-rank adapters in frozen unimodal encoders and jointly optimize an alignment loss with a task-specific loss. Extensive experiments on five multimodal datasets show that our method outperforms state-of-the-art baselines across various missing rates while achieving competitive results in complete-modality settings. Overall, our method offers a flexible and efficient solution for robust multimodal learning. The code for this paper is available at: https://github.com/CSIPlab/Cross-Modal-Proxy-Tokens.
>
---
#### [replaced 090] Long-Tailed Classification by Keeping the Good and Removing the Bad Momentum Causal Effect
- **分类: cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2009.12991v5](http://arxiv.org/pdf/2009.12991v5)**

> **作者:** Kaihua Tang; Jianqiang Huang; Hanwang Zhang
>
> **备注:** This paper is accepted by NeurIPS 2020. The code is available on GitHub: https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch
>
> **摘要:** As the class size grows, maintaining a balanced dataset across many classes is challenging because the data are long-tailed in nature; it is even impossible when the sample-of-interest co-exists with each other in one collectable unit, e.g., multiple visual instances in one image. Therefore, long-tailed classification is the key to deep learning at scale. However, existing methods are mainly based on re-weighting/re-sampling heuristics that lack a fundamental theory. In this paper, we establish a causal inference framework, which not only unravels the whys of previous methods, but also derives a new principled solution. Specifically, our theory shows that the SGD momentum is essentially a confounder in long-tailed classification. On one hand, it has a harmful causal effect that misleads the tail prediction biased towards the head. On the other hand, its induced mediation also benefits the representation learning and head prediction. Our framework elegantly disentangles the paradoxical effects of the momentum, by pursuing the direct causal effect caused by an input sample. In particular, we use causal intervention in training, and counterfactual reasoning in inference, to remove the "bad" while keep the "good". We achieve new state-of-the-arts on three long-tailed visual recognition benchmarks: Long-tailed CIFAR-10/-100, ImageNet-LT for image classification and LVIS for instance segmentation.
>
---
#### [replaced 091] Bootstrapping Referring Multi-Object Tracking
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.05039v2](http://arxiv.org/pdf/2406.05039v2)**

> **作者:** Yani Zhang; Dongming Wu; Wencheng Han; Xingping Dong
>
> **摘要:** Referring understanding is a fundamental task that bridges natural language and visual content by localizing objects described in free-form expressions. However, existing works are constrained by limited language expressiveness, lacking the capacity to model object dynamics in spatial numbers and temporal states. To address these limitations, we introduce a new and general referring understanding task, termed referring multi-object tracking (RMOT). Its core idea is to employ a language expression as a semantic cue to guide the prediction of multi-object tracking, comprehensively accounting for variations in object quantity and temporal semantics. Along with RMOT, we introduce a RMOT benchmark named Refer-KITTI-V2, featuring scalable and diverse language expressions. To efficiently generate high-quality annotations covering object dynamics with minimal manual effort, we propose a semi-automatic labeling pipeline that formulates a total of 9,758 language prompts. In addition, we propose TempRMOT, an elegant end-to-end Transformer-based framework for RMOT. At its core is a query-driven Temporal Enhancement Module that represents each object as a Transformer query, enabling long-term spatial-temporal interactions with other objects and past frames to efficiently refine these queries. TempRMOT achieves state-of-the-art performance on both Refer-KITTI and Refer-KITTI-V2, demonstrating the effectiveness of our approach. The source code and dataset is available at https://github.com/zyn213/TempRMOT.
>
---
#### [replaced 092] A Training-Free Framework for Open-Vocabulary Image Segmentation and Recognition with EfficientNet and CLIP
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.19333v2](http://arxiv.org/pdf/2510.19333v2)**

> **作者:** Ying Dai; Wei Yu Chen
>
> **摘要:** This paper presents a novel training-free framework for open-vocabulary image segmentation and object recognition (OVSR), which leverages EfficientNetB0, a convolutional neural network, for unsupervised segmentation and CLIP, a vision-language model, for open-vocabulary object recognition. The proposed framework adopts a two stage pipeline: unsupervised image segmentation followed by segment-level recognition via vision-language alignment. In the first stage, pixel-wise features extracted from EfficientNetB0 are decomposed using singular value decomposition to obtain latent representations, which are then clustered using hierarchical clustering to segment semantically meaningful regions. The number of clusters is adaptively determined by the distribution of singular values. In the second stage, the segmented regions are localized and encoded into image embeddings using the Vision Transformer backbone of CLIP. Text embeddings are precomputed using CLIP's text encoder from category-specific prompts, including a generic something else prompt to support open set recognition. The image and text embeddings are concatenated and projected into a shared latent feature space via SVD to enhance cross-modal alignment. Recognition is performed by computing the softmax over the similarities between the projected image and text embeddings. The proposed method is evaluated on standard benchmarks, including COCO, ADE20K, and PASCAL VOC, achieving state-of-the-art performance in terms of Hungarian mIoU, precision, recall, and F1-score. These results demonstrate the effectiveness, flexibility, and generalizability of the proposed framework.
>
---
#### [replaced 093] ExpressNet-MoE: A Hybrid Deep Neural Network for Emotion Recognition
- **分类: cs.CV; cs.LG; I.2.10; I.5.2; H.4.2**

- **链接: [http://arxiv.org/pdf/2510.13493v2](http://arxiv.org/pdf/2510.13493v2)**

> **作者:** Deeptimaan Banerjee; Prateek Gothwal; Ashis Kumer Biswas
>
> **备注:** * Current version of the manuscript contains 17 pages including text, 13 figures, and 4 tables. The manuscript is currently under review at a journal
>
> **摘要:** In many domains, including online education, healthcare, security, and human-computer interaction, facial emotion recognition (FER) is essential. Real-world FER is still difficult despite its significance because of some factors such as variable head positions, occlusions, illumination shifts, and demographic diversity. Engagement detection, which is essential for applications like virtual learning and customer services, is frequently challenging due to FER limitations by many current models. In this article, we propose ExpressNet-MoE, a novel hybrid deep learning model that blends both Convolution Neural Networks (CNNs) and Mixture of Experts (MoE) framework, to overcome the difficulties. Our model dynamically chooses the most pertinent expert networks, thus it aids in the generalization and providing flexibility to model across a wide variety of datasets. Our model improves on the accuracy of emotion recognition by utilizing multi-scale feature extraction to collect both global and local facial features. ExpressNet-MoE includes numerous CNN-based feature extractors, a MoE module for adaptive feature selection, and finally a residual network backbone for deep feature learning. To demonstrate efficacy of our proposed model we evaluated on several datasets, and compared with current state-of-the-art methods. Our model achieves accuracies of 74.77% on AffectNet (v7), 72.55% on AffectNet (v8), 84.29% on RAF-DB, and 64.66% on FER-2013. The results show how adaptive our model is and how it may be used to develop end-to-end emotion recognition systems in practical settings. Reproducible codes and results are made publicly accessible at https://github.com/DeeptimaanB/ExpressNet-MoE.
>
---
#### [replaced 094] Blockchain and Biometrics: Survey, GDPR Analysis, and Future Directions
- **分类: cs.CV; cs.CR; cs.DC; cs.LG**

- **链接: [http://arxiv.org/pdf/2302.10883v4](http://arxiv.org/pdf/2302.10883v4)**

> **作者:** Mahdi Ghafourian; Bilgesu Sumer; Ruben Vera-Rodriguez; Julian Fierrez; Ruben Tolosana; Aythami Moralez; Els Kindt
>
> **摘要:** Biometric recognition as an efficient and hard-to-forge way of identification and verification has become an indispensable part of the current digital world. The fast evolution of this technology has been a strong incentive for integration into many applications. Meanwhile, blockchain, the decentralized ledger technology, has been widely received by both research and industry in the past few years, and it is being increasingly deployed today in many different applications, such as money transfer, IoT, healthcare, or logistics. Recently, researchers have started to speculate on the pros and cons and what the best applications would be when these two technologies cross paths. This paper provides a survey of the research literature on the combination of blockchain and biometrics and includes a first legal analysis of this integration based on GDPR to shed light on challenges and potentials. Although the integration of blockchain technology into the biometric sector is still in its infancy, with a growing body of literature discussing specific applications and advanced technological setups, this paper aims to provide a holistic understanding of blockchain applicability in biometrics. Based on published studies, this article discusses, among others, practical examples combining blockchain and biometrics for novel applications in PKI systems, distributed trusted services, and identity management. Challenges and limitations when combining blockchain and biometrics that motivate future work will also be discussed; e.g., blockchain networks at their current stage may not be efficient or economical for some real-time biometric applications. Finally, we also discuss key legal aspects of the EU General Data Protection Regulation (GDPR) related to this combination of technologies (blockchain and biometrics); for example, accountability, immutability, anonymity, and data protection elements.
>
---
#### [replaced 095] Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval
- **分类: cs.IR; cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.04273v3](http://arxiv.org/pdf/2508.04273v3)**

> **作者:** Junan Lin; Daizong Liu; Xianke Chen; Xiaoye Qu; Xun Yang; Jixiang Zhu; Sanyuan Zhang; Jianfeng Dong
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** Video Moment Retrieval (VMR) aims to retrieve a specific moment semantically related to the given query. To tackle this task, most existing VMR methods solely focus on the visual and textual modalities while neglecting the complementary but important audio modality. Although a few recent works try to tackle the joint audio-vision-text reasoning, they treat all modalities equally and simply embed them without fine-grained interaction for moment retrieval. These designs are counter-practical as: Not all audios are helpful for video moment retrieval, and the audio of some videos may be complete noise or background sound that is meaningless to the moment determination. To this end, we propose a novel Importance-aware Multi-Granularity fusion model (IMG), which learns to dynamically and selectively aggregate the audio-vision-text contexts for VMR. Specifically, after integrating the textual guidance with vision and audio separately, we first design a pseudo-label-supervised audio importance predictor that predicts the importance score of the audio, and accordingly assigns weights to mitigate the interference caused by noisy audio. Then, we design a multi-granularity audio fusion module that adaptively fuses audio and visual modalities at local-, event-, and global-level, fully capturing their complementary contexts. We further propose a cross-modal knowledge distillation strategy to address the challenge of missing audio modality during inference. To evaluate our method, we further construct a new VMR dataset, i.e., Charades-AudioMatter, where audio-related samples are manually selected and re-organized from the original Charades-STA to validate the model's capability in utilizing audio modality. Extensive experiments validate the effectiveness of our method, achieving state-of-the-art with audio-video fusion in VMR methods. Our code is available at https://github.com/HuiGuanLab/IMG.
>
---
#### [replaced 096] Pindrop it! Audio and Visual Deepfake Countermeasures for Robust Detection and Fine Grained-Localization
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.08141v2](http://arxiv.org/pdf/2508.08141v2)**

> **作者:** Nicholas Klein; Hemlata Tak; James Fullwood; Krishna Regmi; Leonidas Spinoulas; Ganesh Sivaraman; Tianxiang Chen; Elie Khoury
>
> **摘要:** The field of visual and audio generation is burgeoning with new state-of-the-art methods. This rapid proliferation of new techniques underscores the need for robust solutions for detecting synthetic content in videos. In particular, when fine-grained alterations via localized manipulations are performed in visual, audio, or both domains, these subtle modifications add challenges to the detection algorithms. This paper presents solutions for the problems of deepfake video classification and localization. The methods were submitted to the ACM 1M Deepfakes Detection Challenge, achieving the best performance in the temporal localization task and a top four ranking in the classification task for the TestA split of the evaluation dataset.
>
---
#### [replaced 097] MEXA: Towards General Multimodal Reasoning with Dynamic Multi-Expert Aggregation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.17113v2](http://arxiv.org/pdf/2506.17113v2)**

> **作者:** Shoubin Yu; Yue Zhang; Ziyang Wang; Jaehong Yoon; Mohit Bansal
>
> **备注:** EMNLP 2025 Findings; The first two authors contributed equally; Github link: https://github.com/Yui010206/MEXA
>
> **摘要:** Combining pre-trained expert models offers substantial potential for scalable multimodal reasoning, but building a unified framework remains challenging due to the increasing diversity of input modalities and task complexity. For instance, medical diagnosis requires precise reasoning over structured clinical tables, while financial forecasting depends on interpreting plot-based data to make informed predictions. To tackle this challenge, we introduce MEXA, a training-free framework that performs modality- and task-aware aggregation of multiple expert models to enable effective multimodal reasoning across diverse and distinct domains. MEXA dynamically selects expert models based on the input modality and the task-specific reasoning demands (i.e., skills). Each expert model, specialized in a modality task pair, generates interpretable textual reasoning outputs. MEXA then aggregates and reasons over these outputs using a Large Reasoning Model (LRM) to produce the final answer. This modular design allows flexible and transparent multimodal reasoning across diverse domains without additional training overhead. We extensively evaluate our approach on diverse multimodal benchmarks, including Video Reasoning, Audio Reasoning, 3D Understanding, and Medical QA. MEXA consistently delivers performance improvements over strong multimodal baselines, highlighting the effectiveness and broad applicability of our expert-driven selection and aggregation in diverse multimodal reasoning tasks.
>
---
#### [replaced 098] DEEMO: De-identity Multimodal Emotion Recognition and Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.19549v2](http://arxiv.org/pdf/2504.19549v2)**

> **作者:** Deng Li; Bohao Xing; Xin Liu; Baiqiang Xia; Bihan Wen; Heikki Kälviäinen
>
> **备注:** Accepted by ACMMM 2025
>
> **摘要:** Emotion understanding is a critical yet challenging task. Most existing approaches rely heavily on identity-sensitive information, such as facial expressions and speech, which raises concerns about personal privacy. To address this, we introduce the De-identity Multimodal Emotion Recognition and Reasoning (DEEMO), a novel task designed to enable emotion understanding using de-identified video and audio inputs. The DEEMO dataset consists of two subsets: DEEMO-NFBL, which includes rich annotations of Non-Facial Body Language (NFBL), and DEEMO-MER, an instruction dataset for Multimodal Emotion Recognition and Reasoning using identity-free cues. This design supports emotion understanding without compromising identity privacy. In addition, we propose DEEMO-LLaMA, a Multimodal Large Language Model (MLLM) that integrates de-identified audio, video, and textual information to enhance both emotion recognition and reasoning. Extensive experiments show that DEEMO-LLaMA achieves state-of-the-art performance on both tasks, outperforming existing MLLMs by a significant margin, achieving 74.49% accuracy and 74.45% F1-score in de-identity emotion recognition, and 6.20 clue overlap and 7.66 label overlap in de-identity emotion reasoning. Our work contributes to ethical AI by advancing privacy-preserving emotion understanding and promoting responsible affective computing.
>
---
#### [replaced 099] Psi-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01320v3](http://arxiv.org/pdf/2506.01320v3)**

> **作者:** Taehoon Yoon; Yunhong Min; Kyeongmin Yeo; Minhyuk Sung
>
> **备注:** NeurIPS 2025, Spotlight Presentation
>
> **摘要:** We introduce $\Psi$-Sampler, an SMC-based framework incorporating pCNL-based initial particle sampling for effective inference-time reward alignment with a score-based generative model. Inference-time reward alignment with score-based generative models has recently gained significant traction, following a broader paradigm shift from pre-training to post-training optimization. At the core of this trend is the application of Sequential Monte Carlo (SMC) to the denoising process. However, existing methods typically initialize particles from the Gaussian prior, which inadequately captures reward-relevant regions and results in reduced sampling efficiency. We demonstrate that initializing from the reward-aware posterior significantly improves alignment performance. To enable posterior sampling in high-dimensional latent spaces, we introduce the preconditioned Crank-Nicolson Langevin (pCNL) algorithm, which combines dimension-robust proposals with gradient-informed dynamics. This approach enables efficient and scalable posterior sampling and consistently improves performance across various reward alignment tasks, including layout-to-image generation, quantity-aware generation, and aesthetic-preference generation, as demonstrated in our experiments. Project Webpage: https://psi-sampler.github.io/
>
---
#### [replaced 100] TransFace++: Rethinking the Face Recognition Paradigm with a Focus on Accuracy, Efficiency, and Security
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2308.10133v2](http://arxiv.org/pdf/2308.10133v2)**

> **作者:** Jun Dan; Yang Liu; Baigui Sun; Jiankang Deng; Shan Luo
>
> **备注:** This is an extended version of our previous ICCV paper "TransFace", with significant new experiments, ablation studies, and improvements published in IEEE TPAMI as "TransFace++"
>
> **摘要:** Face Recognition (FR) technology has made significant strides with the emergence of deep learning. Typically, most existing FR models are built upon Convolutional Neural Networks (CNN) and take RGB face images as the model's input. In this work, we take a closer look at existing FR paradigms from high-efficiency, security, and precision perspectives, and identify the following three problems: (i) CNN frameworks are vulnerable in capturing global facial features and modeling the correlations between local facial features. (ii) Selecting RGB face images as the model's input greatly degrades the model's inference efficiency, increasing the extra computation costs. (iii) In the real-world FR system that operates on RGB face images, the integrity of user privacy may be compromised if hackers successfully penetrate and gain access to the input of this model. To solve these three issues, we propose two novel FR frameworks, i.e., TransFace and TransFace++, which successfully explore the feasibility of applying ViTs and image bytes to FR tasks, respectively. Experiments on popular face benchmarks demonstrate the superiority of our TransFace and TransFace++. Code is available at https://github.com/DanJun6737/TransFace_pp.
>
---
#### [replaced 101] GS-ProCams: Gaussian Splatting-based Projector-Camera Systems
- **分类: cs.CV; cs.GR; cs.MM**

- **链接: [http://arxiv.org/pdf/2412.11762v2](http://arxiv.org/pdf/2412.11762v2)**

> **作者:** Qingyue Deng; Jijiang Li; Haibin Ling; Bingyao Huang
>
> **摘要:** We present GS-ProCams, the first Gaussian Splatting-based framework for projector-camera systems (ProCams). GS-ProCams is not only view-agnostic but also significantly enhances the efficiency of projection mapping (PM) that requires establishing geometric and radiometric mappings between the projector and the camera. Previous CNN-based ProCams are constrained to a specific viewpoint, limiting their applicability to novel perspectives. In contrast, NeRF-based ProCams support view-agnostic projection mapping, however, they require an additional co-located light source and demand significant computational and memory resources. To address this issue, we propose GS-ProCams that employs 2D Gaussian for scene representations, and enables efficient view-agnostic ProCams applications. In particular, we explicitly model the complex geometric and photometric mappings of ProCams using projector responses, the projection surface's geometry and materials represented by Gaussians, and the global illumination component. Then, we employ differentiable physically-based rendering to jointly estimate them from captured multi-view projections. Compared to state-of-the-art NeRF-based methods, our GS-ProCams eliminates the need for additional devices, achieving superior ProCams simulation quality. It also uses only 1/10 of the GPU memory for training and is 900 times faster in inference speed. Please refer to our project page for the code and dataset: https://realqingyue.github.io/GS-ProCams/.
>
---
#### [replaced 102] TokenCLIP: Token-wise Prompt Learning for Zero-shot Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.21171v2](http://arxiv.org/pdf/2510.21171v2)**

> **作者:** Qihang Zhou; Binbin Gao; Guansong Pang; Xin Wang; Jiming Chen; Shibo He
>
> **摘要:** Adapting CLIP for anomaly detection on unseen objects has shown strong potential in a zero-shot manner. However, existing methods typically rely on a single textual space to align with visual semantics across diverse objects and domains. The indiscriminate alignment hinders the model from accurately capturing varied anomaly semantics. We propose TokenCLIP, a token-wise adaptation framework that enables dynamic alignment between visual and learnable textual spaces for fine-grained anomaly learning. Rather than mapping all visual tokens to a single, token-agnostic textual space, TokenCLIP aligns each token with a customized textual subspace that represents its visual characteristics. Explicitly assigning a unique learnable textual space to each token is computationally intractable and prone to insufficient optimization. We instead expand the token-agnostic textual space into a set of orthogonal subspaces, and then dynamically assign each token to a subspace combination guided by semantic affinity, which jointly supports customized and efficient token-wise adaptation. To this end, we formulate dynamic alignment as an optimal transport problem, where all visual tokens in an image are transported to textual subspaces based on semantic similarity. The transport constraints of OT ensure sufficient optimization across subspaces and encourage them to focus on different semantics. Solving the problem yields a transport plan that adaptively assigns each token to semantically relevant subspaces. A top-k masking is then applied to sparsify the plan and specialize subspaces for distinct visual regions. Extensive experiments demonstrate the superiority of TokenCLIP.
>
---
#### [replaced 103] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.14350v3](http://arxiv.org/pdf/2503.14350v3)**

> **作者:** Shoubin Yu; Difan Liu; Ziqiao Ma; Yicong Hong; Yang Zhou; Hao Tan; Joyce Chai; Mohit Bansal
>
> **备注:** ICCV 2025; First three authors contributed equally. Project page: https://veggie-gen.github.io/
>
> **摘要:** Recent video diffusion models have enhanced video editing, but it remains challenging to handle instructional editing and diverse tasks (e.g., adding, removing, changing) within a unified framework. In this paper, we introduce VEGGIE, a Video Editor with Grounded Generation from Instructions, a simple end-to-end framework that unifies video concept editing, grounding, and reasoning based on diverse user instructions. Specifically, given a video and text query, VEGGIE first utilizes an MLLM to interpret user intentions in instructions and ground them to the video contexts, generating frame-specific grounded task queries for pixel-space responses. A diffusion model then renders these plans and generates edited videos that align with user intent. To support diverse tasks and complex instructions, we employ a curriculum learning strategy: first aligning the MLLM and video diffusion model with large-scale instructional image editing data, followed by end-to-end fine-tuning on high-quality multitask video data. Additionally, we introduce a novel data synthesis pipeline to generate paired instructional video editing data for model training. It transforms static image data into diverse, high-quality video editing samples by leveraging Image-to-Video models to inject dynamics. VEGGIE shows strong performance in instructional video editing with different editing skills, outperforming the best instructional baseline as a versatile model, while other models struggle with multi-tasking. VEGGIE also excels in video object grounding and reasoning segmentation, where other baselines fail. We further reveal how the multiple tasks help each other and highlight promising applications like zero-shot multimodal instructional and in-context video editing.
>
---
#### [replaced 104] SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent
- **分类: cs.GR; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.20414v2](http://arxiv.org/pdf/2509.20414v2)**

> **作者:** Yandan Yang; Baoxiong Jia; Shujie Zhang; Siyuan Huang
>
> **备注:** Accepted by NeurIPS 2025, 26 pages
>
> **摘要:** Indoor scene synthesis has become increasingly important with the rise of Embodied AI, which requires 3D environments that are not only visually realistic but also physically plausible and functionally diverse. While recent approaches have advanced visual fidelity, they often remain constrained to fixed scene categories, lack sufficient object-level detail and physical consistency, and struggle to align with complex user instructions. In this work, we present SceneWeaver, a reflective agentic framework that unifies diverse scene synthesis paradigms through tool-based iterative refinement. At its core, SceneWeaver employs a language model-based planner to select from a suite of extensible scene generation tools, ranging from data-driven generative models to visual- and LLM-based methods, guided by self-evaluation of physical plausibility, visual realism, and semantic alignment with user input. This closed-loop reason-act-reflect design enables the agent to identify semantic inconsistencies, invoke targeted tools, and update the environment over successive iterations. Extensive experiments on both common and open-vocabulary room types demonstrate that SceneWeaver not only outperforms prior methods on physical, visual, and semantic metrics, but also generalizes effectively to complex scenes with diverse instructions, marking a step toward general-purpose 3D environment generation. Project website: https://scene-weaver.github.io/.
>
---
#### [replaced 105] Weakly Supervised Learning for Facial Behavior Analysis : A Review
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2101.09858v5](http://arxiv.org/pdf/2101.09858v5)**

> **作者:** R. Gnana Praveen; Patrick Cardinal; Eric Granger
>
> **备注:** Provided a link of constantly updated papers \url{https://github.com/praveena2j/ awesome-Weakly-Supervised-Facial-Behavior-Analysis}
>
> **摘要:** In the recent years, there has been a shift in facial behavior analysis from the laboratory-controlled conditions to the challenging in-the-wild conditions due to the superior performance of deep learning based approaches for many real world applications.However, the performance of deep learning approaches relies on the amount of training data. One of the major problems with data acquisition is the requirement of annotations for large amount of training data. Labeling process of huge training data demands lot of human support with strong domain expertise for facial expressions or action units, which is difficult to obtain in real-time environments.Moreover, labeling process is highly vulnerable to ambiguity of expressions or action units, especially for intensities due to the bias induced by the domain experts. Therefore, there is an imperative need to address the problem of facial behavior analysis with weak annotations. In this paper, we provide a comprehensive review of weakly supervised learning (WSL) approaches for facial behavior analysis with both categorical as well as dimensional labels along with the challenges and potential research directions associated with it. First, we introduce various types of weak annotations in the context of facial behavior analysis and the corresponding challenges associated with it. We then systematically review the existing state-of-the-art approaches and provide a taxonomy of these approaches along with their insights and limitations. In addition, widely used data-sets in the reviewed literature and the performance of these approaches along with evaluation principles are summarized. Finally, we discuss the remaining challenges and opportunities along with the potential research directions in order to apply facial behavior analysis with weak labels in real life situations.
>
---
#### [replaced 106] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.18094v3](http://arxiv.org/pdf/2509.18094v3)**

> **作者:** Ye Liu; Zongyang Ma; Junfu Pu; Zhongang Qi; Yang Wu; Ying Shan; Chang Wen Chen
>
> **备注:** NeurIPS 2025 Camera Ready. Project Page: https://polyu-chenlab.github.io/unipixel/
>
> **摘要:** Recent advances in Large Multi-modal Models (LMMs) have demonstrated their remarkable success as general-purpose multi-modal assistants, with particular focuses on holistic image- and video-language understanding. Conversely, less attention has been given to scaling fine-grained pixel-level understanding capabilities, where the models are expected to realize pixel-level alignment between visual signals and language semantics. Some previous studies have applied LMMs to related tasks such as region-level captioning and referring expression segmentation. However, these models are limited to performing either referring or segmentation tasks independently and fail to integrate these fine-grained perception capabilities into visual reasoning. To bridge this gap, we propose UniPixel, a large multi-modal model capable of flexibly comprehending visual prompt inputs and generating mask-grounded responses. Our model distinguishes itself by seamlessly integrating pixel-level perception with general visual understanding capabilities. Specifically, UniPixel processes visual prompts and generates relevant masks on demand, and performs subsequent reasoning conditioning on these intermediate pointers during inference, thereby enabling fine-grained pixel-level reasoning. The effectiveness of our approach has been verified on 10 benchmarks across a diverse set of tasks, including pixel-level referring/segmentation and object-centric understanding in images/videos. A novel PixelQA task that jointly requires referring, segmentation, and question answering is also designed to verify the flexibility of our method.
>
---
#### [replaced 107] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18087v2](http://arxiv.org/pdf/2505.18087v2)**

> **作者:** Hyungyung Lee; Geon Choi; Jung-Oh Lee; Hangyul Yoon; Hyuk Gi Hong; Edward Choi
>
> **备注:** Accepted at NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Recent progress in Large Vision-Language Models (LVLMs) has enabled promising applications in medical tasks, such as report generation and visual question answering. However, existing benchmarks focus mainly on the final diagnostic answer, offering limited insight into whether models engage in clinically meaningful reasoning. To address this, we present CheXStruct and CXReasonBench, a structured pipeline and benchmark built on the publicly available MIMIC-CXR-JPG dataset. CheXStruct automatically derives a sequence of intermediate reasoning steps directly from chest X-rays, such as segmenting anatomical regions, deriving anatomical landmarks and diagnostic measurements, computing diagnostic indices, and applying clinical thresholds. CXReasonBench leverages this pipeline to evaluate whether models can perform clinically valid reasoning steps and to what extent they can learn from structured guidance, enabling fine-grained and transparent assessment of diagnostic reasoning. The benchmark comprises 18,988 QA pairs across 12 diagnostic tasks and 1,200 cases, each paired with up to 4 visual inputs, and supports multi-path, multi-stage evaluation including visual grounding via anatomical region selection and diagnostic measurements. Even the strongest of 12 evaluated LVLMs struggle with structured reasoning and generalization, often failing to link abstract knowledge with anatomically grounded visual interpretation. The code is available at https://github.com/ttumyche/CXReasonBench
>
---
#### [replaced 108] Progressive Multi-Source Domain Adaptation for Personalized Facial Expression Recognition
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04252v2](http://arxiv.org/pdf/2504.04252v2)**

> **作者:** Muhammad Osama Zeeshan; Marco Pedersoli; Alessandro Lameiras Koerich; Eric Granger
>
> **备注:** Transactions on Affective Computing 2025
>
> **摘要:** Personalized facial expression recognition (FER) involves adapting a machine learning model using samples from labeled sources and unlabeled target domains. Given the challenges of recognizing subtle expressions with considerable interpersonal variability, state-of-the-art unsupervised domain adaptation (UDA) methods focus on the multi-source UDA (MSDA) setting, where each domain corresponds to a specific subject, and improve model accuracy and robustness. However, when adapting to a specific target, the diverse nature of multiple source domains translates to a large shift between source and target data. State-of-the-art MSDA methods for FER address this domain shift by considering all the sources to adapt to the target representations. Nevertheless, adapting to a target subject presents significant challenges due to large distributional differences between source and target domains, often resulting in negative transfer. In addition, integrating all sources simultaneously increases computational costs and causes misalignment with the target. To address these issues, we propose a progressive MSDA approach that gradually introduces information from subjects based on their similarity to the target subject. This will ensure that only the most relevant sources from the target are selected, which helps avoid the negative transfer caused by dissimilar sources. We first exploit the closest sources to reduce the distribution shift with the target and then move towards the furthest while only considering the most relevant sources based on the predetermined threshold. Furthermore, to mitigate catastrophic forgetting caused by the incremental introduction of source subjects, we implemented a density-based memory mechanism that preserves the most relevant historical source samples for adaptation. Our extensive experiments on Biovid, UNBC-McMaster, Aff-Wild2, BAH, and in a cross-dataset setting.
>
---
#### [replaced 109] Kernel Density Steering: Inference-Time Scaling via Mode Seeking for Image Restoration
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.05604v2](http://arxiv.org/pdf/2507.05604v2)**

> **作者:** Yuyang Hu; Kangfu Mei; Mojtaba Sahraee-Ardakan; Ulugbek S. Kamilov; Peyman Milanfar; Mauricio Delbracio
>
> **摘要:** Diffusion models show promise for image restoration, but existing methods often struggle with inconsistent fidelity and undesirable artifacts. To address this, we introduce Kernel Density Steering (KDS), a novel inference-time framework promoting robust, high-fidelity outputs through explicit local mode-seeking. KDS employs an $N$-particle ensemble of diffusion samples, computing patch-wise kernel density estimation gradients from their collective outputs. These gradients steer patches in each particle towards shared, higher-density regions identified within the ensemble. This collective local mode-seeking mechanism, acting as "collective wisdom", steers samples away from spurious modes prone to artifacts, arising from independent sampling or model imperfections, and towards more robust, high-fidelity structures. This allows us to obtain better quality samples at the expense of higher compute by simultaneously sampling multiple particles. As a plug-and-play framework, KDS requires no retraining or external verifiers, seamlessly integrating with various diffusion samplers. Extensive numerical validations demonstrate KDS substantially improves both quantitative and qualitative performance on challenging real-world super-resolution and image inpainting tasks.
>
---
#### [replaced 110] GRE Suite: Geo-localization Inference via Fine-Tuned Vision-Language Models and Enhanced Reasoning Chains
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18700v4](http://arxiv.org/pdf/2505.18700v4)**

> **作者:** Chun Wang; Xiaojun Ye; Xiaoran Pan; Zihao Pan; Haofan Wang; Yiren Song
>
> **摘要:** Recent advances in Visual Language Models (VLMs) have demonstrated exceptional performance in visual reasoning tasks. However, geo-localization presents unique challenges, requiring the extraction of multigranular visual cues from images and their integration with external world knowledge for systematic reasoning. Current approaches to geo-localization tasks often lack robust reasoning mechanisms and explainability, limiting their effectiveness. To address these limitations, we propose the Geo Reason Enhancement (GRE) Suite, a novel framework that augments VLMs with structured reasoning chains for accurate and interpretable location inference. The GRE Suite is systematically developed across three key dimensions: dataset, model, and benchmark. First, we introduce GRE30K, a high-quality geo-localization reasoning dataset designed to facilitate fine-grained visual and contextual analysis. Next, we present the GRE model, which employs a multi-stage reasoning strategy to progressively infer scene attributes, local details, and semantic features, thereby narrowing down potential geographic regions with enhanced precision. Finally, we construct the Geo Reason Evaluation Benchmark (GREval-Bench), a comprehensive evaluation framework that assesses VLMs across diverse urban, natural, and landmark scenes to measure both coarse-grained (e.g., country, continent) and fine-grained (e.g., city, street) localization performance. Experimental results demonstrate that GRE significantly outperforms existing methods across all granularities of geo-localization tasks, underscoring the efficacy of reasoning-augmented VLMs in complex geographic inference. Code and data will be released at https://github.com/Thorin215/GRE.
>
---
#### [replaced 111] Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04789v2](http://arxiv.org/pdf/2506.04789v2)**

> **作者:** Gaia Di Lorenzo; Federico Tombari; Marc Pollefeys; Daniel Barath
>
> **摘要:** Learning effective multi-modal 3D representations of objects is essential for numerous applications, such as augmented reality and robotics. Existing methods often rely on task-specific embeddings that are tailored either for semantic understanding or geometric reconstruction. As a result, these embeddings typically cannot be decoded into explicit geometry and simultaneously reused across tasks. In this paper, we propose Object-X, a versatile multi-modal object representation framework capable of encoding rich object embeddings (e.g. images, point cloud, text) and decoding them back into detailed geometric and visual reconstructions. Object-X operates by geometrically grounding the captured modalities in a 3D voxel grid and learning an unstructured embedding fusing the information from the voxels with the object attributes. The learned embedding enables 3D Gaussian Splatting-based object reconstruction, while also supporting a range of downstream tasks, including scene alignment, single-image 3D object reconstruction, and localization. Evaluations on two challenging real-world datasets demonstrate that Object-X produces high-fidelity novel-view synthesis comparable to standard 3D Gaussian Splatting, while significantly improving geometric accuracy. Moreover, Object-X achieves competitive performance with specialized methods in scene alignment and localization. Critically, our object-centric descriptors require 3-4 orders of magnitude less storage compared to traditional image- or point cloud-based approaches, establishing Object-X as a scalable and highly practical solution for multi-modal 3D scene representation.
>
---
#### [replaced 112] Open-Set 3D Semantic Instance Maps for Vision Language Navigation -- O3D-SIM
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2404.17922v2](http://arxiv.org/pdf/2404.17922v2)**

> **作者:** Laksh Nanwani; Kumaraditya Gupta; Aditya Mathur; Swayam Agrawal; A. H. Abdul Hafez; K. Madhava Krishna
>
> **摘要:** Humans excel at forming mental maps of their surroundings, equipping them to understand object relationships and navigate based on language queries. Our previous work, SI Maps (Nanwani L, Agarwal A, Jain K, et al. Instance-level semantic maps for vision language navigation. In: 2023 32nd IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). IEEE; 2023 Aug.), showed that having instance-level information and the semantic understanding of an environment helps significantly improve performance for language-guided tasks. We extend this instance-level approach to 3D while increasing the pipeline's robustness and improving quantitative and qualitative results. Our method leverages foundational models for object recognition, image segmentation, and feature extraction. We propose a representation that results in a 3D point cloud map with instance-level embeddings, which bring in the semantic understanding that natural language commands can query. Quantitatively, the work improves upon the success rate of language-guided tasks. At the same time, we qualitatively observe the ability to identify instances more clearly and leverage the foundational models and language and image-aligned embeddings to identify objects that, otherwise, a closed-set approach wouldn't be able to identify. Project Page - https://smart-wheelchair-rrc.github.io/o3d-sim-webpage
>
---
#### [replaced 113] Steerable Transformers for Volumetric Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.15932v4](http://arxiv.org/pdf/2405.15932v4)**

> **作者:** Soumyabrata Kundu; Risi Kondor
>
> **摘要:** We introduce Steerable Transformers, an extension of the Vision Transformer mechanism that maintains equivariance to the special Euclidean group $\mathrm{SE}(d)$. We propose an equivariant attention mechanism that operates on features extracted by steerable convolutions. Operating in Fourier space, our network utilizes Fourier space non-linearities. Our experiments in both two and three dimensions show that adding steerable transformer layers to steerable convolutional networks enhances performance.
>
---
#### [replaced 114] RealCustom++: Representing Images as Real Textual Word for Real-Time Customization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.09744v3](http://arxiv.org/pdf/2408.09744v3)**

> **作者:** Zhendong Mao; Mengqi Huang; Fei Ding; Mingcong Liu; Qian He; Yongdong Zhang
>
> **备注:** 18 pages
>
> **摘要:** Given a text and an image of a specific subject, text-to-image customization aims to generate new images that align with both the text and the subject's appearance. Existing works follow the pseudo-word paradigm, which represents the subject as a non-existent pseudo word and combines it with other text to generate images. However, the pseudo word causes semantic conflict from its different learning objective and entanglement from overlapping influence scopes with other texts, resulting in a dual-optimum paradox where subject similarity and text controllability cannot be optimal simultaneously. To address this, we propose RealCustom++, a novel real-word paradigm that represents the subject with a non-conflicting real word to firstly generate a coherent guidance image and corresponding subject mask, thereby disentangling the influence scopes of the text and subject for simultaneous optimization. Specifically, RealCustom++ introduces a train-inference decoupled framework: (1) during training, it learns a general alignment between visual conditions and all real words in the text; and (2) during inference, a dual-branch architecture is employed, where the Guidance Branch produces the subject guidance mask and the Generation Branch utilizes this mask to customize the generation of the specific real word exclusively within subject-relevant regions. In contrast to previous methods that excel in either controllability or similarity, RealCustom++ achieves superior performance in both, with improvements of 7.48% in controllability, 3.04% in similarity, and 76.43% in generation quality. For multi-subject customization, RealCustom++ further achieves improvements of 4.6% in controllability and 6.34% in multi-subject similarity. Our work has been applied in JiMeng of ByteDance, and codes are released at https://github.com/bytedance/RealCustom.
>
---
#### [replaced 115] Reducing the Representation Error of GAN Image Priors Using the Deep Decoder
- **分类: cs.LG; cs.CV; eess.IV; stat.ML**

- **链接: [http://arxiv.org/pdf/2001.08747v2](http://arxiv.org/pdf/2001.08747v2)**

> **作者:** Mara Daniels; Paul Hand; Reinhard Heckel
>
> **摘要:** Generative models, such as GANs, learn an explicit low-dimensional representation of a particular class of images, and so they may be used as natural image priors for solving inverse problems such as image restoration and compressive sensing. GAN priors have demonstrated impressive performance on these tasks, but they can exhibit substantial representation error for both in-distribution and out-of-distribution images, because of the mismatch between the learned, approximate image distribution and the data generating distribution. In this paper, we demonstrate a method for reducing the representation error of GAN priors by modeling images as the linear combination of a GAN prior with a Deep Decoder. The deep decoder is an underparameterized and most importantly unlearned natural signal model similar to the Deep Image Prior. No knowledge of the specific inverse problem is needed in the training of the GAN underlying our method. For compressive sensing and image superresolution, our hybrid model exhibits consistently higher PSNRs than both the GAN priors and Deep Decoder separately, both on in-distribution and out-of-distribution images. This model provides a method for extensibly and cheaply leveraging both the benefits of learned and unlearned image recovery priors in inverse problems.
>
---
#### [replaced 116] A Poisson-Guided Decomposition Network for Extreme Low-Light Image Enhancement
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04470v2](http://arxiv.org/pdf/2506.04470v2)**

> **作者:** Isha Rao; Ratul Chakraborty; Sanjay Ghosh
>
> **备注:** 8 pages, 3 figures and 1 table
>
> **摘要:** Low-light image denoising and enhancement are challenging, especially when traditional noise assumptions, such as Gaussian noise, do not hold in majority. In many real-world scenarios, such as low-light imaging, noise is signal-dependent and is better represented as Poisson noise. In this work, we address the problem of denoising images degraded by Poisson noise under extreme low-light conditions. We introduce a light-weight deep learning-based method that integrates Retinex based decomposition with Poisson denoising into a unified encoder-decoder network. The model simultaneously enhances illumination and suppresses noise by incorporating a Poisson denoising loss to address signal-dependent noise. Without prior requirement for reflectance and illumination, the network learns an effective decomposition process while ensuring consistent reflectance and smooth illumination without causing any form of color distortion. The experimental results demonstrate the effectiveness and practicality of the proposed low-light illumination enhancement method. Our method significantly improves visibility and brightness in low-light conditions, while preserving image structure and color constancy under ambient illumination.
>
---
#### [replaced 117] Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.13227v3](http://arxiv.org/pdf/2505.13227v3)**

> **作者:** Tianbao Xie; Jiaqi Deng; Xiaochuan Li; Junlin Yang; Haoyuan Wu; Jixuan Chen; Wenjing Hu; Xinyuan Wang; Yuhui Xu; Zekun Wang; Yiheng Xu; Junli Wang; Doyen Sahoo; Tao Yu; Caiming Xiong
>
> **备注:** 49 pages, 13 figures
>
> **摘要:** Graphical user interface (GUI) grounding, the ability to map natural language instructions to specific actions on graphical user interfaces, remains a critical bottleneck in computer use agent development. Current benchmarks oversimplify grounding tasks as short referring expressions, failing to capture the complexity of real-world interactions that require software commonsense, layout understanding, and fine-grained manipulation capabilities. To address these limitations, we introduce OSWorld-G, a comprehensive benchmark comprising 564 finely annotated samples across diverse task types including text matching, element recognition, layout understanding, and precise manipulation. Additionally, we synthesize and release the largest computer use grounding dataset Jedi, which contains 4 million examples through multi-perspective decoupling of tasks. Our multi-scale models trained on Jedi demonstrate its effectiveness by outperforming existing approaches on ScreenSpot-v2, ScreenSpot-Pro, and our OSWorld-G. Furthermore, we demonstrate that improved grounding with Jedi directly enhances agentic capabilities of general foundation models on complex computer tasks, improving from 5% to 27% on OSWorld. Through detailed ablation studies, we identify key factors contributing to grounding performance and verify that combining specialized data for different interface elements enables compositional generalization to novel interfaces. All benchmark, data, checkpoints, and code are open-sourced and available at https://osworld-grounding.github.io.
>
---
#### [replaced 118] RotaTouille: Rotation Equivariant Deep Learning for Contours
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.16359v2](http://arxiv.org/pdf/2508.16359v2)**

> **作者:** Odin Hoff Gardaa; Nello Blaser
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Contours or closed planar curves are common in many domains. For example, they appear as object boundaries in computer vision, isolines in meteorology, and the orbits of rotating machinery. In many cases when learning from contour data, planar rotations of the input will result in correspondingly rotated outputs. It is therefore desirable that deep learning models be rotationally equivariant. In addition, contours are typically represented as an ordered sequence of edge points, where the choice of starting point is arbitrary. It is therefore also desirable for deep learning methods to be equivariant under cyclic shifts. We present RotaTouille, a deep learning framework for learning from contour data that achieves both rotation and cyclic shift equivariance through complex-valued circular convolution. We further introduce and characterize equivariant non-linearities, coarsening layers, and global pooling layers to obtain invariant representations for downstream tasks. Finally, we demonstrate the effectiveness of RotaTouille through experiments in shape classification, reconstruction, and contour regression.
>
---
#### [replaced 119] TEn-CATG:Text-Enriched Audio-Visual Video Parsing with Multi-Scale Category-Aware Temporal Graph
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2509.04086v2](http://arxiv.org/pdf/2509.04086v2)**

> **作者:** Yaru Chen; Faegheh Sardari; Peiliang Zhang; Ruohao Guo; Yang Xiang; Zhenbo Li; Wenwu Wang
>
> **摘要:** Audio-visual video parsing (AVVP) aims to detect event categories and their temporal boundaries in videos, typically under weak supervision. Existing methods mainly focus on (i) improving temporal modeling using attention-based architectures or (ii) generating richer pseudo-labels to address the absence of frame-level annotations. However, attention-based models often overfit noisy pseudo-labels, leading to cumulative training errors, while pseudo-label generation approaches distribute attention uniformly across frames, weakening temporal localization accuracy. To address these challenges, we propose TEn-CATG, a text-enriched AVVP framework that combines semantic calibration with category-aware temporal reasoning. More specifically, we design a bi-directional text fusion (BiT) module by leveraging audio-visual features as semantic anchors to refine text embeddings, which departs from conventional text-to-feature alignment, thereby mitigating noise and enhancing cross-modal consistency. Furthermore, we introduce the category-aware temporal graph (CATG) module to model temporal relationships by selecting multi-scale temporal neighbors and learning category-specific temporal decay factors, enabling effective event-dependent temporal reasoning. Extensive experiments demonstrate that TEn-CATG achieves state-of-the-art results across multiple evaluation metrics on benchmark datasets LLP and UnAV-100, highlighting its robustness and superior ability to capture complex temporal and semantic dependencies in weakly supervised AVVP tasks.
>
---
#### [replaced 120] Beyond Seeing: Evaluating Multimodal LLMs on Tool-Enabled Image Perception, Transformation, and Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.12712v3](http://arxiv.org/pdf/2510.12712v3)**

> **作者:** Xingang Guo; Utkarsh Tyagi; Advait Gosai; Paula Vergara; Jayeon Park; Ernesto Gabriel Hernández Montoya; Chen Bo Calvin Zhang; Bin Hu; Yunzhong He; Bing Liu; Rakshith Sharma Srinivasa
>
> **摘要:** Multimodal Large Language Models (MLLMs) are increasingly applied in real-world scenarios where user-provided images are often imperfect, requiring active image manipulations such as cropping, editing, or enhancement to uncover salient visual cues. Beyond static visual perception, MLLMs must also think with images: dynamically transforming visual content and integrating it with other tools to solve complex tasks. However, this shift from treating vision as passive context to a manipulable cognitive workspace remains underexplored. Most existing benchmarks still follow a think about images paradigm, where images are regarded as static inputs. To address this gap, we introduce VisualToolBench, a visual tool-use reasoning benchmark that rigorously evaluates MLLMs' ability to perceive, transform, and reason across complex visual-textual tasks under the think-with-images paradigm. VisualToolBench comprises 1,204 challenging, open-ended vision tasks (603 single-turn, 601 multi-turn) spanning across five diverse domains, each paired with detailed rubrics to enable systematic evaluation. Our evaluation shows that current MLLMs struggle with tasks requiring effective integration of vision and general-purpose tools. Even the strongest model (GPT-5-think) reaches only 18.68% pass rate. We further observe divergent tool-use behaviors, with OpenAI models benefiting from diverse image manipulations while Gemini-2.5-pro shows no improvement. By introducing the first benchmark centered on think with images, VisualToolBench offers critical insights for advancing visual intelligence in MLLMs.
>
---
#### [replaced 121] BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.15566v4](http://arxiv.org/pdf/2509.15566v4)**

> **作者:** Shaojie Zhang; Ruoceng Zhang; Pei Fu; Shaokang Wang; Jiahui Yang; Xin Du; Shiqi Cui; Bin Qin; Ying Huang; Zhenbo Luo; Jian Luan
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** In the field of AI-driven human-GUI interaction automation, while rapid advances in multimodal large language models and reinforcement fine-tuning techniques have yielded remarkable progress, a fundamental challenge persists: their interaction logic significantly deviates from natural human-GUI communication patterns. To fill this gap, we propose "Blink-Think-Link" (BTL), a brain-inspired framework for human-GUI interaction that mimics the human cognitive process between users and graphical interfaces. The system decomposes interactions into three biologically plausible phases: (1) Blink - rapid detection and attention to relevant screen areas, analogous to saccadic eye movements; (2) Think - higher-level reasoning and decision-making, mirroring cognitive planning; and (3) Link - generation of executable commands for precise motor control, emulating human action selection mechanisms. Additionally, we introduce two key technical innovations for the BTL framework: (1) Blink Data Generation - an automated annotation pipeline specifically optimized for blink data, and (2) BTL Reward -- the first rule-based reward mechanism that enables reinforcement learning driven by both process and outcome. Building upon this framework, we develop a GUI agent model named BTL-UI, which demonstrates competitive performance across both static GUI understanding and dynamic interaction tasks in comprehensive benchmarks. These results provide conclusive empirical validation of the framework's efficacy in developing advanced GUI Agents.
>
---
#### [replaced 122] VLCE: A Knowledge-Enhanced Framework for Image Description in Disaster Assessment
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.21609v2](http://arxiv.org/pdf/2509.21609v2)**

> **作者:** Md. Mahfuzur Rahman; Kishor Datta Gupta; Marufa Kamal; Fahad Rahman; Sunzida Siddique; Ahmed Rafi Hasan; Mohd Ariful Haque; Roy George
>
> **备注:** 29 pages, 40 figures, 3 algorithms
>
> **摘要:** Immediate damage assessment is essential after natural catastrophes; yet, conventional hand evaluation techniques are sluggish and perilous. Although satellite and unmanned aerial vehicle (UAV) photos offer extensive perspectives of impacted regions, current computer vision methodologies generally yield just classification labels or segmentation masks, so constraining their capacity to deliver a thorough situational comprehension. We introduce the Vision Language Caption Enhancer (VLCE), a multimodal system designed to produce comprehensive, contextually-informed explanations of disaster imagery. VLCE employs a dual-architecture approach: a CNN-LSTM model with a ResNet50 backbone pretrained on EuroSat satellite imagery for the xBD dataset, and a Vision Transformer (ViT) model pretrained on UAV pictures for the RescueNet dataset. Both systems utilize external semantic knowledge from ConceptNet and WordNet to expand vocabulary coverage and improve description accuracy. We assess VLCE in comparison to leading vision-language models (LLaVA and QwenVL) utilizing CLIPScore for semantic alignment and InfoMetIC for caption informativeness. Experimental findings indicate that VLCE markedly surpasses baseline models, attaining a maximum of 95.33% on InfoMetIC while preserving competitive semantic alignment. Our dual-architecture system demonstrates significant potential for improving disaster damage assessment by automating the production of actionable, information-dense descriptions from satellite and drone photos.
>
---
#### [replaced 123] Structural-Spectral Graph Convolution with Evidential Edge Learning for Hyperspectral Image Clustering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09920v3](http://arxiv.org/pdf/2506.09920v3)**

> **作者:** Jianhan Qi; Yuheng Jia; Hui Liu; Junhui Hou
>
> **摘要:** Hyperspectral image (HSI) clustering groups pixels into clusters without labeled data, which is an important yet challenging task. For large-scale HSIs, most methods rely on superpixel segmentation and perform superpixel-level clustering based on graph neural networks (GNNs). However, existing GNNs cannot fully exploit the spectral information of the input HSI, and the inaccurate superpixel topological graph may lead to the confusion of different class semantics during information aggregation. To address these challenges, we first propose a structural-spectral graph convolutional operator (SSGCO) tailored for graph-structured HSI superpixels to improve their representation quality through the co-extraction of spatial and spectral features. Second, we propose an evidence-guided adaptive edge learning (EGAEL) module that adaptively predicts and refines edge weights in the superpixel topological graph. We integrate the proposed method into a contrastive learning framework to achieve clustering, where representation learning and clustering are simultaneously conducted. Experiments demonstrate that the proposed method improves clustering accuracy by 2.61%, 6.06%, 4.96% and 3.15% over the best compared methods on four HSI datasets. Our code is available at https://github.com/jhqi/SSGCO-EGAEL.
>
---
#### [replaced 124] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22204v2](http://arxiv.org/pdf/2503.22204v2)**

> **作者:** Yiren Lu; Yunlai Zhou; Yiran Qiao; Chaoda Song; Tuo Liang; Jing Ma; Huan Wang; Yu Yin
>
> **备注:** NeurIPS 2025. Project page: https://vulab-ai.github.io/Segment-then-Splat/
>
> **摘要:** Open-vocabulary querying in 3D space is crucial for enabling more intelligent perception in applications such as robotics, autonomous systems, and augmented reality. However, most existing methods rely on 2D pixel-level parsing, leading to multi-view inconsistencies and poor 3D object retrieval. Moreover, they are limited to static scenes and struggle with dynamic scenes due to the complexities of motion modeling. In this paper, we propose Segment then Splat, a 3D-aware open vocabulary segmentation approach for both static and dynamic scenes based on Gaussian Splatting. Segment then Splat reverses the long established approach of "segmentation after reconstruction" by dividing Gaussians into distinct object sets before reconstruction. Once reconstruction is complete, the scene is naturally segmented into individual objects, achieving true 3D segmentation. This design eliminates both geometric and semantic ambiguities, as well as Gaussian-object misalignment issues in dynamic scenes. It also accelerates the optimization process, as it eliminates the need for learning a separate language field. After optimization, a CLIP embedding is assigned to each object to enable open-vocabulary querying. Extensive experiments one various datasets demonstrate the effectiveness of our proposed method in both static and dynamic scenarios.
>
---
#### [replaced 125] Optimize the Unseen - Fast NeRF Cleanup with Free Space Prior
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.12772v3](http://arxiv.org/pdf/2412.12772v3)**

> **作者:** Leo Segre; Shai Avidan
>
> **摘要:** Neural Radiance Fields (NeRF) have advanced photorealistic novel view synthesis, but their reliance on photometric reconstruction introduces artifacts, commonly known as "floaters". These artifacts degrade novel view quality, especially in areas unseen by the training cameras. We present a fast, post-hoc NeRF cleanup method that eliminates such artifacts by enforcing our Free Space Prior, effectively minimizing floaters without disrupting the NeRF's representation of observed regions. Unlike existing approaches that rely on either Maximum Likelihood (ML) estimation to fit the data or a complex, local data-driven prior, our method adopts a Maximum-a-Posteriori (MAP) approach, selecting the optimal model parameters under a simple global prior assumption that unseen regions should remain empty. This enables our method to clean artifacts in both seen and unseen areas, enhancing novel view quality even in challenging scene regions. Our method is comparable with existing NeRF cleanup models while being 2.5x faster in inference time, requires no additional memory beyond the original NeRF, and achieves cleanup training in less than 30 seconds. Our code will be made publically available.
>
---
#### [replaced 126] Task-Oriented Feature Compression for Multimodal Understanding via Device-Edge Co-Inference
- **分类: eess.SP; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12926v3](http://arxiv.org/pdf/2503.12926v3)**

> **作者:** Cheng Yuan; Zhening Liu; Jiashu Lv; Jiawei Shao; Yufei Jiang; Jun Zhang; Xuelong Li
>
> **备注:** Accepted by IEEE Transactions on Mobile Computing
>
> **摘要:** With the rapid development of large multimodal models (LMMs), multimodal understanding applications are emerging. As most LMM inference requests originate from edge devices with limited computational capabilities, the predominant inference pipeline involves directly forwarding the input data to an edge server which handles all computations. However, this approach introduces high transmission latency due to limited uplink bandwidth of edge devices and significant computation latency caused by the prohibitive number of visual tokens, thus hindering delay-sensitive tasks and degrading user experience. To address this challenge, we propose a task-oriented feature compression (TOFC) method for multimodal understanding in a device-edge co-inference framework, where visual features are merged by clustering and encoded by a learnable and selective entropy model before feature projection. Specifically, we employ density peaks clustering based on K nearest neighbors to reduce the number of visual features, thereby minimizing both data transmission and computational complexity. Subsequently, a learnable entropy model with hyperprior is utilized to encode and decode merged features, further reducing transmission overhead. To enhance compression efficiency, multiple entropy models are adaptively selected based on the characteristics of the visual features, enabling a more accurate estimation of the probability distribution. Comprehensive experiments on seven visual question answering benchmarks validate the effectiveness of the proposed TOFC method. Results show that TOFC achieves up to 52% reduction in data transmission overhead and 63% reduction in system latency while maintaining identical task performance, compared with neural compression ELIC.
>
---
#### [replaced 127] Learning Knowledge-based Prompts for Robust 3D Mask Presentation Attack Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03610v2](http://arxiv.org/pdf/2505.03610v2)**

> **作者:** Fangling Jiang; Qi Li; Bing Liu; Weining Wang; Caifeng Shan; Zhenan Sun; Ming-Hsuan Yang
>
> **备注:** Accepted by TPAMI
>
> **摘要:** 3D mask presentation attack detection is crucial for protecting face recognition systems against the rising threat of 3D mask attacks. While most existing methods utilize multimodal features or remote photoplethysmography (rPPG) signals to distinguish between real faces and 3D masks, they face significant challenges, such as the high costs associated with multimodal sensors and limited generalization ability. Detection-related text descriptions offer concise, universal information and are cost-effective to obtain. However, the potential of vision-language multimodal features for 3D mask presentation attack detection remains unexplored. In this paper, we propose a novel knowledge-based prompt learning framework to explore the strong generalization capability of vision-language models for 3D mask presentation attack detection. Specifically, our approach incorporates entities and triples from knowledge graphs into the prompt learning process, generating fine-grained, task-specific explicit prompts that effectively harness the knowledge embedded in pre-trained vision-language models. Furthermore, considering different input images may emphasize distinct knowledge graph elements, we introduce a visual-specific knowledge filter based on an attention mechanism to refine relevant elements according to the visual context. Additionally, we leverage causal graph theory insights into the prompt learning process to further enhance the generalization ability of our method. During training, a spurious correlation elimination paradigm is employed, which removes category-irrelevant local image patches using guidance from knowledge-based text features, fostering the learning of generalized causal prompts that align with category-relevant local patches. Experimental results demonstrate that the proposed method achieves state-of-the-art intra- and cross-scenario detection performance on benchmark datasets.
>
---
#### [replaced 128] Editable Noise Map Inversion: Encoding Target-image into Noise For High-Fidelity Image Manipulation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.25776v3](http://arxiv.org/pdf/2509.25776v3)**

> **作者:** Mingyu Kang; Yong Suk Choi
>
> **备注:** ICML 2025
>
> **摘要:** Text-to-image diffusion models have achieved remarkable success in generating high-quality and diverse images. Building on these advancements, diffusion models have also demonstrated exceptional performance in text-guided image editing. A key strategy for effective image editing involves inverting the source image into editable noise maps associated with the target image. However, previous inversion methods face challenges in adhering closely to the target text prompt. The limitation arises because inverted noise maps, while enabling faithful reconstruction of the source image, restrict the flexibility needed for desired edits. To overcome this issue, we propose Editable Noise Map Inversion (ENM Inversion), a novel inversion technique that searches for optimal noise maps to ensure both content preservation and editability. We analyze the properties of noise maps for enhanced editability. Based on this analysis, our method introduces an editable noise refinement that aligns with the desired edits by minimizing the difference between the reconstructed and edited noise maps. Extensive experiments demonstrate that ENM Inversion outperforms existing approaches across a wide range of image editing tasks in both preservation and edit fidelity with target prompts. Our approach can also be easily applied to video editing, enabling temporal consistency and content manipulation across frames.
>
---
#### [replaced 129] ViBED-Net: Video Based Engagement Detection Network Using Face-Aware and Scene-Aware Spatiotemporal Cues
- **分类: cs.CV; cs.LG; I.2.10; I.5.2**

- **链接: [http://arxiv.org/pdf/2510.18016v2](http://arxiv.org/pdf/2510.18016v2)**

> **作者:** Prateek Gothwal; Deeptimaan Banerjee; Ashis Kumer Biswas
>
> **备注:** 10 pages, 4 figures, 2 tables
>
> **摘要:** Engagement detection in online learning environments is vital for improving student outcomes and personalizing instruction. We present ViBED-Net (Video-Based Engagement Detection Network), a novel deep learning framework designed to assess student engagement from video data using a dual-stream architecture. ViBED-Net captures both facial expressions and full-scene context by processing facial crops and entire video frames through EfficientNetV2 for spatial feature extraction. These features are then analyzed over time using two temporal modeling strategies: Long Short-Term Memory (LSTM) networks and Transformer encoders. Our model is evaluated on the DAiSEE dataset, a large-scale benchmark for affective state recognition in e-learning. To enhance performance on underrepresented engagement classes, we apply targeted data augmentation techniques. Among the tested variants, ViBED-Net with LSTM achieves 73.43\% accuracy, outperforming existing state-of-the-art approaches. ViBED-Net demonstrates that combining face-aware and scene-aware spatiotemporal cues significantly improves engagement detection accuracy. Its modular design allows flexibility for application across education, user experience research, and content personalization. This work advances video-based affective computing by offering a scalable, high-performing solution for real-world engagement analysis. The source code for this project is available on https://github.com/prateek-gothwal/ViBED-Net .
>
---
#### [replaced 130] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.15963v2](http://arxiv.org/pdf/2510.15963v2)**

> **作者:** Jiani Huang; Amish Sethi; Matthew Kuo; Mayank Keoliya; Neelay Velingker; JungHo Jung; Ser-Nam Lim; Ziyang Li; Mayur Naik
>
> **备注:** Accepted as a Spotlight Paper at NeurIPS 2025
>
> **摘要:** Multi-modal large language models (MLLMs) are making rapid progress toward general-purpose embodied agents. However, existing MLLMs do not reliably capture fine-grained links between low-level visual features and high-level textual semantics, leading to weak grounding and inaccurate perception. To overcome this challenge, we propose ESCA, a framework that contextualizes embodied agents by grounding their perception in spatial-temporal scene graphs. At its core is SGCLIP, a novel, open-domain, promptable foundation model for generating scene graphs that is based on CLIP. SGCLIP is trained on 87K+ open-domain videos using a neurosymbolic pipeline that aligns automatically generated captions with scene graphs produced by the model itself, eliminating the need for human-labeled annotations. We demonstrate that SGCLIP excels in both prompt-based inference and task-specific fine-tuning, achieving state-of-the-art results on scene graph generation and action localization benchmarks. ESCA with SGCLIP improves perception for embodied agents based on both open-source and commercial MLLMs, achieving state of-the-art performance across two embodied environments. Notably, ESCA significantly reduces agent perception errors and enables open-source models to surpass proprietary baselines. We release the source code for SGCLIP model training at https://github.com/video-fm/LASER and for the embodied agent at https://github.com/video-fm/ESCA.
>
---
#### [replaced 131] Dynamic-Aware Spatio-temporal Representation Learning for Dynamic MRI Reconstruction
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.09049v2](http://arxiv.org/pdf/2501.09049v2)**

> **作者:** Dayoung Baik; Jaejun Yoo
>
> **备注:** MICCAI2025
>
> **摘要:** Dynamic MRI reconstruction, one of inverse problems, has seen a surge by the use of deep learning techniques. Especially, the practical difficulty of obtaining ground truth data has led to the emergence of unsupervised learning approaches. A recent promising method among them is implicit neural representation (INR), which defines the data as a continuous function that maps coordinate values to the corresponding signal values. This allows for filling in missing information only with incomplete measurements and solving the inverse problem effectively. Nevertheless, previous works incorporating this method have faced drawbacks such as long optimization time and the need for extensive hyperparameter tuning. To address these issues, we propose Dynamic-Aware INR (DA-INR), an INR-based model for dynamic MRI reconstruction that captures the spatial and temporal continuity of dynamic MRI data in the image domain and explicitly incorporates the temporal redundancy of the data into the model structure. As a result, DA-INR outperforms other models in reconstruction quality even at extreme undersampling ratios while significantly reducing optimization time and requiring minimal hyperparameter tuning.
>
---
#### [replaced 132] Radiant Triangle Soup with Soft Connectivity Forces for 3D Reconstruction and Novel View Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23642v2](http://arxiv.org/pdf/2505.23642v2)**

> **作者:** Nathaniel Burgdorfer; Philippos Mordohai
>
> **摘要:** We introduce an inference-time scene optimization algorithm utilizing triangle soup, a collection of disconnected translucent triangle primitives, as the representation for the geometry and appearance of a scene. Unlike full-rank Gaussian kernels, triangles are a natural, locally-flat proxy for surfaces that can be connected to achieve highly complex geometry. When coupled with per-vertex Spherical Harmonics (SH), triangles provide a rich visual representation without incurring an expensive increase in primitives. We leverage our new representation to incorporate optimization objectives and enforce spatial regularization directly on the underlying primitives. The main differentiator of our approach is the definition and enforcement of soft connectivity forces between triangles during optimization, encouraging explicit, but soft, surface continuity in 3D. Experiments on representative 3D reconstruction and novel view synthesis datasets show improvements in geometric accuracy compared to current state-of-the-art algorithms without sacrificing visual fidelity.
>
---
#### [replaced 133] DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.14376v2](http://arxiv.org/pdf/2510.14376v2)**

> **作者:** Dongnam Byun; Jungwon Park; Jumgmin Ko; Changin Choi; Wonjong Rhee
>
> **摘要:** Recent progress in text-to-image (T2I) generative models has led to significant improvements in generating high-quality images aligned with text prompts. However, these models still struggle with prompts involving multiple objects, often resulting in object neglect or object mixing. Through extensive studies, we identify four problematic scenarios, Similar Shapes, Similar Textures, Dissimilar Background Biases, and Many Objects, where inter-object relationships frequently lead to such failures. Motivated by two key observations about CLIP embeddings, we propose DOS (Directional Object Separation), a method that modifies three types of CLIP text embeddings before passing them into text-to-image models. Experimental results show that DOS consistently improves the success rate of multi-object image generation and reduces object mixing. In human evaluations, DOS significantly outperforms four competing methods, receiving 26.24%-43.04% more votes across four benchmarks. These results highlight DOS as a practical and effective solution for improving multi-object image generation.
>
---
#### [replaced 134] LayerComposer: Interactive Personalized T2I via Spatially-Aware Layered Canvas
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.20820v2](http://arxiv.org/pdf/2510.20820v2)**

> **作者:** Guocheng Gordon Qian; Ruihang Zhang; Tsai-Shien Chen; Yusuf Dalva; Anujraaj Argo Goyal; Willi Menapace; Ivan Skorokhodov; Meng Dong; Arpit Sahni; Daniil Ostashev; Ju Hu; Sergey Tulyakov; Kuan-Chieh Jackson Wang
>
> **备注:** 9 pages, preprint. Project page: https://snap-research.github.io/layercomposer/
>
> **摘要:** Despite their impressive visual fidelity, existing personalized generative models lack interactive control over spatial composition and scale poorly to multiple subjects. To address these limitations, we present LayerComposer, an interactive framework for personalized, multi-subject text-to-image generation. Our approach introduces two main contributions: (1) a layered canvas, a novel representation in which each subject is placed on a distinct layer, enabling occlusion-free composition; and (2) a locking mechanism that preserves selected layers with high fidelity while allowing the remaining layers to adapt flexibly to the surrounding context. Similar to professional image-editing software, the proposed layered canvas allows users to place, resize, or lock input subjects through intuitive layer manipulation. Our versatile locking mechanism requires no architectural changes, relying instead on inherent positional embeddings combined with a new complementary data sampling strategy. Extensive experiments demonstrate that LayerComposer achieves superior spatial control and identity preservation compared to the state-of-the-art methods in multi-subject personalized image generation.
>
---
#### [replaced 135] ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.22030v2](http://arxiv.org/pdf/2507.22030v2)**

> **作者:** Mohammed Baharoon; Luyang Luo; Michael Moritz; Abhinav Kumar; Sung Eun Kim; Xiaoman Zhang; Miao Zhu; Mahmoud Hussain Alabbad; Maha Sbayel Alhazmi; Neel P. Mistry; Lucas Bijnens; Kent Ryan Kleinschmidt; Brady Chrisler; Sathvik Suryadevara; Sri Sai Dinesh Jaliparthi; Noah Michael Prudlo; Mark David Marino; Jeremy Palacio; Rithvik Akula; Di Zhou; Hong-Yu Zhou; Ibrahim Ethem Hamamci; Scott J. Adams; Hassan Rayhan AlOmaish; Pranav Rajpurkar
>
> **摘要:** We introduce ReXGroundingCT, the first publicly available dataset linking free-text findings to pixel-level 3D segmentations in chest CT scans. The dataset includes 3,142 non-contrast chest CT scans paired with standardized radiology reports from CT-RATE. Construction followed a structured three-stage pipeline. First, GPT-4 was used to extract and standardize findings, descriptors, and metadata from reports originally written in Turkish and machine-translated into English. Second, GPT-4o-mini categorized each finding into a hierarchical ontology of lung and pleural abnormalities. Third, 3D annotations were produced for all CT volumes: the training set was quality-assured by board-certified radiologists, and the validation and test sets were fully annotated by board-certified radiologists. Additionally, a complementary chain-of-thought dataset was created to provide step-by-step hierarchical anatomical reasoning for localizing findings within the CT volume, using GPT-4o and localization coordinates derived from organ segmentation models. ReXGroundingCT contains 16,301 annotated entities across 8,028 text-to-3D-segmentation pairs, covering diverse radiological patterns from 3,142 non-contrast CT scans. About 79% of findings are focal abnormalities and 21% are non-focal. The dataset includes a public validation set of 50 cases and a private test set of 100 cases, both annotated by board-certified radiologists. The dataset establishes a foundation for enabling free-text finding segmentation and grounded radiology report generation in CT imaging. Model performance on the private test set is hosted on a public leaderboard at https://rexrank.ai/ReXGroundingCT. The dataset is available at https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT.
>
---
#### [replaced 136] Neural Stereo Video Compression with Hybrid Disparity Compensation
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.20383v2](http://arxiv.org/pdf/2504.20383v2)**

> **作者:** Shiyin Jiang; Zhenghao Chen; Minghao Han; Shuhang Gu
>
> **摘要:** Disparity compensation represents the primary strategy in stereo video compression (SVC) for exploiting cross-view redundancy. These mechanisms can be broadly categorized into two types: one that employs explicit horizontal shifting, and another that utilizes an implicit cross-attention mechanism to reduce cross-view disparity redundancy. In this work, we propose a hybrid disparity compensation (HDC) strategy that leverages explicit pixel displacement as a robust prior feature to simplify optimization and perform implicit cross-attention mechanisms for subsequent warping operations, thereby capturing a broader range of disparity information. Specifically, HDC first computes a similarity map by fusing the horizontally shifted cross-view features to capture pixel displacement information. This similarity map is then normalized into an "explicit pixel-wise attention score" to perform the cross-attention mechanism, implicitly aligning features from one view to another. Building upon HDC, we introduce a novel end-to-end optimized neural stereo video compression framework, which integrates HDC-based modules into key coding operations, including cross-view feature extraction and reconstruction (HDC-FER) and cross-view entropy modeling (HDC-EM). Extensive experiments on SVC benchmarks, including KITTI 2012, KITTI 2015, and Nagoya, which cover both autonomous driving and general scenes, demonstrate that our framework outperforms both neural and traditional SVC methodologies.
>
---
#### [replaced 137] Generalization Bounds for Robust Contrastive Learning: From Theory to Practice
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2311.09671v2](http://arxiv.org/pdf/2311.09671v2)**

> **作者:** Ngoc N. Tran; Lam Tran; Hoang Phan; Anh Bui; Tung Pham; Toan Tran; Dinh Phung; Trung Le
>
> **备注:** 13 pages, 1 figure, 4 tables
>
> **摘要:** Contrastive Learning first extracts features from unlabeled data, followed by linear probing with labeled data. Adversarial Contrastive Learning (ACL) integrates Adversarial Training into the first phase to enhance feature robustness against attacks in the probing phase. While ACL has shown strong empirical results, its theoretical understanding remains limited. Furthermore, while a fair amount of theoretical works analyze how the unsupervised loss can support the supervised loss in the probing phase, none has examined its role to the robust supervised loss. To fill this gap, our work develops rigorous theories to identify which components in the unsupervised training can help improve the robust supervised loss. Specifically, besides the adversarial contrastive loss, we reveal that the benign one, along with a global divergence between benign and adversarial examples can also improve robustness. Proper experiments are conducted to justify our findings.
>
---
#### [replaced 138] Integrating Reinforcement Learning with Visual Generative Models: Foundations and Advances
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10316v2](http://arxiv.org/pdf/2508.10316v2)**

> **作者:** Yuanzhi Liang; Yijie Fang; Rui Li; Ziqi Ni; Ruijie Su; Chi Zhang
>
> **备注:** Ongoing work
>
> **摘要:** Generative models have made significant progress in synthesizing visual content, including images, videos, and 3D/4D structures. However, they are typically trained with surrogate objectives such as likelihood or reconstruction loss, which often misalign with perceptual quality, semantic accuracy, or physical realism. Reinforcement learning (RL) offers a principled framework for optimizing non-differentiable, preference-driven, and temporally structured objectives. Recent advances demonstrate its effectiveness in enhancing controllability, consistency, and human alignment across generative tasks. This survey provides a systematic overview of RL-based methods for visual content generation. We review the evolution of RL from classical control to its role as a general-purpose optimization tool, and examine its integration into image, video, and 3D/4D generation. Across these domains, RL serves not only as a fine-tuning mechanism but also as a structural component for aligning generation with complex, high-level goals. We conclude with open challenges and future research directions at the intersection of RL and generative modeling.
>
---
#### [replaced 139] ORIGEN: Zero-Shot 3D Orientation Grounding in Text-to-Image Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.22194v3](http://arxiv.org/pdf/2503.22194v3)**

> **作者:** Yunhong Min; Daehyeon Choi; Kyeongmin Yeo; Jihyun Lee; Minhyuk Sung
>
> **备注:** Project Page: https://origen2025.github.io
>
> **摘要:** We introduce ORIGEN, the first zero-shot method for 3D orientation grounding in text-to-image generation across multiple objects and diverse categories. While previous work on spatial grounding in image generation has mainly focused on 2D positioning, it lacks control over 3D orientation. To address this, we propose a reward-guided sampling approach using a pretrained discriminative model for 3D orientation estimation and a one-step text-to-image generative flow model. While gradient-ascent-based optimization is a natural choice for reward-based guidance, it struggles to maintain image realism. Instead, we adopt a sampling-based approach using Langevin dynamics, which extends gradient ascent by simply injecting random noise--requiring just a single additional line of code. Additionally, we introduce adaptive time rescaling based on the reward function to accelerate convergence. Our experiments show that ORIGEN outperforms both training-based and test-time guidance methods across quantitative metrics and user studies.
>
---
#### [replaced 140] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.19331v3](http://arxiv.org/pdf/2503.19331v3)**

> **作者:** Chau Pham; Juan C. Caicedo; Bryan A. Plummer
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Prior work using Masked Autoencoders (MAEs) typically relies on random patch masking based on the assumption that images have significant redundancies across different channels, allowing for the reconstruction of masked content using cross-channel correlations. However, this assumption does not hold in Multi-Channel Imaging (MCI), where channels may provide complementary information with minimal feature overlap. Thus, these MAEs primarily learn local structures within individual channels from patch reconstruction, failing to fully leverage cross-channel interactions and limiting their MCI effectiveness. In this paper, we present ChA-MAEViT, an MAE-based method that enhances feature learning across MCI channels via four key strategies: (1) dynamic channel-patch masking, which compels the model to reconstruct missing channels in addition to masked patches, thereby enhancing cross-channel dependencies and improving robustness to varying channel configurations; (2) memory tokens, which serve as long-term memory aids to promote information sharing across channels, addressing the challenges of reconstructing structurally diverse channels; (3) hybrid token fusion module, which merges fine-grained patch tokens with a global class token to capture richer representations; and (4) Channel-Aware Decoder, a lightweight decoder utilizes channel tokens to effectively reconstruct image patches. Experiments on satellite and microscopy datasets, CHAMMI, JUMP-CP, and So2Sat, show that ChA-MAEViT significantly outperforms state-of-the-art MCI-ViTs by 3.0-21.5%, highlighting the importance of cross-channel interactions in MCI. Our code is publicly available at https://github.com/chaudatascience/cha_mae_vit.
>
---
#### [replaced 141] NVS-SQA: Exploring Self-Supervised Quality Representation Learning for Neurally Synthesized Scenes without References
- **分类: cs.CV; cs.AI; cs.HC; cs.MM; eess.IV**

- **链接: [http://arxiv.org/pdf/2501.06488v3](http://arxiv.org/pdf/2501.06488v3)**

> **作者:** Qiang Qu; Yiran Shen; Xiaoming Chen; Yuk Ying Chung; Weidong Cai; Tongliang Liu
>
> **备注:** Accepted by TPAMI
>
> **摘要:** Neural View Synthesis (NVS), such as NeRF and 3D Gaussian Splatting, effectively creates photorealistic scenes from sparse viewpoints, typically evaluated by quality assessment methods like PSNR, SSIM, and LPIPS. However, these full-reference methods, which compare synthesized views to reference views, may not fully capture the perceptual quality of neurally synthesized scenes (NSS), particularly due to the limited availability of dense reference views. Furthermore, the challenges in acquiring human perceptual labels hinder the creation of extensive labeled datasets, risking model overfitting and reduced generalizability. To address these issues, we propose NVS-SQA, a NSS quality assessment method to learn no-reference quality representations through self-supervision without reliance on human labels. Traditional self-supervised learning predominantly relies on the "same instance, similar representation" assumption and extensive datasets. However, given that these conditions do not apply in NSS quality assessment, we employ heuristic cues and quality scores as learning objectives, along with a specialized contrastive pair preparation process to improve the effectiveness and efficiency of learning. The results show that NVS-SQA outperforms 17 no-reference methods by a large margin (i.e., on average 109.5% in SRCC, 98.6% in PLCC, and 91.5% in KRCC over the second best) and even exceeds 16 full-reference methods across all evaluation metrics (i.e., 22.9% in SRCC, 19.1% in PLCC, and 18.6% in KRCC over the second best).
>
---
#### [replaced 142] Image-Plane Geometric Decoding for View-Invariant Indoor Scene Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.25744v2](http://arxiv.org/pdf/2509.25744v2)**

> **作者:** Mingyang Li; Yimeng Fan; Changsong Liu; Lixue Xu; Xin Wang; Yanyan Liu; Wei Zhang
>
> **摘要:** Volume-based indoor scene reconstruction methods offer superior generalization capability and real-time deployment potential. However, existing methods rely on multi-view pixel back-projection ray intersections as weak geometric constraints to determine spatial positions. This dependence results in reconstruction quality being heavily influenced by input view density. Performance degrades in overlapping regions and unobserved areas.To address these limitations, we reduce dependency on inter-view geometric constraints by exploiting spatial information within individual views. We propose an image-plane decoding framework with three core components: Pixel-level Confidence Encoder, Affine Compensation Module, and Image-Plane Spatial Decoder. These modules decode three-dimensional structural information encoded in images through physical imaging processes. The framework effectively preserves spatial geometric features including edges, hollow structures, and complex textures. It significantly enhances view-invariant reconstruction.Experiments on indoor scene reconstruction datasets confirm superior reconstruction stability. Our method maintains nearly identical quality when view count reduces by 40%. It achieves a coefficient of variation of 0.24%, performance retention rate of 99.7%, and maximum performance drop of 0.42%. These results demonstrate that exploiting intra-view spatial information provides a robust solution for view-limited scenarios in practical applications.
>
---
#### [replaced 143] Smartphone-based iris recognition through high-quality visible-spectrum iris image capture.V2
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.06170v2](http://arxiv.org/pdf/2510.06170v2)**

> **作者:** Naveenkumar G Venkataswamy; Yu Liu; Soumyabrata Dey; Stephanie Schuckers; Masudul H Imtiaz
>
> **备注:** This submission has been withdrawn because it duplicates significant content from another version of the paper already available on arXiv as arXiv:2412.13063
>
> **摘要:** Smartphone-based iris recognition in the visible spectrum (VIS) remains difficult due to illumination variability, pigmentation differences, and the absence of standardized capture controls. This work presents a compact end-to-end pipeline that enforces ISO/IEC 29794-6 quality compliance at acquisition and demonstrates that accurate VIS iris recognition is feasible on commodity devices. Using a custom Android application performing real-time framing, sharpness evaluation, and feedback, we introduce the CUVIRIS dataset of 752 compliant images from 47 subjects. A lightweight MobileNetV3-based multi-task segmentation network (LightIrisNet) is developed for efficient on-device processing, and a transformer matcher (IrisFormer) is adapted to the VIS domain. Under a standardized protocol and comparative benchmarking against prior CNN baselines, OSIRIS attains a TAR of 97.9% at FAR=0.01 (EER=0.76%), while IrisFormer, trained only on UBIRIS.v2, achieves an EER of 0.057% on CUVIRIS. The acquisition app, trained models, and a public subset of the dataset are released to support reproducibility. These results confirm that standardized capture and VIS-adapted lightweight models enable accurate and practical iris recognition on smartphones.
>
---
#### [replaced 144] LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2509.23729v2](http://arxiv.org/pdf/2509.23729v2)**

> **作者:** Shubhang Bhatnagar; Andy Xu; Kar-Han Tan; Narendra Ahuja
>
> **摘要:** Large Language Models (LLMs) with multimodal capabilities have revolutionized vision-language tasks, but their deployment often requires huge memory and computational resources. While post-training quantization (PTQ) has successfully compressed language models to as low as 1-bit precision without significant performance loss, its effectiveness for multimodal LLMs (MLLMs) remains relatively unexplored. In this paper, we present the first study on ultra-low bit (<4-bit) quantization for multimodal LLMs. Our analysis reveals that multimodal tokens and intermediate layer activations produced by them exhibit significantly higher statistical variance and entropy compared to text tokens, making them less tolerant to ultra-low bit quantization. However, the activation distributions of multimodal tokens varies significantly over different layers, with some layers having lower entropy activation distributions. We empirically show that such layers in these models can better tolerate ultra-low bit quantization. Building on these insights, we propose a novel strategy for MLLM quantization, LUQ: Layerwise Ultra-Low Bit Quantization, which selectively applies ultra-low bit quantization to layers that are more resilient to it. Additionally, we also show that using a mix of multimodal tokens (image and text) for PTQ boosts VQA performance in the ultra-low bit regime. We evaluate our method on LLaVA-1.5 and Qwen-2.5-VL across 9 popular VQA benchmarks. The resulting LUQ models use 40% and 31% less memory than their 4-bit counterparts, respectively, while exhibiting a performance degradation of less than 10% on the MME benchmark.
>
---
#### [replaced 145] Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13939v5](http://arxiv.org/pdf/2503.13939v5)**

> **作者:** Yuxiang Lai; Jike Zhong; Ming Li; Shitian Zhao; Yuheng Li; Konstantinos Psounis; Xiaofeng Yang
>
> **摘要:** Vision-language models (VLMs) have achieved impressive progress in natural image reasoning, yet their potential in medical imaging remains underexplored. Medical vision-language tasks demand precise understanding and clinically coherent answers, which are difficult to achieve due to the complexity of medical data and the scarcity of high-quality expert annotations. These challenges limit the effectiveness of conventional supervised fine-tuning (SFT) and Chain-of-Thought (CoT) strategies that work well in general domains. To address these challenges, we propose Med-R1, a reinforcement learning (RL)-enhanced vision-language model designed to improve generalization and reliability in medical reasoning. Built on the DeepSeek strategy, Med-R1 adopts Group Relative Policy Optimization (GRPO) to encourage reward-guided learning beyond static annotations. We comprehensively evaluate Med-R1 across eight distinct medical imaging modalities. Med-R1 achieves a 29.94% improvement in average accuracy over its base model Qwen2-VL-2B, and even outperforms Qwen2-VL-72B-a model with 36x more parameters. To assess cross-task generalization, we further evaluate Med-R1 on five question types. Med-R1 outperforms Qwen2-VL-2B by 32.06% in question-type generalization, also surpassing Qwen2-VL-72B. We further explore the thinking process in Med-R1, a crucial component for the success of Deepseek-R1. Our results show that omitting intermediate rationales (No-Thinking-Med-R1) not only improves in-domain and cross-domain generalization with less training, but also challenges the assumption that more reasoning always helps. These findings suggest that in medical VQA, it is not reasoning itself, but its quality and domain alignment, that determine effectiveness. Together, these results highlight that RL improves medical reasoning and generalization, enabling efficient and reliable VLMs for real-world deployment.
>
---
#### [replaced 146] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.10691v2](http://arxiv.org/pdf/2510.10691v2)**

> **作者:** Xuankai Zhang; Junjin Xiao; Qing Zhang
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** This paper presents a unified framework that allows high-quality dynamic Gaussian Splatting from both defocused and motion-blurred monocular videos. Due to the significant difference between the formation processes of defocus blur and motion blur, existing methods are tailored for either one of them, lacking the ability to simultaneously deal with both of them. Although the two can be jointly modeled as blur kernel-based convolution, the inherent difficulty in estimating accurate blur kernels greatly limits the progress in this direction. In this work, we go a step further towards this direction. Particularly, we propose to estimate per-pixel reliable blur kernels using a blur prediction network that exploits blur-related scene and camera information and is subject to a blur-aware sparsity constraint. Besides, we introduce a dynamic Gaussian densification strategy to mitigate the lack of Gaussians for incomplete regions, and boost the performance of novel view synthesis by incorporating unseen view information to constrain scene optimization. Extensive experiments show that our method outperforms the state-of-the-art methods in generating photorealistic novel view synthesis from defocused and motion-blurred monocular videos. Our code is available at \href{https://github.com/hhhddddddd/dydeblur}{\textcolor{cyan}{https://github.com/hhhddddddd/dydeblur}}.
>
---
#### [replaced 147] Structure-preserving contrastive learning for spatial time series
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06380v5](http://arxiv.org/pdf/2502.06380v5)**

> **作者:** Yiru Jiao; Sander van Cranenburgh; Simeon Calvert; Hans van Lint
>
> **备注:** TL;DR: Preserving certain structures of similarity relations in spatio-temporal data can improve downstream task performance via contrastive learning
>
> **摘要:** The effectiveness of neural network models largely relies on learning meaningful latent patterns from data, where self-supervised learning of informative representations can enhance model performance and generalisability. However, self-supervised representation learning for spatially characterised time series, which are ubiquitous in transportation domain, poses unique challenges due to the necessity of maintaining fine-grained spatio-temporal similarities in the latent space. In this study, we introduce two structure-preserving regularisers for the contrastive learning of spatial time series: one regulariser preserves the topology of similarities between instances, and the other preserves the graph geometry of similarities across spatial and temporal dimensions. To balance the contrastive learning objective and the need for structure preservation, we propose a dynamic weighting mechanism that adaptively manages this trade-off and stabilises training. We validate the proposed method through extensive experiments, including multivariate time series classification to demonstrate its general applicability, as well as macroscopic and microscopic traffic prediction to highlight its particular usefulness in encoding traffic interactions. Across all tasks, our method preserves the similarity structures more effectively and improves state-of-the-art task performances. This method can be integrated with an arbitrary neural network model and is particularly beneficial for time series data with spatial or geographical features. Furthermore, our findings suggest that well-preserved similarity structures in the latent space indicate more informative and useful representations. This provides insights to design more effective neural networks for data-driven transportation research. Our code is made openly accessible with all resulting data at https://github.com/yiru-jiao/spclt
>
---
#### [replaced 148] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13757v2](http://arxiv.org/pdf/2506.13757v2)**

> **作者:** Zewei Zhou; Tianhui Cai; Seth Z. Zhao; Yun Zhang; Zhiyu Huang; Bolei Zhou; Jiaqi Ma
>
> **备注:** NeurIPS 2025; Website link:https://autovla.github.io/
>
> **摘要:** Recent advancements in Vision-Language-Action (VLA) models have shown promise for end-to-end autonomous driving by leveraging world knowledge and reasoning capabilities. However, current VLA models often struggle with physically infeasible action outputs, complex model structures, or unnecessarily long reasoning. In this paper, we propose AutoVLA, a novel VLA model that unifies reasoning and action generation within a single autoregressive generation model for end-to-end autonomous driving. AutoVLA performs semantic reasoning and trajectory planning directly from raw visual inputs and language instructions. We tokenize continuous trajectories into discrete, feasible actions, enabling direct integration into the language model. For training, we employ supervised fine-tuning to equip the model with dual thinking modes: fast thinking (trajectory-only) and slow thinking (enhanced with chain-of-thought reasoning). To further enhance planning performance and efficiency, we introduce a reinforcement fine-tuning method based on Group Relative Policy Optimization (GRPO), reducing unnecessary reasoning in straightforward scenarios. Extensive experiments across real-world and simulated datasets and benchmarks, including nuPlan, nuScenes, Waymo, and CARLA, demonstrate the competitive performance of AutoVLA in both open-loop and closed-loop settings. Qualitative results showcase the adaptive reasoning and accurate planning capabilities of AutoVLA in diverse scenarios.
>
---
#### [replaced 149] Capture, Canonicalize, Splat: Zero-Shot 3D Gaussian Avatars from Unstructured Phone Images
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2510.14081v3](http://arxiv.org/pdf/2510.14081v3)**

> **作者:** Emanuel Garbin; Guy Adam; Oded Krams; Zohar Barzelay; Eran Guendelman; Michael Schwarz; Matteo Presutto; Moran Vatelmacher; Yigal Shenkman; Eli Peker; Itai Druker; Uri Patish; Yoav Blum; Max Bluvstein; Junxuan Li; Rawal Khirodkar; Shunsuke Saito
>
> **备注:** This work received the Best Paper Honorable Mention at the AMFG Workshop, ICCV 2025
>
> **摘要:** We present a novel, zero-shot pipeline for creating hyperrealistic, identity-preserving 3D avatars from a few unstructured phone images. Existing methods face several challenges: single-view approaches suffer from geometric inconsistencies and hallucinations, degrading identity preservation, while models trained on synthetic data fail to capture high-frequency details like skin wrinkles and fine hair, limiting realism. Our method introduces two key contributions: (1) a generative canonicalization module that processes multiple unstructured views into a standardized, consistent representation, and (2) a transformer-based model trained on a new, large-scale dataset of high-fidelity Gaussian splatting avatars derived from dome captures of real people. This "Capture, Canonicalize, Splat" pipeline produces static quarter-body avatars with compelling realism and robust identity preservation from unstructured photos.
>
---
#### [replaced 150] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.20819v2](http://arxiv.org/pdf/2510.20819v2)**

> **作者:** Nimrod Berman; Omkar Joglekar; Eitan Kosman; Dotan Di Castro; Omri Azencot
>
> **备注:** Accepted as a poster at NeurIPS 2025
>
> **摘要:** Recent advances in generative modeling have positioned diffusion models as state-of-the-art tools for sampling from complex data distributions. While these models have shown remarkable success across single-modality domains such as images and audio, extending their capabilities to Modality Translation (MT), translating information across different sensory modalities, remains an open challenge. Existing approaches often rely on restrictive assumptions, including shared dimensionality, Gaussian source priors, and modality-specific architectures, which limit their generality and theoretical grounding. In this work, we propose the Latent Denoising Diffusion Bridge Model (LDDBM), a general-purpose framework for modality translation based on a latent-variable extension of Denoising Diffusion Bridge Models. By operating in a shared latent space, our method learns a bridge between arbitrary modalities without requiring aligned dimensions. We introduce a contrastive alignment loss to enforce semantic consistency between paired samples and design a domain-agnostic encoder-decoder architecture tailored for noise prediction in latent space. Additionally, we propose a predictive loss to guide training toward accurate cross-domain translation and explore several training strategies to improve stability. Our approach supports arbitrary modality pairs and performs strongly on diverse MT tasks, including multi-view to 3D shape generation, image super-resolution, and multi-view scene synthesis. Comprehensive experiments and ablations validate the effectiveness of our framework, establishing a new strong baseline in general modality translation. For more information, see our project page: https://sites.google.com/view/lddbm/home.
>
---
