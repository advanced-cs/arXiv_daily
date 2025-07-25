# 计算机视觉 cs.CV

- **最新发布 136 篇**

- **更新 68 篇**

## 最新发布

#### [new 001] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis
- **分类: cs.CV**

- **简介: 该论文属于文本生成图像任务，旨在解决视觉文本准确渲染问题。提出一种基于分割引导的扩散框架，提升文本内容与风格的一致性。**

- **链接: [http://arxiv.org/pdf/2507.00992v1](http://arxiv.org/pdf/2507.00992v1)**

> **作者:** Yuanrui Wang; Cong Han; YafeiLi; Zhipeng Jin; Xiawei Li; SiNan Du; Wen Tao; Yi Yang; shuanglong li; Chun Yuan; Liu Lin
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Text-to-image generation has greatly advanced content creation, yet accurately rendering visual text remains a key challenge due to blurred glyphs, semantic drift, and limited style control. Existing methods often rely on pre-rendered glyph images as conditions, but these struggle to retain original font styles and color cues, necessitating complex multi-branch designs that increase model overhead and reduce flexibility. To address these issues, we propose a segmentation-guided framework that uses pixel-level visual text masks -- rich in glyph shape, color, and spatial detail -- as unified conditional inputs. Our method introduces two core components: (1) a fine-tuned bilingual segmentation model for precise text mask extraction, and (2) a streamlined diffusion model augmented with adaptive glyph conditioning and a region-specific loss to preserve textual fidelity in both content and style. Our approach achieves state-of-the-art performance on the AnyText benchmark, significantly surpassing prior methods in both Chinese and English settings. To enable more rigorous evaluation, we also introduce two new benchmarks: GlyphMM-benchmark for testing layout and glyph consistency in complex typesetting, and MiniText-benchmark for assessing generation quality in small-scale text regions. Experimental results show that our model outperforms existing methods by a large margin in both scenarios, particularly excelling at small text rendering and complex layout preservation, validating its strong generalization and deployment readiness.
>
---
#### [new 002] PlantSegNeRF: A few-shot, cross-dataset method for plant 3D instance point cloud reconstruction via joint-channel NeRF with multi-view image instance matching
- **分类: cs.CV**

- **简介: 该论文属于植物点云实例分割任务，旨在解决跨数据集、低样本下的高精度3D重建问题。通过结合NeRF与多视角匹配，生成高质量植物实例点云。**

- **链接: [http://arxiv.org/pdf/2507.00371v1](http://arxiv.org/pdf/2507.00371v1)**

> **作者:** Xin Yang; Ruiming Du; Hanyang Huang; Jiayang Xie; Pengyao Xie; Leisen Fang; Ziyue Guo; Nanjun Jiang; Yu Jiang; Haiyan Cen
>
> **摘要:** Organ segmentation of plant point clouds is a prerequisite for the high-resolution and accurate extraction of organ-level phenotypic traits. Although the fast development of deep learning has boosted much research on segmentation of plant point clouds, the existing techniques for organ segmentation still face limitations in resolution, segmentation accuracy, and generalizability across various plant species. In this study, we proposed a novel approach called plant segmentation neural radiance fields (PlantSegNeRF), aiming to directly generate high-precision instance point clouds from multi-view RGB image sequences for a wide range of plant species. PlantSegNeRF performed 2D instance segmentation on the multi-view images to generate instance masks for each organ with a corresponding ID. The multi-view instance IDs corresponding to the same plant organ were then matched and refined using a specially designed instance matching module. The instance NeRF was developed to render an implicit scene, containing color, density, semantic and instance information. The implicit scene was ultimately converted into high-precision plant instance point clouds based on the volume density. The results proved that in semantic segmentation of point clouds, PlantSegNeRF outperformed the commonly used methods, demonstrating an average improvement of 16.1%, 18.3%, 17.8%, and 24.2% in precision, recall, F1-score, and IoU compared to the second-best results on structurally complex datasets. More importantly, PlantSegNeRF exhibited significant advantages in plant point cloud instance segmentation tasks. Across all plant datasets, it achieved average improvements of 11.7%, 38.2%, 32.2% and 25.3% in mPrec, mRec, mCov, mWCov, respectively. This study extends the organ-level plant phenotyping and provides a high-throughput way to supply high-quality 3D data for the development of large-scale models in plant science.
>
---
#### [new 003] OptiPrune: Boosting Prompt-Image Consistency with Attention-Guided Noise and Dynamic Token Selection
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在提升提示与图像的一致性并降低计算成本。提出OptiPrune框架，结合噪声优化与动态令牌选择，实现高效且语义准确的生成。**

- **链接: [http://arxiv.org/pdf/2507.00789v1](http://arxiv.org/pdf/2507.00789v1)**

> **作者:** Ziji Lu
>
> **摘要:** Text-to-image diffusion models often struggle to achieve accurate semantic alignment between generated images and text prompts while maintaining efficiency for deployment on resource-constrained hardware. Existing approaches either incur substantial computational overhead through noise optimization or compromise semantic fidelity by aggressively pruning tokens. In this work, we propose OptiPrune, a unified framework that combines distribution-aware initial noise optimization with similarity-based token pruning to address both challenges simultaneously. Specifically, (1) we introduce a distribution-aware noise optimization module guided by attention scores to steer the initial latent noise toward semantically meaningful regions, mitigating issues such as subject neglect and feature entanglement; (2) we design a hardware-efficient token pruning strategy that selects representative base tokens via patch-wise similarity, injects randomness to enhance generalization, and recovers pruned tokens using maximum similarity copying before attention operations. Our method preserves the Gaussian prior during noise optimization and enables efficient inference without sacrificing alignment quality. Experiments on benchmark datasets, including Animal-Animal, demonstrate that OptiPrune achieves state-of-the-art prompt-image consistency with significantly reduced computational cost.
>
---
#### [new 004] Do Echo Top Heights Improve Deep Learning Nowcasts?
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于降水短时预报任务，旨在提升深度学习模型的预报能力。通过引入回波顶高（ETH）作为辅助输入，研究其对降水强度预测的影响及效果。**

- **链接: [http://arxiv.org/pdf/2507.00845v1](http://arxiv.org/pdf/2507.00845v1)**

> **作者:** Peter Pavlík; Marc Schleiss; Anna Bou Ezzeddine; Viera Rozinajová
>
> **备注:** Pre-review version of an article accepted at Transactions on Large-Scale Data and Knowledge-Centered Systems
>
> **摘要:** Precipitation nowcasting -- the short-term prediction of rainfall using recent radar observations -- is critical for weather-sensitive sectors such as transportation, agriculture, and disaster mitigation. While recent deep learning models have shown promise in improving nowcasting skill, most approaches rely solely on 2D radar reflectivity fields, discarding valuable vertical information available in the full 3D radar volume. In this work, we explore the use of Echo Top Height (ETH), a 2D projection indicating the maximum altitude of radar reflectivity above a given threshold, as an auxiliary input variable for deep learning-based nowcasting. We examine the relationship between ETH and radar reflectivity, confirming its relevance for predicting rainfall intensity. We implement a single-pass 3D U-Net that processes both the radar reflectivity and ETH as separate input channels. While our models are able to leverage ETH to improve skill at low rain-rate thresholds, results are inconsistent at higher intensities and the models with ETH systematically underestimate precipitation intensity. Three case studies are used to illustrate how ETH can help in some cases, but also confuse the models and increase the error variance. Nonetheless, the study serves as a foundation for critically assessing the potential contribution of additional variables to nowcasting performance.
>
---
#### [new 005] AdaDeDup: Adaptive Hybrid Data Pruning for Efficient Large-Scale Object Detection Training
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于目标检测任务，解决大规模数据训练中的计算负担和冗余问题。提出AdaDeDup框架，结合密度与模型反馈实现高效数据剪枝。**

- **链接: [http://arxiv.org/pdf/2507.00049v1](http://arxiv.org/pdf/2507.00049v1)**

> **作者:** Feiyang Kang; Nadine Chang; Maying Shen; Marc T. Law; Rafid Mahmood; Ruoxi Jia; Jose M. Alvarez
>
> **备注:** Preprint
>
> **摘要:** The computational burden and inherent redundancy of large-scale datasets challenge the training of contemporary machine learning models. Data pruning offers a solution by selecting smaller, informative subsets, yet existing methods struggle: density-based approaches can be task-agnostic, while model-based techniques may introduce redundancy or prove computationally prohibitive. We introduce Adaptive De-Duplication (AdaDeDup), a novel hybrid framework that synergistically integrates density-based pruning with model-informed feedback in a cluster-adaptive manner. AdaDeDup first partitions data and applies an initial density-based pruning. It then employs a proxy model to evaluate the impact of this initial pruning within each cluster by comparing losses on kept versus pruned samples. This task-aware signal adaptively adjusts cluster-specific pruning thresholds, enabling more aggressive pruning in redundant clusters while preserving critical data in informative ones. Extensive experiments on large-scale object detection benchmarks (Waymo, COCO, nuScenes) using standard models (BEVFormer, Faster R-CNN) demonstrate AdaDeDup's advantages. It significantly outperforms prominent baselines, substantially reduces performance degradation (e.g., over 54% versus random sampling on Waymo), and achieves near-original model performance while pruning 20% of data, highlighting its efficacy in enhancing data efficiency for large-scale model training. Code is open-sourced.
>
---
#### [new 006] Few-shot Classification as Multi-instance Verification: Effective Backbone-agnostic Transfer across Domains
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于跨域小样本分类任务，解决冻结主干网络下的适应问题。提出MIV-head方法，在不微调主干的情况下实现高效准确的迁移学习。**

- **链接: [http://arxiv.org/pdf/2507.00401v1](http://arxiv.org/pdf/2507.00401v1)**

> **作者:** Xin Xu; Eibe Frank; Geoffrey Holmes
>
> **摘要:** We investigate cross-domain few-shot learning under the constraint that fine-tuning of backbones (i.e., feature extractors) is impossible or infeasible -- a scenario that is increasingly common in practical use cases. Handling the low-quality and static embeddings produced by frozen, "black-box" backbones leads to a problem representation of few-shot classification as a series of multiple instance verification (MIV) tasks. Inspired by this representation, we introduce a novel approach to few-shot domain adaptation, named the "MIV-head", akin to a classification head that is agnostic to any pretrained backbone and computationally efficient. The core components designed for the MIV-head, when trained on few-shot data from a target domain, collectively yield strong performance on test data from that domain. Importantly, it does so without fine-tuning the backbone, and within the "meta-testing" phase. Experimenting under various settings and on an extension of the Meta-dataset benchmark for cross-domain few-shot image classification, using representative off-the-shelf convolutional neural network and vision transformer backbones pretrained on ImageNet1K, we show that the MIV-head achieves highly competitive accuracy when compared to state-of-the-art "adapter" (or partially fine-tuning) methods applied to the same backbones, while incurring substantially lower adaptation cost. We also find well-known "classification head" approaches lag far behind in terms of accuracy. Ablation study empirically justifies the core components of our approach. We share our code at https://github.com/xxweka/MIV-head.
>
---
#### [new 007] Box-QAymo: Box-Referring VQA Dataset for Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Box-QAymo数据集，用于评估和微调视觉语言模型在自动驾驶中的空间时间推理能力，解决用户意图理解与局部查询响应问题。**

- **链接: [http://arxiv.org/pdf/2507.00525v1](http://arxiv.org/pdf/2507.00525v1)**

> **作者:** Djamahl Etchegaray; Yuxia Fu; Zi Huang; Yadan Luo
>
> **摘要:** Interpretable communication is essential for safe and trustworthy autonomous driving, yet current vision-language models (VLMs) often operate under idealized assumptions and struggle to capture user intent in real-world scenarios. Existing driving-oriented VQA datasets are limited to full-scene descriptions or waypoint prediction, preventing the assessment of whether VLMs can respond to localized user-driven queries. We introduce Box-QAymo, a box-referring dataset and benchmark designed to both evaluate and finetune VLMs on spatial and temporal reasoning over user-specified objects. Users express intent by drawing bounding boxes, offering a fast and intuitive interface for focused queries in complex scenes. Specifically, we propose a hierarchical evaluation protocol that begins with binary sanity-check questions to assess basic model capacities, and progresses to (1) attribute prediction for box-referred objects, (2) motion understanding of target instances, and (3) spatiotemporal motion reasoning over inter-object dynamics across frames. To support this, we crowd-sourced fine-grained object classes and visual attributes that reflect the complexity drivers encounter, and extract object trajectories to construct temporally grounded QA pairs. Rigorous quality control through negative sampling, temporal consistency checks, and difficulty-aware balancing guarantee dataset robustness and diversity. Our comprehensive evaluation reveals significant limitations in current VLMs when queried about perception questions, highlighting the gap in achieving real-world performance. This work provides a foundation for developing more robust and interpretable autonomous driving systems that can communicate effectively with users under real-world conditions. Project page and dataset are available at https://djamahl99.github.io/qaymo-pages/.
>
---
#### [new 008] LoD-Loc v2: Aerial Visual Localization over Low Level-of-Detail City Models using Explicit Silhouette Alignment
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，解决低细节度城市模型下的无人机定位问题。提出LoD-Loc v2方法，通过轮廓对齐实现精准定位。**

- **链接: [http://arxiv.org/pdf/2507.00659v1](http://arxiv.org/pdf/2507.00659v1)**

> **作者:** Juelin Zhu; Shuaibang Peng; Long Wang; Hanlin Tan; Yu Liu; Maojun Zhang; Shen Yan
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** We propose a novel method for aerial visual localization over low Level-of-Detail (LoD) city models. Previous wireframe-alignment-based method LoD-Loc has shown promising localization results leveraging LoD models. However, LoD-Loc mainly relies on high-LoD (LoD3 or LoD2) city models, but the majority of available models and those many countries plan to construct nationwide are low-LoD (LoD1). Consequently, enabling localization on low-LoD city models could unlock drones' potential for global urban localization. To address these issues, we introduce LoD-Loc v2, which employs a coarse-to-fine strategy using explicit silhouette alignment to achieve accurate localization over low-LoD city models in the air. Specifically, given a query image, LoD-Loc v2 first applies a building segmentation network to shape building silhouettes. Then, in the coarse pose selection stage, we construct a pose cost volume by uniformly sampling pose hypotheses around a prior pose to represent the pose probability distribution. Each cost of the volume measures the degree of alignment between the projected and predicted silhouettes. We select the pose with maximum value as the coarse pose. In the fine pose estimation stage, a particle filtering method incorporating a multi-beam tracking approach is used to efficiently explore the hypothesis space and obtain the final pose estimation. To further facilitate research in this field, we release two datasets with LoD1 city models covering 10.7 km , along with real RGB queries and ground-truth pose annotations. Experimental results show that LoD-Loc v2 improves estimation accuracy with high-LoD models and enables localization with low-LoD models for the first time. Moreover, it outperforms state-of-the-art baselines by large margins, even surpassing texture-model-based methods, and broadens the convergence basin to accommodate larger prior errors.
>
---
#### [new 009] Just Noticeable Difference for Large Multimodal Models
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于视觉感知研究任务，旨在解决大模型在视觉识别中的感知边界问题。通过提出LMM-JND概念和构建数据集VPA-JND，揭示模型的视觉盲区及性能缺陷。**

- **链接: [http://arxiv.org/pdf/2507.00490v1](http://arxiv.org/pdf/2507.00490v1)**

> **作者:** Zijian Chen; Yuan Tian; Yuze Sun; Wei Sun; Zicheng Zhang; Weisi Lin; Guangtao Zhai; Wenjun Zhang
>
> **备注:** 19 pages, 19 figures
>
> **摘要:** Just noticeable difference (JND), the minimum change that the human visual system (HVS) can perceive, has been studied for decades. Although recent work has extended this line of research into machine vision, there has been a scarcity of studies systematically exploring its perceptual boundaries across multiple tasks and stimulus types, particularly in the current era of rapidly advancing large multimodal models (LMMs), where studying the multifaceted capabilities of models has become a mainstream focus. Moreover, the perceptual defects of LMMs are not investigated thoroughly, resulting in potential security issues and suboptimal response efficiency. In this paper, we take an initial attempt and demonstrate that there exist significant visual blind spots in current LMMs. To systemically quantify this characteristic, we propose a new concept, {\bf LMM-JND}, together with its determination pipeline. Targeting uncovering the behavior commonalities in HVS-aligned visual perception tasks, we delve into several LMM families and construct a large-scale dataset, named VPA-JND, which contains 21.5k reference images with over 489k stimuli across 12 distortion types, to facilitate LMM-JND studies. VPA-JND exposes areas where state-of-the-art LMMs, including GPT-4o and the InternVL2.5 series, struggle with basic comparison queries and fall significantly short of human-level visual performance. We further explore the effects of vision and language backbones and find a notable correlation between their design philosophy that may instruct the future refinement of LMMs for their visual acuity. Together, our research underscores the significance of LMM-JND as a unique perspective for studying LMMs, and predictable LMM-JND is crucial for security concerns. This work will be available at https://github.com/zijianchen98/LMM-JND.
>
---
#### [new 010] Similarity Memory Prior is All You Need for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提升分割精度。提出Sim-MPNet网络，通过相似性记忆先验和动态更新机制，有效提取类别特征。**

- **链接: [http://arxiv.org/pdf/2507.00585v1](http://arxiv.org/pdf/2507.00585v1)**

> **作者:** Tang Hao; Guo ZhiQing; Wang LieJun; Liu Chao
>
> **摘要:** In recent years, it has been found that "grandmother cells" in the primary visual cortex (V1) of macaques can directly recognize visual input with complex shapes. This inspires us to examine the value of these cells in promoting the research of medical image segmentation. In this paper, we design a Similarity Memory Prior Network (Sim-MPNet) for medical image segmentation. Specifically, we propose a Dynamic Memory Weights-Loss Attention (DMW-LA), which matches and remembers the category features of specific lesions or organs in medical images through the similarity memory prior in the prototype memory bank, thus helping the network to learn subtle texture changes between categories. DMW-LA also dynamically updates the similarity memory prior in reverse through Weight-Loss Dynamic (W-LD) update strategy, effectively assisting the network directly extract category features. In addition, we propose the Double-Similarity Global Internal Enhancement Module (DS-GIM) to deeply explore the internal differences in the feature distribution of input data through cosine similarity and euclidean distance. Extensive experiments on four public datasets show that Sim-MPNet has better segmentation performance than other state-of-the-art methods. Our code is available on https://github.com/vpsg-research/Sim-MPNet.
>
---
#### [new 011] Surgical Neural Radiance Fields from One Image
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像重建任务，旨在解决手术中因数据有限无法训练NeRF的问题。通过结合术前MRI与单张术中图像，实现快速高质量的NeRF重建。**

- **链接: [http://arxiv.org/pdf/2507.00969v1](http://arxiv.org/pdf/2507.00969v1)**

> **作者:** Alberto Neri; Maximilan Fehrentz; Veronica Penza; Leonardo S. Mattos; Nazim Haouchine
>
> **摘要:** Purpose: Neural Radiance Fields (NeRF) offer exceptional capabilities for 3D reconstruction and view synthesis, yet their reliance on extensive multi-view data limits their application in surgical intraoperative settings where only limited data is available. In particular, collecting such extensive data intraoperatively is impractical due to time constraints. This work addresses this challenge by leveraging a single intraoperative image and preoperative data to train NeRF efficiently for surgical scenarios. Methods: We leverage preoperative MRI data to define the set of camera viewpoints and images needed for robust and unobstructed training. Intraoperatively, the appearance of the surgical image is transferred to the pre-constructed training set through neural style transfer, specifically combining WTC2 and STROTSS to prevent over-stylization. This process enables the creation of a dataset for instant and fast single-image NeRF training. Results: The method is evaluated with four clinical neurosurgical cases. Quantitative comparisons to NeRF models trained on real surgical microscope images demonstrate strong synthesis agreement, with similarity metrics indicating high reconstruction fidelity and stylistic alignment. When compared with ground truth, our method demonstrates high structural similarity, confirming good reconstruction quality and texture preservation. Conclusion: Our approach demonstrates the feasibility of single-image NeRF training in surgical settings, overcoming the limitations of traditional multi-view methods.
>
---
#### [new 012] Room Scene Discovery and Grouping in Unstructured Vacation Rental Image Collections
- **分类: cs.CV; cs.LG; cs.NE**

- **简介: 该论文属于图像分类与聚类任务，旨在解决度假租赁图片中房间场景识别与分组问题，通过机器学习方法实现高效、准确的图像分组与床型识别。**

- **链接: [http://arxiv.org/pdf/2507.00263v1](http://arxiv.org/pdf/2507.00263v1)**

> **作者:** Vignesh Ram Nithin Kappagantula; Shayan Hassantabar
>
> **摘要:** The rapid growth of vacation rental (VR) platforms has led to an increasing volume of property images, often uploaded without structured categorization. This lack of organization poses significant challenges for travelers attempting to understand the spatial layout of a property, particularly when multiple rooms of the same type are present. To address this issue, we introduce an effective approach for solving the room scene discovery and grouping problem, as well as identifying bed types within each bedroom group. This grouping is valuable for travelers to comprehend the spatial organization, layout, and the sleeping configuration of the property. We propose a computationally efficient machine learning pipeline characterized by low latency and the ability to perform effectively with sample-efficient learning, making it well-suited for real-time and data-scarce environments. The pipeline integrates a supervised room-type detection model, a supervised overlap detection model to identify the overlap similarity between two images, and a clustering algorithm to group the images of the same space together using the similarity scores. Additionally, the pipeline maps each bedroom group to the corresponding bed types specified in the property's metadata, based on the visual content present in the group's images using a Multi-modal Large Language Model (MLLM) model. We evaluate the aforementioned models individually and also assess the pipeline in its entirety, observing strong performance that significantly outperforms established approaches such as contrastive learning and clustering with pretrained embeddings.
>
---
#### [new 013] Self-Supervised Multiview Xray Matching
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多视角X光图像匹配任务，旨在解决不同X光视图间对应关系难建立的问题。通过自监督学习和Transformer方法，自动生成对应矩阵，提升骨折检测效果。**

- **链接: [http://arxiv.org/pdf/2507.00287v1](http://arxiv.org/pdf/2507.00287v1)**

> **作者:** Mohamad Dabboussi; Malo Huard; Yann Gousseau; Pietro Gori
>
> **备注:** MICCAI 2025
>
> **摘要:** Accurate interpretation of multi-view radiographs is crucial for diagnosing fractures, muscular injuries, and other anomalies. While significant advances have been made in AI-based analysis of single images, current methods often struggle to establish robust correspondences between different X-ray views, an essential capability for precise clinical evaluations. In this work, we present a novel self-supervised pipeline that eliminates the need for manual annotation by automatically generating a many-to-many correspondence matrix between synthetic X-ray views. This is achieved using digitally reconstructed radiographs (DRR), which are automatically derived from unannotated CT volumes. Our approach incorporates a transformer-based training phase to accurately predict correspondences across two or more X-ray views. Furthermore, we demonstrate that learning correspondences among synthetic X-ray views can be leveraged as a pretraining strategy to enhance automatic multi-view fracture detection on real data. Extensive evaluations on both synthetic and real X-ray datasets show that incorporating correspondences improves performance in multi-view fracture classification.
>
---
#### [new 014] Improving the Reasoning of Multi-Image Grounding in MLLMs via Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于多图像接地任务，旨在提升MLLMs在复杂多图场景下的推理能力。通过强化学习方法优化模型，显著提高了性能与泛化性。**

- **链接: [http://arxiv.org/pdf/2507.00748v1](http://arxiv.org/pdf/2507.00748v1)**

> **作者:** Bob Zhang; Haoran Li; Tao Zhang; Cilin Yan; Jiayin Cai; Xiaolong Jiang; Yanbin Hao
>
> **备注:** 11 pages
>
> **摘要:** Recently, Multimodal Large Language Models (MLLMs) excel at visual grounding in single-image scenarios with textual references. However, their performance degrades when handling real-world applications involving complex multi-image compositions and multimodal instructions, which reveals limitations in cross-image reasoning and generalization. To address these challenges, we adopt a Reinforcement Learning (RL) based post-training strategy to improve the reasoning performance of MLLMs in multi-image grounding tasks. Our approach begins with synthesizing high-quality chain-of-thought (CoT) data for cold-start initialization, followed by supervised fine-tuning (SFT) using low-rank adaptation (LoRA). The cold-start training stage enables the model to identify correct solutions. Subsequently, we perform rejection sampling using the merged SFT model to curate high-quality RL data and leverage rule-based RL to guide the model toward optimal reasoning paths. Extensive experimental results demonstrate the effectiveness of our approach, achieving +9.04\% improvements on MIG-Bench and +4.98\% improvements on several out-of-domain reasoning grounding benchmarks over the SFT baseline. Furthermore, our approach exhibits strong generalization in multi-image perception, with gains of +3.1\% and +2.4\% over the base model on subsets of the BLINK and MMIU benchmarks, respectively.
>
---
#### [new 015] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement
- **分类: cs.CV**

- **简介: 该论文属于零样本域适应（ZSDA）任务，解决目标域无图像数据时的检测问题。通过联合优化文本提示和视觉表示，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00721v1](http://arxiv.org/pdf/2507.00721v1)**

> **作者:** Xiao Zhang; Fei Wei; Yong Wang; Wenda Zhao; Feiyi Li; Xiangxiang Chu
>
> **备注:** ICCV2025
>
> **摘要:** Zero-shot domain adaptation (ZSDA) presents substantial challenges due to the lack of images in the target domain. Previous approaches leverage Vision-Language Models (VLMs) to tackle this challenge, exploiting their zero-shot learning capabilities. However, these methods primarily address domain distribution shifts and overlook the misalignment between the detection task and VLMs, which rely on manually crafted prompts. To overcome these limitations, we propose the unified prompt and representation enhancement (UPRE) framework, which jointly optimizes both textual prompts and visual representations. Specifically, our approach introduces a multi-view domain prompt that combines linguistic domain priors with detection-specific knowledge, and a visual representation enhancement module that produces domain style variations. Furthermore, we introduce multi-level enhancement strategies, including relative domain distance and positive-negative separation, which align multi-modal representations at the image level and capture diverse visual representations at the instance level, respectively. Extensive experiments conducted on nine benchmark datasets demonstrate the superior performance of our framework in ZSDA detection scenarios. Code is available at https://github.com/AMAP-ML/UPRE.
>
---
#### [new 016] MammoTracker: Mask-Guided Lesion Tracking in Temporal Mammograms
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决乳腺X光片中病灶的自动跟踪问题。提出MammoTracker框架，提升CAD系统的病灶追踪效果。**

- **链接: [http://arxiv.org/pdf/2507.00328v1](http://arxiv.org/pdf/2507.00328v1)**

> **作者:** Xuan Liu; Yinhao Ren; Marc D. Ryser; Lars J. Grimm; Joseph Y. Lo
>
> **摘要:** Accurate lesion tracking in temporal mammograms is essential for monitoring breast cancer progression and facilitating early diagnosis. However, automated lesion correspondence across exams remains a challenges in computer-aided diagnosis (CAD) systems, limiting their effectiveness. We propose MammoTracker, a mask-guided lesion tracking framework that automates lesion localization across consecutively exams. Our approach follows a coarse-to-fine strategy incorporating three key modules: global search, local search, and score refinement. To support large-scale training and evaluation, we introduce a new dataset with curated prior-exam annotations for 730 mass and calcification cases from the public EMBED mammogram dataset, yielding over 20000 lesion pairs, making it the largest known resource for temporal lesion tracking in mammograms. Experimental results demonstrate that MammoTracker achieves 0.455 average overlap and 0.509 accuracy, surpassing baseline models by 8%, highlighting its potential to enhance CAD-based lesion progression analysis. Our dataset will be available at https://gitlab.oit.duke.edu/railabs/LoGroup/mammotracker.
>
---
#### [new 017] SCING:Towards More Efficient and Robust Person Re-Identification through Selective Cross-modal Prompt Tuning
- **分类: cs.CV**

- **简介: 该论文属于行人重识别任务，旨在提升跨模态对齐与鲁棒性。提出SCING框架，通过视觉提示融合和扰动一致性对齐，实现高效准确的ReID。**

- **链接: [http://arxiv.org/pdf/2507.00506v1](http://arxiv.org/pdf/2507.00506v1)**

> **作者:** Yunfei Xie; Yuxuan Cheng; Juncheng Wu; Haoyu Zhang; Yuyin Zhou; Shoudong Han
>
> **摘要:** Recent advancements in adapting vision-language pre-training models like CLIP for person re-identification (ReID) tasks often rely on complex adapter design or modality-specific tuning while neglecting cross-modal interaction, leading to high computational costs or suboptimal alignment. To address these limitations, we propose a simple yet effective framework named Selective Cross-modal Prompt Tuning (SCING) that enhances cross-modal alignment and robustness against real-world perturbations. Our method introduces two key innovations: Firstly, we proposed Selective Visual Prompt Fusion (SVIP), a lightweight module that dynamically injects discriminative visual features into text prompts via a cross-modal gating mechanism. Moreover, the proposed Perturbation-Driven Consistency Alignment (PDCA) is a dual-path training strategy that enforces invariant feature alignment under random image perturbations by regularizing consistency between original and augmented cross-modal embeddings. Extensive experiments are conducted on several popular benchmarks covering Market1501, DukeMTMC-ReID, Occluded-Duke, Occluded-REID, and P-DukeMTMC, which demonstrate the impressive performance of the proposed method. Notably, our framework eliminates heavy adapters while maintaining efficient inference, achieving an optimal trade-off between performance and computational overhead. The code will be released upon acceptance.
>
---
#### [new 018] GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于多模态推理任务，旨在提升模型在多种场景下的综合能力。通过大规模预训练和强化学习，开发出高效且性能优越的视觉语言模型GLM-4.1V-Thinking。**

- **链接: [http://arxiv.org/pdf/2507.01006v1](http://arxiv.org/pdf/2507.01006v1)**

> **作者:** Wenyi Hong; Wenmeng Yu; Xiaotao Gu; Guo Wang; Guobing Gan; Haomiao Tang; Jiale Cheng; Ji Qi; Junhui Ji; Lihang Pan; Shuaiqi Duan; Weihan Wang; Yan Wang; Yean Cheng; Zehai He; Zhe Su; Zhen Yang; Ziyang Pan; Aohan Zeng; Baoxu Wang; Boyan Shi; Changyu Pang; Chenhui Zhang; Da Yin; Fan Yang; Guoqing Chen; Jiazheng Xu; Jiali Chen; Jing Chen; Jinhao Chen; Jinghao Lin; Jinjiang Wang; Junjie Chen; Leqi Lei; Leyi Pan; Mingzhi Zhang; Qinkai Zheng; Sheng Yang; Shi Zhong; Shiyu Huang; Shuyuan Zhao; Siyan Xue; Shangqin Tu; Shengbiao Meng; Tianshu Zhang; Tianwei Luo; Tianxiang Hao; Tianle Gong; Wenkai Li; Wei Jia; Xin Lyu; Xuancheng Huang; Yanling Wang; Yadong Xue; Yanfeng Wang; Yifan An; Yifan Du; Yiming Shi; Yiheng Huang; Yilin Niu; Yuan Wang; Yuanchang Yue; Yuchen Li; Yutao Zhang; Yuxuan Zhang; Zhanxiao Du; Zhenyu Hou; Zhao Xue; Zhengxiao Du; Zihan Wang; Peng Zhang; Debing Liu; Bin Xu; Juanzi Li; Minlie Huang; Yuxiao Dong; Jie Tang
>
> **摘要:** We present GLM-4.1V-Thinking, a vision-language model (VLM) designed to advance general-purpose multimodal reasoning. In this report, we share our key findings in the development of the reasoning-centric training framework. We first develop a capable vision foundation model with significant potential through large-scale pre-training, which arguably sets the upper bound for the final performance. Reinforcement Learning with Curriculum Sampling (RLCS) then unlocks the full potential of the model, leading to comprehensive capability enhancement across a diverse range of tasks, including STEM problem solving, video understanding, content recognition, coding, grounding, GUI-based agents, and long document understanding, among others. To facilitate research in this field, we open-source GLM-4.1V-9B-Thinking, which achieves state-of-the-art performance among models of comparable size. In a comprehensive evaluation across 28 public benchmarks, our model outperforms Qwen2.5-VL-7B on nearly all tasks and achieves comparable or even superior performance on 18 benchmarks relative to the significantly larger Qwen2.5-VL-72B. Notably, GLM-4.1V-9B-Thinking also demonstrates competitive or superior performance compared to closed-source models such as GPT-4o on challenging tasks including long document understanding and STEM reasoning, further underscoring its strong capabilities. Code, models and more information are released at https://github.com/THUDM/GLM-4.1V-Thinking.
>
---
#### [new 019] A Unified Transformer-Based Framework with Pretraining For Whole Body Grasping Motion Generation
- **分类: cs.CV**

- **简介: 该论文属于人体全肢体抓取任务，解决抓取姿态生成与运动补全问题。提出基于Transformer的框架，结合预训练提升性能。**

- **链接: [http://arxiv.org/pdf/2507.00676v1](http://arxiv.org/pdf/2507.00676v1)**

> **作者:** Edward Effendy; Kuan-Wei Tseng; Rei Kawakami
>
> **摘要:** Accepted in the ICIP 2025 We present a novel transformer-based framework for whole-body grasping that addresses both pose generation and motion infilling, enabling realistic and stable object interactions. Our pipeline comprises three stages: Grasp Pose Generation for full-body grasp generation, Temporal Infilling for smooth motion continuity, and a LiftUp Transformer that refines downsampled joints back to high-resolution markers. To overcome the scarcity of hand-object interaction data, we introduce a data-efficient Generalized Pretraining stage on large, diverse motion datasets, yielding robust spatio-temporal representations transferable to grasping tasks. Experiments on the GRAB dataset show that our method outperforms state-of-the-art baselines in terms of coherence, stability, and visual realism. The modular design also supports easy adaptation to other human-motion applications.
>
---
#### [new 020] Not All Attention Heads Are What You Need: Refining CLIP's Image Representation with Attention Ablation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉-语言模型任务，旨在解决CLIP图像编码器中注意力头影响表示质量的问题，通过提出AAT方法优化注意力权重，提升下游任务性能。**

- **链接: [http://arxiv.org/pdf/2507.00537v1](http://arxiv.org/pdf/2507.00537v1)**

> **作者:** Feng Lin; Marco Chen; Haokui Zhang; Xiaotian Yu; Guangming Lu; Rong Xiao
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** This paper studies the role of attention heads in CLIP's image encoder. While CLIP has exhibited robust performance across diverse applications, we hypothesize that certain attention heads negatively affect final representations and that ablating them can improve performance in downstream tasks. To capitalize on this insight, we propose a simple yet effective method, called Attention Ablation Technique (AAT), to suppress the contribution of specific heads by manipulating attention weights. By integrating two alternative strategies tailored for different application scenarios, AAT systematically identifies and ablates detrimental attention heads to enhance representation quality. Experiments demonstrate that AAT consistently improves downstream task performance across various domains, boosting recall rate by up to 11.1% on CLIP-family models for cross-modal retrieval. The results highlight the potential of AAT to effectively refine large-scale vision-language models with virtually no increase in inference cost.
>
---
#### [new 021] VSF-Med:A Vulnerability Scoring Framework for Medical Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医疗视觉语言模型安全评估任务，旨在解决其漏洞检测问题。提出VSF-Med框架，通过攻击模板、视觉扰动和评分体系评估模型风险。**

- **链接: [http://arxiv.org/pdf/2507.00052v1](http://arxiv.org/pdf/2507.00052v1)**

> **作者:** Binesh Sadanandan; Vahid Behzadan
>
> **摘要:** Vision Language Models (VLMs) hold great promise for streamlining labour-intensive medical imaging workflows, yet systematic security evaluations in clinical settings remain scarce. We introduce VSF--Med, an end-to-end vulnerability-scoring framework for medical VLMs that unites three novel components: (i) a rich library of sophisticated text-prompt attack templates targeting emerging threat vectors; (ii) imperceptible visual perturbations calibrated by structural similarity (SSIM) thresholds to preserve clinical realism; and (iii) an eight-dimensional rubric evaluated by two independent judge LLMs, whose raw scores are consolidated via z-score normalization to yield a 0--32 composite risk metric. Built entirely on publicly available datasets and accompanied by open-source code, VSF--Med synthesizes over 30,000 adversarial variants from 5,000 radiology images and enables reproducible benchmarking of any medical VLM with a single command. Our consolidated analysis reports mean z-score shifts of $0.90\sigma$ for persistence-of-attack-effects, $0.74\sigma$ for prompt-injection effectiveness, and $0.63\sigma$ for safety-bypass success across state-of-the-art VLMs. Notably, Llama-3.2-11B-Vision-Instruct exhibits a peak vulnerability increase of $1.29\sigma$ for persistence-of-attack-effects, while GPT-4o shows increases of $0.69\sigma$ for that same vector and $0.28\sigma$ for prompt-injection attacks.
>
---
#### [new 022] Multi-Modal Graph Convolutional Network with Sinusoidal Encoding for Robust Human Action Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人体动作分割任务，旨在解决因噪声导致的过分割问题。通过多模态图卷积网络和正弦编码等方法提升分割精度。**

- **链接: [http://arxiv.org/pdf/2507.00752v1](http://arxiv.org/pdf/2507.00752v1)**

> **作者:** Hao Xing; Kai Zhe Boey; Yuankai Wu; Darius Burschka; Gordon Cheng
>
> **备注:** 7 pages, 4 figures, accepted in IROS25, Hangzhou, China
>
> **摘要:** Accurate temporal segmentation of human actions is critical for intelligent robots in collaborative settings, where a precise understanding of sub-activity labels and their temporal structure is essential. However, the inherent noise in both human pose estimation and object detection often leads to over-segmentation errors, disrupting the coherence of action sequences. To address this, we propose a Multi-Modal Graph Convolutional Network (MMGCN) that integrates low-frame-rate (e.g., 1 fps) visual data with high-frame-rate (e.g., 30 fps) motion data (skeleton and object detections) to mitigate fragmentation. Our framework introduces three key contributions. First, a sinusoidal encoding strategy that maps 3D skeleton coordinates into a continuous sin-cos space to enhance spatial representation robustness. Second, a temporal graph fusion module that aligns multi-modal inputs with differing resolutions via hierarchical feature aggregation, Third, inspired by the smooth transitions inherent to human actions, we design SmoothLabelMix, a data augmentation technique that mixes input sequences and labels to generate synthetic training examples with gradual action transitions, enhancing temporal consistency in predictions and reducing over-segmentation artifacts. Extensive experiments on the Bimanual Actions Dataset, a public benchmark for human-object interaction understanding, demonstrate that our approach outperforms state-of-the-art methods, especially in action segmentation accuracy, achieving F1@10: 94.5% and F1@25: 92.8%.
>
---
#### [new 023] ATSTrack: Enhancing Visual-Language Tracking by Aligning Temporal and Spatial Scales
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言跟踪任务，解决视觉与语言输入在时空尺度上的不匹配问题。通过对语言描述进行细粒度特征调整，并引入跨帧语言信息，提升跟踪效果。**

- **链接: [http://arxiv.org/pdf/2507.00454v1](http://arxiv.org/pdf/2507.00454v1)**

> **作者:** Yihao Zhen; Qiang Wang; Yu Qiao; Liangqiong Qu; Huijie Fan
>
> **摘要:** A main challenge of Visual-Language Tracking (VLT) is the misalignment between visual inputs and language descriptions caused by target movement. Previous trackers have explored many effective feature modification methods to preserve more aligned features. However, an important yet unexplored factor ultimately hinders their capability, which is the inherent differences in the temporal and spatial scale of information between visual and language inputs. To address this issue, we propose a novel visual-language tracker that enhances the effect of feature modification by \textbf{A}ligning \textbf{T}emporal and \textbf{S}patial scale of different input components, named as \textbf{ATSTrack}. Specifically, we decompose each language description into phrases with different attributes based on their temporal and spatial correspondence with visual inputs, and modify their features in a fine-grained manner. Moreover, we introduce a Visual-Language token that comprises modified linguistic information from the previous frame to guide the model to extract visual features that are more relevant to language description, thereby reducing the impact caused by the differences in spatial scale. Experimental results show that our proposed ATSTrack achieves performance comparable to existing methods. Our code will be released.
>
---
#### [new 024] Cage-Based Deformation for Transferable and Undefendable Point Cloud Attack
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于点云对抗攻击任务，旨在解决攻击的可迁移性、不可防御性和真实性问题。提出CageAttack框架，通过结构化变形生成自然的对抗点云。**

- **链接: [http://arxiv.org/pdf/2507.00690v1](http://arxiv.org/pdf/2507.00690v1)**

> **作者:** Keke Tang; Ziyong Du; Weilong Peng; Xiaofei Wang; Peican Zhu; Ligang Liu; Zhihong Tian
>
> **摘要:** Adversarial attacks on point clouds often impose strict geometric constraints to preserve plausibility; however, such constraints inherently limit transferability and undefendability. While deformation offers an alternative, existing unstructured approaches may introduce unnatural distortions, making adversarial point clouds conspicuous and undermining their plausibility. In this paper, we propose CageAttack, a cage-based deformation framework that produces natural adversarial point clouds. It first constructs a cage around the target object, providing a structured basis for smooth, natural-looking deformation. Perturbations are then applied to the cage vertices, which seamlessly propagate to the point cloud, ensuring that the resulting deformations remain intrinsic to the object and preserve plausibility. Extensive experiments on seven 3D deep neural network classifiers across three datasets show that CageAttack achieves a superior balance among transferability, undefendability, and plausibility, outperforming state-of-the-art methods. Codes will be made public upon acceptance.
>
---
#### [new 025] Evolutionary computing-based image segmentation method to detect defects and features in Additive Friction Stir Deposition Process
- **分类: cs.CV; cs.CE**

- **简介: 该论文属于图像分割任务，旨在检测AFSD过程中的缺陷和特征。通过PSO优化阈值，结合多通道可视化，实现材料界面的精确分割与质量评估。**

- **链接: [http://arxiv.org/pdf/2507.00046v1](http://arxiv.org/pdf/2507.00046v1)**

> **作者:** Akshansh Mishra; Eyob Mesele Sefene; Shivraman Thapliyal
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** This work proposes an evolutionary computing-based image segmentation approach for analyzing soundness in Additive Friction Stir Deposition (AFSD) processes. Particle Swarm Optimization (PSO) was employed to determine optimal segmentation thresholds for detecting defects and features in multilayer AFSD builds. The methodology integrates gradient magnitude analysis with distance transforms to create novel attention-weighted visualizations that highlight critical interface regions. Five AFSD samples processed under different conditions were analyzed using multiple visualization techniques i.e. self-attention maps, and multi-channel visualization. These complementary approaches reveal subtle material transition zones and potential defect regions which were not readily observable through conventional imaging. The PSO algorithm automatically identified optimal threshold values (ranging from 156-173) for each sample, enabling precise segmentation of material interfaces. The multi-channel visualization technique effectively combines boundary information (red channel), spatial relationships (green channel), and material density data (blue channel) into cohesive representations that quantify interface quality. The results demonstrate that attention-based analysis successfully identifies regions of incomplete bonding and inhomogeneities in AFSD joints, providing quantitative metrics for process optimization and quality assessment of additively manufactured components.
>
---
#### [new 026] DAM-VSR: Disentanglement of Appearance and Motion for Video Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于视频超分辨率任务，解决视频帧间不一致和细节生成不足的问题。通过分离外观与运动，结合扩散模型与控制网络，提升视频质量。**

- **链接: [http://arxiv.org/pdf/2507.01012v1](http://arxiv.org/pdf/2507.01012v1)**

> **作者:** Zhe Kong; Le Li; Yong Zhang; Feng Gao; Shaoshu Yang; Tao Wang; Kaihao Zhang; Zhuoliang Kang; Xiaoming Wei; Guanying Chen; Wenhan Luo
>
> **备注:** Accepted by ACM SIGGRAPH 2025, Homepage: https://kongzhecn.github.io/projects/dam-vsr/ Github: https://github.com/kongzhecn/DAM-VSR
>
> **摘要:** Real-world video super-resolution (VSR) presents significant challenges due to complex and unpredictable degradations. Although some recent methods utilize image diffusion models for VSR and have shown improved detail generation capabilities, they still struggle to produce temporally consistent frames. We attempt to use Stable Video Diffusion (SVD) combined with ControlNet to address this issue. However, due to the intrinsic image-animation characteristics of SVD, it is challenging to generate fine details using only low-quality videos. To tackle this problem, we propose DAM-VSR, an appearance and motion disentanglement framework for VSR. This framework disentangles VSR into appearance enhancement and motion control problems. Specifically, appearance enhancement is achieved through reference image super-resolution, while motion control is achieved through video ControlNet. This disentanglement fully leverages the generative prior of video diffusion models and the detail generation capabilities of image super-resolution models. Furthermore, equipped with the proposed motion-aligned bidirectional sampling strategy, DAM-VSR can conduct VSR on longer input videos. DAM-VSR achieves state-of-the-art performance on real-world data and AIGC data, demonstrating its powerful detail generation capabilities.
>
---
#### [new 027] Zero-shot Skeleton-based Action Recognition with Prototype-guided Feature Alignment
- **分类: cs.CV**

- **简介: 该论文属于零样本骨架动作识别任务，解决未知动作分类难题。提出PGFA方法，通过原型引导特征对齐提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00566v1](http://arxiv.org/pdf/2507.00566v1)**

> **作者:** Kai Zhou; Shuhai Zhang; Zeng You; Jinwu Hu; Mingkui Tan; Fei Liu
>
> **备注:** This paper is accepted by IEEE TIP 2025. Code is publicly available at https://github.com/kaai520/PGFA
>
> **摘要:** Zero-shot skeleton-based action recognition aims to classify unseen skeleton-based human actions without prior exposure to such categories during training. This task is extremely challenging due to the difficulty in generalizing from known to unknown actions. Previous studies typically use two-stage training: pre-training skeleton encoders on seen action categories using cross-entropy loss and then aligning pre-extracted skeleton and text features, enabling knowledge transfer to unseen classes through skeleton-text alignment and language models' generalization. However, their efficacy is hindered by 1) insufficient discrimination for skeleton features, as the fixed skeleton encoder fails to capture necessary alignment information for effective skeleton-text alignment; 2) the neglect of alignment bias between skeleton and unseen text features during testing. To this end, we propose a prototype-guided feature alignment paradigm for zero-shot skeleton-based action recognition, termed PGFA. Specifically, we develop an end-to-end cross-modal contrastive training framework to improve skeleton-text alignment, ensuring sufficient discrimination for skeleton features. Additionally, we introduce a prototype-guided text feature alignment strategy to mitigate the adverse impact of the distribution discrepancy during testing. We provide a theoretical analysis to support our prototype-guided text feature alignment strategy and empirically evaluate our overall PGFA on three well-known datasets. Compared with the top competitor SMIE method, our PGFA achieves absolute accuracy improvements of 22.96%, 12.53%, and 18.54% on the NTU-60, NTU-120, and PKU-MMD datasets, respectively.
>
---
#### [new 028] CaughtCheating: Is Your MLLM a Good Cheating Detective? Exploring the Boundary of Visual Perception and Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出"CaughtCheating"任务，研究MLLM在视觉推理中的漏洞，探索其检测细微线索的能力，旨在提升模型的人类级侦探推理水平。**

- **链接: [http://arxiv.org/pdf/2507.00045v1](http://arxiv.org/pdf/2507.00045v1)**

> **作者:** Ming Li; Chenguang Wang; Yijun Liang; Xiyao Wang; Yuhang Zhou; Xiyang Wu; Yuqing Zhang; Ruiyi Zhang; Tianyi Zhou
>
> **摘要:** Recent agentic Multi-Modal Large Language Models (MLLMs) such as GPT-o3 have achieved near-ceiling scores on various existing benchmarks, motivating a demand for more challenging test tasks. These MLLMs have been reported to excel in a few expert-level tasks for humans, e.g., GeoGuesser, reflecting their potential as a detective who can notice minuscule cues in an image and weave them into coherent, situational explanations, leading to a reliable answer. But can they match the performance of excellent human detectives? To answer this question, we investigate some hard scenarios where GPT-o3 can still handle, and find a common scenario where o3's performance drops to nearly zero, which we name CaughtCheating. It is inspired by the social media requests that ask others to detect suspicious clues from photos shared by the poster's partner. We conduct extensive experiments and analysis to understand why existing MLLMs lack sufficient capability to solve this kind of task. CaughtCheating provides a class of challenging visual perception and reasoning tasks with great value and practical usage. Success in these tasks paves the way for MLLMs to acquire human-level detective perception and reasoning capabilities.
>
---
#### [new 029] SelvaBox: A high-resolution dataset for tropical tree crown detection
- **分类: cs.CV; I.2.10; I.4.8; I.5.4**

- **简介: 该论文属于热带树木冠层检测任务，旨在解决数据稀缺问题。作者构建了SelvaBox数据集，包含83,000个标注样本，提升了检测精度与模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00170v1](http://arxiv.org/pdf/2507.00170v1)**

> **作者:** Hugo Baudchon; Arthur Ouaknine; Martin Weiss; Mélisande Teng; Thomas R. Walla; Antoine Caron-Guay; Christopher Pal; Etienne Laliberté
>
> **摘要:** Detecting individual tree crowns in tropical forests is essential to study these complex and crucial ecosystems impacted by human interventions and climate change. However, tropical crowns vary widely in size, structure, and pattern and are largely overlapping and intertwined, requiring advanced remote sensing methods applied to high-resolution imagery. Despite growing interest in tropical tree crown detection, annotated datasets remain scarce, hindering robust model development. We introduce SelvaBox, the largest open-access dataset for tropical tree crown detection in high-resolution drone imagery. It spans three countries and contains more than 83,000 manually labeled crowns - an order of magnitude larger than all previous tropical forest datasets combined. Extensive benchmarks on SelvaBox reveal two key findings: (1) higher-resolution inputs consistently boost detection accuracy; and (2) models trained exclusively on SelvaBox achieve competitive zero-shot detection performance on unseen tropical tree crown datasets, matching or exceeding competing methods. Furthermore, jointly training on SelvaBox and three other datasets at resolutions from 3 to 10 cm per pixel within a unified multi-resolution pipeline yields a detector ranking first or second across all evaluated datasets. Our dataset, code, and pre-trained weights are made public.
>
---
#### [new 030] ShapeEmbed: a self-supervised learning framework for 2D contour quantification
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于形状量化任务，旨在解决形状描述符对几何变换不变性的问题。提出ShapeEmbed框架，通过自监督学习生成不变的形状描述符。**

- **链接: [http://arxiv.org/pdf/2507.01009v1](http://arxiv.org/pdf/2507.01009v1)**

> **作者:** Anna Foix Romero; Craig Russell; Alexander Krull; Virginie Uhlmann
>
> **摘要:** The shape of objects is an important source of visual information in a wide range of applications. One of the core challenges of shape quantification is to ensure that the extracted measurements remain invariant to transformations that preserve an object's intrinsic geometry, such as changing its size, orientation, and position in the image. In this work, we introduce ShapeEmbed, a self-supervised representation learning framework designed to encode the contour of objects in 2D images, represented as a Euclidean distance matrix, into a shape descriptor that is invariant to translation, scaling, rotation, reflection, and point indexing. Our approach overcomes the limitations of traditional shape descriptors while improving upon existing state-of-the-art autoencoder-based approaches. We demonstrate that the descriptors learned by our framework outperform their competitors in shape classification tasks on natural and biological images. We envision our approach to be of particular relevance to biological imaging applications.
>
---
#### [new 031] LLaVA-SP: Enhancing Visual Representation with Visual Spatial Tokens for MLLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在提升MLLM的视觉表示能力。针对CLIP-ViT局部关系建模不足的问题，提出LLaVA-SP，通过添加少量空间视觉标记增强视觉理解。**

- **链接: [http://arxiv.org/pdf/2507.00505v1](http://arxiv.org/pdf/2507.00505v1)**

> **作者:** Haoran Lou; Chunxiao Fan; Ziyan Liu; Yuexin Wu; Xinxiang Wang
>
> **备注:** ICCV
>
> **摘要:** The architecture of multimodal large language models (MLLMs) commonly connects a vision encoder, often based on CLIP-ViT, to a large language model. While CLIP-ViT works well for capturing global image features, it struggles to model local relationships between adjacent patches, leading to weaker visual representation, which in turn affects the detailed understanding ability of MLLMs. To solve this, we propose LLaVA-SP, which \textbf{ only adds six spatial visual tokens} to the original visual tokens to enhance the visual representation. Our approach offers three key advantages: 1)We propose a novel Projector, which uses convolutional kernels to derive visual spatial tokens from ViT patch features, simulating two visual spatial ordering approaches: ``from central region to global" and ``from abstract to specific". Then, a cross-attention mechanism is applied to fuse fine-grained visual information, enriching the overall visual representation. 2) We present two model variants: LLaVA-SP-Cropping, which focuses on detail features through progressive cropping, and LLaVA-SP-Pooling, which captures global semantics through adaptive pooling, enabling the model to handle diverse visual understanding tasks. 3) Extensive experiments show that LLaVA-SP, fine-tuned with LoRA, achieves significant performance improvements across various multimodal benchmarks, outperforming the state-of-the-art LLaVA-1.5 model in multiple tasks with nearly identical inference latency. The code and models are available at \href{https://github.com/CnFaker/LLaVA-SP}{\texttt{https://github.com/CnFaker/LLaVA-SP}}.
>
---
#### [new 032] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决无需感知标注的端到端规划问题。通过构建物理潜在世界模型，实现自监督学习下的轨迹生成与选择。**

- **链接: [http://arxiv.org/pdf/2507.00603v1](http://arxiv.org/pdf/2507.00603v1)**

> **作者:** Yupeng Zheng; Pengxuan Yang; Zebin Xing; Qichao Zhang; Yuhang Zheng; Yinfeng Gao; Pengfei Li; Teng Zhang; Zhongpu Xia; Peng Jia; Dongbin Zhao
>
> **备注:** ICCV 2025, first version
>
> **摘要:** End-to-end autonomous driving directly generates planning trajectories from raw sensor data, yet it typically relies on costly perception supervision to extract scene information. A critical research challenge arises: constructing an informative driving world model to enable perception annotation-free, end-to-end planning via self-supervised learning. In this paper, we present World4Drive, an end-to-end autonomous driving framework that employs vision foundation models to build latent world models for generating and evaluating multi-modal planning trajectories. Specifically, World4Drive first extracts scene features, including driving intention and world latent representations enriched with spatial-semantic priors provided by vision foundation models. It then generates multi-modal planning trajectories based on current scene features and driving intentions and predicts multiple intention-driven future states within the latent space. Finally, it introduces a world model selector module to evaluate and select the best trajectory. We achieve perception annotation-free, end-to-end planning through self-supervised alignment between actual future observations and predicted observations reconstructed from the latent space. World4Drive achieves state-of-the-art performance without manual perception annotations on both the open-loop nuScenes and closed-loop NavSim benchmarks, demonstrating an 18.1\% relative reduction in L2 error, 46.7% lower collision rate, and 3.75 faster training convergence. Codes will be accessed at https://github.com/ucaszyp/World4Drive.
>
---
#### [new 033] Robust Component Detection for Flexible Manufacturing: A Deep Learning Approach to Tray-Free Object Recognition under Variable Lighting
- **分类: cs.CV**

- **简介: 该论文属于工业视觉任务，解决无固定托盘下物体检测与抓取问题，采用深度学习方法提升光照变化下的识别鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.00852v1](http://arxiv.org/pdf/2507.00852v1)**

> **作者:** Fatemeh Sadat Daneshmand
>
> **摘要:** Flexible manufacturing systems in Industry 4.0 require robots capable of handling objects in unstructured environments without rigid positioning constraints. This paper presents a computer vision system that enables industrial robots to detect and grasp pen components in arbitrary orientations without requiring structured trays, while maintaining robust performance under varying lighting conditions. We implement and evaluate a Mask R-CNN-based approach on a complete pen manufacturing line at ZHAW, addressing three critical challenges: object detection without positional constraints, robustness to extreme lighting variations, and reliable performance with cost-effective cameras. Our system achieves 95% detection accuracy across diverse lighting conditions while eliminating the need for structured component placement, demonstrating a 30% reduction in setup time and significant improvement in manufacturing flexibility. The approach is validated through extensive testing under four distinct lighting scenarios, showing practical applicability for real-world industrial deployment.
>
---
#### [new 034] Out-of-distribution detection in 3D applications: a review
- **分类: cs.CV**

- **简介: 该论文属于3D应用中的分布外检测任务，旨在解决模型对训练中未见对象识别不足的问题，综述了方法、数据集及评估指标。**

- **链接: [http://arxiv.org/pdf/2507.00570v1](http://arxiv.org/pdf/2507.00570v1)**

> **作者:** Zizhao Li; Xueyang Kang; Joseph West; Kourosh Khoshelham
>
> **摘要:** The ability to detect objects that are not prevalent in the training set is a critical capability in many 3D applications, including autonomous driving. Machine learning methods for object recognition often assume that all object categories encountered during inference belong to a closed set of classes present in the training data. This assumption limits generalization to the real world, as objects not seen during training may be misclassified or entirely ignored. As part of reliable AI, OOD detection identifies inputs that deviate significantly from the training distribution. This paper provides a comprehensive overview of OOD detection within the broader scope of trustworthy and uncertain AI. We begin with key use cases across diverse domains, introduce benchmark datasets spanning multiple modalities, and discuss evaluation metrics. Next, we present a comparative analysis of OOD detection methodologies, exploring model structures, uncertainty indicators, and distributional distance taxonomies, alongside uncertainty calibration techniques. Finally, we highlight promising research directions, including adversarially robust OOD detection and failure identification, particularly relevant to 3D applications. The paper offers both theoretical and practical insights into OOD detection, showcasing emerging research opportunities such as 3D vision integration. These insights help new researchers navigate the field more effectively, contributing to the development of reliable, safe, and robust AI systems.
>
---
#### [new 035] Evaluating Robustness of Monocular Depth Estimation with Procedural Scene Perturbations
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在评估模型的鲁棒性。通过引入PDE基准，系统性地测试不同扰动下的性能，以补充传统准确率评估的不足。**

- **链接: [http://arxiv.org/pdf/2507.00981v1](http://arxiv.org/pdf/2507.00981v1)**

> **作者:** Jack Nugent; Siyang Wu; Zeyu Ma; Beining Han; Meenal Parakh; Abhishek Joshi; Lingjie Mei; Alexander Raistrick; Xinyuan Li; Jia Deng
>
> **摘要:** Recent years have witnessed substantial progress on monocular depth estimation, particularly as measured by the success of large models on standard benchmarks. However, performance on standard benchmarks does not offer a complete assessment, because most evaluate accuracy but not robustness. In this work, we introduce PDE (Procedural Depth Evaluation), a new benchmark which enables systematic robustness evaluation. PDE uses procedural generation to create 3D scenes that test robustness to various controlled perturbations, including object, camera, material and lighting changes. Our analysis yields interesting findings on what perturbations are challenging for state-of-the-art depth models, which we hope will inform further research. Code and data are available at https://github.com/princeton-vl/proc-depth-eval.
>
---
#### [new 036] Diffusion-Based Image Augmentation for Semantic Segmentation in Outdoor Robotics
- **分类: cs.CV**

- **简介: 该论文属于图像增强任务，旨在提升户外机器人在极端环境下的语义分割性能。通过扩散模型生成更贴近真实场景的训练数据，解决环境分布不匹配问题。**

- **链接: [http://arxiv.org/pdf/2507.00153v1](http://arxiv.org/pdf/2507.00153v1)**

> **作者:** Peter Mortimer; Mirko Maehlisch
>
> **备注:** Presented at the 2025 IEEE ICRA Workshop on Field Robotics
>
> **摘要:** The performance of leaning-based perception algorithms suffer when deployed in out-of-distribution and underrepresented environments. Outdoor robots are particularly susceptible to rapid changes in visual scene appearance due to dynamic lighting, seasonality and weather effects that lead to scenes underrepresented in the training data of the learning-based perception system. In this conceptual paper, we focus on preparing our autonomous vehicle for deployment in snow-filled environments. We propose a novel method for diffusion-based image augmentation to more closely represent the deployment environment in our training data. Diffusion-based image augmentations rely on the public availability of vision foundation models learned on internet-scale datasets. The diffusion-based image augmentations allow us to take control over the semantic distribution of the ground surfaces in the training data and to fine-tune our model for its deployment environment. We employ open vocabulary semantic segmentation models to filter out augmentation candidates that contain hallucinations. We believe that diffusion-based image augmentations can be extended to many other environments apart from snow surfaces, like sandy environments and volcanic terrains.
>
---
#### [new 037] MANTA: Cross-Modal Semantic Alignment and Information-Theoretic Optimization for Long-form Multimodal Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出MANTA框架，解决多模态理解中的语义对齐与信息优化问题，提升长视频问答等任务性能。**

- **链接: [http://arxiv.org/pdf/2507.00068v1](http://arxiv.org/pdf/2507.00068v1)**

> **作者:** Ziqi Zhong; Daniel Tang
>
> **摘要:** While multi-modal learning has advanced significantly, current approaches often treat modalities separately, creating inconsistencies in representation and reasoning. We introduce MANTA (Multi-modal Abstraction and Normalization via Textual Alignment), a theoretically-grounded framework that unifies visual and auditory inputs into a structured textual space for seamless processing with large language models. MANTA addresses four key challenges: (1) semantic alignment across modalities with information-theoretic optimization, (2) adaptive temporal synchronization for varying information densities, (3) hierarchical content representation for multi-scale understanding, and (4) context-aware retrieval of sparse information from long sequences. We formalize our approach within a rigorous mathematical framework, proving its optimality for context selection under token constraints. Extensive experiments on the challenging task of Long Video Question Answering show that MANTA improves state-of-the-art models by up to 22.6% in overall accuracy, with particularly significant gains (27.3%) on videos exceeding 30 minutes. Additionally, we demonstrate MANTA's superiority on temporal reasoning tasks (23.8% improvement) and cross-modal understanding (25.1% improvement). Our framework introduces novel density estimation techniques for redundancy minimization while preserving rare signals, establishing new foundations for unifying multimodal representations through structured text.
>
---
#### [new 038] Graph-Based Deep Learning for Component Segmentation of Maize Plants
- **分类: cs.CV**

- **简介: 该论文属于作物成分分割任务，旨在解决3D点云数据中植物组件识别的问题。提出基于图神经网络的方法，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2507.00182v1](http://arxiv.org/pdf/2507.00182v1)**

> **作者:** J. I. Ruíz; A. Méndez; E. Rodríguez
>
> **摘要:** In precision agriculture, one of the most important tasks when exploring crop production is identifying individual plant components. There are several attempts to accomplish this task by the use of traditional 2D imaging, 3D reconstructions, and Convolutional Neural Networks (CNN). However, they have several drawbacks when processing 3D data and identifying individual plant components. Therefore, in this work, we propose a novel Deep Learning architecture to detect components of individual plants on Light Detection and Ranging (LiDAR) 3D Point Cloud (PC) data sets. This architecture is based on the concept of Graph Neural Networks (GNN), and feature enhancing with Principal Component Analysis (PCA). For this, each point is taken as a vertex and by the use of a K-Nearest Neighbors (KNN) layer, the edges are established, thus representing the 3D PC data set. Subsequently, Edge-Conv layers are used to further increase the features of each point. Finally, Graph Attention Networks (GAT) are applied to classify visible phenotypic components of the plant, such as the leaf, stem, and soil. This study demonstrates that our graph-based deep learning approach enhances segmentation accuracy for identifying individual plant components, achieving percentages above 80% in the IoU average, thus outperforming other existing models based on point clouds.
>
---
#### [new 039] Masks make discriminative models great again!
- **分类: cs.CV**

- **简介: 该论文属于单图像3D重建任务，旨在解决从单张图像重建逼真3D场景的问题。通过分离可见区域提升与不可见区域补全，提升重建质量。**

- **链接: [http://arxiv.org/pdf/2507.00916v1](http://arxiv.org/pdf/2507.00916v1)**

> **作者:** Tianshi Cao; Marie-Julie Rakotosaona; Ben Poole; Federico Tombari; Michael Niemeyer
>
> **摘要:** We present Image2GS, a novel approach that addresses the challenging problem of reconstructing photorealistic 3D scenes from a single image by focusing specifically on the image-to-3D lifting component of the reconstruction process. By decoupling the lifting problem (converting an image to a 3D model representing what is visible) from the completion problem (hallucinating content not present in the input), we create a more deterministic task suitable for discriminative models. Our method employs visibility masks derived from optimized 3D Gaussian splats to exclude areas not visible from the source view during training. This masked training strategy significantly improves reconstruction quality in visible regions compared to strong baselines. Notably, despite being trained only on masked regions, Image2GS remains competitive with state-of-the-art discriminative models trained on full target images when evaluated on complete scenes. Our findings highlight the fundamental struggle discriminative models face when fitting unseen regions and demonstrate the advantages of addressing image-to-3D lifting as a distinct problem with specialized techniques.
>
---
#### [new 040] Visual Anagrams Reveal Hidden Differences in Holistic Shape Processing Across Vision Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在解决视觉模型对形状与纹理依赖的问题。通过引入CSS评估模型的全局配置能力，揭示模型在形状处理上的差异。**

- **链接: [http://arxiv.org/pdf/2507.00493v1](http://arxiv.org/pdf/2507.00493v1)**

> **作者:** Fenil R. Doshi; Thomas Fel; Talia Konkle; George Alvarez
>
> **备注:** Project page: https://www.fenildoshi.com/configural-shape/
>
> **摘要:** Humans are able to recognize objects based on both local texture cues and the configuration of object parts, yet contemporary vision models primarily harvest local texture cues, yielding brittle, non-compositional features. Work on shape-vs-texture bias has pitted shape and texture representations in opposition, measuring shape relative to texture, ignoring the possibility that models (and humans) can simultaneously rely on both types of cues, and obscuring the absolute quality of both types of representation. We therefore recast shape evaluation as a matter of absolute configural competence, operationalized by the Configural Shape Score (CSS), which (i) measures the ability to recognize both images in Object-Anagram pairs that preserve local texture while permuting global part arrangement to depict different object categories. Across 86 convolutional, transformer, and hybrid models, CSS (ii) uncovers a broad spectrum of configural sensitivity with fully self-supervised and language-aligned transformers -- exemplified by DINOv2, SigLIP2 and EVA-CLIP -- occupying the top end of the CSS spectrum. Mechanistic probes reveal that (iii) high-CSS networks depend on long-range interactions: radius-controlled attention masks abolish performance showing a distinctive U-shaped integration profile, and representational-similarity analyses expose a mid-depth transition from local to global coding. A BagNet control remains at chance (iv), ruling out "border-hacking" strategies. Finally, (v) we show that configural shape score also predicts other shape-dependent evals. Overall, we propose that the path toward truly robust, generalizable, and human-like vision systems may not lie in forcing an artificial choice between shape and texture, but rather in architectural and learning frameworks that seamlessly integrate both local-texture and global configural shape.
>
---
#### [new 041] Overtake Detection in Trucks Using CAN Bus Signals: A Comparative Study of Machine Learning Methods
- **分类: cs.CV**

- **简介: 该论文属于车辆行为识别任务，旨在通过CAN总线数据检测卡车超车行为。研究比较了多种机器学习方法，分析了数据预处理对性能的影响，并提出融合策略提升准确率。**

- **链接: [http://arxiv.org/pdf/2507.00593v1](http://arxiv.org/pdf/2507.00593v1)**

> **作者:** Fernando Alonso-Fernandez; Talha Hanif Butt; Prayag Tiwari
>
> **备注:** Under review at ESWA
>
> **摘要:** Safe overtaking manoeuvres in trucks are vital for preventing accidents and ensuring efficient traffic flow. Accurate prediction of such manoeuvres is essential for Advanced Driver Assistance Systems (ADAS) to make timely and informed decisions. In this study, we focus on overtake detection using Controller Area Network (CAN) bus data collected from five in-service trucks provided by the Volvo Group. We evaluate three common classifiers for vehicle manoeuvre detection, Artificial Neural Networks (ANN), Random Forest (RF), and Support Vector Machines (SVM), and analyse how different preprocessing configurations affect performance. We find that variability in traffic conditions strongly influences the signal patterns, particularly in the no-overtake class, affecting classification performance if training data lacks adequate diversity. Since the data were collected under unconstrained, real-world conditions, class diversity cannot be guaranteed a priori. However, training with data from multiple vehicles improves generalisation and reduces condition-specific bias. Our pertruck analysis also reveals that classification accuracy, especially for overtakes, depends on the amount of training data per vehicle. To address this, we apply a score-level fusion strategy, which yields the best per-truck performance across most cases. Overall, we achieve an accuracy via fusion of TNR=93% (True Negative Rate) and TPR=86.5% (True Positive Rate). This research has been part of the BIG FUN project, which explores how Artificial Intelligence can be applied to logged vehicle data to understand and predict driver behaviour, particularly in relation to Camera Monitor Systems (CMS), being introduced as digital replacements for traditional exterior mirrors.
>
---
#### [new 042] Learning Dense Feature Matching via Lifting Single 2D Image to 3D Space
- **分类: cs.CV**

- **简介: 该论文属于特征匹配任务，旨在解决多视角数据依赖和3D感知不足的问题。提出L2M框架，通过将2D图像提升至3D空间实现更鲁棒的特征匹配。**

- **链接: [http://arxiv.org/pdf/2507.00392v1](http://arxiv.org/pdf/2507.00392v1)**

> **作者:** Yingping Liang; Yutao Hu; Wenqi Shao; Ying Fu
>
> **摘要:** Feature matching plays a fundamental role in many computer vision tasks, yet existing methods heavily rely on scarce and clean multi-view image collections, which constrains their generalization to diverse and challenging scenarios. Moreover, conventional feature encoders are typically trained on single-view 2D images, limiting their capacity to capture 3D-aware correspondences. In this paper, we propose a novel two-stage framework that lifts 2D images to 3D space, named as \textbf{Lift to Match (L2M)}, taking full advantage of large-scale and diverse single-view images. To be specific, in the first stage, we learn a 3D-aware feature encoder using a combination of multi-view image synthesis and 3D feature Gaussian representation, which injects 3D geometry knowledge into the encoder. In the second stage, a novel-view rendering strategy, combined with large-scale synthetic data generation from single-view images, is employed to learn a feature decoder for robust feature matching, thus achieving generalization across diverse domains. Extensive experiments demonstrate that our method achieves superior generalization across zero-shot evaluation benchmarks, highlighting the effectiveness of the proposed framework for robust feature matching.
>
---
#### [new 043] Out-of-Distribution Detection with Adaptive Top-K Logits Integration
- **分类: cs.CV**

- **简介: 该论文属于OOD检测任务，旨在解决神经网络对分布外样本过度自信的问题。提出ATLI方法，通过自适应整合top-k logits提升检测效果。**

- **链接: [http://arxiv.org/pdf/2507.00368v1](http://arxiv.org/pdf/2507.00368v1)**

> **作者:** Hikaru Shijo; Yutaka Yoshihama; Kenichi Yadani; Norifumi Murata
>
> **摘要:** Neural networks often make overconfident predictions from out-of-distribution (OOD) samples. Detection of OOD data is therefore crucial to improve the safety of machine learning. The simplest and most powerful method for OOD detection is MaxLogit, which uses the model's maximum logit to provide an OOD score. We have discovered that, in addition to the maximum logit, some other logits are also useful for OOD detection. Based on this finding, we propose a new method called ATLI (Adaptive Top-k Logits Integration), which adaptively determines effective top-k logits that are specific to each model and combines the maximum logit with the other top-k logits. In this study we evaluate our proposed method using ImageNet-1K benchmark. Extensive experiments showed our proposed method to reduce the false positive rate (FPR95) by 6.73% compared to the MaxLogit approach, and decreased FPR95 by an additional 2.67% compared to other state-of-the-art methods.
>
---
#### [new 044] Populate-A-Scene: Affordance-Aware Human Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在让模型理解并模拟人与环境的交互。通过单张场景图预测人物行为与位置，实现自然的人景融合。**

- **链接: [http://arxiv.org/pdf/2507.00334v1](http://arxiv.org/pdf/2507.00334v1)**

> **作者:** Mengyi Shan; Zecheng He; Haoyu Ma; Felix Juefei-Xu; Peizhao Zhang; Tingbo Hou; Ching-Yao Chuang
>
> **备注:** Project page: https://shanmy.github.io/Populate-A-Scene
>
> **摘要:** Can a video generation model be repurposed as an interactive world simulator? We explore the affordance perception potential of text-to-video models by teaching them to predict human-environment interaction. Given a scene image and a prompt describing human actions, we fine-tune the model to insert a person into the scene, while ensuring coherent behavior, appearance, harmonization, and scene affordance. Unlike prior work, we infer human affordance for video generation (i.e., where to insert a person and how they should behave) from a single scene image, without explicit conditions like bounding boxes or body poses. An in-depth study of cross-attention heatmaps demonstrates that we can uncover the inherent affordance perception of a pre-trained video model without labeled affordance datasets.
>
---
#### [new 045] ARIG: Autoregressive Interactive Head Generation for Real-time Conversations
- **分类: cs.CV**

- **简介: 该论文属于实时交互头部生成任务，旨在解决传统方法在实时性和交互真实度上的不足。提出ARIG框架，通过自回归和扩散过程实现更准确的运动预测。**

- **链接: [http://arxiv.org/pdf/2507.00472v1](http://arxiv.org/pdf/2507.00472v1)**

> **作者:** Ying Guo; Xi Liu; Cheng Zhen; Pengfei Yan; Xiaoming Wei
>
> **备注:** ICCV 2025. Homepage: https://jinyugy21.github.io/ARIG/
>
> **摘要:** Face-to-face communication, as a common human activity, motivates the research on interactive head generation. A virtual agent can generate motion responses with both listening and speaking capabilities based on the audio or motion signals of the other user and itself. However, previous clip-wise generation paradigm or explicit listener/speaker generator-switching methods have limitations in future signal acquisition, contextual behavioral understanding, and switching smoothness, making it challenging to be real-time and realistic. In this paper, we propose an autoregressive (AR) based frame-wise framework called ARIG to realize the real-time generation with better interaction realism. To achieve real-time generation, we model motion prediction as a non-vector-quantized AR process. Unlike discrete codebook-index prediction, we represent motion distribution using diffusion procedure, achieving more accurate predictions in continuous space. To improve interaction realism, we emphasize interactive behavior understanding (IBU) and detailed conversational state understanding (CSU). In IBU, based on dual-track dual-modal signals, we summarize short-range behaviors through bidirectional-integrated learning and perform contextual understanding over long ranges. In CSU, we use voice activity signals and context features of IBU to understand the various states (interruption, feedback, pause, etc.) that exist in actual conversations. These serve as conditions for the final progressive motion prediction. Extensive experiments have verified the effectiveness of our model.
>
---
#### [new 046] MR-CLIP: Efficient Metadata-Guided Learning of MRI Contrast Representations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决MRI对比度识别问题。通过MR-CLIP框架，利用DICOM元数据学习对比度感知表示，提升图像检索与分类效果。**

- **链接: [http://arxiv.org/pdf/2507.00043v1](http://arxiv.org/pdf/2507.00043v1)**

> **作者:** Mehmet Yigit Avci; Pedro Borges; Paul Wright; Mehmet Yigitsoy; Sebastien Ourselin; Jorge Cardoso
>
> **摘要:** Accurate interpretation of Magnetic Resonance Imaging scans in clinical systems is based on a precise understanding of image contrast. This contrast is primarily governed by acquisition parameters, such as echo time and repetition time, which are stored in the DICOM metadata. To simplify contrast identification, broad labels such as T1-weighted or T2-weighted are commonly used, but these offer only a coarse approximation of the underlying acquisition settings. In many real-world datasets, such labels are entirely missing, leaving raw acquisition parameters as the only indicators of contrast. Adding to this challenge, the available metadata is often incomplete, noisy, or inconsistent. The lack of reliable and standardized metadata complicates tasks such as image interpretation, retrieval, and integration into clinical workflows. Furthermore, robust contrast-aware representations are essential to enable more advanced clinical applications, such as achieving modality-invariant representations and data harmonization. To address these challenges, we propose MR-CLIP, a multimodal contrastive learning framework that aligns MR images with their DICOM metadata to learn contrast-aware representations, without relying on manual labels. Trained on a diverse clinical dataset that spans various scanners and protocols, MR-CLIP captures contrast variations across acquisitions and within scans, enabling anatomy-invariant representations. We demonstrate its effectiveness in cross-modal retrieval and contrast classification, highlighting its scalability and potential for further clinical applications. The code and weights are publicly available at https://github.com/myigitavci/MR-CLIP.
>
---
#### [new 047] Laplace-Mamba: Laplace Frequency Prior-Guided Mamba-CNN Fusion Network for Image Dehazing
- **分类: cs.CV**

- **简介: 该论文属于图像去雾任务，旨在解决SSM在处理局部结构和高维数据时的不足。通过融合Laplace频率先验与Mamba-CNN架构，提升去雾效果与效率。**

- **链接: [http://arxiv.org/pdf/2507.00501v1](http://arxiv.org/pdf/2507.00501v1)**

> **作者:** Yongzhen Wang; Liangliang Chen; Bingwen Hu; Heng Liu; Xiao-Ping Zhang; Mingqiang Wei
>
> **备注:** 12 pages, 11 figures, 6 tables
>
> **摘要:** Recent progress in image restoration has underscored Spatial State Models (SSMs) as powerful tools for modeling long-range dependencies, owing to their appealing linear complexity and computational efficiency. However, SSM-based approaches exhibit limitations in reconstructing localized structures and tend to be less effective when handling high-dimensional data, frequently resulting in suboptimal recovery of fine image features. To tackle these challenges, we introduce Laplace-Mamba, a novel framework that integrates Laplace frequency prior with a hybrid Mamba-CNN architecture for efficient image dehazing. Leveraging the Laplace decomposition, the image is disentangled into low-frequency components capturing global texture and high-frequency components representing edges and fine details. This decomposition enables specialized processing via dual parallel pathways: the low-frequency branch employs SSMs for global context modeling, while the high-frequency branch utilizes CNNs to refine local structural details, effectively addressing diverse haze scenarios. Notably, the Laplace transformation facilitates information-preserving downsampling of low-frequency components in accordance with the Nyquist theory, thereby significantly improving computational efficiency. Extensive evaluations across multiple benchmarks demonstrate that our method outperforms state-of-the-art approaches in both restoration quality and efficiency. The source code and pretrained models are available at https://github.com/yz-wang/Laplace-Mamba.
>
---
#### [new 048] DiGA3D: Coarse-to-Fine Diffusional Propagation of Geometry and Appearance for Versatile 3D Inpainting
- **分类: cs.CV**

- **简介: 该论文属于3D图像修复任务，旨在解决多视角下外观与几何不一致的问题。提出DiGA3D方法，通过扩散模型实现从粗到细的外观和几何传播。**

- **链接: [http://arxiv.org/pdf/2507.00429v1](http://arxiv.org/pdf/2507.00429v1)**

> **作者:** Jingyi Pan; Dan Xu; Qiong Luo
>
> **备注:** ICCV 2025, Project page: https://rorisis.github.io/DiGA3D/
>
> **摘要:** Developing a unified pipeline that enables users to remove, re-texture, or replace objects in a versatile manner is crucial for text-guided 3D inpainting. However, there are still challenges in performing multiple 3D inpainting tasks within a unified framework: 1) Single reference inpainting methods lack robustness when dealing with views that are far from the reference view. 2) Appearance inconsistency arises when independently inpainting multi-view images with 2D diffusion priors; 3) Geometry inconsistency limits performance when there are significant geometric changes in the inpainting regions. To tackle these challenges, we introduce DiGA3D, a novel and versatile 3D inpainting pipeline that leverages diffusion models to propagate consistent appearance and geometry in a coarse-to-fine manner. First, DiGA3D develops a robust strategy for selecting multiple reference views to reduce errors during propagation. Next, DiGA3D designs an Attention Feature Propagation (AFP) mechanism that propagates attention features from the selected reference views to other views via diffusion models to maintain appearance consistency. Furthermore, DiGA3D introduces a Texture-Geometry Score Distillation Sampling (TG-SDS) loss to further improve the geometric consistency of inpainted 3D scenes. Extensive experiments on multiple 3D inpainting tasks demonstrate the effectiveness of our method. The project page is available at https://rorisis.github.io/DiGA3D/.
>
---
#### [new 049] GazeTarget360: Towards Gaze Target Estimation in 360-Degree for Robot Perception
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于机器人视觉任务，旨在解决360度场景下 gaze target 估计问题。提出 GazeTarget360 系统，结合多种模型实现准确预测。**

- **链接: [http://arxiv.org/pdf/2507.00253v1](http://arxiv.org/pdf/2507.00253v1)**

> **作者:** Zhuangzhuang Dai; Vincent Gbouna Zakka; Luis J. Manso; Chen Li
>
> **摘要:** Enabling robots to understand human gaze target is a crucial step to allow capabilities in downstream tasks, for example, attention estimation and movement anticipation in real-world human-robot interactions. Prior works have addressed the in-frame target localization problem with data-driven approaches by carefully removing out-of-frame samples. Vision-based gaze estimation methods, such as OpenFace, do not effectively absorb background information in images and cannot predict gaze target in situations where subjects look away from the camera. In this work, we propose a system to address the problem of 360-degree gaze target estimation from an image in generalized visual scenes. The system, named GazeTarget360, integrates conditional inference engines of an eye-contact detector, a pre-trained vision encoder, and a multi-scale-fusion decoder. Cross validation results show that GazeTarget360 can produce accurate and reliable gaze target predictions in unseen scenarios. This makes a first-of-its-kind system to predict gaze targets from realistic camera footage which is highly efficient and deployable. Our source code is made publicly available at: https://github.com/zdai257/DisengageNet.
>
---
#### [new 050] Unleashing the Potential of All Test Samples: Mean-Shift Guided Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的测试时适应任务，解决模型在分布偏移下的性能下降问题。提出MS-TTA方法，通过均值漂移提升特征表示，增强分类稳定性。**

- **链接: [http://arxiv.org/pdf/2507.00462v1](http://arxiv.org/pdf/2507.00462v1)**

> **作者:** Jizhou Han; Chenhao Ding; SongLin Dong; Yuhang He; Xinyuan Gao; Yihong Gong
>
> **摘要:** Visual-language models (VLMs) like CLIP exhibit strong generalization but struggle with distribution shifts at test time. Existing training-free test-time adaptation (TTA) methods operate strictly within CLIP's original feature space, relying on high-confidence samples while overlooking the potential of low-confidence ones. We propose MS-TTA, a training-free approach that enhances feature representations beyond CLIP's space using a single-step k-nearest neighbors (kNN) Mean-Shift. By refining all test samples, MS-TTA improves feature compactness and class separability, leading to more stable adaptation. Additionally, a cache of refined embeddings further enhances inference by providing Mean Shift enhanced logits. Extensive evaluations on OOD and cross-dataset benchmarks demonstrate that MS-TTA consistently outperforms state-of-the-art training-free TTA methods, achieving robust adaptation without requiring additional training.
>
---
#### [new 051] Holmes: Towards Effective and Harmless Model Ownership Verification to Personalized Large Vision Models via Decoupling Common Features
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于模型所有权验证任务，旨在解决个性化大视觉模型被窃取的问题。通过解耦公共特征，提出一种有效且无害的验证方法。**

- **链接: [http://arxiv.org/pdf/2507.00724v1](http://arxiv.org/pdf/2507.00724v1)**

> **作者:** Linghui Zhu; Yiming Li; Haiqin Weng; Yan Liu; Tianwei Zhang; Shu-Tao Xia; Zhi Wang
>
> **摘要:** Large vision models achieve remarkable performance in various downstream tasks, primarily by personalizing pre-trained models through fine-tuning with private and valuable local data, which makes the personalized model a valuable intellectual property for its owner. Similar to the era of traditional DNNs, model stealing attacks also pose significant risks to these personalized models. However, in this paper, we reveal that most existing defense methods (developed for traditional DNNs), typically designed for models trained from scratch, either introduce additional security risks, are prone to misjudgment, or are even ineffective for fine-tuned models. To alleviate these problems, this paper proposes a harmless model ownership verification method for personalized models by decoupling similar common features. In general, our method consists of three main stages. In the first stage, we create shadow models that retain common features of the victim model while disrupting dataset-specific features. We represent the dataset-specific features of the victim model by the output differences between the shadow and victim models. After that, a meta-classifier is trained to identify stolen models by determining whether suspicious models contain the dataset-specific features of the victim. In the third stage, we conduct model ownership verification by hypothesis test to mitigate randomness and enhance robustness. Extensive experiments on benchmark datasets verify the effectiveness of the proposed method in detecting different types of model stealing simultaneously.
>
---
#### [new 052] MFH: Marrying Frequency Domain with Handwritten Mathematical Expression Recognition
- **分类: cs.CV**

- **简介: 该论文属于手写数学表达式识别任务，旨在解决复杂结构和布局带来的识别难题。通过引入频域分析方法MFH，提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2507.00430v1](http://arxiv.org/pdf/2507.00430v1)**

> **作者:** Huanxin Yang; Qiwen Wang
>
> **摘要:** Handwritten mathematical expression recognition (HMER) suffers from complex formula structures and character layouts in sequence prediction. In this paper, we incorporate frequency domain analysis into HMER and propose a method that marries frequency domain with HMER (MFH), leveraging the discrete cosine transform (DCT). We emphasize the structural analysis assistance of frequency information for recognizing mathematical formulas. When implemented on various baseline models, our network exhibits a consistent performance enhancement, demonstrating the efficacy of frequency domain information. Experiments show that our MFH-CoMER achieves noteworthy accuracyrates of 61.66%/62.07%/63.72% on the CROHME 2014/2016/2019 test sets. The source code is available at https://github.com/Hryxyhe/MFH.
>
---
#### [new 053] CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于对抗攻击任务，针对视频多模态大语言模型（V-MLLMs）的脆弱性进行研究，提出CAVALRY-V框架，通过双目标损失和高效生成结构提升攻击效果。**

- **链接: [http://arxiv.org/pdf/2507.00817v1](http://arxiv.org/pdf/2507.00817v1)**

> **作者:** Jiaming Zhang; Rui Hu; Qing Guo; Wei Yang Bryan Lim
>
> **摘要:** Video Multimodal Large Language Models (V-MLLMs) have shown impressive capabilities in temporal reasoning and cross-modal understanding, yet their vulnerability to adversarial attacks remains underexplored due to unique challenges: complex cross-modal reasoning mechanisms, temporal dependencies, and computational constraints. We present CAVALRY-V (Cross-modal Language-Vision Adversarial Yielding for Videos), a novel framework that directly targets the critical interface between visual perception and language generation in V-MLLMs. Our approach introduces two key innovations: (1) a dual-objective semantic-visual loss function that simultaneously disrupts the model's text generation logits and visual representations to undermine cross-modal integration, and (2) a computationally efficient two-stage generator framework that combines large-scale pre-training for cross-model transferability with specialized fine-tuning for spatiotemporal coherence. Empirical evaluation on comprehensive video understanding benchmarks demonstrates that CAVALRY-V significantly outperforms existing attack methods, achieving 22.8% average improvement over the best baseline attacks on both commercial systems (GPT-4.1, Gemini 2.0) and open-source models (QwenVL-2.5, InternVL-2.5, Llava-Video, Aria, MiniCPM-o-2.6). Our framework achieves flexibility through implicit temporal coherence modeling rather than explicit regularization, enabling significant performance improvements even on image understanding (34.4% average gain). This capability demonstrates CAVALRY-V's potential as a foundational approach for adversarial research across multimodal systems.
>
---
#### [new 054] Biorthogonal Tunable Wavelet Unit with Lifting Scheme in Convolutional Neural Network
- **分类: cs.CV; eess.IV; eess.SP**

- **简介: 该论文属于图像分类与异常检测任务，旨在提升CNN性能。通过引入可调双正交小波单元，优化卷积与下采样操作，提高分类与检测精度。**

- **链接: [http://arxiv.org/pdf/2507.00739v1](http://arxiv.org/pdf/2507.00739v1)**

> **作者:** An Le; Hung Nguyen; Sungbal Seo; You-Suk Bae; Truong Nguyen
>
> **摘要:** This work introduces a novel biorthogonal tunable wavelet unit constructed using a lifting scheme that relaxes both the orthogonality and equal filter length constraints, providing greater flexibility in filter design. The proposed unit enhances convolution, pooling, and downsampling operations, leading to improved image classification and anomaly detection in convolutional neural networks (CNN). When integrated into an 18-layer residual neural network (ResNet-18), the approach improved classification accuracy on CIFAR-10 by 2.12% and on the Describable Textures Dataset (DTD) by 9.73%, demonstrating its effectiveness in capturing fine-grained details. Similar improvements were observed in ResNet-34. For anomaly detection in the hazelnut category of the MVTec Anomaly Detection dataset, the proposed method achieved competitive and wellbalanced performance in both segmentation and detection tasks, outperforming existing approaches in terms of accuracy and robustness.
>
---
#### [new 055] VOCAL: Visual Odometry via ContrAstive Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉里程计任务，旨在解决传统方法依赖几何假设的问题。通过对比学习与贝叶斯推理，构建可解释的相机状态表示。**

- **链接: [http://arxiv.org/pdf/2507.00243v1](http://arxiv.org/pdf/2507.00243v1)**

> **作者:** Chi-Yao Huang; Zeel Bhatt; Yezhou Yang
>
> **摘要:** Breakthroughs in visual odometry (VO) have fundamentally reshaped the landscape of robotics, enabling ultra-precise camera state estimation that is crucial for modern autonomous systems. Despite these advances, many learning-based VO techniques rely on rigid geometric assumptions, which often fall short in interpretability and lack a solid theoretical basis within fully data-driven frameworks. To overcome these limitations, we introduce VOCAL (Visual Odometry via ContrAstive Learning), a novel framework that reimagines VO as a label ranking challenge. By integrating Bayesian inference with a representation learning framework, VOCAL organizes visual features to mirror camera states. The ranking mechanism compels similar camera states to converge into consistent and spatially coherent representations within the latent space. This strategic alignment not only bolsters the interpretability of the learned features but also ensures compatibility with multimodal data sources. Extensive evaluations on the KITTI dataset highlight VOCAL's enhanced interpretability and flexibility, pushing VO toward more general and explainable spatial intelligence.
>
---
#### [new 056] De-Simplifying Pseudo Labels to Enhancing Domain Adaptive Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决无监督域适应中的自标签方法性能不足问题。通过提出DeSimPL方法减少简单样本比例，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2507.00608v1](http://arxiv.org/pdf/2507.00608v1)**

> **作者:** Zehua Fu; Chenguang Liu; Yuyu Chen; Jiaqi Zhou; Qingjie Liu; Yunhong Wang
>
> **备注:** Accepted by IEEE Transactions on Intelligent Transportation Systems. 15 pages, 10 figures
>
> **摘要:** Despite its significant success, object detection in traffic and transportation scenarios requires time-consuming and laborious efforts in acquiring high-quality labeled data. Therefore, Unsupervised Domain Adaptation (UDA) for object detection has recently gained increasing research attention. UDA for object detection has been dominated by domain alignment methods, which achieve top performance. Recently, self-labeling methods have gained popularity due to their simplicity and efficiency. In this paper, we investigate the limitations that prevent self-labeling detectors from achieving commensurate performance with domain alignment methods. Specifically, we identify the high proportion of simple samples during training, i.e., the simple-label bias, as the central cause. We propose a novel approach called De-Simplifying Pseudo Labels (DeSimPL) to mitigate the issue. DeSimPL utilizes an instance-level memory bank to implement an innovative pseudo label updating strategy. Then, adversarial samples are introduced during training to enhance the proportion. Furthermore, we propose an adaptive weighted loss to avoid the model suffering from an abundance of false positive pseudo labels in the late training period. Experimental results demonstrate that DeSimPL effectively reduces the proportion of simple samples during training, leading to a significant performance improvement for self-labeling detectors. Extensive experiments conducted on four benchmarks validate our analysis and conclusions.
>
---
#### [new 057] Language-Unlocked ViT (LUViT): Empowering Self-Supervised Vision Transformers with LLMs
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言融合任务，旨在解决LLM与ViT的模态不匹配问题。通过联合预训练策略，使ViT和LLM协同优化，提升视觉理解性能。**

- **链接: [http://arxiv.org/pdf/2507.00754v1](http://arxiv.org/pdf/2507.00754v1)**

> **作者:** Selim Kuzucu; Muhammad Ferjad Naeem; Anna Kukleva; Federico Tombari; Bernt Schiele
>
> **备注:** 26 pages, 6 figures
>
> **摘要:** The integration of Large Language Model (LLMs) blocks with Vision Transformers (ViTs) holds immense promise for vision-only tasks by leveraging the rich semantic knowledge and reasoning capabilities of LLMs. However, a fundamental challenge lies in the inherent modality mismatch between text-centric pretraining of LLMs and vision-centric training of ViTs. Direct fusion often fails to fully exploit the LLM's potential and suffers from unstable finetuning. As a result, LLM blocks are kept frozen while only the vision components are learned. As a remedy to these challenges, we introduce Language-Unlocked Vision Transformers (LUViT), a novel approach that bridges this modality mismatch through a synergistic pre-training strategy. LUViT co-adapts a ViT backbone and an LLM fusion block by (1) employing Masked Auto-Encoding (MAE) to pre-train the ViT for richer visual representations, and (2) concurrently training Low-Rank Adaptation (LoRA) layers within the LLM block using the MAE objective. This joint optimization guides the ViT to produce LLM-aligned features and the LLM to effectively interpret visual information. We demonstrate through extensive experiments that LUViT significantly improves performance on various downstream vision tasks, showcasing a more effective and efficient pathway to harness LLM knowledge for visual understanding.
>
---
#### [new 058] Developing Lightweight DNN Models With Limited Data For Real-Time Sign Language Recognition
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于实时手语识别任务，解决数据稀缺、计算成本高和帧率不一致问题。通过轻量DNN和MediaPipe实现高效准确识别。**

- **链接: [http://arxiv.org/pdf/2507.00248v1](http://arxiv.org/pdf/2507.00248v1)**

> **作者:** Nikita Nikitin; Eugene Fomin
>
> **备注:** 7 pages, 2 figures, 2 tables, for associated mpeg file, see https://slait.app/static/Screen_Recording.mp4
>
> **摘要:** We present a novel framework for real-time sign language recognition using lightweight DNNs trained on limited data. Our system addresses key challenges in sign language recognition, including data scarcity, high computational costs, and discrepancies in frame rates between training and inference environments. By encoding sign language specific parameters, such as handshape, palm orientation, movement, and location into vectorized inputs, and leveraging MediaPipe for landmark extraction, we achieve highly separable input data representations. Our DNN architecture, optimized for sub 10MB deployment, enables accurate classification of 343 signs with less than 10ms latency on edge devices. The data annotation platform 'slait data' facilitates structured labeling and vector extraction. Our model achieved 92% accuracy in isolated sign recognition and has been integrated into the 'slait ai' web application, where it demonstrates stable inference.
>
---
#### [new 059] Training for X-Ray Vision: Amodal Segmentation, Amodal Content Completion, and View-Invariant Object Representation from Multi-Camera Video
- **分类: cs.CV; cs.AI; 68T45, 68T07; I.2.10; I.2.6; I.4.6**

- **简介: 该论文属于计算机视觉领域，解决多视角下物体的遮挡分割与内容补全问题，提出MOVi-MC-AC数据集以支持相关研究。**

- **链接: [http://arxiv.org/pdf/2507.00339v1](http://arxiv.org/pdf/2507.00339v1)**

> **作者:** Alexander Moore; Amar Saini; Kylie Cancilla; Doug Poland; Carmen Carrano
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Amodal segmentation and amodal content completion require using object priors to estimate occluded masks and features of objects in complex scenes. Until now, no data has provided an additional dimension for object context: the possibility of multiple cameras sharing a view of a scene. We introduce MOVi-MC-AC: Multiple Object Video with Multi-Cameras and Amodal Content, the largest amodal segmentation and first amodal content dataset to date. Cluttered scenes of generic household objects are simulated in multi-camera video. MOVi-MC-AC contributes to the growing literature of object detection, tracking, and segmentation by including two new contributions to the deep learning for computer vision world. Multiple Camera (MC) settings where objects can be identified and tracked between various unique camera perspectives are rare in both synthetic and real-world video. We introduce a new complexity to synthetic video by providing consistent object ids for detections and segmentations between both frames and multiple cameras each with unique features and motion patterns on a single scene. Amodal Content (AC) is a reconstructive task in which models predict the appearance of target objects through occlusions. In the amodal segmentation literature, some datasets have been released with amodal detection, tracking, and segmentation labels. While other methods rely on slow cut-and-paste schemes to generate amodal content pseudo-labels, they do not account for natural occlusions present in the modal masks. MOVi-MC-AC provides labels for ~5.8 million object instances, setting a new maximum in the amodal dataset literature, along with being the first to provide ground-truth amodal content. The full dataset is available at https://huggingface.co/datasets/Amar-S/MOVi-MC-AC ,
>
---
#### [new 060] Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes
- **分类: cs.CV**

- **简介: 该论文属于模型微调任务，解决低数据和领域差异大场景下的适应性问题。提出SR-LoRA框架，利用稳定秩指导秩分配，提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.00327v1](http://arxiv.org/pdf/2507.00327v1)**

> **作者:** Chuyan Zhang; Kefan Wang; Yun Gu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Low-Rank Adaptation (LoRA) has proven effective in reducing computational costs while maintaining performance comparable to fully fine-tuned foundation models across various tasks. However, its fixed low-rank structure restricts its adaptability in scenarios with substantial domain gaps, where higher ranks are often required to capture domain-specific complexities. Current adaptive LoRA methods attempt to overcome this limitation by dynamically expanding or selectively allocating ranks, but these approaches frequently depend on computationally intensive techniques such as iterative pruning, rank searches, or additional regularization. To address these challenges, we introduce Stable Rank-Guided Low-Rank Adaptation (SR-LoRA), a novel framework that utilizes the stable rank of pre-trained weight matrices as a natural prior for layer-wise rank allocation. By leveraging the stable rank, which reflects the intrinsic dimensionality of the weights, SR-LoRA enables a principled and efficient redistribution of ranks across layers, enhancing adaptability without incurring additional search costs. Empirical evaluations on few-shot tasks with significant domain gaps show that SR-LoRA consistently outperforms recent adaptive LoRA variants, achieving a superior trade-off between performance and efficiency. Our code is available at https://github.com/EndoluminalSurgicalVision-IMR/SR-LoRA.
>
---
#### [new 061] Moment Sampling in Video LLMs for Long-Form Video QA
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频问答任务，旨在解决长视频中关键帧丢失和冗余问题。通过引入时刻采样方法，提升视频大模型的长文本问答性能。**

- **链接: [http://arxiv.org/pdf/2507.00033v1](http://arxiv.org/pdf/2507.00033v1)**

> **作者:** Mustafa Chasmai; Gauri Jagatap; Gouthaman KV; Grant Van Horn; Subhransu Maji; Andrea Fanelli
>
> **备注:** Workshop on Video Large Language Models (VidLLMs) at CVPR 2025
>
> **摘要:** Recent advancements in video large language models (Video LLMs) have significantly advanced the field of video question answering (VideoQA). While existing methods perform well on short videos, they often struggle with long-range reasoning in longer videos. To scale Video LLMs for longer video content, frame sub-sampling (selecting frames at regular intervals) is commonly used. However, this approach is suboptimal, often leading to the loss of crucial frames or the inclusion of redundant information from multiple similar frames. Missing key frames impairs the model's ability to answer questions accurately, while redundant frames lead the model to focus on irrelevant video segments and increase computational resource consumption. In this paper, we investigate the use of a general-purpose text-to-video moment retrieval model to guide the frame sampling process. We propose "moment sampling", a novel, model-agnostic approach that enables the model to select the most relevant frames according to the context of the question. Specifically, we employ a lightweight moment retrieval model to prioritize frame selection. By focusing on the frames most pertinent to the given question, our method enhances long-form VideoQA performance in Video LLMs. Through extensive experiments on four long-form VideoQA datasets, using four state-of-the-art Video LLMs, we demonstrate the effectiveness of the proposed approach.
>
---
#### [new 062] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型任务，旨在解决模型幻觉问题。提出ONLY方法，通过单次查询和一层干预有效减少幻觉，提升可靠性与效率。**

- **链接: [http://arxiv.org/pdf/2507.00898v1](http://arxiv.org/pdf/2507.00898v1)**

> **作者:** Zifu Wan; Ce Zhang; Silong Yong; Martin Q. Ma; Simon Stepputtis; Louis-Philippe Morency; Deva Ramanan; Katia Sycara; Yaqi Xie
>
> **备注:** Accepted by ICCV 2025. Project page: https://zifuwan.github.io/ONLY/
>
> **摘要:** Recent Large Vision-Language Models (LVLMs) have introduced a new paradigm for understanding and reasoning about image input through textual responses. Although they have achieved remarkable performance across a range of multi-modal tasks, they face the persistent challenge of hallucination, which introduces practical weaknesses and raises concerns about their reliable deployment in real-world applications. Existing work has explored contrastive decoding approaches to mitigate this issue, where the output of the original LVLM is compared and contrasted with that of a perturbed version. However, these methods require two or more queries that slow down LVLM response generation, making them less suitable for real-time applications. To overcome this limitation, we propose ONLY, a training-free decoding approach that requires only a single query and a one-layer intervention during decoding, enabling efficient real-time deployment. Specifically, we enhance textual outputs by selectively amplifying crucial textual information using a text-to-visual entropy ratio for each token. Extensive experimental results demonstrate that our proposed ONLY consistently outperforms state-of-the-art methods across various benchmarks while requiring minimal implementation effort and computational cost. Code is available at https://github.com/zifuwan/ONLY.
>
---
#### [new 063] Catastrophic Forgetting Mitigation via Discrepancy-Weighted Experience Replay
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于云边协同目标检测任务，解决动态交通环境中模型因新数据分布导致的灾难性遗忘问题。提出ER-EMU算法，通过差异加权经验回放提升知识保留与适应能力。**

- **链接: [http://arxiv.org/pdf/2507.00042v1](http://arxiv.org/pdf/2507.00042v1)**

> **作者:** Xinrun Xu; Jianwen Yang; Qiuhong Zhang; Zhanbiao Lian; Zhiming Ding; Shan Jiang
>
> **备注:** ICANN 2025
>
> **摘要:** Continually adapting edge models in cloud-edge collaborative object detection for traffic monitoring suffers from catastrophic forgetting, where models lose previously learned knowledge when adapting to new data distributions. This is especially problematic in dynamic traffic environments characterised by periodic variations (e.g., day/night, peak hours), where past knowledge remains valuable. Existing approaches like experience replay and visual prompts offer some mitigation, but struggle to effectively prioritize and leverage historical data for optimal knowledge retention and adaptation. Specifically, simply storing and replaying all historical data can be inefficient, while treating all historical experiences as equally important overlooks their varying relevance to the current domain. This paper proposes ER-EMU, an edge model update algorithm based on adaptive experience replay, to address these limitations. ER-EMU utilizes a limited-size experience buffer managed using a First-In-First-Out (FIFO) principle, and a novel Domain Distance Metric-based Experience Selection (DDM-ES) algorithm. DDM-ES employs the multi-kernel maximum mean discrepancy (MK-MMD) to quantify the dissimilarity between target domains, prioritizing the selection of historical data that is most dissimilar to the current target domain. This ensures training diversity and facilitates the retention of knowledge from a wider range of past experiences, while also preventing overfitting to the new domain. The experience buffer is also updated using a simple random sampling strategy to maintain a balanced representation of previous domains. Experiments on the Bellevue traffic video dataset, involving repeated day/night cycles, demonstrate that ER-EMU consistently improves the performance of several state-of-the-art cloud-edge collaborative object detection frameworks.
>
---
#### [new 064] Reducing Variability of Multiple Instance Learning Methods for Digital Pathology
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于数字病理学中的WSI分类任务，旨在解决MIL方法性能波动大的问题。通过模型融合策略降低变异性，提升可重复性与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.00292v1](http://arxiv.org/pdf/2507.00292v1)**

> **作者:** Ali Mammadov; Loïc Le Folgoc; Guillaume Hocquet; Pietro Gori
>
> **备注:** MICCAI 2025
>
> **摘要:** Digital pathology has revolutionized the field by enabling the digitization of tissue samples into whole slide images (WSIs). However, the high resolution and large size of WSIs present significant challenges when it comes to applying Deep Learning models. As a solution, WSIs are often divided into smaller patches with a global label (\textit{i.e., diagnostic}) per slide, instead of a (too) costly pixel-wise annotation. By treating each slide as a bag of patches, Multiple Instance Learning (MIL) methods have emerged as a suitable solution for WSI classification. A major drawback of MIL methods is their high variability in performance across different runs, which can reach up to 10-15 AUC points on the test set, making it difficult to compare different MIL methods reliably. This variability mainly comes from three factors: i) weight initialization, ii) batch (shuffling) ordering, iii) and learning rate. To address that, we introduce a Multi-Fidelity, Model Fusion strategy for MIL methods. We first train multiple models for a few epochs and average the most stable and promising ones based on validation scores. This approach can be applied to any existing MIL model to reduce performance variability. It also simplifies hyperparameter tuning and improves reproducibility while maintaining computational efficiency. We extensively validate our approach on WSI classification tasks using 2 different datasets, 3 initialization strategies and 5 MIL methods, for a total of more than 2000 experiments.
>
---
#### [new 065] Computer Vision for Objects used in Group Work: Challenges and Opportunities
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于6D姿态估计任务，旨在解决协作场景中物体姿态识别难题。通过构建FiboSB数据集并改进YOLO11-x模型，提升协作环境中物体姿态估计的准确性。**

- **链接: [http://arxiv.org/pdf/2507.00224v1](http://arxiv.org/pdf/2507.00224v1)**

> **作者:** Changsoo Jung; Sheikh Mannan; Jack Fitzgerald; Nathaniel Blanchard
>
> **备注:** Accepted to AIED 2025 Late Breaking Results Track
>
> **摘要:** Interactive and spatially aware technologies are transforming educational frameworks, particularly in K-12 settings where hands-on exploration fosters deeper conceptual understanding. However, during collaborative tasks, existing systems often lack the ability to accurately capture real-world interactions between students and physical objects. This issue could be addressed with automatic 6D pose estimation, i.e., estimation of an object's position and orientation in 3D space from RGB images or videos. For collaborative groups that interact with physical objects, 6D pose estimates allow AI systems to relate objects and entities. As part of this work, we introduce FiboSB, a novel and challenging 6D pose video dataset featuring groups of three participants solving an interactive task featuring small hand-held cubes and a weight scale. This setup poses unique challenges for 6D pose because groups are holistically recorded from a distance in order to capture all participants -- this, coupled with the small size of the cubes, makes 6D pose estimation inherently non-trivial. We evaluated four state-of-the-art 6D pose estimation methods on FiboSB, exposing the limitations of current algorithms on collaborative group work. An error analysis of these methods reveals that the 6D pose methods' object detection modules fail. We address this by fine-tuning YOLO11-x for FiboSB, achieving an overall mAP_50 of 0.898. The dataset, benchmark results, and analysis of YOLO11-x errors presented here lay the groundwork for leveraging the estimation of 6D poses in difficult collaborative contexts.
>
---
#### [new 066] Efficient Depth- and Spatially-Varying Image Simulation for Defocus Deblur
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像去模糊任务，解决大光圈相机浅景深导致的模糊问题。通过合成数据模拟深度和空间变化的光学像差，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00372v1](http://arxiv.org/pdf/2507.00372v1)**

> **作者:** Xinge Yang; Chuong Nguyen; Wenbin Wang; Kaizhang Kang; Wolfgang Heidrich; Xiaoxing Li
>
> **摘要:** Modern cameras with large apertures often suffer from a shallow depth of field, resulting in blurry images of objects outside the focal plane. This limitation is particularly problematic for fixed-focus cameras, such as those used in smart glasses, where adding autofocus mechanisms is challenging due to form factor and power constraints. Due to unmatched optical aberrations and defocus properties unique to each camera system, deep learning models trained on existing open-source datasets often face domain gaps and do not perform well in real-world settings. In this paper, we propose an efficient and scalable dataset synthesis approach that does not rely on fine-tuning with real-world data. Our method simultaneously models depth-dependent defocus and spatially varying optical aberrations, addressing both computational complexity and the scarcity of high-quality RGB-D datasets. Experimental results demonstrate that a network trained on our low resolution synthetic images generalizes effectively to high resolution (12MP) real-world images across diverse scenes.
>
---
#### [new 067] BEV-VAE: Multi-view Image Generation with Spatial Consistency for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于多视角图像生成任务，旨在解决自动驾驶中跨视角场景理解不一致的问题。提出BEV-VAE模型，通过3D结构化表示实现一致且可控的视图合成。**

- **链接: [http://arxiv.org/pdf/2507.00707v1](http://arxiv.org/pdf/2507.00707v1)**

> **作者:** Zeming Chen; Hang Zhao
>
> **摘要:** Multi-view image generation in autonomous driving demands consistent 3D scene understanding across camera views. Most existing methods treat this problem as a 2D image set generation task, lacking explicit 3D modeling. However, we argue that a structured representation is crucial for scene generation, especially for autonomous driving applications. This paper proposes BEV-VAE for consistent and controllable view synthesis. BEV-VAE first trains a multi-view image variational autoencoder for a compact and unified BEV latent space and then generates the scene with a latent diffusion transformer. BEV-VAE supports arbitrary view generation given camera configurations, and optionally 3D layouts. Experiments on nuScenes and Argoverse 2 (AV2) show strong performance in both 3D consistent reconstruction and generation. The code is available at: https://github.com/Czm369/bev-vae.
>
---
#### [new 068] Rectifying Magnitude Neglect in Linear Attention
- **分类: cs.CV**

- **简介: 该论文属于视觉与自然语言处理任务，旨在解决Linear Attention性能下降问题。通过引入Query的幅度信息，提出MALA模型，提升其表现。**

- **链接: [http://arxiv.org/pdf/2507.00698v1](http://arxiv.org/pdf/2507.00698v1)**

> **作者:** Qihang Fan; Huaibo Huang; Yuang Ai; ran He
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** As the core operator of Transformers, Softmax Attention exhibits excellent global modeling capabilities. However, its quadratic complexity limits its applicability to vision tasks. In contrast, Linear Attention shares a similar formulation with Softmax Attention while achieving linear complexity, enabling efficient global information modeling. Nevertheless, Linear Attention suffers from a significant performance degradation compared to standard Softmax Attention. In this paper, we analyze the underlying causes of this issue based on the formulation of Linear Attention. We find that, unlike Softmax Attention, Linear Attention entirely disregards the magnitude information of the Query. This prevents the attention score distribution from dynamically adapting as the Query scales. As a result, despite its structural similarity to Softmax Attention, Linear Attention exhibits a significantly different attention score distribution. Based on this observation, we propose Magnitude-Aware Linear Attention (MALA), which modifies the computation of Linear Attention to fully incorporate the Query's magnitude. This adjustment allows MALA to generate an attention score distribution that closely resembles Softmax Attention while exhibiting a more well-balanced structure. We evaluate the effectiveness of MALA on multiple tasks, including image classification, object detection, instance segmentation, semantic segmentation, natural language processing, speech recognition, and image generation. Our MALA achieves strong results on all of these tasks. Code will be available at https://github.com/qhfan/MALA
>
---
#### [new 069] UMDATrack: Unified Multi-Domain Adaptive Tracking Under Adverse Weather Conditions
- **分类: cs.CV**

- **简介: 该论文属于视觉目标跟踪任务，解决恶劣天气下跟踪性能下降的问题。通过域适应框架UMDATrack，实现跨域目标状态预测。**

- **链接: [http://arxiv.org/pdf/2507.00648v1](http://arxiv.org/pdf/2507.00648v1)**

> **作者:** Siyuan Yao; Rui Zhu; Ziqi Wang; Wenqi Ren; Yanyang Yan; Xiaochun Cao
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Visual object tracking has gained promising progress in past decades. Most of the existing approaches focus on learning target representation in well-conditioned daytime data, while for the unconstrained real-world scenarios with adverse weather conditions, e.g. nighttime or foggy environment, the tremendous domain shift leads to significant performance degradation. In this paper, we propose UMDATrack, which is capable of maintaining high-quality target state prediction under various adverse weather conditions within a unified domain adaptation framework. Specifically, we first use a controllable scenario generator to synthesize a small amount of unlabeled videos (less than 2% frames in source daytime datasets) in multiple weather conditions under the guidance of different text prompts. Afterwards, we design a simple yet effective domain-customized adapter (DCA), allowing the target objects' representation to rapidly adapt to various weather conditions without redundant model updating. Furthermore, to enhance the localization consistency between source and target domains, we propose a target-aware confidence alignment module (TCA) following optimal transport theorem. Extensive experiments demonstrate that UMDATrack can surpass existing advanced visual trackers and lead new state-of-the-art performance by a significant margin. Our code is available at https://github.com/Z-Z188/UMDATrack.
>
---
#### [new 070] Context-Aware Academic Emotion Dataset and Benchmark
- **分类: cs.CV**

- **简介: 该论文属于学术情感识别任务，旨在解决真实学习环境中情感识别数据不足与上下文依赖问题。提出RAER数据集和CLIP-CAER方法，提升情感识别准确性。**

- **链接: [http://arxiv.org/pdf/2507.00586v1](http://arxiv.org/pdf/2507.00586v1)**

> **作者:** Luming Zhao; Jingwen Xuan; Jiamin Lou; Yonghui Yu; Wenwu Yang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Academic emotion analysis plays a crucial role in evaluating students' engagement and cognitive states during the learning process. This paper addresses the challenge of automatically recognizing academic emotions through facial expressions in real-world learning environments. While significant progress has been made in facial expression recognition for basic emotions, academic emotion recognition remains underexplored, largely due to the scarcity of publicly available datasets. To bridge this gap, we introduce RAER, a novel dataset comprising approximately 2,700 video clips collected from around 140 students in diverse, natural learning contexts such as classrooms, libraries, laboratories, and dormitories, covering both classroom sessions and individual study. Each clip was annotated independently by approximately ten annotators using two distinct sets of academic emotion labels with varying granularity, enhancing annotation consistency and reliability. To our knowledge, RAER is the first dataset capturing diverse natural learning scenarios. Observing that annotators naturally consider context cues-such as whether a student is looking at a phone or reading a book-alongside facial expressions, we propose CLIP-CAER (CLIP-based Context-aware Academic Emotion Recognition). Our method utilizes learnable text prompts within the vision-language model CLIP to effectively integrate facial expression and context cues from videos. Experimental results demonstrate that CLIP-CAER substantially outperforms state-of-the-art video-based facial expression recognition methods, which are primarily designed for basic emotions, emphasizing the crucial role of context in accurately recognizing academic emotions. Project page: https://zgsfer.github.io/CAER
>
---
#### [new 071] LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像修复任务，旨在解决传统方法依赖配对数据和泛化能力差的问题。提出一种无需数据集的统一方法，利用扩散模型进行后验采样和迭代优化。**

- **链接: [http://arxiv.org/pdf/2507.00790v1](http://arxiv.org/pdf/2507.00790v1)**

> **作者:** Huaqiu Li; Yong Wang; Tongwen Huang; Hailang Huang; Haoqian Wang; Xiangxiang Chu
>
> **摘要:** Unified image restoration is a significantly challenging task in low-level vision. Existing methods either make tailored designs for specific tasks, limiting their generalizability across various types of degradation, or rely on training with paired datasets, thereby suffering from closed-set constraints. To address these issues, we propose a novel, dataset-free, and unified approach through recurrent posterior sampling utilizing a pretrained latent diffusion model. Our method incorporates the multimodal understanding model to provide sematic priors for the generative model under a task-blind condition. Furthermore, it utilizes a lightweight module to align the degraded input with the generated preference of the diffusion model, and employs recurrent refinement for posterior sampling. Extensive experiments demonstrate that our method outperforms state-of-the-art methods, validating its effectiveness and robustness. Our code and data will be available at https://github.com/AMAP-ML/LD-RPS.
>
---
#### [new 072] SafeMap: Robust HD Map Construction from Incomplete Observations
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的高精度地图构建任务，解决多视角数据不完整时地图准确性不足的问题。提出SafeMap框架，结合视图重建与BEV校正模块，提升地图鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.00861v1](http://arxiv.org/pdf/2507.00861v1)**

> **作者:** Xiaoshuai Hao; Lingdong Kong; Rong Yin; Pengwei Wang; Jing Zhang; Yunfeng Diao; Shu Zhao
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Robust high-definition (HD) map construction is vital for autonomous driving, yet existing methods often struggle with incomplete multi-view camera data. This paper presents SafeMap, a novel framework specifically designed to secure accuracy even when certain camera views are missing. SafeMap integrates two key components: the Gaussian-based Perspective View Reconstruction (G-PVR) module and the Distillation-based Bird's-Eye-View (BEV) Correction (D-BEVC) module. G-PVR leverages prior knowledge of view importance to dynamically prioritize the most informative regions based on the relationships among available camera views. Furthermore, D-BEVC utilizes panoramic BEV features to correct the BEV representations derived from incomplete observations. Together, these components facilitate the end-to-end map reconstruction and robust HD map generation. SafeMap is easy to implement and integrates seamlessly into existing systems, offering a plug-and-play solution for enhanced robustness. Experimental results demonstrate that SafeMap significantly outperforms previous methods in both complete and incomplete scenarios, highlighting its superior performance and reliability.
>
---
#### [new 073] UAVD-Mamba: Deformable Token Fusion Vision Mamba for Multimodal UAV Detection
- **分类: cs.CV**

- **简介: 该论文属于多模态无人机目标检测任务，旨在解决遮挡、小目标和不规则形状等问题。提出UAVD-Mamba框架，结合可变形token融合与多尺度特征提取，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.00849v1](http://arxiv.org/pdf/2507.00849v1)**

> **作者:** Wei Li; Jiaman Tang; Yang Li; Beihao Xia; Ligang Tan; Hongmao Qin
>
> **备注:** The paper was accepted by the 36th IEEE Intelligent Vehicles Symposium (IEEE IV 2025)
>
> **摘要:** Unmanned Aerial Vehicle (UAV) object detection has been widely used in traffic management, agriculture, emergency rescue, etc. However, it faces significant challenges, including occlusions, small object sizes, and irregular shapes. These challenges highlight the necessity for a robust and efficient multimodal UAV object detection method. Mamba has demonstrated considerable potential in multimodal image fusion. Leveraging this, we propose UAVD-Mamba, a multimodal UAV object detection framework based on Mamba architectures. To improve geometric adaptability, we propose the Deformable Token Mamba Block (DTMB) to generate deformable tokens by incorporating adaptive patches from deformable convolutions alongside normal patches from normal convolutions, which serve as the inputs to the Mamba Block. To optimize the multimodal feature complementarity, we design two separate DTMBs for the RGB and infrared (IR) modalities, with the outputs from both DTMBs integrated into the Mamba Block for feature extraction and into the Fusion Mamba Block for feature fusion. Additionally, to improve multiscale object detection, especially for small objects, we stack four DTMBs at different scales to produce multiscale feature representations, which are then sent to the Detection Neck for Mamba (DNM). The DNM module, inspired by the YOLO series, includes modifications to the SPPF and C3K2 of YOLOv11 to better handle the multiscale features. In particular, we employ cross-enhanced spatial attention before the DTMB and cross-channel attention after the Fusion Mamba Block to extract more discriminative features. Experimental results on the DroneVehicle dataset show that our method outperforms the baseline OAFA method by 3.6% in the mAP metric. Codes will be released at https://github.com/GreatPlum-hnu/UAVD-Mamba.git.
>
---
#### [new 074] GDGS: 3D Gaussian Splatting Via Geometry-Guided Initialization And Dynamic Density Control
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决3D高斯点云渲染中的初始化、优化和密度控制问题，提出几何引导初始化和动态密度控制方法以提升渲染质量与效率。**

- **链接: [http://arxiv.org/pdf/2507.00363v1](http://arxiv.org/pdf/2507.00363v1)**

> **作者:** Xingjun Wang; Lianlei Shan
>
> **摘要:** We propose a method to enhance 3D Gaussian Splatting (3DGS)~\cite{Kerbl2023}, addressing challenges in initialization, optimization, and density control. Gaussian Splatting is an alternative for rendering realistic images while supporting real-time performance, and it has gained popularity due to its explicit 3D Gaussian representation. However, 3DGS heavily depends on accurate initialization and faces difficulties in optimizing unstructured Gaussian distributions into ordered surfaces, with limited adaptive density control mechanism proposed so far. Our first key contribution is a geometry-guided initialization to predict Gaussian parameters, ensuring precise placement and faster convergence. We then introduce a surface-aligned optimization strategy to refine Gaussian placement, improving geometric accuracy and aligning with the surface normals of the scene. Finally, we present a dynamic adaptive density control mechanism that adjusts Gaussian density based on regional complexity, for visual fidelity. These innovations enable our method to achieve high-fidelity real-time rendering and significant improvements in visual quality, even in complex scenes. Our method demonstrates comparable or superior results to state-of-the-art methods, rendering high-fidelity images in real time.
>
---
#### [new 075] Customizable ROI-Based Deep Image Compression
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像压缩任务，解决传统ROI压缩不可定制的问题。通过文本控制ROI定义、可调掩码和潜在注意力机制，实现灵活的ROI压缩与质量平衡。**

- **链接: [http://arxiv.org/pdf/2507.00373v1](http://arxiv.org/pdf/2507.00373v1)**

> **作者:** Ian Jin; Fanxin Xia; Feng Ding; Xinfeng Zhang; Meiqin Liu; Yao Zhao; Weisi Lin; Lili Meng
>
> **摘要:** Region of Interest (ROI)-based image compression optimizes bit allocation by prioritizing ROI for higher-quality reconstruction. However, as the users (including human clients and downstream machine tasks) become more diverse, ROI-based image compression needs to be customizable to support various preferences. For example, different users may define distinct ROI or require different quality trade-offs between ROI and non-ROI. Existing ROI-based image compression schemes predefine the ROI, making it unchangeable, and lack effective mechanisms to balance reconstruction quality between ROI and non-ROI. This work proposes a paradigm for customizable ROI-based deep image compression. First, we develop a Text-controlled Mask Acquisition (TMA) module, which allows users to easily customize their ROI for compression by just inputting the corresponding semantic \emph{text}. It makes the encoder controlled by text. Second, we design a Customizable Value Assign (CVA) mechanism, which masks the non-ROI with a changeable extent decided by users instead of a constant one to manage the reconstruction quality trade-off between ROI and non-ROI. Finally, we present a Latent Mask Attention (LMA) module, where the latent spatial prior of the mask and the latent Rate-Distortion Optimization (RDO) prior of the image are extracted and fused in the latent space, and further used to optimize the latent representation of the source image. Experimental results demonstrate that our proposed customizable ROI-based deep image compression paradigm effectively addresses the needs of customization for ROI definition and mask acquisition as well as the reconstruction quality trade-off management between the ROI and non-ROI.
>
---
#### [new 076] MedDiff-FT: Data-Efficient Diffusion Model Fine-tuning with Structural Guidance for Controllable Medical Image Synthesis
- **分类: cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决数据稀缺和生成质量不足的问题。通过微调扩散模型，结合结构引导和高效数据利用，提升医学图像合成的准确性与多样性。**

- **链接: [http://arxiv.org/pdf/2507.00377v1](http://arxiv.org/pdf/2507.00377v1)**

> **作者:** Jianhao Xie; Ziang Zhang; Zhenyu Weng; Yuesheng Zhu; Guibo Luo
>
> **备注:** 11 pages,3 figures
>
> **摘要:** Recent advancements in deep learning for medical image segmentation are often limited by the scarcity of high-quality training data.While diffusion models provide a potential solution by generating synthetic images, their effectiveness in medical imaging remains constrained due to their reliance on large-scale medical datasets and the need for higher image quality. To address these challenges, we present MedDiff-FT, a controllable medical image generation method that fine-tunes a diffusion foundation model to produce medical images with structural dependency and domain specificity in a data-efficient manner. During inference, a dynamic adaptive guiding mask enforces spatial constraints to ensure anatomically coherent synthesis, while a lightweight stochastic mask generator enhances diversity through hierarchical randomness injection. Additionally, an automated quality assessment protocol filters suboptimal outputs using feature-space metrics, followed by mask corrosion to refine fidelity. Evaluated on five medical segmentation datasets,MedDiff-FT's synthetic image-mask pairs improve SOTA method's segmentation performance by an average of 1% in Dice score. The framework effectively balances generation quality, diversity, and computational efficiency, offering a practical solution for medical data augmentation. The code is available at https://github.com/JianhaoXie1/MedDiff-FT.
>
---
#### [new 077] Instant Particle Size Distribution Measurement Using CNNs Trained on Synthetic Data
- **分类: cs.CV**

- **简介: 该论文属于颗粒尺寸分布测量任务，旨在解决传统方法效率低、依赖人工的问题。通过训练CNN模型，利用合成数据实现快速准确的PSD预测。**

- **链接: [http://arxiv.org/pdf/2507.00822v1](http://arxiv.org/pdf/2507.00822v1)**

> **作者:** Yasser El Jarida; Youssef Iraqi; Loubna Mekouar
>
> **备注:** Accepted at the Synthetic Data for Computer Vision Workshop @ CVPR 2025. 10 pages, 5 figures. Code available at https://github.com/YasserElj/Synthetic-Granular-Gen
>
> **摘要:** Accurate particle size distribution (PSD) measurement is important in industries such as mining, pharmaceuticals, and fertilizer manufacturing, significantly influencing product quality and operational efficiency. Traditional PSD methods like sieve analysis and laser diffraction are manual, time-consuming, and limited by particle overlap. Recent developments in convolutional neural networks (CNNs) enable automated, real-time PSD estimation directly from particle images. In this work, we present a CNN-based methodology trained on realistic synthetic particle imagery generated using Blender's advanced rendering capabilities. Synthetic data sets using this method can replicate various industrial scenarios by systematically varying particle shapes, textures, lighting, and spatial arrangements that closely resemble the actual configurations. We evaluated three CNN-based architectures, ResNet-50, InceptionV3, and EfficientNet-B0, for predicting critical PSD parameters (d10, d50, d90). Results demonstrated comparable accuracy across models, with EfficientNet-B0 achieving the best computational efficiency suitable for real-time industrial deployment. This approach shows the effectiveness of realistic synthetic data for robust CNN training, which offers significant potential for automated industrial PSD monitoring. The code is released at : https://github.com/YasserElj/Synthetic-Granular-Gen
>
---
#### [new 078] An Improved U-Net Model for Offline handwriting signature denoising
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像去噪任务，旨在解决离线手写签名图像中噪声干扰的问题。通过改进U-Net模型，结合小波和PCA变换提升去噪效果。**

- **链接: [http://arxiv.org/pdf/2507.00365v1](http://arxiv.org/pdf/2507.00365v1)**

> **作者:** Wanghui Xiao
>
> **摘要:** Handwriting signatures, as an important means of identity recognition, are widely used in multiple fields such as financial transactions, commercial contracts and personal affairs due to their legal effect and uniqueness. In forensic science appraisals, the analysis of offline handwriting signatures requires the appraiser to provide a certain number of signature samples, which are usually derived from various historical contracts or archival materials. However, the provided handwriting samples are often mixed with a large amount of interfering information, which brings severe challenges to handwriting identification work. This study proposes a signature handwriting denoising model based on the improved U-net structure, aiming to enhance the robustness of the signature recognition system. By introducing discrete wavelet transform and PCA transform, the model's ability to suppress noise has been enhanced. The experimental results show that this modelis significantly superior to the traditional methods in denoising effect, can effectively improve the clarity and readability of the signed images, and provide more reliable technical support for signature analysis and recognition.
>
---
#### [new 079] An efficient plant disease detection using transfer learning approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于植物病害检测任务，旨在解决早期病害识别问题。通过迁移学习优化YOLOv7和YOLOv8模型，实现高效准确的病害检测。**

- **链接: [http://arxiv.org/pdf/2507.00070v1](http://arxiv.org/pdf/2507.00070v1)**

> **作者:** Bosubabu Sambana; Hillary Sunday Nnadi; Mohd Anas Wajid; Nwosu Ogochukwu Fidelia; Claudia Camacho-Zuñiga; Henry Dozie Ajuzie; Edeh Michael Onyema
>
> **备注:** 15 pages , 4 figures. Scientific Reports 2025
>
> **摘要:** Plant diseases pose significant challenges to farmers and the agricultural sector at large. However, early detection of plant diseases is crucial to mitigating their effects and preventing widespread damage, as outbreaks can severely impact the productivity and quality of crops. With advancements in technology, there are increasing opportunities for automating the monitoring and detection of disease outbreaks in plants. This study proposed a system designed to identify and monitor plant diseases using a transfer learning approach. Specifically, the study utilizes YOLOv7 and YOLOv8, two state-ofthe-art models in the field of object detection. By fine-tuning these models on a dataset of plant leaf images, the system is able to accurately detect the presence of Bacteria, Fungi and Viral diseases such as Powdery Mildew, Angular Leaf Spot, Early blight and Tomato mosaic virus. The model's performance was evaluated using several metrics, including mean Average Precision (mAP), F1-score, Precision, and Recall, yielding values of 91.05, 89.40, 91.22, and 87.66, respectively. The result demonstrates the superior effectiveness and efficiency of YOLOv8 compared to other object detection methods, highlighting its potential for use in modern agricultural practices. The approach provides a scalable, automated solution for early any plant disease detection, contributing to enhanced crop yield, reduced reliance on manual monitoring, and supporting sustainable agricultural practices.
>
---
#### [new 080] VirtualFencer: Generating Fencing Bouts based on Strategies Extracted from In-the-Wild Videos
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于动作生成任务，旨在解决如何从视频中提取击剑动作与策略并生成真实击剑行为的问题。工作包括无监督提取3D动作与策略，并生成对抗性击剑对战。**

- **链接: [http://arxiv.org/pdf/2507.00261v1](http://arxiv.org/pdf/2507.00261v1)**

> **作者:** Zhiyin Lin; Purvi Goel; Joy Yun; C. Karen Liu; Joao Pedro Araujo
>
> **摘要:** Fencing is a sport where athletes engage in diverse yet strategically logical motions. While most motions fall into a few high-level actions (e.g. step, lunge, parry), the execution can vary widely-fast vs. slow, large vs. small, offensive vs. defensive. Moreover, a fencer's actions are informed by a strategy that often comes in response to the opponent's behavior. This combination of motion diversity with underlying two-player strategy motivates the application of data-driven modeling to fencing. We present VirtualFencer, a system capable of extracting 3D fencing motion and strategy from in-the-wild video without supervision, and then using that extracted knowledge to generate realistic fencing behavior. We demonstrate the versatile capabilities of our system by having it (i) fence against itself (self-play), (ii) fence against a real fencer's motion from online video, and (iii) fence interactively against a professional fencer.
>
---
#### [new 081] Topology-Constrained Learning for Efficient Laparoscopic Liver Landmark Detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决腹腔镜肝脏关键点检测难题。通过引入拓扑约束学习框架TopoNet，提升检测精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.00519v1](http://arxiv.org/pdf/2507.00519v1)**

> **作者:** Ruize Cui; Jiaan Zhang; Jialun Pei; Kai Wang; Pheng-Ann Heng; Jing Qin
>
> **备注:** This paper has been accepted by MICCAI 2025
>
> **摘要:** Liver landmarks provide crucial anatomical guidance to the surgeon during laparoscopic liver surgery to minimize surgical risk. However, the tubular structural properties of landmarks and dynamic intraoperative deformations pose significant challenges for automatic landmark detection. In this study, we introduce TopoNet, a novel topology-constrained learning framework for laparoscopic liver landmark detection. Our framework adopts a snake-CNN dual-path encoder to simultaneously capture detailed RGB texture information and depth-informed topological structures. Meanwhile, we propose a boundary-aware topology fusion (BTF) module, which adaptively merges RGB-D features to enhance edge perception while preserving global topology. Additionally, a topological constraint loss function is embedded, which contains a center-line constraint loss and a topological persistence loss to ensure homotopy equivalence between predictions and labels. Extensive experiments on L3D and P2ILF datasets demonstrate that TopoNet achieves outstanding accuracy and computational complexity, highlighting the potential for clinical applications in laparoscopic liver surgery. Our code will be available at https://github.com/cuiruize/TopoNet.
>
---
#### [new 082] Towards Open-World Human Action Segmentation Using Graph Convolutional Networks
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人类动作分割任务，解决开放世界中新动作识别与分割问题。提出一种基于图卷积网络的框架，结合数据增强和时序聚类损失，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00756v1](http://arxiv.org/pdf/2507.00756v1)**

> **作者:** Hao Xing; Kai Zhe Boey; Gordon Cheng
>
> **备注:** 8 pages, 3 figures, accepted in IROS25, Hangzhou, China
>
> **摘要:** Human-object interaction segmentation is a fundamental task of daily activity understanding, which plays a crucial role in applications such as assistive robotics, healthcare, and autonomous systems. Most existing learning-based methods excel in closed-world action segmentation, they struggle to generalize to open-world scenarios where novel actions emerge. Collecting exhaustive action categories for training is impractical due to the dynamic diversity of human activities, necessitating models that detect and segment out-of-distribution actions without manual annotation. To address this issue, we formally define the open-world action segmentation problem and propose a structured framework for detecting and segmenting unseen actions. Our framework introduces three key innovations: 1) an Enhanced Pyramid Graph Convolutional Network (EPGCN) with a novel decoder module for robust spatiotemporal feature upsampling. 2) Mixup-based training to synthesize out-of-distribution data, eliminating reliance on manual annotations. 3) A novel Temporal Clustering loss that groups in-distribution actions while distancing out-of-distribution samples. We evaluate our framework on two challenging human-object interaction recognition datasets: Bimanual Actions and 2 Hands and Object (H2O) datasets. Experimental results demonstrate significant improvements over state-of-the-art action segmentation models across multiple open-set evaluation metrics, achieving 16.9% and 34.6% relative gains in open-set segmentation (F1@50) and out-of-distribution detection performances (AUROC), respectively. Additionally, we conduct an in-depth ablation study to assess the impact of each proposed component, identifying the optimal framework configuration for open-world action segmentation.
>
---
#### [new 083] HistoART: Histopathology Artifact Detection and Reporting Tool
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于病理图像分析任务，旨在解决WSI中的伪影检测问题。通过三种方法检测六类常见伪影，并生成质量报告。**

- **链接: [http://arxiv.org/pdf/2507.00044v1](http://arxiv.org/pdf/2507.00044v1)**

> **作者:** Seyed Kahaki; Alexander R. Webber; Ghada Zamzmi; Adarsh Subbaswamy; Rucha Deshpande; Aldo Badano
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** In modern cancer diagnostics, Whole Slide Imaging (WSI) is widely used to digitize tissue specimens for detailed, high-resolution examination; however, other diagnostic approaches, such as liquid biopsy and molecular testing, are also utilized based on the cancer type and clinical context. While WSI has revolutionized digital histopathology by enabling automated, precise analysis, it remains vulnerable to artifacts introduced during slide preparation and scanning. These artifacts can compromise downstream image analysis. To address this challenge, we propose and compare three robust artifact detection approaches for WSIs: (1) a foundation model-based approach (FMA) using a fine-tuned Unified Neural Image (UNI) architecture, (2) a deep learning approach (DLA) built on a ResNet50 backbone, and (3) a knowledge-based approach (KBA) leveraging handcrafted features from texture, color, and frequency-based metrics. The methods target six common artifact types: tissue folds, out-of-focus regions, air bubbles, tissue damage, marker traces, and blood contamination. Evaluations were conducted on 50,000+ image patches from diverse scanners (Hamamatsu, Philips, Leica Aperio AT2) across multiple sites. The FMA achieved the highest patch-wise AUROC of 0.995 (95% CI [0.994, 0.995]), outperforming the ResNet50-based method (AUROC: 0.977, 95% CI [0.977, 0.978]) and the KBA (AUROC: 0.940, 95% CI [0.933, 0.946]). To translate detection into actionable insights, we developed a quality report scorecard that quantifies high-quality patches and visualizes artifact distributions.
>
---
#### [new 084] Real-Time Inverse Kinematics for Generating Multi-Constrained Movements of Virtual Human Characters
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于实时逆运动学任务，解决虚拟人角色多约束运动生成问题。通过TensorFlow实现高效求解器，提升计算效率与成功率。**

- **链接: [http://arxiv.org/pdf/2507.00792v1](http://arxiv.org/pdf/2507.00792v1)**

> **作者:** Hendric Voss; Stefan Kopp
>
> **摘要:** Generating accurate and realistic virtual human movements in real-time is of high importance for a variety of applications in computer graphics, interactive virtual environments, robotics, and biomechanics. This paper introduces a novel real-time inverse kinematics (IK) solver specifically designed for realistic human-like movement generation. Leveraging the automatic differentiation and just-in-time compilation of TensorFlow, the proposed solver efficiently handles complex articulated human skeletons with high degrees of freedom. By treating forward and inverse kinematics as differentiable operations, our method effectively addresses common challenges such as error accumulation and complicated joint limits in multi-constrained problems, which are critical for realistic human motion modeling. We demonstrate the solver's effectiveness on the SMPLX human skeleton model, evaluating its performance against widely used iterative-based IK algorithms, like Cyclic Coordinate Descent (CCD), FABRIK, and the nonlinear optimization algorithm IPOPT. Our experiments cover both simple end-effector tasks and sophisticated, multi-constrained problems with realistic joint limits. Results indicate that our IK solver achieves real-time performance, exhibiting rapid convergence, minimal computational overhead per iteration, and improved success rates compared to existing methods. The project code is available at https://github.com/hvoss-techfak/TF-JAX-IK
>
---
#### [new 085] Bisecle: Binding and Separation in Continual Learning for Video Language Understanding
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视频语言理解任务，解决持续学习中的灾难性遗忘和更新冲突问题，提出Bisecle方法提升模型记忆与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00469v1](http://arxiv.org/pdf/2507.00469v1)**

> **作者:** Yue Tan; Xiaoqian Hu; Hao Xue; Celso De Melo; Flora D. Salim
>
> **备注:** 23 pages, 12 figures, 10 tables
>
> **摘要:** Frontier vision-language models (VLMs) have made remarkable improvements in video understanding tasks. However, real-world videos typically exist as continuously evolving data streams (e.g., dynamic scenes captured by wearable glasses), necessitating models to continually adapt to shifting data distributions and novel scenarios. Considering the prohibitive computational costs of fine-tuning models on new tasks, usually, a small subset of parameters is updated while the bulk of the model remains frozen. This poses new challenges to existing continual learning frameworks in the context of large multimodal foundation models, i.e., catastrophic forgetting and update conflict. While the foundation models struggle with parameter-efficient continual learning, the hippocampus in the human brain has evolved highly efficient mechanisms for memory formation and consolidation. Inspired by the rapid Binding and pattern separation mechanisms in the hippocampus, in this work, we propose Bisecle for video-language continual learning, where a multi-directional supervision module is used to capture more cross-modal relationships and a contrastive prompt learning scheme is designed to isolate task-specific knowledge to facilitate efficient memory storage. Binding and separation processes further strengthen the ability of VLMs to retain complex experiences, enabling robust and efficient continual learning in video understanding tasks. We perform a thorough evaluation of the proposed Bisecle, demonstrating its ability to mitigate forgetting and enhance cross-task generalization on several VideoQA benchmarks.
>
---
#### [new 086] MVP: Winning Solution to SMP Challenge 2025 Video Track
- **分类: cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于视频热度预测任务，旨在准确预测社交媒体视频的流行度。通过融合多模态特征和用户数据，构建了MVP模型，有效提升了预测效果。**

- **链接: [http://arxiv.org/pdf/2507.00950v1](http://arxiv.org/pdf/2507.00950v1)**

> **作者:** Liliang Ye; Yunyao Zhang; Yafeng Wu; Yi-Ping Phoebe Chen; Junqing Yu; Wei Yang; Zikai Song
>
> **摘要:** Social media platforms serve as central hubs for content dissemination, opinion expression, and public engagement across diverse modalities. Accurately predicting the popularity of social media videos enables valuable applications in content recommendation, trend detection, and audience engagement. In this paper, we present Multimodal Video Predictor (MVP), our winning solution to the Video Track of the SMP Challenge 2025. MVP constructs expressive post representations by integrating deep video features extracted from pretrained models with user metadata and contextual information. The framework applies systematic preprocessing techniques, including log-transformations and outlier removal, to improve model robustness. A gradient-boosted regression model is trained to capture complex patterns across modalities. Our approach ranked first in the official evaluation of the Video Track, demonstrating its effectiveness and reliability for multimodal video popularity prediction on social platforms. The source code is available at https://anonymous.4open.science/r/SMPDVideo.
>
---
#### [new 087] TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶中的车道拓扑推理任务，解决现有方法在时空一致性与多属性学习上的不足，提出TopoStreamer模型提升车道结构感知性能。**

- **链接: [http://arxiv.org/pdf/2507.00709v1](http://arxiv.org/pdf/2507.00709v1)**

> **作者:** Yiming Yang; Yueru Luo; Bingkun He; Hongbin Lin; Suzhong Fu; Chao Yan; Kun Tang; Xinrui Yan; Chao Zheng; Shuguang Cui; Zhen Li
>
> **摘要:** Lane segment topology reasoning constructs a comprehensive road network by capturing the topological relationships between lane segments and their semantic types. This enables end-to-end autonomous driving systems to perform road-dependent maneuvers such as turning and lane changing. However, the limitations in consistent positional embedding and temporal multiple attribute learning in existing methods hinder accurate roadnet reconstruction. To address these issues, we propose TopoStreamer, an end-to-end temporal perception model for lane segment topology reasoning. Specifically, TopoStreamer introduces three key improvements: streaming attribute constraints, dynamic lane boundary positional encoding, and lane segment denoising. The streaming attribute constraints enforce temporal consistency in both centerline and boundary coordinates, along with their classifications. Meanwhile, dynamic lane boundary positional encoding enhances the learning of up-to-date positional information within queries, while lane segment denoising helps capture diverse lane segment patterns, ultimately improving model performance. Additionally, we assess the accuracy of existing models using a lane boundary classification metric, which serves as a crucial measure for lane-changing scenarios in autonomous driving. On the OpenLane-V2 dataset, TopoStreamer demonstrates significant improvements over state-of-the-art methods, achieving substantial performance gains of +3.4% mAP in lane segment perception and +2.1% OLS in centerline perception tasks.
>
---
#### [new 088] High-Frequency Semantics and Geometric Priors for End-to-End Detection Transformers in Challenging UAV Imagery
- **分类: cs.CV; I.2.10; I.4.8; I.5.1**

- **简介: 该论文属于无人机目标检测任务，旨在解决小目标、密集分布和复杂背景下的检测难题。提出HEGS-DETR框架，融合高频率语义与几何先验，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.00825v1](http://arxiv.org/pdf/2507.00825v1)**

> **作者:** Hongxing Peng; Lide Chen; Hui Zhu; Yan Chen
>
> **备注:** 14 pages, 9 figures, to appear in KBS
>
> **摘要:** Unmanned Aerial Vehicle-based Object Detection (UAV-OD) faces substantial challenges, including small target sizes, high-density distributions, and cluttered backgrounds in UAV imagery. Current algorithms often depend on hand-crafted components like anchor boxes, which demand fine-tuning and exhibit limited generalization, and Non-Maximum Suppression (NMS), which is threshold-sensitive and prone to misclassifying dense objects. These generic architectures thus struggle to adapt to aerial imaging characteristics, resulting in performance limitations. Moreover, emerging end-to-end frameworks have yet to effectively mitigate these aerial-specific challenges.To address these issues, we propose HEGS-DETR, a comprehensively enhanced, real-time Detection Transformer framework tailored for UAVs. First, we introduce the High-Frequency Enhanced Semantics Network (HFESNet) as a novel backbone. HFESNet preserves critical high-frequency spatial details to extract robust semantic features, thereby improving discriminative capability for small and occluded targets in complex backgrounds. Second, our Efficient Small Object Pyramid (ESOP) strategy strategically fuses high-resolution feature maps with minimal computational overhead, significantly boosting small object detection. Finally, the proposed Selective Query Recollection (SQR) and Geometry-Aware Positional Encoding (GAPE) modules enhance the detector's decoder stability and localization accuracy, effectively optimizing bounding boxes and providing explicit spatial priors for dense scenes. Experiments on the VisDrone dataset demonstrate that HEGS-DETR achieves a 5.1\% AP$_{50}$ and 3.8\% AP increase over the baseline, while maintaining real-time speed and reducing parameter count by 4M.
>
---
#### [new 089] AI-Generated Video Detection via Perceptual Straightening
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于AI生成视频检测任务，旨在解决真实与AI生成视频区分问题。通过分析神经网络表示中的几何特性，提出ReStraV方法，实现高效准确检测。**

- **链接: [http://arxiv.org/pdf/2507.00583v1](http://arxiv.org/pdf/2507.00583v1)**

> **作者:** Christian Internò; Robert Geirhos; Markus Olhofer; Sunny Liu; Barbara Hammer; David Klindt
>
> **摘要:** The rapid advancement of generative AI enables highly realistic synthetic videos, posing significant challenges for content authentication and raising urgent concerns about misuse. Existing detection methods often struggle with generalization and capturing subtle temporal inconsistencies. We propose ReStraV(Representation Straightening Video), a novel approach to distinguish natural from AI-generated videos. Inspired by the "perceptual straightening" hypothesis -- which suggests real-world video trajectories become more straight in neural representation domain -- we analyze deviations from this expected geometric property. Using a pre-trained self-supervised vision transformer (DINOv2), we quantify the temporal curvature and stepwise distance in the model's representation domain. We aggregate statistics of these measures for each video and train a classifier. Our analysis shows that AI-generated videos exhibit significantly different curvature and distance patterns compared to real videos. A lightweight classifier achieves state-of-the-art detection performance (e.g., 97.17% accuracy and 98.63% AUROC on the VidProM benchmark), substantially outperforming existing image- and video-based methods. ReStraV is computationally efficient, it is offering a low-cost and effective detection solution. This work provides new insights into using neural representation geometry for AI-generated video detection.
>
---
#### [new 090] ExPaMoE: An Expandable Parallel Mixture of Experts for Continual Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文提出ExPaMoE框架，解决持续测试时适应（CTTA）中的领域变化和遗忘问题，通过扩展并行专家架构实现高效适应。**

- **链接: [http://arxiv.org/pdf/2507.00502v1](http://arxiv.org/pdf/2507.00502v1)**

> **作者:** JianChao Zhao; Songlin Dong
>
> **摘要:** Continual Test-Time Adaptation (CTTA) aims to enable models to adapt on-the-fly to a stream of unlabeled data under evolving distribution shifts. However, existing CTTA methods typically rely on shared model parameters across all domains, making them vulnerable to feature entanglement and catastrophic forgetting in the presence of large or non-stationary domain shifts. To address this limitation, we propose \textbf{ExPaMoE}, a novel framework based on an \emph{Expandable Parallel Mixture-of-Experts} architecture. ExPaMoE decouples domain-general and domain-specific knowledge via a dual-branch expert design with token-guided feature separation, and dynamically expands its expert pool based on a \emph{Spectral-Aware Online Domain Discriminator} (SODD) that detects distribution changes in real-time using frequency-domain cues. Extensive experiments demonstrate the superiority of ExPaMoE across diverse CTTA scenarios. We evaluate our method on standard benchmarks including CIFAR-10C, CIFAR-100C, ImageNet-C, and Cityscapes-to-ACDC for semantic segmentation. Additionally, we introduce \textbf{ImageNet++}, a large-scale and realistic CTTA benchmark built from multiple ImageNet-derived datasets, to better reflect long-term adaptation under complex domain evolution. ExPaMoE consistently outperforms prior arts, showing strong robustness, scalability, and resistance to forgetting.
>
---
#### [new 091] FreeLong++: Training-Free Long Video Generation via Multi-band SpectralFusion
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决长视频生成中时间一致性与视觉质量下降的问题。提出FreeLong++框架，在不额外训练的情况下提升长视频的连贯性与细节表现。**

- **链接: [http://arxiv.org/pdf/2507.00162v1](http://arxiv.org/pdf/2507.00162v1)**

> **作者:** Yu Lu; Yi Yang
>
> **备注:** under review
>
> **摘要:** Recent advances in video generation models have enabled high-quality short video generation from text prompts. However, extending these models to longer videos remains a significant challenge, primarily due to degraded temporal consistency and visual fidelity. Our preliminary observations show that naively applying short-video generation models to longer sequences leads to noticeable quality degradation. Further analysis identifies a systematic trend where high-frequency components become increasingly distorted as video length grows, an issue we term high-frequency distortion. To address this, we propose FreeLong, a training-free framework designed to balance the frequency distribution of long video features during the denoising process. FreeLong achieves this by blending global low-frequency features, which capture holistic semantics across the full video, with local high-frequency features extracted from short temporal windows to preserve fine details. Building on this, FreeLong++ extends FreeLong dual-branch design into a multi-branch architecture with multiple attention branches, each operating at a distinct temporal scale. By arranging multiple window sizes from global to local, FreeLong++ enables multi-band frequency fusion from low to high frequencies, ensuring both semantic continuity and fine-grained motion dynamics across longer video sequences. Without any additional training, FreeLong++ can be plugged into existing video generation models (e.g. Wan2.1 and LTX-Video) to produce longer videos with substantially improved temporal consistency and visual fidelity. We demonstrate that our approach outperforms previous methods on longer video generation tasks (e.g. 4x and 8x of native length). It also supports coherent multi-prompt video generation with smooth scene transitions and enables controllable video generation using long depth or pose sequences.
>
---
#### [new 092] LOD-GS: Level-of-Detail-Sensitive 3D Gaussian Splatting for Detail Conserved Anti-Aliasing
- **分类: cs.CV**

- **简介: 该论文属于3D渲染任务，旨在解决3D高斯点云渲染中的走样问题。通过引入与细节层次相关的过滤框架，提升渲染质量并有效消除走样。**

- **链接: [http://arxiv.org/pdf/2507.00554v1](http://arxiv.org/pdf/2507.00554v1)**

> **作者:** Zhenya Yang; Bingchen Gong; Kai Chen; Qi Dou
>
> **摘要:** Despite the advancements in quality and efficiency achieved by 3D Gaussian Splatting (3DGS) in 3D scene rendering, aliasing artifacts remain a persistent challenge. Existing approaches primarily rely on low-pass filtering to mitigate aliasing. However, these methods are not sensitive to the sampling rate, often resulting in under-filtering and over-smoothing renderings. To address this limitation, we propose LOD-GS, a Level-of-Detail-sensitive filtering framework for Gaussian Splatting, which dynamically predicts the optimal filtering strength for each 3D Gaussian primitive. Specifically, we introduce a set of basis functions to each Gaussian, which take the sampling rate as input to model appearance variations, enabling sampling-rate-sensitive filtering. These basis function parameters are jointly optimized with the 3D Gaussian in an end-to-end manner. The sampling rate is influenced by both focal length and camera distance. However, existing methods and datasets rely solely on down-sampling to simulate focal length changes for anti-aliasing evaluation, overlooking the impact of camera distance. To enable a more comprehensive assessment, we introduce a new synthetic dataset featuring objects rendered at varying camera distances. Extensive experiments on both public datasets and our newly collected dataset demonstrate that our method achieves SOTA rendering quality while effectively eliminating aliasing. The code and dataset have been open-sourced.
>
---
#### [new 093] Latent Posterior-Mean Rectified Flow for Higher-Fidelity Perceptual Face Restoration
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像修复任务，旨在解决感知与失真之间的权衡问题。通过在潜在空间中改进PMRF方法，提升修复质量与效率。**

- **链接: [http://arxiv.org/pdf/2507.00447v1](http://arxiv.org/pdf/2507.00447v1)**

> **作者:** Xin Luo; Menglin Zhang; Yunwei Lan; Tianyu Zhang; Rui Li; Chang Liu; Dong Liu
>
> **备注:** Code and Models will be publicly available at https://github.com/Luciennnnnnn/Latent-PMRF
>
> **摘要:** The Perception-Distortion tradeoff (PD-tradeoff) theory suggests that face restoration algorithms must balance perceptual quality and fidelity. To achieve minimal distortion while maintaining perfect perceptual quality, Posterior-Mean Rectified Flow (PMRF) proposes a flow based approach where source distribution is minimum distortion estimations. Although PMRF is shown to be effective, its pixel-space modeling approach limits its ability to align with human perception, where human perception is defined as how humans distinguish between two image distributions. In this work, we propose Latent-PMRF, which reformulates PMRF in the latent space of a variational autoencoder (VAE), facilitating better alignment with human perception during optimization. By defining the source distribution on latent representations of minimum distortion estimation, we bound the minimum distortion by the VAE's reconstruction error. Moreover, we reveal the design of VAE is crucial, and our proposed VAE significantly outperforms existing VAEs in both reconstruction and restoration. Extensive experiments on blind face restoration demonstrate the superiority of Latent-PMRF, offering an improved PD-tradeoff compared to existing methods, along with remarkable convergence efficiency, achieving a 5.79X speedup over PMRF in terms of FID. Our code will be available as open-source.
>
---
#### [new 094] Is Visual in-Context Learning for Compositional Medical Tasks within Reach?
- **分类: cs.CV**

- **简介: 该论文研究视觉上下文学习在组合医学任务中的应用，旨在让模型适应多步骤任务而无需重新训练。通过合成任务生成和掩码训练优化模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00868v1](http://arxiv.org/pdf/2507.00868v1)**

> **作者:** Simon Reiß; Zdravko Marinov; Alexander Jaus; Constantin Seibold; M. Saquib Sarfraz; Erik Rodner; Rainer Stiefelhagen
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** In this paper, we explore the potential of visual in-context learning to enable a single model to handle multiple tasks and adapt to new tasks during test time without re-training. Unlike previous approaches, our focus is on training in-context learners to adapt to sequences of tasks, rather than individual tasks. Our goal is to solve complex tasks that involve multiple intermediate steps using a single model, allowing users to define entire vision pipelines flexibly at test time. To achieve this, we first examine the properties and limitations of visual in-context learning architectures, with a particular focus on the role of codebooks. We then introduce a novel method for training in-context learners using a synthetic compositional task generation engine. This engine bootstraps task sequences from arbitrary segmentation datasets, enabling the training of visual in-context learners for compositional tasks. Additionally, we investigate different masking-based training objectives to gather insights into how to train models better for solving complex, compositional tasks. Our exploration not only provides important insights especially for multi-modal medical task sequences but also highlights challenges that need to be addressed.
>
---
#### [new 095] CGEarthEye:A High-Resolution Remote Sensing Vision Foundation Model Based on the Jilin-1 Satellite Constellation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像智能解译任务，旨在解决高分辨率遥感数据不足导致的模型性能受限问题。通过构建CGEarthEye框架和JLSSD数据集，提升模型表现。**

- **链接: [http://arxiv.org/pdf/2507.00356v1](http://arxiv.org/pdf/2507.00356v1)**

> **作者:** Zhiwei Yi; Xin Cheng; Jingyu Ma; Ruifei Zhu; Junwei Tian; Yuanxiu Zhou; Xinge Zhao; Hongzhe Li
>
> **备注:** A Remote Sensing Fundation Model for Very High Resolution Images
>
> **摘要:** Deep learning methods have significantly advanced the development of intelligent rinterpretation in remote sensing (RS), with foundational model research based on large-scale pre-training paradigms rapidly reshaping various domains of Earth Observation (EO). However, compared to the open accessibility and high spatiotemporal coverage of medium-resolution data, the limited acquisition channels for ultra-high-resolution optical RS imagery have constrained the progress of high-resolution remote sensing vision foundation models (RSVFM). As the world's largest sub-meter-level commercial RS satellite constellation, the Jilin-1 constellation possesses abundant sub-meter-level image resources. This study proposes CGEarthEye, a RSVFM framework specifically designed for Jilin-1 satellite characteristics, comprising five backbones with different parameter scales with totaling 2.1 billion parameters. To enhance the representational capacity of the foundation model, we developed JLSSD, the first 15-million-scale multi-temporal self-supervised learning (SSL) dataset featuring global coverage with quarterly temporal sampling within a single year, constructed through multi-level representation clustering and sampling strategies. The framework integrates seasonal contrast, augmentation-based contrast, and masked patch token contrastive strategies for pre-training. Comprehensive evaluations across 10 benchmark datasets covering four typical RS tasks demonstrate that the CGEarthEye consistently achieves state-of-the-art (SOTA) performance. Further analysis reveals CGEarthEye's superior characteristics in feature visualization, model convergence, parameter efficiency, and practical mapping applications. This study anticipates that the exceptional representation capabilities of CGEarthEye will facilitate broader and more efficient applications of Jilin-1 data in traditional EO application.
>
---
#### [new 096] GaussianVLM: Scene-centric 3D Vision-Language Models using Language-aligned Gaussian Splats for Embodied Reasoning and Beyond
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GaussianVLM，解决3D场景理解中的多模态对齐问题，通过语言对齐高斯点云实现场景表示，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00886v1](http://arxiv.org/pdf/2507.00886v1)**

> **作者:** Anna-Maria Halacheva; Jan-Nico Zaech; Xi Wang; Danda Pani Paudel; Luc Van Gool
>
> **摘要:** As multimodal language models advance, their application to 3D scene understanding is a fast-growing frontier, driving the development of 3D Vision-Language Models (VLMs). Current methods show strong dependence on object detectors, introducing processing bottlenecks and limitations in taxonomic flexibility. To address these limitations, we propose a scene-centric 3D VLM for 3D Gaussian splat scenes that employs language- and task-aware scene representations. Our approach directly embeds rich linguistic features into the 3D scene representation by associating language with each Gaussian primitive, achieving early modality alignment. To process the resulting dense representations, we introduce a dual sparsifier that distills them into compact, task-relevant tokens via task-guided and location-guided pathways, producing sparse, task-aware global and local scene tokens. Notably, we present the first Gaussian splatting-based VLM, leveraging photorealistic 3D representations derived from standard RGB images, demonstrating strong generalization: it improves performance of prior 3D VLM five folds, in out-of-the-domain settings.
>
---
#### [new 097] ADAptation: Reconstruction-based Unsupervised Active Learning for Breast Ultrasound Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医学图像诊断任务，解决跨域数据分布差异导致的模型性能下降问题。提出ADAptation框架，通过无监督主动学习选择关键样本，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00474v1](http://arxiv.org/pdf/2507.00474v1)**

> **作者:** Yaofei Duan; Yuhao Huang; Xin Yang; Luyi Han; Xinyu Xie; Zhiyuan Zhu; Ping He; Ka-Hou Chan; Ligang Cui; Sio-Kei Im; Dong Ni; Tao Tan
>
> **备注:** 11 pages, 4 figures, 4 tables. Accepted by conference MICCAI2025
>
> **摘要:** Deep learning-based diagnostic models often suffer performance drops due to distribution shifts between training (source) and test (target) domains. Collecting and labeling sufficient target domain data for model retraining represents an optimal solution, yet is limited by time and scarce resources. Active learning (AL) offers an efficient approach to reduce annotation costs while maintaining performance, but struggles to handle the challenge posed by distribution variations across different datasets. In this study, we propose a novel unsupervised Active learning framework for Domain Adaptation, named ADAptation, which efficiently selects informative samples from multi-domain data pools under limited annotation budget. As a fundamental step, our method first utilizes the distribution homogenization capabilities of diffusion models to bridge cross-dataset gaps by translating target images into source-domain style. We then introduce two key innovations: (a) a hypersphere-constrained contrastive learning network for compact feature clustering, and (b) a dual-scoring mechanism that quantifies and balances sample uncertainty and representativeness. Extensive experiments on four breast ultrasound datasets (three public and one in-house/multi-center) across five common deep classifiers demonstrate that our method surpasses existing strong AL-based competitors, validating its effectiveness and generalization for clinical domain adaptation. The code is available at the anonymized link: https://github.com/miccai25-966/ADAptation.
>
---
#### [new 098] RTMap: Real-Time Recursive Mapping with Change Detection and Localization
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的高精地图构建任务，解决单次遍历方法的局限性，通过多次遍历融合提升地图精度与实时性。**

- **链接: [http://arxiv.org/pdf/2507.00980v1](http://arxiv.org/pdf/2507.00980v1)**

> **作者:** Yuheng Du; Sheng Yang; Lingxuan Wang; Zhenghua Hou; Chengying Cai; Zhitao Tan; Mingxia Chen; Shi-Sheng Huang; Qiang Li
>
> **摘要:** While recent online HD mapping methods relieve burdened offline pipelines and solve map freshness, they remain limited by perceptual inaccuracies, occlusion in dense traffic, and an inability to fuse multi-agent observations. We propose RTMap to enhance these single-traversal methods by persistently crowdsourcing a multi-traversal HD map as a self-evolutional memory. On onboard agents, RTMap simultaneously addresses three core challenges in an end-to-end fashion: (1) Uncertainty-aware positional modeling for HD map elements, (2) probabilistic-aware localization w.r.t. the crowdsourced prior-map, and (3) real-time detection for possible road structural changes. Experiments on several public autonomous driving datasets demonstrate our solid performance on both the prior-aided map quality and the localization accuracy, demonstrating our effectiveness of robustly serving downstream prediction and planning modules while gradually improving the accuracy and freshness of the crowdsourced prior-map asynchronously. Our source-code will be made publicly available at https://github.com/CN-ADLab/RTMap (Camera ready version incorporating reviewer suggestions will be updated soon).
>
---
#### [new 099] TRACE: Temporally Reliable Anatomically-Conditioned 3D CT Generation with Enhanced Efficiency
- **分类: cs.CV**

- **简介: 该论文属于3D医学图像生成任务，旨在解决 anatomical fidelity、axial length 和计算效率问题。提出TRACE框架，利用2D多模态条件扩散模型生成高质量3D图像。**

- **链接: [http://arxiv.org/pdf/2507.00802v1](http://arxiv.org/pdf/2507.00802v1)**

> **作者:** Minye Shao; Xingyu Miao; Haoran Duan; Zeyu Wang; Jingkun Chen; Yawen Huang; Xian Wu; Jingjing Deng; Yang Long; Yefeng Zheng
>
> **备注:** Accepted to MICCAI 2025 (this version is not peer-reviewed; it is the preprint version). MICCAI proceedings DOI will appear here
>
> **摘要:** 3D medical image generation is essential for data augmentation and patient privacy, calling for reliable and efficient models suited for clinical practice. However, current methods suffer from limited anatomical fidelity, restricted axial length, and substantial computational cost, placing them beyond reach for regions with limited resources and infrastructure. We introduce TRACE, a framework that generates 3D medical images with spatiotemporal alignment using a 2D multimodal-conditioned diffusion approach. TRACE models sequential 2D slices as video frame pairs, combining segmentation priors and radiology reports for anatomical alignment, incorporating optical flow to sustain temporal coherence. During inference, an overlapping-frame strategy links frame pairs into a flexible length sequence, reconstructed into a spatiotemporally and anatomically aligned 3D volume. Experimental results demonstrate that TRACE effectively balances computational efficiency with preserving anatomical fidelity and spatiotemporal consistency. Code is available at: https://github.com/VinyehShaw/TRACE.
>
---
#### [new 100] HiT-JEPA: A Hierarchical Self-supervised Trajectory Embedding Framework for Similarity Computation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于轨迹表示学习任务，旨在解决轨迹细粒度与高层语义融合的问题。提出HiT-JEPA框架，通过分层结构捕捉多尺度轨迹特征。**

- **链接: [http://arxiv.org/pdf/2507.00028v1](http://arxiv.org/pdf/2507.00028v1)**

> **作者:** Lihuan Li; Hao Xue; Shuang Ao; Yang Song; Flora Salim
>
> **摘要:** The representation of urban trajectory data plays a critical role in effectively analyzing spatial movement patterns. Despite considerable progress, the challenge of designing trajectory representations that can capture diverse and complementary information remains an open research problem. Existing methods struggle in incorporating trajectory fine-grained details and high-level summary in a single model, limiting their ability to attend to both long-term dependencies while preserving local nuances. To address this, we propose HiT-JEPA (Hierarchical Interactions of Trajectory Semantics via a Joint Embedding Predictive Architecture), a unified framework for learning multi-scale urban trajectory representations across semantic abstraction levels. HiT-JEPA adopts a three-layer hierarchy that progressively captures point-level fine-grained details, intermediate patterns, and high-level trajectory abstractions, enabling the model to integrate both local dynamics and global semantics in one coherent structure. Extensive experiments on multiple real-world datasets for trajectory similarity computation show that HiT-JEPA's hierarchical design yields richer, multi-scale representations. Code is available at: https://anonymous.4open.science/r/HiT-JEPA.
>
---
#### [new 101] Stable Tracking of Eye Gaze Direction During Ophthalmic Surgery
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文属于眼动追踪任务，旨在解决手术中眼位估计不准确的问题。通过结合机器学习与传统算法，实现稳定的眼球定位与跟踪。**

- **链接: [http://arxiv.org/pdf/2507.00635v1](http://arxiv.org/pdf/2507.00635v1)**

> **作者:** Tinghe Hong; Shenlin Cai; Boyang Li; Kai Huang
>
> **备注:** Accepted by ICRA 2025
>
> **摘要:** Ophthalmic surgical robots offer superior stability and precision by reducing the natural hand tremors of human surgeons, enabling delicate operations in confined surgical spaces. Despite the advancements in developing vision- and force-based control methods for surgical robots, preoperative navigation remains heavily reliant on manual operation, limiting the consistency and increasing the uncertainty. Existing eye gaze estimation techniques in the surgery, whether traditional or deep learning-based, face challenges including dependence on additional sensors, occlusion issues in surgical environments, and the requirement for facial detection. To address these limitations, this study proposes an innovative eye localization and tracking method that combines machine learning with traditional algorithms, eliminating the requirements of landmarks and maintaining stable iris detection and gaze estimation under varying lighting and shadow conditions. Extensive real-world experiment results show that our proposed method has an average estimation error of 0.58 degrees for eye orientation estimation and 2.08-degree average control error for the robotic arm's movement based on the calculated orientation.
>
---
#### [new 102] Advancing Lung Disease Diagnosis in 3D CT Scans
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于肺部疾病诊断任务，旨在提升3D CT扫描中肺部疾病的准确诊断。通过去除非肺区域、使用ResNeSt50和加权交叉熵损失，提高了模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00993v1](http://arxiv.org/pdf/2507.00993v1)**

> **作者:** Qingqiu Li; Runtian Yuan; Junlin Hou; Jilan Xu; Yuejie Zhang; Rui Feng; Hao Chen
>
> **摘要:** To enable more accurate diagnosis of lung disease in chest CT scans, we propose a straightforward yet effective model. Firstly, we analyze the characteristics of 3D CT scans and remove non-lung regions, which helps the model focus on lesion-related areas and reduces computational cost. We adopt ResNeSt50 as a strong feature extractor, and use a weighted cross-entropy loss to mitigate class imbalance, especially for the underrepresented squamous cell carcinoma category. Our model achieves a Macro F1 Score of 0.80 on the validation set of the Fair Disease Diagnosis Challenge, demonstrating its strong performance in distinguishing between different lung conditions.
>
---
#### [new 103] Accurate and Efficient Fetal Birth Weight Estimation from 3D Ultrasound
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于胎儿体重估计任务，旨在解决传统方法效率低、依赖操作者及精度不足的问题。提出一种结合多尺度特征融合与合成样本学习的新方法，提升3D超声图像的胎儿体重预测准确性。**

- **链接: [http://arxiv.org/pdf/2507.00398v1](http://arxiv.org/pdf/2507.00398v1)**

> **作者:** Jian Wang; Qiongying Ni; Hongkui Yu; Ruixuan Yao; Jinqiao Ying; Bin Zhang; Xingyi Yang; Jin Peng; Jiongquan Chen; Junxuan Yu; Wenlong Shi; Chaoyu Chen; Zhongnuo Yan; Mingyuan Luo; Gaocheng Cai; Dong Ni; Jing Lu; Xin Yang
>
> **备注:** Accepted by MICCAI 2025
>
> **摘要:** Accurate fetal birth weight (FBW) estimation is essential for optimizing delivery decisions and reducing perinatal mortality. However, clinical methods for FBW estimation are inefficient, operator-dependent, and challenging to apply in cases of complex fetal anatomy. Existing deep learning methods are based on 2D standard ultrasound (US) images or videos that lack spatial information, limiting their prediction accuracy. In this study, we propose the first method for directly estimating FBW from 3D fetal US volumes. Our approach integrates a multi-scale feature fusion network (MFFN) and a synthetic sample-based learning framework (SSLF). The MFFN effectively extracts and fuses multi-scale features under sparse supervision by incorporating channel attention, spatial attention, and a ranking-based loss function. SSLF generates synthetic samples by simply combining fetal head and abdomen data from different fetuses, utilizing semi-supervised learning to improve prediction performance. Experimental results demonstrate that our method achieves superior performance, with a mean absolute error of $166.4\pm155.9$ $g$ and a mean absolute percentage error of $5.1\pm4.6$%, outperforming existing methods and approaching the accuracy of a senior doctor. Code is available at: https://github.com/Qioy-i/EFW.
>
---
#### [new 104] Real-Time Guidewire Tip Tracking Using a Siamese Network for Image-Guided Endovascular Procedures
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决血管内手术中导丝尖端实时跟踪问题，提出一种基于双注意力机制的Siamese网络框架，提升跟踪精度与速度。**

- **链接: [http://arxiv.org/pdf/2507.00051v1](http://arxiv.org/pdf/2507.00051v1)**

> **作者:** Tianliang Yao; Zhiqiang Pei; Yong Li; Yixuan Yuan; Peng Qi
>
> **备注:** This paper has been accepted by Advanced Intelligent Systems
>
> **摘要:** An ever-growing incorporation of AI solutions into clinical practices enhances the efficiency and effectiveness of healthcare services. This paper focuses on guidewire tip tracking tasks during image-guided therapy for cardiovascular diseases, aiding physicians in improving diagnostic and therapeutic quality. A novel tracking framework based on a Siamese network with dual attention mechanisms combines self- and cross-attention strategies for robust guidewire tip tracking. This design handles visual ambiguities, tissue deformations, and imaging artifacts through enhanced spatial-temporal feature learning. Validation occurred on 3 randomly selected clinical digital subtraction angiography (DSA) sequences from a dataset of 15 sequences, covering multiple interventional scenarios. The results indicate a mean localization error of 0.421 $\pm$ 0.138 mm, with a maximum error of 1.736 mm, and a mean Intersection over Union (IoU) of 0.782. The framework maintains an average processing speed of 57.2 frames per second, meeting the temporal demands of endovascular imaging. Further validations with robotic platforms for automating diagnostics and therapies in clinical routines yielded tracking errors of 0.708 $\pm$ 0.695 mm and 0.148 $\pm$ 0.057 mm in two distinct experimental scenarios.
>
---
#### [new 105] Evo-0: Vision-Language-Action Model with Implicit Spatial Understanding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决VLA模型缺乏精确空间理解的问题。通过引入隐式3D几何特征提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00416v1](http://arxiv.org/pdf/2507.00416v1)**

> **作者:** Tao Lin; Gen Li; Yilei Zhong; Yanwen Zou; Bo Zhao
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising framework for enabling generalist robots capable of perceiving, reasoning, and acting in the real world. These models usually build upon pretrained Vision-Language Models (VLMs), which excel at semantic understanding due to large-scale text pretraining. However, VLMs typically lack precise spatial understanding capabilities, as they are primarily tuned on 2D image-text pairs without 3D supervision. To address this limitation, recent approaches have incorporated explicit 3D inputs such as point clouds or depth maps, but this necessitates additional depth sensors or defective estimation. In contrast, our work introduces a plug-and-play module that implicitly injects 3D geometry features into VLA models by leveraging an off-the-shelf visual geometry foundation models. We design five spatially challenging tasks that require precise spatial understanding ability to validate effectiveness of our method. Extensive evaluations show that our method significantly improves the performance of state-of-the-art VLA models across diverse scenarios.
>
---
#### [new 106] Deep learning-based segmentation of T1 and T2 cardiac MRI maps for automated disease detection
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决手动分割的主观性和简化特征带来的问题。通过深度学习和多特征融合提升心脏MRI疾病检测精度。**

- **链接: [http://arxiv.org/pdf/2507.00903v1](http://arxiv.org/pdf/2507.00903v1)**

> **作者:** Andreea Bianca Popescu; Andreas Seitz; Heiko Mahrholdt; Jens Wetzl; Athira Jacob; Lucian Mihai Itu; Constantin Suciu; Teodora Chitiboi
>
> **备注:** This work has been submitted for consideration at European Radiology (Springer). Upon acceptance, this preprint will be updated with the journal reference
>
> **摘要:** Objectives Parametric tissue mapping enables quantitative cardiac tissue characterization but is limited by inter-observer variability during manual delineation. Traditional approaches relying on average relaxation values and single cutoffs may oversimplify myocardial complexity. This study evaluates whether deep learning (DL) can achieve segmentation accuracy comparable to inter-observer variability, explores the utility of statistical features beyond mean T1/T2 values, and assesses whether machine learning (ML) combining multiple features enhances disease detection. Materials & Methods T1 and T2 maps were manually segmented. The test subset was independently annotated by two observers, and inter-observer variability was assessed. A DL model was trained to segment left ventricle blood pool and myocardium. Average (A), lower quartile (LQ), median (M), and upper quartile (UQ) were computed for the myocardial pixels and employed in classification by applying cutoffs or in ML. Dice similarity coefficient (DICE) and mean absolute percentage error evaluated segmentation performance. Bland-Altman plots assessed inter-user and model-observer agreement. Receiver operating characteristic analysis determined optimal cutoffs. Pearson correlation compared features from model and manual segmentations. F1-score, precision, and recall evaluated classification performance. Wilcoxon test assessed differences between classification methods, with p < 0.05 considered statistically significant. Results 144 subjects were split into training (100), validation (15) and evaluation (29) subsets. Segmentation model achieved a DICE of 85.4%, surpassing inter-observer agreement. Random forest applied to all features increased F1-score (92.7%, p < 0.001). Conclusion DL facilitates segmentation of T1/ T2 maps. Combining multiple features with ML improves disease detection.
>
---
#### [new 107] GANs Secretly Perform Approximate Bayesian Model Selection
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文属于生成模型研究任务，旨在解释GANs的成功与局限性。通过将其视为贝叶斯神经网络，提出优化策略以提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00651v1](http://arxiv.org/pdf/2507.00651v1)**

> **作者:** Maurizio Filippone; Marius P. Linhard
>
> **摘要:** Generative Adversarial Networks (GANs) are popular and successful generative models. Despite their success, optimization is notoriously challenging and they require regularization against overfitting. In this work, we explain the success and limitations of GANs by interpreting them as probabilistic generative models. This interpretation enables us to view GANs as Bayesian neural networks with partial stochasticity, allowing us to establish conditions of universal approximation. We can then cast the adversarial-style optimization of several variants of GANs as the optimization of a proxy for the marginal likelihood. Taking advantage of the connection between marginal likelihood optimization and Occam's razor, we can define regularization and optimization strategies to smooth the loss landscape and search for solutions with minimum description length, which are associated with flat minima and good generalization. The results on a wide range of experiments indicate that these strategies lead to performance improvements and pave the way to a deeper understanding of regularization strategies for GANs.
>
---
#### [new 108] Mind the Detail: Uncovering Clinically Relevant Image Details in Accelerated MRI with Semantically Diverse Reconstructions
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像重建任务，旨在解决加速MRI中遗漏关键临床信息的问题。通过生成语义多样重建提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2507.00670v1](http://arxiv.org/pdf/2507.00670v1)**

> **作者:** Jan Nikolas Morshuis; Christian Schlarmann; Thomas Küstner; Christian F. Baumgartner; Matthias Hein
>
> **备注:** MICCAI 2025
>
> **摘要:** In recent years, accelerated MRI reconstruction based on deep learning has led to significant improvements in image quality with impressive results for high acceleration factors. However, from a clinical perspective image quality is only secondary; much more important is that all clinically relevant information is preserved in the reconstruction from heavily undersampled data. In this paper, we show that existing techniques, even when considering resampling for diffusion-based reconstruction, can fail to reconstruct small and rare pathologies, thus leading to potentially wrong diagnosis decisions (false negatives). To uncover the potentially missing clinical information we propose ``Semantically Diverse Reconstructions'' (\SDR), a method which, given an original reconstruction, generates novel reconstructions with enhanced semantic variability while all of them are fully consistent with the measured data. To evaluate \SDR automatically we train an object detector on the fastMRI+ dataset. We show that \SDR significantly reduces the chance of false-negative diagnoses (higher recall) and improves mean average precision compared to the original reconstructions. The code is available on https://github.com/NikolasMorshuis/SDR
>
---
#### [new 109] BadViM: Backdoor Attack against Vision Mamba
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文属于安全任务，研究ViM模型的后门攻击问题。提出BadViM框架，利用RFT和隐藏状态对齐损失实现高效隐蔽攻击。**

- **链接: [http://arxiv.org/pdf/2507.00577v1](http://arxiv.org/pdf/2507.00577v1)**

> **作者:** Yinghao Wu; Liyan Zhang
>
> **摘要:** Vision State Space Models (SSMs), particularly architectures like Vision Mamba (ViM), have emerged as promising alternatives to Vision Transformers (ViTs). However, the security implications of this novel architecture, especially their vulnerability to backdoor attacks, remain critically underexplored. Backdoor attacks aim to embed hidden triggers into victim models, causing the model to misclassify inputs containing these triggers while maintaining normal behavior on clean inputs. This paper investigates the susceptibility of ViM to backdoor attacks by introducing BadViM, a novel backdoor attack framework specifically designed for Vision Mamba. The proposed BadViM leverages a Resonant Frequency Trigger (RFT) that exploits the frequency sensitivity patterns of the victim model to create stealthy, distributed triggers. To maximize attack efficacy, we propose a Hidden State Alignment loss that strategically manipulates the internal representations of model by aligning the hidden states of backdoor images with those of target classes. Extensive experimental results demonstrate that BadViM achieves superior attack success rates while maintaining clean data accuracy. Meanwhile, BadViM exhibits remarkable resilience against common defensive measures, including PatchDrop, PatchShuffle and JPEG compression, which typically neutralize normal backdoor attacks.
>
---
#### [new 110] FreNBRDF: A Frequency-Rectified Neural Material Representation
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于材质建模任务，解决神经材质表示在频域行为不明确的问题。通过引入频域修正损失，提升材质重建与编辑的准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.00476v1](http://arxiv.org/pdf/2507.00476v1)**

> **作者:** Chenliang Zhou; Zheyuan Hu; Cengiz Oztireli
>
> **摘要:** Accurate material modeling is crucial for achieving photorealistic rendering, bridging the gap between computer-generated imagery and real-world photographs. While traditional approaches rely on tabulated BRDF data, recent work has shifted towards implicit neural representations, which offer compact and flexible frameworks for a range of tasks. However, their behavior in the frequency domain remains poorly understood. To address this, we introduce FreNBRDF, a frequency-rectified neural material representation. By leveraging spherical harmonics, we integrate frequency-domain considerations into neural BRDF modeling. We propose a novel frequency-rectified loss, derived from a frequency analysis of neural materials, and incorporate it into a generalizable and adaptive reconstruction and editing pipeline. This framework enhances fidelity, adaptability, and efficiency. Extensive experiments demonstrate that \ours improves the accuracy and robustness of material appearance reconstruction and editing compared to state-of-the-art baselines, enabling more structured and interpretable downstream tasks and applications.
>
---
#### [new 111] Rethink 3D Object Detection from Physical World
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D目标检测任务，解决实时系统中速度与精度的权衡问题。提出L-AP和P-AP新指标，优化模型与硬件选择。**

- **链接: [http://arxiv.org/pdf/2507.00190v1](http://arxiv.org/pdf/2507.00190v1)**

> **作者:** Satoshi Tanaka; Koji Minoda; Fumiya Watanabe; Takamasa Horibe
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** High-accuracy and low-latency 3D object detection is essential for autonomous driving systems. While previous studies on 3D object detection often evaluate performance based on mean average precision (mAP) and latency, they typically fail to address the trade-off between speed and accuracy, such as 60.0 mAP at 100 ms vs 61.0 mAP at 500 ms. A quantitative assessment of the trade-offs between different hardware devices and accelerators remains unexplored, despite being critical for real-time applications. Furthermore, they overlook the impact on collision avoidance in motion planning, for example, 60.0 mAP leading to safer motion planning or 61.0 mAP leading to high-risk motion planning. In this paper, we introduce latency-aware AP (L-AP) and planning-aware AP (P-AP) as new metrics, which consider the physical world such as the concept of time and physical constraints, offering a more comprehensive evaluation for real-time 3D object detection. We demonstrate the effectiveness of our metrics for the entire autonomous driving system using nuPlan dataset, and evaluate 3D object detection models accounting for hardware differences and accelerators. We also develop a state-of-the-art performance model for real-time 3D object detection through latency-aware hyperparameter optimization (L-HPO) using our metrics. Additionally, we quantitatively demonstrate that the assumption "the more point clouds, the better the recognition performance" is incorrect for real-time applications and optimize both hardware and model selection using our metrics.
>
---
#### [new 112] RoboEval: Where Robotic Manipulation Meets Structured and Scalable Evaluation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出RoboEval，用于评估双臂机器人操作性能。解决传统评估指标不足的问题，通过结构化任务和细粒度指标揭示策略缺陷。**

- **链接: [http://arxiv.org/pdf/2507.00435v1](http://arxiv.org/pdf/2507.00435v1)**

> **作者:** Yi Ru Wang; Carter Ung; Grant Tannert; Jiafei Duan; Josephine Li; Amy Le; Rishabh Oswal; Markus Grotz; Wilbert Pumacay; Yuquan Deng; Ranjay Krishna; Dieter Fox; Siddhartha Srinivasa
>
> **备注:** Project page: https://robo-eval.github.io
>
> **摘要:** We present RoboEval, a simulation benchmark and structured evaluation framework designed to reveal the limitations of current bimanual manipulation policies. While prior benchmarks report only binary task success, we show that such metrics often conceal critical weaknesses in policy behavior -- such as poor coordination, slipping during grasping, or asymmetric arm usage. RoboEval introduces a suite of tiered, semantically grounded tasks decomposed into skill-specific stages, with variations that systematically challenge spatial, physical, and coordination capabilities. Tasks are paired with fine-grained diagnostic metrics and 3000+ human demonstrations to support imitation learning. Our experiments reveal that policies with similar success rates diverge in how tasks are executed -- some struggle with alignment, others with temporally consistent bimanual control. We find that behavioral metrics correlate with success in over half of task-metric pairs, and remain informative even when binary success saturates. By pinpointing when and how policies fail, RoboEval enables a deeper, more actionable understanding of robotic manipulation -- and highlights the need for evaluation tools that go beyond success alone.
>
---
#### [new 113] Scope Meets Screen: Lessons Learned in Designing Composite Visualizations for Marksmanship Training Across Skill Levels
- **分类: cs.HC; cs.CV; cs.GR; eess.IV**

- **简介: 该论文属于人机交互任务，旨在提升射击训练效果。通过设计复合可视化系统，解决教练无法实时观察射手视角的问题，提升不同水平射手的理解与反馈。**

- **链接: [http://arxiv.org/pdf/2507.00333v1](http://arxiv.org/pdf/2507.00333v1)**

> **作者:** Emin Zerman; Jonas Carlsson; Mårten Sjöström
>
> **备注:** 5 pages, accepted at IEEE VIS 2025
>
> **摘要:** Marksmanship practices are required in various professions, including police, military personnel, hunters, as well as sports shooters, such as Olympic shooting, biathlon, and modern pentathlon. The current form of training and coaching is mostly based on repetition, where the coach does not see through the eyes of the shooter, and analysis is limited to stance and accuracy post-session. In this study, we present a shooting visualization system and evaluate its perceived effectiveness for both novice and expert shooters. To achieve this, five composite visualizations were developed using first-person shooting video recordings enriched with overlaid metrics and graphical summaries. These views were evaluated with 10 participants (5 expert marksmen, 5 novices) through a mixed-methods study including shot-count and aiming interpretation tasks, pairwise preference comparisons, and semi-structured interviews. The results show that a dashboard-style composite view, combining raw video with a polar plot and selected graphs, was preferred in 9 of 10 cases and supported understanding across skill levels. The insights gained from this design study point to the broader value of integrating first-person video with visual analytics for coaching, and we suggest directions for applying this approach to other precision-based sports.
>
---
#### [new 114] Multimodal, Multi-Disease Medical Imaging Foundation Model (MerMED-FM)
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学影像多模态任务，旨在解决单模态、单疾病模型准确性不足的问题。通过自监督学习和记忆模块，构建了跨学科的多模态医疗影像基础模型MerMED-FM。**

- **链接: [http://arxiv.org/pdf/2507.00185v1](http://arxiv.org/pdf/2507.00185v1)**

> **作者:** Yang Zhou; Chrystie Wan Ning Quek; Jun Zhou; Yan Wang; Yang Bai; Yuhe Ke; Jie Yao; Laura Gutierrez; Zhen Ling Teo; Darren Shu Jeng Ting; Brian T. Soetikno; Christopher S. Nielsen; Tobias Elze; Zengxiang Li; Linh Le Dinh; Lionel Tim-Ee Cheng; Tran Nguyen Tuan Anh; Chee Leong Cheng; Tien Yin Wong; Nan Liu; Iain Beehuat Tan; Tony Kiat Hon Lim; Rick Siow Mong Goh; Yong Liu; Daniel Shu Wei Ting
>
> **备注:** 42 pages, 3 composite figures, 4 tables
>
> **摘要:** Current artificial intelligence models for medical imaging are predominantly single modality and single disease. Attempts to create multimodal and multi-disease models have resulted in inconsistent clinical accuracy. Furthermore, training these models typically requires large, labour-intensive, well-labelled datasets. We developed MerMED-FM, a state-of-the-art multimodal, multi-specialty foundation model trained using self-supervised learning and a memory module. MerMED-FM was trained on 3.3 million medical images from over ten specialties and seven modalities, including computed tomography (CT), chest X-rays (CXR), ultrasound (US), pathology patches, color fundus photography (CFP), optical coherence tomography (OCT) and dermatology images. MerMED-FM was evaluated across multiple diseases and compared against existing foundational models. Strong performance was achieved across all modalities, with AUROCs of 0.988 (OCT); 0.982 (pathology); 0.951 (US); 0.943 (CT); 0.931 (skin); 0.894 (CFP); 0.858 (CXR). MerMED-FM has the potential to be a highly adaptable, versatile, cross-specialty foundation model that enables robust medical imaging interpretation across diverse medical disciplines.
>
---
#### [new 115] MuteSwap: Silent Face-based Voice Conversion
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **简介: 该论文提出MuteSwap，解决无声视频中的语音转换任务（SFVC），通过视觉信息生成目标说话人语音，克服音频不可用的挑战。**

- **链接: [http://arxiv.org/pdf/2507.00498v1](http://arxiv.org/pdf/2507.00498v1)**

> **作者:** Yifan Liu; Yu Fang; Zhouhan Lin
>
> **摘要:** Conventional voice conversion modifies voice characteristics from a source speaker to a target speaker, relying on audio input from both sides. However, this process becomes infeasible when clean audio is unavailable, such as in silent videos or noisy environments. In this work, we focus on the task of Silent Face-based Voice Conversion (SFVC), which does voice conversion entirely from visual inputs. i.e., given images of a target speaker and a silent video of a source speaker containing lip motion, SFVC generates speech aligning the identity of the target speaker while preserving the speech content in the source silent video. As this task requires generating intelligible speech and converting identity using only visual cues, it is particularly challenging. To address this, we introduce MuteSwap, a novel framework that employs contrastive learning to align cross-modality identities and minimize mutual information to separate shared visual features. Experimental results show that MuteSwap achieves impressive performance in both speech synthesis and identity conversion, especially under noisy conditions where methods dependent on audio input fail to produce intelligible results, demonstrating both the effectiveness of our training approach and the feasibility of SFVC.
>
---
#### [new 116] Automated anatomy-based post-processing reduces false positives and improved interpretability of deep learning intracranial aneurysm detection
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在降低深度学习检测颅内动脉瘤的误报率。通过引入基于解剖结构的后处理方法，有效减少了假阳性，提升了模型的可解释性和临床适用性。**

- **链接: [http://arxiv.org/pdf/2507.00832v1](http://arxiv.org/pdf/2507.00832v1)**

> **作者:** Jisoo Kim; Chu-Hsuan Lin; Alberto Ceballos-Arroyo; Ping Liu; Huaizu Jiang; Shrikanth Yadav; Qi Wan; Lei Qin; Geoffrey S Young
>
> **摘要:** Introduction: Deep learning (DL) models can help detect intracranial aneurysms on CTA, but high false positive (FP) rates remain a barrier to clinical translation, despite improvement in model architectures and strategies like detection threshold tuning. We employed an automated, anatomy-based, heuristic-learning hybrid artery-vein segmentation post-processing method to further reduce FPs. Methods: Two DL models, CPM-Net and a deformable 3D convolutional neural network-transformer hybrid (3D-CNN-TR), were trained with 1,186 open-source CTAs (1,373 annotated aneurysms), and evaluated with 143 held-out private CTAs (218 annotated aneurysms). Brain, artery, vein, and cavernous venous sinus (CVS) segmentation masks were applied to remove possible FPs in the DL outputs that overlapped with: (1) brain mask; (2) vein mask; (3) vein more than artery masks; (4) brain plus vein mask; (5) brain plus vein more than artery masks. Results: CPM-Net yielded 139 true-positives (TP); 79 false-negative (FN); 126 FP. 3D-CNN-TR yielded 179 TP; 39 FN; 182 FP. FPs were commonly extracranial (CPM-Net 27.3%; 3D-CNN-TR 42.3%), venous (CPM-Net 56.3%; 3D-CNN-TR 29.1%), arterial (CPM-Net 11.9%; 3D-CNN-TR 53.3%), and non-vascular (CPM-Net 25.4%; 3D-CNN-TR 9.3%) structures. Method 5 performed best, reducing CPM-Net FP by 70.6% (89/126) and 3D-CNN-TR FP by 51.6% (94/182), without reducing TP, lowering the FP/case rate from 0.88 to 0.26 for CPM-NET, and from 1.27 to 0.62 for the 3D-CNN-TR. Conclusion: Anatomy-based, interpretable post-processing can improve DL-based aneurysm detection model performance. More broadly, automated, domain-informed, hybrid heuristic-learning processing holds promise for improving the performance and clinical acceptance of aneurysm detection models.
>
---
#### [new 117] Bridging Classical and Learning-based Iterative Registration through Deep Equilibrium Models
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决传统方法收敛性差和学习方法内存消耗大的问题。提出DEQReg框架，利用深度均衡模型实现稳定高效配准。**

- **链接: [http://arxiv.org/pdf/2507.00582v1](http://arxiv.org/pdf/2507.00582v1)**

> **作者:** Yi Zhang; Yidong Zhao; Qian Tao
>
> **备注:** Submitted version. Accepted by MICCAI 2025
>
> **摘要:** Deformable medical image registration is traditionally formulated as an optimization problem. While classical methods solve this problem iteratively, recent learning-based approaches use recurrent neural networks (RNNs) to mimic this process by unrolling the prediction of deformation fields in a fixed number of steps. However, classical methods typically converge after sufficient iterations, but learning-based unrolling methods lack a theoretical convergence guarantee and show instability empirically. In addition, unrolling methods have a practical bottleneck at training time: GPU memory usage grows linearly with the unrolling steps due to backpropagation through time (BPTT). To address both theoretical and practical challenges, we propose DEQReg, a novel registration framework based on Deep Equilibrium Models (DEQ), which formulates registration as an equilibrium-seeking problem, establishing a natural connection between classical optimization and learning-based unrolling methods. DEQReg maintains constant memory usage, enabling theoretically unlimited iteration steps. Through extensive evaluation on the public brain MRI and lung CT datasets, we show that DEQReg can achieve competitive registration performance, while substantially reducing memory consumption compared to state-of-the-art unrolling methods. We also reveal an intriguing phenomenon: the performance of existing unrolling methods first increases slightly then degrades irreversibly when the inference steps go beyond the training configuration. In contrast, DEQReg achieves stable convergence with its inbuilt equilibrium-seeking mechanism, bridging the gap between classical optimization-based and modern learning-based registration methods.
>
---
#### [new 118] Gradient-based Fine-Tuning through Pre-trained Model Regularization
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于模型微调任务，旨在降低计算和存储成本。提出GRFT方法，通过梯度和正则化高效更新参数，显著减少需调整的参数比例。**

- **链接: [http://arxiv.org/pdf/2507.00016v1](http://arxiv.org/pdf/2507.00016v1)**

> **作者:** Xuanbo Liu; Liu Liu; Fuxiang Wu; Fusheng Hao; Xianglong Liu
>
> **摘要:** Large pre-trained models have demonstrated extensive applications across various fields. However, fine-tuning these models for specific downstream tasks demands significant computational resources and storage. One fine-tuning method, gradient-based parameter selection (GPS), focuses on fine-tuning only the parameters with high gradients in each neuron, thereby reducing the number of training parameters. Nevertheless, this approach increases computational resource requirements and storage demands. In this paper, we propose an efficient gradient-based and regularized fine-tuning method (GRFT) that updates the rows or columns of the weight matrix. We theoretically demonstrate that the rows or columns with the highest sum of squared gradients are optimal for updating. This strategy effectively reduces storage overhead and improves the efficiency of parameter selection. Additionally, we incorporate regularization to enhance knowledge transfer from the pre-trained model. GRFT achieves state-of-the-art performance, surpassing existing methods such as GPS, Adapter Tuning, and LoRA. Notably, GRFT requires updating only 1.22% and 0.30% of the total parameters on FGVC and VTAB datasets, respectively, demonstrating its high efficiency and effectiveness. The source code will be released soon.
>
---
#### [new 119] Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，解决无需物理演示的机器人学习问题。通过生成视频并提取轨迹，实现机器人复杂操作。**

- **链接: [http://arxiv.org/pdf/2507.00990v1](http://arxiv.org/pdf/2507.00990v1)**

> **作者:** Shivansh Patel; Shraddhaa Mohan; Hanlin Mai; Unnat Jain; Svetlana Lazebnik; Yunzhu Li
>
> **备注:** Project Page: https://rigvid-robot.github.io/
>
> **摘要:** This work introduces Robots Imitating Generated Videos (RIGVid), a system that enables robots to perform complex manipulation tasks--such as pouring, wiping, and mixing--purely by imitating AI-generated videos, without requiring any physical demonstrations or robot-specific training. Given a language command and an initial scene image, a video diffusion model generates potential demonstration videos, and a vision-language model (VLM) automatically filters out results that do not follow the command. A 6D pose tracker then extracts object trajectories from the video, and the trajectories are retargeted to the robot in an embodiment-agnostic fashion. Through extensive real-world evaluations, we show that filtered generated videos are as effective as real demonstrations, and that performance improves with generation quality. We also show that relying on generated videos outperforms more compact alternatives such as keypoint prediction using VLMs, and that strong 6D pose tracking outperforms other ways to extract trajectories, such as dense feature point tracking. These findings suggest that videos produced by a state-of-the-art off-the-shelf model can offer an effective source of supervision for robotic manipulation.
>
---
#### [new 120] DiMo-GUI: Advancing Test-time Scaling in GUI Grounding via Modality-Aware Visual Reasoning
- **分类: cs.AI; cs.CV; cs.HC**

- **简介: 该论文属于GUI接地任务，解决视觉元素多样性和语言模糊性问题。提出DiMo-GUI框架，通过模态分离和区域聚焦推理提升接地效果。**

- **链接: [http://arxiv.org/pdf/2507.00008v1](http://arxiv.org/pdf/2507.00008v1)**

> **作者:** Hang Wu; Hongkai Chen; Yujun Cai; Chang Liu; Qingwen Ye; Ming-Hsuan Yang; Yiwei Wang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Grounding natural language queries in graphical user interfaces (GUIs) poses unique challenges due to the diversity of visual elements, spatial clutter, and the ambiguity of language. In this paper, we introduce DiMo-GUI, a training-free framework for GUI grounding that leverages two core strategies: dynamic visual grounding and modality-aware optimization. Instead of treating the GUI as a monolithic image, our method splits the input into textual elements and iconic elements, allowing the model to reason over each modality independently using general-purpose vision-language models. When predictions are ambiguous or incorrect, DiMo-GUI dynamically focuses attention by generating candidate focal regions centered on the model's initial predictions and incrementally zooms into subregions to refine the grounding result. This hierarchical refinement process helps disambiguate visually crowded layouts without the need for additional training or annotations. We evaluate our approach on standard GUI grounding benchmarks and demonstrate consistent improvements over baseline inference pipelines, highlighting the effectiveness of combining modality separation with region-focused reasoning.
>
---
#### [new 121] Prompt2SegCXR:Prompt to Segment All Organs and Diseases in Chest X-rays
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决多器官和多疾病在胸部X光中的交互式分割问题。工作包括生成标注数据和提出轻量级模型Prompt2SegCXR。**

- **链接: [http://arxiv.org/pdf/2507.00673v1](http://arxiv.org/pdf/2507.00673v1)**

> **作者:** Abduz Zami; Shadman Sobhan; Rounaq Hossain; Md. Sawran Sorker; Mohiuddin Ahmed; Md. Redwan Hossain
>
> **备注:** 29 Pages
>
> **摘要:** Image segmentation plays a vital role in the medical field by isolating organs or regions of interest from surrounding areas. Traditionally, segmentation models are trained on a specific organ or a disease, limiting their ability to handle other organs and diseases. At present, few advanced models can perform multi-organ or multi-disease segmentation, offering greater flexibility. Also, recently, prompt-based image segmentation has gained attention as a more flexible approach. It allows models to segment areas based on user-provided prompts. Despite these advances, there has been no dedicated work on prompt-based interactive multi-organ and multi-disease segmentation, especially for Chest X-rays. This work presents two main contributions: first, generating doodle prompts by medical experts of a collection of datasets from multiple sources with 23 classes, including 6 organs and 17 diseases, specifically designed for prompt-based Chest X-ray segmentation. Second, we introduce Prompt2SegCXR, a lightweight model for accurately segmenting multiple organs and diseases from Chest X-rays. The model incorporates multi-stage feature fusion, enabling it to combine features from various network layers for better spatial and semantic understanding, enhancing segmentation accuracy. Compared to existing pre-trained models for prompt-based image segmentation, our model scores well, providing a reliable solution for segmenting Chest X-rays based on user prompts.
>
---
#### [new 122] Medical Image Segmentation Using Advanced Unet: VMSE-Unet and VM-Unet CBAM+
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在提升分割精度与效率。通过引入SE和CBAM模块改进U-Net架构，提出VMSE-Unet和VM-Unet CBAM+模型，显著提高性能。**

- **链接: [http://arxiv.org/pdf/2507.00511v1](http://arxiv.org/pdf/2507.00511v1)**

> **作者:** Sayandeep Kanrar; Raja Piyush; Qaiser Razi; Debanshi Chakraborty; Vikas Hassija; GSS Chalapathi
>
> **备注:** under review
>
> **摘要:** In this paper, we present the VMSE U-Net and VM-Unet CBAM+ model, two cutting-edge deep learning architectures designed to enhance medical image segmentation. Our approach integrates Squeeze-and-Excitation (SE) and Convolutional Block Attention Module (CBAM) techniques into the traditional VM U-Net framework, significantly improving segmentation accuracy, feature localization, and computational efficiency. Both models show superior performance compared to the baseline VM-Unet across multiple datasets. Notably, VMSEUnet achieves the highest accuracy, IoU, precision, and recall while maintaining low loss values. It also exhibits exceptional computational efficiency with faster inference times and lower memory usage on both GPU and CPU. Overall, the study suggests that the enhanced architecture VMSE-Unet is a valuable tool for medical image analysis. These findings highlight its potential for real-world clinical applications, emphasizing the importance of further research to optimize accuracy, robustness, and computational efficiency.
>
---
#### [new 123] RaGNNarok: A Light-Weight Graph Neural Network for Enhancing Radar Point Clouds on Unmanned Ground Vehicles
- **分类: cs.RO; cs.AR; cs.CV; cs.LG**

- **简介: 该论文属于机器人感知任务，旨在解决雷达点云稀疏和噪声问题。提出RaGNNarok框架，提升雷达数据质量，适用于低成本移动机器人。**

- **链接: [http://arxiv.org/pdf/2507.00937v1](http://arxiv.org/pdf/2507.00937v1)**

> **作者:** David Hunt; Shaocheng Luo; Spencer Hallyburton; Shafii Nillongo; Yi Li; Tingjun Chen; Miroslav Pajic
>
> **备注:** 8 pages, accepted by IROS 2025
>
> **摘要:** Low-cost indoor mobile robots have gained popularity with the increasing adoption of automation in homes and commercial spaces. However, existing lidar and camera-based solutions have limitations such as poor performance in visually obscured environments, high computational overhead for data processing, and high costs for lidars. In contrast, mmWave radar sensors offer a cost-effective and lightweight alternative, providing accurate ranging regardless of visibility. However, existing radar-based localization suffers from sparse point cloud generation, noise, and false detections. Thus, in this work, we introduce RaGNNarok, a real-time, lightweight, and generalizable graph neural network (GNN)-based framework to enhance radar point clouds, even in complex and dynamic environments. With an inference time of just 7.3 ms on the low-cost Raspberry Pi 5, RaGNNarok runs efficiently even on such resource-constrained devices, requiring no additional computational resources. We evaluate its performance across key tasks, including localization, SLAM, and autonomous navigation, in three different environments. Our results demonstrate strong reliability and generalizability, making RaGNNarok a robust solution for low-cost indoor mobile robots.
>
---
#### [new 124] Research on Improving the High Precision and Lightweight Diabetic Retinopathy Detection of YOLOv8n
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像检测任务，旨在提升糖尿病视网膜病变的检测精度与模型轻量化。通过改进YOLOv8n，提出YOLO-KFG模型，解决微小病灶识别难和模型部署受限的问题。**

- **链接: [http://arxiv.org/pdf/2507.00780v1](http://arxiv.org/pdf/2507.00780v1)**

> **作者:** Fei Yuhuan; Sun Xufei; Zang Ran; Wang Gengchen; Su Meng; Liu Fenghao
>
> **备注:** in Chinese language
>
> **摘要:** Early detection and diagnosis of diabetic retinopathy is one of the current research focuses in ophthalmology. However, due to the subtle features of micro-lesions and their susceptibility to background interference, ex-isting detection methods still face many challenges in terms of accuracy and robustness. To address these issues, a lightweight and high-precision detection model based on the improved YOLOv8n, named YOLO-KFG, is proposed. Firstly, a new dynamic convolution KWConv and C2f-KW module are designed to improve the backbone network, enhancing the model's ability to perceive micro-lesions. Secondly, a fea-ture-focused diffusion pyramid network FDPN is designed to fully integrate multi-scale context information, further improving the model's ability to perceive micro-lesions. Finally, a lightweight shared detection head GSDHead is designed to reduce the model's parameter count, making it more deployable on re-source-constrained devices. Experimental results show that compared with the base model YOLOv8n, the improved model reduces the parameter count by 20.7%, increases mAP@0.5 by 4.1%, and improves the recall rate by 7.9%. Compared with single-stage mainstream algorithms such as YOLOv5n and YOLOv10n, YOLO-KFG demonstrates significant advantages in both detection accuracy and efficiency.
>
---
#### [new 125] Tunable Wavelet Unit based Convolutional Neural Network in Optical Coherence Tomography Analysis Enhancement for Classifying Type of Epiretinal Membrane Surgery
- **分类: eess.IV; cs.CV; eess.SP**

- **简介: 该论文属于医学图像分类任务，旨在准确区分视网膜手术类型。通过改进的CNN模型和可调小波单元提升分类性能。**

- **链接: [http://arxiv.org/pdf/2507.00743v1](http://arxiv.org/pdf/2507.00743v1)**

> **作者:** An Le; Nehal Mehta; William Freeman; Ines Nagel; Melanie Tran; Anna Heinke; Akshay Agnihotri; Lingyun Cheng; Dirk-Uwe Bartsch; Hung Nguyen; Truong Nguyen; Cheolhong An
>
> **摘要:** In this study, we developed deep learning-based method to classify the type of surgery performed for epiretinal membrane (ERM) removal, either internal limiting membrane (ILM) removal or ERM-alone removal. Our model, based on the ResNet18 convolutional neural network (CNN) architecture, utilizes postoperative optical coherence tomography (OCT) center scans as inputs. We evaluated the model using both original scans and scans preprocessed with energy crop and wavelet denoising, achieving 72% accuracy on preprocessed inputs, outperforming the 66% accuracy achieved on original scans. To further improve accuracy, we integrated tunable wavelet units with two key adaptations: Orthogonal Lattice-based Wavelet Units (OrthLatt-UwU) and Perfect Reconstruction Relaxation-based Wavelet Units (PR-Relax-UwU). These units allowed the model to automatically adjust filter coefficients during training and were incorporated into downsampling, stride-two convolution, and pooling layers, enhancing its ability to distinguish between ERM-ILM removal and ERM-alone removal, with OrthLattUwU boosting accuracy to 76% and PR-Relax-UwU increasing performance to 78%. Performance comparisons showed that our AI model outperformed a trained human grader, who achieved only 50% accuracy in classifying the removal surgery types from postoperative OCT scans. These findings highlight the potential of CNN based models to improve clinical decision-making by providing more accurate and reliable classifications. To the best of our knowledge, this is the first work to employ tunable wavelets for classifying different types of ERM removal surgery.
>
---
#### [new 126] Twill: Scheduling Compound AI Systems on Heterogeneous Mobile Edge Platforms
- **分类: cs.MA; cs.AI; cs.CV; cs.PF**

- **简介: 该论文属于边缘计算任务，解决cAI系统在异构平台上的调度问题。通过Twill框架优化任务映射、优先级和功耗，降低推理延迟。**

- **链接: [http://arxiv.org/pdf/2507.00491v1](http://arxiv.org/pdf/2507.00491v1)**

> **作者:** Zain Taufique; Aman Vyas; Antonio Miele; Pasi Liljeberg; Anil Kanduri
>
> **备注:** 9 Pages, 9 Figures, Accepted in International Conference on Computer-Aided Design (ICCAD) 2025
>
> **摘要:** Compound AI (cAI) systems chain multiple AI models to solve complex problems. cAI systems are typically composed of deep neural networks (DNNs), transformers, and large language models (LLMs), exhibiting a high degree of computational diversity and dynamic workload variation. Deploying cAI services on mobile edge platforms poses a significant challenge in scheduling concurrent DNN-transformer inference tasks, which arrive dynamically in an unknown sequence. Existing mobile edge AI inference strategies manage multi-DNN or transformer-only workloads, relying on design-time profiling, and cannot handle concurrent inference of DNNs and transformers required by cAI systems. In this work, we address the challenge of scheduling cAI systems on heterogeneous mobile edge platforms. We present Twill, a run-time framework to handle concurrent inference requests of cAI workloads through task affinity-aware cluster mapping and migration, priority-aware task freezing/unfreezing, and DVFS, while minimizing inference latency within power budgets. We implement and deploy our Twill framework on the Nvidia Jetson Orin NX platform. We evaluate Twill against state-of-the-art edge AI inference techniques over contemporary DNNs and LLMs, reducing inference latency by 54% on average, while honoring power budgets.
>
---
#### [new 127] DMCIE: Diffusion Model with Concatenation of Inputs and Errors to Improve the Accuracy of the Segmentation of Brain Tumors in MRI Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提高脑肿瘤MRI图像的分割精度。通过结合输入与误差信息的扩散模型，提升分割准确性。**

- **链接: [http://arxiv.org/pdf/2507.00983v1](http://arxiv.org/pdf/2507.00983v1)**

> **作者:** Sara Yavari; Rahul Nitin Pandya; Jacob Furst
>
> **摘要:** Accurate segmentation of brain tumors in MRI scans is essential for reliable clinical diagnosis and effective treatment planning. Recently, diffusion models have demonstrated remarkable effectiveness in image generation and segmentation tasks. This paper introduces a novel approach to corrective segmentation based on diffusion models. We propose DMCIE (Diffusion Model with Concatenation of Inputs and Errors), a novel framework for accurate brain tumor segmentation in multi-modal MRI scans. We employ a 3D U-Net to generate an initial segmentation mask, from which an error map is generated by identifying the differences between the prediction and the ground truth. The error map, concatenated with the original MRI images, are used to guide a diffusion model. Using multimodal MRI inputs (T1, T1ce, T2, FLAIR), DMCIE effectively enhances segmentation accuracy by focusing on misclassified regions, guided by the original inputs. Evaluated on the BraTS2020 dataset, DMCIE outperforms several state-of-the-art diffusion-based segmentation methods, achieving a Dice Score of 93.46 and an HD95 of 5.94 mm. These results highlight the effectiveness of error-guided diffusion in producing precise and reliable brain tumor segmentations.
>
---
#### [new 128] Towards 3D Semantic Image Synthesis for Medical Imaging
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决数据不足与隐私保护问题。通过提出Med-LSDM模型，实现3D语义图像合成，提升数据增强效果。**

- **链接: [http://arxiv.org/pdf/2507.00206v1](http://arxiv.org/pdf/2507.00206v1)**

> **作者:** Wenwu Tang; Khaled Seyam; Bin Yang
>
> **摘要:** In the medical domain, acquiring large datasets is challenging due to both accessibility issues and stringent privacy regulations. Consequently, data availability and privacy protection are major obstacles to applying machine learning in medical imaging. To address this, our study proposes the Med-LSDM (Latent Semantic Diffusion Model), which operates directly in the 3D domain and leverages de-identified semantic maps to generate synthetic data as a method of privacy preservation and data augmentation. Unlike many existing methods that focus on generating 2D slices, Med-LSDM is designed specifically for 3D semantic image synthesis, making it well-suited for applications requiring full volumetric data. Med-LSDM incorporates a guiding mechanism that controls the 3D image generation process by applying a diffusion model within the latent space of a pre-trained VQ-GAN. By operating in the compressed latent space, the model significantly reduces computational complexity while still preserving critical 3D spatial details. Our approach demonstrates strong performance in 3D semantic medical image synthesis, achieving a 3D-FID score of 0.0054 on the conditional Duke Breast dataset and similar Dice scores (0.70964) to those of real images (0.71496). These results demonstrate that the synthetic data from our model have a small domain gap with real data and are useful for data augmentation.
>
---
#### [new 129] TalentMine: LLM-Based Extraction and Question-Answering from Multimodal Talent Tables
- **分类: cs.AI; cs.CV; cs.IR**

- **简介: 该论文属于表格信息提取与问答任务，解决传统方法在语义理解上的不足。提出TalentMine框架，提升表格语义表示和问答准确率。**

- **链接: [http://arxiv.org/pdf/2507.00041v1](http://arxiv.org/pdf/2507.00041v1)**

> **作者:** Varun Mannam; Fang Wang; Chaochun Liu; Xin Chen
>
> **备注:** Submitted to KDD conference, workshop: Talent and Management Computing (TMC 2025), https://tmcworkshop.github.io/2025/
>
> **摘要:** In talent management systems, critical information often resides in complex tabular formats, presenting significant retrieval challenges for conventional language models. These challenges are pronounced when processing Talent documentation that requires precise interpretation of tabular relationships for accurate information retrieval and downstream decision-making. Current table extraction methods struggle with semantic understanding, resulting in poor performance when integrated into retrieval-augmented chat applications. This paper identifies a key bottleneck - while structural table information can be extracted, the semantic relationships between tabular elements are lost, causing downstream query failures. To address this, we introduce TalentMine, a novel LLM-enhanced framework that transforms extracted tables into semantically enriched representations. Unlike conventional approaches relying on CSV or text linearization, our method employs specialized multimodal reasoning to preserve both structural and semantic dimensions of tabular data. Experimental evaluation across employee benefits document collections demonstrates TalentMine's superior performance, achieving 100% accuracy in query answering tasks compared to 0% for standard AWS Textract extraction and 40% for AWS Textract Visual Q&A capabilities. Our comparative analysis also reveals that the Claude v3 Haiku model achieves optimal performance for talent management applications. The key contributions of this work include (1) a systematic analysis of semantic information loss in current table extraction pipelines, (2) a novel LLM-based method for semantically enriched table representation, (3) an efficient integration framework for retrieval-augmented systems as end-to-end systems, and (4) comprehensive benchmarks on talent analytics tasks showing substantial improvements across multiple categories.
>
---
#### [new 130] Box Pose and Shape Estimation and Domain Adaptation for Large-Scale Warehouse Automation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人视觉任务，解决仓库自动化中箱体位姿与形状估计问题，提出自监督域适应方法，利用未标注数据提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00984v1](http://arxiv.org/pdf/2507.00984v1)**

> **作者:** Xihang Yu; Rajat Talak; Jingnan Shi; Ulrich Viereck; Igor Gilitschenski; Luca Carlone
>
> **备注:** 12 pages, 6 figures. This work will be presented at the 19th International Symposium on Experimental Robotics (ISER2025)
>
> **摘要:** Modern warehouse automation systems rely on fleets of intelligent robots that generate vast amounts of data -- most of which remains unannotated. This paper develops a self-supervised domain adaptation pipeline that leverages real-world, unlabeled data to improve perception models without requiring manual annotations. Our work focuses specifically on estimating the pose and shape of boxes and presents a correct-and-certify pipeline for self-supervised box pose and shape estimation. We extensively evaluate our approach across a range of simulated and real industrial settings, including adaptation to a large-scale real-world dataset of 50,000 images. The self-supervised model significantly outperforms models trained solely in simulation and shows substantial improvements over a zero-shot 3D bounding box estimation baseline.
>
---
#### [new 131] Exploring Theory-Laden Observations in the Brain Basis of Emotional Experience
- **分类: cs.LG; cs.CV; q-bio.NC**

- **简介: 该论文属于情感神经科学领域，旨在检验情绪分类的生物学基础。通过重新分析数据，发现情绪类别内部存在显著个体差异，挑战了传统假设，强调多方法验证的重要性。**

- **链接: [http://arxiv.org/pdf/2507.00320v1](http://arxiv.org/pdf/2507.00320v1)**

> **作者:** Christiana Westlin; Ashutosh Singh; Deniz Erdogmus; Georgios Stratis; Lisa Feldman Barrett
>
> **摘要:** In the science of emotion, it is widely assumed that folk emotion categories form a biological and psychological typology, and studies are routinely designed and analyzed to identify emotion-specific patterns. This approach shapes the observations that studies report, ultimately reinforcing the assumption that guided the investigation. Here, we reanalyzed data from one such typologically-guided study that reported mappings between individual brain patterns and group-averaged ratings of 34 emotion categories. Our reanalysis was guided by an alternative view of emotion categories as populations of variable, situated instances, and which predicts a priori that there will be significant variation in brain patterns within a category across instances. Correspondingly, our analysis made minimal assumptions about the structure of the variance present in the data. As predicted, we did not observe the original mappings and instead observed significant variation across individuals. These findings demonstrate how starting assumptions can ultimately impact scientific conclusions and suggest that a hypothesis must be supported using multiple analytic methods before it is taken seriously.
>
---
#### [new 132] MTCNet: Motion and Topology Consistency Guided Learning for Mitral Valve Segmentationin 4D Ultrasound
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决4D超声心动图中二尖瓣分割的跨相位一致性问题。通过引入运动与拓扑一致性学习，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2507.00660v1](http://arxiv.org/pdf/2507.00660v1)**

> **作者:** Rusi Chen; Yuanting Yang; Jiezhi Yao; Hongning Song; Ji Zhang; Yongsong Zhou; Yuhao Huang; Ronghao Yang; Dan Jia; Yuhan Zhang; Xing Tao; Haoran Dou; Qing Zhou; Xin Yang; Dong Ni
>
> **备注:** Accepted by MICCAI 2025
>
> **摘要:** Mitral regurgitation is one of the most prevalent cardiac disorders. Four-dimensional (4D) ultrasound has emerged as the primary imaging modality for assessing dynamic valvular morphology. However, 4D mitral valve (MV) analysis remains challenging due to limited phase annotations, severe motion artifacts, and poor imaging quality. Yet, the absence of inter-phase dependency in existing methods hinders 4D MV analysis. To bridge this gap, we propose a Motion-Topology guided consistency network (MTCNet) for accurate 4D MV ultrasound segmentation in semi-supervised learning (SSL). MTCNet requires only sparse end-diastolic and end-systolic annotations. First, we design a cross-phase motion-guided consistency learning strategy, utilizing a bi-directional attention memory bank to propagate spatio-temporal features. This enables MTCNet to achieve excellent performance both per- and inter-phase. Second, we devise a novel topology-guided correlation regularization that explores physical prior knowledge to maintain anatomically plausible. Therefore, MTCNet can effectively leverage structural correspondence between labeled and unlabeled phases. Extensive evaluations on the first largest 4D MV dataset, with 1408 phases from 160 patients, show that MTCNet performs superior cross-phase consistency compared to other advanced methods (Dice: 87.30%, HD: 1.75mm). Both the code and the dataset are available at https://github.com/crs524/MTCNet.
>
---
#### [new 133] SurgiSR4K: A High-Resolution Endoscopic Video Dataset for Robotic-Assisted Minimally Invasive Procedures
- **分类: eess.IV; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出SurgiSR4K数据集，解决机器人辅助微创手术中高分辨率影像数据不足的问题，支持多种计算机视觉任务。**

- **链接: [http://arxiv.org/pdf/2507.00209v1](http://arxiv.org/pdf/2507.00209v1)**

> **作者:** Fengyi Jiang; Xiaorui Zhang; Lingbo Jin; Ruixing Liang; Yuxin Chen; Adi Chola Venkatesh; Jason Culman; Tiantian Wu; Lirong Shao; Wenqing Sun; Cong Gao; Hallie McNamara; Jingpei Lu; Omid Mohareri
>
> **摘要:** High-resolution imaging is crucial for enhancing visual clarity and enabling precise computer-assisted guidance in minimally invasive surgery (MIS). Despite the increasing adoption of 4K endoscopic systems, there remains a significant gap in publicly available native 4K datasets tailored specifically for robotic-assisted MIS. We introduce SurgiSR4K, the first publicly accessible surgical imaging and video dataset captured at a native 4K resolution, representing realistic conditions of robotic-assisted procedures. SurgiSR4K comprises diverse visual scenarios including specular reflections, tool occlusions, bleeding, and soft tissue deformations, meticulously designed to reflect common challenges faced during laparoscopic and robotic surgeries. This dataset opens up possibilities for a broad range of computer vision tasks that might benefit from high resolution data, such as super resolution (SR), smoke removal, surgical instrument detection, 3D tissue reconstruction, monocular depth estimation, instance segmentation, novel view synthesis, and vision-language model (VLM) development. SurgiSR4K provides a robust foundation for advancing research in high-resolution surgical imaging and fosters the development of intelligent imaging technologies aimed at enhancing performance, safety, and usability in image-guided robotic surgeries.
>
---
#### [new 134] VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VQ-VLA，解决视觉-语言-动作模型中的动作分词问题，通过大规模数据提升模型性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.01016v1](http://arxiv.org/pdf/2507.01016v1)**

> **作者:** Yating Wang; Haoyi Zhu; Mingyu Liu; Jiange Yang; Hao-Shu Fang; Tong He
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** In this paper, we introduce an innovative vector quantization based action tokenizer built upon the largest-scale action trajectory dataset to date, leveraging over 100 times more data than previous approaches. This extensive dataset enables our tokenizer to capture rich spatiotemporal dynamics, resulting in a model that not only accelerates inference but also generates smoother and more coherent action outputs. Once trained, the tokenizer can be seamlessly adapted to a wide range of downstream tasks in a zero-shot manner, from short-horizon reactive behaviors to long-horizon planning. A key finding of our work is that the domain gap between synthetic and real action trajectories is marginal, allowing us to effectively utilize a vast amount of synthetic data during training without compromising real-world performance. To validate our approach, we conducted extensive experiments in both simulated environments and on real robotic platforms. The results demonstrate that as the volume of synthetic trajectory data increases, the performance of our tokenizer on downstream tasks improves significantly-most notably, achieving up to a 30% higher success rate on two real-world tasks in long-horizon scenarios. These findings highlight the potential of our action tokenizer as a robust and scalable solution for real-time embodied intelligence systems, paving the way for more efficient and reliable robotic control in diverse application domains.Project website: https://xiaoxiao0406.github.io/vqvla.github.io
>
---
#### [new 135] Diffusion Classifier Guidance for Non-robust Classifiers
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成模型中的条件采样任务，解决非鲁棒分类器在扩散过程中引导不稳定的问题，提出稳定方法提升分类器引导的可靠性。**

- **链接: [http://arxiv.org/pdf/2507.00687v1](http://arxiv.org/pdf/2507.00687v1)**

> **作者:** Philipp Vaeth; Dibyanshu Kumar; Benjamin Paassen; Magda Gregorová
>
> **备注:** Accepted at ECML 2025
>
> **摘要:** Classifier guidance is intended to steer a diffusion process such that a given classifier reliably recognizes the generated data point as a certain class. However, most classifier guidance approaches are restricted to robust classifiers, which were specifically trained on the noise of the diffusion forward process. We extend classifier guidance to work with general, non-robust, classifiers that were trained without noise. We analyze the sensitivity of both non-robust and robust classifiers to noise of the diffusion process on the standard CelebA data set, the specialized SportBalls data set and the high-dimensional real-world CelebA-HQ data set. Our findings reveal that non-robust classifiers exhibit significant accuracy degradation under noisy conditions, leading to unstable guidance gradients. To mitigate these issues, we propose a method that utilizes one-step denoised image predictions and implements stabilization techniques inspired by stochastic optimization methods, such as exponential moving averages. Experimental results demonstrate that our approach improves the stability of classifier guidance while maintaining sample diversity and visual quality. This work contributes to advancing conditional sampling techniques in generative models, enabling a broader range of classifiers to be used as guidance classifiers.
>
---
#### [new 136] Audio-3DVG: Unified Audio - Point Cloud Fusion for 3D Visual Grounding
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于3D视觉定位任务，解决音频引导的3D物体定位问题。提出Audio-3DVG框架，融合音频与点云信息，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2507.00669v1](http://arxiv.org/pdf/2507.00669v1)**

> **作者:** Duc Cao-Dinh; Khai Le-Duc; Anh Dao; Bach Phan Tat; Chris Ngo; Duy M. H. Nguyen; Nguyen X. Khanh; Thanh Nguyen-Tang
>
> **备注:** Work in progress, 42 pages
>
> **摘要:** 3D Visual Grounding (3DVG) involves localizing target objects in 3D point clouds based on natural language. While prior work has made strides using textual descriptions, leveraging spoken language-known as Audio-based 3D Visual Grounding-remains underexplored and challenging. Motivated by advances in automatic speech recognition (ASR) and speech representation learning, we propose Audio-3DVG, a simple yet effective framework that integrates audio and spatial information for enhanced grounding. Rather than treating speech as a monolithic input, we decompose the task into two complementary components. First, we introduce Object Mention Detection, a multi-label classification task that explicitly identifies which objects are referred to in the audio, enabling more structured audio-scene reasoning. Second, we propose an Audio-Guided Attention module that captures interactions between candidate objects and relational speech cues, improving target discrimination in cluttered scenes. To support benchmarking, we synthesize audio descriptions for standard 3DVG datasets, including ScanRefer, Sr3D, and Nr3D. Experimental results demonstrate that Audio-3DVG not only achieves new state-of-the-art performance in audio-based grounding, but also competes with text-based methods-highlighting the promise of integrating spoken language into 3D vision tasks.
>
---
## 更新

#### [replaced 001] Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.02424v2](http://arxiv.org/pdf/2503.02424v2)**

> **作者:** Wei Luo; Yunkang Cao; Haiming Yao; Xiaotian Zhang; Jianan Lou; Yuqi Cheng; Weiming Shen; Wenyong Yu
>
> **备注:** Accepted by CVPR2025
>
> **摘要:** Anomaly detection (AD) is essential for industrial inspection, yet existing methods typically rely on ``comparing'' test images to normal references from a training set. However, variations in appearance and positioning often complicate the alignment of these references with the test image, limiting detection accuracy. We observe that most anomalies manifest as local variations, meaning that even within anomalous images, valuable normal information remains. We argue that this information is useful and may be more aligned with the anomalies since both the anomalies and the normal information originate from the same image. Therefore, rather than relying on external normality from the training set, we propose INP-Former, a novel method that extracts Intrinsic Normal Prototypes (INPs) directly from the test image. Specifically, we introduce the INP Extractor, which linearly combines normal tokens to represent INPs. We further propose an INP Coherence Loss to ensure INPs can faithfully represent normality for the testing image. These INPs then guide the INP-Guided Decoder to reconstruct only normal tokens, with reconstruction errors serving as anomaly scores. Additionally, we propose a Soft Mining Loss to prioritize hard-to-optimize samples during training. INP-Former achieves state-of-the-art performance in single-class, multi-class, and few-shot AD tasks across MVTec-AD, VisA, and Real-IAD, positioning it as a versatile and universal solution for AD. Remarkably, INP-Former also demonstrates some zero-shot AD capability. Code is available at:https://github.com/luow23/INP-Former.
>
---
#### [replaced 002] Unsupervised contrastive analysis for anomaly detection in brain MRIs via conditional diffusion models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.00772v3](http://arxiv.org/pdf/2406.00772v3)**

> **作者:** Cristiano Patrício; Carlo Alberto Barbano; Attilio Fiandrotti; Riccardo Renzulli; Marco Grangetto; Luis F. Teixeira; João C. Neves
>
> **备注:** Under consideration at Pattern Recognition Letters
>
> **摘要:** Contrastive Analysis (CA) detects anomalies by contrasting patterns unique to a target group (e.g., unhealthy subjects) from those in a background group (e.g., healthy subjects). In the context of brain MRIs, existing CA approaches rely on supervised contrastive learning or variational autoencoders (VAEs) using both healthy and unhealthy data, but such reliance on target samples is challenging in clinical settings. Unsupervised Anomaly Detection (UAD) offers an alternative by learning a reference representation of healthy anatomy without the need for target samples. Deviations from this reference distribution can indicate potential anomalies. In this context, diffusion models have been increasingly adopted in UAD due to their superior performance in image generation compared to VAEs. Nonetheless, precisely reconstructing the anatomy of the brain remains a challenge. In this work, we propose an unsupervised framework to improve the reconstruction quality by training a self-supervised contrastive encoder on healthy images to extract meaningful anatomical features. These features are used to condition a diffusion model to reconstruct the healthy appearance of a given image, enabling interpretable anomaly localization via pixel-wise comparison. We validate our approach through a proof-of-concept on a facial image dataset and further demonstrate its effectiveness on four brain MRI datasets, achieving state-of-the-art anomaly localization performance on the NOVA benchmark.
>
---
#### [replaced 003] R1-Track: Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21980v2](http://arxiv.org/pdf/2506.21980v2)**

> **作者:** Biao Wang; Wenwen Li
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** Visual single object tracking aims to continuously localize and estimate the scale of a target in subsequent video frames, given only its initial state in the first frame. This task has traditionally been framed as a template matching problem, evolving through major phases including correlation filters, two-stream networks, and one-stream networks with significant progress achieved. However, these methods typically require explicit classification and regression modeling, depend on supervised training with large-scale datasets, and are limited to the single task of tracking, lacking flexibility. In recent years, multi-modal large language models (MLLMs) have advanced rapidly. Open-source models like Qwen2.5-VL, a flagship MLLMs with strong foundational capabilities, demonstrate excellent performance in grounding tasks. This has spurred interest in applying such models directly to visual tracking. However, experiments reveal that Qwen2.5-VL struggles with template matching between image pairs (i.e., tracking tasks). Inspired by deepseek-R1, we fine-tuned Qwen2.5-VL using the group relative policy optimization (GRPO) reinforcement learning method on a small-scale dataset with a rule-based reward function. The resulting model, R1-Track, achieved notable performance on the GOT-10k benchmark. R1-Track supports flexible initialization via bounding boxes or text descriptions while retaining most of the original model's general capabilities. And we further discuss potential improvements for R1-Track. This rough technical report summarizes our findings as of May 2025.
>
---
#### [replaced 004] Towards Generalized and Training-Free Text-Guided Semantic Manipulation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.17269v2](http://arxiv.org/pdf/2504.17269v2)**

> **作者:** Yu Hong; Xiao Cai; Pengpeng Zeng; Shuai Zhang; Jingkuan Song; Lianli Gao; Heng Tao Shen
>
> **备注:** Project Page: https://ayanami-yu.github.io/GTF-Project-Page/
>
> **摘要:** Text-guided semantic manipulation refers to semantically editing an image generated from a source prompt to match a target prompt, enabling the desired semantic changes (e.g., addition, removal, and style transfer) while preserving irrelevant contents. With the powerful generative capabilities of the diffusion model, the task has shown the potential to generate high-fidelity visual content. Nevertheless, existing methods either typically require time-consuming fine-tuning (inefficient), fail to accomplish multiple semantic manipulations (poorly extensible), and/or lack support for different modality tasks (limited generalizability). Upon further investigation, we find that the geometric properties of noises in the diffusion model are strongly correlated with the semantic changes. Motivated by this, we propose a novel $\textit{GTF}$ for text-guided semantic manipulation, which has the following attractive capabilities: 1) $\textbf{Generalized}$: our $\textit{GTF}$ supports multiple semantic manipulations (e.g., addition, removal, and style transfer) and can be seamlessly integrated into all diffusion-based methods (i.e., Plug-and-play) across different modalities (i.e., modality-agnostic); and 2) $\textbf{Training-free}$: $\textit{GTF}$ produces high-fidelity results via simply controlling the geometric relationship between noises without tuning or optimization. Our extensive experiments demonstrate the efficacy of our approach, highlighting its potential to advance the state-of-the-art in semantics manipulation.
>
---
#### [replaced 005] Generating Physically Stable and Buildable Brick Structures from Text
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05469v2](http://arxiv.org/pdf/2505.05469v2)**

> **作者:** Ava Pun; Kangle Deng; Ruixuan Liu; Deva Ramanan; Changliu Liu; Jun-Yan Zhu
>
> **备注:** Project page: https://avalovelace1.github.io/BrickGPT/
>
> **摘要:** We introduce BrickGPT, the first approach for generating physically stable interconnecting brick assembly models from text prompts. To achieve this, we construct a large-scale, physically stable dataset of brick structures, along with their associated captions, and train an autoregressive large language model to predict the next brick to add via next-token prediction. To improve the stability of the resulting designs, we employ an efficient validity check and physics-aware rollback during autoregressive inference, which prunes infeasible token predictions using physics laws and assembly constraints. Our experiments show that BrickGPT produces stable, diverse, and aesthetically pleasing brick structures that align closely with the input text prompts. We also develop a text-based brick texturing method to generate colored and textured designs. We show that our designs can be assembled manually by humans and automatically by robotic arms. We release our new dataset, StableText2Brick, containing over 47,000 brick structures of over 28,000 unique 3D objects accompanied by detailed captions, along with our code and models at the project website: https://avalovelace1.github.io/BrickGPT/.
>
---
#### [replaced 006] ICME 2025 Grand Challenge on Video Super-Resolution for Video Conferencing
- **分类: eess.IV; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.12269v2](http://arxiv.org/pdf/2506.12269v2)**

> **作者:** Babak Naderi; Ross Cutler; Juhee Cho; Nabakumar Khongbantabam; Dejan Ivkovic
>
> **摘要:** Super-Resolution (SR) is a critical task in computer vision, focusing on reconstructing high-resolution (HR) images from low-resolution (LR) inputs. The field has seen significant progress through various challenges, particularly in single-image SR. Video Super-Resolution (VSR) extends this to the temporal domain, aiming to enhance video quality using methods like local, uni-, bi-directional propagation, or traditional upscaling followed by restoration. This challenge addresses VSR for conferencing, where LR videos are encoded with H.265 at fixed QPs. The goal is to upscale videos by a specific factor, providing HR outputs with enhanced perceptual quality under a low-delay scenario using causal models. The challenge included three tracks: general-purpose videos, talking head videos, and screen content videos, with separate datasets provided by the organizers for training, validation, and testing. We open-sourced a new screen content dataset for the SR task in this challenge. Submissions were evaluated through subjective tests using a crowdsourced implementation of the ITU-T Rec P.910.
>
---
#### [replaced 007] Towards Markerless Intraoperative Tracking of Deformable Spine Tissue
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23657v2](http://arxiv.org/pdf/2506.23657v2)**

> **作者:** Connor Daly; Elettra Marconi; Marco Riva; Jinendra Ekanayake; Daniel S. Elson; Ferdinando Rodriguez y Baena
>
> **备注:** An improved version of this manuscript was accepted to MICCAI
>
> **摘要:** Consumer-grade RGB-D imaging for intraoperative orthopedic tissue tracking is a promising method with high translational potential. Unlike bone-mounted tracking devices, markerless tracking can reduce operating time and complexity. However, its use has been limited to cadaveric studies. This paper introduces the first real-world clinical RGB-D dataset for spine surgery and develops SpineAlign, a system for capturing deformation between preoperative and intraoperative spine states. We also present an intraoperative segmentation network trained on this data and introduce CorrespondNet, a multi-task framework for predicting key regions for registration in both intraoperative and preoperative scenes.
>
---
#### [replaced 008] A Good Start Matters: Enhancing Continual Learning with Data-Driven Weight Initialization
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06385v2](http://arxiv.org/pdf/2503.06385v2)**

> **作者:** Md Yousuf Harun; Christopher Kanan
>
> **备注:** Accepted to the Conference on Lifelong Learning Agents (CoLLAs) 2025
>
> **摘要:** To adapt to real-world data streams, continual learning (CL) systems must rapidly learn new concepts while preserving and utilizing prior knowledge. When it comes to adding new information to continually-trained deep neural networks (DNNs), classifier weights for newly encountered categories are typically initialized randomly, leading to high initial training loss (spikes) and instability. Consequently, achieving optimal convergence and accuracy requires prolonged training, increasing computational costs. Inspired by Neural Collapse (NC), we propose a weight initialization strategy to improve learning efficiency in CL. In DNNs trained with mean-squared-error, NC gives rise to a Least-Square (LS) classifier in the last layer, whose weights can be analytically derived from learned features. We leverage this LS formulation to initialize classifier weights in a data-driven manner, aligning them with the feature distribution rather than using random initialization. Our method mitigates initial loss spikes and accelerates adaptation to new tasks. We evaluate our approach in large-scale CL settings, demonstrating faster adaptation and improved CL performance.
>
---
#### [replaced 009] Edit Transfer: Learning Image Editing via Vision In-Context Relations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13327v2](http://arxiv.org/pdf/2503.13327v2)**

> **作者:** Lan Chen; Qi Mao; Yuchao Gu; Mike Zheng Shou
>
> **摘要:** We introduce a new setting, Edit Transfer, where a model learns a transformation from just a single source-target example and applies it to a new query image. While text-based methods excel at semantic manipulations through textual prompts, they often struggle with precise geometric details (e.g., poses and viewpoint changes). Reference-based editing, on the other hand, typically focuses on style or appearance and fails at non-rigid transformations. By explicitly learning the editing transformation from a source-target pair, Edit Transfer mitigates the limitations of both text-only and appearance-centric references. Drawing inspiration from in-context learning in large language models, we propose a visual relation in-context learning paradigm, building upon a DiT-based text-to-image model. We arrange the edited example and the query image into a unified four-panel composite, then apply lightweight LoRA fine-tuning to capture complex spatial transformations from minimal examples. Despite using only 42 training samples, Edit Transfer substantially outperforms state-of-the-art TIE and RIE methods on diverse non-rigid scenarios, demonstrating the effectiveness of few-shot visual relation learning.
>
---
#### [replaced 010] Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23918v2](http://arxiv.org/pdf/2506.23918v2)**

> **作者:** Zhaochen Su; Peng Xia; Hangyu Guo; Zhenhua Liu; Yan Ma; Xiaoye Qu; Jiaqi Liu; Yanshu Li; Kaide Zeng; Zhengyuan Yang; Linjie Li; Yu Cheng; Heng Ji; Junxian He; Yi R. Fung
>
> **备注:** We maintain a real-time GitHub repository tracking progress at: https://github.com/zhaochen0110/Awesome_Think_With_Images
>
> **摘要:** Recent progress in multimodal reasoning has been significantly advanced by textual Chain-of-Thought (CoT), a paradigm where models conduct reasoning within language. This text-centric approach, however, treats vision as a static, initial context, creating a fundamental "semantic gap" between rich perceptual data and discrete symbolic thought. Human cognition often transcends language, utilizing vision as a dynamic mental sketchpad. A similar evolution is now unfolding in AI, marking a fundamental paradigm shift from models that merely think about images to those that can truly think with images. This emerging paradigm is characterized by models leveraging visual information as intermediate steps in their thought process, transforming vision from a passive input into a dynamic, manipulable cognitive workspace. In this survey, we chart this evolution of intelligence along a trajectory of increasing cognitive autonomy, which unfolds across three key stages: from external tool exploration, through programmatic manipulation, to intrinsic imagination. To structure this rapidly evolving field, our survey makes four key contributions. (1) We establish the foundational principles of the think with image paradigm and its three-stage framework. (2) We provide a comprehensive review of the core methods that characterize each stage of this roadmap. (3) We analyze the critical landscape of evaluation benchmarks and transformative applications. (4) We identify significant challenges and outline promising future directions. By providing this structured overview, we aim to offer a clear roadmap for future research towards more powerful and human-aligned multimodal AI.
>
---
#### [replaced 011] Fully Differentiable Lagrangian Convolutional Neural Network for Physics-Informed Precipitation Nowcasting
- **分类: cs.LG; cs.AI; cs.CV; I.2.1; J.2**

- **链接: [http://arxiv.org/pdf/2402.10747v2](http://arxiv.org/pdf/2402.10747v2)**

> **作者:** Peter Pavlík; Martin Výboh; Anna Bou Ezzeddine; Viera Rozinajová
>
> **备注:** Submitted to Applied Computing and Geosciences
>
> **摘要:** This paper presents a convolutional neural network model for precipitation nowcasting that combines data-driven learning with physics-informed domain knowledge. We propose LUPIN, a Lagrangian Double U-Net for Physics-Informed Nowcasting, that draws from existing extrapolation-based nowcasting methods. It consists of a U-Net that dynamically produces mesoscale advection motion fields, a differentiable semi-Lagrangian extrapolation operator, and an advection-free U-Net capturing the growth and decay of precipitation over time. Using our approach, we successfully implement the Lagrangian convolutional neural network for precipitation nowcasting in a fully differentiable and GPU-accelerated manner. This allows for end-to-end training and inference, including the data-driven Lagrangian coordinate system transformation of the data at runtime. We evaluate the model and compare it with other related AI-based models both quantitatively and qualitatively in an extreme event case study. Based on our evaluation, LUPIN matches and even exceeds the performance of the chosen benchmarks, opening the door for other Lagrangian machine learning models.
>
---
#### [replaced 012] Seeking and Updating with Live Visual Knowledge
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05288v2](http://arxiv.org/pdf/2504.05288v2)**

> **作者:** Mingyang Fu; Yuyang Peng; Dongping Chen; Zetong Zhou; Benlin Liu; Yao Wan; Zhou Zhao; Philip S. Yu; Ranjay Krishna
>
> **备注:** Preprint. Under Review
>
> **摘要:** The visual world around us constantly evolves, from real-time news and social media trends to global infrastructure changes visible through satellite imagery and augmented reality enhancements. However, Multimodal Large Language Models (MLLMs), which automate many tasks, struggle to stay current, limited by the cutoff dates in their fixed training datasets. To quantify this stagnation, we introduce LiveVQA, the first-of-its-kind dataset featuring 107,143 samples and 12 categories data specifically designed to support research in both seeking and updating with live visual knowledge. Drawing from recent news articles, video platforms, and academic publications in April 2024-May 2025, LiveVQA enables evaluation of how models handle latest visual information beyond their knowledge boundaries and how current methods help to update them. Our comprehensive benchmarking of 17 state-of-the-art MLLMs reveals significant performance gaps on content beyond knowledge cutoff, and tool-use or agentic visual seeking framework drastically gain an average of 327% improvement. Furthermore, we explore parameter-efficient fine-tuning (PEFT) methods to update MLLMs with new visual knowledge. We dive deeply to the critical balance between adapter capacity and model capability when updating MLLMs with new visual knowledge. All the experimental dataset and source code are publicly available at: https://livevqa.github.io.
>
---
#### [replaced 013] Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.22554v2](http://arxiv.org/pdf/2506.22554v2)**

> **作者:** Vasu Agrawal; Akinniyi Akinyemi; Kathryn Alvero; Morteza Behrooz; Julia Buffalini; Fabio Maria Carlucci; Joy Chen; Junming Chen; Zhang Chen; Shiyang Cheng; Praveen Chowdary; Joe Chuang; Antony D'Avirro; Jon Daly; Ning Dong; Mark Duppenthaler; Cynthia Gao; Jeff Girard; Martin Gleize; Sahir Gomez; Hongyu Gong; Srivathsan Govindarajan; Brandon Han; Sen He; Denise Hernandez; Yordan Hristov; Rongjie Huang; Hirofumi Inaguma; Somya Jain; Raj Janardhan; Qingyao Jia; Christopher Klaiber; Dejan Kovachev; Moneish Kumar; Hang Li; Yilei Li; Pavel Litvin; Wei Liu; Guangyao Ma; Jing Ma; Martin Ma; Xutai Ma; Lucas Mantovani; Sagar Miglani; Sreyas Mohan; Louis-Philippe Morency; Evonne Ng; Kam-Woh Ng; Tu Anh Nguyen; Amia Oberai; Benjamin Peloquin; Juan Pino; Jovan Popovic; Omid Poursaeed; Fabian Prada; Alice Rakotoarison; Rakesh Ranjan; Alexander Richard; Christophe Ropers; Safiyyah Saleem; Vasu Sharma; Alex Shcherbyna; Jia Shen; Jie Shen; Anastasis Stathopoulos; Anna Sun; Paden Tomasello; Tuan Tran; Arina Turkatenko; Bo Wan; Chao Wang; Jeff Wang; Mary Williamson; Carleigh Wood; Tao Xiang; Yilin Yang; Julien Yao; Chen Zhang; Jiemin Zhang; Xinyue Zhang; Jason Zheng; Pavlo Zhyzheria; Jan Zikes; Michael Zollhoefer
>
> **摘要:** Human communication involves a complex interplay of verbal and nonverbal signals, essential for conveying meaning and achieving interpersonal goals. To develop socially intelligent AI technologies, it is crucial to develop models that can both comprehend and generate dyadic behavioral dynamics. To this end, we introduce the Seamless Interaction Dataset, a large-scale collection of over 4,000 hours of face-to-face interaction footage from over 4,000 participants in diverse contexts. This dataset enables the development of AI technologies that understand dyadic embodied dynamics, unlocking breakthroughs in virtual agents, telepresence experiences, and multimodal content analysis tools. We also develop a suite of models that utilize the dataset to generate dyadic motion gestures and facial expressions aligned with human speech. These models can take as input both the speech and visual behavior of their interlocutors. We present a variant with speech from an LLM model and integrations with 2D and 3D rendering methods, bringing us closer to interactive virtual agents. Additionally, we describe controllable variants of our motion models that can adapt emotional responses and expressivity levels, as well as generating more semantically-relevant gestures. Finally, we discuss methods for assessing the quality of these dyadic motion models, which are demonstrating the potential for more intuitive and responsive human-AI interactions.
>
---
#### [replaced 014] Building Rome with Convex Optimization
- **分类: cs.RO; cs.CV; math.OC**

- **链接: [http://arxiv.org/pdf/2502.04640v4](http://arxiv.org/pdf/2502.04640v4)**

> **作者:** Haoyu Han; Heng Yang
>
> **摘要:** Global bundle adjustment is made easy by depth prediction and convex optimization. We (i) propose a scaled bundle adjustment (SBA) formulation that lifts 2D keypoint measurements to 3D with learned depth, (ii) design an empirically tight convex semidfinite program (SDP) relaxation that solves SBA to certfiable global optimality, (iii) solve the SDP relaxations at extreme scale with Burer-Monteiro factorization and a CUDA-based trust-region Riemannian optimizer (dubbed XM), (iv) build a structure from motion (SfM) pipeline with XM as the optimization engine and show that XM-SfM compares favorably with existing pipelines in terms of reconstruction quality while being significantly faster, more scalable, and initialization-free.
>
---
#### [replaced 015] FreqDGT: Frequency-Adaptive Dynamic Graph Networks with Transformer for Cross-subject EEG Emotion Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22807v2](http://arxiv.org/pdf/2506.22807v2)**

> **作者:** Yueyang Li; Shengyu Gong; Weiming Zeng; Nizhuan Wang; Wai Ting Siok
>
> **摘要:** Electroencephalography (EEG) serves as a reliable and objective signal for emotion recognition in affective brain-computer interfaces, offering unique advantages through its high temporal resolution and ability to capture authentic emotional states that cannot be consciously controlled. However, cross-subject generalization remains a fundamental challenge due to individual variability, cognitive traits, and emotional responses. We propose FreqDGT, a frequency-adaptive dynamic graph transformer that systematically addresses these limitations through an integrated framework. FreqDGT introduces frequency-adaptive processing (FAP) to dynamically weight emotion-relevant frequency bands based on neuroscientific evidence, employs adaptive dynamic graph learning (ADGL) to learn input-specific brain connectivity patterns, and implements multi-scale temporal disentanglement network (MTDN) that combines hierarchical temporal transformers with adversarial feature disentanglement to capture both temporal dynamics and ensure cross-subject robustness. Comprehensive experiments demonstrate that FreqDGT significantly improves cross-subject emotion recognition accuracy, confirming the effectiveness of integrating frequency-adaptive, spatial-dynamic, and temporal-hierarchical modeling while ensuring robustness to individual differences. The code is available at https://github.com/NZWANG/FreqDGT.
>
---
#### [replaced 016] Semi-supervised Semantic Segmentation for Remote Sensing Images via Multi-scale Uncertainty Consistency and Cross-Teacher-Student Attention
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.10736v3](http://arxiv.org/pdf/2501.10736v3)**

> **作者:** Shanwen Wang; Xin Sun; Changrui Chen; Danfeng Hong; Jungong Han
>
> **摘要:** Semi-supervised learning offers an appealing solution for remote sensing (RS) image segmentation to relieve the burden of labor-intensive pixel-level labeling. However, RS images pose unique challenges, including rich multi-scale features and high inter-class similarity. To address these problems, this paper proposes a novel semi-supervised Multi-Scale Uncertainty and Cross-Teacher-Student Attention (MUCA) model for RS image semantic segmentation tasks. Specifically, MUCA constrains the consistency among feature maps at different layers of the network by introducing a multi-scale uncertainty consistency regularization. It improves the multi-scale learning capability of semi-supervised algorithms on unlabeled data. Additionally, MUCA utilizes a Cross-Teacher-Student attention mechanism to guide the student network, guiding the student network to construct more discriminative feature representations through complementary features from the teacher network. This design effectively integrates weak and strong augmentations (WA and SA) to further boost segmentation performance. To verify the effectiveness of our model, we conduct extensive experiments on ISPRS-Potsdam and LoveDA datasets. The experimental results show the superiority of our method over state-of-the-art semi-supervised methods. Notably, our model excels in distinguishing highly similar objects, showcasing its potential for advancing semi-supervised RS image segmentation tasks.
>
---
#### [replaced 017] CoCMT: Communication-Efficient Cross-Modal Transformer for Collaborative Perception
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.13504v2](http://arxiv.org/pdf/2503.13504v2)**

> **作者:** Rujia Wang; Xiangbo Gao; Hao Xiang; Runsheng Xu; Zhengzhong Tu
>
> **摘要:** Multi-agent collaborative perception enhances each agent perceptual capabilities by sharing sensing information to cooperatively perform robot perception tasks. This approach has proven effective in addressing challenges such as sensor deficiencies, occlusions, and long-range perception. However, existing representative collaborative perception systems transmit intermediate feature maps, such as bird-eye view (BEV) representations, which contain a significant amount of non-critical information, leading to high communication bandwidth requirements. To enhance communication efficiency while preserving perception capability, we introduce CoCMT, an object-query-based collaboration framework that optimizes communication bandwidth by selectively extracting and transmitting essential features. Within CoCMT, we introduce the Efficient Query Transformer (EQFormer) to effectively fuse multi-agent object queries and implement a synergistic deep supervision to enhance the positive reinforcement between stages, leading to improved overall performance. Experiments on OPV2V and V2V4Real datasets show CoCMT outperforms state-of-the-art methods while drastically reducing communication needs. On V2V4Real, our model (Top-50 object queries) requires only 0.416 Mb bandwidth, 83 times less than SOTA methods, while improving AP70 by 1.1 percent. This efficiency breakthrough enables practical collaborative perception deployment in bandwidth-constrained environments without sacrificing detection accuracy.
>
---
#### [replaced 018] Defensive Adversarial CAPTCHA: A Semantics-Driven Framework for Natural Adversarial Example Generation
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2506.10685v3](http://arxiv.org/pdf/2506.10685v3)**

> **作者:** Xia Du; Xiaoyuan Liu; Jizhe Zhou; Zheng Lin; Chi-man Pun; Cong Wu; Tao Li; Zhe Chen; Wei Ni; Jun Luo
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Traditional CAPTCHA (Completely Automated Public Turing Test to Tell Computers and Humans Apart) schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on the original image characteristics, resulting in distortions that hinder human interpretation and limit their applicability in scenarios where no initial input images are available. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (DAC), a novel framework that generates high-fidelity adversarial examples guided by attacker-specified semantics information. Leveraging a Large Language Model (LLM), DAC enhances CAPTCHA diversity and enriches the semantic information. To address various application scenarios, we examine the white-box targeted attack scenario and the black box untargeted attack scenario. For target attacks, we introduce two latent noise variables that are alternately guided in the diffusion step to achieve robust inversion. The synergy between gradient guidance and latent variable optimization achieved in this way ensures that the generated adversarial examples not only accurately align with the target conditions but also achieve optimal performance in terms of distributional consistency and attack effectiveness. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-DAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show that the defensive adversarial CAPTCHA generated by BP-DAC is able to defend against most of the unknown models, and the generated CAPTCHA is indistinguishable to both humans and DNNs.
>
---
#### [replaced 019] Training Free Stylized Abstraction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22663v2](http://arxiv.org/pdf/2505.22663v2)**

> **作者:** Aimon Rahman; Kartik Narayan; Vishal M. Patel
>
> **备注:** Project Page: https://kartik-3004.github.io/TF-SA/
>
> **摘要:** Stylized abstraction synthesizes visually exaggerated yet semantically faithful representations of subjects, balancing recognizability with perceptual distortion. Unlike image-to-image translation, which prioritizes structural fidelity, stylized abstraction demands selective retention of identity cues while embracing stylistic divergence, especially challenging for out-of-distribution individuals. We propose a training-free framework that generates stylized abstractions from a single image using inference-time scaling in vision-language models (VLLMs) to extract identity-relevant features, and a novel cross-domain rectified flow inversion strategy that reconstructs structure based on style-dependent priors. Our method adapts structural restoration dynamically through style-aware temporal scheduling, enabling high-fidelity reconstructions that honor both subject and style. It supports multi-round abstraction-aware generation without fine-tuning. To evaluate this task, we introduce StyleBench, a GPT-based human-aligned metric suited for abstract styles where pixel-level similarity fails. Experiments across diverse abstraction (e.g., LEGO, knitted dolls, South Park) show strong generalization to unseen identities and styles in a fully open-source setup.
>
---
#### [replaced 020] AirV2X: Unified Air-Ground Vehicle-to-Everything Collaboration
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.19283v2](http://arxiv.org/pdf/2506.19283v2)**

> **作者:** Xiangbo Gao; Yuheng Wu; Xuewen Luo; Keshu Wu; Xinghao Chen; Yuping Wang; Chenxi Liu; Yang Zhou; Zhengzhong Tu
>
> **摘要:** While multi-vehicular collaborative driving demonstrates clear advantages over single-vehicle autonomy, traditional infrastructure-based V2X systems remain constrained by substantial deployment costs and the creation of "uncovered danger zones" in rural and suburban areas. We present AirV2X-Perception, a large-scale dataset that leverages Unmanned Aerial Vehicles (UAVs) as a flexible alternative or complement to fixed Road-Side Units (RSUs). Drones offer unique advantages over ground-based perception: complementary bird's-eye-views that reduce occlusions, dynamic positioning capabilities that enable hovering, patrolling, and escorting navigation rules, and significantly lower deployment costs compared to fixed infrastructure. Our dataset comprises 6.73 hours of drone-assisted driving scenarios across urban, suburban, and rural environments with varied weather and lighting conditions. The AirV2X-Perception dataset facilitates the development and standardized evaluation of Vehicle-to-Drone (V2D) algorithms, addressing a critical gap in the rapidly expanding field of aerial-assisted autonomous driving systems. The dataset and development kits are open-sourced at https://github.com/taco-group/AirV2X-Perception.
>
---
#### [replaced 021] Robust Representation Consistency Model via Contrastive Denoising
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.13094v2](http://arxiv.org/pdf/2501.13094v2)**

> **作者:** Jiachen Lei; Julius Berner; Jiongxiao Wang; Zhongzhu Chen; Zhongjia Ba; Kui Ren; Jun Zhu; Anima Anandkumar
>
> **摘要:** Robustness is essential for deep neural networks, especially in security-sensitive applications. To this end, randomized smoothing provides theoretical guarantees for certifying robustness against adversarial perturbations. Recently, diffusion models have been successfully employed for randomized smoothing to purify noise-perturbed samples before making predictions with a standard classifier. While these methods excel at small perturbation radii, they struggle with larger perturbations and incur a significant computational overhead during inference compared to classical methods. To address this, we reformulate the generative modeling task along the diffusion trajectories in pixel space as a discriminative task in the latent space. Specifically, we use instance discrimination to achieve consistent representations along the trajectories by aligning temporally adjacent points. After fine-tuning based on the learned representations, our model enables implicit denoising-then-classification via a single prediction, substantially reducing inference costs. We conduct extensive experiments on various datasets and achieve state-of-the-art performance with minimal computation budget during inference. For example, our method outperforms the certified accuracy of diffusion-based methods on ImageNet across all perturbation radii by 5.3% on average, with up to 11.6% at larger radii, while reducing inference costs by 85$\times$ on average. Codes are available at: https://github.com/jiachenlei/rRCM.
>
---
#### [replaced 022] Grounding Creativity in Physics: A Brief Survey of Physical Priors in AIGC
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07007v3](http://arxiv.org/pdf/2502.07007v3)**

> **作者:** Siwei Meng; Yawei Luo; Ping Liu
>
> **备注:** Accepted by IJCAI 2025 Survey Track
>
> **摘要:** Recent advancements in AI-generated content have significantly improved the realism of 3D and 4D generation. However, most existing methods prioritize appearance consistency while neglecting underlying physical principles, leading to artifacts such as unrealistic deformations, unstable dynamics, and implausible objects interactions. Incorporating physics priors into generative models has become a crucial research direction to enhance structural integrity and motion realism. This survey provides a review of physics-aware generative methods, systematically analyzing how physical constraints are integrated into 3D and 4D generation. First, we examine recent works in incorporating physical priors into static and dynamic 3D generation, categorizing methods based on representation types, including vision-based, NeRF-based, and Gaussian Splatting-based approaches. Second, we explore emerging techniques in 4D generation, focusing on methods that model temporal dynamics with physical simulations. Finally, we conduct a comparative analysis of major methods, highlighting their strengths, limitations, and suitability for different materials and motion dynamics. By presenting an in-depth analysis of physics-grounded AIGC, this survey aims to bridge the gap between generative models and physical realism, providing insights that inspire future research in physically consistent content generation.
>
---
#### [replaced 023] Lifelong Learning of Video Diffusion Models From a Single Video Stream
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.04814v3](http://arxiv.org/pdf/2406.04814v3)**

> **作者:** Jason Yoo; Yingchen He; Saeid Naderiparizi; Dylan Green; Gido M. van de Ven; Geoff Pleiss; Frank Wood
>
> **备注:** Video samples are available here: https://drive.google.com/drive/folders/1CsmWqug-CS7I6NwGDvHsEN9FqN2QzspN
>
> **摘要:** This work demonstrates that training autoregressive video diffusion models from a single video stream$\unicode{x2013}$resembling the experience of embodied agents$\unicode{x2013}$is not only possible, but can also be as effective as standard offline training given the same number of gradient steps. Our work further reveals that this main result can be achieved using experience replay methods that only retain a subset of the preceding video stream. To support training and evaluation in this setting, we introduce four new datasets for streaming lifelong generative video modeling: Lifelong Bouncing Balls, Lifelong 3D Maze, Lifelong Drive, and Lifelong PLAICraft, each consisting of one million consecutive frames from environments of increasing complexity.
>
---
#### [replaced 024] Maximum Dispersion, Maximum Concentration: Enhancing the Quality of MOP Solutions
- **分类: math.OC; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22568v2](http://arxiv.org/pdf/2506.22568v2)**

> **作者:** Gladston Moreira; Ivan Meneghini; Elizabeth Wanner
>
> **备注:** 11 pages
>
> **摘要:** Multi-objective optimization problems (MOPs) often require a trade-off between conflicting objectives, maximizing diversity and convergence in the objective space. This study presents an approach to improve the quality of MOP solutions by optimizing the dispersion in the decision space and the convergence in a specific region of the objective space. Our approach defines a Region of Interest (ROI) based on a cone representing the decision maker's preferences in the objective space, while enhancing the dispersion of solutions in the decision space using a uniformity measure. Combining solution concentration in the objective space with dispersion in the decision space intensifies the search for Pareto-optimal solutions while increasing solution diversity. When combined, these characteristics improve the quality of solutions and avoid the bias caused by clustering solutions in a specific region of the decision space. Preliminary experiments suggest that this method enhances multi-objective optimization by generating solutions that effectively balance dispersion and concentration, thereby mitigating bias in the decision space.
>
---
#### [replaced 025] Mitigating Hallucinations in YOLO-based Object Detection Models: A Revisit to Out-of-Distribution Detection
- **分类: cs.CV; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2503.07330v2](http://arxiv.org/pdf/2503.07330v2)**

> **作者:** Weicheng He; Changshun Wu; Chih-Hong Cheng; Xiaowei Huang; Saddek Bensalem
>
> **备注:** Camera-ready version for IROS 2025
>
> **摘要:** Object detection systems must reliably perceive objects of interest without being overly confident to ensure safe decision-making in dynamic environments. Filtering techniques based on out-of-distribution (OoD) detection are commonly added as an extra safeguard to filter hallucinations caused by overconfidence in novel objects. Nevertheless, evaluating YOLO-family detectors and their filters under existing OoD benchmarks often leads to unsatisfactory performance. This paper studies the underlying reasons for performance bottlenecks and proposes a methodology to improve performance fundamentally. Our first contribution is a calibration of all existing evaluation results: Although images in existing OoD benchmark datasets are claimed not to have objects within in-distribution (ID) classes (i.e., categories defined in the training dataset), around 13% of objects detected by the object detector are actually ID objects. Dually, the ID dataset containing OoD objects can also negatively impact the decision boundary of filters. These ultimately lead to a significantly imprecise performance estimation. Our second contribution is to consider the task of hallucination reduction as a joint pipeline of detectors and filters. By developing a methodology to carefully synthesize an OoD dataset that semantically resembles the objects to be detected, and using the crafted OoD dataset in the fine-tuning of YOLO detectors to suppress the objectness score, we achieve a 88% reduction in overall hallucination error with a combined fine-tuned detection and filtering system on the self-driving benchmark BDD-100K. Our code and dataset are available at: https://gricad-gitlab.univ-grenoble-alpes.fr/dnn-safety/m-hood.
>
---
#### [replaced 026] Teaching Time Series to See and Speak: Forecasting with Aligned Visual and Textual Perspectives
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.24124v2](http://arxiv.org/pdf/2506.24124v2)**

> **作者:** Sixun Dong; Wei Fan; Teresa Wu; Yanjie Fu
>
> **备注:** Code: https://github.com/Ironieser/TimesCLIP
>
> **摘要:** Time series forecasting traditionally relies on unimodal numerical inputs, which often struggle to capture high-level semantic patterns due to their dense and unstructured nature. While recent approaches have explored representing time series as text using large language models (LLMs), these methods remain limited by the discrete nature of token sequences and lack the perceptual intuition humans typically apply, such as interpreting visual patterns. In this paper, we propose a multimodal contrastive learning framework that transforms raw time series into structured visual and textual perspectives. Rather than using natural language or real-world images, we construct both modalities directly from numerical sequences. We then align these views in a shared semantic space via contrastive learning, enabling the model to capture richer and more complementary representations. Furthermore, we introduce a variate selection module that leverages the aligned representations to identify the most informative variables for multivariate forecasting. Extensive experiments on fifteen short-term and six long-term forecasting benchmarks demonstrate that our approach consistently outperforms strong unimodal and cross-modal baselines, highlighting the effectiveness of multimodal alignment in enhancing time series forecasting. Code is available at: https://github.com/Ironieser/TimesCLIP.
>
---
#### [replaced 027] OMNI-DC: Highly Robust Depth Completion with Multiresolution Depth Integration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.19278v2](http://arxiv.org/pdf/2411.19278v2)**

> **作者:** Yiming Zuo; Willow Yang; Zeyu Ma; Jia Deng
>
> **备注:** Accepted to ICCV 2025. Added additional results and ablations
>
> **摘要:** Depth completion (DC) aims to predict a dense depth map from an RGB image and a sparse depth map. Existing DC methods generalize poorly to new datasets or unseen sparse depth patterns, limiting their real-world applications. We propose OMNI-DC, a highly robust DC model that generalizes well zero-shot to various datasets. The key design is a novel Multi-resolution Depth Integrator, allowing our model to deal with very sparse depth inputs. We also introduce a novel Laplacian loss to model the ambiguity in the training process. Moreover, we train OMNI-DC on a mixture of high-quality datasets with a scale normalization technique and synthetic depth patterns. Extensive experiments on 7 datasets show consistent improvements over baselines, reducing errors by as much as 43%. Codes and checkpoints are available at https://github.com/princeton-vl/OMNI-DC.
>
---
#### [replaced 028] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21541v2](http://arxiv.org/pdf/2506.21541v2)**

> **作者:** Chuxin Wang; Yixin Zha; Wenfei Yang; Tianzhu Zhang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recently, Mamba-based methods have demonstrated impressive performance in point cloud representation learning by leveraging State Space Model (SSM) with the efficient context modeling ability and linear complexity. However, these methods still face two key issues that limit the potential of SSM: Destroying the adjacency of 3D points during SSM processing and failing to retain long-sequence memory as the input length increases in downstream tasks. To address these issues, we propose StruMamba3D, a novel paradigm for self-supervised point cloud representation learning. It enjoys several merits. First, we design spatial states and use them as proxies to preserve spatial dependencies among points. Second, we enhance the SSM with a state-wise update strategy and incorporate a lightweight convolution to facilitate interactions between spatial states for efficient structure modeling. Third, our method reduces the sensitivity of pre-trained Mamba-based models to varying input lengths by introducing a sequence length-adaptive strategy. Experimental results across four downstream tasks showcase the superior performance of our method. In addition, our method attains the SOTA 95.1% accuracy on ModelNet40 and 92.75% accuracy on the most challenging split of ScanObjectNN without voting strategy.
>
---
#### [replaced 029] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23897v2](http://arxiv.org/pdf/2506.23897v2)**

> **作者:** Longliang Liu; Miaojie Feng; Junda Cheng; Jijun Xiang; Xuan Zhu; Xin Yang
>
> **摘要:** Panoramic optical flow enables a comprehensive understanding of temporal dynamics across wide fields of view. However, severe distortions caused by sphere-to-plane projections, such as the equirectangular projection (ERP), significantly degrade the performance of conventional perspective-based optical flow methods, especially in polar regions. To address this challenge, we propose PriOr-Flow, a novel dual-branch framework that leverages the low-distortion nature of the orthogonal view to enhance optical flow estimation in these regions. Specifically, we introduce the Dual-Cost Collaborative Lookup (DCCL) operator, which jointly retrieves correlation information from both the primitive and orthogonal cost volumes, effectively mitigating distortion noise during cost volume construction. Furthermore, our Ortho-Driven Distortion Compensation (ODDC) module iteratively refines motion features from both branches, further suppressing polar distortions. Extensive experiments demonstrate that PriOr-Flow is compatible with various perspective-based iterative optical flow methods and consistently achieves state-of-the-art performance on publicly available panoramic optical flow datasets, setting a new benchmark for wide-field motion estimation. The code is publicly available at: https://github.com/longliangLiu/PriOr-Flow.
>
---
#### [replaced 030] DynaCLR: Contrastive Learning of Cellular Dynamics with Temporal Regularization
- **分类: cs.CV; q-bio.QM; I.2.6; J.3**

- **链接: [http://arxiv.org/pdf/2410.11281v2](http://arxiv.org/pdf/2410.11281v2)**

> **作者:** Eduardo Hirata-Miyasaki; Soorya Pradeep; Ziwen Liu; Alishba Imran; Taylla Milena Theodoro; Ivan E. Ivanov; Sudip Khadka; See-Chi Lee; Michelle Grunberg; Hunter Woosley; Madhura Bhave; Carolina Arias; Shalin B. Mehta
>
> **备注:** 30 pages, 6 figures, 13 appendix figures, 5 videos (ancillary files)
>
> **摘要:** We report DynaCLR, a self-supervised method for embedding cell and organelle Dynamics via Contrastive Learning of Representations of time-lapse images. DynaCLR integrates single-cell tracking and time-aware contrastive sampling to learn robust, temporally regularized representations of cell dynamics. DynaCLR embeddings generalize effectively to in-distribution and out-of-distribution datasets, and can be used for several downstream tasks with sparse human annotations. We demonstrate efficient annotations of cell states with a human-in-the-loop using fluorescence and label-free imaging channels. DynaCLR method enables diverse downstream biological analyses: classification of cell division and infection, clustering heterogeneous cell migration patterns, cross-modal distillation of cell states from fluorescence to label-free channel, alignment of asynchronous cellular responses and broken cell tracks, and discovering organelle response due to infection. DynaCLR is a flexible method for comparative analyses of dynamic cellular responses to pharmacological, microbial, and genetic perturbations. We provide PyTorch-based implementations of the model training and inference pipeline (https://github.com/mehta-lab/viscy) and a GUI (https://github.com/czbiohub-sf/napari-iohub) for the visualization and annotation of trajectories of cells in the real space and the embedding space.
>
---
#### [replaced 031] Contrastive Conditional Latent Diffusion for Audio-visual Segmentation
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2307.16579v2](http://arxiv.org/pdf/2307.16579v2)**

> **作者:** Yuxin Mao; Jing Zhang; Mochu Xiang; Yunqiu Lv; Dong Li; Yiran Zhong; Yuchao Dai
>
> **摘要:** We propose a contrastive conditional latent diffusion model for audio-visual segmentation (AVS) to thoroughly investigate the impact of audio, where the correlation between audio and the final segmentation map is modeled to guarantee the strong correlation between them. To achieve semantic-correlated representation learning, our framework incorporates a latent diffusion model. The diffusion model learns the conditional generation process of the ground-truth segmentation map, resulting in ground-truth aware inference during the denoising process at the test stage. As our model is conditional, it is vital to ensure that the conditional variable contributes to the model output. We thus extensively model the contribution of the audio signal by minimizing the density ratio between the conditional probability of the multimodal data, e.g. conditioned on the audio-visual data, and that of the unimodal data, e.g. conditioned on the audio data only. In this way, our latent diffusion model via density ratio optimization explicitly maximizes the contribution of audio for AVS, which can then be achieved with contrastive learning as a constraint, where the diffusion part serves as the main objective to achieve maximum likelihood estimation, and the density ratio optimization part imposes the constraint. By adopting this latent diffusion model via contrastive learning, we effectively enhance the contribution of audio for AVS. The effectiveness of our solution is validated through experimental results on the benchmark dataset. Code and results are online via our project page: https://github.com/OpenNLPLab/DiffusionAVS.
>
---
#### [replaced 032] UAV-DETR: Efficient End-to-End Object Detection for Unmanned Aerial Vehicle Imagery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.01855v3](http://arxiv.org/pdf/2501.01855v3)**

> **作者:** Huaxiang Zhang; Kai Liu; Zhongxue Gan; Guo-Niu Zhu
>
> **摘要:** Unmanned aerial vehicle object detection (UAV-OD) has been widely used in various scenarios. However, most existing UAV-OD algorithms rely on manually designed components, which require extensive tuning. End-to-end models that do not depend on such manually designed components are mainly designed for natural images, which are less effective for UAV imagery. To address such challenges, this paper proposes an efficient detection transformer (DETR) framework tailored for UAV imagery, i.e., UAV-DETR. The framework includes a multi-scale feature fusion with frequency enhancement module, which captures both spatial and frequency information at different scales. In addition, a frequency-focused down-sampling module is presented to retain critical spatial details during down-sampling. A semantic alignment and calibration module is developed to align and fuse features from different fusion paths. Experimental results demonstrate the effectiveness and generalization of our approach across various UAV imagery datasets. On the VisDrone dataset, our method improves AP by 3.1\% and $\text{AP}_{50}$ by 4.2\% over the baseline. Similar enhancements are observed on the UAVVaste dataset. The project page: https://github.com/ValiantDiligent/UAV-DETR
>
---
#### [replaced 033] PRVQL: Progressive Knowledge-guided Refinement for Robust Egocentric Visual Query Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07707v2](http://arxiv.org/pdf/2502.07707v2)**

> **作者:** Bing Fan; Yunhe Feng; Yapeng Tian; James Chenhao Liang; Yuewei Lin; Yan Huang; Heng Fan
>
> **摘要:** Egocentric visual query localization (EgoVQL) focuses on localizing the target of interest in space and time from first-person videos, given a visual query. Despite recent progressive, existing methods often struggle to handle severe object appearance changes and cluttering background in the video due to lacking sufficient target cues, leading to degradation. Addressing this, we introduce PRVQL, a novel Progressive knowledge-guided Refinement framework for EgoVQL. The core is to continuously exploit target-relevant knowledge directly from videos and utilize it as guidance to refine both query and video features for improving target localization. Our PRVQL contains multiple processing stages. The target knowledge from one stage, comprising appearance and spatial knowledge extracted via two specially designed knowledge learning modules, are utilized as guidance to refine the query and videos features for the next stage, which are used to generate more accurate knowledge for further feature refinement. With such a progressive process, target knowledge in PRVQL can be gradually improved, which, in turn, leads to better refined query and video features for localization in the final stage. Compared to previous methods, our PRVQL, besides the given object cues, enjoys additional crucial target information from a video as guidance to refine features, and hence enhances EgoVQL in complicated scenes. In our experiments on challenging Ego4D, PRVQL achieves state-of-the-art result and largely surpasses other methods, showing its efficacy. Our code, model and results will be released at https://github.com/fb-reps/PRVQL.
>
---
#### [replaced 034] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02091v3](http://arxiv.org/pdf/2502.02091v3)**

> **作者:** Joohyun Kwon; Hanbyel Cho; Junmo Kim
>
> **备注:** Accepted to CVPR 2025. The first two authors contributed equally
>
> **摘要:** Recent 4D dynamic scene editing methods require editing thousands of 2D images used for dynamic scene synthesis and updating the entire scene with additional training loops, resulting in several hours of processing to edit a single dynamic scene. Therefore, these methods are not scalable with respect to the temporal dimension of the dynamic scene (i.e., the number of timesteps). In this work, we propose Instruct-4DGS, an efficient dynamic scene editing method that is more scalable in terms of temporal dimension. To achieve computational efficiency, we leverage a 4D Gaussian representation that models a 4D dynamic scene by combining static 3D Gaussians with a Hexplane-based deformation field, which captures dynamic information. We then perform editing solely on the static 3D Gaussians, which is the minimal but sufficient component required for visual editing. To resolve the misalignment between the edited 3D Gaussians and the deformation field, which may arise from the editing process, we introduce a refinement stage using a score distillation mechanism. Extensive editing results demonstrate that Instruct-4DGS is efficient, reducing editing time by more than half compared to existing methods while achieving high-quality edits that better follow user instructions. Code and results: https://hanbyelcho.info/instruct-4dgs/
>
---
#### [replaced 035] BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22099v2](http://arxiv.org/pdf/2506.22099v2)**

> **作者:** Zipei Ma; Junzhe Jiang; Yurui Chen; Li Zhang
>
> **备注:** Accepted at ICCV 2025, Project Page: https://github.com/fudan-zvg/BezierGS
>
> **摘要:** The realistic reconstruction of street scenes is critical for developing real-world simulators in autonomous driving. Most existing methods rely on object pose annotations, using these poses to reconstruct dynamic objects and move them during the rendering process. This dependence on high-precision object annotations limits large-scale and extensive scene reconstruction. To address this challenge, we propose B\'ezier curve Gaussian splatting (B\'ezierGS), which represents the motion trajectories of dynamic objects using learnable B\'ezier curves. This approach fully leverages the temporal information of dynamic objects and, through learnable curve modeling, automatically corrects pose errors. By introducing additional supervision on dynamic object rendering and inter-curve consistency constraints, we achieve reasonable and accurate separation and reconstruction of scene elements. Extensive experiments on the Waymo Open Dataset and the nuPlan benchmark demonstrate that B\'ezierGS outperforms state-of-the-art alternatives in both dynamic and static scene components reconstruction and novel view synthesis.
>
---
#### [replaced 036] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.12787v3](http://arxiv.org/pdf/2411.12787v3)**

> **作者:** Pengkun Jiao; Bin Zhu; Jingjing Chen; Chong-Wah Ngo; Yu-Gang Jiang
>
> **备注:** ICCV 2025
>
> **摘要:** Efficient Visual Instruction Fine-Tuning (EVIT) seeks to adapt Multimodal Large Language Models (MLLMs) to downstream tasks with minimal computational overhead. However, as task diversity and complexity increase, EVIT faces significant challenges in resolving data conflicts. To address this limitation, we propose the Dual Low-Rank Adaptation (Dual-LoRA), a holistic-to-local framework that enhances the adapter's capacity to address data conflict through dual structural optimization. Specifically, we utilize two subspaces: a skill space for stable, holistic knowledge retention, and a rank-rectified task space that locally activates the holistic knowledge. Additionally, we introduce Visual Cue Enhancement (VCE), a multi-level local feature aggregation module designed to enrich the vision-language projection with local details. Our approach is both memory- and time-efficient, requiring only 1.16$\times$ the inference time of the standard LoRA method (with injection into the query and value projection layers), and just 73\% of the inference time of a 4-expert LoRA-MoE. Extensive experiments on various downstream tasks and general MLLM benchmarks validate the effectiveness of our proposed methods.
>
---
#### [replaced 037] StreakNet-Arch: An Anti-scattering Network-based Architecture for Underwater Carrier LiDAR-Radar Imaging
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.09158v3](http://arxiv.org/pdf/2404.09158v3)**

> **作者:** Xuelong Li; Hongjun An; Haofei Zhao; Guangying Li; Bo Liu; Xing Wang; Guanghua Cheng; Guojun Wu; Zhe Sun
>
> **备注:** Accepted by IEEE Transactions on Image Processing (T-IP)
>
> **摘要:** In this paper, we introduce StreakNet-Arch, a real-time, end-to-end binary-classification framework based on our self-developed Underwater Carrier LiDAR-Radar (UCLR) that embeds Self-Attention and our novel Double Branch Cross Attention (DBC-Attention) to enhance scatter suppression. Under controlled water tank validation conditions, StreakNet-Arch with Self-Attention or DBC-Attention outperforms traditional bandpass filtering and achieves higher $F_1$ scores than learning-based MP networks and CNNs at comparable model size and complexity. Real-time benchmarks on an NVIDIA RTX 3060 show a constant Average Imaging Time (54 to 84 ms) regardless of frame count, versus a linear increase (58 to 1,257 ms) for conventional methods. To facilitate further research, we contribute a publicly available streak-tube camera image dataset contains 2,695,168 real-world underwater 3D point cloud data. More importantly, we validate our UCLR system in a South China Sea trial, reaching an error of 46mm for 3D target at 1,000 m depth and 20 m range. Source code and data are available at https://github.com/BestAnHongjun/StreakNet .
>
---
#### [replaced 038] Enabling Collaborative Parametric Knowledge Calibration for Retrieval-Augmented Vision Question Answering
- **分类: cs.CV; cs.IR; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.04065v2](http://arxiv.org/pdf/2504.04065v2)**

> **作者:** Jiaqi Deng; Kaize Shi; Zonghan Wu; Huan Huo; Dingxian Wang; Guandong Xu
>
> **备注:** 10 pages, 5 figures, Under Review
>
> **摘要:** Knowledge-based Vision Question Answering (KB-VQA) systems address complex visual-grounded questions with knowledge retrieved from external knowledge bases. The tasks of knowledge retrieval and answer generation tasks both necessitate precise multimodal understanding of question context and external knowledge. However, existing methods treat these two stages as separate modules with limited interaction during training, which hinders bi-directional parametric knowledge sharing, ultimately leading to suboptimal performance. To fully exploit the cross-task synergy in KB-VQA, we propose a unified retrieval-augmented VQA framework with collaborative parametric knowledge calibration. The proposed framework can effectively adapt general multimodal pre-trained models for fine-grained, knowledge-intensive tasks while enabling the retriever and generator to collaboratively enhance and share their parametric knowledge during both training and inference. To enhance fine-grained understanding of questions and external documents, we also integrate late interaction mechanism into the proposed training framework. Additionally, we introduce a reflective-answering mechanism that allows the model to explicitly evaluate and refine its knowledge boundary. Our approach achieves competitive performance against state-of-the-art models, delivering a significant 4.7\% improvement in answering accuracy, and brings an average 7.5\% boost in base MLLMs' VQA performance.
>
---
#### [replaced 039] AdaptoVision: A Multi-Resolution Image Recognition Model for Robust and Scalable Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12652v2](http://arxiv.org/pdf/2504.12652v2)**

> **作者:** Md. Sanaullah Chowdhury Lameya Sabrin
>
> **摘要:** This paper introduces AdaptoVision, a novel convolutional neural network (CNN) architecture designed to efficiently balance computational complexity and classification accuracy. By leveraging enhanced residual units, depth-wise separable convolutions, and hierarchical skip connections, AdaptoVision significantly reduces parameter count and computational requirements while preserving competitive performance across various benchmark and medical image datasets. Extensive experimentation demonstrates that AdaptoVision achieves state-of-the-art on BreakHis dataset and comparable accuracy levels, notably 95.3\% on CIFAR-10 and 85.77\% on CIFAR-100, without relying on any pretrained weights. The model's streamlined architecture and strategic simplifications promote effective feature extraction and robust generalization, making it particularly suitable for deployment in real-time and resource-constrained environments.
>
---
#### [replaced 040] Enhanced Controllability of Diffusion Models via Feature Disentanglement and Realism-Enhanced Sampling Methods
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2302.14368v5](http://arxiv.org/pdf/2302.14368v5)**

> **作者:** Wonwoong Cho; Hareesh Ravi; Midhun Harikumar; Vinh Khuc; Krishna Kumar Singh; Jingwan Lu; David I. Inouye; Ajinkya Kale
>
> **备注:** ECCV 2024; Code is available at https://github.com/WonwoongCho/Towards-Enhanced-Controllability-of-Diffusion-Models
>
> **摘要:** As Diffusion Models have shown promising performance, a lot of efforts have been made to improve the controllability of Diffusion Models. However, how to train Diffusion Models to have the disentangled latent spaces and how to naturally incorporate the disentangled conditions during the sampling process have been underexplored. In this paper, we present a training framework for feature disentanglement of Diffusion Models (FDiff). We further propose two sampling methods that can boost the realism of our Diffusion Models and also enhance the controllability. Concisely, we train Diffusion Models conditioned on two latent features, a spatial content mask, and a flattened style embedding. We rely on the inductive bias of the denoising process of Diffusion Models to encode pose/layout information in the content feature and semantic/style information in the style feature. Regarding the sampling methods, we first generalize Composable Diffusion Models (GCDM) by breaking the conditional independence assumption to allow for some dependence between conditional inputs, which is shown to be effective in realistic generation in our experiments. Second, we propose timestep-dependent weight scheduling for content and style features to further improve the performance. We also observe better controllability of our proposed methods compared to existing methods in image manipulation and image translation.
>
---
#### [replaced 041] Listener-Rewarded Thinking in VLMs for Image Preferences
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.22832v2](http://arxiv.org/pdf/2506.22832v2)**

> **作者:** Alexander Gambashidze; Li Pengyi; Matvey Skripkin; Andrey Galichin; Anton Gusarov; Konstantin Sobolev; Andrey Kuznetsov; Ivan Oseledets
>
> **摘要:** Training robust and generalizable reward models for human visual preferences is essential for aligning text-to-image and text-to-video generative models with human intent. However, current reward models often fail to generalize, and supervised fine-tuning leads to memorization, demanding complex annotation pipelines. While reinforcement learning (RL), specifically Group Relative Policy Optimization (GRPO), improves generalization, we uncover a key failure mode: a significant drop in reasoning accuracy occurs when a model's reasoning trace contradicts that of an independent, frozen vision-language model ("listener") evaluating the same output. To address this, we introduce a listener-augmented GRPO framework. Here, the listener re-evaluates the reasoner's chain-of-thought to provide a dense, calibrated confidence score, shaping the RL reward signal. This encourages the reasoner not only to answer correctly, but to produce explanations that are persuasive to an independent model. Our listener-shaped reward scheme achieves best accuracy on the ImageReward benchmark (67.4%), significantly improves out-of-distribution (OOD) performance on a large-scale human preference dataset (1.2M votes, up to +6% over naive reasoner), and reduces reasoning contradictions compared to strong GRPO and SFT baselines. These results demonstrate that listener-based rewards provide a scalable, data-efficient path to aligning vision-language models with nuanced human preferences. We will release our reasoning model here: https://huggingface.co/alexgambashidze/qwen2.5vl_image_preference_reasoner.
>
---
#### [replaced 042] Depth Matters: Exploring Deep Interactions of RGB-D for Semantic Segmentation in Traffic Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.07995v2](http://arxiv.org/pdf/2409.07995v2)**

> **作者:** Siyu Chen; Ting Han; Changshe Zhang; Weiquan Liu; Jinhe Su; Zongyue Wang; Guorong Cai
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** RGB-D has gradually become a crucial data source for understanding complex scenes in assisted driving. However, existing studies have paid insufficient attention to the intrinsic spatial properties of depth maps. This oversight significantly impacts the attention representation, leading to prediction errors caused by attention shift issues. To this end, we propose a novel learnable Depth interaction Pyramid Transformer (DiPFormer) to explore the effectiveness of depth. Firstly, we introduce Depth Spatial-Aware Optimization (Depth SAO) as offset to represent real-world spatial relationships. Secondly, the similarity in the feature space of RGB-D is learned by Depth Linear Cross-Attention (Depth LCA) to clarify spatial differences at the pixel level. Finally, an MLP Decoder is utilized to effectively fuse multi-scale features for meeting real-time requirements. Comprehensive experiments demonstrate that the proposed DiPFormer significantly addresses the issue of attention misalignment in both road detection (+7.5%) and semantic segmentation (+4.9% / +1.5%) tasks. DiPFormer achieves state-of-the-art performance on the KITTI (97.57% F-score on KITTI road and 68.74% mIoU on KITTI-360) and Cityscapes (83.4% mIoU) datasets.
>
---
#### [replaced 043] Soft Dice Confidence: A Near-Optimal Confidence Estimator for Selective Prediction in Semantic Segmentation
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2402.10665v3](http://arxiv.org/pdf/2402.10665v3)**

> **作者:** Bruno Laboissiere Camargos Borges; Bruno Machado Pacheco; Danilo Silva
>
> **备注:** 42 pages, 9 figures
>
> **摘要:** Selective prediction augments a model with the option to abstain from providing unreliable predictions. The key ingredient is a confidence score function, which should be directly related to the conditional risk. In the case of binary semantic segmentation, existing score functions either ignore the particularities of the evaluation metric or demand additional held-out data for tuning. We propose the Soft Dice Confidence (SDC), a simple, tuning-free confidence score function that directly aligns with the Dice coefficient metric. We prove that, under conditional independence, the SDC is near optimal: we establish upper and lower bounds on the ratio between the SDC and the ideal (intractable) confidence score function and show that these bounds are very close to 1. Experiments on six public medical-imaging benchmarks and on synthetic data corroborate our theoretical findings. In fact, SDC outperformed all prior confidence estimators from the literature in all of our experiments, including those that rely on additional data. These results position SDC as a reliable and efficient confidence estimator for selective prediction in semantic segmentation.
>
---
#### [replaced 044] VideoCogQA: A Controllable Benchmark for Evaluating Cognitive Abilities in Video-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.09105v2](http://arxiv.org/pdf/2411.09105v2)**

> **作者:** Chenglin Li; Qianglong Chen; Zhi Li; Feng Tao; Yin Zhang
>
> **摘要:** Recent advancements in Large Video-Language Models (LVLMs) have led to promising results in multimodal video understanding. However, it remains unclear whether these models possess the cognitive capabilities required for high-level tasks, particularly those involving symbolic and abstract perception. Existing benchmarks typically rely on real-world, annotated videos, which lack control over video content and inherent difficulty, limiting their diagnostic power. To bridge this gap, we propose VideoCogQA, a scalable and fully controllable benchmark inspired by game-world environments, designed to evaluate the cognitive abilities of LVLMs. By generating synthetic videos via a programmatic engine, VideoCogQA allows fine-grained control over visual elements, temporal dynamics, and task difficulty. This approach enables a focused evaluation of video cognitive abilities, independent of prior knowledge from visual scene semantics. The dataset includes 800 videos and 3,280 question-answer pairs, featuring tasks related to abstract concepts, symbolic elements, and multimodal integration, with varying levels of difficulty. Experimental results show that even state-of-the-art (SOTA) models, such as GPT-4o, achieve an average performance of 48.8% on tasks involving abstract concepts. Additionally, performance drops by 15% as task complexity increases, highlighting the challenges LVLMs face in maintaining consistent performance. Through this work, we hope to show the limitations of current LVLMs and offer insights into how they can more effectively emulate human cognitive processes in the future.
>
---
#### [replaced 045] Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14156v2](http://arxiv.org/pdf/2502.14156v2)**

> **作者:** Katie Z Luo; Minh-Quan Dao; Zhenzhen Liu; Mark Campbell; Wei-Lun Chao; Kilian Q. Weinberger; Ezio Malis; Vincent Fremont; Bharath Hariharan; Mao Shan; Stewart Worrall; Julie Stephany Berrio Perez
>
> **摘要:** Vehicle-to-everything (V2X) collaborative perception has emerged as a promising solution to address the limitations of single-vehicle perception systems. However, existing V2X datasets are limited in scope, diversity, and quality. To address these gaps, we present Mixed Signals, a comprehensive V2X dataset featuring 45.1k point clouds and 240.6k bounding boxes collected from three connected autonomous vehicles (CAVs) equipped with two different configurations of LiDAR sensors, plus a roadside unit with dual LiDARs. Our dataset provides point clouds and bounding box annotations across 10 classes, ensuring reliable data for perception training. We provide detailed statistical analysis on the quality of our dataset and extensively benchmark existing V2X methods on it. The Mixed Signals dataset is ready-to-use, with precise alignment and consistent annotations across time and viewpoints. Dataset website is available at https://mixedsignalsdataset.cs.cornell.edu/.
>
---
#### [replaced 046] Da Yu: Towards USV-Based Image Captioning for Waterway Surveillance and Scene Understanding
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.19288v2](http://arxiv.org/pdf/2506.19288v2)**

> **作者:** Runwei Guan; Ningwei Ouyang; Tianhao Xu; Shaofeng Liang; Wei Dai; Yafeng Sun; Shang Gao; Songning Lai; Shanliang Yao; Xuming Hu; Ryan Wen Liu; Yutao Yue; Hui Xiong
>
> **备注:** 14 pages, 13 figures
>
> **摘要:** Automated waterway environment perception is crucial for enabling unmanned surface vessels (USVs) to understand their surroundings and make informed decisions. Most existing waterway perception models primarily focus on instance-level object perception paradigms (e.g., detection, segmentation). However, due to the complexity of waterway environments, current perception datasets and models fail to achieve global semantic understanding of waterways, limiting large-scale monitoring and structured log generation. With the advancement of vision-language models (VLMs), we leverage image captioning to introduce WaterCaption, the first captioning dataset specifically designed for waterway environments. WaterCaption focuses on fine-grained, multi-region long-text descriptions, providing a new research direction for visual geo-understanding and spatial scene cognition. Exactly, it includes 20.2k image-text pair data with 1.8 million vocabulary size. Additionally, we propose Da Yu, an edge-deployable multi-modal large language model for USVs, where we propose a novel vision-to-language projector called Nano Transformer Adaptor (NTA). NTA effectively balances computational efficiency with the capacity for both global and fine-grained local modeling of visual features, thereby significantly enhancing the model's ability to generate long-form textual outputs. Da Yu achieves an optimal balance between performance and efficiency, surpassing state-of-the-art models on WaterCaption and several other captioning benchmarks.
>
---
#### [replaced 047] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24625v2](http://arxiv.org/pdf/2505.24625v2)**

> **作者:** Duo Zheng; Shijia Huang; Yanyang Li; Liwei Wang
>
> **摘要:** Previous research has investigated the application of Multimodal Large Language Models (MLLMs) in understanding 3D scenes by interpreting them as videos. These approaches generally depend on comprehensive 3D data inputs, such as point clouds or reconstructed Bird's-Eye View (BEV) maps. In our research, we advance this field by enhancing the capability of MLLMs to understand and reason in 3D spaces directly from video data, without the need for additional 3D input. We propose a novel and efficient method, the Video-3D Geometry Large Language Model (VG LLM). Our approach employs a 3D visual geometry encoder that extracts 3D prior information from video sequences. This information is integrated with visual tokens and fed into the MLLM. Extensive experiments have shown that our method has achieved substantial improvements in various tasks related to 3D scene understanding and spatial reasoning, all directly learned from video sources. Impressively, our 4B model, which does not rely on explicit 3D data inputs, achieves competitive results compared to existing state-of-the-art methods, and even surpasses the Gemini-1.5-Pro in the VSI-Bench evaluations.
>
---
#### [replaced 048] Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.10967v2](http://arxiv.org/pdf/2506.10967v2)**

> **作者:** Qizhe Zhang; Mengzhen Liu; Lichen Li; Ming Lu; Yuan Zhang; Junwen Pan; Qi She; Shanghang Zhang
>
> **备注:** 22 pages, 5 figures, code: https://github.com/Theia-4869/CDPruner, project page: https://theia-4869.github.io/CDPruner
>
> **摘要:** In multimodal large language models (MLLMs), the length of input visual tokens is often significantly greater than that of their textual counterparts, leading to a high inference cost. Many works aim to address this issue by removing redundant visual tokens. However, current approaches either rely on attention-based pruning, which retains numerous duplicate tokens, or use similarity-based pruning, overlooking the instruction relevance, consequently causing suboptimal performance. In this paper, we go beyond attention or similarity by proposing a novel visual token pruning method named CDPruner, which maximizes the conditional diversity of retained tokens. We first define the conditional similarity between visual tokens conditioned on the instruction, and then reformulate the token pruning problem with determinantal point process (DPP) to maximize the conditional diversity of the selected subset. The proposed CDPruner is training-free and model-agnostic, allowing easy application to various MLLMs. Extensive experiments across diverse MLLMs show that CDPruner establishes new state-of-the-art on various vision-language benchmarks. By maximizing conditional diversity through DPP, the selected subset better represents the input images while closely adhering to user instructions, thereby preserving strong performance even with high reduction ratios. When applied to LLaVA, CDPruner reduces FLOPs by 95\% and CUDA latency by 78\%, while maintaining 94\% of the original accuracy. Our code is available at https://github.com/Theia-4869/CDPruner.
>
---
#### [replaced 049] The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.03628v2](http://arxiv.org/pdf/2502.03628v2)**

> **作者:** Zhuowei Li; Haizhou Shi; Yunhe Gao; Di Liu; Zhenting Wang; Yuxiao Chen; Ting Liu; Long Zhao; Hao Wang; Dimitris N. Metaxas
>
> **摘要:** Large Vision-Language Models (LVLMs) can reason effectively over both textual and visual inputs, but they tend to hallucinate syntactically coherent yet visually ungrounded contents. In this paper, we investigate the internal dynamics of hallucination by examining the tokens logits ranking throughout the generation process, revealing three key patterns in how LVLMs process information: (1) gradual visual information loss - visually grounded tokens gradually become less favored throughout generation, and (2) early excitation - semantically meaningful tokens achieve peak activation in the layers earlier than the final layer. (3) hidden genuine information - visually grounded tokens though not being eventually decoded still retain relatively high rankings at inference. Based on these insights, we propose VISTA (Visual Information Steering with Token-logit Augmentation), a training-free inference-time intervention framework that reduces hallucination while promoting genuine information. VISTA works by combining two complementary approaches: reinforcing visual information in activation space and leveraging early layer activations to promote semantically meaningful decoding. Compared to existing methods, VISTA requires no external supervision and is applicable to various decoding strategies. Extensive experiments show that VISTA on average reduces hallucination by about 40% on evaluated open-ended generation task, and it consistently outperforms existing methods on four benchmarks across four architectures under three decoding strategies. Code is available at https://github.com/LzVv123456/VISTA.
>
---
#### [replaced 050] A Dataset for Enhancing MLLMs in Visualization Understanding and Reconstruction
- **分类: cs.HC; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21319v2](http://arxiv.org/pdf/2506.21319v2)**

> **作者:** Can Liu; Chunlin Da; Xiaoxiao Long; Yuxiao Yang; Yu Zhang; Yong Wang
>
> **摘要:** Current multimodal large language models (MLLMs), while effective in natural image understanding, struggle with visualization understanding due to their inability to decode the data-to-visual mapping and extract structured information. To address these challenges, we propose SimVec, a compact and structured vector format that encodes chart elements, including mark types, positions, and sizes. Then, we present a new visualization dataset, which consists of bitmap images of charts, their corresponding SimVec representations, and data-centric question-answering pairs, each accompanied by explanatory chain-of-thought sentences. We fine-tune state-of-the-art MLLMs using our dataset. The experimental results show that fine-tuning leads to substantial improvements in data-centric reasoning tasks compared to their zero-shot versions. SimVec also enables MLLMs to accurately and compactly reconstruct chart structures from images. Our dataset and code are available at: https://github.com/VIDA-Lab/MLLM4VIS.
>
---
#### [replaced 051] ZonUI-3B: A Lightweight Vision-Language Model for Cross-Resolution GUI Grounding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.23491v2](http://arxiv.org/pdf/2506.23491v2)**

> **作者:** ZongHan Hsieh; Tzer-Jen Wei; ShengJing Yang
>
> **摘要:** This paper introduces ZonUI-3B, a lightweight Vision-Language Model (VLM) specifically designed for Graphical User Interface grounding tasks, achieving performance competitive with significantly larger models. Unlike large-scale VLMs (>7B parameters) that are computationally intensive and impractical for consumer-grade hardware, ZonUI-3B delivers strong grounding accuracy while being fully trainable on a single GPU (RTX 4090). The model incorporates several key innovations: (i) combine cross-platform, multi-resolution dataset of 24K examples from diverse sources including mobile, desktop, and web GUI screenshots to effectively address data scarcity in high-resolution desktop environments; (ii) a two-stage fine-tuning strategy, where initial cross-platform training establishes robust GUI understanding, followed by specialized fine-tuning on high-resolution data to significantly enhance model adaptability; and (iii) data curation and redundancy reduction strategies, demonstrating that randomly sampling a smaller subset with reduced redundancy achieves performance comparable to larger datasets, emphasizing data diversity over sheer volume. Empirical evaluation on standard GUI grounding benchmarks-including ScreenSpot, ScreenSpot-v2, and the challenging ScreenSpot-Pro, highlights ZonUI-3B's exceptional accuracy, achieving 84.9% on ScreenSpot and 86.4% on ScreenSpot-v2, surpassing prior models under 4B parameters. Ablation studies validate the critical role of balanced sampling and two-stage fine-tuning in enhancing robustness, particularly in high-resolution desktop scenarios. The ZonUI-3B is available at: https://github.com/Han1018/ZonUI-3B
>
---
#### [replaced 052] Beyond Diagnostic Performance: Revealing and Quantifying Ethical Risks in Pathology Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16889v2](http://arxiv.org/pdf/2502.16889v2)**

> **作者:** Weiping Lin; Shen Liu; Runchen Zhu; Yixuan Lin; Baoshun Wang; Liansheng Wang
>
> **备注:** 33 pages,5 figure,23 tables
>
> **摘要:** Pathology foundation models (PFMs), as large-scale pre-trained models tailored for computational pathology, have significantly advanced a wide range of applications. Their ability to leverage prior knowledge from massive datasets has streamlined the development of intelligent pathology models. However, we identify several critical and interrelated ethical risks that remain underexplored, yet must be addressed to enable the safe translation of PFMs from lab to clinic. These include the potential leakage of patient-sensitive attributes, disparities in model performance across demographic and institutional subgroups, and the reliance on diagnosis-irrelevant features that undermine clinical reliability. In this study, we pioneer the quantitative analysis for ethical risks in PFMs, including privacy leakage, clinical reliability, and group fairness. Specifically, we propose an evaluation framework that systematically measures key dimensions of ethical concern: the degree to which patient-sensitive attributes can be inferred from model representations, the extent of performance disparities across demographic and institutional subgroups, and the influence of diagnostically irrelevant features on model decisions. We further investigate the underlying causes of these ethical risks in PFMs and empirically validate our findings. Then we offer insights into potential directions for mitigating such risks, aiming to inform the development of more ethically robust PFMs. This work provides the first quantitative and systematic evaluation of ethical risks in PFMs. Our findings highlight the urgent need for ethical safeguards in PFMs and offer actionable insights for building more trustworthy and clinically robust PFMs. To facilitate future research and deployment, we will release the assessment framework as an online toolkit to support the development, auditing, and deployment of ethically robust PFMs.
>
---
#### [replaced 053] Unleashing Diffusion and State Space Models for Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.12747v2](http://arxiv.org/pdf/2506.12747v2)**

> **作者:** Rong Wu; Ziqi Chen; Liming Zhong; Heng Li; Hai Shu
>
> **摘要:** Existing segmentation models trained on a single medical imaging dataset often lack robustness when encountering unseen organs or tumors. Developing a robust model capable of identifying rare or novel tumor categories not present during training is crucial for advancing medical imaging applications. We propose DSM, a novel framework that leverages diffusion and state space models to segment unseen tumor categories beyond the training data. DSM utilizes two sets of object queries trained within modified attention decoders to enhance classification accuracy. Initially, the model learns organ queries using an object-aware feature grouping strategy to capture organ-level visual features. It then refines tumor queries by focusing on diffusion-based visual prompts, enabling precise segmentation of previously unseen tumors. Furthermore, we incorporate diffusion-guided feature fusion to improve semantic segmentation performance. By integrating CLIP text embeddings, DSM captures category-sensitive classes to improve linguistic transfer knowledge, thereby enhancing the model's robustness across diverse scenarios and multi-label tasks. Extensive experiments demonstrate the superior performance of DSM in various tumor segmentation tasks. Code is available at https://github.com/Rows21/k-Means_Mask_Mamba.
>
---
#### [replaced 054] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Radiology with Zero-Shot Multi-Task Capability
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.07416v2](http://arxiv.org/pdf/2504.07416v2)**

> **作者:** Jonggwon Park; Soobum Kim; Byungmu Yoon; Kyoyun Choi
>
> **摘要:** Recent advancements in multi-modal models have significantly improved vision-language (VL) alignment in radiology. However, existing approaches struggle to effectively utilize complex radiology reports for learning and offer limited interpretability through attention probability visualizations. To address these challenges, we introduce RadZero, a novel framework for VL alignment in radiology with zero-shot multi-task capability. A key component of our approach is VL-CABS (Vision-Language Cross-Attention Based on Similarity), which aligns text embeddings with local image features for interpretable, fine-grained VL reasoning. RadZero leverages large language models to extract concise semantic sentences from radiology reports and employs multi-positive contrastive training to effectively capture relationships between images and multiple relevant textual descriptions. It uses a pre-trained vision encoder with additional trainable Transformer layers, allowing efficient high-resolution image processing. By computing similarity between text embeddings and local image patch features, VL-CABS enables zero-shot inference with similarity probability for classification, and pixel-level VL similarity maps for grounding and segmentation. Experimental results on public chest radiograph benchmarks show that RadZero outperforms state-of-the-art methods in zero-shot classification, grounding, and segmentation. Furthermore, VL similarity map analysis highlights the potential of VL-CABS for improving explainability in VL alignment. Additionally, qualitative evaluation demonstrates RadZero's capability for open-vocabulary semantic segmentation, further validating its effectiveness in medical imaging.
>
---
#### [replaced 055] Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2401.03302v4](http://arxiv.org/pdf/2401.03302v4)**

> **作者:** Seyed Mohammad Hossein Hashemi; Leila Safari; Mohsen Hooshmand; Amirhossein Dadashzadeh Taromi
>
> **摘要:** Reliable diagnosis of brain tumors remains challenging due to low clinical incidence rates of such cases. However, this low rate is neglected in most of proposed methods. We propose a clinically inspired framework for anomaly-resilient tumor detection and classification. Detection leverages YOLOv8n fine-tuned on a realistically imbalanced dataset (1:9 tumor-to-normal ratio; 30,000 MRI slices from 81 patients). In addition, we propose a novel Patient-to-Patient (PTP) metric that evaluates diagnostic reliability at the patient level. Classification employs knowledge distillation: a Data Efficient Image Transformer (DeiT) student model is distilled from a ResNet152 teacher. The distilled ViT achieves an F1-score of 0.92 within 20 epochs, matching near teacher performance (F1=0.97) with significantly reduced computational resources. This end-to-end framework demonstrates high robustness in clinically representative anomaly-distributed data, offering a viable tool that adheres to realistic situations in clinics.
>
---
#### [replaced 056] Prompt-Guided Latent Diffusion with Predictive Class Conditioning for 3D Prostate MRI Generation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10230v2](http://arxiv.org/pdf/2506.10230v2)**

> **作者:** Emerson P. Grabke; Masoom A. Haider; Babak Taati
>
> **备注:** MAH and BT are co-senior authors on the work. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Objective: Latent diffusion models (LDM) could alleviate data scarcity challenges affecting machine learning development for medical imaging. However, medical LDM strategies typically rely on short-prompt text encoders, non-medical LDMs, or large data volumes. These strategies can limit performance and scientific accessibility. We propose a novel LDM conditioning approach to address these limitations. Methods: We propose Class-Conditioned Efficient Large Language model Adapter (CCELLA), a novel dual-head conditioning approach that simultaneously conditions the LDM U-Net with free-text clinical reports and radiology classification. We also propose a data-efficient LDM framework centered around CCELLA and a proposed joint loss function. We first evaluate our method on 3D prostate MRI against state-of-the-art. We then augment a downstream classifier model training dataset with synthetic images from our method. Results: Our method achieves a 3D FID score of 0.025 on a size-limited 3D prostate MRI dataset, significantly outperforming a recent foundation model with FID 0.071. When training a classifier for prostate cancer prediction, adding synthetic images generated by our method during training improves classifier accuracy from 69% to 74%. Training a classifier solely on our method's synthetic images achieved comparable performance to training on real images alone. Conclusion: We show that our method improved both synthetic image quality and downstream classifier performance using limited data and minimal human annotation. Significance: The proposed CCELLA-centric framework enables radiology report and class-conditioned LDM training for high-quality medical image synthesis given limited data volume and human data annotation, improving LDM performance and scientific accessibility. Code from this study will be available at https://github.com/grabkeem/CCELLA
>
---
#### [replaced 057] Dehazing Light Microscopy Images with Guided Conditional Flow Matching: finding a sweet spot between fidelity and realism
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22397v3](http://arxiv.org/pdf/2506.22397v3)**

> **作者:** Anirban Ray; Ashesh; Florian Jug
>
> **备注:** 4 figures, 10 pages + refs, 40 pages total (including supplement), 24 supplementary figures
>
> **摘要:** Fluorescence microscopy is a major driver of scientific progress in the life sciences. Although high-end confocal microscopes are capable of filtering out-of-focus light, cheaper and more accessible microscopy modalities, such as widefield microscopy, can not, which consequently leads to hazy image data. Computational dehazing is trying to combine the best of both worlds, leading to cheap microscopy but crisp-looking images. The perception-distortion trade-off tells us that we can optimize either for data fidelity, e.g. low MSE or high PSNR, or for data realism, measured by perceptual metrics such as LPIPS or FID. Existing methods either prioritize fidelity at the expense of realism, or produce perceptually convincing results that lack quantitative accuracy. In this work, we propose HazeMatching, a novel iterative method for dehazing light microscopy images, which effectively balances these objectives. Our goal was to find a balanced trade-off between the fidelity of the dehazing results and the realism of individual predictions (samples). We achieve this by adapting the conditional flow matching framework by guiding the generative process with a hazy observation in the conditional velocity field. We evaluate HazeMatching on 5 datasets, covering both synthetic and real data, assessing both distortion and perceptual quality. Our method is compared against 7 baselines, achieving a consistent balance between fidelity and realism on average. Additionally, with calibration analysis, we show that HazeMatching produces well-calibrated predictions. Note that our method does not need an explicit degradation operator to exist, making it easily applicable on real microscopy data. All data used for training and evaluation and our code will be publicly available under a permissive license.
>
---
#### [replaced 058] Bridging SFT and DPO for Diffusion Model Alignment with Self-Sampling Preference Optimization
- **分类: cs.CV; cs.LG; I.2.6; I.2.10; I.4.0; I.5.0**

- **链接: [http://arxiv.org/pdf/2410.05255v2](http://arxiv.org/pdf/2410.05255v2)**

> **作者:** Daoan Zhang; Guangchen Lan; Dong-Jun Han; Wenlin Yao; Xiaoman Pan; Hongming Zhang; Mingxiao Li; Pengcheng Chen; Yu Dong; Christopher Brinton; Jiebo Luo
>
> **摘要:** Existing post-training techniques are broadly categorized into supervised fine-tuning (SFT) and reinforcement learning (RL) methods; the former is stable during training but suffers from limited generalization, while the latter, despite its stronger generalization capability, relies on additional preference data or reward models and carries the risk of reward exploitation. In order to preserve the advantages of both SFT and RL -- namely, eliminating the need for paired data and reward models while retaining the training stability of SFT and the generalization ability of RL -- a new alignment method, Self-Sampling Preference Optimization (SSPO), is proposed in this paper. SSPO introduces a Random Checkpoint Replay (RCR) strategy that utilizes historical checkpoints to construct paired data, thereby effectively mitigating overfitting. Simultaneously, a Self-Sampling Regularization (SSR) strategy is employed to dynamically evaluate the quality of generated samples; when the generated samples are more likely to be winning samples, the approach automatically switches from DPO (Direct Preference Optimization) to SFT, ensuring that the training process accurately reflects the quality of the samples. Experimental results demonstrate that SSPO not only outperforms existing methods on text-to-image benchmarks, but its effectiveness has also been validated in text-to-video tasks. We validate SSPO across both text-to-image and text-to-video benchmarks. SSPO surpasses all previous approaches on the text-to-image benchmarks and demonstrates outstanding performance on the text-to-video benchmarks.
>
---
#### [replaced 059] Ovis-U1 Technical Report
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.23044v2](http://arxiv.org/pdf/2506.23044v2)**

> **作者:** Guo-Hua Wang; Shanshan Zhao; Xinjie Zhang; Liangfu Cao; Pengxin Zhan; Lunhao Duan; Shiyin Lu; Minghao Fu; Xiaohao Chen; Jianshan Zhao; Yang Li; Qing-Guo Chen
>
> **备注:** An unified model for multimodal understanding, text-to-image generation, and image editing. GitHub: https://github.com/AIDC-AI/Ovis-U1
>
> **摘要:** In this report, we introduce Ovis-U1, a 3-billion-parameter unified model that integrates multimodal understanding, text-to-image generation, and image editing capabilities. Building on the foundation of the Ovis series, Ovis-U1 incorporates a diffusion-based visual decoder paired with a bidirectional token refiner, enabling image generation tasks comparable to leading models like GPT-4o. Unlike some previous models that use a frozen MLLM for generation tasks, Ovis-U1 utilizes a new unified training approach starting from a language model. Compared to training solely on understanding or generation tasks, unified training yields better performance, demonstrating the enhancement achieved by integrating these two tasks. Ovis-U1 achieves a score of 69.6 on the OpenCompass Multi-modal Academic Benchmark, surpassing recent state-of-the-art models such as Ristretto-3B and SAIL-VL-1.5-2B. In text-to-image generation, it excels with scores of 83.72 and 0.89 on the DPG-Bench and GenEval benchmarks, respectively. For image editing, it achieves 4.00 and 6.42 on the ImgEdit-Bench and GEdit-Bench-EN, respectively. As the initial version of the Ovis unified model series, Ovis-U1 pushes the boundaries of multimodal understanding, generation, and editing.
>
---
#### [replaced 060] De-LightSAM: Modality-Decoupled Lightweight SAM for Generalizable Medical Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.14153v5](http://arxiv.org/pdf/2407.14153v5)**

> **作者:** Qing Xu; Jiaxuan Li; Xiangjian He; Chenxin Li; Fiseha B. Tesem; Wenting Duan; Zhen Chen; Rong Qu; Jonathan M. Garibaldi; Chang Wen Chen
>
> **备注:** Under Review
>
> **摘要:** The universality of deep neural networks across different modalities and their generalization capabilities to unseen domains play an essential role in medical image segmentation. The recent segment anything model (SAM) has demonstrated strong adaptability across diverse natural scenarios. However, the huge computational costs, demand for manual annotations as prompts and conflict-prone decoding process of SAM degrade its generalization capabilities in medical scenarios. To address these limitations, we propose a modality-decoupled lightweight SAM for domain-generalized medical image segmentation, named De-LightSAM. Specifically, we first devise a lightweight domain-controllable image encoder (DC-Encoder) that produces discriminative visual features for diverse modalities. Further, we introduce the self-patch prompt generator (SP-Generator) to automatically generate high-quality dense prompt embeddings for guiding segmentation decoding. Finally, we design the query-decoupled modality decoder (QM-Decoder) that leverages a one-to-one strategy to provide an independent decoding channel for every modality, preventing mutual knowledge interference of different modalities. Moreover, we design a multi-modal decoupled knowledge distillation (MDKD) strategy to leverage robust common knowledge to complement domain-specific medical feature representations. Extensive experiments indicate that De-LightSAM outperforms state-of-the-arts in diverse medical imaging segmentation tasks, displaying superior modality universality and generalization capabilities. Especially, De-LightSAM uses only 2.0% parameters compared to SAM-H. The source code is available at https://github.com/xq141839/De-LightSAM.
>
---
#### [replaced 061] T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00703v2](http://arxiv.org/pdf/2505.00703v2)**

> **作者:** Dongzhi Jiang; Ziyu Guo; Renrui Zhang; Zhuofan Zong; Hao Li; Le Zhuo; Shilin Yan; Pheng-Ann Heng; Hongsheng Li
>
> **备注:** Project Page: https://github.com/CaraJ7/T2I-R1
>
> **摘要:** Recent advancements in large language models have demonstrated how chain-of-thought (CoT) and reinforcement learning (RL) can improve performance. However, applying such reasoning strategies to the visual generation domain remains largely unexplored. In this paper, we present T2I-R1, a novel reasoning-enhanced text-to-image generation model, powered by RL with a bi-level CoT reasoning process. Specifically, we identify two levels of CoT that can be utilized to enhance different stages of generation: (1) the semantic-level CoT for high-level planning of the prompt and (2) the token-level CoT for low-level pixel processing during patch-by-patch generation. To better coordinate these two levels of CoT, we introduce BiCoT-GRPO with an ensemble of generation rewards, which seamlessly optimizes both generation CoTs within the same training step. By applying our reasoning strategies to the baseline model, Janus-Pro, we achieve superior performance with 13% improvement on T2I-CompBench and 19% improvement on the WISE benchmark, even surpassing the state-of-the-art model FLUX.1. Code is available at: https://github.com/CaraJ7/T2I-R1
>
---
#### [replaced 062] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14351v3](http://arxiv.org/pdf/2502.14351v3)**

> **作者:** Yichi Zhang; Le Xue; Wenbo Zhang; Lanlan Li; Yuchen Liu; Chen Jiang; Yuan Cheng; Yuan Qi
>
> **备注:** Accept for ICCV 2025
>
> **摘要:** Positron Emission Tomography (PET) is a powerful molecular imaging tool that plays a crucial role in modern medical diagnostics by visualizing radio-tracer distribution to reveal physiological processes. Accurate organ segmentation from PET images is essential for comprehensive multi-systemic analysis of interactions between different organs and pathologies. Existing segmentation methods are limited by insufficient annotation data and varying levels of annotation, resulting in weak generalization ability and difficulty in clinical application. Recent developments in segmentation foundation models have shown superior versatility across diverse segmentation tasks. Despite the efforts of medical adaptations, these works primarily focus on structural medical images with detailed physiological structural information and exhibit limited generalization performance on molecular PET imaging. In this paper, we collect and construct PETS-5k, the largest PET segmentation dataset to date, comprising 5,731 three-dimensional whole-body PET images and encompassing over 1.3M 2D images. Based on the established dataset, we develop SegAnyPET, a modality-specific 3D foundation model for universal promptable segmentation from PET images. To issue the challenge of discrepant annotation quality, we adopt a cross prompting confident learning (CPCL) strategy with an uncertainty-guided self-rectification process to robustly learn segmentation from high-quality labeled data and low-quality noisy labeled data for promptable segmentation. Experimental results demonstrate that SegAnyPET can segment seen and unseen target organs using only one or a few prompt points, outperforming state-of-the-art foundation models and task-specific fully supervised models with higher accuracy and strong generalization ability for universal segmentation.
>
---
#### [replaced 063] SurgTPGS: Semantic 3D Surgical Scene Understanding with Text Promptable Gaussian Splatting
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23309v2](http://arxiv.org/pdf/2506.23309v2)**

> **作者:** Yiming Huang; Long Bai; Beilei Cui; Kun Yuan; Guankun Wang; Mobarak I. Hoque; Nicolas Padoy; Nassir Navab; Hongliang Ren
>
> **备注:** MICCAI 2025. Project Page: https://lastbasket.github.io/MICCAI-2025-SurgTPGS/
>
> **摘要:** In contemporary surgical research and practice, accurately comprehending 3D surgical scenes with text-promptable capabilities is particularly crucial for surgical planning and real-time intra-operative guidance, where precisely identifying and interacting with surgical tools and anatomical structures is paramount. However, existing works focus on surgical vision-language model (VLM), 3D reconstruction, and segmentation separately, lacking support for real-time text-promptable 3D queries. In this paper, we present SurgTPGS, a novel text-promptable Gaussian Splatting method to fill this gap. We introduce a 3D semantics feature learning strategy incorporating the Segment Anything model and state-of-the-art vision-language models. We extract the segmented language features for 3D surgical scene reconstruction, enabling a more in-depth understanding of the complex surgical environment. We also propose semantic-aware deformation tracking to capture the seamless deformation of semantic features, providing a more precise reconstruction for both texture and semantic features. Furthermore, we present semantic region-aware optimization, which utilizes regional-based semantic information to supervise the training, particularly promoting the reconstruction quality and semantic smoothness. We conduct comprehensive experiments on two real-world surgical datasets to demonstrate the superiority of SurgTPGS over state-of-the-art methods, highlighting its potential to revolutionize surgical practices. SurgTPGS paves the way for developing next-generation intelligent surgical systems by enhancing surgical precision and safety. Our code is available at: https://github.com/lastbasket/SurgTPGS.
>
---
#### [replaced 064] Exploring Text-Guided Single Image Editing for Remote Sensing Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.05769v4](http://arxiv.org/pdf/2405.05769v4)**

> **作者:** Fangzhou Han; Lingyu Si; Zhizhuo Jiang; Hongwei Dong; Lamei Zhang; Yu Liu; Hao Chen; Bo Du
>
> **备注:** 17 pages, 18 figures, Accepted by IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
>
> **摘要:** Artificial intelligence generative content (AIGC) has significantly impacted image generation in the field of remote sensing. However, the equally important area of remote sensing image (RSI) editing has not received sufficient attention. Deep learning based editing methods generally involve two sequential stages: generation and editing. For natural images, these stages primarily rely on generative backbones pre-trained on large-scale benchmark datasets and text guidance facilitated by vision-language models (VLMs). However, it become less viable for RSIs: First, existing generative RSI benchmark datasets do not fully capture the diversity of RSIs, and is often inadequate for universal editing tasks. Second, the single text semantic corresponds to multiple image semantics, leading to the introduction of incorrect semantics. To solve above problems, this paper proposes a text-guided RSI editing method and can be trained using only a single image. A multi-scale training approach is adopted to preserve consistency without the need for training on extensive benchmarks, while leveraging RSI pre-trained VLMs and prompt ensembling (PE) to ensure accuracy and controllability. Experimental results on multiple RSI editing tasks show that the proposed method offers significant advantages in both CLIP scores and subjective evaluations compared to existing methods. Additionally, we explore the ability of the edited RSIs to support disaster assessment tasks in order to validate their practicality. Codes will be released at https://github.com/HIT-PhilipHan/remote_sensing_image_editing.
>
---
#### [replaced 065] Avoid Forgetting by Preserving Global Knowledge Gradients in Federated Learning with Non-IID Data
- **分类: cs.LG; cs.AI; cs.CV; cs.DC; cs.PF**

- **链接: [http://arxiv.org/pdf/2505.20485v3](http://arxiv.org/pdf/2505.20485v3)**

> **作者:** Abhijit Chunduru; Majid Morafah; Mahdi Morafah; Vishnu Pandi Chellapandi; Ang Li
>
> **摘要:** The inevitable presence of data heterogeneity has made federated learning very challenging. There are numerous methods to deal with this issue, such as local regularization, better model fusion techniques, and data sharing. Though effective, they lack a deep understanding of how data heterogeneity can affect the global decision boundary. In this paper, we bridge this gap by performing an experimental analysis of the learned decision boundary using a toy example. Our observations are surprising: (1) we find that the existing methods suffer from forgetting and clients forget the global decision boundary and only learn the perfect local one, and (2) this happens regardless of the initial weights, and clients forget the global decision boundary even starting from pre-trained optimal weights. In this paper, we present FedProj, a federated learning framework that robustly learns the global decision boundary and avoids its forgetting during local training. To achieve better ensemble knowledge fusion, we design a novel server-side ensemble knowledge transfer loss to further calibrate the learned global decision boundary. To alleviate the issue of learned global decision boundary forgetting, we further propose leveraging an episodic memory of average ensemble logits on a public unlabeled dataset to regulate the gradient updates at each step of local training. Experimental results demonstrate that FedProj outperforms state-of-the-art methods by a large margin.
>
---
#### [replaced 066] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.13949v2](http://arxiv.org/pdf/2411.13949v2)**

> **作者:** Ziqi Wang; Chang Che; Qi Wang; Yangyang Li; Zenglin Shi; Meng Wang
>
> **摘要:** Visual instruction tuning (VIT) enables multimodal large language models (MLLMs) to effectively handle a wide range of vision tasks by framing them as language-based instructions. Building on this, continual visual instruction tuning (CVIT) extends the capability of MLLMs to incrementally learn new tasks, accommodating evolving functionalities. While prior work has advanced CVIT through the development of new benchmarks and approaches to mitigate catastrophic forgetting, these efforts largely follow traditional continual learning paradigms, neglecting the unique challenges specific to CVIT. We identify a dual form of catastrophic forgetting in CVIT, where MLLMs not only forget previously learned visual understanding but also experience a decline in instruction following abilities as they acquire new tasks. To address this, we introduce the Separable Mixture of Low-Rank Adaptation (SMoLoRA) framework, which employs separable routing through two distinct modules-one for visual understanding and another for instruction following. This dual-routing design enables specialized adaptation in both domains, preventing forgetting while improving performance. Furthermore, we propose a new CVIT benchmark that goes beyond existing benchmarks by additionally evaluating a model's ability to generalize to unseen tasks and handle diverse instructions across various tasks. Extensive experiments demonstrate that SMoLoRA outperforms existing methods in mitigating dual forgetting, improving generalization to unseen tasks, and ensuring robustness in following diverse instructions. Code is available at https://github.com/Minato-Zackie/SMoLoRA.
>
---
#### [replaced 067] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.03704v3](http://arxiv.org/pdf/2412.03704v3)**

> **作者:** Xiyao Wang; Zhengyuan Yang; Linjie Li; Hongjin Lu; Yuancheng Xu; Chung-Ching Lin; Kevin Lin; Furong Huang; Lijuan Wang
>
> **摘要:** Despite significant advancements in vision-language models (VLMs), there lacks effective approaches to enhance response quality by scaling inference-time computation. This capability is known to be a core step towards the self-improving models in recent large language model studies. In this paper, we present Vision Value Model (VisVM) that can guide VLM inference-time search to generate responses with better visual comprehension. Specifically, VisVM not only evaluates the generated sentence quality in the current search step, but also anticipates the quality of subsequent sentences that may result from the current step, thus providing a long-term value. In this way, VisVM steers VLMs away from generating sentences prone to hallucinations or insufficient detail, thereby producing higher quality responses. Experimental results demonstrate that VisVM-guided search significantly enhances VLMs' ability to generate descriptive captions with richer visual details and fewer hallucinations, compared with greedy decoding and search methods with other visual reward signals. Furthermore, we find that self-training the model with the VisVM-guided captions improve VLM's performance across a wide range of multimodal benchmarks, indicating the potential for developing self-improving VLMs. Our value model and code are available at https://github.com/si0wang/VisVM.
>
---
#### [replaced 068] Identity Preserving 3D Head Stylization with Multiview Score Distillation
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.13536v2](http://arxiv.org/pdf/2411.13536v2)**

> **作者:** Bahri Batuhan Bilecen; Ahmet Berke Gokmen; Furkan Guzelant; Aysegul Dundar
>
> **备注:** https://three-bee.github.io/head_stylization
>
> **摘要:** 3D head stylization transforms realistic facial features into artistic representations, enhancing user engagement across gaming and virtual reality applications. While 3D-aware generators have made significant advancements, many 3D stylization methods primarily provide near-frontal views and struggle to preserve the unique identities of original subjects, often resulting in outputs that lack diversity and individuality. This paper addresses these challenges by leveraging the PanoHead model, synthesizing images from a comprehensive 360-degree perspective. We propose a novel framework that employs negative log-likelihood distillation (LD) to enhance identity preservation and improve stylization quality. By integrating multi-view grid score and mirror gradients within the 3D GAN architecture and introducing a score rank weighing technique, our approach achieves substantial qualitative and quantitative improvements. Our findings not only advance the state of 3D head stylization but also provide valuable insights into effective distillation processes between diffusion models and GANs, focusing on the critical issue of identity preservation. Please visit the https://three-bee.github.io/head_stylization for more visuals.
>
---
