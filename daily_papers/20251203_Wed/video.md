# 计算机视觉 cs.CV

- **最新发布 141 篇**

- **更新 72 篇**

## 最新发布

#### [new 001] Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D场景-语言理解任务，解决传统因果注意力掩码导致的顺序偏差与对象-指令关注受限问题。提出3D-SLIM掩码策略，通过几何自适应与指令感知掩码，使模型基于空间关系进行推理，无需修改架构或增加参数，显著提升多任务性能。**

- **链接: [https://arxiv.org/pdf/2512.02487v1](https://arxiv.org/pdf/2512.02487v1)**

> **作者:** Yerim Jeon; Miso Lee; WonJun Moon; Jae-Pil Heo
>
> **摘要:** Recent advances in 3D scene-language understanding have leveraged Large Language Models (LLMs) for 3D reasoning by transferring their general reasoning ability to 3D multi-modal contexts. However, existing methods typically adopt standard decoders from language modeling, which rely on a causal attention mask. This design introduces two fundamental conflicts in 3D scene understanding: sequential bias among order-agnostic 3D objects and restricted object-instruction attention, hindering task-specific reasoning. To overcome these limitations, we propose 3D Spatial Language Instruction Mask (3D-SLIM), an effective masking strategy that replaces the causal mask with an adaptive attention mask tailored to the spatial structure of 3D scenes. Our 3D-SLIM introduces two key components: a Geometry-adaptive Mask that constrains attention based on spatial density rather than token order, and an Instruction-aware Mask that enables object tokens to directly access instruction context. This design allows the model to process objects based on their spatial relationships while being guided by the user's task. 3D-SLIM is simple, requires no architectural modifications, and adds no extra parameters, yet it yields substantial performance improvements across diverse 3D scene-language tasks. Extensive experiments across multiple benchmarks and LLM baselines validate its effectiveness and underscore the critical role of decoder design in 3D multi-modal reasoning.
>
---
#### [new 002] DynamicVerse: A Physically-Aware Multimodal Framework for 4D World Modeling
- **分类: cs.CV**

- **简介: 该论文提出DynamicVerse，一个用于动态真实世界视频的物理感知多模态4D建模框架。针对现有数据集在尺度、动态性与描述性上的不足，通过融合视觉、几何与多模态模型，实现从互联网单目视频中重建物理尺度的4D结构、运动与语义信息，构建大规模数据集，并在深度、位姿等任务上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.03000v1](https://arxiv.org/pdf/2512.03000v1)**

> **作者:** Kairun Wen; Yuzhi Huang; Runyu Chen; Hui Zheng; Yunlong Lin; Panwang Pan; Chenxin Li; Wenyan Cong; Jian Zhang; Junbin Lu; Chenguo Lin; Dilin Wang; Zhicheng Yan; Hongyu Xu; Justin Theiss; Yue Huang; Xinghao Ding; Rakesh Ranjan; Zhiwen Fan
>
> **摘要:** Understanding the dynamic physical world, characterized by its evolving 3D structure, real-world motion, and semantic content with textual descriptions, is crucial for human-agent interaction and enables embodied agents to perceive and act within real environments with human-like capabilities. However, existing datasets are often derived from limited simulators or utilize traditional Structurefrom-Motion for up-to-scale annotation and offer limited descriptive captioning, which restricts the capacity of foundation models to accurately interpret real-world dynamics from monocular videos, commonly sourced from the internet. To bridge these gaps, we introduce DynamicVerse, a physical-scale, multimodal 4D world modeling framework for dynamic real-world video. We employ large vision, geometric, and multimodal models to interpret metric-scale static geometry, real-world dynamic motion, instance-level masks, and holistic descriptive captions. By integrating window-based Bundle Adjustment with global optimization, our method converts long real-world video sequences into a comprehensive 4D multimodal format. DynamicVerse delivers a large-scale dataset consists of 100K+ videos with 800K+ annotated masks and 10M+ frames from internet videos. Experimental evaluations on three benchmark tasks, namely video depth estimation, camera pose estimation, and camera intrinsics estimation, demonstrate that our 4D modeling achieves superior performance in capturing physical-scale measurements with greater global accuracy than existing methods.
>
---
#### [new 003] ClusterStyle: Modeling Intra-Style Diversity with Prototypical Clustering for Stylized Motion Generation
- **分类: cs.CV**

- **简介: 该论文针对风格化动作生成中难以捕捉同一风格内多样性的问题，提出ClusterStyle框架。通过原型聚类构建全局与局部结构化风格嵌入空间，增强风格表征能力，并引入SMA模块融合风格特征。实验表明其在风格化动作生成与迁移上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.02453v1](https://arxiv.org/pdf/2512.02453v1)**

> **作者:** Kerui Chen; Jianrong Zhang; Ming Li; Zhonglong Zheng; Hehe Fan
>
> **摘要:** Existing stylized motion generation models have shown their remarkable ability to understand specific style information from the style motion, and insert it into the content motion. However, capturing intra-style diversity, where a single style should correspond to diverse motion variations, remains a significant challenge. In this paper, we propose a clustering-based framework, ClusterStyle, to address this limitation. Instead of learning an unstructured embedding from each style motion, we leverage a set of prototypes to effectively model diverse style patterns across motions belonging to the same style category. We consider two types of style diversity: global-level diversity among style motions of the same category, and local-level diversity within the temporal dynamics of motion sequences. These components jointly shape two structured style embedding spaces, i.e., global and local, optimized via alignment with non-learnable prototype anchors. Furthermore, we augment the pretrained text-to-motion generation model with the Stylistic Modulation Adapter (SMA) to integrate the style features. Extensive experiments demonstrate that our approach outperforms existing state-of-the-art models in stylized motion generation and motion style transfer.
>
---
#### [new 004] HUD: Hierarchical Uncertainty-Aware Disambiguation Network for Composed Video Retrieval
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对多模态视频检索任务（CVR），解决查询中视频与文本信息密度差异导致的指代歧义和语义聚焦不足问题。提出HUD框架，通过整体与细粒度跨模态交互，实现指代消解与细节语义增强，提升组合特征学习精度，并可迁移至图像检索任务。**

- **链接: [https://arxiv.org/pdf/2512.02792v1](https://arxiv.org/pdf/2512.02792v1)**

> **作者:** Zhiwei Chen; Yupeng Hu; Zixu Li; Zhiheng Fu; Haokun Wen; Weili Guan
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Composed Video Retrieval (CVR) is a challenging video retrieval task that utilizes multi-modal queries, consisting of a reference video and modification text, to retrieve the desired target video. The core of this task lies in understanding the multi-modal composed query and achieving accurate composed feature learning. Within multi-modal queries, the video modality typically carries richer semantic content compared to the textual modality. However, previous works have largely overlooked the disparity in information density between these two modalities. This limitation can lead to two critical issues: 1) modification subject referring ambiguity and 2) limited detailed semantic focus, both of which degrade the performance of CVR models. To address the aforementioned issues, we propose a novel CVR framework, namely the Hierarchical Uncertainty-aware Disambiguation network (HUD). HUD is the first framework that leverages the disparity in information density between video and text to enhance multi-modal query understanding. It comprises three key components: (a) Holistic Pronoun Disambiguation, (b) Atomistic Uncertainty Modeling, and (c) Holistic-to-Atomistic Alignment. By exploiting overlapping semantics through holistic cross-modal interaction and fine-grained semantic alignment via atomistic-level cross-modal interaction, HUD enables effective object disambiguation and enhances the focus on detailed semantics, thereby achieving precise composed feature learning. Moreover, our proposed HUD is also applicable to the Composed Image Retrieval (CIR) task and achieves state-of-the-art performance across three benchmark datasets for both CVR and CIR tasks. The codes are available on https://zivchen-ty.github.io/HUD.github.io/.
>
---
#### [new 005] GeoViS: Geospatially Rewarded Visual Search for Remote Sensing Visual Grounding
- **分类: cs.CV**

- **简介: 该论文提出GeoViS，针对遥感图像中目标小、空间关系复杂的问题，将视觉定位任务转化为渐进式搜索与推理过程。通过融合多模态感知、空间推理与奖励驱动探索，实现精准的地理空间定位，显著提升遥感视觉定位性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.02715v1](https://arxiv.org/pdf/2512.02715v1)**

> **作者:** Peirong Zhang; Yidan Zhang; Luxiao Xu; Jinliang Lin; Zonghao Guo; Fengxiang Wang; Xue Yang; Kaiwen Wei; Lei Wang
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Recent advances in multimodal large language models(MLLMs) have led to remarkable progress in visual grounding, enabling fine-grained cross-modal alignment between textual queries and image regions. However, transferring such capabilities to remote sensing imagery remains challenging, as targets are often extremely small within kilometer-scale scenes, and queries typically involve intricate geospatial relations such as relative positions, spatial hierarchies, or contextual dependencies across distant objects. To address these challenges, we propose GeoViS, a Geospatially Rewarded Visual Search framework that reformulates remote sensing visual grounding as a progressive search-and-reasoning process. Rather than directly predicting the target location in a single step, GeoViS actively explores the global image through a tree-structured sequence of visual cues, integrating multimodal perception, spatial reasoning, and reward-guided exploration to refine geospatial hypotheses iteratively. This design enables the model to detect subtle small-scale targets while maintaining holistic scene awareness. Extensive experiments on five remote sensing grounding benchmarks demonstrate that GeoViS achieves precise geospatial understanding and consistently surpasses existing methods across key visual grounding metrics, highlighting its strong cross-domain generalization and interpretability.
>
---
#### [new 006] Attention-guided reference point shifting for Gaussian-mixture-based partial point set registration
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对深度学习与高斯混合模型（GMM）结合的点云配准任务，解决部分点集间配准中特征向量对平移旋转不敏感的问题。提出注意力引导的参考点偏移（ARPS）层，通过识别两部分点集的共同参考点，获得变换不变特征，显著提升DeepGMR及UGMMReg性能，优于现有基于注意力机制的方法。**

- **链接: [https://arxiv.org/pdf/2512.02496v1](https://arxiv.org/pdf/2512.02496v1)**

> **作者:** Mizuki Kikkawa; Tatsuya Yatagawa; Yutaka Ohtake; Hiromasa Suzuki
>
> **备注:** 16 pages, 9 figures, 7 tables
>
> **摘要:** This study investigates the impact of the invariance of feature vectors for partial-to-partial point set registration under translation and rotation of input point sets, particularly in the realm of techniques based on deep learning and Gaussian mixture models (GMMs). We reveal both theoretical and practical problems associated with such deep-learning-based registration methods using GMMs, with a particular focus on the limitations of DeepGMR, a pioneering study in this line, to the partial-to-partial point set registration. Our primary goal is to uncover the causes behind such methods and propose a comprehensible solution for that. To address this, we introduce an attention-based reference point shifting (ARPS) layer, which robustly identifies a common reference point of two partial point sets, thereby acquiring transformation-invariant features. The ARPS layer employs a well-studied attention module to find a common reference point rather than the overlap region. Owing to this, it significantly enhances the performance of DeepGMR and its recent variant, UGMMReg. Furthermore, these extension models outperform even prior deep learning methods using attention blocks and Transformer to extract the overlap region or common reference points. We believe these findings provide deeper insights into registration methods using deep learning and GMMs.
>
---
#### [new 007] MICCAI STSR 2025 Challenge: Semi-Supervised Teeth and Pulp Segmentation and CBCT-IOS Registration
- **分类: cs.CV**

- **简介: 该论文聚焦于数字牙科中的半监督学习任务，解决CBCT与IOS数据标注稀缺问题。组织MICCAI 2025 STSR挑战赛，开展牙齿及牙髓分割与跨模态配准。通过提供多模态数据集，推动基于伪标签与一致性正则的深度学习方法发展，实现高精度分割与配准，代码数据公开可复现。**

- **链接: [https://arxiv.org/pdf/2512.02867v1](https://arxiv.org/pdf/2512.02867v1)**

> **作者:** Yaqi Wang; Zhi Li; Chengyu Wu; Jun Liu; Yifan Zhang; Jialuo Chen; Jiaxue Ni; Qian Luo; Jin Liu; Can Han; Changkai Ji; Zhi Qin Tan; Ajo Babu George; Liangyu Chen; Qianni Zhang; Dahong Qian; Shuai Wang; Huiyu Zhou
>
> **摘要:** Cone-Beam Computed Tomography (CBCT) and Intraoral Scanning (IOS) are essential for digital dentistry, but annotated data scarcity limits automated solutions for pulp canal segmentation and cross-modal registration. To benchmark semi-supervised learning (SSL) in this domain, we organized the STSR 2025 Challenge at MICCAI 2025, featuring two tasks: (1) semi-supervised segmentation of teeth and pulp canals in CBCT, and (2) semi-supervised rigid registration of CBCT and IOS. We provided 60 labeled and 640 unlabeled IOS samples, plus 30 labeled and 250 unlabeled CBCT scans with varying resolutions and fields of view. The challenge attracted strong community participation, with top teams submitting open-source deep learning-based SSL solutions. For segmentation, leading methods used nnU-Net and Mamba-like State Space Models with pseudo-labeling and consistency regularization, achieving a Dice score of 0.967 and Instance Affinity of 0.738 on the hidden test set. For registration, effective approaches combined PointNetLK with differentiable SVD and geometric augmentation to handle modality gaps; hybrid neural-classical refinement enabled accurate alignment despite limited labels. All data and code are publicly available at https://github.com/ricoleehduu/STS-Challenge-2025 to ensure reproducibility.
>
---
#### [new 008] Spatially-Grounded Document Retrieval via Patch-to-Region Relevance Propagation
- **分类: cs.CV; cs.IR**

- **简介: 该论文针对文档检索中精确区域定位问题，提出一种融合视觉语言模型与OCR的混合方法。通过将Patch级相似度作为空间相关性过滤器，实现细粒度区域检索，提升RAG任务中的上下文精准度。无需训练，仅在推理时应用，已开源实现。**

- **链接: [https://arxiv.org/pdf/2512.02660v1](https://arxiv.org/pdf/2512.02660v1)**

> **作者:** Agathoklis Georgiou
>
> **备注:** 13 pages, 1 figure, 2 tables. Open-source implementation available at https://github.com/athrael-soju/Snappy
>
> **摘要:** Vision-language models (VLMs) like ColPali achieve state-of-the-art document retrieval by embedding pages as images and computing fine-grained similarity between query tokens and visual patches. However, they return entire pages rather than specific regions, limiting utility for retrieval-augmented generation (RAG) where precise context is paramount. Conversely, OCR-based systems extract structured text with bounding box coordinates but lack semantic grounding for relevance assessment. We propose a hybrid architecture that unifies these paradigms: using ColPali's patch-level similarity scores as spatial relevance filters over OCR-extracted regions. We formalize the coordinate mapping between vision transformer patch grids and OCR bounding boxes, introduce intersection metrics for relevance propagation, and establish theoretical bounds on retrieval precision. Our approach operates at inference time without additional training. We release Snappy, an open-source implementation demonstrating practical applicability, with empirical evaluation ongoing.
>
---
#### [new 009] A Lightweight Real-Time Low-Light Enhancement Network for Embedded Automotive Vision Systems
- **分类: cs.CV**

- **简介: 该论文针对嵌入式车载视觉系统在低光照下的图像增强任务，提出轻量级实时网络UltraFast-LieNET。通过动态移位卷积与多尺度残差块设计，实现高效特征提取与大感受野，结合梯度感知损失提升稳定性，在仅180参数下显著优于现有方法，兼顾性能与实时性。**

- **链接: [https://arxiv.org/pdf/2512.02965v1](https://arxiv.org/pdf/2512.02965v1)**

> **作者:** Yuhan Chen; Yicui Shi; Guofa Li; Guangrui Bai; Jinyuan Shao; Xiangfei Huang; Wenbo Chu; Keqiang Li
>
> **摘要:** In low-light environments like nighttime driving, image degradation severely challenges in-vehicle camera safety. Since existing enhancement algorithms are often too computationally intensive for vehicular applications, we propose UltraFast-LieNET, a lightweight multi-scale shifted convolutional network for real-time low-light image enhancement. We introduce a Dynamic Shifted Convolution (DSConv) kernel with only 12 learnable parameters for efficient feature extraction. By integrating DSConv with varying shift distances, a Multi-scale Shifted Residual Block (MSRB) is constructed to significantly expand the receptive field. To mitigate lightweight network instability, a residual structure and a novel multi-level gradient-aware loss function are incorporated. UltraFast-LieNET allows flexible parameter configuration, with a minimum size of only 36 parameters. Results on the LOLI-Street dataset show a PSNR of 26.51 dB, outperforming state-of-the-art methods by 4.6 dB while utilizing only 180 parameters. Experiments across four benchmark datasets validate its superior balance of real-time performance and enhancement quality under limited resources. Code is available at https://githubhttps://github.com/YuhanChen2024/UltraFast-LiNET
>
---
#### [new 010] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文针对长视频问答任务，解决模型因上下文容量有限导致的细节丢失与多尺度事件理解困难问题。提出WorldMM，一种动态多模态记忆代理，融合文本、语义与视觉记忆，通过自适应检索实现多粒度信息获取，显著提升长视频推理性能。**

- **链接: [https://arxiv.org/pdf/2512.02425v1](https://arxiv.org/pdf/2512.02425v1)**

> **作者:** Woongyeong Yeo; Kangsan Kim; Jaehong Yoon; Sung Ju Hwang
>
> **备注:** Project page : https://worldmm.github.io
>
> **摘要:** Recent advances in video large language models have demonstrated strong capabilities in understanding short clips. However, scaling them to hours- or days-long videos remains highly challenging due to limited context capacity and the loss of critical visual details during abstraction. Existing memory-augmented methods mitigate this by leveraging textual summaries of video segments, yet they heavily rely on text and fail to utilize visual evidence when reasoning over complex scenes. Moreover, retrieving from fixed temporal scales further limits their flexibility in capturing events that span variable durations. To address this, we introduce WorldMM, a novel multimodal memory agent that constructs and retrieves from multiple complementary memories, encompassing both textual and visual representations. WorldMM comprises three types of memory: episodic memory indexes factual events across multiple temporal scales, semantic memory continuously updates high-level conceptual knowledge, and visual memory preserves detailed information about scenes. During inference, an adaptive retrieval agent iteratively selects the most relevant memory source and leverages multiple temporal granularities based on the query, continuing until it determines that sufficient information has been gathered. WorldMM significantly outperforms existing baselines across five long video question-answering benchmarks, achieving an average 8.4% performance gain over previous state-of-the-art methods, showing its effectiveness on long video reasoning.
>
---
#### [new 011] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对3D视觉基础模型在在线重建中时序不一致性问题，提出TALO框架。通过薄板样条实现高自由度长期对齐，并采用点无关子图注册增强鲁棒性，有效提升几何一致性与轨迹精度，兼容多种模型与相机配置。**

- **链接: [https://arxiv.org/pdf/2512.02341v1](https://arxiv.org/pdf/2512.02341v1)**

> **作者:** Fengyi Zhang; Tianjun Zhang; Kasra Khosoussi; Zheng Zhang; Zi Huang; Yadan Luo
>
> **摘要:** 3D vision foundation models have shown strong generalization in reconstructing key 3D attributes from uncalibrated images through a single feed-forward pass. However, when deployed in online settings such as driving scenarios, predictions are made over temporal windows, making it non-trivial to maintain consistency across time. Recent strategies align consecutive predictions by solving global transformation, yet our analysis reveals their fundamental limitations in assumption validity, local alignment scope, and robustness under noisy geometry. In this work, we propose a higher-DOF and long-term alignment framework based on Thin Plate Spline, leveraging globally propagated control points to correct spatially varying inconsistencies. In addition, we adopt a point-agnostic submap registration design that is inherently robust to noisy geometry predictions. The proposed framework is fully plug-and-play, compatible with diverse 3D foundation models and camera configurations (e.g., monocular or surround-view). Extensive experiments demonstrate that our method consistently yields more coherent geometry and lower trajectory errors across multiple datasets, backbone models, and camera setups, highlighting its robustness and generality. Codes are publicly available at \href{https://github.com/Xian-Bei/TALO}{https://github.com/Xian-Bei/TALO}.
>
---
#### [new 012] Tissue-mask supported inter-subject whole-body image registration in the UK Biobank - A method benchmarking study
- **分类: cs.CV**

- **简介: 该论文针对UK Biobank全身体积MRI图像的跨被试配准任务，提出一种基于皮下脂肪与肌肉掩膜增强的强度-图割配准方法。通过引入71个组织掩膜，提升配准精度，显著改善了重叠度与年龄相关性分析的准确性，解决了大规模影像数据空间标准化难题。**

- **链接: [https://arxiv.org/pdf/2512.02702v1](https://arxiv.org/pdf/2512.02702v1)**

> **作者:** Yasemin Utkueri; Elin Lundström; Håkan Ahlström; Johan Öfverstedt; Joel Kullberg
>
> **摘要:** The UK Biobank is a large-scale study collecting whole-body MR imaging and non-imaging health data. Robust and accurate inter-subject image registration of these whole-body MR images would enable their body-wide spatial standardization, and region-/voxel-wise correlation analysis of non-imaging data with image-derived parameters (e.g., tissue volume or fat content). We propose a sex-stratified inter-subject whole-body MR image registration approach that uses subcutaneous adipose tissue- and muscle-masks from the state-of-the-art VIBESegmentator method to augment intensity-based graph-cut registration. The proposed method was evaluated on a subset of 4000 subjects by comparing it to an intensity-only method as well as two previously published registration methods, uniGradICON and MIRTK. The evaluation comprised overlap measures applied to the 71 VIBESegmentator masks: 1) Dice scores, and 2) voxel-wise label error frequency. Additionally, voxel-wise correlation between age and each of fat content and tissue volume was studied to exemplify the usefulness for medical research. The proposed method exhibited a mean dice score of 0.77 / 0.75 across the cohort and the 71 masks for males/females, respectively. When compared to the intensity-only registration, the mean values were 6 percentage points (pp) higher for both sexes, and the label error frequency was decreased in most tissue regions. These differences were 9pp / 8pp against uniGradICON and 12pp / 13pp against MIRTK. Using the proposed method, the age-correlation maps were less noisy and showed higher anatomical alignment. In conclusion, the image registration method using two tissue masks improves whole-body registration of UK Biobank images.
>
---
#### [new 013] Exploring the Potentials of Spiking Neural Networks for Image Deraining
- **分类: cs.CV**

- **简介: 该论文研究图像去雨任务，针对传统脉冲神经网络（SNN）在低层视觉中缺乏空间上下文理解及频域饱和问题，提出视觉LIF（VLIF）神经元与相关模块，实现高效多尺度特征学习。实验表明，方法性能优于现有SNN方法，能耗仅为13%。**

- **链接: [https://arxiv.org/pdf/2512.02258v1](https://arxiv.org/pdf/2512.02258v1)**

> **作者:** Shuang Chen; Tomas Krajnik; Farshad Arvin; Amir Atapour-Abarghouei
>
> **备注:** Accepted By AAAI2026
>
> **摘要:** Biologically plausible and energy-efficient frameworks such as Spiking Neural Networks (SNNs) have not been sufficiently explored in low-level vision tasks. Taking image deraining as an example, this study addresses the representation of the inherent high-pass characteristics of spiking neurons, specifically in image deraining and innovatively proposes the Visual LIF (VLIF) neuron, overcoming the obstacle of lacking spatial contextual understanding present in traditional spiking neurons. To tackle the limitation of frequency-domain saturation inherent in conventional spiking neurons, we leverage the proposed VLIF to introduce the Spiking Decomposition and Enhancement Module and the lightweight Spiking Multi-scale Unit for hierarchical multi-scale representation learning. Extensive experiments across five benchmark deraining datasets demonstrate that our approach significantly outperforms state-of-the-art SNN-based deraining methods, achieving this superior performance with only 13\% of their energy consumption. These findings establish a solid foundation for deploying SNNs in high-performance, energy-efficient low-level vision tasks.
>
---
#### [new 014] DF-Mamba: Deformable State Space Modeling for 3D Hand Pose Estimation in Interactions
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对3D手部姿态估计中因遮挡导致的特征学习困难问题，提出基于可变形状态空间建模的DF-Mamba框架。通过Mamba的选通状态建模与可变形扫描机制，有效捕捉局部特征与全局上下文关系，提升遮挡下姿态估计精度，优于现有方法且推理速度接近ResNet-50。**

- **链接: [https://arxiv.org/pdf/2512.02727v1](https://arxiv.org/pdf/2512.02727v1)**

> **作者:** Yifan Zhou; Takehiko Ohkawa; Guwenxiao Zhou; Kanoko Goto; Takumi Hirose; Yusuke Sekikawa; Nakamasa Inoue
>
> **备注:** Accepted to WACV 2026. Project page: https://tkhkaeio.github.io/projects/25-dfmamba/index.html
>
> **摘要:** Modeling daily hand interactions often struggles with severe occlusions, such as when two hands overlap, which highlights the need for robust feature learning in 3D hand pose estimation (HPE). To handle such occluded hand images, it is vital to effectively learn the relationship between local image features (e.g., for occluded joints) and global context (e.g., cues from inter-joints, inter-hands, or the scene). However, most current 3D HPE methods still rely on ResNet for feature extraction, and such CNN's inductive bias may not be optimal for 3D HPE due to its limited capability to model the global context. To address this limitation, we propose an effective and efficient framework for visual feature extraction in 3D HPE using recent state space modeling (i.e., Mamba), dubbed Deformable Mamba (DF-Mamba). DF-Mamba is designed to capture global context cues beyond standard convolution through Mamba's selective state modeling and the proposed deformable state scanning. Specifically, for local features after convolution, our deformable scanning aggregates these features within an image while selectively preserving useful cues that represent the global context. This approach significantly improves the accuracy of structured 3D HPE, with comparable inference speed to ResNet-50. Our experiments involve extensive evaluations on five divergent datasets including single-hand and two-hand scenarios, hand-only and hand-object interactions, as well as RGB and depth-based estimation. DF-Mamba outperforms the latest image backbones, including VMamba and Spatial-Mamba, on all datasets and achieves state-of-the-art performance.
>
---
#### [new 015] Multi-Domain Enhanced Map-Free Trajectory Prediction with Selective Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶中的轨迹预测任务，解决复杂交互场景下冗余信息干扰与计算效率低的问题。提出一种无地图的多域增强方法，通过MoE选择关键频率成分，结合多尺度时序特征与可选注意力模块，有效过滤冗余信息，并设计多模态解码器提升预测精度。**

- **链接: [https://arxiv.org/pdf/2512.02368v1](https://arxiv.org/pdf/2512.02368v1)**

> **作者:** Wenyi Xiong; Jian Chen
>
> **摘要:** Trajectory prediction is crucial for the reliability and safety of autonomous driving systems, yet it remains a challenging task in complex interactive scenarios. Existing methods often struggle to efficiently extract valuable scene information from redundant data, thereby reducing computational efficiency and prediction accuracy, especially when dealing with intricate agent interactions. To address these challenges, we propose a novel map-free trajectory prediction algorithm that achieves trajectory prediction across the temporal, spatial, and frequency domains. Specifically, in temporal information processing, We utilize a Mixture of Experts (MoE) mechanism to adaptively select critical frequency components. Concurrently, we extract these components and integrate multi-scale temporal features. Subsequently, a selective attention module is proposed to filter out redundant information in both temporal sequences and spatial interactions. Finally, we design a multimodal decoder. Under the supervision of patch-level and point-level losses, we obtain reasonable trajectory results. Experiments on Nuscences datasets demonstrate the superiority of our algorithm, validating its effectiveness in handling complex interactive scenarios.
>
---
#### [new 016] DiverseAR: Boosting Diversity in Bitwise Autoregressive Image Generation
- **分类: cs.CV**

- **简介: 该论文针对比特级自回归图像生成中样本多样性不足的问题，指出二分类限制与过尖锐的输出分布是主因。提出DiverseAR方法，通过自适应调整逻辑值分布平滑度提升多样性，并设计基于能量的路径搜索以保持图像质量。**

- **链接: [https://arxiv.org/pdf/2512.02931v1](https://arxiv.org/pdf/2512.02931v1)**

> **作者:** Ying Yang; Zhengyao Lv; Tianlin Pan; Haofan Wang; Binxin Yang; Hubery Yin; Chen Li; Chenyang Si
>
> **备注:** 23 pages
>
> **摘要:** In this paper, we investigate the underexplored challenge of sample diversity in autoregressive (AR) generative models with bitwise visual tokenizers. We first analyze the factors that limit diversity in bitwise AR models and identify two key issues: (1) the binary classification nature of bitwise modeling, which restricts the prediction space, and (2) the overly sharp logits distribution, which causes sampling collapse and reduces diversity. Building on these insights, we propose DiverseAR, a principled and effective method that enhances image diversity without sacrificing visual quality. Specifically, we introduce an adaptive logits distribution scaling mechanism that dynamically adjusts the sharpness of the binary output distribution during sampling, resulting in smoother predictions and greater diversity. To mitigate potential fidelity loss caused by distribution smoothing, we further develop an energy-based generation path search algorithm that avoids sampling low-confidence tokens, thereby preserving high visual quality. Extensive experiments demonstrate that DiverseAR substantially improves sample diversity in bitwise autoregressive image generation.
>
---
#### [new 017] G-SHARP: Gaussian Surgical Hardware Accelerated Real-time Pipeline
- **分类: cs.CV**

- **简介: 该论文提出G-SHARP，一种基于GSplat的实时微创手术场景重建框架，解决现有方法依赖非商业工具、部署受限的问题。通过原生集成可微高斯光栅化器，实现高保真、抗遮挡的变形组织建模，支持端到端部署于NVIDIA边缘硬件，适用于术中实时可视化。**

- **链接: [https://arxiv.org/pdf/2512.02482v1](https://arxiv.org/pdf/2512.02482v1)**

> **作者:** Vishwesh Nath; Javier G. Tejero; Ruilong Li; Filippo Filicori; Mahdi Azizian; Sean D. Huver
>
> **摘要:** We propose G-SHARP, a commercially compatible, real-time surgical scene reconstruction framework designed for minimally invasive procedures that require fast and accurate 3D modeling of deformable tissue. While recent Gaussian splatting approaches have advanced real-time endoscopic reconstruction, existing implementations often depend on non-commercial derivatives, limiting deployability. G-SHARP overcomes these constraints by being the first surgical pipeline built natively on the GSplat (Apache-2.0) differentiable Gaussian rasterizer, enabling principled deformation modeling, robust occlusion handling, and high-fidelity reconstructions on the EndoNeRF pulling benchmark. Our results demonstrate state-of-the-art reconstruction quality with strong speed-accuracy trade-offs suitable for intra-operative use. Finally, we provide a Holoscan SDK application that deploys G-SHARP on NVIDIA IGX Orin and Thor edge hardware, enabling real-time surgical visualization in practical operating-room settings.
>
---
#### [new 018] Understanding and Harnessing Sparsity in Unified Multimodal Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究统一多模态模型中的稀疏性问题，旨在提升推理效率。针对理解与生成组件在压缩下的不同敏感性，提出基于专家混合（MoE）的稀疏激活机制，使模型仅激活部分参数即可保持性能，显著提升效率。**

- **链接: [https://arxiv.org/pdf/2512.02351v1](https://arxiv.org/pdf/2512.02351v1)**

> **作者:** Shwai He; Chaorui Deng; Ang Li; Shen Yan
>
> **备注:** 13 pages, 13 figures, 8 tables
>
> **摘要:** Large multimodal models have achieved remarkable progress in both understanding and generation. Recent efforts pursue unified multimodal models that integrate heterogeneous components to support both capabilities within a single framework. However, such unification introduces inference inefficiencies, e.g., specific tasks or samples may not require the full knowledge or capacity of the unified model. Yet, a systematic understanding of how these inefficiencies manifest across different components remains limited. In this work, we first conduct a systematic analysis of unified multimodal model components using training-free pruning as a probing methodology, considering both depth pruning and width reduction. Our study reveals that the understanding component exhibits notable compressibility in both understanding and generation tasks, which is more pronounced in the latter. In contrast, the generation components are highly sensitive to compression, with performance deteriorating sharply even under moderate compression ratios. To address this limitation, we propose the Mixture-of-Experts (MoE) Adaptation, inspired by the dynamic activation patterns observed across different samples. This approach partitions the generation module into multiple experts and enables sparse activation to restore generation quality. We validate the effectiveness of sparse activation through expert-frozen tuning and further demonstrate that a fully trainable adaptation delivers additional gains. As a result, the adapted BAGEL model achieves performance comparable to the full model while activating only about half of its parameters. The code is released at \href{https://github.com/Shwai-He/SparseUnifiedModel}{this link}.
>
---
#### [new 019] Taming Camera-Controlled Video Generation with Verifiable Geometry Reward
- **分类: cs.CV**

- **简介: 该论文针对相机控制视频生成任务，解决现有方法依赖监督微调、缺乏在线强化学习优化的问题。提出基于可验证几何奖励的在线强化学习框架，通过对比生成与参考视频的分段相机位姿，提供密集反馈，提升相机控制精度与几何一致性。**

- **链接: [https://arxiv.org/pdf/2512.02870v1](https://arxiv.org/pdf/2512.02870v1)**

> **作者:** Zhaoqing Wang; Xiaobo Xia; Zhuolin Bie; Jinlin Liu; Dongdong Yu; Jia-Wang Bian; Changhu Wang
>
> **备注:** 11 pages, 4 figures, 7 tables
>
> **摘要:** Recent advances in video diffusion models have remarkably improved camera-controlled video generation, but most methods rely solely on supervised fine-tuning (SFT), leaving online reinforcement learning (RL) post-training largely underexplored. In this work, we introduce an online RL post-training framework that optimizes a pretrained video generator for precise camera control. To make RL effective in this setting, we design a verifiable geometry reward that delivers dense segment-level feedback to guide model optimization. Specifically, we estimate the 3D camera trajectories for both generated and reference videos, divide each trajectory into short segments, and compute segment-wise relative poses. The reward function then compares each generated-reference segment pair and assigns an alignment score as the reward signal, which helps alleviate reward sparsity and improve optimization efficiency. Moreover, we construct a comprehensive dataset featuring diverse large-amplitude camera motions and scenes with varied subject dynamics. Extensive experiments show that our online RL post-training clearly outperforms SFT baselines across multiple aspects, including camera-control accuracy, geometric consistency, and visual quality, demonstrating its superiority in advancing camera-controlled video generation.
>
---
#### [new 020] Layout Anything: One Transformer for Universal Room Layout Estimation
- **分类: cs.CV**

- **简介: 该论文提出Layout Anything，一种基于Transformer的通用室内布局估计框架。针对传统方法依赖复杂后处理、效率低的问题，通过任务条件查询与对比学习，结合拓扑感知数据增强和可微几何损失，实现端到端高精度布局预测，显著提升推理速度与精度，适用于AR与3D重建。**

- **链接: [https://arxiv.org/pdf/2512.02952v1](https://arxiv.org/pdf/2512.02952v1)**

> **作者:** Md Sohag Mia; Muhammad Abdullah Adnan
>
> **备注:** Published at WACV 2026
>
> **摘要:** We present Layout Anything, a transformer-based framework for indoor layout estimation that adapts the OneFormer's universal segmentation architecture to geometric structure prediction. Our approach integrates OneFormer's task-conditioned queries and contrastive learning with two key modules: (1) a layout degeneration strategy that augments training data while preserving Manhattan-world constraints through topology-aware transformations, and (2) differentiable geometric losses that directly enforce planar consistency and sharp boundary predictions during training. By unifying these components in an end-to-end framework, the model eliminates complex post-processing pipelines while achieving high-speed inference at 114ms. Extensive experiments demonstrate state-of-the-art performance across standard benchmarks, with pixel error (PE) of 5.43% and corner error (CE) of 4.02% on the LSUN, PE of 7.04% (CE 5.17%) on the Hedau and PE of 4.03% (CE 3.15%) on the Matterport3D-Layout datasets. The framework's combination of geometric awareness and computational efficiency makes it particularly suitable for augmented reality applications and large-scale 3D scene reconstruction tasks.
>
---
#### [new 021] Temporal Dynamics Enhancer for Directly Trained Spiking Object Detectors
- **分类: cs.CV**

- **简介: 该论文针对脉冲神经网络（SNN）在目标检测中因时间信息建模弱导致性能受限的问题，提出时空动态增强器（TDE）。通过设计脉冲编码器与注意力门控模块，提升输入多样性与时序依赖建模能力，并引入脉冲驱动注意力机制降低能耗。实验表明，TDE显著提升检测精度并降低能量消耗。**

- **链接: [https://arxiv.org/pdf/2512.02447v1](https://arxiv.org/pdf/2512.02447v1)**

> **作者:** Fan Luo; Zeyu Gao; Xinhao Luo; Kai Zhao; Yanfeng Lu
>
> **摘要:** Spiking Neural Networks (SNNs), with their brain-inspired spatiotemporal dynamics and spike-driven computation, have emerged as promising energy-efficient alternatives to Artificial Neural Networks (ANNs). However, existing SNNs typically replicate inputs directly or aggregate them into frames at fixed intervals. Such strategies lead to neurons receiving nearly identical stimuli across time steps, severely limiting the model's expressive power, particularly in complex tasks like object detection. In this work, we propose the Temporal Dynamics Enhancer (TDE) to strengthen SNNs' capacity for temporal information modeling. TDE consists of two modules: a Spiking Encoder (SE) that generates diverse input stimuli across time steps, and an Attention Gating Module (AGM) that guides the SE generation based on inter-temporal dependencies. Moreover, to eliminate the high-energy multiplication operations introduced by the AGM, we propose a Spike-Driven Attention (SDA) to reduce attention-related energy consumption. Extensive experiments demonstrate that TDE can be seamlessly integrated into existing SNN-based detectors and consistently outperforms state-of-the-art methods, achieving mAP50-95 scores of 57.7% on the static PASCAL VOC dataset and 47.6% on the neuromorphic EvDET200K dataset. In terms of energy consumption, the SDA consumes only 0.240 times the energy of conventional attention modules.
>
---
#### [new 022] Nav-$R^2$ Dual-Relation Reasoning for Generalizable Open-Vocabulary Object-Goal Navigation
- **分类: cs.CV**

- **简介: 该论文针对开放词汇物体导航任务，解决未知物体定位成功率低与决策过程不透明问题。提出Nav-R²框架，通过双关系推理与相似性感知记忆，实现高效、可解释的导航，显著提升对未见物体的定位能力，支持实时推理。**

- **链接: [https://arxiv.org/pdf/2512.02400v1](https://arxiv.org/pdf/2512.02400v1)**

> **作者:** Wentao Xiang; Haokang Zhang; Tianhang Yang; Zedong Chu; Ruihang Chu; Shichao Xie; Yujian Yuan; Jian Sun; Zhining Gu; Junjie Wang; Xiaolong Wu; Mu Xu; Yujiu Yang
>
> **摘要:** Object-goal navigation in open-vocabulary settings requires agents to locate novel objects in unseen environments, yet existing approaches suffer from opaque decision-making processes and low success rate on locating unseen objects. To address these challenges, we propose Nav-$R^2$, a framework that explicitly models two critical types of relationships, target-environment modeling and environment-action planning, through structured Chain-of-Thought (CoT) reasoning coupled with a Similarity-Aware Memory. We construct a Nav$R^2$-CoT dataset that teaches the model to perceive the environment, focus on target-related objects in the surrounding context and finally make future action plans. Our SA-Mem preserves the most target-relevant and current observation-relevant features from both temporal and semantic perspectives by compressing video frames and fusing historical observations, while introducing no additional parameters. Compared to previous methods, Nav-R^2 achieves state-of-the-art performance in localizing unseen objects through a streamlined and efficient pipeline, avoiding overfitting to seen object categories while maintaining real-time inference at 2Hz. Resources will be made publicly available at \href{https://github.com/AMAP-EAI/Nav-R2}{github link}.
>
---
#### [new 023] TEXTRIX: Latent Attribute Grid for Native Texture Generation and Beyond
- **分类: cs.CV**

- **简介: 该论文针对3D纹理生成中多视角融合导致的不一致与覆盖不全问题，提出TEXTRIX框架。通过构建潜在3D属性网格与稀疏注意力扩散变换器，实现体素空间内直接着色，提升纹理保真度与完整性，并自然扩展至高精度3D部件分割任务，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.02993v1](https://arxiv.org/pdf/2512.02993v1)**

> **作者:** Yifei Zeng; Yajie Bao; Jiachen Qian; Shuang Wu; Youtian Lin; Hao Zhu; Buyu Li; Feihu Zhang; Xun Cao; Yao Yao
>
> **备注:** Project Page: https://www.neural4d.com/research-page/textrix
>
> **摘要:** Prevailing 3D texture generation methods, which often rely on multi-view fusion, are frequently hindered by inter-view inconsistencies and incomplete coverage of complex surfaces, limiting the fidelity and completeness of the generated content. To overcome these challenges, we introduce TEXTRIX, a native 3D attribute generation framework for high-fidelity texture synthesis and downstream applications such as precise 3D part segmentation. Our approach constructs a latent 3D attribute grid and leverages a Diffusion Transformer equipped with sparse attention, enabling direct coloring of 3D models in volumetric space and fundamentally avoiding the limitations of multi-view fusion. Built upon this native representation, the framework naturally extends to high-precision 3D segmentation by training the same architecture to predict semantic attributes on the grid. Extensive experiments demonstrate state-of-the-art performance on both tasks, producing seamless, high-fidelity textures and accurate 3D part segmentation with precise boundaries.
>
---
#### [new 024] Boosting Medical Vision-Language Pretraining via Momentum Self-Distillation under Limited Computing Resources
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医疗视觉-语言预训练中标注难、计算资源有限的问题，提出基于动量自蒸馏与梯度累积的方法。通过动量机制增强知识提取，提升小样本和零样本性能，实现单GPU高效训练，显著改善模型在少样本适应与检索任务中的表现。**

- **链接: [https://arxiv.org/pdf/2512.02438v1](https://arxiv.org/pdf/2512.02438v1)**

> **作者:** Phuc Pham; Nhu Pham; Ngoc Quoc Ly
>
> **备注:** WACV 2026
>
> **摘要:** In medical healthcare, obtaining detailed annotations is challenging, highlighting the need for robust Vision-Language Models (VLMs). Pretrained VLMs enable fine-tuning on small datasets or zero-shot inference, achieving performance comparable to task-specific models. Contrastive learning (CL) is a key paradigm for training VLMs but inherently requires large batch sizes for effective learning, making it computationally demanding and often limited to well-resourced institutions. Moreover, with limited data in healthcare, it is important to prioritize knowledge extraction from both data and models during training to improve performance. Therefore, we focus on leveraging the momentum method combined with distillation to simultaneously address computational efficiency and knowledge exploitation. Our contributions can be summarized as follows: (1) leveraging momentum self-distillation to enhance multimodal learning, and (2) integrating momentum mechanisms with gradient accumulation to enlarge the effective batch size without increasing resource consumption. Our method attains competitive performance with state-of-the-art (SOTA) approaches in zero-shot classification, while providing a substantial boost in the few-shot adaption, achieving over 90% AUC-ROC and improving retrieval tasks by 2-3%. Importantly, our method achieves high training efficiency with a single GPU while maintaining reasonable training time. Our approach aims to advance efficient multimodal learning by reducing resource requirements while improving performance over SOTA methods. The implementation of our method is available at https://github.com/phphuc612/MSD .
>
---
#### [new 025] MindGPT-4ov: An Enhanced MLLM via a Multi-Stage Post-Training Paradigm
- **分类: cs.CV**

- **简介: 该论文提出MindGPT-4ov，一种基于多阶段后训练的多模态大模型。针对MLLM泛化能力弱、部署成本高等问题，创新性地设计高信息密度数据生成、协同课程微调与混合强化学习策略，并结合训练优化技术，显著提升性能与效率，实现从学术到工业的无缝落地。**

- **链接: [https://arxiv.org/pdf/2512.02895v1](https://arxiv.org/pdf/2512.02895v1)**

> **作者:** Wei Chen; Chaoqun Du; Feng Gu; Wei He; Qizhen Li; Zide Liu; Xuhao Pan; Chang Ren; Xudong Rao; Chenfeng Wang; Tao Wei; Chengjun Yu; Pengfei Yu; Yufei Zheng; Chunpeng Zhou; Pan Zhou; Xuhan Zhu
>
> **备注:** 33 pages, 14 figures
>
> **摘要:** We present MindGPT-4ov, a multimodal large language model (MLLM) that introduces a general post-training paradigm spanning data production, model training, and efficient deployment. It achieves state-of-the-art performance across multiple benchmarks at low cost, effectively enhancing the foundational capabilities of MLLMs and the generalization ability. Focusing on data construction, supervised fine-tuning strategies, and multimodal reinforcement learning methods, this work proposes three key innovations: (1) An information density-based data generation scheme, integrated with a dual-dimensional tree-structured label system, enabling automated generation of high-quality cross-domain data. (2) A collaborative curriculum supervised fine-tuning approach that balances the injection of domain-specific knowledge with the preservation of general capabilities. (3) A hybrid reinforcement learning paradigm that enhances reasoning ability while simultaneously addressing multi-objective optimization such as diversity exploration, maintenance of multimodal perception, and response conciseness. Moreover, we implement a series of infrastructure optimizations, such as 5D parallel training, operator optimization, and inference quantization to enhance training and inference efficiency while reducing the cost of domain adaptation. Experimental results demonstrate that the MindGPT-4ov model outperforms state-of-the-art models on benchmarks such as MMBench, MMStar, MathVision, and MathVista. In addition, MindGPT-4ov also demonstrates superior user experience in vertical domain tasks, enabling a seamless transition from academic research to industrial deployment. MindGPT-4ov provides a general post-training paradigm applicable to a wide range of MLLMs. The model weights, datasets, and code for the Qwen3-VL-based variants will be recently open-sourced to support the community's development of MLLMs.
>
---
#### [new 026] GeoDiT: A Diffusion-based Vision-Language Model for Geospatial Understanding
- **分类: cs.CV**

- **简介: 该论文提出GeoDiT，首个基于扩散模型的地理空间视觉语言模型。针对自回归模型在地理空间理解中序列化生成导致结构不一致的问题，提出并行精炼的生成范式，实现从粗到细的全局合成。在图像描述、视觉定位和多对象检测等任务上取得新基准，验证了生成过程与数据结构对齐的重要性。**

- **链接: [https://arxiv.org/pdf/2512.02505v1](https://arxiv.org/pdf/2512.02505v1)**

> **作者:** Jiaqi Liu; Ronghao Fu; Haoran Liu; Lang Sun; Bo Yang
>
> **摘要:** Autoregressive models are structurally misaligned with the inherently parallel nature of geospatial understanding, forcing a rigid sequential narrative onto scenes and fundamentally hindering the generation of structured and coherent outputs. We challenge this paradigm by reframing geospatial generation as a parallel refinement process, enabling a holistic, coarse-to-fine synthesis that resolves all semantic elements simultaneously. To operationalize this, we introduce GeoDiT, the first diffusion-based vision-language model tailored for the geospatial domain. Extensive experiments demonstrate that GeoDiT establishes a new state-of-the-art on benchmarks requiring structured, object-centric outputs. It achieves significant gains in image captioning, visual grounding, and multi-object detection, precisely the tasks where autoregressive models falter. Our work validates that aligning the generative process with the data's intrinsic structure is key to unlocking superior performance in complex geospatial analysis.
>
---
#### [new 027] Unsupervised Structural Scene Decomposition via Foreground-Aware Slot Attention with Pseudo-Mask Guidance
- **分类: cs.CV**

- **简介: 该论文针对无监督场景分解任务，解决现有方法对前景与背景处理不分、导致干扰和实例发现效果不佳的问题。提出FASA框架，通过两阶段机制：先粗粒度分离前景与背景，再利用伪掩码引导的掩码槽注意力精细建模前景对象，提升对象发现与表示一致性。**

- **链接: [https://arxiv.org/pdf/2512.02685v1](https://arxiv.org/pdf/2512.02685v1)**

> **作者:** Huankun Sheng; Ming Li; Yixiang Wei; Yeying Fan; Yu-Hui Wen; Tieliang Gong; Yong-Jin Liu
>
> **摘要:** Recent advances in object-centric representation learning have shown that slot attention-based methods can effectively decompose visual scenes into object slot representations without supervision. However, existing approaches typically process foreground and background regions indiscriminately, often resulting in background interference and suboptimal instance discovery performance on real-world data. To address this limitation, we propose Foreground-Aware Slot Attention (FASA), a two-stage framework that explicitly separates foreground from background to enable precise object discovery. In the first stage, FASA performs a coarse scene decomposition to distinguish foreground from background regions through a dual-slot competition mechanism. These slots are initialized via a clustering-based strategy, yielding well-structured representations of salient regions. In the second stage, we introduce a masked slot attention mechanism where the first slot captures the background while the remaining slots compete to represent individual foreground objects. To further address over-segmentation of foreground objects, we incorporate pseudo-mask guidance derived from a patch affinity graph constructed with self-supervised image features to guide the learning of foreground slots. Extensive experiments on both synthetic and real-world datasets demonstrate that FASA consistently outperforms state-of-the-art methods, validating the effectiveness of explicit foreground modeling and pseudo-mask guidance for robust scene decomposition and object-coherent representation. Code will be made publicly available.
>
---
#### [new 028] GUI Exploration Lab: Enhancing Screen Navigation in Agents via Multi-Turn Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文聚焦GUI代理的复杂屏幕导航任务，针对真实环境信息难获取的问题，提出GUI Exploration Lab仿真引擎，支持灵活构建与全量访问环境。通过多轮强化学习，逐步提升代理的探索与泛化能力，验证了方法在静态与交互基准上的有效性，为构建更智能的GUI代理提供新路径。**

- **链接: [https://arxiv.org/pdf/2512.02423v1](https://arxiv.org/pdf/2512.02423v1)**

> **作者:** Haolong Yan; Yeqing Shen; Xin Huang; Jia Wang; Kaijun Tan; Zhixuan Liang; Hongxin Li; Zheng Ge; Osamu Yoshie; Si Li; Xiangyu Zhang; Daxin Jiang
>
> **备注:** 26 pages
>
> **摘要:** With the rapid development of Large Vision Language Models, the focus of Graphical User Interface (GUI) agent tasks shifts from single-screen tasks to complex screen navigation challenges. However, real-world GUI environments, such as PC software and mobile Apps, are often complex and proprietary, making it difficult to obtain the comprehensive environment information needed for agent training and evaluation. This limitation hinders systematic investigation and benchmarking of agent navigation capabilities. To address this limitation, we introduce GUI Exploration Lab, a simulation environment engine for GUI agent navigation research that enables flexible definition and composition of screens, icons, and navigation graphs, while providing full access to environment information for comprehensive agent training and evaluation. Through extensive experiments, we find that supervised fine-tuning enables effective memorization of fundamental knowledge, serving as a crucial foundation for subsequent training. Building on this, single-turn reinforcement learning further enhances generalization to unseen scenarios. Finally, multi-turn reinforcement learning encourages the development of exploration strategies through interactive trial and error, leading to further improvements in screen navigation performance. We validate our methods on both static and interactive benchmarks, demonstrating that our findings generalize effectively to real-world scenarios. These findings demonstrate the advantages of reinforcement learning approaches in GUI navigation and offer practical guidance for building more capable and generalizable GUI agents.
>
---
#### [new 029] Content-Aware Texturing for Gaussian Splatting
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对高精度3D重建中的细节表示效率问题，提出基于内容感知的纹理映射方法。通过动态调整每个高斯原语的纹理分辨率，使纹理适配输入图像内容，在保证图像质量的同时减少参数量，提升渲染效率。**

- **链接: [https://arxiv.org/pdf/2512.02621v1](https://arxiv.org/pdf/2512.02621v1)**

> **作者:** Panagiotis Papantonakis; Georgios Kopanas; Fredo Durand; George Drettakis
>
> **备注:** Project Page: https://repo-sam.inria.fr/nerphys/gs-texturing/
>
> **摘要:** Gaussian Splatting has become the method of choice for 3D reconstruction and real-time rendering of captured real scenes. However, fine appearance details need to be represented as a large number of small Gaussian primitives, which can be wasteful when geometry and appearance exhibit different frequency characteristics. Inspired by the long tradition of texture mapping, we propose to use texture to represent detailed appearance where possible. Our main focus is to incorporate per-primitive texture maps that adapt to the scene in a principled manner during Gaussian Splatting optimization. We do this by proposing a new appearance representation for 2D Gaussian primitives with textures where the size of a texel is bounded by the image sampling frequency and adapted to the content of the input images. We achieve this by adaptively upscaling or downscaling the texture resolution during optimization. In addition, our approach enables control of the number of primitives during optimization based on texture resolution. We show that our approach performs favorably in image quality and total number of parameters used compared to alternative solutions for textured Gaussian primitives. Project page: https://repo-sam.inria.fr/nerphys/gs-texturing/
>
---
#### [new 030] Vision to Geometry: 3D Spatial Memory for Sequential Embodied MLLM Reasoning and Exploration
- **分类: cs.CV**

- **简介: 该论文研究顺序性室内具身任务中的空间记忆问题，针对多步任务中前序探索知识难以复用的挑战，提出3DSPMR方法，通过融合视觉、几何与关系线索增强多模态大模型的空间推理能力。构建SEER-Bench基准，验证了方法在顺序型EQA与EMN任务上的显著提升。**

- **链接: [https://arxiv.org/pdf/2512.02458v1](https://arxiv.org/pdf/2512.02458v1)**

> **作者:** Zhongyi Cai; Yi Du; Chen Wang; Yu Kong
>
> **摘要:** Existing research on indoor embodied tasks typically requires agents to actively explore unknown environments and reason about the scene to achieve a specific goal. However, when deployed in real life, agents often face sequential tasks, where each new sub-task follows the completion of the previous one, and certain sub-tasks may be infeasible, such as searching for a non-existent object. Compared with the single-task setting, the core challenge lies in reusing spatial knowledge accumulated from previous explorations to support subsequent reasoning and exploration. In this work, we investigate this underexplored yet practically significant embodied AI challenge. To evaluate this challenge, we introduce SEER-Bench, a new Sequential Embodied Exploration and Reasoning Benchmark encompassing encompassing two classic embodied tasks: Embodied Question Answering (EQA) and Embodied Multi-modal Navigation (EMN). Building on SEER-Bench, we propose 3DSPMR, a 3D SPatial Memory Reasoning approach that exploits relational, visual, and geometric cues from explored regions to augment Multi-Modal Large Language Models (MLLMs) for reasoning and exploration in sequential embodied tasks. To the best of our knowledge, this is the first work to explicitly incorporate geometric information into MLLM-based spatial understanding and reasoning. Extensive experiments verify that 3DSPMR achieves substantial performance gains on both sequential EQA and EMN tasks.
>
---
#### [new 031] Multifractal Recalibration of Neural Networks for Medical Imaging Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医学图像分割任务，提出多分形重校准方法，解决现有深度学习模型在特征提取中忽略复杂尺度结构的问题。通过引入单分形与多分形先验，构建基于通道注意力的统计描述，提升U-Net模型对病理规律的捕捉能力，在多个医学影像数据集上实现性能提升。**

- **链接: [https://arxiv.org/pdf/2512.02198v1](https://arxiv.org/pdf/2512.02198v1)**

> **作者:** Miguel L. Martins; Miguel T. Coimbra; Francesco Renna
>
> **备注:** 30 pages, 9 figures, journal paper
>
> **摘要:** Multifractal analysis has revealed regularities in many self-seeding phenomena, yet its use in modern deep learning remains limited. Existing end-to-end multifractal methods rely on heavy pooling or strong feature-space decimation, which constrain tasks such as semantic segmentation. Motivated by these limitations, we introduce two inductive priors: Monofractal and Multifractal Recalibration. These methods leverage relationships between the probability mass of the exponents and the multifractal spectrum to form statistical descriptions of encoder embeddings, implemented as channel-attention functions in convolutional networks. Using a U-Net-based framework, we show that multifractal recalibration yields substantial gains over a baseline equipped with other channel-attention mechanisms that also use higher-order statistics. Given the proven ability of multifractal analysis to capture pathological regularities, we validate our approach on three public medical-imaging datasets: ISIC18 (dermoscopy), Kvasir-SEG (endoscopy), and BUSI (ultrasound). Our empirical analysis also provides insights into the behavior of these attention layers. We find that excitation responses do not become increasingly specialized with encoder depth in U-Net architectures due to skip connections, and that their effectiveness may relate to global statistics of instance variability.
>
---
#### [new 032] On the Problem of Consistent Anomalies in Zero-Shot Anomaly Detection
- **分类: cs.CV; stat.ML**

- **简介: 该论文研究零样本异常检测（AC/AS）任务，针对一致异常导致距离方法偏差的问题，揭示了特征表示中的相似性缩放与邻居湮灭现象。提出CoDeGraph图框架过滤一致异常，并拓展至3D医学图像，实现无训练的零样本3D异常检测与分割，同时融合文本提示模型提升性能。**

- **链接: [https://arxiv.org/pdf/2512.02520v1](https://arxiv.org/pdf/2512.02520v1)**

> **作者:** Tai Le-Gia
>
> **备注:** PhD Dissertation
>
> **摘要:** Zero-shot anomaly classification and segmentation (AC/AS) aim to detect anomalous samples and regions without any training data, a capability increasingly crucial in industrial inspection and medical imaging. This dissertation aims to investigate the core challenges of zero-shot AC/AS and presents principled solutions rooted in theory and algorithmic design. We first formalize the problem of consistent anomalies, a failure mode in which recurring similar anomalies systematically bias distance-based methods. By analyzing the statistical and geometric behavior of patch representations from pre-trained Vision Transformers, we identify two key phenomena - similarity scaling and neighbor-burnout - that describe how relationships among normal patches change with and without consistent anomalies in settings characterized by highly similar objects. We then introduce CoDeGraph, a graph-based framework for filtering consistent anomalies built on the similarity scaling and neighbor-burnout phenomena. Through multi-stage graph construction, community detection, and structured refinement, CoDeGraph effectively suppresses the influence of consistent anomalies. Next, we extend this framework to 3D medical imaging by proposing a training-free, computationally efficient volumetric tokenization strategy for MRI data. This enables a genuinely zero-shot 3D anomaly detection pipeline and shows that volumetric anomaly segmentation is achievable without any 3D training samples. Finally, we bridge batch-based and text-based zero-shot methods by demonstrating that CoDeGraph-derived pseudo-masks can supervise prompt-driven vision-language models. Together, this dissertation provides theoretical understanding and practical solutions for the zero-shot AC/AS problem.
>
---
#### [new 033] GraphFusion3D: Dynamic Graph Attention Convolution with Adaptive Cross-Modal Transformer for 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文针对3D目标检测中点云稀疏、结构不完整及语义信息不足的问题，提出GraphFusion3D框架。通过自适应跨模态变换器融合图像与点云特征，结合图推理模块动态建模局部几何与全局语义关系，并采用级联解码器提升检测精度，在SUN RGB-D和ScanNetV2上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.02991v1](https://arxiv.org/pdf/2512.02991v1)**

> **作者:** Md Sohag Mia; Md Nahid Hasan; Tawhid Ahmed; Muhammad Abdullah Adnan
>
> **摘要:** Despite significant progress in 3D object detection, point clouds remain challenging due to sparse data, incomplete structures, and limited semantic information. Capturing contextual relationships between distant objects presents additional difficulties. To address these challenges, we propose GraphFusion3D, a unified framework combining multi-modal fusion with advanced feature learning. Our approach introduces the Adaptive Cross-Modal Transformer (ACMT), which adaptively integrates image features into point representations to enrich both geometric and semantic information. For proposal refinement, we introduce the Graph Reasoning Module (GRM), a novel mechanism that models neighborhood relationships to simultaneously capture local geometric structures and global semantic context. The module employs multi-scale graph attention to dynamically weight both spatial proximity and feature similarity between proposals. We further employ a cascade decoder that progressively refines detections through multi-stage predictions. Extensive experiments on SUN RGB-D (70.6\% AP$_{25}$ and 51.2\% AP$_{50}$) and ScanNetV2 (75.1\% AP$_{25}$ and 60.8\% AP$_{50}$) demonstrate a substantial performance improvement over existing approaches.
>
---
#### [new 034] Action Anticipation at a Glimpse: To What Extent Can Multimodal Cues Replace Video?
- **分类: cs.CV**

- **简介: 该论文研究单帧图像下的动作预测任务，旨在减少对视频时序信息的依赖。提出AAG方法，融合单帧RGB、深度特征与文本或单帧动作预测作为上下文，实现高效动作前瞻。在三个装配任务数据集上验证，性能媲美视频基线。**

- **链接: [https://arxiv.org/pdf/2512.02846v1](https://arxiv.org/pdf/2512.02846v1)**

> **作者:** Manuel Benavent-Lledo; Konstantinos Bacharidis; Victoria Manousaki; Konstantinos Papoutsakis; Antonis Argyros; Jose Garcia-Rodriguez
>
> **备注:** Accepted in WACV 2026 - Applications Track
>
> **摘要:** Anticipating actions before they occur is a core challenge in action understanding research. While conventional methods rely on extracting and aggregating temporal information from videos, as humans we can often predict upcoming actions by observing a single moment from a scene, when given sufficient context. Can a model achieve this competence? The short answer is yes, although its effectiveness depends on the complexity of the task. In this work, we investigate to what extent video aggregation can be replaced with alternative modalities. To this end, based on recent advances in visual feature extraction and language-based reasoning, we introduce AAG, a method for Action Anticipation at a Glimpse. AAG combines RGB features with depth cues from a single frame for enhanced spatial reasoning, and incorporates prior action information to provide long-term context. This context is obtained either through textual summaries from Vision-Language Models, or from predictions generated by a single-frame action recognizer. Our results demonstrate that multimodal single-frame action anticipation using AAG can perform competitively compared to both temporally aggregated video baselines and state-of-the-art methods across three instructional activity datasets: IKEA-ASM, Meccano, and Assembly101.
>
---
#### [new 035] ALDI-ray: Adapting the ALDI Framework for Security X-ray Object Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对安全X-ray图像中因设备与环境差异导致的域偏移问题，提出将ALDI++框架应用于跨域目标检测。通过自蒸馏、特征对齐与增强训练策略，显著提升模型在不同场景下的泛化能力，实验表明其在EDS数据集上优于现有方法，尤其在基于ViTDet的架构下表现突出。**

- **链接: [https://arxiv.org/pdf/2512.02696v1](https://arxiv.org/pdf/2512.02696v1)**

> **作者:** Omid Reza Heidari; Yang Wang; Xinxin Zuo
>
> **备注:** Submitted to ICASSP 2026 Conference
>
> **摘要:** Domain adaptation in object detection is critical for real-world applications where distribution shifts degrade model performance. Security X-ray imaging presents a unique challenge due to variations in scanning devices and environmental conditions, leading to significant domain discrepancies. To address this, we apply ALDI++, a domain adaptation framework that integrates self-distillation, feature alignment, and enhanced training strategies to mitigate domain shift effectively in this area. We conduct extensive experiments on the EDS dataset, demonstrating that ALDI++ surpasses the state-of-the-art (SOTA) domain adaptation methods across multiple adaptation scenarios. In particular, ALDI++ with a Vision Transformer for Detection (ViTDet) backbone achieves the highest mean average precision (mAP), confirming the effectiveness of transformer-based architectures for cross-domain object detection. Additionally, our category-wise analysis highlights consistent improvements in detection accuracy, reinforcing the robustness of the model across diverse object classes. Our findings establish ALDI++ as an efficient solution for domain-adaptive object detection, setting a new benchmark for performance stability and cross-domain generalization in security X-ray imagery.
>
---
#### [new 036] dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model
- **分类: cs.CV**

- **简介: 该论文提出dots.ocr，一个统一的多语言文档布局解析模型。针对现有方法依赖多阶段流水线导致误差传播的问题，首次在单一视觉-语言模型中联合学习布局检测、文本识别与关系理解。通过大规模多语言数据训练，在OmniDocBench和新基准XDocParse上均达到领先性能，显著提升跨语言、跨领域文档理解能力。**

- **链接: [https://arxiv.org/pdf/2512.02498v1](https://arxiv.org/pdf/2512.02498v1)**

> **作者:** Yumeng Li; Guang Yang; Hao Liu; Bowen Wang; Colin Zhang
>
> **摘要:** Document Layout Parsing serves as a critical gateway for Artificial Intelligence (AI) to access and interpret the world's vast stores of structured knowledge. This process,which encompasses layout detection, text recognition, and relational understanding, is particularly crucial for empowering next-generation Vision-Language Models. Current methods, however, rely on fragmented, multi-stage pipelines that suffer from error propagation and fail to leverage the synergies of joint training. In this paper, we introduce dots.ocr, a single Vision-Language Model that, for the first time, demonstrates the advantages of jointly learning three core tasks within a unified, end-to-end framework. This is made possible by a highly scalable data engine that synthesizes a vast multilingual corpus, empowering the model to deliver robust performance across a wide array of tasks, encompassing diverse languages, layouts, and domains. The efficacy of our unified paradigm is validated by state-of-the-art performance on the comprehensive OmniDocBench. Furthermore, to catalyze research in global document intelligence, we introduce XDocParse, a challenging new benchmark spanning 126 languages. On this testbed, dots.ocr establishes a powerful new baseline, outperforming the next-best competitor by a remarkable +7.4 point margin and proving its unparalleled multilingual capabilities.
>
---
#### [new 037] Benchmarking Scientific Understanding and Reasoning for Video Generation using VideoScience-Bench
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VideoScience-Bench，首个评估视频生成模型科学理解与推理能力的基准。针对现有基准仅考察物理常识、缺乏科学推理评估的问题，构建涵盖14主题、103概念的200个复合科学场景，通过五维指标量化模型生成视频的科学合理性，验证其零样本推理能力。**

- **链接: [https://arxiv.org/pdf/2512.02942v1](https://arxiv.org/pdf/2512.02942v1)**

> **作者:** Lanxiang Hu; Abhilash Shankarampeta; Yixin Huang; Zilin Dai; Haoyang Yu; Yujie Zhao; Haoqiang Kang; Daniel Zhao; Tajana Rosing; Hao Zhang
>
> **摘要:** The next frontier for video generation lies in developing models capable of zero-shot reasoning, where understanding real-world scientific laws is crucial for accurate physical outcome modeling under diverse conditions. However, existing video benchmarks are physical commonsense-based, offering limited insight into video models' scientific reasoning capability. We introduce VideoScience-Bench, a benchmark designed to evaluate undergraduate-level scientific understanding in video models. Each prompt encodes a composite scientific scenario that requires understanding and reasoning across multiple scientific concepts to generate the correct phenomenon. The benchmark comprises 200 carefully curated prompts spanning 14 topics and 103 concepts in physics and chemistry. We conduct expert-annotated evaluations across seven state-of-the-art video models in T2V and I2V settings along five dimensions: Prompt Consistency, Phenomenon Congruency, Correct Dynamism, Immutability, and Spatio-Temporal Continuity. Using a VLM-as-a-Judge to assess video generations, we observe strong correlation with human assessments. To the best of our knowledge, VideoScience-Bench is the first benchmark to evaluate video models not only as generators but also as reasoners, requiring their generations to demonstrate scientific understanding consistent with expected physical and chemical phenomena. Our data and evaluation code are available at: \href{https://github.com/hao-ai-lab/VideoScience}{github.com/hao-ai-lab/VideoScience}.
>
---
#### [new 038] WorldPack: Compressed Memory Improves Spatial Consistency in Video World Modeling
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视频世界建模中的长期时空一致性难题，提出WorldPack模型。通过轨迹压缩与记忆检索机制，在缩短上下文长度的前提下，显著提升生成结果的空间一致性和质量，有效解决长时序建模的计算成本高问题。**

- **链接: [https://arxiv.org/pdf/2512.02473v1](https://arxiv.org/pdf/2512.02473v1)**

> **作者:** Yuta Oshima; Yusuke Iwasawa; Masahiro Suzuki; Yutaka Matsuo; Hiroki Furuta
>
> **摘要:** Video world models have attracted significant attention for their ability to produce high-fidelity future visual observations conditioned on past observations and navigation actions. Temporally- and spatially-consistent, long-term world modeling has been a long-standing problem, unresolved with even recent state-of-the-art models, due to the prohibitively expensive computational costs for long-context inputs. In this paper, we propose WorldPack, a video world model with efficient compressed memory, which significantly improves spatial consistency, fidelity, and quality in long-term generation despite much shorter context length. Our compressed memory consists of trajectory packing and memory retrieval; trajectory packing realizes high context efficiency, and memory retrieval maintains the consistency in rollouts and helps long-term generations that require spatial reasoning. Our performance is evaluated with LoopNav, a benchmark on Minecraft, specialized for the evaluation of long-term consistency, and we verify that WorldPack notably outperforms strong state-of-the-art models.
>
---
#### [new 039] Rethinking Surgical Smoke: A Smoke-Type-Aware Laparoscopic Video Desmoking Method and Dataset
- **分类: cs.CV**

- **简介: 该论文针对腹腔镜视频中手术烟雾导致视觉干扰的问题，提出首个烟雾类型感知的去烟方法STANet。通过区分扩散型与环境型烟雾，设计双分支网络与解耦模块，实现精准去烟，并构建首个带烟雾类型标注的大规模合成数据集，显著提升去烟效果与下游任务泛化性。**

- **链接: [https://arxiv.org/pdf/2512.02780v1](https://arxiv.org/pdf/2512.02780v1)**

> **作者:** Qifan Liang; Junlin Li; Zhen Han; Xihao Wang; Zhongyuan Wang; Bin Mei
>
> **备注:** 12 pages, 15 figures. Accepted to AAAI-26 (Main Technical Track)
>
> **摘要:** Electrocautery or lasers will inevitably generate surgical smoke, which hinders the visual guidance of laparoscopic videos for surgical procedures. The surgical smoke can be classified into different types based on its motion patterns, leading to distinctive spatio-temporal characteristics across smoky laparoscopic videos. However, existing desmoking methods fail to account for such smoke-type-specific distinctions. Therefore, we propose the first Smoke-Type-Aware Laparoscopic Video Desmoking Network (STANet) by introducing two smoke types: Diffusion Smoke and Ambient Smoke. Specifically, a smoke mask segmentation sub-network is designed to jointly conduct smoke mask and smoke type predictions based on the attention-weighted mask aggregation, while a smokeless video reconstruction sub-network is proposed to perform specially desmoking on smoky features guided by two types of smoke mask. To address the entanglement challenges of two smoke types, we further embed a coarse-to-fine disentanglement module into the mask segmentation sub-network, which yields more accurate disentangled masks through the smoke-type-aware cross attention between non-entangled and entangled regions. In addition, we also construct the first large-scale synthetic video desmoking dataset with smoke type annotations. Extensive experiments demonstrate that our method not only outperforms state-of-the-art approaches in quality evaluations, but also exhibits superior generalization across multiple downstream surgical tasks.
>
---
#### [new 040] GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization
- **分类: cs.CV**

- **简介: 该论文提出GeoBridge，一种跨视图地理定位的多视图基础模型，解决传统卫星主导方法在高分辨率或实时影像缺失时鲁棒性差的问题。通过语义锚机制实现图像与文本间的双向匹配，构建了首个大规模多视图对齐数据集GeoLoc，显著提升定位精度与跨域泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.02697v1](https://arxiv.org/pdf/2512.02697v1)**

> **作者:** Zixuan Song; Jing Zhang; Di Wang; Zidie Zhou; Wenbin Liu; Haonan Guo; En Wang; Bo Du
>
> **摘要:** Cross-view geo-localization infers a location by retrieving geo-tagged reference images that visually correspond to a query image. However, the traditional satellite-centric paradigm limits robustness when high-resolution or up-to-date satellite imagery is unavailable. It further underexploits complementary cues across views (e.g., drone, satellite, and street) and modalities (e.g., language and image). To address these challenges, we propose GeoBridge, a foundation model that performs bidirectional matching across views and supports language-to-image retrieval. Going beyond traditional satellite-centric formulations, GeoBridge builds on a novel semantic-anchor mechanism that bridges multi-view features through textual descriptions for robust, flexible localization. In support of this task, we construct GeoLoc, the first large-scale, cross-modal, and multi-view aligned dataset comprising over 50,000 pairs of drone, street-view panorama, and satellite images as well as their textual descriptions, collected from 36 countries, ensuring both geographic and semantic alignment. We performed broad evaluations across multiple tasks. Experiments confirm that GeoLoc pre-training markedly improves geo-location accuracy for GeoBridge while promoting cross-domain generalization and cross-modal knowledge transfer. The dataset, source code, and pretrained models were released at https://github.com/MiliLab/GeoBridge.
>
---
#### [new 041] VACoT: Rethinking Visual Data Augmentation with VLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型（VLMs）在数据增强方面投入不足的问题，提出VACoT框架，在推理阶段动态应用视觉增强。通过结构化增强与条件奖励机制，提升模型对复杂和分布外输入的鲁棒性，尤其在对抗性OCR任务中表现优异，降低训练成本并增强泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.02361v1](https://arxiv.org/pdf/2512.02361v1)**

> **作者:** Zhengzhuo Xu; Chong Sun; SiNan Du; Chen Li; Jing Lyu; Chun Yuan
>
> **摘要:** While visual data augmentation remains a cornerstone for training robust vision models, it has received limited attention in visual language models (VLMs), which predominantly rely on large-scale real data acquisition or synthetic diversity. Consequently, they may struggle with basic perception tasks that conventional models handle reliably. Given the substantial cost of pre-training and fine-tuning VLMs, continue training on augmented data yields limited and diminishing returns. In this paper, we present Visual Augmentation Chain-of-Thought (VACoT), a framework that dynamically invokes image augmentations during model inference. By incorporating post-hoc transformations such as denoising, VACoT substantially improves robustness on challenging and out-of-distribution inputs, especially in OCR-related adversarial scenarios. Distinct from prior approaches limited to local cropping, VACoT integrates a structured collection of general visual augmentations, broadening the query image views while reducing training complexity and computational overhead with efficient agentic reinforcement learning. We propose a conditional reward scheme that encourages necessary augmentation while penalizing verbose responses, ensuring concise and effective reasoning in perception tasks. We demonstrate the superiority of VACoT with extensive experiments on 13 perception benchmarks and further introduce AdvOCR to highlight the generalization benefits of post-hoc visual augmentations in adversarial scenarios.
>
---
#### [new 042] OneThinker: All-in-one Reasoning Model for Image and Video
- **分类: cs.CV**

- **简介: 该论文提出OneThinker，一个统一图像与视频理解的通用视觉推理模型。针对现有方法分立训练、难以跨模态共享知识的问题，构建涵盖10类任务的600k数据集，采用CoT标注与EMA-GRPO算法实现多任务强化学习优化，在31个基准上展现强性能与知识迁移能力，推动迈向通用多模态推理。**

- **链接: [https://arxiv.org/pdf/2512.03043v1](https://arxiv.org/pdf/2512.03043v1)**

> **作者:** Kaituo Feng; Manyuan Zhang; Hongyu Li; Kaixuan Fan; Shuang Chen; Yilei Jiang; Dian Zheng; Peiwen Sun; Yiyuan Zhang; Haoze Sun; Yan Feng; Peng Pei; Xunliang Cai; Xiangyu Yue
>
> **备注:** Project page: https://github.com/tulerfeng/OneThinker
>
> **摘要:** Reinforcement learning (RL) has recently achieved remarkable success in eliciting visual reasoning within Multimodal Large Language Models (MLLMs). However, existing approaches typically train separate models for different tasks and treat image and video reasoning as disjoint domains. This results in limited scalability toward a multimodal reasoning generalist, which restricts practical versatility and hinders potential knowledge sharing across tasks and modalities. To this end, we propose OneThinker, an all-in-one reasoning model that unifies image and video understanding across diverse fundamental visual tasks, including question answering, captioning, spatial and temporal grounding, tracking, and segmentation. To achieve this, we construct the OneThinker-600k training corpus covering all these tasks and employ commercial models for CoT annotation, resulting in OneThinker-SFT-340k for SFT cold start. Furthermore, we propose EMA-GRPO to handle reward heterogeneity in multi-task RL by tracking task-wise moving averages of reward standard deviations for balanced optimization. Extensive experiments on diverse visual benchmarks show that OneThinker delivers strong performance on 31 benchmarks, across 10 fundamental visual understanding tasks. Moreover, it exhibits effective knowledge transfer between certain tasks and preliminary zero-shot generalization ability, marking a step toward a unified multimodal reasoning generalist. All code, model, and data are released.
>
---
#### [new 043] LumiX: Structured and Coherent Text-to-Intrinsic Generation
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文提出LumiX，一种用于文本到固有属性生成的结构化扩散框架。针对多属性生成不一致问题，引入查询广播注意力与张量LoRA，实现各固有图（如反照率、法线等）的联合生成与结构一致性，显著提升物理合理性与生成质量。**

- **链接: [https://arxiv.org/pdf/2512.02781v1](https://arxiv.org/pdf/2512.02781v1)**

> **作者:** Xu Han; Biao Zhang; Xiangjun Tang; Xianzhi Li; Peter Wonka
>
> **备注:** The code will be available at https://github.com/xhanxu/LumiX
>
> **摘要:** We present LumiX, a structured diffusion framework for coherent text-to-intrinsic generation. Conditioned on text prompts, LumiX jointly generates a comprehensive set of intrinsic maps (e.g., albedo, irradiance, normal, depth, and final color), providing a structured and physically consistent description of an underlying scene. This is enabled by two key contributions: 1) Query-Broadcast Attention, a mechanism that ensures structural consistency by sharing queries across all maps in each self-attention block. 2) Tensor LoRA, a tensor-based adaptation that parameter-efficiently models cross-map relations for efficient joint training. Together, these designs enable stable joint diffusion training and unified generation of multiple intrinsic properties. Experiments show that LumiX produces coherent and physically meaningful results, achieving 23% higher alignment and a better preference score (0.19 vs. -0.41) compared to the state of the art, and it can also perform image-conditioned intrinsic decomposition within the same framework.
>
---
#### [new 044] From Detection to Association: Learning Discriminative Object Embeddings for Multi-Object Tracking
- **分类: cs.CV**

- **简介: 该论文针对端到端多目标跟踪中对象嵌入判别性不足的问题，提出FDTA框架。通过空间、时间与身份三重适配器，增强跨帧实例区分能力，提升关联精度，在多个基准上达到领先性能。**

- **链接: [https://arxiv.org/pdf/2512.02392v1](https://arxiv.org/pdf/2512.02392v1)**

> **作者:** Yuqing Shao; Yuchen Yang; Rui Yu; Weilong Li; Xu Guo; Huaicheng Yan; Wei Wang; Xiao Sun
>
> **摘要:** End-to-end multi-object tracking (MOT) methods have recently achieved remarkable progress by unifying detection and association within a single framework. Despite their strong detection performance, these methods suffer from relatively low association accuracy. Through detailed analysis, we observe that object embeddings produced by the shared DETR architecture display excessively high inter-object similarity, as it emphasizes only category-level discrimination within single frames. In contrast, tracking requires instance-level distinction across frames with spatial and temporal continuity, for which current end-to-end approaches insufficiently optimize object embeddings. To address this, we introduce FDTA (From Detection to Association), an explicit feature refinement framework that enhances object discriminativeness across three complementary perspectives. Specifically, we introduce a Spatial Adapter (SA) to integrate depth-aware cues for spatial continuity, a Temporal Adapter (TA) to aggregate historical information for temporal dependencies, and an Identity Adapter (IA) to leverage quality-aware contrastive learning for instance-level separability. Extensive experiments demonstrate that FDTA achieves state-of-the-art performance on multiple challenging MOT benchmarks, including DanceTrack, SportsMOT, and BFT, highlighting the effectiveness of our proposed discriminative embedding enhancement strategy. The code is available at https://github.com/Spongebobbbbbbbb/FDTA.
>
---
#### [new 045] UnicEdit-10M: A Dataset and Benchmark Breaking the Scale-Quality Barrier via Unified Verification for Reasoning-Enriched Edits
- **分类: cs.CV**

- **简介: 该论文针对图像编辑领域因数据稀缺与评估不足导致的性能差距问题，提出轻量级数据生成管道与统一验证机制，构建10M规模高质量数据集UnicEdit-10M及通用基准UnicBench。通过新指标实现对空间与知识推理能力的细粒度诊断，揭示主流模型局限，推动开放模型发展。**

- **链接: [https://arxiv.org/pdf/2512.02790v1](https://arxiv.org/pdf/2512.02790v1)**

> **作者:** Keming Ye; Zhipeng Huang; Canmiao Fu; Qingyang Liu; Jiani Cai; Zheqi Lv; Chen Li; Jing Lyu; Zhou Zhao; Shengyu Zhang
>
> **备注:** 31 pages, 15 figures, 12 tables
>
> **摘要:** With the rapid advances of powerful multimodal models such as GPT-4o, Nano Banana, and Seedream 4.0 in Image Editing, the performance gap between closed-source and open-source models is widening, primarily due to the scarcity of large-scale, high-quality training data and comprehensive benchmarks capable of diagnosing model weaknesses across diverse editing behaviors. Existing data construction methods face a scale-quality trade-off: human annotations are high-quality but not scalable, while automated pipelines suffer from error propagation and noise. To address this, we introduce a lightweight data pipeline that replaces multi-toolchains with an end-to-end model and a unified post-verification stage. For scalable quality control, we train a 7B dual-task expert model, \textbf{Qwen-Verify}, for efficient failure detection and instruction recaptioning. This pipeline yields \textbf{UnicEdit-10M}, a 10M-scale dataset spanning diverse basic and complex editing tasks. We also propose \textbf{UnicBench}, a general benchmark that extends beyond basic edits to explicitly assess spatial and knowledge-driven reasoning. To enable fine-grained diagnosis, we introduce novel metrics, including \textit{Non-edit Consistency} and \textit{Reasoning Accuracy}. Our analysis of mainstream models on UnicBench reveals their limitations and provides clear directions for future research.
>
---
#### [new 046] Towards Unified Video Quality Assessment
- **分类: cs.CV**

- **简介: 该论文针对视频质量评估（VQA）中模型单一、不可解释、格式专用的问题，提出Unified-VQA框架。通过多专家诊断机制与弱监督多任务学习，实现跨格式、多失真类型的统一评估与可解释的故障诊断，显著优于18种基准方法。**

- **链接: [https://arxiv.org/pdf/2512.02224v1](https://arxiv.org/pdf/2512.02224v1)**

> **作者:** Chen Feng; Tianhao Peng; Fan Zhang; David Bull
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Recent works in video quality assessment (VQA) typically employ monolithic models that typically predict a single quality score for each test video. These approaches cannot provide diagnostic, interpretable feedback, offering little insight into why the video quality is degraded. Most of them are also specialized, format-specific metrics rather than truly ``generic" solutions, as they are designed to learn a compromised representation from disparate perceptual domains. To address these limitations, this paper proposes Unified-VQA, a framework that provides a single, unified quality model applicable to various distortion types within multiple video formats by recasting generic VQA as a Diagnostic Mixture-of-Experts (MoE) problem. Unified-VQA employs multiple ``perceptual experts'' dedicated to distinct perceptual domains. A novel multi-proxy expert training strategy is designed to optimize each expert using a ranking-inspired loss, guided by the most suitable proxy metric for its domain. We also integrated a diagnostic multi-task head into this framework to generate a global quality score and an interpretable multi-dimensional artifact vector, which is optimized using a weakly-supervised learning strategy, leveraging the known properties of the large-scale training database generated for this work. With static model parameters (without retraining or fine-tuning), Unified-VQA demonstrates consistent and superior performance compared to over 18 benchmark methods for both generic VQA and diagnostic artifact detection tasks across 17 databases containing diverse streaming artifacts in HD, UHD, HDR and HFR formats. This work represents an important step towards practical, actionable, and interpretable video quality assessment.
>
---
#### [new 047] On-the-fly Feedback SfM: Online Explore-and-Exploit UAV Photogrammetry with Incremental Mesh Quality-Aware Indicator and Predictive Path Planning
- **分类: cs.CV**

- **简介: 该论文针对实时无人机摄影测量中重建质量评估与反馈不足的问题，提出On-the-fly Feedback SfM框架，集成增量粗网格生成、在线质量评估与预测路径规划，实现近实时3D重建与主动反馈，提升覆盖率并降低重飞成本。**

- **链接: [https://arxiv.org/pdf/2512.02375v1](https://arxiv.org/pdf/2512.02375v1)**

> **作者:** Liyuan Lou; Wanyun Li; Wentian Gan; Yifei Yu; Tengfei Wang; Xin Wang; Zongqian Zhan
>
> **备注:** This work was submitted to IEEE GRSM Journal for consideration.COPYRIGHT would be transferred once it get accepted
>
> **摘要:** Compared with conventional offline UAV photogrammetry, real-time UAV photogrammetry is essential for time-critical geospatial applications such as disaster response and active digital-twin maintenance. However, most existing methods focus on processing captured images or sequential frames in real time, without explicitly evaluating the quality of the on-the-go 3D reconstruction or providing guided feedback to enhance image acquisition in the target area. This work presents On-the-fly Feedback SfM, an explore-and-exploit framework for real-time UAV photogrammetry, enabling iterative exploration of unseen regions and exploitation of already observed and reconstructed areas in near real time. Built upon SfM on-the-fly , the proposed method integrates three modules: (1) online incremental coarse-mesh generation for dynamically expanding sparse 3D point cloud; (2) online mesh quality assessment with actionable indicators; and (3) predictive path planning for on-the-fly trajectory refinement. Comprehensive experiments demonstrate that our method achieves in-situ reconstruction and evaluation in near real time while providing actionable feedback that markedly reduces coverage gaps and re-flight costs. Via the integration of data collection, processing, 3D reconstruction and assessment, and online feedback, our on the-fly feedback SfM could be an alternative for the transition from traditional passive working mode to a more intelligent and adaptive exploration workflow. Code is now available at https://github.com/IRIS-LAB-whu/OntheflySfMFeedback.
>
---
#### [new 048] AutoBrep: Autoregressive B-Rep Generation with Unified Topology and Geometry
- **分类: cs.CV**

- **简介: 该论文提出AutoBrep，一种基于Transformer的自回归B-Rep生成模型，旨在解决端到端生成高精度、拓扑封闭的CAD模型难题。通过统一编码几何与拓扑信息为离散序列，实现高质量、可扩展的B-Rep生成，并支持用户可控的自动补全。**

- **链接: [https://arxiv.org/pdf/2512.03018v1](https://arxiv.org/pdf/2512.03018v1)**

> **作者:** Xiang Xu; Pradeep Kumar Jayaraman; Joseph G. Lambourne; Yilin Liu; Durvesh Malpure; Pete Meltzer
>
> **备注:** Accepted to Siggraph Asia 2025
>
> **摘要:** The boundary representation (B-Rep) is the standard data structure used in Computer-Aided Design (CAD) for defining solid models. Despite recent progress, directly generating B-Reps end-to-end with precise geometry and watertight topology remains a challenge. This paper presents AutoBrep, a novel Transformer model that autoregressively generates B-Reps with high quality and validity. AutoBrep employs a unified tokenization scheme that encodes both geometric and topological characteristics of a B-Rep model as a sequence of discrete tokens. Geometric primitives (i.e., surfaces and curves) are encoded as latent geometry tokens, and their structural relationships are defined as special topological reference tokens. Sequence order in AutoBrep naturally follows a breadth first traversal of the B-Rep face adjacency graph. At inference time, neighboring faces and edges along with their topological structure are progressively generated. Extensive experiments demonstrate the advantages of our unified representation when coupled with next-token prediction for B-Rep generation. AutoBrep outperforms baselines with better quality and watertightness. It is also highly scalable to complex solids with good fidelity and inference speed. We further show that autocompleting B-Reps is natively supported through our unified tokenization, enabling user-controllable CAD generation with minimal changes. Code is available at https://github.com/AutodeskAILab/AutoBrep.
>
---
#### [new 049] AVGGT: Rethinking Global Attention for Accelerating VGGT
- **分类: cs.CV**

- **简介: 该论文针对多视图3D重建任务中VGGT等模型因依赖全局自注意力导致计算开销大的问题，通过分析发现早期层无对应关系、中间层负责跨视图对齐、末层仅微调。提出无需训练的两步加速方案：将早期层转为帧注意力，采用对角保留与均值填充的子采样策略，实现8–10倍加速，同时保持或提升精度。**

- **链接: [https://arxiv.org/pdf/2512.02541v1](https://arxiv.org/pdf/2512.02541v1)**

> **作者:** Xianbing Sun; Zhikai Zhu; Zhengyu Lou; Bo Yang; Jinyang Tang; Liqing Zhang; He Wang; Jianfu Zhang
>
> **摘要:** Since DUSt3R, models such as VGGT and $π^3$ have shown strong multi-view 3D performance, but their heavy reliance on global self-attention results in high computational cost. Existing sparse-attention variants offer partial speedups, yet lack a systematic analysis of how global attention contributes to multi-view reasoning. In this paper, we first conduct an in-depth investigation of the global attention modules in VGGT and $π^3$ to better understand their roles. Our analysis reveals a clear division of roles in the alternating global-frame architecture: early global layers do not form meaningful correspondences, middle layers perform cross-view alignment, and last layers provide only minor refinements. Guided by these findings, we propose a training-free two-step acceleration scheme: (1) converting early global layers into frame attention, and (2) subsampling global attention by subsampling K/V over patch tokens with diagonal preservation and a mean-fill component. We instantiate this strategy on VGGT and $π^3$ and evaluate across standard pose and point-map benchmarks. Our method achieves up to $8$-$10\times$ speedup in inference time while matching or slightly improving the accuracy of the original models, and remains robust even in extremely dense multi-view settings where prior sparse-attention baselines fail.
>
---
#### [new 050] nuScenes Revisited: Progress and Challenges in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文回顾nuScenes数据集的构建与影响，分析其在自动驾驶中的关键作用。针对数据集标准缺失与多模态融合挑战，系统梳理其技术细节、扩展版本及对后续研究的影响，总结主流方法与任务进展，为自动驾驶研究提供全面综述。**

- **链接: [https://arxiv.org/pdf/2512.02448v1](https://arxiv.org/pdf/2512.02448v1)**

> **作者:** Whye Kit Fong; Venice Erin Liong; Kok Seang Tan; Holger Caesar
>
> **备注:** 18 pages, 17 figures
>
> **摘要:** Autonomous Vehicles (AV) and Advanced Driver Assistance Systems (ADAS) have been revolutionized by Deep Learning. As a data-driven approach, Deep Learning relies on vast amounts of driving data, typically labeled in great detail. As a result, datasets, alongside hardware and algorithms, are foundational building blocks for the development of AVs. In this work we revisit one of the most widely used autonomous driving datasets: the nuScenes dataset. nuScenes exemplifies key trends in AV development, being the first dataset to include radar data, to feature diverse urban driving scenes from two continents, and to be collected using a fully autonomous vehicle operating on public roads, while also promoting multi-modal sensor fusion, standardized benchmarks, and a broad range of tasks including perception, localization \& mapping, prediction and planning. We provide an unprecedented look into the creation of nuScenes, as well as its extensions nuImages and Panoptic nuScenes, summarizing many technical details that have hitherto not been revealed in academic publications. Furthermore, we trace how the influence of nuScenes impacted a large number of other datasets that were released later and how it defined numerous standards that are used by the community to this day. Finally, we present an overview of both official and unofficial tasks using the nuScenes dataset and review major methodological developments, thereby offering a comprehensive survey of the autonomous driving literature, with a particular focus on nuScenes.
>
---
#### [new 051] Beyond Paired Data: Self-Supervised UAV Geo-Localization from Reference Imagery Alone
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究无人机在无GNSS环境下的图像定位任务。针对现有方法依赖昂贵配对数据的问题，提出仅用卫星参考图训练的自监督方法，通过模拟视角差异的增强策略，构建高效模型CAEVL，实现无需真实无人机图像即可有效定位，显著降低数据依赖并提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.02737v1](https://arxiv.org/pdf/2512.02737v1)**

> **作者:** Tristan Amadei; Enric Meinhardt-Llopis; Benedicte Bascle; Corentin Abgrall; Gabriele Facciolo
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Image-based localization in GNSS-denied environments is critical for UAV autonomy. Existing state-of-the-art approaches rely on matching UAV images to geo-referenced satellite images; however, they typically require large-scale, paired UAV-satellite datasets for training. Such data are costly to acquire and often unavailable, limiting their applicability. To address this challenge, we adopt a training paradigm that removes the need for UAV imagery during training by learning directly from satellite-view reference images. This is achieved through a dedicated augmentation strategy that simulates the visual domain shift between satellite and real-world UAV views. We introduce CAEVL, an efficient model designed to exploit this paradigm, and validate it on ViLD, a new and challenging dataset of real-world UAV images that we release to the community. Our method achieves competitive performance compared to approaches trained with paired data, demonstrating its effectiveness and strong generalization capabilities.
>
---
#### [new 052] BEVDilation: LiDAR-Centric Multi-Modal Fusion for 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D目标检测中多模态融合因传感器几何精度差异导致性能下降的问题，提出LiDAR-centric的BEVDilation框架。通过图像特征作为隐式引导，缓解深度估计误差带来的空间错位，并利用图像先验增强点云稀疏性与语义信息，提升检测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.02972v1](https://arxiv.org/pdf/2512.02972v1)**

> **作者:** Guowen Zhang; Chenhang He; Liyi Chen; Lei Zhang
>
> **备注:** Accept by AAAI26
>
> **摘要:** Integrating LiDAR and camera information in the bird's eye view (BEV) representation has demonstrated its effectiveness in 3D object detection. However, because of the fundamental disparity in geometric accuracy between these sensors, indiscriminate fusion in previous methods often leads to degraded performance. In this paper, we propose BEVDilation, a novel LiDAR-centric framework that prioritizes LiDAR information in the fusion. By formulating image BEV features as implicit guidance rather than naive concatenation, our strategy effectively alleviates the spatial misalignment caused by image depth estimation errors. Furthermore, the image guidance can effectively help the LiDAR-centric paradigm to address the sparsity and semantic limitations of point clouds. Specifically, we propose a Sparse Voxel Dilation Block that mitigates the inherent point sparsity by densifying foreground voxels through image priors. Moreover, we introduce a Semantic-Guided BEV Dilation Block to enhance the LiDAR feature diffusion processing with image semantic guidance and long-range context capture. On the challenging nuScenes benchmark, BEVDilation achieves better performance than state-of-the-art methods while maintaining competitive computational efficiency. Importantly, our LiDAR-centric strategy demonstrates greater robustness to depth noise compared to naive fusion. The source code is available at https://github.com/gwenzhang/BEVDilation.
>
---
#### [new 053] Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉相似物体追踪难题，提出基于预训练视频扩散模型的自监督追踪方法。利用扩散模型在去噪早期阶段分离运动信息的特性，提取鲁棒运动表示，无需标注数据即可有效区分外观相似物体，在复杂场景下显著提升追踪性能。**

- **链接: [https://arxiv.org/pdf/2512.02339v1](https://arxiv.org/pdf/2512.02339v1)**

> **作者:** Chenshuang Zhang; Kang Zhang; Joon Son Chung; In So Kweon; Junmo Kim; Chengzhi Mao
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Distinguishing visually similar objects by their motion remains a critical challenge in computer vision. Although supervised trackers show promise, contemporary self-supervised trackers struggle when visual cues become ambiguous, limiting their scalability and generalization without extensive labeled data. We find that pre-trained video diffusion models inherently learn motion representations suitable for tracking without task-specific training. This ability arises because their denoising process isolates motion in early, high-noise stages, distinct from later appearance refinement. Capitalizing on this discovery, our self-supervised tracker significantly improves performance in distinguishing visually similar objects, an underexplored failure point for existing methods. Our method achieves up to a 6-point improvement over recent self-supervised approaches on established benchmarks and our newly introduced tests focused on tracking visually similar items. Visualizations confirm that these diffusion-derived motion representations enable robust tracking of even identical objects across challenging viewpoint changes and deformations.
>
---
#### [new 054] RFOP: Rethinking Fusion and Orthogonal Projection for Face-Voice Association
- **分类: cs.CV**

- **简介: 该论文针对多语言环境下人脸与语音关联任务，提出RFOP方法，通过重构融合与正交投影机制，增强跨模态语义对齐。旨在提升跨语言人脸-语音匹配准确率，解决多语言场景下模态差异带来的匹配难题，在FAME 2026挑战中以33.1%的EER排名第三。**

- **链接: [https://arxiv.org/pdf/2512.02860v1](https://arxiv.org/pdf/2512.02860v1)**

> **作者:** Abdul Hannan; Furqan Malik; Hina Jabbar; Syed Suleman Sadiq; Mubashir Noman
>
> **备注:** Ranked 3rd in Fame 2026 Challenge, ICASSP
>
> **摘要:** Face-voice association in multilingual environment challenge 2026 aims to investigate the face-voice association task in multilingual scenario. The challenge introduces English-German face-voice pairs to be utilized in the evaluation phase. To this end, we revisit the fusion and orthogonal projection for face-voice association by effectively focusing on the relevant semantic information within the two modalities. Our method performs favorably on the English-German data split and ranked 3rd in the FAME 2026 challenge by achieving the EER of 33.1.
>
---
#### [new 055] Skywork-R1V4: Toward Agentic Multimodal Intelligence through Interleaved Thinking with Images and DeepResearch
- **分类: cs.CV**

- **简介: 该论文提出Skywork-R1V4，一个30B参数的多模态智能体模型，旨在解决现有系统中图像操作与搜索割裂、依赖强化学习、缺乏真实执行规划等问题。通过监督微调实现视觉操作与外部知识检索的交错推理，统一多模态规划与深度搜索，在多项基准上达到领先性能，证明了无需强化学习即可实现复杂多步任务的长程推理。**

- **链接: [https://arxiv.org/pdf/2512.02395v1](https://arxiv.org/pdf/2512.02395v1)**

> **作者:** Yifan Zhang; Liang Hu; Haofeng Sun; Peiyu Wang; Yichen Wei; Shukang Yin; Jiangbo Pei; Wei Shen; Peng Xia; Yi Peng; Tianyidan Xie; Eric Li; Yang Liu; Xuchen Song; Yahui Zhou
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** Despite recent progress in multimodal agentic systems, existing approaches often treat image manipulation and web search as disjoint capabilities, rely heavily on costly reinforcement learning, and lack planning grounded in real tool-execution traces. To address these limitations, we present Skywork-R1V4, a 30B (A3B) parameter multimodal agentic model that unifies multimodal planning, active image manipulation ("thinking with images"), deep multimodal search, and, most critically, interleaved reasoning that dynamically alternates between visual operations and external knowledge retrieval. Trained solely via supervised fine-tuning on fewer than 30,000 high-quality, planning-execution-consistent trajectories and validated through stepwise consistency filtering, Skywork-R1V4 achieves state-of-the-art results across perception and multimodal search benchmarks: it scores 66.1 on MMSearch and 67.2 on FVQA, surpassing Gemini 2.5 Flash on all 11 metrics. Skywork-R1V4 exhibits emergent long-horizon reasoning at inference time, successfully orchestrating more than 10 tool calls to solve complex, multi-step tasks. Our results demonstrate that sophisticated agentic multimodal intelligence can be achieved through carefully curated supervised learning alone, without any reliance on reinforcement learning.
>
---
#### [new 056] PhyCustom: Towards Realistic Physical Customization in Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文针对文本生成图像中的物理概念定制难题，提出PhyCustom框架。通过引入等距正则化与解耦损失，显式融入物理知识，提升模型对物理属性（如重力、弹性）的准确生成能力。实验表明，该方法在定量与定性上均优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.02794v1](https://arxiv.org/pdf/2512.02794v1)**

> **作者:** Fan Wu; Cheng Chen; Zhoujie Fu; Jiacheng Wei; Yi Xu; Deheng Ye; Guosheng Lin
>
> **备注:** codes:https://github.com/wufan-cse/PhyCustom
>
> **摘要:** Recent diffusion-based text-to-image customization methods have achieved significant success in understanding concrete concepts to control generation processes, such as styles and shapes. However, few efforts dive into the realistic yet challenging customization of physical concepts. The core limitation of current methods arises from the absence of explicitly introducing physical knowledge during training. Even when physics-related words appear in the input text prompts, our experiments consistently demonstrate that these methods fail to accurately reflect the corresponding physical properties in the generated results. In this paper, we propose PhyCustom, a fine-tuning framework comprising two novel regularization losses to activate diffusion model to perform physical customization. Specifically, the proposed isometric loss aims at activating diffusion models to learn physical concepts while decouple loss helps to eliminate the mixture learning of independent concepts. Experiments are conducted on a diverse dataset and our benchmark results demonstrate that PhyCustom outperforms previous state-of-the-art and popular methods in terms of physical customization quantitatively and qualitatively.
>
---
#### [new 057] AttMetNet: Attention-Enhanced Deep Neural Network for Methane Plume Detection in Sentinel-2 Satellite Imagery
- **分类: cs.CV**

- **简介: 该论文提出AttMetNet，用于卫星影像中甲烷羽流检测任务。针对传统方法误报率高、深度学习模型忽略甲烷特征的问题，融合NDMI与注意力增强U-Net，提升特征选择能力，并采用焦点损失缓解样本不平衡，实现更精准的甲烷羽流识别。**

- **链接: [https://arxiv.org/pdf/2512.02751v1](https://arxiv.org/pdf/2512.02751v1)**

> **作者:** Rakib Ahsan; MD Sadik Hossain Shanto; Md Sultanul Arifin; Tanzima Hashem
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Methane is a powerful greenhouse gas that contributes significantly to global warming. Accurate detection of methane emissions is the key to taking timely action and minimizing their impact on climate change. We present AttMetNet, a novel attention-enhanced deep learning framework for methane plume detection with Sentinel-2 satellite imagery. The major challenge in developing a methane detection model is to accurately identify methane plumes from Sentinel-2's B11 and B12 bands while suppressing false positives caused by background variability and diverse land cover types. Traditional detection methods typically depend on the differences or ratios between these bands when comparing the scenes with and without plumes. However, these methods often require verification by a domain expert because they generate numerous false positives. Recent deep learning methods make some improvements using CNN-based architectures, but lack mechanisms to prioritize methane-specific features. AttMetNet introduces a methane-aware architecture that fuses the Normalized Difference Methane Index (NDMI) with an attention-enhanced U-Net. By jointly exploiting NDMI's plume-sensitive cues and attention-driven feature selection, AttMetNet selectively amplifies methane absorption features while suppressing background noise. This integration establishes a first-of-its-kind architecture tailored for robust methane plume detection in real satellite imagery. Additionally, we employ focal loss to address the severe class imbalance arising from both limited positive plume samples and sparse plume pixels within imagery. Furthermore, AttMetNet is trained on the real methane plume dataset, making it more robust to practical scenarios. Extensive experiments show that AttMetNet surpasses recent methods in methane plume detection with a lower false positive rate, better precision recall balance, and higher IoU.
>
---
#### [new 058] In-Context Sync-LoRA for Portrait Video Editing
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文针对人物视频编辑任务，解决如何在修改外观、表情或添加物体时保持时间同步与身份一致性的难题。提出Sync-LoRA方法，通过在上下文内训练LoRA，利用同步配对视频学习运动与视觉变化的分离，实现高质量、帧级精准的视频编辑。**

- **链接: [https://arxiv.org/pdf/2512.03013v1](https://arxiv.org/pdf/2512.03013v1)**

> **作者:** Sagi Polaczek; Or Patashnik; Ali Mahdavi-Amiri; Daniel Cohen-Or
>
> **备注:** Project page: https://sagipolaczek.github.io/Sync-LoRA/
>
> **摘要:** Editing portrait videos is a challenging task that requires flexible yet precise control over a wide range of modifications, such as appearance changes, expression edits, or the addition of objects. The key difficulty lies in preserving the subject's original temporal behavior, demanding that every edited frame remains precisely synchronized with the corresponding source frame. We present Sync-LoRA, a method for editing portrait videos that achieves high-quality visual modifications while maintaining frame-accurate synchronization and identity consistency. Our approach uses an image-to-video diffusion model, where the edit is defined by modifying the first frame and then propagated to the entire sequence. To enable accurate synchronization, we train an in-context LoRA using paired videos that depict identical motion trajectories but differ in appearance. These pairs are automatically generated and curated through a synchronization-based filtering process that selects only the most temporally aligned examples for training. This training setup teaches the model to combine motion cues from the source video with the visual changes introduced in the edited first frame. Trained on a compact, highly curated set of synchronized human portraits, Sync-LoRA generalizes to unseen identities and diverse edits (e.g., modifying appearance, adding objects, or changing backgrounds), robustly handling variations in pose and expression. Our results demonstrate high visual fidelity and strong temporal coherence, achieving a robust balance between edit fidelity and precise motion preservation.
>
---
#### [new 059] SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文针对3D高斯点云渲染中多视角一致性问题，提出SplatSuRe方法。通过分析相机位姿与场景几何关系，仅在低频信息不足区域选择性应用超分辨率，避免全局统一增强导致的视图不一致，显著提升局部细节清晰度与整体渲染质量。**

- **链接: [https://arxiv.org/pdf/2512.02172v1](https://arxiv.org/pdf/2512.02172v1)**

> **作者:** Pranav Asthana; Alex Hanson; Allen Tu; Tom Goldstein; Matthias Zwicker; Amitabh Varshney
>
> **备注:** Project Page: https://splatsure.github.io/
>
> **摘要:** 3D Gaussian Splatting (3DGS) enables high-quality novel view synthesis, motivating interest in generating higher-resolution renders than those available during training. A natural strategy is to apply super-resolution (SR) to low-resolution (LR) input views, but independently enhancing each image introduces multi-view inconsistencies, leading to blurry renders. Prior methods attempt to mitigate these inconsistencies through learned neural components, temporally consistent video priors, or joint optimization on LR and SR views, but all uniformly apply SR across every image. In contrast, our key insight is that close-up LR views may contain high-frequency information for regions also captured in more distant views, and that we can use the camera pose relative to scene geometry to inform where to add SR content. Building from this insight, we propose SplatSuRe, a method that selectively applies SR content only in undersampled regions lacking high-frequency supervision, yielding sharper and more consistent results. Across Tanks & Temples, Deep Blending and Mip-NeRF 360, our approach surpasses baselines in both fidelity and perceptual quality. Notably, our gains are most significant in localized foreground regions where higher detail is desired.
>
---
#### [new 060] UAUTrack: Towards Unified Multimodal Anti-UAV Visual Tracking
- **分类: cs.CV**

- **简介: 该论文针对多模态反无人机视觉跟踪任务，解决现有方法缺乏统一框架、跨模态信息融合不足的问题。提出UAUTrack，一种单流端到端的统一跟踪框架，引入文本先验提示策略，有效融合RGB、TIR等多模态数据，实现高精度与高效能的无人机追踪。**

- **链接: [https://arxiv.org/pdf/2512.02668v1](https://arxiv.org/pdf/2512.02668v1)**

> **作者:** Qionglin Ren; Dawei Zhang; Chunxu Tian; Dan Zhang
>
> **摘要:** Research in Anti-UAV (Unmanned Aerial Vehicle) tracking has explored various modalities, including RGB, TIR, and RGB-T fusion. However, a unified framework for cross-modal collaboration is still lacking. Existing approaches have primarily focused on independent models for individual tasks, often overlooking the potential for cross-modal information sharing. Furthermore, Anti-UAV tracking techniques are still in their infancy, with current solutions struggling to achieve effective multimodal data fusion. To address these challenges, we propose UAUTrack, a unified single-target tracking framework built upon a single-stream, single-stage, end-to-end architecture that effectively integrates multiple modalities. UAUTrack introduces a key component: a text prior prompt strategy that directs the model to focus on UAVs across various scenarios. Experimental results show that UAUTrack achieves state-of-the-art performance on the Anti-UAV and DUT Anti-UAV datasets, and maintains a favourable trade-off between accuracy and speed on the Anti-UAV410 dataset, demonstrating both high accuracy and practical efficiency across diverse Anti-UAV scenarios.
>
---
#### [new 061] Contextual Image Attack: How Visual Context Exposes Multimodal Safety Vulnerabilities
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文针对多模态大模型的安全漏洞，提出图像中心的越狱攻击方法CIA。通过多智能体系统在视觉上下文中隐匿有害指令，结合四种可视化策略与上下文增强技术，显著提升攻击成功率，实验证明其对GPT-4o和Qwen2.5-VL-72B的攻击成功率达91.07%，有效揭示了视觉模态的潜在安全风险。**

- **链接: [https://arxiv.org/pdf/2512.02973v1](https://arxiv.org/pdf/2512.02973v1)**

> **作者:** Yuan Xiong; Ziqi Miao; Lijun Li; Chen Qian; Jie Li; Jing Shao
>
> **摘要:** While Multimodal Large Language Models (MLLMs) show remarkable capabilities, their safety alignments are susceptible to jailbreak attacks. Existing attack methods typically focus on text-image interplay, treating the visual modality as a secondary prompt. This approach underutilizes the unique potential of images to carry complex, contextual information. To address this gap, we propose a new image-centric attack method, Contextual Image Attack (CIA), which employs a multi-agent system to subtly embeds harmful queries into seemingly benign visual contexts using four distinct visualization strategies. To further enhance the attack's efficacy, the system incorporate contextual element enhancement and automatic toxicity obfuscation techniques. Experimental results on the MMSafetyBench-tiny dataset show that CIA achieves high toxicity scores of 4.73 and 4.83 against the GPT-4o and Qwen2.5-VL-72B models, respectively, with Attack Success Rates (ASR) reaching 86.31\% and 91.07\%. Our method significantly outperforms prior work, demonstrating that the visual modality itself is a potent vector for jailbreaking advanced MLLMs.
>
---
#### [new 062] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks
- **分类: cs.CV**

- **简介: 该论文针对视频中基于图像的模型出现的时间不一致问题，提出通用稳定性适配器。通过设计统一的精度-稳定性-鲁棒性损失函数，实现对多种视觉任务（如去噪、增强、深度估计、分割）的稳定推理，显著提升视频序列的时序一致性与抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2512.03014v1](https://arxiv.org/pdf/2512.03014v1)**

> **作者:** Matthew Dutson; Nathan Labiosa; Yin Li; Mohit Gupta
>
> **备注:** NeurIPS 2025
>
> **摘要:** When applied sequentially to video, frame-based networks often exhibit temporal inconsistency - for example, outputs that flicker between frames. This problem is amplified when the network inputs contain time-varying corruptions. In this work, we introduce a general approach for adapting frame-based models for stable and robust inference on video. We describe a class of stability adapters that can be inserted into virtually any architecture and a resource-efficient training process that can be performed with a frozen base network. We introduce a unified conceptual framework for describing temporal stability and corruption robustness, centered on a proposed accuracy-stability-robustness loss. By analyzing the theoretical properties of this loss, we identify the conditions where it produces well-behaved stabilizer training. Our experiments validate our approach on several vision tasks including denoising (NAFNet), image enhancement (HDRNet), monocular depth (Depth Anything v2), and semantic segmentation (DeepLabv3+). Our method improves temporal stability and robustness against a range of image corruptions (including compression artifacts, noise, and adverse weather), while preserving or improving the quality of predictions.
>
---
#### [new 063] Video4Spatial: Towards Visuospatial Intelligence with Context-Guided Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Video4Spatial框架，探索视频生成模型在仅依赖视觉数据下的空间智能。针对场景导航与物体定位任务，通过视频上下文实现端到端的空间规划与语义定位，无需深度或姿态等辅助信息，展现了强空间理解与泛化能力，推动视频生成模型向通用视觉空间推理发展。**

- **链接: [https://arxiv.org/pdf/2512.03040v1](https://arxiv.org/pdf/2512.03040v1)**

> **作者:** Zeqi Xiao; Yiwei Zhao; Lingxiao Li; Yushi Lan; Yu Ning; Rahul Garg; Roshni Cooper; Mohammad H. Taghavi; Xingang Pan
>
> **备注:** Project page at https://xizaoqu.github.io/video4spatial/
>
> **摘要:** We investigate whether video generative models can exhibit visuospatial intelligence, a capability central to human cognition, using only visual data. To this end, we present Video4Spatial, a framework showing that video diffusion models conditioned solely on video-based scene context can perform complex spatial tasks. We validate on two tasks: scene navigation - following camera-pose instructions while remaining consistent with 3D geometry of the scene, and object grounding - which requires semantic localization, instruction following, and planning. Both tasks use video-only inputs, without auxiliary modalities such as depth or poses. With simple yet effective design choices in the framework and data curation, Video4Spatial demonstrates strong spatial understanding from video context: it plans navigation and grounds target objects end-to-end, follows camera-pose instructions while maintaining spatial consistency, and generalizes to long contexts and out-of-domain environments. Taken together, these results advance video generative models toward general visuospatial reasoning.
>
---
#### [new 064] MAViD: A Multimodal Framework for Audio-Visual Dialogue Understanding and Generation
- **分类: cs.CV**

- **简介: 该论文提出MAViD框架，解决音频-视觉对话理解与生成任务中多模态融合难、长时序一致性差的问题。通过“指挥-创作”架构，分离语义理解与交互生成，并结合自回归与扩散模型，实现高质量音视频同步生成，提升对话自然性与连贯性。**

- **链接: [https://arxiv.org/pdf/2512.03034v1](https://arxiv.org/pdf/2512.03034v1)**

> **作者:** Youxin Pang; Jiajun Liu; Lingfeng Tan; Yong Zhang; Feng Gao; Xiang Deng; Zhuoliang Kang; Xiaoming Wei; Yebin Liu
>
> **备注:** Our project website is https://carlyx.github.io/MAViD/
>
> **摘要:** We propose MAViD, a novel Multimodal framework for Audio-Visual Dialogue understanding and generation. Existing approaches primarily focus on non-interactive systems and are limited to producing constrained and unnatural human speech.The primary challenge of this task lies in effectively integrating understanding and generation capabilities, as well as achieving seamless multimodal audio-video fusion. To solve these problems, we propose a Conductor-Creator architecture that divides the dialogue system into two primary components.The Conductor is tasked with understanding, reasoning, and generating instructions by breaking them down into motion and speech components, thereby enabling fine-grained control over interactions. The Creator then delivers interactive responses based on these instructions.Furthermore, to address the difficulty of generating long videos with consistent identity, timbre, and tone using dual DiT structures, the Creator adopts a structure that combines autoregressive (AR) and diffusion models. The AR model is responsible for audio generation, while the diffusion model ensures high-quality video generation.Additionally, we propose a novel fusion module to enhance connections between contextually consecutive clips and modalities, enabling synchronized long-duration audio-visual content generation.Extensive experiments demonstrate that our framework can generate vivid and contextually coherent long-duration dialogue interactions and accurately interpret users' multimodal queries.
>
---
#### [new 065] Hear What Matters! Text-conditioned Selective Video-to-Audio Generation
- **分类: cs.CV; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出文本条件选择性视频转音频（V2A）任务，旨在从多物体视频中分离出用户指定的声音。针对现有方法无法精准定位声源的问题，提出SelVA模型，通过文本提示引导视频编码器提取相关特征，并利用补充令牌增强跨注意力，实现语义与时间上的精准对齐。**

- **链接: [https://arxiv.org/pdf/2512.02650v1](https://arxiv.org/pdf/2512.02650v1)**

> **作者:** Junwon Lee; Juhan Nam; Jiyoung Lee
>
> **摘要:** This work introduces a new task, text-conditioned selective video-to-audio (V2A) generation, which produces only the user-intended sound from a multi-object video. This capability is especially crucial in multimedia production, where audio tracks are handled individually for each sound source for precise editing, mixing, and creative control. However, current approaches generate single source-mixed sounds at once, largely because visual features are entangled, and region cues or prompts often fail to specify the source. We propose SelVA, a novel text-conditioned V2A model that treats the text prompt as an explicit selector of target source and modulates video encoder to distinctly extract prompt-relevant video features. The proposed supplementary tokens promote cross-attention by suppressing text-irrelevant activations with efficient parameter tuning, yielding robust semantic and temporal grounding. SelVA further employs a self-augmentation scheme to overcome the lack of mono audio track supervision. We evaluate SelVA on VGG-MONOAUDIO, a curated benchmark of clean single-source videos for such a task. Extensive experiments and ablations consistently verify its effectiveness across audio quality, semantic alignment, and temporal synchronization. Code and demo are available at https://jnwnlee.github.io/selva-demo/.
>
---
#### [new 066] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D布局估计任务，解决现有模型难以处理真实世界多层复杂建筑的问题。提出HouseLayout3D基准数据集与无需训练的MultiFloor3D基线方法，支持全建筑尺度布局估计，显著提升对多楼层结构的建模能力。**

- **链接: [https://arxiv.org/pdf/2512.02450v1](https://arxiv.org/pdf/2512.02450v1)**

> **作者:** Valentin Bieri; Marie-Julie Rakotosaona; Keisuke Tateno; Francis Engelmann; Leonidas Guibas
>
> **备注:** NeurIPS 2025 (Datasets and Benchmarks Track) Project Page: https://houselayout3d.github.io
>
> **摘要:** Current 3D layout estimation models are primarily trained on synthetic datasets containing simple single room or single floor environments. As a consequence, they cannot natively handle large multi floor buildings and require scenes to be split into individual floors before processing, which removes global spatial context that is essential for reasoning about structures such as staircases that connect multiple levels. In this work, we introduce HouseLayout3D, a real world benchmark designed to support progress toward full building scale layout estimation, including multiple floors and architecturally intricate spaces. We also present MultiFloor3D, a simple training free baseline that leverages recent scene understanding methods and already outperforms existing 3D layout estimation models on both our benchmark and prior datasets, highlighting the need for further research in this direction. Data and code are available at: https://houselayout3d.github.io.
>
---
#### [new 067] Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对少样本与测试时适应任务，提出BOLT框架。通过提取任务相关正交基，在冻结基底的前提下仅训练少量对角系数，实现高效参数化迁移。解决了预训练模型在数据与计算资源受限下难以有效迁移的问题，提供强初始化与轻量微调路径。**

- **链接: [https://arxiv.org/pdf/2512.02441v1](https://arxiv.org/pdf/2512.02441v1)**

> **作者:** Junghwan Park; Woojin Cho; Junhyuk Heo; Darongsae Kwon; Kookjin Lee
>
> **摘要:** Adapting large pre-trained models to unseen tasks under tight data and compute budgets remains challenging. Meta-learning approaches explicitly learn good initializations, but they require an additional meta-training phase over many tasks, incur high training cost, and can be unstable. At the same time, the number of task-specific pre-trained models continues to grow, yet the question of how to transfer them to new tasks with minimal additional training remains relatively underexplored. We propose BOLT (Basis-Oriented Low-rank Transfer), a framework that reuses existing fine-tuned models not by merging weights, but instead by extracting an orthogonal, task-informed spectral basis and adapting within that subspace. In the offline phase, BOLT collects dominant singular directions from multiple task vectors and orthogonalizes them per layer to form reusable bases. In the online phase, we freeze these bases and train only a small set of diagonal coefficients per layer for the new task, yielding a rank-controlled update with very few trainable parameters. This design provides (i) a strong, training-free initialization for unseen tasks, obtained by pooling source-task coefficients, along with a lightweight rescaling step while leveraging the shared orthogonal bases, and (ii) a parameter-efficient fine-tuning (PEFT) path that, in our experiments, achieves robust performance compared to common PEFT baselines as well as a representative meta-learned initialization. Our results show that constraining adaptation to a task-informed orthogonal subspace provides an effective alternative for unseen-task transfer.
>
---
#### [new 068] Leveraging AI multimodal geospatial foundation models for improved near-real-time flood mapping at a global scale
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像分割任务，旨在提升全球范围近实时洪水制图精度。针对现有模型依赖标注数据、泛化能力不足的问题，研究利用多模态卫星数据（SAR与光学）对地球空间基础模型TerraMind进行微调，评估不同配置在85个洪灾事件上的表现，验证了融合多源数据与微调策略的有效性。**

- **链接: [https://arxiv.org/pdf/2512.02055v1](https://arxiv.org/pdf/2512.02055v1)**

> **作者:** Mirela G. Tulbure; Julio Caineta; Mark Broich; Mollie D. Gaines; Philippe Rufin; Leon-Friedrich Thomas; Hamed Alemohammad; Jan Hemmerling; Patrick Hostert
>
> **摘要:** Floods are among the most damaging weather-related hazards, and in 2024, the warmest year on record, extreme flood events affected communities across five continents. Earth observation (EO) satellites provide critical, frequent coverage for mapping inundation, yet operational accuracy depends heavily on labeled datasets and model generalization. Recent Geospatial Foundation Models (GFMs), such as ESA-IBM's TerraMind, offer improved generalizability through large-scale self-supervised pretraining, but their performance on diverse global flood events remains poorly understood. We fine-tune TerraMind for flood extent mapping using FloodsNet, a harmonized multimodal dataset containing co-located Sentinel-1 (Synthetic Aperture Radar, SAR data) and Sentinel-2 (optical) imagery for 85 flood events worldwide. We tested four configurations (base vs. large models; frozen vs. unfrozen backbones) and compared against the TerraMind Sen1Floods11 example and a U-Net trained on both FloodsNet and Sen1Floods11. The base-unfrozen configuration provided the best balance of accuracy, precision, and recall at substantially lower computational cost than the large model. The large unfrozen model achieved the highest recall. Models trained on FloodsNet outperformed the Sen1Floods11-trained example in recall with similar overall accuracy. U-Net achieved higher recall than all GFM configurations, though with slightly lower accuracy and precision. Our results demonstrate that integrating multimodal optical and SAR data and fine-tuning a GFM can enhance near-real-time flood mapping. This study provides one of the first global-scale evaluations of a GFM for flood segmentation, highlighting both its potential and current limitations for climate adaptation and disaster resilience.
>
---
#### [new 069] Progressive Image Restoration via Text-Conditioned Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文将文本到视频模型CogVideo用于图像修复任务，通过构建合成数据集并微调模型生成从退化到清晰的渐进恢复序列。解决了传统图像修复缺乏时序一致性的难题，实现了超分辨率、去模糊和低光增强的高质量恢复，并在真实场景中展现零样本泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.02273v1](https://arxiv.org/pdf/2512.02273v1)**

> **作者:** Peng Kang; Xijun Wang; Yu Yuan
>
> **备注:** First two authors contributed equally to this work. IEEE ICNC Accepted
>
> **摘要:** Recent text-to-video models have demonstrated strong temporal generation capabilities, yet their potential for image restoration remains underexplored. In this work, we repurpose CogVideo for progressive visual restoration tasks by fine-tuning it to generate restoration trajectories rather than natural video motion. Specifically, we construct synthetic datasets for super-resolution, deblurring, and low-light enhancement, where each sample depicts a gradual transition from degraded to clean frames. Two prompting strategies are compared: a uniform text prompt shared across all samples, and a scene-specific prompting scheme generated via LLaVA multi-modal LLM and refined with ChatGPT. Our fine-tuned model learns to associate temporal progression with restoration quality, producing sequences that improve perceptual metrics such as PSNR, SSIM, and LPIPS across frames. Extensive experiments show that CogVideo effectively restores spatial detail and illumination consistency while maintaining temporal coherence. Moreover, the model generalizes to real-world scenarios on the ReLoBlur dataset without additional training, demonstrating strong zero-shot robustness and interpretability through temporal restoration.
>
---
#### [new 070] Generalizing Vision-Language Models with Dedicated Prompt Guidance
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型微调中的领域泛化问题，提出GuiDG框架。通过分域提示微调构建专家模型，并利用跨模态注意力实现专家自适应融合，提升模型在未见领域的泛化能力。实验验证其在标准与新构建的ImageNet-DG数据集上的有效性。**

- **链接: [https://arxiv.org/pdf/2512.02421v1](https://arxiv.org/pdf/2512.02421v1)**

> **作者:** Xinyao Li; Yinjie Min; Hongbo Chen; Zhekai Du; Fengling Li; Jingjing Li
>
> **备注:** Accepted to AAAI26
>
> **摘要:** Fine-tuning large pretrained vision-language models (VLMs) has emerged as a prevalent paradigm for downstream adaptation, yet it faces a critical trade-off between domain specificity and domain generalization (DG) ability. Current methods typically fine-tune a universal model on the entire dataset, which potentially compromises the ability to generalize to unseen domains. To fill this gap, we provide a theoretical understanding of the generalization ability for VLM fine-tuning, which reveals that training multiple parameter-efficient expert models on partitioned source domains leads to better generalization than fine-tuning a universal model. Inspired by this finding, we propose a two-step domain-expert-Guided DG (GuiDG) framework. GuiDG first employs prompt tuning to obtain source domain experts, then introduces a Cross-Modal Attention module to guide the fine-tuning of the vision encoder via adaptive expert integration. To better evaluate few-shot DG, we construct ImageNet-DG from ImageNet and its variants. Extensive experiments on standard DG benchmarks and ImageNet-DG demonstrate that GuiDG improves upon state-of-the-art fine-tuning methods while maintaining efficiency.
>
---
#### [new 071] U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦于自动驾驶与具身AI中的4D LiDAR世界建模任务，针对现有方法忽略场景不确定性导致生成伪影的问题，提出U4D框架。通过分割模型生成不确定性图，分“难到易”两阶段建模，并引入时空混合块增强时序一致性，实现更真实、稳定的4D环境重建。**

- **链接: [https://arxiv.org/pdf/2512.02982v1](https://arxiv.org/pdf/2512.02982v1)**

> **作者:** Xiang Xu; Ao Liang; Youquan Liu; Linfeng Li; Lingdong Kong; Ziwei Liu; Qingshan Liu
>
> **备注:** Preprint; 19 pages, 7 figures, 8 tables
>
> **摘要:** Modeling dynamic 3D environments from LiDAR sequences is central to building reliable 4D worlds for autonomous driving and embodied AI. Existing generative frameworks, however, often treat all spatial regions uniformly, overlooking the varying uncertainty across real-world scenes. This uniform generation leads to artifacts in complex or ambiguous regions, limiting realism and temporal stability. In this work, we present U4D, an uncertainty-aware framework for 4D LiDAR world modeling. Our approach first estimates spatial uncertainty maps from a pretrained segmentation model to localize semantically challenging regions. It then performs generation in a "hard-to-easy" manner through two sequential stages: (1) uncertainty-region modeling, which reconstructs high-entropy regions with fine geometric fidelity, and (2) uncertainty-conditioned completion, which synthesizes the remaining areas under learned structural priors. To further ensure temporal coherence, U4D incorporates a mixture of spatio-temporal (MoST) block that adaptively fuses spatial and temporal representations during diffusion. Extensive experiments show that U4D produces geometrically faithful and temporally consistent LiDAR sequences, advancing the reliability of 4D world modeling for autonomous perception and simulation.
>
---
#### [new 072] FineGRAIN: Evaluating Failure Modes of Text-to-Image Models with Vision Language Model Judges
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成模型（T2I）在属性准确性上的系统性错误，提出FineGRAIN框架，通过27种特定失败模式评估T2I与视觉语言模型（VLM）的性能。构建了包含5个T2I模型生成图像及VLM标注的数据集，揭示现有评估指标的不足，推动生成模型可靠性与可解释性提升。**

- **链接: [https://arxiv.org/pdf/2512.02161v1](https://arxiv.org/pdf/2512.02161v1)**

> **作者:** Kevin David Hayes; Micah Goldblum; Vikash Sehwag; Gowthami Somepalli; Ashwinee Panda; Tom Goldstein
>
> **备注:** Accepted to NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Text-to-image (T2I) models are capable of generating visually impressive images, yet they often fail to accurately capture specific attributes in user prompts, such as the correct number of objects with the specified colors. The diversity of such errors underscores the need for a hierarchical evaluation framework that can compare prompt adherence abilities of different image generation models. Simultaneously, benchmarks of vision language models (VLMs) have not kept pace with the complexity of scenes that VLMs are used to annotate. In this work, we propose a structured methodology for jointly evaluating T2I models and VLMs by testing whether VLMs can identify 27 specific failure modes in the images generated by T2I models conditioned on challenging prompts. Our second contribution is a dataset of prompts and images generated by 5 T2I models (Flux, SD3-Medium, SD3-Large, SD3.5-Medium, SD3.5-Large) and the corresponding annotations from VLMs (Molmo, InternVL3, Pixtral) annotated by an LLM (Llama3) to test whether VLMs correctly identify the failure mode in a generated image. By analyzing failure modes on a curated set of prompts, we reveal systematic errors in attribute fidelity and object representation. Our findings suggest that current metrics are insufficient to capture these nuanced errors, highlighting the importance of targeted benchmarks for advancing generative model reliability and interpretability.
>
---
#### [new 073] DGGT: Feedforward 4D Reconstruction of Dynamic Driving Scenes using Unposed Images
- **分类: cs.CV**

- **简介: 该论文提出DGGT，一种无需相机位姿的前馈式动态驾驶场景4D重建方法。针对现有方法依赖标定、逐场景优化和短时序的问题，通过将位姿作为模型输出，实现从稀疏无位姿图像中联合预测3D高斯分布、相机参数与动态信息，支持长序列、多视角输入，显著提升重建速度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.03004v1](https://arxiv.org/pdf/2512.03004v1)**

> **作者:** Xiaoxue Chen; Ziyi Xiong; Yuantao Chen; Gen Li; Nan Wang; Hongcheng Luo; Long Chen; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye; Hongyang Li; Ya-Qin Zhang; Hao Zhao
>
> **摘要:** Autonomous driving needs fast, scalable 4D reconstruction and re-simulation for training and evaluation, yet most methods for dynamic driving scenes still rely on per-scene optimization, known camera calibration, or short frame windows, making them slow and impractical. We revisit this problem from a feedforward perspective and introduce \textbf{Driving Gaussian Grounded Transformer (DGGT)}, a unified framework for pose-free dynamic scene reconstruction. We note that the existing formulations, treating camera pose as a required input, limit flexibility and scalability. Instead, we reformulate pose as an output of the model, enabling reconstruction directly from sparse, unposed images and supporting an arbitrary number of views for long sequences. Our approach jointly predicts per-frame 3D Gaussian maps and camera parameters, disentangles dynamics with a lightweight dynamic head, and preserves temporal consistency with a lifespan head that modulates visibility over time. A diffusion-based rendering refinement further reduces motion/interpolation artifacts and improves novel-view quality under sparse inputs. The result is a single-pass, pose-free algorithm that achieves state-of-the-art performance and speed. Trained and evaluated on large-scale driving benchmarks (Waymo, nuScenes, Argoverse2), our method outperforms prior work both when trained on each dataset and in zero-shot transfer across datasets, and it scales well as the number of input frames increases.
>
---
#### [new 074] Reasoning-Aware Multimodal Fusion for Hateful Video Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对在线视频中的仇恨言论检测任务，解决多模态语义融合不足与微妙仇恨意图理解困难的问题。提出RAMF框架，通过局部-全局上下文融合与语义交叉注意力增强多模态交互，并引入对抗性推理机制，生成客观描述、假设性推断等多视角语义，提升模型对复杂仇恨内容的识别能力。**

- **链接: [https://arxiv.org/pdf/2512.02743v1](https://arxiv.org/pdf/2512.02743v1)**

> **作者:** Shuonan Yang; Tailin Chen; Jiangbei Yue; Guangliang Cheng; Jianbo Jiao; Zeyu Fu
>
> **摘要:** Hate speech in online videos is posing an increasingly serious threat to digital platforms, especially as video content becomes increasingly multimodal and context-dependent. Existing methods often struggle to effectively fuse the complex semantic relationships between modalities and lack the ability to understand nuanced hateful content. To address these issues, we propose an innovative Reasoning-Aware Multimodal Fusion (RAMF) framework. To tackle the first challenge, we design Local-Global Context Fusion (LGCF) to capture both local salient cues and global temporal structures, and propose Semantic Cross Attention (SCA) to enable fine-grained multimodal semantic interaction. To tackle the second challenge, we introduce adversarial reasoning-a structured three-stage process where a vision-language model generates (i) objective descriptions, (ii) hate-assumed inferences, and (iii) non-hate-assumed inferences-providing complementary semantic perspectives that enrich the model's contextual understanding of nuanced hateful intent. Evaluations on two real-world hateful video datasets demonstrate that our method achieves robust generalisation performance, improving upon state-of-the-art methods by 3% and 7% in Macro-F1 and hate class recall, respectively. We will release the code after the anonymity period ends.
>
---
#### [new 075] A multi-weight self-matching visual explanation for cnns on sar images
- **分类: cs.CV**

- **简介: 该论文针对SAR图像分类中CNN模型缺乏可解释性的问题，提出多权重自匹配类激活映射（MS-CAM）方法。通过融合通道与元素级权重，精准可视化模型决策依据，提升解释性，并验证其在弱监督目标定位中的可行性，分析关键影响因素。**

- **链接: [https://arxiv.org/pdf/2512.02344v1](https://arxiv.org/pdf/2512.02344v1)**

> **作者:** Siyuan Sun; Yongping Zhang; Hongcheng Zeng; Yamin Wang; Wei Yang; Wanting Yang; Jie Chen
>
> **摘要:** In recent years, convolutional neural networks (CNNs) have achieved significant success in various synthetic aperture radar (SAR) tasks. However, the complexity and opacity of their internal mechanisms hinder the fulfillment of high-reliability requirements, thereby limiting their application in SAR. Improving the interpretability of CNNs is thus of great importance for their development and deployment in SAR. In this paper, a visual explanation method termed multi-weight self-matching class activation mapping (MS-CAM) is proposed. MS-CAM matches SAR images with the feature maps and corresponding gradients extracted by the CNN, and combines both channel-wise and element-wise weights to visualize the decision basis learned by the model in SAR images. Extensive experiments conducted on a self-constructed SAR target classification dataset demonstrate that MS-CAM more accurately highlights the network's regions of interest and captures detailed target feature information, thereby enhancing network interpretability. Furthermore, the feasibility of applying MS-CAM to weakly-supervised obiect localization is validated. Key factors affecting localization accuracy, such as pixel thresholds, are analyzed in depth to inform future work.
>
---
#### [new 076] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视觉语言模型（VLM）中视觉令牌冗余与空间覆盖不足的问题，提出无需训练的VLM-Pruner方法。通过中心发散式剪枝与空间稀疏缓冲机制，平衡冗余去除与关键区域保留，结合并行贪心策略与信息融合，实现高效剪枝，在88.9%剪枝率下保持性能并提升推理速度。**

- **链接: [https://arxiv.org/pdf/2512.02700v1](https://arxiv.org/pdf/2512.02700v1)**

> **作者:** Zhenkai Wu; Xiaowen Ma; Zhenliang Ni; Dengming Zhang; Han Shu; Xin Jiang; Xinghao Chen
>
> **摘要:** Vision-language models (VLMs) excel at image understanding tasks, but the large number of visual tokens imposes significant computational costs, hindering deployment on mobile devices. Many pruning methods rely solely on token importance and thus overlook inter-token redundancy, retaining numerous duplicated tokens and wasting capacity. Although some redundancy-aware approaches have been proposed, they often ignore the spatial relationships among visual tokens. This can lead to overly sparse selections of retained tokens that fail to adequately cover the regions of target objects. To address these limitations, we propose VLM-Pruner, a training-free token pruning algorithm that explicitly balances redundancy and spatial sparsity. We introduce a centrifugal token pruning paradigm that enables near-to-far selection while prioritizing the preservation of fine-grained object details. Moreover, we design a Buffering for Spatial Sparsity (BSS) criterion that defers the selection of spatially distant tokens. We further adopt a parallel greedy strategy to conduct token selection efficiently. To mitigate information loss from pruning, we selectively fuse salient information from the discarded tokens into the retained ones. Comprehensive comparisons demonstrate that VLM-Pruner consistently outperforms strong baselines across five VLMs with an 88.9\% pruning rate, while delivering an end-to-end inference speedup.
>
---
#### [new 077] SAGE: Style-Adaptive Generalization for Privacy-Constrained Semantic Segmentation Across Domains
- **分类: cs.CV**

- **简介: 该论文针对隐私约束下的跨域语义分割任务，提出SAGE框架。解决冻结模型在域迁移中性能下降问题，通过学习动态视觉提示自适应融合风格特征，提升模型泛化能力，无需修改模型参数。**

- **链接: [https://arxiv.org/pdf/2512.02369v1](https://arxiv.org/pdf/2512.02369v1)**

> **作者:** Qingmei Li; Yang Zhang; Peifeng Zhang; Haohuan Fu; Juepeng Zheng
>
> **摘要:** Domain generalization for semantic segmentation aims to mitigate the degradation in model performance caused by domain shifts. However, in many real-world scenarios, we are unable to access the model parameters and architectural details due to privacy concerns and security constraints. Traditional fine-tuning or adaptation is hindered, leading to the demand for input-level strategies that can enhance generalization without modifying model weights. To this end, we propose a \textbf{S}tyle-\textbf{A}daptive \textbf{GE}neralization framework (\textbf{SAGE}), which improves the generalization of frozen models under privacy constraints. SAGE learns to synthesize visual prompts that implicitly align feature distributions across styles instead of directly fine-tuning the backbone. Specifically, we first utilize style transfer to construct a diverse style representation of the source domain, thereby learning a set of style characteristics that can cover a wide range of visual features. Then, the model adaptively fuses these style cues according to the visual context of each input, forming a dynamic prompt that harmonizes the image appearance without touching the interior of the model. Through this closed-loop design, SAGE effectively bridges the gap between frozen model invariance and the diversity of unseen domains. Extensive experiments on five benchmark datasets demonstrate that SAGE achieves competitive or superior performance compared to state-of-the-art methods under privacy constraints and outperforms full fine-tuning baselines in all settings.
>
---
#### [new 078] SkyMoE: A Vision-Language Foundation Model for Enhancing Geospatial Interpretation with Mixture of Experts
- **分类: cs.CV**

- **简介: 该论文针对遥感图像多任务、多粒度解释中通用模型表现不佳的问题，提出SkyMoE模型。通过引入自适应路由的专家混合架构与上下文解耦增强策略，实现任务与粒度感知的精细化处理，提升局部细节与全局上下文理解能力，并构建了MGRS-Bench基准进行评估。**

- **链接: [https://arxiv.org/pdf/2512.02517v1](https://arxiv.org/pdf/2512.02517v1)**

> **作者:** Jiaqi Liu; Ronghao Fu; Lang Sun; Haoran Liu; Xiao Yang; Weipeng Zhang; Xu Na; Zhuoran Duan; Bo Yang
>
> **摘要:** The emergence of large vision-language models (VLMs) has significantly enhanced the efficiency and flexibility of geospatial interpretation. However, general-purpose VLMs remain suboptimal for remote sensing (RS) tasks. Existing geospatial VLMs typically adopt a unified modeling strategy and struggle to differentiate between task types and interpretation granularities, limiting their ability to balance local detail perception and global contextual understanding. In this paper, we present SkyMoE, a Mixture-of-Experts (MoE) vision-language model tailored for multimodal, multi-task RS interpretation. SkyMoE employs an adaptive router that generates task- and granularity-aware routing instructions, enabling specialized large language model experts to handle diverse sub-tasks. To further promote expert decoupling and granularity sensitivity, we introduce a context-disentangled augmentation strategy that creates contrastive pairs between local and global features, guiding experts toward level-specific representation learning. We also construct MGRS-Bench, a comprehensive benchmark covering multiple RS interpretation tasks and granularity levels, to evaluate generalization in complex scenarios. Extensive experiments on 21 public datasets demonstrate that SkyMoE achieves state-of-the-art performance across tasks, validating its adaptability, scalability, and superior multi-granularity understanding in remote sensing.
>
---
#### [new 079] Polar Perspectives: Evaluating 2-D LiDAR Projections for Robust Place Recognition with Visual Foundation Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究LiDAR点云投影对视觉基础模型进行地点识别的影响，旨在提升2D投影在复杂环境下的鲁棒性与判别力。通过构建可控的检索流程，系统评估不同投影方式性能，验证了优化投影可有效替代端到端3D学习，适用于实时自主系统。**

- **链接: [https://arxiv.org/pdf/2512.02897v1](https://arxiv.org/pdf/2512.02897v1)**

> **作者:** Pierpaolo Serio; Giulio Pisaneschi; Andrea Dan Ryals; Vincenzo Infantino; Lorenzo Gentilini; Valentina Donzella; Lorenzo Pollini
>
> **备注:** 13 Pages, 5 Figures, 2 Tables Under Review
>
> **摘要:** This work presents a systematic investigation into how alternative LiDAR-to-image projections affect metric place recognition when coupled with a state-of-the-art vision foundation model. We introduce a modular retrieval pipeline that controls for backbone, aggregation, and evaluation protocol, thereby isolating the influence of the 2-D projection itself. Using consistent geometric and structural channels across multiple datasets and deployment scenarios, we identify the projection characteristics that most strongly determine discriminative power, robustness to environmental variation, and suitability for real-time autonomy. Experiments with different datasets, including integration into an operational place recognition policy, validate the practical relevance of these findings and demonstrate that carefully designed projections can serve as an effective surrogate for end-to-end 3-D learning in LiDAR place recognition.
>
---
#### [new 080] SurfFill: Completion of LiDAR Point Clouds via Gaussian Surfel Splatting
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出SurfFill，一种基于高斯表面元的LiDAR点云补全方法。针对LiDAR在细结构和暗材质处漏采的问题，利用点云密度变化识别模糊区域，结合高斯表面元优化，在局部区域生成补充点，实现高质量补全。适用于大尺度场景，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.03010v1](https://arxiv.org/pdf/2512.03010v1)**

> **作者:** Svenja Strobel; Matthias Innmann; Bernhard Egger; Marc Stamminger; Linus Franke
>
> **备注:** Project page: https://lfranke.github.io/surffill
>
> **摘要:** LiDAR-captured point clouds are often considered the gold standard in active 3D reconstruction. While their accuracy is exceptional in flat regions, the capturing is susceptible to miss small geometric structures and may fail with dark, absorbent materials. Alternatively, capturing multiple photos of the scene and applying 3D photogrammetry can infer these details as they often represent feature-rich regions. However, the accuracy of LiDAR for featureless regions is rarely reached. Therefore, we suggest combining the strengths of LiDAR and camera-based capture by introducing SurfFill: a Gaussian surfel-based LiDAR completion scheme. We analyze LiDAR capturings and attribute LiDAR beam divergence as a main factor for artifacts, manifesting mostly at thin structures and edges. We use this insight to introduce an ambiguity heuristic for completed scans by evaluating the change in density in the point cloud. This allows us to identify points close to missed areas, which we can then use to grow additional points from to complete the scan. For this point growing, we constrain Gaussian surfel reconstruction [Huang et al. 2024] to focus optimization and densification on these ambiguous areas. Finally, Gaussian primitives of the reconstruction in ambiguous areas are extracted and sampled for points to complete the point cloud. To address the challenges of large-scale reconstruction, we extend this pipeline with a divide-and-conquer scheme for building-sized point cloud completion. We evaluate on the task of LiDAR point cloud completion of synthetic and real-world scenes and find that our method outperforms previous reconstruction methods.
>
---
#### [new 081] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution
- **分类: cs.CV**

- **简介: 该论文提出TGDD，一种轨迹引导的数据集蒸馏方法，旨在解决传统分布匹配方法忽略训练过程中特征演变的问题。通过动态对齐训练轨迹中的特征分布，并引入分布约束正则化，提升合成数据的语义多样性和代表性，显著改善下游任务性能，在10个数据集上达到最优效果。**

- **链接: [https://arxiv.org/pdf/2512.02469v1](https://arxiv.org/pdf/2512.02469v1)**

> **作者:** Fengli Ran; Xiao Pu; Bo Liu; Xiuli Bi; Bin Xiao
>
> **备注:** Accepted in AAAI 2026
>
> **摘要:** Dataset distillation compresses large datasets into compact synthetic ones to reduce storage and computational costs. Among various approaches, distribution matching (DM)-based methods have attracted attention for their high efficiency. However, they often overlook the evolution of feature representations during training, which limits the expressiveness of synthetic data and weakens downstream performance. To address this issue, we propose Trajectory Guided Dataset Distillation (TGDD), which reformulates distribution matching as a dynamic alignment process along the model's training trajectory. At each training stage, TGDD captures evolving semantics by aligning the feature distribution between the synthetic and original dataset. Meanwhile, it introduces a distribution constraint regularization to reduce class overlap. This design helps synthetic data preserve both semantic diversity and representativeness, improving performance in downstream tasks. Without additional optimization overhead, TGDD achieves a favorable balance between performance and efficiency. Experiments on ten datasets demonstrate that TGDD achieves state-of-the-art performance, notably a 5.0% accuracy gain on high-resolution benchmarks.
>
---
#### [new 082] RULER-Bench: Probing Rule-based Reasoning Abilities of Next-level Video Generation Models for Vision Foundation Intelligence
- **分类: cs.CV**

- **简介: 该论文针对视频生成模型缺乏规则推理能力评估的问题，提出RULER-Bench基准。聚焦文本到视频与图像到视频任务，涵盖6类规则的40项任务，通过多维度指标与GPT-o3评分，揭示当前模型在规则一致性上仅达48.87%，推动视频生成向视觉基础智能演进。**

- **链接: [https://arxiv.org/pdf/2512.02622v1](https://arxiv.org/pdf/2512.02622v1)**

> **作者:** Xuming He; Zehao Fan; Hengjia Li; Fan Zhuo; Hankun Xu; Senlin Cheng; Di Weng; Haifeng Liu; Can Ye; Boxi Wu
>
> **摘要:** Recent advances in video generation have enabled the synthesis of videos with strong temporal consistency and impressive visual quality, marking a crucial step toward vision foundation models. To evaluate these video generation models, existing benchmarks primarily focus on factors related to visual perception and understanding, like visual aesthetics, instruction adherence, and temporal coherence. However, the rule-based reasoning capabilities of video generation models remain largely unexplored. Although recent studies have carried out preliminary explorations into whether video models can serve as zero-shot learners, they still lack a fine-grained decomposition of reasoning capabilities and a comprehensive evaluation protocol. To address this gap, we introduce RULER-Bench, a benchmark designed to evaluate the reasoning ability of video generation models from the perspective of cognitive rules. Built upon two fundamental paradigms: text-to-video and image-to-video, RULER-Bench covers 40 representative tasks spanning six rule categories with 622 high-quality annotated instances. For the evaluation of each generated video, we construct a checklist covering four metrics and leverage GPT-o3 to assign scores to each question, achieving 85% alignment with human judgements. Extensive experiments show that the state-of-the-art model achieves only 48.87% on the rule coherence metric, highlighting significant room for improvement in the reasoning capability of next-level video models. We expect that the insight obtained from RULER-Bench will facilitate further development of reasoning-aware video generation, advancing video generation models toward vision foundation intelligence.
>
---
#### [new 083] Defense That Attacks: How Robust Models Become Better Attackers
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究对抗训练对攻击转移性的影响，发现防御性模型反而生成更易转移的对抗样本，揭示了“防御即攻击”的悖论。通过训练36个模型并系统实验，提出应评估模型生成可转移攻击的能力，推动更全面的鲁棒性评测。**

- **链接: [https://arxiv.org/pdf/2512.02830v1](https://arxiv.org/pdf/2512.02830v1)**

> **作者:** Mohamed Awad; Mahmoud Akrm; Walid Gomaa
>
> **摘要:** Deep learning has achieved great success in computer vision, but remains vulnerable to adversarial attacks. Adversarial training is the leading defense designed to improve model robustness. However, its effect on the transferability of attacks is underexplored. In this work, we ask whether adversarial training unintentionally increases the transferability of adversarial examples. To answer this, we trained a diverse zoo of 36 models, including CNNs and ViTs, and conducted comprehensive transferability experiments. Our results reveal a clear paradox: adversarially trained (AT) models produce perturbations that transfer more effectively than those from standard models, which introduce a new ecosystem risk. To enable reproducibility and further study, we release all models, code, and experimental scripts. Furthermore, we argue that robustness evaluations should assess not only the resistance of a model to transferred attacks but also its propensity to produce transferable adversarial examples.
>
---
#### [new 084] CAMEO: Correspondence-Attention Alignment for Multi-View Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对多视角扩散模型的视图一致性问题，发现其注意力机制可学习几何对应关系但精度随视角变化下降。提出CAMEO方法，通过监督单个注意力层的几何对应，提升训练效率与生成质量，加速收敛并改善新视图合成效果，且适用于任意多视角扩散模型。**

- **链接: [https://arxiv.org/pdf/2512.03045v1](https://arxiv.org/pdf/2512.03045v1)**

> **作者:** Minkyung Kwon; Jinhyeok Choi; Jiho Park; Seonghu Jeon; Jinhyuk Jang; Junyoung Seo; Minseop Kwak; Jin-Hwa Kim; Seungryong Kim
>
> **备注:** Project page: https://cvlab-kaist.github.io/CAMEO/
>
> **摘要:** Multi-view diffusion models have recently emerged as a powerful paradigm for novel view synthesis, yet the underlying mechanism that enables their view-consistency remains unclear. In this work, we first verify that the attention maps of these models acquire geometric correspondence throughout training, attending to the geometrically corresponding regions across reference and target views for view-consistent generation. However, this correspondence signal remains incomplete, with its accuracy degrading under large viewpoint changes. Building on these findings, we introduce CAMEO, a simple yet effective training technique that directly supervises attention maps using geometric correspondence to enhance both the training efficiency and generation quality of multi-view diffusion models. Notably, supervising a single attention layer is sufficient to guide the model toward learning precise correspondences, thereby preserving the geometry and structure of reference images, accelerating convergence, and improving novel view synthesis performance. CAMEO reduces the number of training iterations required for convergence by half while achieving superior performance at the same iteration counts. We further demonstrate that CAMEO is model-agnostic and can be applied to any multi-view diffusion model.
>
---
#### [new 085] TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking
- **分类: cs.CV**

- **简介: 该论文针对快速运动小目标跟踪任务，解决现有方法在遮挡和运动方向歧义上的缺陷。提出TrackNetV5，引入运动方向解耦模块（MDD）恢复方向信息，并设计残差驱动时空精修头（R-STR），实现粗到细的精准跟踪。实验表明其性能显著提升，且计算开销可控。**

- **链接: [https://arxiv.org/pdf/2512.02789v1](https://arxiv.org/pdf/2512.02789v1)**

> **作者:** Tang Haonan; Chen Yanjun; Jiang Lezhi
>
> **摘要:** The TrackNet series has established a strong baseline for fast-moving small object tracking in sports. However, existing iterations face significant limitations: V1-V3 struggle with occlusions due to a reliance on purely visual cues, while TrackNetV4, despite introducing motion inputs, suffers from directional ambiguity as its absolute difference method discards motion polarity. To overcome these bottlenecks, we propose TrackNetV5, a robust architecture integrating two novel mechanisms. First, to recover lost directional priors, we introduce the Motion Direction Decoupling (MDD) module. Unlike V4, MDD decomposes temporal dynamics into signed polarity fields, explicitly encoding both movement occurrence and trajectory direction. Second, we propose the Residual-Driven Spatio-Temporal Refinement (R-STR) head. Operating on a coarse-to-fine paradigm, this Transformer-based module leverages factorized spatio-temporal contexts to estimate a corrective residual, effectively recovering occluded targets. Extensive experiments on the TrackNetV2 dataset demonstrate that TrackNetV5 achieves a new state-of-the-art F1-score of 0.9859 and an accuracy of 0.9733, significantly outperforming previous versions. Notably, this performance leap is achieved with a marginal 3.7% increase in FLOPs compared to V4, maintaining real-time inference capabilities while delivering superior tracking precision.
>
---
#### [new 086] MitUNet: Enhancing Floor Plan Recognition using a Hybrid Mix-Transformer and U-Net Architecture
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D建模中2D平面图墙段分割精度不足的问题，提出MitUNet混合架构。通过分层Mix-Transformer捕捉全局上下文，结合带scSE注意力的U-Net decoder恢复精细边界，并采用Tversky损失优化，提升对细墙结构的识别与边界准确性，显著改善了分割质量。**

- **链接: [https://arxiv.org/pdf/2512.02413v1](https://arxiv.org/pdf/2512.02413v1)**

> **作者:** Dmitriy Parashchuk; Alexey Kapshitskiy; Yuriy Karyakin
>
> **备注:** 9 pages, 4 figures, 3 tables
>
> **摘要:** Automatic 3D reconstruction of indoor spaces from 2D floor plans requires high-precision semantic segmentation of structural elements, particularly walls. However, existing methods optimized for standard metrics often struggle to detect thin structural components and yield masks with irregular boundaries, lacking the geometric precision required for subsequent vectorization. To address this issue, we introduce MitUNet, a hybrid neural network architecture specifically designed for wall segmentation tasks in the context of 3D modeling. In MitUNet, we utilize a hierarchical Mix-Transformer encoder to capture global context and a U-Net decoder enhanced with scSE attention blocks for precise boundary recovery. Furthermore, we propose an optimization strategy based on the Tversky loss function to effectively balance precision and recall. By fine-tuning the hyperparameters of the loss function, we prioritize the suppression of false positive noise along wall boundaries while maintaining high sensitivity to thin structures. Our experiments on the public CubiCasa5k dataset and a proprietary regional dataset demonstrate that the proposed approach ensures the generation of structurally correct masks with high boundary accuracy, outperforming standard single-task models. MitUNet provides a robust tool for data preparation in automated 3D reconstruction pipelines.
>
---
#### [new 087] Enhancing Cross Domain SAR Oil Spill Segmentation via Morphological Region Perturbation and Synthetic Label-to-SAR Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对跨区域SAR油污分割模型泛化能力差的问题，提出MORP–Synth框架。通过形态学区域扰动生成真实几何变化，并利用条件生成模型合成SAR纹理，增强数据多样性。在秘鲁海域数据上验证，显著提升模型性能，尤其改善了少数类分割效果。**

- **链接: [https://arxiv.org/pdf/2512.02290v1](https://arxiv.org/pdf/2512.02290v1)**

> **作者:** Andre Juarez; Luis Salsavilca; Frida Coaquira; Celso Gonzales
>
> **摘要:** Deep learning models for SAR oil spill segmentation often fail to generalize across regions due to differences in sea-state, backscatter statistics, and slick morphology, a limitation that is particularly severe along the Peruvian coast where labeled Sentinel-1 data remain scarce. To address this problem, we propose \textbf{MORP--Synth}, a two-stage synthetic augmentation framework designed to improve transfer from Mediterranean to Peruvian conditions. Stage~A applies Morphological Region Perturbation, a curvature guided label space method that generates realistic geometric variations of oil and look-alike regions. Stage~B renders SAR-like textures from the edited masks using a conditional generative INADE model. We compile a Peruvian dataset of 2112 labeled 512$\times$512 patches from 40 Sentinel-1 scenes (2014--2024), harmonized with the Mediterranean CleanSeaNet benchmark, and evaluate seven segmentation architectures. Models pretrained on Mediterranean data degrade from 67.8\% to 51.8\% mIoU on the Peruvian domain; MORP--Synth improves performance up to +6 mIoU and boosts minority-class IoU (+10.8 oil, +14.6 look-alike).
>
---
#### [new 088] PPTBench: Towards Holistic Evaluation of Large Language Models for PowerPoint Layout and Design Understanding
- **分类: cs.CV**

- **简介: 该论文提出PPTBench，一个面向演示文稿布局与设计理解的多模态评估基准，旨在解决现有评测忽视视觉布局结构的问题。通过4,439个样本覆盖检测、理解、修改与生成四类任务，揭示当前大模型在空间布局推理上的显著不足，推动视觉-结构推理与连贯幻灯片生成研究。**

- **链接: [https://arxiv.org/pdf/2512.02624v1](https://arxiv.org/pdf/2512.02624v1)**

> **作者:** Zheng Huang; Xukai Liu; Tianyu Hu; Kai Zhang; Ye Liu
>
> **摘要:** PowerPoint presentations combine rich textual content with structured visual layouts, making them a natural testbed for evaluating the multimodal reasoning and layout understanding abilities of modern MLLMs. However, existing benchmarks focus solely on narrow subtasks while overlooking layout-centric challenges, which are central to real-world slide creation and editing. To bridge this gap, we introduce PPTBench, a comprehensive multimodal benchmark for evaluating LLMs on PowerPoint-related tasks. Leveraging a diverse source of 958 PPTX files, PPTBench evaluates models across four categories with 4,439 samples, including Detection, Understanding, Modification, and Generation. Our experiments reveal a substantial gap between semantic understanding and visual-layout reasoning in current MLLMs: models can interpret slide content but fail to produce coherent spatial arrangements. Ablation and further analysis show that current MLLMs struggle to combine visual cues with JSON-based layout structures and fail to integrate visual information into their API planning ability. And case studies visually expose systematic layout errors such as misalignment and element overlap. These findings provides a new perspective on evaluating VLLMs in PPT scenarios, highlighting challenges and directions for future research on visual-structural reasoning and coherent slide generation. All datasets and code are fully released to support reproducibility and future research.
>
---
#### [new 089] UCAgents: Unidirectional Convergence for Visual Evidence Anchored Multi-Agent Medical Decision-Making
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医学视觉问答任务，解决VLMs推理脱离图像证据的问题。提出UCAgents框架，通过单轮结构化证据审计实现多代理协同，抑制文本噪声与视觉模糊，提升诊断可靠性与计算效率。**

- **链接: [https://arxiv.org/pdf/2512.02485v1](https://arxiv.org/pdf/2512.02485v1)**

> **作者:** Qianhan Feng; Zhongzhen Huang; Yakun Zhu; Xiaofan Zhang; Qi Dou
>
> **摘要:** Vision-Language Models (VLMs) show promise in medical diagnosis, yet suffer from reasoning detachment, where linguistically fluent explanations drift from verifiable image evidence, undermining clinical trust. Recent multi-agent frameworks simulate Multidisciplinary Team (MDT) debates to mitigate single-model bias, but open-ended discussions amplify textual noise and computational cost while failing to anchor reasoning to visual evidence, the cornerstone of medical decision-making. We propose UCAgents, a hierarchical multi-agent framework enforcing unidirectional convergence through structured evidence auditing. Inspired by clinical workflows, UCAgents forbids position changes and limits agent interactions to targeted evidence verification, suppressing rhetorical drift while amplifying visual signal extraction. In UCAgents, a one-round inquiry discussion is introduced to uncover potential risks of visual-textual misalignment. This design jointly constrains visual ambiguity and textual noise, a dual-noise bottleneck that we formalize via information theory. Extensive experiments on four medical VQA benchmarks show UCAgents achieves superior accuracy (71.3% on PathVQA, +6.0% over state-of-the-art) with 87.7% lower token cost, the evaluation results further confirm that UCAgents strikes a balance between uncovering more visual evidence and avoiding confusing textual interference. These results demonstrate that UCAgents exhibits both diagnostic reliability and computational efficiency critical for real-world clinical deployment. Code is available at https://github.com/fqhank/UCAgents.
>
---
#### [new 090] YingVideo-MV: Music-Driven Multi-Stage Video Generation
- **分类: cs.CV**

- **简介: 该论文提出YingVideo-MV，首个音乐驱动的多阶段长视频生成框架，解决音乐表演视频中相机运动控制与长序列一致性难题。通过音频语义分析、可解释的镜头规划、时序感知扩散Transformer及动态窗口策略，实现高质量、同步的音乐-动作-相机协同生成。**

- **链接: [https://arxiv.org/pdf/2512.02492v1](https://arxiv.org/pdf/2512.02492v1)**

> **作者:** Jiahui Chen; Weida Wang; Runhua Shi; Huan Yang; Chaofan Ding; Zihao Chen
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** While diffusion model for audio-driven avatar video generation have achieved notable process in synthesizing long sequences with natural audio-visual synchronization and identity consistency, the generation of music-performance videos with camera motions remains largely unexplored. We present YingVideo-MV, the first cascaded framework for music-driven long-video generation. Our approach integrates audio semantic analysis, an interpretable shot planning module (MV-Director), temporal-aware diffusion Transformer architectures, and long-sequence consistency modeling to enable automatic synthesis of high-quality music performance videos from audio signals. We construct a large-scale Music-in-the-Wild Dataset by collecting web data to support the achievement of diverse, high-quality results. Observing that existing long-video generation methods lack explicit camera motion control, we introduce a camera adapter module that embeds camera poses into latent noise. To enhance continulity between clips during long-sequence inference, we further propose a time-aware dynamic window range strategy that adaptively adjust denoising ranges based on audio embedding. Comprehensive benchmark tests demonstrate that YingVideo-MV achieves outstanding performance in generating coherent and expressive music videos, and enables precise music-motion-camera synchronization. More videos are available in our project page: https://giantailab.github.io/YingVideo-MV/ .
>
---
#### [new 091] MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文针对高分辨率图像理解任务，解决因图像分块导致目标对象断裂、语义相似性计算偏差的问题。提出无需训练的多分辨率检索-检测融合框架MRD，通过多分辨率语义融合保持对象完整性，并利用开放词汇检测模型实现全局目标定位，显著提升理解准确率。**

- **链接: [https://arxiv.org/pdf/2512.02906v1](https://arxiv.org/pdf/2512.02906v1)**

> **作者:** Fan Yang; Kaihao Zhang
>
> **摘要:** Understanding high-resolution images remains a significant challenge for multimodal large language models (MLLMs). Recent study address this issue by dividing the image into smaller crops and computing the semantic similarity between each crop and a query using a pretrained retrieval-augmented generation (RAG) model. The most relevant crops are then selected to localize the target object and suppress irrelevant information. However, such crop-based processing can fragment complete objects across multiple crops, thereby disrupting the computation of semantic similarity. In our experiments, we find that image crops of objects with different sizes are better handled at different resolutions. Based on this observation, we propose Multi-resolution Retrieval-Detection (MRD), a training-free framework for high-resolution image understanding. To address the issue of semantic similarity bias caused by objects being split across different image crops, we propose a multi-resolution semantic fusion method, which integrates semantic similarity maps obtained at different resolutions to produce more accurate semantic information and preserve the integrity of target objects. Furthermore, to achieve direct localization of target objects at a global scale, we introduce an open-vocalbulary object detection (OVD) model that identifies object regions using a sliding-window approach.Experiments on high-resolution image understanding benchmarks using different MLLMs demonstrate the effectiveness of our approach.
>
---
#### [new 092] LoVoRA: Text-guided and Mask-free Video Object Removal and Addition with Learnable Object-aware Localization
- **分类: cs.CV**

- **简介: 该论文针对文本引导的视频对象移除与添加任务，解决现有方法依赖掩码或参考图导致可扩展性差的问题。提出LoVoRA框架，通过可学习的对象感知定位机制与扩散掩码预测器，实现无需外部控制信号的端到端视频编辑，提升时空一致性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.02933v1](https://arxiv.org/pdf/2512.02933v1)**

> **作者:** Zhihan Xiao; Lin Liu; Yixin Gao; Xiaopeng Zhang; Haoxuan Che; Songping Mai; Qi Tian
>
> **摘要:** Text-guided video editing, particularly for object removal and addition, remains a challenging task due to the need for precise spatial and temporal consistency. Existing methods often rely on auxiliary masks or reference images for editing guidance, which limits their scalability and generalization. To address these issues, we propose LoVoRA, a novel framework for mask-free video object removal and addition using object-aware localization mechanism. Our approach utilizes a unique dataset construction pipeline that integrates image-to-video translation, optical flow-based mask propagation, and video inpainting, enabling temporally consistent edits. The core innovation of LoVoRA is its learnable object-aware localization mechanism, which provides dense spatio-temporal supervision for both object insertion and removal tasks. By leveraging a Diffusion Mask Predictor, LoVoRA achieves end-to-end video editing without requiring external control signals during inference. Extensive experiments and human evaluation demonstrate the effectiveness and high-quality performance of LoVoRA.
>
---
#### [new 093] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对多模态大模型在语音理解上的不足，提出AV-SpeakerBench基准，聚焦说话人中心的音视频联合推理。通过精细化标注与融合式问题设计，评估模型对“谁说、说什么、何时说”的细粒度理解能力。实验表明Gemini系列领先，凸显音频视觉融合的关键作用。**

- **链接: [https://arxiv.org/pdf/2512.02231v1](https://arxiv.org/pdf/2512.02231v1)**

> **作者:** Le Thien Phuc Nguyen; Zhuoran Yu; Samuel Low Yu Hang; Subin An; Jeongik Lee; Yohan Ban; SeungEun Chung; Thanh-Huy Nguyen; JuWan Maeng; Soochahn Lee; Yong Jae Lee
>
> **备注:** preprint
>
> **摘要:** Multimodal large language models (MLLMs) are expected to jointly interpret vision, audio, and language, yet existing video benchmarks rarely assess fine-grained reasoning about human speech. Many tasks remain visually solvable or only coarsely evaluate speech, offering limited insight into whether models can align who speaks, what is said, and when it occurs. We introduce AV-SpeakerBench, a curated benchmark of 3,212 multiple-choice questions focused on speaker-centric audiovisual reasoning in real-world videos. It features: (1) a speaker-centered formulation that treats speakers-not scenes-as the core reasoning unit; (2) fusion-grounded question design embedding audiovisual dependencies into question semantics; and (3) expert-curated annotations ensuring temporal precision and cross-modal validity. Comprehensive evaluations show that the Gemini family consistently outperforms open-source systems, with Gemini 2.5 Pro achieving the best results. Among open models, Qwen3-Omni-30B approaches Gemini 2.0 Flash but remains far behind Gemini 2.5 Pro, primarily due to weaker audiovisual fusion rather than visual perception. We believe AV-SpeakerBench establishes a rigorous foundation for advancing fine-grained audiovisual reasoning in future multimodal systems.
>
---
#### [new 094] See, Think, Learn: A Self-Taught Multimodal Reasoner
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉语言模型在多模态推理中感知与推理能力不足的问题，提出自训练框架See-Think-Learn（STL）。通过结构化推理模板先提取视觉属性再推理，并引入负向解释增强判别力，实现感知与推理的联合优化，显著提升多模态推理性能。**

- **链接: [https://arxiv.org/pdf/2512.02456v1](https://arxiv.org/pdf/2512.02456v1)**

> **作者:** Sourabh Sharma; Sonam Gupta; Sadbhawna
>
> **备注:** Winter Conference on Applications of Computer Vision 2026
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable progress in integrating visual perception with language understanding. However, effective multimodal reasoning requires both accurate perception and robust reasoning, and weakness in either limits the performance of VLMs. Prior efforts to enhance reasoning often depend on high-quality chain-of-thought (CoT) data, obtained via labor-intensive human annotations, costly proprietary models, or self-training methods that overlook perception. To address these limitations, we propose a simple yet effective self-training framework called See-Think-Learn (STL). At its core, STL introduces a structured reasoning template that encourages the model to see before thinking, first extracting visual attributes in textual form, then using them to guide reasoning. The framework jointly improves perception and reasoning by having the model generate and learn from its own structured rationales in a self-training loop. Furthermore, we augment the training data with negative rationales, i.e. explanations that justify why certain answer choices are incorrect, to enhance the model's ability to distinguish between correct and misleading responses. This fosters more discriminative and robust learning. Experiments across diverse domains show that STL consistently outperforms baselines trained directly only on answers or self-generated reasoning, while qualitative analysis confirms the high quality of its rationales. STL thus provides a cost-effective solution to enhance multimodal reasoning ability of VLMs.
>
---
#### [new 095] PPTArena: A Benchmark for Agentic PowerPoint Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PPTArena基准，用于评估自然语言指令下的幻灯片精准编辑能力。针对现有方法在复杂编辑任务中表现不佳的问题，构建了包含真实幻灯片编辑的评测体系，并提出PPTPilot代理，通过结构感知规划与迭代验证，显著提升编辑准确性与视觉一致性，但仍面临长周期文档级任务挑战。**

- **链接: [https://arxiv.org/pdf/2512.03042v1](https://arxiv.org/pdf/2512.03042v1)**

> **作者:** Michael Ofengenden; Yunze Man; Ziqi Pang; Yu-Xiong Wang
>
> **备注:** 25 pages, 26 figures
>
> **摘要:** We introduce PPTArena, a benchmark for PowerPoint editing that measures reliable modifications to real slides under natural-language instructions. In contrast to image-PDF renderings or text-to-slide generation, PPTArena focuses on in-place editing across 100 decks, 2125 slides, and over 800 targeted edits covering text, charts, tables, animations, and master-level styles. Each case includes a ground-truth deck, a fully specified target outcome, and a dual VLM-as-judge pipeline that separately scores instruction following and visual quality using both structural diffs and slide images. Building on this setting, we propose PPTPilot, a structure-aware slide-editing agent that plans semantic edit sequences, routes between high-level programmatic tools and deterministic XML operations for precise control, and verifies outputs through an iterative plan-edit-check loop against task-specific constraints. In our experiments, PPTPilot outperforms strong proprietary agents and frontier VLM systems by over 10 percentage points on compound, layout-sensitive, and cross-slide edits, with particularly large gains in visual fidelity and deck-wide consistency. Despite these improvements, existing agents still underperform on long-horizon, document-scale tasks in PPTArena, highlighting the remaining challenges in reliable PPT editing.
>
---
#### [new 096] Context-Enriched Contrastive Loss: Enhancing Presentation of Inherent Sample Connections in Contrastive Learning Framework
- **分类: cs.CV**

- **简介: 该论文属于自监督学习中的对比学习任务，旨在解决数据增强导致的样本信息失真问题。提出上下文增强的对比损失函数，通过双目标优化：强化类别间区分与拉近同源图像增强样本，提升模型泛化能力与收敛速度，在多个基准数据集上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.02152v1](https://arxiv.org/pdf/2512.02152v1)**

> **作者:** Haojin Deng; Yimin Yang
>
> **备注:** 13 pages, 7 figures. Published in IEEE Transactions on Multimedia. Code available at: https://github.com/hdeng26/Contex
>
> **摘要:** Contrastive learning has gained popularity and pushes state-of-the-art performance across numerous large-scale benchmarks. In contrastive learning, the contrastive loss function plays a pivotal role in discerning similarities between samples through techniques such as rotation or cropping. However, this learning mechanism can also introduce information distortion from the augmented samples. This is because the trained model may develop a significant overreliance on information from samples with identical labels, while concurrently neglecting positive pairs that originate from the same initial image, especially in expansive datasets. This paper proposes a context-enriched contrastive loss function that concurrently improves learning effectiveness and addresses the information distortion by encompassing two convergence targets. The first component, which is notably sensitive to label contrast, differentiates between features of identical and distinct classes which boosts the contrastive training efficiency. Meanwhile, the second component draws closer the augmented samples from the same source image and distances all other samples. We evaluate the proposed approach on image classification tasks, which are among the most widely accepted 8 recognition large-scale benchmark datasets: CIFAR10, CIFAR100, Caltech-101, Caltech-256, ImageNet, BiasedMNIST, UTKFace, and CelebA datasets. The experimental results demonstrate that the proposed method achieves improvements over 16 state-of-the-art contrastive learning methods in terms of both generalization performance and learning convergence speed. Interestingly, our technique stands out in addressing systematic distortion tasks. It demonstrates a 22.9% improvement compared to original contrastive loss functions in the downstream BiasedMNIST dataset, highlighting its promise for more efficient and equitable downstream training.
>
---
#### [new 097] LightHCG: a Lightweight yet powerful HSIC Disentanglement based Causal Glaucoma Detection Model framework
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LightHCG，一种轻量级因果驱动的青光眼检测模型。针对现有AI模型参数多、可靠性差、易产生伪相关等问题，利用HSIC解耦与图自编码器实现无监督因果表征学习，显著减少参数量（93~99%），提升分类性能与干预分析能力。**

- **链接: [https://arxiv.org/pdf/2512.02437v1](https://arxiv.org/pdf/2512.02437v1)**

> **作者:** Daeyoung Kim
>
> **摘要:** As a representative optic degenerative condition, glaucoma has been a threat to millions due to its irreversibility and severe impact on human vision fields. Mainly characterized by dimmed and blurred visions, or peripheral vision loss, glaucoma is well known to occur due to damages in the optic nerve from increased intraocular pressure (IOP) or neovascularization within the retina. Traditionally, most glaucoma related works and clinical diagnosis focused on detecting these damages in the optic nerve by using patient data from perimetry tests, optic papilla inspections and tonometer-based IOP measurements. Recently, with advancements in computer vision AI models, such as VGG16 or Vision Transformers (ViT), AI-automatized glaucoma detection and optic cup segmentation based on retinal fundus images or OCT recently exhibited significant performance in aiding conventional diagnosis with high performance. However, current AI-driven glaucoma detection approaches still have significant room for improvement in terms of reliability, excessive parameter usage, possibility of spurious correlation within detection, and limitations in applications to intervention analysis or clinical simulations. Thus, this research introduced a novel causal representation driven glaucoma detection model: LightHCG, an extremely lightweight Convolutional VAE-based latent glaucoma representation model that can consider the true causality among glaucoma-related physical factors within the optic nerve region. Using HSIC-based latent space disentanglement and Graph Autoencoder based unsupervised causal representation learning, LightHCG not only exhibits higher performance in classifying glaucoma with 93~99% less weights, but also enhances the possibility of AI-driven intervention analysis, compared to existing advanced vision models such as InceptionV3, MobileNetV2 or VGG16.
>
---
#### [new 098] ClimaOoD: Improving Anomaly Segmentation via Physically Realistic Synthetic Data
- **分类: cs.CV**

- **简介: 该论文针对开放世界中异常分割因真实异常数据稀缺导致模型泛化能力差的问题，提出ClimaDrive框架，生成语义一致、物理真实的多天气异常驾驶图像。基于此构建ClimaOoD大规模基准，显著提升多种方法的异常分割性能，尤其降低误报率，增强模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.02686v1](https://arxiv.org/pdf/2512.02686v1)**

> **作者:** Yuxing Liu; Yong Liu
>
> **备注:** Under review;
>
> **摘要:** Anomaly segmentation seeks to detect and localize unknown or out-of-distribution (OoD) objects that fall outside predefined semantic classes a capability essential for safe autonomous driving. However, the scarcity and limited diversity of anomaly data severely constrain model generalization in open-world environments. Existing approaches mitigate this issue through synthetic data generation, either by copy-pasting external objects into driving scenes or by leveraging text-to-image diffusion models to inpaint anomalous regions. While these methods improve anomaly diversity, they often lack contextual coherence and physical realism, resulting in domain gaps between synthetic and real data. In this paper, we present ClimaDrive, a semantics-guided image-to-image framework for synthesizing semantically coherent, weather-diverse, and physically plausible OoD driving data. ClimaDrive unifies structure-guided multi-weather generation with prompt-driven anomaly inpainting, enabling the creation of visually realistic training data. Based on this framework, we construct ClimaOoD, a large-scale benchmark spanning six representative driving scenarios under both clear and adverse weather conditions. Extensive experiments on four state-of-the-art methods show that training with ClimaOoD leads to robust improvements in anomaly segmentation. Across all methods, AUROC, AP, and FPR95 show notable gains, with FPR95 dropping from 3.97 to 3.52 for RbA on Fishyscapes LAF. These results demonstrate that ClimaOoD enhances model robustness, offering valuable training data for better generalization in open-world anomaly detection.
>
---
#### [new 099] WeMMU: Enhanced Bridging of Vision-Language Models and Diffusion Models via Noisy Query Tokens
- **分类: cs.CV**

- **简介: 该论文针对视觉-语言模型（VLM）与扩散模型融合中的任务泛化能力下降问题，提出使用噪声查询令牌（Noisy Query Tokens）构建跨模型表示空间，实现端到端优化。通过引入VAE分支恢复图像细节，提升了多任务持续学习的稳定性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.02536v1](https://arxiv.org/pdf/2512.02536v1)**

> **作者:** Jian Yang; Dacheng Yin; Xiaoxuan He; Yong Li; Fengyun Rao; Jing Lyu; Wei Zhai; Yang Cao; Zheng-Jun Zha
>
> **摘要:** Recent progress in multimodal large language models (MLLMs) has highlighted the challenge of efficiently bridging pre-trained Vision-Language Models (VLMs) with Diffusion Models. While methods using a fixed number of learnable query tokens offer computational efficiency, they suffer from task generalization collapse, failing to adapt to new tasks that are distant from their pre-training tasks. To overcome this, we propose Noisy Query Tokens, which learn a distributed representation space between the VLM and Diffusion Model via end-to-end optimization, enhancing continual learning. Additionally, we introduce a VAE branch with linear projection to recover fine-grained image details. Experimental results confirm our approach mitigates generalization collapse and enables stable continual learning across diverse tasks.
>
---
#### [new 100] Leveraging Large-Scale Pretrained Spatial-Spectral Priors for General Zero-Shot Pansharpening
- **分类: cs.CV**

- **简介: 该论文针对遥感图像融合中的跨域泛化难题，提出基于大规模模拟数据预训练的零样本全色锐化方法。通过构建多样化仿真数据集，学习通用的空间-光谱先验，显著提升模型在未见卫星数据上的适应能力，验证了基础模型在遥感图像融合中的潜力。**

- **链接: [https://arxiv.org/pdf/2512.02643v1](https://arxiv.org/pdf/2512.02643v1)**

> **作者:** Yongchuan Cui; Peng Liu; Yi Zeng
>
> **摘要:** Existing deep learning methods for remote sensing image fusion often suffer from poor generalization when applied to unseen datasets due to the limited availability of real training data and the domain gap between different satellite sensors. To address this challenge, we explore the potential of foundation models by proposing a novel pretraining strategy that leverages large-scale simulated datasets to learn robust spatial-spectral priors. Specifically, our approach first constructs diverse simulated datasets by applying various degradation operations (blur, noise, downsampling) and augmentations (bands generation, channel shuffling, high-pass filtering, color jittering, etc.) to natural images from ImageNet and remote sensing images from SkyScript. We then pretrain fusion models on these simulated data to learn generalizable spatial-spectral representations. The pretrained models are subsequently evaluated on six datasets (WorldView-2/3/4, IKONOS, QuickBird, GaoFen-2) using zero-shot and one-shot paradigms, with both full- and freeze-tuning approaches for fine-tuning. Extensive experiments on different network architectures including convolutional neural networks, Transformer, and Mamba demonstrate that our pretraining strategy significantly improves generalization performance across different satellite sensors and imaging conditions for various fusion models. The pretrained models achieve superior results in zero-shot scenarios and show remarkable adaptation capability with minimal real data in one-shot settings. Our work provides a practical solution for cross-domain pansharpening, establishes a new benchmark for generalization in remote sensing image fusion tasks, and paves the way for leveraging foundation models through advanced training strategies.
>
---
#### [new 101] PGP-DiffSR: Phase-Guided Progressive Pruning for Efficient Diffusion-based Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文针对扩散模型在图像超分辨率任务中计算与内存开销大的问题，提出PGP-DiffSR方法。通过相位引导的渐进剪枝去除冗余模块，并引入相位交换适配器提升重建质量，实现高效低耗的超分辨率。**

- **链接: [https://arxiv.org/pdf/2512.02681v1](https://arxiv.org/pdf/2512.02681v1)**

> **作者:** Zhongbao Yang; Jiangxin Dong; Yazhou Yao; Jinhui Tang; Jinshan Pan
>
> **备注:** 10 pages
>
> **摘要:** Although diffusion-based models have achieved impressive results in image super-resolution, they often rely on large-scale backbones such as Stable Diffusion XL (SDXL) and Diffusion Transformers (DiT), which lead to excessive computational and memory costs during training and inference. To address this issue, we develop a lightweight diffusion method, PGP-DiffSR, by removing redundant information from diffusion models under the guidance of the phase information of inputs for efficient image super-resolution. We first identify the intra-block redundancy within the diffusion backbone and propose a progressive pruning approach that removes redundant blocks while reserving restoration capability. We note that the phase information of the restored images produced by the pruned diffusion model is not well estimated. To solve this problem, we propose a phase-exchange adapter module that explores the phase information of the inputs to guide the pruned diffusion model for better restoration performance. We formulate the progressive pruning approach and the phase-exchange adapter module into a unified model. Extensive experiments demonstrate that our method achieves competitive restoration quality while significantly reducing computational load and memory consumption. The code is available at https://github.com/yzb1997/PGP-DiffSR.
>
---
#### [new 102] From Panel to Pixel: Zoom-In Vision-Language Pretraining from Biomedical Scientific Literature
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对生物医学视觉语言模型预训练中忽略细粒度对应关系的问题，提出Panel2Patch数据管道，从多面板科学图表中提取图、面板、局部区域三级对齐的视觉-语言对，构建层次化监督信号。通过细粒度预训练策略，显著提升模型性能，实现以更少数据获得更强表示。**

- **链接: [https://arxiv.org/pdf/2512.02566v1](https://arxiv.org/pdf/2512.02566v1)**

> **作者:** Kun Yuan; Min Woo Sun; Zhen Chen; Alejandro Lozano; Xiangteng He; Shi Li; Nassir Navab; Xiaoxiao Sun; Nicolas Padoy; Serena Yeung-Levy
>
> **摘要:** There is a growing interest in developing strong biomedical vision-language models. A popular approach to achieve robust representations is to use web-scale scientific data. However, current biomedical vision-language pretraining typically compresses rich scientific figures and text into coarse figure-level pairs, discarding the fine-grained correspondences that clinicians actually rely on when zooming into local structures. To tackle this issue, we introduce Panel2Patch, a novel data pipeline that mines hierarchical structure from existing biomedical scientific literature, i.e., multi-panel, marker-heavy figures and their surrounding text, and converts them into multi-granular supervision. Given scientific figures and captions, Panel2Patch parses layouts, panels, and visual markers, then constructs hierarchical aligned vision-language pairs at the figure, panel, and patch levels, preserving local semantics instead of treating each figure as a single data sample. Built on this hierarchical corpus, we develop a granularity-aware pretraining strategy that unifies heterogeneous objectives from coarse didactic descriptions to fine region-focused phrases. By applying Panel2Patch to only a small set of the literature figures, we extract far more effective supervision than prior pipelines, enabling substantially better performance with less pretraining data.
>
---
#### [new 103] OmniPerson: Unified Identity-Preserving Pedestrian Generation
- **分类: cs.CV**

- **简介: 该论文针对行人重识别（ReID）因数据隐私与标注成本导致的高质量训练数据匮乏问题，提出OmniPerson统一生成框架。通过多参考融合机制实现身份一致性，支持多模态、多姿态、文本控制的行人图像/视频生成，并构建了大规模可控行人生成数据集PersonSyn，显著提升生成质量与下游ReID性能。**

- **链接: [https://arxiv.org/pdf/2512.02554v1](https://arxiv.org/pdf/2512.02554v1)**

> **作者:** Changxiao Ma; Chao Yuan; Xincheng Shi; Yuzhuo Ma; Yongfei Zhang; Longkun Zhou; Yujia Zhang; Shangze Li; Yifan Xu
>
> **摘要:** Person re-identification (ReID) suffers from a lack of large-scale high-quality training data due to challenges in data privacy and annotation costs. While previous approaches have explored pedestrian generation for data augmentation, they often fail to ensure identity consistency and suffer from insufficient controllability, thereby limiting their effectiveness in dataset augmentation. To address this, We introduce OmniPerson, the first unified identity-preserving pedestrian generation pipeline for visible/infrared image/video ReID tasks. Our contributions are threefold: 1) We proposed OmniPerson, a unified generation model, offering holistic and fine-grained control over all key pedestrian attributes. Supporting RGB/IR modality image/video generation with any number of reference images, two kinds of person poses, and text. Also including RGB-to-IR transfer and image super-resolution abilities.2) We designed Multi-Refer Fuser for robust identity preservation with any number of reference images as input, making OmniPerson could distill a unified identity from a set of multi-view reference images, ensuring our generated pedestrians achieve high-fidelity pedestrian generation.3) We introduce PersonSyn, the first large-scale dataset for multi-reference, controllable pedestrian generation, and present its automated curation pipeline which transforms public, ID-only ReID benchmarks into a richly annotated resource with the dense, multi-modal supervision required for this task. Experimental results demonstrate that OmniPerson achieves SoTA in pedestrian generation, excelling in both visual fidelity and identity consistency. Furthermore, augmenting existing datasets with our generated data consistently improves the performance of ReID models. We will open-source the full codebase, pretrained model, and the PersonSyn dataset.
>
---
#### [new 104] PoreTrack3D: A Benchmark for Dynamic 3D Gaussian Splatting in Pore-Scale Facial Trajectory Tracking
- **分类: cs.CV**

- **简介: 该论文提出PoreTrack3D，首个面向孔隙尺度非刚性三维面部轨迹追踪的动态3D高斯溅射基准数据集。旨在解决精细面部运动捕捉难题，通过包含超44万条轨迹（含68条完整150帧标注）的数据集，推动微小皮肤表面运动分析，建立首个性能基准，为高保真面部动态重建提供新框架。**

- **链接: [https://arxiv.org/pdf/2512.02648v1](https://arxiv.org/pdf/2512.02648v1)**

> **作者:** Dong Li; Jiahao Xiong; Yingda Huang; Le Chang
>
> **摘要:** We introduce PoreTrack3D, the first benchmark for dynamic 3D Gaussian splatting in pore-scale, non-rigid 3D facial trajectory tracking. It contains over 440,000 facial trajectories in total, among which more than 52,000 are longer than 10 frames, including 68 manually reviewed trajectories that span the entire 150 frames. To the best of our knowledge, PoreTrack3D is the first benchmark dataset to capture both traditional facial landmarks and pore-scale keypoints trajectory, advancing the study of fine-grained facial expressions through the analysis of subtle skin-surface motion. We systematically evaluate state-of-the-art dynamic 3D Gaussian splatting methods on PoreTrack3D, establishing the first performance baseline in this domain. Overall, the pipeline developed for this benchmark dataset's creation establishes a new framework for high-fidelity facial motion capture and dynamic 3D reconstruction. Our dataset are publicly available at: https://github.com/JHXion9/PoreTrack3D
>
---
#### [new 105] MagicQuillV2: Precise and Interactive Image Editing with Layered Visual Cues
- **分类: cs.CV**

- **简介: 该论文提出MagicQuillV2，针对生成式图像编辑中用户意图模糊的问题，引入分层视觉提示机制，将创作意图分解为内容、空间、结构和颜色四层控制，实现精准、交互式编辑。通过专用数据生成与统一控制模块，提升对局部修改与对象移除的精确控制能力。**

- **链接: [https://arxiv.org/pdf/2512.03046v1](https://arxiv.org/pdf/2512.03046v1)**

> **作者:** Zichen Liu; Yue Yu; Hao Ouyang; Qiuyu Wang; Shuailei Ma; Ka Leong Cheng; Wen Wang; Qingyan Bai; Yuxuan Zhang; Yanhong Zeng; Yixuan Li; Xing Zhu; Yujun Shen; Qifeng Chen
>
> **备注:** Code and demo available at https://magicquill.art/v2/
>
> **摘要:** We propose MagicQuill V2, a novel system that introduces a \textbf{layered composition} paradigm to generative image editing, bridging the gap between the semantic power of diffusion models and the granular control of traditional graphics software. While diffusion transformers excel at holistic generation, their use of singular, monolithic prompts fails to disentangle distinct user intentions for content, position, and appearance. To overcome this, our method deconstructs creative intent into a stack of controllable visual cues: a content layer for what to create, a spatial layer for where to place it, a structural layer for how it is shaped, and a color layer for its palette. Our technical contributions include a specialized data generation pipeline for context-aware content integration, a unified control module to process all visual cues, and a fine-tuned spatial branch for precise local editing, including object removal. Extensive experiments validate that this layered approach effectively resolves the user intention gap, granting creators direct, intuitive control over the generative process.
>
---
#### [new 106] Mapping of Lesion Images to Somatic Mutations
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于跨模态生成任务，旨在通过医学影像预测癌症患者的体细胞突变。针对影像与基因数据间关联建模难题，提出LLOST模型，利用点云表示和双变分自编码器共享潜在空间，融合影像特征与突变计数，通过条件归一化流建模多分布。实验验证了其在突变预测与癌症类型关联分析上的有效性。**

- **链接: [https://arxiv.org/pdf/2512.02162v1](https://arxiv.org/pdf/2512.02162v1)**

> **作者:** Rahul Mehta
>
> **备注:** https://dl.acm.org/doi/abs/10.1145/3340531.3414074#sec-terms
>
> **摘要:** Medical imaging is a critical initial tool used by clinicians to determine a patient's cancer diagnosis, allowing for faster intervention and more reliable patient prognosis. At subsequent stages of patient diagnosis, genetic information is extracted to help select specific patient treatment options. As the efficacy of cancer treatment often relies on early diagnosis and treatment, we build a deep latent variable model to determine patients' somatic mutation profiles based on their corresponding medical images. We first introduce a point cloud representation of lesions images to allow for invariance to the imaging modality. We then propose, LLOST, a model with dual variational autoencoders coupled together by a separate shared latent space that unifies features from the lesion point clouds and counts of distinct somatic mutations. Therefore our model consists of three latent space, each of which is learned with a conditional normalizing flow prior to account for the diverse distributions of each domain. We conduct qualitative and quantitative experiments on de-identified medical images from The Cancer Imaging Archive and the corresponding somatic mutations from the Pan Cancer dataset of The Cancer Genomic Archive. We show the model's predictive performance on the counts of specific mutations as well as it's ability to accurately predict the occurrence of mutations. In particular, shared patterns between the imaging and somatic mutation domain that reflect cancer type. We conclude with a remark on how to improve the model and possible future avenues of research to include other genetic domains.
>
---
#### [new 107] PolarGuide-GSDR: 3D Gaussian Splatting Driven by Polarization Priors and Deferred Reflection for Real-World Reflective Scenes
- **分类: cs.CV**

- **简介: 该论文针对复杂反射场景的三维重建任务，解决传统方法在材质假设、环境图依赖及渲染效率上的瓶颈。提出PolarGuide-GSDR框架，通过极化先验与3D高斯溅射的双向耦合，实现无需环境图的实时高保真反射分离与重建，显著提升正常估计与新视角合成质量。**

- **链接: [https://arxiv.org/pdf/2512.02664v1](https://arxiv.org/pdf/2512.02664v1)**

> **作者:** Derui Shan; Qian Qiao; Hao Lu; Tao Du; Peng Lu
>
> **摘要:** Polarization-aware Neural Radiance Fields (NeRF) enable novel view synthesis of specular-reflection scenes but face challenges in slow training, inefficient rendering, and strong dependencies on material/viewpoint assumptions. However, 3D Gaussian Splatting (3DGS) enables real-time rendering yet struggles with accurate reflection reconstruction from reflection-geometry entanglement, adding a deferred reflection module introduces environment map dependence. We address these limitations by proposing PolarGuide-GSDR, a polarization-forward-guided paradigm establishing a bidirectional coupling mechanism between polarization and 3DGS: first 3DGS's geometric priors are leveraged to resolve polarization ambiguity, and then the refined polarization information cues are used to guide 3DGS's normal and spherical harmonic representation. This process achieves high-fidelity reflection separation and full-scene reconstruction without requiring environment maps or restrictive material assumptions. We demonstrate on public and self-collected datasets that PolarGuide-GSDR achieves state-of-the-art performance in specular reconstruction, normal estimation, and novel view synthesis, all while maintaining real-time rendering capabilities. To our knowledge, this is the first framework embedding polarization priors directly into 3DGS optimization, yielding superior interpretability and real-time performance for modeling complex reflective scenes.
>
---
#### [new 108] Spatiotemporal Pyramid Flow Matching for Climate Emulation
- **分类: cs.CV; cs.AI; cs.LG; eess.IV; stat.ML**

- **简介: 该论文提出时空金字塔流匹配（SPF）模型，用于高效、并行的气候模拟。针对传统方法在长时序模拟中速度慢、不稳定的问题，SPF通过多尺度时空分层建模，实现跨时间尺度快速采样，并结合物理强迫条件提升泛化能力。研究构建了最大规模气候模拟数据集ClimateSuite，验证了模型在多种情景下的有效性。**

- **链接: [https://arxiv.org/pdf/2512.02268v1](https://arxiv.org/pdf/2512.02268v1)**

> **作者:** Jeremy Andrew Irvin; Jiaqi Han; Zikui Wang; Abdulaziz Alharbi; Yufei Zhao; Nomin-Erdene Bayarsaikhan; Daniele Visioni; Andrew Y. Ng; Duncan Watson-Parris
>
> **摘要:** Generative models have the potential to transform the way we emulate Earth's changing climate. Previous generative approaches rely on weather-scale autoregression for climate emulation, but this is inherently slow for long climate horizons and has yet to demonstrate stable rollouts under nonstationary forcings. Here, we introduce Spatiotemporal Pyramid Flows (SPF), a new class of flow matching approaches that model data hierarchically across spatial and temporal scales. Inspired by cascaded video models, SPF partitions the generative trajectory into a spatiotemporal pyramid, progressively increasing spatial resolution to reduce computation and coupling each stage with an associated timescale to enable direct sampling at any temporal level in the pyramid. This design, together with conditioning each stage on prescribed physical forcings (e.g., greenhouse gases or aerosols), enables efficient, parallel climate emulation at multiple timescales. On ClimateBench, SPF outperforms strong flow matching baselines and pre-trained models at yearly and monthly timescales while offering fast sampling, especially at coarser temporal levels. To scale SPF, we curate ClimateSuite, the largest collection of Earth system simulations to date, comprising over 33,000 simulation-years across ten climate models and the first dataset to include simulations of climate interventions. We find that the scaled SPF model demonstrates good generalization to held-out scenarios across climate models. Together, SPF and ClimateSuite provide a foundation for accurate, efficient, probabilistic climate emulation across temporal scales and realistic future scenarios. Data and code is publicly available at https://github.com/stanfordmlgroup/spf .
>
---
#### [new 109] WSCF-MVCC: Weakly-supervised Calibration-free Multi-view Crowd Counting
- **分类: cs.CV**

- **简介: 该论文针对多视角人群计数任务，解决现有方法依赖昂贵标注与相机标定的问题。提出弱监督无标定多视角计数方法WSCF-MVCC，以人数为监督信号，结合自监督排序损失和语义信息，实现无需标注与标定的精准计数，显著提升实用性和性能。**

- **链接: [https://arxiv.org/pdf/2512.02359v1](https://arxiv.org/pdf/2512.02359v1)**

> **作者:** Bin Li; Daijie Chen; Qi Zhang
>
> **备注:** PRCV 2025
>
> **摘要:** Multi-view crowd counting can effectively mitigate occlusion issues that commonly arise in single-image crowd counting. Existing deep-learning multi-view crowd counting methods project different camera view images onto a common space to obtain ground-plane density maps, requiring abundant and costly crowd annotations and camera calibrations. Hence, calibration-free methods are proposed that do not require camera calibrations and scene-level crowd annotations. However, existing calibration-free methods still require expensive image-level crowd annotations for training the single-view counting module. Thus, in this paper, we propose a weakly-supervised calibration-free multi-view crowd counting method (WSCF-MVCC), directly using crowd count as supervision for the single-view counting module rather than density maps constructed from crowd annotations. Instead, a self-supervised ranking loss that leverages multi-scale priors is utilized to enhance the model's perceptual ability without additional annotation costs. What's more, the proposed model leverages semantic information to achieve a more accurate view matching and, consequently, a more precise scene-level crowd count estimation. The proposed method outperforms the state-of-the-art methods on three widely used multi-view counting datasets under weakly supervised settings, indicating that it is more suitable for practical deployment compared with calibrated methods. Code is released in https://github.com/zqyq/Weakly-MVCC.
>
---
#### [new 110] A Large Scale Benchmark for Test Time Adaptation Methods in Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割中的域偏移问题，提出MedSeg-TTA基准，系统评估20种测试时自适应方法在7种影像模态下的表现。通过统一流程，比较输入、特征、输出及先验四类方法，揭示其优劣与适用场景，推动临床可靠自适应方法的研究。**

- **链接: [https://arxiv.org/pdf/2512.02497v1](https://arxiv.org/pdf/2512.02497v1)**

> **作者:** Wenjing Yu; Shuo Jiang; Yifei Chen; Shuo Chang; Yuanhan Wang; Beining Wu; Jie Dong; Mingxuan Liu; Shenghao Zhu; Feiwei Qin; Changmiao Wang; Qiyuan Tian
>
> **备注:** 45 pages, 18 figures
>
> **摘要:** Test time Adaptation is a promising approach for mitigating domain shift in medical image segmentation; however, current evaluations remain limited in terms of modality coverage, task diversity, and methodological consistency. We present MedSeg-TTA, a comprehensive benchmark that examines twenty representative adaptation methods across seven imaging modalities, including MRI, CT, ultrasound, pathology, dermoscopy, OCT, and chest X-ray, under fully unified data preprocessing, backbone configuration, and test time protocols. The benchmark encompasses four significant adaptation paradigms: Input-level Transformation, Feature-level Alignment, Output-level Regularization, and Prior Estimation, enabling the first systematic cross-modality comparison of their reliability and applicability. The results show that no single paradigm performs best in all conditions. Input-level methods are more stable under mild appearance shifts. Feature-level and Output-level methods offer greater advantages in boundary-related metrics, whereas prior-based methods exhibit strong modality dependence. Several methods degrade significantly under large inter-center and inter-device shifts, which highlights the importance of principled method selection for clinical deployment. MedSeg-TTA provides standardized datasets, validated implementations, and a public leaderboard, establishing a rigorous foundation for future research on robust, clinically reliable test-time adaptation. All source codes and open-source datasets are available at https://github.com/wenjing-gg/MedSeg-TTA.
>
---
#### [new 111] RobustSurg: Tackling domain generalisation for out-of-distribution surgical scene segmentation
- **分类: cs.CV**

- **简介: 该论文针对外科手术场景分割中的域泛化问题，旨在提升模型在未见中心和模态下的性能。提出RobustSurg方法，通过风格与内容分离、特征协方差映射及重建模块，增强特征鲁棒性，并构建新数据集以支持多中心训练。实验表明，该方法显著优于基线与当前最优模型。**

- **链接: [https://arxiv.org/pdf/2512.02188v1](https://arxiv.org/pdf/2512.02188v1)**

> **作者:** Mansoor Ali; Maksim Richards; Gilberto Ochoa-Ruiz; Sharib Ali
>
> **备注:** Submitted to Medical Image Analysis
>
> **摘要:** While recent advances in deep learning for surgical scene segmentation have demonstrated promising results on single-centre and single-imaging modality data, these methods usually do not generalise to unseen distribution (i.e., from other centres) and unseen modalities. Current literature for tackling generalisation on out-of-distribution data and domain gaps due to modality changes has been widely researched but mostly for natural scene data. However, these methods cannot be directly applied to the surgical scenes due to limited visual cues and often extremely diverse scenarios compared to the natural scene data. Inspired by these works in natural scenes to push generalisability on OOD data, we hypothesise that exploiting the style and content information in the surgical scenes could minimise the appearances, making it less variable to sudden changes such as blood or imaging artefacts. This can be achieved by performing instance normalisation and feature covariance mapping techniques for robust and generalisable feature representations. Further, to eliminate the risk of removing salient feature representation associated with the objects of interest, we introduce a restitution module within the feature learning ResNet backbone that can enable the retention of useful task-relevant features. To tackle the lack of multiclass and multicentre data for surgical scene segmentation, we also provide a newly curated dataset that can be vital for addressing generalisability in this domain. Our proposed RobustSurg obtained nearly 23% improvement on the baseline DeepLabv3+ and from 10-32% improvement on the SOTA in terms of mean IoU score on an unseen centre HeiCholSeg dataset when trained on CholecSeg8K. Similarly, RobustSurg also obtained nearly 22% improvement over the baseline and nearly 11% improvement on a recent SOTA method for the target set of the EndoUDA polyp dataset.
>
---
#### [new 112] Unrolled Networks are Conditional Probability Flows in MRI Reconstruction
- **分类: cs.CV**

- **简介: 该论文研究MRI图像重建任务，针对传统未卷网络因中间参数自由学习导致演化不稳定的问题，提出将其视为条件概率流微分方程的离散实现。基于此，设计了流对齐训练（FLAT）方法，通过欧拉法离散化构建参数并引导中间重建轨迹，提升稳定性和收敛性，显著减少迭代次数。**

- **链接: [https://arxiv.org/pdf/2512.03020v1](https://arxiv.org/pdf/2512.03020v1)**

> **作者:** Kehan Qi; Saumya Gupta; Qingqiao Hu; Weimin Lyu; Chao Chen
>
> **摘要:** Magnetic Resonance Imaging (MRI) offers excellent soft-tissue contrast without ionizing radiation, but its long acquisition time limits clinical utility. Recent methods accelerate MRI by under-sampling $k$-space and reconstructing the resulting images using deep learning. Unrolled networks have been widely used for the reconstruction task due to their efficiency, but suffer from unstable evolving caused by freely-learnable parameters in intermediate steps. In contrast, diffusion models based on stochastic differential equations offer theoretical stability in both medical and natural image tasks but are computationally expensive. In this work, we introduce flow ODEs to MRI reconstruction by theoretically proving that unrolled networks are discrete implementations of conditional probability flow ODEs. This connection provides explicit formulations for parameters and clarifies how intermediate states should evolve. Building on this insight, we propose Flow-Aligned Training (FLAT), which derives unrolled parameters from the ODE discretization and aligns intermediate reconstructions with the ideal ODE trajectory to improve stability and convergence. Experiments on three MRI datasets show that FLAT achieves high-quality reconstructions with up to $3\times$ fewer iterations than diffusion-based generative models and significantly greater stability than unrolled networks.
>
---
#### [new 113] Two-Stage Vision Transformer for Image Restoration: Colorization Pretraining + Residual Upsampling
- **分类: cs.CV**

- **简介: 该论文针对单图像超分辨率（SISR）任务，提出两阶段视觉变换器ViT-SR。先通过自监督色彩化预训练学习通用视觉表征，再微调用于4倍超分辨率，通过预测高频残差图简化残差学习。在DIV2K数据集上取得PSNR 22.90 dB、SSIM 0.712的优异性能，验证了两阶段策略的有效性。**

- **链接: [https://arxiv.org/pdf/2512.02512v1](https://arxiv.org/pdf/2512.02512v1)**

> **作者:** Aditya Chaudhary; Prachet Dev Singh; Ankit Jha
>
> **备注:** Accepted at the 13th Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP 2025), IIT Mandi, India. 3 pages, 1 figure
>
> **摘要:** In computer vision, Single Image Super-Resolution (SISR) is still a difficult problem. We present ViT-SR, a new technique to improve the performance of a Vision Transformer (ViT) employing a two-stage training strategy. In our method, the model learns rich, generalizable visual representations from the data itself through a self-supervised pretraining phase on a colourization task. The pre-trained model is then adjusted for 4x super-resolution. By predicting the addition of a high-frequency residual image to an initial bicubic interpolation, this design simplifies residual learning. ViT-SR, trained and evaluated on the DIV2K benchmark dataset, achieves an impressive SSIM of 0.712 and PSNR of 22.90 dB. These results demonstrate the efficacy of our two-stage approach and highlight the potential of self-supervised pre-training for complex image restoration tasks. Further improvements may be possible with larger ViT architectures or alternative pretext tasks.
>
---
#### [new 114] Tackling Tuberculosis: A Comparative Dive into Machine Learning for Tuberculosis Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决结核病（TB）诊断效率低的问题。研究对比了ResNet-50与SqueezeNet在胸部X光片上的检测性能，使用4200张图像数据集进行训练与评估，结果表明SqueezeNet表现更优，具备临床应用潜力。**

- **链接: [https://arxiv.org/pdf/2512.02364v1](https://arxiv.org/pdf/2512.02364v1)**

> **作者:** Daanish Hindustani; Sanober Hindustani; Preston Nguyen
>
> **摘要:** This study explores the application of machine learning models, specifically a pretrained ResNet-50 model and a general SqueezeNet model, in diagnosing tuberculosis (TB) using chest X-ray images. TB, a persistent infectious disease affecting humanity for millennia, poses challenges in diagnosis, especially in resource-limited settings. Traditional methods, such as sputum smear microscopy and culture, are inefficient, prompting the exploration of advanced technologies like deep learning and computer vision. The study utilized a dataset from Kaggle, consisting of 4,200 chest X-rays, to develop and compare the performance of the two machine learning models. Preprocessing involved data splitting, augmentation, and resizing to enhance training efficiency. Evaluation metrics, including accuracy, precision, recall, and confusion matrix, were employed to assess model performance. Results showcase that the SqueezeNet achieved a loss of 32%, accuracy of 89%, precision of 98%, recall of 80%, and an F1 score of 87%. In contrast, the ResNet-50 model exhibited a loss of 54%, accuracy of 73%, precision of 88%, recall of 52%, and an F1 score of 65%. This study emphasizes the potential of machine learning in TB detection and possible implications for early identification and treatment initiation. The possibility of integrating such models into mobile devices expands their utility in areas lacking TB detection resources. However, despite promising results, the need for continued development of faster, smaller, and more accurate TB detection models remains crucial in contributing to the global efforts in combating TB.
>
---
#### [new 115] ReVSeg: Incentivizing the Reasoning Chain for Video Segmentation with Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视频对象分割中推理链不透明的问题，提出ReVSeg方法。通过强化学习驱动预训练视觉语言模型，分步执行语义理解、时序证据选择与空间定位，显式建模复杂动态推理，提升分割性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.02835v1](https://arxiv.org/pdf/2512.02835v1)**

> **作者:** Yifan Li; Yingda Yin; Lingting Zhu; Weikai Chen; Shengju Qian; Xin Wang; Yanwei Fu
>
> **摘要:** Reasoning-centric video object segmentation is an inherently complex task: the query often refers to dynamics, causality, and temporal interactions, rather than static appearances. Yet existing solutions generally collapse these factors into simplified reasoning with latent embeddings, rendering the reasoning chain opaque and essentially intractable. We therefore adopt an explicit decomposition perspective and introduce ReVSeg, which executes reasoning as sequential decisions in the native interface of pretrained vision language models (VLMs). Rather than folding all reasoning into a single-step prediction, ReVSeg executes three explicit operations -- semantics interpretation, temporal evidence selection, and spatial grounding -- aligning pretrained capabilities. We further employ reinforcement learning to optimize the multi-step reasoning chain, enabling the model to self-refine its decision quality from outcome-driven signals. Experimental results demonstrate that ReVSeg attains state-of-the-art performances on standard video object segmentation benchmarks and yields interpretable reasoning trajectories. Project page is available at https://clementine24.github.io/ReVSeg/ .
>
---
#### [new 116] Are Detectors Fair to Indian IP-AIGC? A Cross-Generator Study
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究AIGC图像检测中对印度及南亚人脸的公平性问题。针对身份保持型生成图像（IP-AIGC），构建了首个面向印度人群的测试集，评估主流检测器在跨生成器场景下的表现。发现微调导致对印度样本性能显著下降，暴露检测器对训练生成器的过拟合，提出需关注表示一致性与本土化基准建设。**

- **链接: [https://arxiv.org/pdf/2512.02850v1](https://arxiv.org/pdf/2512.02850v1)**

> **作者:** Vishal Dubey; Pallavi Tyagi
>
> **摘要:** Modern image editors can produce identity-preserving AIGC (IP-AIGC), where the same person appears with new attire, background, or lighting. The robustness and fairness of current detectors in this regime remain unclear, especially for under-represented populations. We present what we believe is the first systematic study of IP-AIGC detection for Indian and South-Asian faces, quantifying cross-generator generalization and intra-population performance. We assemble Indian-focused training splits from FairFD and HAV-DF, and construct two held-out IP-AIGC test sets (HIDF-img-ip-genai and HIDF-vid-ip-genai) using commercial web-UI generators (Gemini and ChatGPT) with identity-preserving prompts. We evaluate two state-of-the-art detectors (AIDE and Effort) under pretrained (PT) and fine-tuned (FT) regimes and report AUC, AP, EER, and accuracy. Fine-tuning yields strong in-domain gains (for example, Effort AUC 0.739 to 0.944 on HAV-DF-test; AIDE EER 0.484 to 0.259), but consistently degrades performance on held-out IP-AIGC for Indian cohorts (for example, AIDE AUC 0.923 to 0.563 on HIDF-img-ip-genai; Effort 0.740 to 0.533), which indicates overfitting to training-generator cues. On non-IP HIDF images, PT performance remains high, which suggests a specific brittleness to identity-preserving edits rather than a generic distribution shift. Our study establishes IP-AIGC-Indian as a challenging and practically relevant scenario and motivates representation-preserving adaptation and India-aware benchmark curation to close generalization gaps in AIGC detection.
>
---
#### [new 117] Reproducing and Extending RaDelft 4D Radar with Camera-Assisted Labels
- **分类: cs.CV**

- **简介: 该论文针对4D雷达语义分割中缺乏公开标注数据的问题，提出基于摄像头辅助的雷达点云标注方法。通过将雷达点云投影至相机语义图并结合空间聚类，实现无需人工标注的自动化标签生成，提升了标签准确性，并验证了不同雾天条件对标注性能的影响，构建了可复现的研究框架。**

- **链接: [https://arxiv.org/pdf/2512.02394v1](https://arxiv.org/pdf/2512.02394v1)**

> **作者:** Kejia Hu; Mohammed Alsakabi; John M. Dolan; Ozan K. Tonguz
>
> **摘要:** Recent advances in 4D radar highlight its potential for robust environment perception under adverse conditions, yet progress in radar semantic segmentation remains constrained by the scarcity of open source datasets and labels. The RaDelft data set, although seminal, provides only LiDAR annotations and no public code to generate radar labels, limiting reproducibility and downstream research. In this work, we reproduce the numerical results of the RaDelft group and demonstrate that a camera-guided radar labeling pipeline can generate accurate labels for radar point clouds without relying on human annotations. By projecting radar point clouds into camera-based semantic segmentation and applying spatial clustering, we create labels that significantly enhance the accuracy of radar labels. These results establish a reproducible framework that allows the research community to train and evaluate the labeled 4D radar data. In addition, we study and quantify how different fog levels affect the radar labeling performance.
>
---
#### [new 118] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型中的幻觉问题，提出无需训练的InEx框架。通过内部反思（基于熵的不确定性估计）和跨模态多智能体协作（编辑与自省代理），实现自主纠错，提升推理可靠性。实验表明，InEx在多个基准上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.02981v1](https://arxiv.org/pdf/2512.02981v1)**

> **作者:** Zhongyu Yang; Yingfang Yuan; Xuanming Jiang; Baoyi An; Wei Pang
>
> **备注:** Published in AAAI 2026
>
> **摘要:** Hallucination remains a critical challenge in large language models (LLMs), hindering the development of reliable multimodal LLMs (MLLMs). Existing solutions often rely on human intervention or underutilize the agent's ability to autonomously mitigate hallucination. To address these limitations, we draw inspiration from how humans make reliable decisions in the real world. They begin with introspective reasoning to reduce uncertainty and form an initial judgment, then rely on external verification from diverse perspectives to reach a final decision. Motivated by this cognitive paradigm, we propose InEx, a training-free, multi-agent framework designed to autonomously mitigate hallucination. InEx introduces internal introspective reasoning, guided by entropy-based uncertainty estimation, to improve the reliability of the decision agent's reasoning process. The agent first generates a response, which is then iteratively verified and refined through external cross-modal multi-agent collaboration with the editing agent and self-reflection agents, further enhancing reliability and mitigating hallucination. Extensive experiments show that InEx consistently outperforms existing methods, achieving 4%-27% gains on general and hallucination benchmarks, and demonstrating strong robustness.
>
---
#### [new 119] MultiShotMaster: A Controllable Multi-Shot Video Generation Framework
- **分类: cs.CV**

- **简介: 该论文提出MultiShotMaster框架，解决视频生成中多镜头叙事连贯性与可控性不足的问题。通过引入两种新型RoPE机制，实现灵活镜头安排与时空参考注入，并构建自动化数据标注流程，支持文本驱动、主体可控、场景定制的多镜头视频生成。**

- **链接: [https://arxiv.org/pdf/2512.03041v1](https://arxiv.org/pdf/2512.03041v1)**

> **作者:** Qinghe Wang; Xiaoyu Shi; Baolu Li; Weikang Bian; Quande Liu; Huchuan Lu; Xintao Wang; Pengfei Wan; Kun Gai; Xu Jia
>
> **备注:** Project Page: https://qinghew.github.io/MultiShotMaster
>
> **摘要:** Current video generation techniques excel at single-shot clips but struggle to produce narrative multi-shot videos, which require flexible shot arrangement, coherent narrative, and controllability beyond text prompts. To tackle these challenges, we propose MultiShotMaster, a framework for highly controllable multi-shot video generation. We extend a pretrained single-shot model by integrating two novel variants of RoPE. First, we introduce Multi-Shot Narrative RoPE, which applies explicit phase shift at shot transitions, enabling flexible shot arrangement while preserving the temporal narrative order. Second, we design Spatiotemporal Position-Aware RoPE to incorporate reference tokens and grounding signals, enabling spatiotemporal-grounded reference injection. In addition, to overcome data scarcity, we establish an automated data annotation pipeline to extract multi-shot videos, captions, cross-shot grounding signals and reference images. Our framework leverages the intrinsic architectural properties to support multi-shot video generation, featuring text-driven inter-shot consistency, customized subject with motion control, and background-driven customized scene. Both shot count and duration are flexibly configurable. Extensive experiments demonstrate the superior performance and outstanding controllability of our framework.
>
---
#### [new 120] WISE: Weighted Iterative Society-of-Experts for Robust Multimodal Multi-Agent Debate
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究多模态视觉-语言推理中的多智能体辩论任务。针对现有方法局限于纯文本、难以融合异构模型优势的问题，提出WISE框架，通过求解者与反思者协作，结合加权聚合机制，提升推理鲁棒性。在多个多模态数据集上验证，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.02405v1](https://arxiv.org/pdf/2512.02405v1)**

> **作者:** Anoop Cherian; River Doyle; Eyal Ben-Dov; Suhas Lohit; Kuan-Chuan Peng
>
> **摘要:** Recent large language models (LLMs) are trained on diverse corpora and tasks, leading them to develop complementary strengths. Multi-agent debate (MAD) has emerged as a popular way to leverage these strengths for robust reasoning, though it has mostly been applied to language-only tasks, leaving its efficacy on multimodal problems underexplored. In this paper, we study MAD for solving vision-and-language reasoning problems. Our setup enables generalizing the debate protocol with heterogeneous experts that possess single- and multi-modal capabilities. To this end, we present Weighted Iterative Society-of-Experts (WISE), a generalized and modular MAD framework that partitions the agents into Solvers, that generate solutions, and Reflectors, that verify correctness, assign weights, and provide natural language feedback. To aggregate the agents' solutions across debate rounds, while accounting for variance in their responses and the feedback weights, we present a modified Dawid-Skene algorithm for post-processing that integrates our two-stage debate model. We evaluate WISE on SMART-840, VisualPuzzles, EvoChart-QA, and a new SMART-840++ dataset with programmatically generated problem instances of controlled difficulty. Our results show that WISE consistently improves accuracy by 2-7% over the state-of-the-art MAD setups and aggregation methods across diverse multimodal tasks and LLM configurations.
>
---
#### [new 121] Co-speech Gesture Video Generation via Motion-Based Graph Retrieval
- **分类: cs.CV**

- **简介: 该论文聚焦于生成同步自然的口部伴随手势视频的任务。针对音频与手势间多对多映射导致的同步性与自然性不足问题，提出结合扩散模型生成手势轨迹，并设计运动特征检索算法，通过全局与局部相似性匹配，最终拼接生成连贯手势视频，显著提升生成质量。**

- **链接: [https://arxiv.org/pdf/2512.02576v1](https://arxiv.org/pdf/2512.02576v1)**

> **作者:** Yafei Song; Peng Zhang; Bang Zhang
>
> **摘要:** Synthesizing synchronized and natural co-speech gesture videos remains a formidable challenge. Recent approaches have leveraged motion graphs to harness the potential of existing video data. To retrieve an appropriate trajectory from the graph, previous methods either utilize the distance between features extracted from the input audio and those associated with the motions in the graph or embed both the input audio and motion into a shared feature space. However, these techniques may not be optimal due to the many-to-many mapping nature between audio and gestures, which cannot be adequately addressed by one-to-one mapping. To alleviate this limitation, we propose a novel framework that initially employs a diffusion model to generate gesture motions. The diffusion model implicitly learns the joint distribution of audio and motion, enabling the generation of contextually appropriate gestures from input audio sequences. Furthermore, our method extracts both low-level and high-level features from the input audio to enrich the training process of the diffusion model. Subsequently, a meticulously designed motion-based retrieval algorithm is applied to identify the most suitable path within the graph by assessing both global and local similarities in motion. Given that not all nodes in the retrieved path are sequentially continuous, the final step involves seamlessly stitching together these segments to produce a coherent video output. Experimental results substantiate the efficacy of our proposed method, demonstrating a significant improvement over prior approaches in terms of synchronization accuracy and naturalness of generated gestures.
>
---
#### [new 122] Does Hearing Help Seeing? Investigating Audio-Video Joint Denoising for Video Generation
- **分类: cs.CV**

- **简介: 该论文研究音频-视频联合去噪对视频生成质量的影响，旨在解决“音频辅助是否提升纯视频质量”这一问题。提出参数高效的AVFullDiT架构，在相同条件下对比联合训练与仅视频训练模型，发现联合训练在复杂运动场景下显著提升视频质量，证实音频作为先验信号可增强模型对物理因果关系的理解。**

- **链接: [https://arxiv.org/pdf/2512.02457v1](https://arxiv.org/pdf/2512.02457v1)**

> **作者:** Jianzong Wu; Hao Lian; Dachao Hao; Ye Tian; Qingyu Shi; Biaolong Chen; Hao Jiang
>
> **备注:** Project page at https://jianzongwu.github.io/projects/does-hearing-help-seeing/
>
> **摘要:** Recent audio-video generative systems suggest that coupling modalities benefits not only audio-video synchrony but also the video modality itself. We pose a fundamental question: Does audio-video joint denoising training improve video generation, even when we only care about video quality? To study this, we introduce a parameter-efficient Audio-Video Full DiT (AVFullDiT) architecture that leverages pre-trained text-to-video (T2V) and text-to-audio (T2A) modules for joint denoising. We train (i) a T2AV model with AVFullDiT and (ii) a T2V-only counterpart under identical settings. Our results provide the first systematic evidence that audio-video joint denoising can deliver more than synchrony. We observe consistent improvements on challenging subsets featuring large and object contact motions. We hypothesize that predicting audio acts as a privileged signal, encouraging the model to internalize causal relationships between visual events and their acoustic consequences (e.g., collision $\times$ impact sound), which in turn regularizes video dynamics. Our findings suggest that cross-modal co-training is a promising approach to developing stronger, more physically grounded world models. Code and dataset will be made publicly available.
>
---
#### [new 123] ViSAudio: End-to-End Video-Driven Binaural Spatial Audio Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出端到端的视频驱动双耳空间音频生成任务，解决现有方法多为两阶段、缺乏空间沉浸感的问题。构建了97K规模的BiAudio数据集，提出ViSAudio框架，通过双分支潜流建模与时空条件模块，实现精准声像对齐与动态空间感知，显著提升音视频一致性与空间真实感。**

- **链接: [https://arxiv.org/pdf/2512.03036v1](https://arxiv.org/pdf/2512.03036v1)**

> **作者:** Mengchen Zhang; Qi Chen; Tong Wu; Zihan Liu; Dahua Lin
>
> **摘要:** Despite progress in video-to-audio generation, the field focuses predominantly on mono output, lacking spatial immersion. Existing binaural approaches remain constrained by a two-stage pipeline that first generates mono audio and then performs spatialization, often resulting in error accumulation and spatio-temporal inconsistencies. To address this limitation, we introduce the task of end-to-end binaural spatial audio generation directly from silent video. To support this task, we present the BiAudio dataset, comprising approximately 97K video-binaural audio pairs spanning diverse real-world scenes and camera rotation trajectories, constructed through a semi-automated pipeline. Furthermore, we propose ViSAudio, an end-to-end framework that employs conditional flow matching with a dual-branch audio generation architecture, where two dedicated branches model the audio latent flows. Integrated with a conditional spacetime module, it balances consistency between channels while preserving distinctive spatial characteristics, ensuring precise spatio-temporal alignment between audio and the input video. Comprehensive experiments demonstrate that ViSAudio outperforms existing state-of-the-art methods across both objective metrics and subjective evaluations, generating high-quality binaural audio with spatial immersion that adapts effectively to viewpoint changes, sound-source motion, and diverse acoustic environments. Project website: https://kszpxxzmc.github.io/ViSAudio-project.
>
---
#### [new 124] Glance: Accelerating Diffusion Models with 1 Sample
- **分类: cs.CV**

- **简介: 该论文针对扩散模型推理慢的问题，提出Glance方法，通过分阶段加速（慢阶段小提速、快阶段大提速）实现高效生成。利用仅需1样本训练的轻量LoRA适配器（Slow/Fast-LoRA），在单张V100上一小时内完成训练，实现5倍加速且保持高质量生成与强泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.02899v1](https://arxiv.org/pdf/2512.02899v1)**

> **作者:** Zhuobai Dong; Rui Zhao; Songjie Wu; Junchao Yi; Linjie Li; Zhengyuan Yang; Lijuan Wang; Alex Jinpeng Wang
>
> **摘要:** Diffusion models have achieved remarkable success in image generation, yet their deployment remains constrained by the heavy computational cost and the need for numerous inference steps. Previous efforts on fewer-step distillation attempt to skip redundant steps by training compact student models, yet they often suffer from heavy retraining costs and degraded generalization. In this work, we take a different perspective: we accelerate smartly, not evenly, applying smaller speedups to early semantic stages and larger ones to later redundant phases. We instantiate this phase-aware strategy with two experts that specialize in slow and fast denoising phases. Surprisingly, instead of investing massive effort in retraining student models, we find that simply equipping the base model with lightweight LoRA adapters achieves both efficient acceleration and strong generalization. We refer to these two adapters as Slow-LoRA and Fast-LoRA. Through extensive experiments, our method achieves up to 5 acceleration over the base model while maintaining comparable visual quality across diverse benchmarks. Remarkably, the LoRA experts are trained with only 1 samples on a single V100 within one hour, yet the resulting models generalize strongly on unseen prompts.
>
---
#### [new 125] EGGS: Exchangeable 2D/3D Gaussian Splatting for Geometry-Appearance Balanced Novel View Synthesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对新视角合成任务中3DGS几何不准、2DGS纹理损失的问题，提出EGGS框架，融合2D与3D高斯表示。通过混合光栅化、自适应类型切换和频域解耦优化，平衡了外观与几何精度，实现高效高质量渲染。**

- **链接: [https://arxiv.org/pdf/2512.02932v1](https://arxiv.org/pdf/2512.02932v1)**

> **作者:** Yancheng Zhang; Guangyu Sun; Chen Chen
>
> **摘要:** Novel view synthesis (NVS) is crucial in computer vision and graphics, with wide applications in AR, VR, and autonomous driving. While 3D Gaussian Splatting (3DGS) enables real-time rendering with high appearance fidelity, it suffers from multi-view inconsistencies, limiting geometric accuracy. In contrast, 2D Gaussian Splatting (2DGS) enforces multi-view consistency but compromises texture details. To address these limitations, we propose Exchangeable Gaussian Splatting (EGGS), a hybrid representation that integrates 2D and 3D Gaussians to balance appearance and geometry. To achieve this, we introduce Hybrid Gaussian Rasterization for unified rendering, Adaptive Type Exchange for dynamic adaptation between 2D and 3D Gaussians, and Frequency-Decoupled Optimization that effectively exploits the strengths of each type of Gaussian representation. Our CUDA-accelerated implementation ensures efficient training and inference. Extensive experiments demonstrate that EGGS outperforms existing methods in rendering quality, geometric accuracy, and efficiency, providing a practical solution for high-quality NVS.
>
---
#### [new 126] IC-World: In-Context Generation for Shared World Modeling
- **分类: cs.CV**

- **简介: 该论文提出IC-World，一种基于视频的世界建模框架，旨在从多视角输入图像并行生成一致的动态视频序列。针对共享世界建模中几何与运动一致性难题，利用大模型的上下文生成能力，并通过强化学习与新奖励模型优化，显著提升生成质量，是首个系统研究此问题的工作。**

- **链接: [https://arxiv.org/pdf/2512.02793v1](https://arxiv.org/pdf/2512.02793v1)**

> **作者:** Fan Wu; Jiacheng Wei; Ruibo Li; Yi Xu; Junyou Li; Deheng Ye; Guosheng Lin
>
> **备注:** codes:https://github.com/wufan-cse/IC-World
>
> **摘要:** Video-based world models have recently garnered increasing attention for their ability to synthesize diverse and dynamic visual environments. In this paper, we focus on shared world modeling, where a model generates multiple videos from a set of input images, each representing the same underlying world in different camera poses. We propose IC-World, a novel generation framework, enabling parallel generation for all input images via activating the inherent in-context generation capability of large video models. We further finetune IC-World via reinforcement learning, Group Relative Policy Optimization, together with two proposed novel reward models to enforce scene-level geometry consistency and object-level motion consistency among the set of generated videos. Extensive experiments demonstrate that IC-World substantially outperforms state-of-the-art methods in both geometry and motion consistency. To the best of our knowledge, this is the first work to systematically explore the shared world modeling problem with video-based world models.
>
---
#### [new 127] Learning Multimodal Embeddings for Traffic Accident Prediction and Causal Estimation
- **分类: cs.LG; cs.CV; cs.SI**

- **简介: 该论文针对交通肇事预测与因果分析任务，解决传统方法忽视环境信息的问题。构建了包含卫星图像与道路网络数据的多模态数据集，融合视觉与图结构特征，提升预测准确率（AUROC达90.1%），并揭示降水、车速和季节对事故的影响。**

- **链接: [https://arxiv.org/pdf/2512.02920v1](https://arxiv.org/pdf/2512.02920v1)**

> **作者:** Ziniu Zhang; Minxuan Duan; Haris N. Koutsopoulos; Hongyang R. Zhang
>
> **备注:** 17 pages. To appear in KDD'26 Datasets
>
> **摘要:** We consider analyzing traffic accident patterns using both road network data and satellite images aligned to road graph nodes. Previous work for predicting accident occurrences relies primarily on road network structural features while overlooking physical and environmental information from the road surface and its surroundings. In this work, we construct a large multimodal dataset across six U.S. states, containing nine million traffic accident records from official sources, and one million high-resolution satellite images for each node of the road network. Additionally, every node is annotated with features such as the region's weather statistics and road type (e.g., residential vs. motorway), and each edge is annotated with traffic volume information (i.e., Average Annual Daily Traffic). Utilizing this dataset, we conduct a comprehensive evaluation of multimodal learning methods that integrate both visual and network embeddings. Our findings show that integrating both data modalities improves prediction accuracy, achieving an average AUROC of $90.1\%$, which is a $3.7\%$ gain over graph neural network models that only utilize graph structures. With the improved embeddings, we conduct a causal analysis based on a matching estimator to estimate the key contributing factors influencing traffic accidents. We find that accident rates rise by $24\%$ under higher precipitation, by $22\%$ on higher-speed roads such as motorways, and by $29\%$ due to seasonal patterns, after adjusting for other confounding factors. Ablation studies confirm that satellite imagery features are essential for achieving accurate prediction.
>
---
#### [new 128] OmniGuard: Unified Omni-Modal Guardrails with Deliberate Reasoning
- **分类: cs.AI; cs.CL; cs.CR; cs.CV; cs.LG**

- **简介: 该论文针对多模态大模型的安全防护问题，提出OmniGuard统一框架，解决传统单模态安全机制在跨模态场景下泛化能力差的问题。通过构建超21万样本的多模态安全数据集，结合专家模型提炼安全判别与批判性反馈，实现对文本、图像、视频、音频等全模态的精细化安全管控。**

- **链接: [https://arxiv.org/pdf/2512.02306v1](https://arxiv.org/pdf/2512.02306v1)**

> **作者:** Boyu Zhu; Xiaofei Wen; Wenjie Jacky Mo; Tinghui Zhu; Yanan Xie; Peng Qi; Muhao Chen
>
> **摘要:** Omni-modal Large Language Models (OLLMs) that process text, images, videos, and audio introduce new challenges for safety and value guardrails in human-AI interaction. Prior guardrail research largely targets unimodal settings and typically frames safeguarding as binary classification, which limits robustness across diverse modalities and tasks. To address this gap, we propose OmniGuard, the first family of omni-modal guardrails that performs safeguarding across all modalities with deliberate reasoning ability. To support the training of OMNIGUARD, we curate a large, comprehensive omni-modal safety dataset comprising over 210K diverse samples, with inputs that cover all modalities through both unimodal and cross-modal samples. Each sample is annotated with structured safety labels and carefully curated safety critiques from expert models through targeted distillation. Extensive experiments on 15 benchmarks show that OmniGuard achieves strong effectiveness and generalization across a wide range of multimodal safety scenarios. Importantly, OmniGuard provides a unified framework that enforces policies and mitigates risks in omni-modalities, paving the way toward building more robust and capable omnimodal safeguarding systems.
>
---
#### [new 129] SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control
- **分类: cs.GR; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出SMP，一种可重用的基于评分匹配的运动先验方法，用于物理驱动角色控制。针对传统对抗性运动先验需为每种控制器重新训练、难以复用的问题，SMP利用预训练的运动扩散模型与评分蒸馏采样，构建任务无关的通用奖励函数，支持跨任务复用与风格组合，实现高质量自然运动生成。**

- **链接: [https://arxiv.org/pdf/2512.03028v1](https://arxiv.org/pdf/2512.03028v1)**

> **作者:** Yuxuan Mu; Ziyu Zhang; Yi Shi; Minami Matsumoto; Kotaro Imamura; Guy Tevet; Chuan Guo; Michael Taylor; Chang Shu; Pengcheng Xi; Xue Bin Peng
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Data-driven motion priors that can guide agents toward producing naturalistic behaviors play a pivotal role in creating life-like virtual characters. Adversarial imitation learning has been a highly effective method for learning motion priors from reference motion data. However, adversarial priors, with few exceptions, need to be retrained for each new controller, thereby limiting their reusability and necessitating the retention of the reference motion data when training on downstream tasks. In this work, we present Score-Matching Motion Priors (SMP), which leverages pre-trained motion diffusion models and score distillation sampling (SDS) to create reusable task-agnostic motion priors. SMPs can be pre-trained on a motion dataset, independent of any control policy or task. Once trained, SMPs can be kept frozen and reused as general-purpose reward functions to train policies to produce naturalistic behaviors for downstream tasks. We show that a general motion prior trained on large-scale datasets can be repurposed into a variety of style-specific priors. Furthermore SMP can compose different styles to synthesize new styles not present in the original dataset. Our method produces high-quality motion comparable to state-of-the-art adversarial imitation learning methods through reusable and modular motion priors. We demonstrate the effectiveness of SMP across a diverse suite of control tasks with physically simulated humanoid characters. Video demo available at https://youtu.be/ravlZJteS20
>
---
#### [new 130] Emergent Bayesian Behaviour and Optimal Cue Combination in LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; q-bio.NC**

- **简介: 该论文研究大模型在多模态感知中的隐式贝叶斯行为。通过构建心理物理学基准BayesBench，评估九个LLMs在四类感知任务中的不确定性处理与最优线索整合能力，发现模型准确率高但策略不鲁棒，揭示了准确性与贝叶斯一致性间的分离，提出一致性评分以识别隐含策略。**

- **链接: [https://arxiv.org/pdf/2512.02719v1](https://arxiv.org/pdf/2512.02719v1)**

> **作者:** Julian Ma; Jun Wang; Zafeirios Fountas
>
> **摘要:** Large language models (LLMs) excel at explicit reasoning, but their implicit computational strategies remain underexplored. Decades of psychophysics research show that humans intuitively process and integrate noisy signals using near-optimal Bayesian strategies in perceptual tasks. We ask whether LLMs exhibit similar behaviour and perform optimal multimodal integration without explicit training or instruction. Adopting the psychophysics paradigm, we infer computational principles of LLMs from systematic behavioural studies. We introduce a behavioural benchmark - BayesBench: four magnitude estimation tasks (length, location, distance, and duration) over text and image, inspired by classic psychophysics, and evaluate a diverse set of nine LLMs alongside human judgments for calibration. Through controlled ablations of noise, context, and instruction prompts, we measure performance, behaviour and efficiency in multimodal cue-combination. Beyond accuracy and efficiency metrics, we introduce a Bayesian Consistency Score that detects Bayes-consistent behavioural shifts even when accuracy saturates. Our results show that while capable models often adapt in Bayes-consistent ways, accuracy does not guarantee robustness. Notably, GPT-5 Mini achieves perfect text accuracy but fails to integrate visual cues efficiently. This reveals a critical dissociation between capability and strategy, suggesting accuracy-centric benchmarks may over-index on performance while missing brittle uncertainty handling. These findings reveal emergent principled handling of uncertainty and highlight the correlation between accuracy and Bayesian tendencies. We release our psychophysics benchmark and consistency metric (https://bayes-bench.github.io) as evaluation tools and to inform future multimodal architecture designs.
>
---
#### [new 131] Joint Distillation for Fast Likelihood Evaluation and Sampling in Flow-based Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对流模型中似然评估与采样计算成本高的问题，提出F2D2框架，通过联合蒸馏共享速度场的采样与对数似然轨迹，实现两者NFE减少两个数量级。方法模块化，兼容现有模型，仅增一额外头，显著提升效率并保持样本质量。**

- **链接: [https://arxiv.org/pdf/2512.02636v1](https://arxiv.org/pdf/2512.02636v1)**

> **作者:** Xinyue Ai; Yutong He; Albert Gu; Ruslan Salakhutdinov; J Zico Kolter; Nicholas Matthew Boffi; Max Simchowitz
>
> **摘要:** Log-likelihood evaluation enables important capabilities in generative models, including model comparison, certain fine-tuning objectives, and many downstream applications. Yet paradoxically, some of today's best generative models -- diffusion and flow-based models -- still require hundreds to thousands of neural function evaluations (NFEs) to compute a single likelihood. While recent distillation methods have successfully accelerated sampling to just a few steps, they achieve this at the cost of likelihood tractability: existing approaches either abandon likelihood computation entirely or still require expensive integration over full trajectories. We present fast flow joint distillation (F2D2), a framework that simultaneously reduces the number of NFEs required for both sampling and likelihood evaluation by two orders of magnitude. Our key insight is that in continuous normalizing flows, the coupled ODEs for sampling and likelihood are computed from a shared underlying velocity field, allowing us to jointly distill both the sampling trajectory and cumulative divergence using a single model. F2D2 is modular, compatible with existing flow-based few-step sampling models, and requires only an additional divergence prediction head. Experiments demonstrate F2D2's capability of achieving accurate log-likelihood with few-step evaluations while maintaining high sample quality, solving a long-standing computational bottleneck in flow-based generative models. As an application of our approach, we propose a lightweight self-guidance method that enables a 2-step MeanFlow model to outperform a 1024 step teacher model with only a single additional backward NFE.
>
---
#### [new 132] SAM2Grasp: Resolve Multi-modal Grasping via Prompt-conditioned Temporal Action Prediction
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人抓取中的多模态问题，提出SAM2Grasp框架。通过将任务重构为提示条件下的时序动作预测，利用SAM2的视觉时序追踪能力，仅训练轻量级动作头，实现对指定目标的稳定、唯一抓取轨迹预测，显著提升复杂场景下多物体抓取性能。**

- **链接: [https://arxiv.org/pdf/2512.02609v1](https://arxiv.org/pdf/2512.02609v1)**

> **作者:** Shengkai Wu; Jinrong Yang; Wenqiu Luo; Linfeng Gao; Chaohui Shang; Meiyu Zhi; Mingshan Sun; Fangping Yang; Liangliang Ren; Yong Zhao
>
> **摘要:** Imitation learning for robotic grasping is often plagued by the multimodal problem: when a scene contains multiple valid targets, demonstrations of grasping different objects create conflicting training signals. Standard imitation learning policies fail by averaging these distinct actions into a single, invalid action. In this paper, we introduce SAM2Grasp, a novel framework that resolves this issue by reformulating the task as a uni-modal, prompt-conditioned prediction problem. Our method leverages the frozen SAM2 model to use its powerful visual temporal tracking capability and introduces a lightweight, trainable action head that operates in parallel with its native segmentation head. This design allows for training only the small action head on pre-computed temporal-visual features from SAM2. During inference, an initial prompt, such as a bounding box provided by an upstream object detection model, designates the specific object to be grasped. This prompt conditions the action head to predict a unique, unambiguous grasp trajectory for that object alone. In all subsequent video frames, SAM2's built-in temporal tracking capability automatically maintains stable tracking of the selected object, enabling our model to continuously predict the grasp trajectory from the video stream without further external guidance. This temporal-prompted approach effectively eliminates ambiguity from the visuomotor policy. We demonstrate through extensive experiments that SAM2Grasp achieves state-of-the-art performance in cluttered, multi-object grasping tasks.
>
---
#### [new 133] Real-Time Multimodal Data Collection Using Smartwatches and Its Visualization in Education
- **分类: cs.HC; cs.CV; cs.SE**

- **简介: 该论文属于教育技术中的多模态学习分析任务，旨在解决教育场景下实时、同步、高分辨率多源数据采集难的问题。研究开发了Watch-DMLT与ViSeDOPS系统，实现对65名学生在课堂中生理、行为、视频等多模态数据的实时采集与可视化分析，验证了系统在真实教学环境中的可行性与有效性。**

- **链接: [https://arxiv.org/pdf/2512.02651v1](https://arxiv.org/pdf/2512.02651v1)**

> **作者:** Alvaro Becerra; Pablo Villegas; Ruth Cobos
>
> **备注:** Accepted in Technological Ecosystems for Enhancing Multiculturality (TEEM) 2025
>
> **摘要:** Wearable sensors, such as smartwatches, have become increasingly prevalent across domains like healthcare, sports, and education, enabling continuous monitoring of physiological and behavioral data. In the context of education, these technologies offer new opportunities to study cognitive and affective processes such as engagement, attention, and performance. However, the lack of scalable, synchronized, and high-resolution tools for multimodal data acquisition continues to be a significant barrier to the widespread adoption of Multimodal Learning Analytics in real-world educational settings. This paper presents two complementary tools developed to address these challenges: Watch-DMLT, a data acquisition application for Fitbit Sense 2 smartwatches that enables real-time, multi-user monitoring of physiological and motion signals; and ViSeDOPS, a dashboard-based visualization system for analyzing synchronized multimodal data collected during oral presentations. We report on a classroom deployment involving 65 students and up to 16 smartwatches, where data streams including heart rate, motion, gaze, video, and contextual annotations were captured and analyzed. Results demonstrate the feasibility and utility of the proposed system for supporting fine-grained, scalable, and interpretable Multimodal Learning Analytics in real learning environments.
>
---
#### [new 134] VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VIGS-SLAM，一种视觉惯性3D高斯点云SLAM系统，解决纯视觉方法在运动模糊、低纹理等条件下性能下降的问题。通过融合视觉与惯性信息，在统一优化框架中联合估计相机位姿、深度和IMU状态，实现鲁棒实时跟踪与高质量重建。**

- **链接: [https://arxiv.org/pdf/2512.02293v1](https://arxiv.org/pdf/2512.02293v1)**

> **作者:** Zihan Zhu; Wei Zhang; Norbert Haala; Marc Pollefeys; Daniel Barath
>
> **备注:** Project page: https://vigs-slam.github.io
>
> **摘要:** We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction. Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under motion blur, low texture, and exposure variations. Our method tightly couples visual and inertial cues within a unified optimization framework, jointly refining camera poses, depths, and IMU states. It features robust IMU initialization, time-varying bias modeling, and loop closure with consistent Gaussian updates. Experiments on four challenging datasets demonstrate our superiority over state-of-the-art methods. Project page: https://vigs-slam.github.io
>
---
#### [new 135] Superpixel Attack: Enhancing Black-box Adversarial Attack with Image-driven Division Areas
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文针对黑盒对抗攻击中扰动效率低的问题，提出基于超像素的分割区域与多策略搜索方法。通过图像驱动的超像素划分，实现更精准的扰动定位，显著提升攻击成功率（平均+2.10%），增强了对鲁棒模型的攻击能力。**

- **链接: [https://arxiv.org/pdf/2512.02062v1](https://arxiv.org/pdf/2512.02062v1)**

> **作者:** Issa Oe; Keiichiro Yamamura; Hiroki Ishikura; Ryo Hamahira; Katsuki Fujisawa
>
> **摘要:** Deep learning models are used in safety-critical tasks such as automated driving and face recognition. However, small perturbations in the model input can significantly change the predictions. Adversarial attacks are used to identify small perturbations that can lead to misclassifications. More powerful black-box adversarial attacks are required to develop more effective defenses. A promising approach to black-box adversarial attacks is to repeat the process of extracting a specific image area and changing the perturbations added to it. Existing attacks adopt simple rectangles as the areas where perturbations are changed in a single iteration. We propose applying superpixels instead, which achieve a good balance between color variance and compactness. We also propose a new search method, versatile search, and a novel attack method, Superpixel Attack, which applies superpixels and performs versatile search. Superpixel Attack improves attack success rates by an average of 2.10% compared with existing attacks. Most models used in this study are robust against adversarial attacks, and this improvement is significant for black-box adversarial attacks. The code is avilable at https://github.com/oe1307/SuperpixelAttack.git.
>
---
#### [new 136] Reasoning Path and Latent State Analysis for Multi-view Visual Spatial Reasoning: A Cognitive Science Perspective
- **分类: cs.AI; cs.CV**

- **简介: 该论文聚焦多视图视觉空间推理任务，针对当前视觉语言模型在跨视角一致性与几何保持上的不足，提出认知驱动的ReMindView-Bench基准。通过系统化设计多视图实验，揭示模型在信息整合阶段的性能退化与认知机制缺陷，为提升空间推理能力提供诊断与改进方向。**

- **链接: [https://arxiv.org/pdf/2512.02340v1](https://arxiv.org/pdf/2512.02340v1)**

> **作者:** Qiyao Xue; Weichen Liu; Shiqi Wang; Haoming Wang; Yuyang Wu; Wei Gao
>
> **备注:** 23 pages, 37 figures
>
> **摘要:** Spatial reasoning is a core aspect of human intelligence that allows perception, inference and planning in 3D environments. However, current vision-language models (VLMs) struggle to maintain geometric coherence and cross-view consistency for spatial reasoning in multi-view settings. We attribute this gap to the lack of fine-grained benchmarks that isolate multi-view reasoning from single-view perception and temporal factors. To address this, we present ReMindView-Bench, a cognitively grounded benchmark for evaluating how VLMs construct, align and maintain spatial mental models across complementary viewpoints. ReMindView-Bench systematically varies viewpoint spatial pattern and query type to probe key factors of spatial cognition. Evaluations of 15 current VLMs reveals consistent failures in cross-view alignment and perspective-taking in multi-view spatial reasoning, motivating deeper analysis on the reasoning process. Explicit phase-wise analysis using LLM-as-a-judge and self-consistency prompting shows that VLMs perform well on in-frame perception but degrade sharply when integrating information across views. Implicit analysis, including linear probing and entropy dynamics, further show progressive loss of task-relevant information and uncertainty separation between correct and incorrect trajectories. These results provide a cognitively grounded diagnosis of VLM spatial reasoning and reveal how multi-view spatial mental models are formed, degraded and destabilized across reasoning phases. The ReMindView-Bench benchmark is available at https://huggingface.co/datasets/Xue0823/ReMindView-Bench, and the source codes of benchmark construction and VLM reasoning analysis are available at https://github.com/pittisl/ReMindView-Bench.
>
---
#### [new 137] Bridging the Gap: Toward Cognitive Autonomy in Artificial Intelligence
- **分类: cs.AI; cs.CV**

- **简介: 该论文聚焦于人工智能认知自主性的缺失问题，指出当前AI在自监控、元认知、目标重构等方面存在七项核心缺陷。通过对比生物认知机制，提出需构建类神经认知架构以实现自主适应与目标导向行为，推动AI向具身化、可解释的自主系统演进。**

- **链接: [https://arxiv.org/pdf/2512.02280v1](https://arxiv.org/pdf/2512.02280v1)**

> **作者:** Noorbakhsh Amiri Golilarz; Sindhuja Penchala; Shahram Rahimi
>
> **摘要:** Artificial intelligence has advanced rapidly across perception, language, reasoning, and multimodal domains. Yet despite these achievements, modern AI systems remain fun- damentally limited in their ability to self-monitor, self-correct, and regulate their behavior autonomously in dynamic contexts. This paper identifies and analyzes seven core deficiencies that constrain contemporary AI models: the absence of intrinsic self- monitoring, lack of meta-cognitive awareness, fixed and non- adaptive learning mechanisms, inability to restructure goals, lack of representational maintenance, insufficient embodied feedback, and the absence of intrinsic agency. Alongside identifying these limitations, we also outline a forward-looking perspective on how AI may evolve beyond them through architectures that mirror neurocognitive principles. We argue that these structural limitations prevent current architectures, including deep learning and transformer-based systems, from achieving robust general- ization, lifelong adaptability, and real-world autonomy. Drawing on a comparative analysis of artificial systems and biological cognition [7], and integrating insights from AI research, cognitive science, and neuroscience, we outline how these capabilities are absent in current models and why scaling alone cannot resolve them. We conclude by advocating for a paradigmatic shift toward cognitively grounded AI (cognitive autonomy) capable of self-directed adaptation, dynamic representation management, and intentional, goal-oriented behavior, paired with reformative oversight mechanisms [8] that ensure autonomous systems remain interpretable, governable, and aligned with human values.
>
---
#### [new 138] Comparing Baseline and Day-1 Diffusion MRI Using Multimodal Deep Embeddings for Stroke Outcome Prediction
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于急性缺血性卒中预后预测任务，旨在比较基线（J0）与24小时（J1）弥散MRI对3个月功能结局的预测能力。研究融合三维ResNet-50影像嵌入与临床、病灶体积特征，通过降维与SVM分类，发现J1模型表现更优（AUC=0.923），验证了早期术后MRI的更高预后价值。**

- **链接: [https://arxiv.org/pdf/2512.02088v1](https://arxiv.org/pdf/2512.02088v1)**

> **作者:** Sina Raeisadigh; Myles Joshua Toledo Tan; Henning Müller; Abderrahmane Hedjoudje
>
> **备注:** 5 pages, 5 figures, 2 tables
>
> **摘要:** This study compares baseline (J0) and 24-hour (J1) diffusion magnetic resonance imaging (MRI) for predicting three-month functional outcomes after acute ischemic stroke (AIS). Seventy-four AIS patients with paired apparent diffusion coefficient (ADC) scans and clinical data were analyzed. Three-dimensional ResNet-50 embeddings were fused with structured clinical variables, reduced via principal component analysis (<=12 components), and classified using linear support vector machines with eight-fold stratified group cross-validation. J1 multimodal models achieved the highest predictive performance (AUC = 0.923 +/- 0.085), outperforming J0-based configurations (AUC <= 0.86). Incorporating lesion-volume features further improved model stability and interpretability. These findings demonstrate that early post-treatment diffusion MRI provides superior prognostic value to pre-treatment imaging and that combining MRI, clinical, and lesion-volume features produces a robust and interpretable framework for predicting three-month functional outcomes in AIS patients.
>
---
#### [new 139] CoatFusion: Controllable Material Coating in Images
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文提出“材料涂层”新任务，旨在在保留物体几何细节的前提下模拟薄层材料覆盖。针对现有方法无法精确控制涂层细节的问题，构建了110K张合成数据集DataCoat110K，提出CoatFusion模型，通过联合控制反照率与物理参数（如粗糙度、金属度等），实现高保真、可调控的图像材料涂层生成。**

- **链接: [https://arxiv.org/pdf/2512.02143v1](https://arxiv.org/pdf/2512.02143v1)**

> **作者:** Sagie Levy; Elad Aharoni; Matan Levy; Ariel Shamir; Dani Lischinski
>
> **摘要:** We introduce Material Coating, a novel image editing task that simulates applying a thin material layer onto an object while preserving its underlying coarse and fine geometry. Material coating is fundamentally different from existing "material transfer" methods, which are designed to replace an object's intrinsic material, often overwriting fine details. To address this new task, we construct a large-scale synthetic dataset (110K images) of 3D objects with varied, physically-based coatings, named DataCoat110K. We then propose CoatFusion, a novel architecture that enables this task by conditioning a diffusion model on both a 2D albedo texture and granular, PBR-style parametric controls, including roughness, metalness, transmission, and a key thickness parameter. Experiments and user studies show CoatFusion produces realistic, controllable coatings and significantly outperforms existing material editing and transfer methods on this new task.
>
---
#### [new 140] PhishSnap: Image-Based Phishing Detection Using Perceptual Hashing
- **分类: cs.CR; cs.CV; cs.LG**

- **简介: 该论文提出PhishSnap，一种基于感知哈希的图像化钓鱼网站检测系统，旨在解决传统方法在应对视觉欺骗和混淆时的不足。通过本地捕获网页截图并比对合法模板，实现隐私保护下的高效检测，验证了视觉相似性在反钓鱼中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.02243v1](https://arxiv.org/pdf/2512.02243v1)**

> **作者:** Md Abdul Ahad Minhaz; Zannatul Zahan Meem; Md. Shohrab Hossain
>
> **备注:** IEE Standard Formatting, 3 pages, 3 figures
>
> **摘要:** Phishing remains one of the most prevalent online threats, exploiting human trust to harvest sensitive credentials. Existing URL- and HTML-based detection systems struggle against obfuscation and visual deception. This paper presents \textbf{PhishSnap}, a privacy-preserving, on-device phishing detection system leveraging perceptual hashing (pHash). Implemented as a browser extension, PhishSnap captures webpage screenshots, computes visual hashes, and compares them against legitimate templates to identify visually similar phishing attempts. A \textbf{2024 dataset of 10,000 URLs} (70\%/20\%/10\% train/validation/test) was collected from PhishTank and Netcraft. Due to security takedowns, a subset of phishing pages was unavailable, reducing dataset diversity. The system achieved \textbf{0.79 accuracy}, \textbf{0.76 precision}, and \textbf{0.78 recall}, showing that visual similarity remains a viable anti-phishing measure. The entire inference process occurs locally, ensuring user privacy and minimal latency.
>
---
#### [new 141] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中失败诊断与学习难题，提出ViFailback框架，通过视觉符号提升故障诊断效率。构建了包含58,126个VQA对的真实世界操作数据集，并建立细粒度评估基准ViFailback-Bench。基于此，训练出ViFailback-8B模型，可生成可视化纠正指导，实现在真实场景中辅助机器人从失败中恢复。**

- **链接: [https://arxiv.org/pdf/2512.02787v1](https://arxiv.org/pdf/2512.02787v1)**

> **作者:** Xianchao Zeng; Xinyu Zhou; Youcheng Li; Jiayou Shi; Tianle Li; Liangming Chen; Lei Ren; Yong-Lu Li
>
> **摘要:** Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic manipulation, yet they remain limited in failure diagnosis and learning from failures. Additionally, existing failure datasets are mostly generated programmatically in simulation, which limits their generalization to the real world. In light of these, we introduce ViFailback, a framework designed to diagnose robotic manipulation failures and provide both textual and visual correction guidance. Our framework utilizes explicit visual symbols to enhance annotation efficiency. We further release the ViFailback dataset, a large-scale collection of 58,126 Visual Question Answering (VQA) pairs along with their corresponding 5,202 real-world manipulation trajectories. Based on the dataset, we establish ViFailback-Bench, a benchmark of 11 fine-grained VQA tasks designed to assess the failure diagnosis and correction abilities of Vision-Language Models (VLMs), featuring ViFailback-Bench Lite for closed-ended and ViFailback-Bench Hard for open-ended evaluation. To demonstrate the effectiveness of our framework, we built the ViFailback-8B VLM, which not only achieves significant overall performance improvement on ViFailback-Bench but also generates visual symbols for corrective action guidance. Finally, by integrating ViFailback-8B with a VLA model, we conduct real-world robotic experiments demonstrating its ability to assist the VLA model in recovering from failures. Project Website: https://x1nyuzhou.github.io/vifailback.github.io/
>
---
## 更新

#### [replaced 001] Diffusion-SDPO: Safeguarded Direct Preference Optimization for Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.03317v2](https://arxiv.org/pdf/2511.03317v2)**

> **作者:** Minghao Fu; Guo-Hua Wang; Tianyu Cui; Qing-Guo Chen; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** The code is publicly available at https://github.com/AIDC-AI/Diffusion-SDPO
>
> **摘要:** Text-to-image diffusion models deliver high-quality images, yet aligning them with human preferences remains challenging. We revisit diffusion-based Direct Preference Optimization (DPO) for these models and identify a critical pathology: enlarging the preference margin does not necessarily improve generation quality. In particular, the standard Diffusion-DPO objective can increase the reconstruction error of both winner and loser branches. Consequently, degradation of the less-preferred outputs can become sufficiently severe that the preferred branch is also adversely affected even as the margin grows. To address this, we introduce Diffusion-SDPO, a safeguarded update rule that preserves the winner by adaptively scaling the loser gradient according to its alignment with the winner gradient. A first-order analysis yields a closed-form scaling coefficient that guarantees the error of the preferred output is non-increasing at each optimization step. Our method is simple, model-agnostic, broadly compatible with existing DPO-style alignment frameworks and adds only marginal computational overhead. Across standard text-to-image benchmarks, Diffusion-SDPO delivers consistent gains over preference-learning baselines on automated preference, aesthetic, and prompt alignment metrics. Code is publicly available at https://github.com/AIDC-AI/Diffusion-SDPO.
>
---
#### [replaced 002] Diffusion Model in Latent Space for Medical Image Segmentation Task
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.01292v2](https://arxiv.org/pdf/2512.01292v2)**

> **作者:** Huynh Trinh Ngoc; Toan Nguyen Hai; Ba Luong Son; Long Tran Quoc
>
> **摘要:** Medical image segmentation is crucial for clinical diagnosis and treatment planning. Traditional methods typically produce a single segmentation mask, failing to capture inherent uncertainty. Recent generative models enable the creation of multiple plausible masks per image, mimicking the collaborative interpretation of several clinicians. However, these approaches remain computationally heavy. We propose MedSegLatDiff, a diffusion based framework that combines a variational autoencoder (VAE) with a latent diffusion model for efficient medical image segmentation. The VAE compresses the input into a low dimensional latent space, reducing noise and accelerating training, while the diffusion process operates directly in this compact representation. We further replace the conventional MSE loss with weighted cross entropy in the VAE mask reconstruction path to better preserve tiny structures such as small nodules. MedSegLatDiff is evaluated on ISIC-2018 (skin lesions), CVC-Clinic (polyps), and LIDC-IDRI (lung nodules). It achieves state of the art or highly competitive Dice and IoU scores while simultaneously generating diverse segmentation hypotheses and confidence maps. This provides enhanced interpretability and reliability compared to deterministic baselines, making the model particularly suitable for clinical deployment.
>
---
#### [replaced 003] PRIMU: Uncertainty Estimation for Novel Views in Gaussian Splatting from Primitive-Based Representations of Error and Coverage
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02443v2](https://arxiv.org/pdf/2508.02443v2)**

> **作者:** Thomas Gottwald; Edgar Heinert; Peter Stehr; Chamuditha Jayanga Galappaththige; Matthias Rottmann
>
> **备注:** Revised writing and figures; additional Gaussian Splatting experiments; added baselines and datasets; active view-selection experiments
>
> **摘要:** We introduce Primitive-based Representations of Uncertainty (PRIMU), a post-hoc uncertainty estimation (UE) framework for Gaussian Splatting (GS). Reliable UE is essential for deploying GS in safety-critical domains such as robotics and medicine. Existing approaches typically estimate Gaussian-primitive variances and rely on the rendering process to obtain pixel-wise uncertainties. In contrast, we construct primitive-level representations of error and visibility/coverage from training views, capturing interpretable uncertainty information. These representations are obtained by projecting view-dependent training errors and coverage statistics onto the primitives. Uncertainties for novel views are inferred by rendering these primitive-level representations, producing uncertainty feature maps, which are aggregate through pixel-wise regression on holdout data. We analyze combinations of uncertainty feature maps and regression models to understand how their interactions affect prediction accuracy and generalization. PRIMU also enables an effective active view selection strategy by directly leveraging these uncertainty feature maps. Additionally, we study the effect of separating splatting into foreground and background regions. Our estimates show strong correlations with true errors, outperforming state-of-the-art methods, especially for depth UE and foreground objects. Finally, our regression models show generalization capabilities to unseen scenes, enabling UE without additional holdout data.
>
---
#### [replaced 004] Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18507v2](https://arxiv.org/pdf/2511.18507v2)**

> **作者:** Kai Jiang; Siqi Huang; Xiangyu Chen; Jiawei Shao; Hongyuan Zhang; Xuelong Li
>
> **备注:** 18 pages, 16 figures. This is a preprint version of a paper submitted to CVPR 2026
>
> **摘要:** Continual learning in visual understanding aims to deal with catastrophic forgetting in Multimodal Large Language Models (MLLMs). MLLMs deployed on devices have to continuously adapt to dynamic scenarios in downstream tasks, such as variations in background and perspective, to effectively perform complex visual tasks. To this end, we construct a multimodal visual understanding dataset (MSVQA) encompassing four different scenarios and perspectives including high altitude, underwater, low altitude and indoor, to investigate the catastrophic forgetting in MLLMs under the dynamics of scenario shifts in real-world data streams. Furthermore, we propose mUltimodal coNtInual learning with MLLMs From multi-scenarIo pERspectives (UNIFIER) to address visual discrepancies while learning different scenarios. Specifically, it decouples the visual information from different scenarios into distinct branches within each vision block and projects them into the same feature space. A consistency constraint is imposed on the features of each branch to maintain the stability of visual representations across scenarios. Extensive experiments on the MSVQA dataset demonstrate that UNIFIER effectively alleviates forgetting of cross-scenario tasks and achieves knowledge accumulation within the same scenario.
>
---
#### [replaced 005] Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17722v2](https://arxiv.org/pdf/2511.17722v2)**

> **作者:** Saurav Sengupta; Nazanin Moradinasab; Jiebei Liu; Donald E. Brown
>
> **摘要:** Recent research suggests that Vision Language Models (VLMs) often rely on inherent biases learned during training when responding to queries about visual properties of images. These biases are exacerbated when VLMs are asked highly specific questions that require them to focus on particular areas of the image in tasks such as counting. We build upon this research by developing a synthetic benchmark dataset and evaluation framework to systematically determine how counting performance varies as image and prompt properties change. Using open-source VLMs, we then analyze how attention allocation fluctuates with varying input parameters (e.g. number of objects in the image, objects color, background color, objects texture, background texture, and prompt specificity). We further implement attention-based interventions to modulate focus on visual tokens at different layers and evaluate their impact on counting performance across a range of visual conditions. Our experiments reveal that while VLM counting performance remains challenging, especially under high visual or linguistic complexity, certain attention interventions can lead to modest gains in counting performance.
>
---
#### [replaced 006] Aligning Diffusion Models with Noise-Conditioned Perception
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2406.17636v2](https://arxiv.org/pdf/2406.17636v2)**

> **作者:** Alexander Gambashidze; Anton Kulikov; Yuriy Sosnin; Ilya Makarov
>
> **摘要:** Recent advancements in human preference optimization, initially developed for Language Models (LMs), have shown promise for text-to-image Diffusion Models, enhancing prompt alignment, visual appeal, and user preference. Unlike LMs, Diffusion Models typically optimize in pixel or VAE space, which does not align well with human perception, leading to slower and less efficient training during the preference alignment stage. We propose using a perceptual objective in the U-Net embedding space of the diffusion model to address these issues. Our approach involves fine-tuning Stable Diffusion 1.5 and XL using Direct Preference Optimization (DPO), Contrastive Preference Optimization (CPO), and supervised fine-tuning (SFT) within this embedding space. This method significantly outperforms standard latent-space implementations across various metrics, including quality and computational cost. For SDXL, our approach provides 60.8\% general preference, 62.2\% visual appeal, and 52.1\% prompt following against original open-sourced SDXL-DPO on the PartiPrompts dataset, while significantly reducing compute. Our approach not only improves the efficiency and quality of human preference alignment for diffusion models but is also easily integrable with other optimization techniques. The training code and LoRA weights will be available here: https://huggingface.co/alexgambashidze/SDXL\_NCP-DPO\_v0.1
>
---
#### [replaced 007] Image-Based Relocalization and Alignment for Long-Term Monitoring of Dynamic Underwater Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对长期动态水下环境监测中视觉定位困难的问题，提出融合VPR、特征匹配与图像分割的重定位与对齐方法。旨在实现水下场景的精准回溯与变化分析，并构建首个大规模跨时序水下VPR基准数据集SQUIDLE+，推动自动化水下生态监测发展。**

- **链接: [https://arxiv.org/pdf/2503.04096v2](https://arxiv.org/pdf/2503.04096v2)**

> **作者:** Beverley Gorry; Tobias Fischer; Michael Milford; Alejandro Fontan
>
> **摘要:** Effective monitoring of underwater ecosystems is crucial for tracking environmental changes, guiding conservation efforts, and ensuring long-term ecosystem health. However, automating underwater ecosystem management with robotic platforms remains challenging due to the complexities of underwater imagery, which pose significant difficulties for traditional visual localization methods. We propose an integrated pipeline that combines Visual Place Recognition (VPR), feature matching, and image segmentation on video-derived images. This method enables robust identification of revisited areas, estimation of rigid transformations, and downstream analysis of ecosystem changes. Furthermore, we introduce the SQUIDLE+ VPR Benchmark-the first large-scale underwater VPR benchmark designed to leverage an extensive collection of unstructured data from multiple robotic platforms, spanning time intervals from days to years. The dataset encompasses diverse trajectories, arbitrary overlap and diverse seafloor types captured under varying environmental conditions, including differences in depth, lighting, and turbidity. Our code is available at: https://github.com/bev-gorry/underloc
>
---
#### [replaced 008] Multimodal LLMs See Sentiment
- **分类: cs.CV; cs.SI**

- **链接: [https://arxiv.org/pdf/2508.16873v2](https://arxiv.org/pdf/2508.16873v2)**

> **作者:** Neemias B. da Silva; John Harrison; Rodrigo Minetto; Myriam R. Delgado; Bogdan T. Nassu; Thiago H. Silva
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Understanding how visual content communicates sentiment is critical in an era where online interaction is increasingly dominated by this kind of media on social platforms. However, this remains a challenging problem, as sentiment perception is closely tied to complex, scene-level semantics. In this paper, we propose an original framework, MLLMsent, to investigate the sentiment reasoning capabilities of Multimodal Large Language Models (MLLMs) through three perspectives: (1) using those MLLMs for direct sentiment classification from images; (2) associating them with pre-trained LLMs for sentiment analysis on automatically generated image descriptions; and (3) fine-tuning the LLMs on sentiment-labeled image descriptions. Experiments on a recent and established benchmark demonstrate that our proposal, particularly the fine-tuned approach, achieves state-of-the-art results outperforming Lexicon-, CNN-, and Transformer-based baselines by up to 30.9%, 64.8%, and 42.4%, respectively, across different levels of evaluators' agreement and sentiment polarity categories. Remarkably, in a cross-dataset test, without any training on these new data, our model still outperforms, by up to 8.26%, the best runner-up, which has been trained directly on them. These results highlight the potential of the proposed visual reasoning scheme for advancing affective computing, while also establishing new benchmarks for future research.
>
---
#### [replaced 009] ST-Booster: An Iterative SpatioTemporal Perception Booster for Vision-and-Language Navigation in Continuous Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究视觉-语言导航（VLN-CE）任务，针对连续环境中视觉记忆异构与三维结构噪声问题，提出ST-Booster模型。通过分层时空编码、多粒度对齐融合与价值引导路径生成，实现指令感知的迭代优化，显著提升复杂环境下的导航性能。**

- **链接: [https://arxiv.org/pdf/2504.09843v2](https://arxiv.org/pdf/2504.09843v2)**

> **作者:** Lu Yue; Dongliang Zhou; Liang Xie; Erwei Yin; Feitian Zhang
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires agents to navigate unknown, continuous spaces based on natural language instructions. Compared to discrete settings, VLN-CE poses two core perception challenges. First, the absence of predefined observation points leads to heterogeneous visual memories and weakened global spatial correlations. Second, cumulative reconstruction errors in three-dimensional scenes introduce structural noise, impairing local feature perception. To address these challenges, this paper proposes ST-Booster, an iterative spatiotemporal booster that enhances navigation performance through multi-granularity perception and instruction-aware reasoning. ST-Booster consists of three key modules -- Hierarchical SpatioTemporal Encoding (HSTE), Multi-Granularity Aligned Fusion (MGAF), and ValueGuided Waypoint Generation (VGWG). HSTE encodes long-term global memory using topological graphs and captures shortterm local details via grid maps. MGAF aligns these dualmap representations with instructions through geometry-aware knowledge fusion. The resulting representations are iteratively refined through pretraining tasks. During reasoning, VGWG generates Guided Attention Heatmaps (GAHs) to explicitly model environment-instruction relevance and optimize waypoint selection. Extensive comparative experiments and performance analyses are conducted, demonstrating that ST-Booster outperforms existing state-of-the-art methods, particularly in complex, disturbance-prone environments.
>
---
#### [replaced 010] ReSpace: Text-Driven 3D Indoor Scene Synthesis and Editing with Preference Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.02459v4](https://arxiv.org/pdf/2506.02459v4)**

> **作者:** Martin JJ. Bucher; Iro Armeni
>
> **备注:** 25 pages, 18 figures (incl. appendix)
>
> **摘要:** Scene synthesis and editing has emerged as a promising direction in computer graphics. Current trained approaches for 3D indoor scenes either oversimplify object semantics through one-hot class encodings (e.g., 'chair' or 'table'), require masked diffusion for editing, ignore room boundaries, or rely on floor plan renderings that fail to capture complex layouts. LLM-based methods enable richer semantics via natural language (e.g., 'modern studio with light wood furniture'), but lack editing functionality, are limited to rectangular layouts, or rely on weak spatial reasoning from implicit world models. We introduce ReSpace, a generative framework for text-driven 3D indoor scene synthesis and editing using autoregressive language models. Our approach features a compact structured scene representation with explicit room boundaries that enables asset-agnostic deployment and frames scene editing as a next-token prediction task. We leverage a dual-stage training approach combining supervised fine-tuning and preference alignment, enabling a specially trained language model for object addition that accounts for user instructions, spatial geometry, object semantics, and scene-level composition. For scene editing, we employ a zero-shot LLM to handle object removal and prompts for addition. We further introduce a voxelization-based evaluation capturing fine-grained geometry beyond 3D bounding boxes. Experimental results surpass state-of-the-art on addition and achieve superior human-perceived quality on full scene synthesis.
>
---
#### [replaced 011] Zero-shot self-supervised learning of single breath-hold magnetic resonance cholangiopancreatography (MRCP) reconstruction
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.09200v2](https://arxiv.org/pdf/2508.09200v2)**

> **作者:** Jinho Kim; Marcel Dominik Nickel; Florian Knoll
>
> **备注:** 24 pages, 8 figures, 2 tabels
>
> **摘要:** To investigate the feasibility of zero-shot self-supervised learning reconstruction for reducing breath-hold times in magnetic resonance cholangiopancreatography (MRCP). Breath-hold MRCP was acquired from 11 healthy volunteers on 3T scanners using an incoherent k-space sampling pattern, leading to 14-second acquisition time and an acceleration factor of R=25. Zero-shot reconstruction was compared with parallel imaging of respiratory-triggered MRCP (338s, R=3) and compressed sensing reconstruction. For two volunteers, breath-hold scans (40s, R=6) were additionally acquired and retrospectively undersampled to R=25 to compute peak signal-to-noise ratio (PSNR). To address long zero-shot training time, the n+m full stages of the zero-shot learning were divided into two parts to reduce backpropagation depth during training: 1) n frozen stages initialized with n-stage pretrained network and 2) m trainable stages initialized either randomly or m-stage pretrained network. Efficiency of our approach was assessed by varying initialization strategies and the number of trainable stages using the retrospectively undersampled data. Zero-shot reconstruction significantly improved visual image quality over compressed sensing, particularly in SNR and ductal delineation, and achieved image quality comparable to that of successful respiratory-triggered acquisitions with regular breathing patterns. Improved initializations enhanced PSNR and reduced reconstruction time. Adjusting frozen/trainable configurations demonstrated that PSNR decreased only slightly from 38.25 dB (0/13) to 37.67 dB (12/1), while training time decreased up to 6.7-fold. Zero-shot learning delivers high-fidelity MRCP reconstructions with reduced breath-hold times, and the proposed partially trainable approach offers a practical solution for translation into time-constrained clinical workflows.
>
---
#### [replaced 012] Unleashing Hour-Scale Video Training for Long Video-Language Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对长视频语言理解中训练数据稀缺问题，提出VideoMarathon数据集（9700小时，3-60分钟视频）和Hour-LLaVA模型。通过支持小时级视频训练与推理，解决长期视频理解难题，实现多任务高效建模，显著提升长视频语言理解性能。**

- **链接: [https://arxiv.org/pdf/2506.05332v2](https://arxiv.org/pdf/2506.05332v2)**

> **作者:** Jingyang Lin; Jialian Wu; Ximeng Sun; Ze Wang; Jiang Liu; Yusheng Su; Xiaodong Yu; Hao Chen; Jiebo Luo; Zicheng Liu; Emad Barsoum
>
> **备注:** NeurIPS 2025, Project page: https://videomarathon.github.io/
>
> **摘要:** Recent long-form video-language understanding benchmarks have driven progress in video large multimodal models (Video-LMMs). However, the scarcity of well-annotated long videos has left the training of hour-long Video-LMMs underexplored. To close this gap, we present VideoMarathon, a large-scale hour-long video instruction-following dataset. This dataset includes around 9,700 hours of long videos sourced from diverse domains, ranging from 3 to 60 minutes per video. Specifically, it contains 3.3M high-quality QA pairs, spanning six fundamental topics: temporality, spatiality, object, action, scene, and event. Compared to existing video instruction datasets, VideoMarathon significantly extends training video durations up to 1 hour, and supports 22 diverse tasks requiring both short- and long-term video comprehension. Building on VideoMarathon, we propose Hour-LLaVA, a powerful and efficient Video-LMM for hour-scale video-language modeling. It enables hour-long video training and inference at 1-FPS sampling by leveraging a memory augmentation module, which adaptively integrates question-relevant and spatiotemporally informative semantics from the cached full video context. In our experiments, Hour-LLaVA achieves the best performance on multiple representative long video-language benchmarks, demonstrating the high quality of the VideoMarathon dataset and the superiority of the Hour-LLaVA model.
>
---
#### [replaced 013] WorldMem: Long-term Consistent World Simulation with Memory
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.12369v2](https://arxiv.org/pdf/2504.12369v2)**

> **作者:** Zeqi Xiao; Yushi Lan; Yifan Zhou; Wenqi Ouyang; Shuai Yang; Yanhong Zeng; Xingang Pan
>
> **备注:** Project page at https://xizaoqu.github.io/worldmem/
>
> **摘要:** World simulation has gained increasing popularity due to its ability to model virtual environments and predict the consequences of actions. However, the limited temporal context window often leads to failures in maintaining long-term consistency, particularly in preserving 3D spatial consistency. In this work, we present WorldMem, a framework that enhances scene generation with a memory bank consisting of memory units that store memory frames and states (e.g., poses and timestamps). By employing a memory attention mechanism that effectively extracts relevant information from these memory frames based on their states, our method is capable of accurately reconstructing previously observed scenes, even under significant viewpoint or temporal gaps. Furthermore, by incorporating timestamps into the states, our framework not only models a static world but also captures its dynamic evolution over time, enabling both perception and interaction within the simulated world. Extensive experiments in both virtual and real scenarios validate the effectiveness of our approach.
>
---
#### [replaced 014] SkeletonAgent: An Agentic Interaction Framework for Skeleton-based Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22433v2](https://arxiv.org/pdf/2511.22433v2)**

> **作者:** Hongda Liu; Yunfan Liu; Changlu Wang; Yunlong Wang; Zhenan Sun
>
> **摘要:** Recent advances in skeleton-based action recognition increasingly leverage semantic priors from Large Language Models (LLMs) to enrich skeletal representations. However, the LLM is typically queried in isolation from the recognition model and receives no performance feedback. As a result, it often fails to deliver the targeted discriminative cues critical to distinguish similar actions. To overcome these limitations, we propose SkeletonAgent, a novel framework that bridges the recognition model and the LLM through two cooperative agents, i.e., Questioner and Selector. Specifically, the Questioner identifies the most frequently confused classes and supplies them to the LLM as context for more targeted guidance. Conversely, the Selector parses the LLM's response to extract precise joint-level constraints and feeds them back to the recognizer, enabling finer-grained cross-modal alignment. Comprehensive evaluations on five benchmarks, including NTU RGB+D, NTU RGB+D 120, Kinetics-Skeleton, FineGYM, and UAV-Human, demonstrate that SkeletonAgent consistently outperforms state-of-the-art benchmark methods. The code is available at https://github.com/firework8/SkeletonAgent.
>
---
#### [replaced 015] Rainbow Noise: Stress-Testing Multimodal Harmful-Meme Detectors on LGBTQ Content
- **分类: cs.CY; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.19551v3](https://arxiv.org/pdf/2507.19551v3)**

> **作者:** Ran Tong; Songtao Wei; Jiaqi Liu; Lanruo Wang
>
> **备注:** 14 pages, 1 figure
>
> **摘要:** Hateful memes aimed at LGBTQ\,+ communities often evade detection by tweaking either the caption, the image, or both. We build the first robustness benchmark for this setting, pairing four realistic caption attacks with three canonical image corruptions and testing all combinations on the PrideMM dataset. Two state-of-the-art detectors, MemeCLIP and MemeBLIP2, serve as case studies, and we introduce a lightweight \textbf{Text Denoising Adapter (TDA)} to enhance the latter's resilience. Across the grid, MemeCLIP degrades more gently, while MemeBLIP2 is particularly sensitive to the caption edits that disrupt its language processing. However, the addition of the TDA not only remedies this weakness but makes MemeBLIP2 the most robust model overall. Ablations reveal that all systems lean heavily on text, but architectural choices and pre-training data significantly impact robustness. Our benchmark exposes where current multimodal safety models crack and demonstrates that targeted, lightweight modules like the TDA offer a powerful path towards stronger defences.
>
---
#### [replaced 016] ROGR: Relightable 3D Objects using Generative Relighting
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2510.03163v2](https://arxiv.org/pdf/2510.03163v2)**

> **作者:** Jiapeng Tang; Matthew Lavine; Dor Verbin; Stephan J. Garbin; Matthias Nießner; Ricardo Martin Brualla; Pratul P. Srinivasan; Philipp Henzler
>
> **备注:** NeurIPS 2025 Spotlight. Project page: https://tangjiapeng.github.io/ROGR
>
> **摘要:** We introduce ROGR, a novel approach that reconstructs a relightable 3D model of an object captured from multiple views, driven by a generative relighting model that simulates the effects of placing the object under novel environment illuminations. Our method samples the appearance of the object under multiple lighting environments, creating a dataset that is used to train a lighting-conditioned Neural Radiance Field (NeRF) that outputs the object's appearance under any input environmental lighting. The lighting-conditioned NeRF uses a novel dual-branch architecture to encode the general lighting effects and specularities separately. The optimized lighting-conditioned NeRF enables efficient feed-forward relighting under arbitrary environment maps without requiring per-illumination optimization or light transport simulation. We evaluate our approach on the established TensoIR and Stanford-ORB datasets, where it improves upon the state-of-the-art on most metrics, and showcase our approach on real-world object captures.
>
---
#### [replaced 017] MegaSR: Mining Customized Semantics and Expressive Guidance for Real-World Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.08096v2](https://arxiv.org/pdf/2503.08096v2)**

> **作者:** Xinrui Li; Jinrong Zhang; Jianlong Wu; Chong Chen; Liqiang Nie; Zhouchen Lin
>
> **摘要:** Text-to-image (T2I) models have ushered in a new era of real-world image super-resolution (Real-ISR) due to their rich internal implicit knowledge for multimodal learning. Although bringing high-level semantic priors and dense pixel guidance have led to advances in reconstruction, we identified several critical phenomena by analyzing the behavior of existing T2I-based Real-ISR methods: (1) Fine detail deficiency, which ultimately leads to incorrect reconstruction in local regions. (2) Block-wise semantic inconsistency, which results in distracted semantic interpretations across U-Net blocks. (3) Edge ambiguity, which causes noticeable structural degradation. Building upon these observations, we first introduce MegaSR, which enhances the T2I-based Real-ISR models with fine-grained customized semantics and expressive guidance to unlock semantically rich and structurally consistent reconstruction. Then, we propose the Customized Semantics Module (CSM) to supplement fine-grained semantics from the image modality and regulate the semantic fusion between multi-level knowledge to realize customization for different U-Net blocks. Besides the semantic adaptation, we identify expressive multimodal signals through pair-wise comparisons and introduce the Multimodal Signal Fusion Module (MSFM) to aggregate them for structurally consistent reconstruction. Extensive experiments on real-world and synthetic datasets demonstrate the superiority of the method. Notably, it not only achieves state-of-the-art performance on quality-driven metrics but also remains competitive on fidelity-focused metrics, striking a balance between perceptual realism and faithful content reconstruction.
>
---
#### [replaced 018] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16024v3](https://arxiv.org/pdf/2511.16024v3)**

> **作者:** Xiao He; Zhijun Tu; Kun Cheng; Mingrui Zhu; Jie Hu; Nannan Wang; Xinbo Gao
>
> **备注:** 16 pages, Accepted by AAAI 2026, v2: corrected typos
>
> **摘要:** The demonstrated success of sparsely-gated Mixture-of-Experts (MoE) architectures, exemplified by models such as DeepSeek and Grok, has motivated researchers to investigate their adaptation to diverse domains. In real-world image super-resolution (Real-ISR), existing approaches mainly rely on fine-tuning pre-trained diffusion models through Low-Rank Adaptation (LoRA) module to reconstruct high-resolution (HR) images. However, these dense Real-ISR models are limited in their ability to adaptively capture the heterogeneous characteristics of complex real-world degraded samples or enable knowledge sharing between inputs under equivalent computational budgets. To address this, we investigate the integration of sparse MoE into Real-ISR and propose a Mixture-of-Ranks (MoR) architecture for single-step image super-resolution. We introduce a fine-grained expert partitioning strategy that treats each rank in LoRA as an independent expert. This design enables flexible knowledge recombination while isolating fixed-position ranks as shared experts to preserve common-sense features and minimize routing redundancy. Furthermore, we develop a degradation estimation module leveraging CLIP embeddings and predefined positive-negative text pairs to compute relative degradation scores, dynamically guiding expert activation. To better accommodate varying sample complexities, we incorporate zero-expert slots and propose a degradation-aware load-balancing loss, which dynamically adjusts the number of active experts based on degradation severity, ensuring optimal computational resource allocation. Comprehensive experiments validate our framework's effectiveness and state-of-the-art performance.
>
---
#### [replaced 019] OpenLVLM-MIA: A Controlled Benchmark Revealing the Limits of Membership Inference Attacks on Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.16295v2](https://arxiv.org/pdf/2510.16295v2)**

> **作者:** Ryoto Miyamoto; Xin Fan; Fuyuko Kido; Tsuneo Matsumoto; Hayato Yamana
>
> **备注:** WACV2026 Accepted
>
> **摘要:** OpenLVLM-MIA is a new benchmark that highlights fundamental challenges in evaluating membership inference attacks (MIA) against large vision-language models (LVLMs). While prior work has reported high attack success rates, our analysis suggests that these results often arise from detecting distributional bias introduced during dataset construction rather than from identifying true membership status. To address this issue, we introduce a controlled benchmark of 6{,}000 images where the distributions of member and non-member samples are carefully balanced, and ground-truth membership labels are provided across three distinct training stages. Experiments using OpenLVLM-MIA demonstrated that the performance of state-of-the-art MIA methods approached chance-level. OpenLVLM-MIA, designed to be transparent and unbiased benchmark, clarifies certain limitations of MIA research on LVLMs and provides a solid foundation for developing stronger privacy-preserving techniques.
>
---
#### [replaced 020] Toward Content-based Indexing and Retrieval of Head and Neck CT with Abscess Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01589v2](https://arxiv.org/pdf/2512.01589v2)**

> **作者:** Thao Thi Phuong Dao; Tan-Cong Nguyen; Trong-Le Do; Truong Hoang Viet; Nguyen Chi Thanh; Huynh Nguyen Thuan; Do Vo Cong Nguyen; Minh-Khoi Pham; Mai-Khiem Tran; Viet-Tham Huynh; Trong-Thuan Nguyen; Trung-Nghia Le; Vo Thanh Toan; Tam V. Nguyen; Minh-Triet Tran; Thanh Dinh Le
>
> **备注:** The 2025 IEEE International Conference on Content-Based Multimedia Indexing (IEEE CBMI)
>
> **摘要:** Abscesses in the head and neck represent an acute infectious process that can potentially lead to sepsis or mortality if not diagnosed and managed promptly. Accurate detection and delineation of these lesions on imaging are essential for diagnosis, treatment planning, and surgical intervention. In this study, we introduce AbscessHeNe, a curated and comprehensively annotated dataset comprising 4,926 contrast-enhanced CT slices with clinically confirmed head and neck abscesses. The dataset is designed to facilitate the development of robust semantic segmentation models that can accurately delineate abscess boundaries and evaluate deep neck space involvement, thereby supporting informed clinical decision-making. To establish performance baselines, we evaluate several state-of-the-art segmentation architectures, including CNN, Transformer, and Mamba-based models. The highest-performing model achieved a Dice Similarity Coefficient of 0.39, Intersection-over-Union of 0.27, and Normalized Surface Distance of 0.67, indicating the challenges of this task and the need for further research. Beyond segmentation, AbscessHeNe is structured for future applications in content-based multimedia indexing and case-based retrieval. Each CT scan is linked with pixel-level annotations and clinical metadata, providing a foundation for building intelligent retrieval systems and supporting knowledge-driven clinical workflows. The dataset will be made publicly available at https://github.com/drthaodao3101/AbscessHeNe.git.
>
---
#### [replaced 021] AIDEN: Design and Pilot Study of an AI Assistant for the Visually Impaired
- **分类: cs.CV; cs.CY; cs.HC**

- **链接: [https://arxiv.org/pdf/2511.06080v3](https://arxiv.org/pdf/2511.06080v3)**

> **作者:** Luis Marquez-Carpintero; Francisco Gomez-Donoso; Zuria Bauer; Bessie Dominguez-Dager; Alvaro Belmonte-Baeza; Mónica Pina-Navarro; Francisco Morillas-Espejo; Felix Escalona; Miguel Cazorla
>
> **摘要:** This paper presents AIDEN, an artificial intelligence-based assistant designed to enhance the autonomy and daily quality of life of visually impaired individuals, who often struggle with object identification, text reading, and navigation in unfamiliar environments. Existing solutions such as screen readers or audio-based assistants facilitate access to information but frequently lead to auditory overload and raise privacy concerns in open environments. AIDEN addresses these limitations with a hybrid architecture that integrates You Only Look Once (YOLO) for real-time object detection and a Large Language and Vision Assistant (LLaVA) for scene description and Optical Character Recognition (OCR). A key novelty of the system is a continuous haptic guidance mechanism based on a Geiger-counter metaphor, which supports object centering without occupying the auditory channel, while privacy is preserved by ensuring that no personal data are stored. Empirical evaluations with visually impaired participants assessed perceived ease of use and acceptance using the Technology Acceptance Model (TAM). Results indicate high user satisfaction, particularly regarding intuitiveness and perceived autonomy. Moreover, the ``Find an Object'' achieved effective real-time performance. These findings provide promising evidence that multimodal haptic-visual feedback can improve daily usability and independence compared to traditional audio-centric methods, motivating larger-scale clinical validations.
>
---
#### [replaced 022] MRI Super-Resolution with Deep Learning: A Comprehensive Survey
- **分类: eess.IV; cs.AI; cs.CV; eess.SP**

- **链接: [https://arxiv.org/pdf/2511.16854v3](https://arxiv.org/pdf/2511.16854v3)**

> **作者:** Mohammad Khateri; Serge Vasylechko; Morteza Ghahremani; Liam Timms; Deniz Kocanaogullari; Simon K. Warfield; Camilo Jaimes; Davood Karimi; Alejandra Sierra; Jussi Tohka; Sila Kurugol; Onur Afacan
>
> **备注:** 41 pages
>
> **摘要:** High-resolution (HR) magnetic resonance imaging (MRI) is crucial for many clinical and research applications. However, achieving it remains costly and constrained by technical trade-offs and experimental limitations. Super-resolution (SR) presents a promising computational approach to overcome these challenges by generating HR images from more affordable low-resolution (LR) scans, potentially improving diagnostic accuracy and efficiency without requiring additional hardware. This survey reviews recent advances in MRI SR techniques, with a focus on deep learning (DL) approaches. It examines DL-based MRI SR methods from the perspectives of computer vision, computational imaging, inverse problems, and MR physics, covering theoretical foundations, architectural designs, learning strategies, benchmark datasets, and performance metrics. We propose a systematic taxonomy to categorize these methods and present an in-depth study of both established and emerging SR techniques applicable to MRI, considering unique challenges in clinical and research contexts. We also highlight open challenges and directions that the community needs to address. Additionally, we provide a collection of essential open-access resources, tools, and tutorials, available on our GitHub: https://github.com/mkhateri/Awesome-MRI-Super-Resolution. IEEE keywords: MRI, Super-Resolution, Deep Learning, Computational Imaging, Inverse Problem, Survey.
>
---
#### [replaced 023] Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.04979v2](https://arxiv.org/pdf/2508.04979v2)**

> **作者:** Zheng Chen; Mingde Zhou; Jinpei Guo; Jiale Yuan; Yifei Ji; Yulun Zhang
>
> **备注:** Accepted to AAAI 2026. Code is available at: https://github.com/zhengchen1999/SODEC
>
> **摘要:** Diffusion-based image compression has demonstrated impressive perceptual performance. However, it suffers from two critical drawbacks: (1) excessive decoding latency due to multi-step sampling, and (2) poor fidelity resulting from over-reliance on generative priors. To address these issues, we propose SODEC, a novel single-step diffusion image compression model. We argue that in image compression, a sufficiently informative latent renders multi-step refinement unnecessary. Based on this insight, we leverage a pre-trained VAE-based model to produce latents with rich information, and replace the iterative denoising process with a single-step decoding. Meanwhile, to improve fidelity, we introduce the fidelity guidance module, encouraging output that is faithful to the original image. Furthermore, we design the rate annealing training strategy to enable effective training under extremely low bitrates. Extensive experiments show that SODEC significantly outperforms existing methods, achieving superior rate-distortion-perception performance. Moreover, compared to previous diffusion-based compression models, SODEC improves decoding speed by more than 20$\times$. Code is released at: https://github.com/zhengchen1999/SODEC.
>
---
#### [replaced 024] Emergent Extreme-View Geometry in 3D Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22686v2](https://arxiv.org/pdf/2511.22686v2)**

> **作者:** Yiwen Zhang; Joseph Tung; Ruojin Cai; David Fouhey; Hadar Averbuch-Elor
>
> **备注:** Project page is at https://ext-3dfms.github.io/
>
> **摘要:** 3D foundation models (3DFMs) have recently transformed 3D vision, enabling joint prediction of depths, poses, and point maps directly from images. Yet their ability to reason under extreme, non-overlapping views remains largely unexplored. In this work, we study their internal representations and find that 3DFMs exhibit an emergent understanding of extreme-view geometry, despite never being trained for such conditions. To further enhance these capabilities, we introduce a lightweight alignment scheme that refines their internal 3D representation by tuning only a small subset of backbone bias terms, leaving all decoder heads frozen. This targeted adaptation substantially improves relative pose estimation under extreme viewpoints without degrading per-image depth or point quality. Additionally, we contribute MegaUnScene, a new benchmark of Internet scenes unseen by existing 3DFMs, with dedicated test splits for both relative pose estimation and dense 3D reconstruction. All code and data will be released.
>
---
#### [replaced 025] Permutation-Aware Action Segmentation via Unsupervised Frame-to-Segment Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2305.19478v5](https://arxiv.org/pdf/2305.19478v5)**

> **作者:** Quoc-Huy Tran; Ahmed Mehmood; Muhammad Ahmed; Muhammad Naufil; Anas Zafar; Andrey Konin; M. Zeeshan Zia
>
> **备注:** Accepted to WACV 2024
>
> **摘要:** This paper presents an unsupervised transformer-based framework for temporal activity segmentation which leverages not only frame-level cues but also segment-level cues. This is in contrast with previous methods which often rely on frame-level information only. Our approach begins with a frame-level prediction module which estimates framewise action classes via a transformer encoder. The frame-level prediction module is trained in an unsupervised manner via temporal optimal transport. To exploit segment-level information, we utilize a segment-level prediction module and a frame-to-segment alignment module. The former includes a transformer decoder for estimating video transcripts, while the latter matches frame-level features with segment-level features, yielding permutation-aware segmentation results. Moreover, inspired by temporal optimal transport, we introduce simple-yet-effective pseudo labels for unsupervised training of the above modules. Our experiments on four public datasets, i.e., 50 Salads, YouTube Instructions, Breakfast, and Desktop Assembly show that our approach achieves comparable or better performance than previous methods in unsupervised activity segmentation. Our code and dataset are available on our research website: https://retrocausal.ai/research/.
>
---
#### [replaced 026] SPARK: Sim-ready Part-level Articulated Reconstruction with VLM Knowledge
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SPARK框架，解决从单张RGB图像生成可模拟的关节式3D物体问题。利用视觉语言模型提取粗略URDF参数并生成部件参考图，结合扩散变换器生成一致的部件与完整形状，并通过可微正向运动学与渲染优化关节参数，实现高保真、物理一致的仿真资产生成。**

- **链接: [https://arxiv.org/pdf/2512.01629v2](https://arxiv.org/pdf/2512.01629v2)**

> **作者:** Yumeng He; Ying Jiang; Jiayin Lu; Yin Yang; Chenfanfu Jiang
>
> **备注:** Project page: https://heyumeng.com/SPARK/index.html. 17 pages, 7 figures
>
> **摘要:** Articulated 3D objects are critical for embodied AI, robotics, and interactive scene understanding, yet creating simulation-ready assets remains labor-intensive and requires expert modeling of part hierarchies and motion structures. We introduce SPARK, a framework for reconstructing physically consistent, kinematic part-level articulated objects from a single RGB image. Given an input image, we first leverage VLMs to extract coarse URDF parameters and generate part-level reference images. We then integrate the part-image guidance and the inferred structure graph into a generative diffusion transformer to synthesize consistent part and complete shapes of articulated objects. To further refine the URDF parameters, we incorporate differentiable forward kinematics and differentiable rendering to optimize joint types, axes, and origins under VLM-generated open-state supervision. Extensive experiments show that SPARK produces high-quality, simulation-ready articulated assets across diverse categories, enabling downstream applications such as robotic manipulation and interaction modeling. Project page: https://heyumeng.com/SPARK/index.html.
>
---
#### [replaced 027] ViscNet: Vision-Based In-line Viscometry for Fluid Mixing Process
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01268v2](https://arxiv.org/pdf/2512.01268v2)**

> **作者:** Jongwon Sohn; Juhyeon Moon; Hyunjoon Jung; Jaewook Nam
>
> **摘要:** Viscosity measurement is essential for process monitoring and autonomous laboratory operation, yet conventional viscometers remain invasive and require controlled laboratory environments that differ substantially from real process conditions. We present a computer-vision-based viscometer that infers viscosity by exploiting how a fixed background pattern becomes optically distorted as light refracts through the mixing-driven, continuously deforming free surface. Under diverse lighting conditions, the system achieves a mean absolute error of 0.113 in log m2 s^-1 units for regression and reaches up to 81% accuracy in viscosity-class prediction. Although performance declines for classes with closely clustered viscosity values, a multi-pattern strategy improves robustness by providing enriched visual cues. To ensure sensor reliability, we incorporate uncertainty quantification, enabling viscosity predictions with confidence estimates. This stand-off viscometer offers a practical, automation-ready alternative to existing viscometry methods.
>
---
#### [replaced 028] Fast 3D Surrogate Modeling for Data Center Thermal Management
- **分类: cs.LG; cs.AI; cs.CV; eess.SY**

- **链接: [https://arxiv.org/pdf/2511.11722v2](https://arxiv.org/pdf/2511.11722v2)**

> **作者:** Soumyendu Sarkar; Antonio Guillen-Perez; Zachariah J Carmichael; Avisek Naug; Refik Mert Cam; Vineet Gundecha; Ashwin Ramesh Babu; Sahand Ghorbanpour; Ricardo Luna Gutierrez
>
> **备注:** Submitted to AAAI 2026 Conference
>
> **摘要:** Reducing energy consumption and carbon emissions in data centers by enabling real-time temperature prediction is critical for sustainability and operational efficiency. Achieving this requires accurate modeling of the 3D temperature field to capture airflow dynamics and thermal interactions under varying operating conditions. Traditional thermal CFD solvers, while accurate, are computationally expensive and require expert-crafted meshes and boundary conditions, making them impractical for real-time use. To address these limitations, we develop a vision-based surrogate modeling framework that operates directly on a 3D voxelized representation of the data center, incorporating server workloads, fan speeds, and HVAC temperature set points. We evaluate multiple architectures, including 3D CNN U-Net variants, a 3D Fourier Neural Operator, and 3D vision transformers, to map these thermal inputs to high-fidelity heat maps. Our results show that the surrogate models generalize across data center configurations and significantly speed up computations (20,000x), from hundreds of milliseconds to hours. This fast and accurate estimation of hot spots and temperature distribution enables real-time cooling control and workload redistribution, leading to substantial energy savings (7\%) and reduced carbon footprint.
>
---
#### [replaced 029] FairT2I: Mitigating Social Bias in Text-to-Image Generation via Large Language Model-Assisted Detection and Attribute Rebalancing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.03826v3](https://arxiv.org/pdf/2502.03826v3)**

> **作者:** Jinya Sakurai; Yuki Koyama; Issei Sato
>
> **摘要:** Text-to-image (T2I) models have advanced creative content generation, yet their reliance on large uncurated datasets often reproduces societal biases. We present FairT2I, a training-free and interactive framework grounded in a mathematically principled latent variable guidance formulation. This formulation decomposes the generative score function into attribute-conditioned components and reweights them according to a defined distribution, providing a unified and flexible mechanism for bias-aware generation that also subsumes many existing ad hoc debiasing approaches as special cases. Building upon this foundation, FairT2I incorporates (1) latent variable guidance as the core mechanism, (2) LLM-based bias detection to automatically infer bias-prone categories and attributes from text prompts as part of the latent structure, and (3) attribute resampling, which allows users to adjust or redefine the attribute distribution based on uniform, real-world, or user-specified statistics. The accompanying user interface supports this pipeline by enabling users to inspect detected biases, modify attributes or weights, and generate debiased images in real time. Experimental results show that LLMs outperform average human annotators in the number and granularity of detected bias categories and attributes. Moreover, FairT2I achieves superior performance to baseline models in both societal bias mitigation and image diversity, while preserving image quality and prompt fidelity.
>
---
#### [replaced 030] Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22699v2](https://arxiv.org/pdf/2511.22699v2)**

> **作者:** Z-Image Team; Huanqia Cai; Sihan Cao; Ruoyi Du; Peng Gao; Steven Hoi; Zhaohui Hou; Shijie Huang; Dengyang Jiang; Xin Jin; Liangchen Li; Zhen Li; Zhong-Yu Li; David Liu; Dongyang Liu; Junhan Shi; Qilong Wu; Feng Yu; Chi Zhang; Shifeng Zhang; Shilin Zhou
>
> **摘要:** The landscape of high-performance image generation models is currently dominated by proprietary systems, such as Nano Banana Pro and Seedream 4.0. Leading open-source alternatives, including Qwen-Image, Hunyuan-Image-3.0 and FLUX.2, are characterized by massive parameter counts (20B to 80B), making them impractical for inference, and fine-tuning on consumer-grade hardware. To address this gap, we propose Z-Image, an efficient 6B-parameter foundation generative model built upon a Scalable Single-Stream Diffusion Transformer (S3-DiT) architecture that challenges the "scale-at-all-costs" paradigm. By systematically optimizing the entire model lifecycle -- from a curated data infrastructure to a streamlined training curriculum -- we complete the full training workflow in just 314K H800 GPU hours (approx. $630K). Our few-step distillation scheme with reward post-training further yields Z-Image-Turbo, offering both sub-second inference latency on an enterprise-grade H800 GPU and compatibility with consumer-grade hardware (<16GB VRAM). Additionally, our omni-pre-training paradigm also enables efficient training of Z-Image-Edit, an editing model with impressive instruction-following capabilities. Both qualitative and quantitative experiments demonstrate that our model achieves performance comparable to or surpassing that of leading competitors across various dimensions. Most notably, Z-Image exhibits exceptional capabilities in photorealistic image generation and bilingual text rendering, delivering results that rival top-tier commercial models, thereby demonstrating that state-of-the-art results are achievable with significantly reduced computational overhead. We publicly release our code, weights, and online demo to foster the development of accessible, budget-friendly, yet state-of-the-art generative models.
>
---
#### [replaced 031] Visible Yet Unreadable: A Systematic Blind Spot of Vision Language Models Across Writing Systems
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.06996v5](https://arxiv.org/pdf/2509.06996v5)**

> **作者:** Jie Zhang; Ting Xu; Gelei Deng; Runyi Hu; Han Qiu; Tianwei Zhang; Qing Guo; Ivor Tsang
>
> **备注:** arXiv admin note: This article has been withdrawn by arXiv administrators due to violation of arXiv policy regarding generative AI authorship
>
> **摘要:** Writing is a universal cultural technology that reuses vision for symbolic communication. Humans display striking resilience: we readily recognize words even when characters are fragmented, fused, or partially occluded. This paper investigates whether advanced vision language models (VLMs) share this resilience. We construct two psychophysics inspired benchmarks across distinct writing systems, Chinese logographs and English alphabetic words, by splicing, recombining, and overlaying glyphs to yield ''visible but unreadable'' stimuli for models while remaining legible to humans. Despite strong performance on clean text, contemporary VLMs show a severe drop under these perturbations, frequently producing unrelated or incoherent outputs. The pattern suggests a structural limitation: models heavily leverage generic visual invariances but under rely on compositional priors needed for robust literacy. We release stimuli generation code, prompts, and evaluation protocols to facilitate transparent replication and follow up work. Our findings motivate architectures and training strategies that encode symbol segmentation, composition, and binding across scripts, and they delineate concrete challenges for deploying multimodal systems in education, accessibility, cultural heritage, and security.
>
---
#### [replaced 032] Guardian: Detecting Robotic Planning and Execution Errors with Vision-Language Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中故障检测与恢复难题，提出一种自动合成失败数据的方法，生成多样化的规划与执行错误。基于此构建三个新基准，训练出Guardian模型，实现高精度故障检测与细粒度推理，在仿真与真实机器人上均显著提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2512.01946v2](https://arxiv.org/pdf/2512.01946v2)**

> **作者:** Paul Pacaud; Ricardo Garcia; Shizhe Chen; Cordelia Schmid
>
> **备注:** Code, Data, and Models available at https://www.di.ens.fr/willow/research/guardian/. The paper contains 8 pages, 9 figures, 6 tables
>
> **摘要:** Robust robotic manipulation requires reliable failure detection and recovery. Although current Vision-Language Models (VLMs) show promise, their accuracy and generalization are limited by the scarcity of failure data. To address this data gap, we propose an automatic robot failure synthesis approach that procedurally perturbs successful trajectories to generate diverse planning and execution failures. This method produces not only binary classification labels but also fine-grained failure categories and step-by-step reasoning traces in both simulation and the real world. With it, we construct three new failure detection benchmarks: RLBench-Fail, BridgeDataV2-Fail, and UR5-Fail, substantially expanding the diversity and scale of existing failure datasets. We then train Guardian, a VLM with multi-view images for detailed failure reasoning and detection. Guardian achieves state-of-the-art performance on both existing and newly introduced benchmarks. It also effectively improves task success rates when integrated into a state-of-the-art manipulation system in simulation and real robots, demonstrating the impact of our generated failure data. Code, Data, and Models available at https://www.di.ens.fr/willow/research/guardian/.
>
---
#### [replaced 033] VeLU: Variance-enhanced Learning Unit for Deep Neural Networks
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.15051v2](https://arxiv.org/pdf/2504.15051v2)**

> **作者:** Ashkan Shakarami; Yousef Yeganeh; Azade Farshad; Lorenzo Nicolè; Stefano Ghidoni; Nassir Navab
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Activation functions play a critical role in deep neural networks by shaping gradient flow, optimization stability, and generalization. While ReLU remains widely used due to its simplicity, it suffers from gradient sparsity and dead-neuron issues and offers no adaptivity to input statistics. Smooth alternatives such as Swish and GELU improve gradient propagation but still apply a fixed transformation regardless of the activation distribution. In this paper, we propose VeLU, a Variance-enhanced Learning Unit that introduces variance-aware and distributionally aligned nonlinearity through a principled combination of ArcTan-ArcSin transformations, adaptive scaling, and Wasserstein-2 regularization (Optimal Transport). This design enables VeLU to modulate its response based on local activation variance, mitigate internal covariate shift at the activation level, and improve training stability without adding learnable parameters or architectural overhead. Extensive experiments across six deep neural networks show that VeLU outperforms ReLU, ReLU6, Swish, and GELU on 12 vision benchmarks. The implementation of VeLU is publicly available in GitHub.
>
---
#### [replaced 034] SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08294v2](https://arxiv.org/pdf/2511.08294v2)**

> **作者:** Laura Bragagnolo; Leonardo Barcellona; Stefano Ghidoni
>
> **备注:** WACV 2026
>
> **摘要:** Accurate 3D human pose estimation is fundamental for applications such as augmented reality and human-robot interaction. State-of-the-art multi-view methods learn to fuse predictions across views by training on large annotated datasets, leading to poor generalization when the test scenario differs. To overcome these limitations, we propose SkelSplat, a novel framework for multi-view 3D human pose estimation based on differentiable Gaussian rendering. Human pose is modeled as a skeleton of 3D Gaussians, one per joint, optimized via differentiable rendering to enable seamless fusion of arbitrary camera views without 3D ground-truth supervision. Since Gaussian Splatting was originally designed for dense scene reconstruction, we propose a novel one-hot encoding scheme that enables independent optimization of human joints. SkelSplat outperforms approaches that do not rely on 3D ground truth in Human3.6M and CMU, while reducing the cross-dataset error up to 47.8% compared to learning-based methods. Experiments on Human3.6M-Occ and Occlusion-Person demonstrate robustness to occlusions, without scenario-specific fine-tuning. Our project page is available here: https://skelsplat.github.io.
>
---
#### [replaced 035] Mutually-Aware Feature Learning for Few-Shot Object Counting
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2408.09734v2](https://arxiv.org/pdf/2408.09734v2)**

> **作者:** Yerim Jeon; Subeen Lee; Jihwan Kim; Jae-Pil Heo
>
> **备注:** Accepted to Pattern Recognition 2025
>
> **摘要:** Few-shot object counting has garnered significant attention for its practicality as it aims to count target objects in a query image based on given exemplars without additional training. However, the prevailing extract-and-match approach has a shortcoming: query and exemplar features lack interaction during feature extraction since they are extracted independently and later correlated based on similarity. This can lead to insufficient target awareness and confusion in identifying the actual target when multiple class objects coexist. To address this, we propose a novel framework, Mutually-Aware FEAture learning (MAFEA), which encodes query and exemplar features with mutual awareness from the outset. By encouraging interaction throughout the pipeline, we obtain target-aware features robust to a multi-category scenario. Furthermore, we introduce background token to effectively associate the query's target region with exemplars and decouple its background region. Our extensive experiments demonstrate that our model achieves state-of-the-art performance on FSCD-LVIS and FSC-147 benchmarks with remarkably reduced target confusion.
>
---
#### [replaced 036] Detect Anything 3D in the Wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.07958v3](https://arxiv.org/pdf/2504.07958v3)**

> **作者:** Hanxue Zhang; Haoran Jiang; Qingsong Yao; Yanan Sun; Renrui Zhang; Hao Zhao; Hongyang Li; Hongzi Zhu; Zetong Yang
>
> **摘要:** Despite the success of deep learning in close-set 3D object detection, existing approaches struggle with zero-shot generalization to novel objects and camera configurations. We introduce DetAny3D, a promptable 3D detection foundation model capable of detecting any novel object under arbitrary camera configurations using only monocular inputs. Training a foundation model for 3D detection is fundamentally constrained by the limited availability of annotated 3D data, which motivates DetAny3D to leverage the rich prior knowledge embedded in extensively pre-trained 2D foundation models to compensate for this scarcity. To effectively transfer 2D knowledge to 3D, DetAny3D incorporates two core modules: the 2D Aggregator, which aligns features from different 2D foundation models, and the 3D Interpreter with Zero-Embedding Mapping, which stabilizes early training in 2D-to-3D knowledge transfer. Experimental results validate the strong generalization of our DetAny3D, which not only achieves state-of-the-art performance on unseen categories and novel camera configurations, but also surpasses most competitors on in-domain data. DetAny3D sheds light on the potential of the 3D foundation model for diverse applications in real-world scenarios, e.g., rare object detection in autonomous driving, and demonstrates promise for further exploration of 3D-centric tasks in open-world settings. More visualization results can be found at our code repository.
>
---
#### [replaced 037] Learning Egocentric In-Hand Object Segmentation through Weak Supervision from Human Narrations
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.26004v2](https://arxiv.org/pdf/2509.26004v2)**

> **作者:** Nicola Messina; Rosario Leonardi; Luca Ciampi; Fabio Carrara; Giovanni Maria Farinella; Fabrizio Falchi; Antonino Furnari
>
> **备注:** Under consideration at Pattern Recognition Letters
>
> **摘要:** Pixel-level recognition of objects manipulated by the user from egocentric images enables key applications spanning assistive technologies, industrial safety, and activity monitoring. However, progress in this area is currently hindered by the scarcity of annotated datasets, as existing approaches rely on costly manual labels. In this paper, we propose to learn human-object interaction detection leveraging narrations $\unicode{x2013}$ natural language descriptions of the actions performed by the camera wearer which contain clues about manipulated objects. We introduce Narration-Supervised in-Hand Object Segmentation (NS-iHOS), a novel task where models have to learn to segment in-hand objects by learning from natural-language narrations in a weakly-supervised regime. Narrations are then not employed at inference time. We showcase the potential of the task by proposing Weakly-Supervised In-hand Object Segmentation from Human Narrations (WISH), an end-to-end model distilling knowledge from narrations to learn plausible hand-object associations and enable in-hand object segmentation without using narrations at test time. We benchmark WISH against different baselines based on open-vocabulary object detectors and vision-language models. Experiments on EPIC-Kitchens and Ego4D show that WISH surpasses all baselines, recovering more than 50% of the performance of fully supervised methods, without employing fine-grained pixel-wise annotations. Code and data can be found at https://fpv-iplab.github.io/WISH.
>
---
#### [replaced 038] ContourDiff: Unpaired Medical Image Translation with Structural Consistency
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2403.10786v3](https://arxiv.org/pdf/2403.10786v3)**

> **作者:** Yuwen Chen; Nicholas Konz; Hanxue Gu; Haoyu Dong; Yaqian Chen; Lin Li; Jisoo Lee; Maciej A. Mazurowski
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) https://melba-journal.org/2025:031
>
> **摘要:** Accurately translating medical images between different modalities, such as Computed Tomography (CT) to Magnetic Resonance Imaging (MRI), has numerous downstream clinical and machine learning applications. While several methods have been proposed to achieve this, they often prioritize perceptual quality with respect to output domain features over preserving anatomical fidelity. However, maintaining anatomy during translation is essential for many tasks, e.g., when leveraging masks from the input domain to develop a segmentation model with images translated to the output domain. To address these challenges, we propose ContourDiff with Spatially Coherent Guided Diffusion (SCGD), a novel framework that leverages domain-invariant anatomical contour representations of images. These representations are simple to extract from images, yet form precise spatial constraints on their anatomical content. We introduce a diffusion model that converts contour representations of images from arbitrary input domains into images in the output domain of interest. By applying the contour as a constraint at every diffusion sampling step, we ensure the preservation of anatomical content. We evaluate our method on challenging lumbar spine and hip-and-thigh CT-to-MRI translation tasks, via (1) the performance of segmentation models trained on translated images applied to real MRIs, and (2) the foreground FID and KID of translated images with respect to real MRIs. Our method outperforms other unpaired image translation methods by a significant margin across almost all metrics and scenarios. Moreover, it achieves this without the need to access any input domain information during training and we further verify its zero-shot capability, showing that a model trained on one anatomical region can be directly applied to unseen regions without retraining (GitHub: https://github.com/mazurowski-lab/ContourDiff).
>
---
#### [replaced 039] TempoMaster: Efficient Long Video Generation via Next-Frame-Rate Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12578v3](https://arxiv.org/pdf/2511.12578v3)**

> **作者:** Yukuo Ma; Cong Liu; Junke Wang; Junqi Liu; Haibin Huang; Zuxuan Wu; Chi Zhang; Xuelong Li
>
> **备注:** for more information, see https://scottykma.github.io/tempomaster-gitpage/
>
> **摘要:** We present TempoMaster, a novel framework that formulates long video generation as next-frame-rate prediction. Specifically, we first generate a low-frame-rate clip that serves as a coarse blueprint of the entire video sequence, and then progressively increase the frame rate to refine visual details and motion continuity. During generation, TempoMaster employs bidirectional attention within each frame-rate level while performing autoregression across frame rates, thus achieving long-range temporal coherence while enabling efficient and parallel synthesis. Extensive experiments demonstrate that TempoMaster establishes a new state-of-the-art in long video generation, excelling in both visual and temporal quality.
>
---
#### [replaced 040] APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation
- **分类: cs.NE; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.14270v5](https://arxiv.org/pdf/2507.14270v5)**

> **作者:** Ravin Kumar
>
> **备注:** 12 pages, 2 figures, 1 table. Includes a GitHub repository for MNIST experiments and a PyPI package for APTx Neuron implementation
>
> **摘要:** We propose the APTx Neuron, a novel, unified neural computation unit that integrates non-linear activation and linear transformation into a single trainable expression. The APTx Neuron is derived from the APTx activation function, thereby eliminating the need for separate activation layers and making the architecture both optimization-efficient and elegant. The proposed neuron follows the functional form $y = \sum_{i=1}^{n} ((α_i + \tanh(β_i x_i)) \cdot γ_i x_i) + δ$, where all parameters $α_i$, $β_i$, $γ_i$, and $δ$ are trainable. We validate our APTx Neuron-based architecture on the MNIST dataset, achieving up to $96.69\%$ test accuracy within 11 epochs using approximately 332K trainable parameters. The results highlight the superior expressiveness and training efficiency of the APTx Neuron compared to traditional neurons, pointing toward a new paradigm in unified neuron design and the architectures built upon it. Source code is available at https://github.com/mr-ravin/aptx_neuron.
>
---
#### [replaced 041] Convolution goes higher-order: a biologically inspired mechanism empowers image classification
- **分类: cs.CV; cs.LG; q-bio.NC**

- **链接: [https://arxiv.org/pdf/2412.06740v2](https://arxiv.org/pdf/2412.06740v2)**

> **作者:** Simone Azeglio; Olivier Marre; Peter Neri; Ulisse Ferrari
>
> **摘要:** We propose a novel approach to image classification inspired by complex nonlinear biological visual processing, whereby classical convolutional neural networks (CNNs) are equipped with learnable higher-order convolutions. Our model incorporates a Volterra-like expansion of the convolution operator, capturing multiplicative interactions akin to those observed in early and advanced stages of biological visual processing. We evaluated this approach on synthetic datasets by measuring sensitivity to testing higher-order correlations and performance in standard benchmarks (MNIST, FashionMNIST, CIFAR10, CIFAR100 and Imagenette). Our architecture outperforms traditional CNN baselines, and achieves optimal performance with expansions up to 3rd/4th order, aligning remarkably well with the distribution of pixel intensities in natural images. Through systematic perturbation analysis, we validate this alignment by isolating the contributions of specific image statistics to model performance, demonstrating how different orders of convolution process distinct aspects of visual information. Furthermore, Representational Similarity Analysis reveals distinct geometries across network layers, indicating qualitatively different modes of visual information processing. Our work bridges neuroscience and deep learning, offering a path towards more effective, biologically inspired computer vision models. It provides insights into visual information processing and lays the groundwork for neural networks that better capture complex visual patterns, particularly in resource-constrained scenarios.
>
---
#### [replaced 042] PointCNN++: Performant Convolution on Native Points
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.23227v2](https://arxiv.org/pdf/2511.23227v2)**

> **作者:** Lihan Li; Haofeng Zhong; Rui Bu; Mingchao Sun; Wenzheng Chen; Baoquan Chen; Yangyan Li
>
> **摘要:** Existing convolutional learning methods for 3D point cloud data are divided into two paradigms: point-based methods that preserve geometric precision but often face performance challenges, and voxel-based methods that achieve high efficiency through quantization at the cost of geometric fidelity. This loss of precision is a critical bottleneck for tasks such as point cloud registration. We propose PointCNN++, a novel architectural design that fundamentally mitigates this precision-performance trade-off. It $\textbf{generalizes sparse convolution from voxels to points}$, treating voxel-based convolution as a specialized, degraded case of our more general point-based convolution. First, we introduce a point-centric convolution where the receptive field is centered on the original, high-precision point coordinates. Second, to make this high-fidelity operation performant, we design a computational strategy that operates $\textbf{natively}$ on points. We formulate the convolution on native points as a Matrix-Vector Multiplication and Reduction (MVMR) problem, for which we develop a dedicated, highly-optimized GPU kernel. Experiments demonstrate that PointCNN++ $\textbf{uses an order of magnitude less memory and is several times faster}$ than representative point-based methods. Furthermore, when used as a simple replacement for the voxel-based backbones it generalizes, it $\textbf{significantly improves point cloud registration accuracies while proving both more memory-efficient and faster}$. PointCNN++ shows that preserving geometric detail and achieving high performance are not mutually exclusive, paving the way for a new class of 3D learning with high fidelity and efficiency. Our code will be open sourced.
>
---
#### [replaced 043] COACH: Collaborative Agents for Contextual Highlighting - A Multi-Agent Framework for Sports Video Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01853v2](https://arxiv.org/pdf/2512.01853v2)**

> **作者:** Tsz-To Wong; Ching-Chun Huang; Hong-Han Shuai
>
> **备注:** Accepted by AAAI 2026 Workshop LaMAS
>
> **摘要:** Intelligent sports video analysis demands a comprehensive understanding of temporal context, from micro-level actions to macro-level game strategies. Existing end-to-end models often struggle with this temporal hierarchy, offering solutions that lack generalization, incur high development costs for new tasks, and suffer from poor interpretability. To overcome these limitations, we propose a reconfigurable Multi-Agent System (MAS) as a foundational framework for sports video understanding. In our system, each agent functions as a distinct "cognitive tool" specializing in a specific aspect of analysis. The system's architecture is not confined to a single temporal dimension or task. By leveraging iterative invocation and flexible composition of these agents, our framework can construct adaptive pipelines for both short-term analytic reasoning (e.g., Rally QA) and long-term generative summarization (e.g., match summaries). We demonstrate the adaptability of this framework using two representative tasks in badminton analysis, showcasing its ability to bridge fine-grained event detection and global semantic organization. This work presents a paradigm shift towards a flexible, scalable, and interpretable system for robust, cross-task sports video intelligence. The project homepage is available at https://aiden1020.github.io/COACH-project-page
>
---
#### [replaced 044] TimeSearch: Hierarchical Video Search with Spotlight and Reflection for Human-like Long Video Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.01407v2](https://arxiv.org/pdf/2504.01407v2)**

> **作者:** Junwen Pan; Rui Zhang; Xin Wan; Yuan Zhang; Ming Lu; Qi She
>
> **摘要:** Large video-language models (LVLMs) have shown remarkable performance across various video-language tasks. However, they encounter significant challenges when processing long videos because of the large number of video frames involved. Downsampling long videos in either space or time can lead to visual hallucinations, making it difficult to accurately interpret long videos. Motivated by human hierarchical temporal search strategies, we propose \textbf{TimeSearch}, a novel framework enabling LVLMs to understand long videos in a human-like manner. TimeSearch integrates two human-like primitives into a unified autoregressive LVLM: 1) \textbf{Spotlight} efficiently identifies relevant temporal events through a Temporal-Augmented Frame Representation (TAFR), explicitly binding visual features with timestamps; 2) \textbf{Reflection} evaluates the correctness of the identified events, leveraging the inherent temporal self-reflection capabilities of LVLMs. TimeSearch progressively explores key events and prioritizes temporal search based on reflection confidence. Extensive experiments on challenging long-video benchmarks confirm that TimeSearch substantially surpasses previous state-of-the-art, improving the accuracy from 41.8\% to 51.5\% on the LVBench. Additionally, experiments on temporal grounding demonstrate that appropriate TAFR is adequate to effectively stimulate the surprising temporal grounding ability of LVLMs in a simpler yet versatile manner, which improves mIoU on Charades-STA by 11.8\%. The code will be released.
>
---
#### [replaced 045] NOCTIS: Novel Object Cyclic Threshold based Instance Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.01463v4](https://arxiv.org/pdf/2507.01463v4)**

> **作者:** Max Gandyra; Alessandro Santonicola; Michael Beetz
>
> **备注:** 9 pages, 3 figures, 5 tables
>
> **摘要:** Instance segmentation of novel objects instances in RGB images, given some example images for each object, is a well known problem in computer vision. Designing a model general enough to be employed for all kinds of novel objects without (re-) training has proven to be a difficult task. To handle this, we present a new training-free framework, called: Novel Object Cyclic Threshold based Instance Segmentation (NOCTIS). NOCTIS integrates two pre-trained models: Grounded-SAM 2 for object proposals with precise bounding boxes and corresponding segmentation masks; and DINOv2 for robust class and patch embeddings, due to its zero-shot capabilities. Internally, the proposal-object matching is realized by determining an object matching score based on the similarity of the class embeddings and the average maximum similarity of the patch embeddings with a new cyclic thresholding (CT) mechanism that mitigates unstable matches caused by repetitive textures or visually similar patterns. Beyond CT, NOCTIS introduces: (i) an appearance score that is unaffected by object selection bias; (ii) the usage of the average confidence of the proposals' bounding box and mask as a scoring component; and (iii) an RGB-only pipeline that performs even better than RGB-D ones. We empirically show that NOCTIS, without further training/fine tuning, outperforms the best RGB and RGB-D methods regarding the mean AP score on the seven core datasets of the BOP 2023 challenge for the "Model-based 2D segmentation of unseen objects" task.
>
---
#### [replaced 046] Look, Recite, Then Answer: Enhancing VLM Performance via Self-Generated Knowledge Hints
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.00882v2](https://arxiv.org/pdf/2512.00882v2)**

> **作者:** Xisheng Feng
>
> **摘要:** Vision-Language Models (VLMs) exhibit significant performance plateaus in specialized domains like precision agriculture, primarily due to "Reasoning-Driven Hallucination" where linguistic priors override visual perception. A key bottleneck is the "Modality Gap": visual embeddings fail to reliably activate the fine-grained expert knowledge already encoded in model parameters. We propose "Look, Recite, Then Answer," a parameter-efficient framework that enhances VLMs via self-generated knowledge hints while keeping backbone models frozen. The framework decouples inference into three stages: (1) Look generates objective visual descriptions and candidate sets; (2) Recite employs a lightweight 1.7B router to transform visual cues into targeted queries that trigger candidate-specific parametric knowledge; (3) Answer performs parallel evidence alignment between descriptions and recited knowledge to select the most consistent label. On AgroBench, our method achieves state-of-the-art results, improving Weed Identification accuracy by 23.52% over Qwen2-VL-72B and surpassing GPT-4o without external search overhead. This modular design mitigates hallucinations by transforming passive perception into active, controllable knowledge retrieval
>
---
#### [replaced 047] UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.01846v4](https://arxiv.org/pdf/2502.01846v4)**

> **作者:** Aashish Rai; Dilin Wang; Mihir Jain; Nikolaos Sarafianos; Kefan Chen; Srinath Sridhar; Aayush Prakash
>
> **备注:** https://ivl.cs.brown.edu/uvgs
>
> **摘要:** 3D Gaussian Splatting (3DGS) has demonstrated superior quality in modeling 3D objects and scenes. However, generating 3DGS remains challenging due to their discrete, unstructured, and permutation-invariant nature. In this work, we present a simple yet effective method to overcome these challenges. We utilize spherical mapping to transform 3DGS into a structured 2D representation, termed UVGS. UVGS can be viewed as multi-channel images, with feature dimensions as a concatenation of Gaussian attributes such as position, scale, color, opacity, and rotation. We further find that these heterogeneous features can be compressed into a lower-dimensional (e.g., 3-channel) shared feature space using a carefully designed multi-branch network. The compressed UVGS can be treated as typical RGB images. Remarkably, we discover that typical VAEs trained with latent diffusion models can directly generalize to this new representation without additional training. Our novel representation makes it effortless to leverage foundational 2D models, such as diffusion models, to directly model 3DGS. Additionally, one can simply increase the 2D UV resolution to accommodate more Gaussians, making UVGS a scalable solution compared to typical 3D backbones. This approach immediately unlocks various novel generation applications of 3DGS by inherently utilizing the already developed superior 2D generation capabilities. In our experiments, we demonstrate various unconditional, conditional generation, and inpainting applications of 3DGS based on diffusion models, which were previously non-trivial.
>
---
#### [replaced 048] MaxSup: Overcoming Representation Collapse in Label Smoothing
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2502.15798v3](https://arxiv.org/pdf/2502.15798v3)**

> **作者:** Yuxuan Zhou; Heng Li; Zhi-Qi Cheng; Xudong Yan; Yifei Dong; Mario Fritz; Margret Keuper
>
> **备注:** NeurIPS 2025 Oral (0.36% acceptance); code: https://github.com/ZhouYuxuanYX/Maximum-Suppression-Regularization
>
> **摘要:** Label Smoothing (LS) is widely adopted to reduce overconfidence in neural network predictions and improve generalization. Despite these benefits, recent studies reveal two critical issues with LS. First, LS induces overconfidence in misclassified samples. Second, it compacts feature representations into overly tight clusters, diluting intra-class diversity, although the precise cause of this phenomenon remained elusive. In this paper, we analytically decompose the LS-induced loss, exposing two key terms: (i) a regularization term that dampens overconfidence only when the prediction is correct, and (ii) an error-amplification term that arises under misclassifications. This latter term compels the network to reinforce incorrect predictions with undue certainty, exacerbating representation collapse. To address these shortcomings, we propose Max Suppression (MaxSup), which applies uniform regularization to both correct and incorrect predictions by penalizing the top-1 logit rather than the ground-truth logit. Through extensive feature-space analyses, we show that MaxSup restores intra-class variation and sharpens inter-class boundaries. Experiments on large-scale image classification and multiple downstream tasks confirm that MaxSup is a more robust alternative to LS. Code is available at: https://github.com/ZhouYuxuanYX/Maximum-Suppression-Regularization
>
---
#### [replaced 049] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在动态决策中忽视历史上下文的问题，提出AVA-VLA框架。通过引入基于信念状态的主动视觉注意力机制，利用递归状态动态聚焦关键视觉信息，提升序列决策能力。在LIBERO、CALVIN等基准上实现领先性能，并验证了真实机器人平台上的有效性与仿真到现实的迁移能力。**

- **链接: [https://arxiv.org/pdf/2511.18960v2](https://arxiv.org/pdf/2511.18960v2)**

> **作者:** Lei Xiao; Jifeng Li; Juntao Gao; Feiyang Ye; Yan Jin; Jingjing Qian; Jing Zhang; Yong Wu; Xiaoyuan Yu
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in embodied AI tasks. However, existing VLA models, often built upon Vision-Language Models (VLMs), typically process dense visual inputs independently at each timestep. This approach implicitly models the task as a Markov Decision Process (MDP). However, this history-agnostic design is suboptimal for effective visual token processing in dynamic sequential decision-making, as it fails to leverage the context of history. To address this limitation, we reformulate the problem from a Partially Observable Markov Decision Process (POMDP) perspective and propose a novel framework named AVA-VLA. Inspired by the POMDP that the action generation should be conditioned on the belief state. AVA-VLA introduces Active Visual Attention (AVA) to dynamically modulate visual processing. It achieves this by leveraging the recurrent state, which is a neural approximation of the agent's belief state derived from the previous decision step. Specifically, the AVA module uses the recurrent state to compute the soft weights to actively process task-relevant visual tokens based on its historical context. Comprehensive evaluations demonstrate that AVA-VLA achieves state-of-the-art performance across popular robotic benchmarks, including LIBERO and CALVIN. Furthermore, real-world deployments on a dual-arm robot platform validate the framework's practical applicability and robust sim-to-real transferability.
>
---
#### [replaced 050] Walk Before You Dance: High-fidelity and Editable Dance Synthesis via Generative Masked Motion Prior
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.04634v3](https://arxiv.org/pdf/2504.04634v3)**

> **作者:** Foram N Shah; Parshwa Shah; Muhammad Usama Saleem; Ekkasit Pinyoanuntapong; Pu Wang; Hongfei Xue; Ahmed Helmy
>
> **摘要:** Recent advances in dance generation have enabled the automatic synthesis of 3D dance motions. However, existing methods still face significant challenges in simultaneously achieving high realism, precise dance-music synchronization, diverse motion expression, and physical plausibility. To address these limitations, we propose a novel approach that leverages a generative masked text-to-motion model as a distribution prior to learn a probabilistic mapping from diverse guidance signals, including music, genre, and pose, into high-quality dance motion sequences. Our framework also supports semantic motion editing, such as motion inpainting and body part modification. Specifically, we introduce a multi-tower masked motion model that integrates a text-conditioned masked motion backbone with two parallel, modality-specific branches: a music-guidance tower and a pose-guidance tower. The model is trained using synchronized and progressive masked training, which allows effective infusion of the pretrained text-to-motion prior into the dance synthesis process while enabling each guidance branch to optimize independently through its own loss function, mitigating gradient interference. During inference, we introduce classifier-free logits guidance and pose-guided token optimization to strengthen the influence of music, genre, and pose signals. Extensive experiments demonstrate that our method sets a new state of the art in dance generation, significantly advancing both the quality and editability over existing approaches. Project Page available at https://foram-s1.github.io/DanceMosaic/
>
---
#### [replaced 051] OpenREAD: Reinforced Open-Ended Reasoning for End-to-End Autonomous Driving with LLM-as-Critic
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01830v2](https://arxiv.org/pdf/2512.01830v2)**

> **作者:** Songyan Zhang; Wenhui Huang; Zhan Chen; Chua Jiahao Collister; Qihang Huang; Chen Lv
>
> **摘要:** Recently, two-stage fine-tuning strategies, e.g., acquiring essential driving knowledge through supervised fine-tuning (SFT) and further enhancing decision-making and planning via reinforcement fine-tuning (RFT), have shown strong potential in advancing the knowledge-driven autonomous driving (AD) paradigm. However, the learning nature of SFT still limits the generalization of reasoning, thereby constraining the full potential of driving performance. Meanwhile, current RFT approaches are primarily applied to downstream tasks, since scene understanding is an open-ended problem where corresponding rewards are difficult to quantify. To address these limitations, we propose OpenREAD, an OPEN-ended REasoning reinforced vision-language model (VLM)-based autonomous driving (AD) framework that enables end-to-end RFT across the full spectrum from high-level reasoning to low-level trajectory planning. Specifically, we begin by constructing large-scale Chain-of-Thought (CoT) annotations on open-source driving-related knowledge datasets, and employ the powerful Qwen3 large language model (LLM) as the critic in RFT to quantify reasoning quality for open-ended questions during reward modeling. Extensive experiments confirm that joint end-to-end RFT yields substantial improvements in both upstream and downstream tasks, enabling OpenREAD to achieve state-of-the-art performance on reasoning and planning benchmarks.
>
---
#### [replaced 052] Learning-based 3D Reconstruction in Autonomous Driving: A Comprehensive Survey
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的学习型3D重建任务，旨在解决环境精确建模难题。系统综述了相关技术演进与应用，分析了方法分类、挑战及研究趋势，指出当前研究在车载验证与安全评估方面披露不足，并提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2503.14537v5](https://arxiv.org/pdf/2503.14537v5)**

> **作者:** Liewen Liao; Weihao Yan; Wang Xu; Ming Yang; Songan Zhang; H. Eric Tseng
>
> **备注:** Published in IEEE Trans. on Intelligent Transportation Systems
>
> **摘要:** Learning-based 3D reconstruction has emerged as a transformative technique in autonomous driving, enabling precise modeling of environments through advanced neural representations. It has inspired pioneering solutions for vital tasks in autonomous driving, such as dense mapping and closed-loop simulation, as well as comprehensive scene feature for driving scene understanding and reasoning. Given the rapid growth in related research, this survey provides a comprehensive review of both technical evolutions and practical applications in autonomous driving. We begin with an introduction to the preliminaries of learning-based 3D reconstruction to provide a solid technical background foundation, then progress to a rigorous, multi-dimensional examination of cutting-edge methodologies, systematically organized according to the distinctive technical requirements and fundamental challenges of autonomous driving. Through analyzing and summarizing development trends and cutting-edge research, we identify existing technical challenges, along with insufficient disclosure of on-board validation and safety verification details in the current literature, and ultimately suggest potential directions to guide future studies.
>
---
#### [replaced 053] Ov3R: Open-Vocabulary Semantic 3D Reconstruction from RGB Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.22052v2](https://arxiv.org/pdf/2507.22052v2)**

> **作者:** Ziren Gong; Xiaohan Li; Fabio Tosi; Jiawei Han; Stefano Mattoccia; Jianfei Cai; Matteo Poggi
>
> **摘要:** We present Ov3R, a novel framework for open-vocabulary semantic 3D reconstruction from RGB video streams, designed to advance Spatial AI. The system features two key components: CLIP3R, a CLIP-informed 3D reconstruction module that predicts dense point maps from overlapping clips while embedding object-level semantics; and 2D-3D OVS, a 2D-3D open-vocabulary semantic module that lifts 2D features into 3D by learning fused descriptors integrating spatial, geometric, and semantic cues. Unlike prior methods, Ov3R incorporates CLIP semantics directly into the reconstruction process, enabling globally consistent geometry and fine-grained semantic alignment. Our framework achieves state-of-the-art performance in both dense 3D reconstruction and open-vocabulary 3D segmentation, marking a step forward toward real-time, semantics-aware Spatial AI.
>
---
#### [replaced 054] CT-GLIP: 3D Grounded Language-Image Pretraining with CT Scans and Radiology Reports for Full-Body Scenarios
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出CT-GLIP，一种用于全身体部CT扫描的3D医学视觉语言预训练模型。针对现有方法因全局对齐导致关键细节丢失的问题，通过构建细粒度的CT-报告配对，实现基于语义的跨模态对比学习，显著提升器官与病灶的零样本识别与分割性能。**

- **链接: [https://arxiv.org/pdf/2404.15272v4](https://arxiv.org/pdf/2404.15272v4)**

> **作者:** Jingyang Lin; Yingda Xia; Jianpeng Zhang; Ke Yan; Kai Cao; Le Lu; Jiebo Luo; Ling Zhang
>
> **摘要:** 3D medical vision-language (VL) pretraining has shown potential in radiology by leveraging large-scale multimodal datasets with CT-report pairs. However, existing methods primarily rely on a global VL alignment directly adapted from 2D scenarios. The entire 3D image is transformed into one global embedding, resulting in a loss of sparse but critical semantics essential for accurately aligning with the corresponding diagnosis. To address this limitation, we propose CT-GLIP, a 3D Grounded Language-Image Pretrained model that constructs fine-grained CT-report pairs to enhance \textit{grounded} cross-modal contrastive learning, effectively aligning grounded visual features with precise textual descriptions. Leveraging the grounded cross-modal alignment, CT-GLIP improves performance across diverse downstream tasks and can even identify organs and abnormalities in a zero-shot manner using natural language. CT-GLIP is trained on a multimodal CT dataset comprising 44,011 organ-level CT-report pairs from 17,702 patients, covering 104 organs. Evaluation is conducted on four downstream tasks: zero-shot organ recognition (OR), zero-shot abnormality detection (AD), tumor detection (TD), and tumor segmentation (TS). Empirical results show that it outperforms its counterparts with global VL alignment. Compared to vanilla CLIP, CT-GLIP achieves average performance improvements of 15.1% of F1 score, 1.9% of AUC, and 3.2% of DSC for zero-shot AD, TD, and TS tasks, respectively. This study highlights the significance of grounded VL alignment in enabling 3D medical VL foundation models to understand sparse representations within CT scans.
>
---
#### [replaced 055] PrITTI: Primitive-based Generation of Controllable and Editable 3D Semantic Urban Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.19117v2](https://arxiv.org/pdf/2506.19117v2)**

> **作者:** Christina Ourania Tze; Daniel Dauner; Yiyi Liao; Dzmitry Tsishkou; Andreas Geiger
>
> **备注:** Project page: https://raniatze.github.io/pritti/
>
> **摘要:** Existing approaches to 3D semantic urban scene generation predominantly rely on voxel-based representations, which are bound by fixed resolution, challenging to edit, and memory-intensive in their dense form. In contrast, we advocate for a primitive-based paradigm where urban scenes are represented using compact, semantically meaningful 3D elements that are easy to manipulate and compose. To this end, we introduce PrITTI, a latent diffusion model that leverages vectorized object primitives and rasterized ground surfaces for generating diverse, controllable, and editable 3D semantic urban scenes. This hybrid representation yields a structured latent space that facilitates object- and ground-level manipulation. Experiments on KITTI-360 show that primitive-based representations unlock the full capabilities of diffusion transformers, achieving state-of-the-art 3D scene generation quality with lower memory requirements, faster inference, and greater editability than voxel-based methods. Beyond generation, PrITTI supports a range of downstream applications, including scene editing, inpainting, outpainting, and photo-realistic street-view synthesis. Code and models are publicly available at $\href{https://raniatze.github.io/pritti/}{https://raniatze.github.io/pritti}$.
>
---
#### [replaced 056] Beyond Pixels: Efficient Dataset Distillation via Sparse Gaussian Representation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.26219v2](https://arxiv.org/pdf/2509.26219v2)**

> **作者:** Chenyang Jiang; Zhengcen Li; Hang Zhao; Qiben Shan; Shaocong Wu; Jingyong Su
>
> **备注:** 19 pages; Code is available on https://github.com/j-cyoung/GSDatasetDistillation
>
> **摘要:** Dataset distillation has emerged as a promising paradigm that synthesizes compact, informative datasets capable of retaining the knowledge of large-scale counterparts, thereby addressing the substantial computational and storage burdens of modern model training. Conventional approaches typically rely on dense pixel-level representations, which introduce redundancy and are difficult to scale up. In this work, we propose GSDD, a novel and efficient sparse representation for dataset distillation based on 2D Gaussians. Instead of representing all pixels equally, GSDD encodes critical discriminative information in a distilled image using only a small number of Gaussian primitives. This sparse representation could improve dataset diversity under the same storage budget, enhancing coverage of difficult samples and boosting distillation performance. To ensure both efficiency and scalability, we adapt CUDA-based splatting operators for parallel inference and training, enabling high-quality rendering with minimal computational and memory overhead. Our method is simple yet effective, broadly applicable to different distillation pipelines, and highly scalable. Experiments show that GSDD achieves state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet subsets, while remaining highly efficient encoding and decoding cost. Our code is available at https://github.com/j-cyoung/GSDatasetDistillation.
>
---
#### [replaced 057] Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06741v5](https://arxiv.org/pdf/2511.06741v5)**

> **作者:** Wenbo Huang; Jinghui Zhang; Zhenghao Chen; Guang Li; Lei Zhang; Yang Cao; Fang Dong; Takahiro Ogawa; Miki Haseyama
>
> **备注:** Accepted by AAAI 2026 Oral
>
> **摘要:** Wide-angle videos in few-shot action recognition (FSAR) effectively express actions within specific scenarios. However, without a global understanding of both subjects and background, recognizing actions in such samples remains challenging because of the background distractions. Receptance Weighted Key Value (RWKV), which learns interaction between various dimensions, shows promise for global modeling. While directly applying RWKV to wide-angle FSAR may fail to highlight subjects due to excessive background information. Additionally, temporal relation degraded by frames with similar backgrounds is difficult to reconstruct, further impacting performance. Therefore, we design the CompOund SegmenTation and Temporal REconstructing RWKV (Otter). Specifically, the Compound Segmentation Module~(CSM) is devised to segment and emphasize key patches in each frame, effectively highlighting subjects against background information. The Temporal Reconstruction Module (TRM) is incorporated into the temporal-enhanced prototype construction to enable bidirectional scanning, allowing better reconstruct temporal relation. Furthermore, a regular prototype is combined with the temporal-enhanced prototype to simultaneously enhance subject emphasis and temporal modeling, improving wide-angle FSAR performance. Extensive experiments on benchmarks such as SSv2, Kinetics, UCF101, and HMDB51 demonstrate that Otter achieves state-of-the-art performance. Extra evaluation on the VideoBadminton dataset further validates the superiority of Otter in wide-angle FSAR.
>
---
#### [replaced 058] Rank Matters: Understanding and Defending Model Inversion Attacks via Low-Rank Feature Filtering
- **分类: cs.CR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2410.05814v4](https://arxiv.org/pdf/2410.05814v4)**

> **作者:** Hongyao Yu; Yixiang Qiu; Hao Fang; Tianqu Zhuang; Bin Chen; Sijin Yu; Bin Wang; Shu-Tao Xia; Ke Xu
>
> **备注:** KDD 2026 Accept
>
> **摘要:** Model Inversion Attacks (MIAs) pose a significant threat to data privacy by reconstructing sensitive training samples from the knowledge embedded in trained machine learning models. Despite recent progress in enhancing the effectiveness of MIAs across diverse settings, defense strategies have lagged behind, struggling to balance model utility with robustness against increasingly sophisticated attacks. In this work, we propose the ideal inversion error to measure the privacy leakage, and our theoretical and empirical investigations reveals that higher-rank features are inherently more prone to privacy leakage. Motivated by this insight, we propose a lightweight and effective defense strategy based on low-rank feature filtering, which explicitly reduces the attack surface by constraining the dimension of intermediate representations. Extensive experiments across various model architectures and datasets demonstrate that our method consistently outperforms existing defenses, achieving state-of-the-art performance against a wide range of MIAs. Notably, our approach remains effective even in challenging regimes involving high-resolution data and high-capacity models, where prior defenses fail to provide adequate protection. The code is available at https://github.com/Chrisqcwx/LoFt .
>
---
#### [replaced 059] LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LiDARCrafter，面向动态4D LiDAR场景生成与编辑任务。针对现有方法在可控性、时序一致性及评估标准上的不足，构建基于语言指令的统一框架，通过三分支扩散网络生成物体结构、运动轨迹与几何，并结合自回归模块实现时序连贯生成。建立标准化评估基准，实现在nuScenes数据集上的先进性能。**

- **链接: [https://arxiv.org/pdf/2508.03692v3](https://arxiv.org/pdf/2508.03692v3)**

> **作者:** Ao Liang; Youquan Liu; Yu Yang; Dongyue Lu; Linfeng Li; Lingdong Kong; Huaici Zhao; Wei Tsang Ooi
>
> **备注:** AAAI 2026 Oral Presentation; 38 pages, 18 figures, 12 tables; Project Page at https://lidarcrafter.github.io
>
> **摘要:** Generative world models have become essential data engines for autonomous driving, yet most existing efforts focus on videos or occupancy grids, overlooking the unique LiDAR properties. Extending LiDAR generation to dynamic 4D world modeling presents challenges in controllability, temporal coherence, and evaluation standardization. To this end, we present LiDARCrafter, a unified framework for 4D LiDAR generation and editing. Given free-form natural language inputs, we parse instructions into ego-centric scene graphs, which condition a tri-branch diffusion network to generate object structures, motion trajectories, and geometry. These structured conditions enable diverse and fine-grained scene editing. Additionally, an autoregressive module generates temporally coherent 4D LiDAR sequences with smooth transitions. To support standardized evaluation, we establish a comprehensive benchmark with diverse metrics spanning scene-, object-, and sequence-level aspects. Experiments on the nuScenes dataset using this benchmark demonstrate that LiDARCrafter achieves state-of-the-art performance in fidelity, controllability, and temporal consistency across all levels, paving the way for data augmentation and simulation. The code and benchmark are released to the community.
>
---
#### [replaced 060] AlignBench: Benchmarking Fine-Grained Image-Text Alignment with Synthetic Image-Caption Pairs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20515v3](https://arxiv.org/pdf/2511.20515v3)**

> **作者:** Kuniaki Saito; Risa Shinoda; Shohei Tanaka; Tosho Hirasawa; Fumio Okura; Yoshitaka Ushiku
>
> **备注:** Project Page: https://dahlian00.github.io/AlignBench/
>
> **摘要:** Assessing image-text alignment models such as CLIP is crucial for bridging visual and linguistic representations. Yet existing benchmarks rely on rule-based perturbations or short captions, limiting their ability to measure fine-grained alignment. We introduce AlignBench, a benchmark that provides a new indicator of image-text alignment by evaluating detailed image-caption pairs generated by diverse image-to-text and text-to-image models. Each sentence is annotated for correctness, enabling direct assessment of VLMs as alignment evaluators. Benchmarking a wide range of decoder-based VLMs reveals three key findings: (i) CLIP-based models, even those tailored for compositional reasoning, remain nearly blind; (ii) detectors systematically over-score early sentences; and (iii) they show strong self-preference, favoring their own outputs and harming detection performance. Our project page will be available at https://dahlian00.github.io/AlignBench/.
>
---
#### [replaced 061] Bias Beyond Demographics: Probing Decision Boundaries in Black-Box LVLMs via Counterfactual VQA
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03079v2](https://arxiv.org/pdf/2508.03079v2)**

> **作者:** Zaiying Zhao; Toshihiko Yamasaki
>
> **摘要:** Recent advances in large vision-language models (LVLMs) have amplified concerns about fairness, yet existing evaluations remain confined to demographic attributes and often conflate fairness with refusal behavior. This paper broadens the scope of fairness by introducing a counterfactual VQA benchmark that probes the decision boundaries of closed-source LVLMs under controlled context shifts. Each image pair differs in a single visual attribute that has been validated as irrelevant to the question, enabling ground-truth-free and refusal-aware analysis of reasoning stability. Comprehensive experiments reveal that non-demographic attributes, such as environmental context or social behavior, distort LVLM decision-making more strongly than demographic ones. Moreover, instruction-based debiasing shows limited effectiveness and can even amplify these asymmetries, whereas exposure to a small number of human norm validated examples from our benchmark encourages more consistent and balanced responses, highlighting its potential not only as an evaluative framework but also as a means for understanding and improving model behavior. Together, these results provide a practial basis for auditing contextual biases even in black-box LVLMs and contribute to more transparent and equitable multimodal reasoning.
>
---
#### [replaced 062] End-to-End Multi-Person Pose Estimation with Pose-Aware Video Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13208v2](https://arxiv.org/pdf/2511.13208v2)**

> **作者:** Yonghui Yu; Jiahang Cai; Xun Wang; Wenwu Yang
>
> **摘要:** Existing multi-person video pose estimation methods typically adopt a two-stage pipeline: detecting individuals in each frame, followed by temporal modeling for single person pose estimation. This design relies on heuristic operations such as detection, RoI cropping, and non-maximum suppression (NMS), limiting both accuracy and efficiency. In this paper, we present a fully end-to-end framework for multi-person 2D pose estimation in videos, effectively eliminating heuristic operations. A key challenge is to associate individuals across frames under complex and overlapping temporal trajectories. To address this, we introduce a novel Pose-Aware Video transformEr Network (PAVE-Net), which features a spatial encoder to model intra-frame relations and a spatiotemporal pose decoder to capture global dependencies across frames. To achieve accurate temporal association, we propose a pose-aware attention mechanism that enables each pose query to selectively aggregate features corresponding to the same individual across consecutive frames. Additionally, we explicitly model spatiotemporal dependencies among pose keypoints to improve accuracy. Notably, our approach is the first end-to-end method for multi-frame 2D human pose estimation. Extensive experiments show that PAVE-Net substantially outperforms prior image-based end-to-end methods, achieving a 6.0 mAP improvement on PoseTrack2017, and delivers accuracy competitive with state-of-the-art two-stage video based approaches, while offering significant gains in efficiency. Project page: https://github.com/zgspose/PAVENet.
>
---
#### [replaced 063] Random forest-based out-of-distribution detection for robust lung cancer segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.19112v2](https://arxiv.org/pdf/2508.19112v2)**

> **作者:** Aneesh Rangnekar; Harini Veeraraghavan
>
> **备注:** Accepted at SPIE Medical Imaging 2026
>
> **摘要:** Accurate detection and segmentation of cancerous lesions from computed tomography (CT) scans is essential for automated treatment planning and cancer treatment response assessment. Transformer-based models with self-supervised pretraining have achieved strong performance on in-distribution (ID) data but often generalize poorly on out-of-distribution (OOD) inputs. We investigate this behavior for lung cancer segmentation using an encoder-decoder model. Our encoder is a Swin Transformer pretrained with masked image modeling (SimMIM) on 10,432 unlabeled 3D CT scans spanning cancerous and non-cancerous conditions, and the decoder was randomly initialized. This model was evaluated on an independent ID test set and four OOD scenarios, including chest CT cohorts (pulmonary embolism and negative COVID-19) and abdomen CT cohorts (kidney cancers and non-cancerous pancreas). OOD detection was performed at the scan level using RF-Deep, a random forest classifier applied to contextual tumor-anchored feature representations. We evaluated 920 3D CTs (172,650 images) and observed that RF-Deep achieved FPR95 values of 18.26% and 27.66% on the chest CT cohorts, and near-perfect detection (less than 0.1% FPR95) on the abdomen CT cohorts, consistently outperforming established OOD methods. These results demonstrate that our RF-Deep classifier provides a simple, lightweight, and effective approach for enhancing the reliability of segmentation models in clinical deployment.
>
---
#### [replaced 064] 3DIS: Depth-Driven Decoupled Instance Synthesis for Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.12669v2](https://arxiv.org/pdf/2410.12669v2)**

> **作者:** Dewei Zhou; Ji Xie; Zongxin Yang; Yi Yang
>
> **备注:** 10 pages
>
> **摘要:** The increasing demand for controllable outputs in text-to-image generation has spurred advancements in multi-instance generation (MIG), allowing users to define both instance layouts and attributes. However, unlike image-conditional generation methods such as ControlNet, MIG techniques have not been widely adopted in state-of-the-art models like SD2 and SDXL, primarily due to the challenge of building robust renderers that simultaneously handle instance positioning and attribute rendering. In this paper, we introduce Depth-Driven Decoupled Instance Synthesis (3DIS), a novel framework that decouples the MIG process into two stages: (i) generating a coarse scene depth map for accurate instance positioning and scene composition, and (ii) rendering fine-grained attributes using pre-trained ControlNet on any foundational model, without additional training. Our 3DIS framework integrates a custom adapter into LDM3D for precise depth-based layouts and employs a finetuning-free method for enhanced instance-level attribute rendering. Extensive experiments on COCO-Position and COCO-MIG benchmarks demonstrate that 3DIS significantly outperforms existing methods in both layout precision and attribute rendering. Notably, 3DIS offers seamless compatibility with diverse foundational models, providing a robust, adaptable solution for advanced multi-instance generation. The code is available at: https://github.com/limuloo/3DIS.
>
---
#### [replaced 065] Self-Calibrating BCIs: Ranking and Recovery of Mental Targets Without Labels
- **分类: cs.CV; cs.HC**

- **链接: [https://arxiv.org/pdf/2506.11151v2](https://arxiv.org/pdf/2506.11151v2)**

> **作者:** Jonathan Grizou; Carlos de la Torre-Ortiz; Tuukka Ruotsalo
>
> **备注:** 10 pages, 4 figures, 11 appendix pages, 7 appendix figures
>
> **摘要:** We consider the problem of recovering a mental target (e.g., an image of a face) that a participant has in mind from paired EEG (i.e., brain responses) and image (i.e., perceived faces) data collected during interactive sessions without access to labeled information. The problem has been previously explored with labeled data but not via self-calibration, where labeled data is unavailable. Here, we present the first framework and an algorithm, CURSOR, that learns to recover unknown mental targets without access to labeled data or pre-trained decoders. Our experiments on naturalistic images of faces demonstrate that CURSOR can (1) predict image similarity scores that correlate with human perceptual judgments without any label information, (2) use these scores to rank stimuli against an unknown mental target, and (3) generate new stimuli indistinguishable from the unknown mental target (validated via a user study, N=53).
>
---
#### [replaced 066] MasHeNe: A Benchmark for Head and Neck CT Mass Segmentation using Window-Enhanced Mamba with Frequency-Domain Integration
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.01563v2](https://arxiv.org/pdf/2512.01563v2)**

> **作者:** Thao Thi Phuong Dao; Tan-Cong Nguyen; Nguyen Chi Thanh; Truong Hoang Viet; Trong-Le Do; Mai-Khiem Tran; Minh-Khoi Pham; Trung-Nghia Le; Minh-Triet Tran; Thanh Dinh Le
>
> **备注:** The 14th International Symposium on Information and Communication Technology Conference SoICT 2025
>
> **摘要:** Head and neck masses are space-occupying lesions that can compress the airway and esophagus and may affect nerves and blood vessels. Available public datasets primarily focus on malignant lesions and often overlook other space-occupying conditions in this region. To address this gap, we introduce MasHeNe, an initial dataset of 3,779 contrast-enhanced CT slices that includes both tumors and cysts with pixel-level annotations. We also establish a benchmark using standard segmentation baselines and report common metrics to enable fair comparison. In addition, we propose the Windowing-Enhanced Mamba with Frequency integration (WEMF) model. WEMF applies tri-window enhancement to enrich the input appearance before feature extraction. It further uses multi-frequency attention to fuse information across skip connections within a U-shaped Mamba backbone. On MasHeNe, WEMF attains the best performance among evaluated methods, with a Dice of 70.45%, IoU of 66.89%, NSD of 72.33%, and HD95 of 5.12 mm. This model indicates stable and strong results on this challenging task. MasHeNe provides a benchmark for head-and-neck mass segmentation beyond malignancy-only datasets. The observed error patterns also suggest that this task remains challenging and requires further research. Our dataset and code are available at https://github.com/drthaodao3101/MasHeNe.git.
>
---
#### [replaced 067] Self-Supervised Compression and Artifact Correction for Streaming Underwater Imaging Sonar
- **分类: eess.IV; cs.CV; cs.LG; cs.MM**

- **链接: [https://arxiv.org/pdf/2511.13922v2](https://arxiv.org/pdf/2511.13922v2)**

> **作者:** Rongsheng Qian; Chi Xu; Xiaoqiang Ma; Hao Fang; Yili Jin; William I. Atlas; Jiangchuan Liu
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Real-time imaging sonar is crucial for underwater monitoring where optical sensing fails, but its use is limited by low uplink bandwidth and severe sonar-specific artifacts (speckle, motion blur, reverberation, acoustic shadows) affecting up to 98% of frames. We present SCOPE, a self-supervised framework that jointly performs compression and artifact correction without clean-noise pairs or synthetic assumptions. SCOPE combines (i) Adaptive Codebook Compression (ACC), which learns frequency-encoded latent representations tailored to sonar, with (ii) Frequency-Aware Multiscale Segmentation (FAMS), which decomposes frames into low-frequency structure and sparse high-frequency dynamics while suppressing rapidly fluctuating artifacts. A hedging training strategy further guides frequency-aware learning using low-pass proxy pairs generated without labels. Evaluated on months of in-situ ARIS sonar data, SCOPE achieves a structural similarity index (SSIM) of 0.77, representing a 40% improvement over prior self-supervised denoising baselines, at bitrates down to <= 0.0118 bpp. It reduces uplink bandwidth by more than 80% while improving downstream detection. The system runs in real time, with 3.1 ms encoding on an embedded GPU and 97 ms full multi-layer decoding on the server end. SCOPE has been deployed for months in three Pacific Northwest rivers to support real-time salmon enumeration and environmental monitoring in the wild. Results demonstrate that learning frequency-structured latents enables practical, low-bitrate sonar streaming with preserved signal details under real-world deployment conditions.
>
---
#### [replaced 068] Alligat0R: Pre-Training Through Co-Visibility Segmentation for Relative Camera Pose Regression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.07561v2](https://arxiv.org/pdf/2503.07561v2)**

> **作者:** Thibaut Loiseau; Guillaume Bourmaud; Vincent Lepetit
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Pre-training techniques have greatly advanced computer vision, with CroCo's cross-view completion approach yielding impressive results in tasks like 3D reconstruction and pose regression. However, cross-view completion is ill-posed in non-covisible regions, limiting its effectiveness. We introduce Alligat0R, a novel pre-training approach that replaces cross-view learning with a covisibility segmentation task. Our method predicts whether each pixel in one image is covisible in the second image, occluded, or outside the field of view, making the pre-training effective in both covisible and non-covisible regions, and provides interpretable predictions. To support this, we present Cub3, a large-scale dataset with 5M image pairs and dense covisibility annotations derived from the nuScenes and ScanNet datasets. Cub3 includes diverse scenarios with varying degrees of overlap. The experiments show that our novel pre-training method Alligat0R significantly outperforms CroCo in relative pose regression. Code is available at https://github.com/thibautloiseau/alligat0r.
>
---
#### [replaced 069] Learning Massively Multitask World Models for Continuous Control
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文研究连续控制中的多任务强化学习问题，旨在实现单一智能体在数百个任务间的高效适应。提出新基准与语言条件的多任务世界模型Newt，通过演示预训练和在线联合优化，提升数据效率与泛化能力，支持快速零样本迁移。**

- **链接: [https://arxiv.org/pdf/2511.19584v2](https://arxiv.org/pdf/2511.19584v2)**

> **作者:** Nicklas Hansen; Hao Su; Xiaolong Wang
>
> **备注:** Webpage: https://www.nicklashansen.com/NewtWM
>
> **摘要:** General-purpose control demands agents that act across many tasks and embodiments, yet research on reinforcement learning (RL) for continuous control remains dominated by single-task or offline regimes, reinforcing a view that online RL does not scale. Inspired by the foundation model recipe (large-scale pretraining followed by light RL) we ask whether a single agent can be trained on hundreds of tasks with online interaction. To accelerate research in this direction, we introduce a new benchmark with 200 diverse tasks spanning many domains and embodiments, each with language instructions, demonstrations, and optionally image observations. We then present \emph{Newt}, a language-conditioned multitask world model that is first pretrained on demonstrations to acquire task-aware representations and action priors, and then jointly optimized with online interaction across all tasks. Experiments show that Newt yields better multitask performance and data-efficiency than a set of strong baselines, exhibits strong open-loop control, and enables rapid adaptation to unseen tasks. We release our environments, demonstrations, code for training and evaluation, as well as 200+ checkpoints.
>
---
#### [replaced 070] Cross-Cancer Knowledge Transfer in WSI-based Prognosis Prediction
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13482v4](https://arxiv.org/pdf/2508.13482v4)**

> **作者:** Pei Liu; Luping Ji; Jiaxiang Gou; Xiangxiang Zeng
>
> **备注:** 24 pages (11 figures and 10 tables)
>
> **摘要:** Whole-Slide Image (WSI) is an important tool for estimating cancer prognosis. Current studies generally follow a conventional cancer-specific paradigm in which each cancer corresponds to a single model. However, this paradigm naturally struggles to scale to rare tumors and cannot leverage knowledge from other cancers. While multi-task learning frameworks have been explored recently, they often place high demands on computational resources and require extensive training on ultra-large, multi-cancer WSI datasets. To this end, this paper shifts the paradigm to knowledge transfer and presents the first preliminary yet systematic study on cross-cancer prognosis knowledge transfer in WSIs, called CROPKT. It comprises three major parts. (1) We curate a large dataset (UNI2-h-DSS) with 26 cancers and use it to measure the transferability of WSI-based prognostic knowledge across different cancers (including rare tumors). (2) Beyond a simple evaluation merely for benchmarking, we design a range of experiments to gain deeper insights into the underlying mechanism behind transferability. (3) We further show the utility of cross-cancer knowledge transfer, by proposing a routing-based baseline approach (ROUPKT) that could often efficiently utilize the knowledge transferred from off-the-shelf models of other cancers. CROPKT could serve as an inception that lays the foundation for this nascent paradigm, i.e., WSI-based prognosis prediction with cross-cancer knowledge transfer. Our source code is available at https://github.com/liupei101/CROPKT.
>
---
#### [replaced 071] OmniBench: Towards The Future of Universal Omni-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出OmniBench基准，评估模型在视觉、听觉与文本三模态下的理解与推理能力，旨在推动通用全语言模型（OLMs）发展。针对现有模型在多模态指令遵循和推理上的不足，构建高质量标注数据集并提出OmniInstruct训练数据，以提升模型三模态融合能力。**

- **链接: [https://arxiv.org/pdf/2409.15272v5](https://arxiv.org/pdf/2409.15272v5)**

> **作者:** Yizhi Li; Ge Zhang; Yinghao Ma; Ruibin Yuan; Kang Zhu; Hangyu Guo; Yiming Liang; Jiaheng Liu; Zekun Wang; Jian Yang; Siwei Wu; Xingwei Qu; Jinjie Shi; Xinyue Zhang; Zhenzhu Yang; Xiangzhou Wang; Zhaoxiang Zhang; Zachary Liu; Emmanouil Benetos; Wenhao Huang; Chenghua Lin
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have focused on integrating multiple modalities, yet their ability to simultaneously process and reason across different inputs remains underexplored. We introduce OmniBench, a novel benchmark designed to evaluate models' ability to recognize, interpret, and reason across visual, acoustic, and textual inputs simultaneously. We define language models capable of such tri-modal processing as omni-language models (OLMs). OmniBench features high-quality human annotations that require integrated understanding across all modalities. Our evaluation reveals that: i) open-source OLMs show significant limitations in instruction-following and reasoning in tri-modal contexts; and ii) most baseline models perform poorly (around 50% accuracy) even with textual alternatives to image/audio inputs. To address these limitations, we develop OmniInstruct, an 96K-sample instruction tuning dataset for training OLMs. We advocate for developing more robust tri-modal integration techniques and training strategies to enhance OLM performance. Codes and data could be found at our repo (https://github.com/multimodal-art-projection/OmniBench).
>
---
#### [replaced 072] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.09809v2](https://arxiv.org/pdf/2511.09809v2)**

> **作者:** Konstantinos M. Dafnis; Dimitris N. Metaxas
>
> **备注:** NeurIPS 2025
>
> **摘要:** Vision-Language Models (VLMs) excel at zero-shot inference but often degrade under test-time domain shifts. For this reason, episodic test-time adaptation strategies have recently emerged as powerful techniques for adapting VLMs to a single unlabeled image. However, existing adaptation strategies, such as test-time prompt tuning, typically require backpropagating through large encoder weights or altering core model components. In this work, we introduce Spectrum-Aware Test-Time Steering (STS), a lightweight adaptation framework that extracts a spectral subspace from the textual embeddings to define principal semantic directions and learns to steer latent representations in a spectrum-aware manner by adapting a small number of per-sample shift parameters to minimize entropy across augmented views. STS operates entirely at inference in the latent space, without backpropagation through or modification of the frozen encoders. Building on standard evaluation protocols, our comprehensive experiments demonstrate that STS largely surpasses or compares favorably against state-of-the-art test-time adaptation methods, while introducing only a handful of additional parameters and achieving inference speeds up to 8x faster with a 12x smaller memory footprint than conventional test-time prompt tuning. The code is available at https://github.com/kdafnis/STS.
>
---
