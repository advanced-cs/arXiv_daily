# 计算机视觉 cs.CV

- **最新发布 91 篇**

- **更新 48 篇**

## 最新发布

#### [new 001] Mask-to-Height: A YOLOv11-Based Architecture for Joint Building Instance Segmentation and Height Classification from Satellite Imagery
- **分类: cs.CV**

- **简介: 该论文提出基于YOLOv11的联合建筑实例分割与高度分类方法，解决卫星影像中建筑提取与高度划分难题。利用DFC2023数据集，实现高精度分割与分类，显著提升复杂场景下的检测性能与推理速度，推动城市三维重建与遥感智能分析。**

- **链接: [http://arxiv.org/pdf/2510.27224v1](http://arxiv.org/pdf/2510.27224v1)**

> **作者:** Mahmoud El Hussieni; Bahadır K. Güntürk; Hasan F. Ateş; Oğuz Hanoğlu
>
> **摘要:** Accurate building instance segmentation and height classification are critical for urban planning, 3D city modeling, and infrastructure monitoring. This paper presents a detailed analysis of YOLOv11, the recent advancement in the YOLO series of deep learning models, focusing on its application to joint building extraction and discrete height classification from satellite imagery. YOLOv11 builds on the strengths of earlier YOLO models by introducing a more efficient architecture that better combines features at different scales, improves object localization accuracy, and enhances performance in complex urban scenes. Using the DFC2023 Track 2 dataset -- which includes over 125,000 annotated buildings across 12 cities -- we evaluate YOLOv11's performance using metrics such as precision, recall, F1 score, and mean average precision (mAP). Our findings demonstrate that YOLOv11 achieves strong instance segmentation performance with 60.4\% mAP@50 and 38.3\% mAP@50--95 while maintaining robust classification accuracy across five predefined height tiers. The model excels in handling occlusions, complex building shapes, and class imbalance, particularly for rare high-rise structures. Comparative analysis confirms that YOLOv11 outperforms earlier multitask frameworks in both detection accuracy and inference speed, making it well-suited for real-time, large-scale urban mapping. This research highlights YOLOv11's potential to advance semantic urban reconstruction through streamlined categorical height modeling, offering actionable insights for future developments in remote sensing and geospatial intelligence.
>
---
#### [new 002] Can MLLMs Read the Room? A Multimodal Benchmark for Verifying Truthfulness in Multi-Party Social Interactions
- **分类: cs.CV; cs.CL; cs.SI**

- **简介: 该论文提出多模态互动真实性评估（MIVA）任务，旨在检测多人群体社交互动中的说谎行为。针对现有模型难以融合视觉与语言线索的问题，构建基于狼人杀游戏的同步视频-文本数据集，建立基准测试。结果表明，即使先进MLLMs如GPT-4o也表现不佳，暴露出对社会线索感知不足的缺陷。**

- **链接: [http://arxiv.org/pdf/2510.27195v1](http://arxiv.org/pdf/2510.27195v1)**

> **作者:** Caixin Kang; Yifei Huang; Liangyang Ouyang; Mingfang Zhang; Yoichi Sato
>
> **摘要:** As AI systems become increasingly integrated into human lives, endowing them with robust social intelligence has emerged as a critical frontier. A key aspect of this intelligence is discerning truth from deception, a ubiquitous element of human interaction that is conveyed through a complex interplay of verbal language and non-verbal visual cues. However, automatic deception detection in dynamic, multi-party conversations remains a significant challenge. The recent rise of powerful Multimodal Large Language Models (MLLMs), with their impressive abilities in visual and textual understanding, makes them natural candidates for this task. Consequently, their capabilities in this crucial domain are mostly unquantified. To address this gap, we introduce a new task, Multimodal Interactive Veracity Assessment (MIVA), and present a novel multimodal dataset derived from the social deduction game Werewolf. This dataset provides synchronized video, text, with verifiable ground-truth labels for every statement. We establish a comprehensive benchmark evaluating state-of-the-art MLLMs, revealing a significant performance gap: even powerful models like GPT-4o struggle to distinguish truth from falsehood reliably. Our analysis of failure modes indicates that these models fail to ground language in visual social cues effectively and may be overly conservative in their alignment, highlighting the urgent need for novel approaches to building more perceptive and trustworthy AI systems.
>
---
#### [new 003] HyperClick: Advancing Reliable GUI Grounding via Uncertainty Calibration
- **分类: cs.CV**

- **简介: 该论文针对GUI自动化中模型过自信导致的可靠性问题，提出HyperClick框架。通过双奖励机制与基于Brier分数的不确定性校准，联合优化定位准确率与置信度可靠性，实现可信赖的GUI接地，显著提升动态任务中的执行鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.27266v1](http://arxiv.org/pdf/2510.27266v1)**

> **作者:** Shaojie Zhang; Pei Fu; Ruoceng Zhang; Jiahui Yang; Anan Du; Xiuwen Xi; Shaokang Wang; Ying Huang; Bin Qin; Zhenbo Luo; Jian Luan
>
> **摘要:** Autonomous Graphical User Interface (GUI) agents rely on accurate GUI grounding, which maps language instructions to on-screen coordinates, to execute user commands. However, current models, whether trained via supervised fine-tuning (SFT) or reinforcement fine-tuning (RFT), lack self-awareness of their capability boundaries, leading to overconfidence and unreliable predictions. We first systematically evaluate probabilistic and verbalized confidence in general and GUI-specific models, revealing a misalignment between confidence and actual accuracy, which is particularly critical in dynamic GUI automation tasks, where single errors can cause task failure. To address this, we propose HyperClick, a novel framework that enhances reliable GUI grounding through uncertainty calibration. HyperClick introduces a dual reward mechanism, combining a binary reward for correct actions with a truncated Gaussian-based spatial confidence modeling, calibrated using the Brier score. This approach jointly optimizes grounding accuracy and confidence reliability, fostering introspective self-criticism. Extensive experiments on seven challenge benchmarks show that HyperClick achieves state-of-the-art performance while providing well-calibrated confidence. By enabling explicit confidence calibration and introspective self-criticism, HyperClick reduces overconfidence and supports more reliable GUI automation.
>
---
#### [new 004] Improving Cross-view Object Geo-localization: A Dual Attention Approach with Cross-view Interaction and Multi-Scale Spatial Features
- **分类: cs.CV**

- **简介: 该论文针对“地面到无人机”跨视角物体地理定位任务，解决现有方法信息交互不足、噪声干扰导致定位不准的问题。提出双注意力模块CVCAM实现视图间迭代交互，融合多尺度空间特征，提升特征表达与定位精度，并构建新数据集G2D以弥补数据稀缺。**

- **链接: [http://arxiv.org/pdf/2510.27139v1](http://arxiv.org/pdf/2510.27139v1)**

> **作者:** Xingtao Ling Yingying Zhu
>
> **摘要:** Cross-view object geo-localization has recently gained attention due to potential applications. Existing methods aim to capture spatial dependencies of query objects between different views through attention mechanisms to obtain spatial relationship feature maps, which are then used to predict object locations. Although promising, these approaches fail to effectively transfer information between views and do not further refine the spatial relationship feature maps. This results in the model erroneously focusing on irrelevant edge noise, thereby affecting localization performance. To address these limitations, we introduce a Cross-view and Cross-attention Module (CVCAM), which performs multiple iterations of interaction between the two views, enabling continuous exchange and learning of contextual information about the query object from both perspectives. This facilitates a deeper understanding of cross-view relationships while suppressing the edge noise unrelated to the query object. Furthermore, we integrate a Multi-head Spatial Attention Module (MHSAM), which employs convolutional kernels of various sizes to extract multi-scale spatial features from the feature maps containing implicit correspondences, further enhancing the feature representation of the query object. Additionally, given the scarcity of datasets for cross-view object geo-localization, we created a new dataset called G2D for the "Ground-to-Drone" localization task, enriching existing datasets and filling the gap in "Ground-to-Drone" localization task. Extensive experiments on the CVOGL and G2D datasets demonstrate that our proposed method achieves high localization accuracy, surpassing the current state-of-the-art.
>
---
#### [new 005] AD-SAM: Fine-Tuning the Segment Anything Vision Foundation Model for Autonomous Driving Perception
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶场景下的语义分割任务，提出AD-SAM模型。通过双编码器与可变形解码器增强对复杂道路场景的感知能力，结合混合损失函数提升精度与泛化性。实验表明，该模型在多个基准上显著优于现有方法，具备高效率、强泛化与数据高效特性。**

- **链接: [http://arxiv.org/pdf/2510.27047v1](http://arxiv.org/pdf/2510.27047v1)**

> **作者:** Mario Camarena; Het Patel; Fatemeh Nazari; Evangelos Papalexakis; Mohamadhossein Noruzoliaee; Jia Chen
>
> **备注:** Submitted to IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)
>
> **摘要:** This paper presents the Autonomous Driving Segment Anything Model (AD-SAM), a fine-tuned vision foundation model for semantic segmentation in autonomous driving (AD). AD-SAM extends the Segment Anything Model (SAM) with a dual-encoder and deformable decoder tailored to spatial and geometric complexity of road scenes. The dual-encoder produces multi-scale fused representations by combining global semantic context from SAM's pretrained Vision Transformer (ViT-H) with local spatial detail from a trainable convolutional deep learning backbone (i.e., ResNet-50). A deformable fusion module aligns heterogeneous features across scales and object geometries. The decoder performs progressive multi-stage refinement using deformable attention. Training is guided by a hybrid loss that integrates Focal, Dice, Lovasz-Softmax, and Surface losses, improving semantic class balance, boundary precision, and optimization stability. Experiments on the Cityscapes and Berkeley DeepDrive 100K (BDD100K) benchmarks show that AD-SAM surpasses SAM, Generalized SAM (G-SAM), and a deep learning baseline (DeepLabV3) in segmentation accuracy. It achieves 68.1 mean Intersection over Union (mIoU) on Cityscapes and 59.5 mIoU on BDD100K, outperforming SAM, G-SAM, and DeepLabV3 by margins of up to +22.9 and +19.2 mIoU in structured and diverse road scenes, respectively. AD-SAM demonstrates strong cross-domain generalization with a 0.87 retention score (vs. 0.76 for SAM), and faster, more stable learning dynamics, converging within 30-40 epochs, enjoying double the learning speed of benchmark models. It maintains 0.607 mIoU with only 1000 samples, suggesting data efficiency critical for reducing annotation costs. These results confirm that targeted architectural and optimization enhancements to foundation models enable reliable and scalable AD perception.
>
---
#### [new 006] CoMViT: An Efficient Vision Backbone for Supervised Classification in Medical Imaging
- **分类: cs.CV; cs.AI; I.2.10**

- **简介: 该论文针对医疗图像分类中视觉变压器（ViT）计算成本高、小样本易过拟合的问题，提出轻量级架构CoMViT。通过卷积分词器、对角掩码等设计，在12个MedMNIST数据集上实现高效准确分类，参数仅4.5M，显著优于同类模型，且具备临床可解释性。**

- **链接: [http://arxiv.org/pdf/2510.27442v1](http://arxiv.org/pdf/2510.27442v1)**

> **作者:** Aon Safdar; Mohamed Saadeldin
>
> **备注:** Preprint (submitted manuscript). Accepted at the MICCAI 2025 MIRASOL Workshop; to appear in the Springer proceedings volume. This is the pre-review version (not the Version of Record). DOI will be added after publication. [Optional: 8 pages, 4 figures, 4 tables.]
>
> **摘要:** Vision Transformers (ViTs) have demonstrated strong potential in medical imaging; however, their high computational demands and tendency to overfit on small datasets limit their applicability in real-world clinical scenarios. In this paper, we present CoMViT, a compact and generalizable Vision Transformer architecture optimized for resource-constrained medical image analysis. CoMViT integrates a convolutional tokenizer, diagonal masking, dynamic temperature scaling, and pooling-based sequence aggregation to improve performance and generalization. Through systematic architectural optimization, CoMViT achieves robust performance across twelve MedMNIST datasets while maintaining a lightweight design with only ~4.5M parameters. It matches or outperforms deeper CNN and ViT variants, offering up to 5-20x parameter reduction without sacrificing accuracy. Qualitative Grad-CAM analyses show that CoMViT consistently attends to clinically relevant regions despite its compact size. These results highlight the potential of principled ViT redesign for developing efficient and interpretable models in low-resource medical imaging settings.
>
---
#### [new 007] VitalLens 2.0: High-Fidelity rPPG for Heart Rate Variability Estimation from Face Video
- **分类: cs.CV; cs.HC; 68T45; I.4.9; J.3**

- **简介: 该论文提出VitalLens 2.0，用于从人脸视频中高保真估计生理信号。针对远程光体积描记法（rPPG）精度不足的问题，通过改进模型架构和扩大多样化训练数据，显著提升心率（HR）、呼吸率（RR）及心率变异性（HRV）的估计准确性，达到当前最优性能。**

- **链接: [http://arxiv.org/pdf/2510.27028v1](http://arxiv.org/pdf/2510.27028v1)**

> **作者:** Philipp V. Rouast
>
> **备注:** Technical Report. 8 pages, 5 figures. Introduces the VitalLens 2.0 model for rPPG and Heart Rate Variability (HRV) estimation. Project website: https://rouast.com/api
>
> **摘要:** This report introduces VitalLens 2.0, a new deep learning model for estimating physiological signals from face video. This new model demonstrates a significant leap in accuracy for remote photoplethysmography (rPPG), enabling the robust estimation of not only heart rate (HR) and respiratory rate (RR) but also Heart Rate Variability (HRV) metrics. This advance is achieved through a combination of a new model architecture and a substantial increase in the size and diversity of our training data, now totaling 1,413 unique individuals. We evaluate VitalLens 2.0 on a new, combined test set of 422 unique individuals from four public and private datasets. When averaging results by individual, VitalLens 2.0 achieves a Mean Absolute Error (MAE) of 1.57 bpm for HR, 1.08 bpm for RR, 10.18 ms for HRV-SDNN, and 16.45 ms for HRV-RMSSD. These results represent a new state-of-the-art, significantly outperforming previous methods. This model is now available to developers via the VitalLens API at https://rouast.com/api.
>
---
#### [new 008] Incremental Human-Object Interaction Detection with Invariant Relation Representation Learning
- **分类: cs.CV**

- **简介: 该论文研究增量式人-物交互检测（IHOID）任务，旨在应对开放世界中交互关系动态演化带来的灾难性遗忘、交互漂移及零样本组合检测难题。提出无实例的关系蒸馏框架，通过解耦物体与关系学习，设计两种新蒸馏损失，实现关系特征的不变性建模，有效提升模型在持续学习中的鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.27020v1](http://arxiv.org/pdf/2510.27020v1)**

> **作者:** Yana Wei; Zeen Chi; Chongyu Wang; Yu Wu; Shipeng Yan; Yongfei Liu; Xuming He
>
> **摘要:** In open-world environments, human-object interactions (HOIs) evolve continuously, challenging conventional closed-world HOI detection models. Inspired by humans' ability to progressively acquire knowledge, we explore incremental HOI detection (IHOID) to develop agents capable of discerning human-object relations in such dynamic environments. This setup confronts not only the common issue of catastrophic forgetting in incremental learning but also distinct challenges posed by interaction drift and detecting zero-shot HOI combinations with sequentially arriving data. Therefore, we propose a novel exemplar-free incremental relation distillation (IRD) framework. IRD decouples the learning of objects and relations, and introduces two unique distillation losses for learning invariant relation features across different HOI combinations that share the same relation. Extensive experiments on HICO-DET and V-COCO datasets demonstrate the superiority of our method over state-of-the-art baselines in mitigating forgetting, strengthening robustness against interaction drift, and generalization on zero-shot HOIs. Code is available at \href{https://github.com/weiyana/ContinualHOI}{this HTTP URL}
>
---
#### [new 009] RegionRAG: Region-level Retrieval-Augumented Generation for Visually-Rich Documents
- **分类: cs.CV**

- **简介: 该论文提出RegionRAG，针对多模态RAG中文档级检索引入冗余视觉内容的问题，将检索单元从文档级细化到区域级。通过混合监督训练和动态语义区域聚合，精准定位相关视觉区域，提升问答准确率与检索效率，显著减少无关视觉令牌使用。**

- **链接: [http://arxiv.org/pdf/2510.27261v1](http://arxiv.org/pdf/2510.27261v1)**

> **作者:** Yinglu Li; Zhiying Lu; Zhihang Liu; Chuanbin Liu; Hongtao Xie
>
> **摘要:** Multi-modal Retrieval-Augmented Generation (RAG) has become a critical method for empowering LLMs by leveraging candidate visual documents. However, current methods consider the entire document as the basic retrieval unit, introducing substantial irrelevant visual content in two ways: 1) Relevant documents often contain large regions unrelated to the query, diluting the focus on salient information; 2) Retrieving multiple documents to increase recall further introduces redundant and irrelevant documents. These redundant contexts distract the model's attention and further degrade the performance. To address this challenge, we propose \modelname, a novel framework that shifts the retrieval paradigm from the document level to the region level. During training, we design a hybrid supervision strategy from both labeled data and unlabeled data to pinpoint relevant patches. During inference, we propose a dynamic pipeline that intelligently groups salient patches into complete semantic regions. By delegating the task of identifying relevant regions to the retriever, \modelname enables the generator to focus solely on concise visual content relevant to queries, improving both efficiency and accuracy. Experiments on six benchmarks demonstrate that RegionRAG achieves state-of-the-art performance. Improves retrieval accuracy by 10.02\% in R@1 on average and increases question answering accuracy by 3.56\% while using only 71.42\% visual tokens compared to prior methods. The code will be available at https://github.com/Aeryn666/RegionRAG.
>
---
#### [new 010] NAUTILUS: A Large Multimodal Model for Underwater Scene Understanding
- **分类: cs.CV**

- **简介: 该论文聚焦水下场景理解任务，针对缺乏大规模多任务数据集及水下图像退化问题，构建了包含145万图文对的NautData数据集，并提出可插拔的视觉特征增强模块，融合物理成像先验以提升模型鲁棒性。基于此，构建了新型多模态模型NAUTILUS，显著提升水下感知性能。**

- **链接: [http://arxiv.org/pdf/2510.27481v1](http://arxiv.org/pdf/2510.27481v1)**

> **作者:** Wei Xu; Cheng Wang; Dingkang Liang; Zongchuang Zhao; Xingyu Jiang; Peng Zhang; Xiang Bai
>
> **备注:** Accepted to NeurIPS 2025. Data and models are available at https://github.com/H-EmbodVis/NAUTILUS
>
> **摘要:** Underwater exploration offers critical insights into our planet and attracts increasing attention for its broader applications in resource exploration, national security, etc. We study the underwater scene understanding methods, which aim to achieve automated underwater exploration. The underwater scene understanding task demands multi-task perceptions from multiple granularities. However, the absence of large-scale underwater multi-task instruction-tuning datasets hinders the progress of this research. To bridge this gap, we construct NautData, a dataset containing 1.45 M image-text pairs supporting eight underwater scene understanding tasks. It enables the development and thorough evaluation of the underwater scene understanding models. Underwater image degradation is a widely recognized challenge that interferes with underwater tasks. To improve the robustness of underwater scene understanding, we introduce physical priors derived from underwater imaging models and propose a plug-and-play vision feature enhancement (VFE) module, which explicitly restores clear underwater information. We integrate this module into renowned baselines LLaVA-1.5 and Qwen2.5-VL and build our underwater LMM, NAUTILUS. Experiments conducted on the NautData and public underwater datasets demonstrate the effectiveness of the VFE module, consistently improving the performance of both baselines on the majority of supported tasks, thus ensuring the superiority of NAUTILUS in the underwater scene understanding area. Data and models are available at https://github.com/H-EmbodVis/NAUTILUS.
>
---
#### [new 011] Object-IR: Leveraging Object Consistency and Mesh Deformation for Self-Supervised Image Retargeting
- **分类: cs.CV**

- **简介: 该论文提出Object-IR，针对图像重定尺寸中的语义区域几何失真问题，通过自监督的网格变形优化实现。利用对象一致性与几何保真约束，无需标注数据，高效生成高质量、无畸变的重定尺寸结果。**

- **链接: [http://arxiv.org/pdf/2510.27236v1](http://arxiv.org/pdf/2510.27236v1)**

> **作者:** Tianli Liao; Ran Wang; Siqing Zhang; Lei Li; Guangen Liu; Chenyang Zhao; Heling Cao; Peng Li
>
> **备注:** Publish in Pattern Recognition
>
> **摘要:** Eliminating geometric distortion in semantically important regions remains an intractable challenge in image retargeting. This paper presents Object-IR, a self-supervised architecture that reformulates image retargeting as a learning-based mesh warping optimization problem, where the mesh deformation is guided by object appearance consistency and geometric-preserving constraints. Given an input image and a target aspect ratio, we initialize a uniform rigid mesh at the output resolution and use a convolutional neural network to predict the motion of each mesh grid and obtain the deformed mesh. The retargeted result is generated by warping the input image according to the rigid mesh in the input image and the deformed mesh in the output resolution. To mitigate geometric distortion, we design a comprehensive objective function incorporating a) object-consistent loss to ensure that the important semantic objects retain their appearance, b) geometric-preserving loss to constrain simple scale transform of the important meshes, and c) boundary loss to enforce a clean rectangular output. Notably, our self-supervised paradigm eliminates the need for manually annotated retargeting datasets by deriving supervision directly from the input's geometric and semantic properties. Extensive evaluations on the RetargetMe benchmark demonstrate that our Object-IR achieves state-of-the-art performance, outperforming existing methods in quantitative metrics and subjective visual quality assessments. The framework efficiently processes arbitrary input resolutions (average inference time: 0.009s for 1024x683 resolution) while maintaining real-time performance on consumer-grade GPUs. The source code will soon be available at https://github.com/tlliao/Object-IR.
>
---
#### [new 012] Understanding the Implicit User Intention via Reasoning with Large Language Model for Image Editing
- **分类: cs.CV**

- **简介: 该论文针对复杂图像编辑中需联合微调大语言模型与扩散模型导致的高计算成本问题，提出CIELR方法。通过构建结构化语义表示并迭代优化，将复杂指令分解为简单操作，实现无需联合微调的高效推理。实验表明其在保真度和基准测试上均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.27335v1](http://arxiv.org/pdf/2510.27335v1)**

> **作者:** Yijia Wang; Yiqing Shen; Weiming Chen; Zhihai He
>
> **摘要:** Existing image editing methods can handle simple editing instructions very well. To deal with complex editing instructions, they often need to jointly fine-tune the large language models (LLMs) and diffusion models (DMs), which involves very high computational complexity and training cost. To address this issue, we propose a new method, called \textbf{C}omplex \textbf{I}mage \textbf{E}diting via \textbf{L}LM \textbf{R}easoning (CIELR), which converts a complex user instruction into a set of simple and explicit editing actions, eliminating the need for jointly fine-tuning the large language models and diffusion models. Specifically, we first construct a structured semantic representation of the input image using foundation models. Then, we introduce an iterative update mechanism that can progressively refine this representation, obtaining a fine-grained visual representation of the image scene. This allows us to perform complex and flexible image editing tasks. Extensive experiments on the SmartEdit Reasoning Scenario Set show that our method surpasses the previous state-of-the-art by 9.955 dB in PSNR, indicating its superior preservation of regions that should remain consistent. Due to the limited number of samples of public datasets of complex image editing with reasoning, we construct a benchmark named CIEBench, containing 86 image samples, together with a metric specifically for reasoning-based image editing. CIELR also outperforms previous methods on this benchmark. The code and dataset are available at \href{https://github.com/Jia-shao/Reasoning-Editing}{https://github.com/Jia-shao/Reasoning-Editing}.
>
---
#### [new 013] FPS: Feedforward-based Parameter Selection For Efficient Fine-Tuning
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对大模型微调中的内存效率问题，提出无梯度的前向参数选择方法FPS。通过单次前向传播筛选关键参数，显著降低峰值内存（约9倍）并加速选择过程（约2倍），在24个视觉任务上达到与先进方法相当的性能，实现了高效、实用的参数高效微调。**

- **链接: [http://arxiv.org/pdf/2510.27359v1](http://arxiv.org/pdf/2510.27359v1)**

> **作者:** Kenneth Yang; Wen-Li Wei; Jen-Chun Lin
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) has emerged as a key strategy for adapting large-scale pre-trained models to downstream tasks, but existing approaches face notable limitations. Addition-based methods, such as Adapters [1], introduce inference latency and engineering complexity, while selection-based methods like Gradient-based Parameter Selection (GPS) [2] require a full backward pass, which results in the same peak memory usage as full fine-tuning. To address this dilemma, we propose Feedforward-based Parameter Selection (FPS), a gradient-free method that identifies an optimal parameter subset in a single forward pass. FPS ranks parameters by the product of their magnitudes and corresponding input activations, leveraging both pre-trained knowledge and downstream data. Evaluated on $24$ visual tasks from FGVC and VTAB-1k, FPS achieves performance comparable to state-of-the-art methods while reducing peak memory usage by nearly $9 \times$ and accelerating parameter selection by about $2 \times$, offering a genuinely memory-efficient and practical solution for fine-tuning large-scale pre-trained models.
>
---
#### [new 014] M^3Detection: Multi-Frame Multi-Level Feature Fusion for Multi-Modal 3D Object Detection with Camera and 4D Imaging Radar
- **分类: cs.CV**

- **简介: 该论文针对相机与4D成像雷达融合的多帧3D目标检测任务，解决单帧信息不全、特征融合效率低及计算冗余问题。提出M^3Detection框架，通过多级特征融合与轨迹引导的时空建模，提升检测精度与效率。**

- **链接: [http://arxiv.org/pdf/2510.27166v1](http://arxiv.org/pdf/2510.27166v1)**

> **作者:** Xiaozhi Li; Huijun Di; Jian Li; Feng Liu; Wei Liang
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Recent advances in 4D imaging radar have enabled robust perception in adverse weather, while camera sensors provide dense semantic information. Fusing the these complementary modalities has great potential for cost-effective 3D perception. However, most existing camera-radar fusion methods are limited to single-frame inputs, capturing only a partial view of the scene. The incomplete scene information, compounded by image degradation and 4D radar sparsity, hinders overall detection performance. In contrast, multi-frame fusion offers richer spatiotemporal information but faces two challenges: achieving robust and effective object feature fusion across frames and modalities, and mitigating the computational cost of redundant feature extraction. Consequently, we propose M^3Detection, a unified multi-frame 3D object detection framework that performs multi-level feature fusion on multi-modal data from camera and 4D imaging radar. Our framework leverages intermediate features from the baseline detector and employs the tracker to produce reference trajectories, improving computational efficiency and providing richer information for second-stage. In the second stage, we design a global-level inter-object feature aggregation module guided by radar information to align global features across candidate proposals and a local-level inter-grid feature aggregation module that expands local features along the reference trajectories to enhance fine-grained object representation. The aggregated features are then processed by a trajectory-level multi-frame spatiotemporal reasoning module to encode cross-frame interactions and enhance temporal representation. Extensive experiments on the VoD and TJ4DRadSet datasets demonstrate that M^3Detection achieves state-of-the-art 3D detection performance, validating its effectiveness in multi-frame detection with camera-4D imaging radar fusion.
>
---
#### [new 015] Semantic Frame Aggregation-based Transformer for Live Video Comment Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对直播视频评论生成任务，解决现有方法忽视视频帧语义相关性的问题。提出SFAT模型，通过语义帧加权聚合与跨模态注意力机制，提升评论上下文相关性。构建了大规模多模态英文评论数据集，验证模型有效性。**

- **链接: [http://arxiv.org/pdf/2510.26978v1](http://arxiv.org/pdf/2510.26978v1)**

> **作者:** Anam Fatima; Yi Yu; Janak Kapuriya; Julien Lalanne; Jainendra Shukla
>
> **摘要:** Live commenting on video streams has surged in popularity on platforms like Twitch, enhancing viewer engagement through dynamic interactions. However, automatically generating contextually appropriate comments remains a challenging and exciting task. Video streams can contain a vast amount of data and extraneous content. Existing approaches tend to overlook an important aspect of prioritizing video frames that are most relevant to ongoing viewer interactions. This prioritization is crucial for producing contextually appropriate comments. To address this gap, we introduce a novel Semantic Frame Aggregation-based Transformer (SFAT) model for live video comment generation. This method not only leverages CLIP's visual-text multimodal knowledge to generate comments but also assigns weights to video frames based on their semantic relevance to ongoing viewer conversation. It employs an efficient weighted sum of frames technique to emphasize informative frames while focusing less on irrelevant ones. Finally, our comment decoder with a cross-attention mechanism that attends to each modality ensures that the generated comment reflects contextual cues from both chats and video. Furthermore, to address the limitations of existing datasets, which predominantly focus on Chinese-language content with limited video categories, we have constructed a large scale, diverse, multimodal English video comments dataset. Extracted from Twitch, this dataset covers 11 video categories, totaling 438 hours and 3.2 million comments. We demonstrate the effectiveness of our SFAT model by comparing it to existing methods for generating comments from live video and ongoing dialogue contexts.
>
---
#### [new 016] MoME: Mixture of Visual Language Medical Experts for Medical Imaging Segmentation
- **分类: cs.CV**

- **简介: 该论文提出MoME，一种用于医学图像分割的视觉语言混合专家模型。针对医学影像复杂性与多模态信息融合难题，利用多尺度视觉特征与文本嵌入动态选择专家，结合10个数据集3,410例CT扫描，在多个基准上实现优异分割性能，推动了基础模型在医学图像分析中的应用。**

- **链接: [http://arxiv.org/pdf/2510.26996v1](http://arxiv.org/pdf/2510.26996v1)**

> **作者:** Arghavan Rezvani; Xiangyi Yan; Anthony T. Wu; Kun Han; Pooya Khosravi; Xiaohui Xie
>
> **摘要:** In this study, we propose MoME, a Mixture of Visual Language Medical Experts, for Medical Image Segmentation. MoME adapts the successful Mixture of Experts (MoE) paradigm, widely used in Large Language Models (LLMs), for medical vision-language tasks. The architecture enables dynamic expert selection by effectively utilizing multi-scale visual features tailored to the intricacies of medical imagery, enriched with textual embeddings. This work explores a novel integration of vision-language models for this domain. Utilizing an assembly of 10 datasets, encompassing 3,410 CT scans, MoME demonstrates strong performance on a comprehensive medical imaging segmentation benchmark. Our approach explores the integration of foundation models for medical imaging, benefiting from the established efficacy of MoE in boosting model performance by incorporating textual information. Demonstrating competitive precision across multiple datasets, MoME explores a novel architecture for achieving robust results in medical image analysis.
>
---
#### [new 017] C-LEAD: Contrastive Learning for Enhanced Adversarial Defense
- **分类: cs.CV**

- **简介: 该论文针对深度神经网络在图像分类中易受对抗攻击的问题，提出C-LEAD方法，利用对比学习增强模型鲁棒性。通过联合优化模型参数与对抗扰动，使网络学习更具抗干扰性的特征表示，显著提升对多种对抗攻击的防御能力。**

- **链接: [http://arxiv.org/pdf/2510.27249v1](http://arxiv.org/pdf/2510.27249v1)**

> **作者:** Suklav Ghosh; Sonal Kumar; Arijit Sur
>
> **摘要:** Deep neural networks (DNNs) have achieved remarkable success in computer vision tasks such as image classification, segmentation, and object detection. However, they are vulnerable to adversarial attacks, which can cause incorrect predictions with small perturbations in input images. Addressing this issue is crucial for deploying robust deep-learning systems. This paper presents a novel approach that utilizes contrastive learning for adversarial defense, a previously unexplored area. Our method leverages the contrastive loss function to enhance the robustness of classification models by training them with both clean and adversarially perturbed images. By optimizing the model's parameters alongside the perturbations, our approach enables the network to learn robust representations that are less susceptible to adversarial attacks. Experimental results show significant improvements in the model's robustness against various types of adversarial perturbations. This suggests that contrastive loss helps extract more informative and resilient features, contributing to the field of adversarial robustness in deep learning.
>
---
#### [new 018] CASR-Net: An Image Processing-focused Deep Learning-based Coronary Artery Segmentation and Refinement Network for X-ray Coronary Angiogram
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对X射线冠状动脉造影图像中血管分割难题，提出CASR-Net模型。通过三阶段流程：多通道预处理、基于DenseNet121与Self-ONN的分割网络、轮廓精修，提升狭窄血管连续性保持与分割精度，显著改善了低质量图像下的分割效果。**

- **链接: [http://arxiv.org/pdf/2510.27315v1](http://arxiv.org/pdf/2510.27315v1)**

> **作者:** Alvee Hassan; Rusab Sarmun; Muhammad E. H. Chowdhury; M. Murugappan; Md. Sakib Abrar Hossain; Sakib Mahmud; Abdulrahman Alqahtani; Sohaib Bassam Zoghoul; Amith Khandakar; Susu M. Zughaier; Somaya Al-Maadeed; Anwarul Hasan
>
> **摘要:** Early detection of coronary artery disease (CAD) is critical for reducing mortality and improving patient treatment planning. While angiographic image analysis from X-rays is a common and cost-effective method for identifying cardiac abnormalities, including stenotic coronary arteries, poor image quality can significantly impede clinical diagnosis. We present the Coronary Artery Segmentation and Refinement Network (CASR-Net), a three-stage pipeline comprising image preprocessing, segmentation, and refinement. A novel multichannel preprocessing strategy combining CLAHE and an improved Ben Graham method provides incremental gains, increasing Dice Score Coefficient (DSC) by 0.31-0.89% and Intersection over Union (IoU) by 0.40-1.16% compared with using the techniques individually. The core innovation is a segmentation network built on a UNet with a DenseNet121 encoder and a Self-organized Operational Neural Network (Self-ONN) based decoder, which preserves the continuity of narrow and stenotic vessel branches. A final contour refinement module further suppresses false positives. Evaluated with 5-fold cross-validation on a combination of two public datasets that contain both healthy and stenotic arteries, CASR-Net outperformed several state-of-the-art models, achieving an IoU of 61.43%, a DSC of 76.10%, and clDice of 79.36%. These results highlight a robust approach to automated coronary artery segmentation, offering a valuable tool to support clinicians in diagnosis and treatment planning.
>
---
#### [new 019] Overcoming Prompts Pool Confusion via Parameterized Prompt for Incremental Object Detection
- **分类: cs.CV**

- **简介: 该论文针对增量目标检测（IOD）中提示池混淆问题，提出参数化提示方法P²IOD。通过神经网络参数化提示并引入融合策略，实现跨任务知识自适应整合，有效缓解灾难性遗忘，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.27316v1](http://arxiv.org/pdf/2510.27316v1)**

> **作者:** Zijia An; Boyu Diao; Ruiqi Liu; Libo Huang; Chuanguang Yang; Fei Wang; Zhulin An; Yongjun Xu
>
> **摘要:** Recent studies have demonstrated that incorporating trainable prompts into pretrained models enables effective incremental learning. However, the application of prompts in incremental object detection (IOD) remains underexplored. Existing prompts pool based approaches assume disjoint class sets across incremental tasks, which are unsuitable for IOD as they overlook the inherent co-occurrence phenomenon in detection images. In co-occurring scenarios, unlabeled objects from previous tasks may appear in current task images, leading to confusion in prompts pool. In this paper, we hold that prompt structures should exhibit adaptive consolidation properties across tasks, with constrained updates to prevent catastrophic forgetting. Motivated by this, we introduce Parameterized Prompts for Incremental Object Detection (P$^2$IOD). Leveraging neural networks global evolution properties, P$^2$IOD employs networks as the parameterized prompts to adaptively consolidate knowledge across tasks. To constrain prompts structure updates, P$^2$IOD further engages a parameterized prompts fusion strategy. Extensive experiments on PASCAL VOC2007 and MS COCO datasets demonstrate that P$^2$IOD's effectiveness in IOD and achieves the state-of-the-art performance among existing baselines.
>
---
#### [new 020] NegoCollab: A Common Representation Negotiation Approach for Heterogeneous Collaborative Perception
- **分类: cs.CV**

- **简介: 该论文针对异构协同感知中因模型差异导致的特征域差距问题，提出NegoCollab方法。通过训练协商机制生成共享表示，结合结构与语用对齐损失，实现多模态本地表示到公共表示的双向映射，有效降低域差距，提升协同性能。**

- **链接: [http://arxiv.org/pdf/2510.27647v1](http://arxiv.org/pdf/2510.27647v1)**

> **作者:** Congzhang Shao; Quan Yuan; Guiyang Luo; Yue Hu; Danni Wang; Yilin Liu; Rui Pan; Bo Chen; Jinglin Li
>
> **备注:** 19 pages, Accepted by NeurIPS 2025
>
> **摘要:** Collaborative perception improves task performance by expanding the perception range through information sharing among agents. . Immutable heterogeneity poses a significant challenge in collaborative perception, as participating agents may employ different and fixed perception models. This leads to domain gaps in the intermediate features shared among agents, consequently degrading collaborative performance. Aligning the features of all agents to a common representation can eliminate domain gaps with low training cost. However, in existing methods, the common representation is designated as the representation of a specific agent, making it difficult for agents with significant domain discrepancies from this specific agent to achieve proper alignment. This paper proposes NegoCollab, a heterogeneous collaboration method based on the negotiated common representation. It introduces a negotiator during training to derive the common representation from the local representations of each modality's agent, effectively reducing the inherent domain gap with the various local representations. In NegoCollab, the mutual transformation of features between the local representation space and the common representation space is achieved by a pair of sender and receiver. To better align local representations to the common representation containing multimodal information, we introduce structural alignment loss and pragmatic alignment loss in addition to the distribution alignment loss to supervise the training. This enables the knowledge in the common representation to be fully distilled into the sender.
>
---
#### [new 021] A Hybrid Deep Learning and Forensic Approach for Robust Deepfake Detection
- **分类: cs.CV; cs.NE**

- **简介: 该论文针对深度伪造检测任务，解决现有方法在泛化性与可解释性上的不足。提出融合传统取证特征与深度学习模型的混合框架，显著提升检测性能与鲁棒性，并增强结果可解释性，实现更可靠、可信的深度伪造识别。**

- **链接: [http://arxiv.org/pdf/2510.27392v1](http://arxiv.org/pdf/2510.27392v1)**

> **作者:** Sales Aribe Jr
>
> **备注:** 11 pages, 13 figures, 9 tables, Published with International Journal of Advanced Computer Science and Applications (IJACSA)
>
> **摘要:** The rapid evolution of generative adversarial networks (GANs) and diffusion models has made synthetic media increasingly realistic, raising societal concerns around misinformation, identity fraud, and digital trust. Existing deepfake detection methods either rely on deep learning, which suffers from poor generalization and vulnerability to distortions, or forensic analysis, which is interpretable but limited against new manipulation techniques. This study proposes a hybrid framework that fuses forensic features, including noise residuals, JPEG compression traces, and frequency-domain descriptors, with deep learning representations from convolutional neural networks (CNNs) and vision transformers (ViTs). Evaluated on benchmark datasets (FaceForensics++, Celeb-DF v2, DFDC), the proposed model consistently outperformed single-method baselines and demonstrated superior performance compared to existing state-of-the-art hybrid approaches, achieving F1-scores of 0.96, 0.82, and 0.77, respectively. Robustness tests demonstrated stable performance under compression (F1 = 0.87 at QF = 50), adversarial perturbations (AUC = 0.84), and unseen manipulations (F1 = 0.79). Importantly, explainability analysis showed that Grad-CAM and forensic heatmaps overlapped with ground-truth manipulated regions in 82 percent of cases, enhancing transparency and user trust. These findings confirm that hybrid approaches provide a balanced solution, combining the adaptability of deep models with the interpretability of forensic cues, to develop resilient and trustworthy deepfake detection systems.
>
---
#### [new 022] Trans-defense: Transformer-based Denoiser for Adversarial Defense with Spatial-Frequency Domain Representation
- **分类: cs.CV**

- **简介: 该论文针对深度神经网络在安全关键系统中易受对抗攻击的问题，提出基于空间-频域表示的Transformer去噪防御方法。通过结合DWT与Transformer，有效恢复被攻击图像的高频信息，并重训练分类器，显著提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.27245v1](http://arxiv.org/pdf/2510.27245v1)**

> **作者:** Alik Pramanick; Mayank Bansal; Utkarsh Srivastava; Suklav Ghosh; Arijit Sur
>
> **摘要:** In recent times, deep neural networks (DNNs) have been successfully adopted for various applications. Despite their notable achievements, it has become evident that DNNs are vulnerable to sophisticated adversarial attacks, restricting their applications in security-critical systems. In this paper, we present two-phase training methods to tackle the attack: first, training the denoising network, and second, the deep classifier model. We propose a novel denoising strategy that integrates both spatial and frequency domain approaches to defend against adversarial attacks on images. Our analysis reveals that high-frequency components of attacked images are more severely corrupted compared to their lower-frequency counterparts. To address this, we leverage Discrete Wavelet Transform (DWT) for frequency analysis and develop a denoising network that combines spatial image features with wavelets through a transformer layer. Next, we retrain the classifier using the denoised images, which enhances the classifier's robustness against adversarial attacks. Experimental results across the MNIST, CIFAR-10, and Fashion-MNIST datasets reveal that the proposed method remarkably elevates classification accuracy, substantially exceeding the performance by utilizing a denoising network and adversarial training approaches. The code is available at https://github.com/Mayank94/Trans-Defense.
>
---
#### [new 023] ANCHOR: Integrating Adversarial Training with Hard-mined Supervised Contrastive Learning for Robust Representation Learning
- **分类: cs.CV**

- **简介: 该论文提出ANCHOR框架，旨在提升神经网络在对抗攻击下的鲁棒性。针对模型易受微小扰动影响的问题，结合对抗训练与硬样本挖掘的监督对比学习，使同类图像及其扰动版本在嵌入空间中聚集，从而学习更稳定、鲁棒的表示，显著提升干净与对抗场景下的准确率。**

- **链接: [http://arxiv.org/pdf/2510.27599v1](http://arxiv.org/pdf/2510.27599v1)**

> **作者:** Samarup Bhattacharya; Anubhab Bhattacharya; Abir Chakraborty
>
> **备注:** 11 pages, 1 figure
>
> **摘要:** Neural networks have changed the way machines interpret the world. At their core, they learn by following gradients, adjusting their parameters step by step until they identify the most discriminant patterns in the data. This process gives them their strength, yet it also opens the door to a hidden flaw. The very gradients that help a model learn can also be used to produce small, imperceptible tweaks that cause the model to completely alter its decision. Such tweaks are called adversarial attacks. These attacks exploit this vulnerability by adding tiny, imperceptible changes to images that, while leaving them identical to the human eye, cause the model to make wrong predictions. In this work, we propose Adversarially-trained Contrastive Hard-mining for Optimized Robustness (ANCHOR), a framework that leverages the power of supervised contrastive learning with explicit hard positive mining to enable the model to learn representations for images such that the embeddings for the images, their augmentations, and their perturbed versions cluster together in the embedding space along with those for other images of the same class while being separated from images of other classes. This alignment helps the model focus on stable, meaningful patterns rather than fragile gradient cues. On CIFAR-10, our approach achieves impressive results for both clean and robust accuracy under PGD-20 (epsilon = 0.031), outperforming standard adversarial training methods. Our results indicate that combining adversarial guidance with hard-mined contrastive supervision helps models learn more structured and robust representations, narrowing the gap between accuracy and robustness.
>
---
#### [new 024] SYNAPSE-Net: A Unified Framework with Lesion-Aware Hierarchical Gating for Robust Segmentation of Heterogeneous Brain Lesions
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对多模态MRI中异质性脑病变自动分割任务，提出SYNAPSE-Net统一框架，通过分层门控解码器与动态跨模态融合机制，提升模型泛化与鲁棒性。在多个公开数据集上实现领先性能，显著改善分割精度与边界准确率。**

- **链接: [http://arxiv.org/pdf/2510.26961v1](http://arxiv.org/pdf/2510.26961v1)**

> **作者:** Md. Mehedi Hassan; Shafqat Alam; Shahriar Ahmed Seam; Maruf Ahmed
>
> **备注:** 17 pages, 10 figures, 8 tables, submitted to "Medical Image Analysis" journal
>
> **摘要:** Automated segmentation of heterogeneous brain lesions from multi-modal MRI remains a critical challenge in clinical neuroimaging. Current deep learning models are typically specialized `point solutions' that lack generalization and high performance variance, limiting their clinical reliability. To address these gaps, we propose the Unified Multi-Stream SYNAPSE-Net, an adaptive framework designed for both generalization and robustness. The framework is built on a novel hybrid architecture integrating multi-stream CNN encoders, a Swin Transformer bottleneck for global context, a dynamic cross-modal attention fusion (CMAF) mechanism, and a hierarchical gated decoder for high-fidelity mask reconstruction. The architecture is trained with a variance reduction strategy that combines pathology specific data augmentation and difficulty-aware sampling method. The model was evaluated on three different challenging public datasets: the MICCAI 2017 WMH Challenge, the ISLES 2022 Challenge, and the BraTS 2020 Challenge. Our framework attained a state-of-the-art DSC value of 0.831 with the HD95 value of 3.03 in the WMH dataset. For ISLES 2022, it achieved the best boundary accuracy with a statistically significant difference (HD95 value of 9.69). For BraTS 2020, it reached the highest DSC value for the tumor core region (0.8651). These experimental findings suggest that our unified adaptive framework achieves state-of-the-art performance across multiple brain pathologies, providing a robust and clinically feasible solution for automated segmentation. The source code and the pre-trained models are available at https://github.com/mubid-01/SYNAPSE-Net-pre.
>
---
#### [new 025] Context-Gated Cross-Modal Perception with Visual Mamba for PET-CT Lung Tumor Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对肺肿瘤分割任务，解决PET与CT图像融合中信息互补与噪声干扰问题。提出vMambaX框架，基于视觉Mamba架构，设计上下文门控跨模态感知模块，自适应增强关键区域特征，抑制噪声。在PCLT20K数据集上实现高效高精度分割，兼具轻量化与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.27508v1](http://arxiv.org/pdf/2510.27508v1)**

> **作者:** Elena Mulero Ayllón; Linlin Shen; Pierangelo Veltri; Fabrizia Gelardi; Arturo Chiti; Paolo Soda; Matteo Tortora
>
> **摘要:** Accurate lung tumor segmentation is vital for improving diagnosis and treatment planning, and effectively combining anatomical and functional information from PET and CT remains a major challenge. In this study, we propose vMambaX, a lightweight multimodal framework integrating PET and CT scan images through a Context-Gated Cross-Modal Perception Module (CGM). Built on the Visual Mamba architecture, vMambaX adaptively enhances inter-modality feature interaction, emphasizing informative regions while suppressing noise. Evaluated on the PCLT20K dataset, the model outperforms baseline models while maintaining lower computational complexity. These results highlight the effectiveness of adaptive cross-modal gating for multimodal tumor segmentation and demonstrate the potential of vMambaX as an efficient and scalable framework for advanced lung cancer analysis. The code is available at https://github.com/arco-group/vMambaX.
>
---
#### [new 026] RzenEmbed: Towards Comprehensive Multimodal Retrieval
- **分类: cs.CV**

- **简介: 该论文提出RzenEmbed，一个统一的多模态检索框架，旨在解决现有方法仅支持自然图像、难以处理视频和视觉文档的问题。通过两阶段训练与改进的InfoNCE损失，提升模型在多种模态下的判别能力与指令遵循性能，显著优于现有方法，在多模态检索任务上达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2510.27350v1](http://arxiv.org/pdf/2510.27350v1)**

> **作者:** Weijian Jian; Yajun Zhang; Dawei Liang; Chunyu Xie; Yixiao He; Dawei Leng; Yuhui Yin
>
> **摘要:** The rapid advancement of Multimodal Large Language Models (MLLMs) has extended CLIP-based frameworks to produce powerful, universal embeddings for retrieval tasks. However, existing methods primarily focus on natural images, offering limited support for other crucial visual modalities such as videos and visual documents. To bridge this gap, we introduce RzenEmbed, a unified framework to learn embeddings across a diverse set of modalities, including text, images, videos, and visual documents. We employ a novel two-stage training strategy to learn discriminative representations. The first stage focuses on foundational text and multimodal retrieval. In the second stage, we introduce an improved InfoNCE loss, incorporating two key enhancements. Firstly, a hardness-weighted mechanism guides the model to prioritize challenging samples by assigning them higher weights within each batch. Secondly, we implement an approach to mitigate the impact of false negatives and alleviate data noise. This strategy not only enhances the model's discriminative power but also improves its instruction-following capabilities. We further boost performance with learnable temperature parameter and model souping. RzenEmbed sets a new state-of-the-art on the MMEB benchmark. It not only achieves the best overall score but also outperforms all prior work on the challenging video and visual document retrieval tasks. Our models are available in https://huggingface.co/qihoo360/RzenEmbed.
>
---
#### [new 027] Towards Universal Video Retrieval: Generalizing Video Embedding via Synthesized Multimodal Pyramid Curriculum
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文针对视频检索任务，旨在突破现有方法在多任务、多领域泛化能力上的局限。提出统一评估基准UVRB与合成数据生成流程，构建模态金字塔课程训练通用视频嵌入模型GVE，显著提升零样本泛化性能，揭示了传统基准的不足与部分相关检索的重要性。**

- **链接: [http://arxiv.org/pdf/2510.27571v1](http://arxiv.org/pdf/2510.27571v1)**

> **作者:** Zhuoning Guo; Mingxin Li; Yanzhao Zhang; Dingkun Long; Pengjun Xie; Xiaowen Chu
>
> **摘要:** The prevailing video retrieval paradigm is structurally misaligned, as narrow benchmarks incentivize correspondingly limited data and single-task training. Therefore, universal capability is suppressed due to the absence of a diagnostic evaluation that defines and demands multi-dimensional generalization. To break this cycle, we introduce a framework built on the co-design of evaluation, data, and modeling. First, we establish the Universal Video Retrieval Benchmark (UVRB), a suite of 16 datasets designed not only to measure performance but also to diagnose critical capability gaps across tasks and domains. Second, guided by UVRB's diagnostics, we introduce a scalable synthesis workflow that generates 1.55 million high-quality pairs to populate the semantic space required for universality. Finally, we devise the Modality Pyramid, a curriculum that trains our General Video Embedder (GVE) by explicitly leveraging the latent interconnections within our diverse data. Extensive experiments show GVE achieves state-of-the-art zero-shot generalization on UVRB. In particular, our analysis reveals that popular benchmarks are poor predictors of general ability and that partially relevant retrieval is a dominant but overlooked scenario. Overall, our co-designed framework provides a practical path to escape the limited scope and advance toward truly universal video retrieval.
>
---
#### [new 028] Deep Neural Watermarking for Robust Copyright Protection in 3D Point Clouds
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对3D点云版权保护任务，解决其易受几何与非几何攻击导致水印失效的问题。提出基于SVD嵌入二进制水印、利用PointNet++深度学习网络提取水印的鲁棒框架，在多种攻击下显著提升水印恢复精度与完整性。**

- **链接: [http://arxiv.org/pdf/2510.27533v1](http://arxiv.org/pdf/2510.27533v1)**

> **作者:** Khandoker Ashik Uz Zaman; Mohammad Zahangir Alam; Mohammed N. M. Ali; Mahdi H. Miraz
>
> **摘要:** The protection of intellectual property has become critical due to the rapid growth of three-dimensional content in digital media. Unlike traditional images or videos, 3D point clouds present unique challenges for copyright enforcement, as they are especially vulnerable to a range of geometric and non-geometric attacks that can easily degrade or remove conventional watermark signals. In this paper, we address these challenges by proposing a robust deep neural watermarking framework for 3D point cloud copyright protection and ownership verification. Our approach embeds binary watermarks into the singular values of 3D point cloud blocks using spectral decomposition, i.e. Singular Value Decomposition (SVD), and leverages the extraction capabilities of Deep Learning using PointNet++ neural network architecture. The network is trained to reliably extract watermarks even after the data undergoes various attacks such as rotation, scaling, noise, cropping and signal distortions. We validated our method using the publicly available ModelNet40 dataset, demonstrating that deep learning-based extraction significantly outperforms traditional SVD-based techniques under challenging conditions. Our experimental evaluation demonstrates that the deep learning-based extraction approach significantly outperforms existing SVD-based methods with deep learning achieving bitwise accuracy up to 0.83 and Intersection over Union (IoU) of 0.80, compared to SVD achieving a bitwise accuracy of 0.58 and IoU of 0.26 for the Crop (70%) attack, which is the most severe geometric distortion in our experiment. This demonstrates our method's ability to achieve superior watermark recovery and maintain high fidelity even under severe distortions.
>
---
#### [new 029] Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals
- **分类: cs.CV**

- **简介: 该论文针对生成模型蒸馏任务，解决多步蒸馏中效率低、多样性差的问题。提出Phased DMD框架，通过分阶段分布匹配与子区间得分匹配，提升模型容量与生成多样性，实现高效高质的多步蒸馏。**

- **链接: [http://arxiv.org/pdf/2510.27684v1](http://arxiv.org/pdf/2510.27684v1)**

> **作者:** Xiangyu Fan; Zesong Qiu; Zhuguanyu Wu; Fanzhou Wang; Zhiqian Lin; Tianxiang Ren; Dahua Lin; Ruihao Gong; Lei Yang
>
> **摘要:** Distribution Matching Distillation (DMD) distills score-based generative models into efficient one-step generators, without requiring a one-to-one correspondence with the sampling trajectories of their teachers. However, limited model capacity causes one-step distilled models underperform on complex generative tasks, e.g., synthesizing intricate object motions in text-to-video generation. Directly extending DMD to multi-step distillation increases memory usage and computational depth, leading to instability and reduced efficiency. While prior works propose stochastic gradient truncation as a potential solution, we observe that it substantially reduces the generation diversity of multi-step distilled models, bringing it down to the level of their one-step counterparts. To address these limitations, we propose Phased DMD, a multi-step distillation framework that bridges the idea of phase-wise distillation with Mixture-of-Experts (MoE), reducing learning difficulty while enhancing model capacity. Phased DMD is built upon two key ideas: progressive distribution matching and score matching within subintervals. First, our model divides the SNR range into subintervals, progressively refining the model to higher SNR levels, to better capture complex distributions. Next, to ensure the training objective within each subinterval is accurate, we have conducted rigorous mathematical derivations. We validate Phased DMD by distilling state-of-the-art image and video generation models, including Qwen-Image (20B parameters) and Wan2.2 (28B parameters). Experimental results demonstrate that Phased DMD preserves output diversity better than DMD while retaining key generative capabilities. We will release our code and models.
>
---
#### [new 030] PF-DAformer: Proximal Femur Segmentation via Domain Adaptive Transformer for Dual-Center QCT
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文针对多中心定量CT（QCT）中近端股骨分割因设备与人群差异导致的域偏移问题，提出PF-DAformer框架。基于3D TransUNet，融合梯度反转层与最大均值差异，实现特征域自适应，提升模型跨机构泛化能力，确保骨密度分析结果可重复。**

- **链接: [http://arxiv.org/pdf/2510.26903v1](http://arxiv.org/pdf/2510.26903v1)**

> **作者:** Rochak Dhakal; Chen Zhao; Zixin Shi; Joyce H. Keyak; Tadashi S. Kaneko; Kuan-Jui Su; Hui Shen; Hong-Wen Deng; Weihua Zhou
>
> **备注:** 22 Pages, 5 Tables, 10 Figures. The combination of GRL and MMD achieved the most balanced performance, reducing contour deviations and enhancing surface smoothness
>
> **摘要:** Quantitative computed tomography (QCT) plays a crucial role in assessing bone strength and fracture risk by enabling volumetric analysis of bone density distribution in the proximal femur. However, deploying automated segmentation models in practice remains difficult because deep networks trained on one dataset often fail when applied to another. This failure stems from domain shift, where scanners, reconstruction settings, and patient demographics vary across institutions, leading to unstable predictions and unreliable quantitative metrics. Overcoming this barrier is essential for multi-center osteoporosis research and for ensuring that radiomics and structural finite element analysis results remain reproducible across sites. In this work, we developed a domain-adaptive transformer segmentation framework tailored for multi-institutional QCT. Our model is trained and validated on one of the largest hip fracture related research cohorts to date, comprising 1,024 QCT images scans from Tulane University and 384 scans from Rochester, Minnesota for proximal femur segmentation. To address domain shift, we integrate two complementary strategies within a 3D TransUNet backbone: adversarial alignment via Gradient Reversal Layer (GRL), which discourages the network from encoding site-specific cues, and statistical alignment via Maximum Mean Discrepancy (MMD), which explicitly reduces distributional mismatches between institutions. This dual mechanism balances invariance and fine-grained alignment, enabling scanner-agnostic feature learning while preserving anatomical detail.
>
---
#### [new 031] SAGS: Self-Adaptive Alias-Free Gaussian Splatting for Dynamic Surgical Endoscopic Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对动态腹腔镜场景下的可变形组织重建任务，解决现有3D高斯溅射方法因组织运动导致的混叠与伪影问题。提出SAGS框架，通过自适应注意力驱动的4D形变解码器及平滑滤波机制，提升重建质量与可视化效果，在多个指标上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.27318v1](http://arxiv.org/pdf/2510.27318v1)**

> **作者:** Wenfeng Huang; Xiangyun Liao; Yinling Qian; Hao Liu; Yongming Yang; Wenjing Jia; Qiong Wang
>
> **摘要:** Surgical reconstruction of dynamic tissues from endoscopic videos is a crucial technology in robot-assisted surgery. The development of Neural Radiance Fields (NeRFs) has greatly advanced deformable tissue reconstruction, achieving high-quality results from video and image sequences. However, reconstructing deformable endoscopic scenes remains challenging due to aliasing and artifacts caused by tissue movement, which can significantly degrade visualization quality. The introduction of 3D Gaussian Splatting (3DGS) has improved reconstruction efficiency by enabling a faster rendering pipeline. Nevertheless, existing 3DGS methods often prioritize rendering speed while neglecting these critical issues. To address these challenges, we propose SAGS, a self-adaptive alias-free Gaussian splatting framework. We introduce an attention-driven, dynamically weighted 4D deformation decoder, leveraging 3D smoothing filters and 2D Mip filters to mitigate artifacts in deformable tissue reconstruction and better capture the fine details of tissue movement. Experimental results on two public benchmarks, EndoNeRF and SCARED, demonstrate that our method achieves superior performance in all metrics of PSNR, SSIM, and LPIPS compared to the state of the art while also delivering better visualization quality.
>
---
#### [new 032] Sparse Model Inversion: Efficient Inversion of Vision Transformers for Data-Free Applications
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对视觉Transformer模型在数据不可用时的高效反演问题，提出稀疏模型反演方法。通过仅反演语义前景、抑制背景噪声与虚假关联，显著提升反演效率（最高快3.79倍），且保持或增强下游任务性能，无需修改原有损失函数。**

- **链接: [http://arxiv.org/pdf/2510.27186v1](http://arxiv.org/pdf/2510.27186v1)**

> **作者:** Zixuan Hu; Yongxian Wei; Li Shen; Zhenyi Wang; Lei Li; Chun Yuan; Dacheng Tao
>
> **摘要:** Model inversion, which aims to reconstruct the original training data from pre-trained discriminative models, is especially useful when the original training data is unavailable due to privacy, usage rights, or size constraints. However, existing dense inversion methods attempt to reconstruct the entire image area, making them extremely inefficient when inverting high-resolution images from large-scale Vision Transformers (ViTs). We further identify two underlying causes of this inefficiency: the redundant inversion of noisy backgrounds and the unintended inversion of spurious correlations--a phenomenon we term "hallucination" in model inversion. To address these limitations, we propose a novel sparse model inversion strategy, as a plug-and-play extension to speed up existing dense inversion methods with no need for modifying their original loss functions. Specifically, we selectively invert semantic foregrounds while stopping the inversion of noisy backgrounds and potential spurious correlations. Through both theoretical and empirical studies, we validate the efficacy of our approach in achieving significant inversion acceleration (up to 3.79 faster) while maintaining comparable or even enhanced downstream performance in data-free model quantization and data-free knowledge transfer. Code is available at https://github.com/Egg-Hu/SMI.
>
---
#### [new 033] SpecAware: A Spectral-Content Aware Foundation Model for Unifying Multi-Sensor Learning in Hyperspectral Remote Sensing Mapping
- **分类: cs.CV**

- **简介: 该论文针对高光谱遥感映射中多传感器数据异质性导致的模型泛化难题，提出SpecAware模型。通过融合传感器元属性与图像内容，设计双阶段超网络编码机制，实现跨传感器统一预训练，提升多场景下土地覆盖分类、变化检测等任务性能。**

- **链接: [http://arxiv.org/pdf/2510.27219v1](http://arxiv.org/pdf/2510.27219v1)**

> **作者:** Renjie Ji; Xue Wang; Chao Niu; Wen Zhang; Yong Mei; Kun Tan
>
> **摘要:** Hyperspectral imaging (HSI) is a vital tool for fine-grained land-use and land-cover (LULC) mapping. However, the inherent heterogeneity of HSI data has long posed a major barrier to developing generalized models via joint training. Although HSI foundation models have shown promise for different downstream tasks, the existing approaches typically overlook the critical guiding role of sensor meta-attributes, and struggle with multi-sensor training, limiting their transferability. To address these challenges, we propose SpecAware, which is a novel hyperspectral spectral-content aware foundation model for unifying multi-sensor learning for HSI mapping. We also constructed the Hyper-400K dataset to facilitate this research, which is a new large-scale, high-quality benchmark dataset with over 400k image patches from diverse airborne AVIRIS sensors. The core of SpecAware is a two-step hypernetwork-driven encoding process for HSI data. Firstly, we designed a meta-content aware module to generate a unique conditional input for each HSI patch, tailored to each spectral band of every sample by fusing the sensor meta-attributes and its own image content. Secondly, we designed the HyperEmbedding module, where a sample-conditioned hypernetwork dynamically generates a pair of matrix factors for channel-wise encoding, consisting of adaptive spatial pattern extraction and latent semantic feature re-projection. Thus, SpecAware gains the ability to perceive and interpret spatial-spectral features across diverse scenes and sensors. This, in turn, allows SpecAware to adaptively process a variable number of spectral channels, establishing a unified framework for joint pre-training. Extensive experiments on six datasets demonstrate that SpecAware can learn superior feature representations, excelling in land-cover semantic segmentation classification, change detection, and scene classification.
>
---
#### [new 034] T3: Test-Time Model Merging in VLMs for Zero-Shot Medical Imaging Analysis
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对医疗影像中视觉语言模型在分布外场景下性能下降的问题，提出测试时任务自适应融合框架T³。通过动态计算样本级融合系数，平衡通用性与精度，提升零样本医学影像分析性能。**

- **链接: [http://arxiv.org/pdf/2510.27265v1](http://arxiv.org/pdf/2510.27265v1)**

> **作者:** Raza Imam; Hu Wang; Dwarikanath Mahapatra; Mohammad Yaqub
>
> **备注:** Main: 11 pages, Supplementary: 9 pages 10 tables, 10 figures
>
> **摘要:** In medical imaging, vision-language models face a critical duality: pretrained networks offer broad robustness but lack subtle, modality-specific characteristics, while fine-tuned expert models achieve high in-distribution accuracy yet falter under modality shift. Existing model-merging techniques, designed for natural-image benchmarks, are simple and efficient but fail to deliver consistent gains across diverse medical modalities; their static interpolation limits reliability in varied clinical tasks. To address this, we introduce Test-Time Task adaptive merging (T^3), a backpropagation-free framework that computes per-sample interpolation coefficients via the Jensen-Shannon divergence between the two models' output distributions. T^3 dynamically preserves local precision when models agree and defers to generalist robustness under drift. To overcome the inference costs of sample-wise merging, we further propose a batch-wise extension, T^3_B, that computes a merging coefficient across a batch of samples, dramatically reducing computational bottleneck. Recognizing the lack of a standardized medical-merging benchmark, we present a rigorous cross-evaluation protocol spanning in-domain, base-to-novel, and corruptions across four modalities. Empirically, T^3 sets new state-of-the-art in Top-1 accuracy and error reduction, outperforming strong baselines while maintaining efficiency, paving the way for adaptive MVLM deployment in clinical settings. Our code is available at https://github.com/Razaimam45/TCube.
>
---
#### [new 035] ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ZEBRA框架，解决fMRI脑视觉解码中跨被试泛化难题。针对现有方法依赖个体特异性训练的问题，通过对抗性训练分离出与主体无关的语义表征，实现零样本跨被试图像重建，显著提升通用性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.27128v1](http://arxiv.org/pdf/2510.27128v1)**

> **作者:** Haonan Wang; Jingyu Lu; Hongrui Li; Xiaomeng Li
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Recent advances in neural decoding have enabled the reconstruction of visual experiences from brain activity, positioning fMRI-to-image reconstruction as a promising bridge between neuroscience and computer vision. However, current methods predominantly rely on subject-specific models or require subject-specific fine-tuning, limiting their scalability and real-world applicability. In this work, we introduce ZEBRA, the first zero-shot brain visual decoding framework that eliminates the need for subject-specific adaptation. ZEBRA is built on the key insight that fMRI representations can be decomposed into subject-related and semantic-related components. By leveraging adversarial training, our method explicitly disentangles these components to isolate subject-invariant, semantic-specific representations. This disentanglement allows ZEBRA to generalize to unseen subjects without any additional fMRI data or retraining. Extensive experiments show that ZEBRA significantly outperforms zero-shot baselines and achieves performance comparable to fully finetuned models on several metrics. Our work represents a scalable and practical step toward universal neural decoding. Code and model weights are available at: https://github.com/xmed-lab/ZEBRA.
>
---
#### [new 036] Vision Transformer for Robust Occluded Person Reidentification in Complex Surveillance Scenes
- **分类: cs.CV**

- **简介: 该论文针对复杂监控场景下行人重识别（ReID）中的遮挡、视角畸变和图像质量差问题，提出轻量级Sh-ViT模型。通过引入洗牌模块、场景自适应增强与知识蒸馏，提升对遮挡和模糊的鲁棒性。构建MyTT数据集进行真实场景评估，实验表明其在遮挡场景下显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.27677v1](http://arxiv.org/pdf/2510.27677v1)**

> **作者:** Bo Li; Duyuan Zheng; Xinyang Liu; Qingwen Li; Hong Li; Hongyan Cui; Ge Gao; Chen Liu
>
> **备注:** 12 pages,conference
>
> **摘要:** Person re-identification (ReID) in surveillance is challenged by occlusion, viewpoint distortion, and poor image quality. Most existing methods rely on complex modules or perform well only on clear frontal images. We propose Sh-ViT (Shuffling Vision Transformer), a lightweight and robust model for occluded person ReID. Built on ViT-Base, Sh-ViT introduces three components: First, a Shuffle module in the final Transformer layer to break spatial correlations and enhance robustness to occlusion and blur; Second, scenario-adapted augmentation (geometric transforms, erasing, blur, and color adjustment) to simulate surveillance conditions; Third, DeiT-based knowledge distillation to improve learning with limited labels.To support real-world evaluation, we construct the MyTT dataset, containing over 10,000 pedestrians and 30,000+ images from base station inspections, with frequent equipment occlusion and camera variations. Experiments show that Sh-ViT achieves 83.2% Rank-1 and 80.1% mAP on MyTT, outperforming CNN and ViT baselines, and 94.6% Rank-1 and 87.5% mAP on Market1501, surpassing state-of-the-art methods.In summary, Sh-ViT improves robustness to occlusion and blur without external modules, offering a practical solution for surveillance-based personnel monitoring.
>
---
#### [new 037] Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对大视觉语言模型（LVLMs）空间理解能力弱的问题，提出自监督强化学习框架Spatial-SSRL。通过从普通图像中自动构建五种可验证的预训练任务，实现无需人工标注的规模化的空间结构学习，显著提升模型在多场景下的空间推理性能。**

- **链接: [http://arxiv.org/pdf/2510.27606v1](http://arxiv.org/pdf/2510.27606v1)**

> **作者:** Yuhong Liu; Beichen Zhang; Yuhang Zang; Yuhang Cao; Long Xing; Xiaoyi Dong; Haodong Duan; Dahua Lin; Jiaqi Wang
>
> **备注:** preprint
>
> **摘要:** Spatial understanding remains a weakness of Large Vision-Language Models (LVLMs). Existing supervised fine-tuning (SFT) and recent reinforcement learning with verifiable rewards (RLVR) pipelines depend on costly supervision, specialized tools, or constrained environments that limit scale. We introduce Spatial-SSRL, a self-supervised RL paradigm that derives verifiable signals directly from ordinary RGB or RGB-D images. Spatial-SSRL automatically formulates five pretext tasks that capture 2D and 3D spatial structure: shuffled patch reordering, flipped patch recognition, cropped patch inpainting, regional depth ordering, and relative 3D position prediction. These tasks provide ground-truth answers that are easy to verify and require no human or LVLM annotation. Training on our tasks substantially improves spatial reasoning while preserving general visual capabilities. On seven spatial understanding benchmarks in both image and video settings, Spatial-SSRL delivers average accuracy gains of 4.63% (3B) and 3.89% (7B) over the Qwen2.5-VL baselines. Our results show that simple, intrinsic supervision enables RLVR at scale and provides a practical route to stronger spatial intelligence in LVLMs.
>
---
#### [new 038] Scale-Aware Curriculum Learning for Ddata-Efficient Lung Nodule Detection with YOLOv11
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对肺结节检测任务，解决小样本标注下深度学习模型训练效果差的问题。提出尺度自适应课程学习（SACL）策略，通过动态调整训练周期、注入难样本和尺度感知优化，在数据有限时显著提升检测性能，尤其在10%-50%数据下优于基线模型。**

- **链接: [http://arxiv.org/pdf/2510.26923v1](http://arxiv.org/pdf/2510.26923v1)**

> **作者:** Yi Luo; Yike Guo; Hamed Hooshangnejad; Kai Ding
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Lung nodule detection in chest CT is crucial for early lung cancer diagnosis, yet existing deep learning approaches face challenges when deployed in clinical settings with limited annotated data. While curriculum learning has shown promise in improving model training, traditional static curriculum strategies fail in data-scarce scenarios. We propose Scale Adaptive Curriculum Learning (SACL), a novel training strategy that dynamically adjusts curriculum design based on available data scale. SACL introduces three key mechanisms:(1) adaptive epoch scheduling, (2) hard sample injection, and (3) scale-aware optimization. We evaluate SACL on the LUNA25 dataset using YOLOv11 as the base detector. Experimental results demonstrate that while SACL achieves comparable performance to static curriculum learning on the full dataset in mAP50, it shows significant advantages under data-limited conditions with 4.6%, 3.5%, and 2.0% improvements over baseline at 10%, 20%, and 50% of training data respectively. By enabling robust training across varying data scales without architectural modifications, SACL provides a practical solution for healthcare institutions to develop effective lung nodule detection systems despite limited annotation resources.
>
---
#### [new 039] Do Vision-Language Models Measure Up? Benchmarking Visual Measurement Reading with MeasureBench
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型在测量读数任务中的表现不足，提出MeasureBench基准与数据合成管道。通过真实与合成图像评估发现，模型普遍存在指针定位错误问题，导致数值偏差。研究揭示了当前VLMs在细粒度空间定位上的局限性，旨在推动视觉接地数值理解与精确空间感知的发展。**

- **链接: [http://arxiv.org/pdf/2510.26865v1](http://arxiv.org/pdf/2510.26865v1)**

> **作者:** Fenfen Lin; Yesheng Liu; Haiyu Xu; Chen Yue; Zheqi He; Mingxuan Zhao; Miguel Hu Chen; Jiakang Liu; JG Yao; Xi Yang
>
> **备注:** Project page: https://flageval-baai.github.io/MeasureBenchPage/
>
> **摘要:** Reading measurement instruments is effortless for humans and requires relatively little domain expertise, yet it remains surprisingly challenging for current vision-language models (VLMs) as we find in preliminary evaluation. In this work, we introduce MeasureBench, a benchmark on visual measurement reading covering both real-world and synthesized images of various types of measurements, along with an extensible pipeline for data synthesis. Our pipeline procedurally generates a specified type of gauge with controllable visual appearance, enabling scalable variation in key details such as pointers, scales, fonts, lighting, and clutter. Evaluation on popular proprietary and open-weight VLMs shows that even the strongest frontier VLMs struggle measurement reading in general. A consistent failure mode is indicator localization: models can read digits or labels but misidentify the key positions of pointers or alignments, leading to big numeric errors despite plausible textual reasoning. We have also conducted preliminary experiments with reinforcement learning over synthetic data, and find encouraging results on in-domain synthetic subset but less promising for real-world images. Our analysis highlights a fundamental limitation of current VLMs in fine-grained spatial grounding. We hope this resource can help future advances on visually grounded numeracy and precise spatial perception of VLMs, bridging the gap between recognizing numbers and measuring the world.
>
---
#### [new 040] Who Does Your Algorithm Fail? Investigating Age and Ethnic Bias in the MAMA-MIA Dataset
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究乳腺癌肿瘤分割模型在年龄和种族上的公平性问题，属于医学图像分割任务。针对MAMA-MIA数据集，分析算法在不同年龄、种族群体中的分割质量差异，发现对年轻患者存在固有偏差，并揭示多源数据聚合可能掩盖种族相关偏见，强调需在细粒度层面评估数据公平性。**

- **链接: [http://arxiv.org/pdf/2510.27421v1](http://arxiv.org/pdf/2510.27421v1)**

> **作者:** Aditya Parikh; Sneha Das; Aasa Feragen
>
> **备注:** Medical Imaging Meets EurIPS (NeurIPS-endorsed workshop) - MedEurIPS
>
> **摘要:** Deep learning models aim to improve diagnostic workflows, but fairness evaluation remains underexplored beyond classification, e.g., in image segmentation. Unaddressed segmentation bias can lead to disparities in the quality of care for certain populations, potentially compounded across clinical decision points and amplified through iterative model development. Here, we audit the fairness of the automated segmentation labels provided in the breast cancer tumor segmentation dataset MAMA-MIA. We evaluate automated segmentation quality across age, ethnicity, and data source. Our analysis reveals an intrinsic age-related bias against younger patients that continues to persist even after controlling for confounding factors, such as data source. We hypothesize that this bias may be linked to physiological factors, a known challenge for both radiologists and automated systems. Finally, we show how aggregating data from multiple data sources influences site-specific ethnic biases, underscoring the necessity of investigating data at a granular level.
>
---
#### [new 041] Rethinking Robust Adversarial Concept Erasure in Diffusion Models
- **分类: cs.CV; cs.CR**

- **简介: 该论文针对扩散模型中概念擦除任务，解决现有对抗训练方法因忽视语义导致的擦除不彻底与干扰其他概念的问题。提出S-GRACE，通过语义引导生成更精准的对抗样本，显著提升擦除效果26%，更好保留非目标概念，并大幅减少训练时间。**

- **链接: [http://arxiv.org/pdf/2510.27285v1](http://arxiv.org/pdf/2510.27285v1)**

> **作者:** Qinghong Yin; Yu Tian; Yue Zhang
>
> **摘要:** Concept erasure aims to selectively unlearning undesirable content in diffusion models (DMs) to reduce the risk of sensitive content generation. As a novel paradigm in concept erasure, most existing methods employ adversarial training to identify and suppress target concepts, thus reducing the likelihood of sensitive outputs. However, these methods often neglect the specificity of adversarial training in DMs, resulting in only partial mitigation. In this work, we investigate and quantify this specificity from the perspective of concept space, i.e., can adversarial samples truly fit the target concept space? We observe that existing methods neglect the role of conceptual semantics when generating adversarial samples, resulting in ineffective fitting of concept spaces. This oversight leads to the following issues: 1) when there are few adversarial samples, they fail to comprehensively cover the object concept; 2) conversely, they will disrupt other target concept spaces. Motivated by the analysis of these findings, we introduce S-GRACE (Semantics-Guided Robust Adversarial Concept Erasure), which grace leveraging semantic guidance within the concept space to generate adversarial samples and perform erasure training. Experiments conducted with seven state-of-the-art methods and three adversarial prompt generation strategies across various DM unlearning scenarios demonstrate that S-GRACE significantly improves erasure performance 26%, better preserves non-target concepts, and reduces training time by 90%. Our code is available at https://github.com/Qhong-522/S-GRACE.
>
---
#### [new 042] Who Made This? Fake Detection and Source Attribution with Diffusion Features
- **分类: cs.CV**

- **简介: 该论文针对生成式扩散模型带来的假图像检测与来源溯源难题，提出FRIDA框架。利用预训练扩散模型的内部特征，无需微调即可实现跨生成器的假图检测与源生成器识别，有效解决现有方法泛化性差、依赖标注数据的问题。**

- **链接: [http://arxiv.org/pdf/2510.27602v1](http://arxiv.org/pdf/2510.27602v1)**

> **作者:** Simone Bonechi; Paolo Andreini; Barbara Toniella Corradini
>
> **摘要:** The rapid progress of generative diffusion models has enabled the creation of synthetic images that are increasingly difficult to distinguish from real ones, raising concerns about authenticity, copyright, and misinformation. Existing supervised detectors often struggle to generalize across unseen generators, requiring extensive labeled data and frequent retraining. We introduce FRIDA (Fake-image Recognition and source Identification via Diffusion-features Analysis), a lightweight framework that leverages internal activations from a pre-trained diffusion model for deepfake detection and source generator attribution. A k-nearest-neighbor classifier applied to diffusion features achieves state-of-the-art cross-generator performance without fine-tuning, while a compact neural model enables accurate source attribution. These results show that diffusion representations inherently encode generator-specific patterns, providing a simple and interpretable foundation for synthetic image forensics.
>
---
#### [new 043] Image Hashing via Cross-View Code Alignment in the Age of Foundation Models
- **分类: cs.CV; cs.IR; cs.LG**

- **简介: 该论文针对高效大规模图像检索任务，解决高维嵌入下近似最近邻搜索计算成本高的问题。提出CroVCA框架，通过跨视图代码对齐与单损失优化，实现简洁统一的二值化编码。设计轻量级HashCoder网络，支持快速训练与模型适配，在多个基准上仅用5轮训练即达最优性能。**

- **链接: [http://arxiv.org/pdf/2510.27584v1](http://arxiv.org/pdf/2510.27584v1)**

> **作者:** Ilyass Moummad; Kawtar Zaher; Hervé Goëau; Alexis Joly
>
> **摘要:** Efficient large-scale retrieval requires representations that are both compact and discriminative. Foundation models provide powerful visual and multimodal embeddings, but nearest neighbor search in these high-dimensional spaces is computationally expensive. Hashing offers an efficient alternative by enabling fast Hamming distance search with binary codes, yet existing approaches often rely on complex pipelines, multi-term objectives, designs specialized for a single learning paradigm, and long training times. We introduce CroVCA (Cross-View Code Alignment), a simple and unified principle for learning binary codes that remain consistent across semantically aligned views. A single binary cross-entropy loss enforces alignment, while coding-rate maximization serves as an anti-collapse regularizer to promote balanced and diverse codes. To implement this, we design HashCoder, a lightweight MLP hashing network with a final batch normalization layer to enforce balanced codes. HashCoder can be used as a probing head on frozen embeddings or to adapt encoders efficiently via LoRA fine-tuning. Across benchmarks, CroVCA achieves state-of-the-art results in just 5 training epochs. At 16 bits, it particularly well-for instance, unsupervised hashing on COCO completes in under 2 minutes and supervised hashing on ImageNet100 in about 3 minutes on a single GPU. These results highlight CroVCA's efficiency, adaptability, and broad applicability.
>
---
#### [new 044] Mitigating Semantic Collapse in Partially Relevant Video Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对部分相关视频检索（PRVR）中的语义坍塌问题，提出文本关联保持学习与跨分支视频对齐方法，通过保留文本语义关系和分离多时序视频表示，增强查询与视频片段的匹配准确性，有效提升检索性能。**

- **链接: [http://arxiv.org/pdf/2510.27432v1](http://arxiv.org/pdf/2510.27432v1)**

> **作者:** WonJun Moon; MinSeok Jung; Gilhan Park; Tae-Young Kim; Cheol-Ho Cho; Woojin Jun; Jae-Pil Heo
>
> **备注:** Accpeted to NeurIPS 2025. Code is available at https://github.com/admins97/MSC_PRVR
>
> **摘要:** Partially Relevant Video Retrieval (PRVR) seeks videos where only part of the content matches a text query. Existing methods treat every annotated text-video pair as a positive and all others as negatives, ignoring the rich semantic variation both within a single video and across different videos. Consequently, embeddings of both queries and their corresponding video-clip segments for distinct events within the same video collapse together, while embeddings of semantically similar queries and segments from different videos are driven apart. This limits retrieval performance when videos contain multiple, diverse events. This paper addresses the aforementioned problems, termed as semantic collapse, in both the text and video embedding spaces. We first introduce Text Correlation Preservation Learning, which preserves the semantic relationships encoded by the foundation model across text queries. To address collapse in video embeddings, we propose Cross-Branch Video Alignment (CBVA), a contrastive alignment method that disentangles hierarchical video representations across temporal scales. Subsequently, we introduce order-preserving token merging and adaptive CBVA to enhance alignment by producing video segments that are internally coherent yet mutually distinctive. Extensive experiments on PRVR benchmarks demonstrate that our framework effectively prevents semantic collapse and substantially improves retrieval accuracy.
>
---
#### [new 045] Generative Semantic Coding for Ultra-Low Bitrate Visual Communication and Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对超低比特率视觉通信任务，解决在极低带宽下实现高质量图像重建与视觉分析的问题。提出将文本与编码隐变量联合引导的生成模型，融合深度图像压缩，以极低码率传输语义描述与隐码，实现精准场景还原与分析，显著降低带宽需求。**

- **链接: [http://arxiv.org/pdf/2510.27324v1](http://arxiv.org/pdf/2510.27324v1)**

> **作者:** Weiming Chen; Yijia Wang; Zhihan Zhu; Zhihai He
>
> **摘要:** We consider the problem of ultra-low bit rate visual communication for remote vision analysis, human interactions and control in challenging scenarios with very low communication bandwidth, such as deep space exploration, battlefield intelligence, and robot navigation in complex environments. In this paper, we ask the following important question: can we accurately reconstruct the visual scene using only a very small portion of the bit rate in existing coding methods while not sacrificing the accuracy of vision analysis and performance of human interactions? Existing text-to-image generation models offer a new approach for ultra-low bitrate image description. However, they can only achieve a semantic-level approximation of the visual scene, which is far insufficient for the purpose of visual communication and remote vision analysis and human interactions. To address this important issue, we propose to seamlessly integrate image generation with deep image compression, using joint text and coding latent to guide the rectified flow models for precise generation of the visual scene. The semantic text description and coding latent are both encoded and transmitted to the decoder at a very small bit rate. Experimental results demonstrate that our method can achieve the same image reconstruction quality and vision analysis accuracy as existing methods while using much less bandwidth. The code will be released upon paper acceptance.
>
---
#### [new 046] AFM-Net: Advanced Fusing Hierarchical CNN Visual Priors with Global Sequence Modeling for Remote Sensing Image Scene Classification
- **分类: cs.CV**

- **简介: 该论文针对遥感图像场景分类任务，解决复杂空间结构与多尺度特征融合难题。提出AFM-Net，结合CNN提取层次化视觉先验与Mamba高效建模全局序列，通过层级融合机制实现动态特征交互，并引入混合专家分类器，显著提升分类精度与效率。**

- **链接: [http://arxiv.org/pdf/2510.27155v1](http://arxiv.org/pdf/2510.27155v1)**

> **作者:** Yuanhao Tang; Xuechao Zou; Zhengpei Hu; Junliang Xing; Chengkun Zhang; Jianqiang Huang
>
> **摘要:** Remote sensing image scene classification remains a challenging task, primarily due to the complex spatial structures and multi-scale characteristics of ground objects. Existing approaches see CNNs excel at modeling local textures, while Transformers excel at capturing global context. However, efficiently integrating them remains a bottleneck due to the high computational cost of Transformers. To tackle this, we propose AFM-Net, a novel Advanced Hierarchical Fusing framework that achieves effective local and global co-representation through two pathways: a CNN branch for extracting hierarchical visual priors, and a Mamba branch for efficient global sequence modeling. The core innovation of AFM-Net lies in its Hierarchical Fusion Mechanism, which progressively aggregates multi-scale features from both pathways, enabling dynamic cross-level feature interaction and contextual reconstruction to produce highly discriminative representations. These fused features are then adaptively routed through a Mixture-of-Experts classifier module, which dispatches them to the most suitable experts for fine-grained scene recognition. Experiments on AID, NWPU-RESISC45, and UC Merced show that AFM-Net obtains 93.72, 95.54, and 96.92 percent accuracy, surpassing state-of-the-art methods with balanced performance and efficiency. Code is available at https://github.com/tangyuanhao-qhu/AFM-Net.
>
---
#### [new 047] LifWavNet: Lifting Wavelet-based Network for Non-contact ECG Reconstruction from Radar
- **分类: cs.CV**

- **简介: 该论文提出LifWavNet，用于雷达信号到非接触心电图（ECG）的重建任务。针对传统固定小波方法适应性差的问题，引入可学习的小波提升结构，结合多分辨率STFT损失，提升重建精度与生理可解释性，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.27692v1](http://arxiv.org/pdf/2510.27692v1)**

> **作者:** Soumitra Kundu; Gargi Panda; Saumik Bhattacharya; Aurobinda Routray; Rajlakshmi Guha
>
> **摘要:** Non-contact electrocardiogram (ECG) reconstruction from radar signals offers a promising approach for unobtrusive cardiac monitoring. We present LifWavNet, a lifting wavelet network based on a multi-resolution analysis and synthesis (MRAS) model for radar-to-ECG reconstruction. Unlike prior models that use fixed wavelet approaches, LifWavNet employs learnable lifting wavelets with lifting and inverse lifting units to adaptively capture radar signal features and synthesize physiologically meaningful ECG waveforms. To improve reconstruction fidelity, we introduce a multi-resolution short-time Fourier transform (STFT) loss, that enforces consistency with the ground-truth ECG in both temporal and spectral domains. Evaluations on two public datasets demonstrate that LifWavNet outperforms state-of-the-art methods in ECG reconstruction and downstream vital sign estimation (heart rate and heart rate variability). Furthermore, intermediate feature visualization highlights the interpretability of multi-resolution decomposition and synthesis in radar-to-ECG reconstruction. These results establish LifWavNet as a robust framework for radar-based non-contact ECG measurement.
>
---
#### [new 048] E-MMDiT: Revisiting Multimodal Diffusion Transformer Design for Fast Image Synthesis under Limited Resources
- **分类: cs.CV**

- **简介: 该论文提出E-MMDiT，一种轻量级多模态扩散模型，旨在高效图像生成。针对现有模型训练资源消耗大、计算成本高的问题，通过令牌压缩、位置强化与子区域注意力等设计，实现仅304M参数下快速生成512px图像，显著降低资源需求，推动生成AI的普及。**

- **链接: [http://arxiv.org/pdf/2510.27135v1](http://arxiv.org/pdf/2510.27135v1)**

> **作者:** Tong Shen; Jingai Yu; Dong Zhou; Dong Li; Emad Barsoum
>
> **摘要:** Diffusion models have shown strong capabilities in generating high-quality images from text prompts. However, these models often require large-scale training data and significant computational resources to train, or suffer from heavy structure with high latency. To this end, we propose Efficient Multimodal Diffusion Transformer (E-MMDiT), an efficient and lightweight multimodal diffusion model with only 304M parameters for fast image synthesis requiring low training resources. We provide an easily reproducible baseline with competitive results. Our model for 512px generation, trained with only 25M public data in 1.5 days on a single node of 8 AMD MI300X GPUs, achieves 0.66 on GenEval and easily reaches to 0.72 with some post-training techniques such as GRPO. Our design philosophy centers on token reduction as the computational cost scales significantly with the token count. We adopt a highly compressive visual tokenizer to produce a more compact representation and propose a novel multi-path compression module for further compression of tokens. To enhance our design, we introduce Position Reinforcement, which strengthens positional information to maintain spatial coherence, and Alternating Subregion Attention (ASA), which performs attention within subregions to further reduce computational cost. In addition, we propose AdaLN-affine, an efficient lightweight module for computing modulation parameters in transformer blocks. Our code is available at https://github.com/AMD-AGI/Nitro-E and we hope E-MMDiT serves as a strong and practical baseline for future research and contributes to democratization of generative AI models.
>
---
#### [new 049] Gaussian Combined Distance: A Generic Metric for Object Detection
- **分类: cs.CV**

- **简介: 该论文针对小目标检测中IoU和Wasserstein距离的不足，提出高斯联合距离（GCD）。GCD具备尺度不变性与联合优化能力，提升定位精度与收敛速度。实验表明，GCD在多个数据集上均优于现有方法，显著提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.27649v1](http://arxiv.org/pdf/2510.27649v1)**

> **作者:** Ziqian Guan; Xieyi Fu; Pengjun Huang; Hengyuan Zhang; Hubin Du; Yongtao Liu; Yinglin Wang; Qang Ma
>
> **备注:** This paper is accepted by the GRSL in 2025
>
> **摘要:** In object detection, a well-defined similarity metric can significantly enhance model performance. Currently, the IoU-based similarity metric is the most commonly preferred choice for detectors. However, detectors using IoU as a similarity metric often perform poorly when detecting small objects because of their sensitivity to minor positional deviations. To address this issue, recent studies have proposed the Wasserstein Distance as an alternative to IoU for measuring the similarity of Gaussian-distributed bounding boxes. However, we have observed that the Wasserstein Distance lacks scale invariance, which negatively impacts the model's generalization capability. Additionally, when used as a loss function, its independent optimization of the center attributes leads to slow model convergence and unsatisfactory detection precision. To address these challenges, we introduce the Gaussian Combined Distance (GCD). Through analytical examination of GCD and its gradient, we demonstrate that GCD not only possesses scale invariance but also facilitates joint optimization, which enhances model localization performance. Extensive experiments on the AI-TOD-v2 dataset for tiny object detection show that GCD, as a bounding box regression loss function and label assignment metric, achieves state-of-the-art performance across various detectors. We further validated the generalizability of GCD on the MS-COCO-2017 and Visdrone-2019 datasets, where it outperforms the Wasserstein Distance across diverse scales of datasets. Code is available at https://github.com/MArKkwanGuan/mmdet-GCD.
>
---
#### [new 050] A Multi-tiered Human-in-the-loop Approach for Interactive School Mapping Using Earth Observation and Machine Learning
- **分类: cs.CV**

- **简介: 该论文针对发展地区学校数据缺失问题，提出多层级人机协同的交互式学校制图方法。结合遥感与机器学习，通过多分辨率影像分析与人工校验，提升学校位置识别的准确性与完整性，支持教育设施规划与资源分配。**

- **链接: [http://arxiv.org/pdf/2510.27460v1](http://arxiv.org/pdf/2510.27460v1)**

> **作者:** Casper Fibaek; Abi Riley; Kelsey Doerksen; Do-Hyung Kim; Rochelle Schneider
>
> **摘要:** This paper presents a multi-tiered human-in-the-loop framework for interactive school mapping designed to improve the accuracy and completeness of educational facility records, particularly in developing regions where such data may be scarce and infrequently updated. The first tier involves a machine learning based analysis of population density, land cover, and existing infrastructure compared with known school locations. The first tier identifies potential gaps and "mislabelled" schools. In subsequent tiers, medium-resolution satellite imagery (Sentinel-2) is investigated to pinpoint regions with a high likelihood of school presence, followed by the application of very high-resolution (VHR) imagery and deep learning models to generate detailed candidate locations for schools within these prioritised areas. The medium-resolution approach was later removed due to insignificant improvements. The medium and VHR resolution models build upon global pre-trained steps to improve generalisation. A key component of the proposed approach is an interactive interface to allow human operators to iteratively review, validate, and refine the mapping results. Preliminary evaluations indicate that the multi-tiered strategy provides a scalable and cost-effective solution for educational infrastructure mapping to support planning and resource allocation.
>
---
#### [new 051] MapSAM2: Adapting SAM2 for Automatic Segmentation of Historical Map Images and Time Series
- **分类: cs.CV**

- **简介: 该论文提出MapSAM2，针对历史地图图像与时间序列的自动分割任务，解决其风格多样、标注数据稀缺的问题。通过将地图视为视频，利用记忆注意力机制提升几何精度，并构建伪时间序列以降低标注成本，实现少样本下的准确分割与时空关联。**

- **链接: [http://arxiv.org/pdf/2510.27547v1](http://arxiv.org/pdf/2510.27547v1)**

> **作者:** Xue Xia; Randall Balestriero; Tao Zhang; Yixin Zhou; Andrew Ding; Dev Saini; Lorenz Hurni
>
> **摘要:** Historical maps are unique and valuable archives that document geographic features across different time periods. However, automated analysis of historical map images remains a significant challenge due to their wide stylistic variability and the scarcity of annotated training data. Constructing linked spatio-temporal datasets from historical map time series is even more time-consuming and labor-intensive, as it requires synthesizing information from multiple maps. Such datasets are essential for applications such as dating buildings, analyzing the development of road networks and settlements, studying environmental changes etc. We present MapSAM2, a unified framework for automatically segmenting both historical map images and time series. Built on a visual foundation model, MapSAM2 adapts to diverse segmentation tasks with few-shot fine-tuning. Our key innovation is to treat both historical map images and time series as videos. For images, we process a set of tiles as a video, enabling the memory attention mechanism to incorporate contextual cues from similar tiles, leading to improved geometric accuracy, particularly for areal features. For time series, we introduce the annotated Siegfried Building Time Series Dataset and, to reduce annotation costs, propose generating pseudo time series from single-year maps by simulating common temporal transformations. Experimental results show that MapSAM2 learns temporal associations effectively and can accurately segment and link buildings in time series under limited supervision or using pseudo videos. We will release both our dataset and code to support future research.
>
---
#### [new 052] Referee: Reference-aware Audiovisual Deepfake Detection
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对跨数据集与跨语言场景下音频视频深度伪造检测泛化能力差的问题，提出参考感知的多模态检测方法Referee。通过利用单样本说话人特征，联合建模音视频同步性与身份一致性，实现更鲁棒的伪造检测。**

- **链接: [http://arxiv.org/pdf/2510.27475v1](http://arxiv.org/pdf/2510.27475v1)**

> **作者:** Hyemin Boo; Eunsang Lee; Jiyoung Lee
>
> **备注:** In Progress
>
> **摘要:** Since deepfakes generated by advanced generative models have rapidly posed serious threats, existing audiovisual deepfake detection approaches struggle to generalize to unseen forgeries. We propose a novel reference-aware audiovisual deepfake detection method, called Referee. Speaker-specific cues from only one-shot examples are leveraged to detect manipulations beyond spatiotemporal artifacts. By matching and aligning identity-related queries from reference and target content into cross-modal features, Referee jointly reasons about audiovisual synchrony and identity consistency. Extensive experiments on FakeAVCeleb, FaceForensics++, and KoDF demonstrate that Referee achieves state-of-the-art performance on cross-dataset and cross-language evaluation protocols. Experimental results highlight the importance of cross-modal identity verification for future deepfake detection. The code is available at https://github.com/ewha-mmai/referee.
>
---
#### [new 053] FOCUS: Efficient Keyframe Selection for Long Video Understanding
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对长视频理解中视觉令牌预算过高的问题，提出无需训练、模型无关的FOCUS方法。通过将关键帧选择建模为多臂老虎机中的纯探索问题，高效识别信息量大的时间片段，在低于2%帧数下显著提升问答准确率，实现高效长视频理解。**

- **链接: [http://arxiv.org/pdf/2510.27280v1](http://arxiv.org/pdf/2510.27280v1)**

> **作者:** Zirui Zhu; Hailun Xu; Yang Luo; Yong Liu; Kanchan Sarkar; Zhenheng Yang; Yang You
>
> **摘要:** Multimodal large language models (MLLMs) represent images and video frames as visual tokens. Scaling from single images to hour-long videos, however, inflates the token budget far beyond practical limits. Popular pipelines therefore either uniformly subsample or apply keyframe selection with retrieval-style scoring using smaller vision-language models. However, these keyframe selection methods still rely on pre-filtering before selection to reduce the inference cost and can miss the most informative moments. We propose FOCUS, Frame-Optimistic Confidence Upper-bound Selection, a training-free, model-agnostic keyframe selection module that selects query-relevant frames under a strict token budget. FOCUS formulates keyframe selection as a combinatorial pure-exploration (CPE) problem in multi-armed bandits: it treats short temporal clips as arms, and uses empirical means and Bernstein confidence radius to identify informative regions while preserving exploration of uncertain areas. The resulting two-stage exploration-exploitation procedure reduces from a sequential policy with theoretical guarantees, first identifying high-value temporal regions, then selecting top-scoring frames within each region On two long-video question-answering benchmarks, FOCUS delivers substantial accuracy improvements while processing less than 2% of video frames. For videos longer than 20 minutes, it achieves an 11.9% gain in accuracy on LongVideoBench, demonstrating its effectiveness as a keyframe selection method and providing a simple and general solution for scalable long-video understanding with MLLMs.
>
---
#### [new 054] DANCER: Dance ANimation via Condition Enhancement and Rendering with diffusion model
- **分类: cs.CV**

- **简介: 该论文提出DANCER框架，用于单人舞蹈动作的逼真视频生成。针对人体动作高自由度带来的连续性与细节还原难题，引入外观增强与姿态渲染模块，并构建TikTok-3K数据集，显著提升生成质量与连贯性。**

- **链接: [http://arxiv.org/pdf/2510.27169v1](http://arxiv.org/pdf/2510.27169v1)**

> **作者:** Yucheng Xing; Jinxing Yin; Xiaodong Liu
>
> **摘要:** Recently, diffusion models have shown their impressive ability in visual generation tasks. Besides static images, more and more research attentions have been drawn to the generation of realistic videos. The video generation not only has a higher requirement for the quality, but also brings a challenge in ensuring the video continuity. Among all the video generation tasks, human-involved contents, such as human dancing, are even more difficult to generate due to the high degrees of freedom associated with human motions. In this paper, we propose a novel framework, named as DANCER (Dance ANimation via Condition Enhancement and Rendering with Diffusion Model), for realistic single-person dance synthesis based on the most recent stable video diffusion model. As the video generation is generally guided by a reference image and a video sequence, we introduce two important modules into our framework to fully benefit from the two inputs. More specifically, we design an Appearance Enhancement Module (AEM) to focus more on the details of the reference image during the generation, and extend the motion guidance through a Pose Rendering Module (PRM) to capture pose conditions from extra domains. To further improve the generation capability of our model, we also collect a large amount of video data from Internet, and generate a novel datasetTikTok-3K to enhance the model training. The effectiveness of the proposed model has been evaluated through extensive experiments on real-world datasets, where the performance of our model is superior to that of the state-of-the-art methods. All the data and codes will be released upon acceptance.
>
---
#### [new 055] SilhouetteTell: Practical Video Identification Leveraging Blurred Recordings of Video Subtitles
- **分类: cs.CV; cs.CR**

- **简介: 该论文提出SilhouetteTell，一种基于视频字幕轮廓的视频识别攻击。针对视频观看隐私泄露问题，利用手机拍摄的模糊字幕影像，结合空间与时间特征，实现对在线及离线视频的精准识别，验证了其在远距离（最高40米）下的有效性。**

- **链接: [http://arxiv.org/pdf/2510.27179v1](http://arxiv.org/pdf/2510.27179v1)**

> **作者:** Guanchong Huang; Song Fang
>
> **备注:** 16 pages, 29 figures. Accepted at 26th Privacy Enhancing Technologies Symposium (PETS 2026)
>
> **摘要:** Video identification attacks pose a significant privacy threat that can reveal videos that victims watch, which may disclose their hobbies, religious beliefs, political leanings, sexual orientation, and health status. Also, video watching history can be used for user profiling or advertising and may result in cyberbullying, discrimination, or blackmail. Existing extensive video inference techniques usually depend on analyzing network traffic generated by streaming online videos. In this work, we observe that the content of a subtitle determines its silhouette displayed on the screen, and identifying each subtitle silhouette also derives the temporal difference between two consecutive subtitles. We then propose SilhouetteTell, a novel video identification attack that combines the spatial and time domain information into a spatiotemporal feature of subtitle silhouettes. SilhouetteTell explores the spatiotemporal correlation between recorded subtitle silhouettes of a video and its subtitle file. It can infer both online and offline videos. Comprehensive experiments on off-the-shelf smartphones confirm the high efficacy of SilhouetteTell for inferring video titles and clips under various settings, including from a distance of up to 40 meters.
>
---
#### [new 056] HiGS: Hierarchical Generative Scene Framework for Multi-Step Associative Semantic Spatial Composition
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出HiGS框架，解决3D场景生成中复杂度与用户输入量难以平衡的问题。受人类认知启发，采用分步、层次化生成策略，通过动态构建空间-语义图实现结构一致的场景扩展，支持用户精细控制关键区域，提升布局合理性与风格一致性。**

- **链接: [http://arxiv.org/pdf/2510.27148v1](http://arxiv.org/pdf/2510.27148v1)**

> **作者:** Jiacheng Hong; Kunzhen Wu; Mingrui Yu; Yichao Gu; Shengze Xue; Shuangjiu Xiao; Deli Dong
>
> **摘要:** Three-dimensional scene generation holds significant potential in gaming, film, and virtual reality. However, most existing methods adopt a single-step generation process, making it difficult to balance scene complexity with minimal user input. Inspired by the human cognitive process in scene modeling, which progresses from global to local, focuses on key elements, and completes the scene through semantic association, we propose HiGS, a hierarchical generative framework for multi-step associative semantic spatial composition. HiGS enables users to iteratively expand scenes by selecting key semantic objects, offering fine-grained control over regions of interest while the model completes peripheral areas automatically. To support structured and coherent generation, we introduce the Progressive Hierarchical Spatial-Semantic Graph (PHiSSG), which dynamically organizes spatial relationships and semantic dependencies across the evolving scene structure. PHiSSG ensures spatial and geometric consistency throughout the generation process by maintaining a one-to-one mapping between graph nodes and generated objects and supporting recursive layout optimization. Experiments demonstrate that HiGS outperforms single-stage methods in layout plausibility, style consistency, and user preference, offering a controllable and extensible paradigm for efficient 3D scene construction.
>
---
#### [new 057] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DUST框架，用于增强视觉-语言-动作模型（VLAs）的机器人策略学习。针对观测与动作模态差异导致的联合预测难题，设计双流扩散架构，分离处理模态并支持跨模态共享，引入独立噪声与解耦损失，实现双向联合建模。实验表明其在仿真与真实世界任务中均显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.27607v1](http://arxiv.org/pdf/2510.27607v1)**

> **作者:** John Won; Kyungmin Lee; Huiwon Jang; Dongyoung Kim; Jinwoo Shin
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Recently, augmenting Vision-Language-Action models (VLAs) with world modeling has shown promise in improving robotic policy learning. However, it remains challenging to jointly predict next-state observations and action sequences because of the inherent difference between the two modalities. To address this, we propose DUal-STream diffusion (DUST), a world-model augmented VLA framework that handles the modality conflict and enhances the performance of VLAs across diverse tasks. Specifically, we propose a multimodal diffusion transformer architecture that explicitly maintains separate modality streams while still enabling cross-modal knowledge sharing. In addition, we introduce independent noise perturbations for each modality and a decoupled flow-matching loss. This design enables the model to learn the joint distribution in a bidirectional manner while avoiding the need for a unified latent space. Based on the decoupling of modalities during training, we also introduce a joint sampling method that supports test-time scaling, where action and vision tokens evolve asynchronously at different rates. Through experiments on simulated benchmarks such as RoboCasa and GR-1, DUST achieves up to 6% gains over baseline methods, while our test-time scaling approach provides an additional 2-5% boost. On real-world tasks with the Franka Research 3, DUST improves success rates by 13%, confirming its effectiveness beyond simulation. Furthermore, pre-training on action-free videos from BridgeV2 yields significant transfer gains on RoboCasa, underscoring DUST's potential for large-scale VLA pretraining.
>
---
#### [new 058] How Close Are We? Limitations and Progress of AI Models in Banff Lesion Scoring
- **分类: cs.CV**

- **简介: 该论文研究AI在肾移植活检病理分析中对Banff评分的模拟能力。针对半定量评分标准复杂、人工差异大等问题，提出模块化规则框架，分解病变成分并结合现有分割检测模型计算分数。结果揭示当前AI在结构识别与可解释性上的局限，强调需建立标准化评估体系以推动发展。**

- **链接: [http://arxiv.org/pdf/2510.27158v1](http://arxiv.org/pdf/2510.27158v1)**

> **作者:** Yanfan Zhu; Juming Xiong; Ruining Deng; Yu Wang; Yaohong Wang; Shilin Zhao; Mengmeng Yin; Yuqing Liu; Haichun Yang; Yuankai Huo
>
> **摘要:** The Banff Classification provides the global standard for evaluating renal transplant biopsies, yet its semi-quantitative nature, complex criteria, and inter-observer variability present significant challenges for computational replication. In this study, we explore the feasibility of approximating Banff lesion scores using existing deep learning models through a modular, rule-based framework. We decompose each Banff indicator - such as glomerulitis (g), peritubular capillaritis (ptc), and intimal arteritis (v) - into its constituent structural and inflammatory components, and assess whether current segmentation and detection tools can support their computation. Model outputs are mapped to Banff scores using heuristic rules aligned with expert guidelines, and evaluated against expert-annotated ground truths. Our findings highlight both partial successes and critical failure modes, including structural omission, hallucination, and detection ambiguity. Even when final scores match expert annotations, inconsistencies in intermediate representations often undermine interpretability. These results reveal the limitations of current AI pipelines in replicating computational expert-level grading, and emphasize the importance of modular evaluation and computational Banff grading standard in guiding future model development for transplant pathology.
>
---
#### [new 059] MoRE: 3D Visual Geometry Reconstruction Meets Mixture-of-Experts
- **分类: cs.CV**

- **简介: 该论文提出MoRE，一种基于混合专家（MoE）架构的3D视觉几何重建模型。针对3D模型扩展难、数据多样性强的问题，通过动态路由实现专家专业化，结合置信度深度优化与语义-几何融合，提升模型可扩展性与鲁棒性，在多个基准上达到领先性能。**

- **链接: [http://arxiv.org/pdf/2510.27234v1](http://arxiv.org/pdf/2510.27234v1)**

> **作者:** Jingnan Gao; Zhe Wang; Xianze Fang; Xingyu Ren; Zhuo Chen; Shengqi Liu; Yuhao Cheng; Jiangjing Lyu; Xiaokang Yang; Yichao Yan
>
> **备注:** Project Page: https://g-1nonly.github.io/MoRE_Website/, Code: https://github.com/alibaba/Taobao3D
>
> **摘要:** Recent advances in language and vision have demonstrated that scaling up model capacity consistently improves performance across diverse tasks. In 3D visual geometry reconstruction, large-scale training has likewise proven effective for learning versatile representations. However, further scaling of 3D models is challenging due to the complexity of geometric supervision and the diversity of 3D data. To overcome these limitations, we propose MoRE, a dense 3D visual foundation model based on a Mixture-of-Experts (MoE) architecture that dynamically routes features to task-specific experts, allowing them to specialize in complementary data aspects and enhance both scalability and adaptability. Aiming to improve robustness under real-world conditions, MoRE incorporates a confidence-based depth refinement module that stabilizes and refines geometric estimation. In addition, it integrates dense semantic features with globally aligned 3D backbone representations for high-fidelity surface normal prediction. MoRE is further optimized with tailored loss functions to ensure robust learning across diverse inputs and multiple geometric tasks. Extensive experiments demonstrate that MoRE achieves state-of-the-art performance across multiple benchmarks and supports effective downstream applications without extra computation.
>
---
#### [new 060] Enhancing Spatio-Temporal Zero-shot Action Recognition with Language-driven Description Attributes
- **分类: cs.CV**

- **简介: 该论文聚焦于零样本动作识别任务，针对仅依赖动作类别导致语义模糊的问题，提出利用网络爬取描述文本与大语言模型提取关键词，构建描述属性，并设计时空交互模块对齐属性与视频内容，显著提升了在UCF-101、HMDB-51和Kinetics-600上的识别准确率。**

- **链接: [http://arxiv.org/pdf/2510.27255v1](http://arxiv.org/pdf/2510.27255v1)**

> **作者:** Yehna Kim andYoung-Eun Kim; Seong-Whan Lee
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated impressive capabilities in zero-shot action recognition by learning to associate video embeddings with class embeddings. However, a significant challenge arises when relying solely on action classes to provide semantic context, particularly due to the presence of multi-semantic words, which can introduce ambiguity in understanding the intended concepts of actions. To address this issue, we propose an innovative approach that harnesses web-crawled descriptions, leveraging a large-language model to extract relevant keywords. This method reduces the need for human annotators and eliminates the laborious manual process of attribute data creation. Additionally, we introduce a spatio-temporal interaction module designed to focus on objects and action units, facilitating alignment between description attributes and video content. In our zero-shot experiments, our model achieves impressive results, attaining accuracies of 81.0%, 53.1%, and 68.9% on UCF-101, HMDB-51, and Kinetics-600, respectively, underscoring the model's adaptability and effectiveness across various downstream tasks.
>
---
#### [new 061] Dual-level Progressive Hardness-Aware Reweighting for Cross-View Geo-Localization
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对无人机与卫星图像间的跨视图地理定位任务，解决视角差异大、难负样本多导致的训练不稳问题。提出双层渐进式硬度感知重加权方法（DPHR），在样本级通过相对难度评估精细加权，在批量级利用训练进度信号动态调节损失权重，提升模型鲁棒性与定位精度。**

- **链接: [http://arxiv.org/pdf/2510.27181v1](http://arxiv.org/pdf/2510.27181v1)**

> **作者:** Guozheng Zheng; Jian Guan; Mingjie Xie; Xuanjia Zhao; Congyi Fan; Shiheng Zhang; Pengming Feng
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Cross-view geo-localization (CVGL) between drone and satellite imagery remains challenging due to severe viewpoint gaps and the presence of hard negatives, which are visually similar but geographically mismatched samples. Existing mining or reweighting strategies often use static weighting, which is sensitive to distribution shifts and prone to overemphasizing difficult samples too early, leading to noisy gradients and unstable convergence. In this paper, we present a Dual-level Progressive Hardness-aware Reweighting (DPHR) strategy. At the sample level, a Ratio-based Difficulty-Aware (RDA) module evaluates relative difficulty and assigns fine-grained weights to negatives. At the batch level, a Progressive Adaptive Loss Weighting (PALW) mechanism exploits a training-progress signal to attenuate noisy gradients during early optimization and progressively enhance hard-negative mining as training matures. Experiments on the University-1652 and SUES-200 benchmarks demonstrate the effectiveness and robustness of the proposed DPHR, achieving consistent improvements over state-of-the-art methods.
>
---
#### [new 062] H2-Cache: A Novel Hierarchical Dual-Stage Cache for High-Performance Acceleration of Generative Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对生成式扩散模型推理速度慢的问题，提出H2-Cache，一种分阶段的层次化双阈值缓存机制。通过分离结构定义与细节优化阶段，结合轻量级特征摘要技术，实现高速且高保真的推理，显著提升效率（最高5.08倍）而不损失图像质量。**

- **链接: [http://arxiv.org/pdf/2510.27171v1](http://arxiv.org/pdf/2510.27171v1)**

> **作者:** Mingyu Sung; Il-Min Kim; Sangseok Yun; Jae-Mo Kang
>
> **摘要:** Diffusion models have emerged as state-of-the-art in image generation, but their practical deployment is hindered by the significant computational cost of their iterative denoising process. While existing caching techniques can accelerate inference, they often create a challenging trade-off between speed and fidelity, suffering from quality degradation and high computational overhead. To address these limitations, we introduce H2-Cache, a novel hierarchical caching mechanism designed for modern generative diffusion model architectures. Our method is founded on the key insight that the denoising process can be functionally separated into a structure-defining stage and a detail-refining stage. H2-cache leverages this by employing a dual-threshold system, using independent thresholds to selectively cache each stage. To ensure the efficiency of our dual-check approach, we introduce pooled feature summarization (PFS), a lightweight technique for robust and fast similarity estimation. Extensive experiments on the Flux architecture demonstrate that H2-cache achieves significant acceleration (up to 5.08x) while maintaining image quality nearly identical to the baseline, quantitatively and qualitatively outperforming existing caching methods. Our work presents a robust and practical solution that effectively resolves the speed-quality dilemma, significantly lowering the barrier for the real-world application of high-fidelity diffusion models. Source code is available at https://github.com/Bluear7878/H2-cache-A-Hierarchical-Dual-Stage-Cache.
>
---
#### [new 063] Hierarchical Transformers for Unsupervised 3D Shape Abstraction
- **分类: cs.CV**

- **简介: 该论文提出HiT模型，用于无监督3D形状抽象。针对现有方法固定层级结构的局限，提出层次化Transformer，通过压缩码本自动学习跨类别形状的通用层次关系，实现从粗到细的灵活层次建模，并在ShapeNet上成功完成多粒度无监督分割。**

- **链接: [http://arxiv.org/pdf/2510.27088v1](http://arxiv.org/pdf/2510.27088v1)**

> **作者:** Aditya Vora; Lily Goli; Andrea Tagliasacchi; Hao Zhang
>
> **摘要:** We introduce HiT, a novel hierarchical neural field representation for 3D shapes that learns general hierarchies in a coarse-to-fine manner across different shape categories in an unsupervised setting. Our key contribution is a hierarchical transformer (HiT), where each level learns parent-child relationships of the tree hierarchy using a compressed codebook. This codebook enables the network to automatically identify common substructures across potentially diverse shape categories. Unlike previous works that constrain the task to a fixed hierarchical structure (e.g., binary), we impose no such restriction, except for limiting the total number of nodes at each tree level. This flexibility allows our method to infer the hierarchical structure directly from data, over multiple shape categories, and representing more general and complex hierarchies than prior approaches. When trained at scale with a reconstruction loss, our model captures meaningful containment relationships between parent and child nodes. We demonstrate its effectiveness through an unsupervised shape segmentation task over all 55 ShapeNet categories, where our method successfully segments shapes into multiple levels of granularity.
>
---
#### [new 064] Multi-Modal Feature Fusion for Spatial Morphology Analysis of Traditional Villages via Hierarchical Graph Neural Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对传统村落空间形态分析中数据单一、定性为主的问题，提出基于分层图神经网络的多模态特征融合方法。通过构建含输入与通信节点的双类型图结构，结合GCN与GAT实现多源数据融合，并引入关系池化与联合训练策略，显著提升17类村落形态分类性能，推动了村落空间模式研究的定量发展。**

- **链接: [http://arxiv.org/pdf/2510.27208v1](http://arxiv.org/pdf/2510.27208v1)**

> **作者:** Jiaxin Zhang; Zehong Zhu; Junye Deng; Yunqin Li; and Bowen Wang
>
> **摘要:** Villages areas hold significant importance in the study of human-land relationships. However, with the advancement of urbanization, the gradual disappearance of spatial characteristics and the homogenization of landscapes have emerged as prominent issues. Existing studies primarily adopt a single-disciplinary perspective to analyze villages spatial morphology and its influencing factors, relying heavily on qualitative analysis methods. These efforts are often constrained by the lack of digital infrastructure and insufficient data. To address the current research limitations, this paper proposes a Hierarchical Graph Neural Network (HGNN) model that integrates multi-source data to conduct an in-depth analysis of villages spatial morphology. The framework includes two types of nodes-input nodes and communication nodes-and two types of edges-static input edges and dynamic communication edges. By combining Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), the proposed model efficiently integrates multimodal features under a two-stage feature update mechanism. Additionally, based on existing principles for classifying villages spatial morphology, the paper introduces a relational pooling mechanism and implements a joint training strategy across 17 subtypes. Experimental results demonstrate that this method achieves significant performance improvements over existing approaches in multimodal fusion and classification tasks. Additionally, the proposed joint optimization of all sub-types lifts mean accuracy/F1 from 0.71/0.83 (independent models) to 0.82/0.90, driven by a 6% gain for parcel tasks. Our method provides scientific evidence for exploring villages spatial patterns and generative logic.
>
---
#### [new 065] Deep learning denoising unlocks quantitative insights in operando materials microscopy
- **分类: cs.CV; cond-mat.mtrl-sci**

- **简介: 该论文针对操作显微镜中噪声限制定量分析的问题，提出一种通用的无监督深度学习去噪框架。通过融合物理约束的偏微分方程优化，提升多模态、多尺度显微图像的信噪比，实现纳米级化学/结构异质性解析与自动分割，显著降低噪声干扰，拓展了受限于噪声的技术应用。**

- **链接: [http://arxiv.org/pdf/2510.27667v1](http://arxiv.org/pdf/2510.27667v1)**

> **作者:** Samuel Degnan-Morgenstern; Alexander E. Cohen; Rajeev Gopal; Megan Gober; George J. Nelson; Peng Bai; Martin Z. Bazant
>
> **摘要:** Operando microscopy provides direct insight into the dynamic chemical and physical processes that govern functional materials, yet measurement noise limits the effective resolution and undermines quantitative analysis. Here, we present a general framework for integrating unsupervised deep learning-based denoising into quantitative microscopy workflows across modalities and length scales. Using simulated data, we demonstrate that deep denoising preserves physical fidelity, introduces minimal bias, and reduces uncertainty in model learning with partial differential equation (PDE)-constrained optimization. Applied to experiments, denoising reveals nanoscale chemical and structural heterogeneity in scanning transmission X-ray microscopy (STXM) of lithium iron phosphate (LFP), enables automated particle segmentation and phase classification in optical microscopy of graphite electrodes, and reduces noise-induced variability by nearly 80% in neutron radiography to resolve heterogeneous lithium transport. Collectively, these results establish deep denoising as a powerful, modality-agnostic enhancement that advances quantitative operando imaging and extends the reach of previously noise-limited techniques.
>
---
#### [new 066] Fusion of Heterogeneous Pathology Foundation Models for Whole Slide Image Analysis
- **分类: cs.CV**

- **简介: 该论文针对病理全切片图像（WSI）分析中异构基础模型（FM）特征不一致的问题，提出FuseCPath框架。通过多视角聚类筛选关键样本，设计簇级重嵌入与协同蒸馏策略，融合不同FM的局部与整体特征，显著提升下游任务性能。**

- **链接: [http://arxiv.org/pdf/2510.27237v1](http://arxiv.org/pdf/2510.27237v1)**

> **作者:** Zhidong Yang; Xiuhui Shi; Wei Ba; Zhigang Song; Haijing Luan; Taiyuan Hu; Senlin Lin; Jiguang Wang; Shaohua Kevin Zhou; Rui Yan
>
> **备注:** 22 pages, 9 figures
>
> **摘要:** Whole slide image (WSI) analysis has emerged as an increasingly essential technique in computational pathology. Recent advances in the pathological foundation models (FMs) have demonstrated significant advantages in deriving meaningful patch-level or slide-level feature representations from WSIs. However, current pathological FMs have exhibited substantial heterogeneity caused by diverse private training datasets and different network architectures. This heterogeneity introduces performance variability when we utilize the extracted features from different FMs in the downstream tasks. To fully explore the advantage of multiple FMs effectively, in this work, we propose a novel framework for the fusion of heterogeneous pathological FMs, called FuseCPath, yielding a model with a superior ensemble performance. The main contributions of our framework can be summarized as follows: (i) To guarantee the representativeness of the training patches, we propose a multi-view clustering-based method to filter out the discriminative patches via multiple FMs' embeddings. (ii) To effectively fuse the heterogeneous patch-level FMs, we devise a cluster-level re-embedding strategy to online capture patch-level local features. (iii) To effectively fuse the heterogeneous slide-level FMs, we devise a collaborative distillation strategy to explore the connections between slide-level FMs. Extensive experiments conducted on lung cancer, bladder cancer, and colorectal cancer datasets from The Cancer Genome Atlas (TCGA) have demonstrated that the proposed FuseCPath achieves state-of-the-art performance across multiple tasks on these public datasets.
>
---
#### [new 067] Versatile and Efficient Medical Image Super-Resolution Via Frequency-Gated Mamba
- **分类: cs.CV**

- **简介: 该论文针对医学图像超分辨率任务，旨在提升图像细节与全局结构建模效率。提出FGMamba模型，通过频域感知的门控状态空间模块与金字塔频率融合机制，实现轻量化下高精度重建，在多模态医学影像上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.27296v1](http://arxiv.org/pdf/2510.27296v1)**

> **作者:** Wenfeng Huang; Xiangyun Liao; Wei Cao; Wenjing Jia; Weixin Si
>
> **摘要:** Medical image super-resolution (SR) is essential for enhancing diagnostic accuracy while reducing acquisition cost and scanning time. However, modeling both long-range anatomical structures and fine-grained frequency details with low computational overhead remains challenging. We propose FGMamba, a novel frequency-aware gated state-space model that unifies global dependency modeling and fine-detail enhancement into a lightweight architecture. Our method introduces two key innovations: a Gated Attention-enhanced State-Space Module (GASM) that integrates efficient state-space modeling with dual-branch spatial and channel attention, and a Pyramid Frequency Fusion Module (PFFM) that captures high-frequency details across multiple resolutions via FFT-guided fusion. Extensive evaluations across five medical imaging modalities (Ultrasound, OCT, MRI, CT, and Endoscopic) demonstrate that FGMamba achieves superior PSNR/SSIM while maintaining a compact parameter footprint ($<$0.75M), outperforming CNN-based and Transformer-based SOTAs. Our results validate the effectiveness of frequency-aware state-space modeling for scalable and accurate medical image enhancement.
>
---
#### [new 068] From Pixels to Paths: A Multi-Agent Framework for Editable Scientific Illustration
- **分类: cs.CV**

- **简介: 该论文提出VisPainter框架，解决科学插图生成中信息密度高但缺乏可编辑性的问题。通过多智能体协作实现向量级元素控制，支持高效迭代修改。构建了七维评估基准VisBench，系统验证了架构与模型性能，实现了高质量、可编辑的科学插图生成。**

- **链接: [http://arxiv.org/pdf/2510.27452v1](http://arxiv.org/pdf/2510.27452v1)**

> **作者:** Jianwen Sun; Fanrui Zhang; Yukang Feng; Chuanhao Li; Zizhen Li; Jiaxin Ai; Yifan Chang; Yu Dai; Kaipeng Zhang
>
> **摘要:** Scientific illustrations demand both high information density and post-editability. However, current generative models have two major limitations: Frist, image generation models output rasterized images lacking semantic structure, making it impossible to access, edit, or rearrange independent visual components in the images. Second, code-based generation methods (TikZ or SVG), although providing element-level control, force users into the cumbersome cycle of "writing-compiling-reviewing" and lack the intuitiveness of manipulation. Neither of these two approaches can well meet the needs for efficiency, intuitiveness, and iterative modification in scientific creation. To bridge this gap, we introduce VisPainter, a multi-agent framework for scientific illustration built upon the model context protocol. VisPainter orchestrates three specialized modules-a Manager, a Designer, and a Toolbox-to collaboratively produce diagrams compatible with standard vector graphics software. This modular, role-based design allows each element to be explicitly represented and manipulated, enabling true element-level control and any element can be added and modified later. To systematically evaluate the quality of scientific illustrations, we introduce VisBench, a benchmark with seven-dimensional evaluation metrics. It assesses high-information-density scientific illustrations from four aspects: content, layout, visual perception, and interaction cost. To this end, we conducted extensive ablation experiments to verify the rationality of our architecture and the reliability of our evaluation methods. Finally, we evaluated various vision-language models, presenting fair and credible model rankings along with detailed comparisons of their respective capabilities. Additionally, we isolated and quantified the impacts of role division, step control,and description on the quality of illustrations.
>
---
#### [new 069] Fine-Tuning Open Video Generators for Cinematic Scene Synthesis: A Small-Data Pipeline with LoRA and Wan2.1 I2V
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对影视场景生成任务，提出一种小数据下的微调流水线。利用LoRA与Wan2.1 I2V模型，在单卡上高效迁移视觉风格，并通过关键帧生成与序列扩展实现高质量视频合成，显著提升影像一致性与时间稳定性。**

- **链接: [http://arxiv.org/pdf/2510.27364v1](http://arxiv.org/pdf/2510.27364v1)**

> **作者:** Meftun Akarsu; Kerem Catay; Sedat Bin Vedat; Enes Kutay Yarkan; Ilke Senturk; Arda Sar; Dafne Eksioglu
>
> **备注:** video generation, image-to-video, dif- fusion transformer, LoRA, fine-tuning, cinematic scene synthesis, multi-GPU inference, fully sharded data parallelism, computational efficiency
>
> **摘要:** We present a practical pipeline for fine-tuning open-source video diffusion transformers to synthesize cinematic scenes for television and film production from small datasets. The proposed two-stage process decouples visual style learning from motion generation. In the first stage, Low-Rank Adaptation (LoRA) modules are integrated into the cross-attention layers of the Wan2.1 I2V-14B model to adapt its visual representations using a compact dataset of short clips from Ay Yapim's historical television film El Turco. This enables efficient domain transfer within hours on a single GPU. In the second stage, the fine-tuned model produces stylistically consistent keyframes that preserve costume, lighting, and color grading, which are then temporally expanded into coherent 720p sequences through the model's video decoder. We further apply lightweight parallelization and sequence partitioning strategies to accelerate inference without quality degradation. Quantitative and qualitative evaluations using FVD, CLIP-SIM, and LPIPS metrics, supported by a small expert user study, demonstrate measurable improvements in cinematic fidelity and temporal stability over the base model. The complete training and inference pipeline is released to support reproducibility and adaptation across cinematic domains.
>
---
#### [new 070] MeisenMeister: A Simple Two Stage Pipeline for Breast Cancer Classification on MRI
- **分类: cs.CV**

- **简介: 该论文针对乳腺癌MRI早期检测任务，解决标注数据稀缺下的分类难题。提出简单两阶段流程MeisenMeister，通过迭代优化提升分类性能与临床适用性，实现高效准确的乳腺癌筛查。**

- **链接: [http://arxiv.org/pdf/2510.27326v1](http://arxiv.org/pdf/2510.27326v1)**

> **作者:** Benjamin Hamm; Yannick Kirchhoff; Maximilian Rokuss; Klaus Maier-Hein
>
> **备注:** Winning Solution of the MICCAI 2025 ODELIA Breast MRI Classification Challenge
>
> **摘要:** The ODELIA Breast MRI Challenge 2025 addresses a critical issue in breast cancer screening: improving early detection through more efficient and accurate interpretation of breast MRI scans. Even though methods for general-purpose whole-body lesion segmentation as well as multi-time-point analysis exist, breast cancer detection remains highly challenging, largely due to the limited availability of high-quality segmentation labels. Therefore, developing robust classification-based approaches is crucial for the future of early breast cancer detection, particularly in applications such as large-scale screening. In this write-up, we provide a comprehensive overview of our approach to the challenge. We begin by detailing the underlying concept and foundational assumptions that guided our work. We then describe the iterative development process, highlighting the key stages of experimentation, evaluation, and refinement that shaped the evolution of our solution. Finally, we present the reasoning and evidence that informed the design choices behind our final submission, with a focus on performance, robustness, and clinical relevance. We release our full implementation publicly at https://github.com/MIC-DKFZ/MeisenMeister
>
---
#### [new 071] DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文针对3D高斯点云渲染中的密度控制问题，提出DC4GS方法。通过引入方向一致性（DC）机制，优化了传统自适应密度控制的分裂策略，在减少30%以上点数的同时提升重建精度，有效捕捉局部结构复杂性。**

- **链接: [http://arxiv.org/pdf/2510.26921v1](http://arxiv.org/pdf/2510.26921v1)**

> **作者:** Moonsoo Jeong; Dongbeen Kim; Minseong Kim; Sungkil Lee
>
> **备注:** Accepted to NeurIPS 2025 / Project page: https://github.com/cgskku/dc4gs
>
> **摘要:** We present a Directional Consistency (DC)-driven Adaptive Density Control (ADC) for 3D Gaussian Splatting (DC4GS). Whereas the conventional ADC bases its primitive splitting on the magnitudes of positional gradients, we further incorporate the DC of the gradients into ADC, and realize it through the angular coherence of the gradients. Our DC better captures local structural complexities in ADC, avoiding redundant splitting. When splitting is required, we again utilize the DC to define optimal split positions so that sub-primitives best align with the local structures than the conventional random placement. As a consequence, our DC4GS greatly reduces the number of primitives (up to 30% in our experiments) than the existing ADC, and also enhances reconstruction fidelity greatly.
>
---
#### [new 072] Privacy-Aware Continual Self-Supervised Learning on Multi-Window Chest Computed Tomography for Domain-Shift Robustness
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对医疗影像中因窗宽设置差异导致的领域偏移问题，提出一种隐私感知的持续自监督学习框架。通过隐式回放机制与基于Wasserstein距离的特征蒸馏，实现跨阶段知识保留与隐私保护，在多窗宽胸部CT图像上提升了模型的鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.27213v1](http://arxiv.org/pdf/2510.27213v1)**

> **作者:** Ren Tasai; Guang Li; Ren Togo; Takahiro Ogawa; Kenji Hirata; Minghui Tang; Takaaki Yoshimura; Hiroyuki Sugimori; Noriko Nishioka; Yukie Shimizu; Kohsuke Kudo; Miki Haseyama
>
> **摘要:** We propose a novel continual self-supervised learning (CSSL) framework for simultaneously learning diverse features from multi-window-obtained chest computed tomography (CT) images and ensuring data privacy. Achieving a robust and highly generalizable model in medical image diagnosis is challenging, mainly because of issues, such as the scarcity of large-scale, accurately annotated datasets and domain shifts inherent to dynamic healthcare environments. Specifically, in chest CT, these domain shifts often arise from differences in window settings, which are optimized for distinct clinical purposes. Previous CSSL frameworks often mitigated domain shift by reusing past data, a typically impractical approach owing to privacy constraints. Our approach addresses these challenges by effectively capturing the relationship between previously learned knowledge and new information across different training stages through continual pretraining on unlabeled images. Specifically, by incorporating a latent replay-based mechanism into CSSL, our method mitigates catastrophic forgetting due to domain shifts during continual pretraining while ensuring data privacy. Additionally, we introduce a feature distillation technique that integrates Wasserstein distance-based knowledge distillation (WKD) and batch-knowledge ensemble (BKE), enhancing the ability of the model to learn meaningful, domain-shift-robust representations. Finally, we validate our approach using chest CT images obtained across two different window settings, demonstrating superior performance compared with other approaches.
>
---
#### [new 073] DeblurSDI: Blind Image Deblurring Using Self-diffusion
- **分类: cs.CV**

- **简介: 该论文提出DeblurSDI，解决盲图像去模糊任务中依赖先验或大量训练的问题。通过自扩散机制，零样本、自监督地迭代优化清晰图像与模糊核，结合数据一致性和L1正则，实现无需预训练的高效去模糊，显著提升对复杂模糊的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.27439v1](http://arxiv.org/pdf/2510.27439v1)**

> **作者:** Yanlong Yang; Guanxiong Luo
>
> **摘要:** Blind image deconvolution is a challenging ill-posed inverse problem, where both the latent sharp image and the blur kernel are unknown. Traditional methods often rely on handcrafted priors, while modern deep learning approaches typically require extensive pre-training on large external datasets, limiting their adaptability to real-world scenarios. In this work, we propose DeblurSDI, a zero-shot, self-supervised framework based on self-diffusion (SDI) that requires no prior training. DeblurSDI formulates blind deconvolution as an iterative reverse self-diffusion process that starts from pure noise and progressively refines the solution. At each step, two randomly-initialized neural networks are optimized continuously to refine the sharp image and the blur kernel. The optimization is guided by an objective function combining data consistency with a sparsity-promoting L1-norm for the kernel. A key innovation is our noise scheduling mechanism, which stabilizes the optimization and provides remarkable robustness to variations in blur kernel size. These allow DeblurSDI to dynamically learn an instance-specific prior tailored to the input image. Extensive experiments demonstrate that DeblurSDI consistently achieves superior performance, recovering sharp images and accurate kernels even in highly degraded scenarios.
>
---
#### [new 074] ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning
- **分类: cs.CV**

- **简介: 该论文提出ThinkMorph模型，解决多模态推理中文本与视觉思维如何有效协同的问题。通过24K高质量交错推理数据训练，模型实现文本与图像的互补性推理，提升视觉任务性能并展现自适应切换、视觉操作等涌现能力，推动统一多模态推理模型发展。**

- **链接: [http://arxiv.org/pdf/2510.27492v1](http://arxiv.org/pdf/2510.27492v1)**

> **作者:** Jiawei Gu; Yunzhuo Hao; Huichen Will Wang; Linjie Li; Michael Qizhe Shieh; Yejin Choi; Ranjay Krishna; Yu Cheng
>
> **备注:** project page: https://thinkmorph.github.io/
>
> **摘要:** Multimodal reasoning requires iterative coordination between language and vision, yet it remains unclear what constitutes a meaningful interleaved chain of thought. We posit that text and image thoughts should function as complementary, rather than isomorphic, modalities that mutually advance reasoning. Guided by this principle, we build ThinkMorph, a unified model fine-tuned on 24K high-quality interleaved reasoning traces spanning tasks with varying visual engagement. ThinkMorph learns to generate progressive text-image reasoning steps that concretely manipulate visual content while maintaining coherent verbal logic. It delivers large gains on vision-centric benchmarks (averaging 34.7% over the base model) and generalizes to out-of-domain tasks, matching or surpassing larger and proprietary VLMs. Beyond performance, ThinkMorph exhibits emergent multimodal intelligence, including unseen visual manipulation skills, adaptive switching between reasoning modes, and better test-time scaling through diversified multimodal thoughts.These findings suggest promising directions for characterizing the emergent capabilities of unified models for multimodal reasoning.
>
---
#### [new 075] Sketch-to-Layout: Sketch-Guided Multimodal Layout Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Sketch-to-Layout任务，用用户手绘草图作为直观约束生成高质量版面。为解决标注数据稀缺问题，提出高效合成训练草图的方法。基于多模态Transformer模型，在多个公开数据集上验证了方法优于现有技术，显著提升设计易用性。**

- **链接: [http://arxiv.org/pdf/2510.27632v1](http://arxiv.org/pdf/2510.27632v1)**

> **作者:** Riccardo Brioschi; Aleksandr Alekseev; Emanuele Nevali; Berkay Döner; Omar El Malki; Blagoj Mitrevski; Leandro Kieliger; Mark Collier; Andrii Maksai; Jesse Berent; Claudiu Musat; Efi Kokiopoulou
>
> **备注:** 15 pages, 18 figures, GitHub link: https://github.com/google-deepmind/sketch_to_layout, accept at ICCV 2025 Workshop (HiGen)
>
> **摘要:** Graphic layout generation is a growing research area focusing on generating aesthetically pleasing layouts ranging from poster designs to documents. While recent research has explored ways to incorporate user constraints to guide the layout generation, these constraints often require complex specifications which reduce usability. We introduce an innovative approach exploiting user-provided sketches as intuitive constraints and we demonstrate empirically the effectiveness of this new guidance method, establishing the sketch-to-layout problem as a promising research direction, which is currently under-explored. To tackle the sketch-to-layout problem, we propose a multimodal transformer-based solution using the sketch and the content assets as inputs to produce high quality layouts. Since collecting sketch training data from human annotators to train our model is very costly, we introduce a novel and efficient method to synthetically generate training sketches at scale. We train and evaluate our model on three publicly available datasets: PubLayNet, DocLayNet and SlidesVQA, demonstrating that it outperforms state-of-the-art constraint-based methods, while offering a more intuitive design experience. In order to facilitate future sketch-to-layout research, we release O(200k) synthetically-generated sketches for the public datasets above. The datasets are available at https://github.com/google-deepmind/sketch_to_layout.
>
---
#### [new 076] Generating Accurate and Detailed Captions for High-Resolution Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对高分辨率图像生成准确详细描述的任务，解决视觉语言模型因预训练于低分辨率图像而导致细节丢失和幻觉的问题。提出融合VLM、LLM与目标检测的多阶段优化流程，通过识别关键对象并补全遗漏内容，提升描述准确性与细节丰富度，有效减少幻觉。**

- **链接: [http://arxiv.org/pdf/2510.27164v1](http://arxiv.org/pdf/2510.27164v1)**

> **作者:** Hankyeol Lee; Gawon Seo; Kyounggyu Lee; Dogun Kim; Kyungwoo Song; Jiyoung Jung
>
> **备注:** Work conducted in 2024; released for archival purposes
>
> **摘要:** Vision-language models (VLMs) often struggle to generate accurate and detailed captions for high-resolution images since they are typically pre-trained on low-resolution inputs (e.g., 224x224 or 336x336 pixels). Downscaling high-resolution images to these dimensions may result in the loss of visual details and the omission of important objects. To address this limitation, we propose a novel pipeline that integrates vision-language models, large language models (LLMs), and object detection systems to enhance caption quality. Our proposed pipeline refines captions through a novel, multi-stage process. Given a high-resolution image, an initial caption is first generated using a VLM, and key objects in the image are then identified by an LLM. The LLM predicts additional objects likely to co-occur with the identified key objects, and these predictions are verified by object detection systems. Newly detected objects not mentioned in the initial caption undergo focused, region-specific captioning to ensure they are incorporated. This process enriches caption detail while reducing hallucinations by removing references to undetected objects. We evaluate the enhanced captions using pairwise comparison and quantitative scoring from large multimodal models, along with a benchmark for hallucination detection. Experiments on a curated dataset of high-resolution images demonstrate that our pipeline produces more detailed and reliable image captions while effectively minimizing hallucinations.
>
---
#### [new 077] PETAR: Localized Findings Generation with Mask-Aware Vision-Language Modeling for PET Automated Reporting
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文面向3D PET/CT影像的自动化报告生成任务，旨在解决海量体积数据中微小分散病灶定位与描述难题。提出PETAR-4B模型，融合PET、CT及病灶掩码，实现空间精准的病灶描述生成。构建超1.1万例的标注数据集，显著提升报告准确性与临床可用性。**

- **链接: [http://arxiv.org/pdf/2510.27680v1](http://arxiv.org/pdf/2510.27680v1)**

> **作者:** Danyal Maqbool; Changhee Lee; Zachary Huemann; Samuel D. Church; Matthew E. Larson; Scott B. Perlman; Tomas A. Romero; Joshua D. Warner; Meghan Lubner; Xin Tie; Jameson Merkow; Junjie Hu; Steve Y. Cho; Tyler J. Bradshaw
>
> **摘要:** Recent advances in vision-language models (VLMs) have enabled impressive multimodal reasoning, yet most medical applications remain limited to 2D imaging. In this work, we extend VLMs to 3D positron emission tomography and computed tomography (PET/CT), a domain characterized by large volumetric data, small and dispersed lesions, and lengthy radiology reports. We introduce a large-scale dataset comprising over 11,000 lesion-level descriptions paired with 3D segmentations from more than 5,000 PET/CT exams, extracted via a hybrid rule-based and large language model (LLM) pipeline. Building upon this dataset, we propose PETAR-4B, a 3D mask-aware vision-language model that integrates PET, CT, and lesion contours for spatially grounded report generation. PETAR bridges global contextual reasoning with fine-grained lesion awareness, producing clinically coherent and localized findings. Comprehensive automated and human evaluations demonstrate that PETAR substantially improves PET/CT report generation quality, advancing 3D medical vision-language understanding.
>
---
#### [new 078] WildfireX-SLAM: A Large-scale Low-altitude RGB-D Dataset for Wildfire SLAM and Beyond
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对大尺度森林场景下3DGS-SLAM缺乏高质量数据的问题，构建了首个大规模低空RGB-D合成数据集WildfireX-SLAM，包含16km²森林区域的5.5k张图像。通过引擎实现多模态数据采集与环境可控性，支持火灾应急响应等应用，并开展基准测试揭示挑战与改进方向。**

- **链接: [http://arxiv.org/pdf/2510.27133v1](http://arxiv.org/pdf/2510.27133v1)**

> **作者:** Zhicong Sun; Jacqueline Lo; Jinxing Hu
>
> **备注:** This paper has been accepted by MMM 2026
>
> **摘要:** 3D Gaussian splatting (3DGS) and its subsequent variants have led to remarkable progress in simultaneous localization and mapping (SLAM). While most recent 3DGS-based SLAM works focus on small-scale indoor scenes, developing 3DGS-based SLAM methods for large-scale forest scenes holds great potential for many real-world applications, especially for wildfire emergency response and forest management. However, this line of research is impeded by the absence of a comprehensive and high-quality dataset, and collecting such a dataset over real-world scenes is costly and technically infeasible. To this end, we have built a large-scale, comprehensive, and high-quality synthetic dataset for SLAM in wildfire and forest environments. Leveraging the Unreal Engine 5 Electric Dreams Environment Sample Project, we developed a pipeline to easily collect aerial and ground views, including ground-truth camera poses and a range of additional data modalities from unmanned aerial vehicle. Our pipeline also provides flexible controls on environmental factors such as light, weather, and types and conditions of wildfire, supporting the need for various tasks covering forest mapping, wildfire emergency response, and beyond. The resulting pilot dataset, WildfireX-SLAM, contains 5.5k low-altitude RGB-D aerial images from a large-scale forest map with a total size of 16 km2. On top of WildfireX-SLAM, a thorough benchmark is also conducted, which not only reveals the unique challenges of 3DGS-based SLAM in the forest but also highlights potential improvements for future works. The dataset and code will be publicly available. Project page: https://zhicongsun.github.io/wildfirexslam.
>
---
#### [new 079] VessShape: Few-shot 2D blood vessel segmentation by leveraging shape priors from synthetic images
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对血管语义分割任务，解决标注数据稀缺与模型跨模态泛化差的问题。提出VessShape方法，通过生成含管状结构先验的合成数据，引导模型学习形状特征而非纹理。实验表明，预训练模型在少量真实数据上即可实现高效分割，并具备零样本迁移能力。**

- **链接: [http://arxiv.org/pdf/2510.27646v1](http://arxiv.org/pdf/2510.27646v1)**

> **作者:** Cesar H. Comin; Wesley N. Galvão
>
> **摘要:** Semantic segmentation of blood vessels is an important task in medical image analysis, but its progress is often hindered by the scarcity of large annotated datasets and the poor generalization of models across different imaging modalities. A key aspect is the tendency of Convolutional Neural Networks (CNNs) to learn texture-based features, which limits their performance when applied to new domains with different visual characteristics. We hypothesize that leveraging geometric priors of vessel shapes, such as their tubular and branching nature, can lead to more robust and data-efficient models. To investigate this, we introduce VessShape, a methodology for generating large-scale 2D synthetic datasets designed to instill a shape bias in segmentation models. VessShape images contain procedurally generated tubular geometries combined with a wide variety of foreground and background textures, encouraging models to learn shape cues rather than textures. We demonstrate that a model pre-trained on VessShape images achieves strong few-shot segmentation performance on two real-world datasets from different domains, requiring only four to ten samples for fine-tuning. Furthermore, the model exhibits notable zero-shot capabilities, effectively segmenting vessels in unseen domains without any target-specific training. Our results indicate that pre-training with a strong shape bias can be an effective strategy to overcome data scarcity and improve model generalization in blood vessel segmentation.
>
---
#### [new 080] Modality Alignment across Trees on Heterogeneous Hyperbolic Manifolds
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视觉-语言模型中模态对齐不均衡问题，提出跨树对齐方法。通过构建图像与文本的层次化特征树，并嵌入不同曲率的双曲流形，利用KL距离与中间流形实现异质流形间的对齐，有效提升少样本与跨域分类性能。**

- **链接: [http://arxiv.org/pdf/2510.27391v1](http://arxiv.org/pdf/2510.27391v1)**

> **作者:** Wu Wei; Xiaomeng Fan; Yuwei Wu; Zhi Gao; Pengxiang Li; Yunde Jia; Mehrtash Harandi
>
> **摘要:** Modality alignment is critical for vision-language models (VLMs) to effectively integrate information across modalities. However, existing methods extract hierarchical features from text while representing each image with a single feature, leading to asymmetric and suboptimal alignment. To address this, we propose Alignment across Trees, a method that constructs and aligns tree-like hierarchical features for both image and text modalities. Specifically, we introduce a semantic-aware visual feature extraction framework that applies a cross-attention mechanism to visual class tokens from intermediate Transformer layers, guided by textual cues to extract visual features with coarse-to-fine semantics. We then embed the feature trees of the two modalities into hyperbolic manifolds with distinct curvatures to effectively model their hierarchical structures. To align across the heterogeneous hyperbolic manifolds with different curvatures, we formulate a KL distance measure between distributions on heterogeneous manifolds, and learn an intermediate manifold for manifold alignment by minimizing the distance. We prove the existence and uniqueness of the optimal intermediate manifold. Experiments on taxonomic open-set classification tasks across multiple image datasets demonstrate that our method consistently outperforms strong baselines under few-shot and cross-domain settings.
>
---
#### [new 081] Soft Task-Aware Routing of Experts for Equivariant Representation Learning
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文针对等变表示学习中不变与等变特征冗余问题，提出软任务感知路由（STAR）机制。通过将投影头建模为专家，引导其分工捕获共享或特定任务信息，减少冗余，提升模型效率。实验表明，该方法在多种迁移学习任务中均有效。**

- **链接: [http://arxiv.org/pdf/2510.27222v1](http://arxiv.org/pdf/2510.27222v1)**

> **作者:** Jaebyeong Jeon; Hyeonseo Jang; Jy-yong Sohn; Kibok Lee
>
> **备注:** NeurIPS 2025
>
> **摘要:** Equivariant representation learning aims to capture variations induced by input transformations in the representation space, whereas invariant representation learning encodes semantic information by disregarding such transformations. Recent studies have shown that jointly learning both types of representations is often beneficial for downstream tasks, typically by employing separate projection heads. However, this design overlooks information shared between invariant and equivariant learning, which leads to redundant feature learning and inefficient use of model capacity. To address this, we introduce Soft Task-Aware Routing (STAR), a routing strategy for projection heads that models them as experts. STAR induces the experts to specialize in capturing either shared or task-specific information, thereby reducing redundant feature learning. We validate this effect by observing lower canonical correlations between invariant and equivariant embeddings. Experimental results show consistent improvements across diverse transfer learning tasks. The code is available at https://github.com/YonseiML/star.
>
---
#### [new 082] A fragile zero-watermarking method based on dual quaternion matrix decomposition
- **分类: eess.IV; cs.CV; cs.NA; math.NA; 65F99; G.1.3**

- **简介: 该论文针对医疗图像版权保护与篡改检测问题，提出一种基于双四元数矩阵分解的脆弱零水印方法。通过提取图像稳定特征，利用双四元数的结构关系实现原图与水印的关联，生成不修改原始图像的水印信息，从而实现版权认证与内容篡改检测。**

- **链接: [http://arxiv.org/pdf/2510.27307v1](http://arxiv.org/pdf/2510.27307v1)**

> **作者:** Mingcui Zhang; Zhigang Jia
>
> **备注:** 18 pages, 6 figures, 3 tables
>
> **摘要:** Medical images play a crucial role in assisting diagnosis, remote consultation, and academic research. However, during the transmission and sharing process, they face serious risks of copyright ownership and content tampering. Therefore, protecting medical images is of great importance. As an effective means of image copyright protection, zero-watermarking technology focuses on constructing watermarks without modifying the original carrier by extracting its stable features, which provides an ideal approach for protecting medical images. This paper aims to propose a fragile zero-watermarking model based on dual quaternion matrix decomposition, which utilizes the operational relationship between the standard part and the dual part of dual quaternions to correlate the original carrier image with the watermark image, and generates zero-watermarking information based on the characteristics of dual quaternion matrix decomposition, ultimately achieving copyright protection and content tampering detection for medical images.
>
---
#### [new 083] Generative diffusion modeling protocols for improving the Kikuchi pattern indexing in electron back-scatter diffraction
- **分类: cond-mat.mtrl-sci; cs.CV**

- **简介: 该论文针对电子背散射衍射（EBSD）在高速扫描下因曝光时间短导致的衍射图样噪声大、索引精度低的问题，提出基于生成式机器学习模型的后处理或实时处理方法，用于恢复噪声图样。研究发现该方法不依赖大量数据，有效提升了高扫描速度下的晶体取向索引可靠性。**

- **链接: [http://arxiv.org/pdf/2510.26907v1](http://arxiv.org/pdf/2510.26907v1)**

> **作者:** Meghraj Prajapata; Alankar Alankar
>
> **摘要:** Electron back-scatter diffraction (EBSD) has traditionally relied upon methods such as the Hough transform and dictionary Indexing to interpret diffraction patterns and extract crystallographic orientation. However, these methods encounter significant limitations, particularly when operating at high scanning speeds, where the exposure time per pattern is decreased beyond the operating sensitivity of CCD camera. Hence the signal to noise ratio decreases for the observed pattern which makes the pattern noisy, leading to reduced indexing accuracy. This research work aims to develop generative machine learning models for the post-processing or on-the-fly processing of Kikuchi patterns which are capable of restoring noisy EBSD patterns obtained at high scan speeds. These restored patterns can be used for the determination of crystal orientations to provide reliable indexing results. We compare the performance of such generative models in enhancing the quality of patterns captured at short exposure times (high scan speeds). An interesting observation is that the methodology is not data-hungry as typical machine learning methods.
>
---
#### [new 084] GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation
- **分类: cs.AI; cs.CV**

- **简介: 该论文针对GUI导航中跨域泛化弱、历史利用低的问题，提出GUI-Rise框架，融合结构化推理、动作预测与历史摘要。通过监督微调与GRPO强化学习训练，实现高效推理与泛化，显著提升导航性能，尤其在跨域场景表现优异。**

- **链接: [http://arxiv.org/pdf/2510.27210v1](http://arxiv.org/pdf/2510.27210v1)**

> **作者:** Tao Liu; Chongyu Wang; Rongjie Li; Yingchen Yu; Xuming He; Bai Song
>
> **备注:** Published in NeurIPS 2025
>
> **摘要:** While Multimodal Large Language Models (MLLMs) have advanced GUI navigation agents, current approaches face limitations in cross-domain generalization and effective history utilization. We present a reasoning-enhanced framework that systematically integrates structured reasoning, action prediction, and history summarization. The structured reasoning component generates coherent Chain-of-Thought analyses combining progress estimation and decision reasoning, which inform both immediate action predictions and compact history summaries for future steps. Based on this framework, we train a GUI agent, \textbf{GUI-Rise}, through supervised fine-tuning on pseudo-labeled trajectories and reinforcement learning with Group Relative Policy Optimization (GRPO). This framework employs specialized rewards, including a history-aware objective, directly linking summary quality to subsequent action performance. Comprehensive evaluations on standard benchmarks demonstrate state-of-the-art results under identical training data conditions, with particularly strong performance in out-of-domain scenarios. These findings validate our framework's ability to maintain robust reasoning and generalization across diverse GUI navigation tasks. Code is available at https://leon022.github.io/GUI-Rise.
>
---
#### [new 085] Using Salient Object Detection to Identify Manipulative Cookie Banners that Circumvent GDPR
- **分类: cs.CY; cs.AI; cs.CV; cs.HC**

- **简介: 该论文研究网页饼干公告是否符合GDPR及是否存在美学操纵。通过计算机视觉的显著目标检测，分析2579个网站，发现38%合规公告仍含操纵设计，且欧盟网站更倾向使用此类设计，揭示了监管环境对设计策略的影响。**

- **链接: [http://arxiv.org/pdf/2510.26967v1](http://arxiv.org/pdf/2510.26967v1)**

> **作者:** Riley Grossman; Michael Smith; Cristian Borcea; Yi Chen
>
> **备注:** Accepted to International AAAI Conference on Web and Social Media 2026 (ICWSM'26)
>
> **摘要:** The main goal of this paper is to study how often cookie banners that comply with the General Data Protection Regulation (GDPR) contain aesthetic manipulation, a design tactic to draw users' attention to the button that permits personal data sharing. As a byproduct of this goal, we also evaluate how frequently the banners comply with GDPR and the recommendations of national data protection authorities regarding banner designs. We visited 2,579 websites and identified the type of cookie banner implemented. Although 45% of the relevant websites have fully compliant banners, we found aesthetic manipulation on 38% of the compliant banners. Unlike prior studies of aesthetic manipulation, we use a computer vision model for salient object detection to measure how salient (i.e., attention-drawing) each banner element is. This enables the discovery of new types of aesthetic manipulation (e.g., button placement), and leads us to conclude that aesthetic manipulation is more common than previously reported (38% vs 27% of banners). To study the effects of user and/or website location on cookie banner design, we include websites within the European Union (EU), where privacy regulation enforcement is more stringent, and websites outside the EU. We visited websites from IP addresses in the EU and from IP addresses in the United States (US). We find that 13.9% of EU websites change their banner design when the user is from the US, and EU websites are roughly 48.3% more likely to use aesthetic manipulation than non-EU websites, highlighting their innovative responses to privacy regulation.
>
---
#### [new 086] See the Speaker: Crafting High-Resolution Talking Faces from Speech with Prior Guidance and Region Refinement
- **分类: eess.AS; cs.AI; cs.CV; cs.SD**

- **简介: 该论文聚焦语音驱动高分辨率说话人脸生成任务，解决仅依赖语音输入时外观与动作同步难题。提出两阶段框架：先用语音条件扩散模型结合面部先验生成人脸，再通过区域增强模块优化唇音同步与表情动态，最终融合离散码本与渲染网络实现端到端高清视频生成。**

- **链接: [http://arxiv.org/pdf/2510.26819v1](http://arxiv.org/pdf/2510.26819v1)**

> **作者:** Jinting Wang; Jun Wang; Hei Victor Cheng; Li Liu
>
> **备注:** 16 pages,15 figures, accepted by TASLP
>
> **摘要:** Unlike existing methods that rely on source images as appearance references and use source speech to generate motion, this work proposes a novel approach that directly extracts information from the speech, addressing key challenges in speech-to-talking face. Specifically, we first employ a speech-to-face portrait generation stage, utilizing a speech-conditioned diffusion model combined with statistical facial prior and a sample-adaptive weighting module to achieve high-quality portrait generation. In the subsequent speech-driven talking face generation stage, we embed expressive dynamics such as lip movement, facial expressions, and eye movements into the latent space of the diffusion model and further optimize lip synchronization using a region-enhancement module. To generate high-resolution outputs, we integrate a pre-trained Transformer-based discrete codebook with an image rendering network, enhancing video frame details in an end-to-end manner. Experimental results demonstrate that our method outperforms existing approaches on the HDTF, VoxCeleb, and AVSpeech datasets. Notably, this is the first method capable of generating high-resolution, high-quality talking face videos exclusively from a single speech input.
>
---
#### [new 087] Imbalanced Classification through the Lens of Spurious Correlations
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对类别不平衡问题，提出从虚假相关性（Clever Hans效应）视角分析并解决。通过可解释AI识别和消除不平衡下产生的误导性关联，提升模型可靠性。工作聚焦于改进不平衡分类任务，实现更准确、可信的分类性能。**

- **链接: [http://arxiv.org/pdf/2510.27650v1](http://arxiv.org/pdf/2510.27650v1)**

> **作者:** Jakob Hackstein; Sidney Bender
>
> **摘要:** Class imbalance poses a fundamental challenge in machine learning, frequently leading to unreliable classification performance. While prior methods focus on data- or loss-reweighting schemes, we view imbalance as a data condition that amplifies Clever Hans (CH) effects by underspecification of minority classes. In a counterfactual explanations-based approach, we propose to leverage Explainable AI to jointly identify and eliminate CH effects emerging under imbalance. Our method achieves competitive classification performance on three datasets and demonstrates how CH effects emerge under imbalance, a perspective largely overlooked by existing approaches.
>
---
#### [new 088] Dark-Field X-Ray Imaging Significantly Improves Deep-Learning based Detection of Synthetic Early-Stage Lung Tumors in Preclinical Models
- **分类: physics.med-ph; cs.CV; cs.LG; eess.IV; physics.optics**

- **简介: 该论文属于医学图像分析任务，旨在提升早期肺癌肿瘤检测率。针对低剂量CT普及难、假阳性率高的问题，研究利用暗场X射线成像（DFI）结合深度学习，通过合成肿瘤数据训练U-Net模型，结果表明DFI显著提高检测灵敏度（83.7%），优于传统衰减成像。**

- **链接: [http://arxiv.org/pdf/2510.27679v1](http://arxiv.org/pdf/2510.27679v1)**

> **作者:** Joyoni Dey; Hunter C. Meyer; Murtuza S. Taqi
>
> **摘要:** Low-dose computed tomography (LDCT) is the current standard for lung cancer screening, yet its adoption and accessibility remain limited. Many regions lack LDCT infrastructure, and even among those screened, early-stage cancer detection often yield false positives, as shown in the National Lung Screening Trial (NLST) with a sensitivity of 93.8 percent and a false-positive rate of 26.6 percent. We aim to investigate whether X-ray dark-field imaging (DFI) radiograph, a technique sensitive to small-angle scatter from alveolar microstructure and less susceptible to organ shadowing, can significantly improve early-stage lung tumor detection when coupled with deep-learning segmentation. Using paired attenuation (ATTN) and DFI radiograph images of euthanized mouse lungs, we generated realistic synthetic tumors with irregular boundaries and intensity profiles consistent with physical lung contrast. A U-Net segmentation network was trained on small patches using either ATTN, DFI, or a combination of ATTN and DFI channels.Results show that the DFI-only model achieved a true-positive detection rate of 83.7 percent, compared with 51 percent for ATTN-only, while maintaining comparable specificity (90.5 versus 92.9 percent). The combined ATTN and DFI input achieved 79.6 percent sensitivity and 97.6 percent specificity. In conclusion, DFI substantially improves early-tumor detectability in comparison to standard attenuation radiography and shows potential as an accessible, low-cost, low-dose alternative for pre-clinical or limited-resource screening where LDCT is unavailable.
>
---
#### [new 089] A Multi-Modal Neuro-Symbolic Approach for Spatial Reasoning-Based Visual Grounding in Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对机器人视觉接地中的空间推理任务，解决现有模型依赖图像、缺乏显式逻辑推理的问题。提出多模态神经符号框架，融合全景图与3D点云，通过感知与符号推理模块构建结构化场景图，实现精准可解释的空间关系理解，在复杂环境中表现优异。**

- **链接: [http://arxiv.org/pdf/2510.27033v1](http://arxiv.org/pdf/2510.27033v1)**

> **作者:** Simindokht Jahangard; Mehrzad Mohammadi; Abhinav Dhall; Hamid Rezatofighi
>
> **摘要:** Visual reasoning, particularly spatial reasoning, is a challenging cognitive task that requires understanding object relationships and their interactions within complex environments, especially in robotics domain. Existing vision_language models (VLMs) excel at perception tasks but struggle with fine-grained spatial reasoning due to their implicit, correlation-driven reasoning and reliance solely on images. We propose a novel neuro_symbolic framework that integrates both panoramic-image and 3D point cloud information, combining neural perception with symbolic reasoning to explicitly model spatial and logical relationships. Our framework consists of a perception module for detecting entities and extracting attributes, and a reasoning module that constructs a structured scene graph to support precise, interpretable queries. Evaluated on the JRDB-Reasoning dataset, our approach demonstrates superior performance and reliability in crowded, human_built environments while maintaining a lightweight design suitable for robotics and embodied AI applications.
>
---
#### [new 090] Audio-Visual Speech Enhancement In Complex Scenarios With Separation And Dereverberation Joint Modeling
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文针对复杂场景下的音视频语音增强任务，解决混响与干扰噪声导致的语音质量差问题。提出“先分离后去混响”的联合建模框架，有效提升语音清晰度与可懂度，在第四届音视频语音增强挑战赛中三项客观指标领先，并获主观听感第一名。**

- **链接: [http://arxiv.org/pdf/2510.26825v1](http://arxiv.org/pdf/2510.26825v1)**

> **作者:** Jiarong Du; Zhan Jin; Peijun Yang; Juan Liu; Zhuo Li; Xin Liu; Ming Li
>
> **摘要:** Audio-visual speech enhancement (AVSE) is a task that uses visual auxiliary information to extract a target speaker's speech from mixed audio. In real-world scenarios, there often exist complex acoustic environments, accompanied by various interfering sounds and reverberation. Most previous methods struggle to cope with such complex conditions, resulting in poor perceptual quality of the extracted speech. In this paper, we propose an effective AVSE system that performs well in complex acoustic environments. Specifically, we design a "separation before dereverberation" pipeline that can be extended to other AVSE networks. The 4th COGMHEAR Audio-Visual Speech Enhancement Challenge (AVSEC) aims to explore new approaches to speech processing in multimodal complex environments. We validated the performance of our system in AVSEC-4: we achieved excellent results in the three objective metrics on the competition leaderboard, and ultimately secured first place in the human subjective listening test.
>
---
#### [new 091] Visual Backdoor Attacks on MLLM Embodied Decision Making via Contrastive Trigger Learning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究多模态大模型驱动的具身智能体的视觉后门攻击问题。针对环境物体作为触发器的复杂性，提出BEAT框架，通过构建多样训练集与两阶段训练（含对比触发学习CTL），实现高成功率且鲁棒的后门攻击，揭示了具身智能系统的关键安全风险。**

- **链接: [http://arxiv.org/pdf/2510.27623v1](http://arxiv.org/pdf/2510.27623v1)**

> **作者:** Qiusi Zhan; Hyeonjeong Ha; Rui Yang; Sirui Xu; Hanyang Chen; Liang-Yan Gui; Yu-Xiong Wang; Huan Zhang; Heng Ji; Daniel Kang
>
> **摘要:** Multimodal large language models (MLLMs) have advanced embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into MLLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and MLLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in MLLM-based embodied agents, underscoring the need for robust defenses before real-world deployment.
>
---
## 更新

#### [replaced 001] Continual Vision-and-Language Navigation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.15049v3](http://arxiv.org/pdf/2403.15049v3)**

> **作者:** Seongjun Jeong; Gi-Cheon Kang; Seongho Choi; Joochan Kim; Byoung-Tak Zhang
>
> **摘要:** Developing Vision-and-Language Navigation (VLN) agents typically assumes a \textit{train-once-deploy-once} strategy, which is unrealistic as deployed agents continually encounter novel environments. To address this, we propose the Continual Vision-and-Language Navigation (CVLN) paradigm, where agents learn and adapt incrementally across multiple \textit{scene domains}. CVLN includes two setups: Initial-instruction based CVLN for instruction-following, and Dialogue-based CVLN for dialogue-guided navigation. We also introduce two simple yet effective baselines for sequential decision-making: Perplexity Replay (PerpR), which replays difficult episodes, and Episodic Self-Replay (ESR), which stores and revisits action logits during training. Experiments show that existing continual learning methods fall short for CVLN, while PerpR and ESR achieve better performance by efficiently utilizing replay memory.
>
---
#### [replaced 002] AVA: Towards Agentic Video Analytics with Vision Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.00254v5](http://arxiv.org/pdf/2505.00254v5)**

> **作者:** Yuxuan Yan; Shiqi Jiang; Ting Cao; Yifan Yang; Qianqian Yang; Yuanchao Shu; Yuqing Yang; Lili Qiu
>
> **备注:** Accepted to NDSI 2026, 19pages, 12 figures, complementary evaluations and appendix
>
> **摘要:** AI-driven video analytics has become increasingly important across diverse domains. However, existing systems are often constrained to specific, predefined tasks, limiting their adaptability in open-ended analytical scenarios. The recent emergence of Vision Language Models (VLMs) as transformative technologies offers significant potential for enabling open-ended video understanding, reasoning, and analytics. Nevertheless, their limited context windows present challenges when processing ultra-long video content, which is prevalent in real-world applications. To address this, we introduce AVA, a VLM-powered system designed for open-ended, advanced video analytics. AVA incorporates two key innovations: (1) the near real-time construction of Event Knowledge Graphs (EKGs) for efficient indexing of long or continuous video streams, and (2) an agentic retrieval-generation mechanism that leverages EKGs to handle complex and diverse queries. Comprehensive evaluations on public benchmarks, LVBench and VideoMME-Long, demonstrate that AVA achieves state-of-the-art performance, attaining 62.3% and 64.1% accuracy, respectively-significantly surpassing existing VLM and video Retrieval-Augmented Generation (RAG) systems. Furthermore, to evaluate video analytics in ultra-long and open-world video scenarios, we introduce a new benchmark, AVA-100. This benchmark comprises 8 videos, each exceeding 10 hours in duration, along with 120 manually annotated, diverse, and complex question-answer pairs. On AVA-100, AVA achieves top-tier performance with an accuracy of 75.8%. The source code of AVA is available at https://github.com/I-ESC/Project-Ava. The AVA-100 benchmark can be accessed at https://huggingface.co/datasets/iesc/Ava-100.
>
---
#### [replaced 003] LV-UNet: A Lightweight and Vanilla Model for Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.16886v4](http://arxiv.org/pdf/2408.16886v4)**

> **作者:** Juntao Jiang; Mengmeng Wang; Huizhong Tian; Lingbo Cheng; Yong Liu
>
> **备注:** Accepted by IEEE BIBM2024 ML4BMI workshop
>
> **摘要:** While large models have achieved significant progress in computer vision, challenges such as optimization complexity, the intricacy of transformer architectures, computational constraints, and practical application demands highlight the importance of simpler model designs in medical image segmentation. This need is particularly pronounced in mobile medical devices, which require lightweight, deployable models with real-time performance. However, existing lightweight models often suffer from poor robustness across datasets, limiting their widespread adoption. To address these challenges, this paper introduces LV-UNet, a lightweight and vanilla model that leverages pre-trained MobileNetv3-Large backbones and incorporates fusible modules. LV-UNet employs an enhanced deep training strategy and switches to a deployment mode during inference by re-parametrization, significantly reducing parameter count and computational overhead. Experimental results on ISIC 2016, BUSI, CVC-ClinicDB, CVC-ColonDB, and Kvair-SEG datasets demonstrate a better trade-off between performance and the computational load. The code will be released at https://github.com/juntaoJianggavin/LV-UNet.
>
---
#### [replaced 004] D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06264v2](http://arxiv.org/pdf/2504.06264v2)**

> **作者:** Jisang Han; Honggyu An; Jaewoo Jung; Takuya Narihira; Junyoung Seo; Kazumi Fukuda; Chaehyun Kim; Sunghwan Hong; Yuki Mitsufuji; Seungryong Kim
>
> **备注:** NeurIPS 2025; project page: https://cvlab-kaist.github.io/DDUSt3R/
>
> **摘要:** In this work, we address the task of 3D reconstruction in dynamic scenes, where object motions frequently degrade the quality of previous 3D pointmap regression methods, such as DUSt3R, that are originally designed for static 3D scene reconstruction. Although these methods provide an elegant and powerful solution in static settings, they struggle in the presence of dynamic motions that disrupt alignment based solely on camera poses. To overcome this, we propose $D^2USt3R$ that directly regresses Static-Dynamic Aligned Pointmaps (SDAP) that simultaneiously capture both static and dynamic 3D scene geometry. By explicitly incorporating both spatial and temporal aspects, our approach successfully encapsulates 3D dense correspondence to the proposed pointmaps, enhancing downstream tasks. Extensive experimental evaluations demonstrate that our proposed approach consistently achieves superior 3D reconstruction performance across various datasets featuring complex motions.
>
---
#### [replaced 005] CARE: Contrastive Alignment for ADL Recognition from Event-Triggered Sensor Streams
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.16988v2](http://arxiv.org/pdf/2510.16988v2)**

> **作者:** Junhao Zhao; Zishuai Liu; Ruili Fang; Jin Lu; Linghan Zhang; Fei Dou
>
> **摘要:** The recognition of Activities of Daily Living (ADLs) from event-triggered ambient sensors is an essential task in Ambient Assisted Living, yet existing methods remain constrained by representation-level limitations. Sequence-based approaches preserve temporal order of sensor activations but are sensitive to noise and lack spatial awareness, while image-based approaches capture global patterns and implicit spatial correlations but compress fine-grained temporal dynamics and distort sensor layouts. Naive fusion (e.g., feature concatenation) fail to enforce alignment between sequence- and image-based representation views, underutilizing their complementary strengths. We propose Contrastive Alignment for ADL Recognition from Event-Triggered Sensor Streams (CARE), an end-to-end framework that jointly optimizes representation learning via Sequence-Image Contrastive Alignment (SICA) and classification via cross-entropy, ensuring both cross-representation alignment and task-specific discriminability. CARE integrates (i) time-aware, noise-resilient sequence encoding with (ii) spatially-informed and frequency-sensitive image representations, and employs (iii) a joint contrastive-classification objective for end-to-end learning of aligned and discriminative embeddings. Evaluated on three CASAS datasets, CARE achieves state-of-the-art performance (89.8% on Milan, 88.9% on Cairo, and 73.3% on Kyoto7) and demonstrates robustness to sensor malfunctions and layout variability, highlighting its potential for reliable ADL recognition in smart homes.
>
---
#### [replaced 006] DPA: A one-stop metric to measure bias amplification in classification datasets
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.11060v3](http://arxiv.org/pdf/2412.11060v3)**

> **作者:** Bhanu Tokas; Rahul Nair; Hannah Kerner
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Most ML datasets today contain biases. When we train models on these datasets, they often not only learn these biases but can worsen them -- a phenomenon known as bias amplification. Several co-occurrence-based metrics have been proposed to measure bias amplification in classification datasets. They measure bias amplification between a protected attribute (e.g., gender) and a task (e.g., cooking). These metrics also support fine-grained bias analysis by identifying the direction in which a model amplifies biases. However, co-occurrence-based metrics have limitations -- some fail to measure bias amplification in balanced datasets, while others fail to measure negative bias amplification. To solve these issues, recent work proposed a predictability-based metric called leakage amplification (LA). However, LA cannot identify the direction in which a model amplifies biases. We propose Directional Predictability Amplification (DPA), a predictability-based metric that is (1) directional, (2) works with balanced and unbalanced datasets, and (3) correctly identifies positive and negative bias amplification. DPA eliminates the need to evaluate models on multiple metrics to verify these three aspects. DPA also improves over prior predictability-based metrics like LA: it is less sensitive to the choice of attacker function (a hyperparameter in predictability-based metrics), reports scores within a bounded range, and accounts for dataset bias by measuring relative changes in predictability. Our experiments on well-known datasets like COMPAS (a tabular dataset), COCO, and ImSitu (image datasets) show that DPA is the most reliable metric to measure bias amplification in classification problems. To compare DPA with existing bias amplification metrics, we released a one-stop library of major bias amplification metrics at https://github.com/kerner-lab/Bias-Amplification.
>
---
#### [replaced 007] C3Editor: Achieving Controllable Consistency in 2D Model for 3D Editing
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.04539v2](http://arxiv.org/pdf/2510.04539v2)**

> **作者:** Zeng Tao; Zheng Ding; Zeyuan Chen; Xiang Zhang; Leizhi Li; Zhuowen Tu
>
> **备注:** ICCV 2025 Workshop Wild3D
>
> **摘要:** Existing 2D-lifting-based 3D editing methods often encounter challenges related to inconsistency, stemming from the lack of view-consistent 2D editing models and the difficulty of ensuring consistent editing across multiple views. To address these issues, we propose C3Editor, a controllable and consistent 2D-lifting-based 3D editing framework. Given an original 3D representation and a text-based editing prompt, our method selectively establishes a view-consistent 2D editing model to achieve superior 3D editing results. The process begins with the controlled selection of a ground truth (GT) view and its corresponding edited image as the optimization target, allowing for user-defined manual edits. Next, we fine-tune the 2D editing model within the GT view and across multiple views to align with the GT-edited image while ensuring multi-view consistency. To meet the distinct requirements of GT view fitting and multi-view consistency, we introduce separate LoRA modules for targeted fine-tuning. Our approach delivers more consistent and controllable 2D and 3D editing results than existing 2D-lifting-based methods, outperforming them in both qualitative and quantitative evaluations.
>
---
#### [replaced 008] Tensor Completion via Monotone Inclusion: Generalized Low-Rank Priors Meet Deep Denoisers
- **分类: math.OC; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.12425v2](http://arxiv.org/pdf/2510.12425v2)**

> **作者:** Peng Chen; Deliang Wei; Jiale Yao; Fang Li
>
> **备注:** 14 pages, 8 figures, 6 tables
>
> **摘要:** Missing entries in multi dimensional data pose significant challenges for downstream analysis across diverse real world applications. These data are naturally represented as tensors, and recent completion methods integrating global low rank priors with plug and play denoisers have demonstrated strong empirical performance. However, these approaches often rely on empirical convergence alone or unrealistic assumptions, such as deep denoisers acting as proximal operators of implicit regularizers, which generally does not hold. To address these limitations, we propose a novel tensor completion framework grounded in the monotone inclusion paradigm. Within this framework, deep denoisers are treated as general operators that require far fewer restrictions than in classical optimization based formulations. To better capture holistic structure, we further incorporate generalized low rank priors with weakly convex penalties. Building upon the Davis Yin splitting scheme, we develop the GTCTV DPC algorithm and rigorously establish its global convergence. Extensive experiments demonstrate that GTCTV DPC consistently outperforms existing methods in both quantitative metrics and visual quality, particularly at low sampling rates. For instance, at a sampling rate of 0.05 for multi dimensional image completion, GTCTV DPC achieves an average mean peak signal to noise ratio (MPSNR) that surpasses the second best method by 0.717 dB, and 0.649 dB for multi spectral images, and color videos, respectively.
>
---
#### [replaced 009] Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23285v2](http://arxiv.org/pdf/2510.23285v2)**

> **作者:** Ruoyu Wang; Beier Zhu; Junzhi Li; Liangyu Yuan; Chi Zhang
>
> **备注:** To appear in NeurIPS 2025
>
> **摘要:** Diffusion-based generative processes, formulated as differential equation solving, frequently balance computational speed with sample quality. Our theoretical investigation of ODE- and SDE-based solvers reveals complementary weaknesses: ODE solvers accumulate irreducible gradient error along deterministic trajectories, while SDE methods suffer from amplified discretization errors when the step budget is limited. Building upon this insight, we introduce AdaSDE, a novel single-step SDE solver that aims to unify the efficiency of ODEs with the error resilience of SDEs. Specifically, we introduce a single per-step learnable coefficient, estimated via lightweight distillation, which dynamically regulates the error correction strength to accelerate diffusion sampling. Notably, our framework can be integrated with existing solvers to enhance their capabilities. Extensive experiments demonstrate state-of-the-art performance: at 5 NFE, AdaSDE achieves FID scores of 4.18 on CIFAR-10, 8.05 on FFHQ and 6.96 on LSUN Bedroom. Codes are available in https://github.com/WLU-wry02/AdaSDE.
>
---
#### [replaced 010] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.15392v4](http://arxiv.org/pdf/2410.15392v4)**

> **作者:** Bohao Liao; Wei Zhai; Zengyu Wan; Zhixin Cheng; Wenfei Yang; Tianzhu Zhang; Yang Cao; Zheng-Jun Zha
>
> **备注:** Accepted to NeurIPS 2025,Project Page: https://lbh666.github.io/ef-3dgs/
>
> **摘要:** Scene reconstruction from casually captured videos has wide applications in real-world scenarios. With recent advancements in differentiable rendering techniques, several methods have attempted to simultaneously optimize scene representations (NeRF or 3DGS) and camera poses. Despite recent progress, existing methods relying on traditional camera input tend to fail in high-speed (or equivalently low-frame-rate) scenarios. Event cameras, inspired by biological vision, record pixel-wise intensity changes asynchronously with high temporal resolution, providing valuable scene and motion information in blind inter-frame intervals. In this paper, we introduce the event camera to aid scene construction from a casually captured video for the first time, and propose Event-Aided Free-Trajectory 3DGS, called EF-3DGS, which seamlessly integrates the advantages of event cameras into 3DGS through three key components. First, we leverage the Event Generation Model (EGM) to fuse events and frames, supervising the rendered views observed by the event stream. Second, we adopt the Contrast Maximization (CMax) framework in a piece-wise manner to extract motion information by maximizing the contrast of the Image of Warped Events (IWE), thereby calibrating the estimated poses. Besides, based on the Linear Event Generation Model (LEGM), the brightness information encoded in the IWE is also utilized to constrain the 3DGS in the gradient domain. Third, to mitigate the absence of color information of events, we introduce photometric bundle adjustment (PBA) to ensure view consistency across events and frames. We evaluate our method on the public Tanks and Temples benchmark and a newly collected real-world dataset, RealEv-DAVIS. Our project page is https://lbh666.github.io/ef-3dgs/.
>
---
#### [replaced 011] AMD-Hummingbird: Towards an Efficient Text-to-Video Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18559v3](http://arxiv.org/pdf/2503.18559v3)**

> **作者:** Takashi Isobe; He Cui; Dong Zhou; Mengmeng Ge; Dong Li; Emad Barsoum
>
> **备注:** Homepage: https://www.amd.com/en/developer/resources/technical-articles/amd-hummingbird-0-9b-text-to-video-diffusion-model-with-4-step-inferencing.html| GitHub: https://github.com/AMD-AIG-AIMA/AMD-Hummingbird-T2V
>
> **摘要:** Text-to-Video (T2V) generation has attracted significant attention for its ability to synthesize realistic videos from textual descriptions. However, existing models struggle to balance computational efficiency and high visual quality, particularly on resource-limited devices, e.g.,iGPUs and mobile phones. Most prior work prioritizes visual fidelity while overlooking the need for smaller, more efficient models suitable for real-world deployment. To address this challenge, we propose a lightweight T2V framework, termed Hummingbird, which prunes existing models and enhances visual quality through visual feedback learning. Our approach reduces the size of the U-Net from 1.4 billion to 0.7 billion parameters, significantly improving efficiency while preserving high-quality video generation. Additionally, we introduce a novel data processing pipeline that leverages Large Language Models (LLMs) and Video Quality Assessment (VQA) models to enhance the quality of both text prompts and video data. To support user-driven training and style customization, we publicly release the full training code, including data processing and model training. Extensive experiments show that our method achieves a 31X speedup compared to state-of-the-art models such as VideoCrafter2, while also attaining the highest overall score on VBench. Moreover, our method supports the generation of videos with up to 26 frames, addressing the limitations of existing U-Net-based methods in long video generation. Notably, the entire training process requires only four GPUs, yet delivers performance competitive with existing leading methods. Hummingbird presents a practical and efficient solution for T2V generation, combining high performance, scalability, and flexibility for real-world applications.
>
---
#### [replaced 012] StateSpaceDiffuser: Bringing Long Context to Diffusion World Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22246v3](http://arxiv.org/pdf/2505.22246v3)**

> **作者:** Nedko Savov; Naser Kazemi; Deheng Zhang; Danda Pani Paudel; Xi Wang; Luc Van Gool
>
> **摘要:** World models have recently gained prominence for action-conditioned visual prediction in complex environments. However, relying on only a few recent observations causes them to lose long-term context. Consequently, within a few steps, the generated scenes drift from what was previously observed, undermining temporal coherence. This limitation, common in state-of-the-art world models, which are diffusion-based, stems from the lack of a lasting environment state. To address this problem, we introduce StateSpaceDiffuser, where a diffusion model is enabled to perform long-context tasks by integrating features from a state-space model, representing the entire interaction history. This design restores long-term memory while preserving the high-fidelity synthesis of diffusion models. To rigorously measure temporal consistency, we develop an evaluation protocol that probes a model's ability to reinstantiate seen content in extended rollouts. Comprehensive experiments show that StateSpaceDiffuser significantly outperforms a strong diffusion-only baseline, maintaining a coherent visual context for an order of magnitude more steps. It delivers consistent views in both a 2D maze navigation and a complex 3D environment. These results establish that bringing state-space representations into diffusion models is highly effective in demonstrating both visual details and long-term memory. Project page: https://insait-institute.github.io/StateSpaceDiffuser/.
>
---
#### [replaced 013] Integrating Video and Text: A Balanced Approach to Multimodal Summary Generation and Evaluation
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06594v2](http://arxiv.org/pdf/2505.06594v2)**

> **作者:** Galann Pennec; Zhengyuan Liu; Nicholas Asher; Philippe Muller; Nancy F. Chen
>
> **摘要:** Vision-Language Models (VLMs) often struggle to balance visual and textual information when summarizing complex multimodal inputs, such as entire TV show episodes. In this paper, we propose a zero-shot video-to-text summarization approach that builds its own screenplay representation of an episode, effectively integrating key video moments, dialogue, and character information into a unified document. Unlike previous approaches, we simultaneously generate screenplays and name the characters in zero-shot, using only the audio, video, and transcripts as input. Additionally, we highlight that existing summarization metrics can fail to assess the multimodal content in summaries. To address this, we introduce MFactSum, a multimodal metric that evaluates summaries with respect to both vision and text modalities. Using MFactSum, we evaluate our screenplay summaries on the SummScreen3D dataset, demonstrating superiority against state-of-the-art VLMs such as Gemini 1.5 by generating summaries containing 20% more relevant visual information while requiring 75% less of the video as input.
>
---
#### [replaced 014] Intelligent Software System for Low-Cost, Brightfield Segmentation: Algorithmic Implementation for Cytometric Auto-Analysis
- **分类: q-bio.QM; cs.CV; eess.IV; q-bio.CB**

- **链接: [http://arxiv.org/pdf/2509.11354v5](http://arxiv.org/pdf/2509.11354v5)**

> **作者:** Surajit Das; Pavel Zun
>
> **摘要:** Bright-field microscopy, a cost-effective solution for live-cell culture, is often the only resource available, along with standard CPUs, for many low-budget labs. The inherent challenges of bright-field images -- their noisiness, low contrast, and dynamic morphology -- coupled with a lack of GPU resources and complex software interfaces, hinder the desired research output. This article presents a novel microscopy image analysis framework designed for low-budget labs equipped with a standard CPU desktop. The Python-based program enables cytometric analysis of live, unstained cells in culture through an advanced computer vision and machine learning pipeline. Crucially, the framework operates on label-free data, requiring no manually annotated training data or training phase. It is accessible via a user-friendly, cross-platform GUI that requires no programming skills, while also providing a scripting interface for programmatic control and integration by developers. The end-to-end workflow performs semantic and instance segmentation, feature extraction, analysis, evaluation, and automated report generation. Its modular architecture supports easy maintenance and flexible integration while supporting both single-image and batch processing. Validated on several unstained cell types from the public dataset of livecells, the framework demonstrates superior accuracy and reproducibility compared to contemporary tools like Cellpose and StarDist. Its competitive segmentation speed on a CPU-based platform highlights its significant potential for basic research and clinical applications -- particularly in cell transplantation for personalised medicine and muscle regeneration therapies. The access to the application is available for reproducibility
>
---
#### [replaced 015] DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07464v4](http://arxiv.org/pdf/2506.07464v4)**

> **作者:** Jinyoung Park; Jeehye Na; Jinyoung Kim; Hyunwoo J. Kim
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recent works have demonstrated the effectiveness of reinforcement learning (RL)-based post-training for enhancing the reasoning capabilities of large language models (LLMs). In particular, Group Relative Policy Optimization (GRPO) has shown impressive success using a PPO-style reinforcement algorithm with group-normalized rewards. However, the effectiveness of GRPO in Video Large Language Models (VideoLLMs) has still been less studyed. In this paper, we explore GRPO and identify two problems that deteriorate the effective learning: (1) reliance on safeguards, and (2) vanishing advantage. To mitigate these challenges, we propose DeepVideo-R1, a video large language model trained with Reg-GRPO (Regressive GRPO) and difficulty-aware data augmentation. Reg-GRPO reformulates the GRPO loss function into a regression task that directly predicts the advantage in GRPO, eliminating the need for safeguards such as the clipping and min functions. It directly aligns the model with advantages, providing guidance to prefer better ones. The difficulty-aware data augmentation strategy augments input prompts/videos to locate the difficulty of samples at solvable difficulty levels, enabling diverse reward signals. Our experimental results show that our approach significantly improves video reasoning performance across multiple benchmarks.
>
---
#### [replaced 016] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13055v4](http://arxiv.org/pdf/2504.13055v4)**

> **作者:** Xiangyan Liu; Jinjie Ni; Zijian Wu; Chao Du; Longxu Dou; Haonan Wang; Tianyu Pang; Michael Qizhe Shieh
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recent advances in reinforcement learning (RL) have strengthened the reasoning capabilities of vision-language models (VLMs). However, enhancing policy exploration to better scale test-time compute remains largely underexplored. In addition, VLMs continue to struggle with imperfect visual perception, which in turn affects the subsequent reasoning process. We introduce NoisyRollout, a simple yet effective data augmentation method that addresses these issues by mixing training trajectories from both clean and moderately distorted images. This approach injects perceptual diversity, encouraging better policy exploration and leading to more robust reasoning. A noise annealing schedule gradually reduces distortion strength, aiding exploration early in training while ensuring later stability. Crucially, our method is easy-to-adopt--requiring no additional training cost and no modifications to the RL objective. Extensive experiments on 2 distinct training datasets demonstrate that NoisyRollout achieves state-of-the-art performance among open-source RL-tuned models across 5 out-of-domain reasoning and perception benchmarks. Furthermore, we validate the effectiveness of NoisyRollout across model sizes (7B and 32B), data scales (from 1K to 6K) and image augmentation types (Gaussion noise and rotation), highlighting its generalizability and scalability.
>
---
#### [replaced 017] Conformal Object Detection by Sequential Risk Control
- **分类: stat.ML; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.24038v2](http://arxiv.org/pdf/2505.24038v2)**

> **作者:** Léo andéol; Luca Mossina; Adrien Mazoyer; Sébastien Gerchinovitz
>
> **备注:** 29 pages, 12 figures
>
> **摘要:** Recent advances in object detectors have led to their adoption for industrial uses. However, their deployment in safety-critical applications is hindered by the inherent lack of reliability of neural networks and the complex structure of object detection models. To address these challenges, we turn to Conformal Prediction, a post-hoc predictive uncertainty quantification procedure with statistical guarantees that are valid for any dataset size, without requiring prior knowledge on the model or data distribution. Our contribution is manifold. First, we formally define the problem of Conformal Object Detection (COD). We introduce a novel method, Sequential Conformal Risk Control (SeqCRC), that extends the statistical guarantees of Conformal Risk Control to two sequential tasks with two parameters, as required in the COD setting. Then, we present old and new loss functions and prediction sets suited to applying SeqCRC to different cases and certification requirements. Finally, we present a conformal toolkit for replication and further exploration of our method. Using this toolkit, we perform extensive experiments that validate our approach and emphasize trade-offs and other practical consequences.
>
---
#### [replaced 018] FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.21657v2](http://arxiv.org/pdf/2509.21657v2)**

> **作者:** Yixiang Dai; Fan Jiang; Chiyu Wang; Mu Xu; Yonggang Qi
>
> **摘要:** High-quality 3D world models are pivotal for embodied intelligence and Artificial General Intelligence (AGI), underpinning applications such as AR/VR content creation and robotic navigation. Despite the established strong imaginative priors, current video foundation models lack explicit 3D grounding capabilities, thus being limited in both spatial consistency and their utility for downstream 3D reasoning tasks. In this work, we present FantasyWorld, a geometry-enhanced framework that augments frozen video foundation models with a trainable geometric branch, enabling joint modeling of video latents and an implicit 3D field in a single forward pass. Our approach introduces cross-branch supervision, where geometry cues guide video generation and video priors regularize 3D prediction, thus yielding consistent and generalizable 3D-aware video representations. Notably, the resulting latents from the geometric branch can potentially serve as versatile representations for downstream 3D tasks such as novel view synthesis and navigation, without requiring per-scene optimization or fine-tuning. Extensive experiments show that FantasyWorld effectively bridges video imagination and 3D perception, outperforming recent geometry-consistent baselines in multi-view coherence and style consistency. Ablation studies further confirm that these gains stem from the unified backbone and cross-branch information exchange.
>
---
#### [replaced 019] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.16553v5](http://arxiv.org/pdf/2412.16553v5)**

> **作者:** Yunshan Zhong; Yuyao Zhou; Yuxin Zhang; Wanchen Sui; Shen Li; Yong Li; Fei Chao; Rongrong Ji
>
> **备注:** ICCV2025
>
> **摘要:** Data-free quantization (DFQ) enables model quantization without accessing real data, addressing concerns regarding data security and privacy. With the growing adoption of Vision Transformers (ViTs), DFQ for ViTs has garnered significant attention. However, existing DFQ methods exhibit two limitations: (1) semantic distortion, where the semantics of synthetic images deviate substantially from those of real images, and (2) semantic inadequacy, where synthetic images contain extensive regions with limited content and oversimplified textures, leading to suboptimal quantization performance. To address these limitations, we propose SARDFQ, a novel Semantics Alignment and Reinforcement Data-Free Quantization method for ViTs. To address semantic distortion, SARDFQ incorporates Attention Priors Alignment (APA), which optimizes synthetic images to follow randomly generated structure attention priors. To mitigate semantic inadequacy, SARDFQ introduces Multi-Semantic Reinforcement (MSR), leveraging localized patch optimization to enhance semantic richness across synthetic images. Furthermore, SARDFQ employs Soft-Label Learning (SL), wherein multiple semantic targets are adapted to facilitate the learning of multi-semantic images augmented by MSR. Extensive experiments demonstrate the effectiveness of SARDFQ, significantly surpassing existing methods. For example, SARDFQ improves top-1 accuracy on ImageNet by 15.52% for W4A4 ViT-B. The code is at https://github.com/zysxmu/SARDFQ.
>
---
#### [replaced 020] SafePLUG: Empowering Multimodal LLMs with Pixel-Level Insight and Temporal Grounding for Traffic Accident Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.06763v3](http://arxiv.org/pdf/2508.06763v3)**

> **作者:** Zihao Sheng; Zilin Huang; Yansong Qu; Jiancong Chen; Yuhao Luo; Yen-Jung Chen; Yue Leng; Sikai Chen
>
> **备注:** The code, dataset, and model checkpoints will be made publicly available at: https://zihaosheng.github.io/SafePLUG
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable progress across a range of vision-language tasks and demonstrate strong potential for traffic accident understanding. However, existing MLLMs in this domain primarily focus on coarse-grained image-level or video-level comprehension and often struggle to handle fine-grained visual details or localized scene components, limiting their applicability in complex accident scenarios. To address these limitations, we propose SafePLUG, a novel framework that empowers MLLMs with both Pixel-Level Understanding and temporal Grounding for comprehensive traffic accident analysis. SafePLUG supports both arbitrary-shaped visual prompts for region-aware question answering and pixel-level segmentation based on language instructions, while also enabling the recognition of temporally anchored events in traffic accident scenarios. To advance the development of MLLMs for traffic accident understanding, we curate a new dataset containing multimodal question-answer pairs centered on diverse accident scenarios, with detailed pixel-level annotations and temporal event boundaries. Experimental results show that SafePLUG achieves strong performance on multiple tasks, including region-based question answering, pixel-level segmentation, temporal event localization, and accident event understanding. These capabilities lay a foundation for fine-grained understanding of complex traffic scenes, with the potential to improve driving safety and enhance situational awareness in smart transportation systems. The code, dataset, and model checkpoints will be made publicly available at: https://zihaosheng.github.io/SafePLUG
>
---
#### [replaced 021] DINO-YOLO: Self-Supervised Pre-training for Data-Efficient Object Detection in Civil Engineering Applications
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.25140v2](http://arxiv.org/pdf/2510.25140v2)**

> **作者:** Malaisree P; Youwai S; Kitkobsin T; Janrungautai S; Amorndechaphon D; Rojanavasu P
>
> **摘要:** Object detection in civil engineering applications is constrained by limited annotated data in specialized domains. We introduce DINO-YOLO, a hybrid architecture combining YOLOv12 with DINOv3 self-supervised vision transformers for data-efficient detection. DINOv3 features are strategically integrated at two locations: input preprocessing (P0) and mid-backbone enhancement (P3). Experimental validation demonstrates substantial improvements: Tunnel Segment Crack detection (648 images) achieves 12.4% improvement, Construction PPE (1K images) gains 13.7%, and KITTI (7K images) shows 88.6% improvement, while maintaining real-time inference (30-47 FPS). Systematic ablation across five YOLO scales and nine DINOv3 variants reveals that Medium-scale architectures achieve optimal performance with DualP0P3 integration (55.77% mAP@0.5), while Small-scale requires Triple Integration (53.63%). The 2-4x inference overhead (21-33ms versus 8-16ms baseline) remains acceptable for field deployment on NVIDIA RTX 5090. DINO-YOLO establishes state-of-the-art performance for civil engineering datasets (<10K images) while preserving computational efficiency, providing practical solutions for construction safety monitoring and infrastructure inspection in data-constrained environments.
>
---
#### [replaced 022] SRAGAN: Saliency Regularized and Attended Generative Adversarial Network for Chinese Ink-wash Painting Style Transfer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.15743v3](http://arxiv.org/pdf/2404.15743v3)**

> **作者:** Xiang Gao; Yuqi Zhang
>
> **备注:** Pattern Recognition, Volume 162, June 2025, 111344
>
> **摘要:** Recent style transfer problems are still largely dominated by Generative Adversarial Network (GAN) from the perspective of cross-domain image-to-image (I2I) translation, where the pivotal issue is to learn and transfer target-domain style patterns onto source-domain content images. This paper handles the problem of translating real pictures into traditional Chinese ink-wash paintings, i.e., Chinese ink-wash painting style transfer. Though a wide range of I2I models tackle this problem, a notable challenge is that the content details of the source image could be easily erased or corrupted due to the transfer of ink-wash style elements. To remedy this issue, we propose to incorporate saliency detection into the unpaired I2I framework to regularize image content, where the detected saliency map is utilized from two aspects: (\romannumeral1) we propose saliency IOU (SIOU) loss to explicitly regularize object content structure by enforcing saliency consistency before and after image stylization; (\romannumeral2) we propose saliency adaptive normalization (SANorm) which implicitly enhances object structure integrity of the generated paintings by dynamically injecting image saliency information into the generator to guide stylization process. Besides, we also propose saliency attended discriminator which harnesses image saliency information to focus generative adversarial attention onto the drawn objects, contributing to generating more vivid and delicate brush strokes and ink-wash textures. Extensive qualitative and quantitative experiments demonstrate superiority of our approach over related advanced image stylization methods in both GAN and diffusion model paradigms.
>
---
#### [replaced 023] Neural Posterior Estimation for Cataloging Astronomical Images from the Legacy Survey of Space and Time
- **分类: astro-ph.IM; cs.CV; stat.AP; 85A35 (Primary), 62F15 (Secondary)**

- **链接: [http://arxiv.org/pdf/2510.15315v2](http://arxiv.org/pdf/2510.15315v2)**

> **作者:** Yicun Duan; Xinyue Li; Camille Avestruz; Jeffrey Regier
>
> **摘要:** The Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) will commence full-scale operations in 2026, yielding an unprecedented volume of astronomical images. Constructing an astronomical catalog, a table of imaged stars, galaxies, and their properties, is a fundamental step in most scientific workflows based on astronomical image data. Traditional deterministic cataloging methods lack statistical coherence as cataloging is an ill-posed problem, while existing probabilistic approaches suffer from computational inefficiency, inaccuracy, or the inability to perform inference with multiband coadded images, the primary output format for LSST images. In this article, we explore a recently developed Bayesian inference method called neural posterior estimation (NPE) as an approach to cataloging. NPE leverages deep learning to achieve both computational efficiency and high accuracy. When evaluated on the DC2 Simulated Sky Survey -- a highly realistic synthetic dataset designed to mimic LSST data -- NPE systematically outperforms the standard LSST pipeline in light source detection, flux measurement, star/galaxy classification, and galaxy shape measurement. Additionally, NPE provides well-calibrated posterior approximations. These promising results, obtained using simulated data, illustrate the potential of NPE in the absence of model misspecification. Although some degree of model misspecification is inevitable in the application of NPE to real LSST images, there are a variety of strategies to mitigate its effects.
>
---
#### [replaced 024] Augmented Reality-based Guidance with Deformable Registration in Head and Neck Tumor Resection
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08802v2](http://arxiv.org/pdf/2503.08802v2)**

> **作者:** Qingyun Yang; Fangjie Li; Jiayi Xu; Zixuan Liu; Sindhura Sridhar; Whitney Jin; Jennifer Du; Jon Heiselman; Michael Miga; Michael Topf; Jie Ying Wu
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** Head and neck squamous cell carcinoma (HNSCC) has one of the highest rates of recurrence cases among solid malignancies. Recurrence rates can be reduced by improving positive margins localization. Frozen section analysis (FSA) of resected specimens is the gold standard for intraoperative margin assessment. However, because of the complex 3D anatomy and the significant shrinkage of resected specimens, accurate margin relocation from specimen back onto the resection site based on FSA results remains challenging. We propose a novel deformable registration framework that uses both the pre-resection upper surface and the post-resection site of the specimen to incorporate thickness information into the registration process. The proposed method significantly improves target registration error (TRE), demonstrating enhanced adaptability to thicker specimens. In tongue specimens, the proposed framework improved TRE by up to 33% as compared to prior deformable registration. Notably, tongue specimens exhibit complex 3D anatomies and hold the highest clinical significance compared to other head and neck specimens from the buccal and skin. We analyzed distinct deformation behaviors in different specimens, highlighting the need for tailored deformation strategies. To further aid intraoperative visualization, we also integrated this framework with an augmented reality-based auto-alignment system. The combined system can accurately and automatically overlay the deformed 3D specimen mesh with positive margin annotation onto the resection site. With a pilot study of the AR guided framework involving two surgeons, the integrated system improved the surgeons' average target relocation error from 9.8 cm to 4.8 cm.
>
---
#### [replaced 025] BALR-SAM: Boundary-Aware Low-Rank Adaptation of SAM for Resource-Efficient Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.24204v2](http://arxiv.org/pdf/2509.24204v2)**

> **作者:** Zelin Liu; Sicheng Dong; Bocheng Li; Yixuan Yang; Jiacheng Ruan; Chenxu Zhou; Suncheng Xiang
>
> **摘要:** Vision foundation models like the Segment Anything Model (SAM), pretrained on large-scale natural image datasets, often struggle in medical image segmentation due to a lack of domain-specific adaptation. In clinical practice, fine-tuning such models efficiently for medical downstream tasks with minimal resource demands, while maintaining strong performance, is challenging. To address these issues, we propose BALR-SAM, a boundary-aware low-rank adaptation framework that enhances SAM for medical imaging. It combines three tailored components: (1) a Complementary Detail Enhancement Network (CDEN) using depthwise separable convolutions and multi-scale fusion to capture boundary-sensitive features essential for accurate segmentation; (2) low-rank adapters integrated into SAM's Vision Transformer blocks to optimize feature representation and attention for medical contexts, while simultaneously significantly reducing the parameter space; and (3) a low-rank tensor attention mechanism in the mask decoder, cutting memory usage by 75% and boosting inference speed. Experiments on standard medical segmentation datasets show that BALR-SAM, without requiring prompts, outperforms several state-of-the-art (SOTA) methods, including fully fine-tuned MedSAM, while updating just 1.8% (11.7M) of its parameters.
>
---
#### [replaced 026] IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22706v3](http://arxiv.org/pdf/2510.22706v3)**

> **作者:** Hao Li; Zhengyu Zou; Fangfu Liu; Xuanyang Zhang; Fangzhou Hong; Yukang Cao; Yushi Lan; Manyuan Zhang; Gang Yu; Dingwen Zhang; Ziwei Liu
>
> **备注:** https://github.com/lifuguan/IGGT_official
>
> **摘要:** Humans naturally perceive the geometric structure and semantic content of a 3D world as intertwined dimensions, enabling coherent and accurate understanding of complex scenes. However, most prior approaches prioritize training large geometry models for low-level 3D reconstruction and treat high-level spatial understanding in isolation, overlooking the crucial interplay between these two fundamental aspects of 3D-scene analysis, thereby limiting generalization and leading to poor performance in downstream 3D understanding tasks. Recent attempts have mitigated this issue by simply aligning 3D models with specific language models, thus restricting perception to the aligned model's capacity and limiting adaptability to downstream tasks. In this paper, we propose InstanceGrounded Geometry Transformer (IGGT), an end-to-end large unified transformer to unify the knowledge for both spatial reconstruction and instance-level contextual understanding. Specifically, we design a 3D-Consistent Contrastive Learning strategy that guides IGGT to encode a unified representation with geometric structures and instance-grounded clustering through only 2D visual inputs. This representation supports consistent lifting of 2D visual inputs into a coherent 3D scene with explicitly distinct object instances. To facilitate this task, we further construct InsScene-15K, a large-scale dataset with high-quality RGB images, poses, depth maps, and 3D-consistent instance-level mask annotations with a novel data curation pipeline.
>
---
#### [replaced 027] Manifold Learning for Hyperspectral Images
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.15016v2](http://arxiv.org/pdf/2503.15016v2)**

> **作者:** Fethi Harkat; Guillaume Gey; Valérie Perrier; Kévin Polisano; Tiphaine Deuberet
>
> **摘要:** Traditional feature extraction and projection techniques, such as Principal Component Analysis, struggle to adequately represent X-Ray Transmission (XRT) Multi-Energy (ME) images, limiting the performance of neural networks in decision-making processes. To address this issue, we propose a method that approximates the dataset topology by constructing adjacency graphs using the Uniform Manifold Approximation and Projection. This approach captures nonlinear correlations within the data, significantly improving the performance of machine learning algorithms, particularly in processing Hyperspectral Images (HSI) from X-ray transmission spectroscopy. This technique not only preserves the global structure of the data but also enhances feature separability, leading to more accurate and robust classification results.
>
---
#### [replaced 028] Face Spoofing Detection using Deep Learning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.19223v2](http://arxiv.org/pdf/2503.19223v2)**

> **作者:** Najeebullah; Maaz Salman; Zar Nawab Khan Swati
>
> **备注:** The author's school has a conflict of interest regarding the submission of this article prior to his graduation thesis submission
>
> **摘要:** Digital image spoofing has emerged as a significant security threat in biometric authentication systems, particularly those relying on facial recognition. This study evaluates the performance of three vision based models, MobileNetV2, ResNET50, and Vision Transformer, ViT, for spoof detection in image classification, utilizing a dataset of 150,986 images divided into training , 140,002, testing, 10,984, and validation ,39,574, sets. Spoof detection is critical for enhancing the security of image recognition systems, and this research compares the models effectiveness through accuracy, precision, recall, and F1 score metrics. Results reveal that MobileNetV2 outperforms other architectures on the test dataset, achieving an accuracy of 91.59%, precision of 91.72%, recall of 91.59%, and F1 score of 91.58%, compared to ViT 86.54%, 88.28%, 86.54%, and 86.39%, respectively. On the validation dataset, MobileNetV2, and ViT excel, with MobileNetV2 slightly ahead at 97.17% accuracy versus ViT 96.36%. MobileNetV2 demonstrates faster convergence during training and superior generalization to unseen data, despite both models showing signs of overfitting. These findings highlight MobileNetV2 balanced performance and robustness, making it the preferred choice for spoof detection applications where reliability on new data is essential. The study underscores the importance of model selection in security sensitive contexts and suggests MobileNetV2 as a practical solution for real world deployment.
>
---
#### [replaced 029] ε-Seg: Sparsely Supervised Semantic Segmentation of Microscopy Data
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.18637v2](http://arxiv.org/pdf/2510.18637v2)**

> **作者:** Sheida Rahnamai Kordasiabi; Damian Dalle Nogare; Florian Jug
>
> **备注:** 10 pages main text, 17 pages total
>
> **摘要:** Semantic segmentation of electron microscopy (EM) images of biological samples remains a challenge in the life sciences. EM data captures details of biological structures, sometimes with such complexity that even human observers can find it overwhelming. We introduce {\epsilon}-Seg, a method based on hierarchical variational autoencoders (HVAEs), employing center-region masking, sparse label contrastive learning (CL), a Gaussian mixture model (GMM) prior, and clustering-free label prediction. Center-region masking and the inpainting loss encourage the model to learn robust and representative embeddings to distinguish the desired classes, even if training labels are sparse (0.05% of the total image data or less). For optimal performance, we employ CL and a GMM prior to shape the latent space of the HVAE such that encoded input patches tend to cluster wrt. the semantic classes we wish to distinguish. Finally, instead of clustering latent embeddings for semantic segmentation, we propose a MLP semantic segmentation head to directly predict class labels from latent embeddings. We show empirical results of {\epsilon}-Seg and baseline methods on 2 dense EM datasets of biological tissues and demonstrate the applicability of our method also on fluorescence microscopy data. Our results show that {\epsilon}-Seg is capable of achieving competitive sparsely-supervised segmentation results on complex biological image data, even if only limited amounts of training labels are available.
>
---
#### [replaced 030] $\mathtt{M^3VIR}$: A Large-Scale Multi-Modality Multi-View Synthesized Benchmark Dataset for Image Restoration and Content Creation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.16873v2](http://arxiv.org/pdf/2509.16873v2)**

> **作者:** Yuanzhi Li; Lebin Zhou; Nam Ling; Zhenghao Chen; Wei Wang; Wei Jiang
>
> **摘要:** The gaming and entertainment industry is rapidly evolving, driven by immersive experiences and the integration of generative AI (GAI) technologies. Training such models effectively requires large-scale datasets that capture the diversity and context of gaming environments. However, existing datasets are often limited to specific domains or rely on artificial degradations, which do not accurately capture the unique characteristics of gaming content. Moreover, benchmarks for controllable video generation remain absent. To address these limitations, we introduce $\mathtt{M^3VIR}$, a large-scale, multi-modal, multi-view dataset specifically designed to overcome the shortcomings of current resources. Unlike existing datasets, $\mathtt{M^3VIR}$ provides diverse, high-fidelity gaming content rendered with Unreal Engine 5, offering authentic ground-truth LR-HR paired and multi-view frames across 80 scenes in 8 categories. It includes $\mathtt{M^3VIR\_MR}$ for super-resolution (SR), novel view synthesis (NVS), and combined NVS+SR tasks, and $\mathtt{M^3VIR\_{MS}}$, the first multi-style, object-level ground-truth set enabling research on controlled video generation. Additionally, we benchmark several state-of-the-art SR and NVS methods to establish performance baselines. While no existing approaches directly handle controlled video generation, $\mathtt{M^3VIR}$ provides a benchmark for advancing this area. By releasing the dataset, we aim to facilitate research in AI-powered restoration, compression, and controllable content generation for next-generation cloud gaming and entertainment.
>
---
#### [replaced 031] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.25263v2](http://arxiv.org/pdf/2510.25263v2)**

> **作者:** Yang Miao; Jan-Nico Zaech; Xi Wang; Fabien Despinoy; Danda Pani Paudel; Luc Van Gool
>
> **备注:** 10 pages, 5 figures, 14 tables, Neurips 2025
>
> **摘要:** We propose LangHOPS, the first Multimodal Large Language Model (MLLM) based framework for open-vocabulary object-part instance segmentation. Given an image, LangHOPS can jointly detect and segment hierarchical object and part instances from open-vocabulary candidate categories. Unlike prior approaches that rely on heuristic or learnable visual grouping, our approach grounds object-part hierarchies in language space. It integrates the MLLM into the object-part parsing pipeline to leverage its rich knowledge and reasoning capabilities, and link multi-granularity concepts within the hierarchies. We evaluate LangHOPS across multiple challenging scenarios, including in-domain and cross-dataset object-part instance segmentation, and zero-shot semantic segmentation. LangHOPS achieves state-of-the-art results, surpassing previous methods by 5.5% Average Precision (AP) (in-domain) and 4.8% (cross-dataset) on the PartImageNet dataset and by 2.5% mIOU on unseen object parts in ADE20K (zero-shot). Ablation studies further validate the effectiveness of the language-grounded hierarchy and MLLM driven part query refinement strategy. The code will be released here.
>
---
#### [replaced 032] MixedGaussianAvatar: Realistically and Geometrically Accurate Head Avatar via Mixed 2D-3D Gaussians
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.04955v3](http://arxiv.org/pdf/2412.04955v3)**

> **作者:** Peng Chen; Xiaobao Wei; Qingpo Wuwu; Xinyi Wang; Xingyu Xiao; Ming Lu
>
> **摘要:** Reconstructing high-fidelity 3D head avatars is crucial in various applications such as virtual reality. The pioneering methods reconstruct realistic head avatars with Neural Radiance Fields (NeRF), which have been limited by training and rendering speed. Recent methods based on 3D Gaussian Splatting (3DGS) significantly improve the efficiency of training and rendering. However, the surface inconsistency of 3DGS results in subpar geometric accuracy; later, 2DGS uses 2D surfels to enhance geometric accuracy at the expense of rendering fidelity. To leverage the benefits of both 2DGS and 3DGS, we propose a novel method named MixedGaussianAvatar for realistically and geometrically accurate head avatar reconstruction. Our main idea is to utilize 2D Gaussians to reconstruct the surface of the 3D head, ensuring geometric accuracy. We attach the 2D Gaussians to the triangular mesh of the FLAME model and connect additional 3D Gaussians to those 2D Gaussians where the rendering quality of 2DGS is inadequate, creating a mixed 2D-3D Gaussian representation. These 2D-3D Gaussians can then be animated using FLAME parameters. We further introduce a progressive training strategy that first trains the 2D Gaussians and then fine-tunes the mixed 2D-3D Gaussians. We use a unified mixed Gaussian representation to integrate the two modalities of 2D image and 3D mesh. Furthermore, the comprehensive experiments demonstrate the superiority of MixedGaussianAvatar. The code will be released.
>
---
#### [replaced 033] MMEdge: Accelerating On-device Multimodal Inference via Pipelined Sensing and Encoding
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.25327v3](http://arxiv.org/pdf/2510.25327v3)**

> **作者:** Runxi Huang; Mingxuan Yu; Mingyu Tsoi; Xiaomin Ouyang
>
> **备注:** Code available at: https://github.com/HKUST-MINSys-Lab/MMEdge. Accepted by SenSys 2026
>
> **摘要:** Real-time multimodal inference on resource-constrained edge devices is essential for applications such as autonomous driving, human-computer interaction, and mobile health. However, prior work often overlooks the tight coupling between sensing dynamics and model execution, as well as the complex inter-modality dependencies. In this paper, we propose MMEdge, an new on-device multi-modal inference framework based on pipelined sensing and encoding. Instead of waiting for complete sensor inputs, MMEdge decomposes the entire inference process into a sequence of fine-grained sensing and encoding units, allowing computation to proceed incrementally as data arrive. MMEdge also introduces a lightweight but effective temporal aggregation module that captures rich temporal dynamics across different pipelined units to maintain accuracy performance. Such pipelined design also opens up opportunities for fine-grained cross-modal optimization and early decision-making during inference. To further enhance system performance under resource variability and input data complexity, MMEdge incorporates an adaptive multimodal configuration optimizer that dynamically selects optimal sensing and model configurations for each modality under latency constraints, and a cross-modal speculative skipping mechanism that bypasses future units of slower modalities when early predictions reach sufficient confidence. We evaluate MMEdge using two public multimodal datasets and deploy it on a real-world unmanned aerial vehicle (UAV)-based multimodal testbed. The results show that MMEdge significantly reduces end-to-end latency while maintaining high task accuracy across various system and data dynamics.
>
---
#### [replaced 034] Rethinking Metrics and Benchmarks of Video Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19022v2](http://arxiv.org/pdf/2505.19022v2)**

> **作者:** Zihao Liu; Xiaoyu Wu; Wenna Li; Linlin Yang; Shengjin Wang
>
> **摘要:** Video Anomaly Detection (VAD), which aims to detect anomalies that deviate from expectation, has attracted increasing attention in recent years. Existing advancements in VAD primarily focus on model architectures and training strategies, while devoting insufficient attention to evaluation metrics and benchmarks. In this paper, we rethink VAD evaluation methods through comprehensive analyses, revealing three critical limitations in current practices: 1) existing metrics are significantly influenced by single annotation bias; 2) current metrics fail to reward early detection of anomalies; 3) available benchmarks lack the capability to evaluate scene overfitting of fully/weakly-supervised algorithms. To address these limitations, we propose three novel evaluation methods: first, we establish probabilistic AUC/AP (Prob-AUC/AP) metrics utlizing multi-round annotations to mitigate single annotation bias; second, we develop a Latency-aware Average Precision (LaAP) metric that rewards early and accurate anomaly detection; and finally, we introduce two hard normal benchmarks (UCF-HN, MSAD-HN) with videos specifically designed to evaluate scene overfitting. We report performance comparisons of ten state-of-the-art VAD approaches using our proposed evaluation methods, providing novel perspectives for future VAD model development. We release our data and code in https://github.com/Kamino666/RethinkingVAD.
>
---
#### [replaced 035] PROFIT: A Specialized Optimizer for Deep Fine Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01930v3](http://arxiv.org/pdf/2412.01930v3)**

> **作者:** Anirudh S Chakravarthy; Shuai Kyle Zheng; Xin Huang; Sachithra Hemachandra; Xiao Zhang; Yuning Chai; Zhao Chen
>
> **备注:** technical report, 23 pages, NeurIPS 2025 poster
>
> **摘要:** The fine-tuning of pre-trained models has become ubiquitous in generative AI, computer vision, and robotics. Although much attention has been paid to improving the efficiency of fine-tuning model, there has been less scholarship around fine-tuning specifically for improved model performance. To remedy this gap, we present PROFIT, one of the first optimizers designed to incrementally fine-tune converged models on new tasks and/or datasets. Unlike traditional optimizers such as SGD or Adam, which make minimal assumptions due to random initializations, PROFIT takes the properties of a converged model into account explicitly to regularize the optimization process. Employing a temporal gradient-orthogonalization process, PROFIT outperforms fine-tuning methods in various tasks, from image classification to multimodal language model training to large-scale motion prediction. Moreover, PROFIT is encapsulated as a modular optimizer, which makes it easy to integrate directly into any training pipeline with minimal engineering effort.
>
---
#### [replaced 036] SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training
- **分类: cs.LG; cs.AI; cs.AR; cs.CV; cs.PF**

- **链接: [http://arxiv.org/pdf/2505.11594v2](http://arxiv.org/pdf/2505.11594v2)**

> **作者:** Jintao Zhang; Jia Wei; Pengle Zhang; Xiaoming Xu; Haofeng Huang; Haoxu Wang; Kai Jiang; Jun Zhu; Jianfei Chen
>
> **摘要:** The efficiency of attention is important due to its quadratic time complexity. We enhance the efficiency of attention through two key contributions: First, we leverage the new FP4 Tensor Cores in Blackwell GPUs to accelerate attention computation. Our implementation achieves 1038 TOPS on RTX5090, which is a 5x speedup over the fastest FlashAttention on RTX5090. Experiments show that our FP4 attention can accelerate inference of various models in a plug-and-play way. Second, we pioneer low-bit attention to training tasks. Existing low-bit attention works like FlashAttention3 and SageAttention focus only on inference. However, the efficiency of training large models is also important. To explore whether low-bit attention can be effectively applied to training tasks, we design an accurate and efficient 8-bit attention for both forward and backward propagation. Experiments indicate that 8-bit attention achieves lossless performance in fine-tuning tasks but exhibits slower convergence in pretraining tasks. The code is available at https://github.com/thu-ml/SageAttention.
>
---
#### [replaced 037] Variational Visual Question Answering for Uncertainty-Aware Selective Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.09591v2](http://arxiv.org/pdf/2505.09591v2)**

> **作者:** Tobias Jan Wieczorek; Nathalie Daun; Mohammad Emtiyaz Khan; Marcus Rohrbach
>
> **备注:** under review at TMLR
>
> **摘要:** Despite remarkable progress in recent years, vision language models (VLMs) remain prone to overconfidence and hallucinations on tasks such as Visual Question Answering (VQA) and Visual Reasoning. Bayesian methods can potentially improve reliability by helping models selectively predict, that is, models respond only when they are sufficiently confident. Unfortunately, Bayesian methods are often assumed to be costly and ineffective for large models, and so far there exists little evidence to show otherwise, especially for multimodal applications. Here, we show the effectiveness and competitive edge of variational Bayes for selective prediction in VQA for the first time. We build on recent advances in variational methods for deep learning and propose an extension called "Variational VQA". This method improves calibration and yields significant gains for selective prediction on VQA and Visual Reasoning, particularly when the error tolerance is low ($\leq 1\%$). Often, just one posterior sample can yield more reliable answers than those obtained by models trained with AdamW. In addition, we propose a new risk-averse selector that outperforms standard sample averaging by considering the variance of predictions. Overall, we present compelling evidence that variational learning is a viable option to make large VLMs safer and more trustworthy.
>
---
#### [replaced 038] GASP: Gaussian Splatting for Physic-Based Simulations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.05819v3](http://arxiv.org/pdf/2409.05819v3)**

> **作者:** Piotr Borycki; Weronika Smolak; Joanna Waczyńska; Marcin Mazur; Sławomir Tadeja; Przemysław Spurek
>
> **摘要:** Physics simulation is paramount for modeling and utilizing 3D scenes in various real-world applications. However, integrating with state-of-the-art 3D scene rendering techniques such as Gaussian Splatting (GS) remains challenging. Existing models use additional meshing mechanisms, including triangle or tetrahedron meshing, marching cubes, or cage meshes. Alternatively, we can modify the physics-grounded Newtonian dynamics to align with 3D Gaussian components. Current models take the first-order approximation of a deformation map, which locally approximates the dynamics by linear transformations. In contrast, our GS for Physics-Based Simulations (GASP) pipeline uses parametrized flat Gaussian distributions. Consequently, the problem of modeling Gaussian components using the physics engine is reduced to working with 3D points. In our work, we present additional rules for manipulating Gaussians, demonstrating how to adapt the pipeline to incorporate meshes, control Gaussian sizes during simulations, and enhance simulation efficiency. This is achieved through the Gaussian grouping strategy, which implements hierarchical structuring and enables simulations to be performed exclusively on selected Gaussians. The resulting solution can be integrated into any physics engine that can be treated as a black box. As demonstrated in our studies, the proposed pipeline exhibits superior performance on a diverse range of benchmark datasets designed for 3D object rendering. The project webpage, which includes additional visualizations, can be found at https://waczjoan.github.io/GASP.
>
---
#### [replaced 039] Poisson Informed Retinex Network for Extreme Low-Light Image Enhancement
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04470v3](http://arxiv.org/pdf/2506.04470v3)**

> **作者:** Isha Rao; Ratul Chakraborty; Sanjay Ghosh
>
> **备注:** 10 pages, 5 figures and 1 table
>
> **摘要:** Low-light image denoising and enhancement are challenging, especially when traditional noise assumptions, such as Gaussian noise, do not hold in majority. In many real-world scenarios, such as low-light imaging, noise is signal-dependent and is better represented as Poisson noise. In this work, we address the problem of denoising images degraded by Poisson noise under extreme low-light conditions. We introduce a light-weight deep learning-based method that integrates Retinex based decomposition with Poisson denoising into a unified encoder-decoder network. The model simultaneously enhances illumination and suppresses noise by incorporating a Poisson denoising loss to address signal-dependent noise. Without prior requirement for reflectance and illumination, the network learns an effective decomposition process while ensuring consistent reflectance and smooth illumination without causing any form of color distortion. The experimental results demonstrate the effectiveness and practicality of the proposed low-light illumination enhancement method. Our method significantly improves visibility and brightness in low-light conditions, while preserving image structure and color constancy under ambient illumination.
>
---
#### [replaced 040] Mano Technical Report
- **分类: cs.MM; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.17336v3](http://arxiv.org/pdf/2509.17336v3)**

> **作者:** Tianyu Fu; Anyang Su; Chenxu Zhao; Hanning Wang; Minghui Wu; Zhe Yu; Fei Hu; Mingjia Shi; Wei Dong; Jiayao Wang; Yuyang Chen; Ruiyang Yu; Siran Peng; Menglin Li; Nan Huang; Haitian Wei; Jiawei Yu; Yi Xin; Xilin Zhao; Kai Gu; Ping Jiang; Sifan Zhou; Shuo Wang
>
> **摘要:** Graphical user interfaces (GUIs) are the primary medium for human-computer interaction, yet automating GUI interactions remains challenging due to the complexity of visual elements, dynamic environments, and the need for multi-step reasoning. Existing methods based on vision-language models (VLMs) often suffer from limited resolution, domain mismatch, and insufficient sequential decisionmaking capability. To address these issues, we propose Mano, a robust GUI agent built upon a multi-modal foundation model pre-trained on extensive web and computer system data. Our approach integrates a novel simulated environment for high-fidelity data generation, a three-stage training pipeline (supervised fine-tuning, offline reinforcement learning, and online reinforcement learning), and a verification module for error recovery. Mano demonstrates state-of-the-art performance on multiple GUI benchmarks, including Mind2Web and OSWorld, achieving significant improvements in success rate and operational accuracy. Our work provides new insights into the effective integration of reinforcement learning with VLMs for practical GUI agent deployment, highlighting the importance of domain-specific data, iterative training, and holistic reward design.
>
---
#### [replaced 041] Transformers in Medicine: Improving Vision-Language Alignment for Medical Image Captioning
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.25164v2](http://arxiv.org/pdf/2510.25164v2)**

> **作者:** Yogesh Thakku Suresh; Vishwajeet Shivaji Hogale; Luca-Alexandru Zamfira; Anandavardhana Hegde
>
> **备注:** This work is to appear in the Proceedings of MICAD 2025, the 6th International Conference on Medical Imaging and Computer-Aided Diagnosis
>
> **摘要:** We present a transformer-based multimodal framework for generating clinically relevant captions for MRI scans. Our system combines a DEiT-Small vision transformer as an image encoder, MediCareBERT for caption embedding, and a custom LSTM-based decoder. The architecture is designed to semantically align image and textual embeddings, using hybrid cosine-MSE loss and contrastive inference via vector similarity. We benchmark our method on the MultiCaRe dataset, comparing performance on filtered brain-only MRIs versus general MRI images against state-of-the-art medical image captioning methods including BLIP, R2GenGPT, and recent transformer-based approaches. Results show that focusing on domain-specific data improves caption accuracy and semantic alignment. Our work proposes a scalable, interpretable solution for automated medical image reporting.
>
---
#### [replaced 042] How Should One Evaluate Monocular Depth Estimation?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.19814v2](http://arxiv.org/pdf/2510.19814v2)**

> **作者:** Siyang Wu; Jack Nugent; Willow Yang; Jia Deng
>
> **摘要:** Monocular depth estimation is an important task with rapid progress, but how to evaluate it remains an open question, as evidenced by a lack of standardization in existing literature and a large selection of evaluation metrics whose trade-offs and behaviors are not well understood. This paper contributes a novel, quantitative analysis of existing metrics in terms of their sensitivity to various types of perturbations of ground truth, emphasizing comparison to human judgment. Our analysis reveals that existing metrics are severely under-sensitive to curvature perturbation such as making flat surfaces wavy. To remedy this, we introduce a new metric based on relative surface normals, along with new depth visualization tools and a principled method to create composite metrics with better human alignment. Code and data are available at: https://github.com/princeton-vl/evalmde.
>
---
#### [replaced 043] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.10691v3](http://arxiv.org/pdf/2510.10691v3)**

> **作者:** Xuankai Zhang; Junjin Xiao; Qing Zhang
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** This paper presents a unified framework that allows high-quality dynamic Gaussian Splatting from both defocused and motion-blurred monocular videos. Due to the significant difference between the formation processes of defocus blur and motion blur, existing methods are tailored for either one of them, lacking the ability to simultaneously deal with both of them. Although the two can be jointly modeled as blur kernel-based convolution, the inherent difficulty in estimating accurate blur kernels greatly limits the progress in this direction. In this work, we go a step further towards this direction. Particularly, we propose to estimate per-pixel reliable blur kernels using a blur prediction network that exploits blur-related scene and camera information and is subject to a blur-aware sparsity constraint. Besides, we introduce a dynamic Gaussian densification strategy to mitigate the lack of Gaussians for incomplete regions, and boost the performance of novel view synthesis by incorporating unseen view information to constrain scene optimization. Extensive experiments show that our method outperforms the state-of-the-art methods in generating photorealistic novel view synthesis from defocused and motion-blurred monocular videos. Our code is available at https://github.com/hhhddddddd/dydeblur.
>
---
#### [replaced 044] Scaling Diffusion Transformers Efficiently via $μ$P
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15270v3](http://arxiv.org/pdf/2505.15270v3)**

> **作者:** Chenyu Zheng; Xinyu Zhang; Rongzhen Wang; Wei Huang; Zhi Tian; Weilin Huang; Jun Zhu; Chongxuan Li
>
> **备注:** Accepted by NeurIPS 2025, 38 pages, 10 figures, 17 tables
>
> **摘要:** Diffusion Transformers have emerged as the foundation for vision generative models, but their scalability is limited by the high cost of hyperparameter (HP) tuning at large scales. Recently, Maximal Update Parametrization ($\mu$P) was proposed for vanilla Transformers, which enables stable HP transfer from small to large language models, and dramatically reduces tuning costs. However, it remains unclear whether $\mu$P of vanilla Transformers extends to diffusion Transformers, which differ architecturally and objectively. In this work, we generalize standard $\mu$P to diffusion Transformers and validate its effectiveness through large-scale experiments. First, we rigorously prove that $\mu$P of mainstream diffusion Transformers, including U-ViT, DiT, PixArt-$\alpha$, and MMDiT, aligns with that of the vanilla Transformer, enabling the direct application of existing $\mu$P methodologies. Leveraging this result, we systematically demonstrate that DiT-$\mu$P enjoys robust HP transferability. Notably, DiT-XL-2-$\mu$P with transferred learning rate achieves 2.9 times faster convergence than the original DiT-XL-2. Finally, we validate the effectiveness of $\mu$P on text-to-image generation by scaling PixArt-$\alpha$ from 0.04B to 0.61B and MMDiT from 0.18B to 18B. In both cases, models under $\mu$P outperform their respective baselines while requiring small tuning cost, only 5.5% of one training run for PixArt-$\alpha$ and 3% of consumption by human experts for MMDiT-18B. These results establish $\mu$P as a principled and efficient framework for scaling diffusion Transformers.
>
---
#### [replaced 045] On the Theory of Conditional Feature Alignment for Unsupervised Domain-Adaptive Counting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17137v2](http://arxiv.org/pdf/2506.17137v2)**

> **作者:** Zhuonan Liang; Dongnan Liu; Jianan Fan; Yaxuan Song; Qiang Qu; Runnan Chen; Yu Yao; Peng Fu; Weidong Cai
>
> **备注:** 18 pages, 6 figures, 5 tables
>
> **摘要:** Object counting models suffer when deployed across domains with differing density variety, since density shifts are inherently task-relevant and violate standard domain adaptation assumptions. To address this, we propose a theoretical framework of conditional feature alignment and provide a straightforward implementation. By theoretical analysis, our framework is feasible to achieve superior cross-domain generalization for counting. In the presented network, the features related to density are explicitly preserved across domains. Theoretically, we formalize the notion of conditional divergence by partitioning each domain into subsets and measuring divergences per condition. We then derive a joint error bound showing that, under discrete label spaces treated as condition sets, aligning distributions conditionally leads to tighter bounds on the combined source-target decision error than unconditional alignment. Empirically, we demonstrate the effectiveness of our approach through extensive experiments on multiple counting datasets with varying density distributions. The results show that our method outperforms existing unsupervised domain adaptation methods, empirically validating the theoretical insights on conditional feature alignment.
>
---
#### [replaced 046] Larger Hausdorff Dimension in Scanning Pattern Facilitates Mamba-Based Methods in Low-Light Image Enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.26001v2](http://arxiv.org/pdf/2510.26001v2)**

> **作者:** Xinhua Wang; Caibo Feng; Xiangjun Fu; Chunxiao Liu
>
> **摘要:** We propose an innovative enhancement to the Mamba framework by increasing the Hausdorff dimension of its scanning pattern through a novel Hilbert Selective Scan mechanism. This mechanism explores the feature space more effectively, capturing intricate fine-scale details and improving overall coverage. As a result, it mitigates information inconsistencies while refining spatial locality to better capture subtle local interactions without sacrificing the model's ability to handle long-range dependencies. Extensive experiments on publicly available benchmarks demonstrate that our approach significantly improves both the quantitative metrics and qualitative visual fidelity of existing Mamba-based low-light image enhancement methods, all while reducing computational resource consumption and shortening inference time. We believe that this refined strategy not only advances the state-of-the-art in low-light image enhancement but also holds promise for broader applications in fields that leverage Mamba-based techniques.
>
---
#### [replaced 047] Panoramic Out-of-Distribution Segmentation for Autonomous Driving
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.03539v2](http://arxiv.org/pdf/2505.03539v2)**

> **作者:** Mengfei Duan; Yuheng Zhang; Yihong Cao; Fei Teng; Kai Luo; Jiaming Zhang; Kailun Yang; Zhiyong Li
>
> **备注:** Code and datasets will be available at https://github.com/MengfeiD/PanOoS
>
> **摘要:** Panoramic imaging enables capturing 360{\deg} images with an ultra-wide Field-of-View (FoV) for dense omnidirectional perception, which is critical to applications, such as autonomous driving and augmented reality, etc. However, current panoramic semantic segmentation methods fail to identify outliers, and pinhole Out-of-distribution Segmentation (OoS) models perform unsatisfactorily in the panoramic domain due to background clutter and pixel distortions. To address these issues, we introduce a new task, Panoramic Out-of-distribution Segmentation (PanOoS), with the aim of achieving comprehensive and safe scene understanding. Furthermore, we propose the first solution, POS, which adapts to the characteristics of panoramic images through text-guided prompt distribution learning. Specifically, POS integrates a disentanglement strategy designed to materialize the cross-domain generalization capability of CLIP. The proposed Prompt-based Restoration Attention (PRA) optimizes semantic decoding by prompt guidance and self-adaptive correction, while Bilevel Prompt Distribution Learning (BPDL) refines the manifold of per-pixel mask embeddings via semantic prototype supervision. Besides, to compensate for the scarcity of PanOoS datasets, we establish two benchmarks: DenseOoS, which features diverse outliers in complex environments, and QuadOoS, captured by a quadruped robot with a panoramic annular lens system. Extensive experiments demonstrate superior performance of POS, with AuPRC improving by 34.25% and FPR95 decreasing by 21.42% on DenseOoS, outperforming state-of-the-art pinhole-OoS methods. Moreover, POS achieves leading closed-set segmentation capabilities and advances the development of panoramic understanding. Code and datasets will be available at https://github.com/MengfeiD/PanOoS.
>
---
#### [replaced 048] Human Uncertainty-Aware Data Selection and Automatic Labeling in Visual Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.11295v2](http://arxiv.org/pdf/2510.11295v2)**

> **作者:** Jian Lan; Zhicheng Liu; Udo Schlegel; Raoyuan Zhao; Yihong Liu; Hinrich Schütze; Michael A. Hedderich; Thomas Seidl
>
> **摘要:** Large vision-language models (VLMs) achieve strong performance in Visual Question Answering but still rely heavily on supervised fine-tuning (SFT) with massive labeled datasets, which is costly due to human annotations. Crucially, real-world datasets often exhibit human uncertainty (HU) -- variation in human confidence across annotations -- but standard SFT simply optimizes toward the most frequent label, disregarding HU distributions. This leaves two open questions: How does HU affect SFT, and how can HU be effectively leveraged in training? In this work, we first conduct a systematic evaluation of VLMs across varying HU levels. We have two key findings: (i) surprisingly, high-HU samples contribute little or even degrade model performance, and (ii) naively training on the full dataset yields under-calibrated models that fail to capture HU distributions. Motivated by these findings, we introduce HaDola, a human uncertainty-aware data selection and automatic labeling framework. HaDola operates in four stages -- discriminate, self-annotate, error trigger, and training -- to iteratively identify harmful samples, prioritize informative ones, and bootstrap from a small seed set (5\% of data). Our approach substantially reduces reliance on costly HU annotations and makes VLMs more accurate and better calibrated. Extensive experiments on VQAv2 and VizWiz datasets demonstrate that HaDola consistently matches or outperforms state-of-the-art baselines with less training data. Our work highlights the importance of explicitly modeling HU in SFT, suggesting that better utilization of HU is more effective than merely scaling up dataset size.
>
---
