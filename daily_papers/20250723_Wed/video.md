# 计算机视觉 cs.CV

- **最新发布 100 篇**

- **更新 82 篇**

## 最新发布

#### [new 001] Disrupting Semantic and Abstract Features for Better Adversarial Transferability
- **分类: cs.CV**

- **简介: 该论文属于对抗攻击任务，旨在提升黑盒攻击中的对抗样本可迁移性。现有方法多关注语义特征，而本文提出SAFER，结合语义与高频抽象特征（如纹理、边缘）进行攻击，通过BLOCKMIX和SELF-MIX策略增强特征重要性计算，从而提升攻击效果。**

- **链接: [http://arxiv.org/pdf/2507.16052v1](http://arxiv.org/pdf/2507.16052v1)**

> **作者:** Yuyang Luo; Xiaosen Wang; Zhijin Ge; Yingzhe He
>
> **摘要:** Adversarial examples pose significant threats to deep neural networks (DNNs), and their property of transferability in the black-box setting has led to the emergence of transfer-based attacks, making it feasible to target real-world applications employing DNNs. Among them, feature-level attacks, where intermediate features are perturbed based on feature importance weight matrix computed from transformed images, have gained popularity. In this work, we find that existing feature-level attacks primarily manipulate the semantic information to derive the weight matrix. Inspired by several works that find CNNs tend to focus more on high-frequency components (a.k.a. abstract features, e.g., texture, edge, etc.), we validate that transforming images in the high-frequency space also improves transferability. Based on this finding, we propose a balanced approach called Semantic and Abstract FEatures disRuption (SAFER). Specifically, SAFER conducts BLOCKMIX on the input image and SELF-MIX on the frequency spectrum when computing the weight matrix to highlight crucial features. By using such a weight matrix, we can direct the attacker to disrupt both semantic and abstract features, leading to improved transferability. Extensive experiments on the ImageNet dataset also demonstrate the effectiveness of our method in boosting adversarial transferability.
>
---
#### [new 002] From Flat to Round: Redefining Brain Decoding with Surface-Based fMRI and Cortex Structure
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于脑解码任务，旨在从fMRI数据重建视觉刺激。它试图解决现有方法忽略脑结构功能关系、空间信息和个体差异的问题。论文提出了球形分词器处理fMRI信号，结合sMRI数据建模个体差异，并采用正样本混合法提升重建效果。**

- **链接: [http://arxiv.org/pdf/2507.16389v1](http://arxiv.org/pdf/2507.16389v1)**

> **作者:** Sijin Yu; Zijiao Chen; Wenxuan Wu; Shengxian Chen; Zhongliang Liu; Jingxin Nie; Xiaofen Xing; Xiangmin Xu; Xin Zhang
>
> **备注:** 18 pages, 14 figures, ICCV Findings 2025
>
> **摘要:** Reconstructing visual stimuli from human brain activity (e.g., fMRI) bridges neuroscience and computer vision by decoding neural representations. However, existing methods often overlook critical brain structure-function relationships, flattening spatial information and neglecting individual anatomical variations. To address these issues, we propose (1) a novel sphere tokenizer that explicitly models fMRI signals as spatially coherent 2D spherical data on the cortical surface; (2) integration of structural MRI (sMRI) data, enabling personalized encoding of individual anatomical variations; and (3) a positive-sample mixup strategy for efficiently leveraging multiple fMRI scans associated with the same visual stimulus. Collectively, these innovations enhance reconstruction accuracy, biological interpretability, and generalizability across individuals. Experiments demonstrate superior reconstruction performance compared to SOTA methods, highlighting the effectiveness and interpretability of our biologically informed approach.
>
---
#### [new 003] A Multimodal Deviation Perceiving Framework for Weakly-Supervised Temporal Forgery Localization
- **分类: cs.CV**

- **简介: 该论文属于弱监督时间伪造定位任务，旨在通过视频级标注识别伪造片段。针对现有方法受限、耗时且难以扩展的问题，论文提出多模态偏差感知框架MDP，包含多模态交互机制和偏差感知损失，实现伪造片段的精确起止时间定位。**

- **链接: [http://arxiv.org/pdf/2507.16596v1](http://arxiv.org/pdf/2507.16596v1)**

> **作者:** Wenbo Xu; Junyan Wu; Wei Lu; Xiangyang Luo; Qian Wang
>
> **备注:** 9 pages, 3 figures,conference
>
> **摘要:** Current researches on Deepfake forensics often treat detection as a classification task or temporal forgery localization problem, which are usually restrictive, time-consuming, and challenging to scale for large datasets. To resolve these issues, we present a multimodal deviation perceiving framework for weakly-supervised temporal forgery localization (MDP), which aims to identify temporal partial forged segments using only video-level annotations. The MDP proposes a novel multimodal interaction mechanism (MI) and an extensible deviation perceiving loss to perceive multimodal deviation, which achieves the refined start and end timestamps localization of forged segments. Specifically, MI introduces a temporal property preserving cross-modal attention to measure the relevance between the visual and audio modalities in the probabilistic embedding space. It could identify the inter-modality deviation and construct comprehensive video features for temporal forgery localization. To explore further temporal deviation for weakly-supervised learning, an extensible deviation perceiving loss has been proposed, aiming at enlarging the deviation of adjacent segments of the forged samples and reducing that of genuine samples. Extensive experiments demonstrate the effectiveness of the proposed framework and achieve comparable results to fully-supervised approaches in several evaluation metrics.
>
---
#### [new 004] Survival Modeling from Whole Slide Images via Patch-Level Graph Clustering and Mixture Density Experts
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析与生存预测任务，旨在解决从全切片病理图像（WSIs）中准确预测癌症患者生存期的问题。论文提出一种新框架，结合动态补丁选择、图聚类、注意力机制与混合密度建模，显著提升了预测准确性。**

- **链接: [http://arxiv.org/pdf/2507.16476v1](http://arxiv.org/pdf/2507.16476v1)**

> **作者:** Ardhendu Sekhar; Vasu Soni; Keshav Aske; Garima Jain; Pranav Jeevan; Amit Sethi
>
> **摘要:** We introduce a modular framework for predicting cancer-specific survival from whole slide pathology images (WSIs) that significantly improves upon the state-of-the-art accuracy. Our method integrating four key components. Firstly, to tackle large size of WSIs, we use dynamic patch selection via quantile-based thresholding for isolating prognostically informative tissue regions. Secondly, we use graph-guided k-means clustering to capture phenotype-level heterogeneity through spatial and morphological coherence. Thirdly, we use attention mechanisms that model both intra- and inter-cluster relationships to contextualize local features within global spatial relations between various types of tissue compartments. Finally, we use an expert-guided mixture density modeling for estimating complex survival distributions using Gaussian mixture models. The proposed model achieves a concordance index of $0.712 \pm 0.028$ and Brier score of $0.254 \pm 0.018$ on TCGA-KIRC (renal cancer), and a concordance index of $0.645 \pm 0.017$ and Brier score of $0.281 \pm 0.031$ on TCGA-LUAD (lung adenocarcinoma). These results are significantly better than the state-of-art and demonstrate predictive potential of the proposed method across diverse cancer types.
>
---
#### [new 005] Optimization of DNN-based HSI Segmentation FPGA-based SoC for ADS: A Practical Approach
- **分类: cs.CV; cs.AI; cs.AR; cs.LG; eess.IV**

- **简介: 该论文属于自动驾驶系统中的图像处理任务，旨在解决基于深度学习的高光谱成像（HSI）分割模型在边缘设备上部署时面临的计算资源消耗大、延迟高等问题。作者通过软硬件协同设计、模型压缩与优化等方法，实现了高效实时的HSI分割。**

- **链接: [http://arxiv.org/pdf/2507.16556v1](http://arxiv.org/pdf/2507.16556v1)**

> **作者:** Jon Gutiérrez-Zaballa; Koldo Basterretxea; Javier Echanobe
>
> **摘要:** The use of HSI for autonomous navigation is a promising research field aimed at improving the accuracy and robustness of detection, tracking, and scene understanding systems based on vision sensors. Combining advanced computer algorithms, such as DNNs, with small-size snapshot HSI cameras enhances the reliability of these systems. HSI overcomes intrinsic limitations of greyscale and RGB imaging in depicting physical properties of targets, particularly regarding spectral reflectance and metamerism. Despite promising results in HSI-based vision developments, safety-critical systems like ADS demand strict constraints on latency, resource consumption, and security, motivating the shift of ML workloads to edge platforms. This involves a thorough software/hardware co-design scheme to distribute and optimize the tasks efficiently among the limited resources of computing platforms. With respect to inference, the over-parameterized nature of DNNs poses significant computational challenges for real-time on-the-edge deployment. In addition, the intensive data preprocessing required by HSI, which is frequently overlooked, must be carefully managed in terms of memory arrangement and inter-task communication to enable an efficient integrated pipeline design on a SoC. This work presents a set of optimization techniques for the practical co-design of a DNN-based HSI segmentation processor deployed on a FPGA-based SoC targeted at ADS, including key optimizations such as functional software/hardware task distribution, hardware-aware preprocessing, ML model compression, and a complete pipelined deployment. Applied compression techniques significantly reduce the complexity of the designed DNN to 24.34% of the original operations and to 1.02% of the original number of parameters, achieving a 2.86x speed-up in the inference task without noticeable degradation of the segmentation accuracy.
>
---
#### [new 006] MONITRS: Multimodal Observations of Natural Incidents Through Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于自然灾害监测任务，旨在解决现有遥感数据分析受限于单一灾种、依赖人工解读及缺乏时序标注的问题。作者构建了包含多模态数据的MONITRS数据集，并验证了其在灾害监测中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.16228v1](http://arxiv.org/pdf/2507.16228v1)**

> **作者:** Shreelekha Revankar; Utkarsh Mall; Cheng Perng Phoo; Kavita Bala; Bharath Hariharan
>
> **备注:** 17 pages, 9 figures, 4 tables
>
> **摘要:** Natural disasters cause devastating damage to communities and infrastructure every year. Effective disaster response is hampered by the difficulty of accessing affected areas during and after events. Remote sensing has allowed us to monitor natural disasters in a remote way. More recently there have been advances in computer vision and deep learning that help automate satellite imagery analysis, However, they remain limited by their narrow focus on specific disaster types, reliance on manual expert interpretation, and lack of datasets with sufficient temporal granularity or natural language annotations for tracking disaster progression. We present MONITRS, a novel multimodal dataset of more than 10,000 FEMA disaster events with temporal satellite imagery and natural language annotations from news articles, accompanied by geotagged locations, and question-answer pairs. We demonstrate that fine-tuning existing MLLMs on our dataset yields significant performance improvements for disaster monitoring tasks, establishing a new benchmark for machine learning-assisted disaster response systems. Code can be found at: https://github.com/ShreelekhaR/MONITRS
>
---
#### [new 007] Denoising-While-Completing Network (DWCNet): Robust Point Cloud Completion Under Corruption
- **分类: cs.CV**

- **简介: 该论文属于点云补全任务，旨在解决真实场景中受噪声和遮挡影响的点云数据补全问题。作者提出了DWCNet网络，通过噪声管理模块结合对比学习和自注意力机制，在补全的同时抑制噪声，提升了模型在多种退化情况下的鲁棒性，并构建了新的基准数据集CPCCD。**

- **链接: [http://arxiv.org/pdf/2507.16743v1](http://arxiv.org/pdf/2507.16743v1)**

> **作者:** Keneni W. Tesema; Lyndon Hill; Mark W. Jones; Gary K. L. Tam
>
> **备注:** Accepted for Computers and Graphics and EG Symposium on 3D Object Retrieval 2025 (3DOR'25)
>
> **摘要:** Point cloud completion is crucial for 3D computer vision tasks in autonomous driving, augmented reality, and robotics. However, obtaining clean and complete point clouds from real-world environments is challenging due to noise and occlusions. Consequently, most existing completion networks -- trained on synthetic data -- struggle with real-world degradations. In this work, we tackle the problem of completing and denoising highly corrupted partial point clouds affected by multiple simultaneous degradations. To benchmark robustness, we introduce the Corrupted Point Cloud Completion Dataset (CPCCD), which highlights the limitations of current methods under diverse corruptions. Building on these insights, we propose DWCNet (Denoising-While-Completing Network), a completion framework enhanced with a Noise Management Module (NMM) that leverages contrastive learning and self-attention to suppress noise and model structural relationships. DWCNet achieves state-of-the-art performance on both clean and corrupted, synthetic and real-world datasets. The dataset and code will be publicly available at https://github.com/keneniwt/DWCNET-Robust-Point-Cloud-Completion-against-Corruptions
>
---
#### [new 008] Task-Specific Zero-shot Quantization-Aware Training for Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决无真实数据下的模型量化问题。现有零样本量化方法使用通用合成数据，导致检测性能下降。作者提出任务特定的零样本量化框架，通过合成包含位置、尺寸和类别信息的校准集，并结合知识蒸馏进行训练，有效提升量化模型性能。**

- **链接: [http://arxiv.org/pdf/2507.16782v1](http://arxiv.org/pdf/2507.16782v1)**

> **作者:** Changhao Li; Xinrui Chen; Ji Wang; Kang Zhao; Jianfei Chen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Quantization is a key technique to reduce network size and computational complexity by representing the network parameters with a lower precision. Traditional quantization methods rely on access to original training data, which is often restricted due to privacy concerns or security challenges. Zero-shot Quantization (ZSQ) addresses this by using synthetic data generated from pre-trained models, eliminating the need for real training data. Recently, ZSQ has been extended to object detection. However, existing methods use unlabeled task-agnostic synthetic images that lack the specific information required for object detection, leading to suboptimal performance. In this paper, we propose a novel task-specific ZSQ framework for object detection networks, which consists of two main stages. First, we introduce a bounding box and category sampling strategy to synthesize a task-specific calibration set from the pre-trained network, reconstructing object locations, sizes, and category distributions without any prior knowledge. Second, we integrate task-specific training into the knowledge distillation process to restore the performance of quantized detection networks. Extensive experiments conducted on the MS-COCO and Pascal VOC datasets demonstrate the efficiency and state-of-the-art performance of our method. Our code is publicly available at: https://github.com/DFQ-Dojo/dfq-toolkit .
>
---
#### [new 009] One Polyp Identifies All: One-Shot Polyp Segmentation with SAM via Cascaded Priors and Iterative Prompt Evolution
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决结直肠癌早期检测中的息肉分割问题。现有方法依赖大量标注且泛化能力差。作者提出OP-SAM框架，基于SAM模型，通过相关性生成先验、级联融合和迭代提示优化，在仅需单张标注图像的情况下实现高效、准确的息肉分割。**

- **链接: [http://arxiv.org/pdf/2507.16337v1](http://arxiv.org/pdf/2507.16337v1)**

> **作者:** Xinyu Mao; Xiaohan Xing; Fei Meng; Jianbang Liu; Fan Bai; Qiang Nie; Max Meng
>
> **备注:** accepted by ICCV2025
>
> **摘要:** Polyp segmentation is vital for early colorectal cancer detection, yet traditional fully supervised methods struggle with morphological variability and domain shifts, requiring frequent retraining. Additionally, reliance on large-scale annotations is a major bottleneck due to the time-consuming and error-prone nature of polyp boundary labeling. Recently, vision foundation models like Segment Anything Model (SAM) have demonstrated strong generalizability and fine-grained boundary detection with sparse prompts, effectively addressing key polyp segmentation challenges. However, SAM's prompt-dependent nature limits automation in medical applications, since manually inputting prompts for each image is labor-intensive and time-consuming. We propose OP-SAM, a One-shot Polyp segmentation framework based on SAM that automatically generates prompts from a single annotated image, ensuring accurate and generalizable segmentation without additional annotation burdens. Our method introduces Correlation-based Prior Generation (CPG) for semantic label transfer and Scale-cascaded Prior Fusion (SPF) to adapt to polyp size variations as well as filter out noisy transfers. Instead of dumping all prompts at once, we devise Euclidean Prompt Evolution (EPE) for iterative prompt refinement, progressively enhancing segmentation quality. Extensive evaluations across five datasets validate OP-SAM's effectiveness. Notably, on Kvasir, it achieves 76.93% IoU, surpassing the state-of-the-art by 11.44%.
>
---
#### [new 010] Faithful, Interpretable Chest X-ray Diagnosis with Anti-Aliased B-cos Networks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学影像诊断任务，旨在解决现有B-cos网络在胸片诊断中解释图存在走样伪影且仅适用于多类分类的问题。论文引入抗走样策略并扩展模型以支持多标签分类，从而提升解释质量和临床适用性。**

- **链接: [http://arxiv.org/pdf/2507.16761v1](http://arxiv.org/pdf/2507.16761v1)**

> **作者:** Marcel Kleinmann; Shashank Agnihotri; Margret Keuper
>
> **摘要:** Faithfulness and interpretability are essential for deploying deep neural networks (DNNs) in safety-critical domains such as medical imaging. B-cos networks offer a promising solution by replacing standard linear layers with a weight-input alignment mechanism, producing inherently interpretable, class-specific explanations without post-hoc methods. While maintaining diagnostic performance competitive with state-of-the-art DNNs, standard B-cos models suffer from severe aliasing artifacts in their explanation maps, making them unsuitable for clinical use where clarity is essential. Additionally, the original B-cos formulation is limited to multi-class settings, whereas chest X-ray analysis often requires multi-label classification due to co-occurring abnormalities. In this work, we address both limitations: (1) we introduce anti-aliasing strategies using FLCPooling (FLC) and BlurPool (BP) to significantly improve explanation quality, and (2) we extend B-cos networks to support multi-label classification. Our experiments on chest X-ray datasets demonstrate that the modified $\text{B-cos}_\text{FLC}$ and $\text{B-cos}_\text{BP}$ preserve strong predictive performance while providing faithful and artifact-free explanations suitable for clinical application in multi-label settings. Code available at: $\href{https://github.com/mkleinma/B-cos-medical-paper}{GitHub repository}$.
>
---
#### [new 011] LMM4Edit: Benchmarking and Evaluating Multimodal Image Editing with LMMs
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于图像编辑评估任务，旨在解决现有文本引导图像编辑模型在质量、编辑对齐和一致性方面的不足。作者构建了包含18K编辑图像的大规模基准EBench-18K，并提出LMM4Edit指标，利用多模态大模型综合评估图像编辑效果，验证其与人类偏好的一致性。**

- **链接: [http://arxiv.org/pdf/2507.16193v1](http://arxiv.org/pdf/2507.16193v1)**

> **作者:** Zitong Xu; Huiyu Duan; Bingnan Liu; Guangji Ma; Jiarui Wang; Liu Yang; Shiqi Gao; Xiaoyu Wang; Jia Wang; Xiongkuo Min; Guangtao Zhai; Weisi Lin
>
> **摘要:** The rapid advancement of Text-guided Image Editing (TIE) enables image modifications through text prompts. However, current TIE models still struggle to balance image quality, editing alignment, and consistency with the original image, limiting their practical applications. Existing TIE evaluation benchmarks and metrics have limitations on scale or alignment with human perception. To this end, we introduce EBench-18K, the first large-scale image Editing Benchmark including 18K edited images with fine-grained human preference annotations for evaluating TIE. Specifically, EBench-18K includes 1,080 source images with corresponding editing prompts across 21 tasks, 18K+ edited images produced by 17 state-of-the-art TIE models, 55K+ mean opinion scores (MOSs) assessed from three evaluation dimensions, and 18K+ question-answering (QA) pairs. Based on EBench-18K, we employ outstanding LMMs to assess edited images, while the evaluation results, in turn, provide insights into assessing the alignment between the LMMs' understanding ability and human preferences. Then, we propose LMM4Edit, a LMM-based metric for evaluating image Editing models from perceptual quality, editing alignment, attribute preservation, and task-specific QA accuracy in an all-in-one manner. Extensive experiments show that LMM4Edit achieves outstanding performance and aligns well with human preference. Zero-shot validation on the other datasets also shows the generalization ability of our model. The dataset and code are available at https://github.com/IntMeGroup/LMM4Edit.
>
---
#### [new 012] Synthetic Data Matters: Re-training with Geo-typical Synthetic Labels for Building Detection
- **分类: cs.CV**

- **简介: 该论文属于遥感图像中的建筑物分割任务，旨在解决模型在不同地理区域泛化能力不足的问题。作者提出在测试时使用基于目标区域城市布局生成的合成数据进行再训练，结合对抗域适应方法，有效提升分割性能，缓解纯合成数据的“模型崩溃”问题。**

- **链接: [http://arxiv.org/pdf/2507.16657v1](http://arxiv.org/pdf/2507.16657v1)**

> **作者:** Shuang Song; Yang Tang; Rongjun Qin
>
> **备注:** 14 pages, 5 figures, This work has been submitted to the IEEE for possible publication
>
> **摘要:** Deep learning has significantly advanced building segmentation in remote sensing, yet models struggle to generalize on data of diverse geographic regions due to variations in city layouts and the distribution of building types, sizes and locations. However, the amount of time-consuming annotated data for capturing worldwide diversity may never catch up with the demands of increasingly data-hungry models. Thus, we propose a novel approach: re-training models at test time using synthetic data tailored to the target region's city layout. This method generates geo-typical synthetic data that closely replicates the urban structure of a target area by leveraging geospatial data such as street network from OpenStreetMap. Using procedural modeling and physics-based rendering, very high-resolution synthetic images are created, incorporating domain randomization in building shapes, materials, and environmental illumination. This enables the generation of virtually unlimited training samples that maintain the essential characteristics of the target environment. To overcome synthetic-to-real domain gaps, our approach integrates geo-typical data into an adversarial domain adaptation framework for building segmentation. Experiments demonstrate significant performance enhancements, with median improvements of up to 12%, depending on the domain gap. This scalable and cost-effective method blends partial geographic knowledge with synthetic imagery, providing a promising solution to the "model collapse" issue in purely synthetic datasets. It offers a practical pathway to improving generalization in remote sensing building segmentation without extensive real-world annotations.
>
---
#### [new 013] Dyna3DGR: 4D Cardiac Motion Tracking with Dynamic 3D Gaussian Representation
- **分类: cs.CV**

- **简介: 论文提出Dyna3DGR，用于4D心脏运动追踪任务，旨在解决因心肌组织均匀、缺乏显著特征导致的精细运动追踪难题。现有方法在拓扑一致性或细节保留上存在不足。该文结合3D高斯表示与神经运动场建模，实现无需大量训练数据的自监督优化，提升追踪精度。**

- **链接: [http://arxiv.org/pdf/2507.16608v1](http://arxiv.org/pdf/2507.16608v1)**

> **作者:** Xueming Fu; Pei Wu; Yingtai Li; Xin Luo; Zihang Jiang; Junhao Mei; Jian Lu; Gao-Jun Teng; S. Kevin Zhou
>
> **备注:** Accepted to MICCAI 2025
>
> **摘要:** Accurate analysis of cardiac motion is crucial for evaluating cardiac function. While dynamic cardiac magnetic resonance imaging (CMR) can capture detailed tissue motion throughout the cardiac cycle, the fine-grained 4D cardiac motion tracking remains challenging due to the homogeneous nature of myocardial tissue and the lack of distinctive features. Existing approaches can be broadly categorized into image based and representation-based, each with its limitations. Image-based methods, including both raditional and deep learning-based registration approaches, either struggle with topological consistency or rely heavily on extensive training data. Representation-based methods, while promising, often suffer from loss of image-level details. To address these limitations, we propose Dynamic 3D Gaussian Representation (Dyna3DGR), a novel framework that combines explicit 3D Gaussian representation with implicit neural motion field modeling. Our method simultaneously optimizes cardiac structure and motion in a self-supervised manner, eliminating the need for extensive training data or point-to-point correspondences. Through differentiable volumetric rendering, Dyna3DGR efficiently bridges continuous motion representation with image-space alignment while preserving both topological and temporal consistency. Comprehensive evaluations on the ACDC dataset demonstrate that our approach surpasses state-of-the-art deep learning-based diffeomorphic registration methods in tracking accuracy. The code will be available in https://github.com/windrise/Dyna3DGR.
>
---
#### [new 014] Comparative validation of surgical phase recognition, instrument keypoint estimation, and instrument instance segmentation in endoscopy: Results of the PhaKIR 2024 challenge
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析与手术辅助技术任务，旨在提升微创手术中器械识别与定位的鲁棒性。论文通过组织PhaKIR 2024挑战，引入包含13例腹腔镜胆囊切除术的新多中心数据集，统一标注手术阶段识别、器械关键点估计与实例分割任务，支持结合手术上下文与时间信息的方法研究，推动机器人辅助手术场景理解的发展。**

- **链接: [http://arxiv.org/pdf/2507.16559v1](http://arxiv.org/pdf/2507.16559v1)**

> **作者:** Tobias Rueckert; David Rauber; Raphaela Maerkl; Leonard Klausmann; Suemeyye R. Yildiran; Max Gutbrod; Danilo Weber Nunes; Alvaro Fernandez Moreno; Imanol Luengo; Danail Stoyanov; Nicolas Toussaint; Enki Cho; Hyeon Bae Kim; Oh Sung Choo; Ka Young Kim; Seong Tae Kim; Gonçalo Arantes; Kehan Song; Jianjun Zhu; Junchen Xiong; Tingyi Lin; Shunsuke Kikuchi; Hiroki Matsuzaki; Atsushi Kouno; João Renato Ribeiro Manesco; João Paulo Papa; Tae-Min Choi; Tae Kyeong Jeong; Juyoun Park; Oluwatosin Alabi; Meng Wei; Tom Vercauteren; Runzhi Wu; Mengya Xu; An Wang; Long Bai; Hongliang Ren; Amine Yamlahi; Jakob Hennighausen; Lena Maier-Hein; Satoshi Kondo; Satoshi Kasai; Kousuke Hirasawa; Shu Yang; Yihui Wang; Hao Chen; Santiago Rodríguez; Nicolás Aparicio; Leonardo Manrique; Juan Camilo Lyons; Olivia Hosie; Nicolás Ayobi; Pablo Arbeláez; Yiping Li; Yasmina Al Khalil; Sahar Nasirihaghighi; Stefanie Speidel; Daniel Rueckert; Hubertus Feussner; Dirk Wilhelm; Christoph Palm
>
> **备注:** A challenge report pre-print containing 36 pages, 15 figures, and 13 tables
>
> **摘要:** Reliable recognition and localization of surgical instruments in endoscopic video recordings are foundational for a wide range of applications in computer- and robot-assisted minimally invasive surgery (RAMIS), including surgical training, skill assessment, and autonomous assistance. However, robust performance under real-world conditions remains a significant challenge. Incorporating surgical context - such as the current procedural phase - has emerged as a promising strategy to improve robustness and interpretability. To address these challenges, we organized the Surgical Procedure Phase, Keypoint, and Instrument Recognition (PhaKIR) sub-challenge as part of the Endoscopic Vision (EndoVis) challenge at MICCAI 2024. We introduced a novel, multi-center dataset comprising thirteen full-length laparoscopic cholecystectomy videos collected from three distinct medical institutions, with unified annotations for three interrelated tasks: surgical phase recognition, instrument keypoint estimation, and instrument instance segmentation. Unlike existing datasets, ours enables joint investigation of instrument localization and procedural context within the same data while supporting the integration of temporal information across entire procedures. We report results and findings in accordance with the BIAS guidelines for biomedical image analysis challenges. The PhaKIR sub-challenge advances the field by providing a unique benchmark for developing temporally aware, context-driven methods in RAMIS and offers a high-quality resource to support future research in surgical scene understanding.
>
---
#### [new 015] Local Dense Logit Relations for Enhanced Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文属于知识蒸馏任务，旨在解决现有方法在细粒度类别关系学习上的不足。作者提出了LDRLD方法，通过递归解耦和重组logit信息捕捉局部密集关系，并引入ADW策略动态调整关键类别对的权重，以提升学生模型性能。实验表明该方法在多个数据集上优于现有logit蒸馏方法。**

- **链接: [http://arxiv.org/pdf/2507.15911v1](http://arxiv.org/pdf/2507.15911v1)**

> **作者:** Liuchi Xu; Kang Liu; Jinshuai Liu; Lu Wang; Lisheng Xu; Jun Cheng
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** State-of-the-art logit distillation methods exhibit versatility, simplicity, and efficiency. Despite the advances, existing studies have yet to delve thoroughly into fine-grained relationships within logit knowledge. In this paper, we propose Local Dense Relational Logit Distillation (LDRLD), a novel method that captures inter-class relationships through recursively decoupling and recombining logit information, thereby providing more detailed and clearer insights for student learning. To further optimize the performance, we introduce an Adaptive Decay Weight (ADW) strategy, which can dynamically adjust the weights for critical category pairs using Inverse Rank Weighting (IRW) and Exponential Rank Decay (ERD). Specifically, IRW assigns weights inversely proportional to the rank differences between pairs, while ERD adaptively controls weight decay based on total ranking scores of category pairs. Furthermore, after the recursive decoupling, we distill the remaining non-target knowledge to ensure knowledge completeness and enhance performance. Ultimately, our method improves the student's performance by transferring fine-grained knowledge and emphasizing the most critical relationships. Extensive experiments on datasets such as CIFAR-100, ImageNet-1K, and Tiny-ImageNet demonstrate that our method compares favorably with state-of-the-art logit-based distillation approaches. The code will be made publicly available.
>
---
#### [new 016] PAT++: a cautionary tale about generative visual augmentation for Object Re-identification
- **分类: cs.CV**

- **简介: 该论文属于图像识别任务，旨在解决生成式数据增强在目标重识别中的有效性问题。作者提出PAT++方法，结合扩散自蒸馏与部分感知Transformer，实验发现生成图像导致性能下降，揭示了生成模型在细粒度识别中的局限性。**

- **链接: [http://arxiv.org/pdf/2507.15888v1](http://arxiv.org/pdf/2507.15888v1)**

> **作者:** Leonardo Santiago Benitez Pereira; Arathy Jeevan
>
> **摘要:** Generative data augmentation has demonstrated gains in several vision tasks, but its impact on object re-identification - where preserving fine-grained visual details is essential - remains largely unexplored. In this work, we assess the effectiveness of identity-preserving image generation for object re-identification. Our novel pipeline, named PAT++, incorporates Diffusion Self-Distillation into the well-established Part-Aware Transformer. Using the Urban Elements ReID Challenge dataset, we conduct extensive experiments with generated images used for both model training and query expansion. Our results show consistent performance degradation, driven by domain shifts and failure to retain identity-defining features. These findings challenge assumptions about the transferability of generative models to fine-grained recognition tasks and expose key limitations in current approaches to visual augmentation for identity-preserving applications.
>
---
#### [new 017] Automatic Fine-grained Segmentation-assisted Report Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像报告生成任务，旨在提升生成报告的准确性和可信度。论文提出ASaRG方法，通过将中间特征和细粒度分割图融合进LLaVA架构，显著提升了性能，并实现了报告内容与分割图的可追溯性。**

- **链接: [http://arxiv.org/pdf/2507.16623v1](http://arxiv.org/pdf/2507.16623v1)**

> **作者:** Frederic Jonske; Constantin Seibold; Osman Alperen Koras; Fin Bahnsen; Marie Bauer; Amin Dada; Hamza Kalisch; Anton Schily; Jens Kleesiek
>
> **摘要:** Reliable end-to-end clinical report generation has been a longstanding goal of medical ML research. The end goal for this process is to alleviate radiologists' workloads and provide second opinions to clinicians or patients. Thus, a necessary prerequisite for report generation models is a strong general performance and some type of innate grounding capability, to convince clinicians or patients of the veracity of the generated reports. In this paper, we present ASaRG (\textbf{A}utomatic \textbf{S}egmentation-\textbf{a}ssisted \textbf{R}eport \textbf{G}eneration), an extension of the popular LLaVA architecture that aims to tackle both of these problems. ASaRG proposes to fuse intermediate features and fine-grained segmentation maps created by specialist radiological models into LLaVA's multi-modal projection layer via simple concatenation. With a small number of added parameters, our approach achieves a +0.89\% performance gain ($p=0.012$) in CE F1 score compared to the LLaVA baseline when using only intermediate features, and +2.77\% performance gain ($p<0.001$) when adding a combination of intermediate features and fine-grained segmentation maps. Compared with COMG and ORID, two other report generation methods that utilize segmentations, the performance gain amounts to 6.98\% and 6.28\% in F1 score, respectively. ASaRG is not mutually exclusive with other changes made to the LLaVA architecture, potentially allowing our method to be combined with other advances in the field. Finally, the use of an arbitrary number of segmentations as part of the input demonstrably allows tracing elements of the report to the corresponding segmentation maps and verifying the groundedness of assessments. Our code will be made publicly available at a later date.
>
---
#### [new 018] Artifacts and Attention Sinks: Structured Approximations for Efficient Vision Transformers
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在提升视觉Transformer的效率。它分析了“大规模标记”和“伪影标记”在注意力机制中的作用，并提出了一种无需训练的线性时间和空间复杂度注意力方法FNA，以及去噪策略，从而在降低计算开销的同时保持模型性能。**

- **链接: [http://arxiv.org/pdf/2507.16018v1](http://arxiv.org/pdf/2507.16018v1)**

> **作者:** Andrew Lu; Wentinn Liao; Liuhui Wang; Huzheng Yang; Jianbo Shi
>
> **摘要:** Vision transformers have emerged as a powerful tool across a wide range of applications, yet their inner workings remain only partially understood. In this work, we examine the phenomenon of massive tokens - tokens with exceptionally high activation norms that act as attention sinks - and artifact tokens that emerge as a byproduct during inference. Our analysis reveals that these tokens mutually suppress one another through the attention mechanism, playing a critical role in regulating information flow within the network. Leveraging these insights, we introduce Fast Nystr\"om Attention (FNA), a training-free method that approximates self-attention in linear time and space by exploiting the structured patterns formed by massive and artifact tokens. Additionally, we propose a masking strategy to mitigate noise from these tokens, yielding modest performance gains at virtually no cost. We evaluate our approach on popular pretrained vision backbones and demonstrate competitive performance on retrieval, classification, segmentation, and visual question answering (VQA), all while reducing computational overhead.
>
---
#### [new 019] Benchmarking pig detection and tracking under diverse and challenging conditions
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决猪只检测与跟踪问题。为提升猪场管理和动物福利，作者构建了两个数据集（PigDetect和PigTrack），并在多种复杂场景下对检测与跟踪模型进行系统评估，发现挑战性训练图像和先进模型可显著提升性能，并指出未来改进方向。**

- **链接: [http://arxiv.org/pdf/2507.16639v1](http://arxiv.org/pdf/2507.16639v1)**

> **作者:** Jonathan Henrich; Christian Post; Maximilian Zilke; Parth Shiroya; Emma Chanut; Amir Mollazadeh Yamchi; Ramin Yahyapour; Thomas Kneib; Imke Traulsen
>
> **摘要:** To ensure animal welfare and effective management in pig farming, monitoring individual behavior is a crucial prerequisite. While monitoring tasks have traditionally been carried out manually, advances in machine learning have made it possible to collect individualized information in an increasingly automated way. Central to these methods is the localization of animals across space (object detection) and time (multi-object tracking). Despite extensive research of these two tasks in pig farming, a systematic benchmarking study has not yet been conducted. In this work, we address this gap by curating two datasets: PigDetect for object detection and PigTrack for multi-object tracking. The datasets are based on diverse image and video material from realistic barn conditions, and include challenging scenarios such as occlusions or bad visibility. For object detection, we show that challenging training images improve detection performance beyond what is achievable with randomly sampled images alone. Comparing different approaches, we found that state-of-the-art models offer substantial improvements in detection quality over real-time alternatives. For multi-object tracking, we observed that SORT-based methods achieve superior detection performance compared to end-to-end trainable models. However, end-to-end models show better association performance, suggesting they could become strong alternatives in the future. We also investigate characteristic failure cases of end-to-end models, providing guidance for future improvements. The detection and tracking models trained on our datasets perform well in unseen pens, suggesting good generalization capabilities. This highlights the importance of high-quality training data. The datasets and research code are made publicly available to facilitate reproducibility, re-use and further development.
>
---
#### [new 020] STAR: A Benchmark for Astronomical Star Fields Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于天文图像超分辨率（ASR）任务，旨在解决现有数据集在通量一致性、数据多样性和物体裁剪设置方面的局限性。论文构建了大规模数据集STAR，包含54,738对通量一致的星场图像，并提出通量误差（FE）评估指标及通量不变超分辨率模型FISR，显著提升了超分辨率效果。**

- **链接: [http://arxiv.org/pdf/2507.16385v1](http://arxiv.org/pdf/2507.16385v1)**

> **作者:** Kuo-Cheng Wu; Guohang Zhuang; Jinyang Huang; Xiang Zhang; Wanli Ouyang; Yan Lu
>
> **摘要:** Super-resolution (SR) advances astronomical imaging by enabling cost-effective high-resolution capture, crucial for detecting faraway celestial objects and precise structural analysis. However, existing datasets for astronomical SR (ASR) exhibit three critical limitations: flux inconsistency, object-crop setting, and insufficient data diversity, significantly impeding ASR development. We propose STAR, a large-scale astronomical SR dataset containing 54,738 flux-consistent star field image pairs covering wide celestial regions. These pairs combine Hubble Space Telescope high-resolution observations with physically faithful low-resolution counterparts generated through a flux-preserving data generation pipeline, enabling systematic development of field-level ASR models. To further empower the ASR community, STAR provides a novel Flux Error (FE) to evaluate SR models in physical view. Leveraging this benchmark, we propose a Flux-Invariant Super Resolution (FISR) model that could accurately infer the flux-consistent high-resolution images from input photometry, suppressing several SR state-of-the-art methods by 24.84% on a novel designed flux consistency metric, showing the priority of our method for astrophysics. Extensive experiments demonstrate the effectiveness of our proposed method and the value of our dataset. Code and models are available at https://github.com/GuoCheng12/STAR.
>
---
#### [new 021] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）推理任务，旨在解决现有方法缺乏显式多步规划与复杂任务适应能力的问题。作者提出ThinkAct框架，通过强化视觉潜在规划，实现高层推理与底层动作执行的结合，支持少样本适应、长视野规划与自我修正行为。**

- **链接: [http://arxiv.org/pdf/2507.16815v1](http://arxiv.org/pdf/2507.16815v1)**

> **作者:** Chi-Pin Huang; Yueh-Hua Wu; Min-Hung Chen; Yu-Chiang Frank Wang; Fu-En Yang
>
> **备注:** Project page: https://jasper0314-huang.github.io/thinkact-vla/
>
> **摘要:** Vision-language-action (VLA) reasoning tasks require agents to interpret multimodal instructions, perform long-horizon planning, and act adaptively in dynamic environments. Existing approaches typically train VLA models in an end-to-end fashion, directly mapping inputs to actions without explicit reasoning, which hinders their ability to plan over multiple steps or adapt to complex task variations. In this paper, we propose ThinkAct, a dual-system framework that bridges high-level reasoning with low-level action execution via reinforced visual latent planning. ThinkAct trains a multimodal LLM to generate embodied reasoning plans guided by reinforcing action-aligned visual rewards based on goal completion and trajectory consistency. These reasoning plans are compressed into a visual plan latent that conditions a downstream action model for robust action execution on target environments. Extensive experiments on embodied reasoning and robot manipulation benchmarks demonstrate that ThinkAct enables few-shot adaptation, long-horizon planning, and self-correction behaviors in complex embodied AI tasks.
>
---
#### [new 022] AMMNet: An Asymmetric Multi-Modal Network for Remote Sensing Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于遥感语义分割任务，旨在解决RGB和DSM多模态数据融合中的计算复杂度高与模态不对齐问题。论文提出AMMNet网络，通过不对称编码器、先验融合和分布对齐模块，提升分割精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.16158v1](http://arxiv.org/pdf/2507.16158v1)**

> **作者:** Hui Ye; Haodong Chen; Zeke Zexi Hu; Xiaoming Chen; Yuk Ying Chung
>
> **摘要:** Semantic segmentation in remote sensing (RS) has advanced significantly with the incorporation of multi-modal data, particularly the integration of RGB imagery and the Digital Surface Model (DSM), which provides complementary contextual and structural information about the ground object. However, integrating RGB and DSM often faces two major limitations: increased computational complexity due to architectural redundancy, and degraded segmentation performance caused by modality misalignment. These issues undermine the efficiency and robustness of semantic segmentation, particularly in complex urban environments where precise multi-modal integration is essential. To overcome these limitations, we propose Asymmetric Multi-Modal Network (AMMNet), a novel asymmetric architecture that achieves robust and efficient semantic segmentation through three designs tailored for RGB-DSM input pairs. To reduce architectural redundancy, the Asymmetric Dual Encoder (ADE) module assigns representational capacity based on modality-specific characteristics, employing a deeper encoder for RGB imagery to capture rich contextual information and a lightweight encoder for DSM to extract sparse structural features. Besides, to facilitate modality alignment, the Asymmetric Prior Fuser (APF) integrates a modality-aware prior matrix into the fusion process, enabling the generation of structure-aware contextual features. Additionally, the Distribution Alignment (DA) module enhances cross-modal compatibility by aligning feature distributions through divergence minimization. Extensive experiments on the ISPRS Vaihingen and Potsdam datasets demonstrate that AMMNet attains state-of-the-art segmentation accuracy among multi-modal networks while reducing computational and memory requirements.
>
---
#### [new 023] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation
- **分类: cs.CV**

- **简介: 该论文属于文本生成视频任务，旨在解决参考对象与目标对象间外观和结构差异大导致的运动迁移难题。论文提出MotionShot框架，通过语义特征匹配和形状重定向，实现跨对象的高质量运动迁移，保持外观一致性。**

- **链接: [http://arxiv.org/pdf/2507.16310v1](http://arxiv.org/pdf/2507.16310v1)**

> **作者:** Yanchen Liu; Yanan Sun; Zhening Xing; Junyao Gao; Kai Chen; Wenjie Pei
>
> **摘要:** Existing text-to-video methods struggle to transfer motion smoothly from a reference object to a target object with significant differences in appearance or structure between them. To address this challenge, we introduce MotionShot, a training-free framework capable of parsing reference-target correspondences in a fine-grained manner, thereby achieving high-fidelity motion transfer while preserving coherence in appearance. To be specific, MotionShot first performs semantic feature matching to ensure high-level alignments between the reference and target objects. It then further establishes low-level morphological alignments through reference-to-target shape retargeting. By encoding motion with temporal attention, our MotionShot can coherently transfer motion across objects, even in the presence of significant appearance and structure disparities, demonstrated by extensive experiments. The project page is available at: https://motionshot.github.io/.
>
---
#### [new 024] MAN++: Scaling Momentum Auxiliary Network for Supervised Local Learning in Vision Tasks
- **分类: cs.CV**

- **简介: 该论文属于视觉任务中的模型训练方法研究。针对监督局部学习中因梯度截断导致的性能下降问题，提出MAN++，通过引入动量辅助网络与参数指数滑动平均机制，增强模块间信息流动，并加入可学习缩放偏差解决特征差异，从而提升性能，减少显存消耗。**

- **链接: [http://arxiv.org/pdf/2507.16279v1](http://arxiv.org/pdf/2507.16279v1)**

> **作者:** Junhao Su; Feiyu Zhu; Hengyu Shi; Tianyang Han; Yurui Qiu; Junfeng Luo; Xiaoming Wei; Jialin Gao
>
> **备注:** 14 pages
>
> **摘要:** Deep learning typically relies on end-to-end backpropagation for training, a method that inherently suffers from issues such as update locking during parameter optimization, high GPU memory consumption, and a lack of biological plausibility. In contrast, supervised local learning seeks to mitigate these challenges by partitioning the network into multiple local blocks and designing independent auxiliary networks to update each block separately. However, because gradients are propagated solely within individual local blocks, performance degradation occurs, preventing supervised local learning from supplanting end-to-end backpropagation. To address these limitations and facilitate inter-block information flow, we propose the Momentum Auxiliary Network++ (MAN++). MAN++ introduces a dynamic interaction mechanism by employing the Exponential Moving Average (EMA) of parameters from adjacent blocks to enhance communication across the network. The auxiliary network, updated via EMA, effectively bridges the information gap between blocks. Notably, we observed that directly applying EMA parameters can be suboptimal due to feature discrepancies between local blocks. To resolve this issue, we introduce a learnable scaling bias that balances feature differences, thereby further improving performance. We validate MAN++ through extensive experiments on tasks that include image classification, object detection, and image segmentation, utilizing multiple network architectures. The experimental results demonstrate that MAN++ achieves performance comparable to end-to-end training while significantly reducing GPU memory usage. Consequently, MAN++ offers a novel perspective for supervised local learning and presents a viable alternative to conventional training methods.
>
---
#### [new 025] C2-Evo: Co-Evolving Multimodal Data and Model for Self-Improving Reasoning
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 论文提出C2-Evo框架，用于多模态大语言模型的自提升推理。该工作属于多模态学习与自进化模型任务，旨在解决现有方法在数据增强和模型进化上的分离问题，通过联合演进数据与模型，持续生成复杂多模态问题并动态适配模型训练，从而提升数学推理性能。**

- **链接: [http://arxiv.org/pdf/2507.16518v1](http://arxiv.org/pdf/2507.16518v1)**

> **作者:** Xiuwei Chen; Wentao Hu; Hanhui Li; Jun Zhou; Zisheng Chen; Meng Cao; Yihan Zeng; Kui Zhang; Yu-Jie Yuan; Jianhua Han; Hang Xu; Xiaodan Liang
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have shown impressive reasoning capabilities. However, further enhancing existing MLLMs necessitates high-quality vision-language datasets with carefully curated task complexities, which are both costly and challenging to scale. Although recent self-improving models that iteratively refine themselves offer a feasible solution, they still suffer from two core challenges: (i) most existing methods augment visual or textual data separately, resulting in discrepancies in data complexity (e.g., over-simplified diagrams paired with redundant textual descriptions); and (ii) the evolution of data and models is also separated, leading to scenarios where models are exposed to tasks with mismatched difficulty levels. To address these issues, we propose C2-Evo, an automatic, closed-loop self-improving framework that jointly evolves both training data and model capabilities. Specifically, given a base dataset and a base model, C2-Evo enhances them by a cross-modal data evolution loop and a data-model evolution loop. The former loop expands the base dataset by generating complex multimodal problems that combine structured textual sub-problems with iteratively specified geometric diagrams, while the latter loop adaptively selects the generated problems based on the performance of the base model, to conduct supervised fine-tuning and reinforcement learning alternately. Consequently, our method continuously refines its model and training data, and consistently obtains considerable performance gains across multiple mathematical reasoning benchmarks. Our code, models, and datasets will be released.
>
---
#### [new 026] Positive Style Accumulation: A Style Screening and Continuous Utilization Framework for Federated DG-ReID
- **分类: cs.CV; I.4.9; I.2.10**

- **简介: 该论文属于联邦域泛化行人重识别任务，旨在解决模型在不同域间泛化能力不足的问题。通过筛选并持续利用有助于泛化的正向风格，提出了一种风格筛选与持续利用框架（SSCU），包括动态风格记忆、记忆识别损失和协同风格训练策略，有效提升了模型在源域和目标域的表现。**

- **链接: [http://arxiv.org/pdf/2507.16238v1](http://arxiv.org/pdf/2507.16238v1)**

> **作者:** Xin Xu; Chaoyue Ren; Wei Liu; Wenke Huang; Bin Yang; Zhixi Yu; Kui Jiang
>
> **备注:** 10 pages, 3 figures, accepted at ACM MM 2025, Submission ID: 4394
>
> **摘要:** The Federated Domain Generalization for Person re-identification (FedDG-ReID) aims to learn a global server model that can be effectively generalized to source and target domains through distributed source domain data. Existing methods mainly improve the diversity of samples through style transformation, which to some extent enhances the generalization performance of the model. However, we discover that not all styles contribute to the generalization performance. Therefore, we define styles that are beneficial or harmful to the model's generalization performance as positive or negative styles. Based on this, new issues arise: How to effectively screen and continuously utilize the positive styles. To solve these problems, we propose a Style Screening and Continuous Utilization (SSCU) framework. Firstly, we design a Generalization Gain-guided Dynamic Style Memory (GGDSM) for each client model to screen and accumulate generated positive styles. Meanwhile, we propose a style memory recognition loss to fully leverage the positive styles memorized by Memory. Furthermore, we propose a Collaborative Style Training (CST) strategy to make full use of positive styles. Unlike traditional learning strategies, our approach leverages both newly generated styles and the accumulated positive styles stored in memory to train client models on two distinct branches. This training strategy is designed to effectively promote the rapid acquisition of new styles by the client models, and guarantees the continuous and thorough utilization of positive styles, which is highly beneficial for the model's generalization performance. Extensive experimental results demonstrate that our method outperforms existing methods in both the source domain and the target domain.
>
---
#### [new 027] An empirical study for the early detection of Mpox from skin lesion images using pretrained CNN models leveraging XAI technique
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决Mpox早期检测难题。利用预训练CNN模型（如InceptionV3、MobileNetV2）对皮肤病变图像进行分类，并结合XAI技术（Grad-CAM）提升模型可解释性。实验表明模型表现良好，但仍存在过拟合问题，未来需改进数据与方法以提高可靠性。**

- **链接: [http://arxiv.org/pdf/2507.15915v1](http://arxiv.org/pdf/2507.15915v1)**

> **作者:** Mohammad Asifur Rahim; Muhammad Nazmul Arefin; Md. Mizanur Rahman; Md Ali Hossain; Ahmed Moustafa
>
> **摘要:** Context: Mpox is a zoonotic disease caused by the Mpox virus, which shares similarities with other skin conditions, making accurate early diagnosis challenging. Artificial intelligence (AI), especially Deep Learning (DL), has a strong tool for medical image analysis; however, pre-trained models like CNNs and XAI techniques for mpox detection is underexplored. Objective: This study aims to evaluate the effectiveness of pre-trained CNN models (VGG16, VGG19, InceptionV3, MobileNetV2) for the early detection of monkeypox using binary and multi-class datasets. It also seeks to enhance model interpretability using Grad-CAM an XAI technique. Method: Two datasets, MSLD and MSLD v2.0, were used for training and validation. Transfer learning techniques were applied to fine-tune pre-trained CNN models by freezing initial layers and adding custom layers for adapting the final features for mpox detection task and avoid overfitting. Models performance were evaluated using metrics such as accuracy, precision, recall, F1-score and ROC. Grad-CAM was utilized for visualizing critical features. Results: InceptionV3 demonstrated the best performance on the binary dataset with an accuracy of 95%, while MobileNetV2 outperformed on the multi-class dataset with an accuracy of 93%. Grad-CAM successfully highlighted key image regions. Despite high accuracy, some models showed overfitting tendencies, as videnced by discrepancies between training and validation losses. Conclusion: This study underscores the potential of pre-trained CNN models in monkeypox detection and the value of XAI techniques. Future work should address dataset limitations, incorporate multimodal data, and explore additional interpretability techniques to improve diagnostic reliability and model transparency
>
---
#### [new 028] Edge-case Synthesis for Fisheye Object Detection: A Data-centric Perspective
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于目标检测任务，旨在解决鱼眼相机图像中因畸变导致的检测难题。通过分析模型盲点，合成边缘案例图像，补充训练数据，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.16254v1](http://arxiv.org/pdf/2507.16254v1)**

> **作者:** Seunghyeon Kim; Kyeongryeol Go
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Fisheye cameras introduce significant distortion and pose unique challenges to object detection models trained on conventional datasets. In this work, we propose a data-centric pipeline that systematically improves detection performance by focusing on the key question of identifying the blind spots of the model. Through detailed error analysis, we identify critical edge-cases such as confusing class pairs, peripheral distortions, and underrepresented contexts. Then we directly address them through edge-case synthesis. We fine-tuned an image generative model and guided it with carefully crafted prompts to produce images that replicate real-world failure modes. These synthetic images are pseudo-labeled using a high-quality detector and integrated into training. Our approach results in consistent performance gains, highlighting how deeply understanding data and selectively fixing its weaknesses can be impactful in specialized domains like fisheye object detection.
>
---
#### [new 029] A2Mamba: Attention-augmented State Space Models for Visual Recognition
- **分类: cs.CV**

- **简介: 论文提出A2Mamba，属于视觉识别任务。旨在解决Transformer与Mamba层间缺乏深度融合的问题。工作是设计多尺度注意力增强的状态空间模型（MASS），通过跨注意力机制增强空间依赖，提升动态建模能力。模型在图像分类、分割等任务上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.16624v1](http://arxiv.org/pdf/2507.16624v1)**

> **作者:** Meng Lou; Yunxiang Fu; Yizhou Yu
>
> **备注:** 14 pages, 5 figures, 13 tables
>
> **摘要:** Transformers and Mamba, initially invented for natural language processing, have inspired backbone architectures for visual recognition. Recent studies integrated Local Attention Transformers with Mamba to capture both local details and global contexts. Despite competitive performance, these methods are limited to simple stacking of Transformer and Mamba layers without any interaction mechanism between them. Thus, deep integration between Transformer and Mamba layers remains an open problem. We address this problem by proposing A2Mamba, a powerful Transformer-Mamba hybrid network architecture, featuring a new token mixer termed Multi-scale Attention-augmented State Space Model (MASS), where multi-scale attention maps are integrated into an attention-augmented SSM (A2SSM). A key step of A2SSM performs a variant of cross-attention by spatially aggregating the SSM's hidden states using the multi-scale attention maps, which enhances spatial dependencies pertaining to a two-dimensional space while improving the dynamic modeling capabilities of SSMs. Our A2Mamba outperforms all previous ConvNet-, Transformer-, and Mamba-based architectures in visual recognition tasks. For instance, A2Mamba-L achieves an impressive 86.1% top-1 accuracy on ImageNet-1K. In semantic segmentation, A2Mamba-B exceeds CAFormer-S36 by 2.5% in mIoU, while exhibiting higher efficiency. In object detection and instance segmentation with Cascade Mask R-CNN, A2Mamba-S surpasses MambaVision-B by 1.2%/0.9% in AP^b/AP^m, while having 40% less parameters. Code is publicly available at https://github.com/LMMMEng/A2Mamba.
>
---
#### [new 030] Scale Your Instructions: Enhance the Instruction-Following Fidelity of Unified Image Generation Model by Self-Adaptive Attention Scaling
- **分类: cs.CV**

- **简介: 该论文属于图像生成与编辑任务，旨在解决统一模型处理多子指令时忽视文本的问题。通过注意力机制分析，发现子指令与图像激活冲突，提出自适应注意力缩放方法（SaaS），提升指令遵循准确性，无需额外训练。**

- **链接: [http://arxiv.org/pdf/2507.16240v1](http://arxiv.org/pdf/2507.16240v1)**

> **作者:** Chao Zhou; Tianyi Wei; Nenghai Yu
>
> **备注:** Accept by ICCV2025
>
> **摘要:** Recent advancements in unified image generation models, such as OmniGen, have enabled the handling of diverse image generation and editing tasks within a single framework, accepting multimodal, interleaved texts and images in free form. This unified architecture eliminates the need for text encoders, greatly reducing model complexity and standardizing various image generation and editing tasks, making it more user-friendly. However, we found that it suffers from text instruction neglect, especially when the text instruction contains multiple sub-instructions. To explore this issue, we performed a perturbation analysis on the input to identify critical steps and layers. By examining the cross-attention maps of these key steps, we observed significant conflicts between neglected sub-instructions and the activations of the input image. In response, we propose Self-Adaptive Attention Scaling (SaaS), a method that leverages the consistency of cross-attention between adjacent timesteps to dynamically scale the attention activation for each sub-instruction. Our SaaS enhances instruction-following fidelity without requiring additional training or test-time optimization. Experimental results on instruction-based image editing and visual conditional image generation validate the effectiveness of our SaaS, showing superior instruction-following fidelity over existing methods. The code is available https://github.com/zhouchao-ops/SaaS.
>
---
#### [new 031] Stop-band Energy Constraint for Orthogonal Tunable Wavelet Units in Convolutional Neural Networks for Computer Vision problems
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于计算机视觉任务，旨在提升卷积神经网络在图像分类与异常检测中的性能。作者提出了一种基于正交可调小波单元的滤波器停带能量约束方法，应用于ResNet-18和ResNet-34中，显著提高了CIFAR-10和纹理数据集的分类准确率，并在MVTec榛子异常检测任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.16114v1](http://arxiv.org/pdf/2507.16114v1)**

> **作者:** An D. Le; Hung Nguyen; Sungbal Seo; You-Suk Bae; Truong Q. Nguyen
>
> **摘要:** This work introduces a stop-band energy constraint for filters in orthogonal tunable wavelet units with a lattice structure, aimed at improving image classification and anomaly detection in CNNs, especially on texture-rich datasets. Integrated into ResNet-18, the method enhances convolution, pooling, and downsampling operations, yielding accuracy gains of 2.48% on CIFAR-10 and 13.56% on the Describable Textures dataset. Similar improvements are observed in ResNet-34. On the MVTec hazelnut anomaly detection task, the proposed method achieves competitive results in both segmentation and detection, outperforming existing approaches.
>
---
#### [new 032] CMP: A Composable Meta Prompt for SAM-Based Cross-Domain Few-Shot Segmentation
- **分类: cs.CV**

- **简介: 该论文属于跨域少样本分割任务，旨在解决数据有限和领域差异带来的分割挑战。通过改进SAM模型，提出CMP框架，包含语义扩展、自动提示生成和领域差异缓解三个模块，实现更优跨域分割性能。**

- **链接: [http://arxiv.org/pdf/2507.16753v1](http://arxiv.org/pdf/2507.16753v1)**

> **作者:** Shuai Chen; Fanman Meng; Chunjin Yang; Haoran Wei; Chenhao Wu; Qingbo Wu; Hongliang Li
>
> **备注:** 3 figures
>
> **摘要:** Cross-Domain Few-Shot Segmentation (CD-FSS) remains challenging due to limited data and domain shifts. Recent foundation models like the Segment Anything Model (SAM) have shown remarkable zero-shot generalization capability in general segmentation tasks, making it a promising solution for few-shot scenarios. However, adapting SAM to CD-FSS faces two critical challenges: reliance on manual prompt and limited cross-domain ability. Therefore, we propose the Composable Meta-Prompt (CMP) framework that introduces three key modules: (i) the Reference Complement and Transformation (RCT) module for semantic expansion, (ii) the Composable Meta-Prompt Generation (CMPG) module for automated meta-prompt synthesis, and (iii) the Frequency-Aware Interaction (FAI) module for domain discrepancy mitigation. Evaluations across four cross-domain datasets demonstrate CMP's state-of-the-art performance, achieving 71.8\% and 74.5\% mIoU in 1-shot and 5-shot scenarios respectively.
>
---
#### [new 033] AtrousMamaba: An Atrous-Window Scanning Visual State Space Model for Remote Sensing Change Detection
- **分类: cs.CV**

- **简介: 论文提出AtrousMamba，属于遥感变化检测任务，旨在解决现有模型在捕捉局部细节与全局信息间的不平衡问题。通过引入空洞窗口扫描机制，有效融合局部特征与全局上下文，提升了二值变化检测和语义变化检测效果。**

- **链接: [http://arxiv.org/pdf/2507.16172v1](http://arxiv.org/pdf/2507.16172v1)**

> **作者:** Tao Wang; Tiecheng Bai; Chao Xu; Bin Liu; Erlei Zhang; Jiyun Huang; Hongming Zhang
>
> **摘要:** Recently, a novel visual state space (VSS) model, referred to as Mamba, has demonstrated significant progress in modeling long sequences with linear complexity, comparable to Transformer models, thereby enhancing its adaptability for processing visual data. Although most methods aim to enhance the global receptive field by directly modifying Mamba's scanning mechanism, they tend to overlook the critical importance of local information in dense prediction tasks. Additionally, whether Mamba can effectively extract local features as convolutional neural networks (CNNs) do remains an open question that merits further investigation. In this paper, We propose a novel model, AtrousMamba, which effectively balances the extraction of fine-grained local details with the integration of global contextual information. Specifically, our method incorporates an atrous-window selective scan mechanism, enabling a gradual expansion of the scanning range with adjustable rates. This design shortens the distance between adjacent tokens, enabling the model to effectively capture fine-grained local features and global context. By leveraging the atrous window scan visual state space (AWVSS) module, we design dedicated end-to-end Mamba-based frameworks for binary change detection (BCD) and semantic change detection (SCD), referred to as AWMambaBCD and AWMambaSCD, respectively. Experimental results on six benchmark datasets show that the proposed framework outperforms existing CNN-based, Transformer-based, and Mamba-based methods. These findings clearly demonstrate that Mamba not only captures long-range dependencies in visual data but also effectively preserves fine-grained local details.
>
---
#### [new 034] Enhancing Remote Sensing Vision-Language Models Through MLLM and LLM-Based High-Quality Image-Text Dataset Generation
- **分类: cs.CV**

- **简介: 该论文属于遥感图像与文本匹配任务，旨在解决高质量图像-文本对数据稀缺问题。作者提出MpGI方法，结合多模态大模型与大模型生成高质量文本描述，构建HQRS-IT-210K数据集，并基于此微调CLIP和CoCa模型，显著提升性能，减少训练数据需求。**

- **链接: [http://arxiv.org/pdf/2507.16716v1](http://arxiv.org/pdf/2507.16716v1)**

> **作者:** Yiguo He; Junjie Zhu; Yiying Li; Xiaoyu Zhang; Chunping Qiu; Jun Wang; Qiangjuan Huang; Ke Yang
>
> **备注:** SUBMIT TO IEEE TRANSACTIONS
>
> **摘要:** The application of Vision-language foundation models (VLFMs) to remote sensing (RS) imagery has garnered significant attention due to their superior capability in various downstream tasks. A key challenge lies in the scarcity of high-quality, large-scale, image-text paired training data. Recently, several works introduced extensive image-text datasets for RS and trained their VLFMs. However, due to the rudimentary methods used for generating captions, the quality of datasets is suboptimal, requiring larger volumes of training data, while only yielding modest performance improvements. In this paper, we propose a two-stage method named MpGI(Multi-Perspective Generation and Integration) for generating high-quality text captions for RS images. Firstly, we generate distinct and detailed descriptions from different perspectives using Rule-MLLM(Multimodal Large Language Model) Relay Generation and MLLMs generation methods. Next, we utilize Large Language Models (LLMs) to integrate these diverse descriptions into comprehensive captions, capturing details from multiple perspectives. Finally, we have created the HQRS-IT-210K dataset, including about 210,000 RS images and 1.3 million captions. We fine-tuned two VLFMs using our dataset: CLIP, a discriminative model, and CoCa, an image-to-text generative model. This process resulted in our proposed HQRS-CLIP and RS-CoCa models. Experimental results demonstrate that HQRS-CLIP surpassed the previous SOTA RS CLIP model in various downstream tasks while using only 4.2\% of the training data. RS-CoCa outperforms other advanced approaches across benchmark datasets and can generate captions for RS images that rival or even exceed manual annotations. Dataset, pre-trained models, and codes will be released at https://github.com/YiguoHe/HQRS-210K-and-HQRS-CLIP.
>
---
#### [new 035] Mamba-OTR: a Mamba-based Solution for Online Take and Release Detection from Untrimmed Egocentric Video
- **分类: cs.CV**

- **简介: 该论文属于视频动作检测任务，旨在解决第一视角视频中在线检测物体拿取与释放的问题。由于标注稀疏、类别不平衡及实时性要求，该任务具有挑战性。论文提出Mamba-OTR模型，基于Mamba架构，在训练时使用短片段视频，结合焦点损失与正则化策略，并利用时序递归来提升检测精度与效率，取得了优于Transformer与原生Mamba的效果。**

- **链接: [http://arxiv.org/pdf/2507.16342v1](http://arxiv.org/pdf/2507.16342v1)**

> **作者:** Alessandro Sebastiano Catinello; Giovanni Maria Farinella; Antonino Furnari
>
> **摘要:** This work tackles the problem of Online detection of Take and Release (OTR) of an object in untrimmed egocentric videos. This task is challenging due to severe label imbalance, with temporally sparse positive annotations, and the need for precise temporal predictions. Furthermore, methods need to be computationally efficient in order to be deployed in real-world online settings. To address these challenges, we propose Mamba-OTR, a model based on the Mamba architecture. Mamba-OTR is designed to exploit temporal recurrence during inference while being trained on short video clips. To address label imbalance, our training pipeline incorporates the focal loss and a novel regularization scheme that aligns model predictions with the evaluation metric. Extensive experiments on EPIC-KITCHENS-100, the comparisons with transformer-based approach, and the evaluation of different training and test schemes demonstrate the superiority of Mamba-OTR in both accuracy and efficiency. These finding are particularly evident when evaluating full-length videos or high frame-rate sequences, even when trained on short video snippets for computational convenience. The proposed Mamba-OTR achieves a noteworthy mp-mAP of 45.48 when operating in a sliding-window fashion, and 43.35 in streaming mode, versus the 20.32 of a vanilla transformer and 25.16 of a vanilla Mamba, thus providing a strong baseline for OTR. We will publicly release the source code of Mamba-OTR to support future research.
>
---
#### [new 036] Combined Image Data Augmentations diminish the benefits of Adaptive Label Smoothing
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16427v1](http://arxiv.org/pdf/2507.16427v1)**

> **作者:** Georg Siedel; Ekagra Gupta; Weijia Shao; Silvia Vock; Andrey Morozov
>
> **备注:** Preprint submitted to the Fast Review Track of DAGM German Conference on Pattern Recognition (GCPR) 2025
>
> **摘要:** Soft augmentation regularizes the supervised learning process of image classifiers by reducing label confidence of a training sample based on the magnitude of random-crop augmentation applied to it. This paper extends this adaptive label smoothing framework to other types of aggressive augmentations beyond random-crop. Specifically, we demonstrate the effectiveness of the method for random erasing and noise injection data augmentation. Adaptive label smoothing permits stronger regularization via higher-intensity Random Erasing. However, its benefits vanish when applied with a diverse range of image transformations as in the state-of-the-art TrivialAugment method, and excessive label smoothing harms robustness to common corruptions. Our findings suggest that adaptive label smoothing should only be applied when the training data distribution is dominated by a limited, homogeneous set of image transformation types.
>
---
#### [new 037] Navigating Large-Pose Challenge for High-Fidelity Face Reenactment with Video Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于人脸重演任务，旨在解决大姿态变化下生成高保真说话头视频的问题。现有方法在处理大姿态时效果不佳。作者提出FRVD模型，通过隐式关键点提取和运动对齐，并引入WFM模块，将源图像映射到预训练视频生成模型的潜在空间，从而提升姿态准确性与视觉质量。**

- **链接: [http://arxiv.org/pdf/2507.16341v1](http://arxiv.org/pdf/2507.16341v1)**

> **作者:** Mingtao Guo; Guanyu Xing; Yanci Zhang; Yanli Liu
>
> **摘要:** Face reenactment aims to generate realistic talking head videos by transferring motion from a driving video to a static source image while preserving the source identity. Although existing methods based on either implicit or explicit keypoints have shown promise, they struggle with large pose variations due to warping artifacts or the limitations of coarse facial landmarks. In this paper, we present the Face Reenactment Video Diffusion model (FRVD), a novel framework for high-fidelity face reenactment under large pose changes. Our method first employs a motion extractor to extract implicit facial keypoints from the source and driving images to represent fine-grained motion and to perform motion alignment through a warping module. To address the degradation introduced by warping, we introduce a Warping Feature Mapper (WFM) that maps the warped source image into the motion-aware latent space of a pretrained image-to-video (I2V) model. This latent space encodes rich priors of facial dynamics learned from large-scale video data, enabling effective warping correction and enhancing temporal coherence. Extensive experiments show that FRVD achieves superior performance over existing methods in terms of pose accuracy, identity preservation, and visual quality, especially in challenging scenarios with extreme pose variations.
>
---
#### [new 038] DFR: A Decompose-Fuse-Reconstruct Framework for Multi-Modal Few-Shot Segmentation
- **分类: cs.CV**

- **简介: 该论文属于多模态少样本分割任务，旨在解决如何有效利用多模态信息（视觉、文本、音频）提升分割性能。论文提出DFR框架，通过分解、对比融合与双路径重建，实现多模态语义与几何信息的协同利用，显著提升分割效果。**

- **链接: [http://arxiv.org/pdf/2507.16736v1](http://arxiv.org/pdf/2507.16736v1)**

> **作者:** Shuai Chen; Fanman Meng; Xiwei Zhang; Haoran Wei; Chenhao Wu; Qingbo Wu; Hongliang Li
>
> **备注:** 3 figures
>
> **摘要:** This paper presents DFR (Decompose, Fuse and Reconstruct), a novel framework that addresses the fundamental challenge of effectively utilizing multi-modal guidance in few-shot segmentation (FSS). While existing approaches primarily rely on visual support samples or textual descriptions, their single or dual-modal paradigms limit exploitation of rich perceptual information available in real-world scenarios. To overcome this limitation, the proposed approach leverages the Segment Anything Model (SAM) to systematically integrate visual, textual, and audio modalities for enhanced semantic understanding. The DFR framework introduces three key innovations: 1) Multi-modal Decompose: a hierarchical decomposition scheme that extracts visual region proposals via SAM, expands textual semantics into fine-grained descriptors, and processes audio features for contextual enrichment; 2) Multi-modal Contrastive Fuse: a fusion strategy employing contrastive learning to maintain consistency across visual, textual, and audio modalities while enabling dynamic semantic interactions between foreground and background features; 3) Dual-path Reconstruct: an adaptive integration mechanism combining semantic guidance from tri-modal fused tokens with geometric cues from multi-modal location priors. Extensive experiments across visual, textual, and audio modalities under both synthetic and real settings demonstrate DFR's substantial performance improvements over state-of-the-art methods.
>
---
#### [new 039] Scene Text Detection and Recognition "in light of" Challenging Environmental Conditions using Aria Glasses Egocentric Vision Cameras
- **分类: cs.CV**

- **简介: 该论文属于场景文本检测与识别任务，旨在解决现实环境中光照、距离等因素对识别效果的影响。作者使用Aria眼镜采集数据，评估两种OCR流程，并提出图像超分辨率和眼动追踪优化方法，以提升识别准确率和处理效率。**

- **链接: [http://arxiv.org/pdf/2507.16330v1](http://arxiv.org/pdf/2507.16330v1)**

> **作者:** Joseph De Mathia; Carlos Francisco Moreno-García
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** In an era where wearable technology is reshaping applications, Scene Text Detection and Recognition (STDR) becomes a straightforward choice through the lens of egocentric vision. Leveraging Meta's Project Aria smart glasses, this paper investigates how environmental variables, such as lighting, distance, and resolution, affect the performance of state-of-the-art STDR algorithms in real-world scenarios. We introduce a novel, custom-built dataset captured under controlled conditions and evaluate two OCR pipelines: EAST with CRNN, and EAST with PyTesseract. Our findings reveal that resolution and distance significantly influence recognition accuracy, while lighting plays a less predictable role. Notably, image upscaling emerged as a key pre-processing technique, reducing Character Error Rate (CER) from 0.65 to 0.48. We further demonstrate the potential of integrating eye-gaze tracking to optimise processing efficiency by focusing on user attention zones. This work not only benchmarks STDR performance under realistic conditions but also lays the groundwork for adaptive, user-aware AR systems. Our contributions aim to inspire future research in robust, context-sensitive text recognition for assistive and research-oriented applications, such as asset inspection and nutrition analysis. The code is available at https://github.com/josepDe/Project_Aria_STR.
>
---
#### [new 040] LPTR-AFLNet: Lightweight Integrated Chinese License Plate Rectification and Recognition Network
- **分类: cs.CV**

- **简介: 该论文属于中文车牌识别任务，旨在解决复杂环境下车牌透视变形及识别难题。作者提出LPTR-AFLNet，集成透视校正与识别，优化注意力模块与损失函数，实现高效准确的端到端处理。**

- **链接: [http://arxiv.org/pdf/2507.16362v1](http://arxiv.org/pdf/2507.16362v1)**

> **作者:** Guangzhu Xu; Pengcheng Zuo; Zhi Ke; Bangjun Lei
>
> **备注:** 28 pages, 33 figures
>
> **摘要:** Chinese License Plate Recognition (CLPR) faces numerous challenges in unconstrained and complex environments, particularly due to perspective distortions caused by various shooting angles and the correction of single-line and double-line license plates. Considering the limited computational resources of edge devices, developing a low-complexity, end-to-end integrated network for both correction and recognition is essential for achieving real-time and efficient deployment. In this work, we propose a lightweight, unified network named LPTR-AFLNet for correcting and recognizing Chinese license plates, which combines a perspective transformation correction module (PTR) with an optimized license plate recognition network, AFLNet. The network leverages the recognition output as a weak supervisory signal to effectively guide the correction process, ensuring accurate perspective distortion correction. To enhance recognition accuracy, we introduce several improvements to LPRNet, including an improved attention module to reduce confusion among similar characters and the use of Focal Loss to address class imbalance during training. Experimental results demonstrate the exceptional performance of LPTR-AFLNet in rectifying perspective distortion and recognizing double-line license plate images, maintaining high recognition accuracy across various challenging scenarios. Moreover, on lower-mid-range GPUs platform, the method runs in less than 10 milliseconds, indicating its practical efficiency and broad applicability.
>
---
#### [new 041] FW-VTON: Flattening-and-Warping for Person-to-Person Virtual Try-on
- **分类: cs.CV**

- **简介: 该论文提出FW-VTON方法，属于人对人虚拟试穿任务，旨在将不同人身上的衣物迁移到目标人物身上。通过三阶段流程：衣物提取、变形对齐与融合，实现高质量合成。论文还构建了新数据集，解决了该任务中缺乏高质量数据的问题。**

- **链接: [http://arxiv.org/pdf/2507.16010v1](http://arxiv.org/pdf/2507.16010v1)**

> **作者:** Zheng Wang; Xianbing Sun; Shengyi Wu; Jiahui Zhan; Jianlou Si; Chi Zhang; Liqing Zhang; Jianfu Zhang
>
> **摘要:** Traditional virtual try-on methods primarily focus on the garment-to-person try-on task, which requires flat garment representations. In contrast, this paper introduces a novel approach to the person-to-person try-on task. Unlike the garment-to-person try-on task, the person-to-person task only involves two input images: one depicting the target person and the other showing the garment worn by a different individual. The goal is to generate a realistic combination of the target person with the desired garment. To this end, we propose Flattening-and-Warping Virtual Try-On (\textbf{FW-VTON}), a method that operates in three stages: (1) extracting the flattened garment image from the source image; (2) warping the garment to align with the target pose; and (3) integrating the warped garment seamlessly onto the target person. To overcome the challenges posed by the lack of high-quality datasets for this task, we introduce a new dataset specifically designed for person-to-person try-on scenarios. Experimental evaluations demonstrate that FW-VTON achieves state-of-the-art performance, with superior results in both qualitative and quantitative assessments, and also excels in garment extraction subtasks.
>
---
#### [new 042] Universal Wavelet Units in 3D Retinal Layer Segmentation
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于医学图像分割任务，旨在提升视网膜层分割精度。为解决传统下采样方法丢失高频细节的问题，作者将三种可学习小波模块集成到网络中，实现3D OCT图像更精确的分割，尤其LS-BiorthLattUwU效果最佳。**

- **链接: [http://arxiv.org/pdf/2507.16119v1](http://arxiv.org/pdf/2507.16119v1)**

> **作者:** An D. Le; Hung Nguyen; Melanie Tran; Jesse Most; Dirk-Uwe G. Bartsch; William R Freeman; Shyamanga Borooah; Truong Q. Nguyen; Cheolhong An
>
> **摘要:** This paper presents the first study to apply tunable wavelet units (UwUs) for 3D retinal layer segmentation from Optical Coherence Tomography (OCT) volumes. To overcome the limitations of conventional max-pooling, we integrate three wavelet-based downsampling modules, OrthLattUwU, BiorthLattUwU, and LS-BiorthLattUwU, into a motion-corrected MGU-Net architecture. These modules use learnable lattice filter banks to preserve both low- and high-frequency features, enhancing spatial detail and structural consistency. Evaluated on the Jacobs Retina Center (JRC) OCT dataset, our framework shows significant improvement in accuracy and Dice score, particularly with LS-BiorthLattUwU, highlighting the benefits of tunable wavelet filters in volumetric medical image segmentation.
>
---
#### [new 043] Improving Personalized Image Generation through Social Context Feedback
- **分类: cs.CV**

- **简介: 该论文属于个性化图像生成任务，旨在解决生成图像中人物姿态错误、身份未保留、注视不自然等问题。通过引入社交上下文反馈，利用姿态、人-物交互、人脸识别和注视点估计等检测器微调扩散模型，并按信号层次设计反馈模块，提升了生成图像的质量与真实性。**

- **链接: [http://arxiv.org/pdf/2507.16095v1](http://arxiv.org/pdf/2507.16095v1)**

> **作者:** Parul Gupta; Abhinav Dhall; Thanh-Toan Do
>
> **摘要:** Personalized image generation, where reference images of one or more subjects are used to generate their image according to a scene description, has gathered significant interest in the community. However, such generated images suffer from three major limitations -- complex activities, such as $<$man, pushing, motorcycle$>$ are not generated properly with incorrect human poses, reference human identities are not preserved, and generated human gaze patterns are unnatural/inconsistent with the scene description. In this work, we propose to overcome these shortcomings through feedback-based fine-tuning of existing personalized generation methods, wherein, state-of-art detectors of pose, human-object-interaction, human facial recognition and human gaze-point estimation are used to refine the diffusion model. We also propose timestep-based inculcation of different feedback modules, depending upon whether the signal is low-level (such as human pose), or high-level (such as gaze point). The images generated in this manner show an improvement in the generated interactions, facial identities and image quality over three benchmark datasets.
>
---
#### [new 044] Are Foundation Models All You Need for Zero-shot Face Presentation Attack Detection?
- **分类: cs.CV**

- **简介: 该论文属于人脸识别安全任务，旨在解决零样本场景下的人脸呈现攻击检测问题。现有方法依赖大量数据且泛化能力差，作者评估了基础模型在该任务中的有效性，并提出一种简单有效的零样本PAD框架，实验证明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.16393v1](http://arxiv.org/pdf/2507.16393v1)**

> **作者:** Lazaro Janier Gonzalez-Sole; Juan E. Tapia; Christoph Busch
>
> **备注:** Accepted at FG 2025
>
> **摘要:** Although face recognition systems have undergone an impressive evolution in the last decade, these technologies are vulnerable to attack presentations (AP). These attacks are mostly easy to create and, by executing them against the system's capture device, the malicious actor can impersonate an authorised subject and thus gain access to the latter's information (e.g., financial transactions). To protect facial recognition schemes against presentation attacks, state-of-the-art deep learning presentation attack detection (PAD) approaches require a large amount of data to produce reliable detection performances and even then, they decrease their performance for unknown presentation attack instruments (PAI) or database (information not seen during training), i.e. they lack generalisability. To mitigate the above problems, this paper focuses on zero-shot PAD. To do so, we first assess the effectiveness and generalisability of foundation models in established and challenging experimental scenarios and then propose a simple but effective framework for zero-shot PAD. Experimental results show that these models are able to achieve performance in difficult scenarios with minimal effort of the more advanced PAD mechanisms, whose weights were optimised mainly with training sets that included APs and bona fide presentations. The top-performing foundation model outperforms by a margin the best from the state of the art observed with the leaving-one-out protocol on the SiW-Mv2 database, which contains challenging unknown 2D and 3D attacks
>
---
#### [new 045] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文属于视觉问答（VQA）任务，旨在解决现有数据集缺乏复杂推理和外部知识的问题。作者构建了一个新数据集ReasonVQA，集成结构化百科知识，支持多跳推理问题。实验表明，该数据集对现有模型构成挑战，且规模远超同类数据集，有助于推动VQA领域发展。**

- **链接: [http://arxiv.org/pdf/2507.16403v1](http://arxiv.org/pdf/2507.16403v1)**

> **作者:** Thuy-Duong Tran; Trung-Kien Tran; Manfred Hauswirth; Danh Le Phuoc
>
> **备注:** Accepted at the IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** In this paper, we propose a new dataset, ReasonVQA, for the Visual Question Answering (VQA) task. Our dataset is automatically integrated with structured encyclopedic knowledge and constructed using a low-cost framework, which is capable of generating complex, multi-hop questions. We evaluated state-of-the-art VQA models on ReasonVQA, and the empirical results demonstrate that ReasonVQA poses significant challenges to these models, highlighting its potential for benchmarking and advancing the field of VQA. Additionally, our dataset can be easily scaled with respect to input images; the current version surpasses the largest existing datasets requiring external knowledge by more than an order of magnitude.
>
---
#### [new 046] Explicit Context Reasoning with Supervision for Visual Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉跟踪任务，旨在解决跨帧建模中上下文关联不准确导致目标动态建模困难的问题。论文提出了RSTrack方法，通过上下文推理机制、前向监督策略和高效状态建模三个核心机制，显式建模并监督上下文推理过程，提升跟踪的时序一致性和准确性。**

- **链接: [http://arxiv.org/pdf/2507.16191v1](http://arxiv.org/pdf/2507.16191v1)**

> **作者:** Fansheng Zeng; Bineng Zhong; Haiying Xia; Yufei Tan; Xiantao Hu; Liangtao Shi; Shuxiang Song
>
> **摘要:** Contextual reasoning with constraints is crucial for enhancing temporal consistency in cross-frame modeling for visual tracking. However, mainstream tracking algorithms typically associate context by merely stacking historical information without explicitly supervising the association process, making it difficult to effectively model the target's evolving dynamics. To alleviate this problem, we propose RSTrack, which explicitly models and supervises context reasoning via three core mechanisms. \textit{1) Context Reasoning Mechanism}: Constructs a target state reasoning pipeline, converting unconstrained contextual associations into a temporal reasoning process that predicts the current representation based on historical target states, thereby enhancing temporal consistency. \textit{2) Forward Supervision Strategy}: Utilizes true target features as anchors to constrain the reasoning pipeline, guiding the predicted output toward the true target distribution and suppressing drift in the context reasoning process. \textit{3) Efficient State Modeling}: Employs a compression-reconstruction mechanism to extract the core features of the target, removing redundant information across frames and preventing ineffective contextual associations. These three mechanisms collaborate to effectively alleviate the issue of contextual association divergence in traditional temporal modeling. Experimental results show that RSTrack achieves state-of-the-art performance on multiple benchmark datasets while maintaining real-time running speeds. Our code is available at https://github.com/GXNU-ZhongLab/RSTrack.
>
---
#### [new 047] LongSplat: Online Generalizable 3D Gaussian Splatting from Long Sequence Images
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决长序列图像输入下实时、高效的3D高斯重建问题。现有方法存在优化慢或更新效率低的问题。论文提出LongSplat，通过流式更新机制和Gaussian-Image表示实现在线增量融合与冗余压缩，显著提升效率与质量。**

- **链接: [http://arxiv.org/pdf/2507.16144v1](http://arxiv.org/pdf/2507.16144v1)**

> **作者:** Guichen Huang; Ruoyu Wang; Xiangjun Gao; Che Sun; Yuwei Wu; Shenghua Gao; Yunde Jia
>
> **摘要:** 3D Gaussian Splatting achieves high-fidelity novel view synthesis, but its application to online long-sequence scenarios is still limited. Existing methods either rely on slow per-scene optimization or fail to provide efficient incremental updates, hindering continuous performance. In this paper, we propose LongSplat, an online real-time 3D Gaussian reconstruction framework designed for long-sequence image input. The core idea is a streaming update mechanism that incrementally integrates current-view observations while selectively compressing redundant historical Gaussians. Crucial to this mechanism is our Gaussian-Image Representation (GIR), a representation that encodes 3D Gaussian parameters into a structured, image-like 2D format. GIR simultaneously enables efficient fusion of current-view and historical Gaussians and identity-aware redundancy compression. These functions enable online reconstruction and adapt the model to long sequences without overwhelming memory or computational costs. Furthermore, we leverage an existing image compression method to guide the generation of more compact and higher-quality 3D Gaussians. Extensive evaluations demonstrate that LongSplat achieves state-of-the-art efficiency-quality trade-offs in real-time novel view synthesis, delivering real-time reconstruction while reducing Gaussian counts by 44\% compared to existing per-pixel Gaussian prediction methods.
>
---
#### [new 048] Zebra-CoT: A Dataset for Interleaved Vision Language Reasoning
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态推理任务，旨在解决视觉链式推理（Visual CoT）中数据质量差和模型性能低的问题。作者构建了大规模数据集Zebra-CoT，包含18万+图文交错推理样本，并验证其在提升模型视觉推理能力上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.16746v1](http://arxiv.org/pdf/2507.16746v1)**

> **作者:** Ang Li; Charles Wang; Kaiyu Yue; Zikui Cai; Ollie Liu; Deqing Fu; Peng Guo; Wang Bill Zhu; Vatsal Sharan; Robin Jia; Willie Neiswanger; Furong Huang; Tom Goldstein; Micah Goldblum
>
> **备注:** dataset link: https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT
>
> **摘要:** Humans often use visual aids, for example diagrams or sketches, when solving complex problems. Training multimodal models to do the same, known as Visual Chain of Thought (Visual CoT), is challenging due to: (1) poor off-the-shelf visual CoT performance, which hinders reinforcement learning, and (2) the lack of high-quality visual CoT training data. We introduce $\textbf{Zebra-CoT}$, a diverse large-scale dataset with 182,384 samples, containing logically coherent interleaved text-image reasoning traces. We focus on four categories of tasks where sketching or visual reasoning is especially natural, spanning scientific questions such as geometry, physics, and algorithms; 2D visual reasoning tasks like visual search and jigsaw puzzles; 3D reasoning tasks including 3D multi-hop inference, embodied and robot planning; visual logic problems and strategic games like chess. Fine-tuning the Anole-7B model on the Zebra-CoT training corpus results in an improvement of +12% in our test-set accuracy and yields up to +13% performance gain on standard VLM benchmark evaluations. Fine-tuning Bagel-7B yields a model that generates high-quality interleaved visual reasoning chains, underscoring Zebra-CoT's effectiveness for developing multimodal reasoning abilities. We open-source our dataset and models to support development and evaluation of visual CoT.
>
---
#### [new 049] CTSL: Codebook-based Temporal-Spatial Learning for Accurate Non-Contrast Cardiac Risk Prediction Using Cine MRIs
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决无对比剂条件下心脏MRI预测重大心脏事件（MACE）的难题。作者提出一种自监督框架CTSL，通过时空特征解耦和多视角蒸馏，从原始Cine MRI中学习动态表征，实现精准、无创的心脏风险预测。**

- **链接: [http://arxiv.org/pdf/2507.16612v1](http://arxiv.org/pdf/2507.16612v1)**

> **作者:** Haoyang Su; Shaohao Rui; Jinyi Xiang; Lianming Wu; Xiaosong Wang
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** Accurate and contrast-free Major Adverse Cardiac Events (MACE) prediction from Cine MRI sequences remains a critical challenge. Existing methods typically necessitate supervised learning based on human-refined masks in the ventricular myocardium, which become impractical without contrast agents. We introduce a self-supervised framework, namely Codebook-based Temporal-Spatial Learning (CTSL), that learns dynamic, spatiotemporal representations from raw Cine data without requiring segmentation masks. CTSL decouples temporal and spatial features through a multi-view distillation strategy, where the teacher model processes multiple Cine views, and the student model learns from reduced-dimensional Cine-SA sequences. By leveraging codebook-based feature representations and dynamic lesion self-detection through motion cues, CTSL captures intricate temporal dependencies and motion patterns. High-confidence MACE risk predictions are achieved through our model, providing a rapid, non-invasive solution for cardiac risk assessment that outperforms traditional contrast-dependent methods, thereby enabling timely and accessible heart disease diagnosis in clinical settings.
>
---
#### [new 050] HoliTracer: Holistic Vectorization of Geographic Objects from Large-Size Remote Sensing Imagery
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像处理任务，旨在解决大尺寸遥感图像中地理目标矢量化不完整、碎片化的问题。论文提出了HoliTracer框架，通过上下文注意力网络（CAN）增强分割，并利用MCR和PST模块实现整体矢量化，提升了建筑物、水域和道路等地理对象的高精度矢量提取效果。**

- **链接: [http://arxiv.org/pdf/2507.16251v1](http://arxiv.org/pdf/2507.16251v1)**

> **作者:** Yu Wang; Bo Dang; Wanchun Li; Wei Chen; Yansheng Li
>
> **摘要:** With the increasing resolution of remote sensing imagery (RSI), large-size RSI has emerged as a vital data source for high-precision vector mapping of geographic objects. Existing methods are typically constrained to processing small image patches, which often leads to the loss of contextual information and produces fragmented vector outputs. To address these, this paper introduces HoliTracer, the first framework designed to holistically extract vectorized geographic objects from large-size RSI. In HoliTracer, we enhance segmentation of large-size RSI using the Context Attention Net (CAN), which employs a local-to-global attention mechanism to capture contextual dependencies. Furthermore, we achieve holistic vectorization through a robust pipeline that leverages the Mask Contour Reformer (MCR) to reconstruct polygons and the Polygon Sequence Tracer (PST) to trace vertices. Extensive experiments on large-size RSI datasets, including buildings, water bodies, and roads, demonstrate that HoliTracer outperforms state-of-the-art methods. Our code and data are available in https://github.com/vvangfaye/HoliTracer.
>
---
#### [new 051] QRetinex-Net: Quaternion-Valued Retinex Decomposition for Low-Level Computer Vision Applications
- **分类: cs.CV**

- **简介: 该论文属于图像增强任务，旨在解决低光图像的颜色偏差、低对比度和噪声问题。作者提出了QRetinex-Net，采用四元数Retinex理论进行反射率和光照分解，并设计了反射率一致性指标评估稳定性。实验表明其在多个视觉任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.16683v1](http://arxiv.org/pdf/2507.16683v1)**

> **作者:** Sos Agaian; Vladimir Frants
>
> **摘要:** Images taken in low light often show color shift, low contrast, noise, and other artifacts that hurt computer-vision accuracy. Retinex theory addresses this by viewing an image S as the pixel-wise product of reflectance R and illumination I, mirroring the way people perceive stable object colors under changing light. The decomposition is ill-posed, and classic Retinex models have four key flaws: (i) they treat the red, green, and blue channels independently; (ii) they lack a neuroscientific model of color vision; (iii) they cannot perfectly rebuild the input image; and (iv) they do not explain human color constancy. We introduce the first Quaternion Retinex formulation, in which the scene is written as the Hamilton product of quaternion-valued reflectance and illumination. To gauge how well reflectance stays invariant, we propose the Reflectance Consistency Index. Tests on low-light crack inspection, face detection under varied lighting, and infrared-visible fusion show gains of 2-11 percent over leading methods, with better color fidelity, lower noise, and higher reflectance stability.
>
---
#### [new 052] Document Haystack: A Long Context Multimodal Image/Document Understanding Vision LLM Benchmark
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态文档理解任务，旨在解决现有模型在长文档处理上的不足。作者构建了Document Haystack基准，包含长达200页的复杂文档，并插入文本或图文“针”测试模型检索能力，共包含400种文档变体和8250个问题，提供自动化评估框架，用于评估视觉语言模型的长文档理解性能。**

- **链接: [http://arxiv.org/pdf/2507.15882v1](http://arxiv.org/pdf/2507.15882v1)**

> **作者:** Goeric Huybrechts; Srikanth Ronanki; Sai Muralidhar Jayanthi; Jack Fitzgerald; Srinivasan Veeravanallur
>
> **摘要:** The proliferation of multimodal Large Language Models has significantly advanced the ability to analyze and understand complex data inputs from different modalities. However, the processing of long documents remains under-explored, largely due to a lack of suitable benchmarks. To address this, we introduce Document Haystack, a comprehensive benchmark designed to evaluate the performance of Vision Language Models (VLMs) on long, visually complex documents. Document Haystack features documents ranging from 5 to 200 pages and strategically inserts pure text or multimodal text+image "needles" at various depths within the documents to challenge VLMs' retrieval capabilities. Comprising 400 document variants and a total of 8,250 questions, it is supported by an objective, automated evaluation framework. We detail the construction and characteristics of the Document Haystack dataset, present results from prominent VLMs and discuss potential research avenues in this area.
>
---
#### [new 053] Dens3R: A Foundation Model for 3D Geometry Prediction
- **分类: cs.CV**

- **简介: 该论文属于3D几何预测任务，旨在解决现有方法仅能预测单一几何量导致的不一致问题。作者提出了Dens3R模型，通过统一框架联合预测多种几何属性，并设计了轻量网络结构与位置编码，实现单视图到多视图的几何一致性推理，提升了预测准确性与适用性。**

- **链接: [http://arxiv.org/pdf/2507.16290v1](http://arxiv.org/pdf/2507.16290v1)**

> **作者:** Xianze Fang; Jingnan Gao; Zhe Wang; Zhuo Chen; Xingyu Ren; Jiangjing Lyu; Qiaomu Ren; Zhonglei Yang; Xiaokang Yang; Yichao Yan; Chengfei Lyu
>
> **备注:** Project Page: https://g-1nonly.github.io/Dens3R/, Code: https://github.com/G-1nOnly/Dens3R
>
> **摘要:** Recent advances in dense 3D reconstruction have led to significant progress, yet achieving accurate unified geometric prediction remains a major challenge. Most existing methods are limited to predicting a single geometry quantity from input images. However, geometric quantities such as depth, surface normals, and point maps are inherently correlated, and estimating them in isolation often fails to ensure consistency, thereby limiting both accuracy and practical applicability. This motivates us to explore a unified framework that explicitly models the structural coupling among different geometric properties to enable joint regression. In this paper, we present Dens3R, a 3D foundation model designed for joint geometric dense prediction and adaptable to a wide range of downstream tasks. Dens3R adopts a two-stage training framework to progressively build a pointmap representation that is both generalizable and intrinsically invariant. Specifically, we design a lightweight shared encoder-decoder backbone and introduce position-interpolated rotary positional encoding to maintain expressive power while enhancing robustness to high-resolution inputs. By integrating image-pair matching features with intrinsic invariance modeling, Dens3R accurately regresses multiple geometric quantities such as surface normals and depth, achieving consistent geometry perception from single-view to multi-view inputs. Additionally, we propose a post-processing pipeline that supports geometrically consistent multi-view inference. Extensive experiments demonstrate the superior performance of Dens3R across various dense 3D prediction tasks and highlight its potential for broader applications.
>
---
#### [new 054] SPACT18: Spiking Human Action Recognition Benchmark Dataset with Complementary RGB and Thermal Modalities
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频动作识别任务，旨在解决现有数据集在脉冲相机数据上的缺失问题。作者构建了首个包含脉冲、RGB和热成像三种模态的动作识别数据集SPACT18，用于评估脉冲神经网络在多模态视频理解中的性能，推动低功耗视频理解研究。**

- **链接: [http://arxiv.org/pdf/2507.16151v1](http://arxiv.org/pdf/2507.16151v1)**

> **作者:** Yasser Ashraf; Ahmed Sharshar; Velibor Bojkovic; Bin Gu
>
> **摘要:** Spike cameras, bio-inspired vision sensors, asynchronously fire spikes by accumulating light intensities at each pixel, offering ultra-high energy efficiency and exceptional temporal resolution. Unlike event cameras, which record changes in light intensity to capture motion, spike cameras provide even finer spatiotemporal resolution and a more precise representation of continuous changes. In this paper, we introduce the first video action recognition (VAR) dataset using spike camera, alongside synchronized RGB and thermal modalities, to enable comprehensive benchmarking for Spiking Neural Networks (SNNs). By preserving the inherent sparsity and temporal precision of spiking data, our three datasets offer a unique platform for exploring multimodal video understanding and serve as a valuable resource for directly comparing spiking, thermal, and RGB modalities. This work contributes a novel dataset that will drive research in energy-efficient, ultra-low-power video understanding, specifically for action recognition tasks using spike-based data.
>
---
#### [new 055] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于少样本动作识别任务，旨在解决标注数据稀缺下的动作分类问题。论文提出Language-Guided Action Anatomy（LGA）框架，利用大语言模型解析动作标签，结合视觉模块提取时空特征，并通过细粒度融合与多模态匹配提升识别性能。**

- **链接: [http://arxiv.org/pdf/2507.16287v1](http://arxiv.org/pdf/2507.16287v1)**

> **作者:** Zefeng Qian; Xincheng Yao; Yifei Huang; Chongyang Zhang; Jiangyong Ying; Hong Sun
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Few-shot action recognition (FSAR) aims to classify human actions in videos with only a small number of labeled samples per category. The scarcity of training data has driven recent efforts to incorporate additional modalities, particularly text. However, the subtle variations in human posture, motion dynamics, and the object interactions that occur during different phases, are critical inherent knowledge of actions that cannot be fully exploited by action labels alone. In this work, we propose Language-Guided Action Anatomy (LGA), a novel framework that goes beyond label semantics by leveraging Large Language Models (LLMs) to dissect the essential representational characteristics hidden beneath action labels. Guided by the prior knowledge encoded in LLM, LGA effectively captures rich spatiotemporal cues in few-shot scenarios. Specifically, for text, we prompt an off-the-shelf LLM to anatomize labels into sequences of atomic action descriptions, focusing on the three core elements of action (subject, motion, object). For videos, a Visual Anatomy Module segments actions into atomic video phases to capture the sequential structure of actions. A fine-grained fusion strategy then integrates textual and visual features at the atomic level, resulting in more generalizable prototypes. Finally, we introduce a Multimodal Matching mechanism, comprising both video-video and video-text matching, to ensure robust few-shot classification. Experimental results demonstrate that LGA achieves state-of-the-art performance across multipe FSAR benchmarks.
>
---
#### [new 056] Discovering and using Spelke segments
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉与人工智能任务，旨在解决图像中基于物理因果关系的物体分割问题。受发展心理学启发，提出Spelke对象概念，构建SpelkeBench数据集与SpelkeNet模型，通过预测运动分布发现物体。论文实现了更适用于操作与规划的图像分割方法，并验证其在实际应用中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.16038v1](http://arxiv.org/pdf/2507.16038v1)**

> **作者:** Rahul Venkatesh; Klemen Kotar; Lilian Naing Chen; Seungwoo Kim; Luca Thomas Wheeler; Jared Watrous; Ashley Xu; Gia Ancone; Wanhee Lee; Honglin Chen; Daniel Bear; Stefan Stojanov; Daniel Yamins
>
> **备注:** Project page at: https://neuroailab.github.io/spelke_net
>
> **摘要:** Segments in computer vision are often defined by semantic considerations and are highly dependent on category-specific conventions. In contrast, developmental psychology suggests that humans perceive the world in terms of Spelke objects--groupings of physical things that reliably move together when acted on by physical forces. Spelke objects thus operate on category-agnostic causal motion relationships which potentially better support tasks like manipulation and planning. In this paper, we first benchmark the Spelke object concept, introducing the SpelkeBench dataset that contains a wide variety of well-defined Spelke segments in natural images. Next, to extract Spelke segments from images algorithmically, we build SpelkeNet, a class of visual world models trained to predict distributions over future motions. SpelkeNet supports estimation of two key concepts for Spelke object discovery: (1) the motion affordance map, identifying regions likely to move under a poke, and (2) the expected-displacement map, capturing how the rest of the scene will move. These concepts are used for "statistical counterfactual probing", where diverse "virtual pokes" are applied on regions of high motion-affordance, and the resultant expected displacement maps are used define Spelke segments as statistical aggregates of correlated motion statistics. We find that SpelkeNet outperforms supervised baselines like SegmentAnything (SAM) on SpelkeBench. Finally, we show that the Spelke concept is practically useful for downstream applications, yielding superior performance on the 3DEditBench benchmark for physical object manipulation when used in a variety of off-the-shelf object manipulation models.
>
---
#### [new 057] Salience Adjustment for Context-Based Emotion Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于情感识别任务，旨在解决动态社交场景中面部表情与情境线索融合的问题。通过贝叶斯线索整合与视觉-语言模型，提出显著性调整框架，动态加权面部与情境信息。在囚徒困境场景中验证，提升情感识别效果。**

- **链接: [http://arxiv.org/pdf/2507.15878v1](http://arxiv.org/pdf/2507.15878v1)**

> **作者:** Bin Han; Jonathan Gratch
>
> **摘要:** Emotion recognition in dynamic social contexts requires an understanding of the complex interaction between facial expressions and situational cues. This paper presents a salience-adjusted framework for context-aware emotion recognition with Bayesian Cue Integration (BCI) and Visual-Language Models (VLMs) to dynamically weight facial and contextual information based on the expressivity of facial cues. We evaluate this approach using human annotations and automatic emotion recognition systems in prisoner's dilemma scenarios, which are designed to evoke emotional reactions. Our findings demonstrate that incorporating salience adjustment enhances emotion recognition performance, offering promising directions for future research to extend this framework to broader social contexts and multimodal applications.
>
---
#### [new 058] Advancing Visual Large Language Model for Multi-granular Versatile Perception
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉与自然语言处理交叉任务，旨在解决现有模型在多类型感知任务中适用性受限的问题。作者提出了MVP-LM框架，通过多粒度解码器和数据统一策略，实现对多种感知任务的统一建模与优化。**

- **链接: [http://arxiv.org/pdf/2507.16213v1](http://arxiv.org/pdf/2507.16213v1)**

> **作者:** Wentao Xiang; Haoxian Tan; Cong Wei; Yujie Zhong; Dengjie Li; Yujiu Yang
>
> **备注:** To appear in ICCV 2025
>
> **摘要:** Perception is a fundamental task in the field of computer vision, encompassing a diverse set of subtasks that can be systematically categorized into four distinct groups based on two dimensions: prediction type and instruction type. Notably, existing researches often focus solely on a limited subset of these potential combinations, which constrains their applicability and versatility across various contexts. In response to this challenge, we present MVP-LM, a Multi-granular and Versatile Perception framework incorporating Visual Large Language Model. Our framework is designed to integrate both word-based and sentence-based perception tasks alongside box and mask predictions within a single architecture. MVP-LM features an innovative multi-granularity decoder in conjunction with a CoT-inspired dataset unification strategy, enabling seamless supervised fine-tuning across a wide spectrum of tasks, including but not limited to panoptic segmentation, detection, grounding, and referring expression segmentation. Furthermore, we introduce a query enhancement strategy aimed at harnessing the decoding and generative capabilities inherent in VLLMs. Extensive experiments conducted across a range of benchmarks in both word-based and sentence-based perception tasks substantiate the efficacy of our framework. The code will be available at https://github.com/xiangwentao666/MVP-LM.
>
---
#### [new 059] Is Tracking really more challenging in First Person Egocentric Vision?
- **分类: cs.CV**

- **简介: 该论文属于视觉目标跟踪与分割任务，旨在分析第一视角（egocentric）视觉中跟踪与分割的挑战来源。现有方法认为第一视角更具挑战性，但本文指出这可能源于人类与物体交互场景本身的复杂性，而非视角本身。为此，作者构建新基准，区分视角因素与活动域因素，深入探讨真正影响性能的关键问题。**

- **链接: [http://arxiv.org/pdf/2507.16015v1](http://arxiv.org/pdf/2507.16015v1)**

> **作者:** Matteo Dunnhofer; Zaira Manigrasso; Christian Micheloni
>
> **备注:** 2025 IEEE/CVF International Conference on Computer Vision (ICCV)
>
> **摘要:** Visual object tracking and segmentation are becoming fundamental tasks for understanding human activities in egocentric vision. Recent research has benchmarked state-of-the-art methods and concluded that first person egocentric vision presents challenges compared to previously studied domains. However, these claims are based on evaluations conducted across significantly different scenarios. Many of the challenging characteristics attributed to egocentric vision are also present in third person videos of human-object activities. This raises a critical question: how much of the observed performance drop stems from the unique first person viewpoint inherent to egocentric vision versus the domain of human-object activities? To address this question, we introduce a new benchmark study designed to disentangle such factors. Our evaluation strategy enables a more precise separation of challenges related to the first person perspective from those linked to the broader domain of human-object activity understanding. By doing so, we provide deeper insights into the true sources of difficulty in egocentric tracking and segmentation, facilitating more targeted advancements on this task.
>
---
#### [new 060] DenseSR: Image Shadow Removal as Dense Prediction
- **分类: cs.CV**

- **简介: 该论文属于图像阴影去除任务，旨在解决单张图像在复杂光照下阴影导致的细节丢失与边界模糊问题。论文提出DenseSR框架，结合几何语义先验与新型密集融合模块，提升阴影区域的细节恢复与边界清晰度，从而优化整体图像质量。**

- **链接: [http://arxiv.org/pdf/2507.16472v1](http://arxiv.org/pdf/2507.16472v1)**

> **作者:** Yu-Fan Lin; Chia-Ming Lee; Chih-Chung Hsu
>
> **备注:** Paper accepted to ACMMM 2025
>
> **摘要:** Shadows are a common factor degrading image quality. Single-image shadow removal (SR), particularly under challenging indirect illumination, is hampered by non-uniform content degradation and inherent ambiguity. Consequently, traditional methods often fail to simultaneously recover intra-shadow details and maintain sharp boundaries, resulting in inconsistent restoration and blurring that negatively affect both downstream applications and the overall viewing experience. To overcome these limitations, we propose the DenseSR, approaching the problem from a dense prediction perspective to emphasize restoration quality. This framework uniquely synergizes two key strategies: (1) deep scene understanding guided by geometric-semantic priors to resolve ambiguity and implicitly localize shadows, and (2) high-fidelity restoration via a novel Dense Fusion Block (DFB) in the decoder. The DFB employs adaptive component processing-using an Adaptive Content Smoothing Module (ACSM) for consistent appearance and a Texture-Boundary Recuperation Module (TBRM) for fine textures and sharp boundaries-thereby directly tackling the inconsistent restoration and blurring issues. These purposefully processed components are effectively fused, yielding an optimized feature representation preserving both consistency and fidelity. Extensive experimental results demonstrate the merits of our approach over existing methods. Our code can be available on https://github$.$com/VanLinLin/DenseSR
>
---
#### [new 061] Spatial 3D-LLM: Exploring Spatial Awareness in 3D Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D视觉-语言任务，旨在提升模型的空间感知能力。现有方法因对3D场景表示不足而限制了空间理解。作者提出了Spatial 3D-LLM，通过渐进式空间感知机制增强空间嵌入，并引入新任务和数据集评估性能。实验表明其方法在多个任务上达到最优表现。**

- **链接: [http://arxiv.org/pdf/2507.16524v1](http://arxiv.org/pdf/2507.16524v1)**

> **作者:** Xiaoyan Wang; Zeju Li; Yifan Xu; Jiaxing Qi; Zhifei Yang; Ruifei Ma; Xiangde Liu; Chao Zhang
>
> **备注:** Accepted by ICME2025
>
> **摘要:** New era has unlocked exciting possibilities for extending Large Language Models (LLMs) to tackle 3D vision-language tasks. However, most existing 3D multimodal LLMs (MLLMs) rely on compressing holistic 3D scene information or segmenting independent objects to perform these tasks, which limits their spatial awareness due to insufficient representation of the richness inherent in 3D scenes. To overcome these limitations, we propose Spatial 3D-LLM, a 3D MLLM specifically designed to enhance spatial awareness for 3D vision-language tasks by enriching the spatial embeddings of 3D scenes. Spatial 3D-LLM integrates an LLM backbone with a progressive spatial awareness scheme that progressively captures spatial information as the perception field expands, generating location-enriched 3D scene embeddings to serve as visual prompts. Furthermore, we introduce two novel tasks: 3D object distance measurement and 3D layout editing, and construct a 3D instruction dataset, MODEL, to evaluate the model's spatial awareness capabilities. Experimental results demonstrate that Spatial 3D-LLM achieves state-of-the-art performance across a wide range of 3D vision-language tasks, revealing the improvements stemmed from our progressive spatial awareness scheme of mining more profound spatial information. Our code is available at https://github.com/bjshuyuan/Spatial-3D-LLM.
>
---
#### [new 062] Enhancing Domain Diversity in Synthetic Data Face Recognition with Dataset Fusion
- **分类: cs.CV**

- **简介: 该论文属于人脸识别任务，旨在解决合成数据训练模型性能不足的问题。通过融合两种不同生成模型的合成数据集，减少生成模型偏差，提升数据多样性，从而提高人脸识别模型的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.16790v1](http://arxiv.org/pdf/2507.16790v1)**

> **作者:** Anjith George; Sebastien Marcel
>
> **备注:** Accepted in ICCV Workshops 2025
>
> **摘要:** While the accuracy of face recognition systems has improved significantly in recent years, the datasets used to train these models are often collected through web crawling without the explicit consent of users, raising ethical and privacy concerns. To address this, many recent approaches have explored the use of synthetic data for training face recognition models. However, these models typically underperform compared to those trained on real-world data. A common limitation is that a single generator model is often used to create the entire synthetic dataset, leading to model-specific artifacts that may cause overfitting to the generator's inherent biases and artifacts. In this work, we propose a solution by combining two state-of-the-art synthetic face datasets generated using architecturally distinct backbones. This fusion reduces model-specific artifacts, enhances diversity in pose, lighting, and demographics, and implicitly regularizes the face recognition model by emphasizing identity-relevant features. We evaluate the performance of models trained on this combined dataset using standard face recognition benchmarks and demonstrate that our approach achieves superior performance across many of these benchmarks.
>
---
#### [new 063] HOComp: Interaction-Aware Human-Object Composition
- **分类: cs.CV**

- **简介: 该论文属于图像合成任务，旨在解决人与物体交互场景下的自然合成问题。现有方法在处理人与前景物体的交互时效果不佳。为此，作者提出了HOComp方法，包含区域姿态引导和外观保持设计，并构建了新数据集IHOC。实验表明其在生成自然人-物交互图像上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.16813v1](http://arxiv.org/pdf/2507.16813v1)**

> **作者:** Dong Liang; Jinyuan Jia; Yuhao Liu; Rynson W. H. Lau
>
> **摘要:** While existing image-guided composition methods may help insert a foreground object onto a user-specified region of a background image, achieving natural blending inside the region with the rest of the image unchanged, we observe that these existing methods often struggle in synthesizing seamless interaction-aware compositions when the task involves human-object interactions. In this paper, we first propose HOComp, a novel approach for compositing a foreground object onto a human-centric background image, while ensuring harmonious interactions between the foreground object and the background person and their consistent appearances. Our approach includes two key designs: (1) MLLMs-driven Region-based Pose Guidance (MRPG), which utilizes MLLMs to identify the interaction region as well as the interaction type (e.g., holding and lefting) to provide coarse-to-fine constraints to the generated pose for the interaction while incorporating human pose landmarks to track action variations and enforcing fine-grained pose constraints; and (2) Detail-Consistent Appearance Preservation (DCAP), which unifies a shape-aware attention modulation mechanism, a multi-view appearance loss, and a background consistency loss to ensure consistent shapes/textures of the foreground and faithful reproduction of the background human. We then propose the first dataset, named Interaction-aware Human-Object Composition (IHOC), for the task. Experimental results on our dataset show that HOComp effectively generates harmonious human-object interactions with consistent appearances, and outperforms relevant methods qualitatively and quantitatively.
>
---
#### [new 064] PUSA V1.0: Surpassing Wan-I2V with $500 Training Cost by Vectorized Timestep Adaptation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决传统视频扩散模型在时间建模上的局限性，特别是帧演化同步问题。作者提出Pusa方法，采用向量化时间步适应（VTA），实现细粒度时间控制，并通过微调SOTA模型Wan2.1-T2V-14B，显著降低训练成本和数据需求，同时提升性能并支持多种零样本任务。**

- **链接: [http://arxiv.org/pdf/2507.16116v1](http://arxiv.org/pdf/2507.16116v1)**

> **作者:** Yaofang Liu; Yumeng Ren; Aitor Artola; Yuxuan Hu; Xiaodong Cun; Xiaotong Zhao; Alan Zhao; Raymond H. Chan; Suiyun Zhang; Rui Liu; Dandan Tu; Jean-Michel Morel
>
> **备注:** Code is open-sourced at https://github.com/Yaofang-Liu/Pusa-VidGen
>
> **摘要:** The rapid advancement of video diffusion models has been hindered by fundamental limitations in temporal modeling, particularly the rigid synchronization of frame evolution imposed by conventional scalar timestep variables. While task-specific adaptations and autoregressive models have sought to address these challenges, they remain constrained by computational inefficiency, catastrophic forgetting, or narrow applicability. In this work, we present Pusa, a groundbreaking paradigm that leverages vectorized timestep adaptation (VTA) to enable fine-grained temporal control within a unified video diffusion framework. Besides, VTA is a non-destructive adaptation, which means it fully preserves the capabilities of the base model. By finetuning the SOTA Wan2.1-T2V-14B model with VTA, we achieve unprecedented efficiency -- surpassing the performance of Wan-I2V-14B with $\leq$ 1/200 of the training cost (\$500 vs. $\geq$ \$100,000) and $\leq$ 1/2500 of the dataset size (4K vs. $\geq$ 10M samples). Pusa not only sets a new standard for image-to-video (I2V) generation, achieving a VBench-I2V total score of 87.32\% (vs. 86.86\% of Wan-I2V-14B), but also unlocks many zero-shot multi-task capabilities such as start-end frames and video extension -- all without task-specific training. Meanwhile, Pusa can still perform text-to-video generation. Mechanistic analyses reveal that our approach preserves the foundation model's generative priors while surgically injecting temporal dynamics, avoiding the combinatorial explosion inherent to vectorized timesteps. This work establishes a scalable, efficient, and versatile paradigm for next-generation video synthesis, democratizing high-fidelity video generation for research and industry alike. Code is open-sourced at https://github.com/Yaofang-Liu/Pusa-VidGen
>
---
#### [new 065] ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement
- **分类: cs.CV**

- **简介: 该论文属于文档图像篡改定位任务，旨在解决文档图像中篡改区域难以检测的问题。论文提出ACDC-Net模型，通过自适应DCT特征和分层内容解耦方法，提升在各种失真下的检测鲁棒性和准确性。**

- **链接: [http://arxiv.org/pdf/2507.16397v1](http://arxiv.org/pdf/2507.16397v1)**

> **作者:** Kahim Wong; Jicheng Zhou; Haiwei Wu; Yain-Whar Si; Jiantao Zhou
>
> **摘要:** The advancement of image editing tools has enabled malicious manipulation of sensitive document images, underscoring the need for robust document image forgery detection.Though forgery detectors for natural images have been extensively studied, they struggle with document images, as the tampered regions can be seamlessly blended into the uniform document background (BG) and structured text. On the other hand, existing document-specific methods lack sufficient robustness against various degradations, which limits their practical deployment. This paper presents ADCD-Net, a robust document forgery localization model that adaptively leverages the RGB/DCT forensic traces and integrates key characteristics of document images. Specifically, to address the DCT traces' sensitivity to block misalignment, we adaptively modulate the DCT feature contribution based on a predicted alignment score, resulting in much improved resilience to various distortions, including resizing and cropping. Also, a hierarchical content disentanglement approach is proposed to boost the localization performance via mitigating the text-BG disparities. Furthermore, noticing the predominantly pristine nature of BG regions, we construct a pristine prototype capturing traces of untampered regions, and eventually enhance both the localization accuracy and robustness. Our proposed ADCD-Net demonstrates superior forgery localization performance, consistently outperforming state-of-the-art methods by 20.79\% averaged over 5 types of distortions. The code is available at https://github.com/KAHIMWONG/ACDC-Net.
>
---
#### [new 066] PlantSAM: An Object Detection-Driven Segmentation Pipeline for Herbarium Specimens
- **分类: cs.CV**

- **简介: 该论文属于图像分割与分类任务，旨在解决植物标本图像因背景复杂导致分类模型性能下降的问题。作者提出PlantSAM，结合YOLOv10检测与SAM2分割模型，提升分割精度，并通过去除背景噪声增强分类效果，验证了其在多个植物特征分类任务中的性能提升。**

- **链接: [http://arxiv.org/pdf/2507.16506v1](http://arxiv.org/pdf/2507.16506v1)**

> **作者:** Youcef Sklab; Florian Castanet; Hanane Ariouat; Souhila Arib; Jean-Daniel Zucker; Eric Chenin; Edi Prifti
>
> **备注:** 19 pages, 11 figures, 8 tables
>
> **摘要:** Deep learning-based classification of herbarium images is hampered by background heterogeneity, which introduces noise and artifacts that can potentially mislead models and reduce classification accuracy. Addressing these background-related challenges is critical to improving model performance. We introduce PlantSAM, an automated segmentation pipeline that integrates YOLOv10 for plant region detection and the Segment Anything Model (SAM2) for segmentation. YOLOv10 generates bounding box prompts to guide SAM2, enhancing segmentation accuracy. Both models were fine-tuned on herbarium images and evaluated using Intersection over Union (IoU) and Dice coefficient metrics. PlantSAM achieved state-of-the-art segmentation performance, with an IoU of 0.94 and a Dice coefficient of 0.97. Incorporating segmented images into classification models led to consistent performance improvements across five tested botanical traits, with accuracy gains of up to 4.36% and F1-score improvements of 4.15%. Our findings highlight the importance of background removal in herbarium image analysis, as it significantly enhances classification accuracy by allowing models to focus more effectively on the foreground plant structures.
>
---
#### [new 067] Sparse-View 3D Reconstruction: Recent Advances and Open Challenges
- **分类: cs.CV**

- **简介: 该论文聚焦稀疏视角下的三维重建任务，旨在解决因图像稀疏导致的传统方法失效问题。论文综述了基于神经隐式模型、点云显式建模及结合扩散模型与视觉基础模型的混合方法，分析了几何正则化、形状建模与生成推理对缓解重建伪影的作用，并探讨了重建精度、效率与泛化能力间的权衡。**

- **链接: [http://arxiv.org/pdf/2507.16406v1](http://arxiv.org/pdf/2507.16406v1)**

> **作者:** Tanveer Younis; Zhanglin Cheng
>
> **备注:** 30 pages, 6 figures
>
> **摘要:** Sparse-view 3D reconstruction is essential for applications in which dense image acquisition is impractical, such as robotics, augmented/virtual reality (AR/VR), and autonomous systems. In these settings, minimal image overlap prevents reliable correspondence matching, causing traditional methods, such as structure-from-motion (SfM) and multiview stereo (MVS), to fail. This survey reviews the latest advances in neural implicit models (e.g., NeRF and its regularized versions), explicit point-cloud-based approaches (e.g., 3D Gaussian Splatting), and hybrid frameworks that leverage priors from diffusion and vision foundation models (VFMs).We analyze how geometric regularization, explicit shape modeling, and generative inference are used to mitigate artifacts such as floaters and pose ambiguities in sparse-view settings. Comparative results on standard benchmarks reveal key trade-offs between the reconstruction accuracy, efficiency, and generalization. Unlike previous reviews, our survey provides a unified perspective on geometry-based, neural implicit, and generative (diffusion-based) methods. We highlight the persistent challenges in domain generalization and pose-free reconstruction and outline future directions for developing 3D-native generative priors and achieving real-time, unconstrained sparse-view reconstruction.
>
---
#### [new 068] Robust Noisy Pseudo-label Learning for Semi-supervised Medical Image Segmentation Using Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决半监督学习中伪标签噪声影响模型性能的问题。作者提出一种基于扩散模型的框架，通过引入原型对比一致性约束，优化潜在空间中的语义分布结构，提升在噪声伪标签下的分割鲁棒性，并发布了一个新的X射线血管造影视频多目标分割数据集MOSXAV。**

- **链接: [http://arxiv.org/pdf/2507.16429v1](http://arxiv.org/pdf/2507.16429v1)**

> **作者:** Lin Xi; Yingliang Ma; Cheng Wang; Sandra Howell; Aldo Rinaldi; Kawal S. Rhode
>
> **摘要:** Obtaining pixel-level annotations in the medical domain is both expensive and time-consuming, often requiring close collaboration between clinical experts and developers. Semi-supervised medical image segmentation aims to leverage limited annotated data alongside abundant unlabeled data to achieve accurate segmentation. However, existing semi-supervised methods often struggle to structure semantic distributions in the latent space due to noise introduced by pseudo-labels. In this paper, we propose a novel diffusion-based framework for semi-supervised medical image segmentation. Our method introduces a constraint into the latent structure of semantic labels during the denoising diffusion process by enforcing prototype-based contrastive consistency. Rather than explicitly delineating semantic boundaries, the model leverages class prototypes centralized semantic representations in the latent space as anchors. This strategy improves the robustness of dense predictions, particularly in the presence of noisy pseudo-labels. We also introduce a new publicly available benchmark: Multi-Object Segmentation in X-ray Angiography Videos (MOSXAV), which provides detailed, manually annotated segmentation ground truth for multiple anatomical structures in X-ray angiography videos. Extensive experiments on the EndoScapes2023 and MOSXAV datasets demonstrate that our method outperforms state-of-the-art medical image segmentation approaches under the semi-supervised learning setting. This work presents a robust and data-efficient diffusion model that offers enhanced flexibility and strong potential for a wide range of clinical applications.
>
---
#### [new 069] Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的对抗鲁棒性提升任务。它旨在解决现有对抗训练方法忽视语言在增强视觉鲁棒性中的作用问题。论文提出QT-AFT方法，利用高质量文本引导对抗训练，使模型更稳健地识别图像特征，提高零样本任务的对抗鲁棒性和准确性。**

- **链接: [http://arxiv.org/pdf/2507.16257v1](http://arxiv.org/pdf/2507.16257v1)**

> **作者:** Futa Waseda; Saku Sugawara; Isao Echizen
>
> **备注:** ACMMM 2025 Accepted
>
> **摘要:** Defending pre-trained vision-language models (VLMs), such as CLIP, against adversarial attacks is crucial, as these models are widely used in diverse zero-shot tasks, including image classification. However, existing adversarial training (AT) methods for robust fine-tuning largely overlook the role of language in enhancing visual robustness. Specifically, (1) supervised AT methods rely on short texts (e.g., class labels) to generate adversarial perturbations, leading to overfitting to object classes in the training data, and (2) unsupervised AT avoids this overfitting but remains suboptimal against practical text-guided adversarial attacks due to its lack of semantic guidance. To address these limitations, we propose Quality Text-guided Adversarial Fine-Tuning (QT-AFT), which leverages high-quality captions during training to guide adversarial examples away from diverse semantics present in images. This enables the visual encoder to robustly recognize a broader range of image features even under adversarial noise, thereby enhancing robustness across diverse downstream tasks. QT-AFT overcomes the key weaknesses of prior methods -- overfitting in supervised AT and lack of semantic awareness in unsupervised AT -- achieving state-of-the-art zero-shot adversarial robustness and clean accuracy, evaluated across 16 zero-shot datasets. Furthermore, our comprehensive study uncovers several key insights into the role of language in enhancing vision robustness; for example, describing object properties in addition to object names further enhances zero-shot robustness. Our findings point to an urgent direction for future work -- centering high-quality linguistic supervision in robust visual representation learning.
>
---
#### [new 070] A Single-step Accurate Fingerprint Registration Method Based on Local Feature Matching
- **分类: cs.CV**

- **简介: 该论文属于指纹识别任务，旨在解决指纹图像因形变导致的识别性能下降问题。现有方法因依赖两步注册（细节点初注册和密集点精注册）易在低质量图像上失败。论文提出一种端到端的单步指纹注册算法，通过直接预测半密集匹配点实现精确对齐，避免初注册失败，并结合全局-局部注意力机制提升匹配性能。实验表明该方法在单步注册中达到最优性能，还可与密集注册结合进一步提升效果。**

- **链接: [http://arxiv.org/pdf/2507.16201v1](http://arxiv.org/pdf/2507.16201v1)**

> **作者:** Yuwei Jia; Zhe Cui; Fei Su
>
> **摘要:** Distortion of the fingerprint images leads to a decline in fingerprint recognition performance, and fingerprint registration can mitigate this distortion issue by accurately aligning two fingerprint images. Currently, fingerprint registration methods often consist of two steps: an initial registration based on minutiae, and a dense registration based on matching points. However, when the quality of fingerprint image is low, the number of detected minutiae is reduced, leading to frequent failures in the initial registration, which ultimately causes the entire fingerprint registration process to fail. In this study, we propose an end-to-end single-step fingerprint registration algorithm that aligns two fingerprints by directly predicting the semi-dense matching points correspondences between two fingerprints. Thus, our method minimizes the risk of minutiae registration failure and also leverages global-local attentions to achieve end-to-end pixel-level alignment between the two fingerprints. Experiment results prove that our method can achieve the state-of-the-art matching performance with only single-step registration, and it can also be used in conjunction with dense registration algorithms for further performance improvements.
>
---
#### [new 071] A Lightweight Face Quality Assessment Framework to Improve Face Verification Performance in Real-Time Screening Applications
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人脸识别任务，旨在解决低质量人脸图像影响识别准确率的问题。通过构建一个轻量级的人脸质量评估框架，利用归一化面部关键点和随机森林回归分类器筛选低质量图像，在提升ArcFace模型验证性能的同时，有效降低了拒真率，并解决了实时监控中常见的分辨率和姿态变化问题。**

- **链接: [http://arxiv.org/pdf/2507.15961v1](http://arxiv.org/pdf/2507.15961v1)**

> **作者:** Ahmed Aman Ibrahim; Hamad Mansour Alawar; Abdulnasser Abbas Zehi; Ahmed Mohammad Alkendi; Bilal Shafi Ashfaq Ahmed Mirza; Shan Ullah; Ismail Lujain Jaleel; Hassan Ugail
>
> **摘要:** Face image quality plays a critical role in determining the accuracy and reliability of face verification systems, particularly in real-time screening applications such as surveillance, identity verification, and access control. Low-quality face images, often caused by factors such as motion blur, poor lighting conditions, occlusions, and extreme pose variations, significantly degrade the performance of face recognition models, leading to higher false rejection and false acceptance rates. In this work, we propose a lightweight yet effective framework for automatic face quality assessment, which aims to pre-filter low-quality face images before they are passed to the verification pipeline. Our approach utilises normalised facial landmarks in conjunction with a Random Forest Regression classifier to assess image quality, achieving an accuracy of 96.67\%. By integrating this quality assessment module into the face verification process, we observe a substantial improvement in performance, including a comfortable 99.7\% reduction in the false rejection rate and enhanced cosine similarity scores when paired with the ArcFace face verification model. To validate our approach, we have conducted experiments on a real-world dataset collected comprising over 600 subjects captured from CCTV footage in unconstrained environments within Dubai Police. Our results demonstrate that the proposed framework effectively mitigates the impact of poor-quality face images, outperforming existing face quality assessment techniques while maintaining computational efficiency. Moreover, the framework specifically addresses two critical challenges in real-time screening: variations in face resolution and pose deviations, both of which are prevalent in practical surveillance scenarios.
>
---
#### [new 072] LSSGen: Leveraging Latent Space Scaling in Flow and Diffusion for Efficient Text to Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决传统分辨率缩放方法在像素空间操作导致的图像质量下降问题。作者提出LSSGen框架，通过在潜在空间直接进行分辨率缩放，提升生成效率与视觉质量，实现了更优的多分辨率图像生成效果。**

- **链接: [http://arxiv.org/pdf/2507.16154v1](http://arxiv.org/pdf/2507.16154v1)**

> **作者:** Jyun-Ze Tang; Chih-Fan Hsu; Jeng-Lin Li; Ming-Ching Chang; Wei-Chao Chen
>
> **备注:** ICCV AIGENS 2025
>
> **摘要:** Flow matching and diffusion models have shown impressive results in text-to-image generation, producing photorealistic images through an iterative denoising process. A common strategy to speed up synthesis is to perform early denoising at lower resolutions. However, traditional methods that downscale and upscale in pixel space often introduce artifacts and distortions. These issues arise when the upscaled images are re-encoded into the latent space, leading to degraded final image quality. To address this, we propose {\bf Latent Space Scaling Generation (LSSGen)}, a framework that performs resolution scaling directly in the latent space using a lightweight latent upsampler. Without altering the Transformer or U-Net architecture, LSSGen improves both efficiency and visual quality while supporting flexible multi-resolution generation. Our comprehensive evaluation covering text-image alignment and perceptual quality shows that LSSGen significantly outperforms conventional scaling approaches. When generating $1024^2$ images at similar speeds, it achieves up to 246\% TOPIQ score improvement.
>
---
#### [new 073] Temporally-Constrained Video Reasoning Segmentation and Automated Benchmark Construction
- **分类: cs.CV**

- **简介: 该论文属于视频推理分割任务，旨在解决现有方法无法处理时序上下文中目标对象动态变化的问题。作者提出了时序约束的视频推理分割新任务，并构建了自动化基准测试方法及相应数据集TCVideoRSBenchmark。**

- **链接: [http://arxiv.org/pdf/2507.16718v1](http://arxiv.org/pdf/2507.16718v1)**

> **作者:** Yiqing Shen; Chenjia Li; Chenxiao Fan; Mathias Unberath
>
> **摘要:** Conventional approaches to video segmentation are confined to predefined object categories and cannot identify out-of-vocabulary objects, let alone objects that are not identified explicitly but only referred to implicitly in complex text queries. This shortcoming limits the utility for video segmentation in complex and variable scenarios, where a closed set of object categories is difficult to define and where users may not know the exact object category that will appear in the video. Such scenarios can arise in operating room video analysis, where different health systems may use different workflows and instrumentation, requiring flexible solutions for video analysis. Reasoning segmentation (RS) now offers promise towards such a solution, enabling natural language text queries as interaction for identifying object to segment. However, existing video RS formulation assume that target objects remain contextually relevant throughout entire video sequences. This assumption is inadequate for real-world scenarios in which objects of interest appear, disappear or change relevance dynamically based on temporal context, such as surgical instruments that become relevant only during specific procedural phases or anatomical structures that gain importance at particular moments during surgery. Our first contribution is the introduction of temporally-constrained video reasoning segmentation, a novel task formulation that requires models to implicitly infer when target objects become contextually relevant based on text queries that incorporate temporal reasoning. Since manual annotation of temporally-constrained video RS datasets would be expensive and limit scalability, our second contribution is an innovative automated benchmark construction method. Finally, we present TCVideoRSBenchmark, a temporally-constrained video RS dataset containing 52 samples using the videos from the MVOR dataset.
>
---
#### [new 074] M-SpecGene: Generalized Foundation Model for RGBT Multispectral Vision
- **分类: cs.CV**

- **简介: 该论文属于RGBT多光谱视觉任务，旨在解决传统方法依赖人工定制模型导致的归纳偏置、模态偏置和数据瓶颈问题。作者提出了M-SpecGene，一种通用的自监督多光谱基础模型，并引入CMSS度量与GMM-CMSS策略以优化预训练过程，实现跨任务泛化。**

- **链接: [http://arxiv.org/pdf/2507.16318v1](http://arxiv.org/pdf/2507.16318v1)**

> **作者:** Kailai Zhou; Fuqiang Yang; Shixian Wang; Bihan Wen; Chongde Zi; Linsen Chen; Qiu Shen; Xun Cao
>
> **备注:** accepted by ICCV2025
>
> **摘要:** RGB-Thermal (RGBT) multispectral vision is essential for robust perception in complex environments. Most RGBT tasks follow a case-by-case research paradigm, relying on manually customized models to learn task-oriented representations. Nevertheless, this paradigm is inherently constrained by artificial inductive bias, modality bias, and data bottleneck. To address these limitations, we make the initial attempt to build a Generalized RGBT MultiSpectral foundation model (M-SpecGene), which aims to learn modality-invariant representations from large-scale broad data in a self-supervised manner. M-SpecGene provides new insights into multispectral fusion and integrates prior case-by-case studies into a unified paradigm. Considering the unique characteristic of information imbalance in RGBT data, we introduce the Cross-Modality Structural Sparsity (CMSS) metric to quantify the information density across two modalities. Then we develop the GMM-CMSS progressive masking strategy to facilitate a flexible, easy-to-hard, and object-centric pre-training process. Comprehensive experiments validate M-SpecGene's generalizability across eleven datasets for four RGBT downstream tasks. The code will be available at https://github.com/CalayZhou/M-SpecGene.
>
---
#### [new 075] HarmonPaint: Harmonized Training-Free Diffusion Inpainting
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决现有方法需大量训练且难以保持结构与风格连贯的问题。论文提出HarmonPaint，一种无需训练的扩散模型修复框架，利用注意力机制和扩散模型特性，实现结构保真与风格融合，无需训练即可高质量修复图像。**

- **链接: [http://arxiv.org/pdf/2507.16732v1](http://arxiv.org/pdf/2507.16732v1)**

> **作者:** Ying Li; Xinzhe Li; Yong Du; Yangyang Xu; Junyu Dong; Shengfeng He
>
> **摘要:** Existing inpainting methods often require extensive retraining or fine-tuning to integrate new content seamlessly, yet they struggle to maintain coherence in both structure and style between inpainted regions and the surrounding background. Motivated by these limitations, we introduce HarmonPaint, a training-free inpainting framework that seamlessly integrates with the attention mechanisms of diffusion models to achieve high-quality, harmonized image inpainting without any form of training. By leveraging masking strategies within self-attention, HarmonPaint ensures structural fidelity without model retraining or fine-tuning. Additionally, we exploit intrinsic diffusion model properties to transfer style information from unmasked to masked regions, achieving a harmonious integration of styles. Extensive experiments demonstrate the effectiveness of HarmonPaint across diverse scenes and styles, validating its versatility and performance.
>
---
#### [new 076] VGGT-Long: Chunk it, Loop it, Align it -- Pushing VGGT's Limits on Kilometer-scale Long RGB Sequences
- **分类: cs.CV**

- **简介: 该论文属于3D视觉与自动驾驶任务，旨在解决大规模RGB视频流的3D重建难题。现有方法受限于内存，难以处理千米级长序列。论文提出VGGT-Long，采用分块处理、重叠对齐和轻量回环优化，在无需相机标定、深度监督或模型微调的前提下，实现高效稳定的长序列3D重建。**

- **链接: [http://arxiv.org/pdf/2507.16443v1](http://arxiv.org/pdf/2507.16443v1)**

> **作者:** Kai Deng; Zexin Ti; Jiawei Xu; Jian Yang; Jin Xie
>
> **摘要:** Foundation models for 3D vision have recently demonstrated remarkable capabilities in 3D perception. However, extending these models to large-scale RGB stream 3D reconstruction remains challenging due to memory limitations. In this work, we propose VGGT-Long, a simple yet effective system that pushes the limits of monocular 3D reconstruction to kilometer-scale, unbounded outdoor environments. Our approach addresses the scalability bottlenecks of existing models through a chunk-based processing strategy combined with overlapping alignment and lightweight loop closure optimization. Without requiring camera calibration, depth supervision or model retraining, VGGT-Long achieves trajectory and reconstruction performance comparable to traditional methods. We evaluate our method on KITTI, Waymo, and Virtual KITTI datasets. VGGT-Long not only runs successfully on long RGB sequences where foundation models typically fail, but also produces accurate and consistent geometry across various conditions. Our results highlight the potential of leveraging foundation models for scalable monocular 3D scene in real-world settings, especially for autonomous driving scenarios. Code is available at https://github.com/DengKaiCQ/VGGT-Long.
>
---
#### [new 077] EarthCrafter: Scalable 3D Earth Generation via Dual-Sparse Latent Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D地理生成任务，旨在解决大规模真实地球表面建模难题。作者构建了最大航空3D数据集Aerial-Earth3D，并提出EarthCrafter框架，采用双稀疏潜扩散模型分离结构与纹理生成，实现高效大范围3D地球合成，支持语义引导与无条件地形生成。**

- **链接: [http://arxiv.org/pdf/2507.16535v1](http://arxiv.org/pdf/2507.16535v1)**

> **作者:** Shang Liu; Chenjie Cao; Chaohui Yu; Wen Qian; Jing Wang; Fan Wang
>
> **摘要:** Despite the remarkable developments achieved by recent 3D generation works, scaling these methods to geographic extents, such as modeling thousands of square kilometers of Earth's surface, remains an open challenge. We address this through a dual innovation in data infrastructure and model architecture. First, we introduce Aerial-Earth3D, the largest 3D aerial dataset to date, consisting of 50k curated scenes (each measuring 600m x 600m) captured across the U.S. mainland, comprising 45M multi-view Google Earth frames. Each scene provides pose-annotated multi-view images, depth maps, normals, semantic segmentation, and camera poses, with explicit quality control to ensure terrain diversity. Building on this foundation, we propose EarthCrafter, a tailored framework for large-scale 3D Earth generation via sparse-decoupled latent diffusion. Our architecture separates structural and textural generation: 1) Dual sparse 3D-VAEs compress high-resolution geometric voxels and textural 2D Gaussian Splats (2DGS) into compact latent spaces, largely alleviating the costly computation suffering from vast geographic scales while preserving critical information. 2) We propose condition-aware flow matching models trained on mixed inputs (semantics, images, or neither) to flexibly model latent geometry and texture features independently. Extensive experiments demonstrate that EarthCrafter performs substantially better in extremely large-scale generation. The framework further supports versatile applications, from semantic-guided urban layout generation to unconditional terrain synthesis, while maintaining geographic plausibility through our rich data priors from Aerial-Earth3D.
>
---
#### [new 078] Towards Railway Domain Adaptation for LiDAR-based 3D Detection: Road-to-Rail and Sim-to-Real via SynDRA-BBox
- **分类: cs.CV; cs.ET**

- **简介: 该论文属于3D目标检测任务，旨在解决铁路领域缺乏真实标注数据的问题。作者提出了首个针对铁路场景的合成数据集SynDRA-BBox，并结合半监督域适应方法，实现将合成数据迁移到真实铁路环境中的3D检测，提升了感知能力。**

- **链接: [http://arxiv.org/pdf/2507.16413v1](http://arxiv.org/pdf/2507.16413v1)**

> **作者:** Xavier Diaz; Gianluca D'Amico; Raul Dominguez-Sanchez; Federico Nesti; Max Ronecker; Giorgio Buttazzo
>
> **备注:** IEEE International Conference on Intelligent Rail Transportation (ICIRT) 2025
>
> **摘要:** In recent years, interest in automatic train operations has significantly increased. To enable advanced functionalities, robust vision-based algorithms are essential for perceiving and understanding the surrounding environment. However, the railway sector suffers from a lack of publicly available real-world annotated datasets, making it challenging to test and validate new perception solutions in this domain. To address this gap, we introduce SynDRA-BBox, a synthetic dataset designed to support object detection and other vision-based tasks in realistic railway scenarios. To the best of our knowledge, is the first synthetic dataset specifically tailored for 2D and 3D object detection in the railway domain, the dataset is publicly available at https://syndra.retis.santannapisa.it. In the presented evaluation, a state-of-the-art semi-supervised domain adaptation method, originally developed for automotive perception, is adapted to the railway context, enabling the transferability of synthetic data to 3D object detection. Experimental results demonstrate promising performance, highlighting the effectiveness of synthetic datasets and domain adaptation techniques in advancing perception capabilities for railway environments.
>
---
#### [new 079] ToFe: Lagged Token Freezing and Reusing for Efficient Vision Transformer Inference
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉Transformer推理优化任务，旨在解决其计算量大、难以部署到资源受限设备的问题。论文提出ToFe框架，通过滞后冻结和复用不重要的token，实现各阶段自适应计算，在保持性能的同时显著降低计算成本。**

- **链接: [http://arxiv.org/pdf/2507.16260v1](http://arxiv.org/pdf/2507.16260v1)**

> **作者:** Haoyue Zhang; Jie Zhang; Song Guo
>
> **摘要:** Although vision transformers (ViT) have shown remarkable success in various vision tasks, their computationally expensive self-attention hinder their deployment on resource-constrained devices. Token reduction, which discards less important tokens during forward propagation, has been proposed to enhance the efficiency of transformer models. However, existing methods handle unimportant tokens irreversibly, preventing their reuse in subsequent blocks. Considering that transformers focus on different information among blocks, tokens reduced in early blocks might be useful later. Furthermore, to adapt transformer models for resource-constrained devices, it is crucial to strike a balance between model performance and computational overhead. To address these challenges, in this paper, we introduce a novel Token Freezing and Reusing (ToFe) framework, where we identify important tokens at each stage and temporarily freeze the unimportant ones, allowing their lagged reusing at a later stage. Specifically, we design a prediction module for token identification and an approximate module for recovery of the frozen tokens. By jointly optimizing with the backbone through computation budget-aware end-to-end training, ToFe can adaptively process the necessary tokens at each block, thereby reducing computational cost while maintaining performance. Extensive experiments demonstrate that ToFe reduces the computational cost of LV-ViT model by 50% with less than 2% drop in Top-1 accuracy, achieving a better trade-off between performance and complexity compared to state-of-the-art methods.
>
---
#### [new 080] LDRFusion: A LiDAR-Dominant multimodal refinement framework for 3D object detection
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决激光雷达（LiDAR）与相机融合检测中点云稀疏和伪点噪声问题。论文提出LDRFusion框架，第一阶段仅用LiDAR生成准确提议，第二阶段引入伪点云检测困难样本，并融合两阶段结果。同时设计了层次化伪点残差编码模块，提升局部结构表示。实验表明该方法在KITTI数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.16224v1](http://arxiv.org/pdf/2507.16224v1)**

> **作者:** Jijun Wang; Yan Wu; Yujian Mo; Junqiao Zhao; Jun Yan; Yinghao Hu
>
> **摘要:** Existing LiDAR-Camera fusion methods have achieved strong results in 3D object detection. To address the sparsity of point clouds, previous approaches typically construct spatial pseudo point clouds via depth completion as auxiliary input and adopts a proposal-refinement framework to generate detection results. However, introducing pseudo points inevitably brings noise, potentially resulting in inaccurate predictions. Considering the differing roles and reliability levels of each modality, we propose LDRFusion, a novel Lidar-dominant two-stage refinement framework for multi-sensor fusion. The first stage soley relies on LiDAR to produce accurately localized proposals, followed by a second stage where pseudo point clouds are incorporated to detect challenging instances. The instance-level results from both stages are subsequently merged. To further enhance the representation of local structures in pseudo point clouds, we present a hierarchical pseudo point residual encoding module, which encodes neighborhood sets using both feature and positional residuals. Experiments on the KITTI dataset demonstrate that our framework consistently achieves strong performance across multiple categories and difficulty levels.
>
---
#### [new 081] Systole-Conditioned Generative Cardiac Motion
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决心脏CT影像中缺乏密集运动标注数据的问题。作者提出了一种基于条件变分自编码器的方法，通过单帧CT生成逼真的三维心脏运动序列及光流标注，用于训练更精确的心肌运动模型。**

- **链接: [http://arxiv.org/pdf/2507.15894v1](http://arxiv.org/pdf/2507.15894v1)**

> **作者:** Shahar Zuler; Gal Lifshitz; Hadar Averbuch-Elor; Dan Raviv
>
> **摘要:** Accurate motion estimation in cardiac computed tomography (CT) imaging is critical for assessing cardiac function and surgical planning. Data-driven methods have become the standard approach for dense motion estimation, but they rely on vast amounts of labeled data with dense ground-truth (GT) motion annotations, which are often unfeasible to obtain. To address this limitation, we present a novel approach that synthesizes realistically looking pairs of cardiac CT frames enriched with dense 3D flow field annotations. Our method leverages a conditional Variational Autoencoder (CVAE), which incorporates a novel multi-scale feature conditioning mechanism and is trained to generate 3D flow fields conditioned on a single CT frame. By applying the generated flow field to warp the given frame, we create pairs of frames that simulate realistic myocardium deformations across the cardiac cycle. These pairs serve as fully annotated data samples, providing optical flow GT annotations. Our data generation pipeline could enable the training and validation of more complex and accurate myocardium motion models, allowing for substantially reducing reliance on manual annotations. Our code, along with animated generated samples and additional material, is available on our project page: https://shaharzuler.github.io/GenerativeCardiacMotion_Page.
>
---
#### [new 082] A High Magnifications Histopathology Image Dataset for Oral Squamous Cell Carcinoma Diagnosis and Prognosis
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决现有口腔鳞状细胞癌（OSCC）数据集规模小、任务单一的问题。作者构建了包含1,325名患者、涵盖诊断与预后信息的Multi-OSCC数据集，并进行多任务模型评估，以推动OSCC的自动化分析与临床应用。**

- **链接: [http://arxiv.org/pdf/2507.16360v1](http://arxiv.org/pdf/2507.16360v1)**

> **作者:** Jinquan Guan; Junhong Guo; Qi Chen; Jian Chen; Yongkang Cai; Yilin He; Zhiquan Huang; Yan Wang; Yutong Xie
>
> **备注:** 12 pages, 11 tables, 4 figures
>
> **摘要:** Oral Squamous Cell Carcinoma (OSCC) is a prevalent and aggressive malignancy where deep learning-based computer-aided diagnosis and prognosis can enhance clinical assessments.However, existing publicly available OSCC datasets often suffer from limited patient cohorts and a restricted focus on either diagnostic or prognostic tasks, limiting the development of comprehensive and generalizable models. To bridge this gap, we introduce Multi-OSCC, a new histopathology image dataset comprising 1,325 OSCC patients, integrating both diagnostic and prognostic information to expand existing public resources. Each patient is represented by six high resolution histopathology images captured at x200, x400, and x1000 magnifications-two per magnification-covering both the core and edge tumor regions.The Multi-OSCC dataset is richly annotated for six critical clinical tasks: recurrence prediction (REC), lymph node metastasis (LNM), tumor differentiation (TD), tumor invasion (TI), cancer embolus (CE), and perineural invasion (PI). To benchmark this dataset, we systematically evaluate the impact of different visual encoders, multi-image fusion techniques, stain normalization, and multi-task learning frameworks. Our analysis yields several key insights: (1) The top-performing models achieve excellent results, with an Area Under the Curve (AUC) of 94.72% for REC and 81.23% for TD, while all tasks surpass 70% AUC; (2) Stain normalization benefits diagnostic tasks but negatively affects recurrence prediction; (3) Multi-task learning incurs a 3.34% average AUC degradation compared to single-task models in our multi-task benchmark, underscoring the challenge of balancing multiple tasks in our dataset. To accelerate future research, we publicly release the Multi-OSCC dataset and baseline models at https://github.com/guanjinquan/OSCC-PathologyImageDataset.
>
---
#### [new 083] DREAM: Scalable Red Teaming for Text-to-Image Generative Systems via Distribution Modeling
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文属于文本到图像生成系统的安全性评估任务，旨在解决现有红队测试方法在可扩展性和多样性方面的不足。论文提出了DREAM框架，通过建模问题提示的分布，实现高效、多样化的问题提示生成，提升文本到图像系统在部署前的安全性评估效果。**

- **链接: [http://arxiv.org/pdf/2507.16329v1](http://arxiv.org/pdf/2507.16329v1)**

> **作者:** Boheng Li; Junjie Wang; Yiming Li; Zhiyang Hu; Leyi Qi; Jianshuo Dong; Run Wang; Han Qiu; Zhan Qin; Tianwei Zhang
>
> **备注:** Preprint version. Under review
>
> **摘要:** Despite the integration of safety alignment and external filters, text-to-image (T2I) generative models are still susceptible to producing harmful content, such as sexual or violent imagery. This raises serious concerns about unintended exposure and potential misuse. Red teaming, which aims to proactively identify diverse prompts that can elicit unsafe outputs from the T2I system (including the core generative model as well as potential external safety filters and other processing components), is increasingly recognized as an essential method for assessing and improving safety before real-world deployment. Yet, existing automated red teaming approaches often treat prompt discovery as an isolated, prompt-level optimization task, which limits their scalability, diversity, and overall effectiveness. To bridge this gap, in this paper, we propose DREAM, a scalable red teaming framework to automatically uncover diverse problematic prompts from a given T2I system. Unlike most prior works that optimize prompts individually, DREAM directly models the probabilistic distribution of the target system's problematic prompts, which enables explicit optimization over both effectiveness and diversity, and allows efficient large-scale sampling after training. To achieve this without direct access to representative training samples, we draw inspiration from energy-based models and reformulate the objective into simple and tractable objectives. We further introduce GC-SPSA, an efficient optimization algorithm that provide stable gradient estimates through the long and potentially non-differentiable T2I pipeline. The effectiveness of DREAM is validated through extensive experiments, demonstrating that it surpasses 9 state-of-the-art baselines by a notable margin across a broad range of T2I models and safety filters in terms of prompt success rate and diversity.
>
---
#### [new 084] Semi-off-Policy Reinforcement Learning for Vision-Language Slow-thinking Reasoning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于视觉-语言多模态任务，旨在提升大模型的视觉慢思考推理能力。为解决现有模型受限于初始能力、难以通过策略梯度方法提升推理能力的问题，论文提出SOPHIA，结合在线视觉理解和离线语言推理，利用奖励传播进行半离线强化学习。实验表明该方法在多个基准上效果显著，优于部分闭源模型。**

- **链接: [http://arxiv.org/pdf/2507.16814v1](http://arxiv.org/pdf/2507.16814v1)**

> **作者:** Junhao Shen; Haiteng Zhao; Yuzhe Gu; Songyang Gao; Kuikun Liu; Haian Huang; Jianfei Gao; Dahua Lin; Wenwei Zhang; Kai Chen
>
> **摘要:** Enhancing large vision-language models (LVLMs) with visual slow-thinking reasoning is crucial for solving complex multimodal tasks. However, since LVLMs are mainly trained with vision-language alignment, it is difficult to adopt on-policy reinforcement learning (RL) to develop the slow thinking ability because the rollout space is restricted by its initial abilities. Off-policy RL offers a way to go beyond the current policy, but directly distilling trajectories from external models may cause visual hallucinations due to mismatched visual perception abilities across models. To address these issues, this paper proposes SOPHIA, a simple and scalable Semi-Off-Policy RL for vision-language slow-tHInking reAsoning. SOPHIA builds a semi-off-policy behavior model by combining on-policy visual understanding from a trainable LVLM with off-policy slow-thinking reasoning from a language model, assigns outcome-based rewards to reasoning, and propagates visual rewards backward. Then LVLM learns slow-thinking reasoning ability from the obtained reasoning trajectories using propagated rewards via off-policy RL algorithms. Extensive experiments with InternVL2.5 and InternVL3.0 with 8B and 38B sizes show the effectiveness of SOPHIA. Notably, SOPHIA improves InternVL3.0-38B by 8.50% in average, reaching state-of-the-art performance among open-source LVLMs on multiple multimodal reasoning benchmarks, and even outperforms some closed-source models (e.g., GPT-4.1) on the challenging MathVision and OlympiadBench, achieving 49.08% and 49.95% pass@1 accuracy, respectively. Analysis shows SOPHIA outperforms supervised fine-tuning and direct on-policy RL methods, offering a better policy initialization for further on-policy training.
>
---
#### [new 085] Quantization-Aware Neuromorphic Architecture for Efficient Skin Disease Classification on Resource-Constrained Devices
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决资源受限设备上皮肤病变分类的效率与精度问题。作者提出了QANA架构，结合神经形态计算与量化感知设计，实现低延迟、低能耗的皮肤疾病分类，优于现有CNN-to-SNN模型。**

- **链接: [http://arxiv.org/pdf/2507.15958v1](http://arxiv.org/pdf/2507.15958v1)**

> **作者:** Haitian Wang; Xinyu Wang; Yiren Wang; Karen Lee; Zichen Geng; Xian Zhang; Kehkashan Kiran; Yu Zhang; Bo Miao
>
> **备注:** This manuscript is under review for IEEE BIBM 2025
>
> **摘要:** Accurate and efficient skin lesion classification on edge devices is critical for accessible dermatological care but remains challenging due to computational, energy, and privacy constraints. We introduce QANA, a novel quantization-aware neuromorphic architecture for incremental skin lesion classification on resource-limited hardware. QANA effectively integrates ghost modules, efficient channel attention, and squeeze-and-excitation blocks for robust feature representation with low-latency and energy-efficient inference. Its quantization-aware head and spike-compatible transformations enable seamless conversion to spiking neural networks (SNNs) and deployment on neuromorphic platforms. Evaluation on the large-scale HAM10000 benchmark and a real-world clinical dataset shows that QANA achieves 91.6\% Top-1 accuracy and 82.4\% macro F1 on HAM10000, and 90.8\% / 81.7\% on the clinical dataset, significantly outperforming state-of-the-art CNN-to-SNN models under fair comparison. Deployed on BrainChip Akida hardware, QANA achieves 1.5\,ms inference latency and 1.7\,mJ energy per image, reducing inference latency and energy use by over 94.6\%/98.6\% compared to GPU-based CNNs surpassing state-of-the-art CNN-to-SNN conversion baselines. These results demonstrate the effectiveness of QANA for accurate, real-time, and privacy-sensitive medical analysis in edge environments.
>
---
#### [new 086] Screen2AX: Vision-Based Approach for Automatic macOS Accessibility Generation
- **分类: cs.LG; cs.AI; cs.CV; cs.HC**

- **简介: 论文提出Screen2AX，属于桌面界面可访问性生成任务，旨在解决macOS应用缺乏完整可访问性元数据的问题。通过视觉和语言模型，自动从截图生成层级结构的可访问性信息，并构建相关数据集与评估基准，显著提升自主代理对复杂桌面界面的理解与操作性能。**

- **链接: [http://arxiv.org/pdf/2507.16704v1](http://arxiv.org/pdf/2507.16704v1)**

> **作者:** Viktor Muryn; Marta Sumyk; Mariya Hirna; Sofiya Garkot; Maksym Shamrai
>
> **摘要:** Desktop accessibility metadata enables AI agents to interpret screens and supports users who depend on tools like screen readers. Yet, many applications remain largely inaccessible due to incomplete or missing metadata provided by developers - our investigation shows that only 33% of applications on macOS offer full accessibility support. While recent work on structured screen representation has primarily addressed specific challenges, such as UI element detection or captioning, none has attempted to capture the full complexity of desktop interfaces by replicating their entire hierarchical structure. To bridge this gap, we introduce Screen2AX, the first framework to automatically create real-time, tree-structured accessibility metadata from a single screenshot. Our method uses vision-language and object detection models to detect, describe, and organize UI elements hierarchically, mirroring macOS's system-level accessibility structure. To tackle the limited availability of data for macOS desktop applications, we compiled and publicly released three datasets encompassing 112 macOS applications, each annotated for UI element detection, grouping, and hierarchical accessibility metadata alongside corresponding screenshots. Screen2AX accurately infers hierarchy trees, achieving a 77% F1 score in reconstructing a complete accessibility tree. Crucially, these hierarchy trees improve the ability of autonomous agents to interpret and interact with complex desktop interfaces. We introduce Screen2AX-Task, a benchmark specifically designed for evaluating autonomous agent task execution in macOS desktop environments. Using this benchmark, we demonstrate that Screen2AX delivers a 2.2x performance improvement over native accessibility representations and surpasses the state-of-the-art OmniParser V2 system on the ScreenSpot benchmark.
>
---
#### [new 087] Semantic-Aware Gaussian Process Calibration with Structured Layerwise Kernels for Deep Neural Networks
- **分类: cs.LG; cs.CV**

- **简介: 论文属于模型校准任务，旨在提升神经网络分类器的预测可靠性。现有高斯过程校准方法忽视网络层次结构，导致效果受限。本文提出语义感知分层高斯过程（SAL-GP），通过逐层建模与结构化多层核，联合优化各层校准，增强模型可解释性与不确定性量化能力。**

- **链接: [http://arxiv.org/pdf/2507.15987v1](http://arxiv.org/pdf/2507.15987v1)**

> **作者:** Kyung-hwan Lee; Kyung-tae Kim
>
> **摘要:** Calibrating the confidence of neural network classifiers is essential for quantifying the reliability of their predictions during inference. However, conventional Gaussian Process (GP) calibration methods often fail to capture the internal hierarchical structure of deep neural networks, limiting both interpretability and effectiveness for assessing predictive reliability. We propose a Semantic-Aware Layer-wise Gaussian Process (SAL-GP) framework that mirrors the layered architecture of the target neural network. Instead of applying a single global GP correction, SAL-GP employs a multi-layer GP model, where each layer's feature representation is mapped to a local calibration correction. These layerwise GPs are coupled through a structured multi-layer kernel, enabling joint marginalization across all layers. This design allows SAL-GP to capture both local semantic dependencies and global calibration coherence, while consistently propagating predictive uncertainty through the network. The resulting framework enhances interpretability aligned with the network architecture and enables principled evaluation of confidence consistency and uncertainty quantification in deep models.
>
---
#### [new 088] Understanding Generalization, Robustness, and Interpretability in Low-Capacity Neural Networks
- **分类: cs.LG; cs.AI; cs.CV; 68T07; I.2.6; I.5.1**

- **简介: 该论文研究低容量神经网络的泛化、鲁棒性和可解释性。通过构建不同视觉难度的二分类任务，发现模型容量需求随任务复杂度增加，且稀疏子网络可保持高性能与鲁棒性，过参数化有助于提升对抗输入扰动的能力。**

- **链接: [http://arxiv.org/pdf/2507.16278v1](http://arxiv.org/pdf/2507.16278v1)**

> **作者:** Yash Kumar
>
> **备注:** 15 pages (10 pages main text). 18 figures (8 main, 10 appendix), 1 table
>
> **摘要:** Although modern deep learning often relies on massive over-parameterized models, the fundamental interplay between capacity, sparsity, and robustness in low-capacity networks remains a vital area of study. We introduce a controlled framework to investigate these properties by creating a suite of binary classification tasks from the MNIST dataset with increasing visual difficulty (e.g., 0 and 1 vs. 4 and 9). Our experiments reveal three core findings. First, the minimum model capacity required for successful generalization scales directly with task complexity. Second, these trained networks are robust to extreme magnitude pruning (up to 95% sparsity), revealing the existence of sparse, high-performing subnetworks. Third, we show that over-parameterization provides a significant advantage in robustness against input corruption. Interpretability analysis via saliency maps further confirms that these identified sparse subnetworks preserve the core reasoning process of the original dense models. This work provides a clear, empirical demonstration of the foundational trade-offs governing simple neural networks.
>
---
#### [new 089] SFNet: A Spatio-Frequency Domain Deep Learning Network for Efficient Alzheimer's Disease Diagnosis
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决阿尔茨海默病（AD）的早期诊断问题。现有方法多仅利用MRI的空间或频率单域信息，限制了诊断效果。论文提出SFNet，首个结合3D MRI空间与频率域信息的端到端深度学习模型，融合局部空间特征与全局频率表示，并引入多尺度注意力模块优化特征提取，提升了AD诊断效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.16267v1](http://arxiv.org/pdf/2507.16267v1)**

> **作者:** Xinyue Yang; Meiliang Liu; Yunfang Xu; Xiaoxiao Yang; Zhengye Si; Zijin Li; Zhiwen Zhao
>
> **摘要:** Alzheimer's disease (AD) is a progressive neurodegenerative disorder that predominantly affects the elderly population and currently has no cure. Magnetic Resonance Imaging (MRI), as a non-invasive imaging technique, is essential for the early diagnosis of AD. MRI inherently contains both spatial and frequency information, as raw signals are acquired in the frequency domain and reconstructed into spatial images via the Fourier transform. However, most existing AD diagnostic models extract features from a single domain, limiting their capacity to fully capture the complex neuroimaging characteristics of the disease. While some studies have combined spatial and frequency information, they are mostly confined to 2D MRI, leaving the potential of dual-domain analysis in 3D MRI unexplored. To overcome this limitation, we propose Spatio-Frequency Network (SFNet), the first end-to-end deep learning framework that simultaneously leverages spatial and frequency domain information to enhance 3D MRI-based AD diagnosis. SFNet integrates an enhanced dense convolutional network to extract local spatial features and a global frequency module to capture global frequency-domain representations. Additionally, a novel multi-scale attention module is proposed to further refine spatial feature extraction. Experiments on the Alzheimer's Disease Neuroimaging Initiative (ANDI) dataset demonstrate that SFNet outperforms existing baselines and reduces computational overhead in classifying cognitively normal (CN) and AD, achieving an accuracy of 95.1%.
>
---
#### [new 090] Semantic Segmentation for Preoperative Planning in Transcatheter Aortic Valve Replacement
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在通过语义分割模型辅助经导管主动脉瓣置换术（TAVR）的术前规划。论文提出生成细粒度伪标签，并改进损失函数，提升分割性能，使相关解剖结构在CT扫描中更准确可测。**

- **链接: [http://arxiv.org/pdf/2507.16573v1](http://arxiv.org/pdf/2507.16573v1)**

> **作者:** Cedric Zöllner; Simon Reiß; Alexander Jaus; Amroalalaa Sholi; Ralf Sodian; Rainer Stiefelhagen
>
> **备注:** Accepted at 16th MICCAI Workshop on Statistical Atlases and Computational Modeling of the Heart (STACOM)
>
> **摘要:** When preoperative planning for surgeries is conducted on the basis of medical images, artificial intelligence methods can support medical doctors during assessment. In this work, we consider medical guidelines for preoperative planning of the transcatheter aortic valve replacement (TAVR) and identify tasks, that may be supported via semantic segmentation models by making relevant anatomical structures measurable in computed tomography scans. We first derive fine-grained TAVR-relevant pseudo-labels from coarse-grained anatomical information, in order to train segmentation models and quantify how well they are able to find these structures in the scans. Furthermore, we propose an adaptation to the loss function in training these segmentation models and through this achieve a +1.27% Dice increase in performance. Our fine-grained TAVR-relevant pseudo-labels and the computed tomography scans we build upon are available at https://doi.org/10.5281/zenodo.16274176.
>
---
#### [new 091] Handcrafted vs. Deep Radiomics vs. Fusion vs. Deep Learning: A Comprehensive Review of Machine Learning -Based Cancer Outcome Prediction in PET and SPECT Imaging
- **分类: physics.med-ph; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决癌症预后预测中不同机器学习方法的性能比较问题。论文系统回顾了226项研究，评估手工特征、深度特征、融合方法等在PET/SPECT影像中的表现，发现深度特征准确率最高，融合方法AUC最优，同时指出数据质量和标准化不足等问题。**

- **链接: [http://arxiv.org/pdf/2507.16065v1](http://arxiv.org/pdf/2507.16065v1)**

> **作者:** Mohammad R. Salmanpour; Somayeh Sadat Mehrnia; Sajad Jabarzadeh Ghandilu; Zhino Safahi; Sonya Falahati; Shahram Taeb; Ghazal Mousavi; Mehdi Maghsoudi; Ahmad Shariftabrizi; Ilker Hacihaliloglu; Arman Rahmim
>
> **摘要:** Machine learning (ML), including deep learning (DL) and radiomics-based methods, is increasingly used for cancer outcome prediction with PET and SPECT imaging. However, the comparative performance of handcrafted radiomics features (HRF), deep radiomics features (DRF), DL models, and hybrid fusion approaches remains inconsistent across clinical applications. This systematic review analyzed 226 studies published from 2020 to 2025 that applied ML to PET or SPECT imaging for outcome prediction. Each study was evaluated using a 59-item framework covering dataset construction, feature extraction, validation methods, interpretability, and risk of bias. We extracted key details including model type, cancer site, imaging modality, and performance metrics such as accuracy and area under the curve (AUC). PET-based studies (95%) generally outperformed those using SPECT, likely due to higher spatial resolution and sensitivity. DRF models achieved the highest mean accuracy (0.862), while fusion models yielded the highest AUC (0.861). ANOVA confirmed significant differences in performance (accuracy: p=0.0006, AUC: p=0.0027). Common limitations included inadequate handling of class imbalance (59%), missing data (29%), and low population diversity (19%). Only 48% of studies adhered to IBSI standards. These findings highlight the need for standardized pipelines, improved data quality, and explainable AI to support clinical integration.
>
---
#### [new 092] Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于风险分析任务，旨在识别和评估前沿人工智能模型带来的重大风险。依据“AI-$45^\circ$法”与E-T-C框架，研究对七大风险领域进行评估，判断模型处于绿、黄、红风险区域。结果显示，当前模型尚未突破红线，但部分领域如说服与操控风险较高。论文呼吁加强风险管理与集体应对。**

- **链接: [http://arxiv.org/pdf/2507.16534v1](http://arxiv.org/pdf/2507.16534v1)**

> **作者:** Shanghai AI Lab; :; Xiaoyang Chen; Yunhao Chen; Zeren Chen; Zhiyun Chen; Hanyun Cui; Yawen Duan; Jiaxuan Guo; Qi Guo; Xuhao Hu; Hong Huang; Lige Huang; Chunxiao Li; Juncheng Li; Qihao Lin; Dongrui Liu; Xinmin Liu; Zicheng Liu; Chaochao Lu; Xiaoya Lu; Jingjing Qu; Qibing Ren; Jing Shao; Jingwei Shi; Jingwei Sun; Peng Wang; Weibing Wang; Jia Xu; Lewen Yan; Xiao Yu; Yi Yu; Boxuan Zhang; Jie Zhang; Weichen Zhang; Zhijie Zheng; Tianyi Zhou; Bowen Zhou
>
> **备注:** 97 pages, 37 figures
>
> **摘要:** To understand and identify the unprecedented risks posed by rapidly advancing artificial intelligence (AI) models, this report presents a comprehensive assessment of their frontier risks. Drawing on the E-T-C analysis (deployment environment, threat source, enabling capability) from the Frontier AI Risk Management Framework (v1.0) (SafeWork-F1-Framework), we identify critical risks in seven areas: cyber offense, biological and chemical risks, persuasion and manipulation, uncontrolled autonomous AI R\&D, strategic deception and scheming, self-replication, and collusion. Guided by the "AI-$45^\circ$ Law," we evaluate these risks using "red lines" (intolerable thresholds) and "yellow lines" (early warning indicators) to define risk zones: green (manageable risk for routine deployment and continuous monitoring), yellow (requiring strengthened mitigations and controlled deployment), and red (necessitating suspension of development and/or deployment). Experimental results show that all recent frontier AI models reside in green and yellow zones, without crossing red lines. Specifically, no evaluated models cross the yellow line for cyber offense or uncontrolled AI R\&D risks. For self-replication, and strategic deception and scheming, most models remain in the green zone, except for certain reasoning models in the yellow zone. In persuasion and manipulation, most models are in the yellow zone due to their effective influence on humans. For biological and chemical risks, we are unable to rule out the possibility of most models residing in the yellow zone, although detailed threat modeling and in-depth assessment are required to make further claims. This work reflects our current understanding of AI frontier risks and urges collective action to mitigate these challenges.
>
---
#### [new 093] Improved Semantic Segmentation from Ultra-Low-Resolution RGB Images Applied to Privacy-Preserving Object-Goal Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人隐私保护与语义分割任务，旨在解决超低分辨率RGB图像下语义分割困难的问题，以实现保护隐私的语义目标导航。作者提出一种联合学习方法，融合特征提取与分割判别，提升超低分辨率下的语义分割效果，从而增强隐私保护场景下的导航成功率。**

- **链接: [http://arxiv.org/pdf/2507.16034v1](http://arxiv.org/pdf/2507.16034v1)**

> **作者:** Xuying Huang; Sicong Pan; Olga Zatsarynna; Juergen Gall; Maren Bennewitz
>
> **备注:** Submitted to RA-L
>
> **摘要:** User privacy in mobile robotics has become a critical concern. Existing methods typically prioritize either the performance of downstream robotic tasks or privacy protection, with the latter often constraining the effectiveness of task execution. To jointly address both objectives, we study semantic-based robot navigation in an ultra-low-resolution setting to preserve visual privacy. A key challenge in such scenarios is recovering semantic segmentation from ultra-low-resolution RGB images. In this work, we introduce a novel fully joint-learning method that integrates an agglomerative feature extractor and a segmentation-aware discriminator to solve ultra-low-resolution semantic segmentation, thereby enabling privacy-preserving, semantic object-goal navigation. Our method outperforms different baselines on ultra-low-resolution semantic segmentation and our improved segmentation results increase the success rate of the semantic object-goal navigation in a real-world privacy-constrained scenario.
>
---
#### [new 094] Improving U-Net Confidence on TEM Image Data with L2-Regularization, Transfer Learning, and Deep Fine-Tuning
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像分析任务，旨在提升TEM图像中纳米缺陷检测的准确性。针对标注数据少、误差多的问题，作者采用预训练模型、L2正则化和深度微调方法，改进U-Net模型性能。通过新评估指标验证，缺陷检测率提升了57%。**

- **链接: [http://arxiv.org/pdf/2507.16779v1](http://arxiv.org/pdf/2507.16779v1)**

> **作者:** Aiden Ochoa; Xinyuan Xu; Xing Wang
>
> **备注:** Accepted into the ICCV 2025 CV4MS Workshop
>
> **摘要:** With ever-increasing data volumes, it is essential to develop automated approaches for identifying nanoscale defects in transmission electron microscopy (TEM) images. However, compared to features in conventional photographs, nanoscale defects in TEM images exhibit far greater variation due to the complex contrast mechanisms and intricate defect structures. These challenges often result in much less labeled data and higher rates of annotation errors, posing significant obstacles to improving machine learning model performance for TEM image analysis. To address these limitations, we examined transfer learning by leveraging large, pre-trained models used for natural images. We demonstrated that by using the pre-trained encoder and L2-regularization, semantically complex features are ignored in favor of simpler, more reliable cues, substantially improving the model performance. However, this improvement cannot be captured by conventional evaluation metrics such as F1-score, which can be skewed by human annotation errors treated as ground truth. Instead, we introduced novel evaluation metrics that are independent of the annotation accuracy. Using grain boundary detection in UO2 TEM images as a case study, we found that our approach led to a 57% improvement in defect detection rate, which is a robust and holistic measure of model performance on the TEM dataset used in this work. Finally, we showed that model self-confidence is only achieved through transfer learning and fine-tuning of very deep layers.
>
---
#### [new 095] A Target-based Multi-LiDAR Multi-Camera Extrinsic Calibration System
- **分类: cs.RO; cs.CV**

- **简介: 论文属于自动驾驶中的多传感器外参标定任务，旨在解决多激光雷达与多相机系统间精确对齐的问题。通过使用定制的ChArUco标定板和非线性优化方法，实现了跨模态传感器标定。实验验证了该方法在真实场景中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.16621v1](http://arxiv.org/pdf/2507.16621v1)**

> **作者:** Lorenzo Gentilini; Pierpaolo Serio; Valentina Donzella; Lorenzo Pollini
>
> **摘要:** Extrinsic Calibration represents the cornerstone of autonomous driving. Its accuracy plays a crucial role in the perception pipeline, as any errors can have implications for the safety of the vehicle. Modern sensor systems collect different types of data from the environment, making it harder to align the data. To this end, we propose a target-based extrinsic calibration system tailored for a multi-LiDAR and multi-camera sensor suite. This system enables cross-calibration between LiDARs and cameras with limited prior knowledge using a custom ChArUco board and a tailored nonlinear optimization method. We test the system with real-world data gathered in a warehouse. Results demonstrated the effectiveness of the proposed method, highlighting the feasibility of a unique pipeline tailored for various types of sensors.
>
---
#### [new 096] Pyramid Hierarchical Masked Diffusion Model for Imaging Synthesis
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像合成任务，旨在解决缺失成像模态的问题。论文提出了一种多尺度分层模型PHMDiff，通过结合Transformer和扩散模型，实现跨粒度信息一致性建模，提升了合成图像的质量和结构完整性。实验表明其在PSNR和SSIM指标上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.16579v1](http://arxiv.org/pdf/2507.16579v1)**

> **作者:** Xiaojiao Xiao; Qinmin Vivian Hu; Guanghui Wang
>
> **摘要:** Medical image synthesis plays a crucial role in clinical workflows, addressing the common issue of missing imaging modalities due to factors such as extended scan times, scan corruption, artifacts, patient motion, and intolerance to contrast agents. The paper presents a novel image synthesis network, the Pyramid Hierarchical Masked Diffusion Model (PHMDiff), which employs a multi-scale hierarchical approach for more detailed control over synthesizing high-quality images across different resolutions and layers. Specifically, this model utilizes randomly multi-scale high-proportion masks to speed up diffusion model training, and balances detail fidelity and overall structure. The integration of a Transformer-based Diffusion model process incorporates cross-granularity regularization, modeling the mutual information consistency across each granularity's latent spaces, thereby enhancing pixel-level perceptual accuracy. Comprehensive experiments on two challenging datasets demonstrate that PHMDiff achieves superior performance in both the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM), highlighting its capability to produce high-quality synthesized images with excellent structural integrity. Ablation studies further confirm the contributions of each component. Furthermore, the PHMDiff model, a multi-scale image synthesis framework across and within medical imaging modalities, shows significant advantages over other methods. The source code is available at https://github.com/xiaojiao929/PHMDiff
>
---
#### [new 097] Designing for Difference: How Human Characteristics Shape Perceptions of Collaborative Robots
- **分类: cs.RO; cs.AI; cs.CV; cs.ET; cs.SY; eess.SY**

- **简介: 该论文研究人类特征如何影响对协作机器人行为的接受度，旨在解决辅助机器人在与不同人群协作时的设计责任与包容性问题。通过在线实验，参与者评估不同人机协作场景，结合认知情感映射（CAM）方法获取反思反馈。结果显示，机器人行为类型与协作对象特征显著影响评价，强调亲社会设计的重要性，并验证了反思方法在用户中心设计中的价值。**

- **链接: [http://arxiv.org/pdf/2507.16480v1](http://arxiv.org/pdf/2507.16480v1)**

> **作者:** Sabrina Livanec; Laura Londoño; Michael Gorki; Adrian Röfer; Abhinav Valada; Andrea Kiesel
>
> **摘要:** The development of assistive robots for social collaboration raises critical questions about responsible and inclusive design, especially when interacting with individuals from protected groups such as those with disabilities or advanced age. Currently, research is scarce on how participants assess varying robot behaviors in combination with diverse human needs, likely since participants have limited real-world experience with advanced domestic robots. In the current study, we aim to address this gap while using methods that enable participants to assess robot behavior, as well as methods that support meaningful reflection despite limited experience. In an online study, 112 participants (from both experimental and control groups) evaluated 7 videos from a total of 28 variations of human-robot collaboration types. The experimental group first completed a cognitive-affective mapping (CAM) exercise on human-robot collaboration before providing their ratings. Although CAM reflection did not significantly affect overall ratings, it led to more pronounced assessments for certain combinations of robot behavior and human condition. Most importantly, the type of human-robot collaboration influences the assessment. Antisocial robot behavior was consistently rated as the lowest, while collaboration with aged individuals elicited more sensitive evaluations. Scenarios involving object handovers were viewed more positively than those without them. These findings suggest that both human characteristics and interaction paradigms influence the perceived acceptability of collaborative robots, underscoring the importance of prosocial design. They also highlight the potential of reflective methods, such as CAM, to elicit nuanced feedback, supporting the development of user-centered and socially responsible robotic systems tailored to diverse populations.
>
---
#### [new 098] MultiTaskDeltaNet: Change Detection-based Image Segmentation for Operando ETEM with Application to Carbon Gasification Kinetics
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决固体反应中动态特征的高精度语义分割问题。由于标记数据少、特征模糊和小目标问题，传统方法效果有限。论文提出MultiTaskDeltaNet（MTDN），将分割任务重构为变化检测问题，采用Siamese网络与U-Net主干及多任务学习策略，提升了小特征的分割精度，显著优于传统模型。**

- **链接: [http://arxiv.org/pdf/2507.16803v1](http://arxiv.org/pdf/2507.16803v1)**

> **作者:** Yushuo Niu; Tianyu Li; Yuanyuan Zhu; Qian Yang
>
> **摘要:** Transforming in-situ transmission electron microscopy (TEM) imaging into a tool for spatially-resolved operando characterization of solid-state reactions requires automated, high-precision semantic segmentation of dynamically evolving features. However, traditional deep learning methods for semantic segmentation often encounter limitations due to the scarcity of labeled data, visually ambiguous features of interest, and small-object scenarios. To tackle these challenges, we introduce MultiTaskDeltaNet (MTDN), a novel deep learning architecture that creatively reconceptualizes the segmentation task as a change detection problem. By implementing a unique Siamese network with a U-Net backbone and using paired images to capture feature changes, MTDN effectively utilizes minimal data to produce high-quality segmentations. Furthermore, MTDN utilizes a multi-task learning strategy to leverage correlations between physical features of interest. In an evaluation using data from in-situ environmental TEM (ETEM) videos of filamentous carbon gasification, MTDN demonstrated a significant advantage over conventional segmentation models, particularly in accurately delineating fine structural features. Notably, MTDN achieved a 10.22% performance improvement over conventional segmentation models in predicting small and visually ambiguous physical features. This work bridges several key gaps between deep learning and practical TEM image analysis, advancing automated characterization of nanomaterials in complex experimental settings.
>
---
#### [new 099] MLRU++: Multiscale Lightweight Residual UNETR++ with Attention for Efficient 3D Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决3D医学图像分割中精度与计算效率难以平衡的问题。作者提出MLRU++模型，结合轻量级注意力模块（LCBAM）和多尺度瓶颈块（M2B），在多个数据集上取得优异性能，提升了分割精度并降低了计算成本。**

- **链接: [http://arxiv.org/pdf/2507.16122v1](http://arxiv.org/pdf/2507.16122v1)**

> **作者:** Nand Kumar Yadav; Rodrigue Rizk; Willium WC Chen; KC
>
> **摘要:** Accurate and efficient medical image segmentation is crucial but challenging due to anatomical variability and high computational demands on volumetric data. Recent hybrid CNN-Transformer architectures achieve state-of-the-art results but add significant complexity. In this paper, we propose MLRU++, a Multiscale Lightweight Residual UNETR++ architecture designed to balance segmentation accuracy and computational efficiency. It introduces two key innovations: a Lightweight Channel and Bottleneck Attention Module (LCBAM) that enhances contextual feature encoding with minimal overhead, and a Multiscale Bottleneck Block (M2B) in the decoder that captures fine-grained details via multi-resolution feature aggregation. Experiments on four publicly available benchmark datasets (Synapse, BTCV, ACDC, and Decathlon Lung) demonstrate that MLRU++ achieves state-of-the-art performance, with average Dice scores of 87.57% (Synapse), 93.00% (ACDC), and 81.12% (Lung). Compared to existing leading models, MLRU++ improves Dice scores by 5.38% and 2.12% on Synapse and ACDC, respectively, while significantly reducing parameter count and computational cost. Ablation studies evaluating LCBAM and M2B further confirm the effectiveness of the proposed architectural components. Results suggest that MLRU++ offers a practical and high-performing solution for 3D medical image segmentation tasks. Source code is available at: https://github.com/1027865/MLRUPP
>
---
#### [new 100] Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **简介: 该论文属于文本生成图像任务，旨在解决扩散模型在下游微调时恢复有害行为的问题。作者提出了ResAlign框架，通过建模微调过程并引入元学习策略，增强模型在保持安全性和生成能力方面的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.16302v1](http://arxiv.org/pdf/2507.16302v1)**

> **作者:** Boheng Li; Renjie Gu; Junjie Wang; Leyi Qi; Yiming Li; Run Wang; Zhan Qin; Tianwei Zhang
>
> **备注:** Preprint version. Under review
>
> **摘要:** Text-to-image (T2I) diffusion models have achieved impressive image generation quality and are increasingly fine-tuned for personalized applications. However, these models often inherit unsafe behaviors from toxic pretraining data, raising growing safety concerns. While recent safety-driven unlearning methods have made promising progress in suppressing model toxicity, they are identified to be fragile to downstream fine-tuning, where we reveal that state-of-the-art methods largely fail to retain their effectiveness even when fine-tuned on entirely benign datasets. To mitigate this problem, in this paper, we propose ResAlign, a safety-driven unlearning framework with enhanced resilience against downstream fine-tuning. By modeling downstream fine-tuning as an implicit optimization problem with a Moreau Envelope-based reformulation, ResAlign enables efficient gradient estimation to minimize the recovery of harmful behaviors. Additionally, a meta-learning strategy is proposed to simulate a diverse distribution of fine-tuning scenarios to improve generalization. Extensive experiments across a wide range of datasets, fine-tuning methods, and configurations demonstrate that ResAlign consistently outperforms prior unlearning approaches in retaining safety after downstream fine-tuning while preserving benign generation capability well.
>
---
## 更新

#### [replaced 001] Unveiling the Potential of Segment Anything Model 2 for RGB-Thermal Semantic Segmentation with Language Guidance
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.02581v2](http://arxiv.org/pdf/2503.02581v2)**

> **作者:** Jiayi Zhao; Fei Teng; Kai Luo; Guoqiang Zhao; Zhiyong Li; Xu Zheng; Kailun Yang
>
> **备注:** Accepted to IROS 2025. The source code will be made publicly available at https://github.com/iAsakiT3T/SHIFNet
>
> **摘要:** The perception capability of robotic systems relies on the richness of the dataset. Although Segment Anything Model 2 (SAM2), trained on large datasets, demonstrates strong perception potential in perception tasks, its inherent training paradigm prevents it from being suitable for RGB-T tasks. To address these challenges, we propose SHIFNet, a novel SAM2-driven Hybrid Interaction Paradigm that unlocks the potential of SAM2 with linguistic guidance for efficient RGB-Thermal perception. Our framework consists of two key components: (1) Semantic-Aware Cross-modal Fusion (SACF) module that dynamically balances modality contributions through text-guided affinity learning, overcoming SAM2's inherent RGB bias; (2) Heterogeneous Prompting Decoder (HPD) that enhances global semantic information through a semantic enhancement module and then combined with category embeddings to amplify cross-modal semantic consistency. With 32.27M trainable parameters, SHIFNet achieves state-of-the-art segmentation performance on public benchmarks, reaching 89.8% on PST900 and 67.8% on FMB, respectively. The framework facilitates the adaptation of pre-trained large models to RGB-T segmentation tasks, effectively mitigating the high costs associated with data collection while endowing robotic systems with comprehensive perception capabilities. The source code will be made publicly available at https://github.com/iAsakiT3T/SHIFNet.
>
---
#### [replaced 002] Rethinking Discrete Tokens: Treating Them as Conditions for Continuous Autoregressive Image Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01756v2](http://arxiv.org/pdf/2507.01756v2)**

> **作者:** Peng Zheng; Junke Wang; Yi Chang; Yizhou Yu; Rui Ma; Zuxuan Wu
>
> **备注:** iccv 2025, camera-ready version
>
> **摘要:** Recent advances in large language models (LLMs) have spurred interests in encoding images as discrete tokens and leveraging autoregressive (AR) frameworks for visual generation. However, the quantization process in AR-based visual generation models inherently introduces information loss that degrades image fidelity. To mitigate this limitation, recent studies have explored to autoregressively predict continuous tokens. Unlike discrete tokens that reside in a structured and bounded space, continuous representations exist in an unbounded, high-dimensional space, making density estimation more challenging and increasing the risk of generating out-of-distribution artifacts. Based on the above findings, this work introduces DisCon (Discrete-Conditioned Continuous Autoregressive Model), a novel framework that reinterprets discrete tokens as conditional signals rather than generation targets. By modeling the conditional probability of continuous representations conditioned on discrete tokens, DisCon circumvents the optimization challenges of continuous token modeling while avoiding the information loss caused by quantization. DisCon achieves a gFID score of 1.38 on ImageNet 256$\times$256 generation, outperforming state-of-the-art autoregressive approaches by a clear margin. Project page: https://pengzheng0707.github.io/DisCon.
>
---
#### [replaced 003] One-for-More: Continual Diffusion Model for Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.19848v3](http://arxiv.org/pdf/2502.19848v3)**

> **作者:** Xiaofan Li; Xin Tan; Zhuo Chen; Zhizhong Zhang; Ruixin Zhang; Rizen Guo; Guannan Jiang; Yulong Chen; Yanyun Qu; Lizhuang Ma; Yuan Xie
>
> **备注:** Accepted by CVPR2025
>
> **摘要:** With the rise of generative models, there is a growing interest in unifying all tasks within a generative framework. Anomaly detection methods also fall into this scope and utilize diffusion models to generate or reconstruct normal samples when given arbitrary anomaly images. However, our study found that the diffusion model suffers from severe ``faithfulness hallucination'' and ``catastrophic forgetting'', which can't meet the unpredictable pattern increments. To mitigate the above problems, we propose a continual diffusion model that uses gradient projection to achieve stable continual learning. Gradient projection deploys a regularization on the model updating by modifying the gradient towards the direction protecting the learned knowledge. But as a double-edged sword, it also requires huge memory costs brought by the Markov process. Hence, we propose an iterative singular value decomposition method based on the transitive property of linear representation, which consumes tiny memory and incurs almost no performance loss. Finally, considering the risk of ``over-fitting'' to normal images of the diffusion model, we propose an anomaly-masked network to enhance the condition mechanism of the diffusion model. For continual anomaly detection, ours achieves first place in 17/18 settings on MVTec and VisA. Code is available at https://github.com/FuNz-0/One-for-More
>
---
#### [replaced 004] SeC: Advancing Complex Video Object Segmentation via Progressive Concept Construction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.15852v2](http://arxiv.org/pdf/2507.15852v2)**

> **作者:** Zhixiong Zhang; Shuangrui Ding; Xiaoyi Dong; Songxin He; Jianfan Lin; Junsong Tang; Yuhang Zang; Yuhang Cao; Dahua Lin; Jiaqi Wang
>
> **备注:** project page: https://rookiexiong7.github.io/projects/SeC/ ; code: https://github.com/OpenIXCLab/SeC ; dataset: https://huggingface.co/datasets/OpenIXCLab/SeCVOS
>
> **摘要:** Video Object Segmentation (VOS) is a core task in computer vision, requiring models to track and segment target objects across video frames. Despite notable advances with recent efforts, current techniques still lag behind human capabilities in handling drastic visual variations, occlusions, and complex scene changes. This limitation arises from their reliance on appearance matching, neglecting the human-like conceptual understanding of objects that enables robust identification across temporal dynamics. Motivated by this gap, we propose Segment Concept (SeC), a concept-driven segmentation framework that shifts from conventional feature matching to the progressive construction and utilization of high-level, object-centric representations. SeC employs Large Vision-Language Models (LVLMs) to integrate visual cues across diverse frames, constructing robust conceptual priors. During inference, SeC forms a comprehensive semantic representation of the target based on processed frames, realizing robust segmentation of follow-up frames. Furthermore, SeC adaptively balances LVLM-based semantic reasoning with enhanced feature matching, dynamically adjusting computational efforts based on scene complexity. To rigorously assess VOS methods in scenarios demanding high-level conceptual reasoning and robust semantic understanding, we introduce the Semantic Complex Scenarios Video Object Segmentation benchmark (SeCVOS). SeCVOS comprises 160 manually annotated multi-scenario videos designed to challenge models with substantial appearance variations and dynamic scene transformations. In particular, SeC achieves an 11.8-point improvement over SAM 2.1 on SeCVOS, establishing a new state-of-the-art in concept-aware video object segmentation.
>
---
#### [replaced 005] FBSDiff: Plug-and-Play Frequency Band Substitution of Diffusion Features for Highly Controllable Text-Driven Image Translation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.00998v4](http://arxiv.org/pdf/2408.00998v4)**

> **作者:** Xiang Gao; Jiaying Liu
>
> **备注:** Accepted conference paper of ACM MM 2024
>
> **摘要:** Large-scale text-to-image diffusion models have been a revolutionary milestone in the evolution of generative AI and multimodal technology, allowing wonderful image generation with natural-language text prompt. However, the issue of lacking controllability of such models restricts their practical applicability for real-life content creation. Thus, attention has been focused on leveraging a reference image to control text-to-image synthesis, which is also regarded as manipulating (or editing) a reference image as per a text prompt, namely, text-driven image-to-image translation. This paper contributes a novel, concise, and efficient approach that adapts pre-trained large-scale text-to-image (T2I) diffusion model to the image-to-image (I2I) paradigm in a plug-and-play manner, realizing high-quality and versatile text-driven I2I translation without any model training, model fine-tuning, or online optimization process. To guide T2I generation with a reference image, we propose to decompose diverse guiding factors with different frequency bands of diffusion features in the DCT spectral space, and accordingly devise a novel frequency band substitution layer which realizes dynamic control of the reference image to the T2I generation result in a plug-and-play manner. We demonstrate that our method allows flexible control over both guiding factor and guiding intensity of the reference image simply by tuning the type and bandwidth of the substituted frequency band, respectively. Extensive qualitative and quantitative experiments verify superiority of our approach over related methods in I2I translation visual quality, versatility, and controllability. The code is publicly available at: https://github.com/XiangGao1102/FBSDiff.
>
---
#### [replaced 006] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21401v3](http://arxiv.org/pdf/2506.21401v3)**

> **作者:** Zhirui Gao; Renjiao Yi; Yaqiao Dai; Xuening Zhu; Wei Chen; Chenyang Zhu; Kai Xu
>
> **备注:** Accepted by ICCV 2025, Code: https://github.com/zhirui-gao/Curve-Gaussian
>
> **摘要:** This paper presents an end-to-end framework for reconstructing 3D parametric curves directly from multi-view edge maps. Contrasting with existing two-stage methods that follow a sequential ``edge point cloud reconstruction and parametric curve fitting'' pipeline, our one-stage approach optimizes 3D parametric curves directly from 2D edge maps, eliminating error accumulation caused by the inherent optimization gap between disconnected stages. However, parametric curves inherently lack suitability for rendering-based multi-view optimization, necessitating a complementary representation that preserves their geometric properties while enabling differentiable rendering. We propose a novel bi-directional coupling mechanism between parametric curves and edge-oriented Gaussian components. This tight correspondence formulates a curve-aware Gaussian representation, \textbf{CurveGaussian}, that enables differentiable rendering of 3D curves, allowing direct optimization guided by multi-view evidence. Furthermore, we introduce a dynamically adaptive topology optimization framework during training to refine curve structures through linearization, merging, splitting, and pruning operations. Comprehensive evaluations on the ABC dataset and real-world benchmarks demonstrate our one-stage method's superiority over two-stage alternatives, particularly in producing cleaner and more robust reconstructions. Additionally, by directly optimizing parametric curves, our method significantly reduces the parameter count during training, achieving both higher efficiency and superior performance compared to existing approaches.
>
---
#### [replaced 007] One-Shot Affordance Grounding of Deformable Objects in Egocentric Organizing Scenes
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.01092v2](http://arxiv.org/pdf/2503.01092v2)**

> **作者:** Wanjun Jia; Fan Yang; Mengfei Duan; Xianchi Chen; Yinxi Wang; Yiming Jiang; Wenrui Chen; Kailun Yang; Zhiyong Li
>
> **备注:** Accepted to IROS 2025. Source code and benchmark dataset will be publicly available at https://github.com/Dikay1/OS-AGDO
>
> **摘要:** Deformable object manipulation in robotics presents significant challenges due to uncertainties in component properties, diverse configurations, visual interference, and ambiguous prompts. These factors complicate both perception and control tasks. To address these challenges, we propose a novel method for One-Shot Affordance Grounding of Deformable Objects (OS-AGDO) in egocentric organizing scenes, enabling robots to recognize previously unseen deformable objects with varying colors and shapes using minimal samples. Specifically, we first introduce the Deformable Object Semantic Enhancement Module (DefoSEM), which enhances hierarchical understanding of the internal structure and improves the ability to accurately identify local features, even under conditions of weak component information. Next, we propose the ORB-Enhanced Keypoint Fusion Module (OEKFM), which optimizes feature extraction of key components by leveraging geometric constraints and improves adaptability to diversity and visual interference. Additionally, we propose an instance-conditional prompt based on image data and task context, which effectively mitigates the issue of region ambiguity caused by prompt words. To validate these methods, we construct a diverse real-world dataset, AGDDO15, which includes 15 common types of deformable objects and their associated organizational actions. Experimental results demonstrate that our approach significantly outperforms state-of-the-art methods, achieving improvements of 6.2%, 3.2%, and 2.9% in KLD, SIM, and NSS metrics, respectively, while exhibiting high generalization performance. Source code and benchmark dataset are made publicly available at https://github.com/Dikay1/OS-AGDO.
>
---
#### [replaced 008] A Survey of Deep Learning for Geometry Problem Solving
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11936v2](http://arxiv.org/pdf/2507.11936v2)**

> **作者:** Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in progress
>
> **摘要:** Geometry problem solving is a key area of mathematical reasoning, which is widely involved in many important fields such as education, mathematical ability assessment of artificial intelligence, and multimodal ability assessment. In recent years, the rapid development of deep learning technology, especially the rise of multimodal large language models, has triggered a widespread research boom. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our goal is to provide a comprehensive and practical reference of deep learning for geometry problem solving to promote further developments in this field. We create a continuously updated list of papers on GitHub: https://github.com/majianz/dl4gps.
>
---
#### [replaced 009] VRU-Accident: A Vision-Language Benchmark for Video Question Answering and Dense Captioning for Accident Scene Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09815v2](http://arxiv.org/pdf/2507.09815v2)**

> **作者:** Younggun Kim; Ahmed S. Abdelrahman; Mohamed Abdel-Aty
>
> **备注:** 22 pages, 11 figures, 5 tables
>
> **摘要:** Ensuring the safety of vulnerable road users (VRUs), such as pedestrians and cyclists, is a critical challenge for autonomous driving systems, as crashes involving VRUs often result in severe or fatal consequences. While multimodal large language models (MLLMs) have shown promise in enhancing scene understanding and decision making in autonomous vehicles, there is currently no standardized benchmark to quantitatively evaluate their reasoning abilities in complex, safety-critical scenarios involving VRUs. To address this gap, we present VRU-Accident, a large-scale vision-language benchmark designed to evaluate MLLMs in high-risk traffic scenarios involving VRUs. VRU-Accident comprises 1K real-world dashcam accident videos, annotated with 6K multiple-choice question-answer pairs across six safety-critical categories (with 24K candidate options and 3.4K unique answer choices), as well as 1K dense scene descriptions. Unlike prior works, our benchmark focuses explicitly on VRU-vehicle accidents, providing rich, fine-grained annotations that capture both spatial-temporal dynamics and causal semantics of accidents. To assess the current landscape of MLLMs, we conduct a comprehensive evaluation of 17 state-of-the-art models on the multiple-choice VQA task and on the dense captioning task. Our findings reveal that while MLLMs perform reasonably well on visually grounded attributes, they face significant challenges in reasoning and describing accident causes, types, and preventability.
>
---
#### [replaced 010] Vision-based Conflict Detection within Crowds based on High-Resolution Human Pose Estimation for Smart and Safe Airport
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2207.00477v2](http://arxiv.org/pdf/2207.00477v2)**

> **作者:** Karan Kheta; Claire Delgove; Ruolin Liu; Adeola Aderogba; Marc-Olivier Pokam; Muhammed Mehmet Unal; Yang Xing; Weisi Guo
>
> **备注:** One of the authors has expressed privacy concerns and made a related request
>
> **摘要:** Future airports are becoming more complex and congested with the increasing number of travellers. While the airports are more likely to become hotspots for potential conflicts to break out which can cause serious delays to flights and several safety issues. An intelligent algorithm which renders security surveillance more effective in detecting conflicts would bring many benefits to the passengers in terms of their safety, finance, and travelling efficiency. This paper details the development of a machine learning model to classify conflicting behaviour in a crowd. HRNet is used to segment the images and then two approaches are taken to classify the poses of people in the frame via multiple classifiers. Among them, it was found that the support vector machine (SVM) achieved the most performant achieving precision of 94.37%. Where the model falls short is against ambiguous behaviour such as a hug or losing track of a subject in the frame. The resulting model has potential for deployment within an airport if improvements are made to cope with the vast number of potential passengers in view as well as training against further ambiguous behaviours which will arise in an airport setting. In turn, will provide the capability to enhance security surveillance and improve airport safety.
>
---
#### [replaced 011] CP-uniGuard: A Unified, Probability-Agnostic, and Adaptive Framework for Malicious Agent Detection and Defense in Multi-Agent Embodied Perception Systems
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2506.22890v2](http://arxiv.org/pdf/2506.22890v2)**

> **作者:** Senkang Hu; Yihang Tao; Guowen Xu; Xinyuan Qian; Yiqin Deng; Xianhao Chen; Sam Tak Wu Kwong; Yuguang Fang
>
> **摘要:** Collaborative Perception (CP) has been shown to be a promising technique for multi-agent autonomous driving and multi-agent robotic systems, where multiple agents share their perception information to enhance the overall perception performance and expand the perception range. However, in CP, an ego agent needs to receive messages from its collaborators, which makes it vulnerable to attacks from malicious agents. To address this critical issue, we propose a unified, probability-agnostic, and adaptive framework, namely, CP-uniGuard, which is a tailored defense mechanism for CP deployed by each agent to accurately detect and eliminate malicious agents in its collaboration network. Our key idea is to enable CP to reach a consensus rather than a conflict against an ego agent's perception results. Based on this idea, we first develop a probability-agnostic sample consensus (PASAC) method to effectively sample a subset of the collaborators and verify the consensus without prior probabilities of malicious agents. Furthermore, we define collaborative consistency loss (CCLoss) for object detection task and bird's eye view (BEV) segmentation task to capture the discrepancy between an ego agent and its collaborators, which is used as a verification criterion for consensus. In addition, we propose online adaptive threshold via dual sliding windows to dynamically adjust the threshold for consensus verification and ensure the reliability of the systems in dynamic environments. Finally, we conduct extensive experiments and demonstrate the effectiveness of our framework. Code will be released at https://github.com/CP-Security/CP-uniGuard.
>
---
#### [replaced 012] Physically Consistent Image Augmentation for Deep Learning in Mueller Matrix Polarimetry
- **分类: cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2411.07918v2](http://arxiv.org/pdf/2411.07918v2)**

> **作者:** Christopher Hahne; Omar Rodriguez-Nunez; Éléa Gros; Théotim Lucas; Ekkehard Hewer; Tatiana Novikova; Theoni Maragkou; Philippe Schucht; Richard McKinley
>
> **备注:** preprint
>
> **摘要:** Mueller matrix polarimetry captures essential information about polarized light interactions with a sample, presenting unique challenges for data augmentation in deep learning due to its distinct structure. While augmentations are an effective and affordable way to enhance dataset diversity and reduce overfitting, standard transformations like rotations and flips do not preserve the polarization properties in Mueller matrix images. To this end, we introduce a versatile simulation framework that applies physically consistent rotations and flips to Mueller matrices, tailored to maintain polarization fidelity. Our experimental results across multiple datasets reveal that conventional augmentations can lead to falsified results when applied to polarimetric data, underscoring the necessity of our physics-based approach. In our experiments, we first compare our polarization-specific augmentations against real-world captures to validate their physical consistency. We then apply these augmentations in a semantic segmentation task, achieving substantial improvements in model generalization and performance. This study underscores the necessity of physics-informed data augmentation for polarimetric imaging in deep learning (DL), paving the way for broader adoption and more robust applications across diverse research in the field. In particular, our framework unlocks the potential of DL models for polarimetric datasets with limited sample sizes. Our code implementation is available at github.com/hahnec/polar_augment.
>
---
#### [replaced 013] R1-Track: Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21980v3](http://arxiv.org/pdf/2506.21980v3)**

> **作者:** Biao Wang; Wenwen Li; Jiawei Ge
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** Visual single object tracking aims to continuously localize and estimate the scale of a target in subsequent video frames, given only its initial state in the first frame. This task has traditionally been framed as a template matching problem, evolving through major phases including correlation filters, two-stream networks, and one-stream networks with significant progress achieved. However, these methods typically require explicit classification and regression modeling, depend on supervised training with large-scale datasets, and are limited to the single task of tracking, lacking flexibility. In recent years, multi-modal large language models (MLLMs) have advanced rapidly. Open-source models like Qwen2.5-VL, a flagship MLLMs with strong foundational capabilities, demonstrate excellent performance in grounding tasks. This has spurred interest in applying such models directly to visual tracking. However, experiments reveal that Qwen2.5-VL struggles with template matching between image pairs (i.e., tracking tasks). Inspired by deepseek-R1, we fine-tuned Qwen2.5-VL using the group relative policy optimization (GRPO) reinforcement learning method on a small-scale dataset with a rule-based reward function. The resulting model, R1-Track, achieved notable performance on the GOT-10k benchmark. R1-Track supports flexible initialization via bounding boxes or text descriptions while retaining most of the original model's general capabilities. And we further discuss potential improvements for R1-Track. This rough technical report summarizes our findings as of May 2025.
>
---
#### [replaced 014] Online Episodic Memory Visual Query Localization with Egocentric Streaming Object Memory
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16934v2](http://arxiv.org/pdf/2411.16934v2)**

> **作者:** Zaira Manigrasso; Matteo Dunnhofer; Antonino Furnari; Moritz Nottebaum; Antonio Finocchiaro; Davide Marana; Rosario Forte; Giovanni Maria Farinella; Christian Micheloni
>
> **摘要:** Episodic memory retrieval enables wearable cameras to recall objects or events previously observed in video. However, existing formulations assume an "offline" setting with full video access at query time, limiting their applicability in real-world scenarios with power and storage-constrained wearable devices. Towards more application-ready episodic memory systems, we introduce Online Visual Query 2D (OVQ2D), a task where models process video streams online, observing each frame only once, and retrieve object localizations using a compact memory instead of full video history. We address OVQ2D with ESOM (Egocentric Streaming Object Memory), a novel framework integrating an object discovery module, an object tracking module, and a memory module that find, track, and store spatio-temporal object information for efficient querying. Experiments on Ego4D demonstrate ESOM's superiority over other online approaches, though OVQ2D remains challenging, with top performance at only ~4% success. ESOM's accuracy increases markedly with perfect object tracking (31.91%), discovery (40.55%), or both (81.92%), underscoring the need of applied research on these components.
>
---
#### [replaced 015] Visual-Language Model Knowledge Distillation Method for Image Quality Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15680v2](http://arxiv.org/pdf/2507.15680v2)**

> **作者:** Yongkang Hou; Jiarun Song
>
> **摘要:** Image Quality Assessment (IQA) is a core task in computer vision. Multimodal methods based on vision-language models, such as CLIP, have demonstrated exceptional generalization capabilities in IQA tasks. To address the issues of excessive parameter burden and insufficient ability to identify local distorted features in CLIP for IQA, this study proposes a visual-language model knowledge distillation method aimed at guiding the training of models with architectural advantages using CLIP's IQA knowledge. First, quality-graded prompt templates were designed to guide CLIP to output quality scores. Then, CLIP is fine-tuned to enhance its capabilities in IQA tasks. Finally, a modality-adaptive knowledge distillation strategy is proposed to achieve guidance from the CLIP teacher model to the student model. Our experiments were conducted on multiple IQA datasets, and the results show that the proposed method significantly reduces model complexity while outperforming existing IQA methods, demonstrating strong potential for practical deployment.
>
---
#### [replaced 016] Conformal Predictions for Human Action Recognition with Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.06631v2](http://arxiv.org/pdf/2502.06631v2)**

> **作者:** Bary Tim; Fuchs Clément; Macq Benoît
>
> **备注:** 6 pages, 7 figures, Accepted to ICIP 2025 Workshops
>
> **摘要:** Human-in-the-Loop (HITL) systems are essential in high-stakes, real-world applications where AI must collaborate with human decision-makers. This work investigates how Conformal Prediction (CP) techniques, which provide rigorous coverage guarantees, can enhance the reliability of state-of-the-art human action recognition (HAR) systems built upon Vision-Language Models (VLMs). We demonstrate that CP can significantly reduce the average number of candidate classes without modifying the underlying VLM. However, these reductions often result in distributions with long tails which can hinder their practical utility. To mitigate this, we propose tuning the temperature of the softmax prediction, without using additional calibration data. This work contributes to ongoing efforts for multi-modal human-AI interaction in dynamic real-world environments.
>
---
#### [replaced 017] GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.14456v2](http://arxiv.org/pdf/2507.14456v2)**

> **作者:** Chi Wan; Yixin Cui; Jiatong Du; Shuo Yang; Yulong Bai; Yanjun Huang
>
> **摘要:** End-to-end autonomous driving requires adaptive and robust handling of complex and diverse traffic environments. However, prevalent single-mode planning methods attempt to learn an overall policy while struggling to acquire diversified driving skills to handle diverse scenarios. Therefore, this paper proposes GEMINUS, a Mixture-of-Experts end-to-end autonomous driving framework featuring a Global Expert, a Scene-Adaptive Experts Group, and equipped with a Dual-aware Router. Specifically, the Global Expert is trained on the overall dataset, possessing robust performance. The Scene-Adaptive Experts are trained on corresponding scene subsets, achieving adaptive performance. The Dual-aware Router simultaneously considers scenario-level features and routing uncertainty to dynamically activate expert modules. Through the effective coupling of the Global Expert and the Scene-Adaptive Experts Group via the Dual-aware Router, GEMINUS achieves adaptive and robust performance in diverse scenarios. GEMINUS outperforms existing methods in the Bench2Drive closed-loop benchmark and achieves state-of-the-art performance in Driving Score and Success Rate, even with only monocular vision input. Furthermore, ablation studies demonstrate significant improvements over the original single-expert baseline: 7.67% in Driving Score, 22.06% in Success Rate, and 19.41% in MultiAbility-Mean. The code will be available at https://github.com/newbrains1/GEMINUS.
>
---
#### [replaced 018] VitaGlyph: Vitalizing Artistic Typography with Flexible Dual-branch Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.01738v3](http://arxiv.org/pdf/2410.01738v3)**

> **作者:** Kailai Feng; Yabo Zhang; Haodong Yu; Zhilong Ji; Jinfeng Bai; Hongzhi Zhang; Wangmeng Zuo
>
> **备注:** https://github.com/Carlofkl/VitaGlyph
>
> **摘要:** Artistic typography is a technique to visualize the meaning of input character in an imaginable and readable manner. With powerful text-to-image diffusion models, existing methods directly design the overall geometry and texture of input character, making it challenging to ensure both creativity and legibility. In this paper, we introduce a dual-branch, training-free method called VitaGlyph, enabling flexible artistic typography with controllable geometry changes while maintaining the readability. The key insight of VitaGlyph is to treat input character as a scene composed of a Subject and its Surrounding, which are rendered with varying degrees of geometric transformation. To enhance the visual appeal and creativity of the generated artistic typography, the subject flexibly expresses the essential concept of the input character, while the surrounding enriches relevant background without altering the shape, thus maintaining overall readability. Specifically, we implement VitaGlyph through a three-phase framework: (i) Knowledge Acquisition leverages large language models to design text descriptions for the subject and surrounding. (ii) Regional Interpretation detects the part that most closely matches the subject description and refines the structure via Semantic Typography. (iii) Attentional Compositional Generation separately renders the textures of the Subject and Surrounding regions and blends them in an attention-based manner. Experimental results demonstrate that VitaGlyph not only achieves better artistry and readability but also manages to depict multiple customized concepts, facilitating more creative and pleasing artistic typography generation. Our code will be made publicly available.
>
---
#### [replaced 019] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22139v3](http://arxiv.org/pdf/2506.22139v3)**

> **作者:** Shaojie Zhang; Jiahui Yang; Jianqin Yin; Zhenbo Luo; Jian Luan
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated significant success in visual understanding tasks. However, challenges persist in adapting these models for video comprehension due to the large volume of data and temporal complexity. Existing Video-LLMs using uniform frame sampling often struggle to capture the query-related crucial spatiotemporal clues of videos effectively. In this paper, we introduce Q-Frame, a novel approach for adaptive frame selection and multi-resolution scaling tailored to the video's content and the specific query. Q-Frame employs a training-free, plug-and-play strategy generated by a text-image matching network like CLIP, utilizing the Gumbel-Max trick for efficient frame selection. Q-Frame allows Video-LLMs to process more frames without exceeding computational limits, thereby preserving critical temporal and spatial information. We demonstrate Q-Frame's effectiveness through extensive experiments on benchmark datasets, including MLVU, LongVideoBench, and Video-MME, illustrating its superiority over existing methods and its applicability across various video understanding tasks.
>
---
#### [replaced 020] PEBench: A Fictitious Dataset to Benchmark Machine Unlearning for Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12545v2](http://arxiv.org/pdf/2503.12545v2)**

> **作者:** Zhaopan Xu; Pengfei Zhou; Weidong Tang; Jiaxin Ai; Wangbo Zhao; Kai Wang; Xiaojiang Peng; Wenqi Shao; Hongxun Yao; Kaipeng Zhang
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable success in vision-language tasks, but their reliance on vast, internet-sourced data raises significant privacy and security concerns. Machine unlearning (MU) has emerged as a critical technique to address these issues, enabling the selective removal of targeted information from pre-trained models without costly retraining. However, the evaluation of MU for MLLMs remains inadequate. Existing benchmarks often lack a comprehensive scope, focusing narrowly on entities while overlooking the unlearning of broader visual concepts and the inherent semantic coupling between them. To bridge this gap, we introduce, PEBench, a novel benchmark designed to facilitate a thorough assessment of MU in MLLMs. PEBench features a fictitious dataset of personal entities and corresponding event scenes to evaluate unlearning across these distinct yet entangled concepts. We leverage this benchmark to evaluate five MU methods, revealing their unique strengths and weaknesses. Our findings show that unlearning one concept can unintentionally degrade performance on related concepts within the same image, a challenge we term cross-concept interference. Furthermore, we demonstrate the difficulty of unlearning person and event concepts simultaneously and propose an effective method to mitigate these conflicting objectives. The source code and benchmark are publicly available at https://pebench.github.io.
>
---
#### [replaced 021] Unreal is all you need: Multimodal ISAC Data Simulation with Only One Engine
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08716v2](http://arxiv.org/pdf/2507.08716v2)**

> **作者:** Kongwu Huang; Shiyi Mu; Jun Jiang; Yuan Gao; Shugong Xu
>
> **摘要:** Scaling laws have achieved success in LLM and foundation models. To explore their potential in ISAC research, we propose Great-X. This single-engine multimodal data twin platform reconstructs the ray-tracing computation of Sionna within Unreal Engine and is deeply integrated with autonomous driving tools. This enables efficient and synchronized simulation of multimodal data, including CSI, RGB, Radar, and LiDAR. Based on this platform, we construct an open-source, large-scale, low-altitude UAV multimodal synaesthesia dataset named Great-MSD, and propose a baseline CSI-based UAV 3D localization algorithm, demonstrating its feasibility and generalizability across different CSI simulation engines. The related code and dataset will be made available at: https://github.com/hkw-xg/Great-MCD.
>
---
#### [replaced 022] INTER: Mitigating Hallucination in Large Vision-Language Models by Interaction Guidance Sampling
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05056v2](http://arxiv.org/pdf/2507.05056v2)**

> **作者:** Xin Dong; Shichao Dong; Jin Wang; Jing Huang; Li Zhou; Zenghui Sun; Lihua Jing; Jingsong Lan; Xiaoyong Zhu; Bo Zheng
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Hallucinations in large vision-language models (LVLMs) pose significant challenges for real-world applications, as LVLMs may generate responses that appear plausible yet remain inconsistent with the associated visual content. This issue rarely occurs in human cognition. We argue that this discrepancy arises from humans' ability to effectively leverage multimodal interaction information in data samples. Specifically, humans typically first gather multimodal information, analyze the interactions across modalities for understanding, and then express their understanding through language. Motivated by this observation, we conduct extensive experiments on popular LVLMs and obtained insights that surprisingly reveal human-like, though less pronounced, cognitive behavior of LVLMs on multimodal samples. Building on these findings, we further propose \textbf{INTER}: \textbf{Inter}action Guidance Sampling, a novel training-free algorithm that mitigate hallucinations without requiring additional data. Specifically, INTER explicitly guides LVLMs to effectively reapply their understanding of multimodal interaction information when generating responses, thereby reducing potential hallucinations. On six benchmarks including VQA and image captioning tasks, INTER achieves an average improvement of up to 3.4\% on five LVLMs compared to the state-of-the-art decoding strategy. The code will be released when the paper is accepted.
>
---
#### [replaced 023] Make Me Happier: Evoking Emotions Through Image Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.08255v4](http://arxiv.org/pdf/2403.08255v4)**

> **作者:** Qing Lin; Jingfeng Zhang; Yew-Soon Ong; Mengmi Zhang
>
> **摘要:** Despite the rapid progress in image generation, emotional image editing remains under-explored. The semantics, context, and structure of an image can evoke emotional responses, making emotional image editing techniques valuable for various real-world applications, including treatment of psychological disorders, commercialization of products, and artistic design. First, we present a novel challenge of emotion-evoked image generation, aiming to synthesize images that evoke target emotions while retaining the semantics and structures of the original scenes. To address this challenge, we propose a diffusion model capable of effectively understanding and editing source images to convey desired emotions and sentiments. Moreover, due to the lack of emotion editing datasets, we provide a unique dataset consisting of 340,000 pairs of images and their emotion annotations. Furthermore, we conduct human psychophysics experiments and introduce a new evaluation metric to systematically benchmark all the methods. Experimental results demonstrate that our method surpasses all competitive baselines. Our diffusion model is capable of identifying emotional cues from original images, editing images that elicit desired emotions, and meanwhile, preserving the semantic structure of the original images. All code, model, and dataset are available at GitHub.
>
---
#### [replaced 024] PRISM: High-Resolution & Precise Counterfactual Medical Image Generation using Language-guided Stable Diffusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.00196v2](http://arxiv.org/pdf/2503.00196v2)**

> **作者:** Amar Kumar; Anita Kriz; Mohammad Havaei; Tal Arbel
>
> **备注:** MIDL 2025
>
> **摘要:** Developing reliable and generalizable deep learning systems for medical imaging faces significant obstacles due to spurious correlations, data imbalances, and limited text annotations in datasets. Addressing these challenges requires architectures that are robust to the unique complexities posed by medical imaging data. Rapid advancements in vision-language foundation models within the natural image domain prompt the question of how they can be adapted for medical imaging tasks. In this work, we present PRISM, a framework that leverages foundation models to generate high-resolution, language-guided medical image counterfactuals using Stable Diffusion. Our approach demonstrates unprecedented precision in selectively modifying spurious correlations (the medical devices) and disease features, enabling the removal and addition of specific attributes while preserving other image characteristics. Through extensive evaluation, we show how PRISM advances counterfactual generation and enables the development of more robust downstream classifiers for clinically deployable solutions. To facilitate broader adoption and research, we make our code publicly available at https://github.com/Amarkr1/PRISM.
>
---
#### [replaced 025] Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models
- **分类: cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16104v2](http://arxiv.org/pdf/2505.16104v2)**

> **作者:** Yue Li; Xin Yi; Dongsheng Shi; Gerard de Melo; Xiaoling Wang; Linlin Wang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** With the increasing size of Large Vision-Language Models (LVLMs), network pruning techniques aimed at compressing models for deployment in resource-constrained environments have garnered significant attention. However, we observe that pruning often leads to a degradation in safety performance. To address this issue, we present a novel and lightweight approach, termed Hierarchical Safety Realignment (HSR). HSR operates by first quantifying the contribution of each attention head to safety, identifying the most critical ones, and then selectively restoring neurons directly within these attention heads that play a pivotal role in maintaining safety. This process hierarchically realigns the safety of pruned LVLMs, progressing from the attention head level to the neuron level. We validate HSR across various models and pruning strategies, consistently achieving notable improvements in safety performance. To our knowledge, this is the first work explicitly focused on restoring safety in LVLMs post-pruning.
>
---
#### [replaced 026] GraspDiffusion: Synthesizing Realistic Whole-body Hand-Object Interaction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.13911v2](http://arxiv.org/pdf/2410.13911v2)**

> **作者:** Patrick Kwon; Chen Chen; Hanbyul Joo
>
> **摘要:** Recent generative models can synthesize high-quality images but often fail to generate humans interacting with objects using their hands. This arises mostly from the model's misunderstanding of such interactions, and the hardships of synthesizing intricate regions of the body. In this paper, we propose GraspDiffusion, a novel generative method that creates realistic scenes of human-object interaction. Given a 3D object mesh, GraspDiffusion first constructs life-like whole-body poses with control over the object's location relative to the human body. This is achieved by separately leveraging the generative priors for 3D body and hand poses, optimizing them into a joint grasping pose. The resulting pose guides the image synthesis to correctly reflect the intended interaction, allowing the creation of realistic and diverse human-object interaction scenes. We demonstrate that GraspDiffusion can successfully tackle the relatively uninvestigated problem of generating full-bodied human-object interactions while outperforming previous methods. Code and models will be available at https://webtoon.github.io/GraspDiffusion
>
---
#### [replaced 027] Watermark Anything with Localized Messages
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2411.07231v2](http://arxiv.org/pdf/2411.07231v2)**

> **作者:** Tom Sander; Pierre Fernandez; Alain Durmus; Teddy Furon; Matthijs Douze
>
> **备注:** ICLR 2025
>
> **摘要:** Image watermarking methods are not tailored to handle small watermarked areas. This restricts applications in real-world scenarios where parts of the image may come from different sources or have been edited. We introduce a deep-learning model for localized image watermarking, dubbed the Watermark Anything Model (WAM). The WAM embedder imperceptibly modifies the input image, while the extractor segments the received image into watermarked and non-watermarked areas and recovers one or several hidden messages from the areas found to be watermarked. The models are jointly trained at low resolution and without perceptual constraints, then post-trained for imperceptibility and multiple watermarks. Experiments show that WAM is competitive with state-of-the art methods in terms of imperceptibility and robustness, especially against inpainting and splicing, even on high-resolution images. Moreover, it offers new capabilities: WAM can locate watermarked areas in spliced images and extract distinct 32-bit messages with less than 1 bit error from multiple small regions -- no larger than 10% of the image surface -- even for small 256x256 images. Training and inference code and model weights are available at https://github.com/facebookresearch/watermark-anything.
>
---
#### [replaced 028] Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.05195v3](http://arxiv.org/pdf/2411.05195v3)**

> **作者:** Siting Li; Pang Wei Koh; Simon Shaolei Du
>
> **备注:** ACL 2025; 19 pages, 3 figures
>
> **摘要:** Recent research has shown that CLIP models struggle with visual reasoning tasks that require grounding compositionality, understanding spatial relationships, or capturing fine-grained details. One natural hypothesis is that the CLIP vision encoder does not embed essential information for these tasks. However, we find that this is not always the case: The encoder gathers query-relevant visual information, while CLIP fails to extract it. In particular, we show that another branch of Vision-Language Models (VLMs), Generative Multimodal Large Language Models (MLLMs), achieve significantly higher accuracy than CLIP in many of these tasks using the same vision encoder and weights, indicating that these Generative MLLMs perceive more -- as they extract and utilize visual information more effectively. We conduct a series of controlled experiments and reveal that their success is attributed to multiple key design choices, including patch tokens, position embeddings, and prompt-based weighting. On the other hand, enhancing the training data alone or applying a stronger text encoder does not suffice to solve the task, and additional text tokens offer little benefit. Interestingly, we find that fine-grained visual reasoning is not exclusive to generative models trained by an autoregressive loss: When converted into CLIP-like encoders by contrastive finetuning, these MLLMs still outperform CLIP under the same cosine similarity-based evaluation protocol. Our study highlights the importance of VLM architectural choices and suggests directions for improving the performance of CLIP-like contrastive VLMs.
>
---
#### [replaced 029] Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.09110v2](http://arxiv.org/pdf/2502.09110v2)**

> **作者:** Eylon Mizrahi; Raz Lapid; Moshe Sipper
>
> **备注:** Accepted at SafeMM-AI @ ICCV 2025
>
> **摘要:** Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.
>
---
#### [replaced 030] RDD: Robust Feature Detector and Descriptor using Deformable Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08013v4](http://arxiv.org/pdf/2505.08013v4)**

> **作者:** Gonglin Chen; Tianwen Fu; Haiwei Chen; Wenbin Teng; Hanyuan Xiao; Yajie Zhao
>
> **摘要:** As a core step in structure-from-motion and SLAM, robust feature detection and description under challenging scenarios such as significant viewpoint changes remain unresolved despite their ubiquity. While recent works have identified the importance of local features in modeling geometric transformations, these methods fail to learn the visual cues present in long-range relationships. We present Robust Deformable Detector (RDD), a novel and robust keypoint detector/descriptor leveraging the deformable transformer, which captures global context and geometric invariance through deformable self-attention mechanisms. Specifically, we observed that deformable attention focuses on key locations, effectively reducing the search space complexity and modeling the geometric invariance. Furthermore, we collected an Air-to-Ground dataset for training in addition to the standard MegaDepth dataset. Our proposed method outperforms all state-of-the-art keypoint detection/description methods in sparse matching tasks and is also capable of semi-dense matching. To ensure comprehensive evaluation, we introduce two challenging benchmarks: one emphasizing large viewpoint and scale variations, and the other being an Air-to-Ground benchmark -- an evaluation setting that has recently gaining popularity for 3D reconstruction across different altitudes.
>
---
#### [replaced 031] EndoControlMag: Robust Endoscopic Vascular Motion Magnification with Periodic Reference Resetting and Hierarchical Tissue-aware Dual-Mask Contro
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15292v2](http://arxiv.org/pdf/2507.15292v2)**

> **作者:** An Wanga; Rulin Zhou; Mengya Xu; Yiru Ye; Longfei Gou; Yiting Chang; Hao Chen; Chwee Ming Lim; Jiankun Wang; Hongliang Ren
>
> **摘要:** Visualizing subtle vascular motions in endoscopic surgery is crucial for surgical precision and decision-making, yet remains challenging due to the complex and dynamic nature of surgical scenes. To address this, we introduce EndoControlMag, a training-free, Lagrangian-based framework with mask-conditioned vascular motion magnification tailored to endoscopic environments. Our approach features two key modules: a Periodic Reference Resetting (PRR) scheme that divides videos into short overlapping clips with dynamically updated reference frames to prevent error accumulation while maintaining temporal coherence, and a Hierarchical Tissue-aware Magnification (HTM) framework with dual-mode mask dilation. HTM first tracks vessel cores using a pretrained visual tracking model to maintain accurate localization despite occlusions and view changes. It then applies one of two adaptive softening strategies to surrounding tissues: motion-based softening that modulates magnification strength proportional to observed tissue displacement, or distance-based exponential decay that simulates biomechanical force attenuation. This dual-mode approach accommodates diverse surgical scenarios-motion-based softening excels with complex tissue deformations while distance-based softening provides stability during unreliable optical flow conditions. We evaluate EndoControlMag on our EndoVMM24 dataset spanning four different surgery types and various challenging scenarios, including occlusions, instrument disturbance, view changes, and vessel deformations. Quantitative metrics, visual assessments, and expert surgeon evaluations demonstrate that EndoControlMag significantly outperforms existing methods in both magnification accuracy and visual quality while maintaining robustness across challenging surgical conditions. The code, dataset, and video results are available at https://szupc.github.io/EndoControlMag/.
>
---
#### [replaced 032] Balancing Robustness and Efficiency in Embedded DNNs Through Activation Function Selection
- **分类: cs.LG; cs.AI; cs.AR; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.05119v2](http://arxiv.org/pdf/2504.05119v2)**

> **作者:** Jon Gutiérrez-Zaballa; Koldo Basterretxea; Javier Echanobe
>
> **摘要:** Machine learning-based embedded systems for safety-critical applications, such as aerospace and autonomous driving, must be robust to perturbations caused by soft errors. As transistor geometries shrink and voltages decrease, modern electronic devices become more susceptible to background radiation, increasing the concern about failures produced by soft errors. The resilience of deep neural networks (DNNs) to these errors depends not only on target device technology but also on model structure and the numerical representation and arithmetic precision of their parameters. Compression techniques like pruning and quantization, used to reduce memory footprint and computational complexity, alter both model structure and representation, affecting soft error robustness. In this regard, although often overlooked, the choice of activation functions (AFs) impacts not only accuracy and trainability but also compressibility and error resilience. This paper explores the use of bounded AFs to enhance robustness against parameter perturbations, while evaluating their effects on model accuracy, compressibility, and computational load with a technology-agnostic approach. We focus on encoder-decoder convolutional models developed for semantic segmentation of hyperspectral images with application to autonomous driving systems. Experiments are conducted on an AMD-Xilinx's KV260 SoM.
>
---
#### [replaced 033] Now and Future of Artificial Intelligence-based Signet Ring Cell Diagnosis: A Survey
- **分类: eess.IV; cs.CV; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2311.10118v2](http://arxiv.org/pdf/2311.10118v2)**

> **作者:** Zhu Meng; Junhao Dong; Limei Guo; Fei Su; Jiaxuan Liu; Guangxi Wang; Zhicheng Zhao
>
> **摘要:** Signet ring cells (SRCs), associated with a high propensity for peripheral metastasis and poor prognosis, critically influence surgical decision-making and outcome prediction. However, their detection remains challenging even for experienced pathologists. While artificial intelligence (AI)-based automated SRC diagnosis has gained increasing attention for its potential to enhance diagnostic efficiency and accuracy, existing methodologies lack systematic review. This gap impedes the assessment of disparities between algorithmic capabilities and clinical applicability. This paper presents a comprehensive survey of AI-driven SRC analysis from 2008 through June 2025. We systematically summarize the biological characteristics of SRCs and challenges in their automated identification. Representative algorithms are analyzed and categorized as unimodal or multi-modal approaches. Unimodal algorithms, encompassing image, omics, and text data, are reviewed; image-based ones are further subdivided into classification, detection, segmentation, and foundation model tasks. Multi-modal algorithms integrate two or more data modalities (images, omics, and text). Finally, by evaluating current methodological performance against clinical assistance requirements, we discuss unresolved challenges and future research directions in SRC analysis. This survey aims to assist researchers, particularly those without medical backgrounds, in understanding the landscape of SRC analysis and the prospects for intelligent diagnosis, thereby accelerating the translation of computational algorithms into clinical practice.
>
---
#### [replaced 034] USP: Unified Self-Supervised Pretraining for Image Generation and Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06132v3](http://arxiv.org/pdf/2503.06132v3)**

> **作者:** Xiangxiang Chu; Renda Li; Yong Wang
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** Recent studies have highlighted the interplay between diffusion models and representation learning. Intermediate representations from diffusion models can be leveraged for downstream visual tasks, while self-supervised vision models can enhance the convergence and generation quality of diffusion models. However, transferring pretrained weights from vision models to diffusion models is challenging due to input mismatches and the use of latent spaces. To address these challenges, we propose Unified Self-supervised Pretraining (USP), a framework that initializes diffusion models via masked latent modeling in a Variational Autoencoder (VAE) latent space. USP achieves comparable performance in understanding tasks while significantly improving the convergence speed and generation quality of diffusion models. Our code will be publicly available at https://github.com/AMAP-ML/USP.
>
---
#### [replaced 035] Do large language vision models understand 3D shapes?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.10908v5](http://arxiv.org/pdf/2412.10908v5)**

> **作者:** Sagi Eppel
>
> **摘要:** Large vision language models (LVLM) are the leading A.I approach for achieving a general visual understanding of the world. Models such as GPT, Claude, Gemini, and LLama can use images to understand and analyze complex visual scenes. 3D objects and shapes are the basic building blocks of the world, recognizing them is a fundamental part of human perception. The goal of this work is to test whether LVLMs truly understand 3D shapes by testing the models ability to identify and match objects of the exact same 3D shapes but with different orientations and materials/textures. A large number of test images were created using CGI with a huge number of highly diverse objects, materials, and scenes. The results of this test show that the ability of such models to match 3D shapes is significantly below humans but much higher than random guesses. Suggesting that the models have gained some abstract understanding of 3D shapes but still trail far beyond humans in this task. Mainly it seems that the models can easily identify the same object with a different orientation as well as matching identical 3D shapes of the same orientation but with different materials and textures. However, when both the object material and orientation are changed, all models perform poorly relative to humans. Code and benchmark are available.
>
---
#### [replaced 036] FiVE: A Fine-grained Video Editing Benchmark for Evaluating Emerging Diffusion and Rectified Flow Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13684v2](http://arxiv.org/pdf/2503.13684v2)**

> **作者:** Minghan Li; Chenxi Xie; Yichen Wu; Lei Zhang; Mengyu Wang
>
> **备注:** 24 pages, 14 figures, 16 tables
>
> **摘要:** Numerous text-to-video (T2V) editing methods have emerged recently, but the lack of a standardized benchmark for fair evaluation has led to inconsistent claims and an inability to assess model sensitivity to hyperparameters. Fine-grained video editing is crucial for enabling precise, object-level modifications while maintaining context and temporal consistency. To address this, we introduce FiVE, a Fine-grained Video Editing Benchmark for evaluating emerging diffusion and rectified flow models. Our benchmark includes 74 real-world videos and 26 generated videos, featuring 6 fine-grained editing types, 420 object-level editing prompt pairs, and their corresponding masks. Additionally, we adapt the latest rectified flow (RF) T2V generation models, Pyramid-Flow and Wan2.1, by introducing FlowEdit, resulting in training-free and inversion-free video editing models Pyramid-Edit and Wan-Edit. We evaluate five diffusion-based and two RF-based editing methods on our FiVE benchmark using 15 metrics, covering background preservation, text-video similarity, temporal consistency, video quality, and runtime. To further enhance object-level evaluation, we introduce FiVE-Acc, a novel metric leveraging Vision-Language Models (VLMs) to assess the success of fine-grained video editing. Experimental results demonstrate that RF-based editing significantly outperforms diffusion-based methods, with Wan-Edit achieving the best overall performance and exhibiting the least sensitivity to hyperparameters. More video demo available on the anonymous website: https://sites.google.com/view/five-benchmark
>
---
#### [replaced 037] Illuminating Darkness: Learning to Enhance Low-light Images In-the-Wild
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06898v2](http://arxiv.org/pdf/2503.06898v2)**

> **作者:** S M A Sharif; Abdur Rehman; Zain Ul Abidin; Fayaz Ali Dharejo; Radu Timofte; Rizwan Ali Naqvi
>
> **摘要:** Single-shot low-light image enhancement (SLLIE) remains challenging due to the limited availability of diverse, real-world paired datasets. To bridge this gap, we introduce the Low-Light Smartphone Dataset (LSD), a large-scale, high-resolution (4K+) dataset collected in the wild across a wide range of challenging lighting conditions (0.1 to 200 lux). LSD contains 6,425 precisely aligned low and normal-light image pairs, selected from over 8,000 dynamic indoor and outdoor scenes through multi-frame acquisition and expert evaluation. To evaluate generalization and aesthetic quality, we collect 2,117 unpaired low-light images from previously unseen devices. To fully exploit LSD, we propose TFFormer, a hybrid model that encodes luminance and chrominance (LC) separately to reduce color-structure entanglement. We further propose a cross-attention-driven joint decoder for context-aware fusion of LC representations, along with LC refinement and LC-guided supervision to significantly enhance perceptual fidelity and structural consistency. TFFormer achieves state-of-the-art results on LSD (+2.45 dB PSNR) and substantially improves downstream vision tasks, such as low-light object detection (+6.80 mAP on ExDark).
>
---
#### [replaced 038] VICI: VLM-Instructed Cross-view Image-localisation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04107v2](http://arxiv.org/pdf/2507.04107v2)**

> **作者:** Xiaohan Zhang; Tavis Shore; Chen Chen; Oscar Mendez; Simon Hadfield; Safwan Wshah
>
> **摘要:** In this paper, we present a high-performing solution to the UAVM 2025 Challenge, which focuses on matching narrow FOV street-level images to corresponding satellite imagery using the University-1652 dataset. As panoramic Cross-View Geo-Localisation nears peak performance, it becomes increasingly important to explore more practical problem formulations. Real-world scenarios rarely offer panoramic street-level queries; instead, queries typically consist of limited-FOV images captured with unknown camera parameters. Our work prioritises discovering the highest achievable performance under these constraints, pushing the limits of existing architectures. Our method begins by retrieving candidate satellite image embeddings for a given query, followed by a re-ranking stage that selectively enhances retrieval accuracy within the top candidates. This two-stage approach enables more precise matching, even under the significant viewpoint and scale variations inherent in the task. Through experimentation, we demonstrate that our approach achieves competitive results -specifically attaining R@1 and R@10 retrieval rates of \topone\% and \topten\% respectively. This underscores the potential of optimised retrieval and re-ranking strategies in advancing practical geo-localisation performance. Code is available at https://github.com/tavisshore/VICI.
>
---
#### [replaced 039] CL-Polyp: A Contrastive Learning-Enhanced Network for Accurate Polyp Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07154v2](http://arxiv.org/pdf/2507.07154v2)**

> **作者:** Desheng Li; Chaoliang Liu; Zhiyong Xiao
>
> **摘要:** Accurate segmentation of polyps from colonoscopy images is crucial for the early diagnosis and treatment of colorectal cancer. Most existing deep learning-based polyp segmentation methods adopt an Encoder-Decoder architecture, and some utilize multi-task frameworks that incorporate auxiliary tasks like classification to improve segmentation. However, these methods often need more labeled data and depend on task similarity, potentially limiting generalizability. To address these challenges, we propose CL-Polyp, a contrastive learning-enhanced polyp segmentation network. Our method uses contrastive learning to enhance the encoder's extraction of discriminative features by contrasting positive and negative sample pairs from polyp images. This self-supervised strategy improves visual representation without needing additional annotations. We also introduce two efficient, lightweight modules: the Modified Atrous Spatial Pyramid Pooling (MASPP) module for improved multi-scale feature fusion, and the Channel Concatenate and Element Add (CA) module to merge low-level and upsampled features for {enhanced} boundary reconstruction. Extensive experiments on five benchmark datasets-Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, and ETIS-show that CL-Polyp consistently surpasses state-of-the-art methods. Specifically, it enhances the IoU metric by 0.011 and 0.020 on the Kvasir-SEG and CVC-ClinicDB datasets, respectively, demonstrating its effectiveness in clinical polyp segmentation.
>
---
#### [replaced 040] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2410.13613v3](http://arxiv.org/pdf/2410.13613v3)**

> **作者:** Xinjie Zhang; Zhening Liu; Yifan Zhang; Xingtong Ge; Dailan He; Tongda Xu; Yan Wang; Zehong Lin; Shuicheng Yan; Jun Zhang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** 4D Gaussian Splatting (4DGS) has recently emerged as a promising technique for capturing complex dynamic 3D scenes with high fidelity. It utilizes a 4D Gaussian representation and a GPU-friendly rasterizer, enabling rapid rendering speeds. Despite its advantages, 4DGS faces significant challenges, notably the requirement of millions of 4D Gaussians, each with extensive associated attributes, leading to substantial memory and storage cost. This paper introduces a memory-efficient framework for 4DGS. We streamline the color attribute by decomposing it into a per-Gaussian direct color component with only 3 parameters and a shared lightweight alternating current color predictor. This approach eliminates the need for spherical harmonics coefficients, which typically involve up to 144 parameters in classic 4DGS, thereby creating a memory-efficient 4D Gaussian representation. Furthermore, we introduce an entropy-constrained Gaussian deformation technique that uses a deformation field to expand the action range of each Gaussian and integrates an opacity-based entropy loss to limit the number of Gaussians, thus forcing our model to use as few Gaussians as possible to fit a dynamic scene well. With simple half-precision storage and zip compression, our framework achieves a storage reduction by approximately 190$\times$ and 125$\times$ on the Technicolor and Neural 3D Video datasets, respectively, compared to the original 4DGS. Meanwhile, it maintains comparable rendering speeds and scene representation quality, setting a new standard in the field. Code is available at https://github.com/Xinjie-Q/MEGA.
>
---
#### [replaced 041] Bundle Adjustment in the Eager Mode
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.12190v2](http://arxiv.org/pdf/2409.12190v2)**

> **作者:** Zitong Zhan; Huan Xu; Zihang Fang; Xinpeng Wei; Yaoyu Hu; Chen Wang
>
> **摘要:** Bundle adjustment (BA) is a critical technique in various robotic applications such as simultaneous localization and mapping (SLAM), augmented reality (AR), and photogrammetry. BA optimizes parameters such as camera poses and 3D landmarks to align them with observations. With the growing importance of deep learning in perception systems, there is an increasing need to integrate BA with deep learning frameworks for enhanced reliability and performance. However, widely-used C++-based BA libraries, such as GTSAM, g$^2$o, and Ceres, lack native integration with modern deep learning libraries like PyTorch. This limitation affects their flexibility, adaptability, ease of debugging, and overall implementation efficiency. To address this gap, we introduce an eager-mode BA library seamlessly integrated with PyTorch with high efficiency. Our approach includes GPU-accelerated, differentiable, and sparse operations designed for \nth{2}-order optimization, Lie group and Lie algebra operations, and linear solvers. Our eager-mode BA on GPU demonstrates substantial runtime efficiency, achieving an average speedup of 18.5$\times$, 22$\times$, and 23$\times$ compared to GTSAM, g$^2$o, and Ceres, respectively. The source code will be available at https://github.com/sair-lab/bae.
>
---
#### [replaced 042] RadAlign: Advancing Radiology Report Generation with Vision-Language Concept Alignment
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.07525v2](http://arxiv.org/pdf/2501.07525v2)**

> **作者:** Difei Gu; Yunhe Gao; Yang Zhou; Mu Zhou; Dimitris Metaxas
>
> **备注:** Accepted to MICCAI 2025
>
> **摘要:** Automated chest radiographs interpretation requires both accurate disease classification and detailed radiology report generation, presenting a significant challenge in the clinical workflow. Current approaches either focus on classification accuracy at the expense of interpretability or generate detailed but potentially unreliable reports through image captioning techniques. In this study, we present RadAlign, a novel framework that combines the predictive accuracy of vision-language models (VLMs) with the reasoning capabilities of large language models (LLMs). Inspired by the radiologist's workflow, RadAlign first employs a specialized VLM to align visual features with key medical concepts, achieving superior disease classification with an average AUC of 0.885 across multiple diseases. These recognized medical conditions, represented as text-based concepts in the aligned visual-language space, are then used to prompt LLM-based report generation. Enhanced by a retrieval-augmented generation mechanism that grounds outputs in similar historical cases, RadAlign delivers superior report quality with a GREEN score of 0.678, outperforming state-of-the-art methods' 0.634. Our framework maintains strong clinical interpretability while reducing hallucinations, advancing automated medical imaging and report analysis through integrated predictive and generative AI. Code is available at https://github.com/difeigu/RadAlign.
>
---
#### [replaced 043] Distilling Diversity and Control in Diffusion Models
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10637v3](http://arxiv.org/pdf/2503.10637v3)**

> **作者:** Rohit Gandikota; David Bau
>
> **备注:** Project Page: https://distillation.baulab.info
>
> **摘要:** Distilled diffusion models suffer from a critical limitation: reduced sample diversity compared to their base counterparts. In this work, we uncover that despite this diversity loss, distilled models retain the fundamental concept representations of base models. We demonstrate control distillation - where control mechanisms like Concept Sliders and LoRAs trained on base models can be seamlessly transferred to distilled models and vice-versa, effectively distilling control without any retraining. This preservation of representational structure prompted our investigation into the mechanisms of sample-diversity collapse during distillation. To understand how distillation affects diversity, we utilize $\hat{\mathbf{x}}_{0}$ visualization as an analysis and debugging tool to reveal how models predict final outputs at intermediate steps. Through $\hat{\mathbf{x}}_{0}$ visualization, we identify generation artifacts, inconsistencies, and demonstrate that initial diffusion timesteps disproportionately determine output diversity, while later steps primarily refine details. Based on these insights, we introduce diversity distillation - a hybrid inference approach that strategically employs the base model for only the first critical timestep before transitioning to the efficient distilled model. Our experiments demonstrate that this simple modification not only restores the diversity capabilities from base to distilled models but surprisingly exceeds it, while maintaining nearly the computational efficiency of distilled inference, all without requiring additional training or model modifications. Our code and data are available at https://distillation.baulab.info/
>
---
#### [replaced 044] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.08629v2](http://arxiv.org/pdf/2412.08629v2)**

> **作者:** Vladimir Kulikov; Matan Kleiner; Inbar Huberman-Spiegelglas; Tomer Michaeli
>
> **备注:** ICCV 2025. Project's webpage at https://matankleiner.github.io/flowedit/
>
> **摘要:** Editing real images using a pre-trained text-to-image (T2I) diffusion/flow model often involves inverting the image into its corresponding noise map. However, inversion by itself is typically insufficient for obtaining satisfactory results, and therefore many methods additionally intervene in the sampling process. Such methods achieve improved results but are not seamlessly transferable between model architectures. Here, we introduce FlowEdit, a text-based editing method for pre-trained T2I flow models, which is inversion-free, optimization-free and model agnostic. Our method constructs an ODE that directly maps between the source and target distributions (corresponding to the source and target text prompts) and achieves a lower transport cost than the inversion approach. This leads to state-of-the-art results, as we illustrate with Stable Diffusion 3 and FLUX. Code and examples are available on the project's webpage.
>
---
#### [replaced 045] MCP-MedSAM: A Powerful Lightweight Medical Segment Anything Model Trained with a Single GPU in Just One Day
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05888v3](http://arxiv.org/pdf/2412.05888v3)**

> **作者:** Donghang Lyu; Ruochen Gao; Marius Staring
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA)
>
> **摘要:** Medical image segmentation involves partitioning medical images into meaningful regions, with a focus on identifying anatomical structures and lesions. It has broad applications in healthcare, and deep learning methods have enabled significant advancements in automating this process. Recently, the introduction of the Segmentation Anything Model (SAM), the first foundation model for segmentation task, has prompted researchers to adapt it for the medical domain to improve performance across various tasks. However, SAM's large model size and high GPU requirements hinder its scalability and development in the medical domain. In this work, we propose MCP-MedSAM, a powerful and lightweight medical SAM model designed to be trainable on a single A100 GPU with 40GB of memory within one day while delivering superior segmentation performance. Recognizing the significant internal differences between modalities and the need for direct segmentation target information within bounding boxes, we introduce two kinds of prompts: the modality prompt and the content prompt. After passing through the prompt encoder, their embedding representations can further improve the segmentation performance by incorporating more relevant information without adding significant training overhead. Additionally, we adopt an effective modality-based data sampling strategy to address data imbalance between modalities, ensuring more balanced performance across all modalities. Our method was trained and evaluated using a large-scale challenge dataset, compared to top-ranking methods on the challenge leaderboard, MCP-MedSAM achieved superior performance while requiring only one day of training on a single GPU. The code is publicly available at \textcolor{blue}{https://github.com/dong845/MCP-MedSAM}.}
>
---
#### [replaced 046] AV-Reasoner: Improving and Benchmarking Clue-Grounded Audio-Visual Counting for MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05328v2](http://arxiv.org/pdf/2506.05328v2)**

> **作者:** Lidong Lu; Guo Chen; Zhiqi Li; Yicheng Liu; Tong Lu
>
> **备注:** 21 pages, 11 figures
>
> **摘要:** Despite progress in video understanding, current MLLMs struggle with counting tasks. Existing benchmarks are limited by short videos, close-set queries, lack of clue annotations, and weak multimodal coverage. In this paper, we introduce CG-AV-Counting, a manually-annotated clue-grounded counting benchmark with 1,027 multimodal questions and 5,845 annotated clues over 497 long videos. It supports both black-box and white-box evaluation, serving as a comprehensive testbed for both end-to-end and reasoning-based counting. To explore ways to improve model's counting capability, we propose AV-Reasoner, a model trained with GRPO and curriculum learning to generalize counting ability from related tasks. AV-Reasoner achieves state-of-the-art results across multiple benchmarks, demonstrating the effectiveness of reinforcement learning. However, experiments show that on out-of-domain benchmarks, reasoning in the language space fails to bring performance gains. The code and benchmark have been released on https://av-reasoner.github.io.
>
---
#### [replaced 047] Balanced Image Stylization with Style Matching Score
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.07601v2](http://arxiv.org/pdf/2503.07601v2)**

> **作者:** Yuxin Jiang; Liming Jiang; Shuai Yang; Jia-Wei Liu; Ivor Tsang; Mike Zheng Shou
>
> **备注:** ICCV 2025. Code: https://github.com/showlab/SMS Project page: https://yuxinn-j.github.io/projects/SMS.html
>
> **摘要:** We present Style Matching Score (SMS), a novel optimization method for image stylization with diffusion models. Balancing effective style transfer with content preservation is a long-standing challenge. Unlike existing efforts, our method reframes image stylization as a style distribution matching problem. The target style distribution is estimated from off-the-shelf style-dependent LoRAs via carefully designed score functions. To preserve content information adaptively, we propose Progressive Spectrum Regularization, which operates in the frequency domain to guide stylization progressively from low-frequency layouts to high-frequency details. In addition, we devise a Semantic-Aware Gradient Refinement technique that leverages relevance maps derived from diffusion semantic priors to selectively stylize semantically important regions. The proposed optimization formulation extends stylization from pixel space to parameter space, readily applicable to lightweight feedforward generators for efficient one-step stylization. SMS effectively balances style alignment and content preservation, outperforming state-of-the-art approaches, verified by extensive experiments.
>
---
#### [replaced 048] Revealing Bias Formation in Deep Neural Networks Through the Geometric Mechanisms of Human Visual Decoupling
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11809v3](http://arxiv.org/pdf/2502.11809v3)**

> **作者:** Yanbiao Ma; Bowei Liu; Boyuan Gao; Wei Dai; Jiayi Chen; Shuo Li; Andi Zhang
>
> **摘要:** Deep neural networks (DNNs) often exhibit biases toward certain categories during object recognition, even under balanced training data conditions. The intrinsic mechanisms underlying these biases remain unclear. Inspired by the human visual system, which decouples object manifolds through hierarchical processing to achieve object recognition, we propose a geometric analysis framework linking the geometric complexity of class-specific perceptual manifolds in DNNs to model bias. Our findings reveal that differences in geometric complexity can lead to varying recognition capabilities across categories, introducing biases. To support this analysis, we present the Perceptual-Manifold-Geometry library, designed for calculating the geometric properties of perceptual manifolds.
>
---
#### [replaced 049] InternAgent: When Agent Becomes the Scientist -- Building Closed-Loop System from Hypothesis to Verification
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16938v3](http://arxiv.org/pdf/2505.16938v3)**

> **作者:** InternAgent Team; Bo Zhang; Shiyang Feng; Xiangchao Yan; Jiakang Yuan; Runmin Ma; Yusong Hu; Zhiyin Yu; Xiaohan He; Songtao Huang; Shaowei Hou; Zheng Nie; Zhilong Wang; Jinyao Liu; Tianshuo Peng; Peng Ye; Dongzhan Zhou; Shufei Zhang; Xiaosong Wang; Yilan Zhang; Meng Li; Zhongying Tu; Xiangyu Yue; Wangli Ouyang; Bowen Zhou; Lei Bai
>
> **备注:** Code: https://github.com/Alpha-Innovator/InternAgent, HomePage: https://alpha-innovator.github.io/InternAgent-project-page
>
> **摘要:** Artificial Intelligence (AI) is accelerating the transformation of scientific research paradigms, not only enhancing research efficiency but also driving innovation. We introduce InternAgent, a unified closed-loop multi-agent framework to conduct Autonomous Scientific Research (ASR) across various scientific research fields, enabling researchers to tackle complicated problems in these fields with unprecedented speed and precision. InternAgent highlights three key advantages: 1) Scalability: InternAgent has demonstrated its versatility across 12 scientific research tasks, capable of generating innovative ideas to enhance the performance of baseline code. 2) Interactivity: InternAgent provides an interface for human expert feedback and multi-agent interaction in automated end-to-end processes, allowing for the seamless integration of domain expert knowledge. 3) Efficiency: InternAgent has achieved promising performance gains in several scientific fields with significantly less time cost compared to human efforts. For instance, in reaction yield prediction, it increased from 27.6% to 35.4% in just 12 hours; in enhancer activity prediction, accuracy rose from 0.65 to 0.79 with only 4 hours of processing; and in 2D semantic segmentation, precision advanced from 78.8% to 81.0% in a mere 30 hours.
>
---
#### [replaced 050] Predicting the Reliability of an Image Classifier under Image Distortion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.16881v2](http://arxiv.org/pdf/2412.16881v2)**

> **作者:** Dang Nguyen; Sunil Gupta; Kien Do; Svetha Venkatesh
>
> **摘要:** In image classification tasks, deep learning models are vulnerable to image distortions i.e. their accuracy significantly drops if the input images are distorted. An image-classifier is considered "reliable" if its accuracy on distorted images is above a user-specified threshold. For a quality control purpose, it is important to predict if the image-classifier is unreliable/reliable under a distortion level. In other words, we want to predict whether a distortion level makes the image-classifier "non-reliable" or "reliable". Our solution is to construct a training set consisting of distortion levels along with their "non-reliable" or "reliable" labels, and train a machine learning predictive model (called distortion-classifier) to classify unseen distortion levels. However, learning an effective distortion-classifier is a challenging problem as the training set is highly imbalanced. To address this problem, we propose a Gaussian process based method to rebalance the training set. We conduct extensive experiments to show that our method significantly outperforms several baselines on six popular image datasets.
>
---
#### [replaced 051] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.05710v2](http://arxiv.org/pdf/2501.05710v2)**

> **作者:** Shengqi Dang; Yi He; Long Ling; Ziqing Qian; Nanxuan Zhao; Nan Cao
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Recent research shows that emotions can enhance users' cognition and influence information communication. While research on visual emotion analysis is extensive, limited work has been done on helping users generate emotionally rich image content. Existing work on emotional image generation relies on discrete emotion categories, making it challenging to capture complex and subtle emotional nuances accurately. Additionally, these methods struggle to control the specific content of generated images based on text prompts. In this work, we introduce the new task of continuous emotional image content generation (C-EICG) and present EmotiCrafter, an emotional image generation model that generates images based on text prompts and Valence-Arousal values. Specifically, we propose a novel emotion-embedding mapping network that embeds Valence-Arousal values into textual features, enabling the capture of specific emotions in alignment with intended input prompts. Additionally, we introduce a loss function to enhance emotion expression. The experimental results show that our method effectively generates images representing specific emotions with the desired content and outperforms existing techniques.
>
---
#### [replaced 052] Memory-Augmented SAM2 for Training-Free Surgical Video Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09577v2](http://arxiv.org/pdf/2507.09577v2)**

> **作者:** Ming Yin; Fu Wang; Xujiong Ye; Yanda Meng; Zeyu Fu
>
> **备注:** Accepted in MICCAI 2025
>
> **摘要:** Surgical video segmentation is a critical task in computer-assisted surgery, essential for enhancing surgical quality and patient outcomes. Recently, the Segment Anything Model 2 (SAM2) framework has demonstrated remarkable advancements in both image and video segmentation. However, the inherent limitations of SAM2's greedy selection memory design are amplified by the unique properties of surgical videos-rapid instrument movement, frequent occlusion, and complex instrument-tissue interaction-resulting in diminished performance in the segmentation of complex, long videos. To address these challenges, we introduce Memory Augmented (MA)-SAM2, a training-free video object segmentation strategy, featuring novel context-aware and occlusion-resilient memory models. MA-SAM2 exhibits strong robustness against occlusions and interactions arising from complex instrument movements while maintaining accuracy in segmenting objects throughout videos. Employing a multi-target, single-loop, one-prompt inference further enhances the efficiency of the tracking process in multi-instrument videos. Without introducing any additional parameters or requiring further training, MA-SAM2 achieved performance improvements of 4.36% and 6.1% over SAM2 on the EndoVis2017 and EndoVis2018 datasets, respectively, demonstrating its potential for practical surgical applications.
>
---
#### [replaced 053] GR-3 Technical Report
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15493v2](http://arxiv.org/pdf/2507.15493v2)**

> **作者:** Chilam Cheang; Sijin Chen; Zhongren Cui; Yingdong Hu; Liqun Huang; Tao Kong; Hang Li; Yifeng Li; Yuxiao Liu; Xiao Ma; Hao Niu; Wenxuan Ou; Wanli Peng; Zeyu Ren; Haixin Shi; Jiawen Tian; Hongtao Wu; Xin Xiao; Yuyang Xiao; Jiafeng Xu; Yichu Yang
>
> **备注:** Tech report. Authors are listed in alphabetical order. Project page: https://seed.bytedance.com/GR3/
>
> **摘要:** We report our recent progress towards building generalist robot policies, the development of GR-3. GR-3 is a large-scale vision-language-action (VLA) model. It showcases exceptional capabilities in generalizing to novel objects, environments, and instructions involving abstract concepts. Furthermore, it can be efficiently fine-tuned with minimal human trajectory data, enabling rapid and cost-effective adaptation to new settings. GR-3 also excels in handling long-horizon and dexterous tasks, including those requiring bi-manual manipulation and mobile movement, showcasing robust and reliable performance. These capabilities are achieved through a multi-faceted training recipe that includes co-training with web-scale vision-language data, efficient fine-tuning from human trajectory data collected via VR devices, and effective imitation learning with robot trajectory data. In addition, we introduce ByteMini, a versatile bi-manual mobile robot designed with exceptional flexibility and reliability, capable of accomplishing a wide range of tasks when integrated with GR-3. Through extensive real-world experiments, we show GR-3 surpasses the state-of-the-art baseline method, $\pi_0$, on a wide variety of challenging tasks. We hope GR-3 can serve as a step towards building generalist robots capable of assisting humans in daily life.
>
---
#### [replaced 054] Prompt4Trust: A Reinforcement Learning Prompt Augmentation Framework for Clinically-Aligned Confidence Calibration in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.09279v3](http://arxiv.org/pdf/2507.09279v3)**

> **作者:** Anita Kriz; Elizabeth Laura Janes; Xing Shen; Tal Arbel
>
> **备注:** Accepted to ICCV 2025 Workshop CVAMD
>
> **摘要:** Multimodal large language models (MLLMs) hold considerable promise for applications in healthcare. However, their deployment in safety-critical settings is hindered by two key limitations: (i) sensitivity to prompt design, and (ii) a tendency to generate incorrect responses with high confidence. As clinicians may rely on a model's stated confidence to gauge the reliability of its predictions, it is especially important that when a model expresses high confidence, it is also highly accurate. We introduce Prompt4Trust, the first reinforcement learning (RL) framework for prompt augmentation targeting confidence calibration in MLLMs. A lightweight LLM is trained to produce context-aware auxiliary prompts that guide a downstream task MLLM to generate responses in which the expressed confidence more accurately reflects predictive accuracy. Unlike conventional calibration techniques, Prompt4Trust specifically prioritizes aspects of calibration most critical for safe and trustworthy clinical decision-making. Beyond improvements driven by this clinically motivated calibration objective, our proposed method also improves task accuracy, achieving state-of-the-art medical visual question answering (VQA) performance on the PMC-VQA benchmark, which is composed of multiple-choice questions spanning diverse medical imaging modalities. Moreover, our framework trained with a small downstream task MLLM showed promising zero-shot generalization to larger MLLMs in our experiments, suggesting the potential for scalable calibration without the associated computational costs. This work demonstrates the potential of automated yet human-aligned prompt engineering for improving the the trustworthiness of MLLMs in safety critical settings. Our codebase can be found at https://github.com/xingbpshen/prompt4trust.
>
---
#### [replaced 055] Mind the Gap: Bridging Occlusion in Gait Recognition via Residual Gap Correction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.10978v2](http://arxiv.org/pdf/2507.10978v2)**

> **作者:** Ayush Gupta; Siyuan Huang; Rama Chellappa
>
> **备注:** Accepted at IJCB 2025
>
> **摘要:** Gait is becoming popular as a method of person re-identification because of its ability to identify people at a distance. However, most current works in gait recognition do not address the practical problem of occlusions. Among those which do, some require paired tuples of occluded and holistic sequences, which are impractical to collect in the real world. Further, these approaches work on occlusions but fail to retain performance on holistic inputs. To address these challenges, we propose RG-Gait, a method for residual correction for occluded gait recognition with holistic retention. We model the problem as a residual learning task, conceptualizing the occluded gait signature as a residual deviation from the holistic gait representation. Our proposed network adaptively integrates the learned residual, significantly improving performance on occluded gait sequences without compromising the holistic recognition accuracy. We evaluate our approach on the challenging Gait3D, GREW and BRIAR datasets and show that learning the residual can be an effective technique to tackle occluded gait recognition with holistic retention. We release our code publicly at https://github.com/Ayush-00/rg-gait.
>
---
#### [replaced 056] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23468v2](http://arxiv.org/pdf/2506.23468v2)**

> **作者:** Xuan Yao; Junyu Gao; Changsheng Xu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires agents to execute sequential navigation actions in complex environments guided by natural language instructions. Current approaches often struggle with generalizing to novel environments and adapting to ongoing changes during navigation. Inspired by human cognition, we present NavMorph, a self-evolving world model framework that enhances environmental understanding and decision-making in VLN-CE tasks. NavMorph employs compact latent representations to model environmental dynamics, equipping agents with foresight for adaptive planning and policy refinement. By integrating a novel Contextual Evolution Memory, NavMorph leverages scene-contextual information to support effective navigation while maintaining online adaptability. Extensive experiments demonstrate that our method achieves notable performance improvements on popular VLN-CE benchmarks. Code is available at https://github.com/Feliciaxyao/NavMorph.
>
---
#### [replaced 057] ViP$^2$-CLIP: Visual-Perception Prompting with Unified Alignment for Zero-Shot Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17692v2](http://arxiv.org/pdf/2505.17692v2)**

> **作者:** Ziteng Yang; Jingzehua Xu; Yanshu Li; Zepeng Li; Yeqiang Wang; Xinghui Li
>
> **摘要:** Zero-shot anomaly detection (ZSAD) aims to detect anomalies without any target domain training samples, relying solely on external auxiliary data. Existing CLIP-based methods attempt to activate the model's ZSAD potential via handcrafted or static learnable prompts. The former incur high engineering costs and limited semantic coverage, whereas the latter apply identical descriptions across diverse anomaly types, thus fail to adapt to complex variations. Furthermore, since CLIP is originally pretrained on large-scale classification tasks, its anomaly segmentation quality is highly sensitive to the exact wording of class names, severely constraining prompting strategies that depend on class labels. To address these challenges, we introduce ViP$^{2}$-CLIP. The key insight of ViP$^{2}$-CLIP is a Visual-Perception Prompting (ViP-Prompt) mechanism, which fuses global and multi-scale local visual context to adaptively generate fine-grained textual prompts, eliminating manual templates and class-name priors. This design enables our model to focus on precise abnormal regions, making it particularly valuable when category labels are ambiguous or privacy-constrained. Extensive experiments on 15 industrial and medical benchmarks demonstrate that ViP$^{2}$-CLIP achieves state-of-the-art performance and robust cross-domain generalization.
>
---
#### [replaced 058] Recurrent Inference Machine for Medical Image Registration
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.13413v2](http://arxiv.org/pdf/2406.13413v2)**

> **作者:** Yi Zhang; Yidong Zhao; Hui Xue; Peter Kellman; Stefan Klein; Qian Tao
>
> **备注:** Preprint version. Accepted by Medical Image Analysis
>
> **摘要:** Image registration is essential for medical image applications where alignment of voxels across multiple images is needed for qualitative or quantitative analysis. With recent advancements in deep neural networks and parallel computing, deep learning-based medical image registration methods become competitive with their flexible modelling and fast inference capabilities. However, compared to traditional optimization-based registration methods, the speed advantage may come at the cost of registration performance at inference time. Besides, deep neural networks ideally demand large training datasets while optimization-based methods are training-free. To improve registration accuracy and data efficiency, we propose a novel image registration method, termed Recurrent Inference Image Registration (RIIR) network. RIIR is formulated as a meta-learning solver to the registration problem in an iterative manner. RIIR addresses the accuracy and data efficiency issues, by learning the update rule of optimization, with implicit regularization combined with explicit gradient input. We evaluated RIIR extensively on brain MRI and quantitative cardiac MRI datasets, in terms of both registration accuracy and training data efficiency. Our experiments showed that RIIR outperformed a range of deep learning-based methods, even with only $5\%$ of the training data, demonstrating high data efficiency. Key findings from our ablation studies highlighted the important added value of the hidden states introduced in the recurrent inference framework for meta-learning. Our proposed RIIR offers a highly data-efficient framework for deep learning-based medical image registration.
>
---
#### [replaced 059] MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02358v5](http://arxiv.org/pdf/2502.02358v5)**

> **作者:** Ziyan Guo; Zeyu Hu; De Wen Soh; Na Zhao
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Human motion generation and editing are key components of computer vision. However, current approaches in this field tend to offer isolated solutions tailored to specific tasks, which can be inefficient and impractical for real-world applications. While some efforts have aimed to unify motion-related tasks, these methods simply use different modalities as conditions to guide motion generation. Consequently, they lack editing capabilities, fine-grained control, and fail to facilitate knowledge sharing across tasks. To address these limitations and provide a versatile, unified framework capable of handling both human motion generation and editing, we introduce a novel paradigm: \textbf{Motion-Condition-Motion}, which enables the unified formulation of diverse tasks with three concepts: source motion, condition, and target motion. Based on this paradigm, we propose a unified framework, \textbf{MotionLab}, which incorporates rectified flows to learn the mapping from source motion to target motion, guided by the specified conditions. In MotionLab, we introduce the 1) MotionFlow Transformer to enhance conditional generation and editing without task-specific modules; 2) Aligned Rotational Position Encoding to guarantee the time synchronization between source motion and target motion; 3) Task Specified Instruction Modulation; and 4) Motion Curriculum Learning for effective multi-task learning and knowledge sharing across tasks. Notably, our MotionLab demonstrates promising generalization capabilities and inference efficiency across multiple benchmarks for human motion. Our code and additional video results are available at: https://diouo.github.io/motionlab.github.io/.
>
---
#### [replaced 060] GS-TransUNet: Integrated 2D Gaussian Splatting and Transformer UNet for Accurate Skin Lesion Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16748v2](http://arxiv.org/pdf/2502.16748v2)**

> **作者:** Anand Kumar; Kavinder Roghit Kanthen; Josna John
>
> **备注:** 12 pages, 7 figures, SPIE Medical Imaging 2025. 13407-1340736
>
> **摘要:** We can achieve fast and consistent early skin cancer detection with recent developments in computer vision and deep learning techniques. However, the existing skin lesion segmentation and classification prediction models run independently, thus missing potential efficiencies from their integrated execution. To unify skin lesion analysis, our paper presents the Gaussian Splatting - Transformer UNet (GS-TransUNet), a novel approach that synergistically combines 2D Gaussian splatting with the Transformer UNet architecture for automated skin cancer diagnosis. Our unified deep learning model efficiently delivers dual-function skin lesion classification and segmentation for clinical diagnosis. Evaluated on ISIC-2017 and PH2 datasets, our network demonstrates superior performance compared to existing state-of-the-art models across multiple metrics through 5-fold cross-validation. Our findings illustrate significant advancements in the precision of segmentation and classification. This integration sets new benchmarks in the field and highlights the potential for further research into multi-task medical image analysis methodologies, promising enhancements in automated diagnostic systems.
>
---
#### [replaced 061] AI-Enhanced Precision in Sport Taekwondo: Increasing Fairness, Speed, and Trust in Competition (FST.ai)
- **分类: cs.CV; cs.AI; 68T45; I.2.10**

- **链接: [http://arxiv.org/pdf/2507.14657v2](http://arxiv.org/pdf/2507.14657v2)**

> **作者:** Keivan Shariatmadar; Ahmad Osman
>
> **备注:** 24 pages, 9 figures
>
> **摘要:** The integration of Artificial Intelligence (AI) into sports officiating represents a paradigm shift in how decisions are made in competitive environments. Traditional manual systems, even when supported by Instant Video Replay (IVR), often suffer from latency, subjectivity, and inconsistent enforcement, undermining fairness and athlete trust. This paper introduces 'FST.ai' -- which is developed under the 'R3AL.ai' project, which serves as its Principal Investigator: r3al.ai -- a novel AI-powered framework designed to enhance officiating in Sport Taekwondo, particularly focusing on the complex task of real-time head kick detection and scoring. Leveraging computer vision, deep learning, and edge inference, the system automates the identification and classification of key actions, significantly reducing decision time from minutes to seconds while improving consistency and transparency. Importantly, the methodology is not limited to Taekwondo. The underlying framework -- based on pose estimation, motion classification, and impact analysis -- can be adapted to a wide range of sports requiring action detection, such as judo, karate, fencing, or even team sports like football and basketball, where foul recognition or performance tracking is critical. By addressing one of Taekwondo's most challenging scenarios -- head kick scoring -- we demonstrate the robustness, scalability, and sport-agnostic potential of 'FST.ai' to transform officiating standards across multiple disciplines.
>
---
#### [replaced 062] Rethinking Data Input for Point Cloud Upsampling
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.04476v3](http://arxiv.org/pdf/2407.04476v3)**

> **作者:** Tongxu Zhang
>
> **摘要:** Point cloud upsampling is crucial for tasks like 3D reconstruction. While existing methods rely on patch-based inputs, and there is no research discussing the differences and principles between point cloud model full input and patch based input. Ergo, we propose a novel approach using whole model inputs i.e. Average Segment input. Our experiments on PU1K and ABC datasets reveal that patch-based inputs consistently outperform whole model inputs. To understand this, we will delve into factors in feature extraction, and network architecture that influence upsampling results.
>
---
#### [replaced 063] MGSR: 2D/3D Mutual-boosted Gaussian Splatting for High-fidelity Surface Reconstruction under Various Light Conditions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05182v2](http://arxiv.org/pdf/2503.05182v2)**

> **作者:** Qingyuan Zhou; Yuehu Gong; Weidong Yang; Jiaze Li; Yeqi Luo; Baixin Xu; Shuhao Li; Ben Fei; Ying He
>
> **备注:** Accepted at ICCV'25
>
> **摘要:** Novel view synthesis (NVS) and surface reconstruction (SR) are essential tasks in 3D Gaussian Splatting (3D-GS). Despite recent progress, these tasks are often addressed independently, with GS-based rendering methods struggling under diverse light conditions and failing to produce accurate surfaces, while GS-based reconstruction methods frequently compromise rendering quality. This raises a central question: must rendering and reconstruction always involve a trade-off? To address this, we propose MGSR, a 2D/3D Mutual-boosted Gaussian splatting for Surface Reconstruction that enhances both rendering quality and 3D reconstruction accuracy. MGSR introduces two branches--one based on 2D-GS and the other on 3D-GS. The 2D-GS branch excels in surface reconstruction, providing precise geometry information to the 3D-GS branch. Leveraging this geometry, the 3D-GS branch employs a geometry-guided illumination decomposition module that captures reflected and transmitted components, enabling realistic rendering under varied light conditions. Using the transmitted component as supervision, the 2D-GS branch also achieves high-fidelity surface reconstruction. Throughout the optimization process, the 2D-GS and 3D-GS branches undergo alternating optimization, providing mutual supervision. Prior to this, each branch completes an independent warm-up phase, with an early stopping strategy implemented to reduce computational costs. We evaluate MGSR on a diverse set of synthetic and real-world datasets, at both object and scene levels, demonstrating strong performance in rendering and surface reconstruction. Code is available at https://github.com/TsingyuanChou/MGSR.
>
---
#### [replaced 064] DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2501.08005v5](http://arxiv.org/pdf/2501.08005v5)**

> **作者:** Francisco Caetano; Christiaan Viviers; Luis A. Zavala-Mondragón; Peter H. N. de With; Fons van der Sommen
>
> **备注:** ICCV 2025
>
> **摘要:** Out-of-distribution (OOD) detection holds significant importance across many applications. While semantic and domain-shift OOD problems are well-studied, this work focuses on covariate shifts - subtle variations in the data distribution that can degrade machine learning performance. We hypothesize that detecting these subtle shifts can improve our understanding of in-distribution boundaries, ultimately improving OOD detection. In adversarial discriminators trained with Batch Normalization (BN), real and adversarial samples form distinct domains with unique batch statistics - a property we exploit for OOD detection. We introduce DisCoPatch, an unsupervised Adversarial Variational Autoencoder (VAE) framework that harnesses this mechanism. During inference, batches consist of patches from the same image, ensuring a consistent data distribution that allows the model to rely on batch statistics. DisCoPatch uses the VAE's suboptimal outputs (generated and reconstructed) as negative samples to train the discriminator, thereby improving its ability to delineate the boundary between in-distribution samples and covariate shifts. By tightening this boundary, DisCoPatch achieves state-of-the-art results in public OOD detection benchmarks. The proposed model not only excels in detecting covariate shifts, achieving 95.5% AUROC on ImageNet-1K(-C) but also outperforms all prior methods on public Near-OOD (95.0%) benchmarks. With a compact model size of 25MB, it achieves high OOD detection performance at notably lower latency than existing methods, making it an efficient and practical solution for real-world OOD detection applications. The code is publicly available.
>
---
#### [replaced 065] Analysis of the 2024 BraTS Meningioma Radiotherapy Planning Automated Segmentation Challenge
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.18383v3](http://arxiv.org/pdf/2405.18383v3)**

> **作者:** Dominic LaBella; Valeriia Abramova; Mehdi Astaraki; Andre Ferreira; Zhifan Jiang; Mason C. Cleveland; Ramandeep Kang; Uma M. Lal-Trehan Estrada; Cansu Yalcin; Rachika E. Hamadache; Clara Lisazo; Adrià Casamitjana; Joaquim Salvi; Arnau Oliver; Xavier Lladó; Iuliana Toma-Dasu; Tiago Jesus; Behrus Puladi; Jens Kleesiek; Victor Alves; Jan Egger; Daniel Capellán-Martín; Abhijeet Parida; Austin Tapp; Xinyang Liu; Maria J. Ledesma-Carbayo; Jay B. Patel; Thomas N. McNeal; Maya Viera; Owen McCall; Albert E. Kim; Elizabeth R. Gerstner; Christopher P. Bridge; Katherine Schumacher; Michael Mix; Kevin Leu; Shan McBurney-Lin; Pierre Nedelec; Javier Villanueva-Meyer; David R. Raleigh; Jonathan Shapey; Tom Vercauteren; Kazumi Chia; Marina Ivory; Theodore Barfoot; Omar Al-Salihi; Justin Leu; Lia M. Halasz; Yuri S. Velichko; Chunhao Wang; John P. Kirkpatrick; Scott R. Floyd; Zachary J. Reitman; Trey C. Mullikin; Eugene J. Vaios; Christina Huang; Ulas Bagci; Sean Sachdev; Jona A. Hattangadi-Gluth; Tyler M. Seibert; Nikdokht Farid; Connor Puett; Matthew W. Pease; Kevin Shiue; Syed Muhammad Anwar; Shahriar Faghani; Peter Taylor; Pranav Warman; Jake Albrecht; András Jakab; Mana Moassefi; Verena Chung; Rong Chai; Alejandro Aristizabal; Alexandros Karargyris; Hasan Kassem; Sarthak Pati; Micah Sheller; Nazanin Maleki; Rachit Saluja; Florian Kofler; Christopher G. Schwarz; Philipp Lohmann; Phillipp Vollmuth; Louis Gagnon; Maruf Adewole; Hongwei Bran Li; Anahita Fathi Kazerooni; Nourel Hoda Tahon; Udunna Anazodo; Ahmed W. Moawad; Bjoern Menze; Marius George Linguraru; Mariam Aboian; Benedikt Wiestler; Ujjwal Baid; Gian-Marco Conte; Andreas M. Rauschecker; Ayman Nada; Aly H. Abayazeed; Raymond Huang; Maria Correia de Verdier; Jeffrey D. Rudie; Spyridon Bakas; Evan Calabrese
>
> **备注:** 23 pages, 9 figures, 5 tables
>
> **摘要:** The 2024 Brain Tumor Segmentation Meningioma Radiotherapy (BraTS-MEN-RT) challenge aimed to advance automated segmentation algorithms using the largest known multi-institutional dataset of 750 radiotherapy planning brain MRIs with expert-annotated target labels for patients with intact or postoperative meningioma that underwent either conventional external beam radiotherapy or stereotactic radiosurgery. Each case included a defaced 3D post-contrast T1-weighted radiotherapy planning MRI in its native acquisition space, accompanied by a single-label "target volume" representing the gross tumor volume (GTV) and any at-risk post-operative site. Target volume annotations adhered to established radiotherapy planning protocols, ensuring consistency across cases and institutions, and were approved by expert neuroradiologists and radiation oncologists. Six participating teams developed, containerized, and evaluated automated segmentation models using this comprehensive dataset. Team rankings were assessed using a modified lesion-wise Dice Similarity Coefficient (DSC) and 95% Hausdorff Distance (95HD). The best reported average lesion-wise DSC and 95HD was 0.815 and 26.92 mm, respectively. BraTS-MEN-RT is expected to significantly advance automated radiotherapy planning by enabling precise tumor segmentation and facilitating tailored treatment, ultimately improving patient outcomes. We describe the design and results from the BraTS-MEN-RT challenge.
>
---
#### [replaced 066] Towards Accurate and Efficient 3D Object Detection for Autonomous Driving: A Mixture of Experts Computing System on Edge
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.04123v2](http://arxiv.org/pdf/2507.04123v2)**

> **作者:** Linshen Liu; Boyan Su; Junyue Jiang; Guanlin Wu; Cong Guo; Ceyu Xu; Hao Frank Yang
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** This paper presents Edge-based Mixture of Experts (MoE) Collaborative Computing (EMC2), an optimal computing system designed for autonomous vehicles (AVs) that simultaneously achieves low-latency and high-accuracy 3D object detection. Unlike conventional approaches, EMC2 incorporates a scenario-aware MoE architecture specifically optimized for edge platforms. By effectively fusing LiDAR and camera data, the system leverages the complementary strengths of sparse 3D point clouds and dense 2D images to generate robust multimodal representations. To enable this, EMC2 employs an adaptive multimodal data bridge that performs multi-scale preprocessing on sensor inputs, followed by a scenario-aware routing mechanism that dynamically dispatches features to dedicated expert models based on object visibility and distance. In addition, EMC2 integrates joint hardware-software optimizations, including hardware resource utilization optimization and computational graph simplification, to ensure efficient and real-time inference on resource-constrained edge devices. Experiments on open-source benchmarks clearly show the EMC2 advancements as an end-to-end system. On the KITTI dataset, it achieves an average accuracy improvement of 3.58% and a 159.06% inference speedup compared to 15 baseline methods on Jetson platforms, with similar performance gains on the nuScenes dataset, highlighting its capability to advance reliable, real-time 3D object detection tasks for AVs. The official implementation is available at https://github.com/LinshenLiu622/EMC2.
>
---
#### [replaced 067] Sparrow: Data-Efficient Video-LLM with Text-to-Image Augmentation
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19951v5](http://arxiv.org/pdf/2411.19951v5)**

> **作者:** Shukang Yin; Chaoyou Fu; Sirui Zhao; Chunjiang Ge; Yan Yang; Yuhan Dai; Yongdong Luo; Tong Xu; Caifeng Shan; Enhong Chen
>
> **备注:** Project page: https://github.com/VITA-MLLM/Sparrow
>
> **摘要:** Recent years have seen the success of Multimodal Large Language Models (MLLMs) in the domain of vision understanding. The success of these models can largely be attributed to the dominant scaling law, which states that larger parameter sizes and data volumes contribute to better performance. Notably, data scaling has been primarily driven by automatic data pipelines, which focus on the self-instruction of LLMs. The paradigm has been taken for granted for quite some time, but the study of the effectiveness of scaling with these data has been neglected for a long time. In this context, this work revisits scaling with synthetic data and focuses on developing video-LLMs from a data-centric perspective. Our primary study approach involves fine-tuning pre-trained image-LLMs with video data and examining learning efficiency through data scaling. Results from our preliminary experiments reveal a low learning efficiency phenomenon when simply scaling up video data samples, which, through our probing, can be ascribed to a lack of instruction diversity. Aiming at this issue, we propose a data augmentation method called Sparrow, which synthesizes video-like samples from pure text instruction data. Mixing these synthetic samples with the video data enables a more efficient training scheme. Through comprehensive experiments, we demonstrate that our proposed method achieves performance comparable to or even superior to that of baselines trained with significantly more samples. Meanwhile, we find that incorporating these synthetic samples can enhance the performance of long video understanding without requiring training on long video data. The code and data examples are available at https://github.com/VITA-MLLM/Sparrow.
>
---
#### [replaced 068] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.17262v2](http://arxiv.org/pdf/2503.17262v2)**

> **作者:** Shuang Guo; Friedhelm Hamann; Guillermo Gallego
>
> **备注:** 13 pages, 8 figures, 9 tables. Project page: https://github.com/tub-rip/E2FAI . IEEE/CVF International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Event cameras rely on motion to obtain information about scene appearance. This means that appearance and motion are inherently linked: either both are present and recorded in the event data, or neither is captured. Previous works treat the recovery of these two visual quantities as separate tasks, which does not fit with the above-mentioned nature of event cameras and overlooks the inherent relations between them. We propose an unsupervised learning framework that jointly estimates optical flow (motion) and image intensity (appearance) using a single network. From the data generation model, we newly derive the event-based photometric error as a function of optical flow and image intensity. This error is further combined with the contrast maximization framework to form a comprehensive loss function that provides proper constraints for both flow and intensity estimation. Exhaustive experiments show our method's state-of-the-art performance: in optical flow estimation, it reduces EPE by 20% and AE by 25% compared to unsupervised approaches, while delivering competitive intensity estimation results, particularly in high dynamic range scenarios. Our method also achieves shorter inference time than all other optical flow methods and many of the image reconstruction methods, while they output only one quantity. Project page: https://github.com/tub-rip/E2FAI
>
---
#### [replaced 069] ETCH: Generalizing Body Fitting to Clothed Humans via Equivariant Tightness
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2503.10624v2](http://arxiv.org/pdf/2503.10624v2)**

> **作者:** Boqian Li; Haiwen Feng; Zeyu Cai; Michael J. Black; Yuliang Xiu
>
> **备注:** Page: https://boqian-li.github.io/ETCH/, Code: https://github.com/boqian-li/ETCH
>
> **摘要:** Fitting a body to a 3D clothed human point cloud is a common yet challenging task. Traditional optimization-based approaches use multi-stage pipelines that are sensitive to pose initialization, while recent learning-based methods often struggle with generalization across diverse poses and garment types. We propose Equivariant Tightness Fitting for Clothed Humans, or ETCH, a novel pipeline that estimates cloth-to-body surface mapping through locally approximate SE(3) equivariance, encoding tightness as displacement vectors from the cloth surface to the underlying body. Following this mapping, pose-invariant body features regress sparse body markers, simplifying clothed human fitting into an inner-body marker fitting task. Extensive experiments on CAPE and 4D-Dress show that ETCH significantly outperforms state-of-the-art methods -- both tightness-agnostic and tightness-aware -- in body fitting accuracy on loose clothing (16.7% ~ 69.5%) and shape accuracy (average 49.9%). Our equivariant tightness design can even reduce directional errors by (67.2% ~ 89.8%) in one-shot (or out-of-distribution) settings (~ 1% data). Qualitative results demonstrate strong generalization of ETCH, regardless of challenging poses, unseen shapes, loose clothing, and non-rigid dynamics. We will release the code and models soon for research purposes at https://boqian-li.github.io/ETCH/.
>
---
#### [replaced 070] CogStream: Context-guided Streaming Video Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.10516v2](http://arxiv.org/pdf/2506.10516v2)**

> **作者:** Zicheng Zhao; Kangyu Wang; Shijie Li; Rui Qian; Weiyao Lin; Huabin Liu
>
> **备注:** Project page: https://github.com/LiamZhao326/CogStream
>
> **摘要:** Despite advancements in Video Large Language Models (Vid-LLMs) improving multimodal understanding, challenges persist in streaming video reasoning due to its reliance on contextual information. Existing paradigms feed all available historical contextual information into Vid-LLMs, resulting in a significant computational burden for visual data processing. Furthermore, the inclusion of irrelevant context distracts models from key details. This paper introduces a challenging task called Context-guided Streaming Video Reasoning (CogStream), which simulates real-world streaming video scenarios, requiring models to identify the most relevant historical contextual information to deduce answers for questions about the current stream. To support CogStream, we present a densely annotated dataset featuring extensive and hierarchical question-answer pairs, generated by a semi-automatic pipeline. Additionally, we present CogReasoner as a baseline model. It efficiently tackles this task by leveraging visual stream compression and historical dialogue retrieval. Extensive experiments prove the effectiveness of this method. The project is released on https://github.com/LiamZhao326/CogStream.
>
---
#### [replaced 071] Towards a Universal 3D Medical Multi-modality Generalization via Learning Personalized Invariant Representation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.06106v3](http://arxiv.org/pdf/2411.06106v3)**

> **作者:** Zhaorui Tan; Xi Yang; Tan Pan; Tianyi Liu; Chen Jiang; Xin Guo; Qiufeng Wang; Anh Nguyen; Yuan Qi; Kaizhu Huang; Yuan Cheng
>
> **备注:** Accepted by ICCV25
>
> **摘要:** The differences among medical imaging modalities, driven by distinct underlying principles, pose significant challenges for generalization in multi-modal medical tasks. Beyond modality gaps, individual variations, such as differences in organ size and metabolic rate, further impede a model's ability to generalize effectively across both modalities and diverse populations. Despite the importance of personalization, existing approaches to multi-modal generalization often neglect individual differences, focusing solely on common anatomical features. This limitation may result in weakened generalization in various medical tasks. In this paper, we unveil that personalization is critical for multi-modal generalization. Specifically, we propose an approach to achieve personalized generalization through approximating the underlying personalized invariant representation ${X}_h$ across various modalities by leveraging individual-level constraints and a learnable biological prior. We validate the feasibility and benefits of learning a personalized ${X}_h$, showing that this representation is highly generalizable and transferable across various multi-modal medical tasks. Extensive experimental results consistently show that the additionally incorporated personalization significantly improves performance and generalization across diverse scenarios, confirming its effectiveness.
>
---
#### [replaced 072] Aligning AI with Public Values: Deliberation and Decision-Making for Governing Multimodal LLMs in Political Video Analysis
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2410.01817v2](http://arxiv.org/pdf/2410.01817v2)**

> **作者:** Tanusree Sharma; Yujin Potter; Zachary Kilhoffer; Yun Huang; Dawn Song; Yang Wang
>
> **摘要:** How AI models should deal with political topics has been discussed, but it remains challenging and requires better governance. This paper examines the governance of large language models through individual and collective deliberation, focusing on politically sensitive videos. We conducted a two-step study: interviews with 10 journalists established a baseline understanding of expert video interpretation; 114 individuals through deliberation using InclusiveAI, a platform that facilitates democratic decision-making through decentralized autonomous organization (DAO) mechanisms. Our findings reveal distinct differences in interpretative priorities: while experts emphasized emotion and narrative, the general public prioritized factual clarity, objectivity, and emotional neutrality. Furthermore, we examined how different governance mechanisms - quadratic vs. weighted voting and equal vs. 20/80 voting power - shape users' decision-making regarding AI behavior. Results indicate that voting methods significantly influence outcomes, with quadratic voting reinforcing perceptions of liberal democracy and political equality. Our study underscores the necessity of selecting appropriate governance mechanisms to better capture user perspectives and suggests decentralized AI governance as a potential way to facilitate broader public engagement in AI development, ensuring that varied perspectives meaningfully inform design decisions.
>
---
#### [replaced 073] GroundFlow: A Plug-in Module for Temporal Reasoning on 3D Point Cloud Sequential Grounding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21188v2](http://arxiv.org/pdf/2506.21188v2)**

> **作者:** Zijun Lin; Shuting He; Cheston Tan; Bihan Wen
>
> **摘要:** Sequential grounding in 3D point clouds (SG3D) refers to locating sequences of objects by following text instructions for a daily activity with detailed steps. Current 3D visual grounding (3DVG) methods treat text instructions with multiple steps as a whole, without extracting useful temporal information from each step. However, the instructions in SG3D often contain pronouns such as "it", "here" and "the same" to make language expressions concise. This requires grounding methods to understand the context and retrieve relevant information from previous steps to correctly locate object sequences. Due to the lack of an effective module for collecting related historical information, state-of-the-art 3DVG methods face significant challenges in adapting to the SG3D task. To fill this gap, we propose GroundFlow -- a plug-in module for temporal reasoning on 3D point cloud sequential grounding. Firstly, we demonstrate that integrating GroundFlow improves the task accuracy of 3DVG baseline methods by a large margin (+7.5\% and +10.2\%) in the SG3D benchmark, even outperforming a 3D large language model pre-trained on various datasets. Furthermore, we selectively extract both short-term and long-term step information based on its relevance to the current instruction, enabling GroundFlow to take a comprehensive view of historical information and maintain its temporal understanding advantage as step counts increase. Overall, our work introduces temporal reasoning capabilities to existing 3DVG models and achieves state-of-the-art performance in the SG3D benchmark across five datasets.
>
---
#### [replaced 074] Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02304v2](http://arxiv.org/pdf/2505.02304v2)**

> **作者:** Siyu Liang; Yunan Li; Wentian Xin; Huizhou Chen; Xujie Liu; Kang Liu; Qiguang Miao
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Sign language recognition (SLR) faces fundamental challenges in creating accurate annotations due to the inherent complexity of simultaneous manual and non-manual signals. To the best of our knowledge, this is the first work to integrate generative large language models (LLMs) into SLR tasks. We propose a novel Generative Sign-description Prompts Multi-positive Contrastive learning (GSP-MC) method that leverages retrieval-augmented generation (RAG) with domain-specific LLMs, incorporating multi-step prompt engineering and expert-validated sign language corpora to produce precise multipart descriptions. The GSP-MC method also employs a dual-encoder architecture to bidirectionally align hierarchical skeleton features with multiple text descriptions (global, synonym, and part level) through probabilistic matching. Our approach combines global and part-level losses, optimizing KL divergence to ensure robust alignment across all relevant text-skeleton pairs while capturing both sign-level semantics and detailed part dynamics. Experiments demonstrate state-of-the-art performance against existing methods on the Chinese SLR500 (reaching 97.1%) and Turkish AUTSL datasets (97.07% accuracy). The method's cross-lingual effectiveness highlight its potential for developing inclusive communication technologies.
>
---
#### [replaced 075] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.12332v2](http://arxiv.org/pdf/2410.12332v2)**

> **作者:** Yunqiu Xu; Linchao Zhu; Yi Yang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** While multimodal large language models (MLLMs) have demonstrated extraordinary vision-language understanding capabilities, their abilities to solve instance-level visual-language problems beyond a single image warrant further exploration. To assess these unproven abilities of MLLMs, this paper proposes a new visual grounding task called multi-context visual grounding, which aims to localize instances of interest across multiple images based on open-ended text prompts. In order to facilitate this research, we construct a new dataset MC-Bench that features 2K high-quality and manually annotated samples. Each sample consists of an instance-level labeled image pair and a corresponding text prompt that indicates the target instances in the images. These text prompts are highly open-ended and follow three distinct styles, covering 20 practical skills. We benchmark over 20 state-of-the-art MLLMs and foundation models with potential multi-context visual grounding capabilities, along with our developed simple yet effective agentic baseline and a finetuned baseline by multi-context instruction tuning. Our evaluation reveals a non-trivial performance gap between existing MLLMs and humans, along with some insightful observations that suggest potential future directions. We hope that MC-Bench and our empirical findings encourage the research community to further advance the untapped potentials of MLLMs in instance-level tasks, particularly in multi-image contexts. Project page: https://xuyunqiu.github.io/MC-Bench.
>
---
#### [replaced 076] V-RoAst: Visual Road Assessment. Can VLM be a Road Safety Assessor Using the iRAP Standard?
- **分类: cs.CV; cs.AI; cs.ET**

- **链接: [http://arxiv.org/pdf/2408.10872v4](http://arxiv.org/pdf/2408.10872v4)**

> **作者:** Natchapon Jongwiriyanurak; Zichao Zeng; June Moh Goo; Xinglei Wang; Ilya Ilyankou; Kerkritt Sriroongvikrai; Nicola Christie; Meihui Wang; Huanfa Chen; James Haworth
>
> **摘要:** Road safety assessments are critical yet costly, especially in Low- and Middle-Income Countries (LMICs), where most roads remain unrated. Traditional methods require expert annotation and training data, while supervised learning-based approaches struggle to generalise across regions. In this paper, we introduce \textit{V-RoAst}, a zero-shot Visual Question Answering (VQA) framework using Vision-Language Models (VLMs) to classify road safety attributes defined by the iRAP standard. We introduce the first open-source dataset from ThaiRAP, consisting of over 2,000 curated street-level images from Thailand annotated for this task. We evaluate Gemini-1.5-flash and GPT-4o-mini on this dataset and benchmark their performance against VGGNet and ResNet baselines. While VLMs underperform on spatial awareness, they generalise well to unseen classes and offer flexible prompt-based reasoning without retraining. Our results show that VLMs can serve as automatic road assessment tools when integrated with complementary data. This work is the first to explore VLMs for zero-shot infrastructure risk assessment and opens new directions for automatic, low-cost road safety mapping. Code and dataset: https://github.com/PongNJ/V-RoAst.
>
---
#### [replaced 077] GUI-G$^2$: Gaussian Reward Modeling for GUI Grounding
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.15846v2](http://arxiv.org/pdf/2507.15846v2)**

> **作者:** Fei Tang; Zhangxuan Gu; Zhengxi Lu; Xuyang Liu; Shuheng Shen; Changhua Meng; Wen Wang; Wenqi Zhang; Yongliang Shen; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **摘要:** Graphical User Interface (GUI) grounding maps natural language instructions to precise interface locations for autonomous interaction. Current reinforcement learning approaches use binary rewards that treat elements as hit-or-miss targets, creating sparse signals that ignore the continuous nature of spatial interactions. Motivated by human clicking behavior that naturally forms Gaussian distributions centered on target elements, we introduce GUI Gaussian Grounding Rewards (GUI-G$^2$), a principled reward framework that models GUI elements as continuous Gaussian distributions across the interface plane. GUI-G$^2$ incorporates two synergistic mechanisms: Gaussian point rewards model precise localization through exponentially decaying distributions centered on element centroids, while coverage rewards assess spatial alignment by measuring the overlap between predicted Gaussian distributions and target regions. To handle diverse element scales, we develop an adaptive variance mechanism that calibrates reward distributions based on element dimensions. This framework transforms GUI grounding from sparse binary classification to dense continuous optimization, where Gaussian distributions generate rich gradient signals that guide models toward optimal interaction positions. Extensive experiments across ScreenSpot, ScreenSpot-v2, and ScreenSpot-Pro benchmarks demonstrate that GUI-G$^2$, substantially outperforms state-of-the-art method UI-TARS-72B, with the most significant improvement of 24.7% on ScreenSpot-Pro. Our analysis reveals that continuous modeling provides superior robustness to interface variations and enhanced generalization to unseen layouts, establishing a new paradigm for spatial reasoning in GUI interaction tasks.
>
---
#### [replaced 078] FLLIC: Functionally Lossless Image Compression
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2401.13616v4](http://arxiv.org/pdf/2401.13616v4)**

> **作者:** Xi Zhang; Xiaolin Wu
>
> **摘要:** Recently, DNN models for lossless image coding have surpassed their traditional counterparts in compression performance, reducing the previous lossless bit rate by about ten percent for natural color images. But even with these advances, mathematically lossless image compression (MLLIC) ratios for natural images still fall short of the bandwidth and cost-effectiveness requirements of most practical imaging and vision systems at present and beyond. To overcome the performance barrier of MLLIC, we question the very necessity of MLLIC. Considering that all digital imaging sensors suffer from acquisition noises, why should we insist on mathematically lossless coding, i.e., wasting bits to preserve noises? Instead, we propose a new paradigm of joint denoising and compression called functionally lossless image compression (FLLIC), which performs lossless compression of optimally denoised images (the optimality may be task-specific). Although not literally lossless with respect to the noisy input, FLLIC aims to achieve the best possible reconstruction of the latent noise-free original image. Extensive experiments show that FLLIC achieves state-of-the-art performance in joint denoising and compression of noisy images and does so at a lower computational cost.
>
---
#### [replaced 079] DOFA-CLIP: Multimodal Vision-Language Foundation Models for Earth Observation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06312v2](http://arxiv.org/pdf/2503.06312v2)**

> **作者:** Zhitong Xiong; Yi Wang; Weikang Yu; Adam J Stewart; Jie Zhao; Nils Lehmann; Thomas Dujardin; Zhenghang Yuan; Pedram Ghamisi; Xiao Xiang Zhu
>
> **备注:** code & weights: https://github.com/xiong-zhitong/DOFA-CLIP
>
> **摘要:** Earth observation (EO) spans a broad spectrum of modalities, including optical, radar, multispectral, and hyperspectral data, each capturing distinct environmental signals. However, current vision-language models in EO, particularly CLIP-based variants, remain confined to individual modalities, limiting generalization and scalability across diverse tasks. We present DOFA-CLIP (Dynamic-One-For-All CLIP), a unified vision-language foundation model that dynamically adapts to EO modalities with flexible spectral configurations through a single Transformer backbone. Our approach introduces three key contributions: 1) the construction of GeoLangBind-2M, a large-scale EO image-text dataset covering six heterogeneous modalities with rich natural language descriptions; 2) a novel training strategy called VECT (Vision-models Enhanced Contrastive Text-image pretraining), which enhances the spatial awareness of CLIP features with multiple vision foundation models; and 3) a Modality-aware Knowledge Agglomeration (MaKA) module that refines feature distillation with modality-specific awareness. DOFA-CLIP achieves state-of-the-art zero-shot performance across a wide range of EO benchmarks, including unseen modalities and a diverse number of input spectral bands. Together, these contributions establish a scalable foundation for multimodal EO understanding and open new avenues for integrating heterogeneous EO data with large language models. Code and datasets will be released. Code and datasets are publicly available.
>
---
#### [replaced 080] Learning Streaming Video Representation via Multitask Training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20041v2](http://arxiv.org/pdf/2504.20041v2)**

> **作者:** Yibin Yan; Jilan Xu; Shangzhe Di; Yikun Liu; Yudi Shi; Qirui Chen; Zeqian Li; Yifei Huang; Weidi Xie
>
> **备注:** Technical Report. Project Page: https://go2heart.github.io/streamformer
>
> **摘要:** Understanding continuous video streams plays a fundamental role in real-time applications including embodied AI and autonomous driving. Unlike offline video understanding, streaming video understanding requires the ability to process video streams frame by frame, preserve historical information, and make low-latency decisions. To address these challenges, our main contributions are three-fold. (i) We develop a novel streaming video backbone, termed as StreamFormer, by incorporating causal temporal attention into a pre-trained vision transformer. This enables efficient streaming video processing while maintaining image representation capability. (ii) To train StreamFormer, we propose to unify diverse spatial-temporal video understanding tasks within a multitask visual-language alignment framework. Hence, StreamFormer learns global semantics, temporal dynamics, and fine-grained spatial relationships simultaneously. (iii) We conduct extensive experiments on online action detection, online video instance segmentation, and video question answering. StreamFormer achieves competitive results while maintaining efficiency, demonstrating its potential for real-time applications.
>
---
#### [replaced 081] DeSamba: Decoupled Spectral Adaptive Framework for 3D Multi-Sequence MRI Lesion Classification
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15487v2](http://arxiv.org/pdf/2507.15487v2)**

> **作者:** Dezhen Wang; Sheng Miao; Rongxin Chai; Jiufa Cui
>
> **备注:** 7 figures, 3 tables, submitted to AAAI2026
>
> **摘要:** Magnetic Resonance Imaging (MRI) sequences provide rich spatial and frequency domain information, which is crucial for accurate lesion classification in medical imaging. However, effectively integrating multi-sequence MRI data for robust 3D lesion classification remains a challenge. In this paper, we propose DeSamba (Decoupled Spectral Adaptive Network and Mamba-Based Model), a novel framework designed to extract decoupled representations and adaptively fuse spatial and spectral features for lesion classification. DeSamba introduces a Decoupled Representation Learning Module (DRLM) that decouples features from different MRI sequences through self-reconstruction and cross-reconstruction, and a Spectral Adaptive Modulation Block (SAMB) within the proposed SAMNet, enabling dynamic fusion of spectral and spatial information based on lesion characteristics. We evaluate DeSamba on two clinically relevant 3D datasets. On a six-class spinal metastasis dataset (n=1,448), DeSamba achieves 62.10% Top-1 accuracy, 63.62% F1-score, 87.71% AUC, and 93.55% Top-3 accuracy on an external validation set (n=372), outperforming all state-of-the-art (SOTA) baselines. On a spondylitis dataset (n=251) involving a challenging binary classification task, DeSamba achieves 70.00%/64.52% accuracy and 74.75/73.88 AUC on internal and external validation sets, respectively. Ablation studies demonstrate that both DRLM and SAMB significantly contribute to overall performance, with over 10% relative improvement compared to the baseline. Our results highlight the potential of DeSamba as a generalizable and effective solution for 3D lesion classification in multi-sequence medical imaging.
>
---
#### [replaced 082] FedWSQ: Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23516v3](http://arxiv.org/pdf/2506.23516v3)**

> **作者:** Seung-Wook Kim; Seongyeol Kim; Jiah Kim; Seowon Ji; Se-Ho Lee
>
> **摘要:** Federated learning (FL) often suffers from performance degradation due to key challenges such as data heterogeneity and communication constraints. To address these limitations, we present a novel FL framework called FedWSQ, which integrates weight standardization (WS) and the proposed distribution-aware non-uniform quantization (DANUQ). WS enhances FL performance by filtering out biased components in local updates during training, thereby improving the robustness of the model against data heterogeneity and unstable client participation. In addition, DANUQ minimizes quantization errors by leveraging the statistical properties of local model updates. As a result, FedWSQ significantly reduces communication overhead while maintaining superior model accuracy. Extensive experiments on FL benchmark datasets demonstrate that FedWSQ consistently outperforms existing FL methods across various challenging FL settings, including extreme data heterogeneity and ultra-low-bit communication scenarios.
>
---
