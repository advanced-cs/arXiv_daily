# 计算机视觉 cs.CV

- **最新发布 224 篇**

- **更新 166 篇**

## 最新发布

#### [new 001] ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding
- **分类: cs.CV**

- **简介: 该论文提出ShapeLLM-Omni，旨在解决现有模型缺乏3D内容理解和生成能力的问题。任务是开发原生多模态大模型，支持3D资产与文本的交互。工作包括构建3D-Alpaca数据集，并基于Qwen-2.5-vl-7B-Instruct进行指令训练，实现3D生成、理解和编辑。**

- **链接: [http://arxiv.org/pdf/2506.01853v1](http://arxiv.org/pdf/2506.01853v1)**

> **作者:** Junliang Ye; Zhengyi Wang; Ruowen Zhao; Shenghao Xie; Jun Zhu
>
> **备注:** Project page: https://github.com/JAMESYJL/ShapeLLM-Omni
>
> **摘要:** Recently, the powerful text-to-image capabilities of ChatGPT-4o have led to growing appreciation for native multimodal large language models. However, its multimodal capabilities remain confined to images and text. Yet beyond images, the ability to understand and generate 3D content is equally crucial. To address this gap, we propose ShapeLLM-Omni-a native 3D large language model capable of understanding and generating 3D assets and text in any sequence. First, we train a 3D vector-quantized variational autoencoder (VQVAE), which maps 3D objects into a discrete latent space to achieve efficient and accurate shape representation and reconstruction. Building upon the 3D-aware discrete tokens, we innovatively construct a large-scale continuous training dataset named 3D-Alpaca, encompassing generation, comprehension, and editing, thus providing rich resources for future research and training. Finally, by performing instruction-based training of the Qwen-2.5-vl-7B-Instruct model on the 3D-Alpaca dataset. Our work provides an effective attempt at extending multimodal models with basic 3D capabilities, which contributes to future research in 3D-native AI. Project page: https://github.com/JAMESYJL/ShapeLLM-Omni
>
---
#### [new 002] Active Learning via Vision-Language Model Adaptation with Open Data
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型（VLM）在主动学习（AL）任务中的应用。旨在解决标注数据昂贵且依赖专家的问题，通过利用公开开放数据增强任务相关样本，提出ALOR方法，并结合对比调优（CT）和新策略尾部优先采样（TFS），显著提升性能。**

- **链接: [http://arxiv.org/pdf/2506.01724v1](http://arxiv.org/pdf/2506.01724v1)**

> **作者:** Tong Wang; Jiaqi Wang; Shu Kong
>
> **备注:** Here is the project webpage: https://leowangtong.github.io/ALOR/
>
> **摘要:** Pretrained on web-scale open data, VLMs offer powerful capabilities for solving downstream tasks after being adapted to task-specific labeled data. Yet, data labeling can be expensive and may demand domain expertise. Active Learning (AL) aims to reduce this expense by strategically selecting the most informative data for labeling and model training. Recent AL methods have explored VLMs but have not leveraged publicly available open data, such as VLM's pretraining data. In this work, we leverage such data by retrieving task-relevant examples to augment the task-specific examples. As expected, incorporating them significantly improves AL. Given that our method exploits open-source VLM and open data, we refer to it as Active Learning with Open Resources (ALOR). Additionally, most VLM-based AL methods use prompt tuning (PT) for model adaptation, likely due to its ability to directly utilize pretrained parameters and the assumption that doing so reduces the risk of overfitting to limited labeled data. We rigorously compare popular adaptation approaches, including linear probing (LP), finetuning (FT), and contrastive tuning (CT). We reveal two key findings: (1) All adaptation approaches benefit from incorporating retrieved data, and (2) CT resoundingly outperforms other approaches across AL methods. Further analysis of retrieved data reveals a naturally imbalanced distribution of task-relevant classes, exposing inherent biases within the VLM. This motivates our novel Tail First Sampling (TFS) strategy for AL, an embarrassingly simple yet effective method that prioritizes sampling data from underrepresented classes to label. Extensive experiments demonstrate that our final method, contrastively finetuning VLM on both retrieved and TFS-selected labeled data, significantly outperforms existing methods.
>
---
#### [new 003] Long-Tailed Visual Recognition via Permutation-Invariant Head-to-Tail Feature Fusion
- **分类: cs.CV**

- **简介: 该论文属于视觉识别任务，旨在解决长尾数据分布下模型对尾部类别识别准确率低的问题。通过提出一种可置换不变的头到尾特征融合方法（PI-H2T），优化表示空间和分类器偏差，提升尾类识别效果，并具有良好的适配性与实用性。**

- **链接: [http://arxiv.org/pdf/2506.00625v1](http://arxiv.org/pdf/2506.00625v1)**

> **作者:** Mengke Li; Zhikai Hu; Yang Lu; Weichao Lan; Yiu-ming Cheung; Hui Huang
>
> **摘要:** The imbalanced distribution of long-tailed data presents a significant challenge for deep learning models, causing them to prioritize head classes while neglecting tail classes. Two key factors contributing to low recognition accuracy are the deformed representation space and a biased classifier, stemming from insufficient semantic information in tail classes. To address these issues, we propose permutation-invariant and head-to-tail feature fusion (PI-H2T), a highly adaptable method. PI-H2T enhances the representation space through permutation-invariant representation fusion (PIF), yielding more clustered features and automatic class margins. Additionally, it adjusts the biased classifier by transferring semantic information from head to tail classes via head-to-tail fusion (H2TF), improving tail class diversity. Theoretical analysis and experiments show that PI-H2T optimizes both the representation space and decision boundaries. Its plug-and-play design ensures seamless integration into existing methods, providing a straightforward path to further performance improvements. Extensive experiments on long-tailed benchmarks confirm the effectiveness of PI-H2T.
>
---
#### [new 004] Self-Supervised Multi-View Representation Learning using Vision-Language Model for 3D/4D Facial Expression Recognition
- **分类: cs.CV**

- **简介: 该论文属于情感计算中的面部表情识别任务，旨在解决3D/4D面部表情识别中对标注数据依赖高和多视角变化的问题。作者提出了SMILE-VLM框架，通过自监督学习结合视觉-语言模型，实现多视角解相关、视觉-语言对比对齐和跨模冗余最小化，提升了识别性能，并扩展至4D微表情识别。**

- **链接: [http://arxiv.org/pdf/2506.01203v1](http://arxiv.org/pdf/2506.01203v1)**

> **作者:** Muzammil Behzad
>
> **摘要:** Facial expression recognition (FER) is a fundamental task in affective computing with applications in human-computer interaction, mental health analysis, and behavioral understanding. In this paper, we propose SMILE-VLM, a self-supervised vision-language model for 3D/4D FER that unifies multiview visual representation learning with natural language supervision. SMILE-VLM learns robust, semantically aligned, and view-invariant embeddings by proposing three core components: multiview decorrelation via a Barlow Twins-style loss, vision-language contrastive alignment, and cross-modal redundancy minimization. Our framework achieves the state-of-the-art performance on multiple benchmarks. We further extend SMILE-VLM to the task of 4D micro-expression recognition (MER) to recognize the subtle affective cues. The extensive results demonstrate that SMILE-VLM not only surpasses existing unsupervised methods but also matches or exceeds supervised baselines, offering a scalable and annotation-efficient solution for expressive facial behavior understanding.
>
---
#### [new 005] Target Driven Adaptive Loss For Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决现有损失函数在局部区域检测和小尺度、低对比度目标上的不足。作者提出了目标驱动自适应（TDA）损失，通过基于图像块的机制和自适应调整策略，提升模型对关键局部区域的关注能力。实验表明，该方法在多个数据集上优于现有损失函数。**

- **链接: [http://arxiv.org/pdf/2506.01349v1](http://arxiv.org/pdf/2506.01349v1)**

> **作者:** Yuho Shoji; Takahiro Toizumi; Atsushi Ito
>
> **摘要:** We propose a target driven adaptive (TDA) loss to enhance the performance of infrared small target detection (IRSTD). Prior works have used loss functions, such as binary cross-entropy loss and IoU loss, to train segmentation models for IRSTD. Minimizing these loss functions guides models to extract pixel-level features or global image context. However, they have two issues: improving detection performance for local regions around the targets and enhancing robustness to small scale and low local contrast. To address these issues, the proposed TDA loss introduces a patch-based mechanism, and an adaptive adjustment strategy to scale and local contrast. The proposed TDA loss leads the model to focus on local regions around the targets and pay particular attention to targets with smaller scales and lower local contrast. We evaluate the proposed method on three datasets for IRSTD. The results demonstrate that the proposed TDA loss achieves better detection performance than existing losses on these datasets.
>
---
#### [new 006] Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型（VLM）的指令微调任务，旨在解决训练成本高、依赖大量标注数据和计算资源的问题。作者提出PROGRESS方法，通过动态选择最具信息量的样本进行学习，减少数据和计算需求，提升训练效率，并验证了其在多个数据集上的有效性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2506.01085v1](http://arxiv.org/pdf/2506.01085v1)**

> **作者:** Shivam Chandhok; Qian Yang; Oscar Manas; Kanishk Jain; Leonid Sigal; Aishwarya Agrawal
>
> **备注:** Preprint
>
> **摘要:** Instruction tuning has been central to the success of recent vision-language models (VLMs), but it remains expensive-requiring large-scale datasets, high-quality annotations, and large compute budgets. We propose PRioritized cOncept learninG via Relative Error-driven Sample Selection (PROGRESS), a data- and compute-efficient framework that enables VLMs to dynamically select what to learn next based on their evolving needs during training. At each stage, the model tracks its learning progress across skills and selects the most informative samples-those it has not already mastered and that are not too difficult to learn at the current stage of training. This strategy effectively controls skill acquisition and the order in which skills are learned. Specifically, we sample from skills showing the highest learning progress, prioritizing those with the most rapid improvement. Unlike prior methods, PROGRESS requires no upfront answer annotations, queries answers only on a need basis, avoids reliance on additional supervision from auxiliary VLMs, and does not require compute-heavy gradient computations for data selection. Experiments across multiple instruction-tuning datasets of varying scales demonstrate that PROGRESS consistently outperforms state-of-the-art baselines with much less data and supervision. Additionally, we show strong cross-architecture generalization and transferability to larger models, validating PROGRESS as a scalable solution for efficient learning.
>
---
#### [new 007] 3D Skeleton-Based Action Recognition: A Review
- **分类: cs.CV**

- **简介: 该论文属于3D骨架动作识别任务，旨在解决现有研究忽视预处理等关键步骤的问题。作者提出了一个任务导向的综合框架，涵盖数据预处理、特征提取、时空建模等子任务，并综述了最新方法与公开数据集，为该领域提供了系统性发展路径。**

- **链接: [http://arxiv.org/pdf/2506.00915v1](http://arxiv.org/pdf/2506.00915v1)**

> **作者:** Mengyuan Liu; Hong Liu; Qianshuo Hu; Bin Ren; Junsong Yuan; Jiaying Lin; Jiajun Wen
>
> **摘要:** With the inherent advantages of skeleton representation, 3D skeleton-based action recognition has become a prominent topic in the field of computer vision. However, previous reviews have predominantly adopted a model-oriented perspective, often neglecting the fundamental steps involved in skeleton-based action recognition. This oversight tends to ignore key components of skeleton-based action recognition beyond model design and has hindered deeper, more intrinsic understanding of the task. To bridge this gap, our review aims to address these limitations by presenting a comprehensive, task-oriented framework for understanding skeleton-based action recognition. We begin by decomposing the task into a series of sub-tasks, placing particular emphasis on preprocessing steps such as modality derivation and data augmentation. The subsequent discussion delves into critical sub-tasks, including feature extraction and spatio-temporal modeling techniques. Beyond foundational action recognition networks, recently advanced frameworks such as hybrid architectures, Mamba models, large language models (LLMs), and generative models have also been highlighted. Finally, a comprehensive overview of public 3D skeleton datasets is presented, accompanied by an analysis of state-of-the-art algorithms evaluated on these benchmarks. By integrating task-oriented discussions, comprehensive examinations of sub-tasks, and an emphasis on the latest advancements, our review provides a fundamental and accessible structured roadmap for understanding and advancing the field of 3D skeleton-based action recognition.
>
---
#### [new 008] ReFoCUS: Reinforcement-guided Frame Optimization for Contextual Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频内容理解任务，旨在解决现有方法在帧选择上的不足。作者提出了ReFoCUS框架，通过强化学习优化帧选择策略，以提升视频问答性能，无需显式监督，并确保时间一致性。**

- **链接: [http://arxiv.org/pdf/2506.01274v1](http://arxiv.org/pdf/2506.01274v1)**

> **作者:** Hosu Lee; Junho Kim; Hyunjun Kim; Yong Man Ro
>
> **摘要:** Recent progress in Large Multi-modal Models (LMMs) has enabled effective vision-language reasoning, yet the ability to understand video content remains constrained by suboptimal frame selection strategies. Existing approaches often rely on static heuristics or external retrieval modules to feed frame information into video-LLMs, which may fail to provide the query-relevant information. In this work, we introduce ReFoCUS (Reinforcement-guided Frame Optimization for Contextual UnderStanding), a novel frame-level policy optimization framework that shifts the optimization target from textual responses to visual input selection. ReFoCUS learns a frame selection policy via reinforcement learning, using reward signals derived from a reference LMM to reflect the model's intrinsic preferences for frames that best support temporally grounded responses. To efficiently explore the large combinatorial frame space, we employ an autoregressive, conditional selection architecture that ensures temporal coherence while reducing complexity. Our approach does not require explicit supervision at the frame-level and consistently improves reasoning performance across multiple video QA benchmarks, highlighting the benefits of aligning frame selection with model-internal utility.
>
---
#### [new 009] RAW Image Reconstruction from RGB on Smartphones. NTIRE 2025 Challenge Report
- **分类: cs.CV**

- **简介: 该论文属于图像处理任务，旨在从智能手机的sRGB图像中重建RAW图像，以“反向”ISP变换。由于RAW数据稀缺且昂贵，研究者尝试利用现有sRGB图像生成高质量RAW图像。论文组织了NTIRE 2025挑战赛，吸引150多个团队参与，提出了多种高效模型，建立了当前最佳的RAW图像重建方法和基准。**

- **链接: [http://arxiv.org/pdf/2506.01947v1](http://arxiv.org/pdf/2506.01947v1)**

> **作者:** Marcos V. Conde; Radu Timofte; Radu Berdan; Beril Besbinar; Daisuke Iso; Pengzhou Ji; Xiong Dun; Zeying Fan; Chen Wu; Zhansheng Wang; Pengbo Zhang; Jiazi Huang; Qinglin Liu; Wei Yu; Shengping Zhang; Xiangyang Ji; Kyungsik Kim; Minkyung Kim; Hwalmin Lee; Hekun Ma; Huan Zheng; Yanyan Wei; Zhao Zhang; Jing Fang; Meilin Gao; Xiang Yu; Shangbin Xie; Mengyuan Sun; Huanjing Yue; Jingyu Yang Huize Cheng; Shaomeng Zhang; Zhaoyang Zhang; Haoxiang Liang
>
> **备注:** CVPR 2025 - New Trends in Image Restoration and Enhancement (NTIRE)
>
> **摘要:** Numerous low-level vision tasks operate in the RAW domain due to its linear properties, bit depth, and sensor designs. Despite this, RAW image datasets are scarce and more expensive to collect than the already large and public sRGB datasets. For this reason, many approaches try to generate realistic RAW images using sensor information and sRGB images. This paper covers the second challenge on RAW Reconstruction from sRGB (Reverse ISP). We aim to recover RAW sensor images from smartphones given the corresponding sRGB images without metadata and, by doing this, ``reverse" the ISP transformation. Over 150 participants joined this NTIRE 2025 challenge and submitted efficient models. The proposed methods and benchmark establish the state-of-the-art for generating realistic RAW data.
>
---
#### [new 010] TIME: TabPFN-Integrated Multimodal Engine for Robust Tabular-Image Learning
- **分类: cs.CV; cs.LG**

- **简介: 论文提出TIME框架，用于表格-图像多模态学习任务，旨在解决医疗等领域中表格数据缺乏预训练表示和缺失值处理的问题。工作包括集成TabPFN作为鲁棒表格编码器，结合图像特征，并探索融合策略，验证其在完整与缺失数据上的有效性。**

- **链接: [http://arxiv.org/pdf/2506.00813v1](http://arxiv.org/pdf/2506.00813v1)**

> **作者:** Jiaqi Luo; Yuan Yuan; Shixin Xu
>
> **摘要:** Tabular-image multimodal learning, which integrates structured tabular data with imaging data, holds great promise for a variety of tasks, especially in medical applications. Yet, two key challenges remain: (1) the lack of a standardized, pretrained representation for tabular data, as is commonly available in vision and language domains; and (2) the difficulty of handling missing values in the tabular modality, which are common in real-world medical datasets. To address these issues, we propose the TabPFN-Integrated Multimodal Engine (TIME), a novel multimodal framework that builds on the recently introduced tabular foundation model, TabPFN. TIME leverages TabPFN as a frozen tabular encoder to generate robust, strong embeddings that are naturally resilient to missing data, and combines them with image features from pretrained vision backbones. We explore a range of fusion strategies and tabular encoders, and evaluate our approach on both natural and medical datasets. Extensive experiments demonstrate that TIME consistently outperforms competitive baselines across both complete and incomplete tabular inputs, underscoring its practical value in real-world multimodal learning scenarios.
>
---
#### [new 011] SSAM: Self-Supervised Association Modeling for Test-Time Adaption
- **分类: cs.CV**

- **简介: 论文提出SSAM框架，用于测试时自适应（TTA）任务，旨在解决测试数据分布偏移下模型性能下降问题。现有方法冻结图像编码器，忽视其应对分布变化的重要性。SSAM通过软原型估计和原型锚定图像重建，实现编码器动态优化，提升跨分布识别能力。**

- **链接: [http://arxiv.org/pdf/2506.00513v1](http://arxiv.org/pdf/2506.00513v1)**

> **作者:** Yaxiong Wang; Zhenqiang Zhang; Lechao Cheng; Zhun Zhong; Dan Guo; Meng Wang
>
> **备注:** 10 papges
>
> **摘要:** Test-time adaption (TTA) has witnessed important progress in recent years, the prevailing methods typically first encode the image and the text and design strategies to model the association between them. Meanwhile, the image encoder is usually frozen due to the absence of explicit supervision in TTA scenarios. We identify a critical limitation in this paradigm: While test-time images often exhibit distribution shifts from training data, existing methods persistently freeze the image encoder due to the absence of explicit supervision during adaptation. This practice overlooks the image encoder's crucial role in bridging distribution shift between training and test. To address this challenge, we propose SSAM (Self-Supervised Association Modeling), a new TTA framework that enables dynamic encoder refinement through dual-phase association learning. Our method operates via two synergistic components: 1) Soft Prototype Estimation (SPE), which estimates probabilistic category associations to guide feature space reorganization, and 2) Prototype-anchored Image Reconstruction (PIR), enforcing encoder stability through cluster-conditional image feature reconstruction. Comprehensive experiments across diverse baseline methods and benchmarks demonstrate that SSAM can surpass state-of-the-art TTA baselines by a clear margin while maintaining computational efficiency. The framework's architecture-agnostic design and minimal hyperparameter dependence further enhance its practical applicability.
>
---
#### [new 012] Latent Guidance in Diffusion Models for Perceptual Evaluations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像质量评估任务，旨在解决无参考图像质量评估（NR-IQA）中缺乏感知一致性建模的问题。作者提出了一种基于潜在扩散模型和感知特征的引导方法PMG，通过多尺度特征提取实现感知一致性的建模，并在多个数据集上验证了其优越性能。**

- **链接: [http://arxiv.org/pdf/2506.00327v1](http://arxiv.org/pdf/2506.00327v1)**

> **作者:** Shreshth Saini; Ru-Ling Liao; Yan Ye; Alan C. Bovik
>
> **备注:** 24 Pages, 7 figures, 10 Tables
>
> **摘要:** Despite recent advancements in latent diffusion models that generate high-dimensional image data and perform various downstream tasks, there has been little exploration into perceptual consistency within these models on the task of No-Reference Image Quality Assessment (NR-IQA). In this paper, we hypothesize that latent diffusion models implicitly exhibit perceptually consistent local regions within the data manifold. We leverage this insight to guide on-manifold sampling using perceptual features and input measurements. Specifically, we propose Perceptual Manifold Guidance (PMG), an algorithm that utilizes pretrained latent diffusion models and perceptual quality features to obtain perceptually consistent multi-scale and multi-timestep feature maps from the denoising U-Net. We empirically demonstrate that these hyperfeatures exhibit high correlation with human perception in IQA tasks. Our method can be applied to any existing pretrained latent diffusion model and is straightforward to integrate. To the best of our knowledge, this paper is the first work on guiding diffusion model with perceptual features for NR-IQA. Extensive experiments on IQA datasets show that our method, LGDM, achieves state-of-the-art performance, underscoring the superior generalization capabilities of diffusion models for NR-IQA tasks.
>
---
#### [new 013] Improving Optical Flow and Stereo Depth Estimation by Leveraging Uncertainty-Based Learning Difficulties
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在提升光流和立体深度估计。通过引入基于不确定性的难度感知学习方法，解决传统统一损失函数忽视像素间学习难度差异的问题。提出了难度平衡（DB）损失和遮挡避免（OA）损失，分别关注困难像素并引导网络避开遮挡区域，从而提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.00324v1](http://arxiv.org/pdf/2506.00324v1)**

> **作者:** Jisoo Jeong; Hong Cai; Jamie Menjay Lin; Fatih Porikli
>
> **备注:** CVPRW2025
>
> **摘要:** Conventional training for optical flow and stereo depth models typically employs a uniform loss function across all pixels. However, this one-size-fits-all approach often overlooks the significant variations in learning difficulty among individual pixels and contextual regions. This paper investigates the uncertainty-based confidence maps which capture these spatially varying learning difficulties and introduces tailored solutions to address them. We first present the Difficulty Balancing (DB) loss, which utilizes an error-based confidence measure to encourage the network to focus more on challenging pixels and regions. Moreover, we identify that some difficult pixels and regions are affected by occlusions, resulting from the inherently ill-posed matching problem in the absence of real correspondences. To address this, we propose the Occlusion Avoiding (OA) loss, designed to guide the network into cycle consistency-based confident regions, where feature matching is more reliable. By combining the DB and OA losses, we effectively manage various types of challenging pixels and regions during training. Experiments on both optical flow and stereo depth tasks consistently demonstrate significant performance improvements when applying our proposed combination of the DB and OA losses.
>
---
#### [new 014] EPFL-Smart-Kitchen-30: Densely annotated cooking dataset with 3D kinematics to challenge video and language models
- **分类: cs.CV; cs.AI; cs.LG; q-bio.OT**

- **简介: 该论文属于行为理解与建模任务，旨在解决复杂人类行为的数据缺失问题。作者构建了EPFL-Smart-Kitchen-30数据集，包含多视角动作、眼动、体态等信息，并提出四个基准测试，推动视频与语言模型发展。**

- **链接: [http://arxiv.org/pdf/2506.01608v1](http://arxiv.org/pdf/2506.01608v1)**

> **作者:** Andy Bonnetto; Haozhe Qi; Franklin Leong; Matea Tashkovska; Mahdi Rad; Solaiman Shokur; Friedhelm Hummel; Silvestro Micera; Marc Pollefeys; Alexander Mathis
>
> **备注:** Code and data at: https://github.com/amathislab/EPFL-Smart-Kitchen
>
> **摘要:** Understanding behavior requires datasets that capture humans while carrying out complex tasks. The kitchen is an excellent environment for assessing human motor and cognitive function, as many complex actions are naturally exhibited in kitchens from chopping to cleaning. Here, we introduce the EPFL-Smart-Kitchen-30 dataset, collected in a noninvasive motion capture platform inside a kitchen environment. Nine static RGB-D cameras, inertial measurement units (IMUs) and one head-mounted HoloLens~2 headset were used to capture 3D hand, body, and eye movements. The EPFL-Smart-Kitchen-30 dataset is a multi-view action dataset with synchronized exocentric, egocentric, depth, IMUs, eye gaze, body and hand kinematics spanning 29.7 hours of 16 subjects cooking four different recipes. Action sequences were densely annotated with 33.78 action segments per minute. Leveraging this multi-modal dataset, we propose four benchmarks to advance behavior understanding and modeling through 1) a vision-language benchmark, 2) a semantic text-to-motion generation benchmark, 3) a multi-modal action recognition benchmark, 4) a pose-based action segmentation benchmark. We expect the EPFL-Smart-Kitchen-30 dataset to pave the way for better methods as well as insights to understand the nature of ecologically-valid human behavior. Code and data are available at https://github.com/amathislab/EPFL-Smart-Kitchen
>
---
#### [new 015] PointT2I: LLM-based text-to-image generation via keypoints
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决复杂概念、尤其是人体姿态描述下生成准确图像的挑战。论文提出PointT2I框架，利用大语言模型生成关键点，结合文本和关键点生成图像，并通过反馈系统提升一致性，实现无需微调的姿态对齐图像生成。**

- **链接: [http://arxiv.org/pdf/2506.01370v1](http://arxiv.org/pdf/2506.01370v1)**

> **作者:** Taekyung Lee; Donggyu Lee; Myungjoo Kang
>
> **摘要:** Text-to-image (T2I) generation model has made significant advancements, resulting in high-quality images aligned with an input prompt. However, despite T2I generation's ability to generate fine-grained images, it still faces challenges in accurately generating images when the input prompt contains complex concepts, especially human pose. In this paper, we propose PointT2I, a framework that effectively generates images that accurately correspond to the human pose described in the prompt by using a large language model (LLM). PointT2I consists of three components: Keypoint generation, Image generation, and Feedback system. The keypoint generation uses an LLM to directly generate keypoints corresponding to a human pose, solely based on the input prompt, without external references. Subsequently, the image generation produces images based on both the text prompt and the generated keypoints to accurately reflect the target pose. To refine the outputs of the preceding stages, we incorporate an LLM-based feedback system that assesses the semantic consistency between the generated contents and the given prompts. Our framework is the first approach to leveraging LLM for keypoints-guided image generation without any fine-tuning, producing accurate pose-aligned images based solely on textual prompts.
>
---
#### [new 016] Test-time Vocabulary Adaptation for Language-driven Object Detection
- **分类: cs.CV**

- **简介: 该论文属于开放词汇目标检测任务，旨在解决用户自定义词汇可能过于宽泛或错误指定导致检测性能下降的问题。论文提出了一种无需训练、即插即用的词汇适配器（VocAda），在推理时通过图像描述生成、名词解析和类别筛选三步自动优化用户定义的词汇表，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.00333v1](http://arxiv.org/pdf/2506.00333v1)**

> **作者:** Mingxuan Liu; Tyler L. Hayes; Massimiliano Mancini; Elisa Ricci; Riccardo Volpi; Gabriela Csurka
>
> **备注:** Accepted as a conference paper at ICIP 2025
>
> **摘要:** Open-vocabulary object detection models allow users to freely specify a class vocabulary in natural language at test time, guiding the detection of desired objects. However, vocabularies can be overly broad or even mis-specified, hampering the overall performance of the detector. In this work, we propose a plug-and-play Vocabulary Adapter (VocAda) to refine the user-defined vocabulary, automatically tailoring it to categories that are relevant for a given image. VocAda does not require any training, it operates at inference time in three steps: i) it uses an image captionner to describe visible objects, ii) it parses nouns from those captions, and iii) it selects relevant classes from the user-defined vocabulary, discarding irrelevant ones. Experiments on COCO and Objects365 with three state-of-the-art detectors show that VocAda consistently improves performance, proving its versatility. The code is open source.
>
---
#### [new 017] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在复杂指令跟随中的推理不足问题。通过分解指令、强化学习与专家克隆，提升模型对多约束结构的理解与执行能力。**

- **链接: [http://arxiv.org/pdf/2506.01413v1](http://arxiv.org/pdf/2506.01413v1)**

> **作者:** Yulei Qin; Gang Li; Zongyi Li; Zihan Xu; Yuchen Shi; Zhekai Lin; Xiao Cui; Ke Li; Xing Sun
>
> **备注:** 10 pages of main body, 3 tables, 5 figures, 40 pages of appendix
>
> **摘要:** Existing large language models (LLMs) face challenges of following complex instructions, especially when multiple constraints are present and organized in paralleling, chaining, and branching structures. One intuitive solution, namely chain-of-thought (CoT), is expected to universally improve capabilities of LLMs. However, we find that the vanilla CoT exerts a negative impact on performance due to its superficial reasoning pattern of simply paraphrasing the instructions. It fails to peel back the compositions of constraints for identifying their relationship across hierarchies of types and dimensions. To this end, we propose a systematic method to boost LLMs in dealing with complex instructions via incentivizing reasoning for test-time compute scaling. First, we stem from the decomposition of complex instructions under existing taxonomies and propose a reproducible data acquisition method. Second, we exploit reinforcement learning (RL) with verifiable rule-centric reward signals to cultivate reasoning specifically for instruction following. We address the shallow, non-essential nature of reasoning under complex instructions via sample-wise contrast for superior CoT enforcement. We also exploit behavior cloning of experts to facilitate steady distribution shift from fast-thinking LLMs to skillful reasoners. Extensive evaluations on seven comprehensive benchmarks confirm the validity of the proposed method, where a 1.5B LLM achieves 11.74% gains with performance comparable to a 8B LLM. Codes and data are available at https://github.com/yuleiqin/RAIF.
>
---
#### [new 018] Involution-Infused DenseNet with Two-Step Compression for Resource-Efficient Plant Disease Classification
- **分类: cs.CV**

- **简介: 该论文属于植物病害分类任务，旨在解决传统CNN模型计算量大、难以部署到资源受限设备的问题。工作包括：提出结合权重剪枝与知识蒸馏的两步压缩方法，融合DenseNet与Involution层提升效率。实验表明模型在保持高精度的同时显著减少参数量，适用于实时农业病害检测。**

- **链接: [http://arxiv.org/pdf/2506.00735v1](http://arxiv.org/pdf/2506.00735v1)**

> **作者:** T. Ahmed; S. Jannat; Md. F. Islam; J. Noor
>
> **摘要:** Agriculture is vital for global food security, but crops are vulnerable to diseases that impact yield and quality. While Convolutional Neural Networks (CNNs) accurately classify plant diseases using leaf images, their high computational demands hinder their deployment in resource-constrained settings such as smartphones, edge devices, and real-time monitoring systems. This study proposes a two-step model compression approach integrating Weight Pruning and Knowledge Distillation, along with the hybridization of DenseNet with Involutional Layers. Pruning reduces model size and computational load, while distillation improves the smaller student models performance by transferring knowledge from a larger teacher network. The hybridization enhances the models ability to capture spatial features efficiently. These compressed models are suitable for real-time applications, promoting precision agriculture through rapid disease identification and crop management. The results demonstrate ResNet50s superior performance post-compression, achieving 99.55% and 98.99% accuracy on the PlantVillage and PaddyLeaf datasets, respectively. The DenseNet-based model, optimized for efficiency, recorded 99.21% and 93.96% accuracy with a minimal parameter count. Furthermore, the hybrid model achieved 98.87% and 97.10% accuracy, supporting the practical deployment of energy-efficient devices for timely disease intervention and sustainable farming practices.
>
---
#### [new 019] ProstaTD: A Large-scale Multi-source Dataset for Structured Surgical Triplet Detection
- **分类: cs.CV**

- **简介: 该论文属于手术视频分析任务，旨在解决现有数据集在标注精度、临床相关性和多样性方面的不足。作者构建了大规模多中心数据集ProstaTD，提供精确时空标注和多样化的手术场景，用于推动手术AI系统和培训工具的发展。**

- **链接: [http://arxiv.org/pdf/2506.01130v1](http://arxiv.org/pdf/2506.01130v1)**

> **作者:** Yiliang Chen; Zhixi Li; Cheng Xu; Alex Qinyang Liu; Xuemiao Xu; Jeremy Yuen-Chun Teoh; Shengfeng He; Jing Qin
>
> **摘要:** Surgical triplet detection has emerged as a pivotal task in surgical video analysis, with significant implications for performance assessment and the training of novice surgeons. However, existing datasets such as CholecT50 exhibit critical limitations: they lack precise spatial bounding box annotations, provide inconsistent and clinically ungrounded temporal labels, and rely on a single data source, which limits model generalizability.To address these shortcomings, we introduce ProstaTD, a large-scale, multi-institutional dataset for surgical triplet detection, developed from the technically demanding domain of robot-assisted prostatectomy. ProstaTD offers clinically defined temporal boundaries and high-precision bounding box annotations for each structured triplet action. The dataset comprises 60,529 video frames and 165,567 annotated triplet instances, collected from 21 surgeries performed across multiple institutions, reflecting a broad range of surgical practices and intraoperative conditions. The annotation process was conducted under rigorous medical supervision and involved more than 50 contributors, including practicing surgeons and medically trained annotators, through multiple iterative phases of labeling and verification. ProstaTD is the largest and most diverse surgical triplet dataset to date, providing a robust foundation for fair benchmarking, the development of reliable surgical AI systems, and scalable tools for procedural training.
>
---
#### [new 020] Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control
- **分类: cs.CV**

- **简介: 该论文属于视频生成与机器人控制任务，旨在解决复杂多物体交互场景中视觉质量下降的问题。现有方法难以处理多物体特征融合导致的视觉模糊。论文提出RoboMaster框架，通过将交互过程分解为三个阶段并使用主导物体特征建模，避免多物体特征纠缠，并引入外观和形状感知的潜在表示以保持语义一致性，从而提升轨迹控制下的视频生成效果。**

- **链接: [http://arxiv.org/pdf/2506.01943v1](http://arxiv.org/pdf/2506.01943v1)**

> **作者:** Xiao Fu; Xintao Wang; Xian Liu; Jianhong Bai; Runsen Xu; Pengfei Wan; Di Zhang; Dahua Lin
>
> **备注:** Project Page: https://fuxiao0719.github.io/projects/robomaster/ Code: https://github.com/KwaiVGI/RoboMaster
>
> **摘要:** Recent advances in video diffusion models have demonstrated strong potential for generating robotic decision-making data, with trajectory conditions further enabling fine-grained control. However, existing trajectory-based methods primarily focus on individual object motion and struggle to capture multi-object interaction crucial in complex robotic manipulation. This limitation arises from multi-feature entanglement in overlapping regions, which leads to degraded visual fidelity. To address this, we present RoboMaster, a novel framework that models inter-object dynamics through a collaborative trajectory formulation. Unlike prior methods that decompose objects, our core is to decompose the interaction process into three sub-stages: pre-interaction, interaction, and post-interaction. Each stage is modeled using the feature of the dominant object, specifically the robotic arm in the pre- and post-interaction phases and the manipulated object during interaction, thereby mitigating the drawback of multi-object feature fusion present during interaction in prior work. To further ensure subject semantic consistency throughout the video, we incorporate appearance- and shape-aware latent representations for objects. Extensive experiments on the challenging Bridge V2 dataset, as well as in-the-wild evaluation, demonstrate that our method outperforms existing approaches, establishing new state-of-the-art performance in trajectory-controlled video generation for robotic manipulation.
>
---
#### [new 021] Concept-Centric Token Interpretation for Vector-Quantized Generative Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成模型解释任务，旨在解决向量量化生成模型（VQGMs）中离散码本难以理解的问题。论文提出了CORTEX方法，通过样本级和码本级解释分析关键token组合，提升了模型透明度，并支持图像编辑与特征检测应用。**

- **链接: [http://arxiv.org/pdf/2506.00698v1](http://arxiv.org/pdf/2506.00698v1)**

> **作者:** Tianze Yang; Yucheng Shi; Mengnan Du; Xuansheng Wu; Qiaoyu Tan; Jin Sun; Ninghao Liu
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Vector-Quantized Generative Models (VQGMs) have emerged as powerful tools for image generation. However, the key component of VQGMs -- the codebook of discrete tokens -- is still not well understood, e.g., which tokens are critical to generate an image of a certain concept? This paper introduces Concept-Oriented Token Explanation (CORTEX), a novel approach for interpreting VQGMs by identifying concept-specific token combinations. Our framework employs two methods: (1) a sample-level explanation method that analyzes token importance scores in individual images, and (2) a codebook-level explanation method that explores the entire codebook to find globally relevant tokens. Experimental results demonstrate CORTEX's efficacy in providing clear explanations of token usage in the generative process, outperforming baselines across multiple pretrained VQGMs. Besides enhancing VQGMs transparency, CORTEX is useful in applications such as targeted image editing and shortcut feature detection. Our code is available at https://github.com/YangTianze009/CORTEX.
>
---
#### [new 022] ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决现有方法缺乏动态反馈、难以自适应调整的问题。作者提出ReAgent-V框架，通过引入实时奖励信号和多视角反思机制，实现答案迭代优化与高质量数据筛选，提升模型在复杂场景下的推理能力与泛化表现。**

- **链接: [http://arxiv.org/pdf/2506.01300v1](http://arxiv.org/pdf/2506.01300v1)**

> **作者:** Yiyang Zhou; Yangfan He; Yaofeng Su; Siwei Han; Joel Jang; Gedas Bertasius; Mohit Bansal; Huaxiu Yao
>
> **备注:** 31 pages, 18 figures
>
> **摘要:** Video understanding is fundamental to tasks such as action recognition, video reasoning, and robotic control. Early video understanding methods based on large vision-language models (LVLMs) typically adopt a single-pass reasoning paradigm without dynamic feedback, limiting the model's capacity to self-correct and adapt in complex scenarios. Recent efforts have attempted to address this limitation by incorporating reward models and reinforcement learning to enhance reasoning, or by employing tool-agent frameworks. However, these approaches face several challenges, including high annotation costs, reward signals that fail to capture real-time reasoning states, and low inference efficiency. To overcome these issues, we propose ReAgent-V, a novel agentic video understanding framework that integrates efficient frame selection with real-time reward generation during inference. These reward signals not only guide iterative answer refinement through a multi-perspective reflection mechanism-adjusting predictions from conservative, neutral, and aggressive viewpoints-but also enable automatic filtering of high-quality data for supervised fine-tuning (SFT), direct preference optimization (DPO), and group relative policy optimization (GRPO). ReAgent-V is lightweight, modular, and extensible, supporting flexible tool integration tailored to diverse tasks. Extensive experiments on 12 datasets across three core applications-video understanding, video reasoning enhancement, and vision-language-action model alignment-demonstrate significant gains in generalization and reasoning, with improvements of up to 6.9%, 2.1%, and 9.8%, respectively, highlighting the effectiveness and versatility of the proposed framework.
>
---
#### [new 023] Latent Wavelet Diffusion: Enabling 4K Image Synthesis for Free
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于图像生成任务，旨在解决高分辨率（2K至4K）图像合成中的计算效率与细节保留问题。论文提出了Latent Wavelet Diffusion（LWD），通过频域感知的监督方法，提升潜变量扩散模型在超高清图像生成中的表现，无需修改架构且无额外开销。**

- **链接: [http://arxiv.org/pdf/2506.00433v1](http://arxiv.org/pdf/2506.00433v1)**

> **作者:** Luigi Sigillo; Shengfeng He; Danilo Comminiello
>
> **摘要:** High-resolution image synthesis remains a core challenge in generative modeling, particularly in balancing computational efficiency with the preservation of fine-grained visual detail. We present Latent Wavelet Diffusion (LWD), a lightweight framework that enables any latent diffusion model to scale to ultra-high-resolution image generation (2K to 4K) for free. LWD introduces three key components: (1) a scale-consistent variational autoencoder objective that enhances the spectral fidelity of latent representations; (2) wavelet energy maps that identify and localize detail-rich spatial regions within the latent space; and (3) a time-dependent masking strategy that focuses denoising supervision on high-frequency components during training. LWD requires no architectural modifications and incurs no additional computational overhead. Despite its simplicity, it consistently improves perceptual quality and reduces FID in ultra-high-resolution image synthesis, outperforming strong baseline models. These results highlight the effectiveness of frequency-aware, signal-driven supervision as a principled and efficient approach for high-resolution generative modeling.
>
---
#### [new 024] Visual Sparse Steering: Improving Zero-shot Image Classification with Sparsity Guided Steering Vectors
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像分类任务，旨在提升零样本视觉模型推理性能。提出轻量级方法VS2，通过稀疏自编码器提取引导向量，在测试时无需重训练或对比数据。进一步设计VS2++结合检索增强，并提出PASS改进稀疏特征对齐，显著提升分类准确率，尤其对相似类别效果明显。**

- **链接: [http://arxiv.org/pdf/2506.01247v1](http://arxiv.org/pdf/2506.01247v1)**

> **作者:** Gerasimos Chatzoudis; Zhuowei Li; Gemma E. Moran; Hao Wang; Dimitris N. Metaxas
>
> **摘要:** Steering vision foundation models at inference time without retraining or access to large labeled datasets is a desirable yet challenging objective, particularly in dynamic or resource-constrained settings. In this paper, we introduce Visual Sparse Steering (VS2), a lightweight, test-time method that guides vision models using steering vectors derived from sparse features learned by top-$k$ Sparse Autoencoders without requiring contrastive data. Specifically, VS2 surpasses zero-shot CLIP by 4.12% on CIFAR-100, 1.08% on CUB-200, and 1.84% on Tiny-ImageNet. We further propose VS2++, a retrieval-augmented variant that selectively amplifies relevant sparse features using pseudo-labeled neighbors at inference time. With oracle positive/negative sets, VS2++ achieves absolute top-1 gains over CLIP zero-shot of up to 21.44% on CIFAR-100, 7.08% on CUB-200, and 20.47% on Tiny-ImageNet. Interestingly, VS2 and VS2++ raise per-class accuracy by up to 25% and 38%, respectively, showing that sparse steering benefits specific classes by disambiguating visually or taxonomically proximate categories rather than providing a uniform boost. Finally, to better align the sparse features learned through the SAE reconstruction task with those relevant for downstream performance, we propose Prototype-Aligned Sparse Steering (PASS). By incorporating a prototype-alignment loss during SAE training, using labels only during training while remaining fully test-time unsupervised, PASS consistently, though modestly, outperforms VS2, achieving a 6.12% gain over VS2 only on CIFAR-100 with ViT-B/32.
>
---
#### [new 025] Breaking Latent Prior Bias in Detectors for Generalizable AIGC Image Detection
- **分类: cs.CV**

- **简介: 该论文属于AIGC图像检测任务，旨在解决检测器在不同生成模型间泛化能力差的问题。作者提出On-Manifold Adversarial Training（OMAT）方法，通过优化扩散模型的初始噪声生成对抗样本，提升检测器对未见生成器的鲁棒性，并构建新基准GenImage++用于测试。实验表明该方法有效提升了跨生成器的检测性能。**

- **链接: [http://arxiv.org/pdf/2506.00874v1](http://arxiv.org/pdf/2506.00874v1)**

> **作者:** Yue Zhou; Xinan He; KaiQing Lin; Bin Fan; Feng Ding; Bin Li
>
> **摘要:** Current AIGC detectors often achieve near-perfect accuracy on images produced by the same generator used for training but struggle to generalize to outputs from unseen generators. We trace this failure in part to latent prior bias: detectors learn shortcuts tied to patterns stemming from the initial noise vector rather than learning robust generative artifacts. To address this, we propose On-Manifold Adversarial Training (OMAT): by optimizing the initial latent noise of diffusion models under fixed conditioning, we generate on-manifold adversarial examples that remain on the generator's output manifold-unlike pixel-space attacks, which introduce off-manifold perturbations that the generator itself cannot reproduce and that can obscure the true discriminative artifacts. To test against state-of-the-art generative models, we introduce GenImage++, a test-only benchmark of outputs from advanced generators (Flux.1, SD3) with extended prompts and diverse styles. We apply our adversarial-training paradigm to ResNet50 and CLIP baselines and evaluate across existing AIGC forensic benchmarks and recent challenge datasets. Extensive experiments show that adversarially trained detectors significantly improve cross-generator performance without any network redesign. Our findings on latent-prior bias offer valuable insights for future dataset construction and detector evaluation, guiding the development of more robust and generalizable AIGC forensic methodologies.
>
---
#### [new 026] Playing with Transformer at 30+ FPS via Next-Frame Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决自回归模型在实时视频生成中的效率问题。通过提出Next-Frame Diffusion（NFD）模型，结合一致性蒸馏和推测采样技术，实现高效并行计算，在A100 GPU上达到30+ FPS的实时视频生成效果。**

- **链接: [http://arxiv.org/pdf/2506.01380v1](http://arxiv.org/pdf/2506.01380v1)**

> **作者:** Xinle Cheng; Tianyu He; Jiayi Xu; Junliang Guo; Di He; Jiang Bian
>
> **摘要:** Autoregressive video models offer distinct advantages over bidirectional diffusion models in creating interactive video content and supporting streaming applications with arbitrary duration. In this work, we present Next-Frame Diffusion (NFD), an autoregressive diffusion transformer that incorporates block-wise causal attention, enabling iterative sampling and efficient inference via parallel token generation within each frame. Nonetheless, achieving real-time video generation remains a significant challenge for such models, primarily due to the high computational cost associated with diffusion sampling and the hardware inefficiencies inherent to autoregressive generation. To address this, we introduce two innovations: (1) We extend consistency distillation to the video domain and adapt it specifically for video models, enabling efficient inference with few sampling steps; (2) To fully leverage parallel computation, motivated by the observation that adjacent frames often share the identical action input, we propose speculative sampling. In this approach, the model generates next few frames using current action input, and discard speculatively generated frames if the input action differs. Experiments on a large-scale action-conditioned video generation benchmark demonstrate that NFD beats autoregressive baselines in terms of both visual quality and sampling efficiency. We, for the first time, achieves autoregressive video generation at over 30 Frames Per Second (FPS) on an A100 GPU using a 310M model.
>
---
#### [new 027] Revolutionizing Blood Banks: AI-Driven Fingerprint-Blood Group Correlation for Enhanced Safety
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究指纹类型与ABO血型的关系，旨在提升生物识别技术的安全性。通过分析200人的指纹和血型数据，使用统计方法评估二者关联性。结果显示指纹与血型无显著相关性，表明结合血型无法显著提升指纹识别的准确性。任务属于生物特征识别领域，试图解决身份识别中的安全问题，探索多模态生物识别系统的潜在改进方向。**

- **链接: [http://arxiv.org/pdf/2506.01069v1](http://arxiv.org/pdf/2506.01069v1)**

> **作者:** Malik A. Altayar; Muhyeeddin Alqaraleh; Mowafaq Salem Alzboon; Wesam T. Almagharbeh
>
> **摘要:** Identification of a person is central in forensic science, security, and healthcare. Methods such as iris scanning and genomic profiling are more accurate but expensive, time-consuming, and more difficult to implement. This study focuses on the relationship between the fingerprint patterns and the ABO blood group as a biometric identification tool. A total of 200 subjects were included in the study, and fingerprint types (loops, whorls, and arches) and blood groups were compared. Associations were evaluated with statistical tests, including chi-square and Pearson correlation. The study found that the loops were the most common fingerprint pattern and the O+ blood group was the most prevalent. Even though there was some associative pattern, there was no statistically significant difference in the fingerprint patterns of different blood groups. Overall, the results indicate that blood group data do not significantly improve personal identification when used in conjunction with fingerprinting. Although the study shows weak correlation, it may emphasize the efforts of multi-modal based biometric systems in enhancing the current biometric systems. Future studies may focus on larger and more diverse samples, and possibly machine learning and additional biometrics to improve identification methods. This study addresses an element of the ever-changing nature of the fields of forensic science and biometric identification, highlighting the importance of resilient analytical methods for personal identification.
>
---
#### [new 028] OmniV2V: Versatile Video Generation and Editing via Dynamic Content Manipulation
- **分类: cs.CV**

- **简介: 该论文属于视频生成与编辑任务，旨在解决现有模型场景单一、缺乏动态内容操控能力的问题。作者提出了OmniV2V模型，支持多种视频编辑操作，并设计了内容操控模块和视觉-文本指令模块，构建了多任务数据系统及对应数据集，实现了跨场景的视频生成与编辑。**

- **链接: [http://arxiv.org/pdf/2506.01801v1](http://arxiv.org/pdf/2506.01801v1)**

> **作者:** Sen Liang; Zhentao Yu; Zhengguang Zhou; Teng Hu; Hongmei Wang; Yi Chen; Qin Lin; Yuan Zhou; Xin Li; Qinglin Lu; Zhibo Chen
>
> **摘要:** The emergence of Diffusion Transformers (DiT) has brought significant advancements to video generation, especially in text-to-video and image-to-video tasks. Although video generation is widely applied in various fields, most existing models are limited to single scenarios and cannot perform diverse video generation and editing through dynamic content manipulation. We propose OmniV2V, a video model capable of generating and editing videos across different scenarios based on various operations, including: object movement, object addition, mask-guided video edit, try-on, inpainting, outpainting, human animation, and controllable character video synthesis. We explore a unified dynamic content manipulation injection module, which effectively integrates the requirements of the above tasks. In addition, we design a visual-text instruction module based on LLaVA, enabling the model to effectively understand the correspondence between visual content and instructions. Furthermore, we build a comprehensive multi-task data processing system. Since there is data overlap among various tasks, this system can efficiently provide data augmentation. Using this system, we construct a multi-type, multi-scenario OmniV2V dataset and its corresponding OmniV2V-Test benchmark. Extensive experiments show that OmniV2V works as well as, and sometimes better than, the best existing open-source and commercial models for many video generation and editing tasks.
>
---
#### [new 029] E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于3D几何基础模型评估任务，旨在解决缺乏系统评测的问题。作者构建了首个全面基准E3D-Bench，涵盖5项核心任务和多种数据集，评估16个最先进模型，提供公平、可复现的比较，并指导未来研究方向。**

- **链接: [http://arxiv.org/pdf/2506.01933v1](http://arxiv.org/pdf/2506.01933v1)**

> **作者:** Wenyan Cong; Yiqing Liang; Yancheng Zhang; Ziyi Yang; Yan Wang; Boris Ivanovic; Marco Pavone; Chen Chen; Zhangyang Wang; Zhiwen Fan
>
> **备注:** Project Page: https://e3dbench.github.io/
>
> **摘要:** Spatial intelligence, encompassing 3D reconstruction, perception, and reasoning, is fundamental to applications such as robotics, aerial imaging, and extended reality. A key enabler is the real-time, accurate estimation of core 3D attributes (camera parameters, point clouds, depth maps, and 3D point tracks) from unstructured or streaming imagery. Inspired by the success of large foundation models in language and 2D vision, a new class of end-to-end 3D geometric foundation models (GFMs) has emerged, directly predicting dense 3D representations in a single feed-forward pass, eliminating the need for slow or unavailable precomputed camera parameters. Since late 2023, the field has exploded with diverse variants, but systematic evaluation is lacking. In this work, we present the first comprehensive benchmark for 3D GFMs, covering five core tasks: sparse-view depth estimation, video depth estimation, 3D reconstruction, multi-view pose estimation, novel view synthesis, and spanning both standard and challenging out-of-distribution datasets. Our standardized toolkit automates dataset handling, evaluation protocols, and metric computation to ensure fair, reproducible comparisons. We evaluate 16 state-of-the-art GFMs, revealing their strengths and limitations across tasks and domains, and derive key insights to guide future model scaling and optimization. All code, evaluation scripts, and processed data will be publicly released to accelerate research in 3D spatial intelligence.
>
---
#### [new 030] GSCodec Studio: A Modular Framework for Gaussian Splat Compression
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于3D/4D高斯点压缩任务，旨在解决高斯点数据存储需求高、压缩方法分散难比较的问题。作者提出了GSCodec Studio框架，整合多种重建与压缩模块，支持静态与动态高斯点的高效压缩，并提供统一平台促进相关研究发展。**

- **链接: [http://arxiv.org/pdf/2506.01822v1](http://arxiv.org/pdf/2506.01822v1)**

> **作者:** Sicheng Li; Chengzhen Wu; Hao Li; Xiang Gao; Yiyi Liao; Lu Yu
>
> **备注:** Repository of the project: https://github.com/JasonLSC/GSCodec_Studio
>
> **摘要:** 3D Gaussian Splatting and its extension to 4D dynamic scenes enable photorealistic, real-time rendering from real-world captures, positioning Gaussian Splats (GS) as a promising format for next-generation immersive media. However, their high storage requirements pose significant challenges for practical use in sharing, transmission, and storage. Despite various studies exploring GS compression from different perspectives, these efforts remain scattered across separate repositories, complicating benchmarking and the integration of best practices. To address this gap, we present GSCodec Studio, a unified and modular framework for GS reconstruction, compression, and rendering. The framework incorporates a diverse set of 3D/4D GS reconstruction methods and GS compression techniques as modular components, facilitating flexible combinations and comprehensive comparisons. By integrating best practices from community research and our own explorations, GSCodec Studio supports the development of compact representation and compression solutions for static and dynamic Gaussian Splats, namely our Static and Dynamic GSCodec, achieving competitive rate-distortion performance in static and dynamic GS compression. The code for our framework is publicly available at https://github.com/JasonLSC/GSCodec_Studio , to advance the research on Gaussian Splats compression.
>
---
#### [new 031] MedEBench: Revisiting Text-instructed Image Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本引导的医学图像编辑任务，旨在解决缺乏标准化评估的问题。论文提出了MedEBench基准，包含1,182个临床图像-提示对，覆盖70项任务。提供了包含编辑准确性、上下文保持和视觉质量的评估框架，并系统比较了七种模型的表现，分析了其失败模式。**

- **链接: [http://arxiv.org/pdf/2506.01921v1](http://arxiv.org/pdf/2506.01921v1)**

> **作者:** Minghao Liu; Zhitao He; Zhiyuan Fan; Qingyun Wang; Yi R. Fung
>
> **摘要:** Text-guided image editing has seen rapid progress in natural image domains, but its adaptation to medical imaging remains limited and lacks standardized evaluation. Clinically, such editing holds promise for simulating surgical outcomes, creating personalized teaching materials, and enhancing patient communication. To bridge this gap, we introduce \textbf{MedEBench}, a comprehensive benchmark for evaluating text-guided medical image editing. It consists of 1,182 clinically sourced image-prompt triplets spanning 70 tasks across 13 anatomical regions. MedEBench offers three key contributions: (1) a clinically relevant evaluation framework covering Editing Accuracy, Contextual Preservation, and Visual Quality, supported by detailed descriptions of expected change and ROI (Region of Interest) masks; (2) a systematic comparison of seven state-of-the-art models, revealing common failure patterns; and (3) a failure analysis protocol based on attention grounding, using IoU between attention maps and ROIs to identify mislocalization. MedEBench provides a solid foundation for developing and evaluating reliable, clinically meaningful medical image editing systems.
>
---
#### [new 032] Towards Predicting Any Human Trajectory In Context
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于行人轨迹预测任务，旨在解决不同场景下模型适应性差的问题。作者提出TrajICL框架，通过上下文学习与时空相似性选择示例，无需微调即可实现快速适应。方法包括STES与PG-ES策略，并基于大规模合成数据训练提升预测性能。**

- **链接: [http://arxiv.org/pdf/2506.00871v1](http://arxiv.org/pdf/2506.00871v1)**

> **作者:** Ryo Fujii; Hideo Saito; Ryo Hachiuma
>
> **摘要:** Predicting accurate future trajectories of pedestrians is essential for autonomous systems but remains a challenging task due to the need for adaptability in different environments and domains. A common approach involves collecting scenario-specific data and performing fine-tuning via backpropagation. However, this process is often impractical on edge devices due to constrained computational resources. To address this challenge, we introduce TrajICL, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables rapid adaptation without fine-tuning on the scenario-specific data. We propose a spatio-temporal similarity-based example selection (STES) method that selects relevant examples from previously observed trajectories within the same scene by identifying similar motion patterns at corresponding locations. To further refine this selection, we introduce prediction-guided example selection (PG-ES), which selects examples based on both the past trajectory and the predicted future trajectory, rather than relying solely on the past trajectory. This approach allows the model to account for long-term dynamics when selecting examples. Finally, instead of relying on small real-world datasets with limited scenario diversity, we train our model on a large-scale synthetic dataset to enhance its prediction ability by leveraging in-context examples. Extensive experiments demonstrate that TrajICL achieves remarkable adaptation across both in-domain and cross-domain scenarios, outperforming even fine-tuned approaches across multiple public benchmarks. The code will be released at https://fujiry0.github.io/TrajICL-project-page.
>
---
#### [new 033] ViTA-PAR: Visual and Textual Attribute Alignment with Attribute Prompting for Pedestrian Attribute Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于行人属性识别（PAR）任务，旨在提升模型对行人不同粒度属性的识别能力。现有方法受限于固定区域划分，难以处理属性位置变化问题。论文提出ViTA-PAR，通过视觉与文本属性对齐、多模态提示学习，实现全局到局部特征融合，提升了识别性能。**

- **链接: [http://arxiv.org/pdf/2506.01411v1](http://arxiv.org/pdf/2506.01411v1)**

> **作者:** Minjeong Park; Hongbeen Park; Jinkyu Kim
>
> **备注:** Accepted to IEEE ICIP 2025
>
> **摘要:** The Pedestrian Attribute Recognition (PAR) task aims to identify various detailed attributes of an individual, such as clothing, accessories, and gender. To enhance PAR performance, a model must capture features ranging from coarse-grained global attributes (e.g., for identifying gender) to fine-grained local details (e.g., for recognizing accessories) that may appear in diverse regions. Recent research suggests that body part representation can enhance the model's robustness and accuracy, but these methods are often restricted to attribute classes within fixed horizontal regions, leading to degraded performance when attributes appear in varying or unexpected body locations. In this paper, we propose Visual and Textual Attribute Alignment with Attribute Prompting for Pedestrian Attribute Recognition, dubbed as ViTA-PAR, to enhance attribute recognition through specialized multimodal prompting and vision-language alignment. We introduce visual attribute prompts that capture global-to-local semantics, enabling diverse attribute representations. To enrich textual embeddings, we design a learnable prompt template, termed person and attribute context prompting, to learn person and attributes context. Finally, we align visual and textual attribute features for effective fusion. ViTA-PAR is validated on four PAR benchmarks, achieving competitive performance with efficient inference. We release our code and model at https://github.com/mlnjeongpark/ViTA-PAR.
>
---
#### [new 034] SAM2-LOVE: Segment Anything Model 2 in Language-aided Audio-Visual Scenes
- **分类: cs.CV**

- **简介: 该论文属于参考音频-视觉分割（Ref-AVS）任务，旨在解决多模态场景中目标分割不一致的问题。作者提出SAM2-LOVE框架，融合文本、音频和视觉信息，提升分割的时空一致性。通过多模态融合模块及记忆机制，在Ref-AVS基准上超越现有方法8.5%。**

- **链接: [http://arxiv.org/pdf/2506.01558v1](http://arxiv.org/pdf/2506.01558v1)**

> **作者:** Yuji Wang; Haoran Xu; Yong Liu; Jiaze Li; Yansong Tang
>
> **备注:** CVPR 2025
>
> **摘要:** Reference Audio-Visual Segmentation (Ref-AVS) aims to provide a pixel-wise scene understanding in Language-aided Audio-Visual Scenes (LAVS). This task requires the model to continuously segment objects referred to by text and audio from a video. Previous dual-modality methods always fail due to the lack of a third modality and the existing triple-modality method struggles with spatio-temporal consistency, leading to the target shift of different frames. In this work, we introduce a novel framework, termed SAM2-LOVE, which integrates textual, audio, and visual representations into a learnable token to prompt and align SAM2 for achieving Ref-AVS in the LAVS. Technically, our approach includes a multimodal fusion module aimed at improving multimodal understanding of SAM2, as well as token propagation and accumulation strategies designed to enhance spatio-temporal consistency without forgetting historical information. We conducted extensive experiments to demonstrate that SAM2-LOVE outperforms the SOTA by 8.5\% in $\mathcal{J\&F}$ on the Ref-AVS benchmark and showcase the simplicity and effectiveness of the components. Our code will be available here.
>
---
#### [new 035] Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于Sign Language Translation任务，旨在解决现有方法忽略骨骼数据几何特性的问题。作者提出Geo-Sign，利用双曲几何建模手语运动的层次结构，通过双曲投影层、加权Fréchet均值聚合和几何对比损失，提升模型对细微动作的表达能力，从而改善翻译性能。**

- **链接: [http://arxiv.org/pdf/2506.00129v1](http://arxiv.org/pdf/2506.00129v1)**

> **作者:** Edward Fish; Richard Bowden
>
> **备注:** Under Review
>
> **摘要:** Recent progress in Sign Language Translation (SLT) has focussed primarily on improving the representational capacity of large language models to incorporate Sign Language features. This work explores an alternative direction: enhancing the geometric properties of skeletal representations themselves. We propose Geo-Sign, a method that leverages the properties of hyperbolic geometry to model the hierarchical structure inherent in sign language kinematics. By projecting skeletal features derived from Spatio-Temporal Graph Convolutional Networks (ST-GCNs) into the Poincar\'e ball model, we aim to create more discriminative embeddings, particularly for fine-grained motions like finger articulations. We introduce a hyperbolic projection layer, a weighted Fr\'echet mean aggregation scheme, and a geometric contrastive loss operating directly in hyperbolic space. These components are integrated into an end-to-end translation framework as a regularisation function, to enhance the representations within the language model. This work demonstrates the potential of hyperbolic geometry to improve skeletal representations for Sign Language Translation, improving on SOTA RGB methods while preserving privacy and improving computational efficiency. Code available here: https://github.com/ed-fish/geo-sign.
>
---
#### [new 036] R2SM: Referring and Reasoning for Selective Masks
- **分类: cs.CV**

- **简介: 该论文提出R2SM任务，解决根据自然语言选择生成可见（modal）或完整（amodal）分割掩码的问题。作者构建了R2SM数据集，支持模型训练与评估，推动多模态推理与意图感知分割研究。**

- **链接: [http://arxiv.org/pdf/2506.01795v1](http://arxiv.org/pdf/2506.01795v1)**

> **作者:** Yu-Lin Shih; Wei-En Tai; Cheng Sun; Yu-Chiang Frank Wang; Hwann-Tzong Chen
>
> **摘要:** We introduce a new task, Referring and Reasoning for Selective Masks (R2SM), which extends text-guided segmentation by incorporating mask-type selection driven by user intent. This task challenges vision-language models to determine whether to generate a modal (visible) or amodal (complete) segmentation mask based solely on natural language prompts. To support the R2SM task, we present the R2SM dataset, constructed by augmenting annotations of COCOA-cls, D2SA, and MUVA. The R2SM dataset consists of both modal and amodal text queries, each paired with the corresponding ground-truth mask, enabling model finetuning and evaluation for the ability to segment images as per user intent. Specifically, the task requires the model to interpret whether a given prompt refers to only the visible part of an object or to its complete shape, including occluded regions, and then produce the appropriate segmentation. For example, if a prompt explicitly requests the whole shape of a partially hidden object, the model is expected to output an amodal mask that completes the occluded parts. In contrast, prompts without explicit mention of hidden regions should generate standard modal masks. The R2SM benchmark provides a challenging and insightful testbed for advancing research in multimodal reasoning and intent-aware segmentation.
>
---
#### [new 037] Enhancing Biomedical Multi-modal Representation Learning with Multi-scale Pre-training and Perturbed Report Discrimination
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于生物医学视觉-语言模型预训练任务，旨在解决现有对比学习方法在捕捉生物医学文本复杂语义上的不足。论文提出“扰动报告辨别”方法，通过区分原始与扰动文本，并结合多尺度特征对比，提升模型对细粒度语义的理解能力。**

- **链接: [http://arxiv.org/pdf/2506.01902v1](http://arxiv.org/pdf/2506.01902v1)**

> **作者:** Xinliu Zhong; Kayhan Batmanghelich; Li Sun
>
> **备注:** 6 pages, 1 figure, accepted by 2024 IEEE Conference on Artificial Intelligence (CAI)
>
> **摘要:** Vision-language models pre-trained on large scale of unlabeled biomedical images and associated reports learn generalizable semantic representations. These multi-modal representations can benefit various downstream tasks in the biomedical domain. Contrastive learning is widely used to pre-train vision-language models for general natural images and associated captions. Despite its popularity, we found biomedical texts have complex and domain-specific semantics that are often neglected by common contrastive methods. To address this issue, we propose a novel method, perturbed report discrimination, for pre-train biomedical vision-language models. First, we curate a set of text perturbation methods that keep the same words, but disrupt the semantic structure of the sentence. Next, we apply different types of perturbation to reports, and use the model to distinguish the original report from the perturbed ones given the associated image. Parallel to this, we enhance the sensitivity of our method to higher level of granularity for both modalities by contrasting attention-weighted image sub-regions and sub-words in the image-text pairs. We conduct extensive experiments on multiple downstream tasks, and our method outperforms strong baseline methods. The results demonstrate that our approach learns more semantic meaningful and robust multi-modal representations.
>
---
#### [new 038] Efficiency without Compromise: CLIP-aided Text-to-Image GANs with Increased Diversity
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于文本到图像生成任务，旨在解决大规模GAN训练成本高且生成多样性不足的问题。论文提出SCAD模型，结合预训练模型和双判别器机制，提升生成效果与多样性，同时降低训练成本。**

- **链接: [http://arxiv.org/pdf/2506.01493v1](http://arxiv.org/pdf/2506.01493v1)**

> **作者:** Yuya Kobayashi; Yuhta Takida; Takashi Shibuya; Yuki Mitsufuji
>
> **备注:** Accepted at IJCNN 2025
>
> **摘要:** Recently, Generative Adversarial Networks (GANs) have been successfully scaled to billion-scale large text-to-image datasets. However, training such models entails a high training cost, limiting some applications and research usage. To reduce the cost, one promising direction is the incorporation of pre-trained models. The existing method of utilizing pre-trained models for a generator significantly reduced the training cost compared with the other large-scale GANs, but we found the model loses the diversity of generation for a given prompt by a large margin. To build an efficient and high-fidelity text-to-image GAN without compromise, we propose to use two specialized discriminators with Slicing Adversarial Networks (SANs) adapted for text-to-image tasks. Our proposed model, called SCAD, shows a notable enhancement in diversity for a given prompt with better sample fidelity. We also propose to use a metric called Per-Prompt Diversity (PPD) to evaluate the diversity of text-to-image models quantitatively. SCAD achieved a zero-shot FID competitive with the latest large-scale GANs at two orders of magnitude less training cost.
>
---
#### [new 039] Efficient 3D Brain Tumor Segmentation with Axial-Coronal-Sagittal Embedding
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决脑肿瘤分割中训练耗时和预训练权重利用率低的问题。工作包括：引入轴状-冠状-矢状卷积、迁移ImageNet预训练权重至3D域、构建分类与分割联合模型，从而减少训练参数与周期，并提升分割效果。**

- **链接: [http://arxiv.org/pdf/2506.00434v1](http://arxiv.org/pdf/2506.00434v1)**

> **作者:** Tuan-Luc Huynh; Thanh-Danh Le; Tam V. Nguyen; Trung-Nghia Le; Minh-Triet Tran
>
> **备注:** Accepted by PSIVT 2023. Best paper award. Repo: https://github.com/LouisDo2108/ACS-nnU-Net
>
> **摘要:** In this paper, we address the crucial task of brain tumor segmentation in medical imaging and propose innovative approaches to enhance its performance. The current state-of-the-art nnU-Net has shown promising results but suffers from extensive training requirements and underutilization of pre-trained weights. To overcome these limitations, we integrate Axial-Coronal-Sagittal convolutions and pre-trained weights from ImageNet into the nnU-Net framework, resulting in reduced training epochs, reduced trainable parameters, and improved efficiency. Two strategies for transferring 2D pre-trained weights to the 3D domain are presented, ensuring the preservation of learned relationships and feature representations critical for effective information propagation. Furthermore, we explore a joint classification and segmentation model that leverages pre-trained encoders from a brain glioma grade classification proxy task, leading to enhanced segmentation performance, especially for challenging tumor labels. Experimental results demonstrate that our proposed methods in the fast training settings achieve comparable or even outperform the ensemble of cross-validation models, a common practice in the brain tumor segmentation literature.
>
---
#### [new 040] AuralSAM2: Enabling SAM2 Hear Through Pyramid Audio-Visual Feature Prompting
- **分类: cs.CV**

- **简介: 论文提出AuralSAM2，旨在解决视频中可听目标的分割任务。现有方法在融合音频与视觉特征时存在效率低或定位不准的问题。AuralSAM2通过引入AuralFuser模块和音频引导对比学习，实现多模态特征融合与更精准的语义对齐，提升SAM2在音视频场景中的分割性能。**

- **链接: [http://arxiv.org/pdf/2506.01015v1](http://arxiv.org/pdf/2506.01015v1)**

> **作者:** Yuyuan Liu; Yuanhong Chen; Chong Wang; Junlin Han; Junde Wu; Can Peng; Jingkun Chen; Yu Tian; Gustavo Carneiro
>
> **备注:** 18 pages, 18 Figures and 7 tables
>
> **摘要:** Segment Anything Model 2 (SAM2) exhibits strong generalisation for promptable segmentation in video clips; however, its integration with the audio modality remains underexplored. Existing approaches mainly follow two directions: (1) injecting adapters into the image encoder to receive audio signals, which incurs efficiency costs during prompt engineering, and (2) leveraging additional foundation models to generate visual prompts for the sounding objects, which are often imprecisely localised, leading to misguidance in SAM2. Moreover, these methods overlook the rich semantic interplay between hierarchical visual features and other modalities, resulting in suboptimal cross-modal fusion. In this work, we propose AuralSAM2, comprising the novel AuralFuser module, which externally attaches to SAM2 to integrate features from different modalities and generate feature-level prompts, guiding SAM2's decoder in segmenting sounding targets. Such integration is facilitated by a feature pyramid, further refining semantic understanding and enhancing object awareness in multimodal scenarios. Additionally, the audio-guided contrastive learning is introduced to explicitly align audio and visual representations and to also mitigate biases caused by dominant visual patterns. Results on public benchmarks show that our approach achieves remarkable improvements over the previous methods in the field. Code is available at https://github.com/yyliu01/AuralSAM2.
>
---
#### [new 041] OD3: Optimization-free Dataset Distillation for Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决大规模数据集训练计算资源消耗大的问题。通过提出OD3方法，无需优化即可合成紧凑数据集，在MS COCO和PASCAL VOC上实现高效训练，压缩比1.0%时mAP50超现有方法14%。**

- **链接: [http://arxiv.org/pdf/2506.01942v1](http://arxiv.org/pdf/2506.01942v1)**

> **作者:** Salwa K. Al Khatib; Ahmed ElHagry; Shitong Shao; Zhiqiang Shen
>
> **备注:** Equal Contribution of the first three authors
>
> **摘要:** Training large neural networks on large-scale datasets requires substantial computational resources, particularly for dense prediction tasks such as object detection. Although dataset distillation (DD) has been proposed to alleviate these demands by synthesizing compact datasets from larger ones, most existing work focuses solely on image classification, leaving the more complex detection setting largely unexplored. In this paper, we introduce OD3, a novel optimization-free data distillation framework specifically designed for object detection. Our approach involves two stages: first, a candidate selection process in which object instances are iteratively placed in synthesized images based on their suitable locations, and second, a candidate screening process using a pre-trained observer model to remove low-confidence objects. We perform our data synthesis framework on MS COCO and PASCAL VOC, two popular detection datasets, with compression ratios ranging from 0.25% to 5%. Compared to the prior solely existing dataset distillation method on detection and conventional core set selection methods, OD3 delivers superior accuracy, establishes new state-of-the-art results, surpassing prior best method by more than 14% on COCO mAP50 at a compression ratio of 1.0%. Code and condensed datasets are available at: https://github.com/VILA-Lab/OD3.
>
---
#### [new 042] Efficient Egocentric Action Recognition with Multimodal Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人机交互任务，旨在解决XR设备上实时动作识别的效率问题。通过分析RGB视频与3D手部姿态输入的采样频率对识别性能和计算资源的影响，发现降低RGB帧率并结合高频手部姿态数据可显著减少CPU使用，同时保持高准确率，为高效实时的EAR系统提供了可行方案。**

- **链接: [http://arxiv.org/pdf/2506.01757v1](http://arxiv.org/pdf/2506.01757v1)**

> **作者:** Marco Calzavara; Ard Kastrati; Matteo Macchini; Dushan Vasilevski; Roger Wattenhofer
>
> **备注:** Accepted as an extended abstract at the Second Joint Egocentric Vision (EgoVis) Workshop, 2025
>
> **摘要:** The increasing availability of wearable XR devices opens new perspectives for Egocentric Action Recognition (EAR) systems, which can provide deeper human understanding and situation awareness. However, deploying real-time algorithms on these devices can be challenging due to the inherent trade-offs between portability, battery life, and computational resources. In this work, we systematically analyze the impact of sampling frequency across different input modalities - RGB video and 3D hand pose - on egocentric action recognition performance and CPU usage. By exploring a range of configurations, we provide a comprehensive characterization of the trade-offs between accuracy and computational efficiency. Our findings reveal that reducing the sampling rate of RGB frames, when complemented with higher-frequency 3D hand pose input, can preserve high accuracy while significantly lowering CPU demands. Notably, we observe up to a 3x reduction in CPU usage with minimal to no loss in recognition performance. This highlights the potential of multimodal input strategies as a viable approach to achieving efficient, real-time EAR on XR devices.
>
---
#### [new 043] EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言导航（VLN）任务，旨在解决基于大语言模型（LLM）的导航代理在理解自然语言指令和视觉环境时推理能力不足、决策不可解释的问题。作者提出了EvolveNav框架，通过形式化思维链（CoT）微调和自反思后训练，提升模型的导航性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.01551v1](http://arxiv.org/pdf/2506.01551v1)**

> **作者:** Bingqian Lin; Yunshuang Nie; Khun Loun Zai; Ziming Wei; Mingfei Han; Rongtao Xu; Minzhe Niu; Jianhua Han; Liang Lin; Cewu Lu; Xiaodan Liang
>
> **摘要:** Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at https://github.com/expectorlin/EvolveNav.
>
---
#### [new 044] Chain-of-Frames: Advancing Video Understanding in Multimodal LLMs via Frame-Aware Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决多模态大语言模型（LLMs）在处理视频内容时推理能力不足和易产生幻觉的问题。作者构建了一个包含帧级推理链的数据集CoF-Data，并基于此微调视频LLMs，使其生成的推理过程能准确关联关键帧，从而提升模型在多个视频理解基准上的表现并降低幻觉率。**

- **链接: [http://arxiv.org/pdf/2506.00318v1](http://arxiv.org/pdf/2506.00318v1)**

> **作者:** Sara Ghazanfari; Francesco Croce; Nicolas Flammarion; Prashanth Krishnamurthy; Farshad Khorrami; Siddharth Garg
>
> **摘要:** Recent work has shown that eliciting Large Language Models (LLMs) to generate reasoning traces in natural language before answering the user's request can significantly improve their performance across tasks. This approach has been extended to multimodal LLMs, where the models can produce chain-of-thoughts (CoT) about the content of input images and videos. In this work, we propose to obtain video LLMs whose reasoning steps are grounded in, and explicitly refer to, the relevant video frames. For this, we first create CoF-Data, a large dataset of diverse questions, answers, and corresponding frame-grounded reasoning traces about both natural and synthetic videos, spanning various topics and tasks. Then, we fine-tune existing video LLMs on this chain-of-frames (CoF) data. Our approach is simple and self-contained, and, unlike existing approaches for video CoT, does not require auxiliary networks to select or caption relevant frames. We show that our models based on CoF are able to generate chain-of-thoughts that accurately refer to the key frames to answer the given question. This, in turn, leads to improved performance across multiple video understanding benchmarks, for example, surpassing leading video LLMs on Video-MME, MVBench, and VSI-Bench, and notably reducing the hallucination rate. Code available at https://github.com/SaraGhazanfari/CoF}{github.com/SaraGhazanfari/CoF.
>
---
#### [new 045] MLLMs Need 3D-Aware Representation Supervision for Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于多模态场景理解任务，旨在解决MLLMs缺乏3D感知表示能力的问题。通过引入预训练3D模型的监督，提出3DRS框架，提升MLLM在视觉基础、描述生成和问答等下游任务中的表现。**

- **链接: [http://arxiv.org/pdf/2506.01946v1](http://arxiv.org/pdf/2506.01946v1)**

> **作者:** Xiaohu Huang; Jingjing Wu; Qunyi Xie; Kai Han
>
> **摘要:** Recent advances in scene understanding have leveraged multimodal large language models (MLLMs) for 3D reasoning by capitalizing on their strong 2D pretraining. However, the lack of explicit 3D data during MLLM pretraining limits 3D representation capability. In this paper, we investigate the 3D-awareness of MLLMs by evaluating multi-view correspondence and reveal a strong positive correlation between the quality of 3D-aware representation and downstream task performance. Motivated by this, we propose 3DRS, a framework that enhances MLLM 3D representation learning by introducing supervision from pretrained 3D foundation models. Our approach aligns MLLM visual features with rich 3D knowledge distilled from 3D models, effectively improving scene understanding. Extensive experiments across multiple benchmarks and MLLMs -- including visual grounding, captioning, and question answering -- demonstrate consistent performance gains. Project page: https://visual-ai.github.io/3drs
>
---
#### [new 046] SEED: A Benchmark Dataset for Sequential Facial Attribute Editing with Diffusion Models
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于图像编辑与检测任务，旨在解决扩散模型下连续面部属性编辑的追踪与检测难题。作者构建了大规模数据集SEED，并提出FAITH模型以增强对细微编辑序列的敏感性，推动编辑追踪与真实性分析研究。**

- **链接: [http://arxiv.org/pdf/2506.00562v1](http://arxiv.org/pdf/2506.00562v1)**

> **作者:** Yule Zhu; Ping Liu; Zhedong Zheng; Wei Liu
>
> **摘要:** Diffusion models have recently enabled precise and photorealistic facial editing across a wide range of semantic attributes. Beyond single-step modifications, a growing class of applications now demands the ability to analyze and track sequences of progressive edits, such as stepwise changes to hair, makeup, or accessories. However, sequential editing introduces significant challenges in edit attribution and detection robustness, further complicated by the lack of large-scale, finely annotated benchmarks tailored explicitly for this task. We introduce SEED, a large-scale Sequentially Edited facE Dataset constructed via state-of-the-art diffusion models. SEED contains over 90,000 facial images with one to four sequential attribute modifications, generated using diverse diffusion-based editing pipelines (LEdits, SDXL, SD3). Each image is annotated with detailed edit sequences, attribute masks, and prompts, facilitating research on sequential edit tracking, visual provenance analysis, and manipulation robustness assessment. To benchmark this task, we propose FAITH, a frequency-aware transformer-based model that incorporates high-frequency cues to enhance sensitivity to subtle sequential changes. Comprehensive experiments, including systematic comparisons of multiple frequency-domain methods, demonstrate the effectiveness of FAITH and the unique challenges posed by SEED. SEED offers a challenging and flexible resource for studying progressive diffusion-based edits at scale. Dataset and code will be publicly released at: https://github.com/Zeus1037/SEED.
>
---
#### [new 047] Deep Temporal Reasoning in Video Language Models: A Cross-Linguistic Evaluation of Action Duration and Completion through Perfect Times
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频语言模型的时空推理任务，旨在解决模型对动作持续与完成的理解问题。作者构建了四语种的“Perfect Times”数据集，通过视频与问题配对，检验模型是否真正理解事件的时间动态，而非依赖表面特征。实验表明现有模型在该任务上表现不佳，强调需融合深度多模态线索以提升时序推理能力。**

- **链接: [http://arxiv.org/pdf/2506.00928v1](http://arxiv.org/pdf/2506.00928v1)**

> **作者:** Olga Loginova; Sofía Ortega Loguinova
>
> **摘要:** Human perception of events is intrinsically tied to distinguishing between completed (perfect and telic) and ongoing (durative) actions, a process mediated by both linguistic structure and visual cues. In this work, we introduce the \textbf{Perfect Times} dataset, a novel, quadrilingual (English, Italian, Russian, and Japanese) multiple-choice question-answering benchmark designed to assess video-language models (VLMs) on temporal reasoning. By pairing everyday activity videos with event completion labels and perfectivity-tailored distractors, our dataset probes whether models truly comprehend temporal dynamics or merely latch onto superficial markers. Experimental results indicate that state-of-the-art models, despite their success on text-based tasks, struggle to mirror human-like temporal and causal reasoning grounded in video. This study underscores the necessity of integrating deep multimodal cues to capture the nuances of action duration and completion within temporal and causal video dynamics, setting a new standard for evaluating and advancing temporal reasoning in VLMs.
>
---
#### [new 048] Improving Keystep Recognition in Ego-Video via Dexterous Focus
- **分类: cs.CV**

- **简介: 该论文属于细粒度动作识别任务，旨在解决第一视角视频中因头部运动导致的动作识别难题。作者提出一种新框架，通过稳定视频并聚焦手部区域，在不改变模型结构的前提下，提升了Ego-Exo4D数据集上的关键步骤识别效果。**

- **链接: [http://arxiv.org/pdf/2506.00827v1](http://arxiv.org/pdf/2506.00827v1)**

> **作者:** Zachary Chavis; Stephen J. Guy; Hyun Soo Park
>
> **摘要:** In this paper, we address the challenge of understanding human activities from an egocentric perspective. Traditional activity recognition techniques face unique challenges in egocentric videos due to the highly dynamic nature of the head during many activities. We propose a framework that seeks to address these challenges in a way that is independent of network architecture by restricting the ego-video input to a stabilized, hand-focused video. We demonstrate that this straightforward video transformation alone outperforms existing egocentric video baselines on the Ego-Exo4D Fine-Grained Keystep Recognition benchmark without requiring any alteration of the underlying model infrastructure.
>
---
#### [new 049] NavBench: Probing Multimodal Large Language Models for Embodied Navigation
- **分类: cs.CV**

- **简介: 论文提出NavBench，评估多模态大语言模型在具身导航中的能力，包含理解与执行任务。旨在解决零样本下模型对环境理解与行动能力不足的问题，通过认知任务与真实场景测试，探索模型性能与挑战。**

- **链接: [http://arxiv.org/pdf/2506.01031v1](http://arxiv.org/pdf/2506.01031v1)**

> **作者:** Yanyuan Qiao; Haodong Hong; Wenqi Lyu; Dong An; Siqi Zhang; Yutong Xie; Xinyu Wang; Qi Wu
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated strong generalization in vision-language tasks, yet their ability to understand and act within embodied environments remains underexplored. We present NavBench, a benchmark to evaluate the embodied navigation capabilities of MLLMs under zero-shot settings. NavBench consists of two components: (1) navigation comprehension, assessed through three cognitively grounded tasks including global instruction alignment, temporal progress estimation, and local observation-action reasoning, covering 3,200 question-answer pairs; and (2) step-by-step execution in 432 episodes across 72 indoor scenes, stratified by spatial, cognitive, and execution complexity. To support real-world deployment, we introduce a pipeline that converts MLLMs' outputs into robotic actions. We evaluate both proprietary and open-source models, finding that GPT-4o performs well across tasks, while lighter open-source models succeed in simpler cases. Results also show that models with higher comprehension scores tend to achieve better execution performance. Providing map-based context improves decision accuracy, especially in medium-difficulty scenarios. However, most models struggle with temporal understanding, particularly in estimating progress during navigation, which may pose a key challenge.
>
---
#### [new 050] DiffuseSlide: Training-Free High Frame Rate Video Generation Diffusion
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决高帧率视频生成中的闪烁和质量退化问题。作者提出DiffuseSlide，一种无需训练的方法，利用预训练扩散模型，通过噪声重注入和滑动窗口去噪技术，生成高质量、高帧率视频，提升了时间一致性和空间细节。**

- **链接: [http://arxiv.org/pdf/2506.01454v1](http://arxiv.org/pdf/2506.01454v1)**

> **作者:** Geunmin Hwang; Hyun-kyu Ko; Younghyun Kim; Seungryong Lee; Eunbyung Park
>
> **摘要:** Recent advancements in diffusion models have revolutionized video generation, enabling the creation of high-quality, temporally consistent videos. However, generating high frame-rate (FPS) videos remains a significant challenge due to issues such as flickering and degradation in long sequences, particularly in fast-motion scenarios. Existing methods often suffer from computational inefficiencies and limitations in maintaining video quality over extended frames. In this paper, we present a novel, training-free approach for high FPS video generation using pre-trained diffusion models. Our method, DiffuseSlide, introduces a new pipeline that leverages key frames from low FPS videos and applies innovative techniques, including noise re-injection and sliding window latent denoising, to achieve smooth, consistent video outputs without the need for additional fine-tuning. Through extensive experiments, we demonstrate that our approach significantly improves video quality, offering enhanced temporal coherence and spatial fidelity. The proposed method is not only computationally efficient but also adaptable to various video generation tasks, making it ideal for applications such as virtual reality, video games, and high-quality content creation.
>
---
#### [new 051] TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决细粒度动物图像生成中形态和身份准确性低的问题。作者提出了TaxaDiffusion框架，利用分类学知识，通过多级渐进训练扩散模型，从纲目到科属再到物种层级逐步细化特征，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.01923v1](http://arxiv.org/pdf/2506.01923v1)**

> **作者:** Amin Karimi Monsefi; Mridul Khurana; Rajiv Ramnath; Anuj Karpatne; Wei-Lun Chao; Cheng Zhang
>
> **摘要:** We propose TaxaDiffusion, a taxonomy-informed training framework for diffusion models to generate fine-grained animal images with high morphological and identity accuracy. Unlike standard approaches that treat each species as an independent category, TaxaDiffusion incorporates domain knowledge that many species exhibit strong visual similarities, with distinctions often residing in subtle variations of shape, pattern, and color. To exploit these relationships, TaxaDiffusion progressively trains conditioned diffusion models across different taxonomic levels -- starting from broad classifications such as Class and Order, refining through Family and Genus, and ultimately distinguishing at the Species level. This hierarchical learning strategy first captures coarse-grained morphological traits shared by species with common ancestors, facilitating knowledge transfer before refining fine-grained differences for species-level distinction. As a result, TaxaDiffusion enables accurate generation even with limited training samples per species. Extensive experiments on three fine-grained animal datasets demonstrate that outperforms existing approaches, achieving superior fidelity in fine-grained animal image generation. Project page: https://amink8.github.io/TaxaDiffusion/
>
---
#### [new 052] VRD-IU: Lessons from Visually Rich Document Intelligence and Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文档智能任务，旨在解决复杂表单类文档中关键信息提取与定位的问题。通过VRD-IU竞赛，评估并总结了多模态方法、层级分解和先进检测技术等方案的效果，推动了视觉丰富文档理解的技术发展。**

- **链接: [http://arxiv.org/pdf/2506.01388v1](http://arxiv.org/pdf/2506.01388v1)**

> **作者:** Yihao Ding; Soyeon Caren Han; Yan Li; Josiah Poon
>
> **备注:** Accepted at IJCAI 2025 Demonstrations Track
>
> **摘要:** Visually Rich Document Understanding (VRDU) has emerged as a critical field in document intelligence, enabling automated extraction of key information from complex documents across domains such as medical, financial, and educational applications. However, form-like documents pose unique challenges due to their complex layouts, multi-stakeholder involvement, and high structural variability. Addressing these issues, the VRD-IU Competition was introduced, focusing on extracting and localizing key information from multi-format forms within the Form-NLU dataset, which includes digital, printed, and handwritten documents. This paper presents insights from the competition, which featured two tracks: Track A, emphasizing entity-based key information retrieval, and Track B, targeting end-to-end key information localization from raw document images. With over 20 participating teams, the competition showcased various state-of-the-art methodologies, including hierarchical decomposition, transformer-based retrieval, multimodal feature fusion, and advanced object detection techniques. The top-performing models set new benchmarks in VRDU, providing valuable insights into document intelligence.
>
---
#### [new 053] Unlocking Aha Moments via Reinforcement Learning: Advancing Collaborative Visual Comprehension and Generation
- **分类: cs.CV**

- **简介: 该论文属于多模态生成任务，旨在解决视觉理解和生成能力分离的问题。通过监督微调和强化学习，使模型具备协同进化能力，实现图文生成的统一。**

- **链接: [http://arxiv.org/pdf/2506.01480v1](http://arxiv.org/pdf/2506.01480v1)**

> **作者:** Kaihang Pan; Yang Wu; Wendong Bu; Kai Shen; Juncheng Li; Yingting Wang; Yunfei Li; Siliang Tang; Jun Xiao; Fei Wu; Hang Zhao; Yueting Zhuang
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** Recent endeavors in Multimodal Large Language Models (MLLMs) aim to unify visual comprehension and generation. However, these two capabilities remain largely independent, as if they are two separate functions encapsulated within the same model. Consequently, visual comprehension does not enhance visual generation, and the reasoning mechanisms of LLMs have not been fully integrated to revolutionize image generation. In this paper, we propose to enable the collaborative co-evolution of visual comprehension and generation, advancing image generation into an iterative introspective process. We introduce a two-stage training approach: supervised fine-tuning teaches the MLLM with the foundational ability to generate genuine CoT for visual generation, while reinforcement learning activates its full potential via an exploration-exploitation trade-off. Ultimately, we unlock the Aha moment in visual generation, advancing MLLMs from text-to-image tasks to unified image generation. Extensive experiments demonstrate that our model not only excels in text-to-image generation and image editing, but also functions as a superior image semantic evaluator with enhanced visual comprehension capabilities. Project Page: https://janus-pro-r1.github.io.
>
---
#### [new 054] FlowMo: Variance-Based Flow Guidance for Coherent Motion in Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决文本到视频扩散模型中运动连贯性不足的问题。作者提出了FlowMo方法，通过分析预训练模型的隐空间时间表示，动态减少帧间方差，提升生成视频的运动一致性，且无需额外训练或输入。**

- **链接: [http://arxiv.org/pdf/2506.01144v1](http://arxiv.org/pdf/2506.01144v1)**

> **作者:** Ariel Shaulov; Itay Hazan; Lior Wolf; Hila Chefer
>
> **摘要:** Text-to-video diffusion models are notoriously limited in their ability to model temporal aspects such as motion, physics, and dynamic interactions. Existing approaches address this limitation by retraining the model or introducing external conditioning signals to enforce temporal consistency. In this work, we explore whether a meaningful temporal representation can be extracted directly from the predictions of a pre-trained model without any additional training or auxiliary inputs. We introduce \textbf{FlowMo}, a novel training-free guidance method that enhances motion coherence using only the model's own predictions in each diffusion step. FlowMo first derives an appearance-debiased temporal representation by measuring the distance between latents corresponding to consecutive frames. This highlights the implicit temporal structure predicted by the model. It then estimates motion coherence by measuring the patch-wise variance across the temporal dimension and guides the model to reduce this variance dynamically during sampling. Extensive experiments across multiple text-to-video models demonstrate that FlowMo significantly improves motion coherence without sacrificing visual quality or prompt alignment, offering an effective plug-and-play solution for enhancing the temporal fidelity of pre-trained video diffusion models.
>
---
#### [new 055] Sequence-Based Identification of First-Person Camera Wearers in Third-Person Views
- **分类: cs.CV**

- **简介: 该论文属于多视角视觉任务，旨在解决第一人称摄像头佩戴者在第三人称视角中的识别问题。通过构建同步第一-第三人称数据集TF2025，并结合运动线索与人员重识别技术，实现序列化识别方法。**

- **链接: [http://arxiv.org/pdf/2506.00394v1](http://arxiv.org/pdf/2506.00394v1)**

> **作者:** Ziwei Zhao; Xizi Wang; Yuchen Wang; Feng Cheng; David Crandall
>
> **摘要:** The increasing popularity of egocentric cameras has generated growing interest in studying multi-camera interactions in shared environments. Although large-scale datasets such as Ego4D and Ego-Exo4D have propelled egocentric vision research, interactions between multiple camera wearers remain underexplored-a key gap for applications like immersive learning and collaborative robotics. To bridge this, we present TF2025, an expanded dataset with synchronized first- and third-person views. In addition, we introduce a sequence-based method to identify first-person wearers in third-person footage, combining motion cues and person re-identification.
>
---
#### [new 056] Seg2Any: Open-set Segmentation-Mask-to-Image Generation with Precise Shape and Semantic Control
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决现有模型在空间布局控制上的不足。通过提出Seg2Any框架，结合区域语义与高频形状信息，并引入属性隔离机制，实现更精确的分割掩码到图像生成。**

- **链接: [http://arxiv.org/pdf/2506.00596v1](http://arxiv.org/pdf/2506.00596v1)**

> **作者:** Danfeng li; Hui Zhang; Sheng Wang; Jiacheng Li; Zuxuan Wu
>
> **摘要:** Despite recent advances in diffusion models, top-tier text-to-image (T2I) models still struggle to achieve precise spatial layout control, i.e. accurately generating entities with specified attributes and locations. Segmentation-mask-to-image (S2I) generation has emerged as a promising solution by incorporating pixel-level spatial guidance and regional text prompts. However, existing S2I methods fail to simultaneously ensure semantic consistency and shape consistency. To address these challenges, we propose Seg2Any, a novel S2I framework built upon advanced multimodal diffusion transformers (e.g. FLUX). First, to achieve both semantic and shape consistency, we decouple segmentation mask conditions into regional semantic and high-frequency shape components. The regional semantic condition is introduced by a Semantic Alignment Attention Mask, ensuring that generated entities adhere to their assigned text prompts. The high-frequency shape condition, representing entity boundaries, is encoded as an Entity Contour Map and then introduced as an additional modality via multi-modal attention to guide image spatial structure. Second, to prevent attribute leakage across entities in multi-entity scenarios, we introduce an Attribute Isolation Attention Mask mechanism, which constrains each entity's image tokens to attend exclusively to themselves during image self-attention. To support open-set S2I generation, we construct SACap-1M, a large-scale dataset containing 1 million images with 5.9 million segmented entities and detailed regional captions, along with a SACap-Eval benchmark for comprehensive S2I evaluation. Extensive experiments demonstrate that Seg2Any achieves state-of-the-art performance on both open-set and closed-set S2I benchmarks, particularly in fine-grained spatial and attribute control of entities.
>
---
#### [new 057] Towards Edge-Based Idle State Detection in Construction Machinery Using Surveillance Cameras
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于设备状态监测任务，旨在解决施工机械闲置状态检测问题。通过构建Edge-IMI框架，在边缘设备上实现基于监控视频的对象检测、跟踪与闲置识别，减少对云端依赖和硬件成本，并验证了其在实际场景中的有效性与实时性。**

- **链接: [http://arxiv.org/pdf/2506.00904v1](http://arxiv.org/pdf/2506.00904v1)**

> **作者:** Xander Küpers; Jeroen Klein Brinke; Rob Bemthuis; Ozlem Durmaz Incel
>
> **备注:** 18 pages, 6 figures, 3 tables; to appear in Intelligent Systems and Applications, Lecture Notes in Networks and Systems (LNNS), Springer, 2025. Part of the 11th Intelligent Systems Conference (IntelliSys 2025), 28-29 August 2025, Amsterdam, The Netherlands
>
> **摘要:** The construction industry faces significant challenges in optimizing equipment utilization, as underused machinery leads to increased operational costs and project delays. Accurate and timely monitoring of equipment activity is therefore key to identifying idle periods and improving overall efficiency. This paper presents the Edge-IMI framework for detecting idle construction machinery, specifically designed for integration with surveillance camera systems. The proposed solution consists of three components: object detection, tracking, and idle state identification, which are tailored for execution on resource-constrained, CPU-based edge computing devices. The performance of Edge-IMI is evaluated using a combined dataset derived from the ACID and MOCS benchmarks. Experimental results confirm that the object detector achieves an F1 score of 71.75%, indicating robust real-world detection capabilities. The logistic regression-based idle identification module reliably distinguishes between active and idle machinery with minimal false positives. Integrating all three modules, Edge-IMI enables efficient on-site inference, reducing reliance on high-bandwidth cloud services and costly hardware accelerators. We also evaluate the performance of object detection models on Raspberry Pi 5 and an Intel NUC platforms, as example edge computing platforms. We assess the feasibility of real-time processing and the impact of model optimization techniques.
>
---
#### [new 058] Synthetic Data Augmentation using Pre-trained Diffusion Models for Long-tailed Food Image Classification
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决食品图像数据分布长尾导致的类别不平衡问题。通过使用预训练扩散模型进行合成数据增强，提出两阶段框架，结合正负提示生成多样且区分度高的样本，提升分类性能。实验验证了方法在多个数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2506.01368v1](http://arxiv.org/pdf/2506.01368v1)**

> **作者:** GaYeon Koh; Hyun-Jic Oh; Jeonghyun Noh; Won-Ki Jeong
>
> **备注:** 10 pages
>
> **摘要:** Deep learning-based food image classification enables precise identification of food categories, further facilitating accurate nutritional analysis. However, real-world food images often show a skewed distribution, with some food types being more prevalent than others. This class imbalance can be problematic, causing models to favor the majority (head) classes with overall performance degradation for the less common (tail) classes. Recently, synthetic data augmentation using diffusion-based generative models has emerged as a promising solution to address this issue. By generating high-quality synthetic images, these models can help uniformize the data distribution, potentially improving classification performance. However, existing approaches face challenges: fine-tuning-based methods need a uniformly distributed dataset, while pre-trained model-based approaches often overlook inter-class separation in synthetic data. In this paper, we propose a two-stage synthetic data augmentation framework, leveraging pre-trained diffusion models for long-tailed food classification. We generate a reference set conditioned by a positive prompt on the generation target and then select a class that shares similar features with the generation target as a negative prompt. Subsequently, we generate a synthetic augmentation set using positive and negative prompt conditions by a combined sampling strategy that promotes intra-class diversity and inter-class separation. We demonstrate the efficacy of the proposed method on two long-tailed food benchmark datasets, achieving superior performance compared to previous works in terms of top-1 accuracy.
>
---
#### [new 059] MOOSE: Pay Attention to Temporal Dynamics for Video Understanding via Optical Flows
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决建模时间动态信息效率低和可解释性差的问题。作者提出MOOSE模型，融合光学流与空间嵌入，利用预训练编码器提升性能。实现了高效、可解释的时序建模，并在多个数据集上达到先进水平。**

- **链接: [http://arxiv.org/pdf/2506.01119v1](http://arxiv.org/pdf/2506.01119v1)**

> **作者:** Hong Nguyen; Dung Tran; Hieu Hoang; Phong Nguyen; Shrikanth Narayanan
>
> **摘要:** Many motion-centric video analysis tasks, such as atomic actions, detecting atypical motor behavior in individuals with autism, or analyzing articulatory motion in real-time MRI of human speech, require efficient and interpretable temporal modeling. Capturing temporal dynamics is a central challenge in video analysis, often requiring significant computational resources and fine-grained annotations that are not widely available. This paper presents MOOSE (Motion Flow Over Spatial Space), a novel temporally-centric video encoder explicitly integrating optical flow with spatial embeddings to model temporal information efficiently, inspired by human perception of motion. Unlike prior models, MOOSE takes advantage of rich, widely available pre-trained visual and optical flow encoders instead of training video models from scratch. This significantly reduces computational complexity while enhancing temporal interpretability. Our primary contributions includes (1) proposing a computationally efficient temporally-centric architecture for video understanding (2) demonstrating enhanced interpretability in modeling temporal dynamics; and (3) achieving state-of-the-art performance on diverse benchmarks, including clinical, medical, and standard action recognition datasets, confirming the broad applicability and effectiveness of our approach.
>
---
#### [new 060] Visual Explanation via Similar Feature Activation for Metric Learning
- **分类: cs.CV**

- **简介: 该论文属于图像识别中的模型解释任务，旨在解决现有可视化方法无法用于度量学习模型的问题。作者提出了一种名为相似特征激活图（SFAM）的新方法，通过计算通道重要性分数来生成视觉解释图。实验表明，SFAM在使用欧氏距离或余弦相似度的CNN模型上具有良好的可解释性和应用前景。**

- **链接: [http://arxiv.org/pdf/2506.01636v1](http://arxiv.org/pdf/2506.01636v1)**

> **作者:** Yi Liao; Ugochukwu Ejike Akpudo; Jue Zhang; Yongsheng Gao; Jun Zhou; Wenyi Zeng; Weichuan Zhang
>
> **摘要:** Visual explanation maps enhance the trustworthiness of decisions made by deep learning models and offer valuable guidance for developing new algorithms in image recognition tasks. Class activation maps (CAM) and their variants (e.g., Grad-CAM and Relevance-CAM) have been extensively employed to explore the interpretability of softmax-based convolutional neural networks, which require a fully connected layer as the classifier for decision-making. However, these methods cannot be directly applied to metric learning models, as such models lack a fully connected layer functioning as a classifier. To address this limitation, we propose a novel visual explanation method termed Similar Feature Activation Map (SFAM). This method introduces the channel-wise contribution importance score (CIS) to measure feature importance, derived from the similarity measurement between two image embeddings. The explanation map is constructed by linearly combining the proposed importance weights with the feature map from a CNN model. Quantitative and qualitative experiments show that SFAM provides highly promising interpretable visual explanations for CNN models using Euclidean distance or cosine similarity as the similarity metric.
>
---
#### [new 061] MoDA: Modulation Adapter for Fine-Grained Visual Grounding in Instructional MLLMs
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属于多模态视觉-语言任务，旨在解决现有模型在复杂场景中难以精细理解视觉概念的问题。作者提出MoDA模块，通过指令引导的调制机制优化视觉特征表示，提升视觉基础能力。实验表明其能增强图像驱动的MLLMs生成更合适的上下文响应。**

- **链接: [http://arxiv.org/pdf/2506.01850v1](http://arxiv.org/pdf/2506.01850v1)**

> **作者:** Wayner Barrios; Andrés Villa; Juan León Alcázar; SouYoung Jin; Bernard Ghanem
>
> **摘要:** Recently, Multimodal Large Language Models (MLLMs) have demonstrated impressive performance on instruction-following tasks by integrating pretrained visual encoders with large language models (LLMs). However, existing approaches often struggle to ground fine-grained visual concepts in complex scenes. In this paper, we propose MoDA (Modulation Adapter), a lightweight yet effective module designed to refine pre-aligned visual features through instruction-guided modulation. Our approach follows the standard LLaVA training protocol, consisting of a two-stage process: (1) aligning image features to the LLMs input space via a frozen vision encoder and adapter layers, and (2) refining those features using the MoDA adapter during the instructional tuning stage. MoDA employs a Transformer-based cross-attention mechanism to generate a modulation mask over the aligned visual tokens, thereby emphasizing semantically relevant embedding dimensions based on the language instruction. The modulated features are then passed to the LLM for autoregressive language generation. Our experimental evaluation shows that MoDA improves visual grounding and generates more contextually appropriate responses, demonstrating its effectiveness as a general-purpose enhancement for image-based MLLMs.
>
---
#### [new 062] Efficient Endangered Deer Species Monitoring with UAV Aerial Imagery and Deep Learning
- **分类: cs.CV**

- **简介: 该论文属于野生动物监测任务，旨在解决传统人工识别濒危鹿种效率低的问题。作者利用无人机航拍图像和深度学习技术，开发基于YOLO框架的算法，在阿根廷两个项目中自动识别沼泽鹿和潘帕斯鹿，取得了良好效果，提升了野生动物监测与管理效率。**

- **链接: [http://arxiv.org/pdf/2506.00164v1](http://arxiv.org/pdf/2506.00164v1)**

> **作者:** Agustín Roca; Gabriel Torre; Juan I. Giribet; Gastón Castro; Leonardo Colombo; Ignacio Mas; Javier Pereira
>
> **摘要:** This paper examines the use of Unmanned Aerial Vehicles (UAVs) and deep learning for detecting endangered deer species in their natural habitats. As traditional identification processes require trained manual labor that can be costly in resources and time, there is a need for more efficient solutions. Leveraging high-resolution aerial imagery, advanced computer vision techniques are applied to automate the identification process of deer across two distinct projects in Buenos Aires, Argentina. The first project, Pantano Project, involves the marsh deer in the Paran\'a Delta, while the second, WiMoBo, focuses on the Pampas deer in Campos del Tuy\'u National Park. A tailored algorithm was developed using the YOLO framework, trained on extensive datasets compiled from UAV-captured images. The findings demonstrate that the algorithm effectively identifies marsh deer with a high degree of accuracy and provides initial insights into its applicability to Pampas deer, albeit with noted limitations. This study not only supports ongoing conservation efforts but also highlights the potential of integrating AI with UAV technology to enhance wildlife monitoring and management practices.
>
---
#### [new 063] Dual-Process Image Generation
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决现有方法难以快速学习新控制任务的问题。作者提出一种双过程蒸馏方法，利用视觉语言模型（VLM）指导图像生成器学习多模态控制，实现对颜色、线条、深度等属性的快速调节。**

- **链接: [http://arxiv.org/pdf/2506.01955v1](http://arxiv.org/pdf/2506.01955v1)**

> **作者:** Grace Luo; Jonathan Granskog; Aleksander Holynski; Trevor Darrell
>
> **摘要:** Prior methods for controlling image generation are limited in their ability to be taught new tasks. In contrast, vision-language models, or VLMs, can learn tasks in-context and produce the correct outputs for a given input. We propose a dual-process distillation scheme that allows feed-forward image generators to learn new tasks from deliberative VLMs. Our scheme uses a VLM to rate the generated images and backpropagates this gradient to update the weights of the image generator. Our general framework enables a wide variety of new control tasks through the same text-and-image based interface. We showcase a handful of applications of this technique for different types of control signals, such as commonsense inferences and visual prompts. With our method, users can implement multimodal controls for properties such as color palette, line weight, horizon position, and relative depth within a matter of minutes. Project page: https://dual-process.github.io.
>
---
#### [new 064] A Large Convolutional Neural Network for Clinical Target and Multi-organ Segmentation in Gynecologic Brachytherapy with Multi-stage Learning
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决宫颈癌放疗中临床靶区和多器官分割的难题。作者提出了GynBTNet模型，通过自监督预训练和分阶段微调策略，提升了分割精度，尤其在复杂结构上表现优异，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.01073v1](http://arxiv.org/pdf/2506.01073v1)**

> **作者:** Mingzhe Hu; Yuan Gao; Yuheng Li; Ricahrd LJ Qiu; Chih-Wei Chang; Keyur D. Shah; Priyanka Kapoor; Beth Bradshaw; Yuan Shao; Justin Roper; Jill Remick; Zhen Tian; Xiaofeng Yang
>
> **摘要:** Purpose: Accurate segmentation of clinical target volumes (CTV) and organs-at-risk is crucial for optimizing gynecologic brachytherapy (GYN-BT) treatment planning. However, anatomical variability, low soft-tissue contrast in CT imaging, and limited annotated datasets pose significant challenges. This study presents GynBTNet, a novel multi-stage learning framework designed to enhance segmentation performance through self-supervised pretraining and hierarchical fine-tuning strategies. Methods: GynBTNet employs a three-stage training strategy: (1) self-supervised pretraining on large-scale CT datasets using sparse submanifold convolution to capture robust anatomical representations, (2) supervised fine-tuning on a comprehensive multi-organ segmentation dataset to refine feature extraction, and (3) task-specific fine-tuning on a dedicated GYN-BT dataset to optimize segmentation performance for clinical applications. The model was evaluated against state-of-the-art methods using the Dice Similarity Coefficient (DSC), 95th percentile Hausdorff Distance (HD95), and Average Surface Distance (ASD). Results: Our GynBTNet achieved superior segmentation performance, significantly outperforming nnU-Net and Swin-UNETR. Notably, it yielded a DSC of 0.837 +/- 0.068 for CTV, 0.940 +/- 0.052 for the bladder, 0.842 +/- 0.070 for the rectum, and 0.871 +/- 0.047 for the uterus, with reduced HD95 and ASD compared to baseline models. Self-supervised pretraining led to consistent performance improvements, particularly for structures with complex boundaries. However, segmentation of the sigmoid colon remained challenging, likely due to anatomical ambiguities and inter-patient variability. Statistical significance analysis confirmed that GynBTNet's improvements were significant compared to baseline models.
>
---
#### [new 065] FaceCoT: A Benchmark Dataset for Face Anti-Spoofing with Chain-of-Thought Reasoning
- **分类: cs.CV**

- **简介: 该论文属于人脸识别安全任务，旨在解决传统方法在设备、环境和攻击类型上泛化能力有限的问题。作者构建了首个用于面部反欺骗的大规模视觉问答数据集FaceCoT，并提出CoT增强渐进学习策略，结合多模态大模型提升鲁棒性和可解释性。**

- **链接: [http://arxiv.org/pdf/2506.01783v1](http://arxiv.org/pdf/2506.01783v1)**

> **作者:** Honglu Zhang; Zhiqin Fang; Ningning Zhao; Saihui Hou; Long Ma; Renwang Pei; Zhaofeng He
>
> **摘要:** Face Anti-Spoofing (FAS) typically depends on a single visual modality when defending against presentation attacks such as print attacks, screen replays, and 3D masks, resulting in limited generalization across devices, environments, and attack types. Meanwhile, Multimodal Large Language Models (MLLMs) have recently achieved breakthroughs in image-text understanding and semantic reasoning, suggesting that integrating visual and linguistic co-inference into FAS can substantially improve both robustness and interpretability. However, the lack of a high-quality vision-language multimodal dataset has been a critical bottleneck. To address this, we introduce FaceCoT (Face Chain-of-Thought), the first large-scale Visual Question Answering (VQA) dataset tailored for FAS. FaceCoT covers 14 spoofing attack types and enriches model learning with high-quality CoT VQA annotations. Meanwhile, we develop a caption model refined via reinforcement learning to expand the dataset and enhance annotation quality. Furthermore, we introduce a CoT-Enhanced Progressive Learning (CEPL) strategy to better leverage the CoT data and boost model performance on FAS tasks. Extensive experiments demonstrate that models trained with FaceCoT and CEPL outperform state-of-the-art methods on multiple benchmark datasets.
>
---
#### [new 066] ViVo: A Dataset for Volumetric VideoReconstruction and Compression
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与图形学任务，旨在解决现有数据集在神经体积视频重建与压缩中的局限性。作者提出了一个新数据集ViVo，包含多样化的现实场景和丰富的语义及低级特征，并提供多视角RGB、深度视频及相关标注。通过基准测试展示了其挑战性，强调需更有效的算法来提升重建与压缩性能。**

- **链接: [http://arxiv.org/pdf/2506.00558v1](http://arxiv.org/pdf/2506.00558v1)**

> **作者:** Adrian Azzarelli; Ge Gao; Ho Man Kwan; Fan Zhang; Nantheera Anantrasirichai; Ollie Moolan-Feroze; David Bull
>
> **摘要:** As research on neural volumetric video reconstruction and compression flourishes, there is a need for diverse and realistic datasets, which can be used to develop and validate reconstruction and compression models. However, existing volumetric video datasets lack diverse content in terms of both semantic and low-level features that are commonly present in real-world production pipelines. In this context, we propose a new dataset, ViVo, for VolumetrIc VideO reconstruction and compression. The dataset is faithful to real-world volumetric video production and is the first dataset to extend the definition of diversity to include both human-centric characteristics (skin, hair, etc.) and dynamic visual phenomena (transparent, reflective, liquid, etc.). Each video sequence in this database contains raw data including fourteen multi-view RGB and depth video pairs, synchronized at 30FPS with per-frame calibration and audio data, and their associated 2-D foreground masks and 3-D point clouds. To demonstrate the use of this database, we have benchmarked three state-of-the-art (SotA) 3-D reconstruction methods and two volumetric video compression algorithms. The obtained results evidence the challenging nature of the proposed dataset and the limitations of existing datasets for both volumetric video reconstruction and compression tasks, highlighting the need to develop more effective algorithms for these applications. The database and the associated results are available at https://vivo-bvicr.github.io/
>
---
#### [new 067] Rethinking Image Histogram Matching for Image Classification
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决低对比度图像在恶劣天气条件下分类性能下降的问题。论文提出了一种可微分、参数化的图像直方图匹配预处理方法，通过优化目标像素分布来提升分类器性能，相比传统直方图均衡化方法更具适应性和效果。**

- **链接: [http://arxiv.org/pdf/2506.01346v1](http://arxiv.org/pdf/2506.01346v1)**

> **作者:** Rikuto Otsuka; Yuho Shoji; Yuka Ogino; Takahiro Toizumi; Atsushi Ito
>
> **摘要:** This paper rethinks image histogram matching (HM) and proposes a differentiable and parametric HM preprocessing for a downstream classifier. Convolutional neural networks have demonstrated remarkable achievements in classification tasks. However, they often exhibit degraded performance on low-contrast images captured under adverse weather conditions. To maintain classifier performance under low-contrast images, histogram equalization (HE) is commonly used. HE is a special case of HM using a uniform distribution as a target pixel value distribution. In this paper, we focus on the shape of the target pixel value distribution. Compared to a uniform distribution, a single, well-designed distribution could have potential to improve the performance of the downstream classifier across various adverse weather conditions. Based on this hypothesis, we propose a differentiable and parametric HM that optimizes the target distribution using the loss function of the downstream classifier. This method addresses pixel value imbalances by transforming input images with arbitrary distributions into a target distribution optimized for the classifier. Our HM is trained on only normal weather images using the classifier. Experimental results show that a classifier trained with our proposed HM outperforms conventional preprocessing methods under adverse weather conditions.
>
---
#### [new 068] SVQA-R1: Reinforcing Spatial Reasoning in MLLMs via View-Consistent Reward Optimization
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的**空间推理任务**，旨在解决现有模型在**空间视觉问答（Spatial VQA）**中对相对位置、距离和物体配置理解不足的问题。作者提出**SVQA-R1**框架，结合基于规则的强化学习（R1）与新型分组强化策略**Spatial-GRPO**，通过扰动空间关系构建一致性奖励，提升模型的空间理解与推理能力，且无需监督微调数据。实验表明其在多个基准上表现优异且推理路径可解释。**

- **链接: [http://arxiv.org/pdf/2506.01371v1](http://arxiv.org/pdf/2506.01371v1)**

> **作者:** Peiyao Wang; Haibin Ling
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Spatial reasoning remains a critical yet underdeveloped capability in existing vision-language models (VLMs), especially for Spatial Visual Question Answering (Spatial VQA) tasks that require understanding relative positions, distances, and object configurations. Inspired by the R1 paradigm introduced in DeepSeek-R1, which enhances reasoning in language models through rule-based reinforcement learning (RL), we propose SVQA-R1, the first framework to extend R1-style training to spatial VQA. In particular, we introduce Spatial-GRPO, a novel group-wise RL strategy that constructs view-consistent rewards by perturbing spatial relations between objects, e.g., mirror flipping, thereby encouraging the model to develop a consistent and grounded understanding of space. Our model, SVQA-R1, not only achieves dramatically improved accuracy on spatial VQA benchmarks but also exhibits interpretable reasoning paths even without using supervised fine-tuning (SFT) data. Extensive experiments and visualization demonstrate the effectiveness of SVQA-R1 across multiple spatial reasoning benchmarks.
>
---
#### [new 069] Fourier-Modulated Implicit Neural Representation for Multispectral Satellite Image Compression
- **分类: cs.CV; cs.AI**

- **简介: 论文任务是多光谱卫星图像压缩。针对其高维、多分辨率带来的压缩难题，提出ImpliSat框架，采用隐式神经表示建模图像为连续函数，并引入傅里叶调制算法动态适配各波段特征，实现高效压缩与细节保留。**

- **链接: [http://arxiv.org/pdf/2506.01234v1](http://arxiv.org/pdf/2506.01234v1)**

> **作者:** Woojin Cho; Steve Andreas Immanuel; Junhyuk Heo; Darongsae Kwon
>
> **备注:** Accepted to IGARSS 2025 (Oral)
>
> **摘要:** Multispectral satellite images play a vital role in agriculture, fisheries, and environmental monitoring. However, their high dimensionality, large data volumes, and diverse spatial resolutions across multiple channels pose significant challenges for data compression and analysis. This paper presents ImpliSat, a unified framework specifically designed to address these challenges through efficient compression and reconstruction of multispectral satellite data. ImpliSat leverages Implicit Neural Representations (INR) to model satellite images as continuous functions over coordinate space, capturing fine spatial details across varying spatial resolutions. Furthermore, we introduce a Fourier modulation algorithm that dynamically adjusts to the spectral and spatial characteristics of each band, ensuring optimal compression while preserving critical image details.
>
---
#### [new 070] Continual-MEGA: A Large-scale Benchmark for Generalizable Continual Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于持续学习与异常检测任务，旨在解决现实场景中持续学习时对未见类别异常的零样本泛化问题。作者构建了大规模基准Continual-MEGA，包含新提出的ContinualAD数据集，并提出统一基线算法，在小样本检测和泛化能力上表现更优。**

- **链接: [http://arxiv.org/pdf/2506.00956v1](http://arxiv.org/pdf/2506.00956v1)**

> **作者:** Geonu Lee; Yujeong Oh; Geonhui Jang; Soyoung Lee; Jeonghyo Song; Sungmin Cha; YoungJoon Yoo
>
> **摘要:** In this paper, we introduce a new benchmark for continual learning in anomaly detection, aimed at better reflecting real-world deployment scenarios. Our benchmark, Continual-MEGA, includes a large and diverse dataset that significantly expands existing evaluation settings by combining carefully curated existing datasets with our newly proposed dataset, ContinualAD. In addition to standard continual learning with expanded quantity, we propose a novel scenario that measures zero-shot generalization to unseen classes, those not observed during continual adaptation. This setting poses a new problem setting that continual adaptation also enhances zero-shot performance. We also present a unified baseline algorithm that improves robustness in few-shot detection and maintains strong generalization. Through extensive evaluations, we report three key findings: (1) existing methods show substantial room for improvement, particularly in pixel-level defect localization; (2) our proposed method consistently outperforms prior approaches; and (3) the newly introduced ContinualAD dataset enhances the performance of strong anomaly detection models. We release the benchmark and code in https://github.com/Continual-Mega/Continual-Mega.
>
---
#### [new 071] Temporal In-Context Fine-Tuning for Versatile Control of Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于视频扩散模型的可控生成任务，旨在解决在数据和计算资源有限的情况下实现灵活、高效的条件生成问题。论文提出了Temporal In-Context Fine-Tuning（TIC-FT）方法，通过沿时间轴拼接条件帧与目标帧并插入带噪声的缓冲帧，实现对预训练模型的高效微调，无需修改架构即可适应多种生成任务。**

- **链接: [http://arxiv.org/pdf/2506.00996v1](http://arxiv.org/pdf/2506.00996v1)**

> **作者:** Kinam Kim; Junha Hyung; Jaegul Choo
>
> **备注:** project page: https://kinam0252.github.io/TIC-FT/
>
> **摘要:** Recent advances in text-to-video diffusion models have enabled high-quality video synthesis, but controllable generation remains challenging, particularly under limited data and compute. Existing fine-tuning methods for conditional generation often rely on external encoders or architectural modifications, which demand large datasets and are typically restricted to spatially aligned conditioning, limiting flexibility and scalability. In this work, we introduce Temporal In-Context Fine-Tuning (TIC-FT), an efficient and versatile approach for adapting pretrained video diffusion models to diverse conditional generation tasks. Our key idea is to concatenate condition and target frames along the temporal axis and insert intermediate buffer frames with progressively increasing noise levels. These buffer frames enable smooth transitions, aligning the fine-tuning process with the pretrained model's temporal dynamics. TIC-FT requires no architectural changes and achieves strong performance with as few as 10-30 training samples. We validate our method across a range of tasks, including image-to-video and video-to-video generation, using large-scale base models such as CogVideoX-5B and Wan-14B. Extensive experiments show that TIC-FT outperforms existing baselines in both condition fidelity and visual quality, while remaining highly efficient in both training and inference. For additional results, visit https://kinam0252.github.io/TIC-FT/
>
---
#### [new 072] Generic Token Compression in Multimodal Large Language Models from an Explainability Perspective
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型（MLLM）任务，旨在解决视觉token处理带来的高计算成本问题。作者提出一种基于可解释性的token压缩方法，可在输入阶段有效筛选重要视觉token，并通过轻量卷积网络学习映射关系，实现高效部署。实验表明该方法在多个模型和数据集上均表现优异。**

- **链接: [http://arxiv.org/pdf/2506.01097v1](http://arxiv.org/pdf/2506.01097v1)**

> **作者:** Lei Lei; Jie Gu; Xiaokang Ma; Chu Tang; Jingmin Chen; Tong Xu
>
> **摘要:** Existing Multimodal Large Language Models (MLLMs) process a large number of visual tokens, leading to significant computational costs and inefficiency. Previous works generally assume that all visual tokens are necessary in the shallow layers of LLMs, and therefore token compression typically occurs in intermediate layers. In contrast, our study reveals an interesting insight: with proper selection, token compression is feasible at the input stage of LLM with negligible performance loss. Specifically, we reveal that explainability methods can effectively evaluate the importance of each visual token with respect to the given instruction, which can well guide the token compression. Furthermore, we propose to learn a mapping from the attention map of the first LLM layer to the explanation results, thereby avoiding the need for a full inference pass and facilitating practical deployment. Interestingly, this mapping can be learned using a simple and lightweight convolutional network, whose training is efficient and independent of MLLMs. Extensive experiments on 10 image and video benchmarks across three leading MLLMs (Qwen2-VL, LLaVA-OneVision, and VILA1.5) demonstrate the effectiveness of our approach, e.g., pruning 50% visual tokens while retaining more than 96% of the original performance across all benchmarks for all these three MLLMs. It also exhibits strong generalization, even when the number of tokens in inference far exceeds that used in training.
>
---
#### [new 073] EgoVIS@CVPR: What Changed and What Could Have Changed? State-Change Counterfactuals for Procedure-Aware Video Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于视频表示学习任务，旨在提升模型对 procedural 活动的理解。它通过引入 LLM 生成的状态变化描述及其反事实推理作为监督信号，使模型能更好地理解动作与场景变化之间的因果关系。工作包括构建状态变化描述、生成反事实场景，并在多个程序感知任务上验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2506.00101v1](http://arxiv.org/pdf/2506.00101v1)**

> **作者:** Chi-Hsi Kung; Frangil Ramirez; Juhyung Ha; Yi-Ting Chen; David Crandall; Yi-Hsuan Tsai
>
> **备注:** 4 pages, 1 figure, 4 tables. Full paper is available at arXiv:2503.21055
>
> **摘要:** Understanding a procedural activity requires modeling both how action steps transform the scene, and how evolving scene transformations can influence the sequence of action steps, even those that are accidental or erroneous. Yet, existing work on procedure-aware video representations fails to explicitly learned the state changes (scene transformations). In this work, we study procedure-aware video representation learning by incorporating state-change descriptions generated by LLMs as supervision signals for video encoders. Moreover, we generate state-change counterfactuals that simulate hypothesized failure outcomes, allowing models to learn by imagining the unseen ``What if'' scenarios. This counterfactual reasoning facilitates the model's ability to understand the cause and effect of each step in an activity. To verify the procedure awareness of our model, we conduct extensive experiments on procedure-aware tasks, including temporal action segmentation, error detection, and more. Our results demonstrate the effectiveness of the proposed state-change descriptions and their counterfactuals, and achieve significant improvements on multiple tasks.
>
---
#### [new 074] Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型安全任务，旨在解决大型视觉-语言模型（LVLMs）易受视觉对抗攻击的问题。作者提出F3方法，通过引入简单扰动来净化对抗样本，利用跨模态注意力机制提升模型鲁棒性，具备训练-free、高效等优点。**

- **链接: [http://arxiv.org/pdf/2506.01064v1](http://arxiv.org/pdf/2506.01064v1)**

> **作者:** Yudong Zhang; Ruobing Xie; Yiqing Huang; Jiansheng Chen; Xingwu Sun; Zhanhui Kang; Di Wang; Yu Wang
>
> **摘要:** Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. Despite their potential impact, the development of effective methods for purifying such adversarial examples has received relatively limited attention. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive "fighting fire with fire" strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code will be made publicly available.
>
---
#### [new 075] GOBench: Benchmarking Geometric Optics Generation and Understanding of MLLMs
- **分类: cs.CV**

- **简介: 该论文属于视觉生成与理解任务，旨在评估多模态大模型（MLLMs）在几何光学领域的生成与理解能力。作者构建了GOBench基准，包含生成真实光学图像和理解光学现象两个任务，并发现当前模型在这两方面均存在显著不足。**

- **链接: [http://arxiv.org/pdf/2506.00991v1](http://arxiv.org/pdf/2506.00991v1)**

> **作者:** Xiaorong Zhu; Ziheng Jia; Jiarui Wang; Xiangyu Zhao; Haodong Duan; Xiongkuo Min; Jia Wang; Zicheng Zhang; Guangtao Zhai
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** The rapid evolution of Multi-modality Large Language Models (MLLMs) is driving significant advancements in visual understanding and generation. Nevertheless, a comprehensive assessment of their capabilities, concerning the fine-grained physical principles especially in geometric optics, remains underexplored. To address this gap, we introduce GOBench, the first benchmark to systematically evaluate MLLMs' ability across two tasks: 1) Generating Optically Authentic Imagery and 2) Understanding Underlying Optical Phenomena. We curates high-quality prompts of geometric optical scenarios and use MLLMs to construct GOBench-Gen-1k dataset.We then organize subjective experiments to assess the generated imagery based on Optical Authenticity, Aesthetic Quality, and Instruction Fidelity, revealing MLLMs' generation flaws that violate optical principles. For the understanding task, we apply crafted evaluation instructions to test optical understanding ability of eleven prominent MLLMs. The experimental results demonstrate that current models face significant challenges in both optical generation and understanding. The top-performing generative model, GPT-4o-Image, cannot perfectly complete all generation tasks, and the best-performing MLLM model, Gemini-2.5Pro, attains a mere 37.35\% accuracy in optical understanding.
>
---
#### [new 076] Quotient Network - A Network Similar to ResNet but Learning Quotients
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度学习模型设计任务，旨在解决ResNet中残差学习对特征大小敏感且缺乏独立意义的问题。作者提出“商网络”（Quotient Network），通过学习目标特征与现有特征的商值来改进模型性能，并设计相应规则提升训练效率。实验表明其在多个数据集上优于ResNet，且不增加参数量。**

- **链接: [http://arxiv.org/pdf/2506.00992v1](http://arxiv.org/pdf/2506.00992v1)**

> **作者:** Peng Hui; Jiamuyang Zhao; Changxin Li; Qingzhen Zhu
>
> **备注:** This manuscript is the original version submitted to NeurIPS 2024, which was later revised and published as "Quotient Network: A Network Similar to ResNet but Learning Quotients" in Algorithms 2024, 17(11), 521 (https://doi.org/10.3390/a17110521). Please cite the journal version when referring to this work
>
> **摘要:** The emergence of ResNet provides a powerful tool for training extremely deep networks. The core idea behind it is to change the learning goals of the network. It no longer learns new features from scratch but learns the difference between the target and existing features. However, the difference between the two kinds of features does not have an independent and clear meaning, and the amount of learning is based on the absolute rather than the relative difference, which is sensitive to the size of existing features. We propose a new network that perfectly solves these two problems while still having the advantages of ResNet. Specifically, it chooses to learn the quotient of the target features with the existing features, so we call it the quotient network. In order to enable this network to learn successfully and achieve higher performance, we propose some design rules for this network so that it can be trained efficiently and achieve better performance than ResNet. Experiments on the CIFAR10, CIFAR100, and SVHN datasets prove that this network can stably achieve considerable improvements over ResNet by simply making tiny corresponding changes to the original ResNet network without adding new parameters.
>
---
#### [new 077] HOSIG: Full-Body Human-Object-Scene Interaction Generation with Hierarchical Scene Perception
- **分类: cs.CV**

- **简介: 该论文属于计算机图形学与动画任务，旨在解决人体与物体、场景交互生成中的不自然穿透和协调困难问题。作者提出了HOSIG框架，通过分层场景感知方法实现全身交互合成，包含抓取姿态生成、导航路径规划和动作扩散模型三个核心模块，显著提升了交互的真实感与连贯性。**

- **链接: [http://arxiv.org/pdf/2506.01579v1](http://arxiv.org/pdf/2506.01579v1)**

> **作者:** Wei Yao; Yunlian Sun; Hongwen Zhang; Yebin Liu; Jinhui Tang
>
> **摘要:** Generating high-fidelity full-body human interactions with dynamic objects and static scenes remains a critical challenge in computer graphics and animation. Existing methods for human-object interaction often neglect scene context, leading to implausible penetrations, while human-scene interaction approaches struggle to coordinate fine-grained manipulations with long-range navigation. To address these limitations, we propose HOSIG, a novel framework for synthesizing full-body interactions through hierarchical scene perception. Our method decouples the task into three key components: 1) a scene-aware grasp pose generator that ensures collision-free whole-body postures with precise hand-object contact by integrating local geometry constraints, 2) a heuristic navigation algorithm that autonomously plans obstacle-avoiding paths in complex indoor environments via compressed 2D floor maps and dual-component spatial reasoning, and 3) a scene-guided motion diffusion model that generates trajectory-controlled, full-body motions with finger-level accuracy by incorporating spatial anchors and dual-space classifier-free guidance. Extensive experiments on the TRUMANS dataset demonstrate superior performance over state-of-the-art methods. Notably, our framework supports unlimited motion length through autoregressive generation and requires minimal manual intervention. This work bridges the critical gap between scene-aware navigation and dexterous object manipulation, advancing the frontier of embodied interaction synthesis. Codes will be available after publication. Project page: http://yw0208.github.io/hosig
>
---
#### [new 078] GThinker: Towards General Multimodal Reasoning via Cue-Guided Rethinking
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态推理任务，旨在解决现有模型在视觉中心任务中表现不佳的问题。作者提出了GThinker模型，通过引入基于视觉线索的Cue-Rethinking推理模式和两阶段训练方法，提升了多模态推理能力，并构建了GThinker-11K数据集支持训练。实验表明其在多项基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.01078v1](http://arxiv.org/pdf/2506.01078v1)**

> **作者:** Yufei Zhan; Ziheng Wu; Yousong Zhu; Rongkun Xue; Ruipu Luo; Zhenghao Chen; Can Zhang; Yifan Li; Zhentao He; Zheming Yang; Ming Tang; Minghui Qiu; Jinqiao Wang
>
> **备注:** Tech report
>
> **摘要:** Despite notable advancements in multimodal reasoning, leading Multimodal Large Language Models (MLLMs) still underperform on vision-centric multimodal reasoning tasks in general scenarios. This shortfall stems from their predominant reliance on logic- and knowledge-based slow thinking strategies, while effective for domains like math and science, fail to integrate visual information effectively during reasoning. Consequently, these models often fail to adequately ground visual cues, resulting in suboptimal performance in tasks that require multiple plausible visual interpretations and inferences. To address this, we present GThinker (General Thinker), a novel reasoning MLLM excelling in multimodal reasoning across general scenarios, mathematics, and science. GThinker introduces Cue-Rethinking, a flexible reasoning pattern that grounds inferences in visual cues and iteratively reinterprets these cues to resolve inconsistencies. Building on this pattern, we further propose a two-stage training pipeline, including pattern-guided cold start and incentive reinforcement learning, designed to enable multimodal reasoning capabilities across domains. Furthermore, to support the training, we construct GThinker-11K, comprising 7K high-quality, iteratively-annotated reasoning paths and 4K curated reinforcement learning samples, filling the data gap toward general multimodal reasoning. Extensive experiments demonstrate that GThinker achieves 81.5% on the challenging comprehensive multimodal reasoning benchmark M$^3$CoT, surpassing the latest O4-mini model. It also shows an average improvement of 2.1% on general scenario multimodal reasoning benchmarks, while maintaining on-par performance in mathematical reasoning compared to counterpart advanced reasoning models. The code, model, and data will be released soon at https://github.com/jefferyZhan/GThinker.
>
---
#### [new 079] Self-supervised ControlNet with Spatio-Temporal Mamba for Real-world Video Super-resolution
- **分类: cs.CV; I.4.4; I.2.6**

- **简介: 该论文属于视频超分辨率任务，旨在解决真实场景中低分辨率视频恢复为高分辨率时易引入噪声和伪影的问题。作者提出了一种结合自监督学习与Mamba模型的框架，通过时空注意力机制和自监督ControlNet提升视频内容一致性与细节质量，并采用三阶段训练策略优化模型效果。**

- **链接: [http://arxiv.org/pdf/2506.01037v1](http://arxiv.org/pdf/2506.01037v1)**

> **作者:** Shijun Shi; Jing Xu; Lijing Lu; Zhihang Li; Kai Hu
>
> **备注:** 11 pages, 10 figures, accepted by CVPR 2025
>
> **摘要:** Existing diffusion-based video super-resolution (VSR) methods are susceptible to introducing complex degradations and noticeable artifacts into high-resolution videos due to their inherent randomness. In this paper, we propose a noise-robust real-world VSR framework by incorporating self-supervised learning and Mamba into pre-trained latent diffusion models. To ensure content consistency across adjacent frames, we enhance the diffusion model with a global spatio-temporal attention mechanism using the Video State-Space block with a 3D Selective Scan module, which reinforces coherence at an affordable computational cost. To further reduce artifacts in generated details, we introduce a self-supervised ControlNet that leverages HR features as guidance and employs contrastive learning to extract degradation-insensitive features from LR videos. Finally, a three-stage training strategy based on a mixture of HR-LR videos is proposed to stabilize VSR training. The proposed Self-supervised ControlNet with Spatio-Temporal Continuous Mamba based VSR algorithm achieves superior perceptual quality than state-of-the-arts on real-world VSR benchmark datasets, validating the effectiveness of the proposed model design and training strategies.
>
---
#### [new 080] MotionSight: Boosting Fine-Grained Motion Understanding in Multimodal LLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态视频理解任务，旨在解决现有MLLM在细粒度视频运动分析中的不足。作者提出MotionSight方法，利用视觉提示提升零样本下的运动感知，并构建了大规模数据集MotionVid-QA以推动研究。**

- **链接: [http://arxiv.org/pdf/2506.01674v1](http://arxiv.org/pdf/2506.01674v1)**

> **作者:** Yipeng Du; Tiehan Fan; Kepan Nan; Rui Xie; Penghao Zhou; Xiang Li; Jian Yang; Zhenheng Yang; Ying Tai
>
> **摘要:** Despite advancements in Multimodal Large Language Models (MLLMs), their proficiency in fine-grained video motion understanding remains critically limited. They often lack inter-frame differencing and tend to average or ignore subtle visual cues. Furthermore, while visual prompting has shown potential in static images, its application to video's temporal complexities, particularly for fine-grained motion understanding, remains largely unexplored. We investigate whether inherent capability can be unlocked and boost MLLMs' motion perception and enable distinct visual signatures tailored to decouple object and camera motion cues. In this study, we introduce MotionSight, a novel zero-shot method pioneering object-centric visual spotlight and motion blur as visual prompts to effectively improve fine-grained motion understanding without training. To convert this into valuable data assets, we curated MotionVid-QA, the first large-scale dataset for fine-grained video motion understanding, with hierarchical annotations including SFT and preference data, {\Theta}(40K) video clips and {\Theta}(87K) QAs. Experiments show MotionSight achieves state-of-the-art open-source performance and competitiveness with commercial models. In particular, for fine-grained motion understanding we present a novel zero-shot technique and a large-scale, high-quality dataset. All the code and annotations will be publicly available.
>
---
#### [new 081] Aligned Contrastive Loss for Long-Tailed Recognition
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决长尾分布下的图像识别问题。作者提出了一种对齐对比学习（ACL）算法，通过理论分析发现监督对比学习中的梯度冲突和不平衡问题，并设计方法加以解决，从而提升模型在多个长尾数据集上的性能，取得了新的最优结果。**

- **链接: [http://arxiv.org/pdf/2506.01071v1](http://arxiv.org/pdf/2506.01071v1)**

> **作者:** Jiali Ma; Jiequan Cui; Maeno Kazuki; Lakshmi Subramanian; Karlekar Jayashree; Sugiri Pranata; Hanwang Zhang
>
> **备注:** Accepted by CVPR 2025 DG-EBF Workshop
>
> **摘要:** In this paper, we propose an Aligned Contrastive Learning (ACL) algorithm to address the long-tailed recognition problem. Our findings indicate that while multi-view training boosts the performance, contrastive learning does not consistently enhance model generalization as the number of views increases. Through theoretical gradient analysis of supervised contrastive learning (SCL), we identify gradient conflicts, and imbalanced attraction and repulsion gradients between positive and negative pairs as the underlying issues. Our ACL algorithm is designed to eliminate these problems and demonstrates strong performance across multiple benchmarks. We validate the effectiveness of ACL through experiments on long-tailed CIFAR, ImageNet, Places, and iNaturalist datasets. Results show that ACL achieves new state-of-the-art performance.
>
---
#### [new 082] EcoLens: Leveraging Multi-Objective Bayesian Optimization for Energy-Efficient Video Processing on Edge Devices
- **分类: cs.CV**

- **简介: 该论文属于边缘计算任务，旨在解决资源受限环境下视频处理的能耗与语义保持平衡的问题。工作包括构建配置性能档案，提出基于多目标贝叶斯优化的EcoLens系统，动态调整参数以实现低能耗高精度的视频分析。**

- **链接: [http://arxiv.org/pdf/2506.00754v1](http://arxiv.org/pdf/2506.00754v1)**

> **作者:** Benjamin Civjan; Bo Chen; Ruixiao Zhang; Klara Nahrstedt
>
> **摘要:** Video processing for real-time analytics in resource-constrained environments presents a significant challenge in balancing energy consumption and video semantics. This paper addresses the problem of energy-efficient video processing by proposing a system that dynamically optimizes processing configurations to minimize energy usage on the edge, while preserving essential video features for deep learning inference. We first gather an extensive offline profile of various configurations consisting of device CPU frequencies, frame filtering features, difference thresholds, and video bitrates, to establish apriori knowledge of their impact on energy consumption and inference accuracy. Leveraging this insight, we introduce an online system that employs multi-objective Bayesian optimization to intelligently explore and adapt configurations in real time. Our approach continuously refines processing settings to meet a target inference accuracy with minimal edge device energy expenditure. Experimental results demonstrate the system's effectiveness in reducing video processing energy use while maintaining high analytical performance, offering a practical solution for smart devices and edge computing applications.
>
---
#### [new 083] MS-RAFT-3D: A Multi-Scale Architecture for Recurrent Image-Based Scene Flow
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决图像场景流估计问题。通过引入多尺度架构到循环网络中，改进了现有方法，在KITTI和Spring数据集上取得更好性能。**

- **链接: [http://arxiv.org/pdf/2506.01443v1](http://arxiv.org/pdf/2506.01443v1)**

> **作者:** Jakob Schmid; Azin Jahedi; Noah Berenguel Senn; Andrés Bruhn
>
> **备注:** ICIP 2025
>
> **摘要:** Although multi-scale concepts have recently proven useful for recurrent network architectures in the field of optical flow and stereo, they have not been considered for image-based scene flow so far. Hence, based on a single-scale recurrent scene flow backbone, we develop a multi-scale approach that generalizes successful hierarchical ideas from optical flow to image-based scene flow. By considering suitable concepts for the feature and the context encoder, the overall coarse-to-fine framework and the training loss, we succeed to design a scene flow approach that outperforms the current state of the art on KITTI and Spring by 8.7%(3.89 vs. 4.26) and 65.8% (9.13 vs. 26.71), respectively. Our code is available at https://github.com/cv-stuttgart/MS-RAFT-3D.
>
---
#### [new 084] Low-Rank Head Avatar Personalization with Registers
- **分类: cs.CV**

- **简介: 该论文属于头像生成任务，旨在解决通用模型难以还原个性化面部细节的问题。作者提出了一种低秩自适应架构Register Module，通过在预训练模型的中间特征中存储和重用信息，增强LoRA效果，实现对新身份的高质量适配，仅需少量参数即可捕捉如皱纹、纹身等高频率面部特征。**

- **链接: [http://arxiv.org/pdf/2506.01935v1](http://arxiv.org/pdf/2506.01935v1)**

> **作者:** Sai Tanmay Reddy Chakkera; Aggelina Chatziagapi; Md Moniruzzaman; Chen-Ping Yu; Yi-Hsuan Tsai; Dimitris Samaras
>
> **备注:** 23 pages, 16 figures. Project page: https://starc52.github.io/publications/2025-05-28-LoRAvatar/
>
> **摘要:** We introduce a novel method for low-rank personalization of a generic model for head avatar generation. Prior work proposes generic models that achieve high-quality face animation by leveraging large-scale datasets of multiple identities. However, such generic models usually fail to synthesize unique identity-specific details, since they learn a general domain prior. To adapt to specific subjects, we find that it is still challenging to capture high-frequency facial details via popular solutions like low-rank adaptation (LoRA). This motivates us to propose a specific architecture, a Register Module, that enhances the performance of LoRA, while requiring only a small number of parameters to adapt to an unseen identity. Our module is applied to intermediate features of a pre-trained model, storing and re-purposing information in a learnable 3D feature space. To demonstrate the efficacy of our personalization method, we collect a dataset of talking videos of individuals with distinctive facial details, such as wrinkles and tattoos. Our approach faithfully captures unseen faces, outperforming existing methods quantitatively and qualitatively. We will release the code, models, and dataset to the public.
>
---
#### [new 085] ECP-Mamba: An Efficient Multi-scale Self-supervised Contrastive Learning Method with State Space Model for PolSAR Image Classification
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分类任务，旨在解决PolSAR图像分类中依赖大量标注数据和计算效率低的问题。论文提出ECP-Mamba框架，结合多尺度自监督对比学习与状态空间模型，设计螺旋扫描策略和Cross Mamba模块，提升分类性能与效率。**

- **链接: [http://arxiv.org/pdf/2506.01040v1](http://arxiv.org/pdf/2506.01040v1)**

> **作者:** Zuzheng Kuang; Haixia Bi; Chen Xu; Jian Sun
>
> **摘要:** Recently, polarimetric synthetic aperture radar (PolSAR) image classification has been greatly promoted by deep neural networks. However,current deep learning-based PolSAR classification methods encounter difficulties due to its dependence on extensive labeled data and the computational inefficiency of architectures like Transformers. This paper presents ECP-Mamba, an efficient framework integrating multi-scale self-supervised contrastive learning with a state space model (SSM) backbone. Specifically, ECP-Mamba addresses annotation scarcity through a multi-scale predictive pretext task based on local-to-global feature correspondences, which uses a simplified self-distillation paradigm without negative sample pairs. To enhance computational efficiency,the Mamba architecture (a selective SSM) is first tailored for pixel-wise PolSAR classification task by designing a spiral scan strategy. This strategy prioritizes causally relevant features near the central pixel, leveraging the localized nature of pixel-wise classification tasks. Additionally, the lightweight Cross Mamba module is proposed to facilitates complementary multi-scale feature interaction with minimal overhead. Extensive experiments across four benchmark datasets demonstrate ECP-Mamba's effectiveness in balancing high accuracy with resource efficiency. On the Flevoland 1989 dataset, ECP-Mamba achieves state-of-the-art performance with an overall accuracy of 99.70%, average accuracy of 99.64% and Kappa coefficient of 99.62e-2. Our code will be available at https://github.com/HaixiaBi1982/ECP_Mamba.
>
---
#### [new 086] Ctrl-Crash: Controllable Diffusion for Realistic Car Crashes
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视频生成任务，旨在解决现有扩散模型难以生成逼真车祸视频的问题。作者提出Ctrl-Crash模型，通过引入边界框、碰撞类型和初始帧等条件信号，实现对车祸视频的可控生成，并支持反事实场景模拟，提升了生成视频的质量与现实感。**

- **链接: [http://arxiv.org/pdf/2506.00227v1](http://arxiv.org/pdf/2506.00227v1)**

> **作者:** Anthony Gosselin; Ge Ya Luo; Luis Lara; Florian Golemo; Derek Nowrouzezahrai; Liam Paull; Alexia Jolicoeur-Martineau; Christopher Pal
>
> **备注:** Under review
>
> **摘要:** Video diffusion techniques have advanced significantly in recent years; however, they struggle to generate realistic imagery of car crashes due to the scarcity of accident events in most driving datasets. Improving traffic safety requires realistic and controllable accident simulations. To tackle the problem, we propose Ctrl-Crash, a controllable car crash video generation model that conditions on signals such as bounding boxes, crash types, and an initial image frame. Our approach enables counterfactual scenario generation where minor variations in input can lead to dramatically different crash outcomes. To support fine-grained control at inference time, we leverage classifier-free guidance with independently tunable scales for each conditioning signal. Ctrl-Crash achieves state-of-the-art performance across quantitative video quality metrics (e.g., FVD and JEDi) and qualitative measurements based on a human-evaluation of physical realism and video quality compared to prior diffusion-based methods.
>
---
#### [new 087] QuantFace: Low-Bit Post-Training Quantization for One-Step Diffusion Face Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决扩散模型在手机等设备上部署困难的问题。论文提出QuantFace，通过低比特量化、量化-蒸馏低秩适应和自适应比特分配策略，实现高效的人脸修复。**

- **链接: [http://arxiv.org/pdf/2506.00820v1](http://arxiv.org/pdf/2506.00820v1)**

> **作者:** Jiatong Li; Libo Zhu; Haotong Qin; Jingkai Wang; Linghe Kong; Guihai Chen; Yulun Zhang; Xiaokang Yang
>
> **摘要:** Diffusion models have been achieving remarkable performance in face restoration. However, the heavy computations of diffusion models make it difficult to deploy them on devices like smartphones. In this work, we propose QuantFace, a novel low-bit quantization for one-step diffusion face restoration models, where the full-precision (\ie, 32-bit) weights and activations are quantized to 4$\sim$6-bit. We first analyze the data distribution within activations and find that they are highly variant. To preserve the original data information, we employ rotation-scaling channel balancing. Furthermore, we propose Quantization-Distillation Low-Rank Adaptation (QD-LoRA) that jointly optimizes for quantization and distillation performance. Finally, we propose an adaptive bit-width allocation strategy. We formulate such a strategy as an integer programming problem, which combines quantization error and perceptual metrics to find a satisfactory resource allocation. Extensive experiments on the synthetic and real-world datasets demonstrate the effectiveness of QuantFace under 6-bit and 4-bit. QuantFace achieves significant advantages over recent leading low-bit quantization methods for face restoration. The code is available at https://github.com/jiatongli2024/QuantFace.
>
---
#### [new 088] Perceptual Inductive Bias Is What You Need Before Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决对比学习在表示学习中忽略人类视觉归纳偏置导致的收敛慢、纹理偏差等问题。工作提出基于Marr理论的预训练阶段，先学习边界和表面表示，再进行语义学习，提升了模型性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.01201v1](http://arxiv.org/pdf/2506.01201v1)**

> **作者:** Tianqin Li; Junru Zhao; Dunhan Jiang; Shenghao Wu; Alan Ramirez; Tai Sing Lee
>
> **备注:** CVPR 2025. Tianqin Li and Junru Zhao contributed equally to this work. Due to a formatting error during the CVPR submission, the equal contribution note was omitted in the official proceedings. This arXiv version corrects that oversight. The author order follows alphabetical order by last name
>
> **摘要:** David Marr's seminal theory of human perception stipulates that visual processing is a multi-stage process, prioritizing the derivation of boundary and surface properties before forming semantic object representations. In contrast, contrastive representation learning frameworks typically bypass this explicit multi-stage approach, defining their objective as the direct learning of a semantic representation space for objects. While effective in general contexts, this approach sacrifices the inductive biases of vision, leading to slower convergence speed and learning shortcut resulting in texture bias. In this work, we demonstrate that leveraging Marr's multi-stage theory-by first constructing boundary and surface-level representations using perceptual constructs from early visual processing stages and subsequently training for object semantics-leads to 2x faster convergence on ResNet18, improved final representations on semantic segmentation, depth estimation, and object recognition, and enhanced robustness and out-of-distribution capability. Together, we propose a pretraining stage before the general contrastive representation pretraining to further enhance the final representation quality and reduce the overall convergence time via inductive bias from human vision systems.
>
---
#### [new 089] Semantic Palette-Guided Color Propagation
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决局部颜色编辑难以准确传播到语义相似区域的问题。现有方法依赖低级特征，缺乏内容感知能力。作者提出一种基于语义调色板的传播方法，通过提取语义调色板、优化编辑调色板和在相似语义区域传播编辑，实现了更自然、全局协调的颜色编辑效果。**

- **链接: [http://arxiv.org/pdf/2506.01441v1](http://arxiv.org/pdf/2506.01441v1)**

> **作者:** Zi-Yu Zhang; Bing-Feng Seng; Ya-Feng Du; Kang Li; Zhe-Cheng Wang; Zheng-Jun Du
>
> **备注:** 6 pages,5 figures, IEEE ICME 2025
>
> **摘要:** Color propagation aims to extend local color edits to similar regions across the input image. Conventional approaches often rely on low-level visual cues such as color, texture, or lightness to measure pixel similarity, making it difficult to achieve content-aware color propagation. While some recent approaches attempt to introduce semantic information into color editing, but often lead to unnatural, global color change in color adjustments. To overcome these limitations, we present a semantic palette-guided approach for color propagation. We first extract a semantic palette from an input image. Then, we solve an edited palette by minimizing a well-designed energy function based on user edits. Finally, local edits are accurately propagated to regions that share similar semantics via the solved palette. Our approach enables efficient yet accurate pixel-level color editing and ensures that local color changes are propagated in a content-aware manner. Extensive experiments demonstrated the effectiveness of our method.
>
---
#### [new 090] iDPA: Instance Decoupled Prompt Attention for Incremental Medical Object Detection
- **分类: cs.CV**

- **简介: 该论文提出iDPA框架，用于解决增量式医学目标检测任务中前景-背景耦合及提示信息耦合的问题。通过实例级提示生成和解耦提示注意力机制，提升检测性能，尤其在少样本设置下表现优异。**

- **链接: [http://arxiv.org/pdf/2506.00406v1](http://arxiv.org/pdf/2506.00406v1)**

> **作者:** Huahui Yi; Wei Xu; Ziyuan Qin; Xi Chen; Xiaohu Wu; Kang Li; Qicheng Lao
>
> **备注:** accepted to ICML 2025
>
> **摘要:** Existing prompt-based approaches have demonstrated impressive performance in continual learning, leveraging pre-trained large-scale models for classification tasks; however, the tight coupling between foreground-background information and the coupled attention between prompts and image-text tokens present significant challenges in incremental medical object detection tasks, due to the conceptual gap between medical and natural domains. To overcome these challenges, we introduce the \method~framework, which comprises two main components: 1) Instance-level Prompt Generation (\ipg), which decouples fine-grained instance-level knowledge from images and generates prompts that focus on dense predictions, and 2) Decoupled Prompt Attention (\dpa), which decouples the original prompt attention, enabling a more direct and efficient transfer of prompt information while reducing memory usage and mitigating catastrophic forgetting. We collect 13 clinical, cross-modal, multi-organ, and multi-category datasets, referred to as \dataset, and experiments demonstrate that \method~outperforms existing SOTA methods, with FAP improvements of 5.44\%, 4.83\%, 12.88\%, and 4.59\% in full data, 1-shot, 10-shot, and 50-shot settings, respectively.
>
---
#### [new 091] L3A: Label-Augmented Analytic Adaptation for Multi-Label Class Incremental Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多标签类增量学习任务，旨在解决类别遗忘、标签缺失和类别不平衡问题。作者提出L3A方法，包含伪标签模块和加权解析分类器，无需存储旧样本即可提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.00816v1](http://arxiv.org/pdf/2506.00816v1)**

> **作者:** Xiang Zhang; Run He; Jiao Chen; Di Fang; Ming Li; Ziqian Zeng; Cen Chen; Huiping Zhuang
>
> **备注:** Accepted by ICML2025
>
> **摘要:** Class-incremental learning (CIL) enables models to learn new classes continually without forgetting previously acquired knowledge. Multi-label CIL (MLCIL) extends CIL to a real-world scenario where each sample may belong to multiple classes, introducing several challenges: label absence, which leads to incomplete historical information due to missing labels, and class imbalance, which results in the model bias toward majority classes. To address these challenges, we propose Label-Augmented Analytic Adaptation (L3A), an exemplar-free approach without storing past samples. L3A integrates two key modules. The pseudo-label (PL) module implements label augmentation by generating pseudo-labels for current phase samples, addressing the label absence problem. The weighted analytic classifier (WAC) derives a closed-form solution for neural networks. It introduces sample-specific weights to adaptively balance the class contribution and mitigate class imbalance. Experiments on MS-COCO and PASCAL VOC datasets demonstrate that L3A outperforms existing methods in MLCIL tasks. Our code is available at https://github.com/scut-zx/L3A.
>
---
#### [new 092] Scene Detection Policies and Keyframe Extraction Strategies for Large-Scale Video Analysis
- **分类: cs.CV; cs.MM; 68T07; I.2.10; I.4.8; I.5.1**

- **简介: 该论文属于视频分析任务，旨在解决不同类型的视频在场景分割和关键帧提取上的通用性不足问题。论文提出了一种统一的自适应框架，根据视频长度动态选择场景分割策略，并采用轻量级模块进行关键帧评分，实现了高效、可扩展的视频预处理方案。**

- **链接: [http://arxiv.org/pdf/2506.00667v1](http://arxiv.org/pdf/2506.00667v1)**

> **作者:** Vasilii Korolkov
>
> **备注:** 24 pages, 8 figures, submitted as a preprint. ArXiv preprint only, not submitted to a journal yet
>
> **摘要:** Robust scene segmentation and keyframe extraction are essential preprocessing steps in video understanding pipelines, supporting tasks such as indexing, summarization, and semantic retrieval. However, existing methods often lack generalizability across diverse video types and durations. We present a unified, adaptive framework for automatic scene detection and keyframe selection that handles formats ranging from short-form media to long-form films, archival content, and surveillance footage. Our system dynamically selects segmentation policies based on video length: adaptive thresholding for short videos, hybrid strategies for mid-length ones, and interval-based splitting for extended recordings. This ensures consistent granularity and efficient processing across domains. For keyframe selection, we employ a lightweight module that scores sampled frames using a composite metric of sharpness, luminance, and temporal spread, avoiding complex saliency models while ensuring visual relevance. Designed for high-throughput workflows, the system is deployed in a commercial video analysis platform and has processed content from media, education, research, and security domains. It offers a scalable and interpretable solution suitable for downstream applications such as UI previews, embedding pipelines, and content filtering. We discuss practical implementation details and outline future enhancements, including audio-aware segmentation and reinforcement-learned frame scoring.
>
---
#### [new 093] SVarM: Linear Support Varifold Machines for Classification and Regression on Geometric Data
- **分类: cs.CV; cs.LG; math.DG; math.FA; 49Q15, 53C42, 46N10; I.5.1; I.4.0**

- **简介: 该论文属于几何数据分类与回归任务，旨在解决形状空间非欧几里得特性带来的建模难题。作者提出SVarM方法，利用可训练测试函数h，基于varifold表示构建线性支持向量机框架，在保持性能的同时显著减少参数数量。**

- **链接: [http://arxiv.org/pdf/2506.01189v1](http://arxiv.org/pdf/2506.01189v1)**

> **作者:** Emmanuel Hartman; Nicolas Charon
>
> **备注:** 22 pages, 12 figures
>
> **摘要:** Despite progress in the rapidly developing field of geometric deep learning, performing statistical analysis on geometric data--where each observation is a shape such as a curve, graph, or surface--remains challenging due to the non-Euclidean nature of shape spaces, which are defined as equivalence classes under invariance groups. Building machine learning frameworks that incorporate such invariances, notably to shape parametrization, is often crucial to ensure generalizability of the trained models to new observations. This work proposes SVarM to exploit varifold representations of shapes as measures and their duality with test functions $h:\mathbb{R}^n \times S^{n-1} \to \mathbb{R}$. This method provides a general framework akin to linear support vector machines but operating instead over the infinite-dimensional space of varifolds. We develop classification and regression models on shape datasets by introducing a neural network-based representation of the trainable test function $h$. This approach demonstrates strong performance and robustness across various shape graph and surface datasets, achieving results comparable to state-of-the-art methods while significantly reducing the number of trainable parameters.
>
---
#### [new 094] Video Signature: In-generation Watermarking for Latent Video Diffusion Models
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于视频生成中的水印技术任务，旨在解决AIGC视频的版权保护与内容溯源问题。作者提出VIDSIG方法，在生成过程中嵌入水印，通过微调潜在解码器并引入时间对齐模块，实现高质量、鲁棒性强的视频水印方案。**

- **链接: [http://arxiv.org/pdf/2506.00652v1](http://arxiv.org/pdf/2506.00652v1)**

> **作者:** Yu Huang; Junhao Chen; Qi Zheng; Hanqian Li; Shuliang Liu; Xuming Hu
>
> **摘要:** The rapid development of Artificial Intelligence Generated Content (AIGC) has led to significant progress in video generation but also raises serious concerns about intellectual property protection and reliable content tracing. Watermarking is a widely adopted solution to this issue, but existing methods for video generation mainly follow a post-generation paradigm, which introduces additional computational overhead and often fails to effectively balance the trade-off between video quality and watermark extraction. To address these issues, we propose Video Signature (VIDSIG), an in-generation watermarking method for latent video diffusion models, which enables implicit and adaptive watermark integration during generation. Specifically, we achieve this by partially fine-tuning the latent decoder, where Perturbation-Aware Suppression (PAS) pre-identifies and freezes perceptually sensitive layers to preserve visual quality. Beyond spatial fidelity, we further enhance temporal consistency by introducing a lightweight Temporal Alignment module that guides the decoder to generate coherent frame sequences during fine-tuning. Experimental results show that VIDSIG achieves the best overall performance in watermark extraction, visual quality, and generation efficiency. It also demonstrates strong robustness against both spatial and temporal tampering, highlighting its practicality in real-world scenarios.
>
---
#### [new 095] Ultra-High-Resolution Image Synthesis: Data, Method and Evaluation
- **分类: cs.CV**

- **简介: 该论文属于图像合成任务，旨在解决超高清图像生成缺乏标准数据集和计算瓶颈的问题。工作包括构建Aesthetic-4K数据集、提出Diffusion-4K生成框架及新评估指标，实现高质量4K图像合成。**

- **链接: [http://arxiv.org/pdf/2506.01331v1](http://arxiv.org/pdf/2506.01331v1)**

> **作者:** Jinjin Zhang; Qiuyu Huang; Junjie Liu; Xiefan Guo; Di Huang
>
> **摘要:** Ultra-high-resolution image synthesis holds significant potential, yet remains an underexplored challenge due to the absence of standardized benchmarks and computational constraints. In this paper, we establish Aesthetic-4K, a meticulously curated dataset containing dedicated training and evaluation subsets specifically designed for comprehensive research on ultra-high-resolution image synthesis. This dataset consists of high-quality 4K images accompanied by descriptive captions generated by GPT-4o. Furthermore, we propose Diffusion-4K, an innovative framework for the direct generation of ultra-high-resolution images. Our approach incorporates the Scale Consistent Variational Auto-Encoder (SC-VAE) and Wavelet-based Latent Fine-tuning (WLF), which are designed for efficient visual token compression and the capture of intricate details in ultra-high-resolution images, thereby facilitating direct training with photorealistic 4K data. This method is applicable to various latent diffusion models and demonstrates its efficacy in synthesizing highly detailed 4K images. Additionally, we propose novel metrics, namely the GLCM Score and Compression Ratio, to assess the texture richness and fine details in local patches, in conjunction with holistic measures such as FID, Aesthetics, and CLIPScore, enabling a thorough and multifaceted evaluation of ultra-high-resolution image synthesis. Consequently, Diffusion-4K achieves impressive performance in ultra-high-resolution image synthesis, particularly when powered by state-of-the-art large-scale diffusion models (eg, Flux-12B). The source code is publicly available at https://github.com/zhang0jhon/diffusion-4k.
>
---
#### [new 096] Modality Translation and Registration of MR and Ultrasound Images Using Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决磁共振（MR）与超声（US）图像在前列腺癌诊断中的多模态配准难题。现有方法因模态差异大，难以准确对齐关键边界。作者提出一种基于扩散模型的解剖一致性模态翻译（ACMT）网络，通过构建中间伪模态，统一翻译MR和US图像，从而提升配准精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.01025v1](http://arxiv.org/pdf/2506.01025v1)**

> **作者:** Xudong Ma; Nantheera Anantrasirichai; Stefanos Bolomytis; Alin Achim
>
> **摘要:** Multimodal MR-US registration is critical for prostate cancer diagnosis. However, this task remains challenging due to significant modality discrepancies. Existing methods often fail to align critical boundaries while being overly sensitive to irrelevant details. To address this, we propose an anatomically coherent modality translation (ACMT) network based on a hierarchical feature disentanglement design. We leverage shallow-layer features for texture consistency and deep-layer features for boundary preservation. Unlike conventional modality translation methods that convert one modality into another, our ACMT introduces the customized design of an intermediate pseudo modality. Both MR and US images are translated toward this intermediate domain, effectively addressing the bottlenecks faced by traditional translation methods in the downstream registration task. Experiments demonstrate that our method mitigates modality-specific discrepancies while preserving crucial anatomical boundaries for accurate registration. Quantitative evaluations show superior modality similarity compared to state-of-the-art modality translation methods. Furthermore, downstream registration experiments confirm that our translated images achieve the best alignment performance, highlighting the robustness of our framework for multi-modal prostate image registration.
>
---
#### [new 097] Many-for-Many: Unify the Training of Multiple Video and Image Generation and Manipulation Tasks
- **分类: cs.CV**

- **简介: 该论文属于多任务视觉生成与编辑任务，旨在解决训练多个特定任务模型成本高且泛化能力差的问题。作者提出“Many-for-Many”框架，通过轻量级适配器和图文视频联合训练，实现单模型支持十余种任务，并引入深度图提升3D空间感知能力，训练出性能优异的统一模型。**

- **链接: [http://arxiv.org/pdf/2506.01758v1](http://arxiv.org/pdf/2506.01758v1)**

> **作者:** Tao Yang; Ruibin Li; Yangming Shi; Yuqi Zhang; Qide Dong; Haoran Cheng; Weiguo Feng; Shilei Wen; Bingyue Peng; Lei Zhang
>
> **摘要:** Diffusion models have shown impressive performance in many visual generation and manipulation tasks. Many existing methods focus on training a model for a specific task, especially, text-to-video (T2V) generation, while many other works focus on finetuning the pretrained T2V model for image-to-video (I2V), video-to-video (V2V), image and video manipulation tasks, etc. However, training a strong T2V foundation model requires a large amount of high-quality annotations, which is very costly. In addition, many existing models can perform only one or several tasks. In this work, we introduce a unified framework, namely many-for-many, which leverages the available training data from many different visual generation and manipulation tasks to train a single model for those different tasks. Specifically, we design a lightweight adapter to unify the different conditions in different tasks, then employ a joint image-video learning strategy to progressively train the model from scratch. Our joint learning leads to a unified visual generation and manipulation model with improved video generation performance. In addition, we introduce depth maps as a condition to help our model better perceive the 3D space in visual generation. Two versions of our model are trained with different model sizes (8B and 2B), each of which can perform more than 10 different tasks. In particular, our 8B model demonstrates highly competitive performance in video generation tasks compared to open-source and even commercial engines. Our models and source codes are available at https://github.com/leeruibin/MfM.git.
>
---
#### [new 098] Abstractive Visual Understanding of Multi-modal Structured Knowledge: A New Perspective for MLLM Evaluation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态大语言模型（MLLM）评估任务，旨在解决现有基准测试忽视MLLM对结构化抽象视觉知识理解能力的问题。作者提出了新评估范式M3STR，基于多模态知识图谱生成含复杂关系的合成图像，要求MLLM识别实体并解析其拓扑关系，揭示了当前模型在该能力上的不足。**

- **链接: [http://arxiv.org/pdf/2506.01293v1](http://arxiv.org/pdf/2506.01293v1)**

> **作者:** Yichi Zhang; Zhuo Chen; Lingbing Guo; Yajing Xu; Min Zhang; Wen Zhang; Huajun Chen
>
> **备注:** Work in progress
>
> **摘要:** Multi-modal large language models (MLLMs) incorporate heterogeneous modalities into LLMs, enabling a comprehensive understanding of diverse scenarios and objects. Despite the proliferation of evaluation benchmarks and leaderboards for MLLMs, they predominantly overlook the critical capacity of MLLMs to comprehend world knowledge with structured abstractions that appear in visual form. To address this gap, we propose a novel evaluation paradigm and devise M3STR, an innovative benchmark grounded in the Multi-Modal Map for STRuctured understanding. This benchmark leverages multi-modal knowledge graphs to synthesize images encapsulating subgraph architectures enriched with multi-modal entities. M3STR necessitates that MLLMs not only recognize the multi-modal entities within the visual inputs but also decipher intricate relational topologies among them. We delineate the benchmark's statistical profiles and automated construction pipeline, accompanied by an extensive empirical analysis of 26 state-of-the-art MLLMs. Our findings reveal persistent deficiencies in processing abstractive visual information with structured knowledge, thereby charting a pivotal trajectory for advancing MLLMs' holistic reasoning capacities. Our code and data are released at https://github.com/zjukg/M3STR
>
---
#### [new 099] CLIP-driven rain perception: Adaptive deraining with pattern-aware network routing and mask-guided cross-attention
- **分类: cs.CV**

- **简介: 该论文属于图像去雨任务，旨在解决单一网络难以处理多样雨纹的问题。论文提出CLIP-RPN模型，利用CLIP感知雨纹并动态路由子网络处理不同雨况，同时引入MGCA机制和DLS损失调度方法，提升去雨效果，尤其在复杂混合数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2506.01366v1](http://arxiv.org/pdf/2506.01366v1)**

> **作者:** Cong Guan; Osamu Yoshie
>
> **摘要:** Existing deraining models process all rainy images within a single network. However, different rain patterns have significant variations, which makes it challenging for a single network to handle diverse types of raindrops and streaks. To address this limitation, we propose a novel CLIP-driven rain perception network (CLIP-RPN) that leverages CLIP to automatically perceive rain patterns by computing visual-language matching scores and adaptively routing to sub-networks to handle different rain patterns, such as varying raindrop densities, streak orientations, and rainfall intensity. CLIP-RPN establishes semantic-aware rain pattern recognition through CLIP's cross-modal visual-language alignment capabilities, enabling automatic identification of precipitation characteristics across different rain scenarios. This rain pattern awareness drives an adaptive subnetwork routing mechanism where specialized processing branches are dynamically activated based on the detected rain type, significantly enhancing the model's capacity to handle diverse rainfall conditions. Furthermore, within sub-networks of CLIP-RPN, we introduce a mask-guided cross-attention mechanism (MGCA) that predicts precise rain masks at multi-scale to facilitate contextual interactions between rainy regions and clean background areas by cross-attention. We also introduces a dynamic loss scheduling mechanism (DLS) to adaptively adjust the gradients for the optimization process of CLIP-RPN. Compared with the commonly used $l_1$ or $l_2$ loss, DLS is more compatible with the inherent dynamics of the network training process, thus achieving enhanced outcomes. Our method achieves state-of-the-art performance across multiple datasets, particularly excelling in complex mixed datasets.
>
---
#### [new 100] AceVFI: A Comprehensive Survey of Advances in Video Frame Interpolation
- **分类: cs.CV**

- **简介: 该论文属于视频帧插值（VFI）任务，旨在生成视频中缺失的中间帧以保持时空连贯性。论文系统综述了基于深度学习的多种方法，包括核、流、混合、GAN、Transformer等模型，并提出AceVFI分类框架与挑战分析，覆盖250余篇相关研究，为领域提供全面参考。**

- **链接: [http://arxiv.org/pdf/2506.01061v1](http://arxiv.org/pdf/2506.01061v1)**

> **作者:** Dahyeon Kye; Changhyun Roh; Sukhun Ko; Chanho Eom; Jihyong Oh
>
> **备注:** Please visit our project page at https://github.com/CMLab-Korea/Awesome-Video-Frame-Interpolation
>
> **摘要:** Video Frame Interpolation (VFI) is a fundamental Low-Level Vision (LLV) task that synthesizes intermediate frames between existing ones while maintaining spatial and temporal coherence. VFI techniques have evolved from classical motion compensation-based approach to deep learning-based approach, including kernel-, flow-, hybrid-, phase-, GAN-, Transformer-, Mamba-, and more recently diffusion model-based approach. We introduce AceVFI, the most comprehensive survey on VFI to date, covering over 250+ papers across these approaches. We systematically organize and describe VFI methodologies, detailing the core principles, design assumptions, and technical characteristics of each approach. We categorize the learning paradigm of VFI methods namely, Center-Time Frame Interpolation (CTFI) and Arbitrary-Time Frame Interpolation (ATFI). We analyze key challenges of VFI such as large motion, occlusion, lighting variation, and non-linear motion. In addition, we review standard datasets, loss functions, evaluation metrics. We examine applications of VFI including event-based, cartoon, medical image VFI and joint VFI with other LLV tasks. We conclude by outlining promising future research directions to support continued progress in the field. This survey aims to serve as a unified reference for both newcomers and experts seeking a deep understanding of modern VFI landscapes.
>
---
#### [new 101] SteerPose: Simultaneous Extrinsic Camera Calibration and Matching from Articulation
- **分类: cs.CV**

- **简介: 该论文属于多视角相机标定与姿态估计任务，旨在解决无需专用标定物、仅通过自由运动的人类或动物进行相机参数标定和跨视角匹配的问题。作者提出了SteerPose方法，通过神经网络实现2D姿态旋转与匹配，结合几何一致性损失，统一框架下完成标定与对应点搜索，并可推广到新物种的3D姿态重建。**

- **链接: [http://arxiv.org/pdf/2506.01691v1](http://arxiv.org/pdf/2506.01691v1)**

> **作者:** Sang-Eun Lee; Ko Nishino; Shohei Nobuhara
>
> **备注:** 13 pages
>
> **摘要:** Can freely moving humans or animals themselves serve as calibration targets for multi-camera systems while simultaneously estimating their correspondences across views? We humans can solve this problem by mentally rotating the observed 2D poses and aligning them with those in the target views. Inspired by this cognitive ability, we propose SteerPose, a neural network that performs this rotation of 2D poses into another view. By integrating differentiable matching, SteerPose simultaneously performs extrinsic camera calibration and correspondence search within a single unified framework. We also introduce a novel geometric consistency loss that explicitly ensures that the estimated rotation and correspondences result in a valid translation estimation. Experimental results on diverse in-the-wild datasets of humans and animals validate the effectiveness and robustness of the proposed method. Furthermore, we demonstrate that our method can reconstruct the 3D poses of novel animals in multi-camera setups by leveraging off-the-shelf 2D pose estimators and our class-agnostic model.
>
---
#### [new 102] CAPAA: Classifier-Agnostic Projector-Based Adversarial Attack
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于对抗攻击任务，旨在解决投影式对抗攻击在多分类器和不同相机姿态下效果差的问题。作者提出了CAPAA方法，通过设计分类器无关的损失函数和注意力机制，提升攻击成功率与隐蔽性。**

- **链接: [http://arxiv.org/pdf/2506.00978v1](http://arxiv.org/pdf/2506.00978v1)**

> **作者:** Zhan Li; Mingyu Zhao; Xin Dong; Haibin Ling; Bingyao Huang
>
> **摘要:** Projector-based adversarial attack aims to project carefully designed light patterns (i.e., adversarial projections) onto scenes to deceive deep image classifiers. It has potential applications in privacy protection and the development of more robust classifiers. However, existing approaches primarily focus on individual classifiers and fixed camera poses, often neglecting the complexities of multi-classifier systems and scenarios with varying camera poses. This limitation reduces their effectiveness when introducing new classifiers or camera poses. In this paper, we introduce Classifier-Agnostic Projector-Based Adversarial Attack (CAPAA) to address these issues. First, we develop a novel classifier-agnostic adversarial loss and optimization framework that aggregates adversarial and stealthiness loss gradients from multiple classifiers. Then, we propose an attention-based gradient weighting mechanism that concentrates perturbations on regions of high classification activation, thereby improving the robustness of adversarial projections when applied to scenes with varying camera poses. Our extensive experimental evaluations demonstrate that CAPAA achieves both a higher attack success rate and greater stealthiness compared to existing baselines. Codes are available at: https://github.com/ZhanLiQxQ/CAPAA.
>
---
#### [new 103] BAGNet: A Boundary-Aware Graph Attention Network for 3D Point Cloud Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D点云语义分割任务，旨在解决点云数据不规则、计算复杂的问题。作者提出了BAGNet，通过边界感知图注意力层（BAGLayer）和轻量注意力池化层，提升分割精度并减少计算时间。**

- **链接: [http://arxiv.org/pdf/2506.00475v1](http://arxiv.org/pdf/2506.00475v1)**

> **作者:** Wei Tao; Xiaoyang Qu; Kai Lu; Jiguang Wan; Shenglin He; Jianzong Wang
>
> **备注:** Accepted by the 2025 International Joint Conference on Neural Networks (IJCNN 2025)
>
> **摘要:** Since the point cloud data is inherently irregular and unstructured, point cloud semantic segmentation has always been a challenging task. The graph-based method attempts to model the irregular point cloud by representing it as a graph; however, this approach incurs substantial computational cost due to the necessity of constructing a graph for every point within a large-scale point cloud. In this paper, we observe that boundary points possess more intricate spatial structural information and develop a novel graph attention network known as the Boundary-Aware Graph attention Network (BAGNet). On one hand, BAGNet contains a boundary-aware graph attention layer (BAGLayer), which employs edge vertex fusion and attention coefficients to capture features of boundary points, reducing the computation time. On the other hand, BAGNet employs a lightweight attention pooling layer to extract the global feature of the point cloud to maintain model accuracy. Extensive experiments on standard datasets demonstrate that BAGNet outperforms state-of-the-art methods in point cloud semantic segmentation with higher accuracy and less inference time.
>
---
#### [new 104] Towards Effective and Efficient Adversarial Defense with Diffusion Models for Robust Visual Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉跟踪任务，旨在解决对抗攻击下跟踪性能下降的问题。工作提出了一种基于扩散模型的防御方法DiffDf，结合多尺度机制和多种损失函数，有效抑制对抗扰动，提升跟踪鲁棒性与效率。**

- **链接: [http://arxiv.org/pdf/2506.00325v1](http://arxiv.org/pdf/2506.00325v1)**

> **作者:** Long Xu; Peng Gao; Wen-Jia Tang; Fei Wang; Ru-Yue Yuan
>
> **摘要:** Although deep learning-based visual tracking methods have made significant progress, they exhibit vulnerabilities when facing carefully designed adversarial attacks, which can lead to a sharp decline in tracking performance. To address this issue, this paper proposes for the first time a novel adversarial defense method based on denoise diffusion probabilistic models, termed DiffDf, aimed at effectively improving the robustness of existing visual tracking methods against adversarial attacks. DiffDf establishes a multi-scale defense mechanism by combining pixel-level reconstruction loss, semantic consistency loss, and structural similarity loss, effectively suppressing adversarial perturbations through a gradual denoising process. Extensive experimental results on several mainstream datasets show that the DiffDf method demonstrates excellent generalization performance for trackers with different architectures, significantly improving various evaluation metrics while achieving real-time inference speeds of over 30 FPS, showcasing outstanding defense performance and efficiency. Codes are available at https://github.com/pgao-lab/DiffDf.
>
---
#### [new 105] SemiVT-Surge: Semi-Supervised Video Transformer for Surgical Phase Recognition
- **分类: cs.CV**

- **简介: 该论文属于手术阶段识别任务，旨在解决标注数据稀缺的问题。作者提出了一种半监督视频Transformer模型（SemiVT-Surge），结合伪标签框架、时间一致性正则化和对比学习，利用少量标注数据与未标注数据提升识别性能。实验表明其在RAMIE和Cholec80数据集上表现优异，显著减少了对手工标注的依赖。**

- **链接: [http://arxiv.org/pdf/2506.01471v1](http://arxiv.org/pdf/2506.01471v1)**

> **作者:** Yiping Li; Ronald de Jong; Sahar Nasirihaghighi; Tim Jaspers; Romy van Jaarsveld; Gino Kuiper; Richard van Hillegersberg; Fons van der Sommen; Jelle Ruurda; Marcel Breeuwer; Yasmina Al Khalil
>
> **备注:** Accepted for MICCAI 2025
>
> **摘要:** Accurate surgical phase recognition is crucial for computer-assisted interventions and surgical video analysis. Annotating long surgical videos is labor-intensive, driving research toward leveraging unlabeled data for strong performance with minimal annotations. Although self-supervised learning has gained popularity by enabling large-scale pretraining followed by fine-tuning on small labeled subsets, semi-supervised approaches remain largely underexplored in the surgical domain. In this work, we propose a video transformer-based model with a robust pseudo-labeling framework. Our method incorporates temporal consistency regularization for unlabeled data and contrastive learning with class prototypes, which leverages both labeled data and pseudo-labels to refine the feature space. Through extensive experiments on the private RAMIE (Robot-Assisted Minimally Invasive Esophagectomy) dataset and the public Cholec80 dataset, we demonstrate the effectiveness of our approach. By incorporating unlabeled data, we achieve state-of-the-art performance on RAMIE with a 4.9% accuracy increase and obtain comparable results to full supervision while using only 1/4 of the labeled data on Cholec80. Our findings establish a strong benchmark for semi-supervised surgical phase recognition, paving the way for future research in this domain.
>
---
#### [new 106] Feature Fusion and Knowledge-Distilled Multi-Modal Multi-Target Detection
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于多目标检测任务，旨在解决异构数据源下资源受限设备的多目标检测难题。通过融合RGB与热成像特征，并引入知识蒸馏提升域适应性，实现精度优化。实验表明其方法在保持高精度的同时显著降低推理时间。**

- **链接: [http://arxiv.org/pdf/2506.00365v1](http://arxiv.org/pdf/2506.00365v1)**

> **作者:** Ngoc Tuyen Do; Tri Nhu Do
>
> **摘要:** In the surveillance and defense domain, multi-target detection and classification (MTD) is considered essential yet challenging due to heterogeneous inputs from diverse data sources and the computational complexity of algorithms designed for resource-constrained embedded devices, particularly for Al-based solutions. To address these challenges, we propose a feature fusion and knowledge-distilled framework for multi-modal MTD that leverages data fusion to enhance accuracy and employs knowledge distillation for improved domain adaptation. Specifically, our approach utilizes both RGB and thermal image inputs within a novel fusion-based multi-modal model, coupled with a distillation training pipeline. We formulate the problem as a posterior probability optimization task, which is solved through a multi-stage training pipeline supported by a composite loss function. This loss function effectively transfers knowledge from a teacher model to a student model. Experimental results demonstrate that our student model achieves approximately 95% of the teacher model's mean Average Precision while reducing inference time by approximately 50%, underscoring its suitability for practical MTD deployment scenarios.
>
---
#### [new 107] Keystep Recognition using Graph Neural Networks
- **分类: cs.CV**

- **简介: 该论文属于节点分类任务，旨在解决第一视角视频中精细步骤识别问题。作者提出GLEVR框架，利用图神经网络构建稀疏图模型，融合第三人称视角视频和自动字幕作为额外节点，有效捕捉长期依赖关系，提升了识别性能。**

- **链接: [http://arxiv.org/pdf/2506.01102v1](http://arxiv.org/pdf/2506.01102v1)**

> **作者:** Julia Lee Romero; Kyle Min; Subarna Tripathi; Morteza Karimzadeh
>
> **摘要:** We pose keystep recognition as a node classification task, and propose a flexible graph-learning framework for fine-grained keystep recognition that is able to effectively leverage long-term dependencies in egocentric videos. Our approach, termed GLEVR, consists of constructing a graph where each video clip of the egocentric video corresponds to a node. The constructed graphs are sparse and computationally efficient, outperforming existing larger models substantially. We further leverage alignment between egocentric and exocentric videos during training for improved inference on egocentric videos, as well as adding automatic captioning as an additional modality. We consider each clip of each exocentric video (if available) or video captions as additional nodes during training. We examine several strategies to define connections across these nodes. We perform extensive experiments on the Ego-Exo4D dataset and show that our proposed flexible graph-based framework notably outperforms existing methods.
>
---
#### [new 108] IMAGHarmony: Controllable Image Editing with Consistent Object Quantity and Layout
- **分类: cs.CV**

- **简介: 该论文提出IMAGHarmony，属于图像编辑任务，旨在解决多物体场景下对象数量与布局的一致性控制难题。通过引入结构感知框架和偏好引导的噪声选择策略，实现对复杂场景中物体数量和空间结构的精细编辑，提升了生成稳定性和布局一致性。**

- **链接: [http://arxiv.org/pdf/2506.01949v1](http://arxiv.org/pdf/2506.01949v1)**

> **作者:** Fei Shen; Xiaoyu Du; Yutong Gao; Jian Yu; Yushe Cao; Xing Lei; Jinhui Tang
>
> **摘要:** Recent diffusion models have advanced image editing by enhancing visual quality and control, supporting broad applications across creative and personalized domains. However, current image editing largely overlooks multi-object scenarios, where precise control over object categories, counts, and spatial layouts remains a significant challenge. To address this, we introduce a new task, quantity-and-layout consistent image editing (QL-Edit), which aims to enable fine-grained control of object quantity and spatial structure in complex scenes. We further propose IMAGHarmony, a structure-aware framework that incorporates harmony-aware attention (HA) to integrate multimodal semantics, explicitly modeling object counts and layouts to enhance editing accuracy and structural consistency. In addition, we observe that diffusion models are susceptible to initial noise and exhibit strong preferences for specific noise patterns. Motivated by this, we present a preference-guided noise selection (PNS) strategy that chooses semantically aligned initial noise samples based on vision-language matching, thereby improving generation stability and layout consistency in multi-object editing. To support evaluation, we construct HarmonyBench, a comprehensive benchmark covering diverse quantity and layout control scenarios. Extensive experiments demonstrate that IMAGHarmony consistently outperforms state-of-the-art methods in structural alignment and semantic accuracy. The code and model are available at https://github.com/muzishen/IMAGHarmony.
>
---
#### [new 109] Data Pruning by Information Maximization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于数据修剪（coreset selection）任务，旨在解决如何从大规模数据集中选取最具信息量且非冗余的子集问题。作者提出了InfoMax方法，通过最大化个体样本的信息得分并最小化样本间冗余来提升子集的整体信息量。方法上，将问题建模为离散二次规划问题，并设计高效求解策略，使算法可扩展至百万级数据。实验表明其在多种任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.01701v1](http://arxiv.org/pdf/2506.01701v1)**

> **作者:** Haoru Tan; Sitong Wu; Wei Huang; Shizhen Zhao; Xiaojuan Qi
>
> **备注:** ICLR 2025
>
> **摘要:** In this paper, we present InfoMax, a novel data pruning method, also known as coreset selection, designed to maximize the information content of selected samples while minimizing redundancy. By doing so, InfoMax enhances the overall informativeness of the coreset. The information of individual samples is measured by importance scores, which capture their influence or difficulty in model learning. To quantify redundancy, we use pairwise sample similarities, based on the premise that similar samples contribute similarly to the learning process. We formalize the coreset selection problem as a discrete quadratic programming (DQP) task, with the objective of maximizing the total information content, represented as the sum of individual sample contributions minus the redundancies introduced by similar samples within the coreset. To ensure practical scalability, we introduce an efficient gradient-based solver, complemented by sparsification techniques applied to the similarity matrix and dataset partitioning strategies. This enables InfoMax to seamlessly scale to datasets with millions of samples. Extensive experiments demonstrate the superior performance of InfoMax in various data pruning tasks, including image classification, vision-language pre-training, and instruction tuning for large language models.
>
---
#### [new 110] Aiding Medical Diagnosis through Image Synthesis and Classification
- **分类: cs.CV**

- **简介: 该论文属于医学图像生成与分类任务，旨在解决医疗资源多样性不足和获取困难的问题。通过微调扩散模型生成病理图像，并用分类模型验证质量，实现高精度图像合成，辅助医学诊断与教学。**

- **链接: [http://arxiv.org/pdf/2506.00786v1](http://arxiv.org/pdf/2506.00786v1)**

> **作者:** Kanishk Choudhary
>
> **备注:** 8 pages, 6 figures. Under review
>
> **摘要:** Medical professionals, especially those in training, often depend on visual reference materials to support an accurate diagnosis and develop pattern recognition skills. However, existing resources may lack the diversity and accessibility needed for broad and effective clinical learning. This paper presents a system designed to generate realistic medical images from textual descriptions and validate their accuracy through a classification model. A pretrained stable diffusion model was fine-tuned using Low-Rank Adaptation (LoRA) on the PathMNIST dataset, consisting of nine colorectal histopathology tissue types. The generative model was trained multiple times using different training parameter configurations, guided by domain-specific prompts to capture meaningful features. To ensure quality control, a ResNet-18 classification model was trained on the same dataset, achieving 99.76% accuracy in detecting the correct label of a colorectal histopathological medical image. Generated images were then filtered using the trained classifier and an iterative process, where inaccurate outputs were discarded and regenerated until they were correctly classified. The highest performing version of the generative model from experimentation achieved an F1 score of 0.6727, with precision and recall scores of 0.6817 and 0.7111, respectively. Some types of tissue, such as adipose tissue and lymphocytes, reached perfect classification scores, while others proved more challenging due to structural complexity. The self-validating approach created demonstrates a reliable method for synthesizing domain-specific medical images because of high accuracy in both the generation and classification portions of the system, with potential applications in both diagnostic support and clinical education. Future work includes improving prompt-specific accuracy and extending the system to other areas of medical imaging.
>
---
#### [new 111] IVY-FAKE: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像与视频内容真实性检测任务，旨在解决现有AIGC检测方法缺乏可解释性和统一框架的问题。作者构建了大规模多模态数据集IVY-FAKE，并提出了可解释的统一检测模型IVY-XDETECTOR，实现图像和视频的联合检测与解释。**

- **链接: [http://arxiv.org/pdf/2506.00979v1](http://arxiv.org/pdf/2506.00979v1)**

> **作者:** Wayne Zhang; Changjiang Jiang; Zhonghao Zhang; Chenyang Si; Fengchang Yu; Wei Peng
>
> **备注:** 20pages,13figures,7 tables
>
> **摘要:** The rapid advancement of Artificial Intelligence Generated Content (AIGC) in visual domains has resulted in highly realistic synthetic images and videos, driven by sophisticated generative frameworks such as diffusion-based architectures. While these breakthroughs open substantial opportunities, they simultaneously raise critical concerns about content authenticity and integrity. Many current AIGC detection methods operate as black-box binary classifiers, which offer limited interpretability, and no approach supports detecting both images and videos in a unified framework. This dual limitation compromises model transparency, reduces trustworthiness, and hinders practical deployment. To address these challenges, we introduce IVY-FAKE , a novel, unified, and large-scale dataset specifically designed for explainable multimodal AIGC detection. Unlike prior benchmarks, which suffer from fragmented modality coverage and sparse annotations, IVY-FAKE contains over 150,000 richly annotated training samples (images and videos) and 18,700 evaluation examples, each accompanied by detailed natural-language reasoning beyond simple binary labels. Building on this, we propose Ivy Explainable Detector (IVY-XDETECTOR), a unified AIGC detection and explainable architecture that jointly performs explainable detection for both image and video content. Our unified vision-language model achieves state-of-the-art performance across multiple image and video detection benchmarks, highlighting the significant advancements enabled by our dataset and modeling framework. Our data is publicly available at https://huggingface.co/datasets/AI-Safeguard/Ivy-Fake.
>
---
#### [new 112] ArtiScene: Language-Driven Artistic 3D Scene Generation Through Image Intermediary
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D场景生成任务，旨在解决传统方法需专业技能及数据不足导致效果受限的问题。作者提出ArtiScene，利用文本生成2D图像作为中介，指导高质量3D场景合成，无需训练，提升布局与美学质量，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2506.00742v1](http://arxiv.org/pdf/2506.00742v1)**

> **作者:** Zeqi Gu; Yin Cui; Zhaoshuo Li; Fangyin Wei; Yunhao Ge; Jinwei Gu; Ming-Yu Liu; Abe Davis; Yifan Ding
>
> **备注:** Accepted by CVPR
>
> **摘要:** Designing 3D scenes is traditionally a challenging task that demands both artistic expertise and proficiency with complex software. Recent advances in text-to-3D generation have greatly simplified this process by letting users create scenes based on simple text descriptions. However, as these methods generally require extra training or in-context learning, their performance is often hindered by the limited availability of high-quality 3D data. In contrast, modern text-to-image models learned from web-scale images can generate scenes with diverse, reliable spatial layouts and consistent, visually appealing styles. Our key insight is that instead of learning directly from 3D scenes, we can leverage generated 2D images as an intermediary to guide 3D synthesis. In light of this, we introduce ArtiScene, a training-free automated pipeline for scene design that integrates the flexibility of free-form text-to-image generation with the diversity and reliability of 2D intermediary layouts. First, we generate 2D images from a scene description, then extract the shape and appearance of objects to create 3D models. These models are assembled into the final scene using geometry, position, and pose information derived from the same intermediary image. Being generalizable to a wide range of scenes and styles, ArtiScene outperforms state-of-the-art benchmarks by a large margin in layout and aesthetic quality by quantitative metrics. It also averages a 74.89% winning rate in extensive user studies and 95.07% in GPT-4o evaluation. Project page: https://artiscene-cvpr.github.io/
>
---
#### [new 113] UNSURF: Uncertainty Quantification for Cortical Surface Reconstruction of Clinical Brain MRIs
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决临床脑MRI皮层表面重建中的不确定性建模问题。作者提出了UNSURF方法，通过预测与实际符号距离函数的差异来量化不确定性，并验证其在质量控制和阿尔茨海默病分类中的有效性。**

- **链接: [http://arxiv.org/pdf/2506.00498v1](http://arxiv.org/pdf/2506.00498v1)**

> **作者:** Raghav Mehta; Karthik Gopinath; Ben Glocker; Juan Eugenio Iglesias
>
> **备注:** Raghav Mehta and Karthik Gopinath contributed equally. Ben Glocker and Juan Eugenio Iglesias contributed equally. Paper under review at MICCAI 2025
>
> **摘要:** We propose UNSURF, a novel uncertainty measure for cortical surface reconstruction of clinical brain MRI scans of any orientation, resolution, and contrast. It relies on the discrepancy between predicted voxel-wise signed distance functions (SDFs) and the actual SDFs of the fitted surfaces. Our experiments on real clinical scans show that traditional uncertainty measures, such as voxel-wise Monte Carlo variance, are not suitable for modeling the uncertainty of surface placement. Our results demonstrate that UNSURF estimates correlate well with the ground truth errors and: \textit{(i)}~enable effective automated quality control of surface reconstructions at the subject-, parcel-, mesh node-level; and \textit{(ii)}~improve performance on a downstream Alzheimer's disease classification task.
>
---
#### [new 114] Reinforcement Learning Tuning for VideoLLMs: Reward Design and Data Efficiency
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在提升多模态大语言模型（MLLMs）在复杂语义和长时序依赖视频中的推理能力。通过引入双奖励机制与方差感知数据选择策略，结合强化学习调优（RLT），有效提升模型性能，仅需更少训练数据即可超越现有方法。**

- **链接: [http://arxiv.org/pdf/2506.01908v1](http://arxiv.org/pdf/2506.01908v1)**

> **作者:** Hongyu Li; Songhao Han; Yue Liao; Junfeng Luo; Jialin Gao; Shuicheng Yan; Si Liu
>
> **摘要:** Understanding real-world videos with complex semantics and long temporal dependencies remains a fundamental challenge in computer vision. Recent progress in multimodal large language models (MLLMs) has demonstrated strong capabilities in vision-language tasks, while reinforcement learning tuning (RLT) has further improved their reasoning abilities. In this work, we explore RLT as a post-training strategy to enhance the video-specific reasoning capabilities of MLLMs. Built upon the Group Relative Policy Optimization (GRPO) framework, we propose a dual-reward formulation that supervises both semantic and temporal reasoning through discrete and continuous reward signals. To facilitate effective preference-based optimization, we introduce a variance-aware data selection strategy based on repeated inference to identify samples that provide informative learning signals. We evaluate our approach across eight representative video understanding tasks, including VideoQA, Temporal Video Grounding, and Grounded VideoQA. Our method consistently outperforms supervised fine-tuning and existing RLT baselines, achieving superior performance with significantly less training data. These results underscore the importance of reward design and data selection in advancing reasoning-centric video understanding with MLLMs. Notably, The initial code release (two months ago) has now been expanded with updates, including optimized reward mechanisms and additional datasets. The latest version is available at https://github.com/appletea233/Temporal-R1 .
>
---
#### [new 115] Elucidating the representation of images within an unconditional diffusion model denoiser
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在理解扩散模型中去噪网络的内部机制。作者分析了一个用于ImageNet去噪的UNet模型，发现其中间层能提取稀疏、非线性的图像表示，并提出一种随机重建算法，可从该表示恢复图像。研究还表明，该表示空间中的欧氏距离反映图像语义相似性和条件密度差异，聚类结果揭示了图像间细粒度和全局结构的关联。**

- **链接: [http://arxiv.org/pdf/2506.01912v1](http://arxiv.org/pdf/2506.01912v1)**

> **作者:** Zahra Kadkhodaie; Stéphane Mallat; Eero Simoncelli
>
> **摘要:** Generative diffusion models learn probability densities over diverse image datasets by estimating the score with a neural network trained to remove noise. Despite their remarkable success in generating high-quality images, the internal mechanisms of the underlying score networks are not well understood. Here, we examine a UNet trained for denoising on the ImageNet dataset, to better understand its internal representation and computation of the score. We show that the middle block of the UNet decomposes individual images into sparse subsets of active channels, and that the vector of spatial averages of these channels can provide a nonlinear representation of the underlying clean images. We develop a novel algorithm for stochastic reconstruction of images from this representation and demonstrate that it recovers a sample from a set of images defined by a target image representation. We then study the properties of the representation and demonstrate that Euclidean distances in the latent space correspond to distances between conditional densities induced by representations as well as semantic similarities in the image space. Applying a clustering algorithm in the representation space yields groups of images that share both fine details (e.g., specialized features, textured regions, small objects), as well as global structure, but are only partially aligned with object identities. Thus, we show for the first time that a network trained solely on denoising contains a rich and accessible sparse representation of images.
>
---
#### [new 116] ZeShot-VQA: Zero-Shot Visual Question Answering Framework with Answer Mapping for Natural Disaster Damage Assessment
- **分类: cs.CV; cs.CL; cs.IR; cs.LG; I.2.7; I.2.10; I.5.1**

- **简介: 该论文属于视觉问答（VQA）任务，旨在解决自然灾害灾后评估中模型无法回答开放性问题及需重新训练的问题。作者提出了一种基于大规模视觉语言模型（VLM）的零样本VQA方法ZeShot-VQA，可在无需微调的情况下处理新数据集并生成未见过的答案，提升了灵活性与实用性。**

- **链接: [http://arxiv.org/pdf/2506.00238v1](http://arxiv.org/pdf/2506.00238v1)**

> **作者:** Ehsan Karimi; Maryam Rahnemoonfar
>
> **备注:** Accepted by the 2025 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025)
>
> **摘要:** Natural disasters usually affect vast areas and devastate infrastructures. Performing a timely and efficient response is crucial to minimize the impact on affected communities, and data-driven approaches are the best choice. Visual question answering (VQA) models help management teams to achieve in-depth understanding of damages. However, recently published models do not possess the ability to answer open-ended questions and only select the best answer among a predefined list of answers. If we want to ask questions with new additional possible answers that do not exist in the predefined list, the model needs to be fin-tuned/retrained on a new collected and annotated dataset, which is a time-consuming procedure. In recent years, large-scale Vision-Language Models (VLMs) have earned significant attention. These models are trained on extensive datasets and demonstrate strong performance on both unimodal and multimodal vision/language downstream tasks, often without the need for fine-tuning. In this paper, we propose a VLM-based zero-shot VQA (ZeShot-VQA) method, and investigate the performance of on post-disaster FloodNet dataset. Since the proposed method takes advantage of zero-shot learning, it can be applied on new datasets without fine-tuning. In addition, ZeShot-VQA is able to process and generate answers that has been not seen during the training procedure, which demonstrates its flexibility.
>
---
#### [new 117] XYZ-IBD: High-precision Bin-picking Dataset for Object 6D Pose Estimation Capturing Real-world Industrial Complexity
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与机器人领域任务，旨在解决工业场景中物体6D姿态估计难题。针对现有数据集偏向家用物品、难以反映真实工业复杂性的问题，作者构建了XYZ-IBD数据集，包含15种无纹理、金属、对称物体，在高密度杂乱场景中进行多视角真实与合成数据采集，并提供毫米级精确标注，以推动更具挑战性的研究方向。**

- **链接: [http://arxiv.org/pdf/2506.00599v1](http://arxiv.org/pdf/2506.00599v1)**

> **作者:** Junwen Huang; Jizhong Liang; Jiaqi Hu; Martin Sundermeyer; Peter KT Yu; Nassir Navab; Benjamin Busam
>
> **摘要:** We introduce XYZ-IBD, a bin-picking dataset for 6D pose estimation that captures real-world industrial complexity, including challenging object geometries, reflective materials, severe occlusions, and dense clutter. The dataset reflects authentic robotic manipulation scenarios with millimeter-accurate annotations. Unlike existing datasets that primarily focus on household objects, which approach saturation,XYZ-IBD represents the unsolved realistic industrial conditions. The dataset features 15 texture-less, metallic, and mostly symmetrical objects of varying shapes and sizes. These objects are heavily occluded and randomly arranged in bins with high density, replicating the challenges of real-world bin-picking. XYZ-IBD was collected using two high-precision industrial cameras and one commercially available camera, providing RGB, grayscale, and depth images. It contains 75 multi-view real-world scenes, along with a large-scale synthetic dataset rendered under simulated bin-picking conditions. We employ a meticulous annotation pipeline that includes anti-reflection spray, multi-view depth fusion, and semi-automatic annotation, achieving millimeter-level pose labeling accuracy required for industrial manipulation. Quantification in simulated environments confirms the reliability of the ground-truth annotations. We benchmark state-of-the-art methods on 2D detection, 6D pose estimation, and depth estimation tasks on our dataset, revealing significant performance degradation in our setups compared to current academic household benchmarks. By capturing the complexity of real-world bin-picking scenarios, XYZ-IBD introduces more realistic and challenging problems for future research. The dataset and benchmark are publicly available at https://xyz-ibd.github.io/XYZ-IBD/.
>
---
#### [new 118] 3D Trajectory Reconstruction of Moving Points Based on Asynchronous Cameras
- **分类: cs.CV**

- **简介: 该论文属于三维轨迹重建任务，旨在解决异步相机下的运动点定位问题。现有方法通常仅处理轨迹重建或相机同步之一，而该文提出一种同时优化轨迹、相机时间信息及旋转参数的方法，提高了重建精度，尤其适用于相机旋转不准的情况。**

- **链接: [http://arxiv.org/pdf/2506.00541v1](http://arxiv.org/pdf/2506.00541v1)**

> **作者:** Huayu Huang; Banglei Guan; Yang Shang; Qifeng Yu
>
> **备注:** This paper has been accepted by Acta Mechanica Sinica
>
> **摘要:** Photomechanics is a crucial branch of solid mechanics. The localization of point targets constitutes a fundamental problem in optical experimental mechanics, with extensive applications in various missions of UAVs. Localizing moving targets is crucial for analyzing their motion characteristics and dynamic properties. Reconstructing the trajectories of points from asynchronous cameras is a significant challenge. It encompasses two coupled sub-problems: trajectory reconstruction and camera synchronization. Present methods typically address only one of these sub-problems individually. This paper proposes a 3D trajectory reconstruction method for point targets based on asynchronous cameras, simultaneously solving both sub-problems. Firstly, we extend the trajectory intersection method to asynchronous cameras to resolve the limitation of traditional triangulation that requires camera synchronization. Secondly, we develop models for camera temporal information and target motion, based on imaging mechanisms and target dynamics characteristics. The parameters are optimized simultaneously to achieve trajectory reconstruction without accurate time parameters. Thirdly, we optimize the camera rotations alongside the camera time information and target motion parameters, using tighter and more continuous constraints on moving points. The reconstruction accuracy is significantly improved, especially when the camera rotations are inaccurate. Finally, the simulated and real-world experimental results demonstrate the feasibility and accuracy of the proposed method. The real-world results indicate that the proposed algorithm achieved a localization error of 112.95 m at an observation range of 15 ~ 20 km.
>
---
#### [new 119] SatDreamer360: Geometry Consistent Street-View Video Generation from Satellite Imagery
- **分类: cs.CV**

- **简介: 该论文属于街景视频生成任务，旨在从卫星图像生成几何一致的地面视角视频。现有方法难以保证时间连续性和几何一致性，作者提出SatDreamer360框架，利用三平面表示和注意力机制实现跨视角特征对齐与视频序列一致性建模，并构建了用于评估的大规模数据集VIGOR++。**

- **链接: [http://arxiv.org/pdf/2506.00600v1](http://arxiv.org/pdf/2506.00600v1)**

> **作者:** Xianghui Ze; Beiyi Zhu; Zhenbo Song; Jianfeng Lu; Yujiao Shi
>
> **摘要:** Generating continuous ground-level video from satellite imagery is a challenging task with significant potential for applications in simulation, autonomous navigation, and digital twin cities. Existing approaches primarily focus on synthesizing individual ground-view images, often relying on auxiliary inputs like height maps or handcrafted projections, and fall short in producing temporally consistent sequences. In this paper, we propose {SatDreamer360}, a novel framework that generates geometrically and temporally consistent ground-view video from a single satellite image and a predefined trajectory. To bridge the large viewpoint gap, we introduce a compact tri-plane representation that encodes scene geometry directly from the satellite image. A ray-based pixel attention mechanism retrieves view-dependent features from the tri-plane, enabling accurate cross-view correspondence without requiring additional geometric priors. To ensure multi-frame consistency, we propose an epipolar-constrained temporal attention module that aligns features across frames using the known relative poses along the trajectory. To support evaluation, we introduce {VIGOR++}, a large-scale dataset for cross-view video generation, with dense trajectory annotations and high-quality ground-view sequences. Extensive experiments demonstrate that SatDreamer360 achieves superior performance in fidelity, coherence, and geometric alignment across diverse urban scenes.
>
---
#### [new 120] Dirty and Clean-Label attack detection using GAN discriminators
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉安全任务，旨在解决图像数据集中存在的脏标签和干净标签攻击问题。通过使用生成对抗网络（GAN）的判别器，论文提出了一种无需重新训练模型即可检测恶意图像的方法，并展示了其在不同扰动水平下的检测效果，为保护关键类别提供了新思路。**

- **链接: [http://arxiv.org/pdf/2506.01224v1](http://arxiv.org/pdf/2506.01224v1)**

> **作者:** John Smutny
>
> **备注:** 13 pages total. Appendix starts on page 10
>
> **摘要:** Gathering enough images to train a deep computer vision model is a constant challenge. Unfortunately, collecting images from unknown sources can leave your model s behavior at risk of being manipulated by a dirty-label or clean-label attack unless the images are properly inspected. Manually inspecting each image-label pair is impractical and common poison-detection methods that involve re-training your model can be time consuming. This research uses GAN discriminators to protect a single class against mislabeled and different levels of modified images. The effect of said perturbation on a basic convolutional neural network classifier is also included for reference. The results suggest that after training on a single class, GAN discriminator s confidence scores can provide a threshold to identify mislabeled images and identify 100% of the tested poison starting at a perturbation epsilon magnitude of 0.20, after decision threshold calibration using in-class samples. Developers can use this report as a basis to train their own discriminators to protect high valued classes in their CV models.
>
---
#### [new 121] Text-to-CT Generation via 3D Latent Diffusion Model with Contrastive Vision-Language Pretraining
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像生成任务，旨在解决从文本生成高质量3D CT图像的问题。作者提出一种结合3D对比视觉-语言预训练与潜在扩散模型的新方法，实现更准确的文本到CT生成，并验证其在数据增强和临床模拟中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2506.00633v1](http://arxiv.org/pdf/2506.00633v1)**

> **作者:** Daniele Molino; Camillo Maria Caruso; Filippo Ruffini; Paolo Soda; Valerio Guarrasi
>
> **摘要:** Objective: While recent advances in text-conditioned generative models have enabled the synthesis of realistic medical images, progress has been largely confined to 2D modalities such as chest X-rays. Extending text-to-image generation to volumetric Computed Tomography (CT) remains a significant challenge, due to its high dimensionality, anatomical complexity, and the absence of robust frameworks that align vision-language data in 3D medical imaging. Methods: We introduce a novel architecture for Text-to-CT generation that combines a latent diffusion model with a 3D contrastive vision-language pretraining scheme. Our approach leverages a dual-encoder CLIP-style model trained on paired CT volumes and radiology reports to establish a shared embedding space, which serves as the conditioning input for generation. CT volumes are compressed into a low-dimensional latent space via a pretrained volumetric VAE, enabling efficient 3D denoising diffusion without requiring external super-resolution stages. Results: We evaluate our method on the CT-RATE dataset and conduct a comprehensive assessment of image fidelity, clinical relevance, and semantic alignment. Our model achieves competitive performance across all tasks, significantly outperforming prior baselines for text-to-CT generation. Moreover, we demonstrate that CT scans synthesized by our framework can effectively augment real data, improving downstream diagnostic performance. Conclusion: Our results show that modality-specific vision-language alignment is a key component for high-quality 3D medical image generation. By integrating contrastive pretraining and volumetric diffusion, our method offers a scalable and controllable solution for synthesizing clinically meaningful CT volumes from text, paving the way for new applications in data augmentation, medical education, and automated clinical simulation.
>
---
#### [new 122] Depth-Aware Scoring and Hierarchical Alignment for Multiple Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪（MOT）任务，旨在解决遮挡和外观相似导致的跟踪失效问题。论文提出一种深度感知框架，引入零样本深度估计和分层对齐评分，优化目标关联过程。通过融合3D特征与像素级对齐，提升跟踪精度，且无需训练或微调。**

- **链接: [http://arxiv.org/pdf/2506.00774v1](http://arxiv.org/pdf/2506.00774v1)**

> **作者:** Milad Khanchi; Maria Amer; Charalambos Poullis
>
> **备注:** ICIP 2025
>
> **摘要:** Current motion-based multiple object tracking (MOT) approaches rely heavily on Intersection-over-Union (IoU) for object association. Without using 3D features, they are ineffective in scenarios with occlusions or visually similar objects. To address this, our paper presents a novel depth-aware framework for MOT. We estimate depth using a zero-shot approach and incorporate it as an independent feature in the association process. Additionally, we introduce a Hierarchical Alignment Score that refines IoU by integrating both coarse bounding box overlap and fine-grained (pixel-level) alignment to improve association accuracy without requiring additional learnable parameters. To our knowledge, this is the first MOT framework to incorporate 3D features (monocular depth) as an independent decision matrix in the association step. Our framework achieves state-of-the-art results on challenging benchmarks without any training nor fine-tuning. The code is available at https://github.com/Milad-Khanchi/DepthMOT
>
---
#### [new 123] Uneven Event Modeling for Partially Relevant Video Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于部分相关视频检索（PRVR）任务，旨在解决视频片段划分不准确和文本-视频对齐不佳的问题。作者提出了非均匀事件建模框架（UEM），包括渐进式分组视频分割（PGVS）和上下文感知事件优化（CAER）模块，以提升检索性能。**

- **链接: [http://arxiv.org/pdf/2506.00891v1](http://arxiv.org/pdf/2506.00891v1)**

> **作者:** Sa Zhu; Huashan Chen; Wanqian Zhang; Jinchao Zhang; Zexian Yang; Xiaoshuai Hao; Bo Li
>
> **备注:** Accepted by ICME 2025
>
> **摘要:** Given a text query, partially relevant video retrieval (PRVR) aims to retrieve untrimmed videos containing relevant moments, wherein event modeling is crucial for partitioning the video into smaller temporal events that partially correspond to the text. Previous methods typically segment videos into a fixed number of equal-length clips, resulting in ambiguous event boundaries. Additionally, they rely on mean pooling to compute event representations, inevitably introducing undesired misalignment. To address these, we propose an Uneven Event Modeling (UEM) framework for PRVR. We first introduce the Progressive-Grouped Video Segmentation (PGVS) module, to iteratively formulate events in light of both temporal dependencies and semantic similarity between consecutive frames, enabling clear event boundaries. Furthermore, we also propose the Context-Aware Event Refinement (CAER) module to refine the event representation conditioned the text's cross-attention. This enables event representations to focus on the most relevant frames for a given text, facilitating more precise text-video alignment. Extensive experiments demonstrate that our method achieves state-of-the-art performance on two PRVR benchmarks.
>
---
#### [new 124] FlexSelect: Flexible Token Selection for Efficient Long Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长视频处理中计算和内存开销过大的问题。作者提出了FlexSelect方法，通过跨模态注意力机制筛选关键视频片段，减少冗余信息。该方法无需训练即可评估视频token的重要性，并通过轻量级模型进行学习与筛选，适用于多种视频大模型，如LLaVA-Video、InternVL等，提升了处理效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.00993v1](http://arxiv.org/pdf/2506.00993v1)**

> **作者:** Yunzhu Zhang; Yu Lu; Tianyi Wang; Fengyun Rao; Yi Yang; Linchao Zhu
>
> **摘要:** Long-form video understanding poses a significant challenge for video large language models (VideoLLMs) due to prohibitively high computational and memory demands. In this paper, we propose FlexSelect, a flexible and efficient token selection strategy for processing long videos. FlexSelect identifies and retains the most semantically relevant content by leveraging cross-modal attention patterns from a reference transformer layer. It comprises two key components: (1) a training-free token ranking pipeline that leverages faithful cross-modal attention weights to estimate each video token's importance, and (2) a rank-supervised lightweight selector that is trained to replicate these rankings and filter redundant tokens. This generic approach can be seamlessly integrated into various VideoLLM architectures, such as LLaVA-Video, InternVL and Qwen-VL, serving as a plug-and-play module to extend their temporal context length. Empirically, FlexSelect delivers strong gains across multiple long-video benchmarks including VideoMME, MLVU, LongVB, and LVBench. Moreover, it achieves significant speed-ups (for example, up to 9 times on a LLaVA-Video-7B model), highlighting FlexSelect's promise for efficient long-form video understanding. Project page available at: https://yunzhuzhang0918.github.io/flex_select
>
---
#### [new 125] A Review on Coarse to Fine-Grained Animal Action Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动物行为识别任务，旨在解决从粗粒度到细粒度的动物动作识别问题。论文分析了现有研究进展、挑战及数据集局限性，比较了与人类动作识别的不同，并探讨了未来发展方向以提升识别精度和泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.01214v1](http://arxiv.org/pdf/2506.01214v1)**

> **作者:** Ali Zia; Renuka Sharma; Abdelwahed Khamis; Xuesong Li; Muhammad Husnain; Numan Shafi; Saeed Anwar; Sabine Schmoelzl; Eric Stone; Lars Petersson; Vivien Rolland
>
> **摘要:** This review provides an in-depth exploration of the field of animal action recognition, focusing on coarse-grained (CG) and fine-grained (FG) techniques. The primary aim is to examine the current state of research in animal behaviour recognition and to elucidate the unique challenges associated with recognising subtle animal actions in outdoor environments. These challenges differ significantly from those encountered in human action recognition due to factors such as non-rigid body structures, frequent occlusions, and the lack of large-scale, annotated datasets. The review begins by discussing the evolution of human action recognition, a more established field, highlighting how it progressed from broad, coarse actions in controlled settings to the demand for fine-grained recognition in dynamic environments. This shift is particularly relevant for animal action recognition, where behavioural variability and environmental complexity present unique challenges that human-centric models cannot fully address. The review then underscores the critical differences between human and animal action recognition, with an emphasis on high intra-species variability, unstructured datasets, and the natural complexity of animal habitats. Techniques like spatio-temporal deep learning frameworks (e.g., SlowFast) are evaluated for their effectiveness in animal behaviour analysis, along with the limitations of existing datasets. By assessing the strengths and weaknesses of current methodologies and introducing a recently-published dataset, the review outlines future directions for advancing fine-grained action recognition, aiming to improve accuracy and generalisability in behaviour analysis across species.
>
---
#### [new 126] FDSG: Forecasting Dynamic Scene Graphs
- **分类: cs.CV**

- **简介: 该论文属于动态场景图生成任务，旨在解决现有方法无法有效预测视频中实体与关系动态演化的问题。作者提出了FDSG框架，通过查询分解和神经随机微分方程建模实体与关系的时间动态，并设计时间聚合模块提升预测精度，实现了对未观测帧的完整场景图预测。**

- **链接: [http://arxiv.org/pdf/2506.01487v1](http://arxiv.org/pdf/2506.01487v1)**

> **作者:** Yi Yang; Yuren Cong; Hao Cheng; Bodo Rosenhahn; Michael Ying Yang
>
> **备注:** 21 pages, 9 figures, 15 tables
>
> **摘要:** Dynamic scene graph generation extends scene graph generation from images to videos by modeling entity relationships and their temporal evolution. However, existing methods either generate scene graphs from observed frames without explicitly modeling temporal dynamics, or predict only relationships while assuming static entity labels and locations. These limitations hinder effective extrapolation of both entity and relationship dynamics, restricting video scene understanding. We propose Forecasting Dynamic Scene Graphs (FDSG), a novel framework that predicts future entity labels, bounding boxes, and relationships, for unobserved frames, while also generating scene graphs for observed frames. Our scene graph forecast module leverages query decomposition and neural stochastic differential equations to model entity and relationship dynamics. A temporal aggregation module further refines predictions by integrating forecasted and observed information via cross-attention. To benchmark FDSG, we introduce Scene Graph Forecasting, a new task for full future scene graph prediction. Experiments on Action Genome show that FDSG outperforms state-of-the-art methods on dynamic scene graph generation, scene graph anticipation, and scene graph forecasting. Codes will be released upon publication.
>
---
#### [new 127] TIGeR: Text-Instructed Generation and Refinement for Template-Free Hand-Object Interaction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决手-物交互中依赖预定义3D模板的问题。现有方法需大量手动操作且适应性差，尤其在遮挡严重时效果不佳。论文提出TIGeR框架，利用文本驱动生成形状先验，并通过2D-3D注意力机制优化结果。实验显示其在Dex-YCB和Obman数据集上表现优异，具有抗遮挡能力和兼容多种先验来源的优势。**

- **链接: [http://arxiv.org/pdf/2506.00953v1](http://arxiv.org/pdf/2506.00953v1)**

> **作者:** Yiyao Huang; Zhedong Zheng; Yu Ziwei; Yaxiong Wang; Tze Ho Elden Tse; Angela Yao
>
> **摘要:** Pre-defined 3D object templates are widely used in 3D reconstruction of hand-object interactions. However, they often require substantial manual efforts to capture or source, and inherently restrict the adaptability of models to unconstrained interaction scenarios, e.g., heavily-occluded objects. To overcome this bottleneck, we propose a new Text-Instructed Generation and Refinement (TIGeR) framework, harnessing the power of intuitive text-driven priors to steer the object shape refinement and pose estimation. We use a two-stage framework: a text-instructed prior generation and vision-guided refinement. As the name implies, we first leverage off-the-shelf models to generate shape priors according to the text description without tedious 3D crafting. Considering the geometric gap between the synthesized prototype and the real object interacted with the hand, we further calibrate the synthesized prototype via 2D-3D collaborative attention. TIGeR achieves competitive performance, i.e., 1.979 and 5.468 object Chamfer distance on the widely-used Dex-YCB and Obman datasets, respectively, surpassing existing template-free methods. Notably, the proposed framework shows robustness to occlusion, while maintaining compatibility with heterogeneous prior sources, e.g., retrieved hand-crafted prototypes, in practical deployment scenarios.
>
---
#### [new 128] Parallel Rescaling: Rebalancing Consistency Guidance for Personalized Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决个性化扩散模型在少量参考图像下生成与文本提示一致且保持身份特征的问题。现有方法易过拟合并难以平衡提示贴合与身份保真。论文提出并行重标定技术，通过分解一致性引导信号并调整其平行分量，减少对分类器无关引导的干扰，从而提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.00607v1](http://arxiv.org/pdf/2506.00607v1)**

> **作者:** JungWoo Chae; Jiyoon Kim; Sangheum Hwang
>
> **摘要:** Personalizing diffusion models to specific users or concepts remains challenging, particularly when only a few reference images are available. Existing methods such as DreamBooth and Textual Inversion often overfit to limited data, causing misalignment between generated images and text prompts when attempting to balance identity fidelity with prompt adherence. While Direct Consistency Optimization (DCO) with its consistency-guided sampling partially alleviates this issue, it still struggles with complex or stylized prompts. In this paper, we propose a parallel rescaling technique for personalized diffusion models. Our approach explicitly decomposes the consistency guidance signal into parallel and orthogonal components relative to classifier free guidance (CFG). By rescaling the parallel component, we minimize disruptive interference with CFG while preserving the subject's identity. Unlike prior personalization methods, our technique does not require additional training data or expensive annotations. Extensive experiments show improved prompt alignment and visual fidelity compared to baseline methods, even on challenging stylized prompts. These findings highlight the potential of parallel rescaled guidance to yield more stable and accurate personalization for diverse user inputs.
>
---
#### [new 129] Advancing from Automated to Autonomous Beamline by Leveraging Computer Vision
- **分类: cs.CV**

- **简介: 该论文旨在实现同步辐射光束线从自动化向自主操作的跨越。任务是通过计算机视觉、深度学习和多视角摄像头实现实时碰撞检测，解决当前依赖人工安全监控的问题。工作包括设备分割、跟踪、几何分析及迁移学习提升鲁棒性，并开发交互式标注模块以适应新对象类别。**

- **链接: [http://arxiv.org/pdf/2506.00836v1](http://arxiv.org/pdf/2506.00836v1)**

> **作者:** Baolu Li; Hongkai Yu; Huiming Sun; Jin Ma; Yuewei Lin; Lu Ma; Yonghua Du
>
> **摘要:** The synchrotron light source, a cutting-edge large-scale user facility, requires autonomous synchrotron beamline operations, a crucial technique that should enable experiments to be conducted automatically, reliably, and safely with minimum human intervention. However, current state-of-the-art synchrotron beamlines still heavily rely on human safety oversight. To bridge the gap between automated and autonomous operation, a computer vision-based system is proposed, integrating deep learning and multiview cameras for real-time collision detection. The system utilizes equipment segmentation, tracking, and geometric analysis to assess potential collisions with transfer learning that enhances robustness. In addition, an interactive annotation module has been developed to improve the adaptability to new object classes. Experiments on a real beamline dataset demonstrate high accuracy, real-time performance, and strong potential for autonomous synchrotron beamline operations.
>
---
#### [new 130] Multi-Modal Dataset Distillation in the Wild
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态数据集蒸馏任务，旨在解决大规模、含噪声的多模态数据带来的训练成本高和性能下降问题。作者提出了MDW框架，通过引入可学习的细粒度对应关系和双轨协同学习策略，有效压缩数据并提升模型训练效果。**

- **链接: [http://arxiv.org/pdf/2506.01586v1](http://arxiv.org/pdf/2506.01586v1)**

> **作者:** Zhuohang Dang; Minnan Luo; Chengyou Jia; Hangwei Qian; Xiaojun Chang; Ivor W. Tsang
>
> **摘要:** Recent multi-modal models have shown remarkable versatility in real-world applications. However, their rapid development encounters two critical data challenges. First, the training process requires large-scale datasets, leading to substantial storage and computational costs. Second, these data are typically web-crawled with inevitable noise, i.e., partially mismatched pairs, severely degrading model performance. To these ends, we propose Multi-modal dataset Distillation in the Wild, i.e., MDW, the first framework to distill noisy multi-modal datasets into compact clean ones for effective and efficient model training. Specifically, MDW introduces learnable fine-grained correspondences during distillation and adaptively optimizes distilled data to emphasize correspondence-discriminative regions, thereby enhancing distilled data's information density and efficacy. Moreover, to capture robust cross-modal correspondence prior knowledge from real data, MDW proposes dual-track collaborative learning to avoid the risky data noise, alleviating information loss with certifiable noise tolerance. Extensive experiments validate MDW's theoretical and empirical efficacy with remarkable scalability, surpassing prior methods by over 15% across various compression ratios, highlighting its appealing practicality for applications with diverse efficacy and resource needs.
>
---
#### [new 131] Pseudo-Labeling Driven Refinement of Benchmark Object Detection Datasets via Analysis of Learning Patterns
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决MS-COCO数据集中存在的标注错误问题。通过提出一种伪标签驱动的自动修正框架，改进了标注质量并发布了MJ-COCO数据集，提升了模型性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.00997v1](http://arxiv.org/pdf/2506.00997v1)**

> **作者:** Min Je Kim; Muhammad Munsif; Altaf Hussain; Hikmat Yar; Sung Wook Baik
>
> **摘要:** Benchmark object detection (OD) datasets play a pivotal role in advancing computer vision applications such as autonomous driving, and surveillance, as well as in training and evaluating deep learning-based state-of-the-art detection models. Among them, MS-COCO has become a standard benchmark due to its diverse object categories and complex scenes. However, despite its wide adoption, MS-COCO suffers from various annotation issues, including missing labels, incorrect class assignments, inaccurate bounding boxes, duplicate labels, and group labeling inconsistencies. These errors not only hinder model training but also degrade the reliability and generalization of OD models. To address these challenges, we propose a comprehensive refinement framework and present MJ-COCO, a newly re-annotated version of MS-COCO. Our approach begins with loss and gradient-based error detection to identify potentially mislabeled or hard-to-learn samples. Next, we apply a four-stage pseudo-labeling refinement process: (1) bounding box generation using invertible transformations, (2) IoU-based duplicate removal and confidence merging, (3) class consistency verification via expert objects recognizer, and (4) spatial adjustment based on object region activation map analysis. This integrated pipeline enables scalable and accurate correction of annotation errors without manual re-labeling. Extensive experiments were conducted across four validation datasets: MS-COCO, Sama COCO, Objects365, and PASCAL VOC. Models trained on MJ-COCO consistently outperformed those trained on MS-COCO, achieving improvements in Average Precision (AP) and APS metrics. MJ-COCO also demonstrated significant gains in annotation coverage: for example, the number of small object annotations increased by more than 200,000 compared to MS-COCO.
>
---
#### [new 132] SkyReels-Audio: Omni Audio-Conditioned Talking Portraits in Video Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于音视频生成任务，旨在解决多模态输入条件下生成高质量、时序连贯的说话人肖像视频问题。作者提出了SkyReels-Audio框架，基于预训练视频扩散Transformer，实现无限长度视频的生成与编辑，并引入多种机制提升面部动作与音频对齐效果及局部细节一致性。**

- **链接: [http://arxiv.org/pdf/2506.00830v1](http://arxiv.org/pdf/2506.00830v1)**

> **作者:** Zhengcong Fei; Hao Jiang; Di Qiu; Baoxuan Gu; Youqiang Zhang; Jiahua Wang; Jialin Bai; Debang Li; Mingyuan Fan; Guibin Chen; Yahui Zhou
>
> **摘要:** The generation and editing of audio-conditioned talking portraits guided by multimodal inputs, including text, images, and videos, remains under explored. In this paper, we present SkyReels-Audio, a unified framework for synthesizing high-fidelity and temporally coherent talking portrait videos. Built upon pretrained video diffusion transformers, our framework supports infinite-length generation and editing, while enabling diverse and controllable conditioning through multimodal inputs. We employ a hybrid curriculum learning strategy to progressively align audio with facial motion, enabling fine-grained multimodal control over long video sequences. To enhance local facial coherence, we introduce a facial mask loss and an audio-guided classifier-free guidance mechanism. A sliding-window denoising approach further fuses latent representations across temporal segments, ensuring visual fidelity and temporal consistency across extended durations and diverse identities. More importantly, we construct a dedicated data pipeline for curating high-quality triplets consisting of synchronized audio, video, and textual descriptions. Comprehensive benchmark evaluations show that SkyReels-Audio achieves superior performance in lip-sync accuracy, identity consistency, and realistic facial dynamics, particularly under complex and challenging conditions.
>
---
#### [new 133] From Local Cues to Global Percepts: Emergent Gestalt Organization in Self-Supervised Vision Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究自监督视觉模型是否具备类似人类的格式塔组织能力，分析其对全局空间结构的敏感性。作者提出DiSRT测试基准，并发现使用MAE训练的模型（如ViT、ConvNeXt）表现出与格式塔原则一致的行为，而微调分类会削弱此能力。任务为计算机视觉中的结构感知建模。**

- **链接: [http://arxiv.org/pdf/2506.00718v1](http://arxiv.org/pdf/2506.00718v1)**

> **作者:** Tianqin Li; Ziqi Wen; Leiran Song; Jun Liu; Zhi Jing; Tai Sing Lee
>
> **摘要:** Human vision organizes local cues into coherent global forms using Gestalt principles like closure, proximity, and figure-ground assignment -- functions reliant on global spatial structure. We investigate whether modern vision models show similar behaviors, and under what training conditions these emerge. We find that Vision Transformers (ViTs) trained with Masked Autoencoding (MAE) exhibit activation patterns consistent with Gestalt laws, including illusory contour completion, convexity preference, and dynamic figure-ground segregation. To probe the computational basis, we hypothesize that modeling global dependencies is necessary for Gestalt-like organization. We introduce the Distorted Spatial Relationship Testbench (DiSRT), which evaluates sensitivity to global spatial perturbations while preserving local textures. Using DiSRT, we show that self-supervised models (e.g., MAE, CLIP) outperform supervised baselines and sometimes even exceed human performance. ConvNeXt models trained with MAE also exhibit Gestalt-compatible representations, suggesting such sensitivity can arise without attention architectures. However, classification finetuning degrades this ability. Inspired by biological vision, we show that a Top-K activation sparsity mechanism can restore global sensitivity. Our findings identify training conditions that promote or suppress Gestalt-like perception and establish DiSRT as a diagnostic for global structure sensitivity across models.
>
---
#### [new 134] WorldExplorer: Towards Generating Fully Navigable 3D Scenes
- **分类: cs.CV**

- **简介: 该论文属于3D场景生成任务，旨在解决现有方法在视角移动时产生模糊和噪声的问题。作者提出WorldExplorer，通过自回归视频轨迹生成和多视角融合，实现高质量、可自由探索的3D场景生成。**

- **链接: [http://arxiv.org/pdf/2506.01799v1](http://arxiv.org/pdf/2506.01799v1)**

> **作者:** Manuel-Andreas Schneider; Lukas Höllein; Matthias Nießner
>
> **备注:** project page: see https://the-world-explorer.github.io/, video: see https://youtu.be/c1lBnwJWNmE
>
> **摘要:** Generating 3D worlds from text is a highly anticipated goal in computer vision. Existing works are limited by the degree of exploration they allow inside of a scene, i.e., produce streched-out and noisy artifacts when moving beyond central or panoramic perspectives. To this end, we propose WorldExplorer, a novel method based on autoregressive video trajectory generation, which builds fully navigable 3D scenes with consistent visual quality across a wide range of viewpoints. We initialize our scenes by creating multi-view consistent images corresponding to a 360 degree panorama. Then, we expand it by leveraging video diffusion models in an iterative scene generation pipeline. Concretely, we generate multiple videos along short, pre-defined trajectories, that explore the scene in depth, including motion around objects. Our novel scene memory conditions each video on the most relevant prior views, while a collision-detection mechanism prevents degenerate results, like moving into objects. Finally, we fuse all generated views into a unified 3D representation via 3D Gaussian Splatting optimization. Compared to prior approaches, WorldExplorer produces high-quality scenes that remain stable under large camera motion, enabling for the first time realistic and unrestricted exploration. We believe this marks a significant step toward generating immersive and truly explorable virtual 3D environments.
>
---
#### [new 135] Ridgeformer: Mutli-Stage Contrastive Training For Fine-grained Cross-Domain Fingerprint Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于生物特征识别任务，旨在解决接触式与非接触式指纹识别中的跨域匹配问题。针对非接触式指纹图像存在的模糊、对比度低、形变等问题，作者提出了一种基于Transformer的多阶段对比训练方法（Ridgeformer），通过全局特征提取和局部特征对齐提升识别精度，并在多个公开数据集上验证了其优越性能。**

- **链接: [http://arxiv.org/pdf/2506.01806v1](http://arxiv.org/pdf/2506.01806v1)**

> **作者:** Shubham Pandey; Bhavin Jawade; Srirangaraj Setlur
>
> **备注:** Accepted to IEEE International Conference on Image Processing 2025
>
> **摘要:** The increasing demand for hygienic and portable biometric systems has underscored the critical need for advancements in contactless fingerprint recognition. Despite its potential, this technology faces notable challenges, including out-of-focus image acquisition, reduced contrast between fingerprint ridges and valleys, variations in finger positioning, and perspective distortion. These factors significantly hinder the accuracy and reliability of contactless fingerprint matching. To address these issues, we propose a novel multi-stage transformer-based contactless fingerprint matching approach that first captures global spatial features and subsequently refines localized feature alignment across fingerprint samples. By employing a hierarchical feature extraction and matching pipeline, our method ensures fine-grained, cross-sample alignment while maintaining the robustness of global feature representation. We perform extensive evaluations on publicly available datasets such as HKPolyU and RidgeBase under different evaluation protocols, such as contactless-to-contact matching and contactless-to-contactless matching and demonstrate that our proposed approach outperforms existing methods, including COTS solutions.
>
---
#### [new 136] G4Seg: Generation for Inexact Segmentation Refinement with Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分割任务，旨在解决不精确分割（IS）问题。现有方法依赖判别模型或密集视觉表征，而该文提出G4Seg，利用Stable Diffusion的生成先验，通过对比原图与掩码生成图的差异，实现从粗到精的分割优化。方法为即插即用设计，验证了生成差异在建模密集表征中的潜力。**

- **链接: [http://arxiv.org/pdf/2506.01539v1](http://arxiv.org/pdf/2506.01539v1)**

> **作者:** Tianjiao Zhang; Fei Zhang; Jiangchao Yao; Ya Zhang; Yanfeng Wang
>
> **备注:** 16 pages, 12 figures, IEEE International Conference on Multimedia & Expo 2025
>
> **摘要:** This paper considers the problem of utilizing a large-scale text-to-image diffusion model to tackle the challenging Inexact Segmentation (IS) task. Unlike traditional approaches that rely heavily on discriminative-model-based paradigms or dense visual representations derived from internal attention mechanisms, our method focuses on the intrinsic generative priors in Stable Diffusion~(SD). Specifically, we exploit the pattern discrepancies between original images and mask-conditional generated images to facilitate a coarse-to-fine segmentation refinement by establishing a semantic correspondence alignment and updating the foreground probability. Comprehensive quantitative and qualitative experiments validate the effectiveness and superiority of our plug-and-play design, underscoring the potential of leveraging generation discrepancies to model dense representations and encouraging further exploration of generative approaches for solving discriminative tasks.
>
---
#### [new 137] Beyond black and white: A more nuanced approach to facial recognition with continuous ethnicity labels
- **分类: cs.CV**

- **简介: 该论文属于人脸识别任务，旨在解决数据偏见导致的模型不公平问题。传统方法用离散种族标签平衡数据，但效果有限。本文提出使用连续种族标签，构建更真实的数据平衡方式。通过训练65个模型及创建20个数据子集验证，发现基于连续变量的模型性能更优。**

- **链接: [http://arxiv.org/pdf/2506.01532v1](http://arxiv.org/pdf/2506.01532v1)**

> **作者:** Pedro C. Neto; Naser Damer; Jaime S. Cardoso; Ana F. Sequeira
>
> **备注:** Under review
>
> **摘要:** Bias has been a constant in face recognition models. Over the years, researchers have looked at it from both the model and the data point of view. However, their approach to mitigation of data bias was limited and lacked insight on the real nature of the problem. Here, in this document, we propose to revise our use of ethnicity labels as a continuous variable instead of a discrete value per identity. We validate our formulation both experimentally and theoretically, showcasing that not all identities from one ethnicity contribute equally to the balance of the dataset; thus, having the same number of identities per ethnicity does not represent a balanced dataset. We further show that models trained on datasets balanced in the continuous space consistently outperform models trained on data balanced in the discrete space. We trained more than 65 different models, and created more than 20 subsets of the original datasets.
>
---
#### [new 138] No Train Yet Gain: Towards Generic Multi-Object Tracking in Sports and Beyond
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，旨在解决体育视频中因快速运动、遮挡等因素导致的跟踪难题。论文提出了McByte方法，通过结合检测与分割掩码传播提升跟踪鲁棒性，无需训练或视频特定调优，适用于体育及通用行人跟踪。**

- **链接: [http://arxiv.org/pdf/2506.01373v1](http://arxiv.org/pdf/2506.01373v1)**

> **作者:** Tomasz Stanczyk; Seongro Yoon; Francois Bremond
>
> **摘要:** Multi-object tracking (MOT) is essential for sports analytics, enabling performance evaluation and tactical insights. However, tracking in sports is challenging due to fast movements, occlusions, and camera shifts. Traditional tracking-by-detection methods require extensive tuning, while segmentation-based approaches struggle with track processing. We propose McByte, a tracking-by-detection framework that integrates temporally propagated segmentation mask as an association cue to improve robustness without per-video tuning. Unlike many existing methods, McByte does not require training, relying solely on pre-trained models and object detectors commonly used in the community. Evaluated on SportsMOT, DanceTrack, SoccerNet-tracking 2022 and MOT17, McByte demonstrates strong performance across sports and general pedestrian tracking. Our results highlight the benefits of mask propagation for a more adaptable and generalizable MOT approach. Code will be made available at https://github.com/tstanczyk95/McByte.
>
---
#### [new 139] Motion-Aware Concept Alignment for Consistent Video Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频编辑任务，旨在解决在保持视频原有运动和场景的前提下，将参考图像的语义特征注入视频中特定物体的问题。作者提出了MoCA-Video方法，通过运动感知与概念对齐实现高质量、连贯的视频编辑。**

- **链接: [http://arxiv.org/pdf/2506.01004v1](http://arxiv.org/pdf/2506.01004v1)**

> **作者:** Tong Zhang; Juan C Leon Alcazar; Bernard Ghanem
>
> **摘要:** We introduce MoCA-Video (Motion-Aware Concept Alignment in Video), a training-free framework bridging the gap between image-domain semantic mixing and video. Given a generated video and a user-provided reference image, MoCA-Video injects the semantic features of the reference image into a specific object within the video, while preserving the original motion and visual context. Our approach leverages a diagonal denoising schedule and class-agnostic segmentation to detect and track objects in the latent space and precisely control the spatial location of the blended objects. To ensure temporal coherence, we incorporate momentum-based semantic corrections and gamma residual noise stabilization for smooth frame transitions. We evaluate MoCA's performance using the standard SSIM, image-level LPIPS, temporal LPIPS, and introduce a novel metric CASS (Conceptual Alignment Shift Score) to evaluate the consistency and effectiveness of the visual shifts between the source prompt and the modified video frames. Using self-constructed dataset, MoCA-Video outperforms current baselines, achieving superior spatial consistency, coherent motion, and a significantly higher CASS score, despite having no training or fine-tuning. MoCA-Video demonstrates that structured manipulation in the diffusion noise trajectory allows for controllable, high-quality video synthesis.
>
---
#### [new 140] ABCDEFGH: An Adaptation-Based Convolutional Neural Network-CycleGAN Disease-Courses Evolution Framework Using Generative Models in Health Education
- **分类: cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决医疗教育中高质量教学材料不足的问题。作者利用CNN和CycleGAN生成合成医学图像，以提供多样化且保护隐私的教学资源，支持现代医学教育。**

- **链接: [http://arxiv.org/pdf/2506.00605v1](http://arxiv.org/pdf/2506.00605v1)**

> **作者:** Ruiming Min; Minghao Liu
>
> **摘要:** With the advancement of modern medicine and the development of technologies such as MRI, CT, and cellular analysis, it has become increasingly critical for clinicians to accurately interpret various diagnostic images. However, modern medical education often faces challenges due to limited access to high-quality teaching materials, stemming from privacy concerns and a shortage of educational resources (Balogh et al., 2015). In this context, image data generated by machine learning models, particularly generative models, presents a promising solution. These models can create diverse and comparable imaging datasets without compromising patient privacy, thereby supporting modern medical education. In this study, we explore the use of convolutional neural networks (CNNs) and CycleGAN (Zhu et al., 2017) for generating synthetic medical images. The source code is available at https://github.com/mliuby/COMP4211-Project.
>
---
#### [new 141] Fovea Stacking: Imaging with Dynamic Localized Aberration Correction
- **分类: cs.CV**

- **简介: 该论文属于计算成像任务，旨在解决简化光学系统中严重的离轴像差问题。作者提出Fovea Stacking方法，利用可变形相位板（DPP）实现局部像差校正，并通过不同iable光学模型优化，生成高清晰度的中心注视图像。叠加多个图像获得无像差的全景图，并结合目标检测或眼动追踪实现动态视频聚焦。**

- **链接: [http://arxiv.org/pdf/2506.00716v1](http://arxiv.org/pdf/2506.00716v1)**

> **作者:** Shi Mao; Yogeshwar Mishra; Wolfgang Heidrich
>
> **摘要:** The desire for cameras with smaller form factors has recently lead to a push for exploring computational imaging systems with reduced optical complexity such as a smaller number of lens elements. Unfortunately such simplified optical systems usually suffer from severe aberrations, especially in off-axis regions, which can be difficult to correct purely in software. In this paper we introduce Fovea Stacking, a new type of imaging system that utilizes emerging dynamic optical components called deformable phase plates (DPPs) for localized aberration correction anywhere on the image sensor. By optimizing DPP deformations through a differentiable optical model, off-axis aberrations are corrected locally, producing a foveated image with enhanced sharpness at the fixation point - analogous to the eye's fovea. Stacking multiple such foveated images, each with a different fixation point, yields a composite image free from aberrations. To efficiently cover the entire field of view, we propose joint optimization of DPP deformations under imaging budget constraints. Due to the DPP device's non-linear behavior, we introduce a neural network-based control model for improved alignment between simulation-hardware performance. We further demonstrated that for extended depth-of-field imaging, fovea stacking outperforms traditional focus stacking in image quality. By integrating object detection or eye-tracking, the system can dynamically adjust the lens to track the object of interest-enabling real-time foveated video suitable for downstream applications such as surveillance or foveated virtual reality displays.
>
---
#### [new 142] Neural shape reconstruction from multiple views with static pattern projection
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决传统主动立体系统因需精确标定相机与投影仪位姿而影响使用便捷性的问题。论文提出一种新方法，通过多视角图像捕捉并结合神经SDF与体积微分渲染技术，实现相机与投影仪在运动中自动标定并恢复物体形状。**

- **链接: [http://arxiv.org/pdf/2506.01389v1](http://arxiv.org/pdf/2506.01389v1)**

> **作者:** Ryo Furukawa; Kota Nishihara; Hiroshi Kawasaki
>
> **备注:** 6 pages, CVPR 2025 Workshop on Neural Fields Beyond Conventional Cameras
>
> **摘要:** Active-stereo-based 3D shape measurement is crucial for various purposes, such as industrial inspection, reverse engineering, and medical systems, due to its strong ability to accurately acquire the shape of textureless objects. Active stereo systems typically consist of a camera and a pattern projector, tightly fixed to each other, and precise calibration between a camera and a projector is required, which in turn decreases the usability of the system. If a camera and a projector can be freely moved during shape scanning process, it will drastically increase the convenience of the usability of the system. To realize it, we propose a technique to recover the shape of the target object by capturing multiple images while both the camera and the projector are in motion, and their relative poses are auto-calibrated by our neural signed-distance-field (NeuralSDF) using novel volumetric differential rendering technique. In the experiment, the proposed method is evaluated by performing 3D reconstruction using both synthetic and real images.
>
---
#### [new 143] MR2US-Pro: Prostate MR to Ultrasound Image Translation and Registration Based on Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决前列腺MRI与TRUS图像的跨模态配准难题。作者提出MR2US-Pro框架，先进行TRUS三维重建，再通过扩散模型实现无需监督的模态翻译引导配准。方法不依赖探针追踪信息，并引入聚类特征匹配和解剖感知策略，提升了配准精度与形变合理性。**

- **链接: [http://arxiv.org/pdf/2506.00591v1](http://arxiv.org/pdf/2506.00591v1)**

> **作者:** Xudong Ma; Nantheera Anantrasirichai; Stefanos Bolomytis; Alin Achim
>
> **摘要:** The diagnosis of prostate cancer increasingly depends on multimodal imaging, particularly magnetic resonance imaging (MRI) and transrectal ultrasound (TRUS). However, accurate registration between these modalities remains a fundamental challenge due to the differences in dimensionality and anatomical representations. In this work, we present a novel framework that addresses these challenges through a two-stage process: TRUS 3D reconstruction followed by cross-modal registration. Unlike existing TRUS 3D reconstruction methods that rely heavily on external probe tracking information, we propose a totally probe-location-independent approach that leverages the natural correlation between sagittal and transverse TRUS views. With the help of our clustering-based feature matching method, we enable the spatial localization of 2D frames without any additional probe tracking information. For the registration stage, we introduce an unsupervised diffusion-based framework guided by modality translation. Unlike existing methods that translate one modality into another, we map both MR and US into a pseudo intermediate modality. This design enables us to customize it to retain only registration-critical features, greatly easing registration. To further enhance anatomical alignment, we incorporate an anatomy-aware registration strategy that prioritizes internal structural coherence while adaptively reducing the influence of boundary inconsistencies. Extensive validation demonstrates that our approach outperforms state-of-the-art methods by achieving superior registration accuracy with physically realistic deformations in a completely unsupervised fashion.
>
---
#### [new 144] Camera Trajectory Generation: A Comprehensive Survey of Methods, Metrics, and Future Directions
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于计算机图形学与视觉任务，旨在系统综述相机轨迹生成的方法、评估指标及未来方向。它梳理了从基础定义到先进模型的发展，涵盖规则、优化、机器学习及混合方法，并分析常用数据集与评价标准，指出研究不足与发展方向。**

- **链接: [http://arxiv.org/pdf/2506.00974v1](http://arxiv.org/pdf/2506.00974v1)**

> **作者:** Zahra Dehghanian; Pouya Ardekhani; Amir Vahedi; Hamid Beigy; Hamid R. Rabiee
>
> **摘要:** Camera trajectory generation is a cornerstone in computer graphics, robotics, virtual reality, and cinematography, enabling seamless and adaptive camera movements that enhance visual storytelling and immersive experiences. Despite its growing prominence, the field lacks a systematic and unified survey that consolidates essential knowledge and advancements in this domain. This paper addresses this gap by providing the first comprehensive review of the field, covering from foundational definitions to advanced methodologies. We introduce the different approaches to camera representation and present an in-depth review of available camera trajectory generation models, starting with rule-based approaches and progressing through optimization-based techniques, machine learning advancements, and hybrid methods that integrate multiple strategies. Additionally, we gather and analyze the metrics and datasets commonly used for evaluating camera trajectory systems, offering insights into how these tools measure performance, aesthetic quality, and practical applicability. Finally, we highlight existing limitations, critical gaps in current research, and promising opportunities for investment and innovation in the field. This paper not only serves as a foundational resource for researchers entering the field but also paves the way for advancing adaptive, efficient, and creative camera trajectory systems across diverse applications.
>
---
#### [new 145] CReFT-CAD: Boosting Orthographic Projection Reasoning for CAD via Reinforcement Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文属于CAD领域的正交投影推理任务，旨在解决现有深度学习方法在CAD流程中推理能力不足、泛化性差的问题。作者提出CReFT-CAD方法，结合强化学习与监督微调，并发布大规模数据集TriView2CAD，以提升正交投影推理的准确性和实际应用能力。**

- **链接: [http://arxiv.org/pdf/2506.00568v1](http://arxiv.org/pdf/2506.00568v1)**

> **作者:** Ke Niu; Zhuofan Chen; Haiyang Yu; Yuwen Chen; Teng Fu; Mengyang Zhao; Bin Li; Xiangyang Xue
>
> **摘要:** Computer-Aided Design (CAD) plays a pivotal role in industrial manufacturing. Orthographic projection reasoning underpins the entire CAD workflow, encompassing design, manufacturing, and simulation. However, prevailing deep-learning approaches employ standard 3D reconstruction pipelines as an alternative, which often introduce imprecise dimensions and limit the parametric editability required for CAD workflows. Recently, some researchers adopt vision-language models (VLMs), particularly supervised fine-tuning (SFT), to tackle CAD-related challenges. SFT shows promise but often devolves into pattern memorization, yielding poor out-of-distribution performance on complex reasoning tasks. To address these gaps, we introduce CReFT-CAD, a two-stage fine-tuning paradigm that first employs a curriculum-driven reinforcement learning stage with difficulty-aware rewards to build reasoning ability steadily, and then applies supervised post-tuning to hone instruction following and semantic extraction. Complementing this, we release TriView2CAD, the first large-scale, open-source benchmark for orthographic projection reasoning, comprising 200,000 synthetic and 3,000 real-world orthographic projections with precise dimension annotations and six interoperable data modalities. We benchmark leading VLMs on orthographic projection reasoning and demonstrate that CReFT-CAD substantially improves reasoning accuracy and out-of-distribution generalizability in real-world scenarios, offering valuable insights for advancing CAD reasoning research.
>
---
#### [new 146] FastCAR: Fast Classification And Regression for Task Consolidation in Multi-Task Learning to Model a Continuous Property Variable of Detected Object Class
- **分类: cs.CV**

- **简介: 论文提出FastCAR，一种多任务学习中快速分类与回归的任务整合方法，解决对象分类与连续属性建模的异构任务难以有效联合建模的问题。通过标签转换策略，在单一回归网络中高效实现，训练速度提升2.52倍，推理延迟降低55%，在自建钢材属性数据集上取得优异性能。**

- **链接: [http://arxiv.org/pdf/2506.00208v1](http://arxiv.org/pdf/2506.00208v1)**

> **作者:** Anoop Kini; Andreas Jansche; Timo Bernthaler; Gerhard Schneider
>
> **摘要:** FastCAR is a novel task consolidation approach in Multi-Task Learning (MTL) for a classification and a regression task, despite the non-triviality of task heterogeneity with only a subtle correlation. The approach addresses the classification of a detected object (occupying the entire image frame) and regression for modeling a continuous property variable (for instances of an object class), a crucial use case in science and engineering. FastCAR involves a label transformation approach that is amenable for use with only a single-task regression network architecture. FastCAR outperforms traditional MTL model families, parametrized in the landscape of architecture and loss weighting schemes, when learning both tasks are collectively considered (classification accuracy of 99.54%, regression mean absolute percentage error of 2.4%). The experiments performed used "Advanced Steel Property Dataset" contributed by us https://github.com/fastcandr/AdvancedSteel-Property-Dataset. The dataset comprises 4536 images of 224x224 pixels, annotated with discrete object classes and its hardness property that can take continuous values. Our proposed FastCAR approach for task consolidation achieves training time efficiency (2.52x quicker) and reduced inference latency (55% faster) than benchmark MTL networks.
>
---
#### [new 147] A 2-Stage Model for Vehicle Class and Orientation Detection with Photo-Realistic Image Generation
- **分类: cs.CV**

- **简介: 论文属于车辆类别与朝向检测任务，旨在解决合成数据训练模型在真实场景中表现差及类别分布不均的问题。工作包括：构建包含信息的元表、合成图像转真实风格、分类预测，并结合位置信息提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.01338v1](http://arxiv.org/pdf/2506.01338v1)**

> **作者:** Youngmin Kim; Donghwa Kang; Hyeongboo Baek
>
> **备注:** Accepted to IEEE BigData Conference 2022
>
> **摘要:** We aim to detect the class and orientation of a vehicle by training a model with synthetic data. However, the distribution of the classes in the training data is imbalanced, and the model trained on the synthetic image is difficult to predict in real-world images. We propose a two-stage detection model with photo-realistic image generation to tackle this issue. Our model mainly takes four steps to detect the class and orientation of the vehicle. (1) It builds a table containing the image, class, and location information of objects in the image, (2) transforms the synthetic images into real-world images style, and merges them into the meta table. (3) Classify vehicle class and orientation using images from the meta-table. (4) Finally, the vehicle class and orientation are detected by combining the pre-extracted location information and the predicted classes. We achieved 4th place in IEEE BigData Challenge 2022 Vehicle class and Orientation Detection (VOD) with our approach.
>
---
#### [new 148] Revolutionizing Radiology Workflow with Factual and Efficient CXR Report Generation
- **分类: cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在解决胸片（CXR）诊断效率与准确性不足的问题。作者提出CXR-PathFinder模型，结合临床反馈的对抗微调和知识图谱增强模块，提升报告生成的准确性和可靠性，实验证明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.01118v1](http://arxiv.org/pdf/2506.01118v1)**

> **作者:** Pimchanok Sukjai; Apiradee Boonmee
>
> **摘要:** The escalating demand for medical image interpretation underscores the critical need for advanced artificial intelligence solutions to enhance the efficiency and accuracy of radiological diagnoses. This paper introduces CXR-PathFinder, a novel Large Language Model (LLM)-centric foundation model specifically engineered for automated chest X-ray (CXR) report generation. We propose a unique training paradigm, Clinician-Guided Adversarial Fine-Tuning (CGAFT), which meticulously integrates expert clinical feedback into an adversarial learning framework to mitigate factual inconsistencies and improve diagnostic precision. Complementing this, our Knowledge Graph Augmentation Module (KGAM) acts as an inference-time safeguard, dynamically verifying generated medical statements against authoritative knowledge bases to minimize hallucinations and ensure standardized terminology. Leveraging a comprehensive dataset of millions of paired CXR images and expert reports, our experiments demonstrate that CXR-PathFinder significantly outperforms existing state-of-the-art medical vision-language models across various quantitative metrics, including clinical accuracy (Macro F1 (14): 46.5, Micro F1 (14): 59.5). Furthermore, blinded human evaluation by board-certified radiologists confirms CXR-PathFinder's superior clinical utility, completeness, and accuracy, establishing its potential as a reliable and efficient aid for radiological practice. The developed method effectively balances high diagnostic fidelity with computational efficiency, providing a robust solution for automated medical report generation.
>
---
#### [new 149] SAM-I2V: Upgrading SAM to Support Promptable Video Segmentation with Less than 0.2% Training Cost
- **分类: cs.CV**

- **简介: 该论文属于视频分割任务，旨在解决将图像分割模型扩展到视频时存在的训练成本高、动态场景处理难等问题。作者提出了SAM-I2V方法，通过升级现有模型（SAM）实现高效视频分割，引入了特征提取升级、记忆筛选和记忆提示机制，大幅降低训练成本，达到接近SAM 2性能的同时仅用其0.2%的训练资源。**

- **链接: [http://arxiv.org/pdf/2506.01304v1](http://arxiv.org/pdf/2506.01304v1)**

> **作者:** Haiyang Mei; Pengyu Zhang; Mike Zheng Shou
>
> **备注:** CVPR 2025
>
> **摘要:** Foundation models like the Segment Anything Model (SAM) have significantly advanced promptable image segmentation in computer vision. However, extending these capabilities to videos presents substantial challenges, particularly in ensuring precise and temporally consistent mask propagation in dynamic scenes. SAM 2 attempts to address this by training a model on massive image and video data from scratch to learn complex spatiotemporal associations, resulting in huge training costs that hinder research and practical deployment. In this paper, we introduce SAM-I2V, an effective image-to-video upgradation method for cultivating a promptable video segmentation (PVS) model. Our approach strategically upgrades the pre-trained SAM to support PVS, significantly reducing training complexity and resource requirements. To achieve this, we introduce three key innovations: (i) an image-to-video feature extraction upgrader built upon SAM's static image encoder to enable spatiotemporal video perception, (ii) a memory filtering strategy that selects the most relevant past frames for more effective utilization of historical information, and (iii) a memory-as-prompt mechanism leveraging object memory to ensure temporally consistent mask propagation in dynamic scenes. Comprehensive experiments demonstrate that our method achieves over 90% of SAM 2's performance while using only 0.2% of its training cost. Our work presents a resource-efficient pathway to PVS, lowering barriers for further research in PVS model design and enabling broader applications and advancements in the field. Code and model are available at: https://github.com/showlab/SAM-I2V.
>
---
#### [new 150] Common Inpainted Objects In-N-Out of Context
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉任务，旨在解决现有数据集中缺乏上下文不一致样本的问题。作者构建了COinCO数据集，包含97,722张图像，通过扩散模型修复COCO图像中的物体，生成上下文一致或不一致的场景，并利用多模态大语言模型标注。该数据集支持上下文分类、物体预测和伪造检测等任务，推动上下文感知的视觉理解与图像取证研究。**

- **链接: [http://arxiv.org/pdf/2506.00721v1](http://arxiv.org/pdf/2506.00721v1)**

> **作者:** Tianze Yang; Tyson Jordan; Ninghao Liu; Jin Sun
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** We present Common Inpainted Objects In-N-Out of Context (COinCO), a novel dataset addressing the scarcity of out-of-context examples in existing vision datasets. By systematically replacing objects in COCO images through diffusion-based inpainting, we create 97,722 unique images featuring both contextually coherent and inconsistent scenes, enabling effective context learning. Each inpainted object is meticulously verified and categorized as in- or out-of-context through a multimodal large language model assessment. Our analysis reveals significant patterns in semantic priors that influence inpainting success across object categories. We demonstrate three key tasks enabled by COinCO: (1) training context classifiers that effectively determine whether existing objects belong in their context; (2) a novel Objects-from-Context prediction task that determines which new objects naturally belong in given scenes at both instance and clique levels, and (3) context-enhanced fake detection on state-of-the-art methods without fine-tuning. COinCO provides a controlled testbed with contextual variations, establishing a foundation for advancing context-aware visual understanding in computer vision and image forensics. Our code and data are at: https://github.com/YangTianze009/COinCO.
>
---
#### [new 151] Enhancing Diffusion-based Unrestricted Adversarial Attacks via Adversary Preferences Alignment
- **分类: cs.CV**

- **简介: 该论文属于对抗攻击任务，旨在解决扩散模型中视觉一致性与攻击有效性之间的冲突优化问题。作者提出了APA框架，分两阶段解耦冲突偏好，并引入扩散增强策略，提升黑盒攻击的可迁移性，同时保持高质量生成效果。**

- **链接: [http://arxiv.org/pdf/2506.01511v1](http://arxiv.org/pdf/2506.01511v1)**

> **作者:** Kaixun Jiang; Zhaoyu Chen; Haijing Guo; Jinglun Li; Jiyuan Fu; Pinxue Guo; Hao Tang; Bo Li; Wenqiang Zhang
>
> **摘要:** Preference alignment in diffusion models has primarily focused on benign human preferences (e.g., aesthetic). In this paper, we propose a novel perspective: framing unrestricted adversarial example generation as a problem of aligning with adversary preferences. Unlike benign alignment, adversarial alignment involves two inherently conflicting preferences: visual consistency and attack effectiveness, which often lead to unstable optimization and reward hacking (e.g., reducing visual quality to improve attack success). To address this, we propose APA (Adversary Preferences Alignment), a two-stage framework that decouples conflicting preferences and optimizes each with differentiable rewards. In the first stage, APA fine-tunes LoRA to improve visual consistency using rule-based similarity reward. In the second stage, APA updates either the image latent or prompt embedding based on feedback from a substitute classifier, guided by trajectory-level and step-wise rewards. To enhance black-box transferability, we further incorporate a diffusion augmentation strategy. Experiments demonstrate that APA achieves significantly better attack transferability while maintaining high visual consistency, inspiring further research to approach adversarial attacks from an alignment perspective. Code will be available at https://github.com/deep-kaixun/APA.
>
---
#### [new 152] A Novel Context-Adaptive Fusion of Shadow and Highlight Regions for Efficient Sonar Image Classification
- **分类: cs.CV**

- **简介: 该论文属于水下声呐图像分类任务，旨在解决阴影区域利用不足及噪声干扰问题。作者提出了一种上下文自适应的阴影与亮区融合分类框架，引入阴影专用分类器、区域感知去噪模型，并构建了包含物理噪声的扩展数据集S3Simulator+，以提升声呐图像分类的鲁棒性与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.01445v1](http://arxiv.org/pdf/2506.01445v1)**

> **作者:** Kamal Basha S; Anukul Kiran B; Athira Nambiar; Suresh Rajendran
>
> **摘要:** Sonar imaging is fundamental to underwater exploration, with critical applications in defense, navigation, and marine research. Shadow regions, in particular, provide essential cues for object detection and classification, yet existing studies primarily focus on highlight-based analysis, leaving shadow-based classification underexplored. To bridge this gap, we propose a Context-adaptive sonar image classification framework that leverages advanced image processing techniques to extract and integrate discriminative shadow and highlight features. Our framework introduces a novel shadow-specific classifier and adaptive shadow segmentation, enabling effective classification based on the dominant region. This approach ensures optimal feature representation, improving robustness against noise and occlusions. In addition, we introduce a Region-aware denoising model that enhances sonar image quality by preserving critical structural details while suppressing noise. This model incorporates an explainability-driven optimization strategy, ensuring that denoising is guided by feature importance, thereby improving interpretability and classification reliability. Furthermore, we present S3Simulator+, an extended dataset incorporating naval mine scenarios with physics-informed noise specifically tailored for the underwater sonar domain, fostering the development of robust AI models. By combining novel classification strategies with an enhanced dataset, our work addresses key challenges in sonar image analysis, contributing to the advancement of autonomous underwater perception.
>
---
#### [new 153] Fast and Robust Rotation Averaging with Anisotropic Coordinate Descent
- **分类: cs.CV**

- **简介: 该论文属于结构从运动（SfM）任务，旨在解决各向异性旋转平均问题。现有方法在最优性、鲁棒性或效率方面存在不足。作者提出了一种快速且鲁棒的各向异性坐标下降方法，结合块坐标下降思想，实现了高效求解并提升了大尺度数据下的性能。**

- **链接: [http://arxiv.org/pdf/2506.01940v1](http://arxiv.org/pdf/2506.01940v1)**

> **作者:** Yaroslava Lochman; Carl Olsson; Christopher Zach
>
> **摘要:** Anisotropic rotation averaging has recently been explored as a natural extension of respective isotropic methods. In the anisotropic formulation, uncertainties of the estimated relative rotations -- obtained via standard two-view optimization -- are propagated to the optimization of absolute rotations. The resulting semidefinite relaxations are able to recover global minima but scale poorly with the problem size. Local methods are fast and also admit robust estimation but are sensitive to initialization. They usually employ minimum spanning trees and therefore suffer from drift accumulation and can get trapped in poor local minima. In this paper, we attempt to bridge the gap between optimality, robustness and efficiency of anisotropic rotation averaging. We analyze a family of block coordinate descent methods initially proposed to optimize the standard chordal distances, and derive a much simpler formulation and an anisotropic extension obtaining a fast general solver. We integrate this solver into the extended anisotropic large-scale robust rotation averaging pipeline. The resulting algorithm achieves state-of-the-art performance on public structure-from-motion datasets. Project page: https://ylochman.github.io/acd
>
---
#### [new 154] DNAEdit: Direct Noise Alignment for Text-Guided Rectified Flow Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决文本引导下图像编辑中的误差累积和编辑控制问题。提出了DNAEdit方法，通过直接噪声对齐（DNA）减少误差积累，并引入移动速度引导（MVG）平衡背景保留与对象编辑。此外，构建了DNA-Bench评估基准。**

- **链接: [http://arxiv.org/pdf/2506.01430v1](http://arxiv.org/pdf/2506.01430v1)**

> **作者:** Chenxi Xie; Minghan Li; Shuai Li; Yuhui Wu; Qiaosi Yi; Lei Zhang
>
> **备注:** Project URL: https://xiechenxi99.github.io/DNAEdit
>
> **摘要:** Leveraging the powerful generation capability of large-scale pretrained text-to-image models, training-free methods have demonstrated impressive image editing results. Conventional diffusion-based methods, as well as recent rectified flow (RF)-based methods, typically reverse synthesis trajectories by gradually adding noise to clean images, during which the noisy latent at the current timestep is used to approximate that at the next timesteps, introducing accumulated drift and degrading reconstruction accuracy. Considering the fact that in RF the noisy latent is estimated through direct interpolation between Gaussian noises and clean images at each timestep, we propose Direct Noise Alignment (DNA), which directly refines the desired Gaussian noise in the noise domain, significantly reducing the error accumulation in previous methods. Specifically, DNA estimates the velocity field of the interpolated noised latent at each timestep and adjusts the Gaussian noise by computing the difference between the predicted and expected velocity field. We validate the effectiveness of DNA and reveal its relationship with existing RF-based inversion methods. Additionally, we introduce a Mobile Velocity Guidance (MVG) to control the target prompt-guided generation process, balancing image background preservation and target object editability. DNA and MVG collectively constitute our proposed method, namely DNAEdit. Finally, we introduce DNA-Bench, a long-prompt benchmark, to evaluate the performance of advanced image editing models. Experimental results demonstrate that our DNAEdit achieves superior performance to state-of-the-art text-guided editing methods. Codes and benchmark will be available at \href{ https://xiechenxi99.github.io/DNAEdit/}{https://xiechenxi99.github.io/DNAEdit/}.
>
---
#### [new 155] UMA: Ultra-detailed Human Avatars via Multi-level Surface Alignment
- **分类: cs.CV**

- **简介: 该论文属于计算机图形学与视觉任务，旨在生成高细节、可动画的虚拟人类模型。现有方法在几何对齐上存在误差，导致细节丢失。作者提出UMA框架，通过2D点追踪和级联训练优化3D形变，提升几何精度与渲染质量。**

- **链接: [http://arxiv.org/pdf/2506.01802v1](http://arxiv.org/pdf/2506.01802v1)**

> **作者:** Heming Zhu; Guoxing Sun; Christian Theobalt; Marc Habermann
>
> **备注:** For video results, see https://youtu.be/XMNCy7J2tuc
>
> **摘要:** Learning an animatable and clothed human avatar model with vivid dynamics and photorealistic appearance from multi-view videos is an important foundational research problem in computer graphics and vision. Fueled by recent advances in implicit representations, the quality of the animatable avatars has achieved an unprecedented level by attaching the implicit representation to drivable human template meshes. However, they usually fail to preserve the highest level of detail, particularly apparent when the virtual camera is zoomed in and when rendering at 4K resolution and higher. We argue that this limitation stems from inaccurate surface tracking, specifically, depth misalignment and surface drift between character geometry and the ground truth surface, which forces the detailed appearance model to compensate for geometric errors. To address this, we propose a latent deformation model and supervising the 3D deformation of the animatable character using guidance from foundational 2D video point trackers, which offer improved robustness to shading and surface variations, and are less prone to local minima than differentiable rendering. To mitigate the drift over time and lack of 3D awareness of 2D point trackers, we introduce a cascaded training strategy that generates consistent 3D point tracks by anchoring point tracks to the rendered avatar, which ultimately supervises our avatar at the vertex and texel level. To validate the effectiveness of our approach, we introduce a novel dataset comprising five multi-view video sequences, each over 10 minutes in duration, captured using 40 calibrated 6K-resolution cameras, featuring subjects dressed in clothing with challenging texture patterns and wrinkle deformations. Our approach demonstrates significantly improved performance in rendering quality and geometric accuracy over the prior state of the art.
>
---
#### [new 156] CineMA: A Foundation Model for Cine Cardiac MRI
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决心脏MRI中手动提取临床指标耗时且主观的问题。作者开发了CineMA——首个基于自监督学习的心脏MRI基础模型，通过预训练和微调，在多个任务（如分割、射血分数计算、疾病分类等）上达到或超越传统CNN效果，并减少标注需求。**

- **链接: [http://arxiv.org/pdf/2506.00679v1](http://arxiv.org/pdf/2506.00679v1)**

> **作者:** Yunguan Fu; Weixi Yi; Charlotte Manisty; Anish N Bhuva; Thomas A Treibel; James C Moon; Matthew J Clarkson; Rhodri Huw Davies; Yipeng Hu
>
> **摘要:** Cardiac magnetic resonance (CMR) is a key investigation in clinical cardiovascular medicine and has been used extensively in population research. However, extracting clinically important measurements such as ejection fraction for diagnosing cardiovascular diseases remains time-consuming and subjective. We developed CineMA, a foundation AI model automating these tasks with limited labels. CineMA is a self-supervised autoencoder model trained on 74,916 cine CMR studies to reconstruct images from masked inputs. After fine-tuning, it was evaluated across eight datasets on 23 tasks from four categories: ventricle and myocardium segmentation, left and right ventricle ejection fraction calculation, disease detection and classification, and landmark localisation. CineMA is the first foundation model for cine CMR to match or outperform convolutional neural networks (CNNs). CineMA demonstrated greater label efficiency than CNNs, achieving comparable or better performance with fewer annotations. This reduces the burden of clinician labelling and supports replacing task-specific training with fine-tuning foundation models in future cardiac imaging applications. Models and code for pre-training and fine-tuning are available at https://github.com/mathpluscode/CineMA, democratising access to high-performance models that otherwise require substantial computational resources, promoting reproducibility and accelerating clinical translation.
>
---
#### [new 157] NTIRE 2025 the 2nd Restore Any Image Model (RAIM) in the Wild Challenge
- **分类: cs.CV**

- **简介: 该论文介绍了NTIRE 2025的RAIM挑战赛，旨在解决真实场景中复杂退化图像的恢复问题。比赛包含两个任务：低光去噪与去马赛克、图像细节增强/生成。每个任务分为有监督和无监督子任务，评估兼顾客观指标与主观质量。吸引了近300支队伍参与，推动了图像恢复领域的技术进步。**

- **链接: [http://arxiv.org/pdf/2506.01394v1](http://arxiv.org/pdf/2506.01394v1)**

> **作者:** Jie Liang; Radu Timofte; Qiaosi Yi; Zhengqiang Zhang; Shuaizheng Liu; Lingchen Sun; Rongyuan Wu; Xindong Zhang; Hui Zeng; Lei Zhang
>
> **摘要:** In this paper, we present a comprehensive overview of the NTIRE 2025 challenge on the 2nd Restore Any Image Model (RAIM) in the Wild. This challenge established a new benchmark for real-world image restoration, featuring diverse scenarios with and without reference ground truth. Participants were tasked with restoring real-captured images suffering from complex and unknown degradations, where both perceptual quality and fidelity were critically evaluated. The challenge comprised two tracks: (1) the low-light joint denoising and demosaicing (JDD) task, and (2) the image detail enhancement/generation task. Each track included two sub-tasks. The first sub-task involved paired data with available ground truth, enabling quantitative evaluation. The second sub-task dealt with real-world yet unpaired images, emphasizing restoration efficiency and subjective quality assessed through a comprehensive user study. In total, the challenge attracted nearly 300 registrations, with 51 teams submitting more than 600 results. The top-performing methods advanced the state of the art in image restoration and received unanimous recognition from all 20+ expert judges. The datasets used in Track 1 and Track 2 are available at https://drive.google.com/drive/folders/1Mgqve-yNcE26IIieI8lMIf-25VvZRs_J and https://drive.google.com/drive/folders/1UB7nnzLwqDZOwDmD9aT8J0KVg2ag4Qae, respectively. The official challenge pages for Track 1 and Track 2 can be found at https://codalab.lisn.upsaclay.fr/competitions/21334#learn_the_details and https://codalab.lisn.upsaclay.fr/competitions/21623#learn_the_details.
>
---
#### [new 158] DeepVerse: 4D Autoregressive Video Generation as a World Model
- **分类: cs.CV**

- **简介: 论文提出DeepVerse，一种基于几何感知的4D世界模型，用于视频生成。该模型通过结合几何结构预测与动作条件，提升时空一致性，减少误差累积，实现长期高保真视频预测。任务是视频生成，解决现有方法忽略隐藏几何状态导致的时空不一致问题。**

- **链接: [http://arxiv.org/pdf/2506.01103v1](http://arxiv.org/pdf/2506.01103v1)**

> **作者:** Junyi Chen; Haoyi Zhu; Xianglong He; Yifan Wang; Jianjun Zhou; Wenzheng Chang; Yang Zhou; Zizun Li; Zhoujie Fu; Jiangmiao Pang; Tong He
>
> **摘要:** World models serve as essential building blocks toward Artificial General Intelligence (AGI), enabling intelligent agents to predict future states and plan actions by simulating complex physical interactions. However, existing interactive models primarily predict visual observations, thereby neglecting crucial hidden states like geometric structures and spatial coherence. This leads to rapid error accumulation and temporal inconsistency. To address these limitations, we introduce DeepVerse, a novel 4D interactive world model explicitly incorporating geometric predictions from previous timesteps into current predictions conditioned on actions. Experiments demonstrate that by incorporating explicit geometric constraints, DeepVerse captures richer spatio-temporal relationships and underlying physical dynamics. This capability significantly reduces drift and enhances temporal consistency, enabling the model to reliably generate extended future sequences and achieve substantial improvements in prediction accuracy, visual realism, and scene rationality. Furthermore, our method provides an effective solution for geometry-aware memory retrieval, effectively preserving long-term spatial consistency. We validate the effectiveness of DeepVerse across diverse scenarios, establishing its capacity for high-fidelity, long-horizon predictions grounded in geometry-aware dynamics.
>
---
#### [new 159] Towards Scalable Video Anomaly Retrieval: A Synthetic Video-Text Benchmark
- **分类: cs.CV**

- **简介: 该论文属于视频-文本检索任务，旨在解决真实世界中异常事件数据稀缺和隐私问题。作者构建了大规模合成数据集SVTA，包含41,315个视频及描述，涵盖30类正常活动与68类异常事件。利用生成模型克服数据限制，验证了跨模态检索方法的有效性与挑战性。**

- **链接: [http://arxiv.org/pdf/2506.01466v1](http://arxiv.org/pdf/2506.01466v1)**

> **作者:** Shuyu Yang; Yilun Wang; Yaxiong Wang; Li Zhu; Zhedong Zheng
>
> **摘要:** Video anomaly retrieval aims to localize anomalous events in videos using natural language queries to facilitate public safety. However, existing datasets suffer from severe limitations: (1) data scarcity due to the long-tail nature of real-world anomalies, and (2) privacy constraints that impede large-scale collection. To address the aforementioned issues in one go, we introduce SVTA (Synthetic Video-Text Anomaly benchmark), the first large-scale dataset for cross-modal anomaly retrieval, leveraging generative models to overcome data availability challenges. Specifically, we collect and generate video descriptions via the off-the-shelf LLM (Large Language Model) covering 68 anomaly categories, e.g., throwing, stealing, and shooting. These descriptions encompass common long-tail events. We adopt these texts to guide the video generative model to produce diverse and high-quality videos. Finally, our SVTA involves 41,315 videos (1.36M frames) with paired captions, covering 30 normal activities, e.g., standing, walking, and sports, and 68 anomalous events, e.g., falling, fighting, theft, explosions, and natural disasters. We adopt three widely-used video-text retrieval baselines to comprehensively test our SVTA, revealing SVTA's challenging nature and its effectiveness in evaluating a robust cross-modal retrieval method. SVTA eliminates privacy risks associated with real-world anomaly collection while maintaining realistic scenarios. The dataset demo is available at: [https://svta-mm.github.io/SVTA.github.io/].
>
---
#### [new 160] Zoom-Refine: Boosting High-Resolution Multimodal Understanding via Localized Zoom and Self-Refinement
- **分类: cs.CV**

- **简介: 该论文属于多模态理解任务，旨在解决现有模型在高分辨率图像理解中难以捕捉细节的问题。作者提出Zoom-Refine方法，通过“局部放大”定位关键区域，并结合初始推理结果进行“自精炼”，提升模型对复杂视觉细节的理解能力。整个过程无需额外训练或外部专家支持。**

- **链接: [http://arxiv.org/pdf/2506.01663v1](http://arxiv.org/pdf/2506.01663v1)**

> **作者:** Xuan Yu; Dayan Guan; Michael Ying Yang; Yanfeng Gu
>
> **备注:** Code is available at https://github.com/xavier-yu114/Zoom-Refine
>
> **摘要:** Multimodal Large Language Models (MLLM) often struggle to interpret high-resolution images accurately, where fine-grained details are crucial for complex visual understanding. We introduce Zoom-Refine, a novel training-free method that enhances MLLM capabilities to address this issue. Zoom-Refine operates through a synergistic process of \textit{Localized Zoom} and \textit{Self-Refinement}. In the \textit{Localized Zoom} step, Zoom-Refine leverages the MLLM to provide a preliminary response to an input query and identifies the most task-relevant image region by predicting its bounding box coordinates. During the \textit{Self-Refinement} step, Zoom-Refine then integrates fine-grained details from the high-resolution crop (identified by \textit{Localized Zoom}) with its initial reasoning to re-evaluate and refine its preliminary response. Our method harnesses the MLLM's inherent capabilities for spatial localization, contextual reasoning and comparative analysis without requiring additional training or external experts. Comprehensive experiments demonstrate the efficacy of Zoom-Refine on two challenging high-resolution multimodal benchmarks. Code is available at \href{https://github.com/xavier-yu114/Zoom-Refine}{\color{magenta}github.com/xavier-yu114/Zoom-Refine}
>
---
#### [new 161] VideoCap-R1: Enhancing MLLMs for Video Captioning via Structured Thinking
- **分类: cs.CV**

- **简介: 该论文属于视频描述生成任务，旨在提升多模态大语言模型（MLLM）在视频字幕生成中的动作描述能力。通过引入基于GRPO的强化学习后训练框架，结合结构化思考与双奖励机制，实现更准确的动作和对象描述，显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2506.01725v1](http://arxiv.org/pdf/2506.01725v1)**

> **作者:** Desen Meng; Rui Huang; Zhilin Dai; Xinhao Li; Yifan Xu; Jun Zhang; Zhenpeng Huang; Meng Zhang; Lingshu Zhang; Yi Liu; Limin Wang
>
> **摘要:** While recent advances in reinforcement learning have significantly enhanced reasoning capabilities in large language models (LLMs), these techniques remain underexplored in multi-modal LLMs for video captioning. This paper presents the first systematic investigation of GRPO-based RL post-training for video MLLMs, with the goal of enhancing video MLLMs' capability of describing actions in videos. Specifically, we develop the VideoCap-R1, which is prompted to first perform structured thinking that analyzes video subjects with their attributes and actions before generating complete captions, supported by two specialized reward mechanisms: a LLM-free think scorer evaluating the structured thinking quality and a LLM-assisted caption scorer assessing the output quality. The RL training framework effectively establishes the connection between structured reasoning and comprehensive description generation, enabling the model to produce captions with more accurate actions. Our experiments demonstrate that VideoCap-R1 achieves substantial improvements over the Qwen2VL-7B baseline using limited samples (1.5k) across multiple video caption benchmarks (DREAM1K: +4.4 event F1, VDC: +4.2 Acc, CAREBENCH: +3.1 action F1, +6.9 object F1) while consistently outperforming the SFT-trained counterparts, confirming GRPO's superiority in enhancing MLLMs' captioning capabilities.
>
---
#### [new 162] unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于无监督多目标分割任务，旨在解决单张真实图像中复杂场景的多物体分割问题。现有方法受限于简单合成场景或预训练特征。论文提出unMORE，通过中心-边界推理的两阶段方法，显式学习对象表示，并在无需标注和网络的情况下实现多目标发现，显著优于现有方法，尤其在拥挤场景中表现突出。**

- **链接: [http://arxiv.org/pdf/2506.01778v1](http://arxiv.org/pdf/2506.01778v1)**

> **作者:** Yafei Yang; Zihui Zhang; Bo Yang
>
> **备注:** ICML 2025. Code and data are available at: https://github.com/vLAR-group/unMORE
>
> **摘要:** We study the challenging problem of unsupervised multi-object segmentation on single images. Existing methods, which rely on image reconstruction objectives to learn objectness or leverage pretrained image features to group similar pixels, often succeed only in segmenting simple synthetic objects or discovering a limited number of real-world objects. In this paper, we introduce unMORE, a novel two-stage pipeline designed to identify many complex objects in real-world images. The key to our approach involves explicitly learning three levels of carefully defined object-centric representations in the first stage. Subsequently, our multi-object reasoning module utilizes these learned object priors to discover multiple objects in the second stage. Notably, this reasoning module is entirely network-free and does not require human labels. Extensive experiments demonstrate that unMORE significantly outperforms all existing unsupervised methods across 6 real-world benchmark datasets, including the challenging COCO dataset, achieving state-of-the-art object segmentation results. Remarkably, our method excels in crowded images where all baselines collapse.
>
---
#### [new 163] LongDWM: Cross-Granularity Distillation for Building a Long-Term Driving World Model
- **分类: cs.CV**

- **简介: 论文提出LongDWM，用于构建长期驾驶世界模型。任务是生成连贯的长视频以模拟未来驾驶场景。解决当前模型预测长期未来时误差累积严重、生成不连贯的问题。方法包括分层解耦运动学习与跨粒度蒸馏，提升视频生成质量与效率。**

- **链接: [http://arxiv.org/pdf/2506.01546v1](http://arxiv.org/pdf/2506.01546v1)**

> **作者:** Xiaodong Wang; Zhirong Wu; Peixi Peng
>
> **备注:** project homepage: https://wang-xiaodong1899.github.io/longdwm/
>
> **摘要:** Driving world models are used to simulate futures by video generation based on the condition of the current state and actions. However, current models often suffer serious error accumulations when predicting the long-term future, which limits the practical application. Recent studies utilize the Diffusion Transformer (DiT) as the backbone of driving world models to improve learning flexibility. However, these models are always trained on short video clips (high fps and short duration), and multiple roll-out generations struggle to produce consistent and reasonable long videos due to the training-inference gap. To this end, we propose several solutions to build a simple yet effective long-term driving world model. First, we hierarchically decouple world model learning into large motion learning and bidirectional continuous motion learning. Then, considering the continuity of driving scenes, we propose a simple distillation method where fine-grained video flows are self-supervised signals for coarse-grained flows. The distillation is designed to improve the coherence of infinite video generation. The coarse-grained and fine-grained modules are coordinated to generate long-term and temporally coherent videos. In the public benchmark NuScenes, compared with the state-of-the-art front-view model, our model improves FVD by $27\%$ and reduces inference time by $85\%$ for the video task of generating 110+ frames. More videos (including 90s duration) are available at https://Wang-Xiaodong1899.github.io/longdwm/.
>
---
#### [new 164] CountingFruit: Real-Time 3D Fruit Counting with Language-Guided Semantic Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于农业场景中的3D水果计数任务，旨在解决因遮挡、语义模糊和高计算需求导致的准确水果计数难题。论文提出FruitLangGS框架，结合空间重建、语义嵌入和语言引导实例估计，实现高效的实时3D水果计数。**

- **链接: [http://arxiv.org/pdf/2506.01109v1](http://arxiv.org/pdf/2506.01109v1)**

> **作者:** Fengze Li; Yangle Liu; Jieming Ma; Hai-Ning Liang; Yaochun Shen; Huangxiang Li; Zhijing Wu
>
> **摘要:** Accurate fruit counting in real-world agricultural environments is a longstanding challenge due to visual occlusions, semantic ambiguity, and the high computational demands of 3D reconstruction. Existing methods based on neural radiance fields suffer from low inference speed, limited generalization, and lack support for open-set semantic control. This paper presents FruitLangGS, a real-time 3D fruit counting framework that addresses these limitations through spatial reconstruction, semantic embedding, and language-guided instance estimation. FruitLangGS first reconstructs orchard-scale scenes using an adaptive Gaussian splatting pipeline with radius-aware pruning and tile-based rasterization for efficient rendering. To enable semantic control, each Gaussian encodes a compressed CLIP-aligned language embedding, forming a compact and queryable 3D representation. At inference time, prompt-based semantic filtering is applied directly in 3D space, without relying on image-space segmentation or view-level fusion. The selected Gaussians are then converted into dense point clouds via distribution-aware sampling and clustered to estimate fruit counts. Experimental results on real orchard data demonstrate that FruitLangGS achieves higher rendering speed, semantic flexibility, and counting accuracy compared to prior approaches, offering a new perspective for language-driven, real-time neural rendering across open-world scenarios.
>
---
#### [new 165] Event-based multi-view photogrammetry for high-dynamic, high-velocity target measurement
- **分类: cs.CV**

- **简介: 该论文属于运动测量任务，旨在解决高速目标动态特性测量难题。现有方法存在动态范围小、观测不连续等问题。作者提出基于事件的多视角摄影测量系统，利用时空事件单调性提取特征、通过重投影误差关联轨迹，并结合速度衰减模型实现精确测量，实验验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2506.00578v1](http://arxiv.org/pdf/2506.00578v1)**

> **作者:** Taihang Lei; Banglei Guan; Minzu Liang; Xiangyu Li; Jianbing Liu; Jing Tao; Yang Shang; Qifeng Yu
>
> **备注:** 9 pages, 9 figures, 1 table. This paper was accepted by Acta Mechanica Sinica (Date:30.May 2025)
>
> **摘要:** The characterization of mechanical properties for high-dynamic, high-velocity target motion is essential in industries. It provides crucial data for validating weapon systems and precision manufacturing processes etc. However, existing measurement methods face challenges such as limited dynamic range, discontinuous observations, and high costs. This paper presents a new approach leveraging an event-based multi-view photogrammetric system, which aims to address the aforementioned challenges. First, the monotonicity in the spatiotemporal distribution of events is leveraged to extract the target's leading-edge features, eliminating the tailing effect that complicates motion measurements. Then, reprojection error is used to associate events with the target's trajectory, providing more data than traditional intersection methods. Finally, a target velocity decay model is employed to fit the data, enabling accurate motion measurements via ours multi-view data joint computation. In a light gas gun fragment test, the proposed method showed a measurement deviation of 4.47% compared to the electromagnetic speedometer.
>
---
#### [new 166] Poster: Adapting Pretrained Vision Transformers with LoRA Against Attack Vectors
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与模型安全任务，旨在解决图像分类器易受对抗攻击的问题。作者通过低秩适应（LoRA）调整预训练视觉Transformer的权重和类别，提升其鲁棒性，并实现无需重新训练的可扩展微调。**

- **链接: [http://arxiv.org/pdf/2506.00661v1](http://arxiv.org/pdf/2506.00661v1)**

> **作者:** Richard E. Neddo; Sean Willis; Zander Blasingame; Chen Liu
>
> **备注:** Presented at IEEE MOST 2025
>
> **摘要:** Image classifiers, such as those used for autonomous vehicle navigation, are largely known to be susceptible to adversarial attacks that target the input image set. There is extensive discussion on adversarial attacks including perturbations that alter the input images to cause malicious misclassifications without perceivable modification. This work proposes a countermeasure for such attacks by adjusting the weights and classes of pretrained vision transformers with a low-rank adaptation to become more robust against adversarial attacks and allow for scalable fine-tuning without retraining.
>
---
#### [new 167] Performance Analysis of Few-Shot Learning Approaches for Bangla Handwritten Character and Digit Recognition
- **分类: cs.CV**

- **简介: 该论文属于手写字符识别任务，旨在解决孟加拉语数据稀缺下的少样本学习问题。作者提出了SynergiProtoNet模型，结合聚类与嵌入框架，提升识别准确率，并在多种设置下超越现有方法。**

- **链接: [http://arxiv.org/pdf/2506.00447v1](http://arxiv.org/pdf/2506.00447v1)**

> **作者:** Mehedi Ahamed; Radib Bin Kabir; Tawsif Tashwar Dipto; Mueeze Al Mushabbir; Sabbir Ahmed; Md. Hasanul Kabir
>
> **摘要:** This study investigates the performance of few-shot learning (FSL) approaches in recognizing Bangla handwritten characters and numerals using limited labeled data. It demonstrates the applicability of these methods to scripts with intricate and complex structures, where dataset scarcity is a common challenge. Given the complexity of Bangla script, we hypothesize that models performing well on these characters can generalize effectively to languages of similar or lower structural complexity. To this end, we introduce SynergiProtoNet, a hybrid network designed to improve the recognition accuracy of handwritten characters and digits. The model integrates advanced clustering techniques with a robust embedding framework to capture fine-grained details and contextual nuances. It leverages multi-level (both high- and low-level) feature extraction within a prototypical learning framework. We rigorously benchmark SynergiProtoNet against several state-of-the-art few-shot learning models: BD-CSPN, Prototypical Network, Relation Network, Matching Network, and SimpleShot, across diverse evaluation settings including Monolingual Intra-Dataset Evaluation, Monolingual Inter-Dataset Evaluation, Cross-Lingual Transfer, and Split Digit Testing. Experimental results show that SynergiProtoNet consistently outperforms existing methods, establishing a new benchmark in few-shot learning for handwritten character and digit recognition. The code is available on GitHub: https://github.com/MehediAhamed/SynergiProtoNet.
>
---
#### [new 168] STORM: Benchmarking Visual Rating of MLLMs with a Comprehensive Ordinal Regression Dataset
- **分类: cs.CV**

- **简介: 该论文属于视觉评价任务，旨在解决多模态大语言模型（MLLMs）在视觉有序回归（OR）任务中表现不佳及缺乏相关数据集的问题。论文构建了包含14个数据集的STORM基准，并提出一种可解释的粗到精处理流程，以提升MLLM的视觉评级能力并推动相关研究。**

- **链接: [http://arxiv.org/pdf/2506.01738v1](http://arxiv.org/pdf/2506.01738v1)**

> **作者:** Jinhong Wang; Shuo Tong; Jian liu; Dongqi Tang; Jintai Chen; Haochao Ying; Hongxia Xu; Danny Chen; Jian Wu
>
> **备注:** underreview of NIPS2025 D&B track
>
> **摘要:** Visual rating is an essential capability of artificial intelligence (AI) for multi-dimensional quantification of visual content, primarily applied in ordinal regression (OR) tasks such as image quality assessment, facial age estimation, and medical image grading. However, current multi-modal large language models (MLLMs) under-perform in such visual rating ability while also suffering the lack of relevant datasets and benchmarks. In this work, we collect and present STORM, a data collection and benchmark for Stimulating Trustworthy Ordinal Regression Ability of MLLMs for universal visual rating. STORM encompasses 14 ordinal regression datasets across five common visual rating domains, comprising 655K image-level pairs and the corresponding carefully curated VQAs. Importantly, we also propose a coarse-to-fine processing pipeline that dynamically considers label candidates and provides interpretable thoughts, providing MLLMs with a general and trustworthy ordinal thinking paradigm. This benchmark aims to evaluate the all-in-one and zero-shot performance of MLLMs in scenarios requiring understanding of the essential common ordinal relationships of rating labels. Extensive experiments demonstrate the effectiveness of our framework and shed light on better fine-tuning strategies. The STORM dataset, benchmark, and pre-trained models are available on the following webpage to support further research in this area. Datasets and codes are released on the project page: https://storm-bench.github.io/.
>
---
#### [new 169] RadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的3D场景重建与雷达数据合成任务。旨在解决现有方法在复杂雷达噪声下重建效果差、无法生成真实雷达数据的问题。作者提出RadarSplat，结合高斯点绘与雷达噪声建模，实现高质量的雷达数据合成与3D重建，提升了在恶劣天气下的自动驾驶感知能力。**

- **链接: [http://arxiv.org/pdf/2506.01379v1](http://arxiv.org/pdf/2506.01379v1)**

> **作者:** Pou-Chun Kung; Skanda Harisha; Ram Vasudevan; Aline Eid; Katherine A. Skinner
>
> **摘要:** High-Fidelity 3D scene reconstruction plays a crucial role in autonomous driving by enabling novel data generation from existing datasets. This allows simulating safety-critical scenarios and augmenting training datasets without incurring further data collection costs. While recent advances in radiance fields have demonstrated promising results in 3D reconstruction and sensor data synthesis using cameras and LiDAR, their potential for radar remains largely unexplored. Radar is crucial for autonomous driving due to its robustness in adverse weather conditions like rain, fog, and snow, where optical sensors often struggle. Although the state-of-the-art radar-based neural representation shows promise for 3D driving scene reconstruction, it performs poorly in scenarios with significant radar noise, including receiver saturation and multipath reflection. Moreover, it is limited to synthesizing preprocessed, noise-excluded radar images, failing to address realistic radar data synthesis. To address these limitations, this paper proposes RadarSplat, which integrates Gaussian Splatting with novel radar noise modeling to enable realistic radar data synthesis and enhanced 3D reconstruction. Compared to the state-of-the-art, RadarSplat achieves superior radar image synthesis (+3.4 PSNR / 2.6x SSIM) and improved geometric reconstruction (-40% RMSE / 1.5x Accuracy), demonstrating its effectiveness in generating high-fidelity radar data and scene reconstruction. A project page is available at https://umautobots.github.io/radarsplat.
>
---
#### [new 170] HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医疗视觉-语言模型对齐任务，旨在解决模态不对齐导致的临床响应不可靠问题。作者提出HSCR方法，通过自对比奖励机制生成高质量偏好数据，并设计多层次优化策略，提升模型在问答、图像描述等任务中的对齐性与可信度。**

- **链接: [http://arxiv.org/pdf/2506.00805v1](http://arxiv.org/pdf/2506.00805v1)**

> **作者:** Songtao Jiang; Yan Zhang; Yeying Jin; Zhihang Tang; Yangyang Wu; Yang Feng; Jian Wu; Zuozhu Liu
>
> **摘要:** Medical Vision-Language Models (Med-VLMs) have achieved success across various tasks, yet most existing methods overlook the modality misalignment issue that can lead to untrustworthy responses in clinical settings. In this paper, we propose Hierarchical Self-Contrastive Rewarding (HSCR), a novel approach that addresses two critical challenges in Med-VLM alignment: 1) Cost-effective generation of high-quality preference data; 2) Capturing nuanced and context-aware preferences for improved alignment. HSCR first leverages the inherent capability of Med-VLMs to generate dispreferred responses with higher sampling probability. By analyzing output logit shifts after visual token dropout, we identify modality-coupled tokens that induce misalignment and derive an implicit alignment reward function. This function guides token replacement with hallucinated ones during decoding, producing high-quality dispreferred data. Furthermore, HSCR introduces a multi-level preference optimization strategy, which extends beyond traditional adjacent-level optimization by incorporating nuanced implicit preferences, leveraging relative quality in dispreferred data to capture subtle alignment cues for more precise and context-aware optimization. Extensive experiments across multiple medical tasks, including Med-VQA, medical image captioning and instruction following, demonstrate that HSCR not only enhances zero-shot performance but also significantly improves modality alignment and trustworthiness with just 2,000 training entries.
>
---
#### [new 171] Speed-up of Vision Transformer Models by Attention-aware Token Filtering
- **分类: cs.CV**

- **简介: 该论文属于图像检索任务，旨在解决视觉Transformer模型（ViT）计算开销大的问题。作者提出了一种名为Attention-aware Token Filtering（ATF）的方法，在不修改Transformer编码器的前提下，通过动态和静态策略过滤输入标记，减少计算量。实验表明该方法使SigLIP模型在保持检索准确率的同时实现2.8倍加速。**

- **链接: [http://arxiv.org/pdf/2506.01519v1](http://arxiv.org/pdf/2506.01519v1)**

> **作者:** Takahiro Naruko; Hiroaki Akutsu
>
> **摘要:** Vision Transformer (ViT) models have made breakthroughs in image embedding extraction, which provide state-of-the-art performance in tasks such as zero-shot image classification. However, the models suffer from a high computational burden. In this paper, we propose a novel speed-up method for ViT models called Attention-aware Token Filtering (ATF). ATF consists of two main ideas: a novel token filtering module and a filtering strategy. The token filtering module is introduced between a tokenizer and a transformer encoder of the ViT model, without modifying or fine-tuning of the transformer encoder. The module filters out tokens inputted to the encoder so that it keeps tokens in regions of specific object types dynamically and keeps tokens in regions that statically receive high attention in the transformer encoder. This filtering strategy maintains task accuracy while filtering out tokens inputted to the transformer encoder. Evaluation results on retrieval tasks show that ATF provides $2.8\times$ speed-up to a ViT model, SigLIP, while maintaining the retrieval recall rate.
>
---
#### [new 172] SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决基于流的大规模模型在分布匹配蒸馏中的收敛难题。作者提出隐式分布对齐和段内引导方法，使蒸馏过程在SD 3.5和FLUX等模型上成功收敛，并推出SenseFlow模型提升蒸馏效果。**

- **链接: [http://arxiv.org/pdf/2506.00523v1](http://arxiv.org/pdf/2506.00523v1)**

> **作者:** Xingtong Ge; Xin Zhang; Tongda Xu; Yi Zhang; Xinjie Zhang; Yan Wang; Jun Zhang
>
> **备注:** under review
>
> **摘要:** The Distribution Matching Distillation (DMD) has been successfully applied to text-to-image diffusion models such as Stable Diffusion (SD) 1.5. However, vanilla DMD suffers from convergence difficulties on large-scale flow-based text-to-image models, such as SD 3.5 and FLUX. In this paper, we first analyze the issues when applying vanilla DMD on large-scale models. Then, to overcome the scalability challenge, we propose implicit distribution alignment (IDA) to regularize the distance between the generator and fake distribution. Furthermore, we propose intra-segment guidance (ISG) to relocate the timestep importance distribution from the teacher model. With IDA alone, DMD converges for SD 3.5; employing both IDA and ISG, DMD converges for SD 3.5 and FLUX.1 dev. Along with other improvements such as scaled up discriminator models, our final model, dubbed \textbf{SenseFlow}, achieves superior performance in distillation for both diffusion based text-to-image models such as SDXL, and flow-matching models such as SD 3.5 Large and FLUX. The source code will be avaliable at https://github.com/XingtongGe/SenseFlow.
>
---
#### [new 173] DS-VTON: High-Quality Virtual Try-on via Disentangled Dual-Scale Generation
- **分类: cs.CV**

- **简介: 该论文属于虚拟试穿任务，旨在解决衣物与人体对齐和细节保留问题。DS-VTON提出双尺度生成框架，第一阶段生成低分辨率图像实现结构对齐，第二阶段通过残差扩散提升纹理质量，且无需分割掩码。**

- **链接: [http://arxiv.org/pdf/2506.00908v1](http://arxiv.org/pdf/2506.00908v1)**

> **作者:** Xianbing Sun; Yan Hong; Jiahui Zhan; Jun Lan; Huijia Zhu; Weiqiang Wang; Liqing Zhang; Jianfu Zhang
>
> **摘要:** Despite recent progress, most existing virtual try-on methods still struggle to simultaneously address two core challenges: accurately aligning the garment image with the target human body, and preserving fine-grained garment textures and patterns. In this paper, we propose DS-VTON, a dual-scale virtual try-on framework that explicitly disentangles these objectives for more effective modeling. DS-VTON consists of two stages: the first stage generates a low-resolution try-on result to capture the semantic correspondence between garment and body, where reduced detail facilitates robust structural alignment. The second stage introduces a residual-guided diffusion process that reconstructs high-resolution outputs by refining the residual between the two scales, focusing on texture fidelity. In addition, our method adopts a fully mask-free generation paradigm, eliminating reliance on human parsing maps or segmentation masks. By leveraging the semantic priors embedded in pretrained diffusion models, this design more effectively preserves the person's appearance and geometric consistency. Extensive experiments demonstrate that DS-VTON achieves state-of-the-art performance in both structural alignment and texture preservation across multiple standard virtual try-on benchmarks.
>
---
#### [new 174] Leveraging CLIP Encoder for Multimodal Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文属于多模态情感识别（MER）任务，旨在解决因数据集不足导致的性能瓶颈问题。作者提出基于CLIP模型的MER-CLIP框架，通过引入标签编码器和跨模态解码器，融合语义信息与多模态特征，实现更优的情感识别效果，并在多个基准数据集上取得了优于现有方法的表现。**

- **链接: [http://arxiv.org/pdf/2506.00903v1](http://arxiv.org/pdf/2506.00903v1)**

> **作者:** Yehun Song; Sunyoung Cho
>
> **备注:** Accepted at IEEE/CVF WACV 2025, pp.6115-6124, 2025
>
> **摘要:** Multimodal emotion recognition (MER) aims to identify human emotions by combining data from various modalities such as language, audio, and vision. Despite the recent advances of MER approaches, the limitations in obtaining extensive datasets impede the improvement of performance. To mitigate this issue, we leverage a Contrastive Language-Image Pre-training (CLIP)-based architecture and its semantic knowledge from massive datasets that aims to enhance the discriminative multimodal representation. We propose a label encoder-guided MER framework based on CLIP (MER-CLIP) to learn emotion-related representations across modalities. Our approach introduces a label encoder that treats labels as text embeddings to incorporate their semantic information, leading to the learning of more representative emotional features. To further exploit label semantics, we devise a cross-modal decoder that aligns each modality to a shared embedding space by sequentially fusing modality features based on emotion-related input from the label encoder. Finally, the label encoder-guided prediction enables generalization across diverse labels by embedding their semantic information as well as word labels. Experimental results show that our method outperforms the state-of-the-art MER methods on the benchmark datasets, CMU-MOSI and CMU-MOSEI.
>
---
#### [new 175] EarthMind: Towards Multi-Granular and Multi-Sensor Earth Observation with Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属于地球观测任务，旨在解决多传感器、多粒度遥感数据的理解问题。作者提出了EarthMind框架，包含空间注意力提示和跨模态融合技术，并构建了EarthMind-Bench基准进行评估。实验表明其性能优于GPT-4o，适用于统一处理多源地球观测数据。**

- **链接: [http://arxiv.org/pdf/2506.01667v1](http://arxiv.org/pdf/2506.01667v1)**

> **作者:** Yan Shu; Bin Ren; Zhitong Xiong; Danda Pani Paudel; Luc Van Gool; Begum Demir; Nicu Sebe; Paolo Rota
>
> **摘要:** Large Multimodal Models (LMMs) have demonstrated strong performance in various vision-language tasks. However, they often struggle to comprehensively understand Earth Observation (EO) data, which is critical for monitoring the environment and the effects of human activity on it. In this work, we present EarthMind, a novel vision-language framework for multi-granular and multi-sensor EO data understanding. EarthMind features two core components: (1) Spatial Attention Prompting (SAP), which reallocates attention within the LLM to enhance pixel-level understanding; and (2) Cross-modal Fusion, which aligns heterogeneous modalities into a shared space and adaptively reweighs tokens based on their information density for effective fusion. To facilitate multi-sensor fusion evaluation, we propose EarthMind-Bench, a comprehensive benchmark with over 2,000 human-annotated multi-sensor image-question pairs, covering a wide range of perception and reasoning tasks. Extensive experiments demonstrate the effectiveness of EarthMind. It achieves state-of-the-art performance on EarthMind-Bench, surpassing GPT-4o despite being only 4B in scale. Moreover, EarthMind outperforms existing methods on multiple public EO benchmarks, showcasing its potential to handle both multi-granular and multi-sensor challenges in a unified framework.
>
---
#### [new 176] Detection of Endangered Deer Species Using UAV Imagery: A Comparative Study Between Efficient Deep Learning Approaches
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉与生态保护任务，旨在解决无人机影像中濒危沼泽鹿检测困难的问题。作者比较了YOLOv11和RT-DETR模型的性能，引入精确分割掩码以实现精细化训练，提升了检测效果，为基于无人机的野生动物监测提供了更高效的AI方案。**

- **链接: [http://arxiv.org/pdf/2506.00154v1](http://arxiv.org/pdf/2506.00154v1)**

> **作者:** Agustín Roca; Gastón Castro; Gabriel Torre; Leonardo J. Colombo; Ignacio Mas; Javier Pereira; Juan I. Giribet
>
> **摘要:** This study compares the performance of state-of-the-art neural networks including variants of the YOLOv11 and RT-DETR models for detecting marsh deer in UAV imagery, in scenarios where specimens occupy a very small portion of the image and are occluded by vegetation. We extend previous analysis adding precise segmentation masks for our datasets enabling a fine-grained training of a YOLO model with a segmentation head included. Experimental results show the effectiveness of incorporating the segmentation head achieving superior detection performance. This work contributes valuable insights for improving UAV-based wildlife monitoring and conservation strategies through scalable and accurate AI-driven detection systems.
>
---
#### [new 177] Deformable registration and generative modelling of aortic anatomies by auto-decoders and neural ODEs
- **分类: cs.CV; cs.NA; math.NA; 68T07, 68U05,; J.3; I.2.m; I.4.m**

- **简介: 该论文属于医学图像分析与生成任务，旨在解决血管形状的配准与合成问题。作者提出了AD-SVFD模型，结合自解码器与神经ODE，实现对主动脉几何的形变配准和新解剖结构的生成。通过优化点云匹配并利用隐式表示，提升了精度与效率。**

- **链接: [http://arxiv.org/pdf/2506.00947v1](http://arxiv.org/pdf/2506.00947v1)**

> **作者:** Riccardo Tenderini; Luca Pegolotti; Fanwei Kong; Stefano Pagani; Francesco Regazzoni; Alison L. Marsden; Simone Deparis
>
> **备注:** 29 pages, 7 figures, 6 tables, 2 algorithms. Submitted to "npj Biological Physics and Mechanics". Dataset publicly available at https://doi.org/10.5281/zenodo.15494901
>
> **摘要:** This work introduces AD-SVFD, a deep learning model for the deformable registration of vascular shapes to a pre-defined reference and for the generation of synthetic anatomies. AD-SVFD operates by representing each geometry as a weighted point cloud and models ambient space deformations as solutions at unit time of ODEs, whose time-independent right-hand sides are expressed through artificial neural networks. The model parameters are optimized by minimizing the Chamfer Distance between the deformed and reference point clouds, while backward integration of the ODE defines the inverse transformation. A distinctive feature of AD-SVFD is its auto-decoder structure, that enables generalization across shape cohorts and favors efficient weight sharing. In particular, each anatomy is associated with a low-dimensional code that acts as a self-conditioning field and that is jointly optimized with the network parameters during training. At inference, only the latent codes are fine-tuned, substantially reducing computational overheads. Furthermore, the use of implicit shape representations enables generative applications: new anatomies can be synthesized by suitably sampling from the latent space and applying the corresponding inverse transformations to the reference geometry. Numerical experiments, conducted on healthy aortic anatomies, showcase the high-quality results of AD-SVFD, which yields extremely accurate approximations at competitive computational costs.
>
---
#### [new 178] Sheep Facial Pain Assessment Under Weighted Graph Neural Networks
- **分类: cs.CV**

- **简介: 该论文属于动物疼痛评估任务，旨在解决如何准确自动识别羊面部疼痛的问题。研究提出了一种加权图神经网络模型（WGNN）和新的羊面部关键点数据集，结合YOLOv8n检测模型实现高精度疼痛水平预测，提升了羊面部表情分析的自动化能力。**

- **链接: [http://arxiv.org/pdf/2506.01468v1](http://arxiv.org/pdf/2506.01468v1)**

> **作者:** Alam Noor; Luis Almeida; Mohamed Daoudi; Kai Li; Eduardo Tovar
>
> **备注:** 2025 19th International Conference on Automatic Face and Gesture Recognition (FG)
>
> **摘要:** Accurately recognizing and assessing pain in sheep is key to discern animal health and mitigating harmful situations. However, such accuracy is limited by the ability to manage automatic monitoring of pain in those animals. Facial expression scoring is a widely used and useful method to evaluate pain in both humans and other living beings. Researchers also analyzed the facial expressions of sheep to assess their health state and concluded that facial landmark detection and pain level prediction are essential. For this purpose, we propose a novel weighted graph neural network (WGNN) model to link sheep's detected facial landmarks and define pain levels. Furthermore, we propose a new sheep facial landmarks dataset that adheres to the parameters of the Sheep Facial Expression Scale (SPFES). Currently, there is no comprehensive performance benchmark that specifically evaluates the use of graph neural networks (GNNs) on sheep facial landmark data to detect and measure pain levels. The YOLOv8n detector architecture achieves a mean average precision (mAP) of 59.30% with the sheep facial landmarks dataset, among seven other detection models. The WGNN framework has an accuracy of 92.71% for tracking multiple facial parts expressions with the YOLOv8n lightweight on-board device deployment-capable model.
>
---
#### [new 179] Visual Embodied Brain: Let Multimodal Large Language Models See, Think, and Control in Spaces
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出VeBrain框架，旨在将多模态大语言模型（MLLMs）应用于具身机器人控制任务。要解决的问题是如何统一视觉感知、空间推理与物理交互能力。工作包括设计文本化控制任务映射、开发机器人适配器、构建VeBrain-600k数据集，并验证其在13个多模态和5个空间智能基准上的优越性能。**

- **链接: [http://arxiv.org/pdf/2506.00123v1](http://arxiv.org/pdf/2506.00123v1)**

> **作者:** Gen Luo; Ganlin Yang; Ziyang Gong; Guanzhou Chen; Haonan Duan; Erfei Cui; Ronglei Tong; Zhi Hou; Tianyi Zhang; Zhe Chen; Shenglong Ye; Lewei Lu; Jingbo Wang; Wenhai Wang; Jifeng Dai; Yu Qiao; Rongrong Ji; Xizhou Zhu
>
> **摘要:** The remarkable progress of Multimodal Large Language Models (MLLMs) has attracted increasing attention to extend them to physical entities like legged robot. This typically requires MLLMs to not only grasp multimodal understanding abilities, but also integrate visual-spatial reasoning and physical interaction capabilities. Nevertheless,existing methods struggle to unify these capabilities due to their fundamental differences.In this paper, we present the Visual Embodied Brain (VeBrain), a unified framework for perception, reasoning, and control in real world. VeBrain reformulates robotic control into common text-based MLLM tasks in the 2D visual space, thus unifying the objectives and mapping spaces of different tasks. Then, a novel robotic adapter is proposed to convert textual control signals from MLLMs to motion policies of real robots. From the data perspective, we further introduce VeBrain-600k, a high-quality instruction dataset encompassing various capabilities of VeBrain. In VeBrain-600k, we take hundreds of hours to collect, curate and annotate the data, and adopt multimodal chain-of-thought(CoT) to mix the different capabilities into a single conversation. Extensive experiments on 13 multimodal benchmarks and 5 spatial intelligence benchmarks demonstrate the superior performance of VeBrain to existing MLLMs like Qwen2.5-VL. When deployed to legged robots and robotic arms, VeBrain shows strong adaptability, flexibility, and compositional capabilities compared to existing methods. For example, compared to Qwen2.5-VL, VeBrain not only achieves substantial gains on MMVet by +5.6%, but also excels in legged robot tasks with +50% average gains.
>
---
#### [new 180] Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 论文提出“Foresight”，一种自适应层复用技术，用于加速高质量文本到视频生成。该方法在Diffusion Transformers中动态重用层输出，减少去噪步骤中的计算冗余，提升效率，同时保持生成质量，适用于多种模型，如OpenSora、Latte和CogVideoX。**

- **链接: [http://arxiv.org/pdf/2506.00329v1](http://arxiv.org/pdf/2506.00329v1)**

> **作者:** Muhammad Adnan; Nithesh Kurella; Akhil Arunkumar; Prashant J. Nair
>
> **摘要:** Diffusion Transformers (DiTs) achieve state-of-the-art results in text-to-image, text-to-video generation, and editing. However, their large model size and the quadratic cost of spatial-temporal attention over multiple denoising steps make video generation computationally expensive. Static caching mitigates this by reusing features across fixed steps but fails to adapt to generation dynamics, leading to suboptimal trade-offs between speed and quality. We propose Foresight, an adaptive layer-reuse technique that reduces computational redundancy across denoising steps while preserving baseline performance. Foresight dynamically identifies and reuses DiT block outputs for all layers across steps, adapting to generation parameters such as resolution and denoising schedules to optimize efficiency. Applied to OpenSora, Latte, and CogVideoX, Foresight achieves up to 1.63x end-to-end speedup, while maintaining video quality. The source code of Foresight is available at \texttt{https://github.com/STAR-Laboratory/foresight}.
>
---
#### [new 181] Applying Vision Transformers on Spectral Analysis of Astronomical Objects
- **分类: astro-ph.IM; cs.CV**

- **简介: 该论文属于天文光谱分析任务，旨在解决传统方法在特征提取和模型泛化上的局限性。作者将一维光谱转换为二维图像，利用预训练视觉Transformer进行分类和红移估计，提升了准确性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2506.00294v1](http://arxiv.org/pdf/2506.00294v1)**

> **作者:** Luis Felipe Strano Moraes; Ignacio Becker; Pavlos Protopapas; Guillermo Cabrera-Vives
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** We apply pre-trained Vision Transformers (ViTs), originally developed for image recognition, to the analysis of astronomical spectral data. By converting traditional one-dimensional spectra into two-dimensional image representations, we enable ViTs to capture both local and global spectral features through spatial self-attention. We fine-tune a ViT pretrained on ImageNet using millions of spectra from the SDSS and LAMOST surveys, represented as spectral plots. Our model is evaluated on key tasks including stellar object classification and redshift ($z$) estimation, where it demonstrates strong performance and scalability. We achieve classification accuracy higher than Support Vector Machines and Random Forests, and attain $R^2$ values comparable to AstroCLIP's spectrum encoder, even when generalizing across diverse object types. These results demonstrate the effectiveness of using pretrained vision models for spectroscopic data analysis. To our knowledge, this is the first application of ViTs to large-scale, which also leverages real spectroscopic data and does not rely on synthetic inputs.
>
---
#### [new 182] Using Diffusion Ensembles to Estimate Uncertainty for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶轨迹规划任务，旨在解决传统方法忽视不确定性或依赖专用表示的问题。作者提出EnDfuser，使用扩散模型作为轨迹规划器，结合注意力机制与感知信息，通过集成扩散生成多条候选轨迹，提供不确定性估计和可解释性。**

- **链接: [http://arxiv.org/pdf/2506.00560v1](http://arxiv.org/pdf/2506.00560v1)**

> **作者:** Florian Wintel; Sigmund H. Høeg; Gabriel Kiss; Frank Lindseth
>
> **摘要:** End-to-end planning systems for autonomous driving are improving rapidly, especially in closed-loop simulation environments like CARLA. Many such driving systems either do not consider uncertainty as part of the plan itself, or obtain it by using specialized representations that do not generalize. In this paper, we propose EnDfuser, an end-to-end driving system that uses a diffusion model as the trajectory planner. EnDfuser effectively leverages complex perception information like fused camera and LiDAR features, through combining attention pooling and trajectory planning into a single diffusion transformer module. Instead of committing to a single plan, EnDfuser produces a distribution of candidate trajectories (128 for our case) from a single perception frame through ensemble diffusion. By observing the full set of candidate trajectories, EnDfuser provides interpretability for uncertain, multi-modal future trajectory spaces, where there are multiple plausible options. EnDfuser achieves a competitive driving score of 70.1 on the Longest6 benchmark in CARLA with minimal concessions on inference speed. Our findings suggest that ensemble diffusion, used as a drop-in replacement for traditional point-estimate trajectory planning modules, can help improve the safety of driving decisions by modeling the uncertainty of the posterior trajectory distribution.
>
---
#### [new 183] PerFormer: A Permutation Based Vision Transformer for Remaining Useful Life Prediction
- **分类: cs.LG; cs.CV**

- **简介: 论文属于剩余使用寿命（RUL）预测任务，旨在提升设备退化系统的健康状态估计。针对时间序列数据缺乏空间信息、难以应用视觉Transformer（ViT）的问题，论文提出PerFormer模型，通过排列多变量传感器数据模拟图像结构，并设计新的排列损失函数。实验表明该方法在NASA数据集上优于CNN、RNN和现有Transformer模型。**

- **链接: [http://arxiv.org/pdf/2506.00259v1](http://arxiv.org/pdf/2506.00259v1)**

> **作者:** Zhengyang Fan; Wanru Li; Kuo-chu Chang; Ting Yuan
>
> **摘要:** Accurately estimating the remaining useful life (RUL) for degradation systems is crucial in modern prognostic and health management (PHM). Convolutional Neural Networks (CNNs), initially developed for tasks like image and video recognition, have proven highly effectively in RUL prediction, demonstrating remarkable performance. However, with the emergence of the Vision Transformer (ViT), a Transformer model tailored for computer vision tasks such as image classification, and its demonstrated superiority over CNNs, there is a natural inclination to explore its potential in enhancing RUL prediction accuracy. Nonetheless, applying ViT directly to multivariate sensor data for RUL prediction poses challenges, primarily due to the ambiguous nature of spatial information in time series data. To address this issue, we introduce the PerFormer, a permutation-based vision transformer approach designed to permute multivariate time series data, mimicking spatial characteristics akin to image data, thereby making it suitable for ViT. To generate the desired permutation matrix, we introduce a novel permutation loss function aimed at guiding the convergence of any matrix towards a permutation matrix. Our experiments on NASA's C-MAPSS dataset demonstrate the PerFormer's superior performance in RUL prediction compared to state-of-the-art methods employing CNNs, Recurrent Neural Networks (RNNs), and various Transformer models. This underscores its effectiveness and potential in PHM applications.
>
---
#### [new 184] GaussianFusion: Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 论文提出GaussianFusion，用于端到端自动驾驶的高斯融合框架。该方法通过2D高斯表示聚合多传感器信息，结合显式与隐式特征提升轨迹预测。设计级联规划头优化轨迹，解决现有方法解释性差或计算密集问题。属于自动驾驶感知与决策任务。**

- **链接: [http://arxiv.org/pdf/2506.00034v1](http://arxiv.org/pdf/2506.00034v1)**

> **作者:** Shuai Liu; Quanmin Liang; Zefeng Li; Boyang Li; Kai Huang
>
> **摘要:** Multi-sensor fusion is crucial for improving the performance and robustness of end-to-end autonomous driving systems. Existing methods predominantly adopt either attention-based flatten fusion or bird's eye view fusion through geometric transformations. However, these approaches often suffer from limited interpretability or dense computational overhead. In this paper, we introduce GaussianFusion, a Gaussian-based multi-sensor fusion framework for end-to-end autonomous driving. Our method employs intuitive and compact Gaussian representations as intermediate carriers to aggregate information from diverse sensors. Specifically, we initialize a set of 2D Gaussians uniformly across the driving scene, where each Gaussian is parameterized by physical attributes and equipped with explicit and implicit features. These Gaussians are progressively refined by integrating multi-modal features. The explicit features capture rich semantic and spatial information about the traffic scene, while the implicit features provide complementary cues beneficial for trajectory planning. To fully exploit rich spatial and semantic information in Gaussians, we design a cascade planning head that iteratively refines trajectory predictions through interactions with Gaussians. Extensive experiments on the NAVSIM and Bench2Drive benchmarks demonstrate the effectiveness and robustness of the proposed GaussianFusion framework. The source code will be released at https://github.com/Say2L/GaussianFusion.
>
---
#### [new 185] Silence is Golden: Leveraging Adversarial Examples to Nullify Audio Control in LDM-based Talking-Head Generation
- **分类: cs.GR; cs.CR; cs.CV; cs.SD**

- **简介: 该论文属于AI安全任务，旨在解决基于LDM的语音驱动 talking-head 视频生成技术可能引发的伦理问题。作者提出 Silencer 方法，通过 nullifying loss 和 anti-purification loss 抵抗音频控制与扩散净化，以保护肖像隐私。**

- **链接: [http://arxiv.org/pdf/2506.01591v1](http://arxiv.org/pdf/2506.01591v1)**

> **作者:** Yuan Gan; Jiaxu Miao; Yunze Wang; Yi Yang
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Advances in talking-head animation based on Latent Diffusion Models (LDM) enable the creation of highly realistic, synchronized videos. These fabricated videos are indistinguishable from real ones, increasing the risk of potential misuse for scams, political manipulation, and misinformation. Hence, addressing these ethical concerns has become a pressing issue in AI security. Recent proactive defense studies focused on countering LDM-based models by adding perturbations to portraits. However, these methods are ineffective at protecting reference portraits from advanced image-to-video animation. The limitations are twofold: 1) they fail to prevent images from being manipulated by audio signals, and 2) diffusion-based purification techniques can effectively eliminate protective perturbations. To address these challenges, we propose Silencer, a two-stage method designed to proactively protect the privacy of portraits. First, a nullifying loss is proposed to ignore audio control in talking-head generation. Second, we apply anti-purification loss in LDM to optimize the inverted latent feature to generate robust perturbations. Extensive experiments demonstrate the effectiveness of Silencer in proactively protecting portrait privacy. We hope this work will raise awareness among the AI security community regarding critical ethical issues related to talking-head generation techniques. Code: https://github.com/yuangan/Silencer.
>
---
#### [new 186] OG-VLA: 3D-Aware Vision Language Action Model via Orthographic Image Generation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人视觉语言动作（VLA）任务，旨在解决3D感知策略在新环境中的泛化能力不足问题。作者提出了OG-VLA模型，通过正交图像生成实现视角不变性，结合视觉、语言模型与扩散模型，提升了对未见场景的适应能力，并在多个基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.01196v1](http://arxiv.org/pdf/2506.01196v1)**

> **作者:** Ishika Singh; Ankit Goyal; Stan Birchfield; Dieter Fox; Animesh Garg; Valts Blukis
>
> **备注:** 17 pages
>
> **摘要:** We introduce OG-VLA, a novel architecture and learning framework that combines the generalization strengths of Vision Language Action models (VLAs) with the robustness of 3D-aware policies. We address the challenge of mapping natural language instructions and multi-view RGBD observations to quasi-static robot actions. 3D-aware robot policies achieve state-of-the-art performance on precise robot manipulation tasks, but struggle with generalization to unseen instructions, scenes, and objects. On the other hand, VLAs excel at generalizing across instructions and scenes, but can be sensitive to camera and robot pose variations. We leverage prior knowledge embedded in language and vision foundation models to improve generalization of 3D-aware keyframe policies. OG-VLA projects input observations from diverse views into a point cloud which is then rendered from canonical orthographic views, ensuring input view invariance and consistency between input and output spaces. These canonical views are processed with a vision backbone, a Large Language Model (LLM), and an image diffusion model to generate images that encode the next position and orientation of the end-effector on the input scene. Evaluations on the Arnold and Colosseum benchmarks demonstrate state-of-the-art generalization to unseen environments, with over 40% relative improvements while maintaining robust performance in seen settings. We also show real-world adaption in 3 to 5 demonstrations along with strong generalization. Videos and resources at https://og-vla.github.io/
>
---
#### [new 187] Neural Path Guiding with Distribution Factorization
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文属于计算机图形学中的渲染任务，旨在解决蒙特卡洛积分中采样效率低的问题。现有神经方法在分布表示上难以兼顾速度与表达能力。本文提出一种将2D方向域分布分解为两个1D概率分布函数的方法，并用神经网络建模，通过插值实现高效采样，同时使用缓存网络优化训练，提升了复杂光照场景下的渲染性能。**

- **链接: [http://arxiv.org/pdf/2506.00839v1](http://arxiv.org/pdf/2506.00839v1)**

> **作者:** Pedro Figueiredo; Qihao He; Nima Khademi Kalantari
>
> **备注:** 11 pages, 11 figures. Accepted to EGSR 2025
>
> **摘要:** In this paper, we present a neural path guiding method to aid with Monte Carlo (MC) integration in rendering. Existing neural methods utilize distribution representations that are either fast or expressive, but not both. We propose a simple, but effective, representation that is sufficiently expressive and reasonably fast. Specifically, we break down the 2D distribution over the directional domain into two 1D probability distribution functions (PDF). We propose to model each 1D PDF using a neural network that estimates the distribution at a set of discrete coordinates. The PDF at an arbitrary location can then be evaluated and sampled through interpolation. To train the network, we maximize the similarity of the learned and target distributions. To reduce the variance of the gradient during optimizations and estimate the normalization factor, we propose to cache the incoming radiance using an additional network. Through extensive experiments, we demonstrate that our approach is better than the existing methods, particularly in challenging scenes with complex light transport.
>
---
#### [new 188] $Ψ$-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于生成模型推理优化任务，旨在解决基于得分的生成模型在推理时奖励对齐效率低的问题。作者提出Ψ-Sampler，结合SMC与pCNL算法，通过从奖励感知后验初始化粒子，提升采样效率和对齐效果。**

- **链接: [http://arxiv.org/pdf/2506.01320v1](http://arxiv.org/pdf/2506.01320v1)**

> **作者:** Taehoon Yoon; Yunhong Min; Kyeongmin Yeo; Minhyuk Sung
>
> **摘要:** We introduce $\Psi$-Sampler, an SMC-based framework incorporating pCNL-based initial particle sampling for effective inference-time reward alignment with a score-based generative model. Inference-time reward alignment with score-based generative models has recently gained significant traction, following a broader paradigm shift from pre-training to post-training optimization. At the core of this trend is the application of Sequential Monte Carlo (SMC) to the denoising process. However, existing methods typically initialize particles from the Gaussian prior, which inadequately captures reward-relevant regions and results in reduced sampling efficiency. We demonstrate that initializing from the reward-aware posterior significantly improves alignment performance. To enable posterior sampling in high-dimensional latent spaces, we introduce the preconditioned Crank-Nicolson Langevin (pCNL) algorithm, which combines dimension-robust proposals with gradient-informed dynamics. This approach enables efficient and scalable posterior sampling and consistently improves performance across various reward alignment tasks, including layout-to-image generation, quantity-aware generation, and aesthetic-preference generation, as demonstrated in our experiments.
>
---
#### [new 189] Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System for Dynamic Interactions
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态对话系统任务，旨在解决现有聊天机器人在动态、多轮、多方对话中缺乏自然视听融合的问题。作者构建了多模态多轮多方对话数据集 $M^3C$，并提出具有多模态记忆检索能力的模型，实现更沉浸式的人机交互。**

- **链接: [http://arxiv.org/pdf/2506.00421v1](http://arxiv.org/pdf/2506.00421v1)**

> **作者:** Jihyoung Jang; Minwook Bae; Minji Kim; Dilek Hakkani-Tur; Hyounghun Kim
>
> **备注:** ACL 2025 (32 pages); Project website: https://m3c-dataset.github.io/
>
> **摘要:** As chatbots continue to evolve toward human-like, real-world, interactions, multimodality remains an active area of research and exploration. So far, efforts to integrate multimodality into chatbots have primarily focused on image-centric tasks, such as visual dialogue and image-based instructions, placing emphasis on the "eyes" of human perception while neglecting the "ears", namely auditory aspects. Moreover, these studies often center around static interactions that focus on discussing the modality rather than naturally incorporating it into the conversation, which limits the richness of simultaneous, dynamic engagement. Furthermore, while multimodality has been explored in multi-party and multi-session conversations, task-specific constraints have hindered its seamless integration into dynamic, natural conversations. To address these challenges, this study aims to equip chatbots with "eyes and ears" capable of more immersive interactions with humans. As part of this effort, we introduce a new multimodal conversation dataset, Multimodal Multi-Session Multi-Party Conversation ($M^3C$), and propose a novel multimodal conversation model featuring multimodal memory retrieval. Our model, trained on the $M^3C$, demonstrates the ability to seamlessly engage in long-term conversations with multiple speakers in complex, real-world-like settings, effectively processing visual and auditory inputs to understand and respond appropriately. Human evaluations highlight the model's strong performance in maintaining coherent and dynamic interactions, demonstrating its potential for advanced multimodal conversational agents.
>
---
#### [new 190] DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DualMap，用于自然语言导航的在线开放词汇语义地图系统。任务是机器人在动态变化环境中实现语义理解和导航。解决了3D物体合并成本高和环境动态更新问题。工作包括混合分割前端、对象状态检测与双图表示方法。**

- **链接: [http://arxiv.org/pdf/2506.01950v1](http://arxiv.org/pdf/2506.01950v1)**

> **作者:** Jiajun Jiang; Yiming Zhu; Zirui Wu; Jie Song
>
> **备注:** 8 pages, 6 figures. Code: https://github.com/Eku127/DualMap Project page: https://eku127.github.io/DualMap/
>
> **摘要:** We introduce DualMap, an online open-vocabulary mapping system that enables robots to understand and navigate dynamically changing environments through natural language queries. Designed for efficient semantic mapping and adaptability to changing environments, DualMap meets the essential requirements for real-world robot navigation applications. Our proposed hybrid segmentation frontend and object-level status check eliminate the costly 3D object merging required by prior methods, enabling efficient online scene mapping. The dual-map representation combines a global abstract map for high-level candidate selection with a local concrete map for precise goal-reaching, effectively managing and updating dynamic changes in the environment. Through extensive experiments in both simulation and real-world scenarios, we demonstrate state-of-the-art performance in 3D open-vocabulary segmentation, efficient scene mapping, and online language-guided navigation.
>
---
#### [new 191] Adaptive Plane Reformatting for 4D Flow MRI using Deep Reinforcement Learning
- **分类: cs.LG; cs.CV; I.4.0**

- **简介: 该论文属于医学图像处理任务，旨在解决4D流MRI中平面重格式化的准确性与适应性问题。现有深度强化学习方法受限于训练和测试数据需一致对齐的问题。本文提出一种基于当前状态的灵活坐标系统，结合A3C算法，实现了更精准且适应性强的平面重格式化，提升了其在不同位置和方向影像中的适用性。**

- **链接: [http://arxiv.org/pdf/2506.00727v1](http://arxiv.org/pdf/2506.00727v1)**

> **作者:** Javier Bisbal; Julio Sotelo; Maria I Valdés; Pablo Irarrazaval; Marcelo E Andia; Julio García; José Rodriguez-Palomarez; Francesca Raimondi; Cristián Tejos; Sergio Uribe
>
> **备注:** 11 pages, 4 figures, submitted to IEEE Transactions on Medical Imaging
>
> **摘要:** Deep reinforcement learning (DRL) algorithms have shown robust results in plane reformatting tasks. In these methods, an agent sequentially adjusts the position and orientation of an initial plane towards an objective location. This process allows accurate plane reformatting, without the need for detailed landmarks, which makes it suitable for images with limited contrast and resolution, such as 4D flow MRI. However, current DRL methods require the test dataset to be in the same position and orientation as the training dataset. In this paper, we present a novel technique that utilizes a flexible coordinate system based on the current state, enabling navigation in volumes at any position or orientation. We adopted the Asynchronous Advantage Actor Critic (A3C) algorithm for reinforcement learning, outperforming Deep Q Network (DQN). Experimental results in 4D flow MRI demonstrate improved accuracy in plane reformatting angular and distance errors (6.32 +- 4.15 {\deg} and 3.40 +- 2.75 mm), as well as statistically equivalent flow measurements determined by a plane reformatting process done by an expert (p=0.21). The method's flexibility and adaptability make it a promising candidate for other medical imaging applications beyond 4D flow MRI.
>
---
#### [new 192] Transport Network, Graph, and Air Pollution
- **分类: physics.soc-ph; cs.CV**

- **简介: 该论文研究交通网络结构对城市空气污染的影响，属于城市规划与环境科学交叉任务。通过分析全球城市交通网络的几何和拓扑特征，提取12个指标探索网络与污染的相关性。论文旨在识别有助于缓解污染的交通网络模式，为城市规划提供依据。**

- **链接: [http://arxiv.org/pdf/2506.01164v1](http://arxiv.org/pdf/2506.01164v1)**

> **作者:** Nan Xu
>
> **摘要:** Air pollution can be studied in the urban structure regulated by transport networks. Transport networks can be studied as geometric and topological graph characteristics through designed models. Current studies do not offer a comprehensive view as limited models with insufficient features are examined. Our study finds geometric patterns of pollution-indicated transport networks through 0.3 million image interpretations of global cities. These are then described as part of 12 indices to investigate the network-pollution correlation. Strategies such as improved connectivity, more balanced road types and the avoidance of extreme clustering coefficient are identified as beneficial for alleviated pollution. As a graph-only study, it informs superior urban planning by separating the impact of permanent infrastructure from that of derived development for a more focused and efficient effort toward pollution reduction.
>
---
#### [new 193] A European Multi-Center Breast Cancer MRI Dataset
- **分类: eess.IV; cs.CV**

- **简介: 该论文旨在推动乳腺癌早期检测，任务是开发基于MRI和AI的自动化检测方法。为解决放射科医生阅片耗时长的问题，论文发布了欧洲多中心乳腺癌MRI数据集，供研究AI辅助诊断工具。**

- **链接: [http://arxiv.org/pdf/2506.00474v1](http://arxiv.org/pdf/2506.00474v1)**

> **作者:** Gustav Müller-Franzes; Lorena Escudero Sánchez; Nicholas Payne; Alexandra Athanasiou; Michael Kalogeropoulos; Aitor Lopez; Alfredo Miguel Soro Busto; Julia Camps Herrero; Nika Rasoolzadeh; Tianyu Zhang; Ritse Mann; Debora Jutz; Maike Bode; Christiane Kuhl; Wouter Veldhuis; Oliver Lester Saldanha; JieFu Zhu; Jakob Nikolas Kather; Daniel Truhn; Fiona J. Gilbert
>
> **摘要:** Detecting breast cancer early is of the utmost importance to effectively treat the millions of women afflicted by breast cancer worldwide every year. Although mammography is the primary imaging modality for screening breast cancer, there is an increasing interest in adding magnetic resonance imaging (MRI) to screening programmes, particularly for women at high risk. Recent guidelines by the European Society of Breast Imaging (EUSOBI) recommended breast MRI as a supplemental screening tool for women with dense breast tissue. However, acquiring and reading MRI scans requires significantly more time from expert radiologists. This highlights the need to develop new automated methods to detect cancer accurately using MRI and Artificial Intelligence (AI), which have the potential to support radiologists in breast MRI interpretation and classification and help detect cancer earlier. For this reason, the ODELIA consortium has made this multi-centre dataset publicly available to assist in developing AI tools for the detection of breast cancer on MRI.
>
---
#### [new 194] FreqPolicy: Frequency Autoregressive Visuomotor Policy with Continuous Tokens
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在学习有效的视觉运动策略。现有方法受限于动作表示和网络结构。论文提出FreqPolicy，将动作表示为频域成分，低频捕获全局运动，高频编码细节，并引入连续潜在表示提升精度。实验证明其在准确性和效率上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.01583v1](http://arxiv.org/pdf/2506.01583v1)**

> **作者:** Yiming Zhong; Yumeng Liu; Chuyang Xiao; Zemin Yang; Youzhuo Wang; Yufei Zhu; Ye Shi; Yujing Sun; Xinge Zhu; Yuexin Ma
>
> **摘要:** Learning effective visuomotor policies for robotic manipulation is challenging, as it requires generating precise actions while maintaining computational efficiency. Existing methods remain unsatisfactory due to inherent limitations in the essential action representation and the basic network architectures. We observe that representing actions in the frequency domain captures the structured nature of motion more effectively: low-frequency components reflect global movement patterns, while high-frequency components encode fine local details. Additionally, robotic manipulation tasks of varying complexity demand different levels of modeling precision across these frequency bands. Motivated by this, we propose a novel paradigm for visuomotor policy learning that progressively models hierarchical frequency components. To further enhance precision, we introduce continuous latent representations that maintain smoothness and continuity in the action space. Extensive experiments across diverse 2D and 3D robotic manipulation benchmarks demonstrate that our approach outperforms existing methods in both accuracy and efficiency, showcasing the potential of a frequency-domain autoregressive framework with continuous tokens for generalized robotic manipulation.
>
---
#### [new 195] From Motion to Behavior: Hierarchical Modeling of Humanoid Generative Behavior Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于行为建模任务，旨在解决当前人类运动生成忽视行为目标层次性的问题。作者提出Generative Behavior Control（GBC）框架，结合大语言模型与机器人任务规划，实现由高层意图驱动的多样化人体运动控制，并构建了包含语义与动作计划的GBC-100K数据集。实验表明该方法在生成质量、多样性和时间跨度上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.00043v1](http://arxiv.org/pdf/2506.00043v1)**

> **作者:** Jusheng Zhang; Jinzhou Tang; Sidi Liu; Mingyan Li; Sheng Zhang; Jian Wang; Keze Wang
>
> **摘要:** Human motion generative modeling or synthesis aims to characterize complicated human motions of daily activities in diverse real-world environments. However, current research predominantly focuses on either low-level, short-period motions or high-level action planning, without taking into account the hierarchical goal-oriented nature of human activities. In this work, we take a step forward from human motion generation to human behavior modeling, which is inspired by cognitive science. We present a unified framework, dubbed Generative Behavior Control (GBC), to model diverse human motions driven by various high-level intentions by aligning motions with hierarchical behavior plans generated by large language models (LLMs). Our insight is that human motions can be jointly controlled by task and motion planning in robotics, but guided by LLMs to achieve improved motion diversity and physical fidelity. Meanwhile, to overcome the limitations of existing benchmarks, i.e., lack of behavioral plans, we propose GBC-100K dataset annotated with a hierarchical granularity of semantic and motion plans driven by target goals. Our experiments demonstrate that GBC can generate more diverse and purposeful high-quality human motions with 10* longer horizons compared with existing methods when trained on GBC-100K, laying a foundation for future research on behavioral modeling of human motions. Our dataset and source code will be made publicly available.
>
---
#### [new 196] QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多模态临床决策任务，旨在解决现有模型在跨临床专科泛化能力差、数据分布不平衡的问题。作者提出QoQ-Med及DRPO训练方法，提升诊断性能并增强对医疗图像、文本等多模态数据的联合推理能力。**

- **链接: [http://arxiv.org/pdf/2506.00711v1](http://arxiv.org/pdf/2506.00711v1)**

> **作者:** Wei Dai; Peilin Chen; Chanakya Ekbote; Paul Pu Liang
>
> **摘要:** Clinical decision-making routinely demands reasoning over heterogeneous data, yet existing multimodal language models (MLLMs) remain largely vision-centric and fail to generalize across clinical specialties. To bridge this gap, we introduce QoQ-Med-7B/32B, the first open generalist clinical foundation model that jointly reasons across medical images, time-series signals, and text reports. QoQ-Med is trained with Domain-aware Relative Policy Optimization (DRPO), a novel reinforcement-learning objective that hierarchically scales normalized rewards according to domain rarity and modality difficulty, mitigating performance imbalance caused by skewed clinical data distributions. Trained on 2.61 million instruction tuning pairs spanning 9 clinical domains, we show that DRPO training boosts diagnostic performance by 43% in macro-F1 on average across all visual domains as compared to other critic-free training methods like GRPO. Furthermore, with QoQ-Med trained on intensive segmentation data, it is able to highlight salient regions related to the diagnosis, with an IoU 10x higher than open models while reaching the performance of OpenAI o4-mini. To foster reproducibility and downstream research, we release (i) the full model weights, (ii) the modular training pipeline, and (iii) all intermediate reasoning traces at https://github.com/DDVD233/QoQ_Med.
>
---
#### [new 197] Is Extending Modality The Right Path Towards Omni-Modality?
- **分类: cs.CL; cs.CV**

- **简介: 论文探讨如何实现真正的全模态（Omni-modality）语言模型，研究扩展模态是否会影响核心语言能力、模型融合是否有效，以及全模态扩展是否优于顺序扩展。属于多模态机器学习任务，旨在解决现有模型在多模态输入下泛化能力不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.01872v1](http://arxiv.org/pdf/2506.01872v1)**

> **作者:** Tinghui Zhu; Kai Zhang; Muhao Chen; Yu Su
>
> **摘要:** Omni-modal language models (OLMs) aim to integrate and reason over diverse input modalities--such as text, images, video, and audio--while maintaining strong language capabilities. Despite recent advancements, existing models, especially open-source ones, remain far from true omni-modality, struggling to generalize beyond the specific modality pairs they are trained on or to achieve strong performance when processing multi-modal inputs. We study the effect of extending modality, the dominant technique for training multimodal models, where an off-the-shelf language model is fine-tuned on target-domain and language data. Specifically, we investigate three key questions: (1) Does modality extension compromise core language abilities? (2) Can model merging effectively integrate independently fine-tuned modality-specific models to achieve omni-modality? (3) Does omni-modality extension lead to better knowledge sharing and generalization compared to sequential extension? Through extensive experiments, we analyze these trade-offs and provide insights into the feasibility of achieving true omni-modality using current approaches.
>
---
#### [new 198] SynPO: Synergizing Descriptiveness and Preference Optimization for Video Detailed Captioning
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于视频细粒度描述生成任务，旨在解决现有方法在捕捉视频细节和动态信息上的不足。作者提出SynPO方法，结合偏好学习优化视觉语言模型，提升描述的丰富性和连贯性，并设计新优化策略提高训练效率。**

- **链接: [http://arxiv.org/pdf/2506.00835v1](http://arxiv.org/pdf/2506.00835v1)**

> **作者:** Jisheng Dang; Yizhou Zhang; Hao Ye; Teng Wang; Siming Chen; Huicheng Zheng; Yulan Guo; Jianhuang Lai; Bin Hu
>
> **摘要:** Fine-grained video captioning aims to generate detailed, temporally coherent descriptions of video content. However, existing methods struggle to capture subtle video dynamics and rich detailed information. In this paper, we leverage preference learning to enhance the performance of vision-language models in fine-grained video captioning, while mitigating several limitations inherent to direct preference optimization (DPO). First, we propose a pipeline for constructing preference pairs that leverages the intrinsic properties of VLMs along with partial assistance from large language models, achieving an optimal balance between cost and data quality. Second, we propose Synergistic Preference Optimization (SynPO), a novel optimization method offering significant advantages over DPO and its variants. SynPO prevents negative preferences from dominating the optimization, explicitly preserves the model's language capability to avoid deviation of the optimization objective, and improves training efficiency by eliminating the need for the reference model. We extensively evaluate SynPO not only on video captioning benchmarks (e.g., VDC, VDD, VATEX) but also across well-established NLP tasks, including general language understanding and preference evaluation, using diverse pretrained models. Results demonstrate that SynPO consistently outperforms DPO variants while achieving 20\% improvement in training efficiency. Code is available at https://github.com/longmalongma/SynPO
>
---
#### [new 199] Understanding while Exploring: Semantics-driven Active Mapping
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于主动语义建图任务，旨在解决机器人在未知环境中高效探索与精确建图的问题。作者提出了ActiveSGM框架，基于3D高斯点绘图，结合语义和几何不确定性评估，指导机器人选择最优视角，提升建图的完整性、准确性及对噪声的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.00225v1](http://arxiv.org/pdf/2506.00225v1)**

> **作者:** Liyan Chen; Huangying Zhan; Hairong Yin; Yi Xu; Philippos Mordohai
>
> **摘要:** Effective robotic autonomy in unknown environments demands proactive exploration and precise understanding of both geometry and semantics. In this paper, we propose ActiveSGM, an active semantic mapping framework designed to predict the informativeness of potential observations before execution. Built upon a 3D Gaussian Splatting (3DGS) mapping backbone, our approach employs semantic and geometric uncertainty quantification, coupled with a sparse semantic representation, to guide exploration. By enabling robots to strategically select the most beneficial viewpoints, ActiveSGM efficiently enhances mapping completeness, accuracy, and robustness to noisy semantic data, ultimately supporting more adaptive scene exploration. Our experiments on the Replica and Matterport3D datasets highlight the effectiveness of ActiveSGM in active semantic mapping tasks.
>
---
#### [new 200] Variance-Based Defense Against Blended Backdoor Attacks
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于人工智能安全任务，旨在防御混合后门攻击。它提出一种基于方差的新方法，在无需干净数据集的情况下，检测中毒类别、提取触发器关键部分并识别恶意样本，提升模型可解释性与安全性。**

- **链接: [http://arxiv.org/pdf/2506.01444v1](http://arxiv.org/pdf/2506.01444v1)**

> **作者:** Sujeevan Aseervatham; Achraf Kerzazi; Younès Bennani
>
> **备注:** This paper has been accepted at ECML PKDD 2025
>
> **摘要:** Backdoor attacks represent a subtle yet effective class of cyberattacks targeting AI models, primarily due to their stealthy nature. The model behaves normally on clean data but exhibits malicious behavior only when the attacker embeds a specific trigger into the input. This attack is performed during the training phase, where the adversary corrupts a small subset of the training data by embedding a pattern and modifying the labels to a chosen target. The objective is to make the model associate the pattern with the target label while maintaining normal performance on unaltered data. Several defense mechanisms have been proposed to sanitize training data-sets. However, these methods often rely on the availability of a clean dataset to compute statistical anomalies, which may not always be feasible in real-world scenarios where datasets can be unavailable or compromised. To address this limitation, we propose a novel defense method that trains a model on the given dataset, detects poisoned classes, and extracts the critical part of the attack trigger before identifying the poisoned instances. This approach enhances explainability by explicitly revealing the harmful part of the trigger. The effectiveness of our method is demonstrated through experimental evaluations on well-known image datasets and comparative analysis against three state-of-the-art algorithms: SCAn, ABL, and AGPD.
>
---
#### [new 201] WoMAP: World Models For Embodied Open-Vocabulary Object Localization
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人主动目标定位任务，旨在解决语言指令下在部分可观环境中高效探索与定位物体的问题。作者提出WoMAP方法，结合高斯溅射仿真、开放词汇检测器和潜在世界模型，实现无需专家演示的策略训练，在零样本任务中表现出更高的成功率和泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.01600v1](http://arxiv.org/pdf/2506.01600v1)**

> **作者:** Tenny Yin; Zhiting Mei; Tao Sun; Lihan Zha; Emily Zhou; Jeremy Bao; Miyu Yamane; Ola Shorinwa; Anirudha Majumdar
>
> **摘要:** Language-instructed active object localization is a critical challenge for robots, requiring efficient exploration of partially observable environments. However, state-of-the-art approaches either struggle to generalize beyond demonstration datasets (e.g., imitation learning methods) or fail to generate physically grounded actions (e.g., VLMs). To address these limitations, we introduce WoMAP (World Models for Active Perception): a recipe for training open-vocabulary object localization policies that: (i) uses a Gaussian Splatting-based real-to-sim-to-real pipeline for scalable data generation without the need for expert demonstrations, (ii) distills dense rewards signals from open-vocabulary object detectors, and (iii) leverages a latent world model for dynamics and rewards prediction to ground high-level action proposals at inference time. Rigorous simulation and hardware experiments demonstrate WoMAP's superior performance in a broad range of zero-shot object localization tasks, with more than 9x and 2x higher success rates compared to VLM and diffusion policy baselines, respectively. Further, we show that WoMAP achieves strong generalization and sim-to-real transfer on a TidyBot.
>
---
#### [new 202] Learning Sparsity for Effective and Efficient Music Performance Question Answering
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于音乐表演音视频问答（Music AVQA）任务，旨在解决现有方法在处理密集、冗余数据时效率低下的问题。作者提出了Sparsify框架，通过三种稀疏学习策略，在保持准确率的同时提升训练效率，并设计了关键子集选择算法，仅用25%的数据达到70-80%的性能，显著提高数据效率。**

- **链接: [http://arxiv.org/pdf/2506.01319v1](http://arxiv.org/pdf/2506.01319v1)**

> **作者:** Xingjian Diao; Tianzhen Yang; Chunhui Zhang; Weiyi Wu; Ming Cheng; Jiang Gui
>
> **备注:** Accepted to the main conference of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** Music performances, characterized by dense and continuous audio as well as seamless audio-visual integration, present unique challenges for multimodal scene understanding and reasoning. Recent Music Performance Audio-Visual Question Answering (Music AVQA) datasets have been proposed to reflect these challenges, highlighting the continued need for more effective integration of audio-visual representations in complex question answering. However, existing Music AVQA methods often rely on dense and unoptimized representations, leading to inefficiencies in the isolation of key information, the reduction of redundancy, and the prioritization of critical samples. To address these challenges, we introduce Sparsify, a sparse learning framework specifically designed for Music AVQA. It integrates three sparsification strategies into an end-to-end pipeline and achieves state-of-the-art performance on the Music AVQA datasets. In addition, it reduces training time by 28.32% compared to its fully trained dense counterpart while maintaining accuracy, demonstrating clear efficiency gains. To further improve data efficiency, we propose a key-subset selection algorithm that selects and uses approximately 25% of MUSIC-AVQA v2.0 training data and retains 70-80% of full-data performance across models.
>
---
#### [new 203] Hanfu-Bench: A Multimodal Benchmark on Cross-Temporal Cultural Understanding and Transcreation
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出了Hanfu-Bench，一个关于跨时代文化理解与创意重构的多模态基准数据集。旨在解决现有视觉-语言模型在文化理解中忽视时间维度的问题。包含两个任务：文化视觉理解和文化图像创意重构。评估显示当前模型在这两个任务上仍显著落后于人类表现。**

- **链接: [http://arxiv.org/pdf/2506.01565v1](http://arxiv.org/pdf/2506.01565v1)**

> **作者:** Li Zhou; Lutong Yu; Dongchu Xie; Shaohuan Cheng; Wenyan Li; Haizhou Li
>
> **备注:** cultural analysis, cultural visual understanding, cultural image transcreation
>
> **摘要:** Culture is a rich and dynamic domain that evolves across both geography and time. However, existing studies on cultural understanding with vision-language models (VLMs) primarily emphasize geographic diversity, often overlooking the critical temporal dimensions. To bridge this gap, we introduce Hanfu-Bench, a novel, expert-curated multimodal dataset. Hanfu, a traditional garment spanning ancient Chinese dynasties, serves as a representative cultural heritage that reflects the profound temporal aspects of Chinese culture while remaining highly popular in Chinese contemporary society. Hanfu-Bench comprises two core tasks: cultural visual understanding and cultural image transcreation.The former task examines temporal-cultural feature recognition based on single- or multi-image inputs through multiple-choice visual question answering, while the latter focuses on transforming traditional attire into modern designs through cultural element inheritance and modern context adaptation. Our evaluation shows that closed VLMs perform comparably to non-experts on visual cutural understanding but fall short by 10\% to human experts, while open VLMs lags further behind non-experts. For the transcreation task, multi-faceted human evaluation indicates that the best-performing model achieves a success rate of only 42\%. Our benchmark provides an essential testbed, revealing significant challenges in this new direction of temporal cultural understanding and creative adaptation.
>
---
#### [new 204] LensCraft: Your Professional Virtual Cinematographer
- **分类: cs.GR; cs.CV**

- **简介: 论文提出LensCraft，一种虚拟电影摄影师系统，属于智能摄像任务。解决现有自动摄像系统难以平衡机械执行与创意意图、忽略主体体积的问题。工作包括：构建高仿真训练数据模拟框架，设计结合电影原则与神经模型的算法，支持多模态输入控制摄像，实现低计算、实时高效拍摄。**

- **链接: [http://arxiv.org/pdf/2506.00988v1](http://arxiv.org/pdf/2506.00988v1)**

> **作者:** Zahra Dehghanian; Morteza Abolghasemi; Hossein Azizinaghsh; Amir Vahedi; Hamid Beigy; Hamid R. Rabiee
>
> **摘要:** Digital creators, from indie filmmakers to animation studios, face a persistent bottleneck: translating their creative vision into precise camera movements. Despite significant progress in computer vision and artificial intelligence, current automated filming systems struggle with a fundamental trade-off between mechanical execution and creative intent. Crucially, almost all previous works simplify the subject to a single point-ignoring its orientation and true volume-severely limiting spatial awareness during filming. LensCraft solves this problem by mimicking the expertise of a professional cinematographer, using a data-driven approach that combines cinematographic principles with the flexibility to adapt to dynamic scenes in real time. Our solution combines a specialized simulation framework for generating high-fidelity training data with an advanced neural model that is faithful to the script while being aware of the volume and dynamic behavior of the subject. Additionally, our approach allows for flexible control via various input modalities, including text prompts, subject trajectory and volume, key points, or a full camera trajectory, offering creators a versatile tool to guide camera movements in line with their vision. Leveraging a lightweight real time architecture, LensCraft achieves markedly lower computational complexity and faster inference while maintaining high output quality. Extensive evaluation across static and dynamic scenarios reveals unprecedented accuracy and coherence, setting a new benchmark for intelligent camera systems compared to state-of-the-art models. Extended results, the complete dataset, simulation environment, trained model weights, and source code are publicly accessible on LensCraft Webpage.
>
---
#### [new 205] Understanding Model Reprogramming for CLIP via Decoupling Visual Prompts
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于视觉模型适配任务，旨在解决现有视觉重编程方法在适应下游任务时表现不佳的问题。作者提出解耦视觉提示（DVP）与概率重加权矩阵（PRM），通过分组描述学习多个提示并动态集成，提升CLIP模型的分类性能，并提供可解释性分析。**

- **链接: [http://arxiv.org/pdf/2506.01000v1](http://arxiv.org/pdf/2506.01000v1)**

> **作者:** Chengyi Cai; Zesheng Ye; Lei Feng; Jianzhong Qi; Feng Liu
>
> **摘要:** Model reprogramming adapts pretrained models to downstream tasks by modifying only the input and output spaces. Visual reprogramming (VR) is one instance for vision tasks that adds a trainable noise pattern (i.e., a visual prompt) to input images to facilitate downstream classification. The existing VR approaches for CLIP train a single visual prompt using all descriptions of different downstream classes. However, the limited learning capacity may result in (1) a failure to capture diverse aspects of the descriptions (e.g., shape, color, and texture), and (2) a possible bias toward less informative attributes that do not help distinguish between classes. In this paper, we introduce a decoupling-and-reweighting framework. Our decoupled visual prompts (DVP) are optimized using descriptions grouped by explicit causes (DVP-cse) or unsupervised clusters (DVP-cls). Then, we integrate the outputs of these visual prompts with a probabilistic reweighting matrix (PRM) that measures their contributions to each downstream class. Theoretically, DVP lowers the empirical risk bound. Experimentally, DVP outperforms baselines on average across 11 downstream datasets. Notably, the DVP-PRM integration enables insights into how individual visual prompts influence classification decisions, providing a probabilistic framework for understanding reprogramming. Our code is available at https://github.com/tmlr-group/DecoupledVP.
>
---
#### [new 206] ProtInvTree: Deliberate Protein Inverse Folding with Reward-guided Tree Search
- **分类: q-bio.BM; cs.CV; cs.LG**

- **简介: 该论文属于蛋白质逆折叠任务，旨在设计能折叠成目标结构的多样序列。现有方法多忽视问题的一对多特性，难以生成多样化结果。论文提出ProtInvTree，通过基于奖励的树搜索框架，结合预训练模型与两阶段动作机制，实现结构一致且多样化的序列设计。**

- **链接: [http://arxiv.org/pdf/2506.00925v1](http://arxiv.org/pdf/2506.00925v1)**

> **作者:** Mengdi Liu; Xiaoxue Cheng; Zhangyang Gao; Hong Chang; Cheng Tan; Shiguang Shan; Xilin Chen
>
> **摘要:** Designing protein sequences that fold into a target 3D structure, known as protein inverse folding, is a fundamental challenge in protein engineering. While recent deep learning methods have achieved impressive performance by recovering native sequences, they often overlook the one-to-many nature of the problem: multiple diverse sequences can fold into the same structure. This motivates the need for a generative model capable of designing diverse sequences while preserving structural consistency. To address this trade-off, we introduce ProtInvTree, the first reward-guided tree-search framework for protein inverse folding. ProtInvTree reformulates sequence generation as a deliberate, step-wise decision-making process, enabling the exploration of multiple design paths and exploitation of promising candidates through self-evaluation, lookahead, and backtracking. We propose a two-stage focus-and-grounding action mechanism that decouples position selection and residue generation. To efficiently evaluate intermediate states, we introduce a jumpy denoising strategy that avoids full rollouts. Built upon pretrained protein language models, ProtInvTree supports flexible test-time scaling by expanding the search depth and breadth without retraining. Empirically, ProtInvTree outperforms state-of-the-art baselines across multiple benchmarks, generating structurally consistent yet diverse sequences, including those far from the native ground truth.
>
---
#### [new 207] Multiverse Through Deepfakes: The MultiFakeVerse Dataset of Person-Centric Visual and Conceptual Manipulations
- **分类: cs.MM; cs.CV**

- **简介: 该论文属于图像生成与深度伪造检测任务，旨在解决当前缺乏大规模、语义驱动的深伪基准数据集问题。作者构建了MultiFakeVerse数据集，包含845,286张图像，基于视觉语言模型生成操作指令，实现对人物及场景的高层语义修改，用于评估现有检测方法和人类识别能力。**

- **链接: [http://arxiv.org/pdf/2506.00868v1](http://arxiv.org/pdf/2506.00868v1)**

> **作者:** Parul Gupta; Shreya Ghosh; Tom Gedeon; Thanh-Toan Do; Abhinav Dhall
>
> **摘要:** The rapid advancement of GenAI technology over the past few years has significantly contributed towards highly realistic deepfake content generation. Despite ongoing efforts, the research community still lacks a large-scale and reasoning capability driven deepfake benchmark dataset specifically tailored for person-centric object, context and scene manipulations. In this paper, we address this gap by introducing MultiFakeVerse, a large scale person-centric deepfake dataset, comprising 845,286 images generated through manipulation suggestions and image manipulations both derived from vision-language models (VLM). The VLM instructions were specifically targeted towards modifications to individuals or contextual elements of a scene that influence human perception of importance, intent, or narrative. This VLM-driven approach enables semantic, context-aware alterations such as modifying actions, scenes, and human-object interactions rather than synthetic or low-level identity swaps and region-specific edits that are common in existing datasets. Our experiments reveal that current state-of-the-art deepfake detection models and human observers struggle to detect these subtle yet meaningful manipulations. The code and dataset are available on \href{https://github.com/Parul-Gupta/MultiFakeVerse}{GitHub}.
>
---
#### [new 208] EffiVLM-BENCH: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于模型加速任务，旨在解决大视觉语言模型（LVLM）计算需求高、部署受限的问题。作者系统评估主流加速技术，提出EffiVLM-Bench框架，统一衡量性能、泛化性和保真度，并探索最优权衡策略，推动训练无关的LVLM加速方法研究。**

- **链接: [http://arxiv.org/pdf/2506.00479v1](http://arxiv.org/pdf/2506.00479v1)**

> **作者:** Zekun Wang; Minghua Ma; Zexin Wang; Rongchuan Mu; Liping Shan; Ming Liu; Bing Qin
>
> **备注:** ACL 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved remarkable success, yet their significant computational demands hinder practical deployment. While efforts to improve LVLM efficiency are growing, existing methods lack comprehensive evaluation across diverse backbones, benchmarks, and metrics. In this work, we systematically evaluate mainstream acceleration techniques for LVLMs, categorized into token and parameter compression. We introduce EffiVLM-Bench, a unified framework for assessing not only absolute performance but also generalization and loyalty, while exploring Pareto-optimal trade-offs. Our extensive experiments and in-depth analyses offer insights into optimal strategies for accelerating LVLMs. We open-source code and recipes for EffiVLM-Bench to foster future research.
>
---
#### [new 209] SST: Self-training with Self-adaptive Thresholding for Semi-supervised Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于半监督学习任务，旨在解决标注数据不足的问题。通过提出SST框架及其SAT机制，自适应调整伪标签阈值，提升模型性能。实验表明其在多个数据集和架构上效果优异，尤其在ImageNet-1K上用10%标注数据超越全监督模型。**

- **链接: [http://arxiv.org/pdf/2506.00467v1](http://arxiv.org/pdf/2506.00467v1)**

> **作者:** Shuai Zhao; Heyan Huang; Xinge Li; Xiaokang Chen; Rui Wang
>
> **备注:** Accepted by Information Processing & Management (IP&M)
>
> **摘要:** Neural networks have demonstrated exceptional performance in supervised learning, benefiting from abundant high-quality annotated data. However, obtaining such data in real-world scenarios is costly and labor-intensive. Semi-supervised learning (SSL) offers a solution to this problem. Recent studies, such as Semi-ViT and Noisy Student, which employ consistency regularization or pseudo-labeling, have demonstrated significant achievements. However, they still face challenges, particularly in accurately selecting sufficient high-quality pseudo-labels due to their reliance on fixed thresholds. Recent methods such as FlexMatch and FreeMatch have introduced flexible or self-adaptive thresholding techniques, greatly advancing SSL research. Nonetheless, their process of updating thresholds at each iteration is deemed time-consuming, computationally intensive, and potentially unnecessary. To address these issues, we propose Self-training with Self-adaptive Thresholding (SST), a novel, effective, and efficient SSL framework. SST introduces an innovative Self-Adaptive Thresholding (SAT) mechanism that adaptively adjusts class-specific thresholds based on the model's learning progress. SAT ensures the selection of high-quality pseudo-labeled data, mitigating the risks of inaccurate pseudo-labels and confirmation bias. Extensive experiments demonstrate that SST achieves state-of-the-art performance with remarkable efficiency, generalization, and scalability across various architectures and datasets. Semi-SST-ViT-Huge achieves the best results on competitive ImageNet-1K SSL benchmarks, with 80.7% / 84.9% Top-1 accuracy using only 1% / 10% labeled data. Compared to the fully-supervised DeiT-III-ViT-Huge, which achieves 84.8% Top-1 accuracy using 100% labeled data, our method demonstrates superior performance using only 10% labeled data.
>
---
#### [new 210] PromptVFX: Text-Driven Fields for Open-World 3D Gaussian Animation
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D动画生成任务，旨在解决传统3D特效制作复杂、耗时且需专业技能的问题。作者提出PromptVFX，通过文本驱动的4D流场控制3D高斯分布，实现快速、实时的体积效果动画生成，无需复杂建模或仿真，提升创作效率与可访问性。**

- **链接: [http://arxiv.org/pdf/2506.01091v1](http://arxiv.org/pdf/2506.01091v1)**

> **作者:** Mert Kiray; Paul Uhlenbruck; Nassir Navab; Benjamin Busam
>
> **摘要:** Visual effects (VFX) are key to immersion in modern films, games, and AR/VR. Creating 3D effects requires specialized expertise and training in 3D animation software and can be time consuming. Generative solutions typically rely on computationally intense methods such as diffusion models which can be slow at 4D inference. We reformulate 3D animation as a field prediction task and introduce a text-driven framework that infers a time-varying 4D flow field acting on 3D Gaussians. By leveraging large language models (LLMs) and vision-language models (VLMs) for function generation, our approach interprets arbitrary prompts (e.g., "make the vase glow orange, then explode") and instantly updates color, opacity, and positions of 3D Gaussians in real time. This design avoids overheads such as mesh extraction, manual or physics-based simulations and allows both novice and expert users to animate volumetric scenes with minimal effort on a consumer device even in a web browser. Experimental results show that simple textual instructions suffice to generate compelling time-varying VFX, reducing the manual effort typically required for rigging or advanced modeling. We thus present a fast and accessible pathway to language-driven 3D content creation that can pave the way to democratize VFX further.
>
---
#### [new 211] Flashbacks to Harmonize Stability and Plasticity in Continual Learning
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文属于持续学习任务，旨在解决模型在学习新知识时遗忘旧知识的稳定性-可塑性失衡问题。作者提出了Flashback Learning方法，通过双向正则化机制，在不同知识库间平衡新旧知识的学习与保留。实验表明该方法提升了分类准确率，并优化了稳定性与可塑性的比例。**

- **链接: [http://arxiv.org/pdf/2506.00477v1](http://arxiv.org/pdf/2506.00477v1)**

> **作者:** Leila Mahmoodi; Peyman Moghadam; Munawar Hayat; Christian Simon; Mehrtash Harandi
>
> **备注:** Manuscript submitted to Neural Networks (Elsevier) in August 2024; and accepted in May 2025 for publication. This version is author-accepted manuscript before copyediting and typesetting. The codes of this article will be available at https://github.com/csiro-robotics/Flashback-Learning
>
> **摘要:** We introduce Flashback Learning (FL), a novel method designed to harmonize the stability and plasticity of models in Continual Learning (CL). Unlike prior approaches that primarily focus on regularizing model updates to preserve old information while learning new concepts, FL explicitly balances this trade-off through a bidirectional form of regularization. This approach effectively guides the model to swiftly incorporate new knowledge while actively retaining its old knowledge. FL operates through a two-phase training process and can be seamlessly integrated into various CL methods, including replay, parameter regularization, distillation, and dynamic architecture techniques. In designing FL, we use two distinct knowledge bases: one to enhance plasticity and another to improve stability. FL ensures a more balanced model by utilizing both knowledge bases to regularize model updates. Theoretically, we analyze how the FL mechanism enhances the stability-plasticity balance. Empirically, FL demonstrates tangible improvements over baseline methods within the same training budget. By integrating FL into at least one representative baseline from each CL category, we observed an average accuracy improvement of up to 4.91% in Class-Incremental and 3.51% in Task-Incremental settings on standard image classification benchmarks. Additionally, measurements of the stability-to-plasticity ratio confirm that FL effectively enhances this balance. FL also outperforms state-of-the-art CL methods on more challenging datasets like ImageNet.
>
---
#### [new 212] Vid2Coach: Transforming How-To Videos into Task Assistants
- **分类: cs.HC; cs.CV**

- **简介: 该论文提出了Vid2Coach系统，属于视觉辅助任务。旨在帮助视障人士通过智能眼镜跟随教学视频完成任务。系统将视频转化为可听指令，并结合非视觉技巧与实时反馈。实验表明其能显著减少错误并提升用户体验。**

- **链接: [http://arxiv.org/pdf/2506.00717v1](http://arxiv.org/pdf/2506.00717v1)**

> **作者:** Mina Huh; Zihui Xue; Ujjaini Das; Kumar Ashutosh; Kristen Grauman; Amy Pavel
>
> **摘要:** People use videos to learn new recipes, exercises, and crafts. Such videos remain difficult for blind and low vision (BLV) people to follow as they rely on visual comparison. Our observations of visual rehabilitation therapists (VRTs) guiding BLV people to follow how-to videos revealed that VRTs provide both proactive and responsive support including detailed descriptions, non-visual workarounds, and progress feedback. We propose Vid2Coach, a system that transforms how-to videos into wearable camera-based assistants that provide accessible instructions and mixed-initiative feedback. From the video, Vid2Coach generates accessible instructions by augmenting narrated instructions with demonstration details and completion criteria for each step. It then uses retrieval-augmented-generation to extract relevant non-visual workarounds from BLV-specific resources. Vid2Coach then monitors user progress with a camera embedded in commercial smart glasses to provide context-aware instructions, proactive feedback, and answers to user questions. BLV participants (N=8) using Vid2Coach completed cooking tasks with 58.5\% fewer errors than when using their typical workflow and wanted to use Vid2Coach in their daily lives. Vid2Coach demonstrates an opportunity for AI visual assistance that strengthens rather than replaces non-visual expertise.
>
---
#### [new 213] AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning
- **分类: cs.AI; cs.CL; cs.CV; cs.HC; I.2.8; I.2.7; I.2.10; H.5.2**

- **简介: 该论文属于移动GUI智能代理任务，旨在解决现有模型在多语言界面理解、泛化能力与低延迟执行上的不足。作者提出了AgentCPM-GUI，通过预训练、高质量数据微调及强化学习优化，实现高效精准的移动端GUI交互，并发布多语言基准CAGUI。**

- **链接: [http://arxiv.org/pdf/2506.01391v1](http://arxiv.org/pdf/2506.01391v1)**

> **作者:** Zhong Zhang; Yaxi Lu; Yikun Fu; Yupeng Huo; Shenzhi Yang; Yesai Wu; Han Si; Xin Cong; Haotian Chen; Yankai Lin; Jie Xie; Wei Zhou; Wang Xu; Yuanheng Zhang; Zhou Su; Zhongwu Zhai; Xiaoming Liu; Yudong Mei; Jianming Xu; Hongyan Tian; Chongyi Wang; Chi Chen; Yuan Yao; Zhiyuan Liu; Maosong Sun
>
> **备注:** The project is available at https://github.com/OpenBMB/AgentCPM-GUI
>
> **摘要:** The recent progress of large language model agents has opened new possibilities for automating tasks through graphical user interfaces (GUIs), especially in mobile environments where intelligent interaction can greatly enhance usability. However, practical deployment of such agents remains constrained by several key challenges. Existing training data is often noisy and lack semantic diversity, which hinders the learning of precise grounding and planning. Models trained purely by imitation tend to overfit to seen interface patterns and fail to generalize in unfamiliar scenarios. Moreover, most prior work focuses on English interfaces while overlooks the growing diversity of non-English applications such as those in the Chinese mobile ecosystem. In this work, we present AgentCPM-GUI, an 8B-parameter GUI agent built for robust and efficient on-device GUI interaction. Our training pipeline includes grounding-aware pre-training to enhance perception, supervised fine-tuning on high-quality Chinese and English trajectories to imitate human-like actions, and reinforcement fine-tuning with GRPO to improve reasoning capability. We also introduce a compact action space that reduces output length and supports low-latency execution on mobile devices. AgentCPM-GUI achieves state-of-the-art performance on five public benchmarks and a new Chinese GUI benchmark called CAGUI, reaching $96.9\%$ Type-Match and $91.3\%$ Exact-Match. To facilitate reproducibility and further research, we publicly release all code, model checkpoint, and evaluation data.
>
---
#### [new 214] Image Restoration Learning via Noisy Supervision in the Fourier Domain
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决使用含噪监督信号训练模型时效果不佳的问题，尤其针对空间相关噪声和像素损失局限性。工作提出在傅里叶域中进行监督学习，利用其系数的稀疏性和全局信息特性，建立统一的学习框架。实验表明该方法在多种任务和网络结构中均表现优异。**

- **链接: [http://arxiv.org/pdf/2506.00564v1](http://arxiv.org/pdf/2506.00564v1)**

> **作者:** Haosen Liu; Jiahao Liu; Shan Tan; Edmund Y. Lam
>
> **摘要:** Noisy supervision refers to supervising image restoration learning with noisy targets. It can alleviate the data collection burden and enhance the practical applicability of deep learning techniques. However, existing methods suffer from two key drawbacks. Firstly, they are ineffective in handling spatially correlated noise commonly observed in practical applications such as low-light imaging and remote sensing. Secondly, they rely on pixel-wise loss functions that only provide limited supervision information. This work addresses these challenges by leveraging the Fourier domain. We highlight that the Fourier coefficients of spatially correlated noise exhibit sparsity and independence, making them easier to handle. Additionally, Fourier coefficients contain global information, enabling more significant supervision. Motivated by these insights, we propose to establish noisy supervision in the Fourier domain. We first prove that Fourier coefficients of a wide range of noise converge in distribution to the Gaussian distribution. Exploiting this statistical property, we establish the equivalence between using noisy targets and clean targets in the Fourier domain. This leads to a unified learning framework applicable to various image restoration tasks, diverse network architectures, and different noise models. Extensive experiments validate the outstanding performance of this framework in terms of both quantitative indices and perceptual quality.
>
---
#### [new 215] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态医学推理任务，旨在解决单智能体模型在多样医疗场景中泛化能力不足的问题。作者提出MMedAgent-RL框架，通过强化学习实现动态协作机制，包含分诊医生和主治医生智能体，并引入课程学习策略提升决策一致性，最终在多个医学VQA基准上取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2506.00555v1](http://arxiv.org/pdf/2506.00555v1)**

> **作者:** Peng Xia; Jinglu Wang; Yibo Peng; Kaide Zeng; Xian Wu; Xiangru Tang; Hongtu Zhu; Yun Li; Shujie Liu; Yan Lu; Huaxiu Yao
>
> **摘要:** Medical Large Vision-Language Models (Med-LVLMs) have shown strong potential in multimodal diagnostic tasks. However, existing single-agent models struggle to generalize across diverse medical specialties, limiting their performance. Recent efforts introduce multi-agent collaboration frameworks inspired by clinical workflows, where general practitioners (GPs) and specialists interact in a fixed sequence. Despite improvements, these static pipelines lack flexibility and adaptability in reasoning. To address this, we propose MMedAgent-RL, a reinforcement learning (RL)-based multi-agent framework that enables dynamic, optimized collaboration among medical agents. Specifically, we train two GP agents based on Qwen2.5-VL via RL: the triage doctor learns to assign patients to appropriate specialties, while the attending physician integrates the judgments from multi-specialists and its own knowledge to make final decisions. To address the inconsistency in specialist outputs, we introduce a curriculum learning (CL)-guided RL strategy that progressively teaches the attending physician to balance between imitating specialists and correcting their mistakes. Experiments on five medical VQA benchmarks demonstrate that MMedAgent-RL not only outperforms both open-source and proprietary Med-LVLMs, but also exhibits human-like reasoning patterns. Notably, it achieves an average performance gain of 18.4% over supervised fine-tuning baselines.
>
---
#### [new 216] Image Generation from Contextually-Contradictory Prompts
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型在处理语义矛盾提示时生成不准确的问题。作者提出了一种分阶段的提示分解框架，利用大语言模型分析并重构提示，生成语义连贯的代理提示，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.01929v1](http://arxiv.org/pdf/2506.01929v1)**

> **作者:** Saar Huberman; Or Patashnik; Omer Dahary; Ron Mokady; Daniel Cohen-Or
>
> **备注:** Project page: https://tdpc2025.github.io/SAP/
>
> **摘要:** Text-to-image diffusion models excel at generating high-quality, diverse images from natural language prompts. However, they often fail to produce semantically accurate results when the prompt contains concept combinations that contradict their learned priors. We define this failure mode as contextual contradiction, where one concept implicitly negates another due to entangled associations learned during training. To address this, we propose a stage-aware prompt decomposition framework that guides the denoising process using a sequence of proxy prompts. Each proxy prompt is constructed to match the semantic content expected to emerge at a specific stage of denoising, while ensuring contextual coherence. To construct these proxy prompts, we leverage a large language model (LLM) to analyze the target prompt, identify contradictions, and generate alternative expressions that preserve the original intent while resolving contextual conflicts. By aligning prompt information with the denoising progression, our method enables fine-grained semantic control and accurate image generation in the presence of contextual contradictions. Experiments across a variety of challenging prompts show substantial improvements in alignment to the textual prompt.
>
---
#### [new 217] 3D Gaussian Splat Vulnerabilities
- **分类: cs.CR; cs.CV; cs.LG**

- **简介: 论文研究3D高斯点渲染（3DGS）在安全关键应用中的漏洞，提出两种攻击方法：CLOAK利用视角相关外观嵌入隐蔽内容，DAGGER通过扰动3D高斯欺骗目标检测模型。任务为安全性分析，问题为对抗攻击威胁，工作为设计无需训练数据的定向攻击策略，揭示3DGS在自主导航等应用中的潜在风险。**

- **链接: [http://arxiv.org/pdf/2506.00280v1](http://arxiv.org/pdf/2506.00280v1)**

> **作者:** Matthew Hull; Haoyang Yang; Pratham Mehta; Mansi Phute; Aeree Cho; Haoran Wang; Matthew Lau; Wenke Lee; Willian T. Lunardi; Martin Andreoni; Polo Chau
>
> **备注:** 4 pages, 4 figures, CVPR '25 Workshop on Neural Fields Beyond Conventional Cameras
>
> **摘要:** With 3D Gaussian Splatting (3DGS) being increasingly used in safety-critical applications, how can an adversary manipulate the scene to cause harm? We introduce CLOAK, the first attack that leverages view-dependent Gaussian appearances - colors and textures that change with viewing angle - to embed adversarial content visible only from specific viewpoints. We further demonstrate DAGGER, a targeted adversarial attack directly perturbing 3D Gaussians without access to underlying training data, deceiving multi-stage object detectors e.g., Faster R-CNN, through established methods such as projected gradient descent. These attacks highlight underexplored vulnerabilities in 3DGS, introducing a new potential threat to robotic learning for autonomous navigation and other safety-critical 3DGS applications.
>
---
#### [new 218] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态对话系统任务，旨在解决现有语言模型缺乏非语言交流能力的问题。作者构建了包含视频、文本及非语言信号的大型数据集Venus，并基于此训练多模态模型Mars，实现语言与非语言信息的理解与生成。**

- **链接: [http://arxiv.org/pdf/2506.00958v1](http://arxiv.org/pdf/2506.00958v1)**

> **作者:** Youngmin Kim; Jiwan Chung; Jisoo Kim; Sunghyun Lee; Sangkyu Lee; Junhyeok Kim; Cheoljong Yang; Youngjae Yu
>
> **备注:** Accepted to ACL 2025 (Main), Our code and dataset: https://github.com/winston1214/nonverbal-conversation
>
> **摘要:** Nonverbal communication is integral to human interaction, with gestures, facial expressions, and body language conveying critical aspects of intent and emotion. However, existing large language models (LLMs) fail to effectively incorporate these nonverbal elements, limiting their capacity to create fully immersive conversational experiences. We introduce MARS, a multimodal language model designed to understand and generate nonverbal cues alongside text, bridging this gap in conversational AI. Our key innovation is VENUS, a large-scale dataset comprising annotated videos with time-aligned text, facial expressions, and body language. Leveraging VENUS, we train MARS with a next-token prediction objective, combining text with vector-quantized nonverbal representations to achieve multimodal understanding and generation within a unified framework. Based on various analyses of the VENUS datasets, we validate its substantial scale and high effectiveness. Our quantitative and qualitative results demonstrate that MARS successfully generates text and nonverbal languages, corresponding to conversational input.
>
---
#### [new 219] GeoChain: Multimodal Chain-of-Thought for Geographic Reasoning
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于地理推理任务，旨在解决多模态大语言模型在地理定位与推理中的不足。作者构建了GeoChain基准，包含百万级街景图像及对应的21步推理问题链，并标注多种地理属性。通过测试现有模型，发现其在视觉基础、逻辑推理和精确定位上存在挑战，从而推动复杂地理推理技术的发展。**

- **链接: [http://arxiv.org/pdf/2506.00785v1](http://arxiv.org/pdf/2506.00785v1)**

> **作者:** Sahiti Yerramilli; Nilay Pande; Rynaa Grover; Jayant Sravan Tamarapalli
>
> **摘要:** This paper introduces GeoChain, a large-scale benchmark for evaluating step-by-step geographic reasoning in multimodal large language models (MLLMs). Leveraging 1.46 million Mapillary street-level images, GeoChain pairs each image with a 21-step chain-of-thought (CoT) question sequence (over 30 million Q&A pairs). These sequences guide models from coarse attributes to fine-grained localization across four reasoning categories - visual, spatial, cultural, and precise geolocation - annotated by difficulty. Images are also enriched with semantic segmentation (150 classes) and a visual locatability score. Our benchmarking of contemporary MLLMs (GPT-4.1 variants, Claude 3.7, Gemini 2.5 variants) on a diverse 2,088-image subset reveals consistent challenges: models frequently exhibit weaknesses in visual grounding, display erratic reasoning, and struggle to achieve accurate localization, especially as the reasoning complexity escalates. GeoChain offers a robust diagnostic methodology, critical for fostering significant advancements in complex geographic reasoning within MLLMs.
>
---
#### [new 220] SEMNAV: A Semantic Segmentation-Driven Approach to Visual Semantic Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语义导航（VSN）任务，旨在解决智能体在未知环境中基于视觉信息导航至目标物体的问题。现有方法依赖模拟环境和RGB数据，泛化能力受限。作者提出SEMNAV，利用语义分割作为输入表示，提升感知与决策能力，并构建了专用数据集。实验表明其在模拟和真实环境均优于现有方法，有效缩小了仿真到现实的差距。**

- **链接: [http://arxiv.org/pdf/2506.01418v1](http://arxiv.org/pdf/2506.01418v1)**

> **作者:** Rafael Flor-Rodríguez; Carlos Gutiérrez-Álvarez; Francisco Javier Acevedo-Rodríguez; Sergio Lafuente-Arroyo; Roberto J. López-Sastre
>
> **摘要:** Visual Semantic Navigation (VSN) is a fundamental problem in robotics, where an agent must navigate toward a target object in an unknown environment, mainly using visual information. Most state-of-the-art VSN models are trained in simulation environments, where rendered scenes of the real world are used, at best. These approaches typically rely on raw RGB data from the virtual scenes, which limits their ability to generalize to real-world environments due to domain adaptation issues. To tackle this problem, in this work, we propose SEMNAV, a novel approach that leverages semantic segmentation as the main visual input representation of the environment to enhance the agent's perception and decision-making capabilities. By explicitly incorporating high-level semantic information, our model learns robust navigation policies that improve generalization across unseen environments, both in simulated and real world settings. We also introduce a newly curated dataset, i.e. the SEMNAV dataset, designed for training semantic segmentation-aware navigation models like SEMNAV. Our approach is evaluated extensively in both simulated environments and with real-world robotic platforms. Experimental results demonstrate that SEMNAV outperforms existing state-of-the-art VSN models, achieving higher success rates in the Habitat 2.0 simulation environment, using the HM3D dataset. Furthermore, our real-world experiments highlight the effectiveness of semantic segmentation in mitigating the sim-to-real gap, making our model a promising solution for practical VSN-based robotic applications. We release SEMNAV dataset, code and trained models at https://github.com/gramuah/semnav
>
---
#### [new 221] EgoBrain: Synergizing Minds and Eyes For Human Action Understanding
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于多模态人机交互任务，旨在解决人类行为理解问题。作者构建了首个大规模、时间对齐的脑电与第一视角视觉同步数据集EgoBrain，并开发了融合EEG与视觉的多模态学习框架，实现了跨被试与环境的动作识别，准确率达66.70%。**

- **链接: [http://arxiv.org/pdf/2506.01353v1](http://arxiv.org/pdf/2506.01353v1)**

> **作者:** Nie Lin; Yansen Wang; Dongqi Han; Weibang Jiang; Jingyuan Li; Ryosuke Furuta; Yoichi Sato; Dongsheng Li
>
> **备注:** 21 pages, 12 figures
>
> **摘要:** The integration of brain-computer interfaces (BCIs), in particular electroencephalography (EEG), with artificial intelligence (AI) has shown tremendous promise in decoding human cognition and behavior from neural signals. In particular, the rise of multimodal AI models have brought new possibilities that have never been imagined before. Here, we present EgoBrain --the world's first large-scale, temporally aligned multimodal dataset that synchronizes egocentric vision and EEG of human brain over extended periods of time, establishing a new paradigm for human-centered behavior analysis. This dataset comprises 61 hours of synchronized 32-channel EEG recordings and first-person video from 40 participants engaged in 29 categories of daily activities. We then developed a muiltimodal learning framework to fuse EEG and vision for action understanding, validated across both cross-subject and cross-environment challenges, achieving an action recognition accuracy of 66.70%. EgoBrain paves the way for a unified framework for brain-computer interface with multiple modalities. All data, tools, and acquisition protocols are openly shared to foster open science in cognitive computing.
>
---
#### [new 222] Datasheets Aren't Enough: DataRubrics for Automated Quality Metrics and Accountability
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; eess.AS**

- **简介: 该论文属于数据质量评估任务，旨在解决当前机器学习数据集质量不足、缺乏标准化评价方法的问题。作者提出DataRubrics框架，结合LLM评估技术，提供可复现、可扩展的数据质量评分系统，并推动数据提交流程的透明与问责。**

- **链接: [http://arxiv.org/pdf/2506.01789v1](http://arxiv.org/pdf/2506.01789v1)**

> **作者:** Genta Indra Winata; David Anugraha; Emmy Liu; Alham Fikri Aji; Shou-Yi Hung; Aditya Parashar; Patrick Amadeus Irawan; Ruochen Zhang; Zheng-Xin Yong; Jan Christian Blaise Cruz; Niklas Muennighoff; Seungone Kim; Hanyang Zhao; Sudipta Kar; Kezia Erina Suryoraharjo; M. Farid Adilazuarda; En-Shiun Annie Lee; Ayu Purwarianti; Derry Tanti Wijaya; Monojit Choudhury
>
> **备注:** Preprint
>
> **摘要:** High-quality datasets are fundamental to training and evaluating machine learning models, yet their creation-especially with accurate human annotations-remains a significant challenge. Many dataset paper submissions lack originality, diversity, or rigorous quality control, and these shortcomings are often overlooked during peer review. Submissions also frequently omit essential details about dataset construction and properties. While existing tools such as datasheets aim to promote transparency, they are largely descriptive and do not provide standardized, measurable methods for evaluating data quality. Similarly, metadata requirements at conferences promote accountability but are inconsistently enforced. To address these limitations, this position paper advocates for the integration of systematic, rubric-based evaluation metrics into the dataset review process-particularly as submission volumes continue to grow. We also explore scalable, cost-effective methods for synthetic data generation, including dedicated tools and LLM-as-a-judge approaches, to support more efficient evaluation. As a call to action, we introduce DataRubrics, a structured framework for assessing the quality of both human- and model-generated datasets. Leveraging recent advances in LLM-based evaluation, DataRubrics offers a reproducible, scalable, and actionable solution for dataset quality assessment, enabling both authors and reviewers to uphold higher standards in data-centric research. We also release code to support reproducibility of LLM-based evaluations at https://github.com/datarubrics/datarubrics.
>
---
#### [new 223] Dynamic Domain Adaptation-Driven Physics-Informed Graph Representation Learning for AC-OPF
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于电力系统优化任务，旨在解决AC-OPF问题中变量分布与最优解关系建模困难、约束建模不足及缺乏时空信息整合的问题。作者提出DDA-PIGCN方法，结合物理引导与动态域适应，提升约束一致性与时空特征融合，实现高效可靠的AC-OPF求解。**

- **链接: [http://arxiv.org/pdf/2506.00478v1](http://arxiv.org/pdf/2506.00478v1)**

> **作者:** Hongjie Zhu; Zezheng Zhang; Zeyu Zhang; Yu Bai; Shimin Wen; Huazhang Wang; Daji Ergu; Ying Cai; Yang Zhao
>
> **摘要:** Alternating Current Optimal Power Flow (AC-OPF) aims to optimize generator power outputs by utilizing the non-linear relationships between voltage magnitudes and phase angles in a power system. However, current AC-OPF solvers struggle to effectively represent the complex relationship between variable distributions in the constraint space and their corresponding optimal solutions. This limitation in constraint modeling restricts the system's ability to develop diverse knowledge representations. Additionally, modeling the power grid solely based on spatial topology further limits the integration of additional prior knowledge, such as temporal information. To overcome these challenges, we propose DDA-PIGCN (Dynamic Domain Adaptation-Driven Physics-Informed Graph Convolutional Network), a new method designed to address constraint-related issues and build a graph-based learning framework that incorporates spatiotemporal features. DDA-PIGCN improves consistency optimization for features with varying long-range dependencies by applying multi-layer, hard physics-informed constraints. It also uses a dynamic domain adaptation learning mechanism that iteratively updates and refines key state variables under predefined constraints, enabling precise constraint verification. Moreover, it captures spatiotemporal dependencies between generators and loads by leveraging the physical structure of the power grid, allowing for deep integration of topological information across time and space. Extensive comparative and ablation studies show that DDA-PIGCN delivers strong performance across several IEEE standard test cases (such as case9, case30, and case300), achieving mean absolute errors (MAE) from 0.0011 to 0.0624 and constraint satisfaction rates between 99.6% and 100%, establishing it as a reliable and efficient AC-OPF solver.
>
---
#### [new 224] Sparse Imagination for Efficient Visual World Model Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉世界模型规划任务，旨在解决计算资源受限下的高效决策问题。作者提出“稀疏想象”方法，通过减少前向预测中的处理标记数量，提升推理效率。方法基于稀疏训练的视觉世界模型与随机分组注意力策略，实现高效且自适应的规划，在保持控制精度的同时显著降低计算开销。**

- **链接: [http://arxiv.org/pdf/2506.01392v1](http://arxiv.org/pdf/2506.01392v1)**

> **作者:** Junha Chun; Youngjoon Jeong; Taesup Kim
>
> **摘要:** World model based planning has significantly improved decision-making in complex environments by enabling agents to simulate future states and make informed choices. However, ensuring the prediction accuracy of world models often demands substantial computational resources, posing a major challenge for real-time applications. This computational burden is particularly restrictive in robotics, where resources are severely constrained. To address this limitation, we propose a Sparse Imagination for Efficient Visual World Model Planning, which enhances computational efficiency by reducing the number of tokens processed during forward prediction. Our method leverages a sparsely trained vision-based world model based on transformers with randomized grouped attention strategy, allowing the model to adaptively adjust the number of tokens processed based on the computational resource. By enabling sparse imagination (rollout), our approach significantly accelerates planning while maintaining high control fidelity. Experimental results demonstrate that sparse imagination preserves task performance while dramatically improving inference efficiency, paving the way for the deployment of world models in real-time decision-making scenarios.
>
---
## 更新

#### [replaced 001] Beyond Face Swapping: A Diffusion-Based Digital Human Benchmark for Multimodal Deepfake Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16512v3](http://arxiv.org/pdf/2505.16512v3)**

> **作者:** Jiaxin Liu; Jia Wang; Saihui Hou; Min Ren; Huijia Wu; Zhaofeng He
>
> **摘要:** In recent years, the explosive advancement of deepfake technology has posed a critical and escalating threat to public security: diffusion-based digital human generation. Unlike traditional face manipulation methods, such models can generate highly realistic videos with consistency via multimodal control signals. Their flexibility and covertness pose severe challenges to existing detection strategies. To bridge this gap, we introduce DigiFakeAV, the new large-scale multimodal digital human forgery dataset based on diffusion models. Leveraging five of the latest digital human generation methods and a voice cloning method, we systematically construct a dataset comprising 60,000 videos (8.4 million frames), covering multiple nationalities, skin tones, genders, and real-world scenarios, significantly enhancing data diversity and realism. User studies demonstrate that the misrecognition rate by participants for DigiFakeAV reaches as high as 68%. Moreover, the substantial performance degradation of existing detection models on our dataset further highlights its challenges. To address this problem, we propose DigiShield, an effective detection baseline based on spatiotemporal and cross-modal fusion. By jointly modeling the 3D spatiotemporal features of videos and the semantic-acoustic features of audio, DigiShield achieves state-of-the-art (SOTA) performance on the DigiFakeAV and shows strong generalization on other datasets.
>
---
#### [replaced 002] Leveraging Complementary Attention maps in vision transformers for OCT image analysis
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2310.14005v3](http://arxiv.org/pdf/2310.14005v3)**

> **作者:** Haz Sameen Shahgir; Tanjeem Azwad Zaman; Khondker Salman Sayeed; Md. Asif Haider; Sheikh Saifur Rahman Jony; M. Sohel Rahman
>
> **备注:** Accepted in 2025 IEEE International Conference on Image Processing
>
> **摘要:** Optical Coherence Tomography (OCT) scan yields all possible cross-section images of a retina for detecting biomarkers linked to optical defects. Due to the high volume of data generated, an automated and reliable biomarker detection pipeline is necessary as a primary screening stage. We outline our new state-of-the-art pipeline for identifying biomarkers from OCT scans. In collaboration with trained ophthalmologists, we identify local and global structures in biomarkers. Through a comprehensive and systematic review of existing vision architectures, we evaluate different convolution and attention mechanisms for biomarker detection. We find that MaxViT, a hybrid vision transformer combining convolution layers with strided attention, is better suited for local feature detection, while EVA-02, a standard vision transformer leveraging pure attention and large-scale knowledge distillation, excels at capturing global features. We ensemble the predictions of both models to achieve first place in the IEEE Video and Image Processing Cup 2023 competition on OCT biomarker detection, achieving a patient-wise F1 score of 0.8527 in the final phase of the competition, scoring 3.8\% higher than the next best solution. Finally, we used knowledge distillation to train a single MaxViT to outperform our ensemble at a fraction of the computation cost.
>
---
#### [replaced 003] DiffVLA: Vision-Language Guided Diffusion Planning for Autonomous Driving
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.19381v3](http://arxiv.org/pdf/2505.19381v3)**

> **作者:** Anqing Jiang; Yu Gao; Zhigang Sun; Yiru Wang; Jijun Wang; Jinghao Chai; Qian Cao; Yuweng Heng; Hao Jiang; Zongzheng Zhang; Xianda Guo; Hao Sun; Hao Zhao
>
> **备注:** 4pages
>
> **摘要:** Research interest in end-to-end autonomous driving has surged owing to its fully differentiable design integrating modular tasks, i.e. perception, prediction and planing, which enables optimization in pursuit of the ultimate goal. Despite the great potential of the end-to-end paradigm, existing methods suffer from several aspects including expensive BEV (bird's eye view) computation, action diversity, and sub-optimal decision in complex real-world scenarios. To address these challenges, we propose a novel hybrid sparse-dense diffusion policy, empowered by a Vision-Language Model (VLM), called Diff-VLA. We explore the sparse diffusion representation for efficient multi-modal driving behavior. Moreover, we rethink the effectiveness of VLM driving decision and improve the trajectory generation guidance through deep interaction across agent, map instances and VLM output. Our method shows superior performance in Autonomous Grand Challenge 2025 which contains challenging real and reactive synthetic scenarios. Our methods achieves 45.0 PDMS.
>
---
#### [replaced 004] Efficiency Bottlenecks of Convolutional Kolmogorov-Arnold Networks: A Comprehensive Scrutiny with ImageNet, AlexNet, LeNet and Tabular Classification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.15757v3](http://arxiv.org/pdf/2501.15757v3)**

> **作者:** Ashim Dahal; Saydul Akbar Murad; Nick Rahimi
>
> **摘要:** Algorithmic level developments like Convolutional Neural Networks, transformers, attention mechanism, Retrieval Augmented Generation and so on have changed Artificial Intelligence. Recent such development was observed by Kolmogorov-Arnold Networks that suggested to challenge the fundamental concept of a Neural Network, thus change Multilayer Perceptron, and Convolutional Neural Networks. They received a good reception in terms of scientific modeling, yet had some drawbacks in terms of efficiency. In this paper, we train Convolutional Kolmogorov Arnold Networks (CKANs) with the ImageNet-1k dataset with 1.3 million images, MNIST dataset with 60k images and a tabular biological science related MoA dataset and test the promise of CKANs in terms of FLOPS, Inference Time, number of trainable parameters and training time against the accuracy, precision, recall and f-1 score they produce against the standard industry practice on CNN models. We show that the CKANs perform fair yet slower than CNNs in small size dataset like MoA and MNIST but are not nearly comparable as the dataset gets larger and more complex like the ImageNet. The code implementation of this paper can be found on the link: https://github.com/ashimdahal/Study-of-Convolutional-Kolmogorov-Arnold-networks
>
---
#### [replaced 005] Enhancing Sample Generation of Diffusion Models using Noise Level Correction
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2412.05488v3](http://arxiv.org/pdf/2412.05488v3)**

> **作者:** Abulikemu Abuduweili; Chenyang Yuan; Changliu Liu; Frank Permenter
>
> **摘要:** The denoising process of diffusion models can be interpreted as an approximate projection of noisy samples onto the data manifold. Moreover, the noise level in these samples approximates their distance to the underlying manifold. Building on this insight, we propose a novel method to enhance sample generation by aligning the estimated noise level with the true distance of noisy samples to the manifold. Specifically, we introduce a noise level correction network, leveraging a pre-trained denoising network, to refine noise level estimates during the denoising process. Additionally, we extend this approach to various image restoration tasks by integrating task-specific constraints, including inpainting, deblurring, super-resolution, colorization, and compressed sensing. Experimental results demonstrate that our method significantly improves sample quality in both unconstrained and constrained generation scenarios. Notably, the proposed noise level correction framework is compatible with existing denoising schedulers (e.g., DDIM), offering additional performance improvements.
>
---
#### [replaced 006] Parameter Efficient Fine-Tuning of Segment Anything Model for Biomedical Imaging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00418v2](http://arxiv.org/pdf/2502.00418v2)**

> **作者:** Carolin Teuber; Anwai Archit; Constantin Pape
>
> **备注:** Published in MIDL 2025
>
> **摘要:** Segmentation is an important analysis task for biomedical images, enabling the study of individual organelles, cells or organs. Deep learning has massively improved segmentation methods, but challenges remain in generalization to new conditions, requiring costly data annotation. Vision foundation models, such as Segment Anything Model (SAM), address this issue through improved generalization. However, these models still require finetuning on annotated data, although with less annotations, to achieve optimal results for new conditions. As a downside, they require more computational resources. This makes parameter-efficient finetuning (PEFT) relevant. We contribute the first comprehensive study of PEFT for SAM applied to biomedical images. We find that the placement of PEFT layers is more important for efficiency than the type of layer for vision transformers and we provide a recipe for resource-efficient finetuning. Our code is publicly available at https://github.com/computational-cell-analytics/peft-sam.
>
---
#### [replaced 007] Mixed-View Panorama Synthesis using Geospatially Guided Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.09672v2](http://arxiv.org/pdf/2407.09672v2)**

> **作者:** Zhexiao Xiong; Xin Xing; Scott Workman; Subash Khanal; Nathan Jacobs
>
> **备注:** Accepted by Transactions on Machine Learning Research (TMLR) Project page: https://mixed-view.github.io
>
> **摘要:** We introduce the task of mixed-view panorama synthesis, where the goal is to synthesize a novel panorama given a small set of input panoramas and a satellite image of the area. This contrasts with previous work which only uses input panoramas (same-view synthesis), or an input satellite image (cross-view synthesis). We argue that the mixed-view setting is the most natural to support panorama synthesis for arbitrary locations worldwide. A critical challenge is that the spatial coverage of panoramas is uneven, with few panoramas available in many regions of the world. We introduce an approach that utilizes diffusion-based modeling and an attention-based architecture for extracting information from all available input imagery. Experimental results demonstrate the effectiveness of our proposed method. In particular, our model can handle scenarios when the available panoramas are sparse or far from the location of the panorama we are attempting to synthesize. The project page is available at https://mixed-view.github.io
>
---
#### [replaced 008] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2502.11196v2](http://arxiv.org/pdf/2502.11196v2)**

> **作者:** Yixin Ou; Yunzhi Yao; Ningyu Zhang; Hui Jin; Jiacheng Sun; Shumin Deng; Zhenguo Li; Huajun Chen
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Despite exceptional capabilities in knowledge-intensive tasks, Large Language Models (LLMs) face a critical gap in understanding how they internalize new knowledge, particularly how to structurally embed acquired knowledge in their neural computations. We address this issue through the lens of knowledge circuit evolution, identifying computational subgraphs that facilitate knowledge storage and processing. Our systematic analysis of circuit evolution throughout continual pre-training reveals several key findings: (1) the acquisition of new knowledge is influenced by its relevance to pre-existing knowledge; (2) the evolution of knowledge circuits exhibits a distinct phase shift from formation to optimization; (3) the evolution of knowledge circuits follows a deep-to-shallow pattern. These insights not only advance our theoretical understanding of the mechanisms of new knowledge acquisition in LLMs, but also provide potential implications for improving continual pre-training strategies to enhance model performance. Code and data will be available at https://github.com/zjunlp/DynamicKnowledgeCircuits.
>
---
#### [replaced 009] MoviePuzzle: Visual Narrative Reasoning through Multimodal Order Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2306.02252v3](http://arxiv.org/pdf/2306.02252v3)**

> **作者:** Jianghui Wang; Yuxuan Wang; Dongyan Zhao; Zilong Zheng
>
> **摘要:** We introduce MoviePuzzle, a novel challenge that targets visual narrative reasoning and holistic movie understanding. Despite the notable progress that has been witnessed in the realm of video understanding, most prior works fail to present tasks and models to address holistic video understanding and the innate visual narrative structures existing in long-form videos. To tackle this quandary, we put forth MoviePuzzle task that amplifies the temporal feature learning and structure learning of video models by reshuffling the shot, frame, and clip layers of movie segments in the presence of video-dialogue information. We start by establishing a carefully refined dataset based on MovieNet by dissecting movies into hierarchical layers and randomly permuting the orders. Besides benchmarking the MoviePuzzle with prior arts on movie understanding, we devise a Hierarchical Contrastive Movie Clustering (HCMC) model that considers the underlying structure and visual semantic orders for movie reordering. Specifically, through a pairwise and contrastive learning approach, we train models to predict the correct order of each layer. This equips them with the knack for deciphering the visual narrative structure of movies and handling the disorder lurking in video data. Experiments show that our approach outperforms existing state-of-the-art methods on the \MoviePuzzle benchmark, underscoring its efficacy.
>
---
#### [replaced 010] Enhancing Large Vision Model in Street Scene Semantic Understanding through Leveraging Posterior Optimization Trajectory
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.01710v2](http://arxiv.org/pdf/2501.01710v2)**

> **作者:** Wei-Bin Kou; Qingfeng Lin; Ming Tang; Jingreng Lei; Shuai Wang; Rongguang Ye; Guangxu Zhu; Yik-Chung Wu
>
> **备注:** 7 pages
>
> **摘要:** To improve the generalization of the autonomous driving (AD) perception model, vehicles need to update the model over time based on the continuously collected data. As time progresses, the amount of data fitted by the AD model expands, which helps to improve the AD model generalization substantially. However, such ever-expanding data is a double-edged sword for the AD model. Specifically, as the fitted data volume grows to exceed the the AD model's fitting capacities, the AD model is prone to under-fitting. To address this issue, we propose to use a pretrained Large Vision Models (LVMs) as backbone coupled with downstream perception head to understand AD semantic information. This design can not only surmount the aforementioned under-fitting problem due to LVMs' powerful fitting capabilities, but also enhance the perception generalization thanks to LVMs' vast and diverse training data. On the other hand, to mitigate vehicles' computational burden of training the perception head while running LVM backbone, we introduce a Posterior Optimization Trajectory (POT)-Guided optimization scheme (POTGui) to accelerate the convergence. Concretely, we propose a POT Generator (POTGen) to generate posterior (future) optimization direction in advance to guide the current optimization iteration, through which the model can generally converge within 10 epochs. Extensive experiments demonstrate that the proposed method improves the performance by over 66.48\% and converges faster over 6 times, compared to the existing state-of-the-art approach.
>
---
#### [replaced 011] LEGNet: Lightweight Edge-Gaussian Driven Network for Low-Quality Remote Sensing Image Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14012v2](http://arxiv.org/pdf/2503.14012v2)**

> **作者:** Wei Lu; Si-Bao Chen; Hui-Dong Li; Qing-Ling Shu; Chris H. Q. Ding; Jin Tang; Bin Luo
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Remote sensing object detection (RSOD) often suffers from degradations such as low spatial resolution, sensor noise, motion blur, and adverse illumination. These factors diminish feature distinctiveness, leading to ambiguous object representations and inadequate foreground-background separation. Existing RSOD methods exhibit limitations in robust detection of low-quality objects. To address these pressing challenges, we introduce LEGNet, a lightweight backbone network featuring a novel Edge-Gaussian Aggregation (EGA) module specifically engineered to enhance feature representation derived from low-quality remote sensing images. EGA module integrates: (a) orientation-aware Scharr filters to sharpen crucial edge details often lost in low-contrast or blurred objects, and (b) Gaussian-prior-based feature refinement to suppress noise and regularize ambiguous feature responses, enhancing foreground saliency under challenging conditions. EGA module alleviates prevalent problems in reduced contrast, structural discontinuities, and ambiguous feature responses prevalent in degraded images, effectively improving model robustness while maintaining computational efficiency. Comprehensive evaluations across five benchmarks (DOTA-v1.0, v1.5, DIOR-R, FAIR1M-v1.0, and VisDrone2019) demonstrate that LEGNet achieves state-of-the-art performance, particularly in detecting low-quality objects. The code is available at https://github.com/lwCVer/LEGNet.
>
---
#### [replaced 012] Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.19475v2](http://arxiv.org/pdf/2504.19475v2)**

> **作者:** Sonia Joseph; Praneet Suresh; Lorenz Hufe; Edward Stevinson; Robert Graham; Yash Vadi; Danilo Bzdok; Sebastian Lapuschkin; Lee Sharkey; Blake Aaron Richards
>
> **备注:** 4 pages, 3 figures, 9 tables. Oral and Tutorial at the CVPR Mechanistic Interpretability for Vision (MIV) Workshop
>
> **摘要:** Robust tooling and publicly available pre-trained models have helped drive recent advances in mechanistic interpretability for language models. However, similar progress in vision mechanistic interpretability has been hindered by the lack of accessible frameworks and pre-trained weights. We present Prisma (Access the codebase here: https://github.com/Prisma-Multimodal/ViT-Prisma), an open-source framework designed to accelerate vision mechanistic interpretability research, providing a unified toolkit for accessing 75+ vision and video transformers; support for sparse autoencoder (SAE), transcoder, and crosscoder training; a suite of 80+ pre-trained SAE weights; activation caching, circuit analysis tools, and visualization tools; and educational resources. Our analysis reveals surprising findings, including that effective vision SAEs can exhibit substantially lower sparsity patterns than language SAEs, and that in some instances, SAE reconstructions can decrease model loss. Prisma enables new research directions for understanding vision model internals while lowering barriers to entry in this emerging field.
>
---
#### [replaced 013] Safety at Scale: A Comprehensive Survey of Large Model Safety
- **分类: cs.CR; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.05206v4](http://arxiv.org/pdf/2502.05206v4)**

> **作者:** Xingjun Ma; Yifeng Gao; Yixu Wang; Ruofan Wang; Xin Wang; Ye Sun; Yifan Ding; Hengyuan Xu; Yunhao Chen; Yunhan Zhao; Hanxun Huang; Yige Li; Jiaming Zhang; Xiang Zheng; Yang Bai; Zuxuan Wu; Xipeng Qiu; Jingfeng Zhang; Yiming Li; Xudong Han; Haonan Li; Jun Sun; Cong Wang; Jindong Gu; Baoyuan Wu; Siheng Chen; Tianwei Zhang; Yang Liu; Mingming Gong; Tongliang Liu; Shirui Pan; Cihang Xie; Tianyu Pang; Yinpeng Dong; Ruoxi Jia; Yang Zhang; Shiqing Ma; Xiangyu Zhang; Neil Gong; Chaowei Xiao; Sarah Erfani; Tim Baldwin; Bo Li; Masashi Sugiyama; Dacheng Tao; James Bailey; Yu-Gang Jiang
>
> **备注:** 47 pages, 3 figures, 11 tables; GitHub: https://github.com/xingjunm/Awesome-Large-Model-Safety
>
> **摘要:** The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.
>
---
#### [replaced 014] ITA-MDT: Image-Timestep-Adaptive Masked Diffusion Transformer Framework for Image-Based Virtual Try-On
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20418v2](http://arxiv.org/pdf/2503.20418v2)**

> **作者:** Ji Woo Hong; Tri Ton; Trung X. Pham; Gwanhyeong Koo; Sunjae Yoon; Chang D. Yoo
>
> **备注:** CVPR 2025, Project Page: https://jiwoohong93.github.io/ita-mdt/
>
> **摘要:** This paper introduces ITA-MDT, the Image-Timestep-Adaptive Masked Diffusion Transformer Framework for Image-Based Virtual Try-On (IVTON), designed to overcome the limitations of previous approaches by leveraging the Masked Diffusion Transformer (MDT) for improved handling of both global garment context and fine-grained details. The IVTON task involves seamlessly superimposing a garment from one image onto a person in another, creating a realistic depiction of the person wearing the specified garment. Unlike conventional diffusion-based virtual try-on models that depend on large pre-trained U-Net architectures, ITA-MDT leverages a lightweight, scalable transformer-based denoising diffusion model with a mask latent modeling scheme, achieving competitive results while reducing computational overhead. A key component of ITA-MDT is the Image-Timestep Adaptive Feature Aggregator (ITAFA), a dynamic feature aggregator that combines all of the features from the image encoder into a unified feature of the same size, guided by diffusion timestep and garment image complexity. This enables adaptive weighting of features, allowing the model to emphasize either global information or fine-grained details based on the requirements of the denoising stage. Additionally, the Salient Region Extractor (SRE) module is presented to identify complex region of the garment to provide high-resolution local information to the denoising model as an additional condition alongside the global information of the full garment image. This targeted conditioning strategy enhances detail preservation of fine details in highly salient garment regions, optimizing computational resources by avoiding unnecessarily processing entire garment image. Comparative evaluations confirms that ITA-MDT improves efficiency while maintaining strong performance, reaching state-of-the-art results in several metrics.
>
---
#### [replaced 015] SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.09055v4](http://arxiv.org/pdf/2403.09055v4)**

> **作者:** Jaerin Lee; Daniel Sungho Jung; Kanggeon Lee; Kyoung Mu Lee
>
> **备注:** CVPR 2025 camera ready
>
> **摘要:** We introduce SemanticDraw, a new paradigm of interactive content creation where high-quality images are generated in near real-time from given multiple hand-drawn regions, each encoding prescribed semantic meaning. In order to maximize the productivity of content creators and to fully realize their artistic imagination, it requires both quick interactive interfaces and fine-grained regional controls in their tools. Despite astonishing generation quality from recent diffusion models, we find that existing approaches for regional controllability are very slow (52 seconds for $512 \times 512$ image) while not compatible with acceleration methods such as LCM, blocking their huge potential in interactive content creation. From this observation, we build our solution for interactive content creation in two steps: (1) we establish compatibility between region-based controls and acceleration techniques for diffusion models, maintaining high fidelity of multi-prompt image generation with $\times 10$ reduced number of inference steps, (2) we increase the generation throughput with our new multi-prompt stream batch pipeline, enabling low-latency generation from multiple, region-based text prompts on a single RTX 2080 Ti GPU. Our proposed framework is generalizable to any existing diffusion models and acceleration schedulers, allowing sub-second (0.64 seconds) image content creation application upon well-established image diffusion models. Our project page is: https://jaerinlee.com/research/semantic-draw
>
---
#### [replaced 016] Robust Adaptation of Foundation Models with Black-Box Visual Prompting
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.17491v2](http://arxiv.org/pdf/2407.17491v2)**

> **作者:** Changdae Oh; Gyeongdeok Seo; Geunyoung Jung; Zhi-Qi Cheng; Hosik Choi; Jiyoung Jung; Kyungwoo Song
>
> **备注:** Extended work from the CVPR'23 paper: arxiv:2303.14773; This paper has been submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) for possible publication
>
> **摘要:** With a surge of large-scale pre-trained models, parameter-efficient transfer learning (PETL) of large models has garnered significant attention. While promising, they commonly rely on two optimistic assumptions: 1) full access to the parameters of a PTM, and 2) sufficient memory capacity to cache all intermediate activations for gradient computation. However, in most real-world applications, PTMs serve as black-box APIs or proprietary software without full parameter accessibility. Besides, it is hard to meet a large memory requirement for modern PTMs. This work proposes black-box visual prompting (BlackVIP), which efficiently adapts the PTMs without knowledge of their architectures or parameters. BlackVIP has two components: 1) Coordinator and 2) simultaneous perturbation stochastic approximation with gradient correction (SPSA-GC). The Coordinator designs input-dependent visual prompts, which allow the target PTM to adapt in the wild. SPSA-GC efficiently estimates the gradient of PTM to update Coordinator. Besides, we introduce a variant, BlackVIP-SE, which significantly reduces the runtime and computational cost of BlackVIP. Extensive experiments on 19 datasets demonstrate that BlackVIPs enable robust adaptation to diverse domains and tasks with minimal memory requirements. We further provide a theoretical analysis on the generalization of visual prompting methods by presenting their connection to the certified robustness of randomized smoothing, and presenting an empirical support for improved robustness.
>
---
#### [replaced 017] SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting and a Physically Grounded Image Formation Model
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.17345v2](http://arxiv.org/pdf/2409.17345v2)**

> **作者:** Daniel Yang; John J. Leonard; Yogesh Girdhar
>
> **备注:** ICRA 2025. Project page here: https://seasplat.github.io
>
> **摘要:** We introduce SeaSplat, a method to enable real-time rendering of underwater scenes leveraging recent advances in 3D radiance fields. Underwater scenes are challenging visual environments, as rendering through a medium such as water introduces both range and color dependent effects on image capture. We constrain 3D Gaussian Splatting (3DGS), a recent advance in radiance fields enabling rapid training and real-time rendering of full 3D scenes, with a physically grounded underwater image formation model. Applying SeaSplat to the real-world scenes from SeaThru-NeRF dataset, a scene collected by an underwater vehicle in the US Virgin Islands, and simulation-degraded real-world scenes, not only do we see increased quantitative performance on rendering novel viewpoints from the scene with the medium present, but are also able to recover the underlying true color of the scene and restore renders to be without the presence of the intervening medium. We show that the underwater image formation helps learn scene structure, with better depth maps, as well as show that our improvements maintain the significant computational improvements afforded by leveraging a 3D Gaussian representation.
>
---
#### [replaced 018] Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings
- **分类: cs.AI; cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2310.17451v3](http://arxiv.org/pdf/2310.17451v3)**

> **作者:** Yifei Peng; Zijie Zha; Yu Jin; Zhexu Luo; Wang-Zhou Dai; Zhong Ren; Yao-Xiang Ding; Kun Zhou
>
> **备注:** KDD 2025 research track paper
>
> **摘要:** Making neural visual generative models controllable by logical reasoning systems is promising for improving faithfulness, transparency, and generalizability. We propose the Abductive visual Generation (AbdGen) approach to build such logic-integrated models. A vector-quantized symbol grounding mechanism and the corresponding disentanglement training method are introduced to enhance the controllability of logical symbols over generation. Furthermore, we propose two logical abduction methods to make our approach require few labeled training data and support the induction of latent logical generative rules from data. We experimentally show that our approach can be utilized to integrate various neural generative models with logical reasoning systems, by both learning from scratch or utilizing pre-trained models directly. The code is released at https://github.com/future-item/AbdGen.
>
---
#### [replaced 019] Towards Resource-Efficient Streaming of Large-Scale Medical Image Datasets for Deep Learning
- **分类: cs.CV; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2307.00438v3](http://arxiv.org/pdf/2307.00438v3)**

> **作者:** Pranav Kulkarni; Adway Kanhere; Eliot Siegel; Paul H. Yi; Vishwa S. Parekh
>
> **备注:** 17 pages, 4 figures, 10 tables, accepted to MIDL'25
>
> **摘要:** Large-scale medical imaging datasets have accelerated deep learning (DL) for medical image analysis. However, the large scale of these datasets poses a challenge for researchers, resulting in increased storage and bandwidth requirements for hosting and accessing them. Since different researchers have different use cases and require different resolutions or formats for DL, it is neither feasible to anticipate every researcher's needs nor practical to store data in multiple resolutions and formats. To that end, we propose the Medical Image Streaming Toolkit (MIST), a format-agnostic database that enables streaming of medical images at different resolutions and formats from a single high-resolution copy. We evaluated MIST across eight popular, large-scale medical imaging datasets spanning different body parts, modalities, and formats. Our results showed that our framework reduced the storage and bandwidth requirements for hosting and downloading datasets without impacting image quality. We demonstrate that MIST addresses the challenges posed by large-scale medical imaging datasets by building a data-efficient and format-agnostic database to meet the diverse needs of researchers and reduce barriers to DL research in medical imaging.
>
---
#### [replaced 020] RF4D:Neural Radar Fields for Novel View Synthesis in Outdoor Dynamic Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20967v2](http://arxiv.org/pdf/2505.20967v2)**

> **作者:** Jiarui Zhang; Zhihao Li; Chong Wang; Bihan Wen
>
> **摘要:** Neural fields (NFs) have demonstrated remarkable performance in scene reconstruction, powering various tasks such as novel view synthesis. However, existing NF methods relying on RGB or LiDAR inputs often exhibit severe fragility to adverse weather, particularly when applied in outdoor scenarios like autonomous driving. In contrast, millimeter-wave radar is inherently robust to environmental changes, while unfortunately, its integration with NFs remains largely underexplored. Besides, as outdoor driving scenarios frequently involve moving objects, making spatiotemporal modeling essential for temporally consistent novel view synthesis. To this end, we introduce RF4D, a radar-based neural field framework specifically designed for novel view synthesis in outdoor dynamic scenes. RF4D explicitly incorporates temporal information into its representation, significantly enhancing its capability to model moving objects. We further introduce a feature-level flow module that predicts latent temporal offsets between adjacent frames, enforcing temporal coherence in dynamic scene modeling. Moreover, we propose a radar-specific power rendering formulation closely aligned with radar sensing physics, improving synthesis accuracy and interoperability. Extensive experiments on public radar datasets demonstrate the superior performance of RF4D in terms of radar measurement synthesis quality and occupancy estimation accuracy, achieving especially pronounced improvements in dynamic outdoor scenarios.
>
---
#### [replaced 021] The Dual Power of Interpretable Token Embeddings: Jailbreaking Attacks and Defenses for Diffusion Model Unlearning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21307v2](http://arxiv.org/pdf/2504.21307v2)**

> **作者:** Siyi Chen; Yimeng Zhang; Sijia Liu; Qing Qu
>
> **摘要:** Despite the remarkable generation capabilities of diffusion models, recent studies have shown that they can memorize and create harmful content when given specific text prompts. Although fine-tuning approaches have been developed to mitigate this issue by unlearning harmful concepts, these methods can be easily circumvented through jailbreaking attacks. This implies that the harmful concept has not been fully erased from the model. However, existing jailbreaking attack methods, while effective, lack interpretability regarding why unlearned models still retain the concept, thereby hindering the development of defense strategies. In this work, we address these limitations by proposing an attack method that learns an orthogonal set of interpretable attack token embeddings. The attack token embeddings can be decomposed into human-interpretable textual elements, revealing that unlearned models still retain the target concept through implicit textual components. Furthermore, these attack token embeddings are powerful and transferable across text prompts, initial noises, and unlearned models, emphasizing that unlearned models are more vulnerable than expected. Finally, building on the insights from our interpretable attack, we develop a defense method to protect unlearned models against both our proposed and existing jailbreaking attacks. Extensive experimental results demonstrate the effectiveness of our attack and defense strategies.
>
---
#### [replaced 022] Absolute Coordinates Make Motion Generation Easy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19377v2](http://arxiv.org/pdf/2505.19377v2)**

> **作者:** Zichong Meng; Zeyu Han; Xiaogang Peng; Yiming Xie; Huaizu Jiang
>
> **备注:** Preprint
>
> **摘要:** State-of-the-art text-to-motion generation models rely on the kinematic-aware, local-relative motion representation popularized by HumanML3D, which encodes motion relative to the pelvis and to the previous frame with built-in redundancy. While this design simplifies training for earlier generation models, it introduces critical limitations for diffusion models and hinders applicability to downstream tasks. In this work, we revisit the motion representation and propose a radically simplified and long-abandoned alternative for text-to-motion generation: absolute joint coordinates in global space. Through systematic analysis of design choices, we show that this formulation achieves significantly higher motion fidelity, improved text alignment, and strong scalability, even with a simple Transformer backbone and no auxiliary kinematic-aware losses. Moreover, our formulation naturally supports downstream tasks such as text-driven motion control and temporal/spatial editing without additional task-specific reengineering and costly classifier guidance generation from control signals. Finally, we demonstrate promising generalization to directly generate SMPL-H mesh vertices in motion from text, laying a strong foundation for future research and motion-related applications.
>
---
#### [replaced 023] PADetBench: Towards Benchmarking Physical Attacks against Object Detection
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.09181v3](http://arxiv.org/pdf/2408.09181v3)**

> **作者:** Jiawei Lian; Jianhong Pan; Lefan Wang; Yi Wang; Lap-Pui Chau; Shaohui Mei
>
> **摘要:** Physical attacks against object detection have gained increasing attention due to their significant practical implications. However, conducting physical experiments is extremely time-consuming and labor-intensive. Moreover, physical dynamics and cross-domain transformation are challenging to strictly regulate in the real world, leading to unaligned evaluation and comparison, severely hindering the development of physically robust models. To accommodate these challenges, we explore utilizing realistic simulation to thoroughly and rigorously benchmark physical attacks with fairness under controlled physical dynamics and cross-domain transformation. This resolves the problem of capturing identical adversarial images that cannot be achieved in the real world. Our benchmark includes 20 physical attack methods, 48 object detectors, comprehensive physical dynamics, and evaluation metrics. We also provide end-to-end pipelines for dataset generation, detection, evaluation, and further analysis. In addition, we perform 8064 groups of evaluation based on our benchmark, which includes both overall evaluation and further detailed ablation studies for controlled physical dynamics. Through these experiments, we provide in-depth analyses of physical attack performance and physical adversarial robustness, draw valuable observations, and discuss potential directions for future research. Codebase: https://github.com/JiaweiLian/Benchmarking_Physical_Attack
>
---
#### [replaced 024] FastCAR: Fast Classification And Regression Multi-Task Learning via Task Consolidation for Modelling a Continuous Property Variable of Object Classes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.17926v2](http://arxiv.org/pdf/2403.17926v2)**

> **作者:** Anoop Kini; Andreas Jansche; Timo Bernthaler; Gerhard Schneider
>
> **摘要:** FastCAR is a novel task consolidation approach in Multi-Task Learning (MTL) for a classification and a regression task, despite task heterogeneity with only subtle correlation. It addresses object classification and continuous property variable regression, a crucial use case in science and engineering. FastCAR involves a labeling transformation approach that can be used with a single-task regression network architecture. FastCAR outperforms traditional MTL model families, parametrized in the landscape of architecture and loss weighting schemes, when learning of both tasks are collectively considered (classification accuracy of 99.54\%, regression mean absolute percentage error of 2.4\%). The experiments performed used an Advanced Steel Property dataset https://github.com/fastcandr/Advanced-Steel-Property-Dataset contributed by us. The dataset comprises 4536 images of 224x224 pixels, annotated with object classes and hardness properties that take continuous values. With our designed approach, FastCAR achieves reduced latency and time efficiency.
>
---
#### [replaced 025] OmniCaptioner: One Captioner to Rule Them All
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07089v3](http://arxiv.org/pdf/2504.07089v3)**

> **作者:** Yiting Lu; Jiakang Yuan; Zhen Li; Shitian Zhao; Qi Qin; Xinyue Li; Le Zhuo; Licheng Wen; Dongyang Liu; Yuewen Cao; Xiangchao Yan; Xin Li; Tianshuo Peng; Shufei Zhang; Botian Shi; Tao Chen; Zhibo Chen; Lei Bai; Peng Gao; Bo Zhang
>
> **备注:** More visualizations on Homepage: https://alpha-innovator.github.io/OmniCaptioner-project-page and Official code: https://github.com/Alpha-Innovator/OmniCaptioner
>
> **摘要:** We propose OmniCaptioner, a versatile visual captioning framework for generating fine-grained textual descriptions across a wide variety of visual domains. Unlike prior methods limited to specific image types (e.g., natural images or geometric visuals), our framework provides a unified solution for captioning natural images, visual text (e.g., posters, UIs, textbooks), and structured visuals (e.g., documents, tables, charts). By converting low-level pixel information into semantically rich textual representations, our framework bridges the gap between visual and textual modalities. Our results highlight three key advantages: (i) Enhanced Visual Reasoning with LLMs, where long-context captions of visual modalities empower LLMs, particularly the DeepSeek-R1 series, to reason effectively in multimodal scenarios; (ii) Improved Image Generation, where detailed captions improve tasks like text-to-image generation and image transformation; and (iii) Efficient Supervised Fine-Tuning (SFT), which enables faster convergence with less data. We believe the versatility and adaptability of OmniCaptioner can offer a new perspective for bridging the gap between language and visual modalities.
>
---
#### [replaced 026] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.01377v2](http://arxiv.org/pdf/2501.01377v2)**

> **作者:** Yucheng Zhou; Lingran Song; Jianbing Shen
>
> **备注:** 16 pages
>
> **摘要:** Existing Medical Large Vision-Language Models (Med-LVLMs), encapsulating extensive medical knowledge, demonstrate excellent capabilities in understanding medical images. However, there remain challenges in visual localization in medical images, which is crucial for abnormality detection and interpretation. To address these issues, we propose a novel UMed-LVLM designed to unveil medical abnormalities. Specifically, we collect a Medical Abnormalities Unveiling (MAU) dataset and propose a two-stage training method for UMed-LVLM training. To collect MAU dataset, we propose a prompt method utilizing the GPT-4V to generate diagnoses based on identified abnormal areas in medical images. Moreover, the two-stage training method includes Abnormal-Aware Instruction Tuning and Abnormal-Aware Rewarding, comprising Relevance Reward, Abnormal Localization Reward and Vision Relevance Reward. Experimental results demonstrate that our UMed-LVLM significantly outperforms existing Med-LVLMs in identifying and understanding medical abnormalities, achieving a 58% improvement over the baseline. In addition, this work shows that enhancing the abnormality detection capabilities of Med-LVLMs significantly improves their understanding of medical images and generalization capability.
>
---
#### [replaced 027] CRAVES: Controlling Robotic Arm with a Vision-based Economic System
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1812.00725v3](http://arxiv.org/pdf/1812.00725v3)**

> **作者:** Yiming Zuo; Weichao Qiu; Lingxi Xie; Fangwei Zhong; Yizhou Wang; Alan L. Yuille
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Training a robotic arm to accomplish real-world tasks has been attracting increasing attention in both academia and industry. This work discusses the role of computer vision algorithms in this field. We focus on low-cost arms on which no sensors are equipped and thus all decisions are made upon visual recognition, e.g., real-time 3D pose estimation. This requires annotating a lot of training data, which is not only time-consuming but also laborious. In this paper, we present an alternative solution, which uses a 3D model to create a large number of synthetic data, trains a vision model in this virtual domain, and applies it to real-world images after domain adaptation. To this end, we design a semi-supervised approach, which fully leverages the geometric constraints among keypoints. We apply an iterative algorithm for optimization. Without any annotations on real images, our algorithm generalizes well and produces satisfying results on 3D pose estimation, which is evaluated on two real-world datasets. We also construct a vision-based control system for task accomplishment, for which we train a reinforcement learning agent in a virtual environment and apply it to the real-world. Moreover, our approach, with merely a 3D model being required, has the potential to generalize to other types of multi-rigid-body dynamic systems. Website: https://qiuwch.github.io/craves.ai. Code: https://github.com/zuoym15/craves.ai
>
---
#### [replaced 028] MSDNet: Multi-Scale Decoder for Few-Shot Semantic Segmentation via Transformer-Guided Prototyping
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.11316v3](http://arxiv.org/pdf/2409.11316v3)**

> **作者:** Amirreza Fateh; Mohammad Reza Mohammadi; Mohammad Reza Jahed Motlagh
>
> **摘要:** Few-shot Semantic Segmentation addresses the challenge of segmenting objects in query images with only a handful of annotated examples. However, many previous state-of-the-art methods either have to discard intricate local semantic features or suffer from high computational complexity. To address these challenges, we propose a new Few-shot Semantic Segmentation framework based on the Transformer architecture. Our approach introduces the spatial transformer decoder and the contextual mask generation module to improve the relational understanding between support and query images. Moreover, we introduce a multi scale decoder to refine the segmentation mask by incorporating features from different resolutions in a hierarchical manner. Additionally, our approach integrates global features from intermediate encoder stages to improve contextual understanding, while maintaining a lightweight structure to reduce complexity. This balance between performance and efficiency enables our method to achieve competitive results on benchmark datasets such as PASCAL-5^i and COCO-20^i in both 1-shot and 5-shot settings. Notably, our model with only 1.5 million parameters demonstrates competitive performance while overcoming limitations of existing methodologies.
>
---
#### [replaced 029] SCC-YOLO: An Improved Object Detector for Assisting in Brain Tumor Diagnosis
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.03836v4](http://arxiv.org/pdf/2501.03836v4)**

> **作者:** Runci Bai; Guibao Xu; Yanze Shi
>
> **摘要:** Brain tumors can lead to neurological dysfunction, cognitive and psychological changes, increased intracranial pressure, and seizures, posing significant risks to health. The You Only Look Once (YOLO) series has shown superior accuracy in medical imaging object detection. This paper presents a novel SCC-YOLO architecture that integrates the SCConv module into YOLOv9. The SCConv module optimizes convolutional efficiency by reducing spatial and channel redundancy, enhancing image feature learning. We examine the effects of different attention mechanisms with YOLOv9 for brain tumor detection using the Br35H dataset and our custom dataset (Brain_Tumor_Dataset). Results indicate that SCC-YOLO improved mAP50 by 0.3% on the Br35H dataset and by 0.5% on our custom dataset compared to YOLOv9. SCC-YOLO achieves state-of-the-art performance in brain tumor detection.
>
---
#### [replaced 030] Certified Robustness to Clean-Label Poisoning Using Diffusion Denoising
- **分类: cs.CR; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.11981v2](http://arxiv.org/pdf/2403.11981v2)**

> **作者:** Sanghyun Hong; Nicholas Carlini; Alexey Kurakin
>
> **摘要:** We present a certified defense to clean-label poisoning attacks under $\ell_2$-norm. These attacks work by injecting a small number of poisoning samples (e.g., 1%) that contain bounded adversarial perturbations into the training data to induce a targeted misclassification of a test-time input. Inspired by the adversarial robustness achieved by $randomized$ $smoothing$, we show how an off-the-shelf diffusion denoising model can sanitize the tampered training data. We extensively test our defense against seven clean-label poisoning attacks in both $\ell_2$ and $\ell_{\infty}$-norms and reduce their attack success to 0-16% with only a negligible drop in the test accuracy. We compare our defense with existing countermeasures against clean-label poisoning, showing that the defense reduces the attack success the most and offers the best model utility. Our results highlight the need for future work on developing stronger clean-label attacks and using our certified yet practical defense as a strong baseline to evaluate these attacks.
>
---
#### [replaced 031] Graph-Driven Multimodal Feature Learning Framework for Apparent Personality Assessment
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.11515v2](http://arxiv.org/pdf/2504.11515v2)**

> **作者:** Kangsheng Wang; Chengwei Ye; Huanzhen Zhang; Linuo Xu; Shuyan Liu
>
> **备注:** The article contains serious scientific errors and cannot be corrected by updating the preprint
>
> **摘要:** Predicting personality traits automatically has become a challenging problem in computer vision. This paper introduces an innovative multimodal feature learning framework for personality analysis in short video clips. For visual processing, we construct a facial graph and design a Geo-based two-stream network incorporating an attention mechanism, leveraging both Graph Convolutional Networks (GCN) and Convolutional Neural Networks (CNN) to capture static facial expressions. Additionally, ResNet18 and VGGFace networks are employed to extract global scene and facial appearance features at the frame level. To capture dynamic temporal information, we integrate a BiGRU with a temporal attention module for extracting salient frame representations. To enhance the model's robustness, we incorporate the VGGish CNN for audio-based features and XLM-Roberta for text-based features. Finally, a multimodal channel attention mechanism is introduced to integrate different modalities, and a Multi-Layer Perceptron (MLP) regression model is used to predict personality traits. Experimental results confirm that our proposed framework surpasses existing state-of-the-art approaches in performance.
>
---
#### [replaced 032] M$^3$-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.13803v3](http://arxiv.org/pdf/2412.13803v3)**

> **作者:** Zixuan Chen; Jiaxin Li; Liming Tan; Yejie Guo; Junxuan Liang; Cewu Lu; Yong-Lu Li
>
> **备注:** 18 pages, 12 figures
>
> **摘要:** Intelligent robots need to interact with diverse objects across various environments. The appearance and state of objects frequently undergo complex transformations depending on the object properties, e.g., phase transitions. However, in the vision community, segmenting dynamic objects with phase transitions is overlooked. In light of this, we introduce the concept of phase in segmentation, which categorizes real-world objects based on their visual characteristics and potential morphological and appearance changes. Then, we present a new benchmark, Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation (M$^3$-VOS), to verify the ability of models to understand object phases, which consists of 479 high-resolution videos spanning over 10 distinct everyday scenarios. It provides dense instance mask annotations that capture both object phases and their transitions. We evaluate state-of-the-art methods on M$^3$-VOS, yielding several key insights. Notably, current appearance-based approaches show significant room for improvement when handling objects with phase transitions. The inherent changes in disorder suggest that the predictive performance of the forward entropy-increasing process can be improved through a reverse entropy-reducing process. These findings lead us to propose ReVOS, a new plug-andplay model that improves its performance by reversal refinement. Our data and code will be publicly available at https://zixuan-chen.github.io/M-cube-VOS.github.io/.
>
---
#### [replaced 033] DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23694v2](http://arxiv.org/pdf/2505.23694v2)**

> **作者:** Li Ren; Chen Chen; Liqiang Wang; Kien Hua
>
> **备注:** CVPR 2025
>
> **摘要:** Visual Prompt Tuning (VPT) has become a promising solution for Parameter-Efficient Fine-Tuning (PEFT) approach for Vision Transformer (ViT) models by partially fine-tuning learnable tokens while keeping most model parameters frozen. Recent research has explored modifying the connection structures of the prompts. However, the fundamental correlation and distribution between the prompts and image tokens remain unexplored. In this paper, we leverage metric learning techniques to investigate how the distribution of prompts affects fine-tuning performance. Specifically, we propose a novel framework, Distribution Aware Visual Prompt Tuning (DA-VPT), to guide the distributions of the prompts by learning the distance metric from their class-related semantic data. Our method demonstrates that the prompts can serve as an effective bridge to share semantic information between image patches and the class token. We extensively evaluated our approach on popular benchmarks in both recognition and segmentation tasks. The results demonstrate that our approach enables more effective and efficient fine-tuning of ViT models by leveraging semantic information to guide the learning of the prompts, leading to improved performance on various downstream vision tasks.
>
---
#### [replaced 034] CTRL-GS: Cascaded Temporal Residue Learning for 4D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18306v2](http://arxiv.org/pdf/2505.18306v2)**

> **作者:** Karly Hou; Wanhua Li; Hanspeter Pfister
>
> **备注:** Accepted to 4D Vision Workshop @ CVPR 2025
>
> **摘要:** Recently, Gaussian Splatting methods have emerged as a desirable substitute for prior Radiance Field methods for novel-view synthesis of scenes captured with multi-view images or videos. In this work, we propose a novel extension to 4D Gaussian Splatting for dynamic scenes. Drawing on ideas from residual learning, we hierarchically decompose the dynamic scene into a "video-segment-frame" structure, with segments dynamically adjusted by optical flow. Then, instead of directly predicting the time-dependent signals, we model the signal as the sum of video-constant values, segment-constant values, and frame-specific residuals, as inspired by the success of residual learning. This approach allows more flexible models that adapt to highly variable scenes. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets, with the greatest improvements on complex scenes with large movements, occlusions, and fine details, where current methods degrade most.
>
---
#### [replaced 035] ADS-Edit: A Multimodal Knowledge Editing Dataset for Autonomous Driving Systems
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.20756v2](http://arxiv.org/pdf/2503.20756v2)**

> **作者:** Chenxi Wang; Jizhan Fang; Xiang Chen; Bozhong Tian; Ziwen Xu; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Recent advancements in Large Multimodal Models (LMMs) have shown promise in Autonomous Driving Systems (ADS). However, their direct application to ADS is hindered by challenges such as misunderstanding of traffic knowledge, complex road conditions, and diverse states of vehicle. To address these challenges, we propose the use of Knowledge Editing, which enables targeted modifications to a model's behavior without the need for full retraining. Meanwhile, we introduce ADS-Edit, a multimodal knowledge editing dataset specifically designed for ADS, which includes various real-world scenarios, multiple data types, and comprehensive evaluation metrics. We conduct comprehensive experiments and derive several interesting conclusions. We hope that our work will contribute to the further advancement of knowledge editing applications in the field of autonomous driving. Code and data are available in https://github.com/zjunlp/EasyEdit.
>
---
#### [replaced 036] Benchmarking 3D Human Pose Estimation Models under Occlusions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10350v2](http://arxiv.org/pdf/2504.10350v2)**

> **作者:** Filipa Lino; Carlos Santiago; Manuel Marques
>
> **摘要:** Human Pose Estimation (HPE) involves detecting and localizing keypoints on the human body from visual data. In 3D HPE, occlusions, where parts of the body are not visible in the image, pose a significant challenge for accurate pose reconstruction. This paper presents a benchmark on the robustness of 3D HPE models under realistic occlusion conditions, involving combinations of occluded keypoints commonly observed in real-world scenarios. We evaluate nine state-of-the-art 2D-to-3D HPE models, spanning convolutional, transformer-based, graph-based, and diffusion-based architectures, using the BlendMimic3D dataset, a synthetic dataset with ground-truth 2D/3D annotations and occlusion labels. All models were originally trained on Human3.6M and tested here without retraining to assess their generalization. We introduce a protocol that simulates occlusion by adding noise into 2D keypoints based on real detector behavior, and conduct both global and per-joint sensitivity analyses. Our findings reveal that all models exhibit notable performance degradation under occlusion, with diffusion-based models underperforming despite their stochastic nature. Additionally, a per-joint occlusion analysis identifies consistent vulnerability in distal joints (e.g., wrists, feet) across models. Overall, this work highlights critical limitations of current 3D HPE models in handling occlusions, and provides insights for improving real-world robustness.
>
---
#### [replaced 037] Urban Safety Perception Assessments via Integrating Multimodal Large Language Models with Street View Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.19719v3](http://arxiv.org/pdf/2407.19719v3)**

> **作者:** Jiaxin Zhang; Yunqin Li; Tomohiro Fukuda; Bowen Wang
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Measuring urban safety perception is an important and complex task that traditionally relies heavily on human resources. This process often involves extensive field surveys, manual data collection, and subjective assessments, which can be time-consuming, costly, and sometimes inconsistent. Street View Images (SVIs), along with deep learning methods, provide a way to realize large-scale urban safety detection. However, achieving this goal often requires extensive human annotation to train safety ranking models, and the architectural differences between cities hinder the transferability of these models. Thus, a fully automated method for conducting safety evaluations is essential. Recent advances in multimodal large language models (MLLMs) have demonstrated powerful reasoning and analytical capabilities. Cutting-edge models, e.g., GPT-4 have shown surprising performance in many tasks. We employed these models for urban safety ranking on a human-annotated anchor set and validated that the results from MLLMs align closely with human perceptions. Additionally, we proposed a method based on the pre-trained Contrastive Language-Image Pre-training (CLIP) feature and K-Nearest Neighbors (K-NN) retrieval to quickly assess the safety index of the entire city. Experimental results show that our method outperforms existing training needed deep learning approaches, achieving efficient and accurate urban safety evaluations. The proposed automation for urban safety perception assessment is a valuable tool for city planners, policymakers, and researchers aiming to improve urban environments.
>
---
#### [replaced 038] MIRAGE: Assessing Hallucination in Multimodal Reasoning Chains of MLLM
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.24238v2](http://arxiv.org/pdf/2505.24238v2)**

> **作者:** Bowen Dong; Minheng Ni; Zitong Huang; Guanglei Yang; Wangmeng Zuo; Lei Zhang
>
> **摘要:** Multimodal hallucination in multimodal large language models (MLLMs) restricts the correctness of MLLMs. However, multimodal hallucinations are multi-sourced and arise from diverse causes. Existing benchmarks fail to adequately distinguish between perception-induced hallucinations and reasoning-induced hallucinations. This failure constitutes a significant issue and hinders the diagnosis of multimodal reasoning failures within MLLMs. To address this, we propose the {\dataset} benchmark, which isolates reasoning hallucinations by constructing questions where input images are correctly perceived by MLLMs yet reasoning errors persist. {\dataset} introduces multi-granular evaluation metrics: accuracy, factuality, and LLMs hallucination score for hallucination quantification. Our analysis reveals that (1) the model scale, data scale, and training stages significantly affect the degree of logical, fabrication, and factual hallucinations; (2) current MLLMs show no effective improvement on spatial hallucinations caused by misinterpreted spatial relationships, indicating their limited visual reasoning capabilities; and (3) question types correlate with distinct hallucination patterns, highlighting targeted challenges and potential mitigation strategies. To address these challenges, we propose {\method}, a method that combines curriculum reinforcement fine-tuning to encourage models to generate logic-consistent reasoning chains by stepwise reducing learning difficulty, and collaborative hint inference to reduce reasoning complexity. {\method} establishes a baseline on {\dataset}, and reduces the logical hallucinations in original base models.
>
---
#### [replaced 039] DNTextSpotter: Arbitrary-Shaped Scene Text Spotting via Improved Denoising Training
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.00355v4](http://arxiv.org/pdf/2408.00355v4)**

> **作者:** Yu Xie; Qian Qiao; Jun Gao; Tianxiang Wu; Jiaqing Fan; Yue Zhang; Jielei Zhang; Huyang Sun
>
> **备注:** Accepted by ACM'MM2024
>
> **摘要:** More and more end-to-end text spotting methods based on Transformer architecture have demonstrated superior performance. These methods utilize a bipartite graph matching algorithm to perform one-to-one optimal matching between predicted objects and actual objects. However, the instability of bipartite graph matching can lead to inconsistent optimization targets, thereby affecting the training performance of the model. Existing literature applies denoising training to solve the problem of bipartite graph matching instability in object detection tasks. Unfortunately, this denoising training method cannot be directly applied to text spotting tasks, as these tasks need to perform irregular shape detection tasks and more complex text recognition tasks than classification. To address this issue, we propose a novel denoising training method (DNTextSpotter) for arbitrary-shaped text spotting. Specifically, we decompose the queries of the denoising part into noised positional queries and noised content queries. We use the four Bezier control points of the Bezier center curve to generate the noised positional queries. For the noised content queries, considering that the output of the text in a fixed positional order is not conducive to aligning position with content, we employ a masked character sliding method to initialize noised content queries, thereby assisting in the alignment of text content and position. To improve the model's perception of the background, we further utilize an additional loss function for background characters classification in the denoising training part.Although DNTextSpotter is conceptually simple, it outperforms the state-of-the-art methods on four benchmarks (Total-Text, SCUT-CTW1500, ICDAR15, and Inverse-Text), especially yielding an improvement of 11.3% against the best approach in Inverse-Text dataset.
>
---
#### [replaced 040] Keypoint-Integrated Instruction-Following Data Generation for Enhanced Human Pose and Action Understanding in Multimodal Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.09306v2](http://arxiv.org/pdf/2409.09306v2)**

> **作者:** Dewen Zhang; Wangpeng An; Hayaru Shouno
>
> **备注:** Accepted at the International Conference on Advanced Concepts for Intelligent Vision Systems (ACIVS 2025)
>
> **摘要:** Current vision-language multimodal models are well-adapted for general visual understanding tasks. However, they perform inadequately when handling complex visual tasks related to human poses and actions due to the lack of specialized vision-language instruction-following data. We introduce a method for generating such data by integrating human keypoints with traditional visual features such as captions and bounding boxes, enabling more precise understanding of human-centric scenes. Our approach constructs a dataset comprising 200,328 samples tailored to fine-tune models for human-centric tasks, focusing on three areas: conversation, detailed description, and complex reasoning. We establish a benchmark called Human Pose and Action Understanding Benchmark (HPAUB) to assess model performance on human pose and action understanding. We fine-tune the LLaVA-1.5-7B model using this dataset and evaluate it on the benchmark, achieving significant improvements. Experimental results show an overall improvement of 21.18% compared to the original LLaVA-1.5-7B model. These findings highlight the effectiveness of keypoint-integrated data in enhancing multimodal models. Code is available at https://github.com/Ody-trek/Keypoint-Instruction-Tuning.
>
---
#### [replaced 041] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.07171v3](http://arxiv.org/pdf/2407.07171v3)**

> **作者:** Yuyuan Liu; Yuanhong Chen; Hu Wang; Vasileios Belagiannis; Ian Reid; Gustavo Carneiro
>
> **备注:** 27 pages (15 pages main paper and 12 pages supplementary with references), ECCV 2024 accepted
>
> **摘要:** The costly and time-consuming annotation process to produce large training sets for modelling semantic LiDAR segmentation methods has motivated the development of semi-supervised learning (SSL) methods. However, such SSL approaches often concentrate on employing consistency learning only for individual LiDAR representations. This narrow focus results in limited perturbations that generally fail to enable effective consistency learning. Additionally, these SSL approaches employ contrastive learning based on the sampling from a limited set of positive and negative embedding samples. This paper introduces a novel semi-supervised LiDAR semantic segmentation framework called ItTakesTwo (IT2). IT2 is designed to ensure consistent predictions from peer LiDAR representations, thereby improving the perturbation effectiveness in consistency learning. Furthermore, our contrastive learning employs informative samples drawn from a distribution of positive and negative embeddings learned from the entire training set. Results on public benchmarks show that our approach achieves remarkable improvements over the previous state-of-the-art (SOTA) methods in the field. The code is available at: https://github.com/yyliu01/IT2.
>
---
#### [replaced 042] Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23566v2](http://arxiv.org/pdf/2505.23566v2)**

> **作者:** Yu Li; Jin Jiang; Jianhua Zhu; Shuai Peng; Baole Wei; Yuxuan Zhou; Liangcai Gao
>
> **摘要:** Handwritten Mathematical Expression Recognition (HMER) remains a persistent challenge in Optical Character Recognition (OCR) due to the inherent freedom of symbol layout and variability in handwriting styles. Prior methods have faced performance bottlenecks, proposing isolated architectural modifications that are difficult to integrate coherently into a unified framework. Meanwhile, recent advances in pretrained vision-language models (VLMs) have demonstrated strong cross-task generalization, offering a promising foundation for developing unified solutions. In this paper, we introduce Uni-MuMER, which fully fine-tunes a VLM for the HMER task without modifying its architecture, effectively injecting domain-specific knowledge into a generalist framework. Our method integrates three data-driven tasks: Tree-Aware Chain-of-Thought (Tree-CoT) for structured spatial reasoning, Error-Driven Learning (EDL) for reducing confusion among visually similar characters, and Symbol Counting (SC) for improving recognition consistency in long expressions. Experiments on the CROHME and HME100K datasets show that Uni-MuMER achieves new state-of-the-art performance, surpassing the best lightweight specialized model SSAN by 16.31% and the top-performing VLM Gemini2.5-flash by 24.42% in the zero-shot setting. Our datasets, models, and code are open-sourced at: https://github.com/BFlameSwift/Uni-MuMER
>
---
#### [replaced 043] FreeInsert: Disentangled Text-Guided Object Insertion in 3D Gaussian Scene without Spatial Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01322v2](http://arxiv.org/pdf/2505.01322v2)**

> **作者:** Chenxi Li; Weijie Wang; Qiang Li; Bruno Lepri; Nicu Sebe; Weizhi Nie
>
> **摘要:** Text-driven object insertion in 3D scenes is an emerging task that enables intuitive scene editing through natural language. However, existing 2D editing-based methods often rely on spatial priors such as 2D masks or 3D bounding boxes, and they struggle to ensure consistency of the inserted object. These limitations hinder flexibility and scalability in real-world applications. In this paper, we propose FreeInsert, a novel framework that leverages foundation models including MLLMs, LGMs, and diffusion models to disentangle object generation from spatial placement. This enables unsupervised and flexible object insertion in 3D scenes without spatial priors. FreeInsert starts with an MLLM-based parser that extracts structured semantics, including object types, spatial relationships, and attachment regions, from user instructions. These semantics guide both the reconstruction of the inserted object for 3D consistency and the learning of its degrees of freedom. We leverage the spatial reasoning capabilities of MLLMs to initialize object pose and scale. A hierarchical, spatially aware refinement stage further integrates spatial semantics and MLLM-inferred priors to enhance placement. Finally, the appearance of the object is improved using the inserted-object image to enhance visual fidelity. Experimental results demonstrate that FreeInsert achieves semantically coherent, spatially precise, and visually realistic 3D insertions without relying on spatial priors, offering a user-friendly and flexible editing experience.
>
---
#### [replaced 044] More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21523v2](http://arxiv.org/pdf/2505.21523v2)**

> **作者:** Chengzhi Liu; Zhongxing Xu; Qingyue Wei; Juncheng Wu; James Zou; Xin Eric Wang; Yuyin Zhou; Sheng Liu
>
> **摘要:** Test-time compute has empowered multimodal large language models to generate extended reasoning chains, yielding strong performance on tasks such as multimodal math reasoning. However, this improved reasoning ability often comes with increased hallucination: as generations become longer, models tend to drift away from image-grounded content and rely more heavily on language priors. Attention analysis shows that longer reasoning chains lead to reduced focus on visual inputs, which contributes to hallucination. To systematically study this phenomenon, we introduce RH-AUC, a metric that quantifies how a model's perception accuracy changes with reasoning length, allowing us to evaluate whether the model preserves visual grounding during reasoning. We also release RH-Bench, a diagnostic benchmark that spans a variety of multimodal tasks, designed to assess the trade-off between reasoning ability and hallucination. Our analysis reveals that (i) larger models typically achieve a better balance between reasoning and perception, and (ii) this balance is influenced more by the types and domains of training data than by its overall volume. These findings underscore the importance of evaluation frameworks that jointly consider both reasoning quality and perceptual fidelity.
>
---
#### [replaced 045] An Interpretable Representation Learning Approach for Diffusion Tensor Imaging
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.19110v2](http://arxiv.org/pdf/2505.19110v2)**

> **作者:** Vishwa Mohan Singh; Alberto Gaston Villagran Asiares; Luisa Sophie Schuhmacher; Kate Rendall; Simon Weißbrod; David Rügamer; Inga Körte
>
> **备注:** Accepted for publication at MIDL 2025
>
> **摘要:** Diffusion Tensor Imaging (DTI) tractography offers detailed insights into the structural connectivity of the brain, but presents challenges in effective representation and interpretation in deep learning models. In this work, we propose a novel 2D representation of DTI tractography that encodes tract-level fractional anisotropy (FA) values into a 9x9 grayscale image. This representation is processed through a Beta-Total Correlation Variational Autoencoder with a Spatial Broadcast Decoder to learn a disentangled and interpretable latent embedding. We evaluate the quality of this embedding using supervised and unsupervised representation learning strategies, including auxiliary classification, triplet loss, and SimCLR-based contrastive learning. Compared to the 1D Group deep neural network (DNN) baselines, our approach improves the F1 score in a downstream sex classification task by 15.74% and shows a better disentanglement than the 3D representation.
>
---
#### [replaced 046] Understanding differences in applying DETR to natural and medical images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.17677v2](http://arxiv.org/pdf/2405.17677v2)**

> **作者:** Yanqi Xu; Yiqiu Shen; Carlos Fernandez-Granda; Laura Heacock; Krzysztof J. Geras
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) https://melba-journal.org/2025:009
>
> **摘要:** Transformer-based detectors have shown success in computer vision tasks with natural images. These models, exemplified by the Deformable DETR, are optimized through complex engineering strategies tailored to the typical characteristics of natural scenes. However, medical imaging data presents unique challenges such as extremely large image sizes, fewer and smaller regions of interest, and object classes which can be differentiated only through subtle differences. This study evaluates the applicability of these transformer-based design choices when applied to a screening mammography dataset that represents these distinct medical imaging data characteristics. Our analysis reveals that common design choices from the natural image domain, such as complex encoder architectures, multi-scale feature fusion, query initialization, and iterative bounding box refinement, do not improve and sometimes even impair object detection performance in medical imaging. In contrast, simpler and shallower architectures often achieve equal or superior results. This finding suggests that the adaptation of transformer models for medical imaging data requires a reevaluation of standard practices, potentially leading to more efficient and specialized frameworks for medical diagnosis.
>
---
#### [replaced 047] Monge-Ampere Regularization for Learning Arbitrary Shapes from Point Clouds
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.18477v3](http://arxiv.org/pdf/2410.18477v3)**

> **作者:** Chuanxiang Yang; Yuanfeng Zhou; Guangshun Wei; Long Ma; Junhui Hou; Yuan Liu; Wenping Wang
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Project Page: https://chuanxiang-yang.github.io/S2DF/, Code: https://github.com/chuanxiang-yang/S2DF
>
> **摘要:** As commonly used implicit geometry representations, the signed distance function (SDF) is limited to modeling watertight shapes, while the unsigned distance function (UDF) is capable of representing various surfaces. However, its inherent theoretical shortcoming, i.e., the non-differentiability at the zero level set, would result in sub-optimal reconstruction quality. In this paper, we propose the scaled-squared distance function (S$^{2}$DF), a novel implicit surface representation for modeling arbitrary surface types. S$^{2}$DF does not distinguish between inside and outside regions while effectively addressing the non-differentiability issue of UDF at the zero level set. We demonstrate that S$^{2}$DF satisfies a second-order partial differential equation of Monge-Ampere-type, allowing us to develop a learning pipeline that leverages a novel Monge-Ampere regularization to directly learn S$^{2}$DF from raw unoriented point clouds without supervision from ground-truth S$^{2}$DF values. Extensive experiments across multiple datasets show that our method significantly outperforms state-of-the-art supervised approaches that require ground-truth surface information as supervision for training. The source code is available at https://github.com/chuanxiang-yang/S2DF.
>
---
#### [replaced 048] GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19480v2](http://arxiv.org/pdf/2503.19480v2)**

> **作者:** Shijie Ma; Yuying Ge; Teng Wang; Yuxin Guo; Yixiao Ge; Ying Shan
>
> **备注:** Project released at: https://mashijie1028.github.io/GenHancer/
>
> **摘要:** The synergy between generative and discriminative models receives growing attention. While discriminative Contrastive Language-Image Pre-Training (CLIP) excels in high-level semantics, it struggles with perceiving fine-grained visual details. Generally, to enhance representations, generative models take CLIP's visual features as conditions for reconstruction. However, the underlying principle remains underexplored. In this work, we empirically found that visually perfect generations are not always optimal for representation enhancement. The essence lies in effectively extracting fine-grained knowledge from generative models while mitigating irrelevant information. To explore critical factors, we delve into three aspects: (1) Conditioning mechanisms: We found that even a small number of local tokens can drastically reduce the difficulty of reconstruction, leading to collapsed training. We thus conclude that utilizing only global visual tokens as conditions is the most effective strategy. (2) Denoising configurations: We observed that end-to-end training introduces extraneous information. To address this, we propose a two-stage training strategy to prioritize learning useful visual knowledge. Additionally, we demonstrate that lightweight denoisers can yield remarkable improvements. (3) Generation paradigms: We explore both continuous and discrete denoisers with desirable outcomes, validating the versatility of our method. Through our in-depth explorations, we have finally arrived at an effective method, namely GenHancer, which consistently outperforms prior arts on the MMVP-VLM benchmark, e.g., 6.0% on OpenAICLIP. The enhanced CLIP can be further plugged into multimodal large language models for better vision-centric performance. All the models and codes are made publicly available.
>
---
#### [replaced 049] Deep Learning Framework for Infrastructure Maintenance: Crack Detection and High-Resolution Imaging of Infrastructure Surfaces
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.03974v2](http://arxiv.org/pdf/2505.03974v2)**

> **作者:** Nikhil M. Pawar; Jorge A. Prozzi; Feng Hong; Surya Sarat Chandra Congress
>
> **备注:** Presented :Transportation Research Board 104th Annual Meeting, Washington, D.C
>
> **摘要:** Recently, there has been an impetus for the application of cutting-edge data collection platforms such as drones mounted with camera sensors for infrastructure asset management. However, the sensor characteristics, proximity to the structure, hard-to-reach access, and environmental conditions often limit the resolution of the datasets. A few studies used super-resolution techniques to address the problem of low-resolution images. Nevertheless, these techniques were observed to increase computational cost and false alarms of distress detection due to the consideration of all the infrastructure images i.e., positive and negative distress classes. In order to address the pre-processing of false alarm and achieve efficient super-resolution, this study developed a framework consisting of convolutional neural network (CNN) and efficient sub-pixel convolutional neural network (ESPCNN). CNN accurately classified both the classes. ESPCNN, which is the lightweight super-resolution technique, generated high-resolution infrastructure image of positive distress obtained from CNN. The ESPCNN outperformed bicubic interpolation in all the evaluation metrics for super-resolution. Based on the performance metrics, the combination of CNN and ESPCNN was observed to be effective in preprocessing the infrastructure images with negative distress, reducing the computational cost and false alarms in the next step of super-resolution. The visual inspection showed that EPSCNN is able to capture crack propagation, complex geometry of even minor cracks. The proposed framework is expected to help the highway agencies in accurately performing distress detection and assist in efficient asset management practices.
>
---
#### [replaced 050] Organizing Unstructured Image Collections using Natural Language
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.05217v4](http://arxiv.org/pdf/2410.05217v4)**

> **作者:** Mingxuan Liu; Zhun Zhong; Jun Li; Gianni Franchi; Subhankar Roy; Elisa Ricci
>
> **备注:** Preprint. Project webpage: https://oatmealliu.github.io/opensmc.html
>
> **摘要:** Organizing unstructured image collections into semantic clusters is a long-standing challenge. Traditional deep clustering techniques address this by producing a single data partition, whereas multiple clustering methods uncover diverse alternative partitions-but only when users predefine the clustering criteria. Yet expecting users to specify such criteria a priori for large, unfamiliar datasets is unrealistic. In this work, we introduce the task of Open-ended Semantic Multiple Clustering (OpenSMC), which aims to automatically discover clustering criteria from large, unstructured image collections, revealing interpretable substructures without human input. Our framework, X-Cluster: eXploratory Clustering, treats text as a reasoning proxy: it concurrently scans the entire image collection, proposes candidate criteria in natural language, and groups images into meaningful clusters per criterion. To evaluate progress, we release COCO-4c and Food-4c benchmarks, each annotated with four grouping criteria. Experiments show that X-Cluster effectively reveals meaningful partitions and enables downstream applications such as bias discovery and social media image popularity analysis. We will open-source code and data to encourage reproducibility and further research.
>
---
#### [replaced 051] ART-DECO: Arbitrary Text Guidance for 3D Detailizer Construction
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20431v2](http://arxiv.org/pdf/2505.20431v2)**

> **作者:** Qimin Chen; Yuezhi Yang; Yifang Wang; Vladimir G. Kim; Siddhartha Chaudhuri; Hao Zhang; Zhiqin Chen
>
> **摘要:** We introduce a 3D detailizer, a neural model which can instantaneously (in <1s) transform a coarse 3D shape proxy into a high-quality asset with detailed geometry and texture as guided by an input text prompt. Our model is trained using the text prompt, which defines the shape class and characterizes the appearance and fine-grained style of the generated details. The coarse 3D proxy, which can be easily varied and adjusted (e.g., via user editing), provides structure control over the final shape. Importantly, our detailizer is not optimized for a single shape; it is the result of distilling a generative model, so that it can be reused, without retraining, to generate any number of shapes, with varied structures, whose local details all share a consistent style and appearance. Our detailizer training utilizes a pretrained multi-view image diffusion model, with text conditioning, to distill the foundational knowledge therein into our detailizer via Score Distillation Sampling (SDS). To improve SDS and enable our detailizer architecture to learn generalizable features over complex structures, we train our model in two training stages to generate shapes with increasing structural complexity. Through extensive experiments, we show that our method generates shapes of superior quality and details compared to existing text-to-3D models under varied structure control. Our detailizer can refine a coarse shape in less than a second, making it possible to interactively author and adjust 3D shapes. Furthermore, the user-imposed structure control can lead to creative, and hence out-of-distribution, 3D asset generations that are beyond the current capabilities of leading text-to-3D generative models. We demonstrate an interactive 3D modeling workflow our method enables, and its strong generalizability over styles, structures, and object categories.
>
---
#### [replaced 052] Harnessing PDF Data for Improving Japanese Large Multimodal Models
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14778v2](http://arxiv.org/pdf/2502.14778v2)**

> **作者:** Jeonghun Baek; Akiko Aizawa; Kiyoharu Aizawa
>
> **备注:** Accepted to ACL2025 Findings. Code: https://github.com/ku21fan/PDF-JLMM
>
> **摘要:** Large Multimodal Models (LMMs) have demonstrated strong performance in English, but their effectiveness in Japanese remains limited due to the lack of high-quality training data. Current Japanese LMMs often rely on translated English datasets, restricting their ability to capture Japan-specific cultural knowledge. To address this, we explore the potential of Japanese PDF data as a training resource, an area that remains largely underutilized. We introduce a fully automated pipeline that leverages pretrained models to extract image-text pairs from PDFs through layout analysis, OCR, and vision-language pairing, removing the need for manual annotation. Additionally, we construct instruction data from extracted image-text pairs to enrich the training data. To evaluate the effectiveness of PDF-derived data, we train Japanese LMMs and assess their performance on the Japanese LMM Benchmark. Our results demonstrate substantial improvements, with performance gains ranging from 2.1% to 13.8% on Heron-Bench. Further analysis highlights the impact of PDF-derived data on various factors, such as model size and language models, reinforcing its value as a multimodal resource for Japanese LMMs.
>
---
#### [replaced 053] Jigsaw-R1: A Study of Rule-based Visual Reinforcement Learning with Jigsaw Puzzles
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23590v2](http://arxiv.org/pdf/2505.23590v2)**

> **作者:** Zifu Wang; Junyi Zhu; Bo Tang; Zhiyu Li; Feiyu Xiong; Jiaqian Yu; Matthew B. Blaschko
>
> **摘要:** The application of rule-based reinforcement learning (RL) to multimodal large language models (MLLMs) introduces unique challenges and potential deviations from findings in text-only domains, particularly for perception-heavy tasks. This paper provides a comprehensive study of rule-based visual RL, using jigsaw puzzles as a structured experimental framework. Jigsaw puzzles offer inherent ground truth, adjustable difficulty, and demand complex decision-making, making them ideal for this study. Our research reveals several key findings: \textit{Firstly,} we find that MLLMs, initially performing near to random guessing on the simplest jigsaw puzzles, achieve near-perfect accuracy and generalize to complex, unseen configurations through fine-tuning. \textit{Secondly,} training on jigsaw puzzles can induce generalization to other visual tasks, with effectiveness tied to specific task configurations. \textit{Thirdly,} MLLMs can learn and generalize with or without explicit reasoning, though open-source models often favor direct answering. Consequently, even when trained for step-by-step reasoning, they can ignore the thinking process in deriving the final answer. \textit{Fourthly,} we observe that complex reasoning patterns appear to be pre-existing rather than emergent, with their frequency increasing alongside training and task difficulty. \textit{Finally,} our results demonstrate that RL exhibits more effective generalization than Supervised Fine-Tuning (SFT), and an initial SFT cold start phase can hinder subsequent RL optimization. Although these observations are based on jigsaw puzzles and may vary across other visual tasks, this research contributes a valuable piece of jigsaw to the larger puzzle of collective understanding rule-based visual RL and its potential in multimodal learning. The code is available at: https://github.com/zifuwanggg/Jigsaw-R1.
>
---
#### [replaced 054] Conditional Image Synthesis with Diffusion Models: A Survey
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.19365v3](http://arxiv.org/pdf/2409.19365v3)**

> **作者:** Zheyuan Zhan; Defang Chen; Jian-Ping Mei; Zhenghe Zhao; Jiawei Chen; Chun Chen; Siwei Lyu; Can Wang
>
> **摘要:** Conditional image synthesis based on user-specified requirements is a key component in creating complex visual content. In recent years, diffusion-based generative modeling has become a highly effective way for conditional image synthesis, leading to exponential growth in the literature. However, the complexity of diffusion-based modeling, the wide range of image synthesis tasks, and the diversity of conditioning mechanisms present significant challenges for researchers to keep up with rapid developments and to understand the core concepts on this topic. In this survey, we categorize existing works based on how conditions are integrated into the two fundamental components of diffusion-based modeling, $\textit{i.e.}$, the denoising network and the sampling process. We specifically highlight the underlying principles, advantages, and potential challenges of various conditioning approaches during the training, re-purposing, and specialization stages to construct a desired denoising network. We also summarize six mainstream conditioning mechanisms in the sampling process. All discussions are centered around popular applications. Finally, we pinpoint several critical yet still unsolved problems and suggest some possible solutions for future research. Our reviewed works are itemized at https://github.com/zju-pi/Awesome-Conditional-Diffusion-Models.
>
---
#### [replaced 055] AVadCLIP: Audio-Visual Collaboration for Robust Video Anomaly Detection
- **分类: cs.CV; I.4.9; I.5.4**

- **链接: [http://arxiv.org/pdf/2504.04495v2](http://arxiv.org/pdf/2504.04495v2)**

> **作者:** Peng Wu; Wanshun Su; Guansong Pang; Yujia Sun; Qingsen Yan; Peng Wang; Yanning Zhang
>
> **备注:** 12 pages, 6 figures, 9 tables. This work has been submitted to the IEEE for possible publication
>
> **摘要:** With the increasing adoption of video anomaly detection in intelligent surveillance domains, conventional visual-based detection approaches often struggle with information insufficiency and high false-positive rates in complex environments. To address these limitations, we present a novel weakly supervised framework that leverages audio-visual collaboration for robust video anomaly detection. Capitalizing on the exceptional cross-modal representation learning capabilities of Contrastive Language-Image Pretraining (CLIP) across visual, audio, and textual domains, our framework introduces two major innovations: an efficient audio-visual fusion that enables adaptive cross-modal integration through lightweight parametric adaptation while maintaining the frozen CLIP backbone, and a novel audio-visual prompt that dynamically enhances text embeddings with key multimodal information based on the semantic correlation between audio-visual features and textual labels, significantly improving CLIP's generalization for the video anomaly detection task. Moreover, to enhance robustness against modality deficiency during inference, we further develop an uncertainty-driven feature distillation module that synthesizes audio-visual representations from visual-only inputs. This module employs uncertainty modeling based on the diversity of audio-visual features to dynamically emphasize challenging features during the distillation process. Our framework demonstrates superior performance across multiple benchmarks, with audio integration significantly boosting anomaly detection accuracy in various scenarios. Notably, with unimodal data enhanced by uncertainty-driven distillation, our approach consistently outperforms current unimodal VAD methods.
>
---
#### [replaced 056] InfoChartQA: A Benchmark for Multimodal Question Answering on Infographic Charts
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19028v2](http://arxiv.org/pdf/2505.19028v2)**

> **作者:** Minzhi Lin; Tianchi Xie; Mengchen Liu; Yilin Ye; Changjian Chen; Shixia Liu
>
> **摘要:** Understanding infographic charts with design-driven visual elements (e.g., pictograms, icons) requires both visual recognition and reasoning, posing challenges for multimodal large language models (MLLMs). However, existing visual-question answering benchmarks fall short in evaluating these capabilities of MLLMs due to the lack of paired plain charts and visual-element-based questions. To bridge this gap, we introduce InfoChartQA, a benchmark for evaluating MLLMs on infographic chart understanding. It includes 5,642 pairs of infographic and plain charts, each sharing the same underlying data but differing in visual presentations. We further design visual-element-based questions to capture their unique visual designs and communicative intent. Evaluation of 20 MLLMs reveals a substantial performance decline on infographic charts, particularly for visual-element-based questions related to metaphors. The paired infographic and plain charts enable fine-grained error analysis and ablation studies, which highlight new opportunities for advancing MLLMs in infographic chart understanding. We release InfoChartQA at https://github.com/CoolDawnAnt/InfoChartQA.
>
---
#### [replaced 057] Efficient Open Set Single Image Test Time Adaptation of Vision Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.00481v2](http://arxiv.org/pdf/2406.00481v2)**

> **作者:** Manogna Sreenivas; Soma Biswas
>
> **备注:** Accepted at TMLR
>
> **摘要:** Adapting models to dynamic, real-world environments characterized by shifting data distributions and unseen test scenarios is a critical challenge in deep learning. In this paper, we consider a realistic and challenging Test-Time Adaptation setting, where a model must continuously adapt to test samples that arrive sequentially, one at a time, while distinguishing between known and unknown classes. Current Test-Time Adaptation methods operate under closed-set assumptions or batch processing, differing from the real-world open-set scenarios. We address this limitation by establishing a comprehensive benchmark for {\em Open-set Single-image Test-Time Adaptation using Vision-Language Models}. Furthermore, we propose ROSITA, a novel framework that leverages dynamically updated feature banks to identify reliable test samples and employs a contrastive learning objective to improve the separation between known and unknown classes. Our approach effectively adapts models to domain shifts for known classes while rejecting unfamiliar samples. Extensive experiments across diverse real-world benchmarks demonstrate that ROSITA sets a new state-of-the-art in open-set TTA, achieving both strong performance and computational efficiency for real-time deployment. Our code can be found at the project site https://manogna-s.github.io/rosita/
>
---
#### [replaced 058] MFCLIP: Multi-modal Fine-grained CLIP for Generalizable Diffusion Face Forgery Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.09724v3](http://arxiv.org/pdf/2409.09724v3)**

> **作者:** Yaning Zhang; Tianyi Wang; Zitong Yu; Zan Gao; Linlin Shen; Shengyong Chen
>
> **备注:** Accepted by IEEE Transactions on Information Forensics and Security 2025
>
> **摘要:** The rapid development of photo-realistic face generation methods has raised significant concerns in society and academia, highlighting the urgent need for robust and generalizable face forgery detection (FFD) techniques. Although existing approaches mainly capture face forgery patterns using image modality, other modalities like fine-grained noises and texts are not fully explored, which limits the generalization capability of the model. In addition, most FFD methods tend to identify facial images generated by GAN, but struggle to detect unseen diffusion-synthesized ones. To address the limitations, we aim to leverage the cutting-edge foundation model, contrastive language-image pre-training (CLIP), to achieve generalizable diffusion face forgery detection (DFFD). In this paper, we propose a novel multi-modal fine-grained CLIP (MFCLIP) model, which mines comprehensive and fine-grained forgery traces across image-noise modalities via language-guided face forgery representation learning, to facilitate the advancement of DFFD. Specifically, we devise a fine-grained language encoder (FLE) that extracts fine global language features from hierarchical text prompts. We design a multi-modal vision encoder (MVE) to capture global image forgery embeddings as well as fine-grained noise forgery patterns extracted from the richest patch, and integrate them to mine general visual forgery traces. Moreover, we build an innovative plug-and-play sample pair attention (SPA) method to emphasize relevant negative pairs and suppress irrelevant ones, allowing cross-modality sample pairs to conduct more flexible alignment. Extensive experiments and visualizations show that our model outperforms the state of the arts on different settings like cross-generator, cross-forgery, and cross-dataset evaluations.
>
---
#### [replaced 059] VARD: Efficient and Dense Fine-Tuning for Diffusion Models with Value-based RL
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15791v2](http://arxiv.org/pdf/2505.15791v2)**

> **作者:** Fengyuan Dai; Zifeng Zhuang; Yufei Huang; Siteng Huang; Bangyan Liao; Donglin Wang; Fajie Yuan
>
> **备注:** Under review
>
> **摘要:** Diffusion models have emerged as powerful generative tools across various domains, yet tailoring pre-trained models to exhibit specific desirable properties remains challenging. While reinforcement learning (RL) offers a promising solution,current methods struggle to simultaneously achieve stable, efficient fine-tuning and support non-differentiable rewards. Furthermore, their reliance on sparse rewards provides inadequate supervision during intermediate steps, often resulting in suboptimal generation quality. To address these limitations, dense and differentiable signals are required throughout the diffusion process. Hence, we propose VAlue-based Reinforced Diffusion (VARD): a novel approach that first learns a value function predicting expection of rewards from intermediate states, and subsequently uses this value function with KL regularization to provide dense supervision throughout the generation process. Our method maintains proximity to the pretrained model while enabling effective and stable training via backpropagation. Experimental results demonstrate that our approach facilitates better trajectory guidance, improves training efficiency and extends the applicability of RL to diffusion models optimized for complex, non-differentiable reward functions.
>
---
#### [replaced 060] HandCraft: Anatomically Correct Restoration of Malformed Hands in Diffusion Generated Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.04332v3](http://arxiv.org/pdf/2411.04332v3)**

> **作者:** Zhenyue Qin; Yiqun Zhang; Yang Liu; Dylan Campbell
>
> **备注:** 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
>
> **摘要:** Generative text-to-image models, such as Stable Diffusion, have demonstrated a remarkable ability to generate diverse, high-quality images. However, they are surprisingly inept when it comes to rendering human hands, which are often anatomically incorrect or reside in the "uncanny valley". In this paper, we propose a method HandCraft for restoring such malformed hands. This is achieved by automatically constructing masks and depth images for hands as conditioning signals using a parametric model, allowing a diffusion-based image editor to fix the hand's anatomy and adjust its pose while seamlessly integrating the changes into the original image, preserving pose, color, and style. Our plug-and-play hand restoration solution is compatible with existing pretrained diffusion models, and the restoration process facilitates adoption by eschewing any fine-tuning or training requirements for the diffusion models. We also contribute MalHand datasets that contain generated images with a wide variety of malformed hands in several styles for hand detector training and hand restoration benchmarking, and demonstrate through qualitative and quantitative evaluation that HandCraft not only restores anatomical correctness but also maintains the integrity of the overall image.
>
---
#### [replaced 061] CLEAR: Character Unlearning in Textual and Visual Modalities
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.18057v4](http://arxiv.org/pdf/2410.18057v4)**

> **作者:** Alexey Dontsov; Dmitrii Korzh; Alexey Zhavoronkin; Boris Mikheev; Denis Bobkov; Aibek Alanov; Oleg Y. Rogov; Ivan Oseledets; Elena Tutubalina
>
> **摘要:** Machine Unlearning (MU) is critical for removing private or hazardous information from deep learning models. While MU has advanced significantly in unimodal (text or vision) settings, multimodal unlearning (MMU) remains underexplored due to the lack of open benchmarks for evaluating cross-modal data removal. To address this gap, we introduce CLEAR, the first open-source benchmark designed specifically for MMU. CLEAR contains 200 fictitious individuals and 3,700 images linked with corresponding question-answer pairs, enabling a thorough evaluation across modalities. We conduct a comprehensive analysis of 11 MU methods (e.g., SCRUB, gradient ascent, DPO) across four evaluation sets, demonstrating that jointly unlearning both modalities outperforms single-modality approaches. The dataset is available at https://huggingface.co/datasets/therem/CLEAR
>
---
#### [replaced 062] I see what you mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.00071v2](http://arxiv.org/pdf/2503.00071v2)**

> **作者:** Esam Ghaleb; Bulat Khaertdinov; Aslı Özyürek; Raquel Fernández
>
> **摘要:** In face-to-face interaction, we use multiple modalities, including speech and gestures, to communicate information and resolve references to objects. However, how representational co-speech gestures refer to objects remains understudied from a computational perspective. In this work, we address this gap by introducing a multimodal reference resolution task centred on representational gestures, while simultaneously tackling the challenge of learning robust gesture embeddings. We propose a self-supervised pre-training approach to gesture representation learning that grounds body movements in spoken language. Our experiments show that the learned embeddings align with expert annotations and have significant predictive power. Moreover, reference resolution accuracy further improves when (1) using multimodal gesture representations, even when speech is unavailable at inference time, and (2) leveraging dialogue history. Overall, our findings highlight the complementary roles of gesture and speech in reference resolution, offering a step towards more naturalistic models of human-machine interaction.
>
---
#### [replaced 063] TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18263v4](http://arxiv.org/pdf/2411.18263v4)**

> **作者:** Linwei Dong; Qingnan Fan; Yihong Guo; Zhonghao Wang; Qi Zhang; Jinwei Chen; Yawei Luo; Changqing Zou
>
> **摘要:** Pre-trained text-to-image diffusion models are increasingly applied to real-world image super-resolution (Real-ISR) task. Given the iterative refinement nature of diffusion models, most existing approaches are computationally expensive. While methods such as SinSR and OSEDiff have emerged to condense inference steps via distillation, their performance in image restoration or details recovery is not satisfied. To address this, we propose TSD-SR, a novel distillation framework specifically designed for real-world image super-resolution, aiming to construct an efficient and effective one-step model. We first introduce the Target Score Distillation, which leverages the priors of diffusion models and real image references to achieve more realistic image restoration. Secondly, we propose a Distribution-Aware Sampling Module to make detail-oriented gradients more readily accessible, addressing the challenge of recovering fine details. Extensive experiments demonstrate that our TSD-SR has superior restoration results (most of the metrics perform the best) and the fastest inference speed (e.g. 40 times faster than SeeSR) compared to the past Real-ISR approaches based on pre-trained diffusion priors.
>
---
#### [replaced 064] Translation Consistent Semi-supervised Segmentation for 3D Medical Images
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2203.14523v3](http://arxiv.org/pdf/2203.14523v3)**

> **作者:** Yuyuan Liu; Yu Tian; Chong Wang; Yuanhong Chen; Fengbei Liu; Vasileios Belagiannis; Gustavo Carneiro
>
> **备注:** 17 pages, 10 Figures and 6 Tables
>
> **摘要:** 3D medical image segmentation methods have been successful, but their dependence on large amounts of voxel-level annotated data is a disadvantage that needs to be addressed given the high cost to obtain such annotation. Semi-supervised learning (SSL) solve this issue by training models with a large unlabelled and a small labelled dataset. The most successful SSL approaches are based on consistency learning that minimises the distance between model responses obtained from perturbed views of the unlabelled data. These perturbations usually keep the spatial input context between views fairly consistent, which may cause the model to learn segmentation patterns from the spatial input contexts instead of the segmented objects. In this paper, we introduce the Translation Consistent Co-training (TraCoCo) which is a consistency learning SSL method that perturbs the input data views by varying their spatial input context, allowing the model to learn segmentation patterns from visual objects. Furthermore, we propose the replacement of the commonly used mean squared error (MSE) semi-supervised loss by a new Cross-model confident Binary Cross entropy (CBC) loss, which improves training convergence and keeps the robustness to co-training pseudo-labelling mistakes. We also extend CutMix augmentation to 3D SSL to further improve generalisation. Our TraCoCo shows state-of-the-art results for the Left Atrium (LA) and Brain Tumor Segmentation (BRaTS19) datasets with different backbones. Our code is available at https://github.com/yyliu01/TraCoCo.
>
---
#### [replaced 065] Marine Saliency Segmenter: Object-Focused Conditional Diffusion with Region-Level Semantic Knowledge Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02391v2](http://arxiv.org/pdf/2504.02391v2)**

> **作者:** Laibin Chang; Yunke Wang; JiaXing Huang; Longxiang Deng; Bo Du; Chang Xu
>
> **摘要:** Marine Saliency Segmentation (MSS) plays a pivotal role in various vision-based marine exploration tasks. However, existing marine segmentation techniques face the dilemma of object mislocalization and imprecise boundaries due to the complex underwater environment. Meanwhile, despite the impressive performance of diffusion models in visual segmentation, there remains potential to further leverage contextual semantics to enhance feature learning of region-level salient objects, thereby improving segmentation outcomes. Building on this insight, we propose DiffMSS, a novel marine saliency segmenter based on the diffusion model, which utilizes semantic knowledge distillation to guide the segmentation of marine salient objects. Specifically, we design a region-word similarity matching mechanism to identify salient terms at the word level from the text descriptions. These high-level semantic features guide the conditional feature learning network in generating salient and accurate diffusion conditions with semantic knowledge distillation. To further refine the segmentation of fine-grained structures in unique marine organisms, we develop the dedicated consensus deterministic sampling to suppress overconfident missegmentations. Comprehensive experiments demonstrate the superior performance of DiffMSS over state-of-the-art methods in both quantitative and qualitative evaluations.
>
---
#### [replaced 066] Unveiling the Lack of LVLM Robustness to Fundamental Visual Variations: Why and Path Forward
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.16727v3](http://arxiv.org/pdf/2504.16727v3)**

> **作者:** Zhiyuan Fan; Yumeng Wang; Sandeep Polisetty; Yi R. Fung
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Large Vision Language Models (LVLMs) excel in various vision-language tasks. Yet, their robustness to visual variations in position, scale, orientation, and context that objects in natural scenes inevitably exhibit due to changes in viewpoint and environment remains largely underexplored. To bridge this gap, we introduce V$^2$R-Bench, a comprehensive benchmark framework for evaluating Visual Variation Robustness of LVLMs, which encompasses automated evaluation dataset generation and principled metrics for thorough robustness assessment. Through extensive evaluation on 21 LVLMs, we reveal a surprising vulnerability to visual variations, in which even advanced models that excel at complex vision-language tasks significantly underperform on simple tasks such as object recognition. Interestingly, these models exhibit a distinct visual position bias that contradicts theories of effective receptive fields, and demonstrate a human-like visual acuity threshold. To identify the source of these vulnerabilities, we present a systematic framework for component-level analysis, featuring a novel visualization approach for aligned visual features. Results show that these vulnerabilities stem from error accumulation in the pipeline architecture and inadequate multimodal alignment. Complementary experiments with synthetic data further demonstrate that these limitations are fundamentally architectural deficiencies, scoring the need for architectural innovations in future LVLM designs.
>
---
#### [replaced 067] Accurate Differential Operators for Hybrid Neural Fields
- **分类: cs.CV; cs.AI; cs.GR; cs.LG**

- **链接: [http://arxiv.org/pdf/2312.05984v2](http://arxiv.org/pdf/2312.05984v2)**

> **作者:** Aditya Chetan; Guandao Yang; Zichen Wang; Steve Marschner; Bharath Hariharan
>
> **备注:** Accepted in CVPR 2025. Project page is available at https://justachetan.github.io/hnf-derivatives/
>
> **摘要:** Neural fields have become widely used in various fields, from shape representation to neural rendering, and for solving partial differential equations (PDEs). With the advent of hybrid neural field representations like Instant NGP that leverage small MLPs and explicit representations, these models train quickly and can fit large scenes. Yet in many applications like rendering and simulation, hybrid neural fields can cause noticeable and unreasonable artifacts. This is because they do not yield accurate spatial derivatives needed for these downstream applications. In this work, we propose two ways to circumvent these challenges. Our first approach is a post hoc operator that uses local polynomial fitting to obtain more accurate derivatives from pre-trained hybrid neural fields. Additionally, we also propose a self-supervised fine-tuning approach that refines the hybrid neural field to yield accurate derivatives directly while preserving the initial signal. We show applications of our method to rendering, collision simulation, and solving PDEs. We observe that using our approach yields more accurate derivatives, reducing artifacts and leading to more accurate simulations in downstream applications.
>
---
#### [replaced 068] Autoregressive Models in Vision: A Survey
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.05902v2](http://arxiv.org/pdf/2411.05902v2)**

> **作者:** Jing Xiong; Gongye Liu; Lun Huang; Chengyue Wu; Taiqiang Wu; Yao Mu; Yuan Yao; Hui Shen; Zhongwei Wan; Jinfa Huang; Chaofan Tao; Shen Yan; Huaxiu Yao; Lingpeng Kong; Hongxia Yang; Mi Zhang; Guillermo Sapiro; Jiebo Luo; Ping Luo; Ngai Wong
>
> **备注:** The paper is accepted by TMLR
>
> **摘要:** Autoregressive modeling has been a huge success in the field of natural language processing (NLP). Recently, autoregressive models have emerged as a significant area of focus in computer vision, where they excel in producing high-quality visual content. Autoregressive models in NLP typically operate on subword tokens. However, the representation strategy in computer vision can vary in different levels, i.e., pixel-level, token-level, or scale-level, reflecting the diverse and hierarchical nature of visual data compared to the sequential structure of language. This survey comprehensively examines the literature on autoregressive models applied to vision. To improve readability for researchers from diverse research backgrounds, we start with preliminary sequence representation and modeling in vision. Next, we divide the fundamental frameworks of visual autoregressive models into three general sub-categories, including pixel-based, token-based, and scale-based models based on the representation strategy. We then explore the interconnections between autoregressive models and other generative models. Furthermore, we present a multifaceted categorization of autoregressive models in computer vision, including image generation, video generation, 3D generation, and multimodal generation. We also elaborate on their applications in diverse domains, including emerging domains such as embodied AI and 3D medical AI, with about 250 related references. Finally, we highlight the current challenges to autoregressive models in vision with suggestions about potential research directions. We have also set up a Github repository to organize the papers included in this survey at: https://github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey.
>
---
#### [replaced 069] RAFT: Robust Augmentation of FeaTures for Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04529v2](http://arxiv.org/pdf/2505.04529v2)**

> **作者:** Edward Humes; Xiaomin Lin; Uttej Kallakuri; Tinoosh Mohsenin
>
> **摘要:** Image segmentation is a powerful computer vision technique for scene understanding. However, real-world deployment is stymied by the need for high-quality, meticulously labeled datasets. Synthetic data provides high-quality labels while reducing the need for manual data collection and annotation. However, deep neural networks trained on synthetic data often face the Syn2Real problem, leading to poor performance in real-world deployments. To mitigate the aforementioned gap in image segmentation, we propose RAFT, a novel framework for adapting image segmentation models using minimal labeled real-world data through data and feature augmentations, as well as active learning. To validate RAFT, we perform experiments on the synthetic-to-real "SYNTHIA->Cityscapes" and "GTAV->Cityscapes" benchmarks. We managed to surpass the previous state of the art, HALO. SYNTHIA->Cityscapes experiences an improvement in mIoU* upon domain adaptation of 2.1%/79.9%, and GTAV->Cityscapes experiences a 0.4%/78.2% improvement in mIoU. Furthermore, we test our approach on the real-to-real benchmark of "Cityscapes->ACDC", and again surpass HALO, with a gain in mIoU upon adaptation of 1.3%/73.2%. Finally, we examine the effect of the allocated annotation budget and various components of RAFT upon the final transfer mIoU.
>
---
#### [replaced 070] IMPROVE: Iterative Model Pipeline Refinement and Optimization Leveraging LLM Experts
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.18530v2](http://arxiv.org/pdf/2502.18530v2)**

> **作者:** Eric Xue; Ke Chen; Zeyi Huang; Yuyang Ji; Yong Jae Lee; Haohan Wang
>
> **摘要:** Large language model (LLM) agents have emerged as a promising solution to automate the workflow of machine learning, but most existing methods share a common limitation: they attempt to optimize entire pipelines in a single step before evaluation, making it difficult to attribute improvements to specific changes. This lack of granularity leads to unstable optimization and slower convergence, limiting their effectiveness. To address this, we introduce Iterative Refinement, a novel strategy for LLM-driven ML pipeline design inspired by how human ML experts iteratively refine models, focusing on one component at a time rather than making sweeping changes all at once. By systematically updating individual components based on real training feedback, Iterative Refinement improves overall model performance. We also provide some theoretical edvience of the superior properties of this Iterative Refinement. Further, we implement this strategy in IMPROVE, an end-to-end LLM agent framework for automating and optimizing object classification pipelines. Through extensive evaluations across datasets of varying sizes and domains, we demonstrate that Iterative Refinement enables IMPROVE to consistently achieve better performance over existing zero-shot LLM-based approaches.
>
---
#### [replaced 071] Fact-Checking of AI-Generated Reports
- **分类: cs.AI; cs.CR; cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2307.14634v2](http://arxiv.org/pdf/2307.14634v2)**

> **作者:** Razi Mahmood; Diego Machado Reyes; Ge Wang; Mannudeep Kalra; Pingkun Yan
>
> **备注:** 10 pages, 3 figures, 3 tables
>
> **摘要:** With advances in generative artificial intelligence (AI), it is now possible to produce realistic-looking automated reports for preliminary reads of radiology images. This can expedite clinical workflows, improve accuracy and reduce overall costs. However, it is also well-known that such models often hallucinate, leading to false findings in the generated reports. In this paper, we propose a new method of fact-checking of AI-generated reports using their associated images. Specifically, the developed examiner differentiates real and fake sentences in reports by learning the association between an image and sentences describing real or potentially fake findings. To train such an examiner, we first created a new dataset of fake reports by perturbing the findings in the original ground truth radiology reports associated with images. Text encodings of real and fake sentences drawn from these reports are then paired with image encodings to learn the mapping to real/fake labels. The utility of such an examiner is demonstrated for verifying automatically generated reports by detecting and removing fake sentences. Future generative AI approaches can use the resulting tool to validate their reports leading to a more responsible use of AI in expediting clinical workflows.
>
---
#### [replaced 072] Stochastic Layer-Wise Shuffle for Improving Vision Mamba Training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.17081v2](http://arxiv.org/pdf/2408.17081v2)**

> **作者:** Zizheng Huang; Haoxing Chen; Jiaqi Li; Jun Lan; Huijia Zhu; Weiqiang Wang; Limin Wang
>
> **备注:** accpeted to ICML25
>
> **摘要:** Recent Vision Mamba (Vim) models exhibit nearly linear complexity in sequence length, making them highly attractive for processing visual data. However, the training methodologies and their potential are still not sufficiently explored. In this paper, we investigate strategies for Vim and propose Stochastic Layer-Wise Shuffle (SLWS), a novel regularization method that can effectively improve the Vim training. Without architectural modifications, this approach enables the non-hierarchical Vim to get leading performance on ImageNet-1K compared with the similar type counterparts. Our method operates through four simple steps per layer: probability allocation to assign layer-dependent shuffle rates, operation sampling via Bernoulli trials, sequence shuffling of input tokens, and order restoration of outputs. SLWS distinguishes itself through three principles: \textit{(1) Plug-and-play:} No architectural modifications are needed, and it is deactivated during inference. \textit{(2) Simple but effective:} The four-step process introduces only random permutations and negligible overhead. \textit{(3) Intuitive design:} Shuffling probabilities grow linearly with layer depth, aligning with the hierarchical semantic abstraction in vision models. Our work underscores the importance of tailored training strategies for Vim models and provides a helpful way to explore their scalability.
>
---
#### [replaced 073] Chain-of-Talkers (CoTalk): Fast Human Annotation of Dense Image Captions
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22627v2](http://arxiv.org/pdf/2505.22627v2)**

> **作者:** Yijun Shen; Delong Chen; Fan Liu; Xingyu Wang; Chuanyi Zhang; Liang Yao; Yuhui Zheng
>
> **摘要:** While densely annotated image captions significantly facilitate the learning of robust vision-language alignment, methodologies for systematically optimizing human annotation efforts remain underexplored. We introduce Chain-of-Talkers (CoTalk), an AI-in-the-loop methodology designed to maximize the number of annotated samples and improve their comprehensiveness under fixed budget constraints (e.g., total human annotation time). The framework is built upon two key insights. First, sequential annotation reduces redundant workload compared to conventional parallel annotation, as subsequent annotators only need to annotate the ``residual'' -- the missing visual information that previous annotations have not covered. Second, humans process textual input faster by reading while outputting annotations with much higher throughput via talking; thus a multimodal interface enables optimized efficiency. We evaluate our framework from two aspects: intrinsic evaluations that assess the comprehensiveness of semantic units, obtained by parsing detailed captions into object-attribute trees and analyzing their effective connections; extrinsic evaluation measures the practical usage of the annotated captions in facilitating vision-language alignment. Experiments with eight participants show our Chain-of-Talkers (CoTalk) improves annotation speed (0.42 vs. 0.30 units/sec) and retrieval performance (41.13% vs. 40.52%) over the parallel method.
>
---
#### [replaced 074] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.07093v2](http://arxiv.org/pdf/2504.07093v2)**

> **作者:** Gene Chou; Wenqi Xian; Guandao Yang; Mohamed Abdelfattah; Bharath Hariharan; Noah Snavely; Ning Yu; Paul Debevec
>
> **摘要:** A versatile video depth estimation model should (1) be accurate and consistent across frames, (2) produce high-resolution depth maps, and (3) support real-time streaming. We propose FlashDepth, a method that satisfies all three requirements, performing depth estimation on a 2044x1148 streaming video at 24 FPS. We show that, with careful modifications to pretrained single-image depth models, these capabilities are enabled with relatively little data and training. We evaluate our approach across multiple unseen datasets against state-of-the-art depth models, and find that ours outperforms them in terms of boundary sharpness and speed by a significant margin, while maintaining competitive accuracy. We hope our model will enable various applications that require high-resolution depth, such as video editing, and online decision-making, such as robotics. We release all code and model weights at https://github.com/Eyeline-Research/FlashDepth
>
---
#### [replaced 075] MorphoSeg: An Uncertainty-Aware Deep Learning Method for Biomedical Segmentation of Complex Cellular Morphologies
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.17110v2](http://arxiv.org/pdf/2409.17110v2)**

> **作者:** Tianhao Zhang; Heather J. McCourty; Berardo M. Sanchez-Tafolla; Anton Nikolaev; Lyudmila S. Mihaylova
>
> **摘要:** Deep learning has revolutionized medical and biological imaging, particularly in segmentation tasks. However, segmenting biological cells remains challenging due to the high variability and complexity of cell shapes. Addressing this challenge requires high-quality datasets that accurately represent the diverse morphologies found in biological cells. Existing cell segmentation datasets are often limited by their focus on regular and uniform shapes. In this paper, we introduce a novel benchmark dataset of Ntera-2 (NT2) cells, a pluripotent carcinoma cell line, exhibiting diverse morphologies across multiple stages of differentiation, capturing the intricate and heterogeneous cellular structures that complicate segmentation tasks. To address these challenges, we propose an uncertainty-aware deep learning framework for complex cellular morphology segmentation (MorphoSeg) by incorporating sampling of virtual outliers from low-likelihood regions during training. Our comprehensive experimental evaluations against state-of-the-art baselines demonstrate that MorphoSeg significantly enhances segmentation accuracy, achieving up to a 7.74% increase in the Dice Similarity Coefficient (DSC) and a 28.36% reduction in the Hausdorff Distance. These findings highlight the effectiveness of our dataset and methodology in advancing cell segmentation capabilities, especially for complex and variable cell morphologies. The dataset and source code is publicly available at https://github.com/RanchoGoose/MorphoSeg.
>
---
#### [replaced 076] MDMP: Multi-modal Diffusion for supervised Motion Predictions with uncertainty
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.03860v2](http://arxiv.org/pdf/2410.03860v2)**

> **作者:** Leo Bringer; Joey Wilson; Kira Barton; Maani Ghaffari
>
> **备注:** Accepted to CVPR 2025 - HuMoGen. Minor revisions made based on reviewer feedback
>
> **摘要:** This paper introduces a Multi-modal Diffusion model for Motion Prediction (MDMP) that integrates and synchronizes skeletal data and textual descriptions of actions to generate refined long-term motion predictions with quantifiable uncertainty. Existing methods for motion forecasting or motion generation rely solely on either prior motions or text prompts, facing limitations with precision or control, particularly over extended durations. The multi-modal nature of our approach enhances the contextual understanding of human motion, while our graph-based transformer framework effectively capture both spatial and temporal motion dynamics. As a result, our model consistently outperforms existing generative techniques in accurately predicting long-term motions. Additionally, by leveraging diffusion models' ability to capture different modes of prediction, we estimate uncertainty, significantly improving spatial awareness in human-robot interactions by incorporating zones of presence with varying confidence levels for each body joint.
>
---
#### [replaced 077] Alignment is All You Need: A Training-free Augmentation Strategy for Pose-guided Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.16506v2](http://arxiv.org/pdf/2408.16506v2)**

> **作者:** Xiaoyu Jin; Zunnan Xu; Mingwen Ou; Wenming Yang
>
> **备注:** Accepted to CVG@ICML 2024
>
> **摘要:** Character animation is a transformative field in computer graphics and vision, enabling dynamic and realistic video animations from static images. Despite advancements, maintaining appearance consistency in animations remains a challenge. Our approach addresses this by introducing a training-free framework that ensures the generated video sequence preserves the reference image's subtleties, such as physique and proportions, through a dual alignment strategy. We decouple skeletal and motion priors from pose information, enabling precise control over animation generation. Our method also improves pixel-level alignment for conditional control from the reference character, enhancing the temporal consistency and visual cohesion of animations. Our method significantly enhances the quality of video generation without the need for large datasets or expensive computational resources.
>
---
#### [replaced 078] StarVector: Generating Scalable Vector Graphics Code from Images and Text
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2312.11556v4](http://arxiv.org/pdf/2312.11556v4)**

> **作者:** Juan A. Rodriguez; Abhay Puri; Shubham Agarwal; Issam H. Laradji; Pau Rodriguez; Sai Rajeswar; David Vazquez; Christopher Pal; Marco Pedersoli
>
> **摘要:** Scalable Vector Graphics (SVGs) are vital for modern image rendering due to their scalability and versatility. Previous SVG generation methods have focused on curve-based vectorization, lacking semantic understanding, often producing artifacts, and struggling with SVG primitives beyond path curves. To address these issues, we introduce StarVector, a multimodal large language model for SVG generation. It performs image vectorization by understanding image semantics and using SVG primitives for compact, precise outputs. Unlike traditional methods, StarVector works directly in the SVG code space, leveraging visual understanding to apply accurate SVG primitives. To train StarVector, we create SVG-Stack, a diverse dataset of 2M samples that enables generalization across vectorization tasks and precise use of primitives like ellipses, polygons, and text. We address challenges in SVG evaluation, showing that pixel-based metrics like MSE fail to capture the unique qualities of vector graphics. We introduce SVG-Bench, a benchmark across 10 datasets, and 3 tasks: Image-to-SVG, Text-to-SVG generation, and diagram generation. Using this setup, StarVector achieves state-of-the-art performance, producing more compact and semantically rich SVGs.
>
---
#### [replaced 079] WaterSplatting: Fast Underwater 3D Scene Reconstruction Using Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.08206v2](http://arxiv.org/pdf/2408.08206v2)**

> **作者:** Huapeng Li; Wenxuan Song; Tianao Xu; Alexandre Elsig; Jonas Kulhanek
>
> **备注:** Web: https://water-splatting.github.io
>
> **摘要:** The underwater 3D scene reconstruction is a challenging, yet interesting problem with applications ranging from naval robots to VR experiences. The problem was successfully tackled by fully volumetric NeRF-based methods which can model both the geometry and the medium (water). Unfortunately, these methods are slow to train and do not offer real-time rendering. More recently, 3D Gaussian Splatting (3DGS) method offered a fast alternative to NeRFs. However, because it is an explicit method that renders only the geometry, it cannot render the medium and is therefore unsuited for underwater reconstruction. Therefore, we propose a novel approach that fuses volumetric rendering with 3DGS to handle underwater data effectively. Our method employs 3DGS for explicit geometry representation and a separate volumetric field (queried once per pixel) for capturing the scattering medium. This dual representation further allows the restoration of the scenes by removing the scattering medium. Our method outperforms state-of-the-art NeRF-based methods in rendering quality on the underwater SeaThru-NeRF dataset. Furthermore, it does so while offering real-time rendering performance, addressing the efficiency limitations of existing methods. Web: https://water-splatting.github.io
>
---
#### [replaced 080] Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.04343v2](http://arxiv.org/pdf/2406.04343v2)**

> **作者:** Stanislaw Szymanowicz; Eldar Insafutdinov; Chuanxia Zheng; Dylan Campbell; João F. Henriques; Christian Rupprecht; Andrea Vedaldi
>
> **备注:** Project page: https://www.robots.ox.ac.uk/~vgg/research/flash3d/
>
> **摘要:** We propose Flash3D, a method for scene reconstruction and novel view synthesis from a single image which is both very generalisable and efficient. For generalisability, we start from a "foundation" model for monocular depth estimation and extend it to a full 3D shape and appearance reconstructor. For efficiency, we base this extension on feed-forward Gaussian Splatting. Specifically, we predict a first layer of 3D Gaussians at the predicted depth, and then add additional layers of Gaussians that are offset in space, allowing the model to complete the reconstruction behind occlusions and truncations. Flash3D is very efficient, trainable on a single GPU in a day, and thus accessible to most researchers. It achieves state-of-the-art results when trained and tested on RealEstate10k. When transferred to unseen datasets like NYU it outperforms competitors by a large margin. More impressively, when transferred to KITTI, Flash3D achieves better PSNR than methods trained specifically on that dataset. In some instances, it even outperforms recent methods that use multiple views as input. Code, models, demo, and more results are available at https://www.robots.ox.ac.uk/~vgg/research/flash3d/.
>
---
#### [replaced 081] Generalizing from SIMPLE to HARD Visual Reasoning: Can We Mitigate Modality Imbalance in VLMs?
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.02669v2](http://arxiv.org/pdf/2501.02669v2)**

> **作者:** Simon Park; Abhishek Panigrahi; Yun Cheng; Dingli Yu; Anirudh Goyal; Sanjeev Arora
>
> **摘要:** Vision Language Models (VLMs) are impressive at visual question answering and image captioning. But they underperform on multi-step visual reasoning -- even compared to LLMs on the same tasks presented in text form -- giving rise to perceptions of modality imbalance or brittleness. Towards a systematic study of such issues, we introduce a synthetic framework for assessing the ability of VLMs to perform algorithmic visual reasoning, comprising three tasks: Table Readout, Grid Navigation, and Visual Analogy. Each has two levels of difficulty, SIMPLE and HARD, and even the SIMPLE versions are difficult for frontier VLMs. We propose strategies for training on the SIMPLE version of tasks that improve performance on the corresponding HARD task, i.e., simple-to-hard (S2H) generalization. This controlled setup, where each task also has an equivalent text-only version, allows a quantification of the modality imbalance and how it is impacted by training strategy. We show that 1) explicit image-to-text conversion is important in promoting S2H generalization on images, by transferring reasoning from text; 2) conversion can be internalized at test time. We also report results of mechanistic study of this phenomenon. We identify measures of gradient alignment that can identify training strategies that promote better S2H generalization. Ablations highlight the importance of chain-of-thought.
>
---
#### [replaced 082] DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06029v3](http://arxiv.org/pdf/2502.06029v3)**

> **作者:** Krishna Sri Ipsit Mantri; Carola-Bibiane Schönlieb; Bruno Ribeiro; Chaim Baskin; Moshe Eliasof
>
> **备注:** CVPR 2025, 14 pages
>
> **摘要:** Pre-trained Vision Transformers now serve as powerful tools for computer vision. Yet, efficiently adapting them for multiple tasks remains a challenge that arises from the need to modify the rich hidden representations encoded by the learned weight matrices, without inducing interference between tasks. Current parameter-efficient methods like LoRA, which apply low-rank updates, force tasks to compete within constrained subspaces, ultimately degrading performance. We introduce DiTASK a novel Diffeomorphic Multi-Task Fine-Tuning approach that maintains pre-trained representations by preserving weight matrix singular vectors, while enabling task-specific adaptations through neural diffeomorphic transformations of the singular values. By following this approach, DiTASK enables both shared and task-specific feature modulations with minimal added parameters. Our theoretical analysis shows that DITASK achieves full-rank updates during optimization, preserving the geometric structure of pre-trained features, and establishing a new paradigm for efficient multi-task learning (MTL). Our experiments on PASCAL MTL and NYUD show that DiTASK achieves state-of-the-art performance across four dense prediction tasks, using 75% fewer parameters than existing methods. Our code is available [here](https://github.com/ipsitmantri/DiTASK).
>
---
#### [replaced 083] ARFlow: Human Action-Reaction Flow Matching with Physical Guidance
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.16973v3](http://arxiv.org/pdf/2503.16973v3)**

> **作者:** Wentao Jiang; Jingya Wang; Kaiyang Ji; Baoxiong Jia; Siyuan Huang; Ye Shi
>
> **备注:** Project Page: https://arflow2025.github.io/
>
> **摘要:** Human action-reaction synthesis, a fundamental challenge in modeling causal human interactions, plays a critical role in applications ranging from virtual reality to social robotics. While diffusion-based models have demonstrated promising performance, they exhibit two key limitations for interaction synthesis: reliance on complex noise-to-reaction generators with intricate conditional mechanisms, and frequent physical violations in generated motions. To address these issues, we propose Action-Reaction Flow Matching (ARFlow), a novel framework that establishes direct action-to-reaction mappings, eliminating the need for complex conditional mechanisms. Our approach introduces a physical guidance mechanism specifically designed for Flow Matching (FM) that effectively prevents body penetration artifacts during sampling. Moreover, we discover the bias of traditional flow matching sampling algorithm and employ a reprojection method to revise the sampling direction of FM. To further enhance the reaction diversity, we incorporate randomness into the sampling process. Extensive experiments on NTU120, Chi3D and InterHuman datasets demonstrate that ARFlow not only outperforms existing methods in terms of Fr\'echet Inception Distance and motion diversity but also significantly reduces body collisions, as measured by our new Intersection Volume and Intersection Frequency metrics.
>
---
#### [replaced 084] The Structural Safety Generalization Problem
- **分类: cs.CR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09712v2](http://arxiv.org/pdf/2504.09712v2)**

> **作者:** Julius Broomfield; Tom Gibbs; Ethan Kosak-Hine; George Ingebretsen; Tia Nasir; Jason Zhang; Reihaneh Iranmanesh; Sara Pieri; Reihaneh Rabbany; Kellin Pelrine
>
> **摘要:** LLM jailbreaks are a widespread safety challenge. Given this problem has not yet been tractable, we suggest targeting a key failure mechanism: the failure of safety to generalize across semantically equivalent inputs. We further focus the target by requiring desirable tractability properties of attacks to study: explainability, transferability between models, and transferability between goals. We perform red-teaming within this framework by uncovering new vulnerabilities to multi-turn, multi-image, and translation-based attacks. These attacks are semantically equivalent by our design to their single-turn, single-image, or untranslated counterparts, enabling systematic comparisons; we show that the different structures yield different safety outcomes. We then demonstrate the potential for this framework to enable new defenses by proposing a Structure Rewriting Guardrail, which converts an input to a structure more conducive to safety assessment. This guardrail significantly improves refusal of harmful inputs, without over-refusing benign ones. Thus, by framing this intermediate challenge - more tractable than universal defenses but essential for long-term safety - we highlight a critical milestone for AI safety research.
>
---
#### [replaced 085] ObjectAdd: Adding Objects into Image via a Training-Free Diffusion Modification Fashion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.17230v4](http://arxiv.org/pdf/2404.17230v4)**

> **作者:** Ziyue Zhang; Quanjian Song; Yuxin Zhang; Rongrong Ji
>
> **备注:** 13 pages in total
>
> **摘要:** We introduce ObjectAdd, a training-free diffusion modification method to add user-expected objects into user-specified area. The motive of ObjectAdd stems from: first, describing everything in one prompt can be difficult, and second, users often need to add objects into the generated image. To accommodate with real world, our ObjectAdd maintains accurate image consistency after adding objects with technical innovations in: (1) embedding-level concatenation to ensure correct text embedding coalesce; (2) object-driven layout control with latent and attention injection to ensure objects accessing user-specified area; (3) prompted image inpainting in an attention refocusing & object expansion fashion to ensure rest of the image stays the same. With a text-prompted image, our ObjectAdd allows users to specify a box and an object, and achieves: (1) adding object inside the box area; (2) exact content outside the box area; (3) flawless fusion between the two areas
>
---
#### [replaced 086] Normalized Attention Guidance: Universal Negative Guidance for Diffusion Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21179v2](http://arxiv.org/pdf/2505.21179v2)**

> **作者:** Dar-Yen Chen; Hmrishav Bandyopadhyay; Kai Zou; Yi-Zhe Song
>
> **摘要:** Negative guidance -- explicitly suppressing unwanted attributes -- remains a fundamental challenge in diffusion models, particularly in few-step sampling regimes. While Classifier-Free Guidance (CFG) works well in standard settings, it fails under aggressive sampling step compression due to divergent predictions between positive and negative branches. We present Normalized Attention Guidance (NAG), an efficient, training-free mechanism that applies extrapolation in attention space with L1-based normalization and refinement. NAG restores effective negative guidance where CFG collapses while maintaining fidelity. Unlike existing approaches, NAG generalizes across architectures (UNet, DiT), sampling regimes (few-step, multi-step), and modalities (image, video), functioning as a \textit{universal} plug-in with minimal computational overhead. Through extensive experimentation, we demonstrate consistent improvements in text alignment (CLIP Score), fidelity (FID, PFID), and human-perceived quality (ImageReward). Our ablation studies validate each design component, while user studies confirm significant preference for NAG-guided outputs. As a model-agnostic inference-time approach requiring no retraining, NAG provides effortless negative guidance for all modern diffusion frameworks -- pseudocode in the Appendix!
>
---
#### [replaced 087] Robust Multimodal Learning via Cross-Modal Proxy Tokens
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.17823v3](http://arxiv.org/pdf/2501.17823v3)**

> **作者:** Md Kaykobad Reza; Ameya Patil; Mashhour Solh; M. Salman Asif
>
> **备注:** 21 Pages, 9 Figures, 6 Tables
>
> **摘要:** Multimodal models often experience a significant performance drop when one or more modalities are missing during inference. To address this challenge, we propose a simple yet effective approach that enhances robustness to missing modalities while maintaining strong performance when all modalities are available. Our method introduces cross-modal proxy tokens (CMPTs), which approximate the class token of a missing modality by attending only to the tokens of the available modality without requiring explicit modality generation or auxiliary networks. To efficiently learn these approximations with minimal computational overhead, we employ low-rank adapters in frozen unimodal encoders and jointly optimize an alignment loss with a task-specific loss. Extensive experiments on five multimodal datasets show that our method outperforms state-of-the-art baselines across various missing rates while achieving competitive results in complete-modality settings. Overall, our method offers a flexible and efficient solution for robust multimodal learning. The code and pretrained models will be released on GitHub.
>
---
#### [replaced 088] Beyond Pretty Pictures: Combined Single- and Multi-Image Super-resolution for Sentinel-2 Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24799v2](http://arxiv.org/pdf/2505.24799v2)**

> **作者:** Aditya Retnanto; Son Le; Sebastian Mueller; Armin Leitner; Michael Riffler; Konrad Schindler; Yohan Iddawela
>
> **摘要:** Super-resolution aims to increase the resolution of satellite images by reconstructing high-frequency details, which go beyond na\"ive upsampling. This has particular relevance for Earth observation missions like Sentinel-2, which offer frequent, regular coverage at no cost; but at coarse resolution. Its pixel footprint is too large to capture small features like houses, streets, or hedge rows. To address this, we present SEN4X, a hybrid super-resolution architecture that combines the advantages of single-image and multi-image techniques. It combines temporal oversampling from repeated Sentinel-2 acquisitions with a learned prior from high-resolution Pl\'eiades Neo data. In doing so, SEN4X upgrades Sentinel-2 imagery to 2.5 m ground sampling distance. We test the super-resolved images on urban land-cover classification in Hanoi, Vietnam. We find that they lead to a significant performance improvement over state-of-the-art super-resolution baselines.
>
---
#### [replaced 089] Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.17019v2](http://arxiv.org/pdf/2502.17019v2)**

> **作者:** Maksim Zhdanov; Max Welling; Jan-Willem van de Meent
>
> **备注:** Accepted to ICML 2025. Code: https://github.com/maxxxzdn/erwin
>
> **摘要:** Large-scale physical systems defined on irregular grids pose significant scalability challenges for deep learning methods, especially in the presence of long-range interactions and multi-scale coupling. Traditional approaches that compute all pairwise interactions, such as attention, become computationally prohibitive as they scale quadratically with the number of nodes. We present Erwin, a hierarchical transformer inspired by methods from computational many-body physics, which combines the efficiency of tree-based algorithms with the expressivity of attention mechanisms. Erwin employs ball tree partitioning to organize computation, which enables linear-time attention by processing nodes in parallel within local neighborhoods of fixed size. Through progressive coarsening and refinement of the ball tree structure, complemented by a novel cross-ball interaction mechanism, it captures both fine-grained local details and global features. We demonstrate Erwin's effectiveness across multiple domains, including cosmology, molecular dynamics, PDE solving, and particle fluid dynamics, where it consistently outperforms baseline methods both in accuracy and computational efficiency.
>
---
#### [replaced 090] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.12392v2](http://arxiv.org/pdf/2412.12392v2)**

> **作者:** Riku Murai; Eric Dexheimer; Andrew J. Davison
>
> **备注:** CVPR 2025 Highlight. The first two authors contributed equally to this work. Project Page: https://edexheim.github.io/mast3r-slam/
>
> **摘要:** We present a real-time monocular dense SLAM system designed bottom-up from MASt3R, a two-view 3D reconstruction and matching prior. Equipped with this strong prior, our system is robust on in-the-wild video sequences despite making no assumption on a fixed or parametric camera model beyond a unique camera centre. We introduce efficient methods for pointmap matching, camera tracking and local fusion, graph construction and loop closure, and second-order global optimisation. With known calibration, a simple modification to the system achieves state-of-the-art performance across various benchmarks. Altogether, we propose a plug-and-play monocular SLAM system capable of producing globally-consistent poses and dense geometry while operating at 15 FPS.
>
---
#### [replaced 091] ChatReID: Open-ended Interactive Person Retrieval via Hierarchical Progressive Tuning for Vision Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.19958v3](http://arxiv.org/pdf/2502.19958v3)**

> **作者:** Ke Niu; Haiyang Yu; Mengyang Zhao; Teng Fu; Siyang Yi; Wei Lu; Bin Li; Xuelin Qian; Xiangyang Xue
>
> **摘要:** Person re-identification (Re-ID) is a crucial task in computer vision, aiming to recognize individuals across non-overlapping camera views. While recent advanced vision-language models (VLMs) excel in logical reasoning and multi-task generalization, their applications in Re-ID tasks remain limited. They either struggle to perform accurate matching based on identity-relevant features or assist image-dominated branches as auxiliary semantics. In this paper, we propose a novel framework ChatReID, that shifts the focus towards a text-side-dominated retrieval paradigm, enabling flexible and interactive re-identification. To integrate the reasoning abilities of language models into Re-ID pipelines, We first present a large-scale instruction dataset, which contains more than 8 million prompts to promote the model fine-tuning. Next. we introduce a hierarchical progressive tuning strategy, which endows Re-ID ability through three stages of tuning, i.e., from person attribute understanding to fine-grained image retrieval and to multi-modal task reasoning. Extensive experiments across ten popular benchmarks demonstrate that ChatReID outperforms existing methods, achieving state-of-the-art performance in all Re-ID tasks. More experiments demonstrate that ChatReID not only has the ability to recognize fine-grained details but also to integrate them into a coherent reasoning process.
>
---
#### [replaced 092] CAP-Net: A Unified Network for 6D Pose and Size Estimation of Categorical Articulated Parts from a Single RGB-D Image
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.11230v3](http://arxiv.org/pdf/2504.11230v3)**

> **作者:** Jingshun Huang; Haitao Lin; Tianyu Wang; Yanwei Fu; Xiangyang Xue; Yi Zhu
>
> **备注:** To appear in CVPR 2025 (Highlight)
>
> **摘要:** This paper tackles category-level pose estimation of articulated objects in robotic manipulation tasks and introduces a new benchmark dataset. While recent methods estimate part poses and sizes at the category level, they often rely on geometric cues and complex multi-stage pipelines that first segment parts from the point cloud, followed by Normalized Part Coordinate Space (NPCS) estimation for 6D poses. These approaches overlook dense semantic cues from RGB images, leading to suboptimal accuracy, particularly for objects with small parts. To address these limitations, we propose a single-stage Network, CAP-Net, for estimating the 6D poses and sizes of Categorical Articulated Parts. This method combines RGB-D features to generate instance segmentation and NPCS representations for each part in an end-to-end manner. CAP-Net uses a unified network to simultaneously predict point-wise class labels, centroid offsets, and NPCS maps. A clustering algorithm then groups points of the same predicted class based on their estimated centroid distances to isolate each part. Finally, the NPCS region of each part is aligned with the point cloud to recover its final pose and size. To bridge the sim-to-real domain gap, we introduce the RGBD-Art dataset, the largest RGB-D articulated dataset to date, featuring photorealistic RGB images and depth noise simulated from real sensors. Experimental evaluations on the RGBD-Art dataset demonstrate that our method significantly outperforms the state-of-the-art approach. Real-world deployments of our model in robotic tasks underscore its robustness and exceptional sim-to-real transfer capabilities, confirming its substantial practical utility. Our dataset, code and pre-trained models are available on the project page.
>
---
#### [replaced 093] GAME: Learning Multimodal Interactions via Graph Structures for Personality Trait Estimation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.03846v2](http://arxiv.org/pdf/2505.03846v2)**

> **作者:** Kangsheng Wang; Yuhang Li; Chengwei Ye; Yufei Lin; Huanzhen Zhang; Bohan Hu; Linuo Xu; Shuyan Liu
>
> **备注:** The article contains serious scientific errors and cannot be corrected by updating the preprint
>
> **摘要:** Apparent personality analysis from short videos poses significant chal-lenges due to the complex interplay of visual, auditory, and textual cues. In this paper, we propose GAME, a Graph-Augmented Multimodal Encoder designed to robustly model and fuse multi-source features for automatic personality prediction. For the visual stream, we construct a facial graph and introduce a dual-branch Geo Two-Stream Network, which combines Graph Convolutional Networks (GCNs) and Convolutional Neural Net-works (CNNs) with attention mechanisms to capture both structural and appearance-based facial cues. Complementing this, global context and iden-tity features are extracted using pretrained ResNet18 and VGGFace back-bones. To capture temporal dynamics, frame-level features are processed by a BiGRU enhanced with temporal attention modules. Meanwhile, audio representations are derived from the VGGish network, and linguistic se-mantics are captured via the XLM-Roberta transformer. To achieve effective multimodal integration, we propose a Channel Attention-based Fusion module, followed by a Multi-Layer Perceptron (MLP) regression head for predicting personality traits. Extensive experiments show that GAME con-sistently outperforms existing methods across multiple benchmarks, vali-dating its effectiveness and generalizability.
>
---
#### [replaced 094] S2A: A Unified Framework for Parameter and Memory Efficient Transfer Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08154v3](http://arxiv.org/pdf/2503.08154v3)**

> **作者:** Tian Jin; Enjun Du; Changwei Wang; Wenhao Xu; Ding Luo
>
> **摘要:** Parameter-efficient transfer learning (PETL) aims to reduce the scales of pretrained models for multiple downstream tasks. However, as the models keep scaling up, the memory footprint of existing PETL methods is not significantly reduced compared to the reduction of learnable parameters. This limitation hinders the practical deployment of PETL methods on memory-constrained devices. To this end, we proposed a new PETL framework, called Structure to Activation (S2A), to reduce the memory footprint of activation during fine-tuning. Specifically, our framework consists of: 1) Activation modules design(i.e., bias, prompt and side modules) in the parametric model structure, which results in a significant reduction of adjustable parameters and activation memory; 2) 4-bit quantization of activations based on their derivatives for non-parametric structures (e.g., nonlinear functions), which maintains accuracy while significantly reducing memory usage. Our S2A method consequently offers a lightweight solution in terms of both parameters and memory footprint. We evaluated S2A with different backbones and performed extensive experiments on various datasets to evaluate the effectiveness. The results show that our methods not only outperform existing PETL techniques, achieving a fourfold reduction in GPU memory footprint on average, but also shows competitive performance in accuracy with fewer tunable parameters. These demonstrate that our method is highly suitable for practical transfer learning on hardware-constrained devices.
>
---
#### [replaced 095] Don't Let Your Robot be Harmful: Responsible Robotic Manipulation via Safety-as-Policy
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18289v2](http://arxiv.org/pdf/2411.18289v2)**

> **作者:** Minheng Ni; Lei Zhang; Zihan Chen; Kaixin Bai; Zhaopeng Chen; Jianwei Zhang; Lei Zhang; Wangmeng Zuo
>
> **摘要:** Unthinking execution of human instructions in robotic manipulation can lead to severe safety risks, such as poisonings, fires, and even explosions. In this paper, we present responsible robotic manipulation, which requires robots to consider potential hazards in the real-world environment while completing instructions and performing complex operations safely and efficiently. However, such scenarios in real world are variable and risky for training. To address this challenge, we propose Safety-as-policy, which includes (i) a world model to automatically generate scenarios containing safety risks and conduct virtual interactions, and (ii) a mental model to infer consequences with reflections and gradually develop the cognition of safety, allowing robots to accomplish tasks while avoiding dangers. Additionally, we create the SafeBox synthetic dataset, which includes one hundred responsible robotic manipulation tasks with different safety risk scenarios and instructions, effectively reducing the risks associated with real-world experiments. Experiments demonstrate that Safety-as-policy can avoid risks and efficiently complete tasks in both synthetic dataset and real-world experiments, significantly outperforming baseline methods. Our SafeBox dataset shows consistent evaluation results with real-world scenarios, serving as a safe and effective benchmark for future research.
>
---
#### [replaced 096] Motion-compensated cardiac MRI using low-rank diffeomorphic flow (DMoCo)
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.03149v3](http://arxiv.org/pdf/2505.03149v3)**

> **作者:** Joseph Kettelkamp; Ludovica Romanin; Sarv Priya; Mathews Jacob
>
> **摘要:** We introduce an unsupervised motion-compensated image reconstruction algorithm for free-breathing and ungated 3D cardiac magnetic resonance imaging (MRI). We express the image volume corresponding to each specific motion phase as the deformation of a single static image template. The main contribution of the work is the low-rank model for the compact joint representation of the family of diffeomorphisms, parameterized by the motion phases. The diffeomorphism at a specific motion phase is obtained by integrating a parametric velocity field along a path connecting the reference template phase to the motion phase. The velocity field at different phases is represented using a low-rank model. The static template and the low-rank motion model parameters are learned directly from the k-space data in an unsupervised fashion. The more constrained motion model is observed to offer improved recovery compared to current motion-resolved and motion-compensated algorithms for free-breathing 3D cine MRI.
>
---
#### [replaced 097] PixFoundation: Are We Heading in the Right Direction with Pixel-level Vision Foundation Models?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.04192v3](http://arxiv.org/pdf/2502.04192v3)**

> **作者:** Mennatullah Siam
>
> **备注:** Under Review
>
> **摘要:** Multiple works have emerged to push the boundaries on multi-modal large language models (MLLMs) towards pixel-level understanding. The current trend in pixel-level MLLMs is to train with pixel-level grounding supervision on large-scale labelled data with specialized decoders for the segmentation task. However, we show that such MLLMs when evaluated on recent challenging vision-centric benchmarks, exhibit a weak ability in visual question answering (VQA). Surprisingly, some of these methods even downgrade the grounding ability of MLLMs that were never trained with such pixel-level supervision. In this work, we propose two novel challenging benchmarks with paired evaluation for both VQA and grounding. We show that MLLMs without pixel-level grounding supervision can outperform the state of the art in such tasks. Our paired benchmarks and evaluation enable additional analysis on the reasons for failure with respect to VQA and/or grounding. Furthermore, we propose simple baselines to extract the grounding information that can be plugged into any MLLM, which we call PixFoundation. More importantly, we study the research question of "When does grounding emerge in MLLMs that are not trained with pixel-level grounding supervision?" We show that grounding can coincide with object parts, its location, appearance, context or state, where we show 27-45% of the examples in both benchmarks exhibit this phenomenon. Our code and datasets will be made publicly available and some are in the supplemental.
>
---
#### [replaced 098] Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.19252v2](http://arxiv.org/pdf/2501.19252v2)**

> **作者:** Yuta Oshima; Masahiro Suzuki; Yutaka Matsuo; Hiroki Furuta
>
> **备注:** Code: https://github.com/shim0114/T2V-Diffusion-Search
>
> **摘要:** The remarkable progress in text-to-video diffusion models enables photorealistic generations, although the contents of the generated video often include unnatural movement or deformation, reverse playback, and motionless scenes. Recently, an alignment problem has attracted huge attention, where we steer the output of diffusion models based on some quantity on the goodness of the content. Because there is a large room for improvement of perceptual quality along the frame direction, we should address which metrics we should optimize and how we can optimize them in the video generation. In this paper, we propose diffusion latent beam search with lookahead estimator, which can select a better diffusion latent to maximize a given alignment reward, at inference time. We then point out that the improvement of perceptual video quality considering the alignment to prompts requires reward calibration by weighting existing metrics. This is because when humans or vision language models evaluate outputs, many previous metrics to quantify the naturalness of video do not always correlate with evaluation. We demonstrate that our method improves the perceptual quality evaluated on the calibrated reward, VLMs, and human assessment, without model parameter update, and outputs the best generation compared to greedy search and best-of-N sampling under much more efficient computational cost. The experiments highlight that our method is beneficial to many capable generative models, and provide a practical guideline that we should prioritize the inference-time compute allocation into lookahead steps for reward estimation over search budget or denoising steps.
>
---
#### [replaced 099] NEXT: Multi-Grained Mixture of Experts via Text-Modulation for Multi-Modal Object Re-ID
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20001v3](http://arxiv.org/pdf/2505.20001v3)**

> **作者:** Shihao Li; Chenglong Li; Aihua Zheng; Andong Lu; Jin Tang; Jixin Ma
>
> **摘要:** Multi-modal object re-identification (ReID) aims to extract identity features across heterogeneous spectral modalities to enable accurate recognition and retrieval in complex real-world scenarios. However, most existing methods rely on implicit feature fusion structures, making it difficult to model fine-grained recognition strategies under varying challenging conditions. Benefiting from the powerful semantic understanding capabilities of Multi-modal Large Language Models (MLLMs), the visual appearance of an object can be effectively translated into descriptive text. In this paper, we propose a reliable multi-modal caption generation method based on attribute confidence, which significantly reduces the unknown recognition rate of MLLMs in multi-modal semantic generation and improves the quality of generated text. Additionally, we propose a novel ReID framework NEXT, the Multi-grained Mixture of Experts via Text-Modulation for Multi-modal Object Re-Identification. Specifically, we decouple the recognition problem into semantic and structural expert branches to separately capture modality-specific appearance and intrinsic structure. For semantic recognition, we propose the Text-Modulated Semantic-sampling Experts (TMSE), which leverages randomly sampled high-quality semantic texts to modulate expert-specific sampling of multi-modal features and mining intra-modality fine-grained semantic cues. Then, to recognize coarse-grained structure features, we propose the Context-Shared Structure-aware Experts (CSSE) that focuses on capturing the holistic object structure across modalities and maintains inter-modality structural consistency through a soft routing mechanism. Finally, we propose the Multi-Modal Feature Aggregation (MMFA), which adopts a unified feature fusion strategy to simply and effectively integrate semantic and structural expert outputs into the final identity representations.
>
---
#### [replaced 100] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.12861v3](http://arxiv.org/pdf/2412.12861v3)**

> **作者:** Zhengdi Yu; Stefanos Zafeiriou; Tolga Birdal
>
> **备注:** Project page is available at https://dyn-hamr.github.io/
>
> **摘要:** We propose Dyn-HaMR, to the best of our knowledge, the first approach to reconstruct 4D global hand motion from monocular videos recorded by dynamic cameras in the wild. Reconstructing accurate 3D hand meshes from monocular videos is a crucial task for understanding human behaviour, with significant applications in augmented and virtual reality (AR/VR). However, existing methods for monocular hand reconstruction typically rely on a weak perspective camera model, which simulates hand motion within a limited camera frustum. As a result, these approaches struggle to recover the full 3D global trajectory and often produce noisy or incorrect depth estimations, particularly when the video is captured by dynamic or moving cameras, which is common in egocentric scenarios. Our Dyn-HaMR consists of a multi-stage, multi-objective optimization pipeline, that factors in (i) simultaneous localization and mapping (SLAM) to robustly estimate relative camera motion, (ii) an interacting-hand prior for generative infilling and to refine the interaction dynamics, ensuring plausible recovery under (self-)occlusions, and (iii) hierarchical initialization through a combination of state-of-the-art hand tracking methods. Through extensive evaluations on both in-the-wild and indoor datasets, we show that our approach significantly outperforms state-of-the-art methods in terms of 4D global mesh recovery. This establishes a new benchmark for hand motion reconstruction from monocular video with moving cameras. Our project page is at https://dyn-hamr.github.io/.
>
---
#### [replaced 101] Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.08145v2](http://arxiv.org/pdf/2410.08145v2)**

> **作者:** Xiaoyuan Liu; Wenxuan Wang; Youliang Yuan; Jen-tse Huang; Qiuzhi Liu; Pinjia He; Zhaopeng Tu
>
> **备注:** Accepted by ACL 2025 main
>
> **摘要:** This paper explores the problem of commonsense level vision-knowledge conflict in Multimodal Large Language Models (MLLMs), where visual information contradicts model's internal commonsense knowledge. To study this issue, we introduce an automated framework, augmented with human-in-the-loop quality control, to generate inputs designed to simulate and evaluate these conflicts in MLLMs. Using this framework, we have crafted a diagnostic benchmark consisting of 374 original images and 1,122 high-quality question-answer (QA) pairs. The benchmark covers two aspects of conflict and three question types, providing a thorough assessment tool. We apply this benchmark to assess the conflict-resolution capabilities of nine representative MLLMs from various model families. Our results indicate an evident over-reliance on parametric knowledge for approximately 20% of all queries, especially among Yes-No and action-related problems. Based on these findings, we evaluate the effectiveness of existing approaches to mitigating the conflicts and compare them to our "Focus-on-Vision" prompting strategy. Despite some improvement, the vision-knowledge conflict remains unresolved and can be further scaled through our data construction framework. Our proposed framework, benchmark, and analysis contribute to the understanding and mitigation of vision-knowledge conflicts in MLLMs.
>
---
#### [replaced 102] SurgRIPE challenge: Benchmark of Surgical Robot Instrument Pose Estimation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.02990v2](http://arxiv.org/pdf/2501.02990v2)**

> **作者:** Haozheng Xu; Alistair Weld; Chi Xu; Alfie Roddan; Joao Cartucho; Mert Asim Karaoglu; Alexander Ladikos; Yangke Li; Yiping Li; Daiyun Shen; Geonhee Lee; Seyeon Park; Jongho Shin; Young-Gon Kim; Lucy Fothergill; Dominic Jones; Pietro Valdastri; Duygu Sarikaya; Stamatia Giannarou
>
> **备注:** 35 pages, 18 figures, journal paper
>
> **摘要:** Accurate instrument pose estimation is a crucial step towards the future of robotic surgery, enabling applications such as autonomous surgical task execution. Vision-based methods for surgical instrument pose estimation provide a practical approach to tool tracking, but they often require markers to be attached to the instruments. Recently, more research has focused on the development of marker-less methods based on deep learning. However, acquiring realistic surgical data, with ground truth instrument poses, required for deep learning training, is challenging. To address the issues in surgical instrument pose estimation, we introduce the Surgical Robot Instrument Pose Estimation (SurgRIPE) challenge, hosted at the 26th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) in 2023. The objectives of this challenge are: (1) to provide the surgical vision community with realistic surgical video data paired with ground truth instrument poses, and (2) to establish a benchmark for evaluating markerless pose estimation methods. The challenge led to the development of several novel algorithms that showcased improved accuracy and robustness over existing methods. The performance evaluation study on the SurgRIPE dataset highlights the potential of these advanced algorithms to be integrated into robotic surgery systems, paving the way for more precise and autonomous surgical procedures. The SurgRIPE challenge has successfully established a new benchmark for the field, encouraging further research and development in surgical robot instrument pose estimation.
>
---
#### [replaced 103] Concept Based Explanations and Class Contrasting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.03422v2](http://arxiv.org/pdf/2502.03422v2)**

> **作者:** Rudolf Herdt; Daniel Otero Baguer
>
> **摘要:** Explaining deep neural networks is challenging, due to their large size and non-linearity. In this paper, we introduce a concept-based explanation method, in order to explain the prediction for an individual class, as well as contrasting any two classes, i.e. explain why the model predicts one class over the other. We test it on several openly available classification models trained on ImageNet1K. We perform both qualitative and quantitative tests. For example, for a ResNet50 model from pytorch model zoo, we can use the explanation for why the model predicts a class 'A' to automatically select four dataset crops where the model does not predict class 'A'. The model then predicts class 'A' again for the newly combined image in 91.1% of the cases (works for 911 out of the 1000 classes). The code including an .ipynb example is available on github: https://github.com/rherdt185/concept-based-explanations-and-class-contrasting
>
---
#### [replaced 104] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.01225v2](http://arxiv.org/pdf/2504.01225v2)**

> **作者:** Gonçalo Gomes; Bruno Martins; Chrysoula Zerva
>
> **备注:** Accepted at Findings ACL 2025
>
> **摘要:** This study explores current limitations of learned image captioning evaluation metrics, specifically the lack of granular assessments for errors within captions, and the reliance on single-point quality estimates without considering uncertainty. To address the limitations, we propose a simple yet effective strategy for generating and calibrating distributions of CLIPScore values. Leveraging a model-agnostic conformal risk control framework, we calibrate CLIPScore values for task-specific control variables, tackling the aforementioned limitations. Experimental results demonstrate that using conformal risk control, over score distributions produced with simple methods such as input masking, can achieve competitive performance compared to more complex approaches. Our method effectively detects erroneous words, while providing formal guarantees aligned with desired risk levels. It also improves the correlation between uncertainty estimations and prediction errors, thus enhancing the overall reliability of caption evaluation metrics.
>
---
#### [replaced 105] DIS-CO: Discovering Copyrighted Content in VLMs Training Data
- **分类: cs.CV; cs.AI; cs.LG; I.2**

- **链接: [http://arxiv.org/pdf/2502.17358v3](http://arxiv.org/pdf/2502.17358v3)**

> **作者:** André V. Duarte; Xuandong Zhao; Arlindo L. Oliveira; Lei Li
>
> **摘要:** How can we verify whether copyrighted content was used to train a large vision-language model (VLM) without direct access to its training data? Motivated by the hypothesis that a VLM is able to recognize images from its training corpus, we propose DIS-CO, a novel approach to infer the inclusion of copyrighted content during the model's development. By repeatedly querying a VLM with specific frames from targeted copyrighted material, DIS-CO extracts the content's identity through free-form text completions. To assess its effectiveness, we introduce MovieTection, a benchmark comprising 14,000 frames paired with detailed captions, drawn from films released both before and after a model's training cutoff. Our results show that DIS-CO significantly improves detection performance, nearly doubling the average AUC of the best prior method on models with logits available. Our findings also highlight a broader concern: all tested models appear to have been exposed to some extent to copyrighted content. Our code and data are available at https://github.com/avduarte333/DIS-CO
>
---
#### [replaced 106] Contrastive Alignment with Semantic Gap-Aware Corrections in Text-Video Retrieval
- **分类: cs.CV; cs.IR; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.12499v4](http://arxiv.org/pdf/2505.12499v4)**

> **作者:** Jian Xiao; Zijie Song; Jialong Hu; Hao Cheng; Zhenzhen Hu; Jia Li; Richang Hong
>
> **摘要:** Recent advances in text-video retrieval have been largely driven by contrastive learning frameworks. However, existing methods overlook a key source of optimization tension: the separation between text and video distributions in the representation space (referred to as the modality gap), and the prevalence of false negatives in batch sampling. These factors lead to conflicting gradients under the InfoNCE loss, impeding stable alignment. To mitigate this, we propose GARE, a Gap-Aware Retrieval framework that introduces a learnable, pair-specific increment Delta_ij between text t_i and video v_j to offload the tension from the global anchor representation. We first derive the ideal form of Delta_ij via a coupled multivariate first-order Taylor approximation of the InfoNCE loss under a trust-region constraint, revealing it as a mechanism for resolving gradient conflicts by guiding updates along a locally optimal descent direction. Due to the high cost of directly computing Delta_ij, we introduce a lightweight neural module conditioned on the semantic gap between each video-text pair, enabling structure-aware correction guided by gradient supervision. To further stabilize learning and promote interpretability, we regularize Delta using three components: a trust-region constraint to prevent oscillation, a directional diversity term to promote semantic coverage, and an information bottleneck to limit redundancy. Experiments across four retrieval benchmarks show that GARE consistently improves alignment accuracy and robustness to noisy supervision, confirming the effectiveness of gap-aware tension mitigation.
>
---
#### [replaced 107] A Survey of 3D Reconstruction with Event Cameras
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.08438v2](http://arxiv.org/pdf/2505.08438v2)**

> **作者:** Chuanzhi Xu; Haoxian Zhou; Langyi Chen; Haodong Chen; Ying Zhou; Vera Chung; Qiang Qu; Weidong Cai
>
> **备注:** 24 pages, 16 figures, 11 tables
>
> **摘要:** Event cameras are rapidly emerging as powerful vision sensors for 3D reconstruction, uniquely capable of asynchronously capturing per-pixel brightness changes. Compared to traditional frame-based cameras, event cameras produce sparse yet temporally dense data streams, enabling robust and accurate 3D reconstruction even under challenging conditions such as high-speed motion, low illumination, and extreme dynamic range scenarios. These capabilities offer substantial promise for transformative applications across various fields, including autonomous driving, robotics, aerial navigation, and immersive virtual reality. In this survey, we present the first comprehensive review exclusively dedicated to event-based 3D reconstruction. Existing approaches are systematically categorised based on input modality into stereo, monocular, and multimodal systems, and further classified according to reconstruction methodologies, including geometry-based techniques, deep learning approaches, and neural rendering techniques such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Within each category, methods are chronologically organised to highlight the evolution of key concepts and advancements. Furthermore, we provide a detailed summary of publicly available datasets specifically suited to event-based reconstruction tasks. Finally, we discuss significant open challenges in dataset availability, standardised evaluation, effective representation, and dynamic scene reconstruction, outlining insightful directions for future research. This survey aims to serve as an essential reference and provides a clear and motivating roadmap toward advancing the state of the art in event-driven 3D reconstruction.
>
---
#### [replaced 108] LaWa: Using Latent Space for In-Generation Image Watermarking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.05868v3](http://arxiv.org/pdf/2408.05868v3)**

> **作者:** Ahmad Rezaei; Mohammad Akbari; Saeed Ranjbar Alvar; Arezou Fatemi; Yong Zhang
>
> **备注:** Accepted to ECCV 2024
>
> **摘要:** With generative models producing high quality images that are indistinguishable from real ones, there is growing concern regarding the malicious usage of AI-generated images. Imperceptible image watermarking is one viable solution towards such concerns. Prior watermarking methods map the image to a latent space for adding the watermark. Moreover, Latent Diffusion Models (LDM) generate the image in the latent space of a pre-trained autoencoder. We argue that this latent space can be used to integrate watermarking into the generation process. To this end, we present LaWa, an in-generation image watermarking method designed for LDMs. By using coarse-to-fine watermark embedding modules, LaWa modifies the latent space of pre-trained autoencoders and achieves high robustness against a wide range of image transformations while preserving perceptual quality of the image. We show that LaWa can also be used as a general image watermarking method. Through extensive experiments, we demonstrate that LaWa outperforms previous works in perceptual quality, robustness against attacks, and computational complexity, while having very low false positive rate. Code is available here.
>
---
#### [replaced 109] RemoteSAM: Towards Segment Anything for Earth Observation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18022v3](http://arxiv.org/pdf/2505.18022v3)**

> **作者:** Liang Yao; Fan Liu; Delong Chen; Chuanyi Zhang; Yijun Wang; Ziyun Chen; Wei Xu; Shimin Di; Yuhui Zheng
>
> **摘要:** We aim to develop a robust yet flexible visual foundation model for Earth observation. It should possess strong capabilities in recognizing and localizing diverse visual targets while providing compatibility with various input-output interfaces required across different task scenarios. Current systems cannot meet these requirements, as they typically utilize task-specific architecture trained on narrow data domains with limited semantic coverage. Our study addresses these limitations from two aspects: data and modeling. We first introduce an automatic data engine that enjoys significantly better scalability compared to previous human annotation or rule-based approaches. It has enabled us to create the largest dataset of its kind to date, comprising 270K image-text-mask triplets covering an unprecedented range of diverse semantic categories and attribute specifications. Based on this data foundation, we further propose a task unification paradigm that centers around referring expression segmentation. It effectively handles a wide range of vision-centric perception tasks, including classification, detection, segmentation, grounding, etc, using a single model without any task-specific heads. Combining these innovations on data and modeling, we present RemoteSAM, a foundation model that establishes new SoTA on several earth observation perception benchmarks, outperforming other foundation models such as Falcon, GeoChat, and LHRS-Bot with significantly higher efficiency. Models and data are publicly available at https://github.com/1e12Leon/RemoteSAM.
>
---
#### [replaced 110] Real-time Chest X-Ray Distributed Decision Support for Resource-constrained Clinics
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.07818v2](http://arxiv.org/pdf/2412.07818v2)**

> **作者:** Omar H. Khater; Basem Almadani; Farouq Aliyu
>
> **摘要:** Internet of Things (IoT) based healthcare systems offer significant potential for improving the delivery of healthcare services in humanitarian engineering, providing essential healthcare services to millions of underserved people in remote areas worldwide. However, these areas have poor network infrastructure, making communications difficult for traditional IoT. This paper presents a real-time chest X-ray classification system for hospitals in remote areas using FastDDS real-time middleware, offering reliable real-time communication. We fine-tuned a ResNet50 neural network to an accuracy of 88.61%, a precision of 88.76%, and a recall of 88.49\%. Our system results mark an average throughput of 3.2 KB/s and an average latency of 65 ms. The proposed system demonstrates how middleware-based systems can assist doctors in remote locations.
>
---
#### [replaced 111] RePaViT: Scalable Vision Transformer Acceleration via Structural Reparameterization on Feedforward Network Layers
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.21847v2](http://arxiv.org/pdf/2505.21847v2)**

> **作者:** Xuwei Xu; Yang Li; Yudong Chen; Jiajun Liu; Sen Wang
>
> **备注:** Accepted to ICML2025
>
> **摘要:** We reveal that feedforward network (FFN) layers, rather than attention layers, are the primary contributors to Vision Transformer (ViT) inference latency, with their impact signifying as model size increases. This finding highlights a critical opportunity for optimizing the efficiency of large-scale ViTs by focusing on FFN layers. In this work, we propose a novel channel idle mechanism that facilitates post-training structural reparameterization for efficient FFN layers during testing. Specifically, a set of feature channels remains idle and bypasses the nonlinear activation function in each FFN layer, thereby forming a linear pathway that enables structural reparameterization during inference. This mechanism results in a family of ReParameterizable Vision Transformers (RePaViTs), which achieve remarkable latency reductions with acceptable sacrifices (sometimes gains) in accuracy across various ViTs. The benefits of our method scale consistently with model sizes, demonstrating greater speed improvements and progressively narrowing accuracy gaps or even higher accuracies on larger models. In particular, RePa-ViT-Large and RePa-ViT-Huge enjoy 66.8% and 68.7% speed-ups with +1.7% and +1.1% higher top-1 accuracies under the same training strategy, respectively. RePaViT is the first to employ structural reparameterization on FFN layers to expedite ViTs to our best knowledge, and we believe that it represents an auspicious direction for efficient ViTs. Source code is available at https://github.com/Ackesnal/RePaViT.
>
---
#### [replaced 112] Hierarchical Material Recognition from Local Appearance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22911v2](http://arxiv.org/pdf/2505.22911v2)**

> **作者:** Matthew Beveridge; Shree K. Nayar
>
> **摘要:** We introduce a taxonomy of materials for hierarchical recognition from local appearance. Our taxonomy is motivated by vision applications and is arranged according to the physical traits of materials. We contribute a diverse, in-the-wild dataset with images and depth maps of the taxonomy classes. Utilizing the taxonomy and dataset, we present a method for hierarchical material recognition based on graph attention networks. Our model leverages the taxonomic proximity between classes and achieves state-of-the-art performance. We demonstrate the model's potential to generalize to adverse, real-world imaging conditions, and that novel views rendered using the depth maps can enhance this capability. Finally, we show the model's capacity to rapidly learn new materials in a few-shot learning setting.
>
---
#### [replaced 113] FactCheXcker: Mitigating Measurement Hallucinations in Chest X-ray Report Generation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18672v3](http://arxiv.org/pdf/2411.18672v3)**

> **作者:** Alice Heiman; Xiaoman Zhang; Emma Chen; Sung Eun Kim; Pranav Rajpurkar
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Medical vision-language models often struggle with generating accurate quantitative measurements in radiology reports, leading to hallucinations that undermine clinical reliability. We introduce FactCheXcker, a modular framework that de-hallucinates radiology report measurements by leveraging an improved query-code-update paradigm. Specifically, FactCheXcker employs specialized modules and the code generation capabilities of large language models to solve measurement queries generated based on the original report. After extracting measurable findings, the results are incorporated into an updated report. We evaluate FactCheXcker on endotracheal tube placement, which accounts for an average of 78% of report measurements, using the MIMIC-CXR dataset and 11 medical report-generation models. Our results show that FactCheXcker significantly reduces hallucinations, improves measurement precision, and maintains the quality of the original reports. Specifically, FactCheXcker improves the performance of 10/11 models and achieves an average improvement of 135.0% in reducing measurement hallucinations measured by mean absolute error. Code is available at https://github.com/rajpurkarlab/FactCheXcker.
>
---
#### [replaced 114] Wake Vision: A Tailored Dataset and Benchmark Suite for TinyML Computer Vision Applications
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.00892v5](http://arxiv.org/pdf/2405.00892v5)**

> **作者:** Colby Banbury; Emil Njor; Andrea Mattia Garavagno; Mark Mazumder; Matthew Stewart; Pete Warden; Manjunath Kudlur; Nat Jeffries; Xenofon Fafoutis; Vijay Janapa Reddi
>
> **摘要:** Tiny machine learning (TinyML) for low-power devices lacks systematic methodologies for creating large, high-quality datasets suitable for production-grade systems. We present a novel automated pipeline for generating binary classification datasets that addresses this critical gap through several algorithmic innovations: intelligent multi-source label fusion, confidence-aware filtering, automated label correction, and systematic fine-grained benchmark generation. Crucially, automation is not merely convenient but necessary to cope with TinyML's diverse applications. TinyML requires bespoke datasets tailored to specific deployment constraints and use cases, making manual approaches prohibitively expensive and impractical for widespread adoption. Using our pipeline, we create Wake Vision, a large-scale binary classification dataset of almost 6 million images that demonstrates our methodology through person detection--the canonical vision task for TinyML. Wake Vision achieves up to a 6.6% accuracy improvement over existing datasets via a carefully designed two-stage training strategy and provides 100x more images. We demonstrate our broad applicability for automated large-scale TinyML dataset generation across two additional target categories, and show our label error rates are substantially lower than prior work. Our comprehensive fine-grained benchmark suite evaluates model robustness across five critical dimensions, revealing failure modes masked by aggregate metrics. To ensure continuous improvement, we establish ongoing community engagement through competitions hosted by the Edge AI Foundation. All datasets, benchmarks, and code are available under CC-BY 4.0 license, providing a systematic foundation for advancing TinyML research.
>
---
#### [replaced 115] MaxSup: Overcoming Representation Collapse in Label Smoothing
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.15798v2](http://arxiv.org/pdf/2502.15798v2)**

> **作者:** Yuxuan Zhou; Heng Li; Zhi-Qi Cheng; Xudong Yan; Yifei Dong; Mario Fritz; Margret Keuper
>
> **备注:** 24 pages, 15 tables, 5 figures. Preliminary work under review. Do not distribute
>
> **摘要:** Label Smoothing (LS) is widely adopted to reduce overconfidence in neural network predictions and improve generalization. Despite these benefits, recent studies reveal two critical issues with LS. First, LS induces overconfidence in misclassified samples. Second, it compacts feature representations into overly tight clusters, diluting intra-class diversity, although the precise cause of this phenomenon remained elusive. In this paper, we analytically decompose the LS-induced loss, exposing two key terms: (i) a regularization term that dampens overconfidence only when the prediction is correct, and (ii) an error-amplification term that arises under misclassifications. This latter term compels the network to reinforce incorrect predictions with undue certainty, exacerbating representation collapse. To address these shortcomings, we propose Max Suppression (MaxSup), which applies uniform regularization to both correct and incorrect predictions by penalizing the top-1 logit rather than the ground-truth logit. Through extensive feature-space analyses, we show that MaxSup restores intra-class variation and sharpens inter-class boundaries. Experiments on large-scale image classification and multiple downstream tasks confirm that MaxSup is a more robust alternative to LS, consistently reducing overconfidence while preserving richer feature representations. Code is available at: https://github.com/ZhouYuxuanYX/Maximum-Suppression-Regularization
>
---
#### [replaced 116] SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference
- **分类: cs.LG; cs.AI; cs.CV; cs.PF**

- **链接: [http://arxiv.org/pdf/2502.18137v3](http://arxiv.org/pdf/2502.18137v3)**

> **作者:** Jintao Zhang; Chendong Xiang; Haofeng Huang; Jia Wei; Haocheng Xi; Jun Zhu; Jianfei Chen
>
> **备注:** @inproceedings{zhang2025spargeattn, title={Spargeattn: Accurate sparse attention accelerating any model inference}, author={Zhang, Jintao and Xiang, Chendong and Huang, Haofeng and Wei, Jia and Xi, Haocheng and Zhu, Jun and Chen, Jianfei}, booktitle={International Conference on Machine Learning (ICML)}, year={2025} }
>
> **摘要:** An efficient attention implementation is essential for large models due to its quadratic time complexity. Fortunately, attention commonly exhibits sparsity, i.e., many values in the attention map are near zero, allowing for the omission of corresponding computations. Many studies have utilized the sparse pattern to accelerate attention. However, most existing works focus on optimizing attention within specific models by exploiting certain sparse patterns of the attention map. A universal sparse attention that guarantees both the speedup and end-to-end performance of diverse models remains elusive. In this paper, we propose SpargeAttn, a universal sparse and quantized attention for any model. Our method uses a two-stage online filter: in the first stage, we rapidly and accurately predict the attention map, enabling the skip of some matrix multiplications in attention. In the second stage, we design an online softmax-aware filter that incurs no extra overhead and further skips some matrix multiplications. Experiments show that our method significantly accelerates diverse models, including language, image, and video generation, without sacrificing end-to-end metrics. The codes are available at https://github.com/thu-ml/SpargeAttn.
>
---
#### [replaced 117] Exploring Compositional Generalization of Multimodal LLMs for Medical Imaging
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.20070v2](http://arxiv.org/pdf/2412.20070v2)**

> **作者:** Zhenyang Cai; Junying Chen; Rongsheng Wang; Weihong Wang; Yonglin Deng; Dingjie Song; Yize Chen; Zixu Zhang; Benyou Wang
>
> **摘要:** Medical imaging provides essential visual insights for diagnosis, and multimodal large language models (MLLMs) are increasingly utilized for its analysis due to their strong generalization capabilities; however, the underlying factors driving this generalization remain unclear. Current research suggests that multi-task training outperforms single-task as different tasks can benefit each other, but they often overlook the internal relationships within these tasks. To analyze this phenomenon, we attempted to employ compositional generalization (CG), which refers to the models' ability to understand novel combinations by recombining learned elements, as a guiding framework. Since medical images can be precisely defined by Modality, Anatomical area, and Task, naturally providing an environment for exploring CG, we assembled 106 medical datasets to create Med-MAT for comprehensive experiments. The experiments confirmed that MLLMs can use CG to understand unseen medical images and identified CG as one of the main drivers of the generalization observed in multi-task training. Additionally, further studies demonstrated that CG effectively supports datasets with limited data and confirmed that MLLMs can achieve CG across classification and detection tasks, underscoring its broader generalization potential. Med-MAT is available at https://github.com/FreedomIntelligence/Med-MAT.
>
---
#### [replaced 118] SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2504.03561v3](http://arxiv.org/pdf/2504.03561v3)**

> **作者:** Runnan Fang; Xiaobin Wang; Yuan Liang; Shuofei Qiao; Jialong Wu; Zekun Xi; Ningyu Zhang; Yong Jiang; Pengjun Xie; Fei Huang; Huajun Chen
>
> **备注:** ACL 2025
>
> **摘要:** In the interaction between agents and their environments, agents expand their capabilities by planning and executing actions. However, LLM-based agents face substantial challenges when deployed in novel environments or required to navigate unconventional action spaces. To empower agents to autonomously explore environments, optimize workflows, and enhance their understanding of actions, we propose SynWorld, a framework that allows agents to synthesize possible scenarios with multi-step action invocation within the action space and perform Monte Carlo Tree Search (MCTS) exploration to effectively refine their action knowledge in the current environment. Our experiments demonstrate that SynWorld is an effective and general approach to learning action knowledge in new environments. Code is available at https://github.com/zjunlp/SynWorld.
>
---
#### [replaced 119] Articulate-Anything: Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.13882v4](http://arxiv.org/pdf/2410.13882v4)**

> **作者:** Long Le; Jason Xie; William Liang; Hung-Ju Wang; Yue Yang; Yecheng Jason Ma; Kyle Vedder; Arjun Krishna; Dinesh Jayaraman; Eric Eaton
>
> **备注:** ICLR 2025. Project website and open-source code: https://articulate-anything.github.io/
>
> **摘要:** Interactive 3D simulated objects are crucial in AR/VR, animations, and robotics, driving immersive experiences and advanced automation. However, creating these articulated objects requires extensive human effort and expertise, limiting their broader applications. To overcome this challenge, we present Articulate-Anything, a system that automates the articulation of diverse, complex objects from many input modalities, including text, images, and videos. Articulate-Anything leverages vision-language models (VLMs) to generate code that can be compiled into an interactable digital twin for use in standard 3D simulators. Our system exploits existing 3D asset datasets via a mesh retrieval mechanism, along with an actor-critic system that iteratively proposes, evaluates, and refines solutions for articulating the objects, self-correcting errors to achieve a robust outcome. Qualitative evaluations demonstrate Articulate-Anything's capability to articulate complex and even ambiguous object affordances by leveraging rich grounded inputs. In extensive quantitative experiments on the standard PartNet-Mobility dataset, Articulate-Anything substantially outperforms prior work, increasing the success rate from 8.7-11.6% to 75% and setting a new bar for state-of-the-art performance. We further showcase the utility of our system by generating 3D assets from in-the-wild video inputs, which are then used to train robotic policies for fine-grained manipulation tasks in simulation that go beyond basic pick and place. These policies are then transferred to a real robotic system.
>
---
#### [replaced 120] RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14318v2](http://arxiv.org/pdf/2505.14318v2)**

> **作者:** Wenjun Hou; Yi Cheng; Kaishuai Xu; Heng Li; Yan Hu; Wenjie Li; Jiang Liu
>
> **备注:** Accepted to ACL 2025 main
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in various domains, including radiology report generation. Previous approaches have attempted to utilize multimodal LLMs for this task, enhancing their performance through the integration of domain-specific knowledge retrieval. However, these approaches often overlook the knowledge already embedded within the LLMs, leading to redundant information integration. To address this limitation, we propose Radar, a framework for enhancing radiology report generation with supplementary knowledge injection. Radar improves report generation by systematically leveraging both the internal knowledge of an LLM and externally retrieved information. Specifically, it first extracts the model's acquired knowledge that aligns with expert image-based classification outputs. It then retrieves relevant supplementary knowledge to further enrich this information. Finally, by aggregating both sources, Radar generates more accurate and informative radiology reports. Extensive experiments on MIMIC-CXR, CheXpert-Plus, and IU X-ray demonstrate that our model outperforms state-of-the-art LLMs in both language quality and clinical accuracy.
>
---
#### [replaced 121] Don't Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.17820v2](http://arxiv.org/pdf/2405.17820v2)**

> **作者:** Sangmin Woo; Donguk Kim; Jaehyuk Jang; Yubin Choi; Changick Kim
>
> **备注:** ACL 2025 Findings; Project: https://sangminwoo.github.io/AvisC/
>
> **摘要:** Large Vision Language Models (LVLMs) demonstrate strong capabilities in visual understanding and description, yet often suffer from hallucinations, attributing incorrect or misleading features to images. We observe that LVLMs disproportionately focus on a small subset of image tokens--termed blind tokens--which are typically irrelevant to the query (e.g., background or non-object regions). We hypothesize that such attention misalignment plays a key role in generating hallucinated responses. To mitigate this issue, we propose Attentional Vision Calibration (AvisC), a test-time approach that dynamically recalibrates the influence of blind tokens without modifying the underlying attention mechanism. AvisC first identifies blind tokens by analyzing layer-wise attention distributions over image tokens, then employs a contrastive decoding strategy to balance the influence of original and blind-token-biased logits. Experiments on standard benchmarks, including POPE, MME, and AMBER, demonstrate that AvisC effectively reduces hallucinations in LVLMs.
>
---
#### [replaced 122] NFIG: Autoregressive Image Generation with Next-Frequency Prediction
- **分类: cs.CV; cs.AI; 68T07; I.2.10; I.2.6**

- **链接: [http://arxiv.org/pdf/2503.07076v3](http://arxiv.org/pdf/2503.07076v3)**

> **作者:** Zhihao Huang; Xi Qiu; Yukuo Ma; Yifu Zhou; Junjie Chen; Hongyuan Zhang; Chi Zhang; Xuelong Li
>
> **备注:** 10 pages, 7 figures, 2 tables
>
> **摘要:** Autoregressive models have achieved promising results in natural language processing. However, for image generation tasks, they encounter substantial challenges in effectively capturing long-range dependencies, managing computational costs, and most crucially, defining meaningful autoregressive sequences that reflect natural image hierarchies. To address these issues, we present \textbf{N}ext-\textbf{F}requency \textbf{I}mage \textbf{G}eneration (\textbf{NFIG}), a novel framework that decomposes the image generation process into multiple frequency-guided stages. Our approach first generates low-frequency components to establish global structure with fewer tokens, then progressively adds higher-frequency details, following the natural spectral hierarchy of images. This principled autoregressive sequence not only improves the quality of generated images by better capturing true causal relationships between image components, but also significantly reduces computational overhead during inference. Extensive experiments demonstrate that NFIG achieves state-of-the-art performance with fewer steps, offering a more efficient solution for image generation, with 1.25$\times$ speedup compared to VAR-d20 while achieving better performance (FID: 2.81) on the ImageNet-256 benchmark. We hope that our insight of incorporating frequency-domain knowledge to guide autoregressive sequence design will shed light on future research. We will make our code publicly available upon acceptance of the paper.
>
---
#### [replaced 123] PHT-CAD: Efficient CAD Parametric Primitive Analysis with Progressive Hierarchical Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18147v3](http://arxiv.org/pdf/2503.18147v3)**

> **作者:** Ke Niu; Yuwen Chen; Haiyang Yu; Zhuofan Chen; Xianghui Que; Bin Li; Xiangyang Xue
>
> **摘要:** Computer-Aided Design (CAD) plays a pivotal role in industrial manufacturing, yet 2D Parametric Primitive Analysis (PPA) remains underexplored due to two key challenges: structural constraint reasoning and advanced semantic understanding. To tackle these challenges, we first propose an Efficient Hybrid Parametrization (EHP) for better representing 2D engineering drawings. EHP contains four types of atomic component i.e., point, line, circle, and arc). Additionally, we propose PHT-CAD, a novel 2D PPA framework that harnesses the modality alignment and reasoning capabilities of Vision-Language Models (VLMs) for precise engineering drawing analysis. In PHT-CAD, we introduce four dedicated regression heads to predict corresponding atomic components. To train PHT-CAD, a three-stage training paradigm Progressive Hierarchical Tuning (PHT) is proposed to progressively enhance PHT-CAD's capability to perceive individual primitives, infer structural constraints, and align annotation layers with their corresponding geometric representations. Considering that existing datasets lack complete annotation layers and real-world engineering drawings, we introduce ParaCAD, the first large-scale benchmark that explicitly integrates both the geometric and annotation layers. ParaCAD comprises over 10 million annotated drawings for training and 3,000 real-world industrial drawings with complex topological structures and physical constraints for test. Extensive experiments demonstrate the effectiveness of PHT-CAD and highlight the practical significance of ParaCAD in advancing 2D PPA research.
>
---
#### [replaced 124] IrrMap: A Large-Scale Comprehensive Dataset for Irrigation Method Mapping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08273v2](http://arxiv.org/pdf/2505.08273v2)**

> **作者:** Nibir Chandra Mandal; Oishee Bintey Hoque; Abhijin Adiga; Samarth Swarup; Mandy Wilson; Lu Feng; Yangfeng Ji; Miaomiao Zhang; Geoffrey Fox; Madhav Marathe
>
> **摘要:** We introduce IrrMap, the first large-scale dataset (1.1 million patches) for irrigation method mapping across regions. IrrMap consists of multi-resolution satellite imagery from LandSat and Sentinel, along with key auxiliary data such as crop type, land use, and vegetation indices. The dataset spans 1,687,899 farms and 14,117,330 acres across multiple western U.S. states from 2013 to 2023, providing a rich and diverse foundation for irrigation analysis and ensuring geospatial alignment and quality control. The dataset is ML-ready, with standardized 224x224 GeoTIFF patches, the multiple input modalities, carefully chosen train-test-split data, and accompanying dataloaders for seamless deep learning model training andbenchmarking in irrigation mapping. The dataset is also accompanied by a complete pipeline for dataset generation, enabling researchers to extend IrrMap to new regions for irrigation data collection or adapt it with minimal effort for other similar applications in agricultural and geospatial analysis. We also analyze the irrigation method distribution across crop groups, spatial irrigation patterns (using Shannon diversity indices), and irrigated area variations for both LandSat and Sentinel, providing insights into regional and resolution-based differences. To promote further exploration, we openly release IrrMap, along with the derived datasets, benchmark models, and pipeline code, through a GitHub repository: https://github.com/Nibir088/IrrMap and Data repository: https://huggingface.co/Nibir/IrrMap, providing comprehensive documentation and implementation details.
>
---
#### [replaced 125] VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.17451v2](http://arxiv.org/pdf/2411.17451v2)**

> **作者:** Lei Li; Yuancheng Wei; Zhihui Xie; Xuqing Yang; Yifan Song; Peiyi Wang; Chenxin An; Tianyu Liu; Sujian Li; Bill Yuchen Lin; Lingpeng Kong; Qi Liu
>
> **备注:** CVPR 2025 Camera Ready Version. Project page: https://vl-rewardbench.github.io
>
> **摘要:** Vision-language generative reward models (VL-GenRMs) play a crucial role in aligning and evaluating multimodal AI systems, yet their own evaluation remains under-explored. Current assessment methods primarily rely on AI-annotated preference labels from traditional VL tasks, which can introduce biases and often fail to effectively challenge state-of-the-art models. To address these limitations, we introduce VL-RewardBench, a comprehensive benchmark spanning general multimodal queries, visual hallucination detection, and complex reasoning tasks. Through our AI-assisted annotation pipeline that combines sample selection with human verification, we curate 1,250 high-quality examples specifically designed to probe VL-GenRMs limitations. Comprehensive evaluation across 16 leading large vision-language models demonstrates VL-RewardBench's effectiveness as a challenging testbed, where even GPT-4o achieves only 65.4% accuracy, and state-of-the-art open-source models such as Qwen2-VL-72B, struggle to surpass random-guessing. Importantly, performance on VL-RewardBench strongly correlates (Pearson's r $>$ 0.9) with MMMU-Pro accuracy using Best-of-N sampling with VL-GenRMs. Analysis experiments uncover three critical insights for improving VL-GenRMs: (i) models predominantly fail at basic visual perception tasks rather than reasoning tasks; (ii) inference-time scaling benefits vary dramatically by model capacity; and (iii) training VL-GenRMs to learn to judge substantially boosts judgment capability (+14.7% accuracy for a 7B VL-GenRM). We believe VL-RewardBench along with the experimental insights will become a valuable resource for advancing VL-GenRMs.
>
---
#### [replaced 126] NUC-Net: Non-uniform Cylindrical Partition Network for Efficient LiDAR Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24634v2](http://arxiv.org/pdf/2505.24634v2)**

> **作者:** Xuzhi Wang; Wei Feng; Lingdong Kong; Liang Wan
>
> **备注:** Accepted at TCSVT in 2025.Code available at https://github.com/alanWXZ/NUC-Net
>
> **摘要:** LiDAR semantic segmentation plays a vital role in autonomous driving. Existing voxel-based methods for LiDAR semantic segmentation apply uniform partition to the 3D LiDAR point cloud to form a structured representation based on cartesian/cylindrical coordinates. Although these methods show impressive performance, the drawback of existing voxel-based methods remains in two aspects: (1) it requires a large enough input voxel resolution, which brings a large amount of computation cost and memory consumption. (2) it does not well handle the unbalanced point distribution of LiDAR point cloud. In this paper, we propose a non-uniform cylindrical partition network named NUC-Net to tackle the above challenges. Specifically, we propose the Arithmetic Progression of Interval (API) method to non-uniformly partition the radial axis and generate the voxel representation which is representative and efficient. Moreover, we propose a non-uniform multi-scale aggregation method to improve contextual information. Our method achieves state-of-the-art performance on SemanticKITTI and nuScenes datasets with much faster speed and much less training time. And our method can be a general component for LiDAR semantic segmentation, which significantly improves both the accuracy and efficiency of the uniform counterpart by $4 \times$ training faster and $2 \times$ GPU memory reduction and $3 \times$ inference speedup. We further provide theoretical analysis towards understanding why NUC is effective and how point distribution affects performance. Code is available at \href{https://github.com/alanWXZ/NUC-Net}{https://github.com/alanWXZ/NUC-Net}.
>
---
#### [replaced 127] QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16175v2](http://arxiv.org/pdf/2505.16175v2)**

> **作者:** Benjamin Schneider; Dongfu Jiang; Chao Du; Tianyu Pang; Wenhu Chen
>
> **备注:** 19 pages, 6 figures, 2 tables
>
> **摘要:** Long-video understanding has emerged as a crucial capability in real-world applications such as video surveillance, meeting summarization, educational lecture analysis, and sports broadcasting. However, it remains computationally prohibitive for VideoLLMs, primarily due to two bottlenecks: 1) sequential video decoding, the process of converting the raw bit stream to RGB frames can take up to a minute for hour-long video inputs, and 2) costly prefilling of up to several million tokens for LLM inference, resulting in high latency and memory use. To address these challenges, we propose QuickVideo, a system-algorithm co-design that substantially accelerates long-video understanding to support real-time downstream applications. It comprises three key innovations: QuickDecoder, a parallelized CPU-based video decoder that achieves 2-3 times speedup by splitting videos into keyframe-aligned intervals processed concurrently; QuickPrefill, a memory-efficient prefilling method using KV-cache pruning to support more frames with less GPU memory; and an overlapping scheme that overlaps CPU video decoding with GPU inference. Together, these components infernece time reduce by a minute on long video inputs, enabling scalable, high-quality video understanding even on limited hardware. Experiments show that QuickVideo generalizes across durations and sampling rates, making long video processing feasible in practice.
>
---
#### [replaced 128] Segment Anything for Histopathology
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00408v2](http://arxiv.org/pdf/2502.00408v2)**

> **作者:** Titus Griebel; Anwai Archit; Constantin Pape
>
> **备注:** Published in MIDL 2025
>
> **摘要:** Nucleus segmentation is an important analysis task in digital pathology. However, methods for automatic segmentation often struggle with new data from a different distribution, requiring users to manually annotate nuclei and retrain data-specific models. Vision foundation models (VFMs), such as the Segment Anything Model (SAM), offer a more robust alternative for automatic and interactive segmentation. Despite their success in natural images, a foundation model for nucleus segmentation in histopathology is still missing. Initial efforts to adapt SAM have shown some success, but did not yet introduce a comprehensive model for diverse segmentation tasks. To close this gap, we introduce PathoSAM, a VFM for nucleus segmentation, based on training SAM on a diverse dataset. Our extensive experiments show that it is the new state-of-the-art model for automatic and interactive nucleus instance segmentation in histopathology. We also demonstrate how it can be adapted for other segmentation tasks, including semantic nucleus segmentation. For this task, we show that it yields results better than popular methods, while not yet beating the state-of-the-art, CellViT. Our models are open-source and compatible with popular tools for data annotation. We also provide scripts for whole-slide image segmentation. Our code and models are publicly available at https://github.com/computational-cell-analytics/patho-sam.
>
---
#### [replaced 129] Probing Equivariance and Symmetry Breaking in Convolutional Networks
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.01999v3](http://arxiv.org/pdf/2501.01999v3)**

> **作者:** Sharvaree Vadgama; Mohammad Mohaiminul Islam; Domas Buracas; Christian Shewmake; Artem Moskalev; Erik Bekkers
>
> **备注:** 27 pages, 7 figures
>
> **摘要:** In this work, we explore the trade-offs of explicit structural priors, particularly group equivariance. We address this through theoretical analysis and a comprehensive empirical study. To enable controlled and fair comparisons, we introduce \texttt{Rapidash}, a unified group convolutional architecture that allows for different variants of equivariant and non-equivariant models. Our results suggest that more constrained equivariant models outperform less constrained alternatives when aligned with the geometry of the task, and increasing representation capacity does not fully eliminate performance gaps. We see improved performance of models with equivariance and symmetry-breaking through tasks like segmentation, regression, and generation across diverse datasets. Explicit \textit{symmetry breaking} via geometric reference frames consistently improves performance, while \textit{breaking equivariance} through geometric input features can be helpful when aligned with task geometry. Our results provide task-specific performance trends that offer a more nuanced way for model selection.
>
---
#### [replaced 130] View-Invariant Policy Learning via Zero-Shot Novel View Synthesis
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.03685v3](http://arxiv.org/pdf/2409.03685v3)**

> **作者:** Stephen Tian; Blake Wulfe; Kyle Sargent; Katherine Liu; Sergey Zakharov; Vitor Guizilini; Jiajun Wu
>
> **备注:** Accepted to CoRL 2024
>
> **摘要:** Large-scale visuomotor policy learning is a promising approach toward developing generalizable manipulation systems. Yet, policies that can be deployed on diverse embodiments, environments, and observational modalities remain elusive. In this work, we investigate how knowledge from large-scale visual data of the world may be used to address one axis of variation for generalizable manipulation: observational viewpoint. Specifically, we study single-image novel view synthesis models, which learn 3D-aware scene-level priors by rendering images of the same scene from alternate camera viewpoints given a single input image. For practical application to diverse robotic data, these models must operate zero-shot, performing view synthesis on unseen tasks and environments. We empirically analyze view synthesis models within a simple data-augmentation scheme that we call View Synthesis Augmentation (VISTA) to understand their capabilities for learning viewpoint-invariant policies from single-viewpoint demonstration data. Upon evaluating the robustness of policies trained with our method to out-of-distribution camera viewpoints, we find that they outperform baselines in both simulated and real-world manipulation tasks. Videos and additional visualizations are available at https://s-tian.github.io/projects/vista.
>
---
#### [replaced 131] OpenUni: A Simple Baseline for Unified Multimodal Understanding and Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23661v3](http://arxiv.org/pdf/2505.23661v3)**

> **作者:** Size Wu; Zhonghua Wu; Zerui Gong; Qingyi Tao; Sheng Jin; Qinyue Li; Wei Li; Chen Change Loy
>
> **摘要:** In this report, we present OpenUni, a simple, lightweight, and fully open-source baseline for unifying multimodal understanding and generation. Inspired by prevailing practices in unified model learning, we adopt an efficient training strategy that minimizes the training complexity and overhead by bridging the off-the-shelf multimodal large language models (LLMs) and diffusion models through a set of learnable queries and a light-weight transformer-based connector. With a minimalist choice of architecture, we demonstrate that OpenUni can: 1) generate high-quality and instruction-aligned images, and 2) achieve exceptional performance on standard benchmarks such as GenEval, DPG- Bench, and WISE, with only 1.1B and 3.1B activated parameters. To support open research and community advancement, we release all model weights, training code, and our curated training datasets (including 23M image-text pairs) at https://github.com/wusize/OpenUni.
>
---
#### [replaced 132] Distractor-free Generalizable 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17605v2](http://arxiv.org/pdf/2411.17605v2)**

> **作者:** Yanqi Bao; Jing Liao; Jing Huo; Yang Gao
>
> **摘要:** We present DGGS, a novel framework that addresses the previously unexplored challenge: $\textbf{Distractor-free Generalizable 3D Gaussian Splatting}$ (3DGS). It mitigates 3D inconsistency and training instability caused by distractor data in the cross-scenes generalizable train setting while enabling feedforward inference for 3DGS and distractor masks from references in the unseen scenes. To achieve these objectives, DGGS proposes a scene-agnostic reference-based mask prediction and refinement module during the training phase, effectively eliminating the impact of distractor on training stability. Moreover, we combat distractor-induced artifacts and holes at inference time through a novel two-stage inference framework for references scoring and re-selection, complemented by a distractor pruning mechanism that further removes residual distractor 3DGS-primitive influences. Extensive feedforward experiments on the real and our synthetic data show DGGS's reconstruction capability when dealing with novel distractor scenes. Moreover, our generalizable mask prediction even achieves an accuracy superior to existing scene-specific training methods. Homepage is https://github.com/bbbbby-99/DGGS.
>
---
#### [replaced 133] Open High-Resolution Satellite Imagery: The WorldStrat Dataset -- With Application to Super-Resolution
- **分类: eess.IV; cs.CV; cs.LG; stat.AP; 68-04 (Primary), 68T45, 68U10(Secondary); I.2.10; I.2.6**

- **链接: [http://arxiv.org/pdf/2207.06418v2](http://arxiv.org/pdf/2207.06418v2)**

> **作者:** Julien Cornebise; Ivan Oršolić; Freddie Kalaitzis
>
> **备注:** Published in 36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks
>
> **摘要:** Analyzing the planet at scale with satellite imagery and machine learning is a dream that has been constantly hindered by the cost of difficult-to-access highly-representative high-resolution imagery. To remediate this, we introduce here the WorldStrat dataset. The largest and most varied such publicly available dataset, at Airbus SPOT 6/7 satellites' high resolution of up to 1.5 m/pixel, empowered by European Space Agency's Phi-Lab as part of the ESA-funded QueryPlanet project, we curate nearly 10,000 sqkm of unique locations to ensure stratified representation of all types of land-use across the world: from agriculture to ice caps, from forests to multiple urbanization densities. We also enrich those with locations typically under-represented in ML datasets: sites of humanitarian interest, illegal mining sites, and settlements of persons at risk. We temporally-match each high-resolution image with multiple low-resolution images from the freely accessible lower-resolution Sentinel-2 satellites at 10 m/pixel. We accompany this dataset with an open-source Python package to: rebuild or extend the WorldStrat dataset, train and infer baseline algorithms, and learn with abundant tutorials, all compatible with the popular EO-learn toolbox. We hereby hope to foster broad-spectrum applications of ML to satellite imagery, and possibly develop from free public low-resolution Sentinel2 imagery the same power of analysis allowed by costly private high-resolution imagery. We illustrate this specific point by training and releasing several highly compute-efficient baselines on the task of Multi-Frame Super-Resolution. High-resolution Airbus imagery is CC BY-NC, while the labels and Sentinel2 imagery are CC BY, and the source code and pre-trained models under BSD. The dataset is available at https://zenodo.org/record/6810791 and the software package at https://github.com/worldstrat/worldstrat .
>
---
#### [replaced 134] DragPoser: Motion Reconstruction from Variable Sparse Tracking Signals via Latent Space Optimization
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.14567v3](http://arxiv.org/pdf/2406.14567v3)**

> **作者:** Jose Luis Ponton; Eduard Pujol; Andreas Aristidou; Carlos Andujar; Nuria Pelechano
>
> **备注:** Published on Eurographics 2025. Project page: https://upc-virvig.github.io/DragPoser/
>
> **摘要:** High-quality motion reconstruction that follows the user's movements can be achieved by high-end mocap systems with many sensors. However, obtaining such animation quality with fewer input devices is gaining popularity as it brings mocap closer to the general public. The main challenges include the loss of end-effector accuracy in learning-based approaches, or the lack of naturalness and smoothness in IK-based solutions. In addition, such systems are often finely tuned to a specific number of trackers and are highly sensitive to missing data e.g., in scenarios where a sensor is occluded or malfunctions. In response to these challenges, we introduce DragPoser, a novel deep-learning-based motion reconstruction system that accurately represents hard and dynamic on-the-fly constraints, attaining real-time high end-effectors position accuracy. This is achieved through a pose optimization process within a structured latent space. Our system requires only one-time training on a large human motion dataset, and then constraints can be dynamically defined as losses, while the pose is iteratively refined by computing the gradients of these losses within the latent space. To further enhance our approach, we incorporate a Temporal Predictor network, which employs a Transformer architecture to directly encode temporality within the latent space. This network ensures the pose optimization is confined to the manifold of valid poses and also leverages past pose data to predict temporally coherent poses. Results demonstrate that DragPoser surpasses both IK-based and the latest data-driven methods in achieving precise end-effector positioning, while it produces natural poses and temporally coherent motion. In addition, our system showcases robustness against on-the-fly constraint modifications, and exhibits exceptional adaptability to various input configurations and changes.
>
---
#### [replaced 135] ChitroJera: A Regionally Relevant Visual Question Answering Dataset for Bangla
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.14991v2](http://arxiv.org/pdf/2410.14991v2)**

> **作者:** Deeparghya Dutta Barua; Md Sakib Ul Rahman Sourove; Md Fahim; Fabiha Haider; Fariha Tanjim Shifat; Md Tasmim Rahman Adib; Anam Borhan Uddin; Md Farhan Ishmam; Md Farhad Alam
>
> **备注:** Accepted in ECML PKDD 2025
>
> **摘要:** Visual Question Answer (VQA) poses the problem of answering a natural language question about a visual context. Bangla, despite being a widely spoken language, is considered low-resource in the realm of VQA due to the lack of proper benchmarks, challenging models known to be performant in other languages. Furthermore, existing Bangla VQA datasets offer little regional relevance and are largely adapted from their foreign counterparts. To address these challenges, we introduce a large-scale Bangla VQA dataset, ChitroJera, totaling over 15k samples from diverse and locally relevant data sources. We assess the performance of text encoders, image encoders, multimodal models, and our novel dual-encoder models. The experiments reveal that the pre-trained dual-encoders outperform other models of their scale. We also evaluate the performance of current large vision language models (LVLMs) using prompt-based techniques, achieving the overall best performance. Given the underdeveloped state of existing datasets, we envision ChitroJera expanding the scope of Vision-Language tasks in Bangla.
>
---
#### [replaced 136] SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization
- **分类: cs.LG; cs.AI; cs.CV; cs.NE; cs.PF**

- **链接: [http://arxiv.org/pdf/2411.10958v5](http://arxiv.org/pdf/2411.10958v5)**

> **作者:** Jintao Zhang; Haofeng Huang; Pengle Zhang; Jia Wei; Jun Zhu; Jianfei Chen
>
> **备注:** @inproceedings{zhang2024sageattention2, title={Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization}, author={Zhang, Jintao and Huang, Haofeng and Zhang, Pengle and Wei, Jia and Zhu, Jun and Chen, Jianfei}, booktitle={International Conference on Machine Learning (ICML)}, year={2025} }
>
> **摘要:** Although quantization for linear layers has been widely used, its application to accelerate the attention process remains limited. To further enhance the efficiency of attention computation compared to SageAttention while maintaining precision, we propose SageAttention2, which utilizes significantly faster 4-bit matrix multiplication (Matmul) alongside additional precision-enhancing techniques. First, we propose to quantize matrices $(Q, K)$ to INT4 in a hardware-friendly thread-level granularity and quantize matrices $(\widetilde P, V)$ to FP8. Second, we propose a method to smooth $Q$, enhancing the accuracy of INT4 $QK^\top$. Third, we propose a two-level accumulation strategy for $\widetilde PV$ to enhance the accuracy of FP8 $\widetilde PV$. The operations per second (OPS) of SageAttention2 surpass FlashAttention2 and xformers by about 3x and 4.5x on RTX4090, respectively. Moreover, SageAttention2 matches the speed of FlashAttention3(fp8) on the Hopper GPUs, while delivering much higher accuracy. Comprehensive experiments confirm that our approach incurs negligible end-to-end metrics loss across diverse models, including those for language, image, and video generation. The code is available at https://github.com/thu-ml/SageAttention.
>
---
#### [replaced 137] MultiFlow: A unified deep learning framework for multi-vessel classification, segmentation and clustering of phase-contrast MRI validated on a multi-site single ventricle patient cohort
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.11993v2](http://arxiv.org/pdf/2502.11993v2)**

> **作者:** Tina Yao; Nicole St. Clair; Madeline Gong; Gabriel F. Miller; Jennifer A. Steeden; Rahul H. Rathod; Vivek Muthurangu; FORCE Investigators
>
> **备注:** 6 Figures
>
> **摘要:** We present a deep learning framework with two models for automated segmentation and large-scale flow phenotyping in a registry of single-ventricle patients. MultiFlowSeg simultaneously classifies and segments five key vessels, left and right pulmonary arteries, aorta, superior vena cava, and inferior vena cava, from velocity encoded phase-contrast magnetic resonance (PCMR) data. Trained on 260 CMR exams (5 PCMR scans per exam), it achieved an average Dice score of 0.91 on 50 unseen test cases. The method was then integrated into an automated pipeline where it processed over 5,500 registry exams without human assistance, in exams with all 5 vessels it achieved 98% classification and 90% segmentation accuracy. Flow curves from successful segmentations were used to train MultiFlowDTC, which applied deep temporal clustering to identify distinct flow phenotypes. Survival analysis revealed distinct phenotypes were significantly associated with increased risk of death/transplantation and liver disease, demonstrating the potential of the framework.
>
---
#### [replaced 138] Towards Modality Generalization: A Benchmark and Prospective Analysis
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.18277v2](http://arxiv.org/pdf/2412.18277v2)**

> **作者:** Xiaohao Liu; Xiaobo Xia; Zhuo Huang; See-Kiong Ng; Tat-Seng Chua
>
> **备注:** under-review
>
> **摘要:** Multi-modal learning has achieved remarkable success by integrating information from various modalities, achieving superior performance in tasks like recognition and retrieval compared to uni-modal approaches. However, real-world scenarios often present novel modalities that are unseen during training due to resource and privacy constraints, a challenge current methods struggle to address. This paper introduces Modality Generalization (MG), which focuses on enabling models to generalize to unseen modalities. We define two cases: Weak MG, where both seen and unseen modalities can be mapped into a joint embedding space via existing perceptors, and Strong MG, where no such mappings exist. To facilitate progress, we propose a comprehensive benchmark featuring multi-modal algorithms and adapt existing methods that focus on generalization. Extensive experiments highlight the complexity of MG, exposing the limitations of existing methods and identifying key directions for future research. Our work provides a foundation for advancing robust and adaptable multi-modal models, enabling them to handle unseen modalities in realistic scenarios.
>
---
#### [replaced 139] Exploring Model Kinship for Merging Large Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2410.12613v2](http://arxiv.org/pdf/2410.12613v2)**

> **作者:** Yedi Hu; Yunzhi Yao; Shumin Deng; Huajun Chen; Ningyu Zhang
>
> **备注:** Ongoing work
>
> **摘要:** Model merging has become one of the key technologies for enhancing the capabilities and efficiency of Large Language Models (LLMs). However, our understanding of the expected performance gains and principles when merging any two models remains limited. In this work, we introduce model kinship, the degree of similarity or relatedness between LLMs, analogous to biological evolution. With comprehensive empirical analysis, we find that there is a certain relationship between model kinship and the performance gains after model merging, which can help guide our selection of candidate models. Inspired by this, we propose a new model merging strategy: Top-k Greedy Merging with Model Kinship, which can yield better performance on benchmark datasets. Specifically, we discover that using model kinship as a criterion can assist us in continuously performing model merging, alleviating the degradation (local optima) in model evolution, whereas model kinship can serve as a guide to escape these traps. Code is available at https://github.com/zjunlp/ModelKinship.
>
---
#### [replaced 140] Distill CLIP (DCLIP): Enhancing Image-Text Retrieval via Cross-Modal Transformer Distillation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21549v3](http://arxiv.org/pdf/2505.21549v3)**

> **作者:** Daniel Csizmadia; Andrei Codreanu; Victor Sim; Vighnesh Prabhu; Michael Lu; Kevin Zhu; Sean O'Brien; Vasu Sharma
>
> **摘要:** We present Distill CLIP (DCLIP), a fine-tuned variant of the CLIP model that enhances multimodal image-text retrieval while preserving the original model's strong zero-shot classification capabilities. CLIP models are typically constrained by fixed image resolutions and limited context, which can hinder their effectiveness in retrieval tasks that require fine-grained cross-modal understanding. DCLIP addresses these challenges through a meta teacher-student distillation framework, where a cross-modal transformer teacher is fine-tuned to produce enriched embeddings via bidirectional cross-attention between YOLO-extracted image regions and corresponding textual spans. These semantically and spatially aligned global representations guide the training of a lightweight student model using a hybrid loss that combines contrastive learning and cosine similarity objectives. Despite being trained on only ~67,500 samples curated from MSCOCO, Flickr30k, and Conceptual Captions-just a fraction of CLIP's original dataset-DCLIP significantly improves image-text retrieval metrics (Recall@K, MAP), while retaining approximately 94% of CLIP's zero-shot classification performance. These results demonstrate that DCLIP effectively mitigates the trade-off between task specialization and generalization, offering a resource-efficient, domain-adaptive, and detail-sensitive solution for advanced vision-language tasks. Code available at https://anonymous.4open.science/r/DCLIP-B772/README.md.
>
---
#### [replaced 141] FMNet: Frequency-Assisted Mamba-Like Linear Attention Network for Camouflaged Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.11030v2](http://arxiv.org/pdf/2503.11030v2)**

> **作者:** Ming Deng; Sijin Sun; Zihao Li; Xiaochuan Hu; Xing Wu
>
> **摘要:** Camouflaged Object Detection (COD) is challenging due to the strong similarity between camouflaged objects and their surroundings, which complicates identification. Existing methods mainly rely on spatial local features, failing to capture global information, while Transformers increase computational costs. To address this, the Frequency-Assisted Mamba-Like Linear Attention Network (FMNet) is proposed, which leverages frequency-domain learning to efficiently capture global features and mitigate ambiguity between objects and the background. FMNet introduces the Multi-Scale Frequency-Assisted Mamba-Like Linear Attention (MFM) module, integrating frequency and spatial features through a multi-scale structure to handle scale variations while reducing computational complexity. Additionally, the Pyramidal Frequency Attention Extraction (PFAE) module and the Frequency Reverse Decoder (FRD) enhance semantics and reconstruct features. Experimental results demonstrate that FMNet outperforms existing methods on multiple COD datasets, showcasing its advantages in both performance and efficiency. Code available at https://github.com/Chranos/FMNet.
>
---
#### [replaced 142] ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18668v2](http://arxiv.org/pdf/2505.18668v2)**

> **作者:** Zhen Li; Duan Li; Yukai Guo; Xinyuan Guo; Bowen Li; Lanxi Xiao; Shenyu Qiao; Jiashu Chen; Zijian Wu; Hui Zhang; Xinhuan Shu; Shixia Liu
>
> **备注:** 56 pages
>
> **摘要:** Infographic charts are a powerful medium for communicating abstract data by combining visual elements (e.g., charts, images) with textual information. However, their visual and structural richness poses challenges for large vision-language models (LVLMs), which are typically trained on plain charts. To bridge this gap, we introduce ChartGalaxy, a million-scale dataset designed to advance the understanding and generation of infographic charts. The dataset is constructed through an inductive process that identifies 75 chart types, 330 chart variations, and 68 layout templates from real infographic charts and uses them to create synthetic ones programmatically. We showcase the utility of this dataset through: 1) improving infographic chart understanding via fine-tuning, 2) benchmarking code generation for infographic charts, and 3) enabling example-based infographic chart generation. By capturing the visual and structural complexity of real design, ChartGalaxy provides a useful resource for enhancing multimodal reasoning and generation in LVLMs.
>
---
#### [replaced 143] GSBA$^K$: $top$-$K$ Geometric Score-based Black-box Attack
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12827v3](http://arxiv.org/pdf/2503.12827v3)**

> **作者:** Md Farhamdur Reza; Richeng Jin; Tianfu Wu; Huaiyu Dai
>
> **备注:** License changed to CC BY 4.0 to align with ICLR 2025. No changes to content. Published at: https://openreview.net/forum?id=htX7AoHyln
>
> **摘要:** Existing score-based adversarial attacks mainly focus on crafting $top$-1 adversarial examples against classifiers with single-label classification. Their attack success rate and query efficiency are often less than satisfactory, particularly under small perturbation requirements; moreover, the vulnerability of classifiers with multi-label learning is yet to be studied. In this paper, we propose a comprehensive surrogate free score-based attack, named \b geometric \b score-based \b black-box \b attack (GSBA$^K$), to craft adversarial examples in an aggressive $top$-$K$ setting for both untargeted and targeted attacks, where the goal is to change the $top$-$K$ predictions of the target classifier. We introduce novel gradient-based methods to find a good initial boundary point to attack. Our iterative method employs novel gradient estimation techniques, particularly effective in $top$-$K$ setting, on the decision boundary to effectively exploit the geometry of the decision boundary. Additionally, GSBA$^K$ can be used to attack against classifiers with $top$-$K$ multi-label learning. Extensive experimental results on ImageNet and PASCAL VOC datasets validate the effectiveness of GSBA$^K$ in crafting $top$-$K$ adversarial examples.
>
---
#### [replaced 144] Flex3D: Feed-Forward 3D Generation with Flexible Reconstruction Model and Input View Curation
- **分类: cs.CV; cs.GR; eess.IV**

- **链接: [http://arxiv.org/pdf/2410.00890v3](http://arxiv.org/pdf/2410.00890v3)**

> **作者:** Junlin Han; Jianyuan Wang; Andrea Vedaldi; Philip Torr; Filippos Kokkinos
>
> **备注:** ICML 25. Project page: https://junlinhan.github.io/projects/flex3d/
>
> **摘要:** Generating high-quality 3D content from text, single images, or sparse view images remains a challenging task with broad applications. Existing methods typically employ multi-view diffusion models to synthesize multi-view images, followed by a feed-forward process for 3D reconstruction. However, these approaches are often constrained by a small and fixed number of input views, limiting their ability to capture diverse viewpoints and, even worse, leading to suboptimal generation results if the synthesized views are of poor quality. To address these limitations, we propose Flex3D, a novel two-stage framework capable of leveraging an arbitrary number of high-quality input views. The first stage consists of a candidate view generation and curation pipeline. We employ a fine-tuned multi-view image diffusion model and a video diffusion model to generate a pool of candidate views, enabling a rich representation of the target 3D object. Subsequently, a view selection pipeline filters these views based on quality and consistency, ensuring that only the high-quality and reliable views are used for reconstruction. In the second stage, the curated views are fed into a Flexible Reconstruction Model (FlexRM), built upon a transformer architecture that can effectively process an arbitrary number of inputs. FlemRM directly outputs 3D Gaussian points leveraging a tri-plane representation, enabling efficient and detailed 3D generation. Through extensive exploration of design and training strategies, we optimize FlexRM to achieve superior performance in both reconstruction and generation tasks. Our results demonstrate that Flex3D achieves state-of-the-art performance, with a user study winning rate of over 92% in 3D generation tasks when compared to several of the latest feed-forward 3D generative models.
>
---
#### [replaced 145] Domain-Agnostic Stroke Lesion Segmentation Using Physics-Constrained Synthetic Data
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2412.03318v3](http://arxiv.org/pdf/2412.03318v3)**

> **作者:** Liam Chalcroft; Jenny Crinion; Cathy J. Price; John Ashburner
>
> **摘要:** Segmenting stroke lesions in MRI is challenging due to diverse acquisition protocols that limit model generalisability. In this work, we introduce two physics-constrained approaches to generate synthetic quantitative MRI (qMRI) images that improve segmentation robustness across heterogeneous domains. Our first method, $\texttt{qATLAS}$, trains a neural network to estimate qMRI maps from standard MPRAGE images, enabling the simulation of varied MRI sequences with realistic tissue contrasts. The second method, $\texttt{qSynth}$, synthesises qMRI maps directly from tissue labels using label-conditioned Gaussian mixture models, ensuring physical plausibility. Extensive experiments on multiple out-of-domain datasets show that both methods outperform a baseline UNet, with $\texttt{qSynth}$ notably surpassing previous synthetic data approaches. These results highlight the promise of integrating MRI physics into synthetic data generation for robust, generalisable stroke lesion segmentation. Code is available at https://github.com/liamchalcroft/qsynth
>
---
#### [replaced 146] TextDestroyer: A Training- and Annotation-Free Diffusion Method for Destroying Anomal Text from Images
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.00355v2](http://arxiv.org/pdf/2411.00355v2)**

> **作者:** Mengcheng Li; Fei Chao; Chia-Wen Lin; Rongrong Ji
>
> **摘要:** In this paper, we propose TextDestroyer, the first training- and annotation-free method for scene text destruction using a pre-trained diffusion model. Existing scene text removal models require complex annotation and retraining, and may leave faint yet recognizable text information, compromising privacy protection and content concealment. TextDestroyer addresses these issues by employing a three-stage hierarchical process to obtain accurate text masks. Our method scrambles text areas in the latent start code using a Gaussian distribution before reconstruction. During the diffusion denoising process, self-attention key and value are referenced from the original latent to restore the compromised background. Latent codes saved at each inversion step are used for replacement during reconstruction, ensuring perfect background restoration. The advantages of TextDestroyer include: (1) it eliminates labor-intensive data annotation and resource-intensive training; (2) it achieves more thorough text destruction, preventing recognizable traces; and (3) it demonstrates better generalization capabilities, performing well on both real-world scenes and generated images.
>
---
#### [replaced 147] SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.04301v4](http://arxiv.org/pdf/2412.04301v4)**

> **作者:** Trong-Tung Nguyen; Quang Nguyen; Khoi Nguyen; Anh Tran; Cuong Pham
>
> **备注:** 17 pages, 15 figures
>
> **摘要:** Recent advances in text-guided image editing enable users to perform image edits through simple text inputs, leveraging the extensive priors of multi-step diffusion-based text-to-image models. However, these methods often fall short of the speed demands required for real-world and on-device applications due to the costly multi-step inversion and sampling process involved. In response to this, we introduce SwiftEdit, a simple yet highly efficient editing tool that achieve instant text-guided image editing (in 0.23s). The advancement of SwiftEdit lies in its two novel contributions: a one-step inversion framework that enables one-step image reconstruction via inversion and a mask-guided editing technique with our proposed attention rescaling mechanism to perform localized image editing. Extensive experiments are provided to demonstrate the effectiveness and efficiency of SwiftEdit. In particular, SwiftEdit enables instant text-guided image editing, which is extremely faster than previous multi-step methods (at least 50 times faster) while maintain a competitive performance in editing results. Our project page is at: https://swift-edit.github.io/
>
---
#### [replaced 148] Subpixel Edge Localization Based on Converted Intensity Summation under Stable Edge Region
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16502v2](http://arxiv.org/pdf/2502.16502v2)**

> **作者:** Yingyuan Yang; Guoyuan Liang; Xianwen Wang; Kaiming Wang; Can Wang; Xinyu Wu
>
> **摘要:** To satisfy the rigorous requirements of precise edge detection in critical high-accuracy measurements, this article proposes a series of efficient approaches for localizing subpixel edge. In contrast to the fitting based methods, which consider pixel intensity as a sample value derived from a specific model. We take an innovative perspective by assuming that the intensity at the pixel level can be interpreted as a local integral mapping in the intensity model for subpixel localization. Consequently, we propose a straightforward subpixel edge localization method called Converted Intensity Summation (CIS). To address the limited robustness associated with focusing solely on the localization of individual edge points, a Stable Edge Region (SER) based algorithm is presented to alleviate local interference near edges. Given the observation that the consistency of edge statistics exists in the local region, the algorithm seeks correlated stable regions in the vicinity of edges to facilitate the acquisition of robust parameters and achieve higher precision positioning. In addition, an edge complement method based on extension-adjustment is also introduced to rectify the irregular edges through the efficient migration of SERs. A large number of experiments are conducted on both synthetic and real image datasets which cover common edge patterns as well as various real scenarios such as industrial PCB images, remote sensing and medical images. It is verified that CIS can achieve higher accuracy than the state-of-the-art method, while requiring less execution time. Moreover, by integrating SER into CIS, the proposed algorithm demonstrates excellent performance in further improving the anti-interference capability and positioning accuracy.
>
---
#### [replaced 149] CogAD: Cognitive-Hierarchy Guided End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21581v2](http://arxiv.org/pdf/2505.21581v2)**

> **作者:** Zhennan Wang; Jianing Teng; Canqun Xiang; Kangliang Chen; Xing Pan; Lu Deng; Weihao Gu
>
> **摘要:** While end-to-end autonomous driving has advanced significantly, prevailing methods remain fundamentally misaligned with human cognitive principles in both perception and planning. In this paper, we propose CogAD, a novel end-to-end autonomous driving model that emulates the hierarchical cognition mechanisms of human drivers. CogAD implements dual hierarchical mechanisms: global-to-local context processing for human-like perception and intent-conditioned multi-mode trajectory generation for cognitively-inspired planning. The proposed method demonstrates three principal advantages: comprehensive environmental understanding through hierarchical perception, robust planning exploration enabled by multi-level planning, and diverse yet reasonable multi-modal trajectory generation facilitated by dual-level uncertainty modeling. Extensive experiments on nuScenes and Bench2Drive demonstrate that CogAD achieves state-of-the-art performance in end-to-end planning, exhibiting particular superiority in long-tail scenarios and robust generalization to complex real-world driving conditions.
>
---
#### [replaced 150] OpenGait: A Comprehensive Benchmark Study for Gait Recognition towards Better Practicality
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.09138v2](http://arxiv.org/pdf/2405.09138v2)**

> **作者:** Chao Fan; Saihui Hou; Junhao Liang; Chuanfu Shen; Jingzhe Ma; Dongyang Jin; Yongzhen Huang; Shiqi Yu
>
> **摘要:** Gait recognition, a rapidly advancing vision technology for person identification from a distance, has made significant strides in indoor settings. However, evidence suggests that existing methods often yield unsatisfactory results when applied to newly released real-world gait datasets. Furthermore, conclusions drawn from indoor gait datasets may not easily generalize to outdoor ones. Therefore, the primary goal of this paper is to present a comprehensive benchmark study aimed at improving practicality rather than solely focusing on enhancing performance. To this end, we developed OpenGait, a flexible and efficient gait recognition platform. Using OpenGait, we conducted in-depth ablation experiments to revisit recent developments in gait recognition. Surprisingly, we detected some imperfect parts of some prior methods and thereby uncovered several critical yet previously neglected insights. These findings led us to develop three structurally simple yet empirically powerful and practically robust baseline models: DeepGaitV2, SkeletonGait, and SkeletonGait++, which represent the appearance-based, model-based, and multi-modal methodologies for gait pattern description, respectively. In addition to achieving state-of-the-art performance, our careful exploration provides new perspectives on the modeling experience of deep gait models and the representational capacity of typical gait modalities. In the end, we discuss the key trends and challenges in current gait recognition, aiming to inspire further advancements towards better practicality. The code is available at https://github.com/ShiqiYu/OpenGait.
>
---
#### [replaced 151] Advancing Image Super-resolution Techniques in Remote Sensing: A Comprehensive Survey
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23248v2](http://arxiv.org/pdf/2505.23248v2)**

> **作者:** Yunliang Qi; Meng Lou; Yimin Liu; Lu Li; Zhen Yang; Wen Nie
>
> **备注:** A survey of Remote Sensing Super-resolution Techniques
>
> **摘要:** Remote sensing image super-resolution (RSISR) is a crucial task in remote sensing image processing, aiming to reconstruct high-resolution (HR) images from their low-resolution (LR) counterparts. Despite the growing number of RSISR methods proposed in recent years, a systematic and comprehensive review of these methods is still lacking. This paper presents a thorough review of RSISR algorithms, covering methodologies, datasets, and evaluation metrics. We provide an in-depth analysis of RSISR methods, categorizing them into supervised, unsupervised, and quality evaluation approaches, to help researchers understand current trends and challenges. Our review also discusses the strengths, limitations, and inherent challenges of these techniques. Notably, our analysis reveals significant limitations in existing methods, particularly in preserving fine-grained textures and geometric structures under large-scale degradation. Based on these findings, we outline future research directions, highlighting the need for domain-specific architectures and robust evaluation protocols to bridge the gap between synthetic and real-world RSISR scenarios.
>
---
#### [replaced 152] In the Picture: Medical Imaging Datasets, Artifacts, and their Living Review
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2501.10727v2](http://arxiv.org/pdf/2501.10727v2)**

> **作者:** Amelia Jiménez-Sánchez; Natalia-Rozalia Avlona; Sarah de Boer; Víctor M. Campello; Aasa Feragen; Enzo Ferrante; Melanie Ganz; Judy Wawira Gichoya; Camila González; Steff Groefsema; Alessa Hering; Adam Hulman; Leo Joskowicz; Dovile Juodelyte; Melih Kandemir; Thijs Kooi; Jorge del Pozo Lérida; Livie Yumeng Li; Andre Pacheco; Tim Rädsch; Mauricio Reyes; Théo Sourget; Bram van Ginneken; David Wen; Nina Weng; Jack Junchi Xu; Hubert Dariusz Zając; Maria A. Zuluaga; Veronika Cheplygina
>
> **备注:** ACM Conference on Fairness, Accountability, and Transparency - FAccT 2025
>
> **摘要:** Datasets play a critical role in medical imaging research, yet issues such as label quality, shortcuts, and metadata are often overlooked. This lack of attention may harm the generalizability of algorithms and, consequently, negatively impact patient outcomes. While existing medical imaging literature reviews mostly focus on machine learning (ML) methods, with only a few focusing on datasets for specific applications, these reviews remain static -- they are published once and not updated thereafter. This fails to account for emerging evidence, such as biases, shortcuts, and additional annotations that other researchers may contribute after the dataset is published. We refer to these newly discovered findings of datasets as research artifacts. To address this gap, we propose a living review that continuously tracks public datasets and their associated research artifacts across multiple medical imaging applications. Our approach includes a framework for the living review to monitor data documentation artifacts, and an SQL database to visualize the citation relationships between research artifact and dataset. Lastly, we discuss key considerations for creating medical imaging datasets, review best practices for data annotation, discuss the significance of shortcuts and demographic diversity, and emphasize the importance of managing datasets throughout their entire lifecycle. Our demo is publicly available at http://inthepicture.itu.dk/.
>
---
#### [replaced 153] EasyREG: Easy Depth-Based Markerless Registration and Tracking using Augmented Reality Device for Surgical Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09498v2](http://arxiv.org/pdf/2504.09498v2)**

> **作者:** Yue Yang; Christoph Leuze; Brian Hargreaves; Bruce Daniel; Fred Baik
>
> **摘要:** The use of Augmented Reality (AR) devices for surgical guidance has gained increasing traction in the medical field. Traditional registration methods often rely on external fiducial markers to achieve high accuracy and real-time performance. However, these markers introduce cumbersome calibration procedures and can be challenging to deploy in clinical settings. While commercial solutions have attempted real-time markerless tracking using the native RGB cameras of AR devices, their accuracy remains questionable for medical guidance, primarily due to occlusions and significant outliers between the live sensor data and the preoperative target anatomy point cloud derived from MRI or CT scans. In this work, we present a markerless framework that relies only on the depth sensor of AR devices and consists of two modules: a registration module for high-precision, outlier-robust target anatomy localization, and a tracking module for real-time pose estimation. The registration module integrates depth sensor error correction, a human-in-the-loop region filtering technique, and a robust global alignment with curvature-aware feature sampling, followed by local ICP refinement, for markerless alignment of preoperative models with patient anatomy. The tracking module employs a fast and robust registration algorithm that uses the initial pose from the registration module to estimate the target pose in real-time. We comprehensively evaluated the performance of both modules through simulation and real-world measurements. The results indicate that our markerless system achieves superior performance for registration and comparable performance for tracking to industrial solutions. The two-module design makes our system a one-stop solution for surgical procedures where the target anatomy moves or stays static during surgery.
>
---
#### [replaced 154] ReelWave: Multi-Agentic Movie Sound Generation through Multimodal LLM Conversation
- **分类: cs.SD; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07217v3](http://arxiv.org/pdf/2503.07217v3)**

> **作者:** Zixuan Wang; Chi-Keung Tang; Yu-Wing Tai
>
> **备注:** Project page: https://vincent2311.github.io/ReelWave_demo
>
> **摘要:** Current audio generation conditioned by text or video focuses on aligning audio with text/video modalities. Despite excellent alignment results, these multimodal frameworks still cannot be directly applied to compelling movie storytelling involving multiple scenes, where "on-screen" sounds require temporally-aligned audio generation, while "off-screen" sounds contribute to appropriate environment sounds accompanied by background music when applicable. Inspired by professional movie production, this paper proposes a multi-agentic framework for audio generation supervised by an autonomous Sound Director agent, engaging multi-turn conversations with other agents for on-screen and off-screen sound generation through multimodal LLM. To address on-screen sound generation, after detecting any talking humans in videos, we capture semantically and temporally synchronized sound by training a prediction model that forecasts interpretable, time-varying audio control signals: loudness, pitch, and timbre, which are used by a Foley Artist agent to condition a cross-attention module in the sound generation. The Foley Artist works cooperatively with the Composer and Voice Actor agents, and together they autonomously generate off-screen sound to complement the overall production. Each agent takes on specific roles similar to those of a movie production team. To temporally ground audio language models, in ReelWave, text/video conditions are decomposed into atomic, specific sound generation instructions synchronized with visuals when applicable. Consequently, our framework can generate rich and relevant audio content conditioned on video clips extracted from movies.
>
---
#### [replaced 155] A Survey on Event-driven 3D Reconstruction: Development under Different Categories
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19753v3](http://arxiv.org/pdf/2503.19753v3)**

> **作者:** Chuanzhi Xu; Haoxian Zhou; Haodong Chen; Vera Chung; Qiang Qu
>
> **备注:** We have decided not to submit this article and plan to withdraw it from public display. The content of this article will be presented in a more comprehensive form in another work
>
> **摘要:** Event cameras have gained increasing attention for 3D reconstruction due to their high temporal resolution, low latency, and high dynamic range. They capture per-pixel brightness changes asynchronously, allowing accurate reconstruction under fast motion and challenging lighting conditions. In this survey, we provide a comprehensive review of event-driven 3D reconstruction methods, including stereo, monocular, and multimodal systems. We further categorize recent developments based on geometric, learning-based, and hybrid approaches. Emerging trends, such as neural radiance fields and 3D Gaussian splatting with event data, are also covered. The related works are structured chronologically to illustrate the innovations and progression within the field. To support future research, we also highlight key research gaps and future research directions in dataset, experiment, evaluation, event representation, etc.
>
---
#### [replaced 156] Survey on Vision-Language-Action Models
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06851v3](http://arxiv.org/pdf/2502.06851v3)**

> **作者:** Adilzhan Adilkhanov; Amir Yelenov; Assylkhan Seitzhanov; Ayan Mazhitov; Azamat Abdikarimov; Danissa Sandykbayeva; Daryn Kenzhebek; Dinmukhammed Mukashev; Ilyas Umurbekov; Jabrail Chumakov; Kamila Spanova; Karina Burunchina; Madina Yergibay; Margulan Issa; Moldir Zabirova; Nurdaulet Zhuzbay; Nurlan Kabdyshev; Nurlan Zhaniyar; Rasul Yermagambet; Rustam Chibar; Saltanat Seitzhan; Soibkhon Khajikhanov; Tasbolat Taunyazov; Temirlan Galimzhanov; Temirlan Kaiyrbay; Tleukhan Mussin; Togzhan Syrymova; Valeriya Kostyukova; Yerkebulan Massalim; Yermakhan Kassym; Zerde Nurbayeva; Zhanat Kappassov
>
> **备注:** arXiv admin note: This submission has been withdrawn due to serious violation of arXiv policies for acceptable submissions
>
> **摘要:** This paper presents an AI-generated review of Vision-Language-Action (VLA) models, summarizing key methodologies, findings, and future directions. The content is produced using large language models (LLMs) and is intended only for demonstration purposes. This work does not represent original research, but highlights how AI can help automate literature reviews. As AI-generated content becomes more prevalent, ensuring accuracy, reliability, and proper synthesis remains a challenge. Future research will focus on developing a structured framework for AI-assisted literature reviews, exploring techniques to enhance citation accuracy, source credibility, and contextual understanding. By examining the potential and limitations of LLM in academic writing, this study aims to contribute to the broader discussion of integrating AI into research workflows. This work serves as a preliminary step toward establishing systematic approaches for leveraging AI in literature review generation, making academic knowledge synthesis more efficient and scalable.
>
---
#### [replaced 157] Enhancing Multimodal Unified Representations for Cross Modal Generalization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.05168v3](http://arxiv.org/pdf/2403.05168v3)**

> **作者:** Hai Huang; Yan Xia; Shengpeng Ji; Shulei Wang; Hanting Wang; Minghui Fang; Jieming Zhu; Zhenhua Dong; Sashuai Zhou; Zhou Zhao
>
> **摘要:** To enhance the interpretability of multimodal unified representations, many studies have focused on discrete unified representations. These efforts typically start with contrastive learning and gradually extend to the disentanglement of modal information, achieving solid multimodal discrete unified representations. However, existing research often overlooks two critical issues: 1) The use of Euclidean distance for quantization in discrete representations often overlooks the important distinctions among different dimensions of features, resulting in redundant representations after quantization; 2) Different modalities have unique characteristics, and a uniform alignment approach does not fully exploit these traits. To address these issues, we propose Training-free Optimization of Codebook (TOC) and Fine and Coarse cross-modal Information Disentangling (FCID). These methods refine the unified discrete representations from pretraining and perform fine- and coarse-grained information disentanglement tailored to the specific characteristics of each modality, achieving significant performance improvements over previous state-of-the-art models. The code is available at https://github.com/haihuangcode/CMG.
>
---
#### [replaced 158] AdaWorld: Learning Adaptable World Models with Latent Actions
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.18938v4](http://arxiv.org/pdf/2503.18938v4)**

> **作者:** Shenyuan Gao; Siyuan Zhou; Yilun Du; Jun Zhang; Chuang Gan
>
> **备注:** ICML 2025. Project page: https://adaptable-world-model.github.io/, code: https://github.com/Little-Podi/AdaWorld, model: https://huggingface.co/Little-Podi/AdaWorld
>
> **摘要:** World models aim to learn action-controlled future prediction and have proven essential for the development of intelligent agents. However, most existing world models rely heavily on substantial action-labeled data and costly training, making it challenging to adapt to novel environments with heterogeneous actions through limited interactions. This limitation can hinder their applicability across broader domains. To overcome this limitation, we propose AdaWorld, an innovative world model learning approach that enables efficient adaptation. The key idea is to incorporate action information during the pretraining of world models. This is achieved by extracting latent actions from videos in a self-supervised manner, capturing the most critical transitions between frames. We then develop an autoregressive world model that conditions on these latent actions. This learning paradigm enables highly adaptable world models, facilitating efficient transfer and learning of new actions even with limited interactions and finetuning. Our comprehensive experiments across multiple environments demonstrate that AdaWorld achieves superior performance in both simulation quality and visual planning.
>
---
#### [replaced 159] Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.00420v2](http://arxiv.org/pdf/2504.00420v2)**

> **作者:** Yuanqi Yao; Siao Liu; Haoming Song; Delin Qu; Qizhi Chen; Yan Ding; Bin Zhao; Zhigang Wang; Xuelong Li; Dong Wang
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Building a lifelong robot that can effectively leverage prior knowledge for continuous skill acquisition remains significantly challenging. Despite the success of experience replay and parameter-efficient methods in alleviating catastrophic forgetting problem, naively applying these methods causes a failure to leverage the shared primitives between skills. To tackle these issues, we propose Primitive Prompt Learning (PPL), to achieve lifelong robot manipulation via reusable and extensible primitives. Within our two stage learning scheme, we first learn a set of primitive prompts to represent shared primitives through multi-skills pre-training stage, where motion-aware prompts are learned to capture semantic and motion shared primitives across different skills. Secondly, when acquiring new skills in lifelong span, new prompts are appended and optimized with frozen pretrained prompts, boosting the learning via knowledge transfer from old skills to new ones. For evaluation, we construct a large-scale skill dataset and conduct extensive experiments in both simulation and real-world tasks, demonstrating PPL's superior performance over state-of-the-art methods.
>
---
#### [replaced 160] VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20279v2](http://arxiv.org/pdf/2505.20279v2)**

> **作者:** Zhiwen Fan; Jian Zhang; Renjie Li; Junge Zhang; Runjin Chen; Hezhen Hu; Kevin Wang; Huaizhi Qu; Dilin Wang; Zhicheng Yan; Hongyu Xu; Justin Theiss; Tianlong Chen; Jiachen Li; Zhengzhong Tu; Zhangyang Wang; Rakesh Ranjan
>
> **备注:** Project Page: https://vlm-3r.github.io/
>
> **摘要:** The rapid advancement of Large Multimodal Models (LMMs) for 2D images and videos has motivated extending these models to understand 3D scenes, aiming for human-like visual-spatial intelligence. Nevertheless, achieving deep spatial understanding comparable to human capabilities poses significant challenges in model encoding and data acquisition. Existing methods frequently depend on external depth sensors for geometry capture or utilize off-the-shelf algorithms for pre-constructing 3D maps, thereby limiting their scalability, especially with prevalent monocular video inputs and for time-sensitive applications. In this work, we introduce VLM-3R, a unified framework for Vision-Language Models (VLMs) that incorporates 3D Reconstructive instruction tuning. VLM-3R processes monocular video frames by employing a geometry encoder to derive implicit 3D tokens that represent spatial understanding. Leveraging our Spatial-Visual-View Fusion and over 200K curated 3D reconstructive instruction tuning question-answer (QA) pairs, VLM-3R effectively aligns real-world spatial context with language instructions. This enables monocular 3D spatial assistance and embodied reasoning. To facilitate the evaluation of temporal reasoning, we introduce the Vision-Spatial-Temporal Intelligence benchmark, featuring over 138.6K QA pairs across five distinct tasks focused on evolving spatial relationships. Extensive experiments demonstrate that our model, VLM-3R, not only facilitates robust visual-spatial reasoning but also enables the understanding of temporal 3D context changes, excelling in both accuracy and scalability.
>
---
#### [replaced 161] Point Cloud Mixture-of-Domain-Experts Model for 3D Self-supervised Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.09886v2](http://arxiv.org/pdf/2410.09886v2)**

> **作者:** Yaohua Zha; Tao Dai; Hang Guo; Yanzi Wang; Bin Chen; Ke Chen; Shu-Tao Xia
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Point clouds, as a primary representation of 3D data, can be categorized into scene domain point clouds and object domain point clouds. Point cloud self-supervised learning (SSL) has become a mainstream paradigm for learning 3D representations. However, existing point cloud SSL primarily focuses on learning domain-specific 3D representations within a single domain, neglecting the complementary nature of cross-domain knowledge, which limits the learning of 3D representations. In this paper, we propose to learn a comprehensive Point cloud Mixture-of-Domain-Experts model (Point-MoDE) via a block-to-scene pre-training strategy. Specifically, we first propose a mixture-of-domain-expert model consisting of scene domain experts and multiple shared object domain experts. Furthermore, we propose a block-to-scene pretraining strategy, which leverages the features of point blocks in the object domain to regress their initial positions in the scene domain through object-level block mask reconstruction and scene-level block position regression. By integrating the complementary knowledge between object and scene, this strategy simultaneously facilitates the learning of both object-domain and scene-domain representations, leading to a more comprehensive 3D representation. Extensive experiments in downstream tasks demonstrate the superiority of our model.
>
---
#### [replaced 162] SpatialLLM: A Compound 3D-Informed Design towards Spatially-Intelligent Large Multimodal Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.00788v2](http://arxiv.org/pdf/2505.00788v2)**

> **作者:** Wufei Ma; Luoxin Ye; Nessa McWeeney; Celso M de Melo; Jieneng Chen; Alan Yuille
>
> **备注:** CVPR 2025 highlight
>
> **摘要:** Humans naturally understand 3D spatial relationships, enabling complex reasoning like predicting collisions of vehicles from different directions. Current large multimodal models (LMMs), however, lack of this capability of 3D spatial reasoning. This limitation stems from the scarcity of 3D training data and the bias in current model designs toward 2D data. In this paper, we systematically study the impact of 3D-informed data, architecture, and training setups, introducing SpatialLLM, a large multi-modal model with advanced 3D spatial reasoning abilities. To address data limitations, we develop two types of 3D-informed training datasets: (1) 3D-informed probing data focused on object's 3D location and orientation, and (2) 3D-informed conversation data for complex spatial relationships. Notably, we are the first to curate VQA data that incorporate 3D orientation relationships on real images. Furthermore, we systematically integrate these two types of training data with the architectural and training designs of LMMs, providing a roadmap for optimal design aimed at achieving superior 3D reasoning capabilities. Our SpatialLLM advances machines toward highly capable 3D-informed reasoning, surpassing GPT-4o performance by 8.7%. Our systematic empirical design and the resulting findings offer valuable insights for future research in this direction.
>
---
#### [replaced 163] LLaVA-ST: A Multimodal Large Language Model for Fine-Grained Spatial-Temporal Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08282v2](http://arxiv.org/pdf/2501.08282v2)**

> **作者:** Hongyu Li; Jinyu Chen; Ziyu Wei; Shaofei Huang; Tianrui Hui; Jialin Gao; Xiaoming Wei; Si Liu
>
> **备注:** Accepted by CVPR2025
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have shown promising results, yet existing approaches struggle to effectively handle both temporal and spatial localization simultaneously. This challenge stems from two key issues: first, incorporating spatial-temporal localization introduces a vast number of coordinate combinations, complicating the alignment of linguistic and visual coordinate representations; second, encoding fine-grained temporal and spatial information during video feature compression is inherently difficult. To address these issues, we propose LLaVA-ST, a MLLM for fine-grained spatial-temporal multimodal understanding. In LLaVA-ST, we propose Language-Aligned Positional Embedding, which embeds the textual coordinate special token into the visual space, simplifying the alignment of fine-grained spatial-temporal correspondences. Additionally, we design the Spatial-Temporal Packer, which decouples the feature compression of temporal and spatial resolutions into two distinct point-to-region attention processing streams. Furthermore, we propose ST-Align dataset with 4.3M training samples for fine-grained spatial-temporal multimodal understanding. With ST-align, we present a progressive training pipeline that aligns the visual and textual feature through sequential coarse-to-fine stages.Additionally, we introduce an ST-Align benchmark to evaluate spatial-temporal interleaved fine-grained understanding tasks, which include Spatial-Temporal Video Grounding (STVG) , Event Localization and Captioning (ELC) and Spatial Video Grounding (SVG). LLaVA-ST achieves outstanding performance on 11 benchmarks requiring fine-grained temporal, spatial, or spatial-temporal interleaving multimodal understanding. Our code, data and benchmark will be released at Our code, data and benchmark will be released at https://github.com/appletea233/LLaVA-ST .
>
---
#### [replaced 164] One RL to See Them All: Visual Triple Unified Reinforcement Learning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18129v2](http://arxiv.org/pdf/2505.18129v2)**

> **作者:** Yan Ma; Linge Du; Xuyang Shen; Shaoxiang Chen; Pengfei Li; Qibing Ren; Lizhuang Ma; Yuchao Dai; Pengfei Liu; Junjie Yan
>
> **备注:** Technical Report
>
> **摘要:** Reinforcement learning (RL) has significantly advanced the reasoning capabilities of vision-language models (VLMs). However, the use of RL beyond reasoning tasks remains largely unexplored, especially for perceptionintensive tasks like object detection and grounding. We propose V-Triune, a Visual Triple Unified Reinforcement Learning system that enables VLMs to jointly learn visual reasoning and perception tasks within a single training pipeline. V-Triune comprises triple complementary components: Sample-Level Data Formatting (to unify diverse task inputs), Verifier-Level Reward Computation (to deliver custom rewards via specialized verifiers) , and Source-Level Metric Monitoring (to diagnose problems at the data-source level). We further introduce a novel Dynamic IoU reward, which provides adaptive, progressive, and definite feedback for perception tasks handled by V-Triune. Our approach is instantiated within off-the-shelf RL training framework using open-source 7B and 32B backbone models. The resulting model, dubbed Orsta (One RL to See Them All), demonstrates consistent improvements across both reasoning and perception tasks. This broad capability is significantly shaped by its training on a diverse dataset, constructed around four representative visual reasoning tasks (Math, Puzzle, Chart, and Science) and four visual perception tasks (Grounding, Detection, Counting, and OCR). Subsequently, Orsta achieves substantial gains on MEGA-Bench Core, with improvements ranging from +2.1 to an impressive +14.1 across its various 7B and 32B model variants, with performance benefits extending to a wide range of downstream tasks. These results highlight the effectiveness and scalability of our unified RL approach for VLMs. The V-Triune system, along with the Orsta models, is publicly available at https://github.com/MiniMax-AI.
>
---
#### [replaced 165] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.17534v2](http://arxiv.org/pdf/2505.17534v2)**

> **作者:** Jingjing Jiang; Chongjie Si; Jun Luo; Hanwang Zhang; Chao Ma
>
> **摘要:** This paper presents a pioneering exploration of reinforcement learning (RL) via group relative policy optimization for unified multimodal large language models (ULMs), aimed at simultaneously reinforcing generation and understanding capabilities. Through systematic pilot studies, we uncover the significant potential of ULMs to enable the synergistic co-evolution of dual capabilities within a shared policy optimization framework. Building on this insight, we introduce CoRL, a co-reinforcement learning framework comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement. With the proposed CoRL, our resulting model, ULM-R1, achieves average improvements of 7% on three text-to-image generation datasets and 23% on nine multimodal understanding benchmarks. These results demonstrate the effectiveness of CoRL and highlight the substantial benefit of reinforcement learning in facilitating cross-task synergy and optimization for ULMs. Code is available at https://github.com/mm-vl/ULM-R1.
>
---
#### [replaced 166] RenderBender: A Survey on Adversarial Attacks Using Differentiable Rendering
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.09749v2](http://arxiv.org/pdf/2411.09749v2)**

> **作者:** Matthew Hull; Haoran Wang; Matthew Lau; Alec Helbling; Mansi Phute; Chao Zhang; Zsolt Kira; Willian Lunardi; Martin Andreoni; Wenke Lee; Polo Chau
>
> **备注:** 9 pages, 1 figure, 2 tables, IJCAI '25 Survey Track
>
> **摘要:** Differentiable rendering techniques like Gaussian Splatting and Neural Radiance Fields have become powerful tools for generating high-fidelity models of 3D objects and scenes. Their ability to produce both physically plausible and differentiable models of scenes are key ingredient needed to produce physically plausible adversarial attacks on DNNs. However, the adversarial machine learning community has yet to fully explore these capabilities, partly due to differing attack goals (e.g., misclassification, misdetection) and a wide range of possible scene manipulations used to achieve them (e.g., alter texture, mesh). This survey contributes the first framework that unifies diverse goals and tasks, facilitating easy comparison of existing work, identifying research gaps, and highlighting future directions - ranging from expanding attack goals and tasks to account for new modalities, state-of-the-art models, tools, and pipelines, to underscoring the importance of studying real-world threats in complex scenes.
>
---
