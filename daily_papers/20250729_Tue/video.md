# 计算机视觉 cs.CV

- **最新发布 205 篇**

- **更新 170 篇**

## 最新发布

#### [new 001] KB-DMGen: Knowledge-Based Global Guidance and Dynamic Pose Masking for Human Image Generation
- **分类: cs.CV**

- **简介: 该论文属于人体图像生成任务，旨在解决生成图像中姿态准确性和整体视觉质量难以兼顾的问题。提出了KB-DMGen方法，通过知识库增强姿态精度并维护图像质量，结合动态掩码调整姿态区域重要性，最终在HumanArt数据集上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2507.20083v1](http://arxiv.org/pdf/2507.20083v1)**

> **作者:** Shibang Liu; Xuemei Xie; Guangming Shi
>
> **摘要:** Recent methods using diffusion models have made significant progress in human image generation with various control signals such as pose priors. In portrait generation, both the accuracy of human pose and the overall visual quality are crucial for realistic synthesis. Most existing methods focus on controlling the accuracy of generated poses, but ignore the quality assurance of the entire image. In order to ensure the global image quality and pose accuracy, we propose Knowledge-Based Global Guidance and Dynamic pose Masking for human image Generation (KB-DMGen). The Knowledge Base (KB) is designed not only to enhance pose accuracy but also to leverage image feature information to maintain overall image quality. Dynamic Masking (DM) dynamically adjusts the importance of pose-related regions. Experiments demonstrate the effectiveness of our model, achieving new state-of-the-art results in terms of AP and CAP on the HumanArt dataset. The code will be made publicly available.
>
---
#### [new 002] Compositional Video Synthesis by Temporal Object-Centric Learning
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决现有方法缺乏显式对象结构和生成能力的问题。作者提出一种基于时序对象中心学习的视频合成框架，通过学习姿态不变的对象槽并结合扩散模型，实现高质量、时间连贯的视频生成，并支持对象级编辑操作。**

- **链接: [http://arxiv.org/pdf/2507.20855v1](http://arxiv.org/pdf/2507.20855v1)**

> **作者:** Adil Kaan Akan; Yucel Yemez
>
> **备注:** 12+21 pages, submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), currently under review
>
> **摘要:** We present a novel framework for compositional video synthesis that leverages temporally consistent object-centric representations, extending our previous work, SlotAdapt, from images to video. While existing object-centric approaches either lack generative capabilities entirely or treat video sequences holistically, thus neglecting explicit object-level structure, our approach explicitly captures temporal dynamics by learning pose invariant object-centric slots and conditioning them on pretrained diffusion models. This design enables high-quality, pixel-level video synthesis with superior temporal coherence, and offers intuitive compositional editing capabilities such as object insertion, deletion, or replacement, maintaining consistent object identities across frames. Extensive experiments demonstrate that our method sets new benchmarks in video generation quality and temporal consistency, outperforming previous object-centric generative methods. Although our segmentation performance closely matches state-of-the-art methods, our approach uniquely integrates this capability with robust generative performance, significantly advancing interactive and controllable video generation and opening new possibilities for advanced content creation, semantic editing, and dynamic scene understanding.
>
---
#### [new 003] T2VParser: Adaptive Decomposition Tokens for Partial Alignment in Text to Video Retrieval
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于文本到视频检索任务，旨在解决视频与文本部分对齐的问题。现有方法直接对齐整体表示，忽略信息不等价性。作者提出T2VParser，引入自适应分解Token，实现跨模态语义分解，提升部分对齐精度。**

- **链接: [http://arxiv.org/pdf/2507.20518v1](http://arxiv.org/pdf/2507.20518v1)**

> **作者:** Yili Li; Gang Xiong; Gaopeng Gou; Xiangyan Qu; Jiamin Zhuang; Zhen Li; Junzheng Shi
>
> **摘要:** Text-to-video retrieval essentially aims to train models to align visual content with textual descriptions accurately. Due to the impressive general multimodal knowledge demonstrated by image-text pretrained models such as CLIP, existing work has primarily focused on extending CLIP knowledge for video-text tasks. However, videos typically contain richer information than images. In current video-text datasets, textual descriptions can only reflect a portion of the video content, leading to partial misalignment in video-text matching. Therefore, directly aligning text representations with video representations can result in incorrect supervision, ignoring the inequivalence of information. In this work, we propose T2VParser to extract multiview semantic representations from text and video, achieving adaptive semantic alignment rather than aligning the entire representation. To extract corresponding representations from different modalities, we introduce Adaptive Decomposition Tokens, which consist of a set of learnable tokens shared across modalities. The goal of T2VParser is to emphasize precise alignment between text and video while retaining the knowledge of pretrained models. Experimental results demonstrate that T2VParser achieves accurate partial alignment through effective cross-modal content decomposition. The code is available at https://github.com/Lilidamowang/T2VParser.
>
---
#### [new 004] ATR-UMMIM: A Benchmark Dataset for UAV-Based Multimodal Image Registration under Complex Imaging Conditions
- **分类: cs.CV**

- **简介: 该论文属于图像处理与计算机视觉任务，旨在解决无人机多模态图像配准问题。现有数据集缺乏复杂环境下的真实场景支持，限制了算法发展。作者提出ATR-UMMIM，首个面向无人机多模态配准的公开基准数据集，包含可见光、红外及配准图像三元组，并提供像素级标注、成像条件属性与目标框标注，覆盖多种飞行高度、视角及全天候环境，推动真实场景下多模态融合与感知研究。**

- **链接: [http://arxiv.org/pdf/2507.20764v1](http://arxiv.org/pdf/2507.20764v1)**

> **作者:** Kangcheng Bin; Chen Chen; Ting Hu; Jiahao Qi; Ping Zhong
>
> **摘要:** Multimodal fusion has become a key enabler for UAV-based object detection, as each modality provides complementary cues for robust feature extraction. However, due to significant differences in resolution, field of view, and sensing characteristics across modalities, accurate registration is a prerequisite before fusion. Despite its importance, there is currently no publicly available benchmark specifically designed for multimodal registration in UAV-based aerial scenarios, which severely limits the development and evaluation of advanced registration methods under real-world conditions. To bridge this gap, we present ATR-UMMIM, the first benchmark dataset specifically tailored for multimodal image registration in UAV-based applications. This dataset includes 7,969 triplets of raw visible, infrared, and precisely registered visible images captured covers diverse scenarios including flight altitudes from 80m to 300m, camera angles from 0{\deg} to 75{\deg}, and all-day, all-year temporal variations under rich weather and illumination conditions. To ensure high registration quality, we design a semi-automated annotation pipeline to introduce reliable pixel-level ground truth to each triplet. In addition, each triplet is annotated with six imaging condition attributes, enabling benchmarking of registration robustness under real-world deployment settings. To further support downstream tasks, we provide object-level annotations on all registered images, covering 11 object categories with 77,753 visible and 78,409 infrared bounding boxes. We believe ATR-UMMIM will serve as a foundational benchmark for advancing multimodal registration, fusion, and perception in real-world UAV scenarios. The datatset can be download from https://github.com/supercpy/ATR-UMMIM
>
---
#### [new 005] Endoscopic Depth Estimation Based on Deep Learning: A Survey
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于医学图像分析任务，旨在解决微创手术中内窥镜深度估计问题。论文系统综述了基于深度学习的最新方法，从数据、方法和应用三个角度进行分析，涵盖单目和双目技术，总结了评估指标和公开数据集，并探讨了在机器人辅助手术中的应用与未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.20881v1](http://arxiv.org/pdf/2507.20881v1)**

> **作者:** Ke Niu; Zeyun Liu; Xue Feng; Heng Li; Kaize Shi
>
> **摘要:** Endoscopic depth estimation is a critical technology for improving the safety and precision of minimally invasive surgery. It has attracted considerable attention from researchers in medical imaging, computer vision, and robotics. Over the past decade, a large number of methods have been developed. Despite the existence of several related surveys, a comprehensive overview focusing on recent deep learning-based techniques is still limited. This paper endeavors to bridge this gap by systematically reviewing the state-of-the-art literature. Specifically, we provide a thorough survey of the field from three key perspectives: data, methods, and applications, covering a range of methods including both monocular and stereo approaches. We describe common performance evaluation metrics and summarize publicly available datasets. Furthermore, this review analyzes the specific challenges of endoscopic scenes and categorizes representative techniques based on their supervision strategies and network architectures. The application of endoscopic depth estimation in the important area of robot-assisted surgery is also reviewed. Finally, we outline potential directions for future research, such as domain adaptation, real-time implementation, and enhanced model generalization, thereby providing a valuable starting point for researchers to engage with and advance the field.
>
---
#### [new 006] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决MRI脑肿瘤分割中相邻切片空间关联利用不足的问题。作者提出了M-Net框架，引入Mesh-Cast机制和Two-Phase Sequential训练策略，提升分割连续性和精度。**

- **链接: [http://arxiv.org/pdf/2507.20582v1](http://arxiv.org/pdf/2507.20582v1)**

> **作者:** Jiacheng Lu; Hui Ding; Shiyu Zhang; Guoping Huo
>
> **备注:** ICCV 2025 Accepted
>
> **摘要:** MRI tumor segmentation remains a critical challenge in medical imaging, where volumetric analysis faces unique computational demands due to the complexity of 3D data. The spatially sequential arrangement of adjacent MRI slices provides valuable information that enhances segmentation continuity and accuracy, yet this characteristic remains underutilized in many existing models. The spatial correlations between adjacent MRI slices can be regarded as "temporal-like" data, similar to frame sequences in video segmentation tasks. To bridge this gap, we propose M-Net, a flexible framework specifically designed for sequential image segmentation. M-Net introduces the novel Mesh-Cast mechanism, which seamlessly integrates arbitrary sequential models into the processing of both channel and temporal information, thereby systematically capturing the inherent "temporal-like" spatial correlations between MRI slices. Additionally, we define an MRI sequential input pattern and design a Two-Phase Sequential (TPS) training strategy, which first focuses on learning common patterns across sequences before refining slice-specific feature extraction. This approach leverages temporal modeling techniques to preserve volumetric contextual information while avoiding the high computational cost of full 3D convolutions, thereby enhancing the generalizability and robustness of M-Net in sequential segmentation tasks. Experiments on the BraTS2019 and BraTS2023 datasets demonstrate that M-Net outperforms existing methods across all key metrics, establishing itself as a robust solution for temporally-aware MRI tumor segmentation.
>
---
#### [new 007] Improving Adversarial Robustness Through Adaptive Learning-Driven Multi-Teacher Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在提升卷积神经网络（CNN）在面对对抗攻击时的鲁棒性。通过使用多教师对抗训练模型进行知识蒸馏，并结合自适应学习策略，使学生模型在未接触对抗数据的情况下增强抗攻击能力。实验验证了该方法在多个数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.20996v1](http://arxiv.org/pdf/2507.20996v1)**

> **作者:** Hayat Ullah; Syed Muhammad Talha Zaidi; Arslan Munir
>
> **备注:** 11 pages
>
> **摘要:** Convolutional neural networks (CNNs) excel in computer vision but are susceptible to adversarial attacks, crafted perturbations designed to mislead predictions. Despite advances in adversarial training, a gap persists between model accuracy and robustness. To mitigate this issue, in this paper, we present a multi-teacher adversarial robustness distillation using an adaptive learning strategy. Specifically, our proposed method first trained multiple clones of a baseline CNN model using an adversarial training strategy on a pool of perturbed data acquired through different adversarial attacks. Once trained, these adversarially trained models are used as teacher models to supervise the learning of a student model on clean data using multi-teacher knowledge distillation. To ensure an effective robustness distillation, we design an adaptive learning strategy that controls the knowledge contribution of each model by assigning weights as per their prediction precision. Distilling knowledge from adversarially pre-trained teacher models not only enhances the learning capabilities of the student model but also empowers it with the capacity to withstand different adversarial attacks, despite having no exposure to adversarial data. To verify our claims, we extensively evaluated our proposed method on MNIST-Digits and Fashion-MNIST datasets across diverse experimental settings. The obtained results exhibit the efficacy of our multi-teacher adversarial distillation and adaptive learning strategy, enhancing CNNs' adversarial robustness against various adversarial attacks.
>
---
#### [new 008] Beyond Class Tokens: LLM-guided Dominant Property Mining for Few-shot Classification
- **分类: cs.CV**

- **简介: 该论文属于小样本分类任务，旨在解决数据稀缺下新类识别问题。通过改进对比学习方法，结合大语言模型挖掘主导属性，提升图像表征学习效果。提出了多属性生成器和新对比学习策略，取得更好分类性能。**

- **链接: [http://arxiv.org/pdf/2507.20511v1](http://arxiv.org/pdf/2507.20511v1)**

> **作者:** Wei Zhuo; Runjie Luo; Wufeng Xue; Linlin Shen
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Few-shot Learning (FSL), which endeavors to develop the generalization ability for recognizing novel classes using only a few images, faces significant challenges due to data scarcity. Recent CLIP-like methods based on contrastive language-image pertaining mitigate the issue by leveraging textual representation of the class name for unseen image discovery. Despite the achieved success, simply aligning visual representations to class name embeddings would compromise the visual diversity for novel class discrimination. To this end, we proposed a novel Few-Shot Learning (FSL) method (BCT-CLIP) that explores \textbf{dominating properties} via contrastive learning beyond simply using class tokens. Through leveraging LLM-based prior knowledge, our method pushes forward FSL with comprehensive structural image representations, including both global category representation and the patch-aware property embeddings. In particular, we presented a novel multi-property generator (MPG) with patch-aware cross-attentions to generate multiple visual property tokens, a Large-Language Model (LLM)-assistant retrieval procedure with clustering-based pruning to obtain dominating property descriptions, and a new contrastive learning strategy for property-token learning. The superior performances on the 11 widely used datasets demonstrate that our investigation of dominating properties advances discriminative class-specific representation learning and few-shot classification.
>
---
#### [new 009] AnimalClue: Recognizing Animals by their Traces
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在通过动物痕迹（如足迹、粪便等）识别物种。为解决该问题，作者构建了首个大规模数据集AnimalClue，包含159,605个标注样本，涵盖968个物种，并评估了视觉模型在此任务上的表现，推动野生动物监测的自动化发展。**

- **链接: [http://arxiv.org/pdf/2507.20240v1](http://arxiv.org/pdf/2507.20240v1)**

> **作者:** Risa Shinoda; Nakamasa Inoue; Iro Laina; Christian Rupprecht; Hirokatsu Kataoka
>
> **备注:** ICCV2025 Highlight
>
> **摘要:** Wildlife observation plays an important role in biodiversity conservation, necessitating robust methodologies for monitoring wildlife populations and interspecies interactions. Recent advances in computer vision have significantly contributed to automating fundamental wildlife observation tasks, such as animal detection and species identification. However, accurately identifying species from indirect evidence like footprints and feces remains relatively underexplored, despite its importance in contributing to wildlife monitoring. To bridge this gap, we introduce AnimalClue, the first large-scale dataset for species identification from images of indirect evidence. Our dataset consists of 159,605 bounding boxes encompassing five categories of indirect clues: footprints, feces, eggs, bones, and feathers. It covers 968 species, 200 families, and 65 orders. Each image is annotated with species-level labels, bounding boxes or segmentation masks, and fine-grained trait information, including activity patterns and habitat preferences. Unlike existing datasets primarily focused on direct visual features (e.g., animal appearances), AnimalClue presents unique challenges for classification, detection, and instance segmentation tasks due to the need for recognizing more detailed and subtle visual features. In our experiments, we extensively evaluate representative vision models and identify key challenges in animal identification from their traces. Our dataset and code are available at https://dahlian00.github.io/AnimalCluePage/
>
---
#### [new 010] HumanSAM: Classifying Human-centric Forgery Videos in Human Spatial, Appearance, and Motion Anomaly
- **分类: cs.CV**

- **简介: 该论文属于视频伪造分类任务，旨在解决当前伪造视频检测缺乏细粒度分类与可解释性问题。作者提出HumanSAM框架，将伪造分为空间、外观和运动异常三类，设计双分支模型融合时空与深度信息，并引入置信度增强策略。构建了首个面向人体伪造的HFV数据集，实验表明该方法在二类与多类分类上均表现优异。**

- **链接: [http://arxiv.org/pdf/2507.19924v1](http://arxiv.org/pdf/2507.19924v1)**

> **作者:** Chang Liu; Yunfan Ye; Fan Zhang; Qingyang Zhou; Yuchuan Luo; Zhiping Cai
>
> **备注:** ICCV 2025. Project page: https://dejian-lc.github.io/humansam/
>
> **摘要:** Numerous synthesized videos from generative models, especially human-centric ones that simulate realistic human actions, pose significant threats to human information security and authenticity. While progress has been made in binary forgery video detection, the lack of fine-grained understanding of forgery types raises concerns regarding both reliability and interpretability, which are critical for real-world applications. To address this limitation, we propose HumanSAM, a new framework that builds upon the fundamental challenges of video generation models. Specifically, HumanSAM aims to classify human-centric forgeries into three distinct types of artifacts commonly observed in generated content: spatial, appearance, and motion anomaly.To better capture the features of geometry, semantics and spatiotemporal consistency, we propose to generate the human forgery representation by fusing two branches of video understanding and spatial depth. We also adopt a rank-based confidence enhancement strategy during the training process to learn more robust representation by introducing three prior scores. For training and evaluation, we construct the first public benchmark, the Human-centric Forgery Video (HFV) dataset, with all types of forgeries carefully annotated semi-automatically. In our experiments, HumanSAM yields promising results in comparison with state-of-the-art methods, both in binary and multi-class forgery classification.
>
---
#### [new 011] DriveAgent-R1: Advancing VLM-based Autonomous Driving with Hybrid Thinking and Active Perception
- **分类: cs.CV**

- **简介: 本文属于自动驾驶任务，旨在解决视觉语言模型（VLM）在复杂环境中决策短视和感知被动的问题。作者提出了DriveAgent-R1，包含混合思考框架和主动感知机制，通过三阶段强化学习训练，实现高效可靠的长视野行为决策，提升自动驾驶系统的智能与安全性。**

- **链接: [http://arxiv.org/pdf/2507.20879v1](http://arxiv.org/pdf/2507.20879v1)**

> **作者:** Weicheng Zheng; Xiaofei Mao; Nanfei Ye; Pengxiang Li; Kun Zhan; Xianpeng Lang; Hang Zhao
>
> **摘要:** Vision-Language Models (VLMs) are advancing autonomous driving, yet their potential is constrained by myopic decision-making and passive perception, limiting reliability in complex environments. We introduce DriveAgent-R1 to tackle these challenges in long-horizon, high-level behavioral decision-making. DriveAgent-R1 features two core innovations: a Hybrid-Thinking framework that adaptively switches between efficient text-based and in-depth tool-based reasoning, and an Active Perception mechanism with a vision toolkit to proactively resolve uncertainties, thereby balancing decision-making efficiency and reliability. The agent is trained using a novel, three-stage progressive reinforcement learning strategy designed to master these hybrid capabilities. Extensive experiments demonstrate that DriveAgent-R1 achieves state-of-the-art performance, outperforming even leading proprietary large multimodal models, such as Claude Sonnet 4. Ablation studies validate our approach and confirm that the agent's decisions are robustly grounded in actively perceived visual evidence, paving a path toward safer and more intelligent autonomous systems.
>
---
#### [new 012] MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification
- **分类: cs.CV**

- **简介: 该论文属于脑机接口任务，旨在解决现有EEG模型在运动想象分类中泛化能力不足的问题。作者提出了MIRepNet，一个专为MI范式设计的预训练模型，结合自监督与监督学习策略，适配不同电极配置，提升小样本下的分类性能。**

- **链接: [http://arxiv.org/pdf/2507.20254v1](http://arxiv.org/pdf/2507.20254v1)**

> **作者:** Dingkun Liu; Zhu Chen; Jingwei Luo; Shijie Lian; Dongrui Wu
>
> **摘要:** Brain-computer interfaces (BCIs) enable direct communication between the brain and external devices. Recent EEG foundation models aim to learn generalized representations across diverse BCI paradigms. However, these approaches overlook fundamental paradigm-specific neurophysiological distinctions, limiting their generalization ability. Importantly, in practical BCI deployments, the specific paradigm such as motor imagery (MI) for stroke rehabilitation or assistive robotics, is generally determined prior to data acquisition. This paper proposes MIRepNet, the first EEG foundation model tailored for the MI paradigm. MIRepNet comprises a high-quality EEG preprocessing pipeline incorporating a neurophysiologically-informed channel template, adaptable to EEG headsets with arbitrary electrode configurations. Furthermore, we introduce a hybrid pretraining strategy that combines self-supervised masked token reconstruction and supervised MI classification, facilitating rapid adaptation and accurate decoding on novel downstream MI tasks with fewer than 30 trials per class. Extensive evaluations across five public MI datasets demonstrated that MIRepNet consistently achieved state-of-the-art performance, significantly outperforming both specialized and generalized EEG models. Our code will be available on GitHub\footnote{https://github.com/staraink/MIRepNet}.
>
---
#### [new 013] TransFlow: Motion Knowledge Transfer from Video Diffusion Models to Video Salient Object Detection
- **分类: cs.CV**

- **简介: 该论文属于视频显著目标检测（SOD）任务，旨在解决因视频数据稀缺而难以训练模型的问题。论文提出TransFlow方法，通过从预训练视频扩散模型中迁移运动知识，生成具有语义意识的光流，从而合成高质量的视频训练数据，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.19789v1](http://arxiv.org/pdf/2507.19789v1)**

> **作者:** Suhwan Cho; Minhyeok Lee; Jungho Lee; Sunghun Yang; Sangyoun Lee
>
> **备注:** ICCVW 2025
>
> **摘要:** Video salient object detection (SOD) relies on motion cues to distinguish salient objects from backgrounds, but training such models is limited by scarce video datasets compared to abundant image datasets. Existing approaches that use spatial transformations to create video sequences from static images fail for motion-guided tasks, as these transformations produce unrealistic optical flows that lack semantic understanding of motion. We present TransFlow, which transfers motion knowledge from pre-trained video diffusion models to generate realistic training data for video SOD. Video diffusion models have learned rich semantic motion priors from large-scale video data, understanding how different objects naturally move in real scenes. TransFlow leverages this knowledge to generate semantically-aware optical flows from static images, where objects exhibit natural motion patterns while preserving spatial boundaries and temporal coherence. Our method achieves improved performance across multiple benchmarks, demonstrating effective motion knowledge transfer.
>
---
#### [new 014] Pic2Diagnosis: A Method for Diagnosis of Cardiovascular Diseases from the Printed ECG Pictures
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决从印刷或扫描的ECG图像中自动诊断心血管疾病的问题。现有方法依赖过时数据和低精度算法，作者提出Pic2Diagnosis方法，采用两步课程学习框架，先在分割掩码上预训练分类模型，再在灰度反转ECG图像上微调，通过模型集成提升性能，实现了高准确率和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.19961v1](http://arxiv.org/pdf/2507.19961v1)**

> **作者:** Oğuzhan Büyüksolak; İlkay Öksüz
>
> **备注:** To appear in: Proceedings of the 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), 2025
>
> **摘要:** The electrocardiogram (ECG) is a vital tool for diagnosing heart diseases. However, many disease patterns are derived from outdated datasets and traditional stepwise algorithms with limited accuracy. This study presents a method for direct cardiovascular disease (CVD) diagnosis from ECG images, eliminating the need for digitization. The proposed approach utilizes a two-step curriculum learning framework, beginning with the pre-training of a classification model on segmentation masks, followed by fine-tuning on grayscale, inverted ECG images. Robustness is further enhanced through an ensemble of three models with averaged outputs, achieving an AUC of 0.9534 and an F1 score of 0.7801 on the BHF ECG Challenge dataset, outperforming individual models. By effectively handling real-world artifacts and simplifying the diagnostic process, this method offers a reliable solution for automated CVD diagnosis, particularly in resource-limited settings where printed or scanned ECG images are commonly used. Such an automated procedure enables rapid and accurate diagnosis, which is critical for timely intervention in CVD cases that often demand urgent care.
>
---
#### [new 015] Multi-Attention Stacked Ensemble for Lung Cancer Detection in CT Scans
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决肺结节良恶性分类问题。作者提出了一种多注意力堆叠集成模型，结合三种预训练网络与注意力机制，并采用动态焦点损失和数据增强策略。在LIDC-IDRI数据集上取得了高准确率和AUC，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.20221v1](http://arxiv.org/pdf/2507.20221v1)**

> **作者:** Uzzal Saha; Surya Prakash
>
> **备注:** 26 pages, 14 figures
>
> **摘要:** In this work, we address the challenge of binary lung nodule classification (benign vs malignant) using CT images by proposing a multi-level attention stacked ensemble of deep neural networks. Three pretrained backbones - EfficientNet V2 S, MobileViT XXS, and DenseNet201 - are each adapted with a custom classification head tailored to 96 x 96 pixel inputs. A two-stage attention mechanism learns both model-wise and class-wise importance scores from concatenated logits, and a lightweight meta-learner refines the final prediction. To mitigate class imbalance and improve generalization, we employ dynamic focal loss with empirically calculated class weights, MixUp augmentation during training, and test-time augmentation at inference. Experiments on the LIDC-IDRI dataset demonstrate exceptional performance, achieving 98.09 accuracy and 0.9961 AUC, representing a 35 percent reduction in error rate compared to state-of-the-art methods. The model exhibits balanced performance across sensitivity (98.73) and specificity (98.96), with particularly strong results on challenging cases where radiologist disagreement was high. Statistical significance testing confirms the robustness of these improvements across multiple experimental runs. Our approach can serve as a robust, automated aid for radiologists in lung cancer screening.
>
---
#### [new 016] ForCenNet: Foreground-Centric Network for Document Image Rectification
- **分类: cs.CV**

- **简介: 该论文属于文档图像矫正任务，旨在消除拍摄文档中的几何失真以提升文本识别效果。现有方法忽略前景元素的重要性，而本文提出ForCenNet，通过前景标签生成、前景掩码机制和曲率一致性损失，有效恢复文本和布局结构，提升了矫正效果。**

- **链接: [http://arxiv.org/pdf/2507.19804v1](http://arxiv.org/pdf/2507.19804v1)**

> **作者:** Peng Cai; Qiang Li; Kaicheng Yang; Dong Guo; Jia Li; Nan Zhou; Xiang An; Ninghua Yang; Jiankang Deng
>
> **备注:** Accepted by ICCV25, 16 pages, 14 figures
>
> **摘要:** Document image rectification aims to eliminate geometric deformation in photographed documents to facilitate text recognition. However, existing methods often neglect the significance of foreground elements, which provide essential geometric references and layout information for document image correction. In this paper, we introduce Foreground-Centric Network (ForCenNet) to eliminate geometric distortions in document images. Specifically, we initially propose a foreground-centric label generation method, which extracts detailed foreground elements from an undistorted image. Then we introduce a foreground-centric mask mechanism to enhance the distinction between readable and background regions. Furthermore, we design a curvature consistency loss to leverage the detailed foreground labels to help the model understand the distorted geometric distribution. Extensive experiments demonstrate that ForCenNet achieves new state-of-the-art on four real-world benchmarks, such as DocUNet, DIR300, WarpDoc, and DocReal. Quantitative analysis shows that the proposed method effectively undistorts layout elements, such as text lines and table borders. The resources for further comparison are provided at https://github.com/caipeng328/ForCenNet.
>
---
#### [new 017] DeepJIVE: Learning Joint and Individual Variation Explained from Multimodal Data Using Deep Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DeepJIVE，一种基于深度学习的多模态数据融合方法，用于提取数据中的共享和个体特异性变化。它解决了传统方法难以处理高维数据和非线性结构的问题，通过三种损失函数实现身份性和正交性约束，并在ADNI数据中找到PET与MR图像间的生物学合理关联。**

- **链接: [http://arxiv.org/pdf/2507.19682v1](http://arxiv.org/pdf/2507.19682v1)**

> **作者:** Matthew Drexler; Benjamin Risk; James J Lah; Suprateek Kundu; Deqiang Qiu
>
> **备注:** 26 pages, 10 figures
>
> **摘要:** Conventional multimodal data integration methods provide a comprehensive assessment of the shared or unique structure within each individual data type but suffer from several limitations such as the inability to handle high-dimensional data and identify nonlinear structures. In this paper, we introduce DeepJIVE, a deep-learning approach to performing Joint and Individual Variance Explained (JIVE). We perform mathematical derivation and experimental validations using both synthetic and real-world 1D, 2D, and 3D datasets. Different strategies of achieving the identity and orthogonality constraints for DeepJIVE were explored, resulting in three viable loss functions. We found that DeepJIVE can successfully uncover joint and individual variations of multimodal datasets. Our application of DeepJIVE to the Alzheimer's Disease Neuroimaging Initiative (ADNI) also identified biologically plausible covariation patterns between the amyloid positron emission tomography (PET) and magnetic resonance (MR) images. In conclusion, the proposed DeepJIVE can be a useful tool for multimodal data analysis.
>
---
#### [new 018] Quaternion-Based Robust PCA for Efficient Moving Target Detection and Background Recovery in Color Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在解决彩色视频中移动目标检测与背景恢复问题。针对传统方法计算复杂度高、色彩通道低秩难以保持的问题，作者提出基于四元数鲁棒主成分分析（uQRPCA）及其改进框架uQRPCA+，实现高效准确的目标分割与背景重建，达到当前最优性能。**

- **链接: [http://arxiv.org/pdf/2507.19730v1](http://arxiv.org/pdf/2507.19730v1)**

> **作者:** Liyang Wang; Shiqian Wu; Shun Fang; Qile Zhu; Jiaxin Wu; Sos Again
>
> **摘要:** Moving target detection is a challenging computer vision task aimed at generating accurate segmentation maps in diverse in-the-wild color videos captured by static cameras. If backgrounds and targets can be simultaneously extracted and recombined, such synthetic data can significantly enrich annotated in-the-wild datasets and enhance the generalization ability of deep models. Quaternion-based RPCA (QRPCA) is a promising unsupervised paradigm for color image processing. However, in color video processing, Quaternion Singular Value Decomposition (QSVD) incurs high computational costs, and rank-1 quaternion matrix fails to yield rank-1 color channels. In this paper, we reduce the computational complexity of QSVD to o(1) by utilizing a quaternion Riemannian manifold. Furthermor, we propose the universal QRPCA (uQRPCA) framework, which achieves a balance in simultaneously segmenting targets and recovering backgrounds from color videos. Moreover, we expand to uQRPCA+ by introducing the Color Rank-1 Batch (CR1B) method to further process and obtain the ideal low-rank background across color channels. Experiments demonstrate our uQRPCA+ achieves State Of The Art (SOTA) performance on moving target detection and background recovery tasks compared to existing open-source methods. Our implementation is publicly available on GitHub at https://github.com/Ruchtech/uQRPCA
>
---
#### [new 019] The Devil is in the EOS: Sequence Training for Detailed Image Captioning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像描述生成任务，旨在解决当前模型生成描述缺乏细节的问题。论文发现交叉熵训练导致模型过早结束生成（EOS偏差），提出一种无监督方法缓解该问题，使生成更长、更详细的描述。实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2507.20077v1](http://arxiv.org/pdf/2507.20077v1)**

> **作者:** Abdelrahman Mohamed; Yova Kementchedjhieva
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Despite significant advances in vision-language models (VLMs), image captioning often suffers from a lack of detail, with base models producing short, generic captions. This limitation persists even though VLMs are equipped with strong vision and language backbones. While supervised data and complex reward functions have been proposed to improve detailed image captioning, we identify a simpler underlying issue: a bias towards the end-of-sequence (EOS) token, which is introduced during cross-entropy training. We propose an unsupervised method to debias the model's tendency to predict the EOS token prematurely. By reducing this bias, we encourage the generation of longer, more detailed captions without the need for intricate reward functions or supervision. Our approach is straightforward, effective, and easily applicable to any pretrained model. We demonstrate its effectiveness through experiments with three VLMs and on three detailed captioning benchmarks. Our results show a substantial increase in caption length and relevant details, albeit with an expected increase in the rate of hallucinations.
>
---
#### [new 020] Trust the Model: Compact VLMs as In-Context Judges for Image-Text Data Quality
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型任务，旨在解决图像-文本数据质量筛选问题。作者提出一种基于小型VLM的数据过滤框架，利用其内在评估能力筛选高质量图像-文本对，无需额外模块，降低训练开销，并提升数据质量和模型性能。**

- **链接: [http://arxiv.org/pdf/2507.20156v1](http://arxiv.org/pdf/2507.20156v1)**

> **作者:** Daulet Toibazar; Kesen Wang; Sherif Mohamed; Abdulaziz Al-Badawi; Abdulrahman Alfulayt; Pedro J. Moreno
>
> **摘要:** Vision-language models (VLMs) extend the conventional large language models by integrating visual data, enabling richer multimodal reasoning and significantly broadens the practical applications of AI. However, including visual inputs also brings new challenges in maintaining data quality. Empirical evidence consistently shows that carefully curated and representative training examples often yield superior results compared to simply increasing the quantity of data. Inspired by this observation, we introduce a streamlined data filtration framework that employs a compact VLM, fine-tuned on a high-quality image-caption annotated dataset. This model effectively evaluates and filters potential training samples based on caption and image quality and alignment. Unlike previous approaches, which typically add auxiliary filtration modules on top of existing full-scale VLMs, our method exclusively utilizes the inherent evaluative capability of a purpose-built small VLM. This strategy eliminates the need for extra modules and reduces training overhead. Our lightweight model efficiently filters out inaccurate, noisy web data, improving image-text alignment and caption linguistic fluency. Experimental results show that datasets underwent high-precision filtration using our compact VLM perform on par with, or even surpass, larger and noisier datasets gathered through high-volume web crawling. Thus, our method provides a lightweight yet robust solution for building high-quality vision-language training corpora. \\ \textbf{Availability and implementation:} Our compact VLM filtration model, training data, utility scripts, and Supplementary data (Appendices) are freely available at https://github.com/daulettoibazar/Compact_VLM_Filter.
>
---
#### [new 021] Priority-Aware Pathological Hierarchy Training for Multiple Instance Learning
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析中的多实例学习（MIL）任务，旨在解决临床诊断中病理症状与类别优先级被忽视的问题。作者提出了一种结合垂直与水平层次结构的训练方法，通过层级对齐与特征重用，使模型更关注严重类别，从而减少误诊并提升多症状复杂病例的预测效果。**

- **链接: [http://arxiv.org/pdf/2507.20469v1](http://arxiv.org/pdf/2507.20469v1)**

> **作者:** Sungrae Hong; Kyungeun Kim; Juhyeon Kim; Sol Lee; Jisu Shin; Chanjae Song; Mun Yong Yi
>
> **备注:** 10 pages, 4 figures, Accepted for oral presentation by The 2nd MICCAI Student Board (MSB) EMERGE Workshop
>
> **摘要:** Multiple Instance Learning (MIL) is increasingly being used as a support tool within clinical settings for pathological diagnosis decisions, achieving high performance and removing the annotation burden. However, existing approaches for clinical MIL tasks have not adequately addressed the priority issues that exist in relation to pathological symptoms and diagnostic classes, causing MIL models to ignore priority among classes. To overcome this clinical limitation of MIL, we propose a new method that addresses priority issues using two hierarchies: vertical inter-hierarchy and horizontal intra-hierarchy. The proposed method aligns MIL predictions across each hierarchical level and employs an implicit feature re-usability during training to facilitate clinically more serious classes within the same level. Experiments with real-world patient data show that the proposed method effectively reduces misdiagnosis and prioritizes more important symptoms in multiclass scenarios. Further analysis verifies the efficacy of the proposed components and qualitatively confirms the MIL predictions against challenging cases with multiple symptoms.
>
---
#### [new 022] Tuning adaptive gamma correction (TAGC) for enhancing images in low ligh
- **分类: cs.CV**

- **简介: 该论文属于图像增强任务，旨在解决低光条件下图像质量差的问题。作者提出了一种自适应伽马校正模型（TAGC），通过分析图像的颜色亮度和平均颜色，自动计算适应不同光照条件的伽马系数，从而提升图像质量，保持细节和自然视觉效果。**

- **链接: [http://arxiv.org/pdf/2507.19574v1](http://arxiv.org/pdf/2507.19574v1)**

> **作者:** Ghufran Abualhail Alhamzawi; Ali Saeed Alfoudi; Ali Hakem Alsaeedi; Suha Mohammed Hadi; Amjed Abbas Ahmed; Md. Riad Hassan; Nurhizam Safie Mohd Satar; Waeel Yahya Yasseen
>
> **摘要:** Enhancing images in low-light conditions is an important challenge in computer vision. Insufficient illumination negatively affects the quality of images, resulting in low contrast, intensive noise, and blurred details. This paper presents a model for enhancing low-light images called tuning adaptive gamma correction (TAGC). The model is based on analyzing the color luminance of the low-light image and calculating the average color to determine the adaptive gamma coefficient. The gamma value is calculated automatically and adaptively at different illumination levels suitable for the image without human intervention or manual adjustment. Based on qualitative and quantitative evaluation, tuning adaptive gamma correction model has effectively improved low-light images while maintaining details, natural contrast, and correct color distribution. It also provides natural visual quality. It can be considered a more efficient solution for processing low-light images in multiple applications such as night surveillance, improving the quality of medical images, and photography in low-light environments.
>
---
#### [new 023] Efficient Learning for Product Attributes with Compact Multimodal Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于电商中基于图像的产品属性预测任务，旨在解决标注数据成本高的问题。论文采用高效的小规模视觉语言模型，结合PEFT和DPO方法，利用无标签数据进行半监督微调，显著提升了模型性能。**

- **链接: [http://arxiv.org/pdf/2507.19679v1](http://arxiv.org/pdf/2507.19679v1)**

> **作者:** Mandar Kulkarni
>
> **摘要:** Image-based product attribute prediction in e-commerce is a crucial task with numerous applications. The supervised fine-tuning of Vision Language Models (VLMs) faces significant scale challenges due to the cost of manual or API based annotation. In this paper, we investigate label-efficient semi-supervised fine-tuning strategies for compact VLMs (2B-3B parameters) that leverage unlabeled product listings through Direct Preference Optimization (DPO). Beginning with a small, API-based, annotated, and labeled set, we first employ PEFT to train low-rank adapter modules. To update the adapter weights with unlabeled data, we generate multiple reasoning-and-answer chains per unlabeled sample and segregate these chains into preferred and dispreferred based on self-consistency. We then fine-tune the model with DPO loss and use the updated model for the next iteration. By using PEFT fine-tuning with DPO, our method achieves efficient convergence with minimal compute overhead. On a dataset spanning twelve e-commerce verticals, DPO-based fine-tuning, which utilizes only unlabeled data, demonstrates a significant improvement over the supervised model. Moreover, experiments demonstrate that accuracy with DPO training improves with more unlabeled data, indicating that a large pool of unlabeled samples can be effectively leveraged to improve performance.
>
---
#### [new 024] AV-Deepfake1M++: A Large-Scale Audio-Visual Deepfake Benchmark with Real-World Perturbations
- **分类: cs.CV**

- **简介: 该论文属于音视频深度伪造检测任务，旨在解决生成模型伪造视频日益逼真、检测难度增大的问题。论文扩展了原有数据集，构建了包含200万视频片段的AV-Deepfake1M++，涵盖多种生成方法与真实场景扰动，推动检测技术研究。**

- **链接: [http://arxiv.org/pdf/2507.20579v1](http://arxiv.org/pdf/2507.20579v1)**

> **作者:** Zhixi Cai; Kartik Kuckreja; Shreya Ghosh; Akanksha Chuchra; Muhammad Haris Khan; Usman Tariq; Tom Gedeon; Abhinav Dhall
>
> **摘要:** The rapid surge of text-to-speech and face-voice reenactment models makes video fabrication easier and highly realistic. To encounter this problem, we require datasets that rich in type of generation methods and perturbation strategy which is usually common for online videos. To this end, we propose AV-Deepfake1M++, an extension of the AV-Deepfake1M having 2 million video clips with diversified manipulation strategy and audio-visual perturbation. This paper includes the description of data generation strategies along with benchmarking of AV-Deepfake1M++ using state-of-the-art methods. We believe that this dataset will play a pivotal role in facilitating research in Deepfake domain. Based on this dataset, we host the 2025 1M-Deepfakes Detection Challenge. The challenge details, dataset and evaluation scripts are available online under a research-only license at https://deepfakes1m.github.io/2025.
>
---
#### [new 025] GPT-IMAGE-EDIT-1.5M: A Million-Scale, GPT-Generated Image Dataset
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决开源模型在高质量指令引导图像编辑上的数据匮乏问题。作者利用GPT-4o构建了包含150万高质量三元组的大规模数据集，并通过优化图像质量和提示语义，提升了模型性能，在多个基准上取得领先结果。**

- **链接: [http://arxiv.org/pdf/2507.21033v1](http://arxiv.org/pdf/2507.21033v1)**

> **作者:** Yuhan Wang; Siwei Yang; Bingchen Zhao; Letian Zhang; Qing Liu; Yuyin Zhou; Cihang Xie
>
> **摘要:** Recent advancements in large multimodal models like GPT-4o have set a new standard for high-fidelity, instruction-guided image editing. However, the proprietary nature of these models and their training data creates a significant barrier for open-source research. To bridge this gap, we introduce GPT-IMAGE-EDIT-1.5M, a publicly available, large-scale image-editing corpus containing more than 1.5 million high-quality triplets (instruction, source image, edited image). We systematically construct this dataset by leveraging the versatile capabilities of GPT-4o to unify and refine three popular image-editing datasets: OmniEdit, HQ-Edit, and UltraEdit. Specifically, our methodology involves 1) regenerating output images to enhance visual quality and instruction alignment, and 2) selectively rewriting prompts to improve semantic clarity. To validate the efficacy of our dataset, we fine-tune advanced open-source models on GPT-IMAGE-EDIT-1.5M. The empirical results are exciting, e.g., the fine-tuned FluxKontext achieves highly competitive performance across a comprehensive suite of benchmarks, including 7.24 on GEdit-EN, 3.80 on ImgEdit-Full, and 8.78 on Complex-Edit, showing stronger instruction following and higher perceptual quality while maintaining identity. These scores markedly exceed all previously published open-source methods and substantially narrow the gap to leading proprietary models. We hope the full release of GPT-IMAGE-EDIT-1.5M can help to catalyze further open research in instruction-guided image editing.
>
---
#### [new 026] Local Prompt Adaptation for Style-Consistent Multi-Object Generation in Diffusion Models
- **分类: cs.CV; cs.AI; cs.MA**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型在生成多物体、风格一致场景时的布局控制和风格统一问题。论文提出了一种无需训练的局部提示适应方法（LPA），通过将提示分解为内容和风格标记，并在U-Net注意力层中选择性注入，提升了生成图像的风格一致性和空间连贯性。**

- **链接: [http://arxiv.org/pdf/2507.20094v1](http://arxiv.org/pdf/2507.20094v1)**

> **作者:** Ankit Sanjyal
>
> **备注:** 10 Pages, 8 figures, pre-print
>
> **摘要:** Diffusion models have become a powerful backbone for text-to-image generation, enabling users to synthesize high-quality visuals from natural language prompts. However, they often struggle with complex prompts involving multiple objects and global or local style specifications. In such cases, the generated scenes tend to lack style uniformity and spatial coherence, limiting their utility in creative and controllable content generation. In this paper, we propose a simple, training-free architectural method called Local Prompt Adaptation (LPA). Our method decomposes the prompt into content and style tokens, and injects them selectively into the U-Net's attention layers at different stages. By conditioning object tokens early and style tokens later in the generation process, LPA enhances both layout control and stylistic consistency. We evaluate our method on a custom benchmark of 50 style-rich prompts across five categories and compare against strong baselines including Composer, MultiDiffusion, Attend-and-Excite, LoRA, and SDXL. Our approach outperforms prior work on both CLIP score and style consistency metrics, offering a new direction for controllable, expressive diffusion-based generation.
>
---
#### [new 027] An Automated Deep Segmentation and Spatial-Statistics Approach for Post-Blast Rock Fragmentation Assessment
- **分类: cs.CV; stat.ML**

- **简介: 该论文属于图像分割与空间分析任务，旨在解决爆破后岩石碎裂程度的自动评估问题。作者提出了一种端到端方法，结合改进的YOLO12l-seg模型实现实例分割，并通过空间统计方法提取多指标特征，以快速、准确评估爆破效果，适用于野外实际应用。**

- **链接: [http://arxiv.org/pdf/2507.20126v1](http://arxiv.org/pdf/2507.20126v1)**

> **作者:** Yukun Yang
>
> **摘要:** We introduce an end-to-end pipeline that leverages a fine-tuned YOLO12l-seg model -- trained on over 500 annotated post-blast images -- to deliver real-time instance segmentation (Box mAP@0.5 ~ 0.769, Mask mAP@0.5 ~ 0.800 at ~ 15 FPS). High-fidelity masks are converted into normalized 3D coordinates, from which we extract multi-metric spatial descriptors: principal component directions, kernel density hotspots, size-depth regression, and Delaunay edge statistics. We present four representative examples to illustrate key fragmentation patterns. Experimental results confirm the framework's accuracy, robustness to small-object crowding, and feasibility for rapid, automated blast-effect assessment in field conditions.
>
---
#### [new 028] Automatic camera orientation estimation for a partially calibrated camera above a plane with a line at known planar distance
- **分类: cs.CV**

- **简介: 论文研究如何通过部分标定的摄像头，在仅知其高度和一个平面上参考直线的情况下，估计摄像头的翻滚和俯仰角度。任务是姿态估计，问题为在缺乏完整标定信息时的摄像头方向估计，工作是基于逆投影几何和几何约束推导出有效估计方法。**

- **链接: [http://arxiv.org/pdf/2507.20689v1](http://arxiv.org/pdf/2507.20689v1)**

> **作者:** Gergely Dinya; Anna Gelencsér-Horváth
>
> **摘要:** We present a derivation for estimating the roll and pitch orientation of a partially calibrated camera mounted above a planar surface, using minimal scene information. Specifically, we assume known intrinsic parameters and a fixed height between the camera and the observed plane. By detecting a single straight reference line at a known planar distance -- such as the edge between a floor and a wall -- we estimate the roll and pitch angles via inverse projection geometry. The method leverages geometric constraints and the camera model, including lens distortion correction. This approach is suitable for scenarios where full calibration is impractical and offers a lightweight alternative for multi-camera systems operating in constrained environments.
>
---
#### [new 029] DriveIndia: An Object Detection Dataset for Diverse Indian Traffic Scenes
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务中的目标检测领域，旨在解决印度复杂交通场景下的目标检测问题。作者构建了一个大规模数据集DriveIndia，包含66,986张高分辨率图像和24类交通相关物体，并提供了YOLO格式标注。数据覆盖多种天气、光照、道路环境和交通状况，适用于研究鲁棒性强、泛化能力高的目标检测模型。论文还提供了基于YOLO系列模型的基准结果，最高达到78.7%的mAP₅₀。**

- **链接: [http://arxiv.org/pdf/2507.19912v1](http://arxiv.org/pdf/2507.19912v1)**

> **作者:** Rishav Kumar; D. Santhosh Reddy; P. Rajalakshmi
>
> **备注:** Accepted at ITSC 2025 Conference
>
> **摘要:** We introduce \textbf{DriveIndia}, a large-scale object detection dataset purpose-built to capture the complexity and unpredictability of Indian traffic environments. The dataset contains \textbf{66,986 high-resolution images} annotated in YOLO format across \textbf{24 traffic-relevant object categories}, encompassing diverse conditions such as varied weather (fog, rain), illumination changes, heterogeneous road infrastructure, and dense, mixed traffic patterns and collected over \textbf{120+ hours} and covering \textbf{3,400+ kilometers} across urban, rural, and highway routes. DriveIndia offers a comprehensive benchmark for real-world autonomous driving challenges. We provide baseline results using state-of-the-art \textbf{YOLO family models}, with the top-performing variant achieving a $mAP_{50}$ of \textbf{78.7\%}. Designed to support research in robust, generalizable object detection under uncertain road conditions, DriveIndia will be publicly available via the TiHAN-IIT Hyderabad dataset repository (https://tihan.iith.ac.in/tiand-datasets/).
>
---
#### [new 030] MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于语音驱动三维人脸动画生成任务，旨在解决无需额外先验信息下生成个性化说话风格面部动画的问题。作者提出MemoryTalker，通过音频引导的风格化记忆机制，实现仅凭音频输入生成逼真且具个性化的面部动作。**

- **链接: [http://arxiv.org/pdf/2507.20562v1](http://arxiv.org/pdf/2507.20562v1)**

> **作者:** Hyung Kyu Kim; Sangmin Lee; Hak Gu Kim
>
> **备注:** Accepted for ICCV 2025 Project Page: https://cau-irislab.github.io/ICCV25-MemoryTalker/
>
> **摘要:** Speech-driven 3D facial animation aims to synthesize realistic facial motion sequences from given audio, matching the speaker's speaking style. However, previous works often require priors such as class labels of a speaker or additional 3D facial meshes at inference, which makes them fail to reflect the speaking style and limits their practical use. To address these issues, we propose MemoryTalker which enables realistic and accurate 3D facial motion synthesis by reflecting speaking style only with audio input to maximize usability in applications. Our framework consists of two training stages: 1-stage is storing and retrieving general motion (i.e., Memorizing), and 2-stage is to perform the personalized facial motion synthesis (i.e., Animating) with the motion memory stylized by the audio-driven speaking style feature. In this second stage, our model learns about which facial motion types should be emphasized for a particular piece of audio. As a result, our MemoryTalker can generate a reliable personalized facial animation without additional prior information. With quantitative and qualitative evaluations, as well as user study, we show the effectiveness of our model and its performance enhancement for personalized facial animation over state-of-the-art methods.
>
---
#### [new 031] Bias Analysis for Synthetic Face Detection: A Case Study of the Impact of Facial Attribute
- **分类: cs.CV**

- **简介: 该论文属于图像检测任务，旨在解决合成人脸检测中的偏见问题。通过构建评估框架，使用均匀分布的面部属性数据，分析五个最先进检测模型的偏见水平，并探讨偏见来源与训练数据分布及模型激活图的关系。**

- **链接: [http://arxiv.org/pdf/2507.19705v1](http://arxiv.org/pdf/2507.19705v1)**

> **作者:** Asmae Lamsaf; Lucia Cascone; Hugo Proença; João Neves
>
> **摘要:** Bias analysis for synthetic face detection is bound to become a critical topic in the coming years. Although many detection models have been developed and several datasets have been released to reliably identify synthetic content, one crucial aspect has been largely overlooked: these models and training datasets can be biased, leading to failures in detection for certain demographic groups and raising significant social, legal, and ethical issues. In this work, we introduce an evaluation framework to contribute to the analysis of bias of synthetic face detectors with respect to several facial attributes. This framework exploits synthetic data generation, with evenly distributed attribute labels, for mitigating any skew in the data that could otherwise influence the outcomes of bias analysis. We build on the proposed framework to provide an extensive case study of the bias level of five state-of-the-art detectors in synthetic datasets with 25 controlled facial attributes. While the results confirm that, in general, synthetic face detectors are biased towards the presence/absence of specific facial attributes, our study also sheds light on the origins of the observed bias through the analysis of the correlations with the balancing of facial attributes in the training sets of the detectors, and the analysis of detectors activation maps in image pairs with controlled attribute modifications.
>
---
#### [new 032] Indian Sign Language Detection for Real-Time Translation using Machine Learning
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与机器学习任务，旨在解决印度手语（ISL）实时翻译问题。为帮助听障人士沟通，研究者构建了一个基于CNN的模型，结合MediaPipe实现手势识别与翻译，准确率达99.95%，具有高实用性。**

- **链接: [http://arxiv.org/pdf/2507.20414v1](http://arxiv.org/pdf/2507.20414v1)**

> **作者:** Rajat Singhal; Jatin Gupta; Akhil Sharma; Anushka Gupta; Navya Sharma
>
> **备注:** 7 pages, 6 figures, 2 tables. Accepted for publication at the 6th International Conference on Recent Advances in Information Technology (RAIT 2025). This is the accepted version (preprint); the final published version will appear in IEEE Xplore
>
> **摘要:** Gestural language is used by deaf & mute communities to communicate through hand gestures & body movements that rely on visual-spatial patterns known as sign languages. Sign languages, which rely on visual-spatial patterns of hand gestures & body movements, are the primary mode of communication for deaf & mute communities worldwide. Effective communication is fundamental to human interaction, yet individuals in these communities often face significant barriers due to a scarcity of skilled interpreters & accessible translation technologies. This research specifically addresses these challenges within the Indian context by focusing on Indian Sign Language (ISL). By leveraging machine learning, this study aims to bridge the critical communication gap for the deaf & hard-of-hearing population in India, where technological solutions for ISL are less developed compared to other global sign languages. We propose a robust, real-time ISL detection & translation system built upon a Convolutional Neural Network (CNN). Our model is trained on a comprehensive ISL dataset & demonstrates exceptional performance, achieving a classification accuracy of 99.95%. This high precision underscores the model's capability to discern the nuanced visual features of different signs. The system's effectiveness is rigorously evaluated using key performance metrics, including accuracy, F1 score, precision & recall, ensuring its reliability for real-world applications. For real-time implementation, the framework integrates MediaPipe for precise hand tracking & motion detection, enabling seamless translation of dynamic gestures. This paper provides a detailed account of the model's architecture, the data preprocessing pipeline & the classification methodology. The research elaborates the model architecture, preprocessing & classification methodologies for enhancing communication in deaf & mute communities.
>
---
#### [new 033] Second Competition on Presentation Attack Detection on ID Card
- **分类: cs.CV**

- **简介: 该论文属于身份验证安全任务，旨在解决ID卡片呈现攻击检测（PAD）问题。工作包括举办第二届竞赛，设置两个评估轨道，提供新数据集，搭建自动评估平台。结果显示检测技术有所进步，但仍存在挑战，尤其是真实样本数量不足的问题。**

- **链接: [http://arxiv.org/pdf/2507.20404v1](http://arxiv.org/pdf/2507.20404v1)**

> **作者:** Juan E. Tapia; Mario Nieto; Juan M. Espin; Alvaro S. Rocamora; Javier Barrachina; Naser Damer; Christoph Busch; Marija Ivanovska; Leon Todorov; Renat Khizbullin; Lazar Lazarevich; Aleksei Grishin; Daniel Schulz; Sebastian Gonzalez; Amir Mohammadi; Ketan Kotwal; Sebastien Marcel; Raghavendra Mudgalgundurao; Kiran Raja; Patrick Schuch; Sushrut Patwardhan; Raghavendra Ramachandra; Pedro Couto Pereira; Joao Ribeiro Pinto; Mariana Xavier; Andrés Valenzuela; Rodrigo Lara; Borut Batagelj; Marko Peterlin; Peter Peer; Ajnas Muhammed; Diogo Nunes; Nuno Gonçalves
>
> **摘要:** This work summarises and reports the results of the second Presentation Attack Detection competition on ID cards. This new version includes new elements compared to the previous one. (1) An automatic evaluation platform was enabled for automatic benchmarking; (2) Two tracks were proposed in order to evaluate algorithms and datasets, respectively; and (3) A new ID card dataset was shared with Track 1 teams to serve as the baseline dataset for the training and optimisation. The Hochschule Darmstadt, Fraunhofer-IGD, and Facephi company jointly organised this challenge. 20 teams were registered, and 74 submitted models were evaluated. For Track 1, the "Dragons" team reached first place with an Average Ranking and Equal Error rate (EER) of AV-Rank of 40.48% and 11.44% EER, respectively. For the more challenging approach in Track 2, the "Incode" team reached the best results with an AV-Rank of 14.76% and 6.36% EER, improving on the results of the first edition of 74.30% and 21.87% EER, respectively. These results suggest that PAD on ID cards is improving, but it is still a challenging problem related to the number of images, especially of bona fide images.
>
---
#### [new 034] Predicting Brain Responses To Natural Movies With Multimodal LLMs
- **分类: cs.CV; cs.AI; q-bio.NC**

- **简介: 该论文属于脑响应预测任务，旨在通过多模态大语言模型预测人脑对自然电影的反应。研究团队结合多种预训练模型提取视频、语音、文本等特征，将其映射到fMRI时间序列对应的皮层区域。通过模型集成和优化，最终在挑战赛中取得良好成绩，展示了多模态融合与简单架构的有效性。**

- **链接: [http://arxiv.org/pdf/2507.19956v1](http://arxiv.org/pdf/2507.19956v1)**

> **作者:** Cesar Kadir Torrico Villanueva; Jiaxin Cindy Tu; Mihir Tripathy; Connor Lane; Rishab Iyer; Paul S. Scotti
>
> **备注:** Code available at https://github.com/MedARC-AI/algonauts2025
>
> **摘要:** We present MedARC's team solution to the Algonauts 2025 challenge. Our pipeline leveraged rich multimodal representations from various state-of-the-art pretrained models across video (V-JEPA2), speech (Whisper), text (Llama 3.2), vision-text (InternVL3), and vision-text-audio (Qwen2.5-Omni). These features extracted from the models were linearly projected to a latent space, temporally aligned to the fMRI time series, and finally mapped to cortical parcels through a lightweight encoder comprising a shared group head plus subject-specific residual heads. We trained hundreds of model variants across hyperparameter settings, validated them on held-out movies and assembled ensembles targeted to each parcel in each subject. Our final submission achieved a mean Pearson's correlation of 0.2085 on the test split of withheld out-of-distribution movies, placing our team in fourth place for the competition. We further discuss a last-minute optimization that would have raised us to second place. Our results highlight how combining features from models trained in different modalities, using a simple architecture consisting of shared-subject and single-subject components, and conducting comprehensive model selection and ensembling improves generalization of encoding models to novel movie stimuli. All code is available on GitHub.
>
---
#### [new 035] AutoSign: Direct Pose-to-Text Translation for Continuous Sign Language Recognition
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出AutoSign，用于连续手语识别任务，旨在将姿态序列直接翻译为文本。传统方法依赖多阶段流程，存在误差传播和扩展性差等问题。AutoSign采用解码器-only的Transformer结构，结合时间压缩模块和预训练模型，实现端到端翻译，提升了识别准确率。**

- **链接: [http://arxiv.org/pdf/2507.19840v1](http://arxiv.org/pdf/2507.19840v1)**

> **作者:** Samuel Ebimobowei Johnny; Blessed Guda; Andrew Blayama Stephen; Assane Gueye
>
> **备注:** Paper to appear at the 1st Workshop in Multimodal Sign Language Recognition at ICCV 2025
>
> **摘要:** Continuously recognizing sign gestures and converting them to glosses plays a key role in bridging the gap between the hearing and hearing-impaired communities. This involves recognizing and interpreting the hands, face, and body gestures of the signer, which pose a challenge as it involves a combination of all these features. Continuous Sign Language Recognition (CSLR) methods rely on multi-stage pipelines that first extract visual features, then align variable-length sequences with target glosses using CTC or HMM-based approaches. However, these alignment-based methods suffer from error propagation across stages, overfitting, and struggle with vocabulary scalability due to the intermediate gloss representation bottleneck. To address these limitations, we propose AutoSign, an autoregressive decoder-only transformer that directly translates pose sequences to natural language text, bypassing traditional alignment mechanisms entirely. The use of this decoder-only approach allows the model to directly map between the features and the glosses without the need for CTC loss while also directly learning the textual dependencies in the glosses. Our approach incorporates a temporal compression module using 1D CNNs to efficiently process pose sequences, followed by AraGPT2, a pre-trained Arabic decoder, to generate text (glosses). Through comprehensive ablation studies, we demonstrate that hand and body gestures provide the most discriminative features for signer-independent CSLR. By eliminating the multi-stage pipeline, AutoSign achieves substantial improvements on the Isharah-1000 dataset, achieving an improvement of up to 6.1\% in WER score compared to the best existing method.
>
---
#### [new 036] AgroBench: Vision-Language Model Benchmark in Agriculture
- **分类: cs.CV**

- **简介: 该论文属于农业领域的视觉-语言模型（VLM）基准构建任务，旨在解决农业任务中自动化理解与交互能力评估的问题。作者提出了AgroBench，一个由农业专家标注的基准数据集，涵盖203种作物和682种病害类别，用于全面评估VLM模型在农作物疾病识别等任务中的表现。论文分析了现有VLM模型的不足，并为未来模型改进提供了方向。**

- **链接: [http://arxiv.org/pdf/2507.20519v1](http://arxiv.org/pdf/2507.20519v1)**

> **作者:** Risa Shinoda; Nakamasa Inoue; Hirokatsu Kataoka; Masaki Onishi; Yoshitaka Ushiku
>
> **备注:** ICCV 2025
>
> **摘要:** Precise automated understanding of agricultural tasks such as disease identification is essential for sustainable crop production. Recent advances in vision-language models (VLMs) are expected to further expand the range of agricultural tasks by facilitating human-model interaction through easy, text-based communication. Here, we introduce AgroBench (Agronomist AI Benchmark), a benchmark for evaluating VLM models across seven agricultural topics, covering key areas in agricultural engineering and relevant to real-world farming. Unlike recent agricultural VLM benchmarks, AgroBench is annotated by expert agronomists. Our AgroBench covers a state-of-the-art range of categories, including 203 crop categories and 682 disease categories, to thoroughly evaluate VLM capabilities. In our evaluation on AgroBench, we reveal that VLMs have room for improvement in fine-grained identification tasks. Notably, in weed identification, most open-source VLMs perform close to random. With our wide range of topics and expert-annotated categories, we analyze the types of errors made by VLMs and suggest potential pathways for future VLM development. Our dataset and code are available at https://dahlian00.github.io/AgroBenchPage/ .
>
---
#### [new 037] Learning Only with Images: Visual Reinforcement Learning with Reasoning, Rendering, and Visual Feedback
- **分类: cs.CV**

- **简介: 该论文属于视觉强化学习任务，旨在解决多模态大语言模型（MLLMs）依赖图像-文本监督的问题。作者提出RRVF框架，通过推理、渲染与视觉反馈的闭环过程，结合强化学习，使模型仅凭视觉反馈自我修正，减少了对文本监督的依赖，提升了图像生成代码任务中的性能。**

- **链接: [http://arxiv.org/pdf/2507.20766v1](http://arxiv.org/pdf/2507.20766v1)**

> **作者:** Yang Chen; Yufan Shen; Wenxuan Huang; Shen Zhou; Qunshu Lin; Xinyu Cai; Zhi Yu; Botian Shi; Yu Qiao
>
> **摘要:** Multimodal Large Language Models (MLLMs) have exhibited impressive performance across various visual tasks. Subsequent investigations into enhancing their visual reasoning abilities have significantly expanded their performance envelope. However, a critical bottleneck in the advancement of MLLMs toward deep visual reasoning is their heavy reliance on curated image-text supervision. To solve this problem, we introduce a novel framework termed ``Reasoning-Rendering-Visual-Feedback'' (RRVF), which enables MLLMs to learn complex visual reasoning from only raw images. This framework builds on the ``Asymmetry of Verification'' principle to train MLLMs, i.e., verifying the rendered output against a source image is easier than generating it. We demonstrate that this relative ease provides an ideal reward signal for optimization via Reinforcement Learning (RL) training, reducing the reliance on the image-text supervision. Guided by the above principle, RRVF implements a closed-loop iterative process encompassing reasoning, rendering, and visual feedback components, enabling the model to perform self-correction through multi-turn interactions and tool invocation, while this pipeline can be optimized by the GRPO algorithm in an end-to-end manner. Extensive experiments on image-to-code generation for data charts and web interfaces show that RRVF substantially outperforms existing open-source MLLMs and surpasses supervised fine-tuning baselines. Our findings demonstrate that systems driven by purely visual feedback present a viable path toward more robust and generalizable reasoning models without requiring explicit supervision. Code will be available at https://github.com/L-O-I/RRVF.
>
---
#### [new 038] Latest Object Memory Management for Temporally Consistent Video Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频实例分割（VIS）任务，旨在解决视频中实例的长期跟踪与身份一致性问题。作者提出最新对象记忆管理（LOMM）方法，通过最新对象记忆（LOM）持续更新对象状态，并引入解耦对象关联（DOA）策略，分别处理新出现和已存在的对象，提升匹配准确性和身份一致性，尤其在长视频和动态场景中表现优异，取得了YouTube-VIS 2022数据集上的最优性能。**

- **链接: [http://arxiv.org/pdf/2507.19754v1](http://arxiv.org/pdf/2507.19754v1)**

> **作者:** Seunghun Lee; Jiwan Seo; Minwoo Choi; Kiljoon Han; Jaehoon Jeong; Zane Durante; Ehsan Adeli; Sang Hyun Park; Sunghoon Im
>
> **备注:** ICCV 2025. Code: https://github.com/Seung-Hun-Lee/LOMM
>
> **摘要:** In this paper, we present Latest Object Memory Management (LOMM) for temporally consistent video instance segmentation that significantly improves long-term instance tracking. At the core of our method is Latest Object Memory (LOM), which robustly tracks and continuously updates the latest states of objects by explicitly modeling their presence in each frame. This enables consistent tracking and accurate identity management across frames, enhancing both performance and reliability through the VIS process. Moreover, we introduce Decoupled Object Association (DOA), a strategy that separately handles newly appearing and already existing objects. By leveraging our memory system, DOA accurately assigns object indices, improving matching accuracy and ensuring stable identity consistency, even in dynamic scenes where objects frequently appear and disappear. Extensive experiments and ablation studies demonstrate the superiority of our method over traditional approaches, setting a new benchmark in VIS. Notably, our LOMM achieves state-of-the-art AP score of 54.0 on YouTube-VIS 2022, a dataset known for its challenging long videos. Project page: https://seung-hun-lee.github.io/projects/LOMM/
>
---
#### [new 039] PUMPS: Skeleton-Agnostic Point-based Universal Motion Pre-Training for Synthesis in Human Motion Tasks
- **分类: cs.CV**

- **简介: 该论文属于3D人体动作合成任务，旨在解决不同骨骼结构间动作数据难以迁移的问题。作者提出PUMPS模型，基于时间点云（TPC）实现动作表示与合成，通过自编码架构和点配对优化，实现了跨骨骼的动作预测、过渡生成和关键帧插值，表现出色。**

- **链接: [http://arxiv.org/pdf/2507.20170v1](http://arxiv.org/pdf/2507.20170v1)**

> **作者:** Clinton Ansun Mo; Kun Hu; Chengjiang Long; Dong Yuan; Wan-Chi Siu; Zhiyong Wang
>
> **备注:** Accepted for publication in ICCV 2025
>
> **摘要:** Motion skeletons drive 3D character animation by transforming bone hierarchies, but differences in proportions or structure make motion data hard to transfer across skeletons, posing challenges for data-driven motion synthesis. Temporal Point Clouds (TPCs) offer an unstructured, cross-compatible motion representation. Though reversible with skeletons, TPCs mainly serve for compatibility, not for direct motion task learning. Doing so would require data synthesis capabilities for the TPC format, which presents unexplored challenges regarding its unique temporal consistency and point identifiability. Therefore, we propose PUMPS, the primordial autoencoder architecture for TPC data. PUMPS independently reduces frame-wise point clouds into sampleable feature vectors, from which a decoder extracts distinct temporal points using latent Gaussian noise vectors as sampling identifiers. We introduce linear assignment-based point pairing to optimise the TPC reconstruction process, and negate the use of expensive point-wise attention mechanisms in the architecture. Using these latent features, we pre-train a motion synthesis model capable of performing motion prediction, transition generation, and keyframe interpolation. For these pre-training tasks, PUMPS performs remarkably well even without native dataset supervision, matching state-of-the-art performance. When fine-tuned for motion denoising or estimation, PUMPS outperforms many respective methods without deviating from its generalist architecture.
>
---
#### [new 040] Local2Global query Alignment for Video Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频实例分割任务，旨在解决在线视频分割中时间不一致、噪声积累和遮挡等问题。工作提出Local2Global框架，引入局部与全局查询及L2G-aligner模块，实现帧间对齐与信息融合，提升分割一致性与性能。**

- **链接: [http://arxiv.org/pdf/2507.20120v1](http://arxiv.org/pdf/2507.20120v1)**

> **作者:** Rajat Koner; Zhipeng Wang; Srinivas Parthasarathy; Chinghang Chen
>
> **摘要:** Online video segmentation methods excel at handling long sequences and capturing gradual changes, making them ideal for real-world applications. However, achieving temporally consistent predictions remains a challenge, especially with gradual accumulation of noise or drift in on-line propagation, abrupt occlusions and scene transitions. This paper introduces Local2Global, an online framework, for video instance segmentation, exhibiting state-of-the-art performance with simple baseline and training purely in online fashion. Leveraging the DETR-based query propagation framework, we introduce two novel sets of queries:(1) local queries that capture initial object-specific spatial features from each frame and (2) global queries containing past spatio-temporal representations. We propose the L2G-aligner, a novel lightweight transformer decoder, to facilitate an early alignment between local and global queries. This alignment allows our model to effectively utilize current frame information while maintaining temporal consistency, producing a smooth transition between frames. Furthermore, L2G-aligner is integrated within the segmentation model, without relying on additional complex heuristics, or memory mechanisms. Extensive experiments across various challenging VIS and VPS datasets showcase the superiority of our method with simple online training, surpassing current benchmarks without bells and rings. For instance, we achieve 54.3 and 49.4 AP on Youtube-VIS-19/-21 datasets and 37.0 AP on OVIS dataset respectively withthe ResNet-50 backbone.
>
---
#### [new 041] DS-Det: Single-Query Paradigm and Attention Disentangled Learning for Flexible Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决基于Transformer的目标检测模型中固定查询机制导致的灵活性差和效率低问题。作者提出DS-Det，采用单查询范式和注意力解耦学习，缓解查询歧义和交互干扰，提升检测效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.19807v1](http://arxiv.org/pdf/2507.19807v1)**

> **作者:** Guiping Cao; Xiangyuan Lan; Wenjian Huang; Jianguo Zhang; Dongmei Jiang; Yaowei Wang
>
> **摘要:** Popular transformer detectors have achieved promising performance through query-based learning using attention mechanisms. However, the roles of existing decoder query types (e.g., content query and positional query) are still underexplored. These queries are generally predefined with a fixed number (fixed-query), which limits their flexibility. We find that the learning of these fixed-query is impaired by Recurrent Opposing inTeractions (ROT) between two attention operations: Self-Attention (query-to-query) and Cross-Attention (query-to-encoder), thereby degrading decoder efficiency. Furthermore, "query ambiguity" arises when shared-weight decoder layers are processed with both one-to-one and one-to-many label assignments during training, violating DETR's one-to-one matching principle. To address these challenges, we propose DS-Det, a more efficient detector capable of detecting a flexible number of objects in images. Specifically, we reformulate and introduce a new unified Single-Query paradigm for decoder modeling, transforming the fixed-query into flexible. Furthermore, we propose a simplified decoder framework through attention disentangled learning: locating boxes with Cross-Attention (one-to-many process), deduplicating predictions with Self-Attention (one-to-one process), addressing "query ambiguity" and "ROT" issues directly, and enhancing decoder efficiency. We further introduce a unified PoCoo loss that leverages box size priors to prioritize query learning on hard samples such as small objects. Extensive experiments across five different backbone models on COCO2017 and WiderPerson datasets demonstrate the general effectiveness and superiority of DS-Det. The source codes are available at https://github.com/Med-Process/DS-Det/.
>
---
#### [new 042] Security Tensors as a Cross-Modal Bridge: Extending Text-Aligned Safety to Vision in LVLM
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态安全任务，旨在解决视觉-语言模型（LVLM）中视觉模态缺乏文本对齐安全机制的问题。作者提出“安全张量”方法，通过在推理阶段引入可训练输入向量，将文本安全机制扩展到视觉处理，无需修改模型参数。实验表明该方法能有效增强模型对有害图像的识别与拒绝能力，同时保持对正常任务的性能。**

- **链接: [http://arxiv.org/pdf/2507.20994v1](http://arxiv.org/pdf/2507.20994v1)**

> **作者:** Shen Li; Liuyi Yao; Wujia Niu; Lan Zhang; Yaliang Li
>
> **备注:** Codes and data are available at https://github.com/listen0425/Security-Tensors
>
> **摘要:** Large visual-language models (LVLMs) integrate aligned large language models (LLMs) with visual modules to process multimodal inputs. However, the safety mechanisms developed for text-based LLMs do not naturally extend to visual modalities, leaving LVLMs vulnerable to harmful image inputs. To address this cross-modal safety gap, we introduce security tensors - trainable input vectors applied during inference through either the textual or visual modality. These tensors transfer textual safety alignment to visual processing without modifying the model's parameters. They are optimized using a curated dataset containing (i) malicious image-text pairs requiring rejection, (ii) contrastive benign pairs with text structurally similar to malicious queries, with the purpose of being contrastive examples to guide visual reliance, and (iii) general benign samples preserving model functionality. Experimental results demonstrate that both textual and visual security tensors significantly enhance LVLMs' ability to reject diverse harmful visual inputs while maintaining near-identical performance on benign tasks. Further internal analysis towards hidden-layer representations reveals that security tensors successfully activate the language module's textual "safety layers" in visual inputs, thereby effectively extending text-based safety to the visual modality.
>
---
#### [new 043] The Importance of Facial Features in Vision-based Sign Language Recognition: Eyes, Mouth or Full Face?
- **分类: cs.CV; cs.CL; eess.IV**

- **简介: 该论文属于视觉手语识别任务，旨在解决非手动面部特征在识别中的作用问题。作者通过对比眼睛、嘴巴和全脸的贡献，结合CNN和Transformer模型，发现嘴巴特征最关键，显著提升识别准确率，强调了面部特征对自动手语识别的重要性。**

- **链接: [http://arxiv.org/pdf/2507.20884v1](http://arxiv.org/pdf/2507.20884v1)**

> **作者:** Dinh Nam Pham; Eleftherios Avramidis
>
> **备注:** Accepted at 9th International Workshop on Sign Language Translation and Avatar Technologies @ ACM IVA'25
>
> **摘要:** Non-manual facial features play a crucial role in sign language communication, yet their importance in automatic sign language recognition (ASLR) remains underexplored. While prior studies have shown that incorporating facial features can improve recognition, related work often relies on hand-crafted feature extraction and fails to go beyond the comparison of manual features versus the combination of manual and facial features. In this work, we systematically investigate the contribution of distinct facial regionseyes, mouth, and full faceusing two different deep learning models (a CNN-based model and a transformer-based model) trained on an SLR dataset of isolated signs with randomly selected classes. Through quantitative performance and qualitative saliency map evaluation, we reveal that the mouth is the most important non-manual facial feature, significantly improving accuracy. Our findings highlight the necessity of incorporating facial features in ASLR.
>
---
#### [new 044] SWIFT: A General Sensitive Weight Identification Framework for Fast Sensor-Transfer Pansharpening
- **分类: cs.CV**

- **简介: 该论文属于图像融合任务，旨在解决跨传感器泛化问题。传统方法需耗时重训练，而SWIFT通过采样策略与权重识别，快速适配新传感器数据，显著提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.20311v1](http://arxiv.org/pdf/2507.20311v1)**

> **作者:** Zeyu Xia; Chenxi Sun; Tianyu Xin; Yubo Zeng; Haoyu Chen; Liang-Jian Deng
>
> **摘要:** Pansharpening aims to fuse high-resolution panchromatic (PAN) images with low-resolution multispectral (LRMS) images to generate high-resolution multispectral (HRMS) images. Although deep learning-based methods have achieved promising performance, they generally suffer from severe performance degradation when applied to data from unseen sensors. Adapting these models through full-scale retraining or designing more complex architectures is often prohibitively expensive and impractical for real-world deployment. To address this critical challenge, we propose a fast and general-purpose framework for cross-sensor adaptation, SWIFT (Sensitive Weight Identification for Fast Transfer). Specifically, SWIFT employs an unsupervised sampling strategy based on data manifold structures to balance sample selection while mitigating the bias of traditional Farthest Point Sampling, efficiently selecting only 3\% of the most informative samples from the target domain. This subset is then used to probe a source-domain pre-trained model by analyzing the gradient behavior of its parameters, allowing for the quick identification and subsequent update of only the weight subset most sensitive to the domain shift. As a plug-and-play framework, SWIFT can be applied to various existing pansharpening models. Extensive experiments demonstrate that SWIFT reduces the adaptation time from hours to approximately one minute on a single NVIDIA RTX 4090 GPU. The adapted models not only substantially outperform direct-transfer baselines but also achieve performance competitive with, and in some cases superior to, full retraining, establishing a new state-of-the-art on cross-sensor pansharpening tasks for the WorldView-2 and QuickBird datasets.
>
---
#### [new 045] Dual-Stream Global-Local Feature Collaborative Representation Network for Scene Classification of Mining Area
- **分类: cs.CV**

- **简介: 该论文属于遥感图像场景分类任务，旨在解决矿区复杂空间布局和多尺度特征带来的分类难题。论文提出了一种双流全局-局部特征协同表示网络，融合多源数据，通过多尺度全局Transformer分支、局部增强协同表示分支及双分支融合模块，提升矿区场景分类精度，整体准确率达83.63%。**

- **链接: [http://arxiv.org/pdf/2507.20216v1](http://arxiv.org/pdf/2507.20216v1)**

> **作者:** Shuqi Fan; Haoyi Wang; Xianju Li
>
> **摘要:** Scene classification of mining areas provides accurate foundational data for geological environment monitoring and resource development planning. This study fuses multi-source data to construct a multi-modal mine land cover scene classification dataset. A significant challenge in mining area classification lies in the complex spatial layout and multi-scale characteristics. By extracting global and local features, it becomes possible to comprehensively reflect the spatial distribution, thereby enabling a more accurate capture of the holistic characteristics of mining scenes. We propose a dual-branch fusion model utilizing collaborative representation to decompose global features into a set of key semantic vectors. This model comprises three key components:(1) Multi-scale Global Transformer Branch: It leverages adjacent large-scale features to generate global channel attention features for small-scale features, effectively capturing the multi-scale feature relationships. (2) Local Enhancement Collaborative Representation Branch: It refines the attention weights by leveraging local features and reconstructed key semantic sets, ensuring that the local context and detailed characteristics of the mining area are effectively integrated. This enhances the model's sensitivity to fine-grained spatial variations. (3) Dual-Branch Deep Feature Fusion Module: It fuses the complementary features of the two branches to incorporate more scene information. This fusion strengthens the model's ability to distinguish and classify complex mining landscapes. Finally, this study employs multi-loss computation to ensure a balanced integration of the modules. The overall accuracy of this model is 83.63%, which outperforms other comparative models. Additionally, it achieves the best performance across all other evaluation metrics.
>
---
#### [new 046] AIComposer: Any Style and Content Image Composition via Feature Integration
- **分类: cs.CV**

- **简介: 该论文属于图像合成任务，旨在解决跨域图像合成中的风格差异和依赖文本提示的问题。提出了AIComposer方法，通过特征集成实现无需文本提示的自然风格化合成。设计了多层感知网络与局部交叉注意力策略，有效保持前景内容并稳定风格化。建立了新基准数据集进行评估，结果显示其方法在多个指标上显著优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.20721v1](http://arxiv.org/pdf/2507.20721v1)**

> **作者:** Haowen Li; Zhenfeng Fan; Zhang Wen; Zhengzhou Zhu; Yunjin Li
>
> **摘要:** Image composition has advanced significantly with large-scale pre-trained T2I diffusion models. Despite progress in same-domain composition, cross-domain composition remains under-explored. The main challenges are the stochastic nature of diffusion models and the style gap between input images, leading to failures and artifacts. Additionally, heavy reliance on text prompts limits practical applications. This paper presents the first cross-domain image composition method that does not require text prompts, allowing natural stylization and seamless compositions. Our method is efficient and robust, preserving the diffusion prior, as it involves minor steps for backward inversion and forward denoising without training the diffuser. Our method also uses a simple multilayer perceptron network to integrate CLIP features from foreground and background, manipulating diffusion with a local cross-attention strategy. It effectively preserves foreground content while enabling stable stylization without a pre-stylization network. Finally, we create a benchmark dataset with diverse contents and styles for fair evaluation, addressing the lack of testing datasets for cross-domain image composition. Our method outperforms state-of-the-art techniques in both qualitative and quantitative evaluations, significantly improving the LPIPS score by 30.5% and the CSD metric by 18.1%. We believe our method will advance future research and applications. Code and benchmark at https://github.com/sherlhw/AIComposer.
>
---
#### [new 047] Ensemble Foreground Management for Unsupervised Object Discovery
- **分类: cs.CV**

- **简介: 该论文属于无监督目标发现（UOD）任务，旨在解决无需标注的情况下图像中目标检测与分割的问题。论文提出了UnionCut和UnionSeg方法，用于更准确地识别前景区域并判断目标数量，从而提升现有UOD方法在多个任务上的性能表现。**

- **链接: [http://arxiv.org/pdf/2507.20860v1](http://arxiv.org/pdf/2507.20860v1)**

> **作者:** Ziling Wu; Armaghan Moemeni; Praminda Caleb-Solly
>
> **备注:** Accepted by ICCV2025 (Highlight)
>
> **摘要:** Unsupervised object discovery (UOD) aims to detect and segment objects in 2D images without handcrafted annotations. Recent progress in self-supervised representation learning has led to some success in UOD algorithms. However, the absence of ground truth provides existing UOD methods with two challenges: 1) determining if a discovered region is foreground or background, and 2) knowing how many objects remain undiscovered. To address these two problems, previous solutions rely on foreground priors to distinguish if the discovered region is foreground, and conduct one or fixed iterations of discovery. However, the existing foreground priors are heuristic and not always robust, and a fixed number of discoveries leads to under or over-segmentation, since the number of objects in images varies. This paper introduces UnionCut, a robust and well-grounded foreground prior based on min-cut and ensemble methods that detects the union of foreground areas of an image, allowing UOD algorithms to identify foreground objects and stop discovery once the majority of the foreground union in the image is segmented. In addition, we propose UnionSeg, a distilled transformer of UnionCut that outputs the foreground union more efficiently and accurately. Our experiments show that by combining with UnionCut or UnionSeg, previous state-of-the-art UOD methods witness an increase in the performance of single object discovery, saliency detection and self-supervised instance segmentation on various benchmarks. The code is available at https://github.com/YFaris/UnionCut.
>
---
#### [new 048] A Structure-aware and Motion-adaptive Framework for 3D Human Pose Estimation with Mamba
- **分类: cs.CV**

- **简介: 该论文属于3D人体姿态估计任务，旨在解决现有方法难以建模复杂关节连接和运动差异的问题。作者提出SAMA框架，包含结构感知的状态整合器和运动自适应的状态调节器，分别捕捉空间关节结构和运动特性，提升姿态估计效果。**

- **链接: [http://arxiv.org/pdf/2507.19852v1](http://arxiv.org/pdf/2507.19852v1)**

> **作者:** Ye Lu; Jie Wang; Jianjun Gao; Rui Gong; Chen Cai; Kim-Hui Yap
>
> **备注:** 8 pages, 5 figures, conference
>
> **摘要:** Recent Mamba-based methods for the pose-lifting task tend to model joint dependencies by 2D-to-1D mapping with diverse scanning strategies. Though effective, they struggle to model intricate joint connections and uniformly process all joint motion trajectories while neglecting the intrinsic differences across motion characteristics. In this work, we propose a structure-aware and motion-adaptive framework to capture spatial joint topology along with diverse motion dynamics independently, named as SAMA. Specifically, SAMA consists of a Structure-aware State Integrator (SSI) and a Motion-adaptive State Modulator (MSM). The Structure-aware State Integrator is tasked with leveraging dynamic joint relationships to fuse information at both the joint feature and state levels in the state space, based on pose topology rather than sequential state transitions. The Motion-adaptive State Modulator is responsible for joint-specific motion characteristics recognition, thus applying tailored adjustments to diverse motion patterns across different joints. Through the above key modules, our algorithm enables structure-aware and motion-adaptive pose lifting. Extensive experiments across multiple benchmarks demonstrate that our algorithm achieves advanced results with fewer computational costs.
>
---
#### [new 049] Harnessing Diffusion-Yielded Score Priors for Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决去模糊、生成真实细节和像素级一致性问题。现有方法在质量、保真度和速度间难以平衡。论文提出HYPIR方法，通过预训练扩散模型初始化并进行对抗训练，无需扩散损失或迭代采样，提升了稳定性与收敛速度，实现高效高质量图像恢复。**

- **链接: [http://arxiv.org/pdf/2507.20590v1](http://arxiv.org/pdf/2507.20590v1)**

> **作者:** Xinqi Lin; Fanghua Yu; Jinfan Hu; Zhiyuan You; Wu Shi; Jimmy S. Ren; Jinjin Gu; Chao Dong
>
> **摘要:** Deep image restoration models aim to learn a mapping from degraded image space to natural image space. However, they face several critical challenges: removing degradation, generating realistic details, and ensuring pixel-level consistency. Over time, three major classes of methods have emerged, including MSE-based, GAN-based, and diffusion-based methods. However, they fail to achieve a good balance between restoration quality, fidelity, and speed. We propose a novel method, HYPIR, to address these challenges. Our solution pipeline is straightforward: it involves initializing the image restoration model with a pre-trained diffusion model and then fine-tuning it with adversarial training. This approach does not rely on diffusion loss, iterative sampling, or additional adapters. We theoretically demonstrate that initializing adversarial training from a pre-trained diffusion model positions the initial restoration model very close to the natural image distribution. Consequently, this initialization improves numerical stability, avoids mode collapse, and substantially accelerates the convergence of adversarial training. Moreover, HYPIR inherits the capabilities of diffusion models with rich user control, enabling text-guided restoration and adjustable texture richness. Requiring only a single forward pass, it achieves faster convergence and inference speed than diffusion-based methods. Extensive experiments show that HYPIR outperforms previous state-of-the-art methods, achieving efficient and high-quality image restoration.
>
---
#### [new 050] Style-Aware Blending and Prototype-Based Cross-Contrast Consistency for Semi-Supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于半监督医学图像分割任务，旨在解决标注数据有限下的模型训练问题。现有方法存在训练数据流分离导致的确认偏差和监督信息利用不充分的问题。论文提出风格感知混合和基于原型的跨对比一致性框架，通过统计矩匹配缓解分布差异，并利用伪标签中的有用信息，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2507.20729v1](http://arxiv.org/pdf/2507.20729v1)**

> **作者:** Chaowei Chen; Xiang Zhang; Honglie Guo; Shunfang Wang
>
> **摘要:** Weak-strong consistency learning strategies are widely employed in semi-supervised medical image segmentation to train models by leveraging limited labeled data and enforcing weak-to-strong consistency. However, existing methods primarily focus on designing and combining various perturbation schemes, overlooking the inherent potential and limitations within the framework itself. In this paper, we first identify two critical deficiencies: (1) separated training data streams, which lead to confirmation bias dominated by the labeled stream; and (2) incomplete utilization of supervisory information, which limits exploration of strong-to-weak consistency. To tackle these challenges, we propose a style-aware blending and prototype-based cross-contrast consistency learning framework. Specifically, inspired by the empirical observation that the distribution mismatch between labeled and unlabeled data can be characterized by statistical moments, we design a style-guided distribution blending module to break the independent training data streams. Meanwhile, considering the potential noise in strong pseudo-labels, we introduce a prototype-based cross-contrast strategy to encourage the model to learn informative supervisory signals from both weak-to-strong and strong-to-weak predictions, while mitigating the adverse effects of noise. Experimental results demonstrate the effectiveness and superiority of our framework across multiple medical segmentation benchmarks under various semi-supervised settings.
>
---
#### [new 051] $A^2R^2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像到LaTeX转换任务，旨在解决视觉语言模型（VLM）在精细视觉元素理解上的不足，导致LaTeX生成不准确的问题。作者提出了一种名为$A^2R^2$的新框架，结合注意力定位与迭代优化，提升模型预测质量。此外，还构建了一个用于评估的高难度数据集Img2LaTex-Hard-1K，并通过实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.20890v1](http://arxiv.org/pdf/2507.20890v1)**

> **作者:** Zhecheng Li; Guoxian Song; Yiwei Wang; Zhen Xiong; Junsong Yuan; Yujun Cai
>
> **摘要:** Img2LaTeX is a practically significant task that involves converting mathematical expressions or tabular data from images into LaTeX code. In recent years, vision-language models (VLMs) have demonstrated strong performance across a variety of visual understanding tasks, owing to their generalization capabilities. While some studies have explored the use of VLMs for the Img2LaTeX task, their performance often falls short of expectations. Empirically, VLMs sometimes struggle with fine-grained visual elements, leading to inaccurate LaTeX predictions. To address this challenge, we propose $A^2R^2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement, a framework that effectively integrates attention localization and iterative refinement within a visual reasoning framework, enabling VLMs to perform self-correction and progressively improve prediction quality. For effective evaluation, we introduce a new dataset, Img2LaTex-Hard-1K, consisting of 1,100 carefully curated and challenging examples designed to rigorously evaluate the capabilities of VLMs within this task domain. Extensive experimental results demonstrate that: (1) $A^2R^2$ significantly improves model performance across six evaluation metrics spanning both textual and visual levels, consistently outperforming other baseline methods; (2) Increasing the number of inference rounds yields notable performance gains, underscoring the potential of $A^2R^2$ in test-time scaling scenarios; (3) Ablation studies and human evaluations validate the practical effectiveness of our approach, as well as the strong synergy among its core components during inference.
>
---
#### [new 052] HydraMamba: Multi-Head State Space Model for Global Point Cloud Learning
- **分类: cs.CV**

- **简介: 论文提出HydraMamba，用于点云学习，解决现有方法在长程依赖建模和局部特征学习上的不足。基于状态空间模型（S6），设计了Shuffle序列化策略和ConvBiS6层，提升全局与局部特征融合能力，实现物体级与场景级任务的最优性能。**

- **链接: [http://arxiv.org/pdf/2507.19778v1](http://arxiv.org/pdf/2507.19778v1)**

> **作者:** Kanglin Qu; Pan Gao; Qun Dai; Yuanhao Sun
>
> **备注:** Accepted by MM '25
>
> **摘要:** The attention mechanism has become a dominant operator in point cloud learning, but its quadratic complexity leads to limited inter-point interactions, hindering long-range dependency modeling between objects. Due to excellent long-range modeling capability with linear complexity, the selective state space model (S6), as the core of Mamba, has been exploited in point cloud learning for long-range dependency interactions over the entire point cloud. Despite some significant progress, related works still suffer from imperfect point cloud serialization and lack of locality learning. To this end, we explore a state space model-based point cloud network termed HydraMamba to address the above challenges. Specifically, we design a shuffle serialization strategy, making unordered point sets better adapted to the causal nature of S6. Meanwhile, to overcome the deficiency of existing techniques in locality learning, we propose a ConvBiS6 layer, which is capable of capturing local geometries and global context dependencies synergistically. Besides, we propose MHS6 by extending the multi-head design to S6, further enhancing its modeling capability. HydraMamba achieves state-of-the-art results on various tasks at both object-level and scene-level. The code is available at https://github.com/Point-Cloud-Learning/HydraMamba.
>
---
#### [new 053] Detecting Visual Information Manipulation Attacks in Augmented Reality: A Multimodal Semantic Reasoning Approach
- **分类: cs.CV**

- **简介: 该论文属于安全检测任务，旨在识别增强现实（AR）中的视觉信息操纵（VIM）攻击。作者构建了AR-VIM数据集，并提出多模态语义推理框架VIM-Sense，融合视觉、语言和OCR技术，实现高效攻击检测，准确率达88.94%。**

- **链接: [http://arxiv.org/pdf/2507.20356v1](http://arxiv.org/pdf/2507.20356v1)**

> **作者:** Yanming Xiu; Maria Gorlatova
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** The virtual content in augmented reality (AR) can introduce misleading or harmful information, leading to semantic misunderstandings or user errors. In this work, we focus on visual information manipulation (VIM) attacks in AR where virtual content changes the meaning of real-world scenes in subtle but impactful ways. We introduce a taxonomy that categorizes these attacks into three formats: character, phrase, and pattern manipulation, and three purposes: information replacement, information obfuscation, and extra wrong information. Based on the taxonomy, we construct a dataset, AR-VIM. It consists of 452 raw-AR video pairs spanning 202 different scenes, each simulating a real-world AR scenario. To detect such attacks, we propose a multimodal semantic reasoning framework, VIM-Sense. It combines the language and visual understanding capabilities of vision-language models (VLMs) with optical character recognition (OCR)-based textual analysis. VIM-Sense achieves an attack detection accuracy of 88.94% on AR-VIM, consistently outperforming vision-only and text-only baselines. The system reaches an average attack detection latency of 7.07 seconds in a simulated video processing framework and 7.17 seconds in a real-world evaluation conducted on a mobile Android AR application.
>
---
#### [new 054] Controllable Feature Whitening for Hyperparameter-Free Bias Mitigation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于人工智能可信性任务，旨在解决深度神经网络学习数据集中虚假关联导致的偏差问题。提出了可控特征白化方法，通过消除特征间线性相关性来缓解偏差，无需正则化或对抗学习，实现公平性与性能的权衡控制，并在多个数据集上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2507.20284v1](http://arxiv.org/pdf/2507.20284v1)**

> **作者:** Yooshin Cho; Hanbyel Cho; Janghyeon Lee; HyeongGwon Hong; Jaesung Ahn; Junmo Kim
>
> **备注:** Accepted to ICCV 2025 (Poster)
>
> **摘要:** As the use of artificial intelligence rapidly increases, the development of trustworthy artificial intelligence has become important. However, recent studies have shown that deep neural networks are susceptible to learn spurious correlations present in datasets. To improve the reliability, we propose a simple yet effective framework called controllable feature whitening. We quantify the linear correlation between the target and bias features by the covariance matrix, and eliminate it through the whitening module. Our results systemically demonstrate that removing the linear correlations between features fed into the last linear classifier significantly mitigates the bias, while avoiding the need to model intractable higher-order dependencies. A particular advantage of the proposed method is that it does not require regularization terms or adversarial learning, which often leads to unstable optimization in practice. Furthermore, we show that two fairness criteria, demographic parity and equalized odds, can be effectively handled by whitening with the re-weighted covariance matrix. Consequently, our method controls the trade-off between the utility and fairness of algorithms by adjusting the weighting coefficient. Finally, we validate that our method outperforms existing approaches on four benchmark datasets: Corrupted CIFAR-10, Biased FFHQ, WaterBirds, and Celeb-A.
>
---
#### [new 055] T-MPEDNet: Unveiling the Synergy of Transformer-aware Multiscale Progressive Encoder-Decoder Network with Feature Recalibration for Tumor and Liver Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决肝脏及其肿瘤在CT扫描中的自动分割问题。针对肿瘤异质性和肝脏形态多样性带来的挑战，作者提出了T-MPEDNet网络，结合Transformer机制与多尺度特征，提升分割精度，最终在公开数据集上取得了优异性能。**

- **链接: [http://arxiv.org/pdf/2507.19590v1](http://arxiv.org/pdf/2507.19590v1)**

> **作者:** Chandravardhan Singh Raghaw; Jasmer Singh Sanjotra; Mohammad Zia Ur Rehman; Shubhi Bansal; Shahid Shafi Dar; Nagendra Kumar
>
> **摘要:** Precise and automated segmentation of the liver and its tumor within CT scans plays a pivotal role in swift diagnosis and the development of optimal treatment plans for individuals with liver diseases and malignancies. However, automated liver and tumor segmentation faces significant hurdles arising from the inherent heterogeneity of tumors and the diverse visual characteristics of livers across a broad spectrum of patients. Aiming to address these challenges, we present a novel Transformer-aware Multiscale Progressive Encoder-Decoder Network (T-MPEDNet) for automated segmentation of tumor and liver. T-MPEDNet leverages a deep adaptive features backbone through a progressive encoder-decoder structure, enhanced by skip connections for recalibrating channel-wise features while preserving spatial integrity. A Transformer-inspired dynamic attention mechanism captures long-range contextual relationships within the spatial domain, further enhanced by multi-scale feature utilization for refined local details, leading to accurate prediction. Morphological boundary refinement is then employed to address indistinct boundaries with neighboring organs, capturing finer details and yielding precise boundary labels. The efficacy of T-MPEDNet is comprehensively assessed on two widely utilized public benchmark datasets, LiTS and 3DIRCADb. Extensive quantitative and qualitative analyses demonstrate the superiority of T-MPEDNet compared to twelve state-of-the-art methods. On LiTS, T-MPEDNet achieves outstanding Dice Similarity Coefficients (DSC) of 97.6% and 89.1% for liver and tumor segmentation, respectively. Similar performance is observed on 3DIRCADb, with DSCs of 98.3% and 83.3% for liver and tumor segmentation, respectively. Our findings prove that T-MPEDNet is an efficacious and reliable framework for automated segmentation of the liver and its tumor in CT scans.
>
---
#### [new 056] Exemplar Med-DETR: Toward Generalized and Robust Lesion Detection in Mammogram Images and beyond
- **分类: cs.CV**

- **简介: 该论文属于医学图像检测任务，旨在解决乳腺X线、胸部X光和血管造影中病灶检测的泛化性与鲁棒性问题。作者提出Exemplar Med-DETR模型，通过多模态对比检测方法，结合类特异性示例特征，提升检测性能。实验表明其在多个数据集上表现优异，具有广泛应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.19621v1](http://arxiv.org/pdf/2507.19621v1)**

> **作者:** Sheethal Bhat; Bogdan Georgescu; Adarsh Bhandary Panambur; Mathias Zinnen; Tri-Thien Nguyen; Awais Mansoor; Karim Khalifa Elbarbary; Siming Bayer; Florin-Cristian Ghesu; Sasa Grbic; Andreas Maier
>
> **摘要:** Detecting abnormalities in medical images poses unique challenges due to differences in feature representations and the intricate relationship between anatomical structures and abnormalities. This is especially evident in mammography, where dense breast tissue can obscure lesions, complicating radiological interpretation. Despite leveraging anatomical and semantic context, existing detection methods struggle to learn effective class-specific features, limiting their applicability across different tasks and imaging modalities. In this work, we introduce Exemplar Med-DETR, a novel multi-modal contrastive detector that enables feature-based detection. It employs cross-attention with inherently derived, intuitive class-specific exemplar features and is trained with an iterative strategy. We achieve state-of-the-art performance across three distinct imaging modalities from four public datasets. On Vietnamese dense breast mammograms, we attain an mAP of 0.7 for mass detection and 0.55 for calcifications, yielding an absolute improvement of 16 percentage points. Additionally, a radiologist-supported evaluation of 100 mammograms from an out-of-distribution Chinese cohort demonstrates a twofold gain in lesion detection performance. For chest X-rays and angiography, we achieve an mAP of 0.25 for mass and 0.37 for stenosis detection, improving results by 4 and 7 percentage points, respectively. These results highlight the potential of our approach to advance robust and generalizable detection systems for medical imaging.
>
---
#### [new 057] When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios
- **分类: cs.CV**

- **简介: 该论文属于多模态长上下文建模任务，旨在解决因输入token过多导致的计算瓶颈问题。论文系统综述了图像、视频和音频中的token压缩方法，按模态和机制分类，提供结构化概述并指出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.20198v1](http://arxiv.org/pdf/2507.20198v1)**

> **作者:** Kele Shao; Keda Tao; Kejia Zhang; Sicheng Feng; Mu Cai; Yuzhang Shang; Haoxuan You; Can Qin; Yang Sui; Huan Wang
>
> **备注:** For ongoing updates and to track the latest advances in this promising area, we maintain a public repository: <a href="https://github.com/cokeshao/Awesome-Multimodal-Token-Compression" rel="external noopener nofollow" class="link-external link-https">Awesome-Multimodal-Token-Compression</a>
>
> **摘要:** Multimodal large language models (MLLMs) have made remarkable strides, largely driven by their ability to process increasingly long and complex contexts, such as high-resolution images, extended video sequences, and lengthy audio input. While this ability significantly enhances MLLM capabilities, it introduces substantial computational challenges, primarily due to the quadratic complexity of self-attention mechanisms with numerous input tokens. To mitigate these bottlenecks, token compression has emerged as an auspicious and critical approach, efficiently reducing the number of tokens during both training and inference. In this paper, we present the first systematic survey and synthesis of the burgeoning field of multimodal long context token compression. Recognizing that effective compression strategies are deeply tied to the unique characteristics and redundancies of each modality, we categorize existing approaches by their primary data focus, enabling researchers to quickly access and learn methods tailored to their specific area of interest: (1) image-centric compression, which addresses spatial redundancy in visual data; (2) video-centric compression, which tackles spatio-temporal redundancy in dynamic sequences; and (3) audio-centric compression, which handles temporal and spectral redundancy in acoustic signals. Beyond this modality-driven categorization, we further dissect methods based on their underlying mechanisms, including transformation-based, similarity-based, attention-based, and query-based approaches. By providing a comprehensive and structured overview, this survey aims to consolidate current progress, identify key challenges, and inspire future research directions in this rapidly evolving domain. We also maintain a public repository to continuously track and update the latest advances in this promising area.
>
---
#### [new 058] Self-Supervised Continuous Colormap Recovery from a 2D Scalar Field Visualization without a Legend
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于图像处理与可视化任务，旨在从无色例的2D标量场可视化中恢复连续色图。方法通过解耦与重建策略，联合预测色图与数据，并引入颜色顺序损失确保色图平滑与正确排序。实验验证了方法在合成与真实数据上的有效性，并展示了其在色图调整与迁移中的应用。**

- **链接: [http://arxiv.org/pdf/2507.20632v1](http://arxiv.org/pdf/2507.20632v1)**

> **作者:** Hongxu Liu; Xinyu Chen; Haoyang Zheng; Manyi Li; Zhenfan Liu; Fumeng Yang; Yunhai Wang; Changhe Tu; Qiong Zeng
>
> **备注:** Submitted to IEEE VIS 2025
>
> **摘要:** Recovering a continuous colormap from a single 2D scalar field visualization can be quite challenging, especially in the absence of a corresponding color legend. In this paper, we propose a novel colormap recovery approach that extracts the colormap from a color-encoded 2D scalar field visualization by simultaneously predicting the colormap and underlying data using a decoupling-and-reconstruction strategy. Our approach first separates the input visualization into colormap and data using a decoupling module, then reconstructs the visualization with a differentiable color-mapping module. To guide this process, we design a reconstruction loss between the input and reconstructed visualizations, which serves both as a constraint to ensure strong correlation between colormap and data during training, and as a self-supervised optimizer for fine-tuning the predicted colormap of unseen visualizations during inferencing. To ensure smoothness and correct color ordering in the extracted colormap, we introduce a compact colormap representation using cubic B-spline curves and an associated color order loss. We evaluate our method quantitatively and qualitatively on a synthetic dataset and a collection of real-world visualizations from the VIS30K dataset. Additionally, we demonstrate its utility in two prototype applications -- colormap adjustment and colormap transfer -- and explore its generalization to visualizations with color legends and ones encoded using discrete color palettes.
>
---
#### [new 059] All-in-One Medical Image Restoration with Latent Diffusion-Enhanced Vector-Quantized Codebook Prior
- **分类: cs.CV**

- **简介: 该论文属于医学图像恢复任务，旨在解决多种医学图像恢复任务中低质量图像信息丢失、任务异构性带来的挑战。论文提出了DiffCode框架，结合任务自适应码本库与潜在扩散策略，统一恢复高质量MRI、CT和PET图像，提升了恢复效果与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.19874v1](http://arxiv.org/pdf/2507.19874v1)**

> **作者:** Haowei Chen; Zhiwen Yang; Haotian Hou; Hui Zhang; Bingzheng Wei; Gang Zhou; Yan Xu
>
> **备注:** 11pages, 3figures, MICCAI 2025
>
> **摘要:** All-in-one medical image restoration (MedIR) aims to address multiple MedIR tasks using a unified model, concurrently recovering various high-quality (HQ) medical images (e.g., MRI, CT, and PET) from low-quality (LQ) counterparts. However, all-in-one MedIR presents significant challenges due to the heterogeneity across different tasks. Each task involves distinct degradations, leading to diverse information losses in LQ images. Existing methods struggle to handle these diverse information losses associated with different tasks. To address these challenges, we propose a latent diffusion-enhanced vector-quantized codebook prior and develop \textbf{DiffCode}, a novel framework leveraging this prior for all-in-one MedIR. Specifically, to compensate for diverse information losses associated with different tasks, DiffCode constructs a task-adaptive codebook bank to integrate task-specific HQ prior features across tasks, capturing a comprehensive prior. Furthermore, to enhance prior retrieval from the codebook bank, DiffCode introduces a latent diffusion strategy that utilizes the diffusion model's powerful mapping capabilities to iteratively refine the latent feature distribution, estimating more accurate HQ prior features during restoration. With the help of the task-adaptive codebook bank and latent diffusion strategy, DiffCode achieves superior performance in both quantitative metrics and visual quality across three MedIR tasks: MRI super-resolution, CT denoising, and PET synthesis.
>
---
#### [new 060] Interpretable Open-Vocabulary Referring Object Detection with Reverse Contrast Attention
- **分类: cs.CV; cs.AI**

- **简介: 论文提出Reverse Contrast Attention（RCA），用于提升视觉-语言模型中的物体定位性能，应用于开放词汇指代表达检测（OV-RefOD）任务。该方法无需重新训练即可增强模型表现，并引入了新的评估指标FitAP。实验表明RCA在多个模型上有效提升性能，同时提升模型解释性。**

- **链接: [http://arxiv.org/pdf/2507.19891v1](http://arxiv.org/pdf/2507.19891v1)**

> **作者:** Drandreb Earl O. Juanico; Rowel O. Atienza; Jeffrey Kenneth Go
>
> **备注:** 10 pages with supplementary material, 6 main figures, 2 main tables; github: earl-juanico/rca
>
> **摘要:** We propose Reverse Contrast Attention (RCA), a plug-in method that enhances object localization in vision-language transformers without retraining. RCA reweights final-layer attention by suppressing extremes and amplifying mid-level activations to let semantically relevant but subdued tokens guide predictions. We evaluate it on Open Vocabulary Referring Object Detection (OV-RefOD), introducing FitAP, a confidence-free average precision metric based on IoU and box area. RCA improves FitAP in 11 out of 15 open-source VLMs, with gains up to $+26.6\%$. Effectiveness aligns with attention sharpness and fusion timing; while late-fusion models benefit consistently, models like $\texttt{DeepSeek-VL2}$ also improve, pointing to capacity and disentanglement as key factors. RCA offers both interpretability and performance gains for multimodal transformers.
>
---
#### [new 061] Leveraging Sparse LiDAR for RAFT-Stereo: A Depth Pre-Fill Perspective
- **分类: cs.CV**

- **简介: 本文研究稀疏LiDAR在RAFT-Stereo中的引导作用，旨在通过预填充提升深度估计精度。论文属于立体匹配任务，解决稀疏LiDAR引导效果差的问题，提出结合两种预填充策略的GRAFT-Stereo方法，在多数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.19738v1](http://arxiv.org/pdf/2507.19738v1)**

> **作者:** Jinsu Yoo; Sooyoung Jeon; Zanming Huang; Tai-Yu Pan; Wei-Lun Chao
>
> **摘要:** We investigate LiDAR guidance within the RAFT-Stereo framework, aiming to improve stereo matching accuracy by injecting precise LiDAR depth into the initial disparity map. We find that the effectiveness of LiDAR guidance drastically degrades when the LiDAR points become sparse (e.g., a few hundred points per frame), and we offer a novel explanation from a signal processing perspective. This insight leads to a surprisingly simple solution that enables LiDAR-guided RAFT-Stereo to thrive: pre-filling the sparse initial disparity map with interpolation. Interestingly, we find that pre-filling is also effective when injecting LiDAR depth into image features via early fusion, but for a fundamentally different reason, necessitating a distinct pre-filling approach. By combining both solutions, the proposed Guided RAFT-Stereo (GRAFT-Stereo) significantly outperforms existing LiDAR-guided methods under sparse LiDAR conditions across various datasets. We hope this study inspires more effective LiDAR-guided stereo methods.
>
---
#### [new 062] Co-Win: Joint Object Detection and Instance Segmentation in LiDAR Point Clouds via Collaborative Window Processing
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的环境感知任务，旨在解决复杂城市环境中激光雷达点云的多目标检测与实例分割问题。论文提出了Co-Win框架，结合鸟瞰图感知、窗口特征提取与变分掩码分割，实现对场景的精细理解和实例识别，提升自动驾驶系统的决策能力。**

- **链接: [http://arxiv.org/pdf/2507.19691v1](http://arxiv.org/pdf/2507.19691v1)**

> **作者:** Haichuan Li; Tomi Westerlund
>
> **摘要:** Accurate perception and scene understanding in complex urban environments is a critical challenge for ensuring safe and efficient autonomous navigation. In this paper, we present Co-Win, a novel bird's eye view (BEV) perception framework that integrates point cloud encoding with efficient parallel window-based feature extraction to address the multi-modality inherent in environmental understanding. Our method employs a hierarchical architecture comprising a specialized encoder, a window-based backbone, and a query-based decoder head to effectively capture diverse spatial features and object relationships. Unlike prior approaches that treat perception as a simple regression task, our framework incorporates a variational approach with mask-based instance segmentation, enabling fine-grained scene decomposition and understanding. The Co-Win architecture processes point cloud data through progressive feature extraction stages, ensuring that predicted masks are both data-consistent and contextually relevant. Furthermore, our method produces interpretable and diverse instance predictions, enabling enhanced downstream decision-making and planning in autonomous driving systems.
>
---
#### [new 063] FineMotion: A Dataset and Benchmark with both Spatial and Temporal Annotation for Fine-grained Motion Generation and Editing
- **分类: cs.CV**

- **简介: 该论文属于文本驱动的人体动作生成与编辑任务，旨在解决现有方法忽略身体部位动作细节的问题。作者构建了FineMotion数据集，包含44.2万段人体动作片段及其详细描述，并提出支持空间与时间维度细粒度编辑的方法，显著提升了生成效果。**

- **链接: [http://arxiv.org/pdf/2507.19850v1](http://arxiv.org/pdf/2507.19850v1)**

> **作者:** Bizhu Wu; Jinheng Xie; Meidan Ding; Zhe Kong; Jianfeng Ren; Ruibin Bai; Rong Qu; Linlin Shen
>
> **摘要:** Generating realistic human motions from textual descriptions has undergone significant advancements. However, existing methods often overlook specific body part movements and their timing. In this paper, we address this issue by enriching the textual description with more details. Specifically, we propose the FineMotion dataset, which contains over 442,000 human motion snippets - short segments of human motion sequences - and their corresponding detailed descriptions of human body part movements. Additionally, the dataset includes about 95k detailed paragraphs describing the movements of human body parts of entire motion sequences. Experimental results demonstrate the significance of our dataset on the text-driven finegrained human motion generation task, especially with a remarkable +15.3% improvement in Top-3 accuracy for the MDM model. Notably, we further support a zero-shot pipeline of fine-grained motion editing, which focuses on detailed editing in both spatial and temporal dimensions via text. Dataset and code available at: CVI-SZU/FineMotion
>
---
#### [new 064] RaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cues for 3D Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D目标检测任务，旨在解决4D雷达与单目图像融合的3D目标检测问题。现有方法受限于结构或缺乏整体理解，本文提出RaGS，首次使用3D高斯点阵表示，通过级联流程融合多模态信息，实现高效精准检测。**

- **链接: [http://arxiv.org/pdf/2507.19856v1](http://arxiv.org/pdf/2507.19856v1)**

> **作者:** Xiaokai Bai; Chenxu Zhou; Lianqing Zheng; Si-Yuan Cao; Jianan Liu; Xiaohan Zhang; Zhengzhuang Zhang; Hui-liang Shen
>
> **备注:** 9 pages, 6 figures, conference
>
> **摘要:** 4D millimeter-wave radar has emerged as a promising sensor for autonomous driving, but effective 3D object detection from both 4D radar and monocular images remains a challenge. Existing fusion approaches typically rely on either instance-based proposals or dense BEV grids, which either lack holistic scene understanding or are limited by rigid grid structures. To address these, we propose RaGS, the first framework to leverage 3D Gaussian Splatting (GS) as representation for fusing 4D radar and monocular cues in 3D object detection. 3D GS naturally suits 3D object detection by modeling the scene as a field of Gaussians, dynamically allocating resources on foreground objects and providing a flexible, resource-efficient solution. RaGS uses a cascaded pipeline to construct and refine the Gaussian field. It starts with the Frustum-based Localization Initiation (FLI), which unprojects foreground pixels to initialize coarse 3D Gaussians positions. Then, the Iterative Multimodal Aggregation (IMA) fuses semantics and geometry, refining the limited Gaussians to the regions of interest. Finally, the Multi-level Gaussian Fusion (MGF) renders the Gaussians into multi-level BEV features for 3D object detection. By dynamically focusing on sparse objects within scenes, RaGS enable object concentrating while offering comprehensive scene perception. Extensive experiments on View-of-Delft, TJ4DRadSet, and OmniHD-Scenes benchmarks demonstrate its state-of-the-art performance. Code will be released.
>
---
#### [new 065] MagicAnime: A Hierarchically Annotated, Multimodal and Multitasking Dataset with Benchmarks for Cartoon Animation Generation
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于动画生成任务，旨在解决高质量卡通动画生成中多模态控制难、数据稀缺的问题。论文构建了大规模多模态数据集MagicAnime及基准测试MagicAnime-Bench，支持多种生成任务，验证了其在高保真、细粒度和可控生成方面的有效性。**

- **链接: [http://arxiv.org/pdf/2507.20368v1](http://arxiv.org/pdf/2507.20368v1)**

> **作者:** Shuolin Xu; Bingyuan Wang; Zeyu Cai; Fangteng Fu; Yue Ma; Tongyi Lee; Hongchuan Yu; Zeyu Wang
>
> **备注:** 8 pages,6 figures
>
> **摘要:** Generating high-quality cartoon animations multimodal control is challenging due to the complexity of non-human characters, stylistically diverse motions and fine-grained emotions. There is a huge domain gap between real-world videos and cartoon animation, as cartoon animation is usually abstract and has exaggerated motion. Meanwhile, public multimodal cartoon data are extremely scarce due to the difficulty of large-scale automatic annotation processes compared with real-life scenarios. To bridge this gap, We propose the MagicAnime dataset, a large-scale, hierarchically annotated, and multimodal dataset designed to support multiple video generation tasks, along with the benchmarks it includes. Containing 400k video clips for image-to-video generation, 50k pairs of video clips and keypoints for whole-body annotation, 12k pairs of video clips for video-to-video face animation, and 2.9k pairs of video and audio clips for audio-driven face animation. Meanwhile, we also build a set of multi-modal cartoon animation benchmarks, called MagicAnime-Bench, to support the comparisons of different methods in the tasks above. Comprehensive experiments on four tasks, including video-driven face animation, audio-driven face animation, image-to-video animation, and pose-driven character animation, validate its effectiveness in supporting high-fidelity, fine-grained, and controllable generation.
>
---
#### [new 066] JOLT3D: Joint Learning of Talking Heads and 3DMM Parameters with Application to Lip-Sync
- **分类: cs.CV**

- **简介: 该论文属于 talking head 合成任务，旨在解决生成高质量说话人脸并实现自然唇形同步的问题。作者提出 JOLT3D，联合学习 3D 面部重建模型和说话人头生成模型，优化基于 FACS 的 blendshape 表情表示，并设计新唇同步流程，分离下巴轮廓，减少嘴部闪烁，提升同步质量。**

- **链接: [http://arxiv.org/pdf/2507.20452v1](http://arxiv.org/pdf/2507.20452v1)**

> **作者:** Sungjoon Park; Minsik Park; Haneol Lee; Jaesub Yun; Donggeon Lee
>
> **备注:** 10 + 8 pages, 11 figures
>
> **摘要:** In this work, we revisit the effectiveness of 3DMM for talking head synthesis by jointly learning a 3D face reconstruction model and a talking head synthesis model. This enables us to obtain a FACS-based blendshape representation of facial expressions that is optimized for talking head synthesis. This contrasts with previous methods that either fit 3DMM parameters to 2D landmarks or rely on pretrained face reconstruction models. Not only does our approach increase the quality of the generated face, but it also allows us to take advantage of the blendshape representation to modify just the mouth region for the purpose of audio-based lip-sync. To this end, we propose a novel lip-sync pipeline that, unlike previous methods, decouples the original chin contour from the lip-synced chin contour, and reduces flickering near the mouth.
>
---
#### [new 067] MambaMap: Online Vectorized HD Map Construction using State Space Model
- **分类: cs.CV**

- **简介: 论文提出MambaMap，用于自动驾驶中在线构建矢量高精地图。该方法通过状态空间模型融合长时间序列特征，解决遮挡和感知范围扩展带来的挑战。主要工作包括：引入记忆库、门控机制及多向时空扫描策略，提升地图精度与时间一致性。**

- **链接: [http://arxiv.org/pdf/2507.20224v1](http://arxiv.org/pdf/2507.20224v1)**

> **作者:** Ruizi Yang; Xiaolu Liu; Junbo Chen; Jianke Zhu
>
> **摘要:** High-definition (HD) maps are essential for autonomous driving, as they provide precise road information for downstream tasks. Recent advances highlight the potential of temporal modeling in addressing challenges like occlusions and extended perception range. However, existing methods either fail to fully exploit temporal information or incur substantial computational overhead in handling extended sequences. To tackle these challenges, we propose MambaMap, a novel framework that efficiently fuses long-range temporal features in the state space to construct online vectorized HD maps. Specifically, MambaMap incorporates a memory bank to store and utilize information from historical frames, dynamically updating BEV features and instance queries to improve robustness against noise and occlusions. Moreover, we introduce a gating mechanism in the state space, selectively integrating dependencies of map elements in high computational efficiency. In addition, we design innovative multi-directional and spatial-temporal scanning strategies to enhance feature extraction at both BEV and instance levels. These strategies significantly boost the prediction accuracy of our approach while ensuring robust temporal consistency. Extensive experiments on the nuScenes and Argoverse2 datasets demonstrate that our proposed MambaMap approach outperforms state-of-the-art methods across various splits and perception ranges. Source code will be available at https://github.com/ZiziAmy/MambaMap.
>
---
#### [new 068] L-MCAT: Unpaired Multimodal Transformer with Contrastive Attention for Label-Efficient Satellite Image Classification
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分类任务，旨在解决标签效率低和多模态数据对齐问题。作者提出L-MCAT模型，通过模态光谱适配器和对比注意力机制，在无需像素级对齐或标签的情况下实现高效分类，表现出优越性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.20259v1](http://arxiv.org/pdf/2507.20259v1)**

> **作者:** Mitul Goswami; Mrinal Goswami
>
> **摘要:** We propose the Lightweight Multimodal Contrastive Attention Transformer (L-MCAT), a novel transformer-based framework for label-efficient remote sensing image classification using unpaired multimodal satellite data. L-MCAT introduces two core innovations: (1) Modality-Spectral Adapters (MSA) that compress high-dimensional sensor inputs into a unified embedding space, and (2) Unpaired Multimodal Attention Alignment (U-MAA), a contrastive self-supervised mechanism integrated into the attention layers to align heterogeneous modalities without pixel-level correspondence or labels. L-MCAT achieves 95.4% overall accuracy on the SEN12MS dataset using only 20 labels per class, outperforming state-of-the-art baselines while using 47x fewer parameters and 23x fewer FLOPs than MCTrans. It maintains over 92% accuracy even under 50% spatial misalignment, demonstrating robustness for real-world deployment. The model trains end-to-end in under 5 hours on a single consumer GPU.
>
---
#### [new 069] Frequency-Aware Autoregressive Modeling for Efficient High-Resolution Image Synthesis
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决高分辨率图像合成中的计算效率问题。现有方法在高分辨率阶段计算量大，效率低。论文提出SparseVAR框架，通过动态排除低频冗余标记并保留关键信息，显著加速推理过程，同时保持生成质量，适用于HART和Infinity模型。**

- **链接: [http://arxiv.org/pdf/2507.20454v1](http://arxiv.org/pdf/2507.20454v1)**

> **作者:** Zhuokun Chen; Jugang Fan; Zhuowei Yu; Bohan Zhuang; Mingkui Tan
>
> **摘要:** Visual autoregressive modeling, based on the next-scale prediction paradigm, exhibits notable advantages in image quality and model scalability over traditional autoregressive and diffusion models. It generates images by progressively refining resolution across multiple stages. However, the computational overhead in high-resolution stages remains a critical challenge due to the substantial number of tokens involved. In this paper, we introduce SparseVAR, a plug-and-play acceleration framework for next-scale prediction that dynamically excludes low-frequency tokens during inference without requiring additional training. Our approach is motivated by the observation that tokens in low-frequency regions have a negligible impact on image quality in high-resolution stages and exhibit strong similarity with neighboring tokens. Additionally, we observe that different blocks in the next-scale prediction model focus on distinct regions, with some concentrating on high-frequency areas. SparseVAR leverages these insights by employing lightweight MSE-based metrics to identify low-frequency tokens while preserving the fidelity of excluded regions through a small set of uniformly sampled anchor tokens. By significantly reducing the computational cost while maintaining high image generation quality, SparseVAR achieves notable acceleration in both HART and Infinity. Specifically, SparseVAR achieves up to a 2 times speedup with minimal quality degradation in Infinity-2B.
>
---
#### [new 070] FantasyID: A dataset for detecting digital manipulations of ID-documents
- **分类: cs.CV**

- **简介: 该论文属于图像伪造检测任务，旨在解决身份文档（ID）伪造问题。作者构建了一个名为FantasyID的公开数据集，包含真实人脸和多样式设计的模拟ID卡，用于评估检测算法在现实场景下的性能。实验表明，现有先进算法在该数据集上的检测效果不佳，显示出数据集的挑战性和实用性。**

- **链接: [http://arxiv.org/pdf/2507.20808v1](http://arxiv.org/pdf/2507.20808v1)**

> **作者:** Pavel Korshunov; Amir Mohammadi; Vidit Vidit; Christophe Ecabert; Sébastien Marcel
>
> **备注:** Accepted to IJCB 2025; for project page, see https://www.idiap.ch/paper/fantasyid
>
> **摘要:** Advancements in image generation led to the availability of easy-to-use tools for malicious actors to create forged images. These tools pose a serious threat to the widespread Know Your Customer (KYC) applications, requiring robust systems for detection of the forged Identity Documents (IDs). To facilitate the development of the detection algorithms, in this paper, we propose a novel publicly available (including commercial use) dataset, FantasyID, which mimics real-world IDs but without tampering with legal documents and, compared to previous public datasets, it does not contain generated faces or specimen watermarks. FantasyID contains ID cards with diverse design styles, languages, and faces of real people. To simulate a realistic KYC scenario, the cards from FantasyID were printed and captured with three different devices, constituting the bonafide class. We have emulated digital forgery/injection attacks that could be performed by a malicious actor to tamper the IDs using the existing generative tools. The current state-of-the-art forgery detection algorithms, such as TruFor, MMFusion, UniFD, and FatFormer, are challenged by FantasyID dataset. It especially evident, in the evaluation conditions close to practical, with the operational threshold set on validation set so that false positive rate is at 10%, leading to false negative rates close to 50% across the board on the test set. The evaluation experiments demonstrate that FantasyID dataset is complex enough to be used as an evaluation benchmark for detection algorithms.
>
---
#### [new 071] SCALAR: Scale-wise Controllable Visual Autoregressive Learning
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决现有视觉自回归（VAR）模型在可控生成中控制编码效率低、注入机制干扰生成质量的问题。论文提出SCALAR方法，通过引入逐尺度条件解码机制，实现对生成过程的精细控制，同时保持生成效果与效率。**

- **链接: [http://arxiv.org/pdf/2507.19946v1](http://arxiv.org/pdf/2507.19946v1)**

> **作者:** Ryan Xu; Dongyang Jin; Yancheng Bai; Rui Lan; Xu Duan; Lei Sun; Xiangxiang Chu
>
> **摘要:** Controllable image synthesis, which enables fine-grained control over generated outputs, has emerged as a key focus in visual generative modeling. However, controllable generation remains challenging for Visual Autoregressive (VAR) models due to their hierarchical, next-scale prediction style. Existing VAR-based methods often suffer from inefficient control encoding and disruptive injection mechanisms that compromise both fidelity and efficiency. In this work, we present SCALAR, a controllable generation method based on VAR, incorporating a novel Scale-wise Conditional Decoding mechanism. SCALAR leverages a
>
---
#### [new 072] Region-based Cluster Discrimination for Visual Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉表示学习任务，旨在解决现有模型依赖全局特征、难以胜任密集预测任务的问题。作者提出RICE方法，通过构建大规模候选区域数据集、设计Region Transformer层和统一的区域聚类判别损失，提升区域级视觉理解和OCR能力，实验证明其在分割、检测及多模态大模型中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.20025v1](http://arxiv.org/pdf/2507.20025v1)**

> **作者:** Yin Xie; Kaicheng Yang; Xiang An; Kun Wu; Yongle Zhao; Weimo Deng; Zimin Ran; Yumeng Wang; Ziyong Feng; Roy Miles; Ismail Elezi; Jiankang Deng
>
> **备注:** Accepted as a highlight paper at ICCV 2025
>
> **摘要:** Learning visual representations is foundational for a broad spectrum of downstream tasks. Although recent vision-language contrastive models, such as CLIP and SigLIP, have achieved impressive zero-shot performance via large-scale vision-language alignment, their reliance on global representations constrains their effectiveness for dense prediction tasks, such as grounding, OCR, and segmentation. To address this gap, we introduce Region-Aware Cluster Discrimination (RICE), a novel method that enhances region-level visual and OCR capabilities. We first construct a billion-scale candidate region dataset and propose a Region Transformer layer to extract rich regional semantics. We further design a unified region cluster discrimination loss that jointly supports object and OCR learning within a single classification framework, enabling efficient and scalable distributed training on large-scale data. Extensive experiments show that RICE consistently outperforms previous methods on tasks, including segmentation, dense detection, and visual perception for Multimodal Large Language Models (MLLMs). The pre-trained models have been released at https://github.com/deepglint/MVT.
>
---
#### [new 073] An Efficient Machine Learning Framework for Forest Height Estimation from Multi-Polarimetric Multi-Baseline SAR data
- **分类: cs.CV**

- **简介: 该论文属于遥感与机器学习交叉任务，旨在解决森林高度估计问题。利用多极化多基线SAR数据和LiDAR真实数据，提出FGump框架，通过梯度提升实现高效准确的森林高度估算，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.20798v1](http://arxiv.org/pdf/2507.20798v1)**

> **作者:** Francesca Razzano; Wenyu Yang; Sergio Vitale; Giampaolo Ferraioli; Silvia Liberata Ullo; Gilda Schirinzi
>
> **备注:** 13 pages, 12 figures, This paper has been submitted to IEEE TGRS. At the moment is under review
>
> **摘要:** Accurate forest height estimation is crucial for climate change monitoring and carbon cycle assessment. Synthetic Aperture Radar (SAR), particularly in multi-channel configurations, has provided support for a long time in 3D forest structure reconstruction through model-based techniques. More recently, data-driven approaches using Machine Learning (ML) and Deep Learning (DL) have enabled new opportunities for forest parameter retrieval. This paper introduces FGump, a forest height estimation framework by gradient boosting using multi-channel SAR processing with LiDAR profiles as Ground Truth(GT). Unlike typical ML and DL approaches that require large datasets and complex architectures, FGump ensures a strong balance between accuracy and computational efficiency, using a limited set of hand-designed features and avoiding heavy preprocessing (e.g., calibration and/or quantization). Evaluated under both classification and regression paradigms, the proposed framework demonstrates that the regression formulation enables fine-grained, continuous estimations and avoids quantization artifacts by resulting in more precise measurements without rounding. Experimental results confirm that FGump outperforms State-of-the-Art (SOTA) AI-based and classical methods, achieving higher accuracy and significantly lower training and inference times, as demonstrated in our results.
>
---
#### [new 074] SeeDiff: Off-the-Shelf Seeded Mask Generation from Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于语义分割任务，旨在解决像素级标注耗时问题。利用Stable Diffusion模型，通过分析注意力机制，提出SeeDiff方法，无需额外训练或调优，即可生成高质量标注掩码。核心在于结合交叉注意力与自注意力机制，实现从种子点扩展到完整对象的分割效果。**

- **链接: [http://arxiv.org/pdf/2507.19808v1](http://arxiv.org/pdf/2507.19808v1)**

> **作者:** Joon Hyun Park; Kumju Jo; Sungyong Baik
>
> **备注:** AAAI 2025
>
> **摘要:** Entrusted with the goal of pixel-level object classification, the semantic segmentation networks entail the laborious preparation of pixel-level annotation masks. To obtain pixel-level annotation masks for a given class without human efforts, recent few works have proposed to generate pairs of images and annotation masks by employing image and text relationships modeled by text-to-image generative models, especially Stable Diffusion. However, these works do not fully exploit the capability of text-guided Diffusion models and thus require a pre-trained segmentation network, careful text prompt tuning, or the training of a segmentation network to generate final annotation masks. In this work, we take a closer look at attention mechanisms of Stable Diffusion, from which we draw connections with classical seeded segmentation approaches. In particular, we show that cross-attention alone provides very coarse object localization, which however can provide initial seeds. Then, akin to region expansion in seeded segmentation, we utilize the semantic-correspondence-modeling capability of self-attention to iteratively spread the attention to the whole class from the seeds using multi-scale self-attention maps. We also observe that a simple-text-guided synthetic image often has a uniform background, which is easier to find correspondences, compared to complex-structured objects. Thus, we further refine a mask using a more accurate background mask. Our proposed method, dubbed SeeDiff, generates high-quality masks off-the-shelf from Stable Diffusion, without additional training procedure, prompt tuning, or a pre-trained segmentation network.
>
---
#### [new 075] From Gallery to Wrist: Realistic 3D Bracelet Insertion in Videos
- **分类: cs.CV**

- **简介: 该论文属于视频编辑任务，旨在解决3D物体插入视频时的时空一致性与真实感光照难题。工作提出一种结合3D高斯泼溅与2D扩散模型的混合管线，实现动态手腕场景中手镯插入的真实感与时间连贯性。**

- **链接: [http://arxiv.org/pdf/2507.20331v1](http://arxiv.org/pdf/2507.20331v1)**

> **作者:** Chenjian Gao; Lihe Ding; Rui Han; Zhanpeng Huang; Zibin Wang; Tianfan Xue
>
> **备注:** 12 pages
>
> **摘要:** Inserting 3D objects into videos is a longstanding challenge in computer graphics with applications in augmented reality, virtual try-on, and video composition. Achieving both temporal consistency, or realistic lighting remains difficult, particularly in dynamic scenarios with complex object motion, perspective changes, and varying illumination. While 2D diffusion models have shown promise for producing photorealistic edits, they often struggle with maintaining temporal coherence across frames. Conversely, traditional 3D rendering methods excel in spatial and temporal consistency but fall short in achieving photorealistic lighting. In this work, we propose a hybrid object insertion pipeline that combines the strengths of both paradigms. Specifically, we focus on inserting bracelets into dynamic wrist scenes, leveraging the high temporal consistency of 3D Gaussian Splatting (3DGS) for initial rendering and refining the results using a 2D diffusion-based enhancement model to ensure realistic lighting interactions. Our method introduces a shading-driven pipeline that separates intrinsic object properties (albedo, shading, reflectance) and refines both shading and sRGB images for photorealism. To maintain temporal coherence, we optimize the 3DGS model with multi-frame weighted adjustments. This is the first approach to synergize 3D rendering and 2D diffusion for video object insertion, offering a robust solution for realistic and consistent video editing. Project Page: https://cjeen.github.io/BraceletPaper/
>
---
#### [new 076] RARE: Refine Any Registration of Pairwise Point Clouds via Zero-Shot Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于点云配准任务，旨在提升点云配准的精度。它通过结合深度图像与预训练扩散模型提取的特征，改进点云间的对应关系，从而优化配准结果。方法无需额外训练数据，具有良好的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.19950v1](http://arxiv.org/pdf/2507.19950v1)**

> **作者:** Chengyu Zheng; Jin Huang; Honghua Chen; Mingqiang Wei
>
> **摘要:** Recent research leveraging large-scale pretrained diffusion models has demonstrated the potential of using diffusion features to establish semantic correspondences in images. Inspired by advancements in diffusion-based techniques, we propose a novel zero-shot method for refining point cloud registration algorithms. Our approach leverages correspondences derived from depth images to enhance point feature representations, eliminating the need for a dedicated training dataset. Specifically, we first project the point cloud into depth maps from multiple perspectives and extract implicit knowledge from a pretrained diffusion network as depth diffusion features. These features are then integrated with geometric features obtained from existing methods to establish more accurate correspondences between point clouds. By leveraging these refined correspondences, our approach achieves significantly improved registration accuracy. Extensive experiments demonstrate that our method not only enhances the performance of existing point cloud registration techniques but also exhibits robust generalization capabilities across diverse datasets. Codes are available at https://github.com/zhengcy-lambo/RARE.git.
>
---
#### [new 077] SAMwave: Wavelet-Driven Feature Enrichment for Effective Adaptation of Segment Anything Model
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像分割任务，旨在解决SAM模型在复杂任务中性能下降的问题。论文提出SAMwave方法，利用小波变换提取多尺度高频特征，并引入复数适配器融合这些特征，提升SAM的密集预测效果，验证了方法的有效性和通用性。**

- **链接: [http://arxiv.org/pdf/2507.20186v1](http://arxiv.org/pdf/2507.20186v1)**

> **作者:** Saurabh Yadav; Avi Gupta; Koteswar Rao Jerripothula
>
> **备注:** Accepted to BMVC 2025. The first two authors contributed equally
>
> **摘要:** The emergence of large foundation models has propelled significant advances in various domains. The Segment Anything Model (SAM), a leading model for image segmentation, exemplifies these advances, outperforming traditional methods. However, such foundation models often suffer from performance degradation when applied to complex tasks for which they are not trained. Existing methods typically employ adapter-based fine-tuning strategies to adapt SAM for tasks and leverage high-frequency features extracted from the Fourier domain. However, Our analysis reveals that these approaches offer limited benefits due to constraints in their feature extraction techniques. To overcome this, we propose \textbf{\textit{SAMwave}}, a novel and interpretable approach that utilizes the wavelet transform to extract richer, multi-scale high-frequency features from input data. Extending this, we introduce complex-valued adapters capable of capturing complex-valued spatial-frequency information via complex wavelet transforms. By adaptively integrating these wavelet coefficients, SAMwave enables SAM's encoder to capture information more relevant for dense prediction. Empirical evaluations on four challenging low-level vision tasks demonstrate that SAMwave significantly outperforms existing adaptation methods. This superior performance is consistent across both the SAM and SAM2 backbones and holds for both real and complex-valued adapter variants, highlighting the efficiency, flexibility, and interpretability of our proposed method for adapting segment anything models.
>
---
#### [new 078] Generative Pre-training for Subjective Tasks: A Diffusion Transformer-Based Framework for Facial Beauty Prediction
- **分类: cs.CV**

- **简介: 该论文属于面部美感预测任务，旨在解决现有方法在主观美感特征学习上的不足。论文提出Diff-FBP框架，利用扩散Transformer在无标签面部数据上进行生成预训练，提取更贴合美感评估的特征表示，再通过轻量回归头微调，实现了更优性能。**

- **链接: [http://arxiv.org/pdf/2507.20363v1](http://arxiv.org/pdf/2507.20363v1)**

> **作者:** Djamel Eddine Boukhari; Ali chemsa
>
> **摘要:** Facial Beauty Prediction (FBP) is a challenging computer vision task due to its subjective nature and the subtle, holistic features that influence human perception. Prevailing methods, often based on deep convolutional networks or standard Vision Transformers pre-trained on generic object classification (e.g., ImageNet), struggle to learn feature representations that are truly aligned with high-level aesthetic assessment. In this paper, we propose a novel two-stage framework that leverages the power of generative models to create a superior, domain-specific feature extractor. In the first stage, we pre-train a Diffusion Transformer on a large-scale, unlabeled facial dataset (FFHQ) through a self-supervised denoising task. This process forces the model to learn the fundamental data distribution of human faces, capturing nuanced details and structural priors essential for aesthetic evaluation. In the second stage, the pre-trained and frozen encoder of our Diffusion Transformer is used as a backbone feature extractor, with only a lightweight regression head being fine-tuned on the target FBP dataset (FBP5500). Our method, termed Diff-FBP, sets a new state-of-the-art on the FBP5500 benchmark, achieving a Pearson Correlation Coefficient (PCC) of 0.932, significantly outperforming prior art based on general-purpose pre-training. Extensive ablation studies validate that our generative pre-training strategy is the key contributor to this performance leap, creating feature representations that are more semantically potent for subjective visual tasks.
>
---
#### [new 079] SAViL-Det: Semantic-Aware Vision-Language Model for Multi-Script Text Detection
- **分类: cs.CV**

- **简介: 该论文属于多语种文本检测任务，旨在解决自然场景中多样文字和任意形状文本检测困难的问题。作者提出了SAViL-Det模型，结合视觉与文本语义信息，利用CLIP模型、AFPN特征融合和跨模态注意力机制，实现更精准的文本检测，取得了优异性能。**

- **链接: [http://arxiv.org/pdf/2507.20188v1](http://arxiv.org/pdf/2507.20188v1)**

> **作者:** Mohammed-En-Nadhir Zighem; Abdenour Hadid
>
> **摘要:** Detecting text in natural scenes remains challenging, particularly for diverse scripts and arbitrarily shaped instances where visual cues alone are often insufficient. Existing methods do not fully leverage semantic context. This paper introduces SAViL-Det, a novel semantic-aware vision-language model that enhances multi-script text detection by effectively integrating textual prompts with visual features. SAViL-Det utilizes a pre-trained CLIP model combined with an Asymptotic Feature Pyramid Network (AFPN) for multi-scale visual feature fusion. The core of the proposed framework is a novel language-vision decoder that adaptively propagates fine-grained semantic information from text prompts to visual features via cross-modal attention. Furthermore, a text-to-pixel contrastive learning mechanism explicitly aligns textual and corresponding visual pixel features. Extensive experiments on challenging benchmarks demonstrate the effectiveness of the proposed approach, achieving state-of-the-art performance with F-scores of 84.8% on the benchmark multi-lingual MLT-2019 dataset and 90.2% on the curved-text CTW1500 dataset.
>
---
#### [new 080] Reconstructing 4D Spatial Intelligence: A Survey
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在重建4D空间智能。它系统归纳了现有方法为五个层次，涵盖从低级3D属性到物理规律建模的进展，并分析各层次挑战与未来方向。**

- **链接: [http://arxiv.org/pdf/2507.21045v1](http://arxiv.org/pdf/2507.21045v1)**

> **作者:** Yukang Cao; Jiahao Lu; Zhisheng Huang; Zhuowei Shen; Chengfeng Zhao; Fangzhou Hong; Zhaoxi Chen; Xin Li; Wenping Wang; Yuan Liu; Ziwei Liu
>
> **备注:** Project page: https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence
>
> **摘要:** Reconstructing 4D spatial intelligence from visual observations has long been a central yet challenging task in computer vision, with broad real-world applications. These range from entertainment domains like movies, where the focus is often on reconstructing fundamental visual elements, to embodied AI, which emphasizes interaction modeling and physical realism. Fueled by rapid advances in 3D representations and deep learning architectures, the field has evolved quickly, outpacing the scope of previous surveys. Additionally, existing surveys rarely offer a comprehensive analysis of the hierarchical structure of 4D scene reconstruction. To address this gap, we present a new perspective that organizes existing methods into five progressive levels of 4D spatial intelligence: (1) Level 1 -- reconstruction of low-level 3D attributes (e.g., depth, pose, and point maps); (2) Level 2 -- reconstruction of 3D scene components (e.g., objects, humans, structures); (3) Level 3 -- reconstruction of 4D dynamic scenes; (4) Level 4 -- modeling of interactions among scene components; and (5) Level 5 -- incorporation of physical laws and constraints. We conclude the survey by discussing the key challenges at each level and highlighting promising directions for advancing toward even richer levels of 4D spatial intelligence. To track ongoing developments, we maintain an up-to-date project page: https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence.
>
---
#### [new 081] Implicit Counterfactual Learning for Audio-Visual Segmentation
- **分类: cs.CV**

- **简介: 该论文属于音频-视觉分割任务，旨在解决跨模态表征差异和信息不平衡导致的分割偏差问题。作者提出隐式反事实框架（ICF），通过多粒度隐式文本构建模态共享空间，并引入语义反事实和对比学习策略，实现更准确、无偏的音频-视觉分割。**

- **链接: [http://arxiv.org/pdf/2507.20740v1](http://arxiv.org/pdf/2507.20740v1)**

> **作者:** Mingfeng Zha; Tianyu Li; Guoqing Wang; Peng Wang; Yangyang Wu; Yang Yang; Heng Tao Shen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Audio-visual segmentation (AVS) aims to segment objects in videos based on audio cues. Existing AVS methods are primarily designed to enhance interaction efficiency but pay limited attention to modality representation discrepancies and imbalances. To overcome this, we propose the implicit counterfactual framework (ICF) to achieve unbiased cross-modal understanding. Due to the lack of semantics, heterogeneous representations may lead to erroneous matches, especially in complex scenes with ambiguous visual content or interference from multiple audio sources. We introduce the multi-granularity implicit text (MIT) involving video-, segment- and frame-level as the bridge to establish the modality-shared space, reducing modality gaps and providing prior guidance. Visual content carries more information and typically dominates, thereby marginalizing audio features in the decision-making. To mitigate knowledge preference, we propose the semantic counterfactual (SC) to learn orthogonal representations in the latent space, generating diverse counterfactual samples, thus avoiding biases introduced by complex functional designs and explicit modifications of text structures or attributes. We further formulate the collaborative distribution-aware contrastive learning (CDCL), incorporating factual-counterfactual and inter-modality contrasts to align representations, promoting cohesion and decoupling. Extensive experiments on three public datasets validate that the proposed method achieves state-of-the-art performance.
>
---
#### [new 082] Not Only Grey Matter: OmniBrain for Robust Multimodal Classification of Alzheimer's Disease
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出OmniBrain，用于阿尔茨海默病的多模态分类任务，旨在解决现有方法在准确性、泛化性、模态缺失鲁棒性和可解释性上的不足。模型融合MRI、放射组学、基因表达和临床数据，采用交叉注意力与模态丢弃机制，在多个数据集上取得优异表现，并具备临床可解释性。**

- **链接: [http://arxiv.org/pdf/2507.20872v1](http://arxiv.org/pdf/2507.20872v1)**

> **作者:** Ahmed Sharshar; Yasser Ashraf; Tameem Bakr; Salma Hassan; Hosam Elgendy; Mohammad Yaqub; Mohsen Guizani
>
> **备注:** Published in Third Workshop on Computer Vision for Automated Medical Diagnosis CVAMD 2025 in ICCV 2025
>
> **摘要:** Alzheimer's disease affects over 55 million people worldwide and is projected to more than double by 2050, necessitating rapid, accurate, and scalable diagnostics. However, existing approaches are limited because they cannot achieve clinically acceptable accuracy, generalization across datasets, robustness to missing modalities, and explainability all at the same time. This inability to satisfy all these requirements simultaneously undermines their reliability in clinical settings. We propose OmniBrain, a multimodal framework that integrates brain MRI, radiomics, gene expression, and clinical data using a unified model with cross-attention and modality dropout. OmniBrain achieves $92.2 \pm 2.4\%$accuracy on the ANMerge dataset and generalizes to the MRI-only ADNI dataset with $70.4 \pm 2.7\%$ accuracy, outperforming unimodal and prior multimodal approaches. Explainability analyses highlight neuropathologically relevant brain regions and genes, enhancing clinical trust. OmniBrain offers a robust, interpretable, and practical solution for real-world Alzheimer's diagnosis.
>
---
#### [new 083] $S^3$LAM: Surfel Splatting SLAM for Geometrically Accurate Tracking and Mapping
- **分类: cs.CV**

- **简介: 本文提出了一种名为$S^3$LAM的RGB-D SLAM系统，用于实现高精度的几何跟踪与建图。该方法使用2D高斯surfels作为基本表示单元，通过2D surfel splatting策略提升几何精度，并推导了基于该表示的相机位姿雅可比矩阵，提高了跟踪收敛性。实验表明该方法在合成和真实数据集上均达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2507.20854v1](http://arxiv.org/pdf/2507.20854v1)**

> **作者:** Ruoyu Fan; Yuhui Wen; Jiajia Dai; Tao Zhang; Long Zeng; Yong-jin Liu
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** We propose $S^3$LAM, a novel RGB-D SLAM system that leverages 2D surfel splatting to achieve highly accurate geometric representations for simultaneous tracking and mapping. Unlike existing 3DGS-based SLAM approaches that rely on 3D Gaussian ellipsoids, we utilize 2D Gaussian surfels as primitives for more efficient scene representation. By focusing on the surfaces of objects in the scene, this design enables $S^3$LAM to reconstruct high-quality geometry, benefiting both mapping and tracking. To address inherent SLAM challenges including real-time optimization under limited viewpoints, we introduce a novel adaptive surface rendering strategy that improves mapping accuracy while maintaining computational efficiency. We further derive camera pose Jacobians directly from 2D surfel splatting formulation, highlighting the importance of our geometrically accurate representation that improves tracking convergence. Extensive experiments on both synthetic and real-world datasets validate that $S^3$LAM achieves state-of-the-art performance. Code will be made publicly available.
>
---
#### [new 084] Event-Based De-Snowing for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的图像去雪任务，旨在解决雪天条件下视觉系统可靠性问题。现有方法存在伪影或依赖高帧率。论文提出基于事件相机的去雪方法，利用事件数据中的雪花轨迹特征设计注意力模块，恢复背景强度。在新数据集DSEC-Snow上验证，图像重建PSNR提升3dB，并提升深度估计和光流性能，增强自动驾驶系统在雪天的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.20901v1](http://arxiv.org/pdf/2507.20901v1)**

> **作者:** Manasi Muglikar; Nico Messikommer; Marco Cannici; Davide Scaramuzza
>
> **摘要:** Adverse weather conditions, particularly heavy snowfall, pose significant challenges to both human drivers and autonomous vehicles. Traditional image-based de-snowing methods often introduce hallucination artifacts as they rely solely on spatial information, while video-based approaches require high frame rates and suffer from alignment artifacts at lower frame rates. Camera parameters, such as exposure time, also influence the appearance of snowflakes, making the problem difficult to solve and heavily dependent on network generalization. In this paper, we propose to address the challenge of desnowing by using event cameras, which offer compressed visual information with submillisecond latency, making them ideal for de-snowing images, even in the presence of ego-motion. Our method leverages the fact that snowflake occlusions appear with a very distinctive streak signature in the spatio-temporal representation of event data. We design an attention-based module that focuses on events along these streaks to determine when a background point was occluded and use this information to recover its original intensity. We benchmark our method on DSEC-Snow, a new dataset created using a green-screen technique that overlays pre-recorded snowfall data onto the existing DSEC driving dataset, resulting in precise ground truth and synchronized image and event streams. Our approach outperforms state-of-the-art de-snowing methods by 3 dB in PSNR for image reconstruction. Moreover, we show that off-the-shelf computer vision algorithms can be applied to our reconstructions for tasks such as depth estimation and optical flow, achieving a $20\%$ performance improvement over other de-snowing methods. Our work represents a crucial step towards enhancing the reliability and safety of vision systems in challenging winter conditions, paving the way for more robust, all-weather-capable applications.
>
---
#### [new 085] An Improved YOLOv8 Approach for Small Target Detection of Rice Spikelet Flowering in Field Environments
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决田间环境下水稻颖花因体积小、花期短导致的自动识别难题。论文改进了YOLOv8模型，引入BiFPN结构和小目标检测头，并构建专用数据集。实验表明，模型在检测精度和速度上均有提升，适用于杂交水稻制种中的自动监测需求。**

- **链接: [http://arxiv.org/pdf/2507.20506v1](http://arxiv.org/pdf/2507.20506v1)**

> **作者:** Beizhang Chen; Jinming Liang; Zheng Xiong; Ming Pan; Xiangbao Meng; Qingshan Lin; Qun Ma; Yingping Zhao
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Accurately detecting rice flowering time is crucial for timely pollination in hybrid rice seed production. This not only enhances pollination efficiency but also ensures higher yields. However, due to the complexity of field environments and the characteristics of rice spikelets, such as their small size and short flowering period, automated and precise recognition remains challenging. To address this, this study proposes a rice spikelet flowering recognition method based on an improved YOLOv8 object detection model. First, a Bidirectional Feature Pyramid Network (BiFPN) replaces the original PANet structure to enhance feature fusion and improve multi-scale feature utilization. Second, to boost small object detection, a p2 small-object detection head is added, using finer feature mapping to reduce feature loss commonly seen in detecting small targets. Given the lack of publicly available datasets for rice spikelet flowering in field conditions, a high-resolution RGB camera and data augmentation techniques are used to construct a dedicated dataset, providing reliable support for model training and testing. Experimental results show that the improved YOLOv8s-p2 model achieves an mAP@0.5 of 65.9%, precision of 67.6%, recall of 61.5%, and F1-score of 64.41%, representing improvements of 3.10%, 8.40%, 10.80%, and 9.79%, respectively, over the baseline YOLOv8. The model also runs at 69 f/s on the test set, meeting practical application requirements. Overall, the improved YOLOv8s-p2 offers high accuracy and speed, providing an effective solution for automated monitoring in hybrid rice seed production.
>
---
#### [new 086] SurgPIS: Surgical-instrument-level Instances and Part-level Semantics for Weakly-supervised Part-aware Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决手术器械实例与部件级语义分割的联合建模问题。作者提出SurgPIS模型，通过统一的part-aware instance segmentation方法，结合弱监督学习策略，实现对部分标注数据的有效利用，提升了手术器械的实例分割和部件识别效果。**

- **链接: [http://arxiv.org/pdf/2507.19592v1](http://arxiv.org/pdf/2507.19592v1)**

> **作者:** Meng Wei; Charlie Budd; Oluwatosin Alabi; Miaojing Shi; Tom Vercauteren
>
> **摘要:** Consistent surgical instrument segmentation is critical for automation in robot-assisted surgery. Yet, existing methods only treat instrument-level instance segmentation (IIS) or part-level semantic segmentation (PSS) separately, without interaction between these tasks. In this work, we formulate a surgical tool segmentation as a unified part-aware instance segmentation (PIS) problem and introduce SurgPIS, the first PIS model for surgical instruments. Our method adopts a transformer-based mask classification approach and introduces part-specific queries derived from instrument-level object queries, explicitly linking parts to their parent instrument instances. In order to address the lack of large-scale datasets with both instance- and part-level labels, we propose a weakly-supervised learning strategy for SurgPIS to learn from disjoint datasets labelled for either IIS or PSS purposes. During training, we aggregate our PIS predictions into IIS or PSS masks, thereby allowing us to compute a loss against partially labelled datasets. A student-teacher approach is developed to maintain prediction consistency for missing PIS information in the partially labelled data, e.g., parts of the IIS labelled data. Extensive experiments across multiple datasets validate the effectiveness of SurgPIS, achieving state-of-the-art performance in PIS as well as IIS, PSS, and instrument-level semantic segmentation.
>
---
#### [new 087] FedS2R: One-Shot Federated Domain Generalization for Synthetic-to-Real Semantic Segmentation in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 论文提出FedS2R，首次将联邦域泛化应用于自动驾驶中的合成到真实语义分割任务。旨在解决不共享原始数据的情况下，多客户端协同训练模型的域适应问题。方法包括不一致性驱动的数据增强和多客户端知识蒸馏。实验表明其性能优于单客户端模型，接近集中训练模型。**

- **链接: [http://arxiv.org/pdf/2507.19881v1](http://arxiv.org/pdf/2507.19881v1)**

> **作者:** Tao Lian; Jose L. Gómez; Antonio M. López
>
> **摘要:** Federated domain generalization has shown promising progress in image classification by enabling collaborative training across multiple clients without sharing raw data. However, its potential in the semantic segmentation of autonomous driving remains underexplored. In this paper, we propose FedS2R, the first one-shot federated domain generalization framework for synthetic-to-real semantic segmentation in autonomous driving. FedS2R comprises two components: an inconsistency-driven data augmentation strategy that generates images for unstable classes, and a multi-client knowledge distillation scheme with feature fusion that distills a global model from multiple client models. Experiments on five real-world datasets, Cityscapes, BDD100K, Mapillary, IDD, and ACDC, show that the global model significantly outperforms individual client models and is only 2 mIoU points behind the model trained with simultaneous access to all client data. These results demonstrate the effectiveness of FedS2R in synthetic-to-real semantic segmentation for autonomous driving under federated learning
>
---
#### [new 088] GTAD: Global Temporal Aggregation Denoising Learning for 3D Semantic Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文属于3D语义占用预测任务，旨在解决现有方法对时序信息利用不足的问题。作者提出GTAD网络，通过全局时序特征聚合与去噪学习，提升动态环境感知能力，实现更连贯、全面的3D场景理解。**

- **链接: [http://arxiv.org/pdf/2507.20963v1](http://arxiv.org/pdf/2507.20963v1)**

> **作者:** Tianhao Li; Yang Li; Mengtian Li; Yisheng Deng; Weifeng Ge
>
> **摘要:** Accurately perceiving dynamic environments is a fundamental task for autonomous driving and robotic systems. Existing methods inadequately utilize temporal information, relying mainly on local temporal interactions between adjacent frames and failing to leverage global sequence information effectively. To address this limitation, we investigate how to effectively aggregate global temporal features from temporal sequences, aiming to achieve occupancy representations that efficiently utilize global temporal information from historical observations. For this purpose, we propose a global temporal aggregation denoising network named GTAD, introducing a global temporal information aggregation framework as a new paradigm for holistic 3D scene understanding. Our method employs an in-model latent denoising network to aggregate local temporal features from the current moment and global temporal features from historical sequences. This approach enables the effective perception of both fine-grained temporal information from adjacent frames and global temporal patterns from historical observations. As a result, it provides a more coherent and comprehensive understanding of the environment. Extensive experiments on the nuScenes and Occ3D-nuScenes benchmark and ablation studies demonstrate the superiority of our method.
>
---
#### [new 089] DepthFlow: Exploiting Depth-Flow Structural Correlations for Unsupervised Video Object Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割任务，旨在解决无监督学习中训练数据不足的问题。作者提出DepthFlow方法，通过单张图像生成深度图，并转化为光流场，合成大量训练数据，显著提升模型性能，实现当前最优效果。**

- **链接: [http://arxiv.org/pdf/2507.19790v1](http://arxiv.org/pdf/2507.19790v1)**

> **作者:** Suhwan Cho; Minhyeok Lee; Jungho Lee; Donghyeong Kim; Sangyoun Lee
>
> **备注:** ICCVW 2025
>
> **摘要:** Unsupervised video object segmentation (VOS) aims to detect the most prominent object in a video. Recently, two-stream approaches that leverage both RGB images and optical flow have gained significant attention, but their performance is fundamentally constrained by the scarcity of training data. To address this, we propose DepthFlow, a novel data generation method that synthesizes optical flow from single images. Our approach is driven by the key insight that VOS models depend more on structural information embedded in flow maps than on their geometric accuracy, and that this structure is highly correlated with depth. We first estimate a depth map from a source image and then convert it into a synthetic flow field that preserves essential structural cues. This process enables the transformation of large-scale image-mask pairs into image-flow-mask training pairs, dramatically expanding the data available for network training. By training a simple encoder-decoder architecture with our synthesized data, we achieve new state-of-the-art performance on all public VOS benchmarks, demonstrating a scalable and effective solution to the data scarcity problem.
>
---
#### [new 090] Towards Universal Modal Tracking with Online Dense Temporal Token Learning
- **分类: cs.CV**

- **简介: 该论文属于多模态视频目标跟踪任务，旨在解决现有方法依赖特定模态和独立训练的问题。作者提出了一种通用的视频级跟踪模型ModalTracker，通过在线密集时序令牌学习，实现跨模态、多任务的统一跟踪。**

- **链接: [http://arxiv.org/pdf/2507.20177v1](http://arxiv.org/pdf/2507.20177v1)**

> **作者:** Yaozong Zheng; Bineng Zhong; Qihua Liang; Shengping Zhang; Guorong Li; Xianxian Li; Rongrong Ji
>
> **备注:** arXiv admin note: text overlap with arXiv:2401.01686
>
> **摘要:** We propose a universal video-level modality-awareness tracking model with online dense temporal token learning (called {\modaltracker}). It is designed to support various tracking tasks, including RGB, RGB+Thermal, RGB+Depth, and RGB+Event, utilizing the same model architecture and parameters. Specifically, our model is designed with three core goals: \textbf{Video-level Sampling}. We expand the model's inputs to a video sequence level, aiming to see a richer video context from an near-global perspective. \textbf{Video-level Association}. Furthermore, we introduce two simple yet effective online dense temporal token association mechanisms to propagate the appearance and motion trajectory information of target via a video stream manner. \textbf{Modality Scalable}. We propose two novel gated perceivers that adaptively learn cross-modal representations via a gated attention mechanism, and subsequently compress them into the same set of model parameters via a one-shot training manner for multi-task inference. This new solution brings the following benefits: (i) The purified token sequences can serve as temporal prompts for the inference in the next video frames, whereby previous information is leveraged to guide future inference. (ii) Unlike multi-modal trackers that require independent training, our one-shot training scheme not only alleviates the training burden, but also improves model representation. Extensive experiments on visible and multi-modal benchmarks show that our {\modaltracker} achieves a new \textit{SOTA} performance. The code will be available at https://github.com/GXNU-ZhongLab/ODTrack.
>
---
#### [new 091] Hybrid-Domain Synergistic Transformer for Hyperspectral Image Denoising
- **分类: cs.CV**

- **简介: 该论文属于图像处理任务，旨在解决高光谱图像中的复杂噪声耦合问题。作者提出了HDST网络，结合频域增强与多尺度建模，实现空间、频域和通道的三维协同去噪。方法包含FFT预处理、跨域注意力机制和多级架构，有效提升去噪性能与计算效率。**

- **链接: [http://arxiv.org/pdf/2507.20099v1](http://arxiv.org/pdf/2507.20099v1)**

> **作者:** Haoyue Li; Di Wu
>
> **备注:** 10 pages, 4 figures, 4 tables
>
> **摘要:** Hyperspectral image denoising faces the challenge of multi-dimensional coupling of spatially non-uniform noise and spectral correlation interference. Existing deep learning methods mostly focus on RGB images and struggle to effectively handle the unique spatial-spectral characteristics and complex noise distributions of hyperspectral images (HSI). This paper proposes an HSI denoising framework, Hybrid-Domain Synergistic Transformer Network (HDST), based on frequency domain enhancement and multiscale modeling, achieving three-dimensional collaborative processing of spatial, frequency and channel domains. The method innovatively integrates three key mechanisms: (1) introducing an FFT preprocessing module with multi-band convolution to extract cross-band correlations and decouple spectral noise components; (2) designing a dynamic cross-domain attention module that adaptively fuses spatial domain texture features and frequency domain noise priors through a learnable gating mechanism; (3) building a hierarchical architecture where shallow layers capture global noise statistics using multiscale atrous convolution, and deep layers achieve detail recovery through frequency domain postprocessing. Experiments on both real and synthetic datasets demonstrate that HDST significantly improves denoising performance while maintaining computational efficiency, validating the effectiveness of the proposed method. This research provides new insights and a universal framework for addressing complex noise coupling issues in HSI and other high-dimensional visual data. The code is available at https://github.com/lhy-cn/HDST-HSIDenoise.
>
---
#### [new 092] DAMS:Dual-Branch Adaptive Multiscale Spatiotemporal Framework for Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，旨在定位视频中的时空异常事件。为解决多尺度时间依赖、视觉语义异构性和标注数据稀缺问题，论文提出了DAMS双分支框架，融合多层次特征学习与互补信息，结合AMTPN、CBAM与CLIP，实现高效异常检测。**

- **链接: [http://arxiv.org/pdf/2507.20629v1](http://arxiv.org/pdf/2507.20629v1)**

> **作者:** Dezhi An; Wenqiang Liu; Kefan Wang; Zening Chen; Jun Lu; Shengcai Zhang
>
> **备注:** 13 pages,7 figures
>
> **摘要:** The goal of video anomaly detection is tantamount to performing spatio-temporal localization of abnormal events in the video. The multiscale temporal dependencies, visual-semantic heterogeneity, and the scarcity of labeled data exhibited by video anomalies collectively present a challenging research problem in computer vision. This study offers a dual-path architecture called the Dual-Branch Adaptive Multiscale Spatiotemporal Framework (DAMS), which is based on multilevel feature decoupling and fusion, enabling efficient anomaly detection modeling by integrating hierarchical feature learning and complementary information. The main processing path of this framework integrates the Adaptive Multiscale Time Pyramid Network (AMTPN) with the Convolutional Block Attention Mechanism (CBAM). AMTPN enables multigrained representation and dynamically weighted reconstruction of temporal features through a three-level cascade structure (time pyramid pooling, adaptive feature fusion, and temporal context enhancement). CBAM maximizes the entropy distribution of feature channels and spatial dimensions through dual attention mapping. Simultaneously, the parallel path driven by CLIP introduces a contrastive language-visual pre-training paradigm. Cross-modal semantic alignment and a multiscale instance selection mechanism provide high-order semantic guidance for spatio-temporal features. This creates a complete inference chain from the underlying spatio-temporal features to high-level semantic concepts. The orthogonal complementarity of the two paths and the information fusion mechanism jointly construct a comprehensive representation and identification capability for anomalous events. Extensive experimental results on the UCF-Crime and XD-Violence benchmarks establish the effectiveness of the DAMS framework.
>
---
#### [new 093] VESPA: Towards un(Human)supervised Open-World Pointcloud Labeling for Autonomous Driving
- **分类: cs.CV**

- **简介: 论文提出VESPA，一种多模态自动标注方法，用于自动驾驶中的开放世界点云标注任务。它融合LiDAR几何信息与相机图像语义信息，利用视觉-语言模型实现无需人工标注的高质量3D伪标签生成，解决现有方法在语义粒度、稀疏点云处理及新类别发现上的限制。**

- **链接: [http://arxiv.org/pdf/2507.20397v1](http://arxiv.org/pdf/2507.20397v1)**

> **作者:** Levente Tempfli; Esteban Rivera; Markus Lienkamp
>
> **摘要:** Data collection for autonomous driving is rapidly accelerating, but manual annotation, especially for 3D labels, remains a major bottleneck due to its high cost and labor intensity. Autolabeling has emerged as a scalable alternative, allowing the generation of labels for point clouds with minimal human intervention. While LiDAR-based autolabeling methods leverage geometric information, they struggle with inherent limitations of lidar data, such as sparsity, occlusions, and incomplete object observations. Furthermore, these methods typically operate in a class-agnostic manner, offering limited semantic granularity. To address these challenges, we introduce VESPA, a multimodal autolabeling pipeline that fuses the geometric precision of LiDAR with the semantic richness of camera images. Our approach leverages vision-language models (VLMs) to enable open-vocabulary object labeling and to refine detection quality directly in the point cloud domain. VESPA supports the discovery of novel categories and produces high-quality 3D pseudolabels without requiring ground-truth annotations or HD maps. On Nuscenes dataset, VESPA achieves an AP of 52.95% for object discovery and up to 46.54% for multiclass object detection, demonstrating strong performance in scalable 3D scene understanding. Code will be available upon acceptance.
>
---
#### [new 094] Investigation of Accuracy and Bias in Face Recognition Trained with Synthetic Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究使用合成数据训练人脸识别模型的准确性和偏差问题。任务是评估合成数据对模型性能和公平性的影响。通过生成平衡人脸数据集 FairFaceGen，并结合多种数据增强方法，验证合成数据在标准和挑战性数据集上的表现。结果表明，合成数据在公平性方面有潜力，但泛化能力仍弱于真实数据。**

- **链接: [http://arxiv.org/pdf/2507.20782v1](http://arxiv.org/pdf/2507.20782v1)**

> **作者:** Pavel Korshunov; Ketan Kotwal; Christophe Ecabert; Vidit Vidit; Amir Mohammadi; Sebastien Marcel
>
> **备注:** Accepted for publication in IEEE International Joint Conference on Biometrics (IJCB), 2025
>
> **摘要:** Synthetic data has emerged as a promising alternative for training face recognition (FR) models, offering advantages in scalability, privacy compliance, and potential for bias mitigation. However, critical questions remain on whether both high accuracy and fairness can be achieved with synthetic data. In this work, we evaluate the impact of synthetic data on bias and performance of FR systems. We generate balanced face dataset, FairFaceGen, using two state of the art text-to-image generators, Flux.1-dev and Stable Diffusion v3.5 (SD35), and combine them with several identity augmentation methods, including Arc2Face and four IP-Adapters. By maintaining equal identity count across synthetic and real datasets, we ensure fair comparisons when evaluating FR performance on standard (LFW, AgeDB-30, etc.) and challenging IJB-B/C benchmarks and FR bias on Racial Faces in-the-Wild (RFW) dataset. Our results demonstrate that although synthetic data still lags behind the real datasets in the generalization on IJB-B/C, demographically balanced synthetic datasets, especially those generated with SD35, show potential for bias mitigation. We also observe that the number and quality of intra-class augmentations significantly affect FR accuracy and fairness. These findings provide practical guidelines for constructing fairer FR systems using synthetic data.
>
---
#### [new 095] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型压缩任务，旨在解决大型视觉-语言模型推理成本高的问题。作者提出TransPrune方法，通过评估标记在表示变化（TTV）和指令引导注意力（IGA）来剪枝不重要的视觉标记，从而提升推理效率，同时保持模型性能。**

- **链接: [http://arxiv.org/pdf/2507.20630v1](http://arxiv.org/pdf/2507.20630v1)**

> **作者:** Ao Li; Yuxiang Duan; Jinghui Zhang; Congbo Ma; Yutong Xie; Gustavo Carneiro; Mohammad Yaqub; Hu Wang
>
> **摘要:** Large Vision-Language Models (LVLMs) have advanced multimodal learning but face high computational costs due to the large number of visual tokens, motivating token pruning to improve inference efficiency. The key challenge lies in identifying which tokens are truly important. Most existing approaches rely on attention-based criteria to estimate token importance. However, they inherently suffer from certain limitations, such as positional bias. In this work, we explore a new perspective on token importance based on token transitions in LVLMs. We observe that the transition of token representations provides a meaningful signal of semantic information. Based on this insight, we propose TransPrune, a training-free and efficient token pruning method. Specifically, TransPrune progressively prunes tokens by assessing their importance through a combination of Token Transition Variation (TTV)-which measures changes in both the magnitude and direction of token representations-and Instruction-Guided Attention (IGA), which measures how strongly the instruction attends to image tokens via attention. Extensive experiments demonstrate that TransPrune achieves comparable multimodal performance to original LVLMs, such as LLaVA-v1.5 and LLaVA-Next, across eight benchmarks, while reducing inference TFLOPs by more than half. Moreover, TTV alone can serve as an effective criterion without relying on attention, achieving performance comparable to attention-based methods. The code will be made publicly available upon acceptance of the paper at https://github.com/liaolea/TransPrune.
>
---
#### [new 096] Player-Centric Multimodal Prompt Generation for Large Language Model Based Identity-Aware Basketball Video Captioning
- **分类: cs.CV**

- **简介: 该论文属于体育视频描述生成任务，旨在解决现有方法忽略球员身份的问题。通过设计多模态提示生成网络LLM-IAVC，从视觉角度识别球员身份，并提出IRIEM模块提取球员相关信息，结合VCLM学习视频上下文，最终生成包含球员身份的描述。**

- **链接: [http://arxiv.org/pdf/2507.20163v1](http://arxiv.org/pdf/2507.20163v1)**

> **作者:** Zeyu Xi; Haoying Sun; Yaofei Wu; Junchi Yan; Haoran Zhang; Lifang Wu; Liang Wang; Changwen Chen
>
> **备注:** Accepted by ICCV 2025 (Poster)
>
> **摘要:** Existing sports video captioning methods often focus on the action yet overlook player identities, limiting their applicability. Although some methods integrate extra information to generate identity-aware descriptions, the player identities are sometimes incorrect because the extra information is independent of the video content. This paper proposes a player-centric multimodal prompt generation network for identity-aware sports video captioning (LLM-IAVC), which focuses on recognizing player identities from a visual perspective. Specifically, an identity-related information extraction module (IRIEM) is designed to extract player-related multimodal embeddings. IRIEM includes a player identification network (PIN) for extracting visual features and player names, and a bidirectional semantic interaction module (BSIM) to link player features with video content for mutual enhancement. Additionally, a visual context learning module (VCLM) is designed to capture the key video context information. Finally, by integrating the outputs of the above modules as the multimodal prompt for the large language model (LLM), it facilitates the generation of descriptions with player identities. To support this work, we construct a new benchmark called NBA-Identity, a large identity-aware basketball video captioning dataset with 9,726 videos covering 9 major event types. The experimental results on NBA-Identity and VC-NBA-2022 demonstrate that our proposed model achieves advanced performance. Code and dataset are publicly available at https://github.com/Zeyu1226-mt/LLM-IAVC.
>
---
#### [new 097] Is Exchangeability better than I.I.D to handle Data Distribution Shifts while Pooling Data for Data-scarce Medical image segmentation?
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在解决数据稀缺下多源数据融合导致的分布偏移问题。论文提出通过因果框架控制特征差异的方法，假设数据间可交换而非独立同分布，优化特征表示，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.19575v1](http://arxiv.org/pdf/2507.19575v1)**

> **作者:** Ayush Roy; Samin Enam; Jun Xia; Vishnu Suresh Lokhande; Won Hwa Kim
>
> **摘要:** Data scarcity is a major challenge in medical imaging, particularly for deep learning models. While data pooling (combining datasets from multiple sources) and data addition (adding more data from a new dataset) have been shown to enhance model performance, they are not without complications. Specifically, increasing the size of the training dataset through pooling or addition can induce distributional shifts, negatively affecting downstream model performance, a phenomenon known as the "Data Addition Dilemma". While the traditional i.i.d. assumption may not hold in multi-source contexts, assuming exchangeability across datasets provides a more practical framework for data pooling. In this work, we investigate medical image segmentation under these conditions, drawing insights from causal frameworks to propose a method for controlling foreground-background feature discrepancies across all layers of deep networks. This approach improves feature representations, which are crucial in data-addition scenarios. Our method achieves state-of-the-art segmentation performance on histopathology and ultrasound images across five datasets, including a novel ultrasound dataset that we have curated and contributed. Qualitative results demonstrate more refined and accurate segmentation maps compared to prominent baselines across three model architectures. The code will be available on Github.
>
---
#### [new 098] RingMo-Agent: A Unified Remote Sensing Foundation Model for Multi-Platform and Multi-Modal Reasoning
- **分类: cs.CV**

- **简介: 该论文属于遥感视觉语言任务，旨在解决现有方法在多源遥感数据处理中的局限性。作者提出RingMo-Agent，构建大规模多模态数据集RS-VL3M，采用分离嵌入层学习模态自适应表示，并引入任务特定标记实现多任务统一建模，以提升跨平台、跨模态的遥感图像理解和推理能力。**

- **链接: [http://arxiv.org/pdf/2507.20776v1](http://arxiv.org/pdf/2507.20776v1)**

> **作者:** Huiyang Hu; Peijin Wang; Yingchao Feng; Kaiwen Wei; Wenxin Yin; Wenhui Diao; Mengyu Wang; Hanbo Bi; Kaiyue Kang; Tong Ling; Kun Fu; Xian Sun
>
> **备注:** 21 pages, 6 figures, 20 tables
>
> **摘要:** Remote sensing (RS) images from multiple modalities and platforms exhibit diverse details due to differences in sensor characteristics and imaging perspectives. Existing vision-language research in RS largely relies on relatively homogeneous data sources. Moreover, they still remain limited to conventional visual perception tasks such as classification or captioning. As a result, these methods fail to serve as a unified and standalone framework capable of effectively handling RS imagery from diverse sources in real-world applications. To address these issues, we propose RingMo-Agent, a model designed to handle multi-modal and multi-platform data that performs perception and reasoning tasks based on user textual instructions. Compared with existing models, RingMo-Agent 1) is supported by a large-scale vision-language dataset named RS-VL3M, comprising over 3 million image-text pairs, spanning optical, SAR, and infrared (IR) modalities collected from both satellite and UAV platforms, covering perception and challenging reasoning tasks; 2) learns modality adaptive representations by incorporating separated embedding layers to construct isolated features for heterogeneous modalities and reduce cross-modal interference; 3) unifies task modeling by introducing task-specific tokens and employing a token-based high-dimensional hidden state decoding mechanism designed for long-horizon spatial tasks. Extensive experiments on various RS vision-language tasks demonstrate that RingMo-Agent not only proves effective in both visual understanding and sophisticated analytical tasks, but also exhibits strong generalizability across different platforms and sensing modalities.
>
---
#### [new 099] T2I-Copilot: A Training-Free Multi-Agent Text-to-Image System for Enhanced Prompt Interpretation and Interactive Generation
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文属于文本到图像生成任务，旨在解决生成模型对提示敏感、需多次调整的问题。论文提出T2I-Copilot，一种无需训练的多智能体系统，通过协作实现提示解析、模型选择与迭代优化，提升生成质量与图文对齐，支持自动化与人工干预结合。**

- **链接: [http://arxiv.org/pdf/2507.20536v1](http://arxiv.org/pdf/2507.20536v1)**

> **作者:** Chieh-Yun Chen; Min Shi; Gong Zhang; Humphrey Shi
>
> **备注:** ICCV 2025
>
> **摘要:** Text-to-Image (T2I) generative models have revolutionized content creation but remain highly sensitive to prompt phrasing, often requiring users to repeatedly refine prompts multiple times without clear feedback. While techniques such as automatic prompt engineering, controlled text embeddings, denoising, and multi-turn generation mitigate these issues, they offer limited controllability, or often necessitate additional training, restricting the generalization abilities. Thus, we introduce T2I-Copilot, a training-free multi-agent system that leverages collaboration between (Multimodal) Large Language Models to automate prompt phrasing, model selection, and iterative refinement. This approach significantly simplifies prompt engineering while enhancing generation quality and text-image alignment compared to direct generation. Specifically, T2I-Copilot consists of three agents: (1) Input Interpreter, which parses the input prompt, resolves ambiguities, and generates a standardized report; (2) Generation Engine, which selects the appropriate model from different types of T2I models and organizes visual and textual prompts to initiate generation; and (3) Quality Evaluator, which assesses aesthetic quality and text-image alignment, providing scores and feedback for potential regeneration. T2I-Copilot can operate fully autonomously while also supporting human-in-the-loop intervention for fine-grained control. On GenAI-Bench, using open-source generation models, T2I-Copilot achieves a VQA score comparable to commercial models RecraftV3 and Imagen 3, surpasses FLUX1.1-pro by 6.17% at only 16.59% of its cost, and outperforms FLUX.1-dev and SD 3.5 Large by 9.11% and 6.36%. Code will be released at: https://github.com/SHI-Labs/T2I-Copilot.
>
---
#### [new 100] ModalFormer: Multimodal Transformer for Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决低光图像中噪声多、细节丢失和对比度差的问题。论文提出了ModalFormer，首个大规模多模态框架，结合九种辅助模态，通过跨模态Transformer与新型注意力机制，实现图像增强。**

- **链接: [http://arxiv.org/pdf/2507.20388v1](http://arxiv.org/pdf/2507.20388v1)**

> **作者:** Alexandru Brateanu; Raul Balmez; Ciprian Orhei; Codruta Ancuti; Cosmin Ancuti
>
> **摘要:** Low-light image enhancement (LLIE) is a fundamental yet challenging task due to the presence of noise, loss of detail, and poor contrast in images captured under insufficient lighting conditions. Recent methods often rely solely on pixel-level transformations of RGB images, neglecting the rich contextual information available from multiple visual modalities. In this paper, we present ModalFormer, the first large-scale multimodal framework for LLIE that fully exploits nine auxiliary modalities to achieve state-of-the-art performance. Our model comprises two main components: a Cross-modal Transformer (CM-T) designed to restore corrupted images while seamlessly integrating multimodal information, and multiple auxiliary subnetworks dedicated to multimodal feature reconstruction. Central to the CM-T is our novel Cross-modal Multi-headed Self-Attention mechanism (CM-MSA), which effectively fuses RGB data with modality-specific features--including deep feature embeddings, segmentation information, geometric cues, and color information--to generate information-rich hybrid attention maps. Extensive experiments on multiple benchmark datasets demonstrate ModalFormer's state-of-the-art performance in LLIE. Pre-trained models and results are made available at https://github.com/albrateanu/ModalFormer.
>
---
#### [new 101] LargeMvC-Net: Anchor-based Deep Unfolding Network for Large-scale Multi-view Clustering
- **分类: cs.CV; stat.CO; stat.ML**

- **简介: 该论文属于多视图聚类任务，旨在解决大规模数据下聚类的可扩展性和结构优化问题。作者提出LargeMvC-Net，通过将基于锚点的聚类优化过程展开为深度网络结构，包含表示学习、噪声抑制和锚点指示估计模块，提升聚类效果与可扩展性。**

- **链接: [http://arxiv.org/pdf/2507.20980v1](http://arxiv.org/pdf/2507.20980v1)**

> **作者:** Shide Du; Chunming Wu; Zihan Fang; Wendi Zhao; Yilin Wu; Changwei Wang; Shiping Wang
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Deep anchor-based multi-view clustering methods enhance the scalability of neural networks by utilizing representative anchors to reduce the computational complexity of large-scale clustering. Despite their scalability advantages, existing approaches often incorporate anchor structures in a heuristic or task-agnostic manner, either through post-hoc graph construction or as auxiliary components for message passing. Such designs overlook the core structural demands of anchor-based clustering, neglecting key optimization principles. To bridge this gap, we revisit the underlying optimization problem of large-scale anchor-based multi-view clustering and unfold its iterative solution into a novel deep network architecture, termed LargeMvC-Net. The proposed model decomposes the anchor-based clustering process into three modules: RepresentModule, NoiseModule, and AnchorModule, corresponding to representation learning, noise suppression, and anchor indicator estimation. Each module is derived by unfolding a step of the original optimization procedure into a dedicated network component, providing structural clarity and optimization traceability. In addition, an unsupervised reconstruction loss aligns each view with the anchor-induced latent space, encouraging consistent clustering structures across views. Extensive experiments on several large-scale multi-view benchmarks show that LargeMvC-Net consistently outperforms state-of-the-art methods in terms of both effectiveness and scalability.
>
---
#### [new 102] LRR-Bench: Left, Right or Rotate? Vision-Language models Still Struggle With Spatial Understanding Tasks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型的空间理解任务，旨在解决现有VLMs在空间关系识别和空间运动感知上的不足。作者构建了一个合成数据基准LRR-Bench，分为绝对空间理解和3D空间理解两类任务。实验表明，当前VLMs表现远不如人类，尤其在复杂任务上表现接近零。**

- **链接: [http://arxiv.org/pdf/2507.20174v1](http://arxiv.org/pdf/2507.20174v1)**

> **作者:** Fei Kong; Jinhao Duan; Kaidi Xu; Zhenhua Guo; Xiaofeng Zhu; Xiaoshuang Shi
>
> **摘要:** Real-world applications, such as autonomous driving and humanoid robot manipulation, require precise spatial perception. However, it remains underexplored how Vision-Language Models (VLMs) recognize spatial relationships and perceive spatial movement. In this work, we introduce a spatial evaluation pipeline and construct a corresponding benchmark. Specifically, we categorize spatial understanding into two main types: absolute spatial understanding, which involves querying the absolute spatial position (e.g., left, right) of an object within an image, and 3D spatial understanding, which includes movement and rotation. Notably, our dataset is entirely synthetic, enabling the generation of test samples at a low cost while also preventing dataset contamination. We conduct experiments on multiple state-of-the-art VLMs and observe that there is significant room for improvement in their spatial understanding abilities. Explicitly, in our experiments, humans achieve near-perfect performance on all tasks, whereas current VLMs attain human-level performance only on the two simplest tasks. For the remaining tasks, the performance of VLMs is distinctly lower than that of humans. In fact, the best-performing Vision-Language Models even achieve near-zero scores on multiple tasks. The dataset and code are available on https://github.com/kong13661/LRR-Bench.
>
---
#### [new 103] Deep Learning for Skeleton Based Human Motion Rehabilitation Assessment: A Benchmark
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于人体运动康复评估任务，旨在解决缺乏标准数据集和统一评估方法的问题。作者整合多个康复数据集为统一库Rehab-Pile，提出基准框架并评估多种深度学习模型，推动自动化康复评估研究的发展。**

- **链接: [http://arxiv.org/pdf/2507.21018v1](http://arxiv.org/pdf/2507.21018v1)**

> **作者:** Ali Ismail-Fawaz; Maxime Devanne; Stefano Berretti; Jonathan Weber; Germain Forestier
>
> **摘要:** Automated assessment of human motion plays a vital role in rehabilitation, enabling objective evaluation of patient performance and progress. Unlike general human activity recognition, rehabilitation motion assessment focuses on analyzing the quality of movement within the same action class, requiring the detection of subtle deviations from ideal motion. Recent advances in deep learning and video-based skeleton extraction have opened new possibilities for accessible, scalable motion assessment using affordable devices such as smartphones or webcams. However, the field lacks standardized benchmarks, consistent evaluation protocols, and reproducible methodologies, limiting progress and comparability across studies. In this work, we address these gaps by (i) aggregating existing rehabilitation datasets into a unified archive called Rehab-Pile, (ii) proposing a general benchmarking framework for evaluating deep learning methods in this domain, and (iii) conducting extensive benchmarking of multiple architectures across classification and regression tasks. All datasets and implementations are released to the community to support transparency and reproducibility. This paper aims to establish a solid foundation for future research in automated rehabilitation assessment and foster the development of reliable, accessible, and personalized rehabilitation solutions. The datasets, source-code and results of this article are all publicly available.
>
---
#### [new 104] MoCTEFuse: Illumination-Gated Mixture of Chiral Transformer Experts for Multi-Level Infrared and Visible Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于图像融合任务，旨在解决光照变化导致的模态偏差问题。作者提出MoCTEFuse网络，通过引入光照门控机制和双模态Transformer模块，动态融合红外与可见光图像，提升细节与对比度。实验表明其在多个数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.20180v1](http://arxiv.org/pdf/2507.20180v1)**

> **作者:** Li Jinfu; Song Hong; Xia Jianghan; Lin Yucong; Wang Ting; Shao Long; Fan Jingfan; Yang Jian
>
> **摘要:** While illumination changes inevitably affect the quality of infrared and visible image fusion, many outstanding methods still ignore this factor and directly merge the information from source images, leading to modality bias in the fused results. To this end, we propose a dynamic multi-level image fusion network called MoCTEFuse, which applies an illumination-gated Mixture of Chiral Transformer Experts (MoCTE) to adaptively preserve texture details and object contrasts in balance. MoCTE consists of high- and low-illumination expert subnetworks, each built upon the Chiral Transformer Fusion Block (CTFB). Guided by the illumination gating signals, CTFB dynamically switches between the primary and auxiliary modalities as well as assigning them corresponding weights with its asymmetric cross-attention mechanism. Meanwhile, it is stacked at multiple stages to progressively aggregate and refine modality-specific and cross-modality information. To facilitate robust training, we propose a competitive loss function that integrates illumination distributions with three levels of sub-loss terms. Extensive experiments conducted on the DroneVehicle, MSRS, TNO and RoadScene datasets show MoCTEFuse's superior fusion performance. Finally, it achieves the best detection mean Average Precision (mAP) of 70.93% on the MFNet dataset and 45.14% on the DroneVehicle dataset. The code and model are released at https://github.com/Bitlijinfu/MoCTEFuse.
>
---
#### [new 105] RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters
- **分类: cs.CV**

- **简介: 该论文属于虚拟环境中人群疏散仿真任务，旨在解决现有模型忽略复杂人类行为导致仿真不真实的问题。作者提出RESCUE框架，结合3D自适应社会力模型与个性化步态控制，实现多人并行、动态感知的疏散模拟，并引入局部力可视化辅助分析，提升了疏散仿真的真实感与适用性。**

- **链接: [http://arxiv.org/pdf/2507.20117v1](http://arxiv.org/pdf/2507.20117v1)**

> **作者:** Xiaolin Liu; Tianyi Zhou; Hongbo Kang; Jian Ma; Ziwen Wang; Jing Huang; Wenguo Weng; Yu-Kun Lai; Kun Li
>
> **摘要:** Crowd evacuation simulation is critical for enhancing public safety, and demanded for realistic virtual environments. Current mainstream evacuation models overlook the complex human behaviors that occur during evacuation, such as pedestrian collisions, interpersonal interactions, and variations in behavior influenced by terrain types or individual body shapes. This results in the failure to accurately simulate the escape of people in the real world. In this paper, aligned with the sensory-decision-motor (SDM) flow of the human brain, we propose a real-time 3D crowd evacuation simulation framework that integrates a 3D-adaptive SFM (Social Force Model) Decision Mechanism and a Personalized Gait Control Motor. This framework allows multiple agents to move in parallel and is suitable for various scenarios, with dynamic crowd awareness. Additionally, we introduce Part-level Force Visualization to assist in evacuation analysis. Experimental results demonstrate that our framework supports dynamic trajectory planning and personalized behavior for each agent throughout the evacuation process, and is compatible with uneven terrain. Visually, our method generates evacuation results that are more realistic and plausible, providing enhanced insights for crowd simulation. The code is available at http://cic.tju.edu.cn/faculty/likun/projects/RESCUE.
>
---
#### [new 106] Lightweight Remote Sensing Scene Classification on Edge Devices via Knowledge Distillation and Early-exit
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感场景分类任务，旨在解决边缘设备上模型轻量化与性能优化问题。通过知识蒸馏压缩全局滤波网络，并设计早期退出机制，提升推理速度与能效，实验证明方法有效。**

- **链接: [http://arxiv.org/pdf/2507.20623v1](http://arxiv.org/pdf/2507.20623v1)**

> **作者:** Yang Zhao; Shusheng Li; Xueshang Feng
>
> **备注:** 9 pages, 5 figures, to be published in ACM Multimedia 2025
>
> **摘要:** As the development of lightweight deep learning algorithms, various deep neural network (DNN) models have been proposed for the remote sensing scene classification (RSSC) application. However, it is still challenging for these RSSC models to achieve optimal performance among model accuracy, inference latency, and energy consumption on resource-constrained edge devices. In this paper, we propose a lightweight RSSC framework, which includes a distilled global filter network (GFNet) model and an early-exit mechanism designed for edge devices to achieve state-of-the-art performance. Specifically, we first apply frequency domain distillation on the GFNet model to reduce model size. Then we design a dynamic early-exit model tailored for DNN models on edge devices to further improve model inference efficiency. We evaluate our E3C model on three edge devices across four datasets. Extensive experimental results show that it achieves an average of 1.3x speedup on model inference and over 40% improvement on energy efficiency, while maintaining high classification accuracy.
>
---
#### [new 107] FM-LC: A Hierarchical Framework for Urban Flood Mapping by Land Cover Identification Models
- **分类: cs.CV; 86A32, 62H35; I.4.8; I.2.10; I.5.4**

- **简介: 该论文属于城市洪水制图任务，旨在解决干旱地区洪水 mapping 的挑战。通过提出FM-LC分层框架，结合多类U-Net分割、二值专家模型与贝叶斯平滑，提升洪水识别精度。实验验证了方法在迪拜暴雨事件中的有效性，显著优于传统方法。**

- **链接: [http://arxiv.org/pdf/2507.19818v1](http://arxiv.org/pdf/2507.19818v1)**

> **作者:** Xin Hong; Longchao Da; Hua Wei
>
> **备注:** 5 pages and 4 figures. Submitted to the IEEE for possible publication
>
> **摘要:** Urban flooding in arid regions poses severe risks to infrastructure and communities. Accurate, fine-scale mapping of flood extents and recovery trajectories is therefore essential for improving emergency response and resilience planning. However, arid environments often exhibit limited spectral contrast between water and adjacent surfaces, rapid hydrological dynamics, and highly heterogeneous urban land covers, which challenge traditional flood-mapping approaches. High-resolution, daily PlanetScope imagery provides the temporal and spatial detail needed. In this work, we introduce FM-LC, a hierarchical framework for Flood Mapping by Land Cover identification, for this challenging task. Through a three-stage process, it first uses an initial multi-class U-Net to segment imagery into water, vegetation, built area, and bare ground classes. We identify that this method has confusion between spectrally similar categories (e.g., water vs. vegetation). Second, by early checking, the class with the major misclassified area is flagged, and a lightweight binary expert segmentation model is trained to distinguish the flagged class from the rest. Third, a Bayesian smoothing step refines boundaries and removes spurious noise by leveraging nearby pixel information. We validate the framework on the April 2024 Dubai storm event, using pre- and post-rainfall PlanetScope composites. Experimental results demonstrate average F1-score improvements of up to 29% across all land-cover classes and notably sharper flood delineations, significantly outperforming conventional single-stage U-Net baselines.
>
---
#### [new 108] PIVOTS: Aligning unseen Structures using Preoperative to Intraoperative Volume-To-Surface Registration for Liver Navigation
- **分类: cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决术中肝脏形变导致的导航难题。通过提出PIVOTS神经网络，实现从术前体积数据到术中表面数据的非刚性配准，提升肝手术导航精度。方法包括多分辨率特征提取与跨模态注意力模块，并在合成与真实数据上验证有效性。**

- **链接: [http://arxiv.org/pdf/2507.20337v1](http://arxiv.org/pdf/2507.20337v1)**

> **作者:** Peng Liu; Bianca Güttner; Yutong Su; Chenyang Li; Jinjing Xu; Mingyang Liu; Zhe Min; Andrey Zhylka; Jasper Smit; Karin Olthof; Matteo Fusaglia; Rudi Apolle; Matthias Miederer; Laura Frohneberger; Carina Riediger; Jügen Weitz; Fiona Kolbinger; Stefanie Speidel; Micha Pfeiffer
>
> **摘要:** Non-rigid registration is essential for Augmented Reality guided laparoscopic liver surgery by fusing preoperative information, such as tumor location and vascular structures, into the limited intraoperative view, thereby enhancing surgical navigation. A prerequisite is the accurate prediction of intraoperative liver deformation which remains highly challenging due to factors such as large deformation caused by pneumoperitoneum, respiration and tool interaction as well as noisy intraoperative data, and limited field of view due to occlusion and constrained camera movement. To address these challenges, we introduce PIVOTS, a Preoperative to Intraoperative VOlume-To-Surface registration neural network that directly takes point clouds as input for deformation prediction. The geometric feature extraction encoder allows multi-resolution feature extraction, and the decoder, comprising novel deformation aware cross attention modules, enables pre- and intraoperative information interaction and accurate multi-level displacement prediction. We train the neural network on synthetic data simulated from a biomechanical simulation pipeline and validate its performance on both synthetic and real datasets. Results demonstrate superior registration performance of our method compared to baseline methods, exhibiting strong robustness against high amounts of noise, large deformation, and various levels of intraoperative visibility. We publish the training and test sets as evaluation benchmarks and call for a fair comparison of liver registration methods with volume-to-surface data. Code and datasets are available here https://github.com/pengliu-nct/PIVOTS.
>
---
#### [new 109] GT-Mean Loss: A Simple Yet Effective Solution for Brightness Mismatch in Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决增强图像与真实图像间亮度不一致的问题。作者提出了一种新的损失函数GT-Mean Loss，通过建模图像均值来缓解亮度差异，提升模型训练效果。**

- **链接: [http://arxiv.org/pdf/2507.20148v1](http://arxiv.org/pdf/2507.20148v1)**

> **作者:** Jingxi Liao; Shijie Hao; Richang Hong; Meng Wang
>
> **备注:** Accepted to ICCV2025. GitHub repository: https://github.com/jingxiLiao/GT-mean-loss
>
> **摘要:** Low-light image enhancement (LLIE) aims to improve the visual quality of images captured under poor lighting conditions. In supervised LLIE research, there exists a significant yet often overlooked inconsistency between the overall brightness of an enhanced image and its ground truth counterpart, referred to as brightness mismatch in this study. Brightness mismatch negatively impact supervised LLIE models by misleading model training. However, this issue is largely neglected in current research. In this context, we propose the GT-mean loss, a simple yet effective loss function directly modeling the mean values of images from a probabilistic perspective. The GT-mean loss is flexible, as it extends existing supervised LLIE loss functions into the GT-mean form with minimal additional computational costs. Extensive experiments demonstrate that the incorporation of the GT-mean loss results in consistent performance improvements across various methods and datasets.
>
---
#### [new 110] UniCT Depth: Event-Image Fusion Based Monocular Depth Estimation with Convolution-Compensated ViT Dual SA Block
- **分类: cs.CV**

- **简介: 论文属于单目深度估计任务，旨在解决图像与事件相机数据融合中的深度估计问题。现有方法在局部特征、跨模态融合与细节恢复上存在不足。论文提出UniCT Depth，结合CNN与Transformer建模局部与全局特征；设计CcViT-DA模块，包含CMSA与MFSA，实现空间依赖与跨模态融合；引入DCC模块增强纹理与边缘细节，提升深度估计精度。**

- **链接: [http://arxiv.org/pdf/2507.19948v1](http://arxiv.org/pdf/2507.19948v1)**

> **作者:** Luoxi Jing; Dianxi Shi; Zhe Liu; Songchang Jin; Chunping Qiu; Ziteng Qiao; Yuxian Li; Jianqiang Xia
>
> **备注:** Accepted by IJCAI 2025 (International Joint Conference on Artificial Intelligence)
>
> **摘要:** Depth estimation plays a crucial role in 3D scene understanding and is extensively used in a wide range of vision tasks. Image-based methods struggle in challenging scenarios, while event cameras offer high dynamic range and temporal resolution but face difficulties with sparse data. Combining event and image data provides significant advantages, yet effective integration remains challenging. Existing CNN-based fusion methods struggle with occlusions and depth disparities due to limited receptive fields, while Transformer-based fusion methods often lack deep modality interaction. To address these issues, we propose UniCT Depth, an event-image fusion method that unifies CNNs and Transformers to model local and global features. We propose the Convolution-compensated ViT Dual SA (CcViT-DA) Block, designed for the encoder, which integrates Context Modeling Self-Attention (CMSA) to capture spatial dependencies and Modal Fusion Self-Attention (MFSA) for effective cross-modal fusion. Furthermore, we design the tailored Detail Compensation Convolution (DCC) Block to improve texture details and enhances edge representations. Experiments show that UniCT Depth outperforms existing image, event, and fusion-based monocular depth estimation methods across key metrics.
>
---
#### [new 111] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting
- **分类: cs.CV**

- **简介: 该论文属于CLIP模型的少样本分类任务，旨在解决现有少样本评估不准确的问题。作者提出一种基于遗忘技术的新评估方法，发现现有方法性能显著下降，并提出了新的基准和改进技术。**

- **链接: [http://arxiv.org/pdf/2507.20834v1](http://arxiv.org/pdf/2507.20834v1)**

> **作者:** Alexey Kravets; Da Chen; Vinay P. Namboodiri
>
> **摘要:** CLIP is a foundational model with transferable classification performance in the few-shot setting. Several methods have shown improved performance of CLIP using few-shot examples. However, so far, all these techniques have been benchmarked using standard few-shot datasets. We argue that this mode of evaluation does not provide a true indication of the inductive generalization ability using few-shot examples. As most datasets have been seen by the CLIP model, the resultant setting can be termed as partially transductive. To solve this, we propose a pipeline that uses an unlearning technique to obtain true inductive baselines. In this new inductive setting, the methods show a significant drop in performance (-55% on average among 13 baselines with multiple datasets). We validate the unlearning technique using oracle baselines. An improved few-shot classification technique is proposed that consistently obtains state-of-the-art performance over 13 other recent baseline methods on a comprehensive analysis with 5880 experiments - varying the datasets, differing number of few-shot examples, unlearning setting, and with different seeds. Thus, we identify the issue with the evaluation of CLIP-based few-shot classification, provide a solution using unlearning, propose new benchmarks, and provide an improved method.
>
---
#### [new 112] Decomposing Densification in Gaussian Splatting for Faster 3D Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在解决高斯泼溅（GS）训练收敛慢的问题。通过分析分裂与克隆操作，提出全局到局部的致密化策略、能量引导的多分辨率训练框架及动态剪枝方法，加快训练速度并提升重建质量。**

- **链接: [http://arxiv.org/pdf/2507.20239v1](http://arxiv.org/pdf/2507.20239v1)**

> **作者:** Binxiao Huang; Zhengwu Liu; Ngai Wong
>
> **摘要:** 3D Gaussian Splatting (GS) has emerged as a powerful representation for high-quality scene reconstruction, offering compelling rendering quality. However, the training process of GS often suffers from slow convergence due to inefficient densification and suboptimal spatial distribution of Gaussian primitives. In this work, we present a comprehensive analysis of the split and clone operations during the densification phase, revealing their distinct roles in balancing detail preservation and computational efficiency. Building upon this analysis, we propose a global-to-local densification strategy, which facilitates more efficient growth of Gaussians across the scene space, promoting both global coverage and local refinement. To cooperate with the proposed densification strategy and promote sufficient diffusion of Gaussian primitives in space, we introduce an energy-guided coarse-to-fine multi-resolution training framework, which gradually increases resolution based on energy density in 2D images. Additionally, we dynamically prune unnecessary Gaussian primitives to speed up the training. Extensive experiments on MipNeRF-360, Deep Blending, and Tanks & Temples datasets demonstrate that our approach significantly accelerates training,achieving over 2x speedup with fewer Gaussian primitives and superior reconstruction performance.
>
---
#### [new 113] SCANet: Split Coordinate Attention Network for Building Footprint Extraction
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分析任务，旨在解决建筑物轮廓提取问题。作者提出了一种新的注意力模块SCA，通过捕捉空间远程交互，提升特征提取效率。将SCA嵌入2D CNN中构建SCANet，在多个数据集上超越现有方法，取得更高IoU指标。**

- **链接: [http://arxiv.org/pdf/2507.20809v1](http://arxiv.org/pdf/2507.20809v1)**

> **作者:** Chunshi Wang; Bin Zhao; Shuxue Ding
>
> **备注:** Accepted by ICONIP'24
>
> **摘要:** Building footprint extraction holds immense significance in remote sensing image analysis and has great value in urban planning, land use, environmental protection and disaster assessment. Despite the progress made by conventional and deep learning approaches in this field, they continue to encounter significant challenges. This paper introduces a novel plug-and-play attention module, Split Coordinate Attention (SCA), which ingeniously captures spatially remote interactions by employing two spatial range of pooling kernels, strategically encoding each channel along x and y planes, and separately performs a series of split operations for each feature group, thus enabling more efficient semantic feature extraction. By inserting into a 2D CNN to form an effective SCANet, our SCANet outperforms recent SOTA methods on the public Wuhan University (WHU) Building Dataset and Massachusetts Building Dataset in terms of various metrics. Particularly SCANet achieves the best IoU, 91.61% and 75.49% for the two datasets. Our code is available at https://github.com/AiEson/SCANet
>
---
#### [new 114] Color histogram equalization and fine-tuning to improve expression recognition of (partially occluded) faces on sign language datasets
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于表情识别任务，旨在提升手语数据集中面部表情的识别准确率，尤其针对部分遮挡情况。论文提出结合颜色直方图均衡化与微调的方法，进行颜色归一化处理，并分析上下半脸的表情识别效果。实验表明该方法具有较高敏感度和稳定性。**

- **链接: [http://arxiv.org/pdf/2507.20197v1](http://arxiv.org/pdf/2507.20197v1)**

> **作者:** Fabrizio Nunnari; Alakshendra Jyotsnaditya Ramkrishna Singh; Patrick Gebhard
>
> **摘要:** The goal of this investigation is to quantify to what extent computer vision methods can correctly classify facial expressions on a sign language dataset. We extend our experiments by recognizing expressions using only the upper or lower part of the face, which is needed to further investigate the difference in emotion manifestation between hearing and deaf subjects. To take into account the peculiar color profile of a dataset, our method introduces a color normalization stage based on histogram equalization and fine-tuning. The results show the ability to correctly recognize facial expressions with 83.8% mean sensitivity and very little variance (.042) among classes. Like for humans, recognition of expressions from the lower half of the face (79.6%) is higher than that from the upper half (77.9%). Noticeably, the classification accuracy from the upper half of the face is higher than human level.
>
---
#### [new 115] OW-CLIP: Data-Efficient Visual Supervision for Open-World Object Detection via Human-AI Collaboration
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于开放世界目标检测（OWOD）任务，旨在解决传统方法依赖大量标注、易过拟合及模型修改复杂等问题。作者提出OW-CLIP系统，结合多模态提示调优与“Crop-Smoothing”技术，并利用大语言模型与跨模态相似性优化数据生成与过滤，配合可视化界面提升标注效率与质量，实现高效增量训练。**

- **链接: [http://arxiv.org/pdf/2507.19870v1](http://arxiv.org/pdf/2507.19870v1)**

> **作者:** Junwen Duan; Wei Xue; Ziyao Kang; Shixia Liu; Jiazhi Xia
>
> **备注:** 9 pages, 11 figures
>
> **摘要:** Open-world object detection (OWOD) extends traditional object detection to identifying both known and unknown object, necessitating continuous model adaptation as new annotations emerge. Current approaches face significant limitations: 1) data-hungry training due to reliance on a large number of crowdsourced annotations, 2) susceptibility to "partial feature overfitting," and 3) limited flexibility due to required model architecture modifications. To tackle these issues, we present OW-CLIP, a visual analytics system that provides curated data and enables data-efficient OWOD model incremental training. OW-CLIP implements plug-and-play multimodal prompt tuning tailored for OWOD settings and introduces a novel "Crop-Smoothing" technique to mitigate partial feature overfitting. To meet the data requirements for the training methodology, we propose dual-modal data refinement methods that leverage large language models and cross-modal similarity for data generation and filtering. Simultaneously, we develope a visualization interface that enables users to explore and deliver high-quality annotations: including class-specific visual feature phrases and fine-grained differentiated images. Quantitative evaluation demonstrates that OW-CLIP achieves competitive performance at 89% of state-of-the-art performance while requiring only 3.8% self-generated data, while outperforming SOTA approach when trained with equivalent data volumes. A case study shows the effectiveness of the developed method and the improved annotation quality of our visualization system.
>
---
#### [new 116] Wavelet-guided Misalignment-aware Network for Visible-Infrared Object Detection
- **分类: cs.CV**

- **简介: 论文属于可见光-红外目标检测任务，旨在解决因分辨率差异、空间位移和模态不一致导致的跨模态图像对齐问题。作者提出WMNet，结合小波多频分析和模态感知融合机制，自适应处理不同对齐模式，提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.20146v1](http://arxiv.org/pdf/2507.20146v1)**

> **作者:** Haote Zhang; Lipeng Gu; Wuzhou Quan; Fu Lee Wang; Honghui Fan; Jiali Tang; Dingkun Zhu; Haoran Xie; Xiaoping Zhang; Mingqiang Wei
>
> **摘要:** Visible-infrared object detection aims to enhance the detection robustness by exploiting the complementary information of visible and infrared image pairs. However, its performance is often limited by frequent misalignments caused by resolution disparities, spatial displacements, and modality inconsistencies. To address this issue, we propose the Wavelet-guided Misalignment-aware Network (WMNet), a unified framework designed to adaptively address different cross-modal misalignment patterns. WMNet incorporates wavelet-based multi-frequency analysis and modality-aware fusion mechanisms to improve the alignment and integration of cross-modal features. By jointly exploiting low and high-frequency information and introducing adaptive guidance across modalities, WMNet alleviates the adverse effects of noise, illumination variation, and spatial misalignment. Furthermore, it enhances the representation of salient target features while suppressing spurious or misleading information, thereby promoting more accurate and robust detection. Extensive evaluations on the DVTOD, DroneVehicle, and M3FD datasets demonstrate that WMNet achieves state-of-the-art performance on misaligned cross-modal object detection tasks, confirming its effectiveness and practical applicability.
>
---
#### [new 117] Mask-Free Audio-driven Talking Face Generation for Enhanced Visual Quality and Identity Preservation
- **分类: cs.CV**

- **简介: 该论文属于音频驱动的说话人脸生成任务，旨在生成与音频同步且保留身份特征的高质量视频。现有方法依赖遮罩输入人脸下半部分并使用身份参考图，导致信息丢失、参考图与输入差异影响效果等问题。本文提出无需遮罩和参考图的方法，通过无配对训练的两步地标法闭合嘴巴，再结合音频生成自然唇动，提升了视觉质量和身份保持能力。**

- **链接: [http://arxiv.org/pdf/2507.20953v1](http://arxiv.org/pdf/2507.20953v1)**

> **作者:** Dogucan Yaman; Fevziye Irem Eyiokur; Leonard Bärmann; Hazım Kemal Ekenel; Alexander Waibel
>
> **摘要:** Audio-Driven Talking Face Generation aims at generating realistic videos of talking faces, focusing on accurate audio-lip synchronization without deteriorating any identity-related visual details. Recent state-of-the-art methods are based on inpainting, meaning that the lower half of the input face is masked, and the model fills the masked region by generating lips aligned with the given audio. Hence, to preserve identity-related visual details from the lower half, these approaches additionally require an unmasked identity reference image randomly selected from the same video. However, this common masking strategy suffers from (1) information loss in the input faces, significantly affecting the networks' ability to preserve visual quality and identity details, (2) variation between identity reference and input image degrading reconstruction performance, and (3) the identity reference negatively impacting the model, causing unintended copying of elements unaligned with the audio. To address these issues, we propose a mask-free talking face generation approach while maintaining the 2D-based face editing task. Instead of masking the lower half, we transform the input images to have closed mouths, using a two-step landmark-based approach trained in an unpaired manner. Subsequently, we provide these edited but unmasked faces to a lip adaptation model alongside the audio to generate appropriate lip movements. Thus, our approach needs neither masked input images nor identity reference images. We conduct experiments on the benchmark LRS2 and HDTF datasets and perform various ablation studies to validate our contributions.
>
---
#### [new 118] SynPAIN: A Synthetic Dataset of Pain and Non-Pain Facial Expressions
- **分类: cs.CV**

- **简介: 该论文属于医疗图像数据集构建任务，旨在解决现有疼痛检测数据集种族、年龄多样性不足及隐私问题。作者利用生成式AI创建了大规模合成面部表情数据集SynPAIN，包含不同种族、年龄和性别的疼痛与非疼痛表情图像，并验证其临床有效性。实验表明该数据集可提升模型性能并揭示算法偏见，特别适用于老年人疼痛检测。**

- **链接: [http://arxiv.org/pdf/2507.19673v1](http://arxiv.org/pdf/2507.19673v1)**

> **作者:** Babak Taati; Muhammad Muzammil; Yasamin Zarghami; Abhishek Moturu; Airhossein Kazerouni; Hailey Reimer; Alex Mihailidis; Thomas Hadjistavropoulos
>
> **备注:** 10 pages, 4 figures, submitted to IEEE JBHI
>
> **摘要:** Accurate pain assessment in patients with limited ability to communicate, such as older adults with dementia, represents a critical healthcare challenge. Robust automated systems of pain detection may facilitate such assessments. Existing pain detection datasets, however, suffer from limited ethnic/racial diversity, privacy constraints, and underrepresentation of older adults who are the primary target population for clinical deployment. We present SynPAIN, a large-scale synthetic dataset containing 10,710 facial expression images (5,355 neutral/expressive pairs) across five ethnicities/races, two age groups (young: 20-35, old: 75+), and two genders. Using commercial generative AI tools, we created demographically balanced synthetic identities with clinically meaningful pain expressions. Our validation demonstrates that synthetic pain expressions exhibit expected pain patterns, scoring significantly higher than neutral and non-pain expressions using clinically validated pain assessment tools based on facial action unit analysis. We experimentally demonstrate SynPAIN's utility in identifying algorithmic bias in existing pain detection models. Through comprehensive bias evaluation, we reveal substantial performance disparities across demographic characteristics. These performance disparities were previously undetectable with smaller, less diverse datasets. Furthermore, we demonstrate that age-matched synthetic data augmentation improves pain detection performance on real clinical data, achieving a 7.0% improvement in average precision. SynPAIN addresses critical gaps in pain assessment research by providing the first publicly available, demographically diverse synthetic dataset specifically designed for older adult pain detection, while establishing a framework for measuring and mitigating algorithmic bias. The dataset is available at https://doi.org/10.5683/SP3/WCXMAP
>
---
#### [new 119] Solving Scene Understanding for Autonomous Navigation in Unstructured Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于语义分割任务，旨在解决自动驾驶在非结构化环境中的场景理解问题。使用印度驾驶数据集，对五种深度学习模型进行训练和比较，以提高道路和周围环境的理解能力，从而提升自动驾驶的性能。**

- **链接: [http://arxiv.org/pdf/2507.20389v1](http://arxiv.org/pdf/2507.20389v1)**

> **作者:** Naveen Mathews Renji; Kruthika K; Manasa Keshavamurthy; Pooja Kumari; S. Rajarajeswari
>
> **摘要:** Autonomous vehicles are the next revolution in the automobile industry and they are expected to revolutionize the future of transportation. Understanding the scenario in which the autonomous vehicle will operate is critical for its competent functioning. Deep Learning has played a massive role in the progress that has been made till date. Semantic Segmentation, the process of annotating every pixel of an image with an object class, is one crucial part of this scene comprehension using Deep Learning. It is especially useful in Autonomous Driving Research as it requires comprehension of drivable and non-drivable areas, roadside objects and the like. In this paper semantic segmentation has been performed on the Indian Driving Dataset which has been recently compiled on the urban and rural roads of Bengaluru and Hyderabad. This dataset is more challenging compared to other datasets like Cityscapes, since it is based on unstructured driving environments. It has a four level hierarchy and in this paper segmentation has been performed on the first level. Five different models have been trained and their performance has been compared using the Mean Intersection over Union. These are UNET, UNET+RESNET50, DeepLabsV3, PSPNet and SegNet. The highest MIOU of 0.6496 has been achieved. The paper discusses the dataset, exploratory data analysis, preparation, implementation of the five models and studies the performance and compares the results achieved in the process.
>
---
#### [new 120] Smaller, Faster, Cheaper: Architectural Designs for Efficient Machine Learning
- **分类: cs.CV; cs.AR; cs.LG**

- **简介: 该论文属于机器学习模型架构设计任务，旨在解决模型性能与计算资源消耗之间的矛盾。论文通过优化数据传输、改进视觉Transformer的注意力机制及利用Normalizing Flows结构，提升模型效率，使其更小、更快、更省资源。**

- **链接: [http://arxiv.org/pdf/2507.19795v1](http://arxiv.org/pdf/2507.19795v1)**

> **作者:** Steven Walton
>
> **备注:** Ph.D. Thesis
>
> **摘要:** Major advancements in the capabilities of computer vision models have been primarily fueled by rapid expansion of datasets, model parameters, and computational budgets, leading to ever-increasing demands on computational infrastructure. However, as these models are deployed in increasingly diverse and resource-constrained environments, there is a pressing need for architectures that can deliver high performance while requiring fewer computational resources. This dissertation focuses on architectural principles through which models can achieve increased performance while reducing their computational demands. We discuss strides towards this goal through three directions. First, we focus on data ingress and egress, investigating how information may be passed into and retrieved from our core neural processing units. This ensures that our models make the most of available data, allowing smaller architectures to become more performant. Second, we investigate modifications to the core neural architecture, applied to restricted attention in vision transformers. This section explores how removing uniform context windows in restricted attention increases the expressivity of the underlying neural architecture. Third, we explore the natural structures of Normalizing Flows and how we can leverage these properties to better distill model knowledge. These contributions demonstrate that careful design of neural architectures can increase the efficiency of machine learning algorithms, allowing them to become smaller, faster, and cheaper.
>
---
#### [new 121] FaRMamba: Frequency-based learning and Reconstruction aided Mamba for Medical Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决病灶边界模糊、高频细节丢失和长距离结构建模困难的问题。论文提出FaRMamba模型，引入多尺度频率变换模块和自监督重建辅助编码器，有效恢复高频信息和二维空间结构，提升分割精度与细节保留，优于现有CNN-Transformer混合模型和Mamba变体。**

- **链接: [http://arxiv.org/pdf/2507.20056v1](http://arxiv.org/pdf/2507.20056v1)**

> **作者:** Ze Rong; ZiYue Zhao; Zhaoxin Wang; Lei Ma
>
> **摘要:** Accurate medical image segmentation remains challenging due to blurred lesion boundaries (LBA), loss of high-frequency details (LHD), and difficulty in modeling long-range anatomical structures (DC-LRSS). Vision Mamba employs one-dimensional causal state-space recurrence to efficiently model global dependencies, thereby substantially mitigating DC-LRSS. However, its patch tokenization and 1D serialization disrupt local pixel adjacency and impose a low-pass filtering effect, resulting in Local High-frequency Information Capture Deficiency (LHICD) and two-dimensional Spatial Structure Degradation (2D-SSD), which in turn exacerbate LBA and LHD. In this work, we propose FaRMamba, a novel extension that explicitly addresses LHICD and 2D-SSD through two complementary modules. A Multi-Scale Frequency Transform Module (MSFM) restores attenuated high-frequency cues by isolating and reconstructing multi-band spectra via wavelet, cosine, and Fourier transforms. A Self-Supervised Reconstruction Auxiliary Encoder (SSRAE) enforces pixel-level reconstruction on the shared Mamba encoder to recover full 2D spatial correlations, enhancing both fine textures and global context. Extensive evaluations on CAMUS echocardiography, MRI-based Mouse-cochlea, and Kvasir-Seg endoscopy demonstrate that FaRMamba consistently outperforms competitive CNN-Transformer hybrids and existing Mamba variants, delivering superior boundary accuracy, detail preservation, and global coherence without prohibitive computational overhead. This work provides a flexible frequency-aware framework for future segmentation models that directly mitigates core challenges in medical imaging.
>
---
#### [new 122] LAVA: Language Driven Scalable and Versatile Traffic Video Analytics
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频分析任务，旨在解决大规模交通视频中灵活高效查询的问题。现有方法受限于预定义语义类别，灵活性不足。论文提出LAVA系统，采用自然语言驱动的分析范式，结合采样、检测和轨迹提取技术，实现多粒度、任意类别的交通目标检索，显著提升了查询效果与效率。**

- **链接: [http://arxiv.org/pdf/2507.19821v1](http://arxiv.org/pdf/2507.19821v1)**

> **作者:** Yanrui Yu; Tianfei Zhou; Jiaxin Sun; Lianpeng Qiao; Lizhong Ding; Ye Yuan; Guoren Wang
>
> **摘要:** In modern urban environments, camera networks generate massive amounts of operational footage -- reaching petabytes each day -- making scalable video analytics essential for efficient processing. Many existing approaches adopt an SQL-based paradigm for querying such large-scale video databases; however, this constrains queries to rigid patterns with predefined semantic categories, significantly limiting analytical flexibility. In this work, we explore a language-driven video analytics paradigm aimed at enabling flexible and efficient querying of high-volume video data driven by natural language. Particularly, we build \textsc{Lava}, a system that accepts natural language queries and retrieves traffic targets across multiple levels of granularity and arbitrary categories. \textsc{Lava} comprises three main components: 1) a multi-armed bandit-based efficient sampling method for video segment-level localization; 2) a video-specific open-world detection module for object-level retrieval; and 3) a long-term object trajectory extraction scheme for temporal object association, yielding complete trajectories for object-of-interests. To support comprehensive evaluation, we further develop a novel benchmark by providing diverse, semantically rich natural language predicates and fine-grained annotations for multiple videos. Experiments on this benchmark demonstrate that \textsc{Lava} improves $F_1$-scores for selection queries by $\mathbf{14\%}$, reduces MPAE for aggregation queries by $\mathbf{0.39}$, and achieves top-$k$ precision of $\mathbf{86\%}$, while processing videos $ \mathbf{9.6\times} $ faster than the most accurate baseline.
>
---
#### [new 123] LSFDNet: A Single-Stage Fusion and Detection Network for Ships Using SWIR and LWIR
- **分类: cs.CV**

- **简介: 该论文属于图像融合与船舶检测任务，旨在解决复杂环境下单模态检测方法效果差的问题。作者提出LSFDNet，通过多级交叉融合模块提升红外图像融合效果，并引入对象增强损失函数优化检测性能，同时发布新数据集NSLSR支持相关研究。**

- **链接: [http://arxiv.org/pdf/2507.20574v1](http://arxiv.org/pdf/2507.20574v1)**

> **作者:** Yanyin Guo; Runxuan An; Junwei Li; Zhiyuan Zhang
>
> **备注:** ACMMM2025
>
> **摘要:** Traditional ship detection methods primarily rely on single-modal approaches, such as visible or infrared images, which limit their application in complex scenarios involving varying lighting conditions and heavy fog. To address this issue, we explore the advantages of short-wave infrared (SWIR) and long-wave infrared (LWIR) in ship detection and propose a novel single-stage image fusion detection algorithm called LSFDNet. This algorithm leverages feature interaction between the image fusion and object detection subtask networks, achieving remarkable detection performance and generating visually impressive fused images. To further improve the saliency of objects in the fused images and improve the performance of the downstream detection task, we introduce the Multi-Level Cross-Fusion (MLCF) module. This module combines object-sensitive fused features from the detection task and aggregates features across multiple modalities, scales, and tasks to obtain more semantically rich fused features. Moreover, we utilize the position prior from the detection task in the Object Enhancement (OE) loss function, further increasing the retention of object semantics in the fused images. The detection task also utilizes preliminary fused features from the fusion task to complement SWIR and LWIR features, thereby enhancing detection performance. Additionally, we have established a Nearshore Ship Long-Short Wave Registration (NSLSR) dataset to train effective SWIR and LWIR image fusion and detection networks, bridging a gap in this field. We validated the superiority of our proposed single-stage fusion detection algorithm on two datasets. The source code and dataset are available at https://github.com/Yanyin-Guo/LSFDNet
>
---
#### [new 124] Annotation-Free Human Sketch Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，旨在解决无需人工标注的素描质量评价问题。作者提出了GACL方法，通过特征幅度作为质量指标，结合可识别性学习，实现无监督的质量评估。通过大规模人类实验验证了方法有效性，并拓展至自然图像质量评估及噪声标签清洗等应用。**

- **链接: [http://arxiv.org/pdf/2507.20548v1](http://arxiv.org/pdf/2507.20548v1)**

> **作者:** Lan Yang; Kaiyue Pang; Honggang Zhang; Yi-Zhe Song
>
> **备注:** Accepted by IJCV
>
> **摘要:** As lovely as bunnies are, your sketched version would probably not do them justice (Fig.~\ref{fig:intro}). This paper recognises this very problem and studies sketch quality assessment for the first time -- letting you find these badly drawn ones. Our key discovery lies in exploiting the magnitude ($L_2$ norm) of a sketch feature as a quantitative quality metric. We propose Geometry-Aware Classification Layer (GACL), a generic method that makes feature-magnitude-as-quality-metric possible and importantly does it without the need for specific quality annotations from humans. GACL sees feature magnitude and recognisability learning as a dual task, which can be simultaneously optimised under a neat cross-entropy classification loss with theoretic guarantee. This gives GACL a nice geometric interpretation (the better the quality, the easier the recognition), and makes it agnostic to both network architecture changes and the underlying sketch representation. Through a large scale human study of 160,000 \doublecheck{trials}, we confirm the agreement between our GACL-induced metric and human quality perception. We further demonstrate how such a quality assessment capability can for the first time enable three practical sketch applications. Interestingly, we show GACL not only works on abstract visual representations such as sketch but also extends well to natural images on the problem of image quality assessment (IQA). Last but not least, we spell out the general properties of GACL as general-purpose data re-weighting strategy and demonstrate its applications in vertical problems such as noisy label cleansing. Code will be made publicly available at github.com/yanglan0225/SketchX-Quantifying-Sketch-Quality.
>
---
#### [new 125] Fine-structure Preserved Real-world Image Super-resolution via Transfer VAE Training
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决现有方法在恢复图像细节（如小字符、纹理）上的不足。通过提出Transfer VAE Training（TVT）策略，将8×下采样VAE转为4×，并在保持UNet预训练权重的同时提升细节重建能力。此外，优化网络结构以降低计算成本。**

- **链接: [http://arxiv.org/pdf/2507.20291v1](http://arxiv.org/pdf/2507.20291v1)**

> **作者:** Qiaosi Yi; Shuai Li; Rongyuan Wu; Lingchen Sun; Yuhui Wu; Lei Zhang
>
> **备注:** ICCV 2025
>
> **摘要:** Impressive results on real-world image super-resolution (Real-ISR) have been achieved by employing pre-trained stable diffusion (SD) models. However, one critical issue of such methods lies in their poor reconstruction of image fine structures, such as small characters and textures, due to the aggressive resolution reduction of the VAE (eg., 8$\times$ downsampling) in the SD model. One solution is to employ a VAE with a lower downsampling rate for diffusion; however, adapting its latent features with the pre-trained UNet while mitigating the increased computational cost poses new challenges. To address these issues, we propose a Transfer VAE Training (TVT) strategy to transfer the 8$\times$ downsampled VAE into a 4$\times$ one while adapting to the pre-trained UNet. Specifically, we first train a 4$\times$ decoder based on the output features of the original VAE encoder, then train a 4$\times$ encoder while keeping the newly trained decoder fixed. Such a TVT strategy aligns the new encoder-decoder pair with the original VAE latent space while enhancing image fine details. Additionally, we introduce a compact VAE and compute-efficient UNet by optimizing their network architectures, reducing the computational cost while capturing high-resolution fine-scale features. Experimental results demonstrate that our TVT method significantly improves fine-structure preservation, which is often compromised by other SD-based methods, while requiring fewer FLOPs than state-of-the-art one-step diffusion models. The official code can be found at https://github.com/Joyies/TVT.
>
---
#### [new 126] Multi-output Deep-Supervised Classifier Chains for Plant Pathology
- **分类: cs.CV**

- **简介: 该论文属于植物病理分类任务，旨在解决植物叶片疾病自动识别问题。现有方法未充分研究植物种类与疾病类型间的关系对分类性能的影响。论文提出了一种新的多输出深度监督分类链模型（Mo-DsCC），通过结合植物种类与病害的预测，提升分类准确率和F1分数。**

- **链接: [http://arxiv.org/pdf/2507.20125v1](http://arxiv.org/pdf/2507.20125v1)**

> **作者:** Jianping Yao; Son N. Tran
>
> **摘要:** Plant leaf disease classification is an important task in smart agriculture which plays a critical role in sustainable production. Modern machine learning approaches have shown unprecedented potential in this classification task which offers an array of benefits including time saving and cost reduction. However, most recent approaches directly employ convolutional neural networks where the effect of the relationship between plant species and disease types on prediction performance is not properly studied. In this study, we proposed a new model named Multi-output Deep Supervised Classifier Chains (Mo-DsCC) which weaves the prediction of plant species and disease by chaining the output layers for the two labels. Mo-DsCC consists of three components: A modified VGG-16 network as the backbone, deep supervision training, and a stack of classification chains. To evaluate the advantages of our model, we perform intensive experiments on two benchmark datasets Plant Village and PlantDoc. Comparison to recent approaches, including multi-model, multi-label (Power-set), multi-output and multi-task, demonstrates that Mo-DsCC achieves better accuracy and F1-score. The empirical study in this paper shows that the application of Mo-DsCC could be a useful puzzle for smart agriculture to benefit farms and bring new ideas to industry and academia.
>
---
#### [new 127] SCORPION: Addressing Scanner-Induced Variability in Histopathology
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算病理学中的模型泛化任务，旨在解决数字扫描仪差异导致的模型性能不稳定问题。作者构建了SCORPION数据集，包含2,400个配对图像块，用于评估扫描仪变化下的模型一致性，并提出SimCons方法提升模型跨扫描仪的稳定性，确保诊断结果可靠。**

- **链接: [http://arxiv.org/pdf/2507.20907v1](http://arxiv.org/pdf/2507.20907v1)**

> **作者:** Jeongun Ryu; Heon Song; Seungeun Lee; Soo Ick Cho; Jiwon Shin; Kyunghyun Paeng; Sérgio Pereira
>
> **备注:** Accepted in UNSURE 2025 workshop in MICCAI
>
> **摘要:** Ensuring reliable model performance across diverse domains is a critical challenge in computational pathology. A particular source of variability in Whole-Slide Images is introduced by differences in digital scanners, thus calling for better scanner generalization. This is critical for the real-world adoption of computational pathology, where the scanning devices may differ per institution or hospital, and the model should not be dependent on scanner-induced details, which can ultimately affect the patient's diagnosis and treatment planning. However, past efforts have primarily focused on standard domain generalization settings, evaluating on unseen scanners during training, without directly evaluating consistency across scanners for the same tissue. To overcome this limitation, we introduce SCORPION, a new dataset explicitly designed to evaluate model reliability under scanner variability. SCORPION includes 480 tissue samples, each scanned with 5 scanners, yielding 2,400 spatially aligned patches. This scanner-paired design allows for the isolation of scanner-induced variability, enabling a rigorous evaluation of model consistency while controlling for differences in tissue composition. Furthermore, we propose SimCons, a flexible framework that combines augmentation-based domain generalization techniques with a consistency loss to explicitly address scanner generalization. We empirically show that SimCons improves model consistency on varying scanners without compromising task-specific performance. By releasing the SCORPION dataset and proposing SimCons, we provide the research community with a crucial resource for evaluating and improving model consistency across diverse scanners, setting a new standard for reliability testing.
>
---
#### [new 128] Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在解决通过不透明容器表面振动推断其内部液体水平的问题。论文提出了一种基于散斑振动传感的系统，并结合Transformer模型对振动数据进行分析，实现了对封闭容器液体水平的远程非接触式检测。**

- **链接: [http://arxiv.org/pdf/2507.20757v1](http://arxiv.org/pdf/2507.20757v1)**

> **作者:** Matan Kichler; Shai Bagon; Mark Sheinin
>
> **备注:** ICCV 2025
>
> **摘要:** Computer vision seeks to infer a wide range of information about objects and events. However, vision systems based on conventional imaging are limited to extracting information only from the visible surfaces of scene objects. For instance, a vision system can detect and identify a Coke can in the scene, but it cannot determine whether the can is full or empty. In this paper, we aim to expand the scope of computer vision to include the novel task of inferring the hidden liquid levels of opaque containers by sensing the tiny vibrations on their surfaces. Our method provides a first-of-a-kind way to inspect the fill level of multiple sealed containers remotely, at once, without needing physical manipulation and manual weighing. First, we propose a novel speckle-based vibration sensing system for simultaneously capturing scene vibrations on a 2D grid of points. We use our system to efficiently and remotely capture a dataset of vibration responses for a variety of everyday liquid containers. Then, we develop a transformer-based approach for analyzing the captured vibrations and classifying the container type and its hidden liquid level at the time of measurement. Our architecture is invariant to the vibration source, yielding correct liquid level estimates for controlled and ambient scene sound sources. Moreover, our model generalizes to unseen container instances within known classes (e.g., training on five Coke cans of a six-pack, testing on a sixth) and fluid levels. We demonstrate our method by recovering liquid levels from various everyday containers.
>
---
#### [new 129] ARC-Hunyuan-Video-7B: Structured Video Comprehension of Real-World Shorts
- **分类: cs.CV**

- **简介: 该论文属于多模态视频理解任务，旨在解决现有模型对真实短视频缺乏细粒度、结构化理解的问题。作者提出了ARC-Hunyuan-Video-7B模型，通过端到端方式融合视觉、音频与文本信息，并经多阶段训练，实现视频描述、问答、推理等能力，在真实场景中提升了用户参与度与满意度。**

- **链接: [http://arxiv.org/pdf/2507.20939v1](http://arxiv.org/pdf/2507.20939v1)**

> **作者:** Yuying Ge; Yixiao Ge; Chen Li; Teng Wang; Junfu Pu; Yizhuo Li; Lu Qiu; Jin Ma; Lisheng Duan; Xinyu Zuo; Jinwen Luo; Weibo Gu; Zexuan Li; Xiaojing Zhang; Yangyu Tao; Han Hu; Di Wang; Ying Shan
>
> **备注:** Project Page: https://tencentarc.github.io/posts/arc-video-announcement/
>
> **摘要:** Real-world user-generated short videos, especially those distributed on platforms such as WeChat Channel and TikTok, dominate the mobile internet. However, current large multimodal models lack essential temporally-structured, detailed, and in-depth video comprehension capabilities, which are the cornerstone of effective video search and recommendation, as well as emerging video applications. Understanding real-world shorts is actually challenging due to their complex visual elements, high information density in both visuals and audio, and fast pacing that focuses on emotional expression and viewpoint delivery. This requires advanced reasoning to effectively integrate multimodal information, including visual, audio, and text. In this work, we introduce ARC-Hunyuan-Video, a multimodal model that processes visual, audio, and textual signals from raw video inputs end-to-end for structured comprehension. The model is capable of multi-granularity timestamped video captioning and summarization, open-ended video question answering, temporal video grounding, and video reasoning. Leveraging high-quality data from an automated annotation pipeline, our compact 7B-parameter model is trained through a comprehensive regimen: pre-training, instruction fine-tuning, cold start, reinforcement learning (RL) post-training, and final instruction fine-tuning. Quantitative evaluations on our introduced benchmark ShortVid-Bench and qualitative comparisons demonstrate its strong performance in real-world video comprehension, and it supports zero-shot or fine-tuning with a few samples for diverse downstream applications. The real-world production deployment of our model has yielded tangible and measurable improvements in user engagement and satisfaction, a success supported by its remarkable efficiency, with stress tests indicating an inference time of just 10 seconds for a one-minute video on H20 GPU.
>
---
#### [new 130] Efficient Self-Supervised Neuro-Analytic Visual Servoing for Real-time Quadrotor Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机视觉控制任务，旨在解决传统视觉伺服控制在计算效率和数值稳定性方面的不足。论文提出了一种自监督的神经解析控制方法，通过知识蒸馏将解析模型的能力转移到轻量级神经网络，实现高效实时的四旋翼无人机控制，无需显式几何模型或标记。**

- **链接: [http://arxiv.org/pdf/2507.19878v1](http://arxiv.org/pdf/2507.19878v1)**

> **作者:** Sebastian Mocanu; Sebastian-Ion Nae; Mihai-Eugen Barbu; Marius Leordeanu
>
> **备注:** Accepted at the International Conference on Computer Vision Workshops 2025
>
> **摘要:** This work introduces a self-supervised neuro-analytical, cost efficient, model for visual-based quadrotor control in which a small 1.7M parameters student ConvNet learns automatically from an analytical teacher, an improved image-based visual servoing (IBVS) controller. Our IBVS system solves numerical instabilities by reducing the classical visual servoing equations and enabling efficient stable image feature detection. Through knowledge distillation, the student model achieves 11x faster inference compared to the teacher IBVS pipeline, while demonstrating similar control accuracy at a significantly lower computational and memory cost. Our vision-only self-supervised neuro-analytic control, enables quadrotor orientation and movement without requiring explicit geometric models or fiducial markers. The proposed methodology leverages simulation-to-reality transfer learning and is validated on a small drone platform in GPS-denied indoor environments. Our key contributions include: (1) an analytical IBVS teacher that solves numerical instabilities inherent in classical approaches, (2) a two-stage segmentation pipeline combining YOLOv11 with a U-Net-based mask splitter for robust anterior-posterior vehicle segmentation to correctly estimate the orientation of the target, and (3) an efficient knowledge distillation dual-path system, which transfers geometric visual servoing capabilities from the analytical IBVS teacher to a compact and small student neural network that outperforms the teacher, while being suitable for real-time onboard deployment.
>
---
#### [new 131] Motion-example-controlled Co-speech Gesture Generation Leveraging Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于语音伴随手势生成任务，旨在解决现有方法难以保留运动示例细节的问题。作者提出MECo框架，利用大语言模型同时理解语音和运动示例，实现高质量、可控的手势生成。**

- **链接: [http://arxiv.org/pdf/2507.20220v1](http://arxiv.org/pdf/2507.20220v1)**

> **作者:** Bohong Chen; Yumeng Li; Youyi Zheng; Yao-Xiang Ding; Kun Zhou
>
> **备注:** SIGGRAPH 2025; Project Page: https://robinwitch.github.io/MECo-Page
>
> **摘要:** The automatic generation of controllable co-speech gestures has recently gained growing attention. While existing systems typically achieve gesture control through predefined categorical labels or implicit pseudo-labels derived from motion examples, these approaches often compromise the rich details present in the original motion examples. We present MECo, a framework for motion-example-controlled co-speech gesture generation by leveraging large language models (LLMs). Our method capitalizes on LLMs' comprehension capabilities through fine-tuning to simultaneously interpret speech audio and motion examples, enabling the synthesis of gestures that preserve example-specific characteristics while maintaining speech congruence. Departing from conventional pseudo-labeling paradigms, we position motion examples as explicit query contexts within the prompt structure to guide gesture generation. Experimental results demonstrate state-of-the-art performance across three metrics: Fr\'echet Gesture Distance (FGD), motion diversity, and example-gesture similarity. Furthermore, our framework enables granular control of individual body parts and accommodates diverse input modalities including motion clips, static poses, human video sequences, and textual descriptions. Our code, pre-trained models, and videos are available at https://robinwitch.github.io/MECo-Page.
>
---
#### [new 132] VAMPIRE: Uncovering Vessel Directional and Morphological Information from OCTA Images for Cardiovascular Disease Risk Factor Prediction
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析与心血管疾病预测任务，旨在解决现有视网膜影像预测CVD风险方法无法捕捉详细血管特征、仅进行二分类的问题。作者构建了首个用于CVD风险评估的OCTA数据集OCTA-CVD，并提出VAMPIRE模型，通过方向与形态模块提取血管细粒度特征，实现CVD风险及关联因素的联合预测。**

- **链接: [http://arxiv.org/pdf/2507.20017v1](http://arxiv.org/pdf/2507.20017v1)**

> **作者:** Lehan Wang; Hualiang Wang; Chubin Ou; Lushi Chen; Yunyi Liang; Xiaomeng Li
>
> **备注:** Accepted in MICCAI 2025
>
> **摘要:** Cardiovascular disease (CVD) remains the leading cause of death worldwide, requiring urgent development of effective risk assessment methods for timely intervention. While current research has introduced non-invasive and efficient approaches to predict CVD risk from retinal imaging with deep learning models, the commonly used fundus photographs and Optical Coherence Tomography (OCT) fail to capture detailed vascular features critical for CVD assessment compared with OCT angiography (OCTA) images. Moreover, existing methods typically classify CVD risk only as high or low, without providing a deeper analysis on CVD-related blood factor conditions, thus limiting prediction accuracy and clinical utility. As a result, we propose a novel multi-purpose paradigm of CVD risk assessment that jointly performs CVD risk and CVD-related condition prediction, aligning with clinical experiences. Based on this core idea, we introduce OCTA-CVD, the first OCTA dataset for CVD risk assessment, and a Vessel-Aware Mamba-based Prediction model with Informative Enhancement (VAMPIRE) based on OCTA enface images. Our proposed model aims to extract crucial vascular characteristics through two key components: (1) a Mamba-Based Directional (MBD) Module that captures fine-grained vascular trajectory features and (2) an Information-Enhanced Morphological (IEM) Module that incorporates comprehensive vessel morphology knowledge. Experimental results demonstrate that our method can surpass standard classification backbones, OCTA-based detection methods, and ophthalmologic foundation models. Our codes and the collected OCTA-CVD dataset are available at https://github.com/xmed-lab/VAMPIRE.
>
---
#### [new 133] AnimeColor: Reference-based Animation Colorization with Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于动画着色任务，旨在解决现有方法在颜色准确性与时间一致性上的不足。论文提出了AnimeColor框架，结合扩散变换模型，引入草图序列与参考图像，提高动画着色质量，优化生成效果。**

- **链接: [http://arxiv.org/pdf/2507.20158v1](http://arxiv.org/pdf/2507.20158v1)**

> **作者:** Yuhong Zhang; Liyao Wang; Han Wang; Danni Wu; Zuzeng Lin; Feng Wang; Li Song
>
> **摘要:** Animation colorization plays a vital role in animation production, yet existing methods struggle to achieve color accuracy and temporal consistency. To address these challenges, we propose \textbf{AnimeColor}, a novel reference-based animation colorization framework leveraging Diffusion Transformers (DiT). Our approach integrates sketch sequences into a DiT-based video diffusion model, enabling sketch-controlled animation generation. We introduce two key components: a High-level Color Extractor (HCE) to capture semantic color information and a Low-level Color Guider (LCG) to extract fine-grained color details from reference images. These components work synergistically to guide the video diffusion process. Additionally, we employ a multi-stage training strategy to maximize the utilization of reference image color information. Extensive experiments demonstrate that AnimeColor outperforms existing methods in color accuracy, sketch alignment, temporal consistency, and visual quality. Our framework not only advances the state of the art in animation colorization but also provides a practical solution for industrial applications. The code will be made publicly available at \href{https://github.com/IamCreateAI/AnimeColor}{https://github.com/IamCreateAI/AnimeColor}.
>
---
#### [new 134] AF-CLIP: Zero-Shot Anomaly Detection via Anomaly-Focused CLIP Adaptation
- **分类: cs.CV**

- **简介: 该论文属于视觉异常检测任务，旨在解决零样本/少样本下检测效果差的问题。通过改进CLIP模型，增强其对局部异常的关注能力，引入轻量适配器和多尺度空间聚合机制，并设计可学习文本提示。方法在多种工业和医学数据集上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2507.19949v1](http://arxiv.org/pdf/2507.19949v1)**

> **作者:** Qingqing Fang; Wenxi Lv; Qinliang Su
>
> **备注:** The paper is accepted by ACM MM' 25
>
> **摘要:** Visual anomaly detection has been widely used in industrial inspection and medical diagnosis. Existing methods typically demand substantial training samples, limiting their utility in zero-/few-shot scenarios. While recent efforts have leveraged CLIP's zero-shot recognition capability for this task, they often ignore optimizing visual features to focus on local anomalies, reducing their efficacy. In this work, we propose AF-CLIP (Anomaly-Focused CLIP) by dramatically enhancing its visual representations to focus on local defects. Our approach introduces a lightweight adapter that emphasizes anomaly-relevant patterns in visual features, simultaneously optimizing both class-level features for image classification and patch-level features for precise localization. To capture anomalies of different sizes and improve detection accuracy, prior to the adapter, we develop a multi-scale spatial aggregation mechanism to effectively consolidate neighborhood context. Complementing these visual enhancements, we design learnable textual prompts that generically characterize normal and abnormal states. After optimization on auxiliary datasets using a composite objective function, AF-CLIP demonstrates strong zero-shot detection capability. Our method is also extended to few-shot scenarios by extra memory banks. Experimental results across diverse industrial and medical datasets demonstrate the effectiveness and generalization of our proposed method. Code is available at https://github.com/Faustinaqq/AF-CLIP.
>
---
#### [new 135] HAMLET-FFD: Hierarchical Adaptive Multi-modal Learning Embeddings Transformation for Face Forgery Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像取证任务，旨在解决跨域人脸伪造检测泛化性差的问题。作者提出了HAMLET-FFD框架，通过双向多模态推理，结合视觉与文本线索，提升伪造检测效果。方法利用预训练模型，设计知识优化循环，动态融合多层级特征，实现细粒度伪造识别，且参数固定，便于集成。**

- **链接: [http://arxiv.org/pdf/2507.20913v1](http://arxiv.org/pdf/2507.20913v1)**

> **作者:** Jialei Cui; Jianwei Du; Yanzhe Li; Lei Gao; Hui Jiang; Chenfu Bao
>
> **摘要:** The rapid evolution of face manipulation techniques poses a critical challenge for face forgery detection: cross-domain generalization. Conventional methods, which rely on simple classification objectives, often fail to learn domain-invariant representations. We propose HAMLET-FFD, a cognitively inspired Hierarchical Adaptive Multi-modal Learning framework that tackles this challenge via bidirectional cross-modal reasoning. Building on contrastive vision-language models such as CLIP, HAMLET-FFD introduces a knowledge refinement loop that iteratively assesses authenticity by integrating visual evidence with conceptual cues, emulating expert forensic analysis. A key innovation is a bidirectional fusion mechanism in which textual authenticity embeddings guide the aggregation of hierarchical visual features, while modulated visual features refine text embeddings to generate image-adaptive prompts. This closed-loop process progressively aligns visual observations with semantic priors to enhance authenticity assessment. By design, HAMLET-FFD freezes all pretrained parameters, serving as an external plugin that preserves CLIP's original capabilities. Extensive experiments demonstrate its superior generalization to unseen manipulations across multiple benchmarks, and visual analyses reveal a division of labor among embeddings, with distinct representations specializing in fine-grained artifact recognition.
>
---
#### [new 136] Automated 3D-GS Registration and Fusion via Skeleton Alignment and Gaussian-Adaptive Features
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在解决多3D高斯点云（3D-GS）子地图的自动配准与融合问题。现有方法依赖手动选择参考图且融合质量差，本文提出基于骨架对齐和高斯自适应特征的方法，实现自动化处理，提升配准精度与融合效果，改善3D场景表示一致性与准确性。**

- **链接: [http://arxiv.org/pdf/2507.20480v1](http://arxiv.org/pdf/2507.20480v1)**

> **作者:** Shiyang Liu; Dianyi Yang; Yu Gao; Bohan Ren; Yi Yang; Mengyin Fu
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** In recent years, 3D Gaussian Splatting (3D-GS)-based scene representation demonstrates significant potential in real-time rendering and training efficiency. However, most existing methods primarily focus on single-map reconstruction, while the registration and fusion of multiple 3D-GS sub-maps remain underexplored. Existing methods typically rely on manual intervention to select a reference sub-map as a template and use point cloud matching for registration. Moreover, hard-threshold filtering of 3D-GS primitives often degrades rendering quality after fusion. In this paper, we present a novel approach for automated 3D-GS sub-map alignment and fusion, eliminating the need for manual intervention while enhancing registration accuracy and fusion quality. First, we extract geometric skeletons across multiple scenes and leverage ellipsoid-aware convolution to capture 3D-GS attributes, facilitating robust scene registration. Second, we introduce a multi-factor Gaussian fusion strategy to mitigate the scene element loss caused by rigid thresholding. Experiments on the ScanNet-GSReg and our Coord datasets demonstrate the effectiveness of the proposed method in registration and fusion. For registration, it achieves a 41.9\% reduction in RRE on complex scenes, ensuring more precise pose estimation. For fusion, it improves PSNR by 10.11 dB, highlighting superior structural preservation. These results confirm its ability to enhance scene alignment and reconstruction fidelity, ensuring more consistent and accurate 3D scene representation for robotic perception and autonomous navigation.
>
---
#### [new 137] JWB-DH-V1: Benchmark for Joint Whole-Body Talking Avatar and Speech Generation Version 1
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态生成任务，旨在解决全身体说话虚拟人与语音联合生成中视觉与音频一致性不足、缺乏综合评估框架的问题。作者构建了包含200万视频样本的大规模数据集JWB-DH-V1，并提出评估协议，揭示了现有模型在全身与局部生成上的性能差异。**

- **链接: [http://arxiv.org/pdf/2507.20987v1](http://arxiv.org/pdf/2507.20987v1)**

> **作者:** Xinhan Di; Kristin Qi; Pengqian Yu
>
> **备注:** WiCV @ ICCV 2025
>
> **摘要:** Recent advances in diffusion-based video generation have enabled photo-realistic short clips, but current methods still struggle to achieve multi-modal consistency when jointly generating whole-body motion and natural speech. Current approaches lack comprehensive evaluation frameworks that assess both visual and audio quality, and there are insufficient benchmarks for region-specific performance analysis. To address these gaps, we introduce the Joint Whole-Body Talking Avatar and Speech Generation Version I(JWB-DH-V1), comprising a large-scale multi-modal dataset with 10,000 unique identities across 2 million video samples, and an evaluation protocol for assessing joint audio-video generation of whole-body animatable avatars. Our evaluation of SOTA models reveals consistent performance disparities between face/hand-centric and whole-body performance, which incidates essential areas for future research. The dataset and evaluation tools are publicly available at https://github.com/deepreasonings/WholeBodyBenchmark.
>
---
#### [new 138] RIS-LAD: A Benchmark and Model for Referring Low-Altitude Drone Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言理解任务中的指向性图像分割（RIS）任务，旨在解决低空无人机（LAD）场景中因视角多样、物体密集带来的分割难题。作者构建了首个针对LAD的RIS数据集RIS-LAD，并提出SAARN模型，通过语义感知与自适应推理提升复杂场景的分割效果。**

- **链接: [http://arxiv.org/pdf/2507.20920v1](http://arxiv.org/pdf/2507.20920v1)**

> **作者:** Kai Ye; YingShi Luan; Zhudi Chen; Guangyue Meng; Pingyang Dai; Liujuan Cao
>
> **摘要:** Referring Image Segmentation (RIS), which aims to segment specific objects based on natural language descriptions, plays an essential role in vision-language understanding. Despite its progress in remote sensing applications, RIS in Low-Altitude Drone (LAD) scenarios remains underexplored. Existing datasets and methods are typically designed for high-altitude and static-view imagery. They struggle to handle the unique characteristics of LAD views, such as diverse viewpoints and high object density. To fill this gap, we present RIS-LAD, the first fine-grained RIS benchmark tailored for LAD scenarios. This dataset comprises 13,871 carefully annotated image-text-mask triplets collected from realistic drone footage, with a focus on small, cluttered, and multi-viewpoint scenes. It highlights new challenges absent in previous benchmarks, such as category drift caused by tiny objects and object drift under crowded same-class objects. To tackle these issues, we propose the Semantic-Aware Adaptive Reasoning Network (SAARN). Rather than uniformly injecting all linguistic features, SAARN decomposes and routes semantic information to different stages of the network. Specifically, the Category-Dominated Linguistic Enhancement (CDLE) aligns visual features with object categories during early encoding, while the Adaptive Reasoning Fusion Module (ARFM) dynamically selects semantic cues across scales to improve reasoning in complex scenes. The experimental evaluation reveals that RIS-LAD presents substantial challenges to state-of-the-art RIS algorithms, and also demonstrates the effectiveness of our proposed model in addressing these challenges. The dataset and code will be publicly released soon at: https://github.com/AHideoKuzeA/RIS-LAD/.
>
---
#### [new 139] Detection of Medial Epicondyle Avulsion in Elbow Ultrasound Images via Bone Structure Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在检测肘部超声图像中的内上髁撕脱。通过仅训练正常样本，提出一种基于掩码自编码器的结构感知重建框架，利用撕脱区域的重建误差进行异常检测。在16名棒球运动员的数据集上取得了良好效果。**

- **链接: [http://arxiv.org/pdf/2507.20104v1](http://arxiv.org/pdf/2507.20104v1)**

> **作者:** Shizuka Akahori; Shotaro Teruya; Pragyan Shrestha; Yuichi Yoshii; Satoshi Iizuka; Akira Ikumi; Hiromitsu Tsuge; Itaru Kitahara
>
> **备注:** 19th International Conference on Machine Vision Applications (MVA)
>
> **摘要:** This study proposes a reconstruction-based framework for detecting medial epicondyle avulsion in elbow ultrasound images, trained exclusively on normal cases. Medial epicondyle avulsion, commonly observed in baseball players, involves bone detachment and deformity, often appearing as discontinuities in bone contour. Therefore, learning the structure and continuity of normal bone is essential for detecting such abnormalities. To achieve this, we propose a masked autoencoder-based, structure-aware reconstruction framework that learns the continuity of normal bone structures. Even in the presence of avulsion, the model attempts to reconstruct the normal structure, resulting in large reconstruction errors at the avulsion site. For evaluation, we constructed a novel dataset comprising normal and avulsion ultrasound images from 16 baseball players, with pixel-level annotations under orthopedic supervision. Our method outperformed existing approaches, achieving a pixel-wise AUC of 0.965 and an image-wise AUC of 0.967. The dataset is publicly available at: https://github.com/Akahori000/Ultrasound-Medial-Epicondyle-Avulsion-Dataset.
>
---
#### [new 140] JDATT: A Joint Distillation Framework for Atmospheric Turbulence Mitigation and Target Detection
- **分类: cs.CV**

- **简介: 该论文属于图像处理与目标检测任务，旨在解决大气湍流导致的图像质量下降和检测性能恶化问题。论文提出JDATT框架，联合蒸馏图像恢复和目标检测模块，降低模型复杂度，提升实时性与效率。**

- **链接: [http://arxiv.org/pdf/2507.19780v1](http://arxiv.org/pdf/2507.19780v1)**

> **作者:** Zhiming Liu; Paul Hill; Nantheera Anantrasirichai
>
> **备注:** Accepted by the 36th British Machine Vision Conference
>
> **摘要:** Atmospheric turbulence (AT) introduces severe degradations, such as rippling, blur, and intensity fluctuations, that hinder both image quality and downstream vision tasks like target detection. While recent deep learning-based approaches have advanced AT mitigation using transformer and Mamba architectures, their high complexity and computational cost make them unsuitable for real-time applications, especially in resource-constrained settings such as remote surveillance. Moreover, the common practice of separating turbulence mitigation and object detection leads to inefficiencies and suboptimal performance. To address these challenges, we propose JDATT, a Joint Distillation framework for Atmospheric Turbulence mitigation and Target detection. JDATT integrates state-of-the-art AT mitigation and detection modules and introduces a unified knowledge distillation strategy that compresses both components while minimizing performance loss. We employ a hybrid distillation scheme: feature-level distillation via Channel-Wise Distillation (CWD) and Masked Generative Distillation (MGD), and output-level distillation via Kullback-Leibler divergence. Experiments on synthetic and real-world turbulence datasets demonstrate that JDATT achieves superior visual restoration and detection accuracy while significantly reducing model size and inference time, making it well-suited for real-time deployment.
>
---
#### [new 141] ATCTrack: Aligning Target-Context Cues with Dynamic Target States for Robust Vision-Language Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言跟踪任务，旨在解决复杂长时场景中目标状态变化导致的跟踪不稳定问题。作者提出ATCTrack方法，通过动态建模目标-上下文特征，有效融合视觉和文本信息，实现鲁棒跟踪，并在主流基准上达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2507.19875v1](http://arxiv.org/pdf/2507.19875v1)**

> **作者:** X. Feng; S. Hu; X. Li; D. Zhang; M. Wu; J. Zhang; X. Chen; K. Huang
>
> **备注:** Accepted by ICCV2025 Highlight ~
>
> **摘要:** Vision-language tracking aims to locate the target object in the video sequence using a template patch and a language description provided in the initial frame. To achieve robust tracking, especially in complex long-term scenarios that reflect real-world conditions as recently highlighted by MGIT, it is essential not only to characterize the target features but also to utilize the context features related to the target. However, the visual and textual target-context cues derived from the initial prompts generally align only with the initial target state. Due to their dynamic nature, target states are constantly changing, particularly in complex long-term sequences. It is intractable for these cues to continuously guide Vision-Language Trackers (VLTs). Furthermore, for the text prompts with diverse expressions, our experiments reveal that existing VLTs struggle to discern which words pertain to the target or the context, complicating the utilization of textual cues. In this work, we present a novel tracker named ATCTrack, which can obtain multimodal cues Aligned with the dynamic target states through comprehensive Target-Context feature modeling, thereby achieving robust tracking. Specifically, (1) for the visual modality, we propose an effective temporal visual target-context modeling approach that provides the tracker with timely visual cues. (2) For the textual modality, we achieve precise target words identification solely based on textual content, and design an innovative context words calibration method to adaptively utilize auxiliary context words. (3) We conduct extensive experiments on mainstream benchmarks and ATCTrack achieves a new SOTA performance. The code and models will be released at: https://github.com/XiaokunFeng/ATCTrack.
>
---
#### [new 142] FED-PsyAU: Privacy-Preserving Micro-Expression Recognition via Psychological AU Coordination and Dynamic Facial Motion Modeling
- **分类: cs.CV**

- **简介: 该论文属于微表情识别任务，旨在解决小样本、隐私保护和跨场景识别问题。作者提出FED-PsyAU框架，结合心理先验知识与统计特征，通过联邦学习提升多客户端下的识别性能，同时保护数据隐私。**

- **链接: [http://arxiv.org/pdf/2507.20557v1](http://arxiv.org/pdf/2507.20557v1)**

> **作者:** Jingting Li; Yu Qian; Lin Zhao; Su-Jing Wang
>
> **摘要:** Micro-expressions (MEs) are brief, low-intensity, often localized facial expressions. They could reveal genuine emotions individuals may attempt to conceal, valuable in contexts like criminal interrogation and psychological counseling. However, ME recognition (MER) faces challenges, such as small sample sizes and subtle features, which hinder efficient modeling. Additionally, real-world applications encounter ME data privacy issues, leaving the task of enhancing recognition across settings under privacy constraints largely unexplored. To address these issues, we propose a FED-PsyAU research framework. We begin with a psychological study on the coordination of upper and lower facial action units (AUs) to provide structured prior knowledge of facial muscle dynamics. We then develop a DPK-GAT network that combines these psychological priors with statistical AU patterns, enabling hierarchical learning of facial motion features from regional to global levels, effectively enhancing MER performance. Additionally, our federated learning framework advances MER capabilities across multiple clients without data sharing, preserving privacy and alleviating the limited-sample issue for each client. Extensive experiments on commonly-used ME databases demonstrate the effectiveness of our approach.
>
---
#### [new 143] LLMControl: Grounded Control of Text-to-Image Diffusion-based Synthesis with Multimodal LLMs
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决现有方法在复杂文本控制条件下生成不准确的问题。作者提出了LLM_Control框架，利用多模态大语言模型增强扩散模型的控制能力，通过注入控制信号优化生成结构与外观，提升生成质量。**

- **链接: [http://arxiv.org/pdf/2507.19939v1](http://arxiv.org/pdf/2507.19939v1)**

> **作者:** Jiaze Wang; Rui Chen; Haowang Cui
>
> **摘要:** Recent spatial control methods for text-to-image (T2I) diffusion models have shown compelling results. However, these methods still fail to precisely follow the control conditions and generate the corresponding images, especially when encountering the textual prompts that contain multiple objects or have complex spatial compositions. In this work, we present a LLM-guided framework called LLM\_Control to address the challenges of the controllable T2I generation task. By improving grounding capabilities, LLM\_Control is introduced to accurately modulate the pre-trained diffusion models, where visual conditions and textual prompts influence the structures and appearance generation in a complementary way. We utilize the multimodal LLM as a global controller to arrange spatial layouts, augment semantic descriptions and bind object attributes. The obtained control signals are injected into the denoising network to refocus and enhance attention maps according to novel sampling constraints. Extensive qualitative and quantitative experiments have demonstrated that LLM\_Control achieves competitive synthesis quality compared to other state-of-the-art methods across various pre-trained T2I models. It is noteworthy that LLM\_Control allows the challenging input conditions on which most of the existing methods
>
---
#### [new 144] MambaVesselNet++: A Hybrid CNN-Mamba Architecture for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决传统卷积模型感受野受限和视觉Transformer计算成本高的问题。论文提出MambaVesselNet++，结合CNN提取局部特征与Mamba建模长程依赖，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2507.19931v1](http://arxiv.org/pdf/2507.19931v1)**

> **作者:** Qing Xu; Yanming Chen; Yue Li; Ziyu Liu; Zhenye Lou; Yixuan Zhang; Xiangjian He
>
> **备注:** Accepted by TOMM
>
> **摘要:** Medical image segmentation plays an important role in computer-aided diagnosis. Traditional convolution-based U-shape segmentation architectures are usually limited by the local receptive field. Existing vision transformers have been widely applied to diverse medical segmentation frameworks due to their superior capabilities of capturing global contexts. Despite the advantage, the real-world application of vision transformers is challenged by their non-linear self-attention mechanism, requiring huge computational costs. To address this issue, the selective state space model (SSM) Mamba has gained recognition for its adeptness in modeling long-range dependencies in sequential data, particularly noted for its efficient memory costs. In this paper, we propose MambaVesselNet++, a Hybrid CNN-Mamba framework for medical image segmentation. Our MambaVesselNet++ is comprised of a hybrid image encoder (Hi-Encoder) and a bifocal fusion decoder (BF-Decoder). In Hi-Encoder, we first devise the texture-aware layer to capture low-level semantic features by leveraging convolutions. Then, we utilize Mamba to effectively model long-range dependencies with linear complexity. The Bi-Decoder adopts skip connections to combine local and global information of the Hi-Encoder for the accurate generation of segmentation masks. Extensive experiments demonstrate that MambaVesselNet++ outperforms current convolution-based, transformer-based, and Mamba-based state-of-the-arts across diverse medical 2D, 3D, and instance segmentation tasks. The code is available at https://github.com/CC0117/MambaVesselNet.
>
---
#### [new 145] Multi-Masked Querying Network for Robust Emotion Recognition from Incomplete Multi-Modal Physiological Signals
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文属于情感识别任务，旨在从不完整的多模态生理信号中准确识别情绪。论文提出MMQ-Net模型，通过引入多模态查询机制，解决信号缺失和噪声干扰问题，提升情绪识别的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.20737v1](http://arxiv.org/pdf/2507.20737v1)**

> **作者:** Geng-Xin Xu; Xiang Zuo; Ye Li
>
> **备注:** MICCAI2025
>
> **摘要:** Emotion recognition from physiological data is crucial for mental health assessment, yet it faces two significant challenges: incomplete multi-modal signals and interference from body movements and artifacts. This paper presents a novel Multi-Masked Querying Network (MMQ-Net) to address these issues by integrating multiple querying mechanisms into a unified framework. Specifically, it uses modality queries to reconstruct missing data from incomplete signals, category queries to focus on emotional state features, and interference queries to separate relevant information from noise. Extensive experiment results demonstrate the superior emotion recognition performance of MMQ-Net compared to existing approaches, particularly under high levels of data incompleteness.
>
---
#### [new 146] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决多编码器视觉语言模型计算开销大的问题。作者提出METEOR框架，通过多阶段协同剪枝策略，减少视觉token数量，降低冗余，同时保持高性能。实验表明该方法在多个任务上效果优异。**

- **链接: [http://arxiv.org/pdf/2507.20842v1](http://arxiv.org/pdf/2507.20842v1)**

> **作者:** Yuchen Liu; Yaoming Wang; Bowen Shi; Xiaopeng Zhang; Wenrui Dai; Chenglin Li; Hongkai Xiong; Qi Tian
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Vision encoders serve as the cornerstone of multimodal understanding. Single-encoder architectures like CLIP exhibit inherent constraints in generalizing across diverse multimodal tasks, while recent multi-encoder fusion methods introduce prohibitive computational overhead to achieve superior performance using complementary visual representations from multiple vision encoders. To address this, we propose a progressive pruning framework, namely Multi-Encoder collaboraTivE tOken pRuning (METEOR), that eliminates redundant visual tokens across the encoding, fusion, and decoding stages for multi-encoder MLLMs. For multi-vision encoding, we discard redundant tokens within each encoder via a rank guided collaborative token assignment strategy. Subsequently, for multi-vision fusion, we combine the visual features from different encoders while reducing cross-encoder redundancy with cooperative pruning. Finally, we propose an adaptive token pruning method in the LLM decoding stage to further discard irrelevant tokens based on the text prompts with dynamically adjusting pruning ratios for specific task demands. To our best knowledge, this is the first successful attempt that achieves an efficient multi-encoder based vision language model with multi-stage pruning strategies. Extensive experiments on 11 benchmarks demonstrate the effectiveness of our proposed approach. Compared with EAGLE, a typical multi-encoder MLLMs, METEOR reduces 76% visual tokens with only 0.3% performance drop in average. The code is available at https://github.com/YuchenLiu98/METEOR.
>
---
#### [new 147] Adapting Vehicle Detectors for Aerial Imagery to Unseen Domains with Weak Supervision
- **分类: cs.CV**

- **简介: 该论文属于 aerial imagery 中的车辆检测任务，旨在解决模型在不同地理区域间泛化能力差的问题。论文提出一种基于生成式 AI 的多阶段、多模态知识迁移框架，通过合成高质量遥感图像及标签来增强训练，有效缩小源域与目标域间的分布差异，显著提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.20976v1](http://arxiv.org/pdf/2507.20976v1)**

> **作者:** Xiao Fang; Minhyek Jeon; Zheyang Qin; Stanislav Panev; Celso de Melo; Shuowen Hu; Shayok Chakraborty; Fernando De la Torre
>
> **备注:** ICCV 2025
>
> **摘要:** Detecting vehicles in aerial imagery is a critical task with applications in traffic monitoring, urban planning, and defense intelligence. Deep learning methods have provided state-of-the-art (SOTA) results for this application. However, a significant challenge arises when models trained on data from one geographic region fail to generalize effectively to other areas. Variability in factors such as environmental conditions, urban layouts, road networks, vehicle types, and image acquisition parameters (e.g., resolution, lighting, and angle) leads to domain shifts that degrade model performance. This paper proposes a novel method that uses generative AI to synthesize high-quality aerial images and their labels, improving detector training through data augmentation. Our key contribution is the development of a multi-stage, multi-modal knowledge transfer framework utilizing fine-tuned latent diffusion models (LDMs) to mitigate the distribution gap between the source and target environments. Extensive experiments across diverse aerial imagery domains show consistent performance improvements in AP50 over supervised learning on source domain data, weakly supervised adaptation methods, unsupervised domain adaptation methods, and open-set object detectors by 4-23%, 6-10%, 7-40%, and more than 50%, respectively. Furthermore, we introduce two newly annotated aerial datasets from New Zealand and Utah to support further research in this field. Project page is available at: https://humansensinglab.github.io/AGenDA
>
---
#### [new 148] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务中的图像恢复方向，旨在解决从美化后的图像中恢复原始人脸的问题。作者提出了MoFRR方法，结合扩散模型与专家系统，通过低频与高频分支处理不同类型的人脸美化操作，实现了更准确的恢复效果。**

- **链接: [http://arxiv.org/pdf/2507.19770v1](http://arxiv.org/pdf/2507.19770v1)**

> **作者:** Jiaxin Liu; Qichao Ying; Zhenxing Qian; Sheng Li; Runqi Zhang; Jian Liu; Xinpeng Zhang
>
> **摘要:** The widespread use of face retouching on social media platforms raises concerns about the authenticity of face images. While existing methods focus on detecting face retouching, how to accurately recover the original faces from the retouched ones has yet to be answered. This paper introduces Face Retouching Restoration (FRR), a novel computer vision task aimed at restoring original faces from their retouched counterparts. FRR differs from traditional image restoration tasks by addressing the complex retouching operations with various types and degrees, which focuses more on the restoration of the low-frequency information of the faces. To tackle this challenge, we propose MoFRR, Mixture of Diffusion Models for FRR. Inspired by DeepSeek's expert isolation strategy, the MoFRR uses sparse activation of specialized experts handling distinct retouching types and the engagement of a shared expert dealing with universal retouching traces. Each specialized expert follows a dual-branch structure with a DDIM-based low-frequency branch guided by an Iterative Distortion Evaluation Module (IDEM) and a Cross-Attention-based High-Frequency branch (HFCAM) for detail refinement. Extensive experiments on a newly constructed face retouching dataset, RetouchingFFHQ++, demonstrate the effectiveness of MoFRR for FRR.
>
---
#### [new 149] Lightweight Transformer-Driven Segmentation of Hotspots and Snail Trails in Solar PV Thermal Imagery
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决太阳能电池板缺陷检测问题。作者提出了一种基于轻量级Transformer的语义分割模型（SegFormer），用于识别热点和蜗牛痕迹缺陷。工作包括数据预处理、模型设计与优化，并在实际热成像数据上进行训练与评估，结果显示其在准确性和效率方面优于现有方法，适合无人机系统集成实现大规模太阳能场自动化检测。**

- **链接: [http://arxiv.org/pdf/2507.20680v1](http://arxiv.org/pdf/2507.20680v1)**

> **作者:** Deepak Joshi; Mayukha Pal
>
> **备注:** 31 pages, 6 figures
>
> **摘要:** Accurate detection of defects such as hotspots and snail trails in photovoltaic modules is essential for maintaining energy efficiency and system reliablility. This work presents a supervised deep learning framework for segmenting thermal infrared images of PV panels, using a dataset of 277 aerial thermographic images captured by zenmuse XT infrared camera mounted on a DJI Matrice 100 drone. The preprocessing pipeline includes image resizing, CLAHE based contrast enhancement, denoising, and normalisation. A lightweight semantic segmentation model based on SegFormer is developed, featuring a customised Transformwer encoder and streamlined decoder, and fine-tuned on annotated images with manually labeled defect regions. To evaluate performance, we benchmark our model against U-Net, DeepLabV3, PSPNet, and Mask2Former using consistent preprocessing and augmentation. Evaluation metrices includes per-class Dice score, F1-score, Cohen's kappa, mean IoU, and pixel accuracy. The SegFormer-based model outperforms baselines in accuracy and efficiency, particularly for segmenting small and irregular defects. Its lightweight design real-time deployment on edge devices and seamless integration with drone-based systems for automated inspection of large-scale solar farms.
>
---
#### [new 150] Learning Phonetic Context-Dependent Viseme for Enhancing Speech-Driven 3D Facial Animation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于语音驱动三维面部动画生成任务，旨在解决传统方法生成动画不自然、缺乏连贯性的问题。作者提出了一种基于音素上下文的损失函数，通过建模音素对视觉发音单位（viseme）转换的影响，增强面部动作的动态连贯性，从而生成更自然的动画。**

- **链接: [http://arxiv.org/pdf/2507.20568v1](http://arxiv.org/pdf/2507.20568v1)**

> **作者:** Hyung Kyu Kim; Hak Gu Kim
>
> **备注:** Accepted for Interspeech 2025 Project Page: https://cau-irislab.github.io/interspeech25/
>
> **摘要:** Speech-driven 3D facial animation aims to generate realistic facial movements synchronized with audio. Traditional methods primarily minimize reconstruction loss by aligning each frame with ground-truth. However, this frame-wise approach often fails to capture the continuity of facial motion, leading to jittery and unnatural outputs due to coarticulation. To address this, we propose a novel phonetic context-aware loss, which explicitly models the influence of phonetic context on viseme transitions. By incorporating a viseme coarticulation weight, we assign adaptive importance to facial movements based on their dynamic changes over time, ensuring smoother and perceptually consistent animations. Extensive experiments demonstrate that replacing the conventional reconstruction loss with ours improves both quantitative metrics and visual quality. It highlights the importance of explicitly modeling phonetic context-dependent visemes in synthesizing natural speech-driven 3D facial animation. Project page: https://cau-irislab.github.io/interspeech25/
>
---
#### [new 151] NeuroVoxel-LM: Language-Aligned 3D Perception via Dynamic Voxelization and Meta-Embedding
- **分类: cs.CV; cs.AI; cs.LG; I.4; I.5**

- **简介: 该论文属于3D场景理解任务，旨在解决现有3D语言模型处理大规模点云时效率低、表示精度差的问题。作者提出了NeuroVoxel-LM，结合NeRF与动态体素化（DR-MSV）及轻量元嵌入（TAP-LME），提升特征提取效率与语义表示能力。**

- **链接: [http://arxiv.org/pdf/2507.20110v1](http://arxiv.org/pdf/2507.20110v1)**

> **作者:** Shiyu Liu; Lianlei Shan
>
> **备注:** **14 pages, 3 figures, 2 tables
>
> **摘要:** Recent breakthroughs in Visual Language Models (VLMs) and Multimodal Large Language Models (MLLMs) have significantly advanced 3D scene perception towards language-driven cognition. However, existing 3D language models struggle with sparse, large-scale point clouds due to slow feature extraction and limited representation accuracy. To address these challenges, we propose NeuroVoxel-LM, a novel framework that integrates Neural Radiance Fields (NeRF) with dynamic resolution voxelization and lightweight meta-embedding. Specifically, we introduce a Dynamic Resolution Multiscale Voxelization (DR-MSV) technique that adaptively adjusts voxel granularity based on geometric and structural complexity, reducing computational cost while preserving reconstruction fidelity. In addition, we propose the Token-level Adaptive Pooling for Lightweight Meta-Embedding (TAP-LME) mechanism, which enhances semantic representation through attention-based weighting and residual fusion. Experimental results demonstrate that DR-MSV significantly improves point cloud feature extraction efficiency and accuracy, while TAP-LME outperforms conventional max-pooling in capturing fine-grained semantics from NeRF weights.
>
---
#### [new 152] T$^\text{3}$SVFND: Towards an Evolving Fake News Detector for Emergencies with Test-time Training on Short Video Platforms
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于虚假新闻检测任务，旨在解决突发事件中短视频平台上的虚假新闻检测效果下降的问题。作者提出了T³SVFND框架，引入测试时训练机制和基于多模态信息的掩码语言建模辅助任务，提升模型对突发事件新闻的适应性和检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.20286v1](http://arxiv.org/pdf/2507.20286v1)**

> **作者:** Liyuan Zhang; Zeyun Cheng; Yan Yang; Yong Liu; Jinke Ma
>
> **备注:** 16 pages, 3 figures, published to DASFAA 2025
>
> **摘要:** The existing methods for fake news videos detection may not be generalized, because there is a distribution shift between short video news of different events, and the performance of such techniques greatly drops if news records are coming from emergencies. We propose a new fake news videos detection framework (T$^3$SVFND) using Test-Time Training (TTT) to alleviate this limitation, enhancing the robustness of fake news videos detection. Specifically, we design a self-supervised auxiliary task based on Mask Language Modeling (MLM) that masks a certain percentage of words in text and predicts these masked words by combining contextual information from different modalities (audio and video). In the test-time training phase, the model adapts to the distribution of test data through auxiliary tasks. Extensive experiments on the public benchmark demonstrate the effectiveness of the proposed model, especially for the detection of emergency news.
>
---
#### [new 153] Regularizing Subspace Redundancy of Low-Rank Adaptation
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 论文属于参数高效迁移学习任务，旨在解决低秩适应方法中子空间冗余影响特征适配的问题。作者提出ReSoRA，通过显式建模子空间冗余并施加去冗余约束，提升模型表现，且可无缝集成到现有方法中，无需额外推理成本。**

- **链接: [http://arxiv.org/pdf/2507.20745v1](http://arxiv.org/pdf/2507.20745v1)**

> **作者:** Yue Zhu; Haiwen Diao; Shang Gao; Jiazuo Yu; Jiawen Zhu; Yunzhi Zhuge; Shuai Hao; Xu Jia; Lu Zhang; Ying Zhang; Huchuan Lu
>
> **备注:** 10 pages, 4 figures, Accepted by ACMMM2025
>
> **摘要:** Low-Rank Adaptation (LoRA) and its variants have delivered strong capability in Parameter-Efficient Transfer Learning (PETL) by minimizing trainable parameters and benefiting from reparameterization. However, their projection matrices remain unrestricted during training, causing high representation redundancy and diminishing the effectiveness of feature adaptation in the resulting subspaces. While existing methods mitigate this by manually adjusting the rank or implicitly applying channel-wise masks, they lack flexibility and generalize poorly across various datasets and architectures. Hence, we propose ReSoRA, a method that explicitly models redundancy between mapping subspaces and adaptively Regularizes Subspace redundancy of Low-Rank Adaptation. Specifically, it theoretically decomposes the low-rank submatrices into multiple equivalent subspaces and systematically applies de-redundancy constraints to the feature distributions across different projections. Extensive experiments validate that our proposed method consistently facilitates existing state-of-the-art PETL methods across various backbones and datasets in vision-language retrieval and standard visual classification benchmarks. Besides, as a training supervision, ReSoRA can be seamlessly integrated into existing approaches in a plug-and-play manner, with no additional inference costs. Code is publicly available at: https://github.com/Lucenova/ReSoRA.
>
---
#### [new 154] Pre- and Post-Treatment Glioma Segmentation with the Medical Imaging Segmentation Toolkit
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决预处理和后处理阶段方法缺乏标准化的问题。作者扩展了MIST工具包的后处理模块，支持多种图像变换策略，以提升BraTS 2025挑战中的胶质瘤分割效果，并通过不同策略评估验证其性能。**

- **链接: [http://arxiv.org/pdf/2507.19626v1](http://arxiv.org/pdf/2507.19626v1)**

> **作者:** Adrian Celaya; Tucker Netherton; Dawid Schellingerhout; Caroline Chung; Beatrice Riviere; David Fuentes
>
> **摘要:** Medical image segmentation continues to advance rapidly, yet rigorous comparison between methods remains challenging due to a lack of standardized and customizable tooling. In this work, we present the current state of the Medical Imaging Segmentation Toolkit (MIST), with a particular focus on its flexible and modular postprocessing framework designed for the BraTS 2025 pre- and post-treatment glioma segmentation challenge. Since its debut in the 2024 BraTS adult glioma post-treatment segmentation challenge, MIST's postprocessing module has been significantly extended to support a wide range of transforms, including removal or replacement of small objects, extraction of the largest connected components, and morphological operations such as hole filling and closing. These transforms can be composed into user-defined strategies, enabling fine-grained control over the final segmentation output. We evaluate three such strategies - ranging from simple small-object removal to more complex, class-specific pipelines - and rank their performance using the BraTS ranking protocol. Our results highlight how MIST facilitates rapid experimentation and targeted refinement, ultimately producing high-quality segmentations for the BraTS 2025 challenge. MIST remains open source and extensible, supporting reproducible and scalable research in medical image segmentation.
>
---
#### [new 155] Knowledge Regularized Negative Feature Tuning for Out-of-Distribution Detection with Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的分布外检测任务，旨在解决负提示调优方法在提升OOD检测能力时导致的泛化性能下降问题。论文提出了知识正则化负特征调优（KR-NFT）方法，通过负特征调优（NFT）架构和知识正则化优化策略（KR），提升模型在已见和未见类别上的分类准确性和OOD检测效果，同时减少对预训练知识的遗忘。**

- **链接: [http://arxiv.org/pdf/2507.19847v1](http://arxiv.org/pdf/2507.19847v1)**

> **作者:** Wenjie Zhu; Yabin Zhang; Xin Jin; Wenjun Zeng; Lei Zhang
>
> **备注:** accepted by ACMMM 2025
>
> **摘要:** Out-of-distribution (OOD) detection is crucial for building reliable machine learning models. Although negative prompt tuning has enhanced the OOD detection capabilities of vision-language models, these tuned models often suffer from reduced generalization performance on unseen classes and styles. To address this challenge, we propose a novel method called Knowledge Regularized Negative Feature Tuning (KR-NFT), which integrates an innovative adaptation architecture termed Negative Feature Tuning (NFT) and a corresponding knowledge-regularization (KR) optimization strategy. Specifically, NFT applies distribution-aware transformations to pre-trained text features, effectively separating positive and negative features into distinct spaces. This separation maximizes the distinction between in-distribution (ID) and OOD images. Additionally, we introduce image-conditional learnable factors through a lightweight meta-network, enabling dynamic adaptation to individual images and mitigating sensitivity to class and style shifts. Compared to traditional negative prompt tuning, NFT demonstrates superior efficiency and scalability. To optimize this adaptation architecture, the KR optimization strategy is designed to enhance the discrimination between ID and OOD sets while mitigating pre-trained knowledge forgetting. This enhances OOD detection performance on trained ID classes while simultaneously improving OOD detection on unseen ID datasets. Notably, when trained with few-shot samples from ImageNet dataset, KR-NFT not only improves ID classification accuracy and OOD detection but also significantly reduces the FPR95 by 5.44\% under an unexplored generalization setting with unseen ID categories. Codes can be found at \href{https://github.com/ZhuWenjie98/KRNFT}{https://github.com/ZhuWenjie98/KRNFT}.
>
---
#### [new 156] GaRe: Relightable 3D Gaussian Splatting for Outdoor Scenes from Unconstrained Photo Collections
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建与 relighting 任务，旨在从无约束照片集合中实现户外场景的可重光照。为解决光照分离与动态阴影生成问题，作者提出 GaRe 方法，结合 3D Gaussian Splatting 与内在图像分解，引入残差光照提取、区域监督及光线追踪阴影模拟，实现高质量新视角合成与自然光照效果。**

- **链接: [http://arxiv.org/pdf/2507.20512v1](http://arxiv.org/pdf/2507.20512v1)**

> **作者:** Haiyang Bai; Jiaqi Zhu; Songru Jiang; Wei Huang; Tao Lu; Yuanqi Li; Jie Guo; Runze Fu; Yanwen Guo; Lijun Chen
>
> **摘要:** We propose a 3D Gaussian splatting-based framework for outdoor relighting that leverages intrinsic image decomposition to precisely integrate sunlight, sky radiance, and indirect lighting from unconstrained photo collections. Unlike prior methods that compress the per-image global illumination into a single latent vector, our approach enables simultaneously diverse shading manipulation and the generation of dynamic shadow effects. This is achieved through three key innovations: (1) a residual-based sun visibility extraction method to accurately separate direct sunlight effects, (2) a region-based supervision framework with a structural consistency loss for physically interpretable and coherent illumination decomposition, and (3) a ray-tracing-based technique for realistic shadow simulation. Extensive experiments demonstrate that our framework synthesizes novel views with competitive fidelity against state-of-the-art relighting solutions and produces more natural and multifaceted illumination and shadow effects.
>
---
#### [new 157] A Multimodal Architecture for Endpoint Position Prediction in Team-based Multiplayer Games
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出一种多模态架构，用于预测团队多人游戏中玩家的终点位置。该工作属于玩家行为预测任务，旨在解决复杂游戏环境中玩家移动预测的问题，通过结合图像、数值和动态游戏数据，实现对未来位置的概率预测，支持后续如玩家行为模拟和异常检测等应用。**

- **链接: [http://arxiv.org/pdf/2507.20670v1](http://arxiv.org/pdf/2507.20670v1)**

> **作者:** Jonas Peche; Aliaksei Tsishurou; Alexander Zap; Guenter Wallner
>
> **摘要:** Understanding and predicting player movement in multiplayer games is crucial for achieving use cases such as player-mimicking bot navigation, preemptive bot control, strategy recommendation, and real-time player behavior analytics. However, the complex environments allow for a high degree of navigational freedom, and the interactions and team-play between players require models that make effective use of the available heterogeneous input data. This paper presents a multimodal architecture for predicting future player locations on a dynamic time horizon, using a U-Net-based approach for calculating endpoint location probability heatmaps, conditioned using a multimodal feature encoder. The application of a multi-head attention mechanism for different groups of features allows for communication between agents. In doing so, the architecture makes efficient use of the multimodal game state including image inputs, numerical and categorical features, as well as dynamic game data. Consequently, the presented technique lays the foundation for various downstream tasks that rely on future player positions such as the creation of player-predictive bot behavior or player anomaly detection.
>
---
#### [new 158] TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking
- **分类: cs.CV**

- **简介: 该论文属于3D点云跟踪任务，旨在解决现有方法需为每类物体单独建模、泛化能力差的问题。作者提出TrackAny3D，首次将大规模预训练3D模型迁移至类别无关的单目标跟踪，通过适配器、几何专家混合结构和时序优化策略，实现高效统一跟踪，在多个基准上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2507.19908v1](http://arxiv.org/pdf/2507.19908v1)**

> **作者:** Mengmeng Wang; Haonan Wang; Yulong Li; Xiangjie Kong; Jiaxin Du; Guojiang Shen; Feng Xia
>
> **摘要:** 3D LiDAR-based single object tracking (SOT) relies on sparse and irregular point clouds, posing challenges from geometric variations in scale, motion patterns, and structural complexity across object categories. Current category-specific approaches achieve good accuracy but are impractical for real-world use, requiring separate models for each category and showing limited generalization. To tackle these issues, we propose TrackAny3D, the first framework to transfer large-scale pretrained 3D models for category-agnostic 3D SOT. We first integrate parameter-efficient adapters to bridge the gap between pretraining and tracking tasks while preserving geometric priors. Then, we introduce a Mixture-of-Geometry-Experts (MoGE) architecture that adaptively activates specialized subnetworks based on distinct geometric characteristics. Additionally, we design a temporal context optimization strategy that incorporates learnable temporal tokens and a dynamic mask weighting module to propagate historical information and mitigate temporal drift. Experiments on three commonly-used benchmarks show that TrackAny3D establishes new state-of-the-art performance on category-agnostic 3D SOT, demonstrating strong generalization and competitiveness. We hope this work will enlighten the community on the importance of unified models and further expand the use of large-scale pretrained models in this field.
>
---
#### [new 159] Self-Guided Masked Autoencoder
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的自监督表示学习任务，旨在解决MAE预训练中随机掩码效率低的问题。作者发现MAE在早期阶段即可实现基于模式的块级聚类，据此提出自引导MAE，利用内部聚类进度生成掩码，替代传统随机掩码，从而提升学习效率，且无需外部模型或信息。实验验证了方法在多种下游任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.19773v1](http://arxiv.org/pdf/2507.19773v1)**

> **作者:** Jeongwoo Shin; Inseo Lee; Junho Lee; Joonseok Lee
>
> **摘要:** Masked Autoencoder (MAE) is a self-supervised approach for representation learning, widely applicable to a variety of downstream tasks in computer vision. In spite of its success, it is still not fully uncovered what and how MAE exactly learns. In this paper, with an in-depth analysis, we discover that MAE intrinsically learns pattern-based patch-level clustering from surprisingly early stages of pretraining. Upon this understanding, we propose self-guided masked autoencoder, which internally generates informed mask by utilizing its progress in patch clustering, substituting the naive random masking of the vanilla MAE. Our approach significantly boosts its learning process without relying on any external models or supplementary information, keeping the benefit of self-supervised nature of MAE intact. Comprehensive experiments on various downstream tasks verify the effectiveness of the proposed method.
>
---
#### [new 160] A mini-batch training strategy for deep subspace clustering networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度子空间聚类任务，旨在解决现有方法依赖全批量处理导致的效率与扩展性问题。论文提出了一种小批量训练策略，结合记忆库保存全局特征，并引入无需解码器的对比学习框架，实现高效训练。实验表明其在多个数据集上性能优越。**

- **链接: [http://arxiv.org/pdf/2507.19917v1](http://arxiv.org/pdf/2507.19917v1)**

> **作者:** Yuxuan Jiang; Chenwei Yu; Zhi Lin; Xiaolan Liu
>
> **摘要:** Mini-batch training is a cornerstone of modern deep learning, offering computational efficiency and scalability for training complex architectures. However, existing deep subspace clustering (DSC) methods, which typically combine an autoencoder with a self-expressive layer, rely on full-batch processing. The bottleneck arises from the self-expressive module, which requires representations of the entire dataset to construct a self-representation coefficient matrix. In this work, we introduce a mini-batch training strategy for DSC by integrating a memory bank that preserves global feature representations. Our approach enables scalable training of deep architectures for subspace clustering with high-resolution images, overcoming previous limitations. Additionally, to efficiently fine-tune large-scale pre-trained encoders for subspace clustering, we propose a decoder-free framework that leverages contrastive learning instead of autoencoding for representation learning. This design not only eliminates the computational overhead of decoder training but also provides competitive performance. Extensive experiments demonstrate that our approach not only achieves performance comparable to full-batch methods, but outperforms other state-of-the-art subspace clustering methods on the COIL100 and ORL datasets by fine-tuning deep networks.
>
---
#### [new 161] Can Foundation Models Predict Fitness for Duty?
- **分类: cs.CV**

- **简介: 论文探讨使用基础模型通过近红外虹膜图像预测工作适任性，解决因缺乏足够相关数据而难以训练AI模型的问题。任务属于生物特征分析与深度学习应用领域，旨在提升警觉性评估效果。**

- **链接: [http://arxiv.org/pdf/2507.20418v1](http://arxiv.org/pdf/2507.20418v1)**

> **作者:** Juan E. Tapia; Christoph Busch
>
> **摘要:** Biometric capture devices have been utilised to estimate a person's alertness through near-infrared iris images, expanding their use beyond just biometric recognition. However, capturing a substantial number of corresponding images related to alcohol consumption, drug use, and sleep deprivation to create a dataset for training an AI model presents a significant challenge. Typically, a large quantity of images is required to effectively implement a deep learning approach. Currently, training downstream models with a huge number of images based on foundational models provides a real opportunity to enhance this area, thanks to the generalisation capabilities of self-supervised models. This work examines the application of deep learning and foundational models in predicting fitness for duty, which is defined as the subject condition related to determining the alertness for work.
>
---
#### [new 162] Low-Cost Machine Vision System for Sorting Green Lentils (Lens Culinaris) Based on Pneumatic Ejection and Deep Learning
- **分类: cs.CV**

- **简介: 该论文属于机器视觉与农产品分拣任务，旨在解决绿色扁豆分类效率与准确性问题。作者设计了一套基于YOLOv8模型的双阶段视觉系统，结合气动分选装置与Arduino控制，实现对六类扁豆的实时识别与分拣，验证了低成本平台在该场景中的可行性。**

- **链接: [http://arxiv.org/pdf/2507.20531v1](http://arxiv.org/pdf/2507.20531v1)**

> **作者:** Davy Rojas Yana; Edwin Salcedo
>
> **备注:** Accepted for publication in the Proceedings of the 30th International Conference on Automation and Computing (ICAC 2025)
>
> **摘要:** This paper presents the design, development, and evaluation of a dynamic grain classification system for green lentils (Lens Culinaris), which leverages computer vision and pneumatic ejection. The system integrates a YOLOv8-based detection model that identifies and locates grains on a conveyor belt, together with a second YOLOv8-based classification model that categorises grains into six classes: Good, Yellow, Broken, Peeled, Dotted, and Reject. This two-stage YOLOv8 pipeline enables accurate, real-time, multi-class categorisation of lentils, implemented on a low-cost, modular hardware platform. The pneumatic ejection mechanism separates defective grains, while an Arduino-based control system coordinates real-time interaction between the vision system and mechanical components. The system operates effectively at a conveyor speed of 59 mm/s, achieving a grain separation accuracy of 87.2%. Despite a limited processing rate of 8 grams per minute, the prototype demonstrates the potential of machine vision for grain sorting and provides a modular foundation for future enhancements.
>
---
#### [new 163] Enhancing Spatial Reasoning through Visual and Textual Thinking
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉问答中的空间推理任务，旨在解决现有模型在空间理解上的不足。作者提出了SpatialVTS方法，通过视觉与文本协同推理，结合数据集优化与模型训练，有效提升模型在2D/3D空间关系推理上的表现，无需额外信息输入。**

- **链接: [http://arxiv.org/pdf/2507.20529v1](http://arxiv.org/pdf/2507.20529v1)**

> **作者:** Xun Liang; Xin Guo; Zhongming Jin; Weihang Pan; Penghui Shang; Deng Cai; Binbin Lin; Jieping Ye
>
> **摘要:** The spatial reasoning task aims to reason about the spatial relationships in 2D and 3D space, which is a fundamental capability for Visual Question Answering (VQA) and robotics. Although vision language models (VLMs) have developed rapidly in recent years, they are still struggling with the spatial reasoning task. In this paper, we introduce a method that can enhance Spatial reasoning through Visual and Textual thinking Simultaneously (SpatialVTS). In the spatial visual thinking phase, our model is trained to generate location-related specific tokens of essential targets automatically. Not only are the objects mentioned in the problem addressed, but also the potential objects related to the reasoning are considered. During the spatial textual thinking phase, Our model conducts long-term thinking based on visual cues and dialogues, gradually inferring the answers to spatial reasoning problems. To effectively support the model's training, we perform manual corrections to the existing spatial reasoning dataset, eliminating numerous incorrect labels resulting from automatic annotation, restructuring the data input format to enhance generalization ability, and developing thinking processes with logical reasoning details. Without introducing additional information (such as masks or depth), our model's overall average level in several spatial understanding tasks has significantly improved compared with other models.
>
---
#### [new 164] Enhanced Deep Learning DeepFake Detection Integrating Handcrafted Features
- **分类: cs.CV**

- **简介: 该论文属于图像安全与身份验证任务，旨在解决深度伪造（DeepFake）图像检测问题。针对现有方法难以应对复杂面部篡改的问题，作者提出了一种结合手工提取的频域特征与RGB输入的深度学习检测框架，有效利用图像操作中产生的频域和空间域伪影，提高了检测的准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.20608v1](http://arxiv.org/pdf/2507.20608v1)**

> **作者:** Alejandro Hinke-Navarro; Mario Nieto-Hidalgo; Juan M. Espin; Juan E. Tapia
>
> **摘要:** The rapid advancement of deepfake and face swap technologies has raised significant concerns in digital security, particularly in identity verification and onboarding processes. Conventional detection methods often struggle to generalize against sophisticated facial manipulations. This study proposes an enhanced deep-learning detection framework that combines handcrafted frequency-domain features with conventional RGB inputs. This hybrid approach exploits frequency and spatial domain artifacts introduced during image manipulation, providing richer and more discriminative information to the classifier. Several frequency handcrafted features were evaluated, including the Steganalysis Rich Model, Discrete Cosine Transform, Error Level Analysis, Singular Value Decomposition, and Discrete Fourier Transform
>
---
#### [new 165] Object-centric Video Question Answering with Visual Grounding and Referring
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决现有视频大模型缺乏对象中心交互能力的问题。作者提出了一种支持视觉输入与输出的VideoLLM模型，引入了STOM模块实现跨帧视觉提示传播，并构建了VideoInfer数据集用于评估。实验表明该模型在多项视频问答与分割任务中优于基线方法。**

- **链接: [http://arxiv.org/pdf/2507.19599v1](http://arxiv.org/pdf/2507.19599v1)**

> **作者:** Haochen Wang; Qirui Chen; Cilin Yan; Jiayin Cai; Xiaolong Jiang; Yao Hu; Weidi Xie; Stratis Gavves
>
> **摘要:** Video Large Language Models (VideoLLMs) have recently demonstrated remarkable progress in general video understanding. However, existing models primarily focus on high-level comprehension and are limited to text-only responses, restricting the flexibility for object-centric, multiround interactions. In this paper, we make three contributions: (i) we address these limitations by introducing a VideoLLM model, capable of performing both object referring for input and grounding for output in video reasoning tasks, i.e., allowing users to interact with videos using both textual and visual prompts; (ii) we propose STOM (Spatial-Temporal Overlay Module), a novel approach that propagates arbitrary visual prompts input at any single timestamp to the remaining frames within a video; (iii) we present VideoInfer, a manually curated object-centric video instruction dataset featuring questionanswering pairs that require reasoning. We conduct comprehensive experiments on VideoInfer and other existing benchmarks across video question answering and referring object segmentation. The results on 12 benchmarks of 6 tasks show that our proposed model consistently outperforms baselines in both video question answering and segmentation, underscoring its robustness in multimodal, object-centric video and image understanding. Project page: https://qirui-chen.github.io/RGA3-release/.
>
---
#### [new 166] Learning Transferable Facial Emotion Representations from Large-Scale Semantically Rich Captions
- **分类: cs.CV**

- **简介: 该论文属于面部情感识别任务，旨在解决传统方法依赖固定类别或抽象维度导致的表达受限问题。作者构建了大规模情感描述数据集EmoCap100K，并提出EmoCapCLIP框架，通过全局-局部对比学习和跨模态正样本挖掘，有效利用丰富语义监督，提升情感表征的泛化与细粒度识别能力。**

- **链接: [http://arxiv.org/pdf/2507.21015v1](http://arxiv.org/pdf/2507.21015v1)**

> **作者:** Licai Sun; Xingxun Jiang; Haoyu Chen; Yante Li; Zheng Lian; Biu Liu; Yuan Zong; Wenming Zheng; Jukka M. Leppänen; Guoying Zhao
>
> **摘要:** Current facial emotion recognition systems are predominately trained to predict a fixed set of predefined categories or abstract dimensional values. This constrained form of supervision hinders generalization and applicability, as it reduces the rich and nuanced spectrum of emotions into oversimplified labels or scales. In contrast, natural language provides a more flexible, expressive, and interpretable way to represent emotions, offering a much broader source of supervision. Yet, leveraging semantically rich natural language captions as supervisory signals for facial emotion representation learning remains relatively underexplored, primarily due to two key challenges: 1) the lack of large-scale caption datasets with rich emotional semantics, and 2) the absence of effective frameworks tailored to harness such rich supervision. To this end, we introduce EmoCap100K, a large-scale facial emotion caption dataset comprising over 100,000 samples, featuring rich and structured semantic descriptions that capture both global affective states and fine-grained local facial behaviors. Building upon this dataset, we further propose EmoCapCLIP, which incorporates a joint global-local contrastive learning framework enhanced by a cross-modal guided positive mining module. This design facilitates the comprehensive exploitation of multi-level caption information while accommodating semantic similarities between closely related expressions. Extensive evaluations on over 20 benchmarks covering five tasks demonstrate the superior performance of our method, highlighting the promise of learning facial emotion representations from large-scale semantically rich captions. The code and data will be available at https://github.com/sunlicai/EmoCapCLIP.
>
---
#### [new 167] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images
- **分类: cs.CV**

- **简介: 该论文属于3D语义场景图生成任务，旨在解决现有方法计算量大、难以实时处理的问题。作者提出FROSS，通过将2D场景图提升至3D空间并使用3D高斯分布表示物体，实现了在线、超实时的3D SSG生成，并构建了带关系标注的ReplicaSSG数据集用于评估。**

- **链接: [http://arxiv.org/pdf/2507.19993v1](http://arxiv.org/pdf/2507.19993v1)**

> **作者:** Hao-Yu Hou; Chun-Yi Lee; Motoharu Sonogashira; Yasutomo Kawanishi
>
> **摘要:** The ability to abstract complex 3D environments into simplified and structured representations is crucial across various domains. 3D semantic scene graphs (SSGs) achieve this by representing objects as nodes and their interrelationships as edges, facilitating high-level scene understanding. Existing methods for 3D SSG generation, however, face significant challenges, including high computational demands and non-incremental processing that hinder their suitability for real-time open-world applications. To address this issue, we propose FROSS (Faster-than-Real-Time Online 3D Semantic Scene Graph Generation), an innovative approach for online and faster-than-real-time 3D SSG generation that leverages the direct lifting of 2D scene graphs to 3D space and represents objects as 3D Gaussian distributions. This framework eliminates the dependency on precise and computationally-intensive point cloud processing. Furthermore, we extend the Replica dataset with inter-object relationship annotations, creating the ReplicaSSG dataset for comprehensive evaluation of FROSS. The experimental results from evaluations on ReplicaSSG and 3DSSG datasets show that FROSS can achieve superior performance while operating significantly faster than prior 3D SSG generation methods. Our implementation and dataset are publicly available at https://github.com/Howardkhh/FROSS.
>
---
#### [new 168] KASportsFormer: Kinematic Anatomy Enhanced Transformer for 3D Human Pose Estimation on Short Sports Scene Video
- **分类: cs.CV**

- **简介: 该论文属于3D人体姿态估计任务，旨在解决体育场景中因动作复杂、运动模糊和遮挡导致的估计困难。作者提出了KASportsFormer，通过引入基于运动学解剖信息的特征表示和整合模块，提升了对瞬间体育动作的估计能力。实验表明其在SportsPose和WorldPose数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2507.20763v1](http://arxiv.org/pdf/2507.20763v1)**

> **作者:** Zhuoer Yin; Calvin Yeung; Tomohiro Suzuki; Ryota Tanaka; Keisuke Fujii
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Recent transformer based approaches have demonstrated impressive performance in solving real-world 3D human pose estimation problems. Albeit these approaches achieve fruitful results on benchmark datasets, they tend to fall short of sports scenarios where human movements are more complicated than daily life actions, as being hindered by motion blur, occlusions, and domain shifts. Moreover, due to the fact that critical motions in a sports game often finish in moments of time (e.g., shooting), the ability to focus on momentary actions is becoming a crucial factor in sports analysis, where current methods appear to struggle with instantaneous scenarios. To overcome these limitations, we introduce KASportsFormer, a novel transformer based 3D pose estimation framework for sports that incorporates a kinematic anatomy-informed feature representation and integration module. In which the inherent kinematic motion information is extracted with the Bone Extractor (BoneExt) and Limb Fuser (LimbFus) modules and encoded in a multimodal manner. This improved the capability of comprehending sports poses in short videos. We evaluate our method through two representative sports scene datasets: SportsPose and WorldPose. Experimental results show that our proposed method achieves state-of-the-art results with MPJPE errors of 58.0mm and 34.3mm, respectively. Our code and models are available at: https://github.com/jw0r1n/KASportsFormer
>
---
#### [new 169] Exploring text-to-image generation for historical document image retrieval
- **分类: cs.CV**

- **简介: 该论文属于文档图像检索任务，旨在解决传统基于示例的检索依赖样本查询的问题。作者提出T2I-QBE方法，利用文本到图像生成模型，根据属性描述生成查询图像，结合QBE和ABDIR优势，实现基于文本提示的历史文档图像检索。**

- **链接: [http://arxiv.org/pdf/2507.20934v1](http://arxiv.org/pdf/2507.20934v1)**

> **作者:** Melissa Cote; Alexandra Branzan Albu
>
> **备注:** Accepted and presented as an extended abstract (double-blind review process) at the 2025 Scandinavian Conference on Image Analysis (SCIA). 4 pages
>
> **摘要:** Attribute-based document image retrieval (ABDIR) was recently proposed as an alternative to query-by-example (QBE) searches, the dominant document image retrieval (DIR) paradigm. One drawback of QBE searches is that they require sample query documents on hand that may not be available. ABDIR aims to offer users a flexible way to retrieve document images based on memorable visual features of document contents, describing document images with combinations of visual attributes determined via convolutional neural network (CNN)-based binary classifiers. We present an exploratory study of the use of generative AI to bridge the gap between QBE and ABDIR, focusing on historical documents as a use case for their diversity and uniqueness in visual features. We hypothesize that text-to-image (T2I) generation can be leveraged to create query document images using text prompts based on ABDIR-like attributes. We propose T2I-QBE, which uses Leonardo.Ai as the T2I generator with prompts that include a rough description of the desired document type and a list of the desired ABDIR-style attributes. This creates query images that are then used within the traditional QBE paradigm, which compares CNN-extracted query features to those of the document images in the dataset to retrieve the most relevant documents. Experiments on the HisIR19 dataset of historical documents confirm our hypothesis and suggest that T2I-QBE is a viable option for historical document image retrieval. To the authors' knowledge, this is the first attempt at utilizing T2I generation for DIR.
>
---
#### [new 170] TAPS : Frustratingly Simple Test Time Active Learning for VLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型（VLM）任务，旨在解决测试时如何在连续数据流中实时利用少量样本提升模型性能的问题。作者提出了TTAL框架，通过动态查询不确定样本、更新提示，并引入熵阈值、内存替换策略和分布对齐方法，实现实时、低开销的主动学习，适用于自动驾驶和医疗等安全关键场景。**

- **链接: [http://arxiv.org/pdf/2507.20028v1](http://arxiv.org/pdf/2507.20028v1)**

> **作者:** Dhruv Sarkar; Aprameyo Chakrabartty; Bibhudatta Bhanja
>
> **摘要:** Test-Time Optimization enables models to adapt to new data during inference by updating parameters on-the-fly. Recent advances in Vision-Language Models (VLMs) have explored learning prompts at test time to improve performance in downstream tasks. In this work, we extend this idea by addressing a more general and practical challenge: Can we effectively utilize an oracle in a continuous data stream where only one sample is available at a time, requiring an immediate query decision while respecting latency and memory constraints? To tackle this, we propose a novel Test-Time Active Learning (TTAL) framework that adaptively queries uncertain samples and updates prompts dynamically. Unlike prior methods that assume batched data or multiple gradient updates, our approach operates in a real-time streaming scenario with a single test sample per step. We introduce a dynamically adjusted entropy threshold for active querying, a class-balanced replacement strategy for memory efficiency, and a class-aware distribution alignment technique to enhance adaptation. The design choices are justified using careful theoretical analysis. Extensive experiments across 10 cross-dataset transfer benchmarks and 4 domain generalization datasets demonstrate consistent improvements over state-of-the-art methods while maintaining reasonable latency and memory overhead. Our framework provides a practical and effective solution for real-world deployment in safety-critical applications such as autonomous systems and medical diagnostics.
>
---
#### [new 171] Investigating the Effect of Spatial Context on Multi-Task Sea Ice Segmentation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于遥感图像多任务分割任务，旨在解决不同空间上下文对海冰分割效果的影响问题。作者通过调整空洞卷积率控制感受野大小，结合Sentinel-1 SAR与AMSR2数据，分析多尺度空间上下文对海冰浓度、发展阶段和浮冰尺寸分割的影响，并验证多源数据融合的有效性。**

- **链接: [http://arxiv.org/pdf/2507.20507v1](http://arxiv.org/pdf/2507.20507v1)**

> **作者:** Behzad Vahedi; Rafael Pires de Lima; Sepideh Jalayer; Walter N. Meier; Andrew P. Barrett; Morteza Karimzadeh
>
> **摘要:** Capturing spatial context at multiple scales is crucial for deep learning-based sea ice segmentation. However, the optimal specification of spatial context based on observation resolution and task characteristics remains underexplored. This study investigates the impact of spatial context on the segmentation of sea ice concentration, stage of development, and floe size using a multi-task segmentation model. We implement Atrous Spatial Pyramid Pooling with varying atrous rates to systematically control the receptive field size of convolutional operations, and to capture multi-scale contextual information. We explore the interactions between spatial context and feature resolution for different sea ice properties and examine how spatial context influences segmentation performance across different input feature combinations from Sentinel-1 SAR and Advanced Microwave Radiometer-2 (AMSR2) for multi-task mapping. Using Gradient-weighted Class Activation Mapping, we visualize how atrous rates influence model decisions. Our findings indicate that smaller receptive fields excel for high-resolution Sentinel-1 data, while medium receptive fields yield better performances for stage of development segmentation and larger receptive fields often lead to diminished performances. The fusion of SAR and AMSR2 enhances segmentation across all tasks. We highlight the value of lower-resolution 18.7 and 36.5 GHz AMSR2 channels in sea ice mapping. These findings highlight the importance of selecting appropriate spatial context based on observation resolution and target properties in sea ice mapping. By systematically analyzing receptive field effects in a multi-task setting, our study provides insights for optimizing deep learning models in geospatial applications.
>
---
#### [new 172] A Fast Parallel Median Filtering Algorithm Using Hierarchical Tiling
- **分类: cs.DC; cs.CV; I.3.1; I.4.3**

- **简介: 该论文属于图像处理任务，旨在解决中值滤波算法计算成本高的问题。通过分层分块策略，提出两种新算法：基于选择网络和基于内存的排序方法，显著提升效率。实验表明，其性能优于现有方法，尤其在GPU上表现突出。**

- **链接: [http://arxiv.org/pdf/2507.19926v1](http://arxiv.org/pdf/2507.19926v1)**

> **作者:** Louis Sugy
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Median filtering is a non-linear smoothing technique widely used in digital image processing to remove noise while retaining sharp edges. It is particularly well suited to removing outliers (impulse noise) or granular artifacts (speckle noise). However, the high computational cost of median filtering can be prohibitive. Sorting-based algorithms excel with small kernels but scale poorly with increasing kernel diameter, in contrast to constant-time methods characterized by higher constant factors but better scalability, such as histogram-based approaches or the 2D wavelet matrix. This paper introduces a novel algorithm, leveraging the separability of the sorting problem through hierarchical tiling to minimize redundant computations. We propose two variants: a data-oblivious selection network that can operate entirely within registers, and a data-aware version utilizing random-access memory. These achieve per-pixel complexities of $O(k \log(k))$ and $O(k)$, respectively, for a $k \times k$ kernel - unprecedented for sorting-based methods. Our CUDA implementation is up to 5 times faster than the current state of the art on a modern GPU and is the fastest median filter in most cases for 8-, 16-, and 32-bit data types and kernels from $3 \times 3$ to $75 \times 75$.
>
---
#### [new 173] MAIA: A Collaborative Medical AI Platform for Integrated Healthcare Innovation
- **分类: cs.AI; cs.CV; cs.HC; cs.SE**

- **简介: 该论文提出了MAIA，一个用于医疗AI协作的开源平台，旨在弥合技术与临床应用间的差距。它支持跨学科合作，具备数据管理、模型开发、部署等功能，促进AI研究向临床实践转化，解决医疗AI协作与集成难题。**

- **链接: [http://arxiv.org/pdf/2507.19489v1](http://arxiv.org/pdf/2507.19489v1)**

> **作者:** Simone Bendazzoli; Sanna Persson; Mehdi Astaraki; Sebastian Pettersson; Vitali Grozman; Rodrigo Moreno
>
> **备注:** 26 pages, 12 figures
>
> **摘要:** The integration of Artificial Intelligence (AI) into clinical workflows requires robust collaborative platforms that are able to bridge the gap between technical innovation and practical healthcare applications. This paper introduces MAIA (Medical Artificial Intelligence Assistant), an open-source platform designed to facilitate interdisciplinary collaboration among clinicians, researchers, and AI developers. Built on Kubernetes, MAIA offers a modular, scalable environment with integrated tools for data management, model development, annotation, deployment, and clinical feedback. Key features include project isolation, CI/CD automation, integration with high-computing infrastructures and in clinical workflows. MAIA supports real-world use cases in medical imaging AI, with deployments in both academic and clinical environments. By promoting collaborations and interoperability, MAIA aims to accelerate the translation of AI research into impactful clinical solutions while promoting reproducibility, transparency, and user-centered design. We showcase the use of MAIA with different projects, both at KTH Royal Institute of Technology and Karolinska University Hospital.
>
---
#### [new 174] AR-LIF: Adaptive reset leaky-integrate and fire neuron for spiking neural networks
- **分类: cs.NE; cs.AI; cs.CV**

- **简介: 论文提出AR-LIF神经元，属神经网络设计任务，旨在解决脉冲神经网络中硬重置导致的信息丢失与软重置缺乏自适应性的问题。通过建立输入、输出与重置的关联机制，并引入阈值调整策略，在保持低能耗优势的同时提升了模型性能。**

- **链接: [http://arxiv.org/pdf/2507.20746v1](http://arxiv.org/pdf/2507.20746v1)**

> **作者:** Zeyu Huang; Wei Meng; Quan Liu; Kun Chen; Li Ma
>
> **摘要:** Spiking neural networks possess the advantage of low energy consumption due to their event-driven nature. Compared with binary spike outputs, their inherent floating-point dynamics are more worthy of attention. The threshold level and reset mode of neurons play a crucial role in determining the number and timing of spikes. The existing hard reset method causes information loss, while the improved soft reset method adopts a uniform treatment for neurons. In response to this, this paper designs an adaptive reset neuron, establishing the correlation between input, output and reset, and integrating a simple yet effective threshold adjustment strategy. It achieves excellent performance on various datasets while maintaining the advantage of low energy consumption.
>
---
#### [new 175] GNSP: Gradient Null Space Projection for Preserving Cross-Modal Alignment in VLMs Continual Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决视觉语言模型（如CLIP）在连续微调时出现的灾难性遗忘和跨模态对齐退化问题。作者提出GNSP方法，通过将任务特定梯度投影到先前知识的零空间，避免干扰旧知识，并结合知识蒸馏和模态对齐损失保持嵌入空间结构。实验表明其在MTIL基准上表现优异，有效维持了CLIP的零样本能力和跨模态检索性能。**

- **链接: [http://arxiv.org/pdf/2507.19839v1](http://arxiv.org/pdf/2507.19839v1)**

> **作者:** Tiantian Peng; Yuyang Liu; Shuo Yang; Qiuhe Hong; YongHong Tian
>
> **摘要:** Contrastive Language-Image Pretraining has demonstrated remarkable zero-shot generalization by aligning visual and textual modalities in a shared embedding space. However, when continuously fine-tuned on diverse tasks, CLIP suffers from catastrophic forgetting and degradation of its embedding alignment, undermining its zero-shot capabilities. In this work, we propose Gradient Null Space Projection (GNSP), an efficient continual learning method that projects task-specific gradients onto the null space of previously learned knowledge. This orthogonal projection mathematically prevents interference with previous tasks without relying on rehearsal or architectural modification. Furthermore, to preserve the inherent generalization property of CLIP, we introduce knowledge distillation and combine it with a modality alignment preservation loss inspired by CLIP pre-training to stabilize the structure of the multimodal embedding space during fine-tuning. On the MTIL benchmark consisting of 11 tasks, our method achieved SOTA performance on both the Average and Last key metrics. More importantly, experiments show that our method successfully maintains the original modality gap and cross-modal retrieval performance of CLIP, confirming its effectiveness in maintaining a robust visual-language space throughout the continual learning process.
>
---
#### [new 176] Taming Domain Shift in Multi-source CT-Scan Classification via Input-Space Standardization
- **分类: eess.IV; cs.CE; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决多来源CT扫描图像因域偏移导致的泛化问题。作者通过分析SSFL++和KDS预处理方法，揭示其在输入空间标准化方面的有效性，展示了该方法如何提升跨源泛化能力，并通过实验验证其优越性。**

- **链接: [http://arxiv.org/pdf/2507.19858v1](http://arxiv.org/pdf/2507.19858v1)**

> **作者:** Chia-Ming Lee; Bo-Cheng Qiu; Ting-Yao Chen; Ming-Han Sun; Fang-Ying Lin; Jung-Tse Tsai; I-An Tsai; Yu-Fan Lin; Chih-Chung Hsu
>
> **备注:** Accepted by ICCVW 2025, Winner solution of PHAROS-AFE-AIMI Workshop's Multi-Source Covid-19 Detection Challenge
>
> **摘要:** Multi-source CT-scan classification suffers from domain shifts that impair cross-source generalization. While preprocessing pipelines combining Spatial-Slice Feature Learning (SSFL++) and Kernel-Density-based Slice Sampling (KDS) have shown empirical success, the mechanisms underlying their domain robustness remain underexplored. This study analyzes how this input-space standardization manages the trade-off between local discriminability and cross-source generalization. The SSFL++ and KDS pipeline performs spatial and temporal standardization to reduce inter-source variance, effectively mapping disparate inputs into a consistent target space. This preemptive alignment mitigates domain shift and simplifies the learning task for network optimization. Experimental validation demonstrates consistent improvements across architectures, proving the benefits stem from the preprocessing itself. The approach's effectiveness was validated by securing first place in a competitive challenge, supporting input-space standardization as a robust and practical solution for multi-institutional medical imaging.
>
---
#### [new 177] Salsa as a Nonverbal Embodied Language -- The CoMPAS3D Dataset and Benchmarks
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于人机交互与动作生成任务，旨在解决人机共舞中的非语言沟通问题。论文构建了大规模即兴萨尔萨舞数据集CoMPAS3D，包含动作标注与舞者水平信息，提出SalsaAgent模型，实现人机协同舞蹈生成，推动具身智能在社交互动与创意动作生成中的研究。**

- **链接: [http://arxiv.org/pdf/2507.19684v1](http://arxiv.org/pdf/2507.19684v1)**

> **作者:** Bermet Burkanova; Payam Jome Yazdian; Chuxuan Zhang; Trinity Evans; Paige Tuttösí; Angelica Lim
>
> **备注:** https://rosielab.github.io/compas3d
>
> **摘要:** Imagine a humanoid that can safely and creatively dance with a human, adapting to its partner's proficiency, using haptic signaling as a primary form of communication. While today's AI systems excel at text or voice-based interaction with large language models, human communication extends far beyond text-it includes embodied movement, timing, and physical coordination. Modeling coupled interaction between two agents poses a formidable challenge: it is continuous, bidirectionally reactive, and shaped by individual variation. We present CoMPAS3D, the largest and most diverse motion capture dataset of improvised salsa dancing, designed as a challenging testbed for interactive, expressive humanoid AI. The dataset includes 3 hours of leader-follower salsa dances performed by 18 dancers spanning beginner, intermediate, and professional skill levels. For the first time, we provide fine-grained salsa expert annotations, covering over 2,800 move segments, including move types, combinations, execution errors and stylistic elements. We draw analogies between partner dance communication and natural language, evaluating CoMPAS3D on two benchmark tasks for synthetic humans that parallel key problems in spoken language and dialogue processing: leader or follower generation with proficiency levels (speaker or listener synthesis), and duet (conversation) generation. Towards a long-term goal of partner dance with humans, we release the dataset, annotations, and code, along with a multitask SalsaAgent model capable of performing all benchmark tasks, alongside additional baselines to encourage research in socially interactive embodied AI and creative, expressive humanoid motion generation.
>
---
#### [new 178] WEEP: A Differentiable Nonconvex Sparse Regularizer via Weakly-Convex Envelope
- **分类: cs.LG; cs.CV**

- **简介: 论文提出WEEP，一种可微分的非凸稀疏正则化方法，用于信号处理中的稀疏信号恢复和特征提取。解决传统强稀疏性惩罚不可导与梯度优化器不兼容的问题，兼顾统计性能与计算可行性。**

- **链接: [http://arxiv.org/pdf/2507.20447v1](http://arxiv.org/pdf/2507.20447v1)**

> **作者:** Takanobu Furuhashi; Hidekata Hontani; Tatsuya Yokota
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Sparse regularization is fundamental in signal processing for efficient signal recovery and feature extraction. However, it faces a fundamental dilemma: the most powerful sparsity-inducing penalties are often non-differentiable, conflicting with gradient-based optimizers that dominate the field. We introduce WEEP (Weakly-convex Envelope of Piecewise Penalty), a novel, fully differentiable sparse regularizer derived from the weakly-convex envelope framework. WEEP provides strong, unbiased sparsity while maintaining full differentiability and L-smoothness, making it natively compatible with any gradient-based optimizer. This resolves the conflict between statistical performance and computational tractability. We demonstrate superior performance compared to the L1-norm and other established non-convex sparse regularizers on challenging signal and image denoising tasks.
>
---
#### [new 179] A Machine Learning Framework for Predicting Microphysical Properties of Ice Crystals from Cloud Particle Imagery
- **分类: physics.ao-ph; cs.CV; cs.LG; physics.geo-ph**

- **简介: 该论文属于机器学习与大气科学交叉任务，旨在解决冰晶微物理属性难以测量的问题。作者通过生成合成冰晶数据，训练ML模型，从二维图像预测冰晶的三维属性，如有效密度、表面积和子弹数，验证了模型的高准确率，并评估了双视角成像的改进效果。**

- **链接: [http://arxiv.org/pdf/2507.19759v1](http://arxiv.org/pdf/2507.19759v1)**

> **作者:** Joseph Ko; Jerry Harrington; Kara Sulia; Vanessa Przybylo; Marcus van Lier-Walqui; Kara Lamb
>
> **摘要:** The microphysical properties of ice crystals are important because they significantly alter the radiative properties and spatiotemporal distributions of clouds, which in turn strongly affect Earth's climate. However, it is challenging to measure key properties of ice crystals, such as mass or morphological features. Here, we present a framework for predicting three-dimensional (3D) microphysical properties of ice crystals from in situ two-dimensional (2D) imagery. First, we computationally generate synthetic ice crystals using 3D modeling software along with geometric parameters estimated from the 2021 Ice Cryo-Encapsulation Balloon (ICEBall) field campaign. Then, we use synthetic crystals to train machine learning (ML) models to predict effective density ($\rho_{e}$), effective surface area ($A_e$), and number of bullets ($N_b$) from synthetic rosette imagery. When tested on unseen synthetic images, we find that our ML models can predict microphysical properties with high accuracy. For $\rho_{e}$ and $A_e$, respectively, our best-performing single view models achieved $R^2$ values of 0.99 and 0.98. For $N_b$, our best single view model achieved a balanced accuracy and F1 score of 0.91. We also quantify the marginal prediction improvements from incorporating a second view. A stereo view ResNet-18 model reduced RMSE by 40% for both $\rho_e$ and $A_e$, relative to a single view ResNet-18 model. For $N_b$, we find that a stereo view ResNet-18 model improved the F1 score by 8%. This work provides a novel ML-driven framework for estimating ice microphysical properties from in situ imagery, which will allow for downstream constraints on microphysical parameterizations, such as the mass-size relationship.
>
---
#### [new 180] ConSeg: Contextual Backdoor Attack Against Semantic Segmentation
- **分类: cs.CR; cs.CV**

- **简介: 论文属于语义分割任务，旨在解决模型易受后门攻击的问题。作者提出ConSeg攻击方法，利用上下文信息增强攻击效果，通过模拟目标类的上下文在受害区域重建该关系，提高攻击成功率，并验证其有效性及防御抵抗能力。**

- **链接: [http://arxiv.org/pdf/2507.19905v1](http://arxiv.org/pdf/2507.19905v1)**

> **作者:** Bilal Hussain Abbasi; Zirui Gong; Yanjun Zhang; Shang Gao; Antonio Robles-Kelly; Leo Zhang
>
> **摘要:** Despite significant advancements in computer vision, semantic segmentation models may be susceptible to backdoor attacks. These attacks, involving hidden triggers, aim to cause the models to misclassify instances of the victim class as the target class when triggers are present, posing serious threats to the reliability of these models. To further explore the field of backdoor attacks against semantic segmentation, in this paper, we propose a simple yet effective backdoor attack called Contextual Segmentation Backdoor Attack (ConSeg). ConSeg leverages the contextual information inherent in semantic segmentation models to enhance backdoor performance. Our method is motivated by an intriguing observation, i.e., when the target class is set as the `co-occurring' class of the victim class, the victim class can be more easily `mis-segmented'. Building upon this insight, ConSeg mimics the contextual information of the target class and rebuilds it in the victim region to establish the contextual relationship between the target class and the victim class, making the attack easier. Our experiments reveal that ConSeg achieves improvements in Attack Success Rate (ASR) with increases of 15.55\%, compared to existing methods, while exhibiting resilience against state-of-the-art backdoor defenses.
>
---
#### [new 181] Neural Shell Texture Splatting: More Details and Fewer Primitives
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于三维重建与视图合成任务，旨在解决高保真重建需大量高斯图元的问题。通过引入神经壳纹理，分离几何与外观表示，减少图元数量，同时保持细节与效率。**

- **链接: [http://arxiv.org/pdf/2507.20200v1](http://arxiv.org/pdf/2507.20200v1)**

> **作者:** Xin Zhang; Anpei Chen; Jincheng Xiong; Pinxuan Dai; Yujun Shen; Weiwei Xu
>
> **摘要:** Gaussian splatting techniques have shown promising results in novel view synthesis, achieving high fidelity and efficiency. However, their high reconstruction quality comes at the cost of requiring a large number of primitives. We identify this issue as stemming from the entanglement of geometry and appearance in Gaussian Splatting. To address this, we introduce a neural shell texture, a global representation that encodes texture information around the surface. We use Gaussian primitives as both a geometric representation and texture field samplers, efficiently splatting texture features into image space. Our evaluation demonstrates that this disentanglement enables high parameter efficiency, fine texture detail reconstruction, and easy textured mesh extraction, all while using significantly fewer primitives.
>
---
#### [new 182] Rep-MTL: Unleashing the Power of Representation-level Task Saliency for Multi-Task Learning
- **分类: cs.LG; cs.CV**

- **简介: 论文提出Rep-MTL，属于多任务学习（MTL）领域，旨在解决任务冲突与负迁移问题。通过分析表示层的任务显著性，量化任务间交互，引入熵惩罚与跨任务对齐策略，提升任务互补性，实现更有效的知识共享与独立任务训练平衡。**

- **链接: [http://arxiv.org/pdf/2507.21049v1](http://arxiv.org/pdf/2507.21049v1)**

> **作者:** Zedong Wang; Siyuan Li; Dan Xu
>
> **备注:** ICCV 2025 (Highlight). Project page: https://jacky1128.github.io/RepMTL/
>
> **摘要:** Despite the promise of Multi-Task Learning in leveraging complementary knowledge across tasks, existing multi-task optimization (MTO) techniques remain fixated on resolving conflicts via optimizer-centric loss scaling and gradient manipulation strategies, yet fail to deliver consistent gains. In this paper, we argue that the shared representation space, where task interactions naturally occur, offers rich information and potential for operations complementary to existing optimizers, especially for facilitating the inter-task complementarity, which is rarely explored in MTO. This intuition leads to Rep-MTL, which exploits the representation-level task saliency to quantify interactions between task-specific optimization and shared representation learning. By steering these saliencies through entropy-based penalization and sample-wise cross-task alignment, Rep-MTL aims to mitigate negative transfer by maintaining the effective training of individual tasks instead pure conflict-solving, while explicitly promoting complementary information sharing. Experiments are conducted on four challenging MTL benchmarks covering both task-shift and domain-shift scenarios. The results show that Rep-MTL, even paired with the basic equal weighting policy, achieves competitive performance gains with favorable efficiency. Beyond standard performance metrics, Power Law exponent analysis demonstrates Rep-MTL's efficacy in balancing task-specific learning and cross-task sharing. The project page is available at HERE.
>
---
#### [new 183] A Multi-Agent System for Information Extraction from the Chemical Literature
- **分类: cs.AI; cs.CV; cs.MA**

- **简介: 该论文属于信息抽取任务，旨在解决从化学文献中自动提取信息的问题。由于文献中化学信息具有多模态和样式多变的特点，现有方法受限。论文提出了一种基于多模态大语言模型的多智能体系统，将复杂任务分解为子任务并协同解决。系统在多个关键子任务上表现出色，显著提升了化学信息抽取的自动化水平。**

- **链接: [http://arxiv.org/pdf/2507.20230v1](http://arxiv.org/pdf/2507.20230v1)**

> **作者:** Yufan Chen; Ching Ting Leung; Bowen Yu; Jianwei Sun; Yong Huang; Linyan Li; Hao Chen; Hanyu Gao
>
> **摘要:** To fully expedite AI-powered chemical research, high-quality chemical databases are the cornerstone. Automatic extraction of chemical information from the literature is essential for constructing reaction databases, but it is currently limited by the multimodality and style variability of chemical information. In this work, we developed a multimodal large language model (MLLM)-based multi-agent system for automatic chemical information extraction. We used the MLLM's strong reasoning capability to understand the structure of complex chemical graphics, decompose the extraction task into sub-tasks and coordinate a set of specialized agents to solve them. Our system achieved an F1 score of 80.8% on a benchmark dataset of complex chemical reaction graphics from the literature, surpassing the previous state-of-the-art model (F1 score: 35.6%) by a significant margin. Additionally, it demonstrated consistent improvements in key sub-tasks, including molecular image recognition, reaction image parsing, named entity recognition and text-based reaction extraction. This work is a critical step toward automated chemical information extraction into structured datasets, which will be a strong promoter of AI-driven chemical research.
>
---
#### [new 184] Investigating Structural Pruning and Recovery Techniques for Compressing Multimodal Large Language Models: An Empirical Study
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于模型压缩任务，旨在解决多模态大语言模型（MLLMs）因计算和内存需求高而难以部署的问题。通过结构剪枝与恢复训练方法，如逐层和逐宽剪枝，结合微调和知识蒸馏，实现高效压缩。实验表明，在少量数据下仍可保持高性能。**

- **链接: [http://arxiv.org/pdf/2507.20749v1](http://arxiv.org/pdf/2507.20749v1)**

> **作者:** Yiran Huang; Lukas Thede; Massimiliano Mancini; Wenjia Xu; Zeynep Akata
>
> **备注:** Accepted at GCPR 2025
>
> **摘要:** While Multimodal Large Language Models (MLLMs) demonstrate impressive capabilities, their substantial computational and memory requirements pose significant barriers to practical deployment. Current parameter reduction techniques primarily involve training MLLMs from Small Language Models (SLMs), but these methods offer limited flexibility and remain computationally intensive. To address this gap, we propose to directly compress existing MLLMs through structural pruning combined with efficient recovery training. Specifically, we investigate two structural pruning paradigms--layerwise and widthwise pruning--applied to the language model backbone of MLLMs, alongside supervised finetuning and knowledge distillation. Additionally, we assess the feasibility of conducting recovery training with only a small fraction of the available data. Our results show that widthwise pruning generally maintains better performance in low-resource scenarios with limited computational resources or insufficient finetuning data. As for the recovery training, finetuning only the multimodal projector is sufficient at small compression levels (< 20%). Furthermore, a combination of supervised finetuning and hidden-state distillation yields optimal recovery across various pruning levels. Notably, effective recovery can be achieved with as little as 5% of the original training data, while retaining over 95% of the original performance. Through empirical study on two representative MLLMs, i.e., LLaVA-v1.5-7B and Bunny-v1.0-3B, this study offers actionable insights for practitioners aiming to compress MLLMs effectively without extensive computation resources or sufficient data.
>
---
#### [new 185] Methods for the Segmentation of Reticular Structures Using 3D LiDAR Data: A Comparative Evaluation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于点云分割任务，旨在解决攀爬机器人在网状结构中自主导航时的可导航面检测问题。论文提出了基于特征分解的分析算法和多种深度学习模型（如PointNet、PointTransformerV3）进行二值分割，并在3D点云数据上进行比较评估，结果显示PointTransformerV3分割精度最高，而分析算法在调参和效率上更具优势。**

- **链接: [http://arxiv.org/pdf/2507.20589v1](http://arxiv.org/pdf/2507.20589v1)**

> **作者:** Francisco J. Soler Mora; Adrián Peidró Vidal; Marc Fabregat-Jaén; Luis Payá Castelló; Óscar Reinoso García
>
> **摘要:** Reticular structures form the backbone of major infrastructure like bridges, pylons, and airports, but their inspection and maintenance are costly and hazardous, often requiring human intervention. While prior research has focused on fault detection via images or robotic platform design, the autonomous navigation of robots within these structures is less explored. This study addresses that gap by proposing methods to detect navigable surfaces in truss structures, enhancing the autonomy of climbing robots. The paper introduces several approaches for binary segmentation of navigable surfaces versus background from 3D point clouds of metallic trusses. These methods fall into two categories: analytical algorithms and deep learning models. The analytical approach features a custom algorithm that segments structures by analyzing the eigendecomposition of planar patches in the point cloud. In parallel, advanced deep learning models PointNet, PointNet++, MinkUNet34C, and PointTransformerV3 are trained and evaluated for the same task. Comparative analysis shows that the analytical algorithm offers easier parameter tuning and performance comparable to deep learning models, which, while more computationally intensive, excel in segmentation accuracy. Notably, PointTransformerV3 achieves a Mean Intersection Over Union (mIoU) of about 97%. The study demonstrates the promise of both analytical and deep learning methods for improving autonomous navigation in complex truss environments. The results highlight the trade-offs between computational efficiency and segmentation performance, providing valuable guidance for future research and practical applications in autonomous infrastructure inspection and maintenance.
>
---
#### [new 186] ChoreoMuse: Robust Music-to-Dance Video Generation with Style Transfer and Beat-Adherent Motion
- **分类: cs.GR; cs.AI; cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于音乐驱动舞蹈视频生成任务，旨在解决现有方法在音乐节奏与风格适配性、视频质量及个性化表达上的不足。论文提出ChoreoMuse框架，结合风格迁移与节拍对齐技术，实现高质量、可控风格的舞蹈视频生成。**

- **链接: [http://arxiv.org/pdf/2507.19836v1](http://arxiv.org/pdf/2507.19836v1)**

> **作者:** Xuanchen Wang; Heng Wang; Weidong Cai
>
> **备注:** 10 pages, 5 figures, accepted by the 33rd ACM International Conference on Multimedia (ACM MM 2025), demo page: https://choreomuse.github.io
>
> **摘要:** Modern artistic productions increasingly demand automated choreography generation that adapts to diverse musical styles and individual dancer characteristics. Existing approaches often fail to produce high-quality dance videos that harmonize with both musical rhythm and user-defined choreography styles, limiting their applicability in real-world creative contexts. To address this gap, we introduce ChoreoMuse, a diffusion-based framework that uses SMPL format parameters and their variation version as intermediaries between music and video generation, thereby overcoming the usual constraints imposed by video resolution. Critically, ChoreoMuse supports style-controllable, high-fidelity dance video generation across diverse musical genres and individual dancer characteristics, including the flexibility to handle any reference individual at any resolution. Our method employs a novel music encoder MotionTune to capture motion cues from audio, ensuring that the generated choreography closely follows the beat and expressive qualities of the input music. To quantitatively evaluate how well the generated dances match both musical and choreographic styles, we introduce two new metrics that measure alignment with the intended stylistic cues. Extensive experiments confirm that ChoreoMuse achieves state-of-the-art performance across multiple dimensions, including video quality, beat alignment, dance diversity, and style adherence, demonstrating its potential as a robust solution for a wide range of creative applications. Video results can be found on our project page: https://choreomuse.github.io.
>
---
#### [new 187] Digital and Robotic Twinning for Validation of Proximity Operations and Formation Flying
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于航天器导航与控制验证任务，旨在解决RPO和FF任务中GNC系统难以验证的问题。工作内容是构建数字与机器人双胞胎框架，结合多个测试平台，实现软硬件在环测试，验证系统性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.20034v1](http://arxiv.org/pdf/2507.20034v1)**

> **作者:** Aviad Golan; Gregory Zin; Zahra Ahmed; Emily Bates; Toby Bell; Pol Francesch Huc; Samuel Y. W. Low; Juergen Bosse; Simone D'Amico
>
> **备注:** 23 pages, 12 figures. 2025 Astrodynamics Specialist Conference
>
> **摘要:** In spacecraft Rendezvous, Proximity Operations (RPO), and Formation Flying (FF), the Guidance Navigation and Control (GNC) system is safety-critical and must meet strict performance requirements. However, validating such systems is challenging due to the complexity of the space environment, necessitating a verification and validation (V&V) process that bridges simulation and real-world behavior. The key contribution of this paper is a unified, end-to-end digital and robotic twinning framework that enables software- and hardware-in-the-loop testing for multi-modal GNC systems. The robotic twin includes three testbeds at Stanford's Space Rendezvous Laboratory (SLAB): the GNSS and Radiofrequency Autonomous Navigation Testbed for Distributed Space Systems (GRAND) to validate RF-based navigation techniques, and the Testbed for Rendezvous and Optical Navigation (TRON) and Optical Stimulator (OS) to validate vision-based methods. The test article for this work is an integrated multi-modal GNC software stack for RPO and FF developed at SLAB. This paper introduces the hybrid framework and summarizes calibration and error characterization for the robotic twin. Then, the GNC stack's performance and robustness is characterized using the integrated digital and robotic twinning pipeline for a full-range RPO mission scenario in Low-Earth Orbit (LEO). The results shown in the paper demonstrate consistency between digital and robotic twins, validating the hybrid twinning pipeline as a reliable framework for realistic assessment and verification of GNC systems.
>
---
#### [new 188] Multipath Interference Suppression in Indirect Time-of-Flight Imaging via a Novel Compressed Sensing Framework
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于深度成像任务，旨在提升间接飞行时间（iToF）系统的深度重建精度和多目标分离能力。论文提出了一种新的压缩感知方法，通过多相移和窄占空比连续波构建传感矩阵，并结合K-Means聚类优化正交匹配追踪（OMP）过程，有效抑制多路径干扰，提升重建效果。**

- **链接: [http://arxiv.org/pdf/2507.19546v1](http://arxiv.org/pdf/2507.19546v1)**

> **作者:** Yansong Du; Yutong Deng; Yuting Zhou; Feiyu Jiao; Bangyao Wang; Zhancong Xu; Zhaoxiang Jiang; Xun Guan
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** We propose a novel compressed sensing method to improve the depth reconstruction accuracy and multi-target separation capability of indirect Time-of-Flight (iToF) systems. Unlike traditional approaches that rely on hardware modifications, complex modulation, or cumbersome data-driven reconstruction, our method operates with a single modulation frequency and constructs the sensing matrix using multiple phase shifts and narrow-duty-cycle continuous waves. During matrix construction, we further account for pixel-wise range variation caused by lens distortion, making the sensing matrix better aligned with actual modulation response characteristics. To enhance sparse recovery, we apply K-Means clustering to the distance response dictionary and constrain atom selection within each cluster during the OMP process, which effectively reduces the search space and improves solution stability. Experimental results demonstrate that the proposed method outperforms traditional approaches in both reconstruction accuracy and robustness, without requiring any additional hardware changes.
>
---
#### [new 189] ChartGen: Scaling Chart Understanding Via Code-Guided Synthetic Chart Generation
- **分类: cs.HC; cs.AI; cs.CV**

- **简介: 该论文属于图表理解和代码生成任务，旨在解决现有数据集中图表与代码对应不足的问题。作者提出了ChartGen，一个自动化合成图表生成管道，通过视觉-语言模型和代码大模型生成222.5K个图表-代码对，并构建了一个包含27种图表类型、11种绘图库的开源数据集，用于评估图表到代码的理解能力。**

- **链接: [http://arxiv.org/pdf/2507.19492v1](http://arxiv.org/pdf/2507.19492v1)**

> **作者:** Jovana Kondic; Pengyuan Li; Dhiraj Joshi; Zexue He; Shafiq Abedin; Jennifer Sun; Ben Wiesel; Eli Schwartz; Ahmed Nassar; Bo Wu; Assaf Arbelle; Aude Oliva; Dan Gutfreund; Leonid Karlinsky; Rogerio Feris
>
> **摘要:** Chart-to-code reconstruction -- the task of recovering executable plotting scripts from chart images -- provides important insights into a model's ability to ground data visualizations in precise, machine-readable form. Yet many existing multimodal benchmarks largely focus primarily on answering questions about charts or summarizing them. To bridge this gap, we present ChartGen, a fully-automated pipeline for code-guided synthetic chart generation. Starting from seed chart images, ChartGen (i) prompts a vision-language model (VLM) to reconstruct each image into a python script, and (ii) iteratively augments that script with a code-oriented large language model (LLM). Using ChartGen, we create 222.5K unique chart-image code pairs from 13K seed chart images, and present an open-source synthetic chart dataset covering 27 chart types, 11 plotting libraries, and multiple data modalities (image, code, text, CSV, DocTags). From this corpus, we curate a held-out chart-to-code evaluation subset of 4.3K chart image-code pairs, and evaluate six open-weight VLMs (3B - 26B parameters), highlighting substantial room for progress. We release the pipeline, prompts, and the dataset to help accelerate efforts towards robust chart understanding and vision-conditioned code generation: https://github.com/SD122025/ChartGen/
>
---
#### [new 190] Review of Deep Learning Applications to Structural Proteomics Enabled by Cryogenic Electron Microscopy and Tomography
- **分类: q-bio.QM; cs.CV; cs.LG**

- **简介: 该论文综述了深度学习在冷冻电镜和断层扫描结构蛋白质组学中的应用。任务是提升结构解析效率与质量，解决低信噪比、取向偏差和缺失楔等问题。工作包括AI在粒子挑选、去噪、模型构建等流程的应用与优化。**

- **链接: [http://arxiv.org/pdf/2507.19565v1](http://arxiv.org/pdf/2507.19565v1)**

> **作者:** Brady K. Zhou; Jason J. Hu; Jane K. J. Lee; Z. Hong Zhou; Demetri Terzopoulos
>
> **备注:** 16 pages
>
> **摘要:** The past decade's "cryoEM revolution" has produced exponential growth in high-resolution structural data through advances in cryogenic electron microscopy (cryoEM) and tomography (cryoET). Deep learning integration into structural proteomics workflows addresses longstanding challenges including low signal-to-noise ratios, preferred orientation artifacts, and missing-wedge problems that historically limited efficiency and scalability. This review examines AI applications across the entire cryoEM pipeline, from automated particle picking using convolutional neural networks (Topaz, crYOLO, CryoSegNet) to computational solutions for preferred orientation bias (spIsoNet, cryoPROS) and advanced denoising algorithms (Topaz-Denoise). In cryoET, tools like IsoNet employ U-Net architectures for simultaneous missing-wedge correction and noise reduction, while TomoNet streamlines subtomogram averaging through AI-driven particle detection. The workflow culminates with automated atomic model building using sophisticated tools like ModelAngelo, DeepTracer, and CryoREAD that translate density maps into interpretable biological structures. These AI-enhanced approaches have achieved near-atomic resolution reconstructions with minimal manual intervention, resolved previously intractable datasets suffering from severe orientation bias, and enabled successful application to diverse biological systems from HIV virus-like particles to in situ ribosomal complexes. As deep learning evolves, particularly with large language models and vision transformers, the future promises sophisticated automation and accessibility in structural biology, potentially revolutionizing our understanding of macromolecular architecture and function.
>
---
#### [new 191] Rainbow Noise: Stress-Testing Multimodal Harmful-Meme Detectors on LGBTQ Content
- **分类: cs.CY; cs.AI; cs.CV**

- **简介: 该论文属于多模态内容安全任务，旨在解决针对LGBTQ群体的仇恨模因检测中的鲁棒性问题。作者构建了一个包含文本攻击与图像干扰的测试基准，在PrideMM数据集上评估现有模型，并提出轻量级文本去噪适配器（TDA）提升模型鲁棒性。实验表明，加入TDA的MemeBLIP2表现最优，揭示了文本依赖性及架构、预训练数据对鲁棒性的影响。**

- **链接: [http://arxiv.org/pdf/2507.19551v1](http://arxiv.org/pdf/2507.19551v1)**

> **作者:** Ran Tong; Songtao Wei; Jiaqi Liu; Lanruo Wang
>
> **备注:** 9 pages, 1 figure
>
> **摘要:** Hateful memes aimed at LGBTQ\,+ communities often evade detection by tweaking either the caption, the image, or both. We build the first robustness benchmark for this setting, pairing four realistic caption attacks with three canonical image corruptions and testing all combinations on the PrideMM dataset. Two state-of-the-art detectors, MemeCLIP and MemeBLIP2, serve as case studies, and we introduce a lightweight \textbf{Text Denoising Adapter (TDA)} to enhance the latter's resilience. Across the grid, MemeCLIP degrades more gently, while MemeBLIP2 is particularly sensitive to the caption edits that disrupt its language processing. However, the addition of the TDA not only remedies this weakness but makes MemeBLIP2 the most robust model overall. Ablations reveal that all systems lean heavily on text, but architectural choices and pre-training data significantly impact robustness. Our benchmark exposes where current multimodal safety models crack and demonstrates that targeted, lightweight modules like the TDA offer a powerful path towards stronger defences.
>
---
#### [new 192] Taking Language Embedded 3D Gaussian Splatting into the Wild
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D场景理解与重建任务，旨在解决从无约束照片集合中实现建筑结构的沉浸式理解问题。作者扩展了语言嵌入的3D高斯点绘技术，提出了一种新的开放词汇场景理解框架，并引入新数据集PT-OVS进行评估，实现了更准确的开放词汇分割与多种应用。**

- **链接: [http://arxiv.org/pdf/2507.19830v1](http://arxiv.org/pdf/2507.19830v1)**

> **作者:** Yuze Wang; Yue Qi
>
> **备注:** Visit our project page at https://yuzewang1998.github.io/takinglangsplatw/
>
> **摘要:** Recent advances in leveraging large-scale Internet photo collections for 3D reconstruction have enabled immersive virtual exploration of landmarks and historic sites worldwide. However, little attention has been given to the immersive understanding of architectural styles and structural knowledge, which remains largely confined to browsing static text-image pairs. Therefore, can we draw inspiration from 3D in-the-wild reconstruction techniques and use unconstrained photo collections to create an immersive approach for understanding the 3D structure of architectural components? To this end, we extend language embedded 3D Gaussian splatting (3DGS) and propose a novel framework for open-vocabulary scene understanding from unconstrained photo collections. Specifically, we first render multiple appearance images from the same viewpoint as the unconstrained image with the reconstructed radiance field, then extract multi-appearance CLIP features and two types of language feature uncertainty maps-transient and appearance uncertainty-derived from the multi-appearance features to guide the subsequent optimization process. Next, we propose a transient uncertainty-aware autoencoder, a multi-appearance language field 3DGS representation, and a post-ensemble strategy to effectively compress, learn, and fuse language features from multiple appearances. Finally, to quantitatively evaluate our method, we introduce PT-OVS, a new benchmark dataset for assessing open-vocabulary segmentation performance on unconstrained photo collections. Experimental results show that our method outperforms existing methods, delivering accurate open-vocabulary segmentation and enabling applications such as interactive roaming with open-vocabulary queries, architectural style pattern recognition, and 3D scene editing.
>
---
#### [new 193] MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks
- **分类: cs.CL; cs.AI; cs.CV; cs.SD**

- **简介: 该论文提出了MCIF，一个跨语言、多模态的指令跟随评测基准，旨在评估大语言模型在多语言、多模态及长短上下文中的指令理解能力。现有评测集在语言、模态和上下文长度方面存在局限，MCIF填补了这一空白，支持英文、德文、意大利文和中文，包含语音、视觉和文本三种模态，适用于科学讲座场景。**

- **链接: [http://arxiv.org/pdf/2507.19634v1](http://arxiv.org/pdf/2507.19634v1)**

> **作者:** Sara Papi; Maike Züfle; Marco Gaido; Beatrice Savoldi; Danni Liu; Ioannis Douros; Luisa Bentivogli; Jan Niehues
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in large language models have catalyzed the development of multimodal LLMs (MLLMs) that integrate text, speech, and vision within unified frameworks. As MLLMs evolve from narrow, monolingual, task-specific systems to general-purpose instruction-following models, a key frontier lies in evaluating their multilingual and multimodal capabilities over both long and short contexts. However, existing benchmarks fall short in evaluating these dimensions jointly: they are often limited to English, mostly focus on one single modality at a time, rely on short-form contexts, or lack human annotations -- hindering comprehensive assessment of model performance across languages, modalities, and task complexity. To address these gaps, we introduce MCIF (Multimodal Crosslingual Instruction Following), the first multilingual human-annotated benchmark based on scientific talks that is designed to evaluate instruction-following in crosslingual, multimodal settings over both short- and long-form inputs. MCIF spans three core modalities -- speech, vision, and text -- and four diverse languages (English, German, Italian, and Chinese), enabling a comprehensive evaluation of MLLMs' abilities to interpret instructions across languages and combine them with multimodal contextual information. MCIF is released under a CC-BY 4.0 license to encourage open research and progress in MLLMs development.
>
---
#### [new 194] SpecBPP: A Self-Supervised Learning Approach for Hyperspectral Representation and Soil Organic Carbon Estimation
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于遥感与土壤有机碳估算任务，旨在解决高光谱影像表示学习不足的问题。作者提出SpecBPP，一种基于光谱带排列预测的自监督学习框架，利用高光谱数据的谱段连续性，通过恢复打乱顺序的谱段来学习全局光谱特征。该方法在EnMAP卫星数据上实现了优于现有方法的土壤有机碳估算性能。**

- **链接: [http://arxiv.org/pdf/2507.19781v1](http://arxiv.org/pdf/2507.19781v1)**

> **作者:** Daniel La'ah Ayuba; Jean-Yves Guillemaut; Belen Marti-Cardona; Oscar Mendez Maldonado
>
> **摘要:** Self-supervised learning has revolutionized representation learning in vision and language, but remains underexplored for hyperspectral imagery (HSI), where the sequential structure of spectral bands offers unique opportunities. In this work, we propose Spectral Band Permutation Prediction (SpecBPP), a novel self-supervised learning framework that leverages the inherent spectral continuity in HSI. Instead of reconstructing masked bands, SpecBPP challenges a model to recover the correct order of shuffled spectral segments, encouraging global spectral understanding. We implement a curriculum-based training strategy that progressively increases permutation difficulty to manage the factorial complexity of the permutation space. Applied to Soil Organic Carbon (SOC) estimation using EnMAP satellite data, our method achieves state-of-the-art results, outperforming both masked autoencoder (MAE) and joint-embedding predictive (JEPA) baselines. Fine-tuned on limited labeled samples, our model yields an $R^2$ of 0.9456, RMSE of 1.1053%, and RPD of 4.19, significantly surpassing traditional and self-supervised benchmarks. Our results demonstrate that spectral order prediction is a powerful pretext task for hyperspectral understanding, opening new avenues for scientific representation learning in remote sensing and beyond.
>
---
#### [new 195] A Metabolic-Imaging Integrated Model for Prognostic Prediction in Colorectal Liver Metastases
- **分类: eess.IV; cs.CV; cs.LG; q-bio.QM**

- **简介: 该论文旨在开发一个结合代谢和影像特征的机器学习模型，用于预测结直肠癌肝转移（CRLM）术后复发风险。任务是解决传统模型准确性不足的问题，通过使用术前临床参数和CT影像特征，避免数据泄漏，提高预测可靠性和临床实用性。**

- **链接: [http://arxiv.org/pdf/2507.19734v1](http://arxiv.org/pdf/2507.19734v1)**

> **作者:** Qinlong Li; Pu Sun; Guanlin Zhu; Tianjiao Liang; Honggang QI
>
> **备注:** 8 pages,4 figues
>
> **摘要:** Prognostic evaluation in patients with colorectal liver metastases (CRLM) remains challenging due to suboptimal accuracy of conventional clinical models. This study developed and validated a robust machine learning model for predicting postoperative recurrence risk. Preliminary ensemble models achieved exceptionally high performance (AUC $>$ 0.98) but incorporated postoperative features, introducing data leakage risks. To enhance clinical applicability, we restricted input variables to preoperative baseline clinical parameters and radiomic features from contrast-enhanced CT imaging, specifically targeting recurrence prediction at 3, 6, and 12 months postoperatively. The 3-month recurrence prediction model demonstrated optimal performance with an AUC of 0.723 in cross-validation. Decision curve analysis revealed that across threshold probabilities of 0.55-0.95, the model consistently provided greater net benefit than "treat-all" or "treat-none" strategies, supporting its utility in postoperative surveillance and therapeutic decision-making. This study successfully developed a robust predictive model for early CRLM recurrence with confirmed clinical utility. Importantly, it highlights the critical risk of data leakage in clinical prognostic modeling and proposes a rigorous framework to mitigate this issue, enhancing model reliability and translational value in real-world settings.
>
---
#### [new 196] Model-Agnostic Gender Bias Control for Text-to-Image Generation via Sparse Autoencoder
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型中存在的性别偏见问题。通过提出SAE Debias方法，利用稀疏自编码器识别并抑制性别相关方向，从而生成更公平的结果。方法无需重新训练模型，具有模型无关性和可解释性。**

- **链接: [http://arxiv.org/pdf/2507.20973v1](http://arxiv.org/pdf/2507.20973v1)**

> **作者:** Chao Wu; Zhenyi Wang; Kangxian Xie; Naresh Kumar Devulapally; Vishnu Suresh Lokhande; Mingchen Gao
>
> **摘要:** Text-to-image (T2I) diffusion models often exhibit gender bias, particularly by generating stereotypical associations between professions and gendered subjects. This paper presents SAE Debias, a lightweight and model-agnostic framework for mitigating such bias in T2I generation. Unlike prior approaches that rely on CLIP-based filtering or prompt engineering, which often require model-specific adjustments and offer limited control, SAE Debias operates directly within the feature space without retraining or architectural modifications. By leveraging a k-sparse autoencoder pre-trained on a gender bias dataset, the method identifies gender-relevant directions within the sparse latent space, capturing professional stereotypes. Specifically, a biased direction per profession is constructed from sparse latents and suppressed during inference to steer generations toward more gender-balanced outputs. Trained only once, the sparse autoencoder provides a reusable debiasing direction, offering effective control and interpretable insight into biased subspaces. Extensive evaluations across multiple T2I models, including Stable Diffusion 1.4, 1.5, 2.1, and SDXL, demonstrate that SAE Debias substantially reduces gender bias while preserving generation quality. To the best of our knowledge, this is the first work to apply sparse autoencoders for identifying and intervening in gender bias within T2I models. These findings contribute toward building socially responsible generative AI, providing an interpretable and model-agnostic tool to support fairness in text-to-image generation.
>
---
#### [new 197] Complementarity-driven Representation Learning for Multi-modal Knowledge Graph Completion
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态知识图谱补全（MMKGC）任务，旨在解决多模态知识图中实体表示不平衡、模态互补性未被充分利用的问题。作者提出MoCME框架，包含利用模态互补性的CMKF模块和提升训练效果的EGNS机制，有效融合多模态信息，提升实体表示，取得最优性能。**

- **链接: [http://arxiv.org/pdf/2507.20620v1](http://arxiv.org/pdf/2507.20620v1)**

> **作者:** Lijian Li
>
> **摘要:** Multi-modal Knowledge Graph Completion (MMKGC) aims to uncover hidden world knowledge in multimodal knowledge graphs by leveraging both multimodal and structural entity information. However, the inherent imbalance in multimodal knowledge graphs, where modality distributions vary across entities, poses challenges in utilizing additional modality data for robust entity representation. Existing MMKGC methods typically rely on attention or gate-based fusion mechanisms but overlook complementarity contained in multi-modal data. In this paper, we propose a novel framework named Mixture of Complementary Modality Experts (MoCME), which consists of a Complementarity-guided Modality Knowledge Fusion (CMKF) module and an Entropy-guided Negative Sampling (EGNS) mechanism. The CMKF module exploits both intra-modal and inter-modal complementarity to fuse multi-view and multi-modal embeddings, enhancing representations of entities. Additionally, we introduce an Entropy-guided Negative Sampling mechanism to dynamically prioritize informative and uncertain negative samples to enhance training effectiveness and model robustness. Extensive experiments on five benchmark datasets demonstrate that our MoCME achieves state-of-the-art performance, surpassing existing approaches.
>
---
#### [new 198] Onboard Hyperspectral Super-Resolution with Deep Pushbroom Neural Network
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在提升卫星高光谱图像的空间分辨率。为实现在卫星上实时运行，论文提出了一种轻量级神经网络DPSR，采用因果记忆机制逐行处理图像，降低计算复杂度，并在低功耗硬件上实现实时性能。**

- **链接: [http://arxiv.org/pdf/2507.20765v1](http://arxiv.org/pdf/2507.20765v1)**

> **作者:** Davide Piccinini; Diego Valsesia; Enrico Magli
>
> **摘要:** Hyperspectral imagers on satellites obtain the fine spectral signatures essential for distinguishing one material from another at the expense of limited spatial resolution. Enhancing the latter is thus a desirable preprocessing step in order to further improve the detection capabilities offered by hyperspectral images on downstream tasks. At the same time, there is a growing interest towards deploying inference methods directly onboard of satellites, which calls for lightweight image super-resolution methods that can be run on the payload in real time. In this paper, we present a novel neural network design, called Deep Pushbroom Super-Resolution (DPSR) that matches the pushbroom acquisition of hyperspectral sensors by processing an image line by line in the along-track direction with a causal memory mechanism to exploit previously acquired lines. This design greatly limits memory requirements and computational complexity, achieving onboard real-time performance, i.e., the ability to super-resolve a line in the time it takes to acquire the next one, on low-power hardware. Experiments show that the quality of the super-resolved images is competitive or even outperforms state-of-the-art methods that are significantly more complex.
>
---
#### [new 199] CLoRA: Parameter-Efficient Continual Learning with Low-Rank Adaptation
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出CLoRA，属于类增量语义分割任务，旨在解决持续学习中灾难性遗忘和计算资源受限的问题。通过低秩适应方法，仅微调少量参数，实现高效模型更新，适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2507.19887v1](http://arxiv.org/pdf/2507.19887v1)**

> **作者:** Shishir Muralidhara; Didier Stricker; René Schuster
>
> **备注:** Accepted at CoLLAs 2025
>
> **摘要:** In the past, continual learning (CL) was mostly concerned with the problem of catastrophic forgetting in neural networks, that arises when incrementally learning a sequence of tasks. Current CL methods function within the confines of limited data access, without any restrictions imposed on computational resources. However, in real-world scenarios, the latter takes precedence as deployed systems are often computationally constrained. A major drawback of most CL methods is the need to retrain the entire model for each new task. The computational demands of retraining large models can be prohibitive, limiting the applicability of CL in environments with limited resources. Through CLoRA, we explore the applicability of Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning method for class-incremental semantic segmentation. CLoRA leverages a small set of parameters of the model and uses the same set for learning across all tasks. Results demonstrate the efficacy of CLoRA, achieving performance on par with and exceeding the baseline methods. We further evaluate CLoRA using NetScore, underscoring the need to factor in resource efficiency and evaluate CL methods beyond task performance. CLoRA significantly reduces the hardware requirements for training, making it well-suited for CL in resource-constrained environments after deployment.
>
---
#### [new 200] Text2Vis: A Challenging and Diverse Benchmark for Generating Multimodal Visualizations from Text
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于文本生成可视化任务，旨在解决缺乏全面评估基准的问题。作者构建了Text2Vis数据集，包含1,985个样本，涵盖20多种图表类型和复杂数据科学查询。他们提出跨模态框架提升生成效果，并开发自动化评估系统。论文工作推动了文本到可视化模型的发展与评估标准化。**

- **链接: [http://arxiv.org/pdf/2507.19969v1](http://arxiv.org/pdf/2507.19969v1)**

> **作者:** Mizanur Rahman; Md Tahmid Rahman Laskar; Shafiq Joty; Enamul Hoque
>
> **摘要:** Automated data visualization plays a crucial role in simplifying data interpretation, enhancing decision-making, and improving efficiency. While large language models (LLMs) have shown promise in generating visualizations from natural language, the absence of comprehensive benchmarks limits the rigorous evaluation of their capabilities. We introduce Text2Vis, a benchmark designed to assess text-to-visualization models, covering 20+ chart types and diverse data science queries, including trend analysis, correlation, outlier detection, and predictive analytics. It comprises 1,985 samples, each with a data table, natural language query, short answer, visualization code, and annotated charts. The queries involve complex reasoning, conversational turns, and dynamic data retrieval. We benchmark 11 open-source and closed-source models, revealing significant performance gaps, highlighting key challenges, and offering insights for future advancements. To close this gap, we propose the first cross-modal actor-critic agentic framework that jointly refines the textual answer and visualization code, increasing GPT-4o`s pass rate from 26% to 42% over the direct approach and improving chart quality. We also introduce an automated LLM-based evaluation framework that enables scalable assessment across thousands of samples without human annotation, measuring answer correctness, code execution success, visualization readability, and chart accuracy. We release Text2Vis at https://github.com/vis-nlp/Text2Vis.
>
---
#### [new 201] SkinDualGen: Prompt-Driven Diffusion for Simultaneous Image-Mask Generation in Skin Lesions
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像生成与分割任务，旨在解决皮肤病变数据稀缺和类别不平衡问题。作者基于Stable Diffusion-2.0模型，通过LoRA微调和多目标优化，实现根据文本描述同步生成高质量皮肤病变图像及其分割掩码。实验表明，合成数据提升了分类与分割模型的性能，显著改善了诊断准确性与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.19970v1](http://arxiv.org/pdf/2507.19970v1)**

> **作者:** Zhaobin Xu
>
> **摘要:** Medical image analysis plays a pivotal role in the early diagnosis of diseases such as skin lesions. However, the scarcity of data and the class imbalance significantly hinder the performance of deep learning models. We propose a novel method that leverages the pretrained Stable Diffusion-2.0 model to generate high-quality synthetic skin lesion images and corresponding segmentation masks. This approach augments training datasets for classification and segmentation tasks. We adapt Stable Diffusion-2.0 through domain-specific Low-Rank Adaptation (LoRA) fine-tuning and joint optimization of multi-objective loss functions, enabling the model to simultaneously generate clinically relevant images and segmentation masks conditioned on textual descriptions in a single step. Experimental results show that the generated images, validated by FID scores, closely resemble real images in quality. A hybrid dataset combining real and synthetic data markedly enhances the performance of classification and segmentation models, achieving substantial improvements in accuracy and F1-score of 8% to 15%, with additional positive gains in other key metrics such as the Dice coefficient and IoU. Our approach offers a scalable solution to address the challenges of medical imaging data, contributing to improved accuracy and reliability in diagnosing rare diseases.
>
---
#### [new 202] Humanoid Occupancy: Enabling A Generalized Multimodal Occupancy Perception System on Humanoid Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于环境感知任务，旨在解决人形机器人在复杂环境中实现全面环境理解的问题。论文提出了Humanoid Occupancy系统，结合多模态融合技术与传感器布局策略，生成语义与几何信息融合的网格化占据输出，并构建了全景占据数据集，为人形机器人环境感知提供技术基础与数据支持。**

- **链接: [http://arxiv.org/pdf/2507.20217v1](http://arxiv.org/pdf/2507.20217v1)**

> **作者:** Wei Cui; Haoyu Wang; Wenkang Qin; Yijie Guo; Gang Han; Wen Zhao; Jiahang Cao; Zhang Zhang; Jiaru Zhong; Jingkai Sun; Pihai Sun; Shuai Shi; Botuo Jiang; Jiahao Ma; Jiaxu Wang; Hao Cheng; Zhichao Liu; Yang Wang; Zheng Zhu; Guan Huang; Jian Tang; Qiang Zhang
>
> **备注:** Tech Report
>
> **摘要:** Humanoid robot technology is advancing rapidly, with manufacturers introducing diverse heterogeneous visual perception modules tailored to specific scenarios. Among various perception paradigms, occupancy-based representation has become widely recognized as particularly suitable for humanoid robots, as it provides both rich semantic and 3D geometric information essential for comprehensive environmental understanding. In this work, we present Humanoid Occupancy, a generalized multimodal occupancy perception system that integrates hardware and software components, data acquisition devices, and a dedicated annotation pipeline. Our framework employs advanced multi-modal fusion techniques to generate grid-based occupancy outputs encoding both occupancy status and semantic labels, thereby enabling holistic environmental understanding for downstream tasks such as task planning and navigation. To address the unique challenges of humanoid robots, we overcome issues such as kinematic interference and occlusion, and establish an effective sensor layout strategy. Furthermore, we have developed the first panoramic occupancy dataset specifically for humanoid robots, offering a valuable benchmark and resource for future research and development in this domain. The network architecture incorporates multi-modal feature fusion and temporal information integration to ensure robust perception. Overall, Humanoid Occupancy delivers effective environmental perception for humanoid robots and establishes a technical foundation for standardizing universal visual modules, paving the way for the widespread deployment of humanoid robots in complex real-world scenarios.
>
---
#### [new 203] RISEE: A Highly Interactive Naturalistic Driving Trajectories Dataset with Human Subjective Risk Perception and Eye-tracking Information
- **分类: cs.HC; cs.CV**

- **简介: 该论文构建了一个名为RISEE的自动驾驶数据集，包含自然驾驶轨迹、主观风险评估和眼动数据。旨在解决现有数据缺乏人类因素与安全关键场景的问题，支持自动驾驶系统的研发与验证。**

- **链接: [http://arxiv.org/pdf/2507.19490v1](http://arxiv.org/pdf/2507.19490v1)**

> **作者:** Xinzheng Wu; Junyi Chen; Peiyi Wang; Shunxiang Chen; Yong Shen
>
> **备注:** Submitted for ITSC 2025
>
> **摘要:** In the research and development (R&D) and verification and validation (V&V) phases of autonomous driving decision-making and planning systems, it is necessary to integrate human factors to achieve decision-making and evaluation that align with human cognition. However, most existing datasets primarily focus on vehicle motion states and trajectories, neglecting human-related information. In addition, current naturalistic driving datasets lack sufficient safety-critical scenarios while simulated datasets suffer from low authenticity. To address these issues, this paper constructs the Risk-Informed Subjective Evaluation and Eye-tracking (RISEE) dataset which specifically contains human subjective evaluations and eye-tracking data apart from regular naturalistic driving trajectories. By leveraging the complementary advantages of drone-based (high realism and extensive scenario coverage) and simulation-based (high safety and reproducibility) data collection methods, we first conduct drone-based traffic video recording at a highway ramp merging area. After that, the manually selected highly interactive scenarios are reconstructed in simulation software, and drivers' first-person view (FPV) videos are generated, which are then viewed and evaluated by recruited participants. During the video viewing process, participants' eye-tracking data is collected. After data processing and filtering, 3567 valid subjective risk ratings from 101 participants across 179 scenarios are retained, along with 2045 qualified eye-tracking data segments. The collected data and examples of the generated FPV videos are available in our website.
>
---
#### [new 204] Hybrid Deep Learning and Handcrafted Feature Fusion for Mammographic Breast Cancer Classification
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决乳腺癌良恶性自动分类难题。作者提出一种混合框架，融合深度学习特征（ResNet-50、DINOv2）与手工特征，提升了分类性能，验证了手工特征在增强模型效果上的作用。**

- **链接: [http://arxiv.org/pdf/2507.19843v1](http://arxiv.org/pdf/2507.19843v1)**

> **作者:** Maximilian Tschuchnig; Michael Gadermayr; Khalifa Djemal
>
> **备注:** Accepted at IPTA2025
>
> **摘要:** Automated breast cancer classification from mammography remains a significant challenge due to subtle distinctions between benign and malignant tissue. In this work, we present a hybrid framework combining deep convolutional features from a ResNet-50 backbone with handcrafted descriptors and transformer-based embeddings. Using the CBIS-DDSM dataset, we benchmark our ResNet-50 baseline (AUC: 78.1%) and demonstrate that fusing handcrafted features with deep ResNet-50 and DINOv2 features improves AUC to 79.6% (setup d1), with a peak recall of 80.5% (setup d1) and highest F1 score of 67.4% (setup d1). Our experiments show that handcrafted features not only complement deep representations but also enhance performance beyond transformer-based embeddings. This hybrid fusion approach achieves results comparable to state-of-the-art methods while maintaining architectural simplicity and computational efficiency, making it a practical and effective solution for clinical decision support.
>
---
#### [new 205] LanternNet: A Novel Hub-and-Spoke System to Seek and Suppress Spotted Lanternfly Populations
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出LanternNet，一种基于机器人与AI的中心辐射式系统，用于检测与控制入侵物种斑翅 lanternfly（SLF）种群。该系统通过计算机视觉识别SLF，并利用多个机器人执行灭虫、环境监测等任务，显著减少SLF数量并改善树木健康，属于生态治理与自动化控制任务。**

- **链接: [http://arxiv.org/pdf/2507.20800v1](http://arxiv.org/pdf/2507.20800v1)**

> **作者:** Vinil Polepalli
>
> **摘要:** The invasive spotted lanternfly (SLF) poses a significant threat to agriculture and ecosystems, causing widespread damage. Current control methods, such as egg scraping, pesticides, and quarantines, prove labor-intensive, environmentally hazardous, and inadequate for long-term SLF suppression. This research introduces LanternNet, a novel autonomous robotic Hub-and-Spoke system designed for scalable detection and suppression of SLF populations. A central, tree-mimicking hub utilizes a YOLOv8 computer vision model for precise SLF identification. Three specialized robotic spokes perform targeted tasks: pest neutralization, environmental monitoring, and navigation/mapping. Field deployment across multiple infested sites over 5 weeks demonstrated LanternNet's efficacy. Quantitative analysis revealed significant reductions (p < 0.01, paired t-tests) in SLF populations and corresponding improvements in tree health indicators across the majority of test sites. Compared to conventional methods, LanternNet offers substantial cost advantages and improved scalability. Furthermore, the system's adaptability for enhanced autonomy and targeting of other invasive species presents significant potential for broader ecological impact. LanternNet demonstrates the transformative potential of integrating robotics and AI for advanced invasive species management and improved environmental outcomes.
>
---
## 更新

#### [replaced 001] Point Cloud Self-supervised Learning via 3D to Multi-view Masked Learner
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2311.10887v2](http://arxiv.org/pdf/2311.10887v2)**

> **作者:** Zhimin Chen; Xuewei Chen; Xiao Guo; Yingwei Li; Longlong Jing; Liang Yang; Bing Li
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recently, multi-modal masked autoencoders (MAE) has been introduced in 3D self-supervised learning, offering enhanced feature learning by leveraging both 2D and 3D data to capture richer cross-modal representations. However, these approaches have two limitations: (1) they inefficiently require both 2D and 3D modalities as inputs, even though the inherent multi-view properties of 3D point clouds already contain 2D modality. (2) input 2D modality causes the reconstruction learning to unnecessarily rely on visible 2D information, hindering 3D geometric representation learning. To address these challenges, we propose a 3D to Multi-View Learner (Multi-View ML) that only utilizes 3D modalities as inputs and effectively capture rich spatial information in 3D point clouds. Specifically, we first project 3D point clouds to multi-view 2D images at the feature level based on 3D-based pose. Then, we introduce two components: (1) a 3D to multi-view autoencoder that reconstructs point clouds and multi-view images from 3D and projected 2D features; (2) a multi-scale multi-head (MSMH) attention mechanism that facilitates local-global information interactions in each decoder transformer block through attention heads at various scales. Additionally, a novel two-stage self-training strategy is proposed to align 2D and 3D representations. Our method outperforms state-of-the-art counterparts across various downstream tasks, including 3D classification, part segmentation, and object detection.
>
---
#### [replaced 002] REGRACE: A Robust and Efficient Graph-based Re-localization Algorithm using Consistency Evaluation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.03599v2](http://arxiv.org/pdf/2503.03599v2)**

> **作者:** Débora N. P. Oliveira; Joshua Knights; Sebastián Barbas Laina; Simon Boche; Wolfram Burgard; Stefan Leutenegger
>
> **备注:** Accepted to IROS2025
>
> **摘要:** Loop closures are essential for correcting odometry drift and creating consistent maps, especially in the context of large-scale navigation. Current methods using dense point clouds for accurate place recognition do not scale well due to computationally expensive scan-to-scan comparisons. Alternative object-centric approaches are more efficient but often struggle with sensitivity to viewpoint variation. In this work, we introduce REGRACE, a novel approach that addresses these challenges of scalability and perspective difference in re-localization by using LiDAR-based submaps. We introduce rotation-invariant features for each labeled object and enhance them with neighborhood context through a graph neural network. To identify potential revisits, we employ a scalable bag-of-words approach, pooling one learned global feature per submap. Additionally, we define a revisit with geometrical consistency cues rather than embedding distance, allowing us to recognize far-away loop closures. Our evaluations demonstrate that REGRACE achieves similar results compared to state-of-the-art place recognition and registration baselines while being twice as fast. Code and models are publicly available.
>
---
#### [replaced 003] Decentralized LoRA Augmented Transformer with Context-aware Multi-scale Feature Learning for Secured Eye Diagnosis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06982v2](http://arxiv.org/pdf/2505.06982v2)**

> **作者:** Md. Naimur Asif Borno; Md Sakib Hossain Shovon; MD Hanif Sikder; Iffat Firozy Rimi; Tahani Jaser Alahmadi; Mohammad Ali Moni
>
> **备注:** Under Review at Knowledge-Based Systems
>
> **摘要:** Accurate and privacy-preserving diagnosis of ophthalmic diseases remains a critical challenge in medical imaging, particularly given the limitations of existing deep learning models in handling data imbalance, data privacy concerns, spatial feature diversity, and clinical interpretability. This paper proposes a novel Data efficient Image Transformer (DeiT) based framework that integrates context aware multiscale patch embedding, Low-Rank Adaptation (LoRA), knowledge distillation, and federated learning to address these challenges in a unified manner. The proposed model effectively captures both local and global retinal features by leveraging multi scale patch representations with local and global attention mechanisms. LoRA integration enhances computational efficiency by reducing the number of trainable parameters, while federated learning ensures secure, decentralized training without compromising data privacy. A knowledge distillation strategy further improves generalization in data scarce settings. Comprehensive evaluations on two benchmark datasets OCTDL and the Eye Disease Image Dataset demonstrate that the proposed framework consistently outperforms both traditional CNNs and state of the art transformer architectures across key metrics including AUC, F1 score, and precision. Furthermore, Grad-CAM++ visualizations provide interpretable insights into model predictions, supporting clinical trust. This work establishes a strong foundation for scalable, secure, and explainable AI applications in ophthalmic diagnostics.
>
---
#### [replaced 004] GenM$^3$: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14919v2](http://arxiv.org/pdf/2503.14919v2)**

> **作者:** Junyu Shi; Lijiang Liu; Yong Sun; Zhiyuan Zhang; Jinni Zhou; Qiang Nie
>
> **摘要:** Scaling up motion datasets is crucial to enhance motion generation capabilities. However, training on large-scale multi-source datasets introduces data heterogeneity challenges due to variations in motion content. To address this, we propose Generative Pretrained Multi-path Motion Model (GenM\(^3\)), a comprehensive framework designed to learn unified motion representations. GenM\(^3\) comprises two components: 1) a Multi-Expert VQ-VAE (MEVQ-VAE) that adapts to different dataset distributions to learn a unified discrete motion representation, and 2) a Multi-path Motion Transformer (MMT) that improves intra-modal representations by using separate modality-specific pathways, each with densely activated experts to accommodate variations within that modality, and improves inter-modal alignment by the text-motion shared pathway. To enable large-scale training, we integrate and unify 11 high-quality motion datasets (approximately 220 hours of motion data) and augment it with textual annotations (nearly 10,000 motion sequences labeled by a large language model and 300+ by human experts). After training on our integrated dataset, GenM\(^3\) achieves a state-of-the-art FID of 0.035 on the HumanML3D benchmark, surpassing state-of-the-art methods by a large margin. It also demonstrates strong zero-shot generalization on IDEA400 dataset, highlighting its effectiveness and adaptability across diverse motion scenarios.
>
---
#### [replaced 005] Robotic Visual Instruction
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.00693v3](http://arxiv.org/pdf/2505.00693v3)**

> **作者:** Yanbang Li; Ziyang Gong; Haoyang Li; Xiaoqi Huang; Haolan Kang; Guangping Bai; Xianzheng Ma
>
> **备注:** Project website: https://robotic-visual-instruction.github.io/
>
> **摘要:** Recently, natural language has been the primary medium for human-robot interaction. However, its inherent lack of spatial precision introduces challenges for robotic task definition such as ambiguity and verbosity. Moreover, in some public settings where quiet is required, such as libraries or hospitals, verbal communication with robots is inappropriate. To address these limitations, we introduce the Robotic Visual Instruction (RoVI), a novel paradigm to guide robotic tasks through an object-centric, hand-drawn symbolic representation. RoVI effectively encodes spatial-temporal information into human-interpretable visual instructions through 2D sketches, utilizing arrows, circles, colors, and numbers to direct 3D robotic manipulation. To enable robots to understand RoVI better and generate precise actions based on RoVI, we present Visual Instruction Embodied Workflow (VIEW), a pipeline formulated for RoVI-conditioned policies. This approach leverages Vision-Language Models (VLMs) to interpret RoVI inputs, decode spatial and temporal constraints from 2D pixel space via keypoint extraction, and then transform them into executable 3D action sequences. We additionally curate a specialized dataset of 15K instances to fine-tune small VLMs for edge deployment,enabling them to effectively learn RoVI capabilities. Our approach is rigorously validated across 11 novel tasks in both real and simulated environments, demonstrating significant generalization capability. Notably, VIEW achieves an 87.5% success rate in real-world scenarios involving unseen tasks that feature multi-step actions, with disturbances, and trajectory-following requirements. Project website: https://robotic-visual-instruction.github.io/
>
---
#### [replaced 006] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17762v4](http://arxiv.org/pdf/2411.17762v4)**

> **作者:** Rongchang Xie; Chen Du; Ping Song; Chang Liu
>
> **备注:** ICCV 2025
>
> **摘要:** We introduce MUSE-VL, a Unified Vision-Language Model through Semantic discrete Encoding for multimodal understanding and generation. Recently, the research community has begun exploring unified models for visual generation and understanding. However, existing vision tokenizers (e.g., VQGAN) only consider low-level information, which makes it difficult to align with language tokens. This results in high training complexity and necessitates a large amount of training data to achieve optimal performance. Additionally, their performance is still far from dedicated understanding models. This paper proposes Semantic Discrete Encoding (SDE), which effectively aligns the information of visual tokens and language tokens by adding semantic constraints to the visual tokenizer. This greatly reduces the amount of training data and improves the performance of the unified model. With the same LLM size, our method improved the understanding performance by 4.8% compared to the previous SOTA Emu3 and surpassed the dedicated understanding model LLaVA-NeXT 34B by 3.7%. Our model also surpasses the existing unified models on visual generation benchmarks.
>
---
#### [replaced 007] MTCAE-DFER: Multi-Task Cascaded Autoencoder for Dynamic Facial Expression Recognition
- **分类: cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2412.18988v2](http://arxiv.org/pdf/2412.18988v2)**

> **作者:** Peihao Xiang; Kaida Wu; Ou Bai
>
> **备注:** Camera-ready Version, Accepted by IJCB 2025
>
> **摘要:** This paper expands the cascaded network branch of the autoencoder-based multi-task learning (MTL) framework for dynamic facial expression recognition, namely Multi-Task Cascaded Autoencoder for Dynamic Facial Expression Recognition (MTCAE-DFER). MTCAE-DFER builds a plug-and-play cascaded decoder module, which is based on the Vision Transformer (ViT) architecture and employs the decoder concept of Transformer to reconstruct the multi-head attention module. The decoder output from the previous task serves as the query (Q), representing local dynamic features, while the Video Masked Autoencoder (VideoMAE) shared encoder output acts as both the key (K) and value (V), representing global dynamic features. This setup facilitates interaction between global and local dynamic features across related tasks. Additionally, this proposal aims to alleviate overfitting of complex large model. We utilize autoencoder-based multi-task cascaded learning approach to explore the impact of dynamic face detection and dynamic face landmark on dynamic facial expression recognition, which enhances the model's generalization ability. After we conduct extensive ablation experiments and comparison with state-of-the-art (SOTA) methods on various public datasets for dynamic facial expression recognition, the robustness of the MTCAE-DFER model and the effectiveness of global-local dynamic feature interaction among related tasks have been proven.
>
---
#### [replaced 008] Synomaly Noise and Multi-Stage Diffusion: A Novel Approach for Unsupervised Anomaly Detection in Medical Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.04004v2](http://arxiv.org/pdf/2411.04004v2)**

> **作者:** Yuan Bi; Lucie Huang; Ricarda Clarenbach; Reza Ghotbi; Angelos Karlas; Nassir Navab; Zhongliang Jiang
>
> **摘要:** Anomaly detection in medical imaging plays a crucial role in identifying pathological regions across various imaging modalities, such as brain MRI, liver CT, and carotid ultrasound (US). However, training fully supervised segmentation models is often hindered by the scarcity of expert annotations and the complexity of diverse anatomical structures. To address these issues, we propose a novel unsupervised anomaly detection framework based on a diffusion model that incorporates a synthetic anomaly (Synomaly) noise function and a multi-stage diffusion process. Synomaly noise introduces synthetic anomalies into healthy images during training, allowing the model to effectively learn anomaly removal. The multi-stage diffusion process is introduced to progressively denoise images, preserving fine details while improving the quality of anomaly-free reconstructions. The generated high-fidelity counterfactual healthy images can further enhance the interpretability of the segmentation models, as well as provide a reliable baseline for evaluating the extent of anomalies and supporting clinical decision-making. Notably, the unsupervised anomaly detection model is trained purely on healthy images, eliminating the need for anomalous training samples and pixel-level annotations. We validate the proposed approach on brain MRI, liver CT datasets, and carotid US. The experimental results demonstrate that the proposed framework outperforms existing state-of-the-art unsupervised anomaly detection methods, achieving performance comparable to fully supervised segmentation models in the US dataset. Ablation studies further highlight the contributions of Synomaly noise and the multi-stage diffusion process in improving anomaly segmentation. These findings underscore the potential of our approach as a robust and annotation-efficient alternative for medical anomaly detection.
>
---
#### [replaced 009] Towards Scalable IoT Deployment for Visual Anomaly Detection via Efficient Compression
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07119v3](http://arxiv.org/pdf/2505.07119v3)**

> **作者:** Arianna Stropeni; Francesco Borsatti; Manuel Barusco; Davide Dalle Pezze; Marco Fabris; Gian Antonio Susto
>
> **摘要:** Visual Anomaly Detection (VAD) is a key task in industrial settings, where minimizing operational costs is essential. Deploying deep learning models within Internet of Things (IoT) environments introduces specific challenges due to limited computational power and bandwidth of edge devices. This study investigates how to perform VAD effectively under such constraints by leveraging compact, efficient processing strategies. We evaluate several data compression techniques, examining the tradeoff between system latency and detection accuracy. Experiments on the MVTec AD benchmark demonstrate that significant compression can be achieved with minimal loss in anomaly detection performance compared to uncompressed data. Current results show up to 80% reduction in end-to-end inference time, including edge processing, transmission, and server computation.
>
---
#### [replaced 010] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.00623v3](http://arxiv.org/pdf/2412.00623v3)**

> **作者:** Chensheng Peng; Ido Sobol; Masayoshi Tomizuka; Kurt Keutzer; Chenfeng Xu; Or Litany
>
> **备注:** ICCV 2025
>
> **摘要:** We present a novel framework for training 3D image-conditioned diffusion models using only 2D supervision. Recovering 3D structure from 2D images is inherently ill-posed due to the ambiguity of possible reconstructions, making generative models a natural choice. However, most existing 3D generative models rely on full 3D supervision, which is impractical due to the scarcity of large-scale 3D datasets. To address this, we propose leveraging sparse-view supervision as a scalable alternative. While recent reconstruction models use sparse-view supervision with differentiable rendering to lift 2D images to 3D, they are predominantly deterministic, failing to capture the diverse set of plausible solutions and producing blurry predictions in uncertain regions. A key challenge in training 3D diffusion models with 2D supervision is that the standard training paradigm requires both the denoising process and supervision to be in the same modality. We address this by decoupling the noisy samples being denoised from the supervision signal, allowing the former to remain in 3D while the latter is provided in 2D. Our approach leverages suboptimal predictions from a deterministic image-to-3D model-acting as a "teacher"-to generate noisy 3D inputs, enabling effective 3D diffusion training without requiring full 3D ground truth. We validate our framework on both object-level and scene-level datasets, using two different 3D Gaussian Splat (3DGS) teachers. Our results show that our approach consistently improves upon these deterministic teachers, demonstrating its effectiveness in scalable and high-fidelity 3D generative modeling. See our project page at https://lesson-in-splats.github.io/
>
---
#### [replaced 011] Predicting Neoadjuvant Chemotherapy Response in Triple-Negative Breast Cancer Using Pre-Treatment Histopathologic Images
- **分类: q-bio.QM; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.14730v2](http://arxiv.org/pdf/2505.14730v2)**

> **作者:** Hikmat Khan; Ziyu Su; Huina Zhang; Yihong Wang; Bohan Ning; Shi Wei; Hua Guo; Zaibo Li; Muhammad Khalid Khan Niazi
>
> **摘要:** Triple-negative breast cancer (TNBC) remains a major clinical challenge due to its aggressive behavior and lack of targeted therapies. Accurate early prediction of response to neoadjuvant chemotherapy (NACT) is essential for guiding personalized treatment strategies and improving patient outcomes. In this study, we present an attention-based multiple instance learning (MIL) framework designed to predict pathologic complete response (pCR) directly from pre-treatment hematoxylin and eosin (H&E)-stained biopsy slides. The model was trained on a retrospective in-house cohort of 174 TNBC patients and externally validated on an independent cohort (n = 30). It achieved a mean area under the curve (AUC) of 0.85 during five-fold cross-validation and 0.78 on external testing, demonstrating robust predictive performance and generalizability. To enhance model interpretability, attention maps were spatially co-registered with multiplex immuno-histochemistry (mIHC) data stained for PD-L1, CD8+ T cells, and CD163+ macrophages. The attention regions exhibited moderate spatial overlap with immune-enriched areas, with mean Intersection over Union (IoU) scores of 0.47 for PD-L1, 0.45 for CD8+ T cells, and 0.46 for CD163+ macrophages. The presence of these biomarkers in high-attention regions supports their biological relevance to NACT response in TNBC. This not only improves model interpretability but may also inform future efforts to identify clinically actionable histological biomarkers directly from H&E-stained biopsy slides, further supporting the utility of this approach for accurate NACT response prediction and advancing precision oncology in TNBC.
>
---
#### [replaced 012] Dynamic Try-On: Taming Video Virtual Try-on with Dynamic Attention Mechanism
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09822v2](http://arxiv.org/pdf/2412.09822v2)**

> **作者:** Jun Zheng; Jing Wang; Fuwei Zhao; Xujie Zhang; Xiaodan Liang
>
> **备注:** Project Page: https://zhengjun-ai.github.io/dynamic-tryon-page/. Accepted by The 36th British Machine Vision Conference
>
> **摘要:** Video try-on stands as a promising area for its tremendous real-world potential. Previous research on video try-on has primarily focused on transferring product clothing images to videos with simple human poses, while performing poorly with complex movements. To better preserve clothing details, those approaches are armed with an additional garment encoder, resulting in higher computational resource consumption. The primary challenges in this domain are twofold: (1) leveraging the garment encoder's capabilities in video try-on while lowering computational requirements; (2) ensuring temporal consistency in the synthesis of human body parts, especially during rapid movements. To tackle these issues, we propose a novel video try-on framework based on Diffusion Transformer(DiT), named Dynamic Try-On. To reduce computational overhead, we adopt a straightforward approach by utilizing the DiT backbone itself as the garment encoder and employing a dynamic feature fusion module to store and integrate garment features. To ensure temporal consistency of human body parts, we introduce a limb-aware dynamic attention module that enforces the DiT backbone to focus on the regions of human limbs during the denoising process. Extensive experiments demonstrate the superiority of Dynamic Try-On in generating stable and smooth try-on results, even for videos featuring complicated human postures.
>
---
#### [replaced 013] Chimera: Improving Generalist Model with Domain-Specific Experts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05983v3](http://arxiv.org/pdf/2412.05983v3)**

> **作者:** Tianshuo Peng; Mingsheng Li; Jiakang Yuan; Hongbin Zhou; Renqiu Xia; Renrui Zhang; Lei Bai; Song Mao; Bin Wang; Aojun Zhou; Botian Shi; Tao Chen; Bo Zhang; Xiangyu Yue
>
> **备注:** Accepted by ICCV-2025, Chimera Homepage: https://alpha-innovator.github.io/chimera_page
>
> **摘要:** Recent advancements in Large Multi-modal Models (LMMs) underscore the importance of scaling by increasing image-text paired data, achieving impressive performance on general tasks. Despite their effectiveness in broad applications, generalist models are primarily trained on web-scale datasets dominated by natural images, resulting in the sacrifice of specialized capabilities for domain-specific tasks that require extensive domain prior knowledge. Moreover, directly integrating expert models tailored for specific domains is challenging due to the representational gap and imbalanced optimization between the generalist model and experts. To address these challenges, we introduce Chimera, a scalable and low-cost multi-modal pipeline designed to boost the ability of existing LMMs with domain-specific experts. Specifically, we design a progressive training strategy to integrate features from expert models into the input of a generalist LMM. To address the imbalanced optimization caused by the well-aligned general visual encoder, we introduce a novel Generalist-Specialist Collaboration Masking (GSCM) mechanism. This results in a versatile model that excels across the chart, table, math, and document domains, achieving state-of-the-art performance on multi-modal reasoning and visual content extraction tasks, both of which are challenging tasks for assessing existing LMMs.
>
---
#### [replaced 014] Unlocking Constraints: Source-Free Occlusion-Aware Seamless Segmentation
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.21198v2](http://arxiv.org/pdf/2506.21198v2)**

> **作者:** Yihong Cao; Jiaming Zhang; Xu Zheng; Hao Shi; Kunyu Peng; Hang Liu; Kailun Yang; Hui Zhang
>
> **备注:** Accepted to ICCV 2025. All data and code will be made publicly available at https://github.com/yihong-97/UNLOCK
>
> **摘要:** Panoramic image processing is essential for omni-context perception, yet faces constraints like distortions, perspective occlusions, and limited annotations. Previous unsupervised domain adaptation methods transfer knowledge from labeled pinhole data to unlabeled panoramic images, but they require access to source pinhole data. To address these, we introduce a more practical task, i.e., Source-Free Occlusion-Aware Seamless Segmentation (SFOASS), and propose its first solution, called UNconstrained Learning Omni-Context Knowledge (UNLOCK). Specifically, UNLOCK includes two key modules: Omni Pseudo-Labeling Learning and Amodal-Driven Context Learning. While adapting without relying on source data or target labels, this framework enhances models to achieve segmentation with 360{\deg} viewpoint coverage and occlusion-aware reasoning. Furthermore, we benchmark the proposed SFOASS task through both real-to-real and synthetic-to-real adaptation settings. Experimental results show that our source-free method achieves performance comparable to source-dependent methods, yielding state-of-the-art scores of 10.9 in mAAP and 11.6 in mAP, along with an absolute improvement of +4.3 in mAPQ over the source-only method. All data and code will be made publicly available at https://github.com/yihong-97/UNLOCK.
>
---
#### [replaced 015] Manipulating Multimodal Agents via Cross-Modal Prompt Injection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14348v4](http://arxiv.org/pdf/2504.14348v4)**

> **作者:** Le Wang; Zonghao Ying; Tianyuan Zhang; Siyuan Liang; Shengshan Hu; Mingchuan Zhang; Aishan Liu; Xianglong Liu
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this paper, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach incorporates two key coordinated components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to construct the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms state-of-the-art attacks, achieving at least a +30.1% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.
>
---
#### [replaced 016] EventVAD: Training-Free Event-Aware Video Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13092v3](http://arxiv.org/pdf/2504.13092v3)**

> **作者:** Yihua Shao; Haojin He; Sijie Li; Siyu Chen; Xinwei Long; Fanhu Zeng; Yuxuan Fan; Muyang Zhang; Ziyang Yan; Ao Ma; Xiaochen Wang; Hao Tang; Yan Wang; Shuyan Li
>
> **备注:** Paper was accepted by ACM MM 2025; Code: https://github.com/YihuaJerry/EventVAD
>
> **摘要:** Video Anomaly Detection~(VAD) focuses on identifying anomalies within videos. Supervised methods require an amount of in-domain training data and often struggle to generalize to unseen anomalies. In contrast, training-free methods leverage the intrinsic world knowledge of large language models (LLMs) to detect anomalies but face challenges in localizing fine-grained visual transitions and diverse events. Therefore, we propose EventVAD, an event-aware video anomaly detection framework that combines tailored dynamic graph architectures and multimodal LLMs through temporal-event reasoning. Specifically, EventVAD first employs dynamic spatiotemporal graph modeling with time-decay constraints to capture event-aware video features. Then, it performs adaptive noise filtering and uses signal ratio thresholding to detect event boundaries via unsupervised statistical features. The statistical boundary detection module reduces the complexity of processing long videos for MLLMs and improves their temporal reasoning through event consistency. Finally, it utilizes a hierarchical prompting strategy to guide MLLMs in performing reasoning before determining final decisions. We conducted extensive experiments on the UCF-Crime and XD-Violence datasets. The results demonstrate that EventVAD with a 7B MLLM achieves state-of-the-art (SOTA) in training-free settings, outperforming strong baselines that use 7B or larger MLLMs.
>
---
#### [replaced 017] OpenHuman4D: Open-Vocabulary 4D Human Parsing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09880v2](http://arxiv.org/pdf/2507.09880v2)**

> **作者:** Keito Suzuki; Bang Du; Runfa Blark Li; Kunyao Chen; Lei Wang; Peng Liu; Ning Bi; Truong Nguyen
>
> **备注:** BMVC 2025
>
> **摘要:** Understanding dynamic 3D human representation has become increasingly critical in virtual and extended reality applications. However, existing human part segmentation methods are constrained by reliance on closed-set datasets and prolonged inference times, which significantly restrict their applicability. In this paper, we introduce the first 4D human parsing framework that simultaneously addresses these challenges by reducing the inference time and introducing open-vocabulary capabilities. Building upon state-of-the-art open-vocabulary 3D human parsing techniques, our approach extends the support to 4D human-centric video with three key innovations: 1) We adopt mask-based video object tracking to efficiently establish spatial and temporal correspondences, avoiding the necessity of segmenting all frames. 2) A novel Mask Validation module is designed to manage new target identification and mitigate tracking failures. 3) We propose a 4D Mask Fusion module, integrating memory-conditioned attention and logits equalization for robust embedding fusion. Extensive experiments demonstrate the effectiveness and flexibility of the proposed method on 4D human-centric parsing tasks, achieving up to 93.3% acceleration compared to the previous state-of-the-art method, which was limited to parsing fixed classes.
>
---
#### [replaced 018] RemixFusion: Residual-based Mixed Representation for Large-scale Online RGB-D Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17594v2](http://arxiv.org/pdf/2507.17594v2)**

> **作者:** Yuqing Lan; Chenyang Zhu; Shuaifeng Zhi; Jiazhao Zhang; Zhoufeng Wang; Renjiao Yi; Yijie Wang; Kai Xu
>
> **摘要:** The introduction of the neural implicit representation has notably propelled the advancement of online dense reconstruction techniques. Compared to traditional explicit representations, such as TSDF, it improves the mapping completeness and memory efficiency. However, the lack of reconstruction details and the time-consuming learning of neural representations hinder the widespread application of neural-based methods to large-scale online reconstruction. We introduce RemixFusion, a novel residual-based mixed representation for scene reconstruction and camera pose estimation dedicated to high-quality and large-scale online RGB-D reconstruction. In particular, we propose a residual-based map representation comprised of an explicit coarse TSDF grid and an implicit neural module that produces residuals representing fine-grained details to be added to the coarse grid. Such mixed representation allows for detail-rich reconstruction with bounded time and memory budget, contrasting with the overly-smoothed results by the purely implicit representations, thus paving the way for high-quality camera tracking. Furthermore, we extend the residual-based representation to handle multi-frame joint pose optimization via bundle adjustment (BA). In contrast to the existing methods, which optimize poses directly, we opt to optimize pose changes. Combined with a novel technique for adaptive gradient amplification, our method attains better optimization convergence and global optimality. Furthermore, we adopt a local moving volume to factorize the mixed scene representation with a divide-and-conquer design to facilitate efficient online learning in our residual-based framework. Extensive experiments demonstrate that our method surpasses all state-of-the-art ones, including those based either on explicit or implicit representations, in terms of the accuracy of both mapping and tracking on large-scale scenes.
>
---
#### [replaced 019] Representing 3D Shapes With 64 Latent Vectors for 3D Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.08737v2](http://arxiv.org/pdf/2503.08737v2)**

> **作者:** In Cho; Youngbeom Yoo; Subin Jeon; Seon Joo Kim
>
> **摘要:** Constructing a compressed latent space through a variational autoencoder (VAE) is the key for efficient 3D diffusion models. This paper introduces COD-VAE that encodes 3D shapes into a COmpact set of 1D latent vectors without sacrificing quality. COD-VAE introduces a two-stage autoencoder scheme to improve compression and decoding efficiency. First, our encoder block progressively compresses point clouds into compact latent vectors via intermediate point patches. Second, our triplane-based decoder reconstructs dense triplanes from latent vectors instead of directly decoding neural fields, significantly reducing computational overhead of neural fields decoding. Finally, we propose uncertainty-guided token pruning, which allocates resources adaptively by skipping computations in simpler regions and improves the decoder efficiency. Experimental results demonstrate that COD-VAE achieves 16x compression compared to the baseline while maintaining quality. This enables 20.8x speedup in generation, highlighting that a large number of latent vectors is not a prerequisite for high-quality reconstruction and generation. The code is available at https://github.com/join16/COD-VAE.
>
---
#### [replaced 020] iSEARLE: Improving Textual Inversion for Zero-Shot Composed Image Retrieval
- **分类: cs.CV; cs.IR**

- **链接: [http://arxiv.org/pdf/2405.02951v2](http://arxiv.org/pdf/2405.02951v2)**

> **作者:** Lorenzo Agnolucci; Alberto Baldrati; Alberto Del Bimbo; Marco Bertini
>
> **备注:** Accepted at TPAMI, extended version of the ICCV2023 paper arXiv:2303.15247
>
> **摘要:** Given a query consisting of a reference image and a relative caption, Composed Image Retrieval (CIR) aims to retrieve target images visually similar to the reference one while incorporating the changes specified in the relative caption. The reliance of supervised methods on labor-intensive manually labeled datasets hinders their broad applicability. In this work, we introduce a new task, Zero-Shot CIR (ZS-CIR), that addresses CIR without the need for a labeled training dataset. We propose an approach named iSEARLE (improved zero-Shot composEd imAge Retrieval with textuaL invErsion) that involves mapping the visual information of the reference image into a pseudo-word token in CLIP token embedding space and combining it with the relative caption. To foster research on ZS-CIR, we present an open-domain benchmarking dataset named CIRCO (Composed Image Retrieval on Common Objects in context), the first CIR dataset where each query is labeled with multiple ground truths and a semantic categorization. The experimental results illustrate that iSEARLE obtains state-of-the-art performance on three different CIR datasets -- FashionIQ, CIRR, and the proposed CIRCO -- and two additional evaluation settings, namely domain conversion and object composition. The dataset, the code, and the model are publicly available at https://github.com/miccunifi/SEARLE.
>
---
#### [replaced 021] DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.11921v2](http://arxiv.org/pdf/2411.11921v2)**

> **作者:** Chensheng Peng; Chengwei Zhang; Yixiao Wang; Chenfeng Xu; Yichen Xie; Wenzhao Zheng; Kurt Keutzer; Masayoshi Tomizuka; Wei Zhan
>
> **备注:** CVPR 2025
>
> **摘要:** We present DeSiRe-GS, a self-supervised gaussian splatting representation, enabling effective static-dynamic decomposition and high-fidelity surface reconstruction in complex driving scenarios. Our approach employs a two-stage optimization pipeline of dynamic street Gaussians. In the first stage, we extract 2D motion masks based on the observation that 3D Gaussian Splatting inherently can reconstruct only the static regions in dynamic environments. These extracted 2D motion priors are then mapped into the Gaussian space in a differentiable manner, leveraging an efficient formulation of dynamic Gaussians in the second stage. Combined with the introduced geometric regularizations, our method are able to address the over-fitting issues caused by data sparsity in autonomous driving, reconstructing physically plausible Gaussians that align with object surfaces rather than floating in air. Furthermore, we introduce temporal cross-view consistency to ensure coherence across time and viewpoints, resulting in high-quality surface reconstruction. Comprehensive experiments demonstrate the efficiency and effectiveness of DeSiRe-GS, surpassing prior self-supervised arts and achieving accuracy comparable to methods relying on external 3D bounding box annotations. Code is available at https://github.com/chengweialan/DeSiRe-GS
>
---
#### [replaced 022] CP-LLM: Context and Pixel Aware Large Language Model for Video Quality Assessment
- **分类: cs.CV; cs.MM; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.16025v2](http://arxiv.org/pdf/2505.16025v2)**

> **作者:** Wen Wen; Yaohong Wu; Yue Sheng; Neil Birkbeck; Balu Adsumilli; Yilin Wang
>
> **备注:** Under review
>
> **摘要:** Video quality assessment (VQA) is a challenging research topic with broad applications. Effective VQA necessitates sensitivity to pixel-level distortions and a comprehensive understanding of video context to accurately determine the perceptual impact of distortions. Traditional hand-crafted and learning-based VQA models mainly focus on pixel-level distortions and lack contextual understanding, while recent LLM-based models struggle with sensitivity to small distortions or handle quality scoring and description as separate tasks. To address these shortcomings, we introduce CP-LLM: a Context and Pixel aware Large Language Model. CP-LLM is a novel multimodal LLM architecture featuring dual vision encoders designed to independently analyze perceptual quality at both high-level (video context) and low-level (pixel distortion) granularity, along with a language decoder subsequently reasons about the interplay between these aspects. This design enables CP-LLM to simultaneously produce robust quality scores and interpretable quality descriptions, with enhanced sensitivity to pixel distortions (e.g. compression artifacts). The model is trained via a multi-task pipeline optimizing for score prediction, description generation, and pairwise comparisons. Experiment results demonstrate that CP-LLM achieves state-of-the-art cross-dataset performance on established VQA benchmarks and superior robustness to pixel distortions, confirming its efficacy for comprehensive and practical video quality assessment in real-world scenarios.
>
---
#### [replaced 023] Vidar: Embodied Video Diffusion Model for Generalist Bimanual Manipulation
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.12898v2](http://arxiv.org/pdf/2507.12898v2)**

> **作者:** Yao Feng; Hengkai Tan; Xinyi Mao; Guodong Liu; Shuhe Huang; Chendong Xiang; Hang Su; Jun Zhu
>
> **摘要:** Bimanual robotic manipulation, which involves the coordinated control of two robotic arms, is foundational for solving challenging tasks. Despite recent progress in general-purpose manipulation, data scarcity and embodiment heterogeneity remain serious obstacles to further scaling up in bimanual settings. In this paper, we introduce Video Diffusion for Action Reasoning (Vidar), a two-stage framework that leverages large-scale, diffusion-based video pre-training and a novel masked inverse dynamics model for action prediction. We pre-train the video diffusion model on 750K multi-view videos from three real-world bimanual robot platforms, utilizing a unified observation space that encodes robot, camera, task, and scene contexts. Our masked inverse dynamics model learns masks to extract action-relevant information from generated trajectories without requiring pixel-level labels, and the masks can effectively generalize to unseen backgrounds. Our experiments demonstrate that with only 20 minutes of human demonstrations on an unseen robot platform (only 1% of typical data requirements), Vidar generalizes to unseen tasks and backgrounds with strong semantic understanding, surpassing state-of-the-art methods. Our findings highlight the potential of video foundation models, coupled with masked action prediction, to enable scalable and generalizable robotic manipulation in diverse real-world settings.
>
---
#### [replaced 024] NSegment : Label-specific Deformations for Remote Sensing Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.19634v5](http://arxiv.org/pdf/2504.19634v5)**

> **作者:** Yechan Kim; DongHo Yoon; SooYeon Kim; Moongu Jeon
>
> **摘要:** Labeling errors in remote sensing (RS) image segmentation datasets often remain implicit and subtle due to ambiguous class boundaries, mixed pixels, shadows, complex terrain features, and subjective annotator bias. Furthermore, the scarcity of annotated RS data due to the high cost of labeling complicates training noise-robust models. While sophisticated mechanisms such as label selection or noise correction might address the issue mentioned above, they tend to increase training time and add implementation complexity. In this paper, we propose NSegment-a simple yet effective data augmentation solution to mitigate this issue. Unlike traditional methods, it applies elastic transformations only to segmentation labels, varying deformation intensity per sample in each training epoch to address annotation inconsistencies. Experimental results demonstrate that our approach improves the performance of RS image segmentation over various state-of-the-art models.
>
---
#### [replaced 025] Coherent Online Road Topology Estimation and Reasoning with Standard-Definition Maps
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.01397v2](http://arxiv.org/pdf/2507.01397v2)**

> **作者:** Khanh Son Pham; Christian Witte; Jens Behley; Johannes Betz; Cyrill Stachniss
>
> **备注:** Accepted at IROS 2025
>
> **摘要:** Most autonomous cars rely on the availability of high-definition (HD) maps. Current research aims to address this constraint by directly predicting HD map elements from onboard sensors and reasoning about the relationships between the predicted map and traffic elements. Despite recent advancements, the coherent online construction of HD maps remains a challenging endeavor, as it necessitates modeling the high complexity of road topologies in a unified and consistent manner. To address this challenge, we propose a coherent approach to predict lane segments and their corresponding topology, as well as road boundaries, all by leveraging prior map information represented by commonly available standard-definition (SD) maps. We propose a network architecture, which leverages hybrid lane segment encodings comprising prior information and denoising techniques to enhance training stability and performance. Furthermore, we facilitate past frames for temporal consistency. Our experimental evaluation demonstrates that our approach outperforms previous methods by a large margin, highlighting the benefits of our modeling scheme.
>
---
#### [replaced 026] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16403v2](http://arxiv.org/pdf/2507.16403v2)**

> **作者:** Duong T. Tran; Trung-Kien Tran; Manfred Hauswirth; Danh Le Phuoc
>
> **备注:** Accepted at the IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** In this paper, we propose a new dataset, ReasonVQA, for the Visual Question Answering (VQA) task. Our dataset is automatically integrated with structured encyclopedic knowledge and constructed using a low-cost framework, which is capable of generating complex, multi-hop questions. We evaluated state-of-the-art VQA models on ReasonVQA, and the empirical results demonstrate that ReasonVQA poses significant challenges to these models, highlighting its potential for benchmarking and advancing the field of VQA. Additionally, our dataset can be easily scaled with respect to input images; the current version surpasses the largest existing datasets requiring external knowledge by more than an order of magnitude.
>
---
#### [replaced 027] Multi-Person Interaction Generation from Two-Person Motion Priors
- **分类: cs.GR; cs.CV; cs.LG; I.3.7**

- **链接: [http://arxiv.org/pdf/2505.17860v2](http://arxiv.org/pdf/2505.17860v2)**

> **作者:** Wenning Xu; Shiyu Fan; Paul Henderson; Edmond S. L. Ho
>
> **备注:** SIGGRAPH 2025 Conference Papers, project page at http://wenningxu.github.io/multicharacter/
>
> **摘要:** Generating realistic human motion with high-level controls is a crucial task for social understanding, robotics, and animation. With high-quality MOCAP data becoming more available recently, a wide range of data-driven approaches have been presented. However, modelling multi-person interactions still remains a less explored area. In this paper, we present Graph-driven Interaction Sampling, a method that can generate realistic and diverse multi-person interactions by leveraging existing two-person motion diffusion models as motion priors. Instead of training a new model specific to multi-person interaction synthesis, our key insight is to spatially and temporally separate complex multi-person interactions into a graph structure of two-person interactions, which we name the Pairwise Interaction Graph. We thus decompose the generation task into simultaneous single-person motion generation conditioned on one other's motion. In addition, to reduce artifacts such as interpenetrations of body parts in generated multi-person interactions, we introduce two graph-dependent guidance terms into the diffusion sampling scheme. Unlike previous work, our method can produce various high-quality multi-person interactions without having repetitive individual motions. Extensive experiments demonstrate that our approach consistently outperforms existing methods in reducing artifacts when generating a wide range of two-person and multi-person interactions.
>
---
#### [replaced 028] Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.12789v3](http://arxiv.org/pdf/2411.12789v3)**

> **作者:** Haoyu Zhao; Hao Wang; Xingyue Zhao; Hao Fei; Hongqiu Wang; Chengjiang Long; Hua Zou
>
> **备注:** ICCV 2025
>
> **摘要:** Recent advancements in 3D generation models have opened new possibilities for simulating dynamic 3D object movements and customizing behaviors, yet creating this content remains challenging. Current methods often require manual assignment of precise physical properties for simulations or rely on video generation models to predict them, which is computationally intensive. In this paper, we rethink the usage of multi-modal large language model (MLLM) in physics-based simulation, and present Sim Anything, a physics-based approach that endows static 3D objects with interactive dynamics. We begin with detailed scene reconstruction and object-level 3D open-vocabulary segmentation, progressing to multi-view image in-painting. Inspired by human visual reasoning, we propose MLLM-based Physical Property Perception (MLLM-P3) to predict mean physical properties of objects in a zero-shot manner. Based on the mean values and the object's geometry, the Material Property Distribution Prediction model (MPDP) model then estimates the full distribution, reformulating the problem as probability distribution estimation to reduce computational costs. Finally, we simulate objects in an open-world scene with particles sampled via the Physical-Geometric Adaptive Sampling (PGAS) strategy, efficiently capturing complex deformations and significantly reducing computational costs. Extensive experiments and user studies demonstrate our Sim Anything achieves more realistic motion than state-of-the-art methods within 2 minutes on a single GPU.
>
---
#### [replaced 029] Video Forgery Detection for Surveillance Cameras: A Review
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.03832v2](http://arxiv.org/pdf/2505.03832v2)**

> **作者:** Noor B. Tayfor; Tarik A. Rashid; Shko M. Qader; Bryar A. Hassan; Mohammed H. Abdalla; Jafar Majidpour; Aram M. Ahmed; Hussein M. Ali; Aso M. Aladdin; Abdulhady A. Abdullah; Ahmed S. Shamsaldin; Haval M. Sidqi; Abdulrahman Salih; Zaher M. Yaseen; Azad A. Ameen; Janmenjoy Nayak; Mahmood Yashar Hamza
>
> **摘要:** The widespread availability of video recording through smartphones and digital devices has made video-based evidence more accessible than ever. Surveillance footage plays a crucial role in security, law enforcement, and judicial processes. However, with the rise of advanced video editing tools, tampering with digital recordings has become increasingly easy, raising concerns about their authenticity. Ensuring the integrity of surveillance videos is essential, as manipulated footage can lead to misinformation and undermine judicial decisions. This paper provides a comprehensive review of existing forensic techniques used to detect video forgery, focusing on their effectiveness in verifying the authenticity of surveillance recordings. Various methods, including compression-based analysis, frame duplication detection, and machine learning-based approaches, are explored. The findings highlight the growing necessity for more robust forensic techniques to counteract evolving forgery methods. Strengthening video forensic capabilities will ensure that surveillance recordings remain credible and admissible as legal evidence.
>
---
#### [replaced 030] From General to Specialized: The Need for Foundational Models in Agriculture
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.05390v2](http://arxiv.org/pdf/2507.05390v2)**

> **作者:** Vishal Nedungadi; Xingguo Xiong; Aike Potze; Ron Van Bree; Tao Lin; Marc Rußwurm; Ioannis N. Athanasiadis
>
> **备注:** Accepted to the SEA Workshop (Sustainability with Earth Observation & AI) at ICCV 2025
>
> **摘要:** Food security remains a global concern as population grows and climate change intensifies, demanding innovative solutions for sustainable agricultural productivity. Recent advances in foundation models have demonstrated remarkable performance in remote sensing and climate sciences, and therefore offer new opportunities for agricultural monitoring. However, their application in challenges related to agriculture-such as crop type mapping, crop phenology estimation, and crop yield estimation-remains under-explored. In this work, we quantitatively evaluate existing foundational models to assess their effectivity for a representative set of agricultural tasks. From an agricultural domain perspective, we describe a requirements framework for an ideal agricultural foundation model (CropFM). We then survey and compare existing general-purpose foundational models in this framework and empirically evaluate two exemplary of them in three representative agriculture specific tasks. Finally, we highlight the need for a dedicated foundational model tailored specifically to agriculture.
>
---
#### [replaced 031] Facial Demorphing from a Single Morph Using a Latent Conditional GAN
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18566v2](http://arxiv.org/pdf/2507.18566v2)**

> **作者:** Nitish Shukla; Arun Ross
>
> **摘要:** A morph is created by combining two (or more) face images from two (or more) identities to create a composite image that is highly similar to all constituent identities, allowing the forged morph to be biometrically associated with more than one individual. Morph Attack Detection (MAD) can be used to detect a morph, but does not reveal the constituent images. Demorphing - the process of deducing the constituent images - is thus vital to provide additional evidence about a morph. Existing demorphing methods suffer from the morph replication problem, where the outputs tend to look very similar to the morph itself, or assume that train and test morphs are generated using the same morph technique. The proposed method overcomes these issues. The method decomposes a morph in latent space allowing it to demorph images created from unseen morph techniques and face styles. We train our method on morphs created from synthetic faces and test on morphs created from real faces using different morph techniques. Our method outperforms existing methods by a considerable margin and produces high fidelity demorphed face images.
>
---
#### [replaced 032] VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.07960v2](http://arxiv.org/pdf/2504.07960v2)**

> **作者:** Zhong-Yu Li; Ruoyi Du; Juncheng Yan; Le Zhuo; Zhen Li; Peng Gao; Zhanyu Ma; Ming-Ming Cheng
>
> **备注:** Accepted at ICCV 2025. Project page: https://visualcloze.github.io
>
> **摘要:** Recent progress in diffusion models significantly advances various image generation tasks. However, the current mainstream approach remains focused on building task-specific models, which have limited efficiency when supporting a wide range of different needs. While universal models attempt to address this limitation, they face critical challenges, including generalizable task instruction, appropriate task distributions, and unified architectural design. To tackle these challenges, we propose VisualCloze, a universal image generation framework, which supports a wide range of in-domain tasks, generalization to unseen ones, unseen unification of multiple tasks, and reverse generation. Unlike existing methods that rely on language-based task instruction, leading to task ambiguity and weak generalization, we integrate visual in-context learning, allowing models to identify tasks from visual demonstrations. Meanwhile, the inherent sparsity of visual task distributions hampers the learning of transferable knowledge across tasks. To this end, we introduce Graph200K, a graph-structured dataset that establishes various interrelated tasks, enhancing task density and transferable knowledge. Furthermore, we uncover that our unified image generation formulation shared a consistent objective with image infilling, enabling us to leverage the strong generative priors of pre-trained infilling models without modifying the architectures.
>
---
#### [replaced 033] Hoi2Threat: An Interpretable Threat Detection Method for Human Violence Scenarios Guided by Human-Object Interaction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10508v3](http://arxiv.org/pdf/2503.10508v3)**

> **作者:** Yuhan Wang; Cheng Liu; Daou Zhang; Zihan Zhao; Jinyang Chen; Purui Dong; Zuyuan Yu; Ziru Wang; Weichao Wu
>
> **摘要:** In light of the mounting imperative for public security, the necessity for automated threat detection in high-risk scenarios is becoming increasingly pressing. However, existing methods generally suffer from the problems of uninterpretable inference and biased semantic understanding, which severely limits their reliability in practical deployment. In order to address the aforementioned challenges, this article proposes a threat detection method based on human-object interaction pairs (HOI-pairs), Hoi2Threat. This method is based on the fine-grained multimodal TD-Hoi dataset, enhancing the model's semantic modeling ability for key entities and their behavioral interactions by using structured HOI tags to guide language generation. Furthermore, a set of metrics is designed for the evaluation of text response quality, with the objective of systematically measuring the model's representation accuracy and comprehensibility during threat interpretation. The experimental results have demonstrated that Hoi2Threat attains substantial enhancement in several threat detection tasks, particularly in the core metrics of Correctness of Information (CoI), Behavioral Mapping Accuracy (BMA), and Threat Detailed Orientation (TDO), which are 5.08, 5.04, and 4.76, and 7.10%, 6.80%, and 2.63%, respectively, in comparison with the Gemma3 (4B). The aforementioned results provide comprehensive validation of the merits of this approach in the domains of semantic understanding, entity behavior mapping, and interpretability.
>
---
#### [replaced 034] GLC++: Source-Free Universal Domain Adaptation through Global-Local Clustering and Contrastive Affinity Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.14410v2](http://arxiv.org/pdf/2403.14410v2)**

> **作者:** Sanqing Qu; Tianpei Zou; Florian Röhrbein; Cewu Lu; Guang Chen; Dacheng Tao; Changjun Jiang
>
> **备注:** A substantial extension of the CVPR paper "Upcycling Models under Domain and Category Shift", recently accepted by IEEE-TPAMI. arXiv admin note: text overlap with arXiv:2303.07110
>
> **摘要:** Deep neural networks often exhibit sub-optimal performance under covariate and category shifts. Source-Free Domain Adaptation (SFDA) presents a promising solution to this dilemma, yet most SFDA approaches are restricted to closed-set scenarios. In this paper, we explore Source-Free Universal Domain Adaptation (SF-UniDA) aiming to accurately classify "known" data belonging to common categories and segregate them from target-private "unknown" data. We propose a novel Global and Local Clustering (GLC) technique, which comprises an adaptive one-vs-all global clustering algorithm to discern between target classes, complemented by a local k-NN clustering strategy to mitigate negative transfer. Despite the effectiveness, the inherent closed-set source architecture leads to uniform treatment of "unknown" data, impeding the identification of distinct "unknown" categories. To address this, we evolve GLC to GLC++, integrating a contrastive affinity learning strategy. We examine the superiority of GLC and GLC++ across multiple benchmarks and category shift scenarios. Remarkably, in the most challenging open-partial-set scenarios, GLC and GLC++ surpass GATE by 16.8\% and 18.9\% in H-score on VisDA, respectively. GLC++ enhances the novel category clustering accuracy of GLC by 4.1\% in open-set scenarios on Office-Home. Furthermore, the introduced contrastive learning strategy not only enhances GLC but also significantly facilitates existing methodologies. The code is available at https://github.com/ispc-lab/GLC-plus.
>
---
#### [replaced 035] FREE-Merging: Fourier Transform for Efficient Model Merging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16815v3](http://arxiv.org/pdf/2411.16815v3)**

> **作者:** Shenghe Zheng; Hongzhi Wang
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** With the rapid growth of deep learning, there is an increasing availability of open-source models for various tasks. However, single fine-tuned models often fall short of meeting the diverse needs of users. Model merging has thus emerged as an efficient method to integrate the capabilities of existing models into a unified model. Nevertheless, existing model merging methods face challenging trade-offs between performance and deployment costs, primarily due to task interference. For the first time, we reveal that task interference is evident in the frequency domain of model parameters, yet current efforts only focus on spatial domain solutions, which are largely ineffective in addressing frequency domain interference. To mitigate the impact of frequency domain interference, we propose FR-Merging, an innovative method that effectively filters harmful frequency domain interference on the backbone with minimal computational overhead. Since performance loss is inevitable with cost-free methods, we propose a lightweight task-specific expert module that dynamically compensates for information loss during merging. This proposed framework, FREE-Merging (FR-Merging with experts), strikes a balanced trade-off between training cost, inference latency, storage requirements, and performance. We demonstrate the effectiveness of both FR-Merging and FREE-Merging on multiple tasks across CV, NLP, and Multi-Modal domains and show that they can be flexibly adapted to specific needs.
>
---
#### [replaced 036] Generalizable Targeted Data Poisoning against Varying Physical Objects
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.03908v2](http://arxiv.org/pdf/2412.03908v2)**

> **作者:** Zhizhen Chen; Zhengyu Zhao; Subrat Kishore Dutta; Chenhao Lin; Chao Shen; Xiao Zhang
>
> **备注:** 13 pages, 9 figures, 7 tables
>
> **摘要:** Targeted data poisoning (TDP) aims to compromise the model's prediction on a specific (test) target by perturbing a small subset of training data. Existing work on TDP has focused on an overly ideal threat model in which the same image sample of the target is used during both poisoning and inference stages. However, in the real world, a target object often appears in complex variations due to changes of physical settings such as viewpoint, background, and lighting conditions. In this work, we take the first step toward understanding the real-world threats of TDP by studying its generalizability across varying physical conditions. In particular, we observe that solely optimizing gradient directions, as adopted by the best previous TDP method, achieves limited generalization. To address this limitation, we propose optimizing both the gradient direction and magnitude for more generalizable gradient matching, thereby leading to higher poisoning success rates. For instance, our method outperforms the state of the art by 19.49% when poisoning CIFAR-10 images targeting multi-view cars.
>
---
#### [replaced 037] DDB: Diffusion Driven Balancing to Address Spurious Correlations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.17226v2](http://arxiv.org/pdf/2503.17226v2)**

> **作者:** Aryan Yazdan Parast; Basim Azam; Naveed Akhtar
>
> **摘要:** Deep neural networks trained with Empirical Risk Minimization (ERM) perform well when both training and test data come from the same domain, but they often fail to generalize to out-of-distribution samples. In image classification, these models may rely on spurious correlations that often exist between labels and irrelevant features of images, making predictions unreliable when those features do not exist. We propose a Diffusion Driven Balancing (DDB) technique to generate training samples with text-to-image diffusion models for addressing the spurious correlation problem. First, we compute the best describing token for the visual features pertaining to the causal components of samples by a textual inversion mechanism. Then, leveraging a language segmentation method and a diffusion model, we generate new samples by combining the causal component with the elements from other classes. We also meticulously prune the generated samples based on the prediction probabilities and attribution scores of the ERM model to ensure their correct composition for our objective. Finally, we retrain the ERM model on our augmented dataset. This process reduces the model's reliance on spurious correlations by learning from carefully crafted samples in which this correlation does not exist. Our experiments show that across different benchmarks, our technique achieves better worst-group accuracy than the existing state-of-the-art methods. Our code is available at https://github.com/ArianYp/DDB.
>
---
#### [replaced 038] SafeWork-R1: Coevolving Safety and Intelligence under the AI-45$^{\circ}$ Law
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18576v2](http://arxiv.org/pdf/2507.18576v2)**

> **作者:** Shanghai AI Lab; :; Yicheng Bao; Guanxu Chen; Mingkang Chen; Yunhao Chen; Chiyu Chen; Lingjie Chen; Sirui Chen; Xinquan Chen; Jie Cheng; Yu Cheng; Dengke Deng; Yizhuo Ding; Dan Ding; Xiaoshan Ding; Yi Ding; Zhichen Dong; Lingxiao Du; Yuyu Fan; Xinshun Feng; Yanwei Fu; Yuxuan Gao; Ruijun Ge; Tianle Gu; Lujun Gui; Jiaxuan Guo; Qianxi He; Yuenan Hou; Xuhao Hu; Hong Huang; Kaichen Huang; Shiyang Huang; Yuxian Jiang; Shanzhe Lei; Jie Li; Lijun Li; Hao Li; Juncheng Li; Xiangtian Li; Yafu Li; Lingyu Li; Xueyan Li; Haotian Liang; Dongrui Liu; Qihua Liu; Zhixuan Liu; Bangwei Liu; Huacan Liu; Yuexiao Liu; Zongkai Liu; Chaochao Lu; Yudong Lu; Xiaoya Lu; Zhenghao Lu; Qitan Lv; Caoyuan Ma; Jiachen Ma; Xiaoya Ma; Zhongtian Ma; Lingyu Meng; Ziqi Miao; Yazhe Niu; Yuezhang Peng; Yuan Pu; Han Qi; Chen Qian; Xingge Qiao; Jingjing Qu; Jiashu Qu; Wanying Qu; Wenwen Qu; Xiaoye Qu; Qihan Ren; Qingnan Ren; Qingyu Ren; Jing Shao; Wenqi Shao; Shuai Shao; Dongxing Shi; Xin Song; Xinhao Song; Yan Teng; Xuan Tong; Yingchun Wang; Xuhong Wang; Shujie Wang; Xin Wang; Yige Wang; Yixu Wang; Yuanfu Wang; Futing Wang; Ruofan Wang; Wenjie Wang; Yajie Wang; Muhao Wei; Xiaoyu Wen; Fenghua Weng; Yuqi Wu; Yingtong Xiong; Xingcheng Xu; Chao Yang; Yue Yang; Yang Yao; Yulei Ye; Zhenyun Yin; Yi Yu; Bo Zhang; Qiaosheng Zhang; Jinxuan Zhang; Yexin Zhang; Yinqiang Zheng; Hefeng Zhou; Zhanhui Zhou; Pengyu Zhu; Qingzi Zhu; Yubo Zhu; Bowen Zhou
>
> **备注:** 47 pages, 18 figures, authors are listed in alphabetical order by their last names; v2 modifies minor issues
>
> **摘要:** We introduce SafeWork-R1, a cutting-edge multimodal reasoning model that demonstrates the coevolution of capabilities and safety. It is developed by our proposed SafeLadder framework, which incorporates large-scale, progressive, safety-oriented reinforcement learning post-training, supported by a suite of multi-principled verifiers. Unlike previous alignment methods such as RLHF that simply learn human preferences, SafeLadder enables SafeWork-R1 to develop intrinsic safety reasoning and self-reflection abilities, giving rise to safety `aha' moments. Notably, SafeWork-R1 achieves an average improvement of $46.54\%$ over its base model Qwen2.5-VL-72B on safety-related benchmarks without compromising general capabilities, and delivers state-of-the-art safety performance compared to leading proprietary models such as GPT-4.1 and Claude Opus 4. To further bolster its reliability, we implement two distinct inference-time intervention methods and a deliberative search mechanism, enforcing step-level verification. Finally, we further develop SafeWork-R1-InternVL3-78B, SafeWork-R1-DeepSeek-70B, and SafeWork-R1-Qwen2.5VL-7B. All resulting models demonstrate that safety and capability can co-evolve synergistically, highlighting the generalizability of our framework in building robust, reliable, and trustworthy general-purpose AI.
>
---
#### [replaced 039] RadMamba: Efficient Human Activity Recognition through Radar-based Micro-Doppler-Oriented Mamba State-Space Model
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.12039v2](http://arxiv.org/pdf/2504.12039v2)**

> **作者:** Yizhuo Wu; Francesco Fioranelli; Chang Gao
>
> **备注:** Under Review
>
> **摘要:** Radar-based HAR has emerged as a promising alternative to conventional monitoring approaches, such as wearable devices and camera-based systems, due to its unique privacy preservation and robustness advantages. However, existing solutions based on convolutional and recurrent neural networks, although effective, are computationally demanding during deployment. This limits their applicability in scenarios with constrained resources or those requiring multiple sensors. Advanced architectures, such as Vision Transformer (ViT) and State-Space Model (SSM) architectures, offer improved modeling capabilities and have made efforts toward lightweight designs. However, their computational complexity remains relatively high. To leverage the strengths of transformer architectures while simultaneously enhancing accuracy and reducing computational complexity, this paper introduces RadMamba, a parameter-efficient, radar micro-Doppler-oriented Mamba SSM specifically tailored for radar-based HAR. Across three diverse datasets, RadMamba matches the top-performing previous model's 99.8% classification accuracy on Dataset DIAT with only 1/400 of its parameters and equals the leading models' 92.0% accuracy on Dataset CI4R with merely 1/10 of their parameters. In scenarios with continuous sequences of actions evaluated on Dataset UoG2020, RadMamba surpasses other models with significantly higher parameter counts by at least 3%, achieving this with only 6.7k parameters. Our code is available at: https://github.com/lab-emi/AIRHAR.
>
---
#### [replaced 040] Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models
- **分类: cs.AI; cs.CV; 68T01**

- **链接: [http://arxiv.org/pdf/2507.03916v3](http://arxiv.org/pdf/2507.03916v3)**

> **作者:** Yifan Jiang; Yibo Xue; Yukun Kang; Pin Zheng; Jian Peng; Feiran Wu; Changliang Xu
>
> **备注:** Appendix at: https://github.com/PAMPAS-Lab/ANA-PPT-Anamation/blob/main/Appendix.pdf
>
> **摘要:** Slide animations, such as fade-in, fly-in, and wipe, are critical for audience engagement, efficient information delivery, and vivid visual expression. However, most AI-driven slide-generation tools still lack native animation support, and existing vision-language models (VLMs) struggle with animation tasks due to the absence of public datasets and limited temporal-reasoning capabilities. To address this gap, we release the first public dataset for slide-animation modeling: 12,000 triplets of natural-language descriptions, animation JSON files, and rendered videos, collectively covering every built-in PowerPoint effect. Using this resource, we fine-tune Qwen-2.5-VL-7B with Low-Rank Adaptation (LoRA) and achieve consistent improvements over GPT-4.1 and Gemini-2.5-Pro in BLEU-4, ROUGE-L, SPICE, and our Coverage-Order-Detail Assessment (CODA) metric, which evaluates action coverage, temporal order, and detail fidelity. On a manually created test set of slides, the LoRA model increases BLEU-4 by around 60%, ROUGE-L by 30%, and shows significant improvements in CODA-detail. This demonstrates that low-rank adaptation enables reliable temporal reasoning and generalization beyond synthetic data. Overall, our dataset, LoRA-enhanced model, and CODA metric provide a rigorous benchmark and foundation for future research on VLM-based dynamic slide generation.
>
---
#### [replaced 041] FreeQ-Graph: Free-form Querying with Semantic Consistent Scene Graph for 3D Scene Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13629v2](http://arxiv.org/pdf/2506.13629v2)**

> **作者:** Chenlu Zhan; Yufei Zhang; Gaoang Wang; Hongwei Wang
>
> **摘要:** Semantic querying in complex 3D scenes through free-form language presents a significant challenge. Existing 3D scene understanding methods use large-scale training data and CLIP to align text queries with 3D semantic features. However, their reliance on predefined vocabulary priors from training data hinders free-form semantic querying. Besides, recent advanced methods rely on LLMs for scene understanding but lack comprehensive 3D scene-level information and often overlook the potential inconsistencies in LLM-generated outputs. In our paper, we propose FreeQ-Graph, which enables Free-form Querying with a semantic consistent scene Graph for 3D scene understanding. The core idea is to encode free-form queries from a complete and accurate 3D scene graph without predefined vocabularies, and to align them with 3D consistent semantic labels, which accomplished through three key steps. We initiate by constructing a complete and accurate 3D scene graph that maps free-form objects and their relations through LLM and LVLM guidance, entirely free from training data or predefined priors. Most importantly, we align graph nodes with accurate semantic labels by leveraging 3D semantic aligned features from merged superpoints, enhancing 3D semantic consistency. To enable free-form semantic querying, we then design an LLM-based reasoning algorithm that combines scene-level and object-level information to intricate reasoning. We conducted extensive experiments on 3D semantic grounding, segmentation, and complex querying tasks, while also validating the accuracy of graph generation. Experiments on 6 datasets show that our model excels in both complex free-form semantic queries and intricate relational reasoning.
>
---
#### [replaced 042] Free-form language-based robotic reasoning and grasping
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13082v2](http://arxiv.org/pdf/2503.13082v2)**

> **作者:** Runyu Jiao; Alice Fasoli; Francesco Giuliari; Matteo Bortolon; Sergio Povoli; Guofeng Mei; Yiming Wang; Fabio Poiesi
>
> **备注:** Accepted to IROS 2025. Project website: https://tev-fbk.github.io/FreeGrasp/
>
> **摘要:** Performing robotic grasping from a cluttered bin based on human instructions is a challenging task, as it requires understanding both the nuances of free-form language and the spatial relationships between objects. Vision-Language Models (VLMs) trained on web-scale data, such as GPT-4o, have demonstrated remarkable reasoning capabilities across both text and images. But can they truly be used for this task in a zero-shot setting? And what are their limitations? In this paper, we explore these research questions via the free-form language-based robotic grasping task, and propose a novel method, FreeGrasp, leveraging the pre-trained VLMs' world knowledge to reason about human instructions and object spatial arrangements. Our method detects all objects as keypoints and uses these keypoints to annotate marks on images, aiming to facilitate GPT-4o's zero-shot spatial reasoning. This allows our method to determine whether a requested object is directly graspable or if other objects must be grasped and removed first. Since no existing dataset is specifically designed for this task, we introduce a synthetic dataset FreeGraspData by extending the MetaGraspNetV2 dataset with human-annotated instructions and ground-truth grasping sequences. We conduct extensive analyses with both FreeGraspData and real-world validation with a gripper-equipped robotic arm, demonstrating state-of-the-art performance in grasp reasoning and execution. Project website: https://tev-fbk.github.io/FreeGrasp/.
>
---
#### [replaced 043] GDSR: Global-Detail Integration through Dual-Branch Network with Wavelet Losses for Remote Sensing Image Super-Resolution
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.01460v3](http://arxiv.org/pdf/2501.01460v3)**

> **作者:** Qiwei Zhu; Kai Li; Guojing Zhang; Xiaoying Wang; Jianqiang Huang; Xilai Li
>
> **备注:** GDSR: Global-Detail Integration through Dual-Branch Network with Wavelet Losses for Remote Sensing Image Super-Resolution
>
> **摘要:** In recent years, deep neural networks, including Convolutional Neural Networks, Transformers, and State Space Models, have achieved significant progress in Remote Sensing Image (RSI) Super-Resolution (SR). However, existing SR methods typically overlook the complementary relationship between global and local dependencies. These methods either focus on capturing local information or prioritize global information, which results in models that are unable to effectively capture both global and local features simultaneously. Moreover, their computational cost becomes prohibitive when applied to large-scale RSIs. To address these challenges, we introduce the novel application of Receptance Weighted Key Value (RWKV) to RSI-SR, which captures long-range dependencies with linear complexity. To simultaneously model global and local features, we propose the Global-Detail dual-branch structure, GDSR, which performs SR by paralleling RWKV and convolutional operations to handle large-scale RSIs. Furthermore, we introduce the Global-Detail Reconstruction Module (GDRM) as an intermediary between the two branches to bridge their complementary roles. In addition, we propose the Dual-Group Multi-Scale Wavelet Loss, a wavelet-domain constraint mechanism via dual-group subband strategy and cross-resolution frequency alignment for enhanced reconstruction fidelity in RSI-SR. Extensive experiments under two degradation methods on several benchmarks, including AID, UCMerced, and RSSRD-QH, demonstrate that GSDR outperforms the state-of-the-art Transformer-based method HAT by an average of 0.09 dB in PSNR, while using only 63% of its parameters and 51% of its FLOPs, achieving an inference speed 3.2 times faster.
>
---
#### [replaced 044] LUDVIG: Learning-Free Uplifting of 2D Visual Features to Gaussian Splatting Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.14462v5](http://arxiv.org/pdf/2410.14462v5)**

> **作者:** Juliette Marrie; Romain Menegaux; Michael Arbel; Diane Larlus; Julien Mairal
>
> **备注:** Published at ICCV 2025. Project page: https://juliettemarrie.github.io/ludvig
>
> **摘要:** We address the problem of extending the capabilities of vision foundation models such as DINO, SAM, and CLIP, to 3D tasks. Specifically, we introduce a novel method to uplift 2D image features into Gaussian Splatting representations of 3D scenes. Unlike traditional approaches that rely on minimizing a reconstruction loss, our method employs a simpler and more efficient feature aggregation technique, augmented by a graph diffusion mechanism. Graph diffusion refines 3D features, such as coarse segmentation masks, by leveraging 3D geometry and pairwise similarities induced by DINOv2. Our approach achieves performance comparable to the state of the art on multiple downstream tasks while delivering significant speed-ups. Notably, we obtain competitive segmentation results using only generic DINOv2 features, despite DINOv2 not being trained on millions of annotated segmentation masks like SAM. When applied to CLIP features, our method demonstrates strong performance in open-vocabulary object segmentation tasks, highlighting the versatility of our approach.
>
---
#### [replaced 045] OrthoInsight: Rib Fracture Diagnosis and Report Generation Based on Multi-Modal Large Models
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13993v2](http://arxiv.org/pdf/2507.13993v2)**

> **作者:** Ningyong Wu; Jinzhi Wang; Wenhong Zhao; Chenzhan Yu; Zhigang Xiu; Duwei Dai
>
> **备注:** This paper contains significant issues in the data preprocessing stage, which led to non-reproducible results. We are currently correcting the errors and will submit a revised version in the future.
>
> **摘要:** The growing volume of medical imaging data has increased the need for automated diagnostic tools, especially for musculoskeletal injuries like rib fractures, commonly detected via CT scans. Manual interpretation is time-consuming and error-prone. We propose OrthoInsight, a multi-modal deep learning framework for rib fracture diagnosis and report generation. It integrates a YOLOv9 model for fracture detection, a medical knowledge graph for retrieving clinical context, and a fine-tuned LLaVA language model for generating diagnostic reports. OrthoInsight combines visual features from CT images with expert textual data to deliver clinically useful outputs. Evaluated on 28,675 annotated CT images and expert reports, it achieves high performance across Diagnostic Accuracy, Content Completeness, Logical Coherence, and Clinical Guidance Value, with an average score of 4.28, outperforming models like GPT-4 and Claude-3. This study demonstrates the potential of multi-modal learning in transforming medical image analysis and providing effective support for radiologists.
>
---
#### [replaced 046] Prediction of microstructural representativity from a single image
- **分类: stat.CO; cs.CV; stat.AP**

- **链接: [http://arxiv.org/pdf/2410.19568v2](http://arxiv.org/pdf/2410.19568v2)**

> **作者:** Amir Dahari; Ronan Docherty; Steve Kench; Samuel J. Cooper
>
> **摘要:** In this study, we present a method for predicting the representativity of the phase fraction observed in a single image (2D or 3D) of a material. Traditional approaches often require large datasets and extensive statistical analysis to estimate the Integral Range, a key factor in determining the variance of microstructural properties. Our method leverages the Two-Point Correlation function to directly estimate the variance from a single image, thereby enabling phase fraction prediction with associated confidence levels. We validate our approach using open-source datasets, demonstrating its efficacy across diverse microstructures. This technique significantly reduces the data requirements for representativity analysis, providing a practical tool for material scientists and engineers working with limited microstructural data. To make the method easily accessible, we have created a web-application, www.imagerep.io, for quick, simple and informative use of the method.
>
---
#### [replaced 047] Leveraging Modified Ex Situ Tomography Data for Segmentation of In Situ Synchrotron X-Ray Computed Tomography
- **分类: cond-mat.mtrl-sci; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.19200v3](http://arxiv.org/pdf/2504.19200v3)**

> **作者:** Tristan Manchester; Adam Anders; Julio Spadotto; Hannah Eccleston; William Beavan; Hugues Arcis; Brian J. Connolly
>
> **摘要:** In situ synchrotron X-ray computed tomography enables dynamic material studies. However, automated segmentation remains challenging due to complex imaging artefacts - like ring and cupping effects - and limited training data. We present a methodology for deep learning-based segmentation by transforming high-quality ex situ laboratory data to train models for segmentation of in situ synchrotron data, demonstrated through a metal oxide dissolution study. Using a modified SegFormer architecture, our approach achieves segmentation performance (94.7% IoU) that matches human inter-annotator reliability (94.6% IoU). This indicates the model has reached the practical upper bound for this task, while reducing processing time by 2 orders of magnitude per 3D dataset compared to manual segmentation. The method maintains robust performance over significant morphological changes during experiments, despite training only on static specimens. This methodology can be readily applied to diverse materials systems, enabling the efficient analysis of the large volumes of time-resolved tomographic data generated in typical in situ experiments across scientific disciplines.
>
---
#### [replaced 048] Visual Enumeration Remains Challenging for Multimodal Generative AI
- **分类: cs.CV; cs.AI; cs.NE**

- **链接: [http://arxiv.org/pdf/2402.03328v3](http://arxiv.org/pdf/2402.03328v3)**

> **作者:** Alberto Testolin; Kuinan Hou; Marco Zorzi
>
> **摘要:** Many animal species can approximately judge the number of objects in a visual scene at a single glance, and humans can further determine the exact cardinality of a set by deploying systematic counting procedures. In contrast, it has been observed that even state-of-the-art AI systems have very limited enumeration skills. In this work, we propose two benchmark tasks inspired by cognitive science that allow to precisely evaluate the visual enumeration capabilities of multimodal foundation models, thereby providing an objective measure of their number sense and counting level. We consider popular visual question answering models (BLIP, LLaVA and ViLT) as well as advanced image-to-text (Gemini, GPT and Qwen) and text-to-image (DALL-E, FLUX and Stable Diffusion) AI systems. Our analyses show that even the most advanced models cannot reliably name the number of objects in simple visual stimuli or generate images containing a target number of items, as indexed by their low accuracy in both types of tasks. Especially for numbers outside the subitizing range, their responses are often far from the target numerosity, and, in stark contrast with human behavior, in many cases the distribution of errors depends on the object category. We also observe some striking mistakes with small numbers. Our findings demonstrate that developing an intuitive visual understanding of number remains challenging for AI models and that merely increasing model size might not be a viable strategy to promote the emergence of systematic counting skills. We release the full code of our benchmark to facilitate the evaluation of enumeration skills in future AI systems.
>
---
#### [replaced 049] Handcrafted vs. Deep Radiomics vs. Fusion vs. Deep Learning: A Comprehensive Review of Machine Learning -Based Cancer Outcome Prediction in PET and SPECT Imaging
- **分类: physics.med-ph; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16065v2](http://arxiv.org/pdf/2507.16065v2)**

> **作者:** Mohammad R. Salmanpour; Somayeh Sadat Mehrnia; Sajad Jabarzadeh Ghandilu; Zhino Safahi; Sonya Falahati; Shahram Taeb; Ghazal Mousavi; Mehdi Maghsoudi; Ahmad Shariftabrizi; Ilker Hacihaliloglu; Arman Rahmim
>
> **摘要:** Machine learning (ML), including deep learning (DL) and radiomics-based methods, is increasingly used for cancer outcome prediction with PET and SPECT imaging. However, the comparative performance of handcrafted radiomics features (HRF), deep radiomics features (DRF), DL models, and hybrid fusion approaches remains inconsistent across clinical applications. This systematic review analyzed 226 studies published from 2020 to 2025 that applied ML to PET or SPECT imaging for outcome prediction. Each study was evaluated using a 59-item framework covering dataset construction, feature extraction, validation methods, interpretability, and risk of bias. We extracted key details including model type, cancer site, imaging modality, and performance metrics such as accuracy and area under the curve (AUC). PET-based studies (95%) generally outperformed those using SPECT, likely due to higher spatial resolution and sensitivity. DRF models achieved the highest mean accuracy (0.862), while fusion models yielded the highest AUC (0.861). ANOVA confirmed significant differences in performance (accuracy: p=0.0006, AUC: p=0.0027). Common limitations included inadequate handling of class imbalance (59%), missing data (29%), and low population diversity (19%). Only 48% of studies adhered to IBSI standards. These findings highlight the need for standardized pipelines, improved data quality, and explainable AI to support clinical integration.
>
---
#### [replaced 050] M-SpecGene: Generalized Foundation Model for RGBT Multispectral Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16318v2](http://arxiv.org/pdf/2507.16318v2)**

> **作者:** Kailai Zhou; Fuqiang Yang; Shixian Wang; Bihan Wen; Chongde Zi; Linsen Chen; Qiu Shen; Xun Cao
>
> **备注:** accepted by ICCV2025
>
> **摘要:** RGB-Thermal (RGBT) multispectral vision is essential for robust perception in complex environments. Most RGBT tasks follow a case-by-case research paradigm, relying on manually customized models to learn task-oriented representations. Nevertheless, this paradigm is inherently constrained by artificial inductive bias, modality bias, and data bottleneck. To address these limitations, we make the initial attempt to build a Generalized RGBT MultiSpectral foundation model (M-SpecGene), which aims to learn modality-invariant representations from large-scale broad data in a self-supervised manner. M-SpecGene provides new insights into multispectral fusion and integrates prior case-by-case studies into a unified paradigm. Considering the unique characteristic of information imbalance in RGBT data, we introduce the Cross-Modality Structural Sparsity (CMSS) metric to quantify the information density across two modalities. Then we develop the GMM-CMSS progressive masking strategy to facilitate a flexible, easy-to-hard, and object-centric pre-training process. Comprehensive experiments validate M-SpecGene's generalizability across eleven datasets for four RGBT downstream tasks. The code will be available at https://github.com/CalayZhou/M-SpecGene.
>
---
#### [replaced 051] Rethinking Multi-Modal Object Detection from the Perspective of Mono-Modality Feature Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11780v2](http://arxiv.org/pdf/2503.11780v2)**

> **作者:** Tianyi Zhao; Boyang Liu; Yanglei Gao; Yiming Sun; Maoxun Yuan; Xingxing Wei
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Multi-Modal Object Detection (MMOD), due to its stronger adaptability to various complex environments, has been widely applied in various applications. Extensive research is dedicated to the RGB-IR object detection, primarily focusing on how to integrate complementary features from RGB-IR modalities. However, they neglect the mono-modality insufficient learning problem, which arises from decreased feature extraction capability in multi-modal joint learning. This leads to a prevalent but unreasonable phenomenon\textemdash Fusion Degradation, which hinders the performance improvement of the MMOD model. Motivated by this, in this paper, we introduce linear probing evaluation to the multi-modal detectors and rethink the multi-modal object detection task from the mono-modality learning perspective. Therefore, we construct a novel framework called M$^2$D-LIF, which consists of the Mono-Modality Distillation (M$^2$D) method and the Local Illumination-aware Fusion (LIF) module. The M$^2$D-LIF framework facilitates the sufficient learning of mono-modality during multi-modal joint training and explores a lightweight yet effective feature fusion manner to achieve superior object detection performance. Extensive experiments conducted on three MMOD datasets demonstrate that our M$^2$D-LIF effectively mitigates the Fusion Degradation phenomenon and outperforms the previous SOTA detectors. The codes are available at https://github.com/Zhao-Tian-yi/M2D-LIF.
>
---
#### [replaced 052] Hydra-NeXt: Robust Closed-Loop Driving with Open-Loop Training
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12030v2](http://arxiv.org/pdf/2503.12030v2)**

> **作者:** Zhenxin Li; Shihao Wang; Shiyi Lan; Zhiding Yu; Zuxuan Wu; Jose M. Alvarez
>
> **摘要:** End-to-end autonomous driving research currently faces a critical challenge in bridging the gap between open-loop training and closed-loop deployment. Current approaches are trained to predict trajectories in an open-loop environment, which struggle with quick reactions to other agents in closed-loop environments and risk generating kinematically infeasible plans due to the gap between open-loop training and closed-loop driving. In this paper, we introduce Hydra-NeXt, a novel multi-branch planning framework that unifies trajectory prediction, control prediction, and a trajectory refinement network in one model. Unlike current open-loop trajectory prediction models that only handle general-case planning, Hydra-NeXt further utilizes a control decoder to focus on short-term actions, which enables faster responses to dynamic situations and reactive agents. Moreover, we propose the Trajectory Refinement module to augment and refine the planning decisions by effectively adhering to kinematic constraints in closed-loop environments. This unified approach bridges the gap between open-loop training and closed-loop driving, demonstrating superior performance of 65.89 Driving Score (DS) and 48.20% Success Rate (SR) on the Bench2Drive dataset without relying on external experts for data collection. Hydra-NeXt surpasses the previous state-of-the-art by 22.98 DS and 17.49 SR, marking a significant advancement in autonomous driving. Code will be available at https://github.com/woxihuanjiangguo/Hydra-NeXt.
>
---
#### [replaced 053] Mitigating Object Hallucinations via Sentence-Level Early Intervention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12455v2](http://arxiv.org/pdf/2507.12455v2)**

> **作者:** Shangpin Peng; Senqiao Yang; Li Jiang; Zhuotao Tian
>
> **摘要:** Multimodal large language models (MLLMs) have revolutionized cross-modal understanding but continue to struggle with hallucinations - fabricated content contradicting visual inputs. Existing hallucination mitigation methods either incur prohibitive computational costs or introduce distribution mismatches between training data and model outputs. We identify a critical insight: hallucinations predominantly emerge at the early stages of text generation and propagate through subsequent outputs. To address this, we propose SENTINEL (Sentence-level Early iNtervention Through IN-domain prEference Learning), a framework that eliminates dependency on human annotations. Specifically, we first bootstrap high-quality in-domain preference pairs by iteratively sampling model outputs, validating object existence through cross-checking with two open-vocabulary detectors, and classifying sentences into hallucinated/non-hallucinated categories. Subsequently, we use context-coherent positive samples and hallucinated negative samples to build context-aware preference data iteratively. Finally, we train models using a context-aware preference loss (C-DPO) that emphasizes discriminative learning at the sentence level where hallucinations initially manifest. Experimental results show that SENTINEL can reduce hallucinations by over 90% compared to the original model and outperforms the previous state-of-the-art method on both hallucination benchmarks and general capabilities benchmarks, demonstrating its superiority and generalization ability. The models, datasets, and code are available at https://github.com/pspdada/SENTINEL.
>
---
#### [replaced 054] Perpetua: Multi-Hypothesis Persistence Modeling for Semi-Static Environments
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18808v2](http://arxiv.org/pdf/2507.18808v2)**

> **作者:** Miguel Saavedra-Ruiz; Samer B. Nashed; Charlie Gauthier; Liam Paull
>
> **备注:** Accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) Code available at https://github.com/montrealrobotics/perpetua-code. Webpage and additional videos at https://montrealrobotics.ca/perpetua/
>
> **摘要:** Many robotic systems require extended deployments in complex, dynamic environments. In such deployments, parts of the environment may change between subsequent robot observations. Most robotic mapping or environment modeling algorithms are incapable of representing dynamic features in a way that enables predicting their future state. Instead, they opt to filter certain state observations, either by removing them or some form of weighted averaging. This paper introduces Perpetua, a method for modeling the dynamics of semi-static features. Perpetua is able to: incorporate prior knowledge about the dynamics of the feature if it exists, track multiple hypotheses, and adapt over time to enable predicting of future feature states. Specifically, we chain together mixtures of "persistence" and "emergence" filters to model the probability that features will disappear or reappear in a formal Bayesian framework. The approach is an efficient, scalable, general, and robust method for estimating the states of features in an environment, both in the present as well as at arbitrary future times. Through experiments on simulated and real-world data, we find that Perpetua yields better accuracy than similar approaches while also being online adaptable and robust to missing observations.
>
---
#### [replaced 055] InstructFLIP: Exploring Unified Vision-Language Model for Face Anti-spoofing
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.12060v2](http://arxiv.org/pdf/2507.12060v2)**

> **作者:** Kun-Hsiang Lin; Yu-Wen Tseng; Kang-Yang Huang; Jhih-Ciang Wu; Wen-Huang Cheng
>
> **备注:** Accepted by MM'25
>
> **摘要:** Face anti-spoofing (FAS) aims to construct a robust system that can withstand diverse attacks. While recent efforts have concentrated mainly on cross-domain generalization, two significant challenges persist: limited semantic understanding of attack types and training redundancy across domains. We address the first by integrating vision-language models (VLMs) to enhance the perception of visual input. For the second challenge, we employ a meta-domain strategy to learn a unified model that generalizes well across multiple domains. Our proposed InstructFLIP is a novel instruction-tuned framework that leverages VLMs to enhance generalization via textual guidance trained solely on a single domain. At its core, InstructFLIP explicitly decouples instructions into content and style components, where content-based instructions focus on the essential semantics of spoofing, and style-based instructions consider variations related to the environment and camera characteristics. Extensive experiments demonstrate the effectiveness of InstructFLIP by outperforming SOTA models in accuracy and substantially reducing training redundancy across diverse domains in FAS. Project website is available at https://kunkunlin1221.github.io/InstructFLIP.
>
---
#### [replaced 056] MemeBLIP2: A novel lightweight multimodal system to detect harmful memes
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.21226v3](http://arxiv.org/pdf/2504.21226v3)**

> **作者:** Jiaqi Liu; Ran Tong; Aowei Shen; Shuzheng Li; Changlin Yang; Lisha Xu
>
> **备注:** 11 pages, 3 figures. Accepted at the First Workshop on Multimodal Knowledge and Language Modeling (MKLM), IJCAI-25
>
> **摘要:** Memes often merge visuals with brief text to share humor or opinions, yet some memes contain harmful messages such as hate speech. In this paper, we introduces MemeBLIP2, a light weight multimodal system that detects harmful memes by combining image and text features effectively. We build on previous studies by adding modules that align image and text representations into a shared space and fuse them for better classification. Using BLIP-2 as the core vision-language model, our system is evaluated on the PrideMM datasets. The results show that MemeBLIP2 can capture subtle cues in both modalities, even in cases with ironic or culturally specific content, thereby improving the detection of harmful material.
>
---
#### [replaced 057] "Principal Components" Enable A New Language of Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08685v2](http://arxiv.org/pdf/2503.08685v2)**

> **作者:** Xin Wen; Bingchen Zhao; Ismail Elezi; Jiankang Deng; Xiaojuan Qi
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** We introduce a novel visual tokenization framework that embeds a provable PCA-like structure into the latent token space. While existing visual tokenizers primarily optimize for reconstruction fidelity, they often neglect the structural properties of the latent space--a critical factor for both interpretability and downstream tasks. Our method generates a 1D causal token sequence for images, where each successive token contributes non-overlapping information with mathematically guaranteed decreasing explained variance, analogous to principal component analysis. This structural constraint ensures the tokenizer extracts the most salient visual features first, with each subsequent token adding diminishing yet complementary information. Additionally, we identified and resolved a semantic-spectrum coupling effect that causes the unwanted entanglement of high-level semantic content and low-level spectral details in the tokens by leveraging a diffusion decoder. Experiments demonstrate that our approach achieves state-of-the-art reconstruction performance and enables better interpretability to align with the human vision system. Moreover, autoregressive models trained on our token sequences achieve performance comparable to current state-of-the-art methods while requiring fewer tokens for training and inference.
>
---
#### [replaced 058] Vec2Face+ for Face Dataset Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17192v2](http://arxiv.org/pdf/2507.17192v2)**

> **作者:** Haiyu Wu; Jaskirat Singh; Sicong Tian; Liang Zheng; Kevin W. Bowyer
>
> **摘要:** When synthesizing identities as face recognition training data, it is generally believed that large inter-class separability and intra-class attribute variation are essential for synthesizing a quality dataset. % This belief is generally correct, and this is what we aim for. However, when increasing intra-class variation, existing methods overlook the necessity of maintaining intra-class identity consistency. % To address this and generate high-quality face training data, we propose Vec2Face+, a generative model that creates images directly from image features and allows for continuous and easy control of face identities and attributes. Using Vec2Face+, we obtain datasets with proper inter-class separability and intra-class variation and identity consistency using three strategies: 1) we sample vectors sufficiently different from others to generate well-separated identities; 2) we propose an AttrOP algorithm for increasing general attribute variations; 3) we propose LoRA-based pose control for generating images with profile head poses, which is more efficient and identity-preserving than AttrOP. % Our system generates VFace10K, a synthetic face dataset with 10K identities, which allows an FR model to achieve state-of-the-art accuracy on seven real-world test sets. Scaling the size to 4M and 12M images, the corresponding VFace100K and VFace300K datasets yield higher accuracy than the real-world training dataset, CASIA-WebFace, on five real-world test sets. This is the first time a synthetic dataset beats the CASIA-WebFace in average accuracy. In addition, we find that only 1 out of 11 synthetic datasets outperforms random guessing (\emph{i.e., 50\%}) in twin verification and that models trained with synthetic identities are more biased than those trained with real identities. Both are important aspects for future investigation. Code is available at https://github.com/HaiyuWu/Vec2Face_plus
>
---
#### [replaced 059] PaRCE: Probabilistic and Reconstruction-based Competency Estimation for CNN-based Image Classification
- **分类: cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2411.16715v3](http://arxiv.org/pdf/2411.16715v3)**

> **作者:** Sara Pohland; Claire Tomlin
>
> **备注:** arXiv admin note: text overlap with arXiv:2409.06111
>
> **摘要:** Convolutional neural networks (CNNs) are extremely popular and effective for image classification tasks but tend to be overly confident in their predictions. Various works have sought to quantify uncertainty associated with these models, detect out-of-distribution (OOD) inputs, or identify anomalous regions in an image, but limited work has sought to develop a holistic approach that can accurately estimate perception model confidence across various sources of uncertainty. We develop a probabilistic and reconstruction-based competency estimation (PaRCE) method and compare it to existing approaches for uncertainty quantification and OOD detection. We find that our method can best distinguish between correctly classified, misclassified, and OOD samples with anomalous regions, as well as between samples with visual image modifications resulting in high, medium, and low prediction accuracy. We describe how to extend our approach for anomaly localization tasks and demonstrate the ability of our approach to distinguish between regions in an image that are familiar to the perception model from those that are unfamiliar. We find that our method generates interpretable scores that most reliably capture a holistic notion of perception model confidence.
>
---
#### [replaced 060] Investigation of the Challenges of Underwater-Visual-Monocular-SLAM
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2306.08738v2](http://arxiv.org/pdf/2306.08738v2)**

> **作者:** Michele Grimaldi; David Nakath; Mengkun She; Kevin Köser
>
> **摘要:** In this paper, we present a comprehensive investigation of the challenges of Monocular Visual Simultaneous Localization and Mapping (vSLAM) methods for underwater robots. While significant progress has been made in state estimation methods that utilize visual data in the past decade, most evaluations have been limited to controlled indoor and urban environments, where impressive performance was demonstrated. However, these techniques have not been extensively tested in extremely challenging conditions, such as underwater scenarios where factors such as water and light conditions, robot path, and depth can greatly impact algorithm performance. Hence, our evaluation is conducted in real-world AUV scenarios as well as laboratory settings which provide precise external reference. A focus is laid on understanding the impact of environmental conditions, such as optical properties of the water and illumination scenarios, on the performance of monocular vSLAM methods. To this end, we first show that all methods perform very well in in-air settings and subsequently show the degradation of their performance in challenging underwater environments. The final goal of this study is to identify techniques that can improve accuracy and robustness of SLAM methods in such conditions. To achieve this goal, we investigate the potential of image enhancement techniques to improve the quality of input images used by the SLAM methods, specifically in low visibility and extreme lighting scenarios in scattering media. We present a first evaluation on calibration maneuvers and simple image restoration techniques to determine their ability to enable or enhance the performance of monocular SLAM methods in underwater environments.
>
---
#### [replaced 061] Learning Multi-frame and Monocular Prior for Estimating Geometry in Dynamic Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01737v3](http://arxiv.org/pdf/2505.01737v3)**

> **作者:** Seong Hyeon Park; Jinwoo Shin
>
> **备注:** This paper was supported by RLWRLD
>
> **摘要:** In monocular videos that capture dynamic scenes, estimating the 3D geometry of video contents has been a fundamental challenge in computer vision. Specifically, the task is significantly challenged by the object motion, where existing models are limited to predict only partial attributes of the dynamic scenes, such as depth or pointmaps spanning only over a pair of frames. Since these attributes are inherently noisy under multiple frames, test-time global optimizations are often employed to fully recover the geometry, which is liable to failure and incurs heavy inference costs. To address the challenge, we present a new model, coined MMP, to estimate the geometry in a feed-forward manner, which produces a dynamic pointmap representation that evolves over multiple frames. Specifically, based on the recent Siamese architecture, we introduce a new trajectory encoding module to project point-wise dynamics on the representation for each frame, which can provide significantly improved expressiveness for dynamic scenes. In our experiments, we find MMP can achieve state-of-the-art quality in feed-forward pointmap prediction, e.g., 15.1% enhancement in the regression error.
>
---
#### [replaced 062] PUMA: Layer-Pruned Language Model for Efficient Unified Multimodal Retrieval with Modality-Adaptive Learning
- **分类: cs.MM; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08064v2](http://arxiv.org/pdf/2507.08064v2)**

> **作者:** Yibo Lyu; Rui Shao; Gongwei Chen; Yijie Zhu; Weili Guan; Liqiang Nie
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** As multimedia content expands, the demand for unified multimodal retrieval (UMR) in real-world applications increases. Recent work leverages multimodal large language models (MLLMs) to tackle this task. However, their large parameter size results in high training costs and low inference efficiency. To address this, we propose PUMA: a Layer-Pruned Language Model for Efficient Unified Multimodal Retrieval with Modality-Adaptive Learning. Our approach improves UMR from both structural and learning perspectives. (1) Structurally, we propose Layer-Pruned Self-Distillation, which prunes MLLMs by keeping only shallow layers while distilling features from dropped deep layers as teacher signals. This reduces parameters and preserves representation capability. (2) On the learning side, we introduce Modality-Adaptive Contrastive Learning Loss (MAC-Loss), which separates in-batch negatives into harder intra-modality and easier inter-modality groups based on the target modality, assigning different temperature strategies to enhance learning efficiency. Experiments show our method significantly reduces resource usage while maintaining strong performance.
>
---
#### [replaced 063] Synthetic-to-Real Camouflaged Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18911v2](http://arxiv.org/pdf/2507.18911v2)**

> **作者:** Zhihao Luo; Luojun Lin; Zheng Lin
>
> **摘要:** Due to the high cost of collection and labeling, there are relatively few datasets for camouflaged object detection (COD). In particular, for certain specialized categories, the available image dataset is insufficiently populated. Synthetic datasets can be utilized to alleviate the problem of limited data to some extent. However, directly training with synthetic datasets compared to real datasets can lead to a degradation in model performance. To tackle this problem, in this work, we investigate a new task, namely Syn-to-Real Camouflaged Object Detection (S2R-COD). In order to improve the model performance in real world scenarios, a set of annotated synthetic camouflaged images and a limited number of unannotated real images must be utilized. We propose the Cycling Syn-to-Real Domain Adaptation Framework (CSRDA), a method based on the student-teacher model. Specially, CSRDA propagates class information from the labeled source domain to the unlabeled target domain through pseudo labeling combined with consistency regularization. Considering that narrowing the intra-domain gap can improve the quality of pseudo labeling, CSRDA utilizes a recurrent learning framework to build an evolving real domain for bridging the source and target domain. Extensive experiments demonstrate the effectiveness of our framework, mitigating the problem of limited data and handcraft annotations in COD. Our code is publicly available at: https://github.com/Muscape/S2R-COD.
>
---
#### [replaced 064] Swin-TUNA : A Novel PEFT Approach for Accurate Food Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.17347v3](http://arxiv.org/pdf/2507.17347v3)**

> **作者:** Haotian Chen; Zhiyong Xiao
>
> **备注:** After discussion among the authors, some parts of the paper are deemed inappropriate and will be revised and resubmitted
>
> **摘要:** In the field of food image processing, efficient semantic segmentation techniques are crucial for industrial applications. However, existing large-scale Transformer-based models (such as FoodSAM) face challenges in meeting practical deploymentrequirements due to their massive parameter counts and high computational resource demands. This paper introduces TUNable Adapter module (Swin-TUNA), a Parameter Efficient Fine-Tuning (PEFT) method that integrates multiscale trainable adapters into the Swin Transformer architecture, achieving high-performance food image segmentation by updating only 4% of the parameters. The core innovation of Swin-TUNA lies in its hierarchical feature adaptation mechanism: it designs separable convolutions in depth and dimensional mappings of varying scales to address the differences in features between shallow and deep networks, combined with a dynamic balancing strategy for tasks-agnostic and task-specific features. Experiments demonstrate that this method achieves mIoU of 50.56% and 74.94% on the FoodSeg103 and UECFoodPix Complete datasets, respectively, surpassing the fully parameterized FoodSAM model while reducing the parameter count by 98.7% (to only 8.13M). Furthermore, Swin-TUNA exhibits faster convergence and stronger generalization capabilities in low-data scenarios, providing an efficient solution for assembling lightweight food image.
>
---
#### [replaced 065] DSwinIR: Rethinking Window-based Attention for Image Restoration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.04869v2](http://arxiv.org/pdf/2504.04869v2)**

> **作者:** Gang Wu; Junjun Jiang; Kui Jiang; Xianming Liu; Liqiang Nie
>
> **摘要:** Image restoration has witnessed significant advancements with the development of deep learning models. Especially Transformer-based models, particularly those leveraging window-based self-attention, have become a dominant force in image restoration. However, their performance is fundamentally constrained by the rigid, non-overlapping window partitioning scheme, which leads to two critical limitations: insufficient feature interaction across window boundaries and content-agnostic receptive fields that cannot adapt to diverse image structures. Existing methods often rely on heuristic patterns to mitigate these issues, rather than addressing the root cause. In this paper, we propose the Deformable Sliding Window Transformer (DSwinIR), a new foundational backbone architecture that systematically overcomes these limitations. At the heart of DSwinIR is the proposed novel Deformable Sliding Window (DSwin) Attention. This mechanism introduces two fundamental innovations. First, it replaces the rigid partitioning with a token-centric sliding window paradigm, ensuring seamless cross-window information flow and effectively eliminating boundary artifacts. Second, it incorporates a content-aware deformable sampling strategy, which allows the attention mechanism to learn data-dependent offsets and dynamically shape its receptive fields to focus on the most informative image regions. This synthesis endows the model with both strong locality-aware inductive biases and powerful, adaptive long-range modeling capabilities. Extensive experiments show that DSwinIR sets a new state-of-the-art across a wide spectrum of image restoration tasks. For instance, in all-in-one restoration, our DSwinIR surpasses the most recent backbone GridFormer by over 0.53 dB on the three-task benchmark and a remarkable 0.86 dB on the five-task benchmark.
>
---
#### [replaced 066] MTMamba++: Enhancing Multi-Task Dense Scene Understanding via Mamba-Based Decoders
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.15101v2](http://arxiv.org/pdf/2408.15101v2)**

> **作者:** Baijiong Lin; Weisen Jiang; Pengguang Chen; Shu Liu; Ying-Cong Chen
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** Multi-task dense scene understanding, which trains a model for multiple dense prediction tasks, has a wide range of application scenarios. Capturing long-range dependency and enhancing cross-task interactions are crucial to multi-task dense prediction. In this paper, we propose MTMamba++, a novel architecture for multi-task scene understanding featuring with a Mamba-based decoder. It contains two types of core blocks: self-task Mamba (STM) block and cross-task Mamba (CTM) block. STM handles long-range dependency by leveraging state-space models, while CTM explicitly models task interactions to facilitate information exchange across tasks. We design two types of CTM block, namely F-CTM and S-CTM, to enhance cross-task interaction from feature and semantic perspectives, respectively. Extensive experiments on NYUDv2, PASCAL-Context, and Cityscapes datasets demonstrate the superior performance of MTMamba++ over CNN-based, Transformer-based, and diffusion-based methods while maintaining high computational efficiency. The code is available at https://github.com/EnVision-Research/MTMamba.
>
---
#### [replaced 067] Learning to Unlearn while Retaining: Combating Gradient Conflicts in Machine Unlearning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06339v2](http://arxiv.org/pdf/2503.06339v2)**

> **作者:** Gaurav Patel; Qiang Qiu
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Machine Unlearning has recently garnered significant attention, aiming to selectively remove knowledge associated with specific data while preserving the model's performance on the remaining data. A fundamental challenge in this process is balancing effective unlearning with knowledge retention, as naive optimization of these competing objectives can lead to conflicting gradients, hindering convergence and degrading overall performance. To address this issue, we propose Learning to Unlearn while Retaining, aimed to mitigate gradient conflicts between unlearning and retention objectives. Our approach strategically avoids conflicts through an implicit gradient regularization mechanism that emerges naturally within the proposed framework. This prevents conflicting gradients between unlearning and retention, leading to effective unlearning while preserving the model's utility. We validate our approach across both discriminative and generative tasks, demonstrating its effectiveness in achieving unlearning without compromising performance on remaining data. Our results highlight the advantages of avoiding such gradient conflicts, outperforming existing methods that fail to account for these interactions.
>
---
#### [replaced 068] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03515v3](http://arxiv.org/pdf/2412.03515v3)**

> **作者:** Shengyuan Zhang; An Zhao; Ling Yang; Zejian Li; Chenye Meng; Haoran Xu; Tianrun Chen; AnYang Wei; Perry Pengyun GU; Lingyun Sun
>
> **备注:** This paper is accepted by ICCV'25(Oral), the model and code are publicly available on https://github.com/happyw1nd/ScoreLiDAR
>
> **摘要:** Diffusion models have been applied to 3D LiDAR scene completion due to their strong training stability and high completion quality. However, the slow sampling speed limits the practical application of diffusion-based scene completion models since autonomous vehicles require an efficient perception of surrounding environments. This paper proposes a novel distillation method tailored for 3D Li- DAR scene completion models, dubbed ScoreLiDAR, which achieves efficient yet high-quality scene completion. Score- LiDAR enables the distilled model to sample in significantly fewer steps after distillation. To improve completion quality, we also introduce a novel Structural Loss, which encourages the distilled model to capture the geometric structure of the 3D LiDAR scene. The loss contains a scene-wise term constraining the holistic structure and a point-wise term constraining the key landmark points and their relative configuration. Extensive experiments demonstrate that ScoreLiDAR significantly accelerates the completion time from 30.55 to 5.37 seconds per frame (>5x) on SemanticKITTI and achieves superior performance compared to state-of-the-art 3D LiDAR scene completion models. Our model and code are publicly available on https://github.com/happyw1nd/ScoreLiDAR.
>
---
#### [replaced 069] Explainable Synthetic Image Detection through Diffusion Timestep Ensembling
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.06201v2](http://arxiv.org/pdf/2503.06201v2)**

> **作者:** Yixin Wu; Feiran Zhang; Tianyuan Shi; Ruicheng Yin; Zhenghua Wang; Zhenliang Gan; Xiaohua Wang; Changze Lv; Xiaoqing Zheng; Xuanjing Huang
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** Recent advances in diffusion models have enabled the creation of deceptively real images, posing significant security risks when misused. In this study, we empirically show that different timesteps of DDIM inversion reveal varying subtle distinctions between synthetic and real images that are extractable for detection, in the forms of such as Fourier power spectrum high-frequency discrepancies and inter-pixel variance distributions. Based on these observations, we propose a novel synthetic image detection method that directly utilizes features of intermediately noised images by training an ensemble on multiple noised timesteps, circumventing conventional reconstruction-based strategies. To enhance human comprehension, we introduce a metric-grounded explanation generation and refinement module to identify and explain AI-generated flaws. Additionally, we construct the GenHard and GenExplain benchmarks to provide detection samples of greater difficulty and high-quality rationales for fake images. Extensive experiments show that our method achieves state-of-the-art performance with 98.91% and 95.89% detection accuracy on regular and challenging samples respectively, and demonstrates generalizability and robustness. Our code and datasets are available at https://github.com/Shadowlized/ESIDE.
>
---
#### [replaced 070] An Effective UNet Using Feature Interaction and Fusion for Organ Segmentation in Medical Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.05324v2](http://arxiv.org/pdf/2409.05324v2)**

> **作者:** Xiaolin Gou; Chuanlin Liao; Jizhe Zhou; Fengshuo Ye; Yi Lin
>
> **摘要:** Nowadays, pre-trained encoders are widely used in medical image segmentation due to their strong capability in extracting rich and generalized feature representations. However, existing methods often fail to fully leverage these features, limiting segmentation performance. In this work, a novel U-shaped model is proposed to address the above issue, including three plug-and-play modules. A channel spatial interaction module is introduced to improve the quality of skip connection features by modeling inter-stage interactions between the encoder and decoder. A channel attention-based module integrating squeeze-and-excitation mechanisms with convolutional layers is employed in the decoder blocks to strengthen the representation of critical features while suppressing irrelevant ones. A multi-level fusion module is designed to aggregate multi-scale decoder features, improving spatial detail and consistency in the final prediction. Comprehensive experiments on the synapse multi-organ segmentation dataset and automated cardiac diagnosis challenge dataset demonstrate that the proposed model outperforms existing state-of-the-art methods, achieving the highest average Dice score of 86.05% and 92.58%, yielding improvements of 1.15% and 0.26%, respectively. In addition, the proposed model provides a balance between accuracy and computational complexity, with only 86.91 million parameters and 23.26 giga floating-point operations.
>
---
#### [replaced 071] Everything is a Video: Unifying Modalities through Next-Frame Prediction
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.10503v2](http://arxiv.org/pdf/2411.10503v2)**

> **作者:** G. Thomas Hudson; Dean Slack; Thomas Winterbottom; Jamie Sterling; Chenghao Xiao; Junjie Shentu; Noura Al Moubayed
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Multimodal learning, which involves integrating information from various modalities such as text, images, audio, and video, is pivotal for numerous complex tasks like visual question answering, cross-modal retrieval, and caption generation. Traditional approaches rely on modality-specific encoders and late fusion techniques, which can hinder scalability and flexibility when adapting to new tasks or modalities. To address these limitations, we introduce a novel framework that extends the concept of task reformulation beyond natural language processing (NLP) to multimodal learning. We propose to reformulate diverse multimodal tasks into a unified next-frame prediction problem, allowing a single model to handle different modalities without modality-specific components. This method treats all inputs and outputs as sequential frames in a video, enabling seamless integration of modalities and effective knowledge transfer across tasks. Our approach is evaluated on a range of tasks, including text-to-text, image-to-text, video-to-video, video-to-text, and audio-to-text, demonstrating the model's ability to generalize across modalities with minimal adaptation. We show that task reformulation can significantly simplify multimodal model design across various tasks, laying the groundwork for more generalized multimodal foundation models.
>
---
#### [replaced 072] PatchTraj: Dynamic Patch Representation Learning for Time-Frequency Trajectory Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.19119v2](http://arxiv.org/pdf/2507.19119v2)**

> **作者:** Yanghong Liu; Xingping Dong; Ming Li; Weixing Zhang; Yidong Lou
>
> **摘要:** Pedestrian trajectory prediction is crucial for autonomous driving and robotics. While existing point-based and grid-based methods expose two key limitations: insufficiently modeling human motion dynamics, as they fail to balance local motion details with long-range spatiotemporal dependencies, and the time representation lacks interaction with the frequency domain in modeling trajectory sequences. To address these challenges, we propose PatchTraj, a dynamic patch-based trajectory prediction framework that unifies time-domain and frequency-domain representations. Specifically, we decompose the trajectory into raw time sequences and frequency components, employing dynamic patch partitioning for multi-scale trajectory segmentation to capture hierarchical motion patterns. Each patch is processed by an adaptive embedding layer with scale-aware feature extraction, followed by hierarchical feature aggregation to model both fine-grained and long-range dependencies. The outputs of two branches interact via cross-modal attention, enabling complementary fusion of temporal and spectral cues. Finally, a Transformer encoder-decoder integrates both modalities to autoregressively predict future trajectories. Extensive experiments on ETH-UCY, SDD, NBA, and JRDB datasets demonstrate that our method achieves state-of-the-art performance with high efficiency.
>
---
#### [replaced 073] AutoLungDx: A Hybrid Deep Learning Approach for Early Lung Cancer Diagnosis Using 3D Res-U-Net, YOLOv5, and Vision Transformers
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2305.00046v3](http://arxiv.org/pdf/2305.00046v3)**

> **作者:** Samiul Based Shuvo; Tasnia Binte Mamun
>
> **摘要:** Lung cancer is a leading cause of cancer-related deaths worldwide, and early detection is crucial for improving patient outcomes. Nevertheless, early diagnosis of cancer is a major challenge, particularly in low-resource settings where access to medical resources and trained radiologists is limited. The objective of this study is to propose an automated end-to-end deep learning-based framework for the early detection and classification of lung nodules, specifically for low-resource settings. The proposed framework consists of three stages: lung segmentation using a modified 3D U-Net named 3D Res-U-Net, nodule detection using YOLO-v5, and classification with a Vision Transformer-based architecture. We evaluated the proposed framework on a publicly available dataset, LUNA16. The proposed framework's performance was measured using the respective domain's evaluation matrices. The proposed framework achieved a 98.82% lung segmentation dice score while detecting the lung nodule with 0.76 mAP@50 from the segmented lung, at a low false-positive rate. The performance of both networks of the proposed framework was compared with other studies and found to outperform them regarding segmentation and detection accuracy. Additionally, our proposed Vision transformer network obtained an accuracy of 93.57%, which is 1.21% higher than the state-of-the-art networks. Our proposed end-to-end deep learning-based framework can effectively segment lungs, and detect and classify lung nodules, specifically in low-resource settings with limited access to radiologists. The proposed framework outperforms existing studies regarding all the respective evaluation metrics. The proposed framework can potentially improve the accuracy and efficiency of lung cancer screening in low-resource settings, ultimately leading to better patient outcomes.
>
---
#### [replaced 074] Histogram Layers for Neural Engineered Features
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.17176v2](http://arxiv.org/pdf/2403.17176v2)**

> **作者:** Joshua Peeples; Salim Al Kharsa; Luke Saleh; Alina Zare
>
> **备注:** 13 pages, 8 Figures; Accepted to IEEE Transactions on Artificial Intelligence
>
> **摘要:** In the computer vision literature, many effective histogram-based features have been developed. These engineered features include local binary patterns and edge histogram descriptors among others and they have been shown to be informative features for a variety of computer vision tasks. In this paper, we explore whether these features can be learned through histogram layers embedded in a neural network and, therefore, be leveraged within deep learning frameworks. By using histogram features, local statistics of the feature maps from the convolution neural networks can be used to better represent the data. We present neural versions of local binary pattern and edge histogram descriptors that jointly improve the feature representation and perform image classification. Experiments are presented on benchmark and real-world datasets.
>
---
#### [replaced 075] Versatile Multimodal Controls for Expressive Talking Human Animation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.08714v4](http://arxiv.org/pdf/2503.08714v4)**

> **作者:** Zheng Qin; Ruobing Zheng; Yabing Wang; Tianqi Li; Zixin Zhu; Sanping Zhou; Ming Yang; Le Wang
>
> **备注:** Accepted by ACM MM2025
>
> **摘要:** In filmmaking, directors typically allow actors to perform freely based on the script before providing specific guidance on how to present key actions. AI-generated content faces similar requirements, where users not only need automatic generation of lip synchronization and basic gestures from audio input but also desire semantically accurate and expressive body movement that can be ``directly guided'' through text descriptions. Therefore, we present VersaAnimator, a versatile framework that synthesizes expressive talking human videos from arbitrary portrait images. Specifically, we design a motion generator that produces basic rhythmic movements from audio input and supports text-prompt control for specific actions. The generated whole-body 3D motion tokens can animate portraits of various scales, producing talking heads, half-body gestures and even leg movements for whole-body images. Besides, we introduce a multi-modal controlled video diffusion that generates photorealistic videos, where speech signals govern lip synchronization, facial expressions, and head motions while body movements are guided by the 2D poses. Furthermore, we introduce a token2pose translator to smoothly map 3D motion tokens to 2D pose sequences. This design mitigates the stiffness resulting from direct 3D to 2D conversion and enhances the details of the generated body movements. Extensive experiments shows that VersaAnimator synthesizes lip-synced and identity-preserving videos while generating expressive and semantically meaningful whole-body motions.
>
---
#### [replaced 076] Find First, Track Next: Decoupling Identification and Propagation in Referring Video Object Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.03492v2](http://arxiv.org/pdf/2503.03492v2)**

> **作者:** Suhwan Cho; Seunghoon Lee; Minhyeok Lee; Jungho Lee; Sangyoun Lee
>
> **备注:** ICCVW 2025
>
> **摘要:** Referring video object segmentation aims to segment and track a target object in a video using a natural language prompt. Existing methods typically fuse visual and textual features in a highly entangled manner, processing multi-modal information together to generate per-frame masks. However, this approach often struggles with ambiguous target identification, particularly in scenes with multiple similar objects, and fails to ensure consistent mask propagation across frames. To address these limitations, we introduce FindTrack, an efficient decoupled framework that separates target identification from mask propagation. FindTrack first adaptively selects a key frame by balancing segmentation confidence and vision-text alignment, establishing a robust reference for the target object. This reference is then utilized by a dedicated propagation module to track and segment the object across the entire video. By decoupling these processes, FindTrack effectively reduces ambiguities in target association and enhances segmentation consistency. FindTrack significantly outperforms all existing methods on public benchmarks, demonstrating its superiority.
>
---
#### [replaced 077] Text-to-Image Generation Via Energy-Based CLIP
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.17046v2](http://arxiv.org/pdf/2408.17046v2)**

> **作者:** Roy Ganz; Michael Elad
>
> **备注:** Accepted to TMLR
>
> **摘要:** Joint Energy Models (JEMs), while drawing significant research attention, have not been successfully scaled to real-world, high-resolution datasets. We present CLIP-JEM, a novel approach extending JEMs to the multimodal vision-language domain using CLIP, integrating both generative and discriminative objectives. For the generative one, we introduce an image-text joint-energy function based on Cosine similarity in the CLIP space, training CLIP to assign low energy to real image-caption pairs and high energy otherwise. For the discriminative one, we employ contrastive adversarial loss, extending the adversarial training objective to the multimodal domain. CLIP-JEM not only generates realistic images from text but also achieves competitive results on the compositionality benchmark, outperforming leading methods with fewer parameters. Additionally, we demonstrate the superior guidance capability of CLIP-JEM by enhancing CLIP-based generative frameworks and converting unconditional diffusion models to text-based ones. Lastly, we show that our model can serve as a more robust evaluation metric for text-to-image generative tasks than CLIP.
>
---
#### [replaced 078] MagicMotion: Controllable Video Generation with Dense-to-Sparse Trajectory Guidance
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.16421v2](http://arxiv.org/pdf/2503.16421v2)**

> **作者:** Quanhao Li; Zhen Xing; Rui Wang; Hui Zhang; Qi Dai; Zuxuan Wu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent advances in video generation have led to remarkable improvements in visual quality and temporal coherence. Upon this, trajectory-controllable video generation has emerged to enable precise object motion control through explicitly defined spatial paths. However, existing methods struggle with complex object movements and multi-object motion control, resulting in imprecise trajectory adherence, poor object consistency, and compromised visual quality. Furthermore, these methods only support trajectory control in a single format, limiting their applicability in diverse scenarios. Additionally, there is no publicly available dataset or benchmark specifically tailored for trajectory-controllable video generation, hindering robust training and systematic evaluation. To address these challenges, we introduce MagicMotion, a novel image-to-video generation framework that enables trajectory control through three levels of conditions from dense to sparse: masks, bounding boxes, and sparse boxes. Given an input image and trajectories, MagicMotion seamlessly animates objects along defined trajectories while maintaining object consistency and visual quality. Furthermore, we present MagicData, a large-scale trajectory-controlled video dataset, along with an automated pipeline for annotation and filtering. We also introduce MagicBench, a comprehensive benchmark that assesses both video quality and trajectory control accuracy across different numbers of objects. Extensive experiments demonstrate that MagicMotion outperforms previous methods across various metrics. Our project page are publicly available at https://quanhaol.github.io/magicmotion-site.
>
---
#### [replaced 079] Motion Keyframe Interpolation for Any Human Skeleton via Temporally Consistent Point Cloud Sampling and Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.07444v2](http://arxiv.org/pdf/2405.07444v2)**

> **作者:** Clinton Mo; Kun Hu; Chengjiang Long; Dong Yuan; Zhiyong Wang
>
> **备注:** Published in ECCV 2024
>
> **摘要:** In the character animation field, modern supervised keyframe interpolation models have demonstrated exceptional performance in constructing natural human motions from sparse pose definitions. As supervised models, large motion datasets are necessary to facilitate the learning process; however, since motion is represented with fixed hierarchical skeletons, such datasets are incompatible for skeletons outside the datasets' native configurations. Consequently, the expected availability of a motion dataset for desired skeletons severely hinders the feasibility of learned interpolation in practice. To combat this limitation, we propose Point Cloud-based Motion Representation Learning (PC-MRL), an unsupervised approach to enabling cross-compatibility between skeletons for motion interpolation learning. PC-MRL consists of a skeleton obfuscation strategy using temporal point cloud sampling, and an unsupervised skeleton reconstruction method from point clouds. We devise a temporal point-wise K-nearest neighbors loss for unsupervised learning. Moreover, we propose First-frame Offset Quaternion (FOQ) and Rest Pose Augmentation (RPA) strategies to overcome necessary limitations of our unsupervised point cloud-to-skeletal motion process. Comprehensive experiments demonstrate the effectiveness of PC-MRL in motion interpolation for desired skeletons without supervision from native datasets.
>
---
#### [replaced 080] Competency-Aware Planning for Probabilistically Safe Navigation Under Perception Uncertainty
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2409.06111v5](http://arxiv.org/pdf/2409.06111v5)**

> **作者:** Sara Pohland; Claire Tomlin
>
> **摘要:** Perception-based navigation systems are useful for unmanned ground vehicle (UGV) navigation in complex terrains, where traditional depth-based navigation schemes are insufficient. However, these data-driven methods are highly dependent on their training data and can fail in surprising and dramatic ways with little warning. To ensure the safety of the vehicle and the surrounding environment, it is imperative that the navigation system is able to recognize the predictive uncertainty of the perception model and respond safely and effectively in the face of uncertainty. In an effort to enable safe navigation under perception uncertainty, we develop a probabilistic and reconstruction-based competency estimation (PaRCE) method to estimate the model's level of familiarity with an input image as a whole and with specific regions in the image. We find that the overall competency score can correctly predict correctly classified, misclassified, and out-of-distribution (OOD) samples. We also confirm that the regional competency maps can accurately distinguish between familiar and unfamiliar regions across images. We then use this competency information to develop a planning and control scheme that enables effective navigation while maintaining a low probability of error. We find that the competency-aware scheme greatly reduces the number of collisions with unfamiliar obstacles, compared to a baseline controller with no competency awareness. Furthermore, the regional competency information is very valuable in enabling efficient navigation.
>
---
#### [replaced 081] TextSAM-EUS: Text Prompt Learning for SAM to Accurately Segment Pancreatic Tumor in Endoscopic Ultrasound
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.18082v2](http://arxiv.org/pdf/2507.18082v2)**

> **作者:** Pascal Spiegler; Taha Koleilat; Arash Harirpoush; Corey S. Miller; Hassan Rivaz; Marta Kersten-Oertel; Yiming Xiao
>
> **备注:** Accepted to ICCV 2025 Workshop CVAMD
>
> **摘要:** Pancreatic cancer carries a poor prognosis and relies on endoscopic ultrasound (EUS) for targeted biopsy and radiotherapy. However, the speckle noise, low contrast, and unintuitive appearance of EUS make segmentation of pancreatic tumors with fully supervised deep learning (DL) models both error-prone and dependent on large, expert-curated annotation datasets. To address these challenges, we present TextSAM-EUS, a novel, lightweight, text-driven adaptation of the Segment Anything Model (SAM) that requires no manual geometric prompts at inference. Our approach leverages text prompt learning (context optimization) through the BiomedCLIP text encoder in conjunction with a LoRA-based adaptation of SAM's architecture to enable automatic pancreatic tumor segmentation in EUS, tuning only 0.86% of the total parameters. On the public Endoscopic Ultrasound Database of the Pancreas, TextSAM-EUS with automatic prompts attains 82.69% Dice and 85.28% normalized surface distance (NSD), and with manual geometric prompts reaches 83.10% Dice and 85.70% NSD, outperforming both existing state-of-the-art (SOTA) supervised DL models and foundation models (e.g., SAM and its variants). As the first attempt to incorporate prompt learning in SAM-based medical image segmentation, TextSAM-EUS offers a practical option for efficient and robust automatic EUS segmentation.
>
---
#### [replaced 082] Part Segmentation of Human Meshes via Multi-View Human Parsing
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.18655v2](http://arxiv.org/pdf/2507.18655v2)**

> **作者:** James Dickens; Kamyar Hamad
>
> **摘要:** Recent advances in point cloud deep learning have led to models that achieve high per-part labeling accuracy on large-scale point clouds, using only the raw geometry of unordered point sets. In parallel, the field of human parsing focuses on predicting body part and clothing/accessory labels from images. This work aims to bridge these two domains by enabling per-vertex semantic segmentation of large-scale human meshes. To achieve this, a pseudo-ground truth labeling pipeline is developed for the Thuman2.1 dataset: meshes are first aligned to a canonical pose, segmented from multiple viewpoints, and the resulting point-level labels are then backprojected onto the original mesh to produce per-point pseudo ground truth annotations. Subsequently, a novel, memory-efficient sampling strategy is introduced, a windowed iterative farthest point sampling (FPS) with space-filling curve-based serialization to effectively downsample the point clouds. This is followed by a purely geometric segmentation using PointTransformer, enabling semantic parsing of human meshes without relying on texture information. Experimental results confirm the effectiveness and accuracy of the proposed approach.
>
---
#### [replaced 083] One Last Attention for Your Vision-Language Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15480v2](http://arxiv.org/pdf/2507.15480v2)**

> **作者:** Liang Chen; Ghazi Shazan Ahmad; Tianjun Yao; Lingqiao Liu; Zhiqiang Shen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Pretrained vision-language models (VLMs), such as CLIP, achieve remarkable zero-shot performance, yet their downstream potential hinges on effective fine-tuning. Most adaptation methods typically focus on refining representation from separate modalities (text or vision) but neglect the critical role of their fused representations in the decision-making process, \emph{\ie} rational matrix that drives the final prediction. To bridge the gap, we propose a simple yet effective \textbf{R}ational \textbf{Ada}ptaion ({RAda}) to explicitly exploit the final fused representation during fine-tuning. RAda employs a learned mask, obtained from a lightweight attention layer attached at the end of a VLM, to dynamically calibrate the contribution of each element in the rational matrix, enabling targeted adjustments to the final cross-modal interactions without incurring costly modifications to intermediate features. Experiments in different settings (i.e., updating, or freezing pretrained encoders in adaptation, and test-time training that can only access the unlabeled test data) show that RAda serves as a versatile fine-tuning technique, improving the baseline with minimal code and performing comparably against current arts in most settings. Code is available at \href{https://github.com/khufia/RAda/tree/main}{github.com/khufia/RAda}.
>
---
#### [replaced 084] Knowledge Distillation with Refined Logits
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.07703v3](http://arxiv.org/pdf/2408.07703v3)**

> **作者:** Wujie Sun; Defang Chen; Siwei Lyu; Genlang Chen; Chun Chen; Can Wang
>
> **备注:** ICCV 2025
>
> **摘要:** Recent research on knowledge distillation has increasingly focused on logit distillation because of its simplicity, effectiveness, and versatility in model compression. In this paper, we introduce Refined Logit Distillation (RLD) to address the limitations of current logit distillation methods. Our approach is motivated by the observation that even high-performing teacher models can make incorrect predictions, creating an exacerbated divergence between the standard distillation loss and the cross-entropy loss, which can undermine the consistency of the student model's learning objectives. Previous attempts to use labels to empirically correct teacher predictions may undermine the class correlations. In contrast, our RLD employs labeling information to dynamically refine teacher logits. In this way, our method can effectively eliminate misleading information from the teacher while preserving crucial class correlations, thus enhancing the value and efficiency of distilled knowledge. Experimental results on CIFAR-100 and ImageNet demonstrate its superiority over existing methods. Our code is available at https://github.com/zju-SWJ/RLD.
>
---
#### [replaced 085] VLM-CPL: Consensus Pseudo Labels from Vision-Language Models for Human Annotation-Free Pathological Image Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.15836v2](http://arxiv.org/pdf/2403.15836v2)**

> **作者:** Lanfeng Zhong; Zongyao Huang; Yang Liu; Wenjun Liao; Shichuan Zhang; Guotai Wang; Shaoting Zhang
>
> **备注:** Accepted at TMI
>
> **摘要:** Classification of pathological images is the basis for automatic cancer diagnosis. Despite that deep learning methods have achieved remarkable performance, they heavily rely on labeled data, demanding extensive human annotation efforts. In this study, we present a novel human annotation-free method by leveraging pre-trained Vision-Language Models (VLMs). Without human annotation, pseudo-labels of the training set are obtained by utilizing the zero-shot inference capabilities of VLM, which may contain a lot of noise due to the domain gap between the pre-training and target datasets. To address this issue, we introduce VLM-CPL, a novel approach that contains two noisy label filtering techniques with a semi-supervised learning strategy. Specifically, we first obtain prompt-based pseudo-labels with uncertainty estimation by zero-shot inference with the VLM using multiple augmented views of an input. Then, by leveraging the feature representation ability of VLM, we obtain feature-based pseudo-labels via sample clustering in the feature space. Prompt-feature consensus is introduced to select reliable samples based on the consensus between the two types of pseudo-labels. We further propose High-confidence Cross Supervision by to learn from samples with reliable pseudo-labels and the remaining unlabeled samples. Additionally, we present an innovative open-set prompting strategy that filters irrelevant patches from whole slides to enhance the quality of selected patches. Experimental results on five public pathological image datasets for patch-level and slide-level classification showed that our method substantially outperformed zero-shot classification by VLMs, and was superior to existing noisy label learning methods. The code is publicly available at https://github.com/HiLab-git/VLM-CPL.
>
---
#### [replaced 086] ADAgent: LLM Agent for Alzheimer's Disease Analysis with Collaborative Coordinator
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11150v3](http://arxiv.org/pdf/2506.11150v3)**

> **作者:** Wenlong Hou; Guangqian Yang; Ye Du; Yeung Lau; Lihao Liu; Junjun He; Ling Long; Shujun Wang
>
> **摘要:** Alzheimer's disease (AD) is a progressive and irreversible neurodegenerative disease. Early and precise diagnosis of AD is crucial for timely intervention and treatment planning to alleviate the progressive neurodegeneration. However, most existing methods rely on single-modality data, which contrasts with the multifaceted approach used by medical experts. While some deep learning approaches process multi-modal data, they are limited to specific tasks with a small set of input modalities and cannot handle arbitrary combinations. This highlights the need for a system that can address diverse AD-related tasks, process multi-modal or missing input, and integrate multiple advanced methods for improved performance. In this paper, we propose ADAgent, the first specialized AI agent for AD analysis, built on a large language model (LLM) to address user queries and support decision-making. ADAgent integrates a reasoning engine, specialized medical tools, and a collaborative outcome coordinator to facilitate multi-modal diagnosis and prognosis tasks in AD. Extensive experiments demonstrate that ADAgent outperforms SOTA methods, achieving significant improvements in accuracy, including a 2.7% increase in multi-modal diagnosis, a 0.7% improvement in multi-modal prognosis, and enhancements in MRI and PET diagnosis tasks.
>
---
#### [replaced 087] Model Reveals What to Cache: Profiling-Based Feature Reuse for Video Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.03140v2](http://arxiv.org/pdf/2504.03140v2)**

> **作者:** Xuran Ma; Yexin Liu; Yaofu Liu; Xianfeng Wu; Mingzhe Zheng; Zihao Wang; Ser-Nam Lim; Harry Yang
>
> **摘要:** Recent advances in diffusion models have demonstrated remarkable capabilities in video generation. However, the computational intensity remains a significant challenge for practical applications. While feature caching has been proposed to reduce the computational burden of diffusion models, existing methods typically overlook the heterogeneous significance of individual blocks, resulting in suboptimal reuse and degraded output quality. To this end, we address this gap by introducing ProfilingDiT, a novel adaptive caching strategy that explicitly disentangles foreground and background-focused blocks. Through a systematic analysis of attention distributions in diffusion models, we reveal a key observation: 1) Most layers exhibit a consistent preference for either foreground or background regions. 2) Predicted noise shows low inter-step similarity initially, which stabilizes as denoising progresses. This finding inspires us to formulate a selective caching strategy that preserves full computation for dynamic foreground elements while efficiently caching static background features. Our approach substantially reduces computational overhead while preserving visual fidelity. Extensive experiments demonstrate that our framework achieves significant acceleration (e.g., 2.01 times speedup for Wan2.1) while maintaining visual fidelity across comprehensive quality metrics, establishing a viable method for efficient video generation.
>
---
#### [replaced 088] Distribution-aware Forgetting Compensation for Exemplar-Free Lifelong Person Re-identification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.15041v3](http://arxiv.org/pdf/2504.15041v3)**

> **作者:** Shiben Liu; Huijie Fan; Qiang Wang; Baojie Fan; Yandong Tang; Liangqiong Qu
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Lifelong Person Re-identification (LReID) suffers from a key challenge in preserving old knowledge while adapting to new information. The existing solutions include rehearsal-based and rehearsal-free methods to address this challenge. Rehearsal-based approaches rely on knowledge distillation, continuously accumulating forgetting during the distillation process. Rehearsal-free methods insufficiently learn the distribution of each domain, leading to forgetfulness over time. To solve these issues, we propose a novel Distribution-aware Forgetting Compensation (DAFC) model that explores cross-domain shared representation learning and domain-specific distribution integration without using old exemplars or knowledge distillation. We propose a Text-driven Prompt Aggregation (TPA) that utilizes text features to enrich prompt elements and guide the prompt model to learn fine-grained representations for each instance. This can enhance the differentiation of identity information and establish the foundation for domain distribution awareness. Then, Distribution-based Awareness and Integration (DAI) is designed to capture each domain-specific distribution by a dedicated expert network and adaptively consolidate them into a shared region in high-dimensional space. In this manner, DAI can consolidate and enhance cross-domain shared representation learning while alleviating catastrophic forgetting. Furthermore, we develop a Knowledge Consolidation Mechanism (KCM) that comprises instance-level discrimination and cross-domain consistency alignment strategies to facilitate model adaptive learning of new knowledge from the current domain and promote knowledge consolidation learning between acquired domain-specific distributions, respectively. Experimental results show that our DAFC outperforms state-of-the-art methods. Our code is available at https://github.com/LiuShiBen/DAFC.
>
---
#### [replaced 089] Beyond Walking: A Large-Scale Image-Text Benchmark for Text-based Person Anomaly Search
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.17776v3](http://arxiv.org/pdf/2411.17776v3)**

> **作者:** Shuyu Yang; Yaxiong Wang; Li Zhu; Zhedong Zheng
>
> **摘要:** Text-based person search aims to retrieve specific individuals across camera networks using natural language descriptions. However, current benchmarks often exhibit biases towards common actions like walking or standing, neglecting the critical need for identifying abnormal behaviors in real-world scenarios. To meet such demands, we propose a new task, text-based person anomaly search, locating pedestrians engaged in both routine or anomalous activities via text. To enable the training and evaluation of this new task, we construct a large-scale image-text Pedestrian Anomaly Behavior (PAB) benchmark, featuring a broad spectrum of actions, e.g., running, performing, playing soccer, and the corresponding anomalies, e.g., lying, being hit, and falling of the same identity. The training set of PAB comprises 1,013,605 synthesized image-text pairs of both normalities and anomalies, while the test set includes 1,978 real-world image-text pairs. To validate the potential of PAB, we introduce a cross-modal pose-aware framework, which integrates human pose patterns with identity-based hard negative pair sampling. Extensive experiments on the proposed benchmark show that synthetic training data facilitates the fine-grained behavior retrieval, and the proposed pose-aware method arrives at 84.93% recall@1 accuracy, surpassing other competitive methods. The dataset, model, and code are available at https://github.com/Shuyu-XJTU/CMP.
>
---
#### [replaced 090] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.11586v3](http://arxiv.org/pdf/2412.11586v3)**

> **作者:** Xiaokun Sun; Zeyu Cai; Ying Tai; Jian Yang; Zhenyu Zhang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** While haircut indicates distinct personality, existing avatar generation methods fail to model practical hair due to the data limitation or entangled representation. We propose StrandHead, a novel text-driven method capable of generating 3D hair strands and disentangled head avatars with strand-level attributes. Instead of using large-scale hair-text paired data for supervision, we demonstrate that realistic hair strands can be generated from prompts by distilling 2D generative models pre-trained on human mesh data. To this end, we propose a meshing approach guided by strand geometry to guarantee the gradient flow from the distillation objective to the neural strand representation. The optimization is then regularized by statistically significant haircut features, leading to stable updating of strands against unreasonable drifting. These employed 2D/3D human-centric priors contribute to text-aligned and realistic 3D strand generation. Extensive experiments show that StrandHead achieves the state-of-the-art performance on text to strand generation and disentangled 3D head avatar modeling. The generated 3D hair can be applied on avatars for strand-level editing, as well as implemented in the graphics engine for physical simulation or other applications. Project page: https://xiaokunsun.github.io/StrandHead.github.io/.
>
---
#### [replaced 091] SEAL: Searching Expandable Architectures for Incremental Learning
- **分类: cs.LG; cs.AI; cs.CV; 68T07**

- **链接: [http://arxiv.org/pdf/2505.10457v2](http://arxiv.org/pdf/2505.10457v2)**

> **作者:** Matteo Gambella; Manuel Roveri
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Incremental learning is a machine learning paradigm where a model learns from a sequential stream of tasks. This setting poses a key challenge: balancing plasticity (learning new tasks) and stability (preserving past knowledge). Neural Architecture Search (NAS), a branch of AutoML, automates the design of the architecture of Deep Neural Networks and has shown success in static settings. However, existing NAS-based approaches to incremental learning often rely on expanding the model at every task, making them impractical in resource-constrained environments. In this work, we introduce SEAL, a NAS-based framework tailored for data-incremental learning, a scenario where disjoint data samples arrive sequentially and are not stored for future access. SEAL adapts the model structure dynamically by expanding it only when necessary, based on a capacity estimation metric. Stability is preserved through cross-distillation training after each expansion step. The NAS component jointly searches for both the architecture and the optimal expansion policy. Experiments across multiple benchmarks demonstrate that SEAL effectively reduces forgetting and enhances accuracy while maintaining a lower model size compared to prior methods. These results highlight the promise of combining NAS and selective expansion for efficient, adaptive learning in incremental scenarios.
>
---
#### [replaced 092] Latent Multimodal Reconstruction for Misinformation Detection
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.06010v2](http://arxiv.org/pdf/2504.06010v2)**

> **作者:** Stefanos-Iordanis Papadopoulos; Christos Koutlis; Symeon Papadopoulos; Panagiotis C. Petrantonakis
>
> **摘要:** Multimodal misinformation, such as miscaptioned images, where captions misrepresent an image's origin, context, or meaning, poses a growing challenge in the digital age. To support fact-checkers, researchers have focused on developing datasets and methods for multimodal misinformation detection (MMD). Due to the scarcity of large-scale annotated MMD datasets, recent approaches rely on synthetic training data created via out-of-context pairings or named entity manipulations (e.g., altering names, dates, or locations). However, these often yield simplistic examples that lack real-world complexity, limiting model robustness. Meanwhile, Large Vision-Language Models (LVLMs) remain underexplored for generating diverse and realistic synthetic data for MMD. To address, we introduce "Miscaption This!", a collection of LVLM-generated miscaptioned image datasets. Additionally, we introduce "Latent Multimodal Reconstruction" (LAMAR), a network trained to reconstruct the embeddings of truthful captions, providing a strong auxiliary signal to guide detection. We explore various training strategies (end-to-end vs. large-scale pre-training) and integration mechanisms (direct, mask, gate, and attention). Extensive experiments show that models trained on "MisCaption This!" generalize better to real-world misinformation while LAMAR achieves new state-of-the-art on both NewsCLIPpings and VERITE benchmarks; highlighting the value of LVLM-generated data and reconstruction-based networks for advancing MMD. Our code is available at https://github.com/stevejpapad/miscaptioned-image-reconstruction
>
---
#### [replaced 093] YOLO-PRO: Enhancing Instance-Specific Object Detection with Full-Channel Global Self-Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.02348v2](http://arxiv.org/pdf/2503.02348v2)**

> **作者:** Lin Huang; Yujuan Tan; Weisheng Li; Shitai Shan; Linlin Shen; Jing Yu
>
> **摘要:** This paper addresses the inherent limitations of conventional bottleneck structures (diminished instance discriminability due to overemphasis on batch statistics) and decoupled heads (computational redundancy) in object detection frameworks by proposing two novel modules: the Instance-Specific Bottleneck with full-channel global self-attention (ISB) and the Instance-Specific Asymmetric Decoupled Head (ISADH). The ISB module innovatively reconstructs feature maps to establish an efficient full-channel global attention mechanism through synergistic fusion of batch-statistical and instance-specific features. Complementing this, the ISADH module pioneers an asymmetric decoupled architecture enabling hierarchical multi-dimensional feature integration via dual-stream batch-instance representation fusion. Extensive experiments on the MS-COCO benchmark demonstrate that the coordinated deployment of ISB and ISADH in the YOLO-PRO framework achieves state-of-the-art performance across all computational scales. Specifically, YOLO-PRO surpasses YOLOv8 by 1.0-1.6% AP (N/S/M/L/X scales) and outperforms YOLO11 by 0.1-0.5% AP in critical N/M/L/X groups, while maintaining competitive computational efficiency. This work provides practical insights for developing high-precision detectors deployable on edge devices.
>
---
#### [replaced 094] FlowAlign: Trajectory-Regularized, Inversion-Free Flow-based Image Editing
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23145v4](http://arxiv.org/pdf/2505.23145v4)**

> **作者:** Jeongsol Kim; Yeobin Hong; Jonghyun Park; Jong Chul Ye
>
> **摘要:** Recent inversion-free, flow-based image editing methods such as FlowEdit leverages a pre-trained noise-to-image flow model such as Stable Diffusion 3, enabling text-driven manipulation by solving an ordinary differential equation (ODE). While the lack of exact latent inversion is a core advantage of these methods, it often results in unstable editing trajectories and poor source consistency. To address this limitation, we propose {\em FlowAlign}, a novel inversion-free flow-based framework for consistent image editing with optimal control-based trajectory control. Specifically, FlowAlign introduces source similarity at the terminal point as a regularization term to promote smoother and more consistent trajectories during the editing process. Notably, our terminal point regularization is shown to explicitly balance semantic alignment with the edit prompt and structural consistency with the source image along the trajectory. Furthermore, FlowAlign naturally supports reverse editing by simply reversing the ODE trajectory, highliting the reversible and consistent nature of the transformation. Extensive experiments demonstrate that FlowAlign outperforms existing methods in both source preservation and editing controllability.
>
---
#### [replaced 095] Video2BEV: Transforming Drone Videos to BEVs for Video-based Geo-localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.13610v4](http://arxiv.org/pdf/2411.13610v4)**

> **作者:** Hao Ju; Shaofei Huang; Si Liu; Zhedong Zheng
>
> **摘要:** Existing approaches to drone visual geo-localization predominantly adopt the image-based setting, where a single drone-view snapshot is matched with images from other platforms. Such task formulation, however, underutilizes the inherent video output of the drone and is sensitive to occlusions and viewpoint disparity. To address these limitations, we formulate a new video-based drone geo-localization task and propose the Video2BEV paradigm. This paradigm transforms the video into a Bird's Eye View (BEV), simplifying the subsequent \textbf{inter-platform} matching process. In particular, we employ Gaussian Splatting to reconstruct a 3D scene and obtain the BEV projection. Different from the existing transform methods, \eg, polar transform, our BEVs preserve more fine-grained details without significant distortion. To facilitate the discriminative \textbf{intra-platform} representation learning, our Video2BEV paradigm also incorporates a diffusion-based module for generating hard negative samples. To validate our approach, we introduce UniV, a new video-based geo-localization dataset that extends the image-based University-1652 dataset. UniV features flight paths at $30^\circ$ and $45^\circ$ elevation angles with increased frame rates of up to 10 frames per second (FPS). Extensive experiments on the UniV dataset show that our Video2BEV paradigm achieves competitive recall rates and outperforms conventional video-based methods. Compared to other competitive methods, our proposed approach exhibits robustness at lower elevations with more occlusions.
>
---
#### [replaced 096] LLM2TEA: An Agentic AI Designer for Discovery with Generative Evolutionary Multitasking
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2406.14917v3](http://arxiv.org/pdf/2406.14917v3)**

> **作者:** Melvin Wong; Jiao Liu; Thiago Rios; Stefan Menzel; Yew Soon Ong
>
> **备注:** This work is accepted by IEEE CIM. IEEE copyrights applies
>
> **摘要:** This paper presents LLM2TEA, a Large Language Model (LLM) driven MultiTask Evolutionary Algorithm, representing the first agentic AI designer of its kind operating with generative evolutionary multitasking (GEM). LLM2TEA enables the crossbreeding of solutions from multiple domains, fostering novel solutions that transcend disciplinary boundaries. Of particular interest is the ability to discover designs that are both novel and conforming to real-world physical specifications. LLM2TEA comprises an LLM to generate genotype samples from text prompts describing target objects, a text-to-3D generative model to produce corresponding phenotypes, a classifier to interpret its semantic representations, and a computational simulator to assess its physical properties. Novel LLM-based multitask evolutionary operators are introduced to guide the search towards high-performing, practically viable designs. Experimental results in conceptual design optimization validate the effectiveness of LLM2TEA, showing 97% to 174% improvements in the diversity of novel designs over the current text-to-3D baseline. Moreover, over 73% of the generated designs outperform the top 1% of designs produced by the text-to-3D baseline in terms of physical performance. The designs produced by LLM2TEA are not only aesthetically creative but also functional in real-world contexts. Several of these designs have been successfully 3D printed, demonstrating the ability of our approach to transform AI-generated outputs into tangible, physical designs. These designs underscore the potential of LLM2TEA as a powerful tool for complex design optimization and discovery, capable of producing novel and physically viable designs.
>
---
#### [replaced 097] Crop Pest Classification Using Deep Learning Techniques: A Review
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.01494v2](http://arxiv.org/pdf/2507.01494v2)**

> **作者:** Muhammad Hassam Ejaz; Muhammad Bilal; Usman Habib
>
> **摘要:** Insect pests continue to bring a serious threat to crop yields around the world, and traditional methods for monitoring them are often slow, manual, and difficult to scale. In recent years, deep learning has emerged as a powerful solution, with techniques like convolutional neural networks (CNNs), vision transformers (ViTs), and hybrid models gaining popularity for automating pest detection. This review looks at 37 carefully selected studies published between 2018 and 2025, all focused on AI-based pest classification. The selected research is organized by crop type, pest species, model architecture, dataset usage, and key technical challenges. The early studies relied heavily on CNNs but latest work is shifting toward hybrid and transformer-based models that deliver higher accuracy and better contextual understanding. Still, challenges like imbalanced datasets, difficulty in detecting small pests, limited generalizability, and deployment on edge devices remain significant hurdles. Overall, this review offers a structured overview of the field, highlights useful datasets, and outlines the key challenges and future directions for AI-based pest monitoring systems.
>
---
#### [replaced 098] Implementing Adaptations for Vision AutoRegressive Model
- **分类: cs.CV; cs.LG; I.2.6; I.5.1; I.4.8; I.2.10**

- **链接: [http://arxiv.org/pdf/2507.11441v2](http://arxiv.org/pdf/2507.11441v2)**

> **作者:** Kaif Shaikh; Franziska Boenisch; Adam Dziedzic
>
> **备注:** Accepted at DIG-BUGS: Data in Generative Models Workshop @ ICML 2025
>
> **摘要:** Vision AutoRegressive model (VAR) was recently introduced as an alternative to Diffusion Models (DMs) in image generation domain. In this work we focus on its adaptations, which aim to fine-tune pre-trained models to perform specific downstream tasks, like medical data generation. While for DMs there exist many techniques, adaptations for VAR remain underexplored. Similarly, differentially private (DP) adaptations-ones that aim to preserve privacy of the adaptation data-have been extensively studied for DMs, while VAR lacks such solutions. In our work, we implement and benchmark many strategies for VAR, and compare them to state-of-the-art DM adaptation strategies. We observe that VAR outperforms DMs for non-DP adaptations, however, the performance of DP suffers, which necessitates further research in private adaptations for VAR. Code is available at https://github.com/sprintml/finetuning_var_dp.
>
---
#### [replaced 099] Critiques of World Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05169v3](http://arxiv.org/pdf/2507.05169v3)**

> **作者:** Eric Xing; Mingkai Deng; Jinyu Hou; Zhiting Hu
>
> **摘要:** World Model, the supposed algorithmic surrogate of the real-world environment which biological agents experience with and act upon, has been an emerging topic in recent years because of the rising needs to develop virtual agents with artificial (general) intelligence. There has been much debate on what a world model really is, how to build it, how to use it, and how to evaluate it. In this essay, starting from the imagination in the famed Sci-Fi classic Dune, and drawing inspiration from the concept of "hypothetical thinking" in psychology literature, we offer critiques of several schools of thoughts on world modeling, and argue the primary goal of a world model to be simulating all actionable possibilities of the real world for purposeful reasoning and acting. Building on the critiques, we propose a new architecture for a general-purpose world model, based on hierarchical, multi-level, and mixed continuous/discrete representations, and a generative and self-supervision learning framework, with an outlook of a Physical, Agentic, and Nested (PAN) AGI system enabled by such a model.
>
---
#### [replaced 100] BadPatch: Diffusion-Based Generation of Physical Adversarial Patches
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01440v4](http://arxiv.org/pdf/2412.01440v4)**

> **作者:** Zhixiang Wang; Xingjun Ma; Yu-Gang Jiang
>
> **摘要:** Physical adversarial patches printed on clothing can enable individuals to evade person detectors, but most existing methods prioritize attack effectiveness over stealthiness, resulting in aesthetically unpleasing patches. While generative adversarial networks and diffusion models can produce more natural-looking patches, they often fail to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these limitations, we propose BadPatch, a novel diffusion-based framework for generating customizable and naturalistic adversarial patches. Our approach allows users to start from a reference image (rather than random noise) and incorporates masks to create patches of various shapes, not limited to squares. To preserve the original semantics during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Our method achieves attack performance comparable to state-of-the-art non-naturalistic patches while maintaining a natural appearance. Using BadPatch, we construct AdvT-shirt-1K, the first physical adversarial T-shirt dataset comprising over a thousand images captured in diverse scenarios. AdvT-shirt-1K can serve as a useful dataset for training or testing future defense methods.
>
---
#### [replaced 101] Generative AI for Cel-Animation: A Survey
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.06250v2](http://arxiv.org/pdf/2501.06250v2)**

> **作者:** Yunlong Tang; Junjia Guo; Pinxin Liu; Zhiyuan Wang; Hang Hua; Jia-Xing Zhong; Yunzhong Xiao; Chao Huang; Luchuan Song; Susan Liang; Yizhi Song; Liu He; Jing Bi; Mingqian Feng; Xinyang Li; Zeliang Zhang; Chenliang Xu
>
> **备注:** Accepted by ICCV 2025 AISTORY Workshop
>
> **摘要:** Traditional Celluloid (Cel) Animation production pipeline encompasses multiple essential steps, including storyboarding, layout design, keyframe animation, inbetweening, and colorization, which demand substantial manual effort, technical expertise, and significant time investment. These challenges have historically impeded the efficiency and scalability of Cel-Animation production. The rise of generative artificial intelligence (GenAI), encompassing large language models, multimodal models, and diffusion models, offers innovative solutions by automating tasks such as inbetween frame generation, colorization, and storyboard creation. This survey explores how GenAI integration is revolutionizing traditional animation workflows by lowering technical barriers, broadening accessibility for a wider range of creators through tools like AniDoc, ToonCrafter, and AniSora, and enabling artists to focus more on creative expression and artistic innovation. Despite its potential, challenges like visual consistency, stylistic coherence, and ethical considerations persist. Additionally, this paper explores future directions and advancements in AI-assisted animation.
>
---
#### [replaced 102] Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16392v2](http://arxiv.org/pdf/2411.16392v2)**

> **作者:** Ziyu Zhang; Binbin Huang; Hanqing Jiang; Liyang Zhou; Xiaojun Xiang; Shunhan Shen
>
> **备注:** 16pages,18figures
>
> **摘要:** We propose Quadratic Gaussian Splatting (QGS), a novel representation that replaces static primitives with deformable quadric surfaces (e.g., ellipse, paraboloids) to capture intricate geometry. Unlike prior works that rely on Euclidean distance for primitive density modeling--a metric misaligned with surface geometry under deformation--QGS introduces geodesic distance-based density distributions. This innovation ensures that density weights adapt intrinsically to the primitive curvature, preserving consistency during shape changes (e.g., from planar disks to curved paraboloids). By solving geodesic distances in closed form on quadric surfaces, QGS enables surface-aware splatting, where a single primitive can represent complex curvature that previously required dozens of planar surfels, potentially reducing memory usage while maintaining efficient rendering via fast ray-quadric intersection. Experiments on DTU, Tanks and Temples, and MipNeRF360 datasets demonstrate state-of-the-art surface reconstruction, with QGS reducing geometric error (chamfer distance) by 33% over 2DGS and 27% over GOF on the DTU dataset. Crucially, QGS retains competitive appearance quality, bridging the gap between geometric precision and visual fidelity for applications like robotics and immersive reality.
>
---
#### [replaced 103] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14346v2](http://arxiv.org/pdf/2505.14346v2)**

> **作者:** Mingfang Zhang; Ryo Yonetani; Yifei Huang; Liangyang Ouyang; Ruicong Liu; Yoichi Sato
>
> **备注:** ICCV 2025
>
> **摘要:** This paper presents a novel inertial localization framework named Egocentric Action-aware Inertial Localization (EAIL), which leverages egocentric action cues from head-mounted IMU signals to localize the target individual within a 3D point cloud. Human inertial localization is challenging due to IMU sensor noise that causes trajectory drift over time. The diversity of human actions further complicates IMU signal processing by introducing various motion patterns. Nevertheless, we observe that some actions captured by the head-mounted IMU correlate with spatial environmental structures (e.g., bending down to look inside an oven, washing dishes next to a sink), thereby serving as spatial anchors to compensate for the localization drift. The proposed EAIL framework learns such correlations via hierarchical multi-modal alignment with vision-language guidance. By assuming that the 3D point cloud of the environment is available, it contrastively learns modality encoders that align short-term egocentric action cues in IMU signals with local environmental features in the point cloud. The learning process is enhanced using concurrently collected vision and language signals to improve multimodal alignment. The learned encoders are then used in reasoning the IMU data and the point cloud over time and space to perform inertial localization. Interestingly, these encoders can further be utilized to recognize the corresponding sequence of actions as a by-product. Extensive experiments demonstrate the effectiveness of the proposed framework over state-of-the-art inertial localization and inertial action recognition baselines.
>
---
#### [replaced 104] Architecture-Aware Minimization (A$^2$M): How to Find Flat Minima in Neural Architecture Search
- **分类: cs.LG; cond-mat.dis-nn; cs.CV; 68T07**

- **链接: [http://arxiv.org/pdf/2503.10404v2](http://arxiv.org/pdf/2503.10404v2)**

> **作者:** Matteo Gambella; Fabrizio Pittorino; Manuel Roveri
>
> **备注:** Published in the journal Machine Learning: Science and Technology - IOPscience
>
> **摘要:** Neural Architecture Search (NAS) has become an essential tool for designing effective and efficient neural networks. In this paper, we investigate the geometric properties of neural architecture spaces commonly used in differentiable NAS methods, specifically NAS-Bench-201 and DARTS. By defining flatness metrics such as neighborhoods and loss barriers along paths in architecture space, we reveal locality and flatness characteristics analogous to the well-known properties of neural network loss landscapes in weight space. In particular, we find that highly accurate architectures cluster together in flat regions, while suboptimal architectures remain isolated, unveiling the detailed geometrical structure of the architecture search landscape. Building on these insights, we propose Architecture-Aware Minimization (A$^2$M), a novel analytically derived algorithmic framework that explicitly biases, for the first time, the gradient of differentiable NAS methods towards flat minima in architecture space. A$^2$M consistently improves generalization over state-of-the-art DARTS-based algorithms on benchmark datasets including CIFAR-10, CIFAR-100, and ImageNet16-120, across both NAS-Bench-201 and DARTS search spaces. Notably, A$^2$M is able to increase the test accuracy, on average across different differentiable NAS methods, by +3.60\% on CIFAR-10, +4.60\% on CIFAR-100, and +3.64\% on ImageNet16-120, demonstrating its superior effectiveness in practice. A$^2$M can be easily integrated into existing differentiable NAS frameworks, offering a versatile tool for future research and applications in automated machine learning. We open-source our code at https://github.com/AI-Tech-Research-Lab/AsquaredM.
>
---
#### [replaced 105] Benchmarking Graph Neural Networks for Document Layout Analysis in Public Affairs
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14699v2](http://arxiv.org/pdf/2505.14699v2)**

> **作者:** Miguel Lopez-Duran; Julian Fierrez; Aythami Morales; Ruben Tolosana; Oscar Delgado-Mohatar; Alvaro Ortigosa
>
> **备注:** 15 pages, 2 figures, accepted paper at The Fifth ICDAR International Workshop on Machine Learning
>
> **摘要:** The automatic analysis of document layouts in digital-born PDF documents remains a challenging problem due to the heterogeneous arrangement of textual and nontextual elements and the imprecision of the textual metadata in the Portable Document Format. In this work, we benchmark Graph Neural Network (GNN) architectures for the task of fine-grained layout classification of text blocks from digital native documents. We introduce two graph construction structures: a k-closest-neighbor graph and a fully connected graph, and generate node features via pre-trained text and vision models, thus avoiding manual feature engineering. Three experimental frameworks are evaluated: single-modality (text or visual), concatenated multimodal, and dual-branch multimodal. We evaluated four foundational GNN models and compared them with the baseline. Our experiments are specifically conducted on a rich dataset of public affairs documents that includes more than 20 sources (e.g., regional and national-level official gazettes), 37K PDF documents, with 441K pages in total. Our results demonstrate that GraphSAGE operating on the k-closest-neighbor graph in a dual-branch configuration achieves the highest per-class and overall accuracy, outperforming the baseline in some sources. These findings confirm the importance of local layout relationships and multimodal fusion exploited through GNNs for the analysis of native digital document layouts.
>
---
#### [replaced 106] ZeroReg3D: A Zero-shot Registration Pipeline for 3D Consecutive Histopathology Image Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21923v2](http://arxiv.org/pdf/2506.21923v2)**

> **作者:** Juming Xiong; Ruining Deng; Jialin Yue; Siqi Lu; Junlin Guo; Marilyn Lionts; Tianyuan Yao; Can Cui; Junchao Zhu; Chongyu Qu; Mengmeng Yin; Haichun Yang; Yuankai Huo
>
> **摘要:** Histological analysis plays a crucial role in understanding tissue structure and pathology. While recent advancements in registration methods have improved 2D histological analysis, they often struggle to preserve critical 3D spatial relationships, limiting their utility in both clinical and research applications. Specifically, constructing accurate 3D models from 2D slices remains challenging due to tissue deformation, sectioning artifacts, variability in imaging techniques, and inconsistent illumination. Deep learning-based registration methods have demonstrated improved performance but suffer from limited generalizability and require large-scale training data. In contrast, non-deep-learning approaches offer better generalizability but often compromise on accuracy. In this study, we introduced ZeroReg3D, a novel zero-shot registration pipeline tailored for accurate 3D reconstruction from serial histological sections. By combining zero-shot deep learning-based keypoint matching with optimization-based affine and non-rigid registration techniques, ZeroReg3D effectively addresses critical challenges such as tissue deformation, sectioning artifacts, staining variability, and inconsistent illumination without requiring retraining or fine-tuning. The code has been made publicly available at https://github.com/hrlblab/ZeroReg3D
>
---
#### [replaced 107] Dual Frequency Branch Framework with Reconstructed Sliding Windows Attention for AI-Generated Image Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.15253v2](http://arxiv.org/pdf/2501.15253v2)**

> **作者:** Jiazhen Yan; Ziqiang Li; Fan Wang; Ziwen He; Zhangjie Fu
>
> **备注:** under review
>
> **摘要:** The rapid advancement of Generative Adversarial Networks (GANs) and diffusion models has enabled the creation of highly realistic synthetic images, presenting significant societal risks, such as misinformation and deception. As a result, detecting AI-generated images has emerged as a critical challenge. Existing researches emphasize extracting fine-grained features to enhance detector generalization, yet they often lack consideration for the importance and interdependencies of internal elements within local regions and are limited to a single frequency domain, hindering the capture of general forgery traces. To overcome the aforementioned limitations, we first utilize a sliding window to restrict the attention mechanism to a local window, and reconstruct the features within the window to model the relationships between neighboring internal elements within the local region. Then, we design a dual frequency domain branch framework consisting of four frequency domain subbands of DWT and the phase part of FFT to enrich the extraction of local forgery features from different perspectives. Through feature enrichment of dual frequency domain branches and fine-grained feature extraction of reconstruction sliding window attention, our method achieves superior generalization detection capabilities on both GAN and diffusion model-based generative images. Evaluated on diverse datasets comprising images from 65 distinct generative models, our approach achieves a 2.13\% improvement in detection accuracy over state-of-the-art methods.
>
---
#### [replaced 108] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07317v2](http://arxiv.org/pdf/2507.07317v2)**

> **作者:** Sherry X. Chen; Yi Wei; Luowei Zhou; Suren Kumar
>
> **备注:** International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** Recent advances in instruction-guided image editing underscore the need for effective automated evaluation. While Vision-Language Models (VLMs) have been explored as judges, open-source models struggle with alignment, and proprietary models lack transparency and cost efficiency. Additionally, no public training datasets exist to fine-tune open-source VLMs, only small benchmarks with diverse evaluation schemes. To address this, we introduce ADIEE, an automated dataset creation approach which is then used to train a scoring model for instruction-guided image editing evaluation. We generate a large-scale dataset with over 100K samples and use it to fine-tune a LLaVA-NeXT-8B model modified to decode a numeric score from a custom token. The resulting scorer outperforms all open-source VLMs and Gemini-Pro 1.5 across all benchmarks, achieving a 0.0696 (+17.24%) gain in score correlation with human ratings on AURORA-Bench, and improving pair-wise comparison accuracy by 4.03% (+7.21%) on GenAI-Bench and 4.75% (+9.35%) on AURORA-Bench, respectively, compared to the state-of-the-art. The scorer can act as a reward model, enabling automated best edit selection and model fine-tuning. Notably, the proposed scorer can boost MagicBrush model's average evaluation score on ImagenHub from 5.90 to 6.43 (+8.98%). Our code and models are available at https://github.com/SherryXTChen/ADIEE.git.
>
---
#### [replaced 109] Activator: GLU Activation Function as the Core Component of a Vision Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.15953v3](http://arxiv.org/pdf/2405.15953v3)**

> **作者:** Abdullah Nazhat Abdullah; Tarkan Aydin
>
> **摘要:** The transformer architecture has driven many successes in a variety of tasks within the field of deep learning, in particular the recent advances in natural language processing (NLP) culminating with large language models (LLM). Adding to that success, transformer architecture has found widespread interest from computer vision (CV) researchers and practitioners, allowing for many advancements in vision-related tasks and opening the door for multitask and multi-modal deep learning architectures that share the same principle of operation. One drawback to these architectures is their reliance on the scaled dot product attention mechanism with the softmax activation function, which is computationally expensive and requires large compute capabilities for both training and inference. This paper investigates substituting the MLP and attention mechanism usually adopted for transformer architecture with an architecture based on incorporating a gated linear unit (GLU) activation function structure with the aim of reducing the computational cost. The equalized experimental assessments conducted in this work show that the proposed modification with the targeted reductions in computational complexity offers competitive performance compared to the selected baseline architectures. The results are significantly in support of the aims of this work, in which the focus was to extensively utilize GLU-based MLPs, establishing a more efficient but capable alternative to the traditional MLP and the attention mechanism as the core component in the design of transformer architectures.
>
---
#### [replaced 110] Towards End-to-End Neuromorphic Event-based 3D Object Reconstruction Without Physical Priors
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.00741v4](http://arxiv.org/pdf/2501.00741v4)**

> **作者:** Chuanzhi Xu; Langyi Chen; Haodong Chen; Vera Chung; Qiang Qu
>
> **备注:** 6 pages, 3 figures, 5 tables, accepted by IEEE International Conference on Multimedia & Expo (ICME) 2025
>
> **摘要:** Neuromorphic cameras, also known as event cameras, are asynchronous brightness-change sensors that can capture extremely fast motion without suffering from motion blur, making them particularly promising for 3D reconstruction in extreme environments. However, existing research on 3D reconstruction using monocular neuromorphic cameras is limited, and most of the methods rely on estimating physical priors and employ complex multi-step pipelines. In this work, we propose an end-to-end method for dense voxel 3D reconstruction using neuromorphic cameras that eliminates the need to estimate physical priors. Our method incorporates a novel event representation to enhance edge features, enabling the proposed feature-enhancement model to learn more effectively. Additionally, we introduced Optimal Binarization Threshold Selection Principle as a guideline for future related work, using the optimal reconstruction results achieved with threshold optimization as the benchmark. Our method achieves a 54.6% improvement in reconstruction accuracy compared to the baseline method.
>
---
#### [replaced 111] Eyes Will Shut: A Vision-Based Next GPS Location Prediction Model by Reinforcement Learning from Visual Map Feed Back
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.18661v2](http://arxiv.org/pdf/2507.18661v2)**

> **作者:** Ruixing Zhang; Yang Zhang; Tongyu Zhu; Leilei Sun; Weifeng Lv
>
> **摘要:** Next Location Prediction is a fundamental task in the study of human mobility, with wide-ranging applications in transportation planning, urban governance, and epidemic forecasting. In practice, when humans attempt to predict the next location in a trajectory, they often visualize the trajectory on a map and reason based on road connectivity and movement trends. However, the vast majority of existing next-location prediction models do not reason over maps \textbf{in the way that humans do}. Fortunately, the recent development of Vision-Language Models (VLMs) has demonstrated strong capabilities in visual perception and even visual reasoning. This opens up a new possibility: by rendering both the road network and trajectory onto an image and leveraging the reasoning abilities of VLMs, we can enable models to perform trajectory inference in a human-like manner. To explore this idea, we first propose a method called Vision-Guided Location Search (VGLS), which evaluates whether a general-purpose VLM is capable of trajectory-based reasoning without modifying any of its internal parameters. Based on insights from the VGLS results, we further propose our main approach: VLMLocPredictor, which is composed of two stages: In the first stage, we design two Supervised Fine-Tuning (SFT) tasks that help the VLM understand road network and trajectory structures and acquire basic reasoning ability on such visual inputs. In the second stage, we introduce Reinforcement Learning from Visual Map Feedback, enabling the model to self-improve its next-location prediction ability through interaction with the environment. Experiments conducted on datasets from four different cities show that our method achieves state-of-the-art (SOTA) performance and exhibits superior cross-city generalization compared to other LLM-based approaches.
>
---
#### [replaced 112] Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.11170v2](http://arxiv.org/pdf/2412.11170v2)**

> **作者:** Yujie Zhang; Bingyang Cui; Qi Yang; Zhu Li; Yiling Xu
>
> **摘要:** Text-to-3D generation has achieved remarkable progress in recent years, yet evaluating these methods remains challenging for two reasons: i) Existing benchmarks lack fine-grained evaluation on different prompt categories and evaluation dimensions. ii) Previous evaluation metrics only focus on a single aspect (e.g., text-3D alignment) and fail to perform multi-dimensional quality assessment. To address these problems, we first propose a comprehensive benchmark named MATE-3D. The benchmark contains eight well-designed prompt categories that cover single and multiple object generation, resulting in 1,280 generated textured meshes. We have conducted a large-scale subjective experiment from four different evaluation dimensions and collected 107,520 annotations, followed by detailed analyses of the results. Based on MATE-3D, we propose a novel quality evaluator named HyperScore. Utilizing hypernetwork to generate specified mapping functions for each evaluation dimension, our metric can effectively perform multi-dimensional quality assessment. HyperScore presents superior performance over existing metrics on MATE-3D, making it a promising metric for assessing and improving text-to-3D generation. The project is available at https://mate-3d.github.io/.
>
---
#### [replaced 113] Otter: A Multi-Modal Model with In-Context Instruction Tuning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2305.03726v2](http://arxiv.org/pdf/2305.03726v2)**

> **作者:** Bo Li; Yuanhan Zhang; Liangyu Chen; Jinghao Wang; Fanyi Pu; Joshua Adrian Cahyono; Jingkang Yang; Ziwei Liu
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025
>
> **摘要:** Recent advances in Large Multimodal Models (LMMs) have unveiled great potential as visual assistants. However, most existing works focus on responding to individual instructions or using previous dialogues for contextual understanding. There is little discussion on employing both images and text as in-context examples to enhance the instruction following capability. To bridge this gap, we introduce the \textbf{Otter} model to leverage both textual and visual in-context examples for instruction tuning. Specifically, Otter builds upon Flamingo with Perceiver architecture, and has been instruction tuned for general purpose multi-modal assistant. Otter seamlessly processes multi-modal inputs, supporting modalities including text, multiple images, and dynamic video content. To support the training of Otter, we present the \textbf{MIMIC-IT} (\textbf{M}ult\textbf{I}-\textbf{M}odal \textbf{I}n-\textbf{C}ontext \textbf{I}nstruction \textbf{T}uning) dataset, which encompasses over 3 million multi-modal instruction-response pairs, including approximately 2.2 million unique instructions across a broad spectrum of images and videos. MIMIC-IT has been carefully curated to feature a diverse array of in-context examples for each entry. Comprehensive evaluations suggest that instruction tuning with these in-context examples substantially enhances model convergence and generalization capabilities. Notably, the extensive scenario coverage provided by the MIMIC-IT dataset empowers the Otter model to excel in tasks involving complex video and multi-image understanding.
>
---
#### [replaced 114] Unreal is all you need: Multimodal ISAC Data Simulation with Only One Engine
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08716v3](http://arxiv.org/pdf/2507.08716v3)**

> **作者:** Kongwu Huang; Shiyi Mu; Jun Jiang; Yuan Gao; Shugong Xu
>
> **摘要:** Scaling laws have achieved success in LLM and foundation models. To explore their potential in ISAC research, we propose Great-X. This single-engine multimodal data twin platform reconstructs the ray-tracing computation of Sionna within Unreal Engine and is deeply integrated with autonomous driving tools. This enables efficient and synchronized simulation of multimodal data, including CSI, RGB, Radar, and LiDAR. Based on this platform, we construct an open-source, large-scale, low-altitude UAV multimodal synaesthesia dataset named Great-MSD, and propose a baseline CSI-based UAV 3D localization algorithm, demonstrating its feasibility and generalizability across different CSI simulation engines. The related code and dataset will be made available at: https://github.com/hkw-xg/Great-MCD.
>
---
#### [replaced 115] BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.16907v2](http://arxiv.org/pdf/2504.16907v2)**

> **作者:** Ruotong Wang; Mingli Zhu; Jiarong Ou; Rui Chen; Xin Tao; Pengfei Wan; Baoyuan Wu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Text-to-video (T2V) generative models have rapidly advanced and found widespread applications across fields like entertainment, education, and marketing. However, the adversarial vulnerabilities of these models remain rarely explored. We observe that in T2V generation tasks, the generated videos often contain substantial redundant information not explicitly specified in the text prompts, such as environmental elements, secondary objects, and additional details, providing opportunities for malicious attackers to embed hidden harmful content. Exploiting this inherent redundancy, we introduce BadVideo, the first backdoor attack framework tailored for T2V generation. Our attack focuses on designing target adversarial outputs through two key strategies: (1) Spatio-Temporal Composition, which combines different spatiotemporal features to encode malicious information; (2) Dynamic Element Transformation, which introduces transformations in redundant elements over time to convey malicious information. Based on these strategies, the attacker's malicious target seamlessly integrates with the user's textual instructions, providing high stealthiness. Moreover, by exploiting the temporal dimension of videos, our attack successfully evades traditional content moderation systems that primarily analyze spatial information within individual frames. Extensive experiments demonstrate that BadVideo achieves high attack success rates while preserving original semantics and maintaining excellent performance on clean inputs. Overall, our work reveals the adversarial vulnerability of T2V models, calling attention to potential risks and misuse. Our project page is at https://wrt2000.github.io/BadVideo2025/.
>
---
#### [replaced 116] GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.10473v2](http://arxiv.org/pdf/2507.10473v2)**

> **作者:** David G. Shatwell; Ishan Rajendrakumar Dave; Sirnam Swetha; Mubarak Shah
>
> **备注:** Accepted in ICCV2025
>
> **摘要:** Timestamp prediction aims to determine when an image was captured using only visual information, supporting applications such as metadata correction, retrieval, and digital forensics. In outdoor scenarios, hourly estimates rely on cues like brightness, hue, and shadow positioning, while seasonal changes and weather inform date estimation. However, these visual cues significantly depend on geographic context, closely linking timestamp prediction to geo-localization. To address this interdependence, we introduce GT-Loc, a novel retrieval-based method that jointly predicts the capture time (hour and month) and geo-location (GPS coordinates) of an image. Our approach employs separate encoders for images, time, and location, aligning their embeddings within a shared high-dimensional feature space. Recognizing the cyclical nature of time, instead of conventional contrastive learning with hard positives and negatives, we propose a temporal metric-learning objective providing soft targets by modeling pairwise time differences over a cyclical toroidal surface. We present new benchmarks demonstrating that our joint optimization surpasses previous time prediction methods, even those using the ground-truth geo-location as an input during inference. Additionally, our approach achieves competitive results on standard geo-localization tasks, and the unified embedding space facilitates compositional and text-based image retrieval.
>
---
#### [replaced 117] MASQUE: A Text-Guided Diffusion-Based Framework for Localized and Customized Adversarial Makeup
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2503.10549v2](http://arxiv.org/pdf/2503.10549v2)**

> **作者:** Youngjin Kwon; Xiao Zhang
>
> **摘要:** As facial recognition is increasingly adopted for government and commercial services, its potential misuse has raised serious concerns about privacy and civil rights. To counteract, various anti-facial recognition techniques have been proposed for privacy protection by adversarially perturbing face images, among which generative makeup-based approaches are the most popular. However, these methods, designed primarily to impersonate specific target identities, can only achieve weak dodging success rates while increasing the risk of targeted abuse. In addition, they often introduce global visual artifacts or a lack of adaptability to accommodate diverse makeup prompts, compromising user satisfaction. To address the above limitations, we develop MASQUE, a novel diffusion-based framework that generates localized adversarial makeups guided by user-defined text prompts. Built upon precise null-text inversion, customized cross-attention fusion with masking, and a pairwise adversarial guidance mechanism using images of the same individual, MASQUE achieves robust dodging performance without requiring any external identity. Comprehensive evaluations on open-source facial recognition models and commercial APIs demonstrate that MASQUE significantly improves dodging success rates over all baselines, along with higher perceptual fidelity and stronger adaptability to various text makeup prompts.
>
---
#### [replaced 118] The DeepSpeak Dataset
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.05366v4](http://arxiv.org/pdf/2408.05366v4)**

> **作者:** Sarah Barrington; Matyas Bohacek; Hany Farid
>
> **摘要:** Deepfakes represent a growing concern across domains such as impostor hiring, fraud, and disinformation. Despite significant efforts to develop robust detection classifiers to distinguish the real from the fake, commonly used training datasets remain inadequate: relying on low-quality and outdated deepfake generators, consisting of content scraped from online repositories without participant consent, lacking in multimodal coverage, and rarely employing identity-matching protocols to ensure realistic fakes. To overcome these limitations, we present the DeepSpeak dataset, a diverse and multimodal dataset comprising over 100 hours of authentic and deepfake audiovisual content. We contribute: i) more than 50 hours of real, self-recorded data collected from 500 diverse and consenting participants using a custom-built data collection tool, ii) more than 50 hours of state-of-the-art audio and visual deepfakes generated using 14 video synthesis engines and three voice cloning engines, and iii) an embedding-based, identity-matching approach to ensure the creation of convincing, high-quality identity swaps that realistically simulate adversarial deepfake attacks. We also perform large-scale evaluations of state-of-the-art deepfake detectors and show that, without retraining, these detectors fail to generalize to the DeepSpeak dataset. These evaluations highlight the importance of a large and diverse dataset containing deepfakes from the latest generative-AI tools.
>
---
#### [replaced 119] Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16534v2](http://arxiv.org/pdf/2507.16534v2)**

> **作者:** Shanghai AI Lab; :; Xiaoyang Chen; Yunhao Chen; Zeren Chen; Zhiyun Chen; Hanyun Cui; Yawen Duan; Jiaxuan Guo; Qi Guo; Xuhao Hu; Hong Huang; Lige Huang; Chunxiao Li; Juncheng Li; Qihao Lin; Dongrui Liu; Xinmin Liu; Zicheng Liu; Chaochao Lu; Xiaoya Lu; Jingjing Qu; Qibing Ren; Jing Shao; Jingwei Shi; Jingwei Sun; Peng Wang; Weibing Wang; Jia Xu; Lewen Yan; Xiao Yu; Yi Yu; Boxuan Zhang; Jie Zhang; Weichen Zhang; Zhijie Zheng; Tianyi Zhou; Bowen Zhou
>
> **备注:** 97 pages, 37 figures
>
> **摘要:** To understand and identify the unprecedented risks posed by rapidly advancing artificial intelligence (AI) models, this report presents a comprehensive assessment of their frontier risks. Drawing on the E-T-C analysis (deployment environment, threat source, enabling capability) from the Frontier AI Risk Management Framework (v1.0) (SafeWork-F1-Framework), we identify critical risks in seven areas: cyber offense, biological and chemical risks, persuasion and manipulation, uncontrolled autonomous AI R\&D, strategic deception and scheming, self-replication, and collusion. Guided by the "AI-$45^\circ$ Law," we evaluate these risks using "red lines" (intolerable thresholds) and "yellow lines" (early warning indicators) to define risk zones: green (manageable risk for routine deployment and continuous monitoring), yellow (requiring strengthened mitigations and controlled deployment), and red (necessitating suspension of development and/or deployment). Experimental results show that all recent frontier AI models reside in green and yellow zones, without crossing red lines. Specifically, no evaluated models cross the yellow line for cyber offense or uncontrolled AI R\&D risks. For self-replication, and strategic deception and scheming, most models remain in the green zone, except for certain reasoning models in the yellow zone. In persuasion and manipulation, most models are in the yellow zone due to their effective influence on humans. For biological and chemical risks, we are unable to rule out the possibility of most models residing in the yellow zone, although detailed threat modeling and in-depth assessment are required to make further claims. This work reflects our current understanding of AI frontier risks and urges collective action to mitigate these challenges.
>
---
#### [replaced 120] Adaptive Real-Time Multi-Loss Function Optimization Using Dynamic Memory Fusion Framework: A Case Study on Breast Cancer Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.19745v2](http://arxiv.org/pdf/2410.19745v2)**

> **作者:** Amin Golnari; Mostafa Diba
>
> **摘要:** Deep learning has proven to be a highly effective tool for a wide range of applications, significantly when leveraging the power of multi-loss functions to optimize performance on multiple criteria simultaneously. However, optimal selection and weighting loss functions in deep learning tasks can significantly influence model performance, yet manual tuning of these functions is often inefficient and inflexible. We propose a novel framework called dynamic memory fusion for adaptive multi-loss function penalizing in real-time to address this. This framework leverages historical loss values data to dynamically adjust the weighting of multiple loss functions throughout the training process. Additionally, this framework integrates an auxiliary loss function to enhance model performance in the early stages. To further research horizons, we introduce the class-balanced dice loss function, designed to address class imbalance by prioritizing underrepresented classes. Experiments on breast ultrasound datasets demonstrate that the framework improves segmentation performance across various metrics. These results demonstrate the effectiveness of our proposed framework in ensuring that the model dynamically adjusts its focus to prioritize the most relevant criteria, leading to improved performance in evolving environments. The source code for our proposed methodology is publicly available on GitHub.
>
---
#### [replaced 121] APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation
- **分类: cs.NE; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.14270v3](http://arxiv.org/pdf/2507.14270v3)**

> **作者:** Ravin Kumar
>
> **备注:** 10 pages, 2 figures, 1 table. Includes a GitHub repository for MNIST experiments and a PyPI package for APTx Neuron implementation
>
> **摘要:** We propose the APTx Neuron, a novel, unified neural computation unit that integrates non-linear activation and linear transformation into a single trainable expression. The APTx Neuron is derived from the APTx activation function, thereby eliminating the need for separate activation layers and making the architecture both computationally efficient and elegant. The proposed neuron follows the functional form $y = \sum_{i=1}^{n} ((\alpha_i + \tanh(\beta_i x_i)) \cdot \gamma_i x_i) + \delta$, where all parameters $\alpha_i$, $\beta_i$, $\gamma_i$, and $\delta$ are trainable. We validate our APTx Neuron-based architecture on the MNIST dataset, achieving up to 96.69% test accuracy within 11 epochs using approximately 332K trainable parameters. The results highlight the superior expressiveness and computational efficiency of the APTx Neuron compared to traditional neurons, pointing toward a new paradigm in unified neuron design and the architectures built upon it.
>
---
#### [replaced 122] 3DGen-Bench: Comprehensive Benchmark Suite for 3D Generative Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21745v3](http://arxiv.org/pdf/2503.21745v3)**

> **作者:** Yuhan Zhang; Mengchen Zhang; Tong Wu; Tengfei Wang; Gordon Wetzstein; Dahua Lin; Ziwei Liu
>
> **备注:** Page: https://zyh482.github.io/3DGen-Bench/ ; Code: https://github.com/3DTopia/3DGen-Bench
>
> **摘要:** 3D generation is experiencing rapid advancements, while the development of 3D evaluation has not kept pace. How to keep automatic evaluation equitably aligned with human perception has become a well-recognized challenge. Recent advances in the field of language and image generation have explored human preferences and showcased respectable fitting ability. However, the 3D domain still lacks such a comprehensive preference dataset over generative models. To mitigate this absence, we develop 3DGen-Arena, an integrated platform in a battle manner. Then, we carefully design diverse text and image prompts and leverage the arena platform to gather human preferences from both public users and expert annotators, resulting in a large-scale multi-dimension human preference dataset 3DGen-Bench. Using this dataset, we further train a CLIP-based scoring model, 3DGen-Score, and a MLLM-based automatic evaluator, 3DGen-Eval. These two models innovatively unify the quality evaluation of text-to-3D and image-to-3D generation, and jointly form our automated evaluation system with their respective strengths. Extensive experiments demonstrate the efficacy of our scoring model in predicting human preferences, exhibiting a superior correlation with human ranks compared to existing metrics. We believe that our 3DGen-Bench dataset and automated evaluation system will foster a more equitable evaluation in the field of 3D generation, further promoting the development of 3D generative models and their downstream applications. Project page is available at https://zyh482.github.io/3DGen-Bench/.
>
---
#### [replaced 123] IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09923v3](http://arxiv.org/pdf/2507.09923v3)**

> **作者:** Sejin Park; Sangmin Lee; Kyong Hwan Jin; Seung-Won Jung
>
> **备注:** ICCV 2025
>
> **摘要:** Super-resolution (SR) has been a pivotal task in image processing, aimed at enhancing image resolution across various applications. Recently, look-up table (LUT)-based approaches have attracted interest due to their efficiency and performance. However, these methods are typically designed for fixed scale factors, making them unsuitable for arbitrary-scale image SR (ASISR). Existing ASISR techniques often employ implicit neural representations, which come with considerable computational cost and memory demands. To address these limitations, we propose Interpolation Mixing LUT (IM-LUT), a novel framework that operates ASISR by learning to blend multiple interpolation functions to maximize their representational capacity. Specifically, we introduce IM-Net, a network trained to predict mixing weights for interpolation functions based on local image patterns and the target scale factor. To enhance efficiency of interpolation-based methods, IM-Net is transformed into IM-LUT, where LUTs are employed to replace computationally expensive operations, enabling lightweight and fast inference on CPUs while preserving reconstruction quality. Experimental results on several benchmark datasets demonstrate that IM-LUT consistently achieves a superior balance between image quality and efficiency compared to existing methods, highlighting its potential as a promising solution for resource-constrained applications.
>
---
#### [replaced 124] ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11261v2](http://arxiv.org/pdf/2507.11261v2)**

> **作者:** Ronggang Huang; Haoxin Yang; Yan Cai; Xuemiao Xu; Huaidong Zhang; Shengfeng He
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** 3D visual grounding aims to identify and localize objects in a 3D space based on textual descriptions. However, existing methods struggle with disentangling targets from anchors in complex multi-anchor queries and resolving inconsistencies in spatial descriptions caused by perspective variations. To tackle these challenges, we propose ViewSRD, a framework that formulates 3D visual grounding as a structured multi-view decomposition process. First, the Simple Relation Decoupling (SRD) module restructures complex multi-anchor queries into a set of targeted single-anchor statements, generating a structured set of perspective-aware descriptions that clarify positional relationships. These decomposed representations serve as the foundation for the Multi-view Textual-Scene Interaction (Multi-TSI) module, which integrates textual and scene features across multiple viewpoints using shared, Cross-modal Consistent View Tokens (CCVTs) to preserve spatial correlations. Finally, a Textual-Scene Reasoning module synthesizes multi-view predictions into a unified and robust 3D visual grounding. Experiments on 3D visual grounding datasets show that ViewSRD significantly outperforms state-of-the-art methods, particularly in complex queries requiring precise spatial differentiation. Code is available at https://github.com/visualjason/ViewSRD.
>
---
#### [replaced 125] A Memory-Efficient Framework for Deformable Transformer with Neural Architecture Search
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.11549v2](http://arxiv.org/pdf/2507.11549v2)**

> **作者:** Wendong Mao; Mingfan Zhao; Jianfeng Guan; Qiwei Dong; Zhongfeng Wang
>
> **备注:** 5 pages
>
> **摘要:** Deformable Attention Transformers (DAT) have shown remarkable performance in computer vision tasks by adaptively focusing on informative image regions. However, their data-dependent sampling mechanism introduces irregular memory access patterns, posing significant challenges for efficient hardware deployment. Existing acceleration methods either incur high hardware overhead or compromise model accuracy. To address these issues, this paper proposes a hardware-friendly optimization framework for DAT. First, a neural architecture search (NAS)-based method with a new slicing strategy is proposed to automatically divide the input feature into uniform patches during the inference process, avoiding memory conflicts without modifying model architecture. The method explores the optimal slice configuration by jointly optimizing hardware cost and inference accuracy. Secondly, an FPGA-based verification system is designed to test the performance of this framework on edge-side hardware. Algorithm experiments on the ImageNet-1K dataset demonstrate that our hardware-friendly framework can maintain have only 0.2% accuracy drop compared to the baseline DAT. Hardware experiments on Xilinx FPGA show the proposed method reduces DRAM access times to 18% compared with existing DAT acceleration methods.
>
---
#### [replaced 126] SDTrack: A Baseline for Event-based Tracking via Spiking Neural Networks
- **分类: cs.NE; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08703v3](http://arxiv.org/pdf/2503.08703v3)**

> **作者:** Yimeng Shan; Zhenbang Ren; Haodi Wu; Wenjie Wei; Rui-Jie Zhu; Shuai Wang; Dehao Zhang; Yichen Xiao; Jieyuan Zhang; Kexin Shi; Jingzhinan Wang; Jason K. Eshraghian; Haicheng Qu; Malu Zhang
>
> **备注:** 10 pages,7 figures,3 tables
>
> **摘要:** Event cameras provide superior temporal resolution, dynamic range, power efficiency, and pixel bandwidth. Spiking Neural Networks (SNNs) naturally complement event data through discrete spike signals, making them ideal for event-based tracking. However, current approaches that combine Artificial Neural Networks (ANNs) and SNNs, along with suboptimal architectures, compromise energy efficiency and limit tracking performance. To address these limitations, we propose the first Transformer-based spike-driven tracking pipeline. Our Global Trajectory Prompt (GTP) method effectively captures global trajectory information and aggregates it with event streams into event images to enhance spatiotemporal representation. We then introduce SDTrack, a Transformer-based spike-driven tracker comprising a Spiking MetaFormer backbone and a tracking head that directly predicts normalized coordinates using spike signals. The framework is end-to-end, does not require data augmentation or post-processing. Extensive experiments demonstrate that SDTrack achieves state-of-the-art performance while maintaining the lowest parameter count and energy consumption across multiple event-based tracking benchmarks, establishing a solid baseline for future research in the field of neuromorphic vision.
>
---
#### [replaced 127] Continual Low-Rank Scaled Dot-product Attention
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.03214v4](http://arxiv.org/pdf/2412.03214v4)**

> **作者:** Ginés Carreto Picón; Illia Oleksiienko; Lukas Hedegaard; Arian Bakhtiarnia; Alexandros Iosifidis
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** Transformers are widely used for their ability to capture data relations in sequence processing, with great success for a wide range of static tasks. However, the computational and memory footprint of their main component, i.e., the Scaled Dot-product Attention, is commonly overlooked. This makes their adoption in applications involving stream data processing with constraints in response latency, computational and memory resources infeasible. Some works have proposed methods to lower the computational cost of Transformers, i.e. low-rank approximations, sparsity in attention, and efficient formulations for Continual Inference. In this paper, we introduce a new formulation of the Scaled Dot-product Attention based on the Nystr\"om approximation that is suitable for Continual Inference. In experiments on Online Audio Classification and Online Action Detection tasks, the proposed Continual Scaled Dot-product Attention can lower the number of operations by up to three orders of magnitude compared to the original Transformers while retaining the predictive performance of competing models.
>
---
#### [replaced 128] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15602v2](http://arxiv.org/pdf/2507.15602v2)**

> **作者:** Zihui Gao; Jia-Wang Bian; Guosheng Lin; Hao Chen; Chunhua Shen
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Surface reconstruction and novel view rendering from sparse-view images are challenging. Signed Distance Function (SDF)-based methods struggle with fine details, while 3D Gaussian Splatting (3DGS)-based approaches lack global geometry coherence. We propose a novel hybrid method that combines the strengths of both approaches: SDF captures coarse geometry to enhance 3DGS-based rendering, while newly rendered images from 3DGS refine the details of SDF for accurate surface reconstruction. As a result, our method surpasses state-of-the-art approaches in surface reconstruction and novel view synthesis on the DTU and MobileBrick datasets. Code will be released at https://github.com/aim-uofa/SurfaceSplat.
>
---
#### [replaced 129] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07266v3](http://arxiv.org/pdf/2503.07266v3)**

> **作者:** Fu Rong; Meng Lan; Qian Zhang; Lefei Zhang
>
> **摘要:** Referring Remote Sensing Image Segmentation (RRSIS) aims to segment target objects in remote sensing (RS) images based on textual descriptions. Although Segment Anything Model 2 (SAM2) has shown remarkable performance in various segmentation tasks, its application to RRSIS presents several challenges, including understanding the text-described RS scenes and generating effective prompts from text descriptions. To address these issues, we propose RS2-SAM2, a novel framework that adapts SAM2 to RRSIS by aligning the adapted RS features and textual features, providing pseudo-mask-based dense prompts, and enforcing boundary constraints. Specifically, we employ a union encoder to jointly encode the visual and textual inputs, generating aligned visual and text embeddings as well as multimodal class tokens. A bidirectional hierarchical fusion module is introduced to adapt SAM2 to RS scenes and align adapted visual features with the visually enhanced text embeddings, improving the model's interpretation of text-described RS scenes. To provide precise target cues for SAM2, we design a mask prompt generator, which takes the visual embeddings and class tokens as input and produces a pseudo-mask as the dense prompt of SAM2. Experimental results on several RRSIS benchmarks demonstrate that RS2-SAM2 achieves state-of-the-art performance.
>
---
#### [replaced 130] CopyJudge: Automated Copyright Infringement Identification and Mitigation in Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15278v2](http://arxiv.org/pdf/2502.15278v2)**

> **作者:** Shunchang Liu; Zhuan Shi; Lingjuan Lyu; Yaochu Jin; Boi Faltings
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Assessing whether AI-generated images are substantially similar to source works is a crucial step in resolving copyright disputes. In this paper, we propose CopyJudge, a novel automated infringement identification framework that leverages large vision-language models (LVLMs) to simulate practical court processes for determining substantial similarity between copyrighted images and those generated by text-to-image diffusion models. Specifically, we employ an abstraction-filtration-comparison test framework based on the multi-LVLM debate to assess the likelihood of infringement and provide detailed judgment rationales. Based on these judgments, we further introduce a general LVLM-based mitigation strategy that automatically optimizes infringing prompts by avoiding sensitive expressions while preserving the non-infringing content. Furthermore, assuming the input noise is controllable, our approach can be enhanced by iteratively exploring non-infringing noise vectors within the diffusion latent space, even without modifying the original prompts. Experimental results show that our automated identification method achieves comparable state-of-the-art performance, while offering superior generalization and interpretability across various forms of infringement, and that our mitigation method more effectively mitigates memorization and IP infringement with a high degree of alignment to the original non-infringing expressions.
>
---
#### [replaced 131] ZERO: Industry-ready Vision Foundation Model with Multi-modal Prompts
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.04270v2](http://arxiv.org/pdf/2507.04270v2)**

> **作者:** Sangbum Choi; Kyeongryeol Go; Taewoong Jang
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Foundation models have revolutionized AI, yet they struggle with zero-shot deployment in real-world industrial settings due to a lack of high-quality, domain-specific datasets. To bridge this gap, Superb AI introduces ZERO, an industry-ready vision foundation model that leverages multi-modal prompting (textual and visual) for generalization without retraining. Trained on a compact yet representative 0.9 million annotated samples from a proprietary billion-scale industrial dataset, ZERO demonstrates competitive performance on academic benchmarks like LVIS-Val and significantly outperforms existing models across 37 diverse industrial datasets. Furthermore, ZERO achieved 2nd place in the CVPR 2025 Object Instance Detection Challenge and 4th place in the Foundational Few-shot Object Detection Challenge, highlighting its practical deployability and generalizability with minimal adaptation and limited data. To the best of our knowledge, ZERO is the first vision foundation model explicitly built for domain-specific, zero-shot industrial applications.
>
---
#### [replaced 132] The Origin of Self-Attention: Pairwise Affinity Matrices in Feature Selection and the Emergence of Self-Attention
- **分类: cs.LG; cs.CV; 68T07, 05C50, 15A18; I.2.6; I.2.7; I.5.1**

- **链接: [http://arxiv.org/pdf/2507.14560v2](http://arxiv.org/pdf/2507.14560v2)**

> **作者:** Giorgio Roffo
>
> **备注:** 24 pages, 10 figures, submitted for review. Companion code and reproducibility materials available
>
> **摘要:** The self-attention mechanism, now central to deep learning architectures such as Transformers, is a modern instance of a more general computational principle: learning and using pairwise affinity matrices to control how information flows through a model. This paper traces the conceptual origins of self-attention across multiple domains, including computer vision, natural language processing, and graph learning, through their shared reliance on an affinity matrix, denoted as A. We highlight Infinite Feature Selection (Inf-FS) as a foundational approach that generalizes the idea of affinity-based weighting. Unlike the fixed dot-product structure used in Transformers, Inf-FS defines A either through domain knowledge or by learning, and computes feature relevance through multi-hop propagation over the affinity graph. From this perspective, self-attention can be seen as a special case of Inf-FS: it uses a single-hop affinity computation where A is dynamically built from token similarities. We argue that the underlying structure, reasoning over pairwise relationships, is preserved across both approaches, and the key differences lie in how the affinity matrix is defined and applied. By situating self-attention within the broader paradigm of affinity-based computation, we unify several strands of machine learning research and highlight a common mathematical foundation that underpins diverse models and tasks.
>
---
#### [replaced 133] HLFormer: Enhancing Partially Relevant Video Retrieval with Hyperbolic Learning
- **分类: cs.CV; cs.IR; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.17402v2](http://arxiv.org/pdf/2507.17402v2)**

> **作者:** Jun Li; Jinpeng Wang; Chaolei Tan; Niu Lian; Long Chen; Yaowei Wang; Min Zhang; Shu-Tao Xia; Bin Chen
>
> **备注:** Accepted by ICCV'25. 13 pages, 6 figures, 4 tables
>
> **摘要:** Partially Relevant Video Retrieval (PRVR) addresses the critical challenge of matching untrimmed videos with text queries describing only partial content. Existing methods suffer from geometric distortion in Euclidean space that sometimes misrepresents the intrinsic hierarchical structure of videos and overlooks certain hierarchical semantics, ultimately leading to suboptimal temporal modeling. To address this issue, we propose the first hyperbolic modeling framework for PRVR, namely HLFormer, which leverages hyperbolic space learning to compensate for the suboptimal hierarchical modeling capabilities of Euclidean space. Specifically, HLFormer integrates the Lorentz Attention Block and Euclidean Attention Block to encode video embeddings in hybrid spaces, using the Mean-Guided Adaptive Interaction Module to dynamically fuse features. Additionally, we introduce a Partial Order Preservation Loss to enforce "text < video" hierarchy through Lorentzian cone constraints. This approach further enhances cross-modal matching by reinforcing partial relevance between video content and text queries. Extensive experiments show that HLFormer outperforms state-of-the-art methods. Code is released at https://github.com/lijun2005/ICCV25-HLFormer.
>
---
#### [replaced 134] LM-Gaussian: Boost Sparse-view 3D Gaussian Splatting with Large Model Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.03456v3](http://arxiv.org/pdf/2409.03456v3)**

> **作者:** Hanyang Yu; Xiaoxiao Long; Ping Tan
>
> **备注:** Project page: https://hanyangyu1021.github.io/lm-gaussian.github.io/
>
> **摘要:** We aim to address sparse-view reconstruction of a 3D scene by leveraging priors from large-scale vision models. While recent advancements such as 3D Gaussian Splatting (3DGS) have demonstrated remarkable successes in 3D reconstruction, these methods typically necessitate hundreds of input images that densely capture the underlying scene, making them time-consuming and impractical for real-world applications. However, sparse-view reconstruction is inherently ill-posed and under-constrained, often resulting in inferior and incomplete outcomes. This is due to issues such as failed initialization, overfitting on input images, and a lack of details. To mitigate these challenges, we introduce LM-Gaussian, a method capable of generating high-quality reconstructions from a limited number of images. Specifically, we propose a robust initialization module that leverages stereo priors to aid in the recovery of camera poses and the reliable point clouds. Additionally, a diffusion-based refinement is iteratively applied to incorporate image diffusion priors into the Gaussian optimization process to preserve intricate scene details. Finally, we utilize video diffusion priors to further enhance the rendered images for realistic visual effects. Overall, our approach significantly reduces the data acquisition requirements compared to previous 3DGS methods. We validate the effectiveness of our framework through experiments on various public datasets, demonstrating its potential for high-quality 360-degree scene reconstruction. Visual results are on our website.
>
---
#### [replaced 135] Text-guided multi-stage cross-perception network for medical image segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07475v2](http://arxiv.org/pdf/2506.07475v2)**

> **作者:** Gaoyu Chen; Haixia Pan
>
> **摘要:** Medical image segmentation plays a crucial role in clinical medicine, serving as a tool for auxiliary diagnosis, treatment planning, and disease monitoring, thus facilitating physicians in the study and treatment of diseases. However, existing medical image segmentation methods are limited by the weak semantic expression of the target segmentation regions, which is caused by the low contrast between the target and non-target segmentation regions. To address this limitation, text prompt information has greast potential to capture the lesion location. However, existing text-guided methods suffer from insufficient cross-modal interaction and inadequate cross-modal feature expression. To resolve these issues, we propose the Text-guided Multi-stage Cross-perception network (TMC). In TMC, we introduce a multistage cross-attention module to enhance the model's understanding of semantic details and a multi-stage alignment loss to improve the consistency of cross-modal semantics. The results of the experiments demonstrate that our TMC achieves a superior performance with Dice of 84.77%, 78.50%, 88.73% in three public datasets (QaTa-COV19, MosMedData and Breast), outperforming UNet based networks and text-guided methods.
>
---
#### [replaced 136] Are ECGs enough? Deep learning classification of pulmonary embolism using electrocardiograms
- **分类: cs.CV; cs.AI; cs.LG; I.2**

- **链接: [http://arxiv.org/pdf/2503.08960v2](http://arxiv.org/pdf/2503.08960v2)**

> **作者:** Joao D. S. Marques; Arlindo L. Oliveira
>
> **备注:** Accepted to the MIRASOL 2025 Workshop (MICCAI 2025)
>
> **摘要:** Pulmonary embolism is a leading cause of out of hospital cardiac arrest that requires fast diagnosis. While computed tomography pulmonary angiography is the standard diagnostic tool, it is not always accessible. Electrocardiography is an essential tool for diagnosing multiple cardiac anomalies, as it is affordable, fast and available in many settings. However, the availability of public ECG datasets, specially for PE, is limited and, in practice, these datasets tend to be small, making it essential to optimize learning strategies. In this study, we investigate the performance of multiple neural networks in order to assess the impact of various approaches. Moreover, we check whether these practices enhance model generalization when transfer learning is used to translate information learned in larger ECG datasets, such as PTB-XL, CPSC18 and MedalCare-XL, to a smaller, more challenging dataset for PE. By leveraging transfer learning, we analyze the extent to which we can improve learning efficiency and predictive performance on limited data. Code available at https://github.com/joaodsmarques/Are-ECGs-enough-Deep-Learning-Classifiers .
>
---
#### [replaced 137] Emerging Properties in Unified Multimodal Pretraining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14683v3](http://arxiv.org/pdf/2505.14683v3)**

> **作者:** Chaorui Deng; Deyao Zhu; Kunchang Li; Chenhui Gou; Feng Li; Zeyu Wang; Shu Zhong; Weihao Yu; Xiaonan Nie; Ziang Song; Guang Shi; Haoqi Fan
>
> **备注:** 37 pages, 17 figures
>
> **摘要:** Unifying multimodal understanding and generation has shown impressive capabilities in cutting-edge proprietary systems. In this work, we introduce BAGEL, an open-source foundational model that natively supports multimodal understanding and generation. BAGEL is a unified, decoder-only model pretrained on trillions of tokens curated from large-scale interleaved text, image, video, and web data. When scaled with such diverse multimodal interleaved data, BAGEL exhibits emerging capabilities in complex multimodal reasoning. As a result, it significantly outperforms open-source unified models in both multimodal generation and understanding across standard benchmarks, while exhibiting advanced multimodal reasoning abilities such as free-form image manipulation, future frame prediction, 3D manipulation, and world navigation. In the hope of facilitating further opportunities for multimodal research, we share the key findings, pretraining details, data creation protocal, and release our code and checkpoints to the community. The project page is at https://bagel-ai.org/
>
---
#### [replaced 138] TriDi: Trilateral Diffusion of 3D Humans, Objects, and Interactions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.06334v3](http://arxiv.org/pdf/2412.06334v3)**

> **作者:** Ilya A. Petrov; Riccardo Marin; Julian Chibane; Gerard Pons-Moll
>
> **备注:** 2025 IEEE/CVF International Conference on Computer Vision (ICCV)
>
> **摘要:** Modeling 3D human-object interaction (HOI) is a problem of great interest for computer vision and a key enabler for virtual and mixed-reality applications. Existing methods work in a one-way direction: some recover plausible human interactions conditioned on a 3D object; others recover the object pose conditioned on a human pose. Instead, we provide the first unified model - TriDi which works in any direction. Concretely, we generate Human, Object, and Interaction modalities simultaneously with a new three-way diffusion process, allowing to model seven distributions with one network. We implement TriDi as a transformer attending to the various modalities' tokens, thereby discovering conditional relations between them. The user can control the interaction either as a text description of HOI or a contact map. We embed these two representations into a shared latent space, combining the practicality of text descriptions with the expressiveness of contact maps. Using a single network, TriDi unifies all the special cases of prior work and extends to new ones, modeling a family of seven distributions. Remarkably, despite using a single model, TriDi generated samples surpass one-way specialized baselines on GRAB and BEHAVE in terms of both qualitative and quantitative metrics, and demonstrating better diversity. We show the applicability of TriDi to scene population, generating objects for human-contact datasets, and generalization to unseen object geometry. The project page is available at: https://virtualhumans.mpi-inf.mpg.de/tridi.
>
---
#### [replaced 139] Efficacy of Image Similarity as a Metric for Augmenting Small Dataset Retinal Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04862v2](http://arxiv.org/pdf/2507.04862v2)**

> **作者:** Thomas Wallace; Ik Siong Heng; Senad Subasic; Chris Messenger
>
> **备注:** 30 pages, 10 figures
>
> **摘要:** Synthetic images are an option for augmenting limited medical imaging datasets to improve the performance of various machine learning models. A common metric for evaluating synthetic image quality is the Fr\'echet Inception Distance (FID) which measures the similarity of two image datasets. In this study we evaluate the relationship between this metric and the improvement which synthetic images, generated by a Progressively Growing Generative Adversarial Network (PGGAN), grant when augmenting Diabetes-related Macular Edema (DME) intraretinal fluid segmentation performed by a U-Net model with limited amounts of training data. We find that the behaviour of augmenting with standard and synthetic images agrees with previously conducted experiments. Additionally, we show that dissimilar (high FID) datasets do not improve segmentation significantly. As FID between the training and augmenting datasets decreases, the augmentation datasets are shown to contribute to significant and robust improvements in image segmentation. Finally, we find that there is significant evidence to suggest that synthetic and standard augmentations follow separate log-normal trends between FID and improvements in model performance, with synthetic data proving more effective than standard augmentation techniques. Our findings show that more similar datasets (lower FID) will be more effective at improving U-Net performance, however, the results also suggest that this improvement may only occur when images are sufficiently dissimilar.
>
---
#### [replaced 140] Parameter-Efficient Fine-Tuning in Spectral Domain for Point Cloud Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.08114v2](http://arxiv.org/pdf/2410.08114v2)**

> **作者:** Dingkang Liang; Tianrui Feng; Xin Zhou; Yumeng Zhang; Zhikang Zou; Xiang Bai
>
> **备注:** Accepted by IEEE TPAMI. The code will be made available at https://github.com/jerryfeng2003/PointGST
>
> **摘要:** Recently, leveraging pre-training techniques to enhance point cloud models has become a prominent research topic. However, existing approaches typically require full fine-tuning of pre-trained models to achieve satisfactory performance on downstream tasks, which is storage-intensive and computationally demanding. To address this issue, we propose a novel Parameter-Efficient Fine-Tuning (PEFT) method for point cloud, called \textbf{PointGST} (\textbf{Point} cloud \textbf{G}raph \textbf{S}pectral \textbf{T}uning). PointGST freezes the pre-trained model and introduces a lightweight, trainable Point Cloud Spectral Adapter (PCSA) for fine-tuning parameters in the spectral domain. The core idea is built on two observations: 1) The inner tokens from frozen models might present confusion in the spatial domain; 2) Task-specific intrinsic information is important for transferring the general knowledge to the downstream task. Specifically, PointGST transfers the point tokens from the spatial domain to the spectral domain, effectively de-correlating confusion among tokens by using orthogonal components for separation. Moreover, the generated spectral basis involves intrinsic information about the downstream point clouds, enabling more targeted tuning. As a result, PointGST facilitates the efficient transfer of general knowledge to downstream tasks while significantly reducing training costs. Extensive experiments on challenging point cloud datasets across various tasks demonstrate that PointGST not only outperforms its fully fine-tuning counterpart but also significantly reduces trainable parameters, making it a promising solution for efficient point cloud learning. The code will be made available at https://github.com/jerryfeng2003/PointGST
>
---
#### [replaced 141] Benchmarking and Analyzing Generative Data for Visual Recognition
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2307.13697v2](http://arxiv.org/pdf/2307.13697v2)**

> **作者:** Bo Li; Haotian Liu; Liangyu Chen; Yong Jae Lee; Chunyuan Li; Ziwei Liu
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025
>
> **摘要:** Advancements in large pre-trained generative models have expanded their potential as effective data generators in visual recognition. This work delves into the impact of generative images, primarily comparing paradigms that harness external data (\ie generative \vs retrieval \vs original). Our key contributions are: \textbf{1) GenBench Construction:} We devise \textbf{GenBench}, a broad benchmark comprising 22 datasets with 2548 categories, to appraise generative data across various visual recognition tasks. \textbf{2) CLER Score:} To address the insufficient correlation of existing metrics (\eg, FID, CLIP score) with downstream recognition performance, we propose \textbf{CLER}, a training-free metric indicating generative data's efficiency for recognition tasks prior to training. \textbf{3) New Baselines:} Comparisons of generative data with retrieved data from the same external pool help to elucidate the unique traits of generative data. \textbf{4) External Knowledge Injection:} By fine-tuning special token embeddings for each category via Textual Inversion, performance improves across 17 datasets, except when dealing with low-resolution reference images. Our exhaustive benchmark and analysis spotlight generative data's promise in visual recognition, while identifying key challenges for future investigation.
>
---
#### [replaced 142] Mcity Data Engine: Iterative Model Improvement Through Open-Vocabulary Data Selection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21614v2](http://arxiv.org/pdf/2504.21614v2)**

> **作者:** Daniel Bogdoll; Rajanikant Patnaik Ananta; Abeyankar Giridharan; Isabel Moore; Gregory Stevens; Henry X. Liu
>
> **备注:** Accepted for publication at ITSC 2025
>
> **摘要:** With an ever-increasing availability of data, it has become more and more challenging to select and label appropriate samples for the training of machine learning models. It is especially difficult to detect long-tail classes of interest in large amounts of unlabeled data. This holds especially true for Intelligent Transportation Systems (ITS), where vehicle fleets and roadside perception systems generate an abundance of raw data. While industrial, proprietary data engines for such iterative data selection and model training processes exist, researchers and the open-source community suffer from a lack of an openly available system. We present the Mcity Data Engine, which provides modules for the complete data-based development cycle, beginning at the data acquisition phase and ending at the model deployment stage. The Mcity Data Engine focuses on rare and novel classes through an open-vocabulary data selection process. All code is publicly available on GitHub under an MIT license: https://github.com/mcity/mcity_data_engine
>
---
#### [replaced 143] TAIL: Text-Audio Incremental Learning
- **分类: cs.SD; cs.AI; cs.CV; eess.AS; I.2**

- **链接: [http://arxiv.org/pdf/2503.04258v2](http://arxiv.org/pdf/2503.04258v2)**

> **作者:** Yingfei Sun; Xu Gu; Wei Ji; Hanbin Zhao; Yifang Yin; Roger Zimmermann
>
> **备注:** 6 figures, 4 tables
>
> **摘要:** Many studies combine text and audio to capture multi-modal information but they overlook the model's generalization ability on new datasets. Introducing new datasets may affect the feature space of the original dataset, leading to catastrophic forgetting. Meanwhile, large model parameters can significantly impact training performance. To address these limitations, we introduce a novel task called Text-Audio Incremental Learning (TAIL) task for text-audio retrieval, and propose a new method, PTAT, Prompt Tuning for Audio-Text incremental learning. This method utilizes prompt tuning to optimize the model parameters while incorporating an audio-text similarity and feature distillation module to effectively mitigate catastrophic forgetting. We benchmark our method and previous incremental learning methods on AudioCaps, Clotho, BBC Sound Effects and Audioset datasets, and our method outperforms previous methods significantly, particularly demonstrating stronger resistance to forgetting on older datasets. Compared to the full-parameters Finetune (Sequential) method, our model only requires 2.42\% of its parameters, achieving 4.46\% higher performance.
>
---
#### [replaced 144] MaterialPicker: Multi-Modal DiT-Based Material Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03225v3](http://arxiv.org/pdf/2412.03225v3)**

> **作者:** Xiaohe Ma; Valentin Deschaintre; Miloš Hašan; Fujun Luan; Kun Zhou; Hongzhi Wu; Yiwei Hu
>
> **摘要:** High-quality material generation is key for virtual environment authoring and inverse rendering. We propose MaterialPicker, a multi-modal material generator leveraging a Diffusion Transformer (DiT) architecture, improving and simplifying the creation of high-quality materials from text prompts and/or photographs. Our method can generate a material based on an image crop of a material sample, even if the captured surface is distorted, viewed at an angle or partially occluded, as is often the case in photographs of natural scenes. We further allow the user to specify a text prompt to provide additional guidance for the generation. We finetune a pre-trained DiT-based video generator into a material generator, where each material map is treated as a frame in a video sequence. We evaluate our approach both quantitatively and qualitatively and show that it enables more diverse material generation and better distortion correction than previous work.
>
---
#### [replaced 145] MATE: Motion-Augmented Temporal Consistency for Event-based Point Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01300v2](http://arxiv.org/pdf/2412.01300v2)**

> **作者:** Han Han; Wei Zhai; Yang Cao; Bin Li; Zheng-jun Zha
>
> **摘要:** Tracking Any Point (TAP) plays a crucial role in motion analysis. Video-based approaches rely on iterative local matching for tracking, but they assume linear motion during the blind time between frames, which leads to point loss under large displacements or nonlinear motion. The high temporal resolution and motion blur-free characteristics of event cameras provide continuous, fine-grained motion information, capturing subtle variations with microsecond precision. This paper presents an event-based framework for tracking any point, which tackles the challenges posed by spatial sparsity and motion sensitivity in events through two tailored modules. Specifically, to resolve ambiguities caused by event sparsity, a motion-guidance module incorporates kinematic vectors into the local matching process. Additionally, a variable motion aware module is integrated to ensure temporally consistent responses that are insensitive to varying velocities, thereby enhancing matching precision. To validate the effectiveness of the approach, two event dataset for tracking any point is constructed by simulation. The method improves the $Survival_{50}$ metric by 17.9% over event-only tracking of any point baseline. Moreover, on standard feature tracking benchmarks, it outperforms all existing methods, even those that combine events and video frames.
>
---
#### [replaced 146] PARTE: Part-Guided Texturing for 3D Human Reconstruction from a Single Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17332v3](http://arxiv.org/pdf/2507.17332v3)**

> **作者:** Hyeongjin Nam; Donghwan Kim; Gyeongsik Moon; Kyoung Mu Lee
>
> **备注:** Published at ICCV 2025, 22 pages including the supplementary material
>
> **摘要:** The misaligned human texture across different human parts is one of the main limitations of existing 3D human reconstruction methods. Each human part, such as a jacket or pants, should maintain a distinct texture without blending into others. The structural coherence of human parts serves as a crucial cue to infer human textures in the invisible regions of a single image. However, most existing 3D human reconstruction methods do not explicitly exploit such part segmentation priors, leading to misaligned textures in their reconstructions. In this regard, we present PARTE, which utilizes 3D human part information as a key guide to reconstruct 3D human textures. Our framework comprises two core components. First, to infer 3D human part information from a single image, we propose a 3D part segmentation module (PartSegmenter) that initially reconstructs a textureless human surface and predicts human part labels based on the textureless surface. Second, to incorporate part information into texture reconstruction, we introduce a part-guided texturing module (PartTexturer), which acquires prior knowledge from a pre-trained image generation network on texture alignment of human parts. Extensive experiments demonstrate that our framework achieves state-of-the-art quality in 3D human reconstruction. The project page is available at https://hygenie1228.github.io/PARTE/.
>
---
#### [replaced 147] Uncertainty-Aware Testing-Time Optimization for 3D Human Pose Estimation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.02339v2](http://arxiv.org/pdf/2402.02339v2)**

> **作者:** Ti Wang; Mengyuan Liu; Hong Liu; Bin Ren; Yingxuan You; Wenhao Li; Nicu Sebe; Xia Li
>
> **备注:** Accepted by IEEE Transactions on Multimedia. Open sourced
>
> **摘要:** Although data-driven methods have achieved success in 3D human pose estimation, they often suffer from domain gaps and exhibit limited generalization. In contrast, optimization-based methods excel in fine-tuning for specific cases but are generally inferior to data-driven methods in overall performance. We observe that previous optimization-based methods commonly rely on a projection constraint, which only ensures alignment in 2D space, potentially leading to the overfitting problem. To address this, we propose an Uncertainty-Aware testing-time Optimization (UAO) framework, which keeps the prior information of the pre-trained model and alleviates the overfitting problem using the uncertainty of joints. Specifically, during the training phase, we design an effective 2D-to-3D network for estimating the corresponding 3D pose while quantifying the uncertainty of each 3D joint. For optimization during testing, the proposed optimization framework freezes the pre-trained model and optimizes only a latent state. Projection loss is then employed to ensure the generated poses are well aligned in 2D space for high-quality optimization. Furthermore, we utilize the uncertainty of each joint to determine how much each joint is allowed for optimization. The effectiveness and superiority of the proposed framework are validated through extensive experiments on challenging datasets: Human3.6M, MPI-INF-3DHP, and 3DPW. Notably, our approach outperforms the previous best result by a large margin of 5.5\% on Human3.6M. Code is available at \href{https://github.com/xiu-cs/UAO-Pose3D}{https://github.com/xiu-cs/UAO-Pose3D}.
>
---
#### [replaced 148] Qffusion: Controllable Portrait Video Editing via Quadrant-Grid Attention Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.06438v3](http://arxiv.org/pdf/2501.06438v3)**

> **作者:** Maomao Li; Lijian Lin; Yunfei Liu; Ye Zhu; Yu Li
>
> **备注:** 19 pages
>
> **摘要:** This paper presents Qffusion, a dual-frame-guided framework for portrait video editing. Specifically, we consider a design principle of ``animation for editing'', and train Qffusion as a general animation framework from two still reference images while we can use it for portrait video editing easily by applying modified start and end frames as references during inference. Leveraging the powerful generative power of Stable Diffusion, we propose a Quadrant-grid Arrangement (QGA) scheme for latent re-arrangement, which arranges the latent codes of two reference images and that of four facial conditions into a four-grid fashion, separately. Then, we fuse features of these two modalities and use self-attention for both appearance and temporal learning, where representations at different times are jointly modeled under QGA. Our Qffusion can achieve stable video editing without additional networks or complex training stages, where only the input format of Stable Diffusion is modified. Further, we propose a Quadrant-grid Propagation (QGP) inference strategy, which enjoys a unique advantage on stable arbitrary-length video generation by processing reference and condition frames recursively. Through extensive experiments, Qffusion consistently outperforms state-of-the-art techniques on portrait video editing. Project page: https://qffusion.github.io/page/.
>
---
#### [replaced 149] Deformable Convolution Module with Globally Learned Relative Offsets for Fundus Vessel Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18354v2](http://arxiv.org/pdf/2507.18354v2)**

> **作者:** Lexuan Zhu; Yuxuan Li; Yuning Ren
>
> **备注:** Added a graphical abstract and refined some wording
>
> **摘要:** Deformable convolution can adaptively change the shape of convolution kernel by learning offsets to deal with complex shape features. We propose a novel plug and play deformable convolutional module that uses attention and feedforward networks to learn offsets, so that the deformable patterns can capture long-distance global features. Compared with previously existing deformable convolutions, the proposed module learns the sub pixel displacement field and adaptively warps the feature maps across all channels rather than directly deforms the convolution kernel , which is equivalent to a relative deformation of the kernel sampling grids, achieving global feature deformation and the decoupling of kernel size and learning network. Considering that the fundus blood vessels have globally self similar complex edges, we design a deep learning model for fundus blood vessel segmentation, GDCUnet, based on the proposed convolutional module. Empirical evaluations under the same configuration and unified framework show that GDCUnet has achieved state of the art performance on public datasets. Further ablation experiments demonstrated that the proposed deformable convolutional module could more significantly learn the complex features of fundus blood vessels, enhancing the model representation and generalization capabilities. The proposed module is similar to the interface of conventional convolution, we suggest applying it to more machine vision tasks with complex global self similar features.
>
---
#### [replaced 150] Aether: Geometric-Aware Unified World Modeling
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.18945v3](http://arxiv.org/pdf/2503.18945v3)**

> **作者:** Aether Team; Haoyi Zhu; Yifan Wang; Jianjun Zhou; Wenzheng Chang; Yang Zhou; Zizun Li; Junyi Chen; Chunhua Shen; Jiangmiao Pang; Tong He
>
> **备注:** Project Page: https://aether-world.github.io/
>
> **摘要:** The integration of geometric reconstruction and generative modeling remains a critical challenge in developing AI systems capable of human-like spatial reasoning. This paper proposes Aether, a unified framework that enables geometry-aware reasoning in world models by jointly optimizing three core capabilities: (1) 4D dynamic reconstruction, (2) action-conditioned video prediction, and (3) goal-conditioned visual planning. Through task-interleaved feature learning, Aether achieves synergistic knowledge sharing across reconstruction, prediction, and planning objectives. Building upon video generation models, our framework demonstrates zero-shot synthetic-to-real generalization despite never observing real-world data during training. Furthermore, our approach achieves zero-shot generalization in both action following and reconstruction tasks, thanks to its intrinsic geometric modeling. Notably, even without real-world data, its reconstruction performance is comparable with or even better than that of domain-specific models. Additionally, Aether employs camera trajectories as geometry-informed action spaces, enabling effective action-conditioned prediction and visual planning. We hope our work inspires the community to explore new frontiers in physically-reasonable world modeling and its applications.
>
---
#### [replaced 151] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection
- **分类: cs.CV; I.4.0; I.4.9**

- **链接: [http://arxiv.org/pdf/2507.07994v3](http://arxiv.org/pdf/2507.07994v3)**

> **作者:** Subhajit Maity; Ayan Kumar Bhunia; Subhadeep Koley; Pinaki Nath Chowdhury; Aneeshan Sain; Yi-Zhe Song
>
> **备注:** Accepted at ICCV 2025. Project Page: https://subhajitmaity.me/DYKp
>
> **摘要:** Keypoint detection, integral to modern machine perception, faces challenges in few-shot learning, particularly when source data from the same distribution as the query is unavailable. This gap is addressed by leveraging sketches, a popular form of human expression, providing a source-free alternative. However, challenges arise in mastering cross-modal embeddings and handling user-specific sketch styles. Our proposed framework overcomes these hurdles with a prototypical setup, combined with a grid-based locator and prototypical domain adaptation. We also demonstrate success in few-shot convergence across novel keypoints and classes through extensive experiments.
>
---
#### [replaced 152] End-to-End Fine-Tuning of 3D Texture Generation using Differentiable Rewards
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18331v2](http://arxiv.org/pdf/2506.18331v2)**

> **作者:** AmirHossein Zamani; Tianhao Xie; Amir G. Aghdam; Tiberiu Popa; Eugene Belilovsky
>
> **摘要:** While recent 3D generative models can produce high-quality texture images, they often fail to capture human preferences or meet task-specific requirements. Moreover, a core challenge in the 3D texture generation domain is that most existing approaches rely on repeated calls to 2D text-to-image generative models, which lack an inherent understanding of the 3D structure of the input 3D mesh object. To alleviate these issues, we propose an end-to-end differentiable, reinforcement-learning-free framework that embeds human feedback, expressed as differentiable reward functions, directly into the 3D texture synthesis pipeline. By back-propagating preference signals through both geometric and appearance modules of the proposed framework, our method generates textures that respect the 3D geometry structure and align with desired criteria. To demonstrate its versatility, we introduce three novel geometry-aware reward functions, which offer a more controllable and interpretable pathway for creating high-quality 3D content from natural language. By conducting qualitative, quantitative, and user-preference evaluations against state-of-the-art methods, we demonstrate that our proposed strategy consistently outperforms existing approaches. We will make our implementation code publicly available upon acceptance of the paper.
>
---
#### [replaced 153] GUI-G$^2$: Gaussian Reward Modeling for GUI Grounding
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.15846v3](http://arxiv.org/pdf/2507.15846v3)**

> **作者:** Fei Tang; Zhangxuan Gu; Zhengxi Lu; Xuyang Liu; Shuheng Shen; Changhua Meng; Wen Wang; Wenqi Zhang; Yongliang Shen; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **摘要:** Graphical User Interface (GUI) grounding maps natural language instructions to precise interface locations for autonomous interaction. Current reinforcement learning approaches use binary rewards that treat elements as hit-or-miss targets, creating sparse signals that ignore the continuous nature of spatial interactions. Motivated by human clicking behavior that naturally forms Gaussian distributions centered on target elements, we introduce GUI Gaussian Grounding Rewards (GUI-G$^2$), a principled reward framework that models GUI elements as continuous Gaussian distributions across the interface plane. GUI-G$^2$ incorporates two synergistic mechanisms: Gaussian point rewards model precise localization through exponentially decaying distributions centered on element centroids, while coverage rewards assess spatial alignment by measuring the overlap between predicted Gaussian distributions and target regions. To handle diverse element scales, we develop an adaptive variance mechanism that calibrates reward distributions based on element dimensions. This framework transforms GUI grounding from sparse binary classification to dense continuous optimization, where Gaussian distributions generate rich gradient signals that guide models toward optimal interaction positions. Extensive experiments across ScreenSpot, ScreenSpot-v2, and ScreenSpot-Pro benchmarks demonstrate that GUI-G$^2$, substantially outperforms state-of-the-art method UI-TARS-72B, with the most significant improvement of 24.7% on ScreenSpot-Pro. Our analysis reveals that continuous modeling provides superior robustness to interface variations and enhanced generalization to unseen layouts, establishing a new paradigm for spatial reasoning in GUI interaction tasks.
>
---
#### [replaced 154] Edge-guided Low-light Image Enhancement with Inertial Bregman Alternating Linearized Minimization
- **分类: cs.CV; cs.NA; math.NA; 65K05, 65K15, 94A08**

- **链接: [http://arxiv.org/pdf/2403.01142v2](http://arxiv.org/pdf/2403.01142v2)**

> **作者:** Chaoyan Huang; Zhongming Wu; Tieyong Zeng
>
> **备注:** 16 pages
>
> **摘要:** Prior-based methods for low-light image enhancement often face challenges in extracting available prior information from dim images. To overcome this limitation, we introduce a simple yet effective Retinex model with the proposed edge extraction prior. More specifically, we design an edge extraction network to capture the fine edge features from the low-light image directly. Building upon the Retinex theory, we decompose the low-light image into its illumination and reflectance components and introduce an edge-guided Retinex model for enhancing low-light images. To solve the proposed model, we propose a novel inertial Bregman alternating linearized minimization algorithm. This algorithm addresses the optimization problem associated with the edge-guided Retinex model, enabling effective enhancement of low-light images. Through rigorous theoretical analysis, we establish the convergence properties of the algorithm. Besides, we prove that the proposed algorithm converges to a stationary point of the problem through nonconvex optimization theory. Furthermore, extensive experiments are conducted on multiple real-world low-light image datasets to demonstrate the efficiency and superiority of the proposed scheme.
>
---
#### [replaced 155] Back Home: A Computer Vision Solution to Seashell Identification for Ecological Restoration
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.04873v3](http://arxiv.org/pdf/2501.04873v3)**

> **作者:** Alexander Valverde; Luis Solano; André Montoya
>
> **备注:** ICCV 2025 (CV4E Workshop)
>
> **摘要:** Illegal souvenir collection strips an estimated five tonnes of seashells from Costa Rica's beaches each year. Yet, once these specimens are seized, their coastal origin -- Pacific or Caribbean -- cannot be verified easily due to the lack of information, preventing their return when confiscated by local authorities. To solve this issue, we introduce BackHome19K, the first large-scale image corpus (19{,}058 photographs, 516 species) annotated with coast-level labels, and propose a lightweight pipeline that infers provenance in real time on a mobile-grade CPU. A trained anomaly filter pre-screens uploads, increasing robustness to user-generated noise. On a held-out test set, the classifier attains 86.3\% balanced accuracy, while the filter rejects 93\% of 180 out-of-domain objects with zero false negatives. Deployed as a web application, the system has already processed 70{,}000 shells for wildlife officers in under three seconds per image, enabling confiscated specimens to be safely repatriated to their native ecosystems. The dataset is available at https://huggingface.co/datasets/FIFCO/BackHome19K
>
---
#### [replaced 156] VGS-ATD: Robust Distributed Learning for Multi-Label Medical Image Classification Under Heterogeneous and Imbalanced Conditions
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2507.18657v2](http://arxiv.org/pdf/2507.18657v2)**

> **作者:** Zehui Zhao; Laith Alzubaidi; Haider A. Alwzwazy; Jinglan Zhang; Yuantong Gu
>
> **备注:** The idea is still underdeveloped, not yet enough to be published
>
> **摘要:** In recent years, advanced deep learning architectures have shown strong performance in medical imaging tasks. However, the traditional centralized learning paradigm poses serious privacy risks as all data is collected and trained on a single server. To mitigate this challenge, decentralized approaches such as federated learning and swarm learning have emerged, allowing model training on local nodes while sharing only model weights. While these methods enhance privacy, they struggle with heterogeneous and imbalanced data and suffer from inefficiencies due to frequent communication and the aggregation of weights. More critically, the dynamic and complex nature of clinical environments demands scalable AI systems capable of continuously learning from diverse modalities and multilabels. Yet, both centralized and decentralized models are prone to catastrophic forgetting during system expansion, often requiring full model retraining to incorporate new data. To address these limitations, we propose VGS-ATD, a novel distributed learning framework. To validate VGS-ATD, we evaluate it in experiments spanning 30 datasets and 80 independent labels across distributed nodes, VGS-ATD achieved an overall accuracy of 92.7%, outperforming centralized learning (84.9%) and swarm learning (72.99%), while federated learning failed under these conditions due to high requirements on computational resources. VGS-ATD also demonstrated strong scalability, with only a 1% drop in accuracy on existing nodes after expansion, compared to a 20% drop in centralized learning, highlighting its resilience to catastrophic forgetting. Additionally, it reduced computational costs by up to 50% relative to both centralized and swarm learning, confirming its superior efficiency and scalability.
>
---
#### [replaced 157] Spatiotemporal Multi-Camera Calibration using Freely Moving People
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12546v3](http://arxiv.org/pdf/2502.12546v3)**

> **作者:** Sang-Eun Lee; Ko Nishino; Shohei Nobuhara
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** We propose a novel method for spatiotemporal multi-camera calibration using freely moving people in multiview videos. Since calibrating multiple cameras and finding matches across their views are inherently interdependent, performing both in a unified framework poses a significant challenge. We address these issues as a single registration problem of matching two sets of 3D points, leveraging human motion in dynamic multi-person scenes. To this end, we utilize 3D human poses obtained from an off-the-shelf monocular 3D human pose estimator and transform them into 3D points on a unit sphere, to solve the rotation, time offset, and the association alternatingly. We employ a probabilistic approach that can jointly solve both problems of aligning spatiotemporal data and establishing correspondences through soft assignment between two views. The translation is determined by applying coplanarity constraints. The pairwise registration results are integrated into a multiview setup, and then a nonlinear optimization method is used to improve the accuracy of the camera poses, temporal offsets, and multi-person associations. Extensive experiments on synthetic and real data demonstrate the effectiveness and flexibility of the proposed method as a practical marker-free calibration tool.
>
---
#### [replaced 158] A Survey of Deep Learning for Geometry Problem Solving
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11936v4](http://arxiv.org/pdf/2507.11936v4)**

> **作者:** Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in progress
>
> **摘要:** Geometry problem solving is a key area of mathematical reasoning, which is widely involved in many important fields such as education, mathematical ability assessment of artificial intelligence, and multimodal ability assessment. In recent years, the rapid development of deep learning technology, especially the rise of multimodal large language models, has triggered a widespread research boom. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our goal is to provide a comprehensive and practical reference of deep learning for geometry problem solving to promote further developments in this field. We create a continuously updated list of papers on GitHub: https://github.com/majianz/dl4gps.
>
---
#### [replaced 159] MIGE: Mutually Enhanced Multimodal Instruction-Based Image Generation and Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.21291v4](http://arxiv.org/pdf/2502.21291v4)**

> **作者:** Xueyun Tian; Wei Li; Bingbing Xu; Yige Yuan; Yuanzhuo Wang; Huawei Shen
>
> **备注:** This paper have been accepted by ACM MM25
>
> **摘要:** Despite significant progress in diffusion-based image generation, subject-driven generation and instruction-based editing remain challenging. Existing methods typically treat them separately, struggling with limited high-quality data and poor generalization. However, both tasks require capturing complex visual variations while maintaining consistency between inputs and outputs. Inspired by this, we propose MIGE, a unified framework that standardizes task representations using multimodal instructions. It first treats subject-driven generation as creation on a blank canvas and instruction-based editing as modification of an existing image, establishing a shared input-output formulation, then introduces a novel multimodal encoder that maps free-form multimodal instructions into a unified vision-language space, integrating visual and semantic features through a feature fusion mechanism. This unification enables joint training of both tasks, providing two key advantages: (1) Cross-Task Enhancement: by leveraging shared visual and semantic representations, joint training improves instruction adherence and visual consistency in both subject-driven generation and instruction-based editing. (2) Generalization: learning in a unified format facilitates cross-task knowledge transfer, enabling MIGE to generalize to novel compositional tasks, including instruction-based subject-driven editing. Experiments show that MIGE excels in both subject-driven generation and instruction-based editing while setting a SOTA in the new task of instruction-based subject-driven editing. Code and model have been publicly available at https://github.com/Eureka-Maggie/MIGE.
>
---
#### [replaced 160] Learning from Heterogeneity: Generalizing Dynamic Facial Expression Recognition via Distributionally Robust Optimization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.15765v2](http://arxiv.org/pdf/2507.15765v2)**

> **作者:** Feng-Qi Cui; Anyang Tong; Jinyang Huang; Jie Zhang; Dan Guo; Zhi Liu; Meng Wang
>
> **备注:** Accepted by ACM MM'25
>
> **摘要:** Dynamic Facial Expression Recognition (DFER) plays a critical role in affective computing and human-computer interaction. Although existing methods achieve comparable performance, they inevitably suffer from performance degradation under sample heterogeneity caused by multi-source data and individual expression variability. To address these challenges, we propose a novel framework, called Heterogeneity-aware Distributional Framework (HDF), and design two plug-and-play modules to enhance time-frequency modeling and mitigate optimization imbalance caused by hard samples. Specifically, the Time-Frequency Distributional Attention Module (DAM) captures both temporal consistency and frequency robustness through a dual-branch attention design, improving tolerance to sequence inconsistency and visual style shifts. Then, based on gradient sensitivity and information bottleneck principles, an adaptive optimization module Distribution-aware Scaling Module (DSM) is introduced to dynamically balance classification and contrastive losses, enabling more stable and discriminative representation learning. Extensive experiments on two widely used datasets, DFEW and FERV39k, demonstrate that HDF significantly improves both recognition accuracy and robustness. Our method achieves superior weighted average recall (WAR) and unweighted average recall (UAR) while maintaining strong generalization across diverse and imbalanced scenarios. Codes are released at https://github.com/QIcita/HDF_DFER.
>
---
#### [replaced 161] Compressed Image Generation with Denoising Diffusion Codebook Models
- **分类: eess.IV; cs.AI; cs.CV; cs.IT; eess.SP; math.IT**

- **链接: [http://arxiv.org/pdf/2502.01189v4](http://arxiv.org/pdf/2502.01189v4)**

> **作者:** Guy Ohayon; Hila Manor; Tomer Michaeli; Michael Elad
>
> **备注:** Published in the International Conference on Machine Learning (ICML) 2025. Code and demo are available at https://ddcm-2025.github.io/
>
> **摘要:** We present a novel generative approach based on Denoising Diffusion Models (DDMs), which produces high-quality image samples along with their losslessly compressed bit-stream representations. This is obtained by replacing the standard Gaussian noise sampling in the reverse diffusion with a selection of noise samples from pre-defined codebooks of fixed iid Gaussian vectors. Surprisingly, we find that our method, termed Denoising Diffusion Codebook Model (DDCM), retains sample quality and diversity of standard DDMs, even for extremely small codebooks. We leverage DDCM and pick the noises from the codebooks that best match a given image, converting our generative model into a highly effective lossy image codec achieving state-of-the-art perceptual image compression results. More generally, by setting other noise selections rules, we extend our compression method to any conditional image generation task (e.g., image restoration), where the generated images are produced jointly with their condensed bit-stream representations. Our work is accompanied by a mathematical interpretation of the proposed compressed conditional generation schemes, establishing a connection with score-based approximations of posterior samplers for the tasks considered.
>
---
#### [replaced 162] Grid-LOGAT: Grid Based Local and Global Area Transcription for Video Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24371v3](http://arxiv.org/pdf/2505.24371v3)**

> **作者:** Md Intisar Chowdhury; Kittinun Aukkapinyo; Hiroshi Fujimura; Joo Ann Woo; Wasu Wasusatein; Fadoua Ghourabi
>
> **备注:** Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** In this paper, we propose a Grid-based Local and Global Area Transcription (Grid-LoGAT) system for Video Question Answering (VideoQA). The system operates in two phases. First, extracting text transcripts from video frames using a Vision-Language Model (VLM). Next, processing questions using these transcripts to generate answers through a Large Language Model (LLM). This design ensures image privacy by deploying the VLM on edge devices and the LLM in the cloud. To improve transcript quality, we propose grid-based visual prompting, which extracts intricate local details from each grid cell and integrates them with global information. Evaluation results show that Grid-LoGAT, using the open-source VLM (LLaVA-1.6-7B) and LLM (Llama-3.1-8B), outperforms state-of-the-art methods with similar baseline models on NExT-QA and STAR-QA datasets with an accuracy of 65.9% and 50.11% respectively. Additionally, our method surpasses the non-grid version by 24 points on localization-based questions we created using NExT-QA. (This paper is accepted by IEEE ICIP 2025.)
>
---
#### [replaced 163] Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.03222v4](http://arxiv.org/pdf/2503.03222v4)**

> **作者:** Zhumei Wang; Zechen Hu; Ruoxi Guo; Huaijin Pi; Ziyong Feng; Sida Peng; Xiaowei Zhou; Mingtao Pei; Siyuan Huang
>
> **摘要:** Recovering absolute human motion from monocular inputs is challenging due to two main issues. First, existing methods depend on 3D training data collected from limited environments, constraining out-of-distribution generalization. The second issue is the difficulty of estimating metric-scale poses from monocular input. To address these challenges, we introduce Mocap-2-to-3, a novel framework that performs multi-view lifting from monocular input by leveraging 2D data pre-training, enabling the reconstruction of metrically accurate 3D motions with absolute positions. To leverage abundant 2D data, we decompose complex 3D motion into multi-view syntheses. We first pretrain a single-view diffusion model on extensive 2D datasets, then fine-tune a multi-view model using public 3D data to enable view-consistent motion generation from monocular input, allowing the model to acquire action priors and diversity through 2D data. Furthermore, to recover absolute poses, we propose a novel human motion representation that decouples the learning of local pose and global movements, while encoding geometric priors of the ground to accelerate convergence. This enables progressive recovery of motion in absolute space during inference. Experimental results on in-the-wild benchmarks demonstrate that our method surpasses state-of-the-art approaches in both camera-space motion realism and world-grounded human positioning, while exhibiting superior generalization capability. Our code will be made publicly available.
>
---
#### [replaced 164] Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.04469v2](http://arxiv.org/pdf/2502.04469v2)**

> **作者:** Imad Eddine Marouf; Enzo Tartaglione; Stephane Lathuiliere; Joost van de Weijer
>
> **备注:** ICCV 2025, 8 pages. Code: https://github.com/IemProg/QUAD
>
> **摘要:** Continual Learning in Visual Question Answering (VQACL) requires models to acquire new visual-linguistic skills (plasticity) while preserving previously learned knowledge (stability). The inherent multimodality of VQACL exacerbates this challenge, as models must balance stability across visual and textual domains while adapting to novel objects and reasoning tasks. Existing methods, primarily designed for unimodal settings, often fall short in addressing this dual requirement. In this work, we present QUestion-only replay with Attention Distillation (QUAD), a novel approach for VQACL that leverages only past task questions for regularization. By eliminating the need to store visual data, QUAD not only reduces memory overhead, but also alleviates privacy concerns. Our method introduces a Question-only Replay mechanism that selectively reuses prior task questions to counteract overfitting to the answer space of the current task, addressing the problem out of answer set. Complementing this, we propose Attention Consistency Distillation to enforce both intra-modal and inter-modal attention consistency across tasks, preserving essential visual-linguistic associations. Extensive experiments on VQAv2 and NExT-QA demonstrate that QUAD significantly outperforms state-of-the-art methods, achieving robust performance in continual VQA. Code is available at: https://github.com/IemProg/QUAD.
>
---
#### [replaced 165] KITTEN: A Knowledge-Intensive Evaluation of Image Generation on Visual Entities
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.11824v2](http://arxiv.org/pdf/2410.11824v2)**

> **作者:** Hsin-Ping Huang; Xinyi Wang; Yonatan Bitton; Hagai Taitelbaum; Gaurav Singh Tomar; Ming-Wei Chang; Xuhui Jia; Kelvin C. K. Chan; Hexiang Hu; Yu-Chuan Su; Ming-Hsuan Yang
>
> **备注:** Project page: https://kitten-project.github.io/
>
> **摘要:** Recent advances in text-to-image generation have improved the quality of synthesized images, but evaluations mainly focus on aesthetics or alignment with text prompts. Thus, it remains unclear whether these models can accurately represent a wide variety of realistic visual entities. To bridge this gap, we propose KITTEN, a benchmark for Knowledge-InTensive image generaTion on real-world ENtities. Using KITTEN, we conduct a systematic study of the latest text-to-image models and retrieval-augmented models, focusing on their ability to generate real-world visual entities, such as landmarks and animals. Analysis using carefully designed human evaluations, automatic metrics, and MLLM evaluations show that even advanced text-to-image models fail to generate accurate visual details of entities. While retrieval-augmented models improve entity fidelity by incorporating reference images, they tend to over-rely on them and struggle to create novel configurations of the entity in creative text prompts.
>
---
#### [replaced 166] Survey on Hand Gesture Recognition from Visual Input
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.11992v2](http://arxiv.org/pdf/2501.11992v2)**

> **作者:** Manousos Linardakis; Iraklis Varlamis; Georgios Th. Papadopoulos
>
> **备注:** 37 pages
>
> **摘要:** Hand gesture recognition has become an important research area, driven by the growing demand for human-computer interaction in fields such as sign language recognition, virtual and augmented reality, and robotics. Despite the rapid growth of the field, there are few surveys that comprehensively cover recent research developments, available solutions, and benchmark datasets. This survey addresses this gap by examining the latest advancements in hand gesture and 3D hand pose recognition from various types of camera input data including RGB images, depth images, and videos from monocular or multiview cameras, examining the differing methodological requirements of each approach. Furthermore, an overview of widely used datasets is provided, detailing their main characteristics and application domains. Finally, open challenges such as achieving robust recognition in real-world environments, handling occlusions, ensuring generalization across diverse users, and addressing computational efficiency for real-time applications are highlighted to guide future research directions. By synthesizing the objectives, methodologies, and applications of recent studies, this survey offers valuable insights into current trends, challenges, and opportunities for future research in human hand gesture recognition.
>
---
#### [replaced 167] Evaluating Self-Supervised Learning in Medical Imaging: A Benchmark for Robustness, Generalizability, and Multi-Domain Impact
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.19124v2](http://arxiv.org/pdf/2412.19124v2)**

> **作者:** Valay Bundele; Karahan Sarıtaş; Bora Kargi; Oğuz Ata Çal; Kıvanç Tezören; Zohreh Ghaderi; Hendrik Lensch
>
> **摘要:** Self-supervised learning (SSL) has emerged as a promising paradigm in medical imaging, addressing the chronic challenge of limited labeled data in healthcare settings. While SSL has shown impressive results, existing studies in the medical domain are often limited in scope, focusing on specific datasets or modalities, or evaluating only isolated aspects of model performance. This fragmented evaluation approach poses a significant challenge, as models deployed in critical medical settings must not only achieve high accuracy but also demonstrate robust performance and generalizability across diverse datasets and varying conditions. To address this gap, we present a comprehensive evaluation of SSL methods within the medical domain, with a particular focus on robustness and generalizability. Using the MedMNIST dataset collection as a standardized benchmark, we evaluate 8 major SSL methods across 11 different medical datasets. Our study provides an in-depth analysis of model performance in both in-domain scenarios and the detection of out-of-distribution (OOD) samples, while exploring the effect of various initialization strategies, model architectures, and multi-domain pre-training. We further assess the generalizability of SSL methods through cross-dataset evaluations and the in-domain performance with varying label proportions (1%, 10%, and 100%) to simulate real-world scenarios with limited supervision. We hope this comprehensive benchmark helps practitioners and researchers make more informed decisions when applying SSL methods to medical applications.
>
---
#### [replaced 168] A Unified Image-Dense Annotation Generation Model for Underwater Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21771v2](http://arxiv.org/pdf/2503.21771v2)**

> **作者:** Hongkai Lin; Dingkang Liang; Zhenghao Qi; Xiang Bai
>
> **备注:** Accepted by CVPR 2025. The code is available at https://github.com/HongkLin/TIDE
>
> **摘要:** Underwater dense prediction, especially depth estimation and semantic segmentation, is crucial for gaining a comprehensive understanding of underwater scenes. Nevertheless, high-quality and large-scale underwater datasets with dense annotations remain scarce because of the complex environment and the exorbitant data collection costs. This paper proposes a unified Text-to-Image and DEnse annotation generation method (TIDE) for underwater scenes. It relies solely on text as input to simultaneously generate realistic underwater images and multiple highly consistent dense annotations. Specifically, we unify the generation of text-to-image and text-to-dense annotations within a single model. The Implicit Layout Sharing mechanism (ILS) and cross-modal interaction method called Time Adaptive Normalization (TAN) are introduced to jointly optimize the consistency between image and dense annotations. We synthesize a large-scale underwater dataset using TIDE to validate the effectiveness of our method in underwater dense prediction tasks. The results demonstrate that our method effectively improves the performance of existing underwater dense prediction models and mitigates the scarcity of underwater data with dense annotations. We hope our method can offer new perspectives on alleviating data scarcity issues in other fields. The code is available at https://github.com/HongkLin/TIDE
>
---
#### [replaced 169] A Lightweight Face Quality Assessment Framework to Improve Face Verification Performance in Real-Time Screening Applications
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.15961v2](http://arxiv.org/pdf/2507.15961v2)**

> **作者:** Ahmed Aman Ibrahim; Hamad Mansour Alawar; Abdulnasser Abbas Zehi; Ahmed Mohammad Alkendi; Bilal Shafi Ashfaq Ahmed Mirza; Shan Ullah; Ismail Lujain Jaleel; Hassan Ugail
>
> **摘要:** Face image quality plays a critical role in determining the accuracy and reliability of face verification systems, particularly in real-time screening applications such as surveillance, identity verification, and access control. Low-quality face images, often caused by factors such as motion blur, poor lighting conditions, occlusions, and extreme pose variations, significantly degrade the performance of face recognition models, leading to higher false rejection and false acceptance rates. In this work, we propose a lightweight yet effective framework for automatic face quality assessment, which aims to pre-filter low-quality face images before they are passed to the verification pipeline. Our approach utilises normalised facial landmarks in conjunction with a Random Forest Regression classifier to assess image quality, achieving an accuracy of 96.67%. By integrating this quality assessment module into the face verification process, we observe a substantial improvement in performance, including a comfortable 99.7% reduction in the false rejection rate and enhanced cosine similarity scores when paired with the ArcFace face verification model. To validate our approach, we have conducted experiments on a real-world dataset collected comprising over 600 subjects captured from CCTV footage in unconstrained environments within Dubai Police. Our results demonstrate that the proposed framework effectively mitigates the impact of poor-quality face images, outperforming existing face quality assessment techniques while maintaining computational efficiency. Moreover, the framework specifically addresses two critical challenges in real-time screening: variations in face resolution and pose deviations, both of which are prevalent in practical surveillance scenarios.
>
---
#### [replaced 170] Traveling Across Languages: Benchmarking Cross-Lingual Consistency in Multimodal LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15075v4](http://arxiv.org/pdf/2505.15075v4)**

> **作者:** Hao Wang; Pinzhi Huang; Jihan Yang; Saining Xie; Daisuke Kawahara
>
> **备注:** https://github.com/nlp-waseda/traveling-across-languages
>
> **摘要:** The rapid evolution of multimodal large language models (MLLMs) has significantly enhanced their real-world applications. However, achieving consistent performance across languages, especially when integrating cultural knowledge, remains a significant challenge. To better assess this issue, we introduce two new benchmarks: KnowRecall and VisRecall, which evaluate cross-lingual consistency in MLLMs. KnowRecall is a visual question answering benchmark designed to measure factual knowledge consistency in 15 languages, focusing on cultural and historical questions about global landmarks. VisRecall assesses visual memory consistency by asking models to describe landmark appearances in 9 languages without access to images. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, still struggle to achieve cross-lingual consistency. This underscores the need for more robust approaches that produce truly multilingual and culturally aware models.
>
---
