# 计算机视觉 cs.CV

- **最新发布 218 篇**

- **更新 138 篇**

## 最新发布

#### [new 001] SAMSOD: Rethinking SAM Optimization for RGB-T Salient Object Detection
- **分类: cs.CV**

- **简介: 该论文属于RGB-T显著目标检测任务，旨在解决模态不平衡和梯度冲突问题。作者提出SAMSOD模型，通过单模态监督增强非主导模态学习，引入梯度解耦策略，并设计两个解耦适配器分别处理高、低激活神经元，以提升检测效果。**

- **链接: [http://arxiv.org/pdf/2510.03689v1](http://arxiv.org/pdf/2510.03689v1)**

> **作者:** Zhengyi Liu; Xinrui Wang; Xianyong Fang; Zhengzheng Tu; Linbo Wang
>
> **备注:** Accepted by TMM
>
> **摘要:** RGB-T salient object detection (SOD) aims to segment attractive objects by combining RGB and thermal infrared images. To enhance performance, the Segment Anything Model has been fine-tuned for this task. However, the imbalance convergence of two modalities and significant gradient difference between high- and low- activations are ignored, thereby leaving room for further performance enhancement. In this paper, we propose a model called \textit{SAMSOD}, which utilizes unimodal supervision to enhance the learning of non-dominant modality and employs gradient deconfliction to reduce the impact of conflicting gradients on model convergence. The method also leverages two decoupled adapters to separately mask high- and low-activation neurons, emphasizing foreground objects by enhancing background learning. Fundamental experiments on RGB-T SOD benchmark datasets and generalizability experiments on scribble supervised RGB-T SOD, fully supervised RGB-D SOD datasets and full-supervised RGB-D rail surface defect detection all demonstrate the effectiveness of our proposed method.
>
---
#### [new 002] Exploring the Hierarchical Reasoning Model for Small Natural-Image Classification Without Augmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究图像分类任务，探讨在无数据增强的小型自然图像上，基于Transformer的层级推理模型（HRM）能否作为实用分类器。论文比较了HRM与简单卷积网络的性能，发现HRM在MNIST表现良好，但在CIFAR数据集上过拟合严重、泛化能力差，训练效率也较低。结论指出当前HRM缺乏足够的图像先验知识，但不排除改进潜力。**

- **链接: [http://arxiv.org/pdf/2510.03598v1](http://arxiv.org/pdf/2510.03598v1)**

> **作者:** Alexander V. Mantzaris
>
> **摘要:** This paper asks whether the Hierarchical Reasoning Model (HRM) with the two Transformer-style modules $(f_L,f_H)$, one step (DEQ-style) training, deep supervision, Rotary Position Embeddings, and RMSNorm can serve as a practical image classifier. It is evaluated on MNIST, CIFAR-10, and CIFAR-100 under a deliberately raw regime: no data augmentation, identical optimizer family with one-epoch warmup then cosine-floor decay, and label smoothing. HRM optimizes stably and performs well on MNIST ($\approx 98\%$ test accuracy), but on small natural images it overfits and generalizes poorly: on CIFAR-10, HRM reaches 65.0\% after 25 epochs, whereas a two-stage Conv--BN--ReLU baseline attains 77.2\% while training $\sim 30\times$ faster per epoch; on CIFAR-100, HRM achieves only 29.7\% test accuracy despite 91.5\% train accuracy, while the same CNN reaches 45.3\% test with 50.5\% train accuracy. Loss traces and error analyses indicate healthy optimization but insufficient image-specific inductive bias for HRM in this regime. It is concluded that, for small-resolution image classification without augmentation, HRM is not competitive with even simple convolutional architectures as the HRM currently exist but this does not exclude possibilities that modifications to the model may allow it to improve greatly.
>
---
#### [new 003] Efficiency vs. Efficacy: Assessing the Compression Ratio-Dice Score Relationship through a Simple Benchmarking Framework for Cerebrovascular 3D Segmentation
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于医学图像处理任务，旨在解决3D医学影像数据过大影响协作的问题。研究使用ZFP压缩技术，在不影响脑血管分割性能的前提下，实现高效数据压缩，验证了其在保持高分割精度（Dice系数约0.87656）的同时显著减少数据量（最高压缩比22.89:1），从而促进大规模医学数据的共享与研究。**

- **链接: [http://arxiv.org/pdf/2510.03769v1](http://arxiv.org/pdf/2510.03769v1)**

> **作者:** Shimaa Elbana; Ahmad Kamal; Shahd Ahmed Ali; Ahmad Al-Kabbany
>
> **摘要:** The increasing size and complexity of medical imaging datasets, particularly in 3D formats, present significant barriers to collaborative research and transferability. This study investigates whether the ZFP compression technique can mitigate these challenges without compromising the performance of automated cerebrovascular segmentation, a critical first step in intracranial aneurysm detection. We apply ZFP in both its error tolerance and fixed-rate modes to a large scale, and one of the most recent, datasets in the literature, 3D medical dataset containing ground-truth vascular segmentations. The segmentation quality on the compressed volumes is rigorously compared to the uncompressed baseline (Dice approximately equals 0.8774). Our findings reveal that ZFP can achieve substantial data reduction--up to a 22.89:1 ratio in error tolerance mode--while maintaining a high degree of fidelity, with the mean Dice coefficient remaining high at 0.87656. These results demonstrate that ZFP is a viable and powerful tool for enabling more efficient and accessible research on large-scale medical datasets, fostering broader collaboration across the community.
>
---
#### [new 004] Domain-Robust Marine Plastic Detection Using Vision Models
- **分类: cs.CV**

- **简介: 该论文属于图像检测任务，旨在解决跨域水下塑料检测中的模型泛化问题。作者评估了多种视觉模型（包括CNN和Vision Transformer）在不同数据源间的鲁棒性，比较了其性能差异，并分析了错误案例。结果表明，轻量级CNN模型（如MobileNetV2）在跨域检测中表现最佳，而大模型（如CLIP、Gemini）具有互补优势。**

- **链接: [http://arxiv.org/pdf/2510.03294v1](http://arxiv.org/pdf/2510.03294v1)**

> **作者:** Saanvi Kataria
>
> **备注:** 16 pages, 5 figures, 1 table
>
> **摘要:** Marine plastic pollution is a pressing environmental threat, making reliable automation for underwater debris detection essential. However, vision systems trained on one dataset often degrade on new imagery due to domain shift. This study benchmarks models for cross-domain robustness, training convolutional neural networks - CNNs (MobileNetV2, ResNet-18, EfficientNet-B0) and vision transformers (DeiT-Tiny, ViT-B16) on a labeled underwater dataset and then evaluates them on a balanced cross-domain test set built from plastic-positive images drawn from a different source and negatives from the training domain. Two zero-shot models were assessed, CLIP ViT-L14 and Google's Gemini 2.0 Flash, that leverage pretraining to classify images without fine-tuning. Results show the lightweight MobileNetV2 delivers the strongest cross-domain performance (F1 0.97), surpassing larger models. All fine-tuned models achieved high Precision (around 99%), but differ in Recall, indicating varying sensitivity to plastic instances. Zero-shot CLIP is comparatively sensitive (Recall around 80%) yet prone to false positives (Precision around 56%), whereas Gemini exhibits the inverse profile (Precision around 99%, Recall around 81%). Error analysis highlights recurring confusions with coral textures, suspended particulates, and specular glare. Overall, compact CNNs with supervised training can generalize effectively for cross-domain underwater detection, while large pretrained vision-language models provide complementary strengths.
>
---
#### [new 005] From Scope to Script: An Automated Report Generation Model for Gastrointestinal Endoscopy
- **分类: cs.CV**

- **简介: 该论文旨在解决胃肠镜检查报告撰写效率低的问题。通过构建基于Transformer的视觉编码器与文本解码器模型，分两阶段训练，实现自动生成临床报告。属于医学图像理解与自然语言生成的交叉任务。**

- **链接: [http://arxiv.org/pdf/2510.03543v1](http://arxiv.org/pdf/2510.03543v1)**

> **作者:** Evandros Kaklamanos; Kristjana Kristinsdottir; Jonathan Huang; Dustin Carlson; Rajesh Keswani; John Pandolfino; Mozziyar Etemadi
>
> **摘要:** Endoscopic procedures such as esophagogastroduodenoscopy (EGD) and colonoscopy play a critical role in diagnosing and managing gastrointestinal (GI) disorders. However, the documentation burden associated with these procedures place significant strain on gastroenterologists, contributing to inefficiencies in clinical workflows and physician burnout. To address this challenge, we propose a novel automated report generation model that leverages a transformer-based vision encoder and text decoder within a two-stage training framework. In the first stage, both components are pre-trained on image/text caption pairs to capture generalized vision-language features, followed by fine-tuning on images/report pairs to generate clinically meaningful findings. Our approach not only streamlines the documentation process but also holds promise for reducing physician workload and improving patient care.
>
---
#### [new 006] MonitorVLM:A Vision Language Framework for Safety Violation Detection in Mining Operations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于安全监测任务，旨在解决采矿作业中因工人不安全行为导致的事故问题。论文提出了MonitorVLM框架，结合视觉与语言模型，实现从监控视频中自动检测安全违规行为。工作包括构建专用数据集、设计条款筛选模块和行为放大模块，提升了检测精度与效率。**

- **链接: [http://arxiv.org/pdf/2510.03666v1](http://arxiv.org/pdf/2510.03666v1)**

> **作者:** Jiang Wu; Sichao Wu; Yinsong Ma; Guangyuan Yu; Haoyuan Xu; Lifang Zheng; Jingliang Duan
>
> **摘要:** Industrial accidents, particularly in high-risk domains such as surface and underground mining, are frequently caused by unsafe worker behaviors. Traditional manual inspection remains labor-intensive, error-prone, and insufficient for large-scale, dynamic environments, highlighting the urgent need for intelligent and automated safety monitoring. In this paper, we present MonitorVLM, a novel vision--language framework designed to detect safety violations directly from surveillance video streams. MonitorVLM introduces three key innovations: (1) a domain-specific violation dataset comprising 9,000 vision--question--answer (VQA) samples across 40 high-frequency mining regulations, enriched with augmentation and auxiliary detection cues; (2) a clause filter (CF) module that dynamically selects the Top-$K$ most relevant clauses, reducing inference latency by 13.56\% while maintaining accuracy; and (3) a behavior magnifier (BM) module that enhances worker regions to improve fine-grained action recognition, yielding additional gains of 3.45% in precision and 8.62% in recall. Experimental results demonstrate that MonitorVLM significantly outperforms baseline vision--language models, achieving improvements of 22.01% in precision, 34.22\% in recall, and 28.37% in F1 score over the 72B unfine-tuned baseline. A lightweight web-based interface further integrates MonitorVLM into practical workflows, enabling automatic violation reporting with video timestamping. This study highlights the potential of multimodal large models to enhance occupational safety monitoring in mining and beyond.
>
---
#### [new 007] Provenance Networks: End-to-End Exemplar-Based Explainability
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出“溯源网络”，一种新型神经网络模型，旨在提供端到端、基于训练数据的可解释性。它通过将预测结果直接关联到支持该预测的训练样例，嵌入可解释性于模型结构中，解决了传统模型不透明、难以追溯预测依据的问题，同时提升了模型的可信度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.03361v1](http://arxiv.org/pdf/2510.03361v1)**

> **作者:** Ali Kayyam; Anusha Madan Gopal; M. Anthony Lewis
>
> **摘要:** We introduce provenance networks, a novel class of neural models designed to provide end-to-end, training-data-driven explainability. Unlike conventional post-hoc methods, provenance networks learn to link each prediction directly to its supporting training examples as part of the model's normal operation, embedding interpretability into the architecture itself. Conceptually, the model operates similarly to a learned KNN, where each output is justified by concrete exemplars weighted by relevance in the feature space. This approach facilitates systematic investigations of the trade-off between memorization and generalization, enables verification of whether a given input was included in the training set, aids in the detection of mislabeled or anomalous data points, enhances resilience to input perturbations, and supports the identification of similar inputs contributing to the generation of a new data point. By jointly optimizing the primary task and the explainability objective, provenance networks offer insights into model behavior that traditional deep networks cannot provide. While the model introduces additional computational cost and currently scales to moderately sized datasets, it provides a complementary approach to existing explainability techniques. In particular, it addresses critical challenges in modern deep learning, including model opaqueness, hallucination, and the assignment of credit to data contributors, thereby improving transparency, robustness, and trustworthiness in neural models.
>
---
#### [new 008] GAS-MIL: Group-Aggregative Selection Multi-Instance Learning for Ensemble of Foundation Models in Digital Pathology Image Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于数字病理图像分析任务，旨在解决如何高效整合多个基础模型（FMs）以提升诊断性能的问题。作者提出GAS-MIL方法，通过多实例学习集成多个FMs的特征，无需手动选择或精细调优，实现了跨癌症数据集的优异分类效果。**

- **链接: [http://arxiv.org/pdf/2510.03555v1](http://arxiv.org/pdf/2510.03555v1)**

> **作者:** Peiran Quan; Zifan Gu; Zhuo Zhao; Qin Zhou; Donghan M. Yang; Ruichen Rong; Yang Xie; Guanghua Xiao
>
> **摘要:** Foundation models (FMs) have transformed computational pathology by providing powerful, general-purpose feature extractors. However, adapting and benchmarking individual FMs for specific diagnostic tasks is often time-consuming and resource-intensive, especially given their scale and diversity. To address this challenge, we introduce Group-Aggregative Selection Multi-Instance Learning (GAS-MIL), a flexible ensemble framework that seamlessly integrates features from multiple FMs, preserving their complementary strengths without requiring manual feature selection or extensive task-specific fine-tuning. Across classification tasks in three cancer datasets-prostate (PANDA), ovarian (UBC-OCEAN), and breast (TCGA-BrCa)-GAS-MIL consistently achieves superior or on-par performance relative to individual FMs and established MIL methods, demonstrating its robustness and generalizability. By enabling efficient integration of heterogeneous FMs, GAS-MIL streamlines model deployment for pathology and provides a scalable foundation for future multimodal and precision oncology applications.
>
---
#### [new 009] Character Mixing for Video Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到视频生成任务，旨在解决不同风格角色跨世界自然交互的问题。通过提出跨角色嵌入（CCE）和跨角色增强（CCA）方法，实现角色身份与行为逻辑的融合，克服风格混淆问题，提升交互质量与风格保真度。**

- **链接: [http://arxiv.org/pdf/2510.05093v1](http://arxiv.org/pdf/2510.05093v1)**

> **作者:** Tingting Liao; Chongjian Ge; Guangyi Liu; Hao Li; Yi Zhou
>
> **摘要:** Imagine Mr. Bean stepping into Tom and Jerry--can we generate videos where characters interact naturally across different worlds? We study inter-character interaction in text-to-video generation, where the key challenge is to preserve each character's identity and behaviors while enabling coherent cross-context interaction. This is difficult because characters may never have coexisted and because mixing styles often causes style delusion, where realistic characters appear cartoonish or vice versa. We introduce a framework that tackles these issues with Cross-Character Embedding (CCE), which learns identity and behavioral logic across multimodal sources, and Cross-Character Augmentation (CCA), which enriches training with synthetic co-existence and mixed-style data. Together, these techniques allow natural interactions between previously uncoexistent characters without losing stylistic fidelity. Experiments on a curated benchmark of cartoons and live-action series with 10 characters show clear improvements in identity preservation, interaction quality, and robustness to style delusion, enabling new forms of generative storytelling.Additional results and videos are available on our project page: https://tingtingliao.github.io/mimix/.
>
---
#### [new 010] Factuality Matters: When Image Generation and Editing Meet Structured Visuals
- **分类: cs.CV**

- **简介: 该论文研究图像生成与编辑在结构化视觉内容（如图表、数学图形）中的应用，旨在解决现有模型在事实准确性上的不足。作者构建了大规模数据集，训练了融合视觉语言模型与推理能力的统一模型，并提出新评估基准StructBench与StructScore。**

- **链接: [http://arxiv.org/pdf/2510.05091v1](http://arxiv.org/pdf/2510.05091v1)**

> **作者:** Le Zhuo; Songhao Han; Yuandong Pu; Boxiang Qiu; Sayak Paul; Yue Liao; Yihao Liu; Jie Shao; Xi Chen; Si Liu; Hongsheng Li
>
> **备注:** Project page: https://structvisuals.github.io
>
> **摘要:** While modern visual generation models excel at creating aesthetically pleasing natural images, they struggle with producing or editing structured visuals like charts, diagrams, and mathematical figures, which demand composition planning, text rendering, and multimodal reasoning for factual fidelity. To address this, we present the first comprehensive, systematic investigation of this domain, encompassing data construction, model training, and an evaluation benchmark. First, we construct a large-scale dataset of 1.3 million high-quality structured image pairs derived from executable drawing programs and augmented with chain-of-thought reasoning annotations. Building on it, we train a unified model that integrates a VLM with FLUX.1 Kontext via a lightweight connector for enhanced multimodal understanding. A three-stage training curriculum enables progressive feature alignment, knowledge infusion, and reasoning-augmented generation, further boosted by an external reasoner at inference time. Finally, we introduce StructBench, a novel benchmark for generation and editing with over 1,700 challenging instances, and an accompanying evaluation metric, StructScore, which employs a multi-round Q\&A protocol to assess fine-grained factual accuracy. Evaluations of 15 models reveal that even leading closed-source systems remain far from satisfactory. Our model attains strong editing performance, and inference-time reasoning yields consistent gains across diverse architectures. By releasing the dataset, model, and benchmark, we aim to advance unified multimodal foundations for structured visuals.
>
---
#### [new 011] A Spatial-Spectral-Frequency Interactive Network for Multimodal Remote Sensing Classification
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分类任务，旨在解决多模态遥感图像中结构与细节特征提取困难的问题。作者提出了一种空间-光谱-频率交互网络（S²Fin），融合多域特征并引入频率域学习，通过高频频段增强和双层融合策略，提升了分类性能。**

- **链接: [http://arxiv.org/pdf/2510.04628v1](http://arxiv.org/pdf/2510.04628v1)**

> **作者:** Hao Liu; Yunhao Gao; Wei Li; Mingyang Zhang; Maoguo Gong; Lorenzo Bruzzone
>
> **摘要:** Deep learning-based methods have achieved significant success in remote sensing Earth observation data analysis. Numerous feature fusion techniques address multimodal remote sensing image classification by integrating global and local features. However, these techniques often struggle to extract structural and detail features from heterogeneous and redundant multimodal images. With the goal of introducing frequency domain learning to model key and sparse detail features, this paper introduces the spatial-spectral-frequency interaction network (S$^2$Fin), which integrates pairwise fusion modules across the spatial, spectral, and frequency domains. Specifically, we propose a high-frequency sparse enhancement transformer that employs sparse spatial-spectral attention to optimize the parameters of the high-frequency filter. Subsequently, a two-level spatial-frequency fusion strategy is introduced, comprising an adaptive frequency channel module that fuses low-frequency structures with enhanced high-frequency details, and a high-frequency resonance mask that emphasizes sharp edges via phase similarity. In addition, a spatial-spectral attention fusion module further enhances feature extraction at intermediate layers of the network. Experiments on four benchmark multimodal datasets with limited labeled data demonstrate that S$^2$Fin performs superior classification, outperforming state-of-the-art methods. The code is available at https://github.com/HaoLiu-XDU/SSFin.
>
---
#### [new 012] Talking Tennis: Language Feedback from 3D Biomechanical Action Recognition
- **分类: cs.CV; cs.AI; cs.HC; I.2.10; I.5.4; I.2.7**

- **简介: 该论文属于运动分析与人工智能结合的任务，旨在解决现有网球击球分析系统无法将生物力学特征转化为可操作语言反馈的问题。作者通过CNN-LSTM模型提取生物力学特征，并利用大语言模型生成技术性、可解释的反馈，提升训练指导效果。**

- **链接: [http://arxiv.org/pdf/2510.03921v1](http://arxiv.org/pdf/2510.03921v1)**

> **作者:** Arushi Dashore; Aryan Anumala; Emily Hui; Olivia Yang
>
> **备注:** 10 pages, 4 figures, 2 tables
>
> **摘要:** Automated tennis stroke analysis has advanced significantly with the integration of biomechanical motion cues alongside deep learning techniques, enhancing stroke classification accuracy and player performance evaluation. Despite these advancements, existing systems often fail to connect biomechanical insights with actionable language feedback that is both accessible and meaningful to players and coaches. This research project addresses this gap by developing a novel framework that extracts key biomechanical features (such as joint angles, limb velocities, and kinetic chain patterns) from motion data using Convolutional Neural Network Long Short-Term Memory (CNN-LSTM)-based models. These features are analyzed for relationships influencing stroke effectiveness and injury risk, forming the basis for feedback generation using large language models (LLMs). Leveraging the THETIS dataset and feature extraction techniques, our approach aims to produce feedback that is technically accurate, biomechanically grounded, and actionable for end-users. The experimental setup evaluates this framework on classification performance and interpretability, bridging the gap between explainable AI and sports biomechanics.
>
---
#### [new 013] Detailed Aerial Mapping of Photovoltaic Power Plants Through Semantically Significant Keypoints
- **分类: cs.CV**

- **简介: 该论文属于图像处理与建模任务，旨在解决光伏电站缺乏精确模型的问题。作者提出一种基于航拍图像的自动建模方法，通过语义关键点识别与多视角融合，实现光伏组件级别的三维建模，支持电站维护。**

- **链接: [http://arxiv.org/pdf/2510.04840v1](http://arxiv.org/pdf/2510.04840v1)**

> **作者:** Viktor Kozák; Jan Chudoba; Libor Přeučil
>
> **备注:** 10 pages, 18 figures
>
> **摘要:** An accurate and up-to-date model of a photovoltaic (PV) power plant is essential for its optimal operation and maintenance. However, such a model may not be easily available. This work introduces a novel approach for PV power plant mapping based on aerial overview images. It enables the automation of the mapping process while removing the reliance on third-party data. The presented mapping method takes advantage of the structural layout of the power plants to achieve detailed modeling down to the level of individual PV modules. The approach relies on visual segmentation of PV modules in overview images and the inference of structural information in each image, assigning modules to individual benches, rows, and columns. We identify visual keypoints related to the layout and use these to merge detections from multiple images while maintaining their structural integrity. The presented method was experimentally verified and evaluated on two different power plants. The final fusion of 3D positions and semantic structures results in a compact georeferenced model suitable for power plant maintenance.
>
---
#### [new 014] Mirage: Unveiling Hidden Artifacts in Synthetic Images with Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决当前检测方法对含可见伪影的合成图像识别效果差的问题。作者构建了新数据集Mirage，并探索了大视觉语言模型（LVLMs）在此任务中的应用，发现其对含伪影图像检测效果较好，但在无明显线索时表现下降。**

- **链接: [http://arxiv.org/pdf/2510.03840v1](http://arxiv.org/pdf/2510.03840v1)**

> **作者:** Pranav Sharma; Shivank Garg; Durga Toshniwal
>
> **备注:** ACM MM'25, MALLM Workshop
>
> **摘要:** Recent advances in image generation models have led to models that produce synthetic images that are increasingly difficult for standard AI detectors to identify, even though they often remain distinguishable by humans. To identify this discrepancy, we introduce \textbf{Mirage}, a curated dataset comprising a diverse range of AI-generated images exhibiting visible artifacts, where current state-of-the-art detection methods largely fail. Furthermore, we investigate whether Large Vision-Language Models (LVLMs), which are increasingly employed as substitutes for human judgment in various tasks, can be leveraged for explainable AI image detection. Our experiments on both Mirage and existing benchmark datasets demonstrate that while LVLMs are highly effective at detecting AI-generated images with visible artifacts, their performance declines when confronted with images lacking such cues.
>
---
#### [new 015] Learning Efficient Meshflow and Optical Flow from Event Cameras
- **分类: cs.CV**

- **简介: 该论文属于事件相机下的网格流估计任务，旨在解决现有方法缺乏专用数据集与处理事件密度不足的问题。作者构建了HREM数据集，提出了轻量级EEMFlow网络及其改进版本EEMFlow+，并设计ADM模块提升模型对不同密度数据的适应性，显著提高了性能与效率。**

- **链接: [http://arxiv.org/pdf/2510.04111v1](http://arxiv.org/pdf/2510.04111v1)**

> **作者:** Xinglong Luo; Ao Luo; Kunming Luo; Zhengning Wang; Ping Tan; Bing Zeng; Shuaicheng Liu
>
> **备注:** Accepted by TPAMI 2025
>
> **摘要:** In this paper, we explore the problem of event-based meshflow estimation, a novel task that involves predicting a spatially smooth sparse motion field from event cameras. To start, we review the state-of-the-art in event-based flow estimation, highlighting two key areas for further research: i) the lack of meshflow-specific event datasets and methods, and ii) the underexplored challenge of event data density. First, we generate a large-scale High-Resolution Event Meshflow (HREM) dataset, which showcases its superiority by encompassing the merits of high resolution at 1280x720, handling dynamic objects and complex motion patterns, and offering both optical flow and meshflow labels. These aspects have not been fully explored in previous works. Besides, we propose Efficient Event-based MeshFlow (EEMFlow) network, a lightweight model featuring a specially crafted encoder-decoder architecture to facilitate swift and accurate meshflow estimation. Furthermore, we upgrade EEMFlow network to support dense event optical flow, in which a Confidence-induced Detail Completion (CDC) module is proposed to preserve sharp motion boundaries. We conduct comprehensive experiments to show the exceptional performance and runtime efficiency (30x faster) of our EEMFlow model compared to the recent state-of-the-art flow method. As an extension, we expand HREM into HREM+, a multi-density event dataset contributing to a thorough study of the robustness of existing methods across data with varying densities, and propose an Adaptive Density Module (ADM) to adjust the density of input event data to a more optimal range, enhancing the model's generalization ability. We empirically demonstrate that ADM helps to significantly improve the performance of EEMFlow and EEMFlow+ by 8% and 10%, respectively. Code and dataset are released at https://github.com/boomluo02/EEMFlowPlus.
>
---
#### [new 016] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于长尾分布半监督学习任务，旨在解决未标记数据分布未知情况下的性能下降问题。作者提出了可控伪标签生成框架（CPG），通过动态筛选机制将可靠伪标签加入训练集，并结合贝叶斯最优分类器与自适应增强模块，提升模型性能。实验表明其准确性超越现有方法。**

- **链接: [http://arxiv.org/pdf/2510.03993v1](http://arxiv.org/pdf/2510.03993v1)**

> **作者:** Yaxin Hou; Bo Han; Yuheng Jia; Hui Liu; Junhui Hou
>
> **备注:** The paper is accepted by NeurIPS 2025
>
> **摘要:** Current long-tailed semi-supervised learning methods assume that labeled data exhibit a long-tailed distribution, and unlabeled data adhere to a typical predefined distribution (i.e., long-tailed, uniform, or inverse long-tailed). However, the distribution of the unlabeled data is generally unknown and may follow an arbitrary distribution. To tackle this challenge, we propose a Controllable Pseudo-label Generation (CPG) framework, expanding the labeled dataset with the progressively identified reliable pseudo-labels from the unlabeled dataset and training the model on the updated labeled dataset with a known distribution, making it unaffected by the unlabeled data distribution. Specifically, CPG operates through a controllable self-reinforcing optimization cycle: (i) at each training step, our dynamic controllable filtering mechanism selectively incorporates reliable pseudo-labels from the unlabeled dataset into the labeled dataset, ensuring that the updated labeled dataset follows a known distribution; (ii) we then construct a Bayes-optimal classifier using logit adjustment based on the updated labeled data distribution; (iii) this improved classifier subsequently helps identify more reliable pseudo-labels in the next training step. We further theoretically prove that this optimization cycle can significantly reduce the generalization error under some conditions. Additionally, we propose a class-aware adaptive augmentation module to further improve the representation of minority classes, and an auxiliary branch to maximize data utilization by leveraging all labeled and unlabeled samples. Comprehensive evaluations on various commonly used benchmark datasets show that CPG achieves consistent improvements, surpassing state-of-the-art methods by up to \textbf{15.97\%} in accuracy. The code is available at https://github.com/yaxinhou/CPG.
>
---
#### [new 017] Latent Uncertainty Representations for Video-based Driver Action and Intention Recognition
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视频驾驶员动作与意图识别任务，旨在提升模型在资源受限环境下的不确定性估计与分类性能。论文提出了一种基于预训练DNN扩展的潜在不确定性表示方法（LUR）及其改进版RLUR，用于有效检测分布外样本并优化不确定性估计，相比现有方法更高效且易于调参。**

- **链接: [http://arxiv.org/pdf/2510.05006v1](http://arxiv.org/pdf/2510.05006v1)**

> **作者:** Koen Vellenga; H. Joe Steinhauer; Jonas Andersson; Anders Sjögren
>
> **备注:** 16 pages, 8 figures, 7 tables, under submission
>
> **摘要:** Deep neural networks (DNNs) are increasingly applied to safety-critical tasks in resource-constrained environments, such as video-based driver action and intention recognition. While last layer probabilistic deep learning (LL-PDL) methods can detect out-of-distribution (OOD) instances, their performance varies. As an alternative to last layer approaches, we propose extending pre-trained DNNs with transformation layers to produce multiple latent representations to estimate the uncertainty. We evaluate our latent uncertainty representation (LUR) and repulsively trained LUR (RLUR) approaches against eight PDL methods across four video-based driver action and intention recognition datasets, comparing classification performance, calibration, and uncertainty-based OOD detection. We also contribute 28,000 frame-level action labels and 1,194 video-level intention labels for the NuScenes dataset. Our results show that LUR and RLUR achieve comparable in-distribution classification performance to other LL-PDL approaches. For uncertainty-based OOD detection, LUR matches top-performing PDL methods while being more efficient to train and easier to tune than approaches that require Markov-Chain Monte Carlo sampling or repulsive training procedures.
>
---
#### [new 018] Diffusion Low Rank Hybrid Reconstruction for Sparse View Medical Imaging
- **分类: cs.CV**

- **简介: 该论文属于医学成像任务，旨在解决低剂量稀疏视角CT重建中的图像质量下降问题。作者提出了TV-LoRA方法，结合扩散生成先验与多正则化约束，在ADMM框架下实现高效重建。实验表明其在多个数据集上优于现有方法，具有良好的鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.04069v1](http://arxiv.org/pdf/2510.04069v1)**

> **作者:** Zongyin Deng; Qing Zhou; Yuhao Fang; Zijian Wang; Yao Lu; Ye Zhang; Chun Li
>
> **摘要:** This work presents TV-LoRA, a novel method for low-dose sparse-view CT reconstruction that combines a diffusion generative prior (NCSN++ with SDE modeling) and multi-regularization constraints, including anisotropic TV and nuclear norm (LoRA), within an ADMM framework. To address ill-posedness and texture loss under extremely sparse views, TV-LoRA integrates generative and physical constraints, and utilizes a 2D slice-based strategy with FFT acceleration and tensor-parallel optimization for efficient inference. Experiments on AAPM-2016, CTHD, and LIDC datasets with $N_{\mathrm{view}}=8,4,2$ show that TV-LoRA consistently surpasses benchmarks in SSIM, texture recovery, edge clarity, and artifact suppression, demonstrating strong robustness and generalizability. Ablation studies confirm the complementary effects of LoRA regularization and diffusion priors, while the FFT-PCG module provides a speedup. Overall, Diffusion + TV-LoRA achieves high-fidelity, efficient 3D CT reconstruction and broad clinical applicability in low-dose, sparse-sampling scenarios.
>
---
#### [new 019] MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D资产检索任务，旨在解决元宇宙场景生成中资产检索不一致和缺乏标准范式的问题。论文提出MetaFind框架，支持文本、图像、3D多模态查询，通过建模对象特征与场景结构，提升检索结果的空间与风格一致性，并实现迭代场景构建。**

- **链接: [http://arxiv.org/pdf/2510.04057v1](http://arxiv.org/pdf/2510.04057v1)**

> **作者:** Zhenyu Pan; Yucheng Lu; Han Liu
>
> **备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** We present MetaFind, a scene-aware tri-modal compositional retrieval framework designed to enhance scene generation in the metaverse by retrieving 3D assets from large-scale repositories. MetaFind addresses two core challenges: (i) inconsistent asset retrieval that overlooks spatial, semantic, and stylistic constraints, and (ii) the absence of a standardized retrieval paradigm specifically tailored for 3D asset retrieval, as existing approaches mainly rely on general-purpose 3D shape representation models. Our key innovation is a flexible retrieval mechanism that supports arbitrary combinations of text, image, and 3D modalities as queries, enhancing spatial reasoning and style consistency by jointly modeling object-level features (including appearance) and scene-level layout structures. Methodologically, MetaFind introduces a plug-and-play equivariant layout encoder ESSGNN that captures spatial relationships and object appearance features, ensuring retrieved 3D assets are contextually and stylistically coherent with the existing scene, regardless of coordinate frame transformations. The framework supports iterative scene construction by continuously adapting retrieval results to current scene updates. Empirical evaluations demonstrate the improved spatial and stylistic consistency of MetaFind in various retrieval tasks compared to baseline methods.
>
---
#### [new 020] Streaming Drag-Oriented Interactive Video Manipulation: Drag Anything, Anytime!
- **分类: cs.CV**

- **简介: 该论文提出REVEL任务，旨在实现对视频扩散模型输出的流式、细粒度交互编辑。主要解决拖拽操作导致的潜在空间漂移和上下文干扰问题。作者提出DragStream方法，通过自适应分布校正和空间频率选择优化，实现流畅自然的视频编辑。**

- **链接: [http://arxiv.org/pdf/2510.03550v1](http://arxiv.org/pdf/2510.03550v1)**

> **作者:** Junbao Zhou; Yuan Zhou; Kesen Zhao; Qingshan Xu; Beier Zhu; Richang Hong; Hanwang Zhang
>
> **摘要:** Achieving streaming, fine-grained control over the outputs of autoregressive video diffusion models remains challenging, making it difficult to ensure that they consistently align with user expectations. To bridge this gap, we propose \textbf{stReaming drag-oriEnted interactiVe vidEo manipuLation (REVEL)}, a new task that enables users to modify generated videos \emph{anytime} on \emph{anything} via fine-grained, interactive drag. Beyond DragVideo and SG-I2V, REVEL unifies drag-style video manipulation as editing and animating video frames with both supporting user-specified translation, deformation, and rotation effects, making drag operations versatile. In resolving REVEL, we observe: \emph{i}) drag-induced perturbations accumulate in latent space, causing severe latent distribution drift that halts the drag process; \emph{ii}) streaming drag is easily disturbed by context frames, thereby yielding visually unnatural outcomes. We thus propose a training-free approach, \textbf{DragStream}, comprising: \emph{i}) an adaptive distribution self-rectification strategy that leverages neighboring frames' statistics to effectively constrain the drift of latent embeddings; \emph{ii}) a spatial-frequency selective optimization mechanism, allowing the model to fully exploit contextual information while mitigating its interference via selectively propagating visual cues along generation. Our method can be seamlessly integrated into existing autoregressive video diffusion models, and extensive experiments firmly demonstrate the effectiveness of our DragStream.
>
---
#### [new 021] Beyond Random: Automatic Inner-loop Optimization in Dataset Distillation
- **分类: cs.CV; cs.LG**

- **简介: 论文属于高效深度学习任务，旨在解决数据蒸馏中的内循环优化问题。现有方法依赖随机截断，效果不佳。作者提出AT-BPTT框架，动态调整截断位置与窗口大小，提升模型性能。实验表明其准确率、速度与内存效率均优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.04838v1](http://arxiv.org/pdf/2510.04838v1)**

> **作者:** Muquan Li; Hang Gou; Dongyang Zhang; Shuang Liang; Xiurui Xie; Deqiang Ouyang; Ke Qin
>
> **摘要:** The growing demand for efficient deep learning has positioned dataset distillation as a pivotal technique for compressing training dataset while preserving model performance. However, existing inner-loop optimization methods for dataset distillation typically rely on random truncation strategies, which lack flexibility and often yield suboptimal results. In this work, we observe that neural networks exhibit distinct learning dynamics across different training stages-early, middle, and late-making random truncation ineffective. To address this limitation, we propose Automatic Truncated Backpropagation Through Time (AT-BPTT), a novel framework that dynamically adapts both truncation positions and window sizes according to intrinsic gradient behavior. AT-BPTT introduces three key components: (1) a probabilistic mechanism for stage-aware timestep selection, (2) an adaptive window sizing strategy based on gradient variation, and (3) a low-rank Hessian approximation to reduce computational overhead. Extensive experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet, and ImageNet-1K show that AT-BPTT achieves state-of-the-art performance, improving accuracy by an average of 6.16% over baseline methods. Moreover, our approach accelerates inner-loop optimization by 3.9x while saving 63% memory cost.
>
---
#### [new 022] Artery-Vein Segmentation from Fundus Images using Deep Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决视网膜动脉和静脉的分割问题。为实现这一目标，作者提出了一种基于注意力机制的深度学习模型Attention-WNet，并在HRF和DRIVE数据集上验证了其优越性能。**

- **链接: [http://arxiv.org/pdf/2510.03717v1](http://arxiv.org/pdf/2510.03717v1)**

> **作者:** Sharan SK; Subin Sahayam; Umarani Jayaraman; Lakshmi Priya A
>
> **备注:** 12 pages, 6 figures, preprint under review
>
> **摘要:** Segmenting of clinically important retinal blood vessels into arteries and veins is a prerequisite for retinal vessel analysis. Such analysis can provide potential insights and bio-markers for identifying and diagnosing various retinal eye diseases. Alteration in the regularity and width of the retinal blood vessels can act as an indicator of the health of the vasculature system all over the body. It can help identify patients at high risk of developing vasculature diseases like stroke and myocardial infarction. Over the years, various Deep Learning architectures have been proposed to perform retinal vessel segmentation. Recently, attention mechanisms have been increasingly used in image segmentation tasks. The work proposes a new Deep Learning approach for artery-vein segmentation. The new approach is based on the Attention mechanism that is incorporated into the WNet Deep Learning model, and we call the model as Attention-WNet. The proposed approach has been tested on publicly available datasets such as HRF and DRIVE datasets. The proposed approach has outperformed other state-of-art models available in the literature.
>
---
#### [new 023] Bidirectional Mammogram View Translation with Column-Aware and Implicit 3D Conditional Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像处理任务，旨在解决双视图乳腺X线图像中因缺失或损坏视图导致的诊断困难问题。提出了CA3D-Diff框架，通过列感知注意力机制和隐式3D结构重建，实现双向视图转换，提升图像质量和病变分类效果。**

- **链接: [http://arxiv.org/pdf/2510.04947v1](http://arxiv.org/pdf/2510.04947v1)**

> **作者:** Xin Li; Kaixiang Yang; Qiang Li; Zhiwei Wang
>
> **备注:** BIBM2025 accept, 8 pages, 4 figures
>
> **摘要:** Dual-view mammography, including craniocaudal (CC) and mediolateral oblique (MLO) projections, offers complementary anatomical views crucial for breast cancer diagnosis. However, in real-world clinical workflows, one view may be missing, corrupted, or degraded due to acquisition errors or compression artifacts, limiting the effectiveness of downstream analysis. View-to-view translation can help recover missing views and improve lesion alignment. Unlike natural images, this task in mammography is highly challenging due to large non-rigid deformations and severe tissue overlap in X-ray projections, which obscure pixel-level correspondences. In this paper, we propose Column-Aware and Implicit 3D Diffusion (CA3D-Diff), a novel bidirectional mammogram view translation framework based on conditional diffusion model. To address cross-view structural misalignment, we first design a column-aware cross-attention mechanism that leverages the geometric property that anatomically corresponding regions tend to lie in similar column positions across views. A Gaussian-decayed bias is applied to emphasize local column-wise correlations while suppressing distant mismatches. Furthermore, we introduce an implicit 3D structure reconstruction module that back-projects noisy 2D latents into a coarse 3D feature volume based on breast-view projection geometry. The reconstructed 3D structure is refined and injected into the denoising UNet to guide cross-view generation with enhanced anatomical awareness. Extensive experiments demonstrate that CA3D-Diff achieves superior performance in bidirectional tasks, outperforming state-of-the-art methods in visual fidelity and structural consistency. Furthermore, the synthesized views effectively improve single-view malignancy classification in screening settings, demonstrating the practical value of our method in real-world diagnostics.
>
---
#### [new 024] PoseGaze-AHP: A Knowledge-Based 3D Dataset for AI-Driven Ocular and Postural Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决眼源性异常头位（AHP）缺乏综合数据集的问题。作者构建了PoseGaze-AHP，一个包含头姿与眼动信息的3D数据集，通过大语言模型提取临床数据并转化为3D表示，支持AI诊断工具开发。**

- **链接: [http://arxiv.org/pdf/2510.03873v1](http://arxiv.org/pdf/2510.03873v1)**

> **作者:** Saja Al-Dabet; Sherzod Turaev; Nazar Zaki; Arif O. Khan; Luai Eldweik
>
> **备注:** This is a preprint version of a manuscript under review. All rights reserved by the authors
>
> **摘要:** Diagnosing ocular-induced abnormal head posture (AHP) requires a comprehensive analysis of both head pose and ocular movements. However, existing datasets focus on these aspects separately, limiting the development of integrated diagnostic approaches and restricting AI-driven advancements in AHP analysis. To address this gap, we introduce PoseGaze-AHP, a novel 3D dataset that synchronously captures head pose and gaze movement information for ocular-induced AHP assessment. Structured clinical data were extracted from medical literature using large language models (LLMs) through an iterative process with the Claude 3.5 Sonnet model, combining stepwise, hierarchical, and complex prompting strategies. The extracted records were systematically imputed and transformed into 3D representations using the Neural Head Avatar (NHA) framework. The dataset includes 7,920 images generated from two head textures, covering a broad spectrum of ocular conditions. The extraction method achieved an overall accuracy of 91.92%, demonstrating its reliability for clinical dataset construction. PoseGaze-AHP is the first publicly available resource tailored for AI-driven ocular-induced AHP diagnosis, supporting the development of accurate and privacy-compliant diagnostic tools.
>
---
#### [new 025] REN: Anatomically-Informed Mixture-of-Experts for Interstitial Lung Disease Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决传统模型在考虑解剖结构和区域病理异质性方面的不足。论文提出了一种基于解剖先验的混合专家框架（REN），通过区域专家网络对不同肺叶和双肺组合进行建模，并结合多模态门控机制优化专家集成。该方法在间质性肺病分类中表现优异，提升了性能与临床可解释性。**

- **链接: [http://arxiv.org/pdf/2510.04923v1](http://arxiv.org/pdf/2510.04923v1)**

> **作者:** Alec K. Peltekian; Halil Ertugrul Aktas; Gorkem Durak; Kevin Grudzinski; Bradford C. Bemiss; Carrie Richardson; Jane E. Dematte; G. R. Scott Budinger; Anthony J. Esposito; Alexander Misharin; Alok Choudhary; Ankit Agrawal; Ulas Bagci
>
> **备注:** 10 pages, 4 figures, 2 tables
>
> **摘要:** Mixture-of-Experts (MoE) architectures have significantly contributed to scalable machine learning by enabling specialized subnetworks to tackle complex tasks efficiently. However, traditional MoE systems lack domain-specific constraints essential for medical imaging, where anatomical structure and regional disease heterogeneity strongly influence pathological patterns. Here, we introduce Regional Expert Networks (REN), the first anatomically-informed MoE framework tailored specifically for medical image classification. REN leverages anatomical priors to train seven specialized experts, each dedicated to distinct lung lobes and bilateral lung combinations, enabling precise modeling of region-specific pathological variations. Multi-modal gating mechanisms dynamically integrate radiomics biomarkers and deep learning (DL) features (CNN, ViT, Mamba) to weight expert contributions optimally. Applied to interstitial lung disease (ILD) classification, REN achieves consistently superior performance: the radiomics-guided ensemble reached an average AUC of 0.8646 +/- 0.0467, a +12.5 percent improvement over the SwinUNETR baseline (AUC 0.7685, p = 0.031). Region-specific experts further revealed that lower-lobe models achieved AUCs of 0.88-0.90, surpassing DL counterparts (CNN: 0.76-0.79) and aligning with known disease progression patterns. Through rigorous patient-level cross-validation, REN demonstrates strong generalizability and clinical interpretability, presenting a scalable, anatomically-guided approach readily extensible to other structured medical imaging applications.
>
---
#### [new 026] Multimodal Arabic Captioning with Interpretable Visual Concept Integration
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于图像描述生成任务，旨在解决阿拉伯语图像描述不准确和缺乏文化一致性的问题。作者提出了VLCAP框架，结合视觉标签检索与多模态文本生成，使用多语言编码器提取可解释的视觉概念，并融合通用领域标签，提升阿拉伯语图像描述的质量与文化适配性。**

- **链接: [http://arxiv.org/pdf/2510.03295v1](http://arxiv.org/pdf/2510.03295v1)**

> **作者:** Passant Elchafei; Amany Fashwan
>
> **摘要:** We present VLCAP, an Arabic image captioning framework that integrates CLIP-based visual label retrieval with multimodal text generation. Rather than relying solely on end-to-end captioning, VLCAP grounds generation in interpretable Arabic visual concepts extracted with three multilingual encoders, mCLIP, AraCLIP, and Jina V4, each evaluated separately for label retrieval. A hybrid vocabulary is built from training captions and enriched with about 21K general domain labels translated from the Visual Genome dataset, covering objects, attributes, and scenes. The top-k retrieved labels are transformed into fluent Arabic prompts and passed along with the original image to vision-language models. In the second stage, we tested Qwen-VL and Gemini Pro Vision for caption generation, resulting in six encoder-decoder configurations. The results show that mCLIP + Gemini Pro Vision achieved the best BLEU-1 (5.34%) and cosine similarity (60.01%), while AraCLIP + Qwen-VL obtained the highest LLM-judge score (36.33%). This interpretable pipeline enables culturally coherent and contextually accurate Arabic captions.
>
---
#### [new 027] ReactDiff: Fundamental Multiple Appropriate Facial Reaction Diffusion Model
- **分类: cs.CV; cs.HC; cs.MM**

- **简介: 该论文属于人机交互中的面部反应生成任务，旨在解决对话场景中多样且拟人化面部反应生成不足的问题。作者提出了ReactDiff模型，通过引入时间面部行为动力学和面部动作单元依赖性两个先验知识，优化扩散过程，使生成的面部反应更自然、连贯且符合面部解剖结构。**

- **链接: [http://arxiv.org/pdf/2510.04712v1](http://arxiv.org/pdf/2510.04712v1)**

> **作者:** Luo Cheng; Song Siyang; Yan Siyuan; Yu Zhen; Ge Zongyuan
>
> **备注:** Accepted to ACM Multimedia
>
> **摘要:** The automatic generation of diverse and human-like facial reactions in dyadic dialogue remains a critical challenge for human-computer interaction systems. Existing methods fail to model the stochasticity and dynamics inherent in real human reactions. To address this, we propose ReactDiff, a novel temporal diffusion framework for generating diverse facial reactions that are appropriate for responding to any given dialogue context. Our key insight is that plausible human reactions demonstrate smoothness, and coherence over time, and conform to constraints imposed by human facial anatomy. To achieve this, ReactDiff incorporates two vital priors (spatio-temporal facial kinematics) into the diffusion process: i) temporal facial behavioral kinematics and ii) facial action unit dependencies. These two constraints guide the model toward realistic human reaction manifolds, avoiding visually unrealistic jitters, unstable transitions, unnatural expressions, and other artifacts. Extensive experiments on the REACT2024 dataset demonstrate that our approach not only achieves state-of-the-art reaction quality but also excels in diversity and reaction appropriateness.
>
---
#### [new 028] LoRA Patching: Exposing the Fragility of Proactive Defenses against Deepfakes
- **分类: cs.CV**

- **简介: 该论文属于深度伪造（Deepfake）防御任务，旨在解决现有主动防御方法易被攻击的问题。作者提出了一种名为LoRA Patching的新方法，通过向Deepfake生成器注入低秩适配补丁，结合多模态特征对齐损失和可学习门控机制，有效绕过多种现有防御机制，并提出防御性补丁作为缓解方案。**

- **链接: [http://arxiv.org/pdf/2510.03747v1](http://arxiv.org/pdf/2510.03747v1)**

> **作者:** Zuomin Qu; Yimao Guo; Qianyue Hu; Wei Lu
>
> **摘要:** Deepfakes pose significant societal risks, motivating the development of proactive defenses that embed adversarial perturbations in facial images to prevent manipulation. However, in this paper, we show that these preemptive defenses often lack robustness and reliability. We propose a novel approach, Low-Rank Adaptation (LoRA) patching, which injects a plug-and-play LoRA patch into Deepfake generators to bypass state-of-the-art defenses. A learnable gating mechanism adaptively controls the effect of the LoRA patch and prevents gradient explosions during fine-tuning. We also introduce a Multi-Modal Feature Alignment (MMFA) loss, encouraging the features of adversarial outputs to align with those of the desired outputs at the semantic level. Beyond bypassing, we present defensive LoRA patching, embedding visible warnings in the outputs as a complementary solution to mitigate this newly identified security vulnerability. With only 1,000 facial examples and a single epoch of fine-tuning, LoRA patching successfully defeats multiple proactive defenses. These results reveal a critical weakness in current paradigms and underscore the need for more robust Deepfake defense strategies. Our code is available at https://github.com/ZOMIN28/LoRA-Patching.
>
---
#### [new 029] VChain: Chain-of-Visual-Thought for Reasoning in Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决生成复杂动态视频时缺乏连贯性与逻辑性的问题。作者提出VChain框架，利用多模态大模型生成关键帧作为视觉推理信号，指导视频生成模型在关键时间点进行稀疏调整，从而提升视频质量，实现高效推理。**

- **链接: [http://arxiv.org/pdf/2510.05094v1](http://arxiv.org/pdf/2510.05094v1)**

> **作者:** Ziqi Huang; Ning Yu; Gordon Chen; Haonan Qiu; Paul Debevec; Ziwei Liu
>
> **备注:** Project page: https://eyeline-labs.github.io/VChain Code: https://github.com/Eyeline-Labs/VChain
>
> **摘要:** Recent video generation models can produce smooth and visually appealing clips, but they often struggle to synthesize complex dynamics with a coherent chain of consequences. Accurately modeling visual outcomes and state transitions over time remains a core challenge. In contrast, large language and multimodal models (e.g., GPT-4o) exhibit strong visual state reasoning and future prediction capabilities. To bridge these strengths, we introduce VChain, a novel inference-time chain-of-visual-thought framework that injects visual reasoning signals from multimodal models into video generation. Specifically, VChain contains a dedicated pipeline that leverages large multimodal models to generate a sparse set of critical keyframes as snapshots, which are then used to guide the sparse inference-time tuning of a pre-trained video generator only at these key moments. Our approach is tuning-efficient, introduces minimal overhead and avoids dense supervision. Extensive experiments on complex, multi-step scenarios show that VChain significantly enhances the quality of generated videos.
>
---
#### [new 030] ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决生成身份一致且精确控制面部表情的人脸图像问题。论文提出基于扩散模型的框架，结合FLAME混合形状参数引导注意力模块，实现对复杂表情的精细控制，并支持真实图像中的表情编辑，提升了生成效果的身份一致性和表达多样性。**

- **链接: [http://arxiv.org/pdf/2510.04706v1](http://arxiv.org/pdf/2510.04706v1)**

> **作者:** Foivos Paraperas Papantoniou; Stefanos Zafeiriou
>
> **备注:** ICCVW 2025, Code: https://github.com/foivospar/Arc2Face
>
> **摘要:** Human-centric generative models designed for AI-driven storytelling must bring together two core capabilities: identity consistency and precise control over human performance. While recent diffusion-based approaches have made significant progress in maintaining facial identity, achieving fine-grained expression control without compromising identity remains challenging. In this work, we present a diffusion-based framework that faithfully reimagines any subject under any particular facial expression. Building on an ID-consistent face foundation model, we adopt a compositional design featuring an expression cross-attention module guided by FLAME blendshape parameters for explicit control. Trained on a diverse mixture of image and video data rich in expressive variation, our adapter generalizes beyond basic emotions to subtle micro-expressions and expressive transitions, overlooked by prior works. In addition, a pluggable Reference Adapter enables expression editing in real images by transferring the appearance from a reference frame during synthesis. Extensive quantitative and qualitative evaluations show that our model outperforms existing methods in tailored and identity-consistent expression generation. Code and models can be found at https://github.com/foivospar/Arc2Face.
>
---
#### [new 031] RAP: 3D Rasterization Augmented End-to-End Planning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决端到端驾驶策略在部署后缺乏恢复能力的问题。作者提出RAP方法，通过3D光栅化替代传统渲染，生成多样化视角与轨迹，并引入特征空间对齐策略以提升现实迁移效果，从而增强规划系统的鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.04333v1](http://arxiv.org/pdf/2510.04333v1)**

> **作者:** Lan Feng; Yang Gao; Eloi Zablocki; Quanyi Li; Wuyang Li; Sichao Liu; Matthieu Cord; Alexandre Alahi
>
> **摘要:** Imitation learning for end-to-end driving trains policies only on expert demonstrations. Once deployed in a closed loop, such policies lack recovery data: small mistakes cannot be corrected and quickly compound into failures. A promising direction is to generate alternative viewpoints and trajectories beyond the logged path. Prior work explores photorealistic digital twins via neural rendering or game engines, but these methods are prohibitively slow and costly, and thus mainly used for evaluation. In this work, we argue that photorealism is unnecessary for training end-to-end planners. What matters is semantic fidelity and scalability: driving depends on geometry and dynamics, not textures or lighting. Motivated by this, we propose 3D Rasterization, which replaces costly rendering with lightweight rasterization of annotated primitives, enabling augmentations such as counterfactual recovery maneuvers and cross-agent view synthesis. To transfer these synthetic views effectively to real-world deployment, we introduce a Raster-to-Real feature-space alignment that bridges the sim-to-real gap. Together, these components form Rasterization Augmented Planning (RAP), a scalable data augmentation pipeline for planning. RAP achieves state-of-the-art closed-loop robustness and long-tail generalization, ranking first on four major benchmarks: NAVSIM v1/v2, Waymo Open Dataset Vision-based E2E Driving, and Bench2Drive. Our results show that lightweight rasterization with feature alignment suffices to scale E2E training, offering a practical alternative to photorealistic rendering. Project page: https://alan-lanfeng.github.io/RAP/.
>
---
#### [new 032] Fit Pixels, Get Labels: Meta-learned Implicit Networks for Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决隐式神经表示（INRs）在语义分割中的应用难题。作者提出MetaSeg框架，通过元学习训练INRs同时预测像素值和类别标签，使模型在少量参数下实现与U-Net相当的性能，适用于2D和3D脑部MRI图像分割。**

- **链接: [http://arxiv.org/pdf/2510.04021v1](http://arxiv.org/pdf/2510.04021v1)**

> **作者:** Kushal Vyas; Ashok Veeraraghavan; Guha Balakrishnan
>
> **备注:** MICCAI 2025 (oral). Final peer-reviewed copy accessible at publisher DOI https://link.springer.com/chapter/10.1007/978-3-032-04947-6_19 . Project page, https://kushalvyas.github.io/metaseg.html
>
> **摘要:** Implicit neural representations (INRs) have achieved remarkable successes in learning expressive yet compact signal representations. However, they are not naturally amenable to predictive tasks such as segmentation, where they must learn semantic structures over a distribution of signals. In this study, we introduce MetaSeg, a meta-learning framework to train INRs for medical image segmentation. MetaSeg uses an underlying INR that simultaneously predicts per pixel intensity values and class labels. It then uses a meta-learning procedure to find optimal initial parameters for this INR over a training dataset of images and segmentation maps, such that the INR can simply be fine-tuned to fit pixels of an unseen test image, and automatically decode its class labels. We evaluated MetaSeg on 2D and 3D brain MRI segmentation tasks and report Dice scores comparable to commonly used U-Net models, but with $90\%$ fewer parameters. MetaSeg offers a fresh, scalable alternative to traditional resource-heavy architectures such as U-Nets and vision transformers for medical image segmentation. Our project is available at https://kushalvyas.github.io/metaseg.html .
>
---
#### [new 033] Let Features Decide Their Own Solvers: Hybrid Feature Caching for Diffusion Transformers
- **分类: cs.CV**

- **简介: 论文提出HyCa方法，针对扩散Transformer在图像视频生成中的高效推理问题。不同特征维度采用不同缓存策略，通过混合ODE建模实现近无损加速，显著提升FLUX、HunyuanVideo、Qwen-Image等模型的生成速度。**

- **链接: [http://arxiv.org/pdf/2510.04188v1](http://arxiv.org/pdf/2510.04188v1)**

> **作者:** Shikang Zheng; Guantao Chen; Qinming Zhou; Yuqi Lin; Lixuan He; Chang Zou; Peiliang Cai; Jiacheng Liu; Linfeng Zhang
>
> **摘要:** Diffusion Transformers offer state-of-the-art fidelity in image and video synthesis, but their iterative sampling process remains a major bottleneck due to the high cost of transformer forward passes at each timestep. To mitigate this, feature caching has emerged as a training-free acceleration technique that reuses or forecasts hidden representations. However, existing methods often apply a uniform caching strategy across all feature dimensions, ignoring their heterogeneous dynamic behaviors. Therefore, we adopt a new perspective by modeling hidden feature evolution as a mixture of ODEs across dimensions, and introduce HyCa, a Hybrid ODE solver inspired caching framework that applies dimension-wise caching strategies. HyCa achieves near-lossless acceleration across diverse domains and models, including 5.55 times speedup on FLUX, 5.56 times speedup on HunyuanVideo, 6.24 times speedup on Qwen-Image and Qwen-Image-Edit without retraining.
>
---
#### [new 034] MASC: Boosting Autoregressive Image Generation with a Manifold-Aligned Semantic Clustering
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决自回归模型因使用无结构视觉词表导致的训练效率低和生成质量差的问题。论文提出MASC方法，通过构建语义分层树，利用几何感知距离度量和密度驱动聚类优化嵌入空间结构，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.04220v1](http://arxiv.org/pdf/2510.04220v1)**

> **作者:** Lixuan He; Shikang Zheng; Linfeng Zhang
>
> **摘要:** Autoregressive (AR) models have shown great promise in image generation, yet they face a fundamental inefficiency stemming from their core component: a vast, unstructured vocabulary of visual tokens. This conventional approach treats tokens as a flat vocabulary, disregarding the intrinsic structure of the token embedding space where proximity often correlates with semantic similarity. This oversight results in a highly complex prediction task, which hinders training efficiency and limits final generation quality. To resolve this, we propose Manifold-Aligned Semantic Clustering (MASC), a principled framework that constructs a hierarchical semantic tree directly from the codebook's intrinsic structure. MASC employs a novel geometry-aware distance metric and a density-driven agglomerative construction to model the underlying manifold of the token embeddings. By transforming the flat, high-dimensional prediction task into a structured, hierarchical one, MASC introduces a beneficial inductive bias that significantly simplifies the learning problem for the AR model. MASC is designed as a plug-and-play module, and our extensive experiments validate its effectiveness: it accelerates training by up to 57% and significantly improves generation quality, reducing the FID of LlamaGen-XL from 2.87 to 2.58. MASC elevates existing AR frameworks to be highly competitive with state-of-the-art methods, establishing that structuring the prediction space is as crucial as architectural innovation for scalable generative modeling.
>
---
#### [new 035] No-reference Quality Assessment of Contrast-distorted Images using Contrast-enhanced Pseudo Reference
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，旨在解决无参考图像（NR）下对比度失真图像的质量评估问题。通过生成对比度增强的伪参考图像，将NR问题转化为全参考（FR）评估，提高准确性。论文使用对比度增强算法生成伪参考，并训练分类网络选择合适算法，最终在三个对比度失真数据库上验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2510.05053v1](http://arxiv.org/pdf/2510.05053v1)**

> **作者:** Mohammad-Ali Mahmoudpour; Saeed Mahmoudpour
>
> **摘要:** Contrast change is an important factor that affects the quality of images. During image capturing, unfavorable lighting conditions can cause contrast change and visual quality loss. While various methods have been proposed to assess the quality of images under different distortions such as blur and noise, contrast distortion has been largely overlooked as its visual impact and properties are different from other conventional types of distortions. In this paper, we propose a no-reference image quality assessment (NR-IQA) metric for contrast-distorted images. Using a set of contrast enhancement algorithms, we aim to generate pseudo-reference images that are visually close to the actual reference image, such that the NR problem is transformed to a Full-reference (FR) assessment with higher accuracy. To this end, a large dataset of contrast-enhanced images is produced to train a classification network that can select the most suitable contrast enhancement algorithm based on image content and distortion for pseudo-reference image generation. Finally, the evaluation is performed in the FR manner to assess the quality difference between the contrast-enhanced (pseudoreference) and degraded images. Performance evaluation of the proposed method on three databases containing contrast distortions (CCID2014, TID2013, and CSIQ), indicates the promising performance of the proposed method.
>
---
#### [new 036] DHQA-4D: Perceptual Quality Assessment of Dynamic 4D Digital Human
- **分类: cs.CV**

- **简介: 该论文属于动态4D数字人质量评估任务，旨在解决4D人体网格在采集、压缩和传输过程中受噪声影响导致视觉体验下降的问题。作者构建了大规模数据集DHQA-4D，并提出DynaMesh-Rater模型，融合视觉、动作和几何特征，通过LMM与LoRA技术实现质量评分预测。**

- **链接: [http://arxiv.org/pdf/2510.03874v1](http://arxiv.org/pdf/2510.03874v1)**

> **作者:** Yunhao Li; Sijing Wu; Yucheng Zhu; Huiyu Duan; Zicheng Zhang; Guangtao Zhai
>
> **摘要:** With the rapid development of 3D scanning and reconstruction technologies, dynamic digital human avatars based on 4D meshes have become increasingly popular. A high-precision dynamic digital human avatar can be applied to various fields such as game production, animation generation, and remote immersive communication. However, these 4D human avatar meshes are prone to being degraded by various types of noise during the processes of collection, compression, and transmission, thereby affecting the viewing experience of users. In light of this fact, quality assessment of dynamic 4D digital humans becomes increasingly important. In this paper, we first propose a large-scale dynamic digital human quality assessment dataset, DHQA-4D, which contains 32 high-quality real-scanned 4D human mesh sequences, 1920 distorted textured 4D human meshes degraded by 11 textured distortions, as well as their corresponding textured and non-textured mean opinion scores (MOSs). Equipped with DHQA-4D dataset, we analyze the influence of different types of distortion on human perception for textured dynamic 4D meshes and non-textured dynamic 4D meshes. Additionally, we propose DynaMesh-Rater, a novel large multimodal model (LMM) based approach that is able to assess both textured 4D meshes and non-textured 4D meshes. Concretely, DynaMesh-Rater elaborately extracts multi-dimensional features, including visual features from a projected 2D video, motion features from cropped video clips, and geometry features from the 4D human mesh to provide comprehensive quality-related information. Then we utilize a LMM model to integrate the multi-dimensional features and conduct a LoRA-based instruction tuning technique to teach the LMM model to predict the quality scores. Extensive experimental results on the DHQA-4D dataset demonstrate the superiority of our DynaMesh-Rater method over previous quality assessment methods.
>
---
#### [new 037] A Comprehensive Review on Artificial Intelligence Empowered Solutions for Enhancing Pedestrian and Cyclist Safety
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人工智能与交通安全交叉任务，旨在解决行人和骑行者安全保护问题。论文系统回顾了基于摄像头的AI感知系统在检测分类、跟踪识别、轨迹预测和意图识别等方面的进展，并指出了数据、模型和部署方面的挑战，以指导未来研究方向。**

- **链接: [http://arxiv.org/pdf/2510.03314v1](http://arxiv.org/pdf/2510.03314v1)**

> **作者:** Shucheng Zhang; Yan Shi; Bingzhang Wang; Yuang Zhang; Muhammad Monjurul Karim; Kehua Chen; Chenxi Liu; Mehrdad Nasri; Yinhai Wang
>
> **备注:** 20 pages, 4 figures, 5 tables
>
> **摘要:** Ensuring the safety of vulnerable road users (VRUs), such as pedestrians and cyclists, remains a critical global challenge, as conventional infrastructure-based measures often prove inadequate in dynamic urban environments. Recent advances in artificial intelligence (AI), particularly in visual perception and reasoning, open new opportunities for proactive and context-aware VRU protection. However, existing surveys on AI applications for VRUs predominantly focus on detection, offering limited coverage of other vision-based tasks that are essential for comprehensive VRU understanding and protection. This paper presents a state-of-the-art review of recent progress in camera-based AI sensing systems for VRU safety, with an emphasis on developments from the past five years and emerging research trends. We systematically examine four core tasks, namely detection and classification, tracking and reidentification, trajectory prediction, and intent recognition and prediction, which together form the backbone of AI-empowered proactive solutions for VRU protection in intelligent transportation systems. To guide future research, we highlight four major open challenges from the perspectives of data, model, and deployment. By linking advances in visual AI with practical considerations for real-world implementation, this survey aims to provide a foundational reference for the development of next-generation sensing systems to enhance VRU safety.
>
---
#### [new 038] A Comparative Study of Vision Transformers and CNNs for Few-Shot Rigid Transformation and Fundamental Matrix Estimation
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决在小数据场景下几何估计问题，比较了视觉Transformer（ViT）和卷积神经网络（CNN）在图像刚性变换和基础矩阵估计中的性能。工作包括系统性对比不同模型在多种数据规模下的表现，发现ViT在大数据和跨域场景中更优，而CNN在小数据中表现更好。**

- **链接: [http://arxiv.org/pdf/2510.04794v1](http://arxiv.org/pdf/2510.04794v1)**

> **作者:** Alon Kaya; Igal Bilik; Inna Stainvas
>
> **摘要:** Vision-transformers (ViTs) and large-scale convolution-neural-networks (CNNs) have reshaped computer vision through pretrained feature representations that enable strong transfer learning for diverse tasks. However, their efficiency as backbone architectures for geometric estimation tasks involving image deformations in low-data regimes remains an open question. This work considers two such tasks: 1) estimating 2D rigid transformations between pairs of images and 2) predicting the fundamental matrix for stereo image pairs, an important problem in various applications, such as autonomous mobility, robotics, and 3D scene reconstruction. Addressing this intriguing question, this work systematically compares large-scale CNNs (ResNet, EfficientNet, CLIP-ResNet) with ViT-based foundation models (CLIP-ViT variants and DINO) in various data size settings, including few-shot scenarios. These pretrained models are optimized for classification or contrastive learning, encouraging them to focus mostly on high-level semantics. The considered tasks require balancing local and global features differently, challenging the straightforward adoption of these models as the backbone. Empirical comparative analysis shows that, similar to training from scratch, ViTs outperform CNNs during refinement in large downstream-data scenarios. However, in small data scenarios, the inductive bias and smaller capacity of CNNs improve their performance, allowing them to match that of a ViT. Moreover, ViTs exhibit stronger generalization in cross-domain evaluation where the data distribution changes. These results emphasize the importance of carefully selecting model architectures for refinement, motivating future research towards hybrid architectures that balance local and global representations.
>
---
#### [new 039] Enhancing OCR for Sino-Vietnamese Language Processing via Fine-tuned PaddleOCRv5
- **分类: cs.CV; cs.CL; 68T50, 68T50, 68T10; I.2.7; I.5; I.7.5**

- **简介: 该论文属于OCR（光学字符识别）任务，旨在解决古籍中汉喃文识别准确率低的问题。通过微调PaddleOCRv5模型，使用古代越南汉字手稿数据集进行训练，提升了识别效果，尤其在噪声图像条件下准确率显著提高，并开发了可视化演示工具。**

- **链接: [http://arxiv.org/pdf/2510.04003v1](http://arxiv.org/pdf/2510.04003v1)**

> **作者:** Minh Hoang Nguyen; Su Nguyen Thiet
>
> **备注:** 5 pages, 6 figures, 2 tables
>
> **摘要:** Recognizing and processing Classical Chinese (Han-Nom) texts play a vital role in digitizing Vietnamese historical documents and enabling cross-lingual semantic research. However, existing OCR systems struggle with degraded scans, non-standard glyphs, and handwriting variations common in ancient sources. In this work, we propose a fine-tuning approach for PaddleOCRv5 to improve character recognition on Han-Nom texts. We retrain the text recognition module using a curated subset of ancient Vietnamese Chinese manuscripts, supported by a full training pipeline covering preprocessing, LMDB conversion, evaluation, and visualization. Experimental results show a significant improvement over the base model, with exact accuracy increasing from 37.5 percent to 50.0 percent, particularly under noisy image conditions. Furthermore, we develop an interactive demo that visually compares pre- and post-fine-tuning recognition results, facilitating downstream applications such as Han-Vietnamese semantic alignment, machine translation, and historical linguistics research. The demo is available at https://huggingface.co/spaces/MinhDS/Fine-tuned-PaddleOCRv5.
>
---
#### [new 040] ERDE: Entropy-Regularized Distillation for Early-exit
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像分类任务，旨在降低深度神经网络的计算成本。论文提出ERDE方法，结合早停机制与知识蒸馏，通过熵正则化损失优化学生模型训练。解决了模型压缩中准确率与效率的权衡问题，在CIFAR10、CIFAR100和SVHN数据集上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2510.04856v1](http://arxiv.org/pdf/2510.04856v1)**

> **作者:** Martial Guidez; Stefan Duffner; Yannick Alpou; Oscar Röth; Christophe Garcia
>
> **摘要:** Although deep neural networks and in particular Convolutional Neural Networks have demonstrated state-of-the-art performance in image classification with relatively high efficiency, they still exhibit high computational costs, often rendering them impractical for real-time and edge applications. Therefore, a multitude of compression techniques have been developed to reduce these costs while maintaining accuracy. In addition, dynamic architectures have been introduced to modulate the level of compression at execution time, which is a desirable property in many resource-limited application scenarios. The proposed method effectively integrates two well-established optimization techniques: early exits and knowledge distillation, where a reduced student early-exit model is trained from a more complex teacher early-exit model. The primary contribution of this research lies in the approach for training the student early-exit model. In comparison to the conventional Knowledge Distillation loss, our approach incorporates a new entropy-based loss for images where the teacher's classification was incorrect. The proposed method optimizes the trade-off between accuracy and efficiency, thereby achieving significant reductions in computational complexity without compromising classification performance. The validity of this approach is substantiated by experimental results on image classification datasets CIFAR10, CIFAR100 and SVHN, which further opens new research perspectives for Knowledge Distillation in other contexts.
>
---
#### [new 041] Zoom-In to Sort AI-Generated Images Out
- **分类: cs.CV; cs.AI; cs.CL; 68T45; I.2.10; I.2.7**

- **简介: 该论文属于图像真实性鉴别任务，旨在解决AI生成图像与真实图像难以区分的问题。作者提出ZoomIn框架，通过两阶段分析定位并分析可疑区域，结合VLM提升检测准确性和可解释性，并构建了MagniFake数据集支持训练，实现了高精度和可解释的图像鉴别。**

- **链接: [http://arxiv.org/pdf/2510.04225v1](http://arxiv.org/pdf/2510.04225v1)**

> **作者:** Yikun Ji; Yan Hong; Bowen Deng; jun lan; Huijia Zhu; Weiqiang Wang; Liqing Zhang; Jianfu Zhang
>
> **备注:** 9 pages, 6 images (19 pages, 11 figures including appendix)
>
> **摘要:** The rapid growth of AI-generated imagery has blurred the boundary between real and synthetic content, raising critical concerns for digital integrity. Vision-language models (VLMs) offer interpretability through explanations but often fail to detect subtle artifacts in high-quality synthetic images. We propose ZoomIn, a two-stage forensic framework that improves both accuracy and interpretability. Mimicking human visual inspection, ZoomIn first scans an image to locate suspicious regions and then performs a focused analysis on these zoomed-in areas to deliver a grounded verdict. To support training, we introduce MagniFake, a dataset of 20,000 real and high-quality synthetic images annotated with bounding boxes and forensic explanations, generated through an automated VLM-based pipeline. Our method achieves 96.39% accuracy with robust generalization, while providing human-understandable explanations grounded in visual evidence.
>
---
#### [new 042] Photorealistic Inpainting for Perturbation-based Explanations in Ecological Monitoring
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉与生态监测任务，旨在解决视觉模型预测不透明、缺乏可解释性的问题。作者提出一种基于图像修复的扰动解释方法，生成符合场景背景的局部编辑图像，用于揭示模型依赖的细粒度形态特征。实验在海豹检测模型上验证，通过扰动评分与专家评估，证明方法能提供生态合理的解释，提升模型可信度。**

- **链接: [http://arxiv.org/pdf/2510.03317v1](http://arxiv.org/pdf/2510.03317v1)**

> **作者:** Günel Aghakishiyeva; Jiayi Zhou; Saagar Arya; James David Poling; Holly R. Houliston; Jamie N. Womble; David W. Johnston; Brinnae Bent
>
> **备注:** Accepted to NeurIPS 2025 Imageomics Workshop
>
> **摘要:** Ecological monitoring is increasingly automated by vision models, yet opaque predictions limit trust and field adoption. We present an inpainting-guided, perturbation-based explanation technique that produces photorealistic, mask-localized edits that preserve scene context. Unlike masking or blurring, these edits stay in-distribution and reveal which fine-grained morphological cues drive predictions in tasks such as species recognition and trait attribution. We demonstrate the approach on a YOLOv9 detector fine-tuned for harbor seal detection in Glacier Bay drone imagery, using Segment-Anything-Model-refined masks to support two interventions: (i) object removal/replacement (e.g., replacing seals with plausible ice/water or boats) and (ii) background replacement with original animals composited onto new scenes. Explanations are assessed by re-scoring perturbed images (flip rate, confidence drop) and by expert review for ecological plausibility and interpretability. The resulting explanations localize diagnostic structures, avoid deletion artifacts common to traditional perturbations, and yield domain-relevant insights that support expert validation and more trustworthy deployment of AI in ecology.
>
---
#### [new 043] Neuroplastic Modular Framework: Cross-Domain Image Classification of Garbage and Industrial Surfaces
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决垃圾和工业表面缺陷分类问题。作者提出了一种神经可塑性模块化分类框架，结合ResNet-50和Vision Transformer提取多尺度特征，并引入FAISS相似性检索增强模型适应性。模型具有动态扩展的模块，提升跨领域场景的分类准确率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.05071v1](http://arxiv.org/pdf/2510.05071v1)**

> **作者:** Debojyoti Ghosh; Soumya K Ghosh; Adrijit Goswami
>
> **摘要:** Efficient and accurate classification of waste and industrial surface defects is essential for ensuring sustainable waste management and maintaining high standards in quality control. This paper introduces the Neuroplastic Modular Classifier, a novel hybrid architecture designed for robust and adaptive image classification in dynamic environments. The model combines a ResNet-50 backbone for localized feature extraction with a Vision Transformer (ViT) to capture global semantic context. Additionally, FAISS-based similarity retrieval is incorporated to provide a memory-like reference to previously encountered data, enriching the model's feature space. A key innovation of our architecture is the neuroplastic modular design composed of expandable, learnable blocks that dynamically grow during training when performance plateaus. Inspired by biological learning systems, this mechanism allows the model to adapt to data complexity over time, improving generalization. Beyond garbage classification, we validate the model on the Kolektor Surface Defect Dataset 2 (KolektorSDD2), which involves industrial defect detection on metal surfaces. Experimental results across domains show that the proposed architecture outperforms traditional static models in both accuracy and adaptability. The Neuroplastic Modular Classifier offers a scalable, high-performance solution for real-world image classification, with strong applicability in both environmental and industrial domains.
>
---
#### [new 044] Did you just see that? Arbitrary view synthesis for egocentric replay of operating room workflows from ambient sensors
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于**任意视角合成与手术室工作流重建任务**，旨在解决**手术室中缺乏第一视角视觉记录**的问题。作者提出**EgoSurg框架**，通过**几何驱动的神经渲染与扩散模型**，从固定摄像头视频中重建任意人员的**动态第一视角画面**，无需干扰手术流程。此方法提升了对手术决策、培训与流程优化的可视化分析能力。**

- **链接: [http://arxiv.org/pdf/2510.04802v1](http://arxiv.org/pdf/2510.04802v1)**

> **作者:** Han Zhang; Lalithkumar Seenivasan; Jose L. Porras; Roger D. Soberanis-Mukul; Hao Ding; Hongchao Shu; Benjamin D. Killeen; Ankita Ghosh; Lonny Yarmus; Masaru Ishii; Angela Christine Argento; Mathias Unberath
>
> **摘要:** Observing surgical practice has historically relied on fixed vantage points or recollections, leaving the egocentric visual perspectives that guide clinical decisions undocumented. Fixed-camera video can capture surgical workflows at the room-scale, but cannot reconstruct what each team member actually saw. Thus, these videos only provide limited insights into how decisions that affect surgical safety, training, and workflow optimization are made. Here we introduce EgoSurg, the first framework to reconstruct the dynamic, egocentric replays for any operating room (OR) staff directly from wall-mounted fixed-camera video, and thus, without intervention to clinical workflow. EgoSurg couples geometry-driven neural rendering with diffusion-based view enhancement, enabling high-visual fidelity synthesis of arbitrary and egocentric viewpoints at any moment. In evaluation across multi-site surgical cases and controlled studies, EgoSurg reconstructs person-specific visual fields and arbitrary viewpoints with high visual quality and fidelity. By transforming existing OR camera infrastructure into a navigable dynamic 3D record, EgoSurg establishes a new foundation for immersive surgical data science, enabling surgical practice to be visualized, experienced, and analyzed from every angle.
>
---
#### [new 045] GenAR: Next-Scale Autoregressive Generation for Spatial Gene Expression Prediction
- **分类: cs.CV**

- **简介: 论文提出GenAR，用于空间转录组（ST）中基因表达预测的任务。解决现有方法忽略基因共表达结构和连续回归导致生物不合理输出的问题。工作包括：多尺度自回归框架、基因分层聚类、离散token生成预测计数，并融合组织学与空间信息。**

- **链接: [http://arxiv.org/pdf/2510.04315v1](http://arxiv.org/pdf/2510.04315v1)**

> **作者:** Jiarui Ouyang; Yihui Wang; Yihang Gao; Yingxue Xu; Shu Yang; Hao Chen
>
> **摘要:** Spatial Transcriptomics (ST) offers spatially resolved gene expression but remains costly. Predicting expression directly from widely available Hematoxylin and Eosin (H&E) stained images presents a cost-effective alternative. However, most computational approaches (i) predict each gene independently, overlooking co-expression structure, and (ii) cast the task as continuous regression despite expression being discrete counts. This mismatch can yield biologically implausible outputs and complicate downstream analyses. We introduce GenAR, a multi-scale autoregressive framework that refines predictions from coarse to fine. GenAR clusters genes into hierarchical groups to expose cross-gene dependencies, models expression as codebook-free discrete token generation to directly predict raw counts, and conditions decoding on fused histological and spatial embeddings. From an information-theoretic perspective, the discrete formulation avoids log-induced biases and the coarse-to-fine factorization aligns with a principled conditional decomposition. Extensive experimental results on four Spatial Transcriptomics datasets across different tissue types demonstrate that GenAR achieves state-of-the-art performance, offering potential implications for precision medicine and cost-effective molecular profiling. Code is publicly available at https://github.com/oyjr/genar.
>
---
#### [new 046] A Novel Cloud-Based Diffusion-Guided Hybrid Model for High-Accuracy Accident Detection in Intelligent Transportation Systems
- **分类: cs.CV**

- **简介: 论文提出了一种基于扩散模型的新型混合模型，用于智能交通系统中的高精度事故检测。该研究属于图像分类任务，旨在解决传统方法在复杂数据分布下检测精度低的问题。作者结合ExceptionNet与扩散技术，构建了多条件模块，并通过云端实现高效处理，最终在公开数据集上达到97.32%的准确率。**

- **链接: [http://arxiv.org/pdf/2510.03675v1](http://arxiv.org/pdf/2510.03675v1)**

> **作者:** Siva Sai; Saksham Gupta; Vinay Chamola; Rajkumar Buyya
>
> **摘要:** The integration of Diffusion Models into Intelligent Transportation Systems (ITS) is a substantial improvement in the detection of accidents. We present a novel hybrid model integrating guidance classification with diffusion techniques. By leveraging fine-tuned ExceptionNet architecture outputs as input for our proposed diffusion model and processing image tensors as our conditioning, our approach creates a robust classification framework. Our model consists of multiple conditional modules, which aim to modulate the linear projection of inputs using time embeddings and image covariate embeddings, allowing the network to adapt its behavior dynamically throughout the diffusion process. To address the computationally intensive nature of diffusion models, our implementation is cloud-based, enabling scalable and efficient processing. Our strategy overcomes the shortcomings of conventional classification approaches by leveraging diffusion models inherent capacity to effectively understand complicated data distributions. We investigate important diffusion characteristics, such as timestep schedulers, timestep encoding techniques, timestep count, and architectural design changes, using a thorough ablation study, and have conducted a comprehensive evaluation of the proposed model against the baseline models on a publicly available dataset. The proposed diffusion model performs best in image-based accident detection with an accuracy of 97.32%.
>
---
#### [new 047] SSDD: Single-Step Diffusion Decoder for Efficient Image Tokenization
- **分类: cs.CV**

- **简介: 论文提出SSDD，一种基于扩散解码器的图像分词方法，旨在替代传统的KL-VAE。任务是提升生成模型的解码效率与质量，解决扩散解码器需对抗损失和迭代采样的问题。工作包括设计高效单步解码架构、使用蒸馏技术，实现更高重建质量与更快采样速度。**

- **链接: [http://arxiv.org/pdf/2510.04961v1](http://arxiv.org/pdf/2510.04961v1)**

> **作者:** Théophane Vallaeys; Jakob Verbeek; Matthieu Cord
>
> **摘要:** Tokenizers are a key component of state-of-the-art generative image models, extracting the most important features from the signal while reducing data dimension and redundancy. Most current tokenizers are based on KL-regularized variational autoencoders (KL-VAE), trained with reconstruction, perceptual and adversarial losses. Diffusion decoders have been proposed as a more principled alternative to model the distribution over images conditioned on the latent. However, matching the performance of KL-VAE still requires adversarial losses, as well as a higher decoding time due to iterative sampling. To address these limitations, we introduce a new pixel diffusion decoder architecture for improved scaling and training stability, benefiting from transformer components and GAN-free training. We use distillation to replicate the performance of the diffusion decoder in an efficient single-step decoder. This makes SSDD the first diffusion decoder optimized for single-step reconstruction trained without adversarial losses, reaching higher reconstruction quality and faster sampling than KL-VAE. In particular, SSDD improves reconstruction FID from $0.87$ to $0.50$ with $1.4\times$ higher throughput and preserve generation quality of DiTs with $3.8\times$ faster sampling. As such, SSDD can be used as a drop-in replacement for KL-VAE, and for building higher-quality and faster generative models.
>
---
#### [new 048] Cross-View Open-Vocabulary Object Detection in Aerial Imagery
- **分类: cs.CV**

- **简介: 该论文属于开放词汇目标检测任务，旨在解决传统模型无法识别未训练类别的问题。通过跨域对齐方法，将地面视角的预训练知识迁移到航拍图像中，提升了检测灵活性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.03858v1](http://arxiv.org/pdf/2510.03858v1)**

> **作者:** Jyoti Kini; Rohit Gupta; Mubarak Shah
>
> **摘要:** Traditional object detection models are typically trained on a fixed set of classes, limiting their flexibility and making it costly to incorporate new categories. Open-vocabulary object detection addresses this limitation by enabling models to identify unseen classes without explicit training. Leveraging pretrained models contrastively trained on abundantly available ground-view image-text classification pairs provides a strong foundation for open-vocabulary object detection in aerial imagery. Domain shifts, viewpoint variations, and extreme scale differences make direct knowledge transfer across domains ineffective, requiring specialized adaptation strategies. In this paper, we propose a novel framework for adapting open-vocabulary representations from ground-view images to solve object detection in aerial imagery through structured domain alignment. The method introduces contrastive image-to-image alignment to enhance the similarity between aerial and ground-view embeddings and employs multi-instance vocabulary associations to align aerial images with text embeddings. Extensive experiments on the xView, DOTAv2, VisDrone, DIOR, and HRRSD datasets are used to validate our approach. Our open-vocabulary model achieves improvements of +6.32 mAP on DOTAv2, +4.16 mAP on VisDrone (Images), and +3.46 mAP on HRRSD in the zero-shot setting when compared to finetuned closed-vocabulary dataset-specific model performance, thus paving the way for more flexible and scalable object detection systems in aerial applications.
>
---
#### [new 049] μDeepIQA: deep learning-based fast and robust image quality assessment with local predictions for optical microscopy
- **分类: cs.CV; physics.data-an; q-bio.QM**

- **简介: 该论文属于图像质量评估任务，旨在解决光学显微图像质量评估在速度、稳定性与局部预测方面的局限性。作者基于深度学习提出μDeepIQA，通过迁移学习重新训练模型，实现对光学显微图像的快速、稳健质量评估，并支持局部质量预测与可视化。**

- **链接: [http://arxiv.org/pdf/2510.04859v1](http://arxiv.org/pdf/2510.04859v1)**

> **作者:** Elena Corbetta; Thomas Bocklitz
>
> **备注:** 16 pages, 6 figures. \mu DeepIQA is publicly available at https://git.photonicdata.science/elena.corbetta/udeepiqa
>
> **摘要:** Optical microscopy is one of the most widely used techniques in research studies for life sciences and biomedicine. These applications require reliable experimental pipelines to extract valuable knowledge from the measured samples and must be supported by image quality assessment (IQA) to ensure correct processing and analysis of the image data. IQA methods are implemented with variable complexity. However, while most quality metrics have a straightforward implementation, they might be time consuming and computationally expensive when evaluating a large dataset. In addition, quality metrics are often designed for well-defined image features and may be unstable for images out of the ideal domain. To overcome these limitations, recent works have proposed deep learning-based IQA methods, which can provide superior performance, increased generalizability and fast prediction. Our method, named $\mathrm{\mu}$DeepIQA, is inspired by previous studies and applies a deep convolutional neural network designed for IQA on natural images to optical microscopy measurements. We retrained the same architecture to predict individual quality metrics and global quality scores for optical microscopy data. The resulting models provide fast and stable predictions of image quality by generalizing quality estimation even outside the ideal range of standard methods. In addition, $\mathrm{\mu}$DeepIQA provides patch-wise prediction of image quality and can be used to visualize spatially varying quality in a single image. Our study demonstrates that optical microscopy-based studies can benefit from the generalizability of deep learning models due to their stable performance in the presence of outliers, the ability to assess small image patches, and rapid predictions.
>
---
#### [new 050] BenthiCat: An opti-acoustic dataset for advancing benthic classification and habitat mapping
- **分类: cs.CV; cs.LG; I.2.6; I.4.6; I.5.1; I.5.4**

- **简介: 该论文属于水下生境制图任务，旨在解决缺乏大规模标注数据限制机器学习发展的瓶颈。作者构建了包含百万侧扫声呐瓦片、测深图及同步光学图像的多模态数据集BenthiCat，并提供标注掩码、原始数据和预处理工具，以支持模型训练、跨模态学习与算法开发，推动自主海底分类与多传感器融合研究。**

- **链接: [http://arxiv.org/pdf/2510.04876v1](http://arxiv.org/pdf/2510.04876v1)**

> **作者:** Hayat Rajani; Valerio Franchi; Borja Martinez-Clavel Valles; Raimon Ramos; Rafael Garcia; Nuno Gracias
>
> **备注:** Article under review by IJRR
>
> **摘要:** Benthic habitat mapping is fundamental for understanding marine ecosystems, guiding conservation efforts, and supporting sustainable resource management. Yet, the scarcity of large, annotated datasets limits the development and benchmarking of machine learning models in this domain. This paper introduces a thorough multi-modal dataset, comprising about a million side-scan sonar (SSS) tiles collected along the coast of Catalonia (Spain), complemented by bathymetric maps and a set of co-registered optical images from targeted surveys using an autonomous underwater vehicle (AUV). Approximately \num{36000} of the SSS tiles have been manually annotated with segmentation masks to enable supervised fine-tuning of classification models. All the raw sensor data, together with mosaics, are also released to support further exploration and algorithm development. To address challenges in multi-sensor data fusion for AUVs, we spatially associate optical images with corresponding SSS tiles, facilitating self-supervised, cross-modal representation learning. Accompanying open-source preprocessing and annotation tools are provided to enhance accessibility and encourage research. This resource aims to establish a standardized benchmark for underwater habitat mapping, promoting advancements in autonomous seafloor classification and multi-sensor integration.
>
---
#### [new 051] Convolutional Neural Nets vs Vision Transformers: A SpaceNet Case Study with Balanced vs Imbalanced Regimes
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文比较了卷积神经网络（EfficientNet-B0）与视觉Transformer（ViT-Base）在遥感图像分类任务（SpaceNet数据集）中的表现，探讨在类别平衡与不平衡情况下模型性能差异。研究旨在评估模型准确性、效率及类别分布影响，展示了CNN在效率上的优势，而类别平衡可缩小模型性能差距。**

- **链接: [http://arxiv.org/pdf/2510.03297v1](http://arxiv.org/pdf/2510.03297v1)**

> **作者:** Akshar Gothi
>
> **备注:** 5 pages, 1 figure, 9 tables. Code and artifacts: https://github.com/akshar27/spacenet-cnn-vs-vit (release v1.0.1)
>
> **摘要:** We present a controlled comparison of a convolutional neural network (EfficientNet-B0) and a Vision Transformer (ViT-Base) on SpaceNet under two label-distribution regimes: a naturally imbalanced five-class split and a balanced-resampled split with 700 images per class (70:20:10 train/val/test). With matched preprocessing (224x224, ImageNet normalization), lightweight augmentations, and a 40-epoch budget on a single NVIDIA P100, we report accuracy, macro-F1, balanced accuracy, per-class recall, and deployment metrics (model size and latency). On the imbalanced split, EfficientNet-B0 reaches 93% test accuracy with strong macro-F1 and lower latency; ViT-Base is competitive at 93% with a larger parameter count and runtime. On the balanced split, both models are strong; EfficientNet-B0 reaches 99% while ViT-Base remains competitive, indicating that balancing narrows architecture gaps while CNNs retain an efficiency edge. We release manifests, logs, and per-image predictions to support reproducibility.
>
---
#### [new 052] A.I.R.: Enabling Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering
- **分类: cs.CV**

- **简介: 该论文属于视频问答任务，旨在解决现有帧选择方法在准确性和计算成本间的权衡问题。作者提出A.I.R.方法，通过基于推理的迭代机制，使用强VLM进行语义分析，选择高潜力帧，兼顾性能与效率。**

- **链接: [http://arxiv.org/pdf/2510.04428v1](http://arxiv.org/pdf/2510.04428v1)**

> **作者:** Yuanhao Zou; Shengji Jin; Andong Deng; Youpeng Zhao; Jun Wang; Chen Chen
>
> **摘要:** Effectively applying Vision-Language Models (VLMs) to Video Question Answering (VideoQA) hinges on selecting a concise yet comprehensive set of frames, as processing entire videos is computationally infeasible. However, current frame selection methods face a critical trade-off: approaches relying on lightweight similarity models, such as CLIP, often fail to capture the nuances of complex queries, resulting in inaccurate similarity scores that cannot reflect the authentic query-frame relevance, which further undermines frame selection. Meanwhile, methods that leverage a VLM for deeper analysis achieve higher accuracy but incur prohibitive computational costs. To address these limitations, we propose A.I.R., a training-free approach for Adaptive, Iterative, and Reasoning-based frame selection. We leverage a powerful VLM to perform deep, semantic analysis on complex queries, and this analysis is deployed within a cost-effective iterative loop that processes only a small batch of the most high-potential frames at a time. Extensive experiments on various VideoQA benchmarks demonstrate that our approach outperforms existing frame selection methods, significantly boosts the performance of the foundation VLM, and achieves substantial gains in computational efficiency over other VLM-based techniques.
>
---
#### [new 053] Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction
- **分类: cs.CV; cs.AI**

- **简介: 本文属于3D占用预测任务，旨在解决现有方法在开放词汇场景建模中稀疏表示难以捕捉小物体、稠密表示计算开销大的问题。论文提出PG-Occ框架，采用渐进式高斯增强和各向异性感知采样策略，提升场景细节建模能力与计算效率，实现更精确的3D场景理解。**

- **链接: [http://arxiv.org/pdf/2510.04759v1](http://arxiv.org/pdf/2510.04759v1)**

> **作者:** Chi Yan; Dan Xu
>
> **备注:** Project Page: https://yanchi-3dv.github.io/PG-Occ
>
> **摘要:** The 3D occupancy prediction task has witnessed remarkable progress in recent years, playing a crucial role in vision-based autonomous driving systems. While traditional methods are limited to fixed semantic categories, recent approaches have moved towards predicting text-aligned features to enable open-vocabulary text queries in real-world scenes. However, there exists a trade-off in text-aligned scene modeling: sparse Gaussian representation struggles to capture small objects in the scene, while dense representation incurs significant computational overhead. To address these limitations, we present PG-Occ, an innovative Progressive Gaussian Transformer Framework that enables open-vocabulary 3D occupancy prediction. Our framework employs progressive online densification, a feed-forward strategy that gradually enhances the 3D Gaussian representation to capture fine-grained scene details. By iteratively enhancing the representation, the framework achieves increasingly precise and detailed scene understanding. Another key contribution is the introduction of an anisotropy-aware sampling strategy with spatio-temporal fusion, which adaptively assigns receptive fields to Gaussians at different scales and stages, enabling more effective feature aggregation and richer scene information capture. Through extensive evaluations, we demonstrate that PG-Occ achieves state-of-the-art performance with a relative 14.3% mIoU improvement over the previous best performing method. Code and pretrained models will be released upon publication on our project page: https://yanchi-3dv.github.io/PG-Occ
>
---
#### [new 054] SoC-DT: Standard-of-Care Aligned Digital Twins for Patient-Specific Tumor Dynamics
- **分类: cs.CV**

- **简介: 该论文属于医疗科技任务，旨在解决肿瘤治疗后动态预测的问题。现有模型难以准确模拟不同治疗方案下的肿瘤变化。论文提出了SoC-DT框架，结合基因组、人口统计和治疗个性化信息，统一反应扩散模型与治疗干预，实现更精准的肿瘤动态预测，优于传统模型。**

- **链接: [http://arxiv.org/pdf/2510.03287v1](http://arxiv.org/pdf/2510.03287v1)**

> **作者:** Moinak Bhattacharya; Gagandeep Singh; Prateek Prasanna
>
> **摘要:** Accurate prediction of tumor trajectories under standard-of-care (SoC) therapies remains a major unmet need in oncology. This capability is essential for optimizing treatment planning and anticipating disease progression. Conventional reaction-diffusion models are limited in scope, as they fail to capture tumor dynamics under heterogeneous therapeutic paradigms. There is hence a critical need for computational frameworks that can realistically simulate SoC interventions while accounting for inter-patient variability in genomics, demographics, and treatment regimens. We introduce Standard-of-Care Digital Twin (SoC-DT), a differentiable framework that unifies reaction-diffusion tumor growth models, discrete SoC interventions (surgery, chemotherapy, radiotherapy) along with genomic and demographic personalization to predict post-treatment tumor structure on imaging. An implicit-explicit exponential time-differencing solver, IMEX-SoC, is also proposed, which ensures stability, positivity, and scalability in SoC treatment situations. Evaluated on both synthetic data and real world glioma data, SoC-DT consistently outperforms classical PDE baselines and purely data-driven neural models in predicting tumor dynamics. By bridging mechanistic interpretability with modern differentiable solvers, SoC-DT establishes a principled foundation for patient-specific digital twins in oncology, enabling biologically consistent tumor dynamics estimation. Code will be made available upon acceptance.
>
---
#### [new 055] Diffusion^2: Dual Diffusion Model with Uncertainty-Aware Adaptive Noise for Momentary Trajectory Prediction
- **分类: cs.CV**

- **简介: 该论文属于轨迹预测任务，旨在解决观测数据不足时的行人轨迹预测问题。提出了Diffusion²框架，通过双向扩散模型生成未观测的历史轨迹并预测未来轨迹，引入不确定性估计与自适应噪声机制，提升了预测准确性，在多个数据集上取得最优效果。**

- **链接: [http://arxiv.org/pdf/2510.04365v1](http://arxiv.org/pdf/2510.04365v1)**

> **作者:** Yuhao Luo; Yuang Zhang; Kehua Chen; Xinyu Zheng; Shucheng Zhang; Sikai Chen; Yinhai Wang
>
> **备注:** 13 pages, 7 figures, 3 tables
>
> **摘要:** Accurate pedestrian trajectory prediction is crucial for ensuring safety and efficiency in autonomous driving and human-robot interaction scenarios. Earlier studies primarily utilized sufficient observational data to predict future trajectories. However, in real-world scenarios, such as pedestrians suddenly emerging from blind spots, sufficient observational data is often unavailable (i.e. momentary trajectory), making accurate prediction challenging and increasing the risk of traffic accidents. Therefore, advancing research on pedestrian trajectory prediction under extreme scenarios is critical for enhancing traffic safety. In this work, we propose a novel framework termed Diffusion^2, tailored for momentary trajectory prediction. Diffusion^2 consists of two sequentially connected diffusion models: one for backward prediction, which generates unobserved historical trajectories, and the other for forward prediction, which forecasts future trajectories. Given that the generated unobserved historical trajectories may introduce additional noise, we propose a dual-head parameterization mechanism to estimate their aleatoric uncertainty and design a temporally adaptive noise module that dynamically modulates the noise scale in the forward diffusion process. Empirically, Diffusion^2 sets a new state-of-the-art in momentary trajectory prediction on ETH/UCY and Stanford Drone datasets.
>
---
#### [new 056] Quantization Range Estimation for Convolutional Neural Networks
- **分类: cs.CV; cs.AI; 00-01; I.2.6; K.3.2**

- **简介: 该论文属于模型压缩任务，旨在解决卷积神经网络在低比特量化时保持精度的问题。作者提出了一种基于优化量化误差的范围估计方法，并通过局部凸性证明和高效搜索算法提升量化性能，尤其在8-bit以下设置中显著改善了图像分类准确率。**

- **链接: [http://arxiv.org/pdf/2510.04044v1](http://arxiv.org/pdf/2510.04044v1)**

> **作者:** Bingtao Yang; Yujia Wang; Mengzhi Jiao; Hongwei Huo
>
> **备注:** 11 pages, 5 tables, research report
>
> **摘要:** Post-training quantization for reducing the storage of deep neural network models has been demonstrated to be an effective way in various tasks. However, low-bit quantization while maintaining model accuracy is a challenging problem. In this paper, we present a range estimation method to improve the quantization performance for post-training quantization. We model the range estimation into an optimization problem of minimizing quantization errors by layer-wise local minima. We prove this problem is locally convex and present an efficient search algorithm to find the optimal solution. We propose the application of the above search algorithm to the transformed weights space to do further improvement in practice. Our experiments demonstrate that our method outperforms state-of-the-art performance generally on top-1 accuracy for image classification tasks on the ResNet series models and Inception-v3 model. The experimental results show that the proposed method has almost no loss of top-1 accuracy in 8-bit and 6-bit settings for image classifications, and the accuracy of 4-bit quantization is also significantly improved. The code is available at https://github.com/codeiscommitting/REQuant.
>
---
#### [new 057] MedCLM: Learning to Localize and Reason via a CoT-Curriculum in Medical Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医学视觉-语言模型任务，旨在解决临床诊断推理与AI结合的挑战。作者提出了MedCLM，通过将检测数据转化为具有链式推理的视觉问答数据，结合逐步学习策略，提升模型在医学视觉问答基准上的性能，实现可扩展的、与临床对齐的模型开发。**

- **链接: [http://arxiv.org/pdf/2510.04477v1](http://arxiv.org/pdf/2510.04477v1)**

> **作者:** Soo Yong Kim; Suin Cho; Vincent-Daniel Yun; Gyeongyeon Hwang
>
> **摘要:** Bridging clinical diagnostic reasoning with AI remains a central challenge in medical imaging. We introduce MedCLM, an automated pipeline that converts detection datasets into large-scale medical visual question answering (VQA) data with Chain-of-Thought (CoT) reasoning by linking lesion boxes to organ segmentation and structured rationales. These contextual signals enable medical vision-language models to generate question-answer pairs with step-by-step reasoning. To utilize this data effectively, we propose an Integrated CoT-Curriculum Strategy composed of an Easy stage with explicit lesion boxes for visual grounding, a Medium stage that encourages implicit localization, and a Hard stage for weakly supervised reasoning. Experimental results demonstrate that MedCLM attains state-of-the-art performance on several medical VQA benchmarks, providing a scalable framework for developing clinically aligned medical vision-language models.
>
---
#### [new 058] Adaptively Sampling-Reusing-Mixing Decomposed Gradients to Speed Up Sharpness Aware Minimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于优化算法任务，旨在解决Sharpness-Aware Minimization（SAM）计算成本高的问题。作者提出ARSAM，通过分解梯度、重用和混合策略，在保持模型泛化能力的同时显著加速训练过程。实验表明其在多种任务上具有40%左右的加速效果且性能相当。**

- **链接: [http://arxiv.org/pdf/2510.03763v1](http://arxiv.org/pdf/2510.03763v1)**

> **作者:** Jiaxin Deng; Junbiao Pang
>
> **摘要:** Sharpness-Aware Minimization (SAM) improves model generalization but doubles the computational cost of Stochastic Gradient Descent (SGD) by requiring twice the gradient calculations per optimization step. To mitigate this, we propose Adaptively sampling-Reusing-mixing decomposed gradients to significantly accelerate SAM (ARSAM). Concretely, we firstly discover that SAM's gradient can be decomposed into the SGD gradient and the Projection of the Second-order gradient onto the First-order gradient (PSF). Furthermore, we observe that the SGD gradient and PSF dynamically evolve during training, emphasizing the growing role of the PSF to achieve a flat minima. Therefore, ARSAM is proposed to the reused PSF and the timely updated PSF still maintain the model's generalization ability. Extensive experiments show that ARSAM achieves state-of-the-art accuracies comparable to SAM across diverse network architectures. On CIFAR-10/100, ARSAM is comparable to SAM while providing a speedup of about 40\%. Moreover, ARSAM accelerates optimization for the various challenge tasks (\textit{e.g.}, human pose estimation, and model quantization) without sacrificing performance, demonstrating its broad practicality.% The code is publicly accessible at: https://github.com/ajiaaa/ARSAM.
>
---
#### [new 059] Unsupervised Transformer Pre-Training for Images: Self-Distillation, Mean Teachers, and Random Crops
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于计算机视觉任务，旨在通过自监督学习提升图像特征表示。论文分析了DINOv2方法的核心思想，包括多裁剪增强和自蒸馏技术，比较了其与现有SSL和WSL方法的性能，并探讨了其局限性与未来方向。**

- **链接: [http://arxiv.org/pdf/2510.03606v1](http://arxiv.org/pdf/2510.03606v1)**

> **作者:** Mattia Scardecchia
>
> **摘要:** Recent advances in self-supervised learning (SSL) have made it possible to learn general-purpose visual features that capture both the high-level semantics and the fine-grained spatial structure of images. Most notably, the recent DINOv2 has established a new state of the art by surpassing weakly supervised methods (WSL) like OpenCLIP on most benchmarks. In this survey, we examine the core ideas behind its approach, multi-crop view augmentation and self-distillation with a mean teacher, and trace their development in previous work. We then compare the performance of DINO and DINOv2 with other SSL and WSL methods across various downstream tasks, and highlight some remarkable emergent properties of their learned features with transformer backbones. We conclude by briefly discussing DINOv2's limitations, its impact, and future research directions.
>
---
#### [new 060] Generating Human Motion Videos using a Cascaded Text-to-Video Framework
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决基于文本生成多样化人类运动视频的问题。现有方法受限于输入条件或领域，本文提出CAMEO框架，结合文本到动作与视频扩散模型，优化训练与推理过程，实现更通用、连贯且符合描述的人类动作视频生成。**

- **链接: [http://arxiv.org/pdf/2510.03909v1](http://arxiv.org/pdf/2510.03909v1)**

> **作者:** Hyelin Nam; Hyojun Go; Byeongjun Park; Byung-Hoon Kim; Hyungjin Chung
>
> **备注:** 18 pages, 7 figures, Project Page:https://hyelinnam.github.io/Cameo/
>
> **摘要:** Human video generation is becoming an increasingly important task with broad applications in graphics, entertainment, and embodied AI. Despite the rapid progress of video diffusion models (VDMs), their use for general-purpose human video generation remains underexplored, with most works constrained to image-to-video setups or narrow domains like dance videos. In this work, we propose CAMEO, a cascaded framework for general human motion video generation. It seamlessly bridges Text-to-Motion (T2M) models and conditional VDMs, mitigating suboptimal factors that may arise in this process across both training and inference through carefully designed components. Specifically, we analyze and prepare both textual prompts and visual conditions to effectively train the VDM, ensuring robust alignment between motion descriptions, conditioning signals, and the generated videos. Furthermore, we introduce a camera-aware conditioning module that connects the two stages, automatically selecting viewpoints aligned with the input text to enhance coherence and reduce manual intervention. We demonstrate the effectiveness of our approach on both the MovieGen benchmark and a newly introduced benchmark tailored to the T2M-VDM combination, while highlighting its versatility across diverse use cases.
>
---
#### [new 061] SFANet: Spatial-Frequency Attention Network for Deepfake Detection
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于图像处理与深度伪造检测任务，旨在解决现有方法在跨数据集和生成技术上泛化能力差的问题。作者提出SFANet，结合Transformer和纹理方法，引入数据分割、顺序训练、频率分离等技术，提升检测准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.04630v1](http://arxiv.org/pdf/2510.04630v1)**

> **作者:** Vrushank Ahire; Aniruddh Muley; Shivam Zample; Siddharth Verma; Pranav Menon; Surbhi Madan; Abhinav Dhall
>
> **摘要:** Detecting manipulated media has now become a pressing issue with the recent rise of deepfakes. Most existing approaches fail to generalize across diverse datasets and generation techniques. We thus propose a novel ensemble framework, combining the strengths of transformer-based architectures, such as Swin Transformers and ViTs, and texture-based methods, to achieve better detection accuracy and robustness. Our method introduces innovative data-splitting, sequential training, frequency splitting, patch-based attention, and face segmentation techniques to handle dataset imbalances, enhance high-impact regions (e.g., eyes and mouth), and improve generalization. Our model achieves state-of-the-art performance when tested on the DFWild-Cup dataset, a diverse subset of eight deepfake datasets. The ensemble benefits from the complementarity of these approaches, with transformers excelling in global feature extraction and texturebased methods providing interpretability. This work demonstrates that hybrid models can effectively address the evolving challenges of deepfake detection, offering a robust solution for real-world applications.
>
---
#### [new 062] SPEGNet: Synergistic Perception-Guided Network for Camouflaged Object Detection
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于计算机视觉任务，旨在解决伪装物体检测问题。现有方法依赖复杂组件累积，导致计算负担且丢失细节。论文提出SPEGNet，通过统一设计融合多尺度特征，实现边界精准与区域一致性的平衡，提升了检测性能。**

- **链接: [http://arxiv.org/pdf/2510.04472v1](http://arxiv.org/pdf/2510.04472v1)**

> **作者:** Baber Jan; Saeed Anwar; Aiman H. El-Maleh; Abdul Jabbar Siddiqui; Abdul Bais
>
> **摘要:** Camouflaged object detection segments objects with intrinsic similarity and edge disruption. Current detection methods rely on accumulated complex components. Each approach adds components such as boundary modules, attention mechanisms, and multi-scale processors independently. This accumulation creates a computational burden without proportional gains. To manage this complexity, they process at reduced resolutions, eliminating fine details essential for camouflage. We present SPEGNet, addressing fragmentation through a unified design. The architecture integrates multi-scale features via channel calibration and spatial enhancement. Boundaries emerge directly from context-rich representations, maintaining semantic-spatial alignment. Progressive refinement implements scale-adaptive edge modulation with peak influence at intermediate resolutions. This design strikes a balance between boundary precision and regional consistency. SPEGNet achieves 0.887 $S_\alpha$ on CAMO, 0.890 on COD10K, and 0.895 on NC4K, with real-time inference speed. Our approach excels across scales, from tiny, intricate objects to large, pattern-similar ones, while handling occlusion and ambiguous boundaries. Code, model weights, and results are available on \href{https://github.com/Baber-Jan/SPEGNet}{https://github.com/Baber-Jan/SPEGNet}.
>
---
#### [new 063] LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）模型评估任务，旨在解决当前LIBERO基准测试中模型性能评估不准确、易受记忆影响的问题。作者提出了LIBERO-PRO，通过在多个维度引入合理扰动，更严格地评估模型的泛化与理解能力。实验表明，现有模型在新设置下性能大幅下降，揭示其依赖记忆而非真正理解任务。**

- **链接: [http://arxiv.org/pdf/2510.03827v1](http://arxiv.org/pdf/2510.03827v1)**

> **作者:** Xueyang Zhou; Yangming Xu; Guiyao Tie; Yongchao Chen; Guowen Zhang; Duanfeng Chu; Pan Zhou; Lichao Sun
>
> **备注:** 12 pages,7 figures, 5 tables
>
> **摘要:** LIBERO has emerged as a widely adopted benchmark for evaluating Vision-Language-Action (VLA) models; however, its current training and evaluation settings are problematic, often leading to inflated performance estimates and preventing fair model comparison. To address these issues, we introduce LIBERO-PRO, an extended LIBERO benchmark that systematically evaluates model performance under reasonable perturbations across four dimensions: manipulated objects, initial states, task instructions, and environments. Experimental results reveal that, although existing models achieve over 90% accuracy under the standard LIBERO evaluation, their performance collapses to 0.0% under our generalized setting. Crucially, this discrepancy exposes the models' reliance on rote memorization of action sequences and environment layouts from the training set, rather than genuine task understanding or environmental perception. For instance, models persist in executing grasping actions when the target object is replaced with irrelevant items, and their outputs remain unchanged even when given corrupted instructions or even messy tokens. These findings expose the severe flaws in current evaluation practices, and we call on the community to abandon misleading methodologies in favor of robust assessments of model generalization and comprehension. Our code is available at: https://github.com/Zxy-MLlab/LIBERO-PRO.
>
---
#### [new 064] Skin Lesion Classification Based on ResNet-50 Enhanced With Adaptive Spatial Feature Fusion
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决皮肤癌病变分类中类间相似、类内差异大及图像噪声干扰问题。论文改进ResNet-50模型，引入自适应空间特征融合机制，提升特征表达并减少过拟合，实验表明其分类性能优于经典CNN模型。**

- **链接: [http://arxiv.org/pdf/2510.03876v1](http://arxiv.org/pdf/2510.03876v1)**

> **作者:** Runhao Liu; Ziming Chen; Peng Zhang
>
> **摘要:** Skin cancer classification remains a challenging problem due to high inter-class similarity, intra-class variability, and image noise in dermoscopic images. To address these issues, we propose an improved ResNet-50 model enhanced with Adaptive Spatial Feature Fusion (ASFF), which adaptively integrates multi-scale semantic and surface features to improve feature representation and reduce overfitting. The ResNet-50 model is enhanced with an adaptive feature fusion mechanism to achieve more effective multi-scale feature extraction and improve overall performance. Specifically, a dual-branch design fuses high-level semantic and mid-level detail features, which are processed through global average pooling and fully connected layers to generate adaptive weights for weighted fusion, thereby strengthening feature learning and reducing the impact of noise on classification. The method is evaluated on a subset of the ISIC 2020 dataset containing 3297 benign and malignant skin lesion images. Experimental results show that the proposed ASFF-based ResNet-50 achieves the best overall performance compared with 5 classic convolutional neural networks (CNNs) models. The proposed model reached an accuracy of 93.18% along with higher precision, recall, specificity, and F1 score. The improved model achieves an AUC value of 0.9670 and 0.9717 in the P-R and ROC curve, respectively. Then, the evaluation based on Grad-CAM further proved that the improved model adaptively focuses on lesion-relevant regions while suppressing irrelevant background information, thereby validating its enhanced feature learning capability from a deep representation perspective. These findings demonstrate that the proposed approach provides a more effective and efficient solution for computer-aided skin cancer diagnosis.
>
---
#### [new 065] A Semantics-Aware Hierarchical Self-Supervised Approach to Classification of Remote Sensing Images
- **分类: cs.CV; I.4.6; I.4.8; I.4.10**

- **简介: 该论文属于遥感图像分类任务，旨在解决现有方法忽略类别语义层次关系的问题。作者提出了一种语义感知的层次共识方法（SAHC），通过引入可训练层次矩阵和层次共识机制，使网络在自监督下学习层次结构，提升分类效果。**

- **链接: [http://arxiv.org/pdf/2510.04916v1](http://arxiv.org/pdf/2510.04916v1)**

> **作者:** Giulio Weikmann; Gianmarco Perantoni; Lorenzo Bruzzone
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Deep learning has become increasingly important in remote sensing image classification due to its ability to extract semantic information from complex data. Classification tasks often include predefined label hierarchies that represent the semantic relationships among classes. However, these hierarchies are frequently overlooked, and most approaches focus only on fine-grained classification schemes. In this paper, we present a novel Semantics-Aware Hierarchical Consensus (SAHC) method for learning hierarchical features and relationships by integrating hierarchy-specific classification heads within a deep network architecture, each specialized in different degrees of class granularity. The proposed approach employs trainable hierarchy matrices, which guide the network through the learning of the hierarchical structure in a self-supervised manner. Furthermore, we introduce a hierarchical consensus mechanism to ensure consistent probability distributions across different hierarchical levels. This mechanism acts as a weighted ensemble being able to effectively leverage the inherent structure of the hierarchical classification task. The proposed SAHC method is evaluated on three benchmark datasets with different degrees of hierarchical complexity on different tasks, using distinct backbone architectures to effectively emphasize its adaptability. Experimental results show both the effectiveness of the proposed approach in guiding network learning and the robustness of the hierarchical consensus for remote sensing image classification tasks.
>
---
#### [new 066] Visual Odometry with Transformers
- **分类: cs.CV**

- **简介: 该论文属于视觉里程计任务，旨在解决单目视觉里程计中依赖复杂手工组件和难以泛化的问题。论文提出VoT模型，采用Transformer结构，通过时空注意力机制处理单目视频序列，实现端到端的相机运动预测，无需特征匹配、稠密重建等手工模块，提升了速度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.03348v1](http://arxiv.org/pdf/2510.03348v1)**

> **作者:** Vlardimir Yugay; Duy-Kien Nguyen; Theo Gevers; Cees G. M. Snoek; Martin R. Oswald
>
> **摘要:** Modern monocular visual odometry methods typically combine pre-trained deep learning components with optimization modules, resulting in complex pipelines that rely heavily on camera calibration and hyperparameter tuning, and often struggle in unseen real-world scenarios. Recent large-scale 3D models trained on massive amounts of multi-modal data have partially alleviated these challenges, providing generalizable dense reconstruction and camera pose estimation. Still, they remain limited in handling long videos and providing accurate per-frame estimates, which are required for visual odometry. In this work, we demonstrate that monocular visual odometry can be addressed effectively in an end-to-end manner, thereby eliminating the need for handcrafted components such as bundle adjustment, feature matching, camera calibration, or dense 3D reconstruction. We introduce VoT, short for Visual odometry Transformer, which processes sequences of monocular frames by extracting features and modeling global relationships through temporal and spatial attention. Unlike prior methods, VoT directly predicts camera motion without estimating dense geometry and relies solely on camera poses for supervision. The framework is modular and flexible, allowing seamless integration of various pre-trained encoders as feature extractors. Experimental results demonstrate that VoT scales effectively with larger datasets, benefits substantially from stronger pre-trained backbones, generalizes across diverse camera motions and calibration settings, and outperforms traditional methods while running more than 3 times faster. The code will be released.
>
---
#### [new 067] Comparative Analysis of YOLOv5, Faster R-CNN, SSD, and RetinaNet for Motorbike Detection in Kigali Autonomous Driving Context
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于目标检测任务，旨在解决卢旺达基加利市自动驾驶中摩托车识别问题。作者使用自建数据集比较了YOLOv5、Faster R-CNN、SSD和RetinaNet四种模型在准确率、定位和推理速度上的表现，探讨其在资源受限环境下的适用性，并提出未来需优化模型复杂度以提升在发展中国家自动驾驶系统中的实用性。**

- **链接: [http://arxiv.org/pdf/2510.04912v1](http://arxiv.org/pdf/2510.04912v1)**

> **作者:** Ngeyen Yinkfu; Sunday Nwovu; Jonathan Kayizzi; Angelique Uwamahoro
>
> **备注:** 3 figures, 2 tables
>
> **摘要:** In Kigali, Rwanda, motorcycle taxis are a primary mode of transportation, often navigating unpredictably and disregarding traffic rules, posing significant challenges for autonomous driving systems. This study compares four object detection models--YOLOv5, Faster R-CNN, SSD, and RetinaNet--for motorbike detection using a custom dataset of 198 images collected in Kigali. Implemented in PyTorch with transfer learning, the models were evaluated for accuracy, localization, and inference speed to assess their suitability for real-time navigation in resource-constrained settings. We identify implementation challenges, including dataset limitations and model complexities, and recommend simplified architectures for future work to enhance accessibility for autonomous systems in developing countries like Rwanda.
>
---
#### [new 068] SketchPlan: Diffusion Based Drone Planning From Human Sketches
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机路径规划任务，旨在解决从人类手绘草图生成安全飞行路径的问题。论文提出SketchPlan，结合草图解析与扩散模型，实现从2D草图到3D路径的零样本跨真实环境生成，通过合成数据与真实草图训练提升性能。**

- **链接: [http://arxiv.org/pdf/2510.03545v1](http://arxiv.org/pdf/2510.03545v1)**

> **作者:** Sixten Norelius; Aaron O. Feldman; Mac Schwager
>
> **备注:** Code available at https://github.com/sixnor/SketchPlan
>
> **摘要:** We propose SketchPlan, a diffusion-based planner that interprets 2D hand-drawn sketches over depth images to generate 3D flight paths for drone navigation. SketchPlan comprises two components: a SketchAdapter that learns to map the human sketches to projected 2D paths, and DiffPath, a diffusion model that infers 3D trajectories from 2D projections and a first person view depth image. Our model achieves zero-shot sim-to-real transfer, generating accurate and safe flight paths in previously unseen real-world environments. To train the model, we build a synthetic dataset of 32k flight paths using a diverse set of photorealistic 3D Gaussian Splatting scenes. We automatically label the data by computing 2D projections of the 3D flight paths onto the camera plane, and use this to train the DiffPath diffusion model. However, since real human 2D sketches differ significantly from ideal 2D projections, we additionally label 872 of the 3D flight paths with real human sketches and use this to train the SketchAdapter to infer the 2D projection from the human sketch. We demonstrate SketchPlan's effectiveness in both simulated and real-world experiments, and show through ablations that training on a mix of human labeled and auto-labeled data together with a modular design significantly boosts its capabilities to correctly interpret human intent and infer 3D paths. In real-world drone tests, SketchPlan achieved 100\% success in low/medium clutter and 40\% in unseen high-clutter environments, outperforming key ablations by 20-60\% in task completion.
>
---
#### [new 069] The Overlooked Value of Test-time Reference Sets in Visual Place Recognition
- **分类: cs.CV**

- **简介: 该论文属于视觉地点识别（VPR）任务，旨在解决测试环境与训练数据差异大时识别性能下降的问题。作者提出在测试时利用已知的参考图像集（地图）对模型进行微调（RSF），提升现有最先进方法在挑战性数据集上的表现，平均Recall@1提升约2.3%，同时保持模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.03751v1](http://arxiv.org/pdf/2510.03751v1)**

> **作者:** Mubariz Zaffar; Liangliang Nan; Sebastian Scherer; Julian F. P. Kooij
>
> **备注:** Accepted at ICCV 2025 Workshop CrocoDL
>
> **摘要:** Given a query image, Visual Place Recognition (VPR) is the task of retrieving an image of the same place from a reference database with robustness to viewpoint and appearance changes. Recent works show that some VPR benchmarks are solved by methods using Vision-Foundation-Model backbones and trained on large-scale and diverse VPR-specific datasets. Several benchmarks remain challenging, particularly when the test environments differ significantly from the usual VPR training datasets. We propose a complementary, unexplored source of information to bridge the train-test domain gap, which can further improve the performance of State-of-the-Art (SOTA) VPR methods on such challenging benchmarks. Concretely, we identify that the test-time reference set, the "map", contains images and poses of the target domain, and must be available before the test-time query is received in several VPR applications. Therefore, we propose to perform simple Reference-Set-Finetuning (RSF) of VPR models on the map, boosting the SOTA (~2.3% increase on average for Recall@1) on these challenging datasets. Finetuned models retain generalization, and RSF works across diverse test datasets.
>
---
#### [new 070] Hands-Free Heritage: Automated 3D Scanning for Cultural Heritage Digitization
- **分类: cs.CV**

- **简介: 该论文属于文化遗产数字化任务，旨在解决传统3D扫描依赖人工操作的问题。作者设计了一个双机器人自动化扫描系统，通过协调运动规划和优化轨迹，实现高效、高精度的文物3D重建，减少对专家的依赖。**

- **链接: [http://arxiv.org/pdf/2510.04781v1](http://arxiv.org/pdf/2510.04781v1)**

> **作者:** Javed Ahmad; Federico Dassiè; Selene Frascella; Gabriele Marchello; Ferdinando Cannella; Arianna Traviglia
>
> **备注:** 9 pages
>
> **摘要:** High-fidelity 3D scanning is essential for preserving cultural heritage artefacts, supporting documentation, analysis, and long-term conservation. However, conventional methods typically require specialized expertise and manual intervention to maintain optimal scanning conditions and coverage. We present an automated two-robot scanning system that eliminates the need for handheld or semi-automatic workflows by combining coordinated robotic manipulation with high-resolution 3D scanning. Our system parameterizes the scanning space into distinct regions, enabling coordinated motion planning between a scanner-equipped robot and a tray-handling robot. Optimized trajectory planning and waypoint distribution ensure comprehensive surface coverage, minimize occlusions, and balance reconstruction accuracy with system efficiency. Experimental results show that our approach achieves significantly lower Chamfer Distance and higher F-score compared to baseline methods, offering superior geometric accuracy, improved digitization efficiency, and reduced reliance on expert operators.
>
---
#### [new 071] Conditional Representation Learning for Customized Tasks
- **分类: cs.CV**

- **简介: 该论文属于**表示学习**任务，旨在解决**通用表示难以适配定制化下游任务**的问题。现有方法依赖监督微调，成本高。论文提出**条件表示学习**（CRL），通过大语言模型生成描述文本，构建定制特征空间，再利用视觉-语言模型将图像表示投影其中，提升特定任务性能。实验表明其有效性与通用性。**

- **链接: [http://arxiv.org/pdf/2510.04564v1](http://arxiv.org/pdf/2510.04564v1)**

> **作者:** Honglin Liu; Chao Sun; Peng Hu; Yunfan Li; Xi Peng
>
> **摘要:** Conventional representation learning methods learn a universal representation that primarily captures dominant semantics, which may not always align with customized downstream tasks. For instance, in animal habitat analysis, researchers prioritize scene-related features, whereas universal embeddings emphasize categorical semantics, leading to suboptimal results. As a solution, existing approaches resort to supervised fine-tuning, which however incurs high computational and annotation costs. In this paper, we propose Conditional Representation Learning (CRL), aiming to extract representations tailored to arbitrary user-specified criteria. Specifically, we reveal that the semantics of a space are determined by its basis, thereby enabling a set of descriptive words to approximate the basis for a customized feature space. Building upon this insight, given a user-specified criterion, CRL first employs a large language model (LLM) to generate descriptive texts to construct the semantic basis, then projects the image representation into this conditional feature space leveraging a vision-language model (VLM). The conditional representation better captures semantics for the specific criterion, which could be utilized for multiple customized tasks. Extensive experiments on classification and retrieval tasks demonstrate the superiority and generality of the proposed CRL. The code is available at https://github.com/XLearning-SCU/2025-NeurIPS-CRL.
>
---
#### [new 072] Detection of retinal diseases using an accelerated reused convolutional network
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决视网膜疾病检测中模型计算复杂、难以普及的问题。作者通过优化卷积层，提出ArConv层，构建了一个参数更少、精度更高的轻量化模型，适用于移动端部署。**

- **链接: [http://arxiv.org/pdf/2510.04232v1](http://arxiv.org/pdf/2510.04232v1)**

> **作者:** Amin Ahmadi Kasani; Hedieh Sajedi
>
> **摘要:** Convolutional neural networks are continually evolving, with some efforts aimed at improving accuracy, others at increasing speed, and some at enhancing accessibility. Improving accessibility broadens the application of neural networks across a wider range of tasks, including the detection of eye diseases. Early diagnosis of eye diseases and consulting an ophthalmologist can prevent many vision disorders. Given the importance of this issue, various datasets have been collected from the cornea to facilitate the process of making neural network models. However, most of the methods introduced in the past are computationally complex. In this study, we tried to increase the accessibility of deep neural network models. We did this at the most fundamental level, specifically by redesigning and optimizing the convolutional layers. By doing so, we created a new general model that incorporates our novel convolutional layer named ArConv layers. Thanks to the efficient performance of this new layer, the model has suitable complexity for use in mobile phones and can perform the task of diagnosing the presence of disease with high accuracy. The final model we present contains only 1.3 million parameters. In comparison to the MobileNetV2 model, which has 2.2 million parameters, our model demonstrated better accuracy when trained and evaluated on the RfMiD dataset under identical conditions, achieving an accuracy of 0.9328 versus 0.9266 on the RfMiD test set.
>
---
#### [new 073] A Hybrid Co-Finetuning Approach for Visual Bug Detection in Video Games
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉缺陷检测任务，旨在解决视频游戏中视觉缺陷识别依赖大量标注数据的问题。作者提出了一种混合协同微调方法，结合有标签和无标签数据，提升特征学习效果。实验表明，该方法在多种游戏中表现优异，即使使用50%的标注数据仍保持良好性能。**

- **链接: [http://arxiv.org/pdf/2510.03591v1](http://arxiv.org/pdf/2510.03591v1)**

> **作者:** Faliu Yi; Sherif Abdelfattah; Wei Huang; Adrian Brown
>
> **备注:** Accepted at the 21st AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE 2025)
>
> **摘要:** Manual identification of visual bugs in video games is a resource-intensive and costly process, often demanding specialized domain knowledge. While supervised visual bug detection models offer a promising solution, their reliance on extensive labeled datasets presents a significant challenge due to the infrequent occurrence of such bugs. To overcome this limitation, we propose a hybrid Co-FineTuning (CFT) method that effectively integrates both labeled and unlabeled data. Our approach leverages labeled samples from the target game and diverse co-domain games, additionally incorporating unlabeled data to enhance feature representation learning. This strategy maximizes the utility of all available data, substantially reducing the dependency on labeled examples from the specific target game. The developed framework demonstrates enhanced scalability and adaptability, facilitating efficient visual bug detection across various game titles. Our experimental results show the robustness of the proposed method for game visual bug detection, exhibiting superior performance compared to conventional baselines across multiple gaming environments. Furthermore, CFT maintains competitive performance even when trained with only 50% of the labeled data from the target game.
>
---
#### [new 074] Person-Centric Annotations of LAION-400M: Auditing Bias and Its Transfer to Models
- **分类: cs.CV; cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于视觉-语言模型偏见分析任务，旨在解决训练数据中潜在的偏见问题。作者为LAION-400M数据集创建了以人物为中心的注释，包括性别、种族/民族标签和自动生成的描述。通过这些注释，他们揭示了数据集中存在的偏见，并展示了这些偏见如何传递到CLIP和Stable Diffusion模型中。**

- **链接: [http://arxiv.org/pdf/2510.03721v1](http://arxiv.org/pdf/2510.03721v1)**

> **作者:** Leander Girrbach; Stephan Alaniz; Genevieve Smith; Trevor Darrell; Zeynep Akata
>
> **备注:** 48 pages
>
> **摘要:** Vision-language models trained on large-scale multimodal datasets show strong demographic biases, but the role of training data in producing these biases remains unclear. A major barrier has been the lack of demographic annotations in web-scale datasets such as LAION-400M. We address this gap by creating person-centric annotations for the full dataset, including over 276 million bounding boxes, perceived gender and race/ethnicity labels, and automatically generated captions. These annotations are produced through validated automatic labeling pipelines combining object detection, multimodal captioning, and finetuned classifiers. Using them, we uncover demographic imbalances and harmful associations, such as the disproportionate linking of men and individuals perceived as Black or Middle Eastern with crime-related and negative content. We also show that 60-70% of gender bias in CLIP and Stable Diffusion can be linearly explained by direct co-occurrences in the data. Our resources establish the first large-scale empirical link between dataset composition and downstream model bias.
>
---
#### [new 075] DECOR: Deep Embedding Clustering with Orientation Robustness
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于无监督学习任务，旨在解决半导体制造中晶圆缺陷模式复杂、无标签、不平衡等问题。作者提出了DECOR方法，通过深度嵌入聚类实现对晶圆缺陷的自动分组，具备方向鲁棒性，能有效识别不同方向的缺陷模式，提升了聚类的准确性和可靠性。**

- **链接: [http://arxiv.org/pdf/2510.03328v1](http://arxiv.org/pdf/2510.03328v1)**

> **作者:** Fiona Victoria Stanley Jothiraj; Arunaggiri Pandian Karunanidhi; Seth A. Eichmeyer
>
> **摘要:** In semiconductor manufacturing, early detection of wafer defects is critical for product yield optimization. However, raw wafer data from wafer quality tests are often complex, unlabeled, imbalanced and can contain multiple defects on a single wafer, making it crucial to design clustering methods that remain reliable under such imperfect data conditions. We introduce DECOR, a deep clustering with orientation robustness framework that groups complex defect patterns from wafer maps into consistent clusters. We evaluate our method on the open source MixedWM38 dataset, demonstrating its ability to discover clusters without manual tuning. DECOR explicitly accounts for orientation variations in wafer maps, ensuring that spatially similar defects are consistently clustered regardless of its rotation or alignment. Experiments indicate that our method outperforms existing clustering baseline methods, thus providing a reliable and scalable solution in automated visual inspection systems.
>
---
#### [new 076] A Recursive Pyramidal Algorithm for Solving the Image Registration Problem
- **分类: cs.CV**

- **简介: 该论文属于图像配准任务，旨在解决如何通过变换使两幅图像对应点对齐的问题。论文提出了一种递归金字塔算法，该算法简单、端到端可训练，代码实现仅需几行Python。实验表明其在少量训练数据和短训练时间内仍能取得较好效果，适用于数据、时间或代码复杂度受限的场景。**

- **链接: [http://arxiv.org/pdf/2510.04231v1](http://arxiv.org/pdf/2510.04231v1)**

> **作者:** Stefan Dirnstorfer
>
> **摘要:** The problem of image registration is finding a transformation that aligns two images, such that the corresponding points are in the same location. This paper introduces a simple, end-to-end trainable algorithm that is implementable in a few lines of Python code. The approach is shown to work with very little training data and training time, while achieving accurate results in some settings. An example application to stereo vision was trained from 74 images on a 19x15 input window. With just a dozen lines of Python code this algorithm excels in brevity and may serve as a good start in related scenarios with limitations to training data, training time or code complexity.
>
---
#### [new 077] Road Damage and Manhole Detection using Deep Learning for Smart Cities: A Polygonal Annotation Approach
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像检测任务，旨在解决城市道路损坏和井盖自动识别问题。作者使用YOLOv9算法和多边形标注，构建了一个包含1000多张图像的数据集，训练模型识别“破损”、“未破损”和“井盖”三类目标，实现了较高的准确率，尤其在破损和未破损类别上表现优异。**

- **链接: [http://arxiv.org/pdf/2510.03797v1](http://arxiv.org/pdf/2510.03797v1)**

> **作者:** Rasel Hossen; Diptajoy Mistry; Mushiur Rahman; Waki As Sami Atikur Rahman Hridoy; Sajib Saha; Muhammad Ibrahim
>
> **备注:** 13 pages
>
> **摘要:** Urban safety and infrastructure maintenance are critical components of smart city development. Manual monitoring of road damages is time-consuming, highly costly, and error-prone. This paper presents a deep learning approach for automated road damage and manhole detection using the YOLOv9 algorithm with polygonal annotations. Unlike traditional bounding box annotation, we employ polygonal annotations for more precise localization of road defects. We develop a novel dataset comprising more than one thousand images which are mostly collected from Dhaka, Bangladesh. This dataset is used to train a YOLO-based model for three classes, namely Broken, Not Broken, and Manhole. We achieve 78.1% overall image-level accuracy. The YOLOv9 model demonstrates strong performance for Broken (86.7% F1-score) and Not Broken (89.2% F1-score) classes, with challenges in Manhole detection (18.2% F1-score) due to class imbalance. Our approach offers an efficient and scalable solution for monitoring urban infrastructure in developing countries.
>
---
#### [new 078] Real-Time Threaded Houbara Detection and Segmentation for Wildlife Conservation using Mobile Platforms
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于野生动物保护中的实时目标检测与分割任务，旨在解决计算资源受限下隐秘物种检测难题。工作提出轻量化双阶段框架，结合YOLOv10检测与MobileSAM分割，实现高效并发处理。针对阿拉伯大鸨检测，取得优异性能，并发布4万标注图像数据集。**

- **链接: [http://arxiv.org/pdf/2510.03501v1](http://arxiv.org/pdf/2510.03501v1)**

> **作者:** Lyes Saad Saoud; Loic Lesobre; Enrico Sorato; Irfan Hussain
>
> **摘要:** Real-time animal detection and segmentation in natural environments are vital for wildlife conservation, enabling non-invasive monitoring through remote camera streams. However, these tasks remain challenging due to limited computational resources and the cryptic appearance of many species. We propose a mobile-optimized two-stage deep learning framework that integrates a Threading Detection Model (TDM) to parallelize YOLOv10-based detection and MobileSAM-based segmentation. Unlike prior YOLO+SAM pipelines, our approach improves real-time performance by reducing latency through threading. YOLOv10 handles detection while MobileSAM performs lightweight segmentation, both executed concurrently for efficient resource use. On the cryptic Houbara Bustard, a conservation-priority species, our model achieves mAP50 of 0.9627, mAP75 of 0.7731, mAP95 of 0.7178, and a MobileSAM mIoU of 0.7421. YOLOv10 operates at 43.7 ms per frame, confirming real-time readiness. We introduce a curated Houbara dataset of 40,000 annotated images to support model training and evaluation across diverse conditions. The code and dataset used in this study are publicly available on GitHub at https://github.com/LyesSaadSaoud/mobile-houbara-detseg. For interactive demos and additional resources, visit https://lyessaadsaoud.github.io/LyesSaadSaoud-Threaded-YOLO-SAM-Houbara.
>
---
#### [new 079] Multi-Modal Oral Cancer Detection Using Weighted Ensemble Convolutional Neural Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决口腔鳞状细胞癌（OSCC）早期诊断困难的问题。作者开发了一种多模态深度学习框架，融合临床、放射和病理图像，通过加权集成DenseNet-121模型提升检测性能，最终实现更早、更准确的癌症筛查。**

- **链接: [http://arxiv.org/pdf/2510.03878v1](http://arxiv.org/pdf/2510.03878v1)**

> **作者:** Ajo Babu George; Sreehari J R Ajo Babu George; Sreehari J R Ajo Babu George; Sreehari J R
>
> **摘要:** Aims Late diagnosis of Oral Squamous Cell Carcinoma (OSCC) contributes significantly to its high global mortality rate, with over 50\% of cases detected at advanced stages and a 5-year survival rate below 50\% according to WHO statistics. This study aims to improve early detection of OSCC by developing a multimodal deep learning framework that integrates clinical, radiological, and histopathological images using a weighted ensemble of DenseNet-121 convolutional neural networks (CNNs). Material and Methods A retrospective study was conducted using publicly available datasets representing three distinct medical imaging modalities. Each modality-specific dataset was used to train a DenseNet-121 CNN via transfer learning. Augmentation and modality-specific preprocessing were applied to increase robustness. Predictions were fused using a validation-weighted ensemble strategy. Evaluation was performed using accuracy, precision, recall, F1-score. Results High validation accuracy was achieved for radiological (100\%) and histopathological (95.12\%) modalities, with clinical images performing lower (63.10\%) due to visual heterogeneity. The ensemble model demonstrated improved diagnostic robustness with an overall accuracy of 84.58\% on a multimodal validation dataset of 55 samples. Conclusion The multimodal ensemble framework bridges gaps in the current diagnostic workflow by offering a non-invasive, AI-assisted triage tool that enhances early identification of high-risk lesions. It supports clinicians in decision-making, aligning with global oncology guidelines to reduce diagnostic delays and improve patient outcomes.
>
---
#### [new 080] Real-Time Assessment of Bystander Situation Awareness in Drone-Assisted First Aid
- **分类: cs.CV**

- **简介: 该论文属于人机协同任务，旨在解决无人机辅助急救中旁观者情境感知的实时评估问题。作者构建了DANDSD数据集，并提出基于视频的实时评估框架，结合图嵌入和Transformer模型，有效预测旁观者情境意识，提升应急响应效果。**

- **链接: [http://arxiv.org/pdf/2510.03558v1](http://arxiv.org/pdf/2510.03558v1)**

> **作者:** Shen Chang; Renran Tian; Nicole Adams; Nan Kong
>
> **摘要:** Rapid naloxone delivery via drones offers a promising solution for responding to opioid overdose emergencies (OOEs), by extending lifesaving interventions to medically untrained bystanders before emergency medical services (EMS) arrive. Recognizing the critical role of bystander situational awareness (SA) in human-autonomy teaming (HAT), we address a key research gap in real-time SA assessment by introducing the Drone-Assisted Naloxone Delivery Simulation Dataset (DANDSD). This pioneering dataset captures HAT during simulated OOEs, where college students without medical training act as bystanders tasked with administering intranasal naloxone to a mock overdose victim. Leveraging this dataset, we propose a video-based real-time SA assessment framework that utilizes graph embeddings and transformer models to assess bystander SA in real time. Our approach integrates visual perception and comprehension cues--such as geometric, kinematic, and interaction graph features--and achieves high-performance SA prediction. It also demonstrates strong temporal segmentation accuracy, outperforming the FINCH baseline by 9% in Mean over Frames (MoF) and 5% in Intersection over Union (IoU). This work supports the development of adaptive drone systems capable of guiding bystanders effectively, ultimately improving emergency response outcomes and saving lives.
>
---
#### [new 081] Unified Unsupervised Anomaly Detection via Matching Cost Filtering
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于无监督异常检测任务，旨在使用仅正常数据识别图像和像素级异常。论文提出统一成本过滤（UCF）框架，通过匹配成本卷的构建与可学习过滤模块，减少匹配噪声，提升单模态与多模态场景下的异常检测效果。**

- **链接: [http://arxiv.org/pdf/2510.03363v1](http://arxiv.org/pdf/2510.03363v1)**

> **作者:** Zhe Zhang; Mingxiu Cai; Gaochang Wu; Jing Zhang; Lingqiao Liu; Dacheng Tao; Tianyou Chai; Xiatian Zhu
>
> **备注:** 63 pages (main paper and supplementary material), 39 figures, 58 tables. Submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
>
> **摘要:** Unsupervised anomaly detection (UAD) aims to identify image- and pixel-level anomalies using only normal training data, with wide applications such as industrial inspection and medical analysis, where anomalies are scarce due to privacy concerns and cold-start constraints. Existing methods, whether reconstruction-based (restoring normal counterparts) or embedding-based (pretrained representations), fundamentally conduct image- or feature-level matching to generate anomaly maps. Nonetheless, matching noise has been largely overlooked, limiting their detection ability. Beyond earlier focus on unimodal RGB-based UAD, recent advances expand to multimodal scenarios, e.g., RGB--3D and RGB--Text, enabled by point cloud sensing and vision--language models. Despite shared challenges, these lines remain largely isolated, hindering a comprehensive understanding and knowledge transfer. In this paper, we advocate unified UAD for both unimodal and multimodal settings in the matching perspective. Under this insight, we present Unified Cost Filtering (UCF), a generic post-hoc refinement framework for refining anomaly cost volume of any UAD model. The cost volume is constructed by matching a test sample against normal samples from the same or different modalities, followed by a learnable filtering module with multi-layer attention guidance from the test sample, mitigating matching noise and highlighting subtle anomalies. Comprehensive experiments on 22 diverse benchmarks demonstrate the efficacy of UCF in enhancing a variety of UAD methods, consistently achieving new state-of-the-art results in both unimodal (RGB) and multimodal (RGB--3D, RGB--Text) UAD scenarios. Code and models will be released at https://github.com/ZHE-SAPI/CostFilter-AD.
>
---
#### [new 082] Inference-Time Search using Side Information for Diffusion-based Image Reconstruction
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像重建任务，旨在解决扩散模型在逆问题中忽略辅助信息导致重建质量下降的问题。作者提出了一种新的推理时搜索算法，利用辅助信息平衡探索与利用，提升了重建的准确性和可靠性，并在多种逆问题上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2510.03352v1](http://arxiv.org/pdf/2510.03352v1)**

> **作者:** Mahdi Farahbakhsh; Vishnu Teja Kunde; Dileep Kalathil; Krishna Narayanan; Jean-Francois Chamberland
>
> **摘要:** Diffusion models have emerged as powerful priors for solving inverse problems. However, existing approaches typically overlook side information that could significantly improve reconstruction quality, especially in severely ill-posed settings. In this work, we propose a novel inference-time search algorithm that guides the sampling process using the side information in a manner that balances exploration and exploitation. This enables more accurate and reliable reconstructions, providing an alternative to the gradient-based guidance that is prone to reward-hacking artifacts. Our approach can be seamlessly integrated into a wide range of existing diffusion-based image reconstruction pipelines. Through extensive experiments on a number of inverse problems, such as box inpainting, super-resolution, and various deblurring tasks including motion, Gaussian, nonlinear, and blind deblurring, we show that our approach consistently improves the qualitative and quantitative performance of diffusion-based image reconstruction algorithms. We also show the superior performance of our approach with respect to other baselines, including reward gradient-based guidance algorithms. The code is available at \href{https://github.com/mhdfb/sideinfo-search-reconstruction}{this repository}.
>
---
#### [new 083] CoPA: Hierarchical Concept Prompting and Aggregating Network for Explainable Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医学诊断任务，旨在提升深度学习模型的可解释性。针对现有概念瓶颈模型在多层特征提取和概念编码上的不足，提出CoPA框架，通过多层概念表示提取与提示引导，增强细粒度概念捕捉能力，从而提高疾病预测性能。**

- **链接: [http://arxiv.org/pdf/2510.03767v1](http://arxiv.org/pdf/2510.03767v1)**

> **作者:** Yiheng Dong; Yi Lin; Xin Yang
>
> **备注:** Accepted by MICCAI2025
>
> **摘要:** The transparency of deep learning models is essential for clinical diagnostics. Concept Bottleneck Model provides clear decision-making processes for diagnosis by transforming the latent space of black-box models into human-understandable concepts. However, concept-based methods still face challenges in concept capture capabilities. These methods often rely on encode features solely from the final layer, neglecting shallow and multiscale features, and lack effective guidance in concept encoding, hindering fine-grained concept extraction. To address these issues, we introduce Concept Prompting and Aggregating (CoPA), a novel framework designed to capture multilayer concepts under prompt guidance. This framework utilizes the Concept-aware Embedding Generator (CEG) to extract concept representations from each layer of the visual encoder. Simultaneously, these representations serve as prompts for Concept Prompt Tuning (CPT), steering the model towards amplifying critical concept-related visual cues. Visual representations from each layer are aggregated to align with textual concept representations. With the proposed method, valuable concept-wise information in the images is captured and utilized effectively, thus improving the performance of concept and disease prediction. Extensive experimental results demonstrate that CoPA outperforms state-of-the-art methods on three public datasets. Code is available at https://github.com/yihengd/CoPA.
>
---
#### [new 084] World-To-Image: Grounding Text-to-Image Generation with Agent-Driven World Knowledge
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型在生成新颖或分布外实体时性能下降的问题。作者提出World-To-Image框架，通过智能体驱动的网络搜索获取未知概念的图像，优化多模态提示，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2510.04201v1](http://arxiv.org/pdf/2510.04201v1)**

> **作者:** Moo Hyun Son; Jintaek Oh; Sun Bin Mun; Jaechul Roh; Sehyun Choi
>
> **摘要:** While text-to-image (T2I) models can synthesize high-quality images, their performance degrades significantly when prompted with novel or out-of-distribution (OOD) entities due to inherent knowledge cutoffs. We introduce World-To-Image, a novel framework that bridges this gap by empowering T2I generation with agent-driven world knowledge. We design an agent that dynamically searches the web to retrieve images for concepts unknown to the base model. This information is then used to perform multimodal prompt optimization, steering powerful generative backbones toward an accurate synthesis. Critically, our evaluation goes beyond traditional metrics, utilizing modern assessments like LLMGrader and ImageReward to measure true semantic fidelity. Our experiments show that World-To-Image substantially outperforms state-of-the-art methods in both semantic alignment and visual aesthetics, achieving +8.1% improvement in accuracy-to-prompt on our curated NICE benchmark. Our framework achieves these results with high efficiency in less than three iterations, paving the way for T2I systems that can better reflect the ever-changing real world. Our demo code is available here\footnote{https://github.com/mhson-kyle/World-To-Image}.
>
---
#### [new 085] Benchmark on Monocular Metric Depth Estimation in Wildlife Setting
- **分类: cs.CV**

- **简介: 该论文属于单目度量深度估计任务，旨在解决野生动物监测中单目图像缺乏准确深度信息的问题。论文构建了首个针对野生动物环境的单目深度估计基准，评估了四种先进方法及几何基线的性能，提供了精度与效率的综合分析。**

- **链接: [http://arxiv.org/pdf/2510.04723v1](http://arxiv.org/pdf/2510.04723v1)**

> **作者:** Niccolò Niccoli; Lorenzo Seidenari; Ilaria Greco; Francesco Rovero
>
> **摘要:** Camera traps are widely used for wildlife monitoring, but extracting accurate distance measurements from monocular images remains challenging due to the lack of depth information. While monocular depth estimation (MDE) methods have advanced significantly, their performance in natural wildlife environments has not been systematically evaluated. This work introduces the first benchmark for monocular metric depth estimation in wildlife monitoring conditions. We evaluate four state-of-the-art MDE methods (Depth Anything V2, ML Depth Pro, ZoeDepth, and Metric3D) alongside a geometric baseline on 93 camera trap images with ground truth distances obtained using calibrated ChARUCO patterns. Our results demonstrate that Depth Anything V2 achieves the best overall performance with a mean absolute error of 0.454m and correlation of 0.962, while methods like ZoeDepth show significant degradation in outdoor natural environments (MAE: 3.087m). We find that median-based depth extraction consistently outperforms mean-based approaches across all deep learning methods. Additionally, we analyze computational efficiency, with ZoeDepth being fastest (0.17s per image) but least accurate, while Depth Anything V2 provides an optimal balance of accuracy and speed (0.22s per image). This benchmark establishes performance baselines for wildlife applications and provides practical guidance for implementing depth estimation in conservation monitoring systems.
>
---
#### [new 086] ChronoEdit: Towards Temporal Reasoning for Image Editing and World Simulation
- **分类: cs.CV**

- **简介: 该论文属于图像编辑与世界模拟任务，旨在解决编辑后图像的物理一致性问题。论文提出ChronoEdit框架，将图像编辑转化为视频生成问题，通过引入时间推理阶段，结合预训练视频生成模型，实现更符合物理规律的图像编辑。**

- **链接: [http://arxiv.org/pdf/2510.04290v1](http://arxiv.org/pdf/2510.04290v1)**

> **作者:** Jay Zhangjie Wu; Xuanchi Ren; Tianchang Shen; Tianshi Cao; Kai He; Yifan Lu; Ruiyuan Gao; Enze Xie; Shiyi Lan; Jose M. Alvarez; Jun Gao; Sanja Fidler; Zian Wang; Huan Ling
>
> **备注:** Project Page: https://research.nvidia.com/labs/toronto-ai/chronoedit
>
> **摘要:** Recent advances in large generative models have significantly advanced image editing and in-context image generation, yet a critical gap remains in ensuring physical consistency, where edited objects must remain coherent. This capability is especially vital for world simulation related tasks. In this paper, we present ChronoEdit, a framework that reframes image editing as a video generation problem. First, ChronoEdit treats the input and edited images as the first and last frames of a video, allowing it to leverage large pretrained video generative models that capture not only object appearance but also the implicit physics of motion and interaction through learned temporal consistency. Second, ChronoEdit introduces a temporal reasoning stage that explicitly performs editing at inference time. Under this setting, the target frame is jointly denoised with reasoning tokens to imagine a plausible editing trajectory that constrains the solution space to physically viable transformations. The reasoning tokens are then dropped after a few steps to avoid the high computational cost of rendering a full video. To validate ChronoEdit, we introduce PBench-Edit, a new benchmark of image-prompt pairs for contexts that require physical consistency, and demonstrate that ChronoEdit surpasses state-of-the-art baselines in both visual fidelity and physical plausibility. Code and models for both the 14B and 2B variants of ChronoEdit will be released on the project page: https://research.nvidia.com/labs/toronto-ai/chronoedit
>
---
#### [new 087] Evaluating OCR performance on food packaging labels in South Africa
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于OCR性能评估任务，旨在解决食品包装标签上的文本识别问题。研究对四个开源OCR系统在南非食品包装图像上的表现进行了评估，关注准确率、覆盖率和速度等指标，为该领域提供了基准结果和改进方向。**

- **链接: [http://arxiv.org/pdf/2510.03570v1](http://arxiv.org/pdf/2510.03570v1)**

> **作者:** Mayimunah Nagayi; Alice Khan; Tamryn Frank; Rina Swart; Clement Nyirenda
>
> **备注:** 17 pages
>
> **摘要:** This study evaluates four open-source Optical Character Recognition (OCR) systems which are Tesseract, EasyOCR, PaddleOCR, and TrOCR on real world food packaging images. The aim is to assess their ability to extract ingredient lists and nutrition facts panels. Accurate OCR for packaging is important for compliance and nutrition monitoring but is challenging due to multilingual text, dense layouts, varied fonts, glare, and curved surfaces. A dataset of 231 products (1,628 images) was processed by all four models to assess speed and coverage, and a ground truth subset of 113 images (60 products) was created for accuracy evaluation. Metrics include Character Error Rate (CER), Word Error Rate (WER), BLEU, ROUGE-L, F1, coverage, and execution time. On the ground truth subset, Tesseract achieved the lowest CER (0.912) and the highest BLEU (0.245). EasyOCR provided a good balance between accuracy and multilingual support. PaddleOCR achieved near complete coverage but was slower because it ran on CPU only due to GPU incompatibility, and TrOCR produced the weakest results despite GPU acceleration. These results provide a packaging-specific benchmark, establish a baseline, and highlight directions for layout-aware methods and text localization.
>
---
#### [new 088] ActiveMark: on watermarking of visual foundation models via massive activations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉与模型版权保护任务，旨在解决视觉基础模型（VFMs）被非法复制和分发的问题。作者提出了一种名为ActiveMark的方法，通过微调少量关键层和一个编码器-解码器网络，在模型内部嵌入可检测的数字水印。该方法确保水印在模型用于下游任务时仍可被检测，有效区分合法与非法模型副本。**

- **链接: [http://arxiv.org/pdf/2510.04966v1](http://arxiv.org/pdf/2510.04966v1)**

> **作者:** Anna Chistyakova; Mikhail Pautov
>
> **摘要:** Being trained on large and vast datasets, visual foundation models (VFMs) can be fine-tuned for diverse downstream tasks, achieving remarkable performance and efficiency in various computer vision applications. The high computation cost of data collection and training motivates the owners of some VFMs to distribute them alongside the license to protect their intellectual property rights. However, a dishonest user of the protected model's copy may illegally redistribute it, for example, to make a profit. As a consequence, the development of reliable ownership verification tools is of great importance today, since such methods can be used to differentiate between a redistributed copy of the protected model and an independent model. In this paper, we propose an approach to ownership verification of visual foundation models by fine-tuning a small set of expressive layers of a VFM along with a small encoder-decoder network to embed digital watermarks into an internal representation of a hold-out set of input images. Importantly, the watermarks embedded remain detectable in the functional copies of the protected model, obtained, for example, by fine-tuning the VFM for a particular downstream task. Theoretically and experimentally, we demonstrate that the proposed method yields a low probability of false detection of a non-watermarked model and a low probability of false misdetection of a watermarked model.
>
---
#### [new 089] REAR: Rethinking Visual Autoregressive Models via Generator-Tokenizer Consistency Regularization
- **分类: cs.CV**

- **简介: 该论文属于视觉自回归生成任务，旨在解决其性能落后于扩散模型的问题。作者提出REAR方法，通过引入生成器与分词器一致性的正则化目标，改善视觉自回归生成效果，无需修改分词器或生成顺序。实验表明该方法显著提升了性能，接近扩散模型水平。**

- **链接: [http://arxiv.org/pdf/2510.04450v1](http://arxiv.org/pdf/2510.04450v1)**

> **作者:** Qiyuan He; Yicong Li; Haotian Ye; Jinghao Wang; Xinyao Liao; Pheng-Ann Heng; Stefano Ermon; James Zou; Angela Yao
>
> **备注:** 27 pages, 23 figures, 5 tables
>
> **摘要:** Visual autoregressive (AR) generation offers a promising path toward unifying vision and language models, yet its performance remains suboptimal against diffusion models. Prior work often attributes this gap to tokenizer limitations and rasterization ordering. In this work, we identify a core bottleneck from the perspective of generator-tokenizer inconsistency, i.e., the AR-generated tokens may not be well-decoded by the tokenizer. To address this, we propose reAR, a simple training strategy introducing a token-wise regularization objective: when predicting the next token, the causal transformer is also trained to recover the visual embedding of the current token and predict the embedding of the target token under a noisy context. It requires no changes to the tokenizer, generation order, inference pipeline, or external models. Despite its simplicity, reAR substantially improves performance. On ImageNet, it reduces gFID from 3.02 to 1.86 and improves IS to 316.9 using a standard rasterization-based tokenizer. When applied to advanced tokenizers, it achieves a gFID of 1.42 with only 177M parameters, matching the performance with larger state-of-the-art diffusion models (675M).
>
---
#### [new 090] Unmasking Puppeteers: Leveraging Biometric Leakage to Disarm Impersonation in AI-based Videoconferencing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频会议安全任务，旨在解决AI合成视频中身份伪装的安全问题。通过分析姿态-表情潜在空间中的生物特征泄露，提出了一种无需查看合成视频的防御方法，有效检测非法身份替换。**

- **链接: [http://arxiv.org/pdf/2510.03548v1](http://arxiv.org/pdf/2510.03548v1)**

> **作者:** Danial Samadi Vahdati; Tai Duc Nguyen; Ekta Prashnani; Koki Nagano; David Luebke; Orazio Gallo; Matthew Stamm
>
> **摘要:** AI-based talking-head videoconferencing systems reduce bandwidth by sending a compact pose-expression latent and re-synthesizing RGB at the receiver, but this latent can be puppeteered, letting an attacker hijack a victim's likeness in real time. Because every frame is synthetic, deepfake and synthetic video detectors fail outright. To address this security problem, we exploit a key observation: the pose-expression latent inherently contains biometric information of the driving identity. Therefore, we introduce the first biometric leakage defense without ever looking at the reconstructed RGB video: a pose-conditioned, large-margin contrastive encoder that isolates persistent identity cues inside the transmitted latent while cancelling transient pose and expression. A simple cosine test on this disentangled embedding flags illicit identity swaps as the video is rendered. Our experiments on multiple talking-head generation models show that our method consistently outperforms existing puppeteering defenses, operates in real-time, and shows strong generalization to out-of-distribution scenarios.
>
---
#### [new 091] FrameOracle: Learning What to See and How Much to See in Videos
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决现有帧采样策略在信息密度和任务复杂度上的适应性不足问题。作者提出了FrameOracle模块，通过预测关键帧及数量，优化视觉-语言模型的输入，提升效率与准确率，实现了更优的权衡。**

- **链接: [http://arxiv.org/pdf/2510.03584v1](http://arxiv.org/pdf/2510.03584v1)**

> **作者:** Chaoyu Li; Tianzhi Li; Fei Tao; Zhenyu Zhao; Ziqian Wu; Maozheng Zhao; Juntong Song; Cheng Niu; Pooyan Fazli
>
> **摘要:** Vision-language models (VLMs) have advanced video understanding, but their performance is limited by the number of input frames they can process. Existing frame sampling strategies, such as uniform or fixed-budget selection, often fail to adapt to variations in information density or task complexity, resulting in inefficiency and information loss. To address this, we present FrameOracle, a lightweight and plug-and-play module that predicts both (1) which frames are most relevant to a given query and (2) how many frames are needed. FrameOracle is trained using a four-stage curriculum, with the first three stages relying on weak proxy signals such as cross-modal similarity. In the final stage, it leverages stronger supervision from a new dataset we introduce, FrameOracle-41K, the first large-scale VideoQA collection to provide keyframe annotations specifying the minimal set of frames required to answer each question. Extensive experiments across five VLMs and six benchmarks demonstrate that FrameOracle reduces 16-frame inputs to an average of 10.4 frames without any loss in accuracy. When starting from 64-frame candidates, it reduces the input to an average of 13.9 frames while improving accuracy by 1.4%, achieving state-of-the-art efficiency-accuracy trade-offs for scalable video understanding.
>
---
#### [new 092] Visual Representations inside the Language Model
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态语言模型（MLM）在感知密集型任务中的表现，分析其视觉信息处理机制。通过考察视觉键值令牌的流动，发现模型在零样本下能完成分割、语义匹配等任务，但后期层中存在影响感知能力的伪影。提出通过添加文本前缀提升视觉感知能力，揭示MLM中视觉信息控制的重要性。**

- **链接: [http://arxiv.org/pdf/2510.04819v1](http://arxiv.org/pdf/2510.04819v1)**

> **作者:** Benlin Liu; Amita Kamath; Madeleine Grunde-McLaughlin; Winson Han; Ranjay Krishna
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Despite interpretability work analyzing VIT encoders and transformer activations, we don't yet understand why Multimodal Language Models (MLMs) struggle on perception-heavy tasks. We offer an under-studied perspective by examining how popular MLMs (LLaVA-OneVision, Qwen2.5-VL, and Llama-3-LLaVA-NeXT) process their visual key-value tokens. We first study the flow of visual information through the language model, finding that image value tokens encode sufficient information to perform several perception-heavy tasks zero-shot: segmentation, semantic correspondence, temporal correspondence, and referring expression detection. We find that while the language model does augment the visual information received from the projection of input visual encodings-which we reveal correlates with overall MLM perception capability-it contains less visual information on several tasks than the equivalent visual encoder (SigLIP) that has not undergone MLM finetuning. Further, we find that the visual information corresponding to input-agnostic image key tokens in later layers of language models contains artifacts which reduce perception capability of the overall MLM. Next, we discuss controlling visual information in the language model, showing that adding a text prefix to the image input improves perception capabilities of visual representations. Finally, we reveal that if language models were able to better control their visual information, their perception would significantly improve; e.g., in 33.3% of Art Style questions in the BLINK benchmark, perception information present in the language model is not surfaced to the output! Our findings reveal insights into the role of key-value tokens in multimodal systems, paving the way for deeper mechanistic interpretability of MLMs and suggesting new directions for training their visual encoder and language model components.
>
---
#### [new 093] Do Superpixel Segmentation Methods Influence Deforestation Image Classification?
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分类任务，旨在研究超像素分割方法对热带森林毁林检测的影响。论文探讨了不同分割方法对分类器训练的影响，并通过融合分类器提升了毁林检测的平衡准确率。**

- **链接: [http://arxiv.org/pdf/2510.04645v1](http://arxiv.org/pdf/2510.04645v1)**

> **作者:** Hugo Resende; Fabio A. Faria; Eduardo B. Neto; Isabela Borlido; Victor Sundermann; Silvio Jamil F. Guimarães; Álvaro L. Fazenda
>
> **备注:** 15 pages, 3 figures, paper accepted to present at CIARP 2025
>
> **摘要:** Image segmentation is a crucial step in various visual applications, including environmental monitoring through remote sensing. In the context of the ForestEyes project, which combines citizen science and machine learning to detect deforestation in tropical forests, image segments are used for labeling by volunteers and subsequent model training. Traditionally, the Simple Linear Iterative Clustering (SLIC) algorithm is adopted as the segmentation method. However, recent studies have indicated that other superpixel-based methods outperform SLIC in remote sensing image segmentation, and might suggest that they are more suitable for the task of detecting deforested areas. In this sense, this study investigated the impact of the four best segmentation methods, together with SLIC, on the training of classifiers for the target application. Initially, the results showed little variation in performance among segmentation methods, even when selecting the top five classifiers using the PyCaret AutoML library. However, by applying a classifier fusion approach (ensemble of classifiers), noticeable improvements in balanced accuracy were observed, highlighting the importance of both the choice of segmentation method and the combination of machine learning-based models for deforestation detection tasks.
>
---
#### [new 094] Visualizing Celebrity Dynamics in Video Content: A Proposed Approach Using Face Recognition Timestamp Data
- **分类: cs.CV**

- **简介: 该论文属于视频内容分析任务，旨在解决大规模视频中名人动态理解问题。通过结合多GPU推理系统与可视化平台，高效生成并展示名人出现的时间戳数据，提供多维度可视化分析，揭示其在视频中的出现模式、合作关系及时间动态，助力娱乐分析与内容策略优化。**

- **链接: [http://arxiv.org/pdf/2510.03292v1](http://arxiv.org/pdf/2510.03292v1)**

> **作者:** Doğanay Demir; İlknur Durgar Elkahlout
>
> **摘要:** In an era dominated by video content, understanding its structure and dynamics has become increasingly important. This paper presents a hybrid framework that combines a distributed multi-GPU inference system with an interactive visualization platform for analyzing celebrity dynamics in video episodes. The inference framework efficiently processes large volumes of video data by leveraging optimized ONNX models, heterogeneous batch inference, and high-throughput parallelism, ensuring scalable generation of timestamped appearance records. These records are then transformed into a comprehensive suite of visualizations, including appearance frequency charts, duration analyses, pie charts, co-appearance matrices, network graphs, stacked area charts, seasonal comparisons, and heatmaps. Together, these visualizations provide multi-dimensional insights into video content, revealing patterns in celebrity prominence, screen-time distribution, temporal dynamics, co-appearance relationships, and intensity across episodes and seasons. The interactive nature of the system allows users to dynamically explore data, identify key moments, and uncover evolving relationships between individuals. By bridging distributed recognition with structured, visually-driven analytics, this work enables new possibilities for entertainment analytics, content creation strategies, and audience engagement studies.
>
---
#### [new 095] Beyond Appearance: Transformer-based Person Identification from Conversational Dynamics
- **分类: cs.CV**

- **简介: 该论文属于人物识别任务，旨在通过对话中的姿态和动态信息实现身份识别。作者基于Transformer架构，采用双流框架分别建模空间配置与时间运动模式，并引入多尺度时间Transformer。实验表明空间信息更具判别性，特征融合可提升性能，验证了Transformer在自然交互场景中的潜力。**

- **链接: [http://arxiv.org/pdf/2510.04753v1](http://arxiv.org/pdf/2510.04753v1)**

> **作者:** Masoumeh Chapariniya; Teodora Vukovic; Sarah Ebling; Volker Dellwo
>
> **摘要:** This paper investigates the performance of transformer-based architectures for person identification in natural, face-to-face conversation scenario. We implement and evaluate a two-stream framework that separately models spatial configurations and temporal motion patterns of 133 COCO WholeBody keypoints, extracted from a subset of the CANDOR conversational corpus. Our experiments compare pre-trained and from-scratch training, investigate the use of velocity features, and introduce a multi-scale temporal transformer for hierarchical motion modeling. Results demonstrate that domain-specific training significantly outperforms transfer learning, and that spatial configurations carry more discriminative information than temporal dynamics. The spatial transformer achieves 95.74% accuracy, while the multi-scale temporal transformer achieves 93.90%. Feature-level fusion pushes performance to 98.03%, confirming that postural and dynamic information are complementary. These findings highlight the potential of transformer architectures for person identification in natural interactions and provide insights for future multimodal and cross-cultural studies.
>
---
#### [new 096] BLADE: Bias-Linked Adaptive DEbiasing
- **分类: cs.CV**

- **简介: 论文提出BLADE方法，旨在解决神经网络学习中因数据偏差导致的泛化能力差问题。通过生成模型转换图像的偏差特征，自适应地优化样本，增强模型鲁棒性。实验表明其性能优于现有技术。**

- **链接: [http://arxiv.org/pdf/2510.04174v1](http://arxiv.org/pdf/2510.04174v1)**

> **作者:** Piyush Arora; Navlika Singh; Vasubhya Diwan; Pratik Mazumder
>
> **备注:** The authors have contributed equally
>
> **摘要:** Neural networks have revolutionized numerous fields, yet they remain vulnerable to a critical flaw: the tendency to learn implicit biases, spurious correlations between certain attributes and target labels in training data. These biases are often more prevalent and easier to learn, causing models to rely on superficial patterns rather than task-relevant features necessary for generalization. Existing methods typically rely on strong assumptions, such as prior knowledge of these biases or access to bias-conflicting samples, i.e., samples that contradict spurious correlations and counterbalance bias-aligned samples, samples that conform to these spurious correlations. However, such assumptions are often impractical in real-world settings. We propose BLADE ({B}ias-{L}inked {A}daptive {DE}biasing), a generative debiasing framework that requires no prior knowledge of bias or bias-conflicting samples. BLADE first trains a generative model to translate images across bias domains while preserving task-relevant features. Then, it adaptively refines each image with its synthetic counterpart based on the image's susceptibility to bias. To encourage robust representations, BLADE aligns an image with its bias-translated synthetic counterpart that shares task-relevant features but differs in bias, while misaligning it with samples sharing the same bias. We evaluate BLADE on multiple benchmark datasets and show that it significantly outperforms state-of-the-art methods. Notably, it exceeds the closest baseline by an absolute margin of around 18% on the corrupted CIFAR-10 dataset under the worst group setting, establishing a new benchmark in bias mitigation and demonstrating its potential for developing more robust deep learning models without explicit supervision.
>
---
#### [new 097] QuantDemoire: Quantization with Outlier Aware for Image Demoiréing
- **分类: cs.CV**

- **简介: 本文属于图像去莫尔纹任务，旨在解决深度学习模型在边缘设备部署时计算资源受限的问题。现有量化方法会导致性能下降，为此提出QuantDemoire框架，包含异常感知量化和频域感知校准策略，有效减少参数和计算量，同时保持图像质量。**

- **链接: [http://arxiv.org/pdf/2510.04066v1](http://arxiv.org/pdf/2510.04066v1)**

> **作者:** Zheng Chen; Kewei Zhang; Xiaoyang Liu; Weihang Zhang; Mengfan Wang; Yifan Fu; Yulun Zhang
>
> **备注:** Code is available at: https://github.com/zhengchen1999/QuantDemoire
>
> **摘要:** Demoir\'eing aims to remove moir\'e artifacts that often occur in images. While recent deep learning-based methods have achieved promising results, they typically require substantial computational resources, limiting their deployment on edge devices. Model quantization offers a compelling solution. However, directly applying existing quantization methods to demoir\'eing models introduces severe performance degradation. The main reasons are distribution outliers and weakened representations in smooth regions. To address these issues, we propose QuantDemoire, a post-training quantization framework tailored to demoir\'eing. It contains two key components. **First}, we introduce an outlier-aware quantizer to reduce errors from outliers. It uses sampling-based range estimation to reduce activation outliers, and keeps a few extreme weights in FP16 with negligible cost. **Second**, we design a frequency-aware calibration strategy. It emphasizes low- and mid-frequency components during fine-tuning, which mitigates banding artifacts caused by low-bit quantization. Extensive experiments validate that our QuantDemoire achieves large reductions in parameters and computation while maintaining quality. Meanwhile, it outperforms existing quantization methods by over **4 dB** on W4A4. Code is released at: https://github.com/zhengchen1999/QuantDemoire.
>
---
#### [new 098] Exploring the Challenge and Value of Deep Learning in Automated Skin Disease Diagnosis
- **分类: cs.CV**

- **简介: 该论文综述了深度学习在皮肤疾病自动诊断中的应用，探讨了其面临的挑战如图像噪声、数据不平衡等，并总结了应对方法如数据增强、混合模型等，旨在提升诊断准确性和临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2510.03869v1](http://arxiv.org/pdf/2510.03869v1)**

> **作者:** Runhao Liu; Ziming Chen; Peng Zhang
>
> **摘要:** Skin cancer is one of the most prevalent and deadly forms of cancer worldwide, which highlights the critical importance of early detection and diagnosis in improving patient outcomes. Deep learning (DL) has shown significant promise in enhancing the accuracy and efficiency of automated skin disease diagnosis, particularly in detecting and evaluating skin lesions and classification. However, there are still several challenges for DL-based skin cancer diagnosis, including complex features, image noise, intra-class variation, inter-class similarity, and data imbalance. By synthesizing recent research, this review discusses innovative approaches to cope with these challenges, such as data augmentation, hybrid models, and feature fusion, etc. Furthermore, the review highlights the integration of DL models into clinical workflows, offering insights into the potential of deep learning to revolutionize skin disease diagnosis and improve clinical decision-making. This article follows a comprehensive methodology based on the PRISMA framework and emphasizes the need for continued advancements to fully unlock the transformative potential of DL in dermatological care.
>
---
#### [new 099] Platonic Transformers: A Solid Choice For Equivariance
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于计算机视觉与深度学习任务，旨在解决Transformer模型缺乏几何对称性归纳偏置的问题。作者提出Platonic Transformer，通过引入柏拉图立体对称群的参考框架，在不增加计算成本的前提下，实现对连续平移和柏拉图对称性的等变性，同时保持标准Transformer的灵活性与效率。**

- **链接: [http://arxiv.org/pdf/2510.03511v1](http://arxiv.org/pdf/2510.03511v1)**

> **作者:** Mohammad Mohaiminul Islam; Rishabh Anand; David R. Wessels; Friso de Kruiff; Thijs P. Kuipers; Rex Ying; Clara I. Sánchez; Sharvaree Vadgama; Georg Bökman; Erik J. Bekkers
>
> **摘要:** While widespread, Transformers lack inductive biases for geometric symmetries common in science and computer vision. Existing equivariant methods often sacrifice the efficiency and flexibility that make Transformers so effective through complex, computationally intensive designs. We introduce the Platonic Transformer to resolve this trade-off. By defining attention relative to reference frames from the Platonic solid symmetry groups, our method induces a principled weight-sharing scheme. This enables combined equivariance to continuous translations and Platonic symmetries, while preserving the exact architecture and computational cost of a standard Transformer. Furthermore, we show that this attention is formally equivalent to a dynamic group convolution, which reveals that the model learns adaptive geometric filters and enables a highly scalable, linear-time convolutional variant. Across diverse benchmarks in computer vision (CIFAR-10), 3D point clouds (ScanObjectNN), and molecular property prediction (QM9, OMol25), the Platonic Transformer achieves competitive performance by leveraging these geometric constraints at no additional cost.
>
---
#### [new 100] Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在提升视频-大型多模态模型（Video-LMM）的推理能力。论文系统梳理了Video-LMM的后训练方法，包括监督微调、强化学习与测试时扩展，提出统一框架以解决时空关系建模、长视频处理与多模态融合等挑战，并总结了关键设计原则与评估标准。**

- **链接: [http://arxiv.org/pdf/2510.05034v1](http://arxiv.org/pdf/2510.05034v1)**

> **作者:** Yunlong Tang; Jing Bi; Pinxin Liu; Zhenyu Pan; Zhangyun Tan; Qianxiang Shen; Jiani Liu; Hang Hua; Junjia Guo; Yunzhong Xiao; Chao Huang; Zhiyuan Wang; Susan Liang; Xinyi Liu; Yizhi Song; Yuhe Nie; Jia-Xing Zhong; Bozheng Li; Daiqing Qi; Ziyun Zeng; Ali Vosoughi; Luchuan Song; Zeliang Zhang; Daiki Shimada; Han Liu; Jiebo Luo; Chenliang Xu
>
> **备注:** The 1st version
>
> **摘要:** Video understanding represents the most challenging frontier in computer vision, requiring models to reason about complex spatiotemporal relationships, long-term dependencies, and multimodal evidence. The recent emergence of Video-Large Multimodal Models (Video-LMMs), which integrate visual encoders with powerful decoder-based language models, has demonstrated remarkable capabilities in video understanding tasks. However, the critical phase that transforms these models from basic perception systems into sophisticated reasoning engines, post-training, remains fragmented across the literature. This survey provides the first comprehensive examination of post-training methodologies for Video-LMMs, encompassing three fundamental pillars: supervised fine-tuning (SFT) with chain-of-thought, reinforcement learning (RL) from verifiable objectives, and test-time scaling (TTS) through enhanced inference computation. We present a structured taxonomy that clarifies the roles, interconnections, and video-specific adaptations of these techniques, addressing unique challenges such as temporal localization, spatiotemporal grounding, long video efficiency, and multimodal evidence integration. Through systematic analysis of representative methods, we synthesize key design principles, insights, and evaluation protocols while identifying critical open challenges in reward design, scalability, and cost-performance optimization. We further curate essential benchmarks, datasets, and metrics to facilitate rigorous assessment of post-training effectiveness. This survey aims to provide researchers and practitioners with a unified framework for advancing Video-LMM capabilities. Additional resources and updates are maintained at: https://github.com/yunlong10/Awesome-Video-LMM-Post-Training
>
---
#### [new 101] Your Vision-Language Model Can't Even Count to 20: Exposing the Failures of VLMs in Compositional Counting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型（VLM）任务，旨在解决VLM在组合计数任务中的表现问题。作者构建了一个简化基准VLMCountBench，使用基本几何形状进行计数测试，发现VLM在单一形状计数上表现良好，但在多种形状组合时表现显著下降，暴露出其在组合推理上的局限性。**

- **链接: [http://arxiv.org/pdf/2510.04401v1](http://arxiv.org/pdf/2510.04401v1)**

> **作者:** Xuyang Guo; Zekai Huang; Zhenmei Shi; Zhao Song; Jiahao Zhang
>
> **摘要:** Vision-Language Models (VLMs) have become a central focus of today's AI community, owing to their impressive abilities gained from training on large-scale vision-language data from the Web. These models have demonstrated strong performance across diverse tasks, including image understanding, video understanding, complex visual reasoning, and embodied AI. Despite these noteworthy successes, a fundamental question remains: Can VLMs count objects correctly? In this paper, we introduce a simple yet effective benchmark, VLMCountBench, designed under a minimalist setting with only basic geometric shapes (e.g., triangles, circles) and their compositions, focusing exclusively on counting tasks without interference from other factors. We adopt strict independent variable control and systematically study the effects of simple properties such as color, size, and prompt refinement in a controlled ablation. Our empirical results reveal that while VLMs can count reliably when only one shape type is present, they exhibit substantial failures when multiple shape types are combined (i.e., compositional counting). This highlights a fundamental empirical limitation of current VLMs and motivates important directions for future research.
>
---
#### [new 102] Advances in Medical Image Segmentation: A Comprehensive Survey with a Focus on Lumbar Spine Applications
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提升诊断与治疗的精准度。论文综述了传统方法与深度学习技术，特别关注腰椎应用，探讨了多种模型及挑战，如数据偏差与模型可解释性。**

- **链接: [http://arxiv.org/pdf/2510.03318v1](http://arxiv.org/pdf/2510.03318v1)**

> **作者:** Ahmed Kabil; Ghada Khoriba; Mina Yousef; Essam A. Rashed
>
> **备注:** Computers in Biology and Medicine (to appear)
>
> **摘要:** Medical Image Segmentation (MIS) stands as a cornerstone in medical image analysis, playing a pivotal role in precise diagnostics, treatment planning, and monitoring of various medical conditions. This paper presents a comprehensive and systematic survey of MIS methodologies, bridging the gap between traditional image processing techniques and modern deep learning approaches. The survey encompasses thresholding, edge detection, region-based segmentation, clustering algorithms, and model-based techniques while also delving into state-of-the-art deep learning architectures such as Convolutional Neural Networks (CNNs), Fully Convolutional Networks (FCNs), and the widely adopted U-Net and its variants. Moreover, integrating attention mechanisms, semi-supervised learning, generative adversarial networks (GANs), and Transformer-based models is thoroughly explored. In addition to covering established methods, this survey highlights emerging trends, including hybrid architectures, cross-modality learning, federated and distributed learning frameworks, and active learning strategies, which aim to address challenges such as limited labeled datasets, computational complexity, and model generalizability across diverse imaging modalities. Furthermore, a specialized case study on lumbar spine segmentation is presented, offering insights into the challenges and advancements in this relatively underexplored anatomical region. Despite significant progress in the field, critical challenges persist, including dataset bias, domain adaptation, interpretability of deep learning models, and integration into real-world clinical workflows.
>
---
#### [new 103] No Tokens Wasted: Leveraging Long Context in Biomedical Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于生物医学视觉-语言建模任务，旨在解决因文本编码器上下文长度限制导致的长格式生物医学图注被截断的问题。作者通过扩展文本编码器的上下文长度，构建了支持长文本的BIOMEDICA-LongCAP数据集，并训练了BMC-LongCLIP模型，显著提升了检索与分类性能。**

- **链接: [http://arxiv.org/pdf/2510.03978v1](http://arxiv.org/pdf/2510.03978v1)**

> **作者:** Min Woo Sun; Alejandro Lozano; Javier Gamazo Tejero; Vishwesh Nath; Xiao Xiao Sun; James Burgess; Yuhui Zhang; Kun Yuan; Robert Tibshirani; Sean Huver; Serena Yeung-Levy
>
> **摘要:** Embedding vision-language models (VLMs) are typically pretrained with short text windows (<77 tokens), which forces the truncation of long-format captions. Yet, the distribution of biomedical captions from large-scale open source literature reveals that a huge portion of captions far exceed 77 tokens. To this end, we investigate the impact of pretraining on long-format biomedical captions by extending the context length of text encoders in VLMs. We find that longer context (thus, enabling additional supervision provided in long-format captions) correlates with better retrieval and classification performance. Given this finding, we introduce BIOMEDICA-LongCAP, a dataset of 1M image-caption pairs enriched with context-aware descriptions from full-text articles, providing longer and additional textual supervision. Using BIOMEDICA-LongCAP, we train BMC-LongCLIP, a long-context biomedical VLM with a text encoder supporting windows of up to 512 tokens. Our model extends context capacity by 6.6x, reducing token waste from 55% to just 2.2%. On long-caption retrieval benchmarks, BMC-LongCLIP achieves up to +30% absolute gains in Recall@1 and +2% average improvements in classification, while also converging faster than short-context. Our results demonstrate that long-context modeling is a promising direction for advancing biomedical VLMs.
>
---
#### [new 104] ExposureEngine: Oriented Logo Detection and Sponsor Visibility Analytics in Sports Broadcasts
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于计算机视觉与体育媒体分析任务，旨在解决赞助商标志在动态转播画面中因旋转或透视变形导致的可见性度量不准确问题。作者构建了ExposureEngine系统，采用Oriented Bounding Box（OBB）实现精准检测，并结合自然语言处理提供可视化分析报告，提升了赞助效果评估的自动化与准确性。**

- **链接: [http://arxiv.org/pdf/2510.04739v1](http://arxiv.org/pdf/2510.04739v1)**

> **作者:** Mehdi Houshmand Sarkhoosh; Frøy Øye; Henrik Nestor Sørlie; Nam Hoang Vu; Dag Johansen; Cise Midoglu; Tomas Kupka; Pål Halvorsen
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Quantifying sponsor visibility in sports broadcasts is a critical marketing task traditionally hindered by manual, subjective, and unscalable analysis methods. While automated systems offer an alternative, their reliance on axis-aligned Horizontal Bounding Box (HBB) leads to inaccurate exposuremetrics when logos appear rotated or skewed due to dynamic camera angles and perspective distortions. This paper introduces ExposureEngine, an end-to-end system designed for accurate, rotation-aware sponsor visibility analytics in sports broadcasts, demonstrated in a soccer case study. Our approach predicts Oriented Bounding Box (OBB) to provide a geometrically precise fit to each logo regardless of the orientation on-screen. To train and evaluate our detector, we developed a new dataset comprising 1,103 frames from Swedish elite soccer, featuring 670 unique sponsor logos annotated with OBBs. Our model achieves a mean Average Precision (mAP@0.5) of 0.859, with a precision of 0.96 and recall of 0.87, demonstrating robust performance in localizing logos under diverse broadcast conditions. The system integrates these detections into an analytical pipeline that calculates precise visibility metrics, such as exposure duration and on-screen coverage. Furthermore, we incorporate a language-driven agentic layer, enabling users to generate reports, summaries, and media content through natural language queries. The complete system, including the dataset and the analytics dashboard, provides a comprehensive solution for auditable and interpretable sponsor measurement in sports media. An overview of the ExposureEngine is available online: https://youtu.be/tRw6OBISuW4 .
>
---
#### [new 105] Error correction in multiclass image classification of facial emotion on unbalanced samples
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文属于多类别图像分类任务，旨在解决面部表情识别中类别不平衡导致的分类误差问题。作者采用带有注意力机制的LSTM模型，专注于面部关键区域，并通过在六类子集上训练、对第七类进行误差校正的实验方法，验证了该方法在小类别上的有效性，表明其在实际应用如反欺诈系统中的潜力。**

- **链接: [http://arxiv.org/pdf/2510.03337v1](http://arxiv.org/pdf/2510.03337v1)**

> **作者:** Andrey A. Lebedev; Victor B. Kazantsev; Sergey V. Stasenko
>
> **摘要:** This paper considers the problem of error correction in multi-class classification of face images on unbalanced samples. The study is based on the analysis of a data frame containing images labeled by seven different emotional states of people of different ages. Particular attention is paid to the problem of class imbalance, in which some emotions significantly prevail over others. To solve the classification problem, a neural network model based on LSTM with an attention mechanism focusing on key areas of the face that are informative for emotion recognition is used. As part of the experiments, the model is trained on all possible configurations of subsets of six classes with subsequent error correction for the seventh class, excluded at the training stage. The results show that correction is possible for all classes, although the degree of success varies: some classes are better restored, others are worse. In addition, on the test sample, when correcting some classes, an increase in key quality metrics for small classes was recorded, which indicates the promise of the proposed approach in solving applied problems related to the search for rare events, for example, in anti-fraud systems. Thus, the proposed method can be effectively applied in facial expression analysis systems and in tasks requiring stable classification under skewed class distribution.
>
---
#### [new 106] UGround: Towards Unified Visual Grounding with Unrolled Transformers
- **分类: cs.CV**

- **简介: 该论文属于视觉基础任务，旨在解决现有方法依赖固定最后一层和无空间提示的问题。UGround通过动态选择中间层和引入空间提示，实现统一的视觉基础框架，支持多种属性和任务。**

- **链接: [http://arxiv.org/pdf/2510.03853v1](http://arxiv.org/pdf/2510.03853v1)**

> **作者:** Rui Qian; Xin Yin; Chuanhang Deng; Zhiyuan Peng; Jian Xiong; Wei Zhai; Dejing Dou
>
> **备注:** https://github.com/rui-qian/UGround
>
> **摘要:** We present UGround, a \textbf{U}nified visual \textbf{Ground}ing paradigm that dynamically selects intermediate layers across \textbf{U}nrolled transformers as ``mask as prompt'', diverging from the prevailing pipeline that leverages the fixed last hidden layer as ``\texttt{<SEG>} as prompt''. UGround addresses two primary challenges posed by the prevailing paradigm: (1) its reliance on the fixed last hidden layer, which sequentially amplifies cumulative errors arising from layer-by-layer propagation without intermediate correction, and (2) its use of \texttt{<SEG>} as a prompt, which implicitly projects textual embeddings into visual space without explicit spatial cues (\eg, coordinates). Central to UGround is Policy-Prompted Masking, which comprises two key components: Stochastic Skip Connection (SSC) and Mask as Prompt (MasP). SSC is a reinforcement learning policy that, via stochastic sampling, allows each \texttt{<SEG>} token to slide across unrolled transformer layers, enabling dynamic layer selection at which it connects to the vision model (\eg, SAM) in a skip-connection fashion. Given the selected hidden layer, MasP uses the similarity map derived from the \texttt{<SEG>} token and image tokens as a soft logit mask to prompt SAM for mask generation, offering explicit spatial cues through its activation regions. To validate the effectiveness of UGround, we, for the first time, have unified visual grounding within a single framework from an attribute perspective, spanning from traditional refer expression segmentation to newly proposed reasoning segmentation, single-target to multi-target, positive query to false premise (empty target). All codes and models are publicly available at \href{https://github.com/rui-qian/UGround}{https://github.com/rui-qian/UGround}.
>
---
#### [new 107] Denoising of Two-Phase Optically Sectioned Structured Illumination Reconstructions Using Encoder-Decoder Networks
- **分类: cs.CV**

- **简介: 该论文属于图像去噪任务，旨在解决两相位光学切片结构光照明显微成像中的残余伪影问题。作者使用编码器-解码器网络，通过合成数据训练去噪模型，有效提升了图像清晰度，展示了深度学习在该领域应用的潜力。**

- **链接: [http://arxiv.org/pdf/2510.03452v1](http://arxiv.org/pdf/2510.03452v1)**

> **作者:** Allison Davis; Yezhi Shen; Xiaoyu Ji; Fengqing Zhu
>
> **备注:** 5 pages, 4 figures, submitted to ICASSP 2026
>
> **摘要:** Structured illumination (SI) enhances image resolution and contrast by projecting patterned light onto a sample. In two-phase optical-sectioning SI (OS-SI), reduced acquisition time introduces residual artifacts that conventional denoising struggles to suppress. Deep learning offers an alternative to traditional methods; however, supervised training is limited by the lack of clean, optically sectioned ground-truth data. We investigate encoder-decoder networks for artifact reduction in two-phase OS-SI, using synthetic training pairs formed by applying real artifact fields to synthetic images. An asymmetrical denoising autoencoder (DAE) and a U-Net are trained on the synthetic data, then evaluated on real OS-SI images. Both networks improve image clarity, with each excelling against different artifact types. These results demonstrate that synthetic training enables supervised denoising of OS-SI images and highlight the potential of encoder-decoder networks to streamline reconstruction workflows.
>
---
#### [new 108] MorphoSim: An Interactive, Controllable, and Editable Language-guided 4D World Simulator
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出MorphoSim，一个基于语言引导的4D世界模拟框架。任务是生成可控、可编辑的多视角一致场景。解决现有文本到视频模型缺乏交互性和空间控制的问题。工作包括轨迹引导生成、特征场蒸馏，实现动态编辑与多视角观察。**

- **链接: [http://arxiv.org/pdf/2510.04390v1](http://arxiv.org/pdf/2510.04390v1)**

> **作者:** Xuehai He; Shijie Zhou; Thivyanth Venkateswaran; Kaizhi Zheng; Ziyu Wan; Achuta Kadambi; Xin Eric Wang
>
> **摘要:** World models that support controllable and editable spatiotemporal environments are valuable for robotics, enabling scalable training data, repro ducible evaluation, and flexible task design. While recent text-to-video models generate realistic dynam ics, they are constrained to 2D views and offer limited interaction. We introduce MorphoSim, a language guided framework that generates 4D scenes with multi-view consistency and object-level controls. From natural language instructions, MorphoSim produces dynamic environments where objects can be directed, recolored, or removed, and scenes can be observed from arbitrary viewpoints. The framework integrates trajectory-guided generation with feature field dis tillation, allowing edits to be applied interactively without full re-generation. Experiments show that Mor phoSim maintains high scene fidelity while enabling controllability and editability. The code is available at https://github.com/eric-ai-lab/Morph4D.
>
---
#### [new 109] Referring Expression Comprehension for Small Objects
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言任务中的指代表达理解（REC），旨在解决自动驾驶场景中微小物体定位困难的问题。作者构建了专门针对小物体的SOREC数据集，并提出了PIZA适配器模块，用于高效微调模型，逐步聚焦并精确定位小物体，显著提升了定位准确率。**

- **链接: [http://arxiv.org/pdf/2510.03701v1](http://arxiv.org/pdf/2510.03701v1)**

> **作者:** Kanoko Goto; Takumi Hirose; Mahiro Ukai; Shuhei Kurita; Nakamasa Inoue
>
> **摘要:** Referring expression comprehension (REC) aims to localize the target object described by a natural language expression. Recent advances in vision-language learning have led to significant performance improvements in REC tasks. However, localizing extremely small objects remains a considerable challenge despite its importance in real-world applications such as autonomous driving. To address this issue, we introduce a novel dataset and method for REC targeting small objects. First, we present the small object REC (SOREC) dataset, which consists of 100,000 pairs of referring expressions and corresponding bounding boxes for small objects in driving scenarios. Second, we propose the progressive-iterative zooming adapter (PIZA), an adapter module for parameter-efficient fine-tuning that enables models to progressively zoom in and localize small objects. In a series of experiments, we apply PIZA to GroundingDINO and demonstrate a significant improvement in accuracy on the SOREC dataset. Our dataset, codes and pre-trained models are publicly available on the project page.
>
---
#### [new 110] Scaling Sequence-to-Sequence Generative Neural Rendering
- **分类: cs.CV**

- **简介: 论文提出Kaleido，一种用于高质量神经渲染的生成模型，将3D视为视频的特例，统一物体与场景级渲染。通过序列到序列图像合成，实现无需显式3D表示的视角生成，利用视频数据预训练提升空间一致性，减少对标注3D数据的依赖，在视图合成任务中达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2510.04236v1](http://arxiv.org/pdf/2510.04236v1)**

> **作者:** Shikun Liu; Kam Woh Ng; Wonbong Jang; Jiadong Guo; Junlin Han; Haozhe Liu; Yiannis Douratsos; Juan C. Pérez; Zijian Zhou; Chi Phung; Tao Xiang; Juan-Manuel Pérez-Rúa
>
> **备注:** Project Page: https://shikun.io/projects/kaleido
>
> **摘要:** We present Kaleido, a family of generative models designed for photorealistic, unified object- and scene-level neural rendering. Kaleido operates on the principle that 3D can be regarded as a specialised sub-domain of video, expressed purely as a sequence-to-sequence image synthesis task. Through a systemic study of scaling sequence-to-sequence generative neural rendering, we introduce key architectural innovations that enable our model to: i) perform generative view synthesis without explicit 3D representations; ii) generate any number of 6-DoF target views conditioned on any number of reference views via a masked autoregressive framework; and iii) seamlessly unify 3D and video modelling within a single decoder-only rectified flow transformer. Within this unified framework, Kaleido leverages large-scale video data for pre-training, which significantly improves spatial consistency and reduces reliance on scarce, camera-labelled 3D datasets -- all without any architectural modifications. Kaleido sets a new state-of-the-art on a range of view synthesis benchmarks. Its zero-shot performance substantially outperforms other generative methods in few-view settings, and, for the first time, matches the quality of per-scene optimisation methods in many-view settings.
>
---
#### [new 111] Domain Generalization for Semantic Segmentation: A Survey
- **分类: cs.CV**

- **简介: 该论文属于语义分割任务，旨在解决模型在未知目标域上的泛化问题。论文综述了领域泛化方法，重点分析了基于基础模型的范式转变，并对现有方法进行了性能比较，以推动相关研究发展。**

- **链接: [http://arxiv.org/pdf/2510.03540v1](http://arxiv.org/pdf/2510.03540v1)**

> **作者:** Manuel Schwonberg; Hanno Gottschalk
>
> **备注:** Accepted to CVPR2025W
>
> **摘要:** The generalization of deep neural networks to unknown domains is a major challenge despite their tremendous progress in recent years. For this reason, the dynamic area of domain generalization (DG) has emerged. In contrast to unsupervised domain adaptation, there is no access to or knowledge about the target domains, and DG methods aim to generalize across multiple different unseen target domains. Domain generalization is particularly relevant for the task semantic segmentation which is used in several areas such as biomedicine or automated driving. This survey provides a comprehensive overview of the rapidly evolving topic of domain generalized semantic segmentation. We cluster and review existing approaches and identify the paradigm shift towards foundation-model-based domain generalization. Finally, we provide an extensive performance comparison of all approaches, which highlights the significant influence of foundation models on domain generalization. This survey seeks to advance domain generalization research and inspire scientists to explore new research directions.
>
---
#### [new 112] AvatarVTON: 4D Virtual Try-On for Animatable Avatars
- **分类: cs.CV**

- **简介: 该论文属于虚拟试穿任务，旨在解决单视角下动态衣物交互与高质量虚拟试穿生成的问题。作者提出了AvatarVTON框架，包含光流校正与非线性形变模块，实现自由姿态控制、新视角渲染及多样衣物选择，提升了动态衣物的真实感与适用性。**

- **链接: [http://arxiv.org/pdf/2510.04822v1](http://arxiv.org/pdf/2510.04822v1)**

> **作者:** Zicheng Jiang; Jixin Gao; Shengfeng He; Xinzhe Li; Yulong Zheng; Zhaotong Yang; Junyu Dong; Yong Du
>
> **摘要:** We propose AvatarVTON, the first 4D virtual try-on framework that generates realistic try-on results from a single in-shop garment image, enabling free pose control, novel-view rendering, and diverse garment choices. Unlike existing methods, AvatarVTON supports dynamic garment interactions under single-view supervision, without relying on multi-view garment captures or physics priors. The framework consists of two key modules: (1) a Reciprocal Flow Rectifier, a prior-free optical-flow correction strategy that stabilizes avatar fitting and ensures temporal coherence; and (2) a Non-Linear Deformer, which decomposes Gaussian maps into view-pose-invariant and view-pose-specific components, enabling adaptive, non-linear garment deformations. To establish a benchmark for 4D virtual try-on, we extend existing baselines with unified modules for fair qualitative and quantitative comparisons. Extensive experiments show that AvatarVTON achieves high fidelity, diversity, and dynamic garment realism, making it well-suited for AR/VR, gaming, and digital-human applications.
>
---
#### [new 113] Learning from All: Concept Alignment for Autonomous Distillation from Multiple Drifting MLLMs
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于知识蒸馏任务，旨在解决多模态大语言模型（MLLM）中多个教师模型推理轨迹存在概念漂移导致学生模型性能下降的问题。论文提出“学习、比较、批判”范式与自主偏好优化（APO），实现概念对齐，提升模型一致性、鲁棒性与泛化能力，并贡献了大规模数据集CXR-MAX。**

- **链接: [http://arxiv.org/pdf/2510.04142v1](http://arxiv.org/pdf/2510.04142v1)**

> **作者:** Xiaoyu Yang; Jie Lu; En Yu
>
> **摘要:** This paper identifies a critical yet underexplored challenge in distilling from multimodal large language models (MLLMs): the reasoning trajectories generated by multiple drifting teachers exhibit concept drift, whereby their reasoning distributions evolve unpredictably and transmit biases to the student model, ultimately compromising its performance. To tackle this issue, we pioneer a theoretical connection between concept drift and knowledge distillation, casting the non-stationary reasoning dynamics from multiple MLLM teachers as next-token prediction of multi-stream reasoning trajectories.Guided by concept drift, we introduce the "learn, compare, critique" paradigm, culminating in autonomous preference optimization (APO). Under the active guidance of the teachers, the student model first learns and self-distils preferred thinking by comparing multiple teachers. It then engages in critical reflection over the drifting inference from teachers, performing concept alignment through APO, ultimately yielding a robust, consistent, and generalizable model.Extensive experiments demonstrate our superior performance of consistency, robustness and generalization within knowledge distillation. Besides, we also contributed a large-scale dataset, CXR-MAX (Multi-teachers Alignment X-rays), comprising 170,982 distilled reasoning trajectories derived from publicly accessible MLLMs based on MIMIC-CXR. Our code and data are public at: https://anonymous.4open.science/r/Autonomous-Distillation/.
>
---
#### [new 114] The View From Space: Navigating Instrumentation Differences with EOFMs
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于地球观测领域任务，旨在解决现有EOFMs模型对不同传感器架构敏感的问题。作者分析了传感器差异对模型内部表示的影响，指出当前设计的不足，并为未来模型开发提供方向。**

- **链接: [http://arxiv.org/pdf/2510.03316v1](http://arxiv.org/pdf/2510.03316v1)**

> **作者:** Ryan P. Demilt; Nicholas LaHaye; Karis Tenneson
>
> **摘要:** Earth Observation Foundation Models (EOFMs) have exploded in prevalence as tools for processing the massive volumes of remotely sensed and other earth observation data, and for delivering impact on the many essential earth monitoring tasks. An emerging trend posits using the outputs of pre-trained models as 'embeddings' which summarize high dimensional data to be used for generic tasks such as similarity search and content-specific queries. However, most EOFM models are trained only on single modalities of data and then applied or benchmarked by matching bands across different modalities. It is not clear from existing work what impact diverse sensor architectures have on the internal representations of the present suite of EOFMs. We show in this work that the representation space of EOFMs is highly sensitive to sensor architecture and that understanding this difference gives a vital perspective on the pitfalls of current EOFM design and signals for how to move forward as model developers, users, and a community guided by robust remote-sensing science.
>
---
#### [new 115] Automating construction safety inspections using a multi-modal vision-language RAG framework
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 论文提出SiteShield，一种基于多模态视觉-语言模型的检索增强生成框架，用于自动化建筑安全检查报告生成。任务是解决传统方法效率低、现有模型响应不准确及多模态输入受限的问题。工作包括构建多模态框架，结合视觉与音频输入，提升信息检索与报告生成效果。**

- **链接: [http://arxiv.org/pdf/2510.04145v1](http://arxiv.org/pdf/2510.04145v1)**

> **作者:** Chenxin Wang; Elyas Asadi Shamsabadi; Zhaohui Chen; Luming Shen; Alireza Ahmadian Fard Fini; Daniel Dias-da-Costa
>
> **备注:** 33 pages, 11 figures, 7 tables
>
> **摘要:** Conventional construction safety inspection methods are often inefficient as they require navigating through large volume of information. Recent advances in large vision-language models (LVLMs) provide opportunities to automate safety inspections through enhanced visual and linguistic understanding. However, existing applications face limitations including irrelevant or unspecific responses, restricted modal inputs and hallucinations. Utilisation of Large Language Models (LLMs) for this purpose is constrained by availability of training data and frequently lack real-time adaptability. This study introduces SiteShield, a multi-modal LVLM-based Retrieval-Augmented Generation (RAG) framework for automating construction safety inspection reports by integrating visual and audio inputs. Using real-world data, SiteShield outperformed unimodal LLMs without RAG with an F1 score of 0.82, hamming loss of 0.04, precision of 0.76, and recall of 0.96. The findings indicate that SiteShield offers a novel pathway to enhance information retrieval and efficiency in generating safety reports.
>
---
#### [new 116] From Actions to Kinesics: Extracting Human Psychological States through Bodily Movements
- **分类: cs.CV**

- **简介: 该论文属于行为识别任务，旨在通过人体动作识别心理状态。传统方法依赖问卷或理论模型，难以通用且隐私性差。论文提出一种基于3D骨架数据的kinesics识别框架，结合ST-GCN和CNN，利用迁移学习实现动作到心理状态的自动映射，保护隐私的同时揭示潜在行为结构，提升人机交互建模的准确性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.04844v1](http://arxiv.org/pdf/2510.04844v1)**

> **作者:** Cheyu Lin; Katherine A. Flanigan
>
> **备注:** The 15th International Workshop on Structural Health Monitoring (IWSHM)
>
> **摘要:** Understanding the dynamic relationship between humans and the built environment is a key challenge in disciplines ranging from environmental psychology to reinforcement learning (RL). A central obstacle in modeling these interactions is the inability to capture human psychological states in a way that is both generalizable and privacy preserving. Traditional methods rely on theoretical models or questionnaires, which are limited in scope, static, and labor intensive. We present a kinesics recognition framework that infers the communicative functions of human activity -- known as kinesics -- directly from 3D skeleton joint data. Combining a spatial-temporal graph convolutional network (ST-GCN) with a convolutional neural network (CNN), the framework leverages transfer learning to bypass the need for manually defined mappings between physical actions and psychological categories. The approach preserves user anonymity while uncovering latent structures in bodily movements that reflect cognitive and emotional states. Our results on the Dyadic User EngagemenT (DUET) dataset demonstrate that this method enables scalable, accurate, and human-centered modeling of behavior, offering a new pathway for enhancing RL-driven simulations of human-environment interaction.
>
---
#### [new 117] Bridge Thinking and Acting: Unleashing Physical Potential of VLM with Generalizable Action Expert
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人视觉-语言-行动任务，旨在解决现有模型在跨任务泛化和环境适应上的不足。作者提出一种新框架，通过引入可泛化的动作专家和3D轨迹中间表示，解耦规划与动作，实现高效、鲁棒的跨任务迁移。**

- **链接: [http://arxiv.org/pdf/2510.03896v1](http://arxiv.org/pdf/2510.03896v1)**

> **作者:** Mingyu Liu; Zheng Huang; Xiaoyi Lin; Muzhi Zhu; Canyu Zhao; Zongze Du; Yating Wang; Haoyi Zhu; Hao Chen; Chunhua Shen
>
> **摘要:** Although Vision-Language Models (VLM) have demonstrated impressive planning and reasoning capabilities, translating these abilities into the physical world introduces significant challenges. Conventional Vision-Language-Action (VLA) models, which integrate reasoning and action into a monolithic architecture, generalize poorly because they are constrained by scarce, narrow-domain data. While recent dual-system approaches attempt to decouple "thinking" from "acting", they are often constrained by semantic ambiguities within the action module. This ambiguity makes large-scale, cross-task training infeasible. Consequently, these systems typically necessitate fine-tuning on newly collected data when deployed to novel environments, and the cooperation mechanism between the two systems remains ill-defined. To address these limitations, we introduce, for the first time, a framework centered around a generalizable action expert. Our approach utilizes sparse 3D trajectories as an intermediate representation, effectively bridging the high-level planning capabilities of the VLM with the low-level physical action module. During the planning phase, the VLM is only required to generate coarse 3D waypoints. These waypoints are then processed by our generalizable action expert, which refines them into dense, executable action sequences by sampling real-time point cloud observations of the environment. To promote training efficiency and robust generalization, we introduce a novel "Action Pre-training, Pointcloud Fine-tuning" paradigm. Our method combines the broad generalization capabilities of VLMs in visual understanding and planning with the fine-grained, action-level generalization of action expert.
>
---
#### [new 118] Label-Efficient Cross-Modality Generalization for Liver Segmentation in Multi-Phase MRI
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决多模态、多厂商MRI数据中肝脏分割标注数据稀缺且分布不均的问题。作者提出了一种标签高效的方法，结合基础模型微调、协同训练与伪监督，实现跨模态泛化，无需空间配准即可提升分割性能。**

- **链接: [http://arxiv.org/pdf/2510.04705v1](http://arxiv.org/pdf/2510.04705v1)**

> **作者:** Quang-Khai Bui-Tran; Minh-Toan Dinh; Thanh-Huy Nguyen; Ba-Thinh Lam; Mai-Anh Vu; Ulas Bagci
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Accurate liver segmentation in multi-phase MRI is vital for liver fibrosis assessment, yet labeled data is often scarce and unevenly distributed across imaging modalities and vendor systems. We propose a label-efficient segmentation approach that promotes cross-modality generalization under real-world conditions, where GED4 hepatobiliary-phase annotations are limited, non-contrast sequences (T1WI, T2WI, DWI) are unlabeled, and spatial misalignment and missing phases are common. Our method integrates a foundation-scale 3D segmentation backbone adapted via fine-tuning, co-training with cross pseudo supervision to leverage unlabeled volumes, and a standardized preprocessing pipeline. Without requiring spatial registration, the model learns to generalize across MRI phases and vendors, demonstrating robust segmentation performance in both labeled and unlabeled domains. Our results exhibit the effectiveness of our proposed label-efficient baseline for liver segmentation in multi-phase, multi-vendor MRI and highlight the potential of combining foundation model adaptation with co-training for real-world clinical imaging tasks.
>
---
#### [new 119] VaseVQA-3D: Benchmarking 3D VLMs on Ancient Greek Pottery
- **分类: cs.CV**

- **简介: 该论文属于3D视觉语言多模态任务，旨在解决现有模型在古希腊陶器分析中面临的数据稀缺和领域知识不足问题。论文构建了首个3D陶器视觉问答数据集VaseVQA-3D，并提出了领域自适应训练模型VaseVLM，有效提升了模型性能。**

- **链接: [http://arxiv.org/pdf/2510.04479v1](http://arxiv.org/pdf/2510.04479v1)**

> **作者:** Nonghai Zhang; Zeyu Zhang; Jiazi Wang; Yang Zhao; Hao Tang
>
> **摘要:** Vision-Language Models (VLMs) have achieved significant progress in multimodal understanding tasks, demonstrating strong capabilities particularly in general tasks such as image captioning and visual reasoning. However, when dealing with specialized cultural heritage domains like 3D vase artifacts, existing models face severe data scarcity issues and insufficient domain knowledge limitations. Due to the lack of targeted training data, current VLMs struggle to effectively handle such culturally significant specialized tasks. To address these challenges, we propose the VaseVQA-3D dataset, which serves as the first 3D visual question answering dataset for ancient Greek pottery analysis, collecting 664 ancient Greek vase 3D models with corresponding question-answer data and establishing a complete data construction pipeline. We further develop the VaseVLM model, enhancing model performance in vase artifact analysis through domain-adaptive training. Experimental results validate the effectiveness of our approach, where we improve by 12.8% on R@1 metrics and by 6.6% on lexical similarity compared with previous state-of-the-art on the VaseVQA-3D dataset, significantly improving the recognition and understanding of 3D vase artifacts, providing new technical pathways for digital heritage preservation research.
>
---
#### [new 120] \textsc{GUI-Spotlight}: Adaptive Iterative Focus Refinement for Enhanced GUI Visual Grounding
- **分类: cs.CV; cs.AI**

- **简介: 论文提出GUI-Spotlight，属于多模态视觉接地任务，旨在提升图形用户界面中文本指代表与屏幕元素的精准映射，通过动态调用专用工具迭代聚焦相关区域，显著提高视觉接地准确性，在ScreenSpot-Pro基准上表现优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.04039v1](http://arxiv.org/pdf/2510.04039v1)**

> **作者:** Bin Lei; Nuo Xu; Ali Payani; Mingyi Hong; Chunhua Liao; Yu Cao; Caiwen Ding
>
> **摘要:** Multimodal large language models (MLLMs) have markedly expanded the competence of graphical user-interface (GUI) systems, propelling them beyond controlled simulations into complex, real-world environments across diverse platforms. However, practical usefulness is still bounded by the reliability of visual grounding, i.e., mapping textual references to exact on-screen elements. This limitation prevents the system from accurately performing pointer-level actions such as clicking or dragging. To address it, we introduce GUI-Spotlight -- a model trained for image-grounded reasoning that dynamically invokes multiple specialized tools to iteratively narrow its focus to the relevant region of the screen, thereby substantially improving visual grounding accuracy. On the ScreenSpot-Pro benchmark, GUI-Spotlight trained with only 18.5K training samples achieves 52.8\% accuracy, surpassing V2P-7B (50.6\% with 9.6M training samples) and GTA-1-7B (50.1\% with 1.56M training samples).
>
---
#### [new 121] OpusAnimation: Code-Based Dynamic Chart Generation
- **分类: cs.CV**

- **简介: 该论文属于动态图表生成任务，旨在解决多模态大语言模型在动态图表理解和生成能力不足的问题。作者构建了首个动态图表生成基准DCG-Bench和高质量数据集DCG-8K，并提出两阶段训练方法与联合代码-视觉奖励机制，训练出性能优越的模型Qwen2.5-VL-DCG-3B，验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2510.03341v1](http://arxiv.org/pdf/2510.03341v1)**

> **作者:** Bozheng Li; Miao Yang; Zhenhan Chen; Jiawang Cao; Mushui Liu; Yi Lu; Yongliang Wu; Bin Zhang; Yangguang Ji; Licheng Tang; Jay Wu; Wenbo Zhu
>
> **备注:** working in progress
>
> **摘要:** Dynamic Chart Generation (DCG) involves producing code-rendered animated visualizations as charts. While recent advances in multi-modal large language models (MLLMs) have significantly improved their capability on static chart generation and comprehension, MLLMs' potential for handling dynamic chart generation and understanding remains underexplored. To bridge this research gap, we introduce DCG-Bench (Dynamic Chart Generation Benchmark), the first benchmark evaluating MLLM's capability on dynamic chart generation tasks from three dimensions: Simple Text-to-Chart, Detailed Text-to-Chart, and Video-to-Chart tasks. We construct DCG-8K, a high-quality DCG dataset with annotations covering instruction-code-video triplets and QA pairs for both code and video evaluation. Based on DCG-8K, we explored a two-stage training recipe, proposing Joint-Code-Visual Reward for group relative policy optimization to construct expert MLLM Qwen2.5-VL-DCG-3B for the DCG task. Our benchmarking result reveals shortcomings of existing MLLMs in the visual-to-chart task, and our model beats the best open-sourced MLLM with an average 8.31% performance gain across three tasks, and shows on par performance against proprietary models with only 3B parameters, proving the effectiveness of our training recipe. Our code and dataset will be publicly available.
>
---
#### [new 122] CARE-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment
- **分类: cs.CV**

- **简介: 该论文属于医疗数据集构建任务，旨在解决帕金森病步态评估中缺乏大规模多样本数据的问题。论文工作为：提出CARE-PD，首个多中心帕金森病3D步态数据集，包含9个队列，采用统一预处理生成匿名SMPL网格。支持临床评分预测与无监督动作任务，并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2510.04312v1](http://arxiv.org/pdf/2510.04312v1)**

> **作者:** Vida Adeli; Ivan Klabucar; Javad Rajabi; Benjamin Filtjens; Soroush Mehraban; Diwei Wang; Hyewon Seo; Trung-Hieu Hoang; Minh N. Do; Candice Muller; Claudia Oliveira; Daniel Boari Coelho; Pieter Ginis; Moran Gilat; Alice Nieuwboer; Joke Spildooren; Lucas Mckay; Hyeokhyen Kwon; Gari Clifford; Christine Esper; Stewart Factor; Imari Genias; Amirhossein Dadashzadeh; Leia Shum; Alan Whone; Majid Mirmehdi; Andrea Iaboni; Babak Taati
>
> **备注:** Accepted at the Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Objective gait assessment in Parkinson's Disease (PD) is limited by the absence of large, diverse, and clinically annotated motion datasets. We introduce CARE-PD, the largest publicly available archive of 3D mesh gait data for PD, and the first multi-site collection spanning 9 cohorts from 8 clinical centers. All recordings (RGB video or motion capture) are converted into anonymized SMPL meshes via a harmonized preprocessing pipeline. CARE-PD supports two key benchmarks: supervised clinical score prediction (estimating Unified Parkinson's Disease Rating Scale, UPDRS, gait scores) and unsupervised motion pretext tasks (2D-to-3D keypoint lifting and full-body 3D reconstruction). Clinical prediction is evaluated under four generalization protocols: within-dataset, cross-dataset, leave-one-dataset-out, and multi-dataset in-domain adaptation. To assess clinical relevance, we compare state-of-the-art motion encoders with a traditional gait-feature baseline, finding that encoders consistently outperform handcrafted features. Pretraining on CARE-PD reduces MPJPE (from 60.8mm to 7.5mm) and boosts PD severity macro-F1 by 17 percentage points, underscoring the value of clinically curated, diverse training data. CARE-PD and all benchmark code are released for non-commercial research at https://neurips2025.care-pd.ca/.
>
---
#### [new 123] TOPO-Bench: An Open-Source Topological Mapping Evaluation Framework with Quantifiable Perceptual Aliasing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于机器人或SLAM（同时定位与建图）任务，旨在解决拓扑地图构建缺乏统一评估标准和感知别名量化不足的问题。作者提出了TOPO-Bench评估框架，包含拓扑一致性度量、数据集模糊性量化方法，并提供开源数据与工具，支持公平可复现的拓扑地图研究。**

- **链接: [http://arxiv.org/pdf/2510.04100v1](http://arxiv.org/pdf/2510.04100v1)**

> **作者:** Jiaming Wang; Diwen Liu; Jizhuo Chen; Harold Soh
>
> **备注:** Jiaming Wang, Diwen Liu, and Jizhuo Chen contributed equally
>
> **摘要:** Topological mapping offers a compact and robust representation for navigation, but progress in the field is hindered by the lack of standardized evaluation metrics, datasets, and protocols. Existing systems are assessed using different environments and criteria, preventing fair and reproducible comparisons. Moreover, a key challenge - perceptual aliasing - remains under-quantified, despite its strong influence on system performance. We address these gaps by (1) formalizing topological consistency as the fundamental property of topological maps and showing that localization accuracy provides an efficient and interpretable surrogate metric, and (2) proposing the first quantitative measure of dataset ambiguity to enable fair comparisons across environments. To support this protocol, we curate a diverse benchmark dataset with calibrated ambiguity levels, implement and release deep-learned baseline systems, and evaluate them alongside classical methods. Our experiments and analysis yield new insights into the limitations of current approaches under perceptual aliasing. All datasets, baselines, and evaluation tools are fully open-sourced to foster consistent and reproducible research in topological mapping.
>
---
#### [new 124] Joint Learning of Pose Regression and Denoising Diffusion with Score Scaling Sampling for Category-level 6D Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于6D物体姿态估计任务，旨在解决现有扩散模型训练收敛慢、需额外评估网络的问题。作者提出预训练编码器与联合学习策略，加速训练并提升精度；同时引入基于分数缩放的采样引导策略，省去评估网络，实现高效单次推理，取得最优性能。**

- **链接: [http://arxiv.org/pdf/2510.04125v1](http://arxiv.org/pdf/2510.04125v1)**

> **作者:** Seunghyun Lee; Tae-Kyun Kim
>
> **摘要:** Latest diffusion models have shown promising results in category-level 6D object pose estimation by modeling the conditional pose distribution with depth image input. The existing methods, however, suffer from slow convergence during training, learning its encoder with the diffusion denoising network in end-to-end fashion, and require an additional network that evaluates sampled pose hypotheses to filter out low-quality pose candidates. In this paper, we propose a novel pipeline that tackles these limitations by two key components. First, the proposed method pretrains the encoder with the direct pose regression head, and jointly learns the networks via the regression head and the denoising diffusion head, significantly accelerating training convergence while achieving higher accuracy. Second, sampling guidance via time-dependent score scaling is proposed s.t. the exploration-exploitation trade-off is effectively taken, eliminating the need for the additional evaluation network. The sampling guidance maintains multi-modal characteristics of symmetric objects at early denoising steps while ensuring high-quality pose generation at final steps. Extensive experiments on multiple benchmarks including REAL275, HouseCat6D, and ROPE, demonstrate that the proposed method, simple yet effective, achieves state-of-the-art accuracies even with single-pose inference, while being more efficient in both training and inference.
>
---
#### [new 125] From Segments to Concepts: Interpretable Image Classification via Concept-Guided Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像分类任务，旨在提升模型的可解释性。现有方法缺乏透明度且依赖昂贵的概念标注。作者提出SEG-MIL-CBM，结合概念引导的图像分割与注意力机制的多实例学习，自动识别支持每个概念的图像区域，实现无需标注的、空间定位的概念级解释，并提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.04180v1](http://arxiv.org/pdf/2510.04180v1)**

> **作者:** Ran Eisenberg; Amit Rozner; Ethan Fetaya; Ofir Lindenbaum
>
> **摘要:** Deep neural networks have achieved remarkable success in computer vision; however, their black-box nature in decision-making limits interpretability and trust, particularly in safety-critical applications. Interpretability is crucial in domains where errors have severe consequences. Existing models not only lack transparency but also risk exploiting unreliable or misleading features, which undermines both robustness and the validity of their explanations. Concept Bottleneck Models (CBMs) aim to improve transparency by reasoning through human-interpretable concepts. Still, they require costly concept annotations and lack spatial grounding, often failing to identify which regions support each concept. We propose SEG-MIL-CBM, a novel framework that integrates concept-guided image segmentation into an attention-based multiple instance learning (MIL) framework, where each segmented region is treated as an instance and the model learns to aggregate evidence across them. By reasoning over semantically meaningful regions aligned with high-level concepts, our model highlights task-relevant evidence, down-weights irrelevant cues, and produces spatially grounded, concept-level explanations without requiring annotations of concepts or groups. SEG-MIL-CBM achieves robust performance across settings involving spurious correlations (unintended dependencies between background and label), input corruptions (perturbations that degrade visual quality), and large-scale benchmarks, while providing transparent, concept-level explanations.
>
---
#### [new 126] SegMASt3R: Geometry Grounded Segment Matching
- **分类: cs.CV**

- **简介: 本文提出SegMASt3R，用于解决宽基线段匹配任务，即在极端视角变化下匹配图像中的结构化区域。利用3D基础模型的空间理解能力，实现跨视角的语义与几何一致的区域匹配。实验表明其性能优于现有方法，适用于3D实例分割与图像导航任务。**

- **链接: [http://arxiv.org/pdf/2510.05051v1](http://arxiv.org/pdf/2510.05051v1)**

> **作者:** Rohit Jayanti; Swayam Agrawal; Vansh Garg; Siddharth Tourani; Muhammad Haris Khan; Sourav Garg; Madhava Krishna
>
> **备注:** Accepted to The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025) as a Spotlight (top 3.5%)
>
> **摘要:** Segment matching is an important intermediate task in computer vision that establishes correspondences between semantically or geometrically coherent regions across images. Unlike keypoint matching, which focuses on localized features, segment matching captures structured regions, offering greater robustness to occlusions, lighting variations, and viewpoint changes. In this paper, we leverage the spatial understanding of 3D foundation models to tackle wide-baseline segment matching, a challenging setting involving extreme viewpoint shifts. We propose an architecture that uses the inductive bias of these 3D foundation models to match segments across image pairs with up to 180 degree view-point change. Extensive experiments show that our approach outperforms state-of-the-art methods, including the SAM2 video propagator and local feature matching methods, by upto 30% on the AUPRC metric, on ScanNet++ and Replica datasets. We further demonstrate benefits of the proposed model on relevant downstream tasks, including 3D instance segmentation and image-goal navigation. Project Page: https://segmast3r.github.io/
>
---
#### [new 127] Prompt-to-Prompt: Text-Based Image Editing Via Cross-Attention Mechanisms -- The Research of Hyperparameters and Novel Mechanisms to Enhance Existing Frameworks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像编辑任务，旨在解决文本驱动编辑中结果不一致的问题。通过研究超参数优化和交叉注意力机制，提出了“CL P2P”框架及“注意力重加权”方法，提升编辑精度与可靠性，改善模型架构与超参数的交互影响。**

- **链接: [http://arxiv.org/pdf/2510.04034v1](http://arxiv.org/pdf/2510.04034v1)**

> **作者:** Linn Bieske; Carla Lorente
>
> **摘要:** Recent advances in image editing have shifted from manual pixel manipulation to employing deep learning methods like stable diffusion models, which now leverage cross-attention mechanisms for text-driven control. This transition has simplified the editing process but also introduced variability in results, such as inconsistent hair color changes. Our research aims to enhance the precision and reliability of prompt-to-prompt image editing frameworks by exploring and optimizing hyperparameters. We present a comprehensive study of the "word swap" method, develop an "attention re-weight method" for better adaptability, and propose the "CL P2P" framework to address existing limitations like cycle inconsistency. This work contributes to understanding and improving the interaction between hyperparameter settings and the architectural choices of neural network models, specifically their attention mechanisms, which significantly influence the composition and quality of the generated images.
>
---
#### [new 128] In-Field Mapping of Grape Yield and Quality with Illumination-Invariant Deep Learning
- **分类: cs.CV**

- **简介: 该论文属于农业智能系统任务，旨在解决田间葡萄产量与品质无损实时测绘问题。论文设计了一种端到端、物联网赋能的机器人系统，集成葡萄串检测与重量估计模型，以及基于高光谱数据的品质评估深度学习框架；提出LISA模型克服光照变化导致的领域偏移问题，提升品质预测泛化能力，实现高精度、地理参考化的产量与品质数据采集，为精准葡萄种植提供支持。**

- **链接: [http://arxiv.org/pdf/2510.04864v1](http://arxiv.org/pdf/2510.04864v1)**

> **作者:** Ciem Cornelissen; Sander De Coninck; Axel Willekens; Sam Leroux; Pieter Simoens
>
> **备注:** Accepted manuscript for the IEEE Internet of Things Journal. The final version will be available on IEEE Xplore. \c{opyright} 2025 IEEE
>
> **摘要:** This paper presents an end-to-end, IoT-enabled robotic system for the non-destructive, real-time, and spatially-resolved mapping of grape yield and quality (Brix, Acidity) in vineyards. The system features a comprehensive analytical pipeline that integrates two key modules: a high-performance model for grape bunch detection and weight estimation, and a novel deep learning framework for quality assessment from hyperspectral (HSI) data. A critical barrier to in-field HSI is the ``domain shift" caused by variable illumination. To overcome this, our quality assessment is powered by the Light-Invariant Spectral Autoencoder (LISA), a domain-adversarial framework that learns illumination-invariant features from uncalibrated data. We validated the system's robustness on a purpose-built HSI dataset spanning three distinct illumination domains: controlled artificial lighting (lab), and variable natural sunlight captured in the morning and afternoon. Results show the complete pipeline achieves a recall (0.82) for bunch detection and a $R^2$ (0.76) for weight prediction, while the LISA module improves quality prediction generalization by over 20% compared to the baselines. By combining these robust modules, the system successfully generates high-resolution, georeferenced data of both grape yield and quality, providing actionable, data-driven insights for precision viticulture.
>
---
#### [new 129] Visual Language Model as a Judge for Object Detection in Industrial Diagrams
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于工业图像处理任务，旨在解决工业图纸中物体检测结果缺乏自动评估方法的问题。作者提出利用视觉语言模型（VLM）来评估检测结果并指导优化，实现自动质量评估与性能提升。**

- **链接: [http://arxiv.org/pdf/2510.03376v1](http://arxiv.org/pdf/2510.03376v1)**

> **作者:** Sanjukta Ghosh
>
> **备注:** Pre-review version submitted to IEEE ICASSP 2026
>
> **摘要:** Industrial diagrams such as piping and instrumentation diagrams (P&IDs) are essential for the design, operation, and maintenance of industrial plants. Converting these diagrams into digital form is an important step toward building digital twins and enabling intelligent industrial automation. A central challenge in this digitalization process is accurate object detection. Although recent advances have significantly improved object detection algorithms, there remains a lack of methods to automatically evaluate the quality of their outputs. This paper addresses this gap by introducing a framework that employs Visual Language Models (VLMs) to assess object detection results and guide their refinement. The approach exploits the multimodal capabilities of VLMs to identify missing or inconsistent detections, thereby enabling automated quality assessment and improving overall detection performance on complex industrial diagrams.
>
---
#### [new 130] MambaCAFU: Hybrid Multi-Scale and Multi-Attention Model with Mamba-Based Fusion for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决现有模型任务特异性、性能不稳定及效率问题。作者提出MambaCAFU模型，结合CNN、Transformer与Mamba注意力机制，提升分割精度与泛化能力，同时保持计算效率，实现多尺度特征融合与上下文一致性。**

- **链接: [http://arxiv.org/pdf/2510.03786v1](http://arxiv.org/pdf/2510.03786v1)**

> **作者:** T-Mai Bui; Fares Bougourzi; Fadi Dornaika; Vinh Truong Hoang
>
> **摘要:** In recent years, deep learning has shown near-expert performance in segmenting complex medical tissues and tumors. However, existing models are often task-specific, with performance varying across modalities and anatomical regions. Balancing model complexity and performance remains challenging, particularly in clinical settings where both accuracy and efficiency are critical. To address these issues, we propose a hybrid segmentation architecture featuring a three-branch encoder that integrates CNNs, Transformers, and a Mamba-based Attention Fusion (MAF) mechanism to capture local, global, and long-range dependencies. A multi-scale attention-based CNN decoder reconstructs fine-grained segmentation maps while preserving contextual consistency. Additionally, a co-attention gate enhances feature selection by emphasizing relevant spatial and semantic information across scales during both encoding and decoding, improving feature interaction and cross-scale communication. Extensive experiments on multiple benchmark datasets show that our approach outperforms state-of-the-art methods in accuracy and generalization, while maintaining comparable computational complexity. By effectively balancing efficiency and effectiveness, our architecture offers a practical and scalable solution for diverse medical imaging tasks. Source code and trained models will be publicly released upon acceptance to support reproducibility and further research.
>
---
#### [new 131] Enhancing Fake News Video Detection via LLM-Driven Creative Process Simulation
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于虚假新闻视频检测任务，旨在解决现有检测方法因训练数据不足导致的性能问题。通过提出AgentAug框架，利用LLM模拟创作过程生成多样化的虚假新闻视频，并结合主动学习策略提升检测效果。**

- **链接: [http://arxiv.org/pdf/2510.04024v1](http://arxiv.org/pdf/2510.04024v1)**

> **作者:** Yuyan Bu; Qiang Sheng; Juan Cao; Shaofei Wang; Peng Qi; Yuhui Shi; Beizhe Hu
>
> **备注:** ACM CIKM 2025
>
> **摘要:** The emergence of fake news on short video platforms has become a new significant societal concern, necessitating automatic video-news-specific detection. Current detectors primarily rely on pattern-based features to separate fake news videos from real ones. However, limited and less diversified training data lead to biased patterns and hinder their performance. This weakness stems from the complex many-to-many relationships between video material segments and fabricated news events in real-world scenarios: a single video clip can be utilized in multiple ways to create different fake narratives, while a single fabricated event often combines multiple distinct video segments. However, existing datasets do not adequately reflect such relationships due to the difficulty of collecting and annotating large-scale real-world data, resulting in sparse coverage and non-comprehensive learning of the characteristics of potential fake news video creation. To address this issue, we propose a data augmentation framework, AgentAug, that generates diverse fake news videos by simulating typical creative processes. AgentAug implements multiple LLM-driven pipelines of four fabrication categories for news video creation, combined with an active learning strategy based on uncertainty sampling to select the potentially useful augmented samples during training. Experimental results on two benchmark datasets demonstrate that AgentAug consistently improves the performance of short video fake news detectors.
>
---
#### [new 132] DuPLUS: Dual-Prompt Vision-Language Framework for Universal Medical Image Segmentation and Prognosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决现有模型缺乏通用性和语义理解的问题。作者提出了DuPLUS框架，结合视觉-语言双提示机制，实现跨模态、多任务的医学图像分割与预后预测，具备良好的泛化能力与临床适用性。**

- **链接: [http://arxiv.org/pdf/2510.03483v1](http://arxiv.org/pdf/2510.03483v1)**

> **作者:** Numan Saeed; Tausifa Jan Saleem; Fadillah Maani; Muhammad Ridzuan; Hu Wang; Mohammad Yaqub
>
> **摘要:** Deep learning for medical imaging is hampered by task-specific models that lack generalizability and prognostic capabilities, while existing 'universal' approaches suffer from simplistic conditioning and poor medical semantic understanding. To address these limitations, we introduce DuPLUS, a deep learning framework for efficient multi-modal medical image analysis. DuPLUS introduces a novel vision-language framework that leverages hierarchical semantic prompts for fine-grained control over the analysis task, a capability absent in prior universal models. To enable extensibility to other medical tasks, it includes a hierarchical, text-controlled architecture driven by a unique dual-prompt mechanism. For segmentation, DuPLUS is able to generalize across three imaging modalities, ten different anatomically various medical datasets, encompassing more than 30 organs and tumor types. It outperforms the state-of-the-art task specific and universal models on 8 out of 10 datasets. We demonstrate extensibility of its text-controlled architecture by seamless integration of electronic health record (EHR) data for prognosis prediction, and on a head and neck cancer dataset, DuPLUS achieved a Concordance Index (CI) of 0.69. Parameter-efficient fine-tuning enables rapid adaptation to new tasks and modalities from varying centers, establishing DuPLUS as a versatile and clinically relevant solution for medical image analysis. The code for this work is made available at: https://anonymous.4open.science/r/DuPLUS-6C52
>
---
#### [new 133] Exploring Instruction Data Quality for Explainable Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，旨在解决当前方法依赖大规模指令数据带来的冗余和计算成本问题。作者提出了一种基于聚类的数据选择方法IQA-Select，在保证性能的同时显著减少训练数据量，提升了效率。**

- **链接: [http://arxiv.org/pdf/2510.03880v1](http://arxiv.org/pdf/2510.03880v1)**

> **作者:** Yunhao Li; Sijing Wu; Huiyu Duan; Yucheng Zhu; Qi Jia; Guangtao Zhai
>
> **摘要:** In recent years, with the rapid development of powerful multimodal large language models (MLLMs), explainable image quality assessment (IQA) has gradually become popular, aiming at providing quality-related descriptions and answers of images. To achieve this goal, recent methods seek to construct a large-scale instruction tuning dataset to empower the MLLM with quality perception ability following the well-known scaling law. However, a large amount of instruction tuning data may cause substantial computational costs and redundant data, which in turn will cause harm to the performance of the model. To cope with this problem, in this paper, we challenge the scaling law and systematically investigate the role of data quality of the instruction tuning dataset for explainable IQA. Using a powerful pre-trained MLLM, we first investigate the changes in model performance after fine-tuning with different sizes of instruction tuning data. We find that selecting a subset of the data set randomly using an appropriate ratio can even lead to better results than training with the entire instruction tuning dataset, demonstrating the redundancy of current explainable IQA instruction tuning data. Beyond randomly sampling a subset, we propose a clustering-based data selection framework with three stages: clustering feature extraction, cluster quota allocation, and cluster sampling strategy. Then we systematically analyze the choices of each stage and propose a simple but efficient data selection method IQA-Select for explainable IQA. The experimental results demonstrate that IQA-Select can achieve 102.1% and 103.7% performance of full fine-tuning using only 10% selected data in Q-Bench and AesBench respectively, significantly reducing computational costs while achieving better performance.
>
---
#### [new 134] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型在生成过程中因同步去噪导致的图文对齐不足问题。作者提出异步扩散模型，为不同像素分配不同时步，使关键区域更慢去噪，从而利用更清晰上下文，提升生成图像与文本的对齐效果。**

- **链接: [http://arxiv.org/pdf/2510.04504v1](http://arxiv.org/pdf/2510.04504v1)**

> **作者:** Zijing Hu; Yunze Tong; Fengda Zhang; Junkun Yuan; Jun Xiao; Kun Kuang
>
> **备注:** 22 pages, 11 figures, 5 tables
>
> **摘要:** Diffusion models have achieved impressive results in generating high-quality images. Yet, they often struggle to faithfully align the generated images with the input prompts. This limitation arises from synchronous denoising, where all pixels simultaneously evolve from random noise to clear images. As a result, during generation, the prompt-related regions can only reference the unrelated regions at the same noise level, failing to obtain clear context and ultimately impairing text-to-image alignment. To address this issue, we propose asynchronous diffusion models -- a novel framework that allocates distinct timesteps to different pixels and reformulates the pixel-wise denoising process. By dynamically modulating the timestep schedules of individual pixels, prompt-related regions are denoised more gradually than unrelated regions, thereby allowing them to leverage clearer inter-pixel context. Consequently, these prompt-related regions achieve better alignment in the final images. Extensive experiments demonstrate that our asynchronous diffusion models can significantly improve text-to-image alignment across diverse prompts. The code repository for this work is available at https://github.com/hu-zijing/AsynDM.
>
---
#### [new 135] Paper2Video: Automatic Video Generation from Scientific Papers
- **分类: cs.CV; cs.AI; cs.CL; cs.MA; cs.MM**

- **简介: 该论文属于学术视频生成任务，旨在解决从科研论文自动生成展示视频的问题。作者构建了首个包含论文与对应视频的数据集PaperTalker，并提出多智能体框架PaperTalker，实现幻灯片生成、布局优化、字幕、语音合成与虚拟人物渲染，提升了视频生成的准确性和信息量。**

- **链接: [http://arxiv.org/pdf/2510.05096v1](http://arxiv.org/pdf/2510.05096v1)**

> **作者:** Zeyu Zhu; Kevin Qinghong Lin; Mike Zheng Shou
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Academic presentation videos have become an essential medium for research communication, yet producing them remains highly labor-intensive, often requiring hours of slide design, recording, and editing for a short 2 to 10 minutes video. Unlike natural video, presentation video generation involves distinctive challenges: inputs from research papers, dense multi-modal information (text, figures, tables), and the need to coordinate multiple aligned channels such as slides, subtitles, speech, and human talker. To address these challenges, we introduce PaperTalker, the first benchmark of 101 research papers paired with author-created presentation videos, slides, and speaker metadata. We further design four tailored evaluation metrics--Meta Similarity, PresentArena, PresentQuiz, and IP Memory--to measure how videos convey the paper's information to the audience. Building on this foundation, we propose PaperTalker, the first multi-agent framework for academic presentation video generation. It integrates slide generation with effective layout refinement by a novel effective tree search visual choice, cursor grounding, subtitling, speech synthesis, and talking-head rendering, while parallelizing slide-wise generation for efficiency. Experiments on Paper2Video demonstrate that the presentation videos produced by our approach are more faithful and informative than existing baselines, establishing a practical step toward automated and ready-to-use academic video generation. Our dataset, agent, and code are available at https://github.com/showlab/Paper2Video.
>
---
#### [new 136] Unsupervised Active Learning via Natural Feature Progressive Framework
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于主动学习任务，旨在减少深度学习对大规模标注数据的依赖。它提出了一种无监督主动学习方法NFPF，通过新指标量化样本重要性，解决现有方法在数据代表性与抗噪能力上的不足，最终在视觉数据集上达到与监督方法相当的性能。**

- **链接: [http://arxiv.org/pdf/2510.04939v1](http://arxiv.org/pdf/2510.04939v1)**

> **作者:** Yuxi Liu; Catherine Lalman; Yimin Yang
>
> **备注:** Under review at IEEE TPAMI
>
> **摘要:** The effectiveness of modern deep learning models is predicated on the availability of large-scale, human-annotated datasets, a process that is notoriously expensive and time-consuming. While Active Learning (AL) offers a strategic solution by labeling only the most informative and representative data, its iterative nature still necessitates significant human involvement. Unsupervised Active Learning (UAL) presents an alternative by shifting the annotation burden to a single, post-selection step. Unfortunately, prevailing UAL methods struggle to achieve state-of-the-art performance. These approaches typically rely on local, gradient-based scoring for sample importance estimation, which not only makes them vulnerable to ambiguous and noisy data but also hinders their capacity to select samples that adequately represent the full data distribution. Moreover, their use of shallow, one-shot linear selection falls short of a true UAL paradigm. In this paper, we propose the Natural Feature Progressive Framework (NFPF), a UAL method that revolutionizes how sample importance is measured. At its core, NFPF employs a Specific Feature Learning Machine (SFLM) to effectively quantify each sample's contribution to model performance. We further utilize the SFLM to define a powerful Reconstruction Difference metric for initial sample selection. Our comprehensive experiments show that NFPF significantly outperforms all established UAL methods and achieves performance on par with supervised AL methods on vision datasets. Detailed ablation studies and qualitative visualizations provide compelling evidence for NFPF's superior performance, enhanced robustness, and improved data distribution coverage.
>
---
#### [new 137] Mapping Rio de Janeiro's favelas: general-purpose vs. satellite-specific neural networks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于城市遥感任务，旨在检测里约热内卢的贫民窟（favelas）。研究比较了通用预训练神经网络与专为卫星图像设计的网络在检测效果上的差异，探讨任务特异性与数据量对模型性能的影响。**

- **链接: [http://arxiv.org/pdf/2510.03725v1](http://arxiv.org/pdf/2510.03725v1)**

> **作者:** Thomas Hallopeau; Joris Guérin; Laurent Demagistri; Youssef Fouzai; Renata Gracie; Vanderlei Pascoal De Matos; Helen Gurgel; Nadine Dessay
>
> **备注:** 6 pages, 1 figure, 1 table. Presented at the 21st Brazilian Symposium on Remote Sensing (SBSR 2025)
>
> **摘要:** While deep learning methods for detecting informal settlements have already been developed, they have not yet fully utilized the potential offered by recent pretrained neural networks. We compare two types of pretrained neural networks for detecting the favelas of Rio de Janeiro: 1. Generic networks pretrained on large diverse datasets of unspecific images, 2. A specialized network pretrained on satellite imagery. While the latter is more specific to the target task, the former has been pretrained on significantly more images. Hence, this research investigates whether task specificity or data volume yields superior performance in urban informal settlement detection.
>
---
#### [new 138] CodeFormer++: Blind Face Restoration Using Deformable Registration and Deep Metric Learning
- **分类: cs.CV**

- **简介: 该论文属于盲脸修复（Blind Face Restoration, BFR）任务，旨在解决在不损失身份特征的前提下，提升人脸图像的视觉质量。现有方法在视觉质量与身份保真之间存在权衡问题。论文提出了CodeFormer++，通过可变形配准、纹理引导恢复网络和深度度量学习，动态融合身份特征与生成细节，以实现高质量且身份一致的面部修复。**

- **链接: [http://arxiv.org/pdf/2510.04410v1](http://arxiv.org/pdf/2510.04410v1)**

> **作者:** Venkata Bharath Reddy Reddem; Akshay P Sarashetti; Ranjith Merugu; Amit Satish Unde
>
> **摘要:** Blind face restoration (BFR) has attracted increasing attention with the rise of generative methods. Most existing approaches integrate generative priors into the restoration pro- cess, aiming to jointly address facial detail generation and identity preservation. However, these methods often suffer from a trade-off between visual quality and identity fidelity, leading to either identity distortion or suboptimal degradation removal. In this paper, we present CodeFormer++, a novel framework that maximizes the utility of generative priors for high-quality face restoration while preserving identity. We decompose BFR into three sub-tasks: (i) identity- preserving face restoration, (ii) high-quality face generation, and (iii) dynamic fusion of identity features with realistic texture details. Our method makes three key contributions: (1) a learning-based deformable face registration module that semantically aligns generated and restored faces; (2) a texture guided restoration network to dynamically extract and transfer the texture of generated face to boost the quality of identity-preserving restored face; and (3) the integration of deep metric learning for BFR with the generation of informative positive and hard negative samples to better fuse identity- preserving and generative features. Extensive experiments on real-world and synthetic datasets demonstrate that, the pro- posed CodeFormer++ achieves superior performance in terms of both visual fidelity and identity consistency.
>
---
#### [new 139] Ordinal Encoding as a Regularizer in Binary Loss for Solar Flare Prediction
- **分类: cs.CV; astro-ph.SR**

- **简介: 该论文属于太阳耀斑预测任务，旨在解决二分类模型忽视类别内部有序关系导致的预测误差问题。论文提出了一种结合序数信息的改进损失函数，在二元交叉熵损失中引入序数加权，使模型更关注靠近分类边界附近的预测误差，从而提升预测性能。**

- **链接: [http://arxiv.org/pdf/2510.04063v1](http://arxiv.org/pdf/2510.04063v1)**

> **作者:** Chetraj Pandey; Jinsu Hong; Anli Ji; Rafal A. Angryk; Berkay Aydin
>
> **备注:** This is a preprint submitted to ICDM Workshop (SABID 2025). 6 pages, 2 Figures
>
> **摘要:** The prediction of solar flares is typically formulated as a binary classification task, distinguishing events as either Flare (FL) or No-Flare (NF) according to a specified threshold (for example, greater than or equal to C-class, M-class, or X-class). However, this binary framework neglects the inherent ordinal relationships among the sub-classes contained within each category (FL and NF). Several studies on solar flare prediction have empirically shown that the most frequent misclassifications occur near this prediction threshold. This suggests that the models struggle to differentiate events that are similar in intensity but fall on opposite sides of the binary threshold. To mitigate this limitation, we propose a modified loss function that integrates the ordinal information among the sub-classes of the binarized flare labels into the conventional binary cross-entropy (BCE) loss. This approach serves as an ordinality-aware, data-driven regularization method that penalizes the incorrect predictions of flare events in close proximity to the prediction threshold more heavily than those away from the boundary during model optimization. By incorporating ordinal weighting into the loss function, we aim to enhance the model's learning process by leveraging the ordinal characteristics of the data, thereby improving its overall performance.
>
---
#### [new 140] Flexible and Efficient Spatio-Temporal Transformer for Sequential Visual Place Recognition
- **分类: cs.CV**

- **简介: 该论文属于视觉位置识别任务，旨在解决现有方法在灵活性与效率上的不足。作者提出Adapt-STformer，通过Recurrent-DTE模块实现可变序列长度、快速推理和低内存占用。实验表明其在多个数据集上性能优越，显著提升识别准确率并降低资源消耗。**

- **链接: [http://arxiv.org/pdf/2510.04282v1](http://arxiv.org/pdf/2510.04282v1)**

> **作者:** Yu Kiu; Lau; Chao Chen; Ge Jin; Chen Feng
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Sequential Visual Place Recognition (Seq-VPR) leverages transformers to capture spatio-temporal features effectively; however, existing approaches prioritize performance at the expense of flexibility and efficiency. In practice, a transformer-based Seq-VPR model should be flexible to the number of frames per sequence (seq-length), deliver fast inference, and have low memory usage to meet real-time constraints. To our knowledge, no existing transformer-based Seq-VPR method achieves both flexibility and efficiency. To address this gap, we propose Adapt-STformer, a Seq-VPR method built around our novel Recurrent Deformable Transformer Encoder (Recurrent-DTE), which uses an iterative recurrent mechanism to fuse information from multiple sequential frames. This design naturally supports variable seq-lengths, fast inference, and low memory usage. Experiments on the Nordland, Oxford, and NuScenes datasets show that Adapt-STformer boosts recall by up to 17% while reducing sequence extraction time by 36% and lowering memory usage by 35% compared to the second-best baseline.
>
---
#### [new 141] Zero-Shot Fine-Grained Image Classification Using Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于零样本细粒度图像分类任务，旨在解决视觉相似类别间的精确分类问题。论文提出将分类任务转化为视觉问答框架，并引入注意力干预技术提升性能，同时构建了更精确的类别描述基准，验证了方法有效性，超越了现有最先进方法。**

- **链接: [http://arxiv.org/pdf/2510.03903v1](http://arxiv.org/pdf/2510.03903v1)**

> **作者:** Md. Atabuzzaman; Andrew Zhang; Chris Thomas
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated impressive performance on vision-language reasoning tasks. However, their potential for zero-shot fine-grained image classification, a challenging task requiring precise differentiation between visually similar categories, remains underexplored. We present a novel method that transforms zero-shot fine-grained image classification into a visual question-answering framework, leveraging LVLMs' comprehensive understanding capabilities rather than relying on direct class name generation. We enhance model performance through a novel attention intervention technique. We also address a key limitation in existing datasets by developing more comprehensive and precise class description benchmarks. We validate the effectiveness of our method through extensive experimentation across multiple fine-grained image classification benchmarks. Our proposed method consistently outperforms the current state-of-the-art (SOTA) approach, demonstrating both the effectiveness of our method and the broader potential of LVLMs for zero-shot fine-grained classification tasks. Code and Datasets: https://github.com/Atabuzzaman/Fine-grained-classification
>
---
#### [new 142] The best performance in the CARE 2025 -- Liver Task (LiSeg-Contrast): Contrast-Aware Semi-Supervised Segmentation with Domain Generalization and Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决对比增强MRI中肝脏分割因标注数据少、扫描设备差异和增强协议不一致导致的域迁移问题。论文提出CoSSeg-TTA框架，结合半监督学习、基于直方图的风格迁移和测试时自适应策略，提升分割精度和跨中心泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.04243v1](http://arxiv.org/pdf/2510.04243v1)**

> **作者:** Jincan Lou; Jingkun Chen; Haoquan Li; Hang Li; Wenjian Huang; Weihua Chen; Fan Wang; Jianguo Zhang
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Accurate liver segmentation from contrast-enhanced MRI is essential for diagnosis, treatment planning, and disease monitoring. However, it remains challenging due to limited annotated data, heterogeneous enhancement protocols, and significant domain shifts across scanners and institutions. Traditional image-to-image translation frameworks have made great progress in domain generalization, but their application is not straightforward. For example, Pix2Pix requires image registration, and cycle-GAN cannot be integrated seamlessly into segmentation pipelines. Meanwhile, these methods are originally used to deal with cross-modality scenarios, and often introduce structural distortions and suffer from unstable training, which may pose drawbacks in our single-modality scenario. To address these challenges, we propose CoSSeg-TTA, a compact segmentation framework for the GED4 (Gd-EOB-DTPA enhanced hepatobiliary phase MRI) modality built upon nnU-Netv2 and enhanced with a semi-supervised mean teacher scheme to exploit large amounts of unlabeled volumes. A domain adaptation module, incorporating a randomized histogram-based style appearance transfer function and a trainable contrast-aware network, enriches domain diversity and mitigates cross-center variability. Furthermore, a continual test-time adaptation strategy is employed to improve robustness during inference. Extensive experiments demonstrate that our framework consistently outperforms the nnU-Netv2 baseline, achieving superior Dice score and Hausdorff Distance while exhibiting strong generalization to unseen domains under low-annotation conditions.
>
---
#### [new 143] SDAKD: Student Discriminator Assisted Knowledge Distillation for Super-Resolution Generative Adversarial Networks
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决生成对抗网络（GAN）因计算量大难以部署在资源受限设备上的问题。论文提出SDAKD方法，通过引入学生判别器缓解学生生成器与教师判别器之间的能力不匹配，采用三阶段训练策略和特征图蒸馏，提升了小模型的生成效果。**

- **链接: [http://arxiv.org/pdf/2510.03870v1](http://arxiv.org/pdf/2510.03870v1)**

> **作者:** Nikolaos Kaparinos; Vasileios Mezaris
>
> **备注:** Under review
>
> **摘要:** Generative Adversarial Networks (GANs) achieve excellent performance in generative tasks, such as image super-resolution, but their computational requirements make difficult their deployment on resource-constrained devices. While knowledge distillation is a promising research direction for GAN compression, effectively training a smaller student generator is challenging due to the capacity mismatch between the student generator and the teacher discriminator. In this work, we propose Student Discriminator Assisted Knowledge Distillation (SDAKD), a novel GAN distillation methodology that introduces a student discriminator to mitigate this capacity mismatch. SDAKD follows a three-stage training strategy, and integrates an adapted feature map distillation approach in its last two training stages. We evaluated SDAKD on two well-performing super-resolution GANs, GCFSR and Real-ESRGAN. Our experiments demonstrate consistent improvements over the baselines and SOTA GAN knowledge distillation methods. The SDAKD source code will be made openly available upon acceptance of the paper.
>
---
#### [new 144] Learned Display Radiance Fields with Lensless Cameras
- **分类: cs.CV; cs.ET**

- **简介: 该论文属于显示校准任务，旨在解决传统校准需专业设备和环境的问题。作者设计了一种无透镜相机与隐式神经表示算法，实现了在复杂视角下对显示器发光场的高效重建，为便捷校准提供了新方法。**

- **链接: [http://arxiv.org/pdf/2510.03356v1](http://arxiv.org/pdf/2510.03356v1)**

> **作者:** Ziyang Chen; Yuta Itoh; Kaan Akşit
>
> **摘要:** Calibrating displays is a basic and regular task that content creators must perform to maintain optimal visual experience, yet it remains a troublesome issue. Measuring display characteristics from different viewpoints often requires specialized equipment and a dark room, making it inaccessible to most users. To avoid specialized hardware requirements in display calibrations, our work co-designs a lensless camera and an Implicit Neural Representation based algorithm for capturing display characteristics from various viewpoints. More specifically, our pipeline enables efficient reconstruction of light fields emitted from a display from a viewing cone of 46.6{\deg} X 37.6{\deg}. Our emerging pipeline paves the initial steps towards effortless display calibration and characterization.
>
---
#### [new 145] TBStar-Edit: From Image Editing Pattern Shifting to Consistency Enhancement
- **分类: cs.CV**

- **简介: 该论文提出TBStar-Edit，专注于提升电商场景中图像编辑的一致性。通过数据构建、分层模型架构及两阶段训练策略，实现高保真且保持产品外观一致的图像编辑。**

- **链接: [http://arxiv.org/pdf/2510.04483v1](http://arxiv.org/pdf/2510.04483v1)**

> **作者:** Hao Fang; Zechao Zhan; Weixin Feng; Ziwei Huang; XuBin Li; Tiezheng Ge
>
> **摘要:** Recent advances in image generation and editing technologies have enabled state-of-the-art models to achieve impressive results in general domains. However, when applied to e-commerce scenarios, these general models often encounter consistency limitations. To address this challenge, we introduce TBStar-Edit, an new image editing model tailored for the e-commerce domain. Through rigorous data engineering, model architecture design and training strategy, TBStar-Edit achieves precise and high-fidelity image editing while maintaining the integrity of product appearance and layout. Specifically, for data engineering, we establish a comprehensive data construction pipeline, encompassing data collection, construction, filtering, and augmentation, to acquire high-quality, instruction-following, and strongly consistent editing data to support model training. For model architecture design, we design a hierarchical model framework consisting of a base model, pattern shifting modules, and consistency enhancement modules. For model training, we adopt a two-stage training strategy to enhance the consistency preservation: first stage for editing pattern shifting, and second stage for consistency enhancement. Each stage involves training different modules with separate datasets. Finally, we conduct extensive evaluations of TBStar-Edit on a self-proposed e-commerce benchmark, and the results demonstrate that TBStar-Edit outperforms existing general-domain editing models in both objective metrics (VIE Score) and subjective user preference.
>
---
#### [new 146] ConceptSplit: Decoupled Multi-Concept Personalization of Diffusion Models via Token-wise Adaptation and Attention Disentanglement
- **分类: cs.CV**

- **简介: 该论文属于文本到图像扩散模型的多概念个性化任务，旨在解决多个概念在生成图像时相互干扰的问题。论文提出了ConceptSplit框架，包含Token-wise Value Adaptation（ToVA）和Latent Optimization for Disentangled Attention（LODA）两种方法，分别在训练和推理阶段解耦概念，实现更精确的多主体图像生成。**

- **链接: [http://arxiv.org/pdf/2510.04668v1](http://arxiv.org/pdf/2510.04668v1)**

> **作者:** Habin Lim; Yeongseob Won; Juwon Seo; Gyeong-Moon Park
>
> **备注:** 14 pages, 13 figures, to be published in ICCV 2025
>
> **摘要:** In recent years, multi-concept personalization for text-to-image (T2I) diffusion models to represent several subjects in an image has gained much more attention. The main challenge of this task is "concept mixing", where multiple learned concepts interfere or blend undesirably in the output image. To address this issue, in this paper, we present ConceptSplit, a novel framework to split the individual concepts through training and inference. Our framework comprises two key components. First, we introduce Token-wise Value Adaptation (ToVA), a merging-free training method that focuses exclusively on adapting the value projection in cross-attention. Based on our empirical analysis, we found that modifying the key projection, a common approach in existing methods, can disrupt the attention mechanism and lead to concept mixing. Second, we propose Latent Optimization for Disentangled Attention (LODA), which alleviates attention entanglement during inference by optimizing the input latent. Through extensive qualitative and quantitative experiments, we demonstrate that ConceptSplit achieves robust multi-concept personalization, mitigating unintended concept interference. Code is available at https://github.com/KU-VGI/ConceptSplit
>
---
#### [new 147] Beyond the Seen: Bounded Distribution Estimation for Open-Vocabulary Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于开放词汇学习任务，旨在解决开放环境中数据分布估计问题。现有方法仅用已见类数据估计分布，误差无法识别。作者提出通过生成未见类数据来界定分布估计误差，并设计了数据生成与分布对齐方法。实验表明新方法性能优于基线模型。**

- **链接: [http://arxiv.org/pdf/2510.04770v1](http://arxiv.org/pdf/2510.04770v1)**

> **作者:** Xiaomeng Fan; Yuchuan Mao; Zhi Gao; Yuwei Wu; Jin Chen; Yunde Jia
>
> **摘要:** Open-vocabulary learning requires modeling the data distribution in open environments, which consists of both seen-class and unseen-class data. Existing methods estimate the distribution in open environments using seen-class data, where the absence of unseen classes makes the estimation error inherently unidentifiable. Intuitively, learning beyond the seen classes is crucial for distribution estimation to bound the estimation error. We theoretically demonstrate that the distribution can be effectively estimated by generating unseen-class data, through which the estimation error is upper-bounded. Building on this theoretical insight, we propose a novel open-vocabulary learning method, which generates unseen-class data for estimating the distribution in open environments. The method consists of a class-domain-wise data generation pipeline and a distribution alignment algorithm. The data generation pipeline generates unseen-class data under the guidance of a hierarchical semantic tree and domain information inferred from the seen-class data, facilitating accurate distribution estimation. With the generated data, the distribution alignment algorithm estimates and maximizes the posterior probability to enhance generalization in open-vocabulary learning. Extensive experiments on $11$ datasets demonstrate that our method outperforms baseline approaches by up to $14\%$, highlighting its effectiveness and superiority.
>
---
#### [new 148] Read the Room: Inferring Social Context Through Dyadic Interaction Recognition in Cyber-physical-social Infrastructure Systems
- **分类: cs.CV**

- **简介: 该论文属于人机交互与社会计算任务，旨在通过识别双人互动行为，提升对人类社会性行为的理解。为解决隐私问题，采用深度传感器分析骨架动作，比较了五种算法在12种互动类型上的表现，以期为社会性基础设施系统提供技术支持。**

- **链接: [http://arxiv.org/pdf/2510.04854v1](http://arxiv.org/pdf/2510.04854v1)**

> **作者:** Cheyu Lin; John Martins; Katherine A. Flanigan; Ph. D
>
> **备注:** ASCE International Conference on Computing in Civil Engineering 2024
>
> **摘要:** Cyber-physical systems (CPS) integrate sensing, computing, and control to improve infrastructure performance, focusing on economic goals like performance and safety. However, they often neglect potential human-centered (or ''social'') benefits. Cyber-physical-social infrastructure systems (CPSIS) aim to address this by aligning CPS with social objectives. This involves defining social benefits, understanding human interactions with each other and infrastructure, developing privacy-preserving measurement methods, modeling these interactions for prediction, linking them to social benefits, and actuating the physical environment to foster positive social outcomes. This paper delves into recognizing dyadic human interactions using real-world data, which is the backbone to measuring social behavior. This lays a foundation to address the need to enhance understanding of the deeper meanings and mutual responses inherent in human interactions. While RGB cameras are informative for interaction recognition, privacy concerns arise. Depth sensors offer a privacy-conscious alternative by analyzing skeletal movements. This study compares five skeleton-based interaction recognition algorithms on a dataset of 12 dyadic interactions. Unlike single-person datasets, these interactions, categorized into communication types like emblems and affect displays, offer insights into the cultural and emotional aspects of human interactions.
>
---
#### [new 149] Anomaly-Aware YOLO: A Frugal yet Robust Approach to Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 论文属于红外小目标检测任务，旨在解决复杂背景和小目标导致的误报问题。作者提出AA-YOLO，在检测头中引入统计异常检测，将小目标视为背景中的异常模式，从而有效控制误报率。该方法在多种YOLO模型上通用，尤其适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2510.04741v1](http://arxiv.org/pdf/2510.04741v1)**

> **作者:** Alina Ciocarlan; Sylvie Le Hégarat-Mascle; Sidonie Lefebvre
>
> **摘要:** Infrared Small Target Detection (IRSTD) is a challenging task in defense applications, where complex backgrounds and tiny target sizes often result in numerous false alarms using conventional object detectors. To overcome this limitation, we propose Anomaly-Aware YOLO (AA-YOLO), which integrates a statistical anomaly detection test into its detection head. By treating small targets as unexpected patterns against the background, AA-YOLO effectively controls the false alarm rate. Our approach not only achieves competitive performance on several IRSTD benchmarks, but also demonstrates remarkable robustness in scenarios with limited training data, noise, and domain shifts. Furthermore, since only the detection head is modified, our design is highly generic and has been successfully applied across various YOLO backbones, including lightweight models. It also provides promising results when integrated into an instance segmentation YOLO. This versatility makes AA-YOLO an attractive solution for real-world deployments where resources are constrained. The code will be publicly released.
>
---
#### [new 150] From Filters to VLMs: Benchmarking Defogging Methods through Object Detection and Segmentation Performance
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决雾天条件下自动驾驶感知系统性能下降的问题。作者系统评估了多种去雾方法（包括传统滤波器、深度学习模型及VLMs）在目标检测和语义分割中的效果，揭示了去雾处理在何种情况下能真正提升感知性能。**

- **链接: [http://arxiv.org/pdf/2510.03906v1](http://arxiv.org/pdf/2510.03906v1)**

> **作者:** Ardalan Aryashad; Parsa Razmara; Amin Mahjoub; Seyedarmin Azizi; Mahdi Salmani; Arad Firouzkouhi
>
> **摘要:** Autonomous driving perception systems are particularly vulnerable in foggy conditions, where light scattering reduces contrast and obscures fine details critical for safe operation. While numerous defogging methods exist-from handcrafted filters to learned restoration models-improvements in image fidelity do not consistently translate into better downstream detection and segmentation. Moreover, prior evaluations often rely on synthetic data, leaving questions about real-world transferability. We present a structured empirical study that benchmarks a comprehensive set of pipelines, including (i) classical filters, (ii) modern defogging networks, (iii) chained variants (filter$\rightarrow$model, model$\rightarrow$filter), and (iv) prompt-driven visual--language image editing models (VLM) applied directly to foggy images. Using Foggy Cityscapes, we assess both image quality and downstream performance on object detection (mAP) and segmentation (PQ, RQ, SQ). Our analysis reveals when defogging helps, when chaining yields synergy or degradation, and how VLM-based editors compare to dedicated approaches. In addition, we evaluate qualitative rubric-based scores from a VLM judge and quantify their alignment with task metrics, showing strong correlations with mAP. Together, these results establish a transparent, task-oriented benchmark for defogging methods and highlight the conditions under which preprocessing genuinely improves autonomous perception in adverse weather.
>
---
#### [new 151] DiT-VTON: Diffusion Transformer Framework for Unified Multi-Category Virtual Try-On and Virtual Try-All with Integrated Image Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于虚拟试穿（VTO）任务，旨在解决细节保留、真实图像鲁棒性、采样效率及跨品类泛化等问题。作者提出了DiT-VTON框架，基于扩散Transformer进行图像条件建模，探索多种配置并扩展数据集，实现虚拟试穿与图像编辑功能。**

- **链接: [http://arxiv.org/pdf/2510.04797v1](http://arxiv.org/pdf/2510.04797v1)**

> **作者:** Qi Li; Shuwen Qiu; Julien Han; Xingzi Xu; Mehmet Saygin Seyfioglu; Kee Kiat Koo; Karim Bouyarmane
>
> **备注:** Submitted to CVPR 2025 and Published at CVPR 2025 AI for Content Creation workshop
>
> **摘要:** The rapid growth of e-commerce has intensified the demand for Virtual Try-On (VTO) technologies, enabling customers to realistically visualize products overlaid on their own images. Despite recent advances, existing VTO models face challenges with fine-grained detail preservation, robustness to real-world imagery, efficient sampling, image editing capabilities, and generalization across diverse product categories. In this paper, we present DiT-VTON, a novel VTO framework that leverages a Diffusion Transformer (DiT), renowned for its performance on text-conditioned image generation, adapted here for the image-conditioned VTO task. We systematically explore multiple DiT configurations, including in-context token concatenation, channel concatenation, and ControlNet integration, to determine the best setup for VTO image conditioning. To enhance robustness, we train the model on an expanded dataset encompassing varied backgrounds, unstructured references, and non-garment categories, demonstrating the benefits of data scaling for VTO adaptability. DiT-VTON also redefines the VTO task beyond garment try-on, offering a versatile Virtual Try-All (VTA) solution capable of handling a wide range of product categories and supporting advanced image editing functionalities such as pose preservation, localized editing, texture transfer, and object-level customization. Experimental results show that our model surpasses state-of-the-art methods on VITON-HD, achieving superior detail preservation and robustness without reliance on additional condition encoders. It also outperforms models with VTA and image editing capabilities on a diverse dataset spanning thousands of product categories.
>
---
#### [new 152] Exploring the Efficacy of Modified Transfer Learning in Identifying Parkinson's Disease Through Drawn Image Patterns
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在通过机器学习方法早期诊断帕金森病。利用手绘螺旋和波浪图像作为生物标志物，结合卷积神经网络、迁移学习和集成投票策略，提升模型性能。数据增强提高多样性，最终模型在螺旋图像上达到90%的加权平均精度、召回率和F1分数，在波浪图像上达到96.67%，集成后整体准确率达93.3%。**

- **链接: [http://arxiv.org/pdf/2510.05015v1](http://arxiv.org/pdf/2510.05015v1)**

> **作者:** Nabil Daiyan; Md Rakibul Haque
>
> **备注:** 5 pages, 11 figures, published on 2024 2nd International Conference on Information and Communication Technology (ICICT 2024)
>
> **摘要:** Parkinson's disease (PD) is a progressive neurodegenerative condition characterized by the death of dopaminergic neurons, leading to various movement disorder symptoms. Early diagnosis of PD is crucial to prevent adverse effects, yet traditional diagnostic methods are often cumbersome and costly. In this study, a machine learning-based approach is proposed using hand-drawn spiral and wave images as potential biomarkers for PD detection. Our methodology leverages convolutional neural networks (CNNs), transfer learning, and attention mechanisms to improve model performance and resilience against overfitting. To enhance the diversity and richness of both spiral and wave categories, the training dataset undergoes augmentation to increase the number of images. The proposed architecture comprises three phases: utilizing pre-trained CNNs, incorporating custom convolutional layers, and ensemble voting. Employing hard voting further enhances performance by aggregating predictions from multiple models. Experimental results show promising accuracy rates. For spiral images, weighted average precision, recall, and F1-score are 90%, and for wave images, they are 96.67%. After combining the predictions through ensemble hard voting, the overall accuracy is 93.3%. These findings underscore the potential of machine learning in early PD diagnosis, offering a non-invasive and cost-effective solution to improve patient outcomes.
>
---
#### [new 153] Spatial-ViLT: Enhancing Visual Spatial Reasoning through Multi-Task Learning
- **分类: cs.CV; cs.AI; cs.LG; 68T45, 68T10, 68T40**

- **简介: 该论文属于视觉-语言建模任务，旨在解决现有模型在三维场景和复杂物体构型中的空间推理能力不足的问题。作者提出了SpatialViLT及其变体，通过多任务学习框架融合深度图、三维坐标和边缘图等空间特征，提升模型的空间理解能力，并在视觉空间推理数据集上取得了先进性能。**

- **链接: [http://arxiv.org/pdf/2510.03441v1](http://arxiv.org/pdf/2510.03441v1)**

> **作者:** Chashi Mahiul Islam; Oteo Mamo; Samuel Jacob Chacko; Xiuwen Liu; Weikuan Yu
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Vision-language models (VLMs) have advanced multimodal reasoning but still face challenges in spatial reasoning for 3D scenes and complex object configurations. To address this, we introduce SpatialViLT, an enhanced VLM that integrates spatial features like depth maps, 3D coordinates, and edge maps through a multi-task learning framework. This approach enriches multimodal embeddings with spatial understanding. We propose two variants: SpatialViLT and MaskedSpatialViLT, focusing on full and masked object regions, respectively. Additionally, SpatialEnsemble combines both approaches, achieving state-of-the-art accuracy. Our models excel in spatial reasoning categories such as directional, topological, and proximity relations, as demonstrated on the challenging Visual Spatial Reasoning (VSR) dataset. This work represents a significant step in enhancing the spatial intelligence of AI systems, crucial for advanced multimodal understanding and real-world applications.
>
---
#### [new 154] Concept-Based Masking: A Patch-Agnostic Defense Against Adversarial Patch Attacks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉与模型鲁棒性任务，旨在解决对抗补丁攻击问题。现有方法依赖对补丁大小或位置的先验知识，限制了应用范围。本文提出一种无需补丁先验知识的防御方法，利用基于概念的解释识别并抑制最具影响力的特征向量，从而消除补丁影响。实验表明该方法在多种补丁大小和位置下均表现优异。**

- **链接: [http://arxiv.org/pdf/2510.04245v1](http://arxiv.org/pdf/2510.04245v1)**

> **作者:** Ayushi Mehrotra; Derek Peng; Dipkamal Bhusal; Nidhi Rastogi
>
> **备注:** neurips workshop
>
> **摘要:** Adversarial patch attacks pose a practical threat to deep learning models by forcing targeted misclassifications through localized perturbations, often realized in the physical world. Existing defenses typically assume prior knowledge of patch size or location, limiting their applicability. In this work, we propose a patch-agnostic defense that leverages concept-based explanations to identify and suppress the most influential concept activation vectors, thereby neutralizing patch effects without explicit detection. Evaluated on Imagenette with a ResNet-50, our method achieves higher robust and clean accuracy than the state-of-the-art PatchCleanser, while maintaining strong performance across varying patch sizes and locations. Our results highlight the promise of combining interpretability with robustness and suggest concept-driven defenses as a scalable strategy for securing machine learning models against adversarial patch attacks.
>
---
#### [new 155] Flow Matching for Conditional MRI-CT and CBCT-CT Image Synthesis
- **分类: cs.CV**

- **简介: 该论文属于医学图像合成任务，旨在解决从MRI或CBCT生成合成CT（sCT）图像的问题，以用于放疗中减少辐射。作者采用3D Flow Matching框架，通过学习速度场将噪声转换为sCT，并在SynthRAD2025数据集上验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2510.04823v1](http://arxiv.org/pdf/2510.04823v1)**

> **作者:** Arnela Hadzic; Simon Johannes Joham; Martin Urschler
>
> **摘要:** Generating synthetic CT (sCT) from MRI or CBCT plays a crucial role in enabling MRI-only and CBCT-based adaptive radiotherapy, improving treatment precision while reducing patient radiation exposure. To address this task, we adopt a fully 3D Flow Matching (FM) framework, motivated by recent work demonstrating FM's efficiency in producing high-quality images. In our approach, a Gaussian noise volume is transformed into an sCT image by integrating a learned FM velocity field, conditioned on features extracted from the input MRI or CBCT using a lightweight 3D encoder. We evaluated the method on the SynthRAD2025 Challenge benchmark, training separate models for MRI $\rightarrow$ sCT and CBCT $\rightarrow$ sCT across three anatomical regions: abdomen, head and neck, and thorax. Validation and testing were performed through the challenge submission system. The results indicate that the method accurately reconstructs global anatomical structures; however, preservation of fine details was limited, primarily due to the relatively low training resolution imposed by memory and runtime constraints. Future work will explore patch-based training and latent-space flow models to improve resolution and local structural fidelity.
>
---
#### [new 156] EduPersona: Benchmarking Subjective Ability Boundaries of Virtual Student Agents
- **分类: cs.CV; cs.CY**

- **简介: 该论文属于教育AI任务，旨在解决虚拟学生代理在课堂场景中的主观能力评估问题。作者构建了EduPersona基准，包含多语言、多学科的真实课堂对话数据，并提出三层评估任务：基础连贯性、学生真实感和长期人设一致性。通过实验验证数据集和框架的有效性，推动可信教育AI的发展。**

- **链接: [http://arxiv.org/pdf/2510.04648v1](http://arxiv.org/pdf/2510.04648v1)**

> **作者:** Buyuan Zhu; Shiyu Hu; Yiping Ma; Yuanming Zhang; Kang Hao Cheong
>
> **备注:** Preprint, Under review
>
> **摘要:** As large language models are increasingly integrated into education, virtual student agents are becoming vital for classroom simulation and teacher training. Yet their classroom-oriented subjective abilities remain largely unassessed, limiting understanding of model boundaries and hindering trustworthy deployment. We present EduPersona, a large-scale benchmark spanning two languages, three subjects, and ten persona types based on the Big Five theory. The dataset contains 1,308 authentic classroom dialogue rounds, corresponding to 12,814 teacher-student Q&A turns, and is further expanded through persona stylization into roughly 10 times larger scale (128k turns), providing a solid foundation for evaluation. Building on this resource, we decompose hard-to-quantify subjective performance into three progressive tasks: TASK1 basic coherence (whether behavior, emotion, expression, and voice align with classroom context), TASK2 student realism, and TASK3 long-term persona consistency, thereby establishing an evaluation framework grounded in educational theory and research value. We conduct systematic experiments on three representative LLMs, comparing their original versions with ten persona-fine-tuned variants trained on EduPersona. Results show consistent and significant average improvements across all tasks: TASK1 +33.6%, TASK2 +30.6%, and TASK3 +14.9%. These improvements highlight the dataset's effectiveness and research value, while also revealing the heterogeneous difficulty of persona modeling. In summary, EduPersona delivers the first classroom benchmark centered on subjective abilities, establishes a decoupled and verifiable research paradigm, and we will open-source both the dataset and the framework to support the broader research community in advancing trustworthy and human-like AI for education.
>
---
#### [new 157] Federated Learning for Surgical Vision in Appendicitis Classification: Results of the FedSurg EndoVis 2024 Challenge
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医疗图像分类任务，旨在解决不同临床中心间模型泛化与个性化问题。通过联邦学习（FedSurg）挑战赛，评估多种方法在不共享数据下的阑尾炎视频分类性能，强调模型架构、预处理和损失函数设计的重要性。**

- **链接: [http://arxiv.org/pdf/2510.04772v1](http://arxiv.org/pdf/2510.04772v1)**

> **作者:** Max Kirchner; Hanna Hoffmann; Alexander C. Jenke; Oliver L. Saldanha; Kevin Pfeiffer; Weam Kanjo; Julia Alekseenko; Claas de Boer; Santhi Raj Kolamuri; Lorenzo Mazza; Nicolas Padoy; Sophia Bano; Annika Reinke; Lena Maier-Hein; Danail Stoyanov; Jakob N. Kather; Fiona R. Kolbinger; Sebastian Bodenstedt; Stefanie Speidel
>
> **备注:** A challenge report pre-print (31 pages), including 7 tables and 8 figures
>
> **摘要:** Purpose: The FedSurg challenge was designed to benchmark the state of the art in federated learning for surgical video classification. Its goal was to assess how well current methods generalize to unseen clinical centers and adapt through local fine-tuning while enabling collaborative model development without sharing patient data. Methods: Participants developed strategies to classify inflammation stages in appendicitis using a preliminary version of the multi-center Appendix300 video dataset. The challenge evaluated two tasks: generalization to an unseen center and center-specific adaptation after fine-tuning. Submitted approaches included foundation models with linear probing, metric learning with triplet loss, and various FL aggregation schemes (FedAvg, FedMedian, FedSAM). Performance was assessed using F1-score and Expected Cost, with ranking robustness evaluated via bootstrapping and statistical testing. Results: In the generalization task, performance across centers was limited. In the adaptation task, all teams improved after fine-tuning, though ranking stability was low. The ViViT-based submission achieved the strongest overall performance. The challenge highlighted limitations in generalization, sensitivity to class imbalance, and difficulties in hyperparameter tuning in decentralized training, while spatiotemporal modeling and context-aware preprocessing emerged as promising strategies. Conclusion: The FedSurg Challenge establishes the first benchmark for evaluating FL strategies in surgical video classification. Findings highlight the trade-off between local personalization and global robustness, and underscore the importance of architecture choice, preprocessing, and loss design. This benchmarking offers a reference point for future development of imbalance-aware, adaptive, and robust FL methods in clinical surgical AI.
>
---
#### [new 158] Object-Centric Representation Learning for Enhanced 3D Scene Graph Prediction
- **分类: cs.CV**

- **简介: 该论文属于3D场景图预测任务，旨在提升物体特征表示以增强场景图的准确性。现有方法依赖图神经网络，但物体和关系特征的表达能力不足。为此，作者设计了高判别力的物体特征编码器，并采用对比预训练策略，解耦物体表示学习与场景图预测，同时融合几何与语义特征以提升关系预测。实验表明该方法在3DSSG数据集上显著优于现有技术。**

- **链接: [http://arxiv.org/pdf/2510.04714v1](http://arxiv.org/pdf/2510.04714v1)**

> **作者:** KunHo Heo; GiHyun Kim; SuYeon Kim; MyeongAh Cho
>
> **备注:** Accepted by NeurIPS 2025. Code: https://github.com/VisualScienceLab-KHU/OCRL-3DSSG-Codes
>
> **摘要:** 3D Semantic Scene Graph Prediction aims to detect objects and their semantic relationships in 3D scenes, and has emerged as a crucial technology for robotics and AR/VR applications. While previous research has addressed dataset limitations and explored various approaches including Open-Vocabulary settings, they frequently fail to optimize the representational capacity of object and relationship features, showing excessive reliance on Graph Neural Networks despite insufficient discriminative capability. In this work, we demonstrate through extensive analysis that the quality of object features plays a critical role in determining overall scene graph accuracy. To address this challenge, we design a highly discriminative object feature encoder and employ a contrastive pretraining strategy that decouples object representation learning from the scene graph prediction. This design not only enhances object classification accuracy but also yields direct improvements in relationship prediction. Notably, when plugging in our pretrained encoder into existing frameworks, we observe substantial performance improvements across all evaluation metrics. Additionally, whereas existing approaches have not fully exploited the integration of relationship information, we effectively combine both geometric and semantic features to achieve superior relationship prediction. Comprehensive experiments on the 3DSSG dataset demonstrate that our approach significantly outperforms previous state-of-the-art methods. Our code is publicly available at https://github.com/VisualScienceLab-KHU/OCRL-3DSSG-Codes.
>
---
#### [new 159] Sonar Image Datasets: A Comprehensive Survey of Resources, Challenges, and Applications
- **分类: cs.CV; I.4.9; I.5.0; H.3.1; I.2.6**

- **简介: 该论文属于数据集综述任务，旨在解决水下声呐图像数据集稀缺、标注不足的问题。论文全面调研了多种声呐图像数据集，分析其应用场景与特性，并总结成表格和时间线，为研究者提供参考与研究方向。**

- **链接: [http://arxiv.org/pdf/2510.03353v1](http://arxiv.org/pdf/2510.03353v1)**

> **作者:** Larissa S. Gomes; Gustavo P. Almeida; Bryan U. Moreira; Marco Quiroz; Breno Xavier; Lucas Soares; Stephanie L. Brião; Felipe G. Oliveira; Paulo L. J. Drews-Jr
>
> **备注:** Published in the Conference on Graphics, Patterns and Images (SIBGRAPI). This 4-page paper presents a timeline of publicly available datasets up to the year 2025
>
> **摘要:** Sonar images are relevant for advancing underwater exploration, autonomous navigation, and ecosystem monitoring. However, the progress depends on data availability. The scarcity of publicly available, well-annotated sonar image datasets creates a significant bottleneck for the development of robust machine learning models. This paper presents a comprehensive and concise review of the current landscape of sonar image datasets, seeking not only to catalog existing resources but also to contextualize them, identify gaps, and provide a clear roadmap, serving as a base guide for researchers of any kind who wish to start or advance in the field of underwater acoustic data analysis. We mapped publicly accessible datasets across various sonar modalities, including Side Scan Sonar (SSS), Forward-Looking Sonar (FLS), Synthetic Aperture Sonar (SAS), Multibeam Echo Sounder (MBES), and Dual-Frequency Identification Sonar (DIDSON). An analysis was conducted on applications such as classification, detection, segmentation, and 3D reconstruction. This work focuses on state-of-the-art advancements, incorporating newly released datasets. The findings are synthesized into a master table and a chronological timeline, offering a clear and accessible comparison of characteristics, sizes, and annotation details datasets.
>
---
#### [new 160] Optimized Minimal 4D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于动态场景表示任务，旨在解决4D高斯点绘方法中的存储开销问题。作者提出OMG4框架，通过三阶段逐步剪枝高斯点并引入压缩技术，在保持高质量重建的同时显著减小模型体积，提升了压缩比与视觉效果。**

- **链接: [http://arxiv.org/pdf/2510.03857v1](http://arxiv.org/pdf/2510.03857v1)**

> **作者:** Minseo Lee; Byeonghyeon Lee; Lucas Yunkyu Lee; Eunsoo Lee; Sangmin Kim; Seunghyeon Song; Joo Chan Lee; Jong Hwan Ko; Jaesik Park; Eunbyung Park
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** 4D Gaussian Splatting has emerged as a new paradigm for dynamic scene representation, enabling real-time rendering of scenes with complex motions. However, it faces a major challenge of storage overhead, as millions of Gaussians are required for high-fidelity reconstruction. While several studies have attempted to alleviate this memory burden, they still face limitations in compression ratio or visual quality. In this work, we present OMG4 (Optimized Minimal 4D Gaussian Splatting), a framework that constructs a compact set of salient Gaussians capable of faithfully representing 4D Gaussian models. Our method progressively prunes Gaussians in three stages: (1) Gaussian Sampling to identify primitives critical to reconstruction fidelity, (2) Gaussian Pruning to remove redundancies, and (3) Gaussian Merging to fuse primitives with similar characteristics. In addition, we integrate implicit appearance compression and generalize Sub-Vector Quantization (SVQ) to 4D representations, further reducing storage while preserving quality. Extensive experiments on standard benchmark datasets demonstrate that OMG4 significantly outperforms recent state-of-the-art methods, reducing model sizes by over 60% while maintaining reconstruction quality. These results position OMG4 as a significant step forward in compact 4D scene representation, opening new possibilities for a wide range of applications. Our source code is available at https://minshirley.github.io/OMG4/.
>
---
#### [new 161] Video-in-the-Loop: Span-Grounded Long Video QA with Interleaved Reasoning
- **分类: cs.CV**

- **简介: 该论文属于长视频问答（QA）任务，旨在解决视频理解中高效定位与推理的问题。论文提出了“Video-in-the-Loop”（ViTL）框架，通过两阶段方法在固定计算资源下提升视频问答性能。首先低帧率定位相关视频片段，再在关键片段内重分配视觉资源进行精细推理，输出答案及对应时间区间。同时构建了基于时间区间标注的数据集，提升模型可解释性与效率。**

- **链接: [http://arxiv.org/pdf/2510.04022v1](http://arxiv.org/pdf/2510.04022v1)**

> **作者:** Chendong Wang; Donglin Bai; Yifan Yang; Xiao Jin; Anlan Zhang; Rui Wang; Shiqi Jiang; Yuqing Yang; Hao Wu; Qi Dai; Chong Luo; Ting Cao; Lili Qiu; Suman Banerjee
>
> **摘要:** We present \emph{Video-in-the-Loop} (ViTL), a two-stage long-video QA framework that preserves a fixed token budget by first \emph{localizing} question-relevant interval(s) with a low-fps skim and then \emph{answering} via span-aware reallocation of visual tokens at higher effective frame rate, emitting an interleaved output with both spans and the final option for direct attribution. We also introduce \dataname{}, which converts description based event graphs into \emph{span-grounded} multiple-choice QA by pairing each question with \emph{ground-truth} time span(s) and related reasoning. ViTL is trained end-to-end with an interleaved group-relative objective that couples temporal IoU for localization with answer correctness, allowing credit to flow from answers back to spans without increasing compute. Under fixed token budgets, ViTL attains up to 8.6% with 50% less frame input on long-video QA and temporal grounding (e.g., Charades-STA, ActivityNet-Captions) and ablations show that span-aware token reallocation consistently surpasses uniform sampling. Together, \dataname{} and ViTL provide an interpretable, compute-efficient recipe for scalable long-video QA.
>
---
#### [new 162] Diffusion-Classifier Synergy: Reward-Aligned Learning via Mutual Boosting Loop for FSCIL
- **分类: cs.CV**

- **简介: 该论文属于少样本类增量学习（FSCIL）任务，旨在解决新类学习与旧知识遗忘的平衡问题。作者提出Diffusion-Classifier Synergy（DCS）框架，通过扩散模型与分类器的协同学习，利用动态奖励机制提升数据增强效果，从而改善模型的泛化能力与类别区分度。**

- **链接: [http://arxiv.org/pdf/2510.03608v1](http://arxiv.org/pdf/2510.03608v1)**

> **作者:** Ruitao Wu; Yifan Zhao; Guangyao Chen; Jia Li
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Few-Shot Class-Incremental Learning (FSCIL) challenges models to sequentially learn new classes from minimal examples without forgetting prior knowledge, a task complicated by the stability-plasticity dilemma and data scarcity. Current FSCIL methods often struggle with generalization due to their reliance on limited datasets. While diffusion models offer a path for data augmentation, their direct application can lead to semantic misalignment or ineffective guidance. This paper introduces Diffusion-Classifier Synergy (DCS), a novel framework that establishes a mutual boosting loop between diffusion model and FSCIL classifier. DCS utilizes a reward-aligned learning strategy, where a dynamic, multi-faceted reward function derived from the classifier's state directs the diffusion model. This reward system operates at two levels: the feature level ensures semantic coherence and diversity using prototype-anchored maximum mean discrepancy and dimension-wise variance matching, while the logits level promotes exploratory image generation and enhances inter-class discriminability through confidence recalibration and cross-session confusion-aware mechanisms. This co-evolutionary process, where generated images refine the classifier and an improved classifier state yields better reward signals, demonstrably achieves state-of-the-art performance on FSCIL benchmarks, significantly enhancing both knowledge retention and new class learning.
>
---
#### [new 163] Pathology-CoT: Learning Visual Chain-of-Thought Agent from Expert Whole Slide Image Diagnosis Behavior
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决病理切片图像诊断中缺乏实用化、可解释的AI系统问题。作者通过记录专家浏览行为，构建了Pathology-CoT数据集，并开发了Pathologist-o3代理系统，实现精准的胃肠道淋巴结转移检测，提升了诊断性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.04587v1](http://arxiv.org/pdf/2510.04587v1)**

> **作者:** Sheng Wang; Ruiming Wu; Charles Herndon; Yihang Liu; Shunsuke Koga; Jeanne Shen; Zhi Huang
>
> **摘要:** Diagnosing a whole-slide image is an interactive, multi-stage process involving changes in magnification and movement between fields. Although recent pathology foundation models are strong, practical agentic systems that decide what field to examine next, adjust magnification, and deliver explainable diagnoses are still lacking. The blocker is data: scalable, clinically aligned supervision of expert viewing behavior that is tacit and experience-based, not written in textbooks or online, and therefore absent from large language model training. We introduce the AI Session Recorder, which works with standard WSI viewers to unobtrusively record routine navigation and convert the viewer logs into standardized behavioral commands (inspect or peek at discrete magnifications) and bounding boxes. A lightweight human-in-the-loop review turns AI-drafted rationales into the Pathology-CoT dataset, a form of paired "where to look" and "why it matters" supervision produced at roughly six times lower labeling time. Using this behavioral data, we build Pathologist-o3, a two-stage agent that first proposes regions of interest and then performs behavior-guided reasoning. On gastrointestinal lymph-node metastasis detection, it achieved 84.5% precision, 100.0% recall, and 75.4% accuracy, exceeding the state-of-the-art OpenAI o3 model and generalizing across backbones. To our knowledge, this constitutes one of the first behavior-grounded agentic systems in pathology. Turning everyday viewer logs into scalable, expert-validated supervision, our framework makes agentic pathology practical and establishes a path to human-aligned, upgradeable clinical AI.
>
---
#### [new 164] OpenFLAME: Federated Visual Positioning System to Enable Large-Scale Augmented Reality Applications
- **分类: cs.CV; cs.DC; cs.RO**

- **简介: 论文提出OpenFLAME，一种联邦式视觉定位系统（VPS），用于支持大规模增强现实（AR）应用。任务是解决现有集中式VPS无法覆盖私有室内空间的问题。工作包括构建分布式VPS后端，实现数据分片管理、跨地图定位一致性与隐私保护。**

- **链接: [http://arxiv.org/pdf/2510.03915v1](http://arxiv.org/pdf/2510.03915v1)**

> **作者:** Sagar Bharadwaj; Harrison Williams; Luke Wang; Michael Liang; Tao Jin; Srinivasan Seshan; Anthony Rowe
>
> **摘要:** World-scale augmented reality (AR) applications need a ubiquitous 6DoF localization backend to anchor content to the real world consistently across devices. Large organizations such as Google and Niantic are 3D scanning outdoor public spaces in order to build their own Visual Positioning Systems (VPS). These centralized VPS solutions fail to meet the needs of many future AR applications -- they do not cover private indoor spaces because of privacy concerns, regulations, and the labor bottleneck of updating and maintaining 3D scans. In this paper, we present OpenFLAME, a federated VPS backend that allows independent organizations to 3D scan and maintain a separate VPS service for their own spaces. This enables access control of indoor 3D scans, distributed maintenance of the VPS backend, and encourages larger coverage. Sharding of VPS services introduces several unique challenges -- coherency of localization results across spaces, quality control of VPS services, selection of the right VPS service for a location, and many others. We introduce the concept of federated image-based localization and provide reference solutions for managing and merging data across maps without sharing private data.
>
---
#### [new 165] TAG:Tangential Amplifying Guidance for Hallucination-Resistant Diffusion Sampling
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型生成图像时出现的语义不一致和幻觉问题。作者提出了TAG方法，通过放大中间样本的切向分量来修正采样轨迹，提升生成质量，无需修改模型结构，计算效率高。**

- **链接: [http://arxiv.org/pdf/2510.04533v1](http://arxiv.org/pdf/2510.04533v1)**

> **作者:** Hyunmin Cho; Donghoon Ahn; Susung Hong; Jee Eun Kim; Seungryong Kim; Kyong Hwan Jin
>
> **备注:** 16 pages, 9 figures, 5 tables
>
> **摘要:** Recent diffusion models achieve the state-of-the-art performance in image generation, but often suffer from semantic inconsistencies or hallucinations. While various inference-time guidance methods can enhance generation, they often operate indirectly by relying on external signals or architectural modifications, which introduces additional computational overhead. In this paper, we propose Tangential Amplifying Guidance (TAG), a more efficient and direct guidance method that operates solely on trajectory signals without modifying the underlying diffusion model. TAG leverages an intermediate sample as a projection basis and amplifies the tangential components of the estimated scores with respect to this basis to correct the sampling trajectory. We formalize this guidance process by leveraging a first-order Taylor expansion, which demonstrates that amplifying the tangential component steers the state toward higher-probability regions, thereby reducing inconsistencies and enhancing sample quality. TAG is a plug-and-play, architecture-agnostic module that improves diffusion sampling fidelity with minimal computational addition, offering a new perspective on diffusion guidance.
>
---
#### [new 166] Contrastive-SDE: Guiding Stochastic Differential Equations with Contrastive Learning for Unpaired Image-to-Image Translation
- **分类: cs.CV**

- **简介: 该论文属于无配对图像到图像翻译任务，旨在解决无成对数据的情况下跨域图像转换问题。论文提出Contrastive-SDE方法，结合对比学习与随机微分方程模型，通过预训练对比模型引导图像生成，保留域不变特征，去除域特有特征。方法无需标签监督或分类器训练，收敛更快且效果与当前最优方法相当。**

- **链接: [http://arxiv.org/pdf/2510.03821v1](http://arxiv.org/pdf/2510.03821v1)**

> **作者:** Venkata Narendra Kotyada; Revanth Eranki; Nagesh Bhattu Sristy
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Unpaired image-to-image translation involves learning mappings between source domain and target domain in the absence of aligned or corresponding samples. Score based diffusion models have demonstrated state-of-the-art performance in generative tasks. Their ability to approximate complex data distributions through stochastic differential equations (SDEs) enables them to generate high-fidelity and diverse outputs, making them particularly well-suited for unpaired I2I settings. In parallel, contrastive learning provides a powerful framework for learning semantic similarities without the need for explicit supervision or paired data. By pulling together representations of semantically similar samples and pushing apart dissimilar ones, contrastive methods are inherently aligned with the objectives of unpaired translation. Its ability to selectively enforce semantic consistency at the feature level makes contrastive learning particularly effective for guiding generation in unpaired scenarios. In this work, we propose a time-dependent contrastive learning approach where a model is trained with SimCLR by considering an image and its domain invarient feature as a positive pair, enabling the preservation of domain-invariant features and the discarding of domain-specific ones. The learned contrastive model then guides the inference of a pretrained SDE for the I2I translation task. We empirically compare Contrastive-SDE with several baselines across three common unpaired I2I tasks, using four metrics for evaluation. Constrastive-SDE achieves comparable results to the state-of-the-art on several metrics. Furthermore, we observe that our model converges significantly faster and requires no label supervision or classifier training, making it a more efficient alternative for this task.
>
---
#### [new 167] PEaRL: Pathway-Enhanced Representation Learning for Gene and Pathway Expression Prediction from Histology
- **分类: cs.CV**

- **简介: 该论文属于多模态生物信息学任务，旨在通过整合组织病理学与空间转录组数据，提升基因及通路表达预测的准确性。现有方法依赖高变异基因，忽略了生物通路的作用。论文提出PEaRL框架，利用ssGSEA计算通路激活得分，并通过对比学习与组织特征对齐，增强了模型的生物学一致性与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.03455v1](http://arxiv.org/pdf/2510.03455v1)**

> **作者:** Sejuti Majumder; Saarthak Kapse; Moinak Bhattacharya; Xuan Xu; Alisa Yurovsky; Prateek Prasanna
>
> **摘要:** Integrating histopathology with spatial transcriptomics (ST) provides a powerful opportunity to link tissue morphology with molecular function. Yet most existing multimodal approaches rely on a small set of highly variable genes, which limits predictive scope and overlooks the coordinated biological programs that shape tissue phenotypes. We present PEaRL (Pathway Enhanced Representation Learning), a multimodal framework that represents transcriptomics through pathway activation scores computed with ssGSEA. By encoding biologically coherent pathway signals with a transformer and aligning them with histology features via contrastive learning, PEaRL reduces dimensionality, improves interpretability, and strengthens cross-modal correspondence. Across three cancer ST datasets (breast, skin, and lymph node), PEaRL consistently outperforms SOTA methods, yielding higher accuracy for both gene- and pathway-level expression prediction (up to 58.9 percent and 20.4 percent increase in Pearson correlation coefficient compared to SOTA). These results demonstrate that grounding transcriptomic representation in pathways produces more biologically faithful and interpretable multimodal models, advancing computational pathology beyond gene-level embeddings.
>
---
#### [new 168] MoME: Estimating Psychological Traits from Gait with Multi-Stage Mixture of Movement Experts
- **分类: cs.CV**

- **简介: 该论文属于心理特征估计任务，旨在通过步态分析解决从行走方式中推断心理特征的问题。论文提出了一种多阶段混合运动专家（MoME）架构，利用2D姿态表示的步态序列，结合多任务学习，提升了心理特征估计的性能，并在PsyMo基准数据集上取得了优异结果。**

- **链接: [http://arxiv.org/pdf/2510.04654v1](http://arxiv.org/pdf/2510.04654v1)**

> **作者:** Andy Cǎtrunǎ; Adrian Cosma; Emilian Rǎdoi
>
> **备注:** 4 Figures, 4 Tables
>
> **摘要:** Gait encodes rich biometric and behavioural information, yet leveraging the manner of walking to infer psychological traits remains a challenging and underexplored problem. We introduce a hierarchical Multi-Stage Mixture of Movement Experts (MoME) architecture for multi-task prediction of psychological attributes from gait sequences represented as 2D poses. MoME processes the walking cycle in four stages of movement complexity, employing lightweight expert models to extract spatio-temporal features and task-specific gating modules to adaptively weight experts across traits and stages. Evaluated on the PsyMo benchmark covering 17 psychological traits, our method outperforms state-of-the-art gait analysis models, achieving a 37.47% weighted F1 score at the run level and 44.6% at the subject level. Our experiments show that integrating auxiliary tasks such as identity recognition, gender prediction, and BMI estimation further improves psychological trait estimation. Our findings demonstrate the viability of multi-task gait-based learning for psychological trait estimation and provide a foundation for future research on movement-informed psychological inference.
>
---
#### [new 169] Harnessing Synthetic Preference Data for Enhancing Temporal Understanding of Video-LLMs
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决视频大语言模型（Video-LLMs）在细粒度时间理解上的不足。作者提出TimeWarp方法，构建合成偏好数据集，提升模型对视频时空信息的理解能力，显著改善其在多个时间理解基准上的表现。**

- **链接: [http://arxiv.org/pdf/2510.03955v1](http://arxiv.org/pdf/2510.03955v1)**

> **作者:** Sameep Vani; Shreyas Jena; Maitreya Patel; Chitta Baral; Somak Aditya; Yezhou Yang
>
> **备注:** 17 pages, 9 figures, 6 tables. Presents TimeWarp, a synthetic preference data framework to improve temporal understanding in Video-LLMs, showing consistent gains across seven benchmarks. Includes supplementary material in the Appendix
>
> **摘要:** While Video Large Language Models (Video-LLMs) have demonstrated remarkable performance across general video understanding benchmarks-particularly in video captioning and descriptive tasks-they consistently underperform on tasks that require fine-grained temporal understanding. This limitation arises due to the lack of visual complexity and temporal nuance in current fine-tuning datasets, leading these models to rely heavily on language-based reasoning rather than truly understanding video dynamics. In this work, we propose TimeWarp, a systematic method to create a targeted synthetic temporal dataset to fine-tune the model's responses to encourage it to focus on the given input video. We introduce a large-scale preference dataset, created using TimeWarp, that captures intricate temporal dynamics often overlooked, grounding the model's responses to visual and temporal information. We demonstrate that when our method is applied to existing models, it significantly improves performance on temporal understanding benchmarks, highlighting the effectiveness of our proposed datasets in advancing temporal understanding in Video-LLMs, resulting in an absolute improvement in performance across seven benchmarks. Code is available at https://github.com/sameepv21/timewarp.
>
---
#### [new 170] How We Won BraTS-SSA 2025: Brain Tumor Segmentation in the Sub-Saharan African Population Using Segmentation-Aware Data Augmentation and Model Ensembling
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决非洲人群中脑肿瘤分割效果差的问题。作者通过数据增强和模型集成（MedNeXt、SegMamba、Residual-Encoder U-Net）提升模型泛化能力，最终在 BraTS-Africa 数据集上取得良好效果。**

- **链接: [http://arxiv.org/pdf/2510.03568v1](http://arxiv.org/pdf/2510.03568v1)**

> **作者:** Claudia Takyi Ankomah; Livingstone Eli Ayivor; Ireneaus Nyame; Leslie Wambo; Patrick Yeboah Bonsu; Aondona Moses Iorumbur; Raymond Confidence; Toufiq Musah
>
> **备注:** Brain Tumor Segmentation Challenge, Medical Image Computing and Computer Assisted Intervention (MICCAI) Conference, 11 Pages, 2 Figures, 2 Tables
>
> **摘要:** Brain tumors, particularly gliomas, pose significant chall-enges due to their complex growth patterns, infiltrative nature, and the variability in brain structure across individuals, which makes accurate diagnosis and monitoring difficult. Deep learning models have been developed to accurately delineate these tumors. However, most of these models were trained on relatively homogenous high-resource datasets, limiting their robustness when deployed in underserved regions. In this study, we performed segmentation-aware offline data augmentation on the BraTS-Africa dataset to increase the data sample size and diversity to enhance generalization. We further constructed an ensemble of three distinct architectures, MedNeXt, SegMamba, and Residual-Encoder U-Net, to leverage their complementary strengths. Our best-performing model, MedNeXt, was trained on 1000 epochs and achieved the highest average lesion-wise dice and normalized surface distance scores of 0.86 and 0.81 respectively. However, the ensemble model trained for 500 epochs produced the most balanced segmentation performance across the tumour subregions. This work demonstrates that a combination of advanced augmentation and model ensembling can improve segmentation accuracy and robustness on diverse and underrepresented datasets. Code available at: https://github.com/SPARK-Academy-2025/SPARK-2025/tree/main/SPARK2025_BraTs_MODELS/SPARK_NeuroAshanti
>
---
#### [new 171] Super-resolution image projection over an extended depth of field using a diffractive decoder
- **分类: physics.optics; cs.CV; cs.NE; physics.app-ph**

- **简介: 该论文属于图像投影任务，旨在解决扩展景深与提高分辨率的问题。工作内容为设计了一种结合卷积神经网络编码器与全光衍射解码器的混合图像投影系统，实现超分辨率图像投影，提升空间带宽积，验证于太赫兹波段，并具备跨电磁波谱扩展应用的潜力。**

- **链接: [http://arxiv.org/pdf/2510.03938v1](http://arxiv.org/pdf/2510.03938v1)**

> **作者:** Hanlong Chen; Cagatay Isil; Tianyi Gan; Mona Jarrahi; Aydogan Ozcan
>
> **备注:** 18 Pages, 6 Figures
>
> **摘要:** Image projection systems must be efficient in data storage, computation and transmission while maintaining a large space-bandwidth-product (SBP) at their output. Here, we introduce a hybrid image projection system that achieves extended depth-of-field (DOF) with improved resolution, combining a convolutional neural network (CNN)-based digital encoder with an all-optical diffractive decoder. A CNN-based encoder compresses input images into compact phase representations, which are subsequently displayed by a low-resolution (LR) projector and processed by an analog diffractive decoder for all-optical image reconstruction. This optical decoder is completely passive, designed to synthesize pixel super-resolved image projections that feature an extended DOF while eliminating the need for additional power consumption for super-resolved image reconstruction. Our pixel super-resolution (PSR) image projection system demonstrates high-fidelity image synthesis over an extended DOF of ~267xW, where W is the illumination wavelength, concurrently offering up to ~16-fold SBP improvement at each lateral plane. The proof of concept of this approach is validated through an experiment conducted in the THz spectrum, and the system is scalable across different parts of the electromagnetic spectrum. This image projection architecture can reduce data storage and transmission requirements for display systems without imposing additional power constraints on the optical decoder. Beyond extended DOF PSR image projection, the underlying principles of this approach can be extended to various applications, including optical metrology and microscopy.
>
---
#### [new 172] SONA: Learning Conditional, Unconditional, and Mismatching-Aware Discriminator
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文属于生成对抗网络任务，旨在解决条件生成中鉴别器难以平衡真实性和条件对齐的问题。作者提出SONA模型，通过自然性和对齐性的分离投影、匹配感知监督和自适应加权机制，提升了生成质量和条件一致性。**

- **链接: [http://arxiv.org/pdf/2510.04576v1](http://arxiv.org/pdf/2510.04576v1)**

> **作者:** Yuhta Takida; Satoshi Hayakawa; Takashi Shibuya; Masaaki Imaizumi; Naoki Murata; Bac Nguyen; Toshimitsu Uesaka; Chieh-Hsin Lai; Yuki Mitsufuji
>
> **备注:** 24 pages with 9 figures
>
> **摘要:** Deep generative models have made significant advances in generating complex content, yet conditional generation remains a fundamental challenge. Existing conditional generative adversarial networks often struggle to balance the dual objectives of assessing authenticity and conditional alignment of input samples within their conditional discriminators. To address this, we propose a novel discriminator design that integrates three key capabilities: unconditional discrimination, matching-aware supervision to enhance alignment sensitivity, and adaptive weighting to dynamically balance all objectives. Specifically, we introduce Sum of Naturalness and Alignment (SONA), which employs separate projections for naturalness (authenticity) and alignment in the final layer with an inductive bias, supported by dedicated objective functions and an adaptive weighting mechanism. Extensive experiments on class-conditional generation tasks show that \ours achieves superior sample quality and conditional alignment compared to state-of-the-art methods. Furthermore, we demonstrate its effectiveness in text-to-image generation, confirming the versatility and robustness of our approach.
>
---
#### [new 173] Diverse Text-to-Image Generation via Contrastive Noise Optimization
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于文本生成图像任务，旨在解决文本引导生成图像多样性不足的问题。通过提出对比噪声优化方法，在初始噪声中引入对比损失，优化噪声潜变量，从而在保持图像质量的同时提升生成多样性。**

- **链接: [http://arxiv.org/pdf/2510.03813v1](http://arxiv.org/pdf/2510.03813v1)**

> **作者:** Byungjun Kim; Soobin Um; Jong Chul Ye
>
> **摘要:** Text-to-image (T2I) diffusion models have demonstrated impressive performance in generating high-fidelity images, largely enabled by text-guided inference. However, this advantage often comes with a critical drawback: limited diversity, as outputs tend to collapse into similar modes under strong text guidance. Existing approaches typically optimize intermediate latents or text conditions during inference, but these methods deliver only modest gains or remain sensitive to hyperparameter tuning. In this work, we introduce Contrastive Noise Optimization, a simple yet effective method that addresses the diversity issue from a distinct perspective. Unlike prior techniques that adapt intermediate latents, our approach shapes the initial noise to promote diverse outputs. Specifically, we develop a contrastive loss defined in the Tweedie data space and optimize a batch of noise latents. Our contrastive optimization repels instances within the batch to maximize diversity while keeping them anchored to a reference sample to preserve fidelity. We further provide theoretical insights into the mechanism of this preprocessing to substantiate its effectiveness. Extensive experiments across multiple T2I backbones demonstrate that our approach achieves a superior quality-diversity Pareto frontier while remaining robust to hyperparameter choices.
>
---
#### [new 174] CLEAR-IR: Clarity-Enhanced Active Reconstruction of Infrared Imagery
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于图像增强与机器人视觉任务，旨在解决红外图像中因主动发射器模式干扰影响高层视觉任务的问题。作者提出了一种基于U-Net的架构CLEAR-IR，用于从受干扰的红外图像中重建清晰图像，提升图像质量和机器人在低光环境下的感知性能。**

- **链接: [http://arxiv.org/pdf/2510.04883v1](http://arxiv.org/pdf/2510.04883v1)**

> **作者:** Nathan Shankar; Pawel Ladosz; Hujun Yin
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** This paper presents a novel approach for enabling robust robotic perception in dark environments using infrared (IR) stream. IR stream is less susceptible to noise than RGB in low-light conditions. However, it is dominated by active emitter patterns that hinder high-level tasks such as object detection, tracking and localisation. To address this, a U-Net-based architecture is proposed that reconstructs clean IR images from emitter-populated input, improving both image quality and downstream robotic performance. This approach outperforms existing enhancement techniques and enables reliable operation of vision-driven robotic systems across illumination conditions from well-lit to extreme low-light scenes.
>
---
#### [new 175] VIFO: Visual Feature Empowered Multivariate Time Series Forecasting with Cross-Modal Fusion
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多变量时间序列预测任务，旨在解决现有模型忽略跨通道依赖和未充分利用视觉模型的问题。作者提出VIFO，将时间序列转化为图像，利用预训练大视觉模型提取复杂模式，并与时间序列模态融合，仅微调少量参数，提升预测性能。**

- **链接: [http://arxiv.org/pdf/2510.03244v1](http://arxiv.org/pdf/2510.03244v1)**

> **作者:** Yanlong Wang; Hang Yu; Jian Xu; Fei Ma; Hongkang Zhang; Tongtong Feng; Zijian Zhang; Shao-Lun Huang; Danny Dongning Sun; Xiao-Ping Zhang
>
> **摘要:** Large time series foundation models often adopt channel-independent architectures to handle varying data dimensions, but this design ignores crucial cross-channel dependencies. Concurrently, existing multimodal approaches have not fully exploited the power of large vision models (LVMs) to interpret spatiotemporal data. Additionally, there remains significant unexplored potential in leveraging the advantages of information extraction from different modalities to enhance time series forecasting performance. To address these gaps, we propose the VIFO, a cross-modal forecasting model. VIFO uniquely renders multivariate time series into image, enabling pre-trained LVM to extract complex cross-channel patterns that are invisible to channel-independent models. These visual features are then aligned and fused with representations from the time series modality. By freezing the LVM and training only 7.45% of its parameters, VIFO achieves competitive performance on multiple benchmarks, offering an efficient and effective solution for capturing cross-variable relationships in
>
---
#### [new 176] Bridging Text and Video Generation: A Survey
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于文本到视频生成任务，旨在解决生成高质量、时间连贯的视频内容问题。论文综述了从早期GANs、VAEs到最新DiT架构的发展，分析了模型演进原因，并系统总结了数据集、训练配置、评估指标及当前挑战，提出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2510.04999v1](http://arxiv.org/pdf/2510.04999v1)**

> **作者:** Nilay Kumar; Priyansh Bhandari; G. Maragatham
>
> **摘要:** Text-to-video (T2V) generation technology holds potential to transform multiple domains such as education, marketing, entertainment, and assistive technologies for individuals with visual or reading comprehension challenges, by creating coherent visual content from natural language prompts. From its inception, the field has advanced from adversarial models to diffusion-based models, yielding higher-fidelity, temporally consistent outputs. Yet challenges persist, such as alignment, long-range coherence, and computational efficiency. Addressing this evolving landscape, we present a comprehensive survey of text-to-video generative models, tracing their development from early GANs and VAEs to hybrid Diffusion-Transformer (DiT) architectures, detailing how these models work, what limitations they addressed in their predecessors, and why shifts toward new architectural paradigms were necessary to overcome challenges in quality, coherence, and control. We provide a systematic account of the datasets, which the surveyed text-to-video models were trained and evaluated on, and, to support reproducibility and assess the accessibility of training such models, we detail their training configurations, including their hardware specifications, GPU counts, batch sizes, learning rates, optimizers, epochs, and other key hyperparameters. Further, we outline the evaluation metrics commonly used for evaluating such models and present their performance across standard benchmarks, while also discussing the limitations of these metrics and the emerging shift toward more holistic, perception-aligned evaluation strategies. Finally, drawing from our analysis, we outline the current open challenges and propose a few promising future directions, laying out a perspective for future researchers to explore and build upon in advancing T2V research and applications.
>
---
#### [new 177] Social Agent: Mastering Dyadic Nonverbal Behavior Generation via Conversational LLM Agents
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于人机交互与虚拟对话系统任务，旨在解决对话中非语言行为生成不自然、缺乏协调的问题。作者提出Social Agent框架，利用大语言模型控制对话流程，并通过自回归扩散模型生成协调的肢体动作，实现更自然的双人对话互动效果。**

- **链接: [http://arxiv.org/pdf/2510.04637v1](http://arxiv.org/pdf/2510.04637v1)**

> **作者:** Zeyi Zhang; Yanju Zhou; Heyuan Yao; Tenglong Ao; Xiaohang Zhan; Libin Liu
>
> **备注:** SIGGRAPH ASIA 2025 (Conference Track); Project page: https://pku-mocca.github.io/Social-Agent-Page/
>
> **摘要:** We present Social Agent, a novel framework for synthesizing realistic and contextually appropriate co-speech nonverbal behaviors in dyadic conversations. In this framework, we develop an agentic system driven by a Large Language Model (LLM) to direct the conversation flow and determine appropriate interactive behaviors for both participants. Additionally, we propose a novel dual-person gesture generation model based on an auto-regressive diffusion model, which synthesizes coordinated motions from speech signals. The output of the agentic system is translated into high-level guidance for the gesture generator, resulting in realistic movement at both the behavioral and motion levels. Furthermore, the agentic system periodically examines the movements of interlocutors and infers their intentions, forming a continuous feedback loop that enables dynamic and responsive interactions between the two participants. User studies and quantitative evaluations show that our model significantly improves the quality of dyadic interactions, producing natural, synchronized nonverbal behaviors.
>
---
#### [new 178] C3Editor: Achieving Controllable Consistency in 2D Model for 3D Editing
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D编辑任务，旨在解决现有2D编辑提升至3D时存在的视角不一致问题。作者提出C3Editor框架，通过选择参考视角并微调2D编辑模型，引入LoRA模块实现多视角一致性和精准编辑，提升了3D编辑的可控性与一致性。**

- **链接: [http://arxiv.org/pdf/2510.04539v1](http://arxiv.org/pdf/2510.04539v1)**

> **作者:** Zeng Tao; Zheng Ding; Zeyuan Chen; Xiang Zhang; Leizhi Li; Zhuowen Tu
>
> **摘要:** Existing 2D-lifting-based 3D editing methods often encounter challenges related to inconsistency, stemming from the lack of view-consistent 2D editing models and the difficulty of ensuring consistent editing across multiple views. To address these issues, we propose C3Editor, a controllable and consistent 2D-lifting-based 3D editing framework. Given an original 3D representation and a text-based editing prompt, our method selectively establishes a view-consistent 2D editing model to achieve superior 3D editing results. The process begins with the controlled selection of a ground truth (GT) view and its corresponding edited image as the optimization target, allowing for user-defined manual edits. Next, we fine-tune the 2D editing model within the GT view and across multiple views to align with the GT-edited image while ensuring multi-view consistency. To meet the distinct requirements of GT view fitting and multi-view consistency, we introduce separate LoRA modules for targeted fine-tuning. Our approach delivers more consistent and controllable 2D and 3D editing results than existing 2D-lifting-based methods, outperforming them in both qualitative and quantitative evaluations.
>
---
#### [new 179] Bridging the Gap Between Multimodal Foundation Models and World Models
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文旨在弥合多模态基础模型（MFMs）与世界模型之间的差距。当前MFMs在反事实推理、动态模拟、时空理解等方面存在不足。论文通过提升MFMs的结构化推理能力，并引入可控生成框架，增强其对多模态数据的深度理解和生成能力，从而实现更高效的世界建模。**

- **链接: [http://arxiv.org/pdf/2510.03727v1](http://arxiv.org/pdf/2510.03727v1)**

> **作者:** Xuehai He
>
> **备注:** PhD thesis
>
> **摘要:** Humans understand the world through the integration of multiple sensory modalities, enabling them to perceive, reason about, and imagine dynamic physical processes. Inspired by this capability, multimodal foundation models (MFMs) have emerged as powerful tools for multimodal understanding and generation. However, today's MFMs fall short of serving as effective world models. They lack the essential ability such as perform counterfactual reasoning, simulate dynamics, understand the spatiotemporal information, control generated visual outcomes, and perform multifaceted reasoning. We investigates what it takes to bridge the gap between multimodal foundation models and world models. We begin by improving the reasoning capabilities of MFMs through discriminative tasks and equipping MFMs with structured reasoning skills, such as causal inference, counterfactual thinking, and spatiotemporal reasoning, enabling them to go beyond surface correlations and understand deeper relationships within visual and textual data. Next, we explore generative capabilities of multimodal foundation models across both image and video modalities, introducing new frameworks for structured and controllable generation. Our approaches incorporate scene graphs, multimodal conditioning, and multimodal alignment strategies to guide the generation process, ensuring consistency with high-level semantics and fine-grained user intent. We further extend these techniques to controllable 4D generation, enabling interactive, editable, and morphable object synthesis over time and space.
>
---
#### [new 180] EmbodiSwap for Zero-Shot Robot Imitation Learning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出EmbodiSwap方法，用于零样本机器人模仿学习。通过将人类视频合成为机器人视角视频，解决人类与机器人形态差异问题。利用V-JEPA作为视觉主干训练操作策略，在真实场景中取得82%成功率，优于现有方法。同时开源代码、数据集与模型，促进可重复研究。**

- **链接: [http://arxiv.org/pdf/2510.03706v1](http://arxiv.org/pdf/2510.03706v1)**

> **作者:** Eadom Dessalene; Pavan Mantripragada; Michael Maynord; Yiannis Aloimonos
>
> **备注:** Video link: https://drive.google.com/file/d/1UccngwgPqUwPMhBja7JrXfZoTquCx_Qe/view?usp=sharing
>
> **摘要:** We introduce EmbodiSwap - a method for producing photorealistic synthetic robot overlays over human video. We employ EmbodiSwap for zero-shot imitation learning, bridging the embodiment gap between in-the-wild ego-centric human video and a target robot embodiment. We train a closed-loop robot manipulation policy over the data produced by EmbodiSwap. We make novel use of V-JEPA as a visual backbone, repurposing V-JEPA from the domain of video understanding to imitation learning over synthetic robot videos. Adoption of V-JEPA outperforms alternative vision backbones more conventionally used within robotics. In real-world tests, our zero-shot trained V-JEPA model achieves an $82\%$ success rate, outperforming a few-shot trained $\pi_0$ network as well as $\pi_0$ trained over data produced by EmbodiSwap. We release (i) code for generating the synthetic robot overlays which takes as input human videos and an arbitrary robot URDF and generates a robot dataset, (ii) the robot dataset we synthesize over EPIC-Kitchens, HOI4D and Ego4D, and (iii) model checkpoints and inference code, to facilitate reproducible research and broader adoption.
>
---
#### [new 181] Revoking Amnesia: RL-based Trajectory Optimization to Resurrect Erased Concepts in Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型中已删除概念的恢复问题。现有擦除方法效果不佳，作者提出RevAm框架，利用强化学习优化轨迹，动态引导去噪过程，实现高效概念复活，揭示当前安全机制漏洞。**

- **链接: [http://arxiv.org/pdf/2510.03302v1](http://arxiv.org/pdf/2510.03302v1)**

> **作者:** Daiheng Gao; Nanxiang Jiang; Andi Zhang; Shilin Lu; Yufei Tang; Wenbo Zhou; Weiming Zhang; Zhaoxin Fan
>
> **备注:** 21 pages, 10 figures
>
> **摘要:** Concept erasure techniques have been widely deployed in T2I diffusion models to prevent inappropriate content generation for safety and copyright considerations. However, as models evolve to next-generation architectures like Flux, established erasure methods (\textit{e.g.}, ESD, UCE, AC) exhibit degraded effectiveness, raising questions about their true mechanisms. Through systematic analysis, we reveal that concept erasure creates only an illusion of ``amnesia": rather than genuine forgetting, these methods bias sampling trajectories away from target concepts, making the erasure fundamentally reversible. This insight motivates the need to distinguish superficial safety from genuine concept removal. In this work, we propose \textbf{RevAm} (\underline{Rev}oking \underline{Am}nesia), an RL-based trajectory optimization framework that resurrects erased concepts by dynamically steering the denoising process without modifying model weights. By adapting Group Relative Policy Optimization (GRPO) to diffusion models, RevAm explores diverse recovery trajectories through trajectory-level rewards, overcoming local optima that limit existing methods. Extensive experiments demonstrate that RevAm achieves superior concept resurrection fidelity while reducing computational time by 10$\times$, exposing critical vulnerabilities in current safety mechanisms and underscoring the need for more robust erasure techniques beyond trajectory manipulation.
>
---
#### [new 182] SAEdit: Token-level control for continuous image editing via Sparse AutoEncoder
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决文本驱动编辑中控制不够精细的问题。现有方法难以实现属性解耦和连续调整。论文提出SAEdit，通过稀疏自编码器挖掘文本嵌入中语义独立的方向，实现对图像编辑属性的连续、解耦控制，无需修改扩散模型本身，具有通用性。**

- **链接: [http://arxiv.org/pdf/2510.05081v1](http://arxiv.org/pdf/2510.05081v1)**

> **作者:** Ronen Kamenetsky; Sara Dorfman; Daniel Garibi; Roni Paiss; Or Patashnik; Daniel Cohen-Or
>
> **备注:** Project page at: https://ronen94.github.io/SAEdit/
>
> **摘要:** Large-scale text-to-image diffusion models have become the backbone of modern image editing, yet text prompts alone do not offer adequate control over the editing process. Two properties are especially desirable: disentanglement, where changing one attribute does not unintentionally alter others, and continuous control, where the strength of an edit can be smoothly adjusted. We introduce a method for disentangled and continuous editing through token-level manipulation of text embeddings. The edits are applied by manipulating the embeddings along carefully chosen directions, which control the strength of the target attribute. To identify such directions, we employ a Sparse Autoencoder (SAE), whose sparse latent space exposes semantically isolated dimensions. Our method operates directly on text embeddings without modifying the diffusion process, making it model agnostic and broadly applicable to various image synthesis backbones. Experiments show that it enables intuitive and efficient manipulations with continuous control across diverse attributes and domains.
>
---
#### [new 183] On Structured State-Space Duality
- **分类: cs.LG; cs.CL; cs.CV; stat.ML**

- **简介: 本文研究结构化状态空间模型（SSM）与注意力机制的等价性，旨在建立二者间的理论联系并提升序列建模效率。论文扩展了原有SSD理论，证明对角SSM满足等价条件且保持训练复杂度下界，同时指出标准softmax注意力因秩爆炸无法纳入该框架。工作包括理论推导、条件证明与模型分析，为高效序列模型设计提供新视角。**

- **链接: [http://arxiv.org/pdf/2510.04944v1](http://arxiv.org/pdf/2510.04944v1)**

> **作者:** Jerry Yao-Chieh Hu; Xiwen Zhang; Weimin Wu; Han Liu
>
> **摘要:** Structured State-Space Duality (SSD) [Dao & Gu, ICML 2024] is an equivalence between a simple Structured State-Space Model (SSM) and a masked attention mechanism. In particular, a state-space model with a scalar-times-identity state matrix is equivalent to a masked self-attention with a $1$-semiseparable causal mask. Consequently, the same sequence transformation (model) has two algorithmic realizations: as a linear-time $O(T)$ recurrence or as a quadratic-time $O(T^2)$ attention. In this note, we formalize and generalize this duality: (i) we extend SSD from the scalar-identity case to general diagonal SSMs (diagonal state matrices); (ii) we show that these diagonal SSMs match the scalar case's training complexity lower bounds while supporting richer dynamics; (iii) we establish a necessary and sufficient condition under which an SSM is equivalent to $1$-semiseparable masked attention; and (iv) we show that such duality fails to extend to standard softmax attention due to rank explosion. Together, these results tighten bridge between recurrent SSMs and Transformers, and widen the design space for expressive yet efficient sequence models.
>
---
#### [new 184] AI-Assisted Pleural Effusion Volume Estimation from Contrast-Enhanced CT Images
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决胸腔积液（PE）在CT影像中难以准确分割和量化的问题。研究团队开发了一种半监督深度学习框架TTAS，利用少量标注数据训练模型，在未完全标注的CT数据上实现了高效分割。结果表明，TTAS在分割性能和体积测量精度上均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.03856v1](http://arxiv.org/pdf/2510.03856v1)**

> **作者:** Sanhita Basu; Tomas Fröding; Ali Teymur Kahraman; Dimitris Toumpanakis; Tobias Sjöblom
>
> **摘要:** Background: Pleural Effusions (PE) is a common finding in many different clinical conditions, but accurately measuring their volume from CT scans is challenging. Purpose: To improve PE segmentation and quantification for enhanced clinical management, we have developed and trained a semi-supervised deep learning framework on contrast-enhanced CT volumes. Materials and Methods: This retrospective study collected CT Pulmonary Angiogram (CTPA) data from internal and external datasets. A subset of 100 cases was manually annotated for model training, while the remaining cases were used for testing and validation. A novel semi-supervised deep learning framework, Teacher-Teaching Assistant-Student (TTAS), was developed and used to enable efficient training in non-segmented examinations. Segmentation performance was compared to that of state-of-the-art models. Results: 100 patients (mean age, 72 years, 28 [standard deviation]; 55 men) were included in the study. The TTAS model demonstrated superior segmentation performance compared to state-of-the-art models, achieving a mean Dice score of 0.82 (95% CI, 0.79 - 0.84) versus 0.73 for nnU-Net (p < 0.0001, Student's T test). Additionally, TTAS exhibited a four-fold lower mean Absolute Volume Difference (AbVD) of 6.49 mL (95% CI, 4.80 - 8.20) compared to nnU-Net's AbVD of 23.16 mL (p < 0.0001). Conclusion: The developed TTAS framework offered superior PE segmentation, aiding accurate volume determination from CT scans.
>
---
#### [new 185] Conditional Pseudo-Supervised Contrast for Data-Free Knowledge Distillation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于数据无关知识蒸馏任务，旨在解决无真实数据情况下模型压缩中学生模型学习效果差的问题。通过提出条件伪监督对比学习方法（CPSC-DFKD），利用条件生成对抗网络合成类别多样样本，并引入对比学习提升学生模型性能。**

- **链接: [http://arxiv.org/pdf/2510.03375v1](http://arxiv.org/pdf/2510.03375v1)**

> **作者:** Renrong Shao; Wei Zhang; Jun wang
>
> **备注:** 13 pages
>
> **摘要:** Data-free knowledge distillation~(DFKD) is an effective manner to solve model compression and transmission restrictions while retaining privacy protection, which has attracted extensive attention in recent years. Currently, the majority of existing methods utilize a generator to synthesize images to support the distillation. Although the current methods have achieved great success, there are still many issues to be explored. Firstly, the outstanding performance of supervised learning in deep learning drives us to explore a pseudo-supervised paradigm on DFKD. Secondly, current synthesized methods cannot distinguish the distributions of different categories of samples, thus producing ambiguous samples that may lead to an incorrect evaluation by the teacher. Besides, current methods cannot optimize the category-wise diversity samples, which will hinder the student model learning from diverse samples and further achieving better performance. In this paper, to address the above limitations, we propose a novel learning paradigm, i.e., conditional pseudo-supervised contrast for data-free knowledge distillation~(CPSC-DFKD). The primary innovations of CPSC-DFKD are: (1) introducing a conditional generative adversarial network to synthesize category-specific diverse images for pseudo-supervised learning, (2) improving the modules of the generator to distinguish the distributions of different categories, and (3) proposing pseudo-supervised contrastive learning based on teacher and student views to enhance diversity. Comprehensive experiments on three commonly-used datasets validate the performance lift of both the student and generator brought by CPSC-DFKD. The code is available at https://github.com/RoryShao/CPSC-DFKD.git
>
---
#### [new 186] Use of Quadcopter Wakes to Supplement Strawberry Pollination
- **分类: eess.SY; cs.CV; cs.SY**

- **简介: 该论文旨在解决草莓授粉不足的问题，探索使用四旋翼无人机（quadcopter）辅助自然授粉的可行性。研究通过分析风力传播花粉的效果，进行了田间和实验室实验，虽田间结果不明确，但实验室结果具潜力，属农业授粉技术任务。**

- **链接: [http://arxiv.org/pdf/2510.03974v1](http://arxiv.org/pdf/2510.03974v1)**

> **作者:** Sadie Cutler; Ben DeFay; Scott McArt; Kirstin Petersen
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** Pollinators are critical to the world's ecosystems and food supply, yet recent studies have found pollination shortfalls in several crops, including strawberry. This is troubling because wild and managed pollinators are currently experiencing declines. One possibility is to try and provide supplemental pollination solutions. These solutions should be affordable and simple for farmers to implement if their use is to be widespread; quadcopters are a great example, already used for monitoring on many farms. This paper investigates a new method for artificial pollination based on wind pollination that bears further investigation. After determining the height where the lateral flow is maximized, we performed field experiments with a quadcopter assisting natural pollinators. Although our results in the field were inconclusive, lab studies show that the idea shows promise and could be adapted for better field results.
>
---
#### [new 187] DoRAN: Stabilizing Weight-Decomposed Low-Rank Adaptation via Noise Injection and Auxiliary Networks
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于参数高效微调任务，旨在解决大模型微调中的训练不稳定和样本效率低问题。论文提出DoRAN方法，通过在DoRA基础上引入噪声注入和辅助网络，增强训练稳定性并提升样本效率。实验表明，DoRAN在多个视觉和语言任务上优于LoRA、DoRA等基线方法。**

- **链接: [http://arxiv.org/pdf/2510.04331v1](http://arxiv.org/pdf/2510.04331v1)**

> **作者:** Nghiem T. Diep; Hien Dang; Tuan Truong; Tan Dinh; Huy Nguyen; Nhat Ho
>
> **备注:** Nghiem T. Diep, Hien Dang, and Tuan Truong contributed equally to this work
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) methods have become the standard paradigm for adapting large-scale models. Among these techniques, Weight-Decomposed Low-Rank Adaptation (DoRA) has been shown to improve both the learning capacity and training stability of the vanilla Low-Rank Adaptation (LoRA) method by explicitly decomposing pre-trained weights into magnitude and directional components. In this work, we propose DoRAN, a new variant of DoRA designed to further stabilize training and boost the sample efficiency of DoRA. Our approach includes two key stages: (i) injecting noise into the denominator of DoRA's weight decomposition, which serves as an adaptive regularizer to mitigate instabilities; and (ii) replacing static low-rank matrices with auxiliary networks that generate them dynamically, enabling parameter coupling across layers and yielding better sample efficiency in both theory and practice. Comprehensive experiments on vision and language benchmarks show that DoRAN consistently outperforms LoRA, DoRA, and other PEFT baselines. These results underscore the effectiveness of combining stabilization through noise-based regularization with network-based parameter generation, offering a promising direction for robust and efficient fine-tuning of foundation models.
>
---
#### [new 188] Adaptive double-phase Rudin--Osher--Fatemi denoising model
- **分类: eess.IV; cs.CV; cs.NA; math.NA**

- **简介: 该论文属于图像去噪任务，旨在解决传统Rudin–Osher–Fatemi模型导致的阶梯效应问题。作者提出了一种自适应双相变分正则化去噪模型，在保留图像边缘的同时有效减少阶梯效应，并在多种图像上进行了测试验证。**

- **链接: [http://arxiv.org/pdf/2510.04382v1](http://arxiv.org/pdf/2510.04382v1)**

> **作者:** Wojciech Górny; Michał Łasica; Alexandros Matsoukas
>
> **备注:** 21 pages, 18 figures, supplementary material available at: https://github.com/wojciechgorny/double-phase-ROF-model/
>
> **摘要:** We propose a new image denoising model based on a variable-growth total variation regularization of double-phase type with adaptive weight. It is designed to reduce staircasing with respect to the classical Rudin--Osher--Fatemi model, while preserving the edges of the image in a similar fashion. We implement the model and test its performance on synthetic and natural images in 1D and 2D over a range of noise levels.
>
---
#### [new 189] Towards Robust and Generalizable Continuous Space-Time Video Super-Resolution with Events
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文属于视频超分辨率任务，旨在解决现有方法在任意时空尺度下泛化能力差的问题。作者提出了EvEnhancer及EvEnhancerPlus，结合事件流的高时间分辨率与动态范围，利用事件自适应合成与局部隐式视频变换，实现鲁棒且可泛化的连续时空视频重建，并通过可控切换机制提升效率。**

- **链接: [http://arxiv.org/pdf/2510.03833v1](http://arxiv.org/pdf/2510.03833v1)**

> **作者:** Shuoyan Wei; Feng Li; Shengeng Tang; Runmin Cong; Yao Zhao; Meng Wang; Huihui Bai
>
> **备注:** 17 pages, 12 figures, 14 tables. Under review
>
> **摘要:** Continuous space-time video super-resolution (C-STVSR) has garnered increasing interest for its capability to reconstruct high-resolution and high-frame-rate videos at arbitrary spatial and temporal scales. However, prevailing methods often generalize poorly, producing unsatisfactory results when applied to out-of-distribution (OOD) scales. To overcome this limitation, we present EvEnhancer, a novel approach that marries the unique properties of high temporal resolution and high dynamic range encapsulated in event streams to achieve robust and generalizable C-STVSR. Our approach incorporates event-adapted synthesis that capitalizes on the spatiotemporal correlations between frames and events to capture long-term motion trajectories, enabling adaptive interpolation and fusion across space and time. This is then coupled with a local implicit video transformer that integrates local implicit video neural function with cross-scale spatiotemporal attention to learn continuous video representations and generate plausible videos at arbitrary resolutions and frame rates. We further develop EvEnhancerPlus, which builds a controllable switching mechanism that dynamically determines the reconstruction difficulty for each spatiotemporal pixel based on local event statistics. This allows the model to adaptively route reconstruction along the most suitable pathways at a fine-grained pixel level, substantially reducing computational overhead while maintaining excellent performance. Furthermore, we devise a cross-derivative training strategy that stabilizes the convergence of such a multi-pathway framework through staged cross-optimization. Extensive experiments demonstrate that our method achieves state-of-the-art performance on both synthetic and real-world datasets, while maintaining superior generalizability at OOD scales. The code is available at https://github.com/W-Shuoyan/EvEnhancerPlus.
>
---
#### [new 190] Universal Beta Splatting
- **分类: cs.GR; cs.CV; eess.IV**

- **简介: 该论文提出Universal Beta Splatting（UBS），用于辐射场渲染任务，旨在统一建模空间、角度与时间维度的复杂光照效果。它用Beta核替代传统高斯核，实现各向异性、可解释的场景表示，无需辅助网络即可处理动态与视角依赖外观。论文实现了实时渲染，性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.03312v1](http://arxiv.org/pdf/2510.03312v1)**

> **作者:** Rong Liu; Zhongpai Gao; Benjamin Planche; Meida Chen; Van Nguyen Nguyen; Meng Zheng; Anwesa Choudhuri; Terrence Chen; Yue Wang; Andrew Feng; Ziyan Wu
>
> **摘要:** We introduce Universal Beta Splatting (UBS), a unified framework that generalizes 3D Gaussian Splatting to N-dimensional anisotropic Beta kernels for explicit radiance field rendering. Unlike fixed Gaussian primitives, Beta kernels enable controllable dependency modeling across spatial, angular, and temporal dimensions within a single representation. Our unified approach captures complex light transport effects, handles anisotropic view-dependent appearance, and models scene dynamics without requiring auxiliary networks or specific color encodings. UBS maintains backward compatibility by approximating to Gaussian Splatting as a special case, guaranteeing plug-in usability and lower performance bounds. The learned Beta parameters naturally decompose scene properties into interpretable without explicit supervision: spatial (surface vs. texture), angular (diffuse vs. specular), and temporal (static vs. dynamic). Our CUDA-accelerated implementation achieves real-time rendering while consistently outperforming existing methods across static, view-dependent, and dynamic benchmarks, establishing Beta kernels as a scalable universal primitive for radiance field rendering. Our project website is available at https://rongliu-leo.github.io/universal-beta-splatting/.
>
---
#### [new 191] SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 论文提出SDQ-LLM，一种用于任意规模大语言模型的1-bit量化方法，旨在解决低比特量化中精度损失问题。通过Sigma-Delta量化结合过采样比（OSR）动态调整、Hadamard权重平滑及多粒度OSR分配策略MultiOSR，实现高效推理并保持语言模型性能。**

- **链接: [http://arxiv.org/pdf/2510.03275v1](http://arxiv.org/pdf/2510.03275v1)**

> **作者:** Junhao Xia; Ming Zhao; Limin Xiao; Xiujun Zhang
>
> **摘要:** Large language models (LLMs) face significant computational and memory challenges, making extremely low-bit quantization crucial for their efficient deployment. In this work, we introduce SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size, a novel framework that enables extremely low-bit quantization of LLMs while preserving their linguistic reasoning capabilities. A distinctive feature of SDQ-LLM is the continuous adjustability of the Over-Sampling Ratio (OSR), enabling dynamic adaptation to memory or VRAM constraints by selecting fractional OSR (e.g. 2.5 times) for an optimal trade-off between model size and accuracy. SDQ-LLM uses upsampling combined with Sigma-Delta Quantizer to binarize or ternarize LLMs weights, encoding high-precision parameters into 1-bit or 1.58-bit representations, replacing the multiplication operations within linear layers with addition. This approach significantly enhances inference efficiency under extremely low-bit quantization. To further reduce the loss of quantization precision, we incorporate Hadamard-based weight smoothing prior to quantization, improving the stability and robustness of the weight representations. Furthermore, to fully leverage the continuity of the OSR and reduce precision loss, recognizing the correlation between quantization sensitivity and weight variance, we propose a fine-grained, layer- and linear-wise OSR allocation strategy, MultiOSR. This strategy distributes OSR both across layers and within each layer, based on weight variance and parameter scale. Finally, extensive experiments on OPT and LLaMA model families demonstrate that SDQ-LLM achieves a more efficient and high-precision performance even under highly aggressive low-OSR settings. Our code is available at https://github.com/Dreamlittlecat/LLM-Quant-Factory.
>
---
#### [new 192] Universal Multi-Domain Translation via Diffusion Routers
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多领域翻译任务，旨在解决现有方法需完全对齐数据或仅支持训练中见过的领域对的问题。论文提出通用多领域翻译（UMDT）和基于扩散模型的框架Diffusion Router（DR），通过中心领域路由实现任意领域间翻译，并引入可扩展学习策略，实现高效间接与直接翻译，在多个大规模基准上表现优异。**

- **链接: [http://arxiv.org/pdf/2510.03252v1](http://arxiv.org/pdf/2510.03252v1)**

> **作者:** Duc Kieu; Kien Do; Tuan Hoang; Thao Minh Le; Tung Kieu; Dang Nguyen; Thin Nguyen
>
> **摘要:** Multi-domain translation (MDT) aims to learn translations between multiple domains, yet existing approaches either require fully aligned tuples or can only handle domain pairs seen in training, limiting their practicality and excluding many cross-domain mappings. We introduce universal MDT (UMDT), a generalization of MDT that seeks to translate between any pair of $K$ domains using only $K-1$ paired datasets with a central domain. To tackle this problem, we propose Diffusion Router (DR), a unified diffusion-based framework that models all central$\leftrightarrow$non-central translations with a single noise predictor conditioned on the source and target domain labels. DR enables indirect non-central translations by routing through the central domain. We further introduce a novel scalable learning strategy with a variational-bound objective and an efficient Tweedie refinement procedure to support direct non-central mappings. Through evaluation on three large-scale UMDT benchmarks, DR achieves state-of-the-art results for both indirect and direct translations, while lowering sampling cost and unlocking novel tasks such as sketch$\leftrightarrow$segmentation. These results establish DR as a scalable and versatile framework for universal translation across multiple domains.
>
---
#### [new 193] Rethinking Inter-LoRA Orthogonality in Adapter Merging: Insights from Orthogonal Monte Carlo Dropout
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于模型适配与融合任务，旨在解决多LoRA模块合并时语义干扰的问题。作者提出正交蒙特卡洛丢弃方法，在不增加时间复杂度的前提下保证合并时的正交性，从而避免不同LoRA间的直接干扰。然而实验发现，单纯正交性不足以实现语义解耦和组合性，提示需重新审视正交性在适配融合中的作用。**

- **链接: [http://arxiv.org/pdf/2510.03262v1](http://arxiv.org/pdf/2510.03262v1)**

> **作者:** Andi Zhang; Xuan Ding; Haofan Wang; Steven McDonagh; Samuel Kaski
>
> **摘要:** We propose Orthogonal Monte Carlo Dropout, a mechanism that enforces strict orthogonality when combining sparse semantic vectors without extra time complexity. LoRA, a popular fine-tuning method for large models, typically trains a module to represent a specific concept such as an object or a style. When multiple LoRAs are merged, for example to generate an object in a particular style, their semantic vectors may interfere with each other. Our method guarantees, at the theoretical and runtime levels, that merged LoRAs remain orthogonal and thus free from direct interference. However, empirical analysis reveals that such orthogonality does not lead to the semantic disentanglement or compositionality highlighted in prior work on compositional adaptation. This finding suggests that inter-LoRA orthogonality alone may be insufficient for achieving true semantic compositionality, prompting a re-examination of its role in adapter merging.
>
---
#### [new 194] Real-Time Brain Biomechanics Prediction with Neural Operators: Toward Clinically Deployable Traumatic Brain Injury Models
- **分类: cs.LG; cs.AI; cs.CV; physics.med-ph**

- **简介: 该论文属于医学影像与生物力学建模任务，旨在解决传统有限元模型计算耗时、难以实时应用的问题。作者采用神经算子方法，实现基于患者个体解剖和弹性特征的实时脑位移场预测，提升创伤性脑损伤建模的临床实用性。**

- **链接: [http://arxiv.org/pdf/2510.03248v1](http://arxiv.org/pdf/2510.03248v1)**

> **作者:** Anusha Agarwal; Dibakar Roy Sarkar; Somdatta Goswami
>
> **摘要:** Traumatic brain injury (TBI) remains a major public health concern, with over 69 million cases annually worldwide. Finite element (FE) models offer high-fidelity predictions of brain deformation but are computationally expensive, requiring hours per simulation and limiting their clinical utility for rapid decision-making. This study benchmarks state-of-the-art neural operator (NO) architectures for rapid, patient-specific prediction of brain displacement fields, aiming to enable real-time TBI modeling in clinical and translational settings. We formulated TBI modeling as an operator learning problem, mapping subject-specific anatomical MRI, magnetic resonance elastography (MRE) stiffness maps, and demographic features to full-field 3D brain displacement predictions. Four architectures - Fourier Neural Operator (FNO), Factorized FNO (F-FNO), Multi-Grid FNO (MG-FNO), and Deep Operator Network (DeepONet) were trained and evaluated on 249 MRE datasets across physiologically relevant frequencies (20 - 90 Hz). MG-FNO achieved the highest accuracy (MSE = 0.0023, 94.3\% spatial fidelity) and preserved fine-scale features, while F-FNO converged 2$\times$ faster than standard FNO. DeepONet offered the fastest inference (14.5 iterations/s) with a 7$\times$ computational speed-up over MG-FNO, suggesting utility for embedded or edge computing applications. All NOs reduced computation time from hours to milliseconds without sacrificing anatomical realism. NOs provide an efficient, resolution-invariant approach for predicting brain deformation, opening the door to real-time, patient-specific TBI risk assessment, clinical triage support, and optimization of protective equipment. These results highlight the potential for NO-based digital twins of the human brain, enabling scalable, on-demand biomechanical modeling in both clinical and population health contexts.
>
---
#### [new 195] Joint Neural SDF Reconstruction and Semantic Segmentation for CAD Models
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于三维形状重建与语义分割任务，旨在解决CAD模型重建中几何与语义信息难以联合建模的问题。作者提出一种简单、数据高效的方法，在神经SDF重建网络基础上添加语义分割头，实现几何对齐的部件标签预测，支持任意部件数量的CAD模型，并在重建与分割指标上表现良好。**

- **链接: [http://arxiv.org/pdf/2510.03837v1](http://arxiv.org/pdf/2510.03837v1)**

> **作者:** Shen Fan; Przemyslaw Musialski
>
> **摘要:** We propose a simple, data-efficient pipeline that augments an implicit reconstruction network based on neural SDF-based CAD parts with a part-segmentation head trained under PartField-generated supervision. Unlike methods tied to fixed taxonomies, our model accepts meshes with any number of parts and produces coherent, geometry-aligned labels in a single pass. We evaluate on randomly sampled CAD meshes from the ABC dataset with intentionally varied part cardinalities, including over-segmented shapes, and report strong performance across reconstruction (CDL1/CDL2, F1-micro, NC) and segmentation (mIoU, Accuracy), together with a new Segmentation Consistency metric that captures local label smoothness. We attach a lightweight segmentation head to the Flat-CAD SDF trunk; on a paired evaluation it does not alter reconstruction while providing accurate part labels for meshes with any number of parts. Even under degraded reconstructions on thin or intricate geometries, segmentation remains accurate and label-coherent, often preserving the correct part count. Our approach therefore offers a practical route to semantically structured CAD meshes without requiring curated taxonomies or exact palette matches. We discuss limitations in boundary precision, partly due to per-face supervision, and outline paths toward boundary-aware training and higher resolution labels.
>
---
#### [new 196] Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.IT; math.IT**

- **简介: 该论文属于信息理论与多模态数据分析任务，旨在解决现有部分信息分解（PID）方法在连续高维数据上计算效率低且不准确的问题。论文提出高斯PID（GPID）方法，结合归一化流与梯度优化算法，提高了计算效率与准确性，并推广至非高斯数据场景。**

- **链接: [http://arxiv.org/pdf/2510.04417v1](http://arxiv.org/pdf/2510.04417v1)**

> **作者:** Wenyuan Zhao; Adithya Balachandran; Chao Tian; Paul Pu Liang
>
> **备注:** NeurIPS 2025
>
> **摘要:** The study of multimodality has garnered significant interest in fields where the analysis of interactions among multiple information sources can enhance predictive modeling, data fusion, and interpretability. Partial information decomposition (PID) has emerged as a useful information-theoretic framework to quantify the degree to which individual modalities independently, redundantly, or synergistically convey information about a target variable. However, existing PID methods depend on optimizing over a joint distribution constrained by estimated pairwise probability distributions, which are costly and inaccurate for continuous and high-dimensional modalities. Our first key insight is that the problem can be solved efficiently when the pairwise distributions are multivariate Gaussians, and we refer to this problem as Gaussian PID (GPID). We propose a new gradient-based algorithm that substantially improves the computational efficiency of GPID based on an alternative formulation of the underlying optimization problem. To generalize the applicability to non-Gaussian data, we learn information-preserving encoders to transform random variables of arbitrary input distributions into pairwise Gaussian random variables. Along the way, we resolved an open problem regarding the optimality of joint Gaussian solutions for GPID. Empirical validation in diverse synthetic examples demonstrates that our proposed method provides more accurate and efficient PID estimates than existing baselines. We further evaluate a series of large-scale multimodal benchmarks to show its utility in real-world applications of quantifying PID in multimodal datasets and selecting high-performing models.
>
---
#### [new 197] Efficient Test-Time Scaling for Small Vision-Language Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决小规模模型性能不足的问题。论文提出两种高效的测试时扩展策略：测试时增强（TTAug）和测试时适应（TTAdapt），通过内部特征利用和伪标签优化，提升小模型性能，同时保持计算效率。**

- **链接: [http://arxiv.org/pdf/2510.03574v1](http://arxiv.org/pdf/2510.03574v1)**

> **作者:** Mehmet Onurcan Kaya; Desmond Elliott; Dim P. Papadopoulos
>
> **摘要:** Small Vision-Language Models (VLMs) provide a computationally efficient alternative to larger models, at the cost of weaker generalization abilities and downstream task performance. These shortcomings could be addressed by test-time scaling techniques, but existing methods are typically computationally demanding, contradicting the resource-efficient design goals of small models. To address these limitations, we propose two novel and efficient test-time scaling strategies that leverage the model-internal features rather than external supervision: (i) Test-Time Augmentation (TTAug), which generates multiple augmented inputs and aggregates outputs at the token level without parameter updates, and (ii) Test-Time Adaptation (TTAdapt), which adapts model parameters during inference using consensus-based pseudolabels from TTAug. Through extensive experiments across nine benchmarks, we demonstrate consistent performance improvements while maintaining computational efficiency suitable for resource-constrained environments. The generality of our approach is demonstrated both within models at different scales and across different VLMs without additional tuning.
>
---
#### [new 198] Post-training quantization of vision encoders needs prefixing registers
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于模型压缩任务，旨在解决视觉编码器在后训练量化中因激活值异常（离群点）导致精度下降的问题。作者提出了一种无需训练的方法RegCache，通过引入无语义但易出现离群点的前缀token，抑制其他token的异常值，从而实现更高效的量化。**

- **链接: [http://arxiv.org/pdf/2510.04547v1](http://arxiv.org/pdf/2510.04547v1)**

> **作者:** Seunghyeon Kim; Jinho Kim; Taesun Yeom; Wonpyo Park; Kyuyeun Kim; Jaeho Lee
>
> **摘要:** Transformer-based vision encoders -- such as CLIP -- are central to multimodal intelligence, powering applications from autonomous web agents to robotic control. Since these applications often demand real-time processing of massive visual data, reducing the inference cost of vision encoders is critical. Post-training quantization offers a practical path, but remains challenging even at 8-bit precision due to massive-scale activations (i.e., outliers). In this work, we propose $\textit{RegCache}$, a training-free algorithm to mitigate outliers in vision encoders, enabling quantization with significantly smaller accuracy drops. The proposed RegCache introduces outlier-prone yet semantically meaningless prefix tokens to the target vision encoder, which prevents other tokens from having outliers. Notably, we observe that outliers in vision encoders behave differently from those in language models, motivating two technical innovations: middle-layer prefixing and token deletion. Experiments show that our method consistently improves the accuracy of quantized models across both text-supervised and self-supervised vision encoders.
>
---
#### [new 199] Learning-Based Hashing for ANN Search: Foundations and Early Advances
- **分类: cs.IR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于信息检索任务，旨在解决大规模近似最近邻（ANN）搜索问题。论文综述了基于学习的哈希方法，通过将高维数据映射为紧凑二进制码，加快相似性计算。工作包括回顾监督、无监督与半监督哈希方法，分析投影与量化策略，并探讨跨模态检索的早期进展，以梳理该领域基础思想与发展脉络。**

- **链接: [http://arxiv.org/pdf/2510.04127v1](http://arxiv.org/pdf/2510.04127v1)**

> **作者:** Sean Moran
>
> **摘要:** Approximate Nearest Neighbour (ANN) search is a fundamental problem in information retrieval, underpinning large-scale applications in computer vision, natural language processing, and cross-modal search. Hashing-based methods provide an efficient solution by mapping high-dimensional data into compact binary codes that enable fast similarity computations in Hamming space. Over the past two decades, a substantial body of work has explored learning to hash, where projection and quantisation functions are optimised from data rather than chosen at random. This article offers a foundational survey of early learning-based hashing methods, with an emphasis on the core ideas that shaped the field. We review supervised, unsupervised, and semi-supervised approaches, highlighting how projection functions are designed to generate meaningful embeddings and how quantisation strategies convert these embeddings into binary codes. We also examine extensions to multi-bit and multi-threshold models, as well as early advances in cross-modal retrieval. Rather than providing an exhaustive account of the most recent methods, our goal is to introduce the conceptual foundations of learning-based hashing for ANN search. By situating these early models in their historical context, we aim to equip readers with a structured understanding of the principles, trade-offs, and open challenges that continue to inform current research in this area.
>
---
#### [new 200] Using predefined vector systems as latent space configuration for neural network supervised training on data with arbitrarily large number of classes
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于分类任务，旨在解决神经网络在类别数量极大或未知时的训练难题。通过引入预定义向量系统作为目标潜空间配置，实现不依赖类别数量的模型训练。实验验证了方法在多个数据集上的有效性，并展示了其在百万级类别数据上的应用潜力。**

- **链接: [http://arxiv.org/pdf/2510.04090v1](http://arxiv.org/pdf/2510.04090v1)**

> **作者:** Nikita Gabdullin
>
> **备注:** 28 pages, 12 figures, 10 tables, 12 equations, 1 algorithm
>
> **摘要:** Supervised learning (SL) methods are indispensable for neural network (NN) training used to perform classification tasks. While resulting in very high accuracy, SL training often requires making NN parameter number dependent on the number of classes, limiting their applicability when the number of classes is extremely large or unknown in advance. In this paper we propose a methodology that allows one to train the same NN architecture regardless of the number of classes. This is achieved by using predefined vector systems as the target latent space configuration (LSC) during NN training. We discuss the desired properties of target configurations and choose randomly perturbed vectors of An root system for our experiments. These vectors are used to successfully train encoders and visual transformers (ViT) on Cinic-10 and ImageNet-1K in low- and high-dimensional cases by matching NN predictions with the predefined vectors. Finally, ViT is trained on a dataset with 1.28 million classes illustrating the applicability of the method to training on datasets with extremely large number of classes. In addition, potential applications of LSC in lifelong learning and NN distillation are discussed illustrating versatility of the proposed methodology.
>
---
#### [new 201] Frequency-Aware Model Parameter Explorer: A new attribution method for improving explainability
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于模型解释任务，旨在提升深度神经网络的可解释性。针对现有归因方法效果不佳的问题，作者提出了一种新的归因方法FAMPE，通过可迁移的频率感知攻击，结合高低频成分进行模型参数探索，从而提升解释性能，平均插入得分提高13.02%。**

- **链接: [http://arxiv.org/pdf/2510.03245v1](http://arxiv.org/pdf/2510.03245v1)**

> **作者:** Ali Yavari; Alireza Mohamadi; Elham Beydaghi; Rainer A. Leitgeb
>
> **备注:** Preprint
>
> **摘要:** Ensuring the reliability of deep neural networks (DNNs) in the presence of real world noise and intentional perturbations remains a significant challenge. To address this, attribution methods have been proposed, though their efficacy remains suboptimal and necessitates further refinement. In this paper, we propose a novel category of transferable adversarial attacks, called transferable frequency-aware attacks, enabling frequency-aware exploration via both high-and low-frequency components. Based on this type of attacks, we also propose a novel attribution method, named Frequency-Aware Model Parameter Explorer (FAMPE), which improves the explainability for DNNs. Relative to the current state-of-the-art method AttEXplore, our FAMPE attains an average gain of 13.02% in Insertion Score, thereby outperforming existing approaches. Through detailed ablation studies, we also investigate the role of both high- and low-frequency components in explainability.
>
---
#### [new 202] Real-time nonlinear inversion of magnetic resonance elastography with operator learning
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学成像与机器学习交叉任务，旨在解决磁共振弹性成像（MRE）数据的实时非线性反演问题。传统方法计算慢，论文提出一种基于算子学习的框架（oNLI），实现快速且高精度的弹性图重建，结果表明其性能优于卷积神经网络，达到与非线性反演相当的空间精度。**

- **链接: [http://arxiv.org/pdf/2510.03372v1](http://arxiv.org/pdf/2510.03372v1)**

> **作者:** Juampablo E. Heras Rivera; Caitlin M. Neher; Mehmet Kurt
>
> **摘要:** $\textbf{Purpose:}$ To develop and evaluate an operator learning framework for nonlinear inversion (NLI) of brain magnetic resonance elastography (MRE) data, which enables real-time inversion of elastograms with comparable spatial accuracy to NLI. $\textbf{Materials and Methods:}$ In this retrospective study, 3D MRE data from 61 individuals (mean age, 37.4 years; 34 female) were used for development of the framework. A predictive deep operator learning framework (oNLI) was trained using 10-fold cross-validation, with the complex curl of the measured displacement field as inputs and NLI-derived reference elastograms as outputs. A structural prior mechanism, analogous to Soft Prior Regularization in the MRE literature, was incorporated to improve spatial accuracy. Subject-level evaluation metrics included Pearson's correlation coefficient, absolute relative error, and structural similarity index measure between predicted and reference elastograms across brain regions of different sizes to understand accuracy. Statistical analyses included paired t-tests comparing the proposed oNLI variants to the convolutional neural network baselines. $\textbf{Results:}$ Whole brain absolute percent error was 8.4 $\pm$ 0.5 ($\mu'$) and 10.0 $\pm$ 0.7 ($\mu''$) for oNLI and 15.8 $\pm$ 0.8 ($\mu'$) and 26.1 $\pm$ 1.1 ($\mu''$) for CNNs. Additionally, oNLI outperformed convolutional architectures as per Pearson's correlation coefficient, $r$, in the whole brain and across all subregions for both the storage modulus and loss modulus (p < 0.05). $\textbf{Conclusion:}$ The oNLI framework enables real-time MRE inversion (30,000x speedup), outperforming CNN-based approaches and maintaining the fine-grained spatial accuracy achievable with NLI in the brain.
>
---
#### [new 203] ChartAgent: A Multimodal Agent for Visually Grounded Reasoning in Complex Chart Question Answering
- **分类: cs.AI; cs.CE; cs.CL; cs.CV; stat.ME**

- **简介: 该论文属于图表问答任务，旨在解决现有模型在无标注图表理解中的性能下降问题。作者提出ChartAgent，通过视觉推理和图像操作（如标注、裁剪）逐步分解并处理图表问题，显著提升准确率，尤其在复杂图表上表现优异。**

- **链接: [http://arxiv.org/pdf/2510.04514v1](http://arxiv.org/pdf/2510.04514v1)**

> **作者:** Rachneet Kaur; Nishan Srishankar; Zhen Zeng; Sumitra Ganesh; Manuela Veloso
>
> **备注:** 53 pages, 12 figures, 15 tables
>
> **摘要:** Recent multimodal LLMs have shown promise in chart-based visual question answering, but their performance declines sharply on unannotated charts, those requiring precise visual interpretation rather than relying on textual shortcuts. To address this, we introduce ChartAgent, a novel agentic framework that explicitly performs visual reasoning directly within the chart's spatial domain. Unlike textual chain-of-thought reasoning, ChartAgent iteratively decomposes queries into visual subtasks and actively manipulates and interacts with chart images through specialized actions such as drawing annotations, cropping regions (e.g., segmenting pie slices, isolating bars), and localizing axes, using a library of chart-specific vision tools to fulfill each subtask. This iterative reasoning process closely mirrors human cognitive strategies for chart comprehension. ChartAgent achieves state-of-the-art accuracy on the ChartBench and ChartX benchmarks, surpassing prior methods by up to 16.07% absolute gain overall and 17.31% on unannotated, numerically intensive queries. Furthermore, our analyses show that ChartAgent is (a) effective across diverse chart types, (b) achieve the highest scores across varying visual and reasoning complexity levels, and (c) serves as a plug-and-play framework that boosts performance across diverse underlying LLMs. Our work is among the first to demonstrate visually grounded reasoning for chart understanding using tool-augmented multimodal agents.
>
---
#### [new 204] The method of the approximate inverse for limited-angle CT
- **分类: eess.IV; cs.CV; cs.NA; math.NA**

- **简介: 该论文属于医学成像与工业检测中的图像重建任务，旨在解决有限角度CT重建中的伪影问题。论文提出了一种基于近似逆方法的新模型驱动方法（LARK），结合谱滤波与去噪策略（CLARK），有效抑制伪影并提升重建质量，适用于真实数据。**

- **链接: [http://arxiv.org/pdf/2510.04369v1](http://arxiv.org/pdf/2510.04369v1)**

> **作者:** Bernadette Hahn; Gael Rigaud; Richard Schmähl
>
> **摘要:** Limited-angle computerized tomography stands for one of the most difficult challenges in imaging. Although it opens the way to faster data acquisition in industry and less dangerous scans in medicine, standard approaches, such as the filtered backprojection (FBP) algorithm or the widely used total-variation functional, often produce various artefacts that hinder the diagnosis. With the rise of deep learning, many modern techniques have proven themselves successful in removing such artefacts but at the cost of large datasets. In this paper, we propose a new model-driven approach based on the method of the approximate inverse, which could serve as new starting point for learning strategies in the future. In contrast to FBP-type approaches, our reconstruction step consists in evaluating linear functionals on the measured data using reconstruction kernels that are precomputed as solution of an auxiliary problem. With this problem being uniquely solvable, the derived limited-angle reconstruction kernel (LARK) is able to fully reconstruct the object without the well-known streak artefacts, even for large limited angles. However, it inherits severe ill-conditioning which leads to a different kind of artefacts arising from the singular functions of the limited-angle Radon transform. The problem becomes particularly challenging when working on semi-discrete (real or analytical) measurements. We develop a general regularization strategy, named constrained limited-angle reconstruction kernel (CLARK), by combining spectral filter, the method of the approximate inverse and custom edge-preserving denoising in order to stabilize the whole process. We further derive and interpret error estimates for the application on real, i.e. semi-discrete, data and we validate our approach on synthetic and real data.
>
---
#### [new 205] Pulp Motion: Framing-aware multimodal camera and human motion generation
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于多模态生成任务，旨在解决摄像机轨迹与人物动作联合生成中的画面构图一致性问题。论文提出了一种基于画面构图的联合生成框架，通过共享潜在空间和辅助采样方法，实现了文本条件下的连贯画面生成。同时，论文还引入了PulpMotion数据集，支持高质量的人物动作与摄像机轨迹生成。**

- **链接: [http://arxiv.org/pdf/2510.05097v1](http://arxiv.org/pdf/2510.05097v1)**

> **作者:** Robin Courant; Xi Wang; David Loiseaux; Marc Christie; Vicky Kalogeiton
>
> **备注:** Project page: https://www.lix.polytechnique.fr/vista/projects/2025_pulpmotion_courant/
>
> **摘要:** Treating human motion and camera trajectory generation separately overlooks a core principle of cinematography: the tight interplay between actor performance and camera work in the screen space. In this paper, we are the first to cast this task as a text-conditioned joint generation, aiming to maintain consistent on-screen framing while producing two heterogeneous, yet intrinsically linked, modalities: human motion and camera trajectories. We propose a simple, model-agnostic framework that enforces multimodal coherence via an auxiliary modality: the on-screen framing induced by projecting human joints onto the camera. This on-screen framing provides a natural and effective bridge between modalities, promoting consistency and leading to more precise joint distribution. We first design a joint autoencoder that learns a shared latent space, together with a lightweight linear transform from the human and camera latents to a framing latent. We then introduce auxiliary sampling, which exploits this linear transform to steer generation toward a coherent framing modality. To support this task, we also introduce the PulpMotion dataset, a human-motion and camera-trajectory dataset with rich captions, and high-quality human motions. Extensive experiments across DiT- and MAR-based architectures show the generality and effectiveness of our method in generating on-frame coherent human-camera motions, while also achieving gains on textual alignment for both modalities. Our qualitative results yield more cinematographically meaningful framings setting the new state of the art for this task. Code, models and data are available in our \href{https://www.lix.polytechnique.fr/vista/projects/2025_pulpmotion_courant/}{project page}.
>
---
#### [new 206] Creative synthesis of kinematic mechanisms
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于机械设计与图像生成交叉任务，旨在解决平面连杆机构的运动综合问题。作者构建了一个包含多种机构的图像数据集，采用共享潜在空间的变分自编码器，实现基于轨迹形状与速度的新型机构生成。验证显示该方法能有效合成含转动副、移动副等复杂机构。**

- **链接: [http://arxiv.org/pdf/2510.03308v1](http://arxiv.org/pdf/2510.03308v1)**

> **作者:** Jiong Lin; Jialong Ning; Judah Goldfeder; Hod Lipson
>
> **备注:** 6pages, 6 figures
>
> **摘要:** In this paper, we formulate the problem of kinematic synthesis for planar linkages as a cross-domain image generation task. We develop a planar linkages dataset using RGB image representations, covering a range of mechanisms: from simple types such as crank-rocker and crank-slider to more complex eight-bar linkages like Jansen's mechanism. A shared-latent variational autoencoder (VAE) is employed to explore the potential of image generative models for synthesizing unseen motion curves and simulating novel kinematics. By encoding the drawing speed of trajectory points as color gradients, the same architecture also supports kinematic synthesis conditioned on both trajectory shape and velocity profiles. We validate our method on three datasets of increasing complexity: a standard four-bar linkage set, a mixed set of four-bar and crank-slider mechanisms, and a complex set including multi-loop mechanisms. Preliminary results demonstrate the effectiveness of image-based representations for generative mechanical design, showing that mechanisms with revolute and prismatic joints, and potentially cams and gears, can be represented and synthesized within a unified image generation framework.
>
---
#### [new 207] Model-Guided Microstimulation Steers Primate Visual Behavior
- **分类: q-bio.NC; cs.CV**

- **简介: 该论文属于神经科学与视觉 prosthetics 任务，旨在解决如何通过脑刺激诱发复杂视觉感知的问题。作者构建了一个计算框架，结合神经活动模型与微刺激实验，在猕猴上验证了模型引导刺激高阶视觉皮层的有效性，为下一代视觉假体提供新方法。**

- **链接: [http://arxiv.org/pdf/2510.03684v1](http://arxiv.org/pdf/2510.03684v1)**

> **作者:** Johannes Mehrer; Ben Lonnqvist; Anna Mitola; Abdulkadir Gokce; Paolo Papale; Martin Schrimpf
>
> **摘要:** Brain stimulation is a powerful tool for understanding cortical function and holds promise for therapeutic interventions in neuropsychiatric disorders. Initial visual prosthetics apply electric microstimulation to early visual cortex which can evoke percepts of simple symbols such as letters. However, these approaches are fundamentally limited by hardware constraints and the low-level representational properties of this cortical region. In contrast, higher-level visual areas encode more complex object representations and therefore constitute a promising target for stimulation - but determining representational targets that reliably evoke object-level percepts constitutes a major challenge. We here introduce a computational framework to causally model and guide stimulation of high-level cortex, comprising three key components: (1) a perturbation module that translates microstimulation parameters into spatial changes to neural activity, (2) topographic models that capture the spatial organization of cortical neurons and thus enable prototyping of stimulation experiments, and (3) a mapping procedure that links model-optimized stimulation sites back to primate cortex. Applying this framework in two macaque monkeys performing a visual recognition task, model-predicted stimulation experiments produced significant in-vivo changes in perceptual choices. Per-site model predictions and monkey behavior were strongly correlated, underscoring the promise of model-guided stimulation. Image generation further revealed a qualitative similarity between in-silico stimulation of face-selective sites and a patient's report of facephenes. This proof-of-principle establishes a foundation for model-guided microstimulation and points toward next-generation visual prosthetics capable of inducing more complex visual experiences.
>
---
#### [new 208] Visual Lifelog Retrieval through Captioning-Enhanced Interpretation
- **分类: cs.IR; cs.CL; cs.CV; cs.MM**

- **简介: 该论文属于视觉 lifelog 检索任务，旨在帮助用户通过文本查询快速找到个人 lifelog 中的特定图像。为解决传统方法在理解第一视角图像中的局限，论文提出 CIVIL 系统，结合图像描述生成与文本嵌入，提升检索效果，并构建了相关文本数据集。**

- **链接: [http://arxiv.org/pdf/2510.04010v1](http://arxiv.org/pdf/2510.04010v1)**

> **作者:** Yu-Fei Shih; An-Zi Yen; Hen-Hsen Huang; Hsin-Hsi Chen
>
> **摘要:** People often struggle to remember specific details of past experiences, which can lead to the need to revisit these memories. Consequently, lifelog retrieval has emerged as a crucial application. Various studies have explored methods to facilitate rapid access to personal lifelogs for memory recall assistance. In this paper, we propose a Captioning-Integrated Visual Lifelog (CIVIL) Retrieval System for extracting specific images from a user's visual lifelog based on textual queries. Unlike traditional embedding-based methods, our system first generates captions for visual lifelogs and then utilizes a text embedding model to project both the captions and user queries into a shared vector space. Visual lifelogs, captured through wearable cameras, provide a first-person viewpoint, necessitating the interpretation of the activities of the individual behind the camera rather than merely describing the scene. To address this, we introduce three distinct approaches: the single caption method, the collective caption method, and the merged caption method, each designed to interpret the life experiences of lifeloggers. Experimental results show that our method effectively describes first-person visual images, enhancing the outcomes of lifelog retrieval. Furthermore, we construct a textual dataset that converts visual lifelogs into captions, thereby reconstructing personal life experiences.
>
---
#### [new 209] Sliding Window Attention for Learned Video Compression
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于视频压缩任务，旨在解决Transformer在视频压缩中因局部注意力机制导致的架构缺陷和计算冗余问题。论文提出了一种无分块的三维滑动窗口注意力方法，统一处理时空上下文，提升了压缩性能并降低了计算复杂度。**

- **链接: [http://arxiv.org/pdf/2510.03926v1](http://arxiv.org/pdf/2510.03926v1)**

> **作者:** Alexander Kopte; André Kaup
>
> **备注:** Accepted for PCS 2025
>
> **摘要:** To manage the complexity of transformers in video compression, local attention mechanisms are a practical necessity. The common approach of partitioning frames into patches, however, creates architectural flaws like irregular receptive fields. When adapted for temporal autoregressive models, this paradigm, exemplified by the Video Compression Transformer (VCT), also necessitates computationally redundant overlapping windows. This work introduces 3D Sliding Window Attention (SWA), a patchless form of local attention. By enabling a decoder-only architecture that unifies spatial and temporal context processing, and by providing a uniform receptive field, our method significantly improves rate-distortion performance, achieving Bj{\o}rntegaard Delta-rate savings of up to 18.6 % against the VCT baseline. Simultaneously, by eliminating the need for overlapping windows, our method reduces overall decoder complexity by a factor of 2.8, while its entropy model is nearly 3.5 times more efficient. We further analyze our model's behavior and show that while it benefits from long-range temporal context, excessive context can degrade performance.
>
---
#### [new 210] StaMo: Unsupervised Learning of Generalizable Robot Motion from Compact State Representation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决如何从静态图像中学习紧凑且通用的状态表示，以提升机器人运动控制性能。论文提出StaMo方法，通过轻量编码器和预训练DiT解码器，实现无监督学习两token状态表示，并通过隐空间插值得到有效潜动作，提升任务成功率，减少对复杂架构和视频数据的依赖。**

- **链接: [http://arxiv.org/pdf/2510.05057v1](http://arxiv.org/pdf/2510.05057v1)**

> **作者:** Mingyu Liu; Jiuhe Shu; Hui Chen; Zeju Li; Canyu Zhao; Jiange Yang; Shenyuan Gao; Hao Chen; Chunhua Shen
>
> **摘要:** A fundamental challenge in embodied intelligence is developing expressive and compact state representations for efficient world modeling and decision making. However, existing methods often fail to achieve this balance, yielding representations that are either overly redundant or lacking in task-critical information. We propose an unsupervised approach that learns a highly compressed two-token state representation using a lightweight encoder and a pre-trained Diffusion Transformer (DiT) decoder, capitalizing on its strong generative prior. Our representation is efficient, interpretable, and integrates seamlessly into existing VLA-based models, improving performance by 14.3% on LIBERO and 30% in real-world task success with minimal inference overhead. More importantly, we find that the difference between these tokens, obtained via latent interpolation, naturally serves as a highly effective latent action, which can be further decoded into executable robot actions. This emergent capability reveals that our representation captures structured dynamics without explicit supervision. We name our method StaMo for its ability to learn generalizable robotic Motion from compact State representation, which is encoded from static images, challenging the prevalent dependence to learning latent action on complex architectures and video data. The resulting latent actions also enhance policy co-training, outperforming prior methods by 10.4% with improved interpretability. Moreover, our approach scales effectively across diverse data sources, including real-world robot data, simulation, and human egocentric video.
>
---
#### [new 211] Fast Witness Persistence for MRI Volumes via Hybrid Landmarking
- **分类: cs.CG; cs.CV; cs.LG**

- **简介: 该论文属于医学影像分析任务，旨在高效提取MRI数据的拓扑特征。为解决传统方法计算量大、难以扩展的问题，作者提出了一种结合密度感知与几何覆盖的混合标记选择方法，并设计了适用于GPU的快速过滤算法，实现了快速且拓扑保持的MRI体积数据分析。**

- **链接: [http://arxiv.org/pdf/2510.04553v1](http://arxiv.org/pdf/2510.04553v1)**

> **作者:** Jorge Leonardo Ruiz Williams
>
> **摘要:** We introduce a scalable witness-based persistent homology pipeline for full-brain MRI volumes that couples density-aware landmark selection with a GPU-ready witness filtration. Candidates are scored by a hybrid metric that balances geometric coverage against inverse kernel density, yielding landmark sets that shrink mean pairwise distances by 30-60% over random or density-only baselines while preserving topological features. Benchmarks on BrainWeb, IXI, and synthetic manifolds execute in under ten seconds on a single NVIDIA RTX 4090 GPU, avoiding the combinatorial blow-up of Cech, Vietoris-Rips, and alpha filtrations. The package is distributed on PyPI as whale-tda (installable via pip); source and issues are hosted at https://github.com/jorgeLRW/whale. The release also exposes a fast preset (mri_deep_dive_fast) for exploratory sweeps, and ships with reproducibility-focused scripts and artifacts for drop-in use in medical imaging workflows.
>
---
#### [new 212] 3Dify: a Framework for Procedural 3D-CG Generation Assisted by LLMs Using MCP and RAG
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 论文提出“3Dify”框架，属3D内容生成任务，旨在通过自然语言指令生成3D-CG内容。利用LLM、MCP与RAG技术，自动化操作DCC工具或GUI，支持本地模型部署，并通过用户反馈优化生成效果。**

- **链接: [http://arxiv.org/pdf/2510.04536v1](http://arxiv.org/pdf/2510.04536v1)**

> **作者:** Shun-ichiro Hayashi; Daichi Mukunoki; Tetsuya Hoshino; Satoshi Ohshima; Takahiro Katagiri
>
> **摘要:** This paper proposes "3Dify," a procedural 3D computer graphics (3D-CG) generation framework utilizing Large Language Models (LLMs). The framework enables users to generate 3D-CG content solely through natural language instructions. 3Dify is built upon Dify, an open-source platform for AI application development, and incorporates several state-of-the-art LLM-related technologies such as the Model Context Protocol (MCP) and Retrieval-Augmented Generation (RAG). For 3D-CG generation support, 3Dify automates the operation of various Digital Content Creation (DCC) tools via MCP. When DCC tools do not support MCP-based interaction, the framework employs the Computer-Using Agent (CUA) method to automate Graphical User Interface (GUI) operations. Moreover, to enhance image generation quality, 3Dify allows users to provide feedback by selecting preferred images from multiple candidates. The LLM then learns variable patterns from these selections and applies them to subsequent generations. Furthermore, 3Dify supports the integration of locally deployed LLMs, enabling users to utilize custom-developed models and to reduce both time and monetary costs associated with external API calls by leveraging their own computational resources.
>
---
#### [new 213] NoTVLA: Narrowing of Dense Action Trajectories for Generalizable Robot Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决视觉-语言-动作（VLA）模型在实际部署中因密集动作序列导致的灾难性遗忘问题。作者提出NoTVLA框架，通过稀疏轨迹训练和末端执行器轨迹优化，提升跨任务泛化能力，减少计算资源依赖，同时保持语言理解和零样本迁移能力。**

- **链接: [http://arxiv.org/pdf/2510.03895v1](http://arxiv.org/pdf/2510.03895v1)**

> **作者:** Zheng Huang; Mingyu Liu; Xiaoyi Lin; Muzhi Zhu; Canyu Zhao; Zongze Du; Xiaoman Li; Yiduo Jia; Hao Zhong; Hao Chen; Chunhua Shen
>
> **摘要:** Vision-Language-Action (VLA) models represent a pivotal advance in embodied intelligence, yet they confront critical barriers to real-world deployment, most notably catastrophic forgetting. This issue stems from their overreliance on continuous action sequences or action chunks, which inadvertently create isolated data silos that disrupt knowledge retention across tasks. To tackle these challenges, we propose the Narrowing of Trajectory VLA (NoTVLA) framework: a novel approach that narrows its focus to sparse trajectories, thereby avoiding the catastrophic forgetting associated with dense trajectory fine-tuning. A key innovation of NoTVLA lies in its trajectory planning strategy: instead of centering on the target object's trajectory, it leverages temporal compression and spatial reasoning pruning specifically for the robot end effector's trajectory. Furthermore, training is conducted using these sparse trajectories rather than dense action trajectories, an optimization that delivers remarkable practical advantages with better performance in zero-shot. In multi-task evaluation scenarios, NoTVLA achieves superior performance and generalization compared to pi0 while operating under two critical constraints: it uses over an order of magnitude less computing power than pi0 and requires no wrist-mounted camera. This design ensures that NoTVLA's operational accuracy closely approximates that of single-task expert models. Crucially, it also preserves the model's inherent language capabilities, enabling zero-shot generalization in specific scenarios, supporting unified model deployment across multiple robot platforms, and fostering a degree of generalization even when perceiving tasks from novel perspectives.
>
---
#### [new 214] Real-time Prediction of Urban Sound Propagation with Conditioned Normalizing Flows
- **分类: cs.LG; cs.CV**

- **简介: 论文任务为实时预测城市声音传播。解决传统物理方法速度慢、难以满足城市规划中噪声地图快速生成需求的问题。工作提出基于条件归一化流（Full-Glow）的模型，实现从2D城市布局快速生成高精度声压图，加速超2000倍并提升非视距预测准确性。**

- **链接: [http://arxiv.org/pdf/2510.04510v1](http://arxiv.org/pdf/2510.04510v1)**

> **作者:** Achim Eckerle; Martin Spitznagel; Janis Keuper
>
> **摘要:** Accurate and fast urban noise prediction is pivotal for public health and for regulatory workflows in cities, where the Environmental Noise Directive mandates regular strategic noise maps and action plans, often needed in permission workflows, right-of-way allocation, and construction scheduling. Physics-based solvers are too slow for such time-critical, iterative "what-if" studies. We evaluate conditional Normalizing Flows (Full-Glow) for generating for generating standards-compliant urban sound-pressure maps from 2D urban layouts in real time per 256x256 map on a single RTX 4090), enabling interactive exploration directly on commodity hardware. On datasets covering Baseline, Diffraction, and Reflection regimes, our model accelerates map generation by >2000 times over a reference solver while improving NLoS accuracy by up to 24% versus prior deep models; in Baseline NLoS we reach 0.65 dB MAE with high structural fidelity. The model reproduces diffraction and interference patterns and supports instant recomputation under source or geometry changes, making it a practical engine for urban planning, compliance mapping, and operations (e.g., temporary road closures, night-work variance assessments).
>
---
#### [new 215] Longitudinal Flow Matching for Trajectory Modeling
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文属于轨迹建模任务，旨在解决稀疏采样和高维序列数据建模困难的问题。作者提出了IMMFM方法，通过多时间点联合建模，使用插值路径进行流匹配，同时优化漂移和扩散系数，从而提高预测准确性和下游任务表现。**

- **链接: [http://arxiv.org/pdf/2510.03569v1](http://arxiv.org/pdf/2510.03569v1)**

> **作者:** Mohammad Mohaiminul Islam; Thijs P. Kuipers; Sharvaree Vadgama; Coen de Vente; Afsana Khan; Clara I. Sánchez; Erik J. Bekkers
>
> **摘要:** Generative models for sequential data often struggle with sparsely sampled and high-dimensional trajectories, typically reducing the learning of dynamics to pairwise transitions. We propose \textit{Interpolative Multi-Marginal Flow Matching} (IMMFM), a framework that learns continuous stochastic dynamics jointly consistent with multiple observed time points. IMMFM employs a piecewise-quadratic interpolation path as a smooth target for flow matching and jointly optimizes drift and a data-driven diffusion coefficient, supported by a theoretical condition for stable learning. This design captures intrinsic stochasticity, handles irregular sparse sampling, and yields subject-specific trajectories. Experiments on synthetic benchmarks and real-world longitudinal neuroimaging datasets show that IMMFM outperforms existing methods in both forecasting accuracy and further downstream tasks.
>
---
#### [new 216] Watch and Learn: Learning to Use Computers from Online Videos
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于计算机使用代理（CUA）任务，旨在解决训练数据稀缺问题。通过提出Watch & Learn框架，从在线视频中大规模提取高质量UI操作轨迹，用于提升CUA的任务规划能力。方法基于逆动力学建模，减少人工标注，提升跨应用泛化能力，并在OSWorld基准上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2510.04673v1](http://arxiv.org/pdf/2510.04673v1)**

> **作者:** Chan Hee Song; Yiwen Song; Palash Goyal; Yu Su; Oriana Riva; Hamid Palangi; Tomas Pfister
>
> **摘要:** Computer use agents (CUAs) need to plan task workflows grounded in diverse, ever-changing applications and environments, but learning is hindered by the scarcity of large-scale, high-quality training data in the target application. Existing datasets are domain-specific, static, and costly to annotate, while current synthetic data generation methods often yield simplistic or misaligned task demonstrations. To address these limitations, we introduce Watch & Learn (W&L), a framework that converts human demonstration videos readily available on the Internet into executable UI trajectories at scale. Instead of directly generating trajectories or relying on ad hoc reasoning heuristics, we cast the problem as an inverse dynamics objective: predicting the user's action from consecutive screen states. This formulation reduces manual engineering, is easier to learn, and generalizes more robustly across applications. Concretely, we develop an inverse dynamics labeling pipeline with task-aware video retrieval, generate over 53k high-quality trajectories from raw web videos, and demonstrate that these trajectories improve CUAs both as in-context demonstrations and as supervised training data. On the challenging OSWorld benchmark, UI trajectories extracted with W&L consistently enhance both general-purpose and state-of-the-art frameworks in-context, and deliver stronger gains for open-source models under supervised training. These results highlight web-scale human demonstration videos as a practical and scalable foundation for advancing CUAs towards real-world deployment.
>
---
#### [new 217] Efficient Surgical Robotic Instrument Pose Reconstruction in Real World Conditions Using Unified Feature Detection
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉与控制任务，旨在解决微创手术机器人在真实场景中的位姿重建问题。由于机器人结构复杂且视觉受限，传统标定方法效果不佳。论文提出一种融合关键点与边缘特征的统一检测框架，通过单次推理实现高效准确的位姿估计，适用于在线控制。**

- **链接: [http://arxiv.org/pdf/2510.03532v1](http://arxiv.org/pdf/2510.03532v1)**

> **作者:** Zekai Liang; Kazuya Miyata; Xiao Liang; Florian Richter; Michael C. Yip
>
> **摘要:** Accurate camera-to-robot calibration is essential for any vision-based robotic control system and especially critical in minimally invasive surgical robots, where instruments conduct precise micro-manipulations. However, MIS robots have long kinematic chains and partial visibility of their degrees of freedom in the camera, which introduces challenges for conventional camera-to-robot calibration methods that assume stiff robots with good visibility. Previous works have investigated both keypoint-based and rendering-based approaches to address this challenge in real-world conditions; however, they often struggle with consistent feature detection or have long inference times, neither of which are ideal for online robot control. In this work, we propose a novel framework that unifies the detection of geometric primitives (keypoints and shaft edges) through a shared encoding, enabling efficient pose estimation via projection geometry. This architecture detects both keypoints and edges in a single inference and is trained on large-scale synthetic data with projective labeling. This method is evaluated across both feature detection and pose estimation, with qualitative and quantitative results demonstrating fast performance and state-of-the-art accuracy in challenging surgical environments.
>
---
#### [new 218] MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文属于音频-视觉语音识别（AVSR）任务，旨在解决现有模型计算成本高、压缩率固定、跨尺度泛化能力弱的问题。作者提出MoME框架，结合Matryoshka表示学习与稀疏专家混合机制，实现多粒度压缩与动态资源分配，提升了模型效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.04136v1](http://arxiv.org/pdf/2510.04136v1)**

> **作者:** Umberto Cappellazzo; Minsu Kim; Pingchuan Ma; Honglie Chen; Xubo Liu; Stavros Petridis; Maja Pantic
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large language models (LLMs) have recently shown strong potential in audio-visual speech recognition (AVSR), but their high computational demands and sensitivity to token granularity limit their practicality in resource-constrained settings. Token compression methods can reduce inference cost, but they require fixing a compression rate in advance and produce a single fixed-length output, offering no flexibility to balance information density and efficiency at inference time. Matryoshka representation learning (MRL) addresses this by enabling a single model to operate across multiple token granularities, allowing compression rates to be adjusted dynamically. However, current MRL-based methods treat each scale independently during training, limiting cross-scale generalization, robustness at high compression, and interpretability. To overcome these limitations, we propose MoME (Mixture of Matryoshka Experts), a novel framework that integrates sparse Mixture-of-Experts (MoE) into MRL-based LLMs for AVSR. MoME augments a frozen LLM with top-k routed and shared experts, allowing dynamic capacity allocation across scales and modalities. A shared router promotes consistent expert activation across granularities, enabling compressed sequences to benefit from representations learned at lower compression. Experiments on LRS2 and LRS3 demonstrate that MoME achieves state-of-the-art performance across AVSR, ASR, and VSR tasks, while requiring significantly fewer parameters and maintaining robustness under noise. MoME unifies the adaptability of MRL with the efficiency of MoE, offering a scalable and interpretable solution for resource-aware speech recognition.
>
---
## 更新

#### [replaced 001] What Lurks Within? Concept Auditing for Shared Diffusion Models at Scale
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14815v2](http://arxiv.org/pdf/2504.14815v2)**

> **作者:** Xiaoyong Yuan; Xiaolong Ma; Linke Guo; Lan Zhang
>
> **备注:** Extended version of the paper accepted at CCS 2025
>
> **摘要:** Diffusion models (DMs) have revolutionized text-to-image generation, enabling the creation of highly realistic and customized images from text prompts. With the rise of parameter-efficient fine-tuning (PEFT) techniques, users can now customize powerful pre-trained models using minimal computational resources. However, the widespread sharing of fine-tuned DMs on open platforms raises growing ethical and legal concerns, as these models may inadvertently or deliberately generate sensitive or unauthorized content. Despite increasing regulatory attention on generative AI, there are currently no practical tools for systematically auditing these models before deployment. In this paper, we address the problem of concept auditing: determining whether a fine-tuned DM has learned to generate a specific target concept. Existing approaches typically rely on prompt-based input crafting and output-based image classification but they suffer from critical limitations, including prompt uncertainty, concept drift, and poor scalability. To overcome these challenges, we introduce Prompt-Agnostic Image-Free Auditing (PAIA), a novel, model-centric concept auditing framework. By treating the DM as the object of inspection, PAIA enables direct analysis of internal model behavior, bypassing the need for optimized prompts or generated images. We evaluate PAIA on 320 controlled models trained with curated concept datasets and 771 real-world community models sourced from a public DM sharing platform. Evaluation results show that PAIA achieves over 90% detection accuracy while reducing auditing time by 18 - 40X compared to existing baselines. To our knowledge, PAIA is the first scalable and practical solution for pre-deployment concept auditing of diffusion models, providing a practical foundation for safer and more transparent diffusion model sharing.
>
---
#### [replaced 002] DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.11100v2](http://arxiv.org/pdf/2412.11100v2)**

> **作者:** Jinxiu Liu; Shaoheng Lin; Yinxiao Li; Ming-Hsuan Yang
>
> **备注:** CVPR 2025
>
> **摘要:** The increasing demand for immersive AR/VR applications and spatial intelligence has heightened the need to generate high-quality scene-level and 360${\deg}$ panoramic video. However, most video diffusion models are constrained by limited resolution and aspect ratio, which restricts their applicability to scene-level dynamic content synthesis. In this work, we propose $\textbf{DynamicScaler}$, addressing these challenges by enabling spatially scalable and panoramic dynamic scene synthesis that preserves coherence across panoramic scenes of arbitrary size. Specifically, we introduce a Offset Shifting Denoiser, facilitating efficient, synchronous, and coherent denoising panoramic dynamic scenes via a diffusion model with fixed resolution through a seamless rotating Window, which ensures seamless boundary transitions and consistency across the entire panoramic space, accommodating varying resolutions and aspect ratios. Additionally, we employ a Global Motion Guidance mechanism to ensure both local detail fidelity and global motion continuity. Extensive experiments demonstrate our method achieves superior content and motion quality in panoramic scene-level video generation, offering a training-free, efficient, and scalable solution for immersive dynamic scene creation with constant VRAM consumption regardless of the output video resolution. Project page is available at $\href{https://dynamic-scaler.pages.dev/new}{https://dynamic-scaler.pages.dev/new}$.
>
---
#### [replaced 003] AutoDrive-QA: A Multiple-Choice Benchmark for Vision-Language Evaluation in Urban Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.15778v2](http://arxiv.org/pdf/2503.15778v2)**

> **作者:** Boshra Khalili; Andrew W. Smyth
>
> **备注:** Updated results with larger dataset experiments and expanded discussion
>
> **摘要:** Evaluating vision-language models (VLMs) in urban driving contexts remains challenging, as existing benchmarks rely on open-ended responses that are ambiguous, annotation-intensive, and inconsistent to score. This lack of standardized evaluation slows progress toward safe and reliable AI for urban mobility. We introduce AutoDrive-QA, the first benchmark that systematically converts open-ended driving QA datasets (DriveLM, NuScenes-QA, LingoQA) into structured multiple-choice questions (MCQs) with distractors grounded in five realistic error categories: Driving Domain Misconceptions, Logical Inconsistencies, Misinterpreted Sensor Inputs, Computational Oversights, and Question Ambiguity. This framework enables reproducible and interpretable evaluation of VLMs across perception, prediction, and planning tasks in complex urban scenes. Experiments show that fine-tuning LLaVA-1.5-7B improves accuracy by about six percentage points across tasks, GPT-4V achieves the strongest zero-shot performance with up to 69.8% accuracy, and Qwen2-VL models also perform competitively, particularly in multi-view settings. Moreover, traditional metrics such as BLEU and CIDEr fail to distinguish strong from weak models. By providing an objective, domain-grounded evaluation protocol, AutoDrive-QA contributes to more transparent benchmarking of urban AI systems, supporting the development of safer and more trustworthy autonomous driving technologies for smart cities.
>
---
#### [replaced 004] Fast constrained sampling in pre-trained diffusion models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.18804v3](http://arxiv.org/pdf/2410.18804v3)**

> **作者:** Alexandros Graikos; Nebojsa Jojic; Dimitris Samaras
>
> **摘要:** Large denoising diffusion models, such as Stable Diffusion, have been trained on billions of image-caption pairs to perform text-conditioned image generation. As a byproduct of this training, these models have acquired general knowledge about image statistics, which can be useful for other inference tasks. However, when confronted with sampling an image under new constraints, e.g. generating the missing parts of an image, using large pre-trained text-to-image diffusion models is inefficient and often unreliable. Previous approaches either utilized backpropagation through the denoiser network, making them significantly slower and more memory-demanding than simple text-to-image generation, or only enforced the constraint locally, failing to capture critical long-range correlations in the sampled image. In this work, we propose an algorithm that enables fast, high-quality generation under arbitrary constraints. We show that in denoising diffusion models, we can employ an approximation to Newton's optimization method that allows us to speed up inference and avoid the expensive backpropagation operations. Our approach produces results that rival or surpass the state-of-the-art training-free inference methods while requiring a fraction of the time. We demonstrate the effectiveness of our algorithm under both linear (inpainting, super-resolution) and non-linear (style-guided generation) constraints. An implementation is provided at https://github.com/cvlab-stonybrook/fast-constrained-sampling.
>
---
#### [replaced 005] Humanoid Policy ~ Human Policy
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13441v3](http://arxiv.org/pdf/2503.13441v3)**

> **作者:** Ri-Zhao Qiu; Shiqi Yang; Xuxin Cheng; Chaitanya Chawla; Jialong Li; Tairan He; Ge Yan; David J. Yoon; Ryan Hoque; Lars Paulsen; Ge Yang; Jian Zhang; Sha Yi; Guanya Shi; Xiaolong Wang
>
> **备注:** Code and data: https://human-as-robot.github.io/
>
> **摘要:** Training manipulation policies for humanoid robots with diverse data enhances their robustness and generalization across tasks and platforms. However, learning solely from robot demonstrations is labor-intensive, requiring expensive tele-operated data collection which is difficult to scale. This paper investigates a more scalable data source, egocentric human demonstrations, to serve as cross-embodiment training data for robot learning. We mitigate the embodiment gap between humanoids and humans from both the data and modeling perspectives. We collect an egocentric task-oriented dataset (PH2D) that is directly aligned with humanoid manipulation demonstrations. We then train a human-humanoid behavior policy, which we term Human Action Transformer (HAT). The state-action space of HAT is unified for both humans and humanoid robots and can be differentiably retargeted to robot actions. Co-trained with smaller-scale robot data, HAT directly models humanoid robots and humans as different embodiments without additional supervision. We show that human data improves both generalization and robustness of HAT with significantly better data collection efficiency. Code and data: https://human-as-robot.github.io/
>
---
#### [replaced 006] Physics-Based Motion Imitation with Adversarial Differential Discriminators
- **分类: cs.GR; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.04961v2](http://arxiv.org/pdf/2505.04961v2)**

> **作者:** Ziyu Zhang; Sergey Bashkirov; Dun Yang; Yi Shi; Michael Taylor; Xue Bin Peng
>
> **备注:** SIGGRAPH Asia 2025 Conference Papers
>
> **摘要:** Multi-objective optimization problems, which require the simultaneous optimization of multiple objectives, are prevalent across numerous applications. Existing multi-objective optimization methods often rely on manually-tuned aggregation functions to formulate a joint optimization objective. The performance of such hand-tuned methods is heavily dependent on careful weight selection, a time-consuming and laborious process. These limitations also arise in the setting of reinforcement-learning-based motion tracking methods for physically simulated characters, where intricately crafted reward functions are typically used to achieve high-fidelity results. Such solutions not only require domain expertise and significant manual tuning, but also limit the applicability of the resulting reward function across diverse skills. To bridge this gap, we present a novel adversarial multi-objective optimization technique that is broadly applicable to a range of multi-objective reinforcement-learning tasks, including motion tracking. Our proposed Adversarial Differential Discriminator (ADD) receives a single positive sample, yet is still effective at guiding the optimization process. We demonstrate that our technique can enable characters to closely replicate a variety of acrobatic and agile behaviors, achieving comparable quality to state-of-the-art motion-tracking methods, without relying on manually-designed reward functions. Code and results are available at https://add-moo.github.io/.
>
---
#### [replaced 007] Reasoning-Enhanced Domain-Adaptive Pretraining of Multimodal Large Language Models for Short Video Content Governance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.21486v2](http://arxiv.org/pdf/2509.21486v2)**

> **作者:** Zixuan Wang; Yu Sun; Hongwei Wang; Baoyu Jing; Xiang Shen; Xin Dong; Zhuolin Hao; Hongyu Xiong; Yang Song
>
> **摘要:** Short video platforms are evolving rapidly, making the identification of inappropriate content increasingly critical. Existing approaches typically train separate and small classification models for each type of issue, which requires extensive human-labeled data and lacks cross-issue generalization. We propose a reasoning-enhanced multimodal large language model (MLLM) pretraining paradigm for unified inappropriate content detection. To address the distribution gap between short video content and the original pretraining data of MLLMs, as well as the complex issue definitions, we introduce three targeted pretraining tasks: (1) \textit{Caption}, to enhance the MLLM's perception of video details; (2) \textit{Visual Question Answering (VQA)}, to deepen the MLLM's understanding of issue definitions and annotation guidelines; (3) \textit{Chain-of-Thought (CoT)}, to enhance the MLLM's reasoning capability. Experimental results show that our pretraining approach significantly improves the MLLM's performance in both zero-shot and supervised fine-tuning (SFT) settings. In addition, our pretrained model demonstrates strong generalization capabilities to emergent, previously unseen issues.
>
---
#### [replaced 008] How Far are AI-generated Videos from Simulating the 3D Visual World: A Learned 3D Evaluation Approach
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.19568v2](http://arxiv.org/pdf/2406.19568v2)**

> **作者:** Chirui Chang; Jiahui Liu; Zhengzhe Liu; Xiaoyang Lyu; Yi-Hua Huang; Xin Tao; Pengfei Wan; Di Zhang; Xiaojuan Qi
>
> **备注:** ICCV 2025
>
> **摘要:** Recent advancements in video diffusion models enable the generation of photorealistic videos with impressive 3D consistency and temporal coherence. However, the extent to which these AI-generated videos simulate the 3D visual world remains underexplored. In this paper, we introduce Learned 3D Evaluation (L3DE), an objective, quantifiable, and interpretable method for assessing AI-generated videos' ability to simulate the real world in terms of 3D visual qualities and consistencies, without requiring manually labeled defects or quality annotations. Instead of relying on 3D reconstruction, which is prone to failure with in-the-wild videos, L3DE employs a 3D convolutional network, trained on monocular 3D cues of motion, depth, and appearance, to distinguish real from synthetic videos. Confidence scores from L3DE quantify the gap between real and synthetic videos in terms of 3D visual coherence, while a gradient-based visualization pinpoints unrealistic regions, improving interpretability. We validate L3DE through extensive experiments, demonstrating strong alignment with 3D reconstruction quality and human judgments. Our evaluations on leading generative models (e.g., Kling, Sora, and MiniMax) reveal persistent simulation gaps and subtle inconsistencies. Beyond generative video assessment, L3DE extends to broader applications: benchmarking video generation models, serving as a deepfake detector, and enhancing video synthesis by inpainting flagged inconsistencies. Project page: https://justin-crchang.github.io/l3de-project-page/
>
---
#### [replaced 009] VisualChef: Generating Visual Aids in Cooking via Mask Inpainting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18569v2](http://arxiv.org/pdf/2506.18569v2)**

> **作者:** Oleh Kuzyk; Zuoyue Li; Marc Pollefeys; Xi Wang
>
> **备注:** GCPR 2025 (oral presentation; Best Master's Thesis Award)
>
> **摘要:** Cooking requires not only following instructions but also understanding, executing, and monitoring each step - a process that can be challenging without visual guidance. Although recipe images and videos offer helpful cues, they often lack consistency in focus, tools, and setup. To better support the cooking process, we introduce VisualChef, a method for generating contextual visual aids tailored to cooking scenarios. Given an initial frame and a specified action, VisualChef generates images depicting both the action's execution and the resulting appearance of the object, while preserving the initial frame's environment. Previous work aims to integrate knowledge extracted from large language models by generating detailed textual descriptions to guide image generation, which requires fine-grained visual-textual alignment and involves additional annotations. In contrast, VisualChef simplifies alignment through mask-based visual grounding. Our key insight is identifying action-relevant objects and classifying them to enable targeted modifications that reflect the intended action and outcome while maintaining a consistent environment. In addition, we propose an automated pipeline to extract high-quality initial, action, and final state frames. We evaluate VisualChef quantitatively and qualitatively on three egocentric video datasets and show its improvements over state-of-the-art methods.
>
---
#### [replaced 010] Interactive Test-Time Adaptation with Reliable Spatial-Temporal Voxels for Multi-Modal Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.06461v5](http://arxiv.org/pdf/2403.06461v5)**

> **作者:** Haozhi Cao; Yuecong Xu; Pengyu Yin; Xingyu Ji; Shenghai Yuan; Jianfei Yang; Lihua Xie
>
> **摘要:** Multi-modal test-time adaptation (MM-TTA) adapts models to an unlabeled target domain by leveraging the complementary multi-modal inputs in an online manner. While previous MM-TTA methods for 3D segmentation offer a promising solution by leveraging self-refinement per frame, they suffer from two major limitations: 1) unstable frame-wise predictions caused by temporal inconsistency, and 2) consistently incorrect predictions that violate the assumption of reliable modality guidance. To address these limitations, this work introduces a comprehensive two-fold framework. Firstly, building upon our previous work ReLiable Spatial-temporal Voxels (Latte), we propose Latte++ that better suppresses the unstable frame-wise predictions with more informative geometric correspondences. Instead of utilizing a universal sliding window, Latte++ employs multi-window aggregation to capture more reliable correspondences to better evaluate the local prediction consistency of different semantic categories. Secondly, to tackle the consistently incorrect predictions, we propose Interactive Test-Time Adaptation (ITTA), a flexible add-on to empower effortless human feedback with existing MM-TTA methods. ITTA introduces a novel human-in-the-loop approach that efficiently integrates minimal human feedback through interactive segmentation, requiring only simple point clicks and bounding box annotations. Instead of using independent interactive networks, ITTA employs a lightweight promptable branch with a momentum gradient module to capture and reuse knowledge from scarce human feedback during online inference. Extensive experiments across five MM-TTA benchmarks demonstrate that ITTA achieves consistent and notable improvements with robust performance gains for target classes of interest in challenging imbalanced scenarios, while Latte++ provides complementary benefits for temporal stability.
>
---
#### [replaced 011] UniUIR: Considering Underwater Image Restoration as An All-in-One Learner
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.12981v2](http://arxiv.org/pdf/2501.12981v2)**

> **作者:** Xu Zhang; Huan Zhang; Guoli Wang; Qian Zhang; Lefei Zhang; Bo Du
>
> **备注:** Accepted by IEEE Transactions on Image Processing. Project page at https://house-yuyu.github.io/UniUIR/
>
> **摘要:** Existing underwater image restoration (UIR) methods generally only handle color distortion or jointly address color and haze issues, but they often overlook the more complex degradations that can occur in underwater scenes. To address this limitation, we propose a Universal Underwater Image Restoration method, termed as UniUIR, considering the complex scenario of real-world underwater mixed distortions as an all-in-one manner. To decouple degradation-specific issues and explore the inter-correlations among various degradations in UIR task, we designed the Mamba Mixture-of-Experts module. This module enables each expert to identify distinct types of degradation and collaboratively extract task-specific priors while maintaining global feature representation based on linear complexity. Building upon this foundation, to enhance degradation representation and address the task conflicts that arise when handling multiple types of degradation, we introduce the spatial-frequency prior generator. This module extracts degradation prior information in both spatial and frequency domains, and adaptively selects the most appropriate task-specific prompts based on image content, thereby improving the accuracy of image restoration. Finally, to more effectively address complex, region-dependent distortions in UIR task, we incorporate depth information derived from a large-scale pre-trained depth prediction model, thereby enabling the network to perceive and leverage depth variations across different image regions to handle localized degradation. Extensive experiments demonstrate that UniUIR can produce more attractive results across qualitative and quantitative comparisons, and shows strong generalization than state-of-the-art methods.
>
---
#### [replaced 012] Uncovering Grounding IDs: How External Cues Shape Multi-Modal Binding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.24072v2](http://arxiv.org/pdf/2509.24072v2)**

> **作者:** Hosein Hasani; Amirmohammad Izadi; Fatemeh Askari; Mobin Bagherian; Sadegh Mohammadian; Mohammad Izadi; Mahdieh Soleymani Baghshah
>
> **备注:** Under review as a conference paper at ICLR 2026
>
> **摘要:** Large vision-language models (LVLMs) show strong performance across multimodal benchmarks but remain limited in structured reasoning and precise grounding. Recent work has demonstrated that adding simple visual structures, such as partitions and annotations, improves accuracy, yet the internal mechanisms underlying these gains remain unclear. We investigate this phenomenon and propose the concept of Grounding IDs, latent identifiers induced by external cues that bind objects to their designated partitions across modalities. Through representation analysis, we find that these identifiers emerge as robust within-partition alignment in embedding space and reduce the modality gap between image and text. Causal interventions further confirm that these identifiers mediate binding between objects and symbolic cues. We show that Grounding IDs strengthen attention between related components, which in turn improves cross-modal grounding and reduces hallucinations. Taken together, our results identify Grounding IDs as a key symbolic mechanism explaining how external cues enhance multimodal binding, offering both interpretability and practical improvements in robustness.
>
---
#### [replaced 013] SEE-DPO: Self Entropy Enhanced Direct Preference Optimization
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.04712v2](http://arxiv.org/pdf/2411.04712v2)**

> **作者:** Shivanshu Shekhar; Shreyas Singh; Tong Zhang
>
> **摘要:** Direct Preference Optimization (DPO) has been successfully used to align large language models (LLMs) according to human preferences, and more recently it has also been applied to improving the quality of text-to-image diffusion models. However, DPO-based methods such as SPO, Diffusion-DPO, and D3PO are highly susceptible to overfitting and reward hacking, especially when the generative model is optimized to fit out-of-distribution during prolonged training. To overcome these challenges and stabilize the training of diffusion models, we introduce a self-entropy regularization mechanism in reinforcement learning from human feedback. This enhancement improves DPO training by encouraging broader exploration and greater robustness. Our regularization technique effectively mitigates reward hacking, leading to improved stability and enhanced image quality across the latent space. Extensive experiments demonstrate that integrating human feedback with self-entropy regularization can significantly boost image diversity and specificity, achieving state-of-the-art results on key image generation metrics.
>
---
#### [replaced 014] Bias in Gender Bias Benchmarks: How Spurious Features Distort Evaluation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.07596v2](http://arxiv.org/pdf/2509.07596v2)**

> **作者:** Yusuke Hirota; Ryo Hachiuma; Boyi Li; Ximing Lu; Michael Ross Boone; Boris Ivanovic; Yejin Choi; Marco Pavone; Yu-Chiang Frank Wang; Noa Garcia; Yuta Nakashima; Chao-Han Huck Yang
>
> **备注:** ICCV 2025
>
> **摘要:** Gender bias in vision-language foundation models (VLMs) raises concerns about their safe deployment and is typically evaluated using benchmarks with gender annotations on real-world images. However, as these benchmarks often contain spurious correlations between gender and non-gender features, such as objects and backgrounds, we identify a critical oversight in gender bias evaluation: Do spurious features distort gender bias evaluation? To address this question, we systematically perturb non-gender features across four widely used benchmarks (COCO-gender, FACET, MIAP, and PHASE) and various VLMs to quantify their impact on bias evaluation. Our findings reveal that even minimal perturbations, such as masking just 10% of objects or weakly blurring backgrounds, can dramatically alter bias scores, shifting metrics by up to 175% in generative VLMs and 43% in CLIP variants. This suggests that current bias evaluations often reflect model responses to spurious features rather than gender bias, undermining their reliability. Since creating spurious feature-free benchmarks is fundamentally challenging, we recommend reporting bias metrics alongside feature-sensitivity measurements to enable a more reliable bias assessment.
>
---
#### [replaced 015] Autonomous Imagination: Closed-Loop Decomposition of Visual-to-Textual Conversion in Visual Reasoning for Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18142v4](http://arxiv.org/pdf/2411.18142v4)**

> **作者:** Jingming Liu; Yumeng Li; Boyuan Xiao; Yichang Jian; Ziang Qin; Tianjia Shao; Yao-Xiang Ding; Kun Zhou
>
> **备注:** Published in TMLR
>
> **摘要:** Under pure textual modality, Large Language Models (LLMs) have demonstrated remarkable success in complex reasoning tasks by decomposing them into simpler sub-problems. However, Multimodal Large Language Models (MLLMs) still struggle with some seemingly straightforward visual tasks, such as counting and solving jigsaw puzzles. We argue that these tasks challenge the ability of visual-to-textual conversion, where MLLMs convert visual information perceived from the input scene, to textual information for further reasoning and generating the answer. If the complexity of the visual input is beyond the perceptual capability of the MLLMs, without decomposing this conversion process, simply scaling inference-time reasoning cannot solve the task because it repeatedly encounters the same perceptual bottleneck. We propose an approach, autonomous imagination, to enable MLLMs to iteratively modify visual inputs (e.g. isolating objects, rearranging puzzle pieces) into intermediate visual states, decomposing visual-to-textual conversion into closed-loop visual modification steps. We show that, without any retraining, MLLMs can now solve tasks initially beyond their perceptual capability, highlighting that closed-loop visual modification can be an effective way of decomposing the visual reasoning task into solvable substeps. Our code and data are released at https://future-item.github.io/autoimagine-site/.
>
---
#### [replaced 016] ViP$^2$-CLIP: Visual-Perception Prompting with Unified Alignment for Zero-Shot Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17692v3](http://arxiv.org/pdf/2505.17692v3)**

> **作者:** Ziteng Yang; Jingzehua Xu; Yanshu Li; Zepeng Li; Yeqiang Wang; Xinghui Li
>
> **摘要:** Zero-shot anomaly detection (ZSAD) aims to detect anomalies without any target domain training samples, relying solely on external auxiliary data. Existing CLIP-based methods attempt to activate the model's ZSAD potential via handcrafted or static learnable prompts. The former incur high engineering costs and limited semantic coverage, whereas the latter apply identical descriptions across diverse anomaly types, thus fail to adapt to complex variations. Furthermore, since CLIP is originally pretrained on large-scale classification tasks, its anomaly segmentation quality is highly sensitive to the exact wording of class names, severely constraining prompting strategies that depend on class labels. To address these challenges, we introduce ViP$^{2}$-CLIP. The key insight of ViP$^{2}$-CLIP is a Visual-Perception Prompting (ViP-Prompt) mechanism, which fuses global and multi-scale local visual context to adaptively generate fine-grained textual prompts, eliminating manual templates and class-name priors. This design enables our model to focus on precise abnormal regions, making it particularly valuable when category labels are ambiguous or privacy-constrained. Extensive experiments on 15 industrial and medical benchmarks demonstrate that ViP$^{2}$-CLIP achieves state-of-the-art performance and robust cross-domain generalization.
>
---
#### [replaced 017] Light of Normals: Unified Feature Representation for Universal Photometric Stereo
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18882v4](http://arxiv.org/pdf/2506.18882v4)**

> **作者:** Hong Li; Houyuan Chen; Chongjie Ye; Zhaoxi Chen; Bohan Li; Shaocong Xu; Xianda Guo; Xuhui Liu; Yikai Wang; Baochang Zhang; Satoshi Ikehata; Boxin Shi; Anyi Rao; Hao Zhao
>
> **备注:** Home: https://houyuanchen111.github.io/lino.github.io Github: https://github.com/houyuanchen111/LINO_UniPS HuggingFace Demo: https://huggingface.co/spaces/houyuanchen/lino
>
> **摘要:** Universal photometric stereo (PS) is defined by two factors: it must (i) operate under arbitrary, unknown lighting conditions and (ii) avoid reliance on specific illumination models. Despite progress (e.g., SDM UniPS), two challenges remain. First, current encoders cannot guarantee that illumination and normal information are decoupled. To enforce decoupling, we introduce LINO UniPS with two key components: (i) Light Register Tokens with light alignment supervision to aggregate point, direction, and environment lights; (ii) Interleaved Attention Block featuring global cross-image attention that takes all lighting conditions together so the encoder can factor out lighting while retaining normal-related evidence. Second, high-frequency geometric details are easily lost. We address this with (i) a Wavelet-based Dual-branch Architecture and (ii) a Normal-gradient Perception Loss. These techniques yield a unified feature space in which lighting is explicitly represented by register tokens, while normal details are preserved via wavelet branch. We further introduce PS-Verse, a large-scale synthetic dataset graded by geometric complexity and lighting diversity, and adopt curriculum training from simple to complex scenes. Extensive experiments show new state-of-the-art results on public benchmarks (e.g., DiLiGenT, Luces), stronger generalization to real materials, and improved efficiency; ablations confirm that Light Register Tokens + Interleaved Attention Block drive better feature decoupling, while Wavelet-based Dual-branch Architecture + Normal-gradient Perception Loss recover finer details.
>
---
#### [replaced 018] MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.18470v2](http://arxiv.org/pdf/2503.18470v2)**

> **作者:** Zhenyu Pan; Han Liu
>
> **备注:** Working Paper
>
> **摘要:** We present MetaSpatial, the first reinforcement learning (RL)-based framework designed to enhance 3D spatial reasoning in vision-language models (VLMs), enabling real-time 3D scene generation without the need for hard-coded optimizations. MetaSpatial addresses two core challenges: (i) the lack of internalized 3D spatial reasoning in VLMs, which limits their ability to generate realistic layouts, and (ii) the inefficiency of traditional supervised fine-tuning (SFT) for layout generation tasks, as perfect ground truth annotations are unavailable. Our key innovation is a multi-turn RL-based optimization mechanism that integrates physics-aware constraints and rendered image evaluations, ensuring generated 3D layouts are coherent, physically plausible, and aesthetically consistent. Methodologically, MetaSpatial introduces an adaptive, iterative reasoning process, where the VLM refines spatial arrangements over multiple turns by analyzing rendered outputs, improving scene coherence progressively. Empirical evaluations demonstrate that MetaSpatial significantly enhances the spatial consistency and formatting stability of various scale models. Post-training, object placements are more realistic, aligned, and functionally coherent, validating the effectiveness of RL for 3D spatial reasoning in metaverse, AR/VR, digital twins, and game development applications. Our code, data, and training pipeline are publicly available at https://github.com/PzySeere/MetaSpatial.
>
---
#### [replaced 019] Exploring Representation Invariance in Finetuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07399v3](http://arxiv.org/pdf/2503.07399v3)**

> **作者:** Wenqiang Zu; Shenghao Xie; Hao Chen; Zhiqiang Chen; Liwen Hu; Yuanhao Xi; Yiming Liang; Junliang Ye; Bo Lei; Tiejun Huang; Guoqi Li; Lei Ma
>
> **摘要:** Foundation models pretrained on large-scale natural images are widely adapted to various cross-domain low-resource downstream tasks, benefiting from generalizable and transferable patterns captured by their representations. However, these representations are later found to gradually vanish during finetuning, accompanied by a degradation of model's original generalizability. In this paper, we argue that such tasks can be effectively adapted without sacrificing the benefits of pretrained representations. We approach this by introducing \textit{Representation Invariance FineTuning (RIFT)}, a regularization that maximizes the representation similarity between pretrained and finetuned models by leveraging orthogonal invariance of manifolds in a computationally efficient way. Experiments demonstrate that our method is compatible with mainstream finetuning methods, offering competitive or even enhanced performance and better preservation of the generalizability.
>
---
#### [replaced 020] Robust MRI Reconstruction by Smoothed Unrolling (SMUG)
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2312.07784v3](http://arxiv.org/pdf/2312.07784v3)**

> **作者:** Shijun Liang; Van Hoang Minh Nguyen; Jinghan Jia; Ismail Alkhouri; Sijia Liu; Saiprasad Ravishankar
>
> **摘要:** As the popularity of deep learning (DL) in the field of magnetic resonance imaging (MRI) continues to rise, recent research has indicated that DL-based MRI reconstruction models might be excessively sensitive to minor input disturbances, including worst-case additive perturbations. This sensitivity often leads to unstable, aliased images. This raises the question of how to devise DL techniques for MRI reconstruction that can be robust to train-test variations. To address this problem, we propose a novel image reconstruction framework, termed Smoothed Unrolling (SMUG), which advances a deep unrolling-based MRI reconstruction model using a randomized smoothing (RS)-based robust learning approach. RS, which improves the tolerance of a model against input noises, has been widely used in the design of adversarial defense approaches for image classification tasks. Yet, we find that the conventional design that applies RS to the entire DL-based MRI model is ineffective. In this paper, we show that SMUG and its variants address the above issue by customizing the RS process based on the unrolling architecture of a DL-based MRI reconstruction model. Compared to the vanilla RS approach, we show that SMUG improves the robustness of MRI reconstruction with respect to a diverse set of instability sources, including worst-case and random noise perturbations to input measurements, varying measurement sampling rates, and different numbers of unrolling steps. Furthermore, we theoretically analyze the robustness of our method in the presence of perturbations.
>
---
#### [replaced 021] CBVLM: Training-free Explainable Concept-based Large Vision Language Models for Medical Image Classification
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.12266v3](http://arxiv.org/pdf/2501.12266v3)**

> **作者:** Cristiano Patrício; Isabel Rio-Torto; Jaime S. Cardoso; Luís F. Teixeira; João C. Neves
>
> **备注:** Accepted for publication in Computers in Biology and Medicine
>
> **摘要:** The main challenges limiting the adoption of deep learning-based solutions in medical workflows are the availability of annotated data and the lack of interpretability of such systems. Concept Bottleneck Models (CBMs) tackle the latter by constraining the model output on a set of predefined and human-interpretable concepts. However, the increased interpretability achieved through these concept-based explanations implies a higher annotation burden. Moreover, if a new concept needs to be added, the whole system needs to be retrained. Inspired by the remarkable performance shown by Large Vision-Language Models (LVLMs) in few-shot settings, we propose a simple, yet effective, methodology, CBVLM, which tackles both of the aforementioned challenges. First, for each concept, we prompt the LVLM to answer if the concept is present in the input image. Then, we ask the LVLM to classify the image based on the previous concept predictions. Moreover, in both stages, we incorporate a retrieval module responsible for selecting the best examples for in-context learning. By grounding the final diagnosis on the predicted concepts, we ensure explainability, and by leveraging the few-shot capabilities of LVLMs, we drastically lower the annotation cost. We validate our approach with extensive experiments across four medical datasets and twelve LVLMs (both generic and medical) and show that CBVLM consistently outperforms CBMs and task-specific supervised methods without requiring any training and using just a few annotated examples. More information on our project page: https://cristianopatricio.github.io/CBVLM/.
>
---
#### [replaced 022] What Drives Compositional Generalization in Visual Generative Models?
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.03075v2](http://arxiv.org/pdf/2510.03075v2)**

> **作者:** Karim Farid; Rajat Sahay; Yumna Ali Alnaggar; Simon Schrodi; Volker Fischer; Cordelia Schmid; Thomas Brox
>
> **摘要:** Compositional generalization, the ability to generate novel combinations of known concepts, is a key ingredient for visual generative models. Yet, not all mechanisms that enable or inhibit it are fully understood. In this work, we conduct a systematic study of how various design choices influence compositional generalization in image and video generation in a positive or negative way. Through controlled experiments, we identify two key factors: (i) whether the training objective operates on a discrete or continuous distribution, and (ii) to what extent conditioning provides information about the constituent concepts during training. Building on these insights, we show that relaxing the MaskGIT discrete loss with an auxiliary continuous JEPA-based objective can improve compositional performance in discrete models like MaskGIT.
>
---
#### [replaced 023] Conformal Prediction for Long-Tailed Classification
- **分类: stat.ML; cs.CV; cs.LG; stat.ME**

- **链接: [http://arxiv.org/pdf/2507.06867v2](http://arxiv.org/pdf/2507.06867v2)**

> **作者:** Tiffany Ding; Jean-Baptiste Fermanian; Joseph Salmon
>
> **摘要:** Many real-world classification problems, such as plant identification, have extremely long-tailed class distributions. In order for prediction sets to be useful in such settings, they should (i) provide good class-conditional coverage, ensuring that rare classes are not systematically omitted from the prediction sets, and (ii) be a reasonable size, allowing users to easily verify candidate labels. Unfortunately, existing conformal prediction methods, when applied to the long-tailed setting, force practitioners to make a binary choice between small sets with poor class-conditional coverage or sets with very good class-conditional coverage but that are extremely large. We propose methods with guaranteed marginal coverage that smoothly trade off between set size and class-conditional coverage. First, we introduce a new conformal score function called prevalence-adjusted softmax that targets macro-coverage, a relaxed notion of class-conditional coverage. Second, we propose a new procedure that interpolates between marginal and class-conditional conformal prediction by linearly interpolating their conformal score thresholds. We demonstrate our methods on Pl@ntNet-300K and iNaturalist-2018, two long-tailed image datasets with 1,081 and 8,142 classes, respectively.
>
---
#### [replaced 024] Generative Human Geometry Distribution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.01448v3](http://arxiv.org/pdf/2503.01448v3)**

> **作者:** Xiangjun Tang; Biao Zhang; Peter Wonka
>
> **摘要:** Realistic human geometry generation is an important yet challenging task, requiring both the preservation of fine clothing details and the accurate modeling of clothing-body interactions. To tackle this challenge, we build upon Geometry distributions, a recently proposed representation that can model a single human geometry with high fidelity using a flow matching model. However, extending a single-geometry distribution to a dataset is non-trivial and inefficient for large-scale learning. To address this, we propose a new geometry distribution model by two key techniques: (1) encoding distributions as 2D feature maps rather than network parameters, and (2) using SMPL models as the domain instead of Gaussian and refining the associated flow velocity field. We then design a generative framework adopting a two staged training paradigm analogous to state-of-the-art image and 3D generative models. In the first stage, we compress geometry distributions into a latent space using a diffusion flow model; the second stage trains another flow model on this latent space. We validate our approach on two key tasks: pose-conditioned random avatar generation and avatar-consistent novel pose synthesis. Experimental results demonstrate that our method outperforms existing state-of-the-art methods, achieving a 57% improvement in geometry quality.
>
---
#### [replaced 025] Graph Algorithm Unrolling with Douglas-Rachford Iterations for Image Interpolation with Guaranteed Initialization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11926v2](http://arxiv.org/pdf/2509.11926v2)**

> **作者:** Xue Zhang; Bingshuo Hu; Gene Cheung
>
> **摘要:** Conventional deep neural nets (DNNs) initialize network parameters at random and then optimize each one via stochastic gradient descent (SGD), resulting in substantial risk of poor-performing local minima.Focusing on the image interpolation problem and leveraging a recent theorem that maps a (pseudo-)linear interpolator {\Theta} to a directed graph filter that is a solution to a MAP problem regularized with a graph shift variation (GSV) prior, we first initialize a directed graph adjacency matrix A based on a known interpolator {\Theta}, establishing a baseline performance.Then, towards further gain, we learn perturbation matrices P and P(2) from data to augment A, whose restoration effects are implemented via Douglas-Rachford (DR) iterations, which we unroll into a lightweight interpretable neural net.Experimental results demonstrate state-of-the-art image interpolation results, while drastically reducing network parameters.
>
---
#### [replaced 026] Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.13693v3](http://arxiv.org/pdf/2502.13693v3)**

> **作者:** Omid Nejati Manzari; Hojat Asgariandehkordi; Taha Koleilat; Yiming Xiao; Hassan Rivaz
>
> **摘要:** Convolutional networks, transformers, hybrid models, and Mamba-based architectures have demonstrated strong performance across various medical image classification tasks. However, these methods were primarily designed to classify clean images using labeled data. In contrast, real-world clinical data often involve image corruptions that are unique to multi-center studies and stem from variations in imaging equipment across manufacturers. In this paper, we introduce the Medical Vision Transformer (MedViTV2), a novel architecture incorporating Kolmogorov-Arnold Network (KAN) layers into the transformer architecture for the first time, aiming for generalized medical image classification. We have developed an efficient KAN block to reduce computational load while enhancing the accuracy of the original MedViT. Additionally, to counteract the fragility of our MedViT when scaled up, we propose an enhanced Dilated Neighborhood Attention (DiNA), an adaptation of the efficient fused dot-product attention kernel capable of capturing global context and expanding receptive fields to scale the model effectively and addressing feature collapse issues. Moreover, a hierarchical hybrid strategy is introduced to stack our Local Feature Perception and Global Feature Perception blocks in an efficient manner, which balances local and global feature perceptions to boost performance. Extensive experiments on 17 medical image classification datasets and 12 corrupted medical image datasets demonstrate that MedViTV2 achieved state-of-the-art results in 27 out of 29 experiments with reduced computational complexity. MedViTV2 is 44\% more computationally efficient than the previous version and significantly enhances accuracy, achieving improvements of 4.6\% on MedMNIST, 5.8\% on NonMNIST, and 13.4\% on the MedMNIST-C benchmark.
>
---
#### [replaced 027] Neural Brain: A Neuroscience-inspired Framework for Embodied Agents
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07634v3](http://arxiv.org/pdf/2505.07634v3)**

> **作者:** Jian Liu; Xiongtao Shi; Thai Duy Nguyen; Haitian Zhang; Tianxiang Zhang; Wei Sun; Yanjie Li; Athanasios V. Vasilakos; Giovanni Iacca; Arshad Ali Khan; Arvind Kumar; Jae Won Cho; Ajmal Mian; Lihua Xie; Erik Cambria; Lin Wang
>
> **备注:** 51 pages, 17 figures, 9 tables
>
> **摘要:** The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios.
>
---
#### [replaced 028] Grounding-IQA: Grounding Multimodal Language Model for Image Quality Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17237v3](http://arxiv.org/pdf/2411.17237v3)**

> **作者:** Zheng Chen; Xun Zhang; Wenbo Li; Renjing Pei; Fenglong Song; Xiongkuo Min; Xiaohong Liu; Xin Yuan; Yong Guo; Yulun Zhang
>
> **备注:** Code is available at: https://github.com/zhengchen1999/Grounding-IQA
>
> **摘要:** The development of multimodal large language models (MLLMs) enables the evaluation of image quality through natural language descriptions. This advancement allows for more detailed assessments. However, these MLLM-based IQA methods primarily rely on general contextual descriptions, sometimes limiting fine-grained quality assessment. To address this limitation, we introduce a new image quality assessment (IQA) task paradigm, **grounding-IQA**. This paradigm integrates multimodal referring and grounding with IQA to realize more fine-grained quality perception, thereby extending existing IQA. Specifically, grounding-IQA comprises two subtasks: grounding-IQA-description (GIQA-DES) and visual question answering (GIQA-VQA). GIQA-DES involves detailed descriptions with precise locations (e.g., bounding boxes), while GIQA-VQA focuses on quality QA for local regions. To realize grounding-IQA, we construct a corresponding dataset, GIQA-160K, through our proposed automated annotation pipeline. Furthermore, we develop a well-designed benchmark, GIQA-Bench. The benchmark evaluates the grounding-IQA performance from three perspectives: description quality, VQA accuracy, and grounding precision. Experiments demonstrate that our proposed method facilitates the more fine-grained IQA application. Code: https://github.com/zhengchen1999/Grounding-IQA.
>
---
#### [replaced 029] TACO: Enhancing Multimodal In-context Learning via Task Mapping-Guided Sequence Configuration
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17098v3](http://arxiv.org/pdf/2505.17098v3)**

> **作者:** Yanshu Li; Jianjiang Yang; Tian Yun; Pinyuan Feng; Jinfa Huang; Ruixiang Tang
>
> **备注:** EMNLP2025 Main, 28 pages, 11 figures, 19 tables
>
> **摘要:** Multimodal in-context learning (ICL) has emerged as a key mechanism for harnessing the capabilities of large vision-language models (LVLMs). However, its effectiveness remains highly sensitive to the quality of input ICL sequences, particularly for tasks involving complex reasoning or open-ended generation. A major limitation is our limited understanding of how LVLMs actually exploit these sequences during inference. To bridge this gap, we systematically interpret multimodal ICL through the lens of task mapping, which reveals how local and global relationships within and among demonstrations guide model reasoning. Building on this insight, we present TACO, a lightweight transformer-based model equipped with task-aware attention that dynamically configures ICL sequences. By injecting task-mapping signals into the autoregressive decoding process, TACO creates a bidirectional synergy between sequence construction and task reasoning. Experiments on five LVLMs and nine datasets demonstrate that TACO consistently surpasses baselines across diverse ICL tasks. These results position task mapping as a novel and valuable perspective for interpreting and improving multimodal ICL.
>
---
#### [replaced 030] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16965v3](http://arxiv.org/pdf/2503.16965v3)**

> **作者:** Zhe Hu; Jing Li; Zhongzhu Pu; Hou Pong Chan; Yu Yin
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Vision Language Models exhibit impressive performance for various tasks, yet they often lack the sophisticated situational reasoning required for complex decision-making. This paper shows that VLMs can achieve surprisingly strong decision-making performance when visual scenes are replaced by textual descriptions, suggesting foundational reasoning can be effectively learned from language. Motivated by this insight, we propose Praxis-VLM, a reasoning VLM for vision-grounded decision-making. Praxis-VLM employs the GRPO algorithm on textual scenarios to instill robust reasoning capabilities, where models learn to evaluate actions and their consequences. These reasoning skills, acquired purely from text, successfully transfer to multimodal inference with visual inputs, significantly reducing reliance on scarce paired image-text training data. Experiments across diverse decision-making benchmarks demonstrate that Praxis-VLM substantially outperforms standard supervised fine-tuning, exhibiting superior performance and generalizability. Further analysis confirms that our models engage in explicit and effective reasoning, underpinning their enhanced performance and adaptability.
>
---
#### [replaced 031] WetCat: Enabling Automated Skill Assessment in Wet-Lab Cataract Surgery Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.08896v4](http://arxiv.org/pdf/2506.08896v4)**

> **作者:** Negin Ghamsarian; Raphael Sznitman; Klaus Schoeffmann; Jens Kowal
>
> **备注:** 7 pages, 7 figures, Accepted at ACMMM25
>
> **摘要:** To meet the growing demand for systematic surgical training, wet-lab environments have become indispensable platforms for hands-on practice in ophthalmology. Yet, traditional wet-lab training depends heavily on manual performance evaluations, which are labor-intensive, time-consuming, and often subject to variability. Recent advances in computer vision offer promising avenues for automated skill assessment, enhancing both the efficiency and objectivity of surgical education. Despite notable progress in ophthalmic surgical datasets, existing resources predominantly focus on real surgeries or isolated tasks, falling short of supporting comprehensive skill evaluation in controlled wet-lab settings. To address these limitations, we introduce WetCat, the first dataset of wet-lab cataract surgery videos specifically curated for automated skill assessment. WetCat comprises high-resolution recordings of surgeries performed by trainees on artificial eyes, featuring comprehensive phase annotations and semantic segmentations of key anatomical structures. These annotations are meticulously designed to facilitate skill assessment during the critical capsulorhexis and phacoemulsification phases, adhering to standardized surgical skill assessment frameworks. By focusing on these essential phases, WetCat enables the development of interpretable, AI-driven evaluation tools aligned with established clinical metrics. This dataset lays a strong foundation for advancing objective, scalable surgical education and sets a new benchmark for automated workflow analysis and skill assessment in ophthalmology training. The dataset and annotations are publicly available in Synapse.
>
---
#### [replaced 032] Fast-RF-Shimming: Accelerate RF Shimming in 7T MRI using Deep Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.12157v2](http://arxiv.org/pdf/2501.12157v2)**

> **作者:** Zhengyi Lu; Hao Liang; Ming Lu; Xiao Wang; Xinqiang Yan; Yuankai Huo
>
> **摘要:** Ultrahigh field (UHF) Magnetic Resonance Imaging (MRI) offers an elevated signal-to-noise ratio (SNR), enabling exceptionally high spatial resolution that benefits both clinical diagnostics and advanced research. However, the jump to higher fields introduces complications, particularly transmit radiofrequency (RF) field ($B_{1}^{+}$) inhomogeneities, manifesting as uneven flip angles and image intensity irregularities. These artifacts can degrade image quality and impede broader clinical adoption. Traditional RF shimming methods, such as Magnitude Least Squares (MLS) optimization, effectively mitigate $B_{1}^{+}$ inhomogeneity, but remain time-consuming. Recent machine learning approaches, including RF Shim Prediction by Iteratively Projected Ridge Regression and other deep learning architectures, suggest alternative pathways. Although these approaches show promise, challenges such as extensive training periods, limited network complexity, and practical data requirements persist. In this paper, we introduce a holistic learning-based framework called Fast-RF-Shimming, which achieves a 5000x speed-up compared to the traditional MLS method. In the initial phase, we employ random-initialized Adaptive Moment Estimation (Adam) to derive the desired reference shimming weights from multi-channel $B_{1}^{+}$ fields. Next, we train a Residual Network (ResNet) to map $B_{1}^{+}$ fields directly to the ultimate RF shimming outputs, incorporating the confidence parameter into its loss function. Finally, we design Non-uniformity Field Detector (NFD), an optional post-processing step, to ensure the extreme non-uniform outcomes are identified. Comparative evaluations with standard MLS optimization underscore notable gains in both processing speed and predictive accuracy, which indicates that our technique shows a promising solution for addressing persistent inhomogeneity challenges.
>
---
#### [replaced 033] GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13731v2](http://arxiv.org/pdf/2505.13731v2)**

> **作者:** Pengyue Jia; Seongheon Park; Song Gao; Xiangyu Zhao; Yixuan Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Worldwide image geolocalization-the task of predicting GPS coordinates from images taken anywhere on Earth-poses a fundamental challenge due to the vast diversity in visual content across regions. While recent approaches adopt a two-stage pipeline of retrieving candidates and selecting the best match, they typically rely on simplistic similarity heuristics and point-wise supervision, failing to model spatial relationships among candidates. In this paper, we propose GeoRanker, a distance-aware ranking framework that leverages large vision-language models to jointly encode query-candidate interactions and predict geographic proximity. In addition, we introduce a multi-order distance loss that ranks both absolute and relative distances, enabling the model to reason over structured spatial relationships. To support this, we curate GeoRanking, the first dataset explicitly designed for geographic ranking tasks with multimodal candidate information. GeoRanker achieves state-of-the-art results on two well-established benchmarks (IM2GPS3K and YFCC4K), significantly outperforming current best methods.
>
---
#### [replaced 034] Periodontal Bone Loss Analysis via Keypoint Detection With Heuristic Post-Processing
- **分类: q-bio.TO; cs.AI; cs.CV; I.2.1; I.2.10; J.3**

- **链接: [http://arxiv.org/pdf/2503.13477v2](http://arxiv.org/pdf/2503.13477v2)**

> **作者:** Ryan Banks; Vishal Thengane; María Eugenia Guerrero; Nelly Maria García-Madueño; Yunpeng Li; Hongying Tang; Akhilanand Chaurasia
>
> **备注:** 18 pages, 7 tables, 9 figures, 1 equation, journal paper submitted to Computers in Biology and Medicine
>
> **摘要:** This study proposes a deep learning framework and annotation methodology for the automatic detection of periodontal bone loss landmarks, associated conditions, and staging. 192 periapical radiographs were collected and annotated with a stage agnostic methodology, labelling clinically relevant landmarks regardless of disease presence or extent. We propose a heuristic post-processing module that aligns predicted keypoints to tooth boundaries using an auxiliary instance segmentation model. An evaluation metric, Percentage of Relative Correct Keypoints (PRCK), is proposed to capture keypoint performance in dental imaging domains. Four donor pose estimation models were adapted with fine-tuning for our keypoint problem. Post-processing improved fine-grained localisation, raising average PRCK^{0.05} by +0.028, but reduced coarse performance for PRCK^{0.25} by -0.0523 and PRCK^{0.5} by -0.0345. Orientation estimation shows excellent performance for auxiliary segmentation when filtered with either stage 1 object detection model. Periodontal staging was detected sufficiently, with the best mesial and distal Dice scores of 0.508 and 0.489, while furcation involvement and widened periodontal ligament space tasks remained challenging due to scarce positive samples. Scalability is implied with similar validation and external set performance. The annotation methodology enables stage agnostic training with balanced representation across disease severities for some detection tasks. The PRCK metric provides a domain-specific alternative to generic pose metrics, while the heuristic post-processing module consistently corrected implausible predictions with occasional catastrophic failures. The proposed framework demonstrates the feasibility of clinically interpretable periodontal bone loss assessment, with potential to reduce diagnostic variability and clinician workload.
>
---
#### [replaced 035] The Telephone Game: Evaluating Semantic Drift in Unified Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.04438v2](http://arxiv.org/pdf/2509.04438v2)**

> **作者:** Sabbir Mollah; Rohit Gupta; Sirnam Swetha; Qingyang Liu; Ahnaf Munir; Mubarak Shah
>
> **摘要:** Employing a single, unified model (UM) for both visual understanding (image-to-text: I2T) and visual generation (text-to-image: T2I) has opened a new direction in Visual Language Model (VLM) research. While UMs can also support broader unimodal tasks (e.g., text-to-text, image-to-image), we focus on the core cross-modal pair T2I and I2T. Existing evaluation benchmarks consider these capabilities in isolation: FID and GenEval for T2I, and benchmarks such as MME, MMBench for I2T. These isolated single-pass metrics do not reveal cross-consistency: whether a model that "understands" a concept can also "render" it, nor whether semantic meaning is preserved when cycling between image and text modalities. To address this, we introduce the Semantic Drift Protocol (SDP) for Unified Models, a cyclic evaluation protocol that alternates I2T and T2I over multiple generations to quantify semantic drift. We propose two metrics: (i) Mean Cumulative Drift (MCD), an embedding-based measure of overall semantic drift; and (ii) Multi-Generation GenEval (MGG), an object-level compliance score extending GenEval. To assess generalization beyond COCO dataset, which is widely used in training; we create a new benchmark Nocaps+Docci400, sampled from NoCaps and DOCCI and evaluated on seven recent models. SDP reveals substantial variation in cross-modal stability: some models like BAGEL maintain semantic meaning over many alternations, whereas others like VILA-U drift quickly despite strong single-pass scores. Our results highlight SDP as a necessary complement to standard I2T and T2I evaluations. Code is available at https://github.com/mollahsabbir/Semantic-Drift-in-Unified-Models
>
---
#### [replaced 036] PAL-UI: Planning with Active Look-back for Vision-Based GUI Agents
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.00413v2](http://arxiv.org/pdf/2510.00413v2)**

> **作者:** Zikang Liu; Junyi Li; Wayne Xin Zhao; Dawei Gao; Yaliang Li; Ji-rong Wen
>
> **备注:** Under Review
>
> **摘要:** Graphical User Interface (GUI) agents powered by Multimodal Large Language Models (MLLMs) promise human-like interaction with software applications, yet long-horizon tasks remain challenging due to memory limitations. Existing approaches either truncate history or rely on simple textual summaries, which risk losing critical information when past visual details become necessary for future decisions. In this paper, we propose \textbf{PAL-UI} (\textbf{P}lanning with \textbf{A}ctive \textbf{L}ook-back), a novel framework that enables GUI agents to adaptively retrieve past observations when required. PAL-UI combines a dual-level summarization agent, capturing both observation-level cues and action-level outcomes, with a dedicated retrieval tool that allows the agent to recall specific historical screenshots during planning. We curate a step-level instruction dataset of 8.6K samples from mobile GUI navigation trajectories and train \textbf{PAL-UI-3B} and \textbf{PAL-UI-7B} models based on Qwen2.5-VL. Extensive experiments demonstrate that PAL-UI significantly outperforms baseline models and prior methods in mobile GUI navigation tasks, even under data-efficient settings. Moreover, PAL-UI exhibits strong cross-domain generalization, achieving notable improvements in web navigation without additional training. Our work highlights the potential of active memory retrieval for long-horizon planning capabilities of vision-based GUI agents.
>
---
#### [replaced 037] MambaMoE: Mixture-of-Spectral-Spatial-Experts State Space Model for Hyperspectral Image Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20509v2](http://arxiv.org/pdf/2504.20509v2)**

> **作者:** Yichu Xu; Di Wang; Hongzan Jiao; Lefei Zhang; Liangpei Zhang
>
> **备注:** Accepted by Information Fusion
>
> **摘要:** Mamba-based models have recently demonstrated significant potential in hyperspectral image (HSI) classification, primarily due to their ability to perform contextual modeling with linear computational complexity. However, existing Mamba-based approaches often overlook the directional modeling heterogeneity across different land-cover types, leading to limited classification performance. To address these limitations, we propose MambaMoE, a novel spectral-spatial Mixture-of-Experts (MoE) framework, which represents the first MoE-based approach in the HSI classification domain. Specifically, we design a Mixture of Mamba Expert Block (MoMEB) that performs adaptive spectral-spatial feature modeling via a sparse expert activation mechanism. Additionally, we introduce an uncertainty-guided corrective learning (UGCL) strategy that encourages the model to focus on complex regions prone to prediction ambiguity. This strategy dynamically samples supervision signals from regions with high predictive uncertainty, guiding the model to adaptively refine feature representations and thereby enhancing its focus on challenging areas. Extensive experiments conducted on multiple public HSI benchmark datasets show that MambaMoE achieves state-of-the-art performance in both classification accuracy and computational efficiency compared to existing advanced methods, particularly Mamba-based ones. The code will be available online at https://github.com/YichuXu/MambaMoE.
>
---
#### [replaced 038] A Tale of Two Experts: Cooperative Learning for Source-Free Unsupervised Domain Adaptation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.22229v2](http://arxiv.org/pdf/2509.22229v2)**

> **作者:** Jiaping Yu; Muli Yang; Jiapeng Ji; Jiexi Yan; Cheng Deng
>
> **摘要:** Source-Free Unsupervised Domain Adaptation (SFUDA) addresses the realistic challenge of adapting a source-trained model to a target domain without access to the source data, driven by concerns over privacy and cost. Existing SFUDA methods either exploit only the source model's predictions or fine-tune large multimodal models, yet both neglect complementary insights and the latent structure of target data. In this paper, we propose the Experts Cooperative Learning (EXCL). EXCL contains the Dual Experts framework and Retrieval-Augmentation-Interaction optimization pipeline. The Dual Experts framework places a frozen source-domain model (augmented with Conv-Adapter) and a pretrained vision-language model (with a trainable text prompt) on equal footing to mine consensus knowledge from unlabeled target samples. To effectively train these plug-in modules under purely unsupervised conditions, we introduce Retrieval-Augmented-Interaction(RAIN), a three-stage pipeline that (1) collaboratively retrieves pseudo-source and complex target samples, (2) separately fine-tunes each expert on its respective sample set, and (3) enforces learning object consistency via a shared learning result. Extensive experiments on four benchmark datasets demonstrate that our approach matches state-of-the-art performance.
>
---
#### [replaced 039] AutoPrune: Each Complexity Deserves a Pruning Policy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23931v2](http://arxiv.org/pdf/2509.23931v2)**

> **作者:** Hanshi Wang; Yuhao Xu; Zekun Xu; Jin Gao; Yufan Liu; Weiming Hu; Ke Wang; Zhipeng Zhang
>
> **备注:** 13 pages, 2 figures
>
> **摘要:** The established redundancy in visual tokens within large vision-language models allows pruning to effectively reduce their substantial computational demands. Previous methods typically employ heuristic layer-specific pruning strategies where, although the number of tokens removed may differ across decoder layers, the overall pruning schedule is fixed and applied uniformly to all input samples and tasks, failing to align token elimination with the model's holistic reasoning trajectory. Cognitive science indicates that human visual processing often begins with broad exploration to accumulate evidence before narrowing focus as the target becomes distinct. Our experiments reveal an analogous pattern in these models. This observation suggests that neither a fixed pruning schedule nor a heuristic layer-wise strategy can optimally accommodate the diverse complexities inherent in different inputs. To overcome this limitation, we introduce Complexity-Adaptive Pruning (AutoPrune), a training-free, plug-and-play framework that tailors pruning policies to varying sample and task complexities. Specifically, AutoPrune quantifies the mutual information between visual and textual tokens, then projects this signal to a budget-constrained logistic retention curve. Each such logistic curve, defined by its unique shape, corresponds to the specific complexity of different tasks and can guarantee adherence to predefined computational constraints. We evaluate AutoPrune on standard vision-language tasks and on Vision-Language-Action models for autonomous driving. Notably, when applied to LLaVA-1.5-7B, our method prunes 89% of visual tokens and reduces inference FLOPs by 76.8% while retaining 96.7% of the original accuracy averaged over all tasks. This corresponds to a 9.1% improvement over the recent work PDrop, demonstrating the effectiveness. Code is available at https://github.com/AutoLab-SAI-SJTU/AutoPrune.
>
---
#### [replaced 040] DecompDreamer: A Composition-Aware Curriculum for Structured 3D Asset Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11981v2](http://arxiv.org/pdf/2503.11981v2)**

> **作者:** Utkarsh Nath; Rajeev Goel; Rahul Khurana; Kyle Min; Mark Ollila; Pavan Turaga; Varun Jampani; Tejaswi Gowda
>
> **摘要:** Current text-to-3D methods excel at generating single objects but falter on compositional prompts. We argue this failure is fundamental to their optimization schedules, as simultaneous or iterative heuristics predictably collapse under a combinatorial explosion of conflicting gradients, leading to entangled geometry or catastrophic divergence. In this paper, we reframe the core challenge of compositional generation as one of optimization scheduling. We introduce DecompDreamer, a framework built on a novel staged optimization strategy that functions as an implicit curriculum. Our method first establishes a coherent structural scaffold by prioritizing inter-object relationships before shifting to the high-fidelity refinement of individual components. This temporal decoupling of competing objectives provides a robust solution to gradient conflict. Qualitative and quantitative evaluations on diverse compositional prompts demonstrate that DecompDreamer outperforms state-of-the-art methods in fidelity, disentanglement, and spatial coherence.
>
---
#### [replaced 041] A Comparative Benchmark of Real-time Detectors for Blueberry Detection towards Precision Orchard Management
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20580v2](http://arxiv.org/pdf/2509.20580v2)**

> **作者:** Xinyang Mu; Yuzhen Lu; Boyang Deng
>
> **备注:** 19 pages, 6 figures, 4 tables. Abstract abridged due to arXiv's 1920 character limit
>
> **摘要:** Blueberry detection in natural environments remains challenging due to variable lighting, occlusions, and motion blur due to environmental factors and imaging devices. Deep learning-based object detectors promise to address these challenges, but they demand a large-scale, diverse dataset that captures the real-world complexities. Moreover, deploying these models in practical scenarios often requires the right accuracy/speed/memory trade-off in model selection. This study presents a novel comparative benchmark analysis of advanced real-time object detectors, including YOLO (You Only Look Once) (v8-v12) and RT-DETR (Real-Time Detection Transformers) (v1-v2) families, consisting of 36 model variants, evaluated on a newly curated dataset for blueberry detection. This dataset comprises 661 canopy images collected with smartphones during the 2022-2023 seasons, consisting of 85,879 labelled instances (including 36,256 ripe and 49,623 unripe blueberries) across a wide range of lighting conditions, occlusions, and fruit maturity stages. Among the YOLO models, YOLOv12m achieved the best accuracy with a mAP@50 of 93.3%, while RT-DETRv2-X obtained a mAP@50 of 93.6%, the highest among all the RT-DETR variants. The inference time varied with the model scale and complexity, and the mid-sized models appeared to offer a good accuracy-speed balance. To further enhance detection performance, all the models were fine-tuned using Unbiased Mean Teacher-based semi-supervised learning (SSL) on a separate set of 1,035 unlabeled images acquired by a ground-based machine vision platform in 2024. This resulted in accuracy gains ranging from -1.4% to 2.9%, with RT-DETR-v2-X achieving the best mAP@50 of 94.8%. More in-depth research into SSL is needed to better leverage cross-domain unlabeled data. Both the dataset and software programs of this study are made publicly available to support further research.
>
---
#### [replaced 042] QDFlow: A Python package for physics simulations of quantum dot devices
- **分类: cond-mat.mes-hall; cs.CV; cs.LG; quant-ph**

- **链接: [http://arxiv.org/pdf/2509.13298v2](http://arxiv.org/pdf/2509.13298v2)**

> **作者:** Donovan L. Buterakos; Sandesh S. Kalantre; Joshua Ziegler; Jacob M Taylor; Justyna P. Zwolak
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Recent advances in machine learning (ML) have accelerated progress in calibrating and operating quantum dot (QD) devices. However, most ML approaches rely on access to large, representative datasets designed to capture the full spectrum of data quality encountered in practice, with both high- and low-quality data for training, benchmarking, and validation, with labels capturing key features of the device state. Collating such datasets experimentally is challenging due to limited data availability, slow measurement bandwidths, and the labor-intensive nature of labeling. QDFlow is an open-source physics simulator for multi-QD arrays that generates realistic synthetic data with ground-truth labels. QDFlow combines a self-consistent Thomas-Fermi solver, a dynamic capacitance model, and flexible noise modules to simulate charge stability diagrams and ray-based data closely resembling experiments. With an extensive set of parameters that can be varied and customizable noise models, QDFlow supports the creation of large, diverse datasets for ML development, benchmarking, and quantum device research.
>
---
#### [replaced 043] Segmenting Bi-Atrial Structures Using ResNext Based Framework
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.02892v3](http://arxiv.org/pdf/2503.02892v3)**

> **作者:** Malitha Gunawardhana; Mark L Trew; Gregory B Sands; Jichao Zhao
>
> **备注:** Accepted at STACOM workshop (MICCAI 2025)
>
> **摘要:** Atrial Fibrillation (AF), the most common sustained cardiac arrhythmia worldwide, increasingly requires accurate bi-atrial structural assessment to guide ablation strategies, particularly in persistent AF. Late gadolinium-enhanced magnetic resonance imaging (LGE-MRI) enables visualisation of atrial fibrosis, but precise manual segmentation remains time-consuming, operator-dependent, and prone to variability. We propose TASSNet, a novel two-stage deep learning framework for fully automated segmentation of both left atrium (LA) and right atrium (RA), including atrial walls and cavities, from 3D LGE-MRI. TASSNet introduces two main innovations: (i) a ResNeXt-based encoder to enhance feature extraction from limited medical datasets, and (ii) a cyclical learning rate schedule to address convergence instability in highly imbalanced, small-batch 3D segmentation tasks. We evaluated our method on two datasets, one of which was completely out-of-distribution, without any additional training. In both cases, TASSNet successfully segmented atrial structures with high accuracy. These results highlight TASSNet's potential for robust and reproducible bi-atrial segmentation, enabling advanced fibrosis quantification and personalised ablation planning in clinical AF management.
>
---
#### [replaced 044] Erased, But Not Forgotten: Erased Rectified Flow Transformers Still Remain Unsafe Under Concept Attack
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.00635v2](http://arxiv.org/pdf/2510.00635v2)**

> **作者:** Nanxiang Jiang; Zhaoxin Fan; Enhan Kang; Daiheng Gao; Yun Zhou; Yanxia Chang; Zheng Zhu; Yeying Jin; Wenjun Wu
>
> **摘要:** Recent advances in text-to-image (T2I) diffusion models have enabled impressive generative capabilities, but they also raise significant safety concerns due to the potential to produce harmful or undesirable content. While concept erasure has been explored as a mitigation strategy, most existing approaches and corresponding attack evaluations are tailored to Stable Diffusion (SD) and exhibit limited effectiveness when transferred to next-generation rectified flow transformers such as Flux. In this work, we present ReFlux, the first concept attack method specifically designed to assess the robustness of concept erasure in the latest rectified flow-based T2I framework. Our approach is motivated by the observation that existing concept erasure techniques, when applied to Flux, fundamentally rely on a phenomenon known as attention localization. Building on this insight, we propose a simple yet effective attack strategy that specifically targets this property. At its core, a reverse-attention optimization strategy is introduced to effectively reactivate suppressed signals while stabilizing attention. This is further reinforced by a velocity-guided dynamic that enhances the robustness of concept reactivation by steering the flow matching process, and a consistency-preserving objective that maintains the global layout and preserves unrelated content. Extensive experiments consistently demonstrate the effectiveness and efficiency of the proposed attack method, establishing a reliable benchmark for evaluating the robustness of concept erasure strategies in rectified flow transformers.
>
---
#### [replaced 045] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.06901v2](http://arxiv.org/pdf/2503.06901v2)**

> **作者:** Chikai Shang; Mengke Li; Yiqun Zhang; Zhen Chen; Jinlin Wu; Fangqing Gu; Yang Lu; Yiu-ming Cheung
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Visual prompt tuning (VPT), i.e., fine-tuning some lightweight prompt tokens, provides an efficient and effective approach for adapting pre-trained models to various downstream tasks. However, most prior art indiscriminately uses a fixed prompt distribution across different tasks, neglecting the importance of each block varying depending on the task. In this paper, we introduce adaptive distribution optimization (ADO) by tackling two key questions: (1) How to appropriately and formally define ADO, and (2) How to design an adaptive distribution strategy guided by this definition? Through empirical analysis, we first confirm that properly adjusting the distribution significantly improves VPT performance, and further uncover a key insight that a nested relationship exists between ADO and VPT. Based on these findings, we propose a new VPT framework, termed PRO-VPT (iterative Prompt RelOcation-based VPT), which adaptively adjusts the distribution built upon a nested optimization formulation. Specifically, we develop a prompt relocation strategy derived from this formulation, comprising two steps: pruning idle prompts from prompt-saturated blocks, followed by allocating these prompts to the most prompt-needed blocks. By iteratively performing prompt relocation and VPT, our proposal can adaptively learn the optimal prompt distribution in a nested optimization-based manner, thereby unlocking the full potential of VPT. Extensive experiments demonstrate that our proposal significantly outperforms advanced VPT methods, e.g., PRO-VPT surpasses VPT by 1.6 pp and 2.0 pp average accuracy, leading prompt-based methods to state-of-the-art performance on VTAB-1k and FGVC benchmarks. The code is available at https://github.com/ckshang/PRO-VPT.
>
---
#### [replaced 046] ReMoMask: Retrieval-Augmented Masked Motion Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.02605v2](http://arxiv.org/pdf/2508.02605v2)**

> **作者:** Zhengdao Li; Siheng Wang; Zeyu Zhang; Hao Tang
>
> **摘要:** Text-to-Motion (T2M) generation aims to synthesize realistic and semantically aligned human motion sequences from natural language descriptions. However, current approaches face dual challenges: Generative models (e.g., diffusion models) suffer from limited diversity, error accumulation, and physical implausibility, while Retrieval-Augmented Generation (RAG) methods exhibit diffusion inertia, partial-mode collapse, and asynchronous artifacts. To address these limitations, we propose ReMoMask, a unified framework integrating three key innovations: 1) A Bidirectional Momentum Text-Motion Model decouples negative sample scale from batch size via momentum queues, substantially improving cross-modal retrieval precision; 2) A Semantic Spatio-temporal Attention mechanism enforces biomechanical constraints during part-level fusion to eliminate asynchronous artifacts; 3) RAG-Classier-Free Guidance incorporates minor unconditional generation to enhance generalization. Built upon MoMask's RVQ-VAE, ReMoMask efficiently generates temporally coherent motions in minimal steps. Extensive experiments on standard benchmarks demonstrate the state-of-the-art performance of ReMoMask, achieving a 3.88% and 10.97% improvement in FID scores on HumanML3D and KIT-ML, respectively, compared to the previous SOTA method RAG-T2M. Code: https://github.com/AIGeeksGroup/ReMoMask. Website: https://aigeeksgroup.github.io/ReMoMask.
>
---
#### [replaced 047] RoboSwap: A GAN-driven Video Diffusion Framework For Unsupervised Robot Arm Swapping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.08632v2](http://arxiv.org/pdf/2506.08632v2)**

> **作者:** Yang Bai; Liudi Yang; George Eskandar; Fengyi Shen; Dong Chen; Mohammad Altillawi; Ziyuan Liu; Gitta Kutyniok
>
> **摘要:** Recent advancements in generative models have revolutionized video synthesis and editing. However, the scarcity of diverse, high-quality datasets continues to hinder video-conditioned robotic learning, limiting cross-platform generalization. In this work, we address the challenge of swapping a robotic arm in one video with another: a key step for crossembodiment learning. Unlike previous methods that depend on paired video demonstrations in the same environmental settings, our proposed framework, RoboSwap, operates on unpaired data from diverse environments, alleviating the data collection needs. RoboSwap introduces a novel video editing pipeline integrating both GANs and diffusion models, combining their isolated advantages. Specifically, we segment robotic arms from their backgrounds and train an unpaired GAN model to translate one robotic arm to another. The translated arm is blended with the original video background and refined with a diffusion model to enhance coherence, motion realism and object interaction. The GAN and diffusion stages are trained independently. Our experiments demonstrate that RoboSwap outperforms state-of-the-art video and image editing models on three benchmarks in terms of both structural coherence and motion consistency, thereby offering a robust solution for generating reliable, cross-embodiment data in robotic learning.
>
---
#### [replaced 048] InfMasking: Unleashing Synergistic Information by Contrastive Multimodal Interactions
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.25270v2](http://arxiv.org/pdf/2509.25270v2)**

> **作者:** Liangjian Wen; Qun Dai; Jianzhuang Liu; Jiangtao Zheng; Yong Dai; Dongkai Wang; Zhao Kang; Jun Wang; Zenglin Xu; Jiang Duan
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** In multimodal representation learning, synergistic interactions between modalities not only provide complementary information but also create unique outcomes through specific interaction patterns that no single modality could achieve alone. Existing methods may struggle to effectively capture the full spectrum of synergistic information, leading to suboptimal performance in tasks where such interactions are critical. This is particularly problematic because synergistic information constitutes the fundamental value proposition of multimodal representation. To address this challenge, we introduce InfMasking, a contrastive synergistic information extraction method designed to enhance synergistic information through an Infinite Masking strategy. InfMasking stochastically occludes most features from each modality during fusion, preserving only partial information to create representations with varied synergistic patterns. Unmasked fused representations are then aligned with masked ones through mutual information maximization to encode comprehensive synergistic information. This infinite masking strategy enables capturing richer interactions by exposing the model to diverse partial modality combinations during training. As computing mutual information estimates with infinite masking is computationally prohibitive, we derive an InfMasking loss to approximate this calculation. Through controlled experiments, we demonstrate that InfMasking effectively enhances synergistic information between modalities. In evaluations on large-scale real-world datasets, InfMasking achieves state-of-the-art performance across seven benchmarks. Code is released at https://github.com/brightest66/InfMasking.
>
---
#### [replaced 049] Negative Shanshui: Real-time Interactive Ink Painting Synthesis
- **分类: cs.HC; cs.AI; cs.CV; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.16612v2](http://arxiv.org/pdf/2508.16612v2)**

> **作者:** Aven-Le Zhou
>
> **摘要:** This paper presents Negative Shanshui, a real-time interactive AI synthesis approach that reinterprets classical Chinese landscape ink painting, i.e., shanshui, to engage with ecological crises in the Anthropocene. Negative Shanshui optimizes a fine-tuned Stable Diffusion model for real-time inferences and integrates it with gaze-driven inpainting, frame interpolation; it enables dynamic morphing animations in response to the viewer's gaze and presents as an interactive virtual reality (VR) experience. The paper describes the complete technical pipeline, covering the system framework, optimization strategies, gaze-based interaction, and multimodal deployment in an art festival. Further analysis of audience feedback collected during its public exhibition highlights how participants variously engaged with the work through empathy, ambivalence, and critical reflection.
>
---
#### [replaced 050] SIRI-Bench: Challenging VLMs' Spatial Intelligence through Complex Reasoning Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14512v2](http://arxiv.org/pdf/2506.14512v2)**

> **作者:** Zijian Song; Xiaoxin Lin; Qiuming Huang; Guangrun Wang; Liang Lin
>
> **备注:** 14 pages
>
> **摘要:** Large Language Models (LLMs) have undergone rapid progress, largely attributed to reinforcement learning on complex reasoning tasks. In contrast, while spatial intelligence is fundamental for Vision-Language Models (VLMs) in real-world interaction, the systematic study of their complex spatial reasoning remains underexplored. To bridge this gap, we introduce SIRI-Bench, a benchmark designed to evaluate VLMs' structural spatial intelligence through spatial-grounded reasoning tasks. SIRI-Bench comprises 9,000 video-question-answer triplets, where each problem is embedded in a realistic 3D scene. The benchmark is carefully designed so that solving each problem requires both spatial comprehension and structural reasoning. To facilitate large-scale data synthesis, we develop an Automatic Scene Creation Engine that employs collaborative LLM agents to translate abstract mathematical problems into faithful 3D scenes. Experimental results reveal that state-of-the-art VLMs struggle significantly on SIRI-Bench, underscoring the challenge of structural spatial reasoning. We hope that our study will bring researchers' attention to spatially grounded reasoning and advance VLMs in visual problem-solving.
>
---
#### [replaced 051] CryoCCD: Conditional Cycle-consistent Diffusion with Biophysical Modeling for Cryo-EM Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23444v4](http://arxiv.org/pdf/2505.23444v4)**

> **作者:** Runmin Jiang; Genpei Zhang; Yuntian Yang; Siqi Wu; Minhao Wu; Wanyue Feng; Yizhou Zhao; Xi Xiao; Xiao Wang; Tianyang Wang; Xingjian Li; Muyuan Chen; Min Xu
>
> **摘要:** Single-particle cryo-electron microscopy (cryo-EM) has become a cornerstone of structural biology, enabling near-atomic resolution analysis of macromolecules through advanced computational methods. However, the development of cryo-EM processing tools is constrained by the scarcity of high-quality annotated datasets. Synthetic data generation offers a promising alternative, but existing approaches lack thorough biophysical modeling of heterogeneity and fail to reproduce the complex noise observed in real imaging. To address these limitations, we present CryoCCD, a synthesis framework that unifies versatile biophysical modeling with the first conditional cycle-consistent diffusion model tailored for cryo-EM. The biophysical engine provides multi-functional generation capabilities to capture authentic biological organization, and the diffusion model is enhanced with cycle consistency and mask-guided contrastive learning to ensure realistic noise while preserving structural fidelity. Extensive experiments demonstrate that CryoCCD generates structurally faithful micrographs, enhances particle picking and pose estimation, as well as achieves superior performance over state-of-the-art baselines, while also generalizing effectively to held-out protein families.
>
---
#### [replaced 052] Mask What Matters: Controllable Text-Guided Masking for Self-Supervised Medical Image Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23054v2](http://arxiv.org/pdf/2509.23054v2)**

> **作者:** Ruilang Wang; Shuotong Xu; Bowen Liu; Runlin Huang; Donglong Chen; Weifeng Su
>
> **摘要:** The scarcity of annotated data in specialized domains such as medical imaging presents significant challenges to training robust vision models. While self-supervised masked image modeling (MIM) offers a promising solution, existing approaches largely rely on random high-ratio masking, leading to inefficiency and poor semantic alignment. Moreover, region-aware variants typically depend on reconstruction heuristics or supervised signals, limiting their adaptability across tasks and modalities. We propose Mask What Matters, a controllable text-guided masking framework for self-supervised medical image analysis. By leveraging vision-language models for prompt-based region localization, our method flexibly applies differentiated masking to emphasize diagnostically relevant regions while reducing redundancy in background areas. This controllable design enables better semantic alignment, improved representation learning, and stronger cross-task generalizability. Comprehensive evaluation across multiple medical imaging modalities, including brain MRI, chest CT, and lung X-ray, shows that Mask What Matters consistently outperforms existing MIM methods (e.g., SparK), achieving gains of up to +3.1 percentage points in classification accuracy, +1.3 in box average precision (BoxAP), and +1.1 in mask average precision (MaskAP) for detection. Notably, it achieves these improvements with substantially lower overall masking ratios (e.g., 40\% vs. 70\%). This work demonstrates that controllable, text-driven masking can enable semantically aligned self-supervised learning, advancing the development of robust vision models for medical image analysis.
>
---
#### [replaced 053] Interpreting deep learning-based stellar mass estimation via causal analysis and mutual information decomposition
- **分类: astro-ph.IM; astro-ph.GA; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23901v2](http://arxiv.org/pdf/2509.23901v2)**

> **作者:** Wei Zhang; Qiufan Lin; Yuan-Sen Ting; Shupei Chen; Hengxin Ruan; Song Li; Yifan Wang
>
> **备注:** Accepted at Astronomy & Astrophysics; 23 + 12 pages; 8 + 16 figures
>
> **摘要:** End-to-end deep learning models fed with multi-band galaxy images are powerful data-driven tools used to estimate galaxy physical properties in the absence of spectroscopy. However, due to a lack of interpretability and the associational nature of such models, it is difficult to understand how the information that is included in addition to integrated photometry (e.g., morphology) contributes to the estimation task. Improving our understanding in this field would enable further advances into unraveling the physical connections among galaxy properties and optimizing data exploitation. Therefore, our work is aimed at interpreting the deep learning-based estimation of stellar mass via two interpretability techniques: causal analysis and mutual information decomposition. The former reveals the causal paths between multiple variables beyond nondirectional statistical associations, while the latter quantifies the multicomponent contributions (i.e., redundant, unique, and synergistic) of different input data to the stellar mass estimation. Using data from the Sloan Digital Sky Survey (SDSS) and the Wide-field Infrared Survey Explorer (WISE), we obtained meaningful results that provide physical interpretations for image-based models. Our work demonstrates the gains from combining deep learning with interpretability techniques, and holds promise in promoting more data-driven astrophysical research (e.g., astrophysical parameter estimations and investigations on complex multivariate physical processes).
>
---
#### [replaced 054] VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.02282v2](http://arxiv.org/pdf/2510.02282v2)**

> **作者:** Kyoungjun Park; Yifan Yang; Juheon Yi; Shicheng Zheng; Yifei Shen; Dongqi Han; Caihua Shan; Muhammad Muaz; Lili Qiu
>
> **摘要:** With the rapid advancement of AI-generated videos, there is an urgent need for effective detection tools to mitigate societal risks such as misinformation and reputational harm. In addition to accurate classification, it is essential that detection models provide interpretable explanations to ensure transparency for regulators and end users. To address these challenges, we introduce VidGuard-R1, the first video authenticity detector that fine-tunes a multi-modal large language model (MLLM) using group relative policy optimization (GRPO). Our model delivers both highly accurate judgments and insightful reasoning. We curate a challenging dataset of 140k real and AI-generated videos produced by state-of-the-art generation models, carefully designing the generation process to maximize discrimination difficulty. We then fine-tune Qwen-VL using GRPO with two specialized reward models that target temporal artifacts and generation complexity. Extensive experiments demonstrate that VidGuard-R1 achieves state-of-the-art zero-shot performance on existing benchmarks, with additional training pushing accuracy above 95%. Case studies further show that VidGuard-R1 produces precise and interpretable rationales behind its predictions. The code is publicly available at https://VidGuard-R1.github.io.
>
---
#### [replaced 055] From Gaze to Insight: Bridging Human Visual Attention and Vision Language Model Explanation for Weakly-Supervised Medical Image Segmentation
- **分类: cs.CV; 68T45; I.2.10; I.4.8**

- **链接: [http://arxiv.org/pdf/2504.11368v2](http://arxiv.org/pdf/2504.11368v2)**

> **作者:** Jingkun Chen; Haoran Duan; Xiao Zhang; Boyan Gao; Vicente Grau; Jungong Han
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Medical image segmentation remains challenging due to the high cost of pixel-level annotations for training. In the context of weak supervision, clinician gaze data captures regions of diagnostic interest; however, its sparsity limits its use for segmentation. In contrast, vision-language models (VLMs) provide semantic context through textual descriptions but lack the explanation precision required. Recognizing that neither source alone suffices, we propose a teacher-student framework that integrates both gaze and language supervision, leveraging their complementary strengths. Our key insight is that gaze data indicates where clinicians focus during diagnosis, while VLMs explain why those regions are significant. To implement this, the teacher model first learns from gaze points enhanced by VLM-generated descriptions of lesion morphology, establishing a foundation for guiding the student model. The teacher then directs the student through three strategies: (1) Multi-scale feature alignment to fuse visual cues with textual semantics; (2) Confidence-weighted consistency constraints to focus on reliable predictions; (3) Adaptive masking to limit error propagation in uncertain areas. Experiments on the Kvasir-SEG, NCI-ISBI, and ISIC datasets show that our method achieves Dice scores of 80.78%, 80.53%, and 84.22%, respectively-improving 3-5% over gaze baselines without increasing the annotation burden. By preserving correlations among predictions, gaze data, and lesion descriptions, our framework also maintains clinical interpretability. This work illustrates how integrating human visual attention with AI-generated semantic context can effectively overcome the limitations of individual weak supervision signals, thereby advancing the development of deployable, annotation-efficient medical AI systems. Code is available at: https://github.com/jingkunchen/FGI.
>
---
#### [replaced 056] FreqPolicy: Frequency Autoregressive Visuomotor Policy with Continuous Tokens
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01583v2](http://arxiv.org/pdf/2506.01583v2)**

> **作者:** Yiming Zhong; Yumeng Liu; Chuyang Xiao; Zemin Yang; Youzhuo Wang; Yufei Zhu; Ye Shi; Yujing Sun; Xinge Zhu; Yuexin Ma
>
> **备注:** Comments: Published at Neural Information Processing Systems (NeurIPS) 2025. Project page and code: https://freq-policy.github.io/
>
> **摘要:** Learning effective visuomotor policies for robotic manipulation is challenging, as it requires generating precise actions while maintaining computational efficiency. Existing methods remain unsatisfactory due to inherent limitations in the essential action representation and the basic network architectures. We observe that representing actions in the frequency domain captures the structured nature of motion more effectively: low-frequency components reflect global movement patterns, while high-frequency components encode fine local details. Additionally, robotic manipulation tasks of varying complexity demand different levels of modeling precision across these frequency bands. Motivated by this, we propose a novel paradigm for visuomotor policy learning that progressively models hierarchical frequency components. To further enhance precision, we introduce continuous latent representations that maintain smoothness and continuity in the action space. Extensive experiments across diverse 2D and 3D robotic manipulation benchmarks demonstrate that our approach outperforms existing methods in both accuracy and efficiency, showcasing the potential of a frequency-domain autoregressive framework with continuous tokens for generalized robotic manipulation.Code is available at https://github.com/4DVLab/Freqpolicy
>
---
#### [replaced 057] Deep Spectral Epipolar Representations for Dense Light Field Reconstruction
- **分类: cs.CV; 68T45, 68U10; I.4.8; I.2.10**

- **链接: [http://arxiv.org/pdf/2508.08900v2](http://arxiv.org/pdf/2508.08900v2)**

> **作者:** Noor Islam S. Mohammad
>
> **备注:** There are 14 pages, 8 figures, and 3 tables. This full-length article was processed to be submitted to the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) in October 2025
>
> **摘要:** Accurate and efficient dense depth reconstruction from light field imagery remains a central challenge in computer vision, underpinning applications such as augmented reality, biomedical imaging, and 3D scene reconstruction. Existing deep convolutional approaches, while effective, often incur high computational overhead and are sensitive to noise and disparity inconsistencies in real-world scenarios. This paper introduces a novel Deep Spectral Epipolar Representation (DSER) framework for dense light field reconstruction, which unifies deep spectral feature learning with epipolar-domain regularization. The proposed approach exploits frequency-domain correlations across epipolar plane images to enforce global structural coherence, thereby mitigating artifacts and enhancing depth accuracy. Unlike conventional supervised models, DSER operates efficiently with limited training data while maintaining high reconstruction fidelity. Comprehensive experiments on the 4D Light Field Benchmark and a diverse set of real-world datasets demonstrate that DSER achieves superior performance in terms of precision, structural consistency, and computational efficiency compared to state-of-the-art methods. These results highlight the potential of integrating spectral priors with epipolar geometry for scalable and noise-resilient dense light field depth estimation, establishing DSER as a promising direction for next-generation high-dimensional vision systems.
>
---
#### [replaced 058] Controllable Video Generation with Provable Disentanglement
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.02690v3](http://arxiv.org/pdf/2502.02690v3)**

> **作者:** Yifan Shen; Peiyuan Zhu; Zijian Li; Shaoan Xie; Namrata Deka; Zongfang Liu; Zeyu Tang; Guangyi Chen; Kun Zhang
>
> **摘要:** Controllable video generation remains a significant challenge, despite recent advances in generating high-quality and consistent videos. Most existing methods for controlling video generation treat the video as a whole, neglecting intricate fine-grained spatiotemporal relationships, which limits both control precision and efficiency. In this paper, we propose Controllable Video Generative Adversarial Networks (CoVoGAN) to disentangle the video concepts, thus facilitating efficient and independent control over individual concepts. Specifically, following the minimal change principle, we first disentangle static and dynamic latent variables. We then leverage the sufficient change property to achieve component-wise identifiability of dynamic latent variables, enabling disentangled control of video generation. To establish the theoretical foundation, we provide a rigorous analysis demonstrating the identifiability of our approach. Building on these theoretical insights, we design a Temporal Transition Module to disentangle latent dynamics. To enforce the minimal change principle and sufficient change property, we minimize the dimensionality of latent dynamic variables and impose temporal conditional independence. To validate our approach, we integrate this module as a plug-in for GANs. Extensive qualitative and quantitative experiments on various video generation benchmarks demonstrate that our method significantly improves generation quality and controllability across diverse real-world scenarios.
>
---
#### [replaced 059] Depth-Sequence Transformer (DST) for Segment-Specific ICA Calcification Mapping on Non-Contrast CT
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08214v3](http://arxiv.org/pdf/2507.08214v3)**

> **作者:** Xiangjian Hou; Ebru Yaman Akcicek; Xin Wang; Kazem Hashemizadeh; Scott Mcnally; Chun Yuan; Xiaodong Ma
>
> **备注:** Accept to IEEE BIBM 2025
>
> **摘要:** While total intracranial carotid artery calcification (ICAC) volume is an established stroke biomarker, growing evidence shows this aggregate metric ignores the critical influence of plaque location, since calcification in different segments carries distinct prognostic and procedural risks. However, a finer-grained, segment-specific quantification has remained technically infeasible. Conventional 3D models are forced to process downsampled volumes or isolated patches, sacrificing the global context required to resolve anatomical ambiguity and render reliable landmark localization. To overcome this, we reformulate the 3D challenge as a \textbf{Parallel Probabilistic Landmark Localization} task along the 1D axial dimension. We propose the \textbf{Depth-Sequence Transformer (DST)}, a framework that processes full-resolution CT volumes as sequences of 2D slices, learning to predict $N=6$ independent probability distributions that pinpoint key anatomical landmarks. Our DST framework demonstrates exceptional accuracy and robustness. Evaluated on a 100-patient clinical cohort with rigorous 5-fold cross-validation, it achieves a Mean Absolute Error (MAE) of \textbf{0.1 slices}, with \textbf{96\%} of predictions falling within a $\pm1$ slice tolerance. Furthermore, to validate its architectural power, the DST backbone establishes the best result on the public Clean-CC-CCII classification benchmark under an end-to-end evaluation protocol. Our work delivers the first practical tool for automated segment-specific ICAC analysis. The proposed framework provides a foundation for further studies on the role of location-specific biomarkers in diagnosis, prognosis, and procedural planning.
>
---
#### [replaced 060] One Patch to Caption Them All: A Unified Zero-Shot Captioning Framework
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.02898v2](http://arxiv.org/pdf/2510.02898v2)**

> **作者:** Lorenzo Bianchi; Giacomo Pacini; Fabio Carrara; Nicola Messina; Giuseppe Amato; Fabrizio Falchi
>
> **摘要:** Zero-shot captioners are recently proposed models that utilize common-space vision-language representations to caption images without relying on paired image-text data. To caption an image, they proceed by textually decoding a text-aligned image feature, but they limit their scope to global representations and whole-image captions. We present Patch-ioner, a unified framework for zero-shot captioning that shifts from an image-centric to a patch-centric paradigm, enabling the captioning of arbitrary regions without the need of region-level supervision. Instead of relying on global image representations, we treat individual patches as atomic captioning units and aggregate them to describe arbitrary regions, from single patches to non-contiguous areas and entire images. We analyze the key ingredients that enable current latent captioners to work in our novel proposed framework. Experiments demonstrate that backbones producing meaningful, dense visual features, such as DINO, are key to achieving state-of-the-art performance in multiple region-based captioning tasks. Compared to other baselines and state-of-the-art competitors, our models achieve better performance on zero-shot dense, region-set, and a newly introduced trace captioning task, highlighting the effectiveness of patch-wise semantic representations for scalable caption generation. Project page at https://paciosoft.com/Patch-ioner/ .
>
---
#### [replaced 061] Region-of-Interest Augmentation for Mammography Classification under Patient-Level Cross-Validation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.20585v2](http://arxiv.org/pdf/2509.20585v2)**

> **作者:** Farbod Bigdeli; Mohsen Mohammadagha; Ali Bigdeli
>
> **备注:** 5 pages, 5 figures, 2 tables
>
> **摘要:** Breast cancer screening with mammography remains central to early detection and mortality reduction. Deep learning has shown strong potential for automating mammogram interpretation, yet limited-resolution datasets and small sample sizes continue to restrict performance. We revisit the Mini-DDSM dataset (9,684 images; 2,414 patients) and introduce a lightweight region-of-interest (ROI) augmentation strategy. During training, full images are probabilistically replaced with random ROI crops sampled from a precomputed, label-free bounding-box bank, with optional jitter to increase variability. We evaluate under strict patient-level cross-validation and report ROC-AUC, PR-AUC, and training-time efficiency metrics (throughput and GPU memory). Because ROI augmentation is training-only, inference-time cost remains unchanged. On Mini-DDSM, ROI augmentation (best: p_roi = 0.10, alpha = 0.10) yields modest average ROC-AUC gains, with performance varying across folds; PR-AUC is flat to slightly lower. These results demonstrate that simple, data-centric ROI strategies can enhance mammography classification in constrained settings without requiring additional labels or architectural modifications.
>
---
#### [replaced 062] Divergence Minimization Preference Optimization for Diffusion Model Alignment
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.07510v2](http://arxiv.org/pdf/2507.07510v2)**

> **作者:** Binxu Li; Minkai Xu; Jiaqi Han; Meihua Dang; Stefano Ermon
>
> **摘要:** Diffusion models have achieved remarkable success in generating realistic and versatile images from text prompts. Inspired by the recent advancements of language models, there is an increasing interest in further improving the models by aligning with human preferences. However, we investigate alignment from a divergence minimization perspective and reveal that existing preference optimization methods are typically trapped in suboptimal mean-seeking optimization. In this paper, we introduce Divergence Minimization Preference Optimization (DMPO), a novel and principled method for aligning diffusion models by minimizing reverse KL divergence, which asymptotically enjoys the same optimization direction as original RL. We provide rigorous analysis to justify the effectiveness of DMPO and conduct comprehensive experiments to validate its empirical strength across both human evaluations and automatic metrics. Our extensive results show that diffusion models fine-tuned with DMPO can consistently outperform or match existing techniques, specifically consistently outperforming all baseline models across different base models and test sets, achieving the best PickScore in every case, demonstrating the method's superiority in aligning generative behavior with desired outputs. Overall, DMPO unlocks a robust and elegant pathway for preference alignment, bridging principled theory with practical performance in diffusion models.
>
---
#### [replaced 063] EfficientMIL: Efficient Linear-Complexity MIL Method for WSI Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23640v2](http://arxiv.org/pdf/2509.23640v2)**

> **作者:** Chengying She; Chengwei Chen; Dongjie Fan; Lizhuang Liu; Chengwei Shao; Yun Bian; Ben Wang; Xinran Zhang
>
> **备注:** Submitted to Array
>
> **摘要:** Whole slide images (WSIs) classification represents a fundamental challenge in computational pathology, where multiple instance learning (MIL) has emerged as the dominant paradigm. Current state-of-the-art (SOTA) MIL methods rely on attention mechanisms, achieving good performance but requiring substantial computational resources due to quadratic complexity when processing hundreds of thousands of patches. To address this computational bottleneck, we introduce EfficientMIL, a novel linear-complexity MIL approach for WSIs classification with the patches selection module Adaptive Patch Selector (APS) that we designed, replacing the quadratic-complexity self-attention mechanisms in Transformer-based MIL methods with efficient sequence models including RNN-based GRU, LSTM, and State Space Model (SSM) Mamba. EfficientMIL achieves significant computational efficiency improvements while outperforming other MIL methods across multiple histopathology datasets. On TCGA-Lung dataset, EfficientMIL-Mamba achieved AUC of 0.976 and accuracy of 0.933, while on CAMELYON16 dataset, EfficientMIL-GRU achieved AUC of 0.990 and accuracy of 0.975, surpassing previous state-of-the-art methods. Extensive experiments demonstrate that APS is also more effective for patches selection than conventional selection strategies.
>
---
#### [replaced 064] Reconstructing Topology-Consistent Face Mesh by Volume Rendering from Multi-View Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.05606v2](http://arxiv.org/pdf/2404.05606v2)**

> **作者:** Yating Wang; Ran Yi; Xiaoning Lei; Ke Fan; Jinkun Hao; Lizhuang Ma
>
> **摘要:** Industrial 3D face assets creation typically reconstructs topology-consistent face meshes from multi-view images for downstream production. However, high-quality reconstruction usually requires manual processing or specific capture settings. Recently NeRF has shown great advantages in 3D reconstruction, by representing scenes as density and radiance fields and utilizing neural volume rendering for novel view synthesis. Inspired by this, we introduce a novel method which combines explicit mesh with neural volume rendering to optimize geometry of an artist-made template face mesh from multi-view images while keeping the topology unchanged. Our method derives density fields from meshes using distance fields as an intermediary and encodes radiance field in compact tri-planes. To improve convergence, several adaptions tailored for meshes are introduced to the volume rendering. Experiments demonstrate that our method achieves superior reconstruction quality compared to previous approaches, validating the feasibility of integrating mesh and neural volume rendering.
>
---
#### [replaced 065] Motion Blender Gaussian Splatting for Dynamic Scene Reconstruction
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.09040v4](http://arxiv.org/pdf/2503.09040v4)**

> **作者:** Xinyu Zhang; Haonan Chang; Yuhan Liu; Abdeslam Boularias
>
> **备注:** CoRL 2025
>
> **摘要:** Gaussian splatting has emerged as a powerful tool for high-fidelity reconstruction of dynamic scenes. However, existing methods primarily rely on implicit motion representations, such as encoding motions into neural networks or per-Gaussian parameters, which makes it difficult to further manipulate the reconstructed motions. This lack of explicit controllability limits existing methods to replaying recorded motions only, which hinders a wider application in robotics. To address this, we propose Motion Blender Gaussian Splatting (MBGS), a novel framework that uses motion graphs as an explicit and sparse motion representation. The motion of a graph's links is propagated to individual Gaussians via dual quaternion skinning, with learnable weight painting functions that determine the influence of each link. The motion graphs and 3D Gaussians are jointly optimized from input videos via differentiable rendering. Experiments show that MBGS achieves state-of-the-art performance on the highly challenging iPhone dataset while being competitive on HyperNeRF. We demonstrate the application potential of our method in animating novel object poses, synthesizing real robot demonstrations, and predicting robot actions through visual planning. The source code, models, video demonstrations can be found at http://mlzxy.github.io/motion-blender-gs.
>
---
#### [replaced 066] DS-VTON: An Enhanced Dual-Scale Coarse-to-Fine Framework for Virtual Try-On
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00908v2](http://arxiv.org/pdf/2506.00908v2)**

> **作者:** Xianbing Sun; Yan Hong; Jiahui Zhan; Jun Lan; Huijia Zhu; Weiqiang Wang; Liqing Zhang; Jianfu Zhang
>
> **摘要:** Despite recent progress, most existing virtual try-on methods still struggle to simultaneously address two core challenges: accurately aligning the garment image with the target human body, and preserving fine-grained garment textures and patterns. These two requirements map directly onto a coarse-to-fine generation paradigm, where the coarse stage handles structural alignment and the fine stage recovers rich garment details. Motivated by this observation, we propose DS-VTON, an enhanced dual-scale coarse-to-fine framework that tackles the try-on problem more effectively. DS-VTON consists of two stages: the first stage generates a low-resolution try-on result to capture the semantic correspondence between garment and body, where reduced detail facilitates robust structural alignment. In the second stage, a blend-refine diffusion process reconstructs high-resolution outputs by refining the residual between scales through noise-image blending, emphasizing texture fidelity and effectively correcting fine-detail errors from the low-resolution stage. In addition, our method adopts a fully mask-free generation strategy, eliminating reliance on human parsing maps or segmentation masks. Extensive experiments show that DS-VTON not only achieves state-of-the-art performance but consistently and significantly surpasses prior methods in both structural alignment and texture fidelity across multiple standard virtual try-on benchmarks.
>
---
#### [replaced 067] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05193v3](http://arxiv.org/pdf/2507.05193v3)**

> **作者:** Songxiao Yang; Haolin Wang; Yao Fu; Ye Tian; Tamotsu Kamishima; Masayuki Ikebe; Yafei Ou; Masatoshi Okutomi
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Rheumatoid arthritis (RA) is a common autoimmune disease that has been the focus of research in computer-aided diagnosis (CAD) and disease monitoring. In clinical settings, conventional radiography (CR) is widely used for the screening and evaluation of RA due to its low cost and accessibility. The wrist is a critical region for the diagnosis of RA. However, CAD research in this area remains limited, primarily due to the challenges in acquiring high-quality instance-level annotations. (i) The wrist comprises numerous small bones with narrow joint spaces, complex structures, and frequent overlaps, requiring detailed anatomical knowledge for accurate annotation. (ii) Disease progression in RA often leads to osteophyte, bone erosion (BE), and even bony ankylosis, which alter bone morphology and increase annotation difficulty, necessitating expertise in rheumatology. This work presents a multi-task dataset for wrist bone in CR, including two tasks: (i) wrist bone instance segmentation and (ii) Sharp/van der Heijde (SvdH) BE scoring, which is the first public resource for wrist bone instance segmentation. This dataset comprises 1048 wrist conventional radiographs of 388 patients from six medical centers, with pixel-level instance segmentation annotations for 618 images and SvdH BE scores for 800 images. This dataset can potentially support a wide range of research tasks related to RA, including joint space narrowing (JSN) progression quantification, BE detection, bone deformity evaluation, and osteophyte detection. It may also be applied to other wrist-related tasks, such as carpal bone fracture localization. We hope this dataset will significantly lower the barrier to research on wrist RA and accelerate progress in CAD research within the RA-related domain.
>
---
#### [replaced 068] Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.08134v3](http://arxiv.org/pdf/2508.08134v3)**

> **作者:** Zeqian Long; Mingzhe Zheng; Kunyu Feng; Xinhua Zhang; Hongyu Liu; Harry Yang; Linfeng Zhang; Qifeng Chen; Yue Ma
>
> **备注:** Project webpage is available at https://follow-your-shape.github.io/
>
> **摘要:** While recent flow-based image editing models demonstrate general-purpose capabilities across diverse tasks, they often struggle to specialize in challenging scenarios -- particularly those involving large-scale shape transformations. When performing such structural edits, these methods either fail to achieve the intended shape change or inadvertently alter non-target regions, resulting in degraded background quality. We propose Follow-Your-Shape, a training-free and mask-free framework that supports precise and controllable editing of object shapes while strictly preserving non-target content. Motivated by the divergence between inversion and editing trajectories, we compute a Trajectory Divergence Map (TDM) by comparing token-wise velocity differences between the inversion and denoising paths. The TDM enables precise localization of editable regions and guides a Scheduled KV Injection mechanism that ensures stable and faithful editing. To facilitate a rigorous evaluation, we introduce ReShapeBench, a new benchmark comprising 120 new images and enriched prompt pairs specifically curated for shape-aware editing. Experiments demonstrate that our method achieves superior editability and visual fidelity, particularly in tasks requiring large-scale shape replacement.
>
---
#### [replaced 069] Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10178v2](http://arxiv.org/pdf/2506.10178v2)**

> **作者:** Bill Psomas; Dionysis Christopoulos; Eirini Baltzi; Ioannis Kakogeorgiou; Tilemachos Aravanis; Nikos Komodakis; Konstantinos Karantzalos; Yannis Avrithis; Giorgos Tolias
>
> **备注:** 9 main paper pages, 13 supplementary pages; Code available at https://github.com/billpsomas/efficient-probing
>
> **摘要:** As fine-tuning becomes increasingly impractical at scale, probing is emerging as the preferred evaluation protocol. Yet, the standard linear probing fails to adequately reflect the potential of models whose pre-training optimizes representations of patch tokens rather than an explicit global representation. This motivates the need for attentive probing, an alternative that uses attention to selectively aggregate patch-level features. Despite its growing adoption, attentive probing remains under-explored, with existing methods suffering from excessive parameterization and poor computational efficiency. In this work, we revisit attentive probing through the lens of the accuracy vs. parameter efficiency trade-off. We present the first comprehensive study of existing methods, analyzing their design choices and benchmarking their performance. Building on this, we propose efficient probing (EP), a simple yet effective multi-query cross-attention mechanism that eliminates redundant projections and reduces the number of trainable parameters. Despite its simplicity, EP outperforms linear probing and prior attentive probing approaches across seven benchmarks, generalizes well to diverse pre-training paradigms, and delivers strong low-shot and layer-wise gains. Beyond evaluation, our analysis uncovers emerging properties of EP, such as complementary attention maps, which open new directions for leveraging probing beyond protocol design. Code available at https://github.com/billpsomas/efficient-probing.
>
---
#### [replaced 070] PanDORA: Casual HDR Radiance Acquisition for Indoor Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.06150v2](http://arxiv.org/pdf/2407.06150v2)**

> **作者:** Mohammad Reza Karimi Dastjerdi; Dominique Tanguay-Gaudreau; Frédéric Fortier-Chouinard; Yannick Hold-Geoffroy; Claude Demers; Nima Kalantari; Jean-François Lalonde
>
> **备注:** 13 pages, 11 figures
>
> **摘要:** Most novel view synthesis methods-including Neural Radiance Fields (NeRF)-struggle to capture the true high dynamic range (HDR) radiance of scenes. This is primarily due to their dependence on low dynamic range (LDR) images from conventional cameras. Exposure bracketing techniques aim to address this challenge, but they introduce a considerable time burden during the acquisition process. In this work, we introduce PanDORA: PANoramic Dual-Observer Radiance Acquisition, a system designed for the casual, high quality HDR capture of indoor environments. Our approach uses two 360{\deg} cameras mounted on a portable monopod to simultaneously record two panoramic 360{\deg} videos: one with standard exposure and another at fast shutter speed. The resulting video data is processed by a proposed two-stage NeRF-based algorithm, including an algorithm for the fine alignment of the fast- and well-exposed frames, generating non-saturated HDR radiance maps. Compared to existing methods on a novel dataset of real indoor scenes captured with our apparatus and including HDR ground truth lighting, PanDORA achieves superior visual fidelity and provides a scalable solution for capturing real environments in HDR.
>
---
#### [replaced 071] Novel Object 6D Pose Estimation with a Single Reference View
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.05578v3](http://arxiv.org/pdf/2503.05578v3)**

> **作者:** Jian Liu; Wei Sun; Kai Zeng; Jin Zheng; Hui Yang; Hossein Rahmani; Ajmal Mian; Lin Wang
>
> **备注:** 17 pages, 12 figures (including supplementary material)
>
> **摘要:** Existing novel object 6D pose estimation methods typically rely on CAD models or dense reference views, which are both difficult to acquire. Using only a single reference view is more scalable, but challenging due to large pose discrepancies and limited geometric and spatial information. To address these issues, we propose a Single-Reference-based novel object 6D (SinRef-6D) pose estimation method. Our key idea is to iteratively establish point-wise alignment in a common coordinate system based on state space models (SSMs). Specifically, iterative object-space point-wise alignment can effectively handle large pose discrepancies, while our proposed RGB and Points SSMs can capture long-range dependencies and spatial information from a single view, offering linear complexity and superior spatial modeling capability. Once pre-trained on synthetic data, SinRef-6D can estimate the 6D pose of a novel object using only a single reference view, without requiring retraining or a CAD model. Extensive experiments on six popular datasets and real-world robotic scenes demonstrate that we achieve on-par performance with CAD-based and dense reference view-based methods, despite operating in the more challenging single reference setting. Code will be released at https://github.com/CNJianLiu/SinRef-6D.
>
---
#### [replaced 072] Temporal Source Recovery for Time-Series Source-Free Unsupervised Domain Adaptation
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.19635v2](http://arxiv.org/pdf/2409.19635v2)**

> **作者:** Yucheng Wang; Peiliang Gong; Min Wu; Felix Ott; Xiaoli Li; Lihua Xie; Zhenghua Chen
>
> **摘要:** Time-Series (TS) data has grown in importance with the rise of Internet of Things devices like sensors, but its labeling remains costly and complex. While Unsupervised Domain Adaptation (UDAs) offers an effective solution, growing data privacy concerns have led to the development of Source-Free UDA (SFUDAs), enabling model adaptation to target domains without accessing source data. Despite their potential, applying existing SFUDAs to TS data is challenging due to the difficulty of transferring temporal dependencies, an essential characteristic of TS data, particularly in the absence of source samples. Although prior works attempt to address this by specific source pretraining designs, such requirements are often impractical, as source data owners cannot be expected to adhere to particular pretraining schemes. To address this, we propose Temporal Source Recovery (TemSR), a framework that leverages the intrinsic properties of TS data to generate a source-like domain and recover source temporal dependencies. With this domain, TemSR enables dependency transfer to the target domain without accessing source data or relying on source-specific designs, thereby facilitating effective and practical TS-SFUDA. TemSR features a masking recovery optimization process to generate a source-like distribution with restored temporal dependencies. This distribution is further refined through local context-aware regularization to preserve local dependencies, and anchor-based recovery diversity maximization to promote distributional diversity. Together, these components enable effective temporal dependency recovery and facilitate transfer across domains using standard UDA techniques. Extensive experiments across multiple TS tasks demonstrate the effectiveness of TemSR, which even surpasses existing TS-SFUDA methods that require source-specific designs.
>
---
#### [replaced 073] FrameMind: Frame-Interleaved Video Reasoning via Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.24008v3](http://arxiv.org/pdf/2509.24008v3)**

> **作者:** Haonan Ge; Yiwei Wang; Kai-Wei Chang; Hang Wu; Yujun Cai
>
> **备注:** Underreview
>
> **摘要:** Current video understanding models rely on fixed frame sampling strategies, processing predetermined visual inputs regardless of the specific reasoning requirements of each question. This static approach limits their ability to adaptively gather visual evidence, leading to suboptimal performance on tasks that require either broad temporal coverage or fine-grained spatial detail. In this paper, we introduce FrameMind, an end-to-end framework trained with reinforcement learning that enables models to dynamically request visual information during reasoning through Frame-Interleaved Chain-of-Thought (FiCOT). Unlike traditional approaches, FrameMind operates in multiple turns where the model alternates between textual reasoning and active visual perception, using tools to extract targeted frames or video clips based on identified knowledge gaps. To train effective dynamic sampling policies, we propose Dynamic Resolution Frame Sampling (DRFS), which exposes models to diverse temporal-spatial trade-offs during learning, and DRFS-GRPO, a group-relative policy optimization algorithm that learns from outcome-based rewards without requiring frame-level annotations. Extensive experiments on challenging benchmarks like MLVU and VideoMME demonstrate that our method significantly outperforms existing models, advancing the state of the art in flexible and efficient video understanding.
>
---
#### [replaced 074] Resolving Task Objective Conflicts in Unified Model via Task-Aware Mixture-of-Experts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03591v2](http://arxiv.org/pdf/2506.03591v2)**

> **作者:** Jiaxing Zhang; Hao Tang
>
> **摘要:** Unified multimodal large language models (MLLMs) based on end-to-end autoregressive (AR) transformers effectively integrate both understanding and generation tasks within a single framework. However, intrinsic Task Objective Conflicts between high-level semantic abstraction in understanding and fine-grained detail preservation in generation pose significant challenges, often leading to suboptimal trade-offs and task interference. Existing solutions, such as decoupling shared visual encoders, fall short of fundamentally resolving these conflicts due to inherent AR architecture. In this paper, we propose a novel approach that decouples internal components of AR to resolve task objective conflicts. Specifically, we design UTAMoE, a Unified Task-Aware Mixture-of-Experts (MoE) framework that decouples internal AR modules via a Task-Aware MoE Layer to create task-specific optimization subpaths. To enhance task differentiation while maintaining overall coordination, we introduce a novel Two-Stage Training Strategy. Extensive experiments on multimodal benchmarks demonstrate that UTAMoE mitigates task objective conflicts, achieving state-of-the-art performance across various tasks. Visualizations and ablation studies further validate the effectiveness of our approach.
>
---
#### [replaced 075] SIA: Enhancing Safety via Intent Awareness for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.16856v2](http://arxiv.org/pdf/2507.16856v2)**

> **作者:** Youngjin Na; Sangheon Jeong; Youngwan Lee; Jian Lee; Dawoon Jeong; Youngman Kim
>
> **备注:** Accepted to Safe and Trustworthy Multimodal AI Systems(SafeMM-AI) Workshop at ICCV2025, Non-archival track
>
> **摘要:** With the growing deployment of Vision-Language Models (VLMs) in real-world applications, previously overlooked safety risks are becoming increasingly evident. In particular, seemingly innocuous multimodal inputs can combine to reveal harmful intent, leading to unsafe model outputs. While multimodal safety has received increasing attention, existing approaches often fail to address such latent risks, especially when harmfulness arises only from the interaction between modalities. We propose SIA (Safety via Intent Awareness), a training-free, intent-aware safety framework that proactively detects harmful intent in multimodal inputs and uses it to guide the generation of safe responses. SIA follows a three-stage process: (1) visual abstraction via captioning; (2) intent inference through few-shot chain-of-thought (CoT) prompting; and (3) intent-conditioned response generation. By dynamically adapting to the implicit intent inferred from an image-text pair, SIA mitigates harmful outputs without extensive retraining. Extensive experiments on safety benchmarks, including SIUO, MM-SafetyBench, and HoliSafe, show that SIA consistently improves safety and outperforms prior training-free methods.
>
---
#### [replaced 076] TSLA: A Task-Specific Learning Adaptation for Semantic Segmentation on Autonomous Vehicles Platform
- **分类: cs.CV; cs.AI; cs.AR; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.12279v2](http://arxiv.org/pdf/2508.12279v2)**

> **作者:** Jun Liu; Zhenglun Kong; Pu Zhao; Weihao Zeng; Hao Tang; Xuan Shen; Changdi Yang; Wenbin Zhang; Geng Yuan; Wei Niu; Xue Lin; Yanzhi Wang
>
> **摘要:** Autonomous driving platforms encounter diverse driving scenarios, each with varying hardware resources and precision requirements. Given the computational limitations of embedded devices, it is crucial to consider computing costs when deploying on target platforms like the NVIDIA\textsuperscript{\textregistered} DRIVE PX 2. Our objective is to customize the semantic segmentation network according to the computing power and specific scenarios of autonomous driving hardware. We implement dynamic adaptability through a three-tier control mechanism -- width multiplier, classifier depth, and classifier kernel -- allowing fine-grained control over model components based on hardware constraints and task requirements. This adaptability facilitates broad model scaling, targeted refinement of the final layers, and scenario-specific optimization of kernel sizes, leading to improved resource allocation and performance. Additionally, we leverage Bayesian Optimization with surrogate modeling to efficiently explore hyperparameter spaces under tight computational budgets. Our approach addresses scenario-specific and task-specific requirements through automatic parameter search, accommodating the unique computational complexity and accuracy needs of autonomous driving. It scales its Multiply-Accumulate Operations (MACs) for Task-Specific Learning Adaptation (TSLA), resulting in alternative configurations tailored to diverse self-driving tasks. These TSLA customizations maximize computational capacity and model accuracy, optimizing hardware utilization.
>
---
#### [replaced 077] Automated Defect Detection for Mass-Produced Electronic Components Based on YOLO Object Detection Models
- **分类: cs.CV; cs.AI; cs.LG; eess.SP; 68T07, 68U10; I.4.8; I.2.10**

- **链接: [http://arxiv.org/pdf/2510.01914v2](http://arxiv.org/pdf/2510.01914v2)**

> **作者:** Wei-Lung Mao; Chun-Chi Wang; Po-Heng Chou; Yen-Ting Liu
>
> **备注:** 12 pages, 16 figures, 7 tables, and published in IEEE Sensors Journal
>
> **摘要:** Since the defect detection of conventional industry components is time-consuming and labor-intensive, it leads to a significant burden on quality inspection personnel and makes it difficult to manage product quality. In this paper, we propose an automated defect detection system for the dual in-line package (DIP) that is widely used in industry, using digital camera optics and a deep learning (DL)-based model. The two most common defect categories of DIP are examined: (1) surface defects, and (2) pin-leg defects. However, the lack of defective component images leads to a challenge for detection tasks. To solve this problem, the ConSinGAN is used to generate a suitable-sized dataset for training and testing. Four varieties of the YOLO model are investigated (v3, v4, v7, and v9), both in isolation and with the ConSinGAN augmentation. The proposed YOLOv7 with ConSinGAN is superior to the other YOLO versions in accuracy of 95.50\%, detection time of 285 ms, and is far superior to threshold-based approaches. In addition, the supervisory control and data acquisition (SCADA) system is developed, and the associated sensor architecture is described. The proposed automated defect detection can be easily established with numerous types of defects or insufficient defect data.
>
---
#### [replaced 078] Explaining Human Preferences via Metrics for Structured 3D Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08208v2](http://arxiv.org/pdf/2503.08208v2)**

> **作者:** Jack Langerman; Denys Rozumnyi; Yuzhong Huang; Dmytro Mishkin
>
> **备注:** ICCV 2025 Highlight
>
> **摘要:** "What cannot be measured cannot be improved" while likely never uttered by Lord Kelvin, summarizes effectively the driving force behind this work. This paper presents a detailed discussion of automated metrics for evaluating structured 3D reconstructions. Pitfalls of each metric are discussed, and an analysis through the lens of expert 3D modelers' preferences is presented. A set of systematic "unit tests" are proposed to empirically verify desirable properties, and context aware recommendations regarding which metric to use depending on application are provided. Finally, a learned metric distilled from human expert judgments is proposed and analyzed. The source code is available at https://github.com/s23dr/wireframe-metrics-iccv2025
>
---
#### [replaced 079] Leveraging Confident Image Regions for Source-Free Domain-Adaptive Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.10081v2](http://arxiv.org/pdf/2501.10081v2)**

> **作者:** Mohamed Lamine Mekhalfi; Davide Boscaini; Fabio Poiesi
>
> **摘要:** Source-free domain-adaptive object detection is an interesting but scarcely addressed topic. It aims at adapting a source-pretrained detector to a distinct target domain without resorting to source data during adaptation. So far, there is no data augmentation scheme tailored to source-free domain-adaptive object detection. To this end, this paper presents a novel data augmentation approach that cuts out target image regions where the detector is confident, augments them along with their respective pseudo-labels, and joins them into a challenging target image to adapt the detector. As the source data is out of reach during adaptation, we implement our approach within a teacher-student learning paradigm to ensure that the model does not collapse during the adaptation procedure. We evaluated our approach on three adaptation benchmarks of traffic scenes, scoring new state-of-the-art on two of them.
>
---
#### [replaced 080] RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20117v2](http://arxiv.org/pdf/2507.20117v2)**

> **作者:** Xiaolin Liu; Tianyi Zhou; Hongbo Kang; Jian Ma; Ziwen Wang; Jing Huang; Wenguo Weng; Yu-Kun Lai; Kun Li
>
> **摘要:** Crowd evacuation simulation is critical for enhancing public safety, and demanded for realistic virtual environments. Current mainstream evacuation models overlook the complex human behaviors that occur during evacuation, such as pedestrian collisions, interpersonal interactions, and variations in behavior influenced by terrain types or individual body shapes. This results in the failure to accurately simulate the escape of people in the real world. In this paper, aligned with the sensory-decision-motor (SDM) flow of the human brain, we propose a real-time 3D crowd evacuation simulation framework that integrates a 3D-adaptive SFM (Social Force Model) Decision Mechanism and a Personalized Gait Control Motor. This framework allows multiple agents to move in parallel and is suitable for various scenarios, with dynamic crowd awareness. Additionally, we introduce Part-level Force Visualization to assist in evacuation analysis. Experimental results demonstrate that our framework supports dynamic trajectory planning and personalized behavior for each agent throughout the evacuation process, and is compatible with uneven terrain. Visually, our method generates evacuation results that are more realistic and plausible, providing enhanced insights for crowd simulation. The code is available at http://cic.tju.edu.cn/faculty/likun/projects/RESCUE.
>
---
#### [replaced 081] Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning of Vision Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.20752v3](http://arxiv.org/pdf/2503.20752v3)**

> **作者:** Huajie Tan; Yuheng Ji; Xiaoshuai Hao; Xiansheng Chen; Pengwei Wang; Zhongyuan Wang; Shanghang Zhang
>
> **备注:** 51 pages, 23 figures, NeurIPS'25
>
> **摘要:** Visual reasoning abilities play a crucial role in understanding complex multimodal data, advancing both domain-specific applications and artificial general intelligence (AGI). Existing methods enhance Vision-Language Models (VLMs) through Chain-of-Thought (CoT) supervised fine-tuning using meticulously annotated data. However, this approach may lead to overfitting and cognitive rigidity, limiting the model's generalization ability under domain shifts and reducing real-world applicability. To overcome these limitations, we propose Reason-RFT, a two-stage reinforcement fine-tuning framework for visual reasoning. First, Supervised Fine-Tuning (SFT) with curated CoT data activates the reasoning potential of VLMs. This is followed by reinforcement learning based on Group Relative Policy Optimization (GRPO), which generates multiple reasoning-response pairs to enhance adaptability to domain shifts. To evaluate Reason-RFT, we reconstructed a comprehensive dataset covering visual counting, structural perception, and spatial transformation, serving as a benchmark for systematic assessment across three key dimensions. Experimental results highlight three advantages: (1) performance enhancement, with Reason-RFT achieving state-of-the-art results and outperforming both open-source and proprietary models; (2) generalization superiority, maintaining robust performance under domain shifts across various tasks; and (3) data efficiency, excelling in few-shot learning scenarios and surpassing full-dataset SFT baselines. Reason-RFT introduces a novel training paradigm for visual reasoning and marks a significant step forward in multimodal research. Project website: https://tanhuajie.github.io/ReasonRFT
>
---
#### [replaced 082] Law of Vision Representation in MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.16357v3](http://arxiv.org/pdf/2408.16357v3)**

> **作者:** Shijia Yang; Bohan Zhai; Quanzeng You; Jianbo Yuan; Hongxia Yang; Chenfeng Xu
>
> **备注:** The code is available at https://github.com/bronyayang/Law_of_Vision_Representation_in_MLLMs
>
> **摘要:** We present the "Law of Vision Representation" in multimodal large language models (MLLMs). It reveals a strong correlation between the combination of cross-modal alignment, correspondence in vision representation, and MLLM performance. We quantify the two factors using the cross-modal Alignment and Correspondence score (AC score). Through extensive experiments involving thirteen different vision representation settings and evaluations across eight benchmarks, we find that the AC score is linearly correlated to model performance. By leveraging this relationship, we are able to identify and train the optimal vision representation only, which does not require finetuning the language model every time, resulting in a 99.7% reduction in computational cost.
>
---
#### [replaced 083] How to build a consistency model: Learning flow maps via self-distillation
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18825v2](http://arxiv.org/pdf/2505.18825v2)**

> **作者:** Nicholas M. Boffi; Michael S. Albergo; Eric Vanden-Eijnden
>
> **备注:** NeurIPS 2025
>
> **摘要:** Flow-based generative models achieve state-of-the-art sample quality, but require the expensive solution of a differential equation at inference time. Flow map models, commonly known as consistency models, encompass many recent efforts to improve inference-time efficiency by learning the solution operator of this differential equation. Yet despite their promise, these models lack a unified description that clearly explains how to learn them efficiently in practice. Here, building on the methodology proposed in Boffi et. al. (2024), we present a systematic algorithmic framework for directly learning the flow map associated with a flow or diffusion model. By exploiting a relationship between the velocity field underlying a continuous-time flow and the instantaneous rate of change of the flow map, we show how to convert any distillation scheme into a direct training algorithm via self-distillation, eliminating the need for pre-trained teachers. We introduce three algorithmic families based on different mathematical characterizations of the flow map: Eulerian, Lagrangian, and Progressive methods, which we show encompass and extend all known distillation and direct training schemes for consistency models. We find that the novel class of Lagrangian methods, which avoid both spatial derivatives and bootstrapping from small steps by design, achieve significantly more stable training and higher performance than more standard Eulerian and Progressive schemes. Our methodology unifies existing training schemes under a single common framework and reveals new design principles for accelerated generative modeling. Associated code is available at https://github.com/nmboffi/flow-maps.
>
---
#### [replaced 084] Comprehensive Evaluation of Large Multimodal Models for Nutrition Analysis: A New Benchmark Enriched with Contextual Metadata
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07048v2](http://arxiv.org/pdf/2507.07048v2)**

> **作者:** Bruce Coburn; Jiangpeng He; Megan E. Rollo; Satvinder S. Dhaliwal; Deborah A. Kerr; Fengqing Zhu
>
> **备注:** The extended full version of the accepted paper in 2025 IEEE BHI conference with title: Evaluating Large Multimodal Models for Nutrition Analysis: A New Benchmark Enriched with Contextual Metadata. Dataset is available at: https://skynet.ecn.purdue.edu/~coburn6/ACETADA/
>
> **摘要:** Large Multimodal Models (LMMs) are increasingly applied to meal images for nutrition analysis. However, existing work primarily evaluates proprietary models, such as GPT-4. This leaves the broad range of LLMs underexplored. Additionally, the influence of integrating contextual metadata and its interaction with various reasoning modifiers remains largely uncharted. This work investigates how interpreting contextual metadata derived from GPS coordinates (converted to location/venue type), timestamps (transformed into meal/day type), and the food items present can enhance LMM performance in estimating key nutritional values. These values include calories, macronutrients (protein, carbohydrates, fat), and portion sizes. We also introduce \textbf{ACETADA}, a new food-image dataset slated for public release. This open dataset provides nutrition information verified by the dietitian and serves as the foundation for our analysis. Our evaluation across eight LMMs (four open-weight and four closed-weight) first establishes the benefit of contextual metadata integration over straightforward prompting with images alone. We then demonstrate how this incorporation of contextual information enhances the efficacy of reasoning modifiers, such as Chain-of-Thought, Multimodal Chain-of-Thought, Scale Hint, Few-Shot, and Expert Persona. Empirical results show that integrating metadata intelligently, when applied through straightforward prompting strategies, can significantly reduce the Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) in predicted nutritional values. This work highlights the potential of context-aware LMMs for improved nutrition analysis.
>
---
#### [replaced 085] A Survey of Defenses Against AI-Generated Visual Media: Detection,Disruption, and Authentication
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.10575v3](http://arxiv.org/pdf/2407.10575v3)**

> **作者:** Jingyi Deng; Chenhao Lin; Zhengyu Zhao; Shuai Liu; Zhe Peng; Qian Wang; Chao Shen
>
> **备注:** Accepted by ACM Computing Surveys
>
> **摘要:** Deep generative models have demonstrated impressive performance in various computer vision applications, including image synthesis, video generation, and medical analysis. Despite their significant advancements, these models may be used for malicious purposes, such as misinformation, deception, and copyright violation. In this paper, we provide a systematic and timely review of research efforts on defenses against AI-generated visual media, covering detection, disruption, and authentication. We review existing methods and summarize the mainstream defense-related tasks within a unified passive and proactive framework. Moreover, we survey the derivative tasks concerning the trustworthiness of defenses, such as their robustness and fairness. For each defense strategy, we formulate its general pipeline and propose a multidimensional taxonomy applicable across defense tasks, based on methodological strategies. Additionally, we summarize the commonly used evaluation datasets, criteria, and metrics. Finally, by analyzing the reviewed studies, we provide insights into current research challenges and suggest possible directions for future research.
>
---
#### [replaced 086] FVQ: A Large-Scale Dataset and an LMM-based Method for Face Video Quality Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09255v2](http://arxiv.org/pdf/2504.09255v2)**

> **作者:** Sijing Wu; Yunhao Li; Ziwen Xu; Yixuan Gao; Huiyu Duan; Wei Sun; Guangtao Zhai
>
> **备注:** Accepted by ACM MM 2025. Project page: https://github.com/wsj-sjtu/FVQ
>
> **摘要:** Face video quality assessment (FVQA) deserves to be explored in addition to general video quality assessment (VQA), as face videos are the primary content on social media platforms and human visual system (HVS) is particularly sensitive to human faces. However, FVQA is rarely explored due to the lack of large-scale FVQA datasets. To fill this gap, we present the first large-scale in-the-wild FVQA dataset, FVQ-20K, which contains 20,000 in-the-wild face videos together with corresponding mean opinion score (MOS) annotations. Along with the FVQ-20K dataset, we further propose a specialized FVQA method named FVQ-Rater to achieve human-like rating and scoring for face video, which is the first attempt to explore the potential of large multimodal models (LMMs) for the FVQA task. Concretely, we elaborately extract multi-dimensional features including spatial features, temporal features, and face-specific features (i.e., portrait features and face embeddings) to provide comprehensive visual information, and take advantage of the LoRA-based instruction tuning technique to achieve quality-specific fine-tuning, which shows superior performance on both FVQ-20K and CFVQA datasets. Extensive experiments and comprehensive analysis demonstrate the significant potential of the FVQ-20K dataset and FVQ-Rater method in promoting the development of FVQA.
>
---
#### [replaced 087] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22762v3](http://arxiv.org/pdf/2506.22762v3)**

> **作者:** Dinh Phu Tran; Dao Duy Hung; Daeyoung Kim
>
> **备注:** Arxiv version of ICCV 2025 paper (3rd version)
>
> **摘要:** Video super-resolution remains a major challenge in low-level vision tasks. To date, CNN- and Transformer-based methods have delivered impressive results. However, CNNs are limited by local receptive fields, while Transformers struggle with quadratic complexity, posing challenges for processing long sequences in VSR. Recently, Mamba has drawn attention for its long-sequence modeling, linear complexity, and large receptive fields. In this work, we propose VSRM, a novel \textbf{V}ideo \textbf{S}uper-\textbf{R}esolution framework that leverages the power of \textbf{M}amba. VSRM introduces Spatial-to-Temporal Mamba and Temporal-to-Spatial Mamba blocks to extract long-range spatio-temporal features and enhance receptive fields efficiently. To better align adjacent frames, we propose Deformable Cross-Mamba Alignment module. This module utilizes a deformable cross-mamba mechanism to make the compensation stage more dynamic and flexible, preventing feature distortions. Finally, we minimize the frequency domain gaps between reconstructed and ground-truth frames by proposing a simple yet effective Frequency Charbonnier-like loss that better preserves high-frequency content and enhances visual quality. Through extensive experiments, VSRM achieves state-of-the-art results on diverse benchmarks, establishing itself as a solid foundation for future research.
>
---
#### [replaced 088] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16416v2](http://arxiv.org/pdf/2505.16416v2)**

> **作者:** Chengcheng Wang; Jianyuan Guo; Hongguang Li; Yuchuan Tian; Ying Nie; Chang Xu; Kai Han
>
> **摘要:** Rotary Position Embedding (RoPE) is a widely adopted technique for encoding relative positional information in large language models (LLMs). However, when extended to vision-language models (VLMs), RoPE and its variants enforce relative positional dependencies separately within text and image tokens, introducing unintended cross-modal positional biases. For example, image tokens depicting semantically consistent content are assigned distinct positional encodings solely due to spatial location variations. As a result, such tokens exhibit entirely different relative positional relationships with their corresponding text tokens, ultimately leading to misaligned cross-modal representations. To address this, we propose Per-Token Distance, a simple yet effective metric for quantifying the independence of positional encodings across modalities. Informed by this analysis, we introduce Circle-RoPE, a novel encoding scheme designed to eliminate spurious cross-modal biases. Our key idea is to project image token indices onto a \emph{ring} that is orthogonal to the linear axis of text token indices, thereby forming a cone-like structure in the positional encoding space. In this configuration, each text token (point on the linear text axis) becomes the apex of a cone and maintains an equal distance to all image tokens (points on the circular image \emph{ring}), reducing artificial cross-modal biases while preserving intra-image spatial information. To further enhance performance, we propose a staggered strategy that applies different RoPE variants across layers. Extensive experiments demonstrate that our method effectively preserves spatial information from images while reducing relative positional bias, offering a more robust and flexible positional encoding framework for VLMs. The code is available at https://github.com/lose4578/CircleRoPE.
>
---
#### [replaced 089] VirDA: Reusing Backbone for Unsupervised Domain Adaptation with Visual Reprogramming
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.01660v3](http://arxiv.org/pdf/2510.01660v3)**

> **作者:** Duy Nguyen; Dat Nguyen
>
> **摘要:** Existing UDA pipelines fine-tune already well-trained backbone parameters for every new source-and-target pair, resulting in the number of training parameters and storage memory growing linearly with each new pair, and also preventing the reuse of these well-trained backbone parameters. Inspired by recent implications that existing backbones have textural biases, we propose making use of domain-specific textural bias for domain adaptation via visual reprogramming, namely VirDA. Instead of fine-tuning the full backbone, VirDA prepends a domain-specific visual reprogramming layer to the backbone. This layer produces visual prompts that act as an added textural bias to the input image, adapting its "style" to a target domain. To optimize these visual reprogramming layers, we use multiple objective functions that optimize the intra- and inter-domain distribution differences when domain-adapting visual prompts are applied. This process does not require modifying the backbone parameters, allowing the same backbone to be reused across different domains. We evaluate VirDA on Office-31 and obtain 92.8% mean accuracy with only 1.5M trainable parameters. VirDA surpasses PDA, the state-of-the-art parameter-efficient UDA baseline, by +1.6% accuracy while using just 46% of its parameters. Compared with full-backbone fine-tuning, VirDA outperforms CDTrans and FixBi by +0.2% and +1.4%, respectively, while requiring only 1.7% and 2.8% of their trainable parameters. Relative to the strongest current methods (PMTrans and TVT), VirDA uses ~1.7% of their parameters and trades off only 2.2% and 1.1% accuracy, respectively.
>
---
#### [replaced 090] Constructing a 3D Scene from a Single Image
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.15765v2](http://arxiv.org/pdf/2505.15765v2)**

> **作者:** Kaizhi Zheng; Ruijian Zha; Zishuo Xu; Jing Gu; Jie Yang; Xin Eric Wang
>
> **摘要:** Acquiring detailed 3D scenes typically demands costly equipment, multi-view data, or labor-intensive modeling. Therefore, a lightweight alternative, generating complex 3D scenes from a single top-down image, plays an essential role in real-world applications. While recent 3D generative models have achieved remarkable results at the object level, their extension to full-scene generation often leads to inconsistent geometry, layout hallucinations, and low-quality meshes. In this work, we introduce SceneFuse-3D, a training-free framework designed to synthesize coherent 3D scenes from a single top-down view. Our method is grounded in two principles: region-based generation to improve image-to-3D alignment and resolution, and spatial-aware 3D inpainting to ensure global scene coherence and high-quality geometry generation. Specifically, we decompose the input image into overlapping regions and generate each using a pretrained 3D object generator, followed by a masked rectified flow inpainting process that fills in missing geometry while maintaining structural continuity. This modular design allows us to overcome resolution bottlenecks and preserve spatial structure without requiring 3D supervision or fine-tuning. Extensive experiments across diverse scenes show that SceneFuse-3D outperforms state-of-the-art baselines, including Trellis, Hunyuan3D-2, TripoSG, and LGM, in terms of geometry quality, spatial coherence, and texture fidelity. Our results demonstrate that high-quality coherent 3D scene-level asset generation is achievable from a single top-down image using a principled, training-free pipeline.
>
---
#### [replaced 091] Capsule Network Projectors are Equivariant and Invariant Learners
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.14386v4](http://arxiv.org/pdf/2405.14386v4)**

> **作者:** Miles Everett; Aiden Durrant; Mingjun Zhong; Georgios Leontidis
>
> **备注:** V4: accepted at TMLR; V3: Ignore V1 and V2 as we have fixed a bug; 18 pages, 5 figures, 10 Tables
>
> **摘要:** Learning invariant representations has been the long-standing approach to self-supervised learning. However, recently progress has been made in preserving equivariant properties in representations, yet do so with highly prescribed architectures. In this work, we propose an invariant-equivariant self-supervised architecture that employs Capsule Networks (CapsNets), which have been shown to capture equivariance with respect to novel viewpoints. We demonstrate that the use of CapsNets in equivariant self-supervised architectures achieves improved downstream performance on equivariant tasks with higher efficiency and fewer network parameters. To accommodate the architectural changes of CapsNets, we introduce a new objective function based on entropy minimisation. This approach, which we name CapsIE (Capsule Invariant Equivariant Network), achieves state-of-the-art performance on the equivariant rotation tasks on the 3DIEBench dataset compared to prior equivariant SSL methods, while performing competitively against supervised counterparts. Our results demonstrate the ability of CapsNets to learn complex and generalised representations for large-scale, multi-task datasets compared to previous CapsNet benchmarks. Code is available at https://github.com/AberdeenML/CapsIE.
>
---
#### [replaced 092] ShapeICP: Iterative Category-level Object Pose and Shape Estimation from Depth
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2408.13147v3](http://arxiv.org/pdf/2408.13147v3)**

> **作者:** Yihao Zhang; Harpreet S. Sawhney; John J. Leonard
>
> **摘要:** Category-level object pose and shape estimation from a single depth image has recently drawn research attention due to its potential utility for tasks such as robotics manipulation. The task is particularly challenging because the three unknowns, object pose, object shape, and model-to-measurement correspondences, are compounded together, but only a single view of depth measurements is provided. Most of the prior work heavily relies on data-driven approaches to obtain solutions to at least one of the unknowns, and typically two, risking generalization failures if not designed and trained carefully. The shape representations used in the prior work also mainly focus on point clouds and signed distance fields (SDFs). In stark contrast to the prior work, we approach the problem using an iterative estimation method that does not require learning from pose-annotated data. Moreover, we construct and adopt a novel mesh-based object active shape model (ASM), which additionally maintains vertex connectivity compared to the commonly used point-based object ASM. Our algorithm, ShapeICP, is based on the iterative closest point (ICP) algorithm but is equipped with additional features for the category-level pose and shape estimation task. Although not using pose-annotated data, ShapeICP surpasses many data-driven approaches that rely on pose data for training, opening up a new solution space for researchers to consider.
>
---
#### [replaced 093] Do Vision-Language Models See Urban Scenes as People Do? An Urban Perception Benchmark
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.14574v2](http://arxiv.org/pdf/2509.14574v2)**

> **作者:** Rashid Mushkani
>
> **摘要:** Understanding how people read city scenes can inform design and planning. We introduce a small benchmark for testing vision-language models (VLMs) on urban perception using 100 Montreal street images, evenly split between photographs and photorealistic synthetic scenes. Twelve participants from seven community groups supplied 230 annotation forms across 30 dimensions mixing physical attributes and subjective impressions. French responses were normalized to English. We evaluated seven VLMs in a zero-shot setup with a structured prompt and deterministic parser. We use accuracy for single-choice items and Jaccard overlap for multi-label items; human agreement uses Krippendorff's alpha and pairwise Jaccard. Results suggest stronger model alignment on visible, objective properties than subjective appraisals. The top system (claude-sonnet) reaches macro 0.31 and mean Jaccard 0.48 on multi-label items. Higher human agreement coincides with better model scores. Synthetic images slightly lower scores. We release the benchmark, prompts, and harness for reproducible, uncertainty-aware evaluation in participatory urban analysis.
>
---
#### [replaced 094] RealKIE: Five Novel Datasets for Enterprise Key Information Extraction
- **分类: cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.20101v2](http://arxiv.org/pdf/2403.20101v2)**

> **作者:** Benjamin Townsend; Madison May; Katherine Mackowiak; Christopher Wells
>
> **摘要:** We introduce RealKIE, a benchmark of five challenging datasets aimed at advancing key information extraction methods, with an emphasis on enterprise applications. The datasets include a diverse range of documents including SEC S1 Filings, US Non-disclosure Agreements, UK Charity Reports, FCC Invoices, and Resource Contracts. Each presents unique challenges: poor text serialization, sparse annotations in long documents, and complex tabular layouts. These datasets provide a realistic testing ground for key information extraction tasks like investment analysis and contract analysis. In addition to presenting these datasets, we offer an in-depth description of the annotation process, document processing techniques, and baseline modeling approaches. This contribution facilitates the development of NLP models capable of handling practical challenges and supports further research into information extraction technologies applicable to industry-specific problems. The annotated data, OCR outputs, and code to reproduce baselines are available to download at https://indicodatasolutions.github.io/RealKIE/.
>
---
#### [replaced 095] DiViD: Disentangled Video Diffusion for Static-Dynamic Factorization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13934v2](http://arxiv.org/pdf/2507.13934v2)**

> **作者:** Marzieh Gheisari; Auguste Genovesio
>
> **摘要:** Unsupervised disentanglement of static appearance and dynamic motion in video remains a fundamental challenge, often hindered by information leakage and blurry reconstructions in existing VAE- and GAN-based approaches. We introduce DiViD, the first end-to-end video diffusion framework for explicit static-dynamic factorization. DiViD's sequence encoder extracts a global static token from the first frame and per-frame dynamic tokens, explicitly removing static content from the motion code. Its conditional DDPM decoder incorporates three key inductive biases: a shared-noise schedule for temporal consistency, a time-varying KL-based bottleneck that tightens at early timesteps (compressing static information) and relaxes later (enriching dynamics), and cross-attention that routes the global static token to all frames while keeping dynamic tokens frame-specific. An orthogonality regularizer further prevents residual static-dynamic leakage. We evaluate DiViD on real-world benchmarks using swap-based accuracy and cross-leakage metrics. DiViD outperforms state-of-the-art sequential disentanglement methods: it achieves the highest swap-based joint accuracy, preserves static fidelity while improving dynamic transfer, and reduces average cross-leakage.
>
---
#### [replaced 096] Smaller is Better: Enhancing Transparency in Vehicle AI Systems via Pruning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20148v2](http://arxiv.org/pdf/2509.20148v2)**

> **作者:** Sanish Suwal; Shaurya Garg; Dipkamal Bhusal; Michael Clifford; Nidhi Rastogi
>
> **备注:** 17 pages
>
> **摘要:** Connected and autonomous vehicles continue to heavily rely on AI systems, where transparency and security are critical for trust and operational safety. Post-hoc explanations provide transparency to these black-box like AI models but the quality and reliability of these explanations is often questioned due to inconsistencies and lack of faithfulness in representing model decisions. This paper systematically examines the impact of three widely used training approaches, namely natural training, adversarial training, and pruning, affect the quality of post-hoc explanations for traffic sign classifiers. Through extensive empirical evaluation, we demonstrate that pruning significantly enhances the comprehensibility and faithfulness of explanations (using saliency maps). Our findings reveal that pruning not only improves model efficiency but also enforces sparsity in learned representation, leading to more interpretable and reliable decisions. Additionally, these insights suggest that pruning is a promising strategy for developing transparent deep learning models, especially in resource-constrained vehicular AI systems.
>
---
#### [replaced 097] Filling of incomplete sinograms from sparse PET detector configurations using a residual U-Net
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2506.19600v2](http://arxiv.org/pdf/2506.19600v2)**

> **作者:** Klara Leffler; Luigi Tommaso Luppino; Samuel Kuttner; Karin Söderkvist; Jan Axelsson
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Long axial field-of-view PET scanners offer increased field-of-view and sensitivity compared to traditional PET scanners. However, a significant cost is associated with the densely packed photodetectors required for the extended-coverage systems, limiting clinical utilisation. To mitigate the cost limitations, alternative sparse system configurations have been proposed, allowing an extended field-of-view PET design with detector costs similar to a standard PET system, albeit at the expense of image quality. In this work, we propose a deep sinogram restoration network to fill in the missing sinogram data. Our method utilises a modified Residual U-Net, trained on clinical PET scans from a GE Signa PET/MR, simulating the removal of 50% of the detectors in a chessboard pattern (retaining only 25% of all lines of response). The model successfully recovers missing counts, with a mean absolute error below two events per pixel, outperforming 2D interpolation in both sinogram and reconstructed image domain. Notably, the predicted sinograms exhibit a smoothing effect, leading to reconstructed images lacking sharpness in finer details. Despite these limitations, the model demonstrates a substantial capacity for compensating for the undersampling caused by the sparse detector configuration. This proof-of-concept study suggests that sparse detector configurations, combined with deep learning techniques, offer a viable alternative to conventional PET scanner designs. This approach supports the development of cost-effective, total body PET scanners, allowing a significant step forward in medical imaging technology.
>
---
#### [replaced 098] UGC-VideoCaptioner: An Omni UGC Video Detail Caption Model and New Benchmarks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11336v2](http://arxiv.org/pdf/2507.11336v2)**

> **作者:** Peiran Wu; Yunze Liu; Zhengdong Zhu; Enmin Zhou; Junxiao Shen
>
> **摘要:** Real-world user-generated videos, especially on platforms like TikTok, often feature rich and intertwined audio visual content. However, existing video captioning benchmarks and models remain predominantly visual centric, overlooking the crucial role of audio in conveying scene dynamics, speaker intent, and narrative context. This lack of omni datasets and lightweight, capable models hampers progress in fine grained, multimodal video understanding. To address these challenges, we introduce UGC-VideoCap, a new benchmark and model framework specifically designed for detailed omnimodal captioning of short form user-generated videos. Unlike prior datasets, UGC-VideoCap emphasizes balanced integration of audio and visual modalities, featuring 1000 TikTok videos annotated through a structured three stage human-in-the-loop pipeline covering audio only, visual only, and joint audio visual semantics. The benchmark also includes 4000 carefully crafted QA pairs probing both unimodal and cross modal understanding. Alongside the dataset, we propose UGC-VideoCaptioner(3B), a 3B parameter captioning model distilled from Gemini 2.5 Flash. Using a novel two-stage training strategy supervised fine tuning followed by Group Relative Policy Optimization (GRPO), our approach enables efficient adaptation from limited data while maintaining competitive performance. Together, our benchmark and model offer a high-quality foundation and a data-efficient solution for advancing omnimodal video captioning in unconstrained real-world UGC settings.
>
---
#### [replaced 099] FEB-Cache: Frequency-Guided Exposure Bias Reduction for Enhancing Diffusion Transformer Caching
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.07120v3](http://arxiv.org/pdf/2503.07120v3)**

> **作者:** Zhen Zou; Feng Zhao
>
> **摘要:** Diffusion Transformer (DiT) has exhibited impressive generation capabilities but faces great challenges due to its high computational complexity. To address this issue, various methods, notably feature caching, have been introduced. However, these approaches focus on aligning non-cache diffusion without analyzing why caching damage the generation processes. In this paper, we first confirm that the cache greatly amplifies the exposure bias, resulting in a decline in the generation quality. However, directly applying noise scaling is challenging for this issue due to the non-smoothness of exposure bias. We found that this phenomenon stems from the mismatch between its frequency response characteristics and the simple cache of Attention and MLP. Since these two components exhibit unique preferences for frequency signals, which provides us with a caching strategy to separate Attention and MLP to achieve an enhanced fit of exposure bias and reduce it. Based on this, we introduced FEB-Cache, a joint caching strategy that aligns with the non-exposed bias diffusion process (which gives us a higher performance cap) of caching Attention and MLP based on the frequency-guided cache table. Our approach combines a comprehensive understanding of the caching mechanism and offers a new perspective on leveraging caching to accelerate the diffusion process. Empirical results indicate that FEB-Cache optimizes model performance while concurrently facilitating acceleration. Code is available at https://github.com/aSleepyTree/EB-Cache.
>
---
#### [replaced 100] Evaluating Perceptual Distance Models by Fitting Binomial Distributions to Two-Alternative Forced Choice Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.10390v4](http://arxiv.org/pdf/2403.10390v4)**

> **作者:** Alexander Hepburn; Raul Santos-Rodriguez; Javier Portilla
>
> **摘要:** The Two Alternative Forced Choice (2AFC) paradigm offers advantages over the Mean Opinion Score (MOS) paradigm in psychophysics (PF), such as simplicity and robustness. However, when evaluating perceptual distance models, MOS enables direct correlation between model predictions and PF data. In contrast, 2AFC only allows pairwise comparisons to be converted into a quality ranking similar to MOS when comparisons include shared images. In large datasets, like BAPPS, where image patches and distortions are combined randomly, deriving rankings from 2AFC PF data becomes infeasible, as distorted images included in each comparisons are independent. To address this, instead of relying on MOS correlation, researchers have trained ad-hoc neural networks to reproduce 2AFC PF data based on pairs of model distances - a black-box approach with conceptual and operational limitations. This paper introduces a more robust distance-model evaluation method using a pure probabilistic approach, applying maximum likelihood estimation to a binomial decision model. Our method demonstrates superior simplicity, interpretability, flexibility, and computational efficiency, as shown through evaluations of various visual distance models on two 2AFC PF datasets.
>
---
#### [replaced 101] Do Sparse Subnetworks Exhibit Cognitively Aligned Attention? Effects of Pruning on Saliency Map Fidelity, Sparsity, and Concept Coherence
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.21387v3](http://arxiv.org/pdf/2509.21387v3)**

> **作者:** Sanish Suwal; Dipkamal Bhusal; Michael Clifford; Nidhi Rastogi
>
> **备注:** 4 pages, neurips workshop
>
> **摘要:** Prior works have shown that neural networks can be heavily pruned while preserving performance, but the impact of pruning on model interpretability remains unclear. In this work, we investigate how magnitude-based pruning followed by fine-tuning affects both low-level saliency maps and high-level concept representations. Using a ResNet-18 trained on ImageNette, we compare post-hoc explanations from Vanilla Gradients (VG) and Integrated Gradients (IG) across pruning levels, evaluating sparsity and faithfulness. We further apply CRAFT-based concept extraction to track changes in semantic coherence of learned concepts. Our results show that light-to-moderate pruning improves saliency-map focus and faithfulness while retaining distinct, semantically meaningful concepts. In contrast, aggressive pruning merges heterogeneous features, reducing saliency map sparsity and concept coherence despite maintaining accuracy. These findings suggest that while pruning can shape internal representations toward more human-aligned attention patterns, excessive pruning undermines interpretability.
>
---
#### [replaced 102] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.25787v2](http://arxiv.org/pdf/2509.25787v2)**

> **作者:** Wen Wen; Tianwu Zhi; Kanglong Fan; Yang Li; Xinge Peng; Yabin Zhang; Yiting Liao; Junlin Li; Li Zhang
>
> **备注:** Technical Report
>
> **摘要:** Improving vision-language models (VLMs) in the post-training stage typically relies on supervised fine-tuning or reinforcement learning, methods that necessitate costly, human-annotated data. While self-supervised techniques such as self-consistency have proven effective for enhancing reasoning capabilities, their application to perceptual domains such as image quality assessment (IQA) remains largely unexplored. In this work, we introduce EvoQuality, a novel framework that enables a VLM to autonomously refine its quality perception capabilities without any ground-truth labels. EvoQuality adapts the principle of self-consistency to the ranking-based nature of IQA. It generates pseudo-labels by performing pairwise majority voting on the VLM's own outputs to establish a consensus on relative quality. These pseudo-rankings are then formulated into a fidelity reward that guides the model's iterative evolution through group relative policy optimization (GRPO). By iteratively leveraging its own predictions, EvoQuality progressively refines the VLM's perceptual capability. Extensive experiments show that EvoQuality boosts the base VLM's zero-shot performance by 31.8\% on PLCC across diverse IQA benchmarks. Remarkably, despite being entirely self-supervised, EvoQuality achieves performance that is competitive with, or even surpasses, state-of-the-art supervised VLM-based IQA models, outperforming these models on 5 out of 7 IQA benchmarks.
>
---
#### [replaced 103] OpenCUA: Open Foundations for Computer-Use Agents
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09123v3](http://arxiv.org/pdf/2508.09123v3)**

> **作者:** Xinyuan Wang; Bowen Wang; Dunjie Lu; Junlin Yang; Tianbao Xie; Junli Wang; Jiaqi Deng; Xiaole Guo; Yiheng Xu; Chen Henry Wu; Zhennan Shen; Zhuokai Li; Ryan Li; Xiaochuan Li; Junda Chen; Boyuan Zheng; Peihang Li; Fangyu Lei; Ruisheng Cao; Yeqiao Fu; Dongchan Shin; Martin Shin; Jiarui Hu; Yuyan Wang; Jixuan Chen; Yuxiao Ye; Danyang Zhang; Dikang Du; Hao Hu; Huarong Chen; Zaida Zhou; Haotian Yao; Ziwei Chen; Qizheng Gu; Yipu Wang; Heng Wang; Diyi Yang; Victor Zhong; Flood Sung; Y. Charles; Zhilin Yang; Tao Yu
>
> **备注:** Updata author list, modify first page format, correct typos
>
> **摘要:** Vision-language models have demonstrated impressive capabilities as computer-use agents (CUAs) capable of automating diverse computer tasks. As their commercial potential grows, critical details of the most capable CUA systems remain closed. As these agents will increasingly mediate digital interactions and execute consequential decisions on our behalf, the research community needs access to open CUA frameworks to study their capabilities, limitations, and risks. To bridge this gap, we propose OpenCUA, a comprehensive open-source framework for scaling CUA data and foundation models. Our framework consists of: (1) an annotation infrastructure that seamlessly captures human computer-use demonstrations; (2) AgentNet, the first large-scale computer-use task dataset spanning 3 operating systems and 200+ applications and websites; (3) a scalable pipeline that transforms demonstrations into state-action pairs with reflective long Chain-of-Thought reasoning that sustain robust performance gains as data scales. Our end-to-end agent models demonstrate strong performance across CUA benchmarks. In particular, OpenCUA-72B achieves an average success rate of 45.0% on OSWorld-Verified, establishing a new state-of-the-art (SOTA) among open-source models. Further analysis confirms that our approach generalizes well across domains and benefits significantly from increased test-time computation. We release our annotation tool, datasets, code, and models to build open foundations for further CUA research.
>
---
#### [replaced 104] AutoMiSeg: Automatic Medical Image Segmentation via Test-Time Adaptation of Foundation Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.17931v2](http://arxiv.org/pdf/2505.17931v2)**

> **作者:** Xingjian Li; Qifeng Wu; Adithya S. Ubaradka; Yiran Ding; Colleen Que; Runmin Jiang; Jianhua Xing; Tianyang Wang; Min Xu
>
> **摘要:** Medical image segmentation is vital for clinical diagnosis, yet current deep learning methods often demand extensive expert effort, i.e., either through annotating large training datasets or providing prompts at inference time for each new case. This paper introduces a zero-shot and automatic segmentation pipeline that combines off-the-shelf vision-language and segmentation foundation models. Given a medical image and a task definition (e.g., "segment the optic disc in an eye fundus image"), our method uses a grounding model to generate an initial bounding box, followed by a visual prompt boosting module that enhance the prompts, which are then processed by a promptable segmentation model to produce the final mask. To address the challenges of domain gap and result verification, we introduce a test-time adaptation framework featuring a set of learnable adaptors that align the medical inputs with foundation model representations. Its hyperparameters are optimized via Bayesian Optimization, guided by a proxy validation model without requiring ground-truth labels. Our pipeline offers an annotation-efficient and scalable solution for zero-shot medical image segmentation across diverse tasks. Our pipeline is evaluated on seven diverse medical imaging datasets and shows promising results. By proper decomposition and test-time adaptation, our fully automatic pipeline not only substantially surpasses the previously best-performing method, yielding a 69\% relative improvement in accuracy (Dice Score from 42.53 to 71.81), but also performs competitively with weakly-prompted interactive foundation models.
>
---
#### [replaced 105] Towards Foundation Models for Cryo-ET Subtomogram Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.24311v2](http://arxiv.org/pdf/2509.24311v2)**

> **作者:** Runmin Jiang; Wanyue Feng; Yuntian Yang; Shriya Pingulkar; Hong Wang; Xi Xiao; Xiaoyu Cao; Genpei Zhang; Xiao Wang; Xiaolong Wu; Tianyang Wang; Yang Liu; Xingjian Li; Min Xu
>
> **摘要:** Cryo-electron tomography (cryo-ET) enables in situ visualization of macromolecular structures, where subtomogram analysis tasks such as classification, alignment, and averaging are critical for structural determination. However, effective analysis is hindered by scarce annotations, severe noise, and poor generalization. To address these challenges, we take the first step towards foundation models for cryo-ET subtomograms. First, we introduce CryoEngine, a large-scale synthetic data generator that produces over 904k subtomograms from 452 particle classes for pretraining. Second, we design an Adaptive Phase Tokenization-enhanced Vision Transformer (APT-ViT), which incorporates adaptive phase tokenization as an equivariance-enhancing module that improves robustness to both geometric and semantic variations. Third, we introduce a Noise-Resilient Contrastive Learning (NRCL) strategy to stabilize representation learning under severe noise conditions. Evaluations across 24 synthetic and real datasets demonstrate state-of-the-art (SOTA) performance on all three major subtomogram tasks and strong generalization to unseen datasets, advancing scalable and robust subtomogram analysis in cryo-ET.
>
---
#### [replaced 106] Probabilistic Language-Image Pre-Training
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.18857v4](http://arxiv.org/pdf/2410.18857v4)**

> **作者:** Sanghyuk Chun; Wonjae Kim; Song Park; Sangdoo Yun
>
> **备注:** Code: https://github.com/naver-ai/prolip HuggingFace Hub: https://huggingface.co/collections/SanghyukChun/prolip-6712595dfc87fd8597350291 33 pages, 4.5 MB; LongProLIP paper: arXiv:2503.08048; Multiplicity paper for more background: arxiv.org:2505.19614; v4: fix typos
>
> **摘要:** Vision-language models (VLMs) embed aligned image-text pairs into a joint space but often rely on deterministic embeddings, assuming a one-to-one correspondence between images and texts. This oversimplifies real-world relationships, which are inherently many-to-many, with multiple captions describing a single image and vice versa. We introduce Probabilistic Language-Image Pre-training (ProLIP), the first probabilistic VLM pre-trained on a billion-scale image-text dataset using only probabilistic objectives, achieving a strong zero-shot capability (e.g., 74.6% ImageNet zero-shot accuracy with ViT-B/16). ProLIP efficiently estimates uncertainty by an "uncertainty token" without extra parameters. We also introduce a novel inclusion loss that enforces distributional inclusion relationships between image-text pairs and between original and masked inputs. Experiments demonstrate that, by leveraging uncertainty estimates, ProLIP benefits downstream tasks and aligns with intuitive notions of uncertainty, e.g., shorter texts being more uncertain and more general inputs including specific ones. Utilizing text uncertainties, we further improve ImageNet accuracy from 74.6% to 75.8% (under a few-shot setting), supporting the practical advantages of our probabilistic approach. The code is available at https://github.com/naver-ai/prolip
>
---
#### [replaced 107] Latent Visual Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.24251v2](http://arxiv.org/pdf/2509.24251v2)**

> **作者:** Bangzheng Li; Ximeng Sun; Jiang Liu; Ze Wang; Jialian Wu; Xiaodong Yu; Hao Chen; Emad Barsoum; Muhao Chen; Zicheng Liu
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved notable gains in various tasks by incorporating Chain-of-Thought (CoT) reasoning in language spaces. Recent work extends this direction by leveraging external tools for visual editing, thereby enhancing the visual signal along the reasoning trajectories. Nevertheless, these approaches remain fundamentally constrained: reasoning is still confined to the language space, with visual information treated as static preconditions. We introduce Latent Visual Reasoning (LVR), a new paradigm that enables autoregressive reasoning directly in the visual embedding space. A visual encoder first projects images into visual tokens within a joint semantic space shared with the language model. The language model is then trained to generate latent states that reconstruct key visual tokens critical for answering the query, constituting the process of latent visual reasoning. By interleaving LVR with standard text generation, our model achieves substantial gains on perception-intensive visual question answering tasks. In addition, we adapt the GRPO algorithm to conduct reinforcement learning on latent reasoning, further balancing LVR and textual generation. We show that LVR substantially improves fine-grained visual understanding and perception, achieving 71.67% on MMVP compared to 66.67% with Qwen2.5-VL. Code base and model weights will be released later.
>
---
#### [replaced 108] A Neurosymbolic Agent System for Compositional Visual Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07778v3](http://arxiv.org/pdf/2506.07778v3)**

> **作者:** Yichang Xu; Gaowen Liu; Ramana Rao Kompella; Sihao Hu; Fatih Ilhan; Selim Furkan Tekin; Zachary Yahn; Ling Liu
>
> **摘要:** The advancement in large language models (LLMs) and large vision models has fueled the rapid progress in multi-modal vision-language reasoning capabilities. However, existing vision-language models (VLMs) remain challenged by compositional visual reasoning. This paper presents VLAgent, a neuro-symbolic approach to developing a Vision-Language Agent system for efficient compositional visual reasoning with three novel features. First, VLAgent develops an interpretable visualization-enhanced two-stage neuro-symbolic reasoning system. The first stage is managed by a front-end engine that generates a structured visual reasoning plan (symbolic program script) for each compositional visual reasoning task by utilizing a pre-trained LLM powered with few-shot chain-of-thought in-context learning. The second stage is managed by a high-performance back-end engine. It transforms the planning script into executable code based on visual input (image or video) and the combination of neural models and symbolic functions and then performs a sequence of actions for the compositional visual reason task. Second, to ensure and enhance the quality of mapping the logic plan to a sequence of executable instructions, VLAgent introduces the SS-parser, which examines the syntax and semantic correctness of the planning script, detects and repairs the logic errors found in the LLM-generated logic plan before generating the executable program. Third, VLAgent introduces the execution verifier in critical reasoning steps to validate and refine its compositional reasoning results in a stepwise manner, for example, ensemble methods for critical visual reasoning and caption analysis for low-confidence compositional reasoning. Extensive experiments on six visual benchmarks compared to a dozen SoTA visual reasoning models show that VLAgent outperforms existing representative approaches to compositional visual reasoning.
>
---
#### [replaced 109] Towards Cross-modal Backward-compatible Representation Learning for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.14715v3](http://arxiv.org/pdf/2405.14715v3)**

> **作者:** Young Kyun Jang; Ser-nam Lim
>
> **摘要:** Modern retrieval systems often struggle with upgrading to new and more powerful models due to the incompatibility of embeddings between the old and new models. This necessitates a costly process known as backfilling, which involves re-computing the embeddings for a large number of data samples. In vision, Backward-compatible Training (BT) has been proposed to ensure that the new model aligns with the old model's embeddings. This paper extends the concept of vision-only BT to the field of cross-modal retrieval, marking the first attempt to address Cross-modal BT (XBT). Our goal is to achieve backward-compatibility between Vision-Language Pretraining (VLP) models, such as CLIP, for the cross-modal retrieval task. To address XBT challenges, we propose an efficient solution: a projection module that maps the new model's embeddings to those of the old model. This module, pretrained solely with text data, significantly reduces the number of image-text pairs required for XBT learning, and, once it is pretrained, it avoids using the old model during training. Furthermore, we utilize parameter-efficient training strategies that improve efficiency and preserve the off-the-shelf new model's knowledge by avoiding any modifications. Experimental results on cross-modal retrieval datasets demonstrate the effectiveness of XBT and its potential to enable backfill-free upgrades when a new VLP model emerges.
>
---
#### [replaced 110] CoralSCOP-LAT: Labeling and Analyzing Tool for Coral Reef Images with Dense Mask
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.20436v2](http://arxiv.org/pdf/2410.20436v2)**

> **作者:** Yuk-Kwan Wong; Ziqiang Zheng; Mingzhe Zhang; David Suggett; Sai-Kit Yeung
>
> **备注:** Ecological Informatics Page: https://www.sciencedirect.com/science/article/pii/S157495412500411X
>
> **摘要:** Coral reef imagery offers critical data for monitoring ecosystem health, in particular as the ease of image datasets continues to rapidly expand. Whilst semi-automated analytical platforms for reef imagery are becoming more available, the dominant approaches face fundamental limitations. To address these challenges, we propose CoralSCOP-LAT, a coral reef image analysis and labeling tool that automatically segments and analyzes coral regions. By leveraging advanced machine learning models tailored for coral reef segmentation, CoralSCOP-LAT enables users to generate dense segmentation masks with minimal manual effort, significantly enhancing both the labeling efficiency and precision of coral reef analysis. Our extensive evaluations demonstrate that CoralSCOP-LAT surpasses existing coral reef analysis tools in terms of time efficiency, accuracy, precision, and flexibility. CoralSCOP-LAT, therefore, not only accelerates the coral reef annotation process but also assists users in obtaining high-quality coral reef segmentation and analysis outcomes. Github Page: https://github.com/ykwongaq/CoralSCOP-LAT.
>
---
#### [replaced 111] Generating Findings for Jaw Cysts in Dental Panoramic Radiographs Using GPT-4o: Building a Two-Stage Self-Correction Loop with Structured Output (SLSO) Framework
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.02001v2](http://arxiv.org/pdf/2510.02001v2)**

> **作者:** Nanaka Hosokawa; Ryo Takahashi; Tomoya Kitano; Yukihiro Iida; Chisako Muramatsu; Tatsuro Hayashi; Yuta Seino; Xiangrong Zhou; Takeshi Hara; Akitoshi Katsumata; Hiroshi Fujita
>
> **备注:** Submitted to Scientific Reports
>
> **摘要:** In this study, we utilized the multimodal capabilities of OpenAI GPT-4o to automatically generate jaw cyst findings on dental panoramic radiographs. To improve accuracy, we constructed a Self-correction Loop with Structured Output (SLSO) framework and verified its effectiveness. A 10-step process was implemented for 22 cases of jaw cysts, including image input and analysis, structured data generation, tooth number extraction and consistency checking, iterative regeneration when inconsistencies were detected, and finding generation with subsequent restructuring and consistency verification. A comparative experiment was conducted using the conventional Chain-of-Thought (CoT) method across seven evaluation items: transparency, internal structure, borders, root resorption, tooth movement, relationships with other structures, and tooth number. The results showed that the proposed SLSO framework improved output accuracy for many items, with 66.9%, 33.3%, and 28.6% improvement rates for tooth number, tooth movement, and root resorption, respectively. In the successful cases, a consistently structured output was achieved after up to five regenerations. Although statistical significance was not reached because of the small size of the dataset, the overall SLSO framework enforced negative finding descriptions, suppressed hallucinations, and improved tooth number identification accuracy. However, the accurate identification of extensive lesions spanning multiple teeth is limited. Nevertheless, further refinement is required to enhance overall performance and move toward a practical finding generation system.
>
---
#### [replaced 112] FreeInsert: Disentangled Text-Guided Object Insertion in 3D Gaussian Scene without Spatial Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01322v3](http://arxiv.org/pdf/2505.01322v3)**

> **作者:** Chenxi Li; Weijie Wang; Qiang Li; Bruno Lepri; Nicu Sebe; Weizhi Nie
>
> **备注:** Accepted by ACMMM2025
>
> **摘要:** Text-driven object insertion in 3D scenes is an emerging task that enables intuitive scene editing through natural language. However, existing 2D editing-based methods often rely on spatial priors such as 2D masks or 3D bounding boxes, and they struggle to ensure consistency of the inserted object. These limitations hinder flexibility and scalability in real-world applications. In this paper, we propose FreeInsert, a novel framework that leverages foundation models including MLLMs, LGMs, and diffusion models to disentangle object generation from spatial placement. This enables unsupervised and flexible object insertion in 3D scenes without spatial priors. FreeInsert starts with an MLLM-based parser that extracts structured semantics, including object types, spatial relationships, and attachment regions, from user instructions. These semantics guide both the reconstruction of the inserted object for 3D consistency and the learning of its degrees of freedom. We leverage the spatial reasoning capabilities of MLLMs to initialize object pose and scale. A hierarchical, spatially aware refinement stage further integrates spatial semantics and MLLM-inferred priors to enhance placement. Finally, the appearance of the object is improved using the inserted-object image to enhance visual fidelity. Experimental results demonstrate that FreeInsert achieves semantically coherent, spatially precise, and visually realistic 3D insertions without relying on spatial priors, offering a user-friendly and flexible editing experience.
>
---
#### [replaced 113] MapIQ: Evaluating Multimodal Large Language Models for Map Question Answering
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11625v2](http://arxiv.org/pdf/2507.11625v2)**

> **作者:** Varun Srivastava; Fan Lei; Srija Mukhopadhyay; Vivek Gupta; Ross Maciejewski
>
> **备注:** Published as a conference paper at COLM 2025
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have driven researchers to explore how well these models read data visualizations, e.g., bar charts, scatter plots. More recently, attention has shifted to visual question answering with maps (Map-VQA). However, Map-VQA research has primarily focused on choropleth maps, which cover only a limited range of thematic categories and visual analytical tasks. To address these gaps, we introduce MapIQ, a benchmark dataset comprising 14,706 question-answer pairs across three map types: choropleth maps, cartograms, and proportional symbol maps spanning topics from six distinct themes (e.g., housing, crime). We evaluate multiple MLLMs using six visual analytical tasks, comparing their performance against one another and a human baseline. An additional experiment examining the impact of map design changes (e.g., altered color schemes, modified legend designs, and removal of map elements) provides insights into the robustness and sensitivity of MLLMs, their reliance on internal geographic knowledge, and potential avenues for improving Map-VQA performance.
>
---
#### [replaced 114] Mixture of Contexts for Long Video Generation
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.21058v2](http://arxiv.org/pdf/2508.21058v2)**

> **作者:** Shengqu Cai; Ceyuan Yang; Lvmin Zhang; Yuwei Guo; Junfei Xiao; Ziyan Yang; Yinghao Xu; Zhenheng Yang; Alan Yuille; Leonidas Guibas; Maneesh Agrawala; Lu Jiang; Gordon Wetzstein
>
> **备注:** Project page: https://primecai.github.io/moc/
>
> **摘要:** Long video generation is fundamentally a long context memory problem: models must retain and retrieve salient events across a long range without collapsing or drifting. However, scaling diffusion transformers to generate long-context videos is fundamentally limited by the quadratic cost of self-attention, which makes memory and computation intractable and difficult to optimize for long sequences. We recast long-context video generation as an internal information retrieval task and propose a simple, learnable sparse attention routing module, Mixture of Contexts (MoC), as an effective long-term memory retrieval engine. In MoC, each query dynamically selects a few informative chunks plus mandatory anchors (caption, local windows) to attend to, with causal routing that prevents loop closures. As we scale the data and gradually sparsify the routing, the model allocates compute to salient history, preserving identities, actions, and scenes over minutes of content. Efficiency follows as a byproduct of retrieval (near-linear scaling), which enables practical training and synthesis, and the emergence of memory and consistency at the scale of minutes.
>
---
#### [replaced 115] STIV: Scalable Text and Image Conditioned Video Generation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2412.07730v2](http://arxiv.org/pdf/2412.07730v2)**

> **作者:** Zongyu Lin; Wei Liu; Chen Chen; Jiasen Lu; Wenze Hu; Tsu-Jui Fu; Jesse Allardice; Zhengfeng Lai; Liangchen Song; Bowen Zhang; Cha Chen; Yiran Fei; Lezhi Li; Yizhou Sun; Kai-Wei Chang; Yinfei Yang
>
> **摘要:** The field of video generation has made remarkable advancements, yet there remains a pressing need for a clear, systematic recipe that can guide the development of robust and scalable models. In this work, we present a comprehensive study that systematically explores the interplay of model architectures, training recipes, and data curation strategies, culminating in a simple and scalable text-image-conditioned video generation method, named STIV. Our framework integrates image condition into a Diffusion Transformer (DiT) through frame replacement, while incorporating text conditioning via a joint image-text conditional classifier-free guidance. This design enables STIV to perform both text-to-video (T2V) and text-image-to-video (TI2V) tasks simultaneously. Additionally, STIV can be easily extended to various applications, such as video prediction, frame interpolation, multi-view generation, and long video generation, etc. With comprehensive ablation studies on T2I, T2V, and TI2V, STIV demonstrate strong performance, despite its simple design. An 8.7B model with 512 resolution achieves 83.1 on VBench T2V, surpassing both leading open and closed-source models like CogVideoX-5B, Pika, Kling, and Gen-3. The same-sized model also achieves a state-of-the-art result of 90.1 on VBench I2V task at 512 resolution. By providing a transparent and extensible recipe for building cutting-edge video generation models, we aim to empower future research and accelerate progress toward more versatile and reliable video generation solutions.
>
---
#### [replaced 116] OracleGS: Grounding Generative Priors for Sparse-View Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23258v2](http://arxiv.org/pdf/2509.23258v2)**

> **作者:** Atakan Topaloglu; Kunyi Li; Michael Niemeyer; Nassir Navab; A. Murat Tekalp; Federico Tombari
>
> **备注:** Project page available at: https://atakan-topaloglu.github.io/oraclegs/
>
> **摘要:** Sparse-view novel view synthesis is fundamentally ill-posed due to severe geometric ambiguity. Current methods are caught in a trade-off: regressive models are geometrically faithful but incomplete, whereas generative models can complete scenes but often introduce structural inconsistencies. We propose OracleGS, a novel framework that reconciles generative completeness with regressive fidelity for sparse view Gaussian Splatting. Instead of using generative models to patch incomplete reconstructions, our "propose-and-validate" framework first leverages a pre-trained 3D-aware diffusion model to synthesize novel views to propose a complete scene. We then repurpose a multi-view stereo (MVS) model as a 3D-aware oracle to validate the 3D uncertainties of generated views, using its attention maps to reveal regions where the generated views are well-supported by multi-view evidence versus where they fall into regions of high uncertainty due to occlusion, lack of texture, or direct inconsistency. This uncertainty signal directly guides the optimization of a 3D Gaussian Splatting model via an uncertainty-weighted loss. Our approach conditions the powerful generative prior on multi-view geometric evidence, filtering hallucinatory artifacts while preserving plausible completions in under-constrained regions, outperforming state-of-the-art methods on datasets including Mip-NeRF 360 and NeRF Synthetic.
>
---
#### [replaced 117] TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.11941v2](http://arxiv.org/pdf/2411.11941v2)**

> **作者:** DaDong Jiang; Zhihui Ke; Xiaobo Zhou; Zhi Hou; Xianghui Yang; Wenbo Hu; Tie Qiu; Chunchao Guo
>
> **备注:** ICCV 2025
>
> **摘要:** Dynamic scene reconstruction is a long-term challenge in 3D vision. Recent methods extend 3D Gaussian Splatting to dynamic scenes via additional deformation fields and apply explicit constraints like motion flow to guide the deformation. However, they learn motion changes from individual timestamps independently, making it challenging to reconstruct complex scenes, particularly when dealing with violent movement, extreme-shaped geometries, or reflective surfaces. To address the above issue, we design a plug-and-play module called TimeFormer to enable existing deformable 3D Gaussians reconstruction methods with the ability to implicitly model motion patterns from a learning perspective. Specifically, TimeFormer includes a Cross-Temporal Transformer Encoder, which adaptively learns the temporal relationships of deformable 3D Gaussians. Furthermore, we propose a two-stream optimization strategy that transfers the motion knowledge learned from TimeFormer to the base stream during the training phase. This allows us to remove TimeFormer during inference, thereby preserving the original rendering speed. Extensive experiments in the multi-view and monocular dynamic scenes validate qualitative and quantitative improvement brought by TimeFormer. Project Page: https://patrickddj.github.io/TimeFormer/
>
---
#### [replaced 118] Foveated Retinotopy Improves Classification and Localization in CNNs
- **分类: cs.CV; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2402.15480v4](http://arxiv.org/pdf/2402.15480v4)**

> **作者:** Jean-Nicolas Jérémie; Emmanuel Daucé; Laurent U Perrinet
>
> **摘要:** From a falcon detecting prey to humans recognizing faces, many species exhibit extraordinary abilities in rapid visual localization and classification. These are made possible by a specialized retinal region called the fovea, which provides high acuity at the center of vision while maintaining lower resolution in the periphery. This distinctive spatial organization, preserved along the early visual pathway through retinotopic mapping, is fundamental to biological vision, yet remains largely unexplored in machine learning. Our study investigates how incorporating foveated retinotopy may benefit deep convolutional neural networks (CNNs) in image classification tasks. By implementing a foveated retinotopic transformation in the input layer of standard ResNet models and re-training them, we maintain comparable classification accuracy while enhancing the network's robustness to scale and rotational perturbations. Although this architectural modification introduces increased sensitivity to fixation point shifts, we demonstrate how this apparent limitation becomes advantageous: variations in classification probabilities across different gaze positions serve as effective indicators for object localization. Our findings suggest that foveated retinotopic mapping encodes implicit knowledge about visual object geometry, offering an efficient solution to the visual search problem - a capability crucial for many living species.
>
---
#### [replaced 119] SurfDist: Interpretable Three-Dimensional Instance Segmentation Using Curved Surface Patches
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08223v2](http://arxiv.org/pdf/2507.08223v2)**

> **作者:** Jackson Borchardt; Saul Kato
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** We present SurfDist, a convolutional neural network architecture for three-dimensional volumetric instance segmentation. SurfDist enables prediction of instances represented as closed surfaces composed of smooth parametric surface patches, specifically bicubic B\'ezier triangles. SurfDist is a modification of the popular model architecture StarDist-3D which breaks StarDist-3D's coupling of instance parameterization dimension and instance voxel resolution, and it produces predictions which may be upsampled to arbitrarily high resolutions without introduction of voxelization artifacts. For datasets with blob-shaped instances, common in biomedical imaging, SurfDist can outperform StarDist-3D with more compact instance parameterizations. We detail SurfDist's technical implementation and show one synthetic and one real-world dataset for which it outperforms StarDist-3D. These results demonstrate that interpretable instance surface models can be learned effectively alongside instance membership.
>
---
#### [replaced 120] Jasmine: Harnessing Diffusion Prior for Self-supervised Depth Estimation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15905v2](http://arxiv.org/pdf/2503.15905v2)**

> **作者:** Jiyuan Wang; Chunyu Lin; Cheng Guan; Lang Nie; Jing He; Haodong Li; Kang Liao; Yao Zhao
>
> **备注:** Accepted to NeurIPS 2025. 23 pages, with the appendix
>
> **摘要:** In this paper, we propose Jasmine, the first Stable Diffusion (SD)-based self-supervised framework for monocular depth estimation, which effectively harnesses SD's visual priors to enhance the sharpness and generalization of unsupervised prediction. Previous SD-based methods are all supervised since adapting diffusion models for dense prediction requires high-precision supervision. In contrast, self-supervised reprojection suffers from inherent challenges (e.g., occlusions, texture-less regions, illumination variance), and the predictions exhibit blurs and artifacts that severely compromise SD's latent priors. To resolve this, we construct a novel surrogate task of hybrid image reconstruction. Without any additional supervision, it preserves the detail priors of SD models by reconstructing the images themselves while preventing depth estimation from degradation. Furthermore, to address the inherent misalignment between SD's scale and shift invariant estimation and self-supervised scale-invariant depth estimation, we build the Scale-Shift GRU. It not only bridges this distribution gap but also isolates the fine-grained texture of SD output against the interference of reprojection loss. Extensive experiments demonstrate that Jasmine achieves SoTA performance on the KITTI benchmark and exhibits superior zero-shot generalization across multiple datasets.
>
---
#### [replaced 121] WonderHuman: Hallucinating Unseen Parts in Dynamic 3D Human Reconstruction
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2502.01045v2](http://arxiv.org/pdf/2502.01045v2)**

> **作者:** Zilong Wang; Zhiyang Dou; Yuan Liu; Cheng Lin; Xiao Dong; Yunhui Guo; Chenxu Zhang; Xin Li; Wenping Wang; Xiaohu Guo
>
> **摘要:** In this paper, we present WonderHuman to reconstruct dynamic human avatars from a monocular video for high-fidelity novel view synthesis. Previous dynamic human avatar reconstruction methods typically require the input video to have full coverage of the observed human body. However, in daily practice, one typically has access to limited viewpoints, such as monocular front-view videos, making it a cumbersome task for previous methods to reconstruct the unseen parts of the human avatar. To tackle the issue, we present WonderHuman, which leverages 2D generative diffusion model priors to achieve high-quality, photorealistic reconstructions of dynamic human avatars from monocular videos, including accurate rendering of unseen body parts. Our approach introduces a Dual-Space Optimization technique, applying Score Distillation Sampling (SDS) in both canonical and observation spaces to ensure visual consistency and enhance realism in dynamic human reconstruction. Additionally, we present a View Selection strategy and Pose Feature Injection to enforce the consistency between SDS predictions and observed data, ensuring pose-dependent effects and higher fidelity in the reconstructed avatar. In the experiments, our method achieves SOTA performance in producing photorealistic renderings from the given monocular video, particularly for those challenging unseen parts. The project page and source code can be found at https://wyiguanw.github.io/WonderHuman/.
>
---
#### [replaced 122] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.07389v2](http://arxiv.org/pdf/2503.07389v2)**

> **作者:** Ruidong Chen; Honglin Guo; Lanjun Wang; Chenyu Zhang; Weizhi Nie; An-An Liu
>
> **备注:** accepted by ICCV2025
>
> **摘要:** Recent advances in text-to-image diffusion models enable photorealistic image generation, but they also risk producing malicious content, such as NSFW images. To mitigate risk, concept erasure methods are studied to facilitate the model to unlearn specific concepts. However, current studies struggle to fully erase malicious concepts implicitly embedded in prompts (e.g., metaphorical expressions or adversarial prompts) while preserving the model's normal generation capability. To address this challenge, our study proposes TRCE, using a two-stage concept erasure strategy to achieve an effective trade-off between reliable erasure and knowledge preservation. Firstly, TRCE starts by erasing the malicious semantics implicitly embedded in textual prompts. By identifying a critical mapping objective(i.e., the [EoT] embedding), we optimize the cross-attention layers to map malicious prompts to contextually similar prompts but with safe concepts. This step prevents the model from being overly influenced by malicious semantics during the denoising process. Following this, considering the deterministic properties of the sampling trajectory of the diffusion model, TRCE further steers the early denoising prediction toward the safe direction and away from the unsafe one through contrastive learning, thus further avoiding the generation of malicious content. Finally, we conduct comprehensive evaluations of TRCE on multiple malicious concept erasure benchmarks, and the results demonstrate its effectiveness in erasing malicious concepts while better preserving the model's original generation ability. The code is available at: http://github.com/ddgoodgood/TRCE. CAUTION: This paper includes model-generated content that may contain offensive material.
>
---
#### [replaced 123] Poisson multi-Bernoulli mixture filter for trajectory measurements
- **分类: eess.SP; cs.CV; stat.AP**

- **链接: [http://arxiv.org/pdf/2504.08421v2](http://arxiv.org/pdf/2504.08421v2)**

> **作者:** Marco Fontana; Ángel F. García-Fernández; Simon Maskell
>
> **备注:** 16 pages, 9 figures, journal paper
>
> **摘要:** This paper presents a Poisson multi-Bernoulli mixture (PMBM) filter for multi-target filtering based on sensor measurements that are sets of trajectories in the last two-time step window. The proposed filter, the trajectory measurement PMBM (TM-PMBM) filter, propagates a PMBM density on the set of target states. In prediction, the filter obtains the PMBM density on the set of trajectories over the last two time steps. This density is then updated with the set of trajectory measurements. After the update step, the PMBM posterior on the set of two-step trajectories is marginalised to obtain a PMBM density on the set of target states. The filter provides a closed-form solution for multi-target filtering based on sets of trajectory measurements, estimating the set of target states at the end of each time window. Additionally, the paper proposes computationally lighter alternatives to the TM-PMBM filter by deriving a Poisson multi-Bernoulli (PMB) density through Kullback-Leibler divergence minimisation in an augmented space with auxiliary variables. The performance of the proposed filters are evaluated in a simulation study.
>
---
#### [replaced 124] QGFace: Quality-Guided Joint Training For Mixed-Quality Face Recognition
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2312.17494v2](http://arxiv.org/pdf/2312.17494v2)**

> **作者:** Youzhe Song; Feng Wang
>
> **摘要:** The quality of a face crop in an image is decided by many factors such as camera resolution, distance, and illumination condition. This makes the discrimination of face images with different qualities a challenging problem in realistic applications. However, most existing approaches are designed specifically for high-quality (HQ) or low-quality (LQ) images, and the performances would degrade for the mixed-quality images. Besides, many methods ask for pre-trained feature extractors or other auxiliary structures to support the training and the evaluation. In this paper, we point out that the key to better understand both the HQ and the LQ images simultaneously is to apply different learning methods according to their qualities. We propose a novel quality-guided joint training approach for mixed-quality face recognition, which could simultaneously learn the images of different qualities with a single encoder. Based on quality partition, classification-based method is employed for HQ data learning. Meanwhile, for the LQ images which lack identity information, we learn them with self-supervised image-image contrastive learning. To effectively catch up the model update and improve the discriminability of contrastive learning in our joint training scenario, we further propose a proxy-updated real-time queue to compose the contrastive pairs with features from the genuine encoder. Experiments on the low-quality datasets SCface and Tinyface, the mixed-quality dataset IJB-B, and five high-quality datasets demonstrate the effectiveness of our proposed approach in recognizing face images of different qualities.
>
---
#### [replaced 125] SteerDiff: Steering towards Safe Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2410.02710v2](http://arxiv.org/pdf/2410.02710v2)**

> **作者:** Hongxiang Zhang; Yifeng He; Hao Chen
>
> **摘要:** Text-to-image (T2I) diffusion models have drawn attention for their ability to generate high-quality images with precise text alignment. However, these models can also be misused to produce inappropriate content. Existing safety measures, which typically rely on text classifiers or ControlNet-like approaches, are often insufficient. Traditional text classifiers rely on large-scale labeled datasets and can be easily bypassed by rephrasing. As diffusion models continue to scale, fine-tuning these safeguards becomes increasingly challenging and lacks flexibility. Recent red-teaming attack researches further underscore the need for a new paradigm to prevent the generation of inappropriate content. In this paper, we introduce SteerDiff, a lightweight adaptor module designed to act as an intermediary between user input and the diffusion model, ensuring that generated images adhere to ethical and safety standards with little to no impact on usability. SteerDiff identifies and manipulates inappropriate concepts within the text embedding space to guide the model away from harmful outputs. We conduct extensive experiments across various concept unlearning tasks to evaluate the effectiveness of our approach. Furthermore, we benchmark SteerDiff against multiple red-teaming strategies to assess its robustness. Finally, we explore the potential of SteerDiff for concept forgetting tasks, demonstrating its versatility in text-conditioned image generation.
>
---
#### [replaced 126] EMedNeXt: An Enhanced Brain Tumor Segmentation Framework for Sub-Saharan Africa using MedNeXt V2 with Deep Supervision
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.23256v2](http://arxiv.org/pdf/2507.23256v2)**

> **作者:** Ahmed Jaheen; Abdelrahman Elsayed; Damir Kim; Daniil Tikhonov; Matheus Scatolin; Mohor Banerjee; Qiankun Ji; Mostafa Salem; Hu Wang; Sarim Hashmi; Mohammad Yaqub
>
> **备注:** Submitted to the BraTS-Lighthouse 2025 Challenge (MICCAI 2025)
>
> **摘要:** Brain cancer affects millions worldwide, and in nearly every clinical setting, doctors rely on magnetic resonance imaging (MRI) to diagnose and monitor gliomas. However, the current standard for tumor quantification through manual segmentation of multi-parametric MRI is time-consuming, requires expert radiologists, and is often infeasible in under-resourced healthcare systems. This problem is especially pronounced in low-income regions, where MRI scanners are of lower quality and radiology expertise is scarce, leading to incorrect segmentation and quantification. In addition, the number of acquired MRI scans in Africa is typically small. To address these challenges, the BraTS-Lighthouse 2025 Challenge focuses on robust tumor segmentation in sub-Saharan Africa (SSA), where resource constraints and image quality degradation introduce significant shifts. In this study, we present EMedNeXt -- an enhanced brain tumor segmentation framework based on MedNeXt V2 with deep supervision and optimized post-processing pipelines tailored for SSA. EMedNeXt introduces three key contributions: a larger region of interest, an improved nnU-Net v2-based architectural skeleton, and a robust model ensembling system. Evaluated on the hidden validation set, our solution achieved an average LesionWise DSC of 0.897 with an average LesionWise NSD of 0.541 and 0.84 at a tolerance of 0.5 mm and 1.0 mm, respectively.
>
---
#### [replaced 127] Event-driven Robust Fitting on Neuromorphic Hardware
- **分类: cs.CV; cs.NE**

- **链接: [http://arxiv.org/pdf/2508.09466v2](http://arxiv.org/pdf/2508.09466v2)**

> **作者:** Tam Ngoc-Bang Nguyen; Anh-Dzung Doan; Zhipeng Cai; Tat-Jun Chin
>
> **备注:** 13 pages, accepted in ICCV 2025 Workshop on Neuromorphic Vision (NeVI)
>
> **摘要:** Robust fitting of geometric models is a fundamental task in many computer vision pipelines. Numerous innovations have been produced on the topic, from improving the efficiency and accuracy of random sampling heuristics to generating novel theoretical insights that underpin new approaches with mathematical guarantees. However, one aspect of robust fitting that has received little attention is energy efficiency. This performance metric has become critical as high energy consumption is a growing concern for AI adoption. In this paper, we explore energy-efficient robust fitting via the neuromorphic computing paradigm. Specifically, we designed a novel spiking neural network for robust fitting on real neuromorphic hardware, the Intel Loihi 2. Enabling this are novel event-driven formulations of model estimation that allow robust fitting to be implemented in the unique architecture of Loihi 2, and algorithmic strategies to alleviate the current limited precision and instruction set of the hardware. Results show that our neuromorphic robust fitting consumes only a fraction (15%) of the energy required to run the established robust fitting algorithm on a standard CPU to equivalent accuracy.
>
---
#### [replaced 128] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10610v3](http://arxiv.org/pdf/2505.10610v3)**

> **作者:** Zhaowei Wang; Wenhao Yu; Xiyu Ren; Jipeng Zhang; Yu Zhao; Rohit Saxena; Liang Cheng; Ginny Wong; Simon See; Pasquale Minervini; Yangqiu Song; Mark Steedman
>
> **备注:** Accepted as a spotlight at NeurIPS 2025
>
> **摘要:** The rapid extension of context windows in large vision-language models has given rise to long-context vision-language models (LCVLMs), which are capable of handling hundreds of images with interleaved text tokens in a single forward pass. In this work, we introduce MMLongBench, the first benchmark covering a diverse set of long-context vision-language tasks, to evaluate LCVLMs effectively and thoroughly. MMLongBench is composed of 13,331 examples spanning five different categories of downstream tasks, such as Visual RAG and Many-Shot ICL. It also provides broad coverage of image types, including various natural and synthetic images. To assess the robustness of the models to different input lengths, all examples are delivered at five standardized input lengths (8K-128K tokens) via a cross-modal tokenization scheme that combines vision patches and text tokens. Through a thorough benchmarking of 46 closed-source and open-source LCVLMs, we provide a comprehensive analysis of the current models' vision-language long-context ability. Our results show that: i) performance on a single task is a weak proxy for overall long-context capability; ii) both closed-source and open-source models face challenges in long-context vision-language tasks, indicating substantial room for future improvement; iii) models with stronger reasoning ability tend to exhibit better long-context performance. By offering wide task coverage, various image types, and rigorous length control, MMLongBench provides the missing foundation for diagnosing and advancing the next generation of LCVLMs.
>
---
#### [replaced 129] Do We Need All the Synthetic Data? Targeted Synthetic Image Augmentation via Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21574v2](http://arxiv.org/pdf/2505.21574v2)**

> **作者:** Dang Nguyen; Jiping Li; Jinghao Zheng; Baharan Mirzasoleiman
>
> **摘要:** Synthetically augmenting training datasets with diffusion models has been an effective strategy for improving generalization of image classifiers. However, existing techniques struggle to ensure the diversity of generation and increase the size of the data by up to 10-30x to improve the in-distribution performance. In this work, we show that synthetically augmenting part of the data that is not learned early in training with faithful images-containing same features but different noise-outperforms augmenting the entire dataset. By analyzing a two-layer CNN, we prove that this strategy improves generalization by promoting homogeneity in feature learning speed without amplifying noise. Our extensive experiments show that by augmenting only 30%-40% of the data, our method boosts generalization by up to 2.8% in a variety of scenarios, including training ResNet, ViT, ConvNeXt, and Swin Transformer on CIFAR-10/100, and TinyImageNet, with various optimizers including SGD and SAM. Notably, our method applied with SGD outperforms the SOTA optimizer, SAM, on CIFAR-100 and TinyImageNet.
>
---
#### [replaced 130] Copyright Infringement Detection in Text-to-Image Diffusion Models via Differential Privacy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23022v2](http://arxiv.org/pdf/2509.23022v2)**

> **作者:** Xiafeng Man; Zhipeng Wei; Jingjing Chen
>
> **摘要:** The widespread deployment of large vision models such as Stable Diffusion raises significant legal and ethical concerns, as these models can memorize and reproduce copyrighted content without authorization. Existing detection approaches often lack robustness and fail to provide rigorous theoretical underpinnings. To address these gaps, we formalize the concept of copyright infringement and its detection from the perspective of Differential Privacy (DP), and introduce the conditional sensitivity metric, a concept analogous to sensitivity in DP, that quantifies the deviation in a diffusion model's output caused by the inclusion or exclusion of a specific training data point. To operationalize this metric, we propose D-Plus-Minus (DPM), a novel post-hoc detection framework that identifies copyright infringement in text-to-image diffusion models. Specifically, DPM simulates inclusion and exclusion processes by fine-tuning models in two opposing directions: learning or unlearning. Besides, to disentangle concept-specific influence from the global parameter shifts induced by fine-tuning, DPM computes confidence scores over orthogonal prompt distributions using statistical metrics. Moreover, to facilitate standardized benchmarking, we also construct the Copyright Infringement Detection Dataset (CIDD), a comprehensive resource for evaluating detection across diverse categories. Our results demonstrate that DPM reliably detects infringement content without requiring access to the original training dataset or text prompts, offering an interpretable and practical solution for safeguarding intellectual property in the era of generative AI.
>
---
#### [replaced 131] ImplicitQA: Going beyond frames towards Implicit Video Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21742v2](http://arxiv.org/pdf/2506.21742v2)**

> **作者:** Sirnam Swetha; Rohit Gupta; Parth Parag Kulkarni; David G Shatwell; Jeffrey A Chan Santiago; Nyle Siddiqui; Joseph Fioresi; Mubarak Shah
>
> **摘要:** Video Question Answering (VideoQA) has made significant strides by leveraging multimodal learning to align visual and textual modalities. However, current benchmarks overwhelmingly focus on questions answerable through explicit visual content - actions, objects, and events directly observable within individual frames or short clips. In contrast, creative and cinematic videos - such as movies, TV shows, and narrative-driven content - employ storytelling techniques that deliberately omit certain depictions, requiring viewers to infer motives, relationships across discontinuous frames with disjoint visual contexts. Humans naturally excel at such implicit reasoning, seamlessly integrating information across time and context to construct coherent narratives. Yet current benchmarks fail to capture this essential dimension of human-like understanding. To bridge this gap, we present ImplicitQA, a novel benchmark specifically designed to test VideoQA models on human-like implicit reasoning. ImplicitQA comprises 1K meticulously annotated QA pairs drawn from 1K high-quality creative video clips covering 15 genres across 7 decades of content. Questions are systematically categorized into nine key reasoning dimensions: lateral and vertical spatial reasoning, depth and proximity, viewpoint and visibility, motion and trajectory, causal and motivational reasoning, social interactions, physical context, and inferred counting. These annotations are deliberately challenging, crafted by authors, validated through multiple annotators, and benchmarked against human performance to ensure high quality. Our extensive evaluations on 11 leading VideoQA models reveals consistent and significant performance degradation, underscoring their reliance on surface-level visual cues and highlighting the difficulty of implicit reasoning. https://huggingface.co/datasets/ucf-crcv/ImplicitQA.
>
---
#### [replaced 132] LIAM: Multimodal Transformer for Language Instructions, Images, Actions and Semantic Maps
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.12230v2](http://arxiv.org/pdf/2503.12230v2)**

> **作者:** Yihao Wang; Raphael Memmesheimer; Sven Behnke
>
> **备注:** 12 pages, 4 figures, 2 tables, 19th International Conference on Intelligent Autonomous Systems (IAS), Genoa, Italy, June 2025
>
> **摘要:** The availability of large language models and open-vocabulary object perception methods enables more flexibility for domestic service robots. The large variability of domestic tasks can be addressed without implementing each task individually by providing the robot with a task description along with appropriate environment information. In this work, we propose LIAM - an end-to-end model that predicts action transcripts based on language, image, action, and map inputs. Language and image inputs are encoded with a CLIP backbone, for which we designed two pre-training tasks to fine-tune its weights and pre-align the latent spaces. We evaluate our method on the ALFRED dataset, a simulator-generated benchmark for domestic tasks. Our results demonstrate the importance of pre-aligning embedding spaces from different modalities and the efficacy of incorporating semantic maps.
>
---
#### [replaced 133] ExGS: Extreme 3D Gaussian Compression with Diffusion Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.24758v3](http://arxiv.org/pdf/2509.24758v3)**

> **作者:** Jiaqi Chen; Xinhao Ji; Yuanyuan Gao; Hao Li; Yuning Gong; Yifei Liu; Dan Xu; Zhihang Zhong; Dingwen Zhang; Xiao Sun
>
> **摘要:** Neural scene representations, such as 3D Gaussian Splatting (3DGS), have enabled high-quality neural rendering; however, their large storage and transmission costs hinder deployment in resource-constrained environments. Existing compression methods either rely on costly optimization, which is slow and scene-specific, or adopt training-free pruning and quantization, which degrade rendering quality under high compression ratios. In contrast, recent data-driven approaches provide a promising direction to overcome this trade-off, enabling efficient compression while preserving high rendering quality.We introduce ExGS, a novel feed-forward framework that unifies Universal Gaussian Compression (UGC) with GaussPainter for Extreme 3DGS compression. UGC performs re-optimization-free pruning to aggressively reduce Gaussian primitives while retaining only essential information, whereas GaussPainter leverages powerful diffusion priors with mask-guided refinement to restore high-quality renderings from heavily pruned Gaussian scenes. Unlike conventional inpainting, GaussPainter not only fills in missing regions but also enhances visible pixels, yielding substantial improvements in degraded renderings.To ensure practicality, it adopts a lightweight VAE and a one-step diffusion design, enabling real-time restoration. Our framework can even achieve over 100X compression (reducing a typical 354.77 MB model to about 3.31 MB) while preserving fidelity and significantly improving image quality under challenging conditions. These results highlight the central role of diffusion priors in bridging the gap between extreme compression and high-quality neural rendering.Our code repository will be released at: https://github.com/chenttt2001/ExGS
>
---
#### [replaced 134] TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.18671v4](http://arxiv.org/pdf/2506.18671v4)**

> **作者:** Yuqin Dai; Wanlu Zhu; Ronghui Li; Xiu Li; Zhenyu Zhang; Jun Li; Jian Yang
>
> **摘要:** Music-driven dance generation has garnered significant attention due to its wide range of industrial applications, particularly in the creation of group choreography. During the group dance generation process, however, most existing methods still face three primary issues: multi-dancer collisions, single-dancer foot sliding and abrupt swapping in the generation of long group dance. In this paper, we propose TCDiff++, a music-driven end-to-end framework designed to generate harmonious group dance. Specifically, to mitigate multi-dancer collisions, we utilize a dancer positioning embedding to encode temporal and identity information. Additionally, we incorporate a distance-consistency loss to ensure that inter-dancer distances remain within plausible ranges. To address the issue of single-dancer foot sliding, we introduce a swap mode embedding to indicate dancer swapping patterns and design a Footwork Adaptor to refine raw motion, thereby minimizing foot sliding. For long group dance generation, we present a long group diffusion sampling strategy that reduces abrupt position shifts by injecting positional information into the noisy input. Furthermore, we integrate a Sequence Decoder layer to enhance the model's ability to selectively process long sequences. Extensive experiments demonstrate that our TCDiff++ achieves state-of-the-art performance, particularly in long-duration scenarios, ensuring high-quality and coherent group dance generation.
>
---
#### [replaced 135] RGB-to-Polarization Estimation: A New Task and Benchmark Study
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13050v2](http://arxiv.org/pdf/2505.13050v2)**

> **作者:** Beibei Lin; Zifeng Yuan; Tingting Chen
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Polarization images provide rich physical information that is fundamentally absent from standard RGB images, benefiting a wide range of computer vision applications such as reflection separation and material classification. However, the acquisition of polarization images typically requires additional optical components, which increases both the cost and the complexity of the applications. To bridge this gap, we introduce a new task: RGB-to-polarization image estimation, which aims to infer polarization information directly from RGB images. In this work, we establish the first comprehensive benchmark for this task by leveraging existing polarization datasets and evaluating a diverse set of state-of-the-art deep learning models, including both restoration-oriented and generative architectures. Through extensive quantitative and qualitative analysis, our benchmark not only establishes the current performance ceiling of RGB-to-polarization estimation, but also systematically reveals the respective strengths and limitations of different model families -- such as direct reconstruction versus generative synthesis, and task-specific training versus large-scale pre-training. In addition, we provide some potential directions for future research on polarization estimation. This benchmark is intended to serve as a foundational resource to facilitate the design and evaluation of future methods for polarization estimation from standard RGB inputs.
>
---
#### [replaced 136] SAR-TEXT: A Large-Scale SAR Image-Text Dataset Built with SAR-Narrator and A Progressive Learning Strategy for Downstream Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18743v3](http://arxiv.org/pdf/2507.18743v3)**

> **作者:** Yiguo He; Xinjun Cheng; Junjie Zhu; Chunping Qiu; Jun Wang; Xichuan Zhang; Qiangjuan Huang; Ke Yang
>
> **备注:** IEEE Submission
>
> **摘要:** Vision Language Models (VLMs) have achieved remarkable breakthroughs in the field of remote sensing in recent years. Synthetic Aperture Radar (SAR) imagery, with its all-weather capability, is essential in remote sensing, yet the lack of large-scale, high-quality SAR image-text datasets hinders its semantic understanding. In this paper, we construct SAR-TEXT, a large-scale and high-quality dataset consisting of over 130,000 SAR image-text pairs. To construct the SAR-TEXT dataset, we design the SAR-Narrator framework, which generates textual descriptions for SAR images through a multi-stage strategy. To verify the effectiveness of the SAR-TEXT dataset, we conduct experiments on three typical vision-language tasks: image-text retrieval, image captioning, and visual question answering (VQA). Specifically, we construct three representative models on SAR-TEXT: SAR-RS-CLIP, SAR-RS-CoCa, and SAR-GPT. SAR-RS-CLIP achieves notable improvements in retrieval performance, boosting average recall by 12.97% and 10.0% on the OSdataset_512 and HRSID test sets, respectively. In the captioning task, SAR-RS-CoCa achieves significant improvements over the original CoCa models in terms of BLEU-4, SPICE, and CIDEr scores. In the VQA task, SAR-GPT outperforms baseline and single-stage models on multiple SAR-VQA datasets, demonstrating stronger semantic understanding and reasoning ability, as further confirmed by qualitative results. It is worth noting that, as a flexible captioning tool, SAR-Narrator can be readily adopted by the community to construct larger-scale SAR image-text datasets. All code, pretrained models, and the SAR-Text dataset are publicly available at: https://github.com/YiguoHe/SAR-TEXT.
>
---
#### [replaced 137] Enhancing Transformers Through Conditioned Embedded Tokens
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12789v2](http://arxiv.org/pdf/2505.12789v2)**

> **作者:** Hemanth Saratchandran; Simon Lucey
>
> **备注:** ICCV 2025
>
> **摘要:** Transformers have transformed modern machine learning, driving breakthroughs in computer vision, natural language processing, and robotics. At the core of their success lies the attention mechanism, which enables the modeling of global dependencies among input tokens. However, we reveal that the attention block in transformers suffers from inherent ill-conditioning, which hampers gradient-based optimization and leads to inefficient training. To address this, we develop a theoretical framework that establishes a direct relationship between the conditioning of the attention block and that of the embedded tokenized data. Building on this insight, we introduce conditioned embedded tokens, a method that systematically modifies the embedded tokens to improve the conditioning of the attention mechanism. Our analysis demonstrates that this approach significantly mitigates ill-conditioning, leading to more stable and efficient training. We validate our methodology across various transformer architectures, achieving consistent improvements in image classification, object detection, instance segmentation, and natural language processing, highlighting its broad applicability and effectiveness.
>
---
#### [replaced 138] RowDetr: End-to-End Crop Row Detection Using Polynomials
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.10525v3](http://arxiv.org/pdf/2412.10525v3)**

> **作者:** Rahul Harsha Cheppally; Ajay Sharda
>
> **备注:** Code will be open sourced upon publication
>
> **摘要:** Crop row detection enables autonomous robots to navigate in gps denied environments. Vision based strategies often struggle in the environments due to gaps, curved crop rows and require post-processing steps. Furthermore, labeling crop rows in under the canopy environments accurately is very difficult due to occlusions. This study introduces RowDetr, an efficient end-to-end transformer-based neural network for crop row detection in precision agriculture. RowDetr leverages a lightweight backbone and a hybrid encoder to model straight, curved, or occluded crop rows with high precision. Central to the architecture is a novel polynomial representation that enables direct parameterization of crop rows, eliminating computationally expensive post-processing. Key innovations include a PolySampler module and multi-scale deformable attention, which work together with PolyOptLoss, an energy-based loss function designed to optimize geometric alignment between predicted and the annotated crop rows, while also enhancing robustness against labeling noise. RowDetr was evaluated against other state-of-the-art end-to-end crop row detection methods like AgroNav and RolColAttention on a diverse dataset of 6,962 high-resolution images, used for training, validation, and testing across multiple crop types with annotated crop rows. The system demonstrated superior performance, achieved an F1 score up to 0.74 and a lane position deviation as low as 0.405. Furthermore, RowDetr achieves a real-time inference latency of 6.7ms, which was optimized to 3.5ms with INT8 quantization on an NVIDIA Jetson Orin AGX. This work highlighted the critical efficiency of polynomial parameterization, making RowDetr particularly suitable for deployment on edge computing devices in agricultural robotics and autonomous farming equipment. Index terms > Crop Row Detection, Under Canopy Navigation, Transformers, RT-DETR, RT-DETRv2
>
---
