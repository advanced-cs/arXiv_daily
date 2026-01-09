# 计算机视觉 cs.CV

- **最新发布 97 篇**

- **更新 61 篇**

## 最新发布

#### [new 001] SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection
- **分类: cs.CV**

- **简介: 该论文属于3D车道检测任务，旨在解决传统方法特征对齐不佳及忽略先验信息的问题。提出SparseLaneSTP，融合时空先验与稀疏Transformer，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2601.04968v1](https://arxiv.org/pdf/2601.04968v1)**

> **作者:** Maximilian Pittner; Joel Janai; Mario Faigle; Alexandru Paul Condurache
>
> **备注:** Published at IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** 3D lane detection has emerged as a critical challenge in autonomous driving, encompassing identification and localization of lane markings and the 3D road surface. Conventional 3D methods detect lanes from dense birds-eye-viewed (BEV) features, though erroneous transformations often result in a poor feature representation misaligned with the true 3D road surface. While recent sparse lane detectors have surpassed dense BEV approaches, they completely disregard valuable lane-specific priors. Furthermore, existing methods fail to utilize historic lane observations, which yield the potential to resolve ambiguities in situations of poor visibility. To address these challenges, we present SparseLaneSTP, a novel method that integrates both geometric properties of the lane structure and temporal information into a sparse lane transformer. It introduces a new lane-specific spatio-temporal attention mechanism, a continuous lane representation tailored for sparse architectures as well as temporal regularization. Identifying weaknesses of existing 3D lane datasets, we also introduce a precise and consistent 3D lane dataset using a simple yet effective auto-labeling strategy. Our experimental section proves the benefits of our contributions and demonstrates state-of-the-art performance across all detection and error metrics on existing 3D lane detection benchmarks as well as on our novel dataset.
>
---
#### [new 002] Detection of Deployment Operational Deviations for Safety and Security of AI-Enabled Human-Centric Cyber Physical Systems
- **分类: cs.CV**

- **简介: 该论文属于安全与可靠性任务，旨在解决AI驱动的人机协同系统在部署时可能产生的操作偏差问题，提出框架评估策略，并以糖尿病管理为例展示检测技术。**

- **链接: [https://arxiv.org/pdf/2601.04605v1](https://arxiv.org/pdf/2601.04605v1)**

> **作者:** Bernard Ngabonziza; Ayan Banerjee; Sandeep K. S. Gupta
>
> **摘要:** In recent years, Human-centric cyber-physical systems have increasingly involved artificial intelligence to enable knowledge extraction from sensor-collected data. Examples include medical monitoring and control systems, as well as autonomous cars. Such systems are intended to operate according to the protocols and guidelines for regular system operations. However, in many scenarios, such as closed-loop blood glucose control for Type 1 diabetics, self-driving cars, and monitoring systems for stroke diagnosis. The operations of such AI-enabled human-centric applications can expose them to cases for which their operational mode may be uncertain, for instance, resulting from the interactions with a human with the system. Such cases, in which the system is in uncertain conditions, can violate the system's safety and security requirements. This paper will discuss operational deviations that can lead these systems to operate in unknown conditions. We will then create a framework to evaluate different strategies for ensuring the safety and security of AI-enabled human-centric cyber-physical systems in operation deployment. Then, as an example, we show a personalized image-based novel technique for detecting the non-announcement of meals in closed-loop blood glucose control for Type 1 diabetics.
>
---
#### [new 003] Cutting AI Research Costs: How Task-Aware Compression Makes Large Language Model Agents Affordable
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在降低大模型研究成本。通过任务感知压缩技术，将计算费用减少68.3%，同时保持96.2%的性能。**

- **链接: [https://arxiv.org/pdf/2601.05191v1](https://arxiv.org/pdf/2601.05191v1)**

> **作者:** Zuhair Ahmed Khan Taha; Mohammed Mudassir Uddin; Shahnawaz Alam
>
> **摘要:** When researchers deploy large language models for autonomous tasks like reviewing literature or generating hypotheses, the computational bills add up quickly. A single research session using a 70-billion parameter model can cost around $127 in cloud fees, putting these tools out of reach for many academic labs. We developed AgentCompress to tackle this problem head-on. The core idea came from a simple observation during our own work: writing a novel hypothesis clearly demands more from the model than reformatting a bibliography. Why should both tasks run at full precision? Our system uses a small neural network to gauge how hard each incoming task will be, based only on its opening words, then routes it to a suitably compressed model variant. The decision happens in under a millisecond. Testing across 500 research workflows in four scientific fields, we cut compute costs by 68.3% while keeping 96.2% of the original success rate. For labs watching their budgets, this could mean the difference between running experiments and sitting on the sidelines
>
---
#### [new 004] Defocus Aberration Theory Confirms Gaussian Model in Most Imaging Devices
- **分类: cs.CV**

- **简介: 该论文属于3D恢复任务，解决从2D图像准确估计深度的问题。通过理论分析与实验验证，证明大多数成像设备的散焦操作符符合高斯模型，提升深度估计的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2601.04779v1](https://arxiv.org/pdf/2601.04779v1)**

> **作者:** Akbar Saadat
>
> **备注:** 13 pages, 9 figures, 11 .jpg files
>
> **摘要:** Over the past three decades, defocus has consistently provided groundbreaking depth information in scene images. However, accurately estimating depth from 2D images continues to be a persistent and fundamental challenge in the field of 3D recovery. Heuristic approaches involve with the ill-posed problem for inferring the spatial variant defocusing blur, as the desired blur cannot be distinguished from the inherent blur. Given a prior knowledge of the defocus model, the problem become well-posed with an analytic solution for the relative blur between two images, taken at the same viewpoint with different camera settings for the focus. The Gaussian model stands out as an optimal choice for real-time applications, due to its mathematical simplicity and computational efficiency. And theoretically, it is the only model can be applied at the same time to both the absolute blur caused by depth in a single image and the relative blur resulting from depth differences between two images. This paper introduces the settings, for conventional imaging devices, to ensure that the defocusing operator adheres to the Gaussian model. Defocus analysis begins within the framework of geometric optics and is conducted by defocus aberration theory in diffraction-limited optics to obtain the accuracy of fitting the actual model to its Gaussian approximation. The results for a typical set of focused depths between $1$ and $100$ meters, with a maximum depth variation of $10\%$ at the focused depth, confirm the Gaussian model's applicability for defocus operators in most imaging devices. The findings demonstrate a maximum Mean Absolute Error $(\!M\!A\!E)$ of less than $1\%$, underscoring the model's accuracy and reliability.
>
---
#### [new 005] Vision-Language Introspection: Mitigating Overconfident Hallucinations in MLLMs via Interpretable Bi-Causal Steering
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态大模型任务，解决对象幻觉问题。提出VLI框架，通过可解释的双因果调节减少过自信幻觉，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2601.05159v1](https://arxiv.org/pdf/2601.05159v1)**

> **作者:** Shuliang Liu; Songbo Yang; Dong Fang; Sihang Jia; Yuqi Tang; Lingfeng Su; Ruoshui Peng; Yibo Yan; Xin Zou; Xuming Hu
>
> **摘要:** Object hallucination critically undermines the reliability of Multimodal Large Language Models, often stemming from a fundamental failure in cognitive introspection, where models blindly trust linguistic priors over specific visual evidence. Existing mitigations remain limited: contrastive decoding approaches operate superficially without rectifying internal semantic misalignments, while current latent steering methods rely on static vectors that lack instance-specific precision. We introduce Vision-Language Introspection (VLI), a training-free inference framework that simulates a metacognitive self-correction process. VLI first performs Attributive Introspection to diagnose hallucination risks via probabilistic conflict detection and localize the causal visual anchors. It then employs Interpretable Bi-Causal Steering to actively modulate the inference process, dynamically isolating visual evidence from background noise while neutralizing blind confidence through adaptive calibration. VLI achieves state-of-the-art performance on advanced models, reducing object hallucination rates by 12.67% on MMHal-Bench and improving accuracy by 5.8% on POPE.
>
---
#### [new 006] ReHyAt: Recurrent Hybrid Attention for Video Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文提出ReHyAt，解决视频生成中注意力机制复杂度高的问题，通过混合注意力设计实现高效且高质量的视频生成。**

- **链接: [https://arxiv.org/pdf/2601.04342v1](https://arxiv.org/pdf/2601.04342v1)**

> **作者:** Mohsen Ghafoorian; Amirhossein Habibian
>
> **摘要:** Recent advances in video diffusion models have shifted towards transformer-based architectures, achieving state-of-the-art video generation but at the cost of quadratic attention complexity, which severely limits scalability for longer sequences. We introduce ReHyAt, a Recurrent Hybrid Attention mechanism that combines the fidelity of softmax attention with the efficiency of linear attention, enabling chunk-wise recurrent reformulation and constant memory usage. Unlike the concurrent linear-only SANA Video, ReHyAt's hybrid design allows efficient distillation from existing softmax-based models, reducing the training cost by two orders of magnitude to ~160 GPU hours, while being competitive in the quality. Our light-weight distillation and finetuning pipeline provides a recipe that can be applied to future state-of-the-art bidirectional softmax-based models. Experiments on VBench and VBench-2.0, as well as a human preference study, demonstrate that ReHyAt achieves state-of-the-art video quality while reducing attention cost from quadratic to linear, unlocking practical scalability for long-duration and on-device video generation. Project page is available at https://qualcomm-ai-research.github.io/rehyat.
>
---
#### [new 007] Performance Analysis of Image Classification on Bangladeshi Datasets
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，比较了自定义CNN与预训练模型（如VGG-16、ResNet-50、MobileNet）的性能，分析其在数据有限时的优劣。**

- **链接: [https://arxiv.org/pdf/2601.04397v1](https://arxiv.org/pdf/2601.04397v1)**

> **作者:** Mohammed Sami Khan; Fabiha Muniat; Rowzatul Zannat
>
> **摘要:** Convolutional Neural Networks (CNNs) have demonstrated remarkable success in image classification tasks; however, the choice between designing a custom CNN from scratch and employing established pre-trained architectures remains an important practical consideration. In this work, we present a comparative analysis of a custom-designed CNN and several widely used deep learning architectures, including VGG-16, ResNet-50, and MobileNet, for an image classification task. The custom CNN is developed and trained from scratch, while the popular architectures are employed using transfer learning under identical experimental settings. All models are evaluated using standard performance metrics such as accuracy, precision, recall, and F1-score. Experimental results show that pre-trained CNN architectures consistently outperform the custom CNN in terms of classification accuracy and convergence speed, particularly when training data is limited. However, the custom CNN demonstrates competitive performance with significantly fewer parameters and reduced computational complexity. This study highlights the trade-offs between model complexity, performance, and computational efficiency, and provides practical insights into selecting appropriate CNN architectures for image classification problems.
>
---
#### [new 008] 3D-Agent:Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D物体标注任务，解决空间复杂性、遮挡和视角不一致问题。提出Tri MARF框架，融合多模态输入，提升大规模3D标注效果。**

- **链接: [https://arxiv.org/pdf/2601.04404v1](https://arxiv.org/pdf/2601.04404v1)**

> **作者:** Jusheng Zhang; Yijia Fan; Zimo Wen; Jian Wang; Keze Wang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Driven by applications in autonomous driving robotics and augmented reality 3D object annotation presents challenges beyond 2D annotation including spatial complexity occlusion and viewpoint inconsistency Existing approaches based on single models often struggle to address these issues effectively We propose Tri MARF a novel framework that integrates tri modal inputs including 2D multi view images textual descriptions and 3D point clouds within a multi agent collaborative architecture to enhance large scale 3D annotation Tri MARF consists of three specialized agents a vision language model agent for generating multi view descriptions an information aggregation agent for selecting optimal descriptions and a gating agent that aligns textual semantics with 3D geometry for refined captioning Extensive experiments on Objaverse LVIS Objaverse XL and ABO demonstrate that Tri MARF substantially outperforms existing methods achieving a CLIPScore of 88 point 7 compared to prior state of the art methods retrieval accuracy of 45 point 2 and 43 point 8 on ViLT R at 5 and a throughput of up to 12000 objects per hour on a single NVIDIA A100 GPU
>
---
#### [new 009] Re-Align: Structured Reasoning-guided Alignment for In-Context Image Generation and Editing
- **分类: cs.CV**

- **简介: 该论文属于图像生成与编辑任务，旨在解决用户意图理解与生成结果不一致的问题。提出Re-Align框架，通过结构化推理对齐提升生成效果。**

- **链接: [https://arxiv.org/pdf/2601.05124v1](https://arxiv.org/pdf/2601.05124v1)**

> **作者:** Runze He; Yiji Cheng; Tiankai Hang; Zhimin Li; Yu Xu; Zijin Yin; Shiyi Zhang; Wenxun Dai; Penghui Du; Ao Ma; Chunyu Wang; Qinglin Lu; Jizhong Han; Jiao Dai
>
> **备注:** 13 pages, 9 figures, project page: https://github.com/hrz2000/realign
>
> **摘要:** In-context image generation and editing (ICGE) enables users to specify visual concepts through interleaved image-text prompts, demanding precise understanding and faithful execution of user intent. Although recent unified multimodal models exhibit promising understanding capabilities, these strengths often fail to transfer effectively to image generation. We introduce Re-Align, a unified framework that bridges the gap between understanding and generation through structured reasoning-guided alignment. At its core lies the In-Context Chain-of-Thought (IC-CoT), a structured reasoning paradigm that decouples semantic guidance and reference association, providing clear textual target and mitigating confusion among reference images. Furthermore, Re-Align introduces an effective RL training scheme that leverages a surrogate reward to measure the alignment between structured reasoning text and the generated image, thereby improving the model's overall performance on ICGE tasks. Extensive experiments verify that Re-Align outperforms competitive methods of comparable model scale and resources on both in-context image generation and editing tasks.
>
---
#### [new 010] From Preoperative CT to Postmastoidectomy Mesh Construction:1Mastoidectomy Shape Prediction for Cochlear Implant Surgery
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像分析任务，旨在解决 Cochlear Implant 手术中 Mastoidectomy 形状预测问题。通过自监督和弱监督学习方法，从术前CT图像预测手术区域，提升手术规划效果。**

- **链接: [https://arxiv.org/pdf/2601.04405v1](https://arxiv.org/pdf/2601.04405v1)**

> **作者:** Yike Zhang; Eduardo Davalos; Dingjie Su; Ange Lou; Jack Noble
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2505.18368
>
> **摘要:** Cochlear Implant (CI) surgery treats severe hearing loss by inserting an electrode array into the cochlea to stimulate the auditory nerve. An important step in this procedure is mastoidectomy, which removes part of the mastoid region of the temporal bone to provide surgical access. Accurate mastoidectomy shape prediction from preoperative imaging improves pre-surgical planning, reduces risks, and enhances surgical outcomes. Despite its importance, there are limited deep-learning-based studies regarding this topic due to the challenges of acquiring ground-truth labels. We address this gap by investigating self-supervised and weakly-supervised learning models to predict the mastoidectomy region without human annotations. We propose a hybrid self-supervised and weakly-supervised learning framework to predict the mastoidectomy region directly from preoperative CT scans, where the mastoid remains intact. Our hybrid method achieves a mean Dice score of 0.72 when predicting the complex and boundary-less mastoidectomy shape, surpassing state-of-the-art approaches and demonstrating strong performance. The method provides groundwork for constructing 3D postmastoidectomy surfaces directly from the corresponding preoperative CT scans. To our knowledge, this is the first work that integrating self-supervised and weakly-supervised learning for mastoidectomy shape prediction, offering a robust and efficient solution for CI surgical planning while leveraging 3D T-distribution loss in weakly-supervised medical imaging.
>
---
#### [new 011] DB-MSMUNet:Dual Branch Multi-scale Mamba UNet for Pancreatic CT Scans Segmentation
- **分类: cs.CV**

- **简介: 该论文属于胰腺CT图像分割任务，旨在解决胰腺及其病灶分割困难的问题。提出DB-MSMUNet网络，通过多尺度Mamba模块和双解码器结构提升分割精度与边界保持能力。**

- **链接: [https://arxiv.org/pdf/2601.04676v1](https://arxiv.org/pdf/2601.04676v1)**

> **作者:** Qiu Guan; Zhiqiang Yang; Dezhang Ye; Yang Chen; Xinli Xu; Ying Tang
>
> **摘要:** Accurate segmentation of the pancreas and its lesions in CT scans is crucial for the precise diagnosis and treatment of pancreatic cancer. However, it remains a highly challenging task due to several factors such as low tissue contrast with surrounding organs, blurry anatomical boundaries, irregular organ shapes, and the small size of lesions. To tackle these issues, we propose DB-MSMUNet (Dual-Branch Multi-scale Mamba UNet), a novel encoder-decoder architecture designed specifically for robust pancreatic segmentation. The encoder is constructed using a Multi-scale Mamba Module (MSMM), which combines deformable convolutions and multi-scale state space modeling to enhance both global context modeling and local deformation adaptation. The network employs a dual-decoder design: the edge decoder introduces an Edge Enhancement Path (EEP) to explicitly capture boundary cues and refine fuzzy contours, while the area decoder incorporates a Multi-layer Decoder (MLD) to preserve fine-grained details and accurately reconstruct small lesions by leveraging multi-scale deep semantic features. Furthermore, Auxiliary Deep Supervision (ADS) heads are added at multiple scales to both decoders, providing more accurate gradient feedback and further enhancing the discriminative capability of multi-scale features. We conduct extensive experiments on three datasets: the NIH Pancreas dataset, the MSD dataset, and a clinical pancreatic tumor dataset provided by collaborating hospitals. DB-MSMUNet achieves Dice Similarity Coefficients of 89.47%, 87.59%, and 89.02%, respectively, outperforming most existing state-of-the-art methods in terms of segmentation accuracy, edge preservation, and robustness across different datasets. These results demonstrate the effectiveness and generalizability of the proposed method for real-world pancreatic CT segmentation tasks.
>
---
#### [new 012] CounterVid: Counterfactual Video Generation for Mitigating Action and Temporal Hallucinations in Video-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于视频-语言模型任务，旨在解决动作和时间幻觉问题。通过生成反事实视频，构建合成数据集并提出优化方法，提升模型在动作识别和时间推理上的准确性。**

- **链接: [https://arxiv.org/pdf/2601.04778v1](https://arxiv.org/pdf/2601.04778v1)**

> **作者:** Tobia Poppi; Burak Uzkent; Amanmeet Garg; Lucas Porto; Garin Kessler; Yezhou Yang; Marcella Cornia; Lorenzo Baraldi; Rita Cucchiara; Florian Schiffers
>
> **摘要:** Video-language models (VLMs) achieve strong multimodal understanding but remain prone to hallucinations, especially when reasoning about actions and temporal order. Existing mitigation strategies, such as textual filtering or random video perturbations, often fail to address the root cause: over-reliance on language priors rather than fine-grained visual dynamics. We propose a scalable framework for counterfactual video generation that synthesizes videos differing only in actions or temporal structure while preserving scene context. Our pipeline combines multimodal LLMs for action proposal and editing guidance with diffusion-based image and video models to generate semantic hard negatives at scale. Using this framework, we build CounterVid, a synthetic dataset of ~26k preference pairs targeting action recognition and temporal reasoning. We further introduce MixDPO, a unified Direct Preference Optimization approach that jointly leverages textual and visual preferences. Fine-tuning Qwen2.5-VL with MixDPO yields consistent improvements, notably in temporal ordering, and transfers effectively to standard video hallucination benchmarks. Code and models will be made publicly available.
>
---
#### [new 013] Atlas 2 -- Foundation models for clinical deployment
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决病理基础模型在临床部署中的性能、鲁棒性和计算效率问题。通过构建三个新模型，提升了综合表现。**

- **链接: [https://arxiv.org/pdf/2601.05148v1](https://arxiv.org/pdf/2601.05148v1)**

> **作者:** Maximilian Alber; Timo Milbich; Alexandra Carpen-Amarie; Stephan Tietz; Jonas Dippel; Lukas Muttenthaler; Beatriz Perez Cancer; Alessandro Benetti; Panos Korfiatis; Elias Eulig; Jérôme Lüscher; Jiasen Wu; Sayed Abid Hashimi; Gabriel Dernbach; Simon Schallenberg; Neelay Shah; Moritz Krügener; Aniruddh Jammoria; Jake Matras; Patrick Duffy; Matt Redlon; Philipp Jurmeister; David Horst; Lukas Ruff; Klaus-Robert Müller; Frederick Klauschen; Andrew Norgan
>
> **摘要:** Pathology foundation models substantially advanced the possibilities in computational pathology -- yet tradeoffs in terms of performance, robustness, and computational requirements remained, which limited their clinical deployment. In this report, we present Atlas 2, Atlas 2-B, and Atlas 2-S, three pathology vision foundation models which bridge these shortcomings by showing state-of-the-art performance in prediction performance, robustness, and resource efficiency in a comprehensive evaluation across eighty public benchmarks. Our models were trained on the largest pathology foundation model dataset to date comprising 5.5 million histopathology whole slide images, collected from three medical institutions Charité - Universtätsmedizin Berlin, LMU Munich, and Mayo Clinic.
>
---
#### [new 014] FlowLet: Conditional 3D Brain MRI Synthesis using Wavelet Flow Matching
- **分类: cs.CV**

- **简介: 该论文提出FlowLet，用于生成条件化3D脑MRI，解决数据不足与偏差问题，提升脑龄预测性能。**

- **链接: [https://arxiv.org/pdf/2601.05212v1](https://arxiv.org/pdf/2601.05212v1)**

> **作者:** Danilo Danese; Angela Lombardi; Matteo Attimonelli; Giuseppe Fasano; Tommaso Di Noia
>
> **摘要:** Brain Magnetic Resonance Imaging (MRI) plays a central role in studying neurological development, aging, and diseases. One key application is Brain Age Prediction (BAP), which estimates an individual's biological brain age from MRI data. Effective BAP models require large, diverse, and age-balanced datasets, whereas existing 3D MRI datasets are demographically skewed, limiting fairness and generalizability. Acquiring new data is costly and ethically constrained, motivating generative data augmentation. Current generative methods are often based on latent diffusion models, which operate in learned low dimensional latent spaces to address the memory demands of volumetric MRI data. However, these methods are typically slow at inference, may introduce artifacts due to latent compression, and are rarely conditioned on age, thereby affecting the BAP performance. In this work, we propose FlowLet, a conditional generative framework that synthesizes age-conditioned 3D MRIs by leveraging flow matching within an invertible 3D wavelet domain, helping to avoid reconstruction artifacts and reducing computational demands. Experiments show that FlowLet generates high-fidelity volumes with few sampling steps. Training BAP models with data generated by FlowLet improves performance for underrepresented age groups, and region-based analysis confirms preservation of anatomical structures.
>
---
#### [new 015] Vision-Language Agents for Interactive Forest Change Analysis
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于遥感图像变化分析任务，旨在解决森林动态的像素级变化检测和语义描述问题。提出一种基于大语言模型的视觉-语言代理系统，并构建了Forest-Change数据集进行评估。**

- **链接: [https://arxiv.org/pdf/2601.04497v1](https://arxiv.org/pdf/2601.04497v1)**

> **作者:** James Brock; Ce Zhang; Nantheera Anantrasirichai
>
> **备注:** 5 pages, 4 figures, Submitted to IGARSS 2026
>
> **摘要:** Modern forest monitoring workflows increasingly benefit from the growing availability of high-resolution satellite imagery and advances in deep learning. Two persistent challenges in this context are accurate pixel-level change detection and meaningful semantic change captioning for complex forest dynamics. While large language models (LLMs) are being adapted for interactive data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored. To address this gap, we introduce an LLM-driven agent for integrated forest change analysis that supports natural language querying across multiple RSICI tasks. The proposed system builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration. To facilitate adaptation and evaluation in forest environments, we further introduce the Forest-Change dataset, which comprises bi-temporal satellite imagery, pixel-level change masks, and multi-granularity semantic change captions generated using a combination of human annotation and rule-based methods. Experimental results show that the proposed system achieves mIoU and BLEU-4 scores of 67.10% and 40.17% on the Forest-Change dataset, and 88.13% and 34.41% on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI benchmark for joint change detection and captioning. These results highlight the potential of interactive, LLM-driven RSICI systems to improve accessibility, interpretability, and efficiency of forest change analysis. All data and code are publicly available at https://github.com/JamesBrockUoB/ForestChat.
>
---
#### [new 016] Segmentation-Driven Monocular Shape from Polarization based on Physical Model
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决单目偏振形状（SfP）中的方位角模糊问题。通过分割驱动的方法和多尺度约束，提升重建精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.04776v1](https://arxiv.org/pdf/2601.04776v1)**

> **作者:** Jinyu Zhang; Xu Ma; Weili Chen; Gonzalo R. Arce
>
> **备注:** 11 pages, 10 figures, submittd to IEEE Transactions on Image Processing
>
> **摘要:** Monocular shape-from-polarization (SfP) leverages the intrinsic relationship between light polarization properties and surface geometry to recover surface normals from single-view polarized images, providing a compact and robust approach for three-dimensional (3D) reconstruction. Despite its potential, existing monocular SfP methods suffer from azimuth angle ambiguity, an inherent limitation of polarization analysis, that severely compromises reconstruction accuracy and stability. This paper introduces a novel segmentation-driven monocular SfP (SMSfP) framework that reformulates global shape recovery into a set of local reconstructions over adaptively segmented convex sub-regions. Specifically, a polarization-aided adaptive region growing (PARG) segmentation strategy is proposed to decompose the global convexity assumption into locally convex regions, effectively suppressing azimuth ambiguities and preserving surface continuity. Furthermore, a multi-scale fusion convexity prior (MFCP) constraint is developed to ensure local surface consistency and enhance the recovery of fine textural and structural details. Extensive experiments on both synthetic and real-world datasets validate the proposed approach, showing significant improvements in disambiguation accuracy and geometric fidelity compared with existing physics-based monocular SfP techniques.
>
---
#### [new 017] Beyond Binary Preference: Aligning Diffusion Models to Fine-grained Criteria by Decoupling Attributes
- **分类: cs.CV**

- **简介: 该论文属于扩散模型对齐任务，旨在解决传统方法难以匹配复杂、细粒度人类标准的问题。通过构建层次化评估标准并提出CPO框架，提升生成质量与专家一致性。**

- **链接: [https://arxiv.org/pdf/2601.04300v1](https://arxiv.org/pdf/2601.04300v1)**

> **作者:** Chenye Meng; Zejian Li; Zhongni Liu; Yize Li; Changle Xie; Kaixin Jia; Ling Yang; Huanghuang Deng; Shiying Ding; Shengyuan Zhang; Jiayi Li; Lingyun Sun
>
> **摘要:** Post-training alignment of diffusion models relies on simplified signals, such as scalar rewards or binary preferences. This limits alignment with complex human expertise, which is hierarchical and fine-grained. To address this, we first construct a hierarchical, fine-grained evaluation criteria with domain experts, which decomposes image quality into multiple positive and negative attributes organized in a tree structure. Building on this, we propose a two-stage alignment framework. First, we inject domain knowledge to an auxiliary diffusion model via Supervised Fine-Tuning. Second, we introduce Complex Preference Optimization (CPO) that extends DPO to align the target diffusion to our non-binary, hierarchical criteria. Specifically, we reformulate the alignment problem to simultaneously maximize the probability of positive attributes while minimizing the probability of negative attributes with the auxiliary diffusion. We instantiate our approach in the domain of painting generation and conduct CPO training with an annotated dataset of painting with fine-grained attributes based on our criteria. Extensive experiments demonstrate that CPO significantly enhances generation quality and alignment with expertise, opening new avenues for fine-grained criteria alignment.
>
---
#### [new 018] FaceRefiner: High-Fidelity Facial Texture Refinement with Differentiable Rendering-based Style Transfer
- **分类: cs.CV**

- **简介: 该论文属于面部纹理生成任务，旨在解决单图生成纹理与输入不一致的问题。提出FaceRefiner，通过结合风格迁移与可微渲染，提升纹理质量与身份一致性。**

- **链接: [https://arxiv.org/pdf/2601.04520v1](https://arxiv.org/pdf/2601.04520v1)**

> **作者:** Chengyang Li; Baoping Cheng; Yao Cheng; Haocheng Zhang; Renshuai Liu; Yinglin Zheng; Jing Liao; Xuan Cheng
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** Recent facial texture generation methods prefer to use deep networks to synthesize image content and then fill in the UV map, thus generating a compelling full texture from a single image. Nevertheless, the synthesized texture UV map usually comes from a space constructed by the training data or the 2D face generator, which limits the methods' generalization ability for in-the-wild input images. Consequently, their facial details, structures and identity may not be consistent with the input. In this paper, we address this issue by proposing a style transfer-based facial texture refinement method named FaceRefiner. FaceRefiner treats the 3D sampled texture as style and the output of a texture generation method as content. The photo-realistic style is then expected to be transferred from the style image to the content image. Different from current style transfer methods that only transfer high and middle level information to the result, our style transfer method integrates differentiable rendering to also transfer low level (or pixel level) information in the visible face regions. The main benefit of such multi-level information transfer is that, the details, structures and semantics in the input can thus be well preserved. The extensive experiments on Multi-PIE, CelebA and FFHQ datasets demonstrate that our refinement method can improve the texture quality and the face identity preserving ability, compared with state-of-the-arts.
>
---
#### [new 019] Higher-Order Adversarial Patches for Real-Time Object Detectors
- **分类: cs.CV**

- **简介: 该论文研究对抗攻击对实时目标检测器的影响，提出高阶对抗补丁，并验证其泛化能力。任务为对抗攻击与防御，解决检测器在面对高阶攻击时的脆弱性问题。**

- **链接: [https://arxiv.org/pdf/2601.04991v1](https://arxiv.org/pdf/2601.04991v1)**

> **作者:** Jens Bayer; Stefan Becker; David Münch; Michael Arens; Jürgen Beyerer
>
> **备注:** Under review (ICPR2026)
>
> **摘要:** Higher-order adversarial attacks can directly be considered the result of a cat-and-mouse game -- an elaborate action involving constant pursuit, near captures, and repeated escapes. This idiom describes the enduring circular training of adversarial attack patterns and adversarial training the best. The following work investigates the impact of higher-order adversarial attacks on object detectors by successively training attack patterns and hardening object detectors with adversarial training. The YOLOv10 object detector is chosen as a representative, and adversarial patches are used in an evasion attack manner. Our results indicate that higher-order adversarial patches are not only affecting the object detector directly trained on but rather provide a stronger generalization capacity compared to lower-order adversarial patches. Moreover, the results highlight that solely adversarial training is not sufficient to harden an object detector efficiently against this kind of adversarial attack. Code: https://github.com/JensBayer/HigherOrder
>
---
#### [new 020] VERSE: Visual Embedding Reduction and Space Exploration. Clustering-Guided Insights for Training Data Enhancement in Visually-Rich Document Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VERSE方法，用于改进视觉语言模型在视觉丰富文档理解中的表现。通过分析视觉嵌入空间，识别问题区域并生成合成数据增强模型性能。**

- **链接: [https://arxiv.org/pdf/2601.05125v1](https://arxiv.org/pdf/2601.05125v1)**

> **作者:** Ignacio de Rodrigo; Alvaro J. Lopez-Lopez; Jaime Boal
>
> **摘要:** This work introduces VERSE, a methodology for analyzing and improving Vision-Language Models applied to Visually-rich Document Understanding by exploring their visual embedding space. VERSE enables the visualization of latent representations, supporting the assessment of model feasibility. It also facilitates the identification of problematic regions and guides the generation of synthetic data to enhance performance in those clusters. We validate the methodology by training on the synthetic MERIT Dataset and evaluating on its real-world counterpart, MERIT Secret. Results show that VERSE helps uncover the visual features associated with error-prone clusters, and that retraining with samples containing these features substantially boosts F1 performance without degrading generalization. Furthermore, we demonstrate that on-premise models such as Donut and Idefics2, when optimized with VERSE, match or even surpass the performance of SaaS solutions like GPT-4 and Pixtral.
>
---
#### [new 021] Scaling Vision Language Models for Pharmaceutical Long Form Video Reasoning on Industrial GenAI Platform
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态视频理解任务，针对工业场景下的长视频处理问题，提出一个大规模框架并分析现有模型的性能与限制。**

- **链接: [https://arxiv.org/pdf/2601.04891v1](https://arxiv.org/pdf/2601.04891v1)**

> **作者:** Suyash Mishra; Qiang Li; Srikanth Patil; Satyanarayan Pati; Baddu Narendra
>
> **备注:** Submitted to the Industry Track of Top Tier Conference; currently under peer review
>
> **摘要:** Vision Language Models (VLMs) have shown strong performance on multimodal reasoning tasks, yet most evaluations focus on short videos and assume unconstrained computational resources. In industrial settings such as pharmaceutical content understanding, practitioners must process long-form videos under strict GPU, latency, and cost constraints, where many existing approaches fail to scale. In this work, we present an industrial GenAI framework that processes over 200,000 PDFs, 25,326 videos across eight formats (e.g., MP4, M4V, etc.), and 888 multilingual audio files in more than 20 languages. Our study makes three contributions: (i) an industrial large-scale architecture for multimodal reasoning in pharmaceutical domains; (ii) empirical analysis of over 40 VLMs on two leading benchmarks (Video-MME and MMBench) and proprietary dataset of 25,326 videos across 14 disease areas; and (iii) four findings relevant to long-form video reasoning: the role of multimodality, attention mechanism trade-offs, temporal reasoning limits, and challenges of video splitting under GPU constraints. Results show 3-8 times efficiency gains with SDPA attention on commodity GPUs, multimodality improving up to 8/12 task domains (especially length-dependent tasks), and clear bottlenecks in temporal alignment and keyframe detection across open- and closed-source VLMs. Rather than proposing a new "A+B" model, this paper characterizes practical limits, trade-offs, and failure patterns of current VLMs under realistic deployment constraints, and provide actionable guidance for both researchers and practitioners designing scalable multimodal systems for long-form video understanding in industrial domains.
>
---
#### [new 022] UniDrive-WM: Unified Understanding, Planning and Generation World Model For Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决场景理解与轨迹规划问题。提出UniDrive-WM，整合感知、预测与生成，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2601.04453v1](https://arxiv.org/pdf/2601.04453v1)**

> **作者:** Zhexiao Xiong; Xin Ye; Burhan Yaman; Sheng Cheng; Yiren Lu; Jingru Luo; Nathan Jacobs; Liu Ren
>
> **备注:** Project Page: https://unidrive-wm.github.io/UniDrive-WM
>
> **摘要:** World models have become central to autonomous driving, where accurate scene understanding and future prediction are crucial for safe control. Recent work has explored using vision-language models (VLMs) for planning, yet existing approaches typically treat perception, prediction, and planning as separate modules. We propose UniDrive-WM, a unified VLM-based world model that jointly performs driving-scene understanding, trajectory planning, and trajectory-conditioned future image generation within a single architecture. UniDrive-WM's trajectory planner predicts a future trajectory, which conditions a VLM-based image generator to produce plausible future frames. These predictions provide additional supervisory signals that enhance scene understanding and iteratively refine trajectory generation. We further compare discrete and continuous output representations for future image prediction, analyzing their influence on downstream driving performance. Experiments on the challenging Bench2Drive benchmark show that UniDrive-WM produces high-fidelity future images and improves planning performance by 5.9% in L2 trajectory error and 9.2% in collision rate over the previous best method. These results demonstrate the advantages of tightly integrating VLM-driven reasoning, planning, and generative world modeling for autonomous driving. The project page is available at https://unidrive-wm.github.io/UniDrive-WM .
>
---
#### [new 023] UniLiPs: Unified LiDAR Pseudo-Labeling with Geometry-Grounded Dynamic Scene Decomposition
- **分类: cs.CV**

- **简介: 该论文提出UniLiPs方法，解决自动驾驶中无标签LiDAR数据利用难题，通过几何一致性进行3D语义标注和目标检测。**

- **链接: [https://arxiv.org/pdf/2601.05105v1](https://arxiv.org/pdf/2601.05105v1)**

> **作者:** Filippo Ghilotti; Samuel Brucker; Nahku Saidy; Matteo Matteucci; Mario Bijelic; Felix Heide
>
> **摘要:** Unlabeled LiDAR logs, in autonomous driving applications, are inherently a gold mine of dense 3D geometry hiding in plain sight - yet they are almost useless without human labels, highlighting a dominant cost barrier for autonomous-perception research. In this work we tackle this bottleneck by leveraging temporal-geometric consistency across LiDAR sweeps to lift and fuse cues from text and 2D vision foundation models directly into 3D, without any manual input. We introduce an unsupervised multi-modal pseudo-labeling method relying on strong geometric priors learned from temporally accumulated LiDAR maps, alongside with a novel iterative update rule that enforces joint geometric-semantic consistency, and vice-versa detecting moving objects from inconsistencies. Our method simultaneously produces 3D semantic labels, 3D bounding boxes, and dense LiDAR scans, demonstrating robust generalization across three datasets. We experimentally validate that our method compares favorably to existing semantic segmentation and object detection pseudo-labeling methods, which often require additional manual supervision. We confirm that even a small fraction of our geometrically consistent, densified LiDAR improves depth prediction by 51.5% and 22.0% MAE in the 80-150 and 150-250 meters range, respectively.
>
---
#### [new 024] TEA: Temporal Adaptive Satellite Image Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于卫星图像语义分割任务，解决时序长度变化下的分割性能下降问题。提出TEA方法，通过教师-学生框架和全序列重建提升模型适应不同时间长度的能力。**

- **链接: [https://arxiv.org/pdf/2601.04956v1](https://arxiv.org/pdf/2601.04956v1)**

> **作者:** Juyuan Kang; Hao Zhu; Yan Zhu; Wei Zhang; Jianing Chen; Tianxiang Xiao; Yike Ma; Hao Jiang; Feng Dai
>
> **备注:** Under review. Code will be available at \href{https://github.com/KeplerKang/TEA}{this https URL}
>
> **摘要:** Crop mapping based on satellite images time-series (SITS) holds substantial economic value in agricultural production settings, in which parcel segmentation is an essential step. Existing approaches have achieved notable advancements in SITS segmentation with predetermined sequence lengths. However, we found that these approaches overlooked the generalization capability of models across scenarios with varying temporal length, leading to markedly poor segmentation results in such cases. To address this issue, we propose TEA, a TEmporal Adaptive SITS semantic segmentation method to enhance the model's resilience under varying sequence lengths. We introduce a teacher model that encapsulates the global sequence knowledge to guide a student model with adaptive temporal input lengths. Specifically, teacher shapes the student's feature space via intermediate embedding, prototypes and soft label perspectives to realize knowledge transfer, while dynamically aggregating student model to mitigate knowledge forgetting. Finally, we introduce full-sequence reconstruction as an auxiliary task to further enhance the quality of representations across inputs of varying temporal lengths. Through extensive experiments, we demonstrate that our method brings remarkable improvements across inputs of different temporal lengths on common benchmarks. Our code will be publicly available.
>
---
#### [new 025] QNeRF: Neural Radiance Fields on a Simulated Gate-Based Quantum Computer
- **分类: cs.CV**

- **简介: 该论文提出QNeRF，一种用于从2D图像生成新视角的量子-经典混合模型，解决传统NeRF参数多、训练成本高的问题。**

- **链接: [https://arxiv.org/pdf/2601.05250v1](https://arxiv.org/pdf/2601.05250v1)**

> **作者:** Daniele Lizzio Bosco; Shuteng Wang; Giuseppe Serra; Vladislav Golyanik
>
> **备注:** 30 pages, 15 figures, 11 tables; project page: https://4dqv.mpi-inf.mpg.de/QNeRF/
>
> **摘要:** Recently, Quantum Visual Fields (QVFs) have shown promising improvements in model compactness and convergence speed for learning the provided 2D or 3D signals. Meanwhile, novel-view synthesis has seen major advances with Neural Radiance Fields (NeRFs), where models learn a compact representation from 2D images to render 3D scenes, albeit at the cost of larger models and intensive training. In this work, we extend the approach of QVFs by introducing QNeRF, the first hybrid quantum-classical model designed for novel-view synthesis from 2D images. QNeRF leverages parameterised quantum circuits to encode spatial and view-dependent information via quantum superposition and entanglement, resulting in more compact models compared to the classical counterpart. We present two architectural variants. Full QNeRF maximally exploits all quantum amplitudes to enhance representational capabilities. In contrast, Dual-Branch QNeRF introduces a task-informed inductive bias by branching spatial and view-dependent quantum state preparations, drastically reducing the complexity of this operation and ensuring scalability and potential hardware compatibility. Our experiments demonstrate that -- when trained on images of moderate resolution -- QNeRF matches or outperforms classical NeRF baselines while using less than half the number of parameters. These results suggest that quantum machine learning can serve as a competitive alternative for continuous signal representation in mid-level tasks in computer vision, such as 3D representation learning from 2D observations.
>
---
#### [new 026] Detector-Augmented SAMURAI for Long-Duration Drone Tracking
- **分类: cs.CV**

- **简介: 该论文属于目标跟踪任务，解决无人机长时间跟踪中的稳定性问题。通过改进SAMURAI模型，引入检测器增强，提升复杂环境下的跟踪鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.04798v1](https://arxiv.org/pdf/2601.04798v1)**

> **作者:** Tamara R. Lenhard; Andreas Weinmann; Hichem Snoussi; Tobias Koch
>
> **备注:** Accepted at the WACV 2026 Workshop on "Real World Surveillance: Applications and Challenges"
>
> **摘要:** Robust long-term tracking of drone is a critical requirement for modern surveillance systems, given their increasing threat potential. While detector-based approaches typically achieve strong frame-level accuracy, they often suffer from temporal inconsistencies caused by frequent detection dropouts. Despite its practical relevance, research on RGB-based drone tracking is still limited and largely reliant on conventional motion models. Meanwhile, foundation models like SAMURAI have established their effectiveness across other domains, exhibiting strong category-agnostic tracking performance. However, their applicability in drone-specific scenarios has not been investigated yet. Motivated by this gap, we present the first systematic evaluation of SAMURAI's potential for robust drone tracking in urban surveillance settings. Furthermore, we introduce a detector-augmented extension of SAMURAI to mitigate sensitivity to bounding-box initialization and sequence length. Our findings demonstrate that the proposed extension significantly improves robustness in complex urban environments, with pronounced benefits in long-duration sequences - especially under drone exit-re-entry events. The incorporation of detector cues yields consistent gains over SAMURAI's zero-shot performance across datasets and metrics, with success rate improvements of up to +0.393 and FNR reductions of up to -0.475.
>
---
#### [new 027] HyperAlign: Hyperbolic Entailment Cones for Adaptive Text-to-Image Alignment Assessment
- **分类: cs.CV**

- **简介: 该论文属于文本-图像对齐评估任务，旨在解决现有方法依赖欧几里得空间、缺乏适应性的问题。提出HyperAlign框架，利用双曲几何建模实现更准确的对齐评估。**

- **链接: [https://arxiv.org/pdf/2601.04614v1](https://arxiv.org/pdf/2601.04614v1)**

> **作者:** Wenzhi Chen; Bo Hu; Leida Li; Lihuo He; Wen Lu; Xinbo Gao
>
> **摘要:** With the rapid development of text-to-image generation technology, accurately assessing the alignment between generated images and text prompts has become a critical challenge. Existing methods rely on Euclidean space metrics, neglecting the structured nature of semantic alignment, while lacking adaptive capabilities for different samples. To address these limitations, we propose HyperAlign, an adaptive text-to-image alignment assessment framework based on hyperbolic entailment geometry. First, we extract Euclidean features using CLIP and map them to hyperbolic space. Second, we design a dynamic-supervision entailment modeling mechanism that transforms discrete entailment logic into continuous geometric structure supervision. Finally, we propose an adaptive modulation regressor that utilizes hyperbolic geometric features to generate sample-level modulation parameters, adaptively calibrating Euclidean cosine similarity to predict the final score. HyperAlign achieves highly competitive performance on both single database evaluation and cross-database generalization tasks, fully validating the effectiveness of hyperbolic geometric modeling for image-text alignment assessment.
>
---
#### [new 028] Plenoptic Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决多视角一致性问题。提出PlenopticDreamer框架，通过同步生成和条件检索提升视频时空连贯性与视觉质量。**

- **链接: [https://arxiv.org/pdf/2601.05239v1](https://arxiv.org/pdf/2601.05239v1)**

> **作者:** Xiao Fu; Shitao Tang; Min Shi; Xian Liu; Jinwei Gu; Ming-Yu Liu; Dahua Lin; Chen-Hsuan Lin
>
> **备注:** Project Page: https://research.nvidia.com/labs/dir/plenopticdreamer/
>
> **摘要:** Camera-controlled generative video re-rendering methods, such as ReCamMaster, have achieved remarkable progress. However, despite their success in single-view setting, these works often struggle to maintain consistency across multi-view scenarios. Ensuring spatio-temporal coherence in hallucinated regions remains challenging due to the inherent stochasticity of generative models. To address it, we introduce PlenopticDreamer, a framework that synchronizes generative hallucinations to maintain spatio-temporal memory. The core idea is to train a multi-in-single-out video-conditioned model in an autoregressive manner, aided by a camera-guided video retrieval strategy that adaptively selects salient videos from previous generations as conditional inputs. In addition, Our training incorporates progressive context-scaling to improve convergence, self-conditioning to enhance robustness against long-range visual degradation caused by error accumulation, and a long-video conditioning mechanism to support extended video generation. Extensive experiments on the Basic and Agibot benchmarks demonstrate that PlenopticDreamer achieves state-of-the-art video re-rendering, delivering superior view synchronization, high-fidelity visuals, accurate camera control, and diverse view transformations (e.g., third-person to third-person, and head-view to gripper-view in robotic manipulation). Project page: https://research.nvidia.com/labs/dir/plenopticdreamer/
>
---
#### [new 029] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型任务，解决LVLMs的过度思考问题。通过GPRO方法优化感知与推理路径，提升准确率和效率。**

- **链接: [https://arxiv.org/pdf/2601.04442v1](https://arxiv.org/pdf/2601.04442v1)**

> **作者:** Xingjian Diao; Zheyuan Liu; Chunhui Zhang; Weiyi Wu; Keyi Kong; Lin Shi; Kaize Ding; Soroush Vosoughi; Jiang Gui
>
> **摘要:** Large Vision-Language Models (LVLMs) have exhibited strong reasoning capabilities through chain-of-thought mechanisms that generate step-by-step rationales. However, such slow-thinking approaches often lead to overthinking, where models produce excessively verbose responses even for simple queries, resulting in test-time inefficiency and even degraded accuracy. Prior work has attempted to mitigate this issue via adaptive reasoning strategies, but these methods largely overlook a fundamental bottleneck: visual perception failures. We argue that stable reasoning critically depends on low-level visual grounding, and that reasoning errors often originate from imperfect perception rather than insufficient deliberation. To address this limitation, we propose Gated Perception-Reasoning Optimization (GPRO), a meta-reasoning controller that dynamically routes computation among three decision paths at each generation step: a lightweight fast path, a slow perception path for re-examining visual inputs, and a slow reasoning path for internal self-reflection. To learn this distinction, we derive large-scale failure attribution supervision from approximately 790k samples, using teacher models to distinguish perceptual hallucinations from reasoning errors. We then train the controller with multi-objective reinforcement learning to optimize the trade-off between task accuracy and computational cost under uncertainty. Experiments on five benchmarks demonstrate that GPRO substantially improves both accuracy and efficiency, outperforming recent slow-thinking methods while generating significantly shorter responses.
>
---
#### [new 030] PackCache: A Training-Free Acceleration Method for Unified Autoregressive Video Generation via Compact KV-Cache
- **分类: cs.CV**

- **简介: 该论文提出PackCache，用于加速统一自回归视频生成任务中的KV缓存管理，解决缓存过大导致的效率和生成长度限制问题。通过动态压缩缓存提升生成速度。**

- **链接: [https://arxiv.org/pdf/2601.04359v1](https://arxiv.org/pdf/2601.04359v1)**

> **作者:** Kunyang Li; Mubarak Shah; Yuzhang Shang
>
> **摘要:** A unified autoregressive model is a Transformer-based framework that addresses diverse multimodal tasks (e.g., text, image, video) as a single sequence modeling problem under a shared token space. Such models rely on the KV-cache mechanism to reduce attention computation from O(T^2) to O(T); however, KV-cache size grows linearly with the number of generated tokens, and it rapidly becomes the dominant bottleneck limiting inference efficiency and generative length. Unified autoregressive video generation inherits this limitation. Our analysis reveals that KV-cache tokens exhibit distinct spatiotemporal properties: (i) text and conditioning-image tokens act as persistent semantic anchors that consistently receive high attention, and (ii) attention to previous frames naturally decays with temporal distance. Leveraging these observations, we introduce PackCache, a training-free KV-cache management method that dynamically compacts the KV cache through three coordinated mechanisms: condition anchoring that preserves semantic references, cross-frame decay modeling that allocates cache budget according to temporal distance, and spatially preserving position embedding that maintains coherent 3D structure under cache removal. In terms of efficiency, PackCache accelerates end-to-end generation by 1.7-2.2x on 48-frame long sequences, showcasing its strong potential for enabling longer-sequence video generation. Notably, the final four frames - the portion most impacted by the progressively expanding KV-cache and thus the most expensive segment of the clip - PackCache delivers a 2.6x and 3.7x acceleration on A40 and H200, respectively, for 48-frame videos.
>
---
#### [new 031] Comparative Analysis of Custom CNN Architectures versus Pre-trained Models and Transfer Learning: A Study on Five Bangladesh Datasets
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像分类任务，比较自定义CNN与预训练模型在五个孟加拉国数据集上的表现，旨在解决模型选择与性能优化问题。**

- **链接: [https://arxiv.org/pdf/2601.04352v1](https://arxiv.org/pdf/2601.04352v1)**

> **作者:** Ibrahim Tanvir; Alif Ruslan; Sartaj Solaiman
>
> **摘要:** This study presents a comprehensive comparative analysis of custom-built Convolutional Neural Networks (CNNs) against popular pre-trained architectures (ResNet-18 and VGG-16) using both feature extraction and transfer learning approaches. We evaluated these models across five diverse image classification datasets from Bangladesh: Footpath Vision, Auto Rickshaw Detection, Mango Image Classification, Paddy Variety Recognition, and Road Damage Detection. Our experimental results demonstrate that transfer learning with fine-tuning consistently outperforms both custom CNNs built from scratch and feature extraction methods, achieving accuracy improvements ranging from 3% to 76% across different datasets. Notably, ResNet-18 with fine-tuning achieved perfect 100% accuracy on the Road Damage BD dataset. While custom CNNs offer advantages in model size (3.4M parameters vs. 11-134M for pre-trained models) and training efficiency on simpler tasks, pre-trained models with transfer learning provide superior performance, particularly on complex classification tasks with limited training data. This research provides practical insights for practitioners in selecting appropriate deep learning approaches based on dataset characteristics, computational resources, and performance requirements.
>
---
#### [new 032] PyramidalWan: On Making Pretrained Video Model Pyramidal for Efficient Inference
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在提升预训练扩散模型的推理效率。通过低代价微调将模型转为分层结构，提升效率且不损失质量。**

- **链接: [https://arxiv.org/pdf/2601.04792v1](https://arxiv.org/pdf/2601.04792v1)**

> **作者:** Denis Korzhenkov; Adil Karjauv; Animesh Karnewar; Mohsen Ghafoorian; Amirhossein Habibian
>
> **摘要:** Recently proposed pyramidal models decompose the conventional forward and backward diffusion processes into multiple stages operating at varying resolutions. These models handle inputs with higher noise levels at lower resolutions, while less noisy inputs are processed at higher resolutions. This hierarchical approach significantly reduces the computational cost of inference in multi-step denoising models. However, existing open-source pyramidal video models have been trained from scratch and tend to underperform compared to state-of-the-art systems in terms of visual plausibility. In this work, we present a pipeline that converts a pretrained diffusion model into a pyramidal one through low-cost finetuning, achieving this transformation without degradation in quality of output videos. Furthermore, we investigate and compare various strategies for step distillation within pyramidal models, aiming to further enhance the inference efficiency. Our results are available at https://qualcomm-ai-research.github.io/PyramidalWan.
>
---
#### [new 033] HATIR: Heat-Aware Diffusion for Turbulent Infrared Video Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于红外视频超分辨率任务，解决大气湍流和压缩退化问题。提出HATIR模型，结合热感知先验与扩散过程，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2601.04682v1](https://arxiv.org/pdf/2601.04682v1)**

> **作者:** Yang Zou; Xingyue Zhu; Kaiqi Han; Jun Ma; Xingyuan Li; Zhiying Jiang; Jinyuan Liu
>
> **摘要:** Infrared video has been of great interest in visual tasks under challenging environments, but often suffers from severe atmospheric turbulence and compression degradation. Existing video super-resolution (VSR) methods either neglect the inherent modality gap between infrared and visible images or fail to restore turbulence-induced distortions. Directly cascading turbulence mitigation (TM) algorithms with VSR methods leads to error propagation and accumulation due to the decoupled modeling of degradation between turbulence and resolution. We introduce HATIR, a Heat-Aware Diffusion for Turbulent InfraRed Video Super-Resolution, which injects heat-aware deformation priors into the diffusion sampling path to jointly model the inverse process of turbulent degradation and structural detail loss. Specifically, HATIR constructs a Phasor-Guided Flow Estimator, rooted in the physical principle that thermally active regions exhibit consistent phasor responses over time, enabling reliable turbulence-aware flow to guide the reverse diffusion process. To ensure the fidelity of structural recovery under nonuniform distortions, a Turbulence-Aware Decoder is proposed to selectively suppress unstable temporal cues and enhance edge-aware feature aggregation via turbulence gating and structure-aware attention. We built FLIR-IVSR, the first dataset for turbulent infrared VSR, comprising paired LR-HR sequences from a FLIR T1050sc camera (1024 X 768) spanning 640 diverse scenes with varying camera and object motion conditions. This encourages future research in infrared VSR. Project page: https://github.com/JZ0606/HATIR
>
---
#### [new 034] ProFuse: Efficient Cross-View Context Fusion for Open-Vocabulary 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 论文提出ProFuse，解决开放词汇3D场景理解问题。通过跨视角上下文融合，提升一致性与语义连贯性，实现高效3D高斯点云生成。**

- **链接: [https://arxiv.org/pdf/2601.04754v1](https://arxiv.org/pdf/2601.04754v1)**

> **作者:** Yen-Jen Chiou; Wei-Tse Cheng; Yuan-Fu Yang
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** We present ProFuse, an efficient context-aware framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS). The pipeline enhances cross-view consistency and intra-mask cohesion within a direct registration setup, adding minimal overhead and requiring no render-supervised fine-tuning. Instead of relying on a pretrained 3DGS scene, we introduce a dense correspondence-guided pre-registration phase that initializes Gaussians with accurate geometry while jointly constructing 3D Context Proposals via cross-view clustering. Each proposal carries a global feature obtained through weighted aggregation of member embeddings, and this feature is fused onto Gaussians during direct registration to maintain per-primitive language coherence across views. With associations established in advance, semantic fusion requires no additional optimization beyond standard reconstruction, and the model retains geometric refinement without densification. ProFuse achieves strong open-vocabulary 3DGS understanding while completing semantic attachment in about five minutes per scene, which is two times faster than SOTA.
>
---
#### [new 035] Pixel-Perfect Visual Geometry Estimation
- **分类: cs.CV**

- **简介: 该论文属于单目和视频深度估计任务，旨在解决几何重建中的像素漂移和细节丢失问题。提出Pixel-Perfect Depth模型，结合扩散Transformer提升深度预测精度与效率。**

- **链接: [https://arxiv.org/pdf/2601.05246v1](https://arxiv.org/pdf/2601.05246v1)**

> **作者:** Gangwei Xu; Haotong Lin; Hongcheng Luo; Haiyang Sun; Bing Wang; Guang Chen; Sida Peng; Hangjun Ye; Xin Yang
>
> **备注:** Code: https://github.com/gangweix/pixel-perfect-depth
>
> **摘要:** Recovering clean and accurate geometry from images is essential for robotics and augmented reality. However, existing geometry foundation models still suffer severely from flying pixels and the loss of fine details. In this paper, we present pixel-perfect visual geometry models that can predict high-quality, flying-pixel-free point clouds by leveraging generative modeling in the pixel space. We first introduce Pixel-Perfect Depth (PPD), a monocular depth foundation model built upon pixel-space diffusion transformers (DiT). To address the high computational complexity associated with pixel-space diffusion, we propose two key designs: 1) Semantics-Prompted DiT, which incorporates semantic representations from vision foundation models to prompt the diffusion process, preserving global semantics while enhancing fine-grained visual details; and 2) Cascade DiT architecture that progressively increases the number of image tokens, improving both efficiency and accuracy. To further extend PPD to video (PPVD), we introduce a new Semantics-Consistent DiT, which extracts temporally consistent semantics from a multi-view geometry foundation model. We then perform reference-guided token propagation within the DiT to maintain temporal coherence with minimal computational and memory overhead. Our models achieve the best performance among all generative monocular and video depth estimation models and produce significantly cleaner point clouds than all other models.
>
---
#### [new 036] ObjectForesight: Predicting Future 3D Object Trajectories from Human Videos
- **分类: cs.CV**

- **简介: 该论文提出ObjectForesight，解决从视频中预测3D物体未来运动轨迹的任务，通过建模物体动力学实现准确、几何一致的预测。**

- **链接: [https://arxiv.org/pdf/2601.05237v1](https://arxiv.org/pdf/2601.05237v1)**

> **作者:** Rustin Soraki; Homanga Bharadhwaj; Ali Farhadi; Roozbeh Mottaghi
>
> **备注:** Preprint. Project Website: objectforesight.github.io
>
> **摘要:** Humans can effortlessly anticipate how objects might move or change through interaction--imagining a cup being lifted, a knife slicing, or a lid being closed. We aim to endow computational systems with a similar ability to predict plausible future object motions directly from passive visual observation. We introduce ObjectForesight, a 3D object-centric dynamics model that predicts future 6-DoF poses and trajectories of rigid objects from short egocentric video sequences. Unlike conventional world or dynamics models that operate in pixel or latent space, ObjectForesight represents the world explicitly in 3D at the object level, enabling geometrically grounded and temporally coherent predictions that capture object affordances and trajectories. To train such a model at scale, we leverage recent advances in segmentation, mesh reconstruction, and 3D pose estimation to curate a dataset of 2 million plus short clips with pseudo-ground-truth 3D object trajectories. Through extensive experiments, we show that ObjectForesight achieves significant gains in accuracy, geometric consistency, and generalization to unseen objects and scenes, establishing a scalable framework for learning physically grounded, object-centric dynamics models directly from observation. objectforesight.github.io
>
---
#### [new 037] Character Detection using YOLO for Writer Identification in multiple Medieval books
- **分类: cs.CV**

- **简介: 论文属于手写识别任务，旨在解决中世纪文献中作者识别问题。通过使用YOLO模型替代传统方法，提高字符检测准确性，实现更可靠的作者识别。**

- **链接: [https://arxiv.org/pdf/2601.04834v1](https://arxiv.org/pdf/2601.04834v1)**

> **作者:** Alessandra Scotto di Freca; Tiziana D Alessandro; Francesco Fontanella; Filippo Sarria; Claudio De Stefano
>
> **备注:** 7 pages, 2 figures, 1 table. Accepted at IEEE-CH 2025
>
> **摘要:** Paleography is the study of ancient and historical handwriting, its key objectives include the dating of manuscripts and understanding the evolution of writing. Estimating when a document was written and tracing the development of scripts and writing styles can be aided by identifying the individual scribes who contributed to a medieval manuscript. Although digital technologies have made significant progress in this field, the general problem remains unsolved and continues to pose open challenges. ... We previously proposed an approach focused on identifying specific letters or abbreviations that characterize each writer. In that study, we considered the letter "a", as it was widely present on all pages of text and highly distinctive, according to the suggestions of expert paleographers. We used template matching techniques to detect the occurrences of the character "a" on each page and the convolutional neural network (CNN) to attribute each instance to the correct scribe. Moving from the interesting results achieved from this previous system and being aware of the limitations of the template matching technique, which requires an appropriate threshold to work, we decided to experiment in the same framework with the use of the YOLO object detection model to identify the scribe who contributed to the writing of different medieval books. We considered the fifth version of YOLO to implement the YOLO object detection model, which completely substituted the template matching and CNN used in the previous work. The experimental results demonstrate that YOLO effectively extracts a greater number of letters considered, leading to a more accurate second-stage classification. Furthermore, the YOLO confidence score provides a foundation for developing a system that applies a rejection threshold, enabling reliable writer identification even in unseen manuscripts.
>
---
#### [new 038] GeM-VG: Towards Generalized Multi-image Visual Grounding with Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多图像视觉定位任务，旨在解决现有方法在多图像接地任务中的局限性。提出GeM-VG模型，并引入新数据集和强化学习策略，提升模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.04777v1](https://arxiv.org/pdf/2601.04777v1)**

> **作者:** Shurong Zheng; Yousong Zhu; Hongyin Zhao; Fan Yang; Yufei Zhan; Ming Tang; Jinqiao Wang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated impressive progress in single-image grounding and general multi-image understanding. Recently, some methods begin to address multi-image grounding. However, they are constrained by single-target localization and limited types of practical tasks, due to the lack of unified modeling for generalized grounding tasks. Therefore, we propose GeM-VG, an MLLM capable of Generalized Multi-image Visual Grounding. To support this, we systematically categorize and organize existing multi-image grounding tasks according to their reliance of cross-image cues and reasoning, and introduce the MG-Data-240K dataset, addressing the limitations of existing datasets regarding target quantity and image relation. To tackle the challenges of robustly handling diverse multi-image grounding tasks, we further propose a hybrid reinforcement finetuning strategy that integrates chain-of-thought (CoT) reasoning and direct answering, considering their complementary strengths. This strategy adopts an R1-like algorithm guided by a carefully designed rule-based reward, effectively enhancing the model's overall perception and reasoning capabilities. Extensive experiments demonstrate the superior generalized grounding capabilities of our model. For multi-image grounding, it outperforms the previous leading MLLMs by 2.0% and 9.7% on MIG-Bench and MC-Bench, respectively. In single-image grounding, it achieves a 9.1% improvement over the base model on ODINW. Furthermore, our model retains strong capabilities in general multi-image understanding.
>
---
#### [new 039] Integrated Framework for Selecting and Enhancing Ancient Marathi Inscription Images from Stone, Metal Plate, and Paper Documents
- **分类: cs.CV**

- **简介: 该论文属于图像增强任务，旨在提升古代马拉地铭文图像的可读性。针对背景噪声和文本模糊问题，提出基于二值化和预处理的方法，并在不同材质上验证效果。**

- **链接: [https://arxiv.org/pdf/2601.04800v1](https://arxiv.org/pdf/2601.04800v1)**

> **作者:** Bapu D. Chendage; Rajivkumar S. Mente
>
> **备注:** 9 Pages, 5 figures
>
> **摘要:** Ancient script images often suffer from severe background noise, low contrast, and degradation caused by aging and environmental effects. In many cases, the foreground text and background exhibit similar visual characteristics, making the inscriptions difficult to read. The primary objective of image enhancement is to improve the readability of such degraded ancient images. This paper presents an image enhancement approach based on binarization and complementary preprocessing techniques for removing stains and enhancing unclear ancient text. The proposed methods are evaluated on different types of ancient scripts, including inscriptions on stone, metal plates, and historical documents. Experimental results show that the proposed approach achieves classification accuracies of 55.7%, 62%, and 65.6% for stone, metal plate, and document scripts, respectively, using the K-Nearest Neighbor (K-NN) classifier. Using the Support Vector Machine (SVM) classifier, accuracies of 53.2%, 59.5%, and 67.8% are obtained. The results demonstrate the effectiveness of the proposed enhancement method in improving the readability of ancient Marathi inscription images.
>
---
#### [new 040] Unified Text-Image Generation with Weakness-Targeted Post-Training
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决传统系统需手动切换模态的问题。通过后训练实现统一生成，提升跨模态耦合与自动生成能力。**

- **链接: [https://arxiv.org/pdf/2601.04339v1](https://arxiv.org/pdf/2601.04339v1)**

> **作者:** Jiahui Chen; Philippe Hansen-Estruch; Xiaochuang Han; Yushi Hu; Emily Dinan; Amita Kamath; Michal Drozdzal; Reyhane Askari-Hemmat; Luke Zettlemoyer; Marjan Ghazvininejad
>
> **摘要:** Unified multimodal generation architectures that jointly produce text and images have recently emerged as a promising direction for text-to-image (T2I) synthesis. However, many existing systems rely on explicit modality switching, generating reasoning text before switching manually to image generation. This separate, sequential inference process limits cross-modal coupling and prohibits automatic multimodal generation. This work explores post-training to achieve fully unified text-image generation, where models autonomously transition from textual reasoning to visual synthesis within a single inference process. We examine the impact of joint text-image generation on T2I performance and the relative importance of each modality during post-training. We additionally explore different post-training data strategies, showing that a targeted dataset addressing specific limitations achieves superior results compared to broad image-caption corpora or benchmark-aligned data. Using offline, reward-weighted post-training with fully self-generated synthetic data, our approach enables improvements in multimodal image generation across four diverse T2I benchmarks, demonstrating the effectiveness of reward-weighting both modalities and strategically designed post-training data.
>
---
#### [new 041] Few-Shot LoRA Adaptation of a Flow-Matching Foundation Model for Cross-Spectral Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跨谱目标检测任务，旨在解决可见光与非可见光模态间数据不足的问题。通过少量样本微调流匹配基础模型，生成合成红外和SAR图像，提升目标检测性能。**

- **链接: [https://arxiv.org/pdf/2601.04381v1](https://arxiv.org/pdf/2601.04381v1)**

> **作者:** Maxim Clouser; Kia Khezeli; John Kalantari
>
> **摘要:** Foundation models for vision are predominantly trained on RGB data, while many safety-critical applications rely on non-visible modalities such as infrared (IR) and synthetic aperture radar (SAR). We study whether a single flow-matching foundation model pre-trained primarily on RGB images can be repurposed as a cross-spectral translator using only a few co-measured examples, and whether the resulting synthetic data can enhance downstream detection. Starting from FLUX.1 Kontext, we insert low-rank adaptation (LoRA) modules and fine-tune them on just 100 paired images per domain for two settings: RGB to IR on the KAIST dataset and RGB to SAR on the M4-SAR dataset. The adapted model translates RGB images into pixel-aligned IR/SAR, enabling us to reuse existing bounding boxes and train object detection models purely in the target modality. Across a grid of LoRA hyperparameters, we find that LPIPS computed on only 50 held-out pairs is a strong proxy for downstream performance: lower LPIPS consistently predicts higher mAP for YOLOv11n on both IR and SAR, and for DETR on KAIST IR test data. Using the best LPIPS-selected LoRA adapter, synthetic IR from external RGB datasets (LLVIP, FLIR ADAS) improves KAIST IR pedestrian detection, and synthetic SAR significantly boosts infrastructure detection on M4-SAR when combined with limited real SAR. Our results suggest that few-shot LoRA adaptation of flow-matching foundation models is a promising path toward foundation-style support for non-visible modalities.
>
---
#### [new 042] 3D Conditional Image Synthesis of Left Atrial LGE MRI from Composite Semantic Masks
- **分类: cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决左心房LGE MRI数据不足导致的分割困难问题。通过3D条件生成模型合成高质量图像，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2601.04588v1](https://arxiv.org/pdf/2601.04588v1)**

> **作者:** Yusri Al-Sanaani; Rebecca Thornhill; Sreeraman Rajan
>
> **备注:** This work has been published in the Proceedings of the 2025 IEEE International Conference on Imaging Systems and Techniques (IST). The final published version is available via IEEE Xplore
>
> **摘要:** Segmentation of the left atrial (LA) wall and endocardium from late gadolinium-enhanced (LGE) MRI is essential for quantifying atrial fibrosis in patients with atrial fibrillation. The development of accurate machine learning-based segmentation models remains challenging due to the limited availability of data and the complexity of anatomical structures. In this work, we investigate 3D conditional generative models as potential solution for augmenting scarce LGE training data and improving LA segmentation performance. We develop a pipeline to synthesize high-fidelity 3D LGE MRI volumes from composite semantic label maps combining anatomical expert annotations with unsupervised tissue clusters, using three 3D conditional generators (Pix2Pix GAN, SPADE-GAN, and SPADE-LDM). The synthetic images are evaluated for realism and their impact on downstream LA segmentation. SPADE-LDM generates the most realistic and structurally accurate images, achieving an FID of 4.063 and surpassing GAN models, which have FIDs of 40.821 and 7.652 for Pix2Pix and SPADE-GAN, respectively. When augmented with synthetic LGE images, the Dice score for LA cavity segmentation with a 3D U-Net model improved from 0.908 to 0.936, showing a statistically significant improvement (p < 0.05) over the baseline.These findings demonstrate the potential of label-conditioned 3D synthesis to enhance the segmentation of under-represented cardiac structures.
>
---
#### [new 043] Prototypicality Bias Reveals Blindspots in Multimodal Evaluation Metrics
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本-图像生成模型评估任务，旨在解决现有评价指标可能偏向视觉原型而非语义正确性的问题。通过构建对比基准，发现主流指标存在偏差，并提出更可靠的ProtoScore。**

- **链接: [https://arxiv.org/pdf/2601.04946v1](https://arxiv.org/pdf/2601.04946v1)**

> **作者:** Subhadeep Roy; Gagan Bhatia; Steffen Eger
>
> **备注:** First version
>
> **摘要:** Automatic metrics are now central to evaluating text-to-image models, often substituting for human judgment in benchmarking and large-scale filtering. However, it remains unclear whether these metrics truly prioritize semantic correctness or instead favor visually and socially prototypical images learned from biased data distributions. We identify and study \emph{prototypicality bias} as a systematic failure mode in multimodal evaluation. We introduce a controlled contrastive benchmark \textsc{\textbf{ProtoBias}} (\textit{\textbf{Proto}typical \textbf{Bias}}), spanning Animals, Objects, and Demography images, where semantically correct but non-prototypical images are paired with subtly incorrect yet prototypical adversarial counterparts. This setup enables a directional evaluation of whether metrics follow textual semantics or default to prototypes. Our results show that widely used metrics, including CLIPScore, PickScore, and VQA-based scores, frequently misrank these pairs, while even LLM-as-Judge systems exhibit uneven robustness in socially grounded cases. Human evaluations consistently favour semantic correctness with larger decision margins. Motivated by these findings, we propose \textbf{\textsc{ProtoScore}}, a robust 7B-parameter metric that substantially reduces failure rates and suppresses misranking, while running at orders of magnitude faster than the inference time of GPT-5, approaching the robustness of much larger closed-source judges.
>
---
#### [new 044] A Lightweight and Explainable Vision-Language Framework for Crop Disease Visual Question Answering
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于作物病害视觉问答任务，旨在提升作物和病害识别的准确性与可解释性。通过结合Swin Transformer和序列解码器，提出轻量级框架，优化跨模态对齐，实现高效准确的视觉语言理解。**

- **链接: [https://arxiv.org/pdf/2601.05143v1](https://arxiv.org/pdf/2601.05143v1)**

> **作者:** Md. Zahid Hossain; Most. Sharmin Sultana Samu; Md. Rakibul Islam; Md. Siam Ansary
>
> **备注:** Preprint, manuscript is under review
>
> **摘要:** Visual question answering for crop disease analysis requires accurate visual understanding and reliable language generation. This work presents a lightweight vision-language framework for crop and disease identification from leaf images. The proposed approach combines a Swin Transformer vision encoder with sequence-to-sequence language decoders. A two-stage training strategy is adopted to improve visual representation learning and cross-modal alignment. The model is evaluated on a large-scale crop disease dataset using classification and natural language generation metrics. Experimental results show high accuracy for both crop and disease identification. The framework also achieves strong performance on BLEU, ROUGE and BERTScore. Our proposed models outperform large-scale vision-language baselines while using significantly fewer parameters. Explainability is assessed using Grad-CAM and token-level attribution. Qualitative results demonstrate robust performance under diverse user-driven queries. These findings highlight the effectiveness of task-specific visual pretraining for crop disease visual question answering.
>
---
#### [new 045] SCAR-GS: Spatial Context Attention for Residuals in Progressive Gaussian Splatting
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D重建任务，解决大场景下高存储需求问题。提出SCAR-GS方法，用残差向量量化替代传统方法，提升压缩效率。**

- **链接: [https://arxiv.org/pdf/2601.04348v1](https://arxiv.org/pdf/2601.04348v1)**

> **作者:** Diego Revilla; Pooja Suresh; Anand Bhojan; Ooi Wei Tsang
>
> **摘要:** Recent advances in 3D Gaussian Splatting have allowed for real-time, high-fidelity novel view synthesis. Nonetheless, these models have significant storage requirements for large and medium-sized scenes, hindering their deployment over cloud and streaming services. Some of the most recent progressive compression techniques for these models rely on progressive masking and scalar quantization techniques to reduce the bitrate of Gaussian attributes using spatial context models. While effective, scalar quantization may not optimally capture the correlations of high-dimensional feature vectors, which can potentially limit the rate-distortion performance. In this work, we introduce a novel progressive codec for 3D Gaussian Splatting that replaces traditional methods with a more powerful Residual Vector Quantization approach to compress the primitive features. Our key contribution is an auto-regressive entropy model, guided by a multi-resolution hash grid, that accurately predicts the conditional probability of each successive transmitted index, allowing for coarse and refinement layers to be compressed with high efficiency.
>
---
#### [new 046] MoE3D: A Mixture-of-Experts Module for 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决深度边界模糊和飞点伪影问题。提出MoE3D模块，通过预测多个深度图并动态融合提升重建质量。**

- **链接: [https://arxiv.org/pdf/2601.05208v1](https://arxiv.org/pdf/2601.05208v1)**

> **作者:** Zichen Wang; Ang Cao; Liam J. Wang; Jeong Joon Park
>
> **摘要:** MoE3D is a mixture-of-experts module designed to sharpen depth boundaries and mitigate flying-point artifacts (highlighted in red) of existing feed-forward 3D reconstruction models (left side). MoE3D predicts multiple candidate depth maps and fuses them via dynamic weighting (visualized by MoE weights on the right side). When integrated with a pre-trained 3D reconstruction backbone such as VGGT, it substantially enhances reconstruction quality with minimal additional computational overhead. Best viewed digitally.
>
---
#### [new 047] Forge-and-Quench: Enhancing Image Generation for Higher Fidelity in Unified Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在提升图像生成的保真度。通过引入Forge-and-Quench框架，将理解模型的洞察融入生成过程，增强图像细节与真实性。**

- **链接: [https://arxiv.org/pdf/2601.04706v1](https://arxiv.org/pdf/2601.04706v1)**

> **作者:** Yanbing Zeng; Jia Wang; Hanghang Ma; Junqiang Wu; Jie Zhu; Xiaoming Wei; Jie Hu
>
> **摘要:** Integrating image generation and understanding into a single framework has become a pivotal goal in the multimodal domain. However, how understanding can effectively assist generation has not been fully explored. Unlike previous works that focus on leveraging reasoning abilities and world knowledge from understanding models, this paper introduces a novel perspective: leveraging understanding to enhance the fidelity and detail richness of generated images. To this end, we propose Forge-and-Quench, a new unified framework that puts this principle into practice. In the generation process of our framework, an MLLM first reasons over the entire conversational context, including text instructions, to produce an enhanced text instruction. This refined instruction is then mapped to a virtual visual representation, termed the Bridge Feature, via a novel Bridge Adapter. This feature acts as a crucial link, forging insights from the understanding model to quench and refine the generation process. It is subsequently injected into the T2I backbone as a visual guidance signal, alongside the enhanced text instruction that replaces the original input. To validate this paradigm, we conduct comprehensive studies on the design of the Bridge Feature and Bridge Adapter. Our framework demonstrates exceptional extensibility and flexibility, enabling efficient migration across different MLLM and T2I models with significant savings in training overhead, all without compromising the MLLM's inherent multimodal understanding capabilities. Experiments show that Forge-and-Quench significantly improves image fidelity and detail across multiple models, while also maintaining instruction-following accuracy and enhancing world knowledge application. Models and codes are available at https://github.com/YanbingZeng/Forge-and-Quench.
>
---
#### [new 048] From Rays to Projections: Better Inputs for Feed-Forward View Synthesis
- **分类: cs.CV**

- **简介: 该论文属于视图合成任务，解决现有方法对相机变换敏感、几何不一致的问题。通过引入投影条件代替原始相机参数，提升视图合成的鲁棒性和一致性。**

- **链接: [https://arxiv.org/pdf/2601.05116v1](https://arxiv.org/pdf/2601.05116v1)**

> **作者:** Zirui Wu; Zeren Jiang; Martin R. Oswald; Jie Song
>
> **备注:** Project Page: https://wuzirui.github.io/pvsm-web
>
> **摘要:** Feed-forward view synthesis models predict a novel view in a single pass with minimal 3D inductive bias. Existing works encode cameras as Plücker ray maps, which tie predictions to the arbitrary world coordinate gauge and make them sensitive to small camera transformations, thereby undermining geometric consistency. In this paper, we ask what inputs best condition a model for robust and consistent view synthesis. We propose projective conditioning, which replaces raw camera parameters with a target-view projective cue that provides a stable 2D input. This reframes the task from a brittle geometric regression problem in ray space to a well-conditioned target-view image-to-image translation problem. Additionally, we introduce a masked autoencoding pretraining strategy tailored to this cue, enabling the use of large-scale uncalibrated data for pretraining. Our method shows improved fidelity and stronger cross-view consistency compared to ray-conditioned baselines on our view-consistency benchmark. It also achieves state-of-the-art quality on standard novel view synthesis benchmarks.
>
---
#### [new 049] Rotation-Robust Regression with Convolutional Model Trees
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究图像旋转鲁棒性问题，提出基于卷积模型树的回归方法，通过几何归纳偏置和部署时方向搜索提升模型在旋转下的稳定性。**

- **链接: [https://arxiv.org/pdf/2601.04899v1](https://arxiv.org/pdf/2601.04899v1)**

> **作者:** Hongyi Li; William Ward Armstrong; Jun Xu
>
> **摘要:** We study rotation-robust learning for image inputs using Convolutional Model Trees (CMTs) [1], whose split and leaf coefficients can be structured on the image grid and transformed geometrically at deployment time. In a controlled MNIST setting with a rotation-invariant regression target, we introduce three geometry-aware inductive biases for split directions -- convolutional smoothing, a tilt dominance constraint, and importance-based pruning -- and quantify their impact on robustness under in-plane rotations. We further evaluate a deployment-time orientation search that selects a discrete rotation maximizing a forest-level confidence proxy without updating model parameters. Orientation search improves robustness under severe rotations but can be harmful near the canonical orientation when confidence is misaligned with correctness. Finally, we observe consistent trends on MNIST digit recognition implemented as one-vs-rest regression, highlighting both the promise and limitations of confidence-based orientation selection for model-tree ensembles.
>
---
#### [new 050] Multi-Scale Local Speculative Decoding for Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决AR模型生成速度慢的问题。提出MuLo-SD方法，通过多尺度草稿与空间验证加速生成，提升效率并保持质量。**

- **链接: [https://arxiv.org/pdf/2601.05149v1](https://arxiv.org/pdf/2601.05149v1)**

> **作者:** Elia Peruzzo; Guillaume Sautière; Amirhossein Habibian
>
> **备注:** Project page is available at https://qualcomm-ai-research.github.io/mulo-sd-webpage
>
> **摘要:** Autoregressive (AR) models have achieved remarkable success in image synthesis, yet their sequential nature imposes significant latency constraints. Speculative Decoding offers a promising avenue for acceleration, but existing approaches are limited by token-level ambiguity and lack of spatial awareness. In this work, we introduce Multi-Scale Local Speculative Decoding (MuLo-SD), a novel framework that combines multi-resolution drafting with spatially informed verification to accelerate AR image generation. Our method leverages a low-resolution drafter paired with learned up-samplers to propose candidate image tokens, which are then verified in parallel by a high-resolution target model. Crucially, we incorporate a local rejection and resampling mechanism, enabling efficient correction of draft errors by focusing on spatial neighborhoods rather than raster-scan resampling after the first rejection. We demonstrate that MuLo-SD achieves substantial speedups - up to $\mathbf{1.7\times}$ - outperforming strong speculative decoding baselines such as EAGLE-2 and LANTERN in terms of acceleration, while maintaining comparable semantic alignment and perceptual quality. These results are validated using GenEval, DPG-Bench, and FID/HPSv2 on the MS-COCO 5k validation split. Extensive ablations highlight the impact of up-sampling design, probability pooling, and local rejection and resampling with neighborhood expansion. Our approach sets a new state-of-the-art in speculative decoding for image synthesis, bridging the gap between efficiency and fidelity.
>
---
#### [new 051] Skeletonization-Based Adversarial Perturbations on Large Vision Language Model's Mathematical Text Recognition
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型安全研究任务，旨在解决数学文本识别中的对抗攻击问题。通过骨架化方法生成有效扰动，评估模型的视觉理解能力。**

- **链接: [https://arxiv.org/pdf/2601.04752v1](https://arxiv.org/pdf/2601.04752v1)**

> **作者:** Masatomo Yoshida; Haruto Namura; Nicola Adami; Masahiro Okuda
>
> **备注:** accepted to ITC-CSCC 2025
>
> **摘要:** This work explores the visual capabilities and limitations of foundation models by introducing a novel adversarial attack method utilizing skeletonization to reduce the search space effectively. Our approach specifically targets images containing text, particularly mathematical formula images, which are more challenging due to their LaTeX conversion and intricate structure. We conduct a detailed evaluation of both character and semantic changes between original and adversarially perturbed outputs to provide insights into the models' visual interpretation and reasoning abilities. The effectiveness of our method is further demonstrated through its application to ChatGPT, which shows its practical implications in real-world scenarios.
>
---
#### [new 052] HUR-MACL: High-Uncertainty Region-Guided Multi-Architecture Collaborative Learning for Head and Neck Multi-Organ Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于头颈多器官分割任务，旨在解决小而复杂器官分割精度低的问题。提出HUR-MACL模型，结合Vision Mamba和Deformable CNN，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2601.04607v1](https://arxiv.org/pdf/2601.04607v1)**

> **作者:** Xiaoyu Liu; Siwen Wei; Linhao Qu; Mingyuan Pan; Chengsheng Zhang; Yonghong Shi; Zhijian Song
>
> **摘要:** Accurate segmentation of organs at risk in the head and neck is essential for radiation therapy, yet deep learning models often fail on small, complexly shaped organs. While hybrid architectures that combine different models show promise, they typically just concatenate features without exploiting the unique strengths of each component. This results in functional overlap and limited segmentation accuracy. To address these issues, we propose a high uncertainty region-guided multi-architecture collaborative learning (HUR-MACL) model for multi-organ segmentation in the head and neck. This model adaptively identifies high uncertainty regions using a convolutional neural network, and for these regions, Vision Mamba as well as Deformable CNN are utilized to jointly improve their segmentation accuracy. Additionally, a heterogeneous feature distillation loss was proposed to promote collaborative learning between the two architectures in high uncertainty regions to further enhance performance. Our method achieves SOTA results on two public datasets and one private dataset.
>
---
#### [new 053] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction
- **分类: cs.CV**

- **简介: 论文提出OceanSplat，用于水下场景重建。解决水下光学退化导致的多视角不一致问题，通过三目视图一致性与深度先验优化3D高斯分布，提升重建精度与结构保持。**

- **链接: [https://arxiv.org/pdf/2601.04984v1](https://arxiv.org/pdf/2601.04984v1)**

> **作者:** Minseong Kweon; Jinsun Park
>
> **备注:** Accepted to AAAI 2026. Project page: https://oceansplat.github.io
>
> **摘要:** We introduce OceanSplat, a novel 3D Gaussian Splatting-based approach for accurately representing 3D geometry in underwater scenes. To overcome multi-view inconsistencies caused by underwater optical degradation, our method enforces trinocular view consistency by rendering horizontally and vertically translated camera views relative to each input view and aligning them via inverse warping. Furthermore, these translated camera views are used to derive a synthetic epipolar depth prior through triangulation, which serves as a self-supervised depth regularizer. These geometric constraints facilitate the spatial optimization of 3D Gaussians and preserve scene structure in underwater environments. We also propose a depth-aware alpha adjustment that modulates the opacity of 3D Gaussians during early training based on their $z$-component and viewing direction, deterring the formation of medium-induced primitives. With our contributions, 3D Gaussians are disentangled from the scattering medium, enabling robust representation of object geometry and significantly reducing floating artifacts in reconstructed underwater scenes. Experiments on real-world underwater and simulated scenes demonstrate that OceanSplat substantially outperforms existing methods for both scene reconstruction and restoration in scattering media.
>
---
#### [new 054] All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction
- **分类: cs.CV**

- **简介: 该论文属于有害模因检测任务，旨在解决模因类型和时间动态变化带来的检测难题。通过设计概念再现方法，构建设计概念图，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.04567v1](https://arxiv.org/pdf/2601.04567v1)**

> **作者:** Ziyou Jiang; Mingyang Li; Junjie Wang; Yuekai Huang; Jie Huang; Zhiyuan Chang; Zhaoyang Li; Qing Wang
>
> **备注:** 18 pages, 11 figures
>
> **摘要:** Harmful memes are ever-shifting in the Internet communities, which are difficult to analyze due to their type-shifting and temporal-evolving nature. Although these memes are shifting, we find that different memes may share invariant principles, i.e., the underlying design concept of malicious users, which can help us analyze why these memes are harmful. In this paper, we propose RepMD, an ever-shifting harmful meme detection method based on the design concept reproduction. We first refer to the attack tree to define the Design Concept Graph (DCG), which describes steps that people may take to design a harmful meme. Then, we derive the DCG from historical memes with design step reproduction and graph pruning. Finally, we use DCG to guide the Multimodal Large Language Model (MLLM) to detect harmful memes. The evaluation results show that RepMD achieves the highest accuracy with 81.1% and has slight accuracy decreases when generalized to type-shifting and temporal-evolving memes. Human evaluation shows that RepMD can improve the efficiency of human discovery on harmful memes, with 15$\sim$30 seconds per meme.
>
---
#### [new 055] MiLDEdit: Reasoning-Based Multi-Layer Design Document Editing
- **分类: cs.CV**

- **简介: 该论文提出MiLDEdit，解决多层设计文档编辑问题，通过推理框架实现精准修改，填补了该领域的研究空白。**

- **链接: [https://arxiv.org/pdf/2601.04589v1](https://arxiv.org/pdf/2601.04589v1)**

> **作者:** Zihao Lin; Wanrong Zhu; Jiuxiang Gu; Jihyung Kil; Christopher Tensmeyer; Lin Zhang; Shilong Liu; Ruiyi Zhang; Lifu Huang; Vlad I. Morariu; Tong Sun
>
> **摘要:** Real-world design documents (e.g., posters) are inherently multi-layered, combining decoration, text, and images. Editing them from natural-language instructions requires fine-grained, layer-aware reasoning to identify relevant layers and coordinate modifications. Prior work largely overlooks multi-layer design document editing, focusing instead on single-layer image editing or multi-layer generation, which assume a flat canvas and lack the reasoning needed to determine what and where to modify. To address this gap, we introduce the Multi-Layer Document Editing Agent (MiLDEAgent), a reasoning-based framework that combines an RL-trained multimodal reasoner for layer-wise understanding with an image editor for targeted modifications. To systematically benchmark this setting, we introduce the MiLDEBench, a human-in-the-loop corpus of over 20K design documents paired with diverse editing instructions. The benchmark is complemented by a task-specific evaluation protocol, MiLDEEval, which spans four dimensions including instruction following, layout consistency, aesthetics, and text rendering. Extensive experiments on 14 open-source and 2 closed-source models reveal that existing approaches fail to generalize: open-source models often cannot complete multi-layer document editing tasks, while closed-source models suffer from format violations. In contrast, MiLDEAgent achieves strong layer-aware reasoning and precise editing, significantly outperforming all open-source baselines and attaining performance comparable to closed-source models, thereby establishing the first strong baseline for multi-layer document editing.
>
---
#### [new 056] RoboVIP: Multi-View Video Generation with Visual Identity Prompting Augments Robot Manipulation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决真实操作数据收集困难的问题。通过引入视觉身份提示，生成多视角、时间一致的场景，提升策略模型性能。**

- **链接: [https://arxiv.org/pdf/2601.05241v1](https://arxiv.org/pdf/2601.05241v1)**

> **作者:** Boyang Wang; Haoran Zhang; Shujie Zhang; Jinkun Hao; Mingda Jia; Qi Lv; Yucheng Mao; Zhaoyang Lyu; Jia Zeng; Xudong Xu; Jiangmiao Pang
>
> **摘要:** The diversity, quantity, and quality of manipulation data are critical for training effective robot policies. However, due to hardware and physical setup constraints, collecting large-scale real-world manipulation data remains difficult to scale across diverse environments. Recent work uses text-prompt conditioned image diffusion models to augment manipulation data by altering the backgrounds and tabletop objects in the visual observations. However, these approaches often overlook the practical need for multi-view and temporally coherent observations required by state-of-the-art policy models. Further, text prompts alone cannot reliably specify the scene setup. To provide the diffusion model with explicit visual guidance, we introduce visual identity prompting, which supplies exemplar images as conditioning inputs to guide the generation of the desired scene setup. To this end, we also build a scalable pipeline to curate a visual identity pool from large robotics datasets. Using our augmented manipulation data to train downstream vision-language-action and visuomotor policy models yields consistent performance gains in both simulation and real-robot settings.
>
---
#### [new 057] VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control
- **分类: cs.CV**

- **简介: 该论文提出VerseCrafter，解决视频世界建模中相机与物体动态控制不精确的问题。通过4D几何控制表示，实现统一的动态生成，提升视频真实感与一致性。**

- **链接: [https://arxiv.org/pdf/2601.05138v1](https://arxiv.org/pdf/2601.05138v1)**

> **作者:** Sixiao Zheng; Minghao Yin; Wenbo Hu; Xiaoyu Li; Ying Shan; Yanwei Fu
>
> **备注:** Project Page: https://sixiaozheng.github.io/VerseCrafter_page/
>
> **摘要:** Video world models aim to simulate dynamic, real-world environments, yet existing methods struggle to provide unified and precise control over camera and multi-object motion, as videos inherently operate dynamics in the projected 2D image plane. To bridge this gap, we introduce VerseCrafter, a 4D-aware video world model that enables explicit and coherent control over both camera and object dynamics within a unified 4D geometric world state. Our approach is centered on a novel 4D Geometric Control representation, which encodes the world state through a static background point cloud and per-object 3D Gaussian trajectories. This representation captures not only an object's path but also its probabilistic 3D occupancy over time, offering a flexible, category-agnostic alternative to rigid bounding boxes or parametric models. These 4D controls are rendered into conditioning signals for a pretrained video diffusion model, enabling the generation of high-fidelity, view-consistent videos that precisely adhere to the specified dynamics. Unfortunately, another major challenge lies in the scarcity of large-scale training data with explicit 4D annotations. We address this by developing an automatic data engine that extracts the required 4D controls from in-the-wild videos, allowing us to train our model on a massive and diverse dataset.
>
---
#### [new 058] SOVABench: A Vehicle Surveillance Action Retrieval Benchmark for Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出SOVABench，用于视频监控中的行为检索任务，解决现有基准在动作区分上的不足。通过定义评估协议并利用多模态大语言模型生成可解释嵌入，提升动作识别性能。**

- **链接: [https://arxiv.org/pdf/2601.04824v1](https://arxiv.org/pdf/2601.04824v1)**

> **作者:** Oriol Rabasseda; Zenjie Li; Kamal Nasrollahi; Sergio Escalera
>
> **备注:** This work has been accepted at Real World Surveillance: Applications and Challenges, 6th (in WACV Workshops)
>
> **摘要:** Automatic identification of events and recurrent behavior analysis are critical for video surveillance. However, most existing content-based video retrieval benchmarks focus on scene-level similarity and do not evaluate the action discrimination required in surveillance. To address this gap, we introduce SOVABench (Surveillance Opposite Vehicle Actions Benchmark), a real-world retrieval benchmark built from surveillance footage and centered on vehicle-related actions. SOVABench defines two evaluation protocols (inter-pair and intra-pair) to assess cross-action discrimination and temporal direction understanding. Although action distinctions are generally intuitive for human observers, our experiments show that they remain challenging for state-of-the-art vision and multimodal models. Leveraging the visual reasoning and instruction-following capabilities of Multimodal Large Language Models (MLLMs), we present a training-free framework for producing interpretable embeddings from MLLM-generated descriptions for both images and videos. The framework achieves strong performance on SOVABench as well as on several spatial and counting benchmarks where contrastive Vision-Language Models often fail. The code, annotations, and instructions to construct the benchmark are publicly available.
>
---
#### [new 059] Mesh4D: 4D Mesh Reconstruction and Tracking from Monocular Video
- **分类: cs.CV**

- **简介: 论文提出Mesh4D，用于从单目视频中重建和跟踪4D网格。该任务旨在恢复动态物体的3D形状与运动。工作包括构建紧凑潜在空间和使用扩散模型进行快速动画预测。**

- **链接: [https://arxiv.org/pdf/2601.05251v1](https://arxiv.org/pdf/2601.05251v1)**

> **作者:** Zeren Jiang; Chuanxia Zheng; Iro Laina; Diane Larlus; Andrea Vedaldi
>
> **备注:** 15 pages, 8 figures, project page: https://mesh-4d.github.io/
>
> **摘要:** We propose Mesh4D, a feed-forward model for monocular 4D mesh reconstruction. Given a monocular video of a dynamic object, our model reconstructs the object's complete 3D shape and motion, represented as a deformation field. Our key contribution is a compact latent space that encodes the entire animation sequence in a single pass. This latent space is learned by an autoencoder that, during training, is guided by the skeletal structure of the training objects, providing strong priors on plausible deformations. Crucially, skeletal information is not required at inference time. The encoder employs spatio-temporal attention, yielding a more stable representation of the object's overall deformation. Building on this representation, we train a latent diffusion model that, conditioned on the input video and the mesh reconstructed from the first frame, predicts the full animation in one shot. We evaluate Mesh4D on reconstruction and novel view synthesis benchmarks, outperforming prior methods in recovering accurate 3D shape and deformation.
>
---
#### [new 060] On the Holistic Approach for Detecting Human Image Forgery
- **分类: cs.CV**

- **简介: 该论文属于图像伪造检测任务，旨在解决现有方法在面部与全身合成图像检测中泛化能力不足的问题。提出HuForDet框架，结合面部与上下文分析，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.04715v1](https://arxiv.org/pdf/2601.04715v1)**

> **作者:** Xiao Guo; Jie Zhu; Anil Jain; Xiaoming Liu
>
> **备注:** 6 figures, 5 tables
>
> **摘要:** The rapid advancement of AI-generated content (AIGC) has escalated the threat of deepfakes, from facial manipulations to the synthesis of entire photorealistic human bodies. However, existing detection methods remain fragmented, specializing either in facial-region forgeries or full-body synthetic images, and consequently fail to generalize across the full spectrum of human image manipulations. We introduce HuForDet, a holistic framework for human image forgery detection, which features a dual-branch architecture comprising: (1) a face forgery detection branch that employs heterogeneous experts operating in both RGB and frequency domains, including an adaptive Laplacian-of-Gaussian (LoG) module designed to capture artifacts ranging from fine-grained blending boundaries to coarse-scale texture irregularities; and (2) a contextualized forgery detection branch that leverages a Multi-Modal Large Language Model (MLLM) to analyze full-body semantic consistency, enhanced with a confidence estimation mechanism that dynamically weights its contribution during feature fusion. We curate a human image forgery (HuFor) dataset that unifies existing face forgery data with a new corpus of full-body synthetic humans. Extensive experiments show that our HuForDet achieves state-of-the-art forgery detection performance and superior robustness across diverse human image forgeries.
>
---
#### [new 061] Training a Custom CNN on Five Heterogeneous Image Datasets
- **分类: cs.CV; cs.NE**

- **简介: 论文研究在农业和城市领域五种不同图像数据集上训练定制的CNN模型，解决多领域图像分类问题。工作包括设计轻量级CNN、对比不同架构性能，并分析迁移学习效果。**

- **链接: [https://arxiv.org/pdf/2601.04727v1](https://arxiv.org/pdf/2601.04727v1)**

> **作者:** Anika Tabassum; Tasnuva Mahazabin Tuba; Nafisa Naznin
>
> **摘要:** Deep learning has transformed visual data analysis, with Convolutional Neural Networks (CNNs) becoming highly effective in learning meaningful feature representations directly from images. Unlike traditional manual feature engineering methods, CNNs automatically extract hierarchical visual patterns, enabling strong performance across diverse real-world contexts. This study investigates the effectiveness of CNN-based architectures across five heterogeneous datasets spanning agricultural and urban domains: mango variety classification, paddy variety identification, road surface condition assessment, auto-rickshaw detection, and footpath encroachment monitoring. These datasets introduce varying challenges, including differences in illumination, resolution, environmental complexity, and class imbalance, necessitating adaptable and robust learning models. We evaluate a lightweight, task-specific custom CNN alongside established deep architectures, including ResNet-18 and VGG-16, trained both from scratch and using transfer learning. Through systematic preprocessing, augmentation, and controlled experimentation, we analyze how architectural complexity, model depth, and pre-training influence convergence, generalization, and performance across datasets of differing scale and difficulty. The key contributions of this work are: (1) the development of an efficient custom CNN that achieves competitive performance across multiple application domains, and (2) a comprehensive comparative analysis highlighting when transfer learning and deep architectures provide substantial advantages, particularly in data-constrained environments. These findings offer practical insights for deploying deep learning models in resource-limited yet high-impact real-world visual classification tasks.
>
---
#### [new 062] DivAS: Interactive 3D Segmentation of NeRFs via Depth-Weighted Voxel Aggregation
- **分类: cs.CV**

- **简介: 该论文提出DivAS，用于NeRF的3D分割任务，解决传统方法依赖优化、速度慢的问题，通过深度加权体素聚合实现快速交互式分割。**

- **链接: [https://arxiv.org/pdf/2601.04860v1](https://arxiv.org/pdf/2601.04860v1)**

> **作者:** Ayush Pande
>
> **摘要:** Existing methods for segmenting Neural Radiance Fields (NeRFs) are often optimization-based, requiring slow per-scene training that sacrifices the zero-shot capabilities of 2D foundation models. We introduce DivAS (Depth-interactive Voxel Aggregation Segmentation), an optimization-free, fully interactive framework that addresses these limitations. Our method operates via a fast GUI-based workflow where 2D SAM masks, generated from user point prompts, are refined using NeRF-derived depth priors to improve geometric accuracy and foreground-background separation. The core of our contribution is a custom CUDA kernel that aggregates these refined multi-view masks into a unified 3D voxel grid in under 200ms, enabling real-time visual feedback. This optimization-free design eliminates the need for per-scene training. Experiments on Mip-NeRF 360° and LLFF show that DivAS achieves segmentation quality comparable to optimization-based methods, while being 2-2.5x faster end-to-end, and up to an order of magnitude faster when excluding user prompting time.
>
---
#### [new 063] CoV: Chain-of-View Prompting for Spatial Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D环境下的视觉问答任务，解决多视角信息获取与空间推理问题。提出CoV框架，通过主动视角选择与调整提升模型表现。**

- **链接: [https://arxiv.org/pdf/2601.05172v1](https://arxiv.org/pdf/2601.05172v1)**

> **作者:** Haoyu Zhao; Akide Liu; Zeyu Zhang; Weijie Wang; Feng Chen; Ruihan Zhu; Gholamreza Haffari; Bohan Zhuang
>
> **摘要:** Embodied question answering (EQA) in 3D environments often requires collecting context that is distributed across multiple viewpoints and partially occluded. However, most recent vision--language models (VLMs) are constrained to a fixed and finite set of input views, which limits their ability to acquire question-relevant context at inference time and hinders complex spatial reasoning. We propose Chain-of-View (CoV) prompting, a training-free, test-time reasoning framework that transforms a VLM into an active viewpoint reasoner through a coarse-to-fine exploration process. CoV first employs a View Selection agent to filter redundant frames and identify question-aligned anchor views. It then performs fine-grained view adjustment by interleaving iterative reasoning with discrete camera actions, obtaining new observations from the underlying 3D scene representation until sufficient context is gathered or a step budget is reached. We evaluate CoV on OpenEQA across four mainstream VLMs and obtain an average +11.56\% improvement in LLM-Match, with a maximum gain of +13.62\% on Qwen3-VL-Flash. CoV further exhibits test-time scaling: increasing the minimum action budget yields an additional +2.51\% average improvement, peaking at +3.73\% on Gemini-2.5-Flash. On ScanQA and SQA3D, CoV delivers strong performance (e.g., 116 CIDEr / 31.9 EM@1 on ScanQA and 51.1 EM@1 on SQA3D). Overall, these results suggest that question-aligned view selection coupled with open-view search is an effective, model-agnostic strategy for improving spatial reasoning in 3D EQA without additional training.
>
---
#### [new 064] Agri-R1: Empowering Generalizable Agricultural Reasoning in Vision-Language Models with Reinforcement Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于农业视觉-语言模型任务，解决疾病诊断中标签依赖和泛化能力差的问题。通过强化学习与推理数据生成，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.04672v1](https://arxiv.org/pdf/2601.04672v1)**

> **作者:** Wentao Zhang; Lifei Wang; Lina Lu; MingKun Xu; Shangyang Li; Yanchao Yang; Tao Fang
>
> **备注:** This paper is submitted for review to ACL 2026. It is 17 pages long and includes 5 figures. The corresponding authors are Tao Fang and Lina Lu
>
> **摘要:** Agricultural disease diagnosis challenges VLMs, as conventional fine-tuning requires extensive labels, lacks interpretability, and generalizes poorly. While reasoning improves model robustness, existing methods rely on costly expert annotations and rarely address the open-ended, diverse nature of agricultural queries. To address these limitations, we propose \textbf{Agri-R1}, a reasoning-enhanced large model for agriculture. Our framework automates high-quality reasoning data generation via vision-language synthesis and LLM-based filtering, using only 19\% of available samples. Training employs Group Relative Policy Optimization (GRPO) with a novel proposed reward function that integrates domain-specific lexicons and fuzzy matching to assess both correctness and linguistic flexibility in open-ended responses. Evaluated on CDDMBench, our resulting 3B-parameter model achieves performance competitive with 7B- to 13B-parameter baselines, showing a +23.2\% relative gain in disease recognition accuracy, +33.3\% in agricultural knowledge QA, and a +26.10-point improvement in cross-domain generalization over standard fine-tuning. Ablation studies confirm that the synergy between structured reasoning data and GRPO-driven exploration underpins these gains, with benefits scaling as question complexity increases.
>
---
#### [new 065] SRU-Pix2Pix: A Fusion-Driven Generator Network for Medical Image Translation with Few-Shot Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像翻译任务，旨在解决MRI采集时间长、成本高问题。通过融合SEResNet和U-Net++提升生成质量与结构保真度。**

- **链接: [https://arxiv.org/pdf/2601.04785v1](https://arxiv.org/pdf/2601.04785v1)**

> **作者:** Xihe Qiu; Yang Dai; Xiaoyu Tan; Sijia Li; Fenghao Sun; Lu Gan; Liang Liu
>
> **摘要:** Magnetic Resonance Imaging (MRI) provides detailed tissue information, but its clinical application is limited by long acquisition time, high cost, and restricted resolution. Image translation has recently gained attention as a strategy to address these limitations. Although Pix2Pix has been widely applied in medical image translation, its potential has not been fully explored. In this study, we propose an enhanced Pix2Pix framework that integrates Squeeze-and-Excitation Residual Networks (SEResNet) and U-Net++ to improve image generation quality and structural fidelity. SEResNet strengthens critical feature representation through channel attention, while U-Net++ enhances multi-scale feature fusion. A simplified PatchGAN discriminator further stabilizes training and refines local anatomical realism. Experimental results demonstrate that under few-shot conditions with fewer than 500 images, the proposed method achieves consistent structural fidelity and superior image quality across multiple intra-modality MRI translation tasks, showing strong generalization ability. These results suggest an effective extension of Pix2Pix for medical image translation.
>
---
#### [new 066] VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，解决CoT推理与直接回答的效率与效果问题。提出VideoAuto-R1框架，通过“思考一次，回答两次”策略提升效率和准确性。**

- **链接: [https://arxiv.org/pdf/2601.05175v1](https://arxiv.org/pdf/2601.05175v1)**

> **作者:** Shuming Liu; Mingchen Zhuge; Changsheng Zhao; Jun Chen; Lemeng Wu; Zechun Liu; Chenchen Zhu; Zhipeng Cai; Chong Zhou; Haozhe Liu; Ernie Chang; Saksham Suri; Hongyu Xu; Qi Qian; Wei Wen; Balakrishnan Varadarajan; Zhuang Liu; Hu Xu; Florian Bordes; Raghuraman Krishnamoorthi; Bernard Ghanem; Vikas Chandra; Yunyang Xiong
>
> **备注:** Project page: https://ivul-kaust.github.io/projects/videoauto-r1/
>
> **摘要:** Chain-of-thought (CoT) reasoning has emerged as a powerful tool for multimodal large language models on video understanding tasks. However, its necessity and advantages over direct answering remain underexplored. In this paper, we first demonstrate that for RL-trained video models, direct answering often matches or even surpasses CoT performance, despite CoT producing step-by-step analyses at a higher computational cost. Motivated by this, we propose VideoAuto-R1, a video understanding framework that adopts a reason-when-necessary strategy. During training, our approach follows a Thinking Once, Answering Twice paradigm: the model first generates an initial answer, then performs reasoning, and finally outputs a reviewed answer. Both answers are supervised via verifiable rewards. During inference, the model uses the confidence score of the initial answer to determine whether to proceed with reasoning. Across video QA and grounding benchmarks, VideoAuto-R1 achieves state-of-the-art accuracy with significantly improved efficiency, reducing the average response length by ~3.3x, e.g., from 149 to just 44 tokens. Moreover, we observe a low rate of thinking-mode activation on perception-oriented tasks, but a higher rate on reasoning-intensive tasks. This suggests that explicit language-based reasoning is generally beneficial but not always necessary.
>
---
#### [new 067] TokenSeg: Efficient 3D Medical Image Segmentation via Hierarchical Visual Token Compression
- **分类: cs.CV**

- **简介: 该论文提出TokenSeg，用于高效3D医学图像分割。针对计算量大和冗余问题，设计了层次编码器、边界感知分词器和稀疏到稠密解码器，提升分割效率与精度。**

- **链接: [https://arxiv.org/pdf/2601.04519v1](https://arxiv.org/pdf/2601.04519v1)**

> **作者:** Sen Zeng; Hong Zhou; Zheng Zhu; Yang Liu
>
> **摘要:** Three-dimensional medical image segmentation is a fundamental yet computationally demanding task due to the cubic growth of voxel processing and the redundant computation on homogeneous regions. To address these limitations, we propose \textbf{TokenSeg}, a boundary-aware sparse token representation framework for efficient 3D medical volume segmentation. Specifically, (1) we design a \emph{multi-scale hierarchical encoder} that extracts 400 candidate tokens across four resolution levels to capture both global anatomical context and fine boundary details; (2) we introduce a \emph{boundary-aware tokenizer} that combines VQ-VAE quantization with importance scoring to select 100 salient tokens, over 60\% of which lie near tumor boundaries; and (3) we develop a \emph{sparse-to-dense decoder} that reconstructs full-resolution masks through token reprojection, progressive upsampling, and skip connections. Extensive experiments on a 3D breast DCE-MRI dataset comprising 960 cases demonstrate that TokenSeg achieves state-of-the-art performance with 94.49\% Dice and 89.61\% IoU, while reducing GPU memory and inference latency by 64\% and 68\%, respectively. To verify the generalization capability, our evaluations on MSD cardiac and brain MRI benchmark datasets demonstrate that TokenSeg consistently delivers optimal performance across heterogeneous anatomical structures. These results highlight the effectiveness of anatomically informed sparse representation for accurate and efficient 3D medical image segmentation.
>
---
#### [new 068] Driving on Registers
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决端到端驾驶的高效性与准确性问题。提出DrivoR架构，利用视觉Transformer和相机感知的注册令牌压缩特征，提升计算效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2601.05083v1](https://arxiv.org/pdf/2601.05083v1)**

> **作者:** Ellington Kirby; Alexandre Boulch; Yihong Xu; Yuan Yin; Gilles Puy; Éloi Zablocki; Andrei Bursuc; Spyros Gidaris; Renaud Marlet; Florent Bartoccioni; Anh-Quan Cao; Nermin Samet; Tuan-Hung VU; Matthieu Cord
>
> **摘要:** We present DrivoR, a simple and efficient transformer-based architecture for end-to-end autonomous driving. Our approach builds on pretrained Vision Transformers (ViTs) and introduces camera-aware register tokens that compress multi-camera features into a compact scene representation, significantly reducing downstream computation without sacrificing accuracy. These tokens drive two lightweight transformer decoders that generate and then score candidate trajectories. The scoring decoder learns to mimic an oracle and predicts interpretable sub-scores representing aspects such as safety, comfort, and efficiency, enabling behavior-conditioned driving at inference. Despite its minimal design, DrivoR outperforms or matches strong contemporary baselines across NAVSIM-v1, NAVSIM-v2, and the photorealistic closed-loop HUGSIM benchmark. Our results show that a pure-transformer architecture, combined with targeted token compression, is sufficient for accurate, efficient, and adaptive end-to-end driving. Code and checkpoints will be made available via the project page.
>
---
#### [new 069] AIVD: Adaptive Edge-Cloud Collaboration for Accurate and Efficient Industrial Visual Detection
- **分类: cs.CV**

- **简介: 该论文提出AIVD框架，解决工业视觉检测中精准定位与资源受限问题，通过边缘-云协同实现高效准确的检测与语义生成。**

- **链接: [https://arxiv.org/pdf/2601.04734v1](https://arxiv.org/pdf/2601.04734v1)**

> **作者:** Yunqing Hu; Zheming Yang; Chang Zhao; Qi Guo; Meng Gao; Pengcheng Li; Wen Ji
>
> **摘要:** Multimodal large language models (MLLMs) demonstrate exceptional capabilities in semantic understanding and visual reasoning, yet they still face challenges in precise object localization and resource-constrained edge-cloud deployment. To address this, this paper proposes the AIVD framework, which achieves unified precise localization and high-quality semantic generation through the collaboration between lightweight edge detectors and cloud-based MLLMs. To enhance the cloud MLLM's robustness against edge cropped-box noise and scenario variations, we design an efficient fine-tuning strategy with visual-semantic collaborative augmentation, significantly improving classification accuracy and semantic consistency. Furthermore, to maintain high throughput and low latency across heterogeneous edge devices and dynamic network conditions, we propose a heterogeneous resource-aware dynamic scheduling algorithm. Experimental results demonstrate that AIVD substantially reduces resource consumption while improving MLLM classification performance and semantic generation quality. The proposed scheduling strategy also achieves higher throughput and lower latency across diverse scenarios.
>
---
#### [new 070] WebCryptoAgent: Agentic Crypto Trading with Web Informatics
- **分类: cs.CV**

- **简介: 该论文属于加密货币交易任务，旨在解决多源信息融合与风险控制问题。提出WebCryptoAgent框架，通过分模块代理和分离控制架构提升交易稳定性与风险应对能力。**

- **链接: [https://arxiv.org/pdf/2601.04687v1](https://arxiv.org/pdf/2601.04687v1)**

> **作者:** Ali Kurban; Wei Luo; Liangyu Zuo; Zeyu Zhang; Renda Han; Zhaolu Kang; Hao Tang
>
> **摘要:** Cryptocurrency trading increasingly depends on timely integration of heterogeneous web information and market microstructure signals to support short-horizon decision making under extreme volatility. However, existing trading systems struggle to jointly reason over noisy multi-source web evidence while maintaining robustness to rapid price shocks at sub-second timescales. The first challenge lies in synthesizing unstructured web content, social sentiment, and structured OHLCV signals into coherent and interpretable trading decisions without amplifying spurious correlations, while the second challenge concerns risk control, as slow deliberative reasoning pipelines are ill-suited for handling abrupt market shocks that require immediate defensive responses. To address these challenges, we propose WebCryptoAgent, an agentic trading framework that decomposes web-informed decision making into modality-specific agents and consolidates their outputs into a unified evidence document for confidence-calibrated reasoning. We further introduce a decoupled control architecture that separates strategic hourly reasoning from a real-time second-level risk model, enabling fast shock detection and protective intervention independent of the trading loop. Extensive experiments on real-world cryptocurrency markets demonstrate that WebCryptoAgent improves trading stability, reduces spurious activity, and enhances tail-risk handling compared to existing baselines. Code will be available at https://github.com/AIGeeksGroup/WebCryptoAgent.
>
---
#### [new 071] From Understanding to Engagement: Personalized pharmacy Video Clips via Vision Language Models (VLMs)
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视频摘要任务，旨在解决药品行业视频内容处理效率低、成本高的问题。通过融合VLM和ALM，生成个性化视频剪辑，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2601.05059v1](https://arxiv.org/pdf/2601.05059v1)**

> **作者:** Suyash Mishra; Qiang Li; Srikanth Patil; Anubhav Girdhar
>
> **备注:** Contributed original research to top tier conference in VLM; currently undergoing peer review
>
> **摘要:** Vision Language Models (VLMs) are poised to revolutionize the digital transformation of pharmacyceutical industry by enabling intelligent, scalable, and automated multi-modality content processing. Traditional manual annotation of heterogeneous data modalities (text, images, video, audio, and web links), is prone to inconsistencies, quality degradation, and inefficiencies in content utilization. The sheer volume of long video and audio data further exacerbates these challenges, (e.g. long clinical trial interviews and educational seminars). Here, we introduce a domain adapted Video to Video Clip Generation framework that integrates Audio Language Models (ALMs) and Vision Language Models (VLMs) to produce highlight clips. Our contributions are threefold: (i) a reproducible Cut & Merge algorithm with fade in/out and timestamp normalization, ensuring smooth transitions and audio/visual alignment; (ii) a personalization mechanism based on role definition and prompt injection for tailored outputs (marketing, training, regulatory); (iii) a cost efficient e2e pipeline strategy balancing ALM/VLM enhanced processing. Evaluations on Video MME benchmark (900) and our proprietary dataset of 16,159 pharmacy videos across 14 disease areas demonstrate 3 to 4 times speedup, 4 times cost reduction, and competitive clip quality. Beyond efficiency gains, we also report our methods improved clip coherence scores (0.348) and informativeness scores (0.721) over state of the art VLM baselines (e.g., Gemini 2.5 Pro), highlighting the potential of transparent, custom extractive, and compliance supporting video summarization for life sciences.
>
---
#### [new 072] Combining facial videos and biosignals for stress estimation during driving
- **分类: cs.CV**

- **简介: 该论文属于应力识别任务，旨在解决驾驶中应力检测难题。通过结合面部视频与生物信号，提出基于Transformer的时序建模框架，提升应力识别效果。**

- **链接: [https://arxiv.org/pdf/2601.04376v1](https://arxiv.org/pdf/2601.04376v1)**

> **作者:** Paraskevi Valergaki; Vassilis C. Nicodemou; Iason Oikonomidis; Antonis Argyros; Anastasios Roussos
>
> **备注:** UNDER SUBMISSION TO ICPR 2026
>
> **摘要:** Reliable stress recognition from facial videos is challenging due to stress's subjective nature and voluntary facial control. While most methods rely on Facial Action Units, the role of disentangled 3D facial geometry remains underexplored. We address this by analyzing stress during distracted driving using EMOCA-derived 3D expression and pose coefficients. Paired hypothesis tests between baseline and stressor phases reveal that 41 of 56 coefficients show consistent, phase-specific stress responses comparable to physiological markers. Building on this, we propose a Transformer-based temporal modeling framework and assess unimodal, early-fusion, and cross-modal attention strategies. Cross-Modal Attention fusion of EMOCA and physiological signals achieves best performance (AUROC 92\%, Accuracy 86.7\%), with EMOCA-gaze fusion also competitive (AUROC 91.8\%). This highlights the effectiveness of temporal modeling and cross-modal attention for stress recognition.
>
---
#### [new 073] GREx: Generalized Referring Expression Segmentation, Comprehension, and Generation
- **分类: cs.CV**

- **简介: 该论文提出GREx任务，解决传统Referring Expression Segmentation/Comprehension/Generation（REx）仅支持单目标表达的问题，扩展至多目标和无目标场景，并构建了gRefCOCO数据集及ReLA方法。**

- **链接: [https://arxiv.org/pdf/2601.05244v1](https://arxiv.org/pdf/2601.05244v1)**

> **作者:** Henghui Ding; Chang Liu; Shuting He; Xudong Jiang; Yu-Gang Jiang
>
> **备注:** IJCV, Project Page: https://henghuiding.com/GREx/
>
> **摘要:** Referring Expression Segmentation (RES) and Comprehension (REC) respectively segment and detect the object described by an expression, while Referring Expression Generation (REG) generates an expression for the selected object. Existing datasets and methods commonly support single-target expressions only, i.e., one expression refers to one object, not considering multi-target and no-target expressions. This greatly limits the real applications of REx (RES/REC/REG). This paper introduces three new benchmarks called Generalized Referring Expression Segmentation (GRES), Comprehension (GREC), and Generation (GREG), collectively denoted as GREx, which extend the classic REx to allow expressions to identify an arbitrary number of objects. We construct the first large-scale GREx dataset gRefCOCO that contains multi-target, no-target, and single-target expressions and their corresponding images with labeled targets. GREx and gRefCOCO are designed to be backward-compatible with REx, facilitating extensive experiments to study the performance gap of the existing REx methods on GREx tasks. One of the challenges of GRES/GREC is complex relationship modeling, for which we propose a baseline ReLA that adaptively divides the image into regions with sub-instance clues and explicitly models the region-region and region-language dependencies. The proposed ReLA achieves the state-of-the-art results on the both GRES and GREC tasks. The proposed gRefCOCO dataset and method are available at https://henghuiding.github.io/GREx.
>
---
#### [new 074] CRUNet-MR-Univ: A Foundation Model for Diverse Cardiac MRI Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于心脏MRI重建任务，旨在解决模型泛化能力不足的问题。通过构建CRUNet-MR-Univ模型，提升对多样CMR场景的适应能力。**

- **链接: [https://arxiv.org/pdf/2601.04428v1](https://arxiv.org/pdf/2601.04428v1)**

> **作者:** Donghang Lyu; Marius Staring; Hildo Lamb; Mariya Doneva
>
> **备注:** STACOM 2025
>
> **摘要:** In recent years, deep learning has attracted increasing attention in the field of Cardiac MRI (CMR) reconstruction due to its superior performance over traditional methods, particularly in handling higher acceleration factors, highlighting its potential for real-world clinical applications. However, current deep learning methods remain limited in generalizability. CMR scans exhibit wide variability in image contrast, sampling patterns, scanner vendors, anatomical structures, and disease types. Most existing models are designed to handle only a single or narrow subset of these variations, leading to performance degradation when faced with distribution shifts. Therefore, it is beneficial to develop a unified model capable of generalizing across diverse CMR scenarios. To this end, we propose CRUNet-MR-Univ, a foundation model that leverages spatio-temporal correlations and prompt-based priors to effectively handle the full diversity of CMR scans. Our approach consistently outperforms baseline methods across a wide range of settings, highlighting its effectiveness and promise.
>
---
#### [new 075] Mechanisms of Prompt-Induced Hallucination in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型研究，旨在解决提示诱导幻觉问题。通过分析模型注意力机制，发现特定注意力头导致幻觉，并通过消融实验有效减少幻觉。**

- **链接: [https://arxiv.org/pdf/2601.05201v1](https://arxiv.org/pdf/2601.05201v1)**

> **作者:** William Rudman; Michal Golovanevsky; Dana Arad; Yonatan Belinkov; Ritambhara Singh; Carsten Eickhoff; Kyle Mahowald
>
> **摘要:** Large vision-language models (VLMs) are highly capable, yet often hallucinate by favoring textual prompts over visual evidence. We study this failure mode in a controlled object-counting setting, where the prompt overstates the number of objects in the image (e.g., asking a model to describe four waterlilies when only three are present). At low object counts, models often correct the overestimation, but as the number of objects increases, they increasingly conform to the prompt regardless of the discrepancy. Through mechanistic analysis of three VLMs, we identify a small set of attention heads whose ablation substantially reduces prompt-induced hallucinations (PIH) by at least 40% without additional training. Across models, PIH-heads mediate prompt copying in model-specific ways. We characterize these differences and show that PIH ablation increases correction toward visual evidence. Our findings offer insights into the internal mechanisms driving prompt-induced hallucinations, revealing model-specific differences in how these behaviors are implemented.
>
---
#### [new 076] Embedding Textual Information in Images Using Quinary Pixel Combinations
- **分类: cs.CV**

- **简介: 该论文属于信息隐藏任务，旨在解决文本嵌入图像中的效率与隐蔽性问题。提出一种基于RGB五元像素组合的文本嵌入方法，提升嵌入效率并减少图像失真。**

- **链接: [https://arxiv.org/pdf/2601.04302v1](https://arxiv.org/pdf/2601.04302v1)**

> **作者:** A V Uday Kiran Kandala
>
> **摘要:** This paper presents a novel technique for embedding textual data into images using quinary combinations of pixel intensities in RGB space. Existing methods predominantly rely on least and most significant bit (LSB & MSB) manipulation, Pixel Value Differencing (PVD), spatial perturbations in RGB channels, transform domain based methods, Quantization methods, Edge and Region based methods and more recently through deep learning methods and generative AI techniques for hiding textual information in spatial domain of images. Most of them are dependent on pixel intensity flipping over multiple pixels, such as LSB and combination of LSB based methodologies, and on transform coefficients, often resulting in the form of noise. Encoding and Decoding are deterministic in most of the existing approaches and are computationally heavy in case of higher models such as deep learning and gen AI approaches. The proposed method works on quinary pixel intensity combinations in RGB space, where five controlled different pixel intensity variations in each of the R, G, and B channels formulate up to one hundred and twenty five distinct pixel intensity combinations. These combinations are mapped to textual symbols, enabling the representation of uppercase and lowercase alphabetic characters, numeric digits, whitespace, and commonly used special characters. Different metrics such as MSE, MAE, SNR, PSNR, SSIM, Histogram Comparison and Heatmap analysis, were evaluated for both original and encoded images resulting in no significant distortion in the images. Furthermore, the method achieves improved embedding efficiency by encoding a complete textual symbol within a single RGB pixel, in contrast to LSB and MSB based approaches that typically require multiple pixels or multi-step processes, as well as transform and learning based methods that incur higher computational overhead.
>
---
#### [new 077] Patch-based Representation and Learning for Efficient Deformation Modeling
- **分类: cs.CV**

- **简介: 该论文提出一种基于补丁的表面表示方法PolyFit，用于高效变形建模。解决传统方法计算量大的问题，通过学习局部jet系数实现快速表面变形，应用于形状恢复和服装拟合任务。**

- **链接: [https://arxiv.org/pdf/2601.05035v1](https://arxiv.org/pdf/2601.05035v1)**

> **作者:** Ruochen Chen; Thuy Tran; Shaifali Parashar
>
> **摘要:** In this paper, we present a patch-based representation of surfaces, PolyFit, which is obtained by fitting jet functions locally on surface patches. Such a representation can be learned efficiently in a supervised fashion from both analytic functions and real data. Once learned, it can be generalized to various types of surfaces. Using PolyFit, the surfaces can be efficiently deformed by updating a compact set of jet coefficients rather than optimizing per-vertex degrees of freedom for many downstream tasks in computer vision and graphics. We demonstrate the capabilities of our proposed methodologies with two applications: 1) Shape-from-template (SfT): where the goal is to deform the input 3D template of an object as seen in image/video. Using PolyFit, we adopt test-time optimization that delivers competitive accuracy while being markedly faster than offline physics-based solvers, and outperforms recent physics-guided neural simulators in accuracy at modest additional runtime. 2) Garment draping. We train a self-supervised, mesh- and garment-agnostic model that generalizes across resolutions and garment types, delivering up to an order-of-magnitude faster inference than strong baselines.
>
---
#### [new 078] RL-AWB: Deep Reinforcement Learning for Auto White Balance Correction in Low-Light Night-time Scenes
- **分类: cs.CV**

- **简介: 该论文属于图像处理任务，旨在解决低光夜间场景下的自动白平衡问题。通过结合统计方法与深度强化学习，提出RL-AWB框架，提升白平衡的准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.05249v1](https://arxiv.org/pdf/2601.05249v1)**

> **作者:** Yuan-Kang Lee; Kuan-Lin Chen; Chia-Che Chang; Yu-Lun Liu
>
> **备注:** Project page: https://ntuneillee.github.io/research/rl-awb/
>
> **摘要:** Nighttime color constancy remains a challenging problem in computational photography due to low-light noise and complex illumination conditions. We present RL-AWB, a novel framework combining statistical methods with deep reinforcement learning for nighttime white balance. Our method begins with a statistical algorithm tailored for nighttime scenes, integrating salient gray pixel detection with novel illumination estimation. Building on this foundation, we develop the first deep reinforcement learning approach for color constancy that leverages the statistical algorithm as its core, mimicking professional AWB tuning experts by dynamically optimizing parameters for each image. To facilitate cross-sensor evaluation, we introduce the first multi-sensor nighttime dataset. Experiment results demonstrate that our method achieves superior generalization capability across low-light and well-illuminated images. Project page: https://ntuneillee.github.io/research/rl-awb/
>
---
#### [new 079] Measurement-Consistent Langevin Corrector: A Remedy for Latent Diffusion Inverse Solvers
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像恢复任务，旨在解决潜在扩散模型在逆问题求解中的不稳定问题。提出MCLC模块，通过测量一致的Langevin更新提升稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2601.04791v1](https://arxiv.org/pdf/2601.04791v1)**

> **作者:** Lee Hyoseok; Sohwi Lim; Eunju Cha; Tae-Hyun Oh
>
> **备注:** Under Review
>
> **摘要:** With recent advances in generative models, diffusion models have emerged as powerful priors for solving inverse problems in each domain. Since Latent Diffusion Models (LDMs) provide generic priors, several studies have explored their potential as domain-agnostic zero-shot inverse solvers. Despite these efforts, existing latent diffusion inverse solvers suffer from their instability, exhibiting undesirable artifacts and degraded quality. In this work, we first identify the instability as a discrepancy between the solver's and true reverse diffusion dynamics, and show that reducing this gap stabilizes the solver. Building on this, we introduce Measurement-Consistent Langevin Corrector (MCLC), a theoretically grounded plug-and-play correction module that remedies the LDM-based inverse solvers through measurement-consistent Langevin updates. Compared to prior approaches that rely on linear manifold assumptions, which often do not hold in latent space, MCLC operates without this assumption, leading to more stable and reliable behavior. We experimentally demonstrate the effectiveness of MCLC and its compatibility with existing solvers across diverse image restoration tasks. Additionally, we analyze blob artifacts and offer insights into their underlying causes. We highlight that MCLC is a key step toward more robust zero-shot inverse problem solvers.
>
---
#### [new 080] See, Explain, and Intervene: A Few-Shot Multimodal Agent Framework for Hateful Meme Moderation
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于仇恨模因检测任务，旨在解决有限数据下如何检测、解释和干预仇恨模因的问题。通过构建多模态代理框架，结合生成模型实现少样本适应。**

- **链接: [https://arxiv.org/pdf/2601.04692v1](https://arxiv.org/pdf/2601.04692v1)**

> **作者:** Naquee Rizwan; Subhankar Swain; Paramananda Bhaskar; Gagan Aryan; Shehryaar Shah Khan; Animesh Mukherjee
>
> **摘要:** In this work, we examine hateful memes from three complementary angles - how to detect them, how to explain their content and how to intervene them prior to being posted - by applying a range of strategies built on top of generative AI models. To the best of our knowledge, explanation and intervention have typically been studied separately from detection, which does not reflect real-world conditions. Further, since curating large annotated datasets for meme moderation is prohibitively expensive, we propose a novel framework that leverages task-specific generative multimodal agents and the few-shot adaptability of large multimodal models to cater to different types of memes. We believe this is the first work focused on generalizable hateful meme moderation under limited data conditions, and has strong potential for deployment in real-world production scenarios. Warning: Contains potentially toxic contents.
>
---
#### [new 081] FronTalk: Benchmarking Front-End Development as Conversational Code Generation with Multi-Modal Feedback
- **分类: cs.CL; cs.CV; cs.LG; cs.SE**

- **简介: 该论文提出FronTalk，用于前端代码生成的基准任务，解决多轮对话中视觉反馈理解与记忆遗忘问题，通过构建数据集和评估框架提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.04203v1](https://arxiv.org/pdf/2601.04203v1)**

> **作者:** Xueqing Wu; Zihan Xue; Da Yin; Shuyan Zhou; Kai-Wei Chang; Nanyun Peng; Yeming Wen
>
> **摘要:** We present FronTalk, a benchmark for front-end code generation that pioneers the study of a unique interaction dynamic: conversational code generation with multi-modal feedback. In front-end development, visual artifacts such as sketches, mockups and annotated creenshots are essential for conveying design intent, yet their role in multi-turn code generation remains largely unexplored. To address this gap, we focus on the front-end development task and curate FronTalk, a collection of 100 multi-turn dialogues derived from real-world websites across diverse domains such as news, finance, and art. Each turn features both a textual instruction and an equivalent visual instruction, each representing the same user intent. To comprehensively evaluate model performance, we propose a novel agent-based evaluation framework leveraging a web agent to simulate users and explore the website, and thus measuring both functional correctness and user experience. Evaluation of 20 models reveals two key challenges that are under-explored systematically in the literature: (1) a significant forgetting issue where models overwrite previously implemented features, resulting in task failures, and (2) a persistent challenge in interpreting visual feedback, especially for open-source vision-language models (VLMs). We propose a strong baseline to tackle the forgetting issue with AceCoder, a method that critiques the implementation of every past instruction using an autonomous web agent. This approach significantly reduces forgetting to nearly zero and improves the performance by up to 9.3% (56.0% to 65.3%). Overall, we aim to provide a solid foundation for future research in front-end development and the general interaction dynamics of multi-turn, multi-modal code generation. Code and data are released at https://github.com/shirley-wu/frontalk
>
---
#### [new 082] V-FAT: Benchmarking Visual Fidelity Against Text-bias
- **分类: cs.CL; cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于多模态语言模型任务，旨在解决模型依赖文本而非视觉的问题。通过构建V-FAT基准和VRS指标，评估模型在视觉与文本冲突下的表现。**

- **链接: [https://arxiv.org/pdf/2601.04897v1](https://arxiv.org/pdf/2601.04897v1)**

> **作者:** Ziteng Wang; Yujie He; Guanliang Li; Siqi Yang; Jiaqi Xiong; Songxiang Liu
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated impressive performance on standard visual reasoning benchmarks. However, there is growing concern that these models rely excessively on linguistic shortcuts rather than genuine visual grounding, a phenomenon we term Text Bias. In this paper, we investigate the fundamental tension between visual perception and linguistic priors. We decouple the sources of this bias into two dimensions: Internal Corpus Bias, stemming from statistical correlations in pretraining, and External Instruction Bias, arising from the alignment-induced tendency toward sycophancy. To quantify this effect, we introduce V-FAT (Visual Fidelity Against Text-bias), a diagnostic benchmark comprising 4,026 VQA instances across six semantic domains. V-FAT employs a Three-Level Evaluation Framework that systematically increases the conflict between visual evidence and textual information: (L1) internal bias from atypical images, (L2) external bias from misleading instructions, and (L3) synergistic bias where both coincide. We introduce the Visual Robustness Score (VRS), a metric designed to penalize "lucky" linguistic guesses and reward true visual fidelity. Our evaluation of 12 frontier MLLMs reveals that while models excel in existing benchmarks, they experience significant visual collapse under high linguistic dominance.
>
---
#### [new 083] In-SRAM Radiant Foam Rendering on a Graph Processor
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于图形渲染任务，解决在分布式内存加速器上高效渲染体积场景的问题。工作包括设计一种基于SRAM的分布式渲染系统，实现高效数据与计算分布。**

- **链接: [https://arxiv.org/pdf/2601.04382v1](https://arxiv.org/pdf/2601.04382v1)**

> **作者:** Zulkhuu Tuya; Ignacio Alzugaray; Nicholas Fry; Andrew J. Davison
>
> **备注:** 24 pages, 26 figures
>
> **摘要:** Many emerging many-core accelerators replace a single large device memory with hundreds to thousands of lightweight cores, each owning only a small local SRAM and exchanging data via explicit on-chip communication. This organization offers high aggregate bandwidth, but it breaks a key assumption behind many volumetric rendering techniques: that rays can randomly access a large, unified scene representation. Rendering efficiently on such hardware therefore requires distributing both data and computation, keeping ray traversal mostly local, and structuring communication into predictable routes. We present a fully in-SRAM, distributed renderer for the \emph{Radiant Foam} Voronoi-cell volumetric representation on the Graphcore Mk2 IPU, a many-core accelerator with tile-local SRAM and explicit inter-tile communication. Our system shards the scene across tiles and forwards rays between shards through a hierarchical routing overlay, enabling ray marching entirely from on-chip SRAM with predictable communication. On Mip-NeRF~360 scenes, the system attains near-interactive throughput (\(\approx\)1\,fps at \mbox{$640\times480$}) with image and depth quality close to the original GPU-based Radiant Foam implementation, while keeping all scene data and ray state in on-chip SRAM. Beyond demonstrating feasibility, we analyze routing, memory, and scheduling bottlenecks that inform how future distributed-memory accelerators can better support irregular, data-movement-heavy rendering workloads.
>
---
#### [new 084] Decentralized Privacy-Preserving Federal Learning of Computer Vision Models on Edge Devices
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于联邦学习任务，旨在解决隐私泄露问题。通过分析加密、梯度噪声等方法，提升模型训练中的数据隐私性，并在边缘设备上进行了实验验证。**

- **链接: [https://arxiv.org/pdf/2601.04912v1](https://arxiv.org/pdf/2601.04912v1)**

> **作者:** Damian Harenčák; Lukáš Gajdošech; Martin Madaras
>
> **备注:** Accepted to VISAPP 2026 as Position Paper
>
> **摘要:** Collaborative training of a machine learning model comes with a risk of sharing sensitive or private data. Federated learning offers a way of collectively training a single global model without the need to share client data, by sharing only the updated parameters from each client's local model. A central server is then used to aggregate parameters from all clients and redistribute the aggregated model back to the clients. Recent findings have shown that even in this scenario, private data can be reconstructed only using information about model parameters. Current efforts to mitigate this are mainly focused on reducing privacy risks on the server side, assuming that other clients will not act maliciously. In this work, we analyzed various methods for improving the privacy of client data concerning both the server and other clients for neural networks. Some of these methods include homomorphic encryption, gradient compression, gradient noising, and discussion on possible usage of modified federated learning systems such as split learning, swarm learning or fully encrypted models. We have analyzed the negative effects of gradient compression and gradient noising on the accuracy of convolutional neural networks used for classification. We have shown the difficulty of data reconstruction in the case of segmentation networks. We have also implemented a proof of concept on the NVIDIA Jetson TX2 module used in edge devices and simulated a federated learning process.
>
---
#### [new 085] Learning Latent Action World Models In The Wild
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于强化学习任务，旨在解决在真实视频中学习潜在动作世界模型的问题。通过分析视频内容，构建能预测动作后果的模型，提升现实环境中的智能体规划能力。**

- **链接: [https://arxiv.org/pdf/2601.05230v1](https://arxiv.org/pdf/2601.05230v1)**

> **作者:** Quentin Garrido; Tushar Nagarajan; Basile Terver; Nicolas Ballas; Yann LeCun; Michael Rabbat
>
> **备注:** 37 pages, 25 figures
>
> **摘要:** Agents capable of reasoning and planning in the real world require the ability of predicting the consequences of their actions. While world models possess this capability, they most often require action labels, that can be complex to obtain at scale. This motivates the learning of latent action models, that can learn an action space from videos alone. Our work addresses the problem of learning latent actions world models on in-the-wild videos, expanding the scope of existing works that focus on simple robotics simulations, video games, or manipulation data. While this allows us to capture richer actions, it also introduces challenges stemming from the video diversity, such as environmental noise, or the lack of a common embodiment across videos. To address some of the challenges, we discuss properties that actions should follow as well as relevant architectural choices and evaluations. We find that continuous, but constrained, latent actions are able to capture the complexity of actions from in-the-wild videos, something that the common vector quantization does not. We for example find that changes in the environment coming from agents, such as humans entering the room, can be transferred across videos. This highlights the capability of learning actions that are specific to in-the-wild videos. In the absence of a common embodiment across videos, we are mainly able to learn latent actions that become localized in space, relative to the camera. Nonetheless, we are able to train a controller that maps known actions to latent ones, allowing us to use latent actions as a universal interface and solve planning tasks with our world model with similar performance as action-conditioned baselines. Our analyses and experiments provide a step towards scaling latent action models to the real world.
>
---
#### [new 086] A Vision for Multisensory Intelligence: Sensing, Synergy, and Science
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出多感官人工智能的研究愿景，旨在解决AI与人类感官融合的问题。通过感知、科学和协同三个方向，推动AI更全面地理解世界。**

- **链接: [https://arxiv.org/pdf/2601.04563v1](https://arxiv.org/pdf/2601.04563v1)**

> **作者:** Paul Pu Liang
>
> **摘要:** Our experience of the world is multisensory, spanning a synthesis of language, sight, sound, touch, taste, and smell. Yet, artificial intelligence has primarily advanced in digital modalities like text, vision, and audio. This paper outlines a research vision for multisensory artificial intelligence over the next decade. This new set of technologies can change how humans and AI experience and interact with one another, by connecting AI to the human senses and a rich spectrum of signals from physiological and tactile cues on the body, to physical and social signals in homes, cities, and the environment. We outline how this field must advance through three interrelated themes of sensing, science, and synergy. Firstly, research in sensing should extend how AI captures the world in richer ways beyond the digital medium. Secondly, developing a principled science for quantifying multimodal heterogeneity and interactions, developing unified modeling architectures and representations, and understanding cross-modal transfer. Finally, we present new technical challenges to learn synergy between modalities and between humans and AI, covering multisensory integration, alignment, reasoning, generation, generalization, and experience. Accompanying this vision paper are a series of projects, resources, and demos of latest advances from the Multisensory Intelligence group at the MIT Media Lab, see https://mit-mi.github.io/.
>
---
#### [new 087] Towards Spatio-Temporal Extrapolation of Phase-Field Simulations with Convolution-Only Neural Networks
- **分类: cs.CE; cs.AI; cs.CV; cs.LG; math.NA**

- **简介: 该论文属于相场模拟的时空外推任务，旨在解决大尺度和长时间模拟计算成本高的问题。通过构建卷积U-Net网络和条件扩散模型，实现高效准确的模拟外推。**

- **链接: [https://arxiv.org/pdf/2601.04510v1](https://arxiv.org/pdf/2601.04510v1)**

> **作者:** Christophe Bonneville; Nathan Bieberdorf; Pieterjan Robbe; Mark Asta; Habib Najm; Laurent Capolungo; Cosmin Safta
>
> **摘要:** Phase-field simulations of liquid metal dealloying (LMD) can capture complex microstructural evolutions but can be prohibitively expensive for large domains and long time horizons. In this paper, we introduce a fully convolutional, conditionally parameterized U-Net surrogate designed to extrapolate far beyond its training data in both space and time. The architecture integrates convolutional self-attention, physically informed padding, and a flood-fill corrector method to maintain accuracy under extreme extrapolation, while conditioning on simulation parameters allows for flexible time-step skipping and adaptation to varying alloy compositions. To remove the need for costly solver-based initialization, we couple the surrogate with a conditional diffusion model that generates synthetic, physically consistent initial conditions. We train our surrogate on simulations generated over small domain sizes and short time spans, but, by taking advantage of the convolutional nature of U-Nets, we are able to run and extrapolate surrogate simulations for longer time horizons than what would be achievable with classic numerical solvers. Across multiple alloy compositions, the framework is able to reproduce the LMD physics accurately. It predicts key quantities of interest and spatial statistics with relative errors typically below 5% in the training regime and under 15% during large-scale, long time-horizon extrapolations. Our framework can also deliver speed-ups of up to 36,000 times, bringing the time to run weeks-long simulations down to a few seconds. This work is a first stepping stone towards high-fidelity extrapolation in both space and time of phase-field simulation for LMD.
>
---
#### [new 088] Scalable neural pushbroom architectures for real-time denoising of hyperspectral images onboard satellites
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于卫星实时去噪任务，解决 onboard 处理中的低复杂度、动态功耗和容错问题。提出一种线性处理的神经网络架构，实现高效去噪。**

- **链接: [https://arxiv.org/pdf/2601.05020v1](https://arxiv.org/pdf/2601.05020v1)**

> **作者:** Ziyao Yi; Davide Piccinini; Diego Valsesia; Tiziano Bianchi; Enrico Magli
>
> **摘要:** The next generation of Earth observation satellites will seek to deploy intelligent models directly onboard the payload in order to minimize the latency incurred by the transmission and processing chain of the ground segment, for time-critical applications. Designing neural architectures for onboard execution, particularly for satellite-based hyperspectral imagers, poses novel challenges due to the unique constraints of this environment and imaging system that are largely unexplored by the traditional computer vision literature. In this paper, we show that this setting requires addressing three competing objectives, namely high-quality inference with low complexity, dynamic power scalability and fault tolerance. We focus on the problem of hyperspectral image denoising, which is a critical task to enable effective downstream inference, and highlights the constraints of the onboard processing scenario. We propose a neural network design that addresses the three aforementioned objectives with several novel contributions. In particular, we propose a mixture of denoisers that can be resilient to radiation-induced faults as well as allowing for time-varying power scaling. Moreover, each denoiser employs an innovative architecture where an image is processed line-by-line in a causal way, with a memory of past lines, in order to match the acquisition process of pushbroom hyperspectral sensors and greatly limit memory requirements. We show that the proposed architecture can run in real-time, i.e., process one line in the time it takes to acquire the next one, on low-power hardware and provide competitive denoising quality with respect to significantly more complex state-of-the-art models. We also show that the power scalability and fault tolerance objectives provide a design space with multiple tradeoffs between those properties and denoising quality.
>
---
#### [new 089] Quantitative mapping from conventional MRI using self-supervised physics-guided deep learning: applications to a large-scale, clinically heterogeneous dataset
- **分类: physics.med-ph; cs.CV; cs.LG**

- **简介: 该论文属于医学影像处理任务，旨在解决传统MRI定量信息不足的问题。通过自监督物理引导的深度学习框架，从常规MRI生成定量T1、T2和PD图，提升 biomarker 研究的可行性。**

- **链接: [https://arxiv.org/pdf/2601.05063v1](https://arxiv.org/pdf/2601.05063v1)**

> **作者:** Jelmer van Lune; Stefano Mandija; Oscar van der Heide; Matteo Maspero; Martin B. Schilder; Jan Willem Dankbaar; Cornelis A. T. van den Berg; Alessandro Sbrizzi
>
> **备注:** 30 pages, 13 figures, full paper
>
> **摘要:** Magnetic resonance imaging (MRI) is a cornerstone of clinical neuroimaging, yet conventional MRIs provide qualitative information heavily dependent on scanner hardware and acquisition settings. While quantitative MRI (qMRI) offers intrinsic tissue parameters, the requirement for specialized acquisition protocols and reconstruction algorithms restricts its availability and impedes large-scale biomarker research. This study presents a self-supervised physics-guided deep learning framework to infer quantitative T1, T2, and proton-density (PD) maps directly from widely available clinical conventional T1-weighted, T2-weighted, and FLAIR MRIs. The framework was trained and evaluated on a large-scale, clinically heterogeneous dataset comprising 4,121 scan sessions acquired at our institution over six years on four different 3 T MRI scanner systems, capturing real-world clinical variability. The framework integrates Bloch-based signal models directly into the training objective. Across more than 600 test sessions, the generated maps exhibited white matter and gray matter values consistent with literature ranges. Additionally, the generated maps showed invariance to scanner hardware and acquisition protocol groups, with inter-group coefficients of variation $\leq$ 1.1%. Subject-specific analyses demonstrated excellent voxel-wise reproducibility across scanner systems and sequence parameters, with Pearson $r$ and concordance correlation coefficients exceeding 0.82 for T1 and T2. Mean relative voxel-wise differences were low across all quantitative parameters, especially for T2 ($<$ 6%). These results indicate that the proposed framework can robustly transform diverse clinical conventional MRI data into quantitative maps, potentially paving the way for large-scale quantitative biomarker research.
>
---
#### [new 090] GenAI-DrawIO-Creator: A Framework for Automated Diagram Generation
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于自动化绘图任务，旨在解决手动创建和修改图表耗时的问题。工作包括设计框架，利用LLM生成和操作XML格式的图表。**

- **链接: [https://arxiv.org/pdf/2601.05162v1](https://arxiv.org/pdf/2601.05162v1)**

> **作者:** Jinze Yu; Dayuan Jiang
>
> **摘要:** Diagrams are crucial for communicating complex information, yet creating and modifying them remains a labor-intensive task. We present GenAI-DrawIO-Creator, a novel framework that leverages Large Language Models (LLMs) to automate diagram generation and manipulation in the structured XML format used by draw.io. Our system integrates Claude 3.7 to reason about structured visual data and produce valid diagram representations. Key contributions include a high-level system design enabling real-time diagram updates, specialized prompt engineering and error-checking to ensure well-formed XML outputs. We demonstrate a working prototype capable of generating accurate diagrams (such as network architectures and flowcharts) from natural language or code, and even replicating diagrams from images. Simulated evaluations show that our approach significantly reduces diagram creation time and produces outputs with high structural fidelity. Our results highlight the promise of Claude 3.7 in handling structured visual reasoning tasks and lay the groundwork for future research in AI-assisted diagramming applications.
>
---
#### [new 091] Generate, Transfer, Adapt: Learning Functional Dexterous Grasping from a Single Human Demonstration
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取任务，旨在解决功能性抓取的泛化与数据稀缺问题。通过单次人类示范生成高质量训练数据，并结合多模态网络实现高效抓取预测。**

- **链接: [https://arxiv.org/pdf/2601.05243v1](https://arxiv.org/pdf/2601.05243v1)**

> **作者:** Xingyi He; Adhitya Polavaram; Yunhao Cao; Om Deshmukh; Tianrui Wang; Xiaowei Zhou; Kuan Fang
>
> **备注:** Project Page: https://cordex-manipulation.github.io/
>
> **摘要:** Functional grasping with dexterous robotic hands is a key capability for enabling tool use and complex manipulation, yet progress has been constrained by two persistent bottlenecks: the scarcity of large-scale datasets and the absence of integrated semantic and geometric reasoning in learned models. In this work, we present CorDex, a framework that robustly learns dexterous functional grasps of novel objects from synthetic data generated from just a single human demonstration. At the core of our approach is a correspondence-based data engine that generates diverse, high-quality training data in simulation. Based on the human demonstration, our data engine generates diverse object instances of the same category, transfers the expert grasp to the generated objects through correspondence estimation, and adapts the grasp through optimization. Building on the generated data, we introduce a multimodal prediction network that integrates visual and geometric information. By devising a local-global fusion module and an importance-aware sampling mechanism, we enable robust and computationally efficient prediction of functional dexterous grasps. Through extensive experiments across various object categories, we demonstrate that CorDex generalizes well to unseen object instances and significantly outperforms state-of-the-art baselines.
>
---
#### [new 092] End-to-end differentiable design of geometric waveguide displays
- **分类: physics.optics; cs.CV; cs.GR**

- **简介: 该论文属于光学设计任务，解决几何波导显示中光传输与镀层优化难题。通过端到端可微框架联合优化非顺序光线追踪与薄膜涂层，提升显示效率与均匀性。**

- **链接: [https://arxiv.org/pdf/2601.04370v1](https://arxiv.org/pdf/2601.04370v1)**

> **作者:** Xinge Yang; Zhaocheng Liu; Zhaoyu Nie; Qingyuan Fan; Zhimin Shi; Jim Bonar; Wolfgang Heidrich
>
> **摘要:** Geometric waveguides are a promising architecture for optical see-through augmented reality displays, but their performance is severely bottlenecked by the difficulty of jointly optimizing non-sequential light transport and polarization-dependent multilayer thin-film coatings. Here we present the first end-to-end differentiable optimization framework for geometric waveguide that couples non-sequential Monte Carlo polarization ray tracing with a differentiable transfer-matrix thin-film solver. A differentiable Monte Carlo ray tracer avoids the exponential growth of deterministic ray splitting while enabling gradients backpropagation from eyebox metrics to design parameters. With memory-saving strategies, we optimize more than one thousand layer-thickness parameters and billions of non-sequential ray-surface intersections on a single multi-GPU workstation. Automated layer pruning is achieved by starting from over-parameterized stacks and driving redundant layers to zero thickness under discrete manufacturability constraints, effectively performing topology optimization to discover optimal coating structures. On a representative design, starting from random initialization within thickness bounds, our method increases light efficiency from 4.1\% to 33.5\% and improves eyebox and FoV uniformity by $\sim$17$\times$ and $\sim$11$\times$, respectively. Furthermore, we jointly optimize the waveguide and an image preprocessing network to improve perceived image quality. Our framework not only enables system-level, high-dimensional coating optimization inside the waveguide, but also expands the scope of differentiable optics for next-generation optical design.
>
---
#### [new 093] IGenBench: Benchmarking the Reliability of Text-to-Infographic Generation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于文本到信息图生成任务，旨在评估生成模型的可靠性。通过构建基准测试集和自动化评估框架，分析模型在数据准确性和内容正确性上的表现。**

- **链接: [https://arxiv.org/pdf/2601.04498v1](https://arxiv.org/pdf/2601.04498v1)**

> **作者:** Yinghao Tang; Xueding Liu; Boyuan Zhang; Tingfeng Lan; Yupeng Xie; Jiale Lao; Yiyao Wang; Haoxuan Li; Tingting Gao; Bo Pan; Luoxuan Weng; Xiuqi Huang; Minfeng Zhu; Yingchaojie Feng; Yuyu Luo; Wei Chen
>
> **摘要:** Infographics are composite visual artifacts that combine data visualizations with textual and illustrative elements to communicate information. While recent text-to-image (T2I) models can generate aesthetically appealing images, their reliability in generating infographics remains unclear. Generated infographics may appear correct at first glance but contain easily overlooked issues, such as distorted data encoding or incorrect textual content. We present IGENBENCH, the first benchmark for evaluating the reliability of text-to-infographic generation, comprising 600 curated test cases spanning 30 infographic types. We design an automated evaluation framework that decomposes reliability verification into atomic yes/no questions based on a taxonomy of 10 question types. We employ multimodal large language models (MLLMs) to verify each question, yielding question-level accuracy (Q-ACC) and infographic-level accuracy (I-ACC). We comprehensively evaluate 10 state-of-the-art T2I models on IGENBENCH. Our systematic analysis reveals key insights for future model development: (i) a three-tier performance hierarchy with the top model achieving Q-ACC of 0.90 but I-ACC of only 0.49; (ii) data-related dimensions emerging as universal bottlenecks (e.g., Data Completeness: 0.21); and (iii) the challenge of achieving end-to-end correctness across all models. We release IGENBENCH at https://igen-bench.vercel.app/.
>
---
#### [new 094] Aligned explanations in neural networks
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文属于解释深度学习模型的任务，旨在解决解释与预测不一致的问题。提出PiNets框架，使模型预测与解释直接对齐，提升可解释性。**

- **链接: [https://arxiv.org/pdf/2601.04378v1](https://arxiv.org/pdf/2601.04378v1)**

> **作者:** Corentin Lobet; Francesca Chiaromonte
>
> **摘要:** Feature attribution is the dominant paradigm for explaining deep neural networks. However, most existing methods only loosely reflect the model's prediction-making process, thereby merely white-painting the black box. We argue that explanatory alignment is a key aspect of trustworthiness in prediction tasks: explanations must be directly linked to predictions, rather than serving as post-hoc rationalizations. We present model readability as a design principle enabling alignment, and PiNets as a modeling framework to pursue it in a deep learning context. PiNets are pseudo-linear networks that produce instance-wise linear predictions in an arbitrary feature space, making them linearly readable. We illustrate their use on image classification and segmentation tasks, demonstrating how PiNets produce explanations that are faithful across multiple criteria in addition to alignment.
>
---
#### [new 095] ArtCognition: A Multimodal AI Framework for Affective State Sensing from Visual and Kinematic Drawing Cues
- **分类: cs.LG; cs.CV; cs.HC; cs.IR**

- **简介: 该论文属于情感状态感知任务，旨在通过视觉和动态绘画数据评估心理状态。提出ArtCognition框架，融合视觉与行为特征，提升心理评估的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.04297v1](https://arxiv.org/pdf/2601.04297v1)**

> **作者:** Behrad Binaei-Haghighi; Nafiseh Sadat Sajadi; Mehrad Liviyan; Reyhane Akhavan Kharazi; Fatemeh Amirkhani; Behnam Bahrak
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** The objective assessment of human affective and psychological states presents a significant challenge, particularly through non-verbal channels. This paper introduces digital drawing as a rich and underexplored modality for affective sensing. We present a novel multimodal framework, named ArtCognition, for the automated analysis of the House-Tree-Person (HTP) test, a widely used psychological instrument. ArtCognition uniquely fuses two distinct data streams: static visual features from the final artwork, captured by computer vision models, and dynamic behavioral kinematic cues derived from the drawing process itself, such as stroke speed, pauses, and smoothness. To bridge the gap between low-level features and high-level psychological interpretation, we employ a Retrieval-Augmented Generation (RAG) architecture. This grounds the analysis in established psychological knowledge, enhancing explainability and reducing the potential for model hallucination. Our results demonstrate that the fusion of visual and behavioral kinematic cues provides a more nuanced assessment than either modality alone. We show significant correlations between the extracted multimodal features and standardized psychological metrics, validating the framework's potential as a scalable tool to support clinicians. This work contributes a new methodology for non-intrusive affective state assessment and opens new avenues for technology-assisted mental healthcare.
>
---
#### [new 096] UNIC: Learning Unified Multimodal Extrinsic Contact Estimation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于接触估计任务，旨在解决无先验知识下的外部接触估计问题。提出UNIC框架，融合多模态数据，实现可靠、通用的接触感知。**

- **链接: [https://arxiv.org/pdf/2601.04356v1](https://arxiv.org/pdf/2601.04356v1)**

> **作者:** Zhengtong Xu; Yuki Shirai
>
> **摘要:** Contact-rich manipulation requires reliable estimation of extrinsic contacts-the interactions between a grasped object and its environment which provide essential contextual information for planning, control, and policy learning. However, existing approaches often rely on restrictive assumptions, such as predefined contact types, fixed grasp configurations, or camera calibration, that hinder generalization to novel objects and deployment in unstructured environments. In this paper, we present UNIC, a unified multimodal framework for extrinsic contact estimation that operates without any prior knowledge or camera calibration. UNIC directly encodes visual observations in the camera frame and integrates them with proprioceptive and tactile modalities in a fully data-driven manner. It introduces a unified contact representation based on scene affordance maps that captures diverse contact formations and employs a multimodal fusion mechanism with random masking, enabling robust multimodal representation learning. Extensive experiments demonstrate that UNIC performs reliably. It achieves a 9.6 mm average Chamfer distance error on unseen contact locations, performs well on unseen objects, remains robust under missing modalities, and adapts to dynamic camera viewpoints. These results establish extrinsic contact estimation as a practical and versatile capability for contact-rich manipulation.
>
---
#### [new 097] Illumination Angular Spectrum Encoding for Controlling the Functionality of Diffractive Networks
- **分类: physics.optics; cs.CV; cs.LG**

- **简介: 该论文属于多任务光学计算领域，旨在解决 diffractive 网络单一功能限制问题。通过调节照明角度谱实现多功能控制，训练网络完成多种图像转换任务。**

- **链接: [https://arxiv.org/pdf/2601.04825v1](https://arxiv.org/pdf/2601.04825v1)**

> **作者:** Matan Kleiner; Lior Michaeli; Tomer Michaeli
>
> **备注:** Project's code https://github.com/matankleiner/Angular-Spectrum-Encoding
>
> **摘要:** Diffractive neural networks have recently emerged as a promising framework for all-optical computing. However, these networks are typically trained for a single task, limiting their potential adoption in systems requiring multiple functionalities. Existing approaches to achieving multi-task functionality either modify the mechanical configuration of the network per task or use a different illumination wavelength or polarization state for each task. In this work, we propose a new control mechanism, which is based on the illumination's angular spectrum. Specifically, we shape the illumination using an amplitude mask that selectively controls its angular spectrum. We employ different illumination masks for achieving different network functionalities, so that the mask serves as a unique task encoder. Interestingly, we show that effective control can be achieved over a very narrow angular range, within the paraxial regime. We numerically illustrate the proposed approach by training a single diffractive network to perform multiple image-to-image translation tasks. In particular, we demonstrate translating handwritten digits into typeset digits of different values, and translating handwritten English letters into typeset numbers and typeset Greek letters, where the type of the output is determined by the illumination's angular components. As we show, the proposed framework can work under different coherence conditions, and can be combined with existing control strategies, such as different wavelengths. Our results establish the illumination angular spectrum as a powerful degree of freedom for controlling diffractive networks, enabling a scalable and versatile framework for multi-task all-optical computing.
>
---
## 更新

#### [replaced 001] Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于OCR任务，旨在评估DeepSeek-OCR的性能驱动因素。通过语义扰动实验，发现其性能依赖语言先验而非视觉能力，揭示了视觉-文本压缩的局限性。**

- **链接: [https://arxiv.org/pdf/2601.03714v2](https://arxiv.org/pdf/2601.03714v2)**

> **作者:** Yunhao Liang; Ruixuan Ying; Bo Li; Hong Li; Kai Yan; Qingwen Li; Min Yang; Okamoto Satoshi; Zhe Cui; Shiwen Ni
>
> **摘要:** DeepSeek-OCR utilizes an optical 2D mapping approach to achieve high-ratio vision-text compression, claiming to decode text tokens exceeding ten times the input visual tokens. While this suggests a promising solution for the LLM long-context bottleneck, we investigate a critical question: "Visual merit or linguistic crutch - which drives DeepSeek-OCR's performance?" By employing sentence-level and word-level semantic corruption, we isolate the model's intrinsic OCR capabilities from its language priors. Results demonstrate that without linguistic support, DeepSeek-OCR's performance plummets from approximately 90% to 20%. Comparative benchmarking against 13 baseline models reveals that traditional pipeline OCR methods exhibit significantly higher robustness to such semantic perturbations than end-to-end methods. Furthermore, we find that lower visual token counts correlate with increased reliance on priors, exacerbating hallucination risks. Context stress testing also reveals a total model collapse around 10,000 text tokens, suggesting that current optical compression techniques may paradoxically aggravate the long-context bottleneck. This study empirically defines DeepSeek-OCR's capability boundaries and offers essential insights for future optimizations of the vision-text compression paradigm. We release all data, results and scripts used in this study at https://github.com/dududuck00/DeepSeekOCR.
>
---
#### [replaced 002] CADmium: Fine-Tuning Code Language Models for Text-Driven Sequential CAD Design
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.09792v3](https://arxiv.org/pdf/2507.09792v3)**

> **作者:** Prashant Govindarajan; Davide Baldelli; Jay Pathak; Quentin Fournier; Sarath Chandar
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR) 01/2026
>
> **摘要:** Computer-aided design (CAD) is the digital construction of 2D and 3D objects, and is central to a wide range of engineering and manufacturing applications like automobile and aviation. Despite its importance, CAD modeling remains largely a time-intensive, manual task. Recent works have attempted to automate this process with small transformer-based models and handcrafted CAD sequence representations. However, there has been little effort to leverage the potential of large language models (LLMs) for sequential CAD design. In this work, we introduce a new large-scale dataset of more than 170k CAD models annotated with high-quality, human-like descriptions generated with our pipeline based on GPT-4.1. Using this dataset, we fine-tune powerful code-LLMs to generate CAD sequences represented in a JSON-based format from natural language descriptions, demonstrating the viability and effectiveness of this approach for text-conditioned CAD generation. Because simple metrics often fail to reflect the quality of generated objects, we introduce geometric and topological metrics based on sphericity, mean curvature, and Euler characteristic to provide richer structural insights. Our experiments and ablation studies on both synthetic and human-annotated data demonstrate that CADmium is able to automate CAD design, drastically speeding up the design of new objects. The dataset, code, and fine-tuned models are available online.
>
---
#### [replaced 003] MM-Sonate: Multimodal Controllable Audio-Video Generation with Zero-Shot Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于多模态生成任务，解决音频视频同步与零样本语音克隆问题。提出MM-Sonate框架，实现精准语音控制与高质量语音克隆。**

- **链接: [https://arxiv.org/pdf/2601.01568v2](https://arxiv.org/pdf/2601.01568v2)**

> **作者:** Chunyu Qiang; Jun Wang; Xiaopeng Wang; Kang Yin; Yuxin Guo
>
> **摘要:** Joint audio-video generation aims to synthesize synchronized multisensory content, yet current unified models struggle with fine-grained acoustic control, particularly for identity-preserving speech. Existing approaches either suffer from temporal misalignment due to cascaded generation or lack the capability to perform zero-shot voice cloning within a joint synthesis framework. In this work, we present MM-Sonate, a multimodal flow-matching framework that unifies controllable audio-video joint generation with zero-shot voice cloning capabilities. Unlike prior works that rely on coarse semantic descriptions, MM-Sonate utilizes a unified instruction-phoneme input to enforce strict linguistic and temporal alignment. To enable zero-shot voice cloning, we introduce a timbre injection mechanism that effectively decouples speaker identity from linguistic content. Furthermore, addressing the limitations of standard classifier-free guidance in multimodal settings, we propose a noise-based negative conditioning strategy that utilizes natural noise priors to significantly enhance acoustic fidelity. Empirical evaluations demonstrate that MM-Sonate establishes new state-of-the-art performance in joint generation benchmarks, significantly outperforming baselines in lip synchronization and speech intelligibility, while achieving voice cloning fidelity comparable to specialized Text-to-Speech systems.
>
---
#### [replaced 004] UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.03193v2](https://arxiv.org/pdf/2601.03193v2)**

> **作者:** Ruiyan Han; Zhen Fang; XinYu Sun; Yuchen Ma; Ziheng Wang; Yu Zeng; Zehui Chen; Lin Chen; Wenxuan Huang; Wei-Jie Xu; Yi Cao; Feng Zhao
>
> **摘要:** While Unified Multimodal Models (UMMs) have achieved remarkable success in cross-modal comprehension, a significant gap persists in their ability to leverage such internal knowledge for high-quality generation. We formalize this discrepancy as Conduction Aphasia, a phenomenon where models accurately interpret multimodal inputs but struggle to translate that understanding into faithful and controllable synthesis. To address this, we propose UniCorn, a simple yet elegant self-improvement framework that eliminates the need for external data or teacher supervision. By partitioning a single UMM into three collaborative roles: Proposer, Solver, and Judge, UniCorn generates high-quality interactions via self-play and employs cognitive pattern reconstruction to distill latent understanding into explicit generative signals. To validate the restoration of multimodal coherence, we introduce UniCycle, a cycle-consistency benchmark based on a Text to Image to Text reconstruction loop. Extensive experiments demonstrate that UniCorn achieves comprehensive and substantial improvements over the base model across six general image generation benchmarks. Notably, it achieves SOTA performance on TIIF(73.8), DPG(86.8), CompBench(88.5), and UniCycle while further delivering substantial gains of +5.0 on WISE and +6.5 on OneIG. These results highlight that our method significantly enhances T2I generation while maintaining robust comprehension, demonstrating the scalability of fully self-supervised refinement for unified multimodal intelligence.
>
---
#### [replaced 005] CrackSegFlow: Controllable Flow Matching Synthesis for Generalizable Crack Segmentation with a 50K Image-Mask Benchmark
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.03637v2](https://arxiv.org/pdf/2601.03637v2)**

> **作者:** Babak Asadi; Peiyang Wu; Mani Golparvar-Fard; Ramez Hajj
>
> **摘要:** Automated crack segmentation is essential for condition assessment, yet deployment is limited by scarce pixel-level labels and domain shift. We present CrackSegFlow, a controllable flow-matching synthesis framework that generates crack images conditioned on binary masks with mask-image alignment. The renderer combines topology-preserving mask injection with edge gating to maintain thin-structure continuity and suppress false positives. A class-conditional flow-matching mask model synthesizes masks with control over crack coverage, enabling balanced, topology-diverse data without manual annotation. We inject masks into crack-free backgrounds to diversify illumination and reduce false positives. On five datasets with a CNN-Transformer backbone, incorporating synthesized pairs improves in-domain performance by 5.37 mIoU and 5.13 F1, and target-guided cross-domain synthesis yields gains of 13.12 mIoU and 14.82 F1 using target mask statistics. We also release CSF-50K, 50,000 image-mask pairs for benchmarking.
>
---
#### [replaced 006] Crafting Adversarial Inputs for Large Vision-Language Models Using Black-Box Optimization
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.01747v2](https://arxiv.org/pdf/2601.01747v2)**

> **作者:** Jiwei Guan; Haibo Jin; Haohan Wang
>
> **备注:** EACL
>
> **摘要:** Recent advancements in Large Vision-Language Models (LVLMs) have shown groundbreaking capabilities across diverse multimodal tasks. However, these models remain vulnerable to adversarial jailbreak attacks, where adversaries craft subtle perturbations to bypass safety mechanisms and trigger harmful outputs. Existing white-box attacks methods require full model accessibility, suffer from computing costs and exhibit insufficient adversarial transferability, making them impractical for real-world, black-box settings. To address these limitations, we propose a black-box jailbreak attack on LVLMs via Zeroth-Order optimization using Simultaneous Perturbation Stochastic Approximation (ZO-SPSA). ZO-SPSA provides three key advantages: (i) gradient-free approximation by input-output interactions without requiring model knowledge, (ii) model-agnostic optimization without the surrogate model and (iii) lower resource requirements with reduced GPU memory consumption. We evaluate ZO-SPSA on three LVLMs, including InstructBLIP, LLaVA and MiniGPT-4, achieving the highest jailbreak success rate of 83.0% on InstructBLIP, while maintaining imperceptible perturbations comparable to white-box methods. Moreover, adversarial examples generated from MiniGPT-4 exhibit strong transferability to other LVLMs, with ASR reaching 64.18%. These findings underscore the real-world feasibility of black-box jailbreaks and expose critical weaknesses in the safety mechanisms of current LVLMs
>
---
#### [replaced 007] NASTaR: NovaSAR Automated Ship Target Recognition Dataset
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.18503v2](https://arxiv.org/pdf/2512.18503v2)**

> **作者:** Benyamin Hosseiny; Kamirul Kamirul; Odysseas Pappas; Alin Achim
>
> **摘要:** Synthetic Aperture Radar (SAR) offers a unique capability for all-weather, space-based maritime activity monitoring by capturing and imaging strong reflections from ships at sea. A well-defined challenge in this domain is ship type classification. Due to the high diversity and complexity of ship types, accurate recognition is difficult and typically requires specialized deep learning models. These models, however, depend on large, high-quality ground-truth datasets to achieve robust performance and generalization. Furthermore, the growing variety of SAR satellites operating at different frequencies and spatial resolutions has amplified the need for more annotated datasets to enhance model accuracy. To address this, we present the NovaSAR Automated Ship Target Recognition (NASTaR) dataset. This dataset comprises of 3415 ship patches extracted from NovaSAR S-band imagery, with labels matched to AIS data. It includes distinctive features such as 23 unique classes, inshore/offshore separation, and an auxiliary wake dataset for patches where ship wakes are visible. We validated the dataset applicability across prominent ship-type classification scenarios using benchmark deep learning models. Results demonstrate over 60% accuracy for classifying four major ship types, over 70% for a three-class scenario, more than 75% for distinguishing cargo from tanker ships, and over 87% for identifying fishing vessels. The NASTaR dataset is available at https://doi.org/10.5523/bris.2tfa6x37oerz2lyiw6hp47058, while relevant codes for benchmarking and analysis are available at https://github.com/benyaminhosseiny/nastar.
>
---
#### [replaced 008] Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.12766v2](https://arxiv.org/pdf/2506.12766v2)**

> **作者:** Ruojing Li; Wei An; Yingqian Wang; Xinyi Ying; Yimian Dai; Longguang Wang; Miao Li; Yulan Guo; Li Liu
>
> **摘要:** Infrared small target (IRST) detection is challenging in simultaneously achieving precise, robust, and efficient performance due to extremely dim targets and strong interference. Current learning-based methods attempt to leverage ``more" information from both the spatial and the short-term temporal domains, but suffer from unreliable performance under complex conditions while incurring computational redundancy. In this paper, we explore the ``more essential" information from a more crucial domain for the detection. Through theoretical analysis, we reveal that the global temporal saliency and correlation information in the temporal profile demonstrate significant superiority in distinguishing target signals from other signals. To investigate whether such superiority is preferentially leveraged by well-trained networks, we built the first prediction attribution tool in this field and verified the importance of the temporal profile information. Inspired by the above conclusions, we remodel the IRST detection task as a one-dimensional signal anomaly detection task, and propose an efficient deep temporal probe network (DeepPro) that only performs calculations in the time dimension for IRST detection. We conducted extensive experiments to fully validate the effectiveness of our method. The experimental results are exciting, as our DeepPro outperforms existing state-of-the-art IRST detection methods on widely-used benchmarks with extremely high efficiency, and achieves a significant improvement on dim targets and in complex scenarios. We provide a new modeling domain, a new insight, a new method, and a new performance, which can promote the development of IRST detection. Codes are available at https://github.com/TinaLRJ/DeepPro.
>
---
#### [replaced 009] Explainable Binary Classification of Separable Shape Ensembles
- **分类: cs.CV; math.ST**

- **链接: [https://arxiv.org/pdf/2410.12994v2](https://arxiv.org/pdf/2410.12994v2)**

> **作者:** Zachary Grey; Nicholas Fisher; Andrew Glaws
>
> **备注:** 32 pages, 16 figures
>
> **摘要:** Scientists, engineers, biologists, and technology specialists universally leverage image segmentation to extract shape ensembles containing many thousands of curves representing patterns in observations and measurements. These large curve ensembles facilitate inferences about important changes when comparing and contrasting images. We introduce novel pattern recognition formalisms combined with inference methods over large ensembles of segmented curves. Our formalism involves accurately approximating eigenspaces of composite integral operators to motivate discrete, dual representations of curves collocated at quadrature nodes. Approximations are projected onto underlying matrix manifolds and the resulting separable shape tensors constitute rigid-invariant decompositions of curves into generalized (linear) scale variations and complementary (nonlinear) undulations. With thousands of curves segmented from pairs of images, we demonstrate how data-driven features of separable shape tensors inform explainable binary classification utilizing a product maximum mean discrepancy; absent labeled data, building interpretable feature spaces in seconds without high performance computation, and detecting discrepancies below cursory visual inspections.
>
---
#### [replaced 010] Single Image Reflection Separation via Dual Prior Interaction Transformer
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.12641v2](https://arxiv.org/pdf/2505.12641v2)**

> **作者:** Yue Huang; Zi'ang Li; Tianle Hu; Jie Wen; Guanbin Li; Jinglin Zhang; Guoxu Zhou; Xiaozhao Fang
>
> **摘要:** Single image reflection separation aims to separate the transmission and reflection layers from a mixed image. Existing methods typically combine general priors from pre-trained models with task-specific priors such as text prompts and reflection detection. However, the transmission prior, as the most direct task-specific prior for the target transmission layer, has not been effectively modeled or fully utilized, limiting performance in complex scenarios. To address this issue, we propose a dual-prior interaction framework based on lightweight transmission prior generation and effective prior fusion. First, we design a Local Linear Correction Network (LLCN) that finetunes pre-trained models based on the physical constraint T=SI+B, where S and B represent pixel-wise and channel-wise scaling and bias transformations. LLCN efficiently generates high-quality transmission priors with minimal parameters. Second, we construct a Dual-Prior Interaction Transformer (DPIT) that employs a dual-stream channel reorganization attention mechanism. By reorganizing features from general and transmission priors for attention computation, DPIT achieves deep fusion of both priors, fully exploiting their complementary information. Experimental results on multiple benchmark datasets demonstrate that the proposed method achieves state-of-the-art performance.
>
---
#### [replaced 011] Cognitive-Hierarchy Guided End-to-End Planning for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决传统方法与人类认知不匹配的问题。提出CogAD模型，通过双层次机制实现更接近人类的感知与规划。**

- **链接: [https://arxiv.org/pdf/2505.21581v3](https://arxiv.org/pdf/2505.21581v3)**

> **作者:** Zhennan Wang; Jianing Teng; Canqun Xiang; Kangliang Chen; Xing Pan; Lu Deng; Weihao Gu
>
> **摘要:** While end-to-end autonomous driving has advanced significantly, prevailing methods remain fundamentally misaligned with human cognitive principles in both perception and planning. In this paper, we propose CogAD, a novel end-to-end autonomous driving model that emulates the hierarchical cognition mechanisms of human drivers. CogAD implements dual hierarchical mechanisms: global-to-local context processing for human-like perception and intent-conditioned multi-mode trajectory generation for cognitively-inspired planning. The proposed method demonstrates three principal advantages: comprehensive environmental understanding through hierarchical perception, robust planning exploration enabled by multi-level planning, and diverse yet reasonable multi-modal trajectory generation facilitated by dual-level uncertainty modeling. Extensive experiments on nuScenes and Bench2Drive demonstrate that CogAD achieves state-of-the-art performance in end-to-end planning, exhibiting particular superiority in long-tail scenarios and robust generalization to complex real-world driving conditions.
>
---
#### [replaced 012] Jailbreaking Safeguarded Text-to-Image Models via Large Language Models
- **分类: cs.CR; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于安全防护任务，旨在解决文本生成图像模型被攻击绕过安全机制的问题。通过微调大语言模型生成对抗提示，有效突破安全防护。**

- **链接: [https://arxiv.org/pdf/2503.01839v2](https://arxiv.org/pdf/2503.01839v2)**

> **作者:** Zhengyuan Jiang; Yuepeng Hu; Yuchen Yang; Yinzhi Cao; Neil Zhenqiang Gong
>
> **备注:** Accepted by EACL 2026 Findings
>
> **摘要:** Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose \alg, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.
>
---
#### [replaced 013] MT-Video-Bench: A Holistic Video Understanding Benchmark for Evaluating Multimodal LLMs in Multi-Turn Dialogues
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.17722v2](https://arxiv.org/pdf/2510.17722v2)**

> **作者:** Yaning Pan; Qianqian Xie; Guohui Zhang; Zekun Wang; Yongqian Wen; Yuanxing Zhang; Haoxuan Hu; Zhiyu Pan; Yibing Huang; Zhidong Gan; Yonghong Lin; An Ping; Shihao Li; Yanghai Wang; Tianhao Peng; Jiaheng Liu
>
> **备注:** Project Website: https://github.com/NJU-LINK/MT-Video-Bench
>
> **摘要:** The recent development of Multimodal Large Language Models (MLLMs) has significantly advanced AI's ability to understand visual modalities. However, existing evaluation benchmarks remain limited to single-turn question answering, overlooking the complexity of multi-turn dialogues in real-world scenarios. To bridge this gap, we introduce MT-Video-Bench, a holistic video understanding benchmark for evaluating MLLMs in multi-turn dialogues. Specifically, our MT-Video-Bench mainly assesses 6 core competencies that focus on perceptivity and interactivity, encompassing 1,000 meticulously curated multi-turn dialogues from diverse domains. These capabilities are rigorously aligned with real-world applications, such as interactive sports analysis and multi-turn video-based intelligent tutoring. With MT-Video-Bench, we extensively evaluate various state-of-the-art open-source and closed-source MLLMs, revealing their significant performance discrepancies and limitations in handling multi-turn video dialogues. The benchmark will be publicly available to foster future research.
>
---
#### [replaced 014] MoIIE: Mixture of Intra- and Inter-Modality Experts for Large Vision Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.09779v2](https://arxiv.org/pdf/2508.09779v2)**

> **作者:** Dianyi Wang; Siyuan Wang; Zejun Li; Yikun Wang; Yitong Li; Duyu Tang; Xiaoyu Shen; Xuanjing Huang; Zhongyu Wei
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across multi-modal tasks by scaling model size and training data. However, these dense LVLMs incur significant computational costs and motivate the exploration of sparse Mixture of Experts (MoE) architectures. While MoE improve parameter efficiency, effectively applying MoE to simultaneously model modality-specific features and cross-modal associations in LVLMs remains challenging. In this work, we propose to incorporate Mixture of Intra- and Inter-Modality Experts (MoIIE) to LVLMs. For each token, expert routing is guided by its modality, directing tokens to their respective intra-modality experts as well as a shared pool of inter-modality experts, enabling the model to jointly learn rich intra-modal features and cross-modal interactions. We further introduce an effective and straightforward two-stage training strategy, which facilitates the direct activation of both MoE and multi-modal capabilities. Extensive experiments across different data scales and LLM backbone demonstrate the effectiveness, efficiency and generality of our approach. Notably, our MoIIE models with 5.5B and 11.3B activated parameters match or even surpass the performance of existing advanced open-source MoE-LLMs based multi-modal models that involve more activated parameters. The code is available at https://github.com/AlenjandroWang/MoIIE.
>
---
#### [replaced 015] Beyond Fixed Topologies: Unregistered Training and Comprehensive Evaluation Metrics for 3D Talking Heads
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.11041v3](https://arxiv.org/pdf/2410.11041v3)**

> **作者:** Federico Nocentini; Thomas Besnier; Claudio Ferrari; Sylvain Arguillere; Mohamed Daoudi; Stefano Berretti
>
> **备注:** https://fedenoce.github.io/scantalk/
>
> **摘要:** Generating speech-driven 3D talking heads presents numerous challenges; among those is dealing with varying mesh topologies where no point-wise correspondence exists across the meshes the model can animate. While previous literature works assume fixed mesh structures, in this work we present the first framework capable of animating 3D faces in arbitrary topologies, including real scanned data. Our approach leverages heat diffusion to predict features that are robust to the mesh topology. We explore two training settings: a registered one, in which meshes in a training sequences share a fixed topology but any mesh can be animated at test time, and an fully unregistered one, which allows effective training with varying mesh structures. Additionally, we highlight the limitations of current evaluation metrics and propose new metrics for better lip-syncing evaluation. An extensive evaluation shows our approach performs favorably compared to fixed topology techniques, setting a new benchmark by offering a versatile and high-fidelity solution for 3D talking heads where the topology constraint is dropped. The code along with the pre-trained model are available.
>
---
#### [replaced 016] Talk2Move: Reinforcement Learning for Text-Instructed Object-Level Geometric Transformation in Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.02356v2](https://arxiv.org/pdf/2601.02356v2)**

> **作者:** Jing Tan; Zhaoyang Zhang; Yantao Shen; Jiarui Cai; Shuo Yang; Jiajun Wu; Wei Xia; Zhuowen Tu; Stefano Soatto
>
> **备注:** Project page: https://sparkstj.github.io/talk2move
>
> **摘要:** We introduce Talk2Move, a reinforcement learning (RL) based diffusion framework for text-instructed spatial transformation of objects within scenes. Spatially manipulating objects in a scene through natural language poses a challenge for multimodal generation systems. While existing text-based manipulation methods can adjust appearance or style, they struggle to perform object-level geometric transformations-such as translating, rotating, or resizing objects-due to scarce paired supervision and pixel-level optimization limits. Talk2Move employs Group Relative Policy Optimization (GRPO) to explore geometric actions through diverse rollouts generated from input images and lightweight textual variations, removing the need for costly paired data. A spatial reward guided model aligns geometric transformations with linguistic description, while off-policy step evaluation and active step sampling improve learning efficiency by focusing on informative transformation stages. Furthermore, we design object-centric spatial rewards that evaluate displacement, rotation, and scaling behaviors directly, enabling interpretable and coherent transformations. Experiments on curated benchmarks demonstrate that Talk2Move achieves precise, consistent, and semantically faithful object transformations, outperforming existing text-guided editing approaches in both spatial accuracy and scene coherence.
>
---
#### [replaced 017] GCR: Geometry-Consistent Routing for Task-Agnostic Continual Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.01856v2](https://arxiv.org/pdf/2601.01856v2)**

> **作者:** Joongwon Chae; Lihui Luo; Yang Liu; Runming Wang; Dongmei Yu; Zeming Liang; Xi Yuan; Dayan Zhang; Zhenglin Chen; Peiwu Qin; Ilmoon Chae
>
> **摘要:** Feature-based anomaly detection is widely adopted in industrial inspection due to the strong representational power of large pre-trained vision encoders. While most existing methods focus on improving within-category anomaly scoring, practical deployments increasingly require task-agnostic operation under continual category expansion, where the category identity is unknown at test time. In this setting, overall performance is often dominated by expert selection, namely routing an input to an appropriate normality model before any head-specific scoring is applied. However, routing rules that compare head-specific anomaly scores across independently constructed heads are unreliable in practice, as score distributions can differ substantially across categories in scale and tail behavior. We propose GCR, a lightweight mixture-of-experts framework for stabilizing task-agnostic continual anomaly detection through geometry-consistent routing. GCR routes each test image directly in a shared frozen patch-embedding space by minimizing an accumulated nearest-prototype distance to category-specific prototype banks, and then computes anomaly maps only within the routed expert using a standard prototype-based scoring rule. By separating cross-head decision making from within-head anomaly scoring, GCR avoids cross-head score comparability issues without requiring end-to-end representation learning. Experiments on MVTec AD and VisA show that geometry-consistent routing substantially improves routing stability and mitigates continual performance collapse, achieving near-zero forgetting while maintaining competitive detection and localization performance. These results indicate that many failures previously attributed to representation forgetting can instead be explained by decision-rule instability in cross-head routing. Code is available at https://github.com/jw-chae/GCR
>
---
#### [replaced 018] BlurDM: A Blur Diffusion Model for Image Deblurring
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.03979v3](https://arxiv.org/pdf/2512.03979v3)**

> **作者:** Jin-Ting He; Fu-Jen Tsai; Yan-Tsung Peng; Min-Hung Chen; Chia-Wen Lin; Yen-Yu Lin
>
> **备注:** NeurIPS 2025. Project Page: https://jin-ting-he.github.io/BlurDM/
>
> **摘要:** Diffusion models show promise for dynamic scene deblurring; however, existing studies often fail to leverage the intrinsic nature of the blurring process within diffusion models, limiting their full potential. To address it, we present a Blur Diffusion Model (BlurDM), which seamlessly integrates the blur formation process into diffusion for image deblurring. Observing that motion blur stems from continuous exposure, BlurDM implicitly models the blur formation process through a dual-diffusion forward scheme, diffusing both noise and blur onto a sharp image. During the reverse generation process, we derive a dual denoising and deblurring formulation, enabling BlurDM to recover the sharp image by simultaneously denoising and deblurring, given pure Gaussian noise conditioned on the blurred image as input. Additionally, to efficiently integrate BlurDM into deblurring networks, we perform BlurDM in the latent space, forming a flexible prior generation network for deblurring. Extensive experiments demonstrate that BlurDM significantly and consistently enhances existing deblurring methods on four benchmark datasets. The project page is available at https://jin-ting-he.github.io/BlurDM/.
>
---
#### [replaced 019] Forging Spatial Intelligence: A Roadmap of Multi-Modal Data Pre-Training for Autonomous Systems
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多模态预训练任务，旨在解决自动驾驶系统中空间智能构建问题。通过分析传感器数据与学习策略，提出统一的预训练框架，提升3D目标检测等能力。**

- **链接: [https://arxiv.org/pdf/2512.24385v2](https://arxiv.org/pdf/2512.24385v2)**

> **作者:** Song Wang; Lingdong Kong; Xiaolu Liu; Hao Shi; Wentong Li; Jianke Zhu; Steven C. H. Hoi
>
> **备注:** Survey; 40 pages, 7 figures, 9 tables; GitHub Repo at https://github.com/worldbench/awesome-spatial-intelligence
>
> **摘要:** The rapid advancement of autonomous systems, including self-driving vehicles and drones, has intensified the need to forge true Spatial Intelligence from multi-modal onboard sensor data. While foundation models excel in single-modal contexts, integrating their capabilities across diverse sensors like cameras and LiDAR to create a unified understanding remains a formidable challenge. This paper presents a comprehensive framework for multi-modal pre-training, identifying the core set of techniques driving progress toward this goal. We dissect the interplay between foundational sensor characteristics and learning strategies, evaluating the role of platform-specific datasets in enabling these advancements. Our central contribution is the formulation of a unified taxonomy for pre-training paradigms: ranging from single-modality baselines to sophisticated unified frameworks that learn holistic representations for advanced tasks like 3D object detection and semantic occupancy prediction. Furthermore, we investigate the integration of textual inputs and occupancy representations to facilitate open-world perception and planning. Finally, we identify critical bottlenecks, such as computational efficiency and model scalability, and propose a roadmap toward general-purpose multi-modal foundation models capable of achieving robust Spatial Intelligence for real-world deployment.
>
---
#### [replaced 020] Clinically-Validated Innovative Mobile Application for Assessing Blinking and Eyelid Movements
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.14361v2](https://arxiv.org/pdf/2511.14361v2)**

> **作者:** Gustavo Adolpho Bonesso; Carlos Marcelo Gurjão de Godoy; Tammy Hentona Osaki; Midori Hentona Osaki; Bárbara Moreira Ribeiro Trindade dos Santos; Juliana Yuka Washiya; Regina Célia Coelho
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** Blinking is a vital physiological process that protects and maintains the health of the ocular surface. Objective assessment of eyelid movements remains challenging due to the complexity, cost, and limited clinical applicability of existing tools. This study presents the Bapp (Blink Application), a mobile application developed using the Flutter framework and integrated with Google ML Kit for on-device, real-time analysis of eyelid movements, and its clinical validation. The validation was performed using 45 videos from patients, whose blinks were manually annotated by an ophthalmology specialist as the ground truth. The Bapp's performance was evaluated using standard metrics, with results demonstrating 98.4% precision, 96.9% recall, and an overall accuracy of 98.3%. These outcomes confirm the reliability of the Bapp as a portable, accessible, and objective tool for monitoring eyelid movements. The application offers a promising alternative to traditional manual blink counting, supporting continuous ocular health monitoring and postoperative evaluation in clinical environments.
>
---
#### [replaced 021] QUIET-SR: Quantum Image Enhancement Transformer for Single Image Super-Resolution
- **分类: quant-ph; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2503.08759v2](https://arxiv.org/pdf/2503.08759v2)**

> **作者:** Siddhant Dutta; Nouhaila Innan; Khadijeh Najafi; Sadok Ben Yahia; Muhammad Shafique
>
> **备注:** 13 Pages, 7 Figures (5 Main figures, 2 Sub-figures), 2 Tables, Under Review
>
> **摘要:** Recent advancements in Single-Image Super-Resolution (SISR) using deep learning have significantly improved image restoration quality. However, the high computational cost of processing high-resolution images due to the large number of parameters in classical models, along with the scalability challenges of quantum algorithms for image processing, remains a major obstacle. In this paper, we propose the Quantum Image Enhancement Transformer for Super-Resolution (QUIET-SR), a hybrid framework that extends the Swin transformer architecture with a novel shifted quantum window attention mechanism, built upon variational quantum neural networks. QUIET-SR effectively captures complex residual mappings between low-resolution and high-resolution images, leveraging quantum attention mechanisms to enhance feature extraction and image restoration while requiring a minimal number of qubits, making it suitable for the Noisy Intermediate-Scale Quantum (NISQ) era. We evaluate our framework in MNIST (30.24 PSNR, 0.989 SSIM), FashionMNIST (29.76 PSNR, 0.976 SSIM) and the MedMNIST dataset collection, demonstrating that QUIET-SR achieves PSNR and SSIM scores comparable to state-of-the-art methods while using fewer parameters. Our efficient batching strategy directly enables massive parallelization on multiple QPU's paving the way for practical quantum-enhanced image super-resolution through coordinated QPU-GPU quantum supercomputing.
>
---
#### [replaced 022] MVT: Mask-Grounded Vision-Language Models for Taxonomy-Aligned Land-Cover Tagging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.18693v3](https://arxiv.org/pdf/2509.18693v3)**

> **作者:** Siyi Chen; Kai Wang; Weicong Pang; Ruiming Yang; Ziru Chen; Renjun Gao; Alexis Kai Hon Lau; Dasa Gu; Chenchen Zhang; Cheng Li
>
> **备注:** The project is available at https://charlescsyyy.github.io/MVT
>
> **摘要:** Land-cover understanding in remote sensing increasingly demands class-agnostic systems that generalize across datasets while remaining spatially precise and interpretable. We study a geometry-first discovery-and-interpretation setting under domain shift, where candidate regions are delineated class-agnostically and supervision avoids lexical class names via anonymized identifiers. Complementary to open-set recognition and open-world learning, we focus on coupling class-agnostic mask evidence with taxonomy-grounded scene interpretation, rather than unknown rejection or continual class expansion. We propose MVT, a three-stage framework that (i) extracts boundary-faithful region masks using SAM2 with domain adaptation, (ii) performs mask-grounded semantic tagging and scene description generation via dual-step LoRA fine-tuning of multimodal LLMs, and (iii) evaluates outputs with LLM-as-judge scoring calibrated by stratified expert ratings. On cross-dataset segmentation transfer (train on OpenEarthMap, evaluate on LoveDA), domain-adapted SAM2 improves mask quality; meanwhile, dual-step MLLM fine-tuning yields more accurate taxonomy-aligned tags and more informative mask-grounded scene descriptions.
>
---
#### [replaced 023] Is Contrastive Distillation Enough for Learning Comprehensive 3D Representations?
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2412.08973v3](https://arxiv.org/pdf/2412.08973v3)**

> **作者:** Yifan Zhang; Junhui Hou
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** Cross-modal contrastive distillation has recently been explored for learning effective 3D representations. However, existing methods focus primarily on modality-shared features, neglecting the modality-specific features during the pre-training process, which leads to suboptimal representations. In this paper, we theoretically analyze the limitations of current contrastive methods for 3D representation learning and propose a new framework, namely CMCR (Cross-Modal Comprehensive Representation Learning), to address these shortcomings. Our approach improves upon traditional methods by better integrating both modality-shared and modality-specific features. Specifically, we introduce masked image modeling and occupancy estimation tasks to guide the network in learning more comprehensive modality-specific features. Furthermore, we propose a novel multi-modal unified codebook that learns an embedding space shared across different modalities. Besides, we introduce geometry-enhanced masked image modeling to further boost 3D representation learning. Extensive experiments demonstrate that our method mitigates the challenges faced by traditional approaches and consistently outperforms existing image-to-LiDAR contrastive distillation methods in downstream tasks. Code will be available at https://github.com/Eaphan/CMCR.
>
---
#### [replaced 024] Agentic Retoucher for Text-To-Image Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.02046v2](https://arxiv.org/pdf/2601.02046v2)**

> **作者:** Shaocheng Shen; Jianfeng Liang; Chunlei Cai; Cong Geng; Huiyu Duan; Xiaoyun Zhang; Qiang Hu; Guangtao Zhai
>
> **摘要:** Text-to-image (T2I) diffusion models such as SDXL and FLUX have achieved impressive photorealism, yet small-scale distortions remain pervasive in limbs, face, text and so on. Existing refinement approaches either perform costly iterative re-generation or rely on vision-language models (VLMs) with weak spatial grounding, leading to semantic drift and unreliable local edits. To close this gap, we propose Agentic Retoucher, a hierarchical decision-driven framework that reformulates post-generation correction as a human-like perception-reasoning-action loop. Specifically, we design (1) a perception agent that learns contextual saliency for fine-grained distortion localization under text-image consistency cues, (2) a reasoning agent that performs human-aligned inferential diagnosis via progressive preference alignment, and (3) an action agent that adaptively plans localized inpainting guided by user preference. This design integrates perceptual evidence, linguistic reasoning, and controllable correction into a unified, self-corrective decision process. To enable fine-grained supervision and quantitative evaluation, we further construct GenBlemish-27K, a dataset of 6K T2I images with 27K annotated artifact regions across 12 categories. Extensive experiments demonstrate that Agentic Retoucher consistently outperforms state-of-the-art methods in perceptual quality, distortion localization and human preference alignment, establishing a new paradigm for self-corrective and perceptually reliable T2I generation.
>
---
#### [replaced 025] WeatherDiffusion: Controllable Weather Editing in Intrinsic Space
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.06982v5](https://arxiv.org/pdf/2508.06982v5)**

> **作者:** Yixin Zhu; Zuoliang Zhu; Jian Yang; Miloš Hašan; Jin Xie; Beibei Wang
>
> **摘要:** We present WeatherDiffusion, a diffusion-based framework for controllable weather editing in intrinsic space. Our framework includes two components based on diffusion priors: an inverse renderer that estimates material properties, scene geometry, and lighting as intrinsic maps from an input image, and a forward renderer that utilizes these geometry and material maps along with a text prompt that describes specific weather conditions to generate a final image. The intrinsic maps enhance controllability compared to traditional pixel-space editing approaches. We propose an intrinsic map-aware attention mechanism that improves spatial correspondence and decomposition quality in large outdoor scenes. For forward rendering, we leverage CLIP-space interpolation of weather prompts to achieve fine-grained weather control. We also introduce a synthetic and a real-world dataset, containing 38k and 18k images under various weather conditions, each with intrinsic map annotations. WeatherDiffusion outperforms state-of-the-art pixel-space editing approaches, weather restoration methods, and rendering-based methods, showing promise for downstream tasks such as autonomous driving, enhancing the robustness of detection and segmentation in challenging weather scenarios.
>
---
#### [replaced 026] Automated Invoice Data Extraction: Using LLM and OCR
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.05547v2](https://arxiv.org/pdf/2511.05547v2)**

> **作者:** Khushi Khanchandani; Advait Thakur; Akshita Shetty; Chaitravi Reddy; Ritisa Behera
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Conventional Optical Character Recognition (OCR) systems are challenged by variant invoice layouts, handwritten text, and low-quality scans, which are often caused by strong template dependencies that restrict their flexibility across different document structures and layouts. Newer solutions utilize advanced deep learning models such as Convolutional Neural Networks (CNN) as well as Transformers, and domain-specific models for better layout analysis and accuracy across various sections over varied document types. Large Language Models (LLMs) have revolutionized extraction pipelines at their core with sophisticated entity recognition and semantic comprehension to support complex contextual relationship mapping without direct programming specification. Visual Named Entity Recognition (NER) capabilities permit extraction from invoice images with greater contextual sensitivity and much higher accuracy rates than older approaches. Existing industry best practices utilize hybrid architectures that blend OCR technology and LLM for maximum scalability and minimal human intervention. This work introduces a holistic Artificial Intelligence (AI) platform combining OCR, deep learning, LLMs, and graph analytics to achieve unprecedented extraction quality and consistency.
>
---
#### [replaced 027] Full segmentation annotations of 3D time-lapse microscopy images of MDA231 cells
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.10797v2](https://arxiv.org/pdf/2510.10797v2)**

> **作者:** Aleksandra Melnikova; Petr Matula
>
> **备注:** 6 pages, 2 figures, 4 tables
>
> **摘要:** High-quality, publicly available segmentation annotations of image and video datasets are critical for advancing the field of image processing. In particular, annotations of volumetric images of a large number of targets are time-consuming and challenging. In (Melnikova, A., & Matula, P., 2025), we presented the first publicly available full 3D time-lapse segmentation annotations of migrating cells with complex dynamic shapes. Concretely, three distinct humans annotated two sequences of MDA231 human breast carcinoma cells (Fluo-C3DL-MDA231) from the Cell Tracking Challenge (CTC). This paper aims to provide a comprehensive description of the dataset and accompanying experiments that were not included in (Melnikova, A., & Matula, P., 2025) due to limitations in publication space. Namely, we show that the created annotations are consistent with the previously published tracking markers provided by the CTC organizers and the segmentation accuracy measured based on the 2D gold truth of CTC is within the inter-annotator variability margins. We compared the created 3D annotations with automatically created silver truth provided by CTC. We have found the proposed annotations better represent the complexity of the input images. The presented annotations can be used for testing and training cell segmentation, or analyzing 3D shapes of highly dynamic objects.
>
---
#### [replaced 028] Minimal Clips, Maximum Salience: Long Video Summarization via Key Moment Extraction
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于长视频摘要任务，旨在解决关键视觉信息丢失问题。通过提取关键片段并利用轻量模型生成描述，提升摘要质量与效率。**

- **链接: [https://arxiv.org/pdf/2512.11399v2](https://arxiv.org/pdf/2512.11399v2)**

> **作者:** Galann Pennec; Zhengyuan Liu; Nicholas Asher; Philippe Muller; Nancy F. Chen
>
> **摘要:** Vision-Language Models (VLMs) are able to process increasingly longer videos. Yet, important visual information is easily lost throughout the entire context and missed by VLMs. Also, it is important to design tools that enable cost-effective analysis of lengthy video content. In this paper, we propose a clip selection method that targets key video moments to be included in a multimodal summary. We divide the video into short clips and generate compact visual descriptions of each using a lightweight video captioning model. These are then passed to a large language model (LLM), which selects the K clips containing the most relevant visual information for a multimodal summary. We evaluate our approach on reference clips for the task, automatically derived from full human-annotated screenplays and summaries in the MovieSum dataset. We further show that these reference clips (less than 6% of the movie) are sufficient to build a complete multimodal summary of the movies in MovieSum. Using our clip selection method, we achieve a summarization performance close to that of these reference clips while capturing substantially more relevant video information than random clip selection. Importantly, we maintain low computational cost by relying on a lightweight captioning model.
>
---
#### [replaced 029] Name That Part: 3D Part Segmentation and Naming
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.18003v2](https://arxiv.org/pdf/2512.18003v2)**

> **作者:** Soumava Paul; Prakhar Kaushik; Ankit Vaidya; Anand Bhattad; Alan Yuille
>
> **备注:** Project page at https://name-that-part.github.io
>
> **摘要:** We address semantic 3D part segmentation: decomposing objects into parts with meaningful names. While datasets exist with part annotations, their definitions are inconsistent across datasets, limiting robust training. Previous methods produce unlabeled decompositions or retrieve single parts without complete shape annotations. We propose ALIGN-Parts, which formulates part naming as a direct set alignment task. Our method decomposes shapes into partlets - implicit 3D part representations - matched to part descriptions via bipartite assignment. We combine geometric cues from 3D part fields, appearance cues from multi-view vision features, and semantic knowledge from language-model-generated affordance descriptions. Text-alignment loss ensures partlets share embedding space with text, enabling a theoretically open-vocabulary matching setup, given sufficient data. Our efficient and novel, one-shot, 3D part segmentation and naming method finds applications in several downstream tasks, including serving as a scalable annotation engine. As our model supports zero-shot matching to arbitrary descriptions and confidence-calibrated predictions for known categories, with human verification, we create a unified ontology that aligns PartNet, 3DCoMPaT++, and Find3D, consisting of 1,794 unique 3D parts. We introduce two novel metrics appropriate for the named 3D part segmentation task. We also show examples from our newly created TexParts dataset.
>
---
#### [replaced 030] GeoReason: Aligning Thinking And Answering In Remote Sensing Vision-Language Models Via Logical Consistency Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.04118v2](https://arxiv.org/pdf/2601.04118v2)**

> **作者:** Wenshuai Li; Xiantai Xiang; Zixiao Wen; Guangyao Zhou; Ben Niu; Feng Wang; Lijia Huang; Qiantong Wang; Yuxin Hu
>
> **摘要:** The evolution of Remote Sensing Vision-Language Models(RS-VLMs) emphasizes the importance of transitioning from perception-centric recognition toward high-level deductive reasoning to enhance cognitive reliability in complex spatial tasks. However, current models often suffer from logical hallucinations, where correct answers are derived from flawed reasoning chains or rely on positional shortcuts rather than spatial logic. This decoupling undermines reliability in strategic spatial decision-making. To address this, we present GeoReason, a framework designed to synchronize internal thinking with final decisions. We first construct GeoReason-Bench, a logic-driven dataset containing 4,000 reasoning trajectories synthesized from geometric primitives and expert knowledge. We then formulate a two-stage training strategy: (1) Supervised Knowledge Initialization to equip the model with reasoning syntax and domain expertise, and (2) Consistency-Aware Reinforcement Learning to refine deductive reliability. This second stage integrates a novel Logical Consistency Reward, which penalizes logical drift via an option permutation strategy to anchor decisions in verifiable reasoning traces. Experimental results demonstrate that our framework significantly enhances the cognitive reliability and interpretability of RS-VLMs, achieving state-of-the-art performance compared to other advanced methods.
>
---
#### [replaced 031] SynDroneVision: A Synthetic Dataset for Image-Based Drone Detection
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SynDroneVision，一个用于RGB无人机检测的合成数据集，解决真实数据不足和采集成本高的问题。通过仿真生成多样数据，提升检测模型性能。**

- **链接: [https://arxiv.org/pdf/2411.05633v2](https://arxiv.org/pdf/2411.05633v2)**

> **作者:** Tamara R. Lenhard; Andreas Weinmann; Kai Franke; Tobias Koch
>
> **摘要:** Developing robust drone detection systems is often constrained by the limited availability of large-scale annotated training data and the high costs associated with real-world data collection. However, leveraging synthetic data generated via game engine-based simulations provides a promising and cost-effective solution to overcome this issue. Therefore, we present SynDroneVision, a synthetic dataset specifically designed for RGB-based drone detection in surveillance applications. Featuring diverse backgrounds, lighting conditions, and drone models, SynDroneVision offers a comprehensive training foundation for deep learning algorithms. To evaluate the dataset's effectiveness, we perform a comparative analysis across a selection of recent YOLO detection models. Our findings demonstrate that SynDroneVision is a valuable resource for real-world data enrichment, achieving notable enhancements in model performance and robustness, while significantly reducing the time and costs of real-world data acquisition. SynDroneVision will be publicly released upon paper acceptance.
>
---
#### [replaced 032] Leveraging Clinical Text and Class Conditioning for 3D Prostate MRI Generation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.10230v3](https://arxiv.org/pdf/2506.10230v3)**

> **作者:** Emerson P. Grabke; Babak Taati; Masoom A. Haider
>
> **备注:** Accepted for publication in IEEE Transactions on Biomedical Engineering, 2025. This is the accepted author version. The final published version is available at https://doi.org/10.1109/TBME.2025.3648426
>
> **摘要:** Objective: Latent diffusion models (LDM) could alleviate data scarcity challenges affecting machine learning development for medical imaging. However, medical LDM strategies typically rely on short-prompt text encoders, nonmedical LDMs, or large data volumes. These strategies can limit performance and scientific accessibility. We propose a novel LDM conditioning approach to address these limitations. Methods: We propose Class-Conditioned Efficient Large Language model Adapter (CCELLA), a novel dual-head conditioning approach that simultaneously conditions the LDM U-Net with free-text clinical reports and radiology classification. We also propose a data-efficient LDM pipeline centered around CCELLA and a proposed joint loss function. We first evaluate our method on 3D prostate MRI against state-of-the-art. We then augment a downstream classifier model training dataset with synthetic images from our method. Results: Our method achieves a 3D FID score of 0.025 on a size-limited 3D prostate MRI dataset, significantly outperforming a recent foundation model with FID 0.070. When training a classifier for prostate cancer prediction, adding synthetic images generated by our method during training improves classifier accuracy from 69% to 74% and outperforms classifiers trained on images generated by prior state-of-the-art. Classifier training solely on our method's synthetic images achieved comparable performance to real image training. Conclusion: We show that our method improved both synthetic image quality and downstream classifier performance using limited data and minimal human annotation. Significance: The proposed CCELLA-centric pipeline enables radiology report and class-conditioned LDM training for high-quality medical image synthesis given limited data volume and human data annotation, improving LDM performance and scientific accessibility.
>
---
#### [replaced 033] Spontaneous emergence of linguistic statistical laws in images via artificial neural networks
- **分类: cs.CV; physics.comp-ph**

- **链接: [https://arxiv.org/pdf/2501.18620v2](https://arxiv.org/pdf/2501.18620v2)**

> **作者:** Ping-Rui Tsai; Chi-hsiang Wang; Yu-Cheng Liao; Hong-Yue Huang; Tzay-Ming Hong
>
> **备注:** 10 figures
>
> **摘要:** As a core element of culture, images transform perception into structured representations and undergo evolution similar to natural languages. Given that visual input accounts for 60% of human sensory experience, it is natural to ask whether images follow statistical regularities similar to those in linguistic systems. Guided by symbol-grounding theory, which posits that meaningful symbols originate from perception, we treat images as vision-centric artifacts and employ pre-trained neural networks to model visual processing. By detecting kernel activations and extracting pixels, we obtain text-like units, which reveal that these image-derived representations adhere to statistical laws such as Zipf's, Heaps', and Benford's laws, analogous to linguistic data. Notably, these statistical regularities emerge spontaneously, without the need for explicit symbols or hybrid architectures. Our results indicate that connectionist networks can automatically develop structured, quasi-symbolic units through perceptual processing alone, suggesting that text- and symbol-like properties can naturally emerge from neural networks and providing a novel perspective for interpretation.
>
---
#### [replaced 034] Controllable Generation with Text-to-Image Diffusion Models: A Survey
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2403.04279v2](https://arxiv.org/pdf/2403.04279v2)**

> **作者:** Pu Cao; Feng Zhou; Qing Song; Lu Yang
>
> **备注:** TPAMI 2025; A collection of resources on controllable generation with text-to-image diffusion models: https://github.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models
>
> **摘要:** In the rapidly advancing realm of visual generation, diffusion models have revolutionized the landscape, marking a significant shift in capabilities with their impressive text-guided generative functions. However, relying solely on text for conditioning these models does not fully cater to the varied and complex requirements of different applications and scenarios. Acknowledging this shortfall, a variety of studies aim to control pre-trained text-to-image (T2I) models to support novel conditions. In this survey, we undertake a thorough review of the literature on controllable generation with T2I diffusion models, covering both the theoretical foundations and practical advancements in this domain. Our review begins with a brief introduction to the basics of denoising diffusion probabilistic models (DDPMs) and widely used T2I diffusion models. We then reveal the controlling mechanisms of diffusion models, theoretically analyzing how novel conditions are introduced into the denoising process for conditional generation. Additionally, we offer a detailed overview of research in this area, organizing it into distinct categories from the condition perspective: generation with specific conditions, generation with multiple conditions, and universal controllable generation. For an exhaustive list of the controllable generation literature surveyed, please refer to our curated repository at https://github.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models.
>
---
#### [replaced 035] Robust Scene Coordinate Regression via Geometrically-Consistent Global Descriptors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17226v2](https://arxiv.org/pdf/2512.17226v2)**

> **作者:** Son Tung Nguyen; Alejandro Fontan; Michael Milford; Tobias Fischer
>
> **备注:** Accepted at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Recent learning-based visual localization methods use global descriptors to disambiguate visually similar places, but existing approaches often derive these descriptors from geometric cues alone (e.g., covisibility graphs), limiting their discriminative power and reducing robustness in the presence of noisy geometric constraints. We propose an aggregator module that learns global descriptors consistent with both geometrical structure and visual similarity, ensuring that images are close in descriptor space only when they are visually similar and spatially connected. This corrects erroneous associations caused by unreliable overlap scores. Using a batch-mining strategy based solely on the overlap scores and a modified contrastive loss, our method trains without manual place labels and generalizes across diverse environments. Experiments on challenging benchmarks show substantial localization gains in large-scale environments while preserving computational and memory efficiency. Code is available at https://github.com/sontung/robust_scr.
>
---
#### [replaced 036] Boosting HDR Image Reconstruction via Semantic Knowledge Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.15361v2](https://arxiv.org/pdf/2503.15361v2)**

> **作者:** Tao Hu; Longyao Wu; Wei Dong; Peng Wu; Jinqiu Sun; Xiaogang Xu; Qingsen Yan; Yanning Zhang
>
> **摘要:** Recovering High Dynamic Range (HDR) images from multiple Standard Dynamic Range (SDR) images become challenging when the SDR images exhibit noticeable degradation and missing content. Leveraging scene-specific semantic priors offers a promising solution for restoring heavily degraded regions. However, these priors are typically extracted from sRGB SDR images, the domain/format gap poses a significant challenge when applying it to HDR imaging. To address this issue, we propose a general framework that transfers semantic knowledge derived from SDR domain via self-distillation to boost existing HDR reconstruction. Specifically, the proposed framework first introduces the Semantic Priors Guided Reconstruction Model (SPGRM), which leverages SDR image semantic knowledge to address ill-posed problems in the initial HDR reconstruction results. Subsequently, we leverage a self-distillation mechanism that constrains the color and content information with semantic knowledge, aligning the external outputs between the baseline and SPGRM. Furthermore, to transfer the semantic knowledge of the internal features, we utilize a Semantic Knowledge Alignment Module (SKAM) to fill the missing semantic contents with the complementary masks. Extensive experiments demonstrate that our framework significantly boosts HDR imaging quality for existing methods without altering the network architecture.
>
---
#### [replaced 037] BOP-Distrib: Revisiting 6D Pose Estimation Benchmarks for Better Evaluation under Visual Ambiguities
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.17297v4](https://arxiv.org/pdf/2408.17297v4)**

> **作者:** Boris Meden; Asma Brazi; Fabrice Mayran de Chamisso; Steve Bourgeois; Vincent Lepetit
>
> **摘要:** 6D pose estimation aims at determining the object pose that best explains the camera observation. The unique solution for non-ambiguous objects can turn into a multi-modal pose distribution for symmetrical objects or when occlusions of symmetry-breaking elements happen, depending on the viewpoint. Currently, 6D pose estimation methods are benchmarked on datasets that consider, for their ground truth annotations, visual ambiguities as only related to global object symmetries, whereas they should be defined per-image to account for the camera viewpoint. We thus first propose an automatic method to re-annotate those datasets with a 6D pose distribution specific to each image, taking into account the object surface visibility in the image to correctly determine the visual ambiguities. Second, given this improved ground truth, we re-evaluate the state-of-the-art single pose methods and show that this greatly modifies the ranking of these methods. Third, as some recent works focus on estimating the complete set of solutions, we derive a precision/recall formulation to evaluate them against our image-wise distribution ground truth, making it the first benchmark for pose distribution methods on real images.
>
---
#### [replaced 038] OneVision: An End-to-End Generative Framework for Multi-view E-commerce Vision Search
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.05759v4](https://arxiv.org/pdf/2510.05759v4)**

> **作者:** Zexin Zheng; Huangyu Dai; Lingtao Mao; Xinyu Sun; Zihan Liang; Ben Chen; Yuqing Ding; Chenyi Lei; Wenwu Ou; Han Li; Kun Gai
>
> **摘要:** Traditional vision search, similar to search and recommendation systems, follows the multi-stage cascading architecture (MCA) paradigm to balance efficiency and conversion. Specifically, the query image undergoes feature extraction, recall, pre-ranking, and ranking stages, ultimately presenting the user with semantically similar products that meet their preferences. This multi-view representation discrepancy of the same object in the query and the optimization objective collide across these stages, making it difficult to achieve Pareto optimality in both user experience and conversion. In this paper, an end-to-end generative framework, OneVision, is proposed to address these problems. OneVision builds on VRQ, a vision-aligned residual quantization encoding, which can align the vastly different representations of an object across multiple viewpoints while preserving the distinctive features of each product as much as possible. Then a multi-stage semantic alignment scheme is adopted to maintain strong visual similarity priors while effectively incorporating user-specific information for personalized preference generation. In offline evaluations, OneVision performs on par with online MCA, while improving inference efficiency by 21% through dynamic pruning. In A/B tests, it achieves significant online improvements: +2.15% item CTR, +2.27% CVR, and +3.12% order volume. These results demonstrate that a semantic ID centric, generative architecture can unify retrieval and personalization while simplifying the serving pathway.
>
---
#### [replaced 039] POLYCHARTQA: Benchmarking Large Vision-Language Models with Multilingual Chart Question Answering
- **分类: cs.CL; cs.AI; cs.CV; cs.MM**

- **简介: 该论文属于多语言图表问答任务，旨在解决现有基准英语主导的问题。通过构建多语言图表问答基准PolyChartQA，提升模型在全球语言环境下的图表理解能力。**

- **链接: [https://arxiv.org/pdf/2507.11939v2](https://arxiv.org/pdf/2507.11939v2)**

> **作者:** Yichen Xu; Liangyu Chen; Liang Zhang; Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in Progress
>
> **摘要:** Charts are a universally adopted medium for data communication, yet existing chart understanding benchmarks are overwhelmingly English-centric, limiting their accessibility and relevance to global audiences. To address this limitation, we introduce PolyChartQA, the first large-scale multilingual benchmark for chart question answering, comprising 22,606 charts and 26,151 QA pairs across 10 diverse languages. PolyChartQA is constructed through a scalable pipeline that enables efficient multilingual chart generation via data translation and code reuse, supported by LLM-based translation and rigorous quality control. We systematically evaluate multilingual chart understanding with PolyChartQA on state-of-the-art LVLMs and reveal a significant performance gap between English and other languages, particularly low-resource ones. Additionally, we introduce a companion multilingual chart question answering training set, PolyChartQA-Train, on which fine-tuning LVLMs yields substantial gains in multilingual chart understanding across diverse model sizes and architectures. Together, our benchmark provides a foundation for developing globally inclusive vision-language models capable of understanding charts across diverse linguistic contexts.
>
---
#### [replaced 040] FluencyVE: Marrying Temporal-Aware Mamba with Bypass Attention for Video Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.21015v2](https://arxiv.org/pdf/2512.21015v2)**

> **作者:** Mingshu Cai; Yixuan Li; Osamu Yoshie; Yuya Ieiri
>
> **备注:** Accepted by IEEE Transactions on Multimedia (TMM)
>
> **摘要:** Large-scale text-to-image diffusion models have achieved unprecedented success in image generation and editing. However, extending this success to video editing remains challenging. Recent video editing efforts have adapted pretrained text-to-image models by adding temporal attention mechanisms to handle video tasks. Unfortunately, these methods continue to suffer from temporal inconsistency issues and high computational overheads. In this study, we propose FluencyVE, which is a simple yet effective one-shot video editing approach. FluencyVE integrates the linear time-series module, Mamba, into a video editing model based on pretrained Stable Diffusion models, replacing the temporal attention layer. This enables global frame-level attention while reducing the computational costs. In addition, we employ low-rank approximation matrices to replace the query and key weight matrices in the causal attention, and use a weighted averaging technique during training to update the attention scores. This approach significantly preserves the generative power of the text-to-image model while effectively reducing the computational burden. Experiments and analyses demonstrate promising results in editing various attributes, subjects, and locations in real-world videos.
>
---
#### [replaced 041] Two-Stream Thermal Imaging Fusion for Enhanced Time of Birth Detection in Neonatal Care
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.03244v2](https://arxiv.org/pdf/2503.03244v2)**

> **作者:** Jorge García-Torres; Øyvind Meinich-Bache; Sara Brunner; Siren Rettedal; Vilde Kolstad; Kjersti Engan
>
> **备注:** This work has been accepted at IEEE 25th International Conference on Digital Signal Processing
>
> **摘要:** Around 10% of newborns require some help to initiate breathing, and 5\% need ventilation assistance. Accurate Time of Birth (ToB) documentation is essential for optimizing neonatal care, as timely interventions are vital for proper resuscitation. However, current clinical methods for recording ToB often rely on manual processes, which can be prone to inaccuracies. In this study, we present a novel two-stream fusion system that combines the power of image and video analysis to accurately detect the ToB from thermal recordings in the delivery room and operating theater. By integrating static and dynamic streams, our approach captures richer birth-related spatiotemporal features, leading to more robust and precise ToB estimation. We demonstrate that this synergy between data modalities enhances performance over single-stream approaches. Our system achieves 95.7% precision and 84.8% recall in detecting birth within short video clips. Additionally, with the help of a score aggregation module, it successfully identifies ToB in 100% of test cases, with a median absolute error of 2 seconds and an absolute mean deviation of 4.5 seconds compared to manual annotations.
>
---
#### [replaced 042] Improving VisNet for Object Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08897v3](https://arxiv.org/pdf/2511.08897v3)**

> **作者:** Mehdi Fatan Serj; C. Alejandro Parraga; Xavier Otazu
>
> **摘要:** Object recognition plays a fundamental role in how biological organisms perceive and interact with their environment. While the human visual system performs this task with remarkable efficiency, reproducing similar capabilities in artificial systems remains challenging. This study investigates VisNet, a biologically inspired neural network model, and several enhanced variants incorporating radial basis function neurons, Mahalanobis distance based learning, and retinal like preprocessing for both general object recognition and symmetry classification. By leveraging principles of Hebbian learning and temporal continuity associating temporally adjacent views to build invariant representations. VisNet and its extensions capture robust and transformation invariant features. Experimental results across multiple datasets, including MNIST, CIFAR10, and custom symmetric object sets, show that these enhanced VisNet variants substantially improve recognition accuracy compared with the baseline model. These findings underscore the adaptability and biological relevance of VisNet inspired architectures, offering a powerful and interpretable framework for visual recognition in both neuroscience and artificial intelligence. Keywords: VisNet, Object Recognition, Symmetry Detection, Hebbian Learning, RBF Neurons, Mahalanobis Distance, Biologically Inspired Models, Invariant Representations
>
---
#### [replaced 043] Extended OpenTT Games Dataset: A table tennis dataset for fine-grained shot type and point outcome
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19327v2](https://arxiv.org/pdf/2512.19327v2)**

> **作者:** Moamal Fadhil Abdul-Mahdi; Jonas Bruun Hubrechts; Thomas Martini Jørgensen; Emil Hovad
>
> **备注:** Thomas Martini Jørgensen and Emil Hovad contributed equally and share last authorship
>
> **摘要:** Automatically detecting and classifying strokes in table tennis video can streamline training workflows, enrich broadcast overlays, and enable fine-grained performance analytics. For this to be possible, annotated video data of table tennis is needed. We extend the public OpenTTGames dataset with highly detailed, frame-accurate shot type annotations (forehand, backhand with subtypes), player posture labels (body lean and leg stance), and rally outcome tags at point end. OpenTTGames is a set of recordings from the side of the table with official labels for bounces, when the ball is above the net, or hitting the net. The dataset already contains ball coordinates near events, which are either "bounce", "net", or "empty_event" in the original OpenTTGames dataset, and semantic masks (humans, table, scoreboard). Our extension adds the types of stroke to the events and a per-player taxonomy so models can move beyond event spotting toward tactical understanding (e.g., whether a stroke is likely to win the point or set up an advantage). We provide a compact coding scheme and code-assisted labeling procedure to support reproducible annotations and baselines for fine-grained stroke understanding in racket sports. This fills a practical gap in the community, where many prior video resources are either not publicly released or carry restrictive/unclear licenses that hinder reuse and benchmarking. Our annotations are released under the same CC BY-NC-SA 4.0 license as OpenTTGames, allowing free non-commercial use, modification, and redistribution, with appropriate attribution.
>
---
#### [replaced 044] FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.19850v3](https://arxiv.org/pdf/2503.19850v3)**

> **作者:** Carlos Plou; Cesar Borja; Ruben Martinez-Cantin; Ana C. Murillo
>
> **摘要:** Finding information in hour-long videos is a challenging task even for top-performing Vision Language Models (VLMs), as encoding visual content quickly exceeds available context windows. To tackle this challenge, we present FALCONEye, a novel video agent based on a training-free, model-agnostic meta-architecture composed of a VLM and a Large Language Model (LLM). FALCONEye answers open-ended questions using an exploration-based search algorithm guided by calibrated confidence from the VLM's answers. We also introduce the FALCON-Bench benchmark, extending Question Answering problem to Video Answer Search-requiring models to return both the answer and its supporting temporal window for open-ended questions in hour-long videos. With just a 7B VLM and a lightweight LLM, FALCONEye outscores all open-source 7B VLMs and comparable agents in FALCON-Bench. It further demonstrates its generalization capability in MLVU benchmark with shorter videos and different tasks, surpassing GPT-4o on single-detail tasks while slashing inference cost by roughly an order of magnitude.
>
---
#### [replaced 045] Simulation of prosthetic vision with PRIMA system and enhancement of face representation
- **分类: cs.HC; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.11677v3](https://arxiv.org/pdf/2503.11677v3)**

> **作者:** Anna Kochnev Goldstein; Jungyeon Park; Yueming Zhuo; Nathan Jensen; Daniel Palanker
>
> **摘要:** Objective. Patients implanted with the PRIMA photovoltaic subretinal prosthesis in geographic atrophy report form vision with the average acuity matching the 100um pixel size. Although this remarkable outcome enables them to read and write, they report difficulty with perceiving faces. Despite the pixelated stimulation, patients see smooth patterns rather than dots. We present a novel, non-pixelated algorithm for simulating prosthetic vision, compare its predictions to clinical outcomes, and describe computer vision and machine learning (ML) methods to improve face representation. Approach. Our simulation algorithm (ProViSim) integrates a spatial resolution filter based on sampling density limited by the pixel pitch and a contrast filter representing reduced contrast sensitivity of prosthetic vision. Patterns of Landolt C and human faces created using this simulator are compared to reports from actual PRIMA users. To recover the facial features lost in prosthetic vision due to limited resolution or contrast, we apply an ML facial landmarking model, as well as contrast-adjusting tone curves to the image prior to its projection onto the photovoltaic retinal implant. Main results. Prosthetic vision simulated using the above algorithm matches the letter acuity observed in clinical studies, as well as the patients' descriptions of perceived facial features. Applying the inversed contrast filter to images prior to projection onto the implant and accentuating the facial features using an ML facial landmarking model helps preserve the contrast in prosthetic vision, improves emotion recognition and reduces the response time. Significance. Spatial and contrast constraints of prosthetic vision limit the resolvable features and degrade natural images. ML based methods and contrast adjustments prior to image projection onto the implant mitigate some limitations and improve face representation.
>
---
#### [replaced 046] Towards Real-world Lens Active Alignment with Unlabeled Data via Domain Adaptation
- **分类: cs.CV; eess.IV; physics.optics**

- **链接: [https://arxiv.org/pdf/2601.03718v2](https://arxiv.org/pdf/2601.03718v2)**

> **作者:** Wenyong Li; Qi Jiang; Weijian Hu; Kailun Yang; Zhanjun Zhang; Wenjun Tian; Kaiwei Wang; Jian Bai
>
> **摘要:** Active Alignment (AA) is a key technology for the large-scale automated assembly of high-precision optical systems. Compared with labor-intensive per-model on-device calibration, a digital-twin pipeline built on optical simulation offers a substantial advantage in generating large-scale labeled data. However, complex imaging conditions induce a domain gap between simulation and real-world images, limiting the generalization of simulation-trained models. To address this, we propose augmenting a simulation baseline with minimal unlabeled real-world images captured at random misalignment positions, mitigating the gap from a domain adaptation perspective. We introduce Domain Adaptive Active Alignment (DA3), which utilizes an autoregressive domain transformation generator and an adversarial-based feature alignment strategy to distill real-world domain information via self-supervised learning. This enables the extraction of domain-invariant image degradation features to facilitate robust misalignment prediction. Experiments on two lens types reveal that DA3 improves accuracy by 46% over a purely simulation pipeline. Notably, it approaches the performance achieved with precisely labeled real-world data collected on 3 lens samples, while reducing on-device data collection time by 98.7%. The results demonstrate that domain adaptation effectively endows simulation-trained models with robust real-world performance, validating the digital-twin pipeline as a practical solution to significantly enhance the efficiency of large-scale optical assembly.
>
---
#### [replaced 047] Novel View Synthesis using DDIM Inversion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10688v2](https://arxiv.org/pdf/2508.10688v2)**

> **作者:** Sehajdeep Singh; A V Subramanyam; Aditya Gupta; Sahil Gupta
>
> **摘要:** Synthesizing novel views from a single input image is a challenging task. It requires extrapolating the 3D structure of a scene while inferring details in occluded regions, and maintaining geometric consistency across viewpoints. Many existing methods must fine-tune large diffusion backbones using multiple views or train a diffusion model from scratch, which is extremely expensive. Additionally, they suffer from blurry reconstruction and poor generalization. This gap presents the opportunity to explore an explicit lightweight view translation framework that can directly utilize the high-fidelity generative capabilities of a pretrained diffusion model while reconstructing a scene from a novel view. Given the DDIM-inverted latent of a single input image, we employ a camera pose-conditioned translation U-Net, TUNet, to predict the inverted latent corresponding to the desired target view. However, the image sampled using the predicted latent may result in a blurry reconstruction. To this end, we propose a novel fusion strategy that exploits the inherent noise correlation structure observed in DDIM inversion. The proposed fusion strategy helps preserve the texture and fine-grained details. To synthesize the novel view, we use the fused latent as the initial condition for DDIM sampling, leveraging the generative prior of the pretrained diffusion model. Extensive experiments on MVImgNet demonstrate that our method outperforms existing methods.
>
---
#### [replaced 048] Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.04068v2](https://arxiv.org/pdf/2601.04068v2)**

> **作者:** Zitong Huang; Kaidong Zhang; Yukang Ding; Chao Gao; Rui Ding; Ying Chen; Wangmeng Zuo
>
> **备注:** Under Review
>
> **摘要:** Aligning text-to-video diffusion models with human preferences is crucial for generating high-quality videos. Existing Direct Preference Otimization (DPO) methods rely on multi-sample ranking and task-specific critic models, which is inefficient and often yields ambiguous global supervision. To address these limitations, we propose LocalDPO, a novel post-training framework that constructs localized preference pairs from real videos and optimizes alignment at the spatio-temporal region level. We design an automated pipeline to efficiently collect preference pair data that generates preference pairs with a single inference per prompt, eliminating the need for external critic models or manual annotation. Specifically, we treat high-quality real videos as positive samples and generate corresponding negatives by locally corrupting them with random spatio-temporal masks and restoring only the masked regions using the frozen base model. During training, we introduce a region-aware DPO loss that restricts preference learning to corrupted areas for rapid convergence. Experiments on Wan2.1 and CogVideoX demonstrate that LocalDPO consistently improves video fidelity, temporal coherence and human preference scores over other post-training approaches, establishing a more efficient and fine-grained paradigm for video generator alignment.
>
---
#### [replaced 049] InfiniteWeb: Scalable Web Environment Synthesis for GUI Agent Training
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出InfiniteWeb，解决GUI代理训练环境不足的问题。通过自动生成大规模功能网页环境，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2601.04126v2](https://arxiv.org/pdf/2601.04126v2)**

> **作者:** Ziyun Zhang; Zezhou Wang; Xiaoyi Zhang; Zongyu Guo; Jiahao Li; Bin Li; Yan Lu
>
> **备注:** Work In Progress
>
> **摘要:** GUI agents that interact with graphical interfaces on behalf of users represent a promising direction for practical AI assistants. However, training such agents is hindered by the scarcity of suitable environments. We present InfiniteWeb, a system that automatically generates functional web environments at scale for GUI agent training. While LLMs perform well on generating a single webpage, building a realistic and functional website with many interconnected pages faces challenges. We address these challenges through unified specification, task-centric test-driven development, and a combination of website seed with reference design image to ensure diversity. Our system also generates verifiable task evaluators enabling dense reward signals for reinforcement learning. Experiments show that InfiniteWeb surpasses commercial coding agents at realistic website construction, and GUI agents trained on our generated environments achieve significant performance improvements on OSWorld and Online-Mind2Web, demonstrating the effectiveness of proposed system.
>
---
#### [replaced 050] Comparative Analysis of Binarization Methods For Medical Image Hashing On Odir Dataset
- **分类: eess.IV; cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2601.02564v2](https://arxiv.org/pdf/2601.02564v2)**

> **作者:** Nedim Muzoglu
>
> **备注:** After publication of the conference version, we identified fundamental methodological and evaluation issues that affect the validity of the reported results. These issues are intrinsic to the current work and cannot be addressed through a simple revision. Therefore, we request full withdrawal of this submission rather than replacement
>
> **摘要:** In this study, we evaluated four binarization methods. Locality-Sensitive Hashing (LSH), Iterative Quantization (ITQ), Kernel-based Supervised Hashing (KSH), and Supervised Discrete Hashing (SDH) on the ODIR dataset using deep feature embeddings. Experimental results show that SDH achieved the best performance, with an mAP@100 of 0.9184 using only 32-bit codes, outperforming LSH, ITQ, and KSH. Compared with prior studies, our method proved highly competitive: Fang et al. reported 0.7528 (Fundus-iSee, 48 bits) and 0.8856 (ASOCT-Cataract, 48 bits), while Wijesinghe et al. achieved 94.01 (KVASIR, 256 bits). Despite using significantly fewer bits, our SDH-based framework reached retrieval accuracy close to the state-of-the-art. These findings demonstrate that SDH is the most effective approach among those tested, offering a practical balance of accuracy, storage, and efficiency for medical image retrieval and device inventory management.
>
---
#### [replaced 051] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.13425v2](https://arxiv.org/pdf/2507.13425v2)**

> **作者:** Sirui Wang; Zhou Guan; Bingxi Zhao; Tongjia Gu; Jie Liu
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Accurate prediction of driving intention is key to enhancing the safety and interactive efficiency of human-machine co-driving systems. It serves as a cornerstone for achieving high-level autonomous driving. However, current approaches remain inadequate for accurately modeling the complex spatiotemporal interdependencies and the unpredictable variability of human driving behavior. To address these challenges, we propose CaTFormer, a causal Temporal Transformer that explicitly models causal interactions between driver behavior and environmental context for robust intention prediction. Specifically, CaTFormer introduces a novel Reciprocal Delayed Fusion (RDF) mechanism for precise temporal alignment of interior and exterior feature streams, a Counterfactual Residual Encoding (CRE) module that systematically eliminates spurious correlations to reveal authentic causal dependencies, and an innovative Feature Synthesis Network (FSN) that adaptively synthesizes these purified representations into coherent temporal representations. Experimental results demonstrate that CaTFormer attains state-of-the-art performance on the Brain4Cars dataset. It effectively captures complex causal temporal dependencies and enhances both the accuracy and transparency of driving intention prediction.
>
---
#### [replaced 052] DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders for Clinical AI Research
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.06099v2](https://arxiv.org/pdf/2506.06099v2)**

> **作者:** Shanawaj S Madarkar; Mahajabeen Madarkar; Madhumitha Venkatesh; Deepanshu Bansal; Teli Prakash; Konda Reddy Mopuri; Vinaykumar MV; KVL Sathwika; Adarsh Kasturi; Gandla Dilip Raj; PVN Supranitha; Harsh Udai
>
> **备注:** Accepted at NeurIPS 2025 (D&B Track)
>
> **摘要:** Artificial intelligence is poised to augment dermatological care by enabling scalable image-based diagnostics. Yet, the development of robust and equitable models remains hindered by datasets that fail to capture the clinical and demographic complexity of real-world practice. This complexity stems from region-specific disease distributions, wide variation in skin tones, and the underrepresentation of outpatient scenarios from non-Western populations. We introduce DermaCon-IN, a prospectively curated dermatology dataset comprising 5,450 clinical images from 3,002 patients across outpatient clinics in South India. Each image is annotated by board-certified dermatologists with 245 distinct diagnoses, structured under a hierarchical, aetiology-based taxonomy adapted from Rook's classification. The dataset captures a wide spectrum of dermatologic conditions and tonal variation commonly seen in Indian outpatient care. We benchmark a range of architectures, including convolutional models (ResNet, DenseNet, EfficientNet), transformer-based models (ViT, MaxViT, Swin), and Concept Bottleneck Models to establish baseline performance and explore how anatomical and concept-level cues may be integrated. These results are intended to guide future efforts toward interpretable and clinically realistic models. DermaCon-IN provides a scalable and representative foundation for advancing dermatology AI.
>
---
#### [replaced 053] Interleaved Latent Visual Reasoning with Selective Perceptual Modeling
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出ILVR框架，解决多模态大语言模型中视觉反馈计算成本高与感知建模不足的问题，通过交错推理和选择性感知建模提升性能。**

- **链接: [https://arxiv.org/pdf/2512.05665v2](https://arxiv.org/pdf/2512.05665v2)**

> **作者:** Shuai Dong; Siyuan Wang; Xingyu Liu; Chenglin Li; Haowen Hou; Zhongyu Wei
>
> **备注:** 18 pages, 11 figures. Code available at https://github.com/XD111ds/ILVR
>
> **摘要:** Interleaved reasoning paradigms enhance Multimodal Large Language Models (MLLMs) with visual feedback but are hindered by the prohibitive computational cost of re-encoding pixel-dense images. A promising alternative, latent visual reasoning, circumvents this bottleneck yet faces limitations: methods either fail to capture intermediate state evolution due to single-step, non-interleaved structures, or sacrifice precise perceptual modeling by over-compressing features. We introduce Interleaved Latent Visual Reasoning (ILVR), a framework that unifies dynamic state evolution with precise perceptual modeling. ILVR interleaves textual generation with latent visual representations that act as specific, evolving cues for subsequent reasoning. Specifically, we employ a self-supervision strategy where a momentum teacher model selectively distills relevant features from ground-truth intermediate images into sparse supervision targets. This adaptive selection mechanism guides the model to autonomously generate context-aware visual signals. Extensive experiments on multimodal reasoning benchmarks demonstrate that ILVR outperforms existing approaches, effectively bridging the gap between fine-grained perception and sequential multimodal reasoning.
>
---
#### [replaced 054] TRec: Egocentric Action Recognition using 2D Point Tracks
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.03667v2](https://arxiv.org/pdf/2601.03667v2)**

> **作者:** Dennis Holzmann; Sven Wachsmuth
>
> **备注:** submitted to ICPR 2026
>
> **摘要:** We present a novel approach for egocentric action recognition that leverages 2D point tracks as an additional motion cue. While most existing methods rely on RGB appearance, human pose estimation, or their combination, our work demonstrates that tracking randomly sampled image points across video frames can substantially improve recognition accuracy. Unlike prior approaches, we do not detect hands, objects, or interaction regions. Instead, we employ CoTracker to follow a set of randomly initialized points through each video and use the resulting trajectories, together with the corresponding image frames, as input to a Transformer-based recognition model. Surprisingly, our method achieves notable gains even when only the initial frame and its associated point tracks are provided, without incorporating the full video sequence. Experimental results confirm that integrating 2D point tracks consistently enhances performance compared to the same model trained without motion information, highlighting their potential as a lightweight yet effective representation for egocentric action understanding.
>
---
#### [replaced 055] StreamFlow: Theory, Algorithm, and Implementation for High-Efficiency Rectified Flow Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22009v2](https://arxiv.org/pdf/2511.22009v2)**

> **作者:** Sen Fang; Hongbin Zhong; Yalin Feng; Yanxin Zhang; Dimitris N. Metaxas
>
> **备注:** Improved the quality. Project Page at https://world-snapshot.github.io/StreamFlow/
>
> **摘要:** New technologies such as Rectified Flow and Flow Matching have significantly improved the performance of generative models in the past two years, especially in terms of control accuracy, generation quality, and generation efficiency. However, due to some differences in its theory, design, and existing diffusion models, the existing acceleration methods cannot be directly applied to the Rectified Flow model. In this article, we have comprehensively implemented an overall acceleration pipeline from the aspects of theory, design, and reasoning strategies. This pipeline uses new methods such as batch processing with a new velocity field, vectorization of heterogeneous time-step batch processing, and dynamic TensorRT compilation for the new methods to comprehensively accelerate related models based on flow models. Currently, the existing public methods usually achieve an acceleration of 18%, while experiments have proved that our new method can accelerate the 512*512 image generation speed to up to 611%, which is far beyond the current non-generalized acceleration methods.
>
---
#### [replaced 056] PrismVAU: Prompt-Refined Inference System for Multimodal Video Anomaly Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.02927v2](https://arxiv.org/pdf/2601.02927v2)**

> **作者:** Iñaki Erregue; Kamal Nasrollahi; Sergio Escalera
>
> **备注:** This paper has been accepted to the 6th Workshop on Real-World Surveillance: Applications and Challenges (WACV 2026)
>
> **摘要:** Video Anomaly Understanding (VAU) extends traditional Video Anomaly Detection (VAD) by not only localizing anomalies but also describing and reasoning about their context. Existing VAU approaches often rely on fine-tuned multimodal large language models (MLLMs) or external modules such as video captioners, which introduce costly annotations, complex training pipelines, and high inference overhead. In this work, we introduce PrismVAU, a lightweight yet effective system for real-time VAU that leverages a single off-the-shelf MLLM for anomaly scoring, explanation, and prompt optimization. PrismVAU operates in two complementary stages: (1) a coarse anomaly scoring module that computes frame-level anomaly scores via similarity to textual anchors, and (2) an MLLM-based refinement module that contextualizes anomalies through system and user prompts. Both textual anchors and prompts are optimized with a weakly supervised Automatic Prompt Engineering (APE) framework. Extensive experiments on standard VAD benchmarks demonstrate that PrismVAU delivers competitive detection performance and interpretable anomaly explanations -- without relying on instruction tuning, frame-level annotations, and external modules or dense processing -- making it an efficient and practical solution for real-world applications.
>
---
#### [replaced 057] MAFNet:Multi-frequency Adaptive Fusion Network for Real-time Stereo Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04358v2](https://arxiv.org/pdf/2512.04358v2)**

> **作者:** Ao Xu; Rujin Zhao; Xiong Xu; Boceng Huang; Yujia Jia; Hongfeng Long; Fuxuan Chen; Zilong Cao; Fangyuan Chen
>
> **摘要:** Existing stereo matching networks typically rely on either cost-volume construction based on 3D convolutions or deformation methods based on iterative optimization. The former incurs significant computational overhead during cost aggregation, whereas the latter often lacks the ability to model non-local contextual information. These methods exhibit poor compatibility on resource-constrained mobile devices, limiting their deployment in real-time applications. To address this, we propose a Multi-frequency Adaptive Fusion Network (MAFNet), which can produce high-quality disparity maps using only efficient 2D convolutions. Specifically, we design an adaptive frequency-domain filtering attention module that decomposes the full cost volume into high-frequency and low-frequency volumes, performing frequency-aware feature aggregation separately. Subsequently, we introduce a Linformer-based low-rank attention mechanism to adaptively fuse high- and low-frequency information, yielding more robust disparity estimation. Extensive experiments demonstrate that the proposed MAFNet significantly outperforms existing real-time methods on public datasets such as Scene Flow and KITTI 2015, showing a favorable balance between accuracy and real-time performance.
>
---
#### [replaced 058] MobileGeo: Exploring Hierarchical Knowledge Distillation for Resource-Efficient Cross-view Drone Geo-Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.22582v3](https://arxiv.org/pdf/2510.22582v3)**

> **作者:** Jian Sun; Kangdao Liu; Chi Zhang; Chuangquan Chen; Junge Shen; C. L. Philip Chen; Chi-Man Vong
>
> **摘要:** Cross-view geo-localization (CVGL) plays a vital role in drone-based multimedia applications, enabling precise localization by matching drone-captured aerial images against geo-tagged satellite databases in GNSS-denied environments. However, existing methods rely on resource-intensive feature alignment and multi-branch architectures, incurring high inference costs that limit their deployment on edge devices. We propose MobileGeo, a mobile-friendly framework designed for efficient on-device CVGL: 1) During training, a Hierarchical Distillation (HD-CVGL) paradigm, coupled with Uncertainty-Aware Prediction Alignment (UAPA), distills essential information into a compact model without incurring inference overhead. 2) During inference, an efficient Multi-view Selection Refinement Module (MSRM) leverages mutual information to filter redundant views and reduce computational load. Extensive experiments demonstrate that MobileGeo outperforms previous state-of-the-art methods, achieving a 4.19% improvement in AP on University1652 dataset while being over 5 times efficient in FLOPs and 3 times faster. Crucially, MobileGeo runs at 251.5 FPS on an NVIDIA AGX Orin edge device, demonstrating its practical viability for real-time on-device drone geo-localization. The code is available at https://github.com/SkyEyeLoc/MobileGeo.
>
---
#### [replaced 059] From Dataset to Real-world: General 3D Object Detection via Generalized Cross-domain Few-shot Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.06282v2](https://arxiv.org/pdf/2503.06282v2)**

> **作者:** Shuangzhi Li; Junlong Shen; Lei Ma; Xingyu Li
>
> **备注:** The latest version refines the few-shot setting on common classes, enforcing a stricter object-level definition
>
> **摘要:** LiDAR-based 3D object detection models often struggle to generalize to real-world environments due to limited object diversity in existing datasets. To tackle it, we introduce the first generalized cross-domain few-shot (GCFS) task in 3D object detection, aiming to adapt a source-pretrained model to both common and novel classes in a new domain with only few-shot annotations. We propose a unified framework that learns stable target semantics under limited supervision by bridging 2D open-set semantics with 3D spatial reasoning. Specifically, an image-guided multi-modal fusion injects transferable 2D semantic cues into the 3D pipeline via vision-language models, while a physically-aware box search enhances 2D-to-3D alignment via LiDAR priors. To capture class-specific semantics from sparse data, we further introduce contrastive-enhanced prototype learning, which encodes few-shot instances into discriminative semantic anchors and stabilizes representation learning. Extensive experiments on GCFS benchmarks demonstrate the effectiveness and generality of our approach in realistic deployment settings.
>
---
#### [replaced 060] CHIMERA: Adaptive Cache Injection and Semantic Anchor Prompting for Zero-shot Image Morphing with Morphing-oriented Metrics
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07155v4](https://arxiv.org/pdf/2512.07155v4)**

> **作者:** Dahyeon Kye; Jeahun Sung; Minkyu Jeon; Jihyong Oh
>
> **备注:** Please visit our project page at https://cmlab-korea.github.io/CHIMERA/
>
> **摘要:** Diffusion models exhibit remarkable generative ability, yet achieving smooth and semantically consistent image morphing remains a challenge. Existing approaches often yield abrupt transitions or over-saturated appearances due to the lack of adaptive structural and semantic alignments. We propose CHIMERA, a zero-shot diffusion-based framework that formulates morphing as a cached inversion-guided denoising process. To handle large semantic and appearance disparities, we propose Adaptive Cache Injection and Semantic Anchor Prompting. Adaptive Cache Injection (ACI) caches down, mid, and up blocks features from both inputs during DDIM inversion and re-injects them adaptively during denoising, enabling spatial and semantic alignment in depth- and time-adaptive manners and enabling natural feature fusion and smooth transitions. Semantic Anchor Prompting (SAP) leverages a vision-language model to generate a shared anchor prompt that serves as a semantic anchor, bridging dissimilar inputs and guiding the denoising process toward coherent results. Finally, we introduce the Global-Local Consistency Score (GLCS), a morphing-oriented metric that simultaneously evaluates the global harmonization of the two inputs and the smoothness of the local morphing transition. Extensive experiments and user studies show that CHIMERA achieves smoother and more semantically aligned transitions than existing methods, establishing a new state of the art in image morphing. The code and project page will be publicly released.
>
---
#### [replaced 061] FoldNet: Learning Generalizable Closed-Loop Policy for Garment Folding via Keypoint-Driven Asset and Demonstration Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人服装折叠任务，解决数据生成与策略学习难题。通过关键点驱动的合成数据和模仿学习，提升折叠策略的泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2505.09109v2](https://arxiv.org/pdf/2505.09109v2)**

> **作者:** Yuxing Chen; Bowen Xiao; He Wang
>
> **摘要:** Due to the deformability of garments, generating a large amount of high-quality data for robotic garment manipulation tasks is highly challenging. In this paper, we present a synthetic garment dataset that can be used for robotic garment folding. We begin by constructing geometric garment templates based on keypoints and applying generative models to generate realistic texture patterns. Leveraging these keypoint annotations, we generate folding demonstrations in simulation and train folding policies via closed-loop imitation learning. To improve robustness, we propose KG-DAgger, which uses a keypoint-based strategy to generate demonstration data for recovering from failures. KG-DAgger significantly improves the model performance, boosting the real-world success rate by 25\%. After training with 15K trajectories (about 2M image-action pairs), the model achieves a 75\% success rate in the real world. Experiments in both simulation and real-world settings validate the effectiveness of our proposed framework.
>
---
