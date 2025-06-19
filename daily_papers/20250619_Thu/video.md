# 计算机视觉 cs.CV

- **最新发布 93 篇**

- **更新 58 篇**

## 最新发布

#### [new 001] Open-World Object Counting in Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出开放世界视频目标计数任务，旨在根据文本或图像描述准确统计视频中目标物体数量，解决拥挤场景下的重复计数与重识别问题。**

- **链接: [http://arxiv.org/pdf/2506.15368v1](http://arxiv.org/pdf/2506.15368v1)**

> **作者:** Niki Amini-Naieni; Andrew Zisserman
>
> **摘要:** We introduce a new task of open-world object counting in videos: given a text description, or an image example, that specifies the target object, the objective is to enumerate all the unique instances of the target objects in the video. This task is especially challenging in crowded scenes with occlusions and similar objects, where avoiding double counting and identifying reappearances is crucial. To this end, we make the following contributions: we introduce a model, CountVid, for this task. It leverages an image-based counting model, and a promptable video segmentation and tracking model to enable automated, open-world object counting across video frames. To evaluate its performance, we introduce VideoCount, a new dataset for our novel task built from the TAO and MOT20 tracking datasets, as well as from videos of penguins and metal alloy crystallization captured by x-rays. Using this dataset, we demonstrate that CountVid provides accurate object counts, and significantly outperforms strong baselines. The VideoCount dataset, the CountVid model, and all the code are available at https://github.com/niki-amini-naieni/CountVid/.
>
---
#### [new 002] Efficient Retail Video Annotation: A Robust Key Frame Generation Approach for Product and Customer Interaction Analysis
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于视频标注任务，旨在解决零售视频中人工标注效率低、成本高的问题。通过深度学习自动识别关键帧并实现产品与顾客互动的自动化标注。**

- **链接: [http://arxiv.org/pdf/2506.14854v1](http://arxiv.org/pdf/2506.14854v1)**

> **作者:** Varun Mannam; Zhenyu Shi
>
> **备注:** Submitting to ICCV 2025 workshop: https://retailvisionworkshop.github.io/
>
> **摘要:** Accurate video annotation plays a vital role in modern retail applications, including customer behavior analysis, product interaction detection, and in-store activity recognition. However, conventional annotation methods heavily rely on time-consuming manual labeling by human annotators, introducing non-robust frame selection and increasing operational costs. To address these challenges in the retail domain, we propose a deep learning-based approach that automates key-frame identification in retail videos and provides automatic annotations of products and customers. Our method leverages deep neural networks to learn discriminative features by embedding video frames and incorporating object detection-based techniques tailored for retail environments. Experimental results showcase the superiority of our approach over traditional methods, achieving accuracy comparable to human annotator labeling while enhancing the overall efficiency of retail video annotation. Remarkably, our approach leads to an average of 2 times cost savings in video annotation. By allowing human annotators to verify/adjust less than 5% of detected frames in the video dataset, while automating the annotation process for the remaining frames without reducing annotation quality, retailers can significantly reduce operational costs. The automation of key-frame detection enables substantial time and effort savings in retail video labeling tasks, proving highly valuable for diverse retail applications such as shopper journey analysis, product interaction detection, and in-store security monitoring.
>
---
#### [new 003] PictSure: Pretraining Embeddings Matters for In-Context Learning Image Classifiers
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分类任务，解决数据稀缺领域中的少样本分类问题。通过优化嵌入模型的预训练和架构，提升跨域泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.14842v1](http://arxiv.org/pdf/2506.14842v1)**

> **作者:** Lukas Schiesser; Cornelius Wolff; Sophie Haas; Simon Pukrop
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Building image classification models remains cumbersome in data-scarce domains, where collecting large labeled datasets is impractical. In-context learning (ICL) has emerged as a promising paradigm for few-shot image classification (FSIC), enabling models to generalize across domains without gradient-based adaptation. However, prior work has largely overlooked a critical component of ICL-based FSIC pipelines: the role of image embeddings. In this work, we present PictSure, an ICL framework that places the embedding model -- its architecture, pretraining, and training dynamics -- at the center of analysis. We systematically examine the effects of different visual encoder types, pretraining objectives, and fine-tuning strategies on downstream FSIC performance. Our experiments show that the training success and the out-of-domain performance are highly dependent on how the embedding models are pretrained. Consequently, PictSure manages to outperform existing ICL-based FSIC models on out-of-domain benchmarks that differ significantly from the training distribution, while maintaining comparable results on in-domain tasks. Code can be found at https://github.com/PictSure/pictsure-library.
>
---
#### [new 004] Unsupervised Pelage Pattern Unwrapping for Animal Re-identification
- **分类: cs.CV**

- **简介: 该论文属于动物重识别任务，解决动物毛发图案因姿态变化导致的匹配困难问题。提出几何感知的纹理映射方法，将毛发图案映射到标准空间以提升匹配精度。**

- **链接: [http://arxiv.org/pdf/2506.15369v1](http://arxiv.org/pdf/2506.15369v1)**

> **作者:** Aleksandr Algasov; Ekaterina Nepovinnykh; Fedor Zolotarev; Tuomas Eerola; Heikki Kälviäinen; Pavel Zemčík; Charles V. Stewart
>
> **摘要:** Existing individual re-identification methods often struggle with the deformable nature of animal fur or skin patterns which undergo geometric distortions due to body movement and posture changes. In this paper, we propose a geometry-aware texture mapping approach that unwarps pelage patterns, the unique markings found on an animal's skin or fur, into a canonical UV space, enabling more robust feature matching. Our method uses surface normal estimation to guide the unwrapping process while preserving the geometric consistency between the 3D surface and the 2D texture space. We focus on two challenging species: Saimaa ringed seals (Pusa hispida saimensis) and leopards (Panthera pardus). Both species have distinctive yet highly deformable fur patterns. By integrating our pattern-preserving UV mapping with existing re-identification techniques, we demonstrate improved accuracy across diverse poses and viewing angles. Our framework does not require ground truth UV annotations and can be trained in a self-supervised manner. Experiments on seal and leopard datasets show up to a 5.4% improvement in re-identification accuracy.
>
---
#### [new 005] ArchShapeNet:An Interpretable 3D-CNN Framework for Evaluating Architectural Shapes
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D形状分类任务，旨在解决人机设计形式差异分析难题。构建了数据集，提出ArchShapeNet模型，提升区分准确率。**

- **链接: [http://arxiv.org/pdf/2506.14832v1](http://arxiv.org/pdf/2506.14832v1)**

> **作者:** Jun Yin; Jing Zhong; Pengyu Zeng; Peilin Li; Zixuan Dai; Miao Zhang; Shuai Lu
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** In contemporary architectural design, the growing complexity and diversity of design demands have made generative plugin tools essential for quickly producing initial concepts and exploring novel 3D forms. However, objectively analyzing the differences between human-designed and machine-generated 3D forms remains a challenge, limiting our understanding of their respective strengths and hindering the advancement of generative tools. To address this, we built ArchForms-4000, a dataset containing 2,000 architect-designed and 2,000 Evomass-generated 3D forms; Proposed ArchShapeNet, a 3D convolutional neural network tailored for classifying and analyzing architectural forms, incorporating a saliency module to highlight key spatial features aligned with architectural reasoning; And conducted comparative experiments showing our model outperforms human experts in distinguishing form origins, achieving 94.29% accuracy, 96.2% precision, and 98.51% recall. This study not only highlights the distinctive advantages of human-designed forms in spatial organization, proportional harmony, and detail refinement but also provides valuable insights for enhancing generative design tools in the future.
>
---
#### [new 006] Retrospective Memory for Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 该论文属于伪装目标检测任务，旨在解决现有方法缺乏历史上下文机制的问题。提出RetroMem架构，通过记忆机制提升模型对复杂伪装场景的理解与适应能力。**

- **链接: [http://arxiv.org/pdf/2506.15244v1](http://arxiv.org/pdf/2506.15244v1)**

> **作者:** Chenxi Zhang; Jiayun Wu; Qing Zhang; Yazhe Zhai; Youwei Pang
>
> **摘要:** Camouflaged object detection (COD) primarily focuses on learning subtle yet discriminative representations from complex scenes. Existing methods predominantly follow the parametric feedforward architecture based on static visual representation modeling. However, they lack explicit mechanisms for acquiring historical context, limiting their adaptation and effectiveness in handling challenging camouflage scenes. In this paper, we propose a recall-augmented COD architecture, namely RetroMem, which dynamically modulates camouflage pattern perception and inference by integrating relevant historical knowledge into the process. Specifically, RetroMem employs a two-stage training paradigm consisting of a learning stage and a recall stage to construct, update, and utilize memory representations effectively. During the learning stage, we design a dense multi-scale adapter (DMA) to improve the pretrained encoder's capability to capture rich multi-scale visual information with very few trainable parameters, thereby providing foundational inferences. In the recall stage, we propose a dynamic memory mechanism (DMM) and an inference pattern reconstruction (IPR). These components fully leverage the latent relationships between learned knowledge and current sample context to reconstruct the inference of camouflage patterns, thereby significantly improving the model's understanding of camouflage scenes. Extensive experiments on several widely used datasets demonstrate that our RetroMem significantly outperforms existing state-of-the-art methods.
>
---
#### [new 007] Frequency-Calibrated Membership Inference Attacks on Medical Image Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于隐私安全任务，旨在解决医疗图像扩散模型的成员推理攻击问题。通过提出频率校准的重建误差方法，提升攻击效果。**

- **链接: [http://arxiv.org/pdf/2506.14919v1](http://arxiv.org/pdf/2506.14919v1)**

> **作者:** Xinkai Zhao; Yuta Tokuoka; Junichiro Iwasawa; Keita Oda
>
> **摘要:** The increasing use of diffusion models for image generation, especially in sensitive areas like medical imaging, has raised significant privacy concerns. Membership Inference Attack (MIA) has emerged as a potential approach to determine if a specific image was used to train a diffusion model, thus quantifying privacy risks. Existing MIA methods often rely on diffusion reconstruction errors, where member images are expected to have lower reconstruction errors than non-member images. However, applying these methods directly to medical images faces challenges. Reconstruction error is influenced by inherent image difficulty, and diffusion models struggle with high-frequency detail reconstruction. To address these issues, we propose a Frequency-Calibrated Reconstruction Error (FCRE) method for MIAs on medical image diffusion models. By focusing on reconstruction errors within a specific mid-frequency range and excluding both high-frequency (difficult to reconstruct) and low-frequency (less informative) regions, our frequency-selective approach mitigates the confounding factor of inherent image difficulty. Specifically, we analyze the reverse diffusion process, obtain the mid-frequency reconstruction error, and compute the structural similarity index score between the reconstructed and original images. Membership is determined by comparing this score to a threshold. Experiments on several medical image datasets demonstrate that our FCRE method outperforms existing MIA methods.
>
---
#### [new 008] Vision Transformers for End-to-End Quark-Gluon Jet Classification from Calorimeter Images
- **分类: cs.CV**

- **简介: 该论文属于高能物理中的粒子分类任务，旨在解决夸克与胶子喷注的区分问题。通过构建多通道图像并应用ViT模型，提升分类性能。**

- **链接: [http://arxiv.org/pdf/2506.14934v1](http://arxiv.org/pdf/2506.14934v1)**

> **作者:** Md Abrar Jahin; Shahriar Soudeep; Arian Rahman Aditta; M. F. Mridha; Nafiz Fahad; Md. Jakir Hossen
>
> **备注:** Accepted in Third International Workshop on Generalizing from Limited Resources in the Open World Workshop at International Joint Conference on Artificial Intelligence (IJCAI) 2025
>
> **摘要:** Distinguishing between quark- and gluon-initiated jets is a critical and challenging task in high-energy physics, pivotal for improving new physics searches and precision measurements at the Large Hadron Collider. While deep learning, particularly Convolutional Neural Networks (CNNs), has advanced jet tagging using image-based representations, the potential of Vision Transformer (ViT) architectures, renowned for modeling global contextual information, remains largely underexplored for direct calorimeter image analysis, especially under realistic detector and pileup conditions. This paper presents a systematic evaluation of ViTs and ViT-CNN hybrid models for quark-gluon jet classification using simulated 2012 CMS Open Data. We construct multi-channel jet-view images from detector-level energy deposits (ECAL, HCAL) and reconstructed tracks, enabling an end-to-end learning approach. Our comprehensive benchmarking demonstrates that ViT-based models, notably ViT+MaxViT and ViT+ConvNeXt hybrids, consistently outperform established CNN baselines in F1-score, ROC-AUC, and accuracy, highlighting the advantage of capturing long-range spatial correlations within jet substructure. This work establishes the first systematic framework and robust performance baselines for applying ViT architectures to calorimeter image-based jet classification using public collider data, alongside a structured dataset suitable for further deep learning research in this domain.
>
---
#### [new 009] DETONATE: A Benchmark for Text-to-Image Alignment and Kernelized Direct Preference Optimization
- **分类: cs.CV**

- **简介: 该论文属于文本到图像对齐任务，旨在提升模型生成内容的安全性与公平性。提出DPO-Kernels方法，结合多种损失函数和核函数优化模型对齐效果，并构建了DETONATE基准测试集。**

- **链接: [http://arxiv.org/pdf/2506.14903v1](http://arxiv.org/pdf/2506.14903v1)**

> **作者:** Renjith Prasad; Abhilekh Borah; Hasnat Md Abdullah; Chathurangi Shyalika; Gurpreet Singh; Ritvik Garimella; Rajarshi Roy; Harshul Surana; Nasrin Imanpour; Suranjana Trivedy; Amit Sheth; Amitava Das
>
> **备注:** 59 pages, 10 figures
>
> **摘要:** Alignment is crucial for text-to-image (T2I) models to ensure that generated images faithfully capture user intent while maintaining safety and fairness. Direct Preference Optimization (DPO), prominent in large language models (LLMs), is extending its influence to T2I systems. This paper introduces DPO-Kernels for T2I models, a novel extension enhancing alignment across three dimensions: (i) Hybrid Loss, integrating embedding-based objectives with traditional probability-based loss for improved optimization; (ii) Kernelized Representations, employing Radial Basis Function (RBF), Polynomial, and Wavelet kernels for richer feature transformations and better separation between safe and unsafe inputs; and (iii) Divergence Selection, expanding beyond DPO's default Kullback-Leibler (KL) regularizer by incorporating Wasserstein and R'enyi divergences for enhanced stability and robustness. We introduce DETONATE, the first large-scale benchmark of its kind, comprising approximately 100K curated image pairs categorized as chosen and rejected. DETONATE encapsulates three axes of social bias and discrimination: Race, Gender, and Disability. Prompts are sourced from hate speech datasets, with images generated by leading T2I models including Stable Diffusion 3.5 Large, Stable Diffusion XL, and Midjourney. Additionally, we propose the Alignment Quality Index (AQI), a novel geometric measure quantifying latent-space separability of safe/unsafe image activations, revealing hidden vulnerabilities. Empirically, we demonstrate that DPO-Kernels maintain strong generalization bounds via Heavy-Tailed Self-Regularization (HT-SR). DETONATE and complete code are publicly released.
>
---
#### [new 010] ViLLa: A Neuro-Symbolic approach for Animal Monitoring
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ViLLa，一种用于动物监测的神经符号框架，解决视觉与语言结合的动物识别问题，通过整合视觉检测、语言解析和逻辑推理实现可解释的监控。**

- **链接: [http://arxiv.org/pdf/2506.14823v1](http://arxiv.org/pdf/2506.14823v1)**

> **作者:** Harsha Koduri
>
> **摘要:** Monitoring animal populations in natural environments requires systems that can interpret both visual data and human language queries. This work introduces ViLLa (Vision-Language-Logic Approach), a neuro-symbolic framework designed for interpretable animal monitoring. ViLLa integrates three core components: a visual detection module for identifying animals and their spatial locations in images, a language parser for understanding natural language queries, and a symbolic reasoning layer that applies logic-based inference to answer those queries. Given an image and a question such as "How many dogs are in the scene?" or "Where is the buffalo?", the system grounds visual detections into symbolic facts and uses predefined rules to compute accurate answers related to count, presence, and location. Unlike end-to-end black-box models, ViLLa separates perception, understanding, and reasoning, offering modularity and transparency. The system was evaluated on a range of animal imagery tasks and demonstrates the ability to bridge visual content with structured, human-interpretable queries.
>
---
#### [new 011] Mono-Modalizing Extremely Heterogeneous Multi-Modal Medical Image Registration
- **分类: cs.CV; I.4.5; I.4.9; J.3**

- **简介: 该论文属于医学图像配准任务，解决多模态图像（如PET、FA与MRI）因差异大导致配准困难的问题。提出M2M-Reg框架，提升配准精度。**

- **链接: [http://arxiv.org/pdf/2506.15596v1](http://arxiv.org/pdf/2506.15596v1)**

> **作者:** Kyobin Choo; Hyunkyung Han; Jinyeong Kim; Chanyong Yoon; Seong Jae Hwang
>
> **备注:** 11 pages, 3 figures, 2 tables, Accepted at Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025
>
> **摘要:** In clinical practice, imaging modalities with functional characteristics, such as positron emission tomography (PET) and fractional anisotropy (FA), are often aligned with a structural reference (e.g., MRI, CT) for accurate interpretation or group analysis, necessitating multi-modal deformable image registration (DIR). However, due to the extreme heterogeneity of these modalities compared to standard structural scans, conventional unsupervised DIR methods struggle to learn reliable spatial mappings and often distort images. We find that the similarity metrics guiding these models fail to capture alignment between highly disparate modalities. To address this, we propose M2M-Reg (Multi-to-Mono Registration), a novel framework that trains multi-modal DIR models using only mono-modal similarity while preserving the established architectural paradigm for seamless integration into existing models. We also introduce GradCyCon, a regularizer that leverages M2M-Reg's cyclic training scheme to promote diffeomorphism. Furthermore, our framework naturally extends to a semi-supervised setting, integrating pre-aligned and unaligned pairs only, without requiring ground-truth transformations or segmentation masks. Experiments on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset demonstrate that M2M-Reg achieves up to 2x higher DSC than prior methods for PET-MRI and FA-MRI registration, highlighting its effectiveness in handling highly heterogeneous multi-modal DIR. Our code is available at https://github.com/MICV-yonsei/M2M-Reg.
>
---
#### [new 012] GenHOI: Generalizing Text-driven 4D Human-Object Interaction Synthesis for Unseen Objects
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于4D人体-物体交互生成任务，旨在解决缺乏大规模4D HOI数据的问题。提出GenHOI框架，通过两阶段方法实现未见物体的高质量4D交互合成。**

- **链接: [http://arxiv.org/pdf/2506.15483v1](http://arxiv.org/pdf/2506.15483v1)**

> **作者:** Shujia Li; Haiyu Zhang; Xinyuan Chen; Yaohui Wang; Yutong Ban
>
> **摘要:** While diffusion models and large-scale motion datasets have advanced text-driven human motion synthesis, extending these advances to 4D human-object interaction (HOI) remains challenging, mainly due to the limited availability of large-scale 4D HOI datasets. In our study, we introduce GenHOI, a novel two-stage framework aimed at achieving two key objectives: 1) generalization to unseen objects and 2) the synthesis of high-fidelity 4D HOI sequences. In the initial stage of our framework, we employ an Object-AnchorNet to reconstruct sparse 3D HOI keyframes for unseen objects, learning solely from 3D HOI datasets, thereby mitigating the dependence on large-scale 4D HOI datasets. Subsequently, we introduce a Contact-Aware Diffusion Model (ContactDM) in the second stage to seamlessly interpolate sparse 3D HOI keyframes into densely temporally coherent 4D HOI sequences. To enhance the quality of generated 4D HOI sequences, we propose a novel Contact-Aware Encoder within ContactDM to extract human-object contact patterns and a novel Contact-Aware HOI Attention to effectively integrate the contact signals into diffusion models. Experimental results show that we achieve state-of-the-art results on the publicly available OMOMO and 3D-FUTURE datasets, demonstrating strong generalization abilities to unseen objects, while enabling high-fidelity 4D HOI generation.
>
---
#### [new 013] Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在加速扩散模型的推理。通过进化缓存方法ECAD，学习高效缓存策略，在不修改模型的情况下提升速度并控制质量与延迟的权衡。**

- **链接: [http://arxiv.org/pdf/2506.15682v1](http://arxiv.org/pdf/2506.15682v1)**

> **作者:** Anirud Aggarwal; Abhinav Shrivastava; Matthew Gwilliam
>
> **备注:** 29 pages, 22 figures, 9 tables
>
> **摘要:** Diffusion-based image generation models excel at producing high-quality synthetic content, but suffer from slow and computationally expensive inference. Prior work has attempted to mitigate this by caching and reusing features within diffusion transformers across inference steps. These methods, however, often rely on rigid heuristics that result in limited acceleration or poor generalization across architectures. We propose Evolutionary Caching to Accelerate Diffusion models (ECAD), a genetic algorithm that learns efficient, per-model, caching schedules forming a Pareto frontier, using only a small set of calibration prompts. ECAD requires no modifications to network parameters or reference images. It offers significant inference speedups, enables fine-grained control over the quality-latency trade-off, and adapts seamlessly to different diffusion models. Notably, ECAD's learned schedules can generalize effectively to resolutions and model variants not seen during calibration. We evaluate ECAD on PixArt-alpha, PixArt-Sigma, and FLUX-1.dev using multiple metrics (FID, CLIP, Image Reward) across diverse benchmarks (COCO, MJHQ-30k, PartiPrompts), demonstrating consistent improvements over previous approaches. On PixArt-alpha, ECAD identifies a schedule that outperforms the previous state-of-the-art method by 4.47 COCO FID while increasing inference speedup from 2.35x to 2.58x. Our results establish ECAD as a scalable and generalizable approach for accelerating diffusion inference. Our project website is available at https://aniaggarwal.github.io/ecad and our code is available at https://github.com/aniaggarwal/ecad.
>
---
#### [new 014] FindingDory: A Benchmark to Evaluate Memory in Embodied Agents
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人记忆评估任务，旨在解决长时记忆在具身智能体中的应用问题。提出一个基准测试，评估智能体在复杂环境中的记忆与推理能力。**

- **链接: [http://arxiv.org/pdf/2506.15635v1](http://arxiv.org/pdf/2506.15635v1)**

> **作者:** Karmesh Yadav; Yusuf Ali; Gunshi Gupta; Yarin Gal; Zsolt Kira
>
> **备注:** Our dataset and code will be made available at: https://findingdory-benchmark.github.io/
>
> **摘要:** Large vision-language models have recently demonstrated impressive performance in planning and control tasks, driving interest in their application to real-world robotics. However, deploying these models for reasoning in embodied contexts is limited by their ability to incorporate long-term experience collected across multiple days and represented by vast collections of images. Current VLMs typically struggle to process more than a few hundred images concurrently, highlighting the need for more efficient mechanisms to handle long-term memory in embodied settings. To effectively evaluate these models for long-horizon control, a benchmark must specifically target scenarios where memory is crucial for success. Existing long-video QA benchmarks overlook embodied challenges like object manipulation and navigation, which demand low-level skills and fine-grained reasoning over past interactions. Moreover, effective memory integration in embodied agents involves both recalling relevant historical information and executing actions based on that information, making it essential to study these aspects together rather than in isolation. In this work, we introduce a new benchmark for long-range embodied tasks in the Habitat simulator. This benchmark evaluates memory-based capabilities across 60 tasks requiring sustained engagement and contextual awareness in an environment. The tasks can also be procedurally extended to longer and more challenging versions, enabling scalable evaluation of memory and reasoning. We also present baselines that integrate state-of-the-art VLMs with low level navigation policies, assessing their performance on these memory-intensive tasks and highlight areas for improvement.
>
---
#### [new 015] SemIRNet: A Semantic Irony Recognition Network for Multimodal Sarcasm Detection
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态讽刺检测任务，旨在解决图形与文本隐含关联识别困难的问题。提出SemIRNet模型，融合知识库、设计语义相似度模块并引入对比损失函数，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.14791v1](http://arxiv.org/pdf/2506.14791v1)**

> **作者:** Jingxuan Zhou; Yuehao Wu; Yibo Zhang; Yeyubei Zhang; Yunchong Liu; Bolin Huang; Chunhong Yuan
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Aiming at the problem of difficulty in accurately identifying graphical implicit correlations in multimodal irony detection tasks, this paper proposes a Semantic Irony Recognition Network (SemIRNet). The model contains three main innovations: (1) The ConceptNet knowledge base is introduced for the first time to acquire conceptual knowledge, which enhances the model's common-sense reasoning ability; (2) Two cross-modal semantic similarity detection modules at the word level and sample level are designed to model graphic-textual correlations at different granularities; and (3) A contrastive learning loss function is introduced to optimize the spatial distribution of the sample features, which improves the separability of positive and negative samples. Experiments on a publicly available multimodal irony detection benchmark dataset show that the accuracy and F1 value of this model are improved by 1.64% and 2.88% to 88.87% and 86.33%, respectively, compared with the existing optimal methods. Further ablation experiments verify the important role of knowledge fusion and semantic similarity detection in improving the model performance.
>
---
#### [new 016] RA-NeRF: Robust Neural Radiance Field Reconstruction with Accurate Camera Pose Estimation under Complex Trajectories
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决复杂轨迹下相机位姿估计不准确的问题。提出RA-NeRF方法，结合NeRF和流驱动调节，提升重建精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.15242v1](http://arxiv.org/pdf/2506.15242v1)**

> **作者:** Qingsong Yan; Qiang Wang; Kaiyong Zhao; Jie Chen; Bo Li; Xiaowen Chu; Fei Deng
>
> **备注:** IROS 2025
>
> **摘要:** Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have emerged as powerful tools for 3D reconstruction and SLAM tasks. However, their performance depends heavily on accurate camera pose priors. Existing approaches attempt to address this issue by introducing external constraints but fall short of achieving satisfactory accuracy, particularly when camera trajectories are complex. In this paper, we propose a novel method, RA-NeRF, capable of predicting highly accurate camera poses even with complex camera trajectories. Following the incremental pipeline, RA-NeRF reconstructs the scene using NeRF with photometric consistency and incorporates flow-driven pose regulation to enhance robustness during initialization and localization. Additionally, RA-NeRF employs an implicit pose filter to capture the camera movement pattern and eliminate the noise for pose estimation. To validate our method, we conduct extensive experiments on the Tanks\&Temple dataset for standard evaluation, as well as the NeRFBuster dataset, which presents challenging camera pose trajectories. On both datasets, RA-NeRF achieves state-of-the-art results in both camera pose estimation and visual quality, demonstrating its effectiveness and robustness in scene reconstruction under complex pose trajectories.
>
---
#### [new 017] AI-driven visual monitoring of industrial assembly tasks
- **分类: cs.CV**

- **简介: 该论文属于工业装配任务的视觉监控领域，旨在解决传统方法依赖固定环境和标记的问题。提出ViMAT系统，结合感知与推理模块，实现实时无约束监控。**

- **链接: [http://arxiv.org/pdf/2506.15285v1](http://arxiv.org/pdf/2506.15285v1)**

> **作者:** Mattia Nardon; Stefano Messelodi; Antonio Granata; Fabio Poiesi; Alberto Danese; Davide Boscaini
>
> **摘要:** Visual monitoring of industrial assembly tasks is critical for preventing equipment damage due to procedural errors and ensuring worker safety. Although commercial solutions exist, they typically require rigid workspace setups or the application of visual markers to simplify the problem. We introduce ViMAT, a novel AI-driven system for real-time visual monitoring of assembly tasks that operates without these constraints. ViMAT combines a perception module that extracts visual observations from multi-view video streams with a reasoning module that infers the most likely action being performed based on the observed assembly state and prior task knowledge. We validate ViMAT on two assembly tasks, involving the replacement of LEGO components and the reconfiguration of hydraulic press molds, demonstrating its effectiveness through quantitative and qualitative analysis in challenging real-world scenarios characterized by partial and uncertain visual observations. Project page: https://tev-fbk.github.io/ViMAT
>
---
#### [new 018] Hyper-Local Deformable Transformers for Text Spotting on Historical Maps
- **分类: cs.CV**

- **简介: 该论文属于文本检测任务，旨在解决历史地图中文本提取困难的问题。提出PALETTE模型和SynthMap+数据生成方法，提升长文本和倾斜文本的识别效果。**

- **链接: [http://arxiv.org/pdf/2506.15010v1](http://arxiv.org/pdf/2506.15010v1)**

> **作者:** Yijun Lin; Yao-Yi Chiang
>
> **备注:** Published in KDD2024
>
> **摘要:** Text on historical maps contains valuable information providing georeferenced historical, political, and cultural contexts. However, text extraction from historical maps is challenging due to the lack of (1) effective methods and (2) training data. Previous approaches use ad-hoc steps tailored to only specific map styles. Recent machine learning-based text spotters (e.g., for scene images) have the potential to solve these challenges because of their flexibility in supporting various types of text instances. However, these methods remain challenges in extracting precise image features for predicting every sub-component (boundary points and characters) in a text instance. This is critical because map text can be lengthy and highly rotated with complex backgrounds, posing difficulties in detecting relevant image features from a rough text region. This paper proposes PALETTE, an end-to-end text spotter for scanned historical maps of a wide variety. PALETTE introduces a novel hyper-local sampling module to explicitly learn localized image features around the target boundary points and characters of a text instance for detection and recognition. PALETTE also enables hyper-local positional embeddings to learn spatial interactions between boundary points and characters within and across text instances. In addition, this paper presents a novel approach to automatically generate synthetic map images, SynthMap+, for training text spotters for historical maps. The experiment shows that PALETTE with SynthMap+ outperforms SOTA text spotters on two new benchmark datasets of historical maps, particularly for long and angled text. We have deployed PALETTE with SynthMap+ to process over 60,000 maps in the David Rumsey Historical Map collection and generated over 100 million text labels to support map searching. The project is released at https://github.com/kartta-foundation/mapkurator-palette-doc.
>
---
#### [new 019] Multimodal Large Language Models for Medical Report Generation via Customized Prompt Tuning
- **分类: cs.CV**

- **简介: 该论文属于医学报告生成任务，旨在解决如何有效整合医疗影像与大语言模型的问题。通过定制化提示调优方法，提升报告生成效果。**

- **链接: [http://arxiv.org/pdf/2506.15477v1](http://arxiv.org/pdf/2506.15477v1)**

> **作者:** Chunlei Li; Jingyang Hou; Yilei Shi; Jingliang Hu; Xiao Xiang Zhu; Lichao Mou
>
> **摘要:** Medical report generation from imaging data remains a challenging task in clinical practice. While large language models (LLMs) show great promise in addressing this challenge, their effective integration with medical imaging data still deserves in-depth exploration. In this paper, we present MRG-LLM, a novel multimodal large language model (MLLM) that combines a frozen LLM with a learnable visual encoder and introduces a dynamic prompt customization mechanism. Our key innovation lies in generating instance-specific prompts tailored to individual medical images through conditional affine transformations derived from visual features. We propose two implementations: prompt-wise and promptbook-wise customization, enabling precise and targeted report generation. Extensive experiments on IU X-ray and MIMIC-CXR datasets demonstrate that MRG-LLM achieves state-of-the-art performance in medical report generation. Our code will be made publicly available.
>
---
#### [new 020] GraphGSOcc: Semantic and Geometric Graph Transformer for 3D Gaussian Splating-based Occupancy Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D语义占据预测任务，解决3DGS方法中特征聚合不足和边界模糊问题，提出GraphGSOcc模型结合语义与几何图Transformer提升精度并降低内存消耗。**

- **链接: [http://arxiv.org/pdf/2506.14825v1](http://arxiv.org/pdf/2506.14825v1)**

> **作者:** Ke Song; Yunhe Wu; Chunchit Siu; Huiyuan Xiong
>
> **摘要:** Addressing the task of 3D semantic occupancy prediction for autonomous driving, we tackle two key issues in existing 3D Gaussian Splating (3DGS) methods: (1) unified feature aggregation neglecting semantic correlations among similar categories and across regions, and (2) boundary ambiguities caused by the lack of geometric constraints in MLP iterative optimization. We propose the GraphGSOcc model, a novel framework that combines semantic and geometric graph Transformer for 3D Gaussian Splating-based Occupancy Prediction. We propose the Dual Gaussians Graph Attenntion, which dynamically constructs dual graph structures: a geometric graph adaptively calculating KNN search radii based on Gaussian poses, enabling large-scale Gaussians to aggregate features from broader neighborhoods while compact Gaussians focus on local geometric consistency; a semantic graph retaining top-M highly correlated nodes via cosine similarity to explicitly encode semantic relationships within and across instances. Coupled with the Multi-scale Graph Attention framework, fine-grained attention at lower layers optimizes boundary details, while coarse-grained attention at higher layers models object-level topology. Experiments on the SurroundOcc dataset achieve an mIoU of 24.10%, reducing GPU memory to 6.1 GB, demonstrating a 1.97% mIoU improvement and 13.7% memory reduction compared to GaussianWorld
>
---
#### [new 021] A Hybrid ConvNeXt-EfficientNet AI Solution for Precise Falcon Disease Detection
- **分类: cs.CV**

- **简介: 该论文属于疾病分类任务，旨在精准检测猎隼的三种疾病。通过融合ConvNeXt和EfficientNet模型提升诊断效果。**

- **链接: [http://arxiv.org/pdf/2506.14816v1](http://arxiv.org/pdf/2506.14816v1)**

> **作者:** Alavikunhu Panthakkan; Zubair Medammal; S M Anzar; Fatma Taher; Hussain Al-Ahmad
>
> **摘要:** Falconry, a revered tradition involving the training and hunting with falcons, requires meticulous health surveillance to ensure the health and safety of these prized birds, particularly in hunting scenarios. This paper presents an innovative method employing a hybrid of ConvNeXt and EfficientNet AI models for the classification of falcon diseases. The study focuses on accurately identifying three conditions: Normal, Liver Disease and 'Aspergillosis'. A substantial dataset was utilized for training and validating the model, with an emphasis on key performance metrics such as accuracy, precision, recall, and F1-score. Extensive testing and analysis have shown that our concatenated AI model outperforms traditional diagnostic methods and individual model architectures. The successful implementation of this hybrid AI model marks a significant step forward in precise falcon disease detection and paves the way for future developments in AI-powered avian healthcare solutions.
>
---
#### [new 022] PeRL: Permutation-Enhanced Reinforcement Learning for Interleaved Vision-Language Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态视觉-语言推理任务，旨在解决多图像位置关系理解问题。提出PeRL方法，通过序列置换和轨迹筛选提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.14907v1](http://arxiv.org/pdf/2506.14907v1)**

> **作者:** Yizhen Zhang; Yang Ding; Shuoshuo Zhang; Xinchen Zhang; Haoling Li; Zhong-zhi Li; Peijie Wang; Jie Wu; Lei Ji; Yelong Shen; Yujiu Yang; Yeyun Gong
>
> **摘要:** Inspired by the impressive reasoning capabilities demonstrated by reinforcement learning approaches like DeepSeek-R1, recent emerging research has begun exploring the use of reinforcement learning (RL) to enhance vision-language models (VLMs) for multimodal reasoning tasks. However, most existing multimodal reinforcement learning approaches remain limited to spatial reasoning within single-image contexts, yet still struggle to generalize to more complex and real-world scenarios involving multi-image positional reasoning, where understanding the relationships across images is crucial. To address this challenge, we propose a general reinforcement learning approach PeRL tailored for interleaved multimodal tasks, and a multi-stage strategy designed to enhance the exploration-exploitation trade-off, thereby improving learning efficiency and task performance. Specifically, we introduce permutation of image sequences to simulate varied positional relationships to explore more spatial and positional diversity. Furthermore, we design a rollout filtering mechanism for resampling to focus on trajectories that contribute most to learning optimal behaviors to exploit learned policies effectively. We evaluate our model on 5 widely-used multi-image benchmarks and 3 single-image benchmarks. Our experiments confirm that PeRL trained model consistently surpasses R1-related and interleaved VLM baselines by a large margin, achieving state-of-the-art performance on multi-image benchmarks, while preserving comparable performance on single-image tasks.
>
---
#### [new 023] BoxFusion: Reconstruction-Free Open-Vocabulary 3D Object Detection via Real-Time Multi-View Box Fusion
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决传统方法依赖点云重建导致的计算和内存瓶颈问题。提出一种无需重建的实时多视角框融合方法，提升检测效率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.15610v1](http://arxiv.org/pdf/2506.15610v1)**

> **作者:** Yuqing Lan; Chenyang Zhu; Zhirui Gao; Jiazhao Zhang; Yihan Cao; Renjiao Yi; Yijie Wang; Kai Xu
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Open-vocabulary 3D object detection has gained significant interest due to its critical applications in autonomous driving and embodied AI. Existing detection methods, whether offline or online, typically rely on dense point cloud reconstruction, which imposes substantial computational overhead and memory constraints, hindering real-time deployment in downstream tasks. To address this, we propose a novel reconstruction-free online framework tailored for memory-efficient and real-time 3D detection. Specifically, given streaming posed RGB-D video input, we leverage Cubify Anything as a pre-trained visual foundation model (VFM) for single-view 3D object detection by bounding boxes, coupled with CLIP to capture open-vocabulary semantics of detected objects. To fuse all detected bounding boxes across different views into a unified one, we employ an association module for correspondences of multi-views and an optimization module to fuse the 3D bounding boxes of the same instance predicted in multi-views. The association module utilizes 3D Non-Maximum Suppression (NMS) and a box correspondence matching module, while the optimization module uses an IoU-guided efficient random optimization technique based on particle filtering to enforce multi-view consistency of the 3D bounding boxes while minimizing computational complexity. Extensive experiments on ScanNetV2 and CA-1M datasets demonstrate that our method achieves state-of-the-art performance among online methods. Benefiting from this novel reconstruction-free paradigm for 3D object detection, our method exhibits great generalization abilities in various scenarios, enabling real-time perception even in environments exceeding 1000 square meters.
>
---
#### [new 024] Improved Iterative Refinement for Chart-to-Code Generation via Structured Instruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图表到代码生成任务，旨在解决MLLM在该任务中表现不佳的问题。通过结构化指令和迭代优化提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.14837v1](http://arxiv.org/pdf/2506.14837v1)**

> **作者:** Chengzhi Xu; Yuyang Wang; Lai Wei; Lichao Sun; Weiran Huang
>
> **摘要:** Recently, multimodal large language models (MLLMs) have attracted increasing research attention due to their powerful visual understanding capabilities. While they have achieved impressive results on various vision tasks, their performance on chart-to-code generation remains suboptimal. This task requires MLLMs to generate executable code that can reproduce a given chart, demanding not only precise visual understanding but also accurate translation of visual elements into structured code. Directly prompting MLLMs to perform this complex task often yields unsatisfactory results. To address this challenge, we propose {ChartIR}, an iterative refinement method based on structured instruction. First, we distinguish two tasks: visual understanding and code translation. To accomplish the visual understanding component, we design two types of structured instructions: description and difference. The description instruction captures the visual elements of the reference chart, while the difference instruction characterizes the discrepancies between the reference chart and the generated chart. These instructions effectively transform visual features into language representations, thereby facilitating the subsequent code translation process. Second, we decompose the overall chart generation pipeline into two stages: initial code generation and iterative refinement, enabling progressive enhancement of the final output. Experimental results show that, compared to other method, our method achieves superior performance on both the open-source model Qwen2-VL and the closed-source model GPT-4o.
>
---
#### [new 025] When Model Knowledge meets Diffusion Model: Diffusion-assisted Data-free Image Synthesis with Alignment of Domain and Class
- **分类: cs.CV**

- **简介: 该论文属于数据无关图像生成任务，旨在解决DFIS中生成图像偏离训练数据分布的问题。通过引入扩散模型作为先验，结合领域和类别对齐策略，提升生成质量。**

- **链接: [http://arxiv.org/pdf/2506.15381v1](http://arxiv.org/pdf/2506.15381v1)**

> **作者:** Yujin Kim; Hyunsoo Kim; Hyunwoo J. Kim; Suhyun Kim
>
> **备注:** Published at ICML 2025
>
> **摘要:** Open-source pre-trained models hold great potential for diverse applications, but their utility declines when their training data is unavailable. Data-Free Image Synthesis (DFIS) aims to generate images that approximate the learned data distribution of a pre-trained model without accessing the original data. However, existing DFIS meth ods produce samples that deviate from the training data distribution due to the lack of prior knowl edge about natural images. To overcome this limitation, we propose DDIS, the first Diffusion-assisted Data-free Image Synthesis method that leverages a text-to-image diffusion model as a powerful image prior, improving synthetic image quality. DDIS extracts knowledge about the learned distribution from the given model and uses it to guide the diffusion model, enabling the generation of images that accurately align with the training data distribution. To achieve this, we introduce Domain Alignment Guidance (DAG) that aligns the synthetic data domain with the training data domain during the diffusion sampling process. Furthermore, we optimize a single Class Alignment Token (CAT) embedding to effectively capture class-specific attributes in the training dataset. Experiments on PACS and Ima geNet demonstrate that DDIS outperforms prior DFIS methods by generating samples that better reflect the training data distribution, achieving SOTA performance in data-free applications.
>
---
#### [new 026] Conquering the Retina: Bringing Visual in-Context Learning to OCT
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决OCT图像中通用模型的少样本泛化问题，通过视觉上下文学习方法进行训练与评估。**

- **链接: [http://arxiv.org/pdf/2506.15200v1](http://arxiv.org/pdf/2506.15200v1)**

> **作者:** Alessio Negrini; Simon Reiß
>
> **摘要:** Recent advancements in medical image analysis have led to the development of highly specialized models tailored to specific clinical tasks. These models have demonstrated exceptional performance and remain a crucial research direction. Yet, their applicability is limited to predefined tasks, requiring expertise and extensive resources for development and adaptation. In contrast, generalist models offer a different form of utility: allowing medical practitioners to define tasks on the fly without the need for task-specific model development. In this work, we explore how to train generalist models for the domain of retinal optical coherence tomography using visual in-context learning (VICL), i.e., training models to generalize across tasks based on a few examples provided at inference time. To facilitate rigorous assessment, we propose a broad evaluation protocol tailored to VICL in OCT. We extensively evaluate a state-of-the-art medical VICL approach on multiple retinal OCT datasets, establishing a first baseline to highlight the potential and current limitations of in-context learning for OCT. To foster further research and practical adoption, we openly release our code.
>
---
#### [new 027] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D重建任务，解决主动视角选择问题。通过神经不确定性图预测有效视角，提升重建效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.14856v1](http://arxiv.org/pdf/2506.14856v1)**

> **作者:** Zhengquan Zhang; Feng Xu; Mengmi Zhang
>
> **备注:** 9 pages, 3 figures in the main text. Under review for NeurIPS 2025
>
> **摘要:** Some perspectives naturally provide more information than others. How can an AI system determine which viewpoint offers the most valuable insight for accurate and efficient 3D object reconstruction? Active view selection (AVS) for 3D reconstruction remains a fundamental challenge in computer vision. The aim is to identify the minimal set of views that yields the most accurate 3D reconstruction. Instead of learning radiance fields, like NeRF or 3D Gaussian Splatting, from a current observation and computing uncertainty for each candidate viewpoint, we introduce a novel AVS approach guided by neural uncertainty maps predicted by a lightweight feedforward deep neural network, named UPNet. UPNet takes a single input image of a 3D object and outputs a predicted uncertainty map, representing uncertainty values across all possible candidate viewpoints. By leveraging heuristics derived from observing many natural objects and their associated uncertainty patterns, we train UPNet to learn a direct mapping from viewpoint appearance to uncertainty in the underlying volumetric representations. Next, our approach aggregates all previously predicted neural uncertainty maps to suppress redundant candidate viewpoints and effectively select the most informative one. Using these selected viewpoints, we train 3D neural rendering models and evaluate the quality of novel view synthesis against other competitive AVS methods. Remarkably, despite using half of the viewpoints than the upper bound, our method achieves comparable reconstruction accuracy. In addition, it significantly reduces computational overhead during AVS, achieving up to a 400 times speedup along with over 50\% reductions in CPU, RAM, and GPU usage compared to baseline methods. Notably, our approach generalizes effectively to AVS tasks involving novel object categories, without requiring any additional training.
>
---
#### [new 028] NTIRE 2025 Image Shadow Removal Challenge Report
- **分类: cs.CV**

- **简介: 该论文报告了NTIRE 2025图像去阴影挑战，旨在解决自阴影与投射阴影的去除问题，通过两个评估轨道验证方法效果。**

- **链接: [http://arxiv.org/pdf/2506.15524v1](http://arxiv.org/pdf/2506.15524v1)**

> **作者:** Florin-Alexandru Vasluianu; Tim Seizinger; Zhuyun Zhou; Cailian Chen; Zongwei Wu; Radu Timofte; Mingjia Li; Jin Hu; Hainuo Wang; Hengxing Liu; Jiarui Wang; Qiming Hu; Xiaojie Guo; Xin Lu; Jiarong Yang; Yuanfei Bao; Anya Hu; Zihao Fan; Kunyu Wang; Jie Xiao; Xi Wang; Xueyang Fu; Zheng-Jun Zha; Yu-Fan Lin; Chia-Ming Lee; Chih-Chung Hsu; Xingbo Wang; Dong Li; Yuxu Chen; Bin Chen; Yuanbo Zhou; Yuanbin Chen; Hongwei Wang; Jiannan Lin; Qinquan Gao; Tong Tong; Zhao Zhang; Yanyan Wei; Wei Dong; Han Zhou; Seyed Amirreza Mousavi; Jun Chen; Haobo Liang; Jiajie Jing; Junyu Li; Yan Yang; Seoyeon Lee; Chaewon Kim; Ziyu Feng; Shidi Chen; Bowen Luan; Zewen Chen; Vijayalaxmi Ashok Aralikatti; G Gyaneshwar Rao; Nikhil Akalwadi; Chaitra Desai; Ramesh Ashok Tabib; Uma Mudenagudi; Anas M. Ali; Bilel Benjdira; Wadii Boulila; Alexandru Brateanu; Cosmin Ancuti; Tanmay Chaturvedi; Manish Kumar; Anmol Srivastav; Daksh Trivedi; Shashwat Thakur; Kishor Upla; Zeyu Xiao; Zhuoyuan Li; Boda Zhou; Shashank Shekhar; Kele Xu; Qisheng Xu; Zijian Gao; Tianjiao Wan; Suiyi Zhao; Bo Wang; Yan Luo; Mingshen Wang; Yilin Zhang
>
> **摘要:** This work examines the findings of the NTIRE 2025 Shadow Removal Challenge. A total of 306 participants have registered, with 17 teams successfully submitting their solutions during the final evaluation phase. Following the last two editions, this challenge had two evaluation tracks: one focusing on reconstruction fidelity and the other on visual perception through a user study. Both tracks were evaluated with images from the WSRD+ dataset, simulating interactions between self- and cast-shadows with a large number of diverse objects, textures, and materials.
>
---
#### [new 029] video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models
- **分类: cs.CV; cs.CL; cs.SD**

- **简介: 该论文属于视频描述生成任务，旨在提升视频字幕的准确性。通过改进的DPO方法和LoRA技术，优化模型性能，显著降低错误率。**

- **链接: [http://arxiv.org/pdf/2506.15220v1](http://arxiv.org/pdf/2506.15220v1)**

> **作者:** Changli Tang; Yixuan Li; Yudong Yang; Jimin Zhuang; Guangzhi Sun; Wei Li; Zejun Ma; Chao Zhang
>
> **摘要:** Videos contain a wealth of information, and generating detailed and accurate descriptions in natural language is a key aspect of video understanding. In this paper, we present video-SALMONN 2, an advanced audio-visual large language model (LLM) with low-rank adaptation (LoRA) designed for enhanced video (with paired audio) captioning through directed preference optimisation (DPO). We propose new metrics to evaluate the completeness and accuracy of video descriptions, which are optimised using DPO. To further improve training, we propose a novel multi-round DPO (MrDPO) approach, which involves periodically updating the DPO reference model, merging and re-initialising the LoRA module as a proxy for parameter updates after each training round (1,000 steps), and incorporating guidance from ground-truth video captions to stabilise the process. Experimental results show that MrDPO significantly enhances video-SALMONN 2's captioning accuracy, reducing the captioning error rates by 28\%. The final video-SALMONN 2 model, with just 7 billion parameters, surpasses leading models such as GPT-4o and Gemini-1.5-Pro in video captioning tasks, while maintaining highly competitive performance to the state-of-the-art on widely used video question-answering benchmarks among models of similar size. Codes are available at \href{https://github.com/bytedance/video-SALMONN-2}{https://github.com/bytedance/video-SALMONN-2}.
>
---
#### [new 030] Enhancing point cloud analysis via neighbor aggregation correction based on cross-stage structure correlation
- **分类: cs.CV**

- **简介: 该论文属于点云分析任务，旨在解决邻域聚合中的干扰和层次差距问题。提出PDSA模块，通过高维相关性校正特征分布，提升效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.15160v1](http://arxiv.org/pdf/2506.15160v1)**

> **作者:** Jiaqi Shi; Jin Xiao; Xiaoguang Hu; Boyang Song; Hao Jiang; Tianyou Chen; Baochang Zhang
>
> **备注:** 17 papes, 7 figures
>
> **摘要:** Point cloud analysis is the cornerstone of many downstream tasks, among which aggregating local structures is the basis for understanding point cloud data. While numerous works aggregate neighbor using three-dimensional relative coordinates, there are irrelevant point interference and feature hierarchy gap problems due to the limitation of local coordinates. Although some works address this limitation by refining spatial description though explicit modeling of cross-stage structure, these enhancement methods based on direct geometric structure encoding have problems of high computational overhead and noise sensitivity. To overcome these problems, we propose the Point Distribution Set Abstraction module (PDSA) that utilizes the correlation in the high-dimensional space to correct the feature distribution during aggregation, which improves the computational efficiency and robustness. PDSA distinguishes the point correlation based on a lightweight cross-stage structural descriptor, and enhances structural homogeneity by reducing the variance of the neighbor feature matrix and increasing classes separability though long-distance modeling. Additionally, we introducing a key point mechanism to optimize the computational overhead. The experimental result on semantic segmentation and classification tasks based on different baselines verify the generalization of the method we proposed, and achieve significant performance improvement with less parameter cost. The corresponding ablation and visualization results demonstrate the effectiveness and rationality of our method. The code and training weight is available at: https://github.com/AGENT9717/PointDistribution
>
---
#### [new 031] UniRelight: Learning Joint Decomposition and Synthesis for Video Relighting
- **分类: cs.CV**

- **简介: 该论文属于视频重光照任务，旨在解决单图像或视频重光照中的数据不足与效果不真实问题。提出联合分解与合成方法，提升光照效果和材质表现。**

- **链接: [http://arxiv.org/pdf/2506.15673v1](http://arxiv.org/pdf/2506.15673v1)**

> **作者:** Kai He; Ruofan Liang; Jacob Munkberg; Jon Hasselgren; Nandita Vijaykumar; Alexander Keller; Sanja Fidler; Igor Gilitschenski; Zan Gojcic; Zian Wang
>
> **备注:** Project page: https://research.nvidia.com/labs/toronto-ai/UniRelight/
>
> **摘要:** We address the challenge of relighting a single image or video, a task that demands precise scene intrinsic understanding and high-quality light transport synthesis. Existing end-to-end relighting models are often limited by the scarcity of paired multi-illumination data, restricting their ability to generalize across diverse scenes. Conversely, two-stage pipelines that combine inverse and forward rendering can mitigate data requirements but are susceptible to error accumulation and often fail to produce realistic outputs under complex lighting conditions or with sophisticated materials. In this work, we introduce a general-purpose approach that jointly estimates albedo and synthesizes relit outputs in a single pass, harnessing the generative capabilities of video diffusion models. This joint formulation enhances implicit scene comprehension and facilitates the creation of realistic lighting effects and intricate material interactions, such as shadows, reflections, and transparency. Trained on synthetic multi-illumination data and extensive automatically labeled real-world videos, our model demonstrates strong generalization across diverse domains and surpasses previous methods in both visual fidelity and temporal consistency.
>
---
#### [new 032] Convolutional Feature Enhancement and Attention Fusion BiFPN for Ship Detection in SAR Images
- **分类: cs.CV**

- **简介: 该论文属于SAR图像中的船舶检测任务，旨在解决尺度变化大、小船混杂噪声和复杂背景等问题。提出C-AFBiFPN框架，融合卷积增强与注意力机制提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.15231v1](http://arxiv.org/pdf/2506.15231v1)**

> **作者:** Liangjie Meng; Danxia Li; Jinrong He; Lili Ma; Zhixin Li
>
> **备注:** 5 pages, 4 figures, 2 tables. Code available at https://github.com/mlj666219/C-AFBiFPN/tree/master
>
> **摘要:** Synthetic Aperture Radar (SAR) enables submeter-resolution imaging and all-weather monitoring via active microwave and advanced signal processing. Currently, SAR has found extensive applications in critical maritime domains such as ship detection. However, SAR ship detection faces several challenges, including significant scale variations among ships, the presence of small offshore vessels mixed with noise, and complex backgrounds for large nearshore ships. To address these issues, this paper proposes a novel feature enhancement and fusion framework named C-AFBiFPN. C-AFBiFPN constructs a Convolutional Feature Enhancement (CFE) module following the backbone network, aiming to enrich feature representation and enhance the ability to capture and represent local details and contextual information. Furthermore, C-AFBiFPN innovatively integrates BiFormer attention within the fusion strategy of BiFPN, creating the AFBiFPN network. AFBiFPN improves the global modeling capability of cross-scale feature fusion and can adaptively focus on critical feature regions. The experimental results on SAR Ship Detection Dataset (SSDD) indicate that the proposed approach substantially enhances detection accuracy for small targets, robustness against occlusions, and adaptability to multi-scale features.
>
---
#### [new 033] HOIDiNi: Human-Object Interaction through Diffusion Noise Optimization
- **分类: cs.CV**

- **简介: 该论文属于人-物交互生成任务，解决如何合成真实且物理合理的交互动作。通过扩散噪声优化方法，在预训练模型的噪声空间中直接优化，提升接触准确性和运动自然度。**

- **链接: [http://arxiv.org/pdf/2506.15625v1](http://arxiv.org/pdf/2506.15625v1)**

> **作者:** Roey Ron; Guy Tevet; Haim Sawdayee; Amit H. Bermano
>
> **备注:** Project page: https://hoidini.github.io
>
> **摘要:** We present HOIDiNi, a text-driven diffusion framework for synthesizing realistic and plausible human-object interaction (HOI). HOI generation is extremely challenging since it induces strict contact accuracies alongside a diverse motion manifold. While current literature trades off between realism and physical correctness, HOIDiNi optimizes directly in the noise space of a pretrained diffusion model using Diffusion Noise Optimization (DNO), achieving both. This is made feasible thanks to our observation that the problem can be separated into two phases: an object-centric phase, primarily making discrete choices of hand-object contact locations, and a human-centric phase that refines the full-body motion to realize this blueprint. This structured approach allows for precise hand-object contact without compromising motion naturalness. Quantitative, qualitative, and subjective evaluations on the GRAB dataset alone clearly indicate HOIDiNi outperforms prior works and baselines in contact accuracy, physical validity, and overall quality. Our results demonstrate the ability to generate complex, controllable interactions, including grasping, placing, and full-body coordination, driven solely by textual prompts. https://hoidini.github.io.
>
---
#### [new 034] Domain Adaptation for Image Classification of Defects in Semiconductor Manufacturing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分类中的领域自适应任务，旨在解决半导体缺陷检测中模型迁移效率低的问题，提出DBACS方法提升性能。**

- **链接: [http://arxiv.org/pdf/2506.15260v1](http://arxiv.org/pdf/2506.15260v1)**

> **作者:** Adrian Poniatowski; Natalie Gentner; Manuel Barusco; Davide Dalle Pezze; Samuele Salti; Gian Antonio Susto
>
> **摘要:** In the semiconductor sector, due to high demand but also strong and increasing competition, time to market and quality are key factors in securing significant market share in various application areas. Thanks to the success of deep learning methods in recent years in the computer vision domain, Industry 4.0 and 5.0 applications, such as defect classification, have achieved remarkable success. In particular, Domain Adaptation (DA) has proven highly effective since it focuses on using the knowledge learned on a (source) domain to adapt and perform effectively on a different but related (target) domain. By improving robustness and scalability, DA minimizes the need for extensive manual re-labeling or re-training of models. This not only reduces computational and resource costs but also allows human experts to focus on high-value tasks. Therefore, we tested the efficacy of DA techniques in semi-supervised and unsupervised settings within the context of the semiconductor field. Moreover, we propose the DBACS approach, a CycleGAN-inspired model enhanced with additional loss terms to improve performance. All the approaches are studied and validated on real-world Electron Microscope images considering the unsupervised and semi-supervised settings, proving the usefulness of our method in advancing DA techniques for the semiconductor field.
>
---
#### [new 035] One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频超分辨率任务，旨在解决真实视频中细节丰富与时间一致性难以兼顾的问题。通过提出DLoRAL框架，实现高效高质量的视频修复。**

- **链接: [http://arxiv.org/pdf/2506.15591v1](http://arxiv.org/pdf/2506.15591v1)**

> **作者:** Yujing Sun; Lingchen Sun; Shuaizheng Liu; Rongyuan Wu; Zhengqiang Zhang; Lei Zhang
>
> **摘要:** It is a challenging problem to reproduce rich spatial details while maintaining temporal consistency in real-world video super-resolution (Real-VSR), especially when we leverage pre-trained generative models such as stable diffusion (SD) for realistic details synthesis. Existing SD-based Real-VSR methods often compromise spatial details for temporal coherence, resulting in suboptimal visual quality. We argue that the key lies in how to effectively extract the degradation-robust temporal consistency priors from the low-quality (LQ) input video and enhance the video details while maintaining the extracted consistency priors. To achieve this, we propose a Dual LoRA Learning (DLoRAL) paradigm to train an effective SD-based one-step diffusion model, achieving realistic frame details and temporal consistency simultaneously. Specifically, we introduce a Cross-Frame Retrieval (CFR) module to aggregate complementary information across frames, and train a Consistency-LoRA (C-LoRA) to learn robust temporal representations from degraded inputs. After consistency learning, we fix the CFR and C-LoRA modules and train a Detail-LoRA (D-LoRA) to enhance spatial details while aligning with the temporal space defined by C-LoRA to keep temporal coherence. The two phases alternate iteratively for optimization, collaboratively delivering consistent and detail-rich outputs. During inference, the two LoRA branches are merged into the SD model, allowing efficient and high-quality video restoration in a single diffusion step. Experiments show that DLoRAL achieves strong performance in both accuracy and speed. Code and models are available at https://github.com/yjsunnn/DLoRAL.
>
---
#### [new 036] CLAIM: Clinically-Guided LGE Augmentation for Realistic and Diverse Myocardial Scar Synthesis and Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决LGE图像稀缺和标注不一致的问题。提出CLAIM框架，结合临床知识生成逼真心肌瘢痕，并提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.15549v1](http://arxiv.org/pdf/2506.15549v1)**

> **作者:** Farheen Ramzan; Yusuf Kiberu; Nikesh Jathanna; Shahnaz Jamil-Copley; Richard H. Clayton; Chen; Chen
>
> **备注:** 14 Pages
>
> **摘要:** Deep learning-based myocardial scar segmentation from late gadolinium enhancement (LGE) cardiac MRI has shown great potential for accurate and timely diagnosis and treatment planning for structural cardiac diseases. However, the limited availability and variability of LGE images with high-quality scar labels restrict the development of robust segmentation models. To address this, we introduce CLAIM: \textbf{C}linically-Guided \textbf{L}GE \textbf{A}ugmentation for Real\textbf{i}stic and Diverse \textbf{M}yocardial Scar Synthesis and Segmentation framework, a framework for anatomically grounded scar generation and segmentation. At its core is the SMILE module (Scar Mask generation guided by cLinical knowledgE), which conditions a diffusion-based generator on the clinically adopted AHA 17-segment model to synthesize images with anatomically consistent and spatially diverse scar patterns. In addition, CLAIM employs a joint training strategy in which the scar segmentation network is optimized alongside the generator, aiming to enhance both the realism of synthesized scars and the accuracy of the scar segmentation performance. Experimental results show that CLAIM produces anatomically coherent scar patterns and achieves higher Dice similarity with real scar distributions compared to baseline models. Our approach enables controllable and realistic myocardial scar synthesis and has demonstrated utility for downstream medical imaging task.
>
---
#### [new 037] Finding Optimal Kernel Size and Dimension in Convolutional Neural Networks An Architecture Optimization Approach
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于卷积神经网络优化任务，旨在解决核尺寸选择问题。通过提出BKSEF框架，实现层间核尺寸的最优设计，提升模型性能与效率。**

- **链接: [http://arxiv.org/pdf/2506.14846v1](http://arxiv.org/pdf/2506.14846v1)**

> **作者:** Shreyas Rajeev; B Sathish Babu
>
> **摘要:** Kernel size selection in Convolutional Neural Networks (CNNs) is a critical but often overlooked design decision that affects receptive field, feature extraction, computational cost, and model accuracy. This paper proposes the Best Kernel Size Estimation Function (BKSEF), a mathematically grounded and empirically validated framework for optimal, layer-wise kernel size determination. BKSEF balances information gain, computational efficiency, and accuracy improvements by integrating principles from information theory, signal processing, and learning theory. Extensive experiments on CIFAR-10, CIFAR-100, ImageNet-lite, ChestX-ray14, and GTSRB datasets demonstrate that BKSEF-guided architectures achieve up to 3.1 percent accuracy improvement and 42.8 percent reduction in FLOPs compared to traditional models using uniform 3x3 kernels. Two real-world case studies further validate the approach: one for medical image classification in a cloud-based setup, and another for traffic sign recognition on edge devices. The former achieved enhanced interpretability and accuracy, while the latter reduced latency and model size significantly, with minimal accuracy trade-off. These results show that kernel size can be an active, optimizable parameter rather than a fixed heuristic. BKSEF provides practical heuristics and theoretical support for researchers and developers seeking efficient and application-aware CNN designs. It is suitable for integration into neural architecture search pipelines and real-time systems, offering a new perspective on CNN optimization.
>
---
#### [new 038] Demystifying the Visual Quality Paradox in Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究多模态大语言模型的视觉质量问题，揭示视觉质量与模型性能之间的矛盾，提出VQ-TTT方法优化输入图像以提升模型表现。**

- **链接: [http://arxiv.org/pdf/2506.15645v1](http://arxiv.org/pdf/2506.15645v1)**

> **作者:** Shuo Xing; Lanqing Guo; Hongyuan Hua; Seoyoung Lee; Peiran Li; Yufei Wang; Zhangyang Wang; Zhengzhong Tu
>
> **备注:** 18 pages
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) excel on benchmark vision-language tasks, yet little is known about how input visual quality shapes their responses. Does higher perceptual quality of images already translate to better MLLM understanding? We conduct the first systematic study spanning leading MLLMs and a suite of vision-language benchmarks, applying controlled degradations and stylistic shifts to each image. Surprisingly, we uncover a visual-quality paradox: model, task, and even individual-instance performance can improve when images deviate from human-perceived fidelity. Off-the-shelf restoration pipelines fail to reconcile these idiosyncratic preferences. To close the gap, we introduce Visual-Quality Test-Time Tuning (VQ-TTT)-a lightweight adaptation module that: (1) inserts a learnable, low-rank kernel before the frozen vision encoder to modulate frequency content; and (2) fine-tunes only shallow vision-encoder layers via LoRA. VQ-TTT dynamically adjusts each input image in a single forward pass, aligning it with task-specific model preferences. Across the evaluated MLLMs and all datasets, VQ-TTT lifts significant average accuracy, with no external models, cached features, or extra training data. These findings redefine ``better'' visual inputs for MLLMs and highlight the need for adaptive, rather than universally ``clean'', imagery, in the new era of AI being the main data customer.
>
---
#### [new 039] MSNeRV: Neural Video Representation with Multi-Scale Feature Fusion
- **分类: cs.CV; cs.MM; eess.IV**

- **简介: 该论文属于视频表示与压缩任务，旨在解决INR方法在细节和快速变化视频中的表现不足问题。通过多尺度特征融合框架MSNeRV提升表示能力和压缩效率。**

- **链接: [http://arxiv.org/pdf/2506.15276v1](http://arxiv.org/pdf/2506.15276v1)**

> **作者:** Jun Zhu; Xinfeng Zhang; Lv Tang; JunHao Jiang
>
> **摘要:** Implicit Neural representations (INRs) have emerged as a promising approach for video compression, and have achieved comparable performance to the state-of-the-art codecs such as H.266/VVC. However, existing INR-based methods struggle to effectively represent detail-intensive and fast-changing video content. This limitation mainly stems from the underutilization of internal network features and the absence of video-specific considerations in network design. To address these challenges, we propose a multi-scale feature fusion framework, MSNeRV, for neural video representation. In the encoding stage, we enhance temporal consistency by employing temporal windows, and divide the video into multiple Groups of Pictures (GoPs), where a GoP-level grid is used for background representation. Additionally, we design a multi-scale spatial decoder with a scale-adaptive loss function to integrate multi-resolution and multi-frequency information. To further improve feature extraction, we introduce a multi-scale feature block that fully leverages hidden features. We evaluate MSNeRV on HEVC ClassB and UVG datasets for video representation and compression. Experimental results demonstrate that our model exhibits superior representation capability among INR-based approaches and surpasses VTM-23.7 (Random Access) in dynamic scenarios in terms of compression efficiency.
>
---
#### [new 040] Baltimore Atlas: FreqWeaver Adapter for Semi-supervised Ultra-high Spatial Resolution Land Cover Classification
- **分类: cs.CV**

- **简介: 该论文属于土地覆盖分类任务，旨在解决高分辨率图像下标注成本高、模型适应性差的问题。提出一种轻量级半监督框架，提升细粒度建模效果。**

- **链接: [http://arxiv.org/pdf/2506.15565v1](http://arxiv.org/pdf/2506.15565v1)**

> **作者:** Junhao Wu; Aboagye-Ntow Stephen; Chuyuan Wang; Gang Chen; Xin Huang
>
> **摘要:** Ultra-high Spatial Resolution Land Cover Classification is essential for fine-grained land cover analysis, yet it remains challenging due to the high cost of pixel-level annotations, significant scale variation, and the limited adaptability of large-scale vision models. Existing methods typically focus on 1-meter spatial resolution imagery and rely heavily on annotated data, whereas practical applications often require processing higher-resolution imagery under weak supervision. To address this, we propose a parameter-efficient semi-supervised segmentation framework for 0.3 m spatial resolution imagery, which leverages the knowledge of SAM2 and introduces a remote sensing-specific FreqWeaver Adapter to enhance fine-grained detail modeling while maintaining a lightweight design at only 5.96% of the total model parameters. By effectively leveraging unlabeled data and maintaining minimal parameter overhead, the proposed method delivers robust segmentation results with superior structural consistency, achieving a 1.78% improvement over existing parameter-efficient tuning strategies and a 3.44% gain compared to state-of-the-art high-resolution remote sensing segmentation approaches.
>
---
#### [new 041] Real-Time, Low-Latency Surveillance Using Entropy-Based Adaptive Buffering and MobileNetV2 on Edge Devices
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频监控任务，旨在解决资源受限设备上的低延迟高精度视频分析问题。通过结合熵基自适应缓冲与MobileNetV2，实现高效实时处理。**

- **链接: [http://arxiv.org/pdf/2506.14833v1](http://arxiv.org/pdf/2506.14833v1)**

> **作者:** Poojashree Chandrashekar Pankaj M Sajjanar
>
> **备注:** & pages
>
> **摘要:** This paper describes a high-performance, low-latency video surveillance system designed for resource-constrained environments. We have proposed a formal entropy-based adaptive frame buffering algorithm and integrated that with MobileNetV2 to achieve high throughput with low latency. The system is capable of processing live streams of video with sub-50ms end-to-end inference latency on resource-constrained devices (embedding platforms) such as Raspberry Pi, Amazon, and NVIDIA Jetson Nano. Our method maintains over 92% detection accuracy on standard datasets focused on video surveillance and exhibits robustness to varying lighting, backgrounds, and speeds. A number of comparative and ablation experiments validate the effectiveness of our design. Finally, our architecture is scalable, inexpensive, and compliant with stricter data privacy regulations than common surveillance systems, so that the system could coexist in a smart city or embedded security architecture.
>
---
#### [new 042] Argus Inspection: Do Multimodal Large Language Models Possess the Eye of Panoptes?
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.14805v1](http://arxiv.org/pdf/2506.14805v1)**

> **作者:** Yang Yao; Lingyu Li; Jiaxin Song; Chiyu Chen; Zhenqi He; Yixu Wang; Xin Wang; Tianle Gu; Jie Li; Yan Teng; Yingchun Wang
>
> **摘要:** As Multimodal Large Language Models (MLLMs) continue to evolve, their cognitive and reasoning capabilities have seen remarkable progress. However, challenges in visual fine-grained perception and commonsense causal inference persist. This paper introduces Argus Inspection, a multimodal benchmark with two levels of difficulty, emphasizing detailed visual recognition while incorporating real-world commonsense understanding to evaluate causal reasoning abilities. Expanding on it, we present the Eye of Panoptes framework, which integrates a binary parametric Sigmoid metric with an indicator function, enabling a more holistic evaluation of MLLMs' responses in opinion-based reasoning tasks. Experiments conducted on 26 mainstream MLLMs reveal that the highest performance in visual fine-grained reasoning reaches only 0.46, highlighting considerable potential for enhancement. Our research offers valuable perspectives for the continued refinement of MLLMs.
>
---
#### [new 043] MonoVQD: Monocular 3D Object Detection with Variational Query Denoising and Self-Distillation
- **分类: cs.CV**

- **简介: 该论文属于单目3D目标检测任务，旨在解决DETR架构在该任务中的性能限制。通过引入三种关键技术提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.14835v1](http://arxiv.org/pdf/2506.14835v1)**

> **作者:** Kiet Dang Vu; Trung Thai Tran; Duc Dung Nguyen
>
> **摘要:** Precisely localizing 3D objects from a single image constitutes a central challenge in monocular 3D detection. While DETR-like architectures offer a powerful paradigm, their direct application in this domain encounters inherent limitations, preventing optimal performance. Our work addresses these challenges by introducing MonoVQD, a novel framework designed to fundamentally advance DETR-based monocular 3D detection. We propose three main contributions. First, we propose the Mask Separated Self-Attention mechanism that enables the integration of the denoising process into a DETR architecture. This improves the stability of Hungarian matching to achieve a consistent optimization objective. Second, we present the Variational Query Denoising technique to address the gradient vanishing problem of conventional denoising methods, which severely restricts the efficiency of the denoising process. This explicitly introduces stochastic properties to mitigate this fundamental limitation and unlock substantial performance gains. Finally, we introduce a sophisticated self-distillation strategy, leveraging insights from later decoder layers to synergistically improve query quality in earlier layers, thereby amplifying the iterative refinement process. Rigorous experimentation demonstrates that MonoVQD achieves superior performance on the challenging KITTI monocular benchmark. Highlighting its broad applicability, MonoVQD's core components seamlessly integrate into other architectures, delivering significant performance gains even in multi-view 3D detection scenarios on the nuScenes dataset and underscoring its robust generalization capabilities.
>
---
#### [new 044] OpenPath: Open-Set Active Learning for Pathology Image Classification via Pre-trained Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于病理图像分类任务，解决开放集主动学习中的样本选择问题。提出OpenPath方法，通过预训练视觉语言模型高效筛选信息样本，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.15318v1](http://arxiv.org/pdf/2506.15318v1)**

> **作者:** Lanfeng Zhong; Xin Liao; Shichuan Zhang; Shaoting Zhang; Guotai Wang
>
> **备注:** MICCAI 2025 early accept
>
> **摘要:** Pathology image classification plays a crucial role in accurate medical diagnosis and treatment planning. Training high-performance models for this task typically requires large-scale annotated datasets, which are both expensive and time-consuming to acquire. Active Learning (AL) offers a solution by iteratively selecting the most informative samples for annotation, thereby reducing the labeling effort. However, most AL methods are designed under the assumption of a closed-set scenario, where all the unannotated images belong to target classes. In real-world clinical environments, the unlabeled pool often contains a substantial amount of Out-Of-Distribution (OOD) data, leading to low efficiency of annotation in traditional AL methods. Furthermore, most existing AL methods start with random selection in the first query round, leading to a significant waste of labeling costs in open-set scenarios. To address these challenges, we propose OpenPath, a novel open-set active learning approach for pathological image classification leveraging a pre-trained Vision-Language Model (VLM). In the first query, we propose task-specific prompts that combine target and relevant non-target class prompts to effectively select In-Distribution (ID) and informative samples from the unlabeled pool. In subsequent queries, Diverse Informative ID Sampling (DIS) that includes Prototype-based ID candidate Selection (PIS) and Entropy-Guided Stochastic Sampling (EGSS) is proposed to ensure both purity and informativeness in a query, avoiding the selection of OOD samples. Experiments on two public pathology image datasets show that OpenPath significantly enhances the model's performance due to its high purity of selected samples, and outperforms several state-of-the-art open-set AL methods. The code is available at \href{https://github.com/HiLab-git/OpenPath}{https://github.com/HiLab-git/OpenPath}..
>
---
#### [new 045] Control and Realism: Best of Both Worlds in Layout-to-Image without Training
- **分类: cs.CV**

- **简介: 该论文属于布局到图像生成任务，解决控制精度与真实感不足的问题。提出WinWinLay方法，通过改进注意力机制和更新策略提升效果。**

- **链接: [http://arxiv.org/pdf/2506.15563v1](http://arxiv.org/pdf/2506.15563v1)**

> **作者:** Bonan Li; Yinhan Hu; Songhua Liu; Xinchao Wang
>
> **备注:** Accepted by ICML2025
>
> **摘要:** Layout-to-Image generation aims to create complex scenes with precise control over the placement and arrangement of subjects. Existing works have demonstrated that pre-trained Text-to-Image diffusion models can achieve this goal without training on any specific data; however, they often face challenges with imprecise localization and unrealistic artifacts. Focusing on these drawbacks, we propose a novel training-free method, WinWinLay. At its core, WinWinLay presents two key strategies, Non-local Attention Energy Function and Adaptive Update, that collaboratively enhance control precision and realism. On one hand, we theoretically demonstrate that the commonly used attention energy function introduces inherent spatial distribution biases, hindering objects from being uniformly aligned with layout instructions. To overcome this issue, non-local attention prior is explored to redistribute attention scores, facilitating objects to better conform to the specified spatial conditions. On the other hand, we identify that the vanilla backpropagation update rule can cause deviations from the pre-trained domain, leading to out-of-distribution artifacts. We accordingly introduce a Langevin dynamics-based adaptive update scheme as a remedy that promotes in-domain updating while respecting layout constraints. Extensive experiments demonstrate that WinWinLay excels in controlling element placement and achieving photorealistic visual fidelity, outperforming the current state-of-the-art methods.
>
---
#### [new 046] Privacy-Shielded Image Compression: Defending Against Exploitation from Vision-Language Pretrained Models
- **分类: cs.CV**

- **简介: 该论文属于图像隐私保护任务，旨在防止VLP模型对公开图像的滥用。通过提出PSIC方法，在图像压缩阶段实现隐私防护与质量保持。**

- **链接: [http://arxiv.org/pdf/2506.15201v1](http://arxiv.org/pdf/2506.15201v1)**

> **作者:** Xuelin Shen; Jiayin Xu; Kangsheng Yin; Wenhan Yang
>
> **备注:** 11 pages, 6 figures, publised to ICML 2025
>
> **摘要:** The improved semantic understanding of vision-language pretrained (VLP) models has made it increasingly difficult to protect publicly posted images from being exploited by search engines and other similar tools. In this context, this paper seeks to protect users' privacy by implementing defenses at the image compression stage to prevent exploitation. Specifically, we propose a flexible coding method, termed Privacy-Shielded Image Compression (PSIC), that can produce bitstreams with multiple decoding options. By default, the bitstream is decoded to preserve satisfactory perceptual quality while preventing interpretation by VLP models. Our method also retains the original image compression functionality. With a customizable input condition, the proposed scheme can reconstruct the image that preserves its full semantic information. A Conditional Latent Trigger Generation (CLTG) module is proposed to produce bias information based on customizable conditions to guide the decoding process into different reconstructed versions, and an Uncertainty-Aware Encryption-Oriented (UAEO) optimization function is designed to leverage the soft labels inferred from the target VLP model's uncertainty on the training data. This paper further incorporates an adaptive multi-objective optimization strategy to obtain improved encrypting performance and perceptual quality simultaneously within a unified training process. The proposed scheme is plug-and-play and can be seamlessly integrated into most existing Learned Image Compression (LIC) models. Extensive experiments across multiple downstream tasks have demonstrated the effectiveness of our design.
>
---
#### [new 047] Advances in Compliance Detection: Novel Models Using Vision-Based Tactile Sensors
- **分类: cs.CV; cs.RO; I.2.9**

- **简介: 该论文属于机器人感知任务，旨在解决传统方法在检测物体柔顺性上的不足。通过引入LRCN和Transformer模型，提升基于视觉触觉传感器的柔顺性预测精度。**

- **链接: [http://arxiv.org/pdf/2506.14980v1](http://arxiv.org/pdf/2506.14980v1)**

> **作者:** Ziteng Li; Malte Kuhlmann; Ilana Nisky; Nicolás Navarro-Guerrero
>
> **备注:** Accepted in the IEEE International Conference on Development and Learning (ICDL). The paper contains 8 pages and 7 figures
>
> **摘要:** Compliance is a critical parameter for describing objects in engineering, agriculture, and biomedical applications. Traditional compliance detection methods are limited by their lack of portability and scalability, rely on specialized, often expensive equipment, and are unsuitable for robotic applications. Moreover, existing neural network-based approaches using vision-based tactile sensors still suffer from insufficient prediction accuracy. In this paper, we propose two models based on Long-term Recurrent Convolutional Networks (LRCNs) and Transformer architectures that leverage RGB tactile images and other information captured by the vision-based sensor GelSight to predict compliance metrics accurately. We validate the performance of these models using multiple metrics and demonstrate their effectiveness in accurately estimating compliance. The proposed models exhibit significant performance improvement over the baseline. Additionally, we investigated the correlation between sensor compliance and object compliance estimation, which revealed that objects that are harder than the sensor are more challenging to estimate.
>
---
#### [new 048] ReSeDis: A Dataset for Referring-based Object Search across Large-Scale Image Collections
- **分类: cs.CV**

- **简介: 该论文提出ReSeDis任务，解决跨大规模图像集的基于指代的对象搜索问题，结合图像检索与像素级定位，构建了基准数据集和评估指标。**

- **链接: [http://arxiv.org/pdf/2506.15180v1](http://arxiv.org/pdf/2506.15180v1)**

> **作者:** Ziling Huang; Yidan Zhang; Shin'ichi Satoh
>
> **摘要:** Large-scale visual search engines are expected to solve a dual problem at once: (i) locate every image that truly contains the object described by a sentence and (ii) identify the object's bounding box or exact pixels within each hit. Existing techniques address only one side of this challenge. Visual grounding yields tight boxes and masks but rests on the unrealistic assumption that the object is present in every test image, producing a flood of false alarms when applied to web-scale collections. Text-to-image retrieval excels at sifting through massive databases to rank relevant images, yet it stops at whole-image matches and offers no fine-grained localization. We introduce Referring Search and Discovery (ReSeDis), the first task that unifies corpus-level retrieval with pixel-level grounding. Given a free-form description, a ReSeDis model must decide whether the queried object appears in each image and, if so, where it is, returning bounding boxes or segmentation masks. To enable rigorous study, we curate a benchmark in which every description maps uniquely to object instances scattered across a large, diverse corpus, eliminating unintended matches. We further design a task-specific metric that jointly scores retrieval recall and localization precision. Finally, we provide a straightforward zero-shot baseline using a frozen vision-language model, revealing significant headroom for future study. ReSeDis offers a realistic, end-to-end testbed for building the next generation of robust and scalable multimodal search systems.
>
---
#### [new 049] BCRNet: Enhancing Landmark Detection in Laparoscopic Liver Surgery via Bezier Curve Refinement
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决腹腔镜肝手术中关键解剖结构的精确定位问题。提出BCRNet框架，通过贝塞尔曲线优化提升定位精度。**

- **链接: [http://arxiv.org/pdf/2506.15279v1](http://arxiv.org/pdf/2506.15279v1)**

> **作者:** Qian Li; Feng Liu; Shuojue Yang; Daiyun Shen; Yueming Jin
>
> **备注:** Accepted at MICCAI 2025, 11 pages, 2 figures
>
> **摘要:** Laparoscopic liver surgery, while minimally invasive, poses significant challenges in accurately identifying critical anatomical structures. Augmented reality (AR) systems, integrating MRI/CT with laparoscopic images based on 2D-3D registration, offer a promising solution for enhancing surgical navigation. A vital aspect of the registration progress is the precise detection of curvilinear anatomical landmarks in laparoscopic images. In this paper, we propose BCRNet (Bezier Curve Refinement Net), a novel framework that significantly enhances landmark detection in laparoscopic liver surgery primarily via the Bezier curve refinement strategy. The framework starts with a Multi-modal Feature Extraction (MFE) module designed to robustly capture semantic features. Then we propose Adaptive Curve Proposal Initialization (ACPI) to generate pixel-aligned Bezier curves and confidence scores for reliable initial proposals. Additionally, we design the Hierarchical Curve Refinement (HCR) mechanism to enhance these proposals iteratively through a multi-stage process, capturing fine-grained contextual details from multi-scale pixel-level features for precise Bezier curve adjustment. Extensive evaluations on the L3D and P2ILF datasets demonstrate that BCRNet outperforms state-of-the-art methods, achieving significant performance improvements. Code will be available.
>
---
#### [new 050] Dual-Stage Value-Guided Inference with Margin-Based Reward Adjustment for Fast and Faithful VLM Captioning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉语言模型的图像描述生成任务，旨在解决生成效率低和事实错误问题。提出ViMaR框架，通过两阶段推理提升准确性和速度。**

- **链接: [http://arxiv.org/pdf/2506.15649v1](http://arxiv.org/pdf/2506.15649v1)**

> **作者:** Ankan Deria; Adinath Madhavrao Dukre; Feilong Tang; Sara Atito; Sudipta Roy; Muhammad Awais; Muhammad Haris Khan; Imran Razzak
>
> **摘要:** Despite significant advances in inference-time search for vision-language models (VLMs), existing approaches remain both computationally expensive and prone to unpenalized, low-confidence generations which often lead to persistent hallucinations. We introduce \textbf{Value-guided Inference with Margin-based Reward (ViMaR)}, a two-stage inference framework that improves both efficiency and output fidelity by combining a temporal-difference value model with a margin-aware reward adjustment. In the first stage, we perform a single pass to identify the highest-value caption among diverse candidates. In the second stage, we selectively refine only those segments that were overlooked or exhibit weak visual grounding, thereby eliminating frequently rewarded evaluations. A calibrated margin-based penalty discourages low-confidence continuations while preserving descriptive richness. Extensive experiments across multiple VLM architectures demonstrate that ViMaR generates captions that are significantly more reliable, factually accurate, detailed, and explanatory, while achieving over 4$\times$ speedup compared to existing value-guided methods. Specifically, we show that ViMaR trained solely on LLaVA Mistral-7B, \textit{generalizes effectively to guide decoding in a stronger unseen model}. To further validate this, we adapt the ViMaR to steer generation in LLaVA-OneVision-Qwen2-7B, leading to consistent improvements in caption quality and demonstrating robust cross-model guidance. This cross-model generalization highlights ViMaR's flexibility and modularity, positioning it as a scalable and transferable inference-time decoding strategy. Furthermore, when ViMaR-generated captions are used for self-training, the underlying models achieve substantial gains across a broad suite of visual comprehension benchmarks, underscoring the potential of fast, accurate, and self-improving VLM pipelines.
>
---
#### [new 051] Echo-DND: A dual noise diffusion model for robust and precise left ventricle segmentation in echocardiography
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决超声心动图中左心室边界模糊、噪声大导致的分割难题。提出Echo-DND模型，结合双噪声和多尺度融合，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.15166v1](http://arxiv.org/pdf/2506.15166v1)**

> **作者:** Abdur Rahman; Keerthiveena Balraj; Manojkumar Ramteke; Anurag Singh Rathore
>
> **备注:** Version of record published in Discover Applied Sciences (Springer Nature). The definitive article is available at https://doi.org/10.1007/s42452-025-07055-5
>
> **摘要:** Recent advancements in diffusion probabilistic models (DPMs) have revolutionized image processing, demonstrating significant potential in medical applications. Accurate segmentation of the left ventricle (LV) in echocardiograms is crucial for diagnostic procedures and necessary treatments. However, ultrasound images are notoriously noisy with low contrast and ambiguous LV boundaries, thereby complicating the segmentation process. To address these challenges, this paper introduces Echo-DND, a novel dual-noise diffusion model specifically designed for this task. Echo-DND leverages a unique combination of Gaussian and Bernoulli noises. It also incorporates a multi-scale fusion conditioning module to improve segmentation precision. Furthermore, it utilizes spatial coherence calibration to maintain spatial integrity in segmentation masks. The model's performance was rigorously validated on the CAMUS and EchoNet-Dynamic datasets. Extensive evaluations demonstrate that the proposed framework outperforms existing SOTA models. It achieves high Dice scores of 0.962 and 0.939 on these datasets, respectively. The proposed Echo-DND model establishes a new standard in echocardiogram segmentation, and its architecture holds promise for broader applicability in other medical imaging tasks, potentially improving diagnostic accuracy across various medical domains. Project page: https://abdur75648.github.io/Echo-DND
>
---
#### [new 052] Show-o2: Improved Native Unified Multimodal Models
- **分类: cs.CV**

- **简介: 该论文提出Show-o2模型，解决多模态理解和生成任务，通过融合视觉与语言信息，提升跨模态处理能力。**

- **链接: [http://arxiv.org/pdf/2506.15564v1](http://arxiv.org/pdf/2506.15564v1)**

> **作者:** Jinheng Xie; Zhenheng Yang; Mike Zheng Shou
>
> **备注:** Technical report
>
> **摘要:** This paper presents improved native unified multimodal models, \emph{i.e.,} Show-o2, that leverage autoregressive modeling and flow matching. Built upon a 3D causal variational autoencoder space, unified visual representations are constructed through a dual-path of spatial (-temporal) fusion, enabling scalability across image and video modalities while ensuring effective multimodal understanding and generation. Based on a language model, autoregressive modeling and flow matching are natively applied to the language head and flow head, respectively, to facilitate text token prediction and image/video generation. A two-stage training recipe is designed to effectively learn and scale to larger models. The resulting Show-o2 models demonstrate versatility in handling a wide range of multimodal understanding and generation tasks across diverse modalities, including text, images, and videos. Code and models are released at https://github.com/showlab/Show-o.
>
---
#### [new 053] Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 本文属于多智能体人类轨迹预测任务，旨在提升对多主体交互的理解。论文综述了2020至2024年的深度学习方法，分析其架构、输入与策略，并探讨未来方向。**

- **链接: [http://arxiv.org/pdf/2506.14831v1](http://arxiv.org/pdf/2506.14831v1)**

> **作者:** Céline Finet; Stephane Da Silva Martins; Jean-Bernard Hayet; Ioannis Karamouzas; Javad Amirian; Sylvie Le Hégarat-Mascle; Julien Pettré; Emanuel Aldea
>
> **备注:** 30 pages
>
> **摘要:** With the emergence of powerful data-driven methods in human trajectory prediction (HTP), gaining a finer understanding of multi-agent interactions lies within hand's reach, with important implications in areas such as autonomous navigation and crowd modeling. This survey reviews some of the most recent advancements in deep learning-based multi-agent trajectory prediction, focusing on studies published between 2020 and 2024. We categorize the existing methods based on their architectural design, their input representations, and their overall prediction strategies, placing a particular emphasis on models evaluated using the ETH/UCY benchmark. Furthermore, we highlight key challenges and future research directions in the field of multi-agent HTP.
>
---
#### [new 054] Break Stylistic Sophon: Are We Really Meant to Confine the Imagination in Style Transfer?
- **分类: cs.CV**

- **简介: 该论文属于风格迁移任务，旨在解决传统方法中的风格注入不准确和内容丢失问题。提出语义风格注入、数据增强和训练-free 三重扩散过程，实现高质量风格迁移与文本控制。**

- **链接: [http://arxiv.org/pdf/2506.15033v1](http://arxiv.org/pdf/2506.15033v1)**

> **作者:** Gary Song Yan; Yusen Zhang; Jinyu Zhao; Hao Zhang; Zhangping Yang; Guanye Xiong; Yanfei Liu; Tao Zhang; Yujie He; Siyuan Tian; Yao Gou; Min Li
>
> **摘要:** In this pioneering study, we introduce StyleWallfacer, a groundbreaking unified training and inference framework, which not only addresses various issues encountered in the style transfer process of traditional methods but also unifies the framework for different tasks. This framework is designed to revolutionize the field by enabling artist level style transfer and text driven stylization. First, we propose a semantic-based style injection method that uses BLIP to generate text descriptions strictly aligned with the semantics of the style image in CLIP space. By leveraging a large language model to remove style-related descriptions from these descriptions, we create a semantic gap. This gap is then used to fine-tune the model, enabling efficient and drift-free injection of style knowledge. Second, we propose a data augmentation strategy based on human feedback, incorporating high-quality samples generated early in the fine-tuning process into the training set to facilitate progressive learning and significantly reduce its overfitting. Finally, we design a training-free triple diffusion process using the fine-tuned model, which manipulates the features of self-attention layers in a manner similar to the cross-attention mechanism. Specifically, in the generation process, the key and value of the content-related process are replaced with those of the style-related process to inject style while maintaining text control over the model. We also introduce query preservation to mitigate disruptions to the original content. Under such a design, we have achieved high-quality image-driven style transfer and text-driven stylization, delivering artist-level style transfer results while preserving the original image content. Moreover, we achieve image color editing during the style transfer process for the first time.
>
---
#### [new 055] Sekai: A Video Dataset towards World Exploration
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Sekai数据集，用于解决视频生成中世界探索训练的不足。通过收集全球多场景视频并进行详细标注，支持交互式世界探索模型训练。**

- **链接: [http://arxiv.org/pdf/2506.15675v1](http://arxiv.org/pdf/2506.15675v1)**

> **作者:** Zhen Li; Chuanhao Li; Xiaofeng Mao; Shaoheng Lin; Ming Li; Shitian Zhao; Zhaopan Xu; Xinyue Li; Yukang Feng; Jianwen Sun; Zizhen Li; Fanrui Zhang; Jiaxin Ai; Zhixiang Wang; Yuwei Wu; Tong He; Jiangmiao Pang; Yu Qiao; Yunde Jia; Kaipeng Zhang
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Video generation techniques have made remarkable progress, promising to be the foundation of interactive world exploration. However, existing video generation datasets are not well-suited for world exploration training as they suffer from some limitations: limited locations, short duration, static scenes, and a lack of annotations about exploration and the world. In this paper, we introduce Sekai (meaning ``world'' in Japanese), a high-quality first-person view worldwide video dataset with rich annotations for world exploration. It consists of over 5,000 hours of walking or drone view (FPV and UVA) videos from over 100 countries and regions across 750 cities. We develop an efficient and effective toolbox to collect, pre-process and annotate videos with location, scene, weather, crowd density, captions, and camera trajectories. Experiments demonstrate the quality of the dataset. And, we use a subset to train an interactive video world exploration model, named YUME (meaning ``dream'' in Japanese). We believe Sekai will benefit the area of video generation and world exploration, and motivate valuable applications.
>
---
#### [new 056] SynPo: Boosting Training-Free Few-Shot Medical Segmentation via High-Quality Negative Prompts
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决低对比度图像中负提示质量差的问题。通过设计新模块提升负提示质量，实现高效分割。**

- **链接: [http://arxiv.org/pdf/2506.15153v1](http://arxiv.org/pdf/2506.15153v1)**

> **作者:** Yufei Liu; Haoke Xiao; Jiaxing Chai; Yongcun Zhang; Rong Wang; Zijie Meng; Zhiming Luo
>
> **摘要:** The advent of Large Vision Models (LVMs) offers new opportunities for few-shot medical image segmentation. However, existing training-free methods based on LVMs fail to effectively utilize negative prompts, leading to poor performance on low-contrast medical images. To address this issue, we propose SynPo, a training-free few-shot method based on LVMs (e.g., SAM), with the core insight: improving the quality of negative prompts. To select point prompts in a more reliable confidence map, we design a novel Confidence Map Synergy Module by combining the strengths of DINOv2 and SAM. Based on the confidence map, we select the top-k pixels as the positive points set and choose the negative points set using a Gaussian distribution, followed by independent K-means clustering for both sets. Then, these selected points are leveraged as high-quality prompts for SAM to get the segmentation results. Extensive experiments demonstrate that SynPo achieves performance comparable to state-of-the-art training-based few-shot methods.
>
---
#### [new 057] MEGC2025: Micro-Expression Grand Challenge on Spot Then Recognize and Visual Question Answering
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于微表情分析任务，旨在解决 spotting 和 recognition 分离导致的效率问题，并通过 ME-STR 和 ME-VQA 两个任务提升微表情理解与识别能力。**

- **链接: [http://arxiv.org/pdf/2506.15298v1](http://arxiv.org/pdf/2506.15298v1)**

> **作者:** Xinqi Fan; Jingting Li; John See; Moi Hoon Yap; Wen-Huang Cheng; Xiaobai Li; Xiaopeng Hong; Su-Jing Wang; Adrian K. Davision
>
> **备注:** Micro-Expression Grand Challenge (MEGC) at ACM MM 2025
>
> **摘要:** Facial micro-expressions (MEs) are involuntary movements of the face that occur spontaneously when a person experiences an emotion but attempts to suppress or repress the facial expression, typically found in a high-stakes environment. In recent years, substantial advancements have been made in the areas of ME recognition, spotting, and generation. However, conventional approaches that treat spotting and recognition as separate tasks are suboptimal, particularly for analyzing long-duration videos in realistic settings. Concurrently, the emergence of multimodal large language models (MLLMs) and large vision-language models (LVLMs) offers promising new avenues for enhancing ME analysis through their powerful multimodal reasoning capabilities. The ME grand challenge (MEGC) 2025 introduces two tasks that reflect these evolving research directions: (1) ME spot-then-recognize (ME-STR), which integrates ME spotting and subsequent recognition in a unified sequential pipeline; and (2) ME visual question answering (ME-VQA), which explores ME understanding through visual question answering, leveraging MLLMs or LVLMs to address diverse question types related to MEs. All participating algorithms are required to run on this test set and submit their results on a leaderboard. More details are available at https://megc2025.github.io.
>
---
#### [new 058] DAVID-XR1: Detecting AI-Generated Videos with Explainable Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于AI生成视频检测任务，旨在解决如何准确识别并解释AI生成视频的问题。工作包括构建DAVID-X数据集和提出可解释的检测模型DAVID-XR1。**

- **链接: [http://arxiv.org/pdf/2506.14827v1](http://arxiv.org/pdf/2506.14827v1)**

> **作者:** Yifeng Gao; Yifan Ding; Hongyu Su; Juncheng Li; Yunhan Zhao; Lin Luo; Zixing Chen; Li Wang; Xin Wang; Yixu Wang; Xingjun Ma; Yu-Gang Jiang
>
> **摘要:** As AI-generated video becomes increasingly pervasive across media platforms, the ability to reliably distinguish synthetic content from authentic footage has become both urgent and essential. Existing approaches have primarily treated this challenge as a binary classification task, offering limited insight into where or why a model identifies a video as AI-generated. However, the core challenge extends beyond simply detecting subtle artifacts; it requires providing fine-grained, persuasive evidence that can convince auditors and end-users alike. To address this critical gap, we introduce DAVID-X, the first dataset to pair AI-generated videos with detailed defect-level, temporal-spatial annotations and written rationales. Leveraging these rich annotations, we present DAVID-XR1, a video-language model designed to deliver an interpretable chain of visual reasoning-including defect categorization, temporal-spatial localization, and natural language explanations. This approach fundamentally transforms AI-generated video detection from an opaque black-box decision into a transparent and verifiable diagnostic process. We demonstrate that a general-purpose backbone, fine-tuned on our compact dataset and enhanced with chain-of-thought distillation, achieves strong generalization across a variety of generators and generation modes. Our results highlight the promise of explainable detection methods for trustworthy identification of AI-generated video content.
>
---
#### [new 059] Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D生成任务，旨在解决3D内容创作复杂的问题。提出Hunyuan3D 2.1系统，包含形状生成与纹理合成模块，提供完整生成流程指导。**

- **链接: [http://arxiv.org/pdf/2506.15442v1](http://arxiv.org/pdf/2506.15442v1)**

> **作者:** Team Hunyuan3D; Shuhui Yang; Mingxin Yang; Yifei Feng; Xin Huang; Sheng Zhang; Zebin He; Di Luo; Haolin Liu; Yunfei Zhao; Qingxiang Lin; Zeqiang Lai; Xianghui Yang; Huiwen Shi; Zibo Zhao; Bowen Zhang; Hongyu Yan; Lifu Wang; Sicong Liu; Jihong Zhang; Meng Chen; Liang Dong; Yiwen Jia; Yulin Cai; Jiaao Yu; Yixuan Tang; Dongyuan Guo; Junlin Yu; Hao Zhang; Zheng Ye; Peng He; Runzhou Wu; Shida Wei; Chao Zhang; Yonghao Tan; Yifu Sun; Lin Niu; Shirui Huang; Bojian Zheng; Shu Liu; Shilin Chen; Xiang Yuan; Xiaofeng Yang; Kai Liu; Jianchen Zhu; Peng Chen; Tian Liu; Di Wang; Yuhong Liu; Linus; Jie Jiang; Jingwei Huang; Chunchao Guo
>
> **备注:** Github link: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
>
> **摘要:** 3D AI-generated content (AIGC) is a passionate field that has significantly accelerated the creation of 3D models in gaming, film, and design. Despite the development of several groundbreaking models that have revolutionized 3D generation, the field remains largely accessible only to researchers, developers, and designers due to the complexities involved in collecting, processing, and training 3D models. To address these challenges, we introduce Hunyuan3D 2.1 as a case study in this tutorial. This tutorial offers a comprehensive, step-by-step guide on processing 3D data, training a 3D generative model, and evaluating its performance using Hunyuan3D 2.1, an advanced system for producing high-resolution, textured 3D assets. The system comprises two core components: the Hunyuan3D-DiT for shape generation and the Hunyuan3D-Paint for texture synthesis. We will explore the entire workflow, including data preparation, model architecture, training strategies, evaluation metrics, and deployment. By the conclusion of this tutorial, you will have the knowledge to finetune or develop a robust 3D generative model suitable for applications in gaming, virtual reality, and industrial design.
>
---
#### [new 060] RaCalNet: Radar Calibration Network for Sparse-Supervised Metric Depth Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于深度估计任务，解决稀疏监督下高精度深度图生成问题。提出RaCalNet框架，通过稀疏LiDAR监督提升雷达点精度，实现无密集标注的高质量深度预测。**

- **链接: [http://arxiv.org/pdf/2506.15560v1](http://arxiv.org/pdf/2506.15560v1)**

> **作者:** Xingrui Qin; Wentao Zhao; Chuan Cao; Yihe Niu; Houcheng Jiang; Jingchuan Wang
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Dense metric depth estimation using millimeter-wave radar typically requires dense LiDAR supervision, generated via multi-frame projection and interpolation, to guide the learning of accurate depth from sparse radar measurements and RGB images. However, this paradigm is both costly and data-intensive. To address this, we propose RaCalNet, a novel framework that eliminates the need for dense supervision by using sparse LiDAR to supervise the learning of refined radar measurements, resulting in a supervision density of merely around 1% compared to dense-supervised methods. Unlike previous approaches that associate radar points with broad image regions and rely heavily on dense labels, RaCalNet first recalibrates and refines sparse radar points to construct accurate depth priors. These priors then serve as reliable anchors to guide monocular depth prediction, enabling metric-scale estimation without resorting to dense supervision. This design improves structural consistency and preserves fine details. Despite relying solely on sparse supervision, RaCalNet surpasses state-of-the-art dense-supervised methods, producing depth maps with clear object contours and fine-grained textures. Extensive experiments on the ZJU-4DRadarCam dataset and real-world deployment scenarios demonstrate its effectiveness, reducing RMSE by 35.30% and 34.89%, respectively.
>
---
#### [new 061] MapFM: Foundation Model-Driven HD Mapping with Multi-Task Contextual Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶中的高精地图生成任务，旨在提升矢量地图的准确性。通过融合基础模型和多任务学习，增强环境理解与预测质量。**

- **链接: [http://arxiv.org/pdf/2506.15313v1](http://arxiv.org/pdf/2506.15313v1)**

> **作者:** Leonid Ivanov; Vasily Yuryev; Dmitry Yudin
>
> **备注:** Preprint. Submitted. 12 pages, 4 figures
>
> **摘要:** In autonomous driving, high-definition (HD) maps and semantic maps in bird's-eye view (BEV) are essential for accurate localization, planning, and decision-making. This paper introduces an enhanced End-to-End model named MapFM for online vectorized HD map generation. We show significantly boost feature representation quality by incorporating powerful foundation model for encoding camera images. To further enrich the model's understanding of the environment and improve prediction quality, we integrate auxiliary prediction heads for semantic segmentation in the BEV representation. This multi-task learning approach provides richer contextual supervision, leading to a more comprehensive scene representation and ultimately resulting in higher accuracy and improved quality of the predicted vectorized HD maps. The source code is available at https://github.com/LIvanoff/MapFM.
>
---
#### [new 062] DM-FNet: Unified multimodal medical image fusion via diffusion process-trained encoder-decoder
- **分类: cs.CV**

- **简介: 该论文属于医学图像融合任务，旨在解决传统方法在细节捕捉和跨模态交互上的不足。提出DM-FNet网络，通过两阶段扩散模型提升融合质量。**

- **链接: [http://arxiv.org/pdf/2506.15218v1](http://arxiv.org/pdf/2506.15218v1)**

> **作者:** Dan He; Weisheng Li; Guofen Wang; Yuping Huang; Shiqiang Liu
>
> **备注:** This paper has been accepted by IEEE Transactions on Multimedia (TMM) in March 2025
>
> **摘要:** Multimodal medical image fusion (MMIF) extracts the most meaningful information from multiple source images, enabling a more comprehensive and accurate diagnosis. Achieving high-quality fusion results requires a careful balance of brightness, color, contrast, and detail; this ensures that the fused images effectively display relevant anatomical structures and reflect the functional status of the tissues. However, existing MMIF methods have limited capacity to capture detailed features during conventional training and suffer from insufficient cross-modal feature interaction, leading to suboptimal fused image quality. To address these issues, this study proposes a two-stage diffusion model-based fusion network (DM-FNet) to achieve unified MMIF. In Stage I, a diffusion process trains UNet for image reconstruction. UNet captures detailed information through progressive denoising and represents multilevel data, providing a rich set of feature representations for the subsequent fusion network. In Stage II, noisy images at various steps are input into the fusion network to enhance the model's feature recognition capability. Three key fusion modules are also integrated to process medical images from different modalities adaptively. Ultimately, the robust network structure and a hybrid loss function are integrated to harmonize the fused image's brightness, color, contrast, and detail, enhancing its quality and information density. The experimental results across various medical image types demonstrate that the proposed method performs exceptionally well regarding objective evaluation metrics. The fused image preserves appropriate brightness, a comprehensive distribution of radioactive tracers, rich textures, and clear edges. The code is available at https://github.com/HeDan-11/DM-FNet.
>
---
#### [new 063] NERO: Explainable Out-of-Distribution Detection with Neuron-level Relevance
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于OOD检测任务，旨在提升模型可靠性。通过神经元级相关性分析，提出NERO方法增强OOD样本区分能力，并实现可解释性。**

- **链接: [http://arxiv.org/pdf/2506.15404v1](http://arxiv.org/pdf/2506.15404v1)**

> **作者:** Anju Chhetri; Jari Korhonen; Prashnna Gyawali; Binod Bhattarai
>
> **摘要:** Ensuring reliability is paramount in deep learning, particularly within the domain of medical imaging, where diagnostic decisions often hinge on model outputs. The capacity to separate out-of-distribution (OOD) samples has proven to be a valuable indicator of a model's reliability in research. In medical imaging, this is especially critical, as identifying OOD inputs can help flag potential anomalies that might otherwise go undetected. While many OOD detection methods rely on feature or logit space representations, recent works suggest these approaches may not fully capture OOD diversity. To address this, we propose a novel OOD scoring mechanism, called NERO, that leverages neuron-level relevance at the feature layer. Specifically, we cluster neuron-level relevance for each in-distribution (ID) class to form representative centroids and introduce a relevance distance metric to quantify a new sample's deviation from these centroids, enhancing OOD separability. Additionally, we refine performance by incorporating scaled relevance in the bias term and combining feature norms. Our framework also enables explainable OOD detection. We validate its effectiveness across multiple deep learning architectures on the gastrointestinal imaging benchmarks Kvasir and GastroVision, achieving improvements over state-of-the-art OOD detection methods.
>
---
#### [new 064] A Unified Graph-based Framework for Scalable 3D Tree Reconstruction and Non-Destructive Biomass Estimation from Point Clouds
- **分类: cs.CV**

- **简介: 该论文属于3D树重建与生物量估算任务，解决传统QSM方法依赖高质量数据和复杂预处理的问题，提出一种统一的图基框架，实现大规模点云的端到端处理。**

- **链接: [http://arxiv.org/pdf/2506.15577v1](http://arxiv.org/pdf/2506.15577v1)**

> **作者:** Di Wang; Shi Li
>
> **备注:** 17 pages,19 figures
>
> **摘要:** Estimating forest above-ground biomass (AGB) is crucial for assessing carbon storage and supporting sustainable forest management. Quantitative Structural Model (QSM) offers a non-destructive approach to AGB estimation through 3D tree structural reconstruction. However, current QSM methods face significant limitations, as they are primarily designed for individual trees,depend on high-quality point cloud data from terrestrial laser scanning (TLS), and also require multiple pre-processing steps that hinder scalability and practical deployment. This study presents a novel unified framework that enables end-to-end processing of large-scale point clouds using an innovative graph-based pipeline. The proposed approach seamlessly integrates tree segmentation,leaf-wood separation and 3D skeletal reconstruction through dedicated graph operations including pathing and abstracting for tree topology reasoning. Comprehensive validation was conducted on datasets with varying leaf conditions (leaf-on and leaf-off), spatial scales (tree- and plot-level), and data sources (TLS and UAV-based laser scanning, ULS). Experimental results demonstrate strong performance under challenging conditions, particularly in leaf-on scenarios (~20% relative error) and low-density ULS datasets with partial coverage (~30% relative error). These findings indicate that the proposed framework provides a robust and scalable solution for large-scale, non-destructive AGB estimation. It significantly reduces dependency on specialized pre-processing tools and establishes ULS as a viable alternative to TLS. To our knowledge, this is the first method capable of enabling seamless, end-to-end 3D tree reconstruction at operational scales. This advancement substantially improves the feasibility of QSM-based AGB estimation, paving the way for broader applications in forest inventory and climate change research.
>
---
#### [new 065] Enhancing Vector Quantization with Distributional Matching: A Theoretical and Empirical Study
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于深度学习中的向量量化任务，旨在解决训练不稳定和代码本崩溃问题。通过分布对齐提升量化效果。**

- **链接: [http://arxiv.org/pdf/2506.15078v1](http://arxiv.org/pdf/2506.15078v1)**

> **作者:** Xianghong Fang; Litao Guo; Hengchao Chen; Yuxuan Zhang; XiaofanXia; Dingjie Song; Yexin Liu; Hao Wang; Harry Yang; Yuan Yuan; Qiang Sun
>
> **摘要:** The success of autoregressive models largely depends on the effectiveness of vector quantization, a technique that discretizes continuous features by mapping them to the nearest code vectors within a learnable codebook. Two critical issues in existing vector quantization methods are training instability and codebook collapse. Training instability arises from the gradient discrepancy introduced by the straight-through estimator, especially in the presence of significant quantization errors, while codebook collapse occurs when only a small subset of code vectors are utilized during training. A closer examination of these issues reveals that they are primarily driven by a mismatch between the distributions of the features and code vectors, leading to unrepresentative code vectors and significant data information loss during compression. To address this, we employ the Wasserstein distance to align these two distributions, achieving near 100\% codebook utilization and significantly reducing the quantization error. Both empirical and theoretical analyses validate the effectiveness of the proposed approach.
>
---
#### [new 066] FedWSIDD: Federated Whole Slide Image Classification via Dataset Distillation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分类任务，解决联邦学习中资源异构和隐私保护问题。提出FedWSIDD，通过数据蒸馏生成合成切片进行模型训练，提升分类性能并保护患者隐私。**

- **链接: [http://arxiv.org/pdf/2506.15365v1](http://arxiv.org/pdf/2506.15365v1)**

> **作者:** Haolong Jin; Shenglin Liu; Cong Cong; Qingmin Feng; Yongzhi Liu; Lina Huang; Yingzi Hu
>
> **备注:** MICCAI 2025
>
> **摘要:** Federated learning (FL) has emerged as a promising approach for collaborative medical image analysis, enabling multiple institutions to build robust predictive models while preserving sensitive patient data. In the context of Whole Slide Image (WSI) classification, FL faces significant challenges, including heterogeneous computational resources across participating medical institutes and privacy concerns. To address these challenges, we propose FedWSIDD, a novel FL paradigm that leverages dataset distillation (DD) to learn and transmit synthetic slides. On the server side, FedWSIDD aggregates synthetic slides from participating centres and distributes them across all centres. On the client side, we introduce a novel DD algorithm tailored to histopathology datasets which incorporates stain normalisation into the distillation process to generate a compact set of highly informative synthetic slides. These synthetic slides, rather than model parameters, are transmitted to the server. After communication, the received synthetic slides are combined with original slides for local tasks. Extensive experiments on multiple WSI classification tasks, including CAMELYON16 and CAMELYON17, demonstrate that FedWSIDD offers flexibility for heterogeneous local models, enhances local WSI classification performance, and preserves patient privacy. This makes it a highly effective solution for complex WSI classification tasks. The code is available at FedWSIDD.
>
---
#### [new 067] Human Motion Capture from Loose and Sparse Inertial Sensors with Garment-aware Diffusion Models
- **分类: cs.GR; cs.AI; cs.CV; cs.HC; 68T07, 68T45, 68U01; I.2; I.3; I.4; I.5**

- **简介: 该论文属于人体姿态估计任务，解决松散佩戴IMU传感器的全身运动捕捉问题。通过模拟数据和扩散模型提升估计精度。**

- **链接: [http://arxiv.org/pdf/2506.15290v1](http://arxiv.org/pdf/2506.15290v1)**

> **作者:** Andela Ilic; Jiaxi Jiang; Paul Streli; Xintong Liu; Christian Holz
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** Motion capture using sparse inertial sensors has shown great promise due to its portability and lack of occlusion issues compared to camera-based tracking. Existing approaches typically assume that IMU sensors are tightly attached to the human body. However, this assumption often does not hold in real-world scenarios. In this paper, we present a new task of full-body human pose estimation using sparse, loosely attached IMU sensors. To solve this task, we simulate IMU recordings from an existing garment-aware human motion dataset. We developed transformer-based diffusion models to synthesize loose IMU data and estimate human poses based on this challenging loose IMU data. In addition, we show that incorporating garment-related parameters while training the model on simulated loose data effectively maintains expressiveness and enhances the ability to capture variations introduced by looser or tighter garments. Experiments show that our proposed diffusion methods trained on simulated and synthetic data outperformed the state-of-the-art methods quantitatively and qualitatively, opening up a promising direction for future research.
>
---
#### [new 068] Automated MRI Tumor Segmentation using hybrid U-Net with Transformer and Efficient Attention
- **分类: eess.IV; cs.CV; I.4.6; I.2.6; I.4.9**

- **简介: 该论文属于医学图像分割任务，旨在解决肿瘤自动分割问题。针对本地数据集，提出混合U-Net与Transformer的模型，提升分割精度与效率。**

- **链接: [http://arxiv.org/pdf/2506.15562v1](http://arxiv.org/pdf/2506.15562v1)**

> **作者:** Syed Haider Ali; Asrar Ahmad; Muhammad Ali; Asifullah Khan; Muhammad Shahban; Nadeem Shaukat
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Cancer is an abnormal growth with potential to invade locally and metastasize to distant organs. Accurate auto-segmentation of the tumor and surrounding normal tissues is required for radiotherapy treatment plan optimization. Recent AI-based segmentation models are generally trained on large public datasets, which lack the heterogeneity of local patient populations. While these studies advance AI-based medical image segmentation, research on local datasets is necessary to develop and integrate AI tumor segmentation models directly into hospital software for efficient and accurate oncology treatment planning and execution. This study enhances tumor segmentation using computationally efficient hybrid UNet-Transformer models on magnetic resonance imaging (MRI) datasets acquired from a local hospital under strict privacy protection. We developed a robust data pipeline for seamless DICOM extraction and preprocessing, followed by extensive image augmentation to ensure model generalization across diverse clinical settings, resulting in a total dataset of 6080 images for training. Our novel architecture integrates UNet-based convolutional neural networks with a transformer bottleneck and complementary attention modules, including efficient attention, Squeeze-and-Excitation (SE) blocks, Convolutional Block Attention Module (CBAM), and ResNeXt blocks. To accelerate convergence and reduce computational demands, we used a maximum batch size of 8 and initialized the encoder with pretrained ImageNet weights, training the model on dual NVIDIA T4 GPUs via checkpointing to overcome Kaggle's runtime limits. Quantitative evaluation on the local MRI dataset yielded a Dice similarity coefficient of 0.764 and an Intersection over Union (IoU) of 0.736, demonstrating competitive performance despite limited data and underscoring the importance of site-specific model development for clinical deployment.
>
---
#### [new 069] Privacy-Preserving Chest X-ray Classification in Latent Space with Homomorphically Encrypted Neural Inference
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医疗图像分类任务，旨在解决隐私保护问题。通过VQGAN压缩图像并使用同态加密进行推理，降低计算成本，提升安全性。**

- **链接: [http://arxiv.org/pdf/2506.15258v1](http://arxiv.org/pdf/2506.15258v1)**

> **作者:** Jonghun Kim; Gyeongdeok Jo; Shinyoung Ra; Hyunjin Park
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Medical imaging data contain sensitive patient information requiring strong privacy protection. Many analytical setups require data to be sent to a server for inference purposes. Homomorphic encryption (HE) provides a solution by allowing computations to be performed on encrypted data without revealing the original information. However, HE inference is computationally expensive, particularly for large images (e.g., chest X-rays). In this study, we propose an HE inference framework for medical images that uses VQGAN to compress images into latent representations, thereby significantly reducing the computational burden while preserving image quality. We approximate the activation functions with lower-degree polynomials to balance the accuracy and efficiency in compliance with HE requirements. We observed that a downsampling factor of eight for compression achieved an optimal balance between performance and computational cost. We further adapted the squeeze and excitation module, which is known to improve traditional CNNs, to enhance the HE framework. Our method was tested on two chest X-ray datasets for multi-label classification tasks using vanilla CNN backbones. Although HE inference remains relatively slow and introduces minor performance differences compared with unencrypted inference, our approach shows strong potential for practical use in medical images
>
---
#### [new 070] Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于变形物体建模任务，旨在从RGB-D视频中学习动态模型。通过结合粒子与网格的混合表示，解决物体状态估计和动态建模难题。**

- **链接: [http://arxiv.org/pdf/2506.15680v1](http://arxiv.org/pdf/2506.15680v1)**

> **作者:** Kaifeng Zhang; Baoyu Li; Kris Hauser; Yunzhu Li
>
> **备注:** Project page: https://kywind.github.io/pgnd
>
> **摘要:** Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information. We address these challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation. Our particle-grid model captures global shape and motion information while predicting dense particle movements, enabling the modeling of objects with varied shapes and materials. Particles represent object shapes, while the spatial grid discretizes the 3D space to ensure spatial continuity and enhance learning efficiency. Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos. Through experiments, we demonstrate that our model learns the dynamics of diverse objects -- such as ropes, cloths, stuffed animals, and paper bags -- from sparse-view RGB-D recordings of robot-object interactions, while also generalizing at the category level to unseen instances. Our approach outperforms state-of-the-art learning-based and physics-based simulators, particularly in scenarios with limited camera views. Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks. The project page is available at https://kywind.github.io/pgnd .
>
---
#### [new 071] NeuroMoE: A Transformer-Based Mixture-of-Experts Framework for Multi-Modal Neurological Disorder Classification
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.14970v1](http://arxiv.org/pdf/2506.14970v1)**

> **作者:** Wajih Hassan Raza; Aamir Bader Shah; Yu Wen; Yidan Shen; Juan Diego Martinez Lemus; Mya Caryn Schiess; Timothy Michael Ellmore; Renjie Hu; Xin Fu
>
> **备注:** Accepted at the 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society
>
> **摘要:** The integration of multi-modal Magnetic Resonance Imaging (MRI) and clinical data holds great promise for enhancing the diagnosis of neurological disorders (NDs) in real-world clinical settings. Deep Learning (DL) has recently emerged as a powerful tool for extracting meaningful patterns from medical data to aid in diagnosis. However, existing DL approaches struggle to effectively leverage multi-modal MRI and clinical data, leading to suboptimal performance. To address this challenge, we utilize a unique, proprietary multi-modal clinical dataset curated for ND research. Based on this dataset, we propose a novel transformer-based Mixture-of-Experts (MoE) framework for ND classification, leveraging multiple MRI modalities-anatomical (aMRI), Diffusion Tensor Imaging (DTI), and functional (fMRI)-alongside clinical assessments. Our framework employs transformer encoders to capture spatial relationships within volumetric MRI data while utilizing modality-specific experts for targeted feature extraction. A gating mechanism with adaptive fusion dynamically integrates expert outputs, ensuring optimal predictive performance. Comprehensive experiments and comparisons with multiple baselines demonstrate that our multi-modal approach significantly enhances diagnostic accuracy, particularly in distinguishing overlapping disease states. Our framework achieves a validation accuracy of 82.47\%, outperforming baseline methods by over 10\%, highlighting its potential to improve ND diagnosis by applying multi-modal learning to real-world clinical data.
>
---
#### [new 072] Robust Instant Policy: Leveraging Student's t-Regression Model for Robust In-context Imitation Learning of Robot Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人模仿学习任务，解决LLM生成轨迹中的幻觉问题。提出RIP算法，利用Student's t回归模型提升轨迹可靠性。**

- **链接: [http://arxiv.org/pdf/2506.15157v1](http://arxiv.org/pdf/2506.15157v1)**

> **作者:** Hanbit Oh; Andrea M. Salcedo-Vázquez; Ixchel G. Ramirez-Alpizar; Yukiyasu Domae
>
> **备注:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025 accepted
>
> **摘要:** Imitation learning (IL) aims to enable robots to perform tasks autonomously by observing a few human demonstrations. Recently, a variant of IL, called In-Context IL, utilized off-the-shelf large language models (LLMs) as instant policies that understand the context from a few given demonstrations to perform a new task, rather than explicitly updating network models with large-scale demonstrations. However, its reliability in the robotics domain is undermined by hallucination issues such as LLM-based instant policy, which occasionally generates poor trajectories that deviate from the given demonstrations. To alleviate this problem, we propose a new robust in-context imitation learning algorithm called the robust instant policy (RIP), which utilizes a Student's t-regression model to be robust against the hallucinated trajectories of instant policies to allow reliable trajectory generation. Specifically, RIP generates several candidate robot trajectories to complete a given task from an LLM and aggregates them using the Student's t-distribution, which is beneficial for ignoring outliers (i.e., hallucinations); thereby, a robust trajectory against hallucinations is generated. Our experiments, conducted in both simulated and real-world environments, show that RIP significantly outperforms state-of-the-art IL methods, with at least $26\%$ improvement in task success rates, particularly in low-data scenarios for everyday tasks. Video results available at https://sites.google.com/view/robustinstantpolicy.
>
---
#### [new 073] Advanced cervical cancer classification: enhancing pap smear images with hybrid PMD Filter-CLAHE
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在提升宫颈癌检测的准确性。通过融合PMD滤波与CLAHE增强技术改善涂片图像质量，从而提高CNN模型的分类性能。**

- **链接: [http://arxiv.org/pdf/2506.15489v1](http://arxiv.org/pdf/2506.15489v1)**

> **作者:** Ach Khozaimi; Isnani Darti; Syaiful Anam; Wuryansari Muharini Kusumawinahyu
>
> **摘要:** Cervical cancer remains a significant health problem, especially in developing countries. Early detection is critical for effective treatment. Convolutional neural networks (CNN) have shown promise in automated cervical cancer screening, but their performance depends on Pap smear image quality. This study investigates the impact of various image preprocessing techniques on CNN performance for cervical cancer classification using the SIPaKMeD dataset. Three preprocessing techniques were evaluated: perona-malik diffusion (PMD) filter for noise reduction, contrast-limited adaptive histogram equalization (CLAHE) for image contrast enhancement, and the proposed hybrid PMD filter-CLAHE approach. The enhanced image datasets were evaluated on pretrained models, such as ResNet-34, ResNet-50, SqueezeNet-1.0, MobileNet-V2, EfficientNet-B0, EfficientNet-B1, DenseNet-121, and DenseNet-201. The results show that hybrid preprocessing PMD filter-CLAHE can improve the Pap smear image quality and CNN architecture performance compared to the original images. The maximum metric improvements are 13.62% for accuracy, 10.04% for precision, 13.08% for recall, and 14.34% for F1-score. The proposed hybrid PMD filter-CLAHE technique offers a new perspective in improving cervical cancer classification performance using CNN architectures.
>
---
#### [new 074] Deploying and Evaluating Multiple Deep Learning Models on Edge Devices for Diabetic Retinopathy Detection
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决糖尿病视网膜病变的早期检测问题。通过在边缘设备上部署多个深度学习模型实现快速准确诊断。**

- **链接: [http://arxiv.org/pdf/2506.14834v1](http://arxiv.org/pdf/2506.14834v1)**

> **作者:** Akwasi Asare; Dennis Agyemanh Nana Gookyi; Derrick Boateng; Fortunatus Aabangbio Wulnye
>
> **摘要:** Diabetic Retinopathy (DR), a leading cause of vision impairment in individuals with diabetes, affects approximately 34.6% of diabetes patients globally, with the number of cases projected to reach 242 million by 2045. Traditional DR diagnosis relies on the manual examination of retinal fundus images, which is both time-consuming and resource intensive. This study presents a novel solution using Edge Impulse to deploy multiple deep learning models for real-time DR detection on edge devices. A robust dataset of over 3,662 retinal fundus images, sourced from the Kaggle EyePACS dataset, was curated, and enhanced through preprocessing techniques, including augmentation and normalization. Using TensorFlow, various Convolutional Neural Networks (CNNs), such as MobileNet, ShuffleNet, SqueezeNet, and a custom Deep Neural Network (DNN), were designed, trained, and optimized for edge deployment. The models were converted to TensorFlowLite and quantized to 8-bit integers to reduce their size and enhance inference speed, with minimal trade-offs in accuracy. Performance evaluations across different edge hardware platforms, including smartphones and microcontrollers, highlighted key metrics such as inference speed, accuracy, precision, and resource utilization. MobileNet achieved an accuracy of 96.45%, while SqueezeNet demonstrated strong real-time performance with a small model size of 176 KB and latency of just 17 ms on GPU. ShuffleNet and the custom DNN achieved moderate accuracy but excelled in resource efficiency, making them suitable for lower-end devices. This integration of edge AI technology into healthcare presents a scalable, cost-effective solution for early DR detection, providing timely and accurate diagnosis, especially in resource-constrained and remote healthcare settings.
>
---
#### [new 075] Recursive Variational Autoencoders for 3D Blood Vessel Generative Modeling
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于3D血管生成建模任务，旨在解决传统方法难以捕捉血管复杂性和多样性的问题。通过递归变分自编码器学习血管结构的低维表示，生成准确且多样的3D血管模型。**

- **链接: [http://arxiv.org/pdf/2506.14914v1](http://arxiv.org/pdf/2506.14914v1)**

> **作者:** Paula Feldman; Miguel Fainstein; Viviana Siless; Claudio Delrieux; Emmanuel Iarussi
>
> **摘要:** Anatomical trees play an important role in clinical diagnosis and treatment planning. Yet, accurately representing these structures poses significant challenges owing to their intricate and varied topology and geometry. Most existing methods to synthesize vasculature are rule based, and despite providing some degree of control and variation in the structures produced, they fail to capture the diversity and complexity of actual anatomical data. We developed a Recursive variational Neural Network (RvNN) that fully exploits the hierarchical organization of the vessel and learns a low-dimensional manifold encoding branch connectivity along with geometry features describing the target surface. After training, the RvNN latent space can be sampled to generate new vessel geometries. By leveraging the power of generative neural networks, we generate 3D models of blood vessels that are both accurate and diverse, which is crucial for medical and surgical training, hemodynamic simulations, and many other purposes. These results closely resemble real data, achieving high similarity in vessel radii, length, and tortuosity across various datasets, including those with aneurysms. To the best of our knowledge, this work is the first to utilize this technique for synthesizing blood vessels.
>
---
#### [new 076] Towards Perception-based Collision Avoidance for UAVs when Guiding the Visually Impaired
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于无人机辅助视障人士导航任务，旨在解决无人机在复杂环境中避障问题。工作包括设计感知路径规划系统和多DNN避障框架。**

- **链接: [http://arxiv.org/pdf/2506.14857v1](http://arxiv.org/pdf/2506.14857v1)**

> **作者:** Suman Raj; Swapnil Padhi; Ruchi Bhoot; Prince Modi; Yogesh Simmhan
>
> **备注:** 16 pages, 7 figures; Accepted as Late-Breaking Results at the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2023
>
> **摘要:** Autonomous navigation by drones using onboard sensors combined with machine learning and computer vision algorithms is impacting a number of domains, including agriculture, logistics, and disaster management. In this paper, we examine the use of drones for assisting visually impaired people (VIPs) in navigating through outdoor urban environments. Specifically, we present a perception-based path planning system for local planning around the neighborhood of the VIP, integrated with a global planner based on GPS and maps for coarse planning. We represent the problem using a geometric formulation and propose a multi DNN based framework for obstacle avoidance of the UAV as well as the VIP. Our evaluations conducted on a drone human system in a university campus environment verifies the feasibility of our algorithms in three scenarios; when the VIP walks on a footpath, near parked vehicles, and in a crowded street.
>
---
#### [new 077] Nabla-R2D3: Effective and Efficient 3D Diffusion Alignment with 2D Rewards
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文属于3D生成任务，旨在解决3D扩散模型对齐和优化问题。通过引入Nabla-R2D3框架，利用2D奖励信号高效提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.15684v1](http://arxiv.org/pdf/2506.15684v1)**

> **作者:** Qingming Liu; Zhen Liu; Dinghuai Zhang; Kui Jia
>
> **备注:** Technical Report (21 pages, 21 figures)
>
> **摘要:** Generating high-quality and photorealistic 3D assets remains a longstanding challenge in 3D vision and computer graphics. Although state-of-the-art generative models, such as diffusion models, have made significant progress in 3D generation, they often fall short of human-designed content due to limited ability to follow instructions, align with human preferences, or produce realistic textures, geometries, and physical attributes. In this paper, we introduce Nabla-R2D3, a highly effective and sample-efficient reinforcement learning alignment framework for 3D-native diffusion models using 2D rewards. Built upon the recently proposed Nabla-GFlowNet method, which matches the score function to reward gradients in a principled manner for reward finetuning, our Nabla-R2D3 enables effective adaptation of 3D diffusion models using only 2D reward signals. Extensive experiments show that, unlike vanilla finetuning baselines which either struggle to converge or suffer from reward hacking, Nabla-R2D3 consistently achieves higher rewards and reduced prior forgetting within a few finetuning steps.
>
---
#### [new 078] Pixel-level Certified Explanations via Randomized Smoothing
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于解释深度学习模型的任务，解决像素级解释不鲁棒的问题。通过随机平滑方法，提出首个保证像素级鲁棒性的框架，并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.15499v1](http://arxiv.org/pdf/2506.15499v1)**

> **作者:** Alaa Anani; Tobias Lorenz; Mario Fritz; Bernt Schiele
>
> **摘要:** Post-hoc attribution methods aim to explain deep learning predictions by highlighting influential input pixels. However, these explanations are highly non-robust: small, imperceptible input perturbations can drastically alter the attribution map while maintaining the same prediction. This vulnerability undermines their trustworthiness and calls for rigorous robustness guarantees of pixel-level attribution scores. We introduce the first certification framework that guarantees pixel-level robustness for any black-box attribution method using randomized smoothing. By sparsifying and smoothing attribution maps, we reformulate the task as a segmentation problem and certify each pixel's importance against $\ell_2$-bounded perturbations. We further propose three evaluation metrics to assess certified robustness, localization, and faithfulness. An extensive evaluation of 12 attribution methods across 5 ImageNet models shows that our certified attributions are robust, interpretable, and faithful, enabling reliable use in downstream tasks. Our code is at https://github.com/AlaaAnani/certified-attributions.
>
---
#### [new 079] Omnidirectional Video Super-Resolution using Deep Learning
- **分类: cs.MM; cs.CV; cs.LG**

- **简介: 该论文属于视频超分辨率任务，旨在解决360°视频空间分辨率不足的问题。通过构建数据集和提出新模型S3PO提升360°视频质量。**

- **链接: [http://arxiv.org/pdf/2506.14803v1](http://arxiv.org/pdf/2506.14803v1)**

> **作者:** Arbind Agrahari Baniya; Tsz-Kwan Lee; Peter W. Eklund; Sunil Aryal
>
> **摘要:** Omnidirectional Videos (or 360{\deg} videos) are widely used in Virtual Reality (VR) to facilitate immersive and interactive viewing experiences. However, the limited spatial resolution in 360{\deg} videos does not allow for each degree of view to be represented with adequate pixels, limiting the visual quality offered in the immersive experience. Deep learning Video Super-Resolution (VSR) techniques used for conventional videos could provide a promising software-based solution; however, these techniques do not tackle the distortion present in equirectangular projections of 360{\deg} video signals. An additional obstacle is the limited availability of 360{\deg} video datasets for study. To address these issues, this paper creates a novel 360{\deg} Video Dataset (360VDS) with a study of the extensibility of conventional VSR models to 360{\deg} videos. This paper further proposes a novel deep learning model for 360{\deg} Video Super-Resolution (360{\deg} VSR), called Spherical Signal Super-resolution with a Proportioned Optimisation (S3PO). S3PO adopts recurrent modelling with an attention mechanism, unbound from conventional VSR techniques like alignment. With a purpose-built feature extractor and a novel loss function addressing spherical distortion, S3PO outperforms most state-of-the-art conventional VSR models and 360{\deg}~specific super-resolution models on 360{\deg} video datasets. A step-wise ablation study is presented to understand and demonstrate the impact of the chosen architectural sub-components, targeted training and optimisation.
>
---
#### [new 080] One-shot Face Sketch Synthesis in the Wild via Generative Diffusion Prior and Instruction Tuning
- **分类: cs.GR; cs.CR; cs.CV; cs.CY**

- **简介: 该论文属于人脸草图生成任务，解决数据稀缺下的一次性草图合成问题。通过扩散模型和指令调优，实现高效、通用的草图生成。**

- **链接: [http://arxiv.org/pdf/2506.15312v1](http://arxiv.org/pdf/2506.15312v1)**

> **作者:** Han Wu; Junyao Li; Kangbo Zhao; Sen Zhang; Yukai Shi; Liang Lin
>
> **备注:** We propose a novel framework for face sketch synthesis, where merely a single pair of samples suffices to enable in-the-wild face sketch synthesis
>
> **摘要:** Face sketch synthesis is a technique aimed at converting face photos into sketches. Existing face sketch synthesis research mainly relies on training with numerous photo-sketch sample pairs from existing datasets. However, these large-scale discriminative learning methods will have to face problems such as data scarcity and high human labor costs. Once the training data becomes scarce, their generative performance significantly degrades. In this paper, we propose a one-shot face sketch synthesis method based on diffusion models. We optimize text instructions on a diffusion model using face photo-sketch image pairs. Then, the instructions derived through gradient-based optimization are used for inference. To simulate real-world scenarios more accurately and evaluate method effectiveness more comprehensively, we introduce a new benchmark named One-shot Face Sketch Dataset (OS-Sketch). The benchmark consists of 400 pairs of face photo-sketch images, including sketches with different styles and photos with different backgrounds, ages, sexes, expressions, illumination, etc. For a solid out-of-distribution evaluation, we select only one pair of images for training at each time, with the rest used for inference. Extensive experiments demonstrate that the proposed method can convert various photos into realistic and highly consistent sketches in a one-shot context. Compared to other methods, our approach offers greater convenience and broader applicability. The dataset will be available at: https://github.com/HanWu3125/OS-Sketch
>
---
#### [new 081] Embodied Web Agents: Bridging Physical-Digital Realms for Integrated Agent Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.MM; cs.RO**

- **简介: 该论文提出Embodied Web Agents，解决物理与数字智能融合问题。构建了集成3D环境与网络接口的仿真平台，发布基准测试任务，评估跨领域智能。**

- **链接: [http://arxiv.org/pdf/2506.15677v1](http://arxiv.org/pdf/2506.15677v1)**

> **作者:** Yining Hong; Rui Sun; Bingxuan Li; Xingcheng Yao; Maxine Wu; Alexander Chien; Da Yin; Ying Nian Wu; Zhecan James Wang; Kai-Wei Chang
>
> **摘要:** AI agents today are mostly siloed - they either retrieve and reason over vast amount of digital information and knowledge obtained online; or interact with the physical world through embodied perception, planning and action - but rarely both. This separation limits their ability to solve tasks that require integrated physical and digital intelligence, such as cooking from online recipes, navigating with dynamic map data, or interpreting real-world landmarks using web knowledge. We introduce Embodied Web Agents, a novel paradigm for AI agents that fluidly bridge embodiment and web-scale reasoning. To operationalize this concept, we first develop the Embodied Web Agents task environments, a unified simulation platform that tightly integrates realistic 3D indoor and outdoor environments with functional web interfaces. Building upon this platform, we construct and release the Embodied Web Agents Benchmark, which encompasses a diverse suite of tasks including cooking, navigation, shopping, tourism, and geolocation - all requiring coordinated reasoning across physical and digital realms for systematic assessment of cross-domain intelligence. Experimental results reveal significant performance gaps between state-of-the-art AI systems and human capabilities, establishing both challenges and opportunities at the intersection of embodied cognition and web-scale knowledge access. All datasets, codes and websites are publicly available at our project page https://embodied-web-agent.github.io/.
>
---
#### [new 082] An Empirical Study of Bugs in Data Visualization Libraries
- **分类: cs.SE; cs.CV; cs.HC**

- **简介: 该论文属于数据可视化库的缺陷分析任务，旨在解决视觉错误检测问题。通过分析564个bug，提出分类和测试方法，探索VLM在检测中的应用。**

- **链接: [http://arxiv.org/pdf/2506.15084v1](http://arxiv.org/pdf/2506.15084v1)**

> **作者:** Weiqi Lu; Yongqiang Tian; Xiaohan Zhong; Haoyang Ma; Zhenyang Xu; Shing-Chi Cheung; Chengnian Sun
>
> **备注:** Proc. ACM Softw. Eng. 2, FSE
>
> **摘要:** Data visualization (DataViz) libraries play a crucial role in presentation, data analysis, and application development, underscoring the importance of their accuracy in transforming data into visual representations. Incorrect visualizations can adversely impact user experience, distort information conveyance, and influence user perception and decision-making processes. Visual bugs in these libraries can be particularly insidious as they may not cause obvious errors like crashes, but instead mislead users of the underlying data graphically, resulting in wrong decision making. Consequently, a good understanding of the unique characteristics of bugs in DataViz libraries is essential for researchers and developers to detect and fix bugs in DataViz libraries. This study presents the first comprehensive analysis of bugs in DataViz libraries, examining 564 bugs collected from five widely-used libraries. Our study systematically analyzes their symptoms and root causes, and provides a detailed taxonomy. We found that incorrect/inaccurate plots are pervasive in DataViz libraries and incorrect graphic computation is the major root cause, which necessitates further automated testing methods for DataViz libraries. Moreover, we identified eight key steps to trigger such bugs and two test oracles specific to DataViz libraries, which may inspire future research in designing effective automated testing techniques. Furthermore, with the recent advancements in Vision Language Models (VLMs), we explored the feasibility of applying these models to detect incorrect/inaccurate plots. The results show that the effectiveness of VLMs in bug detection varies from 29% to 57%, depending on the prompts, and adding more information in prompts does not necessarily increase the effectiveness. More findings can be found in our manuscript.
>
---
#### [new 083] CACTUS as a Reliable Tool for Early Classification of Age-related Macular Degeneration
- **分类: cs.LG; cs.CV; stat.AP**

- **简介: 该论文属于疾病分类任务，旨在解决AMD早期诊断问题。提出CACTUS工具，结合多源数据提升分类准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.14843v1](http://arxiv.org/pdf/2506.14843v1)**

> **作者:** Luca Gherardini; Imre Lengyel; Tunde Peto; Caroline C. W. Klaverd; Magda A. Meester-Smoord; Johanna Maria Colijnd; EYE-RISK Consortium; E3 Consortium; Jose Sousa
>
> **摘要:** Machine Learning (ML) is used to tackle various tasks, such as disease classification and prediction. The effectiveness of ML models relies heavily on having large amounts of complete data. However, healthcare data is often limited or incomplete, which can hinder model performance. Additionally, issues like the trustworthiness of solutions vary with the datasets used. The lack of transparency in some ML models further complicates their understanding and use. In healthcare, particularly in the case of Age-related Macular Degeneration (AMD), which affects millions of older adults, early diagnosis is crucial due to the absence of effective treatments for reversing progression. Diagnosing AMD involves assessing retinal images along with patients' symptom reports. There is a need for classification approaches that consider genetic, dietary, clinical, and demographic factors. Recently, we introduced the -Comprehensive Abstraction and Classification Tool for Uncovering Structures-(CACTUS), aimed at improving AMD stage classification. CACTUS offers explainability and flexibility, outperforming standard ML models. It enhances decision-making by identifying key factors and providing confidence in its results. The important features identified by CACTUS allow us to compare with existing medical knowledge. By eliminating less relevant or biased data, we created a clinical scenario for clinicians to offer feedback and address biases.
>
---
#### [new 084] Classification of Multi-Parametric Body MRI Series Using Deep Learning
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决mpMRI系列类型识别问题。通过深度学习模型（如DenseNet-121）进行分类，提升放射科医生的诊断效率。**

- **链接: [http://arxiv.org/pdf/2506.15182v1](http://arxiv.org/pdf/2506.15182v1)**

> **作者:** Boah Kim; Tejas Sudharshan Mathai; Kimberly Helm; Peter A. Pinto; Ronald M. Summers
>
> **摘要:** Multi-parametric magnetic resonance imaging (mpMRI) exams have various series types acquired with different imaging protocols. The DICOM headers of these series often have incorrect information due to the sheer diversity of protocols and occasional technologist errors. To address this, we present a deep learning-based classification model to classify 8 different body mpMRI series types so that radiologists read the exams efficiently. Using mpMRI data from various institutions, multiple deep learning-based classifiers of ResNet, EfficientNet, and DenseNet are trained to classify 8 different MRI series, and their performance is compared. Then, the best-performing classifier is identified, and its classification capability under the setting of different training data quantities is studied. Also, the model is evaluated on the out-of-training-distribution datasets. Moreover, the model is trained using mpMRI exams obtained from different scanners in two training strategies, and its performance is tested. Experimental results show that the DenseNet-121 model achieves the highest F1-score and accuracy of 0.966 and 0.972 over the other classification models with p-value$<$0.05. The model shows greater than 0.95 accuracy when trained with over 729 studies of the training data, whose performance improves as the training data quantities grew larger. On the external data with the DLDS and CPTAC-UCEC datasets, the model yields 0.872 and 0.810 accuracy for each. These results indicate that in both the internal and external datasets, the DenseNet-121 model attains high accuracy for the task of classifying 8 body MRI series types.
>
---
#### [new 085] An accurate and revised version of optical character recognition-based speech synthesis using LabVIEW
- **分类: cs.SD; cs.CL; cs.CV; eess.AS; 14J60; I.2.7; I.4; I.5; I.7.5**

- **简介: 该论文属于语音合成任务，旨在解决视障人士获取书籍困难的问题。通过OCR技术与LabVIEW实现准确的语音转换系统。**

- **链接: [http://arxiv.org/pdf/2506.15029v1](http://arxiv.org/pdf/2506.15029v1)**

> **作者:** Prateek Mehta; Anasuya Patil
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Knowledge extraction through sound is a distinctive property. Visually impaired individuals often rely solely on Braille books and audio recordings provided by NGOs. Due to limitations in these approaches, blind individuals often cannot access books of their choice. Speech is a more effective mode of communication than text for blind and visually impaired persons, as they can easily respond to sounds. This paper presents the development of an accurate, reliable, cost-effective, and user-friendly optical character recognition (OCR)-based speech synthesis system. The OCR-based system has been implemented using Laboratory Virtual Instrument Engineering Workbench (LabVIEW).
>
---
#### [new 086] A Real-time Endoscopic Image Denoising System
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于图像去噪任务，旨在解决微型内窥镜图像噪声问题。通过建立噪声模型并结合传统与学习方法，实现高效实时去噪。**

- **链接: [http://arxiv.org/pdf/2506.15395v1](http://arxiv.org/pdf/2506.15395v1)**

> **作者:** Yu Xing; Shishi Huang; Meng Lv; Guo Chen; Huailiang Wang; Lingzhi Sui
>
> **摘要:** Endoscopes featuring a miniaturized design have significantly enhanced operational flexibility, portability, and diagnostic capability while substantially reducing the invasiveness of medical procedures. Recently, single-use endoscopes equipped with an ultra-compact analogue image sensor measuring less than 1mm x 1mm bring revolutionary advancements to medical diagnosis. They reduce the structural redundancy and large capital expenditures associated with reusable devices, eliminate the risk of patient infections caused by inadequate disinfection, and alleviate patient suffering. However, the limited photosensitive area results in reduced photon capture per pixel, requiring higher photon sensitivity settings to maintain adequate brightness. In high-contrast medical imaging scenarios, the small-sized sensor exhibits a constrained dynamic range, making it difficult to simultaneously capture details in both highlights and shadows, and additional localized digital gain is required to compensate. Moreover, the simplified circuit design and analog signal transmission introduce additional noise sources. These factors collectively contribute to significant noise issues in processed endoscopic images. In this work, we developed a comprehensive noise model for analog image sensors in medical endoscopes, addressing three primary noise types: fixed-pattern noise, periodic banding noise, and mixed Poisson-Gaussian noise. Building on this analysis, we propose a hybrid denoising system that synergistically combines traditional image processing algorithms with advanced learning-based techniques for captured raw frames from sensors. Experiments demonstrate that our approach effectively reduces image noise without fine detail loss or color distortion, while achieving real-time performance on FPGA platforms and an average PSNR improvement from 21.16 to 33.05 on our test dataset.
>
---
#### [new 087] Foundation Artificial Intelligence Models for Health Recognition Using Face Photographs (FAHR-Face)
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于健康识别任务，旨在通过面部照片预测生物年龄和生存风险。研究构建了FAHR-Face模型，进行了多方面验证与优化。**

- **链接: [http://arxiv.org/pdf/2506.14909v1](http://arxiv.org/pdf/2506.14909v1)**

> **作者:** Fridolin Haugg; Grace Lee; John He; Leonard Nürnberg; Dennis Bontempi; Danielle S. Bitterman; Paul Catalano; Vasco Prudente; Dmitrii Glubokov; Andrew Warrington; Suraj Pai; Dirk De Ruysscher; Christian Guthier; Benjamin H. Kann; Vadim N. Gladyshev; Hugo JWL Aerts; Raymond H. Mak
>
> **摘要:** Background: Facial appearance offers a noninvasive window into health. We built FAHR-Face, a foundation model trained on >40 million facial images and fine-tuned it for two distinct tasks: biological age estimation (FAHR-FaceAge) and survival risk prediction (FAHR-FaceSurvival). Methods: FAHR-FaceAge underwent a two-stage, age-balanced fine-tuning on 749,935 public images; FAHR-FaceSurvival was fine-tuned on 34,389 photos of cancer patients. Model robustness (cosmetic surgery, makeup, pose, lighting) and independence (saliency mapping) was tested extensively. Both models were clinically tested in two independent cancer patient datasets with survival analyzed by multivariable Cox models and adjusted for clinical prognostic factors. Findings: For age estimation, FAHR-FaceAge had the lowest mean absolute error of 5.1 years on public datasets, outperforming benchmark models and maintaining accuracy across the full human lifespan. In cancer patients, FAHR-FaceAge outperformed a prior facial age estimation model in survival prognostication. FAHR-FaceSurvival demonstrated robust prediction of mortality, and the highest-risk quartile had more than triple the mortality of the lowest (adjusted hazard ratio 3.22; P<0.001). These findings were validated in the independent cohort and both models showed generalizability across age, sex, race and cancer subgroups. The two algorithms provided distinct, complementary prognostic information; saliency mapping revealed each model relied on distinct facial regions. The combination of FAHR-FaceAge and FAHR-FaceSurvival improved prognostic accuracy. Interpretation: A single foundation model can generate inexpensive, scalable facial biomarkers that capture both biological ageing and disease-related mortality risk. The foundation model enabled effective training using relatively small clinical datasets.
>
---
#### [new 088] Empirical Studies of Large Scale Environment Scanning by Consumer Electronics
- **分类: eess.IV; cs.CV; cs.ET; cs.MM**

- **简介: 该论文属于3D环境重建任务，研究消费级设备Matterport Pro3在大规模场景中的性能，评估其效果并提出改进方法。**

- **链接: [http://arxiv.org/pdf/2506.14771v1](http://arxiv.org/pdf/2506.14771v1)**

> **作者:** Mengyuan Wang; Yang Liu; Haopeng Wang; Haiwei Dong; Abdulmotaleb El Saddik
>
> **备注:** Accepted by IEEE Consumer Electronics Magazine
>
> **摘要:** This paper presents an empirical evaluation of the Matterport Pro3, a consumer-grade 3D scanning device, for large-scale environment reconstruction. We conduct detailed scanning (1,099 scanning points) of a six-floor building (17,567 square meters) and assess the device's effectiveness, limitations, and performance enhancements in diverse scenarios. Challenges encountered during the scanning are addressed through proposed solutions, while we also explore advanced methods to overcome them more effectively. Comparative analysis with another consumer-grade device (iPhone) highlights the Pro3's balance between cost-effectiveness and performance. The Matterport Pro3 achieves a denser point cloud with 1,877,324 points compared to the iPhone's 506,961 points and higher alignment accuracy with an RMSE of 0.0118 meters. The cloud-to-cloud (C2C) average distance error between the two point cloud models is 0.0408 meters, with a standard deviation of 0.0715 meters. The study demonstrates the Pro3's ability to generate high-quality 3D models suitable for large-scale applications, leveraging features such as LiDAR and advanced alignment techniques.
>
---
#### [new 089] PIPE: Physics-Informed Position Encoding for Alignment of Satellite Images and Time Series
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多模态时间序列预测任务，旨在解决视觉数据与时间序列对齐问题。提出PIPE方法，通过物理信息编码提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.14786v1](http://arxiv.org/pdf/2506.14786v1)**

> **作者:** Haobo Li; Eunseo Jung; Zixin Chen; Zhaowei Wang; Yueya Wang; Huamin Qu; Alexis Kai Hon Lau
>
> **摘要:** Multimodal time series forecasting is foundational in various fields, such as utilizing satellite imagery and numerical data for predicting typhoons in climate science. However, existing multimodal approaches primarily focus on utilizing text data to help time series forecasting, leaving the visual data in existing time series datasets untouched. Furthermore, it is challenging for models to effectively capture the physical information embedded in visual data, such as satellite imagery's temporal and geospatial context, which extends beyond images themselves. To address this gap, we propose physics-informed positional encoding (PIPE), a lightweight method that embeds physical information into vision language models (VLMs). PIPE introduces two key innovations: (1) a physics-informed positional indexing scheme for mapping physics to positional IDs, and (2) a variant-frequency positional encoding mechanism for encoding frequency information of physical variables and sequential order of tokens within the embedding space. By preserving both the physical information and sequential order information, PIPE significantly improves multimodal alignment and forecasting accuracy. Through the experiments on the most representative and the largest open-sourced satellite image dataset, PIPE achieves state-of-the-art performance in both deep learning forecasting and climate domain methods, demonstrating superiority across benchmarks, including a 12% improvement in typhoon intensity forecasting over prior works. Our code is provided in the supplementary material.
>
---
#### [new 090] Reinforcing VLMs to Use Tools for Detailed Visual Reasoning Under Resource Constraints
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决资源受限下VLMs的详细视觉推理问题。通过训练小模型结合外部工具，提升推理性能。**

- **链接: [http://arxiv.org/pdf/2506.14821v1](http://arxiv.org/pdf/2506.14821v1)**

> **作者:** Sunil Kumar; Bowen Zhao; Leo Dirac; Paulina Varshavskaya
>
> **摘要:** Despite tremendous recent advances in large model reasoning ability, vision-language models (VLMs) still struggle with detailed visual reasoning, especially when compute resources are limited. To address this challenge, we draw inspiration from methods like Deepseek-r1 for VLMs and train smaller-scale models with Group Relative Policy Optimization (GRPO) to use external tools such as zoom. The greatest benefit is obtained with a combination of GRPO learning, a simple reward structure, a simplified tool-calling interface, allocating additional tokens to the result of the tool call, and a training data mix that over-represents visually difficult examples. Compared to similarly-sized baseline models, our method achieves better performance on some visual question-answering (VQA) tasks, thanks to the detailed visual information gathered from the external tool.
>
---
#### [new 091] pycnet-audio: A Python package to support bioacoustics data processing
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文介绍了一个用于生物声学数据处理的Python包pycnet-audio，旨在解决大规模音频数据自动化分析问题，支持物种叫声检测与环境噪声识别。**

- **链接: [http://arxiv.org/pdf/2506.14864v1](http://arxiv.org/pdf/2506.14864v1)**

> **作者:** Zachary J. Ruff; Damon B. Lesmeister
>
> **摘要:** Passive acoustic monitoring is an emerging approach in wildlife research that leverages recent improvements in purpose-made automated recording units (ARUs). The general approach is to deploy ARUs in the field to record on a programmed schedule for extended periods (weeks or months), after which the audio data are retrieved. These data must then be processed, typically either by measuring or analyzing characteristics of the audio itself (e.g. calculating acoustic indices), or by searching for some signal of interest within the recordings, e.g. vocalizations or other sounds produced by some target species, anthropogenic or environmental noise, etc. In the latter case, some method is required to locate the signal(s) of interest within the audio. While very small datasets can simply be searched manually, even modest projects can produce audio datasets on the order of 105 hours of recordings, making manual review impractical and necessitating some form of automated detection. pycnet-audio (Ruff 2024) is intended to provide a practical processing workflow for acoustic data, built around the PNW-Cnet model, which was initially developed by the U.S. Forest Service to support population monitoring of northern spotted owls (Strix occidentalis caurina) and other forest owls (Lesmeister and Jenkins 2022; Ruff et al. 2020). PNW-Cnet has been expanded to detect vocalizations of ca. 80 forest wildlife species and numerous forms of anthropogenic and environmental noise (Ruff et al. 2021, 2023).
>
---
#### [new 092] Improving Prostate Gland Segmenting Using Transformer based Architectures
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在解决前列腺腺体自动分割中的读者差异和跨机构数据偏差问题。通过对比Transformer模型与传统CNN模型，验证了其在精度和鲁棒性上的优势。**

- **链接: [http://arxiv.org/pdf/2506.14844v1](http://arxiv.org/pdf/2506.14844v1)**

> **作者:** Shatha Abudalou
>
> **摘要:** Inter reader variability and cross site domain shift challenge the automatic segmentation of prostate anatomy using T2 weighted MRI images. This study investigates whether transformer models can retain precision amid such heterogeneity. We compare the performance of UNETR and SwinUNETR in prostate gland segmentation against our previous 3D UNet model [1], based on 546 MRI (T2weighted) volumes annotated by two independent experts. Three training strategies were analyzed: single cohort dataset, 5 fold cross validated mixed cohort, and gland size based dataset. Hyperparameters were tuned by Optuna. The test set, from an independent population of readers, served as the evaluation endpoint (Dice Similarity Coefficient). In single reader training, SwinUNETR achieved an average dice score of 0.816 for Reader#1 and 0.860 for Reader#2, while UNETR scored 0.8 and 0.833 for Readers #1 and #2, respectively, compared to the baseline UNets 0.825 for Reader #1 and 0.851 for Reader #2. SwinUNETR had an average dice score of 0.8583 for Reader#1 and 0.867 for Reader#2 in cross-validated mixed training. For the gland size-based dataset, SwinUNETR achieved an average dice score of 0.902 for Reader#1 subset and 0.894 for Reader#2, using the five-fold mixed training strategy (Reader#1, n=53; Reader#2, n=87) at larger gland size-based subsets, where UNETR performed poorly. Our findings demonstrate that global and shifted-window self-attention effectively reduces label noise and class imbalance sensitivity, resulting in improvements in the Dice score over CNNs by up to five points while maintaining computational efficiency. This contributes to the high robustness of SwinUNETR for clinical deployment.
>
---
#### [new 093] MCOO-SLAM: A Multi-Camera Omnidirectional Object SLAM System
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于SLAM任务，旨在解决大场景下对象建模不准确的问题。通过多视角相机和语义融合策略，提升对象关联与地图一致性。**

- **链接: [http://arxiv.org/pdf/2506.15402v1](http://arxiv.org/pdf/2506.15402v1)**

> **作者:** Miaoxin Pan; Jinnan Li; Yaowen Zhang; Yi Yang; Yufeng Yue
>
> **摘要:** Object-level SLAM offers structured and semantically meaningful environment representations, making it more interpretable and suitable for high-level robotic tasks. However, most existing approaches rely on RGB-D sensors or monocular views, which suffer from narrow fields of view, occlusion sensitivity, and limited depth perception-especially in large-scale or outdoor environments. These limitations often restrict the system to observing only partial views of objects from limited perspectives, leading to inaccurate object modeling and unreliable data association. In this work, we propose MCOO-SLAM, a novel Multi-Camera Omnidirectional Object SLAM system that fully leverages surround-view camera configurations to achieve robust, consistent, and semantically enriched mapping in complex outdoor scenarios. Our approach integrates point features and object-level landmarks enhanced with open-vocabulary semantics. A semantic-geometric-temporal fusion strategy is introduced for robust object association across multiple views, leading to improved consistency and accurate object modeling, and an omnidirectional loop closure module is designed to enable viewpoint-invariant place recognition using scene-level descriptors. Furthermore, the constructed map is abstracted into a hierarchical 3D scene graph to support downstream reasoning tasks. Extensive experiments in real-world demonstrate that MCOO-SLAM achieves accurate localization and scalable object-level mapping with improved robustness to occlusion, pose variation, and environmental complexity.
>
---
## 更新

#### [replaced 001] ImmerseGen: Agent-Guided Immersive World Generation with Alpha-Textured Proxies
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14315v2](http://arxiv.org/pdf/2506.14315v2)**

> **作者:** Jinyan Yuan; Bangbang Yang; Keke Wang; Panwang Pan; Lin Ma; Xuehai Zhang; Xiao Liu; Zhaopeng Cui; Yuewen Ma
>
> **备注:** Project webpage: https://immersegen.github.io
>
> **摘要:** Automatic creation of 3D scenes for immersive VR presence has been a significant research focus for decades. However, existing methods often rely on either high-poly mesh modeling with post-hoc simplification or massive 3D Gaussians, resulting in a complex pipeline or limited visual realism. In this paper, we demonstrate that such exhaustive modeling is unnecessary for achieving compelling immersive experience. We introduce ImmerseGen, a novel agent-guided framework for compact and photorealistic world modeling. ImmerseGen represents scenes as hierarchical compositions of lightweight geometric proxies, i.e., simplified terrain and billboard meshes, and generates photorealistic appearance by synthesizing RGBA textures onto these proxies. Specifically, we propose terrain-conditioned texturing for user-centric base world synthesis, and RGBA asset texturing for midground and foreground scenery. This reformulation offers several advantages: (i) it simplifies modeling by enabling agents to guide generative models in producing coherent textures that integrate seamlessly with the scene; (ii) it bypasses complex geometry creation and decimation by directly synthesizing photorealistic textures on proxies, preserving visual quality without degradation; (iii) it enables compact representations suitable for real-time rendering on mobile VR headsets. To automate scene creation from text prompts, we introduce VLM-based modeling agents enhanced with semantic grid-based analysis for improved spatial reasoning and accurate asset placement. ImmerseGen further enriches scenes with dynamic effects and ambient audio to support multisensory immersion. Experiments on scene generation and live VR showcases demonstrate that ImmerseGen achieves superior photorealism, spatial coherence and rendering efficiency compared to prior methods. Project webpage: https://immersegen.github.io.
>
---
#### [replaced 002] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13589v2](http://arxiv.org/pdf/2506.13589v2)**

> **作者:** Zhucun Xue; Jiangning Zhang; Xurong Xie; Yuxuan Cai; Yong Liu; Xiangtai Li; Dacheng Tao
>
> **摘要:** Multimodal Large Language Models (MLLMs) struggle with long videos due to fixed context windows and weak long-term dependency modeling. Existing Retrieval-Augmented Generation (RAG) methods for videos use static retrieval strategies, leading to inefficiencies for simple queries and information loss for complex tasks. To address this, we propose AdaVideoRAG, a novel framework that dynamically adapts retrieval granularity based on query complexity using a lightweight intent classifier. Our framework employs an Omni-Knowledge Indexing module to build hierarchical databases from text (captions, ASR, OCR), visual features, and semantic graphs, enabling optimal resource allocation across tasks. We also introduce the HiVU benchmark for comprehensive evaluation. Experiments demonstrate improved efficiency and accuracy for long-video understanding, with seamless integration into existing MLLMs. AdaVideoRAG establishes a new paradigm for adaptive retrieval in video analysis. Codes will be open-sourced at https://github.com/xzc-zju/AdaVideoRAG.
>
---
#### [replaced 003] A Comprehensive Survey on Continual Learning in Generative Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13045v2](http://arxiv.org/pdf/2506.13045v2)**

> **作者:** Haiyang Guo; Fanhu Zeng; Fei Zhu; Jiayi Wang; Xukai Wang; Jingang Zhou; Hongbo Zhao; Wenzhuo Liu; Shijie Ma; Da-Han Wang; Xu-Yao Zhang; Cheng-Lin Liu
>
> **备注:** Preprint
>
> **摘要:** The rapid advancement of generative models has enabled modern AI systems to comprehend and produce highly sophisticated content, even achieving human-level performance in specific domains. However, these models remain fundamentally constrained by catastrophic forgetting - a persistent challenge where adapting to new tasks typically leads to significant degradation in performance on previously learned tasks. To address this practical limitation, numerous approaches have been proposed to enhance the adaptability and scalability of generative models in real-world applications. In this work, we present a comprehensive survey of continual learning methods for mainstream generative models, including large language models, multimodal large language models, vision language action models, and diffusion models. Drawing inspiration from the memory mechanisms of the human brain, we systematically categorize these approaches into three paradigms: architecture-based, regularization-based, and replay-based methods, while elucidating their underlying methodologies and motivations. We further analyze continual learning setups for different generative models, including training objectives, benchmarks, and core backbones, offering deeper insights into the field. The project page of this paper is available at https://github.com/Ghy0501/Awesome-Continual-Learning-in-Generative-Models.
>
---
#### [replaced 004] EgoBlind: Towards Egocentric Visual Assistance for the Blind
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.08221v2](http://arxiv.org/pdf/2503.08221v2)**

> **作者:** Junbin Xiao; Nanxin Huang; Hao Qiu; Zhulin Tao; Xun Yang; Richang Hong; Meng Wang; Angela Yao
>
> **备注:** We extend and resplit the dataset
>
> **摘要:** We present EgoBlind, the first egocentric VideoQA dataset collected from blind individuals to evaluate the assistive capabilities of contemporary multimodal large language models (MLLMs). EgoBlind comprises 1,392 videos that record the daily lives of real blind users from a first-person perspective. It also features 5,311 questions directly posed or generated and verified by blind individuals to reflect their in-situation needs for visual assistance under various scenarios. We provide each question with an average of 3 reference answers to alleviate subjective evaluation. Using EgoBlind, we comprehensively evaluate 16 advanced MLLMs and find that all models struggle, with the best performers achieving accuracy near 60\%, far behind human performance of 87.4\%. To guide future advancements, we identify and summarize major limitations of existing MLLMs in egocentric visual assistance for the blind and explore heuristic solutions for improvement. With these efforts, we hope EgoBlind can serve as a valuable foundation for developing more effective AI assistants to enhance the independence of the blind individuals' lives. Data and evaluation code are available at https://github.com/doc-doc/EgoBlind.
>
---
#### [replaced 005] Instance-Adaptive Keypoint Learning with Local-to-Global Geometric Aggregation for Category-Level Object Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.15134v3](http://arxiv.org/pdf/2504.15134v3)**

> **作者:** Xiao Zhang; Lu Zou; Tao Lu; Yuan Yao; Zhangjin Huang; Guoping Wang
>
> **摘要:** Category-level object pose estimation aims to predict the 6D pose and size of previously unseen instances from predefined categories, requiring strong generalization across diverse object instances. Although many previous methods attempt to mitigate intra-class variations, they often struggle with instances exhibiting complex geometries or significant deviations from canonical shapes. To address this issue, we propose INKL-Pose, a novel category-level object pose estimation framework that enables INstance-adaptive Keypoint Learning with local-to-global geometric aggregation. Specifically, our method first predicts semantically consistent and geometrically informative keypoints using an Instance-Adaptive Keypoint Detector, then refines them: (1) a Local Keypoint Feature Aggregator capturing fine-grained geometries, and (2) a Global Keypoint Feature Aggregator using bidirectional Mamba for structural consistency. To enable bidirectional modeling in Mamba, we introduce a simple yet effective Feature Sequence Flipping strategy that preserves spatial coherence while constructing backward feature sequence. Additionally, we design a surface loss and a separation loss to encourage uniform coverage and spatial diversity in keypoint distribution. The resulting keypoints are mapped to a canonical space for 6D pose and size regression. Extensive experiments on CAMERA25, REAL275, and HouseCat6D show that INKL-Pose achieves state-of-the-art performance with 16.7M parameters and runs at 36 FPS on an NVIDIA RTX 4090D GPU.
>
---
#### [replaced 006] Patch distribution modeling framework adaptive cosine estimator (PaDiM-ACE) for anomaly detection and localization in synthetic aperture radar imagery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.08049v3](http://arxiv.org/pdf/2504.08049v3)**

> **作者:** Angelina Ibarra; Joshua Peeples
>
> **备注:** Accepted to SPIE, Defense and Commercial Sensing, Algorithms for Synthetic Aperture Radar Imagery XXXII (April 2025)
>
> **摘要:** This work presents a new approach to anomaly detection and localization in synthetic aperture radar imagery (SAR), expanding upon the existing patch distribution modeling framework (PaDiM). We introduce the adaptive cosine estimator (ACE) detection statistic. PaDiM uses the Mahalanobis distance at inference, an unbounded metric. ACE instead uses the cosine similarity metric, providing bounded anomaly detection scores. The proposed method is evaluated across multiple SAR datasets, with performance metrics including the area under the receiver operating curve (AUROC) at the image and pixel level, aiming for increased performance in anomaly detection and localization of SAR imagery. The code is publicly available: https://github.com/Advanced-Vision-and-Learning-Lab/PaDiM-ACE.
>
---
#### [replaced 007] A dataset of high-resolution plantar pressures for gait analysis across varying footwear and walking speeds
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.17244v4](http://arxiv.org/pdf/2502.17244v4)**

> **作者:** Robyn Larracy; Angkoon Phinyomark; Ala Salehi; Eve MacDonald; Saeed Kazemi; Shikder Shafiul Bashar; Aaron Tabor; Erik Scheme
>
> **摘要:** Gait refers to the patterns of limb movement generated during walking, which are unique to each individual due to both physical and behavioral traits. Walking patterns have been widely studied in biometrics, biomechanics, sports, and rehabilitation. While traditional methods rely on video and motion capture, advances in plantar pressure sensing technology now offer deeper insights into gait. However, underfoot pressures during walking remain underexplored due to the lack of large, publicly accessible datasets. To address this, we introduce the UNB StepUP-P150 dataset: a footStep database for gait analysis and recognition using Underfoot Pressure, including data from 150 individuals. This dataset comprises high-resolution plantar pressure data (4 sensors per cm-squared) collected using a 1.2m by 3.6m pressure-sensing walkway. It contains over 200,000 footsteps from participants walking with various speeds (preferred, slow-to-stop, fast, and slow) and footwear conditions (barefoot, standard shoes, and two personal shoes), supporting advancements in biometric gait recognition and presenting new research opportunities in biomechanics and deep learning. UNB StepUP-P150 establishes a new benchmark for plantar pressure-based gait analysis and recognition.
>
---
#### [replaced 008] Exploring Personalized Federated Learning Architectures for Violence Detection in Surveillance Videos
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.00857v2](http://arxiv.org/pdf/2504.00857v2)**

> **作者:** Mohammad Kassir; Siba Haidar; Antoun Yaacoub
>
> **备注:** 7 pages, 5 figures, 4 tables
>
> **摘要:** The challenge of detecting violent incidents in urban surveillance systems is compounded by the voluminous and diverse nature of video data. This paper presents a targeted approach using Personalized Federated Learning (PFL) to address these issues, specifically employing the Federated Learning with Personalization Layers method within the Flower framework. Our methodology adapts learning models to the unique data characteristics of each surveillance node, effectively managing the heterogeneous and non-IID nature of surveillance video data. Through rigorous experiments conducted on balanced and imbalanced datasets, our PFL models demonstrated enhanced accuracy and efficiency, achieving up to 99.3% accuracy. This study underscores the potential of PFL to significantly improve the scalability and effectiveness of surveillance systems, offering a robust, privacy-preserving solution for violence detection in complex urban environments.
>
---
#### [replaced 009] PLD: A Choice-Theoretic List-Wise Knowledge Distillation
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2506.12542v2](http://arxiv.org/pdf/2506.12542v2)**

> **作者:** Ejafa Bassam; Dawei Zhu; Kaigui Bian
>
> **摘要:** Knowledge distillation is a model compression technique in which a compact "student" network is trained to replicate the predictive behavior of a larger "teacher" network. In logit-based knowledge distillation it has become the de facto approach to augment cross-entropy with a distillation term. Typically this term is either a KL divergence-matching marginal probabilities or a correlation-based loss capturing intra- and inter-class relationships but in every case it sits as an add-on to cross-entropy with its own weight that must be carefully tuned. In this paper we adopt a choice-theoretic perspective and recast knowledge distillation under the Plackett-Luce model by interpreting teacher logits as "worth" scores. We introduce Plackett-Luce Distillation (PLD), a weighted list-wise ranking loss in which the teacher model transfers knowledge of its full ranking of classes, weighting each ranked choice by its own confidence. PLD directly optimizes a single teacher-optimal ranking of the true label first, followed by the remaining classes in descending teacher confidence, yielding a convex, translation-invariant surrogate that subsumes weighted cross-entropy. Empirically on standard image classification benchmarks, PLD improves Top-1 accuracy by an average of +0.42% over DIST (arXiv:2205.10536) and +1.04% over KD (arXiv:1503.02531) in homogeneous settings and by +0.48% and +1.09% over DIST and KD, respectively, in heterogeneous settings.
>
---
#### [replaced 010] Style-Preserving Lip Sync via Audio-Aware Style Reference
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2408.05412v2](http://arxiv.org/pdf/2408.05412v2)**

> **作者:** Weizhi Zhong; Jichang Li; Yinqi Cai; Ming Li; Feng Gao; Liang Lin; Guanbin Li
>
> **备注:** submitted to IEEE Transactions on Multimedia(TMM)
>
> **摘要:** Audio-driven lip sync has recently drawn significant attention due to its widespread application in the multimedia domain. Individuals exhibit distinct lip shapes when speaking the same utterance, attributed to the unique speaking styles of individuals, posing a notable challenge for audio-driven lip sync. Earlier methods for such task often bypassed the modeling of personalized speaking styles, resulting in sub-optimal lip sync conforming to the general styles. Recent lip sync techniques attempt to guide the lip sync for arbitrary audio by aggregating information from a style reference video, yet they can not preserve the speaking styles well due to their inaccuracy in style aggregation. This work proposes an innovative audio-aware style reference scheme that effectively leverages the relationships between input audio and reference audio from style reference video to address the style-preserving audio-driven lip sync. Specifically, we first develop an advanced Transformer-based model adept at predicting lip motion corresponding to the input audio, augmented by the style information aggregated through cross-attention layers from style reference video. Afterwards, to better render the lip motion into realistic talking face video, we devise a conditional latent diffusion model, integrating lip motion through modulated convolutional layers and fusing reference facial images via spatial cross-attention layers. Extensive experiments validate the efficacy of the proposed approach in achieving precise lip sync, preserving speaking styles, and generating high-fidelity, realistic talking face videos.
>
---
#### [replaced 011] Hierarchical Multi-Positive Contrastive Learning for Patent Image Retrieval
- **分类: cs.CV; cs.IR; cs.LG; 68T45, 68T07; H.3.3; I.4.10; I.2.10**

- **链接: [http://arxiv.org/pdf/2506.13496v2](http://arxiv.org/pdf/2506.13496v2)**

> **作者:** Kshitij Kavimandan; Angelos Nalmpantis; Emma Beauxis-Aussalet; Robert-Jan Sips
>
> **备注:** 5 pages, 3 figures, Accepted as a short paper at the 6th Workshop on Patent Text Mining and Semantic Technologies (PatentSemTech 2025), co-located with SIGIR 2025
>
> **摘要:** Patent images are technical drawings that convey information about a patent's innovation. Patent image retrieval systems aim to search in vast collections and retrieve the most relevant images. Despite recent advances in information retrieval, patent images still pose significant challenges due to their technical intricacies and complex semantic information, requiring efficient fine-tuning for domain adaptation. Current methods neglect patents' hierarchical relationships, such as those defined by the Locarno International Classification (LIC) system, which groups broad categories (e.g., "furnishing") into subclasses (e.g., "seats" and "beds") and further into specific patent designs. In this work, we introduce a hierarchical multi-positive contrastive loss that leverages the LIC's taxonomy to induce such relations in the retrieval process. Our approach assigns multiple positive pairs to each patent image within a batch, with varying similarity scores based on the hierarchical taxonomy. Our experimental analysis with various vision and multimodal models on the DeepPatent2 dataset shows that the proposed method enhances the retrieval results. Notably, our method is effective with low-parameter models, which require fewer computational resources and can be deployed on environments with limited hardware.
>
---
#### [replaced 012] Data Augmentation Through Random Style Replacement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10563v2](http://arxiv.org/pdf/2504.10563v2)**

> **作者:** Qikai Yang; Cheng Ji; Huaiying Luo; Panfeng Li; Zhicheng Ding
>
> **备注:** Accepted by 2025 6th International Conference on Computer Vision, Image and Deep Learning
>
> **摘要:** In this paper, we introduce a novel data augmentation technique that combines the advantages of style augmentation and random erasing by selectively replacing image subregions with style-transferred patches. Our approach first applies a random style transfer to training images, then randomly substitutes selected areas of these images with patches derived from the style-transferred versions. This method is able to seamlessly accommodate a wide range of existing style transfer algorithms and can be readily integrated into diverse data augmentation pipelines. By incorporating our strategy, the training process becomes more robust and less prone to overfitting. Comparative experiments demonstrate that, relative to previous style augmentation methods, our technique achieves superior performance and faster convergence.
>
---
#### [replaced 013] Translation-Equivariance of Normalization Layers and Aliasing in Convolutional Neural Networks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19805v2](http://arxiv.org/pdf/2505.19805v2)**

> **作者:** Jérémy Scanvic; Quentin Barthélemy; Julián Tachella
>
> **备注:** Accepted at the Workshop on the Theory of AI for Scientific Computing (COLT 2025)
>
> **摘要:** The design of convolutional neural architectures that are exactly equivariant to continuous translations is an active field of research. It promises to benefit scientific computing, notably by making existing imaging systems more physically accurate. Most efforts focus on the design of downsampling/pooling layers, upsampling layers and activation functions, but little attention is dedicated to normalization layers. In this work, we present a novel theoretical framework for understanding the equivariance of normalization layers to discrete shifts and continuous translations. We also determine necessary and sufficient conditions for normalization layers to be equivariant in terms of the dimensions they operate on. Using real feature maps from ResNet-18 and ImageNet, we test those theoretical results empirically and find that they are consistent with our predictions.
>
---
#### [replaced 014] Rasterizing Wireless Radiance Field via Deformable 2D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12787v2](http://arxiv.org/pdf/2506.12787v2)**

> **作者:** Mufan Liu; Cixiao Zhang; Qi Yang; Yujie Cao; Yiling Xu; Yin Xu; Shu Sun; Mingzeng Dai; Yunfeng Guan
>
> **摘要:** Modeling the wireless radiance field (WRF) is fundamental to modern communication systems, enabling key tasks such as localization, sensing, and channel estimation. Traditional approaches, which rely on empirical formulas or physical simulations, often suffer from limited accuracy or require strong scene priors. Recent neural radiance field (NeRF-based) methods improve reconstruction fidelity through differentiable volumetric rendering, but their reliance on computationally expensive multilayer perceptron (MLP) queries hinders real-time deployment. To overcome these challenges, we introduce Gaussian splatting (GS) to the wireless domain, leveraging its efficiency in modeling optical radiance fields to enable compact and accurate WRF reconstruction. Specifically, we propose SwiftWRF, a deformable 2D Gaussian splatting framework that synthesizes WRF spectra at arbitrary positions under single-sided transceiver mobility. SwiftWRF employs CUDA-accelerated rasterization to render spectra at over 100000 fps and uses a lightweight MLP to model the deformation of 2D Gaussians, effectively capturing mobility-induced WRF variations. In addition to novel spectrum synthesis, the efficacy of SwiftWRF is further underscored in its applications in angle-of-arrival (AoA) and received signal strength indicator (RSSI) prediction. Experiments conducted on both real-world and synthetic indoor scenes demonstrate that SwiftWRF can reconstruct WRF spectra up to 500x faster than existing state-of-the-art methods, while significantly enhancing its signal quality. The project page is https://evan-sudo.github.io/swiftwrf/.
>
---
#### [replaced 015] DRL-Based Resource Allocation for Motion Blur Resistant Federated Self-Supervised Learning in IoV
- **分类: cs.CV; cs.LG; cs.NI**

- **链接: [http://arxiv.org/pdf/2408.09194v2](http://arxiv.org/pdf/2408.09194v2)**

> **作者:** Xueying Gu; Qiong Wu; Pingyi Fan; Qiang Fan; Nan Cheng; Wen Chen; Khaled B. Letaief
>
> **备注:** This paper has been accepted by IEEE Internet of Things Journal. The source code has been released at: https://github.com/qiongwu86/DRL-BFSSL
>
> **摘要:** In the Internet of Vehicles (IoV), Federated Learning (FL) provides a privacy-preserving solution by aggregating local models without sharing data. Traditional supervised learning requires image data with labels, but data labeling involves significant manual effort. Federated Self-Supervised Learning (FSSL) utilizes Self-Supervised Learning (SSL) for local training in FL, eliminating the need for labels while protecting privacy. Compared to other SSL methods, Momentum Contrast (MoCo) reduces the demand for computing resources and storage space by creating a dictionary. However, using MoCo in FSSL requires uploading the local dictionary from vehicles to Base Station (BS), which poses a risk of privacy leakage. Simplified Contrast (SimCo) addresses the privacy leakage issue in MoCo-based FSSL by using dual temperature instead of a dictionary to control sample distribution. Additionally, considering the negative impact of motion blur on model aggregation, and based on SimCo, we propose a motion blur-resistant FSSL method, referred to as BFSSL. Furthermore, we address energy consumption and delay in the BFSSL process by proposing a Deep Reinforcement Learning (DRL)-based resource allocation scheme, called DRL-BFSSL. In this scheme, BS allocates the Central Processing Unit (CPU) frequency and transmission power of vehicles to minimize energy consumption and latency, while aggregating received models based on the motion blur level. Simulation results validate the effectiveness of our proposed aggregation and resource allocation methods.
>
---
#### [replaced 016] Ophora: A Large-Scale Data-Driven Text-Guided Ophthalmic Surgical Video Generation Model
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07449v5](http://arxiv.org/pdf/2505.07449v5)**

> **作者:** Wei Li; Ming Hu; Guoan Wang; Lihao Liu; Kaijin Zhou; Junzhi Ning; Xin Guo; Zongyuan Ge; Lixu Gu; Junjun He
>
> **备注:** Early accepted in MICCAI25
>
> **摘要:** In ophthalmic surgery, developing an AI system capable of interpreting surgical videos and predicting subsequent operations requires numerous ophthalmic surgical videos with high-quality annotations, which are difficult to collect due to privacy concerns and labor consumption. Text-guided video generation (T2V) emerges as a promising solution to overcome this issue by generating ophthalmic surgical videos based on surgeon instructions. In this paper, we present Ophora, a pioneering model that can generate ophthalmic surgical videos following natural language instructions. To construct Ophora, we first propose a Comprehensive Data Curation pipeline to convert narrative ophthalmic surgical videos into a large-scale, high-quality dataset comprising over 160K video-instruction pairs, Ophora-160K. Then, we propose a Progressive Video-Instruction Tuning scheme to transfer rich spatial-temporal knowledge from a T2V model pre-trained on natural video-text datasets for privacy-preserved ophthalmic surgical video generation based on Ophora-160K. Experiments on video quality evaluation via quantitative analysis and ophthalmologist feedback demonstrate that Ophora can generate realistic and reliable ophthalmic surgical videos based on surgeon instructions. We also validate the capability of Ophora for empowering downstream tasks of ophthalmic surgical workflow understanding. Code is available at https://github.com/mar-cry/Ophora.
>
---
#### [replaced 017] The OCR Quest for Generalization: Learning to recognize low-resource alphabets with model editing
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06761v2](http://arxiv.org/pdf/2506.06761v2)**

> **作者:** Adrià Molina Rodríguez; Oriol Ramos Terrades; Josep Lladós
>
> **备注:** Preprint (under review) For Journal
>
> **摘要:** Achieving robustness in recognition systems across diverse domains is crucial for their practical utility. While ample data availability is usually assumed, low-resource languages, such as ancient manuscripts and non-western languages, tend to be kept out of the equations of massive pretraining and foundational techniques due to an under representation. In this work, we aim for building models which can generalize to new distributions of data, such as alphabets, faster than centralized fine-tune strategies. For doing so, we take advantage of the recent advancements in model editing to enhance the incorporation of unseen scripts (low-resource learning). In contrast to state-of-the-art meta-learning, we showcase the effectiveness of domain merging in sparse distributions of data, with agnosticity of its relation to the overall distribution or any other prototyping necessity. Even when using the same exact training data, our experiments showcase significant performance boosts in \textbf{transfer learning} to new alphabets and \textbf{out-of-domain evaluation} in challenging domain shifts, including historical ciphered texts and non-Latin scripts. This research contributes a novel approach into building models that can easily adopt under-represented alphabets and, therefore, enable document recognition to a wider set of contexts and cultures.
>
---
#### [replaced 018] Click-Calib: A Robust Extrinsic Calibration Method for Surround-View Systems
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.01557v3](http://arxiv.org/pdf/2501.01557v3)**

> **作者:** Lihao Wang
>
> **摘要:** Surround-View System (SVS) is an essential component in Advanced Driver Assistance System (ADAS) and requires precise calibrations. However, conventional offline extrinsic calibration methods are cumbersome and time-consuming as they rely heavily on physical patterns. Additionally, these methods primarily focus on short-range areas surrounding the vehicle, resulting in lower calibration quality in more distant zones. To address these limitations, we propose Click-Calib, a pattern-free approach for offline SVS extrinsic calibration. Without requiring any special setup, the user only needs to click a few keypoints on the ground in natural scenes. Unlike other offline calibration approaches, Click-Calib optimizes camera poses over a wide range by minimizing reprojection distance errors of keypoints, thereby achieving accurate calibrations at both short and long distances. Furthermore, Click-Calib supports both single-frame and multiple-frame modes, with the latter offering even better results. Evaluations on our in-house dataset and the public WoodScape dataset demonstrate its superior accuracy and robustness compared to baseline methods. Code is available at https://github.com/lwangvaleo/click_calib.
>
---
#### [replaced 019] Multiclass Post-Earthquake Building Assessment Integrating High-Resolution Optical and SAR Satellite Imagery, Ground Motion, and Soil Data with Transformers
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2412.04664v3](http://arxiv.org/pdf/2412.04664v3)**

> **作者:** Deepank Singh; Vedhus Hoskere; Pietro Milillo
>
> **备注:** 28 Pages, 12 Figures
>
> **摘要:** Timely and accurate assessments of building damage are crucial for effective response and recovery in the aftermath of earthquakes. Conventional preliminary damage assessments (PDA) often rely on manual door-to-door inspections, which are not only time-consuming but also pose significant safety risks. To safely expedite the PDA process, researchers have studied the applicability of satellite imagery processed with heuristic and machine learning approaches. These approaches output binary or, more recently, multiclass damage states at the scale of a block or a single building. However, the current performance of such approaches limits practical applicability. To address this limitation, we introduce a metadata-enriched, transformer based framework that combines high-resolution post-earthquake satellite imagery with building-specific metadata relevant to the seismic performance of the structure. Our model achieves state-of-the-art performance in multiclass post-earthquake damage identification for buildings from the Turkey-Syria earthquake on February 6, 2023. Specifically, we demonstrate that incorporating metadata, such as seismic intensity indicators, soil properties, and SAR damage proxy maps not only enhances the model's accuracy and ability to distinguish between damage classes, but also improves its generalizability across various regions. Furthermore, we conducted a detailed, class-wise analysis of feature importance to understand the model's decision-making across different levels of building damage. This analysis reveals how individual metadata features uniquely contribute to predictions for each damage class. By leveraging both satellite imagery and metadata, our proposed framework enables faster and more accurate damage assessments for precise, multiclass, building-level evaluations that can improve disaster response and accelerate recovery efforts for affected communities.
>
---
#### [replaced 020] Improving LLM Video Understanding with 16 Frames Per Second
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13956v2](http://arxiv.org/pdf/2503.13956v2)**

> **作者:** Yixuan Li; Changli Tang; Jimin Zhuang; Yudong Yang; Guangzhi Sun; Wei Li; Zejun Ma; Chao Zhang
>
> **摘要:** Human vision is dynamic and continuous. However, in video understanding with multimodal large language models (LLMs), existing methods primarily rely on static features extracted from images sampled at a fixed low frame rate of frame-per-second (FPS) $\leqslant$2, leading to critical visual information loss. In this paper, we introduce F-16, the first multimodal LLM designed for high-frame-rate video understanding. By increasing the frame rate to 16 FPS and compressing visual tokens within each 1-second clip, F-16 efficiently captures dynamic visual features while preserving key semantic information. Experimental results demonstrate that higher frame rates considerably enhance video understanding across multiple benchmarks, providing a new approach to improving video LLMs beyond scaling model size or training data. F-16 achieves state-of-the-art performance among 7-billion-parameter video LLMs on both general and fine-grained video understanding benchmarks, such as Video-MME and TemporalBench. Furthermore, F-16 excels in complex spatiotemporal tasks, including high-speed sports analysis (\textit{e.g.}, basketball, football, gymnastics, and diving), outperforming SOTA proprietary visual models like GPT-4o and Gemini-1.5-pro. Additionally, we introduce a novel decoding method for F-16 that enables highly efficient low-frame-rate inference without requiring model retraining. We will release the source code, model checkpoints, and data at \href{https://github.com/bytedance/F-16}{https://github.com/bytedance/F-16}.
>
---
#### [replaced 021] A Bird Song Detector for improving bird identification through Deep Learning: a case study from Doñana
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; cs.NE; I.5.4; I.2.6; I.4.8**

- **链接: [http://arxiv.org/pdf/2503.15576v2](http://arxiv.org/pdf/2503.15576v2)**

> **作者:** Alba Márquez-Rodríguez; Miguel Ángel Mohedano-Munoz; Manuel J. Marín-Jiménez; Eduardo Santamaría-García; Giulia Bastianelli; Pedro Jordano; Irene Mendoza
>
> **备注:** 23 pages, 14 images, for associated dataset see https://huggingface.co/datasets/GrunCrow/BIRDeep_AudioAnnotations , for associated code see https://github.com/GrunCrow/BIRDeep_BirdSongDetector_NeuralNetworks and https://github.com/GrunCrow/Bird-Song-Detector
>
> **摘要:** Passive Acoustic Monitoring is a key tool for biodiversity conservation, but the large volumes of unsupervised audio it generates present major challenges for extracting meaningful information. Deep Learning offers promising solutions. BirdNET, a widely used bird identification model, has shown success in many study systems but is limited at local scale due to biases in its training data, which focus on specific locations and target sounds rather than entire soundscapes. A key challenge in bird species identification is that many recordings either lack target species or contain overlapping vocalizations, complicating automatic identification. To address these problems, we developed a multi-stage pipeline for automatic bird vocalization identification in Do\~nana National Park (SW Spain), a wetland of high conservation concern. We deployed AudioMoth recorders in three main habitats across nine locations and manually annotated 461 minutes of audio, resulting in 3749 labeled segments spanning 34 classes. We first applied a Bird Song Detector to isolate bird vocalizations using spectrogram-based image processing. Then, species were classified using custom models trained at the local scale. Applying the Bird Song Detector before classification improved species identification, as all models performed better when analyzing only the segments where birds were detected. Specifically, the combination of detector and fine-tuned BirdNET outperformed the baseline without detection. This approach demonstrates the effectiveness of integrating a Bird Song Detector with local classification models. These findings highlight the need to adapt general-purpose tools to specific ecological challenges. Automatically detecting bird species helps track the health of this threatened ecosystem, given birds sensitivity to environmental change, and supports conservation planning to reduce biodiversity loss.
>
---
#### [replaced 022] Vision Transformers Don't Need Trained Registers
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.08010v3](http://arxiv.org/pdf/2506.08010v3)**

> **作者:** Nick Jiang; Amil Dravid; Alexei Efros; Yossi Gandelsman
>
> **备注:** Project page and code: https://avdravid.github.io/test-time-registers
>
> **摘要:** We investigate the mechanism underlying a previously identified phenomenon in Vision Transformers -- the emergence of high-norm tokens that lead to noisy attention maps. We observe that in multiple models (e.g., CLIP, DINOv2), a sparse set of neurons is responsible for concentrating high-norm activations on outlier tokens, leading to irregular attention patterns and degrading downstream visual processing. While the existing solution for removing these outliers involves retraining models from scratch with additional learned register tokens, we use our findings to create a training-free approach to mitigate these artifacts. By shifting the high-norm activations from our discovered register neurons into an additional untrained token, we can mimic the effect of register tokens on a model already trained without registers. We demonstrate that our method produces cleaner attention and feature maps, enhances performance over base models across multiple downstream visual tasks, and achieves results comparable to models explicitly trained with register tokens. We then extend test-time registers to off-the-shelf vision-language models to improve their interpretability. Our results suggest that test-time registers effectively take on the role of register tokens at test-time, offering a training-free solution for any pre-trained model released without them.
>
---
#### [replaced 023] Bi-VLDoc: Bidirectional Vision-Language Modeling for Visually-Rich Document Understanding
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2206.13155v2](http://arxiv.org/pdf/2206.13155v2)**

> **作者:** Chuwei Luo; Guozhi Tang; Qi Zheng; Cong Yao; Lianwen Jin; Chenliang Li; Yang Xue; Luo Si
>
> **备注:** IJDAR 2025
>
> **摘要:** Multi-modal document pre-trained models have proven to be very effective in a variety of visually-rich document understanding (VrDU) tasks. Though existing document pre-trained models have achieved excellent performance on standard benchmarks for VrDU, the way they model and exploit the interactions between vision and language on documents has hindered them from better generalization ability and higher accuracy. In this work, we investigate the problem of vision-language joint representation learning for VrDU mainly from the perspective of supervisory signals. Specifically, a pre-training paradigm called Bi-VLDoc is proposed, in which a bidirectional vision-language supervision strategy and a vision-language hybrid-attention mechanism are devised to fully explore and utilize the interactions between these two modalities, to learn stronger cross-modal document representations with richer semantics. Benefiting from the learned informative cross-modal document representations, Bi-VLDoc significantly advances the state-of-the-art performance on three widely-used document understanding benchmarks, including Form Understanding (from 85.14% to 93.44%), Receipt Information Extraction (from 96.01% to 97.84%), and Document Classification (from 96.08% to 97.12%). On Document Visual QA, Bi-VLDoc achieves the state-of-the-art performance compared to previous single model methods.
>
---
#### [replaced 024] RoCA: Robust Cross-Domain End-to-End Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10145v2](http://arxiv.org/pdf/2506.10145v2)**

> **作者:** Rajeev Yasarla; Shizhong Han; Hsin-Pai Cheng; Litian Liu; Shweta Mahajan; Apratim Bhattacharyya; Yunxiao Shi; Risheek Garrepalli; Hong Cai; Fatih Porikli
>
> **摘要:** End-to-end (E2E) autonomous driving has recently emerged as a new paradigm, offering significant potential. However, few studies have looked into the practical challenge of deployment across domains (e.g., cities). Although several works have incorporated Large Language Models (LLMs) to leverage their open-world knowledge, LLMs do not guarantee cross-domain driving performance and may incur prohibitive retraining costs during domain adaptation. In this paper, we propose RoCA, a novel framework for robust cross-domain E2E autonomous driving. RoCA formulates the joint probabilistic distribution over the tokens that encode ego and surrounding vehicle information in the E2E pipeline. Instantiating with a Gaussian process (GP), RoCA learns a set of basis tokens with corresponding trajectories, which span diverse driving scenarios. Then, given any driving scene, it is able to probabilistically infer the future trajectory. By using RoCA together with a base E2E model in source-domain training, we improve the generalizability of the base model, without requiring extra inference computation. In addition, RoCA enables robust adaptation on new target domains, significantly outperforming direct finetuning. We extensively evaluate RoCA on various cross-domain scenarios and show that it achieves strong domain generalization and adaptation performance.
>
---
#### [replaced 025] VideoMAR: Autoregressive Video Generatio with Continuous Tokens
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.14168v2](http://arxiv.org/pdf/2506.14168v2)**

> **作者:** Hu Yu; Biao Gong; Hangjie Yuan; DanDan Zheng; Weilong Chai; Jingdong Chen; Kecheng Zheng; Feng Zhao
>
> **摘要:** Masked-based autoregressive models have demonstrated promising image generation capability in continuous space. However, their potential for video generation remains under-explored. In this paper, we propose \textbf{VideoMAR}, a concise and efficient decoder-only autoregressive image-to-video model with continuous tokens, composing temporal frame-by-frame and spatial masked generation. We first identify temporal causality and spatial bi-directionality as the first principle of video AR models, and propose the next-frame diffusion loss for the integration of mask and video generation. Besides, the huge cost and difficulty of long sequence autoregressive modeling is a basic but crucial issue. To this end, we propose the temporal short-to-long curriculum learning and spatial progressive resolution training, and employ progressive temperature strategy at inference time to mitigate the accumulation error. Furthermore, VideoMAR replicates several unique capacities of language models to video generation. It inherently bears high efficiency due to simultaneous temporal-wise KV cache and spatial-wise parallel generation, and presents the capacity of spatial and temporal extrapolation via 3D rotary embeddings. On the VBench-I2V benchmark, VideoMAR surpasses the previous state-of-the-art (Cosmos I2V) while requiring significantly fewer parameters ($9.3\%$), training data ($0.5\%$), and GPU resources ($0.2\%$).
>
---
#### [replaced 026] Generative diffusion model surrogates for mechanistic agent-based biological models
- **分类: q-bio.QM; cs.CV; cs.ET; cs.PF**

- **链接: [http://arxiv.org/pdf/2505.09630v2](http://arxiv.org/pdf/2505.09630v2)**

> **作者:** Tien Comlekoglu; J. Quetzalcoatl Toledo-Marín; Douglas W. DeSimone; Shayn M. Peirce; Geoffrey Fox; James A. Glazier
>
> **摘要:** Mechanistic, multicellular, agent-based models are commonly used to investigate tissue, organ, and organism-scale biology at single-cell resolution. The Cellular-Potts Model (CPM) is a powerful and popular framework for developing and interrogating these models. CPMs become computationally expensive at large space- and time- scales making application and investigation of developed models difficult. Surrogate models may allow for the accelerated evaluation of CPMs of complex biological systems. However, the stochastic nature of these models means each set of parameters may give rise to different model configurations, complicating surrogate model development. In this work, we leverage denoising diffusion probabilistic models to train a generative AI surrogate of a CPM used to investigate in vitro vasculogenesis. We describe the use of an image classifier to learn the characteristics that define unique areas of a 2-dimensional parameter space. We then apply this classifier to aid in surrogate model selection and verification. Our CPM model surrogate generates model configurations 20,000 timesteps ahead of a reference configuration and demonstrates approximately a 22x reduction in computational time as compared to native code execution. Our work represents a step towards the implementation of DDPMs to develop digital twins of stochastic biological systems.
>
---
#### [replaced 027] Pro-AD: Learning Comprehensive Prototypes with Prototype-based Constraint for Multi-class Unsupervised Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13097v2](http://arxiv.org/pdf/2506.13097v2)**

> **作者:** Ziqing Zhou; Bin-Bin Gao; Yurui Pan; Lidong Wang; Wenbing Zhu; Yong Liu; Jun Liu; Mingmin Chi; Dong Wu; Bo Peng; Chengjie Wang
>
> **摘要:** Prototype-based reconstruction methods for unsupervised anomaly detection utilize a limited set of learnable prototypes which only aggregates insufficient normal information, resulting in undesirable reconstruction. However, increasing the number of prototypes may lead to anomalies being well reconstructed through the attention mechanism, which we refer to as the "Soft Identity Mapping" problem. In this paper, we propose Pro-AD to address these issues and fully utilize the prototypes to boost the performance of anomaly detection. Specifically, we first introduce an expanded set of learnable prototypes to provide sufficient capacity for semantic information. Then we employ a Dynamic Bidirectional Decoder which integrates the process of the normal information aggregation and the target feature reconstruction via prototypes, with the aim of allowing the prototypes to aggregate more comprehensive normal semantic information from different levels of the image features and the target feature reconstruction to not only utilize its contextual information but also dynamically leverage the learned comprehensive prototypes. Additionally, to prevent the anomalies from being well reconstructed using sufficient semantic information through the attention mechanism, Pro-AD introduces a Prototype-based Constraint that applied within the target feature reconstruction process of the decoder, which further improves the performance of our approach. Extensive experiments on multiple challenging benchmarks demonstrate that our Pro-AD achieve state-of-the-art performance, highlighting its superior robustness and practical effectiveness for Multi-class Unsupervised Anomaly Detection task.
>
---
#### [replaced 028] LaViDa: A Large Diffusion Language Model for Multimodal Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16839v3](http://arxiv.org/pdf/2505.16839v3)**

> **作者:** Shufan Li; Konstantinos Kallidromitis; Hritik Bansal; Akash Gokul; Yusuke Kato; Kazuki Kozuka; Jason Kuen; Zhe Lin; Kai-Wei Chang; Aditya Grover
>
> **备注:** 26 pages, 8 figures
>
> **摘要:** Modern Vision-Language Models (VLMs) can solve a wide range of tasks requiring visual reasoning. In real-world scenarios, desirable properties for VLMs include fast inference and controllable generation (e.g., constraining outputs to adhere to a desired format). However, existing autoregressive (AR) VLMs like LLaVA struggle in these aspects. Discrete diffusion models (DMs) offer a promising alternative, enabling parallel decoding for faster inference and bidirectional context for controllable generation through text-infilling. While effective in language-only settings, DMs' potential for multimodal tasks is underexplored. We introduce LaViDa, a family of VLMs built on DMs. We build LaViDa by equipping DMs with a vision encoder and jointly fine-tune the combined parts for multimodal instruction following. To address challenges encountered, LaViDa incorporates novel techniques such as complementary masking for effective training, prefix KV cache for efficient inference, and timestep shifting for high-quality sampling. Experiments show that LaViDa achieves competitive or superior performance to AR VLMs on multi-modal benchmarks such as MMMU, while offering unique advantages of DMs, including flexible speed-quality tradeoff, controllability, and bidirectional reasoning. On COCO captioning, LaViDa surpasses Open-LLaVa-Next-8B by +4.1 CIDEr with 1.92x speedup. On bidirectional tasks, it achieves +59% improvement on Constrained Poem Completion. These results demonstrate LaViDa as a strong alternative to AR VLMs. Code and models will be released in the camera-ready version.
>
---
#### [replaced 029] Think Twice before Adaptation: Improving Adaptability of DeepFake Detection via Online Test-Time Adaptation
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2505.18787v2](http://arxiv.org/pdf/2505.18787v2)**

> **作者:** Hong-Hanh Nguyen-Le; Van-Tuan Tran; Dinh-Thuc Nguyen; Nhien-An Le-Khac
>
> **备注:** Accepted at 34th International Joint Conference on Artificial Intelligence (IJCAI-25)
>
> **摘要:** Deepfake (DF) detectors face significant challenges when deployed in real-world environments, particularly when encountering test samples deviated from training data through either postprocessing manipulations or distribution shifts. We demonstrate postprocessing techniques can completely obscure generation artifacts presented in DF samples, leading to performance degradation of DF detectors. To address these challenges, we propose Think Twice before Adaptation (\texttt{T$^2$A}), a novel online test-time adaptation method that enhances the adaptability of detectors during inference without requiring access to source training data or labels. Our key idea is to enable the model to explore alternative options through an Uncertainty-aware Negative Learning objective rather than solely relying on its initial predictions as commonly seen in entropy minimization (EM)-based approaches. We also introduce an Uncertain Sample Prioritization strategy and Gradients Masking technique to improve the adaptation by focusing on important samples and model parameters. Our theoretical analysis demonstrates that the proposed negative learning objective exhibits complementary behavior to EM, facilitating better adaptation capability. Empirically, our method achieves state-of-the-art results compared to existing test-time adaptation (TTA) approaches and significantly enhances the resilience and generalization of DF detectors during inference. Code is available \href{https://github.com/HongHanh2104/T2A-Think-Twice-Before-Adaptation}{here}.
>
---
#### [replaced 030] TARDIS STRIDE: A Spatio-Temporal Road Image Dataset and World Model for Autonomy
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.11302v2](http://arxiv.org/pdf/2506.11302v2)**

> **作者:** Héctor Carrión; Yutong Bai; Víctor A. Hernández Castro; Kishan Panaganti; Ayush Zenith; Matthew Trang; Tony Zhang; Pietro Perona; Jitendra Malik
>
> **备注:** Computer Vision, Pattern Recognition, Early-Fusion, Dataset, Data Augmentation
>
> **摘要:** World models aim to simulate environments and enable effective agent behavior. However, modeling real-world environments presents unique challenges as they dynamically change across both space and, crucially, time. To capture these composed dynamics, we introduce a Spatio-Temporal Road Image Dataset for Exploration (STRIDE) permuting 360-degree panoramic imagery into rich interconnected observation, state and action nodes. Leveraging this structure, we can simultaneously model the relationship between egocentric views, positional coordinates, and movement commands across both space and time. We benchmark this dataset via TARDIS, a transformer-based generative world model that integrates spatial and temporal dynamics through a unified autoregressive framework trained on STRIDE. We demonstrate robust performance across a range of agentic tasks such as controllable photorealistic image synthesis, instruction following, autonomous self-control, and state-of-the-art georeferencing. These results suggest a promising direction towards sophisticated generalist agents--capable of understanding and manipulating the spatial and temporal aspects of their material environments--with enhanced embodied reasoning capabilities. Training code, datasets, and model checkpoints are made available at https://huggingface.co/datasets/Tera-AI/STRIDE.
>
---
#### [replaced 031] A Curated and Re-annotated Peripheral Blood Cell Dataset Integrating Four Public Resources
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.13214v2](http://arxiv.org/pdf/2407.13214v2)**

> **作者:** Lu Gan; Xi Li; Xichun Wang
>
> **摘要:** We present TXL-PBC, a curated and re-annotated peripheral blood cell dataset constructed by integrating four publicly available resources: Blood Cell Count and Detection (BCCD), Blood Cell Detection Dataset (BCDD), Peripheral Blood Cells (PBC), and Raabin White Blood Cell (Raabin-WBC). Through rigorous sample selection, semi-automatic annotation using the YOLOv8n model, and comprehensive manual review, we ensured high annotation accuracy and consistency. The final dataset contains 1,260 images and 18,143 bounding box annotations for three major blood cell types: white blood cells (WBC), red blood cells (RBC), and platelets. We provide detailed visual analyses of the data distribution, demonstrating the diversity and balance of the dataset. To further validate the quality and utility of TXL-PBC, we trained several mainstream object detection models, including YOLOv5s, YOLOv8s, YOLOv11s, SSD300, Faster R-CNN, and RetinaNet, and report their baseline performance. The TXL-PBC dataset is openly available on Figshare and GitHub, offering a valuable resource for the development and benchmarking of blood cell detection models and related machine learning research.
>
---
#### [replaced 032] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.00473v5](http://arxiv.org/pdf/2412.00473v5)**

> **作者:** Yu Wang; Xiaofei Zhou; Yichen Wang; Geyuan Zhang; Tianxing He
>
> **摘要:** With the significant advancement of Large Vision-Language Models (VLMs), concerns about their potential misuse and abuse have grown rapidly. Previous studies have highlighted VLMs' vulnerability to jailbreak attacks, where carefully crafted inputs can lead the model to produce content that violates ethical and legal standards. However, existing methods struggle against state-of-the-art VLMs like GPT-4o, due to the over-exposure of harmful content and lack of stealthy malicious guidance. In this work, we propose a novel jailbreak attack framework: Multi-Modal Linkage (MML) Attack. Drawing inspiration from cryptography, MML utilizes an encryption-decryption process across text and image modalities to mitigate over-exposure of malicious information. To align the model's output with malicious intent covertly, MML employs a technique called "evil alignment", framing the attack within a video game production scenario. Comprehensive experiments demonstrate MML's effectiveness. Specifically, MML jailbreaks GPT-4o with attack success rates of 97.80% on SafeBench, 98.81% on MM-SafeBench and 99.07% on HADES-Dataset. Our code is available at https://github.com/wangyu-ovo/MML.
>
---
#### [replaced 033] I2I-Mamba: Multi-modal medical image synthesis via selective state space modeling
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.14022v5](http://arxiv.org/pdf/2405.14022v5)**

> **作者:** Omer F. Atli; Bilal Kabas; Fuat Arslan; Arda C. Demirtas; Mahmut Yurt; Onat Dalmaz; Tolga Çukur
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Multi-modal medical image synthesis involves nonlinear transformation of tissue signals between source and target modalities, where tissues exhibit contextual interactions across diverse spatial distances. As such, the utility of a network architecture in synthesis depends on its ability to express these contextual features. Convolutional neural networks (CNNs) offer high local precision at the expense of poor sensitivity to long-range context. While transformers promise to alleviate this issue, they suffer from an unfavorable trade-off between sensitivity to long- versus short-range context due to the intrinsic complexity of attention filters. To effectively capture contextual features while avoiding the complexity-driven trade-offs, here we introduce a novel multi-modal synthesis method, I2I-Mamba, based on the state space modeling (SSM) framework. Focusing on semantic representations across a hybrid residual architecture, I2I-Mamba leverages novel dual-domain Mamba (ddMamba) blocks for complementary contextual modeling in image and Fourier domains, while maintaining spatial precision with convolutional layers. Diverting from conventional raster-scan trajectories, ddMamba leverages novel SSM operators based on a spiral-scan trajectory to learn context with enhanced radial coverage and angular isotropy, and a channel-mixing layer to aggregate context across the channel dimension. Comprehensive demonstrations on multi-contrast MRI and MRI-CT protocols indicate that I2I-Mamba offers superior performance against state-of-the-art CNNs, transformers and SSMs.
>
---
#### [replaced 034] Unsourced Adversarial CAPTCHA: A Bi-Phase Adversarial CAPTCHA Framework
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2506.10685v2](http://arxiv.org/pdf/2506.10685v2)**

> **作者:** Xia Du; Xiaoyuan Liu; Jizhe Zhou; Zheng Lin; Chi-man Pun; Cong Wu; Tao Li; Zhe Chen; Wei Ni; Jun Luo
>
> **摘要:** With the rapid advancements in deep learning, traditional CAPTCHA schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on original image characteristics, resulting in distortions that hinder human interpretation and limit applicability in scenarios lacking initial input images. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (UAC), a novel framework generating high-fidelity adversarial examples guided by attacker-specified text prompts. Leveraging a Large Language Model (LLM), UAC enhances CAPTCHA diversity and supports both targeted and untargeted attacks. For targeted attacks, the EDICT method optimizes dual latent variables in a diffusion model for superior image quality. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-UAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show BP-UAC achieves high attack success rates across diverse systems, generating natural CAPTCHAs indistinguishable to humans and DNNs.
>
---
#### [replaced 035] FLARE: Towards Universal Dataset Purification against Backdoor Attacks
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19479v2](http://arxiv.org/pdf/2411.19479v2)**

> **作者:** Linshan Hou; Wei Luo; Zhongyun Hua; Songhua Chen; Leo Yu Zhang; Yiming Li
>
> **备注:** 15 pages, This paper is accepted and will appear in TIFS (CCF-A)
>
> **摘要:** Deep neural networks (DNNs) are susceptible to backdoor attacks, where adversaries poison datasets with adversary-specified triggers to implant hidden backdoors, enabling malicious manipulation of model predictions. Dataset purification serves as a proactive defense by removing malicious training samples to prevent backdoor injection at its source. We first reveal that the current advanced purification methods rely on a latent assumption that the backdoor connections between triggers and target labels in backdoor attacks are simpler to learn than the benign features. We demonstrate that this assumption, however, does not always hold, especially in all-to-all (A2A) and untargeted (UT) attacks. As a result, purification methods that analyze the separation between the poisoned and benign samples in the input-output space or the final hidden layer space are less effective. We observe that this separability is not confined to a single layer but varies across different hidden layers. Motivated by this understanding, we propose FLARE, a universal purification method to counter various backdoor attacks. FLARE aggregates abnormal activations from all hidden layers to construct representations for clustering. To enhance separation, FLARE develops an adaptive subspace selection algorithm to isolate the optimal space for dividing an entire dataset into two clusters. FLARE assesses the stability of each cluster and identifies the cluster with higher stability as poisoned. Extensive evaluations on benchmark datasets demonstrate the effectiveness of FLARE against 22 representative backdoor attacks, including all-to-one (A2O), all-to-all (A2A), and untargeted (UT) attacks, and its robustness to adaptive attacks. Codes are available at \href{https://github.com/THUYimingLi/BackdoorBox}{BackdoorBox} and \href{https://github.com/vtu81/backdoor-toolbox}{backdoor-toolbox}.
>
---
#### [replaced 036] Towards Cross-Subject EMG Pattern Recognition via Dual-Branch Adversarial Feature Disentanglement
- **分类: cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.08555v2](http://arxiv.org/pdf/2506.08555v2)**

> **作者:** Xinyue Niu; Akira Furui
>
> **备注:** 6 pages, 3 figures. This work has been accepted for presentation at the IEEE Engineering in Medicine and Biology Conference (EMBC) 2025. New version corrects numerical errors in Table 1. Conclusions are unaffected
>
> **摘要:** Cross-subject electromyography (EMG) pattern recognition faces significant challenges due to inter-subject variability in muscle anatomy, electrode placement, and signal characteristics. Traditional methods rely on subject-specific calibration data to adapt models to new users, an approach that is both time-consuming and impractical for large-scale, real-world deployment. This paper presents an approach to eliminate calibration requirements through feature disentanglement, enabling effective cross-subject generalization. We propose an end-to-end dual-branch adversarial neural network that simultaneously performs pattern recognition and individual identification by disentangling EMG features into pattern-specific and subject-specific components. The pattern-specific components facilitate robust pattern recognition for new users without model calibration, while the subject-specific components enable downstream applications such as task-invariant biometric identification. Experimental results demonstrate that the proposed model achieves robust performance on data from unseen users, outperforming various baseline methods in cross-subject scenarios. Overall, this study offers a new perspective for cross-subject EMG pattern recognition without model calibration and highlights the proposed model's potential for broader applications, such as task-independent biometric systems.
>
---
#### [replaced 037] SemVink: Advancing VLMs' Semantic Understanding of Optical Illusions via Visual Global Thinking
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02803v2](http://arxiv.org/pdf/2506.02803v2)**

> **作者:** Sifan Li; Yujun Cai; Yiwei Wang
>
> **摘要:** Vision-language models (VLMs) excel in semantic tasks but falter at a core human capability: detecting hidden content in optical illusions or AI-generated images through perceptual adjustments like zooming. We introduce HC-Bench, a benchmark of 112 images with hidden text, objects, and illusions, revealing that leading VLMs achieve near-zero accuracy (0-5.36%)-even with explicit prompting. Humans resolve such ambiguities instinctively, yet VLMs fail due to an overreliance on high-level semantics. Strikingly, we propose SemVink (Semantic Visual Thinking) by simply scaling images to low resolutions (32-128 pixels), which unlocks >99% accuracy by eliminating redundant visual noise. This exposes a critical architectural flaw: VLMs prioritize abstract reasoning over low-level visual operations crucial for real-world robustness. Our work urges a shift toward hybrid models integrating multi-scale processing, bridging the gap between computational vision and human cognition for applications in medical imaging, security, and beyond.
>
---
#### [replaced 038] MOS: Model Surgery for Pre-Trained Model-Based Class-Incremental Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09441v2](http://arxiv.org/pdf/2412.09441v2)**

> **作者:** Hai-Long Sun; Da-Wei Zhou; Hanbin Zhao; Le Gan; De-Chuan Zhan; Han-Jia Ye
>
> **备注:** Accepted to AAAI 2025. Code is available at: https://github.com/sun-hailong/AAAI25-MOS
>
> **摘要:** Class-Incremental Learning (CIL) requires models to continually acquire knowledge of new classes without forgetting old ones. Despite Pre-trained Models (PTMs) have shown excellent performance in CIL, catastrophic forgetting still occurs as the model learns new concepts. Existing work seeks to utilize lightweight components to adjust the PTM, while the forgetting phenomenon still comes from {\em parameter and retrieval} levels. Specifically, iterative updates of the model result in parameter drift, while mistakenly retrieving irrelevant modules leads to the mismatch during inference. To this end, we propose MOdel Surgery (MOS) to rescue the model from forgetting previous knowledge. By training task-specific adapters, we continually adjust the PTM to downstream tasks. To mitigate parameter-level forgetting, we present an adapter merging approach to learn task-specific adapters, which aims to bridge the gap between different components while reserve task-specific information. Besides, to address retrieval-level forgetting, we introduce a training-free self-refined adapter retrieval mechanism during inference, which leverages the model's inherent ability for better adapter retrieval. By jointly rectifying the model with those steps, MOS can robustly resist catastrophic forgetting in the learning process. Extensive experiments on seven benchmark datasets validate MOS's state-of-the-art performance. Code is available at: https://github.com/sun-hailong/AAAI25-MOS
>
---
#### [replaced 039] RDD: Robust Feature Detector and Descriptor using Deformable Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08013v2](http://arxiv.org/pdf/2505.08013v2)**

> **作者:** Gonglin Chen; Tianwen Fu; Haiwei Chen; Wenbin Teng; Hanyuan Xiao; Yajie Zhao
>
> **摘要:** As a core step in structure-from-motion and SLAM, robust feature detection and description under challenging scenarios such as significant viewpoint changes remain unresolved despite their ubiquity. While recent works have identified the importance of local features in modeling geometric transformations, these methods fail to learn the visual cues present in long-range relationships. We present Robust Deformable Detector (RDD), a novel and robust keypoint detector/descriptor leveraging the deformable transformer, which captures global context and geometric invariance through deformable self-attention mechanisms. Specifically, we observed that deformable attention focuses on key locations, effectively reducing the search space complexity and modeling the geometric invariance. Furthermore, we collected an Air-to-Ground dataset for training in addition to the standard MegaDepth dataset. Our proposed method outperforms all state-of-the-art keypoint detection/description methods in sparse matching tasks and is also capable of semi-dense matching. To ensure comprehensive evaluation, we introduce two challenging benchmarks: one emphasizing large viewpoint and scale variations, and the other being an Air-to-Ground benchmark -- an evaluation setting that has recently gaining popularity for 3D reconstruction across different altitudes.
>
---
#### [replaced 040] PanopticNeRF-360: Panoramic 3D-to-2D Label Transfer in Urban Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2309.10815v4](http://arxiv.org/pdf/2309.10815v4)**

> **作者:** Xiao Fu; Shangzhan Zhang; Tianrun Chen; Yichong Lu; Xiaowei Zhou; Andreas Geiger; Yiyi Liao
>
> **备注:** TPAMI 2025. Project page: http://fuxiao0719.github.io/projects/panopticnerf360/ Code: https://github.com/fuxiao0719/PanopticNeRF/tree/panopticnerf360
>
> **摘要:** Training perception systems for self-driving cars requires substantial 2D annotations that are labor-intensive to manual label. While existing datasets provide rich annotations on pre-recorded sequences, they fall short in labeling rarely encountered viewpoints, potentially hampering the generalization ability for perception models. In this paper, we present PanopticNeRF-360, a novel approach that combines coarse 3D annotations with noisy 2D semantic cues to generate high-quality panoptic labels and images from any viewpoint. Our key insight lies in exploiting the complementarity of 3D and 2D priors to mutually enhance geometry and semantics. Specifically, we propose to leverage coarse 3D bounding primitives and noisy 2D semantic and instance predictions to guide geometry optimization, by encouraging predicted labels to match panoptic pseudo ground truth. Simultaneously, the improved geometry assists in filtering 3D&2D annotation noise by fusing semantics in 3D space via a learned semantic field. To further enhance appearance, we combine MLP and hash grids to yield hybrid scene features, striking a balance between high-frequency appearance and contiguous semantics. Our experiments demonstrate PanopticNeRF-360's state-of-the-art performance over label transfer methods on the challenging urban scenes of the KITTI-360 dataset. Moreover, PanopticNeRF-360 enables omnidirectional rendering of high-fidelity, multi-view and spatiotemporally consistent appearance, semantic and instance labels. We make our code and data available at https://github.com/fuxiao0719/PanopticNeRF
>
---
#### [replaced 041] SurgSora: Object-Aware Diffusion Model for Controllable Surgical Video Generation
- **分类: cs.CV; cs.AI; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.14018v2](http://arxiv.org/pdf/2412.14018v2)**

> **作者:** Tong Chen; Shuya Yang; Junyi Wang; Long Bai; Hongliang Ren; Luping Zhou
>
> **摘要:** Surgical video generation can enhance medical education and research, but existing methods lack fine-grained motion control and realism. We introduce SurgSora, a framework that generates high-fidelity, motion-controllable surgical videos from a single input frame and user-specified motion cues. Unlike prior approaches that treat objects indiscriminately or rely on ground-truth segmentation masks, SurgSora leverages self-predicted object features and depth information to refine RGB appearance and optical flow for precise video synthesis. It consists of three key modules: (1) the Dual Semantic Injector, which extracts object-specific RGB-D features and segmentation cues to enhance spatial representations; (2) the Decoupled Flow Mapper, which fuses multi-scale optical flow with semantic features for realistic motion dynamics; and (3) the Trajectory Controller, which estimates sparse optical flow and enables user-guided object movement. By conditioning these enriched features within the Stable Video Diffusion, SurgSora achieves state-of-the-art visual authenticity and controllability in advancing surgical video synthesis, as demonstrated by extensive quantitative and qualitative comparisons. Our human evaluation in collaboration with expert surgeons further demonstrates the high realism of SurgSora-generated videos, highlighting the potential of our method for surgical training and education. Our project is available at https://surgsora.github.io/surgsora.github.io.
>
---
#### [replaced 042] VideoHallu: Evaluating and Mitigating Multi-modal Hallucinations on Synthetic Video Understanding
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.01481v3](http://arxiv.org/pdf/2505.01481v3)**

> **作者:** Zongxia Li; Xiyang Wu; Guangyao Shi; Yubin Qin; Hongyang Du; Tianyi Zhou; Dinesh Manocha; Jordan Lee Boyd-Graber
>
> **摘要:** Synthetic video generation has gained significant attention for its realism and broad applications, but remains prone to violations of common sense and physical laws. This highlights the need for reliable abnormality detectors that understand such principles and are robust to hallucinations. To address this, we introduce VideoHallu, a benchmark of over 3,000 video QA pairs built from synthetic videos generated by models like Veo2, Sora, and Kling, paired with expert-crafted counterintuitive QA to evaluate the critical thinking abilities of Multi-modal Large Language Models (MLLMs) on abnormalities that are perceptually obvious to humans but often hallucinated due to language priors. VideoHallu evaluates MLLMs' abnormality detection abilities with examples across alignment, consistency, commonsense, and physics. We benchmark SOTA MLLMs, including GPT-4o, Gemini-2.5-Pro, Qwen2.5-VL, Video-R1, and VideoChat-R1. We observe that these models perform well on many real-world benchmarks like MVBench and MovieChat, but still struggle with basic physics-based and commonsense reasoning in synthetic videos. We further show that post-training with Group Relative Policy Optimization (GRPO), using curriculum learning on datasets combining video QA with counterintuitive commonsense and physics reasoning over real and synthetic videos, improves MLLMs' abnormality detection and critical thinking, demonstrating the value of targeted training for improving their understanding of commonsense and physical laws. Our code is available at https://github.com/zli12321/VideoHallu.git.
>
---
#### [replaced 043] RefChartQA: Grounding Visual Answer on Chart Images through Instruction Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23131v2](http://arxiv.org/pdf/2503.23131v2)**

> **作者:** Alexander Vogel; Omar Moured; Yufan Chen; Jiaming Zhang; Rainer Stiefelhagen
>
> **备注:** Accepted by ICDAR 2025. All models and code will be publicly available at https://github.com/moured/RefChartQA
>
> **摘要:** Recently, Vision Language Models (VLMs) have increasingly emphasized document visual grounding to achieve better human-computer interaction, accessibility, and detailed understanding. However, its application to visualizations such as charts remains under-explored due to the inherent complexity of interleaved visual-numerical relationships in chart images. Existing chart understanding methods primarily focus on answering questions without explicitly identifying the visual elements that support their predictions. To bridge this gap, we introduce RefChartQA, a novel benchmark that integrates Chart Question Answering (ChartQA) with visual grounding, enabling models to refer elements at multiple granularities within chart images. Furthermore, we conduct a comprehensive evaluation by instruction-tuning 5 state-of-the-art VLMs across different categories. Our experiments demonstrate that incorporating spatial awareness via grounding improves response accuracy by over 15%, reducing hallucinations, and improving model reliability. Additionally, we identify key factors influencing text-spatial alignment, such as architectural improvements in TinyChart, which leverages a token-merging module for enhanced feature fusion. Our dataset is open-sourced for community development and further advancements. All models and code will be publicly available at https://github.com/moured/RefChartQA.
>
---
#### [replaced 044] Semantic Mapping in Indoor Embodied AI -- A Survey on Advances, Challenges, and Future Directions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.05750v2](http://arxiv.org/pdf/2501.05750v2)**

> **作者:** Sonia Raychaudhuri; Angel X. Chang
>
> **摘要:** Intelligent embodied agents (e.g. robots) need to perform complex semantic tasks in unfamiliar environments. Among many skills that the agents need to possess, building and maintaining a semantic map of the environment is most crucial in long-horizon tasks. A semantic map captures information about the environment in a structured way, allowing the agent to reference it for advanced reasoning throughout the task. While existing surveys in embodied AI focus on general advancements or specific tasks like navigation and manipulation, this paper provides a comprehensive review of semantic map-building approaches in embodied AI, specifically for indoor navigation. We categorize these approaches based on their structural representation (spatial grids, topological graphs, dense point-clouds or hybrid maps) and the type of information they encode (implicit features or explicit environmental data). We also explore the strengths and limitations of the map building techniques, highlight current challenges, and propose future research directions. We identify that the field is moving towards developing open-vocabulary, queryable, task-agnostic map representations, while high memory demands and computational inefficiency still remaining to be open challenges. This survey aims to guide current and future researchers in advancing semantic mapping techniques for embodied AI systems.
>
---
#### [replaced 045] CooPre: Cooperative Pretraining for V2X Cooperative Perception
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.11241v2](http://arxiv.org/pdf/2408.11241v2)**

> **作者:** Seth Z. Zhao; Hao Xiang; Chenfeng Xu; Xin Xia; Bolei Zhou; Jiaqi Ma
>
> **摘要:** Existing Vehicle-to-Everything (V2X) cooperative perception methods rely on accurate multi-agent 3D annotations. Nevertheless, it is time-consuming and expensive to collect and annotate real-world data, especially for V2X systems. In this paper, we present a self-supervised learning framwork for V2X cooperative perception, which utilizes the vast amount of unlabeled 3D V2X data to enhance the perception performance. Specifically, multi-agent sensing information is aggregated to form a holistic view and a novel proxy task is formulated to reconstruct the LiDAR point clouds across multiple connected agents to better reason multi-agent spatial correlations. Besides, we develop a V2X bird-eye-view (BEV) guided masking strategy which effectively allows the model to pay attention to 3D features across heterogeneous V2X agents (i.e., vehicles and infrastructure) in the BEV space. Noticeably, such a masking strategy effectively pretrains the 3D encoder with a multi-agent LiDAR point cloud reconstruction objective and is compatible with mainstream cooperative perception backbones. Our approach, validated through extensive experiments on representative datasets (i.e., V2X-Real, V2V4Real, and OPV2V) and multiple state-of-the-art cooperative perception methods (i.e., AttFuse, F-Cooper, and V2X-ViT), leads to a performance boost across all V2X settings. Notably, CooPre achieves a 4% mAP improvement on V2X-Real dataset and surpasses baseline performance using only 50% of the training data, highlighting its data efficiency. Additionally, we demonstrate the framework's powerful performance in cross-domain transferability and robustness under challenging scenarios. The code will be made publicly available at https://github.com/ucla-mobility/CooPre.
>
---
#### [replaced 046] SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04893v5](http://arxiv.org/pdf/2504.04893v5)**

> **作者:** Justus Westerhoff; Erblina Purelku; Jakob Hackstein; Jonas Loos; Leo Pinetzki; Lorenz Hufe
>
> **备注:** Accepted at CVPR 2025 Workshop EVAL-FoMo-2
>
> **摘要:** Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper along with the code for evaluations at www.bliss.berlin/research/scam.
>
---
#### [replaced 047] Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09881v2](http://arxiv.org/pdf/2506.09881v2)**

> **作者:** Siyu Chen; Ting Han; Chengzheng Fu; Changshe Zhang; Chaolei Wang; Jinhe Su; Guorong Cai; Meiliu Wu
>
> **摘要:** Open-Vocabulary semantic segmentation (OVSS) and domain generalization in semantic segmentation (DGSS) highlight a subtle complementarity that motivates Open-Vocabulary Domain-Generalized Semantic Segmentation (OV-DGSS). OV-DGSS aims to generate pixel-level masks for unseen categories while maintaining robustness across unseen domains, a critical capability for real-world scenarios such as autonomous driving in adverse conditions. We introduce Vireo, a novel single-stage framework for OV-DGSS that unifies the strengths of OVSS and DGSS for the first time. Vireo builds upon the frozen Visual Foundation Models (VFMs) and incorporates scene geometry via Depth VFMs to extract domain-invariant structural features. To bridge the gap between visual and textual modalities under domain shift, we propose three key components: (1) GeoText Prompts, which align geometric features with language cues and progressively refine VFM encoder representations; (2) Coarse Mask Prior Embedding (CMPE) for enhancing gradient flow for faster convergence and stronger textual influence; and (3) the Domain-Open-Vocabulary Vector Embedding Head (DOV-VEH), which fuses refined structural and semantic features for robust prediction. Comprehensive evaluation on these components demonstrates the effectiveness of our designs. Our proposed Vireo achieves the state-of-the-art performance and surpasses existing methods by a large margin in both domain generalization and open-vocabulary recognition, offering a unified and scalable solution for robust visual understanding in diverse and dynamic environments. Code is available at https://github.com/anonymouse-9c53tp182bvz/Vireo.
>
---
#### [replaced 048] Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09042v3](http://arxiv.org/pdf/2506.09042v3)**

> **作者:** Xuanchi Ren; Yifan Lu; Tianshi Cao; Ruiyuan Gao; Shengyu Huang; Amirmojtaba Sabour; Tianchang Shen; Tobias Pfaff; Jay Zhangjie Wu; Runjian Chen; Seung Wook Kim; Jun Gao; Laura Leal-Taixe; Mike Chen; Sanja Fidler; Huan Ling
>
> **备注:** Only the core contributors are listed. The full list of contributors can be found in Appendix A of this paper
>
> **摘要:** Collecting and annotating real-world data for safety-critical physical AI systems, such as Autonomous Vehicle (AV), is time-consuming and costly. It is especially challenging to capture rare edge cases, which play a critical role in training and testing of an AV system. To address this challenge, we introduce the Cosmos-Drive-Dreams - a synthetic data generation (SDG) pipeline that aims to generate challenging scenarios to facilitate downstream tasks such as perception and driving policy training. Powering this pipeline is Cosmos-Drive, a suite of models specialized from NVIDIA Cosmos world foundation model for the driving domain and are capable of controllable, high-fidelity, multi-view, and spatiotemporally consistent driving video generation. We showcase the utility of these models by applying Cosmos-Drive-Dreams to scale the quantity and diversity of driving datasets with high-fidelity and challenging scenarios. Experimentally, we demonstrate that our generated data helps in mitigating long-tail distribution problems and enhances generalization in downstream tasks such as 3D lane detection, 3D object detection and driving policy learning. We open source our pipeline toolkit, dataset and model weights through the NVIDIA's Cosmos platform. Project page: https://research.nvidia.com/labs/toronto-ai/cosmos_drive_dreams
>
---
#### [replaced 049] MSVIT: Improving Spiking Vision Transformer Using Multi-scale Attention Fusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14719v3](http://arxiv.org/pdf/2505.14719v3)**

> **作者:** Wei Hua; Chenlin Zhou; Jibin Wu; Yansong Chua; Yangyang Shu
>
> **备注:** 11pages, 2figures, accepted by IJCAI'25 (34th International Joint Conference on Artificial Intelligence)
>
> **摘要:** The combination of Spiking Neural Networks (SNNs) with Vision Transformer architectures has garnered significant attention due to their potential for energy-efficient and high-performance computing paradigms. However, a substantial performance gap still exists between SNN-based and ANN-based transformer architectures. While existing methods propose spiking self-attention mechanisms that are successfully combined with SNNs, the overall architectures proposed by these methods suffer from a bottleneck in effectively extracting features from different image scales. In this paper, we address this issue and propose MSVIT. This novel spike-driven Transformer architecture firstly uses multi-scale spiking attention (MSSA) to enhance the capabilities of spiking attention blocks. We validate our approach across various main datasets. The experimental results show that MSVIT outperforms existing SNN-based models, positioning itself as a state-of-the-art solution among SNN-transformer architectures. The codes are available at https://github.com/Nanhu-AI-Lab/MSViT.
>
---
#### [replaced 050] Incorporating Pre-training Data Matters in Unsupervised Domain Adaptation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2308.03097v2](http://arxiv.org/pdf/2308.03097v2)**

> **作者:** Yinsong Xu; Aidong Men; Yang Liu; Xiahai Zhuang; Qingchao Chen
>
> **摘要:** In deep learning, initializing models with pre-trained weights has become the de facto practice for various downstream tasks. Many unsupervised domain adaptation (UDA) methods typically adopt a backbone pre-trained on ImageNet, and focus on reducing the source-target domain discrepancy. However, the impact of pre-training on adaptation received little attention. In this study, we delve into UDA from the novel perspective of pre-training. We first demonstrate the impact of pre-training by analyzing the dynamic distribution discrepancies between pre-training data domain and the source/ target domain during adaptation. Then, we reveal that the target error also stems from the pre-training in the following two factors: 1) empirically, target error arises from the gradually degenerative pre-trained knowledge during adaptation; 2) theoretically, the error bound depends on difference between the gradient of loss function, \ie, on the target domain and pre-training data domain. To address these two issues, we redefine UDA as a three-domain problem, \ie, source domain, target domain, and pre-training data domain; then we propose a novel framework, named TriDA. We maintain the pre-trained knowledge and improve the error bound by incorporating pre-training data into adaptation for both vanilla UDA and source-free UDA scenarios. For efficiency, we introduce a selection strategy for pre-training data, and offer a solution with synthesized images when pre-training data is unavailable during adaptation. Notably, TriDA is effective even with a small amount of pre-training or synthesized images, and seamlessly complements the two scenario UDA methods, demonstrating state-of-the-art performance across multiple benchmarks. We hope our work provides new insights for better understanding and application of domain adaptation.
>
---
#### [replaced 051] PRO: Projection Domain Synthesis for CT Imaging
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13443v2](http://arxiv.org/pdf/2506.13443v2)**

> **作者:** Kang Chen; Bin Huang; Xuebin Yang; Junyan Zhang; Qiegen Liu
>
> **摘要:** Synthesizing high quality CT projection data remains a significant challenge due to the limited availability of annotated data and the complex nature of CT imaging. In this work, we present PRO, a projection domain synthesis foundation model for CT imaging. To the best of our knowledge, this is the first study that performs CT synthesis in the projection domain. Unlike previous approaches that operate in the image domain, PRO learns rich structural representations from raw projection data and leverages anatomical text prompts for controllable synthesis. This projection domain strategy enables more faithful modeling of underlying imaging physics and anatomical structures. Moreover, PRO functions as a foundation model, capable of generalizing across diverse downstream tasks by adjusting its generative behavior via prompt inputs. Experimental results demonstrated that incorporating our synthesized data significantly improves performance across multiple downstream tasks, including low-dose and sparse-view reconstruction. These findings underscore the versatility and scalability of PRO in data generation for various CT applications. These results highlight the potential of projection domain synthesis as a powerful tool for data augmentation and robust CT imaging. Our source code is publicly available at: https://github.com/yqx7150/PRO.
>
---
#### [replaced 052] SUEDE:Shared Unified Experts for Physical-Digital Face Attack Detection Enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.04818v2](http://arxiv.org/pdf/2504.04818v2)**

> **作者:** Zuying Xie; Changtao Miao; Ajian Liu; Jiabao Guo; Feng Li; Dan Guo; Yunfeng Diao
>
> **备注:** Accepted in ICME 2025 (Oral)
>
> **摘要:** Face recognition systems are vulnerable to physical attacks (e.g., printed photos) and digital threats (e.g., DeepFake), which are currently being studied as independent visual tasks, such as Face Anti-Spoofing and Forgery Detection. The inherent differences among various attack types present significant challenges in identifying a common feature space, making it difficult to develop a unified framework for detecting data from both attack modalities simultaneously. Inspired by the efficacy of Mixture-of-Experts (MoE) in learning across diverse domains, we explore utilizing multiple experts to learn the distinct features of various attack types. However, the feature distributions of physical and digital attacks overlap and differ. This suggests that relying solely on distinct experts to learn the unique features of each attack type may overlook shared knowledge between them. To address these issues, we propose SUEDE, the Shared Unified Experts for Physical-Digital Face Attack Detection Enhancement. SUEDE combines a shared expert (always activated) to capture common features for both attack types and multiple routed experts (selectively activated) for specific attack types. Further, we integrate CLIP as the base network to ensure the shared expert benefits from prior visual knowledge and align visual-text representations in a unified space. Extensive results demonstrate SUEDE achieves superior performance compared to state-of-the-art unified detection methods.
>
---
#### [replaced 053] YOLOv11-RGBT: Towards a Comprehensive Single-Stage Multispectral Object Detection Framework
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14696v2](http://arxiv.org/pdf/2506.14696v2)**

> **作者:** Dahang Wan; Rongsheng Lu; Yang Fang; Xianli Lang; Shuangbao Shu; Jingjing Chen; Siyuan Shen; Ting Xu; Zecong Ye
>
> **备注:** 29 pages, 8 figures . The errors in the first version have been corrected, and no new version will be submitted in the near future. The next version will include more experiments
>
> **摘要:** Multispectral object detection, which integrates information from multiple bands, can enhance detection accuracy and environmental adaptability, holding great application potential across various fields. Although existing methods have made progress in cross-modal interaction, low-light conditions, and model lightweight, there are still challenges like the lack of a unified single-stage framework, difficulty in balancing performance and fusion strategy, and unreasonable modality weight allocation. To address these, based on the YOLOv11 framework, we present YOLOv11-RGBT, a new comprehensive multimodal object detection framework. We designed six multispectral fusion modes and successfully applied them to models from YOLOv3 to YOLOv12 and RT-DETR. After reevaluating the importance of the two modalities, we proposed a P3 mid-fusion strategy and multispectral controllable fine-tuning (MCF) strategy for multispectral models. These improvements optimize feature fusion, reduce redundancy and mismatches, and boost overall model performance. Experiments show our framework excels on three major open-source multispectral object detection datasets, like LLVIP and FLIR. Particularly, the multispectral controllable fine-tuning strategy significantly enhanced model adaptability and robustness. On the FLIR dataset, it consistently improved YOLOv11 models' mAP by 3.41%-5.65%, reaching a maximum of 47.61%, verifying the framework and strategies' effectiveness. The code is available at: https://github.com/wandahangFY/YOLOv11-RGBT.
>
---
#### [replaced 054] SFDLA: Source-Free Document Layout Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18742v2](http://arxiv.org/pdf/2503.18742v2)**

> **作者:** Sebastian Tewes; Yufan Chen; Omar Moured; Jiaming Zhang; Rainer Stiefelhagen
>
> **备注:** Accepted by ICDAR 2025. The benchmark, models, and code will be publicly available at https://github.com/s3setewe/sfdla-DLAdapter
>
> **摘要:** Document Layout Analysis (DLA) is a fundamental task in document understanding. However, existing DLA and adaptation methods often require access to large-scale source data and target labels. This requirements severely limiting their real-world applicability, particularly in privacy-sensitive and resource-constrained domains, such as financial statements, medical records, and proprietary business documents. According to our observation, directly transferring source-domain fine-tuned models on target domains often results in a significant performance drop (Avg. -32.64%). In this work, we introduce Source-Free Document Layout Analysis (SFDLA), aiming for adapting a pre-trained source DLA models to an unlabeled target domain, without access to any source data. To address this challenge, we establish the first SFDLA benchmark, covering three major DLA datasets for geometric- and content-aware adaptation. Furthermore, we propose Document Layout Analysis Adapter (DLAdapter), a novel framework that is designed to improve source-free adaptation across document domains. Our method achieves a +4.21% improvement over the source-only baseline and a +2.26% gain over existing source-free methods from PubLayNet to DocLayNet. We believe this work will inspire the DLA community to further investigate source-free document understanding. To support future research of the community, the benchmark, models, and code will be publicly available at https://github.com/s3setewe/sfdla-DLAdapter.
>
---
#### [replaced 055] Improved Convex Decomposition with Ensembling and Boolean Primitives
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.19569v3](http://arxiv.org/pdf/2405.19569v3)**

> **作者:** Vaibhav Vavilala; Florian Kluger; Seemandhar Jain; Bodo Rosenhahn; Anand Bhattad; David Forsyth
>
> **备注:** 25 pages, 16 figures, 9 tables
>
> **摘要:** Describing a scene in terms of primitives -- geometrically simple shapes that offer a parsimonious but accurate abstraction of structure -- is an established and difficult fitting problem. Different scenes require different numbers of primitives, and these primitives interact strongly. Existing methods are evaluated by predicting depth, normals and segmentation from the primitives, then evaluating the accuracy of those predictions. The state of the art method involves a learned regression procedure to predict a start point consisting of a fixed number of primitives, followed by a descent method to refine the geometry and remove redundant primitives. CSG (Constructive Solid Geometry) representations are significantly enhanced by a set-differencing operation. Our representation incorporates negative primitives, which are differenced from the positive primitives. These notably enrich the geometry that the model can encode, while complicating the fitting problem. This paper demonstrates a method that can (a) incorporate these negative primitives and (b) choose the overall number of positive and negative primitives by ensembling. Extensive experiments on the standard NYUv2 dataset confirm that (a) this approach results in substantial improvements in depth representation and segmentation over SOTA and (b) negative primitives make a notable contribution to accuracy. Our method is robustly applicable across datasets: in a first, we evaluate primitive prediction for LAION images.
>
---
#### [replaced 056] EmoEdit: Evoking Emotions through Image Manipulation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.12661v3](http://arxiv.org/pdf/2405.12661v3)**

> **作者:** Jingyuan Yang; Jiawei Feng; Weibin Luo; Dani Lischinski; Daniel Cohen-Or; Hui Huang
>
> **摘要:** Affective Image Manipulation (AIM) seeks to modify user-provided images to evoke specific emotional responses. This task is inherently complex due to its twofold objective: significantly evoking the intended emotion, while preserving the original image composition. Existing AIM methods primarily adjust color and style, often failing to elicit precise and profound emotional shifts. Drawing on psychological insights, we introduce EmoEdit, which extends AIM by incorporating content modifications to enhance emotional impact. Specifically, we first construct EmoEditSet, a large-scale AIM dataset comprising 40,120 paired data through emotion attribution and data construction. To make existing generative models emotion-aware, we design the Emotion adapter and train it using EmoEditSet. We further propose an instruction loss to capture the semantic variations in data pairs. Our method is evaluated both qualitatively and quantitatively, demonstrating superior performance compared to existing state-of-the-art techniques. Additionally, we showcase the portability of our Emotion adapter to other diffusion-based models, enhancing their emotion knowledge with diverse semantics.
>
---
#### [replaced 057] Generalized Out-of-Distribution Detection and Beyond in Vision Language Model Era: A Survey
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.21794v2](http://arxiv.org/pdf/2407.21794v2)**

> **作者:** Atsuyuki Miyai; Jingkang Yang; Jingyang Zhang; Yifei Ming; Yueqian Lin; Qing Yu; Go Irie; Shafiq Joty; Yixuan Li; Hai Li; Ziwei Liu; Toshihiko Yamasaki; Kiyoharu Aizawa
>
> **备注:** Accepted at TMLR2025. Survey paper. We welcome questions, issues, and paper requests via https://github.com/AtsuMiyai/Awesome-OOD-VLM
>
> **摘要:** Detecting out-of-distribution (OOD) samples is crucial for ensuring the safety of machine learning systems and has shaped the field of OOD detection. Meanwhile, several other problems are closely related to OOD detection, including anomaly detection (AD), novelty detection (ND), open set recognition (OSR), and outlier detection (OD). To unify these problems, a generalized OOD detection framework was proposed, taxonomically categorizing these five problems. However, Vision Language Models (VLMs) such as CLIP have significantly changed the paradigm and blurred the boundaries between these fields, again confusing researchers. In this survey, we first present a generalized OOD detection v2, encapsulating the evolution of these fields in the VLM era. Our framework reveals that, with some field inactivity and integration, the demanding challenges have become OOD detection and AD. Then, we highlight the significant shift in the definition, problem settings, and benchmarks; we thus feature a comprehensive review of the methodology for OOD detection and related tasks to clarify their relationship to OOD detection. Finally, we explore the advancements in the emerging Large Vision Language Model (LVLM) era, such as GPT-4V. We conclude with open challenges and future directions. The resource is available at https://github.com/AtsuMiyai/Awesome-OOD-VLM.
>
---
#### [replaced 058] Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.06940v4](http://arxiv.org/pdf/2410.06940v4)**

> **作者:** Sihyun Yu; Sangkyung Kwak; Huiwon Jang; Jongheon Jeong; Jonathan Huang; Jinwoo Shin; Saining Xie
>
> **备注:** ICLR 2025 (Oral). Project page: https://sihyun.me/REPA
>
> **摘要:** Recent studies have shown that the denoising process in (generative) diffusion models can induce meaningful (discriminative) representations inside the model, though the quality of these representations still lags behind those learned through recent self-supervised learning methods. We argue that one main bottleneck in training large-scale diffusion models for generation lies in effectively learning these representations. Moreover, training can be made easier by incorporating high-quality external visual representations, rather than relying solely on the diffusion models to learn them independently. We study this by introducing a straightforward regularization called REPresentation Alignment (REPA), which aligns the projections of noisy input hidden states in denoising networks with clean image representations obtained from external, pretrained visual encoders. The results are striking: our simple strategy yields significant improvements in both training efficiency and generation quality when applied to popular diffusion and flow-based transformers, such as DiTs and SiTs. For instance, our method can speed up SiT training by over 17.5$\times$, matching the performance (without classifier-free guidance) of a SiT-XL model trained for 7M steps in less than 400K steps. In terms of final generation quality, our approach achieves state-of-the-art results of FID=1.42 using classifier-free guidance with the guidance interval.
>
---
