# 计算机视觉 cs.CV

- **最新发布 71 篇**

- **更新 39 篇**

## 最新发布

#### [new 001] How Well Do Vision--Language Models Understand Cities? A Comparative Study on Spatial Reasoning from Street-View Images
- **分类: cs.CV**

- **简介: 论文评估视觉语言模型在城市场景中的空间推理能力，解决通用模型向城市领域的迁移问题，通过构建合成数据集与微调实验，比较BLIP-2、InstructBLIP和LLaVA-1.5的性能差异。**

- **链接: [http://arxiv.org/pdf/2508.21565v1](http://arxiv.org/pdf/2508.21565v1)**

> **作者:** Juneyoung Ro; Namwoo Kim; Yoonjin Yoon
>
> **备注:** Accepted to ICCV Workshop 2025
>
> **摘要:** Effectively understanding urban scenes requires fine-grained spatial reasoning about objects, layouts, and depth cues. However, how well current vision-language models (VLMs), pretrained on general scenes, transfer these abilities to urban domain remains underexplored. To address this gap, we conduct a comparative study of three off-the-shelf VLMs-BLIP-2, InstructBLIP, and LLaVA-1.5-evaluating both zero-shot performance and the effects of fine-tuning with a synthetic VQA dataset specific to urban scenes. We construct such dataset from segmentation, depth, and object detection predictions of street-view images, pairing each question with LLM-generated Chain-of-Thought (CoT) answers for step-by-step reasoning supervision. Results show that while VLMs perform reasonably well in zero-shot settings, fine-tuning with our synthetic CoT-supervised dataset substantially boosts performance, especially for challenging question types such as negation and counterfactuals. This study introduces urban spatial reasoning as a new challenge for VLMs and demonstrates synthetic dataset construction as a practical path for adapting general-purpose models to specialized domains.
>
---
#### [new 002] Benchmarking GPT-5 in Radiation Oncology: Measurable Gains, but Persistent Need for Expert Oversight
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文通过基准测试评估GPT-5在放射肿瘤学中的表现，发现其准确率优于前代模型，但复杂场景下仍存在错误，需专家监督。**

- **链接: [http://arxiv.org/pdf/2508.21777v1](http://arxiv.org/pdf/2508.21777v1)**

> **作者:** Ugur Dinc; Jibak Sarkar; Philipp Schubert; Sabine Semrau; Thomas Weissmann; Andre Karius; Johann Brand; Bernd-Niklas Axer; Ahmed Gomaa; Pluvio Stephan; Ishita Sheth; Sogand Beirami; Annette Schwarz; Udo Gaipl; Benjamin Frey; Christoph Bert; Stefanie Corradini; Rainer Fietkau; Florian Putz
>
> **备注:** Under review in Frontiers in Artificial Intelligence
>
> **摘要:** Introduction: Large language models (LLM) have shown great potential in clinical decision support. GPT-5 is a novel LLM system that has been specifically marketed towards oncology use. Methods: Performance was assessed using two complementary benchmarks: (i) the ACR Radiation Oncology In-Training Examination (TXIT, 2021), comprising 300 multiple-choice items, and (ii) a curated set of 60 authentic radiation oncologic vignettes representing diverse disease sites and treatment indications. For the vignette evaluation, GPT-5 was instructed to generate concise therapeutic plans. Four board-certified radiation oncologists rated correctness, comprehensiveness, and hallucinations. Inter-rater reliability was quantified using Fleiss' \k{appa}. Results: On the TXIT benchmark, GPT-5 achieved a mean accuracy of 92.8%, outperforming GPT-4 (78.8%) and GPT-3.5 (62.1%). Domain-specific gains were most pronounced in Dose and Diagnosis. In the vignette evaluation, GPT-5's treatment recommendations were rated highly for correctness (mean 3.24/4, 95% CI: 3.11-3.38) and comprehensiveness (3.59/4, 95% CI: 3.49-3.69). Hallucinations were rare with no case reaching majority consensus for their presence. Inter-rater agreement was low (Fleiss' \k{appa} 0.083 for correctness), reflecting inherent variability in clinical judgment. Errors clustered in complex scenarios requiring precise trial knowledge or detailed clinical adaptation. Discussion: GPT-5 clearly outperformed prior model variants on the radiation oncology multiple-choice benchmark. Although GPT-5 exhibited favorable performance in generating real-world radiation oncology treatment recommendations, correctness ratings indicate room for further improvement. While hallucinations were infrequent, the presence of substantive errors underscores that GPT-5-generated recommendations require rigorous expert oversight before clinical implementation.
>
---
#### [new 003] What Can We Learn from Harry Potter? An Exploratory Study of Visual Representation Learning from Atypical Videos
- **分类: cs.CV**

- **简介: 该论文研究开放世界视觉表示学习，探索不典型视频对模型性能的影响。通过新数据集和实验，发现不典型数据提升OOD检测、NCD和ZSAR任务表现，尤其强调语义多样性的重要性。**

- **链接: [http://arxiv.org/pdf/2508.21770v1](http://arxiv.org/pdf/2508.21770v1)**

> **作者:** Qiyue Sun; Qiming Huang; Yang Yang; Hongjun Wang; Jianbo Jiao
>
> **摘要:** Humans usually show exceptional generalisation and discovery ability in the open world, when being shown uncommon new concepts. Whereas most existing studies in the literature focus on common typical data from closed sets, open-world novel discovery is under-explored in videos. In this paper, we are interested in asking: \textit{What if atypical unusual videos are exposed in the learning process?} To this end, we collect a new video dataset consisting of various types of unusual atypical data (\eg sci-fi, animation, \etc). To study how such atypical data may benefit open-world learning, we feed them into the model training process for representation learning. Focusing on three key tasks in open-world learning: out-of-distribution (OOD) detection, novel category discovery (NCD), and zero-shot action recognition (ZSAR), we found that even straightforward learning approaches with atypical data consistently improve performance across various settings. Furthermore, we found that increasing the categorical diversity of the atypical samples further boosts OOD detection performance. Additionally, in the NCD task, using a smaller yet more semantically diverse set of atypical samples leads to better performance compared to using a larger but more typical dataset. In the ZSAR setting, the semantic diversity of atypical videos helps the model generalise better to unseen action classes. These observations in our extensive experimental evaluations reveal the benefits of atypical videos for visual representation learning in the open world, together with the newly proposed dataset, encouraging further studies in this direction.
>
---
#### [new 004] HiddenObject: Modality-Agnostic Fusion for Multimodal Hidden Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文提出HiddenObject框架，通过Mamba融合RGB、热成像与深度数据，解决多模态环境下隐藏物体检测问题，提升遮挡与复杂条件下的检测性能。**

- **链接: [http://arxiv.org/pdf/2508.21135v1](http://arxiv.org/pdf/2508.21135v1)**

> **作者:** Harris Song; Tuan-Anh Vu; Sanjith Menon; Sriram Narasimhan; M. Khalid Jawed
>
> **摘要:** Detecting hidden or partially concealed objects remains a fundamental challenge in multimodal environments, where factors like occlusion, camouflage, and lighting variations significantly hinder performance. Traditional RGB-based detection methods often fail under such adverse conditions, motivating the need for more robust, modality-agnostic approaches. In this work, we present HiddenObject, a fusion framework that integrates RGB, thermal, and depth data using a Mamba-based fusion mechanism. Our method captures complementary signals across modalities, enabling enhanced detection of obscured or camouflaged targets. Specifically, the proposed approach identifies modality-specific features and fuses them in a unified representation that generalizes well across challenging scenarios. We validate HiddenObject across multiple benchmark datasets, demonstrating state-of-the-art or competitive performance compared to existing methods. These results highlight the efficacy of our fusion design and expose key limitations in current unimodal and na\"ive fusion strategies. More broadly, our findings suggest that Mamba-based fusion architectures can significantly advance the field of multimodal object detection, especially under visually degraded or complex conditions.
>
---
#### [new 005] Learning from Silence and Noise for Visual Sound Source Localization
- **分类: cs.CV; cs.MM**

- **简介: 该论文聚焦视觉声音定位任务，解决现有方法在处理沉默、噪音等负面音频及单一声音源场景时表现不佳的问题。提出新训练策略、度量标准及改进数据集IS3+，提升模型在正负音频场景下的鲁棒性与定位精度。**

- **链接: [http://arxiv.org/pdf/2508.21761v1](http://arxiv.org/pdf/2508.21761v1)**

> **作者:** Xavier Juanola; Giovana Morais; Magdalena Fuentes; Gloria Haro
>
> **备注:** 10 pages, 2 figures, 4 tables + Supplementary Material
>
> **摘要:** Visual sound source localization is a fundamental perception task that aims to detect the location of sounding sources in a video given its audio. Despite recent progress, we identify two shortcomings in current methods: 1) most approaches perform poorly in cases with low audio-visual semantic correspondence such as silence, noise, and offscreen sounds, i.e. in the presence of negative audio; and 2) most prior evaluations are limited to positive cases, where both datasets and metrics convey scenarios with a single visible sound source in the scene. To address this, we introduce three key contributions. First, we propose a new training strategy that incorporates silence and noise, which improves performance in positive cases, while being more robust against negative sounds. Our resulting self-supervised model, SSL-SaN, achieves state-of-the-art performance compared to other self-supervised models, both in sound localization and cross-modal retrieval. Second, we propose a new metric that quantifies the trade-off between alignment and separability of auditory and visual features across positive and negative audio-visual pairs. Third, we present IS3+, an extended and improved version of the IS3 synthetic dataset with negative audio. Our data, metrics and code are available on the https://xavijuanola.github.io/SSL-SaN/.
>
---
#### [new 006] Adversarial Patch Attack for Ship Detection via Localized Augmentation
- **分类: cs.CV**

- **简介: 该论文针对船舶检测中的对抗补丁攻击问题，提出局部增强方法，通过仅对目标区域进行数据增强，减少背景干扰，提升攻击成功率与可迁移性。**

- **链接: [http://arxiv.org/pdf/2508.21472v1](http://arxiv.org/pdf/2508.21472v1)**

> **作者:** Chun Liu; Panpan Ding; Zheng Zheng; Hailong Wang; Bingqian Zhu; Tao Xu; Zhigang Han; Jiayao Wang
>
> **摘要:** Current ship detection techniques based on remote sensing imagery primarily rely on the object detection capabilities of deep neural networks (DNNs). However, DNNs are vulnerable to adversarial patch attacks, which can lead to misclassification by the detection model or complete evasion of the targets. Numerous studies have demonstrated that data transformation-based methods can improve the transferability of adversarial examples. However, excessive augmentation of image backgrounds or irrelevant regions may introduce unnecessary interference, resulting in false detections of the object detection model. These errors are not caused by the adversarial patches themselves but rather by the over-augmentation of background and non-target areas. This paper proposes a localized augmentation method that applies augmentation only to the target regions, avoiding any influence on non-target areas. By reducing background interference, this approach enables the loss function to focus more directly on the impact of the adversarial patch on the detection model, thereby improving the attack success rate. Experiments conducted on the HRSC2016 dataset demonstrate that the proposed method effectively increases the success rate of adversarial patch attacks and enhances their transferability.
>
---
#### [new 007] ECHO: Ego-Centric modeling of Human-Object interactions
- **分类: cs.CV**

- **简介: 该论文提出ECHO框架，通过头和手腕追踪数据统一建模人-物交互的三模态信息（姿态、运动、接触），采用扩散Transformer和三元扩散过程实现灵活序列处理，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.21556v1](http://arxiv.org/pdf/2508.21556v1)**

> **作者:** Ilya A. Petrov; Vladimir Guzov; Riccardo Marin; Emre Aksan; Xu Chen; Daniel Cremers; Thabo Beeler; Gerard Pons-Moll
>
> **摘要:** Modeling human-object interactions (HOI) from an egocentric perspective is a largely unexplored yet important problem due to the increasing adoption of wearable devices, such as smart glasses and watches. We investigate how much information about interaction can be recovered from only head and wrists tracking. Our answer is ECHO (Ego-Centric modeling of Human-Object interactions), which, for the first time, proposes a unified framework to recover three modalities: human pose, object motion, and contact from such minimal observation. ECHO employs a Diffusion Transformer architecture and a unique three-variate diffusion process, which jointly models human motion, object trajectory, and contact sequence, allowing for flexible input configurations. Our method operates in a head-centric canonical space, enhancing robustness to global orientation. We propose a conveyor-based inference, which progressively increases the diffusion timestamp with the frame position, allowing us to process sequences of any length. Through extensive evaluation, we demonstrate that ECHO outperforms existing methods that do not offer the same flexibility, setting a state-of-the-art in egocentric HOI reconstruction.
>
---
#### [new 008] UItron: Foundational GUI Agent with Advanced Perception and Planning
- **分类: cs.CV**

- **简介: 论文提出UItron，解决GUI代理中操作轨迹稀缺、基础设施不足及模型能力限制问题，通过系统数据工程、交互环境构建及课程强化学习提升性能，尤其在中文APP场景表现突出。**

- **链接: [http://arxiv.org/pdf/2508.21767v1](http://arxiv.org/pdf/2508.21767v1)**

> **作者:** Zhixiong Zeng; Jing Huang; Liming Zheng; Wenkang Han; Yufeng Zhong; Lei Chen; Longrong Yang; Yingjie Chu; Yuzhi He; Lin Ma
>
> **备注:** 24 pages
>
> **摘要:** GUI agent aims to enable automated operations on Mobile/PC devices, which is an important task toward achieving artificial general intelligence. The rapid advancement of VLMs accelerates the development of GUI agents, owing to their powerful capabilities in visual understanding and task planning. However, building a GUI agent remains a challenging task due to the scarcity of operation trajectories, the availability of interactive infrastructure, and the limitation of initial capabilities in foundation models. In this work, we introduce UItron, an open-source foundational model for automatic GUI agents, featuring advanced GUI perception, grounding, and planning capabilities. UItron highlights the necessity of systemic data engineering and interactive infrastructure as foundational components for advancing GUI agent development. It not only systematically studies a series of data engineering strategies to enhance training effects, but also establishes an interactive environment connecting both Mobile and PC devices. In training, UItron adopts supervised finetuning over perception and planning tasks in various GUI scenarios, and then develop a curriculum reinforcement learning framework to enable complex reasoning and exploration for online environments. As a result, UItron achieves superior performance in benchmarks of GUI perception, grounding, and planning. In particular, UItron highlights the interaction proficiency with top-tier Chinese mobile APPs, as we identified a general lack of Chinese capabilities even in state-of-the-art solutions. To this end, we manually collect over one million steps of operation trajectories across the top 100 most popular apps, and build the offline and online agent evaluation environments. Experimental results demonstrate that UItron achieves significant progress in Chinese app scenarios, propelling GUI agents one step closer to real-world application.
>
---
#### [new 009] Unfolding Framework with Complex-Valued Deformable Attention for High-Quality Computer-Generated Hologram Generation
- **分类: cs.CV**

- **简介: 该论文提出深度展开网络（DUN），通过复值变形自注意力模块和自适应带宽保持模型，解决CGH生成中重建不准确、接收场受限及近场约束问题，实现高PSNR（>35dB）的高质量全息图生成。**

- **链接: [http://arxiv.org/pdf/2508.21657v1](http://arxiv.org/pdf/2508.21657v1)**

> **作者:** Haomiao Zhang; Zhangyuan Li; Yanling Piao; Zhi Li; Xiaodong Wang; Miao Cao; Xiongfei Su; Qiang Song; Xin Yuan
>
> **摘要:** Computer-generated holography (CGH) has gained wide attention with deep learning-based algorithms. However, due to its nonlinear and ill-posed nature, challenges remain in achieving accurate and stable reconstruction. Specifically, ($i$) the widely used end-to-end networks treat the reconstruction model as a black box, ignoring underlying physical relationships, which reduces interpretability and flexibility. ($ii$) CNN-based CGH algorithms have limited receptive fields, hindering their ability to capture long-range dependencies and global context. ($iii$) Angular spectrum method (ASM)-based models are constrained to finite near-fields.In this paper, we propose a Deep Unfolding Network (DUN) that decomposes gradient descent into two modules: an adaptive bandwidth-preserving model (ABPM) and a phase-domain complex-valued denoiser (PCD), providing more flexibility. ABPM allows for wider working distances compared to ASM-based methods. At the same time, PCD leverages its complex-valued deformable self-attention module to capture global features and enhance performance, achieving a PSNR over 35 dB. Experiments on simulated and real data show state-of-the-art results.
>
---
#### [new 010] Print2Volume: Generating Synthetic OCT-based 3D Fingerprint Volume from 2D Fingerprint Image
- **分类: cs.CV**

- **简介: 该论文提出Print2Volume框架，通过三阶段生成合成OCT三维指纹数据，解决OCT数据稀缺问题，提升生物识别性能。**

- **链接: [http://arxiv.org/pdf/2508.21371v1](http://arxiv.org/pdf/2508.21371v1)**

> **作者:** Qingran Miao; Haixia Wang; Haohao Sun; Yilong Zhang
>
> **摘要:** Optical Coherence Tomography (OCT) enables the acquisition of high-resolution, three-dimensional fingerprint data, capturing rich subsurface structures for robust biometric recognition. However, the high cost and time-consuming nature of OCT data acquisition have led to a scarcity of large-scale public datasets, significantly hindering the development of advanced algorithms, particularly data-hungry deep learning models. To address this critical bottleneck, this paper introduces Print2Volume, a novel framework for generating realistic, synthetic OCT-based 3D fingerprints from 2D fingerprint image. Our framework operates in three sequential stages: (1) a 2D style transfer module that converts a binary fingerprint into a grayscale images mimicking the style of a Z-direction mean-projected OCT scan; (2) a 3D Structure Expansion Network that extrapolates the 2D im-age into a plausible 3D anatomical volume; and (3) an OCT Realism Refiner, based on a 3D GAN, that renders the structural volume with authentic textures, speckle noise, and other imaging characteristics. Using Print2Volume, we generated a large-scale synthetic dataset of 420,000 samples. Quantitative experiments demonstrate the high quality of our synthetic data and its significant impact on recognition performance. By pre-training a recognition model on our synthetic data and fine-tuning it on a small real-world dataset, we achieved a remarkable reduction in the Equal Error Rate (EER) from 15.62% to 2.50% on the ZJUT-EIFD benchmark, proving the effectiveness of our approach in overcoming data scarcity.
>
---
#### [new 011] SatDINO: A Deep Dive into Self-Supervised Pretraining for Remote Sensing
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出SatDINO模型，基于DINO的对比自监督方法，解决遥感图像表示学习问题，通过实验验证其优越性并引入GSD编码等增强技术。**

- **链接: [http://arxiv.org/pdf/2508.21402v1](http://arxiv.org/pdf/2508.21402v1)**

> **作者:** Jakub Straka; Ivan Gruber
>
> **摘要:** Self-supervised learning has emerged as a powerful tool for remote sensing, where large amounts of unlabeled data are available. In this work, we investigate the use of DINO, a contrastive self-supervised method, for pretraining on remote sensing imagery. We introduce SatDINO, a model tailored for representation learning in satellite imagery. Through extensive experiments on multiple datasets in multiple testing setups, we demonstrate that SatDINO outperforms other state-of-the-art methods based on much more common masked autoencoders (MAE) and achieves competitive results in multiple benchmarks. We also provide a rigorous ablation study evaluating SatDINO's individual components. Finally, we propose a few novel enhancements, such as a new way to incorporate ground sample distance (GSD) encoding and adaptive view sampling. These enhancements can be used independently on our SatDINO model. Our code and trained models are available at: https://github.com/strakaj/SatDINO.
>
---
#### [new 012] Q-Align: Alleviating Attention Leakage in Zero-Shot Appearance Transfer via Query-Query Alignment
- **分类: cs.CV**

- **简介: 该论文针对零样本外观迁移中的注意力泄漏问题，提出Q-Align方法，通过查询-查询对齐、键值重排和注意力细化，提升语义对齐与外观保真度。**

- **链接: [http://arxiv.org/pdf/2508.21090v1](http://arxiv.org/pdf/2508.21090v1)**

> **作者:** Namu Kim; Wonbin Kweon; Minsoo Kim; Hwanjo Yu
>
> **摘要:** We observe that zero-shot appearance transfer with large-scale image generation models faces a significant challenge: Attention Leakage. This challenge arises when the semantic mapping between two images is captured by the Query-Key alignment. To tackle this issue, we introduce Q-Align, utilizing Query-Query alignment to mitigate attention leakage and improve the semantic alignment in zero-shot appearance transfer. Q-Align incorporates three core contributions: (1) Query-Query alignment, facilitating the sophisticated spatial semantic mapping between two images; (2) Key-Value rearrangement, enhancing feature correspondence through realignment; and (3) Attention refinement using rearranged keys and values to maintain semantic consistency. We validate the effectiveness of Q-Align through extensive experiments and analysis, and Q-Align outperforms state-of-the-art methods in appearance fidelity while maintaining competitive structure preservation.
>
---
#### [new 013] GENNAV: Polygon Mask Generation for Generalized Referring Navigable Regions
- **分类: cs.CV; cs.RO**

- **简介: 论文提出GENNAV方法，解决基于自然语言和图像识别模糊边界目标区域的问题，通过预测存在性并生成分割掩码，构建GRiN-Drive基准并验证其在真实环境中的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.21102v1](http://arxiv.org/pdf/2508.21102v1)**

> **作者:** Kei Katsumata; Yui Iioka; Naoki Hosomi; Teruhisa Misu; Kentaro Yamada; Komei Sugiura
>
> **备注:** Accepted for presentation at CoRL2025
>
> **摘要:** We focus on the task of identifying the location of target regions from a natural language instruction and a front camera image captured by a mobility. This task is challenging because it requires both existence prediction and segmentation, particularly for stuff-type target regions with ambiguous boundaries. Existing methods often underperform in handling stuff-type target regions, in addition to absent or multiple targets. To overcome these limitations, we propose GENNAV, which predicts target existence and generates segmentation masks for multiple stuff-type target regions. To evaluate GENNAV, we constructed a novel benchmark called GRiN-Drive, which includes three distinct types of samples: no-target, single-target, and multi-target. GENNAV achieved superior performance over baseline methods on standard evaluation metrics. Furthermore, we conducted real-world experiments with four automobiles operated in five geographically distinct urban areas to validate its zero-shot transfer performance. In these experiments, GENNAV outperformed baseline methods and demonstrated its robustness across diverse real-world environments. The project page is available at https://gennav.vercel.app/.
>
---
#### [new 014] ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion
- **分类: cs.CV**

- **简介: 该论文提出ERTACache框架，针对扩散模型推理效率低的问题，通过分析缓存引入的特征偏移与步骤放大误差，设计误差校正与时间步动态调整方法，在保持质量的前提下实现2倍推理加速。**

- **链接: [http://arxiv.org/pdf/2508.21091v1](http://arxiv.org/pdf/2508.21091v1)**

> **作者:** Xurui Peng; Hong Liu; Chenqian Yan; Rui Ma; Fangmin Chen; Xing Wang; Zhihua Wu; Songwei Liu; Mingbao Lin
>
> **摘要:** Diffusion models suffer from substantial computational overhead due to their inherently iterative inference process. While feature caching offers a promising acceleration strategy by reusing intermediate outputs across timesteps, naive reuse often incurs noticeable quality degradation. In this work, we formally analyze the cumulative error introduced by caching and decompose it into two principal components: feature shift error, caused by inaccuracies in cached outputs, and step amplification error, which arises from error propagation under fixed timestep schedules. To address these issues, we propose ERTACache, a principled caching framework that jointly rectifies both error types. Our method employs an offline residual profiling stage to identify reusable steps, dynamically adjusts integration intervals via a trajectory-aware correction coefficient, and analytically approximates cache-induced errors through a closed-form residual linearization model. Together, these components enable accurate and efficient sampling under aggressive cache reuse. Extensive experiments across standard image and video generation benchmarks show that ERTACache achieves up to 2x inference speedup while consistently preserving or even improving visual quality. Notably, on the state-of-the-art Wan2.1 video diffusion model, ERTACache delivers 2x acceleration with minimal VBench degradation, effectively maintaining baseline fidelity while significantly improving efficiency. The code is available at https://github.com/bytedance/ERTACache.
>
---
#### [new 015] ROBUST-MIPS: A Combined Skeletal Pose and Instance Segmentation Dataset for Laparoscopic Surgical Instruments
- **分类: cs.CV**

- **简介: 该论文构建ROBUST-MIPS数据集，结合骨骼姿态与实例分割任务，解决腹腔镜手术器械定位数据不足及标注效率低的问题，推动姿态注释在医疗场景的应用。**

- **链接: [http://arxiv.org/pdf/2508.21096v1](http://arxiv.org/pdf/2508.21096v1)**

> **作者:** Zhe Han; Charlie Budd; Gongyu Zhang; Huanyu Tian; Christos Bergeles; Tom Vercauteren
>
> **摘要:** Localisation of surgical tools constitutes a foundational building block for computer-assisted interventional technologies. Works in this field typically focus on training deep learning models to perform segmentation tasks. Performance of learning-based approaches is limited by the availability of diverse annotated data. We argue that skeletal pose annotations are a more efficient annotation approach for surgical tools, striking a balance between richness of semantic information and ease of annotation, thus allowing for accelerated growth of available annotated data. To encourage adoption of this annotation style, we present, ROBUST-MIPS, a combined tool pose and tool instance segmentation dataset derived from the existing ROBUST-MIS dataset. Our enriched dataset facilitates the joint study of these two annotation styles and allow head-to-head comparison on various downstream tasks. To demonstrate the adequacy of pose annotations for surgical tool localisation, we set up a simple benchmark using popular pose estimation methods and observe high-quality results. To ease adoption, together with the dataset, we release our benchmark models and custom tool pose annotation software.
>
---
#### [new 016] CAD2DMD-SET: Synthetic Generation Tool of Digital Measurement Device CAD Model Datasets for fine-tuning Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 论文提出CAD2DMD-SET工具，通过3D CAD模型和高保真渲染生成合成DMD数据集，解决LVLMs在复杂现实场景中读取DMD数值的难题，提升模型鲁棒性与性能。**

- **链接: [http://arxiv.org/pdf/2508.21732v1](http://arxiv.org/pdf/2508.21732v1)**

> **作者:** João Valente; Atabak Dehban; Rodrigo Ventura
>
> **摘要:** Recent advancements in Large Vision-Language Models (LVLMs) have demonstrated impressive capabilities across various multimodal tasks. They continue, however, to struggle with trivial scenarios such as reading values from Digital Measurement Devices (DMDs), particularly in real-world conditions involving clutter, occlusions, extreme viewpoints, and motion blur; common in head-mounted cameras and Augmented Reality (AR) applications. Motivated by these limitations, this work introduces CAD2DMD-SET, a synthetic data generation tool designed to support visual question answering (VQA) tasks involving DMDs. By leveraging 3D CAD models, advanced rendering, and high-fidelity image composition, our tool produces diverse, VQA-labelled synthetic DMD datasets suitable for fine-tuning LVLMs. Additionally, we present DMDBench, a curated validation set of 1,000 annotated real-world images designed to evaluate model performance under practical constraints. Benchmarking three state-of-the-art LVLMs using Average Normalised Levenshtein Similarity (ANLS) and further fine-tuning LoRA's of these models with CAD2DMD-SET's generated dataset yielded substantial improvements, with InternVL showcasing a score increase of 200% without degrading on other tasks. This demonstrates that the CAD2DMD-SET training dataset substantially improves the robustness and performance of LVLMs when operating under the previously stated challenging conditions. The CAD2DMD-SET tool is expected to be released as open-source once the final version of this manuscript is prepared, allowing the community to add different measurement devices and generate their own datasets.
>
---
#### [new 017] GLENDA: Gynecologic Laparoscopy Endometriosis Dataset
- **分类: cs.CV; cs.MM**

- **简介: 论文提出GLENDA数据集，解决医疗数据稀缺及手动分析效率低的问题，通过提供子宫内膜异位症的区域标注图像，支持计算机视觉与机器学习在妇科腹腔镜手术中的应用研究。**

- **链接: [http://arxiv.org/pdf/2508.21398v1](http://arxiv.org/pdf/2508.21398v1)**

> **作者:** Andreas Leibetseder; Sabrina Kletz; Klaus Schoeffmann; Simon Keckstein; Jörg Keckstein
>
> **摘要:** Gynecologic laparoscopy as a type of minimally invasive surgery (MIS) is performed via a live feed of a patient's abdomen surveying the insertion and handling of various instruments for conducting treatment. Adopting this kind of surgical intervention not only facilitates a great variety of treatments, the possibility of recording said video streams is as well essential for numerous post-surgical activities, such as treatment planning, case documentation and education. Nonetheless, the process of manually analyzing surgical recordings, as it is carried out in current practice, usually proves tediously time-consuming. In order to improve upon this situation, more sophisticated computer vision as well as machine learning approaches are actively developed. Since most of such approaches heavily rely on sample data, which especially in the medical field is only sparsely available, with this work we publish the Gynecologic Laparoscopy ENdometriosis DAtaset (GLENDA) - an image dataset containing region-based annotations of a common medical condition named endometriosis, i.e. the dislocation of uterine-like tissue. The dataset is the first of its kind and it has been created in collaboration with leading medical experts in the field.
>
---
#### [new 018] R-4B: Incentivizing General-Purpose Auto-Thinking Capability in MLLMs via Bi-Mode Annealing and Reinforce Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出R-4B模型，解决MLLMs在简单问题上冗余思考的问题。通过双模式退火与强化学习，使模型自适应判断是否进行思考，提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2508.21113v1](http://arxiv.org/pdf/2508.21113v1)**

> **作者:** Jie Jiang; Qi Yang; Bolin Ni; Shiming Xiang; Han Hu; Houwen Peng
>
> **备注:** 20 pages, 14 figures, 5 tables
>
> **摘要:** Multimodal Large Language Models (MLLMs) equipped with step-by-step thinking capabilities have demonstrated remarkable performance on complex reasoning problems. However, this thinking process is redundant for simple problems solvable without complex reasoning. To address this inefficiency, we propose R-4B, an auto-thinking MLLM, which can adaptively decide when to think based on problem complexity. The central idea of R-4B is to empower the model with both thinking and non-thinking capabilities using bi-mode annealing, and apply Bi-mode Policy Optimization~(BPO) to improve the model's accuracy in determining whether to activate the thinking process. Specifically, we first train the model on a carefully curated dataset spanning various topics, which contains samples from both thinking and non-thinking modes. Then it undergoes a second phase of training under an improved GRPO framework, where the policy model is forced to generate responses from both modes for each input query. Experimental results show that R-4B achieves state-of-the-art performance across 25 challenging benchmarks. It outperforms Qwen2.5-VL-7B in most tasks and achieves performance comparable to larger models such as Kimi-VL-A3B-Thinking-2506 (16B) on reasoning-intensive benchmarks with lower computational cost.
>
---
#### [new 019] Maybe you don't need a U-Net: convolutional feature upsampling for materials micrograph segmentation
- **分类: cs.CV; cond-mat.mtrl-sci**

- **简介: 该论文提出基于卷积特征上采样的材料显微图像分割方法，解决传统U-Net在处理大尺寸图像和细小特征时的效率问题，通过上采样低分辨率特征实现高效分割，减少标注需求。**

- **链接: [http://arxiv.org/pdf/2508.21529v1](http://arxiv.org/pdf/2508.21529v1)**

> **作者:** Ronan Docherty; Antonis Vamvakeros; Samuel J. Cooper
>
> **摘要:** Feature foundation models - usually vision transformers - offer rich semantic descriptors of images, useful for downstream tasks such as (interactive) segmentation and object detection. For computational efficiency these descriptors are often patch-based, and so struggle to represent the fine features often present in micrographs; they also struggle with the large image sizes present in materials and biological image analysis. In this work, we train a convolutional neural network to upsample low-resolution (i.e, large patch size) foundation model features with reference to the input image. We apply this upsampler network (without any further training) to efficiently featurise and then segment a variety of microscopy images, including plant cells, a lithium-ion battery cathode and organic crystals. The richness of these upsampled features admits separation of hard to segment phases, like hairline cracks. We demonstrate that interactive segmentation with these deep features produces high-quality segmentations far faster and with far fewer labels than training or finetuning a more traditional convolutional network.
>
---
#### [new 020] MedShift: Implicit Conditional Transport for X-Ray Domain Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MedShift模型，解决合成与真实X光图像的跨域翻译问题，通过流匹配和薛定谔桥构建共享潜在空间，提升领域适应性。**

- **链接: [http://arxiv.org/pdf/2508.21435v1](http://arxiv.org/pdf/2508.21435v1)**

> **作者:** Francisco Caetano; Christiaan Viviers; Peter H. H. de With; Fons van der Sommen
>
> **备注:** Accepted at the ICCV 2025 AIM Workshop
>
> **摘要:** Synthetic medical data offers a scalable solution for training robust models, but significant domain gaps limit its generalizability to real-world clinical settings. This paper addresses the challenge of cross-domain translation between synthetic and real X-ray images of the head, focusing on bridging discrepancies in attenuation behavior, noise characteristics, and soft tissue representation. We propose MedShift, a unified class-conditional generative model based on Flow Matching and Schrodinger Bridges, which enables high-fidelity, unpaired image translation across multiple domains. Unlike prior approaches that require domain-specific training or rely on paired data, MedShift learns a shared domain-agnostic latent space and supports seamless translation between any pair of domains seen during training. We introduce X-DigiSkull, a new dataset comprising aligned synthetic and real skull X-rays under varying radiation doses, to benchmark domain translation models. Experimental results demonstrate that, despite its smaller model size compared to diffusion-based approaches, MedShift offers strong performance and remains flexible at inference time, as it can be tuned to prioritize either perceptual fidelity or structural consistency, making it a scalable and generalizable solution for domain adaptation in medical imaging. The code and dataset are available at https://caetas.github.io/medshift.html
>
---
#### [new 021] Advanced Deep Learning Techniques for Classifying Dental Conditions Using Panoramic X-Ray Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究利用混合CNN-随机森林模型和预训练架构，对全景X光片中的牙科状况进行自动分类，提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2508.21088v1](http://arxiv.org/pdf/2508.21088v1)**

> **作者:** Alireza Golkarieh; Kiana Kiashemshaki; Sajjad Rezvani Boroujeni
>
> **备注:** 14 pages, 8 figures, 8 tables
>
> **摘要:** This study investigates deep learning methods for automated classification of dental conditions in panoramic X-ray images. A dataset of 1,512 radiographs with 11,137 expert-verified annotations across four conditions fillings, cavities, implants, and impacted teeth was used. After preprocessing and class balancing, three approaches were evaluated: a custom convolutional neural network (CNN), hybrid models combining CNN feature extraction with traditional classifiers, and fine-tuned pre-trained architectures. Experiments employed 5 fold cross validation with accuracy, precision, recall, and F1 score as evaluation metrics. The hybrid CNN Random Forest model achieved the highest performance with 85.4% accuracy, surpassing the custom CNN baseline of 74.3%. Among pre-trained models, VGG16 performed best at 82.3% accuracy, followed by Xception and ResNet50. Results show that hybrid models improve discrimination of morphologically similar conditions and provide efficient, reliable performance. These findings suggest that combining CNN-based feature extraction with ensemble classifiers offers a practical path toward automated dental diagnostic support, while also highlighting the need for larger datasets and further clinical validation.
>
---
#### [new 022] Domain Generalization in-the-Wild: Disentangling Classification from Domain-Aware Representations
- **分类: cs.CV; cs.LG**

- **简介: 论文针对领域泛化（DG）评估不足问题，提出CLIP-DCA方法，通过增强领域感知表示并解耦分类，提升基础模型在野外未见过数据上的泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.21769v1](http://arxiv.org/pdf/2508.21769v1)**

> **作者:** Ha Min Son; Zhe Zhao; Shahbaz Rezaei; Xin Liu
>
> **摘要:** Evaluating domain generalization (DG) for foundational models like CLIP is challenging, as web-scale pretraining data potentially covers many existing benchmarks. Consequently, current DG evaluation may neither be sufficiently challenging nor adequately test genuinely unseen data scenarios. To better assess the performance of CLIP on DG in-the-wild, a scenario where CLIP encounters challenging unseen data, we consider two approaches: (1) evaluating on 33 diverse datasets with quantified out-of-distribution (OOD) scores after fine-tuning CLIP on ImageNet, and (2) using unlearning to make CLIP `forget' some domains as an approximation. We observe that CLIP's performance deteriorates significantly on more OOD datasets. To address this, we present CLIP-DCA (Disentangling Classification from enhanced domain Aware representations). Our approach is motivated by the observation that while standard domain invariance losses aim to make representations domain-invariant, this can be harmful to foundation models by forcing the discarding of domain-aware representations beneficial for generalization. We instead hypothesize that enhancing domain awareness is a prerequisite for effective domain-invariant classification in foundation models. CLIP-DCA identifies and enhances domain awareness within CLIP's encoders using a separate domain head and synthetically generated diverse domain data. Simultaneously, it encourages domain-invariant classification through disentanglement from the domain features. CLIP-DCA shows significant improvements within this challenging evaluation compared to existing methods, particularly on datasets that are more OOD.
>
---
#### [new 023] Mapping like a Skeptic: Probabilistic BEV Projection for Online HD Mapping
- **分类: cs.CV**

- **简介: 该论文针对在线高精地图构建中的BEV投影问题，提出基于几何映射与概率机制的改进方法，通过置信度评分优化映射精度并过滤冗余信息，提升长距离感知下的地图生成质量。**

- **链接: [http://arxiv.org/pdf/2508.21689v1](http://arxiv.org/pdf/2508.21689v1)**

> **作者:** Fatih Erdoğan; Merve Rabia Barın; Fatma Güney
>
> **备注:** BMVC 2025. GitHub: https://github.com/Fatih-Erdogan/mapping-like-skeptic
>
> **摘要:** Constructing high-definition (HD) maps from sensory input requires accurately mapping the road elements in image space to the Bird's Eye View (BEV) space. The precision of this mapping directly impacts the quality of the final vectorized HD map. Existing HD mapping approaches outsource the projection to standard mapping techniques, such as attention-based ones. However, these methods struggle with accuracy due to generalization problems, often hallucinating non-existent road elements. Our key idea is to start with a geometric mapping based on camera parameters and adapt it to the scene to extract relevant map information from camera images. To implement this, we propose a novel probabilistic projection mechanism with confidence scores to (i) refine the mapping to better align with the scene and (ii) filter out irrelevant elements that should not influence HD map generation. In addition, we improve temporal processing by using confidence scores to selectively accumulate reliable information over time. Experiments on new splits of the nuScenes and Argoverse2 datasets demonstrate improved performance over state-of-the-art approaches, indicating better generalization. The improvements are particularly pronounced on nuScenes and in the challenging long perception range. Our code and model checkpoints are available at https://github.com/Fatih-Erdogan/mapping-like-skeptic .
>
---
#### [new 024] TMUAD: Enhancing Logical Capabilities in Unified Anomaly Detection Models with a Text Memory Bank
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TMUAD框架，通过文本、图像和片段三级记忆库，结合逻辑与结构分析，解决工业和医疗领域异常检测中正常数据不足的问题，实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2508.21795v1](http://arxiv.org/pdf/2508.21795v1)**

> **作者:** Jiawei Liu; Jiahe Hou; Wei Wang; Jinsong Du; Yang Cong; Huijie Fan
>
> **摘要:** Anomaly detection, which aims to identify anomalies deviating from normal patterns, is challenging due to the limited amount of normal data available. Unlike most existing unified methods that rely on carefully designed image feature extractors and memory banks to capture logical relationships between objects, we introduce a text memory bank to enhance the detection of logical anomalies. Specifically, we propose a Three-Memory framework for Unified structural and logical Anomaly Detection (TMUAD). First, we build a class-level text memory bank for logical anomaly detection by the proposed logic-aware text extractor, which can capture rich logical descriptions of objects from input images. Second, we construct an object-level image memory bank that preserves complete object contours by extracting features from segmented objects. Third, we employ visual encoders to extract patch-level image features for constructing a patch-level memory bank for structural anomaly detection. These three complementary memory banks are used to retrieve and compare normal images that are most similar to the query image, compute anomaly scores at multiple levels, and fuse them into a final anomaly score. By unifying structural and logical anomaly detection through collaborative memory banks, TMUAD achieves state-of-the-art performance across seven publicly available datasets involving industrial and medical domains. The model and code are available at https://github.com/SIA-IDE/TMUAD.
>
---
#### [new 025] Complete Gaussian Splats from a Single Image with Denoising Diffusion Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出基于扩散模型的单图像3D场景重建方法，解决遮挡区域重建问题，通过生成式模型和自监督学习实现完整高斯点云生成。**

- **链接: [http://arxiv.org/pdf/2508.21542v1](http://arxiv.org/pdf/2508.21542v1)**

> **作者:** Ziwei Liao; Mohamed Sayed; Steven L. Waslander; Sara Vicente; Daniyar Turmukhambetov; Michael Firman
>
> **备注:** Main paper: 11 pages; Supplementary materials: 7 pages
>
> **摘要:** Gaussian splatting typically requires dense observations of the scene and can fail to reconstruct occluded and unobserved areas. We propose a latent diffusion model to reconstruct a complete 3D scene with Gaussian splats, including the occluded parts, from only a single image during inference. Completing the unobserved surfaces of a scene is challenging due to the ambiguity of the plausible surfaces. Conventional methods use a regression-based formulation to predict a single "mode" for occluded and out-of-frustum surfaces, leading to blurriness, implausibility, and failure to capture multiple possible explanations. Thus, they often address this problem partially, focusing either on objects isolated from the background, reconstructing only visible surfaces, or failing to extrapolate far from the input views. In contrast, we propose a generative formulation to learn a distribution of 3D representations of Gaussian splats conditioned on a single input image. To address the lack of ground-truth training data, we propose a Variational AutoReconstructor to learn a latent space only from 2D images in a self-supervised manner, over which a diffusion model is trained. Our method generates faithful reconstructions and diverse samples with the ability to complete the occluded surfaces for high-quality 360-degree renderings.
>
---
#### [new 026] Integrating Pathology and CT Imaging for Personalized Recurrence Risk Prediction in Renal Cancer
- **分类: cs.CV**

- **简介: 该论文旨在通过整合术前CT与术后病理切片图像，构建多模态深度学习模型，解决肾癌复发风险预测中传统评分系统（如Leibovich）分辨率不足的问题。研究比较了不同融合策略，证实病理数据对预后判断的关键作用，并探索了更优的融合方法以提升个性化风险预测准确性。**

- **链接: [http://arxiv.org/pdf/2508.21581v1](http://arxiv.org/pdf/2508.21581v1)**

> **作者:** Daniël Boeke; Cedrik Blommestijn; Rebecca N. Wray; Kalina Chupetlovska; Shangqi Gao; Zeyu Gao; Regina G. H. Beets-Tan; Mireia Crispin-Ortuzar; James O. Jones; Wilson Silva; Ines P. Machado
>
> **备注:** 12 pages, 2 figures, 1 table. Accepted at the Multimodal Learning and Fusion Across Scales for Clinical Decision Support (ML-CDS) Workshop, MICCAI 2025. This is the submitted version with authors, affiliations, and acknowledgements included; it has not undergone peer review or revisions. The final version will appear in the Springer Lecture Notes in Computer Science (LNCS) proceedings
>
> **摘要:** Recurrence risk estimation in clear cell renal cell carcinoma (ccRCC) is essential for guiding postoperative surveillance and treatment. The Leibovich score remains widely used for stratifying distant recurrence risk but offers limited patient-level resolution and excludes imaging information. This study evaluates multimodal recurrence prediction by integrating preoperative computed tomography (CT) and postoperative histopathology whole-slide images (WSIs). A modular deep learning framework with pretrained encoders and Cox-based survival modeling was tested across unimodal, late fusion, and intermediate fusion setups. In a real-world ccRCC cohort, WSI-based models consistently outperformed CT-only models, underscoring the prognostic strength of pathology. Intermediate fusion further improved performance, with the best model (TITAN-CONCH with ResNet-18) approaching the adjusted Leibovich score. Random tie-breaking narrowed the gap between the clinical baseline and learned models, suggesting discretization may overstate individualized performance. Using simple embedding concatenation, radiology added value primarily through fusion. These findings demonstrate the feasibility of foundation model-based multimodal integration for personalized ccRCC risk prediction. Future work should explore more expressive fusion strategies, larger multimodal datasets, and general-purpose CT encoders to better match pathology modeling capacity.
>
---
#### [new 027] The Demon is in Ambiguity: Revisiting Situation Recognition with Single Positive Multi-Label Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对情境识别中的动词分类歧义问题，提出单正多标签学习框架（SPMLL）及GE-VerbMLP模型，结合图神经网络与对抗训练，提升多标签识别性能。**

- **链接: [http://arxiv.org/pdf/2508.21816v1](http://arxiv.org/pdf/2508.21816v1)**

> **作者:** Yiming Lin; Yuchen Niu; Shang Wang; Kaizhu Huang; Qiufeng Wang; Xiao-Bo Jin
>
> **备注:** Accepted by ICDM 2025
>
> **摘要:** Context recognition (SR) is a fundamental task in computer vision that aims to extract structured semantic summaries from images by identifying key events and their associated entities. Specifically, given an input image, the model must first classify the main visual events (verb classification), then identify the participating entities and their semantic roles (semantic role labeling), and finally localize these entities in the image (semantic role localization). Existing methods treat verb classification as a single-label problem, but we show through a comprehensive analysis that this formulation fails to address the inherent ambiguity in visual event recognition, as multiple verb categories may reasonably describe the same image. This paper makes three key contributions: First, we reveal through empirical analysis that verb classification is inherently a multi-label problem due to the ubiquitous semantic overlap between verb categories. Second, given the impracticality of fully annotating large-scale datasets with multiple labels, we propose to reformulate verb classification as a single positive multi-label learning (SPMLL) problem - a novel perspective in SR research. Third, we design a comprehensive multi-label evaluation benchmark for SR that is carefully designed to fairly evaluate model performance in a multi-label setting. To address the challenges of SPMLL, we futher develop the Graph Enhanced Verb Multilayer Perceptron (GE-VerbMLP), which combines graph neural networks to capture label correlations and adversarial training to optimize decision boundaries. Extensive experiments on real-world datasets show that our approach achieves more than 3\% MAP improvement while remaining competitive on traditional top-1 and top-5 accuracy metrics.
>
---
#### [new 028] DriveQA: Passing the Driving Knowledge Test
- **分类: cs.CV**

- **简介: 该论文提出DriveQA基准，测试大模型在驾驶规则理解与复杂场景处理能力，揭示其在数值推理、让路规则等任务的不足，并通过微调与预训练提升性能，推动自动驾驶发展。**

- **链接: [http://arxiv.org/pdf/2508.21824v1](http://arxiv.org/pdf/2508.21824v1)**

> **作者:** Maolin Wei; Wanzhou Liu; Eshed Ohn-Bar
>
> **备注:** Accepted by ICCV 2025. Project page: https://driveqaiccv.github.io/
>
> **摘要:** If a Large Language Model (LLM) were to take a driving knowledge test today, would it pass? Beyond standard spatial and visual question-answering (QA) tasks on current autonomous driving benchmarks, driving knowledge tests require a complete understanding of all traffic rules, signage, and right-of-way principles. To pass this test, human drivers must discern various edge cases that rarely appear in real-world datasets. In this work, we present DriveQA, an extensive open-source text and vision-based benchmark that exhaustively covers traffic regulations and scenarios. Through our experiments using DriveQA, we show that (1) state-of-the-art LLMs and Multimodal LLMs (MLLMs) perform well on basic traffic rules but exhibit significant weaknesses in numerical reasoning and complex right-of-way scenarios, traffic sign variations, and spatial layouts, (2) fine-tuning on DriveQA improves accuracy across multiple categories, particularly in regulatory sign recognition and intersection decision-making, (3) controlled variations in DriveQA-V provide insights into model sensitivity to environmental factors such as lighting, perspective, distance, and weather conditions, and (4) pretraining on DriveQA enhances downstream driving task performance, leading to improved results on real-world datasets such as nuScenes and BDD, while also demonstrating that models can internalize text and synthetic traffic knowledge to generalize effectively across downstream QA tasks.
>
---
#### [new 029] Federated Fine-tuning of SAM-Med3D for MRI-based Dementia Classification
- **分类: cs.CV**

- **简介: 该论文研究联邦学习中基础模型（SAM-Med3D）的微调策略，针对MRI数据的痴呆分类任务，系统评估分类头架构、微调策略和聚合方法对模型性能的影响，提出高效部署方案及优化方向。**

- **链接: [http://arxiv.org/pdf/2508.21458v1](http://arxiv.org/pdf/2508.21458v1)**

> **作者:** Kaouther Mouheb; Marawan Elbatel; Janne Papma; Geert Jan Biessels; Jurgen Claassen; Huub Middelkoop; Barbara van Munster; Wiesje van der Flier; Inez Ramakers; Stefan Klein; Esther E. Bron
>
> **备注:** Accepted at the MICCAI 2025 Workshop on Distributed, Collaborative and Federated Learning (DeCAF)
>
> **摘要:** While foundation models (FMs) offer strong potential for AI-based dementia diagnosis, their integration into federated learning (FL) systems remains underexplored. In this benchmarking study, we systematically evaluate the impact of key design choices: classification head architecture, fine-tuning strategy, and aggregation method, on the performance and efficiency of federated FM tuning using brain MRI data. Using a large multi-cohort dataset, we find that the architecture of the classification head substantially influences performance, freezing the FM encoder achieves comparable results to full fine-tuning, and advanced aggregation methods outperform standard federated averaging. Our results offer practical insights for deploying FMs in decentralized clinical settings and highlight trade-offs that should guide future method development.
>
---
#### [new 030] Standardized Multi-Layer Tissue Maps for Enhanced Artificial Intelligence Integration and Search in Large-Scale Whole Slide Image Archives
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出标准化多层组织图谱框架，解决WSI元数据缺乏问题，通过三层结构（来源/组织类型/病理改变）提升AI训练数据筛选效率，实现跨领域病理图像的高效检索与分析。**

- **链接: [http://arxiv.org/pdf/2508.21418v1](http://arxiv.org/pdf/2508.21418v1)**

> **作者:** Gernot Fiala; Markus Plass; Robert Harb; Peter Regitnig; Kristijan Skok; Wael Al Zoughbi; Carmen Zerner; Paul Torke; Michaela Kargl; Heimo Müller; Tomas Brazdil; Matej Gallo; Jaroslav Kubín; Roman Stoklasa; Rudolf Nenutil; Norman Zerbe; Andreas Holzinger; Petr Holub
>
> **摘要:** A Whole Slide Image (WSI) is a high-resolution digital image created by scanning an entire glass slide containing a biological specimen, such as tissue sections or cell samples, at multiple magnifications. These images can be viewed, analyzed, shared digitally, and are used today for Artificial Intelligence (AI) algorithm development. WSIs are used in a variety of fields, including pathology for diagnosing diseases and oncology for cancer research. They are also utilized in neurology, veterinary medicine, hematology, microbiology, dermatology, pharmacology, toxicology, immunology, and forensic science. When assembling cohorts for the training or validation of an AI algorithm, it is essential to know what is present on such a WSI. However, there is currently no standard for this metadata, so such selection has mainly been done through manual inspection, which is not suitable for large collections with several million objects. We propose a general framework to generate a 2D index map for WSI and a profiling mechanism for specific application domains. We demonstrate this approach in the field of clinical pathology, using common syntax and semantics to achieve interoperability between different catalogs. Our approach augments each WSI collection with a detailed tissue map that provides fine-grained information about the WSI content. The tissue map is organized into three layers: source, tissue type, and pathological alterations, with each layer assigning segments of the WSI to specific classes. We illustrate the advantages and applicability of the proposed standard through specific examples in WSI catalogs, Machine Learning (ML), and graph-based WSI representations.
>
---
#### [new 031] Generalizable Object Re-Identification via Visual In-Context Prompting
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VICP框架，结合LLM与视觉基础模型，通过上下文示例作为提示，使模型无需参数调整即可泛化到新类别，解决ReID中泛化差和依赖标注数据的问题，并引入ShopID10K数据集验证效果。**

- **链接: [http://arxiv.org/pdf/2508.21222v1](http://arxiv.org/pdf/2508.21222v1)**

> **作者:** Zhizhong Huang; Xiaoming Liu
>
> **备注:** ICCV 2025
>
> **摘要:** Current object re-identification (ReID) methods train domain-specific models (e.g., for persons or vehicles), which lack generalization and demand costly labeled data for new categories. While self-supervised learning reduces annotation needs by learning instance-wise invariance, it struggles to capture \textit{identity-sensitive} features critical for ReID. This paper proposes Visual In-Context Prompting~(VICP), a novel framework where models trained on seen categories can directly generalize to unseen novel categories using only \textit{in-context examples} as prompts, without requiring parameter adaptation. VICP synergizes LLMs and vision foundation models~(VFM): LLMs infer semantic identity rules from few-shot positive/negative pairs through task-specific prompting, which then guides a VFM (\eg, DINO) to extract ID-discriminative features via \textit{dynamic visual prompts}. By aligning LLM-derived semantic concepts with the VFM's pre-trained prior, VICP enables generalization to novel categories, eliminating the need for dataset-specific retraining. To support evaluation, we introduce ShopID10K, a dataset of 10K object instances from e-commerce platforms, featuring multi-view images and cross-domain testing. Experiments on ShopID10K and diverse ReID benchmarks demonstrate that VICP outperforms baselines by a clear margin on unseen categories. Code is available at https://github.com/Hzzone/VICP.
>
---
#### [new 032] PHD: Personalized 3D Human Body Fitting with Point Diffusion
- **分类: cs.CV**

- **简介: 该论文提出PHD方法，解决个性化3D人体姿态估计中传统方法未结合用户体形导致的准确性不足问题。通过分步校准体形与姿态拟合，利用Point Diffusion Transformer提升精度，数据高效且可集成。**

- **链接: [http://arxiv.org/pdf/2508.21257v1](http://arxiv.org/pdf/2508.21257v1)**

> **作者:** Hsuan-I Ho; Chen Guo; Po-Chen Wu; Ivan Shugurov; Chengcheng Tang; Abhay Mittal; Sizhe An; Manuel Kaufmann; Linguang Zhang
>
> **备注:** ICCV 2025, 19 pages, 18 figures
>
> **摘要:** We introduce PHD, a novel approach for personalized 3D human mesh recovery (HMR) and body fitting that leverages user-specific shape information to improve pose estimation accuracy from videos. Traditional HMR methods are designed to be user-agnostic and optimized for generalization. While these methods often refine poses using constraints derived from the 2D image to improve alignment, this process compromises 3D accuracy by failing to jointly account for person-specific body shapes and the plausibility of 3D poses. In contrast, our pipeline decouples this process by first calibrating the user's body shape and then employing a personalized pose fitting process conditioned on that shape. To achieve this, we develop a body shape-conditioned 3D pose prior, implemented as a Point Diffusion Transformer, which iteratively guides the pose fitting via a Point Distillation Sampling loss. This learned 3D pose prior effectively mitigates errors arising from an over-reliance on 2D constraints. Consequently, our approach improves not only pelvis-aligned pose accuracy but also absolute pose accuracy -- an important metric often overlooked by prior work. Furthermore, our method is highly data-efficient, requiring only synthetic data for training, and serves as a versatile plug-and-play module that can be seamlessly integrated with existing 3D pose estimators to enhance their performance. Project page: https://phd-pose.github.io/
>
---
#### [new 033] Unsupervised Incremental Learning Using Confidence-Based Pseudo-Labels
- **分类: cs.CV**

- **简介: 该论文提出ICPL方法，解决无监督增量学习中未标记数据的类别扩展问题。通过置信度筛选伪标签，实现无需人工标注的增量学习，超越现有方法5%以上准确率，适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2508.21424v1](http://arxiv.org/pdf/2508.21424v1)**

> **作者:** Lucas Rakotoarivony
>
> **备注:** Submitted to WACV 2026
>
> **摘要:** Deep learning models have achieved state-of-the-art performance in many computer vision tasks. However, in real-world scenarios, novel classes that were unseen during training often emerge, requiring models to acquire new knowledge incrementally. Class-Incremental Learning (CIL) methods enable a model to learn novel classes while retaining knowledge of previous classes. However, these methods make the strong assumption that the incremental dataset is fully labeled, which is unrealistic in practice. In this work, we propose an unsupervised Incremental Learning method using Confidence-based Pseudo-labels (ICPL), which replaces human annotations with pseudo-labels, enabling incremental learning from unlabeled datasets. We integrate these pseudo-labels into various CIL methods with confidence-based selection and evaluate performance degradation on CIFAR100 and ImageNet100. Then, we compare our approach to popular Class Incremental Novel Category Discovery (class-iNCD) methods addressing similar challenges. Additionally, we apply our method to fine-grained datasets to demonstrate its real-world practicality and measure its computational complexity to validate its suitability for resource-constrained environments. ICPL achieves competitive results compared to supervised methods and outperforms state-of-the-art class-iNCD methods by more than 5% in final accuracy.
>
---
#### [new 034] One More Glance with Sharp Eyes: Rethinking Lightweight Captioning as a Practical Visual Specialist
- **分类: cs.CV**

- **简介: 论文提出轻量级图像描述模型，解决多模态大模型部署难题及视觉盲点问题，通过改进视觉接地提升性能。**

- **链接: [http://arxiv.org/pdf/2508.21451v1](http://arxiv.org/pdf/2508.21451v1)**

> **作者:** Junha Song; Yongsik Jo; So Yeon Min; Quanting Xie; Taehwan Kim; Yonatan Bisk; Jaegul Choo
>
> **备注:** Project page: https://sites.google.com/view/junha/lightweightcaptioner
>
> **摘要:** Image captioning is fundamental for applications like video instruction systems and exploration robots, yet deploying such models on local devices is challenging due to the high computational demands of multimodal large language models (MLLMs). To address this, we first explore lightweight captioning by implementing a specialist based on a 125M-parameter language model, 56 times smaller than LLaMA-7B, and evaluating its performance on both single-sentence and detailed captioning tasks. Surprisingly, we find that our model can achieve performance comparable to large multimodal generalists, suggesting its potential to serve as a strong visual specialist for on-device applications. While promising, our model also exhibits a limitation: like other MLLMs, it suffers from visual blindness, occasionally resulting in semantic captioning errors. We carry out toy experiments and investigate the underlying causes, where we observe that the problems arise from ineffective attention mechanisms and limited visual representations. To alleviate them, we develop a novel captioning framework, Sharp-Eyed Refinement, which enhances caption quality through improved visual grounding. At its core, our DeepLens extracts detailed visual representations by concentrating on informative regions identified during the initial glance. Our experiments confirm both the advantages of our specialist over prior small captioning models and large generalists and the effectiveness of our framework.
>
---
#### [new 035] EZ-Sort: Efficient Pairwise Comparison via Zero-Shot CLIP-Based Pre-Ordering and Human-in-the-Loop Sorting
- **分类: cs.CV; cs.AI; 68T05, 68T09; I.5.4**

- **简介: 论文提出EZ-Sort，通过零样本CLIP预排序和人类反馈排序，解决高效成对比较问题，显著降低标注成本，保持评分可靠性。**

- **链接: [http://arxiv.org/pdf/2508.21550v1](http://arxiv.org/pdf/2508.21550v1)**

> **作者:** Yujin Park; Haejun Chung; Ikbeom Jang
>
> **备注:** 5 pages, 2 figures, Accepted at CIKM 2025 (ACM International Conference on Information and Knowledge Management)
>
> **摘要:** Pairwise comparison is often favored over absolute rating or ordinal classification in subjective or difficult annotation tasks due to its improved reliability. However, exhaustive comparisons require a massive number of annotations (O(n^2)). Recent work has greatly reduced the annotation burden (O(n log n)) by actively sampling pairwise comparisons using a sorting algorithm. We further improve annotation efficiency by (1) roughly pre-ordering items using the Contrastive Language-Image Pre-training (CLIP) model hierarchically without training, and (2) replacing easy, obvious human comparisons with automated comparisons. The proposed EZ-Sort first produces a CLIP-based zero-shot pre-ordering, then initializes bucket-aware Elo scores, and finally runs an uncertainty-guided human-in-the-loop MergeSort. Validation was conducted using various datasets: face-age estimation (FGNET), historical image chronology (DHCI), and retinal image quality assessment (EyePACS). It showed that EZ-Sort reduced human annotation cost by 90.5% compared to exhaustive pairwise comparisons and by 19.8% compared to prior work (when n = 100), while improving or maintaining inter-rater reliability. These results demonstrate that combining CLIP-based priors with uncertainty-aware sampling yields an efficient and scalable solution for pairwise ranking.
>
---
#### [new 036] 2COOOL: 2nd Workshop on the Challenge Of Out-Of-Label Hazards in Autonomous Driving
- **分类: cs.CV; cs.RO; 68T45 (Machine vision and scene understanding); I.2.10; I.4.8**

- **简介: 该研讨会聚焦自动驾驶中标签外危险问题，旨在解决新场景导致的安全隐患。通过推动分布外检测、视觉-语言模型等技术，促进安全算法与基准测试研究，提升自动驾驶可靠性。**

- **链接: [http://arxiv.org/pdf/2508.21080v1](http://arxiv.org/pdf/2508.21080v1)**

> **作者:** Ali K. AlShami; Ryan Rabinowitz; Maged Shoman; Jianwu Fang; Lukas Picek; Shao-Yuan Lo; Steve Cruz; Khang Nhut Lam; Nachiket Kamod; Lei-Lei Li; Jugal Kalita; Terrance E. Boult
>
> **备注:** 11 pages, 2 figures, Accepted to ICCV 2025 Workshop on Out-of-Label Hazards in Autonomous Driving (2COOOL)
>
> **摘要:** As the computer vision community advances autonomous driving algorithms, integrating vision-based insights with sensor data remains essential for improving perception, decision making, planning, prediction, simulation, and control. Yet we must ask: Why don't we have entirely safe self-driving cars yet? A key part of the answer lies in addressing novel scenarios, one of the most critical barriers to real-world deployment. Our 2COOOL workshop provides a dedicated forum for researchers and industry experts to push the state of the art in novelty handling, including out-of-distribution hazard detection, vision-language models for hazard understanding, new benchmarking and methodologies, and safe autonomous driving practices. The 2nd Workshop on the Challenge of Out-of-Label Hazards in Autonomous Driving (2COOOL) will be held at the International Conference on Computer Vision (ICCV) 2025 in Honolulu, Hawaii, on October 19, 2025. We aim to inspire the development of new algorithms and systems for hazard avoidance, drawing on ideas from anomaly detection, open-set recognition, open-vocabulary modeling, domain adaptation, and related fields. Building on the success of its inaugural edition at the Winter Conference on Applications of Computer Vision (WACV) 2025, the workshop will feature a mix of academic and industry participation.
>
---
#### [new 037] Unsupervised Video Continual Learning via Non-Parametric Deep Embedded Clustering
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出无监督视频持续学习（uVCL），解决无标签和任务边界下的连续学习问题。通过非参数深度嵌入聚类、KDE和迁移学习，动态扩展记忆簇以提升多任务学习性能。**

- **链接: [http://arxiv.org/pdf/2508.21773v1](http://arxiv.org/pdf/2508.21773v1)**

> **作者:** Nattapong Kurpukdee; Adrian G. Bors
>
> **备注:** Accepted to The 36th British Machine Vision Conference (BMVC 2025), Sheffield, UK
>
> **摘要:** We propose a realistic scenario for the unsupervised video learning where neither task boundaries nor labels are provided when learning a succession of tasks. We also provide a non-parametric learning solution for the under-explored problem of unsupervised video continual learning. Videos represent a complex and rich spatio-temporal media information, widely used in many applications, but which have not been sufficiently explored in unsupervised continual learning. Prior studies have only focused on supervised continual learning, relying on the knowledge of labels and task boundaries, while having labeled data is costly and not practical. To address this gap, we study the unsupervised video continual learning (uVCL). uVCL raises more challenges due to the additional computational and memory requirements of processing videos when compared to images. We introduce a general benchmark experimental protocol for uVCL by considering the learning of unstructured video data categories during each task. We propose to use the Kernel Density Estimation (KDE) of deep embedded video features extracted by unsupervised video transformer networks as a non-parametric probabilistic representation of the data. We introduce a novelty detection criterion for the incoming new task data, dynamically enabling the expansion of memory clusters, aiming to capture new knowledge when learning a succession of tasks. We leverage the use of transfer learning from the previous tasks as an initial state for the knowledge transfer to the current learning task. We found that the proposed methodology substantially enhances the performance of the model when successively learning many tasks. We perform in-depth evaluations on three standard video action recognition datasets, including UCF101, HMDB51, and Something-to-Something V2, without using any labels or class boundaries.
>
---
#### [new 038] Radially Distorted Homographies, Revisited
- **分类: cs.CV**

- **简介: 该论文针对几何计算机视觉中的同构估计问题，提出统一方法处理径向失真下的三种配置，开发了更快、更准确的求解器，测试于鱼眼图像基准。**

- **链接: [http://arxiv.org/pdf/2508.21190v1](http://arxiv.org/pdf/2508.21190v1)**

> **作者:** Mårten Wadenbäck; Marcus Valtonen Örnhag; Johan Edstedt
>
> **摘要:** Homographies are among the most prevalent transformations occurring in geometric computer vision and projective geometry, and homography estimation is consequently a crucial step in a wide assortment of computer vision tasks. When working with real images, which are often afflicted with geometric distortions caused by the camera lens, it may be necessary to determine both the homography and the lens distortion-particularly the radial component, called radial distortion-simultaneously to obtain anything resembling useful estimates. When considering a homography with radial distortion between two images, there are three conceptually distinct configurations for the radial distortion; (i) distortion in only one image, (ii) identical distortion in the two images, and (iii) independent distortion in the two images. While these cases have been addressed separately in the past, the present paper provides a novel and unified approach to solve all three cases. We demonstrate how the proposed approach can be used to construct new fast, stable, and accurate minimal solvers for radially distorted homographies. In all three cases, our proposed solvers are faster than the existing state-of-the-art solvers while maintaining similar accuracy. The solvers are tested on well-established benchmarks including images taken with fisheye cameras. The source code for our solvers will be made available in the event our paper is accepted for publication.
>
---
#### [new 039] HCCM: Hierarchical Cross-Granularity Contrastive and Matching Learning for Natural Language-Guided Drones
- **分类: cs.CV**

- **简介: 论文提出HCCM框架，解决自然语言引导无人机任务中视觉-语言理解的细粒度语义缺失与动态环境适应问题。通过区域-全局对比学习与匹配机制，结合动量对比蒸馏，提升目标匹配与导航的鲁棒性与零样本泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.21539v1](http://arxiv.org/pdf/2508.21539v1)**

> **作者:** Hao Ruan; Jinliang Lin; Yingxin Lai; Zhiming Luo; Shaozi Li
>
> **备注:** Accepted by ACM MM'25
>
> **摘要:** Natural Language-Guided Drones (NLGD) provide a novel paradigm for tasks such as target matching and navigation. However, the wide field of view and complex compositional semantics in drone scenarios pose challenges for vision-language understanding. Mainstream Vision-Language Models (VLMs) emphasize global alignment while lacking fine-grained semantics, and existing hierarchical methods depend on precise entity partitioning and strict containment, limiting effectiveness in dynamic environments. To address this, we propose the Hierarchical Cross-Granularity Contrastive and Matching learning (HCCM) framework with two components: (1) Region-Global Image-Text Contrastive Learning (RG-ITC), which avoids precise scene partitioning and captures hierarchical local-to-global semantics by contrasting local visual regions with global text and vice versa; (2) Region-Global Image-Text Matching (RG-ITM), which dispenses with rigid constraints and instead evaluates local semantic consistency within global cross-modal representations, enhancing compositional reasoning. Moreover, drone text descriptions are often incomplete or ambiguous, destabilizing alignment. HCCM introduces a Momentum Contrast and Distillation (MCD) mechanism to improve robustness. Experiments on GeoText-1652 show HCCM achieves state-of-the-art Recall@1 of 28.8% (image retrieval) and 14.7% (text retrieval). On the unseen ERA dataset, HCCM demonstrates strong zero-shot generalization with 39.93% mean recall (mR), outperforming fine-tuned baselines.
>
---
#### [new 040] ELV-Halluc: Benchmarking Semantic Aggregation Hallucinations in Long Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对长视频理解中的语义聚合幻觉（SAH）问题，提出ELV-Halluc基准，系统研究SAH成因并提出缓解策略，包括位置编码优化与DPO方法，通过8K对抗数据验证效果提升。**

- **链接: [http://arxiv.org/pdf/2508.21496v1](http://arxiv.org/pdf/2508.21496v1)**

> **作者:** Hao Lu; Jiahao Wang; Yaolun Zhang; Ruohui Wang; Xuanyu Zheng; Yepeng Tang; Dahua Lin; Lewei Lu
>
> **摘要:** Video multimodal large language models (Video-MLLMs) have achieved remarkable progress in video understanding. However, they remain vulnerable to hallucination-producing content inconsistent with or unrelated to video inputs. Previous video hallucination benchmarks primarily focus on short-videos. They attribute hallucinations to factors such as strong language priors, missing frames, or vision-language biases introduced by the visual encoder. While these causes indeed account for most hallucinations in short videos, they still oversimplify the cause of hallucinations. Sometimes, models generate incorrect outputs but with correct frame-level semantics. We refer to this type of hallucination as Semantic Aggregation Hallucination (SAH), which arises during the process of aggregating frame-level semantics into event-level semantic groups. Given that SAH becomes particularly critical in long videos due to increased semantic complexity across multiple events, it is essential to separate and thoroughly investigate the causes of this type of hallucination. To address the above issues, we introduce ELV-Halluc, the first benchmark dedicated to long-video hallucination, enabling a systematic investigation of SAH. Our experiments confirm the existence of SAH and show that it increases with semantic complexity. Additionally, we find that models are more prone to SAH on rapidly changing semantics. Moreover, we discuss potential approaches to mitigate SAH. We demonstrate that positional encoding strategy contributes to alleviating SAH, and further adopt DPO strategy to enhance the model's ability to distinguish semantics within and across events. To support this, we curate a dataset of 8K adversarial data pairs and achieve improvements on both ELV-Halluc and Video-MME, including a substantial 27.7% reduction in SAH ratio.
>
---
#### [new 041] Safe-Control: A Safety Patch for Mitigating Unsafe Content in Text-to-Image Generation Models
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 本研究提出Safe-Control，一种插件式安全补丁，用于减少文本到图像生成模型中的不安全内容。通过数据驱动策略注入安全信号，有效降低不安全生成概率（7% vs. 20%），兼容多种模型架构，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.21099v1](http://arxiv.org/pdf/2508.21099v1)**

> **作者:** Xiangtao Meng; Yingkai Dong; Ning Yu; Li Wang; Zheng Li; Shanqing Guo
>
> **摘要:** Despite the advancements in Text-to-Image (T2I) generation models, their potential for misuse or even abuse raises serious safety concerns. Model developers have made tremendous efforts to introduce safety mechanisms that can address these concerns in T2I models. However, the existing safety mechanisms, whether external or internal, either remain susceptible to evasion under distribution shifts or require extensive model-specific adjustments. To address these limitations, we introduce Safe-Control, an innovative plug-and-play safety patch designed to mitigate unsafe content generation in T2I models. Using data-driven strategies and safety-aware conditions, Safe-Control injects safety control signals into the locked T2I model, acting as an update in a patch-like manner. Model developers can also construct various safety patches to meet the evolving safety requirements, which can be flexibly merged into a single, unified patch. Its plug-and-play design further ensures adaptability, making it compatible with other T2I models of similar denoising architecture. We conduct extensive evaluations on six diverse and public T2I models. Empirical results highlight that Safe-Control is effective in reducing unsafe content generation across six diverse T2I models with similar generative architectures, yet it successfully maintains the quality and text alignment of benign images. Compared to seven state-of-the-art safety mechanisms, including both external and internal defenses, Safe-Control significantly outperforms all baselines in reducing unsafe content generation. For example, it reduces the probability of unsafe content generation to 7%, compared to approximately 20% for most baseline methods, under both unsafe prompts and the latest adversarial attacks.
>
---
#### [new 042] Entropy-Based Non-Invasive Reliability Monitoring of Convolutional Neural Networks
- **分类: cs.CV; cs.AI; cs.CR; cs.IT; eess.IV; math.IT**

- **简介: 该论文提出基于激活熵的非侵入式对抗样本检测方法，无需修改模型即可实时监控CNN可靠性，通过熵变化区分清洁与对抗输入，实现高准确率检测。**

- **链接: [http://arxiv.org/pdf/2508.21715v1](http://arxiv.org/pdf/2508.21715v1)**

> **作者:** Amirhossein Nazeri; Wael Hafez
>
> **备注:** 8 pages, 3 figures, 2 tables
>
> **摘要:** Convolutional Neural Networks (CNNs) have become the foundation of modern computer vision, achieving unprecedented accuracy across diverse image recognition tasks. While these networks excel on in-distribution data, they remain vulnerable to adversarial perturbations imperceptible input modifications that cause misclassification with high confidence. However, existing detection methods either require expensive retraining, modify network architecture, or degrade performance on clean inputs. Here we show that adversarial perturbations create immediate, detectable entropy signatures in CNN activations that can be monitored without any model modification. Using parallel entropy monitoring on VGG-16, we demonstrate that adversarial inputs consistently shift activation entropy by 7% in early convolutional layers, enabling 90% detection accuracy with false positives and false negative rates below 20%. The complete separation between clean and adversarial entropy distributions reveals that CNNs inherently encode distribution shifts in their activation patterns. This work establishes that CNN reliability can be assessed through activation entropy alone, enabling practical deployment of self-diagnostic vision systems that detect adversarial inputs in real-time without compromising original model performance.
>
---
#### [new 043] FLORA: Efficient Synthetic Data Generation for Object Detection in Low-Data Regimes via finetuning Flux LoRA
- **分类: cs.CV**

- **简介: 该论文提出FLORA方法，解决目标检测中合成数据生成资源密集、需大量图像的问题。通过LoRA微调Flux模型，显著降低计算成本，仅用500张合成图像即超越基准方法（5000张），mAP提升21.3%，实现高效高质量数据生成。**

- **链接: [http://arxiv.org/pdf/2508.21712v1](http://arxiv.org/pdf/2508.21712v1)**

> **作者:** Alvaro Patricio; Atabak Dehban; Rodrigo Ventura
>
> **摘要:** Recent advances in diffusion-based generative models have demonstrated significant potential in augmenting scarce datasets for object detection tasks. Nevertheless, most recent models rely on resource-intensive full fine-tuning of large-scale diffusion models, requiring enterprise-grade GPUs (e.g., NVIDIA V100) and thousands of synthetic images. To address these limitations, we propose Flux LoRA Augmentation (FLORA), a lightweight synthetic data generation pipeline. Our approach uses the Flux 1.1 Dev diffusion model, fine-tuned exclusively through Low-Rank Adaptation (LoRA). This dramatically reduces computational requirements, enabling synthetic dataset generation with a consumer-grade GPU (e.g., NVIDIA RTX 4090). We empirically evaluate our approach on seven diverse object detection datasets. Our results demonstrate that training object detectors with just 500 synthetic images generated by our approach yields superior detection performance compared to models trained on 5000 synthetic images from the ODGEN baseline, achieving improvements of up to 21.3% in mAP@.50:.95. This work demonstrates that it is possible to surpass state-of-the-art performance with far greater efficiency, as FLORA achieves superior results using only 10% of the data and a fraction of the computational cost. This work demonstrates that a quality and efficiency-focused approach is more effective than brute-force generation, making advanced synthetic data creation more practical and accessible for real-world scenarios.
>
---
#### [new 044] SYNBUILD-3D: A large, multi-modal, and semantically rich synthetic dataset of 3D building models at Level of Detail 4
- **分类: cs.CV**

- **简介: 该论文构建多模态合成数据集SYNBUILD-3D，包含620万LoD4建筑模型，涵盖3D线框、平面图和点云，解决自动建模语义与几何一致性难题，推动生成AI算法发展。**

- **链接: [http://arxiv.org/pdf/2508.21169v1](http://arxiv.org/pdf/2508.21169v1)**

> **作者:** Kevin Mayer; Alex Vesel; Xinyi Zhao; Martin Fischer
>
> **摘要:** 3D building models are critical for applications in architecture, energy simulation, and navigation. Yet, generating accurate and semantically rich 3D buildings automatically remains a major challenge due to the lack of large-scale annotated datasets in the public domain. Inspired by the success of synthetic data in computer vision, we introduce SYNBUILD-3D, a large, diverse, and multi-modal dataset of over 6.2 million synthetic 3D residential buildings at Level of Detail (LoD) 4. In the dataset, each building is represented through three distinct modalities: a semantically enriched 3D wireframe graph at LoD 4 (Modality I), the corresponding floor plan images (Modality II), and a LiDAR-like roof point cloud (Modality III). The semantic annotations for each building wireframe are derived from the corresponding floor plan images and include information on rooms, doors, and windows. Through its tri-modal nature, future work can use SYNBUILD-3D to develop novel generative AI algorithms that automate the creation of 3D building models at LoD 4, subject to predefined floor plan layouts and roof geometries, while enforcing semantic-geometric consistency. Dataset and code samples are publicly available at https://github.com/kdmayer/SYNBUILD-3D.
>
---
#### [new 045] Trees as Gaussians: Large-Scale Individual Tree Mapping
- **分类: cs.CV**

- **简介: 该论文提出一种深度学习方法，利用高斯核模拟树冠，结合多源数据训练模型，实现全球高分辨率单株树木检测与映射，提升生态监测精度。**

- **链接: [http://arxiv.org/pdf/2508.21437v1](http://arxiv.org/pdf/2508.21437v1)**

> **作者:** Dimitri Gominski; Martin Brandt; Xiaoye Tong; Siyu Liu; Maurice Mugabowindekwe; Sizhuo Li; Florian Reiner; Andrew Davies; Rasmus Fensholt
>
> **摘要:** Trees are key components of the terrestrial biosphere, playing vital roles in ecosystem function, climate regulation, and the bioeconomy. However, large-scale monitoring of individual trees remains limited by inadequate modelling. Available global products have focused on binary tree cover or canopy height, which do not explicitely identify trees at individual level. In this study, we present a deep learning approach for detecting large individual trees in 3-m resolution PlanetScope imagery at a global scale. We simulate tree crowns with Gaussian kernels of scalable size, allowing the extraction of crown centers and the generation of binary tree cover maps. Training is based on billions of points automatically extracted from airborne lidar data, enabling the model to successfully identify trees both inside and outside forests. We compare against existing tree cover maps and airborne lidar with state-of-the-art performance (fractional cover R$^2 = 0.81$ against aerial lidar), report balanced detection metrics across biomes, and demonstrate how detection can be further improved through fine-tuning with manual labels. Our method offers a scalable framework for global, high-resolution tree monitoring, and is adaptable to future satellite missions offering improved imagery.
>
---
#### [new 046] RadGS-Reg: Registering Spine CT with Biplanar X-rays via Joint 3D Radiative Gaussians Reconstruction and 3D/3D Registration
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出RadGS-Reg框架，解决脊柱CT与双平面X光的高精度实时配准问题。通过联合3D Radiative Gaussians重建与3D/3D配准，结合Counterfactual Attention Learning机制和患者预训练策略，克服传统方法的空间信息损失及噪声干扰，提升配准性能。**

- **链接: [http://arxiv.org/pdf/2508.21154v1](http://arxiv.org/pdf/2508.21154v1)**

> **作者:** Ao Shen; Xueming Fu; Junfeng Jiang; Qiang Zeng; Ye Tang; Zhengming Chen; Luming Nong; Feng Wang; S. Kevin Zhou
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** Computed Tomography (CT)/X-ray registration in image-guided navigation remains challenging because of its stringent requirements for high accuracy and real-time performance. Traditional "render and compare" methods, relying on iterative projection and comparison, suffer from spatial information loss and domain gap. 3D reconstruction from biplanar X-rays supplements spatial and shape information for 2D/3D registration, but current methods are limited by dense-view requirements and struggles with noisy X-rays. To address these limitations, we introduce RadGS-Reg, a novel framework for vertebral-level CT/X-ray registration through joint 3D Radiative Gaussians (RadGS) reconstruction and 3D/3D registration. Specifically, our biplanar X-rays vertebral RadGS reconstruction module explores learning-based RadGS reconstruction method with a Counterfactual Attention Learning (CAL) mechanism, focusing on vertebral regions in noisy X-rays. Additionally, a patient-specific pre-training strategy progressively adapts the RadGS-Reg from simulated to real data while simultaneously learning vertebral shape prior knowledge. Experiments on in-house datasets demonstrate the state-of-the-art performance for both tasks, surpassing existing methods. The code is available at: https://github.com/shenao1995/RadGS_Reg.
>
---
#### [new 047] GCAV: A Global Concept Activation Vector Framework for Cross-Layer Consistency in Interpretability
- **分类: cs.CV**

- **简介: 该论文提出GCAV框架，解决深度学习模型跨层概念激活向量（CAV）不一致问题。通过对比学习对齐层间概念表示，并结合注意力机制融合，提升概念解释的稳定性与可靠性，增强模型可解释性。**

- **链接: [http://arxiv.org/pdf/2508.21197v1](http://arxiv.org/pdf/2508.21197v1)**

> **作者:** Zhenghao He; Sanchit Sinha; Guangzhi Xiong; Aidong Zhang
>
> **摘要:** Concept Activation Vectors (CAVs) provide a powerful approach for interpreting deep neural networks by quantifying their sensitivity to human-defined concepts. However, when computed independently at different layers, CAVs often exhibit inconsistencies, making cross-layer comparisons unreliable. To address this issue, we propose the Global Concept Activation Vector (GCAV), a novel framework that unifies CAVs into a single, semantically consistent representation. Our method leverages contrastive learning to align concept representations across layers and employs an attention-based fusion mechanism to construct a globally integrated CAV. By doing so, our method significantly reduces the variance in TCAV scores while preserving concept relevance, ensuring more stable and reliable concept attributions. To evaluate the effectiveness of GCAV, we introduce Testing with Global Concept Activation Vectors (TGCAV) as a method to apply TCAV to GCAV-based representations. We conduct extensive experiments on multiple deep neural networks, demonstrating that our method effectively mitigates concept inconsistency across layers, enhances concept localization, and improves robustness against adversarial perturbations. By integrating cross-layer information into a coherent framework, our method offers a more comprehensive and interpretable understanding of how deep learning models encode human-defined concepts. Code and models are available at https://github.com/Zhenghao-He/GCAV.
>
---
#### [new 048] Towards Interactive Lesion Segmentation in Whole-Body PET/CT with Promptable Models
- **分类: cs.CV**

- **简介: 论文提出交互式病灶分割方法，解决全身体PET/CT中多中心、示踪剂异质性导致的分割难题。基于nnU-Net框架，引入用户提示输入，采用EDT编码提升提示表示，并模拟用户交互优化分割，有效减少假阳/假阴。**

- **链接: [http://arxiv.org/pdf/2508.21680v1](http://arxiv.org/pdf/2508.21680v1)**

> **作者:** Maximilian Rokuss; Yannick Kirchhoff; Fabian Isensee; Klaus H. Maier-Hein
>
> **备注:** atuoPET4 Team LesionLocator
>
> **摘要:** Whole-body PET/CT is a cornerstone of oncological imaging, yet accurate lesion segmentation remains challenging due to tracer heterogeneity, physiological uptake, and multi-center variability. While fully automated methods have advanced substantially, clinical practice benefits from approaches that keep humans in the loop to efficiently refine predicted masks. The autoPET/CT IV challenge addresses this need by introducing interactive segmentation tasks based on simulated user prompts. In this work, we present our submission to Task 1. Building on the winning autoPET III nnU-Net pipeline, we extend the framework with promptable capabilities by encoding user-provided foreground and background clicks as additional input channels. We systematically investigate representations for spatial prompts and demonstrate that Euclidean Distance Transform (EDT) encodings consistently outperform Gaussian kernels. Furthermore, we propose online simulation of user interactions and a custom point sampling strategy to improve robustness under realistic prompting conditions. Our ensemble of EDT-based models, trained with and without external data, achieves the strongest cross-validation performance, reducing both false positives and false negatives compared to baseline models. These results highlight the potential of promptable models to enable efficient, user-guided segmentation workflows in multi-tracer, multi-center PET/CT. Code is publicly available at https://github.com/MIC-DKFZ/autoPET-interactive
>
---
#### [new 049] Multi-Method Ensemble for Out-of-Distribution Detection
- **分类: cs.CV**

- **简介: 论文提出多方法集成框架，结合特征截断与得分函数，提升OOD检测性能，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.21463v1](http://arxiv.org/pdf/2508.21463v1)**

> **作者:** Lucas Rakotoarivony
>
> **备注:** Accepted paper for BMVC 2025
>
> **摘要:** Detecting out-of-distribution (OOD) samples is essential for neural networks operating in open-world settings, particularly in safety-critical applications. Existing methods have improved OOD detection by leveraging two main techniques: feature truncation, which increases the separation between in-distribution (ID) and OOD samples, and scoring functions, which assign scores to distinguish between ID and OOD data. However, most approaches either focus on a single family of techniques or evaluate their effectiveness on a specific type of OOD dataset, overlooking the potential of combining multiple existing solutions. Motivated by this observation, we theoretically and empirically demonstrate that state-of-the-art feature truncation and scoring functions can be effectively combined. Moreover, we show that aggregating multiple scoring functions enhances robustness against various types of OOD samples. Based on these insights, we propose the Multi-Method Ensemble (MME) score, which unifies state-of-the-art OOD detectors into a single, more effective scoring function. Extensive experiments on both large-scale and small-scale benchmarks, covering near-OOD and far-OOD scenarios, show that MME significantly outperforms recent state-of-the-art methods across all benchmarks. Notably, using the BiT model, our method achieves an average FPR95 of 27.57% on the challenging ImageNet-1K benchmark, improving performance by 6% over the best existing baseline.
>
---
#### [new 050] Scale-GS: Efficient Scalable Gaussian Splatting via Redundancy-filtering Training on Streaming Content
- **分类: cs.CV**

- **简介: 论文提出Scale-GS框架，通过分层高斯结构与混合变形策略，解决动态场景3DGS训练效率低的问题，提升实时渲染性能。**

- **链接: [http://arxiv.org/pdf/2508.21444v1](http://arxiv.org/pdf/2508.21444v1)**

> **作者:** Jiayu Yang; Weijian Su; Songqian Zhang; Yuqi Han; Jinli Suo; Qiang Zhang
>
> **摘要:** 3D Gaussian Splatting (3DGS) enables high-fidelity real-time rendering, a key requirement for immersive applications. However, the extension of 3DGS to dynamic scenes remains limitations on the substantial data volume of dense Gaussians and the prolonged training time required for each frame. This paper presents \M, a scalable Gaussian Splatting framework designed for efficient training in streaming tasks. Specifically, Gaussian spheres are hierarchically organized by scale within an anchor-based structure. Coarser-level Gaussians represent the low-resolution structure of the scene, while finer-level Gaussians, responsible for detailed high-fidelity rendering, are selectively activated by the coarser-level Gaussians. To further reduce computational overhead, we introduce a hybrid deformation and spawning strategy that models motion of inter-frame through Gaussian deformation and triggers Gaussian spawning to characterize wide-range motion. Additionally, a bidirectional adaptive masking mechanism enhances training efficiency by removing static regions and prioritizing informative viewpoints. Extensive experiments demonstrate that \M~ achieves superior visual quality while significantly reducing training time compared to state-of-the-art methods.
>
---
#### [new 051] Why Stop at Words? Unveiling the Bigger Picture through Line-Level OCR
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出行级OCR方法，解决传统字符分割错误和上下文不足问题，构建数据集并验证，提升准确率5.4%和效率4倍。**

- **链接: [http://arxiv.org/pdf/2508.21693v1](http://arxiv.org/pdf/2508.21693v1)**

> **作者:** Shashank Vempati; Nishit Anand; Gaurav Talebailkar; Arpan Garai; Chetan Arora
>
> **备注:** 11 pages. Project Website: https://nishitanand.github.io/line-level-ocr-website
>
> **摘要:** Conventional optical character recognition (OCR) techniques segmented each character and then recognized. This made them prone to error in character segmentation, and devoid of context to exploit language models. Advances in sequence to sequence translation in last decade led to modern techniques first detecting words and then inputting one word at a time to a model to directly output full words as sequence of characters. This allowed better utilization of language models and bypass error-prone character segmentation step. We observe that the above transition in style has moved the bottleneck in accuracy to word segmentation. Hence, in this paper, we propose a natural and logical progression from word level OCR to line-level OCR. The proposal allows to bypass errors in word detection, and provides larger sentence context for better utilization of language models. We show that the proposed technique not only improves the accuracy but also efficiency of OCR. Despite our thorough literature survey, we did not find any public dataset to train and benchmark such shift from word to line-level OCR. Hence, we also contribute a meticulously curated dataset of 251 English page images with line-level annotations. Our experimentation revealed a notable end-to-end accuracy improvement of 5.4%, underscoring the potential benefits of transitioning towards line-level OCR, especially for document images. We also report a 4 times improvement in efficiency compared to word-based pipelines. With continuous improvements in large language models, our methodology also holds potential to exploit such advances. Project Website: https://nishitanand.github.io/line-level-ocr-website
>
---
#### [new 052] Efficient Diffusion-Based 3D Human Pose Estimation with Hierarchical Temporal Pruning
- **分类: cs.CV**

- **简介: 该论文提出一种基于扩散模型的高效3D人体姿态估计框架，通过分层时间剪枝策略降低计算成本。针对扩散模型迭代计算量大的问题，设计三级剪枝模块，动态去除冗余姿态标记，提升推理速度并保持高精度，在Human3.6M和MPI-INF-3DHP数据集上实现效率与性能的平衡。**

- **链接: [http://arxiv.org/pdf/2508.21363v1](http://arxiv.org/pdf/2508.21363v1)**

> **作者:** Yuquan Bi; Hongsong Wang; Xinli Shi; Zhipeng Gui; Jie Gui; Yuan Yan Tang
>
> **摘要:** Diffusion models have demonstrated strong capabilities in generating high-fidelity 3D human poses, yet their iterative nature and multi-hypothesis requirements incur substantial computational cost. In this paper, we propose an Efficient Diffusion-Based 3D Human Pose Estimation framework with a Hierarchical Temporal Pruning (HTP) strategy, which dynamically prunes redundant pose tokens across both frame and semantic levels while preserving critical motion dynamics. HTP operates in a staged, top-down manner: (1) Temporal Correlation-Enhanced Pruning (TCEP) identifies essential frames by analyzing inter-frame motion correlations through adaptive temporal graph construction; (2) Sparse-Focused Temporal MHSA (SFT MHSA) leverages the resulting frame-level sparsity to reduce attention computation, focusing on motion-relevant tokens; and (3) Mask-Guided Pose Token Pruner (MGPTP) performs fine-grained semantic pruning via clustering, retaining only the most informative pose tokens. Experiments on Human3.6M and MPI-INF-3DHP show that HTP reduces training MACs by 38.5\%, inference MACs by 56.8\%, and improves inference speed by an average of 81.1\% compared to prior diffusion-based methods, while achieving state-of-the-art performance.
>
---
#### [new 053] Temporal Flow Matching for Learning Spatio-Temporal Trajectories in 4D Longitudinal Medical Imaging
- **分类: cs.CV**

- **简介: 该论文提出Temporal Flow Matching（TFM），解决4D医学影像时空轨迹预测中现有方法局限问题，支持3D体积、多先验扫描及不规则采样，建立新SOTA基准。**

- **链接: [http://arxiv.org/pdf/2508.21580v1](http://arxiv.org/pdf/2508.21580v1)**

> **作者:** Nico Albert Disch; Yannick Kirchhoff; Robin Peretzke; Maximilian Rokuss; Saikat Roy; Constantin Ulrich; David Zimmerer; Klaus Maier-Hein
>
> **摘要:** Understanding temporal dynamics in medical imaging is crucial for applications such as disease progression modeling, treatment planning and anatomical development tracking. However, most deep learning methods either consider only single temporal contexts, or focus on tasks like classification or regression, limiting their ability for fine-grained spatial predictions. While some approaches have been explored, they are often limited to single timepoints, specific diseases or have other technical restrictions. To address this fundamental gap, we introduce Temporal Flow Matching (TFM), a unified generative trajectory method that (i) aims to learn the underlying temporal distribution, (ii) by design can fall back to a nearest image predictor, i.e. predicting the last context image (LCI), as a special case, and (iii) supports $3D$ volumes, multiple prior scans, and irregular sampling. Extensive benchmarks on three public longitudinal datasets show that TFM consistently surpasses spatio-temporal methods from natural imaging, establishing a new state-of-the-art and robust baseline for $4D$ medical image prediction.
>
---
#### [new 054] VoCap: Video Object Captioning and Segmentation from Any Prompt
- **分类: cs.CV**

- **简介: 该论文提出VoCap模型，通过多模态提示实现视频对象分割与描述，解决数据稀缺问题，生成伪标签数据集并取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2508.21809v1](http://arxiv.org/pdf/2508.21809v1)**

> **作者:** Jasper Uijlings; Xingyi Zhou; Xiuye Gu; Arsha Nagrani; Anurag Arnab; Alireza Fathi; David Ross; Cordelia Schmid
>
> **摘要:** Understanding objects in videos in terms of fine-grained localization masks and detailed semantic properties is a fundamental task in video understanding. In this paper, we propose VoCap, a flexible video model that consumes a video and a prompt of various modalities (text, box or mask), and produces a spatio-temporal masklet with a corresponding object-centric caption. As such our model addresses simultaneously the tasks of promptable video object segmentation, referring expression segmentation, and object captioning. Since obtaining data for this task is tedious and expensive, we propose to annotate an existing large-scale segmentation dataset (SAV) with pseudo object captions. We do so by preprocessing videos with their ground-truth masks to highlight the object of interest and feed this to a large Vision Language Model (VLM). For an unbiased evaluation, we collect manual annotations on the validation set. We call the resulting dataset SAV-Caption. We train our VoCap model at scale on a SAV-Caption together with a mix of other image and video datasets. Our model yields state-of-the-art results on referring expression video object segmentation, is competitive on semi-supervised video object segmentation, and establishes a benchmark for video object captioning. Our dataset will be made available at https://github.com/google-deepmind/vocap.
>
---
#### [new 055] Video-LLMs with Temporal Visual Screening
- **分类: cs.CV**

- **简介: 该论文提出TVS任务，解决Video-LLMs在时间语义理解和帧间推理上的不足。通过关键片段筛选、查询重构和答案一致性保持，设计预处理模块，开发ReSimplifyIt基线模型，提升视频语言理解效果。**

- **链接: [http://arxiv.org/pdf/2508.21094v1](http://arxiv.org/pdf/2508.21094v1)**

> **作者:** Zheyu Fan; Jiateng Liu; Yuji Zhang; Zihan Wang; Yi R.; Fung; Manling Li; Heng Ji
>
> **摘要:** Humans naturally perform temporal screening by dragging the progress bar and focusing on salient temporal segments, but current Video Large Language Models (Video-LLMs) struggle to capture fine-grained temporal semantics due to sparse frame sampling and insufficient inter-frame reasoning supervision during their training. To address this, Inspired by well-established cognitive science principles, we propose Temporal Visual Screening (TVS), a new task that universally pre-processes video question answering and instruction tuning data by: (1) retaining focus-critical video segments, (2) synchronously reconstructing queries to their most direct form while preserving answer consistency, and (3) keeping the invariance and consistency for any possible answer. TVS is formulated as a modular front-end adapter task that can be seamlessly integrated into both Video Instruction Tuning (training) and Video Question Answering (inference) pipelines. TVS optimizes distribution of reasoning burden and cognitive load; during training, it aligns queries with focus-critical visual information; at inference, it enables query-aware segment focus and streamlined query representations. In particular, we curate the first benchmark for TVS and propose ReSimplifyIt, a baseline outperforming prior approaches on seemingly similar tasks by 0.47 in F-1 score on video trimming while achieving competitive query rewriting performance. Experiments demonstrate that incorporating TVS yields relative gains of 7.33% (training) and 34.6% (inference), demonstrating the effectiveness of temporal information screening for improving video-language understanding.
>
---
#### [new 056] Lightweight MRI-Based Automated Segmentation of Pancreatic Cancer with Auto3DSeg
- **分类: cs.CV; 68T07, 68U10; I.4.6; I.5.4; J.3**

- **简介: 论文针对MRI胰腺肿瘤自动分割任务，解决解剖变异与数据不足挑战，采用SegResNet和Auto3DSeg架构，通过五折交叉验证与STAPLE集成，评估DSC等指标。结果显示小数据集下性能差异，强调需更大标准化数据提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.21227v1](http://arxiv.org/pdf/2508.21227v1)**

> **作者:** Keshav Jha; William Sharp; Dominic LaBella
>
> **备注:** 11 pages, 3 figures, 3 tables, MICCAI
>
> **摘要:** Accurate delineation of pancreatic tumors is critical for diagnosis, treatment planning, and outcome assessment, yet automated segmentation remains challenging due to anatomical variability and limited dataset availability. In this study, SegResNet models, as part of the Auto3DSeg architecture, were trained and evaluated on two MRI-based pancreatic tumor segmentation tasks as part of the 2025 PANTHER Challenge. Algorithm methodology included 5-fold cross-validation with STAPLE ensembling after focusing on an anatomically relevant region-of-interest. The Pancreatic Tumor Segmentation on Diagnostic MRI task 1 training set included 91 T1-weighted arterial contrast-enhanced MRI with expert annotated pancreas and tumor labels. The Pancreatic Tumor Segmentation on MR-Linac task 2 training set used 50 T2-weighted MR-Linac cases with expert annotated pancreas and tumor labels. Algorithm-automated segmentation performance of pancreatic tumor was assessed using Dice Similarity Coefficient (DSC), 5 mm DSC, 95th percentile Hausdorff Distance (HD95), Mean Average Surface Distance (MASD), and Root Mean Square Error (RMSE). For Task 1, the algorithm achieved a DSC of 0.56, 5 mm DSC of 0.73, HD95 of 41.1 mm, MASD of 26.0 mm, and RMSE of 5164 mm. For Task 2, performance decreased, with a DSC of 0.33, 5 mm DSC of 0.50, HD95 of 20.1 mm, MASD of 7.2 mm, and RMSE of 17,203 mm. These findings illustrate the challenges of MRI-based pancreatic tumor segmentation with small datasets, highlighting variability introduced by different MRI sequences. Despite modest performance, the results demonstrate potential for automated delineation and emphasize the need for larger, standardized MRI datasets to improve model robustness and clinical utility.
>
---
#### [new 057] Reverse Imaging for Wide-spectrum Generalization of Cardiac MRI Segmentation
- **分类: cs.CV**

- **简介: 该论文提出Reverse Imaging方法，解决心脏MRI分割在不同成像序列下的泛化问题。通过逆向推断自旋属性并结合生成模型，实现跨协议和对比度的高精度分割。**

- **链接: [http://arxiv.org/pdf/2508.21254v1](http://arxiv.org/pdf/2508.21254v1)**

> **作者:** Yidong Zhao; Peter Kellman; Hui Xue; Tongyun Yang; Yi Zhang; Yuchi Han; Orlando Simonetti; Qian Tao
>
> **摘要:** Pretrained segmentation models for cardiac magnetic resonance imaging (MRI) struggle to generalize across different imaging sequences due to significant variations in image contrast. These variations arise from changes in imaging protocols, yet the same fundamental spin properties, including proton density, T1, and T2 values, govern all acquired images. With this core principle, we introduce Reverse Imaging, a novel physics-driven method for cardiac MRI data augmentation and domain adaptation to fundamentally solve the generalization problem. Our method reversely infers the underlying spin properties from observed cardiac MRI images, by solving ill-posed nonlinear inverse problems regularized by the prior distribution of spin properties. We acquire this "spin prior" by learning a generative diffusion model from the multiparametric SAturation-recovery single-SHot acquisition sequence (mSASHA) dataset, which offers joint cardiac T1 and T2 maps. Our method enables approximate but meaningful spin-property estimates from MR images, which provide an interpretable "latent variable" that lead to highly flexible image synthesis of arbitrary novel sequences. We show that Reverse Imaging enables highly accurate segmentation across vastly different image contrasts and imaging protocols, realizing wide-spectrum generalization of cardiac MRI segmentation.
>
---
#### [new 058] Identifying Surgical Instruments in Laparoscopy Using Deep Learning Instance Segmentation
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对腹腔镜手术视频中的器械分割与识别问题，采用深度学习实例分割方法，评估了在有限训练样本下的性能。结果表明，分割准确度较高，但器械类型识别仍具挑战，因器械间高度相似。**

- **链接: [http://arxiv.org/pdf/2508.21399v1](http://arxiv.org/pdf/2508.21399v1)**

> **作者:** Sabrina Kletz; Klaus Schoeffmann; Jenny Benois-Pineau; Heinrich Husslein
>
> **摘要:** Recorded videos from surgeries have become an increasingly important information source for the field of medical endoscopy, since the recorded footage shows every single detail of the surgery. However, while video recording is straightforward these days, automatic content indexing - the basis for content-based search in a medical video archive - is still a great challenge due to the very special video content. In this work, we investigate segmentation and recognition of surgical instruments in videos recorded from laparoscopic gynecology. More precisely, we evaluate the achievable performance of segmenting surgical instruments from their background by using a region-based fully convolutional network for instance-aware (1) instrument segmentation as well as (2) instrument recognition. While the first part addresses only binary segmentation of instances (i.e., distinguishing between instrument or background) we also investigate multi-class instrument recognition (i.e., identifying the type of instrument). Our evaluation results show that even with a moderately low number of training examples, we are able to localize and segment instrument regions with a pretty high accuracy. However, the results also reveal that determining the particular instrument is still very challenging, due to the inherently high similarity of surgical instruments.
>
---
#### [new 059] A Multi-Stage Fine-Tuning and Ensembling Strategy for Pancreatic Tumor Segmentation in Diagnostic and Therapeutic MRI
- **分类: cs.CV**

- **简介: 该论文针对MRI胰腺肿瘤分割任务，解决对比度差与数据稀缺问题，提出基于nnU-Net的多阶段预训练与集成策略，通过数据增强和混合专家模型提升分割精度，达到任务1 Dice 0.661，任务2 0.523。**

- **链接: [http://arxiv.org/pdf/2508.21775v1](http://arxiv.org/pdf/2508.21775v1)**

> **作者:** Omer Faruk Durugol; Maximilian Rokuss; Yannick Kirchhoff; Klaus H. Maier-Hein
>
> **备注:** 11 pages, 1 figure, PANTHER Challenge submission
>
> **摘要:** Automated segmentation of Pancreatic Ductal Adenocarcinoma (PDAC) from MRI is critical for clinical workflows but is hindered by poor tumor-tissue contrast and a scarcity of annotated data. This paper details our submission to the PANTHER challenge, addressing both diagnostic T1-weighted (Task 1) and therapeutic T2-weighted (Task 2) segmentation. Our approach is built upon the nnU-Net framework and leverages a deep, multi-stage cascaded pre-training strategy, starting from a general anatomical foundation model and sequentially fine-tuning on CT pancreatic lesion datasets and the target MRI modalities. Through extensive five-fold cross-validation, we systematically evaluated data augmentation schemes and training schedules. Our analysis revealed a critical trade-off, where aggressive data augmentation produced the highest volumetric accuracy, while default augmentations yielded superior boundary precision (achieving a state-of-the-art MASD of 5.46 mm and HD95 of 17.33 mm for Task 1). For our final submission, we exploited this finding by constructing custom, heterogeneous ensembles of specialist models, essentially creating a mix of experts. This metric-aware ensembling strategy proved highly effective, achieving a top cross-validation Tumor Dice score of 0.661 for Task 1 and 0.523 for Task 2. Our work presents a robust methodology for developing specialized, high-performance models in the context of limited data and complex medical imaging tasks (Team MIC-DKFZ).
>
---
#### [new 060] Can Multimodal LLMs Solve the Basic Perception Problems of Percept-V?
- **分类: cs.CL; cs.CV**

- **简介: 该论文评估多模态大语言模型（MLLMs）在基础视觉感知任务中的表现，解决其在简单感知任务上的性能评估问题。作者构建了Percept-V数据集，测试GPT-4o、Gemini等模型，发现模型性能随任务复杂度显著下降。**

- **链接: [http://arxiv.org/pdf/2508.21143v1](http://arxiv.org/pdf/2508.21143v1)**

> **作者:** Samrajnee Ghosh; Naman Agarwal; Hemanshu Garg; Chinmay Mittal; Mausam; Parag Singla
>
> **摘要:** The reasoning abilities of Multimodal Large Language Models (MLLMs) have garnered a lot of attention in recent times, with advances made in frontiers like coding, mathematics, and science. However, very limited experiments have been done to assess their performance in simple perception tasks performed over uncontaminated, generated images containing basic shapes and structures. To address this issue, the paper introduces a dataset, Percept-V, containing a total of 7200 program-generated images equally divided into 30 categories, each testing a combination of visual perception skills. Unlike previously proposed datasets, Percept-V comprises very basic tasks of varying complexity that test the perception abilities of MLLMs. This dataset is then tested on state-of-the-art MLLMs like GPT-4o, Gemini, and Claude as well as Large Reasoning Models (LRMs) like OpenAI o4-mini and DeepSeek R1 to gauge their performance. Contrary to the evidence that MLLMs excel in many complex tasks, our experiments show a significant drop in the models' performance with increasing problem complexity across all categories. An analysis of the performances also reveals that the tested MLLMs exhibit a similar trend in accuracy across categories, testing a particular cognitive skill and find some skills to be more difficult than others.
>
---
#### [new 061] From Drone Imagery to Livability Mapping: AI-powered Environment Perception in Rural China
- **分类: cs.CY; cs.CV**

- **简介: 该论文提出基于无人机影像与多模态大语言模型的农村宜居性评估框架，解决传统问卷与城市视觉方法在农村应用的局限，分析中国农村宜居性空间异质性及财政支出等核心影响因素，为乡村振兴政策提供数据支持。**

- **链接: [http://arxiv.org/pdf/2508.21738v1](http://arxiv.org/pdf/2508.21738v1)**

> **作者:** Weihuan Deng; Yaofu Huang; Luan Chen; Xun Li; Yao Yao
>
> **摘要:** With the deepening of poverty alleviation and rural revitalization strategies, improving the rural living environment and enhancing the quality of life have become key priorities. Rural livability is a key indicator for measuring the effectiveness of these efforts. Current measurement approaches face significant limitations, as questionnaire-based methods are difficult to scale, while urban-oriented visual perception methods are poorly suited for rural contexts. In this paper, a rural-specific livability assessment framework was proposed based on drone imagery and multimodal large language models (MLLMs). To comprehensively assess village livability, this study first used a top-down approach to collect large-scale drone imagery of 1,766 villages in 146 counties across China. In terms of the model framework, an efficient image comparison mechanism was developed, incorporating binary search interpolation to determine effective image pairs while reducing comparison iterations. Building on expert knowledge, a chain-of-thought prompting suitable for nationwide rural livability measurement was constructed, considering both living quality and ecological habitability dimensions. This approach enhanced the rationality and reliability of the livability assessment. Finally, this study characterized the spatial heterogeneity of rural livability across China and thoroughly analyzed its influential factors. The results show that: (1) The rural livability in China demonstrates a dual-core-periphery spatial pattern, radiating outward from Sichuan and Zhejiang provinces with declining gradients; (2) Among various influential factors, government fiscal expenditure emerged as the core determinant, with each unit increase corresponding to a 3.9 - 4.9 unit enhancement in livability. The findings provide valuable insights for rural construction policy-making.
>
---
#### [new 062] Mini Autonomous Car Driving based on 3D Convolutional Neural Networks
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对微型自动驾驶汽车控制任务，提出基于RGB-D数据与3D卷积神经网络的方法，解决传统模型训练复杂、效率低的问题，通过模拟环境对比实验验证其在任务完成率与驾驶一致性上的优越性。**

- **链接: [http://arxiv.org/pdf/2508.21271v1](http://arxiv.org/pdf/2508.21271v1)**

> **作者:** Pablo Moraes; Monica Rodriguez; Kristofer S. Kappel; Hiago Sodre; Santiago Fernandez; Igor Nunes; Bruna Guterres; Ricardo Grando
>
> **摘要:** Autonomous driving applications have become increasingly relevant in the automotive industry due to their potential to enhance vehicle safety, efficiency, and user experience, thereby meeting the growing demand for sophisticated driving assistance features. However, the development of reliable and trustworthy autonomous systems poses challenges such as high complexity, prolonged training periods, and intrinsic levels of uncertainty. Mini Autonomous Cars (MACs) are used as a practical testbed, enabling validation of autonomous control methodologies on small-scale setups. This simplified and cost-effective environment facilitates rapid evaluation and comparison of machine learning models, which is particularly useful for algorithms requiring online training. To address these challenges, this work presents a methodology based on RGB-D information and three-dimensional convolutional neural networks (3D CNNs) for MAC autonomous driving in simulated environments. We evaluate the proposed approach against recurrent neural networks (RNNs), with architectures trained and tested on two simulated tracks with distinct environmental features. Performance was assessed using task completion success, lap-time metrics, and driving consistency. Results highlight how architectural modifications and track complexity influence the models' generalization capability and vehicle control performance. The proposed 3D CNN demonstrated promising results when compared with RNNs.
>
---
#### [new 063] ARGS: Advanced Regularization on Aligning Gaussians over the Surface
- **分类: cs.GR; cs.CV**

- **简介: 该论文针对3D高斯溅射中高斯形状各向异性和表面不一致问题，提出两种正则化方法：秩正则化抑制针状形状，神经SDF与Eikonal损失提升全局一致性，从而提高重建质量和视觉保真度。**

- **链接: [http://arxiv.org/pdf/2508.21344v1](http://arxiv.org/pdf/2508.21344v1)**

> **作者:** Jeong Uk Lee; Sung Hee Choi
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Reconstructing high-quality 3D meshes and visuals from 3D Gaussian Splatting(3DGS) still remains a central challenge in computer graphics. Although existing models such as SuGaR offer effective solutions for rendering, there is is still room to improve improve both visual fidelity and scene consistency. This work builds upon SuGaR by introducing two complementary regularization strategies that address common limitations in both the shape of individual Gaussians and the coherence of the overall surface. The first strategy introduces an effective rank regularization, motivated by recent studies on Gaussian primitive structures. This regularization discourages extreme anisotropy-specifically, "needle-like" shapes-by favoring more balanced, "disk-like" forms that are better suited for stable surface reconstruction. The second strategy integrates a neural Signed Distance Function (SDF) into the optimization process. The SDF is regularized with an Eikonal loss to maintain proper distance properties and provides a continuous global surface prior, guiding Gaussians toward better alignment with the underlying geometry. These two regularizations aim to improve both the fidelity of individual Gaussian primitives and their collective surface behavior. The final model can make more accurate and coherent visuals from 3DGS data.
>
---
#### [new 064] Learning Unified Representations from Heterogeneous Data for Robust Heart Rate Modeling
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对心率预测中的数据异构性问题，提出统一表示学习框架，通过随机特征丢弃和时间注意力模块处理设备与用户差异，结合对比学习提升鲁棒性，并发布ParroTao数据集验证效果。**

- **链接: [http://arxiv.org/pdf/2508.21785v1](http://arxiv.org/pdf/2508.21785v1)**

> **作者:** Peng Yang; Zhengdong Huang; Zicheng Xie; Wentao Tian; Jingyu Liu; Lunhong Dong
>
> **摘要:** Heart rate prediction is vital for personalized health monitoring and fitness, while it frequently faces a critical challenge when deploying in real-world: data heterogeneity. We classify it in two key dimensions: source heterogeneity from fragmented device markets with varying feature sets, and user heterogeneity reflecting distinct physiological patterns across individuals and activities. Existing methods either discard device-specific information, or fail to model user-specific differences, limiting their real-world performance. To address this, we propose a framework that learns latent representations agnostic to both heterogeneity, enabling downstream predictors to work consistently under heterogeneous data patterns. Specifically, we introduce a random feature dropout strategy to handle source heterogeneity, making the model robust to various feature sets. To manage user heterogeneity, we employ a time-aware attention module to capture long-term physiological traits and use a contrastive learning objective to build a discriminative representation space. To reflect the heterogeneous nature of real-world data, we created and publicly released a new benchmark dataset, ParroTao. Evaluations on both ParroTao and the public FitRec dataset show that our model significantly outperforms existing baselines by 17% and 15%, respectively. Furthermore, analysis of the learned representations demonstrates their strong discriminative power, and one downstream application task confirm the practical value of our model.
>
---
#### [new 065] Morae: Proactively Pausing UI Agents for User Choices
- **分类: cs.HC; cs.CL; cs.CV**

- **简介: 该论文提出Morae系统，解决现有UI代理缺乏用户决策参与的问题。通过自动识别任务中的决策点并暂停，结合大模型解析用户意图与界面信息，让用户明确选择，提升任务完成率与偏好匹配度。**

- **链接: [http://arxiv.org/pdf/2508.21456v1](http://arxiv.org/pdf/2508.21456v1)**

> **作者:** Yi-Hao Peng; Dingzeyu Li; Jeffrey P. Bigham; Amy Pavel
>
> **备注:** ACM UIST 2025
>
> **摘要:** User interface (UI) agents promise to make inaccessible or complex UIs easier to access for blind and low-vision (BLV) users. However, current UI agents typically perform tasks end-to-end without involving users in critical choices or making them aware of important contextual information, thus reducing user agency. For example, in our field study, a BLV participant asked to buy the cheapest available sparkling water, and the agent automatically chose one from several equally priced options, without mentioning alternative products with different flavors or better ratings. To address this problem, we introduce Morae, a UI agent that automatically identifies decision points during task execution and pauses so that users can make choices. Morae uses large multimodal models to interpret user queries alongside UI code and screenshots, and prompt users for clarification when there is a choice to be made. In a study over real-world web tasks with BLV participants, Morae helped users complete more tasks and select options that better matched their preferences, as compared to baseline agents, including OpenAI Operator. More broadly, this work exemplifies a mixed-initiative approach in which users benefit from the automation of UI agents while being able to express their preferences.
>
---
#### [new 066] The Rosario Dataset v2: Multimodal Dataset for Agricultural Robotics
- **分类: cs.RO; cs.CV; cs.SY; eess.SY; I.2.9**

- **简介: 该论文提出农业机器人多模态数据集Rosario v2，解决自然光照、地形复杂等挑战，用于评估SLAM算法，发布数据集及工具。**

- **链接: [http://arxiv.org/pdf/2508.21635v1](http://arxiv.org/pdf/2508.21635v1)**

> **作者:** Nicolas Soncini; Javier Cremona; Erica Vidal; Maximiliano García; Gastón Castro; Taihú Pire
>
> **备注:** First published on The International Journal of Robotics Research: https://journals.sagepub.com/doi/10.1177/02783649251368909
>
> **摘要:** We present a multi-modal dataset collected in a soybean crop field, comprising over two hours of recorded data from sensors such as stereo infrared camera, color camera, accelerometer, gyroscope, magnetometer, GNSS (Single Point Positioning, Real-Time Kinematic and Post-Processed Kinematic), and wheel odometry. This dataset captures key challenges inherent to robotics in agricultural environments, including variations in natural lighting, motion blur, rough terrain, and long, perceptually aliased sequences. By addressing these complexities, the dataset aims to support the development and benchmarking of advanced algorithms for localization, mapping, perception, and navigation in agricultural robotics. The platform and data collection system is designed to meet the key requirements for evaluating multi-modal SLAM systems, including hardware synchronization of sensors, 6-DOF ground truth and loops on long trajectories. We run multimodal state-of-the art SLAM methods on the dataset, showcasing the existing limitations in their application on agricultural settings. The dataset and utilities to work with it are released on https://cifasis.github.io/rosariov2/.
>
---
#### [new 067] Is this chart lying to me? Automating the detection of misleading visualizations
- **分类: cs.CL; cs.CV; cs.GR**

- **简介: 论文提出Misviz和Misviz-synth数据集，用于自动检测误导性图表，解决数据不足问题，评估多模型性能，发现任务挑战性，推动该领域发展。**

- **链接: [http://arxiv.org/pdf/2508.21675v1](http://arxiv.org/pdf/2508.21675v1)**

> **作者:** Jonathan Tonglet; Jan Zimny; Tinne Tuytelaars; Iryna Gurevych
>
> **备注:** Preprint under review. Code and data available at: https://github.com/UKPLab/arxiv2025-misviz
>
> **摘要:** Misleading visualizations are a potent driver of misinformation on social media and the web. By violating chart design principles, they distort data and lead readers to draw inaccurate conclusions. Prior work has shown that both humans and multimodal large language models (MLLMs) are frequently deceived by such visualizations. Automatically detecting misleading visualizations and identifying the specific design rules they violate could help protect readers and reduce the spread of misinformation. However, the training and evaluation of AI models has been limited by the absence of large, diverse, and openly available datasets. In this work, we introduce Misviz, a benchmark of 2,604 real-world visualizations annotated with 12 types of misleaders. To support model training, we also release Misviz-synth, a synthetic dataset of 81,814 visualizations generated using Matplotlib and based on real-world data tables. We perform a comprehensive evaluation on both datasets using state-of-the-art MLLMs, rule-based systems, and fine-tuned classifiers. Our results reveal that the task remains highly challenging. We release Misviz, Misviz-synth, and the accompanying code.
>
---
#### [new 068] Med-RewardBench: Benchmarking Reward Models and Judges for Medical Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出Med-RewardBench基准，专门评估医疗多模态大模型的奖励模型与法官，解决现有基准忽视临床准确性与相关性的不足，通过专家标注数据与多维度评估，分析32个模型表现并开发改进基线模型。**

- **链接: [http://arxiv.org/pdf/2508.21430v1](http://arxiv.org/pdf/2508.21430v1)**

> **作者:** Meidan Ding; Jipeng Zhang; Wenxuan Wang; Cheng-Yi Li; Wei-Chieh Fang; Hsin-Yu Wu; Haiqin Zhong; Wenting Chen; Linlin Shen
>
> **备注:** 19 pages, 5 figures, 3 tables
>
> **摘要:** Multimodal large language models (MLLMs) hold significant potential in medical applications, including disease diagnosis and clinical decision-making. However, these tasks require highly accurate, context-sensitive, and professionally aligned responses, making reliable reward models and judges critical. Despite their importance, medical reward models (MRMs) and judges remain underexplored, with no dedicated benchmarks addressing clinical requirements. Existing benchmarks focus on general MLLM capabilities or evaluate models as solvers, neglecting essential evaluation dimensions like diagnostic accuracy and clinical relevance. To address this, we introduce Med-RewardBench, the first benchmark specifically designed to evaluate MRMs and judges in medical scenarios. Med-RewardBench features a multimodal dataset spanning 13 organ systems and 8 clinical departments, with 1,026 expert-annotated cases. A rigorous three-step process ensures high-quality evaluation data across six clinically critical dimensions. We evaluate 32 state-of-the-art MLLMs, including open-source, proprietary, and medical-specific models, revealing substantial challenges in aligning outputs with expert judgment. Additionally, we develop baseline models that demonstrate substantial performance improvements through fine-tuning.
>
---
#### [new 069] ScanMove: Motion Prediction and Transfer for Unregistered Body Meshes
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出ScanMove框架，解决未注册3D网格的运动预测与传输问题，通过结合运动嵌入网络与顶点特征场生成时空变形场，实现鲁棒的变形计算，适用于行走、跑步等任务。**

- **链接: [http://arxiv.org/pdf/2508.21095v1](http://arxiv.org/pdf/2508.21095v1)**

> **作者:** Thomas Besnier; Sylvain Arguillère; Mohamed Daoudi
>
> **摘要:** Unregistered surface meshes, especially raw 3D scans, present significant challenges for automatic computation of plausible deformations due to the lack of established point-wise correspondences and the presence of noise in the data. In this paper, we propose a new, rig-free, data-driven framework for motion prediction and transfer on such body meshes. Our method couples a robust motion embedding network with a learned per-vertex feature field to generate a spatio-temporal deformation field, which drives the mesh deformation. Extensive evaluations, including quantitative benchmarks and qualitative visuals on tasks such as walking and running, demonstrate the effectiveness and versatility of our approach on challenging unregistered meshes.
>
---
#### [new 070] QuadKAN: KAN-Enhanced Quadruped Motion Control via End-to-End Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.IV; eess.SY**

- **简介: 论文提出QuadKAN框架，用于四足视觉运动控制，结合本体感觉与视觉输入，通过KAN和样条参数化策略提升样本效率与稳定性，采用MMDR+PPO训练，在复杂地形中实现更优运动表现。**

- **链接: [http://arxiv.org/pdf/2508.19153v1](http://arxiv.org/pdf/2508.19153v1)**

> **作者:** Allen Wang; Gavin Tao
>
> **备注:** 14pages, 9 figures, Journal paper
>
> **摘要:** We address vision-guided quadruped motion control with reinforcement learning (RL) and highlight the necessity of combining proprioception with vision for robust control. We propose QuadKAN, a spline-parameterized cross-modal policy instantiated with Kolmogorov-Arnold Networks (KANs). The framework incorporates a spline encoder for proprioception and a spline fusion head for proprioception-vision inputs. This structured function class aligns the state-to-action mapping with the piecewise-smooth nature of gait, improving sample efficiency, reducing action jitter and energy consumption, and providing interpretable posture-action sensitivities. We adopt Multi-Modal Delay Randomization (MMDR) and perform end-to-end training with Proximal Policy Optimization (PPO). Evaluations across diverse terrains, including both even and uneven surfaces and scenarios with static or dynamic obstacles, demonstrate that QuadKAN achieves consistently higher returns, greater distances, and fewer collisions than state-of-the-art (SOTA) baselines. These results show that spline-parameterized policies offer a simple, effective, and interpretable alternative for robust vision-guided locomotion. A repository will be made available upon acceptance.
>
---
#### [new 071] Activation Subspaces for Out-of-Distribution Detection
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出ActSub方法，通过奇异值分解分析模型激活子空间，区分ID与OOD数据。针对大/小分布偏移场景，分别利用不显著子空间和决定性子空间，实现高效OOD检测，在多个基准测试中达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2508.21695v1](http://arxiv.org/pdf/2508.21695v1)**

> **作者:** Barış Zöngür; Robin Hesse; Stefan Roth
>
> **摘要:** To ensure the reliability of deep models in real-world applications, out-of-distribution (OOD) detection methods aim to distinguish samples close to the training distribution (in-distribution, ID) from those farther away (OOD). In this work, we propose a novel OOD detection method that utilizes singular value decomposition of the weight matrix of the classification head to decompose the model's activations into decisive and insignificant components, which contribute maximally, respectively minimally, to the final classifier output. We find that the subspace of insignificant components more effectively distinguishes ID from OOD data than raw activations in regimes of large distribution shifts (Far-OOD). This occurs because the classification objective leaves the insignificant subspace largely unaffected, yielding features that are ''untainted'' by the target classification task. Conversely, in regimes of smaller distribution shifts (Near-OOD), we find that activation shaping methods profit from only considering the decisive subspace, as the insignificant component can cause interference in the activation space. By combining two findings into a single approach, termed ActSub, we achieve state-of-the-art results in various standard OOD benchmarks.
>
---
## 更新

#### [replaced 001] Explicit Residual-Based Scalable Image Coding for Humans and Machines
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19297v2](http://arxiv.org/pdf/2506.19297v2)**

> **作者:** Yui Tatsumi; Ziyue Zeng; Hiroshi Watanabe
>
> **备注:** Accepted to IEEE 27th International Workshop on Multimedia Signal Processing (MMSP 2025)
>
> **摘要:** Scalable image compression is a technique that progressively reconstructs multiple versions of an image for different requirements. In recent years, images have increasingly been consumed not only by humans but also by image recognition models. This shift has drawn growing attention to scalable image compression methods that serve both machine and human vision (ICMH). Many existing models employ neural network-based codecs, known as learned image compression, and have made significant strides in this field by carefully designing the loss functions. In some cases, however, models are overly reliant on their learning capacity, and their architectural design is not sufficiently considered. In this paper, we enhance the coding efficiency and interpretability of ICMH framework by integrating an explicit residual compression mechanism, which is commonly employed in resolution scalable coding methods such as JPEG2000. Specifically, we propose two complementary methods: Feature Residual-based Scalable Coding (FR-ICMH) and Pixel Residual-based Scalable Coding (PR-ICMH). These proposed methods are applicable to various machine vision tasks. Moreover, they provide flexibility to choose between encoder complexity and compression performance, making it adaptable to diverse application requirements. Experimental results demonstrate the effectiveness of our proposed methods, with PR-ICMH achieving up to 29.57% BD-rate savings over the previous work.
>
---
#### [replaced 002] Gaussian is All You Need: A Unified Framework for Solving Inverse Problems via Diffusion Posterior Sampling
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.08906v3](http://arxiv.org/pdf/2409.08906v3)**

> **作者:** Nebiyou Yismaw; Ulugbek S. Kamilov; M. Salman Asif
>
> **摘要:** Diffusion models can generate a variety of high-quality images by modeling complex data distributions. Trained diffusion models can also be very effective image priors for solving inverse problems. Most of the existing diffusion-based methods integrate data consistency steps by approximating the likelihood function within the diffusion reverse sampling process. In this paper, we show that the existing approximations are either insufficient or computationally inefficient. To address these issues, we propose a unified likelihood approximation method that incorporates a covariance correction term to enhance the performance and avoids propagating gradients through the diffusion model. The correction term, when integrated into the reverse diffusion sampling process, achieves better convergence towards the true data posterior for selected distributions and improves performance on real-world natural image datasets. Furthermore, we present an efficient way to factorize and invert the covariance matrix of the likelihood function for several inverse problems. Our comprehensive experiments demonstrate the effectiveness of our method over several existing approaches. Code available at https://github.com/CSIPlab/CoDPS.
>
---
#### [replaced 003] DocR1: Evidence Page-Guided GRPO for Multi-Page Document Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.07313v2](http://arxiv.org/pdf/2508.07313v2)**

> **作者:** Junyu Xiong; Yonghui Wang; Weichao Zhao; Chenyu Liu; Bing Yin; Wengang Zhou; Houqiang Li
>
> **摘要:** Understanding multi-page documents poses a significant challenge for multimodal large language models (MLLMs), as it requires fine-grained visual comprehension and multi-hop reasoning across pages. While prior work has explored reinforcement learning (RL) for enhancing advanced reasoning in MLLMs, its application to multi-page document understanding remains underexplored. In this paper, we introduce DocR1, an MLLM trained with a novel RL framework, Evidence Page-Guided GRPO (EviGRPO). EviGRPO incorporates an evidence-aware reward mechanism that promotes a coarse-to-fine reasoning strategy, guiding the model to first retrieve relevant pages before generating answers. This training paradigm enables us to build high-quality models with limited supervision. To support this, we design a two-stage annotation pipeline and a curriculum learning strategy, based on which we construct two datasets: EviBench, a high-quality training set with 4.8k examples, and ArxivFullQA, an evaluation benchmark with 8.6k QA pairs based on scientific papers. Extensive experiments across a wide range of benchmarks demonstrate that DocR1 achieves state-of-the-art performance on multi-page tasks, while consistently maintaining strong results on single-page benchmarks.
>
---
#### [replaced 004] Bringing Attention to CAD: Boundary Representation Learning via Transformer
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.07134v2](http://arxiv.org/pdf/2504.07134v2)**

> **作者:** Qiang Zou; Lizhen Zhu
>
> **摘要:** The recent rise of generative artificial intelligence (AI), powered by Transformer networks, has achieved remarkable success in natural language processing, computer vision, and graphics. However, the application of Transformers in computer-aided design (CAD), particularly for processing boundary representation (B-rep) models, remains largely unexplored. To bridge this gap, we propose a novel approach for adapting Transformers to B-rep learning, called the Boundary Representation Transformer (BRT). B-rep models pose unique challenges due to their irregular topology and continuous geometric definitions, which are fundamentally different from the structured and discrete data Transformers are designed for. To address this, BRT proposes a continuous geometric embedding method that encodes B-rep surfaces (trimmed and untrimmed) into Bezier triangles, preserving their shape and continuity without discretization. Additionally, BRT employs a topology-aware embedding method that organizes these geometric embeddings into a sequence of discrete tokens suitable for Transformers, capturing both geometric and topological characteristics within B-rep models. This enables the Transformer's attention mechanism to effectively learn shape patterns and contextual semantics of boundary elements in a B-rep model. Extensive experiments demonstrate that BRT achieves state-of-the-art performance in part classification and feature recognition tasks.
>
---
#### [replaced 005] WebInject: Prompt Injection Attack to Web Agents
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11717v3](http://arxiv.org/pdf/2505.11717v3)**

> **作者:** Xilong Wang; John Bloch; Zedian Shao; Yuepeng Hu; Shuyan Zhou; Neil Zhenqiang Gong
>
> **备注:** EMNLP 2025 main
>
> **摘要:** Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.
>
---
#### [replaced 006] From stability of Langevin diffusion to convergence of proximal MCMC for non-log-concave sampling
- **分类: stat.ML; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14177v2](http://arxiv.org/pdf/2505.14177v2)**

> **作者:** Marien Renaud; Valentin De Bortoli; Arthur Leclaire; Nicolas Papadakis
>
> **摘要:** We consider the problem of sampling distributions stemming from non-convex potentials with Unadjusted Langevin Algorithm (ULA). We prove the stability of the discrete-time ULA to drift approximations under the assumption that the potential is strongly convex at infinity. In many context, e.g. imaging inverse problems, potentials are non-convex and non-smooth. Proximal Stochastic Gradient Langevin Algorithm (PSGLA) is a popular algorithm to handle such potentials. It combines the forward-backward optimization algorithm with a ULA step. Our main stability result combined with properties of the Moreau envelope allows us to derive the first proof of convergence of the PSGLA for non-convex potentials. We empirically validate our methodology on synthetic data and in the context of imaging inverse problems. In particular, we observe that PSGLA exhibits faster convergence rates than Stochastic Gradient Langevin Algorithm for posterior sampling while preserving its restoration properties.
>
---
#### [replaced 007] PlantVillageVQA: A Visual Question Answering Dataset for Benchmarking Vision-Language Models in Plant Science
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.17117v2](http://arxiv.org/pdf/2508.17117v2)**

> **作者:** Syed Nazmus Sakib; Nafiul Haque; Mohammad Zabed Hossain; Shifat E. Arman
>
> **备注:** 17 pages, 15 figures and Submittd to Nature Scientific Data
>
> **摘要:** PlantVillageVQA is a large-scale visual question answering (VQA) dataset derived from the widely used PlantVillage image corpus. It was designed to advance the development and evaluation of vision-language models for agricultural decision-making and analysis. The PlantVillageVQA dataset comprises 193,609 high-quality question-answer (QA) pairs grounded over 55,448 images spanning 14 crop species and 38 disease conditions. Questions are organised into 3 levels of cognitive complexity and 9 distinct categories. Each question category was phrased manually following expert guidance and generated via an automated two-stage pipeline: (1) template-based QA synthesis from image metadata and (2) multi-stage linguistic re-engineering. The dataset was iteratively reviewed by domain experts for scientific accuracy and relevancy. The final dataset was evaluated using three state-of-the-art models for quality assessment. Our objective remains to provide a publicly available, standardised and expert-verified database to enhance diagnostic accuracy for plant disease identifications and advance scientific research in the agricultural domain. Our dataset will be open-sourced at https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA.
>
---
#### [replaced 008] Saliency-Guided Training for Fingerprint Presentation Attack Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02176v2](http://arxiv.org/pdf/2505.02176v2)**

> **作者:** Samuel Webster; Adam Czajka
>
> **备注:** 17 pages (8 main, 2 references, 7 appendix), 2 figures, 19 tables (2 main, 17 appendix); updated to camera-ready version for IJCB 2025, results unchanged
>
> **摘要:** Saliency-guided training, which directs model learning to important regions of images, has demonstrated generalization improvements across various biometric presentation attack detection (PAD) tasks. This paper presents its first application to fingerprint PAD. We conducted a 50-participant study to create a dataset of 800 human-annotated fingerprint perceptually-important maps, explored alongside algorithmically-generated "pseudosaliency," including minutiae-based, image quality-based, and autoencoder-based saliency maps. Evaluating on the 2021 Fingerprint Liveness Detection Competition testing set, we explore various configurations within five distinct training scenarios to assess the impact of saliency-guided training on accuracy and generalization. Our findings demonstrate the effectiveness of saliency-guided training for fingerprint PAD in both limited and large data contexts, and we present a configuration capable of earning the first place on the LivDet-2021 benchmark. Our results highlight saliency-guided training's promise for increased model generalization capabilities, its effectiveness when data is limited, and its potential to scale to larger datasets in fingerprint PAD. All collected saliency data and trained models are released with the paper to support reproducible research.
>
---
#### [replaced 009] Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14156v3](http://arxiv.org/pdf/2502.14156v3)**

> **作者:** Katie Z Luo; Minh-Quan Dao; Zhenzhen Liu; Mark Campbell; Wei-Lun Chao; Kilian Q. Weinberger; Ezio Malis; Vincent Fremont; Bharath Hariharan; Mao Shan; Stewart Worrall; Julie Stephany Berrio Perez
>
> **备注:** International Conference on Computer Vision, ICCV 2025
>
> **摘要:** Vehicle-to-everything (V2X) collaborative perception has emerged as a promising solution to address the limitations of single-vehicle perception systems. However, existing V2X datasets are limited in scope, diversity, and quality. To address these gaps, we present Mixed Signals, a comprehensive V2X dataset featuring 45.1k point clouds and 240.6k bounding boxes collected from three connected autonomous vehicles (CAVs) equipped with two different configurations of LiDAR sensors, plus a roadside unit with dual LiDARs. Our dataset provides point clouds and bounding box annotations across 10 classes, ensuring reliable data for perception training. We provide detailed statistical analysis on the quality of our dataset and extensively benchmark existing V2X methods on it. The Mixed Signals dataset is ready-to-use, with precise alignment and consistent annotations across time and viewpoints. Dataset website is available at https://mixedsignalsdataset.cs.cornell.edu/.
>
---
#### [replaced 010] CHaRM: Conditioned Heatmap Regression Methodology for Accurate and Fast Dental Landmark Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.13073v5](http://arxiv.org/pdf/2501.13073v5)**

> **作者:** José Rodríguez-Ortega; Francisco Pérez-Hernández; Siham Tabik
>
> **摘要:** Identifying anatomical landmarks in 3D dental models is essential for orthodontic treatment, yet manual placement is labor-intensive and requires expert knowledge. While machine learning methods have been proposed for automatic landmark detection in 3D Intraoral Scans (IOS), none provide a fully end-to-end solution that avoids costly tooth segmentation. We present CHaRM (Conditioned Heatmap Regression Methodology), the first fully end-to-end deep learning approach for tooth landmark detection in 3D IOS. CHaRM integrates four components: a point cloud encoder, a decoder with a heatmap regression head, a teeth-presence classification head, and the novel CHaR module. The CHaR module leverages teeth-presence information to adapt to missing teeth, improving detection accuracy in complex dental cases. Unlike two-stage workflows that segment teeth before landmarking, CHaRM operates directly on IOS point clouds, reducing complexity, avoiding error propagation, and lowering computational cost. We evaluated CHaRM with five point cloud learning backbones on IOSLandmarks-1k, a new dataset of 1,214 annotated 3D dental models. Both the dataset and code will be publicly released to address the scarcity of open data in orthodontics and foster reproducible research. CHaRM with PointMLP, named CHaRNet, achieved the best accuracy and efficiency. Compared to state-of-the-art methods (TSMDL and ALIIOS), CHaRNet reduced mean Euclidean distance error to 0.56 mm on standard dental models and 1.12 mm across all dentition type, while delivering up to 14.8x faster inference on GPU. This end-to-end approach streamlines orthodontic workflows, enhances the precision of 3D IOS analysis, and enables efficient computer-assisted treatment planning.
>
---
#### [replaced 011] ALow-Cost Real-Time Framework for Industrial Action Recognition Using Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.08420v2](http://arxiv.org/pdf/2403.08420v2)**

> **作者:** Zhicheng Wang; Wensheng Liang; Ruiyan Zhuang; Shuai Li; Jianwei Tan; Xiaoguang Ma
>
> **摘要:** Action recognition (AR) in industrial environments -- particularly for identifying actions and operational gestures -- faces persistent challenges due to high deployment costs, poor cross-scenario generalization, and limited real-time performance. To address these issues, we propose a low-cost real-time framework for industrial action recognition using foundation models, denoted as LRIAR, to enhance recognition accuracy and transferability while minimizing human annotation and computational overhead. The proposed framework constructs an automatically labeled dataset by coupling Grounding DINO with the pretrained BLIP-2 image encoder, enabling efficient and scalable action labeling. Leveraging the constructed dataset, we train YOLOv5 for real-time action detection, and a Vision Transformer (ViT) classifier is deceloped via LoRA-based fine-tuning for action classification. Extensive experiments conducted in real-world industrial settings validate the effectiveness of LRIAR, demonstrating consistent improvements over state-of-the-art methods in recognition accuracy, scenario generalization, and deployment efficiency.
>
---
#### [replaced 012] Consistent and Invariant Generalization Learning for Short-video Misinformation Detection
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.04061v3](http://arxiv.org/pdf/2507.04061v3)**

> **作者:** Hanghui Guo; Weijie Shi; Mengze Li; Juncheng Li; Hao Chen; Yue Cui; Jiajie Xu; Jia Zhu; Jiawei Shen; Zhangze Chen; Sirui Han
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** Short-video misinformation detection has attracted wide attention in the multi-modal domain, aiming to accurately identify the misinformation in the video format accompanied by the corresponding audio. Despite significant advancements, current models in this field, trained on particular domains (source domains), often exhibit unsatisfactory performance on unseen domains (target domains) due to domain gaps. To effectively realize such domain generalization on the short-video misinformation detection task, we propose deep insights into the characteristics of different domains: (1) The detection on various domains may mainly rely on different modalities (i.e., mainly focusing on videos or audios). To enhance domain generalization, it is crucial to achieve optimal model performance on all modalities simultaneously. (2) For some domains focusing on cross-modal joint fraud, a comprehensive analysis relying on cross-modal fusion is necessary. However, domain biases located in each modality (especially in each frame of videos) will be accumulated in this fusion process, which may seriously damage the final identification of misinformation. To address these issues, we propose a new DOmain generalization model via ConsisTency and invariance learning for shORt-video misinformation detection (named DOCTOR), which contains two characteristic modules: (1) We involve the cross-modal feature interpolation to map multiple modalities into a shared space and the interpolation distillation to synchronize multi-modal learning; (2) We design the diffusion model to add noise to retain core features of multi modal and enhance domain invariant features through cross-modal guided denoising. Extensive experiments demonstrate the effectiveness of our proposed DOCTOR model. Our code is public available at https://github.com/ghh1125/DOCTOR.
>
---
#### [replaced 013] Interpretation of Deep Learning Model in Embryo Selection for In Vitro Fertilization (IVF) Treatment
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06680v3](http://arxiv.org/pdf/2506.06680v3)**

> **作者:** Radha Kodali; Venkata Rao Dhulipalla; Venkata Siva Kishor Tatavarty; Madhavi Nadakuditi; Bharadwaj Thiruveedhula; Suryanarayana Gunnam; Durga Prasad Bavirisetti; Gogulamudi Pradeep Reddy
>
> **摘要:** Infertility has a considerable impact on individuals' quality of life, affecting them socially and psychologically, with projections indicating a rise in the upcoming years. In vitro fertilization (IVF) emerges as one of the primary techniques within economically developed nations, employed to address the rising problem of low fertility. Expert embryologists conventionally grade embryos by reviewing blastocyst images to select the most optimal for transfer, yet this process is time-consuming and lacks efficiency. Blastocyst images provide a valuable resource for assessing embryo viability. In this study, we introduce an explainable artificial intelligence (XAI) framework for classifying embryos, employing a fusion of convolutional neural network (CNN) and long short-term memory (LSTM) architecture, referred to as CNN-LSTM. Utilizing deep learning, our model achieves high accuracy in embryo classification while maintaining interpretability through XAI.
>
---
#### [replaced 014] Mask & Match: Learning to Recognize Handwritten Math with Self-Supervised Attention
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.06107v2](http://arxiv.org/pdf/2508.06107v2)**

> **作者:** Shree Mitra; Ritabrata Chakraborty; Nilkanta Sahu
>
> **备注:** We have concluded that the current results, while promising, require substantial improvement and further validation to be competitive with the latest state-of-the-art methods in Handwritten Mathematical Expression Recognition (HMER)
>
> **摘要:** Recognizing handwritten mathematical expressions (HMER) is a challenging task due to the inherent two-dimensional structure, varying symbol scales, and complex spatial relationships among symbols. In this paper, we present a self-supervised learning (SSL) framework for HMER that eliminates the need for expensive labeled data. Our approach begins by pretraining an image encoder using a combination of global and local contrastive loss, enabling the model to learn both holistic and fine-grained representations. A key contribution of this work is a novel self-supervised attention network, which is trained using a progressive spatial masking strategy. This attention mechanism is designed to learn semantically meaningful focus regions, such as operators, exponents, and nested mathematical notation, without requiring any supervision. The progressive masking curriculum encourages the network to become increasingly robust to missing or occluded visual information, ultimately improving structural understanding. Our complete pipeline consists of (1) self-supervised pretraining of the encoder, (2) self-supervised attention learning, and (3) supervised fine-tuning with a transformer decoder to generate LATEX sequences. Extensive experiments on CROHME benchmarks demonstrate that our method outperforms existing SSL and fully supervised baselines, validating the effectiveness of our progressive attention mechanism in enhancing HMER performance. Our codebase can be found here.
>
---
#### [replaced 015] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16430v3](http://arxiv.org/pdf/2503.16430v3)**

> **作者:** Yuqing Wang; Zhijie Lin; Yao Teng; Yuanzhi Zhu; Shuhuai Ren; Jiashi Feng; Xihui Liu
>
> **备注:** Accepted by ICCV 2025. Project page: https://yuqingwang1029.github.io/TokenBridge
>
> **摘要:** Autoregressive visual generation models typically rely on tokenizers to compress images into tokens that can be predicted sequentially. A fundamental dilemma exists in token representation: discrete tokens enable straightforward modeling with standard cross-entropy loss, but suffer from information loss and tokenizer training instability; continuous tokens better preserve visual details, but require complex distribution modeling, complicating the generation pipeline. In this paper, we propose TokenBridge, which bridges this gap by maintaining the strong representation capacity of continuous tokens while preserving the modeling simplicity of discrete tokens. To achieve this, we decouple discretization from the tokenizer training process through post-training quantization that directly obtains discrete tokens from continuous representations. Specifically, we introduce a dimension-wise quantization strategy that independently discretizes each feature dimension, paired with a lightweight autoregressive prediction mechanism that efficiently model the resulting large token space. Extensive experiments show that our approach achieves reconstruction and generation quality on par with continuous methods while using standard categorical prediction. This work demonstrates that bridging discrete and continuous paradigms can effectively harness the strengths of both approaches, providing a promising direction for high-quality visual generation with simple autoregressive modeling. Project page: https://yuqingwang1029.github.io/TokenBridge.
>
---
#### [replaced 016] InterpIoU: Rethinking Bounding Box Regression with Interpolation-Based IoU Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12420v2](http://arxiv.org/pdf/2507.12420v2)**

> **作者:** Haoyuan Liu; Hiroshi Watanabe
>
> **摘要:** Bounding box regression (BBR) is fundamental to object detection, where the regression loss is crucial for accurate localization. Existing IoU-based losses often incorporate handcrafted geometric penalties to address IoU's non-differentiability in non-overlapping cases and enhance BBR performance. However, these penalties are sensitive to box shape, size, and distribution, often leading to suboptimal optimization for small objects and undesired behaviors such as bounding box enlargement due to misalignment with the IoU objective. To address these limitations, we propose InterpIoU, a novel loss function that replaces handcrafted geometric penalties with a term based on the IoU between interpolated boxes and the target. By using interpolated boxes to bridge the gap between predictions and ground truth, InterpIoU provides meaningful gradients in non-overlapping cases and inherently avoids the box enlargement issue caused by misaligned penalties. Simulation results further show that IoU itself serves as an ideal regression target, while existing geometric penalties are both unnecessary and suboptimal. Building on InterpIoU, we introduce Dynamic InterpIoU, which dynamically adjusts interpolation coefficients based on IoU values, enhancing adaptability to scenarios with diverse object distributions. Experiments on COCO, VisDrone, and PASCAL VOC show that our methods consistently outperform state-of-the-art IoU-based losses across various detection frameworks, with particularly notable improvements in small object detection, confirming their effectiveness.
>
---
#### [replaced 017] Region-Level Context-Aware Multimodal Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.12263v2](http://arxiv.org/pdf/2508.12263v2)**

> **作者:** Hongliang Wei; Xianqi Zhang; Xingtao Wang; Xiaopeng Fan; Debin Zhao
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Despite significant progress, existing research on Multimodal Large Language Models (MLLMs) mainly focuses on general visual understanding, overlooking the ability to integrate textual context associated with objects for a more context-aware multimodal understanding -- an ability we refer to as Region-level Context-aware Multimodal Understanding (RCMU). To address this limitation, we first formulate the RCMU task, which requires models to respond to user instructions by integrating both image content and textual information of regions or objects. To equip MLLMs with RCMU capabilities, we propose Region-level Context-aware Visual Instruction Tuning (RCVIT), which incorporates object information into the model input and enables the model to utilize bounding box coordinates to effectively associate objects' visual content with their textual information. To address the lack of datasets, we introduce the RCMU dataset, a large-scale visual instruction tuning dataset that covers multiple RCMU tasks. We also propose RC\&P-Bench, a comprehensive benchmark that can evaluate the performance of MLLMs in RCMU and multimodal personalized understanding tasks. Additionally, we propose a reference-free evaluation metric to perform a comprehensive and fine-grained evaluation of the region-level context-aware image descriptions. By performing RCVIT on Qwen2-VL models with the RCMU dataset, we developed RC-Qwen2-VL models. Experimental results indicate that RC-Qwen2-VL models not only achieve outstanding performance on multiple RCMU tasks but also demonstrate successful applications in multimodal RAG and personalized conversation. Our data, model and benchmark are available at https://github.com/hongliang-wei/RC-MLLM
>
---
#### [replaced 018] Merging and Disentangling Views in Visual Reinforcement Learning for Robotic Manipulation
- **分类: cs.LG; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.04619v2](http://arxiv.org/pdf/2505.04619v2)**

> **作者:** Abdulaziz Almuzairee; Rohan Patil; Dwait Bhatt; Henrik I. Christensen
>
> **备注:** Accepted at CoRL 2025. For project website and code, see https://aalmuzairee.github.io/mad
>
> **摘要:** Vision is well-known for its use in manipulation, especially using visual servoing. Due to the 3D nature of the world, using multiple camera views and merging them creates better representations for Q-learning and in turn, trains more sample efficient policies. Nevertheless, these multi-view policies are sensitive to failing cameras and can be burdensome to deploy. To mitigate these issues, we introduce a Merge And Disentanglement (MAD) algorithm that efficiently merges views to increase sample efficiency while simultaneously disentangling views by augmenting multi-view feature inputs with single-view features. This produces robust policies and allows lightweight deployment. We demonstrate the efficiency and robustness of our approach using Meta-World and ManiSkill3. For project website and code, see https://aalmuzairee.github.io/mad
>
---
#### [replaced 019] Video-LevelGauge: Investigating Contextual Positional Bias in Large Video Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.19650v3](http://arxiv.org/pdf/2508.19650v3)**

> **作者:** Hou Xia; Zheren Fu; Fangcan Ling; Jiajun Li; Yi Tu; Zhendong Mao; Yongdong Zhang
>
> **摘要:** Large video language models (LVLMs) have made notable progress in video understanding, spurring the development of corresponding evaluation benchmarks. However, existing benchmarks generally assess overall performance across entire video sequences, overlooking nuanced behaviors such as contextual positional bias, a critical yet under-explored aspect of LVLM performance. We present Video-LevelGauge, a dedicated benchmark designed to systematically assess positional bias in LVLMs. We employ standardized probes and customized contextual setups, allowing flexible control over context length, probe position, and contextual types to simulate diverse real-world scenarios. In addition, we introduce a comprehensive analysis method that combines statistical measures with morphological pattern recognition to characterize bias. Our benchmark comprises 438 manually curated videos spanning multiple types, yielding 1,177 high-quality multiple-choice questions and 120 open-ended questions, validated for their effectiveness in exposing positional bias. Based on these, we evaluate 27 state-of-the-art LVLMs, including both commercial and open-source models. Our findings reveal significant positional biases in many leading open-source models, typically exhibiting head or neighbor-content preferences. In contrast, commercial models such as Gemini2.5-Pro show impressive, consistent performance across entire video sequences. Further analyses on context length, context variation, and model scale provide actionable insights for mitigating bias and guiding model enhancement . https://github.com/Cola-any/Video-LevelGauge
>
---
#### [replaced 020] Towards Understanding Camera Motions in Any Video
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.15376v2](http://arxiv.org/pdf/2504.15376v2)**

> **作者:** Zhiqiu Lin; Siyuan Cen; Daniel Jiang; Jay Karhade; Hewei Wang; Chancharik Mitra; Tiffany Ling; Yuhan Huang; Sifan Liu; Mingyu Chen; Rushikesh Zawar; Xue Bai; Yilun Du; Chuang Gan; Deva Ramanan
>
> **备注:** Project site: https://linzhiqiu.github.io/papers/camerabench/
>
> **摘要:** We introduce CameraBench, a large-scale dataset and benchmark designed to assess and improve camera motion understanding. CameraBench consists of ~3,000 diverse internet videos, annotated by experts through a rigorous multi-stage quality control process. One of our contributions is a taxonomy of camera motion primitives, designed in collaboration with cinematographers. We find, for example, that some motions like "follow" (or tracking) require understanding scene content like moving subjects. We conduct a large-scale human study to quantify human annotation performance, revealing that domain expertise and tutorial-based training can significantly enhance accuracy. For example, a novice may confuse zoom-in (a change of intrinsics) with translating forward (a change of extrinsics), but can be trained to differentiate the two. Using CameraBench, we evaluate Structure-from-Motion (SfM) and Video-Language Models (VLMs), finding that SfM models struggle to capture semantic primitives that depend on scene content, while VLMs struggle to capture geometric primitives that require precise estimation of trajectories. We then fine-tune a generative VLM on CameraBench to achieve the best of both worlds and showcase its applications, including motion-augmented captioning, video question answering, and video-text retrieval. We hope our taxonomy, benchmark, and tutorials will drive future efforts towards the ultimate goal of understanding camera motions in any video.
>
---
#### [replaced 021] Adaptive Visual Navigation Assistant in 3D RPGs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.18539v2](http://arxiv.org/pdf/2508.18539v2)**

> **作者:** Kaijie Xu; Clark Verbrugge
>
> **摘要:** In complex 3D game environments, players rely on visual affordances to spot map transition points. Efficient identification of such points is important to client-side auto-mapping, and provides an objective basis for evaluating map cue presentation. In this work, we formalize the task of detecting traversable Spatial Transition Points (STPs)-connectors between two sub regions-and selecting the singular Main STP (MSTP), the unique STP that lies on the designer-intended critical path toward the player's current macro-objective, from a single game frame, proposing this as a new research focus. We introduce a two-stage deep-learning pipeline that first detects potential STPs using Faster R-CNN and then ranks them with a lightweight MSTP selector that fuses local and global visual features. Both stages benefit from parameter-efficient adapters, and we further introduce an optional retrieval-augmented fusion step. Our primary goal is to establish the feasibility of this problem and set baseline performance metrics. We validate our approach on a custom-built, diverse dataset collected from five Action RPG titles. Our experiments reveal a key trade-off: while full-network fine-tuning produces superior STP detection with sufficient data, adapter-only transfer is significantly more robust and effective in low-data scenarios and for the MSTP selection task. By defining this novel problem, providing a baseline pipeline and dataset, and offering initial insights into efficient model adaptation, we aim to contribute to future AI-driven navigation aids and data-informed level-design tools.
>
---
#### [replaced 022] Visual Imitation Enables Contextual Humanoid Control
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03729v5](http://arxiv.org/pdf/2505.03729v5)**

> **作者:** Arthur Allshire; Hongsuk Choi; Junyi Zhang; David McAllister; Anthony Zhang; Chung Min Kim; Trevor Darrell; Pieter Abbeel; Jitendra Malik; Angjoo Kanazawa
>
> **备注:** Project website: https://www.videomimic.net/
>
> **摘要:** How can we teach humanoids to climb staircases and sit on chairs using the surrounding environment context? Arguably, the simplest way is to just show them-casually capture a human motion video and feed it to humanoids. We introduce VIDEOMIMIC, a real-to-sim-to-real pipeline that mines everyday videos, jointly reconstructs the humans and the environment, and produces whole-body control policies for humanoid robots that perform the corresponding skills. We demonstrate the results of our pipeline on real humanoid robots, showing robust, repeatable contextual control such as staircase ascents and descents, sitting and standing from chairs and benches, as well as other dynamic whole-body skills-all from a single policy, conditioned on the environment and global root commands. VIDEOMIMIC offers a scalable path towards teaching humanoids to operate in diverse real-world environments.
>
---
#### [replaced 023] Guiding a diffusion model using sliding windows
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.10257v3](http://arxiv.org/pdf/2411.10257v3)**

> **作者:** Nikolas Adaloglou; Tim Kaiser; Damir Iagudin; Markus Kollmann
>
> **备注:** Accepted at BMVC 2025. 30 pages, 16 figures in total, including appendix
>
> **摘要:** Guidance is a widely used technique for diffusion models to enhance sample quality. Technically, guidance is realised by using an auxiliary model that generalises more broadly than the primary model. Using a 2D toy example, we first show that it is highly beneficial when the auxiliary model exhibits similar but stronger generalisation errors than the primary model. Based on this insight, we introduce \emph{masked sliding window guidance (M-SWG)}, a novel, training-free method. M-SWG upweights long-range spatial dependencies by guiding the primary model with itself by selectively restricting its receptive field. M-SWG requires neither access to model weights from previous iterations, additional training, nor class conditioning. M-SWG achieves a superior Inception score (IS) compared to previous state-of-the-art training-free approaches, without introducing sample oversaturation. In conjunction with existing guidance methods, M-SWG reaches state-of-the-art Frechet DINOv2 distance on ImageNet using EDM2-XXL and DiT-XL. The code is available at https://github.com/HHU-MMBS/swg_bmvc2025_official.
>
---
#### [replaced 024] C-Flat++: Towards a More Efficient and Powerful Framework for Continual Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.18860v2](http://arxiv.org/pdf/2508.18860v2)**

> **作者:** Wei Li; Hangjie Yuan; Zixiang Zhao; Yifan Zhu; Aojun Lu; Tao Feng; Yanan Sun
>
> **摘要:** Balancing sensitivity to new tasks and stability for retaining past knowledge is crucial in continual learning (CL). Recently, sharpness-aware minimization has proven effective in transfer learning and has also been adopted in continual learning (CL) to improve memory retention and learning efficiency. However, relying on zeroth-order sharpness alone may favor sharper minima over flatter ones in certain settings, leading to less robust and potentially suboptimal solutions. In this paper, we propose \textbf{C}ontinual \textbf{Flat}ness (\textbf{C-Flat}), a method that promotes flatter loss landscapes tailored for CL. C-Flat offers plug-and-play compatibility, enabling easy integration with minimal modifications to the code pipeline. Besides, we present a general framework that integrates C-Flat into all major CL paradigms and conduct comprehensive comparisons with loss-minima optimizers and flat-minima-based CL methods. Our results show that C-Flat consistently improves performance across a wide range of settings. In addition, we introduce C-Flat++, an efficient yet effective framework that leverages selective flatness-driven promotion, significantly reducing the update cost required by C-Flat. Extensive experiments across multiple CL methods, datasets, and scenarios demonstrate the effectiveness and efficiency of our proposed approaches. Code is available at https://github.com/WanNaa/C-Flat.
>
---
#### [replaced 025] PicoPose: Progressive Pixel-to-Pixel Correspondence Learning for Novel Object Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02617v2](http://arxiv.org/pdf/2504.02617v2)**

> **作者:** Lihua Liu; Jiehong Lin; Zhenxin Liu; Kui Jia
>
> **备注:** CoRL2025
>
> **摘要:** RGB-based novel object pose estimation is critical for rapid deployment in robotic applications, yet zero-shot generalization remains a key challenge. In this paper, we introduce PicoPose, a novel framework designed to tackle this task using a three-stage pixel-to-pixel correspondence learning process. Firstly, PicoPose matches features from the RGB observation with those from rendered object templates, identifying the best-matched template and establishing coarse correspondences. Secondly, PicoPose smooths the correspondences by globally regressing a 2D affine transformation, including in-plane rotation, scale, and 2D translation, from the coarse correspondence map. Thirdly, PicoPose applies the affine transformation to the feature map of the best-matched template and learns correspondence offsets within local regions to achieve fine-grained correspondences. By progressively refining the correspondences, PicoPose significantly improves the accuracy of object poses computed via PnP/RANSAC. PicoPose achieves state-of-the-art performance on the seven core datasets of the BOP benchmark, demonstrating exceptional generalization to novel objects. Code and trained models are available at https://github.com/foollh/PicoPose.
>
---
#### [replaced 026] JambaTalk: Speech-Driven 3D Talking Head Generation Based on Hybrid Transformer-Mamba Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.01627v2](http://arxiv.org/pdf/2408.01627v2)**

> **作者:** Farzaneh Jafari; Stefano Berretti; Anup Basu
>
> **备注:** 23 pages with 8 figures
>
> **摘要:** In recent years, the talking head generation has become a focal point for researchers. Considerable effort is being made to refine lip-sync motion, capture expressive facial expressions, generate natural head poses, and achieve high-quality video. However, no single model has yet achieved equivalence across all quantitative and qualitative metrics. We introduce Jamba, a hybrid Transformer-Mamba model, to animate a 3D face. Mamba, a pioneering Structured State Space Model (SSM) architecture, was developed to overcome the limitations of conventional Transformer architectures, particularly in handling long sequences. This challenge has constrained traditional models. Jamba combines the advantages of both the Transformer and Mamba approaches, offering a comprehensive solution. Based on the foundational Jamba block, we present JambaTalk to enhance motion variety and lip sync through multimodal integration. Extensive experiments reveal that our method achieves performance comparable or superior to state-of-the-art models.
>
---
#### [replaced 027] PointDGRWKV: Generalizing RWKV-like Architecture to Unseen Domains for Point Cloud Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.20835v2](http://arxiv.org/pdf/2508.20835v2)**

> **作者:** Hao Yang; Qianyu Zhou; Haijia Sun; Xiangtai Li; Xuequan Lu; Lizhuang Ma; Shuicheng Yan
>
> **摘要:** Domain Generalization (DG) has been recently explored to enhance the generalizability of Point Cloud Classification (PCC) models toward unseen domains. Prior works are based on convolutional networks, Transformer or Mamba architectures, either suffering from limited receptive fields or high computational cost, or insufficient long-range dependency modeling. RWKV, as an emerging architecture, possesses superior linear complexity, global receptive fields, and long-range dependency. In this paper, we present the first work that studies the generalizability of RWKV models in DG PCC. We find that directly applying RWKV to DG PCC encounters two significant challenges: RWKV's fixed direction token shift methods, like Q-Shift, introduce spatial distortions when applied to unstructured point clouds, weakening local geometric modeling and reducing robustness. In addition, the Bi-WKV attention in RWKV amplifies slight cross-domain differences in key distributions through exponential weighting, leading to attention shifts and degraded generalization. To this end, we propose PointDGRWKV, the first RWKV-based framework tailored for DG PCC. It introduces two key modules to enhance spatial modeling and cross-domain robustness, while maintaining RWKV's linear efficiency. In particular, we present Adaptive Geometric Token Shift to model local neighborhood structures to improve geometric context awareness. In addition, Cross-Domain key feature Distribution Alignment is designed to mitigate attention drift by aligning key feature distributions across domains. Extensive experiments on multiple benchmarks demonstrate that PointDGRWKV achieves state-of-the-art performance on DG PCC.
>
---
#### [replaced 028] Computer-Aided Design of Personalized Occlusal Positioning Splints Using Multimodal 3D Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12868v2](http://arxiv.org/pdf/2504.12868v2)**

> **作者:** Agnieszka Anna Tomaka; Leszek Luchowski; Michał Tarnawski; Dariusz Pojda
>
> **摘要:** Digital technology plays a crucial role in designing customized medical devices, such as occlusal splints, commonly used in the management of disorders of the stomatognathic system. This methodological proof-of-concept study presents a computer-aided approach for designing and evaluating occlusal positioning splints. The primary aim is to demonstrate the feasibility and geometric accuracy of the proposed method at the preclinical stage. In this approach, a three-dimensional splint is generated using a transformation matrix to represent the therapeutic mandibular position. An experienced operator defines this position using a virtual patient model reconstructed from intraoral scans, CBCT, 3D facial scans, and a digitized plaster model. We introduce a novel method for generating splints that reproduces occlusal conditions in the therapeutic position and resolves surface conflicts through virtual embossing. The process for obtaining transformation matrices using dental tools and intraoral devices commonly employed in dental and laboratory workflows is described, and the geometric accuracy of both designed and printed splints is evaluated using profile and surface deviation analysis. The method supports reproducible, patient-specific splint fabrication and provides a transparent foundation for future validation studies, supporting multimodal image registration and quantification of occlusal discrepancies in research settings.
>
---
#### [replaced 029] DDaTR: Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.03401v2](http://arxiv.org/pdf/2505.03401v2)**

> **作者:** Shanshan Song; Hui Tang; Honglong Yang; Xiaomeng Li
>
> **备注:** Accepted in IEEE Transactions on Medical Imaging (TMI). Code is available at https://github.com/xmed-lab/DDaTR
>
> **摘要:** Radiology Report Generation (RRG) automates the creation of radiology reports from medical imaging, enhancing the efficiency of the reporting process. Longitudinal Radiology Report Generation (LRRG) extends RRG by incorporating the ability to compare current and prior exams, facilitating the tracking of temporal changes in clinical findings. Existing LRRG approaches only extract features from prior and current images using a visual pre-trained encoder, which are then concatenated to generate the final report. However, these methods struggle to effectively capture both spatial and temporal correlations during the feature extraction process. Consequently, the extracted features inadequately capture the information of difference across exams and thus underrepresent the expected progressions, leading to sub-optimal performance in LRRG. To address this, we develop a novel dynamic difference-aware temporal residual network (DDaTR). In DDaTR, we introduce two modules at each stage of the visual encoder to capture multi-level spatial correlations. The Dynamic Feature Alignment Module (DFAM) is designed to align prior features across modalities for the integrity of prior clinical information. Prompted by the enriched prior features, the dynamic difference-aware module (DDAM) captures favorable difference information by identifying relationships across exams. Furthermore, our DDaTR employs the dynamic residual network to unidirectionally transmit longitudinal information, effectively modelling temporal correlations. Extensive experiments demonstrated superior performance over existing methods on three benchmarks, proving its efficacy in both RRG and LRRG tasks.
>
---
#### [replaced 030] Single Domain Generalization for Multimodal Cross-Cancer Prognosis via Dirac Rebalancer and Distribution Entanglement
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08340v2](http://arxiv.org/pdf/2507.08340v2)**

> **作者:** Jia-Xuan Jiang; Jiashuai Liu; Hongtao Wu; Yifeng Wu; Zhong Wang; Qi Bi; Yefeng Zheng
>
> **备注:** Accepted by ACMMM 25
>
> **摘要:** Deep learning has shown remarkable performance in integrating multimodal data for survival prediction. However, existing multimodal methods mainly focus on single cancer types and overlook the challenge of generalization across cancers. In this work, we are the first to reveal that multimodal prognosis models often generalize worse than unimodal ones in cross-cancer scenarios, despite the critical need for such robustness in clinical practice. To address this, we propose a new task: Cross-Cancer Single Domain Generalization for Multimodal Prognosis, which evaluates whether models trained on a single cancer type can generalize to unseen cancers. We identify two key challenges: degraded features from weaker modalities and ineffective multimodal integration. To tackle these, we introduce two plug-and-play modules: Sparse Dirac Information Rebalancer (SDIR) and Cancer-aware Distribution Entanglement (CADE). SDIR mitigates the dominance of strong features by applying Bernoulli-based sparsification and Dirac-inspired stabilization to enhance weaker modality signals. CADE, designed to synthesize the target domain distribution, fuses local morphological cues and global gene expression in latent space. Experiments on a four-cancer-type benchmark demonstrate superior generalization, laying the foundation for practical, robust cross-cancer multimodal prognosis. Code is available at https://github.com/HopkinsKwong/MCCSDG
>
---
#### [replaced 031] CE-RS-SBCIT A Novel Channel Enhanced Hybrid CNN Transformer with Residual, Spatial, and Boundary-Aware Learning for Brain Tumor MRI Analysis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.17128v2](http://arxiv.org/pdf/2508.17128v2)**

> **作者:** Mirza Mumtaz Zahoor; Saddam Hussain Khan
>
> **备注:** 37 Pages, 12 Figures
>
> **摘要:** Brain tumors remain among the most lethal human diseases, where early detection and accurate classification are critical for effective diagnosis and treatment planning. Although deep learning-based computer-aided diagnostic (CADx) systems have shown remarkable progress. However, conventional convolutional neural networks (CNNs) and Transformers face persistent challenges, including high computational cost, sensitivity to minor contrast variations, structural heterogeneity, and texture inconsistencies in MRI data. Therefore, a novel hybrid framework, CE-RS-SBCIT, is introduced, integrating residual and spatial learning-based CNNs with transformer-driven modules. The proposed framework exploits local fine-grained and global contextual cues through four core innovations: (i) a smoothing and boundary-based CNN-integrated Transformer (SBCIT), (ii) tailored residual and spatial learning CNNs, (iii) a channel enhancement (CE) strategy, and (iv) a novel spatial attention mechanism. The developed SBCIT employs stem convolution and contextual interaction transformer blocks with systematic smoothing and boundary operations, enabling efficient global feature modeling. Moreover, Residual and spatial CNNs, enhanced by auxiliary transfer-learned feature maps, enrich the representation space, while the CE module amplifies discriminative channels and mitigates redundancy. Furthermore, the spatial attention mechanism selectively emphasizes subtle contrast and textural variations across tumor classes. Extensive evaluation on challenging MRI datasets from Kaggle and Figshare, encompassing glioma, meningioma, pituitary tumors, and healthy controls, demonstrates superior performance, achieving 98.30% accuracy, 98.08% sensitivity, 98.25% F1-score, and 98.43% precision.
>
---
#### [replaced 032] Maximising Kidney Glomeruli Segmentation using Minimal Labels via Self-Supervision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.15389v2](http://arxiv.org/pdf/2412.15389v2)**

> **作者:** Zeeshan Nisar; Thomas Lampert
>
> **备注:** 35 pages, 10 figures, 3 Tables
>
> **摘要:** Histopathology, the microscopic examination of tissue samples, is essential for disease diagnosis and prognosis. Accurate segmentation and identification of key regions in histopathology images are crucial for developing automated solutions. However, state-of-art deep learning segmentation methods like UNet require extensive labels, which is both costly and time-consuming, particularly when dealing with multiple stainings. To mitigate this, various multi-stain segmentation methods such as UDA-GAN have been developed, which reduce the need for labels by requiring only one (source) stain to be labelled. Nonetheless, obtaining source stain labels can still be challenging, and segmentation models fail when they are unavailable. This article shows that through self-supervised pre-training$-$including SimCLR, BYOL, and a novel approach, HR-CS-CO$-$the performance of these segmentation methods (UNet, and UDAGAN) can be retained even with 95% fewer labels. Notably, with self-supervised pre-training and using only 5% labels, the performance drops are minimal: 5.9% for UNet and 6.2% for UDAGAN, compared to their respective fully supervised counterparts (without pre-training, using 100% labels). Furthermore, these findings are shown to generalise beyond their training distribution to public benchmark datasets. Im-
>
---
#### [replaced 033] BuzzSet v1.0: A Dataset for Pollinator Detection in Field Conditions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.19762v2](http://arxiv.org/pdf/2508.19762v2)**

> **作者:** Ahmed Emam; Mohamed Elbassiouny; Julius Miller; Patrick Donworth; Sabine Seidel; Ribana Roscher
>
> **摘要:** Pollinator insects such as honeybees and bumblebees are vital to global food production and ecosystem stability, yet their populations are declining due to anthropogenic and environmental stressors. Scalable, automated monitoring in agricultural environments remains an open challenge due to the difficulty of detecting small, fast-moving, and often camouflaged insects. To address this, we present BuzzSet v1.0, a large-scale dataset of high-resolution pollinator images collected under real field conditions. BuzzSet contains 7,856 manually verified images with more than 8,000 annotated instances across three classes: honeybees, bumblebees, and unidentified insects. Initial annotations were produced using a YOLOv12 model trained on external data and refined through human verification with open-source tools. All images were preprocessed into 256 x 256 tiles to improve the detection of small insects. We provide baselines using the RF-DETR transformer-based object detector. The model achieves strong classification accuracy with F1 scores of 0.94 and 0.92 for honeybees and bumblebees, with minimal confusion between these categories. The unidentified class remains more difficult due to label ambiguity and fewer samples, yet still contributes insights for robustness evaluation. Overall detection performance (mAP at 0.50 of 0.559) illustrates the challenging nature of the dataset and its potential to drive advances in small object detection under realistic ecological conditions. Future work focuses on expanding the dataset to version 2.0 with additional annotations and evaluating further detection strategies. BuzzSet establishes a benchmark for ecological computer vision, with the primary challenge being reliable detection of insects frequently camouflaged within natural vegetation, highlighting an open problem for future research.
>
---
#### [replaced 034] TorchCP: A Python Library for Conformal Prediction
- **分类: cs.LG; cs.CV; math.ST; stat.TH**

- **链接: [http://arxiv.org/pdf/2402.12683v4](http://arxiv.org/pdf/2402.12683v4)**

> **作者:** Jianguo Huang; Jianqing Song; Xuanning Zhou; Bingyi Jing; Hongxin Wei
>
> **摘要:** Conformal prediction (CP) is a powerful statistical framework that generates prediction intervals or sets with guaranteed coverage probability. While CP algorithms have evolved beyond traditional classifiers and regressors to sophisticated deep learning models like deep neural networks (DNNs), graph neural networks (GNNs), and large language models (LLMs), existing CP libraries often lack the model support and scalability for large-scale DL scenarios. This paper introduces TorchCP, a PyTorch-native library designed to integrate state-of-the-art CP algorithms into deep learning techniques, including DNN-based classifier/regressor, GNN, and LLM. Released under the LGPL-3.0 license, TorchCP comprises about 16k lines of code, validated with 100% unit test coverage and detailed documentation. Notably, TorchCP enables CP-specific training algorithms, online prediction, and GPU-accelerated batch processing, achieving up to 90% reduction in inference time on large datasets. With its low-coupling design, comprehensive suite of advanced methods, and full GPU scalability, TorchCP empowers researchers and practitioners to enhance uncertainty quantification across cutting-edge applications.
>
---
#### [replaced 035] Large Intestine 3D Shape Refinement Using Point Diffusion Models for Digital Phantom Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2309.08289v3](http://arxiv.org/pdf/2309.08289v3)**

> **作者:** Kaouther Mouheb; Mobina Ghojogh Nejad; Lavsen Dahal; Ehsan Samei; Kyle J. Lafata; W. Paul Segars; Joseph Y. Lo
>
> **摘要:** Accurate 3D modeling of human organs is critical for constructing digital phantoms in virtual imaging trials. However, organs such as the large intestine remain particularly challenging due to their complex geometry and shape variability. We propose CLAP, a novel Conditional LAtent Point-diffusion model that combines geometric deep learning with denoising diffusion models to enhance 3D representations of the large intestine. Given point clouds sampled from segmentation masks, we employ a hierarchical variational autoencoder to learn both global and local latent shape representations. Two conditional diffusion models operate within this latent space to refine the organ shape. A pretrained surface reconstruction model is then used to convert the refined point clouds into meshes. CLAP achieves substantial improvements in shape modeling accuracy, reducing Chamfer distance by 26% and Hausdorff distance by 36% relative to the initial suboptimal shapes. This approach offers a robust and extensible solution for high-fidelity organ modeling, with potential applicability to a wide range of anatomical structures.
>
---
#### [replaced 036] A Hybrid Fully Convolutional CNN-Transformer Model for Inherently Interpretable Disease Detection from Retinal Fundus Images
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.08481v3](http://arxiv.org/pdf/2504.08481v3)**

> **作者:** Kerol Djoumessi; Samuel Ofosu Mensah; Philipp Berens
>
> **备注:** I made an error in this version
>
> **摘要:** In many medical imaging tasks, convolutional neural networks (CNNs) efficiently extract local features hierarchically. More recently, vision transformers (ViTs) have gained popularity, using self-attention mechanisms to capture global dependencies, but lacking the inherent spatial localization of convolutions. Therefore, hybrid models combining CNNs and ViTs have been developed to combine the strengths of both architectures. However, such hybrid models are difficult to interpret, which hinders their application in medical imaging. In this work, we introduce an interpretable-by-design hybrid fully convolutional CNN-Transformer architecture for retinal disease detection. Unlike widely used post-hoc saliency methods for ViTs, our approach generates faithful and localized evidence maps that directly reflect the mode's decision process. We evaluated our method on two medical tasks focused on disease detection using color fundus images. Our model achieves state-of-the-art predictive performance compared to black-box and interpretable models and provides class-specific sparse evidence maps in a single forward pass. The code is available at: https://github.com/kdjoumessi/Self-Explainable-CNN-Transformer.
>
---
#### [replaced 037] mmFlux: Crowd Flow Analytics with Commodity mmWave MIMO Radar
- **分类: eess.SP; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07331v2](http://arxiv.org/pdf/2507.07331v2)**

> **作者:** Anurag Pallaprolu; Winston Hurst; Yasamin Mostofi
>
> **摘要:** In this paper, we present mmFlux: a novel framework for extracting underlying crowd motion patterns and inferring crowd semantics using mmWave radar. First, our proposed signal processing pipeline combines optical flow estimation concepts from vision with novel statistical and morphological noise filtering. This approach generates high-fidelity mmWave flow fields-compact 2D vector representations of crowd motion. We then introduce a novel approach that transforms these fields into directed geometric graphs. In these graphs, edges capture dominant flow currents, vertices mark crowd splitting or merging, and flow distribution is quantified across edges. Finally, we show that analyzing the local Jacobian and computing the corresponding curl and divergence enables extraction of key crowd semantics for both structured and diffused crowds. We conduct 21 experiments on crowds of up to 20 people across 3 areas, using commodity mmWave radar. Our framework achieves high-fidelity graph reconstruction of the underlying flow structure, even for complex crowd patterns, demonstrating strong spatial alignment and precise quantitative characterization of flow split ratios. Finally, our curl and divergence analysis accurately infers key crowd semantics, e.g., abrupt turns, boundaries where flow directions shift, dispersions, and gatherings. Overall, these findings validate mmFlux, underscoring its potential for various crowd analytics applications.
>
---
#### [replaced 038] QuaDreamer: Controllable Panoramic Video Generation for Quadruped Robots
- **分类: cs.RO; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.02512v2](http://arxiv.org/pdf/2508.02512v2)**

> **作者:** Sheng Wu; Fei Teng; Hao Shi; Qi Jiang; Kai Luo; Kaiwei Wang; Kailun Yang
>
> **备注:** Accepted to CoRL 2025. The source code and model weights will be publicly available at \url{https://github.com/losehu/QuaDreamer
>
> **摘要:** Panoramic cameras, capturing comprehensive 360-degree environmental data, are suitable for quadruped robots in surrounding perception and interaction with complex environments. However, the scarcity of high-quality panoramic training data-caused by inherent kinematic constraints and complex sensor calibration challenges-fundamentally limits the development of robust perception systems tailored to these embodied platforms. To address this issue, we propose QuaDreamer-the first panoramic data generation engine specifically designed for quadruped robots. QuaDreamer focuses on mimicking the motion paradigm of quadruped robots to generate highly controllable, realistic panoramic videos, providing a data source for downstream tasks. Specifically, to effectively capture the unique vertical vibration characteristics exhibited during quadruped locomotion, we introduce Vertical Jitter Encoding (VJE). VJE extracts controllable vertical signals through frequency-domain feature filtering and provides high-quality prompts. To facilitate high-quality panoramic video generation under jitter signal control, we propose a Scene-Object Controller (SOC) that effectively manages object motion and boosts background jitter control through the attention mechanism. To address panoramic distortions in wide-FoV video generation, we propose the Panoramic Enhancer (PE)-a dual-stream architecture that synergizes frequency-texture refinement for local detail enhancement with spatial-structure correction for global geometric consistency. We further demonstrate that the generated video sequences can serve as training data for the quadruped robot's panoramic visual perception model, enhancing the performance of multi-object tracking in 360-degree scenes. The source code and model weights will be publicly available at https://github.com/losehu/QuaDreamer.
>
---
#### [replaced 039] Convolutional Rectangular Attention Module
- **分类: cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2503.10875v2](http://arxiv.org/pdf/2503.10875v2)**

> **作者:** Hai-Vy Nguyen; Fabrice Gamboa; Sixin Zhang; Reda Chhaibi; Serge Gratton; Thierry Giaccone
>
> **摘要:** In this paper, we introduce a novel spatial attention module that can be easily integrated to any convolutional network. This module guides the model to pay attention to the most discriminative part of an image. This enables the model to attain a better performance by an end-to-end training. In conventional approaches, a spatial attention map is typically generated in a position-wise manner. Thus, it is often resulting in irregular boundaries and so can hamper generalization to new samples. In our method, the attention region is constrained to be rectangular. This rectangle is parametrized by only 5 parameters, allowing for a better stability and generalization to new samples. In our experiments, our method systematically outperforms the position-wise counterpart. So that, we provide a novel useful spatial attention mechanism for convolutional models. Besides, our module also provides the interpretability regarding the \textit{where to look} question, as it helps to know the part of the input on which the model focuses to produce the prediction.
>
---
