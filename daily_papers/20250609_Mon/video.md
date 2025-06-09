# 计算机视觉 cs.CV

- **最新发布 141 篇**

- **更新 78 篇**

## 最新发布

#### [new 001] DeformCL: Learning Deformable Centerline Representation for Vessel Extraction in 3D Medical Image
- **分类: cs.CV**

- **简介: 该论文属于3D医学图像中血管分割任务，旨在解决传统方法易导致血管结构断裂或碎片的问题。作者提出DeformCL，采用可变形中心线的连续表示方法，提升血管结构的连通性和抗噪能力，并设计级联训练流程验证其有效性与临床意义。**

- **链接: [http://arxiv.org/pdf/2506.05820v1](http://arxiv.org/pdf/2506.05820v1)**

> **作者:** Ziwei Zhao; Zhixing Zhang; Yuhang Liu; Zhao Zhang; Haojun Yu; Dong Wang; Liwei Wang
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** In the field of 3D medical imaging, accurately extracting and representing the blood vessels with curvilinear structures holds paramount importance for clinical diagnosis. Previous methods have commonly relied on discrete representation like mask, often resulting in local fractures or scattered fragments due to the inherent limitations of the per-pixel classification paradigm. In this work, we introduce DeformCL, a new continuous representation based on Deformable Centerlines, where centerline points act as nodes connected by edges that capture spatial relationships. Compared with previous representations, DeformCL offers three key advantages: natural connectivity, noise robustness, and interaction facility. We present a comprehensive training pipeline structured in a cascaded manner to fully exploit these favorable properties of DeformCL. Extensive experiments on four 3D vessel segmentation datasets demonstrate the effectiveness and superiority of our method. Furthermore, the visualization of curved planar reformation images validates the clinical significance of the proposed framework. We release the code in https://github.com/barry664/DeformCL
>
---
#### [new 002] ExAct: A Video-Language Benchmark for Expert Action Analysis
- **分类: cs.CV**

- **简介: 该论文提出了ExAct，一个用于视频-语言专家动作分析的基准测试，旨在评估模型对人类复杂技能的精细理解能力。任务是选择正确答案，需具备专业级理解。论文构建了包含3521个视频问答对的数据集，覆盖6大领域。当前最先进模型如GPT-4o表现远低于人类专家。**

- **链接: [http://arxiv.org/pdf/2506.06277v1](http://arxiv.org/pdf/2506.06277v1)**

> **作者:** Han Yi; Yulu Pan; Feihong He; Xinyu Liu; Benjamin Zhang; Oluwatumininu Oguntola; Gedas Bertasius
>
> **摘要:** We present ExAct, a new video-language benchmark for expert-level understanding of skilled physical human activities. Our new benchmark contains 3521 expert-curated video question-answer pairs spanning 11 physical activities in 6 domains: Sports, Bike Repair, Cooking, Health, Music, and Dance. ExAct requires the correct answer to be selected from five carefully designed candidate options, thus necessitating a nuanced, fine-grained, expert-level understanding of physical human skills. Evaluating the recent state-of-the-art VLMs on ExAct reveals a substantial performance gap relative to human expert performance. Specifically, the best-performing GPT-4o model achieves only 44.70% accuracy, well below the 82.02% attained by trained human specialists/experts. We believe that ExAct will be beneficial for developing and evaluating VLMs capable of precise understanding of human skills in various physical and procedural domains. Dataset and code are available at https://texaser.github.io/exact_project_page/
>
---
#### [new 003] A Novel Large-scale Crop Dataset and Dual-stream Transformer Method for Fine-grained Hierarchical Crop Classification from Integrated Hyperspectral EnMAP Data and Multispectral Sentinel-2 Time Series
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感与精准农业任务，旨在解决细粒度作物分类问题。作者构建了大规模多源数据集H2Crop，并提出双流Transformer模型，融合EnMAP高光谱与Sentinel-2时序数据，提升作物分类精度。实验表明该方法比现有深度学习方法更优。**

- **链接: [http://arxiv.org/pdf/2506.06155v1](http://arxiv.org/pdf/2506.06155v1)**

> **作者:** Wenyuan Li; Shunlin Liang; Yuxiang Zhang; Liqin Liu; Keyan Chen; Yongzhe Chen; Han Ma; Jianglei Xu; Yichuan Ma; Shikang Guan; Zhenwei Shi
>
> **备注:** 28 pages, 12 figures
>
> **摘要:** Fine-grained crop classification is crucial for precision agriculture and food security monitoring. It requires simultaneous capture of both phenological dynamics (obtained from multi-temporal satellite data like Sentinel-2) and subtle spectral variations (demanding nanometer-scale spectral resolution from hyperspectral imagery). Research combining these two modalities remains scarce currently due to challenges in hyperspectral data acquisition and crop types annotation costs. To address these issues, we construct a hierarchical hyperspectral crop dataset (H2Crop) by integrating 30m-resolution EnMAP hyperspectral data with Sentinel-2 time series. With over one million annotated field parcels organized in a four-tier crop taxonomy, H2Crop establishes a vital benchmark for fine-grained agricultural crop classification and hyperspectral image processing. We propose a dual-stream Transformer architecture that synergistically processes these modalities. It coordinates two specialized pathways: a spectral-spatial Transformer extracts fine-grained signatures from hyperspectral EnMAP data, while a temporal Swin Transformer extracts crop growth patterns from Sentinel-2 time series. The designed hierarchy classification heads with hierarchical fusion then simultaneously delivers multi-level classification across all taxonomic tiers. Experiments demonstrate that adding hyperspectral EnMAP data to Sentinel-2 time series yields a 4.2% average F1-scores improvement (peaking at 6.3%). Extensive comparisons also confirming our method's higher accuracy over existing deep learning approaches for crop type classification and the consistent benefits of hyperspectral data across varying temporal windows and crop change scenarios. Codes and dataset will be available at https://github.com/flyakon/H2Crop and www.glass.hku.hk Keywords: Crop type classification, precision agriculture, remote sensing, deep learning, hyperspectral data, Sentinel-2 time series, fine-grained crops
>
---
#### [new 004] MOGO: Residual Quantized Hierarchical Causal Transformer for High-Quality and Real-Time 3D Human Motion Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D人体运动生成任务，旨在解决高质量、实时性和流式生成的挑战。作者提出MOGO框架，包含运动量化模块和层次化因果Transformer，实现高效单次生成，并引入文本对齐机制提升语义准确性，从而在多个数据集上取得良好表现。**

- **链接: [http://arxiv.org/pdf/2506.05952v1](http://arxiv.org/pdf/2506.05952v1)**

> **作者:** Dongjie Fu; Tengjiao Sun; Pengcheng Fang; Xiaohao Cai; Hansung Kim
>
> **备注:** 9 pages, 4 figures, conference
>
> **摘要:** Recent advances in transformer-based text-to-motion generation have led to impressive progress in synthesizing high-quality human motion. Nevertheless, jointly achieving high fidelity, streaming capability, real-time responsiveness, and scalability remains a fundamental challenge. In this paper, we propose MOGO (Motion Generation with One-pass), a novel autoregressive framework tailored for efficient and real-time 3D motion generation. MOGO comprises two key components: (1) MoSA-VQ, a motion scale-adaptive residual vector quantization module that hierarchically discretizes motion sequences with learnable scaling to produce compact yet expressive representations; and (2) RQHC-Transformer, a residual quantized hierarchical causal transformer that generates multi-layer motion tokens in a single forward pass, significantly reducing inference latency. To enhance semantic fidelity, we further introduce a text condition alignment mechanism that improves motion decoding under textual control. Extensive experiments on benchmark datasets including HumanML3D, KIT-ML, and CMP demonstrate that MOGO achieves competitive or superior generation quality compared to state-of-the-art transformer-based methods, while offering substantial improvements in real-time performance, streaming generation, and generalization under zero-shot settings.
>
---
#### [new 005] CoMemo: LVLMs Need Image Context with Image Memory
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言建模任务，旨在解决现有大型视觉语言模型（LVLMs）在处理多模态信息时忽视中间视觉内容及空间结构的问题。作者提出CoMemo架构与RoPE-DHR位置编码方法，提升长上下文、多图像理解和视觉问答等任务的性能。**

- **链接: [http://arxiv.org/pdf/2506.06279v1](http://arxiv.org/pdf/2506.06279v1)**

> **作者:** Shi Liu; Weijie Su; Xizhou Zhu; Wenhai Wang; Jifeng Dai
>
> **备注:** ICML 2025
>
> **摘要:** Recent advancements in Large Vision-Language Models built upon Large Language Models have established aligning visual features with LLM representations as the dominant paradigm. However, inherited LLM architectural designs introduce suboptimal characteristics for multimodal processing. First, LVLMs exhibit a bimodal distribution in attention allocation, leading to the progressive neglect of middle visual content as context expands. Second, conventional positional encoding schemes fail to preserve vital 2D structural relationships when processing dynamic high-resolution images. To address these limitations, we propose CoMemo - a dual-path architecture that combines a Context image path with an image Memory path for visual processing, effectively alleviating visual information neglect. Additionally, we introduce RoPE-DHR, a novel positional encoding mechanism that employs thumbnail-based positional aggregation to maintain 2D spatial awareness while mitigating remote decay in extended sequences. Evaluations across seven benchmarks,including long-context comprehension, multi-image reasoning, and visual question answering, demonstrate CoMemo's superior performance compared to conventional LVLM architectures. Project page is available at https://lalbj.github.io/projects/CoMemo/.
>
---
#### [new 006] UniRes: Universal Image Restoration for Complex Degradations
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决真实场景中复杂退化（如多种退化混合）的图像复原问题。作者提出了一种基于扩散模型的通用框架UniRes，通过整合多个专用模型的知识，实现对复杂退化图像的端到端恢复，并可在不同退化类型上灵活扩展和调整质量-保真度权衡。**

- **链接: [http://arxiv.org/pdf/2506.05599v1](http://arxiv.org/pdf/2506.05599v1)**

> **作者:** Mo Zhou; Keren Ye; Mauricio Delbracio; Peyman Milanfar; Vishal M. Patel; Hossein Talebi
>
> **摘要:** Real-world image restoration is hampered by diverse degradations stemming from varying capture conditions, capture devices and post-processing pipelines. Existing works make improvements through simulating those degradations and leveraging image generative priors, however generalization to in-the-wild data remains an unresolved problem. In this paper, we focus on complex degradations, i.e., arbitrary mixtures of multiple types of known degradations, which is frequently seen in the wild. A simple yet flexible diffusionbased framework, named UniRes, is proposed to address such degradations in an end-to-end manner. It combines several specialized models during the diffusion sampling steps, hence transferring the knowledge from several well-isolated restoration tasks to the restoration of complex in-the-wild degradations. This only requires well-isolated training data for several degradation types. The framework is flexible as extensions can be added through a unified formulation, and the fidelity-quality trade-off can be adjusted through a new paradigm. Our proposed method is evaluated on both complex-degradation and single-degradation image restoration datasets. Extensive qualitative and quantitative experimental results show consistent performance gain especially for images with complex degradations.
>
---
#### [new 007] Dream to Generalize: Zero-Shot Model-Based Reinforcement Learning for Unseen Visual Distractions
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于模型基础强化学习任务，旨在解决视觉干扰下策略泛化能力差的问题。作者提出Dr. G方法，通过自监督学习与对比学习提取任务相关特征，并引入动态逆模型提升时序理解，增强模型对视觉干扰的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.05419v1](http://arxiv.org/pdf/2506.05419v1)**

> **作者:** Jeongsoo Ha; Kyungsoo Kim; Yusung Kim
>
> **备注:** AAAI 2023
>
> **摘要:** Model-based reinforcement learning (MBRL) has been used to efficiently solve vision-based control tasks in highdimensional image observations. Although recent MBRL algorithms perform well in trained observations, they fail when faced with visual distractions in observations. These task-irrelevant distractions (e.g., clouds, shadows, and light) may be constantly present in real-world scenarios. In this study, we propose a novel self-supervised method, Dream to Generalize (Dr. G), for zero-shot MBRL. Dr. G trains its encoder and world model with dual contrastive learning which efficiently captures task-relevant features among multi-view data augmentations. We also introduce a recurrent state inverse dynamics model that helps the world model to better understand the temporal structure. The proposed methods can enhance the robustness of the world model against visual distractions. To evaluate the generalization performance, we first train Dr. G on simple backgrounds and then test it on complex natural video backgrounds in the DeepMind Control suite, and the randomizing environments in Robosuite. Dr. G yields a performance improvement of 117% and 14% over prior works, respectively. Our code is open-sourced and available at https://github.com/JeongsooHa/DrG.git
>
---
#### [new 008] Robust sensor fusion against on-vehicle sensor staleness
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶中的感知任务，旨在解决多传感器数据因时间不同步导致的感知退化问题。通过引入时间戳偏移特征和模拟传感器延迟的数据增强策略，提升了模型在传感器延迟下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.05780v1](http://arxiv.org/pdf/2506.05780v1)**

> **作者:** Meng Fan; Yifan Zuo; Patrick Blaes; Harley Montgomery; Subhasis Das
>
> **备注:** This paper has been accepted by CVPR 2025 Precognition Workshop
>
> **摘要:** Sensor fusion is crucial for a performant and robust Perception system in autonomous vehicles, but sensor staleness, where data from different sensors arrives with varying delays, poses significant challenges. Temporal misalignment between sensor modalities leads to inconsistent object state estimates, severely degrading the quality of trajectory predictions that are critical for safety. We present a novel and model-agnostic approach to address this problem via (1) a per-point timestamp offset feature (for LiDAR and radar both relative to camera) that enables fine-grained temporal awareness in sensor fusion, and (2) a data augmentation strategy that simulates realistic sensor staleness patterns observed in deployed vehicles. Our method is integrated into a perspective-view detection model that consumes sensor data from multiple LiDARs, radars and cameras. We demonstrate that while a conventional model shows significant regressions when one sensor modality is stale, our approach reaches consistently good performance across both synchronized and stale conditions.
>
---
#### [new 009] HAVIR: HierArchical Vision to Image Reconstruction using CLIP-Guided Versatile Diffusion
- **分类: cs.CV; cs.AI; I.2**

- **简介: 该论文属于脑活动到图像重建任务，旨在解决复杂视觉刺激准确恢复难的问题。通过提出HAVIR模型，结合AutoKL Adapter和CLIP Adapter提取拓扑结构与语义信息，并利用Versatile Diffusion生成图像，有效提升了重建效果。**

- **链接: [http://arxiv.org/pdf/2506.06035v1](http://arxiv.org/pdf/2506.06035v1)**

> **作者:** Shiyi Zhang; Dong Liang; Hairong Zheng; Yihang Zhou
>
> **备注:** 15 pages, 6 figures, 3 tabs
>
> **摘要:** Reconstructing visual information from brain activity bridges the gap between neuroscience and computer vision. Even though progress has been made in decoding images from fMRI using generative models, a challenge remains in accurately recovering highly complex visual stimuli. This difficulty stems from their elemental density and diversity, sophisticated spatial structures, and multifaceted semantic information. To address these challenges, we propose HAVIR that contains two adapters: (1) The AutoKL Adapter transforms fMRI voxels into a latent diffusion prior, capturing topological structures; (2) The CLIP Adapter converts the voxels to CLIP text and image embeddings, containing semantic information. These complementary representations are fused by Versatile Diffusion to generate the final reconstructed image. To extract the most essential semantic information from complex scenarios, the CLIP Adapter is trained with text captions describing the visual stimuli and their corresponding semantic images synthesized from these captions. The experimental results demonstrate that HAVIR effectively reconstructs both structural features and semantic information of visual stimuli even in complex scenarios, outperforming existing models.
>
---
#### [new 010] You Only Estimate Once: Unified, One-stage, Real-Time Category-level Articulated Object 6D Pose Estimation for Robotic Grasping
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人抓取中的类别级关节物体6D姿态估计任务。它旨在解决现有方法计算成本高、实时性差的问题。论文提出YOEO方法，通过单阶段网络同时输出实例分割和NPCS表示，实现端到端的高效姿态估计。**

- **链接: [http://arxiv.org/pdf/2506.05719v1](http://arxiv.org/pdf/2506.05719v1)**

> **作者:** Jingshun Huang; Haitao Lin; Tianyu Wang; Yanwei Fu; Yu-Gang Jiang; Xiangyang Xue
>
> **备注:** To appear in ICRA 2025
>
> **摘要:** This paper addresses the problem of category-level pose estimation for articulated objects in robotic manipulation tasks. Recent works have shown promising results in estimating part pose and size at the category level. However, these approaches primarily follow a complex multi-stage pipeline that first segments part instances in the point cloud and then estimates the Normalized Part Coordinate Space (NPCS) representation for 6D poses. These approaches suffer from high computational costs and low performance in real-time robotic tasks. To address these limitations, we propose YOEO, a single-stage method that simultaneously outputs instance segmentation and NPCS representations in an end-to-end manner. We use a unified network to generate point-wise semantic labels and centroid offsets, allowing points from the same part instance to vote for the same centroid. We further utilize a clustering algorithm to distinguish points based on their estimated centroid distances. Finally, we first separate the NPCS region of each instance. Then, we align the separated regions with the real point cloud to recover the final pose and size. Experimental results on the GAPart dataset demonstrate the pose estimation capabilities of our proposed single-shot method. We also deploy our synthetically-trained model in a real-world setting, providing real-time visual feedback at 200Hz, enabling a physical Kinova robot to interact with unseen articulated objects. This showcases the utility and effectiveness of our proposed method.
>
---
#### [new 011] ChronoTailor: Harnessing Attention Guidance for Fine-Grained Video Virtual Try-On
- **分类: cs.CV**

- **简介: 该论文属于视频虚拟试穿任务，旨在解决生成服装细节保留不足和时序不连贯的问题。论文提出ChronoTailor框架，通过时空注意力机制实现精细编辑与连续特征融合，并引入新数据集StyleDress，显著提升了效果。**

- **链接: [http://arxiv.org/pdf/2506.05858v1](http://arxiv.org/pdf/2506.05858v1)**

> **作者:** Jinjuan Wang; Wenzhang Sun; Ming Li; Yun Zheng; Fanyao Li; Zhulin Tao; Donglin Di; Hao Li; Wei Chen; Xianglin Huang
>
> **摘要:** Video virtual try-on aims to seamlessly replace the clothing of a person in a source video with a target garment. Despite significant progress in this field, existing approaches still struggle to maintain continuity and reproduce garment details. In this paper, we introduce ChronoTailor, a diffusion-based framework that generates temporally consistent videos while preserving fine-grained garment details. By employing a precise spatio-temporal attention mechanism to guide the integration of fine-grained garment features, ChronoTailor achieves robust try-on performance. First, ChronoTailor leverages region-aware spatial guidance to steer the evolution of spatial attention and employs an attention-driven temporal feature fusion mechanism to generate more continuous temporal features. This dual approach not only enables fine-grained local editing but also effectively mitigates artifacts arising from video dynamics. Second, ChronoTailor integrates multi-scale garment features to preserve low-level visual details and incorporates a garment-pose feature alignment to ensure temporal continuity during dynamic motion. Additionally, we collect StyleDress, a new dataset featuring intricate garments, varied environments, and diverse poses, offering advantages over existing public datasets, and will be publicly available for research. Extensive experiments show that ChronoTailor maintains spatio-temporal continuity and preserves garment details during motion, significantly outperforming previous methods.
>
---
#### [new 012] A Neural Network Model of Spatial and Feature-Based Attention
- **分类: cs.CV; cs.CE**

- **简介: 该论文属于计算机视觉与认知科学交叉任务，旨在探索人类视觉注意力机制。论文设计了一个受人类视觉注意力启发的神经网络模型，通过两个网络协作处理复杂任务，发现模型学习到的注意力模式与人类空间和特征注意力相似，为研究人类认知提供了新方向。**

- **链接: [http://arxiv.org/pdf/2506.05487v1](http://arxiv.org/pdf/2506.05487v1)**

> **作者:** Ruoyang Hu; Robert A. Jacobs
>
> **备注:** 6 pages, 9 figures
>
> **摘要:** Visual attention is a mechanism closely intertwined with vision and memory. Top-down information influences visual processing through attention. We designed a neural network model inspired by aspects of human visual attention. This model consists of two networks: one serves as a basic processor performing a simple task, while the other processes contextual information and guides the first network through attention to adapt to more complex tasks. After training the model and visualizing the learned attention response, we discovered that the model's emergent attention patterns corresponded to spatial and feature-based attention. This similarity between human visual attention and attention in computer vision suggests a promising direction for studying human cognition using neural network models.
>
---
#### [new 013] FontAdapter: Instant Font Adaptation in Visual Text Generation
- **分类: cs.CV**

- **简介: 该论文属于视觉文本生成任务，旨在解决现有方法在实时定制未见过字体时计算成本高的问题。作者提出了FontAdapter框架，通过参考字形图像实现秒级字体适配，采用两阶段课程学习提升对新字体的泛化能力，并构建了合成数据集支持训练，实现了高质量、无需微调的字体定制与编辑功能。**

- **链接: [http://arxiv.org/pdf/2506.05843v1](http://arxiv.org/pdf/2506.05843v1)**

> **作者:** Myungkyu Koo; Subin Kim; Sangkyung Kwak; Jaehyun Nam; Seojin Kim; Jinwoo Shin
>
> **备注:** Project page: https://fontadapter.github.io/
>
> **摘要:** Text-to-image diffusion models have significantly improved the seamless integration of visual text into diverse image contexts. Recent approaches further improve control over font styles through fine-tuning with predefined font dictionaries. However, adapting unseen fonts outside the preset is computationally expensive, often requiring tens of minutes, making real-time customization impractical. In this paper, we present FontAdapter, a framework that enables visual text generation in unseen fonts within seconds, conditioned on a reference glyph image. To this end, we find that direct training on font datasets fails to capture nuanced font attributes, limiting generalization to new glyphs. To overcome this, we propose a two-stage curriculum learning approach: FontAdapter first learns to extract font attributes from isolated glyphs and then integrates these styles into diverse natural backgrounds. To support this two-stage training scheme, we construct synthetic datasets tailored to each stage, leveraging large-scale online fonts effectively. Experiments demonstrate that FontAdapter enables high-quality, robust font customization across unseen fonts without additional fine-tuning during inference. Furthermore, it supports visual text editing, font style blending, and cross-lingual font transfer, positioning FontAdapter as a versatile framework for font customization tasks.
>
---
#### [new 014] MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks
- **分类: cs.CV; I.4.9**

- **简介: 该论文属于安全评估任务，旨在解决现有CAPTCHA方案缺乏统一、大规模多模评估基准的问题。作者构建了MCA-Bench，整合多种CAPTCHA类型，基于视觉-语言模型进行微调，实现跨模态安全性评估，并提出设计原则与未来挑战。**

- **链接: [http://arxiv.org/pdf/2506.05982v1](http://arxiv.org/pdf/2506.05982v1)**

> **作者:** Zonglin Wu; Yule Xue; Xin Wei; Yiren Song
>
> **备注:** 31 pages, 8 figures
>
> **摘要:** As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.
>
---
#### [new 015] PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于3D生成任务，旨在根据单张RGB图像生成结构化的多部件3D网格。现有方法生成整体形状或依赖分割，而PartCrafter提出统一的生成架构，无需预分割输入。论文创新包括：组合隐空间与分层注意力机制，并构建了新数据集以支持部件级监督。实验表明其在生成可分解3D网格方面优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.05573v1](http://arxiv.org/pdf/2506.05573v1)**

> **作者:** Yuchen Lin; Chenguo Lin; Panwang Pan; Honglei Yan; Yiqiang Feng; Yadong Mu; Katerina Fragkiadaki
>
> **备注:** Project Page: https://wgsxm.github.io/projects/partcrafter/
>
> **摘要:** We introduce PartCrafter, the first structured 3D generative model that jointly synthesizes multiple semantically meaningful and geometrically distinct 3D meshes from a single RGB image. Unlike existing methods that either produce monolithic 3D shapes or follow two-stage pipelines, i.e., first segmenting an image and then reconstructing each segment, PartCrafter adopts a unified, compositional generation architecture that does not rely on pre-segmented inputs. Conditioned on a single image, it simultaneously denoises multiple 3D parts, enabling end-to-end part-aware generation of both individual objects and complex multi-object scenes. PartCrafter builds upon a pretrained 3D mesh diffusion transformer (DiT) trained on whole objects, inheriting the pretrained weights, encoder, and decoder, and introduces two key innovations: (1) A compositional latent space, where each 3D part is represented by a set of disentangled latent tokens; (2) A hierarchical attention mechanism that enables structured information flow both within individual parts and across all parts, ensuring global coherence while preserving part-level detail during generation. To support part-level supervision, we curate a new dataset by mining part-level annotations from large-scale 3D object datasets. Experiments show that PartCrafter outperforms existing approaches in generating decomposable 3D meshes, including parts that are not directly visible in input images, demonstrating the strength of part-aware generative priors for 3D understanding and synthesis. Code and training data will be released.
>
---
#### [new 016] Technical Report for Egocentric Mistake Detection for the HoloAssist Challenge
- **分类: cs.CV**

- **简介: 该论文属于在线错误检测任务，旨在实时识别工业或教育场景中的操作错误。研究提出了一种结合程序性与执行性错误检测的框架，并利用大语言模型生成反馈。实验验证了方法在HoloAssist数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2506.06174v1](http://arxiv.org/pdf/2506.06174v1)**

> **作者:** Constantin Patsch; Marsil Zakour; Yuankai Wu; Eckehard Steinbach
>
> **摘要:** In this report, we address the task of online mistake detection, which is vital in domains like industrial automation and education, where real-time video analysis allows human operators to correct errors as they occur. While previous work focuses on procedural errors involving action order, broader error types must be addressed for real-world use. We introduce an online mistake detection framework that handles both procedural and execution errors (e.g., motor slips or tool misuse). Upon detecting an error, we use a large language model (LLM) to generate explanatory feedback. Experiments on the HoloAssist benchmark confirm the effectiveness of our approach, where our approach is placed second on the mistake detection task.
>
---
#### [new 017] STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决高分辨率图像合成中生成质量与可扩展性的问题。作者提出了STARFlow模型，结合了归一化流与自回归Transformer的优势，并通过深度-浅层设计、预训练自编码器的潜在空间建模及新引导算法，实现了高效且高质量的图像生成，首次在该规模和分辨率上成功应用归一化流方法。**

- **链接: [http://arxiv.org/pdf/2506.06276v1](http://arxiv.org/pdf/2506.06276v1)**

> **作者:** Jiatao Gu; Tianrong Chen; David Berthelot; Huangjie Zheng; Yuyang Wang; Ruixiang Zhang; Laurent Dinh; Miguel Angel Bautista; Josh Susskind; Shuangfei Zhai
>
> **备注:** TLDR: We show for the first time that normalizing flows can be scaled for high-resolution and text-conditioned image synthesis
>
> **摘要:** We present STARFlow, a scalable generative model based on normalizing flows that achieves strong performance in high-resolution image synthesis. The core of STARFlow is Transformer Autoregressive Flow (TARFlow), which combines the expressive power of normalizing flows with the structured modeling capabilities of Autoregressive Transformers. We first establish the theoretical universality of TARFlow for modeling continuous distributions. Building on this foundation, we introduce several key architectural and algorithmic innovations to significantly enhance scalability: (1) a deep-shallow design, wherein a deep Transformer block captures most of the model representational capacity, complemented by a few shallow Transformer blocks that are computationally efficient yet substantially beneficial; (2) modeling in the latent space of pretrained autoencoders, which proves more effective than direct pixel-level modeling; and (3) a novel guidance algorithm that significantly boosts sample quality. Crucially, our model remains an end-to-end normalizing flow, enabling exact maximum likelihood training in continuous spaces without discretization. STARFlow achieves competitive performance in both class-conditional and text-conditional image generation tasks, approaching state-of-the-art diffusion models in sample quality. To our knowledge, this work is the first successful demonstration of normalizing flows operating effectively at this scale and resolution.
>
---
#### [new 018] Domain Adaptation in Agricultural Image Analysis: A Comprehensive Review from Shallow Models to Deep Learning
- **分类: cs.CV**

- **简介: 该论文属于农业图像分析中的领域自适应（Domain Adaptation, DA）任务，旨在解决模型在不同农业环境（如区域、季节）间泛化能力差的问题。论文系统回顾了从浅层模型到深度学习的DA方法，特别关注对抗学习方法，并分析其在作物健康监测、病虫害检测等任务中的应用效果，同时讨论了公开数据集的优缺点，为未来研究提供指导框架。**

- **链接: [http://arxiv.org/pdf/2506.05972v1](http://arxiv.org/pdf/2506.05972v1)**

> **作者:** Xing Hu; Siyuan Chen; Dawei Zhang
>
> **摘要:** With the increasing use of computer vision in agriculture, image analysis has become crucial for tasks like crop health monitoring and pest detection. However, significant domain shifts between source and target domains-due to environmental differences, crop types, and data acquisition methods-pose challenges. These domain gaps limit the ability of models to generalize across regions, seasons, and complex agricultural environments. This paper explores how Domain Adaptation (DA) techniques can address these challenges, focusing on their role in enhancing the cross-domain transferability of agricultural image analysis. DA has gained attention in agricultural vision tasks due to its potential to mitigate domain heterogeneity. The paper systematically reviews recent advances in DA for agricultural imagery, particularly its practical applications in complex agricultural environments. We examine the key drivers for adopting DA in agriculture, such as limited labeled data, weak model transferability, and dynamic environmental conditions. We also discuss its use in crop health monitoring, pest detection, and fruit recognition, highlighting improvements in performance across regions and seasons. The paper categorizes DA methods into shallow and deep learning models, with further divisions into supervised, semi-supervised, and unsupervised approaches. A special focus is given to adversarial learning-based DA methods, which have shown great promise in challenging agricultural scenarios. Finally, we review key public datasets in agricultural imagery, analyzing their value and limitations in DA research. This review provides a comprehensive framework for researchers, offering insights into current research gaps and supporting the advancement of DA methods in agricultural image analysis.
>
---
#### [new 019] Self-supervised One-Stage Learning for RF-based Multi-Person Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于多人体姿态估计（MPPE）任务，旨在解决现有射频（RF）方法在复杂场景下精度和效率不足的问题。作者提出了一种基于原始RF信号的轻量级单阶段模型，并引入新的自监督学习方法，通过子组信号预测被掩码部分的潜在表示，提升了准确率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.05420v1](http://arxiv.org/pdf/2506.05420v1)**

> **作者:** Seunghwan Shin; Yusung Kim
>
> **备注:** CIKM 2024
>
> **摘要:** In the field of Multi-Person Pose Estimation (MPPE), Radio Frequency (RF)-based methods can operate effectively regardless of lighting conditions and obscured line-of-sight situations. Existing RF-based MPPE methods typically involve either 1) converting RF signals into heatmap images through complex preprocessing, or 2) applying a deep embedding network directly to raw RF signals. The first approach, while delivering decent performance, is computationally intensive and time-consuming. The second method, though simpler in preprocessing, results in lower MPPE accuracy and generalization performance. This paper proposes an efficient and lightweight one-stage MPPE model based on raw RF signals. By sub-grouping RF signals and embedding them using a shared single-layer CNN followed by multi-head attention, this model outperforms previous methods that embed all signals at once through a large and deep CNN. Additionally, we propose a new self-supervised learning (SSL) method that takes inputs from both one unmasked subgroup and the remaining masked subgroups to predict the latent representations of the masked data. Empirical results demonstrate that our model improves MPPE accuracy by up to 15 in PCKh@0.5 compared to previous methods using raw RF signals. Especially, the proposed SSL method has shown to significantly enhance performance improvements when placed in new locations or in front of obstacles at RF antennas, contributing to greater performance gains as the number of people increases. Our code and dataset is open at Github. https://github.com/sshnan7/SOSPE .
>
---
#### [new 020] Robustness Evaluation for Video Models with Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视频模型鲁棒性评估任务，旨在解决视频分类模型在时空维度上的脆弱性问题。作者提出了一种基于多智能体强化学习的攻击方法，协同定位敏感时空区域，生成低扰动、视觉不可察的对抗样本。方法在Lp度量和查询次数上优于现有技术，并支持定制化失真类型，提升了评估实用性。实验验证了其在HMDB-51和UCF-101数据集上对4种主流动作识别模型的有效性。**

- **链接: [http://arxiv.org/pdf/2506.05431v1](http://arxiv.org/pdf/2506.05431v1)**

> **作者:** Ashwin Ramesh Babu; Sajad Mousavi; Vineet Gundecha; Sahand Ghorbanpour; Avisek Naug; Antonio Guillen; Ricardo Luna Gutierrez; Soumyendu Sarkar
>
> **备注:** Accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) 2025
>
> **摘要:** Evaluating the robustness of Video classification models is very challenging, specifically when compared to image-based models. With their increased temporal dimension, there is a significant increase in complexity and computational cost. One of the key challenges is to keep the perturbations to a minimum to induce misclassification. In this work, we propose a multi-agent reinforcement learning approach (spatial and temporal) that cooperatively learns to identify the given video's sensitive spatial and temporal regions. The agents consider temporal coherence in generating fine perturbations, leading to a more effective and visually imperceptible attack. Our method outperforms the state-of-the-art solutions on the Lp metric and the average queries. Our method enables custom distortion types, making the robustness evaluation more relevant to the use case. We extensively evaluate 4 popular models for video action recognition on two popular datasets, HMDB-51 and UCF-101.
>
---
#### [new 021] MoralCLIP: Contrastive Alignment of Vision-and-Language Representations with Moral Foundations Theory
- **分类: cs.CV**

- **简介: 该论文属于多模态语义理解任务，旨在解决现有视觉-语言模型缺乏道德维度理解的问题。作者提出MoralCLIP，通过融合道德基础理论（MFT），将视觉与文本中的道德线索统一表示，提升模型对道德内容的认知与跨模态对齐能力。**

- **链接: [http://arxiv.org/pdf/2506.05696v1](http://arxiv.org/pdf/2506.05696v1)**

> **作者:** Ana Carolina Condez; Diogo Tavares; João Magalhães
>
> **摘要:** Recent advances in vision-language models have enabled rich semantic understanding across modalities. However, these encoding methods lack the ability to interpret or reason about the moral dimensions of content-a crucial aspect of human cognition. In this paper, we address this gap by introducing MoralCLIP, a novel embedding representation method that extends multimodal learning with explicit moral grounding based on Moral Foundations Theory (MFT). Our approach integrates visual and textual moral cues into a unified embedding space, enabling cross-modal moral alignment. MoralCLIP is grounded on the multi-label dataset Social-Moral Image Database to identify co-occurring moral foundations in visual content. For MoralCLIP training, we design a moral data augmentation strategy to scale our annotated dataset to 15,000 image-text pairs labeled with MFT-aligned dimensions. Our results demonstrate that explicit moral supervision improves both unimodal and multimodal understanding of moral content, establishing a foundation for morally-aware AI systems capable of recognizing and aligning with human moral values.
>
---
#### [new 022] FuseUNet: A Multi-Scale Feature Fusion Method for U-like Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决UNet网络中跨尺度特征交互不足、信息融合效率低的问题。作者提出FuseUNet，通过将解码过程建模为初值问题，使用线性多步法实现自适应多尺度特征融合，提升了特征利用率和模型效率。**

- **链接: [http://arxiv.org/pdf/2506.05821v1](http://arxiv.org/pdf/2506.05821v1)**

> **作者:** Quansong He; Xiangde Min; Kaishen Wang; Tao He
>
> **备注:** ICML2025
>
> **摘要:** Medical image segmentation is a critical task in computer vision, with UNet serving as a milestone architecture. The typical component of UNet family is the skip connection, however, their skip connections face two significant limitations: (1) they lack effective interaction between features at different scales, and (2) they rely on simple concatenation or addition operations, which constrain efficient information integration. While recent improvements to UNet have focused on enhancing encoder and decoder capabilities, these limitations remain overlooked. To overcome these challenges, we propose a novel multi-scale feature fusion method that reimagines the UNet decoding process as solving an initial value problem (IVP), treating skip connections as discrete nodes. By leveraging principles from the linear multistep method, we propose an adaptive ordinary differential equation method to enable effective multi-scale feature fusion. Our approach is independent of the encoder and decoder architectures, making it adaptable to various U-Net-like networks. Experiments on ACDC, KiTS2023, MSD brain tumor, and ISIC2017/2018 skin lesion segmentation datasets demonstrate improved feature utilization, reduced network parameters, and maintained high performance. The code is available at https://github.com/nayutayuki/FuseUNet.
>
---
#### [new 023] Investigating the Relationship between Weighted Figure of Merit and Rosin's Measure
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决数字边界多边形逼近的评估问题。作者通过理论、实验和统计分析，研究加权品质因数与Rosin度量之间的关系，发现二者在理论上无关且统计上不相关，因此不能互相替代。**

- **链接: [http://arxiv.org/pdf/2506.05749v1](http://arxiv.org/pdf/2506.05749v1)**

> **作者:** Bimal Kumar Ray
>
> **摘要:** Many studies had been conducted to solve the problem of approximating a digital boundary by piece straight-line segments for further processing required in computer vision applications. The authors of these studies compared their schemes to determine the best one. The initial measure used to assess the goodness of a polygonal approximation was figure of merit. Later, it was pointed out that this measure was not an appropriate metric for a valid reason and this is why Rosin - through mathematical analysis - introduced a measure called merit. However, this measure involves optimal scheme of polygonal approximation and so it is time-consuming to compute it to assess the goodness of an approximation. This led many researchers to use weighted figure of merit as a substitute for Rosin's measure to compare among sub-optimal schemes. An attempt is made in this communication to investigate whether the two measures - weighted figure of merit and Rosin's measure - are related so that one can be used instead of the other and towards this end theoretical analysis, experimental investigation and statistical analysis are carried out. The mathematical formula for weighted figure of merit and Rosin's measure are analyzed and through proof of theorems it is found that the two measures are independent of each other theoretically. The graphical analysis of experiments carried out using public dataset supports theoretical analysis. The statistical analysis using Pearson's correlation coefficient also establishes that the two measures are uncorrelated. This analysis leads one to conclude that if a sub-optimal scheme is found to be better (worse) than some other sub-optimal scheme as indicated by Rosin's measure then the same conclusion cannot be drawn using weighted figure of merit and so one cannot use weighted figure of merit instead of Rosin's measure.
>
---
#### [new 024] VoxelSplat: Dynamic Gaussian Splatting as an Effective Loss for Occupancy and Flow Prediction
- **分类: cs.CV**

- **简介: 论文提出VoxelSplat，用于基于相机的3D语义和场景流预测任务。解决动态环境中遮挡和不平衡带来的挑战。通过3D高斯点云投影提供2D监督，并利用相邻帧标签自监督学习场景流。方法可集成到现有模型中，提升性能且不增加推理时间。**

- **链接: [http://arxiv.org/pdf/2506.05563v1](http://arxiv.org/pdf/2506.05563v1)**

> **作者:** Ziyue Zhu; Shenlong Wang; Jin Xie; Jiang-jiang Liu; Jingdong Wang; Jian Yang
>
> **备注:** Accepted by CVPR 2025 Project Page: https://zzy816.github.io/VoxelSplat-Demo/
>
> **摘要:** Recent advancements in camera-based occupancy prediction have focused on the simultaneous prediction of 3D semantics and scene flow, a task that presents significant challenges due to specific difficulties, e.g., occlusions and unbalanced dynamic environments. In this paper, we analyze these challenges and their underlying causes. To address them, we propose a novel regularization framework called VoxelSplat. This framework leverages recent developments in 3D Gaussian Splatting to enhance model performance in two key ways: (i) Enhanced Semantics Supervision through 2D Projection: During training, our method decodes sparse semantic 3D Gaussians from 3D representations and projects them onto the 2D camera view. This provides additional supervision signals in the camera-visible space, allowing 2D labels to improve the learning of 3D semantics. (ii) Scene Flow Learning: Our framework uses the predicted scene flow to model the motion of Gaussians, and is thus able to learn the scene flow of moving objects in a self-supervised manner using the labels of adjacent frames. Our method can be seamlessly integrated into various existing occupancy models, enhancing performance without increasing inference time. Extensive experiments on benchmark datasets demonstrate the effectiveness of VoxelSplat in improving the accuracy of both semantic occupancy and scene flow estimation. The project page and codes are available at https://zzy816.github.io/VoxelSplat-Demo/.
>
---
#### [new 025] High Throughput Event Filtering: The Interpolation-based DIF Algorithm Hardware Architecture
- **分类: cs.CV**

- **简介: 论文提出了一种基于插值的DIF算法硬件架构，用于事件视觉中的噪声过滤任务。为解决事件传感器数据流中因光照和温度变化引起的噪声问题，作者设计了可在FPGA芯片上运行的DIF滤波器架构，并发布了一个高分辨率事件数据集用于评估。实验表明该方法在不同分辨率下均实现了超过400MEPS的吞吐量，且AUROC指标表现优异，具备高吞吐量和强适应性优势。**

- **链接: [http://arxiv.org/pdf/2506.05825v1](http://arxiv.org/pdf/2506.05825v1)**

> **作者:** Marcin Kowalczyk; Tomasz Kryjak
>
> **备注:** Accepted in the Microprocessors and Microsystems journal
>
> **摘要:** In recent years, there has been rapid development in the field of event vision. It manifests itself both on the technical side, as better and better event sensors are available, and on the algorithmic side, as more and more applications of this technology are proposed and scientific papers are published. However, the data stream from these sensors typically contains a significant amount of noise, which varies depending on factors such as the degree of illumination in the observed scene or the temperature of the sensor. We propose a hardware architecture of the Distance-based Interpolation with Frequency Weights (DIF) filter and implement it on an FPGA chip. To evaluate the algorithm and compare it with other solutions, we have prepared a new high-resolution event dataset, which we are also releasing to the community. Our architecture achieved a throughput of 403.39 million events per second (MEPS) for a sensor resolution of 1280 x 720 and 428.45 MEPS for a resolution of 640 x 480. The average values of the Area Under the Receiver Operating Characteristic (AUROC) index ranged from 0.844 to 0.999, depending on the dataset, which is comparable to the state-of-the-art filtering solutions, but with much higher throughput and better operation over a wide range of noise levels.
>
---
#### [new 026] Full Conformal Adaptation of Medical Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于医学视觉-语言模型（VLM）任务，旨在提升其在有限样本下的预测可靠性。现有方法依赖于严格的假设，在实际应用中受限。论文提出“全共形适配”框架，结合无需训练的线性探针方法SS-Text，通过少量样本对每个测试点进行适配，提升了预测集效率，同时保持误差控制保证。**

- **链接: [http://arxiv.org/pdf/2506.06076v1](http://arxiv.org/pdf/2506.06076v1)**

> **作者:** Julio Silva-Rodríguez; Leo Fillioux; Paul-Henry Cournède; Maria Vakalopoulou; Stergios Christodoulidis; Ismail Ben Ayed; Jose Dolz
>
> **备注:** IPMI 2025. Code: https://github.com/jusiro/FCA
>
> **摘要:** Vision-language models (VLMs) pre-trained at large scale have shown unprecedented transferability capabilities and are being progressively integrated into medical image analysis. Although its discriminative potential has been widely explored, its reliability aspect remains overlooked. This work investigates their behavior under the increasingly popular split conformal prediction (SCP) framework, which theoretically guarantees a given error level on output sets by leveraging a labeled calibration set. However, the zero-shot performance of VLMs is inherently limited, and common practice involves few-shot transfer learning pipelines, which cannot absorb the rigid exchangeability assumptions of SCP. To alleviate this issue, we propose full conformal adaptation, a novel setting for jointly adapting and conformalizing pre-trained foundation models, which operates transductively over each test data point using a few-shot adaptation set. Moreover, we complement this framework with SS-Text, a novel training-free linear probe solver for VLMs that alleviates the computational cost of such a transductive approach. We provide comprehensive experiments using 3 different modality-specialized medical VLMs and 9 adaptation tasks. Our framework requires exactly the same data as SCP, and provides consistent relative improvements of up to 27% on set efficiency while maintaining the same coverage guarantees.
>
---
#### [new 027] A Compendium of Autonomous Navigation using Object Detection and Tracking in Unmanned Aerial Vehicles
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文综述了基于目标检测与跟踪的无人机自主导航技术。任务是提升无人机在复杂环境中的自主导航能力，解决信号干扰、实时处理等问题。工作包括分析多种算法在灾害管理、交通监控等场景的应用。**

- **链接: [http://arxiv.org/pdf/2506.05378v1](http://arxiv.org/pdf/2506.05378v1)**

> **作者:** Mohit Arora; Pratyush Shukla; Shivali Chopra
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are one of the most revolutionary inventions of 21st century. At the core of a UAV lies the central processing system that uses wireless signals to control their movement. The most popular UAVs are quadcopters that use a set of four motors, arranged as two on either side with opposite spin. An autonomous UAV is called a drone. Drones have been in service in the US army since the 90's for covert missions critical to national security. It would not be wrong to claim that drones make up an integral part of the national security and provide the most valuable service during surveillance operations. While UAVs are controlled using wireless signals, there reside some challenges that disrupt the operation of such vehicles such as signal quality and range, real time processing, human expertise, robust hardware and data security. These challenges can be solved by programming UAVs to be autonomous, using object detection and tracking, through Computer Vision algorithms. Computer Vision is an interdisciplinary field that seeks the use of deep learning to gain a high-level understanding of digital images and videos for the purpose of automating the task of human visual system. Using computer vision, algorithms for detecting and tracking various objects can be developed suitable to the hardware so as to allow real time processing for immediate judgement. This paper attempts to review the various approaches several authors have proposed for the purpose of autonomous navigation of UAVs by through various algorithms of object detection and tracking in real time, for the purpose of applications in various fields such as disaster management, dense area exploration, traffic vehicle surveillance etc.
>
---
#### [new 028] HMVLM: Multistage Reasoning-Enhanced Vision-Language Model for Long-Tailed Driving Scenarios
- **分类: cs.CV; cs.AI**

- **简介: 论文提出HMVLM，一种用于自动驾驶的多阶段推理增强型视觉-语言模型，旨在解决长尾驾驶场景中的复杂决策问题。通过结合快速控制与慢速规划架构，引入选择性五视图提示、多阶段推理链和样条轨迹后处理，提升驾驶决策质量与平稳性。在Waymo数据集上验证了其优越性能。**

- **链接: [http://arxiv.org/pdf/2506.05883v1](http://arxiv.org/pdf/2506.05883v1)**

> **作者:** Daming Wang; Yuhao Song; Zijian He; Kangliang Chen; Xing Pan; Lu Deng; Weihao Gu
>
> **备注:** WOD Vision-based End-to-End Driving Challenge
>
> **摘要:** We present HaoMo Vision-Language Model (HMVLM), an end-to-end driving framework that implements the slow branch of a cognitively inspired fast-slow architecture. A fast controller outputs low-level steering, throttle, and brake commands, while a slow planner-a large vision-language model-generates high-level intents such as "yield to pedestrian" or "merge after the truck" without compromising latency. HMVLM introduces three upgrades: (1) selective five-view prompting with an embedded 4s history of ego kinematics, (2) multi-stage chain-of-thought (CoT) prompting that enforces a Scene Understanding -> Driving Decision -> Trajectory Inference reasoning flow, and (3) spline-based trajectory post-processing that removes late-stage jitter and sharp turns. Trained on the Waymo Open Dataset, these upgrades enable HMVLM to achieve a Rater Feedback Score (RFS) of 7.7367, securing 2nd place in the 2025 Waymo Vision-based End-to-End (E2E) Driving Challenge and surpassing the public baseline by 2.77%.
>
---
#### [new 029] FRAME: Pre-Training Video Feature Representations via Anticipation and Memory
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决现有视频编码器在时空一致性与密集预测上的不足。作者提出FRAME框架，通过自监督学习结合图像模型DINO和CLIP，提升视频的时空特征表达能力，增强细粒度视觉对应和语义对齐，适用于多种下游任务。**

- **链接: [http://arxiv.org/pdf/2506.05543v1](http://arxiv.org/pdf/2506.05543v1)**

> **作者:** Sethuraman TV; Savya Khosla; Vignesh Srinivasakumar; Jiahui Huang; Seoung Wug Oh; Simon Jenni; Derek Hoiem; Joon-Young Lee
>
> **摘要:** Dense video prediction tasks, such as object tracking and semantic segmentation, require video encoders that generate temporally consistent, spatially dense features for every frame. However, existing approaches fall short: image encoders like DINO or CLIP lack temporal awareness, while video models such as VideoMAE underperform compared to image encoders on dense prediction tasks. We address this gap with FRAME, a self-supervised video frame encoder tailored for dense video understanding. FRAME learns to predict current and future DINO patch features from past and present RGB frames, leading to spatially precise and temporally coherent representations. To our knowledge, FRAME is the first video encoder to leverage image-based models for dense prediction while outperforming them on tasks requiring fine-grained visual correspondence. As an auxiliary capability, FRAME aligns its class token with CLIP's semantic space, supporting language-driven tasks such as video classification. We evaluate FRAME across six dense prediction tasks on seven datasets, where it consistently outperforms image encoders and existing self-supervised video models. Despite its versatility, FRAME maintains a compact architecture suitable for a range of downstream applications.
>
---
#### [new 030] GazeNLQ @ Ego4D Natural Language Queries Challenge 2025
- **分类: cs.CV**

- **简介: 该论文属于视频内容检索任务，旨在解决基于自然语言查询的视频片段定位问题。作者提出了GazeNLQ方法，利用注视信息增强视频表示，提升定位准确率，并通过对比学习预训练优化注视估计，最终在Ego4D数据集上取得良好效果。**

- **链接: [http://arxiv.org/pdf/2506.05782v1](http://arxiv.org/pdf/2506.05782v1)**

> **作者:** Wei-Cheng Lin; Chih-Ming Lien; Chen Lo; Chia-Hung Yeh
>
> **摘要:** This report presents our solution to the Ego4D Natural Language Queries (NLQ) Challenge at CVPR 2025. Egocentric video captures the scene from the wearer's perspective, where gaze serves as a key non-verbal communication cue that reflects visual attention and offer insights into human intention and cognition. Motivated by this, we propose a novel approach, GazeNLQ, which leverages gaze to retrieve video segments that match given natural language queries. Specifically, we introduce a contrastive learning-based pretraining strategy for gaze estimation directly from video. The estimated gaze is used to augment video representations within proposed model, thereby enhancing localization accuracy. Experimental results show that GazeNLQ achieves R1@IoU0.3 and R1@IoU0.5 scores of 27.82 and 18.68, respectively. Our code is available at https://github.com/stevenlin510/GazeNLQ.
>
---
#### [new 031] Bootstrapping World Models from Dynamics Models in Multimodal Foundation Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态基础模型任务，旨在解决视觉-语言模型缺乏现实世界模型的问题。通过动力学模型引导世界模型构建，采用合成数据训练与推理验证策略，实现更优的以动作为中心的图像编辑效果，在Aurora-Bench评测中表现优越。**

- **链接: [http://arxiv.org/pdf/2506.06006v1](http://arxiv.org/pdf/2506.06006v1)**

> **作者:** Yifu Qiu; Yftah Ziser; Anna Korhonen; Shay B. Cohen; Edoardo M. Ponti
>
> **摘要:** To what extent do vision-and-language foundation models possess a realistic world model (observation $\times$ action $\rightarrow$ observation) and a dynamics model (observation $\times$ observation $\rightarrow$ action), when actions are expressed through language? While open-source foundation models struggle with both, we find that fine-tuning them to acquire a dynamics model through supervision is significantly easier than acquiring a world model. In turn, dynamics models can be used to bootstrap world models through two main strategies: 1) weakly supervised learning from synthetic data and 2) inference time verification. Firstly, the dynamics model can annotate actions for unlabelled pairs of video frame observations to expand the training data. We further propose a new objective, where image tokens in observation pairs are weighted by their importance, as predicted by a recognition model. Secondly, the dynamics models can assign rewards to multiple samples of the world model to score them, effectively guiding search at inference time. We evaluate the world models resulting from both strategies through the task of action-centric image editing on Aurora-Bench. Our best model achieves a performance competitive with state-of-the-art image editing models, improving on them by a margin of $15\%$ on real-world subsets according to GPT4o-as-judge, and achieving the best average human evaluation across all subsets of Aurora-Bench.
>
---
#### [new 032] Movie Facts and Fibs (MF$^2$): A Benchmark for Long Movie Understanding
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视频理解任务，旨在解决当前模型对长视频内容缺乏深层理解的问题。作者构建了新基准MF²，包含50部长电影及850对真假陈述，评估模型对叙事核心要素的理解与推理能力，发现现有模型表现远不如人类。**

- **链接: [http://arxiv.org/pdf/2506.06275v1](http://arxiv.org/pdf/2506.06275v1)**

> **作者:** Emmanouil Zaranis; António Farinhas; Saul Santos; Beatriz Canaverde; Miguel Moura Ramos; Aditya K Surikuchi; André Viveiros; Baohao Liao; Elena Bueno-Benito; Nithin Sivakumaran; Pavlo Vasylenko; Shoubin Yu; Sonal Sannigrahi; Wafaa Mohammed; Ben Peters; Danae Sánchez Villegas; Elias Stengel-Eskin; Giuseppe Attanasio; Jaehong Yoon; Stella Frank; Alessandro Suglia; Chrysoula Zerva; Desmond Elliott; Mariella Dimiccoli; Mohit Bansal; Oswald Lanz; Raffaella Bernardi; Raquel Fernández; Sandro Pezzelle; Vlad Niculae; André F. T. Martins
>
> **备注:** Under Review
>
> **摘要:** Despite recent progress in vision-language models (VLMs), holistic understanding of long-form video content remains a significant challenge, partly due to limitations in current benchmarks. Many focus on peripheral, ``needle-in-a-haystack'' details, encouraging context-insensitive retrieval over deep comprehension. Others rely on large-scale, semi-automatically generated questions (often produced by language models themselves) that are easier for models to answer but fail to reflect genuine understanding. In this paper, we introduce MF$^2$, a new benchmark for evaluating whether models can comprehend, consolidate, and recall key narrative information from full-length movies (50-170 minutes long). MF$^2$ includes over 50 full-length, open-licensed movies, each paired with manually constructed sets of claim pairs -- one true (fact) and one plausible but false (fib), totalling over 850 pairs. These claims target core narrative elements such as character motivations and emotions, causal chains, and event order, and refer to memorable moments that humans can recall without rewatching the movie. Instead of multiple-choice formats, we adopt a binary claim evaluation protocol: for each pair, models must correctly identify both the true and false claims. This reduces biases like answer ordering and enables a more precise assessment of reasoning. Our experiments demonstrate that both open-weight and closed state-of-the-art models fall well short of human performance, underscoring the relative ease of the task for humans and their superior ability to retain and reason over critical narrative information -- an ability current VLMs lack.
>
---
#### [new 033] State Estimation and Control of Dynamic Systems from High-Dimensional Image Data
- **分类: cs.CV**

- **简介: 该论文属于状态估计与控制任务，旨在解决无法直接获取动态系统真实状态的问题。作者提出一种结合CNN与GRU的神经网络架构，从高维图像数据中学习状态表示，并用于训练DQN强化学习代理，实现系统的实时估计与控制。**

- **链接: [http://arxiv.org/pdf/2506.05375v1](http://arxiv.org/pdf/2506.05375v1)**

> **作者:** Ashik E Rasul; Hyung-Jin Yoon
>
> **摘要:** Accurate state estimation is critical for optimal policy design in dynamic systems. However, obtaining true system states is often impractical or infeasible, complicating the policy learning process. This paper introduces a novel neural architecture that integrates spatial feature extraction using convolutional neural networks (CNNs) and temporal modeling through gated recurrent units (GRUs), enabling effective state representation from sequences of images and corresponding actions. These learned state representations are used to train a reinforcement learning agent with a Deep Q-Network (DQN). Experimental results demonstrate that our proposed approach enables real-time, accurate estimation and control without direct access to ground-truth states. Additionally, we provide a quantitative evaluation methodology for assessing the accuracy of the learned states, highlighting their impact on policy performance and control stability.
>
---
#### [new 034] STSBench: A Spatio-temporal Scenario Benchmark for Multi-modal Large Language Models in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出了STSBench，一个用于自动驾驶中多模态大语言模型的时空场景基准。任务是评估视觉-语言模型（VLMs）在复杂交通场景中的整体理解能力，解决现有模型在时空推理上的不足。工作包括自动挖掘交通场景、生成多选题，并构建包含971个问题的STSnu基准，揭示了当前模型在交通动态推理上的关键缺陷。**

- **链接: [http://arxiv.org/pdf/2506.06218v1](http://arxiv.org/pdf/2506.06218v1)**

> **作者:** Christian Fruhwirth-Reisinger; Dušan Malić; Wei Lin; David Schinagl; Samuel Schulter; Horst Possegger
>
> **备注:** Dataset: https://huggingface.co/datasets/ivc-lrp/STSBench, Code: https://github.com/LRP-IVC/STSBench
>
> **摘要:** We introduce STSBench, a scenario-based framework to benchmark the holistic understanding of vision-language models (VLMs) for autonomous driving. The framework automatically mines pre-defined traffic scenarios from any dataset using ground-truth annotations, provides an intuitive user interface for efficient human verification, and generates multiple-choice questions for model evaluation. Applied to the NuScenes dataset, we present STSnu, the first benchmark that evaluates the spatio-temporal reasoning capabilities of VLMs based on comprehensive 3D perception. Existing benchmarks typically target off-the-shelf or fine-tuned VLMs for images or videos from a single viewpoint and focus on semantic tasks such as object recognition, dense captioning, risk assessment, or scene understanding. In contrast, STSnu evaluates driving expert VLMs for end-to-end driving, operating on videos from multi-view cameras or LiDAR. It specifically assesses their ability to reason about both ego-vehicle actions and complex interactions among traffic participants, a crucial capability for autonomous vehicles. The benchmark features 43 diverse scenarios spanning multiple views and frames, resulting in 971 human-verified multiple-choice questions. A thorough evaluation uncovers critical shortcomings in existing models' ability to reason about fundamental traffic dynamics in complex environments. These findings highlight the urgent need for architectural advances that explicitly model spatio-temporal reasoning. By addressing a core gap in spatio-temporal evaluation, STSBench enables the development of more robust and explainable VLMs for autonomous driving.
>
---
#### [new 035] FocusDiff: Advancing Fine-Grained Text-Image Alignment for Autoregressive Visual Generation through RL
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决现有模型在细粒度图文对齐上的不足。作者提出了FocusDiff方法，通过构建包含细微语义差异的配对数据集，并引入强化学习算法，以提升对视觉标记的精确控制能力，从而改善生成效果。**

- **链接: [http://arxiv.org/pdf/2506.05501v1](http://arxiv.org/pdf/2506.05501v1)**

> **作者:** Kaihang Pan; Wendong Bu; Yuruo Wu; Yang Wu; Kai Shen; Yunfei Li; Hang Zhao; Juncheng Li; Siliang Tang; Yueting Zhuang
>
> **备注:** 15 pages, 8 figures. Project Page: https://focusdiff.github.io/
>
> **摘要:** Recent studies extend the autoregression paradigm to text-to-image generation, achieving performance comparable to diffusion models. However, our new PairComp benchmark -- featuring test cases of paired prompts with similar syntax but different fine-grained semantics -- reveals that existing models struggle with fine-grained text-image alignment thus failing to realize precise control over visual tokens. To address this, we propose FocusDiff, which enhances fine-grained text-image semantic alignment by focusing on subtle differences between similar text-image pairs. We construct a new dataset of paired texts and images with similar overall expressions but distinct local semantics, further introducing a novel reinforcement learning algorithm to emphasize such fine-grained semantic differences for desired image generation. Our approach achieves state-of-the-art performance on existing text-to-image benchmarks and significantly outperforms prior methods on PairComp.
>
---
#### [new 036] O-MaMa @ EgoExo4D Correspondence Challenge: Learning Object Mask Matching between Egocentric and Exocentric Views
- **分类: cs.CV**

- **简介: 该论文属于跨视角分割任务，旨在解决不同视角（第一人称与第三人称）间对象对应问题。方法包括：Mask-Context编码器提取对象特征，跨视角注意力融合多视角信息，对比损失对齐特征空间，并采用难负样本挖掘提升区分能力。**

- **链接: [http://arxiv.org/pdf/2506.06026v1](http://arxiv.org/pdf/2506.06026v1)**

> **作者:** Lorenzo Mur-Labadia; Maria Santos-Villafranca; Alejandro Perez-Yus; Jesus Bermudez-Cameo; Ruben Martinez-Cantin; Jose J. Guerrero
>
> **摘要:** The goal of the correspondence task is to segment specific objects across different views. This technical report re-defines cross-image segmentation by treating it as a mask matching task. Our method consists of: (1) A Mask-Context Encoder that pools dense DINOv2 semantic features to obtain discriminative object-level representations from FastSAM mask candidates, (2) an Ego$\leftrightarrow$Exo Cross-Attention that fuses multi-perspective observations, (3) a Mask Matching contrastive loss that aligns cross-view features in a shared latent space, and (4) a Hard Negative Adjacent Mining strategy to encourage the model to better differentiate between nearby objects.
>
---
#### [new 037] On-the-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images
- **分类: cs.CV**

- **简介: 该论文属于三维重建与视图合成任务，旨在解决大规模无姿态图像的快速新颖视角生成问题。现有方法在姿态估计和优化上耗时较长，尤其面对大场景和宽基线拍摄时效果不佳。论文提出一种实时重建方法，结合快速姿态估计与增量式高斯点采样，实现捕获后即时生成相机姿态与高质量3D高斯点模型，并支持大规模场景的渐进式优化与存储。**

- **链接: [http://arxiv.org/pdf/2506.05558v1](http://arxiv.org/pdf/2506.05558v1)**

> **作者:** Andreas Meuleman; Ishaan Shah; Alexandre Lanvin; Bernhard Kerbl; George Drettakis
>
> **摘要:** Radiance field methods such as 3D Gaussian Splatting (3DGS) allow easy reconstruction from photos, enabling free-viewpoint navigation. Nonetheless, pose estimation using Structure from Motion and 3DGS optimization can still each take between minutes and hours of computation after capture is complete. SLAM methods combined with 3DGS are fast but struggle with wide camera baselines and large scenes. We present an on-the-fly method to produce camera poses and a trained 3DGS immediately after capture. Our method can handle dense and wide-baseline captures of ordered photo sequences and large-scale scenes. To do this, we first introduce fast initial pose estimation, exploiting learned features and a GPU-friendly mini bundle adjustment. We then introduce direct sampling of Gaussian primitive positions and shapes, incrementally spawning primitives where required, significantly accelerating training. These two efficient steps allow fast and robust joint optimization of poses and Gaussian primitives. Our incremental approach handles large-scale scenes by introducing scalable radiance field construction, progressively clustering 3DGS primitives, storing them in anchors, and offloading them from the GPU. Clustered primitives are progressively merged, keeping the required scale of 3DGS at any viewpoint. We evaluate our solution on a variety of datasets and show that our solution can provide on-the-fly processing of all the capture scenarios and scene sizes we target while remaining competitive with other methods that only handle specific capture styles or scene sizes in speed, image quality, or both.
>
---
#### [new 038] Tensor-to-Tensor Models with Fast Iterated Sum Features
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于深度学习模型设计任务，旨在解决高维张量数据处理中计算复杂度高的问题。作者提出了一种新的张量到张量层（FIS层），基于“corner trees”和迭代和的多参数推广，实现线性时间复杂度。该层可用于图像处理，有效减少参数量和计算操作，在分类和异常检测任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.06041v1](http://arxiv.org/pdf/2506.06041v1)**

> **作者:** Joscha Diehl; Rasheed Ibraheem; Leonard Schmitz; Yue Wu
>
> **摘要:** Data in the form of images or higher-order tensors is ubiquitous in modern deep learning applications. Owing to their inherent high dimensionality, the need for subquadratic layers processing such data is even more pressing than for sequence data. We propose a novel tensor-to-tensor layer with linear cost in the input size, utilizing the mathematical gadget of ``corner trees'' from the field of permutation counting. In particular, for order-two tensors, we provide an image-to-image layer that can be plugged into image processing pipelines. On the one hand, our method can be seen as a higher-order generalization of state-space models. On the other hand, it is based on a multiparameter generalization of the signature of iterated integrals (or sums). The proposed tensor-to-tensor concept is used to build a neural network layer called the Fast Iterated Sums (FIS) layer which integrates seamlessly with other layer types. We demonstrate the usability of the FIS layer with both classification and anomaly detection tasks. By replacing some layers of a smaller ResNet architecture with FIS, a similar accuracy (with a difference of only 0.1\%) was achieved in comparison to a larger ResNet while reducing the number of trainable parameters and multi-add operations. The FIS layer was also used to build an anomaly detection model that achieved an average AUROC of 97.3\% on the texture images of the popular MVTec AD dataset. The processing and modelling codes are publicly available at https://github.com/diehlj/fast-iterated-sums.
>
---
#### [new 039] Bridging Perspectives: A Survey on Cross-view Collaborative Intelligence with Egocentric-Exocentric Vision
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决如何结合以自我为中心（第一视角）和外部视角（第三视角）视觉信息，提升机器对动态环境的理解。论文系统综述了两种视角的协同方法，包括数据利用、学习框架及基准数据集，并提出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2506.06253v1](http://arxiv.org/pdf/2506.06253v1)**

> **作者:** Yuping He; Yifei Huang; Guo Chen; Lidong Lu; Baoqi Pei; Jilan Xu; Tong Lu; Yoichi Sato
>
> **摘要:** Perceiving the world from both egocentric (first-person) and exocentric (third-person) perspectives is fundamental to human cognition, enabling rich and complementary understanding of dynamic environments. In recent years, allowing the machines to leverage the synergistic potential of these dual perspectives has emerged as a compelling research direction in video understanding. In this survey, we provide a comprehensive review of video understanding from both exocentric and egocentric viewpoints. We begin by highlighting the practical applications of integrating egocentric and exocentric techniques, envisioning their potential collaboration across domains. We then identify key research tasks to realize these applications. Next, we systematically organize and review recent advancements into three main research directions: (1) leveraging egocentric data to enhance exocentric understanding, (2) utilizing exocentric data to improve egocentric analysis, and (3) joint learning frameworks that unify both perspectives. For each direction, we analyze a diverse set of tasks and relevant works. Additionally, we discuss benchmark datasets that support research in both perspectives, evaluating their scope, diversity, and applicability. Finally, we discuss limitations in current works and propose promising future research directions. By synthesizing insights from both perspectives, our goal is to inspire advancements in video understanding and artificial intelligence, bringing machines closer to perceiving the world in a human-like manner. A GitHub repo of related works can be found at https://github.com/ayiyayi/Awesome-Egocentric-and-Exocentric-Vision.
>
---
#### [new 040] SDS-Net: Shallow-Deep Synergism-detection Network for infrared small target detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于红外小目标检测（IRSTD）任务，旨在解决现有方法忽略浅层与深层特征异构性、跨层级特征融合不足导致的检测性能差和计算成本高的问题。论文提出了SDS-Net网络，通过双分支结构分别建模结构信息与语义信息，并引入自适应特征融合模块，提升检测精度与推理效率。**

- **链接: [http://arxiv.org/pdf/2506.06042v1](http://arxiv.org/pdf/2506.06042v1)**

> **作者:** Taoran Yue; Xiaojin Lu; Jiaxi Cai; Yuanping Chen; Shibing Chu
>
> **备注:** 13 pages,9 figures, Submitted IEEE Transactions on Geoscience and Remote Sensing
>
> **摘要:** Current CNN-based infrared small target detection(IRSTD) methods generally overlook the heterogeneity between shallow and deep features, leading to inefficient collaboration between shallow fine grained structural information and deep high-level semantic representations. Additionally, the dependency relationships and fusion mechanisms across different feature hierarchies lack systematic modeling, which fails to fully exploit the complementarity of multilevel features. These limitations hinder IRSTD performance while incurring substantial computational costs. To address these challenges, this paper proposes a shallow-deep synergistic detection network (SDS-Net) that efficiently models multilevel feature representations to increase both the detection accuracy and computational efficiency in IRSTD tasks. SDS-Net introduces a dual-branch architecture that separately models the structural characteristics and semantic properties of features, effectively preserving shallow spatial details while capturing deep semantic representations, thereby achieving high-precision detection with significantly improved inference speed. Furthermore, the network incorporates an adaptive feature fusion module to dynamically model cross-layer feature correlations, enhancing overall feature collaboration and representation capability. Comprehensive experiments on three public datasets (NUAA-SIRST, NUDT-SIRST, and IRSTD-1K) demonstrate that SDS-Net outperforms state-of-the-art IRSTD methods while maintaining low computational complexity and high inference efficiency, showing superior detection performance and broad application prospects. Our code will be made public at https://github.com/PhysiLearn/SDS-Net.
>
---
#### [new 041] Do Large Vision-Language Models Distinguish between the Actual and Apparent Features of Illusions?
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言理解任务，旨在探究大型视觉语言模型（LVLMs）是否能区分真实与表观特征的错觉。为解决现有研究未区分实际与表观特征的问题，作者构建了一个包含真实与虚假错觉图像的视觉问答数据集，并通过实验发现模型回答可能依赖先验知识而非真实感知。**

- **链接: [http://arxiv.org/pdf/2506.05765v1](http://arxiv.org/pdf/2506.05765v1)**

> **作者:** Taiga Shinozaki; Tomoki Doi; Satoshi Nishida; Hitomi Yanaka
>
> **备注:** To appear in the Proceedings of the 47th Annual Meeting of the Cognitive Science Society (COGSCI 2025)
>
> **摘要:** Humans are susceptible to optical illusions, which serve as valuable tools for investigating sensory and cognitive processes. Inspired by human vision studies, research has begun exploring whether machines, such as large vision language models (LVLMs), exhibit similar susceptibilities to visual illusions. However, studies often have used non-abstract images and have not distinguished actual and apparent features, leading to ambiguous assessments of machine cognition. To address these limitations, we introduce a visual question answering (VQA) dataset, categorized into genuine and fake illusions, along with corresponding control images. Genuine illusions present discrepancies between actual and apparent features, whereas fake illusions have the same actual and apparent features even though they look illusory due to the similar geometric configuration. We evaluate the performance of LVLMs for genuine and fake illusion VQA tasks and investigate whether the models discern actual and apparent features. Our findings indicate that although LVLMs may appear to recognize illusions by correctly answering questions about both feature types, they predict the same answers for both Genuine Illusion and Fake Illusion VQA questions. This suggests that their responses might be based on prior knowledge of illusions rather than genuine visual understanding. The dataset is available at https://github.com/ynklab/FILM
>
---
#### [new 042] Hallucinate, Ground, Repeat: A Framework for Generalized Visual Relationship Detection
- **分类: cs.CV**

- **简介: 该论文属于视觉关系检测任务，旨在解决现有方法依赖固定谓词集、难以泛化到新关系的问题。作者提出一种迭代框架，结合大语言模型生成候选关系，并通过视觉模型进行对齐训练，从而提升对未见谓词的检测能力。**

- **链接: [http://arxiv.org/pdf/2506.05651v1](http://arxiv.org/pdf/2506.05651v1)**

> **作者:** Shanmukha Vellamcheti; Sanjoy Kundu; Sathyanarayanan N. Aakur
>
> **备注:** 22 pages, 9 figures, 5 tables
>
> **摘要:** Understanding relationships between objects is central to visual intelligence, with applications in embodied AI, assistive systems, and scene understanding. Yet, most visual relationship detection (VRD) models rely on a fixed predicate set, limiting their generalization to novel interactions. A key challenge is the inability to visually ground semantically plausible, but unannotated, relationships hypothesized from external knowledge. This work introduces an iterative visual grounding framework that leverages large language models (LLMs) as structured relational priors. Inspired by expectation-maximization (EM), our method alternates between generating candidate scene graphs from detected objects using an LLM (expectation) and training a visual model to align these hypotheses with perceptual evidence (maximization). This process bootstraps relational understanding beyond annotated data and enables generalization to unseen predicates. Additionally, we introduce a new benchmark for open-world VRD on Visual Genome with 21 held-out predicates and evaluate under three settings: seen, unseen, and mixed. Our model outperforms LLM-only, few-shot, and debiased baselines, achieving mean recall (mR@50) of 15.9, 13.1, and 11.7 on predicate classification on these three sets. These results highlight the promise of grounded LLM priors for scalable open-world visual understanding.
>
---
#### [new 043] Aerial Multi-View Stereo via Adaptive Depth Range Inference and Normal Cues
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于三维城市重建任务，旨在解决航拍图像多视角立体匹配中深度范围固定和特征匹配困难的问题。论文提出了ADR-MVS方法，通过自适应深度范围预测、法线引导的代价聚合与深度优化，提升了重建精度和效率。**

- **链接: [http://arxiv.org/pdf/2506.05655v1](http://arxiv.org/pdf/2506.05655v1)**

> **作者:** Yimei Liu; Yakun Ju; Yuan Rao; Hao Fan; Junyu Dong; Feng Gao; Qian Du
>
> **备注:** IEEE TGRS 2025
>
> **摘要:** Three-dimensional digital urban reconstruction from multi-view aerial images is a critical application where deep multi-view stereo (MVS) methods outperform traditional techniques. However, existing methods commonly overlook the key differences between aerial and close-range settings, such as varying depth ranges along epipolar lines and insensitive feature-matching associated with low-detailed aerial images. To address these issues, we propose an Adaptive Depth Range MVS (ADR-MVS), which integrates monocular geometric cues to improve multi-view depth estimation accuracy. The key component of ADR-MVS is the depth range predictor, which generates adaptive range maps from depth and normal estimates using cross-attention discrepancy learning. In the first stage, the range map derived from monocular cues breaks through predefined depth boundaries, improving feature-matching discriminability and mitigating convergence to local optima. In later stages, the inferred range maps are progressively narrowed, ultimately aligning with the cascaded MVS framework for precise depth regression. Moreover, a normal-guided cost aggregation operation is specially devised for aerial stereo images to improve geometric awareness within the cost volume. Finally, we introduce a normal-guided depth refinement module that surpasses existing RGB-guided techniques. Experimental results demonstrate that ADR-MVS achieves state-of-the-art performance on the WHU, LuoJia-MVS, and M\"unchen datasets, while exhibits superior computational complexity.
>
---
#### [new 044] GenIR: Generative Visual Feedback for Mental Image Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 论文提出“心理图像检索”（MIR）任务，旨在通过多轮交互帮助用户根据脑海中的图像寻找目标图像。现有方法依赖抽象语言反馈，不够直观。为此，作者提出GenIR，利用扩散模型生成视觉反馈，使用户能更清晰地调整查询。论文还构建了自动化的多轮MIR数据集，实验表明GenIR优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.06220v1](http://arxiv.org/pdf/2506.06220v1)**

> **作者:** Diji Yang; Minghao Liu; Chung-Hsiang Lo; Yi Zhang; James Davis
>
> **摘要:** Vision-language models (VLMs) have shown strong performance on text-to-image retrieval benchmarks. However, bridging this success to real-world applications remains a challenge. In practice, human search behavior is rarely a one-shot action. Instead, it is often a multi-round process guided by clues in mind, that is, a mental image ranging from vague recollections to vivid mental representations of the target image. Motivated by this gap, we study the task of Mental Image Retrieval (MIR), which targets the realistic yet underexplored setting where users refine their search for a mentally envisioned image through multi-round interactions with an image search engine. Central to successful interactive retrieval is the capability of machines to provide users with clear, actionable feedback; however, existing methods rely on indirect or abstract verbal feedback, which can be ambiguous, misleading, or ineffective for users to refine the query. To overcome this, we propose GenIR, a generative multi-round retrieval paradigm leveraging diffusion-based image generation to explicitly reify the AI system's understanding at each round. These synthetic visual representations provide clear, interpretable feedback, enabling users to refine their queries intuitively and effectively. We further introduce a fully automated pipeline to generate a high-quality multi-round MIR dataset. Experimental results demonstrate that GenIR significantly outperforms existing interactive methods in the MIR scenario. This work establishes a new task with a dataset and an effective generative retrieval method, providing a foundation for future research in this direction.
>
---
#### [new 045] Token Transforming: A Unified and Training-Free Token Compression Framework for Vision Transformer Acceleration
- **分类: cs.CV**

- **简介: 该论文属于视觉Transformer加速任务，旨在解决其计算成本高的问题。通过提出一种统一且无需训练的Token压缩框架——Token Transforming，有效减少视觉Transformer中的冗余信息，实现推理加速与性能保持之间的平衡，并在多种视觉任务中验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.05709v1](http://arxiv.org/pdf/2506.05709v1)**

> **作者:** Fanhu Zeng; Deli Yu; Zhenglun Kong; Hao Tang
>
> **摘要:** Vision transformers have been widely explored in various vision tasks. Due to heavy computational cost, much interest has aroused for compressing vision transformer dynamically in the aspect of tokens. Current methods mainly pay attention to token pruning or merging to reduce token numbers, in which tokens are compressed exclusively, causing great information loss and therefore post-training is inevitably required to recover the performance. In this paper, we rethink token reduction and unify the process as an explicit form of token matrix transformation, in which all existing methods are constructing special forms of matrices within the framework. Furthermore, we propose a many-to-many Token Transforming framework that serves as a generalization of all existing methods and reserves the most information, even enabling training-free acceleration. We conduct extensive experiments to validate our framework. Specifically, we reduce 40% FLOPs and accelerate DeiT-S by $\times$1.5 with marginal 0.1% accuracy drop. Furthermore, we extend the method to dense prediction tasks including segmentation, object detection, depth estimation, and language model generation. Results demonstrate that the proposed method consistently achieves substantial improvements, offering a better computation-performance trade-off, impressive budget reduction and inference acceleration.
>
---
#### [new 046] AD-EE: Early Exiting for Fast and Reliable Vision-Language Models in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决视觉-语言模型（VLM）在实时应用中的高延迟和计算开销问题。通过引入基于因果推理的早期退出框架AD-EE，识别最优退出层，减少冗余计算。实验表明，该方法显著降低了延迟并提升了检测精度。**

- **链接: [http://arxiv.org/pdf/2506.05404v1](http://arxiv.org/pdf/2506.05404v1)**

> **作者:** Lianming Huang; Haibo Hu; Yufei Cui; Jiacheng Zuo; Shangyu Wu; Nan Guan; Chun Jason Xue
>
> **备注:** 8 pages
>
> **摘要:** With the rapid advancement of autonomous driving, deploying Vision-Language Models (VLMs) to enhance perception and decision-making has become increasingly common. However, the real-time application of VLMs is hindered by high latency and computational overhead, limiting their effectiveness in time-critical driving scenarios. This challenge is particularly evident when VLMs exhibit over-inference, continuing to process unnecessary layers even after confident predictions have been reached. To address this inefficiency, we propose AD-EE, an Early Exit framework that incorporates domain characteristics of autonomous driving and leverages causal inference to identify optimal exit layers. We evaluate our method on large-scale real-world autonomous driving datasets, including Waymo and the corner-case-focused CODA, as well as on a real vehicle running the Autoware Universe platform. Extensive experiments across multiple VLMs show that our method significantly reduces latency, with maximum improvements reaching up to 57.58%, and enhances object detection accuracy, with maximum gains of up to 44%.
>
---
#### [new 047] Controlled Data Rebalancing in Multi-Task Learning for Real-World Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决真实场景中低分辨率图像的复杂退化问题。通过多任务学习框架，提出数据重平衡方法，优化模型在不同退化模式下的表现，提升超分辨率效果。**

- **链接: [http://arxiv.org/pdf/2506.05607v1](http://arxiv.org/pdf/2506.05607v1)**

> **作者:** Shuchen Lin; Mingtao Feng; Weisheng Dong; Fangfang Wu; Jianqiao Luo; Yaonan Wang; Guangming Shi
>
> **摘要:** Real-world image super-resolution (Real-SR) is a challenging problem due to the complex degradation patterns in low-resolution images. Unlike approaches that assume a broadly encompassing degradation space, we focus specifically on achieving an optimal balance in how SR networks handle different degradation patterns within a fixed degradation space. We propose an improved paradigm that frames Real-SR as a data-heterogeneous multi-task learning problem, our work addresses task imbalance in the paradigm through coordinated advancements in task definition, imbalance quantification, and adaptive data rebalancing. Specifically, we introduce a novel task definition framework that segments the degradation space by setting parameter-specific boundaries for degradation operators, effectively reducing the task quantity while maintaining task discrimination. We then develop a focal loss based multi-task weighting mechanism that precisely quantifies task imbalance dynamics during model training. Furthermore, to prevent sporadic outlier samples from dominating the gradient optimization of the shared multi-task SR model, we strategically convert the quantified task imbalance into controlled data rebalancing through deliberate regulation of task-specific training volumes. Extensive quantitative and qualitative experiments demonstrate that our method achieves consistent superiority across all degradation tasks.
>
---
#### [new 048] Better STEP, a format and dataset for boundary representation
- **分类: cs.CV**

- **简介: 论文提出了一种基于HDF5的新格式及开源工具，用于替代工业中常用的STEP文件格式。该工作旨在解决STEP依赖CAD内核、难以大规模学习处理的问题。作者转换了多个现有数据集，并开发了标准功能和使用案例，验证新格式在几何完整性和任务兼容性方面的有效性。属于数据表示与处理任务。**

- **链接: [http://arxiv.org/pdf/2506.05417v1](http://arxiv.org/pdf/2506.05417v1)**

> **作者:** Nafiseh Izadyar; Sai Chandra Madduri; Teseo Schneider
>
> **摘要:** Boundary representation (B-rep) generated from computer-aided design (CAD) is widely used in industry, with several large datasets available. However, the data in these datasets is represented in STEP format, requiring a CAD kernel to read and process it. This dramatically limits their scope and usage in large learning pipelines, as it constrains the possibility of deploying them on computing clusters due to the high cost of per-node licenses. This paper introduces an alternative format based on the open, cross-platform format HDF5 and a corresponding dataset for STEP files, paired with an open-source library to query and process them. Our Python package also provides standard functionalities such as sampling, normals, and curvature to ease integration in existing pipelines. To demonstrate the effectiveness of our format, we converted the Fusion 360 dataset and the ABC dataset. We developed four standard use cases (normal estimation, denoising, surface reconstruction, and segmentation) to assess the integrity of the data and its compliance with the original STEP files.
>
---
#### [new 049] Object-level Self-Distillation for Vision Pretraining
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉预训练任务，旨在解决现有方法依赖单物体图像的局限性。通过提出对象级自蒸馏（ODIS），在多物体场景中实现更细粒度的预训练，提升图像和补丁级别的表示能力。实验表明其在ImageNet1k上表现优异。**

- **链接: [http://arxiv.org/pdf/2506.05409v1](http://arxiv.org/pdf/2506.05409v1)**

> **作者:** Çağlar Hızlı; Çağatay Yıldız; Pekka Marttinen
>
> **摘要:** State-of-the-art vision pretraining methods rely on image-level self-distillation from object-centric datasets such as ImageNet, implicitly assuming each image contains a single object. This assumption does not always hold: many ImageNet images already contain multiple objects. Further, it limits scalability to scene-centric datasets that better mirror real-world complexity. We address these challenges by introducing Object-level Self-DIStillation (ODIS), a pretraining approach that shifts the self-distillation granularity from whole images to individual objects. Using object-aware cropping and masked attention, ODIS isolates object-specific regions, guiding the transformer toward semantically meaningful content and transforming a noisy, scene-level task into simpler object-level sub-tasks. We show that this approach improves visual representations both at the image and patch levels. Using masks at inference time, our method achieves an impressive $82.6\%$ $k$-NN accuracy on ImageNet1k with ViT-Large.
>
---
#### [new 050] NTIRE 2025 Challenge on HR Depth from Images of Specular and Transparent Surfaces
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决高分辨率下非朗伯体表面（镜面和透明物体）的深度估计问题。工作内容为举办NTIRE 2025挑战赛，设立双目与单目深度估计两个赛道，推动相关技术发展。**

- **链接: [http://arxiv.org/pdf/2506.05815v1](http://arxiv.org/pdf/2506.05815v1)**

> **作者:** Pierluigi Zama Ramirez; Fabio Tosi; Luigi Di Stefano; Radu Timofte; Alex Costanzino; Matteo Poggi; Samuele Salti; Stefano Mattoccia; Zhe Zhang; Yang Yang; Wu Chen; Anlong Ming; Mingshuai Zhao; Mengying Yu; Shida Gao; Xiangfeng Wang; Feng Xue; Jun Shi; Yong Yang; Yong A; Yixiang Jin; Dingzhe Li; Aryan Shukla; Liam Frija-Altarac; Matthew Toews; Hui Geng; Tianjiao Wan; Zijian Gao; Qisheng Xu; Kele Xu; Zijian Zang; Jameer Babu Pinjari; Kuldeep Purohit; Mykola Lavreniuk; Jing Cao; Shenyi Li; Kui Jiang; Junjun Jiang; Yong Huang
>
> **备注:** NTIRE Workshop Challenge Report, CVPR 2025
>
> **摘要:** This paper reports on the NTIRE 2025 challenge on HR Depth From images of Specular and Transparent surfaces, held in conjunction with the New Trends in Image Restoration and Enhancement (NTIRE) workshop at CVPR 2025. This challenge aims to advance the research on depth estimation, specifically to address two of the main open issues in the field: high-resolution and non-Lambertian surfaces. The challenge proposes two tracks on stereo and single-image depth estimation, attracting about 177 registered participants. In the final testing stage, 4 and 4 participating teams submitted their models and fact sheets for the two tracks.
>
---
#### [new 051] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations in Scene Text Spotting and Understanding
- **分类: cs.CV**

- **简介: 该论文属于视觉与语言理解任务，旨在解决大型多模态模型在场景文本识别与理解中出现的语义幻觉问题。作者提出了一种无需训练的框架，结合文本区域聚焦与内部表示纠正，并构建了专门评估数据集TextHalu-Bench，有效缓解语义误导，提升模型准确性。**

- **链接: [http://arxiv.org/pdf/2506.05551v1](http://arxiv.org/pdf/2506.05551v1)**

> **作者:** Yan Shu; Hangui Lin; Yexin Liu; Yan Zhang; Gangyan Zeng; Yan Li; Yu Zhou; Ser-Nam Lim; Harry Yang; Nicu Sebe
>
> **摘要:** Large Multimodal Models (LMMs) have achieved impressive progress in visual perception and reasoning. However, when confronted with visually ambiguous or non-semantic scene text, they often struggle to accurately spot and understand the content, frequently generating semantically plausible yet visually incorrect answers, which we refer to as semantic hallucination. In this work, we investigate the underlying causes of semantic hallucination and identify a key finding: Transformer layers in LLM with stronger attention focus on scene text regions are less prone to producing semantic hallucinations. Thus, we propose a training-free semantic hallucination mitigation framework comprising two key components: (1) ZoomText, a coarse-to-fine strategy that identifies potential text regions without external detectors; and (2) Grounded Layer Correction, which adaptively leverages the internal representations from layers less prone to hallucination to guide decoding, correcting hallucinated outputs for non-semantic samples while preserving the semantics of meaningful ones. To enable rigorous evaluation, we introduce TextHalu-Bench, a benchmark of over 1,730 samples spanning both semantic and non-semantic cases, with manually curated question-answer pairs designed to probe model hallucinations. Extensive experiments demonstrate that our method not only effectively mitigates semantic hallucination but also achieves strong performance on public benchmarks for scene text spotting and understanding.
>
---
#### [new 052] Towards Reliable Identification of Diffusion-based Image Manipulations
- **分类: cs.CV**

- **简介: 该论文属于图像篡改检测任务，旨在解决扩散模型生成的图像编辑难以识别的问题。作者提出了RADAR方法，结合多模态特征与对比损失，提升篡改区域的识别精度，并构建了新基准BBC-PAIR进行评估，结果显示该方法在检测多种扩散模型编辑方面表现优异。**

- **链接: [http://arxiv.org/pdf/2506.05466v1](http://arxiv.org/pdf/2506.05466v1)**

> **作者:** Alex Costanzino; Woody Bayliss; Juil Sock; Marc Gorriz Blanch; Danijela Horak; Ivan Laptev; Philip Torr; Fabio Pizzati
>
> **摘要:** Changing facial expressions, gestures, or background details may dramatically alter the meaning conveyed by an image. Notably, recent advances in diffusion models greatly improve the quality of image manipulation while also opening the door to misuse. Identifying changes made to authentic images, thus, becomes an important task, constantly challenged by new diffusion-based editing tools. To this end, we propose a novel approach for ReliAble iDentification of inpainted AReas (RADAR). RADAR builds on existing foundation models and combines features from different image modalities. It also incorporates an auxiliary contrastive loss that helps to isolate manipulated image patches. We demonstrate these techniques to significantly improve both the accuracy of our method and its generalisation to a large number of diffusion models. To support realistic evaluation, we further introduce BBC-PAIR, a new comprehensive benchmark, with images tampered by 28 diffusion models. Our experiments show that RADAR achieves excellent results, outperforming the state-of-the-art in detecting and localising image edits made by both seen and unseen diffusion models. Our code, data and models will be publicly available at alex-costanzino.github.io/radar.
>
---
#### [new 053] Optimizing Cloud-to-GPU Throughput for Deep Learning With Earth Observation Data
- **分类: cs.CV**

- **简介: 该论文属于深度学习训练优化任务，旨在解决从云存储流式加载地球观测数据时GPU利用率低的问题。通过优化GeoTIFF文件的读取方式和线程配置，显著提升了数据加载吞吐量与模型训练效率。**

- **链接: [http://arxiv.org/pdf/2506.06235v1](http://arxiv.org/pdf/2506.06235v1)**

> **作者:** Akram Zaytar; Caleb Robinson; Girmaw Abebe Tadesse; Tammy Glazer; Gilles Hacheme; Anthony Ortiz; Rahul M Dodhia; Juan M Lavista Ferres
>
> **摘要:** Training deep learning models on petabyte-scale Earth observation (EO) data requires separating compute resources from data storage. However, standard PyTorch data loaders cannot keep modern GPUs utilized when streaming GeoTIFF files directly from cloud storage. In this work, we benchmark GeoTIFF loading throughput from both cloud object storage and local SSD, systematically testing different loader configurations and data parameters. We focus on tile-aligned reads and worker thread pools, using Bayesian optimization to find optimal settings for each storage type. Our optimized configurations increase remote data loading throughput by 20x and local throughput by 4x compared to default settings. On three public EO benchmarks, models trained with optimized remote loading achieve the same accuracy as local training within identical time budgets. We improve validation IoU by 6-15% and maintain 85-95% GPU utilization versus 0-30% with standard configurations. Code is publicly available at https://github.com/microsoft/pytorch-cloud-geotiff-optimization
>
---
#### [new 054] Attention-based transformer models for image captioning across languages: An in-depth survey and evaluation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像描述生成任务，旨在解决多语言场景下图像生成文本描述的问题。论文系统综述了基于注意力机制的Transformer模型在跨语言图像描述中的应用，分析了现有方法、数据集与评估指标，并指出当前模型在语义一致性、非英语数据稀缺和推理能力方面的局限性，提出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2506.05399v1](http://arxiv.org/pdf/2506.05399v1)**

> **作者:** Israa A. Albadarneh; Bassam H. Hammo; Omar S. Al-Kadi
>
> **备注:** 31 pages, 15 figures, 6 tables
>
> **摘要:** Image captioning involves generating textual descriptions from input images, bridging the gap between computer vision and natural language processing. Recent advancements in transformer-based models have significantly improved caption generation by leveraging attention mechanisms for better scene understanding. While various surveys have explored deep learning-based approaches for image captioning, few have comprehensively analyzed attention-based transformer models across multiple languages. This survey reviews attention-based image captioning models, categorizing them into transformer-based, deep learning-based, and hybrid approaches. It explores benchmark datasets, discusses evaluation metrics such as BLEU, METEOR, CIDEr, and ROUGE, and highlights challenges in multilingual captioning. Additionally, this paper identifies key limitations in current models, including semantic inconsistencies, data scarcity in non-English languages, and limitations in reasoning ability. Finally, we outline future research directions, such as multimodal learning, real-time applications in AI-powered assistants, healthcare, and forensic analysis. This survey serves as a comprehensive reference for researchers aiming to advance the field of attention-based image captioning.
>
---
#### [new 055] BYO-Eval: Build Your Own Dataset for Fine-Grained Visual Assessment of Multimodal Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决现有评估方法标注成本高、易信息泄露及难以定位模型缺陷的问题。工作提出BYO-Eval方法，通过合成图像进行细粒度视觉感知评估，实现对模型能力的系统性诊断与分析。**

- **链接: [http://arxiv.org/pdf/2506.05440v1](http://arxiv.org/pdf/2506.05440v1)**

> **作者:** Ludovic Arnould; Salim Khazem; Hugues Ali Mehenni
>
> **摘要:** Visual Language Models (VLMs) are now sufficiently advanced to support a broad range of applications, including answering complex visual questions, and are increasingly expected to interact with images in varied ways. To evaluate them, current benchmarks often focus on specific domains (e.g., reading charts), constructing datasets of annotated real images paired with pre-defined Multiple Choice Questions (MCQs) to report aggregate accuracy scores. However, such benchmarks entail high annotation costs, risk information leakage, and do not clarify whether failures stem from limitations in visual perception, reasoning, or general knowledge. We propose a new evaluation methodology, inspired by ophthalmologic diagnostics, leveraging procedural generation of synthetic images to obtain control over visual attributes and precisely reveal perception failures in VLMs. Specifically, we build collections of images with gradually more challenging variations in the content of interest (e.g., number of objects in a counting task) while holding other visual parameters constant. This diagnostic allows systematic stress testing and fine-grained failure analysis, shifting the focus from coarse benchmarking toward targeted and interpretable assessment of VLM capabilities. Our code is available at https://github.com/byoeval/BYO-EVAL.
>
---
#### [new 056] Personalized Interpretability -- Interactive Alignment of Prototypical Parts Networks
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于可解释性机器学习任务，旨在解决概念不一致问题。现有基于案例推理的模型解释可能因视觉特征混杂而不易理解。作者提出YoursProtoP方法，通过用户交互个性化调整模型使用的原型部分，提升概念一致性与用户理解，实验证明其有效性。**

- **链接: [http://arxiv.org/pdf/2506.05533v1](http://arxiv.org/pdf/2506.05533v1)**

> **作者:** Tomasz Michalski; Adam Wróbel; Andrea Bontempelli; Jakub Luśtyk; Mikolaj Kniejski; Stefano Teso; Andrea Passerini; Bartosz Zieliński; Dawid Rymarczyk
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Concept-based interpretable neural networks have gained significant attention due to their intuitive and easy-to-understand explanations based on case-based reasoning, such as "this bird looks like those sparrows". However, a major limitation is that these explanations may not always be comprehensible to users due to concept inconsistency, where multiple visual features are inappropriately mixed (e.g., a bird's head and wings treated as a single concept). This inconsistency breaks the alignment between model reasoning and human understanding. Furthermore, users have specific preferences for how concepts should look, yet current approaches provide no mechanism for incorporating their feedback. To address these issues, we introduce YoursProtoP, a novel interactive strategy that enables the personalization of prototypical parts - the visual concepts used by the model - according to user needs. By incorporating user supervision, YoursProtoP adapts and splits concepts used for both prediction and explanation to better match the user's preferences and understanding. Through experiments on both the synthetic FunnyBirds dataset and a real-world scenario using the CUB, CARS, and PETS datasets in a comprehensive user study, we demonstrate the effectiveness of YoursProtoP in achieving concept consistency without compromising the accuracy of the model.
>
---
#### [new 057] Rethinking Semi-supervised Segmentation Beyond Accuracy: Reliability and Robustness
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于语义分割任务，旨在解决半监督分割模型在实际部署中缺乏可靠性和鲁棒性的问题。作者提出新指标RSS，综合评估准确性、置信度校准和不确定性质量，发现现有方法牺牲可靠性换取准确性的现象，并倡导更全面的评估标准。**

- **链接: [http://arxiv.org/pdf/2506.05917v1](http://arxiv.org/pdf/2506.05917v1)**

> **作者:** Steven Landgraf; Markus Hillemann; Markus Ulrich
>
> **摘要:** Semantic segmentation is critical for scene understanding but demands costly pixel-wise annotations, attracting increasing attention to semi-supervised approaches to leverage abundant unlabeled data. While semi-supervised segmentation is often promoted as a path toward scalable, real-world deployment, it is astonishing that current evaluation protocols exclusively focus on segmentation accuracy, entirely overlooking reliability and robustness. These qualities, which ensure consistent performance under diverse conditions (robustness) and well-calibrated model confidences as well as meaningful uncertainties (reliability), are essential for safety-critical applications like autonomous driving, where models must handle unpredictable environments and avoid sudden failures at all costs. To address this gap, we introduce the Reliable Segmentation Score (RSS), a novel metric that combines predictive accuracy, calibration, and uncertainty quality measures via a harmonic mean. RSS penalizes deficiencies in any of its components, providing an easy and intuitive way of holistically judging segmentation models. Comprehensive evaluations of UniMatchV2 against its predecessor and a supervised baseline show that semi-supervised methods often trade reliability for accuracy. While out-of-domain evaluations demonstrate UniMatchV2's robustness, they further expose persistent reliability shortcomings. We advocate for a shift in evaluation protocols toward more holistic metrics like RSS to better align semi-supervised learning research with real-world deployment needs.
>
---
#### [new 058] Speaking images. A novel framework for the automated self-description of artworks
- **分类: cs.CV; cs.AI**

- **简介: 该论文旨在解决数字艺术藏品内容展示与传播的问题。通过构建一个自动化框架，结合生成式AI、大语言模型、人脸检测等技术，使数字化艺术品能自动生成讲解视频。属于图像生成与文化传承任务，探索人工智能在艺术解读中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2506.05368v1](http://arxiv.org/pdf/2506.05368v1)**

> **作者:** Valentine Bernasconi; Gustavo Marfia
>
> **摘要:** Recent breakthroughs in generative AI have opened the door to new research perspectives in the domain of art and cultural heritage, where a large number of artifacts have been digitized. There is a need for innovation to ease the access and highlight the content of digital collections. Such innovations develop into creative explorations of the digital image in relation to its malleability and contemporary interpretation, in confrontation to the original historical object. Based on the concept of the autonomous image, we propose a new framework towards the production of self-explaining cultural artifacts using open-source large-language, face detection, text-to-speech and audio-to-animation models. The goal is to start from a digitized artwork and to automatically assemble a short video of the latter where the main character animates to explain its content. The whole process questions cultural biases encapsulated in large-language models, the potential of digital images and deepfakes of artworks for educational purposes, along with concerns of the field of art history regarding such creative diversions.
>
---
#### [new 059] MORSE-500: A Programmatically Controllable Video Benchmark to Stress-Test Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态推理任务，旨在解决现有基准测试在时序复杂性、推理类型覆盖和可扩展性方面的不足。作者构建了MORSE-500，一个包含500个脚本视频的基准，支持系统控制难度并动态生成更具挑战性的实例，以全面评估和推动多模态模型的发展。**

- **链接: [http://arxiv.org/pdf/2506.05523v1](http://arxiv.org/pdf/2506.05523v1)**

> **作者:** Zikui Cai; Andrew Wang; Anirudh Satheesh; Ankit Nakhawa; Hyunwoo Jae; Keenan Powell; Minghui Liu; Neel Jay; Sungbin Oh; Xiyao Wang; Yongyuan Liang; Tom Goldstein; Furong Huang
>
> **摘要:** Despite rapid advances in vision-language models (VLMs), current benchmarks for multimodal reasoning fall short in three key dimensions. First, they overwhelmingly rely on static images, failing to capture the temporal complexity of real-world environments. Second, they narrowly focus on mathematical problem-solving, neglecting the broader spectrum of reasoning skills -- including abstract, physical, planning, spatial, and temporal capabilities -- required for robust multimodal intelligence. Third, many benchmarks quickly saturate, offering limited headroom for diagnosing failure modes or measuring continued progress. We introduce MORSE-500 (Multimodal Reasoning Stress-test Environment), a video benchmark composed of 500 fully scripted clips with embedded questions spanning six complementary reasoning categories. Each instance is programmatically generated using deterministic Python scripts (via Manim, Matplotlib, MoviePy), generative video models, and curated real footage. This script-driven design allows fine-grained control over visual complexity, distractor density, and temporal dynamics -- enabling difficulty to be scaled systematically as models improve. Unlike static benchmarks that become obsolete once saturated, MORSE-500 is built to evolve: its controllable generation pipeline supports the creation of arbitrarily challenging new instances, making it ideally suited for stress-testing next-generation models. Initial experiments with state-of-the-art systems -- including various Gemini 2.5 Pro and OpenAI o3 which represent the strongest available at the time, alongside strong open-source models -- reveal substantial performance gaps across all categories, with particularly large deficits in abstract and planning tasks. We release the full dataset, generation scripts, and evaluation harness to support transparent, reproducible, and forward-looking multimodal reasoning research.
>
---
#### [new 060] Unleashing the Potential of Consistency Learning for Detecting and Grounding Multi-Modal Media Manipulation
- **分类: cs.CV**

- **简介: 该论文属于多模态媒体伪造检测与定位任务，旨在解决现有方法对局部内容一致性挖掘不足导致的伪造细节感知弱、结果不可靠问题。作者提出了一种上下文-语义一致性学习（CSCL）方法，通过两个级联解码器分别捕捉模态内上下文一致性和跨模态语义一致性，以提升伪造检测性能。实验表明该方法在DGM4数据集上取得了新的最优结果。**

- **链接: [http://arxiv.org/pdf/2506.05890v1](http://arxiv.org/pdf/2506.05890v1)**

> **作者:** Yiheng Li; Yang Yang; Zichang Tan; Huan Liu; Weihua Chen; Xu Zhou; Zhen Lei
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** To tackle the threat of fake news, the task of detecting and grounding multi-modal media manipulation DGM4 has received increasing attention. However, most state-of-the-art methods fail to explore the fine-grained consistency within local content, usually resulting in an inadequate perception of detailed forgery and unreliable results. In this paper, we propose a novel approach named Contextual-Semantic Consistency Learning (CSCL) to enhance the fine-grained perception ability of forgery for DGM4. Two branches for image and text modalities are established, each of which contains two cascaded decoders, i.e., Contextual Consistency Decoder (CCD) and Semantic Consistency Decoder (SCD), to capture within-modality contextual consistency and across-modality semantic consistency, respectively. Both CCD and SCD adhere to the same criteria for capturing fine-grained forgery details. To be specific, each module first constructs consistency features by leveraging additional supervision from the heterogeneous information of each token pair. Then, the forgery-aware reasoning or aggregating is adopted to deeply seek forgery cues based on the consistency features. Extensive experiments on DGM4 datasets prove that CSCL achieves new state-of-the-art performance, especially for the results of grounding manipulated content. Codes and weights are avaliable at https://github.com/liyih/CSCL.
>
---
#### [new 061] Challenging Vision-Language Models with Surgical Data: A New Dataset and Broad Benchmarking Study
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型（VLMs）在腹腔镜手术图像上的性能，评估其在基础感知任务和复杂场景理解中的表现。论文属于计算机视觉与自然语言处理交叉任务，旨在解决当前VLMs在医疗图像理解中泛化能力不足的问题。作者通过多模型、多数据集测试发现，VLMs在基本任务上表现良好，但在需要医学知识的任务上性能下降，且专用医疗VLMs不如通用模型。**

- **链接: [http://arxiv.org/pdf/2506.06232v1](http://arxiv.org/pdf/2506.06232v1)**

> **作者:** Leon Mayer; Tim Rädsch; Dominik Michael; Lucas Luttner; Amine Yamlahi; Evangelia Christodoulou; Patrick Godau; Marcel Knopp; Annika Reinke; Fiona Kolbinger; Lena Maier-Hein
>
> **摘要:** While traditional computer vision models have historically struggled to generalize to endoscopic domains, the emergence of foundation models has shown promising cross-domain performance. In this work, we present the first large-scale study assessing the capabilities of Vision Language Models (VLMs) for endoscopic tasks with a specific focus on laparoscopic surgery. Using a diverse set of state-of-the-art models, multiple surgical datasets, and extensive human reference annotations, we address three key research questions: (1) Can current VLMs solve basic perception tasks on surgical images? (2) Can they handle advanced frame-based endoscopic scene understanding tasks? and (3) How do specialized medical VLMs compare to generalist models in this context? Our results reveal that VLMs can effectively perform basic surgical perception tasks, such as object counting and localization, with performance levels comparable to general domain tasks. However, their performance deteriorates significantly when the tasks require medical knowledge. Notably, we find that specialized medical VLMs currently underperform compared to generalist models across both basic and advanced surgical tasks, suggesting that they are not yet optimized for the complexity of surgical environments. These findings highlight the need for further advancements to enable VLMs to handle the unique challenges posed by surgery. Overall, our work provides important insights for the development of next-generation endoscopic AI systems and identifies key areas for improvement in medical visual language models.
>
---
#### [new 062] Self-Predictive Dynamics for Generalization of Vision-based Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉强化学习任务，旨在解决图像中任务无关干扰元素影响模型泛化的问题。作者提出Self-Predictive Dynamics（SPD）方法，通过并行使用强弱增强，预测双向转换关系，提取任务相关特征，提升在未见观测中的泛化性能。**

- **链接: [http://arxiv.org/pdf/2506.05418v1](http://arxiv.org/pdf/2506.05418v1)**

> **作者:** Kyungsoo Kim; Jeongsoo Ha; Yusung Kim
>
> **备注:** IJCAI 2022
>
> **摘要:** Vision-based reinforcement learning requires efficient and robust representations of image-based observations, especially when the images contain distracting (task-irrelevant) elements such as shadows, clouds, and light. It becomes more important if those distractions are not exposed during training. We design a Self-Predictive Dynamics (SPD) method to extract task-relevant features efficiently, even in unseen observations after training. SPD uses weak and strong augmentations in parallel, and learns representations by predicting inverse and forward transitions across the two-way augmented versions. In a set of MuJoCo visual control tasks and an autonomous driving task (CARLA), SPD outperforms previous studies in complex observations, and significantly improves the generalization performance for unseen observations. Our code is available at https://github.com/unigary/SPD.
>
---
#### [new 063] DriveAction: A Benchmark for Exploring Human-like Driving Decisions in VLA Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决现有基准在场景多样性、动作标注和评估协议上的不足。作者构建了DriveAction，首个面向VLA模型的动作驱动基准，包含16,185 QA对，基于真实驾驶数据，提供高阶离散动作标签与树状评估框架。实验表明，视觉与语言输入对动作预测均至关重要。**

- **链接: [http://arxiv.org/pdf/2506.05667v1](http://arxiv.org/pdf/2506.05667v1)**

> **作者:** Yuhan Hao; Zhengning Li; Lei Sun; Weilong Wang; Naixin Yi; Sheng Song; Caihong Qin; Mofan Zhou; Yifei Zhan; Peng Jia; Xianpeng Lang
>
> **备注:** Benchmark: https://huggingface.co/datasets/LiAuto-DriveAction/drive-action
>
> **摘要:** Vision-Language-Action (VLA) models have advanced autonomous driving, but existing benchmarks still lack scenario diversity, reliable action-level annotation, and evaluation protocols aligned with human preferences. To address these limitations, we introduce DriveAction, the first action-driven benchmark specifically designed for VLA models, comprising 16,185 QA pairs generated from 2,610 driving scenarios. DriveAction leverages real-world driving data proactively collected by users of production-level autonomous vehicles to ensure broad and representative scenario coverage, offers high-level discrete action labels collected directly from users' actual driving operations, and implements an action-rooted tree-structured evaluation framework that explicitly links vision, language, and action tasks, supporting both comprehensive and task-specific assessment. Our experiments demonstrate that state-of-the-art vision-language models (VLMs) require both vision and language guidance for accurate action prediction: on average, accuracy drops by 3.3% without vision input, by 4.1% without language input, and by 8.0% without either. Our evaluation supports precise identification of model bottlenecks with robust and consistent results, thus providing new insights and a rigorous foundation for advancing human-like decisions in autonomous driving.
>
---
#### [new 064] CLaMR: Contextualized Late-Interaction for Multimodal Content Retrieval
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于多模态视频内容检索任务，旨在解决传统方法因独立处理多模态信息导致的噪声和检索效果差问题。作者提出了CLaMR模型，通过统一编码四类模态并动态选择关键模态，提升了检索性能。实验表明其在多个数据集上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.06144v1](http://arxiv.org/pdf/2506.06144v1)**

> **作者:** David Wan; Han Wang; Elias Stengel-Eskin; Jaemin Cho; Mohit Bansal
>
> **备注:** 18 pages. Code and data: https://github.com/meetdavidwan/clamr
>
> **摘要:** Online video web content is richly multimodal: a single video blends vision, speech, ambient audio, and on-screen text. Retrieval systems typically treat these modalities as independent retrieval sources, which can lead to noisy and subpar retrieval. We explore multimodal video content retrieval, where relevance can be scored from one particular modality or jointly across multiple modalities simultaneously. Consequently, an effective retriever must dynamically choose which modality (or set of modalities) best addresses the query. We introduce CLaMR, a multimodal, late-interaction retriever that jointly indexes 4 modalities: video frames, transcribed speech, on-screen text, and metadata. CLaMR jointly encodes all modalities with a unified multimodal backbone for improved contextualization and is trained to enhance dynamic modality selection via two key innovations. First, given the lack of training data for multimodal retrieval, we introduce MultiVENT 2.0++, a large-scale synthetic training dataset built on MultiVENT 2.0 (event-centric videos in various languages paired with queries) with modality-targeted queries. Next, we propose a modality-aware loss that jointly trains according to a standard contrastive objective alongside an objective for learning correct modality usage. On the test sets of MultiVENT 2.0++ and MSRVTT, conventional aggregation strategies, such as averaging similarities for baseline retrievers, degrade performance by introducing noise from irrelevant modalities. In contrast, CLaMR consistently outperforms existing retrievers: on MultiVENT 2.0++, CLaMR improves nDCG@10 by 25.6 over the best single-modality retriever and by 35.4 over the best multi-modality retriever. We illustrate CLaMR's downstream utility on long-video QA, retrieving relevant frames and obtaining a 3.50% boost over LanguageBind on Video-MME and 1.42% over dense sampling on LongVideoBench.
>
---
#### [new 065] Pts3D-LLM: Studying the Impact of Token Structure for 3D Scene Understanding With Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决如何有效表示3D数据以提升多模态大语言模型性能的问题。作者系统研究了点云与视频两种token结构，提出融合3D点云特征的方法，提升了模型表现，并展示了点结构在合理设计下可媲美视频结构的效果。**

- **链接: [http://arxiv.org/pdf/2506.05689v1](http://arxiv.org/pdf/2506.05689v1)**

> **作者:** Hugues Thomas; Chen Chen; Jian Zhang
>
> **备注:** Main paper and appendix
>
> **摘要:** Effectively representing 3D scenes for Multimodal Large Language Models (MLLMs) is crucial yet challenging. Existing approaches commonly only rely on 2D image features and use varied tokenization approaches. This work presents a rigorous study of 3D token structures, systematically comparing video-based and point-based representations while maintaining consistent model backbones and parameters. We propose a novel approach that enriches visual tokens by incorporating 3D point cloud features from a Sonata pretrained Point Transformer V3 encoder. Our experiments demonstrate that merging explicit 3D features significantly boosts performance. Furthermore, we show that point-based token structures can rival video-based ones when the points are cleverly sampled and ordered. Our best models from both structures achieve state-of-the-art results on multiple 3D understanding benchmarks. We emphasize our analysis of token structures as a key contribution, alongside transparent reporting of results averaged over multiple seeds, a practice we believe is vital for robust progress in the field.
>
---
#### [new 066] MR.NAVI: Mixed-Reality Navigation Assistant for the Visually Impaired
- **分类: cs.CV**

- **简介: 论文提出MR.NAVI系统，属于辅助导航任务，旨在帮助视障人士在陌生环境中导航。系统通过混合现实技术，结合计算机视觉与自然语言处理，提供实时场景描述、避障及导航指引。使用MobileNet、RANSAC和DBSCAN等方法处理图像与空间数据，并集成公共交通API，提升实用性。实验验证了其在场景理解和导航中的有效性。**

- **链接: [http://arxiv.org/pdf/2506.05369v1](http://arxiv.org/pdf/2506.05369v1)**

> **作者:** Nicolas Pfitzer; Yifan Zhou; Marco Poggensee; Defne Kurtulus; Bessie Dominguez-Dager; Mihai Dusmanu; Marc Pollefeys; Zuria Bauer
>
> **摘要:** Over 43 million people worldwide live with severe visual impairment, facing significant challenges in navigating unfamiliar environments. We present MR.NAVI, a mixed reality system that enhances spatial awareness for visually impaired users through real-time scene understanding and intuitive audio feedback. Our system combines computer vision algorithms for object detection and depth estimation with natural language processing to provide contextual scene descriptions, proactive collision avoidance, and navigation instructions. The distributed architecture processes sensor data through MobileNet for object detection and employs RANSAC-based floor detection with DBSCAN clustering for obstacle avoidance. Integration with public transit APIs enables navigation with public transportation directions. Through our experiments with user studies, we evaluated both scene description and navigation functionalities in unfamiliar environments, showing promising usability and effectiveness.
>
---
#### [new 067] Query Nearby: Offset-Adjusted Mask2Former enhances small-organ segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提升小器官分割精度。针对现有模型在小器官分割中效果差、计算量大的问题，作者改进了Mask2Former模型，引入偏移调整策略和辅助头，优化注意力机制中的采样点分布，并融合粗略定位信息，从而提升了对小器官的分割性能，达到了当前最优水平。**

- **链接: [http://arxiv.org/pdf/2506.05897v1](http://arxiv.org/pdf/2506.05897v1)**

> **作者:** Xin Zhang; Dongdong Meng; Sheng Li
>
> **摘要:** Medical segmentation plays an important role in clinical applications like radiation therapy and surgical guidance, but acquiring clinically acceptable results is difficult. In recent years, progress has been witnessed with the success of utilizing transformer-like models, such as combining the attention mechanism with CNN. In particular, transformer-based segmentation models can extract global information more effectively, compensating for the drawbacks of CNN modules that focus on local features. However, utilizing transformer architecture is not easy, because training transformer-based models can be resource-demanding. Moreover, due to the distinct characteristics in the medical field, especially when encountering mid-sized and small organs with compact regions, their results often seem unsatisfactory. For example, using ViT to segment medical images directly only gives a DSC of less than 50\%, which is far lower than the clinically acceptable score of 80\%. In this paper, we used Mask2Former with deformable attention to reduce computation and proposed offset adjustment strategies to encourage sampling points within the same organs during attention weights computation, thereby integrating compact foreground information better. Additionally, we utilized the 4th feature map in Mask2Former to provide a coarse location of organs, and employed an FCN-based auxiliary head to help train Mask2Former more quickly using Dice loss. We show that our model achieves SOTA (State-of-the-Art) performance on the HaNSeg and SegRap2023 datasets, especially on mid-sized and small organs.Our code is available at link https://github.com/earis/Offsetadjustment\_Background-location\_Decoder\_Mask2former.
>
---
#### [new 068] EX-4D: EXtreme Viewpoint 4D Video Synthesis via Depth Watertight Mesh
- **分类: cs.CV**

- **简介: 该论文属于4D视频生成任务，旨在解决极端视角下几何不一致和遮挡问题。作者提出EX-4D框架，采用Depth Watertight Mesh建模可见与遮挡区域，结合模拟掩码策略和LoRA视频扩散适配器，实现高质量、视角可控的4D视频合成。**

- **链接: [http://arxiv.org/pdf/2506.05554v1](http://arxiv.org/pdf/2506.05554v1)**

> **作者:** Tao Hu; Haoyang Peng; Xiao Liu; Yuewen Ma
>
> **摘要:** Generating high-quality camera-controllable videos from monocular input is a challenging task, particularly under extreme viewpoint. Existing methods often struggle with geometric inconsistencies and occlusion artifacts in boundaries, leading to degraded visual quality. In this paper, we introduce EX-4D, a novel framework that addresses these challenges through a Depth Watertight Mesh representation. The representation serves as a robust geometric prior by explicitly modeling both visible and occluded regions, ensuring geometric consistency in extreme camera pose. To overcome the lack of paired multi-view datasets, we propose a simulated masking strategy that generates effective training data only from monocular videos. Additionally, a lightweight LoRA-based video diffusion adapter is employed to synthesize high-quality, physically consistent, and temporally coherent videos. Extensive experiments demonstrate that EX-4D outperforms state-of-the-art methods in terms of physical consistency and extreme-view quality, enabling practical 4D video generation.
>
---
#### [new 069] Seed Selection for Human-Oriented Image Reconstruction via Guided Diffusion
- **分类: cs.CV**

- **简介: 该论文属于图像重建任务，旨在解决基于扩散模型的人机协同图像编码中图像质量不佳的问题。现有方法使用单一随机种子生成图像，效果有限。本文提出通过早期扩散过程的中间结果选择最优种子，在不增加比特率的前提下提升重建质量。**

- **链接: [http://arxiv.org/pdf/2506.05363v1](http://arxiv.org/pdf/2506.05363v1)**

> **作者:** Yui Tatsumi; Ziyue Zeng; Hiroshi Watanabe
>
> **摘要:** Conventional methods for scalable image coding for humans and machines require the transmission of additional information to achieve scalability. A recent diffusion-based method avoids this by generating human-oriented images from machine-oriented images without extra bitrate. This method, however, uses a single random seed, which may lead to suboptimal image quality. In this paper, we propose a seed selection method that identifies the optimal seed from multiple candidates to improve image quality without increasing the bitrate. To reduce computational cost, the selection is performed based on intermediate outputs obtained from early steps of the reverse diffusion process. Experimental results demonstrate that our method outperforms the baseline across multiple metrics.
>
---
#### [new 070] SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于多模态3D空间推理任务，旨在解决现有AV-LLMs在动态音频-视觉环境中缺乏3D空间推理能力的问题。作者提出了SAVVY-Bench基准和SAVVY方法，通过空间轨迹估计与全局地图构建，实现动态3D空间理解，并显著提升现有模型的表现。**

- **链接: [http://arxiv.org/pdf/2506.05414v1](http://arxiv.org/pdf/2506.05414v1)**

> **作者:** Mingfei Chen; Zijun Cui; Xiulong Liu; Jinlin Xiang; Caleb Zheng; Jingyuan Li; Eli Shlizerman
>
> **备注:** Project website with demo videos: https://zijuncui02.github.io/SAVVY/
>
> **摘要:** 3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs.
>
---
#### [new 071] Sample-Specific Noise Injection For Diffusion-Based Adversarial Purification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像鲁棒性增强任务，旨在解决扩散对抗净化（DBP）中固定噪声注入级别效果不佳的问题。作者提出SSNI框架，利用预训练得分网络估计样本偏离程度，并据此自适应调整每个样本的噪声注入水平，从而提升模型准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.06027v1](http://arxiv.org/pdf/2506.06027v1)**

> **作者:** Yuhao Sun; Jiacheng Zhang; Zesheng Ye; Chaowei Xiao; Feng Liu
>
> **摘要:** Diffusion-based purification (DBP) methods aim to remove adversarial noise from the input sample by first injecting Gaussian noise through a forward diffusion process, and then recovering the clean example through a reverse generative process. In the above process, how much Gaussian noise is injected to the input sample is key to the success of DBP methods, which is controlled by a constant noise level $t^*$ for all samples in existing methods. In this paper, we discover that an optimal $t^*$ for each sample indeed could be different. Intuitively, the cleaner a sample is, the less the noise it should be injected, and vice versa. Motivated by this finding, we propose a new framework, called Sample-specific Score-aware Noise Injection (SSNI). Specifically, SSNI uses a pre-trained score network to estimate how much a data point deviates from the clean data distribution (i.e., score norms). Then, based on the magnitude of score norms, SSNI applies a reweighting function to adaptively adjust $t^*$ for each sample, achieving sample-specific noise injections. Empirically, incorporating our framework with existing DBP methods results in a notable improvement in both accuracy and robustness on CIFAR-10 and ImageNet-1K, highlighting the necessity to allocate distinct noise levels to different samples in DBP methods. Our code is available at: https://github.com/tmlr-group/SSNI.
>
---
#### [new 072] U-NetMN and SegNetMN: Modified U-Net and SegNet models for bimodal SAR image segmentation
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 论文研究SAR遥感图像分割任务，旨在解决深度学习模型在该任务中收敛速度慢、稳定性差的问题。作者在U-Net和SegNet中引入模态归一化方法，有效提升收敛速度与跨区域稳定性，验证了归一化对SAR图像分割的优化效果。**

- **链接: [http://arxiv.org/pdf/2506.05444v1](http://arxiv.org/pdf/2506.05444v1)**

> **作者:** Marwane Kzadri; Franco Alberto Cardillo; Nanée Chahinian; Carole Delenne; Renaud Hostache; Jamal Riffi
>
> **摘要:** Segmenting Synthetic Aperture Radar (SAR) images is crucial for many remote sensing applications, particularly water body detection. However, deep learning-based segmentation models often face challenges related to convergence speed and stability, mainly due to the complex statistical distribution of this type of data. In this study, we evaluate the impact of mode normalization on two widely used semantic segmentation models, U-Net and SegNet. Specifically, we integrate mode normalization, to reduce convergence time while maintaining the performance of the baseline models. Experimental results demonstrate that mode normalization significantly accelerates convergence. Furthermore, cross-validation results indicate that normalized models exhibit increased stability in different zones. These findings highlight the effectiveness of normalization in improving computational efficiency and generalization in SAR image segmentation.
>
---
#### [new 073] Can Vision Transformers with ResNet's Global Features Fairly Authenticate Demographic Faces?
- **分类: cs.CV**

- **简介: 该论文属于生物特征认证任务，旨在解决不同人群在人脸识别中的公平性与泛化性问题。作者结合预训练的ViT与ResNet全局特征，设计了一种少样本原型网络，并构建了新的支持和查询数据集，以评估模型在不同种族、性别和年龄组上的性能。实验表明Swin Transformer效果最佳。**

- **链接: [http://arxiv.org/pdf/2506.05383v1](http://arxiv.org/pdf/2506.05383v1)**

> **作者:** Abu Sufian; Marco Leo; Cosimo Distante; Anirudha Ghosh; Debaditya Barman
>
> **备注:** 14 pages, 6 Figures, ICPR 2024 Workshop FAIRBIO
>
> **摘要:** Biometric face authentication is crucial in computer vision, but ensuring fairness and generalization across demographic groups remains a big challenge. Therefore, we investigated whether Vision Transformer (ViT) and ResNet, leveraging pre-trained global features, can fairly authenticate different demographic faces while relying minimally on local features. In this investigation, we used three pre-trained state-of-the-art (SOTA) ViT foundation models from Facebook, Google, and Microsoft for global features as well as ResNet-18. We concatenated the features from ViT and ResNet, passed them through two fully connected layers, and trained on customized face image datasets to capture the local features. Then, we designed a novel few-shot prototype network with backbone features embedding. We also developed new demographic face image support and query datasets for this empirical study. The network's testing was conducted on this dataset in one-shot, three-shot, and five-shot scenarios to assess how performance improves as the size of the support set increases. We observed results across datasets with varying races/ethnicities, genders, and age groups. The Microsoft Swin Transformer backbone performed better among the three SOTA ViT for this task. The code and data are available at: https://github.com/Sufianlab/FairVitBio.
>
---
#### [new 074] CarboNeXT and CarboFormer: Dual Semantic Segmentation Architectures for Detecting and Quantifying Carbon Dioxide Emissions Using Optical Gas Imaging
- **分类: cs.CV**

- **简介: 论文提出CarboNeXT和CarboFormer两种语义分割模型，用于光学气体成像中二氧化碳排放的检测与量化。任务为环境监测与畜牧业管理中的气体泄漏识别。解决了低流量CO₂泄漏检测难、实时性不足及资源受限设备部署问题。构建了两个新数据集，并在性能与速度上实现领先。**

- **链接: [http://arxiv.org/pdf/2506.05360v1](http://arxiv.org/pdf/2506.05360v1)**

> **作者:** Taminul Islam; Toqi Tahamid Sarker; Mohamed G Embaby; Khaled R Ahmed; Amer AbuGhazaleh
>
> **摘要:** Carbon dioxide (CO$_2$) emissions are critical indicators of both environmental impact and various industrial processes, including livestock management. We introduce CarboNeXT, a semantic segmentation framework for Optical Gas Imaging (OGI), designed to detect and quantify CO$_2$ emissions across diverse applications. Our approach integrates a multi-scale context aggregation network with UPerHead and auxiliary FCN components to effectively model both local details and global relationships in gas plume imagery. We contribute two novel datasets: (1) the Controlled Carbon Dioxide Release (CCR) dataset, which simulates gas leaks with systematically varied flow rates (10-100 SCCM), and (2) the Real Time Ankom (RTA) dataset, focusing on emissions from dairy cow rumen fluid in vitro experiments. Extensive evaluations demonstrate that CarboNeXT outperforms state-of-the-art methods, achieving 88.46% mIoU on CCR and 92.95% mIoU on RTA, with particular effectiveness in challenging low-flow scenarios. The model operates at 60.95 FPS, enabling real-time monitoring applications. Additionally, we propose CarboFormer, a lightweight variant with only 5.07M parameters that achieves 84.68 FPS, with competitive performance of 84.88% mIoU on CCR and 92.98% on RTA, making it suitable for resource-constrained platforms such as programmable drones. Our work advances both environmental sensing and precision livestock management by providing robust tools for CO$_2$ emission analysis, with a specific focus on livestock applications.
>
---
#### [new 075] WisWheat: A Three-Tiered Vision-Language Dataset for Wheat Management
- **分类: cs.CV**

- **简介: 该论文属于农业与人工智能交叉任务，旨在解决小麦管理中依赖人工、效率低的问题。作者构建了专用数据集WisWheat，包含三层结构：预训练、定量分析和指令微调，提升视觉语言模型在小麦管理中的性能，实验显示其效果优于通用模型。**

- **链接: [http://arxiv.org/pdf/2506.06084v1](http://arxiv.org/pdf/2506.06084v1)**

> **作者:** Bowen Yuan; Selena Song; Javier Fernandez; Yadan Luo; Mahsa Baktashmotlagh; Zijian Wang
>
> **摘要:** Wheat management strategies play a critical role in determining yield. Traditional management decisions often rely on labour-intensive expert inspections, which are expensive, subjective and difficult to scale. Recently, Vision-Language Models (VLMs) have emerged as a promising solution to enable scalable, data-driven management support. However, due to a lack of domain-specific knowledge, directly applying VLMs to wheat management tasks results in poor quantification and reasoning capabilities, ultimately producing vague or even misleading management recommendations. In response, we propose WisWheat, a wheat-specific dataset with a three-layered design to enhance VLM performance on wheat management tasks: (1) a foundational pretraining dataset of 47,871 image-caption pairs for coarsely adapting VLMs to wheat morphology; (2) a quantitative dataset comprising 7,263 VQA-style image-question-answer triplets for quantitative trait measuring tasks; and (3) an Instruction Fine-tuning dataset with 4,888 samples targeting biotic and abiotic stress diagnosis and management plan for different phenological stages. Extensive experimental results demonstrate that fine-tuning open-source VLMs (e.g., Qwen2.5 7B) on our dataset leads to significant performance improvements. Specifically, the Qwen2.5 VL 7B fine-tuned on our wheat instruction dataset achieves accuracy scores of 79.2% and 84.6% on wheat stress and growth stage conversation tasks respectively, surpassing even general-purpose commercial models such as GPT-4o by a margin of 11.9% and 34.6%.
>
---
#### [new 076] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频编辑任务，旨在解决现有视频扩散模型难以高效进行动态编辑的问题。作者提出了FADE方法，通过频率感知因子分解和频谱引导调制，充分利用预训练模型的先验知识，实现高质量、时间连贯的视频编辑。**

- **链接: [http://arxiv.org/pdf/2506.05934v1](http://arxiv.org/pdf/2506.05934v1)**

> **作者:** Yixuan Zhu; Haolin Wang; Shilin Ma; Wenliang Zhao; Yansong Tang; Lei Chen; Jie Zhou
>
> **备注:** Accepted by IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025
>
> **摘要:** Recent advancements in diffusion frameworks have significantly enhanced video editing, achieving high fidelity and strong alignment with textual prompts. However, conventional approaches using image diffusion models fall short in handling video dynamics, particularly for challenging temporal edits like motion adjustments. While current video diffusion models produce high-quality results, adapting them for efficient editing remains difficult due to the heavy computational demands that prevent the direct application of previous image editing techniques. To overcome these limitations, we introduce FADE, a training-free yet highly effective video editing approach that fully leverages the inherent priors from pre-trained video diffusion models via frequency-aware factorization. Rather than simply using these models, we first analyze the attention patterns within the video model to reveal how video priors are distributed across different components. Building on these insights, we propose a factorization strategy to optimize each component's specialized role. Furthermore, we devise spectrum-guided modulation to refine the sampling trajectory with frequency domain cues, preventing information leakage and supporting efficient, versatile edits while preserving the basic spatial and temporal structure. Extensive experiments on real-world videos demonstrate that our method consistently delivers high-quality, realistic and temporally coherent editing results both qualitatively and quantitatively. Code is available at https://github.com/EternalEvan/FADE .
>
---
#### [new 077] Where Is The Ball: 3D Ball Trajectory Estimation From 2D Monocular Tracking
- **分类: cs.CV**

- **简介: 该论文属于3D轨迹估计任务，旨在解决从2D单目跟踪序列中恢复3D球体轨迹的问题。作者提出了一种基于LSTM的管道，使用与相机位置无关的规范3D表示和中间表示，以提高估计的不变性和重投影一致性。方法在多个合成与真实数据集上表现优异，具备实际应用潜力。**

- **链接: [http://arxiv.org/pdf/2506.05763v1](http://arxiv.org/pdf/2506.05763v1)**

> **作者:** Puntawat Ponglertnapakorn; Supasorn Suwajanakorn
>
> **备注:** 11th International Workshop on Computer Vision in Sports (CVsports) at CVPR 2025
>
> **摘要:** We present a method for 3D ball trajectory estimation from a 2D tracking sequence. To overcome the ambiguity in 3D from 2D estimation, we design an LSTM-based pipeline that utilizes a novel canonical 3D representation that is independent of the camera's location to handle arbitrary views and a series of intermediate representations that encourage crucial invariance and reprojection consistency. We evaluated our method on four synthetic and three real datasets and conducted extensive ablation studies on our design choices. Despite training solely on simulated data, our method achieves state-of-the-art performance and can generalize to real-world scenarios with multiple trajectories, opening up a range of applications in sport analysis and virtual replay. Please visit our page: https://where-is-the-ball.github.io.
>
---
#### [new 078] S2GO: Streaming Sparse Gaussian Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文属于3D场景感知任务，旨在解决现有方法在动态驾驶场景建模中效率低、表达能力弱的问题。作者提出S2GO框架，采用流式稀疏查询传播与语义高斯解码，结合去噪渲染目标，实现高效、灵活的时序3D占据预测，取得更优性能与推理速度。**

- **链接: [http://arxiv.org/pdf/2506.05473v1](http://arxiv.org/pdf/2506.05473v1)**

> **作者:** Jinhyung Park; Yihan Hu; Chensheng Peng; Wenzhao Zheng; Kris Kitani; Wei Zhan
>
> **摘要:** Despite the demonstrated efficiency and performance of sparse query-based representations for perception, state-of-the-art 3D occupancy prediction methods still rely on voxel-based or dense Gaussian-based 3D representations. However, dense representations are slow, and they lack flexibility in capturing the temporal dynamics of driving scenes. Distinct from prior work, we instead summarize the scene into a compact set of 3D queries which are propagated through time in an online, streaming fashion. These queries are then decoded into semantic Gaussians at each timestep. We couple our framework with a denoising rendering objective to guide the queries and their constituent Gaussians in effectively capturing scene geometry. Owing to its efficient, query-based representation, S2GO achieves state-of-the-art performance on the nuScenes and KITTI occupancy benchmarks, outperforming prior art (e.g., GaussianWorld) by 1.5 IoU with 5.9x faster inference.
>
---
#### [new 079] Structured Labeling Enables Faster Vision-Language Models for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决当前视觉-语言模型（VLMs）在实际应用中的效率与数据冗余问题。作者构建了结构化数据集NuScenes-S，并提出了轻量级模型FastDrive，提升了推理速度与决策准确性。**

- **链接: [http://arxiv.org/pdf/2506.05442v1](http://arxiv.org/pdf/2506.05442v1)**

> **作者:** Hao Jiang; Chuan Hu; Yukang Shi; Yuan He; Ke Wang; Xi Zhang; Zhipeng Zhang
>
> **摘要:** Vision-Language Models (VLMs) offer a promising approach to end-to-end autonomous driving due to their human-like reasoning capabilities. However, troublesome gaps remains between current VLMs and real-world autonomous driving applications. One major limitation is that existing datasets with loosely formatted language descriptions are not machine-friendly and may introduce redundancy. Additionally, high computational cost and massive scale of VLMs hinder the inference speed and real-world deployment. To bridge the gap, this paper introduces a structured and concise benchmark dataset, NuScenes-S, which is derived from the NuScenes dataset and contains machine-friendly structured representations. Moreover, we present FastDrive, a compact VLM baseline with 0.9B parameters. In contrast to existing VLMs with over 7B parameters and unstructured language processing(e.g., LLaVA-1.5), FastDrive understands structured and concise descriptions and generates machine-friendly driving decisions with high efficiency. Extensive experiments show that FastDrive achieves competitive performance on structured dataset, with approximately 20% accuracy improvement on decision-making tasks, while surpassing massive parameter baseline in inference speed with over 10x speedup. Additionally, ablation studies further focus on the impact of scene annotations (e.g., weather, time of day) on decision-making tasks, demonstrating their importance on decision-making tasks in autonomous driving.
>
---
#### [new 080] CCLSTM: Coupled Convolutional Long-Short Term Memory Network for Occupancy Flow Forecasting
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的运动预测任务，旨在解决现有方法依赖高质量向量化输入和计算成本高的问题。作者提出CCLSTM模型，基于卷积操作，无需向量化输入或自注意力机制，实现高效、轻量级的端到端预测，在占用流场预测中达到最优性能。**

- **链接: [http://arxiv.org/pdf/2506.06128v1](http://arxiv.org/pdf/2506.06128v1)**

> **作者:** Peter Lengyel
>
> **摘要:** Predicting future states of dynamic agents is a fundamental task in autonomous driving. An expressive representation for this purpose is Occupancy Flow Fields, which provide a scalable and unified format for modeling motion, spatial extent, and multi-modal future distributions. While recent methods have achieved strong results using this representation, they often depend on high-quality vectorized inputs, which are unavailable or difficult to generate in practice, and the use of transformer-based architectures, which are computationally intensive and costly to deploy. To address these issues, we propose \textbf{Coupled Convolutional LSTM (CCLSTM)}, a lightweight, end-to-end trainable architecture based solely on convolutional operations. Without relying on vectorized inputs or self-attention mechanisms, CCLSTM effectively captures temporal dynamics and spatial occupancy-flow correlations using a compact recurrent convolutional structure. Despite its simplicity, CCLSTM achieves state-of-the-art performance on occupancy flow metrics and, as of this submission, ranks \(1^{\text{st}}\) in all metrics on the 2024 Waymo Occupancy and Flow Prediction Challenge leaderboard.
>
---
#### [new 081] Domain-RAG: Retrieval-Guided Compositional Image Generation for Cross-Domain Few-Shot Object Detection
- **分类: cs.CV**

- **简介: 该论文属于跨域少样本目标检测（CD-FSOD）任务，旨在仅用少量标注样本检测来自新域的未知目标。现有方法在保持类别准确和背景一致性方面存在不足。论文提出Domain-RAG，一种无需训练的检索引导图像生成框架，通过背景检索、生成与前景组合，实现高质量、域一致的样本生成，有效提升CD-FSOD性能。**

- **链接: [http://arxiv.org/pdf/2506.05872v1](http://arxiv.org/pdf/2506.05872v1)**

> **作者:** Yu Li; Xingyu Qiu; Yuqian Fu; Jie Chen; Tianwen Qian; Xu Zheng; Danda Pani Paudel; Yanwei Fu; Xuanjing Huang; Luc Van Gool; Yu-Gang Jiang
>
> **摘要:** Cross-Domain Few-Shot Object Detection (CD-FSOD) aims to detect novel objects with only a handful of labeled samples from previously unseen domains. While data augmentation and generative methods have shown promise in few-shot learning, their effectiveness for CD-FSOD remains unclear due to the need for both visual realism and domain alignment. Existing strategies, such as copy-paste augmentation and text-to-image generation, often fail to preserve the correct object category or produce backgrounds coherent with the target domain, making them non-trivial to apply directly to CD-FSOD. To address these challenges, we propose Domain-RAG, a training-free, retrieval-guided compositional image generation framework tailored for CD-FSOD. Domain-RAG consists of three stages: domain-aware background retrieval, domain-guided background generation, and foreground-background composition. Specifically, the input image is first decomposed into foreground and background regions. We then retrieve semantically and stylistically similar images to guide a generative model in synthesizing a new background, conditioned on both the original and retrieved contexts. Finally, the preserved foreground is composed with the newly generated domain-aligned background to form the generated image. Without requiring any additional supervision or training, Domain-RAG produces high-quality, domain-consistent samples across diverse tasks, including CD-FSOD, remote sensing FSOD, and camouflaged FSOD. Extensive experiments show consistent improvements over strong baselines and establish new state-of-the-art results. Codes will be released upon acceptance.
>
---
#### [new 082] Feedback Guidance of Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型中分类器无关引导（CFG）导致的多样性下降与记忆偏差问题。作者提出了反馈引导（FBG），通过动态调整引导强度，依据样本需求进行自调节，理论基础更强且表现优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.06085v1](http://arxiv.org/pdf/2506.06085v1)**

> **作者:** Koulischer Felix; Handke Florian; Deleu Johannes; Demeester Thomas; Ambrogioni Luca
>
> **备注:** Preprint. Article currently under review. Code is available at: https://github.com/FelixKoulischer/FBG_using_edm2
>
> **摘要:** While Classifier-Free Guidance (CFG) has become standard for improving sample fidelity in conditional diffusion models, it can harm diversity and induce memorization by applying constant guidance regardless of whether a particular sample needs correction. We propose FeedBack Guidance (FBG), which uses a state-dependent coefficient to self-regulate guidance amounts based on need. Our approach is derived from first principles by assuming the learned conditional distribution is linearly corrupted by the unconditional distribution, contrasting with CFG's implicit multiplicative assumption. Our scheme relies on feedback of its own predictions about the conditional signal informativeness to adapt guidance dynamically during inference, challenging the view of guidance as a fixed hyperparameter. The approach is benchmarked on ImageNet512x512, where it significantly outperforms Classifier-Free Guidance and is competitive to Limited Interval Guidance (LIG) while benefitting from a strong mathematical framework. On Text-To-Image generation, we demonstrate that, as anticipated, our approach automatically applies higher guidance scales for complex prompts than for simpler ones and that it can be easily combined with existing guidance schemes such as CFG or LIG.
>
---
#### [new 083] Dy3DGS-SLAM: Monocular 3D Gaussian Splatting SLAM for Dynamic Environments
- **分类: cs.CV**

- **简介: 该论文属于SLAM任务，旨在解决动态环境中基于单目RGB输入的实时定位与三维重建问题。现有方法多依赖RGB-D输入或难以应对动态干扰。为此，作者提出Dy3DGS-SLAM，融合光流和深度信息构建动态掩码，并设计运动损失优化位姿估计，实现高效跟踪与建图。**

- **链接: [http://arxiv.org/pdf/2506.05965v1](http://arxiv.org/pdf/2506.05965v1)**

> **作者:** Mingrui Li; Yiming Zhou; Hongxing Zhou; Xinggang Hu; Florian Roemer; Hongyu Wang; Ahmad Osman
>
> **摘要:** Current Simultaneous Localization and Mapping (SLAM) methods based on Neural Radiance Fields (NeRF) or 3D Gaussian Splatting excel in reconstructing static 3D scenes but struggle with tracking and reconstruction in dynamic environments, such as real-world scenes with moving elements. Existing NeRF-based SLAM approaches addressing dynamic challenges typically rely on RGB-D inputs, with few methods accommodating pure RGB input. To overcome these limitations, we propose Dy3DGS-SLAM, the first 3D Gaussian Splatting (3DGS) SLAM method for dynamic scenes using monocular RGB input. To address dynamic interference, we fuse optical flow masks and depth masks through a probabilistic model to obtain a fused dynamic mask. With only a single network iteration, this can constrain tracking scales and refine rendered geometry. Based on the fused dynamic mask, we designed a novel motion loss to constrain the pose estimation network for tracking. In mapping, we use the rendering loss of dynamic pixels, color, and depth to eliminate transient interference and occlusion caused by dynamic objects. Experimental results demonstrate that Dy3DGS-SLAM achieves state-of-the-art tracking and rendering in dynamic environments, outperforming or matching existing RGB-D methods.
>
---
#### [new 084] Talk2SAM: Text-Guided Semantic Enhancement for Complex-Shaped Object Segmentation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像分割任务，旨在解决复杂形状物体（如电线、自行车等）分割效果差的问题。通过结合文本引导的语义增强方法Talk2SAM，将CLIP文本特征映射到DINO空间并作为额外提示输入SAM-HQ模型，提升其对目标的关注能力。在多个数据集上验证了该方法优于现有模型，尤其在边界分割方面表现突出。**

- **链接: [http://arxiv.org/pdf/2506.05396v1](http://arxiv.org/pdf/2506.05396v1)**

> **作者:** Luka Vetoshkin; Dmitry Yudin
>
> **备注:** 14 pages, 7 figures, Submitted to the conference
>
> **摘要:** Segmenting objects with complex shapes, such as wires, bicycles, or structural grids, remains a significant challenge for current segmentation models, including the Segment Anything Model (SAM) and its high-quality variant SAM-HQ. These models often struggle with thin structures and fine boundaries, leading to poor segmentation quality. We propose Talk2SAM, a novel approach that integrates textual guidance to improve segmentation of such challenging objects. The method uses CLIP-based embeddings derived from user-provided text prompts to identify relevant semantic regions, which are then projected into the DINO feature space. These features serve as additional prompts for SAM-HQ, enhancing its ability to focus on the target object. Beyond improving segmentation accuracy, Talk2SAM allows user-controllable segmentation, enabling disambiguation of objects within a single bounding box based on textual input. We evaluate our approach on three benchmarks: BIG, ThinObject5K, and DIS5K. Talk2SAM consistently outperforms SAM-HQ, achieving up to +5.9\% IoU and +8.3\% boundary IoU improvements. Our results demonstrate that incorporating natural language guidance provides a flexible and effective means for precise object segmentation, particularly in cases where traditional prompt-based methods fail. The source code is available on GitHub: https://github.com/richlukich/Talk2SAM
>
---
#### [new 085] CryoFastAR: Fast Cryo-EM Ab Initio Reconstruction Made Easy
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决冷冻电镜（cryo-EM）图像中低信噪比和对比度传递函数失真导致的快速从头重建难题。作者提出了CryoFastAR模型，通过端到端预测姿态并结合渐进训练策略，提升了重建速度与鲁棒性，实现了媲美传统迭代方法的高质量结果。**

- **链接: [http://arxiv.org/pdf/2506.05864v1](http://arxiv.org/pdf/2506.05864v1)**

> **作者:** Jiakai Zhang; Shouchen Zhou; Haizhao Dai; Xinhang Liu; Peihao Wang; Zhiwen Fan; Yuan Pei; Jingyi Yu
>
> **摘要:** Pose estimation from unordered images is fundamental for 3D reconstruction, robotics, and scientific imaging. Recent geometric foundation models, such as DUSt3R, enable end-to-end dense 3D reconstruction but remain underexplored in scientific imaging fields like cryo-electron microscopy (cryo-EM) for near-atomic protein reconstruction. In cryo-EM, pose estimation and 3D reconstruction from unordered particle images still depend on time-consuming iterative optimization, primarily due to challenges such as low signal-to-noise ratios (SNR) and distortions from the contrast transfer function (CTF). We introduce CryoFastAR, the first geometric foundation model that can directly predict poses from Cryo-EM noisy images for Fast ab initio Reconstruction. By integrating multi-view features and training on large-scale simulated cryo-EM data with realistic noise and CTF modulations, CryoFastAR enhances pose estimation accuracy and generalization. To enhance training stability, we propose a progressive training strategy that first allows the model to extract essential features under simpler conditions before gradually increasing difficulty to improve robustness. Experiments show that CryoFastAR achieves comparable quality while significantly accelerating inference over traditional iterative approaches on both synthetic and real datasets.
>
---
#### [new 086] BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading
- **分类: cs.CV**

- **简介: 该论文属于三维人脸建模与渲染任务，旨在解决高分辨率、可动态控制且支持复杂光照变化的虚拟头像重建问题。作者提出了BecomingLit方法，使用3D高斯图元和混合神经着色技术，结合新构建的多视角、多光照数据集，实现了高质量、可实时渲染的可重光照头像，并能通过单目视频驱动动画。**

- **链接: [http://arxiv.org/pdf/2506.06271v1](http://arxiv.org/pdf/2506.06271v1)**

> **作者:** Jonathan Schmidt; Simon Giebenhain; Matthias Niessner
>
> **备注:** Project Page: see https://jonathsch.github.io/becominglit/ ; YouTube Video: see https://youtu.be/xPyeIqKdszA
>
> **摘要:** We introduce BecomingLit, a novel method for reconstructing relightable, high-resolution head avatars that can be rendered from novel viewpoints at interactive rates. Therefore, we propose a new low-cost light stage capture setup, tailored specifically towards capturing faces. Using this setup, we collect a novel dataset consisting of diverse multi-view sequences of numerous subjects under varying illumination conditions and facial expressions. By leveraging our new dataset, we introduce a new relightable avatar representation based on 3D Gaussian primitives that we animate with a parametric head model and an expression-dependent dynamics module. We propose a new hybrid neural shading approach, combining a neural diffuse BRDF with an analytical specular term. Our method reconstructs disentangled materials from our dynamic light stage recordings and enables all-frequency relighting of our avatars with both point lights and environment maps. In addition, our avatars can easily be animated and controlled from monocular videos. We validate our approach in extensive experiments on our dataset, where we consistently outperform existing state-of-the-art methods in relighting and reenactment by a significant margin.
>
---
#### [new 087] A VLM-based Method for Visual Anomaly Detection in Robotic Scientific Laboratories
- **分类: cs.CV**

- **简介: 该论文属于视觉异常检测任务，旨在解决机器人科学实验室中实验流程异常识别问题。作者提出了一种基于视觉语言模型（VLM）的推理方法，通过不同监督级别的提示配置进行检测，并构建了专用基准进行评估。实验表明，随着上下文信息增加，检测准确性提升，验证了方法的有效性与适应性。**

- **链接: [http://arxiv.org/pdf/2506.05405v1](http://arxiv.org/pdf/2506.05405v1)**

> **作者:** Shiwei Lin; Chenxu Wang; Xiaozhen Ding; Yi Wang; Boyuan Du; Lei Song; Chenggang Wang; Huaping Liu
>
> **摘要:** In robot scientific laboratories, visual anomaly detection is important for the timely identification and resolution of potential faults or deviations. It has become a key factor in ensuring the stability and safety of experimental processes. To address this challenge, this paper proposes a VLM-based visual reasoning approach that supports different levels of supervision through four progressively informative prompt configurations. To systematically evaluate its effectiveness, we construct a visual benchmark tailored for process anomaly detection in scientific workflows. Experiments on two representative vision-language models show that detection accuracy improves as more contextual information is provided, confirming the effectiveness and adaptability of the proposed reasoning approach for process anomaly detection in scientific workflows. Furthermore, real-world validations at selected experimental steps confirm that first-person visual observation can effectively identify process-level anomalies. This work provides both a data-driven foundation and an evaluation framework for vision anomaly detection in scientific experiment workflows.
>
---
#### [new 088] Text2Stereo: Repurposing Stable Diffusion for Stereo Generation with Consistency Rewards
- **分类: cs.CV**

- **简介: 该论文属于文本生成立体图像任务，旨在解决缺乏大规模立体数据导致模型难以训练的问题。论文提出Text2Stereo方法，基于Stable Diffusion进行微调，并引入提示对齐与立体一致性奖励函数，以提升生成质量与文本匹配度。**

- **链接: [http://arxiv.org/pdf/2506.05367v1](http://arxiv.org/pdf/2506.05367v1)**

> **作者:** Aakash Garg; Libing Zeng; Andrii Tsarov; Nima Khademi Kalantari
>
> **摘要:** In this paper, we propose a novel diffusion-based approach to generate stereo images given a text prompt. Since stereo image datasets with large baselines are scarce, training a diffusion model from scratch is not feasible. Therefore, we propose leveraging the strong priors learned by Stable Diffusion and fine-tuning it on stereo image datasets to adapt it to the task of stereo generation. To improve stereo consistency and text-to-image alignment, we further tune the model using prompt alignment and our proposed stereo consistency reward functions. Comprehensive experiments demonstrate the superiority of our approach in generating high-quality stereo images across diverse scenarios, outperforming existing methods.
>
---
#### [new 089] Q-Ponder: A Unified Training Pipeline for Reasoning-based Visual Quality Assessment
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于视觉质量评估任务，旨在解决质量评分与可解释性难以兼顾的问题。作者提出统一训练框架Q-Ponder，通过两阶段训练提升评分准确性和推理一致性，实现跨域性能优化，并在多个基准上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2506.05384v1](http://arxiv.org/pdf/2506.05384v1)**

> **作者:** Zhuoxuan Cai; Jian Zhang; Xinbin Yuan; Pengtao Jiang; Wenxiang Chen; Bowen Tang; Lujian Yao; Qiyuan Wang; Jinwen Chen; Bo Li
>
> **摘要:** Recent studies demonstrate that multimodal large language models (MLLMs) can proficiently evaluate visual quality through interpretable assessments. However, existing approaches typically treat quality scoring and reasoning descriptions as separate tasks with disjoint optimization objectives, leading to a trade-off: models adept at quality reasoning descriptions struggle with precise score regression, while score-focused models lack interpretability. This limitation hinders the full potential of MLLMs in visual quality assessment, where accuracy and interpretability should be mutually reinforcing. To address this, we propose a unified two-stage training framework comprising a cold-start stage and a reinforcement learning-based fine-tuning stage. Specifically, in the first stage, we distill high-quality data from a teacher model through expert-designed prompts, initializing reasoning capabilities via cross-entropy loss supervision. In the second stage, we introduce a novel reward with Group Relative Policy Optimization (GRPO) to jointly optimize scoring accuracy and reasoning consistency. We designate the models derived from these two stages as Q-Ponder-CI and Q-Ponder. Extensive experiments show that Q-Ponder achieves state-of-the-art (SOTA) performance on quality score regression benchmarks, delivering up to 6.5% higher SRCC on cross-domain datasets. Furthermore, Q-Ponder significantly outperforms description-based SOTA models, including its teacher model Qwen-2.5-VL-72B, particularly in description accuracy and reasonableness, demonstrating the generalization potential over diverse tasks.
>
---
#### [new 090] TissUnet: Improved Extracranial Tissue and Cranium Segmentation for Children through Adulthood
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TissUnet，用于改进儿童至成人的颅外组织和颅骨分割。任务是医学图像分割，旨在解决脑MRI中颅外组织量化不足的问题。作者训练并验证了一个深度学习模型，使用155对MRI-CT扫描数据，展示了其在多个数据集上的优异性能，优于现有方法，支持大规模研究应用。**

- **链接: [http://arxiv.org/pdf/2506.05660v1](http://arxiv.org/pdf/2506.05660v1)**

> **作者:** Markian Mandzak; Elvira Yang; Anna Zapaishchykova; Yu-Hui Chen; Lucas Heilbroner; John Zielke; Divyanshu Tak; Reza Mojahed-Yazdi; Francesca Romana Mussa; Zezhong Ye; Sridhar Vajapeyam; Viviana Benitez; Ralph Salloum; Susan N. Chi; Houman Sotoudeh; Jakob Seidlitz; Sabine Mueller; Hugo J. W. L. Aerts; Tina Y. Poussaint; Benjamin H. Kann
>
> **备注:** 44 pages, 4 tables, 6 figures, supplementary material
>
> **摘要:** Extracranial tissues visible on brain magnetic resonance imaging (MRI) may hold significant value for characterizing health conditions and clinical decision-making, yet they are rarely quantified. Current tools have not been widely validated, particularly in settings of developing brains or underlying pathology. We present TissUnet, a deep learning model that segments skull bone, subcutaneous fat, and muscle from routine three-dimensional T1-weighted MRI, with or without contrast enhancement. The model was trained on 155 paired MRI-computed tomography (CT) scans and validated across nine datasets covering a wide age range and including individuals with brain tumors. In comparison to AI-CT-derived labels from 37 MRI-CT pairs, TissUnet achieved a median Dice coefficient of 0.79 [IQR: 0.77-0.81] in a healthy adult cohort. In a second validation using expert manual annotations, median Dice was 0.83 [IQR: 0.83-0.84] in healthy individuals and 0.81 [IQR: 0.78-0.83] in tumor cases, outperforming previous state-of-the-art method. Acceptability testing resulted in an 89% acceptance rate after adjudication by a tie-breaker(N=108 MRIs), and TissUnet demonstrated excellent performance in the blinded comparative review (N=45 MRIs), including both healthy and tumor cases in pediatric populations. TissUnet enables fast, accurate, and reproducible segmentation of extracranial tissues, supporting large-scale studies on craniofacial morphology, treatment effects, and cardiometabolic risk using standard brain T1w MRI.
>
---
#### [new 091] Can ChatGPT Perform Image Splicing Detection? A Preliminary Study
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文研究GPT-4V在图像拼接检测中的能力，属于图像取证任务。旨在探索其无需微调即可检测图像篡改的潜力。作者采用多种提示策略，在CASIA数据集上进行实验，结果显示GPT-4V具备一定检测能力，尤其在零样本设置下表现良好。**

- **链接: [http://arxiv.org/pdf/2506.05358v1](http://arxiv.org/pdf/2506.05358v1)**

> **作者:** Souradip Nath
>
> **摘要:** Multimodal Large Language Models (MLLMs) like GPT-4V are capable of reasoning across text and image modalities, showing promise in a variety of complex vision-language tasks. In this preliminary study, we investigate the out-of-the-box capabilities of GPT-4V in the domain of image forensics, specifically, in detecting image splicing manipulations. Without any task-specific fine-tuning, we evaluate GPT-4V using three prompting strategies: Zero-Shot (ZS), Few-Shot (FS), and Chain-of-Thought (CoT), applied over a curated subset of the CASIA v2.0 splicing dataset. Our results show that GPT-4V achieves competitive detection performance in zero-shot settings (more than 85% accuracy), with CoT prompting yielding the most balanced trade-off across authentic and spliced images. Qualitative analysis further reveals that the model not only detects low-level visual artifacts but also draws upon real-world contextual knowledge such as object scale, semantic consistency, and architectural facts, to identify implausible composites. While GPT-4V lags behind specialized state-of-the-art splicing detection models, its generalizability, interpretability, and encyclopedic reasoning highlight its potential as a flexible tool in image forensics.
>
---
#### [new 092] OpenRR-5k: A Large-Scale Benchmark for Reflection Removal in the Wild
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的单图像反射去除（SIRR）任务，旨在解决现有方法因缺乏高质量、大规模数据集而受限的问题。作者构建了一个包含5,300对反射与干净图像的数据集OpenRR-5k，并基于U-Net模型验证其有效性，推动相关研究发展。**

- **链接: [http://arxiv.org/pdf/2506.05482v1](http://arxiv.org/pdf/2506.05482v1)**

> **作者:** Jie Cai; Kangning Yang; Ling Ouyang; Lan Fu; Jiaming Ding; Jinglin Shen; Zibo Meng
>
> **摘要:** Removing reflections is a crucial task in computer vision, with significant applications in photography and image enhancement. Nevertheless, existing methods are constrained by the absence of large-scale, high-quality, and diverse datasets. In this paper, we present a novel benchmark for Single Image Reflection Removal (SIRR). We have developed a large-scale dataset containing 5,300 high-quality, pixel-aligned image pairs, each consisting of a reflection image and its corresponding clean version. Specifically, the dataset is divided into two parts: 5,000 images are used for training, and 300 images are used for validation. Additionally, we have included 100 real-world testing images without ground truth (GT) to further evaluate the practical performance of reflection removal methods. All image pairs are precisely aligned at the pixel level to guarantee accurate supervision. The dataset encompasses a broad spectrum of real-world scenarios, featuring various lighting conditions, object types, and reflection patterns, and is segmented into training, validation, and test sets to facilitate thorough evaluation. To validate the usefulness of our dataset, we train a U-Net-based model and evaluate it using five widely-used metrics, including PSNR, SSIM, LPIPS, DISTS, and NIQE. We will release both the dataset and the code on https://github.com/caijie0620/OpenRR-5k to facilitate future research in this field.
>
---
#### [new 093] SatelliteFormula: Multi-Modal Symbolic Regression from Remote Sensing Imagery for Physics Discovery
- **分类: cs.CV**

- **简介: 论文提出“SatelliteFormula”，一种从多光谱遥感图像中进行多模态符号回归的框架，旨在发现物理可解释的表达式。任务是解决传统方法在处理高维遥感数据时缺乏可解释性和准确性的问题。工作包括结合视觉Transformer提取空间-光谱特征，并引入物理约束优化符号表达式，提升环境变量建模的稳定性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.06176v1](http://arxiv.org/pdf/2506.06176v1)**

> **作者:** Zhenyu Yu; Mohd. Yamani Idna Idris; Pei Wang; Yuelong Xia; Fei Ma; Rizwan Qureshi
>
> **摘要:** We propose SatelliteFormula, a novel symbolic regression framework that derives physically interpretable expressions directly from multi-spectral remote sensing imagery. Unlike traditional empirical indices or black-box learning models, SatelliteFormula combines a Vision Transformer-based encoder for spatial-spectral feature extraction with physics-guided constraints to ensure consistency and interpretability. Existing symbolic regression methods struggle with the high-dimensional complexity of multi-spectral data; our method addresses this by integrating transformer representations into a symbolic optimizer that balances accuracy and physical plausibility. Extensive experiments on benchmark datasets and remote sensing tasks demonstrate superior performance, stability, and generalization compared to state-of-the-art baselines. SatelliteFormula enables interpretable modeling of complex environmental variables, bridging the gap between data-driven learning and physical understanding.
>
---
#### [new 094] VideoChat-A1: Thinking with Long Videos by Chain-of-Shot Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决现有模型在长视频多模态分析中的上下文误判问题。作者提出了VideoChat-A1，通过“链式镜头推理”范式，逐步选择并深入分析相关视频片段，提升长视频问答性能，实验显示其在多个基准上达到SOTA，并优于主流模型。**

- **链接: [http://arxiv.org/pdf/2506.06097v1](http://arxiv.org/pdf/2506.06097v1)**

> **作者:** Zikang Wang; Boyu Chen; Zhengrong Yue; Yi Wang; Yu Qiao; Limin Wang; Yali Wang
>
> **摘要:** The recent advance in video understanding has been driven by multimodal large language models (MLLMs). But these MLLMs are good at analyzing short videos, while suffering from difficulties in understanding videos with a longer context. To address this difficulty, several agent paradigms have recently been proposed, using MLLMs as agents for retrieving extra contextual knowledge in a long video. However, most existing agents ignore the key fact that a long video is composed with multiple shots, i.e., to answer the user question from a long video, it is critical to deeply understand its relevant shots like human. Without such insight, these agents often mistakenly find redundant even noisy temporal context, restricting their capacity for long video understanding. To fill this gap, we propose VideoChat-A1, a novel long video agent paradigm. Different from the previous works, our VideoChat-A1 can deeply think with long videos, via a distinct chain-of-shot reasoning paradigm. More specifically, it can progressively select the relevant shots of user question, and look into these shots in a coarse-to-fine partition. By multi-modal reasoning along the shot chain, VideoChat-A1 can effectively mimic step-by-step human thinking process, allowing to interactively discover preferable temporal context for thoughtful understanding in long videos. Extensive experiments show that, our VideoChat-A1 achieves the state-of-the-art performance on the mainstream long video QA benchmarks, e.g., it achieves 77.0 on VideoMME and 70.1 on EgoSchema, outperforming its strong baselines (e.g., Intern2.5VL-8B and InternVideo2.5-8B), by up to 10.8\% and 6.2\%. Compared to leading close-source GPT-4o and Gemini 1.5 Pro, VideoChat-A1 offers competitive accuracy, but with 7\% input frames and 12\% inference time on average.
>
---
#### [new 095] Cross-View Multi-Modal Segmentation @ Ego-Exo4D Challenges 2025
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态语义分割任务，旨在解决跨视角（如第一人称与第三人称）物体对应问题。作者提出了一种多模态条件融合模块和跨视图物体对齐模块，以提升跨视角下物体定位和一致性匹配的效果。最终在Ego-Exo4D挑战赛中取得第二名。**

- **链接: [http://arxiv.org/pdf/2506.05856v1](http://arxiv.org/pdf/2506.05856v1)**

> **作者:** Yuqian Fu; Runze Wang; Yanwei Fu; Danda Pani Paudel; Luc Van Gool
>
> **备注:** The 2nd Price Award of EgoExo4D Relations, Second Joint EgoVis Workshop with CVPR2025, technical report paper is accepted by CVPRW 25
>
> **摘要:** In this report, we present a cross-view multi-modal object segmentation approach for the object correspondence task in the Ego-Exo4D Correspondence Challenges 2025. Given object queries from one perspective (e.g., ego view), the goal is to predict the corresponding object masks in another perspective (e.g., exo view). To tackle this task, we propose a multimodal condition fusion module that enhances object localization by leveraging both visual masks and textual descriptions as segmentation conditions. Furthermore, to address the visual domain gap between ego and exo views, we introduce a cross-view object alignment module that enforces object-level consistency across perspectives, thereby improving the model's robustness to viewpoint changes. Our proposed method ranked second on the leaderboard of the large-scale Ego-Exo4D object correspondence benchmark. Code will be made available at https://github.com/lovelyqian/ObjectRelator.
>
---
#### [new 096] Enhancing Orthopox Image Classification Using Hybrid Machine Learning and Deep Learning Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决正痘病毒（Orthopoxvirus）感染的准确诊断问题。由于传统方法耗时且依赖专家，论文提出一种结合机器学习与预训练深度学习模型的混合方法，用于提取深层特征并提升分类性能，无需数据增强，同时保持高效训练和推理能力。**

- **链接: [http://arxiv.org/pdf/2506.06007v1](http://arxiv.org/pdf/2506.06007v1)**

> **作者:** Alejandro Puente-Castro; Enrique Fernandez-Blanco; Daniel Rivero; Andres Molares-Ulloa
>
> **摘要:** Orthopoxvirus infections must be accurately classified from medical pictures for an easy and early diagnosis and epidemic prevention. The necessity for automated and scalable solutions is highlighted by the fact that traditional diagnostic techniques can be time-consuming and require expert interpretation and there are few and biased data sets of the different types of Orthopox. In order to improve classification performance and lower computational costs, a hybrid strategy is put forth in this paper that uses Machine Learning models combined with pretrained Deep Learning models to extract deep feature representations without the need for augmented data. The findings show that this feature extraction method, when paired with other methods in the state-of-the-art, produces excellent classification outcomes while preserving training and inference efficiency. The proposed approach demonstrates strong generalization and robustness across multiple evaluation settings, offering a scalable and interpretable solution for real-world clinical deployment.
>
---
#### [new 097] Coordinated Robustness Evaluation Framework for Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言模型的鲁棒性评估任务，旨在解决模型在图像和文本扰动下的脆弱性问题。作者提出了一种协调攻击策略，通过训练通用代理模型生成跨模态对抗扰动，并在多个数据集上验证了其对多模态模型（如instruct-BLIP、ViLT）的有效性。**

- **链接: [http://arxiv.org/pdf/2506.05429v1](http://arxiv.org/pdf/2506.05429v1)**

> **作者:** Ashwin Ramesh Babu; Sajad Mousavi; Vineet Gundecha; Sahand Ghorbanpour; Avisek Naug; Antonio Guillen; Ricardo Luna Gutierrez; Soumyendu Sarkar
>
> **备注:** Accepted: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) 2025
>
> **摘要:** Vision-language models, which integrate computer vision and natural language processing capabilities, have demonstrated significant advancements in tasks such as image captioning and visual question and answering. However, similar to traditional models, they are susceptible to small perturbations, posing a challenge to their robustness, particularly in deployment scenarios. Evaluating the robustness of these models requires perturbations in both the vision and language modalities to learn their inter-modal dependencies. In this work, we train a generic surrogate model that can take both image and text as input and generate joint representation which is further used to generate adversarial perturbations for both the text and image modalities. This coordinated attack strategy is evaluated on the visual question and answering and visual reasoning datasets using various state-of-the-art vision-language models. Our results indicate that the proposed strategy outperforms other multi-modal attacks and single-modality attacks from the recent literature. Our results demonstrate their effectiveness in compromising the robustness of several state-of-the-art pre-trained multi-modal models such as instruct-BLIP, ViLT and others.
>
---
#### [new 098] TerraFM: A Scalable Foundation Model for Unified Multisensor Earth Observation
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分析任务，旨在解决现有地球观测模型因训练数据规模和光谱多样性不足导致的泛化能力有限问题。作者提出了TerraFM，一种可扩展的自监督学习模型，通过融合Sentinel-1和Sentinel-2全球影像，并采用跨模态注意力机制与对比学习策略，实现对雷达与光学数据的统一处理，提升了分类与分割任务的性能。**

- **链接: [http://arxiv.org/pdf/2506.06281v1](http://arxiv.org/pdf/2506.06281v1)**

> **作者:** Muhammad Sohail Danish; Muhammad Akhtar Munir; Syed Roshaan Ali Shah; Muhammad Haris Khan; Rao Muhammad Anwer; Jorma Laaksonen; Fahad Shahbaz Khan; Salman Khan
>
> **摘要:** Modern Earth observation (EO) increasingly leverages deep learning to harness the scale and diversity of satellite imagery across sensors and regions. While recent foundation models have demonstrated promising generalization across EO tasks, many remain limited by the scale, geographical coverage, and spectral diversity of their training data, factors critical for learning globally transferable representations. In this work, we introduce TerraFM, a scalable self-supervised learning model that leverages globally distributed Sentinel-1 and Sentinel-2 imagery, combined with large spatial tiles and land-cover aware sampling to enrich spatial and semantic coverage. By treating sensing modalities as natural augmentations in our self-supervised approach, we unify radar and optical inputs via modality-specific patch embeddings and adaptive cross-attention fusion. Our training strategy integrates local-global contrastive learning and introduces a dual-centering mechanism that incorporates class-frequency-aware regularization to address long-tailed distributions in land cover.TerraFM achieves strong generalization on both classification and segmentation tasks, outperforming prior models on GEO-Bench and Copernicus-Bench. Our code and pretrained models are publicly available at: https://github.com/mbzuai-oryx/TerraFM .
>
---
#### [new 099] Can Vision Language Models Infer Human Gaze Direction? A Controlled Study
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型（VLMs）是否具备推断人类注视方向的能力。任务属于人机交互与人工智能领域，旨在解决VLMs在理解人类注意力方面的局限性。作者通过控制实验评估111个VLMs的表现，并与人类对比，分析其行为特征。**

- **链接: [http://arxiv.org/pdf/2506.05412v1](http://arxiv.org/pdf/2506.05412v1)**

> **作者:** Zory Zhang; Pinyuan Feng; Bingyang Wang; Tianwei Zhao; Suyang Yu; Qingying Gao; Hokin Deng; Ziqiao Ma; Yijiang Li; Dezhi Luo
>
> **备注:** Preprint under review. Project page at https://grow-ai-like-a-child.github.io/gaze/
>
> **摘要:** Gaze-referential inference--the ability to infer what others are looking at--is a critical component of a theory of mind that underpins natural human-AI interaction. In a controlled study, we evaluated this skill across 111 Vision Language Models (VLMs) using photos taken with manipulated difficulty and variability, comparing performance with that of human participants (N = 65), and analyzed behaviors using mixed-effects models. We found that 94 of the 111 VLMs failed to do better than random guessing, while humans achieved near-ceiling accuracy. VLMs even respond with each choice almost equally frequently. Are they randomly guessing? Although most VLMs struggle, when we zoom in on five of the top-tier VLMs with above-chance performance, we find that their performance declined with increasing task difficulty but varied only slightly across different prompts and scene objects. These behavioral features cannot be explained by considering them as random guessers. Instead, they likely use a combination of heuristics and guessing such that their performance is subject to the task difficulty but robust to perceptual variations. This suggests that VLMs, lacking gaze inference capability, have yet to become technologies that can naturally interact with humans, but the potential remains.
>
---
#### [new 100] An Independent Discriminant Network Towards Identification of Counterfeit Images and Videos
- **分类: cs.CV**

- **简介: 该论文属于图像与视频取证任务，旨在解决在线平台上伪造内容泛滥的问题。作者提出了一种基于InceptionResNetV2的独立判别网络，用于识别由生成对抗网络（GAN）生成的虚假图像或视频，并构建了一个可检测伪造内容的平台，以辅助刑事侦查和证据鉴定。**

- **链接: [http://arxiv.org/pdf/2506.05377v1](http://arxiv.org/pdf/2506.05377v1)**

> **作者:** Shayantani Kar; B. Shresth Bhimrajka; Aditya Kumar; Sahil Gupta; Sourav Ghosh; Subhamita Mukherjee; Shauvik Paul
>
> **备注:** This research was conducted by student and professor co-authors from Techno Main Salt Lake, with co-author Sourav Ghosh serving as an alumni mentor in an invited capacity -- distinct from his primary affiliation and pre-approved by his employer. This preprint presents research originally completed in early 2023 and published in IETE Journal of Research in 2025
>
> **摘要:** Rapid spread of false images and videos on online platforms is an emerging problem. Anyone may add, delete, clone or modify people and entities from an image using various editing software which are readily available. This generates false and misleading proof to hide the crime. Now-a-days, these false and counterfeit images and videos are flooding on the internet. These spread false information. Many methods are available in literature for detecting those counterfeit contents but new methods of counterfeiting are also evolving. Generative Adversarial Networks (GAN) are observed to be one effective method as it modifies the context and definition of images producing plausible results via image-to-image translation. This work uses an independent discriminant network that can identify GAN generated image or video. A discriminant network has been created using a convolutional neural network based on InceptionResNetV2. The article also proposes a platform where users can detect forged images and videos. This proposed work has the potential to help the forensics domain to detect counterfeit videos and hidden criminal evidence towards the identification of criminal activities.
>
---
#### [new 101] Implicit Neural Representation for Video Restoration
- **分类: cs.CV**

- **简介: 该论文属于视频超分辨率任务，旨在解决现有方法在未见尺度和噪声情况下泛化能力差的问题。作者提出VR-INR，基于隐式神经表示，仅训练于×4上采样，却能零样本处理任意放大倍数与去噪，提升重建质量与细节保留效果。**

- **链接: [http://arxiv.org/pdf/2506.05488v1](http://arxiv.org/pdf/2506.05488v1)**

> **作者:** Mary Aiyetigbo; Wanqi Yuan; Feng Luo; Nianyi Li
>
> **摘要:** High-resolution (HR) videos play a crucial role in many computer vision applications. Although existing video restoration (VR) methods can significantly enhance video quality by exploiting temporal information across video frames, they are typically trained for fixed upscaling factors and lack the flexibility to handle scales or degradations beyond their training distribution. In this paper, we introduce VR-INR, a novel video restoration approach based on Implicit Neural Representations (INRs) that is trained only on a single upscaling factor ($\times 4$) but generalizes effectively to arbitrary, unseen super-resolution scales at test time. Notably, VR-INR also performs zero-shot denoising on noisy input, despite never having seen noisy data during training. Our method employs a hierarchical spatial-temporal-texture encoding framework coupled with multi-resolution implicit hash encoding, enabling adaptive decoding of high-resolution and noise-suppressed frames from low-resolution inputs at any desired magnification. Experimental results show that VR-INR consistently maintains high-quality reconstructions at unseen scales and noise during training, significantly outperforming state-of-the-art approaches in sharpness, detail preservation, and denoising efficacy.
>
---
#### [new 102] EASG-Bench: Video Q&A Benchmark with Egocentric Action Scene Graphs
- **分类: cs.CV**

- **简介: 该论文属于视频问答任务，旨在解决第一视角视频中复杂场景的时空关系理解问题。作者构建了基于动态场景图的EASG-Bench基准，并提出系统评估框架，评估语言模型与视频大模型表现，发现时序理解存在显著差距，推动长时视频理解研究。**

- **链接: [http://arxiv.org/pdf/2506.05787v1](http://arxiv.org/pdf/2506.05787v1)**

> **作者:** Ivan Rodin; Tz-Ying Wu; Kyle Min; Sharath Nittur Sridhar; Antonino Furnari; Subarna Tripathi; Giovanni Maria Farinella
>
> **摘要:** We introduce EASG-Bench, a question-answering benchmark for egocentric videos where the question-answering pairs are created from spatio-temporally grounded dynamic scene graphs capturing intricate relationships among actors, actions, and objects. We propose a systematic evaluation framework and evaluate several language-only and video large language models (video-LLMs) on this benchmark. We observe a performance gap in language-only and video-LLMs, especially on questions focusing on temporal ordering, thus identifying a research gap in the area of long-context video understanding. To promote the reproducibility of our findings and facilitate further research, the benchmark and accompanying code are available at the following GitHub page: https://github.com/fpv-iplab/EASG-bench.
>
---
#### [new 103] Bidirectional Image-Event Guided Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决传统相机在极端低光条件下细节丢失和运动模糊的问题。论文提出了一种双向引导的低光图像增强框架BiLIE，通过事件特征增强模块和双向交叉注意力融合机制，抑制噪声并提升图像质量。此外，还构建了一个高质量的新数据集RELIE，实验表明其方法在PSNR和LPIPS指标上优于现有技术。**

- **链接: [http://arxiv.org/pdf/2506.06120v1](http://arxiv.org/pdf/2506.06120v1)**

> **作者:** Zhanwen Liu; Huanna Song; Yang Wang; Nan Yang; Shangyu Xie; Yisheng An; Xiangmo Zhao
>
> **摘要:** Under extreme low-light conditions, traditional frame-based cameras, due to their limited dynamic range and temporal resolution, face detail loss and motion blur in captured images. To overcome this bottleneck, researchers have introduced event cameras and proposed event-guided low-light image enhancement algorithms. However, these methods neglect the influence of global low-frequency noise caused by dynamic lighting conditions and local structural discontinuities in sparse event data. To address these issues, we propose an innovative Bidirectional guided Low-light Image Enhancement framework (BiLIE). Specifically, to mitigate the significant low-frequency noise introduced by global illumination step changes, we introduce the frequency high-pass filtering-based Event Feature Enhancement (EFE) module at the event representation level to suppress the interference of low-frequency information, and preserve and highlight the high-frequency edges.Furthermore, we design a Bidirectional Cross Attention Fusion (BCAF) mechanism to acquire high-frequency structures and edges while suppressing structural discontinuities and local noise introduced by sparse event guidance, thereby generating smoother fused representations.Additionally, considering the poor visual quality and color bias in existing datasets, we provide a new dataset (RELIE), with high-quality ground truth through a reliable enhancement scheme. Extensive experimental results demonstrate that our proposed BiLIE outperforms state-of-the-art methods by 0.96dB in PSNR and 0.03 in LPIPS.
>
---
#### [new 104] F2T2-HiT: A U-Shaped FFT Transformer and Hierarchical Transformer for Reflection Removal
- **分类: cs.CV**

- **简介: 该论文属于图像处理任务，旨在解决单幅图像中因玻璃反射导致的反射去除问题。针对反射在强度、形状、光源等方面的复杂性，作者提出了一种结合快速傅里叶变换（FFT）与层次化Transformer的U型网络结构F2T2-HiT，通过频域信息和多尺度特征提取提升反射去除效果。**

- **链接: [http://arxiv.org/pdf/2506.05489v1](http://arxiv.org/pdf/2506.05489v1)**

> **作者:** Jie Cai; Kangning Yang; Ling Ouyang; Lan Fu; Jiaming Ding; Huiming Sun; Chiu Man Ho; Zibo Meng
>
> **摘要:** Single Image Reflection Removal (SIRR) technique plays a crucial role in image processing by eliminating unwanted reflections from the background. These reflections, often caused by photographs taken through glass surfaces, can significantly degrade image quality. SIRR remains a challenging problem due to the complex and varied reflections encountered in real-world scenarios. These reflections vary significantly in intensity, shapes, light sources, sizes, and coverage areas across the image, posing challenges for most existing methods to effectively handle all cases. To address these challenges, this paper introduces a U-shaped Fast Fourier Transform Transformer and Hierarchical Transformer (F2T2-HiT) architecture, an innovative Transformer-based design for SIRR. Our approach uniquely combines Fast Fourier Transform (FFT) Transformer blocks and Hierarchical Transformer blocks within a UNet framework. The FFT Transformer blocks leverage the global frequency domain information to effectively capture and separate reflection patterns, while the Hierarchical Transformer blocks utilize multi-scale feature extraction to handle reflections of varying sizes and complexities. Extensive experiments conducted on three publicly available testing datasets demonstrate state-of-the-art performance, validating the effectiveness of our approach.
>
---
#### [new 105] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching
- **分类: cs.CV; q-bio.GN**

- **简介: 该论文属于生成式建模任务，旨在解决从组织切片图像高效生成空间转录组数据的问题。现有方法忽略细胞间相互作用且内存消耗大。论文提出STFlow，通过全切片流匹配建模联合分布，并引入局部空间注意力机制，在保持低内存开销的同时显著提升性能。**

- **链接: [http://arxiv.org/pdf/2506.05361v1](http://arxiv.org/pdf/2506.05361v1)**

> **作者:** Tinglin Huang; Tianyu Liu; Mehrtash Babadi; Wengong Jin; Rex Ying
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Spatial transcriptomics (ST) has emerged as a powerful technology for bridging histology imaging with gene expression profiling. However, its application has been limited by low throughput and the need for specialized experimental facilities. Prior works sought to predict ST from whole-slide histology images to accelerate this process, but they suffer from two major limitations. First, they do not explicitly model cell-cell interaction as they factorize the joint distribution of whole-slide ST data and predict the gene expression of each spot independently. Second, their encoders struggle with memory constraints due to the large number of spots (often exceeding 10,000) in typical ST datasets. Herein, we propose STFlow, a flow matching generative model that considers cell-cell interaction by modeling the joint distribution of gene expression of an entire slide. It also employs an efficient slide-level encoder with local spatial attention, enabling whole-slide processing without excessive memory overhead. On the recently curated HEST-1k and STImage-1K4M benchmarks, STFlow substantially outperforms state-of-the-art baselines and achieves over 18% relative improvements over the pathology foundation models.
>
---
#### [new 106] Improved Allergy Wheal Detection for the Skin Prick Automated Test Device
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在提升皮肤点刺试验（SPT）中过敏丘疹的自动检测。为解决传统方法依赖单一图像导致准确性不足的问题，作者利用SPAT设备采集的32张多光源图像，设计了一种结合神经网络与可解释算法的自动化检测方法，并验证其比单图方法更准确。**

- **链接: [http://arxiv.org/pdf/2506.05862v1](http://arxiv.org/pdf/2506.05862v1)**

> **作者:** Rembert Daems; Sven Seys; Valérie Hox; Adam Chaker; Glynnis De Greve; Winde Lemmens; Anne-Lise Poirrier; Eline Beckers; Zuzana Diamant; Carmen Dierickx; Peter W. Hellings; Caroline Huart; Claudia Jerin; Mark Jorissen; Hanne Oscé; Karolien Roux; Mark Thompson; Sophie Tombu; Saartje Uyttebroek; Andrzej Zarowski; Senne Gorris; Laura Van Gerven; Dirk Loeckx; Thomas Demeester
>
> **备注:** This work is presented at Artificial Intelligence in Medicine 2025, this is the longer (10 pages) version
>
> **摘要:** Background: The skin prick test (SPT) is the gold standard for diagnosing sensitization to inhalant allergies. The Skin Prick Automated Test (SPAT) device was designed for increased consistency in test results, and captures 32 images to be jointly used for allergy wheal detection and delineation, which leads to a diagnosis. Materials and Methods: Using SPAT data from $868$ patients with suspected inhalant allergies, we designed an automated method to detect and delineate wheals on these images. To this end, $10,416$ wheals were manually annotated by drawing detailed polygons along the edges. The unique data-modality of the SPAT device, with $32$ images taken under distinct lighting conditions, requires a custom-made approach. Our proposed method consists of two parts: a neural network component that segments the wheals on the pixel level, followed by an algorithmic and interpretable approach for detecting and delineating the wheals. Results: We evaluate the performance of our method on a hold-out validation set of $217$ patients. As a baseline we use a single conventionally lighted image per SPT as input to our method. Conclusion: Using the $32$ SPAT images under various lighting conditions offers a considerably higher accuracy than a single image in conventional, uniform light.
>
---
#### [new 107] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos
- **分类: cs.CV**

- **简介: 该论文属于三维动态分割任务，旨在解决动态场景中3D技术无法有效分割运动物体的问题。作者提出“分层运动融合”方法，将2D运动分割结果融合到分层辐射场中，并通过测试时优化减少数据复杂性，显著提升3D分割效果。**

- **链接: [http://arxiv.org/pdf/2506.05546v1](http://arxiv.org/pdf/2506.05546v1)**

> **作者:** Vadim Tschernezki; Diane Larlus; Andrea Vedaldi; Iro Laina
>
> **备注:** Camera-ready for CVPR25
>
> **摘要:** Computer vision is largely based on 2D techniques, with 3D vision still relegated to a relatively narrow subset of applications. However, by building on recent advances in 3D models such as neural radiance fields, some authors have shown that 3D techniques can at last improve outputs extracted from independent 2D views, by fusing them into 3D and denoising them. This is particularly helpful in egocentric videos, where the camera motion is significant, but only under the assumption that the scene itself is static. In fact, as shown in the recent analysis conducted by EPIC Fields, 3D techniques are ineffective when it comes to studying dynamic phenomena, and, in particular, when segmenting moving objects. In this paper, we look into this issue in more detail. First, we propose to improve dynamic segmentation in 3D by fusing motion segmentation predictions from a 2D-based model into layered radiance fields (Layered Motion Fusion). However, the high complexity of long, dynamic videos makes it challenging to capture the underlying geometric structure, and, as a result, hinders the fusion of motion cues into the (incomplete) scene geometry. We address this issue through test-time refinement, which helps the model to focus on specific frames, thereby reducing the data complexity. This results in a synergy between motion fusion and the refinement, and in turn leads to segmentation predictions of the 3D model that surpass the 2D baseline by a large margin. This demonstrates that 3D techniques can enhance 2D analysis even for dynamic phenomena in a challenging and realistic setting.
>
---
#### [new 108] TriPSS: A Tri-Modal Keyframe Extraction Framework Using Perceptual, Structural, and Semantic Representations
- **分类: cs.CV; cs.IR; cs.MM; eess.IV**

- **简介: 该论文属于视频摘要任务，旨在解决有效提取关键帧以实现视频内容理解的问题。论文提出TriPSS框架，融合感知、结构和语义三种模态信息，通过PCA降维与HDBSCAN聚类实现关键帧提取，并结合质量评估与去重优化结果，显著提升了视频摘要效果。**

- **链接: [http://arxiv.org/pdf/2506.05395v1](http://arxiv.org/pdf/2506.05395v1)**

> **作者:** Mert Can Cakmak; Nitin Agarwal; Diwash Poudel
>
> **摘要:** Efficient keyframe extraction is critical for effective video summarization and retrieval, yet capturing the complete richness of video content remains challenging. In this work, we present TriPSS, a novel tri-modal framework that effectively integrates perceptual cues from color features in the CIELAB space, deep structural embeddings derived from ResNet-50, and semantic context from frame-level captions generated by Llama-3.2-11B-Vision-Instruct. By fusing these diverse modalities using principal component analysis, TriPSS constructs robust multi-modal embeddings that enable adaptive segmentation of video content via HDBSCAN clustering. A subsequent refinement stage incorporating quality assessment and duplicate filtering ensures that the final keyframe set is both concise and semantically rich. Comprehensive evaluations on benchmark datasets TVSum20 and SumMe demonstrate that TriPSS achieves state-of-the-art performance, substantially outperforming traditional unimodal and previous multi-modal methods. These results underscore TriPSS's ability to capture nuanced visual and semantic information, thereby setting a new benchmark for video content understanding in large-scale retrieval scenarios.
>
---
#### [new 109] Visual Graph Arena: Evaluating Visual Conceptualization of Vision and Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉与多模态大模型评估任务，旨在解决AI在视觉概念化（即识别和推理不同视觉形式下的同一概念）上的不足。作者构建了Visual Graph Arena（VGA）数据集，包含六种基于图的任务，用于测试模型的视觉抽象能力。实验表明，当前模型在图同构检测等任务上表现差，揭示其缺乏人类般的概念理解。**

- **链接: [http://arxiv.org/pdf/2506.06242v1](http://arxiv.org/pdf/2506.06242v1)**

> **作者:** Zahra Babaiee; Peyman M. Kiasari; Daniela Rus; Radu Grosu
>
> **摘要:** Recent advancements in multimodal large language models have driven breakthroughs in visual question answering. Yet, a critical gap persists, `conceptualization'-the ability to recognize and reason about the same concept despite variations in visual form, a basic ability of human reasoning. To address this challenge, we introduce the Visual Graph Arena (VGA), a dataset featuring six graph-based tasks designed to evaluate and improve AI systems' capacity for visual abstraction. VGA uses diverse graph layouts (e.g., Kamada-Kawai vs. planar) to test reasoning independent of visual form. Experiments with state-of-the-art vision models and multimodal LLMs reveal a striking divide: humans achieved near-perfect accuracy across tasks, while models totally failed on isomorphism detection and showed limited success in path/cycle tasks. We further identify behavioral anomalies suggesting pseudo-intelligent pattern matching rather than genuine understanding. These findings underscore fundamental limitations in current AI models for visual understanding. By isolating the challenge of representation-invariant reasoning, the VGA provides a framework to drive progress toward human-like conceptualization in AI visual models. The Visual Graph Arena is available at: \href{https://vga.csail.mit.edu/}{vga.csail.mit.edu}
>
---
#### [new 110] DVD: A Comprehensive Dataset for Advancing Violence Detection in Real-World Scenarios
- **分类: cs.CV**

- **简介: 该论文属于暴力检测任务，旨在解决现有数据集标注粗糙、规模小、多样性不足的问题。作者构建了DVD数据集，包含500个视频、2.7M帧，提供帧级标注和丰富元数据，以提升模型在真实场景中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.05372v1](http://arxiv.org/pdf/2506.05372v1)**

> **作者:** Dimitrios Kollias; Damith C. Senadeera; Jianian Zheng; Kaushal K. K. Yadav; Greg Slabaugh; Muhammad Awais; Xiaoyun Yang
>
> **摘要:** Violence Detection (VD) has become an increasingly vital area of research. Existing automated VD efforts are hindered by the limited availability of diverse, well-annotated databases. Existing databases suffer from coarse video-level annotations, limited scale and diversity, and lack of metadata, restricting the generalization of models. To address these challenges, we introduce DVD, a large-scale (500 videos, 2.7M frames), frame-level annotated VD database with diverse environments, varying lighting conditions, multiple camera sources, complex social interactions, and rich metadata. DVD is designed to capture the complexities of real-world violent events.
>
---
#### [new 111] LLIA -- Enabling Low-Latency Interactive Avatars: Real-Time Audio-Driven Portrait Video Generation with Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于音频驱动的虚拟人视频生成任务，旨在解决扩散模型在实时交互应用中的高延迟问题。通过提出变量长度生成、一致性训练策略、量化加速及表情控制等方法，实现了低延迟、高保真的实时头像视频生成。**

- **链接: [http://arxiv.org/pdf/2506.05806v1](http://arxiv.org/pdf/2506.05806v1)**

> **作者:** Haojie Yu; Zhaonian Wang; Yihan Pan; Meng Cheng; Hao Yang; Chao Wang; Tao Xie; Xiaoming Xu; Xiaoming Wei; Xunliang Cai
>
> **摘要:** Diffusion-based models have gained wide adoption in the virtual human generation due to their outstanding expressiveness. However, their substantial computational requirements have constrained their deployment in real-time interactive avatar applications, where stringent speed, latency, and duration requirements are paramount. We present a novel audio-driven portrait video generation framework based on the diffusion model to address these challenges. Firstly, we propose robust variable-length video generation to reduce the minimum time required to generate the initial video clip or state transitions, which significantly enhances the user experience. Secondly, we propose a consistency model training strategy for Audio-Image-to-Video to ensure real-time performance, enabling a fast few-step generation. Model quantization and pipeline parallelism are further employed to accelerate the inference speed. To mitigate the stability loss incurred by the diffusion process and model quantization, we introduce a new inference strategy tailored for long-duration video generation. These methods ensure real-time performance and low latency while maintaining high-fidelity output. Thirdly, we incorporate class labels as a conditional input to seamlessly switch between speaking, listening, and idle states. Lastly, we design a novel mechanism for fine-grained facial expression control to exploit our model's inherent capacity. Extensive experiments demonstrate that our approach achieves low-latency, fluid, and authentic two-way communication. On an NVIDIA RTX 4090D, our model achieves a maximum of 78 FPS at a resolution of 384x384 and 45 FPS at a resolution of 512x512, with an initial video generation latency of 140 ms and 215 ms, respectively.
>
---
#### [new 112] LLMs Can Compensate for Deficiencies in Visual Representations
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言模型任务，旨在解决CLIP视觉编码器的表示缺陷问题。通过注意力机制实验，研究发现强大的语言解码器可补偿弱视觉特征，恢复性能。这表明模型中存在动态分工，为未来设计提供新思路。**

- **链接: [http://arxiv.org/pdf/2506.05439v1](http://arxiv.org/pdf/2506.05439v1)**

> **作者:** Sho Takishita; Jay Gala; Abdelrahman Mohamed; Kentaro Inui; Yova Kementchedjhieva
>
> **摘要:** Many vision-language models (VLMs) that prove very effective at a range of multimodal task, build on CLIP-based vision encoders, which are known to have various limitations. We investigate the hypothesis that the strong language backbone in VLMs compensates for possibly weak visual features by contextualizing or enriching them. Using three CLIP-based VLMs, we perform controlled self-attention ablations on a carefully designed probing task. Our findings show that despite known limitations, CLIP visual representations offer ready-to-read semantic information to the language decoder. However, in scenarios of reduced contextualization in the visual representations, the language decoder can largely compensate for the deficiency and recover performance. This suggests a dynamic division of labor in VLMs and motivates future architectures that offload more visual processing to the language decoder.
>
---
#### [new 113] Restereo: Diffusion stereo video generation and restoration
- **分类: cs.CV**

- **简介: 该论文属于立体视频生成与修复任务，旨在解决从低质量单目视频生成高质量立体视频的问题。现有方法依赖高质量输入，而该文提出Restereo方法，通过在退化数据上微调模型并结合掩码条件，实现立体视频生成与视图修复的统一。实验表明其在低分辨率输入下效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.06023v1](http://arxiv.org/pdf/2506.06023v1)**

> **作者:** Xingchang Huang; Ashish Kumar Singh; Florian Dubost; Cristina Nader Vasconcelos; Sakar Khattar; Liang Shi; Christian Theobalt; Cengiz Oztireli; Gurprit Singh
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Stereo video generation has been gaining increasing attention with recent advancements in video diffusion models. However, most existing methods focus on generating 3D stereoscopic videos from monocular 2D videos. These approaches typically assume that the input monocular video is of high quality, making the task primarily about inpainting occluded regions in the warped video while preserving disoccluded areas. In this paper, we introduce a new pipeline that not only generates stereo videos but also enhances both left-view and right-view videos consistently with a single model. Our approach achieves this by fine-tuning the model on degraded data for restoration, as well as conditioning the model on warped masks for consistent stereo generation. As a result, our method can be fine-tuned on a relatively small synthetic stereo video datasets and applied to low-quality real-world videos, performing both stereo video generation and restoration. Experiments demonstrate that our method outperforms existing approaches both qualitatively and quantitatively in stereo video generation from low-resolution inputs.
>
---
#### [new 114] Degradation-Aware Image Enhancement via Vision-Language Classification
- **分类: cs.CV**

- **简介: 该论文属于图像增强任务，旨在解决实际应用中图像退化影响视觉质量和后续处理的问题。论文提出了一种基于视觉语言模型的分类框架，将退化图像分为四类（超分辨率退化、反射伪影、运动模糊、无明显退化），并根据分类结果对图像进行针对性恢复，从而提升图像质量。**

- **链接: [http://arxiv.org/pdf/2506.05450v1](http://arxiv.org/pdf/2506.05450v1)**

> **作者:** Jie Cai; Kangning Yang; Jiaming Ding; Lan Fu; Ling Ouyang; Jiang Li; Jinglin Shen; Zibo Meng
>
> **摘要:** Image degradation is a prevalent issue in various real-world applications, affecting visual quality and downstream processing tasks. In this study, we propose a novel framework that employs a Vision-Language Model (VLM) to automatically classify degraded images into predefined categories. The VLM categorizes an input image into one of four degradation types: (A) super-resolution degradation (including noise, blur, and JPEG compression), (B) reflection artifacts, (C) motion blur, or (D) no visible degradation (high-quality image). Once classified, images assigned to categories A, B, or C undergo targeted restoration using dedicated models tailored for each specific degradation type. The final output is a restored image with improved visual quality. Experimental results demonstrate the effectiveness of our approach in accurately classifying image degradations and enhancing image quality through specialized restoration models. Our method presents a scalable and automated solution for real-world image enhancement tasks, leveraging the capabilities of VLMs in conjunction with state-of-the-art restoration techniques.
>
---
#### [new 115] SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出了SIV-Bench，一个用于评估多模态大语言模型在社交场景理解、社交状态推理和社交动态预测方面能力的视频基准。论文旨在解决人工智能在理解复杂人类社交互动中的挑战，揭示当前模型在关系推理上的瓶颈，并强调对话文本对社交理解的重要性。**

- **链接: [http://arxiv.org/pdf/2506.05425v1](http://arxiv.org/pdf/2506.05425v1)**

> **作者:** Fanqi Kong; Weiqin Zu; Xinyu Chen; Yaodong Yang; Song-Chun Zhu; Xue Feng
>
> **摘要:** The rich and multifaceted nature of human social interaction, encompassing multimodal cues, unobservable relations and mental states, and dynamical behavior, presents a formidable challenge for artificial intelligence. To advance research in this area, we introduce SIV-Bench, a novel video benchmark for rigorously evaluating the capabilities of Multimodal Large Language Models (MLLMs) across Social Scene Understanding (SSU), Social State Reasoning (SSR), and Social Dynamics Prediction (SDP). SIV-Bench features 2,792 video clips and 8,792 meticulously generated question-answer pairs derived from a human-LLM collaborative pipeline. It is originally collected from TikTok and YouTube, covering a wide range of video genres, presentation styles, and linguistic and cultural backgrounds. It also includes a dedicated setup for analyzing the impact of different textual cues-original on-screen text, added dialogue, or no text. Our comprehensive experiments on leading MLLMs reveal that while models adeptly handle SSU, they significantly struggle with SSR and SDP, where Relation Inference (RI) is an acute bottleneck, as further examined in our analysis. Our study also confirms the critical role of transcribed dialogue in aiding comprehension of complex social interactions. By systematically identifying current MLLMs' strengths and limitations, SIV-Bench offers crucial insights to steer the development of more socially intelligent AI. The dataset and code are available at https://kfq20.github.io/sivbench/.
>
---
#### [new 116] Any-Class Presence Likelihood for Robust Multi-Label Classification with Abundant Negative Data
- **分类: cs.LG; cs.AI; cs.CV; 68T05 (Primary) 62H30 (Secondary); I.2.6; I.5.4**

- **简介: 该论文属于多标签分类任务，旨在解决负样本过多影响正样本识别的问题。作者提出一种新的损失函数设计方法，通过计算任意类别存在的似然性，并引入正则化参数调节负类概率的影响，从而提升模型对隐含正样本的识别能力。实验表明该方法在多个大规模数据集上显著提升分类性能。**

- **链接: [http://arxiv.org/pdf/2506.05721v1](http://arxiv.org/pdf/2506.05721v1)**

> **作者:** Dumindu Tissera; Omar Awadallah; Muhammad Umair Danish; Ayan Sadhu; Katarina Grolinger
>
> **摘要:** Multi-label Classification (MLC) assigns an instance to one or more non-exclusive classes. A challenge arises when the dataset contains a large proportion of instances with no assigned class, referred to as negative data, which can overwhelm the learning process and hinder the accurate identification and classification of positive instances. Nevertheless, it is common in MLC applications such as industrial defect detection, agricultural disease identification, and healthcare diagnosis to encounter large amounts of negative data. Assigning a separate negative class to these instances further complicates the learning objective and introduces unnecessary redundancies. To address this challenge, we redesign standard MLC loss functions by deriving a likelihood of any class being present, formulated by a normalized weighted geometric mean of the predicted class probabilities. We introduce a regularization parameter that controls the relative contribution of the absent class probabilities to the any-class presence likelihood in positive instances. The any-class presence likelihood complements the multi-label learning by encouraging the network to become more aware of implicit positive instances and improve the label classification within those positive instances. Experiments on large-scale datasets with negative data: SewerML, modified COCO, and ChestX-ray14, across various networks and base loss functions show that our loss functions consistently improve MLC performance of their standard loss counterparts, achieving gains of up to 6.01 percentage points in F1, 8.06 in F2, and 3.11 in mean average precision, all without additional parameters or computational complexity. Code available at: https://github.com/ML-for-Sensor-Data-Western/gmean-mlc
>
---
#### [new 117] AI-powered Contextual 3D Environment Generation: A Systematic Review
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文属于AI生成3D环境任务，旨在解决传统方法资源消耗大的问题。通过系统综述分析现有生成AI技术在3D场景生成中的应用，探讨其优势、局限及改进方向，重点关注真实性、文本输入影响、风格融合与数据质量，并总结评估指标与行业应用情况。**

- **链接: [http://arxiv.org/pdf/2506.05449v1](http://arxiv.org/pdf/2506.05449v1)**

> **作者:** Miguel Silva; Alexandre Valle de Carvalho
>
> **摘要:** The generation of high-quality 3D environments is crucial for industries such as gaming, virtual reality, and cinema, yet remains resource-intensive due to the reliance on manual processes. This study performs a systematic review of existing generative AI techniques for 3D scene generation, analyzing their characteristics, strengths, limitations, and potential for improvement. By examining state-of-the-art approaches, it presents key challenges such as scene authenticity and the influence of textual inputs. Special attention is given to how AI can blend different stylistic domains while maintaining coherence, the impact of training data on output quality, and the limitations of current models. In addition, this review surveys existing evaluation metrics for assessing realism and explores how industry professionals incorporate AI into their workflows. The findings of this study aim to provide a comprehensive understanding of the current landscape and serve as a foundation for future research on AI-driven 3D content generation. Key findings include that advanced generative architectures enable high-quality 3D content creation at a high computational cost, effective multi-modal integration techniques like cross-attention and latent space alignment facilitate text-to-3D tasks, and the quality and diversity of training data combined with comprehensive evaluation metrics are critical to achieving scalable, robust 3D scene generation.
>
---
#### [new 118] LinGuinE: Longitudinal Guidance Estimation for Volumetric Lung Tumour Segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在解决肺癌肿瘤在纵向CT扫描中的自动化分割问题。提出LinGuinE方法，通过刚性配准和点击有效性分类器，实现从一个时间点的初始标注传播到其他时间点，提高了分割精度，并验证了其在多个数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2506.06092v1](http://arxiv.org/pdf/2506.06092v1)**

> **作者:** Nadine Garibli; Mayank Patwari; Bence Csiba; Yi Wei; Kostas Sidiropoulos
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Segmentation of lung gross tumour volumes is an important first step in radiotherapy and surgical intervention, and is starting to play a role in assessing chemotherapy response. Response to a drug is measured by tracking the tumour volumes over a series of CT scans over a time period i.e. a longitudinal study. However, there currently exist few solutions for automated or semi-automated longitudinal tumour segmentation. This paper introduces LinGuinE, an automated method to segment a longitudinal series of lung tumours. A radiologist must provide an initial input, indicating the location of the tumour in a CT scan at an arbitrary time point. LinGuinE samples points inside this tumour and propagates them to another time point using rigid registration. A click validity classifier selects points which still fall within the tumour; these are used to automatically create a segmentation in the new time point. We test LinGuinE on a dataset acquired from a phase 3 clinical trial for lung tumours and the publicly available 4-D lung CBCT dataset. We find that LinGuinE improves the Dice on both test sets by over 20% (p< 0.05) across 63 longitudinal studies. We show that any time point can be used as a starting point, conduct ablation experiments, and find that our LinGuinE setup yields the best results on both test datasets.
>
---
#### [new 119] Robust Anti-Backdoor Instruction Tuning in LVLMs
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于视觉语言模型安全任务，旨在解决冻结大模型参数下适配器微调时的后门攻击问题。作者提出了一种无需修改核心参数或了解攻击先验的防御框架，通过输入多样性和异常激活正则化，抑制模型对后门触发器的记忆，提升了模型的安全性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.05401v1](http://arxiv.org/pdf/2506.05401v1)**

> **作者:** Yuan Xun; Siyuan Liang; Xiaojun Jia; Xinwei Liu; Xiaochun Cao
>
> **摘要:** Large visual language models (LVLMs) have demonstrated excellent instruction-following capabilities, yet remain vulnerable to stealthy backdoor attacks when finetuned using contaminated data. Existing backdoor defense techniques are usually developed for single-modal visual or language models under fully parameter-adjustable settings or rely on supervisory knowledge during training. However, in real-world scenarios, defenders cannot modify frozen visual encoders or core LLM parameters, nor possess prior knowledge of unknown trigger patterns or target responses. Motivated by the empirical finding that LVLMs readily overfit to fixed, unknown triggers, which can embed malicious associations during adapter-level tuning, we aim to design a defense that operates without access to core weights or attack priors. To this end, we introduce a lightweight, certified-agnostic defense framework, Robust Instruction Tuning, that finetunes only adapter modules and text embedding layers under instruction tuning. Our method integrates two complementary regularizations: (1) Input Diversity Regularization, which perturbs trigger components across training samples to disrupt consistent spurious cues; and (2) Anomalous Activation Regularization, which dynamically sparses adapter weights exhibiting abnormally sharp activations linked to backdoor patterns. These mechanisms jointly guide the model toward learning semantically grounded representations rather than memorizing superficial trigger-response mappings. Extensive experiments against seven attacks on Flickr30k and MSCOCO demonstrate that ours reduces their attack success rate to nearly zero, with an increase in training cost of less than 15%.
>
---
#### [new 120] TRUST: Test-time Resource Utilization for Superior Trustworthiness
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于机器学习模型可信度评估任务，旨在解决标准不确定性估计方法（如dropout）难以区分可靠与不可靠预测的问题。作者提出TRUST方法，在测试时优化资源利用，提升置信度估计可靠性，有效识别分布差异，并区分CNN与ViT分类器表现。**

- **链接: [http://arxiv.org/pdf/2506.06048v1](http://arxiv.org/pdf/2506.06048v1)**

> **作者:** Haripriya Harikumar; Santu Rana
>
> **摘要:** Standard uncertainty estimation techniques, such as dropout, often struggle to clearly distinguish reliable predictions from unreliable ones. We attribute this limitation to noisy classifier weights, which, while not impairing overall class-level predictions, render finer-level statistics less informative. To address this, we propose a novel test-time optimization method that accounts for the impact of such noise to produce more reliable confidence estimates. This score defines a monotonic subset-selection function, where population accuracy consistently increases as samples with lower scores are removed, and it demonstrates superior performance in standard risk-based metrics such as AUSE and AURC. Additionally, our method effectively identifies discrepancies between training and test distributions, reliably differentiates in-distribution from out-of-distribution samples, and elucidates key differences between CNN and ViT classifiers across various vision datasets.
>
---
#### [new 121] FPDANet: A Multi-Section Classification Model for Intelligent Screening of Fetal Ultrasound
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决胎儿超声图像因低对比度、高相似性和噪声导致的分类难题。作者提出了FPDANet模型，融合多尺度上下文信息，并设计位置注意力机制增强特征表达，提升了分类准确率。**

- **链接: [http://arxiv.org/pdf/2506.06054v1](http://arxiv.org/pdf/2506.06054v1)**

> **作者:** Minglang Chen; Jie He; Caixu Xu; Bocheng Liang; Shengli Li; Guannan He; Xiongjie Tao
>
> **摘要:** ResNet has been widely used in image classification tasks due to its ability to model the residual dependence of constant mappings for linear computation. However, the ResNet method adopts a unidirectional transfer of features and lacks an effective method to correlate contextual information, which is not effective in classifying fetal ultrasound images in the classification task, and fetal ultrasound images have problems such as low contrast, high similarity, and high noise. Therefore, we propose a bilateral multi-scale information fusion network-based FPDANet to address the above challenges. Specifically, we design the positional attention mechanism (DAN) module, which utilizes the similarity of features to establish the dependency of different spatial positional features and enhance the feature representation. In addition, we design a bilateral multi-scale (FPAN) information fusion module to capture contextual and global feature dependencies at different feature scales, thereby further improving the model representation. FPDANet classification results obtained 91.05\% and 100\% in Top-1 and Top-5 metrics, respectively, and the experimental results proved the effectiveness and robustness of FPDANet.
>
---
#### [new 122] Learning to Weight Parameters for Data Attribution
- **分类: cs.LG; cs.CV**

- **简介: 论文研究生成模型中的数据归因任务，旨在识别哪些训练样本对输出影响最大。现有方法忽略不同网络层的信息差异，该文提出一种无需标签的学习参数权重方法，实现对训练样本的细粒度归因，提升扩散模型的归因准确性，并揭示输出在主题、风格等方面的来源。**

- **链接: [http://arxiv.org/pdf/2506.05647v1](http://arxiv.org/pdf/2506.05647v1)**

> **作者:** Shuangqi Li; Hieu Le; Jingyi Xu; Mathieu Salzmann
>
> **摘要:** We study data attribution in generative models, aiming to identify which training examples most influence a given output. Existing methods achieve this by tracing gradients back to training data. However, they typically treat all network parameters uniformly, ignoring the fact that different layers encode different types of information and may thus draw information differently from the training set. We propose a method that models this by learning parameter importance weights tailored for attribution, without requiring labeled data. This allows the attribution process to adapt to the structure of the model, capturing which training examples contribute to specific semantic aspects of an output, such as subject, style, or background. Our method improves attribution accuracy across diffusion models and enables fine-grained insights into how outputs borrow from training data.
>
---
#### [new 123] 3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操控任务，旨在解决跨形态操作中的动作统一性和泛化性问题。作者提出了3DFlowAction方法，通过构建一个基于3D光流的世界模型，从人类和机器人操作数据中学习，实现对不同机器人在多样场景下的操作动作规划与适应。**

- **链接: [http://arxiv.org/pdf/2506.06199v1](http://arxiv.org/pdf/2506.06199v1)**

> **作者:** Hongyan Zhi; Peihao Chen; Siyuan Zhou; Yubo Dong; Quanxi Wu; Lei Han; Mingkui Tan
>
> **摘要:** Manipulation has long been a challenging task for robots, while humans can effortlessly perform complex interactions with objects, such as hanging a cup on the mug rack. A key reason is the lack of a large and uniform dataset for teaching robots manipulation skills. Current robot datasets often record robot action in different action spaces within a simple scene. This hinders the robot to learn a unified and robust action representation for different robots within diverse scenes. Observing how humans understand a manipulation task, we find that understanding how the objects should move in the 3D space is a critical clue for guiding actions. This clue is embodiment-agnostic and suitable for both humans and different robots. Motivated by this, we aim to learn a 3D flow world model from both human and robot manipulation data. This model predicts the future movement of the interacting objects in 3D space, guiding action planning for manipulation. Specifically, we synthesize a large-scale 3D optical flow dataset, named ManiFlow-110k, through a moving object auto-detect pipeline. A video diffusion-based world model then learns manipulation physics from these data, generating 3D optical flow trajectories conditioned on language instructions. With the generated 3D object optical flow, we propose a flow-guided rendering mechanism, which renders the predicted final state and leverages GPT-4o to assess whether the predicted flow aligns with the task description. This equips the robot with a closed-loop planning ability. Finally, we consider the predicted 3D optical flow as constraints for an optimization policy to determine a chunk of robot actions for manipulation. Extensive experiments demonstrate strong generalization across diverse robotic manipulation tasks and reliable cross-embodiment adaptation without hardware-specific training.
>
---
#### [new 124] QA-HFL: Quality-Aware Hierarchical Federated Learning for Resource-Constrained Mobile Devices with Heterogeneous Image Quality
- **分类: cs.CR; cs.CV**

- **简介: 该论文提出QA-HFL，一种面向资源受限移动设备的层次化联邦学习框架，解决异构图像质量问题。通过质量感知的模型训练与融合、差分隐私保护和高效通信压缩，在MNIST数据集上实现了高准确率并显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.05411v1](http://arxiv.org/pdf/2506.05411v1)**

> **作者:** Sajid Hussain; Muhammad Sohail; Nauman Ali Khan
>
> **摘要:** This paper introduces QA-HFL, a quality-aware hierarchical federated learning framework that efficiently handles heterogeneous image quality across resource-constrained mobile devices. Our approach trains specialized local models for different image quality levels and aggregates their features using a quality-weighted fusion mechanism, while incorporating differential privacy protection. Experiments on MNIST demonstrate that QA-HFL achieves 92.31% accuracy after just three federation rounds, significantly outperforming state-of-the-art methods like FedRolex (86.42%). Under strict privacy constraints, our approach maintains 30.77% accuracy with formal differential privacy guarantees. Counter-intuitively, low-end devices contributed most significantly (63.5%) to the final model despite using 100 fewer parameters than high-end counterparts. Our quality-aware approach addresses accuracy decline through device-specific regularization, adaptive weighting, intelligent client selection, and server-side knowledge distillation, while maintaining efficient communication with a 4.71% compression ratio. Statistical analysis confirms that our approach significantly outperforms baseline methods (p 0.01) under both standard and privacy-constrained conditions.
>
---
#### [new 125] SurGSplat: Progressive Geometry-Constrained Gaussian Splatting for Surgical Scene Reconstruction
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决手术场景中因特征稀疏和光照不一致导致的重建难题。作者提出了SurGSplat方法，通过引入几何约束逐步优化3D高斯点绘技术，提升血管等关键结构的重建精度，从而为术中导航提供高保真视觉支持。**

- **链接: [http://arxiv.org/pdf/2506.05935v1](http://arxiv.org/pdf/2506.05935v1)**

> **作者:** Yuchao Zheng; Jianing Zhang; Guochen Ning; Hongen Liao
>
> **摘要:** Intraoperative navigation relies heavily on precise 3D reconstruction to ensure accuracy and safety during surgical procedures. However, endoscopic scenarios present unique challenges, including sparse features and inconsistent lighting, which render many existing Structure-from-Motion (SfM)-based methods inadequate and prone to reconstruction failure. To mitigate these constraints, we propose SurGSplat, a novel paradigm designed to progressively refine 3D Gaussian Splatting (3DGS) through the integration of geometric constraints. By enabling the detailed reconstruction of vascular structures and other critical features, SurGSplat provides surgeons with enhanced visual clarity, facilitating precise intraoperative decision-making. Experimental evaluations demonstrate that SurGSplat achieves superior performance in both novel view synthesis (NVS) and pose estimation accuracy, establishing it as a high-fidelity and efficient solution for surgical scene reconstruction. More information and results can be found on the page https://surgsplat.github.io/.
>
---
#### [new 126] MLLM-CL: Continual Learning for Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态大语言模型的持续学习任务，旨在解决模型在动态场景中持续集成新知识与技能时出现的灾难性遗忘问题。作者提出了MLLM-CL基准测试及基于参数隔离和路由机制的方法，有效减少了遗忘，显著提升了持续学习效果。**

- **链接: [http://arxiv.org/pdf/2506.05453v1](http://arxiv.org/pdf/2506.05453v1)**

> **作者:** Hongbo Zhao; Fei Zhu; Rundong Wang; Gaofeng Meng; Zhaoxiang Zhang
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) excel in vision-language understanding but face challenges in adapting to dynamic real-world scenarios that require continuous integration of new knowledge and skills. While continual learning (CL) offers a potential solution, existing benchmarks and methods suffer from critical limitations. In this paper, we introduce MLLM-CL, a novel benchmark encompassing domain and ability continual learning, where the former focuses on independently and identically distributed (IID) evaluation across evolving mainstream domains, whereas the latter evaluates on non-IID scenarios with emerging model ability. Methodologically, we propose preventing catastrophic interference through parameter isolation, along with an MLLM-based routing mechanism. Extensive experiments demonstrate that our approach can integrate domain-specific knowledge and functional abilities with minimal forgetting, significantly outperforming existing methods.
>
---
#### [new 127] Proactive Assistant Dialogue Generation from Streaming Egocentric Videos
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于对话生成任务，旨在解决实时感知任务指导中缺乏有效数据和评估方法的问题。作者提出了一个框架，包括合成对话数据集、自动评估指标及处理流视频的端到端模型，以支持开发能主动辅助用户的AI系统。**

- **链接: [http://arxiv.org/pdf/2506.05904v1](http://arxiv.org/pdf/2506.05904v1)**

> **作者:** Yichi Zhang; Xin Luna Dong; Zhaojiang Lin; Andrea Madotto; Anuj Kumar; Babak Damavandi; Joyce Chai; Seungwhan Moon
>
> **摘要:** Recent advances in conversational AI have been substantial, but developing real-time systems for perceptual task guidance remains challenging. These systems must provide interactive, proactive assistance based on streaming visual inputs, yet their development is constrained by the costly and labor-intensive process of data collection and system evaluation. To address these limitations, we present a comprehensive framework with three key contributions. First, we introduce a novel data curation pipeline that synthesizes dialogues from annotated egocentric videos, resulting in \dataset, a large-scale synthetic dialogue dataset spanning multiple domains. Second, we develop a suite of automatic evaluation metrics, validated through extensive human studies. Third, we propose an end-to-end model that processes streaming video inputs to generate contextually appropriate responses, incorporating novel techniques for handling data imbalance and long-duration videos. This work lays the foundation for developing real-time, proactive AI assistants capable of guiding users through diverse tasks. Project page: https://pro-assist.github.io/
>
---
#### [new 128] Towards an Explainable Comparison and Alignment of Feature Embeddings
- **分类: cs.LG; cs.AI; cs.CV; math.SP**

- **简介: 该论文属于特征嵌入比较与对齐任务，旨在解决不同嵌入在聚类上的差异问题。作者提出SPEC框架，通过核矩阵的谱分解识别样本聚类差异，并设计优化方法实现嵌入对齐，提升可解释性。应用于ImageNet和MS-COCO等大型数据集。**

- **链接: [http://arxiv.org/pdf/2506.06231v1](http://arxiv.org/pdf/2506.06231v1)**

> **作者:** Mohammad Jalali; Bahar Dibaei Nia; Farzan Farnia
>
> **摘要:** While several feature embedding models have been developed in the literature, comparisons of these embeddings have largely focused on their numerical performance in classification-related downstream applications. However, an interpretable comparison of different embeddings requires identifying and analyzing mismatches between sample groups clustered within the embedding spaces. In this work, we propose the \emph{Spectral Pairwise Embedding Comparison (SPEC)} framework to compare embeddings and identify their differences in clustering a reference dataset. Our approach examines the kernel matrices derived from two embeddings and leverages the eigendecomposition of the difference kernel matrix to detect sample clusters that are captured differently by the two embeddings. We present a scalable implementation of this kernel-based approach, with computational complexity that grows linearly with the sample size. Furthermore, we introduce an optimization problem using this framework to align two embeddings, ensuring that clusters identified in one embedding are also captured in the other model. We provide numerical results demonstrating the SPEC's application to compare and align embeddings on large-scale datasets such as ImageNet and MS-COCO. The code is available at [https://github.com/mjalali/embedding-comparison](github.com/mjalali/embedding-comparison).
>
---
#### [new 129] Object Navigation with Structure-Semantic Reasoning-Based Multi-level Map and Multimodal Decision-Making LLM
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于零样本目标导航任务，旨在解决未知环境中语义新颖目标导航性能下降的问题。通过构建环境属性地图（EAM）并结合多模态层级推理模块（MHR），提升场景映射准确率与路径效率，实验证明在MP3D和HM3D数据集上取得显著改进。**

- **链接: [http://arxiv.org/pdf/2506.05896v1](http://arxiv.org/pdf/2506.05896v1)**

> **作者:** Chongshang Yan; Jiaxuan He; Delun Li; Yi Yang; Wenjie Song
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** The zero-shot object navigation (ZSON) in unknown open-ended environments coupled with semantically novel target often suffers from the significant decline in performance due to the neglect of high-dimensional implicit scene information and the long-range target searching task. To address this, we proposed an active object navigation framework with Environmental Attributes Map (EAM) and MLLM Hierarchical Reasoning module (MHR) to improve its success rate and efficiency. EAM is constructed by reasoning observed environments with SBERT and predicting unobserved ones with Diffusion, utilizing human space regularities that underlie object-room correlations and area adjacencies. MHR is inspired by EAM to perform frontier exploration decision-making, avoiding the circuitous trajectories in long-range scenarios to improve path efficiency. Experimental results demonstrate that the EAM module achieves 64.5\% scene mapping accuracy on MP3D dataset, while the navigation task attains SPLs of 28.4\% and 26.3\% on HM3D and MP3D benchmarks respectively - representing absolute improvements of 21.4\% and 46.0\% over baseline methods.
>
---
#### [new 130] Loss Functions for Predictor-based Neural Architecture Search
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于神经架构搜索（NAS）任务，旨在解决如何有效选择和设计性能预测器的损失函数问题。论文系统研究了三类损失函数（回归、排序、加权），评估其在13个任务上的表现，揭示了组合使用损失函数的有效性，并为NAS社区提供了实用指导。**

- **链接: [http://arxiv.org/pdf/2506.05869v1](http://arxiv.org/pdf/2506.05869v1)**

> **作者:** Han Ji; Yuqi Feng; Jiahao Fan; Yanan Sun
>
> **摘要:** Evaluation is a critical but costly procedure in neural architecture search (NAS). Performance predictors have been widely adopted to reduce evaluation costs by directly estimating architecture performance. The effectiveness of predictors is heavily influenced by the choice of loss functions. While traditional predictors employ regression loss functions to evaluate the absolute accuracy of architectures, recent approaches have explored various ranking-based loss functions, such as pairwise and listwise ranking losses, to focus on the ranking of architecture performance. Despite their success in NAS, the effectiveness and characteristics of these loss functions have not been thoroughly investigated. In this paper, we conduct the first comprehensive study on loss functions in performance predictors, categorizing them into three main types: regression, ranking, and weighted loss functions. Specifically, we assess eight loss functions using a range of NAS-relevant metrics on 13 tasks across five search spaces. Our results reveal that specific categories of loss functions can be effectively combined to enhance predictor-based NAS. Furthermore, our findings could provide practical guidance for selecting appropriate loss functions for various tasks. We hope this work provides meaningful insights to guide the development of loss functions for predictor-based methods in the NAS community.
>
---
#### [new 131] DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders for Clinical AI Research
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决现有皮肤疾病数据集地域与人群代表性不足的问题。作者构建了DermaCon-IN，一个包含3,000名印度患者、5,450张临床图像的多标注皮肤病数据集，并基于Rook分类法进行层次化标注。论文还评估了多种模型性能，探索结合解剖与概念级特征的方法，为开发可解释、适用于真实临床环境的皮肤病AI模型提供基础。**

- **链接: [http://arxiv.org/pdf/2506.06099v1](http://arxiv.org/pdf/2506.06099v1)**

> **作者:** Shanawaj S Madarkar; Mahajabeen Madarkar; Madhumitha V; Teli Prakash; Konda Reddy Mopuri; Vinaykumar MV; KVL Sathwika; Adarsh Kasturi; Gandla Dilip Raj; PVN Supranitha; Harsh Udai
>
> **摘要:** Artificial intelligence is poised to augment dermatological care by enabling scalable image-based diagnostics. Yet, the development of robust and equitable models remains hindered by datasets that fail to capture the clinical and demographic complexity of real-world practice. This complexity stems from region-specific disease distributions, wide variation in skin tones, and the underrepresentation of outpatient scenarios from non-Western populations. We introduce DermaCon-IN, a prospectively curated dermatology dataset comprising over 5,450 clinical images from approximately 3,000 patients across outpatient clinics in South India. Each image is annotated by board-certified dermatologists with over 240 distinct diagnoses, structured under a hierarchical, etiology-based taxonomy adapted from Rook's classification. The dataset captures a wide spectrum of dermatologic conditions and tonal variation commonly seen in Indian outpatient care. We benchmark a range of architectures including convolutional models (ResNet, DenseNet, EfficientNet), transformer-based models (ViT, MaxViT, Swin), and Concept Bottleneck Models to establish baseline performance and explore how anatomical and concept-level cues may be integrated. These results are intended to guide future efforts toward interpretable and clinically realistic models. DermaCon-IN provides a scalable and representative foundation for advancing dermatology AI in real-world settings.
>
---
#### [new 132] Deep histological synthesis from mass spectrometry imaging for multimodal registration
- **分类: eess.IV; cs.CV; cs.LG; I.2; I.4**

- **简介: 该论文属于医学图像处理任务，旨在解决组织切片的组学图像与质谱成像（MSI）之间的多模态配准问题。作者提出了一种基于pix2pix模型的方法，从MSI合成组学图像，实现单模态配准，提升了互信息和结构相似性指标。**

- **链接: [http://arxiv.org/pdf/2506.05441v1](http://arxiv.org/pdf/2506.05441v1)**

> **作者:** Kimberley M. Bird; Xujiong Ye; Alan M. Race; James M. Brown
>
> **备注:** Medical Image Understanding and Analysis (MIUA) 2025 Extended Abstract Submission
>
> **摘要:** Registration of histological and mass spectrometry imaging (MSI) allows for more precise identification of structural changes and chemical interactions in tissue. With histology and MSI having entirely different image formation processes and dimensionalities, registration of the two modalities remains an ongoing challenge. This work proposes a solution that synthesises histological images from MSI, using a pix2pix model, to effectively enable unimodal registration. Preliminary results show promising synthetic histology images with limited artifacts, achieving increases in mutual information (MI) and structural similarity index measures (SSIM) of +0.924 and +0.419, respectively, compared to a baseline U-Net model. Our source code is available on GitHub: https://github.com/kimberley/MIUA2025.
>
---
#### [new 133] Gradient Similarity Surgery in Multi-Task Deep Learning
- **分类: cs.LG; cs.CV**

- **简介: 论文属于多任务深度学习（MTDL）领域，旨在解决多任务训练中梯度冲突导致的收敛困难问题。作者提出了一种新的梯度手术方法SAM-GS，通过梯度幅值相似性衡量并调整梯度方向和大小，优化学习过程。实验表明该方法在合成问题和MTL基准任务上均有效。**

- **链接: [http://arxiv.org/pdf/2506.06130v1](http://arxiv.org/pdf/2506.06130v1)**

> **作者:** Thomas Borsani; Andrea Rosani; Giuseppe Nicosia; Giuseppe Di Fatta
>
> **备注:** Paper accepted at ECMLPKDD 2025
>
> **摘要:** The multi-task learning ($MTL$) paradigm aims to simultaneously learn multiple tasks within a single model capturing higher-level, more general hidden patterns that are shared by the tasks. In deep learning, a significant challenge in the backpropagation training process is the design of advanced optimisers to improve the convergence speed and stability of the gradient descent learning rule. In particular, in multi-task deep learning ($MTDL$) the multitude of tasks may generate potentially conflicting gradients that would hinder the concurrent convergence of the diverse loss functions. This challenge arises when the gradients of the task objectives have either different magnitudes or opposite directions, causing one or a few to dominate or to interfere with each other, thus degrading the training process. Gradient surgery methods address the problem explicitly dealing with conflicting gradients by adjusting the overall gradient trajectory. This work introduces a novel gradient surgery method, the Similarity-Aware Momentum Gradient Surgery (SAM-GS), which provides an effective and scalable approach based on a gradient magnitude similarity measure to guide the optimisation process. The SAM-GS surgery adopts gradient equalisation and modulation of the first-order momentum. A series of experimental tests have shown the effectiveness of SAM-GS on synthetic problems and $MTL$ benchmarks. Gradient magnitude similarity plays a crucial role in regularising gradient aggregation in $MTDL$ for the optimisation of the learning process.
>
---
#### [new 134] WoundAIssist: A Patient-Centered Mobile App for AI-Assisted Wound Care With Physicians in the Loop
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于医疗健康与人工智能交叉任务，旨在解决慢性伤口护理中资源紧张、患者不便的问题。作者开发了WoundAIssist移动应用，通过AI辅助伤口图像分析和远程医疗，实现患者居家自我监测，并促进医生参与。论文贡献包括系统实现、可用性评估及远程医疗设计启示。**

- **链接: [http://arxiv.org/pdf/2506.06104v1](http://arxiv.org/pdf/2506.06104v1)**

> **作者:** Vanessa Borst; Anna Riedmann; Tassilo Dege; Konstantin Müller; Astrid Schmieder; Birgit Lugrin; Samuel Kounev
>
> **备注:** Submitted to ACM Health (Special Issue)
>
> **摘要:** The rising prevalence of chronic wounds, especially in aging populations, presents a significant healthcare challenge due to prolonged hospitalizations, elevated costs, and reduced patient quality of life. Traditional wound care is resource-intensive, requiring frequent in-person visits that strain both patients and healthcare professionals (HCPs). Therefore, we present WoundAIssist, a patient-centered, AI-driven mobile application designed to support telemedical wound care. WoundAIssist enables patients to regularly document wounds at home via photographs and questionnaires, while physicians remain actively engaged in the care process through remote monitoring and video consultations. A distinguishing feature is an integrated lightweight deep learning model for on-device wound segmentation, which, combined with patient-reported data, enables continuous monitoring of wound healing progression. Developed through an iterative, user-centered process involving both patients and domain experts, WoundAIssist prioritizes an user-friendly design, particularly for elderly patients. A conclusive usability study with patients and dermatologists reported excellent usability, good app quality, and favorable perceptions of the AI-driven wound recognition. Our main contribution is two-fold: (I) the implementation and (II) evaluation of WoundAIssist, an easy-to-use yet comprehensive telehealth solution designed to bridge the gap between patients and HCPs. Additionally, we synthesize design insights for remote patient monitoring apps, derived from over three years of interdisciplinary research, that may inform the development of similar digital health tools across clinical domains.
>
---
#### [new 135] ODE-GS: Latent ODEs for Dynamic Scene Extrapolation with 3D Gaussian Splatting
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 本文提出ODE-GS，结合3D高斯点绘与潜神经微分方程，解决动态3D场景外推问题。现有方法依赖时间嵌入，难以预测未来。ODE-GS先训练高保真变形模型，再用Transformer编码历史轨迹为潜状态，并由神经ODE建模其连续演化，实现未来任意时刻的实时渲染。该方法在D-NeRF和NVFI数据集上表现优异，显著提升PSNR和LPIPS指标。**

- **链接: [http://arxiv.org/pdf/2506.05480v1](http://arxiv.org/pdf/2506.05480v1)**

> **作者:** Daniel Wang; Patrick Rim; Tian Tian; Alex Wong; Ganesh Sundaramoorthi
>
> **摘要:** We present ODE-GS, a novel method that unifies 3D Gaussian Splatting with latent neural ordinary differential equations (ODEs) to forecast dynamic 3D scenes far beyond the time span seen during training. Existing neural rendering systems - whether NeRF- or 3DGS-based - embed time directly in a deformation network and therefore excel at interpolation but collapse when asked to predict the future, where timestamps are strictly out-of-distribution. ODE-GS eliminates this dependency: after learning a high-fidelity, time-conditioned deformation model for the training window, we freeze it and train a Transformer encoder that summarizes past Gaussian trajectories into a latent state whose continuous evolution is governed by a neural ODE. Numerical integration of this latent flow yields smooth, physically plausible Gaussian trajectories that can be queried at any future instant and rendered in real time. Coupled with a variational objective and a lightweight second-derivative regularizer, ODE-GS attains state-of-the-art extrapolation on D-NeRF and NVFI benchmarks, improving PSNR by up to 10 dB and halving perceptual error (LPIPS) relative to the strongest baselines. Our results demonstrate that continuous-time latent dynamics are a powerful, practical route to photorealistic prediction of complex 3D scenes.
>
---
#### [new 136] Enhancing Neural Autoregressive Distribution Estimators for Image Reconstruction
- **分类: eess.IV; cs.CV; cs.LG; stat.AP**

- **简介: 该论文属于图像重建任务，旨在通过观测少量像素预测完整图像。作者改进了ConvNADE模型，使其适用于彩色和实值图像，并引入低差异像素采样策略，提升重建效果与效率。**

- **链接: [http://arxiv.org/pdf/2506.05391v1](http://arxiv.org/pdf/2506.05391v1)**

> **作者:** Ambrose Emmett-Iwaniw; Nathan Kirk
>
> **备注:** Accepted for publication in conference proceedings, MCQMC 2024
>
> **摘要:** Autoregressive models are often employed to learn distributions of image data by decomposing the $D$-dimensional density function into a product of one-dimensional conditional distributions. Each conditional depends on preceding variables (pixels, in the case of image data), making the order in which variables are processed fundamental to the model performance. In this paper, we study the problem of observing a small subset of image pixels (referred to as a pixel patch) to predict the unobserved parts of the image. As our prediction mechanism, we propose a generalized and computationally efficient version of the convolutional neural autoregressive distribution estimator (ConvNADE) model adapted for real-valued and color images. Moreover, we investigate the quality of image reconstruction when observing both random pixel patches and low-discrepancy pixel patches inspired by quasi-Monte Carlo theory. Experiments on benchmark datasets demonstrate that choosing the pixels akin to a low-discrepancy sequence reduces test loss and produces more realistic reconstructed images.
>
---
#### [new 137] Integer Binary-Range Alignment Neuron for Spiking Neural Networks
- **分类: cs.NE; cs.CV**

- **简介: 该论文属于神经网络领域，旨在提升脉冲神经网络（SNN）的性能。通过设计新型整数二值范围对齐神经元，解决SNN表达能力不足的问题，在图像分类和目标检测任务中取得了更高准确率，同时保持了能效优势。**

- **链接: [http://arxiv.org/pdf/2506.05679v1](http://arxiv.org/pdf/2506.05679v1)**

> **作者:** Binghao Ye; Wenjuan Li; Dong Wang; Man Yao; Bing Li; Weiming Hu; Dong Liang; Kun Shang
>
> **备注:** 11 pages
>
> **摘要:** Spiking Neural Networks (SNNs) are noted for their brain-like computation and energy efficiency, but their performance lags behind Artificial Neural Networks (ANNs) in tasks like image classification and object detection due to the limited representational capacity. To address this, we propose a novel spiking neuron, Integer Binary-Range Alignment Leaky Integrate-and-Fire to exponentially expand the information expression capacity of spiking neurons with only a slight energy increase. This is achieved through Integer Binary Leaky Integrate-and-Fire and range alignment strategy. The Integer Binary Leaky Integrate-and-Fire allows integer value activation during training and maintains spike-driven dynamics with binary conversion expands virtual timesteps during inference. The range alignment strategy is designed to solve the spike activation limitation problem where neurons fail to activate high integer values. Experiments show our method outperforms previous SNNs, achieving 74.19% accuracy on ImageNet and 66.2% mAP@50 and 49.1% mAP@50:95 on COCO, surpassing previous bests with the same architecture by +3.45% and +1.6% and +1.8%, respectively. Notably, our SNNs match or exceed ANNs' performance with the same architecture, and the energy efficiency is improved by 6.3${\times}$.
>
---
#### [new 138] Noninvasive precision modulation of high-level neural population activity via natural vision perturbations
- **分类: q-bio.NC; cs.CV; cs.NE**

- **简介: 该论文属于神经科学任务，旨在解决如何无创精准调控大脑深层神经活动的问题。通过在自然视觉输入中引入扰动，研究实现了对猕猴腹侧视觉流中高阶神经元群的精确调制，并验证了模型预测与实际生物效应的一致性，展示了可设计、非侵入、视觉传递的神经干预方法。**

- **链接: [http://arxiv.org/pdf/2506.05633v1](http://arxiv.org/pdf/2506.05633v1)**

> **作者:** Guy Gaziv; Sarah Goulding; Ani Ayvazian-Hancock; Yoon Bai; James J. DiCarlo
>
> **摘要:** Precise control of neural activity -- modulating target neurons deep in the brain while leaving nearby neurons unaffected -- is an outstanding challenge in neuroscience, generally achieved through invasive techniques. This study investigates the possibility of precisely and noninvasively modulating neural activity in the high-level primate ventral visual stream via perturbations on one's natural visual feed. When tested on macaque inferior temporal (IT) neural populations, we found quantitative agreement between the model-predicted and biologically realized effect: strong modulation concentrated on targeted neural sites. We extended this to demonstrate accurate injection of experimenter-chosen neural population patterns via subtle perturbations applied on the background of typical natural visual feeds. These results highlight that current machine-executable models of the ventral stream can now design noninvasive, visually-delivered, possibly imperceptible neural interventions at the resolution of individual neurons.
>
---
#### [new 139] Peer-Ranked Precision: Creating a Foundational Dataset for Fine-Tuning Vision Models from DataSeeds' Annotated Imagery
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在推动从“模型中心”向“数据中心”范式的转变。它通过引入高质量、人工标注的DataSeeds.AI样本数据集（DSD），提升图像生成与识别模型性能，并提供可扩展的基础用于商业AI开发。**

- **链接: [http://arxiv.org/pdf/2506.05673v1](http://arxiv.org/pdf/2506.05673v1)**

> **作者:** Sajjad Abdoli; Freeman Lewin; Gediminas Vasiliauskas; Fabian Schonholz
>
> **备注:** 28 pages, 12 figures
>
> **摘要:** The development of modern Artificial Intelligence (AI) models, particularly diffusion-based models employed in computer vision and image generation tasks, is undergoing a paradigmatic shift in development methodologies. Traditionally dominated by a "Model Centric" approach, in which performance gains were primarily pursued through increasingly complex model architectures and hyperparameter optimization, the field is now recognizing a more nuanced "Data-Centric" approach. This emergent framework foregrounds the quality, structure, and relevance of training data as the principal driver of model performance. To operationalize this paradigm shift, we introduce the DataSeeds.AI sample dataset (the "DSD"), initially comprised of approximately 10,610 high-quality human peer-ranked photography images accompanied by extensive multi-tier annotations. The DSD is a foundational computer vision dataset designed to usher in a new standard for commercial image datasets. Representing a small fraction of DataSeed.AI's 100 million-plus image catalog, the DSD provides a scalable foundation necessary for robust commercial and multimodal AI development. Through this in-depth exploratory analysis, we document the quantitative improvements generated by the DSD on specific models against known benchmarks and make the code and the trained models used in our evaluation publicly available.
>
---
#### [new 140] PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出了PuzzleWorld，一个包含667个谜题的多模态开放推理基准任务，旨在评估模型在无明确问题定义下的逐步推理、创造性思维和多模态理解能力。论文分析了当前模型在此类任务上的表现瓶颈，并通过精细标注与微调实验展示了提升推理能力的潜力。**

- **链接: [http://arxiv.org/pdf/2506.06211v1](http://arxiv.org/pdf/2506.06211v1)**

> **作者:** Hengzhi Li; Brendon Jiang; Alexander Naehu; Regan Song; Justin Zhang; Megan Tjandrasuwita; Chanakya Ekbote; Steven-Shine Chen; Adithya Balachandran; Wei Dai; Rebecca Chang; Paul Pu Liang
>
> **摘要:** Puzzlehunts are a genre of complex, multi-step puzzles lacking well-defined problem definitions. In contrast to conventional reasoning benchmarks consisting of tasks with clear instructions, puzzlehunts require models to discover the underlying problem structure from multimodal evidence and iterative reasoning, mirroring real-world domains such as scientific discovery, exploratory data analysis, or investigative problem-solving. Despite recent progress in foundation models, their performance on such open-ended settings remains largely untested. In this paper, we introduce PuzzleWorld, a large-scale benchmark of 667 puzzlehunt-style problems designed to assess step-by-step, open-ended, and creative multimodal reasoning. Each puzzle is annotated with the final solution, detailed reasoning traces, and cognitive skill labels, enabling holistic benchmarking and fine-grained diagnostic analysis. Most state-of-the-art models achieve only 1-2% final answer accuracy, with the best model solving only 14% of puzzles and reaching 40% stepwise accuracy. To demonstrate the value of our reasoning annotations, we show that fine-tuning a small model on reasoning traces improves stepwise reasoning from 4% to 11%, while training on final answers alone degrades performance to near zero. Our error analysis reveals that current models exhibit myopic reasoning, are bottlenecked by the limitations of language-based inference, and lack sketching capabilities crucial for visual and spatial reasoning. We release PuzzleWorld at https://github.com/MIT-MI/PuzzleWorld to support future work on building more general, open-ended, and creative reasoning systems.
>
---
#### [new 141] QualitEye: Public and Privacy-preserving Gaze Data Quality Verification
- **分类: cs.HC; cs.CR; cs.CV**

- **简介: 该论文属于图像质量验证任务，旨在解决大规模注视数据收集中的隐私与质量问题。作者提出了QualitEye方法，通过语义表示和隐私协议实现公开及隐私保护场景下的注视数据质量验证。**

- **链接: [http://arxiv.org/pdf/2506.05908v1](http://arxiv.org/pdf/2506.05908v1)**

> **作者:** Mayar Elfares; Pascal Reisert; Ralf Küsters; Andreas Bulling
>
> **摘要:** Gaze-based applications are increasingly advancing with the availability of large datasets but ensuring data quality presents a substantial challenge when collecting data at scale. It further requires different parties to collaborate, therefore, privacy concerns arise. We propose QualitEye--the first method for verifying image-based gaze data quality. QualitEye employs a new semantic representation of eye images that contains the information required for verification while excluding irrelevant information for better domain adaptation. QualitEye covers a public setting where parties can freely exchange data and a privacy-preserving setting where parties cannot reveal their raw data nor derive gaze features/labels of others with adapted private set intersection protocols. We evaluate QualitEye on the MPIIFaceGaze and GazeCapture datasets and achieve a high verification performance (with a small overhead in runtime for privacy-preserving versions). Hence, QualitEye paves the way for new gaze analysis methods at the intersection of machine learning, human-computer interaction, and cryptography.
>
---
## 更新

#### [replaced 001] Flexiffusion: Segment-wise Neural Architecture Search for Flexible Denoising Schedule
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.17566v2](http://arxiv.org/pdf/2409.17566v2)**

> **作者:** Hongtao Huang; Xiaojun Chang; Lina Yao
>
> **摘要:** Diffusion models are cutting-edge generative models adept at producing diverse, high-quality images. Despite their effectiveness, these models often require significant computational resources owing to their numerous sequential denoising steps and the significant inference cost of each step. Recently, Neural Architecture Search (NAS) techniques have been employed to automatically search for faster generation processes. However, NAS for diffusion is inherently time-consuming as it requires estimating thousands of diffusion models to search for the optimal one. In this paper, we introduce Flexiffusion, a novel training-free NAS paradigm designed to accelerate diffusion models by concurrently optimizing generation steps and network structures. Specifically, we partition the generation process into isometric step segments, each sequentially composed of a full step, multiple partial steps, and several null steps. The full step computes all network blocks, while the partial step involves part of the blocks, and the null step entails no computation. Flexiffusion autonomously explores flexible step combinations for each segment, substantially reducing search costs and enabling greater acceleration compared to the state-of-the-art (SOTA) method for diffusion models. Our searched models reported speedup factors of $2.6\times$ and $1.5\times$ for the original LDM-4-G and the SOTA, respectively. The factors for Stable Diffusion V1.5 and the SOTA are $5.1\times$ and $2.0\times$. We also verified the performance of Flexiffusion on multiple datasets, and positive experiment results indicate that Flexiffusion can effectively reduce redundancy in diffusion models.
>
---
#### [replaced 002] Leopard: A Vision Language Model For Text-Rich Multi-Image Tasks
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.01744v3](http://arxiv.org/pdf/2410.01744v3)**

> **作者:** Mengzhao Jia; Wenhao Yu; Kaixin Ma; Tianqing Fang; Zhihan Zhang; Siru Ouyang; Hongming Zhang; Dong Yu; Meng Jiang
>
> **备注:** Our code is available at https://github.com/tencent-ailab/Leopard
>
> **摘要:** Text-rich images, where text serves as the central visual element guiding the overall understanding, are prevalent in real-world applications, such as presentation slides, scanned documents, and webpage snapshots. Tasks involving multiple text-rich images are especially challenging, as they require not only understanding the content of individual images but reasoning about inter-relationships and logical flows across multiple visual inputs. Despite the importance of these scenarios, current multimodal large language models (MLLMs) struggle to handle such tasks due to two key challenges: (1) the scarcity of high-quality instruction tuning datasets for text-rich multi-image scenarios, and (2) the difficulty in balancing image resolution with visual feature sequence length. To address these challenges, we propose Leopard, an MLLM tailored for handling vision-language tasks involving multiple text-rich images. First, we curated about one million high-quality multimodal instruction-tuning data, tailored to text-rich, multi-image scenarios. Second, we proposed an adaptive high-resolution multi-image encoding module to dynamically optimize the allocation of visual sequence length based on the original aspect ratios and resolutions of images. Experiments on a diverse set of benchmarks reveal that our model consistently outperforms state-of-the-art systems, such as Llama-3.2 and Qwen2-VL, in challenging text-rich, multi-image evaluations. Remarkably, our approach achieves outstanding performance using only 1.2M training instances, all of which are fully open-sourced, demonstrating both high efficiency and effectiveness compared to models trained on large-scale in-house data. Our code and data are available at https://github.com/tencent-ailab/Leopard.
>
---
#### [replaced 003] Astraea: A GPU-Oriented Token-wise Acceleration Framework for Video Diffusion Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05096v2](http://arxiv.org/pdf/2506.05096v2)**

> **作者:** Haosong Liu; Yuge Cheng; Zihan Liu; Aiyue Chen; Yiwu Yao; Chen Chen; Jingwen Leng; Yu Feng; Minyi Guo
>
> **摘要:** Video diffusion transformers (vDiTs) have made impressive progress in text-to-video generation, but their high computational demands present major challenges for practical deployment. While existing acceleration methods reduce workload at various granularities, they often rely on heuristics, limiting their applicability. We introduce ASTRAEA, an automatic framework that searches for near-optimal configurations for vDiT-based video generation. At its core, ASTRAEA proposes a lightweight token selection mechanism and a memory-efficient, GPU-parallel sparse attention strategy, enabling linear reductions in execution time with minimal impact on generation quality. To determine optimal token reduction for different timesteps, we further design a search framework that leverages a classic evolutionary algorithm to automatically determine the distribution of the token budget effectively. Together, ASTRAEA achieves up to 2.4x inference speedup on a single GPU with great scalability (up to 13.2x speedup on 8 GPUs) while retaining better video quality compared to the state-of-the-art methods (<0.5% loss on the VBench score compared to the baseline vDiT models).
>
---
#### [replaced 004] VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.17253v4](http://arxiv.org/pdf/2408.17253v4)**

> **作者:** Mouxiang Chen; Lefei Shen; Zhuo Li; Xiaoyun Joy Wang; Jianling Sun; Chenghao Liu
>
> **备注:** v4: accepted by ICML 2025
>
> **摘要:** Foundation models have emerged as a promising approach in time series forecasting (TSF). Existing approaches either repurpose large language models (LLMs) or build large-scale time series datasets to develop TSF foundation models for universal forecasting. However, these methods face challenges due to the severe cross-domain gap or in-domain heterogeneity. This paper explores a new road to building a TSF foundation model from rich, high-quality natural images. Our key insight is that a visual masked autoencoder, pre-trained on the ImageNet dataset, can naturally be a numeric series forecaster. By reformulating TSF as an image reconstruction task, we bridge the gap between image pre-training and TSF downstream tasks. Surprisingly, without further adaptation in the time series domain, the proposed VisionTS could achieve better zero-shot forecast performance than existing TSF foundation models. With fine-tuning for one epoch, VisionTS could further improve the forecasting and achieve state-of-the-art performance in most cases. Extensive experiments reveal intrinsic similarities between images and real-world time series, suggesting that visual models may offer a "free lunch" for TSF and highlight the potential for future cross-modality research. Our code is publicly available at https://github.com/Keytoyze/VisionTS.
>
---
#### [replaced 005] MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16671v2](http://arxiv.org/pdf/2502.16671v2)**

> **作者:** Hengzhi Li; Megan Tjandrasuwita; Yi R. Fung; Armando Solar-Lezama; Paul Pu Liang
>
> **摘要:** As AI becomes more closely integrated with peoples' daily activities, socially intelligent AI that can understand and interact seamlessly with humans in daily lives is increasingly important. However, current works in AI social reasoning all rely on language-only or language-dominant approaches to benchmark and training models, resulting in systems that are improving in verbal communication but struggle with nonverbal social understanding. To address this limitation, we tap into a novel data source rich in nonverbal social interactions -- mime videos. Mimes refer to the art of expression through gesture and movement without spoken words, which presents unique challenges and opportunities in interpreting nonverbal social communication. We contribute a new dataset called MimeQA, obtained by sourcing 8 hours of videos clips from YouTube and developing a comprehensive video question-answering benchmark comprising 806 carefully annotated and verified question-answer pairs, designed to probe nonverbal social reasoning capabilities. Using MimeQA, we evaluate state-of-the-art video large language models (vLLMs) and find that they achieve low overall accuracy, ranging from 20-30%, while humans score 86%. Our analysis reveals that vLLMs often fail to ground imagined objects and over-rely on the text prompt while ignoring subtle nonverbal interactions. We hope to inspire future work in AI models that embody true social intelligence capable of interpreting non-verbal human interactions.
>
---
#### [replaced 006] FreeTimeGS: Free Gaussian Primitives at Anytime and Anywhere for Dynamic Scene Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05348v2](http://arxiv.org/pdf/2506.05348v2)**

> **作者:** Yifan Wang; Peishan Yang; Zhen Xu; Jiaming Sun; Zhanhua Zhang; Yong Chen; Hujun Bao; Sida Peng; Xiaowei Zhou
>
> **备注:** CVPR 2025; Project page: https://zju3dv.github.io/freetimegs/
>
> **摘要:** This paper addresses the challenge of reconstructing dynamic 3D scenes with complex motions. Some recent works define 3D Gaussian primitives in the canonical space and use deformation fields to map canonical primitives to observation spaces, achieving real-time dynamic view synthesis. However, these methods often struggle to handle scenes with complex motions due to the difficulty of optimizing deformation fields. To overcome this problem, we propose FreeTimeGS, a novel 4D representation that allows Gaussian primitives to appear at arbitrary time and locations. In contrast to canonical Gaussian primitives, our representation possesses the strong flexibility, thus improving the ability to model dynamic 3D scenes. In addition, we endow each Gaussian primitive with an motion function, allowing it to move to neighboring regions over time, which reduces the temporal redundancy. Experiments results on several datasets show that the rendering quality of our method outperforms recent methods by a large margin. Project page: https://zju3dv.github.io/freetimegs/ .
>
---
#### [replaced 007] TT-Occ: Test-Time Compute for Self-Supervised Occupancy via Spatio-Temporal Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08485v2](http://arxiv.org/pdf/2503.08485v2)**

> **作者:** Fengyi Zhang; Huitong Yang; Zheng Zhang; Zi Huang; Yadan Luo
>
> **摘要:** Self-supervised 3D occupancy prediction offers a promising solution for understanding complex driving scenes without requiring costly 3D annotations. However, training dense occupancy decoders to capture fine-grained geometry and semantics can demand hundreds of GPU hours, and once trained, such models struggle to adapt to varying voxel resolutions or novel object categories without extensive retraining. To overcome these limitations, we propose a practical and flexible test-time occupancy prediction framework termed TT-Occ. Our method incrementally constructs, optimizes and voxelizes time-aware 3D Gaussians from raw sensor streams by integrating vision foundation models (VLMs) at runtime. The flexible nature of 3D Gaussians allows voxelization at arbitrary user-specified resolutions, while the generalization ability of VLMs enables accurate perception and open-vocabulary recognition, without any network training or fine-tuning. Specifically, TT-Occ operates in a lift-track-voxelize symphony: We first lift the geometry and semantics of surrounding-view extracted from VLMs to instantiate Gaussians at 3D space; Next, we track dynamic Gaussians while accumulating static ones to complete the scene and enforce temporal consistency; Finally, we voxelize the optimized Gaussians to generate occupancy prediction. Optionally, inherent noise in VLM predictions and tracking is mitigated by periodically smoothing neighboring Gaussians during optimization. To validate the generality and effectiveness of our framework, we offer two variants: one LiDAR-based and one vision-centric, and conduct extensive experiments on Occ3D and nuCraft benchmarks with varying voxel resolutions. Code will be available at https://github.com/Xian-Bei/TT-Occ.
>
---
#### [replaced 008] Assessing Intersectional Bias in Representations of Pre-Trained Image Recognition Models
- **分类: cs.CV; cs.CY; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.03664v2](http://arxiv.org/pdf/2506.03664v2)**

> **作者:** Valerie Krug; Sebastian Stober
>
> **备注:** Summary paper accepted at the 3rd TRR 318 Conference: Contextualizing Explanations 2025
>
> **摘要:** Deep Learning models have achieved remarkable success. Training them is often accelerated by building on top of pre-trained models which poses the risk of perpetuating encoded biases. Here, we investigate biases in the representations of commonly used ImageNet classifiers for facial images while considering intersections of sensitive variables age, race and gender. To assess the biases, we use linear classifier probes and visualize activations as topographic maps. We find that representations in ImageNet classifiers particularly allow differentiation between ages. Less strongly pronounced, the models appear to associate certain ethnicities and distinguish genders in middle-aged groups.
>
---
#### [replaced 009] Does Your 3D Encoder Really Work? When Pretrain-SFT from 2D VLMs Meets 3D VLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05318v2](http://arxiv.org/pdf/2506.05318v2)**

> **作者:** Haoyuan Li; Yanpeng Zhou; Yufei Gao; Tao Tang; Jianhua Han; Yujie Yuan; Dave Zhenyu Chen; Jiawang Bian; Hang Xu; Xiaodan Liang
>
> **摘要:** Remarkable progress in 2D Vision-Language Models (VLMs) has spurred interest in extending them to 3D settings for tasks like 3D Question Answering, Dense Captioning, and Visual Grounding. Unlike 2D VLMs that typically process images through an image encoder, 3D scenes, with their intricate spatial structures, allow for diverse model architectures. Based on their encoder design, this paper categorizes recent 3D VLMs into 3D object-centric, 2D image-based, and 3D scene-centric approaches. Despite the architectural similarity of 3D scene-centric VLMs to their 2D counterparts, they have exhibited comparatively lower performance compared with the latest 3D object-centric and 2D image-based approaches. To understand this gap, we conduct an in-depth analysis, revealing that 3D scene-centric VLMs show limited reliance on the 3D scene encoder, and the pre-train stage appears less effective than in 2D VLMs. Furthermore, we observe that data scaling benefits are less pronounced on larger datasets. Our investigation suggests that while these models possess cross-modal alignment capabilities, they tend to over-rely on linguistic cues and overfit to frequent answer distributions, thereby diminishing the effective utilization of the 3D encoder. To address these limitations and encourage genuine 3D scene understanding, we introduce a novel 3D Relevance Discrimination QA dataset designed to disrupt shortcut learning and improve 3D understanding. Our findings highlight the need for advanced evaluation and improved strategies for better 3D understanding in 3D VLMs.
>
---
#### [replaced 010] Improving Generalization in MRI-Based Deep Learning Models for Total Knee Replacement Prediction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.19203v3](http://arxiv.org/pdf/2504.19203v3)**

> **作者:** Ehsan Karami; Hamid Soltanian-Zadeh
>
> **摘要:** Knee osteoarthritis (KOA) is a common joint disease that causes pain and mobility issues. While MRI-based deep learning models have demonstrated superior performance in predicting total knee replacement (TKR) and disease progression, their generalizability remains challenging, particularly when applied to imaging data from different sources. In this study, we have shown that replacing batch normalization with instance normalization, using data augmentation, and applying contrastive loss improves model generalization in a baseline deep learning model for knee osteoarthritis (KOA) prediction. We trained and evaluated our model using MRI data from the Osteoarthritis Initiative (OAI) database, considering sagittal fat-suppressed intermediate-weighted turbo spin-echo (FS-IW-TSE) images as the source domain and sagittal fat-suppressed three-dimensional (3D) dual-echo in steady state (DESS) images as the target domain. The results demonstrate a statistically significant improvement in classification accuracy across both domains, with our approach outperforming the baseline model.
>
---
#### [replaced 011] Federated Foundation Model for GI Endoscopy Images
- **分类: cs.CV; cs.LG; I.2.10; I.4; I.5**

- **链接: [http://arxiv.org/pdf/2505.24108v2](http://arxiv.org/pdf/2505.24108v2)**

> **作者:** Alina Devkota; Annahita Amireskandari; Joel Palko; Shyam Thakkar; Donald Adjeroh; Xiajun Jiang; Binod Bhattarai; Prashnna K. Gyawali
>
> **备注:** 11 pages, 11 figures, submitted to BHI2025
>
> **摘要:** Gastrointestinal (GI) endoscopy is essential in identifying GI tract abnormalities in order to detect diseases in their early stages and improve patient outcomes. Although deep learning has shown success in supporting GI diagnostics and decision-making, these models require curated datasets with labels that are expensive to acquire. Foundation models offer a promising solution by learning general-purpose representations, which can be finetuned for specific tasks, overcoming data scarcity. Developing foundation models for medical imaging holds significant potential, but the sensitive and protected nature of medical data presents unique challenges. Foundation model training typically requires extensive datasets, and while hospitals generate large volumes of data, privacy restrictions prevent direct data sharing, making foundation model training infeasible in most scenarios. In this work, we propose a FL framework for training foundation models for gastroendoscopy imaging, enabling data to remain within local hospital environments while contributing to a shared model. We explore several established FL algorithms, assessing their suitability for training foundation models without relying on task-specific labels, conducting experiments in both homogeneous and heterogeneous settings. We evaluate the trained foundation model on three critical downstream tasks--classification, detection, and segmentation--and demonstrate that it achieves improved performance across all tasks, highlighting the effectiveness of our approach in a federated, privacy-preserving setting.
>
---
#### [replaced 012] ARMOR: Empowering Multimodal Understanding Model with Interleaved Multimodal Generation Capability
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06542v2](http://arxiv.org/pdf/2503.06542v2)**

> **作者:** Jianwen Sun; Yukang Feng; Chuanhao Li; Fanrui Zhang; Zizhen Li; Jiaxin Ai; Sizhuo Zhou; Yu Dai; Shenglin Zhang; Kaipeng Zhang
>
> **摘要:** Unified multimodal understanding and generation have recently received much attention in the area of vision and language. Existing UniMs are designed to simultaneously learn both multimodal understanding and generation capabilities, demanding substantial computational resources, and often struggle to generate interleaved text-image. We present ARMOR, a resource-efficient and pure autoregressive framework that achieves both understanding and generation by fine-tuning existing multimodal large language models (MLLMs). Specifically, ARMOR extends existing MLLMs from three perspectives: (1) For model architecture, an asymmetric encoder-decoder architecture with a forward-switching mechanism is introduced to unify embedding space integrating textual and visual modalities for enabling natural text-image interleaved generation with minimal computational overhead. (2) For training data, a meticulously curated, high-quality interleaved dataset is collected for fine-tuning MLLMs. (3) For the training algorithm, we propose a ``what or how to generate'' algorithm to empower existing MLLMs with multimodal generation capabilities while preserving their multimodal understanding capabilities, through three progressive training stages based on the collected dataset. Experimental results demonstrate that ARMOR upgrades existing MLLMs to UniMs with promising image generation capabilities, using limited training resources. Our code will be released soon at https://github.com/finyorko/armor.
>
---
#### [replaced 013] Ophora: A Large-Scale Data-Driven Text-Guided Ophthalmic Surgical Video Generation Model
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07449v4](http://arxiv.org/pdf/2505.07449v4)**

> **作者:** Wei Li; Ming Hu; Guoan Wang; Lihao Liu; Kaijin Zhou; Junzhi Ning; Xin Guo; Zongyuan Ge; Lixu Gu; Junjun He
>
> **备注:** Early accepted in MICCAI25
>
> **摘要:** In ophthalmic surgery, developing an AI system capable of interpreting surgical videos and predicting subsequent operations requires numerous ophthalmic surgical videos with high-quality annotations, which are difficult to collect due to privacy concerns and labor consumption. Text-guided video generation (T2V) emerges as a promising solution to overcome this issue by generating ophthalmic surgical videos based on surgeon instructions. In this paper, we present Ophora, a pioneering model that can generate ophthalmic surgical videos following natural language instructions. To construct Ophora, we first propose a Comprehensive Data Curation pipeline to convert narrative ophthalmic surgical videos into a large-scale, high-quality dataset comprising over 160K video-instruction pairs, Ophora-160K. Then, we propose a Progressive Video-Instruction Tuning scheme to transfer rich spatial-temporal knowledge from a T2V model pre-trained on natural video-text datasets for privacy-preserved ophthalmic surgical video generation based on Ophora-160K. Experiments on video quality evaluation via quantitative analysis and ophthalmologist feedback demonstrate that Ophora can generate realistic and reliable ophthalmic surgical videos based on surgeon instructions. We also validate the capability of Ophora for empowering downstream tasks of ophthalmic surgical workflow understanding. Code is available at https://github.com/mar-cry/Ophora.
>
---
#### [replaced 014] Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05280v2](http://arxiv.org/pdf/2506.05280v2)**

> **作者:** Nan Wang; Yuantao Chen; Lixing Xiao; Weiqing Xiao; Bohan Li; Zhaoxi Chen; Chongjie Ye; Shaocong Xu; Saining Zhang; Ziyang Yan; Pierre Merriaux; Lei Lei; Tianfan Xue; Hao Zhao
>
> **备注:** Project page: https://bigcileng.github.io/bilateral-driving ; Code: https://github.com/BigCiLeng/bilateral-driving
>
> **摘要:** Neural rendering techniques, including NeRF and Gaussian Splatting (GS), rely on photometric consistency to produce high-quality reconstructions. However, in real-world scenarios, it is challenging to guarantee perfect photometric consistency in acquired images. Appearance codes have been widely used to address this issue, but their modeling capability is limited, as a single code is applied to the entire image. Recently, the bilateral grid was introduced to perform pixel-wise color mapping, but it is difficult to optimize and constrain effectively. In this paper, we propose a novel multi-scale bilateral grid that unifies appearance codes and bilateral grids. We demonstrate that this approach significantly improves geometric accuracy in dynamic, decoupled autonomous driving scene reconstruction, outperforming both appearance codes and bilateral grids. This is crucial for autonomous driving, where accurate geometry is important for obstacle avoidance and control. Our method shows strong results across four datasets: Waymo, NuScenes, Argoverse, and PandaSet. We further demonstrate that the improvement in geometry is driven by the multi-scale bilateral grid, which effectively reduces floaters caused by photometric inconsistency.
>
---
#### [replaced 015] Defurnishing with X-Ray Vision: Joint Removal of Furniture from Panoramas and Mesh
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05338v2](http://arxiv.org/pdf/2506.05338v2)**

> **作者:** Alan Dolhasz; Chen Ma; Dave Gausebeck; Kevin Chen; Gregor Miller; Lucas Hayne; Gunnar Hovden; Azwad Sabik; Olaf Brandt; Mira Slavcheva
>
> **备注:** Paper website: https://matterport.github.io/defurnishing-with-x-ray-vision/
>
> **摘要:** We present a pipeline for generating defurnished replicas of indoor spaces represented as textured meshes and corresponding multi-view panoramic images. To achieve this, we first segment and remove furniture from the mesh representation, extend planes, and fill holes, obtaining a simplified defurnished mesh (SDM). This SDM acts as an ``X-ray'' of the scene's underlying structure, guiding the defurnishing process. We extract Canny edges from depth and normal images rendered from the SDM. We then use these as a guide to remove the furniture from panorama images via ControlNet inpainting. This control signal ensures the availability of global geometric information that may be hidden from a particular panoramic view by the furniture being removed. The inpainted panoramas are used to texture the mesh. We show that our approach produces higher quality assets than methods that rely on neural radiance fields, which tend to produce blurry low-resolution images, or RGB-D inpainting, which is highly susceptible to hallucinations.
>
---
#### [replaced 016] TMT: Tri-Modal Translation between Speech, Image, and Text by Processing Different Modalities as Different Languages
- **分类: cs.CL; cs.AI; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2402.16021v2](http://arxiv.org/pdf/2402.16021v2)**

> **作者:** Minsu Kim; Jee-weon Jung; Hyeongseop Rha; Soumi Maiti; Siddhant Arora; Xuankai Chang; Shinji Watanabe; Yong Man Ro
>
> **备注:** IEEE TMM
>
> **摘要:** The capability to jointly process multi-modal information is becoming an essential task. However, the limited number of paired multi-modal data and the large computational requirements in multi-modal learning hinder the development. We propose a novel Tri-Modal Translation (TMT) model that translates between arbitrary modalities spanning speech, image, and text. We introduce a novel viewpoint, where we interpret different modalities as different languages, and treat multi-modal translation as a well-established machine translation problem. To this end, we tokenize speech and image data into discrete tokens, which provide a unified interface across modalities and significantly decrease the computational cost. In the proposed TMT, a multi-modal encoder-decoder conducts the core translation, whereas modality-specific processing is conducted only within the tokenization and detokenization stages. We evaluate the proposed TMT on all six modality translation tasks. TMT outperforms single model counterparts consistently, demonstrating that unifying tasks is beneficial not only for practicality but also for performance.
>
---
#### [replaced 017] SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference
- **分类: cs.LG; cs.AI; cs.CV; cs.PF**

- **链接: [http://arxiv.org/pdf/2502.18137v5](http://arxiv.org/pdf/2502.18137v5)**

> **作者:** Jintao Zhang; Chendong Xiang; Haofeng Huang; Jia Wei; Haocheng Xi; Jun Zhu; Jianfei Chen
>
> **备注:** @inproceedings{zhang2025spargeattn, title={Spargeattn: Accurate sparse attention accelerating any model inference}, author={Zhang, Jintao and Xiang, Chendong and Huang, Haofeng and Wei, Jia and Xi, Haocheng and Zhu, Jun and Chen, Jianfei}, booktitle={International Conference on Machine Learning (ICML)}, year={2025} }
>
> **摘要:** An efficient attention implementation is essential for large models due to its quadratic time complexity. Fortunately, attention commonly exhibits sparsity, i.e., many values in the attention map are near zero, allowing for the omission of corresponding computations. Many studies have utilized the sparse pattern to accelerate attention. However, most existing works focus on optimizing attention within specific models by exploiting certain sparse patterns of the attention map. A universal sparse attention that guarantees both the speedup and end-to-end performance of diverse models remains elusive. In this paper, we propose SpargeAttn, a universal sparse and quantized attention for any model. Our method uses a two-stage online filter: in the first stage, we rapidly and accurately predict the attention map, enabling the skip of some matrix multiplications in attention. In the second stage, we design an online softmax-aware filter that incurs no extra overhead and further skips some matrix multiplications. Experiments show that our method significantly accelerates diverse models, including language, image, and video generation, without sacrificing end-to-end metrics. The codes are available at https://github.com/thu-ml/SpargeAttn.
>
---
#### [replaced 018] In Search of Forgotten Domain Generalization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.08258v2](http://arxiv.org/pdf/2410.08258v2)**

> **作者:** Prasanna Mayilvahanan; Roland S. Zimmermann; Thaddäus Wiedemer; Evgenia Rusak; Attila Juhos; Matthias Bethge; Wieland Brendel
>
> **备注:** ICLR 2025 camera-ready version
>
> **摘要:** Out-of-Domain (OOD) generalization is the ability of a model trained on one or more domains to generalize to unseen domains. In the ImageNet era of computer vision, evaluation sets for measuring a model's OOD performance were designed to be strictly OOD with respect to style. However, the emergence of foundation models and expansive web-scale datasets has obfuscated this evaluation process, as datasets cover a broad range of domains and risk test domain contamination. In search of the forgotten domain generalization, we create large-scale datasets subsampled from LAION -- LAION-Natural and LAION-Rendition -- that are strictly OOD to corresponding ImageNet and DomainNet test sets in terms of style. Training CLIP models on these datasets reveals that a significant portion of their performance is explained by in-domain examples. This indicates that the OOD generalization challenges from the ImageNet era still prevail and that training on web-scale data merely creates the illusion of OOD generalization. Furthermore, through a systematic exploration of combining natural and rendition datasets in varying proportions, we identify optimal mixing ratios for model generalization across these domains. Our datasets and results re-enable meaningful assessment of OOD robustness at scale -- a crucial prerequisite for improving model robustness.
>
---
#### [replaced 019] Enhancing pretraining efficiency for medical image segmentation via transferability metrics
- **分类: cs.CV; cs.LG; eess.IV; I.4.6**

- **链接: [http://arxiv.org/pdf/2410.18677v2](http://arxiv.org/pdf/2410.18677v2)**

> **作者:** Gábor Hidy; Bence Bakos; András Lukács
>
> **备注:** An error was discovered in the aggregation process of our results, particularly affecting the experiments involving the advanced pretraining method. This impacts the main conclusions of the paper, and we are therefore withdrawing the submission
>
> **摘要:** In medical image segmentation tasks, the scarcity of labeled training data poses a significant challenge when training deep neural networks. When using U-Net-style architectures, it is common practice to address this problem by pretraining the encoder part on a large general-purpose dataset like ImageNet. However, these methods are resource-intensive and do not guarantee improved performance on the downstream task. In this paper we investigate a variety of training setups on medical image segmentation datasets, using ImageNet-pretrained models. By examining over 300 combinations of models, datasets, and training methods, we find that shorter pretraining often leads to better results on the downstream task, providing additional proof to the well-known fact that the accuracy of the model on ImageNet is a poor indicator for downstream performance. As our main contribution, we introduce a novel transferability metric, based on contrastive learning, that measures how robustly a pretrained model is able to represent the target data. In contrast to other transferability scores, our method is applicable to the case of transferring from ImageNet classification to medical image segmentation. We apply our robustness score by measuring it throughout the pretraining phase to indicate when the model weights are optimal for downstream transfer. This reduces pretraining time and improves results on the target task.
>
---
#### [replaced 020] Feature-Based Lie Group Transformer for Real-World Applications
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.04668v2](http://arxiv.org/pdf/2506.04668v2)**

> **作者:** Takayuki Komatsu; Yoshiyuki Ohmura; Kayato Nishitsunoi; Yasuo Kuniyoshi
>
> **摘要:** The main goal of representation learning is to acquire meaningful representations from real-world sensory inputs without supervision. Representation learning explains some aspects of human development. Various neural network (NN) models have been proposed that acquire empirically good representations. However, the formulation of a good representation has not been established. We recently proposed a method for categorizing changes between a pair of sensory inputs. A unique feature of this approach is that transformations between two sensory inputs are learned to satisfy algebraic structural constraints. Conventional representation learning often assumes that disentangled independent feature axes is a good representation; however, we found that such a representation cannot account for conditional independence. To overcome this problem, we proposed a new method using group decomposition in Galois algebra theory. Although this method is promising for defining a more general representation, it assumes pixel-to-pixel translation without feature extraction, and can only process low-resolution images with no background, which prevents real-world application. In this study, we provide a simple method to apply our group decomposition theory to a more realistic scenario by combining feature extraction and object segmentation. We replace pixel translation with feature translation and formulate object segmentation as grouping features under the same transformation. We validated the proposed method on a practical dataset containing both real-world object and background. We believe that our model will lead to a better understanding of human development of object recognition in the real world.
>
---
#### [replaced 021] SemiOccam: A Robust Semi-Supervised Image Recognition Network Using Sparse Labels
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.03582v2](http://arxiv.org/pdf/2506.03582v2)**

> **作者:** Rui Yann; Xianglei Xing
>
> **备注:** CleanSTL-10 available at https://huggingface.co/datasets/Shu1L0n9/CleanSTL-10
>
> **摘要:** We present SemiOccam, an image recognition network that leverages semi-supervised learning in a highly efficient manner. Existing works often rely on complex training techniques and architectures, requiring hundreds of GPU hours for training, while their generalization ability when dealing with extremely limited labeled data remains to be improved. To address these limitations, we construct a hierarchical mixture density classification decision mechanism by optimizing mutual information between feature representations and target classes, compressing redundant information while retaining crucial discriminative components. Experimental results demonstrate that our method achieves state-of-the-art performance on various datasets when using negligible labeled samples, and its simple architecture keeps training time to minute-level. Notably, this paper reveals a long-overlooked data leakage issue in the STL-10 dataset for semi-supervised learning tasks and removes duplicates to ensure the reliability of experimental results. We also release the deduplicated CleanSTL-10 dataset to facilitate fair and reliable research in future semi-supervised learning. Code available at https://github.com/Shu1L0n9/SemiOccam.
>
---
#### [replaced 022] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.13540v3](http://arxiv.org/pdf/2412.13540v3)**

> **作者:** Yingjie Zhu; Xuefeng Bai; Kehai Chen; Yang Xiang; Jun Yu; Min Zhang
>
> **备注:** Accepted by ACL2025 main conference
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across diverse tasks. Despite great success, recent studies show that LVLMs encounter substantial limitations when engaging with visual graphs. To study the reason behind these limitations, we propose VGCure, a comprehensive benchmark covering 22 tasks for examining the fundamental graph understanding and reasoning capacities of LVLMs. Extensive evaluations conducted on 14 LVLMs reveal that LVLMs are weak in basic graph understanding and reasoning tasks, particularly those concerning relational or structurally complex information. Based on this observation, we propose a structure-aware fine-tuning framework to enhance LVLMs with structure learning abilities through three self-supervised learning tasks. Experiments validate the effectiveness of our method in improving LVLMs' performance on fundamental and downstream graph learning tasks, as well as enhancing their robustness against complex visual graphs.
>
---
#### [replaced 023] Explainable Concept Generation through Vision-Language Preference Learning for Understanding Neural Networks' Internal Representations
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.13438v3](http://arxiv.org/pdf/2408.13438v3)**

> **作者:** Aditya Taparia; Som Sagar; Ransalu Senanayake
>
> **备注:** 28 pages, 31 figures
>
> **摘要:** Understanding the inner representation of a neural network helps users improve models. Concept-based methods have become a popular choice for explaining deep neural networks post-hoc because, unlike most other explainable AI techniques, they can be used to test high-level visual "concepts" that are not directly related to feature attributes. For instance, the concept of "stripes" is important to classify an image as a zebra. Concept-based explanation methods, however, require practitioners to guess and manually collect multiple candidate concept image sets, making the process labor-intensive and prone to overlooking important concepts. Addressing this limitation, in this paper, we frame concept image set creation as an image generation problem. However, since naively using a standard generative model does not result in meaningful concepts, we devise a reinforcement learning-based preference optimization (RLPO) algorithm that fine-tunes a vision-language generative model from approximate textual descriptions of concepts. Through a series of experiments, we demonstrate our method's ability to efficiently and reliably articulate diverse concepts that are otherwise challenging to craft manually.
>
---
#### [replaced 024] ZeroFlow: Overcoming Catastrophic Forgetting is Easier than You Think
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.01045v4](http://arxiv.org/pdf/2501.01045v4)**

> **作者:** Tao Feng; Wei Li; Didi Zhu; Hangjie Yuan; Wendi Zheng; Dan Zhang; Jie Tang
>
> **摘要:** Backpropagation provides a generalized configuration for overcoming catastrophic forgetting. Optimizers such as SGD and Adam are commonly used for weight updates in continual learning and continual pre-training. However, access to gradient information is not always feasible in practice due to black-box APIs, hardware constraints, or non-differentiable systems, a challenge we refer to as the gradient bans. To bridge this gap, we introduce ZeroFlow, the first benchmark designed to evaluate gradient-free optimization algorithms for overcoming forgetting. ZeroFlow examines a suite of forward pass-based methods across various algorithms, forgetting scenarios, and datasets. Our results show that forward passes alone can be sufficient to mitigate forgetting. We uncover novel optimization principles that highlight the potential of forward pass-based methods in mitigating forgetting, managing task conflicts, and reducing memory demands. Additionally, we propose new enhancements that further improve forgetting resistance using only forward passes. This work provides essential tools and insights to advance the development of forward-pass-based methods for continual learning.
>
---
#### [replaced 025] TASTE-Rob: Advancing Video Generation of Task-Oriented Hand-Object Interaction for Generalizable Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.11423v2](http://arxiv.org/pdf/2503.11423v2)**

> **作者:** Hongxiang Zhao; Xingchen Liu; Mutian Xu; Yiming Hao; Weikai Chen; Xiaoguang Han
>
> **备注:** CVPR 2025; Project Page: https://taste-rob.github.io
>
> **摘要:** We address key limitations in existing datasets and models for task-oriented hand-object interaction video generation, a critical approach of generating video demonstrations for robotic imitation learning. Current datasets, such as Ego4D, often suffer from inconsistent view perspectives and misaligned interactions, leading to reduced video quality and limiting their applicability for precise imitation learning tasks. Towards this end, we introduce TASTE-Rob -- a pioneering large-scale dataset of 100,856 ego-centric hand-object interaction videos. Each video is meticulously aligned with language instructions and recorded from a consistent camera viewpoint to ensure interaction clarity. By fine-tuning a Video Diffusion Model (VDM) on TASTE-Rob, we achieve realistic object interactions, though we observed occasional inconsistencies in hand grasping postures. To enhance realism, we introduce a three-stage pose-refinement pipeline that improves hand posture accuracy in generated videos. Our curated dataset, coupled with the specialized pose-refinement framework, provides notable performance gains in generating high-quality, task-oriented hand-object interaction videos, resulting in achieving superior generalizable robotic manipulation. The TASTE-Rob dataset is publicly available to foster further advancements in the field, TASTE-Rob dataset and source code will be made publicly available on our website https://taste-rob.github.io.
>
---
#### [replaced 026] RB-SCD: A New Benchmark for Semantic Change Detection of Roads and Bridges in Traffic Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13212v2](http://arxiv.org/pdf/2505.13212v2)**

> **作者:** Qingling Shu; Sibao Chen; Zhihui You; Wei Lu; Jin Tang; Bin Luo
>
> **摘要:** With the rapid modernization of urban transportation, accurately detecting changes such as road and bridge construction, renovation, and demolition is crucial for urban planning and traffic management. However, existing methods often struggle to extract fine-grained semantic changes in complex traffic scenes, largely due to the lack of high-quality annotated change detection (CD) datasets. To address this, we introduce the Road and Bridge Semantic Change Detection (RB-SCD) dataset, a comprehensive benchmark consisting of 260 pairs of high-resolution remote sensing images. RB-SCD spans diverse geographic areas and includes a wide variety of road and bridge types across over ten cities in multiple countries. It covers 11 distinct categories of semantic changes, enabling detailed structural and functional analysis. Based on this challenging dataset, we propose a novel framework called the Multimodal Frequency-Driven Change Detector (MFDCD). For the first time, MFDCD integrates multimodal feature characteristics in the frequency domain. It comprises two key components: the Dynamic Frequency Coupler (DFC) and the Textual Frequency Filter (TFF). DFC couples hierarchical visual features with wavelet-based frequency components, enhancing the perception of fine-grained and cross-temporal structural changes. TFF transforms textual features extracted by the CLIP model into the frequency domain via Fourier transform and applies graph-based filtering to extract salient frequency responses. These are then fused with visual features to enable effective multimodal representation learning. Extensive experiments show that MFDCD achieves strong performance on RB-SCD and three public benchmarks. The RB-SCD dataset, with its rich and diverse annotations, serves as a valuable resource for advancing research in road and bridge change detection under complex traffic conditions.
>
---
#### [replaced 027] Seeing like a Cephalopod: Colour Vision with a Monochrome Event Camera
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10984v2](http://arxiv.org/pdf/2504.10984v2)**

> **作者:** Sami Arja; Nimrod Kruger; Alexandre Marcireau; Nicholas Owen Ralph; Saeed Afshar; Gregory Cohen
>
> **备注:** 15 pages, 14 figures, 1 table. Accepted at CVPR 2025 (Workshop on Event-based Vision)
>
> **摘要:** Cephalopods exhibit unique colour discrimination capabilities despite having one type of photoreceptor, relying instead on chromatic aberration induced by their ocular optics and pupil shapes to perceive spectral information. We took inspiration from this biological mechanism to design a spectral imaging system that combines a ball lens with an event-based camera. Our approach relies on a motorised system that shifts the focal position, mirroring the adaptive lens motion in cephalopods. This approach has enabled us to achieve wavelength-dependent focusing across the visible light and near-infrared spectrum, making the event a spectral sensor. We characterise chromatic aberration effects, using both event-based and conventional frame-based sensors, validating the effectiveness of bio-inspired spectral discrimination both in simulation and in a real setup as well as assessing the spectral discrimination performance. Our proposed approach provides a robust spectral sensing capability without conventional colour filters or computational demosaicing. This approach opens new pathways toward new spectral sensing systems inspired by nature's evolutionary solutions. Code and analysis are available at: https://samiarja.github.io/neuromorphic_octopus_eye/
>
---
#### [replaced 028] Diving into Self-Evolving Training for Multimodal Reasoning
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.17451v3](http://arxiv.org/pdf/2412.17451v3)**

> **作者:** Wei Liu; Junlong Li; Xiwen Zhang; Fan Zhou; Yu Cheng; Junxian He
>
> **备注:** ICML 2025, Project Page: https://mstar-lmm.github.io
>
> **摘要:** Self-evolving trainin--where models iteratively learn from their own outputs--has emerged as a key approach for complex reasoning tasks, addressing the scarcity of high-quality chain-of-thought data. However, its effectiveness in multimodal reasoning, a domain more intricate than text-only reasoning, remains underexplored, and the understanding of critical factors in this training paradigm remains limited. Furthermore, a central challenge for this training method is performance saturation, which impedes further improvements and scalability. Inspired by reinforcement learning (RL), in this paper, we reframe self-evolving training for multimodal reasoning through the lens of RL, identifying three pivotal factors: Training Method, Reward Model, and Prompt Variation. Through systematic analysis, we establish relatively optimal design principles that significantly enhance multimodal reasoning capabilities. Moreover, delving deeper into training dynamics, we uncover the roots of saturation and propose a new automatic balancing mechanism to mitigate this limitation. Building on these insights, we propose M-STAR (Multimodal Self-evolving Training for Reasoning), a framework that achieves consistent performance gains across models of varying sizes and diverse benchmarks. All resources are made publicly available at https://mstar-lmm.github.io.
>
---
#### [replaced 029] From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.10685v2](http://arxiv.org/pdf/2411.10685v2)**

> **作者:** Jinhong Lin; Cheng-En Wu; Huanran Li; Jifan Zhang; Yu Hen Hu; Pedro Morgado
>
> **备注:** Accepted to CVPR2025
>
> **摘要:** Masked Image Modeling (MIM) has emerged as a powerful self-supervised learning paradigm for visual representation learning, enabling models to acquire rich visual representations by predicting masked portions of images from their visible regions. While this approach has shown promising results, we hypothesize that its effectiveness may be limited by optimization challenges during early training stages, where models are expected to learn complex image distributions from partial observations before developing basic visual processing capabilities. To address this limitation, we propose a prototype-driven curriculum leagrning framework that structures the learning process to progress from prototypical examples to more complex variations in the dataset. Our approach introduces a temperature-based annealing scheme that gradually expands the training distribution, enabling more stable and efficient learning trajectories. Through extensive experiments on ImageNet-1K, we demonstrate that our curriculum learning strategy significantly improves both training efficiency and representation quality while requiring substantially fewer training epochs compared to standard Masked Auto-Encoding. Our findings suggest that carefully controlling the order of training examples plays a crucial role in self-supervised visual learning, providing a practical solution to the early-stage optimization challenges in MIM.
>
---
#### [replaced 030] CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.23330v2](http://arxiv.org/pdf/2410.23330v2)**

> **作者:** Tianyu Yang; Lisen Dai; Xiangqi Wang; Minhao Cheng; Yapeng Tian; Xiangliang Zhang
>
> **备注:** ACL main 2025
>
> **摘要:** Machine unlearning (MU) has gained significant attention as a means to remove specific data from trained models without requiring a full retraining process. While progress has been made in unimodal domains like text and image classification, unlearning in multimodal models remains relatively underexplored. In this work, we address the unique challenges of unlearning in CLIP, a prominent multimodal model that aligns visual and textual representations. We introduce CLIPErase, a novel approach that disentangles and selectively forgets both visual and textual associations, ensuring that unlearning does not compromise model performance. CLIPErase consists of three key modules: a Forgetting Module that disrupts the associations in the forget set, a Retention Module that preserves performance on the retain set, and a Consistency Module that maintains consistency with the original model. Extensive experiments on the CIFAR-100 and Flickr30K datasets across four CLIP downstream tasks demonstrate that CLIPErase effectively forgets designated associations in zero-shot tasks for multimodal samples, while preserving the model's performance on the retain set after unlearning.
>
---
#### [replaced 031] SageAttention2++: A More Efficient Implementation of SageAttention2
- **分类: cs.LG; cs.AI; cs.AR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21136v3](http://arxiv.org/pdf/2505.21136v3)**

> **作者:** Jintao Zhang; Xiaoming Xu; Jia Wei; Haofeng Huang; Pengle Zhang; Chendong Xiang; Jun Zhu; Jianfei Chen
>
> **摘要:** The efficiency of attention is critical because its time complexity grows quadratically with sequence length. SageAttention2 addresses this by utilizing quantization to accelerate matrix multiplications (Matmul) in attention. To further accelerate SageAttention2, we propose to utilize the faster instruction of FP8 Matmul accumulated in FP16. The instruction is 2x faster than the FP8 Matmul used in SageAttention2. Our experiments show that SageAttention2++ achieves a 3.9x speedup over FlashAttention while maintaining the same attention accuracy as SageAttention2. This means SageAttention2++ effectively accelerates various models, including those for language, image, and video generation, with negligible end-to-end metrics loss. The code will be available at https://github.com/thu-ml/SageAttention.
>
---
#### [replaced 032] An Ensemble-Based Two-Step Framework for Classification of Pap Smear Cell Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10312v3](http://arxiv.org/pdf/2503.10312v3)**

> **作者:** Theo Di Piazza; Loic Boussel
>
> **备注:** 7 pages, 3 figures, Grand Challenge paper accepted for publication at ISBI 2025
>
> **摘要:** Early detection of cervical cancer is crucial for improving patient outcomes and reducing mortality by identifying precancerous lesions as soon as possible. As a result, the use of pap smear screening has significantly increased, leading to a growing demand for automated tools that can assist cytologists managing their rising workload. To address this, the Pap Smear Cell Classification Challenge (PS3C) has been organized in association with ISBI in 2025. This project aims to promote the development of automated tools for pap smear images classification. The analyzed images are grouped into four categories: healthy, unhealthy, both, and rubbish images which are considered as unsuitable for diagnosis. In this work, we propose a two-stage ensemble approach: first, a neural network determines whether an image is rubbish or not. If not, a second neural network classifies the image as containing a healthy cell, an unhealthy cell, or both.
>
---
#### [replaced 033] GENIUS: A Generative Framework for Universal Multimodal Search
- **分类: cs.IR; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.19868v2](http://arxiv.org/pdf/2503.19868v2)**

> **作者:** Sungyeon Kim; Xinliang Zhu; Xiaofan Lin; Muhammet Bastan; Douglas Gray; Suha Kwak
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Generative retrieval is an emerging approach in information retrieval that generates identifiers (IDs) of target data based on a query, providing an efficient alternative to traditional embedding-based retrieval methods. However, existing models are task-specific and fall short of embedding-based retrieval in performance. This paper proposes GENIUS, a universal generative retrieval framework supporting diverse tasks across multiple modalities and domains. At its core, GENIUS introduces modality-decoupled semantic quantization, transforming multimodal data into discrete IDs encoding both modality and semantics. Moreover, to enhance generalization, we propose a query augmentation that interpolates between a query and its target, allowing GENIUS to adapt to varied query forms. Evaluated on the M-BEIR benchmark, it surpasses prior generative methods by a clear margin. Unlike embedding-based retrieval, GENIUS consistently maintains high retrieval speed across database size, with competitive performance across multiple benchmarks. With additional re-ranking, GENIUS often achieves results close to those of embedding-based methods while preserving efficiency.
>
---
#### [replaced 034] Skywork R1V2: Multimodal Hybrid Reinforcement Learning for Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.16656v4](http://arxiv.org/pdf/2504.16656v4)**

> **作者:** Peiyu Wang; Yichen Wei; Yi Peng; Xiaokun Wang; Weijie Qiu; Wei Shen; Tianyidan Xie; Jiangbo Pei; Jianhao Zhang; Yunzhuo Hao; Xuchen Song; Yang Liu; Yahui Zhou
>
> **摘要:** We present Skywork R1V2, a next-generation multimodal reasoning model and a major leap forward from its predecessor, Skywork R1V. At its core, R1V2 introduces a hybrid reinforcement learning paradigm that jointly leverages the Mixed Preference Optimization (MPO) and the Group Relative Policy Optimization (GRPO), which harmonizes reward-model guidance with rule-based strategies, thereby addressing the long-standing challenge of balancing sophisticated reasoning capabilities with broad generalization. To further enhance training efficiency, we propose the Selective Sample Buffer (SSB) mechanism, which effectively addresses the vanishing advantages dilemma inherent in GRPO by prioritizing high-value samples throughout the optimization process. Notably, we observe that excessive reinforcement signals can induce visual hallucinations--a phenomenon we systematically monitor and mitigate through calibrated reward thresholds throughout the training process. Empirical results affirm the exceptional capability of R1V2, with benchmark-leading performances such as 62.6 on OlympiadBench, 78.9 on AIME2024, 63.6 on LiveCodeBench, and 73.6 on MMMU. These results underscore R1V2's superiority over existing open-source models and demonstrate significant progress in closing the performance gap with premier proprietary systems, including Gemini 2.5 and OpenAI-o4-mini. The Skywork R1V2 model weights have been publicly released to promote openness and reproducibility https://huggingface.co/Skywork/Skywork-R1V2-38B.
>
---
#### [replaced 035] CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.14914v3](http://arxiv.org/pdf/2502.14914v3)**

> **作者:** Zhihang Liu; Chen-Wei Xie; Bin Wen; Feiwu Yu; Jixuan Chen; Pandeng Li; Boqiang Zhang; Nianzu Yang; Yinglu Li; Zuan Gao; Yun Zheng; Hongtao Xie
>
> **摘要:** Visual captioning benchmarks have become outdated with the emergence of modern multimodal large language models (MLLMs), as the brief ground-truth sentences and traditional metrics fail to assess detailed captions effectively. While recent benchmarks attempt to address this by focusing on keyword extraction or object-centric evaluation, they remain limited to vague-view or object-view analyses and incomplete visual element coverage. In this paper, we introduce CAPability, a comprehensive multi-view benchmark for evaluating visual captioning across 12 dimensions spanning six critical views. We curate nearly 11K human-annotated images and videos with visual element annotations to evaluate the generated captions. CAPability stably assesses both the correctness and thoroughness of captions with \textit{precision} and \textit{hit} metrics. By converting annotations to QA pairs, we further introduce a heuristic metric, \textit{know but cannot tell} ($K\bar{T}$), indicating a significant performance gap between QA and caption capabilities. Our work provides a holistic analysis of MLLMs' captioning abilities, as we identify their strengths and weaknesses across various dimensions, guiding future research to enhance specific aspects of their capabilities.
>
---
#### [replaced 036] Balancing Beyond Discrete Categories: Continuous Demographic Labels for Fair Face Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01532v4](http://arxiv.org/pdf/2506.01532v4)**

> **作者:** Pedro C. Neto; Naser Damer; Jaime S. Cardoso; Ana F. Sequeira
>
> **备注:** Under review
>
> **摘要:** Bias has been a constant in face recognition models. Over the years, researchers have looked at it from both the model and the data point of view. However, their approach to mitigation of data bias was limited and lacked insight on the real nature of the problem. Here, in this document, we propose to revise our use of ethnicity labels as a continuous variable instead of a discrete value per identity. We validate our formulation both experimentally and theoretically, showcasing that not all identities from one ethnicity contribute equally to the balance of the dataset; thus, having the same number of identities per ethnicity does not represent a balanced dataset. We further show that models trained on datasets balanced in the continuous space consistently outperform models trained on data balanced in the discrete space. We trained more than 65 different models, and created more than 20 subsets of the original datasets.
>
---
#### [replaced 037] Universal Domain Adaptation for Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22458v2](http://arxiv.org/pdf/2505.22458v2)**

> **作者:** Seun-An Choe; Keon-Hee Park; Jinwoo Choi; Gyeong-Moon Park
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Unsupervised domain adaptation for semantic segmentation (UDA-SS) aims to transfer knowledge from labeled source data to unlabeled target data. However, traditional UDA-SS methods assume that category settings between source and target domains are known, which is unrealistic in real-world scenarios. This leads to performance degradation if private classes exist. To address this limitation, we propose Universal Domain Adaptation for Semantic Segmentation (UniDA-SS), achieving robust adaptation even without prior knowledge of category settings. We define the problem in the UniDA-SS scenario as low confidence scores of common classes in the target domain, which leads to confusion with private classes. To solve this problem, we propose UniMAP: UniDA-SS with Image Matching and Prototype-based Distinction, a novel framework composed of two key components. First, Domain-Specific Prototype-based Distinction (DSPD) divides each class into two domain-specific prototypes, enabling finer separation of domain-specific features and enhancing the identification of common classes across domains. Second, Target-based Image Matching (TIM) selects a source image containing the most common-class pixels based on the target pseudo-label and pairs it in a batch to promote effective learning of common classes. We also introduce a new UniDA-SS benchmark and demonstrate through various experiments that UniMAP significantly outperforms baselines. The code is available at https://github.com/KU-VGI/UniMAP.
>
---
#### [replaced 038] Feedforward Few-shot Species Range Estimation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.14977v2](http://arxiv.org/pdf/2502.14977v2)**

> **作者:** Christian Lange; Max Hamilton; Elijah Cole; Alexander Shepard; Samuel Heinrich; Angela Zhu; Subhransu Maji; Grant Van Horn; Oisin Mac Aodha
>
> **备注:** Published in the Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)
>
> **摘要:** Knowing where a particular species can or cannot be found on Earth is crucial for ecological research and conservation efforts. By mapping the spatial ranges of all species, we would obtain deeper insights into how global biodiversity is affected by climate change and habitat loss. However, accurate range estimates are only available for a relatively small proportion of all known species. For the majority of the remaining species, we typically only have a small number of records denoting the spatial locations where they have previously been observed. We outline a new approach for few-shot species range estimation to address the challenge of accurately estimating the range of a species from limited data. During inference, our model takes a set of spatial locations as input, along with optional metadata such as text or an image, and outputs a species encoding that can be used to predict the range of a previously unseen species in a feedforward manner. We evaluate our approach on two challenging benchmarks, where we obtain state-of-the-art range estimation performance, in a fraction of the compute time, compared to recent alternative approaches.
>
---
#### [replaced 039] Application of convolutional neural networks in image super-resolution
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.02604v2](http://arxiv.org/pdf/2506.02604v2)**

> **作者:** Chunwei Tian; Mingjian Song; Wangmeng Zuo; Bo Du; Yanning Zhang; Shichao Zhang
>
> **备注:** It has been accepted by CAAI transactions on intelligent systems, in Chinese language
>
> **摘要:** Due to strong learning abilities of convolutional neural networks (CNNs), they have become mainstream methods for image super-resolution. However, there are big differences of different deep learning methods with different types. There is little literature to summarize relations and differences of different methods in image super-resolution. Thus, summarizing these literatures are important, according to loading capacity and execution speed of devices. This paper first introduces principles of CNNs in image super-resolution, then introduces CNNs based bicubic interpolation, nearest neighbor interpolation, bilinear interpolation, transposed convolution, sub-pixel layer, meta up-sampling for image super-resolution to analyze differences and relations of different CNNs based interpolations and modules, and compare performance of these methods by experiments. Finally, this paper gives potential research points and drawbacks and summarizes the whole paper, which can facilitate developments of CNNs in image super-resolution.
>
---
#### [replaced 040] MARS: Radio Map Super-resolution and Reconstruction Method under Sparse Channel Measurements
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.04682v2](http://arxiv.org/pdf/2506.04682v2)**

> **作者:** Chuyun Deng; Na Liu; Wei Xie; Lianming Xu; Li Wang
>
> **摘要:** Radio maps reflect the spatial distribution of signal strength and are essential for applications like smart cities, IoT, and wireless network planning. However, reconstructing accurate radio maps from sparse measurements remains challenging. Traditional interpolation and inpainting methods lack environmental awareness, while many deep learning approaches depend on detailed scene data, limiting generalization. To address this, we propose MARS, a Multi-scale Aware Radiomap Super-resolution method that combines CNNs and Transformers with multi-scale feature fusion and residual connections. MARS focuses on both global and local feature extraction, enhancing feature representation across different receptive fields and improving reconstruction accuracy. Experiments across different scenes and antenna locations show that MARS outperforms baseline models in both MSE and SSIM, while maintaining low computational cost, demonstrating strong practical potential.
>
---
#### [replaced 041] GenSpace: Benchmarking Spatially-Aware Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24870v2](http://arxiv.org/pdf/2505.24870v2)**

> **作者:** Zehan Wang; Jiayang Xu; Ziang Zhang; Tianyu Pang; Chao Du; Hengshuang Zhao; Zhou Zhao
>
> **摘要:** Humans can intuitively compose and arrange scenes in the 3D space for photography. However, can advanced AI image generators plan scenes with similar 3D spatial awareness when creating images from text or image prompts? We present GenSpace, a novel benchmark and evaluation pipeline to comprehensively assess the spatial awareness of current image generation models. Furthermore, standard evaluations using general Vision-Language Models (VLMs) frequently fail to capture the detailed spatial errors. To handle this challenge, we propose a specialized evaluation pipeline and metric, which reconstructs 3D scene geometry using multiple visual foundation models and provides a more accurate and human-aligned metric of spatial faithfulness. Our findings show that while AI models create visually appealing images and can follow general instructions, they struggle with specific 3D details like object placement, relationships, and measurements. We summarize three core limitations in the spatial perception of current state-of-the-art image generation models: 1) Object Perspective Understanding, 2) Egocentric-Allocentric Transformation and 3) Metric Measurement Adherence, highlighting possible directions for improving spatial intelligence in image generation.
>
---
#### [replaced 042] Rethinking Machine Unlearning in Image Generation Models
- **分类: cs.AI; cs.CL; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02761v2](http://arxiv.org/pdf/2506.02761v2)**

> **作者:** Renyang Liu; Wenjie Feng; Tianwei Zhang; Wei Zhou; Xueqi Cheng; See-Kiong Ng
>
> **备注:** Accepted by ACM CCS 2025
>
> **摘要:** With the surge and widespread application of image generation models, data privacy and content safety have become major concerns and attracted great attention from users, service providers, and policymakers. Machine unlearning (MU) is recognized as a cost-effective and promising means to address these challenges. Despite some advancements, image generation model unlearning (IGMU) still faces remarkable gaps in practice, e.g., unclear task discrimination and unlearning guidelines, lack of an effective evaluation framework, and unreliable evaluation metrics. These can hinder the understanding of unlearning mechanisms and the design of practical unlearning algorithms. We perform exhaustive assessments over existing state-of-the-art unlearning algorithms and evaluation standards, and discover several critical flaws and challenges in IGMU tasks. Driven by these limitations, we make several core contributions, to facilitate the comprehensive understanding, standardized categorization, and reliable evaluation of IGMU. Specifically, (1) We design CatIGMU, a novel hierarchical task categorization framework. It provides detailed implementation guidance for IGMU, assisting in the design of unlearning algorithms and the construction of testbeds. (2) We introduce EvalIGMU, a comprehensive evaluation framework. It includes reliable quantitative metrics across five critical aspects. (3) We construct DataIGM, a high-quality unlearning dataset, which can be used for extensive evaluations of IGMU, training content detectors for judgment, and benchmarking the state-of-the-art unlearning algorithms. With EvalIGMU and DataIGM, we discover that most existing IGMU algorithms cannot handle the unlearning well across different evaluation dimensions, especially for preservation and robustness. Code and models are available at https://github.com/ryliu68/IGMU.
>
---
#### [replaced 043] Self-Supervised Generative-Contrastive Learning of Multi-Modal Euclidean Input for 3D Shape Latent Representations: A Dynamic Switching Approach
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2301.04612v2](http://arxiv.org/pdf/2301.04612v2)**

> **作者:** Chengzhi Wu; Julius Pfrommer; Mingyuan Zhou; Jürgen Beyerer
>
> **摘要:** We propose a combined generative and contrastive neural architecture for learning latent representations of 3D volumetric shapes. The architecture uses two encoder branches for voxel grids and multi-view images from the same underlying shape. The main idea is to combine a contrastive loss between the resulting latent representations with an additional reconstruction loss. That helps to avoid collapsing the latent representations as a trivial solution for minimizing the contrastive loss. A novel dynamic switching approach is used to cross-train two encoders with a shared decoder. The switching approach also enables the stop gradient operation on a random branch. Further classification experiments show that the latent representations learned with our self-supervised method integrate more useful information from the additional input data implicitly, thus leading to better reconstruction and classification performance.
>
---
#### [replaced 044] Fréchet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets
- **分类: cs.CV; cs.LG; eess.IV; stat.ML**

- **链接: [http://arxiv.org/pdf/2412.01496v2](http://arxiv.org/pdf/2412.01496v2)**

> **作者:** Nicholas Konz; Richard Osuala; Preeti Verma; Yuwen Chen; Hanxue Gu; Haoyu Dong; Yaqian Chen; Andrew Marshall; Lidia Garrucho; Kaisar Kushibar; Daniel M. Lang; Gene S. Kim; Lars J. Grimm; John M. Lewin; James S. Duncan; Julia A. Schnabel; Oliver Diaz; Karim Lekadir; Maciej A. Mazurowski
>
> **备注:** Codebase for FRD computation: https://github.com/RichardObi/frd-score. Codebase for medical image similarity metric evaluation framework: https://github.com/mazurowski-lab/medical-image-similarity-metrics
>
> **摘要:** Determining whether two sets of images belong to the same or different distributions or domains is a crucial task in modern medical image analysis and deep learning; for example, to evaluate the output quality of image generative models. Currently, metrics used for this task either rely on the (potentially biased) choice of some downstream task, such as segmentation, or adopt task-independent perceptual metrics (e.g., Fr\'echet Inception Distance/FID) from natural imaging, which we show insufficiently capture anatomical features. To this end, we introduce a new perceptual metric tailored for medical images, FRD (Fr\'echet Radiomic Distance), which utilizes standardized, clinically meaningful, and interpretable image features. We show that FRD is superior to other image distribution metrics for a range of medical imaging applications, including out-of-domain (OOD) detection, the evaluation of image-to-image translation (by correlating more with downstream task performance as well as anatomical consistency and realism), and the evaluation of unconditional image generation. Moreover, FRD offers additional benefits such as stability and computational efficiency at low sample sizes, sensitivity to image corruptions and adversarial attacks, feature interpretability, and correlation with radiologist-perceived image quality. Additionally, we address key gaps in the literature by presenting an extensive framework for the multifaceted evaluation of image similarity metrics in medical imaging -- including the first large-scale comparative study of generative models for medical image translation -- and release an accessible codebase to facilitate future research. Our results are supported by thorough experiments spanning a variety of datasets, modalities, and downstream tasks, highlighting the broad potential of FRD for medical image analysis.
>
---
#### [replaced 045] MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.18362v3](http://arxiv.org/pdf/2501.18362v3)**

> **作者:** Yuxin Zuo; Shang Qu; Yifei Li; Zhangren Chen; Xuekai Zhu; Ermo Hua; Kaiyan Zhang; Ning Ding; Bowen Zhou
>
> **备注:** ICML 2025
>
> **摘要:** We introduce MedXpertQA, a highly challenging and comprehensive benchmark to evaluate expert-level medical knowledge and advanced reasoning. MedXpertQA includes 4,460 questions spanning 17 specialties and 11 body systems. It includes two subsets, Text for text evaluation and MM for multimodal evaluation. Notably, MM introduces expert-level exam questions with diverse images and rich clinical information, including patient records and examination results, setting it apart from traditional medical multimodal benchmarks with simple QA pairs generated from image captions. MedXpertQA applies rigorous filtering and augmentation to address the insufficient difficulty of existing benchmarks like MedQA, and incorporates specialty board questions to improve clinical relevance and comprehensiveness. We perform data synthesis to mitigate data leakage risk and conduct multiple rounds of expert reviews to ensure accuracy and reliability. We evaluate 18 leading models on \benchmark. Moreover, medicine is deeply connected to real-world decision-making, providing a rich and representative setting for assessing reasoning abilities beyond mathematics and code. To this end, we develop a reasoning-oriented subset to facilitate the assessment of o1-like models. Code and data are available at: https://github.com/TsinghuaC3I/MedXpertQA
>
---
#### [replaced 046] YOLO-RS: Remote Sensing Enhanced Crop Detection Methods
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.11165v2](http://arxiv.org/pdf/2504.11165v2)**

> **作者:** Linlin Xiao; Zhang Tiancong; Yutong Jia; Xinyu Nie; Mengyao Wang; Xiaohang Shao
>
> **摘要:** With the rapid development of remote sensing technology, crop classification and health detection based on deep learning have gradually become a research hotspot. However, the existing target detection methods show poor performance when dealing with small targets in remote sensing images, especially in the case of complex background and image mixing, which is difficult to meet the practical application requirementsite. To address this problem, a novel target detection model YOLO-RS is proposed in this paper. The model is based on the latest Yolov11 which significantly enhances the detection of small targets by introducing the Context Anchor Attention (CAA) mechanism and an efficient multi-field multi-scale feature fusion network. YOLO-RS adopts a bidirectional feature fusion strategy in the feature fusion process, which effectively enhances the model's performance in the detection of small targets. Small target detection. Meanwhile, the ACmix module at the end of the model backbone network solves the category imbalance problem by adaptively adjusting the contrast and sample mixing, thus enhancing the detection accuracy in complex scenes. In the experiments on the PDT remote sensing crop health detection dataset and the CWC crop classification dataset, YOLO-RS improves both the recall and the mean average precision (mAP) by about 2-3\% or so compared with the existing state-of-the-art methods, while the F1-score is also significantly improved. Moreover, the computational complexity of the model only increases by about 5.2 GFLOPs, indicating its significant advantages in both performance and efficiency. The experimental results validate the effectiveness and application potential of YOLO-RS in the task of detecting small targets in remote sensing images.
>
---
#### [replaced 047] Toward a Low-Cost Perception System in Autonomous Vehicles: A Spectrum Learning Approach
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.01940v2](http://arxiv.org/pdf/2502.01940v2)**

> **作者:** Mohammed Alsakabi; Aidan Erickson; John M. Dolan; Ozan K. Tonguz
>
> **摘要:** We present a cost-effective new approach for generating denser depth maps for Autonomous Driving (AD) and Autonomous Vehicles (AVs) by integrating the images obtained from deep neural network (DNN) 4D radar detectors with conventional camera RGB images. Our approach introduces a novel pixel positional encoding algorithm inspired by Bartlett's spatial spectrum estimation technique. This algorithm transforms both radar depth maps and RGB images into a unified pixel image subspace called the Spatial Spectrum, facilitating effective learning based on their similarities and differences. Our method effectively leverages high-resolution camera images to train radar depth map generative models, addressing the limitations of conventional radar detectors in complex vehicular environments, thus sharpening the radar output. We develop spectrum estimation algorithms tailored for radar depth maps and RGB images, a comprehensive training framework for data-driven generative models, and a camera-radar deployment scheme for AV operation. Our results demonstrate that our approach also outperforms the state-of-the-art (SOTA) by 27.95% in terms of Unidirectional Chamfer Distance (UCD).
>
---
#### [replaced 048] LDPM: Towards undersampled MRI reconstruction with MR-VAE and Latent Diffusion Prior
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.02951v3](http://arxiv.org/pdf/2411.02951v3)**

> **作者:** Xingjian Tang; Jingwei Guan; Linge Li; Ran Shi; Youmei Zhang; Mengye Lyu; Li Yan
>
> **备注:** accepted as oral presentation at EMBC 2025
>
> **摘要:** Diffusion models, as powerful generative models, have found a wide range of applications and shown great potential in solving image reconstruction problems. Some works attempted to solve MRI reconstruction with diffusion models, but these methods operate directly in pixel space, leading to higher computational costs for optimization and inference. Latent diffusion models, pre-trained on natural images with rich visual priors, are expected to solve the high computational cost problem in MRI reconstruction by operating in a lower-dimensional latent space. However, direct application to MRI reconstruction faces three key challenges: (1) absence of explicit control mechanisms for medical fidelity, (2) domain gap between natural images and MR physics, and (3) undefined data consistency in latent space. To address these challenges, a novel Latent Diffusion Prior-based undersampled MRI reconstruction (LDPM) method is proposed. Our LDPM framework addresses these challenges by: (1) a sketch-guided pipeline with a two-step reconstruction strategy, which balances perceptual quality and anatomical fidelity, (2) an MRI-optimized VAE (MR-VAE), which achieves an improvement of approximately 3.92 dB in PSNR for undersampled MRI reconstruction compared to that with SD-VAE \cite{sd}, and (3) Dual-Stage Sampler, a modified version of spaced DDPM sampler, which enforces high-fidelity reconstruction in the latent space. Experiments on the fastMRI dataset\cite{fastmri} demonstrate the state-of-the-art performance of the proposed method and its robustness across various scenarios. The effectiveness of each module is also verified through ablation experiments.
>
---
#### [replaced 049] Progressive Data Dropout: An Embarrassingly Simple Approach to Faster Training
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22342v2](http://arxiv.org/pdf/2505.22342v2)**

> **作者:** Shriram M S; Xinyue Hao; Shihao Hou; Yang Lu; Laura Sevilla-Lara; Anurag Arnab; Shreyank N Gowda
>
> **摘要:** The success of the machine learning field has reliably depended on training on large datasets. While effective, this trend comes at an extraordinary cost. This is due to two deeply intertwined factors: the size of models and the size of datasets. While promising research efforts focus on reducing the size of models, the other half of the equation remains fairly mysterious. Indeed, it is surprising that the standard approach to training remains to iterate over and over, uniformly sampling the training dataset. In this paper we explore a series of alternative training paradigms that leverage insights from hard-data-mining and dropout, simple enough to implement and use that can become the new training standard. The proposed Progressive Data Dropout reduces the number of effective epochs to as little as 12.4% of the baseline. This savings actually do not come at any cost for accuracy. Surprisingly, the proposed method improves accuracy by up to 4.82%. Our approach requires no changes to model architecture or optimizer, and can be applied across standard training pipelines, thus posing an excellent opportunity for wide adoption. Code can be found here: https://github.com/bazyagami/LearningWithRevision
>
---
#### [replaced 050] Subspecialty-Specific Foundation Model for Intelligent Gastrointestinal Pathology
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21928v2](http://arxiv.org/pdf/2505.21928v2)**

> **作者:** Lianghui Zhu; Xitong Ling; Minxi Ouyang; Xiaoping Liu; Tian Guan; Mingxi Fu; Zhiqiang Cheng; Fanglei Fu; Maomao Zeng; Liming Liu; Song Duan; Qiang Huang; Ying Xiao; Jianming Li; Shanming Lu; Zhenghua Piao; Mingxi Zhu; Yibo Jin; Shan Xu; Qiming He; Yizhi Wang; Junru Cheng; Xuanyu Wang; Luxi Xie; Houqiang Li; Sufang Tian; Yonghong He
>
> **摘要:** Gastrointestinal (GI) diseases represent a clinically significant burden, necessitating precise diagnostic approaches to optimize patient outcomes. Conventional histopathological diagnosis suffers from limited reproducibility and diagnostic variability. To overcome these limitations, we develop Digepath, a specialized foundation model for GI pathology. Our framework introduces a dual-phase iterative optimization strategy combining pretraining with fine-screening, specifically designed to address the detection of sparsely distributed lesion areas in whole-slide images. Digepath is pretrained on over 353 million multi-scale images from 210,043 H&E-stained slides of GI diseases. It attains state-of-the-art performance on 33 out of 34 tasks related to GI pathology, including pathological diagnosis, protein expression status prediction, gene mutation prediction, and prognosis evaluation. We further translate the intelligent screening module for early GI cancer and achieve near-perfect 99.70% sensitivity across nine independent medical institutions. This work not only advances AI-driven precision pathology for GI diseases but also bridge critical gaps in histopathological practice.
>
---
#### [replaced 051] On the Importance of Text Preprocessing for Multimodal Representation Learning and Pathology Report Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.19285v3](http://arxiv.org/pdf/2502.19285v3)**

> **作者:** Ruben T. Lucassen; Tijn van de Luijtgaarden; Sander P. J. Moonemans; Gerben E. Breimer; Willeke A. M. Blokx; Mitko Veta
>
> **备注:** 11 pages, 1 figure
>
> **摘要:** Vision-language models in pathology enable multimodal case retrieval and automated report generation. Many of the models developed so far, however, have been trained on pathology reports that include information which cannot be inferred from paired whole slide images (e.g., patient history), potentially leading to hallucinated sentences in generated reports. To this end, we investigate how the selection of information from pathology reports for vision-language modeling affects the quality of the multimodal representations and generated reports. More concretely, we compare a model trained on full reports against a model trained on preprocessed reports that only include sentences describing the cell and tissue appearances based on the H&E-stained slides. For the experiments, we built upon the BLIP-2 framework and used a cutaneous melanocytic lesion dataset of 42,433 H&E-stained whole slide images and 19,636 corresponding pathology reports. Model performance was assessed using image-to-text and text-to-image retrieval, as well as qualitative evaluation of the generated reports by an expert pathologist. Our results demonstrate that text preprocessing prevents hallucination in report generation. Despite the improvement in the quality of the generated reports, training the vision-language model on full reports showed better cross-modal retrieval performance.
>
---
#### [replaced 052] Gaussian Building Mesh (GBM): Extract a Building's 3D Mesh with Google Earth and Gaussian Splatting
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2501.00625v3](http://arxiv.org/pdf/2501.00625v3)**

> **作者:** Kyle Gao; Liangzhi Li; Hongjie He; Dening Lu; Linlin Xu; Jonathan Li
>
> **摘要:** Recently released open-source pre-trained foundational image segmentation and object detection models (SAM2+GroundingDINO) allow for geometrically consistent segmentation of objects of interest in multi-view 2D images. Users can use text-based or click-based prompts to segment objects of interest without requiring labeled training datasets. Gaussian Splatting allows for the learning of the 3D representation of a scene's geometry and radiance based on 2D images. Combining Google Earth Studio, SAM2+GroundingDINO, 2D Gaussian Splatting, and our improvements in mask refinement based on morphological operations and contour simplification, we created a pipeline to extract the 3D mesh of any building based on its name, address, or geographic coordinates.
>
---
#### [replaced 053] SeedEdit 3.0: Fast and High-Quality Generative Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05083v2](http://arxiv.org/pdf/2506.05083v2)**

> **作者:** Peng Wang; Yichun Shi; Xiaochen Lian; Zhonghua Zhai; Xin Xia; Xuefeng Xiao; Weilin Huang; Jianchao Yang
>
> **备注:** Website: https://seed.bytedance.com/tech/seededit
>
> **摘要:** We introduce SeedEdit 3.0, in companion with our T2I model Seedream 3.0, which significantly improves over our previous SeedEdit versions in both aspects of edit instruction following and image content (e.g., ID/IP) preservation on real image inputs. Additional to model upgrading with T2I, in this report, we present several key improvements. First, we develop an enhanced data curation pipeline with a meta-info paradigm and meta-info embedding strategy that help mix images from multiple data sources. This allows us to scale editing data effectively, and meta information is helpfult to connect VLM with diffusion model more closely. Second, we introduce a joint learning pipeline for computing a diffusion loss and reward losses. Finally, we evaluate SeedEdit 3.0 on our testing benchmarks, for real/synthetic image editing, where it achieves a best trade-off between multiple aspects, yielding a high usability rate of 56.1%, compared to SeedEdit 1.6 (38.4%), GPT4o (37.1%) and Gemini 2.0 (30.3%).
>
---
#### [replaced 054] Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.08266v3](http://arxiv.org/pdf/2505.08266v3)**

> **作者:** Yanbin Wei; Xuehao Wang; Zhan Zhuang; Yang Chen; Shuhao Chen; Yulong Zhang; Yu Zhang; James Kwok
>
> **备注:** ICML 2025
>
> **摘要:** Message-passing graph neural networks (MPNNs) and structural features (SFs) are cornerstones for the link prediction task. However, as a common and intuitive mode of understanding, the potential of visual perception has been overlooked in the MPNN community. For the first time, we equip MPNNs with vision structural awareness by proposing an effective framework called Graph Vision Network (GVN), along with a more efficient variant (E-GVN). Extensive empirical results demonstrate that with the proposed frameworks, GVN consistently benefits from the vision enhancement across seven link prediction datasets, including challenging large-scale graphs. Such improvements are compatible with existing state-of-the-art (SOTA) methods and GVNs achieve new SOTA results, thereby underscoring a promising novel direction for link prediction.
>
---
#### [replaced 055] Visual Text Processing: A Comprehensive Review and Unified Evaluation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21682v2](http://arxiv.org/pdf/2504.21682v2)**

> **作者:** Yan Shu; Weichao Zeng; Fangmin Zhao; Zeyu Chen; Zhenhang Li; Xiaomeng Yang; Yu Zhou; Paolo Rota; Xiang Bai; Lianwen Jin; Xu-Cheng Yin; Nicu Sebe
>
> **摘要:** Visual text is a crucial component in both document and scene images, conveying rich semantic information and attracting significant attention in the computer vision community. Beyond traditional tasks such as text detection and recognition, visual text processing has witnessed rapid advancements driven by the emergence of foundation models, including text image reconstruction and text image manipulation. Despite significant progress, challenges remain due to the unique properties that differentiate text from general objects. Effectively capturing and leveraging these distinct textual characteristics is essential for developing robust visual text processing models. In this survey, we present a comprehensive, multi-perspective analysis of recent advancements in visual text processing, focusing on two key questions: (1) What textual features are most suitable for different visual text processing tasks? (2) How can these distinctive text features be effectively incorporated into processing frameworks? Furthermore, we introduce VTPBench, a new benchmark that encompasses a broad range of visual text processing datasets. Leveraging the advanced visual quality assessment capabilities of multimodal large language models (MLLMs), we propose VTPScore, a novel evaluation metric designed to ensure fair and reliable evaluation. Our empirical study with more than 20 specific models reveals substantial room for improvement in the current techniques. Our aim is to establish this work as a fundamental resource that fosters future exploration and innovation in the dynamic field of visual text processing. The relevant repository is available at https://github.com/shuyansy/Visual-Text-Processing-survey.
>
---
#### [replaced 056] Normalizing Flows are Capable Generative Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.06329v3](http://arxiv.org/pdf/2412.06329v3)**

> **作者:** Shuangfei Zhai; Ruixiang Zhang; Preetum Nakkiran; David Berthelot; Jiatao Gu; Huangjie Zheng; Tianrong Chen; Miguel Angel Bautista; Navdeep Jaitly; Josh Susskind
>
> **备注:** ICML 2025
>
> **摘要:** Normalizing Flows (NFs) are likelihood-based models for continuous inputs. They have demonstrated promising results on both density estimation and generative modeling tasks, but have received relatively little attention in recent years. In this work, we demonstrate that NFs are more powerful than previously believed. We present TarFlow: a simple and scalable architecture that enables highly performant NF models. TarFlow can be thought of as a Transformer-based variant of Masked Autoregressive Flows (MAFs): it consists of a stack of autoregressive Transformer blocks on image patches, alternating the autoregression direction between layers. TarFlow is straightforward to train end-to-end, and capable of directly modeling and generating pixels. We also propose three key techniques to improve sample quality: Gaussian noise augmentation during training, a post training denoising procedure, and an effective guidance method for both class-conditional and unconditional settings. Putting these together, TarFlow sets new state-of-the-art results on likelihood estimation for images, beating the previous best methods by a large margin, and generates samples with quality and diversity comparable to diffusion models, for the first time with a stand-alone NF model. We make our code available at https://github.com/apple/ml-tarflow.
>
---
#### [replaced 057] diffDemorph: Extending Reference-Free Demorphing to Unseen Faces
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14527v3](http://arxiv.org/pdf/2505.14527v3)**

> **作者:** Nitish Shukla; Arun Ross
>
> **摘要:** A face morph is created by combining two face images corresponding to two identities to produce a composite that successfully matches both the constituent identities. Reference-free (RF) demorphing reverses this process using only the morph image, without the need for additional reference images. Previous RF demorphing methods are overly constrained, as they rely on assumptions about the distributions of training and testing morphs such as the morphing technique used (e.g., landmark-based) and face image style (e.g., passport photos). In this paper, we introduce a novel diffusion-based approach, referred to as diffDeMorph, that effectively disentangles component images from a composite morph image with high visual fidelity. Our method is the first to generalize across morph techniques and face styles, beating the current state of the art by $\geq 59.46\%$ under a common training protocol across all datasets tested. We train our method on morphs created using synthetically generated face images and test on real morphs, thereby enhancing the practicality of the technique. Experiments on six datasets and two face matchers establish the utility and efficacy of our method.
>
---
#### [replaced 058] GroMo: Plant Growth Modeling with Multiview Images
- **分类: cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.06608v2](http://arxiv.org/pdf/2503.06608v2)**

> **作者:** Ruchi Bhatt; Shreya Bansal; Amanpreet Chander; Rupinder Kaur; Malya Singh; Mohan Kankanhalli; Abdulmotaleb El Saddik; Mukesh Kumar Saini
>
> **备注:** 7 pages, 5 Figures, 3 Tables
>
> **摘要:** Understanding plant growth dynamics is essential for applications in agriculture and plant phenotyping. We present the Growth Modelling (GroMo) challenge, which is designed for two primary tasks: (1) plant age prediction and (2) leaf count estimation, both essential for crop monitoring and precision agriculture. For this challenge, we introduce GroMo25, a dataset with images of four crops: radish, okra, wheat, and mustard. Each crop consists of multiple plants (p1, p2, ..., pn) captured over different days (d1, d2, ..., dm) and categorized into five levels (L1, L2, L3, L4, L5). Each plant is captured from 24 different angles with a 15-degree gap between images. Participants are required to perform both tasks for all four crops with these multiview images. We proposed a Multiview Vision Transformer (MVVT) model for the GroMo challenge and evaluated the crop-wise performance on GroMo25. MVVT reports an average MAE of 7.74 for age prediction and an MAE of 5.52 for leaf count. The GroMo Challenge aims to advance plant phenotyping research by encouraging innovative solutions for tracking and predicting plant growth. The GitHub repository is publicly available at https://github.com/mriglab/GroMo-Plant-Growth-Modeling-with-Multiview-Images.
>
---
#### [replaced 059] Images Speak Louder than Words: Understanding and Mitigating Bias in Vision-Language Model from a Causal Mediation Perspective
- **分类: cs.AI; cs.CL; cs.CV; I.2.7**

- **链接: [http://arxiv.org/pdf/2407.02814v3](http://arxiv.org/pdf/2407.02814v3)**

> **作者:** Zhaotian Weng; Zijun Gao; Jerone Andrews; Jieyu Zhao
>
> **摘要:** Vision-language models (VLMs) pre-trained on extensive datasets can inadvertently learn biases by correlating gender information with specific objects or scenarios. Current methods, which focus on modifying inputs and monitoring changes in the model's output probability scores, often struggle to comprehensively understand bias from the perspective of model components. We propose a framework that incorporates causal mediation analysis to measure and map the pathways of bias generation and propagation within VLMs. This approach allows us to identify the direct effects of interventions on model bias and the indirect effects of interventions on bias mediated through different model components. Our results show that image features are the primary contributors to bias, with significantly higher impacts than text features, specifically accounting for 32.57% and 12.63% of the bias in the MSCOCO and PASCAL-SENTENCE datasets, respectively. Notably, the image encoder's contribution surpasses that of the text encoder and the deep fusion encoder. Further experimentation confirms that contributions from both language and vision modalities are aligned and non-conflicting. Consequently, focusing on blurring gender representations within the image encoder, which contributes most to the model bias, reduces bias efficiently by 22.03% and 9.04% in the MSCOCO and PASCAL-SENTENCE datasets, respectively, with minimal performance loss or increased computational demands.
>
---
#### [replaced 060] LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.05113v3](http://arxiv.org/pdf/2406.05113v3)**

> **作者:** Lukas Helff; Felix Friedrich; Manuel Brack; Kristian Kersting; Patrick Schramowski
>
> **备注:** In Proceedings of the 42st International Conference on Machine Learning (ICML 2025), Project page at https://ml-research.github.io/human-centered-genai/projects/llavaguard/index.html
>
> **摘要:** This paper introduces LlavaGuard, a suite of VLM-based vision safeguards that address the critical need for reliable guardrails in the era of large-scale data and models. To this end, we establish a novel open framework, describing a customizable safety taxonomy, data preprocessing, augmentation, and training setup. For teaching a VLM safeguard on safety, we further create a multimodal safety dataset with high-quality human expert annotations, where each image is labeled with a safety rating, category, and rationale. We also employ advanced augmentations to support context-specific assessments. The resulting LlavaGuard models, ranging from 0.5B to 7B, serve as a versatile tool for evaluating the safety compliance of visual content against flexible policies. In comprehensive experiments, LlavaGuard outperforms both state-of-the-art safeguards and VLMs in accuracy and in flexibly handling different policies. Additionally, we demonstrate LlavaGuard's performance in two real-world applications: large-scale dataset annotation and moderation of text-to-image models. We make our entire framework, including the dataset, model weights, and training code.
>
---
#### [replaced 061] Illusion3D: 3D Multiview Illusion with 2D Diffusion Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09625v2](http://arxiv.org/pdf/2412.09625v2)**

> **作者:** Yue Feng; Vaibhav Sanjay; Spencer Lutz; Badour AlBahar; Songwei Ge; Jia-Bin Huang
>
> **备注:** Project page: https://3d-multiview-illusion.github.io/
>
> **摘要:** Automatically generating multiview illusions is a compelling challenge, where a single piece of visual content offers distinct interpretations from different viewing perspectives. Traditional methods, such as shadow art and wire art, create interesting 3D illusions but are limited to simple visual outputs (i.e., figure-ground or line drawing), restricting their artistic expressiveness and practical versatility. Recent diffusion-based illusion generation methods can generate more intricate designs but are confined to 2D images. In this work, we present a simple yet effective approach for creating 3D multiview illusions based on user-provided text prompts or images. Our method leverages a pre-trained text-to-image diffusion model to optimize the textures and geometry of neural 3D representations through differentiable rendering. When viewed from multiple angles, this produces different interpretations. We develop several techniques to improve the quality of the generated 3D multiview illusions. We demonstrate the effectiveness of our approach through extensive experiments and showcase illusion generation with diverse 3D forms.
>
---
#### [replaced 062] UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control
- **分类: cs.CV; cs.AI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2502.05749v5](http://arxiv.org/pdf/2502.05749v5)**

> **作者:** Kaizhen Zhu; Mokai Pan; Yuexin Ma; Yanwei Fu; Jingyi Yu; Jingya Wang; Ye Shi
>
> **摘要:** Recent advances in diffusion bridge models leverage Doob's $h$-transform to establish fixed endpoints between distributions, demonstrating promising results in image translation and restoration tasks. However, these approaches frequently produce blurred or excessively smoothed image details and lack a comprehensive theoretical foundation to explain these shortcomings. To address these limitations, we propose UniDB, a unified framework for diffusion bridges based on Stochastic Optimal Control (SOC). UniDB formulates the problem through an SOC-based optimization and derives a closed-form solution for the optimal controller, thereby unifying and generalizing existing diffusion bridge models. We demonstrate that existing diffusion bridges employing Doob's $h$-transform constitute a special case of our framework, emerging when the terminal penalty coefficient in the SOC cost function tends to infinity. By incorporating a tunable terminal penalty coefficient, UniDB achieves an optimal balance between control costs and terminal penalties, substantially improving detail preservation and output quality. Notably, UniDB seamlessly integrates with existing diffusion bridge models, requiring only minimal code modifications. Extensive experiments across diverse image restoration tasks validate the superiority and adaptability of the proposed framework. Our code is available at https://github.com/UniDB-SOC/UniDB/.
>
---
#### [replaced 063] Imitating Radiological Scrolling: A Global-Local Attention Model for 3D Chest CT Volumes Multi-Label Anomaly Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20652v4](http://arxiv.org/pdf/2503.20652v4)**

> **作者:** Theo Di Piazza; Carole Lazarus; Olivier Nempont; Loic Boussel
>
> **备注:** 13 pages, 4 figures. Accepted for publication at MIDL 2025
>
> **摘要:** The rapid increase in the number of Computed Tomography (CT) scan examinations has created an urgent need for automated tools, such as organ segmentation, anomaly classification, and report generation, to assist radiologists with their growing workload. Multi-label classification of Three-Dimensional (3D) CT scans is a challenging task due to the volumetric nature of the data and the variety of anomalies to be detected. Existing deep learning methods based on Convolutional Neural Networks (CNNs) struggle to capture long-range dependencies effectively, while Vision Transformers require extensive pre-training, posing challenges for practical use. Additionally, these existing methods do not explicitly model the radiologist's navigational behavior while scrolling through CT scan slices, which requires both global context understanding and local detail awareness. In this study, we present CT-Scroll, a novel global-local attention model specifically designed to emulate the scrolling behavior of radiologists during the analysis of 3D CT scans. Our approach is evaluated on two public datasets, demonstrating its efficacy through comprehensive experiments and an ablation study that highlights the contribution of each model component.
>
---
#### [replaced 064] A Comprehensive Survey on Concept Erasure in Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14896v2](http://arxiv.org/pdf/2502.14896v2)**

> **作者:** Changhoon Kim; Yanjun Qi
>
> **摘要:** Text-to-Image (T2I) models have made remarkable progress in generating high-quality, diverse visual content from natural language prompts. However, their ability to reproduce copyrighted styles, sensitive imagery, and harmful content raises significant ethical and legal concerns. Concept erasure offers a proactive alternative to external filtering by modifying T2I models to prevent the generation of undesired content. In this survey, we provide a structured overview of concept erasure, categorizing existing methods based on their optimization strategies and the architectural components they modify. We categorize concept erasure methods into fine-tuning for parameter updates, closed-form solutions for efficient edits, and inference-time interventions for content restriction without weight modification. Additionally, we explore adversarial attacks that bypass erasure techniques and discuss emerging defenses. To support further research, we consolidate key datasets, evaluation metrics, and benchmarks for assessing erasure effectiveness and model robustness. This survey serves as a comprehensive resource, offering insights into the evolving landscape of concept erasure, its challenges, and future directions.
>
---
#### [replaced 065] Sketched Equivariant Imaging Regularization and Deep Internal Learning for Inverse Problems
- **分类: eess.IV; cs.CV; cs.LG; math.OC**

- **链接: [http://arxiv.org/pdf/2411.05771v4](http://arxiv.org/pdf/2411.05771v4)**

> **作者:** Guixian Xu; Jinglai Li; Junqi Tang
>
> **备注:** 22 pages
>
> **摘要:** Equivariant Imaging (EI) regularization has become the de-facto technique for unsupervised training of deep imaging networks, without any need of ground-truth data. Observing that the EI-based unsupervised training paradigm currently has significant computational redundancy leading to inefficiency in high-dimensional applications, we propose a sketched EI regularization which leverages the randomized sketching techniques for acceleration. We apply our sketched EI regularization to develop an accelerated deep internal learning framework, which can be efficiently applied for test-time network adaptation. Additionally, for network adaptation tasks, we propose a parameter-efficient approach to accelerate both EI and Sketched-EI via optimizing only the normalization layers. Our numerical study on X-ray CT and multicoil magnetic resonance image reconstruction tasks demonstrate that our approach can achieve significant computational acceleration over standard EI counterpart in single-input setting and network adaptation at test time.
>
---
#### [replaced 066] Bridging Annotation Gaps: Transferring Labels to Align Object Detection Datasets
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04737v2](http://arxiv.org/pdf/2506.04737v2)**

> **作者:** Mikhail Kennerley; Angelica Aviles-Rivero; Carola-Bibiane Schönlieb; Robby T. Tan
>
> **摘要:** Combining multiple object detection datasets offers a path to improved generalisation but is hindered by inconsistencies in class semantics and bounding box annotations. Some methods to address this assume shared label taxonomies and address only spatial inconsistencies; others require manual relabelling, or produce a unified label space, which may be unsuitable when a fixed target label space is required. We propose Label-Aligned Transfer (LAT), a label transfer framework that systematically projects annotations from diverse source datasets into the label space of a target dataset. LAT begins by training dataset-specific detectors to generate pseudo-labels, which are then combined with ground-truth annotations via a Privileged Proposal Generator (PPG) that replaces the region proposal network in two-stage detectors. To further refine region features, a Semantic Feature Fusion (SFF) module injects class-aware context and features from overlapping proposals using a confidence-weighted attention mechanism. This pipeline preserves dataset-specific annotation granularity while enabling many-to-one label space transfer across heterogeneous datasets, resulting in a semantically and spatially aligned representation suitable for training a downstream detector. LAT thus jointly addresses both class-level misalignments and bounding box inconsistencies without relying on shared label spaces or manual annotations. Across multiple benchmarks, LAT demonstrates consistent improvements in target-domain detection performance, achieving gains of up to +4.8AP over semi-supervised baselines.
>
---
#### [replaced 067] Birth and Death of a Rose
- **分类: cs.CV; cs.GR; I.2.10**

- **链接: [http://arxiv.org/pdf/2412.05278v2](http://arxiv.org/pdf/2412.05278v2)**

> **作者:** Chen Geng; Yunzhi Zhang; Shangzhe Wu; Jiajun Wu
>
> **备注:** CVPR 2025 Oral. Project website: https://chen-geng.com/rose4d
>
> **摘要:** We study the problem of generating temporal object intrinsics -- temporally evolving sequences of object geometry, reflectance, and texture, such as a blooming rose -- from pre-trained 2D foundation models. Unlike conventional 3D modeling and animation techniques that require extensive manual effort and expertise, we introduce a method that generates such assets with signals distilled from pre-trained 2D diffusion models. To ensure the temporal consistency of object intrinsics, we propose Neural Templates for temporal-state-guided distillation, derived automatically from image features from self-supervised learning. Our method can generate high-quality temporal object intrinsics for several natural phenomena and enable the sampling and controllable rendering of these dynamic objects from any viewpoint, under any environmental lighting conditions, at any time of their lifespan. Project website: https://chen-geng.com/rose4d
>
---
#### [replaced 068] Modality-Fair Preference Optimization for Trustworthy MLLM Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.15334v2](http://arxiv.org/pdf/2410.15334v2)**

> **作者:** Songtao Jiang; Yan Zhang; Ruizhe Chen; Tianxiang Hu; Yeying Jin; Qinglin He; Yang Feng; Jian Wu; Zuozhu Liu
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable success across various tasks. However, separate training of visual and textual encoders often results in a misalignment of the modality. Such misalignment may lead models to generate content that is absent from the input image, a phenomenon referred to as hallucination. These inaccuracies severely undermine the trustworthiness of MLLMs in real-world applications. Despite attempts to optimize text preferences to mitigate this issue, our initial investigation indicates that the trustworthiness of MLLMs remains inadequate. Specifically, these models tend to provide preferred answers even when the input image is heavily distorted. Analysis of visual token attention also indicates that the model focuses primarily on the surrounding context rather than the key object referenced in the question. These findings highlight a misalignment between the modalities, where answers inadequately leverage input images. Motivated by our findings, we propose Modality-Fair Preference Optimization (MFPO), which comprises three components: the construction of a multimodal preference dataset in which dispreferred images differ from originals solely in key regions; an image reward loss function encouraging the model to generate answers better aligned with the input images; and an easy-to-hard iterative alignment strategy to stabilize joint modality training. Extensive experiments on three trustworthiness benchmarks demonstrate that MFPO significantly enhances the trustworthiness of MLLMs. In particular, it enables the 7B models to attain trustworthiness levels on par with, or even surpass, those of the 13B, 34B, and larger models.
>
---
#### [replaced 069] DPCore: Dynamic Prompt Coreset for Continual Test-Time Adaptation
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.10737v4](http://arxiv.org/pdf/2406.10737v4)**

> **作者:** Yunbei Zhang; Akshay Mehra; Shuaicheng Niu; Jihun Hamm
>
> **备注:** ICML2025
>
> **摘要:** Continual Test-Time Adaptation (CTTA) seeks to adapt source pre-trained models to continually changing, unseen target domains. While existing CTTA methods assume structured domain changes with uniform durations, real-world environments often exhibit dynamic patterns where domains recur with varying frequencies and durations. Current approaches, which adapt the same parameters across different domains, struggle in such dynamic conditions-they face convergence issues with brief domain exposures, risk forgetting previously learned knowledge, or misapplying it to irrelevant domains. To remedy this, we propose DPCore, a method designed for robust performance across diverse domain change patterns while ensuring computational efficiency. DPCore integrates three key components: Visual Prompt Adaptation for efficient domain alignment, a Prompt Coreset for knowledge preservation, and a Dynamic Update mechanism that intelligently adjusts existing prompts for similar domains while creating new ones for substantially different domains. Extensive experiments on four benchmarks demonstrate that DPCore consistently outperforms various CTTA methods, achieving state-of-the-art performance in both structured and dynamic settings while reducing trainable parameters by 99% and computation time by 64% compared to previous approaches.
>
---
#### [replaced 070] HilbertMamba: Local-Global Reciprocal Network for Uterine Fibroid Segmentation in Ultrasound Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.05703v2](http://arxiv.org/pdf/2407.05703v2)**

> **作者:** Huihui Xu; Yijun Yang; Angelica I Aviles-Rivero; Guang Yang; Jing Qin; Lei Zhu
>
> **备注:** MICCAI2024 Early Accept
>
> **摘要:** Regular screening and early discovery of uterine fibroid are crucial for preventing potential malignant transformations and ensuring timely, life-saving interventions. To this end, we collect and annotate the first ultrasound video dataset with 100 videos for uterine fibroid segmentation (UFUV). We also present Local-Global Reciprocal Network (LGRNet) to efficiently and effectively propagate the long-term temporal context which is crucial to help distinguish between uninformative noisy surrounding tissues and target lesion regions. Specifically, the Cyclic Neighborhood Propagation (CNP) is introduced to propagate the inter-frame local temporal context in a cyclic manner. Moreover, to aggregate global temporal context, we first condense each frame into a set of frame bottleneck queries and devise Hilbert Selective Scan (HilbertSS) to both efficiently path connect each frame and preserve the locality bias. A distribute layer is then utilized to disseminate back the global context for reciprocal refinement. Extensive experiments on UFUV and three public Video Polyp Segmentation (VPS) datasets demonstrate consistent improvements compared to state-of-the-art segmentation methods, indicating the effectiveness and versatility of LGRNet. Code, checkpoints, and dataset are available at https://github.com/bio-mlhui/LGRNet
>
---
#### [replaced 071] RoPETR: Improving Temporal Camera-Only 3D Detection by Integrating Enhanced Rotary Position Embedding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12643v3](http://arxiv.org/pdf/2504.12643v3)**

> **作者:** Hang Ji; Tao Ni; Xufeng Huang; Zhan Shi; Tao Luo; Xin Zhan; Junbo Chen
>
> **摘要:** This technical report introduces a targeted improvement to the StreamPETR framework, specifically aimed at enhancing velocity estimation, a critical factor influencing the overall NuScenes Detection Score. While StreamPETR exhibits strong 3D bounding box detection performance as reflected by its high mean Average Precision our analysis identified velocity estimation as a substantial bottleneck when evaluated on the NuScenes dataset. To overcome this limitation, we propose a customized positional embedding strategy tailored to enhance temporal modeling capabilities. Experimental evaluations conducted on the NuScenes test set demonstrate that our improved approach achieves a state-of-the-art NDS of 70.86% using the ViT-L backbone, setting a new benchmark for camera-only 3D object detection.
>
---
#### [replaced 072] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02698v2](http://arxiv.org/pdf/2506.02698v2)**

> **作者:** Yunhong Lu; Qichao Wang; Hengyuan Cao; Xiaoyin Xu; Min Zhang
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Direct Preference Optimization (DPO) aligns text-to-image (T2I) generation models with human preferences using pairwise preference data. Although substantial resources are expended in collecting and labeling datasets, a critical aspect is often neglected: \textit{preferences vary across individuals and should be represented with more granularity.} To address this, we propose SmPO-Diffusion, a novel method for modeling preference distributions to improve the DPO objective, along with a numerical upper bound estimation for the diffusion optimization objective. First, we introduce a smoothed preference distribution to replace the original binary distribution. We employ a reward model to simulate human preferences and apply preference likelihood averaging to improve the DPO loss, such that the loss function approaches zero when preferences are similar. Furthermore, we utilize an inversion technique to simulate the trajectory preference distribution of the diffusion model, enabling more accurate alignment with the optimization objective. Our approach effectively mitigates issues of excessive optimization and objective misalignment present in existing methods through straightforward modifications. Our SmPO-Diffusion achieves state-of-the-art performance in preference evaluation, outperforming baselines across metrics with lower training costs. The project page is https://jaydenlyh.github.io/SmPO-project-page/.
>
---
#### [replaced 073] FPSAttention: Training-Aware FP8 and Sparsity Co-Design for Fast Video Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04648v2](http://arxiv.org/pdf/2506.04648v2)**

> **作者:** Akide Liu; Zeyu Zhang; Zhexin Li; Xuehai Bai; Yizeng Han; Jiasheng Tang; Yuanjie Xing; Jichao Wu; Mingyang Yang; Weihua Chen; Jiahao He; Yuanyu He; Fan Wang; Gholamreza Haffari; Bohan Zhuang
>
> **备注:** Project Page: https://fps.ziplab.co
>
> **摘要:** Diffusion generative models have become the standard for producing high-quality, coherent video content, yet their slow inference speeds and high computational demands hinder practical deployment. Although both quantization and sparsity can independently accelerate inference while maintaining generation quality, naively combining these techniques in existing training-free approaches leads to significant performance degradation due to the lack of joint optimization. We introduce FPSAttention, a novel training-aware co-design of FP8 quantization and sparsity for video generation, with a focus on the 3D bi-directional attention mechanism. Our approach features three key innovations: 1) A unified 3D tile-wise granularity that simultaneously supports both quantization and sparsity; 2) A denoising step-aware strategy that adapts to the noise schedule, addressing the strong correlation between quantization/sparsity errors and denoising steps; 3) A native, hardware-friendly kernel that leverages FlashAttention and is implemented with optimized Hopper architecture features for highly efficient execution. Trained on Wan2.1's 1.3B and 14B models and evaluated on the VBench benchmark, FPSAttention achieves a 7.09x kernel speedup for attention operations and a 4.96x end-to-end speedup for video generation compared to the BF16 baseline at 720p resolution-without sacrificing generation quality.
>
---
#### [replaced 074] A novel non-convex minimax $p$-th order concave penalty function approach to low-rank tensor completion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.19979v2](http://arxiv.org/pdf/2502.19979v2)**

> **作者:** Hongbing Zhang; Bing Zheng
>
> **备注:** 30 pages,14 figures
>
> **摘要:** The low-rank tensor completion (LRTC) problem aims to reconstruct a tensor from partial sample information, which has attracted significant interest in a wide range of practical applications such as image processing and computer vision. Among the various techniques employed for the LRTC problem, non-convex relaxation methods have been widely studied for their effectiveness in handling tensor singular values, which are crucial for accurate tensor recovery. While the minimax concave penalty (MCP) non-convex relaxation method has achieved promising results in tackling the LRTC problem and gained widely adopted, it exhibits a notable limitation: insufficient penalty on small singular values during the singular value handling process, resulting in inefficient tensor recovery. To address this issue and enhance recovery performance, a novel minimax $p$-th order concave penalty (MPCP) function is proposed. Based on this novel function, a tensor $p$-th order $\tau$ norm is proposed as a non-convex relaxation for tensor rank approximation, thereby establishing an MPCP-based LRTC model. Furthermore, theoretical convergence guarantees are rigorously established for the proposed method. Extensive numerical experiments conducted on multiple real datasets demonstrate that the proposed method outperforms the state-of-the-art methods in both visual quality and quantitative metrics.
>
---
#### [replaced 075] Multivariate Temporal Regression at Scale: A Three-Pillar Framework Combining ML, XAI, and NLP
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02151v2](http://arxiv.org/pdf/2504.02151v2)**

> **作者:** Jiztom Kavalakkatt Francis; Matthew J Darr
>
> **备注:** 7 pages
>
> **摘要:** This paper introduces a novel framework that accelerates the discovery of actionable relationships in high-dimensional temporal data by integrating machine learning (ML), explainable AI (XAI), and natural language processing (NLP) to enhance data quality and streamline workflows. Traditional methods often fail to recognize complex temporal relationships, leading to noisy, redundant, or biased datasets. Our approach combines ML-driven pruning to identify and mitigate low-quality samples, XAI-based interpretability to validate critical feature interactions, and NLP for future contextual validation, reducing the time required to uncover actionable insights by 40-60%. Evaluated on real-world agricultural and synthetic datasets, the framework significantly improves performance metrics (e.g., MSE, R2, MAE) and computational efficiency, with hardware-agnostic scalability across diverse platforms. While long-term real-world impacts (e.g., cost savings, sustainability gains) are pending, this methodology provides an immediate pathway to accelerate data-centric AI in dynamic domains like agriculture and energy, enabling faster iteration cycles for domain experts.
>
---
#### [replaced 076] SALVE: A 3D Reconstruction Benchmark of Wounds from Consumer-grade Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.19652v3](http://arxiv.org/pdf/2407.19652v3)**

> **作者:** Remi Chierchia; Leo Lebrat; David Ahmedt-Aristizabal; Olivier Salvado; Clinton Fookes; Rodrigo Santa Cruz
>
> **摘要:** Managing chronic wounds is a global challenge that can be alleviated by the adoption of automatic systems for clinical wound assessment from consumer-grade videos. While 2D image analysis approaches are insufficient for handling the 3D features of wounds, existing approaches utilizing 3D reconstruction methods have not been thoroughly evaluated. To address this gap, this paper presents a comprehensive study on 3D wound reconstruction from consumer-grade videos. Specifically, we introduce the SALVE dataset, comprising video recordings of realistic wound phantoms captured with different cameras. Using this dataset, we assess the accuracy and precision of state-of-the-art methods for 3D reconstruction, ranging from traditional photogrammetry pipelines to advanced neural rendering approaches. In our experiments, we observe that photogrammetry approaches do not provide smooth surfaces suitable for precise clinical measurements of wounds. Neural rendering approaches show promise in addressing this issue, advancing the use of this technology in wound care practices. We encourage the readers to visit the project page: https://remichierchia.github.io/SALVE/.
>
---
#### [replaced 077] Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.02821v2](http://arxiv.org/pdf/2504.02821v2)**

> **作者:** Mateusz Pach; Shyamgopal Karthik; Quentin Bouniot; Serge Belongie; Zeynep Akata
>
> **备注:** Preprint
>
> **摘要:** Given that interpretability and steerability are crucial to AI safety, Sparse Autoencoders (SAEs) have emerged as a tool to enhance them in Large Language Models (LLMs). In this work, we extend the application of SAEs to Vision-Language Models (VLMs), such as CLIP, and introduce a comprehensive framework for evaluating monosemanticity at the neuron-level in vision representations. To ensure that our evaluation aligns with human perception, we propose a benchmark derived from a large-scale user study. Our experimental results reveal that SAEs trained on VLMs significantly enhance the monosemanticity of individual neurons, with sparsity and wide latents being the most influential factors. Notably, we demonstrate that applying SAE interventions on CLIP's vision encoder directly steers multimodal LLM outputs (e.g., LLaVA), without any modifications to the underlying model. These findings emphasize the practicality and efficacy of SAEs as an unsupervised tool for enhancing both interpretability and control of VLMs. Code is available at https://github.com/ExplainableML/sae-for-vlm.
>
---
#### [replaced 078] Pseudo-labelling meets Label Smoothing for Noisy Partial Label Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.04835v3](http://arxiv.org/pdf/2402.04835v3)**

> **作者:** Darshana Saravanan; Naresh Manwani; Vineet Gandhi
>
> **备注:** Best Paper Award at The 12th Workshop on Fine-Grained Visual Categorization (CVPRW 2025)
>
> **摘要:** We motivate weakly supervised learning as an effective learning paradigm for problems where curating perfectly annotated datasets is expensive and may require domain expertise such as fine-grained classification. We focus on Partial Label Learning (PLL), a weakly-supervised learning paradigm where each training instance is paired with a set of candidate labels (partial label), one of which is the true label. Noisy PLL (NPLL) relaxes this constraint by allowing some partial labels to not contain the true label, enhancing the practicality of the problem. Our work centres on NPLL and presents a framework that initially assigns pseudo-labels to images by exploiting the noisy partial labels through a weighted nearest neighbour algorithm. These pseudo-label and image pairs are then used to train a deep neural network classifier with label smoothing. The classifier's features and predictions are subsequently employed to refine and enhance the accuracy of pseudo-labels. We perform thorough experiments on seven datasets and compare against nine NPLL and PLL methods. We achieve state-of-the-art results in all studied settings from the prior literature, obtaining substantial gains in the simulated fine-grained benchmarks. Further, we show the promising generalisation capability of our framework in realistic, fine-grained, crowd-sourced datasets.
>
---
