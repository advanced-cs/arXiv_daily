# 计算机视觉 cs.CV

- **最新发布 69 篇**

- **更新 55 篇**

## 最新发布

#### [new 001] WinT3R: Window-Based Streaming Reconstruction with Camera Token Pool
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出WinT3R模型，解决在线三维重建中重建质量与实时性的矛盾。通过滑动窗口机制和全局相机令牌池，提升几何预测精度与姿态估计可靠性，实现高效高质量的实时重建。**

- **链接: [http://arxiv.org/pdf/2509.05296v1](http://arxiv.org/pdf/2509.05296v1)**

> **作者:** Zizun Li; Jianjun Zhou; Yifan Wang; Haoyu Guo; Wenzheng Chang; Yang Zhou; Haoyi Zhu; Junyi Chen; Chunhua Shen; Tong He
>
> **摘要:** We present WinT3R, a feed-forward reconstruction model capable of online prediction of precise camera poses and high-quality point maps. Previous methods suffer from a trade-off between reconstruction quality and real-time performance. To address this, we first introduce a sliding window mechanism that ensures sufficient information exchange among frames within the window, thereby improving the quality of geometric predictions without large computation. In addition, we leverage a compact representation of cameras and maintain a global camera token pool, which enhances the reliability of camera pose estimation without sacrificing efficiency. These designs enable WinT3R to achieve state-of-the-art performance in terms of online reconstruction quality, camera pose estimation, and reconstruction speed, as validated by extensive experiments on diverse datasets. Code and model are publicly available at https://github.com/LiZizun/WinT3R.
>
---
#### [new 002] TemporalFlowViz: Parameter-Aware Visual Analytics for Interpreting Scramjet Combustion Evolution
- **分类: cs.CV**

- **简介: 该论文提出TemporalFlowViz系统，解决scramjet燃烧模拟数据高维、难分析的问题。通过参数感知的视觉分析，结合Vision Transformer提取特征、聚类与轨迹追踪，生成可解释的自然语言总结，支持多视图探索与专家交互，提升燃烧动态理解效率。**

- **链接: [http://arxiv.org/pdf/2509.04834v1](http://arxiv.org/pdf/2509.04834v1)**

> **作者:** Yifei Jia; Shiyu Cheng; Yu Dong; Guan Li; Dong Tian; Ruixiao Peng; Xuyi Lu; Yu Wang; Wei Yao; Guihua Shan
>
> **摘要:** Understanding the complex combustion dynamics within scramjet engines is critical for advancing high-speed propulsion technologies. However, the large scale and high dimensionality of simulation-generated temporal flow field data present significant challenges for visual interpretation, feature differentiation, and cross-case comparison. In this paper, we present TemporalFlowViz, a parameter-aware visual analytics workflow and system designed to support expert-driven clustering, visualization, and interpretation of temporal flow fields from scramjet combustion simulations. Our approach leverages hundreds of simulated combustion cases with varying initial conditions, each producing time-sequenced flow field images. We use pretrained Vision Transformers to extract high-dimensional embeddings from these frames, apply dimensionality reduction and density-based clustering to uncover latent combustion modes, and construct temporal trajectories in the embedding space to track the evolution of each simulation over time. To bridge the gap between latent representations and expert reasoning, domain specialists annotate representative cluster centroids with descriptive labels. These annotations are used as contextual prompts for a vision-language model, which generates natural-language summaries for individual frames and full simulation cases. The system also supports parameter-based filtering, similarity-based case retrieval, and coordinated multi-view exploration to facilitate in-depth analysis. We demonstrate the effectiveness of TemporalFlowViz through two expert-informed case studies and expert feedback, showing TemporalFlowViz enhances hypothesis generation, supports interpretable pattern discovery, and enhances knowledge discovery in large-scale scramjet combustion analysis.
>
---
#### [new 003] Systematic Review and Meta-analysis of AI-driven MRI Motion Artifact Detection and Correction
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文通过系统综述和元分析，评估AI驱动的MRI运动伪影检测与纠正方法，总结深度学习生成模型的潜力与挑战，提出标准化数据集和改进技术的建议。**

- **链接: [http://arxiv.org/pdf/2509.05071v1](http://arxiv.org/pdf/2509.05071v1)**

> **作者:** Mojtaba Safari; Zach Eidex; Richard L. J. Qiu; Matthew Goette; Tonghe Wang; Xiaofeng Yang
>
> **摘要:** Background: To systematically review and perform a meta-analysis of artificial intelligence (AI)-driven methods for detecting and correcting magnetic resonance imaging (MRI) motion artifacts, assessing current developments, effectiveness, challenges, and future research directions. Methods: A comprehensive systematic review and meta-analysis were conducted, focusing on deep learning (DL) approaches, particularly generative models, for the detection and correction of MRI motion artifacts. Quantitative data were extracted regarding utilized datasets, DL architectures, and performance metrics. Results: DL, particularly generative models, show promise for reducing motion artifacts and improving image quality; however, limited generalizability, reliance on paired training data, and risk of visual distortions remain key challenges that motivate standardized datasets and reporting. Conclusions: AI-driven methods, particularly DL generative models, show significant potential for improving MRI image quality by effectively addressing motion artifacts. However, critical challenges must be addressed, including the need for comprehensive public datasets, standardized reporting protocols for artifact levels, and more advanced, adaptable DL techniques to reduce reliance on extensive paired datasets. Addressing these aspects could substantially enhance MRI diagnostic accuracy, reduce healthcare costs, and improve patient care outcomes.
>
---
#### [new 004] PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting
- **分类: cs.CV**

- **简介: 该论文提出PromptEnhancer框架，通过链式思维提示重写提升文本到图像模型的生成质量，解决复杂提示下属性绑定、否定等导致的语义偏差问题。方法无需修改模型权重，利用强化学习和专门奖励模型优化提示表达。**

- **链接: [http://arxiv.org/pdf/2509.04545v1](http://arxiv.org/pdf/2509.04545v1)**

> **作者:** Linqing Wang; Ximing Xing; Yiji Cheng; Zhiyuan Zhao; Jiale Tao; Qixun Wang; Ruihuang Li; Xin Li; Mingrui Wu; Xinchi Deng; Chunyu Wang; Qinglin Lu
>
> **备注:** technical report
>
> **摘要:** Recent advancements in text-to-image (T2I) diffusion models have demonstrated remarkable capabilities in generating high-fidelity images. However, these models often struggle to faithfully render complex user prompts, particularly in aspects like attribute binding, negation, and compositional relationships. This leads to a significant mismatch between user intent and the generated output. To address this challenge, we introduce PromptEnhancer, a novel and universal prompt rewriting framework that enhances any pretrained T2I model without requiring modifications to its weights. Unlike prior methods that rely on model-specific fine-tuning or implicit reward signals like image-reward scores, our framework decouples the rewriter from the generator. We achieve this by training a Chain-of-Thought (CoT) rewriter through reinforcement learning, guided by a dedicated reward model we term the AlignEvaluator. The AlignEvaluator is trained to provide explicit and fine-grained feedback based on a systematic taxonomy of 24 key points, which are derived from a comprehensive analysis of common T2I failure modes. By optimizing the CoT rewriter to maximize the reward from our AlignEvaluator, our framework learns to generate prompts that are more precisely interpreted by T2I models. Extensive experiments on the HunyuanImage 2.1 model demonstrate that PromptEnhancer significantly improves image-text alignment across a wide range of semantic and compositional challenges. Furthermore, we introduce a new, high-quality human preference benchmark to facilitate future research in this direction.
>
---
#### [new 005] Hybrid-Tower: Fine-grained Pseudo-query Interaction and Generation for Text-to-Video Retrieval
- **分类: cs.CV**

- **简介: 该论文针对文本到视频检索任务，解决现有Two-Tower框架效果差、Single-Tower框架效率低的问题，提出Hybrid-Tower框架，通过细粒度伪查询生成与交互，兼顾高效果与高效性，实验验证其性能优于基线。**

- **链接: [http://arxiv.org/pdf/2509.04773v1](http://arxiv.org/pdf/2509.04773v1)**

> **作者:** Bangxiang Lan; Ruobing Xie; Ruixiang Zhao; Xingwu Sun; Zhanhui Kang; Gang Yang; Xirong Li
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** The Text-to-Video Retrieval (T2VR) task aims to retrieve unlabeled videos by textual queries with the same semantic meanings. Recent CLIP-based approaches have explored two frameworks: Two-Tower versus Single-Tower framework, yet the former suffers from low effectiveness, while the latter suffers from low efficiency. In this study, we explore a new Hybrid-Tower framework that can hybridize the advantages of the Two-Tower and Single-Tower framework, achieving high effectiveness and efficiency simultaneously. We propose a novel hybrid method, Fine-grained Pseudo-query Interaction and Generation for T2VR, ie, PIG, which includes a new pseudo-query generator designed to generate a pseudo-query for each video. This enables the video feature and the textual features of pseudo-query to interact in a fine-grained manner, similar to the Single-Tower approaches to hold high effectiveness, even before the real textual query is received. Simultaneously, our method introduces no additional storage or computational overhead compared to the Two-Tower framework during the inference stage, thus maintaining high efficiency. Extensive experiments on five commonly used text-video retrieval benchmarks demonstrate that our method achieves a significant improvement over the baseline, with an increase of $1.6\% \sim 3.9\%$ in R@1. Furthermore, our method matches the efficiency of Two-Tower models while achieving near state-of-the-art performance, highlighting the advantages of the Hybrid-Tower framework.
>
---
#### [new 006] VCMamba: Bridging Convolutions with Multi-Directional Mamba for Efficient Visual Representation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出VCMamba，融合卷积与多方向Mamba，解决视觉任务中局部特征与全局建模的平衡问题，提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2509.04669v1](http://arxiv.org/pdf/2509.04669v1)**

> **作者:** Mustafa Munir; Alex Zhang; Radu Marculescu
>
> **备注:** Proceedings of the 2025 IEEE/CVF International Conference on Computer Vision (ICCV) Workshops
>
> **摘要:** Recent advances in Vision Transformers (ViTs) and State Space Models (SSMs) have challenged the dominance of Convolutional Neural Networks (CNNs) in computer vision. ViTs excel at capturing global context, and SSMs like Mamba offer linear complexity for long sequences, yet they do not capture fine-grained local features as effectively as CNNs. Conversely, CNNs possess strong inductive biases for local features but lack the global reasoning capabilities of transformers and Mamba. To bridge this gap, we introduce \textit{VCMamba}, a novel vision backbone that integrates the strengths of CNNs and multi-directional Mamba SSMs. VCMamba employs a convolutional stem and a hierarchical structure with convolutional blocks in its early stages to extract rich local features. These convolutional blocks are then processed by later stages incorporating multi-directional Mamba blocks designed to efficiently model long-range dependencies and global context. This hybrid design allows for superior feature representation while maintaining linear complexity with respect to image resolution. We demonstrate VCMamba's effectiveness through extensive experiments on ImageNet-1K classification and ADE20K semantic segmentation. Our VCMamba-B achieves 82.6% top-1 accuracy on ImageNet-1K, surpassing PlainMamba-L3 by 0.3% with 37% fewer parameters, and outperforming Vision GNN-B by 0.3% with 64% fewer parameters. Furthermore, VCMamba-B obtains 47.1 mIoU on ADE20K, exceeding EfficientFormer-L7 by 2.0 mIoU while utilizing 62% fewer parameters. Code is available at https://github.com/Wertyuui345/VCMamba.
>
---
#### [new 007] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases
- **分类: cs.CV**

- **简介: 该论文提出FlowSeek框架，解决光流估计中硬件资源消耗高的问题。通过结合深度基础模型与运动基参数化，设计轻量架构，在单GPU上训练，实现跨数据集高精度光流估计，超越现有SOTA方法。**

- **链接: [http://arxiv.org/pdf/2509.05297v1](http://arxiv.org/pdf/2509.05297v1)**

> **作者:** Matteo Poggi; Fabio Tosi
>
> **备注:** ICCV 2025 - Project Page: https://flowseek25.github.io/ - Code: https://github.com/mattpoggi/flowseek
>
> **摘要:** We present FlowSeek, a novel framework for optical flow requiring minimal hardware resources for training. FlowSeek marries the latest advances on the design space of optical flow networks with cutting-edge single-image depth foundation models and classical low-dimensional motion parametrization, implementing a compact, yet accurate architecture. FlowSeek is trained on a single consumer-grade GPU, a hardware budget about 8x lower compared to most recent methods, and still achieves superior cross-dataset generalization on Sintel Final and KITTI, with a relative improvement of 10 and 15% over the previous state-of-the-art SEA-RAFT, as well as on Spring and LayeredFlow datasets.
>
---
#### [new 008] Cryo-RL: automating prostate cancer cryoablation planning with reinforcement learning
- **分类: cs.CV**

- **简介: 论文提出Cryo-RL框架，利用强化学习自动化前列腺癌冷冻消融治疗计划，解决手动规划效率低、专家依赖问题。通过模拟环境训练代理优化探针位置，提升肿瘤覆盖度，超越现有方法并匹配专家表现。**

- **链接: [http://arxiv.org/pdf/2509.04886v1](http://arxiv.org/pdf/2509.04886v1)**

> **作者:** Trixia Simangan; Ahmed Nadeem Abbasi; Yipeng Hu; Shaheer U. Saeed
>
> **备注:** Accepted at MICAD (Medical Imaging and Computer-Aided Diagnosis) 2025
>
> **摘要:** Cryoablation is a minimally invasive localised treatment for prostate cancer that destroys malignant tissue during de-freezing, while sparing surrounding healthy structures. Its success depends on accurate preoperative planning of cryoprobe placements to fully cover the tumour and avoid critical anatomy. This planning is currently manual, expertise-dependent, and time-consuming, leading to variability in treatment quality and limited scalability. In this work, we introduce Cryo-RL, a reinforcement learning framework that models cryoablation planning as a Markov decision process and learns an optimal policy for cryoprobe placement. Within a simulated environment that models clinical constraints and stochastic intraoperative variability, an agent sequentially selects cryoprobe positions and ice sphere diameters. Guided by a reward function based on tumour coverage, this agent learns a cryoablation strategy that leads to optimal cryoprobe placements without the need for any manually-designed plans. Evaluated on 583 retrospective prostate cancer cases, Cryo-RL achieved over 8 percentage-point Dice improvements compared with the best automated baselines, based on geometric optimisation, and matched human expert performance while requiring substantially less planning time. These results highlight the potential of reinforcement learning to deliver clinically viable, reproducible, and efficient cryoablation plans.
>
---
#### [new 009] DisPatch: Disarming Adversarial Patches in Object Detection with Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出DISPATCH框架，利用扩散模型生成和修正图像以防御目标检测中的对抗性贴片攻击，有效提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.04597v1](http://arxiv.org/pdf/2509.04597v1)**

> **作者:** Jin Ma; Mohammed Aldeen; Christopher Salas; Feng Luo; Mashrur Chowdhury; Mert Pesé; Long Cheng
>
> **摘要:** Object detection is fundamental to various real-world applications, such as security monitoring and surveillance video analysis. Despite their advancements, state-of-theart object detectors are still vulnerable to adversarial patch attacks, which can be easily applied to real-world objects to either conceal actual items or create non-existent ones, leading to severe consequences. Given the current diversity of adversarial patch attacks and potential unknown threats, an ideal defense method should be effective, generalizable, and robust against adaptive attacks. In this work, we introduce DISPATCH, the first diffusion-based defense framework for object detection. Unlike previous works that aim to "detect and remove" adversarial patches, DISPATCH adopts a "regenerate and rectify" strategy, leveraging generative models to disarm attack effects while preserving the integrity of the input image. Specifically, we utilize the in-distribution generative power of diffusion models to regenerate the entire image, aligning it with benign data. A rectification process is then employed to identify and replace adversarial regions with their regenerated benign counterparts. DISPATCH is attack-agnostic and requires no prior knowledge of the existing patches. Extensive experiments across multiple detectors and attacks demonstrate that DISPATCH consistently outperforms state-of-the-art defenses on both hiding attacks and creating attacks, achieving the best overall mAP.5 score of 89.3% on hiding attacks, and lowering the attack success rate to 24.8% on untargeted creating attacks. Moreover, it maintains strong robustness against adaptive attacks, making it a practical and reliable defense for object detection systems.
>
---
#### [new 010] SynGen-Vision: Synthetic Data Generation for training industrial vision models
- **分类: cs.CV; cs.LG; I.4**

- **简介: 该论文提出SynGen-Vision方法，通过视觉语言模型与3D渲染生成工业磨损检测合成数据，解决真实数据稀缺问题，提升锈蚀检测性能（mAP50 0.87）。**

- **链接: [http://arxiv.org/pdf/2509.04894v1](http://arxiv.org/pdf/2509.04894v1)**

> **作者:** Alpana Dubey; Suma Mani Kuriakose; Nitish Bhardwaj
>
> **摘要:** We propose an approach to generate synthetic data to train computer vision (CV) models for industrial wear and tear detection. Wear and tear detection is an important CV problem for predictive maintenance tasks in any industry. However, data curation for training such models is expensive and time-consuming due to the unavailability of datasets for different wear and tear scenarios. Our approach employs a vision language model along with a 3D simulation and rendering engine to generate synthetic data for varying rust conditions. We evaluate our approach by training a CV model for rust detection using the generated dataset and tested the trained model on real images of rusted industrial objects. The model trained with the synthetic data generated by our approach, outperforms the other approaches with a mAP50 score of 0.87. The approach is customizable and can be easily extended to other industrial wear and tear detection scenarios
>
---
#### [new 011] MCANet: A Multi-Scale Class-Specific Attention Network for Multi-Label Post-Hurricane Damage Assessment using UAV Imagery
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MCANet，解决飓风后无人机图像多标签损害评估中多尺度特征提取与类别区分难题。通过多尺度Res2Net骨干和类特定注意力模块，实现92.35%的高精度，提升对复杂损害类型的识别能力，支持灾后决策。**

- **链接: [http://arxiv.org/pdf/2509.04757v1](http://arxiv.org/pdf/2509.04757v1)**

> **作者:** Zhangding Liu; Neda Mohammadi; John E. Taylor
>
> **备注:** 34 pages, 7 figures
>
> **摘要:** Rapid and accurate post-hurricane damage assessment is vital for disaster response and recovery. Yet existing CNN-based methods struggle to capture multi-scale spatial features and to distinguish visually similar or co-occurring damage types. To address these issues, we propose MCANet, a multi-label classification framework that learns multi-scale representations and adaptively attends to spatially relevant regions for each damage category. MCANet employs a Res2Net-based hierarchical backbone to enrich spatial context across scales and a multi-head class-specific residual attention module to enhance discrimination. Each attention branch focuses on different spatial granularities, balancing local detail with global context. We evaluate MCANet on the RescueNet dataset of 4,494 UAV images collected after Hurricane Michael. MCANet achieves a mean average precision (mAP) of 91.75%, outperforming ResNet, Res2Net, VGG, MobileNet, EfficientNet, and ViT. With eight attention heads, performance further improves to 92.35%, boosting average precision for challenging classes such as Road Blocked by over 6%. Class activation mapping confirms MCANet's ability to localize damage-relevant regions, supporting interpretability. Outputs from MCANet can inform post-disaster risk mapping, emergency routing, and digital twin-based disaster response. Future work could integrate disaster-specific knowledge graphs and multimodal large language models to improve adaptability to unseen disasters and enrich semantic understanding for real-world decision-making.
>
---
#### [new 012] UAV-Based Intelligent Traffic Surveillance System: Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis
- **分类: cs.CV; cs.ET; cs.RO; cs.SY; eess.IV; eess.SY**

- **简介: 论文提出基于无人机的智能交通监控系统，解决传统方法覆盖不足、适应性差的问题，实现实时车辆检测、分类、跟踪及违规行为分析，通过多技术融合和案例验证系统有效性。**

- **链接: [http://arxiv.org/pdf/2509.04624v1](http://arxiv.org/pdf/2509.04624v1)**

> **作者:** Ali Khanpour; Tianyi Wang; Afra Vahidi-Shams; Wim Ectors; Farzam Nakhaie; Amirhossein Taheri; Christian Claudel
>
> **备注:** 15 pages, 8 figures, 2 tables
>
> **摘要:** Traffic congestion and violations pose significant challenges for urban mobility and road safety. Traditional traffic monitoring systems, such as fixed cameras and sensor-based methods, are often constrained by limited coverage, low adaptability, and poor scalability. To address these challenges, this paper introduces an advanced unmanned aerial vehicle (UAV)-based traffic surveillance system capable of accurate vehicle detection, classification, tracking, and behavioral analysis in real-world, unconstrained urban environments. The system leverages multi-scale and multi-angle template matching, Kalman filtering, and homography-based calibration to process aerial video data collected from altitudes of approximately 200 meters. A case study in urban area demonstrates robust performance, achieving a detection precision of 91.8%, an F1-score of 90.5%, and tracking metrics (MOTA/MOTP) of 92.1% and 93.7%, respectively. Beyond precise detection, the system classifies five vehicle types and automatically detects critical traffic violations, including unsafe lane changes, illegal double parking, and crosswalk obstructions, through the fusion of geofencing, motion filtering, and trajectory deviation analysis. The integrated analytics module supports origin-destination tracking, vehicle count visualization, inter-class correlation analysis, and heatmap-based congestion modeling. Additionally, the system enables entry-exit trajectory profiling, vehicle density estimation across road segments, and movement direction logging, supporting comprehensive multi-scale urban mobility analytics. Experimental results confirms the system's scalability, accuracy, and practical relevance, highlighting its potential as an enforcement-aware, infrastructure-independent traffic monitoring solution for next-generation smart cities.
>
---
#### [new 013] SGS-3D: High-Fidelity 3D Instance Segmentation via Reliable Semantic Mask Splitting and Growing
- **分类: cs.CV**

- **简介: 该论文提出SGS-3D方法，解决3D实例分割中因2D到3D提升误差导致的不准确问题，通过"分割-生长"框架融合语义与几何信息，提升分割精度和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.05144v1](http://arxiv.org/pdf/2509.05144v1)**

> **作者:** Chaolei Wang; Yang Luo; Jing Du; Siyu Chen; Yiping Chen; Ting Han
>
> **摘要:** Accurate 3D instance segmentation is crucial for high-quality scene understanding in the 3D vision domain. However, 3D instance segmentation based on 2D-to-3D lifting approaches struggle to produce precise instance-level segmentation, due to accumulated errors introduced during the lifting process from ambiguous semantic guidance and insufficient depth constraints. To tackle these challenges, we propose splitting and growing reliable semantic mask for high-fidelity 3D instance segmentation (SGS-3D), a novel "split-then-grow" framework that first purifies and splits ambiguous lifted masks using geometric primitives, and then grows them into complete instances within the scene. Unlike existing approaches that directly rely on raw lifted masks and sacrifice segmentation accuracy, SGS-3D serves as a training-free refinement method that jointly fuses semantic and geometric information, enabling effective cooperation between the two levels of representation. Specifically, for semantic guidance, we introduce a mask filtering strategy that leverages the co-occurrence of 3D geometry primitives to identify and remove ambiguous masks, thereby ensuring more reliable semantic consistency with the 3D object instances. For the geometric refinement, we construct fine-grained object instances by exploiting both spatial continuity and high-level features, particularly in the case of semantic ambiguity between distinct objects. Experimental results on ScanNet200, ScanNet++, and KITTI-360 demonstrate that SGS-3D substantially improves segmentation accuracy and robustness against inaccurate masks from pre-trained models, yielding high-fidelity object instances while maintaining strong generalization across diverse indoor and outdoor environments. Code is available in the supplementary materials.
>
---
#### [new 014] Efficient Video-to-Audio Generation via Multiple Foundation Models Mapper
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: 本文提出MFM-Mapper，通过融合双视觉编码器与GPT-2，解决V2A生成中训练成本高的问题，实现高效且性能优越的音频生成。**

- **链接: [http://arxiv.org/pdf/2509.04957v1](http://arxiv.org/pdf/2509.04957v1)**

> **作者:** Gehui Chen; Guan'an Wang; Xiaowen Huang; Jitao Sang
>
> **摘要:** Recent Video-to-Audio (V2A) generation relies on extracting semantic and temporal features from video to condition generative models. Training these models from scratch is resource intensive. Consequently, leveraging foundation models (FMs) has gained traction due to their cross-modal knowledge transfer and generalization capabilities. One prior work has explored fine-tuning a lightweight mapper network to connect a pre-trained visual encoder with a text-to-audio generation model for V2A. Inspired by this, we introduce the Multiple Foundation Model Mapper (MFM-Mapper). Compared to the previous mapper approach, MFM-Mapper benefits from richer semantic and temporal information by fusing features from dual visual encoders. Furthermore, by replacing a linear mapper with GPT-2, MFM-Mapper improves feature alignment, drawing parallels between cross-modal features mapping and autoregressive translation tasks. Our MFM-Mapper exhibits remarkable training efficiency. It achieves better performance in semantic and temporal consistency with fewer training consuming, requiring only 16\% of the training scale compared to previous mapper-based work, yet achieves competitive performance with models trained on a much larger scale.
>
---
#### [new 015] Pose-Free 3D Quantitative Phase Imaging of Flowing Cellular Populations
- **分类: cs.CV; physics.bio-ph; physics.optics; q-bio.QM**

- **简介: 该论文提出OmniFHT框架，解决传统3D定量相位成像对细胞姿态依赖的问题，实现无姿态假设的高通量流式细胞多轴旋转与稀疏投影重建，支持复杂细胞形态分析。**

- **链接: [http://arxiv.org/pdf/2509.04848v1](http://arxiv.org/pdf/2509.04848v1)**

> **作者:** Enze Ye; Wei Lin; Shaochi Ren; Yakun Liu; Xiaoping Li; Hao Wang; He Sun; Feng Pan
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** High-throughput 3D quantitative phase imaging (QPI) in flow cytometry enables label-free, volumetric characterization of individual cells by reconstructing their refractive index (RI) distributions from multiple viewing angles during flow through microfluidic channels. However, current imaging methods assume that cells undergo uniform, single-axis rotation, which require their poses to be known at each frame. This assumption restricts applicability to near-spherical cells and prevents accurate imaging of irregularly shaped cells with complex rotations. As a result, only a subset of the cellular population can be analyzed, limiting the ability of flow-based assays to perform robust statistical analysis. We introduce OmniFHT, a pose-free 3D RI reconstruction framework that leverages the Fourier diffraction theorem and implicit neural representations (INRs) for high-throughput flow cytometry tomographic imaging. By jointly optimizing each cell's unknown rotational trajectory and volumetric structure under weak scattering assumptions, OmniFHT supports arbitrary cell geometries and multi-axis rotations. Its continuous representation also allows accurate reconstruction from sparsely sampled projections and restricted angular coverage, producing high-fidelity results with as few as 10 views or only 120 degrees of angular range. OmniFHT enables, for the first time, in situ, high-throughput tomographic imaging of entire flowing cell populations, providing a scalable and unbiased solution for label-free morphometric analysis in flow cytometry platforms.
>
---
#### [new 016] WATCH: World-aware Allied Trajectory and pose reconstruction for Camera and Human
- **分类: cs.CV**

- **简介: 论文提出WATCH框架，解决全球人类运动重建中相机与人体运动纠缠及深度歧义问题，通过航向角分解和相机轨迹整合机制，实现高效准确的轨迹与姿态重建，达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.04600v1](http://arxiv.org/pdf/2509.04600v1)**

> **作者:** Qijun Ying; Zhongyuan Hu; Rui Zhang; Ronghui Li; Yu Lu; Zijiao Zeng
>
> **摘要:** Global human motion reconstruction from in-the-wild monocular videos is increasingly demanded across VR, graphics, and robotics applications, yet requires accurate mapping of human poses from camera to world coordinates-a task challenged by depth ambiguity, motion ambiguity, and the entanglement between camera and human movements. While human-motion-centric approaches excel in preserving motion details and physical plausibility, they suffer from two critical limitations: insufficient exploitation of camera orientation information and ineffective integration of camera translation cues. We present WATCH (World-aware Allied Trajectory and pose reconstruction for Camera and Human), a unified framework addressing both challenges. Our approach introduces an analytical heading angle decomposition technique that offers superior efficiency and extensibility compared to existing geometric methods. Additionally, we design a camera trajectory integration mechanism inspired by world models, providing an effective pathway for leveraging camera translation information beyond naive hard-decoding approaches. Through experiments on in-the-wild benchmarks, WATCH achieves state-of-the-art performance in end-to-end trajectory reconstruction. Our work demonstrates the effectiveness of jointly modeling camera-human motion relationships and offers new insights for addressing the long-standing challenge of camera translation integration in global human motion reconstruction. The code will be available publicly.
>
---
#### [new 017] LUIVITON: Learned Universal Interoperable VIrtual Try-ON
- **分类: cs.CV**

- **简介: 该论文提出LUIVITON系统，解决复杂衣物与多样化人体的自动虚拟试穿问题。通过SMPL分解为衣物-人体对应任务，结合几何学习与扩散模型，实现无需人工的高质3D试穿，支持衣物尺寸定制及非流形网格处理。**

- **链接: [http://arxiv.org/pdf/2509.05030v1](http://arxiv.org/pdf/2509.05030v1)**

> **作者:** Cong Cao; Xianhang Cheng; Jingyuan Liu; Yujian Zheng; Zhenhui Lin; Meriem Chkir; Hao Li
>
> **摘要:** We present LUIVITON, an end-to-end system for fully automated virtual try-on, capable of draping complex, multi-layer clothing onto diverse and arbitrarily posed humanoid characters. To address the challenge of aligning complex garments with arbitrary and highly diverse body shapes, we use SMPL as a proxy representation and separate the clothing-to-body draping problem into two correspondence tasks: 1) clothing-to-SMPL and 2) body-to-SMPL correspondence, where each has its unique challenges. While we address the clothing-to-SMPL fitting problem using a geometric learning-based approach for partial-to-complete shape correspondence prediction, we introduce a diffusion model-based approach for body-to-SMPL correspondence using multi-view consistent appearance features and a pre-trained 2D foundation model. Our method can handle complex geometries, non-manifold meshes, and generalizes effectively to a wide range of humanoid characters -- including humans, robots, cartoon subjects, creatures, and aliens, while maintaining computational efficiency for practical adoption. In addition to offering a fully automatic fitting solution, LUIVITON supports fast customization of clothing size, allowing users to adjust clothing sizes and material properties after they have been draped. We show that our system can produce high-quality 3D clothing fittings without any human labor, even when 2D clothing sewing patterns are not available.
>
---
#### [new 018] Dual-Domain Perspective on Degradation-Aware Fusion: A VLM-Guided Robust Infrared and Visible Image Fusion Framework
- **分类: cs.CV**

- **简介: 该论文提出GD²Fusion框架，解决红外-可见光图像融合中退化输入导致的性能下降问题，通过VLM引导的双域（频域/空域）联合优化，实现鲁棒融合。**

- **链接: [http://arxiv.org/pdf/2509.05000v1](http://arxiv.org/pdf/2509.05000v1)**

> **作者:** Tianpei Zhang; Jufeng Zhao; Yiming Zhu; Guangmang Cui
>
> **摘要:** Most existing infrared-visible image fusion (IVIF) methods assume high-quality inputs, and therefore struggle to handle dual-source degraded scenarios, typically requiring manual selection and sequential application of multiple pre-enhancement steps. This decoupled pre-enhancement-to-fusion pipeline inevitably leads to error accumulation and performance degradation. To overcome these limitations, we propose Guided Dual-Domain Fusion (GD^2Fusion), a novel framework that synergistically integrates vision-language models (VLMs) for degradation perception with dual-domain (frequency/spatial) joint optimization. Concretely, the designed Guided Frequency Modality-Specific Extraction (GFMSE) module performs frequency-domain degradation perception and suppression and discriminatively extracts fusion-relevant sub-band features. Meanwhile, the Guided Spatial Modality-Aggregated Fusion (GSMAF) module carries out cross-modal degradation filtering and adaptive multi-source feature aggregation in the spatial domain to enhance modality complementarity and structural consistency. Extensive qualitative and quantitative experiments demonstrate that GD^2Fusion achieves superior fusion performance compared with existing algorithms and strategies in dual-source degraded scenarios. The code will be publicly released after acceptance of this paper.
>
---
#### [new 019] Facial Emotion Recognition does not detect feeling unsafe in automated driving
- **分类: cs.CV; J.4; I.2.10**

- **简介: 该论文研究面部表情识别在自动驾驶风险感知中的有效性，发现其不可靠，转而提出基于运动和皮肤电导的神经网络模型，用于客观评估驾驶者感知风险。**

- **链接: [http://arxiv.org/pdf/2509.04490v1](http://arxiv.org/pdf/2509.04490v1)**

> **作者:** Abel van Elburg; Konstantinos Gkentsidis; Mathieu Sarrazin; Sarah Barendswaard; Varun Kotian; Riender Happee
>
> **摘要:** Trust and perceived safety play a crucial role in the public acceptance of automated vehicles. To understand perceived risk, an experiment was conducted using a driving simulator under two automated driving styles and optionally introducing a crossing pedestrian. Data was collected from 32 participants, consisting of continuous subjective comfort ratings, motion, webcam footage for facial expression, skin conductance, heart rate, and eye tracking. The continuous subjective perceived risk ratings showed significant discomfort associated with perceived risk during cornering and braking followed by relief or even positive comfort on continuing the ride. The dynamic driving style induced a stronger discomfort as compared to the calm driving style. The crossing pedestrian did not affect discomfort with the calm driving style but doubled the comfort decrement with the dynamic driving style. This illustrates the importance of consequences of critical interactions in risk perception. Facial expression was successfully analyzed for 24 participants but most (15/24) did not show any detectable facial reaction to the critical event. Among the 9 participants who did, 8 showed a Happy expression, and only 4 showed a Surprise expression. Fear was never dominant. This indicates that facial expression recognition is not a reliable method for assessing perceived risk in automated vehicles. To predict perceived risk a neural network model was implemented using vehicle motion and skin conductance. The model correlated well with reported perceived risk, demonstrating its potential for objective perceived risk assessment in automated vehicles, reducing subjective bias and highlighting areas for future research.
>
---
#### [new 020] A Scalable Attention-Based Approach for Image-to-3D Texture Mapping
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出基于Transformer的图像到3D纹理映射方法，解决现有技术依赖UV图、生成速度慢、纹理失真的问题。通过直接预测3D纹理场并结合三平面表示，实现高效高保真纹理生成，适用于大规模高质量3D内容创作。**

- **链接: [http://arxiv.org/pdf/2509.05131v1](http://arxiv.org/pdf/2509.05131v1)**

> **作者:** Arianna Rampini; Kanika Madan; Bruno Roy; AmirHossein Zamani; Derek Cheung
>
> **摘要:** High-quality textures are critical for realistic 3D content creation, yet existing generative methods are slow, rely on UV maps, and often fail to remain faithful to a reference image. To address these challenges, we propose a transformer-based framework that predicts a 3D texture field directly from a single image and a mesh, eliminating the need for UV mapping and differentiable rendering, and enabling faster texture generation. Our method integrates a triplane representation with depth-based backprojection losses, enabling efficient training and faster inference. Once trained, it generates high-fidelity textures in a single forward pass, requiring only 0.2s per shape. Extensive qualitative, quantitative, and user preference evaluations demonstrate that our method outperforms state-of-the-art baselines on single-image texture reconstruction in terms of both fidelity to the input image and perceptual quality, highlighting its practicality for scalable, high-quality, and controllable 3D content creation.
>
---
#### [new 021] Scale-interaction transformer: a hybrid cnn-transformer model for facial beauty prediction
- **分类: cs.CV**

- **简介: 该论文提出Scale-Interaction Transformer（SIT），解决面部美预测中CNN固定尺度导致的多粒度特征依赖问题。通过多尺度CNN提取特征并结合Transformer建模交互关系，实现高精度预测，在SCUT-FBP5500数据集上达到0.9187的Pearson相关系数。**

- **链接: [http://arxiv.org/pdf/2509.05078v1](http://arxiv.org/pdf/2509.05078v1)**

> **作者:** Djamel Eddine Boukhari
>
> **摘要:** Automated Facial Beauty Prediction (FBP) is a challenging computer vision task due to the complex interplay of local and global facial features that influence human perception. While Convolutional Neural Networks (CNNs) excel at feature extraction, they often process information at a fixed scale, potentially overlooking the critical inter-dependencies between features at different levels of granularity. To address this limitation, we introduce the Scale-Interaction Transformer (SIT), a novel hybrid deep learning architecture that synergizes the feature extraction power of CNNs with the relational modeling capabilities of Transformers. The SIT first employs a multi-scale module with parallel convolutions to capture facial characteristics at varying receptive fields. These multi-scale representations are then framed as a sequence and processed by a Transformer encoder, which explicitly models their interactions and contextual relationships via a self-attention mechanism. We conduct extensive experiments on the widely-used SCUT-FBP5500 benchmark dataset, where the proposed SIT model establishes a new state-of-the-art. It achieves a Pearson Correlation of 0.9187, outperforming previous methods. Our findings demonstrate that explicitly modeling the interplay between multi-scale visual cues is crucial for high-performance FBP. The success of the SIT architecture highlights the potential of hybrid CNN-Transformer models for complex image regression tasks that demand a holistic, context-aware understanding.
>
---
#### [new 022] Extracting Uncertainty Estimates from Mixtures of Experts for Semantic Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对语义分割中的不确定性估计问题，提出利用混合专家模型（MoE）通过门控网络动态加权专家预测，评估预测熵、互信息和专家方差等方法，发现MoE在分布外数据上优于集成方法，并验证增加专家数量可提升不确定性校准。**

- **链接: [http://arxiv.org/pdf/2509.04816v1](http://arxiv.org/pdf/2509.04816v1)**

> **作者:** Svetlana Pavlitska; Beyza Keskin; Alwin Faßbender; Christian Hubschneider; J. Marius Zöllner
>
> **备注:** Accepted for publication at the STREAM workshop at ICCV2025
>
> **摘要:** Estimating accurate and well-calibrated predictive uncertainty is important for enhancing the reliability of computer vision models, especially in safety-critical applications like traffic scene perception. While ensemble methods are commonly used to quantify uncertainty by combining multiple models, a mixture of experts (MoE) offers an efficient alternative by leveraging a gating network to dynamically weight expert predictions based on the input. Building on the promising use of MoEs for semantic segmentation in our previous works, we show that well-calibrated predictive uncertainty estimates can be extracted from MoEs without architectural modifications. We investigate three methods to extract predictive uncertainty estimates: predictive entropy, mutual information, and expert variance. We evaluate these methods for an MoE with two experts trained on a semantical split of the A2D2 dataset. Our results show that MoEs yield more reliable uncertainty estimates than ensembles in terms of conditional correctness metrics under out-of-distribution (OOD) data. Additionally, we evaluate routing uncertainty computed via gate entropy and find that simple gating mechanisms lead to better calibration of routing uncertainty estimates than more complex classwise gates. Finally, our experiments on the Cityscapes dataset suggest that increasing the number of experts can further enhance uncertainty calibration. Our code is available at https://github.com/KASTEL-MobilityLab/mixtures-of-experts/.
>
---
#### [new 023] Comparative Evaluation of Traditional and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar Data
- **分类: cs.CV**

- **简介: 论文比较传统与深度学习特征匹配算法在月球多传感器图像配准中的表现，评估SIFT、ASIFT、AKAZE、RIFT2和SuperGlue，提出预处理方法，发现SuperGlue在精度与速度上最优，强调预处理与学习方法的重要性。**

- **链接: [http://arxiv.org/pdf/2509.04775v1](http://arxiv.org/pdf/2509.04775v1)**

> **作者:** R. Makharia; J. G. Singla; Amitabh; N. Dube; H. Sharma
>
> **备注:** 27 pages, 11 figures, 3 tables
>
> **摘要:** Accurate image registration is critical for lunar exploration, enabling surface mapping, resource localization, and mission planning. Aligning data from diverse lunar sensors -- optical (e.g., Orbital High Resolution Camera, Narrow and Wide Angle Cameras), hyperspectral (Imaging Infrared Spectrometer), and radar (e.g., Dual-Frequency Synthetic Aperture Radar, Selene/Kaguya mission) -- is challenging due to differences in resolution, illumination, and sensor distortion. We evaluate five feature matching algorithms: SIFT, ASIFT, AKAZE, RIFT2, and SuperGlue (a deep learning-based matcher), using cross-modality image pairs from equatorial and polar regions. A preprocessing pipeline is proposed, including georeferencing, resolution alignment, intensity normalization, and enhancements like adaptive histogram equalization, principal component analysis, and shadow correction. SuperGlue consistently yields the lowest root mean square error and fastest runtimes. Classical methods such as SIFT and AKAZE perform well near the equator but degrade under polar lighting. The results highlight the importance of preprocessing and learning-based approaches for robust lunar image registration across diverse conditions.
>
---
#### [new 024] COGITAO: A Visual Reasoning Framework To Study Compositionality & Generalization
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出COGITAO框架，旨在研究视觉任务中的组合性与泛化能力。针对现有模型在新组合场景下泛化不足的问题，通过规则生成大量可调节深度的网格任务，支持多变换组合与参数控制，提供基线实验验证模型性能。**

- **链接: [http://arxiv.org/pdf/2509.05249v1](http://arxiv.org/pdf/2509.05249v1)**

> **作者:** Yassine Taoudi-Benchekroun; Klim Troyan; Pascal Sager; Stefan Gerber; Lukas Tuggener; Benjamin Grewe
>
> **备注:** 10 main pages, 3 figure, appendix available
>
> **摘要:** The ability to compose learned concepts and apply them in novel settings is key to human intelligence, but remains a persistent limitation in state-of-the-art machine learning models. To address this issue, we introduce COGITAO, a modular and extensible data generation framework and benchmark designed to systematically study compositionality and generalization in visual domains. Drawing inspiration from ARC-AGI's problem-setting, COGITAO constructs rule-based tasks which apply a set of transformations to objects in grid-like environments. It supports composition, at adjustable depth, over a set of 28 interoperable transformations, along with extensive control over grid parametrization and object properties. This flexibility enables the creation of millions of unique task rules -- surpassing concurrent datasets by several orders of magnitude -- across a wide range of difficulties, while allowing virtually unlimited sample generation per rule. We provide baseline experiments using state-of-the-art vision models, highlighting their consistent failures to generalize to novel combinations of familiar elements, despite strong in-domain performance. COGITAO is fully open-sourced, including all code and datasets, to support continued research in this field.
>
---
#### [new 025] Toward Accessible Dermatology: Skin Lesion Classification Using Deep Learning Models on Mobile-Acquired Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文旨在解决传统皮肤病诊断成本高、资源匮乏地区的挑战，通过构建移动设备采集的50类皮肤病变数据集，对比多种深度学习模型，验证Swin Transformer在捕捉全局特征上的优势，并结合Grad-CAM提升模型可解释性，推动AI辅助皮肤病筛查的可行性。**

- **链接: [http://arxiv.org/pdf/2509.04800v1](http://arxiv.org/pdf/2509.04800v1)**

> **作者:** Asif Newaz; Masum Mushfiq Ishti; A Z M Ashraful Azam; Asif Ur Rahman Adib
>
> **备注:** Under Review in ICSigSys 2025
>
> **摘要:** Skin diseases are among the most prevalent health concerns worldwide, yet conventional diagnostic methods are often costly, complex, and unavailable in low-resource settings. Automated classification using deep learning has emerged as a promising alternative, but existing studies are mostly limited to dermoscopic datasets and a narrow range of disease classes. In this work, we curate a large dataset of over 50 skin disease categories captured with mobile devices, making it more representative of real-world conditions. We evaluate multiple convolutional neural networks and Transformer-based architectures, demonstrating that Transformer models, particularly the Swin Transformer, achieve superior performance by effectively capturing global contextual features. To enhance interpretability, we incorporate Gradient-weighted Class Activation Mapping (Grad-CAM), which highlights clinically relevant regions and provides transparency in model predictions. Our results underscore the potential of Transformer-based approaches for mobile-acquired skin lesion classification, paving the way toward accessible AI-assisted dermatological screening and early diagnosis in resource-limited environments.
>
---
#### [new 026] SpiderNets: Estimating Fear Ratings of Spider-Related Images with Vision Models
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **简介: 论文提出SpiderNets，利用视觉模型预测蜘蛛图像的恐惧评分，解决临床治疗中动态调整刺激的问题。通过迁移学习优化模型，评估其准确性，并分析误差来源，强调数据量和可解释性的重要性。**

- **链接: [http://arxiv.org/pdf/2509.04889v1](http://arxiv.org/pdf/2509.04889v1)**

> **作者:** Dominik Pegler; David Steyrl; Mengfan Zhang; Alexander Karner; Jozsef Arato; Frank Scharnowski; Filip Melinscak
>
> **备注:** 60 pages (30 main text, 30 appendix), 20 figures (5 in main text, 15 in appendix)
>
> **摘要:** Advances in computer vision have opened new avenues for clinical applications, particularly in computerized exposure therapy where visual stimuli can be dynamically adjusted based on patient responses. As a critical step toward such adaptive systems, we investigated whether pretrained computer vision models can accurately predict fear levels from spider-related images. We adapted three diverse models using transfer learning to predict human fear ratings (on a 0-100 scale) from a standardized dataset of 313 images. The models were evaluated using cross-validation, achieving an average mean absolute error (MAE) between 10.1 and 11.0. Our learning curve analysis revealed that reducing the dataset size significantly harmed performance, though further increases yielded no substantial gains. Explainability assessments showed the models' predictions were based on spider-related features. A category-wise error analysis further identified visual conditions associated with higher errors (e.g., distant views and artificial/painted spiders). These findings demonstrate the potential of explainable computer vision models in predicting fear ratings, highlighting the importance of both model explainability and a sufficient dataset size for developing effective emotion-aware therapeutic technologies.
>
---
#### [new 027] CD-Mamba: Cloud detection with long-range spatial dependency modeling
- **分类: cs.CV**

- **简介: 该论文提出CD-Mamba模型，解决遥感图像云检测问题，通过融合卷积网络与Mamba的长程依赖建模，同时捕捉局部纹理与全局patch特征，提升多尺度检测精度。**

- **链接: [http://arxiv.org/pdf/2509.04729v1](http://arxiv.org/pdf/2509.04729v1)**

> **作者:** Tianxiang Xue; Jiayi Zhao; Jingsheng Li; Changlu Chen; Kun Zhan
>
> **备注:** Journal of Applied Remote Sensing
>
> **摘要:** Remote sensing images are frequently obscured by cloud cover, posing significant challenges to data integrity and reliability. Effective cloud detection requires addressing both short-range spatial redundancies and long-range atmospheric similarities among cloud patches. Convolutional neural networks are effective at capturing local spatial dependencies, while Mamba has strong capabilities in modeling long-range dependencies. To fully leverage both local spatial relations and long-range dependencies, we propose CD-Mamba, a hybrid model that integrates convolution and Mamba's state-space modeling into a unified cloud detection network. CD-Mamba is designed to comprehensively capture pixelwise textural details and long term patchwise dependencies for cloud detection. This design enables CD-Mamba to manage both pixel-wise interactions and extensive patch-wise dependencies simultaneously, improving detection accuracy across diverse spatial scales. Extensive experiments validate the effectiveness of CD-Mamba and demonstrate its superior performance over existing methods.
>
---
#### [new 028] GeoSplat: A Deep Dive into Geometry-Constrained Gaussian Splatting
- **分类: cs.CV**

- **简介: 论文提出GeoSplat框架，针对高斯泼溅中低阶几何先验不稳定的问题，结合一阶/二阶几何量优化初始化与更新，提升新视角合成性能。**

- **链接: [http://arxiv.org/pdf/2509.05075v1](http://arxiv.org/pdf/2509.05075v1)**

> **作者:** Yangming Li; Chaoyu Liu; Lihao Liu; Simon Masnou; Carola-Bibian Schönlieb
>
> **摘要:** A few recent works explored incorporating geometric priors to regularize the optimization of Gaussian splatting, further improving its performance. However, those early studies mainly focused on the use of low-order geometric priors (e.g., normal vector), and they are also unreliably estimated by noise-sensitive methods, like local principal component analysis. To address their limitations, we first present GeoSplat, a general geometry-constrained optimization framework that exploits both first-order and second-order geometric quantities to improve the entire training pipeline of Gaussian splatting, including Gaussian initialization, gradient update, and densification. As an example, we initialize the scales of 3D Gaussian primitives in terms of principal curvatures, leading to a better coverage of the object surface than random initialization. Secondly, based on certain geometric structures (e.g., local manifold), we introduce efficient and noise-robust estimation methods that provide dynamic geometric priors for our framework. We conduct extensive experiments on multiple datasets for novel view synthesis, showing that our framework: GeoSplat, significantly improves the performance of Gaussian splatting and outperforms previous baselines.
>
---
#### [new 029] Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 本论文针对光场超分辨率任务，提出LFMT框架，通过Sub-SS策略和双阶段建模，结合Mamba与Transformer模型，高效提取非局部空间-角度特征，解决现有方法效率低、冗余及信息丢失问题，显著提升性能并降低计算复杂度。**

- **链接: [http://arxiv.org/pdf/2509.04824v1](http://arxiv.org/pdf/2509.04824v1)**

> **作者:** Haosong Liu; Xiancheng Zhu; Huanqiang Zeng; Jianqing Zhu; Jiuwen Cao; Junhui Hou
>
> **摘要:** Recently, Mamba-based methods, with its advantage in long-range information modeling and linear complexity, have shown great potential in optimizing both computational cost and performance of light field image super-resolution (LFSR). However, current multi-directional scanning strategies lead to inefficient and redundant feature extraction when applied to complex LF data. To overcome this challenge, we propose a Subspace Simple Scanning (Sub-SS) strategy, based on which we design the Subspace Simple Mamba Block (SSMB) to achieve more efficient and precise feature extraction. Furthermore, we propose a dual-stage modeling strategy to address the limitation of state space in preserving spatial-angular and disparity information, thereby enabling a more comprehensive exploration of non-local spatial-angular correlations. Specifically, in stage I, we introduce the Spatial-Angular Residual Subspace Mamba Block (SA-RSMB) for shallow spatial-angular feature extraction; in stage II, we use a dual-branch parallel structure combining the Epipolar Plane Mamba Block (EPMB) and Epipolar Plane Transformer Block (EPTB) for deep epipolar feature refinement. Building upon meticulously designed modules and strategies, we introduce a hybrid Mamba-Transformer framework, termed LFMT. LFMT integrates the strengths of Mamba and Transformer models for LFSR, enabling comprehensive information exploration across spatial, angular, and epipolar-plane domains. Experimental results demonstrate that LFMT significantly outperforms current state-of-the-art methods in LFSR, achieving substantial improvements in performance while maintaining low computational complexity on both real-word and synthetic LF datasets.
>
---
#### [new 030] UniView: Enhancing Novel View Synthesis From A Single Image By Unifying Reference Features
- **分类: cs.CV**

- **简介: 论文提出UniView模型，通过统一参考特征提升单图像新型视图合成质量。解决现有方法因模糊先验和插值导致的失真问题，采用检索系统、多模态语言模型、多级隔离层适配器及三重注意力机制，显著提升合成效果。**

- **链接: [http://arxiv.org/pdf/2509.04932v1](http://arxiv.org/pdf/2509.04932v1)**

> **作者:** Haowang Cui; Rui Chen; Tao Luo; Rui Li; Jiaze Wang
>
> **备注:** Submitted to ACM TOMM
>
> **摘要:** The task of synthesizing novel views from a single image is highly ill-posed due to multiple explanations for unobserved areas. Most current methods tend to generate unseen regions from ambiguity priors and interpolation near input views, which often lead to severe distortions. To address this limitation, we propose a novel model dubbed as UniView, which can leverage reference images from a similar object to provide strong prior information during view synthesis. More specifically, we construct a retrieval and augmentation system and employ a multimodal large language model (MLLM) to assist in selecting reference images that meet our requirements. Additionally, a plug-and-play adapter module with multi-level isolation layers is introduced to dynamically generate reference features for the target views. Moreover, in order to preserve the details of an original input image, we design a decoupled triple attention mechanism, which can effectively align and integrate multi-branch features into the synthesis process. Extensive experiments have demonstrated that our UniView significantly improves novel view synthesis performance and outperforms state-of-the-art methods on the challenging datasets.
>
---
#### [new 031] Guideline-Consistent Segmentation via Multi-Agent Refinement
- **分类: cs.CV**

- **简介: 论文提出多智能体框架，解决语义分割中遵循复杂文本指南的问题，通过Worker-Supervisor迭代优化，无需训练，提升指南一致性。**

- **链接: [http://arxiv.org/pdf/2509.04687v1](http://arxiv.org/pdf/2509.04687v1)**

> **作者:** Vanshika Vats; Ashwani Rathee; James Davis
>
> **摘要:** Semantic segmentation in real-world applications often requires not only accurate masks but also strict adherence to textual labeling guidelines. These guidelines are typically complex and long, and both human and automated labeling often fail to follow them faithfully. Traditional approaches depend on expensive task-specific retraining that must be repeated as the guidelines evolve. Although recent open-vocabulary segmentation methods excel with simple prompts, they often fail when confronted with sets of paragraph-length guidelines that specify intricate segmentation rules. To address this, we introduce a multi-agent, training-free framework that coordinates general-purpose vision-language models within an iterative Worker-Supervisor refinement architecture. The Worker performs the segmentation, the Supervisor critiques it against the retrieved guidelines, and a lightweight reinforcement learning stop policy decides when to terminate the loop, ensuring guideline-consistent masks while balancing resource use. Evaluated on the Waymo and ReasonSeg datasets, our method notably outperforms state-of-the-art baselines, demonstrating strong generalization and instruction adherence.
>
---
#### [new 032] Skywork UniPic 2.0: Building Kontext Model with Online RL for Unified Multimodal Model
- **分类: cs.CV**

- **简介: 该论文提出Skywork UniPic 2.0，通过架构优化和在线RL策略（PDTR），解决多模态模型参数冗余与训练效率低问题，实现图像生成/编辑与理解的统一框架，超越大参数模型表现。**

- **链接: [http://arxiv.org/pdf/2509.04548v1](http://arxiv.org/pdf/2509.04548v1)**

> **作者:** Hongyang Wei; Baixin Xu; Hongbo Liu; Cyrus Wu; Jie Liu; Yi Peng; Peiyu Wang; Zexiang Liu; Jingwen He; Yidan Xietian; Chuanxin Tang; Zidong Wang; Yichen Wei; Liang Hu; Boyi Jiang; William Li; Ying He; Yang Liu; Xuchen Song; Eric Li; Yahui Zhou
>
> **摘要:** Recent advances in multimodal models have demonstrated impressive capabilities in unified image generation and editing. However, many prominent open-source models prioritize scaling model parameters over optimizing training strategies, limiting their efficiency and performance. In this work, we present UniPic2-SD3.5M-Kontext, a 2B-parameter DiT model based on SD3.5-Medium, which achieves state-of-the-art image generation and editing while extending seamlessly into a unified multimodal framework. Our approach begins with architectural modifications to SD3.5-Medium and large-scale pre-training on high-quality data, enabling joint text-to-image generation and editing capabilities. To enhance instruction following and editing consistency, we propose a novel Progressive Dual-Task Reinforcement strategy (PDTR), which effectively strengthens both tasks in a staged manner. We empirically validate that the reinforcement phases for different tasks are mutually beneficial and do not induce negative interference. After pre-training and reinforcement strategies, UniPic2-SD3.5M-Kontext demonstrates stronger image generation and editing capabilities than models with significantly larger generation parameters-including BAGEL (7B) and Flux-Kontext (12B). Furthermore, following the MetaQuery, we connect the UniPic2-SD3.5M-Kontext and Qwen2.5-VL-7B via a connector and perform joint training to launch a unified multimodal model UniPic2-Metaquery. UniPic2-Metaquery integrates understanding, generation, and editing, achieving top-tier performance across diverse tasks with a simple and scalable training paradigm. This consistently validates the effectiveness and generalizability of our proposed training paradigm, which we formalize as Skywork UniPic 2.0.
>
---
#### [new 033] Leveraging Transfer Learning and Mobile-enabled Convolutional Neural Networks for Improved Arabic Handwritten Character Recognition
- **分类: cs.CV**

- **简介: 该论文针对阿拉伯语手写字符识别任务，解决计算资源高和数据稀缺问题，结合迁移学习与轻量级MbNets，评估多种策略和模型，发现MobileNet性能最佳，ShuffleNet泛化能力强，为资源高效识别提供新方案。**

- **链接: [http://arxiv.org/pdf/2509.05019v1](http://arxiv.org/pdf/2509.05019v1)**

> **作者:** Mohsine El Khayati; Ayyad Maafiri; Yassine Himeur; Hamzah Ali Alkhazaleh; Shadi Atalla; Wathiq Mansoor
>
> **备注:** 20pages, 9 figures and 11 tables
>
> **摘要:** The study explores the integration of transfer learning (TL) with mobile-enabled convolutional neural networks (MbNets) to enhance Arabic Handwritten Character Recognition (AHCR). Addressing challenges like extensive computational requirements and dataset scarcity, this research evaluates three TL strategies--full fine-tuning, partial fine-tuning, and training from scratch--using four lightweight MbNets: MobileNet, SqueezeNet, MnasNet, and ShuffleNet. Experiments were conducted on three benchmark datasets: AHCD, HIJJA, and IFHCDB. MobileNet emerged as the top-performing model, consistently achieving superior accuracy, robustness, and efficiency, with ShuffleNet excelling in generalization, particularly under full fine-tuning. The IFHCDB dataset yielded the highest results, with 99% accuracy using MnasNet under full fine-tuning, highlighting its suitability for robust character recognition. The AHCD dataset achieved competitive accuracy (97%) with ShuffleNet, while HIJJA posed significant challenges due to its variability, achieving a peak accuracy of 92% with ShuffleNet. Notably, full fine-tuning demonstrated the best overall performance, balancing accuracy and convergence speed, while partial fine-tuning underperformed across metrics. These findings underscore the potential of combining TL and MbNets for resource-efficient AHCR, paving the way for further optimizations and broader applications. Future work will explore architectural modifications, in-depth dataset feature analysis, data augmentation, and advanced sensitivity analysis to enhance model robustness and generalizability.
>
---
#### [new 034] Symbolic Graphics Programming with Large Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究使用大语言模型从自然语言生成符号图形程序（SGPs），特别是SVG。针对LLM生成SGPs能力不足的问题，提出SGP-GenBench基准与强化学习方法，结合跨模态奖励优化生成质量，提升模型表现至前沿水平。**

- **链接: [http://arxiv.org/pdf/2509.05208v1](http://arxiv.org/pdf/2509.05208v1)**

> **作者:** Yamei Chen; Haoquan Zhang; Yangyi Huang; Zeju Qiu; Kaipeng Zhang; Yandong Wen; Weiyang Liu
>
> **备注:** Technical report (32 pages, 12 figures, project page: https://spherelab.ai/SGP-Gen/)
>
> **摘要:** Large language models (LLMs) excel at program synthesis, yet their ability to produce symbolic graphics programs (SGPs) that render into precise visual content remains underexplored. We study symbolic graphics programming, where the goal is to generate an SGP from a natural-language description. This task also serves as a lens into how LLMs understand the visual world by prompting them to generate images rendered from SGPs. Among various SGPs, our paper sticks to scalable vector graphics (SVGs). We begin by examining the extent to which LLMs can generate SGPs. To this end, we introduce SGP-GenBench, a comprehensive benchmark covering object fidelity, scene fidelity, and compositionality (attribute binding, spatial relations, numeracy). On SGP-GenBench, we discover that frontier proprietary models substantially outperform open-source models, and performance correlates well with general coding capabilities. Motivated by this gap, we aim to improve LLMs' ability to generate SGPs. We propose a reinforcement learning (RL) with verifiable rewards approach, where a format-validity gate ensures renderable SVG, and a cross-modal reward aligns text and the rendered image via strong vision encoders (e.g., SigLIP for text-image and DINO for image-image). Applied to Qwen-2.5-7B, our method substantially improves SVG generation quality and semantics, achieving performance on par with frontier systems. We further analyze training dynamics, showing that RL induces (i) finer decomposition of objects into controllable primitives and (ii) contextual details that improve scene coherence. Our results demonstrate that symbolic graphics programming offers a precise and interpretable lens on cross-modal grounding.
>
---
#### [new 035] FloodVision: Urban Flood Depth Estimation Using Foundation Vision-Language Models and Domain Knowledge Graph
- **分类: cs.CV; cs.AI**

- **简介: 论文提出FloodVision，结合GPT-4o与领域知识图谱，通过动态识别参考物、知识图谱验证高度及统计滤波，实现城市洪水深度估计，提升准确性和泛化能力，适用于智能城市应用。**

- **链接: [http://arxiv.org/pdf/2509.04772v1](http://arxiv.org/pdf/2509.04772v1)**

> **作者:** Zhangding Liu; Neda Mohammadi; John E. Taylor
>
> **摘要:** Timely and accurate floodwater depth estimation is critical for road accessibility and emergency response. While recent computer vision methods have enabled flood detection, they suffer from both accuracy limitations and poor generalization due to dependence on fixed object detectors and task-specific training. To enable accurate depth estimation that can generalize across diverse flood scenarios, this paper presents FloodVision, a zero-shot framework that combines the semantic reasoning abilities of the foundation vision-language model GPT-4o with a structured domain knowledge graph. The knowledge graph encodes canonical real-world dimensions for common urban objects including vehicles, people, and infrastructure elements to ground the model's reasoning in physical reality. FloodVision dynamically identifies visible reference objects in RGB images, retrieves verified heights from the knowledge graph to mitigate hallucination, estimates submergence ratios, and applies statistical outlier filtering to compute final depth values. Evaluated on 110 crowdsourced images from MyCoast New York, FloodVision achieves a mean absolute error of 8.17 cm, reducing the GPT-4o baseline 10.28 cm by 20.5% and surpassing prior CNN-based methods. The system generalizes well across varying scenes and operates in near real-time, making it suitable for future integration into digital twin platforms and citizen-reporting apps for smart city flood resilience.
>
---
#### [new 036] PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PropVG框架，解决视觉定位任务中现有方法效率低、依赖单一目标及缺乏多粒度判别的问题。通过整合前景提议生成与参考理解，引入对比学习模块和多粒度判别机制，提升复杂场景下的目标识别效果。**

- **链接: [http://arxiv.org/pdf/2509.04833v1](http://arxiv.org/pdf/2509.04833v1)**

> **作者:** Ming Dai; Wenxuan Cheng; Jiedong Zhuang; Jiang-jiang Liu; Hongshen Zhao; Zhenhua Feng; Wankou Yang
>
> **备注:** ICCV2025
>
> **摘要:** Recent advances in visual grounding have largely shifted away from traditional proposal-based two-stage frameworks due to their inefficiency and high computational complexity, favoring end-to-end direct reference paradigms. However, these methods rely exclusively on the referred target for supervision, overlooking the potential benefits of prominent prospective targets. Moreover, existing approaches often fail to incorporate multi-granularity discrimination, which is crucial for robust object identification in complex scenarios. To address these limitations, we propose PropVG, an end-to-end proposal-based framework that, to the best of our knowledge, is the first to seamlessly integrate foreground object proposal generation with referential object comprehension without requiring additional detectors. Furthermore, we introduce a Contrastive-based Refer Scoring (CRS) module, which employs contrastive learning at both sentence and word levels to enhance the capability in understanding and distinguishing referred objects. Additionally, we design a Multi-granularity Target Discrimination (MTD) module that fuses object- and semantic-level information to improve the recognition of absent targets. Extensive experiments on gRefCOCO (GREC/GRES), Ref-ZOM, R-RefCOCO, and RefCOCO (REC/RES) benchmarks demonstrate the effectiveness of PropVG. The codes and models are available at https://github.com/Dmmm1997/PropVG.
>
---
#### [new 037] Robust Experts: the Effect of Adversarial Training on CNNs with Sparse Mixture-of-Experts Layers
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出在CNN中引入稀疏MoE层，通过对抗训练提升对抗鲁棒性，解决传统方法资源消耗大的问题。实验表明，插入MoE层可增强ResNet-CIFAR-100的鲁棒性，且switch loss使专家路径专业化，提升特定专家的对抗能力。**

- **链接: [http://arxiv.org/pdf/2509.05086v1](http://arxiv.org/pdf/2509.05086v1)**

> **作者:** Svetlana Pavlitska; Haixi Fan; Konstantin Ditschuneit; J. Marius Zöllner
>
> **备注:** Accepted for publication at the STREAM workshop at ICCV 2025
>
> **摘要:** Robustifying convolutional neural networks (CNNs) against adversarial attacks remains challenging and often requires resource-intensive countermeasures. We explore the use of sparse mixture-of-experts (MoE) layers to improve robustness by replacing selected residual blocks or convolutional layers, thereby increasing model capacity without additional inference cost. On ResNet architectures trained on CIFAR-100, we find that inserting a single MoE layer in the deeper stages leads to consistent improvements in robustness under PGD and AutoPGD attacks when combined with adversarial training. Furthermore, we discover that when switch loss is used for balancing, it causes routing to collapse onto a small set of overused experts, thereby concentrating adversarial training on these paths and inadvertently making them more robust. As a result, some individual experts outperform the gated MoE model in robustness, suggesting that robust subpaths emerge through specialization. Our code is available at https://github.com/KASTEL-MobilityLab/robust-sparse-moes.
>
---
#### [new 038] A biologically inspired separable learning vision model for real-time traffic object perception in Dark
- **分类: cs.CV**

- **简介: 该论文针对低光交通场景下的目标感知任务，解决光照退化与数据不足问题，提出Dark-traffic数据集和SLVM模型，通过生物启发机制提升多任务性能。**

- **链接: [http://arxiv.org/pdf/2509.05012v1](http://arxiv.org/pdf/2509.05012v1)**

> **作者:** Hulin Li; Qiliang Ren; Jun Li; Hanbing Wei; Zheng Liu; Linfang Fan
>
> **摘要:** Fast and accurate object perception in low-light traffic scenes has attracted increasing attention. However, due to severe illumination degradation and the lack of reliable visual cues, existing perception models and methods struggle to quickly adapt to and accurately predict in low-light environments. Moreover, there is the absence of available large-scale benchmark specifically focused on low-light traffic scenes. To bridge this gap, we introduce a physically grounded illumination degradation method tailored to real-world low-light settings and construct Dark-traffic, the largest densely annotated dataset to date for low-light traffic scenes, supporting object detection, instance segmentation, and optical flow estimation. We further propose the Separable Learning Vision Model (SLVM), a biologically inspired framework designed to enhance perception under adverse lighting. SLVM integrates four key components: a light-adaptive pupillary mechanism for illumination-sensitive feature extraction, a feature-level separable learning strategy for efficient representation, task-specific decoupled branches for multi-task separable learning, and a spatial misalignment-aware fusion module for precise multi-feature alignment. Extensive experiments demonstrate that SLVM achieves state-of-the-art performance with reduced computational overhead. Notably, it outperforms RT-DETR by 11.2 percentage points in detection, YOLOv12 by 6.1 percentage points in instance segmentation, and reduces endpoint error (EPE) of baseline by 12.37% on Dark-traffic. On the LIS benchmark, the end-to-end trained SLVM surpasses Swin Transformer+EnlightenGAN and ConvNeXt-T+EnlightenGAN by an average of 11 percentage points across key metrics, and exceeds Mask RCNN (with light enhancement) by 3.1 percentage points. The Dark-traffic dataset and complete code is released at https://github.com/alanli1997/slvm.
>
---
#### [new 039] Enhancing Self-Driving Segmentation in Adverse Weather Conditions: A Dual Uncertainty-Aware Training Approach to SAM Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶图像分割在恶劣天气下的性能不足问题，提出双不确定性意识训练方法优化SAM模型，通过不确定性度量融入损失函数和适配医疗领域UAT框架，提升极端天气与复杂场景的分割鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.04735v1](http://arxiv.org/pdf/2509.04735v1)**

> **作者:** Dharsan Ravindran; Kevin Wang; Zhuoyuan Cao; Saleh Abdelrahman; Jeffery Wu
>
> **摘要:** Recent advances in vision foundation models, such as the Segment Anything Model (SAM) and its successor SAM2, have achieved state-of-the-art performance on general image segmentation benchmarks. However, these models struggle in adverse weather conditions where visual ambiguity is high, largely due to their lack of uncertainty quantification. Inspired by progress in medical imaging, where uncertainty-aware training has improved reliability in ambiguous cases, we investigate two approaches to enhance segmentation robustness for autonomous driving. First, we introduce a multi-step finetuning procedure for SAM2 that incorporates uncertainty metrics directly into the loss function, improving overall scene recognition. Second, we adapt the Uncertainty-Aware Adapter (UAT), originally designed for medical image segmentation, to driving contexts. We evaluate both methods on CamVid, BDD100K, and GTA driving datasets. Experiments show that UAT-SAM outperforms standard SAM in extreme weather, while SAM2 with uncertainty-aware loss achieves improved performance across diverse driving scenes. These findings underscore the value of explicit uncertainty modeling for safety-critical autonomous driving in challenging environments.
>
---
#### [new 040] Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping
- **分类: cs.CV; I.3.6; I.3.3**

- **简介: 该论文提出Inpaint4Drag框架，解决拖拽式图像编辑中潜在空间操作导致的精度低、反馈慢问题。通过像素级双向变形与图像修复结合，实现实时高效编辑，兼容任意修复模型，无需架构修改。**

- **链接: [http://arxiv.org/pdf/2509.04582v1](http://arxiv.org/pdf/2509.04582v1)**

> **作者:** Jingyi Lu; Kai Han
>
> **备注:** Accepted to ICCV 2025. Project page: https://visual-ai.github.io/inpaint4drag/
>
> **摘要:** Drag-based image editing has emerged as a powerful paradigm for intuitive image manipulation. However, existing approaches predominantly rely on manipulating the latent space of generative models, leading to limited precision, delayed feedback, and model-specific constraints. Accordingly, we present Inpaint4Drag, a novel framework that decomposes drag-based editing into pixel-space bidirectional warping and image inpainting. Inspired by elastic object deformation in the physical world, we treat image regions as deformable materials that maintain natural shape under user manipulation. Our method achieves real-time warping previews (0.01s) and efficient inpainting (0.3s) at 512x512 resolution, significantly improving the interaction experience compared to existing methods that require minutes per edit. By transforming drag inputs directly into standard inpainting formats, our approach serves as a universal adapter for any inpainting model without architecture modification, automatically inheriting all future improvements in inpainting technology. Extensive experiments demonstrate that our method achieves superior visual quality and precise control while maintaining real-time performance. Project page: https://visual-ai.github.io/inpaint4drag/
>
---
#### [new 041] SL-SLR: Self-Supervised Representation Learning for Sign Language Recognition
- **分类: cs.CV**

- **简介: 本论文提出SL-SLR框架，针对手语识别中对比学习的两个问题：不区分视频部分重要性及负样本相似性，设计自监督方法与数据增强技术，提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2509.05188v1](http://arxiv.org/pdf/2509.05188v1)**

> **作者:** Ariel Basso Madjoukeng; Jérôme Fink; Pierre Poitier; Edith Belise Kenmogne; Benoit Frenay
>
> **摘要:** Sign language recognition (SLR) is a machine learning task aiming to identify signs in videos. Due to the scarcity of annotated data, unsupervised methods like contrastive learning have become promising in this field. They learn meaningful representations by pulling positive pairs (two augmented versions of the same instance) closer and pushing negative pairs (different from the positive pairs) apart. In SLR, in a sign video, only certain parts provide information that is truly useful for its recognition. Applying contrastive methods to SLR raises two issues: (i) contrastive learning methods treat all parts of a video in the same way, without taking into account the relevance of certain parts over others; (ii) shared movements between different signs make negative pairs highly similar, complicating sign discrimination. These issues lead to learning non-discriminative features for sign recognition and poor results in downstream tasks. In response, this paper proposes a self-supervised learning framework designed to learn meaningful representations for SLR. This framework consists of two key components designed to work together: (i) a new self-supervised approach with free-negative pairs; (ii) a new data augmentation technique. This approach shows a considerable gain in accuracy compared to several contrastive and self-supervised methods, across linear evaluation, semi-supervised learning, and transferability between sign languages.
>
---
#### [new 042] Semi-supervised Deep Transfer for Regression without Domain Alignment
- **分类: cs.CV**

- **简介: 该论文提出CRAFT方法，解决源数据不可用、标签稀缺下的源无关半监督回归迁移问题。无需域对齐，利用未标记目标数据，在神经科学和基准数据集上提升回归性能，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.05092v1](http://arxiv.org/pdf/2509.05092v1)**

> **作者:** Mainak Biswas; Ambedkar Dukkipati; Devarajan Sridharan
>
> **备注:** 15 pages, 6 figures, International Conference on Computer Vision 2025
>
> **摘要:** Deep learning models deployed in real-world applications (e.g., medicine) face challenges because source models do not generalize well to domain-shifted target data. Many successful domain adaptation (DA) approaches require full access to source data. Yet, such requirements are unrealistic in scenarios where source data cannot be shared either because of privacy concerns or because it is too large and incurs prohibitive storage or computational costs. Moreover, resource constraints may limit the availability of labeled targets. We illustrate this challenge in a neuroscience setting where source data are unavailable, labeled target data are meager, and predictions involve continuous-valued outputs. We build upon Contradistinguisher (CUDA), an efficient framework that learns a shared model across the labeled source and unlabeled target samples, without intermediate representation alignment. Yet, CUDA was designed for unsupervised DA, with full access to source data, and for classification tasks. We develop CRAFT -- a Contradistinguisher-based Regularization Approach for Flexible Training -- for source-free (SF), semi-supervised transfer of pretrained models in regression tasks. We showcase the efficacy of CRAFT in two neuroscience settings: gaze prediction with electroencephalography (EEG) data and ``brain age'' prediction with structural MRI data. For both datasets, CRAFT yielded up to 9% improvement in root-mean-squared error (RMSE) over fine-tuned models when labeled training examples were scarce. Moreover, CRAFT leveraged unlabeled target data and outperformed four competing state-of-the-art source-free domain adaptation models by more than 3%. Lastly, we demonstrate the efficacy of CRAFT on two other real-world regression benchmarks. We propose CRAFT as an efficient approach for source-free, semi-supervised deep transfer for regression that is ubiquitous in biology and medicine.
>
---
#### [new 043] WatchHAR: Real-time On-device Human Activity Recognition System for Smartwatches
- **分类: cs.CV; I.2.10; H.5.2**

- **简介: 该论文提出WatchHAR系统，解决智能手表实时人体活动识别的隐私与延迟问题。通过端到端优化和多模态数据处理，实现高速度（9.3ms事件检测）和高精度（>90%）的活动分类，提升设备端独立运行能力。**

- **链接: [http://arxiv.org/pdf/2509.04736v1](http://arxiv.org/pdf/2509.04736v1)**

> **作者:** Taeyoung Yeon; Vasco Xu; Henry Hoffmann; Karan Ahuja
>
> **备注:** 8 pages, 4 figures, ICMI '25 (27th International Conference on Multimodal Interaction), October 13-17, 2025, Canberra, ACT, Australia
>
> **摘要:** Despite advances in practical and multimodal fine-grained Human Activity Recognition (HAR), a system that runs entirely on smartwatches in unconstrained environments remains elusive. We present WatchHAR, an audio and inertial-based HAR system that operates fully on smartwatches, addressing privacy and latency issues associated with external data processing. By optimizing each component of the pipeline, WatchHAR achieves compounding performance gains. We introduce a novel architecture that unifies sensor data preprocessing and inference into an end-to-end trainable module, achieving 5x faster processing while maintaining over 90% accuracy across more than 25 activity classes. WatchHAR outperforms state-of-the-art models for event detection and activity classification while running directly on the smartwatch, achieving 9.3 ms processing time for activity event detection and 11.8 ms for multimodal activity classification. This research advances on-device activity recognition, realizing smartwatches' potential as standalone, privacy-aware, and minimally-invasive continuous activity tracking devices.
>
---
#### [new 044] Sali4Vid: Saliency-Aware Video Reweighting and Adaptive Caption Retrieval for Dense Video Captioning
- **分类: cs.CV**

- **简介: 该论文针对密集视频字幕生成任务，解决现有方法帧处理不均与场景转换忽略的问题。提出Sali4Vid框架，通过saliency-aware视频重加权和语义自适应字幕检索，提升视频权重与检索效果，在YouCook2和ViTT上取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2509.04602v1](http://arxiv.org/pdf/2509.04602v1)**

> **作者:** MinJu Jeon; Si-Woo Kim; Ye-Chan Kim; HyunGee Kim; Dong-Jin Kim
>
> **备注:** Accepted in EMNLP 2025
>
> **摘要:** Dense video captioning aims to temporally localize events in video and generate captions for each event. While recent works propose end-to-end models, they suffer from two limitations: (1) applying timestamp supervision only to text while treating all video frames equally, and (2) retrieving captions from fixed-size video chunks, overlooking scene transitions. To address these, we propose Sali4Vid, a simple yet effective saliency-aware framework. We introduce Saliency-aware Video Reweighting, which converts timestamp annotations into sigmoid-based frame importance weights, and Semantic-based Adaptive Caption Retrieval, which segments videos by frame similarity to capture scene transitions and improve caption retrieval. Sali4Vid achieves state-of-the-art results on YouCook2 and ViTT, demonstrating the benefit of jointly improving video weighting and retrieval for dense video captioning
>
---
#### [new 045] Exploiting Unlabeled Structures through Task Consistency Training for Versatile Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对多类别医学图像分割中的类别不平衡问题，提出Task Consistency Training框架，通过主分割头与辅助任务头的预测一致性约束，结合过滤策略和不确定性加权损失，有效利用部分标注数据提升分割性能。**

- **链接: [http://arxiv.org/pdf/2509.04732v1](http://arxiv.org/pdf/2509.04732v1)**

> **作者:** Shengqian Zhu; Jiafei Wu; Xiaogang Xu; Chengrong Yu; Ying Song; Zhang Yi; Guangjun Li; Junjie Hu
>
> **摘要:** Versatile medical image segmentation (VMIS) targets the segmentation of multiple classes, while obtaining full annotations for all classes is often impractical due to the time and labor required. Leveraging partially labeled datasets (PLDs) presents a promising alternative; however, current VMIS approaches face significant class imbalance due to the unequal category distribution in PLDs. Existing methods attempt to address this by generating pseudo-full labels. Nevertheless, these typically require additional models and often result in potential performance degradation from label noise. In this work, we introduce a Task Consistency Training (TCT) framework to address class imbalance without requiring extra models. TCT includes a backbone network with a main segmentation head (MSH) for multi-channel predictions and multiple auxiliary task heads (ATHs) for task-specific predictions. By enforcing a consistency constraint between the MSH and ATH predictions, TCT effectively utilizes unlabeled anatomical structures. To avoid error propagation from low-consistency, potentially noisy data, we propose a filtering strategy to exclude such data. Additionally, we introduce a unified auxiliary uncertainty-weighted loss (UAUWL) to mitigate segmentation quality declines caused by the dominance of specific tasks. Extensive experiments on eight abdominal datasets from diverse clinical sites demonstrate our approach's effectiveness.
>
---
#### [new 046] Dynamic Group Detection using VLM-augmented Temporal Groupness Graph
- **分类: cs.CV**

- **简介: 该论文提出动态群体检测方法，解决传统方法无法处理视频中动态变化群体的问题。通过VLM增强特征与全局图优化，实现跨帧群体结构一致性检测，提升复杂场景下群体识别效果。**

- **链接: [http://arxiv.org/pdf/2509.04758v1](http://arxiv.org/pdf/2509.04758v1)**

> **作者:** Kaname Yokoyama; Chihiro Nakatani; Norimichi Ukita
>
> **备注:** 10 pages, Accepted to ICCV2025
>
> **摘要:** This paper proposes dynamic human group detection in videos. For detecting complex groups, not only the local appearance features of in-group members but also the global context of the scene are important. Such local and global appearance features in each frame are extracted using a Vision-Language Model (VLM) augmented for group detection in our method. For further improvement, the group structure should be consistent over time. While previous methods are stabilized on the assumption that groups are not changed in a video, our method detects dynamically changing groups by global optimization using a graph with all frames' groupness probabilities estimated by our groupness-augmented CLIP features. Our experimental results demonstrate that our method outperforms state-of-the-art group detection methods on public datasets. Code: https://github.com/irajisamurai/VLM-GroupDetection.git
>
---
#### [new 047] Domain Adaptation for Different Sensor Configurations in 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶中不同传感器配置导致的3D目标检测领域偏差问题，提出下游微调与部分层微调方法，通过联合训练提升跨配置模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.04711v1](http://arxiv.org/pdf/2509.04711v1)**

> **作者:** Satoshi Tanaka; Kok Seang Tan; Isamu Yamashita
>
> **摘要:** Recent advances in autonomous driving have underscored the importance of accurate 3D object detection, with LiDAR playing a central role due to its robustness under diverse visibility conditions. However, different vehicle platforms often deploy distinct sensor configurations, causing performance degradation when models trained on one configuration are applied to another because of shifts in the point cloud distribution. Prior work on multi-dataset training and domain adaptation for 3D object detection has largely addressed environmental domain gaps and density variation within a single LiDAR; in contrast, the domain gap for different sensor configurations remains largely unexplored. In this work, we address domain adaptation across different sensor configurations in 3D object detection. We propose two techniques: Downstream Fine-tuning (dataset-specific fine-tuning after multi-dataset training) and Partial Layer Fine-tuning (updating only a subset of layers to improve cross-configuration generalization). Using paired datasets collected in the same geographic region with multiple sensor configurations, we show that joint training with Downstream Fine-tuning and Partial Layer Fine-tuning consistently outperforms naive joint training for each configuration. Our findings provide a practical and scalable solution for adapting 3D object detection models to the diverse vehicle platforms.
>
---
#### [new 048] CoRe-GS: Coarse-to-Refined Gaussian Splatting with Semantic Object Focus
- **分类: cs.CV**

- **简介: 该论文提出CoRe-GS方法，针对自主飞行器的移动场景重建任务，解决高质3D重建与快速训练的矛盾。通过语义分割与颜色过滤结合，先粗分割后精修，降低训练时间约25%，提升新视角合成质量。**

- **链接: [http://arxiv.org/pdf/2509.04859v1](http://arxiv.org/pdf/2509.04859v1)**

> **作者:** Hannah Schieber; Dominik Frischmann; Simon Boche; Victor Schaack; Angela Schoellig; Stefan Leutenegger; Daniel Roth
>
> **摘要:** Mobile reconstruction for autonomous aerial robotics holds strong potential for critical applications such as tele-guidance and disaster response. These tasks demand both accurate 3D reconstruction and fast scene processing. Instead of reconstructing the entire scene in detail, it is often more efficient to focus on specific objects, i.e., points of interest (PoIs). Mobile robots equipped with advanced sensing can usually detect these early during data acquisition or preliminary analysis, reducing the need for full-scene optimization. Gaussian Splatting (GS) has recently shown promise in delivering high-quality novel view synthesis and 3D representation by an incremental learning process. Extending GS with scene editing, semantics adds useful per-splat features to isolate objects effectively. Semantic 3D Gaussian editing can already be achieved before the full training cycle is completed, reducing the overall training time. Moreover, the semantically relevant area, the PoI, is usually already known during capturing. To balance high-quality reconstruction with reduced training time, we propose CoRe-GS. We first generate a coarse segmentation-ready scene with semantic GS and then refine it for the semantic object using our novel color-based effective filtering for effective object isolation. This is speeding up the training process to be about a quarter less than a full training cycle for semantic GS. We evaluate our approach on two datasets, SCRREAM (real-world, outdoor) and NeRDS 360 (synthetic, indoor), showing reduced runtime and higher novel-view-synthesis quality.
>
---
#### [new 049] Evaluating Multiple Instance Learning Strategies for Automated Sebocyte Droplet Counting
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对皮脂细胞脂滴计数任务，提出基于注意力机制的多实例学习框架，对比评估了MLP与MIL模型性能，发现简单聚合方法更稳健，而MIL需优化池化与正则化以提升效果。**

- **链接: [http://arxiv.org/pdf/2509.04895v1](http://arxiv.org/pdf/2509.04895v1)**

> **作者:** Maryam Adelipour; Gustavo Carneiro; Jeongkwon Kim
>
> **备注:** 8 pages, 1 figure, 2 tables
>
> **摘要:** Sebocytes are lipid-secreting cells whose differentiation is marked by the accumulation of intracellular lipid droplets, making their quantification a key readout in sebocyte biology. Manual counting is labor-intensive and subjective, motivating automated solutions. Here, we introduce a simple attention-based multiple instance learning (MIL) framework for sebocyte image analysis. Nile Red-stained sebocyte images were annotated into 14 classes according to droplet counts, expanded via data augmentation to about 50,000 cells. Two models were benchmarked: a baseline multi-layer perceptron (MLP) trained on aggregated patch-level counts, and an attention-based MIL model leveraging ResNet-50 features with instance weighting. Experiments using five-fold cross-validation showed that the baseline MLP achieved more stable performance (mean MAE = 5.6) compared with the attention-based MIL, which was less consistent (mean MAE = 10.7) but occasionally superior in specific folds. These findings indicate that simple bag-level aggregation provides a robust baseline for slide-level droplet counting, while attention-based MIL requires task-aligned pooling and regularization to fully realize its potential in sebocyte image analysis.
>
---
#### [new 050] Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 本论文针对3D点云分类任务，解决ModelNet40数据集的标签不一致、2D数据等问题，提出改进数据集ModelNet-R和轻量网络Point-SkipNet，提升分类准确率并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2509.05198v1](http://arxiv.org/pdf/2509.05198v1)**

> **作者:** Mohammad Saeid; Amir Salarpour; Pedram MohajerAnsari
>
> **备注:** This paper has been accepted for presentation at the 7th International Conference on Pattern Recognition and Image Analysis (IPRIA 2025)
>
> **摘要:** The classification of 3D point clouds is crucial for applications such as autonomous driving, robotics, and augmented reality. However, the commonly used ModelNet40 dataset suffers from limitations such as inconsistent labeling, 2D data, size mismatches, and inadequate class differentiation, which hinder model performance. This paper introduces ModelNet-R, a meticulously refined version of ModelNet40 designed to address these issues and serve as a more reliable benchmark. Additionally, this paper proposes Point-SkipNet, a lightweight graph-based neural network that leverages efficient sampling, neighborhood grouping, and skip connections to achieve high classification accuracy with reduced computational overhead. Extensive experiments demonstrate that models trained in ModelNet-R exhibit significant performance improvements. Notably, Point-SkipNet achieves state-of-the-art accuracy on ModelNet-R with a substantially lower parameter count compared to contemporary models. This research highlights the crucial role of dataset quality in optimizing model efficiency for 3D point cloud classification. For more details, see the code at: https://github.com/m-saeid/ModeNetR_PointSkipNet.
>
---
#### [new 051] Interpretable Deep Transfer Learning for Breast Ultrasound Cancer Detection: A Multi-Dataset Study
- **分类: cs.CV**

- **简介: 该论文提出基于深度迁移学习的乳腺超声癌症检测方法，通过多数据集评估传统ML与CNN模型，结合Grad-CAM提升可解释性，旨在提高检测准确性和临床实用性。**

- **链接: [http://arxiv.org/pdf/2509.05004v1](http://arxiv.org/pdf/2509.05004v1)**

> **作者:** Mohammad Abbadi; Yassine Himeur; Shadi Atalla; Wathiq Mansoor
>
> **备注:** 6 pages, 2 figures and 1 table
>
> **摘要:** Breast cancer remains a leading cause of cancer-related mortality among women worldwide. Ultrasound imaging, widely used due to its safety and cost-effectiveness, plays a key role in early detection, especially in patients with dense breast tissue. This paper presents a comprehensive study on the application of machine learning and deep learning techniques for breast cancer classification using ultrasound images. Using datasets such as BUSI, BUS-BRA, and BrEaST-Lesions USG, we evaluate classical machine learning models (SVM, KNN) and deep convolutional neural networks (ResNet-18, EfficientNet-B0, GoogLeNet). Experimental results show that ResNet-18 achieves the highest accuracy (99.7%) and perfect sensitivity for malignant lesions. Classical ML models, though outperformed by CNNs, achieve competitive performance when enhanced with deep feature extraction. Grad-CAM visualizations further improve model transparency by highlighting diagnostically relevant image regions. These findings support the integration of AI-based diagnostic tools into clinical workflows and demonstrate the feasibility of deploying high-performing, interpretable systems for ultrasound-based breast cancer detection.
>
---
#### [new 052] Towards Efficient Pixel Labeling for Industrial Anomaly Detection and Localization
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对工业异常检测与定位任务，解决传统方法依赖大量像素标注导致的扩展性差问题。提出ADClick交互式分割算法和ADClick-Seg跨模态框架，通过用户点击与文本提示生成高效标注，显著提升检测精度（AP=96.1%）和定位性能（Pixel-AUROC=99.1%）。**

- **链接: [http://arxiv.org/pdf/2509.05034v1](http://arxiv.org/pdf/2509.05034v1)**

> **作者:** Jingqi Wu; Hanxi Li; Lin Yuanbo Wu; Hao Chen; Deyin Liu; Peng Wang
>
> **摘要:** Industrial product inspection is often performed using Anomaly Detection (AD) frameworks trained solely on non-defective samples. Although defective samples can be collected during production, leveraging them usually requires pixel-level annotations, limiting scalability. To address this, we propose ADClick, an Interactive Image Segmentation (IIS) algorithm for industrial anomaly detection. ADClick generates pixel-wise anomaly annotations from only a few user clicks and a brief textual description, enabling precise and efficient labeling that significantly improves AD model performance (e.g., AP = 96.1\% on MVTec AD). We further introduce ADClick-Seg, a cross-modal framework that aligns visual features and textual prompts via a prototype-based approach for anomaly detection and localization. By combining pixel-level priors with language-guided cues, ADClick-Seg achieves state-of-the-art results on the challenging ``Multi-class'' AD task (AP = 80.0\%, PRO = 97.5\%, Pixel-AUROC = 99.1\% on MVTec AD).
>
---
#### [new 053] Inferring the Graph Structure of Images for Graph Neural Networks
- **分类: eess.IV; cs.CV; cs.LG; eess.SP**

- **简介: 该论文旨在改进图像的图结构表示，以提升图神经网络的分类性能。通过构建基于像素相关性的行、列和乘积图，替代传统网格图和超像素方法，实验表明新方法在MNIST和Fashion-MNIST数据集上提高了准确性。**

- **链接: [http://arxiv.org/pdf/2509.04677v1](http://arxiv.org/pdf/2509.04677v1)**

> **作者:** Mayur S Gowda; John Shi; Augusto Santos; José M. F. Moura
>
> **摘要:** Image datasets such as MNIST are a key benchmark for testing Graph Neural Network (GNN) architectures. The images are traditionally represented as a grid graph with each node representing a pixel and edges connecting neighboring pixels (vertically and horizontally). The graph signal is the values (intensities) of each pixel in the image. The graphs are commonly used as input to graph neural networks (e.g., Graph Convolutional Neural Networks (Graph CNNs) [1, 2], Graph Attention Networks (GAT) [3], GatedGCN [4]) to classify the images. In this work, we improve the accuracy of downstream graph neural network tasks by finding alternative graphs to the grid graph and superpixel methods to represent the dataset images, following the approach in [5, 6]. We find row correlation, column correlation, and product graphs for each image in MNIST and Fashion-MNIST using correlations between the pixel values building on the method in [5, 6]. Experiments show that using these different graph representations and features as input into downstream GNN models improves the accuracy over using the traditional grid graph and superpixel methods in the literature.
>
---
#### [new 054] Beyond I-Con: Exploring New Dimension of Distance Measures in Representation Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对表示学习中KL散度的局限性，提出Beyond I-Con框架，探索替代统计分歧和相似性核。通过引入TV距离和f-散度，在聚类、对比学习和降维任务中提升性能。**

- **链接: [http://arxiv.org/pdf/2509.04734v1](http://arxiv.org/pdf/2509.04734v1)**

> **作者:** Jasmine Shone; Shaden Alshammari; Mark Hamilton; Zhening Li; William Freeman
>
> **摘要:** The Information Contrastive (I-Con) framework revealed that over 23 representation learning methods implicitly minimize KL divergence between data and learned distributions that encode similarities between data points. However, a KL-based loss may be misaligned with the true objective, and properties of KL divergence such as asymmetry and unboundedness may create optimization challenges. We present Beyond I-Con, a framework that enables systematic discovery of novel loss functions by exploring alternative statistical divergences and similarity kernels. Key findings: (1) on unsupervised clustering of DINO-ViT embeddings, we achieve state-of-the-art results by modifying the PMI algorithm to use total variation (TV) distance; (2) on supervised contrastive learning, we outperform the standard approach by using TV and a distance-based similarity kernel instead of KL and an angular kernel; (3) on dimensionality reduction, we achieve superior qualitative results and better performance on downstream tasks than SNE by replacing KL with a bounded f-divergence. Our results highlight the importance of considering divergence and similarity kernel choices in representation learning optimization.
>
---
#### [new 055] Sample-efficient Integration of New Modalities into Large Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出SEMI方法，解决低资源模态集成效率低的问题。通过超网络动态适配投影器，结合等距变换扩展编码器多样性，显著降低新模态集成所需数据量，提升大模型多模态扩展能力。**

- **链接: [http://arxiv.org/pdf/2509.04606v1](http://arxiv.org/pdf/2509.04606v1)**

> **作者:** Osman Batur İnce; André F. T. Martins; Oisin Mac Aodha; Edoardo M. Ponti
>
> **备注:** Pre-print
>
> **摘要:** Multimodal foundation models can process several modalities. However, since the space of possible modalities is large and evolving over time, training a model from scratch to encompass all modalities is unfeasible. Moreover, integrating a modality into a pre-existing foundation model currently requires a significant amount of paired data, which is often not available for low-resource modalities. In this paper, we introduce a method for sample-efficient modality integration (SEMI) into Large Language Models (LLMs). To this end, we devise a hypernetwork that can adapt a shared projector -- placed between modality-specific encoders and an LLM -- to any modality. The hypernetwork, trained on high-resource modalities (i.e., text, speech, audio, video), is conditioned on a few samples from any arbitrary modality at inference time to generate a suitable adapter. To increase the diversity of training modalities, we artificially multiply the number of encoders through isometric transformations. We find that SEMI achieves a significant boost in sample efficiency during few-shot integration of new modalities (i.e., satellite images, astronomical images, inertial measurements, and molecules) with encoders of arbitrary embedding dimensionality. For instance, to reach the same accuracy as 32-shot SEMI, training the projector from scratch needs 64$\times$ more data. As a result, SEMI holds promise to extend the modality coverage of foundation models.
>
---
#### [new 056] VLSM-Ensemble: Ensembling CLIP-based Vision-Language Models for Enhanced Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对医学图像分割任务，提出集成CLIP-based视觉-语言模型与低复杂度CNN的方法，提升分割精度。在BKAI数据集上Dice得分提升6.3%，并在多个数据集验证集成效果差异，为未来研究提供方向。**

- **链接: [http://arxiv.org/pdf/2509.05154v1](http://arxiv.org/pdf/2509.05154v1)**

> **作者:** Julia Dietlmeier; Oluwabukola Grace Adegboro; Vayangi Ganepola; Claudia Mazo; Noel E. O'Connor
>
> **备注:** Medical Imaging with Deep Learning (MIDL 2025) short paper
>
> **摘要:** Vision-language models and their adaptations to image segmentation tasks present enormous potential for producing highly accurate and interpretable results. However, implementations based on CLIP and BiomedCLIP are still lagging behind more sophisticated architectures such as CRIS. In this work, instead of focusing on text prompt engineering as is the norm, we attempt to narrow this gap by showing how to ensemble vision-language segmentation models (VLSMs) with a low-complexity CNN. By doing so, we achieve a significant Dice score improvement of 6.3% on the BKAI polyp dataset using the ensembled BiomedCLIPSeg, while other datasets exhibit gains ranging from 1% to 6%. Furthermore, we provide initial results on additional four radiology and non-radiology datasets. We conclude that ensembling works differently across these datasets (from outperforming to underperforming the CRIS model), indicating a topic for future investigation by the community. The code is available at https://github.com/juliadietlmeier/VLSM-Ensemble.
>
---
#### [new 057] PRIM: Towards Practical In-Image Multilingual Machine Translation
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出PRIM数据集与VisTrans模型，解决图像内多语言翻译任务中合成数据与真实场景差距的问题，提升多语言翻译质量与视觉效果。**

- **链接: [http://arxiv.org/pdf/2509.05146v1](http://arxiv.org/pdf/2509.05146v1)**

> **作者:** Yanzhi Tian; Zeming Liu; Zhengyang Liu; Chong Feng; Xin Li; Heyan Huang; Yuhang Guo
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** In-Image Machine Translation (IIMT) aims to translate images containing texts from one language to another. Current research of end-to-end IIMT mainly conducts on synthetic data, with simple background, single font, fixed text position, and bilingual translation, which can not fully reflect real world, causing a significant gap between the research and practical conditions. To facilitate research of IIMT in real-world scenarios, we explore Practical In-Image Multilingual Machine Translation (IIMMT). In order to convince the lack of publicly available data, we annotate the PRIM dataset, which contains real-world captured one-line text images with complex background, various fonts, diverse text positions, and supports multilingual translation directions. We propose an end-to-end model VisTrans to handle the challenge of practical conditions in PRIM, which processes visual text and background information in the image separately, ensuring the capability of multilingual translation while improving the visual quality. Experimental results indicate the VisTrans achieves a better translation quality and visual effect compared to other models. The code and dataset are available at: https://github.com/BITHLP/PRIM.
>
---
#### [new 058] Histogram Driven Amplitude Embedding for Qubit Efficient Quantum Image Compression
- **分类: quant-ph; cs.CV; cs.ET; cs.IT; math.IT**

- **简介: 该论文提出基于直方图驱动的幅度嵌入方法，用于量子图像压缩。通过分块计算强度、构建直方图并编码至量子态，实现在NISQ设备上高效压缩，减少量子比特需求，优于传统像素级编码。**

- **链接: [http://arxiv.org/pdf/2509.04849v1](http://arxiv.org/pdf/2509.04849v1)**

> **作者:** Sahil Tomar; Sandeep Kumar
>
> **备注:** 7 pages
>
> **摘要:** This work introduces a compact and hardware efficient method for compressing color images using near term quantum devices. The approach segments the image into fixed size blocks called bixels, and computes the total intensity within each block. A global histogram with B bins is then constructed from these block intensities, and the normalized square roots of the bin counts are encoded as amplitudes into an n qubit quantum state. Amplitude embedding is performed using PennyLane and executed on real IBM Quantum hardware. The resulting state is measured to reconstruct the histogram, enabling approximate recovery of block intensities and full image reassembly. The method maintains a constant qubit requirement based solely on the number of histogram bins, independent of the resolution of the image. By adjusting B, users can control the trade off between fidelity and resource usage. Empirical results demonstrate high quality reconstructions using as few as 5 to 7 qubits, significantly outperforming conventional pixel level encodings in terms of qubit efficiency and validating the practical application of the method for current NISQ era quantum systems.
>
---
#### [new 059] Phonological Representation Learning for Isolated Signs Improves Out-of-Vocabulary Generalization
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对手语孤立词识别的未见词汇泛化问题，提出结合参数解耦与语音半监督的向量量化自编码器，提升单次重建与识别效果。**

- **链接: [http://arxiv.org/pdf/2509.04745v1](http://arxiv.org/pdf/2509.04745v1)**

> **作者:** Lee Kezar; Zed Sehyr; Jesse Thomason
>
> **摘要:** Sign language datasets are often not representative in terms of vocabulary, underscoring the need for models that generalize to unseen signs. Vector quantization is a promising approach for learning discrete, token-like representations, but it has not been evaluated whether the learned units capture spurious correlations that hinder out-of-vocabulary performance. This work investigates two phonological inductive biases: Parameter Disentanglement, an architectural bias, and Phonological Semi-Supervision, a regularization technique, to improve isolated sign recognition of known signs and reconstruction quality of unseen signs with a vector-quantized autoencoder. The primary finding is that the learned representations from the proposed model are more effective for one-shot reconstruction of unseen signs and more discriminative for sign identification compared to a controlled baseline. This work provides a quantitative analysis of how explicit, linguistically-motivated biases can improve the generalization of learned representations of sign language.
>
---
#### [new 060] Ecologically Valid Benchmarking and Adaptive Attention: Scalable Marine Bioacoustic Monitoring
- **分类: cs.SD; cs.AI; cs.CV; cs.IR; cs.LG; eess.AS**

- **简介: 该论文提出GetNetUPAM框架和ARPA-N网络，解决海洋生物声学监测中噪声干扰和信号依赖问题，提升模型稳定性与泛化能力，实现高精度、可扩展的生态监测。**

- **链接: [http://arxiv.org/pdf/2509.04682v1](http://arxiv.org/pdf/2509.04682v1)**

> **作者:** Nicholas R. Rasmussen; Rodrigue Rizk; Longwei Wang; KC Santosh
>
> **备注:** Under review as an anonymous submission to IEEETAI - We are allowed an archive submission. Final formatting is yet to be determined
>
> **摘要:** Underwater Passive Acoustic Monitoring (UPAM) provides rich spatiotemporal data for long-term ecological analysis, but intrinsic noise and complex signal dependencies hinder model stability and generalization. Multilayered windowing has improved target sound localization, yet variability from shifting ambient noise, diverse propagation effects, and mixed biological and anthropogenic sources demands robust architectures and rigorous evaluation. We introduce GetNetUPAM, a hierarchical nested cross-validation framework designed to quantify model stability under ecologically realistic variability. Data are partitioned into distinct site-year segments, preserving recording heterogeneity and ensuring each validation fold reflects a unique environmental subset, reducing overfitting to localized noise and sensor artifacts. Site-year blocking enforces evaluation against genuine environmental diversity, while standard cross-validation on random subsets measures generalization across UPAM's full signal distribution, a dimension absent from current benchmarks. Using GetNetUPAM as the evaluation backbone, we propose the Adaptive Resolution Pooling and Attention Network (ARPA-N), a neural architecture for irregular spectrogram dimensions. Adaptive pooling with spatial attention extends the receptive field, capturing global context without excessive parameters. Under GetNetUPAM, ARPA-N achieves a 14.4% gain in average precision over DenseNet baselines and a log2-scale order-of-magnitude drop in variability across all metrics, enabling consistent detection across site-year folds and advancing scalable, accurate bioacoustic monitoring.
>
---
#### [new 061] Multi-modal Uncertainty Robust Tree Cover Segmentation For High-Resolution Remote Sensing Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出MURTreeFormer框架，解决多模态遥感图像树覆盖分割中的跨模态不确定性问题。通过概率表示建模辅助模态不确定性，结合VAE重采样与注意力机制，提升分割鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.04870v1](http://arxiv.org/pdf/2509.04870v1)**

> **作者:** Yuanyuan Gui; Wei Li; Yinjian Wang; Xiang-Gen Xia; Mauro Marty; Christian Ginzler; Zuyuan Wang
>
> **摘要:** Recent advances in semantic segmentation of multi-modal remote sensing images have significantly improved the accuracy of tree cover mapping, supporting applications in urban planning, forest monitoring, and ecological assessment. Integrating data from multiple modalities-such as optical imagery, light detection and ranging (LiDAR), and synthetic aperture radar (SAR)-has shown superior performance over single-modality methods. However, these data are often acquired days or even months apart, during which various changes may occur, such as vegetation disturbances (e.g., logging, and wildfires) and variations in imaging quality. Such temporal misalignments introduce cross-modal uncertainty, especially in high-resolution imagery, which can severely degrade segmentation accuracy. To address this challenge, we propose MURTreeFormer, a novel multi-modal segmentation framework that mitigates and leverages aleatoric uncertainty for robust tree cover mapping. MURTreeFormer treats one modality as primary and others as auxiliary, explicitly modeling patch-level uncertainty in the auxiliary modalities via a probabilistic latent representation. Uncertain patches are identified and reconstructed from the primary modality's distribution through a VAE-based resampling mechanism, producing enhanced auxiliary features for fusion. In the decoder, a gradient magnitude attention (GMA) module and a lightweight refinement head (RH) are further integrated to guide attention toward tree-like structures and to preserve fine-grained spatial details. Extensive experiments on multi-modal datasets from Shanghai and Zurich demonstrate that MURTreeFormer significantly improves segmentation performance and effectively reduces the impact of temporally induced aleatoric uncertainty.
>
---
#### [new 062] Robust Model Predictive Control Design for Autonomous Vehicles with Perception-based Observers
- **分类: cs.RO; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文提出一种鲁棒MPC框架，针对自动驾驶车辆感知模块的非高斯噪声问题，结合约束区间状态估计与线性规划优化，确保稳定性，实验验证其在重尾噪声下的控制性能优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.05201v1](http://arxiv.org/pdf/2509.05201v1)**

> **作者:** Nariman Niknejad; Gokul S. Sankar; Bahare Kiumarsi; Hamidreza Modares
>
> **摘要:** This paper presents a robust model predictive control (MPC) framework that explicitly addresses the non-Gaussian noise inherent in deep learning-based perception modules used for state estimation. Recognizing that accurate uncertainty quantification of the perception module is essential for safe feedback control, our approach departs from the conventional assumption of zero-mean noise quantification of the perception error. Instead, it employs set-based state estimation with constrained zonotopes to capture biased, heavy-tailed uncertainties while maintaining bounded estimation errors. To improve computational efficiency, the robust MPC is reformulated as a linear program (LP), using a Minkowski-Lyapunov-based cost function with an added slack variable to prevent degenerate solutions. Closed-loop stability is ensured through Minkowski-Lyapunov inequalities and contractive zonotopic invariant sets. The largest stabilizing terminal set and its corresponding feedback gain are then derived via an ellipsoidal approximation of the zonotopes. The proposed framework is validated through both simulations and hardware experiments on an omnidirectional mobile robot along with a camera and a convolutional neural network-based perception module implemented within a ROS2 framework. The results demonstrate that the perception-aware MPC provides stable and accurate control performance under heavy-tailed noise conditions, significantly outperforming traditional Gaussian-noise-based designs in terms of both state estimation error bounding and overall control performance.
>
---
#### [new 063] SparkUI-Parser: Enhancing GUI Perception with Robust Grounding and Parsing
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文提出SparkUI-Parser，解决GUI感知中定位不准确、无法全面解析界面的问题。通过连续坐标建模与拒绝机制提升精度与速度，并构建ScreenParse基准测试，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2509.04908v1](http://arxiv.org/pdf/2509.04908v1)**

> **作者:** Hongyi Jing; Jiafu Chen; Chen Rao; Ziqiang Dang; Jiajie Teng; Tianyi Chu; Juncheng Mo; Shuo Fang; Huaizhong Lin; Rui Lv; Chenguang Ma; Lei Zhao
>
> **摘要:** The existing Multimodal Large Language Models (MLLMs) for GUI perception have made great progress. However, the following challenges still exist in prior methods: 1) They model discrete coordinates based on text autoregressive mechanism, which results in lower grounding accuracy and slower inference speed. 2) They can only locate predefined sets of elements and are not capable of parsing the entire interface, which hampers the broad application and support for downstream tasks. To address the above issues, we propose SparkUI-Parser, a novel end-to-end framework where higher localization precision and fine-grained parsing capability of the entire interface are simultaneously achieved. Specifically, instead of using probability-based discrete modeling, we perform continuous modeling of coordinates based on a pre-trained Multimodal Large Language Model (MLLM) with an additional token router and coordinate decoder. This effectively mitigates the limitations inherent in the discrete output characteristics and the token-by-token generation process of MLLMs, consequently boosting both the accuracy and the inference speed. To further enhance robustness, a rejection mechanism based on a modified Hungarian matching algorithm is introduced, which empowers the model to identify and reject non-existent elements, thereby reducing false positives. Moreover, we present ScreenParse, a rigorously constructed benchmark to systematically assess structural perception capabilities of GUI models across diverse scenarios. Extensive experiments demonstrate that our approach consistently outperforms SOTA methods on ScreenSpot, ScreenSpot-v2, CAGUI-Grounding and ScreenParse benchmarks. The resources are available at https://github.com/antgroup/SparkUI-Parser.
>
---
#### [new 064] Towards an Accurate and Effective Robot Vision (The Problem of Topological Localization for Mobile Robots)
- **分类: cs.RO; cs.CV**

- **简介: 论文针对移动机器人拓扑定位问题，通过比较多种视觉描述符（如SIFT、Bag-of-Visual-Words）在办公室环境中的性能，优化配置以提升定位准确性与鲁棒性，适应不同光照条件。**

- **链接: [http://arxiv.org/pdf/2509.04948v1](http://arxiv.org/pdf/2509.04948v1)**

> **作者:** Emanuela Boros
>
> **备注:** Master's thesis
>
> **摘要:** Topological localization is a fundamental problem in mobile robotics, since robots must be able to determine their position in order to accomplish tasks. Visual localization and place recognition are challenging due to perceptual ambiguity, sensor noise, and illumination variations. This work addresses topological localization in an office environment using only images acquired with a perspective color camera mounted on a robot platform, without relying on temporal continuity of image sequences. We evaluate state-of-the-art visual descriptors, including Color Histograms, SIFT, ASIFT, RGB-SIFT, and Bag-of-Visual-Words approaches inspired by text retrieval. Our contributions include a systematic, quantitative comparison of these features, distance measures, and classifiers. Performance was analyzed using standard evaluation metrics and visualizations, extending previous experiments. Results demonstrate the advantages of proper configurations of appearance descriptors, similarity measures, and classifiers. The quality of these configurations was further validated in the Robot Vision task of the ImageCLEF evaluation campaign, where the system identified the most likely location of novel image sequences. Future work will explore hierarchical models, ranking methods, and feature combinations to build more robust localization systems, reducing training and runtime while avoiding the curse of dimensionality. Ultimately, this aims toward integrated, real-time localization across varied illumination and longer routes.
>
---
#### [new 065] Improved 3D Scene Stylization via Text-Guided Generative Image Editing with Region-Based Control
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出文本引导的3D场景风格化方法，解决风格一致性与视图一致性问题。通过改进深度条件生成框架和多区域加权损失函数，实现区域可控的风格迁移。**

- **链接: [http://arxiv.org/pdf/2509.05285v1](http://arxiv.org/pdf/2509.05285v1)**

> **作者:** Haruo Fujiwara; Yusuke Mukuta; Tatsuya Harada
>
> **摘要:** Recent advances in text-driven 3D scene editing and stylization, which leverage the powerful capabilities of 2D generative models, have demonstrated promising outcomes. However, challenges remain in ensuring high-quality stylization and view consistency simultaneously. Moreover, applying style consistently to different regions or objects in the scene with semantic correspondence is a challenging task. To address these limitations, we introduce techniques that enhance the quality of 3D stylization while maintaining view consistency and providing optional region-controlled style transfer. Our method achieves stylization by re-training an initial 3D representation using stylized multi-view 2D images of the source views. Therefore, ensuring both style consistency and view consistency of stylized multi-view images is crucial. We achieve this by extending the style-aligned depth-conditioned view generation framework, replacing the fully shared attention mechanism with a single reference-based attention-sharing mechanism, which effectively aligns style across different viewpoints. Additionally, inspired by recent 3D inpainting methods, we utilize a grid of multiple depth maps as a single-image reference to further strengthen view consistency among stylized images. Finally, we propose Multi-Region Importance-Weighted Sliced Wasserstein Distance Loss, allowing styles to be applied to distinct image regions using segmentation masks from off-the-shelf models. We demonstrate that this optional feature enhances the faithfulness of style transfer and enables the mixing of different styles across distinct regions of the scene. Experimental evaluations, both qualitative and quantitative, demonstrate that our pipeline effectively improves the results of text-driven 3D stylization.
>
---
#### [new 066] Pointing-Guided Target Estimation via Transformer-Based Attention
- **分类: cs.RO; cs.AI; cs.CV; I.2.9; I.2.10; I.2.6**

- **简介: 该论文提出MM-ITF模型，通过Transformer注意力机制将指向手势与视觉数据结合，解决人机协作中机器人理解人类意图的目标预测问题，提升交互准确性。**

- **链接: [http://arxiv.org/pdf/2509.05031v1](http://arxiv.org/pdf/2509.05031v1)**

> **作者:** Luca Müller; Hassan Ali; Philipp Allgeuer; Lukáš Gajdošech; Stefan Wermter
>
> **备注:** Accepted at the 34th International Conference on Artificial Neural Networks (ICANN) 2025,12 pages,4 figures,1 table; work was co-funded by Horizon Europe project TERAIS under Grant agreement number 101079338
>
> **摘要:** Deictic gestures, like pointing, are a fundamental form of non-verbal communication, enabling humans to direct attention to specific objects or locations. This capability is essential in Human-Robot Interaction (HRI), where robots should be able to predict human intent and anticipate appropriate responses. In this work, we propose the Multi-Modality Inter-TransFormer (MM-ITF), a modular architecture to predict objects in a controlled tabletop scenario with the NICOL robot, where humans indicate targets through natural pointing gestures. Leveraging inter-modality attention, MM-ITF maps 2D pointing gestures to object locations, assigns a likelihood score to each, and identifies the most likely target. Our results demonstrate that the method can accurately predict the intended object using monocular RGB data, thus enabling intuitive and accessible human-robot collaboration. To evaluate the performance, we introduce a patch confusion matrix, providing insights into the model's predictions across candidate object locations. Code available at: https://github.com/lucamuellercode/MMITF.
>
---
#### [new 067] LatticeWorld: A Multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出LatticeWorld框架，解决传统3D建模效率低、质量不足问题，通过多模态输入（文本/视觉）结合轻量LLM与渲染引擎，实现高效、高保真交互式复杂世界生成。**

- **链接: [http://arxiv.org/pdf/2509.05263v1](http://arxiv.org/pdf/2509.05263v1)**

> **作者:** Yinglin Duan; Zhengxia Zou; Tongwei Gu; Wei Jia; Zhan Zhao; Luyi Xu; Xinzhu Liu; Hao Jiang; Kang Chen; Shuang Qiu
>
> **摘要:** Recent research has been increasingly focusing on developing 3D world models that simulate complex real-world scenarios. World models have found broad applications across various domains, including embodied AI, autonomous driving, entertainment, etc. A more realistic simulation with accurate physics will effectively narrow the sim-to-real gap and allow us to gather rich information about the real world conveniently. While traditional manual modeling has enabled the creation of virtual 3D scenes, modern approaches have leveraged advanced machine learning algorithms for 3D world generation, with most recent advances focusing on generative methods that can create virtual worlds based on user instructions. This work explores such a research direction by proposing LatticeWorld, a simple yet effective 3D world generation framework that streamlines the industrial production pipeline of 3D environments. LatticeWorld leverages lightweight LLMs (LLaMA-2-7B) alongside the industry-grade rendering engine (e.g., Unreal Engine 5) to generate a dynamic environment. Our proposed framework accepts textual descriptions and visual instructions as multimodal inputs and creates large-scale 3D interactive worlds with dynamic agents, featuring competitive multi-agent interaction, high-fidelity physics simulation, and real-time rendering. We conduct comprehensive experiments to evaluate LatticeWorld, showing that it achieves superior accuracy in scene layout generation and visual fidelity. Moreover, LatticeWorld achieves over a $90\times$ increase in industrial production efficiency while maintaining high creative quality compared with traditional manual production methods. Our demo video is available at https://youtu.be/8VWZXpERR18
>
---
#### [new 068] STADI: Fine-Grained Step-Patch Diffusion Parallelism for Heterogeneous GPUs
- **分类: cs.DC; cs.CV**

- **简介: 该论文提出STADI框架，解决异构多GPU环境下扩散模型推理的负载不平衡与资源利用率低问题。通过时空混合调度，实现细粒度步级与补丁级并行，优化计算分配与同步，显著降低推理延迟并提升异构GPU利用率。**

- **链接: [http://arxiv.org/pdf/2509.04719v1](http://arxiv.org/pdf/2509.04719v1)**

> **作者:** Han Liang; Jiahui Zhou; Zicheng Zhou; Xiaoxi Zhang; Xu Chen
>
> **摘要:** The escalating adoption of diffusion models for applications such as image generation demands efficient parallel inference techniques to manage their substantial computational cost. However, existing diffusion parallelism inference schemes often underutilize resources in heterogeneous multi-GPU environments, where varying hardware capabilities or background tasks cause workload imbalance. This paper introduces Spatio-Temporal Adaptive Diffusion Inference (STADI), a novel framework to accelerate diffusion model inference in such settings. At its core is a hybrid scheduler that orchestrates fine-grained parallelism across both temporal and spatial dimensions. Temporally, STADI introduces a novel computation-aware step allocator applied after warmup phases, using a least-common-multiple-minimizing quantization technique to reduce denoising steps on slower GPUs and execution synchronization. To further minimize GPU idle periods, STADI executes an elastic patch parallelism mechanism that allocates variably sized image patches to GPUs according to their computational capability, ensuring balanced workload distribution through a complementary spatial mechanism. Extensive experiments on both load-imbalanced and heterogeneous multi-GPU clusters validate STADI's efficacy, demonstrating improved load balancing and mitigation of performance bottlenecks. Compared to patch parallelism, a state-of-the-art diffusion inference framework, our method significantly reduces end-to-end inference latency by up to 45% and significantly improves resource utilization on heterogeneous GPUs.
>
---
#### [new 069] AURAD: Anatomy-Pathology Unified Radiology Synthesis with Progressive Representations
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出AURAD框架，解决医学图像合成中可控性差、多病理共存及解剖-病理一致性问题。通过渐进式生成伪掩码与高保真X光片，结合预训练模型确保临床合理性，生成的图像与掩码兼具视觉真实性和临床价值。**

- **链接: [http://arxiv.org/pdf/2509.04819v1](http://arxiv.org/pdf/2509.04819v1)**

> **作者:** Shuhan Ding; Jingjing Fu; Yu Gu; Naiteek Sangani; Mu Wei; Paul Vozila; Nan Liu; Jiang Bian; Hoifung Poon
>
> **摘要:** Medical image synthesis has become an essential strategy for augmenting datasets and improving model generalization in data-scarce clinical settings. However, fine-grained and controllable synthesis remains difficult due to limited high-quality annotations and domain shifts across datasets. Existing methods, often designed for natural images or well-defined tumors, struggle to generalize to chest radiographs, where disease patterns are morphologically diverse and tightly intertwined with anatomical structures. To address these challenges, we propose AURAD, a controllable radiology synthesis framework that jointly generates high-fidelity chest X-rays and pseudo semantic masks. Unlike prior approaches that rely on randomly sampled masks-limiting diversity, controllability, and clinical relevance-our method learns to generate masks that capture multi-pathology coexistence and anatomical-pathological consistency. It follows a progressive pipeline: pseudo masks are first generated from clinical prompts conditioned on anatomical structures, and then used to guide image synthesis. We also leverage pretrained expert medical models to filter outputs and ensure clinical plausibility. Beyond visual realism, the synthesized masks also serve as labels for downstream tasks such as detection and segmentation, bridging the gap between generative modeling and real-world clinical applications. Extensive experiments and blinded radiologist evaluations demonstrate the effectiveness and generalizability of our method across tasks and datasets. In particular, 78% of our synthesized images are classified as authentic by board-certified radiologists, and over 40% of predicted segmentation overlays are rated as clinically useful. All code, pre-trained models, and the synthesized dataset will be released upon publication.
>
---
## 更新

#### [replaced 001] GCRPNet: Graph-Enhanced Contextual and Regional Perception Network For Salient Object Detection in Optical Remote Sensing Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10542v2](http://arxiv.org/pdf/2508.10542v2)**

> **作者:** Mengyu Ren; Yutong Li; Hua Li; Runmin Cong; Sam Kwong
>
> **摘要:** Salient object detection (SOD) in optical remote sensing images (ORSIs) faces numerous challenges, including significant variations in target scales and low contrast between targets and the background. Existing methods based on vision transformers (ViTs) and convolutional neural networks (CNNs) architectures aim to leverage both global and local features, but the difficulty in effectively integrating these heterogeneous features limits their overall performance. To overcome these limitations, we propose a graph-enhanced contextual and regional perception network (GCRPNet), which builds upon the Mamba architecture to simultaneously capture long-range dependencies and enhance regional feature representation. Specifically, we employ the visual state space (VSS) encoder to extract multi-scale features. To further achieve deep guidance and enhancement of these features, we first design a difference-similarity guided hierarchical graph attention module (DS-HGAM). This module strengthens cross-layer interaction capabilities between features of different scales while enhancing the model's structural perception,allowing it to distinguish between foreground and background more effectively. Then, we design the LEVSS block as the decoder of GCRPNet. This module integrates our proposed adaptive scanning strategy and multi-granularity collaborative attention enhancement module (MCAEM). It performs adaptive patch scanning on feature maps processed via multi-scale convolutions, thereby capturing rich local region information and enhancing Mamba's local modeling capability. Extensive experimental results demonstrate that the proposed model achieves state-of-the-art performance, validating its effectiveness and superiority.
>
---
#### [replaced 002] RailGoerl24: Görlitz Rail Test Center CV Dataset 2024
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.00204v2](http://arxiv.org/pdf/2504.00204v2)**

> **作者:** Rustam Tagiew; Ilkay Wunderlich; Mark Sastuba; Kilian Göller; Steffen Seitz
>
> **备注:** 4 pages, 5 figures, presented at Engineering Reliable Autonomous Systems 2025
>
> **摘要:** Driverless train operation for open tracks on urban guided transport and mainline railways requires, among other things automatic detection of actual and potential obstacles, especially humans, in the danger zone of the train's path. Machine learning algorithms have proven to be powerful state-of-the-art tools for this task. However, these algorithms require large amounts of high-quality annotated data containing human beings in railway-specific environments as training data. Unfortunately, the amount of publicly available datasets is not yet sufficient and is significantly inferior to the datasets in the road domain. Therefore, this paper presents RailGoerl24, an on-board visual light Full HD camera dataset of 12205 frames recorded in a railway test center of T\"UV S\"UD Rail, in G\"orlitz, Germany. Its main purpose is to support the development of driverless train operation for guided transport. RailGoerl24 also includes a terrestrial LiDAR scan covering parts of the area used to acquire the RGB data. In addition to the raw data, the dataset contains 33556 boxwise annotations in total for the object class 'person'. The faces of recorded actors are not blurred or altered in any other way. RailGoerl24, available at data.fid-move.de/dataset/railgoerl24, can also be used for tasks beyond collision prediction.
>
---
#### [replaced 003] Disentangled Clothed Avatar Generation with Layered Representation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.04631v2](http://arxiv.org/pdf/2501.04631v2)**

> **作者:** Weitian Zhang; Yichao Yan; Sijing Wu; Manwen Liao; Xiaokang Yang
>
> **备注:** ICCV 2025 highlight, project page: https://olivia23333.github.io/LayerAvatar/
>
> **摘要:** Clothed avatar generation has wide applications in virtual and augmented reality, filmmaking, and more. Previous methods have achieved success in generating diverse digital avatars, however, generating avatars with disentangled components (\eg, body, hair, and clothes) has long been a challenge. In this paper, we propose LayerAvatar, the first feed-forward diffusion-based method for generating component-disentangled clothed avatars. To achieve this, we first propose a layered UV feature plane representation, where components are distributed in different layers of the Gaussian-based UV feature plane with corresponding semantic labels. This representation supports high-resolution and real-time rendering, as well as expressive animation including controllable gestures and facial expressions. Based on the well-designed representation, we train a single-stage diffusion model and introduce constrain terms to address the severe occlusion problem of the innermost human body layer. Extensive experiments demonstrate the impressive performances of our method in generating disentangled clothed avatars, and we further explore its applications in component transfer. The project page is available at: https://olivia23333.github.io/LayerAvatar/
>
---
#### [replaced 004] Global-to-Local or Local-to-Global? Enhancing Image Retrieval with Efficient Local Search and Effective Global Re-ranking
- **分类: cs.IR; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04351v2](http://arxiv.org/pdf/2509.04351v2)**

> **作者:** Dror Aiger; Bingyi Cao; Kaifeng Chen; Andre Araujo
>
> **摘要:** The dominant paradigm in image retrieval systems today is to search large databases using global image features, and re-rank those initial results with local image feature matching techniques. This design, dubbed global-to-local, stems from the computational cost of local matching approaches, which can only be afforded for a small number of retrieved images. However, emerging efficient local feature search approaches have opened up new possibilities, in particular enabling detailed retrieval at large scale, to find partial matches which are often missed by global feature search. In parallel, global feature-based re-ranking has shown promising results with high computational efficiency. In this work, we leverage these building blocks to introduce a local-to-global retrieval paradigm, where efficient local feature search meets effective global feature re-ranking. Critically, we propose a re-ranking method where global features are computed on-the-fly, based on the local feature retrieval similarities. Such re-ranking-only global features leverage multidimensional scaling techniques to create embeddings which respect the local similarities obtained during search, enabling a significant re-ranking boost. Experimentally, we demonstrate solid retrieval performance, setting new state-of-the-art results on the Revisited Oxford and Paris datasets.
>
---
#### [replaced 005] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20309v2](http://arxiv.org/pdf/2503.20309v2)**

> **作者:** Zitian Wang; Yue Liao; Kang Rong; Fengyun Rao; Yibo Yang; Si Liu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Preference alignment has emerged as an effective strategy to enhance the performance of Multimodal Large Language Models (MLLMs) following supervised fine-tuning. While existing preference alignment methods predominantly target hallucination factors, they overlook the factors essential for multi-modal comprehension capabilities, often narrowing their improvements on hallucination mitigation. To bridge this gap, we propose Instruction-oriented Preference Alignment (IPA), a scalable framework designed to automatically construct alignment preferences grounded in instruction fulfillment efficacy. Our method involves an automated preference construction coupled with a dedicated verification process that identifies instruction-oriented factors, avoiding significant variability in response representations. Additionally, IPA incorporates a progressive preference collection pipeline, further recalling challenging samples through model self-evolution and reference-guided refinement. Experiments conducted on Qwen2VL-7B demonstrate IPA's effectiveness across multiple benchmarks, including hallucination evaluation, visual question answering, and text understanding tasks, highlighting its capability to enhance general comprehension.
>
---
#### [replaced 006] RAVEN: Query-Guided Representation Alignment for Question Answering over Audio, Video, Embedded Sensors, and Natural Language
- **分类: cs.CL; cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.17114v3](http://arxiv.org/pdf/2505.17114v3)**

> **作者:** Subrata Biswas; Mohammad Nur Hossain Khan; Bashima Islam
>
> **摘要:** Multimodal question answering (QA) often requires identifying which video, audio, or sensor tokens are relevant to the question. Yet modality disagreements are common: off-camera speech, background noise, or motion outside the field of view often mislead fusion models that weight all streams equally. We present RAVEN, a unified QA architecture whose core is QuART, a query-conditioned cross-modal gating module that assigns scalar relevance scores to each token across modalities, enabling the model to amplify informative signals and suppress distractors before fusion. RAVEN is trained through a three-stage pipeline comprising unimodal pretraining, query-aligned fusion, and disagreement-oriented fine-tuning -- each stage targeting a distinct challenge in multi-modal reasoning: representation quality, cross-modal relevance, and robustness to modality mismatch. To support training and evaluation, we release AVS-QA, a dataset of 300K synchronized Audio--Video-Sensor streams paired with automatically generated question-answer pairs. Experimental results on seven multi-modal QA benchmarks -- including egocentric and exocentric tasks -- show that RAVEN achieves up to 14.5\% and 8.0\% gains in accuracy compared to state-of-the-art multi-modal large language models, respectively. Incorporating sensor data provides an additional 16.4\% boost, and the model remains robust under modality corruption, outperforming SOTA baselines by 50.23\%. Our code and dataset are available at https://github.com/BASHLab/RAVEN.
>
---
#### [replaced 007] Towards High-Fidelity, Identity-Preserving Real-Time Makeup Transfer: Decoupling Style Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02445v2](http://arxiv.org/pdf/2509.02445v2)**

> **作者:** Lydia Kin Ching Chau; Zhi Yu; Ruowei Jiang
>
> **摘要:** We present a novel framework for real-time virtual makeup try-on that achieves high-fidelity, identity-preserving cosmetic transfer with robust temporal consistency. In live makeup transfer applications, it is critical to synthesize temporally coherent results that accurately replicate fine-grained makeup and preserve user's identity. However, existing methods often struggle to disentangle semitransparent cosmetics from skin tones and other identify features, causing identity shifts and raising fairness concerns. Furthermore, current methods lack real-time capabilities and fail to maintain temporal consistency, limiting practical adoption. To address these challenges, we decouple makeup transfer into two steps: transparent makeup mask extraction and graphics-based mask rendering. After the makeup extraction step, the makeup rendering can be performed in real time, enabling live makeup try-on. Our makeup extraction model trained on pseudo-ground-truth data generated via two complementary methods: a graphics-based rendering pipeline and an unsupervised k-means clustering approach. To further enhance transparency estimation and color fidelity, we propose specialized training objectives, including alpha-weighted reconstruction and lip color losses. Our method achieves robust makeup transfer across diverse poses, expressions, and skin tones while preserving temporal smoothness. Extensive experiments demonstrate that our approach outperforms existing baselines in capturing fine details, maintaining temporal stability, and preserving identity integrity.
>
---
#### [replaced 008] ArtRAG: Retrieval-Augmented Generation with Structured Context for Visual Art Understanding
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06020v2](http://arxiv.org/pdf/2505.06020v2)**

> **作者:** Shuai Wang; Ivona Najdenkoska; Hongyi Zhu; Stevan Rudinac; Monika Kackovic; Nachoem Wijnberg; Marcel Worring
>
> **摘要:** Understanding visual art requires reasoning across multiple perspectives -- cultural, historical, and stylistic -- beyond mere object recognition. While recent multimodal large language models (MLLMs) perform well on general image captioning, they often fail to capture the nuanced interpretations that fine art demands. We propose ArtRAG, a novel, training-free framework that combines structured knowledge with retrieval-augmented generation (RAG) for multi-perspective artwork explanation. ArtRAG automatically constructs an Art Context Knowledge Graph (ACKG) from domain-specific textual sources, organizing entities such as artists, movements, themes, and historical events into a rich, interpretable graph. At inference time, a multi-granular structured retriever selects semantically and topologically relevant subgraphs to guide generation. This enables MLLMs to produce contextually grounded, culturally informed art descriptions. Experiments on the SemArt and Artpedia datasets show that ArtRAG outperforms several heavily trained baselines. Human evaluations further confirm that ArtRAG generates coherent, insightful, and culturally enriched interpretations.
>
---
#### [replaced 009] UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2509.02544v2](http://arxiv.org/pdf/2509.02544v2)**

> **作者:** Haoming Wang; Haoyang Zou; Huatong Song; Jiazhan Feng; Junjie Fang; Junting Lu; Longxiang Liu; Qinyu Luo; Shihao Liang; Shijue Huang; Wanjun Zhong; Yining Ye; Yujia Qin; Yuwen Xiong; Yuxin Song; Zhiyong Wu; Aoyan Li; Bo Li; Chen Dun; Chong Liu; Daoguang Zan; Fuxing Leng; Hanbin Wang; Hao Yu; Haobin Chen; Hongyi Guo; Jing Su; Jingjia Huang; Kai Shen; Kaiyu Shi; Lin Yan; Peiyao Zhao; Pengfei Liu; Qinghao Ye; Renjie Zheng; Shulin Xin; Wayne Xin Zhao; Wen Heng; Wenhao Huang; Wenqian Wang; Xiaobo Qin; Yi Lin; Youbin Wu; Zehui Chen; Zihao Wang; Baoquan Zhong; Xinchun Zhang; Xujing Li; Yuanfan Li; Zhongkai Zhao; Chengquan Jiang; Faming Wu; Haotian Zhou; Jinlin Pang; Li Han; Qi Liu; Qianli Ma; Siyao Liu; Songhua Cai; Wenqi Fu; Xin Liu; Yaohui Wang; Zhi Zhang; Bo Zhou; Guoliang Li; Jiajun Shi; Jiale Yang; Jie Tang; Li Li; Qihua Han; Taoran Lu; Woyu Lin; Xiaokang Tong; Xinyao Li; Yichi Zhang; Yu Miao; Zhengxuan Jiang; Zili Li; Ziyuan Zhao; Chenxin Li; Dehua Ma; Feng Lin; Ge Zhang; Haihua Yang; Hangyu Guo; Hongda Zhu; Jiaheng Liu; Junda Du; Kai Cai; Kuanye Li; Lichen Yuan; Meilan Han; Minchao Wang; Shuyue Guo; Tianhao Cheng; Xiaobo Ma; Xiaojun Xiao; Xiaolong Huang; Xinjie Chen; Yidi Du; Yilin Chen; Yiwen Wang; Zhaojian Li; Zhenzhu Yang; Zhiyuan Zeng; Chaolin Jin; Chen Li; Hao Chen; Haoli Chen; Jian Chen; Qinghao Zhao; Guang Shi
>
> **摘要:** The development of autonomous agents for graphical user interfaces (GUIs) presents major challenges in artificial intelligence. While recent advances in native agent models have shown promise by unifying perception, reasoning, action, and memory through end-to-end learning, open problems remain in data scalability, multi-turn reinforcement learning (RL), the limitations of GUI-only operation, and environment stability. In this technical report, we present UI-TARS-2, a native GUI-centered agent model that addresses these challenges through a systematic training methodology: a data flywheel for scalable data generation, a stabilized multi-turn RL framework, a hybrid GUI environment that integrates file systems and terminals, and a unified sandbox platform for large-scale rollouts. Empirical evaluation demonstrates that UI-TARS-2 achieves significant improvements over its predecessor UI-TARS-1.5. On GUI benchmarks, it reaches 88.2 on Online-Mind2Web, 47.5 on OSWorld, 50.6 on WindowsAgentArena, and 73.3 on AndroidWorld, outperforming strong baselines such as Claude and OpenAI agents. In game environments, it attains a mean normalized score of 59.8 across a 15-game suite-roughly 60% of human-level performance-and remains competitive with frontier proprietary models (e.g., OpenAI o3) on LMGame-Bench. Additionally, the model can generalize to long-horizon information-seeking tasks and software engineering benchmarks, highlighting its robustness across diverse agent tasks. Detailed analyses of training dynamics further provide insights into achieving stability and efficiency in large-scale agent RL. These results underscore UI-TARS-2's potential to advance the state of GUI agents and exhibit strong generalization to real-world interactive scenarios.
>
---
#### [replaced 010] MitoDetect++: A Domain-Robust Pipeline for Mitosis Detection and Atypical Subtyping
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02586v2](http://arxiv.org/pdf/2509.02586v2)**

> **作者:** Esha Sadia Nasir; Jiaqi Lv; Mostafa Jahanifar; Shan E Ahmed Raza
>
> **摘要:** Automated detection and classification of mitotic figures especially distinguishing atypical from normal remain critical challenges in computational pathology. We present MitoDetect++, a unified deep learning pipeline designed for the MIDOG 2025 challenge, addressing both mitosis detection and atypical mitosis classification. For detection (Track 1), we employ a U-Net-based encoder-decoder architecture with EfficientNetV2-L as the backbone, enhanced with attention modules, and trained via combined segmentation losses. For classification (Track 2), we leverage the Virchow2 vision transformer, fine-tuned efficiently using Low-Rank Adaptation (LoRA) to minimize resource consumption. To improve generalization and mitigate domain shifts, we integrate strong augmentations, focal loss, and group-aware stratified 5-fold cross-validation. At inference, we deploy test-time augmentation (TTA) to boost robustness. Our method achieves a balanced accuracy of 0.892 across validation domains, highlighting its clinical applicability and scalability across tasks.
>
---
#### [replaced 011] MultiStream-LLM: Bridging Modalities for Robust Sign Language Translation
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00030v2](http://arxiv.org/pdf/2509.00030v2)**

> **作者:** Marshall Thomas; Edward Fish; Richard Bowden
>
> **摘要:** Despite progress in gloss-free Sign Language Translation (SLT), monolithic end-to-end models consistently fail on two critical components of natural signing: the precise recognition of high-speed fingerspelling and the integration of asynchronous non-manual cues from the face. Recent progress in Automated Sign Language Translation with Large Language Models has side stepped this challenge, forcing a single network to learn these simultaneously resulting in poor performance when tasked with translating crucial information such as names,places, and technical terms. We introduce MultiStream-LLM, a modular framework designed to overcome these limitations. Our approach employs separate, specialized predictors for continuous signing, fingerspelling, and lipreading. Each expert network first decodes its specific modality into a sequence of tokens. These parallel streams are then fused by a lightweight transformer that resolves temporal misalignments before passing the combined representation to a Large Language Model (LLM) for final sentence generation. Our method establishes a new state-of-the-art on the How2Sign benchmark with a BLEU-4 score of 23.5 and achieves 73.2% letter accuracy on the challenging ChicagoFSWildPlus fingerspelling dataset. These results validate our core hypothesis: by isolating and solving distinct recogni tion tasks before fusion, our multi-expert approach provides a more powerful and effective pathway to robust, high-fidelity sign language translation.
>
---
#### [replaced 012] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17784v2](http://arxiv.org/pdf/2411.17784v2)**

> **作者:** Lingxiao Li; Kaixuan Fan; Boqing Gong; Xiangyu Yue
>
> **备注:** ICCV 2025, Code is available at: https://github.com/lingxiao-li/HypDAE
>
> **摘要:** Few-shot image generation aims to generate diverse and high-quality images for an unseen class given only a few examples in that class. A key challenge in this task is balancing category consistency and image diversity, which often compete with each other. Moreover, existing methods offer limited control over the attributes of newly generated images. In this work, we propose Hyperbolic Diffusion Autoencoders (HypDAE), a novel approach that operates in hyperbolic space to capture hierarchical relationships among images from seen categories. By leveraging pre-trained foundation models, HypDAE generates diverse new images for unseen categories with exceptional quality by varying stochastic subcodes or semantic codes. Most importantly, the hyperbolic representation introduces an additional degree of control over semantic diversity through the adjustment of radii within the hyperbolic disk. Extensive experiments and visualizations demonstrate that HypDAE significantly outperforms prior methods by achieving a better balance between preserving category-relevant features and promoting image diversity with limited data. Furthermore, HypDAE offers a highly controllable and interpretable generation process.
>
---
#### [replaced 013] InfoScale: Unleashing Training-free Variable-scaled Image Generation via Effective Utilization of Information
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01421v2](http://arxiv.org/pdf/2509.01421v2)**

> **作者:** Guohui Zhang; Jiangtong Tan; Linjiang Huang; Zhonghang Yuan; Naishan Zheng; Jie Huang; Feng Zhao
>
> **摘要:** Diffusion models (DMs) have become dominant in visual generation but suffer performance drop when tested on resolutions that differ from the training scale, whether lower or higher. In fact, the key challenge in generating variable-scale images lies in the differing amounts of information across resolutions, which requires information conversion procedures to be varied for generating variable-scaled images. In this paper, we investigate the issues of three critical aspects in DMs for a unified analysis in variable-scaled generation: dilated convolution, attention mechanisms, and initial noise. Specifically, 1) dilated convolution in DMs for the higher-resolution generation loses high-frequency information. 2) Attention for variable-scaled image generation struggles to adjust the information aggregation adaptively. 3) The spatial distribution of information in the initial noise is misaligned with variable-scaled image. To solve the above problems, we propose \textbf{InfoScale}, an information-centric framework for variable-scaled image generation by effectively utilizing information from three aspects correspondingly. For information loss in 1), we introduce Progressive Frequency Compensation module to compensate for high-frequency information lost by dilated convolution in higher-resolution generation. For information aggregation inflexibility in 2), we introduce Adaptive Information Aggregation module to adaptively aggregate information in lower-resolution generation and achieve an effective balance between local and global information in higher-resolution generation. For information distribution misalignment in 3), we design Noise Adaptation module to re-distribute information in initial noise for variable-scaled generation. Our method is plug-and-play for DMs and extensive experiments demonstrate the effectiveness in variable-scaled image generation.
>
---
#### [replaced 014] Aesthetic Image Captioning with Saliency Enhanced MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04378v2](http://arxiv.org/pdf/2509.04378v2)**

> **作者:** Yilin Tao; Jiashui Huang; Huaze Xu; Ling Shao
>
> **摘要:** Aesthetic Image Captioning (AIC) aims to generate textual descriptions of image aesthetics, becoming a key research direction in the field of computational aesthetics. In recent years, pretrained Multimodal Large Language Models (MLLMs) have advanced rapidly, leading to a significant increase in image aesthetics research that integrates both visual and textual modalities. However, most existing studies on image aesthetics primarily focus on predicting aesthetic ratings and have shown limited application in AIC. Existing AIC works leveraging MLLMs predominantly rely on fine-tuning methods without specifically adapting MLLMs to focus on target aesthetic content. To address this limitation, we propose the Aesthetic Saliency Enhanced Multimodal Large Language Model (ASE-MLLM), an end-to-end framework that explicitly incorporates aesthetic saliency into MLLMs. Within this framework, we introduce the Image Aesthetic Saliency Module (IASM), which efficiently and effectively extracts aesthetic saliency features from images. Additionally, we design IAS-ViT as the image encoder for MLLMs, this module fuses aesthetic saliency features with original image features via a cross-attention mechanism. To the best of our knowledge, ASE-MLLM is the first framework to integrate image aesthetic saliency into MLLMs specifically for AIC tasks. Extensive experiments demonstrated that our approach significantly outperformed traditional methods and generic MLLMs on current mainstream AIC benchmarks, achieving state-of-the-art (SOTA) performance.
>
---
#### [replaced 015] YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17733v2](http://arxiv.org/pdf/2506.17733v2)**

> **作者:** Mengqi Lei; Siqi Li; Yihong Wu; Han Hu; You Zhou; Xinhu Zheng; Guiguang Ding; Shaoyi Du; Zongze Wu; Yue Gao
>
> **摘要:** The YOLO series models reign supreme in real-time object detection due to their superior accuracy and computational efficiency. However, both the convolutional architectures of YOLO11 and earlier versions and the area-based self-attention mechanism introduced in YOLOv12 are limited to local information aggregation and pairwise correlation modeling, lacking the capability to capture global multi-to-multi high-order correlations, which limits detection performance in complex scenarios. In this paper, we propose YOLOv13, an accurate and lightweight object detector. To address the above-mentioned challenges, we propose a Hypergraph-based Adaptive Correlation Enhancement (HyperACE) mechanism that adaptively exploits latent high-order correlations and overcomes the limitation of previous methods that are restricted to pairwise correlation modeling based on hypergraph computation, achieving efficient global cross-location and cross-scale feature fusion and enhancement. Subsequently, we propose a Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm based on HyperACE, which effectively achieves fine-grained information flow and representation synergy within the entire network by distributing correlation-enhanced features to the full pipeline. Finally, we propose to leverage depthwise separable convolutions to replace vanilla large-kernel convolutions, and design a series of blocks that significantly reduce parameters and computational complexity without sacrificing performance. We conduct extensive experiments on the widely used MS COCO benchmark, and the experimental results demonstrate that our method achieves state-of-the-art performance with fewer parameters and FLOPs. Specifically, our YOLOv13-N improves mAP by 3.0\% over YOLO11-N and by 1.5\% over YOLOv12-N. The code and models of our YOLOv13 model are available at: https://github.com/iMoonLab/yolov13.
>
---
#### [replaced 016] PromptGuard: Soft Prompt-Guided Unsafe Content Moderation for Text-to-Image Models
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2501.03544v3](http://arxiv.org/pdf/2501.03544v3)**

> **作者:** Lingzhi Yuan; Xinfeng Li; Chejian Xu; Guanhong Tao; Xiaojun Jia; Yihao Huang; Wei Dong; Yang Liu; Bo Li
>
> **备注:** 15 pages, 8 figures, 14 tables
>
> **摘要:** Recent text-to-image (T2I) models have exhibited remarkable performance in generating high-quality images from text descriptions. However, these models are vulnerable to misuse, particularly generating not-safe-for-work (NSFW) content, such as sexually explicit, violent, political, and disturbing images, raising serious ethical concerns. In this work, we present PromptGuard, a novel content moderation technique that draws inspiration from the system prompt mechanism in large language models (LLMs) for safety alignment. Unlike LLMs, T2I models lack a direct interface for enforcing behavioral guidelines. Our key idea is to optimize a safety soft prompt that functions as an implicit system prompt within the T2I model's textual embedding space. This universal soft prompt (P*) directly moderates NSFW inputs, enabling safe yet realistic image generation without altering the inference efficiency or requiring proxy models. We further enhance its reliability and helpfulness through a divide-and-conquer strategy, which optimizes category-specific soft prompts and combines them into holistic safety guidance. Extensive experiments across five datasets demonstrate that PromptGuard effectively mitigates NSFW content generation while preserving high-quality benign outputs. PromptGuard achieves 3.8 times faster than prior content moderation methods, surpassing eight state-of-the-art defenses with an optimal unsafe ratio down to 5.84%.
>
---
#### [replaced 017] Towards Interpretable Geo-localization: a Concept-Aware Global Image-GPS Alignment Framework
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.01910v2](http://arxiv.org/pdf/2509.01910v2)**

> **作者:** Furong Jia; Lanxin Liu; Ce Hou; Fan Zhang; Xinyan Liu; Yu Liu
>
> **摘要:** Worldwide geo-localization involves determining the exact geographic location of images captured globally, typically guided by geographic cues such as climate, landmarks, and architectural styles. Despite advancements in geo-localization models like GeoCLIP, which leverages images and location alignment via contrastive learning for accurate predictions, the interpretability of these models remains insufficiently explored. Current concept-based interpretability methods fail to align effectively with Geo-alignment image-location embedding objectives, resulting in suboptimal interpretability and performance. To address this gap, we propose a novel framework integrating global geo-localization with concept bottlenecks. Our method inserts a Concept-Aware Alignment Module that jointly projects image and location embeddings onto a shared bank of geographic concepts (e.g., tropical climate, mountain, cathedral) and minimizes a concept-level loss, enhancing alignment in a concept-specific subspace and enabling robust interpretability. To our knowledge, this is the first work to introduce interpretability into geo-localization. Extensive experiments demonstrate that our approach surpasses GeoCLIP in geo-localization accuracy and boosts performance across diverse geospatial prediction tasks, revealing richer semantic insights into geographic decision-making processes.
>
---
#### [replaced 018] Drawing2CAD: Sequence-to-Sequence Learning for CAD Generation from Vector Drawings
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.18733v4](http://arxiv.org/pdf/2508.18733v4)**

> **作者:** Feiwei Qin; Shichao Lu; Junhao Hou; Changmiao Wang; Meie Fang; Ligang Liu
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** Computer-Aided Design (CAD) generative modeling is driving significant innovations across industrial applications. Recent works have shown remarkable progress in creating solid models from various inputs such as point clouds, meshes, and text descriptions. However, these methods fundamentally diverge from traditional industrial workflows that begin with 2D engineering drawings. The automatic generation of parametric CAD models from these 2D vector drawings remains underexplored despite being a critical step in engineering design. To address this gap, our key insight is to reframe CAD generation as a sequence-to-sequence learning problem where vector drawing primitives directly inform the generation of parametric CAD operations, preserving geometric precision and design intent throughout the transformation process. We propose Drawing2CAD, a framework with three key technical components: a network-friendly vector primitive representation that preserves precise geometric information, a dual-decoder transformer architecture that decouples command type and parameter generation while maintaining precise correspondence, and a soft target distribution loss function accommodating inherent flexibility in CAD parameters. To train and evaluate Drawing2CAD, we create CAD-VGDrawing, a dataset of paired engineering drawings and parametric CAD models, and conduct thorough experiments to demonstrate the effectiveness of our method. Code and dataset are available at https://github.com/lllssc/Drawing2CAD.
>
---
#### [replaced 019] Histo-Miner: Deep Learning based Tissue Features Extraction Pipeline from H&E Whole Slide Images of Cutaneous Squamous Cell Carcinoma
- **分类: cs.CV; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2505.04672v2](http://arxiv.org/pdf/2505.04672v2)**

> **作者:** Lucas Sancéré; Carina Lorenz; Doris Helbig; Oana-Diana Persa; Sonja Dengler; Alexander Kreuter; Martim Laimer; Anne Fröhlich; Jennifer Landsberg; Johannes Brägelmann; Katarzyna Bozek
>
> **备注:** 31 pages including supplement, 5 core figures, 5 supplement figures. Version 2: change sections order, add new supplementary sections, minor text updates
>
> **摘要:** Recent advancements in digital pathology have enabled comprehensive analysis of Whole-Slide Images (WSI) from tissue samples, leveraging high-resolution microscopy and computational capabilities. Despite this progress, there is a lack of labeled datasets and open source pipelines specifically tailored for analysis of skin tissue. Here we propose Histo-Miner, a deep learning-based pipeline for analysis of skin WSIs and generate two datasets with labeled nuclei and tumor regions. We develop our pipeline for the analysis of patient samples of cutaneous squamous cell carcinoma (cSCC), a frequent non-melanoma skin cancer. Utilizing the two datasets, comprising 47,392 annotated cell nuclei and 144 tumor-segmented WSIs respectively, both from cSCC patients, Histo-Miner employs convolutional neural networks and vision transformers for nucleus segmentation and classification as well as tumor region segmentation. Performance of trained models positively compares to state of the art with multi-class Panoptic Quality (mPQ) of 0.569 for nucleus segmentation, macro-averaged F1 of 0.832 for nucleus classification and mean Intersection over Union (mIoU) of 0.884 for tumor region segmentation. From these predictions we generate a compact feature vector summarizing tissue morphology and cellular interactions, which can be used for various downstream tasks. Here, we use Histo-Miner to predict cSCC patient response to immunotherapy based on pre-treatment WSIs from 45 patients. Histo-Miner identifies percentages of lymphocytes, the granulocyte to lymphocyte ratio in tumor vicinity and the distances between granulocytes and plasma cells in tumors as predictive features for therapy response. This highlights the applicability of Histo-Miner to clinically relevant scenarios, providing direct interpretation of the classification and insights into the underlying biology.
>
---
#### [replaced 020] FADE: A Dataset for Detecting Falling Objects around Buildings in Video
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.05750v3](http://arxiv.org/pdf/2408.05750v3)**

> **作者:** Zhigang Tu; Zhengbo Zhang; Zitao Gao; Chunluan Zhou; Junsong Yuan; Bo Du
>
> **备注:** Accepted by IEEE Transactions on Information Forensics and Security (TIFS), 2025
>
> **摘要:** Objects falling from buildings, a frequently occurring event in daily life, can cause severe injuries to pedestrians due to the high impact force they exert. Surveillance cameras are often installed around buildings to detect falling objects, but such detection remains challenging due to the small size and fast motion of the objects. Moreover, the field of falling object detection around buildings (FODB) lacks a large-scale dataset for training learning-based detection methods and for standardized evaluation. To address these challenges, we propose a large and diverse video benchmark dataset named FADE. Specifically, FADE contains 2,611 videos from 25 scenes, featuring 8 falling object categories, 4 weather conditions, and 4 video resolutions. Additionally, we develop a novel detection method for FODB that effectively leverages motion information and generates small-sized yet high-quality detection proposals. The efficacy of our method is evaluated on the proposed FADE dataset by comparing it with state-of-the-art approaches in generic object detection, video object detection, and moving object detection. The dataset and code are publicly available at https://fadedataset.github.io/FADE.github.io/.
>
---
#### [replaced 021] DiMo-GUI: Advancing Test-time Scaling in GUI Grounding via Modality-Aware Visual Reasoning
- **分类: cs.AI; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.00008v2](http://arxiv.org/pdf/2507.00008v2)**

> **作者:** Hang Wu; Hongkai Chen; Yujun Cai; Chang Liu; Qingwen Ye; Ming-Hsuan Yang; Yiwei Wang
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Grounding natural language queries in graphical user interfaces (GUIs) poses unique challenges due to the diversity of visual elements, spatial clutter, and the ambiguity of language. In this paper, we introduce DiMo-GUI, a training-free framework for GUI grounding that leverages two core strategies: dynamic visual grounding and modality-aware optimization. Instead of treating the GUI as a monolithic image, our method splits the input into textual elements and iconic elements, allowing the model to reason over each modality independently using general-purpose vision-language models. When predictions are ambiguous or incorrect, DiMo-GUI dynamically focuses attention by generating candidate focal regions centered on the model's initial predictions and incrementally zooms into subregions to refine the grounding result. This hierarchical refinement process helps disambiguate visually crowded layouts without the need for additional training or annotations. We evaluate our approach on standard GUI grounding benchmarks and demonstrate consistent improvements over baseline inference pipelines, highlighting the effectiveness of combining modality separation with region-focused reasoning.
>
---
#### [replaced 022] AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.12226v4](http://arxiv.org/pdf/2402.12226v4)**

> **作者:** Jun Zhan; Junqi Dai; Jiasheng Ye; Yunhua Zhou; Dong Zhang; Zhigeng Liu; Xin Zhang; Ruibin Yuan; Ge Zhang; Linyang Li; Hang Yan; Jie Fu; Tao Gui; Tianxiang Sun; Yugang Jiang; Xipeng Qiu
>
> **备注:** 28 pages, 16 figures, under review, work in progress
>
> **摘要:** We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. Experimental results demonstrate that AnyGPT is capable of facilitating any-to-any multimodal conversation while achieving performance comparable to specialized models across all modalities, proving that discrete representations can effectively and conveniently unify multiple modalities within a language model. Demos are shown in https://junzhan2000.github.io/AnyGPT.github.io/
>
---
#### [replaced 023] GeoArena: An Open Platform for Benchmarking Large Vision-language Models on WorldWide Image Geolocalization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04334v2](http://arxiv.org/pdf/2509.04334v2)**

> **作者:** Pengyue Jia; Yingyi Zhang; Xiangyu Zhao; Yixuan Li
>
> **摘要:** Image geolocalization aims to predict the geographic location of images captured anywhere on Earth, but its global nature presents significant challenges. Current evaluation methodologies suffer from two major limitations. First, data leakage: advanced approaches often rely on large vision-language models (LVLMs) to predict image locations, yet these models are frequently pretrained on the test datasets, compromising the accuracy of evaluating a model's actual geolocalization capability. Second, existing metrics primarily rely on exact geographic coordinates to assess predictions, which not only neglects the reasoning process but also raises privacy concerns when user-level location data is required. To address these issues, we propose GeoArena, a first open platform for evaluating LVLMs on worldwide image geolocalization tasks, offering true in-the-wild and human-centered benchmarking. GeoArena enables users to upload in-the-wild images for a more diverse evaluation corpus, and it leverages pairwise human judgments to determine which model output better aligns with human expectations. Our platform has been deployed online for two months, during which we collected over thousands voting records. Based on this data, we conduct a detailed analysis and establish a leaderboard of different LVLMs on the image geolocalization task.
>
---
#### [replaced 024] Food safety trends across Europe: insights from the 392-million-entry CompreHensive European Food Safety (CHEFS) database
- **分类: cs.CY; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13802v2](http://arxiv.org/pdf/2507.13802v2)**

> **作者:** Nehir Kizililsoley; Floor van Meer; Osman Mutlu; Wouter F Hoenderdaal; Rosan G. Hobé; Wenjuan Mu; Arjen Gerssen; H. J. van der Fels-Klerx; Ákos Jóźwiak; Ioannis Manikas; Ali Hürriyetoǧlu; Bas H. M. van der Velden
>
> **摘要:** In the European Union, official food safety monitoring data collected by member states are submitted to the European Food Safety Authority (EFSA) and published on Zenodo. This data includes 392 million analytical results derived from over 15.2 million samples covering more than 4,000 different types of food products, offering great opportunities for artificial intelligence to analyze trends, predict hazards, and support early warning systems. However, the current format with data distributed across approximately 1000 files totaling several hundred gigabytes hinders accessibility and analysis. To address this, we introduce the CompreHensive European Food Safety (CHEFS) database, which consolidates EFSA monitoring data on pesticide residues, veterinary medicinal product residues, and chemical contaminants into a unified and structured dataset. We describe the creation and structure of the CHEFS database and demonstrate its potential by analyzing trends in European food safety monitoring data from 2000 to 2024. Our analyses explore changes in monitoring activities, the most frequently tested products, which products were most often non-compliant and which contaminants were most often found, and differences across countries. These findings highlight the CHEFS database as both a centralized data source and a strategic tool for guiding food safety policy, research, and regulation.
>
---
#### [replaced 025] Improving Vessel Segmentation with Multi-Task Learning and Auxiliary Data Available Only During Model Training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.03975v2](http://arxiv.org/pdf/2509.03975v2)**

> **作者:** Daniel Sobotka; Alexander Herold; Matthias Perkonigg; Lucian Beer; Nina Bastati; Alina Sablatnig; Ahmed Ba-Ssalamah; Georg Langs
>
> **摘要:** Liver vessel segmentation in magnetic resonance imaging data is important for the computational analysis of vascular remodelling, associated with a wide spectrum of diffuse liver diseases. Existing approaches rely on contrast enhanced imaging data, but the necessary dedicated imaging sequences are not uniformly acquired. Images without contrast enhancement are acquired more frequently, but vessel segmentation is challenging, and requires large-scale annotated data. We propose a multi-task learning framework to segment vessels in liver MRI without contrast. It exploits auxiliary contrast enhanced MRI data available only during training to reduce the need for annotated training examples. Our approach draws on paired native and contrast enhanced data with and without vessel annotations for model training. Results show that auxiliary data improves the accuracy of vessel segmentation, even if they are not available during inference. The advantage is most pronounced if only few annotations are available for training, since the feature representation benefits from the shared task structure. A validation of this approach to augment a model for brain tumor segmentation confirms its benefits across different domains. An auxiliary informative imaging modality can augment expert annotations even if it is only available during training.
>
---
#### [replaced 026] Unveiling the Response of Large Vision-Language Models to Visually Absent Tokens
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.03025v2](http://arxiv.org/pdf/2509.03025v2)**

> **作者:** Sohee Kim; Soohyun Ryu; Joonhyung Park; Eunho Yang
>
> **备注:** accepted to EMNLP 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) generate contextually relevant responses by jointly interpreting visual and textual inputs. However, our finding reveals they often mistakenly perceive text inputs lacking visual evidence as being part of the image, leading to erroneous responses. In light of this finding, we probe whether LVLMs possess an internal capability to determine if textual concepts are grounded in the image, and discover a specific subset of Feed-Forward Network (FFN) neurons, termed Visual Absence-aware (VA) neurons, that consistently signal the visual absence through a distinctive activation pattern. Leveraging these patterns, we develop a detection module that systematically classifies whether an input token is visually grounded. Guided by its prediction, we propose a method to refine the outputs by reinterpreting question prompts or replacing the detected absent tokens during generation. Extensive experiments show that our method effectively mitigates the models' tendency to falsely presume the visual presence of text input and its generality across various LVLMs.
>
---
#### [replaced 027] RS-TinyNet: Stage-wise Feature Fusion Network for Detecting Tiny Objects in Remote Sensing Images
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.13120v2](http://arxiv.org/pdf/2507.13120v2)**

> **作者:** Xiaozheng Jiang; Wei Zhang; Xuerui Mao
>
> **备注:** The content of the thesis requires supplementation to make it more substantial
>
> **摘要:** Detecting tiny objects in remote sensing (RS) imagery has been a long-standing challenge due to their extremely limited spatial information, weak feature representations, and dense distributions across complex backgrounds. Despite numerous efforts devoted, mainstream detectors still underperform in such scenarios. To bridge this gap, we introduce RS-TinyNet, a multi-stage feature fusion and enhancement model explicitly tailored for RS tiny object detection in various RS scenarios. RS-TinyNet comes with two novel designs: tiny object saliency modeling and feature integrity reconstruction. Guided by these principles, we design three step-wise feature enhancement modules. Among them, the multi-dimensional collaborative attention (MDCA) module employs multi-dimensional attention to enhance the saliency of tiny objects. Additionally, the auxiliary reversible branch (ARB) and a progressive fusion detection head (PFDH) module are introduced to preserve information flow and fuse multi-level features to bridge semantic gaps and retain structural detail. Comprehensive experiments on public RS dataset AI-TOD show that our RS-TinyNet surpasses existing state-of-the-art (SOTA) detectors by 4.0% AP and 6.5% AP75. Evaluations on DIOR benchmark dataset further validate its superior detection performance in diverse RS scenarios. These results demonstrate that the proposed multi-stage feature fusion strategy offers an effective and practical solution for tiny object detection in complex RS environments.
>
---
#### [replaced 028] High-resolution efficient image generation from WiFi CSI using a pretrained latent diffusion model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10605v3](http://arxiv.org/pdf/2506.10605v3)**

> **作者:** Eshan Ramesh; Takayuki Nishio
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** We present LatentCSI, a novel method for generating images of the physical environment from WiFi CSI measurements that leverages a pretrained latent diffusion model (LDM). Unlike prior approaches that rely on complex and computationally intensive techniques such as GANs, our method employs a lightweight neural network to map CSI amplitudes directly into the latent space of an LDM. We then apply the LDM's denoising diffusion model to the latent representation with text-based guidance before decoding using the LDM's pretrained decoder to obtain a high-resolution image. This design bypasses the challenges of pixel-space image generation and avoids the explicit image encoding stage typically required in conventional image-to-image pipelines, enabling efficient and high-quality image synthesis. We validate our approach on two datasets: a wide-band CSI dataset we collected with off-the-shelf WiFi devices and cameras; and a subset of the publicly available MM-Fi dataset. The results demonstrate that LatentCSI outperforms baselines of comparable complexity trained directly on ground-truth images in both computational efficiency and perceptual quality, while additionally providing practical advantages through its unique capacity for text-guided controllability.
>
---
#### [replaced 029] ActiveGAMER: Active GAussian Mapping through Efficient Rendering
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.06897v3](http://arxiv.org/pdf/2501.06897v3)**

> **作者:** Liyan Chen; Huangying Zhan; Kevin Chen; Xiangyu Xu; Qingan Yan; Changjiang Cai; Yi Xu
>
> **备注:** Accepted to CVPR2025. Project page: https://oppo-us-research.github.io/ActiveGAMER-website/. Code: https://github.com/oppo-us-research/ActiveGAMER
>
> **摘要:** We introduce ActiveGAMER, an active mapping system that utilizes 3D Gaussian Splatting (3DGS) to achieve high-quality, real-time scene mapping and exploration. Unlike traditional NeRF-based methods, which are computationally demanding and restrict active mapping performance, our approach leverages the efficient rendering capabilities of 3DGS, allowing effective and efficient exploration in complex environments. The core of our system is a rendering-based information gain module that dynamically identifies the most informative viewpoints for next-best-view planning, enhancing both geometric and photometric reconstruction accuracy. ActiveGAMER also integrates a carefully balanced framework, combining coarse-to-fine exploration, post-refinement, and a global-local keyframe selection strategy to maximize reconstruction completeness and fidelity. Our system autonomously explores and reconstructs environments with state-of-the-art geometric and photometric accuracy and completeness, significantly surpassing existing approaches in both aspects. Extensive evaluations on benchmark datasets such as Replica and MP3D highlight ActiveGAMER's effectiveness in active mapping tasks.
>
---
#### [replaced 030] Online 3D Gaussian Splatting Modeling with Novel View Selection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.14014v2](http://arxiv.org/pdf/2508.14014v2)**

> **作者:** Byeonggwon Lee; Junkyu Park; Khang Truong Giang; Soohwan Song
>
> **摘要:** This study addresses the challenge of generating online 3D Gaussian Splatting (3DGS) models from RGB-only frames. Previous studies have employed dense SLAM techniques to estimate 3D scenes from keyframes for 3DGS model construction. However, these methods are limited by their reliance solely on keyframes, which are insufficient to capture an entire scene, resulting in incomplete reconstructions. Moreover, building a generalizable model requires incorporating frames from diverse viewpoints to achieve broader scene coverage. However, online processing restricts the use of many frames or extensive training iterations. Therefore, we propose a novel method for high-quality 3DGS modeling that improves model completeness through adaptive view selection. By analyzing reconstruction quality online, our approach selects optimal non-keyframes for additional training. By integrating both keyframes and selected non-keyframes, the method refines incomplete regions from diverse viewpoints, significantly enhancing completeness. We also present a framework that incorporates an online multi-view stereo approach, ensuring consistency in 3D information throughout the 3DGS modeling process. Experimental results demonstrate that our method outperforms state-of-the-art methods, delivering exceptional performance in complex outdoor scenes.
>
---
#### [replaced 031] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.23567v2](http://arxiv.org/pdf/2507.23567v2)**

> **作者:** Yung-Hsu Yang; Luigi Piccinelli; Mattia Segu; Siyuan Li; Rui Huang; Yuqian Fu; Marc Pollefeys; Hermann Blum; Zuria Bauer
>
> **备注:** ICCV 2025
>
> **摘要:** Monocular 3D object detection is valuable for various applications such as robotics and AR/VR. Existing methods are confined to closed-set settings, where the training and testing sets consist of the same scenes and/or object categories. However, real-world applications often introduce new environments and novel object categories, posing a challenge to these methods. In this paper, we address monocular 3D object detection in an open-set setting and introduce the first end-to-end 3D Monocular Open-set Object Detector (3D-MOOD). We propose to lift the open-set 2D detection into 3D space through our designed 3D bounding box head, enabling end-to-end joint training for both 2D and 3D tasks to yield better overall performance. We condition the object queries with geometry prior and overcome the generalization for 3D estimation across diverse scenes. To further improve performance, we design the canonical image space for more efficient cross-dataset training. We evaluate 3D-MOOD on both closed-set settings (Omni3D) and open-set settings (Omni3D to Argoverse 2, ScanNet), and achieve new state-of-the-art results. Code and models are available at royyang0714.github.io/3D-MOOD.
>
---
#### [replaced 032] Automatic segmentation of Organs at Risk in Head and Neck cancer patients from CT and MRI scans
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.10833v3](http://arxiv.org/pdf/2405.10833v3)**

> **作者:** Sébastien Quetin; Andrew Heschl; Mauricio Murillo; Rohit Murali; Piotr Pater; George Shenouda; Shirin A. Enger; Farhad Maleki
>
> **摘要:** Purpose: To present a high-performing, robust, and flexible deep learning pipeline for automatic segmentation of 30 organs-at-risk (OARs) in head and neck (H&N) cancer patients, using MRI, CT, or both. Method: We trained a segmentation pipeline on paired CT and MRI-T1 scans from 296 patients. We combined data from the H&N OARs CT and MR segmentation (HaN-Seg) challenge and the Burdenko and GLIS-RT datasets from the Cancer Imaging Archive (TCIA). MRI was rigidly registered to CT, and both were stacked as input to an nnU-Net pipeline. Left and right OARs were merged into single classes during training and separated at inference time based on anatomical position. Modality Dropout was applied during the training, ensuring the model would learn from both modalities and robustly handle missing modalities during inference. The trained model was evaluated on the HaN-Seg test set and three TCIA datasets. Predictions were also compared with Limbus AI software. Dice Score (DS) and Hausdorff Distance (HD) were used as evaluation metrics. Results: The pipeline achieved state-of-the-art performance on the HaN-Seg challenge with a mean DS of 78.12% and HD of 3.42 mm. On TCIA datasets, the model maintained strong agreement with Limbus AI software (DS: 77.43% , HD: 3.27 mm), while also flagging low-quality contours. The pipeline can segment seamlessly from the CT, the MRI scan, or both. Conclusion: The proposed pipeline achieved the best DS and HD scores among all HaN-Seg challenge participants and establishes a new state-of-the-art for fully automated, multi-modal segmentation of H&N OARs.
>
---
#### [replaced 033] Unlocking Smarter Device Control: Foresighted Planning with a World Model-Driven Code Execution Approach
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16422v3](http://arxiv.org/pdf/2505.16422v3)**

> **作者:** Xiaoran Yin; Xu Luo; Hao Wu; Lianli Gao; Jingkuan Song
>
> **备注:** Accepted to Findings of EMNLP 2025. This is the camera-ready version
>
> **摘要:** The automatic control of mobile devices is essential for efficiently performing complex tasks that involve multiple sequential steps. However, these tasks pose significant challenges due to the limited environmental information available at each step, primarily through visual observations. As a result, current approaches, which typically rely on reactive policies, focus solely on immediate observations and often lead to suboptimal decision-making. To address this problem, we propose \textbf{Foresighted Planning with World Model-Driven Code Execution (FPWC)},a framework that prioritizes natural language understanding and structured reasoning to enhance the agent's global understanding of the environment by developing a task-oriented, refinable \emph{world model} at the outset of the task. Foresighted actions are subsequently generated through iterative planning within this world model, executed in the form of executable code. Extensive experiments conducted in simulated environments and on real mobile devices demonstrate that our method outperforms previous approaches, particularly achieving a 44.4\% relative improvement in task success rate compared to the state-of-the-art in the simulated environment. Code and demo are provided in the supplementary material.
>
---
#### [replaced 034] Multimodal LLM Guided Exploration and Active Mapping using Fisher Information
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.17422v3](http://arxiv.org/pdf/2410.17422v3)**

> **作者:** Wen Jiang; Boshu Lei; Katrina Ashton; Kostas Daniilidis
>
> **备注:** ICCV 2025
>
> **摘要:** We present an active mapping system that plans for both long-horizon exploration goals and short-term actions using a 3D Gaussian Splatting (3DGS) representation. Existing methods either do not take advantage of recent developments in multimodal Large Language Models (LLM) or do not consider challenges in localization uncertainty, which is critical in embodied agents. We propose employing multimodal LLMs for long-horizon planning in conjunction with detailed motion planning using our information-based objective. By leveraging high-quality view synthesis from our 3DGS representation, our method employs a multimodal LLM as a zero-shot planner for long-horizon exploration goals from the semantic perspective. We also introduce an uncertainty-aware path proposal and selection algorithm that balances the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.
>
---
#### [replaced 035] DRIVE-T: A Methodology for Discriminative and Representative Data Viz Item Selection for Literacy Construct and Assessment
- **分类: cs.HC; cs.CV; K.3; K.3.2**

- **链接: [http://arxiv.org/pdf/2508.04160v2](http://arxiv.org/pdf/2508.04160v2)**

> **作者:** Angela Locoro; Silvia Golia; Davide Falessi
>
> **摘要:** The underspecification of progressive levels of difficulty in measurement constructs design and assessment tests for data visualization literacy may hinder the expressivity of measurements in both test design and test reuse. To mitigate this problem, this paper proposes DRIVE-T (Discriminating and Representative Items for Validating Expressive Tests), a methodology designed to drive the construction and evaluation of assessment items. Given a data vizualization, DRIVE-T supports the identification of task-based items discriminability and representativeness for measuring levels of data visualization literacy. DRIVE-T consists of three steps: (1) tagging task-based items associated with a set of data vizualizations; (2) rating them by independent raters for their difficulty; (3) analysing raters' raw scores through a Many-Facet Rasch Measurement model. In this way, we can observe the emergence of difficulty levels of the measurement construct, derived from the discriminability and representativeness of task-based items for each data vizualization, ordered into Many-Facets construct levels. In this study, we show and apply each step of the methodology to an item bank, which models the difficulty levels of a measurement construct approximating a latent construct for data visualization literacy. This measurement construct is drawn from semiotics, i.e., based on the syntax, semantics and pragmatics knowledge that each data visualization may require to be mastered by people. The DRIVE-T methodology operationalises an inductive approach, observable in a post-design phase of the items preparation, for formative-style and practice-based measurement construct emergence. A pilot study with items selected through the application of DRIVE-T is also presented to test our approach.
>
---
#### [replaced 036] Colorectal Cancer Tumor Grade Segmentation in Digital Histopathology Images: From Giga to Mini Challenge
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04681v3](http://arxiv.org/pdf/2507.04681v3)**

> **作者:** Alper Bahcekapili; Duygu Arslan; Umut Ozdemir; Berkay Ozkirli; Emre Akbas; Ahmet Acar; Gozde B. Akar; Bingdou He; Shuoyu Xu; Umit Mert Caglar; Alptekin Temizel; Guillaume Picaud; Marc Chaumont; Gérard Subsol; Luc Téot; Fahad Alsharekh; Shahad Alghannam; Hexiang Mao; Wenhua Zhang
>
> **备注:** Accepted Grand Challenge Paper ICIP 2025
>
> **摘要:** Colorectal cancer (CRC) is the third most diagnosed cancer and the second leading cause of cancer-related death worldwide. Accurate histopathological grading of CRC is essential for prognosis and treatment planning but remains a subjective process prone to observer variability and limited by global shortages of trained pathologists. To promote automated and standardized solutions, we organized the ICIP Grand Challenge on Colorectal Cancer Tumor Grading and Segmentation using the publicly available METU CCTGS dataset. The dataset comprises 103 whole-slide images with expert pixel-level annotations for five tissue classes. Participants submitted segmentation masks via Codalab, evaluated using metrics such as macro F-score and mIoU. Among 39 participating teams, six outperformed the Swin Transformer baseline (62.92 F-score). This paper presents an overview of the challenge, dataset, and the top-performing methods
>
---
#### [replaced 037] Spoof Trace Discovery for Deep Learning Based Explainable Face Anti-Spoofing
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.17541v4](http://arxiv.org/pdf/2412.17541v4)**

> **作者:** Haoyuan Zhang; Xiangyu Zhu; Li Gao; Jiawei Pan; Kai Pang; Guoying Zhao; Zhen Lei
>
> **备注:** Accepted by IJCB 2025. Keywords: explainable artificial intelligence, face anti-spoofing, explainable face anti-spoofing, interpretable
>
> **摘要:** With the rapid growth usage of face recognition in people's daily life, face anti-spoofing becomes increasingly important to avoid malicious attacks. Recent face anti-spoofing models can reach a high classification accuracy on multiple datasets but these models can only tell people "this face is fake" while lacking the explanation to answer "why it is fake". Such a system undermines trustworthiness and causes user confusion, as it denies their requests without providing any explanations. In this paper, we incorporate XAI into face anti-spoofing and propose a new problem termed X-FAS (eXplainable Face Anti-Spoofing) empowering face anti-spoofing models to provide an explanation. We propose SPTD (SPoof Trace Discovery), an X-FAS method which can discover spoof concepts and provide reliable explanations on the basis of discovered concepts. To evaluate the quality of X-FAS methods, we propose an X-FAS benchmark with annotated spoof traces by experts. We analyze SPTD explanations on face anti-spoofing dataset and compare SPTD quantitatively and qualitatively with previous XAI methods on proposed X-FAS benchmark. Experimental results demonstrate SPTD's ability to generate reliable explanations.
>
---
#### [replaced 038] Beyond the Linear Separability Ceiling: Aligning Representations in VLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07574v2](http://arxiv.org/pdf/2507.07574v2)**

> **作者:** Enrico Vompa; Tanel Tammet; Mohit Vaishnav
>
> **备注:** preprint
>
> **摘要:** A challenge in advancing Visual-Language Models (VLMs) is determining whether their failures on abstract reasoning tasks, such as Bongard problems, stem from flawed perception or faulty top-down reasoning. To disentangle these factors, we introduce a diagnostic framework centered on the Linear Separability Ceiling (LSC), the performance achievable by a linear classifier on a VLM's raw visual embeddings. Applying this framework to state-of-the-art VLMs, we uncover a pervasive "alignment gap", where most models fail to generatively outperform the linear separability of their own representations. We find that the few models surpassing this ceiling do so via two mechanisms: by further refining visual representations into a more linearly separable format or by executing non-linear decision logic. We demonstrate that this bottleneck is not a fundamental limitation but a solvable alignment issue. By augmenting standard next-token prediction with a contrastive objective, our fine-tuning method activates dormant reasoning pathways, systematically improving the linear structure of representations to significantly surpass the LSC.
>
---
#### [replaced 039] Net2Brain: A Toolbox to compare artificial vision models with human brain responses
- **分类: cs.CV; cs.AI; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2208.09677v4](http://arxiv.org/pdf/2208.09677v4)**

> **作者:** Domenic Bersch; Kshitij Dwivedi; Martina Vilas; Radoslaw M. Cichy; Gemma Roig
>
> **备注:** Published in Frontiers in Neuroinformatics (2025), Article 1515873. Version of record: https://doi.org/10.3389/fninf.2025.1515873 4 Pages, 3 figures, submitted and accepted to CCNeuro 2022. For associated repository, see https://github.com/ToastyDom/Net2Brain Update 1: Changed Citation
>
> **摘要:** We introduce Net2Brain, a graphical and command-line user interface toolbox for comparing the representational spaces of artificial deep neural networks (DNNs) and human brain recordings. While different toolboxes facilitate only single functionalities or only focus on a small subset of supervised image classification models, Net2Brain allows the extraction of activations of more than 600 DNNs trained to perform a diverse range of vision-related tasks (e.g semantic segmentation, depth estimation, action recognition, etc.), over both image and video datasets. The toolbox computes the representational dissimilarity matrices (RDMs) over those activations and compares them to brain recordings using representational similarity analysis (RSA), weighted RSA, both in specific ROIs and with searchlight search. In addition, it is possible to add a new data set of stimuli and brain recordings to the toolbox for evaluation. We demonstrate the functionality and advantages of Net2Brain with an example showcasing how it can be used to test hypotheses of cognitive computational neuroscience.
>
---
#### [replaced 040] Empowering Bridge Digital Twins by Bridging the Data Gap with a Unified Synthesis Framework
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05814v3](http://arxiv.org/pdf/2507.05814v3)**

> **作者:** Wang Wang; Mingyu Shi; Jun Jiang; Wenqian Ma; Chong Liu; Yasutaka Narazaki; Xuguang Wang
>
> **备注:** Due to the authors' failure to reach an agreement on the manuscript quality, they voluntarily waive their rights to be credited as authors
>
> **摘要:** As critical transportation infrastructure, bridges face escalating challenges from aging and deterioration, while traditional manual inspection methods suffer from low efficiency. Although 3D point cloud technology provides a new data-driven paradigm, its application potential is often constrained by the incompleteness of real-world data, which results from missing labels and scanning occlusions. To overcome the bottleneck of insufficient generalization in existing synthetic data methods, this paper proposes a systematic framework for generating 3D bridge data. This framework can automatically generate complete point clouds featuring component-level instance annotations, high-fidelity color, and precise normal vectors. It can be further extended to simulate the creation of diverse and physically realistic incomplete point clouds, designed to support the training of segmentation and completion networks, respectively. Experiments demonstrate that a PointNet++ model trained with our synthetic data achieves a mean Intersection over Union (mIoU) of 84.2% in real-world bridge semantic segmentation. Concurrently, a fine-tuned KT-Net exhibits superior performance on the component completion task. This research offers an innovative methodology and a foundational dataset for the 3D visual analysis of bridge structures, holding significant implications for advancing the automated management and maintenance of infrastructure.
>
---
#### [replaced 041] DirectorLLM for Human-Centric Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.14484v2](http://arxiv.org/pdf/2412.14484v2)**

> **作者:** Kunpeng Song; Tingbo Hou; Zecheng He; Haoyu Ma; Jialiang Wang; Animesh Sinha; Sam Tsai; Yaqiao Luo; Xiaoliang Dai; Li Chen; Xide Xia; Peizhao Zhang; Peter Vajda; Ahmed Elgammal; Felix Juefei-Xu
>
> **摘要:** In this paper, we introduce DirectorLLM, a novel video generation model that employs a large language model (LLM) to orchestrate human poses within videos. As foundational text-to-video models rapidly evolve, the demand for high-quality human motion and interaction grows. To address this need and enhance the authenticity of human motions, we extend the LLM from a text generator to a video director and human motion simulator. Utilizing open-source resources from Llama 3, we train the DirectorLLM to generate detailed instructional signals, such as human poses, to guide video generation. This approach offloads the simulation of human motion from the video generator to the LLM, effectively creating informative outlines for human-centric scenes. These signals are used as conditions by the video renderer, facilitating more realistic and prompt-following video generation. As an independent LLM module, it can be applied to different video renderers, including UNet and DiT, with minimal effort. Experiments on automatic evaluation benchmarks and human evaluations show that our model outperforms existing ones in generating videos with higher human motion fidelity, improved prompt faithfulness, and enhanced rendered subject naturalness.
>
---
#### [replaced 042] SEA: Supervised Embedding Alignment for Token-Level Visual-Textual Integration in MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.11813v2](http://arxiv.org/pdf/2408.11813v2)**

> **作者:** Yuanyang Yin; Yaqi Zhao; Yajie Zhang; Yuanxing Zhang; Ke Lin; Jiahao Wang; Xin Tao; Pengfei Wan; Wentao Zhang; Feng Zhao
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities by integrating visual and textual inputs, yet modality alignment remains one of the most challenging aspects. Current MLLMs typically rely on simple adapter architectures and pretraining approaches to bridge vision encoders with large language models (LLM), guided by image-level supervision. We identify this paradigm often leads to suboptimal alignment between modalities, significantly constraining the LLM's ability to properly interpret and reason with visual features particularly for smaller language models. This limitation degrades overall performance-particularly for smaller language models where capacity constraints are more pronounced and adaptation capabilities are limited. To address this fundamental limitation, we propose Supervised Embedding Alignment (SEA), a token-level supervision alignment method that enables more precise visual-text alignment during pretraining. SEA introduces minimal computational overhead while preserving language capabilities and substantially improving cross-modal understanding. Our comprehensive analyses reveal critical insights into the adapter's role in multimodal integration, and extensive experiments demonstrate that SEA consistently improves performance across various model sizes, with smaller models benefiting the most (average performance gain of 7.61% for Gemma-2B). This work establishes a foundation for developing more effective alignment strategies for future multimodal systems.
>
---
#### [replaced 043] Auto-Connect: Connectivity-Preserving RigFormer with Direct Preference Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11430v2](http://arxiv.org/pdf/2506.11430v2)**

> **作者:** Jingfeng Guo; Jian Liu; Jinnan Chen; Shiwei Mao; Changrong Hu; Puhua Jiang; Junlin Yu; Jing Xu; Qi Liu; Lixin Xu; Zhuo Chen; Chunchao Guo
>
> **摘要:** We introduce Auto-Connect, a novel approach for automatic rigging that explicitly preserves skeletal connectivity through a connectivity-preserving tokenization scheme. Unlike previous methods that predict bone positions represented as two joints or first predict points before determining connectivity, our method employs special tokens to define endpoints for each joint's children and for each hierarchical layer, effectively automating connectivity relationships. This approach significantly enhances topological accuracy by integrating connectivity information directly into the prediction framework. To further guarantee high-quality topology, we implement a topology-aware reward function that quantifies topological correctness, which is then utilized in a post-training phase through reward-guided Direct Preference Optimization. Additionally, we incorporate implicit geodesic features for latent top-k bone selection, which substantially improves skinning quality. By leveraging geodesic distance information within the model's latent space, our approach intelligently determines the most influential bones for each vertex, effectively mitigating common skinning artifacts. This combination of connectivity-preserving tokenization, reward-guided fine-tuning, and geodesic-aware bone selection enables our model to consistently generate more anatomically plausible skeletal structures with superior deformation properties.
>
---
#### [replaced 044] WikiAutoGen: Towards Multi-Modal Wikipedia-Style Article Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19065v3](http://arxiv.org/pdf/2503.19065v3)**

> **作者:** Zhongyu Yang; Jun Chen; Dannong Xu; Junjie Fei; Xiaoqian Shen; Liangbing Zhao; Chun-Mei Feng; Mohamed Elhoseiny
>
> **备注:** ICCV 2025, Project in https://wikiautogen.github.io/
>
> **摘要:** Knowledge discovery and collection are intelligence-intensive tasks that traditionally require significant human effort to ensure high-quality outputs. Recent research has explored multi-agent frameworks for automating Wikipedia-style article generation by retrieving and synthesizing information from the internet. However, these methods primarily focus on text-only generation, overlooking the importance of multimodal content in enhancing informativeness and engagement. In this work, we introduce WikiAutoGen, a novel system for automated multimodal Wikipedia-style article generation. Unlike prior approaches, WikiAutoGen retrieves and integrates relevant images alongside text, enriching both the depth and visual appeal of generated content. To further improve factual accuracy and comprehensiveness, we propose a multi-perspective self-reflection mechanism, which critically assesses retrieved content from diverse viewpoints to enhance reliability, breadth, and coherence, etc. Additionally, we introduce WikiSeek, a benchmark comprising Wikipedia articles with topics paired with both textual and image-based representations, designed to evaluate multimodal knowledge generation on more challenging topics. Experimental results show that WikiAutoGen outperforms previous methods by 8%-29% on our WikiSeek benchmark, producing more accurate, coherent, and visually enriched Wikipedia-style articles. Our code and examples are available at https://wikiautogen.github.io/
>
---
#### [replaced 045] TPA: Temporal Prompt Alignment for Fetal Congenital Heart Defect Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.15298v5](http://arxiv.org/pdf/2508.15298v5)**

> **作者:** Darya Taratynova; Alya Almsouti; Beknur Kalmakhanbet; Numan Saeed; Mohammad Yaqub
>
> **摘要:** Congenital heart defect (CHD) detection in ultrasound videos is hindered by image noise and probe positioning variability. While automated methods can reduce operator dependence, current machine learning approaches often neglect temporal information, limit themselves to binary classification, and do not account for prediction calibration. We propose Temporal Prompt Alignment (TPA), a method leveraging foundation image-text model and prompt-aware contrastive learning to classify fetal CHD on cardiac ultrasound videos. TPA extracts features from each frame of video subclips using an image encoder, aggregates them with a trainable temporal extractor to capture heart motion, and aligns the video representation with class-specific text prompts via a margin-hinge contrastive loss. To enhance calibration for clinical reliability, we introduce a Conditional Variational Autoencoder Style Modulation (CVAESM) module, which learns a latent style vector to modulate embeddings and quantifies classification uncertainty. Evaluated on a private dataset for CHD detection and on a large public dataset, EchoNet-Dynamic, for systolic dysfunction, TPA achieves state-of-the-art macro F1 scores of 85.40% for CHD diagnosis, while also reducing expected calibration error by 5.38% and adaptive ECE by 6.8%. On EchoNet-Dynamic's three-class task, it boosts macro F1 by 4.73% (from 53.89% to 58.62%). Temporal Prompt Alignment (TPA) is a framework for fetal congenital heart defect (CHD) classification in ultrasound videos that integrates temporal modeling, prompt-aware contrastive learning, and uncertainty quantification.
>
---
#### [replaced 046] Adaptive Learning Strategies for Mitotic Figure Classification in MIDOG2025 Challenge
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02640v2](http://arxiv.org/pdf/2509.02640v2)**

> **作者:** Biwen Meng; Xi Long; Jingxin Liu
>
> **摘要:** Atypical mitotic figures (AMFs) are clinically relevant indicators of abnormal cell division, yet their reliable detection remains challenging due to morphological ambiguity and scanner variability. In this work, we investigated three variants of adapting the pathology foundation model UNI2 for the MIDOG2025 Track 2 challenge: (1) LoRA + UNI2, (2) VPT + UNI2 + Vahadane Normalizer, and (3) VPT + UNI2 + GRL + Stain TTA. We observed that the integration of Visual Prompt Tuning (VPT) with stain normalization techniques contributed to improved generalization. The best robustness was achieved by further incorporating test-time augmentation (TTA) with Vahadane and Macenko stain normalization. Our final submission achieved a balanced accuracy of 0.8837 and an ROC-AUC of 0.9513 on the preliminary leaderboard, ranking within the top 10 teams. These results suggest that prompt-based adaptation combined with stain-normalization TTA offers a promising strategy for atypical mitosis classification under diverse imaging conditions.
>
---
#### [replaced 047] SLENet: A Guidance-Enhanced Network for Underwater Camouflaged Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.03786v2](http://arxiv.org/pdf/2509.03786v2)**

> **作者:** Xinxin Huang; Han Sun; Ningzhong Liu; Huiyu Zhou; Yinan Yao
>
> **备注:** 14pages, accepted by PRCV2025
>
> **摘要:** Underwater Camouflaged Object Detection (UCOD) aims to identify objects that blend seamlessly into underwater environments. This task is critically important to marine ecology. However, it remains largely underexplored and accurate identification is severely hindered by optical distortions, water turbidity, and the complex traits of marine organisms. To address these challenges, we introduce the UCOD task and present DeepCamo, a benchmark dataset designed for this domain. We also propose Semantic Localization and Enhancement Network (SLENet), a novel framework for UCOD. We first benchmark state-of-the-art COD models on DeepCamo to reveal key issues, upon which SLENet is built. In particular, we incorporate Gamma-Asymmetric Enhancement (GAE) module and a Localization Guidance Branch (LGB) to enhance multi-scale feature representation while generating a location map enriched with global semantic information. This map guides the Multi-Scale Supervised Decoder (MSSD) to produce more accurate predictions. Experiments on our DeepCamo dataset and three benchmark COD datasets confirm SLENet's superior performance over SOTA methods, and underscore its high generality for the broader COD task.
>
---
#### [replaced 048] Automated detection of underdiagnosed medical conditions via opportunistic imaging
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.11686v4](http://arxiv.org/pdf/2409.11686v4)**

> **作者:** Asad Aali; Andrew Johnston; Louis Blankemeier; Dave Van Veen; Laura T Derry; David Svec; Jason Hom; Robert D. Boutin; Akshay S. Chaudhari
>
> **摘要:** Abdominal computed tomography (CT) scans are frequently performed in clinical settings. Opportunistic CT involves repurposing routine CT images to extract diagnostic information and is an emerging tool for detecting underdiagnosed conditions such as sarcopenia, hepatic steatosis, and ascites. This study utilizes deep learning methods to promote accurate diagnosis and clinical documentation. We analyze 2,674 inpatient CT scans to identify discrepancies between imaging phenotypes (characteristics derived from opportunistic CT scans) and their corresponding documentation in radiology reports and ICD coding. Through our analysis, we find that only 0.5%, 3.2%, and 30.7% of scans diagnosed with sarcopenia, hepatic steatosis, and ascites (respectively) through either opportunistic imaging or radiology reports were ICD-coded. Our findings demonstrate opportunistic CT's potential to enhance diagnostic precision and accuracy of risk adjustment models, offering advancements in precision medicine.
>
---
#### [replaced 049] AnomalyLMM: Bridging Generative Knowledge and Discriminative Retrieval for Text-Based Person Anomaly Search
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04376v2](http://arxiv.org/pdf/2509.04376v2)**

> **作者:** Hao Ju; Hu Zhang; Zhedong Zheng
>
> **摘要:** With growing public safety demands, text-based person anomaly search has emerged as a critical task, aiming to retrieve individuals with abnormal behaviors via natural language descriptions. Unlike conventional person search, this task presents two unique challenges: (1) fine-grained cross-modal alignment between textual anomalies and visual behaviors, and (2) anomaly recognition under sparse real-world samples. While Large Multi-modal Models (LMMs) excel in multi-modal understanding, their potential for fine-grained anomaly retrieval remains underexplored, hindered by: (1) a domain gap between generative knowledge and discriminative retrieval, and (2) the absence of efficient adaptation strategies for deployment. In this work, we propose AnomalyLMM, the first framework that harnesses LMMs for text-based person anomaly search. Our key contributions are: (1) A novel coarse-to-fine pipeline integrating LMMs to bridge generative world knowledge with retrieval-centric anomaly detection; (2) A training-free adaptation cookbook featuring masked cross-modal prompting, behavioral saliency prediction, and knowledge-aware re-ranking, enabling zero-shot focus on subtle anomaly cues. As the first study to explore LMMs for this task, we conduct a rigorous evaluation on the PAB dataset, the only publicly available benchmark for text-based person anomaly search, with its curated real-world anomalies covering diverse scenarios (e.g., falling, collision, and being hit). Experiments show the effectiveness of the proposed method, surpassing the competitive baseline by +0.96% Recall@1 accuracy. Notably, our method reveals interpretable alignment between textual anomalies and visual behaviors, validated via qualitative analysis. Our code and models will be released for future research.
>
---
#### [replaced 050] Representation-Centric Survey of Skeletal Action Recognition and the ANUBIS Benchmark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2205.02071v4](http://arxiv.org/pdf/2205.02071v4)**

> **作者:** Yang Liu; Jiyao Yang; Madhawa Perera; Pan Ji; Dongwoo Kim; Min Xu; Tianyang Wang; Saeed Anwar; Tom Gedeon; Lei Wang; Zhenyue Qin
>
> **摘要:** 3D skeleton-based human action recognition has emerged as a powerful alternative to traditional RGB and depth-based approaches, offering robustness to environmental variations, computational efficiency, and enhanced privacy. Despite remarkable progress, current research remains fragmented across diverse input representations and lacks evaluation under scenarios that reflect modern real-world challenges. This paper presents a representation-centric survey of skeleton-based action recognition, systematically categorizing state-of-the-art methods by their input feature types: joint coordinates, bone vectors, motion flows, and extended representations, and analyzing how these choices influence spatial-temporal modeling strategies. Building on the insights from this review, we introduce ANUBIS, a large-scale, challenging skeleton action dataset designed to address critical gaps in existing benchmarks. ANUBIS incorporates multi-view recordings with back-view perspectives, complex multi-person interactions, fine-grained and violent actions, and contemporary social behaviors. We benchmark a diverse set of state-of-the-art models on ANUBIS and conduct an in-depth analysis of how different feature types affect recognition performance across 102 action categories. Our results show strong action-feature dependencies, highlight the limitations of na\"ive multi-representational fusion, and point toward the need for task-aware, semantically aligned integration strategies. This work offers both a comprehensive foundation and a practical benchmarking resource, aiming to guide the next generation of robust, generalizable skeleton-based action recognition systems for complex real-world scenarios. The dataset website, benchmarking framework, and download link are available at \href{https://yliu1082.github.io/ANUBIS/}{https://yliu1082.github.io/ANUBIS/
>
---
#### [replaced 051] Generating Synthetic Contrast-Enhanced Chest CT Images from Non-Contrast Scans Using Slice-Consistent Brownian Bridge Diffusion Network
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2508.16897v2](http://arxiv.org/pdf/2508.16897v2)**

> **作者:** Pouya Shiri; Xin Yi; Neel P. Mistry; Samaneh Javadinia; Mohammad Chegini; Seok-Bum Ko; Amirali Baniasadi; Scott J. Adams
>
> **摘要:** Contrast-enhanced computed tomography (CT) imaging is essential for diagnosing and monitoring thoracic diseases, including aortic pathologies. However, contrast agents pose risks such as nephrotoxicity and allergic-like reactions. The ability to generate high-fidelity synthetic contrast-enhanced CT angiography (CTA) images without contrast administration would be transformative, enhancing patient safety and accessibility while reducing healthcare costs. In this study, we propose the first bridge diffusion-based solution for synthesizing contrast-enhanced CTA images from non-contrast CT scans. Our approach builds on the Slice-Consistent Brownian Bridge Diffusion Model (SC-BBDM), leveraging its ability to model complex mappings while maintaining consistency across slices. Unlike conventional slice-wise synthesis methods, our framework preserves full 3D anatomical integrity while operating in a high-resolution 2D fashion, allowing seamless volumetric interpretation under a low memory budget. To ensure robust spatial alignment, we implement a comprehensive preprocessing pipeline that includes resampling, registration using the Symmetric Normalization method, and a sophisticated dilated segmentation mask to extract the aorta and surrounding structures. We create two datasets from the Coltea-Lung dataset: one containing only the aorta and another including both the aorta and heart, enabling a detailed analysis of anatomical context. We compare our approach against baseline methods on both datasets, demonstrating its effectiveness in preserving vascular structures while enhancing contrast fidelity.
>
---
#### [replaced 052] FAGC:Feature Augmentation on Geodesic Curve in the Pre-Shape Space
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2312.03325v4](http://arxiv.org/pdf/2312.03325v4)**

> **作者:** Yuexing Han; Gan Hu; Guanxin Wan; Bing Wang
>
> **摘要:** Due to the constraints on model performance imposed by the size of the training data, data augmentation has become an essential technique in deep learning. However, most existing data augmentation methods are affected by information loss and perform poorly in small-sample scenarios, which limits their application. To overcome the limitation, we propose a Feature Augmentation method on Geodesic Curve in the pre-shape space, called the FAGC. First, a pre-trained neural network model is employed to extract features from the input images. Then, the image features as a vector is projected into the pre-shape space by removing its position and scale information. In the pre-shape space, an optimal Geodesic curve is constructed to fit the feature vectors. Finally, new feature vectors are generated for model learning by interpolating along the constructed Geodesic curve. We conducted extensive experiments to demonstrate the effectiveness and versatility of the FAGC. The results demonstrate that applying the FAGC to deep learning or machine learning methods can significantly improve their performance in small-sample tasks.
>
---
#### [replaced 053] 3D Densification for Multi-Map Monocular VSLAM in Endoscopy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14346v2](http://arxiv.org/pdf/2503.14346v2)**

> **作者:** X. Anadón; Javier Rodríguez-Puigvert; J. M. M. Montiel
>
> **摘要:** Multi-map Sparse Monocular visual Simultaneous Localization and Mapping applied to monocular endoscopic sequences has proven efficient to robustly recover tracking after the frequent losses in endoscopy due to motion blur, temporal occlusion, tools interaction or water jets. The sparse multi-maps are adequate for robust camera localization, however they are very poor for environment representation, they are noisy, with a high percentage of inaccurately reconstructed 3D points, including significant outliers, and more importantly with an unacceptable low density for clinical applications. We propose a method to remove outliers and densify the maps of the state of the art for sparse endoscopy multi-map CudaSIFT-SLAM. The NN LightDepth for up-to-scale depth dense predictions are aligned with the sparse CudaSIFT submaps by means of the robust to spurious LMedS. Our system mitigates the inherent scale ambiguity in monocular depth estimation while filtering outliers, leading to reliable densified 3D maps. We provide experimental evidence of accurate densified maps 4.15 mm RMS accuracy at affordable computing time in the C3VD phantom colon dataset. We report qualitative results on the real colonoscopy from the Endomapper dataset.
>
---
#### [replaced 054] PKF: Probabilistic Data Association Kalman Filter for Multi-Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.06378v2](http://arxiv.org/pdf/2411.06378v2)**

> **作者:** Hanwen Cao; George J. Pappas; Nikolay Atanasov
>
> **摘要:** In this paper, we derive a new Kalman filter with probabilistic data association between measurements and states. We formulate a variational inference problem to approximate the posterior density of the state conditioned on the measurement data. We view the unknown data association as a latent variable and apply Expectation Maximization (EM) to obtain a filter with update step in the same form as the Kalman filter but with expanded measurement vector of all potential associations. We show that the association probabilities can be computed as permanents of matrices with measurement likelihood entries. We also propose an ambiguity check that associates only a subset of ambiguous measurements and states probabilistically, thus reducing the association time and preventing low-probability measurements from harming the estimation accuracy. Experiments in simulation show that our filter achieves lower tracking errors than the well-established joint probabilistic data association filter (JPDAF), while running at comparable rate. We also demonstrate the effectiveness of our filter in multi-object tracking (MOT) on multiple real-world datasets, including MOT17, MOT20, and DanceTrack. We achieve better higher order tracking accuracy (HOTA) than previous Kalman-filter methods and remain real-time. Associating only bounding boxes without deep features or velocities, our method ranks top-10 on both MOT17 and MOT20 in terms of HOTA. Given offline detections, our algorithm tracks at 250+ fps on a single laptop CPU. Code is available at https://github.com/hwcao17/pkf.
>
---
#### [replaced 055] BayesSDF: Surface-Based Laplacian Uncertainty Estimation for 3D Geometry with Neural Signed Distance Fields
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06269v3](http://arxiv.org/pdf/2507.06269v3)**

> **作者:** Rushil Desai
>
> **备注:** ICCV 2025 Workshops (11 Pages, 6 Figures, 2 Tables)
>
> **摘要:** Accurate surface estimation is critical for downstream tasks in scientific simulation, and quantifying uncertainty in implicit neural 3D representations still remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. However, current neural implicit surface models do not offer a principled way to quantify uncertainty, limiting their reliability in real-world applications. Inspired by recent probabilistic rendering approaches, we introduce BayesSDF, a novel probabilistic framework for uncertainty estimation in neural implicit 3D representations. Unlike radiance-based models such as Neural Radiance Fields (NeRF) or 3D Gaussian Splatting, Signed Distance Functions (SDFs) provide continuous, differentiable surface representations, making them especially well-suited for uncertainty-aware modeling. BayesSDF applies a Laplace approximation over SDF weights and derives Hessian-based metrics to estimate local geometric instability. We empirically demonstrate that these uncertainty estimates correlate strongly with surface reconstruction error across both synthetic and real-world benchmarks. By enabling surface-aware uncertainty quantification, BayesSDF lays the groundwork for more robust, interpretable, and actionable 3D perception systems.
>
---
