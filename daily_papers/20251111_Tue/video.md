# 计算机视觉 cs.CV

- **最新发布 255 篇**

- **更新 138 篇**

## 最新发布

#### [new 001] Reperio-rPPG: Relational Temporal Graph Neural Networks for Periodicity Learning in Remote Physiological Measurement
- **分类: cs.CV**

- **简介: 该论文提出Reperio-rPPG，用于远程光体积描记术（rPPG）中的周期性学习，解决现有方法对生理信号时序周期建模不足的问题，通过关系图神经网络与CutMix增强，提升多场景下的心率与呼吸率估计精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.05946v1](http://arxiv.org/pdf/2511.05946v1)**

> **作者:** Ba-Thinh Nguyen; Thach-Ha Ngoc Pham; Hoang-Long Duc Nguyen; Thi-Duyen Ngo; Thanh-Ha Le
>
> **摘要:** Remote photoplethysmography (rPPG) is an emerging contactless physiological sensing technique that leverages subtle color variations in facial videos to estimate vital signs such as heart rate and respiratory rate. This non-invasive method has gained traction across diverse domains, including telemedicine, affective computing, driver fatigue detection, and health monitoring, owing to its scalability and convenience. Despite significant progress in remote physiological signal measurement, a crucial characteristic - the intrinsic periodicity - has often been underexplored or insufficiently modeled in previous approaches, limiting their ability to capture fine-grained temporal dynamics under real-world conditions. To bridge this gap, we propose Reperio-rPPG, a novel framework that strategically integrates Relational Convolutional Networks with a Graph Transformer to effectively capture the periodic structure inherent in physiological signals. Additionally, recognizing the limited diversity of existing rPPG datasets, we further introduce a tailored CutMix augmentation to enhance the model's generalizability. Extensive experiments conducted on three widely used benchmark datasets - PURE, UBFC-rPPG, and MMPD - demonstrate that Reperio-rPPG not only achieves state-of-the-art performance but also exhibits remarkable robustness under various motion (e.g., stationary, rotation, talking, walking) and illumination conditions (e.g., nature, low LED, high LED). The code is publicly available at https://github.com/deconasser/Reperio-rPPG.
>
---
#### [new 002] Video Text Preservation with Synthetic Text-Rich Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到视频（T2V）模型生成文本模糊的问题，提出用合成文本丰富视频进行轻量级微调，无需修改架构，显著提升短文本清晰度与时序一致性。**

- **链接: [http://arxiv.org/pdf/2511.05573v1](http://arxiv.org/pdf/2511.05573v1)**

> **作者:** Ziyang Liu; Kevin Valencia; Justin Cui
>
> **摘要:** While Text-To-Video (T2V) models have advanced rapidly, they continue to struggle with generating legible and coherent text within videos. In particular, existing models often fail to render correctly even short phrases or words and previous attempts to address this problem are computationally expensive and not suitable for video generation. In this work, we investigate a lightweight approach to improve T2V diffusion models using synthetic supervision. We first generate text-rich images using a text-to-image (T2I) diffusion model, then animate them into short videos using a text-agnostic image-to-video (I2v) model. These synthetic video-prompt pairs are used to fine-tune Wan2.1, a pre-trained T2V model, without any architectural changes. Our results show improvement in short-text legibility and temporal consistency with emerging structural priors for longer text. These findings suggest that curated synthetic data and weak supervision offer a practical path toward improving textual fidelity in T2V generation.
>
---
#### [new 003] Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV
- **分类: cs.CV**

- **简介: 论文针对宽视角小样本动作识别中的背景干扰问题，提出Otter模型，通过复合分割模块突出主体，时序重建模块增强时间关系建模，结合双重原型提升识别性能，在多个基准上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2511.06741v1](http://arxiv.org/pdf/2511.06741v1)**

> **作者:** Wenbo Huang; Jinghui Zhang; Zhenghao Chen; Guang Li; Lei Zhang; Yang Cao; Fang Dong; Takahiro Ogawa; Miki Haseyama
>
> **备注:** Accepted by AAAI 2026 Oral
>
> **摘要:** Wide-angle videos in few-shot action recognition (FSAR) effectively express actions within specific scenarios. However, without a global understanding of both subjects and background, recognizing actions in such samples remains challenging because of the background distractions. Receptance Weighted Key Value (RWKV), which learns interaction between various dimensions, shows promise for global modeling. While directly applying RWKV to wide-angle FSAR may fail to highlight subjects due to excessive background information. Additionally, temporal relation degraded by frames with similar backgrounds is difficult to reconstruct, further impacting performance. Therefore, we design the CompOund SegmenTation and Temporal REconstructing RWKV (Otter). Specifically, the Compound Segmentation Module~(CSM) is devised to segment and emphasize key patches in each frame, effectively highlighting subjects against background information. The Temporal Reconstruction Module (TRM) is incorporated into the temporal-enhanced prototype construction to enable bidirectional scanning, allowing better reconstruct temporal relation. Furthermore, a regular prototype is combined with the temporal-enhanced prototype to simultaneously enhance subject emphasis and temporal modeling, improving wide-angle FSAR performance. Extensive experiments on benchmarks such as SSv2, Kinetics, UCF101, and HMDB51 demonstrate that Otter achieves state-of-the-art performance. Extra evaluation on the VideoBadminton dataset further validates the superiority of Otter in wide-angle FSAR.
>
---
#### [new 004] Long Grounded Thoughts: Distilling Compositional Visual Reasoning Chains at Scale
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出一种大规模视觉推理数据生成框架，构建超百万高质量视觉推理样本，支持多模态推理训练。通过蒸馏复杂思维链，显著提升Qwen2.5-VL-7B性能，并实现跨模态迁移，解决视觉推理数据稀缺与泛化不足问题。**

- **链接: [http://arxiv.org/pdf/2511.05705v1](http://arxiv.org/pdf/2511.05705v1)**

> **作者:** David Acuna; Chao-Han Huck Yang; Yuntian Deng; Jaehun Jung; Ximing Lu; Prithviraj Ammanabrolu; Hyunwoo Kim; Yuan-Hong Liao; Yejin Choi
>
> **备注:** Project Page: https://nvlabs.github.io/LongGroundedThoughts/
>
> **摘要:** Recent progress in multimodal reasoning has been driven largely by undisclosed datasets and proprietary data synthesis recipes, leaving open questions about how to systematically build large-scale, vision-centric reasoning datasets, particularly for tasks that go beyond visual math. In this work, we introduce a new reasoning data generation framework spanning diverse skills and levels of complexity with over 1M high-quality synthetic vision-centric questions. The dataset also includes preference data and instruction prompts supporting both offline and online RL. Our synthesis framework proceeds in two stages: (1) scale; and (2) complexity. Reasoning traces are then synthesized through a two-stage process that leverages VLMs and reasoning LLMs, producing CoT traces for VLMs that capture the richness and diverse cognitive behaviors found in frontier reasoning models. Remarkably, we show that finetuning Qwen2.5-VL-7B on our data outperforms all open-data baselines across all evaluated vision-centric benchmarks, and even surpasses strong closed-data models such as MiMo-VL-7B-RL on V* Bench, CV-Bench and MMStar-V. Perhaps most surprising, despite being entirely vision-centric, our data transfers positively to text-only reasoning (MMLU-Pro) and audio reasoning (MMAU), demonstrating its effectiveness. Similarly, despite not containing videos or embodied visual data, we observe notable gains when evaluating on a single-evidence embodied QA benchmark (NiEH). Finally, we use our data to analyze the entire VLM post-training pipeline. Our empirical analysis highlights that (i) SFT on high-quality data with non-linear reasoning traces is essential for effective online RL, (ii) staged offline RL matches online RL's performance while reducing compute demands, and (iii) careful SFT on high quality data can substantially improve out-of-domain, cross-modality transfer.
>
---
#### [new 005] MRT: Learning Compact Representations with Mixed RWKV-Transformer for Extreme Image Compression
- **分类: cs.CV**

- **简介: 该论文面向极端图像压缩任务，解决传统2D潜空间冗余高问题，提出MRT架构，融合RWKV与Transformer，将图像编码为紧凑1D表示，并设计专用RCM模型，显著提升低比特率下的重建质量。**

- **链接: [http://arxiv.org/pdf/2511.06717v1](http://arxiv.org/pdf/2511.06717v1)**

> **作者:** Han Liu; Hengyu Man; Xingtao Wang; Wenrui Li; Debin Zhao
>
> **摘要:** Recent advances in extreme image compression have revealed that mapping pixel data into highly compact latent representations can significantly improve coding efficiency. However, most existing methods compress images into 2-D latent spaces via convolutional neural networks (CNNs) or Swin Transformers, which tend to retain substantial spatial redundancy, thereby limiting overall compression performance. In this paper, we propose a novel Mixed RWKV-Transformer (MRT) architecture that encodes images into more compact 1-D latent representations by synergistically integrating the complementary strengths of linear-attention-based RWKV and self-attention-based Transformer models. Specifically, MRT partitions each image into fixed-size windows, utilizing RWKV modules to capture global dependencies across windows and Transformer blocks to model local redundancies within each window. The hierarchical attention mechanism enables more efficient and compact representation learning in the 1-D domain. To further enhance compression efficiency, we introduce a dedicated RWKV Compression Model (RCM) tailored to the structure characteristics of the intermediate 1-D latent features in MRT. Extensive experiments on standard image compression benchmarks validate the effectiveness of our approach. The proposed MRT framework consistently achieves superior reconstruction quality at bitrates below 0.02 bits per pixel (bpp). Quantitative results based on the DISTS metric show that MRT significantly outperforms the state-of-the-art 2-D architecture GLC, achieving bitrate savings of 43.75%, 30.59% on the Kodak and CLIC2020 test datasets, respectively.
>
---
#### [new 006] Distributed Deep Learning for Medical Image Denoising with Data Obfuscation
- **分类: cs.CV; cs.DC**

- **简介: 该论文面向医学图像去噪任务，解决敏感数据暴露与训练效率问题，提出结合高斯模糊脱敏与分布式深度学习（DDP+AMP）的U-Net++框架，在保持高质量去噪的同时加速训练超60%。**

- **链接: [http://arxiv.org/pdf/2511.06006v1](http://arxiv.org/pdf/2511.06006v1)**

> **作者:** Sulaimon Oyeniyi Adebayo; Ayaz H. Khan
>
> **摘要:** Medical image denoising is essential for improving image quality while minimizing the exposure of sensitive information, particularly when working with large-scale clinical datasets. This study explores distributed deep learning for denoising chest X-ray images from the NIH Chest X-ray14 dataset, using additive Gaussian noise as a lightweight obfuscation technique. We implement and evaluate U-Net and U-Net++ architectures under single-GPU, standard multi-GPU (DataParallel), and optimized multi-GPU training configurations using PyTorch's DistributedDataParallel (DDP) and Automatic Mixed Precision (AMP). Our results show that U-Net++ consistently delivers superior denoising performance, achieving competitive Peak Signal to Noise Ratio (PSNR) and Structured Similarity Index Method (SSIM) scores, though with less performance in Learned Perceptual Image Patch Similarity (LPIPS) compared to U-Net under low and moderate noise levels. This indicates U-Net++'s enhanced structural fidelity and low perceptual similarity. Meanwhile, our optimized training pipeline reduces training time by over 60% for both models compared to single-GPU training, and outperforms standard DataParallel by over 40%, with only a minor accuracy drop for both models (trading some accuracy for speed). These findings highlight the effectiveness of software-level optimization in distributed learning for medical imaging. This work demonstrates the practical viability of combining architectural design, lightweight obfuscation, and advanced distributed training strategies to accelerate and enhance medical image processing pipelines in real-world clinical and research environments. The full implementation is publicly available at: https://github.com/Suadey/medical-image-denoising-ddp.
>
---
#### [new 007] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出SpatialThinker，面向多模态大模型的3D空间理解任务，解决其空间推理能力弱的问题。通过合成STVQA-7K数据集与强化学习密集空间奖励机制，实现无需大规模数据的高效空间 grounding 与多步推理，显著超越基线与GPT-4o。**

- **链接: [http://arxiv.org/pdf/2511.07403v1](http://arxiv.org/pdf/2511.07403v1)**

> **作者:** Hunar Batra; Haoqin Tu; Hardy Chen; Yuanze Lin; Cihang Xie; Ronald Clark
>
> **备注:** Preprint. Accepted at NeurIPS 2025 Workshops on SPACE in Vision, Language, and Embodied AI (SpaVLE), Embodied World Models for Decision Making (EWM), Aligning Reinforcement Learning Experimentalists and Theorists (ARLET), and Scaling Environments for Agents (SEA)
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable progress in vision-language tasks, but they continue to struggle with spatial understanding. Existing spatial MLLMs often rely on explicit 3D inputs or architecture-specific modifications, and remain constrained by large-scale datasets or sparse supervision. To address these limitations, we introduce SpatialThinker, a 3D-aware MLLM trained with RL to integrate structured spatial grounding with multi-step reasoning. The model simulates human-like spatial perception by constructing a scene graph of task-relevant objects and spatial relations, and reasoning towards an answer via dense spatial rewards. SpatialThinker consists of two key contributions: (1) a data synthesis pipeline that generates STVQA-7K, a high-quality spatial VQA dataset, and (2) online RL with a multi-objective dense spatial reward enforcing spatial grounding. SpatialThinker-7B outperforms supervised fine-tuning and the sparse RL baseline on spatial understanding and real-world VQA benchmarks, nearly doubling the base-model gain compared to sparse RL, and surpassing GPT-4o. These results showcase the effectiveness of combining spatial supervision with reward-aligned reasoning in enabling robust 3D spatial understanding with limited data and advancing MLLMs towards human-level visual reasoning.
>
---
#### [new 008] Towards Better Ultrasound Video Segmentation Foundation Model: An Empirical study on SAM2 Finetuning from Data Perspective
- **分类: cs.CV**

- **简介: 该论文针对超声视频分割任务，研究SAM2模型在医疗数据上的适应问题，系统分析数据规模、时序上下文与增强策略对微调效果的影响，提出六种医学专用增强方法，揭示数据因素比模型结构更关键。**

- **链接: [http://arxiv.org/pdf/2511.05731v1](http://arxiv.org/pdf/2511.05731v1)**

> **作者:** Xing Yao; Ahana Gangopadhyay; Hsi-Ming Chang; Ravi Soni
>
> **摘要:** Ultrasound (US) video segmentation remains a challenging problem due to strong inter- and intra-dataset variability, motion artifacts, and limited annotated data. Although foundation models such as Segment Anything Model 2 (SAM2) demonstrate strong zero-shot and prompt-guided segmentation capabilities, their performance deteriorates substantially when transferred to medical imaging domains. Current adaptation studies mainly emphasize architectural modifications, while the influence of data characteristics and training regimes has not been systematically examined. In this study, we present a comprehensive, data-centric investigation of SAM2 adaptation for ultrasound video segmentation. We analyze how training-set size, video duration, and augmentation schemes affect adaptation performance under three paradigms: task-specific fine-tuning, intermediate adaptation, and multi-task joint training, across five SAM2 variants and multiple prompting modes. We further design six ultrasound-specific augmentations, assessing their effect relative to generic strategies. Experiments on three representative ultrasound datasets reveal that data scale and temporal context play a more decisive role than model architecture or initialization. Moreover, joint training offers an efficient compromise between modality alignment and task specialization. This work aims to provide empirical insights for developing efficient, data-aware adaptation pipelines for SAM2 in ultrasound video analysis.
>
---
#### [new 009] MPJudge: Towards Perceptual Assessment of Music-Induced Paintings
- **分类: cs.CV**

- **简介: 论文提出MPJudge，解决音乐诱导绘画的感知一致性评估问题，构建首个专家标注的音乐-绘画配对数据集MPD，通过音乐特征调制视觉编码器与偏好优化，提升模型对音乐相关视觉区域的识别准确性。**

- **链接: [http://arxiv.org/pdf/2511.07137v1](http://arxiv.org/pdf/2511.07137v1)**

> **作者:** Shiqi Jiang; Tianyi Liang; Changbo Wang; Chenhui Li
>
> **摘要:** Music induced painting is a unique artistic practice, where visual artworks are created under the influence of music. Evaluating whether a painting faithfully reflects the music that inspired it poses a challenging perceptual assessment task. Existing methods primarily rely on emotion recognition models to assess the similarity between music and painting, but such models introduce considerable noise and overlook broader perceptual cues beyond emotion. To address these limitations, we propose a novel framework for music induced painting assessment that directly models perceptual coherence between music and visual art. We introduce MPD, the first large scale dataset of music painting pairs annotated by domain experts based on perceptual coherence. To better handle ambiguous cases, we further collect pairwise preference annotations. Building on this dataset, we present MPJudge, a model that integrates music features into a visual encoder via a modulation based fusion mechanism. To effectively learn from ambiguous cases, we adopt Direct Preference Optimization for training. Extensive experiments demonstrate that our method outperforms existing approaches. Qualitative results further show that our model more accurately identifies music relevant regions in paintings.
>
---
#### [new 010] Adaptive Agent Selection and Interaction Network for Image-to-point cloud Registration
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对图像与点云配准任务，解决噪声下特征匹配不准确问题。提出IAS与RAI模块，通过强化学习选择可靠代理并引导跨模态交互，显著提升配准鲁棒性与精度。**

- **链接: [http://arxiv.org/pdf/2511.05965v1](http://arxiv.org/pdf/2511.05965v1)**

> **作者:** Zhixin Cheng; Xiaotian Yin; Jiacheng Deng; Bohao Liao; Yujia Chen; Xu Zhou; Baoqun Yin; Tianzhu Zhang
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Typical detection-free methods for image-to-point cloud registration leverage transformer-based architectures to aggregate cross-modal features and establish correspondences. However, they often struggle under challenging conditions, where noise disrupts similarity computation and leads to incorrect correspondences. Moreover, without dedicated designs, it remains difficult to effectively select informative and correlated representations across modalities, thereby limiting the robustness and accuracy of registration. To address these challenges, we propose a novel cross-modal registration framework composed of two key modules: the Iterative Agents Selection (IAS) module and the Reliable Agents Interaction (RAI) module. IAS enhances structural feature awareness with phase maps and employs reinforcement learning principles to efficiently select reliable agents. RAI then leverages these selected agents to guide cross-modal interactions, effectively reducing mismatches and improving overall robustness. Extensive experiments on the RGB-D Scenes v2 and 7-Scenes benchmarks demonstrate that our method consistently achieves state-of-the-art performance.
>
---
#### [new 011] Exploring the "Great Unseen" in Medieval Manuscripts: Instance-Level Labeling of Legacy Image Collections with Zero-Shot Models
- **分类: cs.CV; cs.HC**

- **简介: 该论文利用零样本模型对中世纪手稿图像进行实例级标注，解决传统标注数据稀缺问题，旨在构建更丰富的视觉数据集，支持手稿内容的分割与多模态理解。**

- **链接: [http://arxiv.org/pdf/2511.07004v1](http://arxiv.org/pdf/2511.07004v1)**

> **作者:** Christofer Meinecke; Estelle Guéville; David Joseph Wrisley
>
> **摘要:** We aim to theorize the medieval manuscript page and its contents more holistically, using state-of-the-art techniques to segment and describe the entire manuscript folio, for the purpose of creating richer training data for computer vision techniques, namely instance segmentation, and multimodal models for medieval-specific visual content.
>
---
#### [new 012] Automated Estimation of Anatomical Risk Metrics for Endoscopic Sinus Surgery Using Deep Learning
- **分类: cs.CV**

- **简介: 该论文提出一种基于深度学习的自动化方法，通过热图回归定位关键解剖标志，估算内窥镜鼻窦手术中的Keros、Gera和TMS风险评分，解决人工测量耗时问题，实现亚毫米级精度的解剖风险评估。**

- **链接: [http://arxiv.org/pdf/2511.07199v1](http://arxiv.org/pdf/2511.07199v1)**

> **作者:** Konrad Reuter; Lennart Thaysen; Bilkay Doruk; Sarah Latus; Brigitte Holst; Benjamin Becker; Dennis Eggert; Christian Betz; Anna-Sophie Hoffmann; Alexander Schlaefer
>
> **备注:** Accepted to SPIE Medical Imaging conference 2026
>
> **摘要:** Endoscopic sinus surgery requires careful preoperative assessment of the skull base anatomy to minimize risks such as cerebrospinal fluid leakage. Anatomical risk scores like the Keros, Gera and Thailand-Malaysia-Singapore score offer a standardized approach but require time-consuming manual measurements on coronal CT or CBCT scans. We propose an automated deep learning pipeline that estimates these risk scores by localizing key anatomical landmarks via heatmap regression. We compare a direct approach to a specialized global-to-local learning strategy and find mean absolute errors on the relevant anatomical measurements of 0.506mm for the Keros, 4.516{\deg} for the Gera and 0.802mm / 0.777mm for the TMS classification.
>
---
#### [new 013] MoEGCL: Mixture of Ego-Graphs Contrastive Representation Learning for Multi-View Clustering
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多视图聚类中的粗粒度图融合问题，提出MoEGCL，通过样本级的ego图混合专家融合与对比学习，实现细粒度表示对齐，显著提升聚类性能。**

- **链接: [http://arxiv.org/pdf/2511.05876v1](http://arxiv.org/pdf/2511.05876v1)**

> **作者:** Jian Zhu; Xin Zou; Jun Sun; Cheng Luo; Lei Liu; Lingfang Zeng; Ning Zhang; Bian Wu; Chang Tang; Lirong Dai
>
> **备注:** AAAI'2026 oral paper
>
> **摘要:** In recent years, the advancement of Graph Neural Networks (GNNs) has significantly propelled progress in Multi-View Clustering (MVC). However, existing methods face the problem of coarse-grained graph fusion. Specifically, current approaches typically generate a separate graph structure for each view and then perform weighted fusion of graph structures at the view level, which is a relatively rough strategy. To address this limitation, we present a novel Mixture of Ego-Graphs Contrastive Representation Learning (MoEGCL). It mainly consists of two modules. In particular, we propose an innovative Mixture of Ego-Graphs Fusion (MoEGF), which constructs ego graphs and utilizes a Mixture-of-Experts network to implement fine-grained fusion of ego graphs at the sample level, rather than the conventional view-level fusion. Additionally, we present the Ego Graph Contrastive Learning (EGCL) module to align the fused representation with the view-specific representation. The EGCL module enhances the representation similarity of samples from the same cluster, not merely from the same sample, further boosting fine-grained graph representation. Extensive experiments demonstrate that MoEGCL achieves state-of-the-art results in deep multi-view clustering tasks. The source code is publicly available at https://github.com/HackerHyper/MoEGCL.
>
---
#### [new 014] LaneDiffusion: Improving Centerline Graph Learning via Prior Injected BEV Feature Generation
- **分类: cs.CV; cs.AI**

- **简介: LaneDiffusion提出一种生成式方法，通过扩散模型在BEV空间生成车道先验，解决传统方法在遮挡场景下中心线学习不足的问题，显著提升点级与段级评估指标，实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.06272v1](http://arxiv.org/pdf/2511.06272v1)**

> **作者:** Zijie Wang; Weiming Zhang; Wei Zhang; Xiao Tan; Hongxing Liu; Yaowei Wang; Guanbin Li
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Centerline graphs, crucial for path planning in autonomous driving, are traditionally learned using deterministic methods. However, these methods often lack spatial reasoning and struggle with occluded or invisible centerlines. Generative approaches, despite their potential, remain underexplored in this domain. We introduce LaneDiffusion, a novel generative paradigm for centerline graph learning. LaneDiffusion innovatively employs diffusion models to generate lane centerline priors at the Bird's Eye View (BEV) feature level, instead of directly predicting vectorized centerlines. Our method integrates a Lane Prior Injection Module (LPIM) and a Lane Prior Diffusion Module (LPDM) to effectively construct diffusion targets and manage the diffusion process. Furthermore, vectorized centerlines and topologies are then decoded from these prior-injected BEV features. Extensive evaluations on the nuScenes and Argoverse2 datasets demonstrate that LaneDiffusion significantly outperforms existing methods, achieving improvements of 4.2%, 4.6%, 4.7%, 6.4% and 1.8% on fine-grained point-level metrics (GEO F1, TOPO F1, JTOPO F1, APLS and SDA) and 2.3%, 6.4%, 6.8% and 2.1% on segment-level metrics (IoU, mAP_cf, DET_l and TOP_ll). These results establish state-of-the-art performance in centerline graph learning, offering new insights into generative models for this task.
>
---
#### [new 015] Towards Frequency-Adaptive Learning for SAR Despeckling
- **分类: cs.CV; 68T10; I.4**

- **简介: 该论文针对SAR图像去斑点噪声任务，提出SAR-FAH模型，通过小波分解将图像按频率分段，为低频和高频分量设计专用网络，分别基于神经微分方程和变形卷积U-Net，提升去噪效果并保留边缘纹理。**

- **链接: [http://arxiv.org/pdf/2511.05890v1](http://arxiv.org/pdf/2511.05890v1)**

> **作者:** Ziqing Ma; Chang Yang; Zhichang Guo; Yao Li
>
> **备注:** 13 pages, 14 figures,9 tables
>
> **摘要:** Synthetic Aperture Radar (SAR) images are inherently corrupted by speckle noise, limiting their utility in high-precision applications. While deep learning methods have shown promise in SAR despeckling, most methods employ a single unified network to process the entire image, failing to account for the distinct speckle statistics associated with different spatial physical characteristics. It often leads to artifacts, blurred edges, and texture distortion. To address these issues, we propose SAR-FAH, a frequency-adaptive heterogeneous despeckling model based on a divide-and-conquer architecture. First, wavelet decomposition is used to separate the image into frequency sub-bands carrying different intrinsic characteristics. Inspired by their differing noise characteristics, we design specialized sub-networks for different frequency components. The tailored approach leverages statistical variations across frequencies, improving edge and texture preservation while suppressing noise. Specifically, for the low-frequency part, denoising is formulated as a continuous dynamic system via neural ordinary differential equations, ensuring structural fidelity and sufficient smoothness that prevents artifacts. For high-frequency sub-bands rich in edges and textures, we introduce an enhanced U-Net with deformable convolutions for noise suppression and enhanced features. Extensive experiments on synthetic and real SAR images validate the superior performance of the proposed model in noise removal and structural preservation.
>
---
#### [new 016] Classification of Microplastic Particles in Water using Polarized Light Scattering and Machine Learning Methods
- **分类: cs.CV**

- **简介: 该论文提出一种基于偏振光散射与深度学习的水中微塑料分类方法，解决传统透射法受水体干扰的问题。通过偏振相机捕获反射信号，利用CNN识别三种聚合物，准确率达80%，并发现AOLP信号更优，尤其区分两种聚乙烯。**

- **链接: [http://arxiv.org/pdf/2511.06901v1](http://arxiv.org/pdf/2511.06901v1)**

> **作者:** Leonard Saur; Marc von Pawlowski; Ulrich Gengenbach; Ingo Sieber; Hossein Shirali; Lorenz Wührl; Rainer Kiko; Christian Pylatiuk
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Facing the critical need for continuous, large-scale microplastic monitoring, which is hindered by the limitations of gold-standard methods in aquatic environments, this paper introduces and validates a novel, reflection-based approach for the in-situ classification and identification of microplastics directly in water bodies, which is based on polarized light scattering. In this experiment, we classify colorless microplastic particles (50-300 $\mu$m) by illuminating them with linearly polarized laser light and capturing their reflected signals using a polarization-sensitive camera. This reflection-based technique successfully circumvents the transmission-based interference issues that plague many conventional methods when applied in water. Using a deep convolutional neural network (CNN) for image-based classification, we successfully identified three common polymer types, high-density polyethylene, low-density polyethylene, and polypropylene, achieving a peak mean classification accuracy of 80% on the test dataset. A subsequent feature hierarchy analysis demonstrated that the CNN's decision-making process relies mainly on the microstructural integrity and internal texture (polarization patterns) of the particle rather than its macroshape. Critically, we found that the Angle of Linear Polarization (AOLP) signal is significantly more robust against contextual noise than the Degree of Linear Polarization (DOLP) signal. While the AOLP-based classification achieved superior overall performance, its strength lies in distinguishing between the two polyethylene plastics, showing a lower confusion rate between high-density and low-density polyethylene. Conversely, the DOLP signal demonstrated slightly worse overall classification results but excels at accurately identifying the polypropylene class, which it isolated with greater success than AOLP.
>
---
#### [new 017] REOcc: Camera-Radar Fusion with Radar Feature Enrichment for 3D Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文提出REOcc，用于相机-雷达融合的3D占位预测任务，解决雷达数据稀疏噪声导致融合效果差的问题。通过雷达密度增强器和放大器，提升雷达特征质量，显著优化动态目标预测性能。**

- **链接: [http://arxiv.org/pdf/2511.06666v1](http://arxiv.org/pdf/2511.06666v1)**

> **作者:** Chaehee Song; Sanmin Kim; Hyeonjun Jeong; Juyeb Shin; Joonhee Lim; Dongsuk Kum
>
> **备注:** IROS 2025
>
> **摘要:** Vision-based 3D occupancy prediction has made significant advancements, but its reliance on cameras alone struggles in challenging environments. This limitation has driven the adoption of sensor fusion, among which camera-radar fusion stands out as a promising solution due to their complementary strengths. However, the sparsity and noise of the radar data limits its effectiveness, leading to suboptimal fusion performance. In this paper, we propose REOcc, a novel camera-radar fusion network designed to enrich radar feature representations for 3D occupancy prediction. Our approach introduces two main components, a Radar Densifier and a Radar Amplifier, which refine radar features by integrating spatial and contextual information, effectively enhancing spatial density and quality. Extensive experiments on the Occ3D-nuScenes benchmark demonstrate that REOcc achieves significant performance gains over the camera-only baseline model, particularly in dynamic object classes. These results underscore REOcc's capability to mitigate the sparsity and noise of the radar data. Consequently, radar complements camera data more effectively, unlocking the full potential of camera-radar fusion for robust and reliable 3D occupancy prediction.
>
---
#### [new 018] A Second-Order Attention Mechanism For Prostate Cancer Segmentation and Detection in Bi-Parametric MRI
- **分类: cs.CV; I.4.6; I.5.4**

- **简介: 该论文针对前列腺癌在双参数MRI中的分割与检测任务，提出一种基于黎曼流形的二阶几何注意力机制（SOGA），提升小样本标注下对高变异性病灶的识别能力，在PI-CAI和Prostate158数据集上显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2511.05760v1](http://arxiv.org/pdf/2511.05760v1)**

> **作者:** Mateo Ortiz; Juan Olmos; Fabio Martínez
>
> **备注:** Accepted at the 28th Iberoamerican Congress on Pattern Recognition (CIARP 2025). To appear in Lecture Notes in Computer Science (LNCS), Springer
>
> **摘要:** The detection of clinically significant prostate cancer lesions (csPCa) from biparametric magnetic resonance imaging (bp-MRI) has emerged as a noninvasive imaging technique for improving accurate diagnosis. Nevertheless, the analysis of such images remains highly dependent on the subjective expert interpretation. Deep learning approaches have been proposed for csPCa lesions detection and segmentation, but they remain limited due to their reliance on extensively annotated datasets. Moreover, the high lesion variability across prostate zones poses additional challenges, even for expert radiologists. This work introduces a second-order geometric attention (SOGA) mechanism that guides a dedicated segmentation network, through skip connections, to detect csPCa lesions. The proposed attention is modeled on the Riemannian manifold, learning from symmetric positive definitive (SPD) representations. The proposed mechanism was integrated into standard U-Net and nnU-Net backbones, and was validated on the publicly available PI-CAI dataset, achieving an Average Precision (AP) of 0.37 and an Area Under the ROC Curve (AUC-ROC) of 0.83, outperforming baseline networks and attention-based methods. Furthermore, the approach was evaluated on the Prostate158 dataset as an independent test cohort, achieving an AP of 0.37 and an AUC-ROC of 0.75, confirming robust generalization and suggesting discriminative learned representations.
>
---
#### [new 019] AesTest: Measuring Aesthetic Intelligence from Perception to Production
- **分类: cs.CV**

- **简介: 论文提出AesTest基准，用于评估多模态大模型的审美智能，解决现有评测集感知范围窄、生产多样性不足的问题。该工作整合心理学理论与多源数据，构建涵盖感知、欣赏、创作的十任务评测体系，支持多维度审美推理。**

- **链接: [http://arxiv.org/pdf/2511.06360v1](http://arxiv.org/pdf/2511.06360v1)**

> **作者:** Guolong Wang; Heng Huang; Zhiqiang Zhang; Wentian Li; Feilong Ma; Xin Jin
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** Perceiving and producing aesthetic judgments is a fundamental yet underexplored capability for multimodal large language models (MLLMs). However, existing benchmarks for image aesthetic assessment (IAA) are narrow in perception scope or lack the diversity needed to evaluate systematic aesthetic production. To address this gap, we introduce AesTest, a comprehensive benchmark for multimodal aesthetic perception and production, distinguished by the following features: 1) It consists of curated multiple-choice questions spanning ten tasks, covering perception, appreciation, creation, and photography. These tasks are grounded in psychological theories of generative learning. 2) It integrates data from diverse sources, including professional editing workflows, photographic composition tutorials, and crowdsourced preferences. It ensures coverage of both expert-level principles and real-world variation. 3) It supports various aesthetic query types, such as attribute-based analysis, emotional resonance, compositional choice, and stylistic reasoning. We evaluate both instruction-tuned IAA MLLMs and general MLLMs on AesTest, revealing significant challenges in building aesthetic intelligence. We will publicly release AesTest to support future research in this area.
>
---
#### [new 020] Registration-Free Monitoring of Unstructured Point Cloud Data via Intrinsic Geometrical Properties
- **分类: cs.CV; cs.LG; stat.ME; stat.ML**

- **简介: 该论文提出一种无需配准与网格重建的点云监测方法，利用拉普拉斯与测地距离等固有几何特征，通过阈值筛选识别缺陷，解决传统预处理易出错、耗时的问题。**

- **链接: [http://arxiv.org/pdf/2511.05623v1](http://arxiv.org/pdf/2511.05623v1)**

> **作者:** Mariafrancesca Patalano; Giovanna Capizzi; Kamran Paynabar
>
> **摘要:** Modern sensing technologies have enabled the collection of unstructured point cloud data (PCD) of varying sizes, which are used to monitor the geometric accuracy of 3D objects. PCD are widely applied in advanced manufacturing processes, including additive, subtractive, and hybrid manufacturing. To ensure the consistency of analysis and avoid false alarms, preprocessing steps such as registration and mesh reconstruction are commonly applied prior to monitoring. However, these steps are error-prone, time-consuming and may introduce artifacts, potentially affecting monitoring outcomes. In this paper, we present a novel registration-free approach for monitoring PCD of complex shapes, eliminating the need for both registration and mesh reconstruction. Our proposal consists of two alternative feature learning methods and a common monitoring scheme. Feature learning methods leverage intrinsic geometric properties of the shape, captured via the Laplacian and geodesic distances. In the monitoring scheme, thresholding techniques are used to further select intrinsic features most indicative of potential out-of-control conditions. Numerical experiments and case studies highlight the effectiveness of the proposed approach in identifying different types of defects.
>
---
#### [new 021] Culture in Action: Evaluating Text-to-Image Models through Social Activities
- **分类: cs.CV**

- **简介: 该论文提出CULTIVate基准，评估文本到图像模型在跨文化社会活动中的文化准确性，解决现有评估忽视日常文化行为的问题，构建多维度指标并揭示模型对全球南方的系统性偏差。**

- **链接: [http://arxiv.org/pdf/2511.05681v1](http://arxiv.org/pdf/2511.05681v1)**

> **作者:** Sina Malakouti; Boqing Gong; Adriana Kovashka
>
> **摘要:** Text-to-image (T2I) diffusion models achieve impressive photorealism by training on large-scale web data, but models inherit cultural biases and fail to depict underrepresented regions faithfully. Existing cultural benchmarks focus mainly on object-centric categories (e.g., food, attire, and architecture), overlooking the social and daily activities that more clearly reflect cultural norms. Few metrics exist for measuring cultural faithfulness. We introduce CULTIVate, a benchmark for evaluating T2I models on cross-cultural activities (e.g., greetings, dining, games, traditional dances, and cultural celebrations). CULTIVate spans 16 countries with 576 prompts and more than 19,000 images, and provides an explainable descriptor-based evaluation framework across multiple cultural dimensions, including background, attire, objects, and interactions. We propose four metrics to measure cultural alignment, hallucination, exaggerated elements, and diversity. Our findings reveal systematic disparities: models perform better for global north countries than for the global south, with distinct failure modes across T2I systems. Human studies confirm that our metrics correlate more strongly with human judgments than existing text-image metrics.
>
---
#### [new 022] MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition
- **分类: cs.CV**

- **简介: MoRA提出一种参数高效的视觉识别微调方法，解决多模态输入缺失问题，通过共享与特异参数建模跨模态交互，在保持性能的同时显著降低计算与参数开销。**

- **链接: [http://arxiv.org/pdf/2511.06225v1](http://arxiv.org/pdf/2511.06225v1)**

> **作者:** Shu Zhao; Nilesh Ahuja; Tan Yu; Tianyi Shen; Vijaykrishnan Narayanan
>
> **摘要:** Pre-trained vision language models have shown remarkable performance on visual recognition tasks, but they typically assume the availability of complete multimodal inputs during both training and inference. In real-world scenarios, however, modalities may be missing due to privacy constraints, collection difficulties, or resource limitations. While previous approaches have addressed this challenge using prompt learning techniques, they fail to capture the cross-modal relationships necessary for effective multimodal visual recognition and suffer from inevitable computational overhead. In this paper, we introduce MoRA, a parameter-efficient fine-tuning method that explicitly models cross-modal interactions while maintaining modality-specific adaptations. MoRA introduces modality-common parameters between text and vision encoders, enabling bidirectional knowledge transfer. Additionally, combined with the modality-specific parameters, MoRA allows the backbone model to maintain inter-modality interaction and enable intra-modality flexibility. Extensive experiments on standard benchmarks demonstrate that MoRA achieves an average performance improvement in missing-modality scenarios by 5.24% and uses only 25.90% of the inference time compared to the SOTA method while requiring only 0.11% of trainable parameters compared to full fine-tuning.
>
---
#### [new 023] Beyond Softmax: Dual-Branch Sigmoid Architecture for Accurate Class Activation Maps
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出双分支Sigmoid架构，解决传统CAM因Softmax导致的显著性偏差与符号混淆问题。通过解耦分类与定位，保留特征贡献的幅值与符号，提升激活图准确性，且不损失分类性能。**

- **链接: [http://arxiv.org/pdf/2511.05590v1](http://arxiv.org/pdf/2511.05590v1)**

> **作者:** Yoojin Oh; Junhyug Noh
>
> **备注:** Accepted at BMVC 2025
>
> **摘要:** Class Activation Mapping (CAM) and its extensions have become indispensable tools for visualizing the evidence behind deep network predictions. However, by relying on a final softmax classifier, these methods suffer from two fundamental distortions: additive logit shifts that arbitrarily bias importance scores, and sign collapse that conflates excitatory and inhibitory features. We propose a simple, architecture-agnostic dual-branch sigmoid head that decouples localization from classification. Given any pretrained model, we clone its classification head into a parallel branch ending in per-class sigmoid outputs, freeze the original softmax head, and fine-tune only the sigmoid branch with class-balanced binary supervision. At inference, softmax retains recognition accuracy, while class evidence maps are generated from the sigmoid branch -- preserving both magnitude and sign of feature contributions. Our method integrates seamlessly with most CAM variants and incurs negligible overhead. Extensive evaluations on fine-grained tasks (CUB-200-2011, Stanford Cars) and WSOL benchmarks (ImageNet-1K, OpenImages30K) show improved explanation fidelity and consistent Top-1 Localization gains -- without any drop in classification accuracy. Code is available at https://github.com/finallyupper/beyond-softmax.
>
---
#### [new 024] How Reasoning Influences Intersectional Biases in Vision Language Models
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型（VLMs）在职业预测任务中因统计关联导致的交叉性偏见，通过分析五种VLM的推理过程，揭示其偏见源于与人类推理的偏差，呼吁在部署前对齐模型推理与人类价值观。**

- **链接: [http://arxiv.org/pdf/2511.06005v1](http://arxiv.org/pdf/2511.06005v1)**

> **作者:** Adit Desai; Sudipta Roy; Mohna Chakraborty
>
> **摘要:** Vision Language Models (VLMs) are increasingly deployed across downstream tasks, yet their training data often encode social biases that surface in outputs. Unlike humans, who interpret images through contextual and social cues, VLMs process them through statistical associations, often leading to reasoning that diverges from human reasoning. By analyzing how a VLM reasons, we can understand how inherent biases are perpetuated and can adversely affect downstream performance. To examine this gap, we systematically analyze social biases in five open-source VLMs for an occupation prediction task, on the FairFace dataset. Across 32 occupations and three different prompting styles, we elicit both predictions and reasoning. Our findings reveal that the biased reasoning patterns systematically underlie intersectional disparities, highlighting the need to align VLM reasoning with human values prior to its downstream deployment.
>
---
#### [new 025] StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression
- **分类: cs.CV**

- **简介: StreamKV面向流式视频问答任务，解决长视频中KV缓存检索与压缩效率低的问题，提出基于语义分段的动态缓存检索与压缩框架，无需训练，显著提升准确率与资源效率。**

- **链接: [http://arxiv.org/pdf/2511.07278v1](http://arxiv.org/pdf/2511.07278v1)**

> **作者:** Yilong Chen; Xiang Bai; Zhibin Wang; Chengyu Bai; Yuhan Dai; Ming Lu; Shanghang Zhang
>
> **摘要:** Video Large Language Models (Video-LLMs) have demonstrated significant potential in the areas of video captioning, search, and summarization. However, current Video-LLMs still face challenges with long real-world videos. Recent methods have introduced a retrieval mechanism that retrieves query-relevant KV caches for question answering, enhancing the efficiency and accuracy of long real-world videos. However, the compression and retrieval of KV caches are still not fully explored. In this paper, we propose \textbf{StreamKV}, a training-free framework that seamlessly equips Video-LLMs with advanced KV cache retrieval and compression. Compared to previous methods that used uniform partitioning, StreamKV dynamically partitions video streams into semantic segments, which better preserves semantic information. For KV cache retrieval, StreamKV calculates a summary vector for each segment to retain segment-level information essential for retrieval. For KV cache compression, StreamKV introduces a guidance prompt designed to capture the key semantic elements within each segment, ensuring only the most informative KV caches are retained for answering questions. Moreover, StreamKV unifies KV cache retrieval and compression within a single module, performing both in a layer-adaptive manner, thereby further improving the effectiveness of streaming video question answering. Extensive experiments on public StreamingVQA benchmarks demonstrate that StreamKV significantly outperforms existing Online Video-LLMs, achieving superior accuracy while substantially improving both memory efficiency and computational latency. The code has been released at https://github.com/sou1p0wer/StreamKV.
>
---
#### [new 026] Global Multiple Extraction Network for Low-Resolution Facial Expression Recognition
- **分类: cs.CV**

- **简介: 该论文面向低分辨率人脸表情识别任务，针对细节缺失与全局建模薄弱问题，提出GME-Net，融合注意力局部特征提取与多尺度全局特征模块，提升特征判别力，在多个数据集上实现最优性能。**

- **链接: [http://arxiv.org/pdf/2511.05938v1](http://arxiv.org/pdf/2511.05938v1)**

> **作者:** Jingyi Shi
>
> **备注:** 12 pages
>
> **摘要:** Facial expression recognition, as a vital computer vision task, is garnering significant attention and undergoing extensive research. Although facial expression recognition algorithms demonstrate impressive performance on high-resolution images, their effectiveness tends to degrade when confronted with low-resolution images. We find it is because: 1) low-resolution images lack detail information; 2) current methods complete weak global modeling, which make it difficult to extract discriminative features. To alleviate the above issues, we proposed a novel global multiple extraction network (GME-Net) for low-resolution facial expression recognition, which incorporates 1) a hybrid attention-based local feature extraction module with attention similarity knowledge distillation to learn image details from high-resolution network; 2) a multi-scale global feature extraction module with quasi-symmetric structure to mitigate the influence of local image noise and facilitate capturing global image features. As a result, our GME-Net is capable of extracting expression-related discriminative features. Extensive experiments conducted on several widely-used datasets demonstrate that the proposed GME-Net can better recognize low-resolution facial expression and obtain superior performance than existing solutions.
>
---
#### [new 027] 3D-ANC: Adaptive Neural Collapse for Robust 3D Point Cloud Recognition
- **分类: cs.CV; cs.CR**

- **简介: 该论文针对3D点云识别中的对抗攻击脆弱性问题，提出3D-ANC方法，利用神经坍缩机制构建解耦特征空间，结合自适应训练策略缓解类别不平衡与几何相似性挑战，显著提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.07040v1](http://arxiv.org/pdf/2511.07040v1)**

> **作者:** Yuanmin Huang; Wenxuan Li; Mi Zhang; Xiaohan Zhang; Xiaoyu You; Min Yang
>
> **备注:** AAAI 2026
>
> **摘要:** Deep neural networks have recently achieved notable progress in 3D point cloud recognition, yet their vulnerability to adversarial perturbations poses critical security challenges in practical deployments. Conventional defense mechanisms struggle to address the evolving landscape of multifaceted attack patterns. Through systematic analysis of existing defenses, we identify that their unsatisfactory performance primarily originates from an entangled feature space, where adversarial attacks can be performed easily. To this end, we present 3D-ANC, a novel approach that capitalizes on the Neural Collapse (NC) mechanism to orchestrate discriminative feature learning. In particular, NC depicts where last-layer features and classifier weights jointly evolve into a simplex equiangular tight frame (ETF) arrangement, establishing maximally separable class prototypes. However, leveraging this advantage in 3D recognition confronts two substantial challenges: (1) prevalent class imbalance in point cloud datasets, and (2) complex geometric similarities between object categories. To tackle these obstacles, our solution combines an ETF-aligned classification module with an adaptive training framework consisting of representation-balanced learning (RBL) and dynamic feature direction loss (FDL). 3D-ANC seamlessly empowers existing models to develop disentangled feature spaces despite the complexity in 3D data distribution. Comprehensive evaluations state that 3D-ANC significantly improves the robustness of models with various structures on two datasets. For instance, DGCNN's classification accuracy is elevated from 27.2% to 80.9% on ModelNet40 -- a 53.7% absolute gain that surpasses leading baselines by 34.0%.
>
---
#### [new 028] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers
- **分类: cs.CV**

- **简介: 该论文研究视觉Transformer（ViT）中特征蒸馏失效问题，提出“蒸馏动态”框架，揭示教师与学生模型表征范式不匹配导致负迁移，指出需超越简单特征模仿，设计符合ViT表示约束的压缩方法。**

- **链接: [http://arxiv.org/pdf/2511.06848v1](http://arxiv.org/pdf/2511.06848v1)**

> **作者:** Huiyuan Tian; Bonan Xu Shijian Li
>
> **备注:** Accepted to AAAI 2026. Submitted version
>
> **摘要:** While feature-based knowledge distillation has proven highly effective for compressing CNNs, these techniques unexpectedly fail when applied to Vision Transformers (ViTs), often performing worse than simple logit-based distillation. We provide the first comprehensive analysis of this phenomenon through a novel analytical framework termed as ``distillation dynamics", combining frequency spectrum analysis, information entropy metrics, and activation magnitude tracking. Our investigation reveals that ViTs exhibit a distinctive U-shaped information processing pattern: initial compression followed by expansion. We identify the root cause of negative transfer in feature distillation: a fundamental representational paradigm mismatch between teacher and student models. Through frequency-domain analysis, we show that teacher models employ distributed, high-dimensional encoding strategies in later layers that smaller student models cannot replicate due to limited channel capacity. This mismatch causes late-layer feature alignment to actively harm student performance. Our findings reveal that successful knowledge transfer in ViTs requires moving beyond naive feature mimicry to methods that respect these fundamental representational constraints, providing essential theoretical guidance for designing effective ViTs compression strategies. All source code and experimental logs are provided in the supplementary material.
>
---
#### [new 029] Robust Nearest Neighbour Retrieval Using Targeted Manifold Manipulation
- **分类: cs.CV**

- **简介: 该论文提出TMM-NN，用于改进最近邻检索。通过在查询图像添加触发补丁，诱导模型将语义相似样本导向虚拟类别，以响应置信度排序邻居，替代传统距离度量，提升噪声鲁棒性与跨任务性能。**

- **链接: [http://arxiv.org/pdf/2511.06261v1](http://arxiv.org/pdf/2511.06261v1)**

> **作者:** B. Ghosh; H. Harikumar; S. Rana
>
> **摘要:** Nearest-neighbour retrieval is central to classification and explainable-AI pipelines, but current practice relies on hand-tuning feature layers and distance metrics. We propose Targeted Manifold Manipulation-Nearest Neighbour (TMM-NN), which reconceptualises retrieval by assessing how readily each sample can be nudged into a designated region of the feature manifold; neighbourhoods are defined by a sample's responsiveness to a targeted perturbation rather than absolute geometric distance. TMM-NN implements this through a lightweight, query-specific trigger patch. The patch is added to the query image, and the network is weakly ``backdoored'' so that any input with the patch is steered toward a dummy class. Images similar to the query need only a slight shift and are classified as the dummy class with high probability, while dissimilar ones are less affected. By ranking candidates by this confidence, TMM-NN retrieves the most semantically related neighbours. Robustness analysis and benchmark experiments confirm this trigger-based ranking outperforms traditional metrics under noise and across diverse tasks.
>
---
#### [new 030] Automated Invoice Data Extraction: Using LLM and OCR
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种融合OCR与LLM的AI平台，解决传统OCR在发票布局多变、手写文本和低质扫描下的提取难题，通过视觉NER与图分析实现高精度、自动化数据抽取。**

- **链接: [http://arxiv.org/pdf/2511.05547v1](http://arxiv.org/pdf/2511.05547v1)**

> **作者:** Advait Thakur; Khushi Khanchandani; Akshita Shetty; Chaitravi Reddy; Ritisa Behera
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Conventional Optical Character Recognition (OCR) systems are challenged by variant invoice layouts, handwritten text, and low- quality scans, which are often caused by strong template dependencies that restrict their flexibility across different document structures and layouts. Newer solutions utilize advanced deep learning models such as Convolutional Neural Networks (CNN) as well as Transformers, and domain-specific models for better layout analysis and accuracy across various sections over varied document types. Large Language Models (LLMs) have revolutionized extraction pipelines at their core with sophisticated entity recognition and semantic comprehension to support complex contextual relationship mapping without direct programming specification. Visual Named Entity Recognition (NER) capabilities permit extraction from invoice images with greater contextual sensitivity and much higher accuracy rates than older approaches. Existing industry best practices utilize hybrid architectures that blend OCR technology and LLM for maximum scalability and minimal human intervention. This work introduces a holistic Artificial Intelligence (AI) platform combining OCR, deep learning, LLMs, and graph analytics to achieve unprecedented extraction quality and consistency.
>
---
#### [new 031] Real-Time LiDAR Super-Resolution via Frequency-Aware Multi-Scale Fusion
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出FLASH，用于LiDAR超分辨率任务，解决低分辨率传感器感知质量差的问题。通过频域-空域双域融合与自适应多尺度机制，在单次前向传播下实现高精度、实时的3D点云增强，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.07377v1](http://arxiv.org/pdf/2511.07377v1)**

> **作者:** June Moh Goo; Zichao Zeng; Jan Boehm
>
> **摘要:** LiDAR super-resolution addresses the challenge of achieving high-quality 3D perception from cost-effective, low-resolution sensors. While recent transformer-based approaches like TULIP show promise, they remain limited to spatial-domain processing with restricted receptive fields. We introduce FLASH (Frequency-aware LiDAR Adaptive Super-resolution with Hierarchical fusion), a novel framework that overcomes these limitations through dual-domain processing. FLASH integrates two key innovations: (i) Frequency-Aware Window Attention that combines local spatial attention with global frequency-domain analysis via FFT, capturing both fine-grained geometry and periodic scanning patterns at log-linear complexity. (ii) Adaptive Multi-Scale Fusion that replaces conventional skip connections with learned position-specific feature aggregation, enhanced by CBAM attention for dynamic feature selection. Extensive experiments on KITTI demonstrate that FLASH achieves state-of-the-art performance across all evaluation metrics, surpassing even uncertainty-enhanced baselines that require multiple forward passes. Notably, FLASH outperforms TULIP with Monte Carlo Dropout while maintaining single-pass efficiency, which enables real-time deployment. The consistent superiority across all distance ranges validates that our dual-domain approach effectively handles uncertainty through architectural design rather than computationally expensive stochastic inference, making it practical for autonomous systems.
>
---
#### [new 032] Video Dataset for Surgical Phase, Keypoint, and Instrument Recognition in Laparoscopic Surgery (PhaKIR)
- **分类: cs.CV**

- **简介: 该论文发布PhaKIR数据集，面向腹腔镜手术场景理解，解决多任务联合标注与跨中心数据缺失问题，首次提供手术阶段、器械关键点与分割的同步标注，并支持时序分析，助力计算机辅助手术研究。**

- **链接: [http://arxiv.org/pdf/2511.06549v1](http://arxiv.org/pdf/2511.06549v1)**

> **作者:** Tobias Rueckert; Raphaela Maerkl; David Rauber; Leonard Klausmann; Max Gutbrod; Daniel Rueckert; Hubertus Feussner; Dirk Wilhelm; Christoph Palm
>
> **备注:** 9 pages, 5 figures, 4 tables
>
> **摘要:** Robotic- and computer-assisted minimally invasive surgery (RAMIS) is increasingly relying on computer vision methods for reliable instrument recognition and surgical workflow understanding. Developing such systems often requires large, well-annotated datasets, but existing resources often address isolated tasks, neglect temporal dependencies, or lack multi-center variability. We present the Surgical Procedure Phase, Keypoint, and Instrument Recognition (PhaKIR) dataset, comprising eight complete laparoscopic cholecystectomy videos recorded at three medical centers. The dataset provides frame-level annotations for three interconnected tasks: surgical phase recognition (485,875 frames), instrument keypoint estimation (19,435 frames), and instrument instance segmentation (19,435 frames). PhaKIR is, to our knowledge, the first multi-institutional dataset to jointly provide phase labels, instrument pose information, and pixel-accurate instrument segmentations, while also enabling the exploitation of temporal context since full surgical procedure sequences are available. It served as the basis for the PhaKIR Challenge as part of the Endoscopic Vision (EndoVis) Challenge at MICCAI 2024 to benchmark methods in surgical scene understanding, thereby further validating the dataset's quality and relevance. The dataset is publicly available upon request via the Zenodo platform.
>
---
#### [new 033] ProcGen3D: Learning Neural Procedural Graph Representations for Image-to-3D Reconstruction
- **分类: cs.CV**

- **简介: 论文提出ProcGen3D，将3D重建转化为生成程序化图结构的任务，通过图序列化与Transformer预测，结合MCTS采样，从单张RGB图像生成高保真3D资产，超越现有方法并实现跨域泛化。**

- **链接: [http://arxiv.org/pdf/2511.07142v1](http://arxiv.org/pdf/2511.07142v1)**

> **作者:** Xinyi Zhang; Daoyi Gao; Naiqi Li; Angela Dai
>
> **备注:** Project Page: https://xzhang-t.github.io/project/ProcGen3D/
>
> **摘要:** We introduce ProcGen3D, a new approach for 3D content creation by generating procedural graph abstractions of 3D objects, which can then be decoded into rich, complex 3D assets. Inspired by the prevalent use of procedural generators in production 3D applications, we propose a sequentialized, graph-based procedural graph representation for 3D assets. We use this to learn to approximate the landscape of a procedural generator for image-based 3D reconstruction. We employ edge-based tokenization to encode the procedural graphs, and train a transformer prior to predict the next token conditioned on an input RGB image. Crucially, to enable better alignment of our generated outputs to an input image, we incorporate Monte Carlo Tree Search (MCTS) guided sampling into our generation process, steering output procedural graphs towards more image-faithful reconstructions. Our approach is applicable across a variety of objects that can be synthesized with procedural generators. Extensive experiments on cacti, trees, and bridges show that our neural procedural graph generation outperforms both state-of-the-art generative 3D methods and domain-specific modeling techniques. Furthermore, this enables improved generalization on real-world input images, despite training only on synthetic data.
>
---
#### [new 034] VideoSSR: Video Self-Supervised Reinforcement Learning
- **分类: cs.CV**

- **简介: 论文提出VideoSSR，一种视频自监督强化学习框架，解决MLLMs在视频理解中依赖昂贵人工标注的问题，通过三个自监督任务生成高质量训练数据，显著提升模型在多视频任务上的性能。**

- **链接: [http://arxiv.org/pdf/2511.06281v1](http://arxiv.org/pdf/2511.06281v1)**

> **作者:** Zefeng He; Xiaoye Qu; Yafu Li; Siyuan Huang; Daizong Liu; Yu Cheng
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has substantially advanced the video understanding capabilities of Multimodal Large Language Models (MLLMs). However, the rapid progress of MLLMs is outpacing the complexity of existing video datasets, while the manual annotation of new, high-quality data remains prohibitively expensive. This work investigates a pivotal question: Can the rich, intrinsic information within videos be harnessed to self-generate high-quality, verifiable training data? To investigate this, we introduce three self-supervised pretext tasks: Anomaly Grounding, Object Counting, and Temporal Jigsaw. We construct the Video Intrinsic Understanding Benchmark (VIUBench) to validate their difficulty, revealing that current state-of-the-art MLLMs struggle significantly on these tasks. Building upon these pretext tasks, we develop the VideoSSR-30K dataset and propose VideoSSR, a novel video self-supervised reinforcement learning framework for RLVR. Extensive experiments across 17 benchmarks, spanning four major video domains (General Video QA, Long Video QA, Temporal Grounding, and Complex Reasoning), demonstrate that VideoSSR consistently enhances model performance, yielding an average improvement of over 5\%. These results establish VideoSSR as a potent foundational framework for developing more advanced video understanding in MLLMs. The code is available at https://github.com/lcqysl/VideoSSR.
>
---
#### [new 035] EVLP:Learning Unified Embodied Vision-Language Planner with Reinforced Supervised Fine-Tuning
- **分类: cs.CV; cs.AI**

- **简介: EVLP提出统一的视觉-语言规划框架，解决多模态长时程操作任务中推理与生成不一致问题，通过统一生成、动态预训练和强化微调，实现语言指令与视觉动作的协同建模。**

- **链接: [http://arxiv.org/pdf/2511.05553v1](http://arxiv.org/pdf/2511.05553v1)**

> **作者:** Xinyan Cai; Shiguang Wu; Dafeng Chi; Yuzheng Zhuang; Xingyue Quan; Jianye Hao; Qiang Guan
>
> **摘要:** In complex embodied long-horizon manipulation tasks, effective task decomposition and execution require synergistic integration of textual logical reasoning and visual-spatial imagination to ensure efficient and accurate operation. Current methods fail to adopt a unified generation framework for multimodal planning, lead to inconsistent in multimodal planning. To address this challenge, we present \textbf{EVLP (Embodied Vision-Language Planner)}, an innovative multimodal unified generation framework that jointly models linguistic reasoning and visual generation. Our approach achieves multimodal planning for long-horizon tasks through a novel training pipeline incorporating dynamic pretraining and reinforced alignment. Our core innovations consist of three key components: \textbf{1) Unified Multimodal Generation Framework}: For understanding, We integrate semantic information with spatial features to provide comprehensive visual perception. For generation, we directly learn the joint distribution of discrete images for one-step visual synthesis, enabling coordinated language-visual modeling through learnable cross-modal attention mechanisms. \textbf{2) Dynamic Perception Pretraining}: We propose a bidirectional dynamic alignment strategy employing inverse dynamics tasks and forward dynamics tasks, effectively strengthening multimodal correlations within a unified feature space. \textbf{3) Reinforced Supervised Fine-Tuning}: While conducting instruction-based fine-tuning in the unified generation space, we construct a reinforce loss to align the spatial logic between textual actions and generated images, enabling the model to acquire spatio-awared multimodal planning capabilities.
>
---
#### [new 036] Light-Field Dataset for Disparity Based Depth Estimation
- **分类: cs.CV**

- **简介: 该论文面向基于视差的光场深度估计任务，解决现有数据集缺失与焦距影响未系统研究的问题，构建并发布了包含285个真实与13个合成光场图像的公开数据集，支持算法开发与评估。**

- **链接: [http://arxiv.org/pdf/2511.05866v1](http://arxiv.org/pdf/2511.05866v1)**

> **作者:** Suresh Nehra; Aupendu Kar; Jayanta Mukhopadhyay; Prabir Kumar Biswas
>
> **备注:** This paper has been accepted to ACM ICVGIP 2025
>
> **摘要:** A Light Field (LF) camera consists of an additional two-dimensional array of micro-lenses placed between the main lens and sensor, compared to a conventional camera. The sensor pixels under each micro-lens receive light from a sub-aperture of the main lens. This enables the image sensor to capture both spatial information and the angular resolution of a scene point. This additional angular information is used to estimate the depth of a 3-D scene. The continuum of virtual viewpoints in light field data enables efficient depth estimation using Epipolar Line Images (EPIs) with robust occlusion handling. However, the trade-off between angular information and spatial information is very critical and depends on the focal position of the camera. To design, develop, implement, and test novel disparity-based light field depth estimation algorithms, the availability of suitable light field image datasets is essential. In this paper, a publicly available light field image dataset is introduced and thoroughly described. We have also demonstrated the effect of focal position on the disparity of a 3-D point as well as the shortcomings of the currently available light field dataset. The proposed dataset contains 285 light field images captured using a Lytro Illum LF camera and 13 synthetic LF images. The proposed dataset also comprises a synthetic dataset with similar disparity characteristics to those of a real light field camera. A real and synthetic stereo light field dataset is also created by using a mechanical gantry system and Blender. The dataset is available at https://github.com/aupendu/light-field-dataset.
>
---
#### [new 037] MiVID: Multi-Strategic Self-Supervision for Video Frame Interpolation using Diffusion Model
- **分类: cs.CV; cs.AI**

- **简介: MiVID提出一种自监督扩散模型，用于视频帧插值，无需光流或高帧率标注，通过3D U-Net与时空注意力机制，结合混合掩码策略，实现低资源下高效帧合成，解决遮挡与运动模糊问题。**

- **链接: [http://arxiv.org/pdf/2511.06019v1](http://arxiv.org/pdf/2511.06019v1)**

> **作者:** Priyansh Srivastava; Romit Chatterjee; Abir Sen; Aradhana Behura; Ratnakar Dash
>
> **摘要:** Video Frame Interpolation (VFI) remains a cornerstone in video enhancement, enabling temporal upscaling for tasks like slow-motion rendering, frame rate conversion, and video restoration. While classical methods rely on optical flow and learning-based models assume access to dense ground-truth, both struggle with occlusions, domain shifts, and ambiguous motion. This article introduces MiVID, a lightweight, self-supervised, diffusion-based framework for video interpolation. Our model eliminates the need for explicit motion estimation by combining a 3D U-Net backbone with transformer-style temporal attention, trained under a hybrid masking regime that simulates occlusions and motion uncertainty. The use of cosine-based progressive masking and adaptive loss scheduling allows our network to learn robust spatiotemporal representations without any high-frame-rate supervision. Our framework is evaluated on UCF101-7 and DAVIS-7 datasets. MiVID is trained entirely on CPU using the datasets and 9-frame video segments, making it a low-resource yet highly effective pipeline. Despite these constraints, our model achieves optimal results at just 50 epochs, competitive with several supervised baselines.This work demonstrates the power of self-supervised diffusion priors for temporally coherent frame synthesis and provides a scalable path toward accessible and generalizable VFI systems.
>
---
#### [new 038] YoNoSplat: You Only Need One Model for Feedforward 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: YoNoSplat提出一种单模型feedforward方法，实现从任意数量图像（含未标定输入）快速重建3D高斯泼溅场景，解决联合优化高斯与相机参数的训练不稳定与尺度歧义问题，兼具高效性与SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.07321v1](http://arxiv.org/pdf/2511.07321v1)**

> **作者:** Botao Ye; Boqi Chen; Haofei Xu; Daniel Barath; Marc Pollefeys
>
> **摘要:** Fast and flexible 3D scene reconstruction from unstructured image collections remains a significant challenge. We present YoNoSplat, a feedforward model that reconstructs high-quality 3D Gaussian Splatting representations from an arbitrary number of images. Our model is highly versatile, operating effectively with both posed and unposed, calibrated and uncalibrated inputs. YoNoSplat predicts local Gaussians and camera poses for each view, which are aggregated into a global representation using either predicted or provided poses. To overcome the inherent difficulty of jointly learning 3D Gaussians and camera parameters, we introduce a novel mixing training strategy. This approach mitigates the entanglement between the two tasks by initially using ground-truth poses to aggregate local Gaussians and gradually transitioning to a mix of predicted and ground-truth poses, which prevents both training instability and exposure bias. We further resolve the scale ambiguity problem by a novel pairwise camera-distance normalization scheme and by embedding camera intrinsics into the network. Moreover, YoNoSplat also predicts intrinsic parameters, making it feasible for uncalibrated inputs. YoNoSplat demonstrates exceptional efficiency, reconstructing a scene from 100 views (at 280x518 resolution) in just 2.69 seconds on an NVIDIA GH200 GPU. It achieves state-of-the-art performance on standard benchmarks in both pose-free and pose-dependent settings. Our project page is at https://botaoye.github.io/yonosplat/.
>
---
#### [new 039] VLDrive: Vision-Augmented Lightweight MLLMs for Efficient Language-grounded Autonomous Driving
- **分类: cs.CV**

- **简介: VLDrive面向语言引导的自动驾驶任务，解决视觉表征弱与LLM参数量大的问题，提出轻量多模态模型，通过动态视觉剪枝与距离解耦注意力机制，实现参数减少81%的同时提升驾驶性能。**

- **链接: [http://arxiv.org/pdf/2511.06256v1](http://arxiv.org/pdf/2511.06256v1)**

> **作者:** Ruifei Zhang; Wei Zhang; Xiao Tan; Sibei Yang; Xiang Wan; Xiaonan Luo; Guanbin Li
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Recent advancements in language-grounded autonomous driving have been significantly promoted by the sophisticated cognition and reasoning capabilities of large language models (LLMs). However, current LLM-based approaches encounter critical challenges: (1) Failure analysis reveals that frequent collisions and obstructions, stemming from limitations in visual representations, remain primary obstacles to robust driving performance. (2) The substantial parameters of LLMs pose considerable deployment hurdles. To address these limitations, we introduce VLDrive, a novel approach featuring a lightweight MLLM architecture with enhanced vision components. VLDrive achieves compact visual tokens through innovative strategies, including cycle-consistent dynamic visual pruning and memory-enhanced feature aggregation. Furthermore, we propose a distance-decoupled instruction attention mechanism to improve joint visual-linguistic feature learning, particularly for long-range visual tokens. Extensive experiments conducted in the CARLA simulator demonstrate VLDrive`s effectiveness. Notably, VLDrive achieves state-of-the-art driving performance while reducing parameters by 81% (from 7B to 1.3B), yielding substantial driving score improvements of 15.4%, 16.8%, and 7.6% at tiny, short, and long distances, respectively, in closed-loop evaluations. Code is available at https://github.com/ReaFly/VLDrive.
>
---
#### [new 040] Performance Decay in Deepfake Detection: The Limitations of Training on Outdated Data
- **分类: cs.CV; 68T07, 68T45**

- **简介: 该论文研究深度伪造检测任务，揭示模型在过时数据上训练后性能急剧下降的问题，提出两阶段检测方法并发现帧级静态特征是关键，强调需持续更新数据集与开发帧级特征检测器以应对快速演进的伪造技术。**

- **链接: [http://arxiv.org/pdf/2511.07009v1](http://arxiv.org/pdf/2511.07009v1)**

> **作者:** Jack Richings; Margaux Leblanc; Ian Groves; Victoria Nockles
>
> **摘要:** The continually advancing quality of deepfake technology exacerbates the threats of disinformation, fraud, and harassment by making maliciously-generated synthetic content increasingly difficult to distinguish from reality. We introduce a simple yet effective two-stage detection method that achieves an AUROC of over 99.8% on contemporary deepfakes. However, this high performance is short-lived. We show that models trained on this data suffer a recall drop of over 30% when evaluated on deepfakes created with generation techniques from just six months later, demonstrating significant decay as threats evolve. Our analysis reveals two key insights for robust detection. Firstly, continued performance requires the ongoing curation of large, diverse datasets. Second, predictive power comes primarily from static, frame-level artifacts, not temporal inconsistencies. The future of effective deepfake detection therefore depends on rapid data collection and the development of advanced frame-level feature detectors.
>
---
#### [new 041] EIDSeg: A Pixel-Level Semantic Segmentation Dataset for Post-Earthquake Damage Assessment from Social Media Images
- **分类: cs.CV**

- **简介: 论文提出EIDSeg，首个用于地震后社交媒体图像的像素级语义分割数据集，解决传统遥感方法依赖昂贵数据、粒度粗的问题，构建了3,266张五类损毁标注图像，并验证了EoMT模型的优越性能。**

- **链接: [http://arxiv.org/pdf/2511.06456v1](http://arxiv.org/pdf/2511.06456v1)**

> **作者:** Huili Huang; Chengeng Liu; Danrong Zhang; Shail Patel; Anastasiya Masalava; Sagar Sadak; Parisa Babolhavaeji; WeiHong Low; Max Mahdi Roozbahani; J. David Frost
>
> **备注:** Camera-Ready for AAAI-AISI26
>
> **摘要:** Rapid post-earthquake damage assessment is crucial for rescue and resource planning. Still, existing remote sensing methods depend on costly aerial images, expert labeling, and produce only binary damage maps for early-stage evaluation. Although ground-level images from social networks provide a valuable source to fill this gap, a large pixel-level annotated dataset for this task is still unavailable. We introduce EIDSeg, the first large-scale semantic segmentation dataset specifically for post-earthquake social media imagery. The dataset comprises 3,266 images from nine major earthquakes (2008-2023), annotated across five classes of infrastructure damage: Undamaged Building, Damaged Building, Destroyed Building, Undamaged Road, and Damaged Road. We propose a practical three-phase cross-disciplinary annotation protocol with labeling guidelines that enables consistent segmentation by non-expert annotators, achieving over 70% inter-annotator agreement. We benchmark several state-of-the-art segmentation models, identifying Encoder-only Mask Transformer (EoMT) as the top-performing method with a Mean Intersection over Union (mIoU) of 80.8%. By unlocking social networks' rich ground-level perspective, our work paves the way for a faster, finer-grained damage assessment in the post-earthquake scenario.
>
---
#### [new 042] K-Stain: Keypoint-Driven Correspondence for H&E-to-IHC Virtual Staining
- **分类: cs.CV**

- **简介: K-Stain提出一种基于关键点的H&E到IHC虚拟染色方法，解决组织切片空间错位导致的结构失真问题，通过关键点检测、增强生成与引导判别，提升合成图像的精度与视觉一致性。**

- **链接: [http://arxiv.org/pdf/2511.06709v1](http://arxiv.org/pdf/2511.06709v1)**

> **作者:** Sicheng Yang; Zhaohu Xing; Haipeng Zhou; Lei Zhu
>
> **摘要:** Virtual staining offers a promising method for converting Hematoxylin and Eosin (H&E) images into Immunohistochemical (IHC) images, eliminating the need for costly chemical processes. However, existing methods often struggle to utilize spatial information effectively due to misalignment in tissue slices. To overcome this challenge, we leverage keypoints as robust indicators of spatial correspondence, enabling more precise alignment and integration of structural details in synthesized IHC images. We introduce K-Stain, a novel framework that employs keypoint-based spatial and semantic relationships to enhance synthesized IHC image fidelity. K-Stain comprises three main components: (1) a Hierarchical Spatial Keypoint Detector (HSKD) for identifying keypoints in stain images, (2) a Keypoint-aware Enhancement Generator (KEG) that integrates these keypoints during image generation, and (3) a Keypoint Guided Discriminator (KGD) that improves the discriminator's sensitivity to spatial details. Our approach leverages contextual information from adjacent slices, resulting in more accurate and visually consistent IHC images. Extensive experiments show that K-Stain outperforms state-of-the-art methods in quantitative metrics and visual quality.
>
---
#### [new 043] Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
- **分类: cs.CV**

- **简介: 论文提出Inpaint360GS，面向360°场景的三维目标感知修复任务，解决多目标遮挡与视图一致性难题，通过2D分割蒸馏与虚拟相机引导，实现高保真三维补全，并构建专用数据集推动领域发展。**

- **链接: [http://arxiv.org/pdf/2511.06457v1](http://arxiv.org/pdf/2511.06457v1)**

> **作者:** Shaoxiang Wang; Shihong Zhang; Christen Millerdurai; Rüdiger Westermann; Didier Stricker; Alain Pagani
>
> **备注:** WACV 2026, project page: https://dfki-av.github.io/inpaint360gs/
>
> **摘要:** Despite recent advances in single-object front-facing inpainting using NeRF and 3D Gaussian Splatting (3DGS), inpainting in complex 360{\deg} scenes remains largely underexplored. This is primarily due to three key challenges: (i) identifying target objects in the 3D field of 360{\deg} environments, (ii) dealing with severe occlusions in multi-object scenes, which makes it hard to define regions to inpaint, and (iii) maintaining consistent and high-quality appearance across views effectively. To tackle these challenges, we propose Inpaint360GS, a flexible 360{\deg} editing framework based on 3DGS that supports multi-object removal and high-fidelity inpainting in 3D space. By distilling 2D segmentation into 3D and leveraging virtual camera views for contextual guidance, our method enables accurate object-level editing and consistent scene completion. We further introduce a new dataset tailored for 360{\deg} inpainting, addressing the lack of ground truth object-free scenes. Experiments demonstrate that Inpaint360GS outperforms existing baselines and achieves state-of-the-art performance. Project page: https://dfki-av.github.io/inpaint360gs/
>
---
#### [new 044] Noise & pattern: identity-anchored Tikhonov regularization for robust structural anomaly detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对工业视觉检测中异常样本稀缺问题，提出一种身份锚定Tikhonov正则化自编码器，通过结构化扰动模拟缺陷并引入高斯噪声稳定重构，实现高精度结构异常检测与分割，在MVTec AD上达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.07233v1](http://arxiv.org/pdf/2511.07233v1)**

> **作者:** Alexander Bauer; Klaus-Robert Müller
>
> **摘要:** Anomaly detection plays a pivotal role in automated industrial inspection, aiming to identify subtle or rare defects in otherwise uniform visual patterns. As collecting representative examples of all possible anomalies is infeasible, we tackle structural anomaly detection using a self-supervised autoencoder that learns to repair corrupted inputs. To this end, we introduce a corruption model that injects artificial disruptions into training images to mimic structural defects. While reminiscent of denoising autoencoders, our approach differs in two key aspects. First, instead of unstructured i.i.d.\ noise, we apply structured, spatially coherent perturbations that make the task a hybrid of segmentation and inpainting. Second, and counterintuitively, we add and preserve Gaussian noise on top of the occlusions, which acts as a Tikhonov regularizer anchoring the Jacobian of the reconstruction function toward identity. This identity-anchored regularization stabilizes reconstruction and further improves both detection and segmentation accuracy. On the MVTec AD benchmark, our method achieves state-of-the-art results (I/P-AUROC: 99.9/99.4), supporting our theoretical framework and demonstrating its practical relevance for automatic inspection.
>
---
#### [new 045] CoMA: Complementary Masking and Hierarchical Dynamic Multi-Window Self-Attention in a Unified Pre-training Framework
- **分类: cs.CV; cs.AI; I.2.0**

- **简介: 论文提出CoMA框架，通过互补掩码与动态多窗口自注意力（DM-MSA）提升视觉Transformer预训练效率，解决MAE收敛慢、参数冗余问题，在ImageNet上仅用12%训练周期即达到同等性能。**

- **链接: [http://arxiv.org/pdf/2511.05929v1](http://arxiv.org/pdf/2511.05929v1)**

> **作者:** Jiaxuan Li; Qing Xu; Xiangjian He; Ziyu Liu; Chang Xing; Zhen Chen; Daokun Zhang; Rong Qu; Chang Wen Chen
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Masked Autoencoders (MAE) achieve self-supervised learning of image representations by randomly removing a portion of visual tokens and reconstructing the original image as a pretext task, thereby significantly enhancing pretraining efficiency and yielding excellent adaptability across downstream tasks. However, MAE and other MAE-style paradigms that adopt random masking generally require more pre-training epochs to maintain adaptability. Meanwhile, ViT in MAE suffers from inefficient parameter use due to fixed spatial resolution across layers. To overcome these limitations, we propose the Complementary Masked Autoencoders (CoMA), which employ a complementary masking strategy to ensure uniform sampling across all pixels, thereby improving effective learning of all features and enhancing the model's adaptability. Furthermore, we introduce DyViT, a hierarchical vision transformer that employs a Dynamic Multi-Window Self-Attention (DM-MSA), significantly reducing the parameters and FLOPs while improving fine-grained feature learning. Pre-trained on ImageNet-1K with CoMA, DyViT matches the downstream performance of MAE using only 12% of the pre-training epochs, demonstrating more effective learning. It also attains a 10% reduction in pre-training time per epoch, further underscoring its superior pre-training efficiency.
>
---
#### [new 046] Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving
- **分类: cs.CV**

- **简介: 该论文提出LFlow，一种无需训练的线性逆问题求解框架，利用潜在流先验在潜空间中进行ODE采样，并引入理论保障的后验协方差，提升重建质量，克服像素空间计算开销大与引导失效问题。**

- **链接: [http://arxiv.org/pdf/2511.06138v1](http://arxiv.org/pdf/2511.06138v1)**

> **作者:** Hossein Askari; Yadan Luo; Hongfu Sun; Fred Roosta
>
> **备注:** 37 pages, 16 figures,
>
> **摘要:** Recent advances in inverse problem solving have increasingly adopted flow priors over diffusion models due to their ability to construct straight probability paths from noise to data, thereby enhancing efficiency in both training and inference. However, current flow-based inverse solvers face two primary limitations: (i) they operate directly in pixel space, which demands heavy computational resources for training and restricts scalability to high-resolution images, and (ii) they employ guidance strategies with prior-agnostic posterior covariances, which can weaken alignment with the generative trajectory and degrade posterior coverage. In this paper, we propose LFlow (Latent Refinement via Flows), a training-free framework for solving linear inverse problems via pretrained latent flow priors. LFlow leverages the efficiency of flow matching to perform ODE sampling in latent space along an optimal path. This latent formulation further allows us to introduce a theoretically grounded posterior covariance, derived from the optimal vector field, enabling effective flow guidance. Experimental results demonstrate that our proposed method outperforms state-of-the-art latent diffusion solvers in reconstruction quality across most tasks. The code will be publicly available at https://github.com/hosseinaskari-cs/LFlow .
>
---
#### [new 047] Position-Prior-Guided Network for System Matrix Super-Resolution in Magnetic Particle Imaging
- **分类: cs.CV; 68T10; I.4.5**

- **简介: 该论文针对磁粒子成像（MPI）系统矩阵校准耗时问题，提出一种融合位置先验的超分辨率网络，利用物理对称性先验提升系统矩阵重建效率与精度，实现快速、准确的SM超分辨率重建。**

- **链接: [http://arxiv.org/pdf/2511.05795v1](http://arxiv.org/pdf/2511.05795v1)**

> **作者:** Xuqing Geng; Lei Su; Zhongwei Bian; Zewen Sun; Jiaxuan Wen; Jie Tian; Yang Du
>
> **备注:** accepted as oral presentation at EMBC 2025
>
> **摘要:** Magnetic Particle Imaging (MPI) is a novel medical imaging modality. One of the established methods for MPI reconstruction is based on the System Matrix (SM). However, the calibration of the SM is often time-consuming and requires repeated measurements whenever the system parameters change. Current methodologies utilize deep learning-based super-resolution (SR) techniques to expedite SM calibration; nevertheless, these strategies do not fully exploit physical prior knowledge associated with the SM, such as symmetric positional priors. Consequently, we integrated positional priors into existing frameworks for SM calibration. Underpinned by theoretical justification, we empirically validated the efficacy of incorporating positional priors through experiments involving both 2D and 3D SM SR methods.
>
---
#### [new 048] A Two-Stage System for Layout-Controlled Image Generation using Large Language Models and Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出一种两阶段系统，结合LLM与扩散模型，解决文本生成图像中对象数量与空间布局控制不足的问题。LLM生成结构化布局，扩散模型据此合成图像，显著提升布局精度与真实感。**

- **链接: [http://arxiv.org/pdf/2511.06888v1](http://arxiv.org/pdf/2511.06888v1)**

> **作者:** Jan-Hendrik Koch; Jonas Krumme; Konrad Gadzicki
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Text-to-image diffusion models exhibit remarkable generative capabilities, but lack precise control over object counts and spatial arrangements. This work introduces a two-stage system to address these compositional limitations. The first stage employs a Large Language Model (LLM) to generate a structured layout from a list of objects. The second stage uses a layout-conditioned diffusion model to synthesize a photorealistic image adhering to this layout. We find that task decomposition is critical for LLM-based spatial planning; by simplifying the initial generation to core objects and completing the layout with rule-based insertion, we improve object recall from 57.2% to 99.9% for complex scenes. For image synthesis, we compare two leading conditioning methods: ControlNet and GLIGEN. After domain-specific finetuning on table-setting datasets, we identify a key trade-off: ControlNet preserves text-based stylistic control but suffers from object hallucination, while GLIGEN provides superior layout fidelity at the cost of reduced prompt-based controllability. Our end-to-end system successfully generates images with specified object counts and plausible spatial arrangements, demonstrating the viability of a decoupled approach for compositionally controlled synthesis.
>
---
#### [new 049] A Mixture-of-Experts Framework with Log-Logistic Components for Survival Analysis on Histopathology Images
- **分类: cs.CV**

- **简介: 该论文针对病理图像的生存分析任务，提出一种混合专家框架，通过量化门控选区、图聚类、层次注意力和对数逻辑分布建模，精准预测癌症特异性生存期，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06266v1](http://arxiv.org/pdf/2511.06266v1)**

> **作者:** Ardhendu Sekhar; Vasu Soni; Keshav Aske; Shivam Madnoorkar; Pranav Jeevan; Amit Sethi
>
> **摘要:** We propose a modular framework for predicting cancer specific survival from whole slide pathology images (WSIs). The method integrates four components: (i) Quantile Gated Patch Selection via quantile based thresholding to isolate prognostically informative tissue regions; (ii) Graph Guided Clustering using a k nearest neighbor graph to capture phenotype level heterogeneity through spatial and morphological coherence; (iii) Hierarchical Context Attention to learn intra and inter cluster interactions; and (iv) an Expert Driven Mixture of Log logistics framework to estimate complex survival distributions using Log logistics distributions. The model attains a concordance index of 0.644 on TCGA LUAD, 0.751 on TCGA KIRC, and 0.752 on TCGA BRCA respectively, outperforming existing state of the art approaches.
>
---
#### [new 050] In-Context Adaptation of VLMs for Few-Shot Cell Detection in Optical Microscopy
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉语言模型（VLMs）在光学显微图像少样本细胞检测中的上下文适应能力，提出Micro-OD基准并构建混合检测管道，解决生物医学图像标注稀缺下的开放词汇检测问题。**

- **链接: [http://arxiv.org/pdf/2511.05565v1](http://arxiv.org/pdf/2511.05565v1)**

> **作者:** Shreyan Ganguly; Angona Biswas; Jaydeep Rade; Md Hasibul Hasan Hasib; Nabila Masud; Nitish Singla; Abhipsa Dash; Ushashi Bhattacharjee; Aditya Balu; Anwesha Sarkar; Adarsh Krishnamurthy; Soumik Sarkar
>
> **摘要:** Foundation vision-language models (VLMs) excel on natural images, but their utility for biomedical microscopy remains underexplored. In this paper, we investigate how in-context learning enables state-of-the-art VLMs to perform few-shot object detection when large annotated datasets are unavailable, as is often the case with microscopic images. We introduce the Micro-OD benchmark, a curated collection of 252 images specifically curated for in-context learning, with bounding-box annotations spanning 11 cell types across four sources, including two in-lab expert-annotated sets. We systematically evaluate eight VLMs under few-shot conditions and compare variants with and without implicit test-time reasoning tokens. We further implement a hybrid Few-Shot Object Detection (FSOD) pipeline that combines a detection head with a VLM-based few-shot classifier, which enhances the few-shot performance of recent VLMs on our benchmark. Across datasets, we observe that zero-shot performance is weak due to the domain gap; however, few-shot support consistently improves detection, with marginal gains achieved after six shots. We observe that models with reasoning tokens are more effective for end-to-end localization, whereas simpler variants are more suitable for classifying pre-localized crops. Our results highlight in-context adaptation as a practical path for microscopy, and our benchmark provides a reproducible testbed for advancing open-vocabulary detection in biomedical imaging.
>
---
#### [new 051] MALeR: Improving Compositional Fidelity in Layout-Guided Generation
- **分类: cs.CV**

- **简介: 该论文提出MALeR，面向布局引导的文本到图像生成任务，解决多主体场景中主体越界、属性泄漏和生成失真问题，通过掩码属性绑定机制提升 compositional fidelity 与生成准确性。**

- **链接: [http://arxiv.org/pdf/2511.06002v1](http://arxiv.org/pdf/2511.06002v1)**

> **作者:** Shivank Saxena; Dhruv Srivastava; Makarand Tapaswi
>
> **备注:** ACM TOG Dec 2025, Siggraph Asia, Project page: https://katha-ai.github.io/projects/maler/
>
> **摘要:** Recent advances in text-to-image models have enabled a new era of creative and controllable image generation. However, generating compositional scenes with multiple subjects and attributes remains a significant challenge. To enhance user control over subject placement, several layout-guided methods have been proposed. However, these methods face numerous challenges, particularly in compositional scenes. Unintended subjects often appear outside the layouts, generated images can be out-of-distribution and contain unnatural artifacts, or attributes bleed across subjects, leading to incorrect visual outputs. In this work, we propose MALeR, a method that addresses each of these challenges. Given a text prompt and corresponding layouts, our method prevents subjects from appearing outside the given layouts while being in-distribution. Additionally, we propose a masked, attribute-aware binding mechanism that prevents attribute leakage, enabling accurate rendering of subjects with multiple attributes, even in complex compositional scenes. Qualitative and quantitative evaluation demonstrates that our method achieves superior performance in compositional accuracy, generation consistency, and attribute binding compared to previous work. MALeR is particularly adept at generating images of scenes with multiple subjects and multiple attributes per subject.
>
---
#### [new 052] A Dual-Mode ViT-Conditioned Diffusion Framework with an Adaptive Conditioning Bridge for Breast Cancer Segmentation
- **分类: cs.CV**

- **简介: 该论文针对乳腺超声图像分割中低对比度与边界模糊问题，提出一种双模式ViT条件扩散模型，创新引入自适应条件桥、拓扑一致性损失和双头架构，显著提升分割精度与解剖合理性。**

- **链接: [http://arxiv.org/pdf/2511.05989v1](http://arxiv.org/pdf/2511.05989v1)**

> **作者:** Prateek Singh; Moumita Dholey; P. K. Vinod
>
> **备注:** 5 pages, 2 figures, 3 tables, submitted to ISBI 2026
>
> **摘要:** In breast ultrasound images, precise lesion segmentation is essential for early diagnosis; however, low contrast, speckle noise, and unclear boundaries make this difficult. Even though deep learning models have demonstrated potential, standard convolutional architectures frequently fall short in capturing enough global context, resulting in segmentations that are anatomically inconsistent. To overcome these drawbacks, we suggest a flexible, conditional Denoising Diffusion Model that combines an enhanced UNet-based generative decoder with a Vision Transformer (ViT) encoder for global feature extraction. We introduce three primary innovations: 1) an Adaptive Conditioning Bridge (ACB) for efficient, multi-scale fusion of semantic features; 2) a novel Topological Denoising Consistency (TDC) loss component that regularizes training by penalizing structural inconsistencies during denoising; and 3) a dual-head architecture that leverages the denoising objective as a powerful regularizer, enabling a lightweight auxiliary head to perform rapid and accurate inference on smaller datasets and a noise prediction head. Our framework establishes a new state-of-the-art on public breast ultrasound datasets, achieving Dice scores of 0.96 on BUSI, 0.90 on BrEaST and 0.97 on BUS-UCLM. Comprehensive ablation studies empirically validate that the model components are critical for achieving these results and for producing segmentations that are not only accurate but also anatomically plausible.
>
---
#### [new 053] SportR: A Benchmark for Multimodal Large Language Model Reasoning in Sports
- **分类: cs.CV**

- **简介: 论文提出SportR，首个多运动多模态大语言模型推理基准，解决现有数据集缺乏细粒度视觉 grounding 与复杂推理链的问题，构建5K+图像/视频与7K+思维链标注，推动体育智能的多模态推理研究。**

- **链接: [http://arxiv.org/pdf/2511.06499v1](http://arxiv.org/pdf/2511.06499v1)**

> **作者:** Haotian Xia; Haonan Ge; Junbo Zou; Hyun Woo Choi; Xuebin Zhang; Danny Suradja; Botao Rui; Ethan Tran; Wendy Jin; Zhen Ye; Xiyang Lin; Christopher Lai; Shengjie Zhang; Junwen Miao; Shichao Chen; Rhys Tracy; Vicente Ordonez; Weining Shen; Hanjie Chen
>
> **摘要:** Deeply understanding sports requires an intricate blend of fine-grained visual perception and rule-based reasoning - a challenge that pushes the limits of current multimodal models. To succeed, models must master three critical capabilities: perceiving nuanced visual details, applying abstract sport rule knowledge, and grounding that knowledge in specific visual evidence. Current sports benchmarks either cover single sports or lack the detailed reasoning chains and precise visual grounding needed to robustly evaluate these core capabilities in a multi-sport context. To address this gap, we introduce SportR, the first multi-sports large-scale benchmark designed to train and evaluate MLLMs on the fundamental reasoning required for sports intelligence. Our benchmark provides a dataset of 5,017 images and 2,101 videos. To enable granular evaluation, we structure our benchmark around a progressive hierarchy of question-answer (QA) pairs designed to probe reasoning at increasing depths - from simple infraction identification to complex penalty prediction. For the most advanced tasks requiring multi-step reasoning, such as determining penalties or explaining tactics, we provide 7,118 high-quality, human-authored Chain of Thought (CoT) annotations. In addition, our benchmark incorporates both image and video modalities and provides manual bounding box annotations to test visual grounding in the image part directly. Extensive experiments demonstrate the profound difficulty of our benchmark. State-of-the-art baseline models perform poorly on our most challenging tasks. While training on our data via Supervised Fine-Tuning and Reinforcement Learning improves these scores, they remain relatively low, highlighting a significant gap in current model capabilities. SportR presents a new challenge for the community, providing a critical resource to drive future research in multimodal sports reasoning.
>
---
#### [new 054] Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning
- **分类: cs.CV**

- **简介: 该论文提出一种开放世界3D场景图生成框架，结合视觉语言模型与检索增强推理，解决封闭词汇限制下的通用3D场景理解问题，实现无固定标签的对象检测、关系推理与多模态查询。**

- **链接: [http://arxiv.org/pdf/2511.05894v1](http://arxiv.org/pdf/2511.05894v1)**

> **作者:** Fei Yu; Quan Deng; Shengeng Tang; Yuehua Li; Lechao Cheng
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Understanding 3D scenes in open-world settings poses fundamental challenges for vision and robotics, particularly due to the limitations of closed-vocabulary supervision and static annotations. To address this, we propose a unified framework for Open-World 3D Scene Graph Generation with Retrieval-Augmented Reasoning, which enables generalizable and interactive 3D scene understanding. Our method integrates Vision-Language Models (VLMs) with retrieval-based reasoning to support multimodal exploration and language-guided interaction. The framework comprises two key components: (1) a dynamic scene graph generation module that detects objects and infers semantic relationships without fixed label sets, and (2) a retrieval-augmented reasoning pipeline that encodes scene graphs into a vector database to support text/image-conditioned queries. We evaluate our method on 3DSSG and Replica benchmarks across four tasks-scene question answering, visual grounding, instance retrieval, and task planning-demonstrating robust generalization and superior performance in diverse environments. Our results highlight the effectiveness of combining open-vocabulary perception with retrieval-based reasoning for scalable 3D scene understanding.
>
---
#### [new 055] Relative Energy Learning for LiDAR Out-of-Distribution Detection
- **分类: cs.CV**

- **简介: 该论文针对LiDAR点云的分布外检测任务，提出相对能量学习（REL）框架，利用正负对数似然差提升异常检测鲁棒性，并设计Point Raise合成异常样本，有效解决无OOD训练数据下的高误报问题，在多个基准上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06720v1](http://arxiv.org/pdf/2511.06720v1)**

> **作者:** Zizhao Li; Zhengkang Xiang; Jiayang Ao; Joseph West; Kourosh Khoshelham
>
> **摘要:** Out-of-distribution (OOD) detection is a critical requirement for reliable autonomous driving, where safety depends on recognizing road obstacles and unexpected objects beyond the training distribution. Despite extensive research on OOD detection in 2D images, direct transfer to 3D LiDAR point clouds has been proven ineffective. Current LiDAR OOD methods struggle to distinguish rare anomalies from common classes, leading to high false-positive rates and overconfident errors in safety-critical settings. We propose Relative Energy Learning (REL), a simple yet effective framework for OOD detection in LiDAR point clouds. REL leverages the energy gap between positive (in-distribution) and negative logits as a relative scoring function, mitigating calibration issues in raw energy values and improving robustness across various scenes. To address the absence of OOD samples during training, we propose a lightweight data synthesis strategy called Point Raise, which perturbs existing point clouds to generate auxiliary anomalies without altering the inlier semantics. Evaluated on SemanticKITTI and the Spotting the Unexpected (STU) benchmark, REL consistently outperforms existing methods by a large margin. Our results highlight that modeling relative energy, combined with simple synthetic outliers, provides a principled and scalable solution for reliable OOD detection in open-world autonomous driving.
>
---
#### [new 056] Garbage Vulnerable Point Monitoring using IoT and Computer Vision
- **分类: cs.CV; cs.LG**

- **简介: 该论文利用IoT与计算机视觉，针对城市垃圾脆弱点（GVP）的非法倾倒问题，构建智能监测系统，通过YOLO系列模型实现高精度垃圾检测，其中YOLO11m达92.39%准确率，有效捕捉倾倒时空规律。**

- **链接: [http://arxiv.org/pdf/2511.07325v1](http://arxiv.org/pdf/2511.07325v1)**

> **作者:** R. Kumar; A. Lall; S. Chaudhari; M. Kale; A. Vattem
>
> **摘要:** This paper proposes a smart way to manage municipal solid waste by using the Internet of Things (IoT) and computer vision (CV) to monitor illegal waste dumping at garbage vulnerable points (GVPs) in urban areas. The system can quickly detect and monitor dumped waste using a street-level camera and object detection algorithm. Data was collected from the Sangareddy district in Telangana, India. A series of comprehensive experiments was carried out using the proposed dataset to assess the accuracy and overall performance of various object detection models. Specifically, we performed an in-depth evaluation of YOLOv8, YOLOv10, YOLO11m, and RT-DETR on our dataset. Among these models, YOLO11m achieved the highest accuracy of 92.39\% in waste detection, demonstrating its effectiveness in detecting waste. Additionally, it attains an mAP@50 of 0.91, highlighting its high precision. These findings confirm that the object detection model is well-suited for monitoring and tracking waste dumping events at GVP locations. Furthermore, the system effectively captures waste disposal patterns, including hourly, daily, and weekly dumping trends, ensuring comprehensive daily and nightly monitoring.
>
---
#### [new 057] LRANet++: Low-Rank Approximation Network for Accurate and Efficient Text Spotting
- **分类: cs.CV**

- **简介: LRANet++提出一种基于低秩近似的文本检测方法，结合三重分配头，实现任意形状文本的高效精准端到端识别，解决传统方法效率低、鲁棒性差的问题。**

- **链接: [http://arxiv.org/pdf/2511.05818v1](http://arxiv.org/pdf/2511.05818v1)**

> **作者:** Yuchen Su; Zhineng Chen; Yongkun Du; Zuxuan Wu; Hongtao Xie; Yu-Gang Jiang
>
> **摘要:** End-to-end text spotting aims to jointly optimize text detection and recognition within a unified framework. Despite significant progress, designing an accurate and efficient end-to-end text spotter for arbitrary-shaped text remains largely unsolved. We identify the primary bottleneck as the lack of a reliable and efficient text detection method. To address this, we propose a novel parameterized text shape method based on low-rank approximation for precise detection and a triple assignment detection head to enable fast inference. Specifically, unlike other shape representation methods that employ data-irrelevant parameterization, our data-driven approach derives a low-rank subspace directly from labeled text boundaries. To ensure this process is robust against the inherent annotation noise in this data, we utilize a specialized recovery method based on an $\ell_1$-norm formulation, which accurately reconstructs the text shape with only a few key orthogonal vectors. By exploiting the inherent shape correlation among different text contours, our method achieves consistency and compactness in shape representation. Next, the triple assignment scheme introduces a novel architecture where a deep sparse branch (for stabilized training) is used to guide the learning of an ultra-lightweight sparse branch (for accelerated inference), while a dense branch provides rich parallel supervision. Building upon these advancements, we integrate the enhanced detection module with a lightweight recognition branch to form an end-to-end text spotting framework, termed LRANet++, capable of accurately and efficiently spotting arbitrary-shaped text. Extensive experiments on several challenging benchmarks demonstrate the superiority of LRANet++ compared to state-of-the-art methods. Code will be available at: https://github.com/ychensu/LRANet-PP.git
>
---
#### [new 058] Exploring Category-level Articulated Object Pose Tracking on SE(3) Manifolds
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对关节物体的位姿跟踪难题，提出PPF-Tracker框架，在SE(3)流形上利用点对特征与关节语义约束，实现多帧鲁棒位姿估计，提升机器人与AR应用中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.05996v1](http://arxiv.org/pdf/2511.05996v1)**

> **作者:** Xianhui Meng; Yukang Huo; Li Zhang; Liu Liu; Haonan Jiang; Yan Zhong; Pingrui Zhang; Cewu Lu; Jun Liu
>
> **摘要:** Articulated objects are prevalent in daily life and robotic manipulation tasks. However, compared to rigid objects, pose tracking for articulated objects remains an underexplored problem due to their inherent kinematic constraints. To address these challenges, this work proposes a novel point-pair-based pose tracking framework, termed \textbf{PPF-Tracker}. The proposed framework first performs quasi-canonicalization of point clouds in the SE(3) Lie group space, and then models articulated objects using Point Pair Features (PPF) to predict pose voting parameters by leveraging the invariance properties of SE(3). Finally, semantic information of joint axes is incorporated to impose unified kinematic constraints across all parts of the articulated object. PPF-Tracker is systematically evaluated on both synthetic datasets and real-world scenarios, demonstrating strong generalization across diverse and challenging environments. Experimental results highlight the effectiveness and robustness of PPF-Tracker in multi-frame pose tracking of articulated objects. We believe this work can foster advances in robotics, embodied intelligence, and augmented reality. Codes are available at https://github.com/mengxh20/PPFTracker.
>
---
#### [new 059] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection
- **分类: cs.CV**

- **简介: 该论文针对多模态情感分析中模态不平衡与静态主模态选择问题，提出MODS框架，通过图结构序列压缩器降噪，动态主模态选择器自适应选主模态，并设计主模态中心交叉注意力机制提升融合效果。**

- **链接: [http://arxiv.org/pdf/2511.06328v1](http://arxiv.org/pdf/2511.06328v1)**

> **作者:** Dingkang Yang; Mingcheng Li; Xuecheng Wu; Zhaoyu Chen; Kaixun Jiang; Keliang Liu; Peng Zhai; Lihua Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Multimodal Sentiment Analysis (MSA) aims to predict sentiment from language, acoustic, and visual data in videos. However, imbalanced unimodal performance often leads to suboptimal fused representations. Existing approaches typically adopt fixed primary modality strategies to maximize dominant modality advantages, yet fail to adapt to dynamic variations in modality importance across different samples. Moreover, non-language modalities suffer from sequential redundancy and noise, degrading model performance when they serve as primary inputs. To address these issues, this paper proposes a modality optimization and dynamic primary modality selection framework (MODS). First, a Graph-based Dynamic Sequence Compressor (GDC) is constructed, which employs capsule networks and graph convolution to reduce sequential redundancy in acoustic/visual modalities. Then, we develop a sample-adaptive Primary Modality Selector (MSelector) for dynamic dominance determination. Finally, a Primary-modality-Centric Cross-Attention (PCCA) module is designed to enhance dominant modalities while facilitating cross-modal interaction. Extensive experiments on four benchmark datasets demonstrate that MODS outperforms state-of-the-art methods, achieving superior performance by effectively balancing modality contributions and eliminating redundant noise.
>
---
#### [new 060] LMM-IQA: Image Quality Assessment for Low-Dose CT Imaging
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LMM-IQA，用于低剂量CT图像质量评估，解决噪声、模糊与对比度损失导致的诊断质量下降问题。基于LLM生成数值评分与可解释文本描述，系统对比多种推理策略，提升评估一致性与临床可用性。**

- **链接: [http://arxiv.org/pdf/2511.07298v1](http://arxiv.org/pdf/2511.07298v1)**

> **作者:** Kagan Celik; Mehmet Ozan Unal; Metin Ertas; Isa Yildirim
>
> **摘要:** Low-dose computed tomography (CT) represents a significant improvement in patient safety through lower radiation doses, but increased noise, blur, and contrast loss can diminish diagnostic quality. Therefore, consistency and robustness in image quality assessment become essential for clinical applications. In this study, we propose an LLM-based quality assessment system that generates both numerical scores and textual descriptions of degradations such as noise, blur, and contrast loss. Furthermore, various inference strategies - from the zero-shot approach to metadata integration and error feedback - are systematically examined, demonstrating the progressive contribution of each method to overall performance. The resultant assessments yield not only highly correlated scores but also interpretable output, thereby adding value to clinical workflows. The source codes of our study are available at https://github.com/itu-biai/lmms_ldct_iqa.
>
---
#### [new 061] CSGaze: Context-aware Social Gaze Prediction
- **分类: cs.CV; cs.LG**

- **简介: CSGaze提出一种上下文感知的多模态社交凝视预测方法，融合面部与场景信息，结合主讲人注意力机制，提升多人群体对话中的凝视预测精度，并验证其泛化性与可解释性。**

- **链接: [http://arxiv.org/pdf/2511.05955v1](http://arxiv.org/pdf/2511.05955v1)**

> **作者:** Surbhi Madan; Shreya Ghosh; Ramanathan Subramanian; Abhinav Dhall; Tom Gedeon
>
> **摘要:** A person's gaze offers valuable insights into their focus of attention, level of social engagement, and confidence. In this work, we investigate how contextual cues combined with visual scene and facial information can be effectively utilized to predict and interpret social gaze patterns during conversational interactions. We introduce CSGaze, a context aware multimodal approach that leverages facial, scene information as complementary inputs to enhance social gaze pattern prediction from multi-person images. The model also incorporates a fine-grained attention mechanism centered on the principal speaker, which helps in better modeling social gaze dynamics. Experimental results show that CSGaze performs competitively with state-of-the-art methods on GP-Static, UCO-LAEO and AVA-LAEO. Our findings highlight the role of contextual cues in improving social gaze prediction. Additionally, we provide initial explainability through generated attention scores, offering insights into the model's decision-making process. We also demonstrate our model's generalizability by testing our model on open set datasets that demonstrating its robustness across diverse scenarios.
>
---
#### [new 062] DIAL-GS: Dynamic Instance Aware Reconstruction for Label-free Street Scenes with 4D Gaussian Splatting
- **分类: cs.CV**

- **简介: DIAL-GS提出一种无标签街景的动态实例感知重建方法，基于4D高斯溅射，通过外观-位置不一致性识别动态对象，并实现实例级自适应重建与交互增强，提升自监督场景建模的精度与编辑能力。**

- **链接: [http://arxiv.org/pdf/2511.06632v1](http://arxiv.org/pdf/2511.06632v1)**

> **作者:** Chenpeng Su; Wenhua Wu; Chensheng Peng; Tianchen Deng; Zhe Liu; Hesheng Wang
>
> **摘要:** Urban scene reconstruction is critical for autonomous driving, enabling structured 3D representations for data synthesis and closed-loop testing. Supervised approaches rely on costly human annotations and lack scalability, while current self-supervised methods often confuse static and dynamic elements and fail to distinguish individual dynamic objects, limiting fine-grained editing. We propose DIAL-GS, a novel dynamic instance-aware reconstruction method for label-free street scenes with 4D Gaussian Splatting. We first accurately identify dynamic instances by exploiting appearance-position inconsistency between warped rendering and actual observation. Guided by instance-level dynamic perception, we employ instance-aware 4D Gaussians as the unified volumetric representation, realizing dynamic-adaptive and instance-aware reconstruction. Furthermore, we introduce a reciprocal mechanism through which identity and dynamics reinforce each other, enhancing both integrity and consistency. Experiments on urban driving scenarios show that DIAL-GS surpasses existing self-supervised baselines in reconstruction quality and instance-level editing, offering a concise yet powerful solution for urban scene modeling.
>
---
#### [new 063] DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出DiA-gnostic VLVAE，用于缺失模态下的放射科报告生成，通过混合专家变分自编码器解耦共享与模态特异性特征，约束对齐以减少幻觉，实现高效鲁棒的多模态融合与报告生成。**

- **链接: [http://arxiv.org/pdf/2511.05968v1](http://arxiv.org/pdf/2511.05968v1)**

> **作者:** Nagur Shareef Shaik; Teja Krishna Cherukuri; Adnan Masood; Dong Hye Ye
>
> **备注:** Accepted for Oral Presentation at the 40th AAAI Conference on Artificial Intelligence (AAAI-26), Main Technical Track
>
> **摘要:** The integration of medical images with clinical context is essential for generating accurate and clinically interpretable radiology reports. However, current automated methods often rely on resource-heavy Large Language Models (LLMs) or static knowledge graphs and struggle with two fundamental challenges in real-world clinical data: (1) missing modalities, such as incomplete clinical context , and (2) feature entanglement, where mixed modality-specific and shared information leads to suboptimal fusion and clinically unfaithful hallucinated findings. To address these challenges, we propose the DiA-gnostic VLVAE, which achieves robust radiology reporting through Disentangled Alignment. Our framework is designed to be resilient to missing modalities by disentangling shared and modality-specific features using a Mixture-of-Experts (MoE) based Vision-Language Variational Autoencoder (VLVAE). A constrained optimization objective enforces orthogonality and alignment between these latent representations to prevent suboptimal fusion. A compact LLaMA-X decoder then uses these disentangled representations to generate reports efficiently. On the IU X-Ray and MIMIC-CXR datasets, DiA has achieved competetive BLEU@4 scores of 0.266 and 0.134, respectively. Experimental results show that the proposed method significantly outperforms state-of-the-art models.
>
---
#### [new 064] GazeVLM: A Vision-Language Model for Multi-Task Gaze Understanding
- **分类: cs.CV; cs.AI**

- **简介: GazeVLM是首个用于多任务眼动理解的视觉语言模型，融合RGB与深度图及文本提示，统一解决人物检测、注视目标定位与注视对象识别问题，在GazeFollow等数据集上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2511.06348v1](http://arxiv.org/pdf/2511.06348v1)**

> **作者:** Athul M. Mathew; Haithem Hermassi; Thariq Khalid; Arshad Ali Khan; Riad Souissi
>
> **摘要:** Gaze understanding unifies the detection of people, their gaze targets, and objects of interest into a single framework, offering critical insight into visual attention and intent estimation. Although prior research has modelled gaze cues in visual scenes, a unified system is still needed for gaze understanding using both visual and language prompts. This paper introduces GazeVLM, a novel Vision-Language Model (VLM) for multi-task gaze understanding in images, addressing person detection, gaze target detection, and gaze object identification. While other transformer-based methods exist for gaze analysis, GazeVLM represents, to our knowledge, the first application of a VLM to these combined tasks, allowing for selective execution of each task. Through the integration of visual (RGB and depth) and textual modalities, our ablation study on visual input combinations revealed that a fusion of RGB images with HHA-encoded depth maps, guided by text prompts, yields superior performance. We also introduce an object-level gaze detection metric for gaze object identification ($AP_{ob}$). Through experiments, GazeVLM demonstrates significant improvements, notably achieving state-of-the-art evaluation scores on GazeFollow and VideoAttentionTarget datasets.
>
---
#### [new 065] One-Shot Knowledge Transfer for Scalable Person Re-Identification
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向可扩展行人重识别（ReID），解决多尺寸模型重复压缩的高计算开销问题，提出OSKT方法，通过一次性知识注入权重链，按需扩展为不同规模模型，无需重复训练。**

- **链接: [http://arxiv.org/pdf/2511.06016v1](http://arxiv.org/pdf/2511.06016v1)**

> **作者:** Longhua Li; Lei Qi; Xin Geng
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Edge computing in person re-identification (ReID) is crucial for reducing the load on central cloud servers and ensuring user privacy. Conventional compression methods for obtaining compact models require computations for each individual student model. When multiple models of varying sizes are needed to accommodate different resource conditions, this leads to repetitive and cumbersome computations. To address this challenge, we propose a novel knowledge inheritance approach named OSKT (One-Shot Knowledge Transfer), which consolidates the knowledge of the teacher model into an intermediate carrier called a weight chain. When a downstream scenario demands a model that meets specific resource constraints, this weight chain can be expanded to the target model size without additional computation. OSKT significantly outperforms state-of-the-art compression methods, with the added advantage of one-time knowledge transfer that eliminates the need for frequent computations for each target model.
>
---
#### [new 066] Hybrid second-order gradient histogram based global low-rank sparse regression for robust face recognition
- **分类: cs.CV; math.OC**

- **简介: 该论文面向鲁棒人脸识别任务，针对遮挡与光照变化问题，提出H2H-GLRSR模型：设计混合二阶梯度直方图（H2H）特征描述子，并结合全局低秩约束的稀疏核范数回归，提升噪声建模与判别能力。**

- **链接: [http://arxiv.org/pdf/2511.05893v1](http://arxiv.org/pdf/2511.05893v1)**

> **作者:** Hongxia Li; Ying Ji; Yongxin Dong; Yuehua Feng
>
> **摘要:** Low-rank sparse regression models have been widely applied in the field of face recognition. To further address the challenges caused by complex occlusions and illumination variations, this paper proposes a Hybrid Second-Order Gradient Histogram based Global Low-Rank Sparse Regression (H2H-GLRSR) model. Specifically, a novel feature descriptor called the Hybrid Second-Order Gradient Histogram (H2H) is first designed to more effectively characterize the local structural features of facial images. Then, this descriptor is integrated with the Sparse Regularized Nuclear Norm based Matrix Regression (SR$\_$NMR). Moreover, a global low-rank constraint is imposed on the residual matrix, enabling the model to better capture the global correlations inherent in structured noise. Experimental results demonstrate that the proposed method significantly outperforms existing regression-based classification approaches under challenging scenarios involving occlusions, illumination changes, and unconstrained environments.
>
---
#### [new 067] SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文针对单目3D目标检测中属性解耦导致的几何不一致问题，提出Spatial-Projection Alignment（SPAN），通过空间点对齐与3D-2D投影对齐约束，提升检测几何一致性，并引入分层任务学习策略稳定训练，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2511.06702v1](http://arxiv.org/pdf/2511.06702v1)**

> **作者:** Yifan Wang; Yian Zhao; Fanqi Pu; Xiaochen Yang; Yang Tang; Xi Chen; Wenming Yang
>
> **摘要:** Existing monocular 3D detectors typically tame the pronounced nonlinear regression of 3D bounding box through decoupled prediction paradigm, which employs multiple branches to estimate geometric center, depth, dimensions, and rotation angle separately. Although this decoupling strategy simplifies the learning process, it inherently ignores the geometric collaborative constraints between different attributes, resulting in the lack of geometric consistency prior, thereby leading to suboptimal performance. To address this issue, we propose novel Spatial-Projection Alignment (SPAN) with two pivotal components: (i). Spatial Point Alignment enforces an explicit global spatial constraint between the predicted and ground-truth 3D bounding boxes, thereby rectifying spatial drift caused by decoupled attribute regression. (ii). 3D-2D Projection Alignment ensures that the projected 3D box is aligned tightly within its corresponding 2D detection bounding box on the image plane, mitigating projection misalignment overlooked in previous works. To ensure training stability, we further introduce a Hierarchical Task Learning strategy that progressively incorporates spatial-projection alignment as 3D attribute predictions refine, preventing early stage error propagation across attributes. Extensive experiments demonstrate that the proposed method can be easily integrated into any established monocular 3D detector and delivers significant performance improvements.
>
---
#### [new 068] ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reconstruction with Fewer Primitives
- **分类: cs.CV**

- **简介: ConeGS面向3D高斯泼溅的重建任务，解决传统密度增强依赖几何克隆导致效率低的问题，提出基于像素锥的误差引导新高斯插入方法，结合预算控制，显著提升低原元数下的重建质量与效率。**

- **链接: [http://arxiv.org/pdf/2511.06810v1](http://arxiv.org/pdf/2511.06810v1)**

> **作者:** Bartłomiej Baranowski; Stefano Esposito; Patricia Gschoßmann; Anpei Chen; Andreas Geiger
>
> **摘要:** 3D Gaussian Splatting (3DGS) achieves state-of-the-art image quality and real-time performance in novel view synthesis but often suffers from a suboptimal spatial distribution of primitives. This issue stems from cloning-based densification, which propagates Gaussians along existing geometry, limiting exploration and requiring many primitives to adequately cover the scene. We present ConeGS, an image-space-informed densification framework that is independent of existing scene geometry state. ConeGS first creates a fast Instant Neural Graphics Primitives (iNGP) reconstruction as a geometric proxy to estimate per-pixel depth. During the subsequent 3DGS optimization, it identifies high-error pixels and inserts new Gaussians along the corresponding viewing cones at the predicted depth values, initializing their size according to the cone diameter. A pre-activation opacity penalty rapidly removes redundant Gaussians, while a primitive budgeting strategy controls the total number of primitives, either by a fixed budget or by adapting to scene complexity, ensuring high reconstruction quality. Experiments show that ConeGS consistently enhances reconstruction quality and rendering performance across Gaussian budgets, with especially strong gains under tight primitive constraints where efficient placement is crucial.
>
---
#### [new 069] AvatarTex: High-Fidelity Facial Texture Reconstruction from Single-Image Stylized Avatars
- **分类: cs.CV**

- **简介: AvatarTex提出一种扩散与GAN融合的三阶段框架，从单图重建高保真风格化与写实人脸纹理，解决多风格数据缺失与几何不一致问题，并构建了20,000张UV对齐的风格化纹理数据集TexHub。**

- **链接: [http://arxiv.org/pdf/2511.06721v1](http://arxiv.org/pdf/2511.06721v1)**

> **作者:** Yuda Qiu; Zitong Xiao; Yiwei Zuo; Zisheng Ye; Weikai Chen; Xiaoguang Han
>
> **备注:** 3DV 2026 Accepted
>
> **摘要:** We present AvatarTex, a high-fidelity facial texture reconstruction framework capable of generating both stylized and photorealistic textures from a single image. Existing methods struggle with stylized avatars due to the lack of diverse multi-style datasets and challenges in maintaining geometric consistency in non-standard textures. To address these limitations, AvatarTex introduces a novel three-stage diffusion-to-GAN pipeline. Our key insight is that while diffusion models excel at generating diversified textures, they lack explicit UV constraints, whereas GANs provide a well-structured latent space that ensures style and topology consistency. By integrating these strengths, AvatarTex achieves high-quality topology-aligned texture synthesis with both artistic and geometric coherence. Specifically, our three-stage pipeline first completes missing texture regions via diffusion-based inpainting, refines style and structure consistency using GAN-based latent optimization, and enhances fine details through diffusion-based repainting. To address the need for a stylized texture dataset, we introduce TexHub, a high-resolution collection of 20,000 multi-style UV textures with precise UV-aligned layouts. By leveraging TexHub and our structured diffusion-to-GAN pipeline, AvatarTex establishes a new state-of-the-art in multi-style facial texture reconstruction. TexHub will be released upon publication to facilitate future research in this field.
>
---
#### [new 070] BuildingWorld: A Structured 3D Building Dataset for Urban Foundation Models
- **分类: cs.CV**

- **简介: 论文提出BuildingWorld，一个全球多样化的结构化3D建筑数据集，解决现有数据集建筑风格单一导致模型泛化差的问题，支持城市基础模型在重建、检测与分割任务上的训练与评估。**

- **链接: [http://arxiv.org/pdf/2511.06337v1](http://arxiv.org/pdf/2511.06337v1)**

> **作者:** Shangfeng Huang; Ruisheng Wang; Xin Wang
>
> **摘要:** As digital twins become central to the transformation of modern cities, accurate and structured 3D building models emerge as a key enabler of high-fidelity, updatable urban representations. These models underpin diverse applications including energy modeling, urban planning, autonomous navigation, and real-time reasoning. Despite recent advances in 3D urban modeling, most learning-based models are trained on building datasets with limited architectural diversity, which significantly undermines their generalizability across heterogeneous urban environments. To address this limitation, we present BuildingWorld, a comprehensive and structured 3D building dataset designed to bridge the gap in stylistic diversity. It encompasses buildings from geographically and architecturally diverse regions -- including North America, Europe, Asia, Africa, and Oceania -- offering a globally representative dataset for urban-scale foundation modeling and analysis. Specifically, BuildingWorld provides about five million LOD2 building models collected from diverse sources, accompanied by real and simulated airborne LiDAR point clouds. This enables comprehensive research on 3D building reconstruction, detection and segmentation. Cyber City, a virtual city model, is introduced to enable the generation of unlimited training data with customized and structurally diverse point cloud distributions. Furthermore, we provide standardized evaluation metrics tailored for building reconstruction, aiming to facilitate the training, evaluation, and comparison of large-scale vision models and foundation models in structured 3D urban environments.
>
---
#### [new 071] In-process 3D Deviation Mapping and Defect Monitoring (3D-DM2) in High Production-rate Robotic Additive Manufacturing
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出实时3D偏差监测系统（3D-DM2），用于高产率机器人增材制造中在线检测形状偏差。通过比对实时构建体与参考模型，实现偏差定位与追踪，支持过程即时补偿，提升成形精度与质量。**

- **链接: [http://arxiv.org/pdf/2511.05604v1](http://arxiv.org/pdf/2511.05604v1)**

> **作者:** Subash Gautam; Alejandro Vargas-Uscategui; Peter King; Hans Lohr; Alireza Bab-Hadiashar; Ivan Cole; Ehsan Asadi
>
> **摘要:** Additive manufacturing (AM) is an emerging digital manufacturing technology to produce complex and freeform objects through a layer-wise deposition. High deposition rate robotic AM (HDRRAM) processes, such as cold spray additive manufacturing (CSAM), offer significantly increased build speeds by delivering large volumes of material per unit time. However, maintaining shape accuracy remains a critical challenge, particularly due to process instabilities in current open-loop systems. Detecting these deviations as they occur is essential to prevent error propagation, ensure part quality, and minimize post-processing requirements. This study presents a real-time monitoring system to acquire and reconstruct the growing part and directly compares it with a near-net reference model to detect the shape deviation during the manufacturing process. The early identification of shape inconsistencies, followed by segmenting and tracking each deviation region, paves the way for timely intervention and compensation to achieve consistent part quality.
>
---
#### [new 072] TCSA-UDA: Text-Driven Cross-Semantic Alignment for Unsupervised Domain Adaptation in Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出TCSA-UDA，用于医学图像分割的无监督域自适应，解决CT与MRI等模态间域偏移问题。通过文本语义引导视觉特征对齐，结合协方差余弦损失与原型对齐模块，提升跨模态一致性与分割性能。**

- **链接: [http://arxiv.org/pdf/2511.05782v1](http://arxiv.org/pdf/2511.05782v1)**

> **作者:** Lalit Maurya; Honghai Liu; Reyer Zwiggelaar
>
> **摘要:** Unsupervised domain adaptation for medical image segmentation remains a significant challenge due to substantial domain shifts across imaging modalities, such as CT and MRI. While recent vision-language representation learning methods have shown promise, their potential in UDA segmentation tasks remains underexplored. To address this gap, we propose TCSA-UDA, a Text-driven Cross-Semantic Alignment framework that leverages domain-invariant textual class descriptions to guide visual representation learning. Our approach introduces a vision-language covariance cosine loss to directly align image encoder features with inter-class textual semantic relations, encouraging semantically meaningful and modality-invariant feature representations. Additionally, we incorporate a prototype alignment module that aligns class-wise pixel-level feature distributions across domains using high-level semantic prototypes. This mitigates residual category-level discrepancies and enhances cross-modal consistency. Extensive experiments on challenging cross-modality cardiac, abdominal, and brain tumor segmentation benchmarks demonstrate that our TCSA-UDA framework significantly reduces domain shift and consistently outperforms state-of-the-art UDA methods, establishing a new paradigm for integrating language-driven semantics into domain-adaptive medical image analysis.
>
---
#### [new 073] VAEVQ: Enhancing Discrete Visual Tokenization through Variational Modeling
- **分类: cs.CV**

- **简介: 论文提出VAEVQ，用于改进离散视觉分词任务。针对VQ导致的潜在空间不平滑、编码本利用率低等问题，引入变分自编码器、表示一致性策略与分布正则化，提升重建与生成性能。**

- **链接: [http://arxiv.org/pdf/2511.06863v1](http://arxiv.org/pdf/2511.06863v1)**

> **作者:** Sicheng Yang; Xing Hu; Qiang Wu; Dawei Yang
>
> **摘要:** Vector quantization (VQ) transforms continuous image features into discrete representations, providing compressed, tokenized inputs for generative models. However, VQ-based frameworks suffer from several issues, such as non-smooth latent spaces, weak alignment between representations before and after quantization, and poor coherence between the continuous and discrete domains. These issues lead to unstable codeword learning and underutilized codebooks, ultimately degrading the performance of both reconstruction and downstream generation tasks. To this end, we propose VAEVQ, which comprises three key components: (1) Variational Latent Quantization (VLQ), replacing the AE with a VAE for quantization to leverage its structured and smooth latent space, thereby facilitating more effective codeword activation; (2) Representation Coherence Strategy (RCS), adaptively modulating the alignment strength between pre- and post-quantization features to enhance consistency and prevent overfitting to noise; and (3) Distribution Consistency Regularization (DCR), aligning the entire codebook distribution with the continuous latent distribution to improve utilization. Extensive experiments on two benchmark datasets demonstrate that VAEVQ outperforms state-of-the-art methods.
>
---
#### [new 074] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory
- **分类: cs.CV**

- **简介: 该论文提出CIF方法，解决少样本多模态工业异常检测中样本不足导致的结构缺失问题，通过超图建模训练样本的高阶结构共性，结合记忆库与无训练消息传递，提升检测精度。**

- **链接: [http://arxiv.org/pdf/2511.05966v1](http://arxiv.org/pdf/2511.05966v1)**

> **作者:** Yuxuan Lin; Hanjing Yan; Xuan Tong; Yang Chang; Huanzhen Wang; Ziheng Zhou; Shuyong Gao; Yan Wang; Wenqiang Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Few-shot multimodal industrial anomaly detection is a critical yet underexplored task, offering the ability to quickly adapt to complex industrial scenarios. In few-shot settings, insufficient training samples often fail to cover the diverse patterns present in test samples. This challenge can be mitigated by extracting structural commonality from a small number of training samples. In this paper, we propose a novel few-shot unsupervised multimodal industrial anomaly detection method based on structural commonality, CIF (Commonality In Few). To extract intra-class structural information, we employ hypergraphs, which are capable of modeling higher-order correlations, to capture the structural commonality within training samples, and use a memory bank to store this intra-class structural prior. Firstly, we design a semantic-aware hypergraph construction module tailored for single-semantic industrial images, from which we extract common structures to guide the construction of the memory bank. Secondly, we use a training-free hypergraph message passing module to update the visual features of test samples, reducing the distribution gap between test features and features in the memory bank. We further propose a hyperedge-guided memory search module, which utilizes structural information to assist the memory search process and reduce the false positive rate. Experimental results on the MVTec 3D-AD dataset and the Eyecandies dataset show that our method outperforms the state-of-the-art (SOTA) methods in few-shot settings. Code is available at https://github.com/Sunny5250/CIF.
>
---
#### [new 075] Elements of Active Continuous Learning and Uncertainty Self-Awareness: a Narrow Implementation for Face and Facial Expression Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向人脸与表情识别任务，提出一种自感知机制：通过监督神经网络监测主CNN的激活模式，识别预测不确定性，触发主动学习以请求人工协助，赋予模型自我反思与决策能力。**

- **链接: [http://arxiv.org/pdf/2511.05574v1](http://arxiv.org/pdf/2511.05574v1)**

> **作者:** Stanislav Selitskiy
>
> **摘要:** Reflection on one's thought process and making corrections to it if there exists dissatisfaction in its performance is, perhaps, one of the essential traits of intelligence. However, such high-level abstract concepts mandatory for Artificial General Intelligence can be modelled even at the low level of narrow Machine Learning algorithms. Here, we present the self-awareness mechanism emulation in the form of a supervising artificial neural network (ANN) observing patterns in activations of another underlying ANN in a search for indications of the high uncertainty of the underlying ANN and, therefore, the trustworthiness of its predictions. The underlying ANN is a convolutional neural network (CNN) ensemble employed for face recognition and facial expression tasks. The self-awareness ANN has a memory region where its past performance information is stored, and its learnable parameters are adjusted during the training to optimize the performance. The trustworthiness verdict triggers the active learning mode, giving elements of agency to the machine learning algorithm that asks for human help in high uncertainty and confusion conditions.
>
---
#### [new 076] Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks
- **分类: cs.CV; cs.AI**

- **简介: 论文提出医疗诊断分割（MDS）任务，联合实现医学图像分割与诊断解释。构建M3DS数据集，设计Sim4Seg框架，利用区域感知视觉-语言相似性提升分割与诊断性能，并引入测试时扩展策略。**

- **链接: [http://arxiv.org/pdf/2511.06665v1](http://arxiv.org/pdf/2511.06665v1)**

> **作者:** Lingran Song; Yucheng Zhou; Jianbing Shen
>
> **备注:** AAAI 2026
>
> **摘要:** Despite significant progress in pixel-level medical image analysis, existing medical image segmentation models rarely explore medical segmentation and diagnosis tasks jointly. However, it is crucial for patients that models can provide explainable diagnoses along with medical segmentation results. In this paper, we introduce a medical vision-language task named Medical Diagnosis Segmentation (MDS), which aims to understand clinical queries for medical images and generate the corresponding segmentation masks as well as diagnostic results. To facilitate this task, we first present the Multimodal Multi-disease Medical Diagnosis Segmentation (M3DS) dataset, containing diverse multimodal multi-disease medical images paired with their corresponding segmentation masks and diagnosis chain-of-thought, created via an automated diagnosis chain-of-thought generation pipeline. Moreover, we propose Sim4Seg, a novel framework that improves the performance of diagnosis segmentation by taking advantage of the Region-Aware Vision-Language Similarity to Mask (RVLS2M) module. To improve overall performance, we investigate a test-time scaling strategy for MDS tasks. Experimental results demonstrate that our method outperforms the baselines in both segmentation and diagnosis.
>
---
#### [new 077] TYrPPG: Uncomplicated and Enhanced Learning Capability rPPG for Remote Heart Rate Estimation
- **分类: cs.CV**

- **简介: 论文提出TYrPPG，用于远程心率估计，解决传统Transformer模型计算效率低的问题。基于Mambaout结构，设计GVB模块融合2D/3D-CNN提升视频理解，并引入CSL损失函数增强学习能力，实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.05833v1](http://arxiv.org/pdf/2511.05833v1)**

> **作者:** Taixi Chen; Yiu-ming Cheung
>
> **备注:** The 6th International Workshop on AI for Social Good in the Connected World (AI4SG)@ IEEE WI-IAT 2025
>
> **摘要:** Remote photoplethysmography (rPPG) can remotely extract physiological signals from RGB video, which has many advantages in detecting heart rate, such as low cost and no invasion to patients. The existing rPPG model is usually based on the transformer module, which has low computation efficiency. Recently, the Mamba model has garnered increasing attention due to its efficient performance in natural language processing tasks, demonstrating potential as a substitute for transformer-based algorithms. However, the Mambaout model and its variants prove that the SSM module, which is the core component of the Mamba model, is unnecessary for the vision task. Therefore, we hope to prove the feasibility of using the Mambaout-based module to remotely learn the heart rate. Specifically, we propose a novel rPPG algorithm called uncomplicated and enhanced learning capability rPPG (TYrPPG). This paper introduces an innovative gated video understanding block (GVB) designed for efficient analysis of RGB videos. Based on the Mambaout structure, this block integrates 2D-CNN and 3D-CNN to enhance video understanding for analysis. In addition, we propose a comprehensive supervised loss function (CSL) to improve the model's learning capability, along with its weakly supervised variants. The experiments show that our TYrPPG can achieve state-of-the-art performance in commonly used datasets, indicating its prospects and superiority in remote heart rate estimation. The source code is available at https://github.com/Taixi-CHEN/TYrPPG.
>
---
#### [new 078] TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research
- **分类: cs.CV; cs.RO**

- **简介: 论文提出TwinOR框架，构建高保真动态手术室数字孪生体，解决实体手术室中AI代理训练受限问题，通过多视角感知融合静态几何与动态行为，实现传感器级真实感仿真，支持Embodied AI的高效训练与评估。**

- **链接: [http://arxiv.org/pdf/2511.07412v1](http://arxiv.org/pdf/2511.07412v1)**

> **作者:** Han Zhang; Yiqing Shen; Roger D. Soberanis-Mukul; Ankita Ghosh; Hao Ding; Lalithkumar Seenivasan; Jose L. Porras; Zhekai Mao; Chenjia Li; Wenjie Xiao; Lonny Yarmus; Angela Christine Argento; Masaru Ishii; Mathias Unberath
>
> **摘要:** Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit embodied agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create photorealistic and dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains unclear. We introduce TwinOR, a framework for constructing photorealistic, dynamic digital twins of ORs for embodied AI research. The system reconstructs static geometry from pre-scan videos and continuously models human and equipment motion through multi-view perception of OR activities. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter level accuracy while preserving dynamic interaction across surgical workflows, enabling realistic renderings and a virtual playground for embodied AI systems. In our experiments, TwinOR simulates stereo and monocular sensor streams for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 on TwinOR-synthesized data achieve performance within their reported accuracy on real indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for perception and localization challenges. By establishing a real-to-sim pipeline for constructing dynamic, photorealistic digital twins of OR environments, TwinOR enables the safe, scalable, and data-efficient development and benchmarking of embodied AI, ultimately accelerating the deployment of embodied AI from sim-to-real.
>
---
#### [new 079] AD-DAE: Unsupervised Modeling of Longitudinal Alzheimer's Disease Progression with Diffusion Auto-Encoder
- **分类: cs.CV**

- **简介: 该论文提出AD-DAE，一种无监督扩散自编码器框架，用于建模阿尔茨海默病纵向进展。通过约束潜在空间中的可控迁移，实现从基线图像生成随访图像，无需个体纵向标注，有效分离病程与个体特征。**

- **链接: [http://arxiv.org/pdf/2511.05934v1](http://arxiv.org/pdf/2511.05934v1)**

> **作者:** Ayantika Das; Arunima Sarkar; Keerthi Ram; Mohanasankar Sivaprakasam
>
> **备注:** Under Review
>
> **摘要:** Generative modeling frameworks have emerged as an effective approach to capture high-dimensional image distributions from large datasets without requiring domain-specific knowledge, a capability essential for longitudinal disease progression modeling. Recent generative modeling approaches have attempted to capture progression by mapping images into a latent representational space and then controlling and guiding the representations to generate follow-up images from a baseline image. However, existing approaches impose constraints on distribution learning, leading to latent spaces with limited controllability to generate follow-up images without explicit supervision from subject-specific longitudinal images. In order to enable controlled movements in the latent representational space and generate progression images from a baseline image in an unsupervised manner, we introduce a conditionable Diffusion Auto-encoder framework. The explicit encoding mechanism of image-diffusion auto-encoders forms a compact latent space capturing high-level semantics, providing means to disentangle information relevant for progression. Our approach leverages this latent space to condition and apply controlled shifts to baseline representations for generating follow-up. Controllability is induced by restricting these shifts to a subspace, thereby isolating progression-related factors from subject identity-preserving components. The shifts are implicitly guided by correlating with progression attributes, without requiring subject-specific longitudinal supervision. We validate the generations through image quality metrics, volumetric progression analysis, and downstream classification in Alzheimer's disease datasets from two different sources and disease categories. This demonstrates the effectiveness of our approach for Alzheimer's progression modeling and longitudinal image generation.
>
---
#### [new 080] SinSEMI: A One-Shot Image Generation Model and Data-Efficient Evaluation Framework for Semiconductor Inspection Equipment
- **分类: cs.CV**

- **简介: 论文提出SinSEMI，一种单样本图像生成模型，解决半导体检测设备训练数据稀缺问题。通过多尺度流模型与LPIPS引导，生成高真实感、多样化的光学图像，并配套轻量评估框架，显著提升AI训练数据质量。**

- **链接: [http://arxiv.org/pdf/2511.06740v1](http://arxiv.org/pdf/2511.06740v1)**

> **作者:** ChunLiang Wu; Xiaochun Li
>
> **摘要:** In the early stages of semiconductor equipment development, obtaining large quantities of raw optical images poses a significant challenge. This data scarcity hinder the advancement of AI-powered solutions in semiconductor manufacturing. To address this challenge, we introduce SinSEMI, a novel one-shot learning approach that generates diverse and highly realistic images from single optical image. SinSEMI employs a multi-scale flow-based model enhanced with LPIPS (Learned Perceptual Image Patch Similarity) energy guidance during sampling, ensuring both perceptual realism and output variety. We also introduce a comprehensive evaluation framework tailored for this application, which enables a thorough assessment using just two reference images. Through the evaluation against multiple one-shot generation techniques, we demonstrate SinSEMI's superior performance in visual quality, quantitative measures, and downstream tasks. Our experimental results demonstrate that SinSEMI-generated images achieve both high fidelity and meaningful diversity, making them suitable as training data for semiconductor AI applications.
>
---
#### [new 081] Inference-Time Scaling of Diffusion Models for Infrared Data Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文面向红外图像生成任务，解决标注数据稀缺问题。通过微调FLUX.1模型并引入CLIP验证器进行推理时引导，提升文本到红外图像生成质量，在KAIST基准上FID降低10%。**

- **链接: [http://arxiv.org/pdf/2511.07362v1](http://arxiv.org/pdf/2511.07362v1)**

> **作者:** Kai A. Horstmann; Maxim Clouser; Kia Khezeli
>
> **备注:** Peer-reviewed workshop paper
>
> **摘要:** Infrared imagery enables temperature-based scene understanding using passive sensors, particularly under conditions of low visibility where traditional RGB imaging fails. Yet, developing downstream vision models for infrared applications is hindered by the scarcity of high-quality annotated data, due to the specialized expertise required for infrared annotation. While synthetic infrared image generation has the potential to accelerate model development by providing large-scale, diverse training data, training foundation-level generative diffusion models in the infrared domain has remained elusive due to limited datasets. In light of such data constraints, we explore an inference-time scaling approach using a domain-adapted CLIP-based verifier for enhanced infrared image generation quality. We adapt FLUX.1-dev, a state-of-the-art text-to-image diffusion model, to the infrared domain by finetuning it on a small sample of infrared images using parameter-efficient techniques. The trained verifier is then employed during inference to guide the diffusion sampling process toward higher quality infrared generations that better align with input text prompts. Empirically, we find that our approach leads to consistent improvements in generation quality, reducing FID scores on the KAIST Multispectral Pedestrian Detection Benchmark dataset by 10% compared to unguided baseline samples. Our results suggest that inference-time guidance offers a promising direction for bridging the domain gap in low-data infrared settings.
>
---
#### [new 082] Glioma C6: A Novel Dataset for Training and Benchmarking Cell Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Glioma C6数据集，用于胶质瘤C6细胞的实例分割任务，解决生物图像中细胞精准分割与泛化性不足的问题。工作包括构建12,000+标注细胞的高分辨率数据集，分基准与泛化两部分，并验证其提升模型性能的有效性。**

- **链接: [http://arxiv.org/pdf/2511.07286v1](http://arxiv.org/pdf/2511.07286v1)**

> **作者:** Roman Malashin; Svetlana Pashkevich; Daniil Ilyukhin; Arseniy Volkov; Valeria Yachnaya; Andrey Denisov; Maria Mikhalkova
>
> **摘要:** We present Glioma C6, a new open dataset for instance segmentation of glioma C6 cells, designed as both a benchmark and a training resource for deep learning models. The dataset comprises 75 high-resolution phase-contrast microscopy images with over 12,000 annotated cells, providing a realistic testbed for biomedical image analysis. It includes soma annotations and morphological cell categorization provided by biologists. Additional categorization of cells, based on morphology, aims to enhance the utilization of image data for cancer cell research. Glioma C6 consists of two parts: the first is curated with controlled parameters for benchmarking, while the second supports generalization testing under varying conditions. We evaluate the performance of several generalist segmentation models, highlighting their limitations on our dataset. Our experiments demonstrate that training on Glioma C6 significantly enhances segmentation performance, reinforcing its value for developing robust and generalizable models. The dataset is publicly available for researchers.
>
---
#### [new 083] 4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation
- **分类: cs.CV**

- **简介: 论文提出4DSTR，面向动态4D内容生成任务，解决时空不一致与快速运动适应差问题。通过时空校正与自适应高斯点增删，提升生成质量与一致性，实现高性能视频到4D生成。**

- **链接: [http://arxiv.org/pdf/2511.07241v1](http://arxiv.org/pdf/2511.07241v1)**

> **作者:** Mengmeng Liu; Jiuming Liu; Yunpeng Zhang; Jiangtao Li; Michael Ying Yang; Francesco Nex; Hao Cheng
>
> **备注:** Accepted by AAAI 2026.The first two authors contributed equally
>
> **摘要:** Remarkable advances in recent 2D image and 3D shape generation have induced a significant focus on dynamic 4D content generation. However, previous 4D generation methods commonly struggle to maintain spatial-temporal consistency and adapt poorly to rapid temporal variations, due to the lack of effective spatial-temporal modeling. To address these problems, we propose a novel 4D generation network called 4DSTR, which modulates generative 4D Gaussian Splatting with spatial-temporal rectification. Specifically, temporal correlation across generated 4D sequences is designed to rectify deformable scales and rotations and guarantee temporal consistency. Furthermore, an adaptive spatial densification and pruning strategy is proposed to address significant temporal variations by dynamically adding or deleting Gaussian points with the awareness of their pre-frame movements. Extensive experiments demonstrate that our 4DSTR achieves state-of-the-art performance in video-to-4D generation, excelling in reconstruction quality, spatial-temporal consistency, and adaptation to rapid temporal movements.
>
---
#### [new 084] Hilbert-Guided Block-Sparse Local Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对图像局部注意力计算效率低的问题，提出基于希尔伯特曲线重排序token的方法，提升块稀疏注意力的连续性，实现4×~18×加速，构建的Transformer模型在保持精度的同时显著提升推理速度。**

- **链接: [http://arxiv.org/pdf/2511.05832v1](http://arxiv.org/pdf/2511.05832v1)**

> **作者:** Yunge Li; Lanyu Xu
>
> **摘要:** The quadratic compute and memory costs of global self-attention severely limit its use in high-resolution images. Local attention reduces complexity by restricting attention to neighborhoods. Block-sparse kernels can further improve the efficiency of local attention, but conventional local attention patterns often fail to deliver significant speedups because tokens within a window are not contiguous in the 1D sequence. This work proposes a novel method for constructing windows and neighborhoods based on the Hilbert curve. Image tokens are first reordered along a Hilbert curve, and windows and neighborhoods are then formed on the reordered 1D sequence. From a block-sparse perspective, this strategy significantly increases block sparsity and can be combined with existing block-sparse kernels to improve the efficiency of 2D local attention. Experiments show that the proposed Hilbert Window Attention and Hilbert Slide Attention can accelerate window attention and slide attention by about $4\times$ and $18\times$, respectively. To assess practicality, the strategy is instantiated as the Hilbert Window Transformer and the Hilbert Neighborhood Transformer, both of which achieve end-to-end speedups with minimal accuracy loss. Overall, combining Hilbert-guided local attention with block-sparse kernels offers a general and practical approach to enhancing the efficiency of 2D local attention for images. The code is available at https://github.com/Yunge6666/Hilbert-Local-Attention.
>
---
#### [new 085] Understanding Cross Task Generalization in Handwriting-Based Alzheimer's Screening via Vision Language Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CLFA框架，利用CLIP实现手写行为的零样本阿尔茨海默病筛查，首次系统研究任务类型对跨任务泛化的影响，揭示关键书写特征，构建手写认知评估基准。**

- **链接: [http://arxiv.org/pdf/2511.05841v1](http://arxiv.org/pdf/2511.05841v1)**

> **作者:** Changqing Gong; Huafeng Qin; Mounim A. El-Yacoubi
>
> **摘要:** Alzheimer's disease is a prevalent neurodegenerative disorder for which early detection is critical. Handwriting-often disrupted in prodromal AD-provides a non-invasive and cost-effective window into subtle motor and cognitive decline. Existing handwriting-based AD studies, mostly relying on online trajectories and hand-crafted features, have not systematically examined how task type influences diagnostic performance and cross-task generalization. Meanwhile, large-scale vision language models have demonstrated remarkable zero or few-shot anomaly detection in natural images and strong adaptability across medical modalities such as chest X-ray and brain MRI. However, handwriting-based disease detection remains largely unexplored within this paradigm. To close this gap, we introduce a lightweight Cross-Layer Fusion Adapter framework that repurposes CLIP for handwriting-based AD screening. CLFA implants multi-level fusion adapters within the visual encoder to progressively align representations toward handwriting-specific medical cues, enabling prompt-free and efficient zero-shot inference. Using this framework, we systematically investigate cross-task generalization-training on a specific handwriting task and evaluating on unseen ones-to reveal which task types and writing patterns most effectively discriminate AD. Extensive analyses further highlight characteristic stroke patterns and task-level factors that contribute to early AD identification, offering both diagnostic insights and a benchmark for handwriting-based cognitive assessment.
>
---
#### [new 086] Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文面向鲁棒动作识别任务，解决RGB模型依赖统计模式、忽视物理空间姿态的问题。提出融合V-JEPA2动态与CoMotion人体姿态的模型，在高遮挡场景下显著提升性能，强调动作识别需基于空间理解。**

- **链接: [http://arxiv.org/pdf/2511.05622v1](http://arxiv.org/pdf/2511.05622v1)**

> **作者:** Nicholas Babey; Tiffany Gu; Yiheng Li; Cristian Meo; Kevin Zhu
>
> **备注:** Accepted at NeurIPS 2025 SpaVLE, for code see https://github.com/nbabey20/groundactrec , 9 pages, 1 figure
>
> **摘要:** For embodied agents to effectively understand and interact within the world around them, they require a nuanced comprehension of human actions grounded in physical space. Current action recognition models, often relying on RGB video, learn superficial correlations between patterns and action labels, so they struggle to capture underlying physical interaction dynamics and human poses in complex scenes. We propose a model architecture that grounds action recognition in physical space by fusing two powerful, complementary representations: V-JEPA 2's contextual, predictive world dynamics and CoMotion's explicit, occlusion-tolerant human pose data. Our model is validated on both the InHARD and UCF-19-Y-OCC benchmarks for general action recognition and high-occlusion action recognition, respectively. Our model outperforms three other baselines, especially within complex, occlusive scenes. Our findings emphasize a need for action recognition to be supported by spatial understanding instead of statistical pattern recognition.
>
---
#### [new 087] UniADC: A Unified Framework for Anomaly Detection and Classification
- **分类: cs.CV**

- **简介: 论文提出UniADC，首次统一异常检测与分类任务，解决二者割裂导致性能受限问题。通过无训练的可控修复网络生成异常样本，结合多任务判别器实现少样本甚至零样本下的高精度检测与分类。**

- **链接: [http://arxiv.org/pdf/2511.06644v1](http://arxiv.org/pdf/2511.06644v1)**

> **作者:** Ximiao Zhang; Min Xu; Zheng Zhang; Junlin Hu; Xiuzhuang Zhou
>
> **摘要:** In this paper, we introduce the task of unified anomaly detection and classification, which aims to simultaneously detect anomalous regions in images and identify their specific categories. Existing methods typically treat anomaly detection and classification as separate tasks, thereby neglecting their inherent correlation, limiting information sharing, and resulting in suboptimal performance. To address this, we propose UniADC, a unified anomaly detection and classification model that can effectively perform both tasks with only a few or even no anomaly images. Specifically, UniADC consists of two key components: a training-free controllable inpainting network and a multi-task discriminator. The inpainting network can synthesize anomaly images of specific categories by repainting normal regions guided by anomaly priors, and can also repaint few-shot anomaly samples to augment the available anomaly data. The multi-task discriminator is then trained on these synthesized samples, enabling precise anomaly detection and classification by aligning fine-grained image features with anomaly-category embeddings. We conduct extensive experiments on three anomaly detection and classification datasets, including MVTec-FS, MTD, and WFDD, and the results demonstrate that UniADC consistently outperforms existing methods in anomaly detection, localization, and classification. The code is available at https://github.com/cnulab/UniADC.
>
---
#### [new 088] VDNeRF: Vision-only Dynamic Neural Radiance Field for Urban Scenes
- **分类: cs.CV**

- **简介: VDNeRF提出一种仅用视觉信息的动态神经辐射场，无需相机位姿，联合优化静态背景与动态物体的时空表示，解决大尺度城市场景中位姿估计与动态建模难题，实现高精度新视角合成。**

- **链接: [http://arxiv.org/pdf/2511.06408v1](http://arxiv.org/pdf/2511.06408v1)**

> **作者:** Zhengyu Zou; Jingfeng Li; Hao Li; Xiaolei Hou; Jinwen Hu; Jingkun Chen; Lechao Cheng; Dingwen Zhang
>
> **摘要:** Neural Radiance Fields (NeRFs) implicitly model continuous three-dimensional scenes using a set of images with known camera poses, enabling the rendering of photorealistic novel views. However, existing NeRF-based methods encounter challenges in applications such as autonomous driving and robotic perception, primarily due to the difficulty of capturing accurate camera poses and limitations in handling large-scale dynamic environments. To address these issues, we propose Vision-only Dynamic NeRF (VDNeRF), a method that accurately recovers camera trajectories and learns spatiotemporal representations for dynamic urban scenes without requiring additional camera pose information or expensive sensor data. VDNeRF employs two separate NeRF models to jointly reconstruct the scene. The static NeRF model optimizes camera poses and static background, while the dynamic NeRF model incorporates the 3D scene flow to ensure accurate and consistent reconstruction of dynamic objects. To address the ambiguity between camera motion and independent object motion, we design an effective and powerful training framework to achieve robust camera pose estimation and self-supervised decomposition of static and dynamic elements in a scene. Extensive evaluations on mainstream urban driving datasets demonstrate that VDNeRF surpasses state-of-the-art NeRF-based pose-free methods in both camera pose estimation and dynamic novel view synthesis.
>
---
#### [new 089] DIMO: Diverse 3D Motion Generation for Arbitrary Objects
- **分类: cs.CV**

- **简介: DIMO提出一种从单张图像生成任意物体多样3D运动的方法，利用视频模型先验学习低维运动潜空间，通过神经关键点驱动高斯模型实现快速采样，支持插值与语言引导生成。**

- **链接: [http://arxiv.org/pdf/2511.07409v1](http://arxiv.org/pdf/2511.07409v1)**

> **作者:** Linzhan Mou; Jiahui Lei; Chen Wang; Lingjie Liu; Kostas Daniilidis
>
> **备注:** Published in ICCV 2025, project page https://linzhanm.github.io/dimo
>
> **摘要:** We present DIMO, a generative approach capable of generating diverse 3D motions for arbitrary objects from a single image. The core idea of our work is to leverage the rich priors in well-trained video models to extract the common motion patterns and then embed them into a shared low-dimensional latent space. Specifically, we first generate multiple videos of the same object with diverse motions. We then embed each motion into a latent vector and train a shared motion decoder to learn the distribution of motions represented by a structured and compact motion representation, i.e., neural key point trajectories. The canonical 3D Gaussians are then driven by these key points and fused to model the geometry and appearance. During inference time with learned latent space, we can instantly sample diverse 3D motions in a single-forward pass and support several interesting applications including 3D motion interpolation and language-guided motion generation. Our project page is available at https://linzhanm.github.io/dimo.
>
---
#### [new 090] HENet++: Hybrid Encoding and Multi-task Learning for 3D Perception and End-to-end Autonomous Driving
- **分类: cs.CV**

- **简介: HENet++提出混合编码与多任务学习框架，解决3D感知任务中特征表示冲突与计算资源受限问题，通过长短时图像编码与稠密/稀疏特征协同提取，实现高精度端到端自动驾驶，在nuScenes上达SOTA。**

- **链接: [http://arxiv.org/pdf/2511.07106v1](http://arxiv.org/pdf/2511.07106v1)**

> **作者:** Zhongyu Xia; Zhiwei Lin; Yongtao Wang; Ming-Hsuan Yang
>
> **备注:** Preliminary version, 19 pages
>
> **摘要:** Three-dimensional feature extraction is a critical component of autonomous driving systems, where perception tasks such as 3D object detection, bird's-eye-view (BEV) semantic segmentation, and occupancy prediction serve as important constraints on 3D features. While large image encoders, high-resolution images, and long-term temporal inputs can significantly enhance feature quality and deliver remarkable performance gains, these techniques are often incompatible in both training and inference due to computational resource constraints. Moreover, different tasks favor distinct feature representations, making it difficult for a single model to perform end-to-end inference across multiple tasks while maintaining accuracy comparable to that of single-task models. To alleviate these issues, we present the HENet and HENet++ framework for multi-task 3D perception and end-to-end autonomous driving. Specifically, we propose a hybrid image encoding network that uses a large image encoder for short-term frames and a small one for long-term frames. Furthermore, our framework simultaneously extracts both dense and sparse features, providing more suitable representations for different tasks, reducing cumulative errors, and delivering more comprehensive information to the planning module. The proposed architecture maintains compatibility with various existing 3D feature extraction methods and supports multimodal inputs. HENet++ achieves state-of-the-art end-to-end multi-task 3D perception results on the nuScenes benchmark, while also attaining the lowest collision rate on the nuScenes end-to-end autonomous driving benchmark.
>
---
#### [new 091] From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models Without Task Knowledge
- **分类: cs.CV; cs.CR**

- **简介: 该论文提出Transferable Video Attack（TVA），利用视频基础模型的时序表征动态，在无任务知识下生成可迁移对抗扰动，攻击下游模型与多模态大模型，揭示了视频模型部署中的新型安全漏洞。**

- **链接: [http://arxiv.org/pdf/2511.07049v1](http://arxiv.org/pdf/2511.07049v1)**

> **作者:** Hui Lu; Yi Yu; Song Xia; Yiming Yang; Deepu Rajan; Boon Poh Ng; Alex Kot; Xudong Jiang
>
> **备注:** AAAI 2026 (Oral presentation)
>
> **摘要:** Large-scale Video Foundation Models (VFMs) has significantly advanced various video-related tasks, either through task-specific models or Multi-modal Large Language Models (MLLMs). However, the open accessibility of VFMs also introduces critical security risks, as adversaries can exploit full knowledge of the VFMs to launch potent attacks. This paper investigates a novel and practical adversarial threat scenario: attacking downstream models or MLLMs fine-tuned from open-source VFMs, without requiring access to the victim task, training data, model query, and architecture. In contrast to conventional transfer-based attacks that rely on task-aligned surrogate models, we demonstrate that adversarial vulnerabilities can be exploited directly from the VFMs. To this end, we propose the Transferable Video Attack (TVA), a temporal-aware adversarial attack method that leverages the temporal representation dynamics of VFMs to craft effective perturbations. TVA integrates a bidirectional contrastive learning mechanism to maximize the discrepancy between the clean and adversarial features, and introduces a temporal consistency loss that exploits motion cues to enhance the sequential impact of perturbations. TVA avoids the need to train expensive surrogate models or access to domain-specific data, thereby offering a more practical and efficient attack strategy. Extensive experiments across 24 video-related tasks demonstrate the efficacy of TVA against downstream models and MLLMs, revealing a previously underexplored security vulnerability in the deployment of video models.
>
---
#### [new 092] M2S2L: Mamba-based Multi-Scale Spatial-temporal Learning for Video Anomaly Detection
- **分类: cs.CV**

- **简介: 论文提出M2S2L框架，用于视频异常检测，解决传统方法在时空建模与效率间的平衡难题。基于Mamba结构，构建多尺度时空编码器与特征分解机制，实现高精度（最高98.5% AUC）与高效推理（45 FPS）。**

- **链接: [http://arxiv.org/pdf/2511.05564v1](http://arxiv.org/pdf/2511.05564v1)**

> **作者:** Yang Liu; Boan Chen; Xiaoguang Zhu; Jing Liu; Peng Sun; Wei Zhou
>
> **备注:** IEEE VCIP 2025
>
> **摘要:** Video anomaly detection (VAD) is an essential task in the image processing community with prospects in video surveillance, which faces fundamental challenges in balancing detection accuracy with computational efficiency. As video content becomes increasingly complex with diverse behavioral patterns and contextual scenarios, traditional VAD approaches struggle to provide robust assessment for modern surveillance systems. Existing methods either lack comprehensive spatial-temporal modeling or require excessive computational resources for real-time applications. In this regard, we present a Mamba-based multi-scale spatial-temporal learning (M2S2L) framework in this paper. The proposed method employs hierarchical spatial encoders operating at multiple granularities and multi-temporal encoders capturing motion dynamics across different time scales. We also introduce a feature decomposition mechanism to enable task-specific optimization for appearance and motion reconstruction, facilitating more nuanced behavioral modeling and quality-aware anomaly assessment. Experiments on three benchmark datasets demonstrate that M2S2L framework achieves 98.5%, 92.1%, and 77.9% frame-level AUCs on UCSD Ped2, CUHK Avenue, and ShanghaiTech respectively, while maintaining efficiency with 20.1G FLOPs and 45 FPS inference speed, making it suitable for practical surveillance deployment.
>
---
#### [new 093] Sign language recognition from skeletal data using graph and recurrent neural networks
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文面向孤立手语手势识别任务，利用骨骼数据建模时空依赖，提出Graph-GRU网络，融合图结构空间表征与循环时序建模，在AUTSL数据集上实现高精度识别，验证了姿态驱动方法的有效性。**

- **链接: [http://arxiv.org/pdf/2511.05772v1](http://arxiv.org/pdf/2511.05772v1)**

> **作者:** B. Mederos; J. Mejía; A. Medina-Reyes; Y. Espinosa-Almeyda; J. D. Díaz-Roman; I. Rodríguez-Mederos; M. Mejía-Carreon; F. Gonzalez-Lopez
>
> **备注:** 15 pages, 2 figures
>
> **摘要:** This work presents an approach for recognizing isolated sign language gestures using skeleton-based pose data extracted from video sequences. A Graph-GRU temporal network is proposed to model both spatial and temporal dependencies between frames, enabling accurate classification. The model is trained and evaluated on the AUTSL (Ankara university Turkish sign language) dataset, achieving high accuracy. Experimental results demonstrate the effectiveness of integrating graph-based spatial representations with temporal modeling, providing a scalable framework for sign language recognition. The results of this approach highlight the potential of pose-driven methods for sign language understanding.
>
---
#### [new 094] Randomized-MLP Regularization Improves Domain Adaptation and Interpretability in DINOv2
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Randomized-MLP正则化方法，用于改进DINOv2在域适应中的可解释性与性能，解决ViT在医学影像中低信息patch滥用导致的注意力不透明问题，通过对比学习提升表征语义对齐。**

- **链接: [http://arxiv.org/pdf/2511.05509v1](http://arxiv.org/pdf/2511.05509v1)**

> **作者:** Joel Valdivia Ortega; Lorenz Lamm; Franziska Eckardt; Benedikt Schworm; Marion Jasnin; Tingying Peng
>
> **摘要:** Vision Transformers (ViTs), such as DINOv2, achieve strong performance across domains but often repurpose low-informative patch tokens in ways that reduce the interpretability of attention and feature maps. This challenge is especially evident in medical imaging, where domain shifts can degrade both performance and transparency. In this paper, we introduce Randomized-MLP (RMLP) regularization, a contrastive learning-based method that encourages more semantically aligned representations. We use RMLPs when fine-tuning DINOv2 to both medical and natural image modalities, showing that it improves or maintains downstream performance while producing more interpretable attention maps. We also provide a mathematical analysis of RMLPs, offering insights into its role in enhancing ViT-based models and advancing our understanding of contrastive learning.
>
---
#### [new 095] Compressing Multi-Task Model for Autonomous Driving via Pruning and Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶多任务感知模型参数过大问题，提出融合任务感知剪枝与特征级知识蒸馏的压缩方法，在BDD100K上实现32.7%参数减少，性能几乎无损，仍保持32.7 FPS实时推理。**

- **链接: [http://arxiv.org/pdf/2511.05557v1](http://arxiv.org/pdf/2511.05557v1)**

> **作者:** Jiayuan Wang; Q. M. Jonathan Wu; Ning Zhang; Katsuya Suto; Lei Zhong
>
> **摘要:** Autonomous driving systems rely on panoptic perception to jointly handle object detection, drivable area segmentation, and lane line segmentation. Although multi-task learning is an effective way to integrate these tasks, its increasing model parameters and complexity make deployment on on-board devices difficult. To address this challenge, we propose a multi-task model compression framework that combines task-aware safe pruning with feature-level knowledge distillation. Our safe pruning strategy integrates Taylor-based channel importance with gradient conflict penalty to keep important channels while removing redundant and conflicting channels. To mitigate performance degradation after pruning, we further design a task head-agnostic distillation method that transfers intermediate backbone and encoder features from a teacher to a student model as guidance. Experiments on the BDD100K dataset demonstrate that our compressed model achieves a 32.7% reduction in parameters while segmentation performance shows negligible accuracy loss and only a minor decrease in detection (-1.2% for Recall and -1.8% for mAP50) compared to the teacher. The compressed model still runs at 32.7 FPS in real-time. These results show that combining pruning and knowledge distillation provides an effective compression solution for multi-task panoptic perception.
>
---
#### [new 096] MambaOVSR: Multiscale Fusion with Global Motion Modeling for Chinese Opera Video Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向中国戏曲视频超分辨率任务，解决低质老视频因运动复杂、数据稀缺导致的细节丢失问题，提出MambaOVSR模型，构建首个COVC数据集，引入多尺度融合与全局运动建模模块，显著提升重建质量。**

- **链接: [http://arxiv.org/pdf/2511.06172v1](http://arxiv.org/pdf/2511.06172v1)**

> **作者:** Hua Chang; Xin Xu; Wei Liu; Wei Wang; Xin Yuan; Kui Jiang
>
> **摘要:** Chinese opera is celebrated for preserving classical art. However, early filming equipment limitations have degraded videos of last-century performances by renowned artists (e.g., low frame rates and resolution), hindering archival efforts. Although space-time video super-resolution (STVSR) has advanced significantly, applying it directly to opera videos remains challenging. The scarcity of datasets impedes the recovery of high frequency details, and existing STVSR methods lack global modeling capabilities, compromising visual quality when handling opera's characteristic large motions. To address these challenges, we pioneer a large scale Chinese Opera Video Clip (COVC) dataset and propose the Mamba-based multiscale fusion network for space-time Opera Video Super-Resolution (MambaOVSR). Specifically, MambaOVSR involves three novel components: the Global Fusion Module (GFM) for motion modeling through a multiscale alternating scanning mechanism, and the Multiscale Synergistic Mamba Module (MSMM) for alignment across different sequence lengths. Additionally, our MambaVR block resolves feature artifacts and positional information loss during alignment. Experimental results on the COVC dataset show that MambaOVSR significantly outperforms the SOTA STVSR method by an average of 1.86 dB in terms of PSNR. Dataset and Code will be publicly released.
>
---
#### [new 097] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction
- **分类: cs.CV**

- **简介: Sparse4DGS首次实现稀疏帧动态场景重建，针对纹理丰富区域提出纹理感知变形正则化与 Canonical 优化，解决传统方法在稀疏帧下重建失败问题，显著提升重建质量。**

- **链接: [http://arxiv.org/pdf/2511.07122v1](http://arxiv.org/pdf/2511.07122v1)**

> **作者:** Changyue Shi; Chuxiao Yang; Xinyuan Hu; Minghao Chen; Wenwen Pan; Yan Yang; Jiajun Ding; Zhou Yu; Jun Yu
>
> **备注:** AAAI 2026
>
> **摘要:** Dynamic Gaussian Splatting approaches have achieved remarkable performance for 4D scene reconstruction. However, these approaches rely on dense-frame video sequences for photorealistic reconstruction. In real-world scenarios, due to equipment constraints, sometimes only sparse frames are accessible. In this paper, we propose Sparse4DGS, the first method for sparse-frame dynamic scene reconstruction. We observe that dynamic reconstruction methods fail in both canonical and deformed spaces under sparse-frame settings, especially in areas with high texture richness. Sparse4DGS tackles this challenge by focusing on texture-rich areas. For the deformation network, we propose Texture-Aware Deformation Regularization, which introduces a texture-based depth alignment loss to regulate Gaussian deformation. For the canonical Gaussian field, we introduce Texture-Aware Canonical Optimization, which incorporates texture-based noise into the gradient descent process of canonical Gaussians. Extensive experiments show that when taking sparse frames as inputs, our method outperforms existing dynamic or few-shot techniques on NeRF-Synthetic, HyperNeRF, NeRF-DS, and our iPhone-4D datasets.
>
---
#### [new 098] Gait Recognition via Collaborating Discriminative and Generative Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出CoD²框架，用于步态识别任务，解决生成模型潜力未被充分挖掘的问题。通过联合判别与生成扩散模型，利用多级条件控制提升特征鲁棒性，实现性能突破并兼容现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06245v1](http://arxiv.org/pdf/2511.06245v1)**

> **作者:** Haijun Xiong; Bin Feng; Bang Wang; Xinggang Wang; Wenyu Liu
>
> **备注:** 14 pages, 4figures
>
> **摘要:** Gait recognition offers a non-intrusive biometric solution by identifying individuals through their walking patterns. Although discriminative models have achieved notable success in this domain, the full potential of generative models remains largely underexplored. In this paper, we introduce \textbf{CoD$^2$}, a novel framework that combines the data distribution modeling capabilities of diffusion models with the semantic representation learning strengths of discriminative models to extract robust gait features. We propose a Multi-level Conditional Control strategy that incorporates both high-level identity-aware semantic conditions and low-level visual details. Specifically, the high-level condition, extracted by the discriminative extractor, guides the generation of identity-consistent gait sequences, whereas low-level visual details, such as appearance and motion, are preserved to enhance consistency. Furthermore, the generated sequences facilitate the discriminative extractor's learning, enabling it to capture more comprehensive high-level semantic features. Extensive experiments on four datasets (SUSTech1K, CCPG, GREW, and Gait3D) demonstrate that CoD$^2$ achieves state-of-the-art performance and can be seamlessly integrated with existing discriminative methods, yielding consistent improvements.
>
---
#### [new 099] LiteUpdate: A Lightweight Framework for Updating AI-Generated Image Detectors
- **分类: cs.CV; cs.CR**

- **简介: 该论文提出LiteUpdate框架，用于高效更新AI生成图像检测器，解决新生成模型导致的检测性能下降与灾难性遗忘问题。通过边界样本选择与多轨迹模型融合，显著提升检测准确率与更新效率。**

- **链接: [http://arxiv.org/pdf/2511.07192v1](http://arxiv.org/pdf/2511.07192v1)**

> **作者:** Jiajie Lu; Zhenkan Fu; Na Zhao; Long Xing; Kejiang Chen; Weiming Zhang; Nenghai Yu
>
> **摘要:** The rapid progress of generative AI has led to the emergence of new generative models, while existing detection methods struggle to keep pace, resulting in significant degradation in the detection performance. This highlights the urgent need for continuously updating AI-generated image detectors to adapt to new generators. To overcome low efficiency and catastrophic forgetting in detector updates, we propose LiteUpdate, a lightweight framework for updating AI-generated image detectors. LiteUpdate employs a representative sample selection module that leverages image confidence and gradient-based discriminative features to precisely select boundary samples. This approach improves learning and detection accuracy on new distributions with limited generated images, significantly enhancing detector update efficiency. Additionally, LiteUpdate incorporates a model merging module that fuses weights from multiple fine-tuning trajectories, including pre-trained, representative, and random updates. This balances the adaptability to new generators and mitigates the catastrophic forgetting of prior knowledge. Experiments demonstrate that LiteUpdate substantially boosts detection performance in various detectors. Specifically, on AIDE, the average detection accuracy on Midjourney improved from 87.63% to 93.03%, a 6.16% relative increase.
>
---
#### [new 100] PlantTraitNet: An Uncertainty-Aware Multimodal Framework for Global-Scale Plant Trait Inference from Citizen Science Data
- **分类: cs.CV; cs.AI**

- **简介: 论文提出PlantTraitNet，一种多模态不确定性感知深度学习框架，利用全球公民科学植物照片预测四种植物性状，构建高精度全球性状分布图，解决传统测量成本高、覆盖稀疏问题，显著超越现有产品。**

- **链接: [http://arxiv.org/pdf/2511.06943v1](http://arxiv.org/pdf/2511.06943v1)**

> **作者:** Ayushi Sharma; Johanna Trost; Daniel Lusk; Johannes Dollinger; Julian Schrader; Christian Rossi; Javier Lopatin; Etienne Laliberté; Simon Haberstroh; Jana Eichel; Daniel Mederer; Jose Miguel Cerda-Paredes; Shyam S. Phartyal; Lisa-Maricia Schwarz; Anja Linstädter; Maria Conceição Caldeira; Teja Kattenborn
>
> **备注:** Preprint version of the paper accepted at the 40th AAAI Conference on Artificial Intelligence (AAAI-26), organized by the Association for the Advancement of Artificial Intelligence
>
> **摘要:** Global plant maps of plant traits, such as leaf nitrogen or plant height, are essential for understanding ecosystem processes, including the carbon and energy cycles of the Earth system. However, existing trait maps remain limited by the high cost and sparse geographic coverage of field-based measurements. Citizen science initiatives offer a largely untapped resource to overcome these limitations, with over 50 million geotagged plant photographs worldwide capturing valuable visual information on plant morphology and physiology. In this study, we introduce PlantTraitNet, a multi-modal, multi-task uncertainty-aware deep learning framework that predictsfour key plant traits (plant height, leaf area, specific leaf area, and nitrogen content) from citizen science photos using weak supervision. By aggregating individual trait predictions across space, we generate global maps of trait distributions. We validate these maps against independent vegetation survey data (sPlotOpen) and benchmark them against leading global trait products. Our results show that PlantTraitNet consistently outperforms existing trait maps across all evaluated traits, demonstrating that citizen science imagery, when integrated with computer vision and geospatial AI, enables not only scalable but also more accurate global trait mapping. This approach offers a powerful new pathway for ecological research and Earth system modeling.
>
---
#### [new 101] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling
- **分类: cs.CV**

- **简介: NURBGen首次实现从文本直接生成高保真NURBS格式CAD模型，解决现有方法输出网格或依赖稀疏历史数据的问题。通过微调LLM生成NURBS参数，结合混合表示与新数据集partABC，提升几何精度与可编辑性。**

- **链接: [http://arxiv.org/pdf/2511.06194v1](http://arxiv.org/pdf/2511.06194v1)**

> **作者:** Muhammad Usama; Mohammad Sadil Khan; Didier Stricker; Muhammad Zeshan Afzal
>
> **备注:** Accepted in AAAI 2026
>
> **摘要:** Generating editable 3D CAD models from natural language remains challenging, as existing text-to-CAD systems either produce meshes or rely on scarce design-history data. We present NURBGen, the first framework to generate high-fidelity 3D CAD models directly from text using Non-Uniform Rational B-Splines (NURBS). To achieve this, we fine-tune a large language model (LLM) to translate free-form texts into JSON representations containing NURBS surface parameters (\textit{i.e}, control points, knot vectors, degrees, and rational weights) which can be directly converted into BRep format using Python. We further propose a hybrid representation that combines untrimmed NURBS with analytic primitives to handle trimmed surfaces and degenerate regions more robustly, while reducing token complexity. Additionally, we introduce partABC, a curated subset of the ABC dataset consisting of individual CAD components, annotated with detailed captions using an automated annotation pipeline. NURBGen demonstrates strong performance on diverse prompts, surpassing prior methods in geometric fidelity and dimensional accuracy, as confirmed by expert evaluations. Code and dataset will be released publicly.
>
---
#### [new 102] Breaking the Stealth-Potency Trade-off in Clean-Image Backdoors with Generative Trigger Optimization
- **分类: cs.CV; cs.CR; cs.LG; 68T07; I.2.6**

- **简介: 该论文提出Generative Clean-Image Backdoors（GCB），通过生成式触发器优化，在不引入明显视觉异常的前提下，实现极低毒化率的干净图像后门攻击，解决 Stealth-Potency 权衡问题，显著降低清洁准确率损失至1%以下。**

- **链接: [http://arxiv.org/pdf/2511.07210v1](http://arxiv.org/pdf/2511.07210v1)**

> **作者:** Binyan Xu; Fan Yang; Di Tang; Xilin Dai; Kehuan Zhang
>
> **备注:** 19 pages, 22 figures, 15 tables. To appear in AAAI '26 (Oral). This paper extends the AAAI-2026 version by including the Appendix
>
> **摘要:** Clean-image backdoor attacks, which use only label manipulation in training datasets to compromise deep neural networks, pose a significant threat to security-critical applications. A critical flaw in existing methods is that the poison rate required for a successful attack induces a proportional, and thus noticeable, drop in Clean Accuracy (CA), undermining their stealthiness. This paper presents a new paradigm for clean-image attacks that minimizes this accuracy degradation by optimizing the trigger itself. We introduce Generative Clean-Image Backdoors (GCB), a framework that uses a conditional InfoGAN to identify naturally occurring image features that can serve as potent and stealthy triggers. By ensuring these triggers are easily separable from benign task-related features, GCB enables a victim model to learn the backdoor from an extremely small set of poisoned examples, resulting in a CA drop of less than 1%. Our experiments demonstrate GCB's remarkable versatility, successfully adapting to six datasets, five architectures, and four tasks, including the first demonstration of clean-image backdoors in regression and segmentation. GCB also exhibits resilience against most of the existing backdoor defenses.
>
---
#### [new 103] MirrorMamba: Towards Scalable and Robust Mirror Detection in Videos
- **分类: cs.CV; cs.AI**

- **简介: 论文提出MirrorMamba，首次将Mamba架构应用于视频镜面检测任务，解决现有方法依赖单一动态特征、计算复杂高、边界模糊等问题，通过多线索融合与Mamba模块提升性能与鲁棒性，显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06716v1](http://arxiv.org/pdf/2511.06716v1)**

> **作者:** Rui Song; Jiaying Lin; Rynson W. H. Lau
>
> **摘要:** Video mirror detection has received significant research attention, yet existing methods suffer from limited performance and robustness. These approaches often over-rely on single, unreliable dynamic features, and are typically built on CNNs with limited receptive fields or Transformers with quadratic computational complexity. To address these limitations, we propose a new effective and scalable video mirror detection method, called MirrorMamba. Our approach leverages multiple cues to adapt to diverse conditions, incorporating perceived depth, correspondence and optical. We also introduce an innovative Mamba-based Multidirection Correspondence Extractor, which benefits from the global receptive field and linear complexity of the emerging Mamba spatial state model to effectively capture correspondence properties. Additionally, we design a Mamba-based layer-wise boundary enforcement decoder to resolve the unclear boundary caused by the blurred depth map. Notably, this work marks the first successful application of the Mamba-based architecture in the field of mirror detection. Extensive experiments demonstrate that our method outperforms existing state-of-the-art approaches for video mirror detection on the benchmark datasets. Furthermore, on the most challenging and representative image-based mirror detection dataset, our approach achieves state-of-the-art performance, proving its robustness and generalizability.
>
---
#### [new 104] Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning
- **分类: cs.CV**

- **简介: 该论文针对雨天多视角图像导致的3D场景重建质量下降问题，构建了包含视角异质性与亮度动态性的OmniRain3D数据集，并提出REVR-GSNet框架，通过联合优化实现雨滴消除与高保真重建。**

- **链接: [http://arxiv.org/pdf/2511.06734v1](http://arxiv.org/pdf/2511.06734v1)**

> **作者:** Qianfeng Yang; Xiang Chen; Pengpeng Li; Qiyuan Guan; Guiyue Jin; Jiyu Jin
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Rain degrades the visual quality of multi-view images, which are essential for 3D scene reconstruction, resulting in inaccurate and incomplete reconstruction results. Existing datasets often overlook two critical characteristics of real rainy 3D scenes: the viewpoint-dependent variation in the appearance of rain streaks caused by their projection onto 2D images, and the reduction in ambient brightness resulting from cloud coverage during rainfall. To improve data realism, we construct a new dataset named OmniRain3D that incorporates perspective heterogeneity and brightness dynamicity, enabling more faithful simulation of rain degradation in 3D scenes. Based on this dataset, we propose an end-to-end reconstruction framework named REVR-GSNet (Rain Elimination and Visibility Recovery for 3D Gaussian Splatting). Specifically, REVR-GSNet integrates recursive brightness enhancement, Gaussian primitive optimization, and GS-guided rain elimination into a unified architecture through joint alternating optimization, achieving high-fidelity reconstruction of clean 3D scenes from rain-degraded inputs. Extensive experiments show the effectiveness of our dataset and method. Our dataset and method provide a foundation for future research on multi-view image deraining and rainy 3D scene reconstruction.
>
---
#### [new 105] C3-Diff: Super-resolving Spatial Transcriptomics via Cross-modal Cross-content Contrastive Diffusion Modelling
- **分类: cs.CV; cs.AI**

- **简介: C3-Diff提出一种跨模态跨内容对比扩散模型，解决空间转录组学分辨率低的问题，通过整合组织图像与基因表达数据，提升空间基因表达图谱精度，并增强下游生物分析能力。**

- **链接: [http://arxiv.org/pdf/2511.05571v1](http://arxiv.org/pdf/2511.05571v1)**

> **作者:** Xiaofei Wang; Stephen Price; Chao Li
>
> **摘要:** The rapid advancement of spatial transcriptomics (ST), i.e., spatial gene expressions, has made it possible to measure gene expression within original tissue, enabling us to discover molecular mechanisms. However, current ST platforms frequently suffer from low resolution, limiting the in-depth understanding of spatial gene expression. Super-resolution approaches promise to enhance ST maps by integrating histology images with gene expressions of profiled tissue spots. However, it remains a challenge to model the interactions between histology images and gene expressions for effective ST enhancement. This study presents a cross-modal cross-content contrastive diffusion framework, called C3-Diff, for ST enhancement with histology images as guidance. In C3-Diff, we firstly analyze the deficiency of traditional contrastive learning paradigm, which is then refined to extract both modal-invariant and content-invariant features of ST maps and histology images. Further, to overcome the problem of low sequencing sensitivity in ST maps, we perform nosing-based information augmentation on the surface of feature unit hypersphere. Finally, we propose a dynamic cross-modal imputation-based training strategy to mitigate ST data scarcity. We tested C3-Diff by benchmarking its performance on four public datasets, where it achieves significant improvements over competing methods. Moreover, we evaluate C3-Diff on downstream tasks of cell type localization, gene expression correlation and single-cell-level gene expression prediction, promoting AI-enhanced biotechnology for biomedical research and clinical applications. Codes are available at https://github.com/XiaofeiWang2018/C3-Diff.
>
---
#### [new 106] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video
- **分类: cs.CV**

- **简介: 论文提出StreamSTGS，用于实时流式自由视点视频，解决3D高斯泼溅存储过大问题。通过将高斯属性编码为图像、时序特征编码为视频，结合滑动窗口与Transformer模块，实现高效压缩与自适应码率，帧大小降至170KB，PSNR提升1dB。**

- **链接: [http://arxiv.org/pdf/2511.06046v1](http://arxiv.org/pdf/2511.06046v1)**

> **作者:** Zhihui Ke; Yuyang Liu; Xiaobo Zhou; Tie Qiu
>
> **备注:** Accepted by AAAI 2026. Code will be released at https://www.github.com/kkkzh/StreamSTGS
>
> **摘要:** Streaming free-viewpoint video~(FVV) in real-time still faces significant challenges, particularly in training, rendering, and transmission efficiency. Harnessing superior performance of 3D Gaussian Splatting~(3DGS), recent 3DGS-based FVV methods have achieved notable breakthroughs in both training and rendering. However, the storage requirements of these methods can reach up to $10$MB per frame, making stream FVV in real-time impossible. To address this problem, we propose a novel FVV representation, dubbed StreamSTGS, designed for real-time streaming. StreamSTGS represents a dynamic scene using canonical 3D Gaussians, temporal features, and a deformation field. For high compression efficiency, we encode canonical Gaussian attributes as 2D images and temporal features as a video. This design not only enables real-time streaming, but also inherently supports adaptive bitrate control based on network condition without any extra training. Moreover, we propose a sliding window scheme to aggregate adjacent temporal features to learn local motions, and then introduce a transformer-guided auxiliary training module to learn global motions. On diverse FVV benchmarks, StreamSTGS demonstrates competitive performance on all metrics compared to state-of-the-art methods. Notably, StreamSTGS increases the PSNR by an average of $1$dB while reducing the average frame size to just $170$KB. The code is publicly available on https://github.com/kkkzh/StreamSTGS.
>
---
#### [new 107] On Accurate and Robust Estimation of 3D and 2D Circular Center: Method and Application to Camera-Lidar Calibration
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对LiDAR-相机外参标定中3D-2D圆心匹配不准确的问题，提出基于共形几何代数与弦长方差最小化的双端圆心估计算法，提升标定精度与鲁棒性，支持自然圆形目标。**

- **链接: [http://arxiv.org/pdf/2511.06611v1](http://arxiv.org/pdf/2511.06611v1)**

> **作者:** Jiajun Jiang; Xiao Hu; Wancheng Liu; Wei Jiang
>
> **摘要:** Circular targets are widely used in LiDAR-camera extrinsic calibration due to their geometric consistency and ease of detection. However, achieving accurate 3D-2D circular center correspondence remains challenging. Existing methods often fail due to decoupled 3D fitting and erroneous 2D ellipse-center estimation. To address this, we propose a geometrically principled framework featuring two innovations: (i) a robust 3D circle center estimator based on conformal geometric algebra and RANSAC; and (ii) a chord-length variance minimization method to recover the true 2D projected center, resolving its dual-minima ambi- guity via homography validation or a quasi-RANSAC fallback. Evaluated on synthetic and real-world datasets, our framework significantly outperforms state-of-the-art approaches. It reduces extrinsic estimation error and enables robust calibration across diverse sensors and target types, including natural circular objects. Our code will be publicly released for reproducibility.
>
---
#### [new 108] Physics-Informed Image Restoration via Progressive PDE Integration
- **分类: cs.CV**

- **简介: 该论文针对运动模糊图像恢复任务，提出一种基于偏微分方程（PDE）的物理引导框架，通过扩散-对流方程建模模糊方向性，增强主流网络的全局建模能力，在仅增加1%计算开销下显著提升恢复质量。**

- **链接: [http://arxiv.org/pdf/2511.06244v1](http://arxiv.org/pdf/2511.06244v1)**

> **作者:** Shamika Likhite; Santiago López-Tapia; Aggelos K. Katsaggelos
>
> **摘要:** Motion blur, caused by relative movement between camera and scene during exposure, significantly degrades image quality and impairs downstream computer vision tasks such as object detection, tracking, and recognition in dynamic environments. While deep learning-based motion deblurring methods have achieved remarkable progress, existing approaches face fundamental challenges in capturing the long-range spatial dependencies inherent in motion blur patterns. Traditional convolutional methods rely on limited receptive fields and require extremely deep networks to model global spatial relationships. These limitations motivate the need for alternative approaches that incorporate physical priors to guide feature evolution during restoration. In this paper, we propose a progressive training framework that integrates physics-informed PDE dynamics into state-of-the-art restoration architectures. By leveraging advection-diffusion equations to model feature evolution, our approach naturally captures the directional flow characteristics of motion blur while enabling principled global spatial modeling. Our PDE-enhanced deblurring models achieve superior restoration quality with minimal overhead, adding only approximately 1\% to inference GMACs while providing consistent improvements in perceptual quality across multiple state-of-the-art architectures. Comprehensive experiments on standard motion deblurring benchmarks demonstrate that our physics-informed approach improves PSNR and SSIM significantly across four diverse architectures, including FFTformer, NAFNet, Restormer, and Stripformer. These results validate that incorporating mathematical physics principles through PDE-based global layers can enhance deep learning-based image restoration, establishing a promising direction for physics-informed neural network design in computer vision applications.
>
---
#### [new 109] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出一种人机协同的都市设计框架，利用视觉-语言模型与共现嵌入，从真实场景中挖掘物体搭配模式，为用户提供三物组合设计建议，以低干预方式支持本地化、经验驱动的微更新，突破传统顶层规划模式。**

- **链接: [http://arxiv.org/pdf/2511.06201v1](http://arxiv.org/pdf/2511.06201v1)**

> **作者:** Rodrigo Gallardo; Oz Fishman; Alexander Htet Kyaw
>
> **备注:** Accepted to NEURIPS 2025 Creative AI Track
>
> **摘要:** This paper introduces a human-in-the-loop computer vision framework that uses generative AI to propose micro-scale design interventions in public space and support more continuous, local participation. Using Grounding DINO and a curated subset of the ADE20K dataset as a proxy for the urban built environment, the system detects urban objects and builds co-occurrence embeddings that reveal common spatial configurations. From this analysis, the user receives five statistically likely complements to a chosen anchor object. A vision language model then reasons over the scene image and the selected pair to suggest a third object that completes a more complex urban tactic. The workflow keeps people in control of selection and refinement and aims to move beyond top-down master planning by grounding choices in everyday patterns and lived experience.
>
---
#### [new 110] MCFCN: Multi-View Clustering via a Fusion-Consensus Graph Convolutional Network
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出MCFCN，用于多视图聚类任务，解决现有方法忽略拓扑结构、噪声敏感和优化不一致等问题。通过融合共识图卷积网络与对齐损失，实现端到端的跨视图一致表示学习，显著提升聚类性能。**

- **链接: [http://arxiv.org/pdf/2511.05554v1](http://arxiv.org/pdf/2511.05554v1)**

> **作者:** Chenping Pei; Fadi Dornaika; Jingjun Bi
>
> **摘要:** Existing Multi-view Clustering (MVC) methods based on subspace learning focus on consensus representation learning while neglecting the inherent topological structure of data. Despite the integration of Graph Neural Networks (GNNs) into MVC, their input graph structures remain susceptible to noise interference. Methods based on Multi-view Graph Refinement (MGRC) also have limitations such as insufficient consideration of cross-view consistency, difficulty in handling hard-to-distinguish samples in the feature space, and disjointed optimization processes caused by graph construction algorithms. To address these issues, a Multi-View Clustering method via a Fusion-Consensus Graph Convolutional Network (MCFCN) is proposed. The network learns the consensus graph of multi-view data in an end-to-end manner and learns effective consensus representations through a view feature fusion model and a Unified Graph Structure Adapter (UGA). It designs Similarity Matrix Alignment Loss (SMAL) and Feature Representation Alignment Loss (FRAL). With the guidance of consensus, it optimizes view-specific graphs, preserves cross-view topological consistency, promotes the construction of intra-class edges, and realizes effective consensus representation learning with the help of GCN to improve clustering performance. MCFCN demonstrates state-of-the-art performance on eight multi-view benchmark datasets, and its effectiveness is verified by extensive qualitative and quantitative implementations. The code will be provided at https://github.com/texttao/MCFCN.
>
---
#### [new 111] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment
- **分类: cs.CV; cs.CL**

- **简介: HiMo-CLIP面向视觉-语言对齐任务，解决CLIP忽略文本语义层次与单调性的问题，提出HiDe模块提取语义成分，结合MoLo损失函数，实现多粒度、有序的跨模态对齐，显著提升长文本检索性能。**

- **链接: [http://arxiv.org/pdf/2511.06653v1](http://arxiv.org/pdf/2511.06653v1)**

> **作者:** Ruijia Wu; Ping Chen; Fei Shen; Shaoan Zhao; Qiang Hui; Huanlin Gao; Ting Lu; Zhaoxiang Liu; Fang Zhao; Kai Wang; Shiguo Lian
>
> **备注:** Accepted by AAAI 2026 as an Oral Presentation (13 pages, 7 figures, 7 tables)
>
> **摘要:** Contrastive vision-language models like CLIP have achieved impressive results in image-text retrieval by aligning image and text representations in a shared embedding space. However, these models often treat text as flat sequences, limiting their ability to handle complex, compositional, and long-form descriptions. In particular, they fail to capture two essential properties of language: semantic hierarchy, which reflects the multi-level compositional structure of text, and semantic monotonicity, where richer descriptions should result in stronger alignment with visual content.To address these limitations, we propose HiMo-CLIP, a representation-level framework that enhances CLIP-style models without modifying the encoder architecture. HiMo-CLIP introduces two key components: a hierarchical decomposition (HiDe) module that extracts latent semantic components from long-form text via in-batch PCA, enabling flexible, batch-aware alignment across different semantic granularities, and a monotonicity-aware contrastive loss (MoLo) that jointly aligns global and component-level representations, encouraging the model to internalize semantic ordering and alignment strength as a function of textual completeness.These components work in concert to produce structured, cognitively-aligned cross-modal representations. Experiments on multiple image-text retrieval benchmarks show that HiMo-CLIP consistently outperforms strong baselines, particularly under long or compositional descriptions. The code is available at https://github.com/UnicomAI/HiMo-CLIP.
>
---
#### [new 112] Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到3D生成中的过饱和与过平滑问题，提出TraCe框架，将生成过程建模为Schrödinger Bridge最优传输轨迹，通过LoRA训练轨迹得分动态，实现低CFG下的高质量3D生成。**

- **链接: [http://arxiv.org/pdf/2511.05609v1](http://arxiv.org/pdf/2511.05609v1)**

> **作者:** Ziying Li; Xuequan Lu; Xinkui Zhao; Guanjie Cheng; Shuiguang Deng; Jianwei Yin
>
> **备注:** NeurIPS 2025; https://github.com/emmaleee789/TraCe.git
>
> **摘要:** Recent advancements in optimization-based text-to-3D generation heavily rely on distilling knowledge from pre-trained text-to-image diffusion models using techniques like Score Distillation Sampling (SDS), which often introduce artifacts such as over-saturation and over-smoothing into the generated 3D assets. In this paper, we address this essential problem by formulating the generation process as learning an optimal, direct transport trajectory between the distribution of the current rendering and the desired target distribution, thereby enabling high-quality generation with smaller Classifier-free Guidance (CFG) values. At first, we theoretically establish SDS as a simplified instance of the Schr\"odinger Bridge framework. We prove that SDS employs the reverse process of an Schr\"odinger Bridge, which, under specific conditions (e.g., a Gaussian noise as one end), collapses to SDS's score function of the pre-trained diffusion model. Based upon this, we introduce Trajectory-Centric Distillation (TraCe), a novel text-to-3D generation framework, which reformulates the mathematically trackable framework of Schr\"odinger Bridge to explicitly construct a diffusion bridge from the current rendering to its text-conditioned, denoised target, and trains a LoRA-adapted model on this trajectory's score dynamics for robust 3D optimization. Comprehensive experiments demonstrate that TraCe consistently achieves superior quality and fidelity to state-of-the-art techniques.
>
---
#### [new 113] DiffSwap++: 3D Latent-Controlled Diffusion for Identity-Preserving Face Swapping
- **分类: cs.CV**

- **简介: DiffSwap++提出一种基于扩散模型的面部交换方法，利用3D面部潜变量增强身份保持与几何一致性，解决现有方法在复杂姿态下身份失真和细节伪影问题。**

- **链接: [http://arxiv.org/pdf/2511.05575v1](http://arxiv.org/pdf/2511.05575v1)**

> **作者:** Weston Bondurant; Arkaprava Sinha; Hieu Le; Srijan Das; Stephanie Schuckers
>
> **摘要:** Diffusion-based approaches have recently achieved strong results in face swapping, offering improved visual quality over traditional GAN-based methods. However, even state-of-the-art models often suffer from fine-grained artifacts and poor identity preservation, particularly under challenging poses and expressions. A key limitation of existing approaches is their failure to meaningfully leverage 3D facial structure, which is crucial for disentangling identity from pose and expression. In this work, we propose DiffSwap++, a novel diffusion-based face-swapping pipeline that incorporates 3D facial latent features during training. By guiding the generation process with 3D-aware representations, our method enhances geometric consistency and improves the disentanglement of facial identity from appearance attributes. We further design a diffusion architecture that conditions the denoising process on both identity embeddings and facial landmarks, enabling high-fidelity and identity-preserving face swaps. Extensive experiments on CelebA, FFHQ, and CelebV-Text demonstrate that DiffSwap++ outperforms prior methods in preserving source identity while maintaining target pose and expression. Additionally, we introduce a biometric-style evaluation and conduct a user study to further validate the realism and effectiveness of our approach. Code will be made publicly available at https://github.com/WestonBond/DiffSwapPP
>
---
#### [new 114] Point Cloud Segmentation of Integrated Circuits Package Substrates Surface Defects Using Causal Inference: Dataset Construction and Methodology
- **分类: cs.CV**

- **简介: 该论文面向集成电路陶瓷封装基板表面缺陷的3D点云分割任务，构建了高精度公开数据集CPS3D-Seg，并提出基于因果推断的CINet方法，有效消除点云干扰因素，显著提升分割精度。**

- **链接: [http://arxiv.org/pdf/2511.05853v1](http://arxiv.org/pdf/2511.05853v1)**

> **作者:** Bingyang Guo; Qiang Zuo; Ruiyun Yu
>
> **摘要:** The effective segmentation of 3D data is crucial for a wide range of industrial applications, especially for detecting subtle defects in the field of integrated circuits (IC). Ceramic package substrates (CPS), as an important electronic material, are essential in IC packaging owing to their superior physical and chemical properties. However, the complex structure and minor defects of CPS, along with the absence of a publically available dataset, significantly hinder the development of CPS surface defect detection. In this study, we construct a high-quality point cloud dataset for 3D segmentation of surface defects in CPS, i.e., CPS3D-Seg, which has the best point resolution and precision compared to existing 3D industrial datasets. CPS3D-Seg consists of 1300 point cloud samples under 20 product categories, and each sample provides accurate point-level annotations. Meanwhile, we conduct a comprehensive benchmark based on SOTA point cloud segmentation algorithms to validate the effectiveness of CPS3D-Seg. Additionally, we propose a novel 3D segmentation method based on causal inference (CINet), which quantifies potential confounders in point clouds through Structural Refine (SR) and Quality Assessment (QA) Modules. Extensive experiments demonstrate that CINet significantly outperforms existing algorithms in both mIoU and accuracy.
>
---
#### [new 115] Gaussian-Augmented Physics Simulation and System Identification with Complex Colliders
- **分类: cs.CV**

- **简介: 该论文提出AS-DiffMPM，一种可微分材料点法框架，解决复杂非平面碰撞场景下的物理系统辨识问题，通过可微碰撞处理实现几何、材质与物理参数的端到端优化。**

- **链接: [http://arxiv.org/pdf/2511.06846v1](http://arxiv.org/pdf/2511.06846v1)**

> **作者:** Federico Vasile; Ri-Zhao Qiu; Lorenzo Natale; Xiaolong Wang
>
> **备注:** Accepted to NeurIPS 2025. Project website: https://as-diffmpm.github.io/
>
> **摘要:** System identification involving the geometry, appearance, and physical properties from video observations is a challenging task with applications in robotics and graphics. Recent approaches have relied on fully differentiable Material Point Method (MPM) and rendering for simultaneous optimization of these properties. However, they are limited to simplified object-environment interactions with planar colliders and fail in more challenging scenarios where objects collide with non-planar surfaces. We propose AS-DiffMPM, a differentiable MPM framework that enables physical property estimation with arbitrarily shaped colliders. Our approach extends existing methods by incorporating a differentiable collision handling mechanism, allowing the target object to interact with complex rigid bodies while maintaining end-to-end optimization. We show AS-DiffMPM can be easily interfaced with various novel view synthesis methods as a framework for system identification from visual observations.
>
---
#### [new 116] Zooming into Comics: Region-Aware RL Improves Fine-Grained Comic Understanding in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型在漫画理解中的不足，构建首个细粒度漫画基准AI4VA-FG，并提出区域感知强化学习（RARL），通过动态聚焦漫画区域提升模型对角色、叙事和布局的理解能力。**

- **链接: [http://arxiv.org/pdf/2511.06490v1](http://arxiv.org/pdf/2511.06490v1)**

> **作者:** Yule Chen; Yufan Ren; Sabine Süsstrunk
>
> **摘要:** Complex visual narratives, such as comics, present a significant challenge to Vision-Language Models (VLMs). Despite excelling on natural images, VLMs often struggle with stylized line art, onomatopoeia, and densely packed multi-panel layouts. To address this gap, we introduce AI4VA-FG, the first fine-grained and comprehensive benchmark for VLM-based comic understanding. It spans tasks from foundational recognition and detection to high-level character reasoning and narrative construction, supported by dense annotations for characters, poses, and depth. Beyond that, we evaluate state-of-the-art proprietary models, including GPT-4o and Gemini-2.5, and open-source models such as Qwen2.5-VL, revealing substantial performance deficits across core tasks of our benchmarks and underscoring that comic understanding remains an unsolved challenge. To enhance VLMs' capabilities in this domain, we systematically investigate post-training strategies, including supervised fine-tuning on solutions (SFT-S), supervised fine-tuning on reasoning trajectories (SFT-R), and reinforcement learning (RL). Beyond that, inspired by the emerging "Thinking with Images" paradigm, we propose Region-Aware Reinforcement Learning (RARL) for VLMs, which trains models to dynamically attend to relevant regions through zoom-in operations. We observe that when applied to the Qwen2.5-VL model, RL and RARL yield significant gains in low-level entity recognition and high-level storyline ordering, paving the way for more accurate and efficient VLM applications in the comics domain.
>
---
#### [new 117] An Artificial Intelligence-based Assistant for the Visually Impaired
- **分类: cs.CV; cs.CY; cs.HC**

- **简介: 该论文提出AI辅助系统AIDEN，面向视觉障碍者，解决其物体识别、文本阅读与环境导航难题。基于YOLO与多模态大模型，实现环境感知与交互，提升生活自主性，经用户反馈验证有效性。**

- **链接: [http://arxiv.org/pdf/2511.06080v1](http://arxiv.org/pdf/2511.06080v1)**

> **作者:** Luis Marquez-Carpintero; Francisco Gomez-Donoso; Zuria Bauer; Bessie Dominguez-Dager; Alvaro Belmonte-Baeza; Mónica Pina-Navarro; Francisco Morillas-Espejo; Felix Escalona; Miguel Cazorla
>
> **摘要:** This paper describes an artificial intelligence-based assistant application, AIDEN, developed during 2023 and 2024, aimed at improving the quality of life for visually impaired individuals. Visually impaired individuals face challenges in identifying objects, reading text, and navigating unfamiliar environments, which can limit their independence and reduce their quality of life. Although solutions such as Braille, audio books, and screen readers exist, they may not be effective in all situations. This application leverages state-of-the-art machine learning algorithms to identify and describe objects, read text, and answer questions about the environment. Specifically, it uses You Only Look Once architectures and a Large Language and Vision Assistant. The system incorporates several methods to facilitate the user's interaction with the system and access to textual and visual information in an appropriate manner. AIDEN aims to enhance user autonomy and access to information, contributing to an improved perception of daily usability, as supported by user feedback.
>
---
#### [new 118] How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究文本到图像生成中的语义绑定偏差问题，提出偏差契合分数与无训练的上下文偏差控制框架，通过词元解耦提升复合提示的去偏效果，揭示现有方法在语义关联场景中的局限性。**

- **链接: [http://arxiv.org/pdf/2511.07091v1](http://arxiv.org/pdf/2511.07091v1)**

> **作者:** Jeng-Lin Li; Ming-Ching Chang; Wei-Chao Chen
>
> **备注:** Accepted for publication at the Alignment Track of The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Text-to-image generative models often exhibit bias related to sensitive attributes. However, current research tends to focus narrowly on single-object prompts with limited contextual diversity. In reality, each object or attribute within a prompt can contribute to bias. For example, the prompt "an assistant wearing a pink hat" may reflect female-inclined biases associated with a pink hat. The neglected joint effects of the semantic binding in the prompts cause significant failures in current debiasing approaches. This work initiates a preliminary investigation on how bias manifests under semantic binding, where contextual associations between objects and attributes influence generative outcomes. We demonstrate that the underlying bias distribution can be amplified based on these associations. Therefore, we introduce a bias adherence score that quantifies how specific object-attribute bindings activate bias. To delve deeper, we develop a training-free context-bias control framework to explore how token decoupling can facilitate the debiasing of semantic bindings. This framework achieves over 10% debiasing improvement in compositional generation tasks. Our analysis of bias scores across various attribute-object bindings and token decorrelation highlights a fundamental challenge: reducing bias without disrupting essential semantic relationships. These findings expose critical limitations in current debiasing approaches when applied to semantically bound contexts, underscoring the need to reassess prevailing bias mitigation strategies.
>
---
#### [new 119] NOVO: Bridging LLaVA and SAM with Visual-only Prompts for Reasoning Segmentation
- **分类: cs.CV**

- **简介: NOVO提出一种视觉仅提示框架，无需文本输入，将LLaVA的输出转化为SAM兼容的掩码与点提示，实现推理分割。无需训练的后处理模块提升边界质量，并构建新基准RISeg评估性能。**

- **链接: [http://arxiv.org/pdf/2511.06651v1](http://arxiv.org/pdf/2511.06651v1)**

> **作者:** Kyung-Yoon Yoon; Yeong-Jun Cho
>
> **摘要:** In this study, we propose NOVO (NO text, Visual-Only prompts), a novel framework that bridges vision-language models (VLMs) and segmentation models through visual-only prompts. Unlike prior approaches that feed text-derived SEG token embeddings into segmentation models, NOVO instead generates a coarse mask and point prompts from the VLM output. These visual prompts are compatible with the Segment Anything Model (SAM), preserving alignment with its pretrained capabilities. To further enhance boundary quality and enable instance-level segmentation, we introduce a training-free refinement module that reduces visual artifacts and improves the quality of segmentation masks. We also present RISeg, a new benchmark comprising 918 images, 2,533 instance-level masks, and diverse reasoning queries to evaluate this task. Experiments demonstrate that NOVO achieves state-of-the-art performance across multiple metrics and model sizes, demonstrating its effectiveness and scalability in reasoning segmentation.
>
---
#### [new 120] Diagnose Like A REAL Pathologist: An Uncertainty-Focused Approach for Trustworthy Multi-Resolution Multiple Instance Learning
- **分类: cs.CV**

- **简介: 该论文提出UFC-MIL，面向病理诊断的多分辨率多实例学习任务，解决模型缺乏可信校准的问题，通过patch级不确定性建模与注意力聚合，实现高精度且校准良好的诊断预测，更贴近病理学家决策行为。**

- **链接: [http://arxiv.org/pdf/2511.06433v1](http://arxiv.org/pdf/2511.06433v1)**

> **作者:** Sungrae Hong; Sol Lee; Jisu Shin; Mun Yong Yi
>
> **备注:** Accepted by IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** With the increasing demand for histopathological specimen examination and diagnostic reporting, Multiple Instance Learning (MIL) has received heightened research focus as a viable solution for AI-centric diagnostic aid. Recently, to improve its performance and make it work more like a pathologist, several MIL approaches based on the use of multiple-resolution images have been proposed, delivering often higher performance than those that use single-resolution images. Despite impressive recent developments of multiple-resolution MIL, previous approaches only focus on improving performance, thereby lacking research on well-calibrated MIL that clinical experts can rely on for trustworthy diagnostic results. In this study, we propose Uncertainty-Focused Calibrated MIL (UFC-MIL), which more closely mimics the pathologists' examination behaviors while providing calibrated diagnostic predictions, using multiple images with different resolutions. UFC-MIL includes a novel patch-wise loss that learns the latent patterns of instances and expresses their uncertainty for classification. Also, the attention-based architecture with a neighbor patch aggregation module collects features for the classifier. In addition, aggregated predictions are calibrated through patch-level uncertainty without requiring multiple iterative inferences, which is a key practical advantage. Against challenging public datasets, UFC-MIL shows superior performance in model calibration while achieving classification accuracy comparable to that of state-of-the-art methods.
>
---
#### [new 121] PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory
- **分类: cs.CV; cs.RO**

- **简介: PanoNav提出一种无地图、仅用RGB的零样本目标导航框架，通过全景场景解析与动态记忆机制，提升多模态大模型的空间理解与长期决策能力，避免局部死锁，显著提升导航成功率。**

- **链接: [http://arxiv.org/pdf/2511.06840v1](http://arxiv.org/pdf/2511.06840v1)**

> **作者:** Qunchao Jin; Yilin Wu; Changhao Chen
>
> **备注:** Accepted as a poster in AAAI 2026
>
> **摘要:** Zero-shot object navigation (ZSON) in unseen environments remains a challenging problem for household robots, requiring strong perceptual understanding and decision-making capabilities. While recent methods leverage metric maps and Large Language Models (LLMs), they often depend on depth sensors or prebuilt maps, limiting the spatial reasoning ability of Multimodal Large Language Models (MLLMs). Mapless ZSON approaches have emerged to address this, but they typically make short-sighted decisions, leading to local deadlocks due to a lack of historical context. We propose PanoNav, a fully RGB-only, mapless ZSON framework that integrates a Panoramic Scene Parsing module to unlock the spatial parsing potential of MLLMs from panoramic RGB inputs, and a Memory-guided Decision-Making mechanism enhanced by a Dynamic Bounded Memory Queue to incorporate exploration history and avoid local deadlocks. Experiments on the public navigation benchmark show that PanoNav significantly outperforms representative baselines in both SR and SPL metrics.
>
---
#### [new 122] Learning-Based Vision Systems for Semi-Autonomous Forklift Operation in Industrial Warehouse Environments
- **分类: cs.CV**

- **简介: 该论文针对工业仓库中叉车半自动操作需求，提出基于单目视觉的托盘与孔洞检测与映射系统，采用优化的YOLOv8/v11模型实现高精度感知，构建可低成本部署的视觉模块，提升仓储自动化安全性与经济性。**

- **链接: [http://arxiv.org/pdf/2511.06295v1](http://arxiv.org/pdf/2511.06295v1)**

> **作者:** Vamshika Sutar; Mahek Maheshwari; Archak Mittal
>
> **摘要:** The automation of material handling in warehouses increasingly relies on robust, low cost perception systems for forklifts and Automated Guided Vehicles (AGVs). This work presents a vision based framework for pallet and pallet hole detection and mapping using a single standard camera. We utilized YOLOv8 and YOLOv11 architectures, enhanced through Optuna driven hyperparameter optimization and spatial post processing. An innovative pallet hole mapping module converts the detections into actionable spatial representations, enabling accurate pallet and pallet hole association for forklift operation. Experiments on a custom dataset augmented with real warehouse imagery show that YOLOv8 achieves high pallet and pallet hole detection accuracy, while YOLOv11, particularly under optimized configurations, offers superior precision and stable convergence. The results demonstrate the feasibility of a cost effective, retrofittable visual perception module for forklifts. This study proposes a scalable approach to advancing warehouse automation, promoting safer, economical, and intelligent logistics operations.
>
---
#### [new 123] RelightMaster: Precise Video Relighting with Multi-plane Light Images
- **分类: cs.CV**

- **简介: RelightMaster提出首个精确视频重光照框架，解决文本控制光照不足与数据稀缺问题。构建RelightVideo数据集，创新Multi-plane Light Image（MPLI）表征光照，通过Light Image Adapter微调视频扩散模型，实现内容一致、物理合理的动态重光照。**

- **链接: [http://arxiv.org/pdf/2511.06271v1](http://arxiv.org/pdf/2511.06271v1)**

> **作者:** Weikang Bian; Xiaoyu Shi; Zhaoyang Huang; Jianhong Bai; Qinghe Wang; Xintao Wang; Pengfei Wan; Kun Gai; Hongsheng Li
>
> **备注:** Project Page: https://wkbian.github.io/Projects/RelightMaster/
>
> **摘要:** Recent advances in diffusion models enable high-quality video generation and editing, but precise relighting with consistent video contents, which is critical for shaping scene atmosphere and viewer attention, remains unexplored. Mainstream text-to-video (T2V) models lack fine-grained lighting control due to text's inherent limitation in describing lighting details and insufficient pre-training on lighting-related prompts. Additionally, constructing high-quality relighting training data is challenging, as real-world controllable lighting data is scarce. To address these issues, we propose RelightMaster, a novel framework for accurate and controllable video relighting. First, we build RelightVideo, the first dataset with identical dynamic content under varying precise lighting conditions based on the Unreal Engine. Then, we introduce Multi-plane Light Image (MPLI), a novel visual prompt inspired by Multi-Plane Image (MPI). MPLI models lighting via K depth-aligned planes, representing 3D light source positions, intensities, and colors while supporting multi-source scenarios and generalizing to unseen light setups. Third, we design a Light Image Adapter that seamlessly injects MPLI into pre-trained Video Diffusion Transformers (DiT): it compresses MPLI via a pre-trained Video VAE and injects latent light features into DiT blocks, leveraging the base model's generative prior without catastrophic forgetting. Experiments show that RelightMaster generates physically plausible lighting and shadows and preserves original scene content. Demos are available at https://wkbian.github.io/Projects/RelightMaster/.
>
---
#### [new 124] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era
- **分类: cs.CV**

- **简介: 该论文面向视觉地点识别任务，提出在Transformer架构中无需显式聚合器，仅通过引入可学习聚合标记并利用自注意力机制隐式聚合特征，即可生成鲁棒全局描述符，显著提升性能与效率。**

- **链接: [http://arxiv.org/pdf/2511.06024v1](http://arxiv.org/pdf/2511.06024v1)**

> **作者:** Feng Lu; Tong Jin; Canming Ye; Yunpeng Liu; Xiangyuan Lan; Chun Yuan
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Visual place recognition (VPR) is typically regarded as a specific image retrieval task, whose core lies in representing images as global descriptors. Over the past decade, dominant VPR methods (e.g., NetVLAD) have followed a paradigm that first extracts the patch features/tokens of the input image using a backbone, and then aggregates these patch features into a global descriptor via an aggregator. This backbone-plus-aggregator paradigm has achieved overwhelming dominance in the CNN era and remains widely used in transformer-based models. In this paper, however, we argue that a dedicated aggregator is not necessary in the transformer era, that is, we can obtain robust global descriptors only with the backbone. Specifically, we introduce some learnable aggregation tokens, which are prepended to the patch tokens before a particular transformer block. All these tokens will be jointly processed and interact globally via the intrinsic self-attention mechanism, implicitly aggregating useful information within the patch tokens to the aggregation tokens. Finally, we only take these aggregation tokens from the last output tokens and concatenate them as the global representation. Although implicit aggregation can provide robust global descriptors in an extremely simple manner, where and how to insert additional tokens, as well as the initialization of tokens, remains an open issue worthy of further exploration. To this end, we also propose the optimal token insertion strategy and token initialization method derived from empirical studies. Experimental results show that our method outperforms state-of-the-art methods on several VPR datasets with higher efficiency and ranks 1st on the MSLS challenge leaderboard. The code is available at https://github.com/lu-feng/image.
>
---
#### [new 125] MUGSQA: Novel Multi-Uncertainty-Based Gaussian Splatting Quality Assessment Method, Dataset, and Benchmarks
- **分类: cs.CV**

- **简介: 该论文提出MUGSQA，面向高斯溅射（GS）3D重建的感知质量评估任务，解决现有方法缺乏多不确定性评估的问题，构建了含多源不确定性的数据集与双基准，支持模型鲁棒性与评估指标的系统评测。**

- **链接: [http://arxiv.org/pdf/2511.06830v1](http://arxiv.org/pdf/2511.06830v1)**

> **作者:** Tianang Chen; Jian Jin; Shilv Cai; Zhuangzi Li; Weisi Lin
>
> **摘要:** Gaussian Splatting (GS) has recently emerged as a promising technique for 3D object reconstruction, delivering high-quality rendering results with significantly improved reconstruction speed. As variants continue to appear, assessing the perceptual quality of 3D objects reconstructed with different GS-based methods remains an open challenge. To address this issue, we first propose a unified multi-distance subjective quality assessment method that closely mimics human viewing behavior for objects reconstructed with GS-based methods in actual applications, thereby better collecting perceptual experiences. Based on it, we also construct a novel GS quality assessment dataset named MUGSQA, which is constructed considering multiple uncertainties of the input data. These uncertainties include the quantity and resolution of input views, the view distance, and the accuracy of the initial point cloud. Moreover, we construct two benchmarks: one to evaluate the robustness of various GS-based reconstruction methods under multiple uncertainties, and the other to evaluate the performance of existing quality assessment metrics. Our dataset and benchmark code will be released soon.
>
---
#### [new 126] Personalized Image Editing in Text-to-Image Diffusion Models via Collaborative Direct Preference Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出C-DPO方法，解决文本到图像扩散模型缺乏个性化编辑的问题，通过构建用户偏好图网络，共享相似用户偏好，优化编辑结果与个体审美的一致性。**

- **链接: [http://arxiv.org/pdf/2511.05616v1](http://arxiv.org/pdf/2511.05616v1)**

> **作者:** Connor Dunlop; Matthew Zheng; Kavana Venkatesh; Pinar Yanardag
>
> **备注:** Published at NeurIPS'25 Main Conference
>
> **摘要:** Text-to-image (T2I) diffusion models have made remarkable strides in generating and editing high-fidelity images from text. Yet, these models remain fundamentally generic, failing to adapt to the nuanced aesthetic preferences of individual users. In this work, we present the first framework for personalized image editing in diffusion models, introducing Collaborative Direct Preference Optimization (C-DPO), a novel method that aligns image edits with user-specific preferences while leveraging collaborative signals from like-minded individuals. Our approach encodes each user as a node in a dynamic preference graph and learns embeddings via a lightweight graph neural network, enabling information sharing across users with overlapping visual tastes. We enhance a diffusion model's editing capabilities by integrating these personalized embeddings into a novel DPO objective, which jointly optimizes for individual alignment and neighborhood coherence. Comprehensive experiments, including user studies and quantitative benchmarks, demonstrate that our method consistently outperforms baselines in generating edits that are aligned with user preferences.
>
---
#### [new 127] Temporal-Guided Visual Foundation Models for Event-Based Vision
- **分类: cs.CV**

- **简介: 该论文提出Temporal-Guided VFM（TGVFM），将图像预训练视觉基础模型用于事件视觉任务，通过引入时序注意力与特征融合机制，解决异步事件流建模难题，在分割、深度估计和检测任务上显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06238v1](http://arxiv.org/pdf/2511.06238v1)**

> **作者:** Ruihao Xia; Junhong Cai; Luziwei Leng; Liuyi Wang; Chengju Liu; Ran Cheng; Yang Tang; Pan Zhou
>
> **摘要:** Event cameras offer unique advantages for vision tasks in challenging environments, yet processing asynchronous event streams remains an open challenge. While existing methods rely on specialized architectures or resource-intensive training, the potential of leveraging modern Visual Foundation Models (VFMs) pretrained on image data remains under-explored for event-based vision. To address this, we propose Temporal-Guided VFM (TGVFM), a novel framework that integrates VFMs with our temporal context fusion block seamlessly to bridge this gap. Our temporal block introduces three key components: (1) Long-Range Temporal Attention to model global temporal dependencies, (2) Dual Spatiotemporal Attention for multi-scale frame correlation, and (3) Deep Feature Guidance Mechanism to fuse semantic-temporal features. By retraining event-to-video models on real-world data and leveraging transformer-based VFMs, TGVFM preserves spatiotemporal dynamics while harnessing pretrained representations. Experiments demonstrate SoTA performance across semantic segmentation, depth estimation, and object detection, with improvements of 16%, 21%, and 16% over existing methods, respectively. Overall, this work unlocks the cross-modality potential of image-based VFMs for event-based vision with temporal reasoning. Code is available at https://github.com/XiaRho/TGVFM.
>
---
#### [new 128] CAST-LUT: Tokenizer-Guided HSV Look-Up Tables for Purple Flare Removal
- **分类: cs.CV**

- **简介: 该论文针对图像紫斑flare问题，提出CAST-LUT方法，通过染色感知分词器提取HSV通道语义令牌，动态生成独立查表校正曲线，首次构建大规模紫斑数据集并设计专用评估指标，显著提升校正效果。**

- **链接: [http://arxiv.org/pdf/2511.06764v1](http://arxiv.org/pdf/2511.06764v1)**

> **作者:** Pu Wang; Shuning Sun; Jialang Lu; Chen Wu; Zhihua Zhang; Youshan Zhang; Chenggang Shan; Dianjie Lu; Guijuan Zhang; Zhuoran Zheng
>
> **摘要:** Purple flare, a diffuse chromatic aberration artifact commonly found around highlight areas, severely degrades the tone transition and color of the image. Existing traditional methods are based on hand-crafted features, which lack flexibility and rely entirely on fixed priors, while the scarcity of paired training data critically hampers deep learning. To address this issue, we propose a novel network built upon decoupled HSV Look-Up Tables (LUTs). The method aims to simplify color correction by adjusting the Hue (H), Saturation (S), and Value (V) components independently. This approach resolves the inherent color coupling problems in traditional methods. Our model adopts a two-stage architecture: First, a Chroma-Aware Spectral Tokenizer (CAST) converts the input image from RGB space to HSV space and independently encodes the Hue (H) and Value (V) channels into a set of semantic tokens describing the Purple flare status; second, the HSV-LUT module takes these tokens as input and dynamically generates independent correction curves (1D-LUTs) for the three channels H, S, and V. To effectively train and validate our model, we built the first large-scale purple flare dataset with diverse scenes. We also proposed new metrics and a loss function specifically designed for this task. Extensive experiments demonstrate that our model not only significantly outperforms existing methods in visual effects but also achieves state-of-the-art performance on all quantitative metrics.
>
---
#### [new 129] Real-Time Bundle Adjustment for Ultra-High-Resolution UAV Imagery Using Adaptive Patch-Based Feature Tracking
- **分类: cs.CV; math.OC**

- **简介: 该论文针对超高分辨率无人机影像实时处理难题，提出一种自适应分块特征跟踪的实时光束法平差方法，无需下采样，利用GNSS/IMU与DSM实现局部优化，在无GPU下2秒内完成高精度三维重建。**

- **链接: [http://arxiv.org/pdf/2511.06152v1](http://arxiv.org/pdf/2511.06152v1)**

> **作者:** Selim Ahmet Iz; Francesco Nex; Norman Kerle; Henry Meissner; Ralf Berger
>
> **摘要:** Real-time processing of UAV imagery is crucial for applications requiring urgent geospatial information, such as disaster response, where rapid decision-making and accurate spatial data are essential. However, processing high-resolution imagery in real time presents significant challenges due to the computational demands of feature extraction, matching, and bundle adjustment (BA). Conventional BA methods either downsample images, sacrificing important details, or require extensive processing time, making them unsuitable for time-critical missions. To overcome these limitations, we propose a novel real-time BA framework that operates directly on fullresolution UAV imagery without downsampling. Our lightweight, onboard-compatible approach divides each image into user-defined patches (e.g., NxN grids, default 150x150 pixels) and dynamically tracks them across frames using UAV GNSS/IMU data and a coarse, globally available digital surface model (DSM). This ensures spatial consistency for robust feature extraction and matching between patches. Overlapping relationships between images are determined in real time using UAV navigation system, enabling the rapid selection of relevant neighbouring images for localized BA. By limiting optimization to a sliding cluster of overlapping images, including those from adjacent flight strips, the method achieves real-time performance while preserving the accuracy of global BA. The proposed algorithm is designed for seamless integration into the DLR Modular Aerial Camera System (MACS), supporting largearea mapping in real time for disaster response, infrastructure monitoring, and coastal protection. Validation on MACS datasets with 50MP images demonstrates that the method maintains precise camera orientations and high-fidelity mapping across multiple strips, running full bundle adjustment in under 2 seconds without GPU acceleration.
>
---
#### [new 130] Hybrid CNN-ViT Framework for Motion-Blurred Scene Text Restoration
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种CNN-ViT混合框架，用于恢复运动模糊的场景文本图像，解决传统方法难以处理空间变化模糊与长程依赖问题。通过局部特征提取与全局注意力结合，在TextOCR数据集上实现高效高质重建。**

- **链接: [http://arxiv.org/pdf/2511.06087v1](http://arxiv.org/pdf/2511.06087v1)**

> **作者:** Umar Rashid; Muhammad Arslan Arshad; Ghulam Ahmad; Muhammad Zeeshan Anjum; Rizwan Khan; Muhammad Akmal
>
> **摘要:** Motion blur in scene text images severely impairs readability and hinders the reliability of computer vision tasks, including autonomous driving, document digitization, and visual information retrieval. Conventional deblurring approaches are often inadequate in handling spatially varying blur and typically fall short in modeling the long-range dependencies necessary for restoring textual clarity. To overcome these limitations, we introduce a hybrid deep learning framework that combines convolutional neural networks (CNNs) with vision transformers (ViTs), thereby leveraging both local feature extraction and global contextual reasoning. The architecture employs a CNN-based encoder-decoder to preserve structural details, while a transformer module enhances global awareness through self-attention. Training is conducted on a curated dataset derived from TextOCR, where sharp scene-text samples are paired with synthetically blurred versions generated using realistic motion-blur kernels of multiple sizes and orientations. Model optimization is guided by a composite loss that incorporates mean absolute error (MAE), squared error (MSE), perceptual similarity, and structural similarity (SSIM). Quantitative eval- uations show that the proposed method attains 32.20 dB in PSNR and 0.934 in SSIM, while remaining lightweight with 2.83 million parameters and an average inference time of 61 ms. These results highlight the effectiveness and computational efficiency of the CNN-ViT hybrid design, establishing its practicality for real-world motion-blurred scene-text restoration.
>
---
#### [new 131] LLM-Driven Completeness and Consistency Evaluation for Cultural Heritage Data Augmentation in Cross-Modal Retrieval
- **分类: cs.CV; cs.CY**

- **简介: 该论文针对文化遗产跨模态检索中文本描述不完整与不一致的问题，提出C³框架，利用LLM生成增强描述，并通过视觉引导的完备性评估与马尔可夫决策过程约束一致性，显著提升检索性能。**

- **链接: [http://arxiv.org/pdf/2511.06268v1](http://arxiv.org/pdf/2511.06268v1)**

> **作者:** Jian Zhang; Junyi Guo; Junyi Yuan; Huanda Lu; Yanlin Zhou; Fangyu Wu; Qiufeng Wang; Dongming Lu
>
> **摘要:** Cross-modal retrieval is essential for interpreting cultural heritage data, but its effectiveness is often limited by incomplete or inconsistent textual descriptions, caused by historical data loss and the high cost of expert annotation. While large language models (LLMs) offer a promising solution by enriching textual descriptions, their outputs frequently suffer from hallucinations or miss visually grounded details. To address these challenges, we propose $C^3$, a data augmentation framework that enhances cross-modal retrieval performance by improving the completeness and consistency of LLM-generated descriptions. $C^3$ introduces a completeness evaluation module to assess semantic coverage using both visual cues and language-model outputs. Furthermore, to mitigate factual inconsistencies, we formulate a Markov Decision Process to supervise Chain-of-Thought reasoning, guiding consistency evaluation through adaptive query control. Experiments on the cultural heritage datasets CulTi and TimeTravel, as well as on general benchmarks MSCOCO and Flickr30K, demonstrate that $C^3$ achieves state-of-the-art performance in both fine-tuned and zero-shot settings.
>
---
#### [new 132] AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer
- **分类: cs.CV**

- **简介: AnoStyler提出一种轻量级文本驱动的局部异常生成方法，解决真实异常数据稀缺问题，通过CLIP引导的风格迁移，在单张正常图上生成语义一致、视觉逼真的局部异常图像，提升异常检测性能。**

- **链接: [http://arxiv.org/pdf/2511.06687v1](http://arxiv.org/pdf/2511.06687v1)**

> **作者:** Yulim So; Seokho Kang
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Anomaly generation has been widely explored to address the scarcity of anomaly images in real-world data. However, existing methods typically suffer from at least one of the following limitations, hindering their practical deployment: (1) lack of visual realism in generated anomalies; (2) dependence on large amounts of real images; and (3) use of memory-intensive, heavyweight model architectures. To overcome these limitations, we propose AnoStyler, a lightweight yet effective method that frames zero-shot anomaly generation as text-guided style transfer. Given a single normal image along with its category label and expected defect type, an anomaly mask indicating the localized anomaly regions and two-class text prompts representing the normal and anomaly states are generated using generalizable category-agnostic procedures. A lightweight U-Net model trained with CLIP-based loss functions is used to stylize the normal image into a visually realistic anomaly image, where anomalies are localized by the anomaly mask and semantically aligned with the text prompts. Extensive experiments on the MVTec-AD and VisA datasets show that AnoStyler outperforms existing anomaly generation methods in generating high-quality and diverse anomaly images. Furthermore, using these generated anomalies helps enhance anomaly detection performance.
>
---
#### [new 133] Convolutional Fully-Connected Capsule Network (CFC-CapsNet): A Novel and Fast Capsule Network
- **分类: cs.CV**

- **简介: 该论文提出CFC-CapsNet，一种新型胶囊网络，用于图像分类任务。针对传统CapsNet参数多、速度慢、性能受限的问题，通过引入CFC层生成更高效胶囊，在CIFAR-10等数据集上实现更高精度、更快训练与更少参数。**

- **链接: [http://arxiv.org/pdf/2511.05617v1](http://arxiv.org/pdf/2511.05617v1)**

> **作者:** Pouya Shiri; Amirali Baniasadi
>
> **摘要:** A Capsule Network (CapsNet) is a relatively new classifier and one of the possible successors of Convolutional Neural Networks (CNNs). CapsNet maintains the spatial hierarchies between the features and outperforms CNNs at classifying images including overlapping categories. Even though CapsNet works well on small-scale datasets such as MNIST, it fails to achieve a similar level of performance on more complicated datasets and real applications. In addition, CapsNet is slow compared to CNNs when performing the same task and relies on a higher number of parameters. In this work, we introduce Convolutional Fully-Connected Capsule Network (CFC-CapsNet) to address the shortcomings of CapsNet by creating capsules using a different method. We introduce a new layer (CFC layer) as an alternative solution to creating capsules. CFC-CapsNet produces fewer, yet more powerful capsules resulting in higher network accuracy. Our experiments show that CFC-CapsNet achieves competitive accuracy, faster training and inference and uses less number of parameters on the CIFAR-10, SVHN and Fashion-MNIST datasets compared to conventional CapsNet.
>
---
#### [new 134] Label-Efficient 3D Forest Mapping: Self-Supervised and Transfer Learning for Individual, Structural, and Species Analysis
- **分类: cs.CV**

- **简介: 该论文面向3D森林点云分析，解决标注数据稀缺问题，提出自监督与迁移学习框架，提升单木实例分割、语义分割与物种分类性能，构建统一开源工具链，降低能耗与碳排放。**

- **链接: [http://arxiv.org/pdf/2511.06331v1](http://arxiv.org/pdf/2511.06331v1)**

> **作者:** Aldino Rizaldy; Fabian Ewald Fassnacht; Ahmed Jamal Afifi; Hua Jiang; Richard Gloaguen; Pedram Ghamisi
>
> **摘要:** Detailed structural and species information on individual tree level is increasingly important to support precision forestry, biodiversity conservation, and provide reference data for biomass and carbon mapping. Point clouds from airborne and ground-based laser scanning are currently the most suitable data source to rapidly derive such information at scale. Recent advancements in deep learning improved segmenting and classifying individual trees and identifying semantic tree components. However, deep learning models typically require large amounts of annotated training data which limits further improvement. Producing dense, high-quality annotations for 3D point clouds, especially in complex forests, is labor-intensive and challenging to scale. We explore strategies to reduce dependence on large annotated datasets using self-supervised and transfer learning architectures. Our objective is to improve performance across three tasks: instance segmentation, semantic segmentation, and tree classification using realistic and operational training sets. Our findings indicate that combining self-supervised learning with domain adaptation significantly enhances instance segmentation compared to training from scratch (AP50 +16.98%), self-supervised learning suffices for semantic segmentation (mIoU +1.79%), and hierarchical transfer learning enables accurate classification of unseen species (Jaccard +6.07%). To simplify use and encourage uptake, we integrated the tasks into a unified framework, streamlining the process from raw point clouds to tree delineation, structural analysis, and species classification. Pretrained models reduce energy consumption and carbon emissions by ~21%. This open-source contribution aims to accelerate operational extraction of individual tree information from laser scanning point clouds to support forestry, biodiversity, and carbon mapping.
>
---
#### [new 135] TrueCity: Real and Simulated Urban Data for Cross-Domain 3D Scene Understanding
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出TrueCity，首个同步真实与模拟城市点云的3D语义分割基准，解决合成数据与真实数据间的域差距问题，通过高精度标注数据推动跨域3D场景理解研究。**

- **链接: [http://arxiv.org/pdf/2511.07007v1](http://arxiv.org/pdf/2511.07007v1)**

> **作者:** Duc Nguyen; Yan-Ling Lai; Qilin Zhang; Prabin Gyawali; Benedikt Schwab; Olaf Wysocki; Thomas H. Kolbe
>
> **备注:** The paper accepted for 3DV 2026 (International Conference on 3D Vision 2026)
>
> **摘要:** 3D semantic scene understanding remains a long-standing challenge in the 3D computer vision community. One of the key issues pertains to limited real-world annotated data to facilitate generalizable models. The common practice to tackle this issue is to simulate new data. Although synthetic datasets offer scalability and perfect labels, their designer-crafted scenes fail to capture real-world complexity and sensor noise, resulting in a synthetic-to-real domain gap. Moreover, no benchmark provides synchronized real and simulated point clouds for segmentation-oriented domain shift analysis. We introduce TrueCity, the first urban semantic segmentation benchmark with cm-accurate annotated real-world point clouds, semantic 3D city models, and annotated simulated point clouds representing the same city. TrueCity proposes segmentation classes aligned with international 3D city modeling standards, enabling consistent evaluation of synthetic-to-real gap. Our extensive experiments on common baselines quantify domain shift and highlight strategies for exploiting synthetic data to enhance real-world 3D scene understanding. We are convinced that the TrueCity dataset will foster further development of sim-to-real gap quantification and enable generalizable data-driven models. The data, code, and 3D models are available online: https://tum-gis.github.io/TrueCity/
>
---
#### [new 136] NOAH: Benchmarking Narrative Prior driven Hallucination and Omission in Video Large Language Models
- **分类: cs.CV; I.2.10; I.4.8**

- **简介: 该论文提出NOAH基准，评估视频大语言模型因叙事先验导致的幻觉与遗漏问题，通过构造插入视频片段的可控场景，系统分析模型在captioning与QA任务中的错误模式，揭示其过度依赖叙事连贯性而非视觉证据的偏差。**

- **链接: [http://arxiv.org/pdf/2511.06475v1](http://arxiv.org/pdf/2511.06475v1)**

> **作者:** Kyuho Lee; Euntae Kim; Jinwoo Choi; Buru Chang
>
> **备注:** 18 pages, 9 figures. Preprint
>
> **摘要:** Video large language models (Video LLMs) have recently achieved strong performance on tasks such as captioning, summarization, and question answering. Many models and training methods explicitly encourage continuity across events to enhance narrative coherence. While this improves fluency, it also introduces an inductive bias that prioritizes storyline consistency over strict grounding in visual evidence. We identify this bias, which we call narrative prior, as a key driver of two errors: hallucinations, where non-existent events are introduced or existing ones are misinterpreted, and omissions, where factual events are suppressed because they are misaligned with surrounding context. To systematically evaluate narrative prior-induced errors, we introduce NOAH, a large-scale benchmark that constructs composite videos by inserting clips from other sources into target videos. By varying semantic similarity and insertion position, our benchmark enables controlled and scalable analysis of narrative priors. We design one captioning task with tailored metrics and three QA tasks - Existence, Temporal, and Narrative - yielding more than 60K evaluation samples. Extensive experiments yield three key findings: (i) most Video LLMs exhibit hallucinations and omissions driven by narrative priors, (ii) the patterns of these errors vary across architectures and depend on event similarity and insertion position, and (iii) reliance on narrative priors intensifies under sampling with fewer frames, amplifying errors when event continuity is weak. We establish NOAH as the first standardized evaluation of narrative prior-induced hallucination and omission in Video LLMs, providing a foundation for developing more reliable and trustworthy models. Our benchmark and code are available at https://anonymous550520.github.io/.
>
---
#### [new 137] From Attribution to Action: Jointly ALIGNing Predictions and Explanations
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ALIGN框架，用于联合优化分类器与掩码生成器，解决解释引导学习中低质量监督信号导致性能下降的问题，提升模型预测准确性与解释质量，适用于域泛化任务。**

- **链接: [http://arxiv.org/pdf/2511.06944v1](http://arxiv.org/pdf/2511.06944v1)**

> **作者:** Dongsheng Hong; Chao Chen; Yanhui Chen; Shanshan Lin; Zhihao Chen; Xiangwen Liao
>
> **备注:** Accepted in AAAI 2026
>
> **摘要:** Explanation-guided learning (EGL) has shown promise in aligning model predictions with interpretable reasoning, particularly in computer vision tasks. However, most approaches rely on external annotations or heuristic-based segmentation to supervise model explanations, which can be noisy, imprecise and difficult to scale. In this work, we provide both empirical and theoretical evidence that low-quality supervision signals can degrade model performance rather than improve it. In response, we propose ALIGN, a novel framework that jointly trains a classifier and a masker in an iterative manner. The masker learns to produce soft, task-relevant masks that highlight informative regions, while the classifier is optimized for both prediction accuracy and alignment between its saliency maps and the learned masks. By leveraging high-quality masks as guidance, ALIGN improves both interpretability and generalizability, showing its superiority across various settings. Experiments on the two domain generalization benchmarks, VLCS and Terra Incognita, show that ALIGN consistently outperforms six strong baselines in both in-distribution and out-of-distribution settings. Besides, ALIGN also yields superior explanation quality concerning sufficiency and comprehensiveness, highlighting its effectiveness in producing accurate and interpretable models.
>
---
#### [new 138] V-Shuffle: Zero-Shot Style Transfer via Value Shuffle
- **分类: cs.CV**

- **简介: V-Shuffle提出一种零样本风格迁移方法，通过在扩散模型自注意力层中打乱值特征，抑制风格图语义泄露，结合混合风格正则化提升风格保真度，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06365v1](http://arxiv.org/pdf/2511.06365v1)**

> **作者:** Haojun Tang; Qiwei Lin; Tongda Xu; Lida Huang; Yan Wang
>
> **摘要:** Attention injection-based style transfer has achieved remarkable progress in recent years. However, existing methods often suffer from content leakage, where the undesired semantic content of the style image mistakenly appears in the stylized output. In this paper, we propose V-Shuffle, a zero-shot style transfer method that leverages multiple style images from the same style domain to effectively navigate the trade-off between content preservation and style fidelity. V-Shuffle implicitly disrupts the semantic content of the style images by shuffling the value features within the self-attention layers of the diffusion model, thereby preserving low-level style representations. We further introduce a Hybrid Style Regularization that complements these low-level representations with high-level style textures to enhance style fidelity. Empirical results demonstrate that V-Shuffle achieves excellent performance when utilizing multiple style images. Moreover, when applied to a single style image, V-Shuffle outperforms previous state-of-the-art methods.
>
---
#### [new 139] Certified L2-Norm Robustness of 3D Point Cloud Recognition in the Frequency Domain
- **分类: cs.CV**

- **简介: 该论文针对3D点云分类的对抗鲁棒性问题，提出FreqCert框架，通过频域分析（图傅里叶变换）实现L2范数鲁棒性认证，利用频域相似性采样与投票提升抗结构扰动能力，理论证明并实验验证了其优越的认证准确率。**

- **链接: [http://arxiv.org/pdf/2511.07029v1](http://arxiv.org/pdf/2511.07029v1)**

> **作者:** Liang Zhou; Qiming Wang; Tianze Chen
>
> **备注:** Accepted by AAAI26
>
> **摘要:** 3D point cloud classification is a fundamental task in safety-critical applications such as autonomous driving, robotics, and augmented reality. However, recent studies reveal that point cloud classifiers are vulnerable to structured adversarial perturbations and geometric corruptions, posing risks to their deployment in safety-critical scenarios. Existing certified defenses limit point-wise perturbations but overlook subtle geometric distortions that preserve individual points yet alter the overall structure, potentially leading to misclassification. In this work, we propose FreqCert, a novel certification framework that departs from conventional spatial domain defenses by shifting robustness analysis to the frequency domain, enabling structured certification against global L2-bounded perturbations. FreqCert first transforms the input point cloud via the graph Fourier transform (GFT), then applies structured frequency-aware subsampling to generate multiple sub-point clouds. Each sub-cloud is independently classified by a standard model, and the final prediction is obtained through majority voting, where sub-clouds are constructed based on spectral similarity rather than spatial proximity, making the partitioning more stable under L2 perturbations and better aligned with the object's intrinsic structure. We derive a closed-form lower bound on the certified L2 robustness radius and prove its tightness under minimal and interpretable assumptions, establishing a theoretical foundation for frequency domain certification. Extensive experiments on the ModelNet40 and ScanObjectNN datasets demonstrate that FreqCert consistently achieves higher certified accuracy and empirical accuracy under strong perturbations. Our results suggest that spectral representations provide an effective pathway toward certifiable robustness in 3D point cloud recognition.
>
---
#### [new 140] AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving
- **分类: cs.CV**

- **简介: AdaDrive提出一种自适应慢-快系统，用于语言引导的自动驾驶，解决LLM频繁调用导致的效率低下问题，通过动态激活与连续融合策略，在复杂场景中精准调用LLM，兼顾决策精度与实时性。**

- **链接: [http://arxiv.org/pdf/2511.06253v1](http://arxiv.org/pdf/2511.06253v1)**

> **作者:** Ruifei Zhang; Junlin Xie; Wei Zhang; Weikai Chen; Xiao Tan; Xiang Wan; Guanbin Li
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Effectively integrating Large Language Models (LLMs) into autonomous driving requires a balance between leveraging high-level reasoning and maintaining real-time efficiency. Existing approaches either activate LLMs too frequently, causing excessive computational overhead, or use fixed schedules, failing to adapt to dynamic driving conditions. To address these challenges, we propose AdaDrive, an adaptively collaborative slow-fast framework that optimally determines when and how LLMs contribute to decision-making. (1) When to activate the LLM: AdaDrive employs a novel adaptive activation loss that dynamically determines LLM invocation based on a comparative learning mechanism, ensuring activation only in complex or critical scenarios. (2) How to integrate LLM assistance: Instead of rigid binary activation, AdaDrive introduces an adaptive fusion strategy that modulates a continuous, scaled LLM influence based on scene complexity and prediction confidence, ensuring seamless collaboration with conventional planners. Through these strategies, AdaDrive provides a flexible, context-aware framework that maximizes decision accuracy without compromising real-time performance. Extensive experiments on language-grounded autonomous driving benchmarks demonstrate that AdaDrive state-of-the-art performance in terms of both driving accuracy and computational efficiency. Code is available at https://github.com/ReaFly/AdaDrive.
>
---
#### [new 141] CGCE: Classifier-Guided Concept Erasure in Generative Models
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 论文提出CGCE，一种无需修改模型权重的即插即用框架，通过文本嵌入分类器检测并修正有害提示，实现鲁棒的概念擦除，在保障生成质量的同时抵御对抗攻击，解决安全与性能的权衡难题。**

- **链接: [http://arxiv.org/pdf/2511.05865v1](http://arxiv.org/pdf/2511.05865v1)**

> **作者:** Viet Nguyen; Vishal M. Patel
>
> **备注:** 24 pages, 15 figures
>
> **摘要:** Recent advancements in large-scale generative models have enabled the creation of high-quality images and videos, but have also raised significant safety concerns regarding the generation of unsafe content. To mitigate this, concept erasure methods have been developed to remove undesirable concepts from pre-trained models. However, existing methods remain vulnerable to adversarial attacks that can regenerate the erased content. Moreover, achieving robust erasure often degrades the model's generative quality for safe, unrelated concepts, creating a difficult trade-off between safety and performance. To address this challenge, we introduce Classifier-Guided Concept Erasure (CGCE), an efficient plug-and-play framework that provides robust concept erasure for diverse generative models without altering their original weights. CGCE uses a lightweight classifier operating on text embeddings to first detect and then refine prompts containing undesired concepts. This approach is highly scalable, allowing for multi-concept erasure by aggregating guidance from several classifiers. By modifying only unsafe embeddings at inference time, our method prevents harmful content generation while preserving the model's original quality on benign prompts. Extensive experiments show that CGCE achieves state-of-the-art robustness against a wide range of red-teaming attacks. Our approach also maintains high generative utility, demonstrating a superior balance between safety and performance. We showcase the versatility of CGCE through its successful application to various modern T2I and T2V models, establishing it as a practical and effective solution for safe generative AI.
>
---
#### [new 142] Runtime Safety Monitoring of Deep Neural Networks for Perception: A Survey
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文综述了深度神经网络感知系统运行时安全监测方法，旨在解决DNN在推理中因泛化错误、OOD输入和对抗攻击导致的安全风险，分类梳理了输入、中间表征与输出三类监测技术，分析优劣并指明未来方向。**

- **链接: [http://arxiv.org/pdf/2511.05982v1](http://arxiv.org/pdf/2511.05982v1)**

> **作者:** Albert Schotschneider; Svetlana Pavlitska; J. Marius Zöllner
>
> **备注:** 6 pages, 1 figure, 2 tables, accepted at IEEE SMC 2025 in Vienna, presented on 8th October 2025
>
> **摘要:** Deep neural networks (DNNs) are widely used in perception systems for safety-critical applications, such as autonomous driving and robotics. However, DNNs remain vulnerable to various safety concerns, including generalization errors, out-of-distribution (OOD) inputs, and adversarial attacks, which can lead to hazardous failures. This survey provides a comprehensive overview of runtime safety monitoring approaches, which operate in parallel to DNNs during inference to detect these safety concerns without modifying the DNN itself. We categorize existing methods into three main groups: Monitoring inputs, internal representations, and outputs. We analyze the state-of-the-art for each category, identify strengths and limitations, and map methods to the safety concerns they address. In addition, we highlight open challenges and future research directions.
>
---
#### [new 143] Spatial-Frequency Enhanced Mamba for Multi-Modal Image Fusion
- **分类: cs.CV**

- **简介: 该论文针对多模态图像融合任务，提出SFMFusion框架，解决Mamba模型缺乏空间与频率感知的问题。通过三分支结构耦合重建任务，设计空间-频率增强Mamba块与动态融合块，提升特征提取与融合效果。**

- **链接: [http://arxiv.org/pdf/2511.06593v1](http://arxiv.org/pdf/2511.06593v1)**

> **作者:** Hui Sun; Long Lv; Pingping Zhang; Tongdan Tang; Feng Tian; Weibing Sun; Huchuan Lu
>
> **备注:** This work is accepted by IEEE Transactions on Image Processing. More modifications may be performed
>
> **摘要:** Multi-Modal Image Fusion (MMIF) aims to integrate complementary image information from different modalities to produce informative images. Previous deep learning-based MMIF methods generally adopt Convolutional Neural Networks (CNNs) or Transformers for feature extraction. However, these methods deliver unsatisfactory performances due to the limited receptive field of CNNs and the high computational cost of Transformers. Recently, Mamba has demonstrated a powerful potential for modeling long-range dependencies with linear complexity, providing a promising solution to MMIF. Unfortunately, Mamba lacks full spatial and frequency perceptions, which are very important for MMIF. Moreover, employing Image Reconstruction (IR) as an auxiliary task has been proven beneficial for MMIF. However, a primary challenge is how to leverage IR efficiently and effectively. To address the above issues, we propose a novel framework named Spatial-Frequency Enhanced Mamba Fusion (SFMFusion) for MMIF. More specifically, we first propose a three-branch structure to couple MMIF and IR, which can retain complete contents from source images. Then, we propose the Spatial-Frequency Enhanced Mamba Block (SFMB), which can enhance Mamba in both spatial and frequency domains for comprehensive feature extraction. Finally, we propose the Dynamic Fusion Mamba Block (DFMB), which can be deployed across different branches for dynamic feature fusion. Extensive experiments show that our method achieves better results than most state-of-the-art methods on six MMIF datasets. The source code is available at https://github.com/SunHui1216/SFMFusion.
>
---
#### [new 144] Pandar128 dataset for lane line detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Pandar128大规模LiDAR车道线检测数据集，解决现有数据不足与评估标准缺失问题，构建轻量基线方法SimpleLidarLane，并提出新评估指标IAM-F1，推动LiDAR车道检测的可复现研究。**

- **链接: [http://arxiv.org/pdf/2511.07084v1](http://arxiv.org/pdf/2511.07084v1)**

> **作者:** Filip Beránek; Václav Diviš; Ivan Gruber
>
> **摘要:** We present Pandar128, the largest public dataset for lane line detection using a 128-beam LiDAR. It contains over 52,000 camera frames and 34,000 LiDAR scans, captured in diverse real-world conditions in Germany. The dataset includes full sensor calibration (intrinsics, extrinsics) and synchronized odometry, supporting tasks such as projection, fusion, and temporal modeling. To complement the dataset, we also introduce SimpleLidarLane, a light-weight baseline method for lane line reconstruction that combines BEV segmentation, clustering, and polyline fitting. Despite its simplicity, our method achieves strong performance under challenging various conditions (e.g., rain, sparse returns), showing that modular pipelines paired with high-quality data and principled evaluation can compete with more complex approaches. Furthermore, to address the lack of standardized evaluation, we propose a novel polyline-based metric - Interpolation-Aware Matching F1 (IAM-F1) - that employs interpolation-aware lateral matching in BEV space. All data and code are publicly released to support reproducibility in LiDAR-based lane detection.
>
---
#### [new 145] Mono3DVG-EnSD: Enhanced Spatial-aware and Dimension-decoupled Text Encoding for Monocular 3D Visual Grounding
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对单目3D视觉定位任务，解决文本描述中过度依赖显式关键词与跨维度特征干扰问题，提出CLIP-LCA与D2M模块，动态保留空间隐含信息并解耦2D/3D文本特征，实现更精准的跨模态对齐，显著提升定位精度。**

- **链接: [http://arxiv.org/pdf/2511.06908v1](http://arxiv.org/pdf/2511.06908v1)**

> **作者:** Yuzhen Li; Min Liu; Zhaoyang Li; Yuan Bian; Xueping Wang; Erbo Zhai; Yaonan Wang
>
> **备注:** 10 pages
>
> **摘要:** Monocular 3D Visual Grounding (Mono3DVG) is an emerging task that locates 3D objects in RGB images using text descriptions with geometric cues. However, existing methods face two key limitations. Firstly, they often over-rely on high-certainty keywords that explicitly identify the target object while neglecting critical spatial descriptions. Secondly, generalized textual features contain both 2D and 3D descriptive information, thereby capturing an additional dimension of details compared to singular 2D or 3D visual features. This characteristic leads to cross-dimensional interference when refining visual features under text guidance. To overcome these challenges, we propose Mono3DVG-EnSD, a novel framework that integrates two key components: the CLIP-Guided Lexical Certainty Adapter (CLIP-LCA) and the Dimension-Decoupled Module (D2M). The CLIP-LCA dynamically masks high-certainty keywords while retaining low-certainty implicit spatial descriptions, thereby forcing the model to develop a deeper understanding of spatial relationships in captions for object localization. Meanwhile, the D2M decouples dimension-specific (2D/3D) textual features from generalized textual features to guide corresponding visual features at same dimension, which mitigates cross-dimensional interference by ensuring dimensionally-consistent cross-modal interactions. Through comprehensive comparisons and ablation studies on the Mono3DRefer dataset, our method achieves state-of-the-art (SOTA) performance across all metrics. Notably, it improves the challenging Far(Acc@0.5) scenario by a significant +13.54%.
>
---
#### [new 146] Improving Deepfake Detection with Reinforcement Learning-Based Adaptive Data Augmentation
- **分类: cs.CV; cs.CR**

- **简介: 该论文面向深度伪造检测任务，提出CRDA框架，利用强化学习与因果推断动态生成自适应数据增强样本，解决固定增强策略难以应对复杂多变伪造特征的问题，显著提升检测器跨域泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.07051v1](http://arxiv.org/pdf/2511.07051v1)**

> **作者:** Yuxuan Zhou; Tao Yu; Wen Huang; Yuheng Zhang; Tao Dai; Shu-Tao Xia
>
> **摘要:** The generalization capability of deepfake detectors is critical for real-world use. Data augmentation via synthetic fake face generation effectively enhances generalization, yet current SoTA methods rely on fixed strategies-raising a key question: Is a single static augmentation sufficient, or does the diversity of forgery features demand dynamic approaches? We argue existing methods overlook the evolving complexity of real-world forgeries (e.g., facial warping, expression manipulation), which fixed policies cannot fully simulate. To address this, we propose CRDA (Curriculum Reinforcement-Learning Data Augmentation), a novel framework guiding detectors to progressively master multi-domain forgery features from simple to complex. CRDA synthesizes augmented samples via a configurable pool of forgery operations and dynamically generates adversarial samples tailored to the detector's current learning state. Central to our approach is integrating reinforcement learning (RL) and causal inference. An RL agent dynamically selects augmentation actions based on detector performance to efficiently explore the vast augmentation space, adapting to increasingly challenging forgeries. Simultaneously, the agent introduces action space variations to generate heterogeneous forgery patterns, guided by causal inference to mitigate spurious correlations-suppressing task-irrelevant biases and focusing on causally invariant features. This integration ensures robust generalization by decoupling synthetic augmentation patterns from the model's learned representations. Extensive experiments show our method significantly improves detector generalizability, outperforming SOTA methods across multiple cross-domain datasets.
>
---
#### [new 147] Neodragon: Mobile Video Generation using Diffusion Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2511.06055v1](http://arxiv.org/pdf/2511.06055v1)**

> **作者:** Animesh Karnewar; Denis Korzhenkov; Ioannis Lelekas; Adil Karjauv; Noor Fathima; Hanwen Xiong; Vancheeswaran Vaidyanathan; Will Zeng; Rafael Esteves; Tushar Singhal; Fatih Porikli; Mohsen Ghafoorian; Amirhossein Habibian
>
> **摘要:** We introduce Neodragon, a text-to-video system capable of generating 2s (49 frames @24 fps) videos at the 640x1024 resolution directly on a Qualcomm Hexagon NPU in a record 6.7s (7 FPS). Differing from existing transformer-based offline text-to-video generation models, Neodragon is the first to have been specifically optimised for mobile hardware to achieve efficient and high-fidelity video synthesis. We achieve this through four key technical contributions: (1) Replacing the original large 4.762B T5xxl Text-Encoder with a much smaller 0.2B DT5 (DistilT5) with minimal quality loss, enabled through a novel Text-Encoder Distillation procedure. (2) Proposing an Asymmetric Decoder Distillation approach allowing us to replace the native codec-latent-VAE decoder with a more efficient one, without disturbing the generative latent-space of the generation pipeline. (3) Pruning of MMDiT blocks within the denoiser backbone based on their relative importance, with recovery of original performance through a two-stage distillation process. (4) Reducing the NFE (Neural Functional Evaluation) requirement of the denoiser by performing step distillation using DMD adapted for pyramidal flow-matching, thereby substantially accelerating video generation. When paired with an optimised SSD1B first-frame image generator and QuickSRNet for 2x super-resolution, our end-to-end Neodragon system becomes a highly parameter (4.945B full model), memory (3.5GB peak RAM usage), and runtime (6.7s E2E latency) efficient mobile-friendly model, while achieving a VBench total score of 81.61. By enabling low-cost, private, and on-device text-to-video synthesis, Neodragon democratizes AI-based video content creation, empowering creators to generate high-quality videos without reliance on cloud services. Code and model will be made publicly available at our website: https://qualcomm-ai-research.github.io/neodragon
>
---
#### [new 148] ConsistTalk: Intensity Controllable Temporally Consistent Talking Head Generation with Diffusion Noise Search
- **分类: cs.CV**

- **简介: ConsistTalk针对音频驱动的说话头生成任务，解决帧间闪烁、身份漂移和音画不同步问题，提出光流引导时序模块、音频-强度模型与扩散噪声初始化策略，实现强度可控且时序一致的高保真生成。**

- **链接: [http://arxiv.org/pdf/2511.06833v1](http://arxiv.org/pdf/2511.06833v1)**

> **作者:** Zhenjie Liu; Jianzhang Lu; Renjie Lu; Cong Liang; Shangfei Wang
>
> **备注:** AAAI26 poster
>
> **摘要:** Recent advancements in video diffusion models have significantly enhanced audio-driven portrait animation. However, current methods still suffer from flickering, identity drift, and poor audio-visual synchronization. These issues primarily stem from entangled appearance-motion representations and unstable inference strategies. In this paper, we introduce \textbf{ConsistTalk}, a novel intensity-controllable and temporally consistent talking head generation framework with diffusion noise search inference. First, we propose \textbf{an optical flow-guided temporal module (OFT)} that decouples motion features from static appearance by leveraging facial optical flow, thereby reducing visual flicker and improving temporal consistency. Second, we present an \textbf{Audio-to-Intensity (A2I) model} obtained through multimodal teacher-student knowledge distillation. By transforming audio and facial velocity features into a frame-wise intensity sequence, the A2I model enables joint modeling of audio and visual motion, resulting in more natural dynamics. This further enables fine-grained, frame-wise control of motion dynamics while maintaining tight audio-visual synchronization. Third, we introduce a \textbf{diffusion noise initialization strategy (IC-Init)}. By enforcing explicit constraints on background coherence and motion continuity during inference-time noise search, we achieve better identity preservation and refine motion dynamics compared to the current autoregressive strategy. Extensive experiments demonstrate that ConsistTalk significantly outperforms prior methods in reducing flicker, preserving identity, and delivering temporally stable, high-fidelity talking head videos.
>
---
#### [new 149] Segmentation of Ischemic Stroke Lesions using Transfer Learning on Multi-sequence MRI
- **分类: cs.CV**

- **简介: 该论文针对缺血性脑卒中病灶分割任务，提出基于迁移学习的Res-Unet框架，利用多序列MRI自动精准分割病灶，通过预训练权重提升模型性能，并融合多轴投票策略，实现80.5%的Dice分数，解决人工分割耗时且不一致的问题。**

- **链接: [http://arxiv.org/pdf/2511.07281v1](http://arxiv.org/pdf/2511.07281v1)**

> **作者:** R. P. Chowdhury; T. Rahman
>
> **备注:** Ischemic Stroke, Segmentation, Transfer Learning, Magnetic Resonance Imaging, Deep Learning, Res-UNet
>
> **摘要:** The accurate understanding of ischemic stroke lesions is critical for efficient therapy and prognosis of stroke patients. Magnetic resonance imaging (MRI) is sensitive to acute ischemic stroke and is a common diagnostic method for stroke. However, manual lesion segmentation performed by experts is tedious, time-consuming, and prone to observer inconsistency. Automatic medical image analysis methods have been proposed to overcome this challenge. However, previous approaches have relied on hand-crafted features that may not capture the irregular and physiologically complex shapes of ischemic stroke lesions. In this study, we present a novel framework for quickly and automatically segmenting ischemic stroke lesions on various MRI sequences, including T1-weighted, T2-weighted, DWI, and FLAIR. The proposed methodology is validated on the ISLES 2015 Brain Stroke sequence dataset, where we trained our model using the Res-Unet architecture twice: first, with pre-existing weights, and then without, to explore the benefits of transfer learning. Evaluation metrics, including the Dice score and sensitivity, were computed across 3D volumes. Finally, a Majority Voting Classifier was integrated to amalgamate the outcomes from each axis, resulting in a comprehensive segmentation method. Our efforts culminated in achieving a Dice score of 80.5\% and an accuracy of 74.03\%, showcasing the efficacy of our segmentation approach.
>
---
#### [new 150] Do Street View Imagery and Public Participation GIS align: Comparative Analysis of Urban Attractiveness
- **分类: cs.CV; cs.CY; cs.LG**

- **简介: 该论文比较Street View图像与公众参与GIS对城市吸引力的感知，发现二者仅部分一致。研究通过机器学习预测视觉吸引力，揭示非视觉因素（如噪音、活动）导致偏差，主张二者互补而非替代。**

- **链接: [http://arxiv.org/pdf/2511.05570v1](http://arxiv.org/pdf/2511.05570v1)**

> **作者:** Milad Malekzadeh; Elias Willberg; Jussi Torkko; Silviya Korpilo; Kamyar Hasanzadeh; Olle Järv; Tuuli Toivonen
>
> **摘要:** As digital tools increasingly shape spatial planning practices, understanding how different data sources reflect human experiences of urban environments is essential. Street View Imagery (SVI) and Public Participation GIS (PPGIS) represent two prominent approaches for capturing place-based perceptions that can support urban planning decisions, yet their comparability remains underexplored. This study investigates the alignment between SVI-based perceived attractiveness and residents' reported experiences gathered via a city-wide PPGIS survey in Helsinki, Finland. Using participant-rated SVI data and semantic image segmentation, we trained a machine learning model to predict perceived attractiveness based on visual features. We compared these predictions to PPGIS-identified locations marked as attractive or unattractive, calculating agreement using two sets of strict and moderate criteria. Our findings reveal only partial alignment between the two datasets. While agreement (with a moderate threshold) reached 67% for attractive and 77% for unattractive places, agreement (with a strict threshold) dropped to 27% and 29%, respectively. By analysing a range of contextual variables, including noise, traffic, population presence, and land use, we found that non-visual cues significantly contributed to mismatches. The model failed to account for experiential dimensions such as activity levels and environmental stressors that shape perceptions but are not visible in images. These results suggest that while SVI offers a scalable and visual proxy for urban perception, it cannot fully substitute the experiential richness captured through PPGIS. We argue that both methods are valuable but serve different purposes; therefore, a more integrated approach is needed to holistically capture how people perceive urban environments.
>
---
#### [new 151] DiLO: Disentangled Latent Optimization for Learning Shape and Deformation in Grouped Deforming 3D Objects
- **分类: cs.CV**

- **简介: 该论文提出DiLO方法，通过无监督学习将群组变形3D对象的形状与变形因素解耦，联合优化生成器与编码，并引入PoinNet编码器实现高效推断，应用于变形传递、分类与可解释性分析，效果优于复杂方法。**

- **链接: [http://arxiv.org/pdf/2511.06115v1](http://arxiv.org/pdf/2511.06115v1)**

> **作者:** Mostofa Rafid Uddin; Jana Armouti; Umong Sain; Md Asib Rahman; Xingjian Li; Min Xu
>
> **摘要:** In this work, we propose a disentangled latent optimization-based method for parameterizing grouped deforming 3D objects into shape and deformation factors in an unsupervised manner. Our approach involves the joint optimization of a generator network along with the shape and deformation factors, supported by specific regularization techniques. For efficient amortized inference of disentangled shape and deformation codes, we train two order-invariant PoinNet-based encoder networks in the second stage of our method. We demonstrate several significant downstream applications of our method, including unsupervised deformation transfer, deformation classification, and explainability analysis. Extensive experiments conducted on 3D human, animal, and facial expression datasets demonstrate that our simple approach is highly effective in these downstream tasks, comparable or superior to existing methods with much higher complexity.
>
---
#### [new 152] ClusterMine: Robust Label-Free Visual Out-Of-Distribution Detection via Concept Mining from Text Corpora
- **分类: cs.CV; cs.LG**

- **简介: 论文提出ClusterMine，用于无监督视觉离群检测，无需预定义正类标签。通过文本语料挖掘概念，结合聚类与零样本图文一致性，实现SOTA性能，且对分布偏移鲁棒。**

- **链接: [http://arxiv.org/pdf/2511.07068v1](http://arxiv.org/pdf/2511.07068v1)**

> **作者:** Nikolas Adaloglou; Diana Petrusheva; Mohamed Asker; Felix Michels; Markus Kollmann
>
> **备注:** Accepted in WACV 2026. Code in https://github.com/HHU-MMBS/clustermine_wacv_official 9 Tables, 11 Figures
>
> **摘要:** Large-scale visual out-of-distribution (OOD) detection has witnessed remarkable progress by leveraging vision-language models such as CLIP. However, a significant limitation of current methods is their reliance on a pre-defined set of in-distribution (ID) ground-truth label names (positives). These fixed label names can be unavailable, unreliable at scale, or become less relevant due to in-distribution shifts after deployment. Towards truly unsupervised OOD detection, we utilize widely available text corpora for positive label mining, bypassing the need for positives. In this paper, we utilize widely available text corpora for positive label mining under a general concept mining paradigm. Within this framework, we propose ClusterMine, a novel positive label mining method. ClusterMine is the first method to achieve state-of-the-art OOD detection performance without access to positive labels. It extracts positive concepts from a large text corpus by combining visual-only sample consistency (via clustering) and zero-shot image-text consistency. Our experimental study reveals that ClusterMine is scalable across a plethora of CLIP models and achieves state-of-the-art robustness to covariate in-distribution shifts. The code is available at https://github.com/HHU-MMBS/clustermine_wacv_official.
>
---
#### [new 153] Geometric implicit neural representations for signed distance functions
- **分类: cs.CV; cs.CG; cs.GR**

- **简介: 该论文综述了几何隐式神经表示（Geometric INRs）在符号距离函数（SDF）重建中的应用，通过引入微分几何正则项（如单位梯度约束）提升从点云或图像重建三维表面的精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2511.07206v1](http://arxiv.org/pdf/2511.07206v1)**

> **作者:** Luiz Schirmer; Tiago Novello; Vinícius da Silva; Guilherme Schardong; Daniel Perazzo; Hélio Lopes; Nuno Gonçalves; Luiz Velho
>
> **摘要:** \textit{Implicit neural representations} (INRs) have emerged as a promising framework for representing signals in low-dimensional spaces. This survey reviews the existing literature on the specialized INR problem of approximating \textit{signed distance functions} (SDFs) for surface scenes, using either oriented point clouds or a set of posed images. We refer to neural SDFs that incorporate differential geometry tools, such as normals and curvatures, in their loss functions as \textit{geometric} INRs. The key idea behind this 3D reconstruction approach is to include additional \textit{regularization} terms in the loss function, ensuring that the INR satisfies certain global properties that the function should hold -- such as having unit gradient in the case of SDFs. We explore key methodological components, including the definition of INR, the construction of geometric loss functions, and sampling schemes from a differential geometry perspective. Our review highlights the significant advancements enabled by geometric INRs in surface reconstruction from oriented point clouds and posed images.
>
---
#### [new 154] FreqGRL: Suppressing Low-Frequency Bias and Mining High-Frequency Knowledge for Cross-Domain Few-Shot Learning
- **分类: cs.CV**

- **简介: 论文针对跨域小样本学习中的数据不平衡问题，提出FreqGRL框架，通过频率域分析抑制源域低频偏差，增强目标域高频特征，引入LFR、HFE和GFF模块提升泛化能力，实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.06648v1](http://arxiv.org/pdf/2511.06648v1)**

> **作者:** Siqi Hui; Sanping Zhou; Ye deng; Wenli Huang; Jinjun Wang
>
> **摘要:** Cross-domain few-shot learning (CD-FSL) aims to recognize novel classes with only a few labeled examples under significant domain shifts. While recent approaches leverage a limited amount of labeled target-domain data to improve performance, the severe imbalance between abundant source data and scarce target data remains a critical challenge for effective representation learning. We present the first frequency-space perspective to analyze this issue and identify two key challenges: (1) models are easily biased toward source-specific knowledge encoded in the low-frequency components of source data, and (2) the sparsity of target data hinders the learning of high-frequency, domain-generalizable features. To address these challenges, we propose \textbf{FreqGRL}, a novel CD-FSL framework that mitigates the impact of data imbalance in the frequency space. Specifically, we introduce a Low-Frequency Replacement (LFR) module that substitutes the low-frequency components of source tasks with those from the target domain to create new source tasks that better align with target characteristics, thus reducing source-specific biases and promoting generalizable representation learning. We further design a High-Frequency Enhancement (HFE) module that filters out low-frequency components and performs learning directly on high-frequency features in the frequency space to improve cross-domain generalization. Additionally, a Global Frequency Filter (GFF) is incorporated to suppress noisy or irrelevant frequencies and emphasize informative ones, mitigating overfitting risks under limited target supervision. Extensive experiments on five standard CD-FSL benchmarks demonstrate that our frequency-guided framework achieves state-of-the-art performance.
>
---
#### [new 155] FilletRec: A Lightweight Graph Neural Network with Intrinsic Features for Automated Fillet Recognition
- **分类: cs.CV**

- **简介: 论文提出FilletRec，一种轻量图神经网络，用于CAD模型中倒角特征的自动识别与简化。通过构建大规模数据集并引入不变本征几何特征（如曲率），实现高精度、强泛化且参数极少的端到端解决方案。**

- **链接: [http://arxiv.org/pdf/2511.05561v1](http://arxiv.org/pdf/2511.05561v1)**

> **作者:** Jiali Gao; Taoran Liu; Hongfei Ye; Jianjun Chen
>
> **摘要:** Automated recognition and simplification of fillet features in CAD models is critical for CAE analysis, yet it remains an open challenge. Traditional rule-based methods lack robustness, while existing deep learning models suffer from poor generalization and low accuracy on complex fillets due to their generic design and inadequate training data. To address these issues, this paper proposes an end-to-end, data-driven framework specifically for fillet features. We first construct and release a large-scale, diverse benchmark dataset for fillet recognition to address the inadequacy of existing data. Based on it, we propose FilletRec, a lightweight graph neural network. The core innovation of this network is its use of pose-invariant intrinsic geometric features, such as curvature, enabling it to learn more fundamental geometric patterns and thereby achieve high-precision recognition of complex geometric topologies. Experiments show that FilletRec surpasses state-of-the-art methods in both accuracy and generalization, while using only 0.2\%-5.4\% of the parameters of baseline models, demonstrating high model efficiency. Finally, the framework completes the automated workflow from recognition to simplification by integrating an effective geometric simplification algorithm.
>
---
#### [new 156] PADM: A Physics-aware Diffusion Model for Attenuation Correction
- **分类: cs.CV**

- **简介: 该论文提出PADM，一种无需CT的物理感知扩散模型，用于心肌灌注SPECT图像的衰减校正，解决传统方法成本高、辐射大的问题，基于新构建的CardiAC数据集，仅用NAC输入实现高保真校正。**

- **链接: [http://arxiv.org/pdf/2511.06948v1](http://arxiv.org/pdf/2511.06948v1)**

> **作者:** Trung Kien Pham; Hoang Minh Vu; Anh Duc Chu; Dac Thai Nguyen; Trung Thanh Nguyen; Thao Nguyen Truong; Mai Hong Son; Thanh Trung Nguyen; Phi Le Nguyen
>
> **备注:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Attenuation artifacts remain a significant challenge in cardiac Myocardial Perfusion Imaging (MPI) using Single-Photon Emission Computed Tomography (SPECT), often compromising diagnostic accuracy and reducing clinical interpretability. While hybrid SPECT/CT systems mitigate these artifacts through CT-derived attenuation maps, their high cost, limited accessibility, and added radiation exposure hinder widespread clinical adoption. In this study, we propose a novel CT-free solution to attenuation correction in cardiac SPECT. Specifically, we introduce Physics-aware Attenuation Correction Diffusion Model (PADM), a diffusion-based generative method that incorporates explicit physics priors via a teacher--student distillation mechanism. This approach enables attenuation artifact correction using only Non-Attenuation-Corrected (NAC) input, while still benefiting from physics-informed supervision during training. To support this work, we also introduce CardiAC, a comprehensive dataset comprising 424 patient studies with paired NAC and Attenuation-Corrected (AC) reconstructions, alongside high-resolution CT-based attenuation maps. Extensive experiments demonstrate that PADM outperforms state-of-the-art generative models, delivering superior reconstruction fidelity across both quantitative metrics and visual assessment.
>
---
#### [new 157] GEWDiff: Geometric Enhanced Wavelet-based Diffusion Model for Hyperspectral Image Super-resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出GEWDiff，用于高光谱图像4倍超分辨率重建。针对传统扩散模型内存高、忽略几何结构、收敛不稳定等问题，引入小波编码、几何增强扩散与多级损失函数，显著提升重建质量。**

- **链接: [http://arxiv.org/pdf/2511.07103v1](http://arxiv.org/pdf/2511.07103v1)**

> **作者:** Sirui Wang; Jiang He; Natàlia Blasco Andreo; Xiao Xiang Zhu
>
> **备注:** This manuscript has been accepted for publication in AAAI 2026
>
> **摘要:** Improving the quality of hyperspectral images (HSIs), such as through super-resolution, is a crucial research area. However, generative modeling for HSIs presents several challenges. Due to their high spectral dimensionality, HSIs are too memory-intensive for direct input into conventional diffusion models. Furthermore, general generative models lack an understanding of the topological and geometric structures of ground objects in remote sensing imagery. In addition, most diffusion models optimize loss functions at the noise level, leading to a non-intuitive convergence behavior and suboptimal generation quality for complex data. To address these challenges, we propose a Geometric Enhanced Wavelet-based Diffusion Model (GEWDiff), a novel framework for reconstructing hyperspectral images at 4-times super-resolution. A wavelet-based encoder-decoder is introduced that efficiently compresses HSIs into a latent space while preserving spectral-spatial information. To avoid distortion during generation, we incorporate a geometry-enhanced diffusion process that preserves the geometric features. Furthermore, a multi-level loss function was designed to guide the diffusion process, promoting stable convergence and improved reconstruction fidelity. Our model demonstrated state-of-the-art results across multiple dimensions, including fidelity, spectral accuracy, visual realism, and clarity.
>
---
#### [new 158] LeCoT: revisiting network architecture for two-view correspondence pruning
- **分类: cs.CV**

- **简介: 论文提出LeCoT，用于两视图匹配点筛选任务，解决MLP难以捕获全局上下文的问题。设计Spatial-Channel Fusion Transformer块与渐进预测模块，无需额外组件即可高效利用全局信息，显著提升 pruning 及下游任务性能。**

- **链接: [http://arxiv.org/pdf/2511.07078v1](http://arxiv.org/pdf/2511.07078v1)**

> **作者:** Luanyuan Dai; Xiaoyu Du; Jinhui Tang
>
> **备注:** Just accepted at SCIENCE CHINA Information Sciences
>
> **摘要:** Two-view correspondence pruning aims to accurately remove incorrect correspondences (outliers) from initial ones and is widely applied to various computer vision tasks. Current popular strategies adopt multilayer perceptron (MLP) as the backbone, supplemented by additional modules to enhance the network ability to handle context information, which is a known limitation of MLPs. In contrast, we introduce a novel perspective for capturing correspondence context information without extra design modules. To this end, we design a two-view correspondence pruning network called LeCoT, which can naturally leverage global context information at different stages. Specifically, the core design of LeCoT is the Spatial-Channel Fusion Transformer block, a newly proposed component that efficiently utilizes both spatial and channel global context information among sparse correspondences. In addition, we integrate the proposed prediction block that utilizes correspondence features from intermediate stages to generate a probability set, which acts as guiding information for subsequent learning phases, allowing the network to more effectively capture robust global context information. Notably, this prediction block progressively refines the probability set, thereby mitigating the issue of information loss that is common in the traditional one. Extensive experiments prove that the proposed LeCoT outperforms state-of-the-art methods in correspondence pruning, relative pose estimation, homography estimation, visual localization, and $3$D~reconstruction tasks. The code is provided in https://github.com/Dailuanyuan2024/LeCoT-Revisiting-Network-Architecture-for-Two-View-Correspondence-Pruning.
>
---
#### [new 159] U(PM)$^2$:Unsupervised polygon matching with pre-trained models for challenging stereo images
- **分类: cs.CV**

- **简介: 该论文提出U(PM)²，面向立体图像的无监督多边形匹配任务，解决视差不连续、尺度变化等挑战。利用预训练模型自动提取特征，结合全局与局部匹配策略，无需训练即实现高精度、强泛化匹配。**

- **链接: [http://arxiv.org/pdf/2511.05949v1](http://arxiv.org/pdf/2511.05949v1)**

> **作者:** Chang Li; Xingtao Peng
>
> **摘要:** Stereo image matching is a fundamental task in computer vision, photogrammetry and remote sensing, but there is an almost unexplored field, i.e., polygon matching, which faces the following challenges: disparity discontinuity, scale variation, training requirement, and generalization. To address the above-mentioned issues, this paper proposes a novel U(PM)$^2$: low-cost unsupervised polygon matching with pre-trained models by uniting automatically learned and handcrafted features, of which pipeline is as follows: firstly, the detector leverages the pre-trained segment anything model to obtain masks; then, the vectorizer converts the masks to polygons and graphic structure; secondly, the global matcher addresses challenges from global viewpoint changes and scale variation based on bidirectional-pyramid strategy with pre-trained LoFTR; finally, the local matcher further overcomes local disparity discontinuity and topology inconsistency of polygon matching by local-joint geometry and multi-feature matching strategy with Hungarian algorithm. We benchmark our U(PM)$^2$ on the ScanNet and SceneFlow datasets using our proposed new metric, which achieved state-of-the-art accuracy at a competitive speed and satisfactory generalization performance at low cost without any training requirement.
>
---
#### [new 160] MVU-Eval: Towards Multi-Video Understanding Evaluation for Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **简介: 论文提出MVU-Eval，首个面向多视频理解的多模态大模型评估基准，解决现有评测仅限单视频的局限，构建1824个QA对覆盖4959个视频，评估八项核心能力，推动多视频场景下的AI理解研究。**

- **链接: [http://arxiv.org/pdf/2511.07250v1](http://arxiv.org/pdf/2511.07250v1)**

> **作者:** Tianhao Peng; Haochen Wang; Yuanxing Zhang; Zekun Wang; Zili Wang; Ge Zhang; Jian Yang; Shihao Li; Yanghai Wang; Xintao Wang; Houyi Li; Wei Ji; Pengfei Wan; Wenhao Huang; Zhaoxiang Zhang; Jiaheng Liu
>
> **摘要:** The advent of Multimodal Large Language Models (MLLMs) has expanded AI capabilities to visual modalities, yet existing evaluation benchmarks remain limited to single-video understanding, overlooking the critical need for multi-video understanding in real-world scenarios (e.g., sports analytics and autonomous driving). To address this significant gap, we introduce MVU-Eval, the first comprehensive benchmark for evaluating Multi-Video Understanding for MLLMs. Specifically, our MVU-Eval mainly assesses eight core competencies through 1,824 meticulously curated question-answer pairs spanning 4,959 videos from diverse domains, addressing both fundamental perception tasks and high-order reasoning tasks. These capabilities are rigorously aligned with real-world applications such as multi-sensor synthesis in autonomous systems and cross-angle sports analytics. Through extensive evaluation of state-of-the-art open-source and closed-source models, we reveal significant performance discrepancies and limitations in current MLLMs' ability to perform understanding across multiple videos. The benchmark will be made publicly available to foster future research.
>
---
#### [new 161] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field
- **分类: cs.CV**

- **简介: 论文提出Physics-Informed Deformable Gaussian Splatting（PIDG），将高斯点建模为拉格朗日材料点，融合物理守恒方程约束，提升单目视频动态场景重建的物理一致性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.06299v1](http://arxiv.org/pdf/2511.06299v1)**

> **作者:** Haoqin Hong; Ding Fan; Fubin Dou; Zhi-Li Zhou; Haoran Sun; Congcong Zhu; Jingrun Chen
>
> **备注:** Accepted by AAAI-26
>
> **摘要:** Recently, 3D Gaussian Splatting (3DGS), an explicit scene representation technique, has shown significant promise for dynamic novel-view synthesis from monocular video input. However, purely data-driven 3DGS often struggles to capture the diverse physics-driven motion patterns in dynamic scenes. To fill this gap, we propose Physics-Informed Deformable Gaussian Splatting (PIDG), which treats each Gaussian particle as a Lagrangian material point with time-varying constitutive parameters and is supervised by 2D optical flow via motion projection. Specifically, we adopt static-dynamic decoupled 4D decomposed hash encoding to reconstruct geometry and motion efficiently. Subsequently, we impose the Cauchy momentum residual as a physics constraint, enabling independent prediction of each particle's velocity and constitutive stress via a time-evolving material field. Finally, we further supervise data fitting by matching Lagrangian particle flow to camera-compensated optical flow, which accelerates convergence and improves generalization. Experiments on a custom physics-driven dataset as well as on standard synthetic and real-world datasets demonstrate significant gains in physical consistency and monocular dynamic reconstruction quality.
>
---
#### [new 162] From ACR O-RADS 2022 to Explainable Deep Learning: Comparative Performance of Expert Radiologists, Convolutional Neural Networks, Vision Transformers, and Fusion Models in Ovarian Masses
- **分类: cs.CV**

- **简介: 该论文研究卵巢肿块良恶性分类任务，比较放射科医生使用O-RADS 2022与多种深度学习模型（CNN、ViT）的诊断性能，并构建人机融合模型。结果表明ViT表现最优，人机融合显著提升CNN性能，助力标准化超声诊断。**

- **链接: [http://arxiv.org/pdf/2511.06282v1](http://arxiv.org/pdf/2511.06282v1)**

> **作者:** Ali Abbasian Ardakani; Afshin Mohammadi; Alisa Mohebbi; Anushya Vijayananthan; Sook Sam Leong; Lim Yi Ting; Mohd Kamil Bin Mohamad Fabell; U Rajendra Acharya; Sepideh Hatamikia
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Background: The 2022 update of the Ovarian-Adnexal Reporting and Data System (O-RADS) ultrasound classification refines risk stratification for adnexal lesions, yet human interpretation remains subject to variability and conservative thresholds. Concurrently, deep learning (DL) models have demonstrated promise in image-based ovarian lesion characterization. This study evaluates radiologist performance applying O-RADS v2022, compares it to leading convolutional neural network (CNN) and Vision Transformer (ViT) models, and investigates the diagnostic gains achieved by hybrid human-AI frameworks. Methods: In this single-center, retrospective cohort study, a total of 512 adnexal mass images from 227 patients (110 with at least one malignant cyst) were included. Sixteen DL models, including DenseNets, EfficientNets, ResNets, VGGs, Xception, and ViTs, were trained and validated. A hybrid model integrating radiologist O-RADS scores with DL-predicted probabilities was also built for each scheme. Results: Radiologist-only O-RADS assessment achieved an AUC of 0.683 and an overall accuracy of 68.0%. CNN models yielded AUCs of 0.620 to 0.908 and accuracies of 59.2% to 86.4%, while ViT16-384 reached the best performance, with an AUC of 0.941 and an accuracy of 87.4%. Hybrid human-AI frameworks further significantly enhanced the performance of CNN models; however, the improvement for ViT models was not statistically significant (P-value >0.05). Conclusions: DL models markedly outperform radiologist-only O-RADS v2022 assessment, and the integration of expert scores with AI yields the highest diagnostic accuracy and discrimination. Hybrid human-AI paradigms hold substantial potential to standardize pelvic ultrasound interpretation, reduce false positives, and improve detection of high-risk lesions.
>
---
#### [new 163] Federated Learning for Video Violence Detection: Complementary Roles of Lightweight CNNs and Vision-Language Models for Energy-Efficient Use
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究联邦学习下视频暴力检测任务，解决大模型能耗高问题，对比了轻量3D CNN与视觉语言模型的能效与精度，提出CNN为主、VLM为辅的混合部署策略，实现高精度与低能耗平衡。**

- **链接: [http://arxiv.org/pdf/2511.07171v1](http://arxiv.org/pdf/2511.07171v1)**

> **作者:** Sébastien Thuau; Siba Haidar; Rachid Chelouah
>
> **备注:** 5 pages, 3 figures, ICTAI 2025
>
> **摘要:** Deep learning-based video surveillance increasingly demands privacy-preserving architectures with low computational and environmental overhead. Federated learning preserves privacy but deploying large vision-language models (VLMs) introduces major energy and sustainability challenges. We compare three strategies for federated violence detection under realistic non-IID splits on the RWF-2000 and RLVS datasets: zero-shot inference with pretrained VLMs, LoRA-based fine-tuning of LLaVA-NeXT-Video-7B, and personalized federated learning of a 65.8M-parameter 3D CNN. All methods exceed 90% accuracy in binary violence detection. The 3D CNN achieves superior calibration (ROC AUC 92.59%) at roughly half the energy cost (240 Wh vs. 570 Wh) of federated LoRA, while VLMs provide richer multimodal reasoning. Hierarchical category grouping (based on semantic similarity and class exclusion) boosts VLM multiclass accuracy from 65.31% to 81% on the UCF-Crime dataset. To our knowledge, this is the first comparative simulation study of LoRA-tuned VLMs and personalized CNNs for federated violence detection, with explicit energy and CO2e quantification. Our results inform hybrid deployment strategies that default to efficient CNNs for routine inference and selectively engage VLMs for complex contextual reasoning.
>
---
#### [new 164] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对多模态大模型后训练中样本难度未量化、感知与推理未协同优化的问题，提出PISM与CMAB两种难度感知采样策略，构建分层训练框架，无需SFT即可提升模型性能。**

- **链接: [http://arxiv.org/pdf/2511.06722v1](http://arxiv.org/pdf/2511.06722v1)**

> **作者:** Jianyu Qi; Ding Zou; Wenrui Yan; Rui Ma; Jiaxu Li; Zhijie Zheng; Zhiguo Yang; Rongchang Zhao
>
> **备注:** Accpeted by AAAI 2026
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have spurred significant progress in Chain-of-Thought (CoT) reasoning. Building on the success of Deepseek-R1, researchers extended multimodal reasoning to post-training paradigms based on reinforcement learning (RL), focusing predominantly on mathematical datasets. However, existing post-training paradigms tend to neglect two critical aspects: (1) The lack of quantifiable difficulty metrics capable of strategically screening samples for post-training optimization. (2) Suboptimal post-training paradigms that fail to jointly optimize perception and reasoning capabilities. To address this gap, we propose two novel difficulty-aware sampling strategies: Progressive Image Semantic Masking (PISM) quantifies sample hardness through systematic image degradation, while Cross-Modality Attention Balance (CMAB) assesses cross-modal interaction complexity via attention distribution analysis. Leveraging these metrics, we design a hierarchical training framework that incorporates both GRPO-only and SFT+GRPO hybrid training paradigms, and evaluate them across six benchmark datasets. Experiments demonstrate consistent superiority of GRPO applied to difficulty-stratified samples compared to conventional SFT+GRPO pipelines, indicating that strategic data sampling can obviate the need for supervised fine-tuning while improving model accuracy. Our code will be released at https://github.com/qijianyu277/DifficultySampling.
>
---
#### [new 165] Adapted Foundation Models for Breast MRI Triaging in Contrast-Enhanced and Non-Contrast Enhanced Protocols
- **分类: cs.CV; cs.AI**

- **简介: 该论文利用DINOv2驱动的Medical Slice Transformer，在乳腺MRI中实现自动化初筛，目标是以≥97.5%敏感性排除BI-RADS≥4的显著病变，评估了对比增强与非增强协议的性能，最高特异性达19%。**

- **链接: [http://arxiv.org/pdf/2511.05967v1](http://arxiv.org/pdf/2511.05967v1)**

> **作者:** Tri-Thien Nguyen; Lorenz A. Kapsner; Tobias Hepp; Shirin Heidarikahkesh; Hannes Schreiter; Luise Brock; Dominika Skwierawska; Dominique Hadler; Julian Hossbach; Evelyn Wenkel; Sabine Ohlmeyer; Frederik B. Laun; Andrzej Liebert; Andreas Maier; Michael Uder; Sebastian Bickelhaupt
>
> **备注:** 23 pages, 6 figures, 4 tables. Originally submitted to Radiology (RAD-25-2541); under consideration for transfer to Radiology: Artificial Intelligence (RSNA Portfolio Journal)
>
> **摘要:** Background: Magnetic resonance imaging (MRI) has high sensitivity for breast cancer detection, but interpretation is time-consuming. Artificial intelligence may aid in pre-screening. Purpose: To evaluate the DINOv2-based Medical Slice Transformer (MST) for ruling out significant findings (Breast Imaging Reporting and Data System [BI-RADS] >=4) in contrast-enhanced and non-contrast-enhanced abbreviated breast MRI. Materials and Methods: This institutional review board approved retrospective study included 1,847 single-breast MRI examinations (377 BI-RADS >=4) from an in-house dataset and 924 from an external validation dataset (Duke). Four abbreviated protocols were tested: T1-weighted early subtraction (T1sub), diffusion-weighted imaging with b=1500 s/mm2 (DWI1500), DWI1500+T2-weighted (T2w), and T1sub+T2w. Performance was assessed at 90%, 95%, and 97.5% sensitivity using five-fold cross-validation and area under the receiver operating characteristic curve (AUC) analysis. AUC differences were compared with the DeLong test. False negatives were characterized, and attention maps of true positives were rated in the external dataset. Results: A total of 1,448 female patients (mean age, 49 +/- 12 years) were included. T1sub+T2w achieved an AUC of 0.77 +/- 0.04; DWI1500+T2w, 0.74 +/- 0.04 (p=0.15). At 97.5% sensitivity, T1sub+T2w had the highest specificity (19% +/- 7%), followed by DWI1500+T2w (17% +/- 11%). Missed lesions had a mean diameter <10 mm at 95% and 97.5% thresholds for both T1sub and DWI1500, predominantly non-mass enhancements. External validation yielded an AUC of 0.77, with 88% of attention maps rated good or moderate. Conclusion: At 97.5% sensitivity, the MST framework correctly triaged cases without BI-RADS >=4, achieving 19% specificity for contrast-enhanced and 17% for non-contrast-enhanced MRI. Further research is warranted before clinical implementation.
>
---
#### [new 166] Pose-Aware Multi-Level Motion Parsing for Action Quality Assessment
- **分类: cs.CV**

- **简介: 该论文针对动作质量评估（AQA）任务，提出多级运动解析框架，通过姿态感知的分层解析（动作单元、运动变化、环境条件）与加权评分模块，提升动作分割与评分精度，尤其适用于跳水等复杂场景。**

- **链接: [http://arxiv.org/pdf/2511.05611v1](http://arxiv.org/pdf/2511.05611v1)**

> **作者:** Shuaikang Zhu; Yang Yang; Chen Sun
>
> **摘要:** Human pose serves as a cornerstone of action quality assessment (AQA), where subtle spatial-temporal variations in pose often distinguish excellence from mediocrity. In high-level competitions, these nuanced differences become decisive factors in scoring. In this paper, we propose a novel multi-level motion parsing framework for AQA based on enhanced spatial-temporal pose features. On the first level, the Action-Unit Parser is designed with the help of pose extraction to achieve precise action segmentation and comprehensive local-global pose representations. On the second level, Motion Parser is used by spatial-temporal feature learning to capture pose changes and appearance details for each action-unit. Meanwhile, some special conditions other than body-related will impact action scoring, like water splash in diving. In this work, we design an additional Condition Parser to offer users more flexibility in their choices. Finally, Weight-Adjust Scoring Module is introduced to better accommodate the diverse requirements of various action types and the multi-scale nature of action-units. Extensive evaluations on large-scale diving sports datasets demonstrate that our multi-level motion parsing framework achieves state-of-the-art performance in both action segmentation and action scoring tasks.
>
---
#### [new 167] Countering Multi-modal Representation Collapse through Rank-targeted Fusion
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多模态融合中的特征与模态坍缩问题，提出Rank-enhancing Token Fuser，利用有效秩量化并协同缓解两类坍缩，构建R3D框架提升动作预测性能，在多个数据集上显著超越SOTA方法。**

- **链接: [http://arxiv.org/pdf/2511.06450v1](http://arxiv.org/pdf/2511.06450v1)**

> **作者:** Seulgi Kim; Kiran Kokilepersaud; Mohit Prabhushankar; Ghassan AlRegib
>
> **备注:** Accepted in 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
>
> **摘要:** Multi-modal fusion methods often suffer from two types of representation collapse: feature collapse where individual dimensions lose their discriminative power (as measured by eigenspectra), and modality collapse where one dominant modality overwhelms the other. Applications like human action anticipation that require fusing multifarious sensor data are hindered by both feature and modality collapse. However, existing methods attempt to counter feature collapse and modality collapse separately. This is because there is no unifying framework that efficiently addresses feature and modality collapse in conjunction. In this paper, we posit the utility of effective rank as an informative measure that can be utilized to quantify and counter both the representation collapses. We propose \textit{Rank-enhancing Token Fuser}, a theoretically grounded fusion framework that selectively blends less informative features from one modality with complementary features from another modality. We show that our method increases the effective rank of the fused representation. To address modality collapse, we evaluate modality combinations that mutually increase each others' effective rank. We show that depth maintains representational balance when fused with RGB, avoiding modality collapse. We validate our method on action anticipation, where we present \texttt{R3D}, a depth-informed fusion framework. Extensive experiments on NTURGBD, UTKinect, and DARai demonstrate that our approach significantly outperforms prior state-of-the-art methods by up to 3.74\%. Our code is available at: \href{https://github.com/olivesgatech/R3D}{https://github.com/olivesgatech/R3D}.
>
---
#### [new 168] Explainable Cross-Disease Reasoning for Cardiovascular Risk Assessment from LDCT
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出一种可解释的跨疾病推理框架，从单次LDCT扫描中联合评估肺部与心血管风险，解决传统方法忽视器官间生理关联的问题。通过感知-推理-表征三模块协同，实现准确且可解释的心血管风险预测。**

- **链接: [http://arxiv.org/pdf/2511.06625v1](http://arxiv.org/pdf/2511.06625v1)**

> **作者:** Yifei Zhang; Jiashuo Zhang; Xiaofeng Yang; Liang Zhao
>
> **摘要:** Low-dose chest computed tomography (LDCT) inherently captures both pulmonary and cardiac structures, offering a unique opportunity for joint assessment of lung and cardiovascular health. However, most existing approaches treat these domains as independent tasks, overlooking their physiological interplay and shared imaging biomarkers. We propose an Explainable Cross-Disease Reasoning Framework that enables interpretable cardiopulmonary risk assessment from a single LDCT scan. The framework introduces an agentic reasoning process that emulates clinical diagnostic thinking-first perceiving pulmonary findings, then reasoning through established medical knowledge, and finally deriving a cardiovascular judgment with explanatory rationale. It integrates three synergistic components: a pulmonary perception module that summarizes lung abnormalities, a knowledge-guided reasoning module that infers their cardiovascular implications, and a cardiac representation module that encodes structural biomarkers. Their outputs are fused to produce a holistic cardiovascular risk prediction that is both accurate and physiologically grounded. Experiments on the NLST cohort demonstrate that the proposed framework achieves state-of-the-art performance for CVD screening and mortality prediction, outperforming single-disease and purely image-based baselines. Beyond quantitative gains, the framework provides human-verifiable reasoning that aligns with cardiological understanding, revealing coherent links between pulmonary abnormalities and cardiac stress mechanisms. Overall, this work establishes a unified and explainable paradigm for cardiovascular analysis from LDCT, bridging the gap between image-based prediction and mechanism-based medical interpretation.
>
---
#### [new 169] Leveraging Text-Driven Semantic Variation for Robust OOD Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种文本驱动的异常分割方法，利用视觉-语言模型的语义多样性，通过距离感知提示和语义增强，提升自动驾驶中未知物体的检测鲁棒性，在多个基准上实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.07238v1](http://arxiv.org/pdf/2511.07238v1)**

> **作者:** Seungheon Song; Jaekoo Lee
>
> **备注:** 8 pages, 5 figure references, 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) submission
>
> **摘要:** In autonomous driving and robotics, ensuring road safety and reliable decision-making critically depends on out-of-distribution (OOD) segmentation. While numerous methods have been proposed to detect anomalous objects on the road, leveraging the vision-language space-which provides rich linguistic knowledge-remains an underexplored field. We hypothesize that incorporating these linguistic cues can be especially beneficial in the complex contexts found in real-world autonomous driving scenarios. To this end, we present a novel approach that trains a Text-Driven OOD Segmentation model to learn a semantically diverse set of objects in the vision-language space. Concretely, our approach combines a vision-language model's encoder with a transformer decoder, employs Distance-Based OOD prompts located at varying semantic distances from in-distribution (ID) classes, and utilizes OOD Semantic Augmentation for OOD representations. By aligning visual and textual information, our approach effectively generalizes to unseen objects and provides robust OOD segmentation in diverse driving environments. We conduct extensive experiments on publicly available OOD segmentation datasets such as Fishyscapes, Segment-Me-If-You-Can, and Road Anomaly datasets, demonstrating that our approach achieves state-of-the-art performance across both pixel-level and object-level evaluations. This result underscores the potential of vision-language-based OOD segmentation to bolster the safety and reliability of future autonomous driving systems.
>
---
#### [new 170] VADER: Towards Causal Video Anomaly Understanding with Relation-Aware Large Language Models
- **分类: cs.CV**

- **简介: VADER提出一种基于大语言模型的视频异常理解框架，通过建模对象间因果关系与动态交互，提升异常事件的语义解释与因果推理能力，解决传统方法忽视深层因果联系的问题。**

- **链接: [http://arxiv.org/pdf/2511.07299v1](http://arxiv.org/pdf/2511.07299v1)**

> **作者:** Ying Cheng; Yu-Ho Lin; Min-Hung Chen; Fu-En Yang; Shang-Hong Lai
>
> **摘要:** Video anomaly understanding (VAU) aims to provide detailed interpretation and semantic comprehension of anomalous events within videos, addressing limitations of traditional methods that focus solely on detecting and localizing anomalies. However, existing approaches often neglect the deeper causal relationships and interactions between objects, which are critical for understanding anomalous behaviors. In this paper, we propose VADER, an LLM-driven framework for Video Anomaly unDErstanding, which integrates keyframe object Relation features with visual cues to enhance anomaly comprehension from video. Specifically, VADER first applies an Anomaly Scorer to assign per-frame anomaly scores, followed by a Context-AwarE Sampling (CAES) strategy to capture the causal context of each anomalous event. A Relation Feature Extractor and a COntrastive Relation Encoder (CORE) jointly model dynamic object interactions, producing compact relational representations for downstream reasoning. These visual and relational cues are integrated with LLMs to generate detailed, causally grounded descriptions and support robust anomaly-related question answering. Experiments on multiple real-world VAU benchmarks demonstrate that VADER achieves strong results across anomaly description, explanation, and causal reasoning tasks, advancing the frontier of explainable video anomaly analysis.
>
---
#### [new 171] Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对无源域目标检测（SFOD）任务，提出利用视觉基础模型（VFM）作为外部知识源，通过特征对齐与伪标签融合三模块，提升跨域迁移性与判别力，显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.07301v1](http://arxiv.org/pdf/2511.07301v1)**

> **作者:** Huizai Yao; Sicheng Zhao; Pengteng Li; Yi Cui; Shuo Lu; Weiyu Guo; Yunfan Lu; Yijie Xu; Hui Xiong
>
> **备注:** Accepted to AAAI 2026. Extended version with full Appendix
>
> **摘要:** Source-Free Object Detection (SFOD) aims to adapt a source-pretrained object detector to a target domain without access to source data. However, existing SFOD methods predominantly rely on internal knowledge from the source model, which limits their capacity to generalize across domains and often results in biased pseudo-labels, thereby hindering both transferability and discriminability. In contrast, Vision Foundation Models (VFMs), pretrained on massive and diverse data, exhibit strong perception capabilities and broad generalization, yet their potential remains largely untapped in the SFOD setting. In this paper, we propose a novel SFOD framework that leverages VFMs as external knowledge sources to jointly enhance feature alignment and label quality. Specifically, we design three VFM-based modules: (1) Patch-weighted Global Feature Alignment (PGFA) distills global features from VFMs using patch-similarity-based weighting to enhance global feature transferability; (2) Prototype-based Instance Feature Alignment (PIFA) performs instance-level contrastive learning guided by momentum-updated VFM prototypes; and (3) Dual-source Enhanced Pseudo-label Fusion (DEPF) fuses predictions from detection VFMs and teacher models via an entropy-aware strategy to yield more reliable supervision. Extensive experiments on six benchmarks demonstrate that our method achieves state-of-the-art SFOD performance, validating the effectiveness of integrating VFMs to simultaneously improve transferability and discriminability.
>
---
#### [new 172] In-Context-Learning-Assisted Quality Assessment Vision-Language Models for Metal Additive Manufacturing
- **分类: cs.CV**

- **简介: 该论文利用视觉语言模型（VLMs）结合上下文学习（ICL），在金属增材制造中实现小样本质量评估，无需大量标注数据，同时生成可解释的推理依据，并提出新指标评估解释质量。**

- **链接: [http://arxiv.org/pdf/2511.05551v1](http://arxiv.org/pdf/2511.05551v1)**

> **作者:** Qiaojie Zheng; Jiucai Zhang; Xiaoli Zhang
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Vision-based quality assessment in additive manufacturing often requires dedicated machine learning models and application-specific datasets. However, data collection and model training can be expensive and time-consuming. In this paper, we leverage vision-language models' (VLMs') reasoning capabilities to assess the quality of printed parts and introduce in-context learning (ICL) to provide VLMs with necessary application-specific knowledge and demonstration samples. This method eliminates the requirement for large application-specific datasets for training models. We explored different sampling strategies for ICL to search for the optimal configuration that makes use of limited samples. We evaluated these strategies on two VLMs, Gemini-2.5-flash and Gemma3:27b, with quality assessment tasks in wire-laser direct energy deposition processes. The results show that ICL-assisted VLMs can reach quality classification accuracies similar to those of traditional machine learning models while requiring only a minimal number of samples. In addition, unlike traditional classification models that lack transparency, VLMs can generate human-interpretable rationales to enhance trust. Since there are no metrics to evaluate their interpretability in manufacturing applications, we propose two metrics, knowledge relevance and rationale validity, to evaluate the quality of VLMs' supporting rationales. Our results show that ICL-assisted VLMs can address application-specific tasks with limited data, achieving relatively high accuracy while also providing valid supporting rationales for improved decision transparency.
>
---
#### [new 173] GABFusion: Rethinking Feature Fusion for Low-Bit Quantization of Multi-Task Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多任务网络低比特量化中的梯度冲突与特征差异问题，提出GABFusion动态平衡梯度并融合特征，结合ADA实现量化友好特征蒸馏，显著提升各架构与比特下的量化性能，且无需修改原网络结构。**

- **链接: [http://arxiv.org/pdf/2511.05898v1](http://arxiv.org/pdf/2511.05898v1)**

> **作者:** Zhaoyang Wang; Dong Wang
>
> **备注:** 9 pages,6 figures
>
> **摘要:** Despite the effectiveness of quantization-aware training (QAT) in compressing deep neural networks, its performance on multi-task architectures often degrades significantly due to task-specific feature discrepancies and gradient conflicts. To address these challenges, we propose Gradient-Aware Balanced Feature Fusion (GABFusion), which dynamically balances gradient magnitudes and fuses task-specific features in a quantization-friendly manner. We further introduce Attention Distribution Alignment (ADA), a feature-level distillation strategy tailored for quantized models. Our method demonstrates strong generalization across network architectures and QAT algorithms, with theoretical guarantees on gradient bias reduction. Extensive experiments demonstrate that our strategy consistently enhances a variety of QAT methods across different network architectures and bit-widths. On PASCAL VOC and COCO datasets, the proposed approach achieves average mAP improvements of approximately 3.3% and 1.6%, respectively. When applied to YOLOv5 under 4-bit quantization, our method narrows the accuracy gap with the full-precision model to only 1.7% on VOC, showcasing its effectiveness in preserving performance under low-bit constraints. Notably, the proposed framework is modular, easy to integrate, and compatible with any existing QAT technique-enhancing the performance of quantized models without requiring modifications to the original network architecture.
>
---
#### [new 174] TiS-TSL: Image-Label Supervised Surgical Video Stereo Matching via Time-Switchable Teacher-Student Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对微创手术视频的立体匹配任务，解决稀疏图像标签下时序不稳定问题。提出TiS-TSL框架，通过时序可切换的师生学习，融合前后向预测实现时空一致性约束，提升稠密视差预测的稳定性与精度。**

- **链接: [http://arxiv.org/pdf/2511.06817v1](http://arxiv.org/pdf/2511.06817v1)**

> **作者:** Rui Wang; Ying Zhou; Hao Wang; Wenwei Zhang; Qiang Li; Zhiwei Wang
>
> **备注:** 8 pages, 4 figures, accepted by BiBM2025
>
> **摘要:** Stereo matching in minimally invasive surgery (MIS) is essential for next-generation navigation and augmented reality. Yet, dense disparity supervision is nearly impossible due to anatomical constraints, typically limiting annotations to only a few image-level labels acquired before the endoscope enters deep body cavities. Teacher-Student Learning (TSL) offers a promising solution by leveraging a teacher trained on sparse labels to generate pseudo labels and associated confidence maps from abundant unlabeled surgical videos. However, existing TSL methods are confined to image-level supervision, providing only spatial confidence and lacking temporal consistency estimation. This absence of spatio-temporal reliability results in unstable disparity predictions and severe flickering artifacts across video frames. To overcome these challenges, we propose TiS-TSL, a novel time-switchable teacher-student learning framework for video stereo matching under minimal supervision. At its core is a unified model that operates in three distinct modes: Image-Prediction (IP), Forward Video-Prediction (FVP), and Backward Video-Prediction (BVP), enabling flexible temporal modeling within a single architecture. Enabled by this unified model, TiS-TSL adopts a two-stage learning strategy. The Image-to-Video (I2V) stage transfers sparse image-level knowledge to initialize temporal modeling. The subsequent Video-to-Video (V2V) stage refines temporal disparity predictions by comparing forward and backward predictions to calculate bidirectional spatio-temporal consistency. This consistency identifies unreliable regions across frames, filters noisy video-level pseudo labels, and enforces temporal coherence. Experimental results on two public datasets demonstrate that TiS-TSL exceeds other image-based state-of-the-arts by improving TEPE and EPE by at least 2.11% and 4.54%, respectively..
>
---
#### [new 175] CINEMAE: Leveraging Frozen Masked Autoencoders for Cross-Generator AI Image Detection
- **分类: cs.CV; cs.AI; cs.CY; 68T07; I.4.8**

- **简介: CINEMAE提出一种基于冻结MAE的AI生成图像检测方法，通过计算局部语义重建负对数似然捕捉生成异常，实现跨生成器泛化。仅在Stable Diffusion上训练，即可在8个 unseen 生成器上超95%准确率。**

- **链接: [http://arxiv.org/pdf/2511.06325v1](http://arxiv.org/pdf/2511.06325v1)**

> **作者:** Minsuk Jang; Hyeonseo Jeong; Minseok Son; Changick Kim
>
> **摘要:** While context-based detectors have achieved strong generalization for AI-generated text by measuring distributional inconsistencies, image-based detectors still struggle with overfitting to generator-specific artifacts. We introduce CINEMAE, a novel paradigm for AIGC image detection that adapts the core principles of text detection methods to the visual domain. Our key insight is that Masked AutoEncoder (MAE), trained to reconstruct masked patches conditioned on visible context, naturally encodes semantic consistency expectations. We formalize this reconstruction process probabilistically, computing conditional Negative Log-Likelihood (NLL, p(masked | visible)) to quantify local semantic anomalies. By aggregating these patch-level statistics with global MAE features through learned fusion, CINEMAE achieves strong cross-generator generalization. Trained exclusively on Stable Diffusion v1.4, our method achieves over 95% accuracy on all eight unseen generators in the GenImage benchmark, substantially outperforming state-of-the-art detectors. This demonstrates that context-conditional reconstruction uncertainty provides a robust, transferable signal for AIGC detection.
>
---
#### [new 176] S2ML: Spatio-Spectral Mutual Learning for Depth Completion
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出S2ML框架，用于深度补全任务，解决RGB-D相机获取的深度图因反射弱、阴影等导致的缺失问题。通过空间与频域互学习，融合幅值与相位谱特性，提升补全精度，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06033v1](http://arxiv.org/pdf/2511.06033v1)**

> **作者:** Zihui Zhao; Yifei Zhang; Zheng Wang; Yang Li; Kui Jiang; Zihan Geng; Chia-Wen Lin
>
> **摘要:** The raw depth images captured by RGB-D cameras using Time-of-Flight (TOF) or structured light often suffer from incomplete depth values due to weak reflections, boundary shadows, and artifacts, which limit their applications in downstream vision tasks. Existing methods address this problem through depth completion in the image domain, but they overlook the physical characteristics of raw depth images. It has been observed that the presence of invalid depth areas alters the frequency distribution pattern. In this work, we propose a Spatio-Spectral Mutual Learning framework (S2ML) to harmonize the advantages of both spatial and frequency domains for depth completion. Specifically, we consider the distinct properties of amplitude and phase spectra and devise a dedicated spectral fusion module. Meanwhile, the local and global correlations between spatial-domain and frequency-domain features are calculated in a unified embedding space. The gradual mutual representation and refinement encourage the network to fully explore complementary physical characteristics and priors for more accurate depth completion. Extensive experiments demonstrate the effectiveness of our proposed S2ML method, outperforming the state-of-the-art method CFormer by 0.828 dB and 0.834 dB on the NYU-Depth V2 and SUN RGB-D datasets, respectively.
>
---
#### [new 177] Mapping Reduced Accessibility to WASH Facilities in Rohingya Refugee Camps with Sub-Meter Imagery
- **分类: cs.CV**

- **简介: 该论文利用亚米级遥感影像与半监督分割模型，量化罗兴亚难民营中WASH设施（水泵、厕所、淋浴间）的可达性，揭示人口增长导致的资源短缺与性别不平等，为有限资源下的公平分配提供数据支持。**

- **链接: [http://arxiv.org/pdf/2511.07231v1](http://arxiv.org/pdf/2511.07231v1)**

> **作者:** Kyeongjin Ahn; YongHun Suh; Sungwon Han; Jeasurk Yang; Hannes Taubenböck; Meeyoung Cha
>
> **备注:** 23 pages, 13 figures, 2 tables
>
> **摘要:** Access to Water, Sanitation, and Hygiene (WASH) services remains a major public health concern in refugee camps. This study introduces a remote sensing-driven framework to quantify WASH accessibility-specifically to water pumps, latrines, and bathing cubicles-in the Rohingya camps of Cox's Bazar, one of the world's most densely populated displacement settings. Detecting refugee shelters in such emergent camps presents substantial challenges, primarily due to their dense spatial configuration and irregular geometric patterns. Using sub-meter satellite images, we develop a semi-supervised segmentation framework that achieves an F1-score of 76.4% in detecting individual refugee shelters. Applying the framework across multi-year data reveals declining WASH accessibility, driven by rapid refugee population growth and reduced facility availability, rising from 25 people per facility in 2022 to 29.4 in 2025. Gender-disaggregated analysis further shows that women and girls experience reduced accessibility, in scenarios with inadequate safety-related segregation in WASH facilities. These findings suggest the importance of demand-responsive allocation strategies that can identify areas with under-served populations-such as women and girls-and ensure that limited infrastructure serves the greatest number of people in settings with fixed or shrinking budgets. We also discuss the value of high-resolution remote sensing and machine learning to detect inequality and inform equitable resource planning in complex humanitarian environments.
>
---
#### [new 178] Token Is All You Need: Cognitive Planning through Sparse Intent Alignment
- **分类: cs.CV; cs.AI; cs.LG; cs.RO; 68T40; I.2.9; I.2.6; I.2.10**

- **简介: 该论文面向端到端自动驾驶规划任务，提出“Token Is All You Need”框架，用稀疏语义令牌替代复杂场景建模，仅靠感知信息即可实现高效规划，显著降低轨迹误差，并揭示模型可自适应关注任务相关语义，实现认知式想象规划。**

- **链接: [http://arxiv.org/pdf/2511.05540v1](http://arxiv.org/pdf/2511.05540v1)**

> **作者:** Shiyao Sang
>
> **备注:** 6 pages, 2 figures. Preprint exploring a new cognitive paradigm for autonomous planning
>
> **摘要:** We challenge the long-standing assumption that exhaustive scene modeling is required for high-performance end-to-end autonomous driving (E2EAD). Unlike world-model approaches that rely on computationally intensive future scene generation or vision-language-action (VLA) systems constrained by Markov assumptions, we show that a minimal set of semantically rich tokens is sufficient for effective planning. Experiments on the nuPlan benchmark (720 scenarios, over 11,000 samples) using perception-informed BEV representations yield three key findings: (1) even without future prediction, our sparse representation achieves 0.548 m ADE, comparable to or surpassing prior methods reporting around 0.75 m on nuScenes; (2) conditioning trajectory decoding on predicted future tokens reduces ADE to 0.479 m, a 12.6% improvement over current-state baselines; and (3) explicit reconstruction loss offers no benefit and may degrade performance under reliable perception inputs. Notably, we observe the emergence of temporal fuzziness, where the model adaptively attends to task-relevant semantics rather than aligning rigidly to fixed timestamps, providing a cognitive advantage for planning under uncertainty. Our "token is all you need" principle marks a paradigm shift from reconstructing the world to understanding it, laying a foundation for cognitively inspired systems that plan through imagination rather than reaction.
>
---
#### [new 179] Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images
- **分类: cs.CV**

- **简介: Omni-View提出一种基于多视角图像的统一3D模型，通过生成（视图合成与几何估计）促进理解，联合建模场景理解、视图合成与几何恢复，实现生成与理解的协同优化，在VSI-Bench上达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.07222v1](http://arxiv.org/pdf/2511.07222v1)**

> **作者:** JiaKui Hu; Shanshan Zhao; Qing-Guo Chen; Xuerui Qiu; Jialun Liu; Zhao Xu; Weihua Luo; Kaifu Zhang; Yanye Lu
>
> **备注:** Under review
>
> **摘要:** This paper presents Omni-View, which extends the unified multimodal understanding and generation to 3D scenes based on multiview images, exploring the principle that "generation facilitates understanding". Consisting of understanding model, texture module, and geometry module, Omni-View jointly models scene understanding, novel view synthesis, and geometry estimation, enabling synergistic interaction between 3D scene understanding and generation tasks. By design, it leverages the spatiotemporal modeling capabilities of its texture module responsible for appearance synthesis, alongside the explicit geometric constraints provided by its dedicated geometry module, thereby enriching the model's holistic understanding of 3D scenes. Trained with a two-stage strategy, Omni-View achieves a state-of-the-art score of 55.4 on the VSI-Bench benchmark, outperforming existing specialized 3D understanding models, while simultaneously delivering strong performance in both novel view synthesis and 3D scene generation.
>
---
#### [new 180] Image Restoration via Primal Dual Hybrid Gradient and Flow Generative Model
- **分类: cs.CV**

- **简介: 该论文提出一种基于PDHG的Plug-and-Play框架，将流生成模型作为先验，替代传统正则项，支持ℓ₁和ℓ₂保真项，提升非高斯噪声（如泊松、脉冲噪声）下图像恢复性能。**

- **链接: [http://arxiv.org/pdf/2511.06748v1](http://arxiv.org/pdf/2511.06748v1)**

> **作者:** Ji Li; Chao Wang
>
> **备注:** 13 pages; AAAI26 version with appendix
>
> **摘要:** Regularized optimization has been a classical approach to solving imaging inverse problems, where the regularization term enforces desirable properties of the unknown image. Recently, the integration of flow matching generative models into image restoration has garnered significant attention, owing to their powerful prior modeling capabilities. In this work, we incorporate such generative priors into a Plug-and-Play (PnP) framework based on proximal splitting, where the proximal operator associated with the regularizer is replaced by a time-dependent denoiser derived from the generative model. While existing PnP methods have achieved notable success in inverse problems with smooth squared $\ell_2$ data fidelity--typically associated with Gaussian noise--their applicability to more general data fidelity terms remains underexplored. To address this, we propose a general and efficient PnP algorithm inspired by the primal-dual hybrid gradient (PDHG) method. Our approach is computationally efficient, memory-friendly, and accommodates a wide range of fidelity terms. In particular, it supports both $\ell_1$ and $\ell_2$ norm-based losses, enabling robustness to non-Gaussian noise types such as Poisson and impulse noise. We validate our method on several image restoration tasks, including denoising, super-resolution, deblurring, and inpainting, and demonstrate that $\ell_1$ and $\ell_2$ fidelity terms outperform the conventional squared $\ell_2$ loss in the presence of non-Gaussian noise.
>
---
#### [new 181] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation
- **分类: cs.CV**

- **简介: 该论文针对开放词汇场景图生成（OVSGG）中因缺乏交互建模导致的误匹配问题，提出交互中心框架ACC，通过交互引导的知识注入与迁移，提升模型对交互对象的识别与关系建模能力，实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.05935v1](http://arxiv.org/pdf/2511.05935v1)**

> **作者:** Lin Li; Chuhan Zhang; Dong Zhang; Chong Sun; Chen Li; Long Chen
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Open-vocabulary scene graph generation (OVSGG) extends traditional SGG by recognizing novel objects and relationships beyond predefined categories, leveraging the knowledge from pre-trained large-scale models. Existing OVSGG methods always adopt a two-stage pipeline: 1) \textit{Infusing knowledge} into large-scale models via pre-training on large datasets; 2) \textit{Transferring knowledge} from pre-trained models with fully annotated scene graphs during supervised fine-tuning. However, due to a lack of explicit interaction modeling, these methods struggle to distinguish between interacting and non-interacting instances of the same object category. This limitation induces critical issues in both stages of OVSGG: it generates noisy pseudo-supervision from mismatched objects during knowledge infusion, and causes ambiguous query matching during knowledge transfer. To this end, in this paper, we propose an inter\textbf{AC}tion-\textbf{C}entric end-to-end OVSGG framework (\textbf{ACC}) in an interaction-driven paradigm to minimize these mismatches. For \textit{interaction-centric knowledge infusion}, ACC employs a bidirectional interaction prompt for robust pseudo-supervision generation to enhance the model's interaction knowledge. For \textit{interaction-centric knowledge transfer}, ACC first adopts interaction-guided query selection that prioritizes pairing interacting objects to reduce interference from non-interacting ones. Then, it integrates interaction-consistent knowledge distillation to bolster robustness by pushing relational foreground away from the background while retaining general knowledge. Extensive experimental results on three benchmarks show that ACC achieves state-of-the-art performance, demonstrating the potential of interaction-centric paradigms for real-world applications.
>
---
#### [new 182] Pedicle Screw Pairing and Registration for Screw Pose Estimation from Dual C-arm Images Using CAD Models
- **分类: cs.CV**

- **简介: 该论文针对脊柱手术中双C臂影像的椎弓根螺钉配准与位姿估计问题，提出基于CAD模型的2D-3D对齐方法，通过对比螺钉组合实现准确配对与姿态估算，显著降低投影误差，提升手术定位精度。**

- **链接: [http://arxiv.org/pdf/2511.05702v1](http://arxiv.org/pdf/2511.05702v1)**

> **作者:** Yehyun Suh; Lin Li; Aric Plumley; Chaochao Zhou; Daniel Moyer; Kongbin Kang
>
> **摘要:** Accurate matching of pedicle screws in both anteroposterior (AP) and lateral (LAT) images is critical for successful spinal decompression and stabilization during surgery. However, establishing screw correspondence, especially in LAT views, remains a significant clinical challenge. This paper introduces a method to address pedicle screw correspondence and pose estimation from dual C-arm images. By comparing screw combinations, the approach demonstrates consistent accuracy in both pairing and registration tasks. The method also employs 2D-3D alignment with screw CAD 3D models to accurately pair and estimate screw pose from dual views. Our results show that the correct screw combination consistently outperforms incorrect pairings across all test cases, even prior to registration. After registration, the correct combination further enhances alignment between projections and images, significantly reducing projection error. This approach shows promise for improving surgical outcomes in spinal procedures by providing reliable feedback on screw positioning.
>
---
#### [new 183] TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks
- **分类: cs.CV**

- **简介: TinyChemVL提出一种高效化学视觉语言模型，通过视觉令牌压缩与反应级任务提升效率与推理能力，解决传统模型计算冗余和任务单一问题，并构建ChemRxn-V基准，实现以1/16令牌量超越现有模型。**

- **链接: [http://arxiv.org/pdf/2511.06283v1](http://arxiv.org/pdf/2511.06283v1)**

> **作者:** Xuanle Zhao; Shuxin Zeng; Yinyuan Cai; Xiang Cheng; Duzhen Zhang; Xiuyi Chen; Bo Xu
>
> **备注:** Accepted by AAAI 2026, Preprint Version
>
> **摘要:** While Vision Language Models (VLMs) have demonstrated remarkable capabilities in general visual understanding, their application in the chemical domain has been limited, with previous works predominantly focusing on text and thus overlooking critical visual information, such as molecular structures. Current approaches that directly adopt standard VLMs for chemical tasks suffer from two primary issues: (i) computational inefficiency of processing entire chemical images with non-informative backgrounds. (ii) a narrow scope on molecular-level tasks that restricts progress in chemical reasoning. In this work, we propose \textbf{TinyChemVL}, an efficient and powerful chemical VLM that leverages visual token reduction and reaction-level tasks to improve model efficiency and reasoning capacity. Also, we propose \textbf{ChemRxn-V}, a reaction-level benchmark for assessing vision-based reaction recognition and prediction tasks. Directly predicting reaction products from molecular images poses a non-trivial challenge, as it requires models to integrate both recognition and reasoning capacities. Our results demonstrate that with only 4B parameters, TinyChemVL achieves superior performance on both molecular and reaction tasks while demonstrating faster inference and training speeds compared to existing models. Notably, TinyChemVL outperforms ChemVLM while utilizing only 1/16th of the visual tokens. This work builds efficient yet powerful VLMs for chemical domains by co-designing model architecture and task complexity.
>
---
#### [new 184] InfoAffect: A Dataset for Affective Analysis of Infographics
- **分类: cs.CV**

- **简介: 该论文提出InfoAffect数据集，解决图文信息图情感分析数据匮乏问题，构建3.5k样本多模态标注数据，结合五种MLLM与RRF融合算法，实现高精度情感分析，用户验证CACI达0.986。**

- **链接: [http://arxiv.org/pdf/2511.06404v1](http://arxiv.org/pdf/2511.06404v1)**

> **作者:** Zihang Fu; Yunchao Wang; Chenyu Huang; Guodao Sun; Ronghua Liang
>
> **摘要:** Infographics are widely used to convey complex information, yet their affective dimensions remain underexplored due to the scarcity of data resources. We introduce a 3.5k-sample affect-annotated InfoAffect dataset, which combines textual content with real-world infographics. We first collect the raw data from six domains and aligned them via preprocessing, the accompanied-text-priority method, and three strategies to guarantee the quality and compliance. After that we construct an affect table and use it to constrain annotation. Five state-of-the-art multimodal large language models (MLLMs) then analyze both modalities, and their outputs are fused with Reciprocal Rank Fusion (RRF) algorithm to yield robust affects and confidences. We conducted a user study with two experiments to validate usability and assess InfoAffect dataset using the Composite Affect Consistency Index (CACI), achieving an overall score of 0.986, which indicates high accuracy.
>
---
#### [new 185] Causal Tracing of Object Representations in Large Vision Language Models: Mechanistic Interpretability and Hallucination Mitigation
- **分类: cs.CV**

- **简介: 该论文针对大视觉语言模型（LVLMs）的可解释性不足与幻觉问题，提出FCCT框架量化跨模态因果效应，发现关键组件作用，并设计无训练的IRI方法，在不牺牲速度下精准增强视觉表征，有效抑制幻觉。**

- **链接: [http://arxiv.org/pdf/2511.05923v1](http://arxiv.org/pdf/2511.05923v1)**

> **作者:** Qiming Li; Zekai Ye; Xiaocheng Feng; Weihong Zhong; Weitao Ma; Xiachong Feng
>
> **备注:** AAAI2026 Oral
>
> **摘要:** Despite the remarkable advancements of Large Vision-Language Models (LVLMs), the mechanistic interpretability remains underexplored. Existing analyses are insufficiently comprehensive and lack examination covering visual and textual tokens, model components, and the full range of layers. This limitation restricts actionable insights to improve the faithfulness of model output and the development of downstream tasks, such as hallucination mitigation. To address this limitation, we introduce Fine-grained Cross-modal Causal Tracing (FCCT) framework, which systematically quantifies the causal effects on visual object perception. FCCT conducts fine-grained analysis covering the full range of visual and textual tokens, three core model components including multi-head self-attention (MHSA), feed-forward networks (FFNs), and hidden states, across all decoder layers. Our analysis is the first to demonstrate that MHSAs of the last token in middle layers play a critical role in aggregating cross-modal information, while FFNs exhibit a three-stage hierarchical progression for the storage and transfer of visual object representations. Building on these insights, we propose Intermediate Representation Injection (IRI), a training-free inference-time technique that reinforces visual object information flow by precisely intervening on cross-modal representations at specific components and layers, thereby enhancing perception and mitigating hallucination. Consistent improvements across five widely used benchmarks and LVLMs demonstrate IRI achieves state-of-the-art performance, while preserving inference speed and other foundational performance.
>
---
#### [new 186] Argus: Quality-Aware High-Throughput Text-to-Image Inference Serving System
- **分类: cs.CV; cs.DC**

- **简介: Argus针对文本生成图像（T2I）推理延迟高、吞吐低的问题，提出一种质量感知的自适应推理系统，动态选择最优近似策略，在保证图像质量前提下提升吞吐量40%，降低延迟违规90%。**

- **链接: [http://arxiv.org/pdf/2511.06724v1](http://arxiv.org/pdf/2511.06724v1)**

> **作者:** Shubham Agarwal; Subrata Mitra; Saud Iqbal
>
> **备注:** Accepted at Middleware 2025
>
> **摘要:** Text-to-image (T2I) models have gained significant popularity. Most of these are diffusion models with unique computational characteristics, distinct from both traditional small-scale ML models and large language models. They are highly compute-bound and use an iterative denoising process to generate images, leading to very high inference time. This creates significant challenges in designing a high-throughput system. We discovered that a large fraction of prompts can be served using faster, approximated models. However, the approximation setting must be carefully calibrated for each prompt to avoid quality degradation. Designing a high-throughput system that assigns each prompt to the appropriate model and compatible approximation setting remains a challenging problem. We present Argus, a high-throughput T2I inference system that selects the right level of approximation for each prompt to maintain quality while meeting throughput targets on a fixed-size cluster. Argus intelligently switches between different approximation strategies to satisfy both throughput and quality requirements. Overall, Argus achieves 10x fewer latency service-level objective (SLO) violations, 10% higher average quality, and 40% higher throughput compared to baselines on two real-world workload traces.
>
---
#### [new 187] NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment
- **分类: cs.CV; cs.AI**

- **简介: 论文提出NeuroBridge，用于自监督EEG到图像解码，解决神经数据与视觉内容语义不匹配及标注数据稀缺问题。通过认知先验增强与双向语义对齐，实现跨模态高效学习，在零样本检索任务中显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06836v1](http://arxiv.org/pdf/2511.06836v1)**

> **作者:** Wenjiang Zhang; Sifeng Wang; Yuwei Su; Xinyu Li; Chen Zhang; Suyu Zhong
>
> **备注:** AAAI 2026
>
> **摘要:** Visual neural decoding seeks to reconstruct or infer perceived visual stimuli from brain activity patterns, providing critical insights into human cognition and enabling transformative applications in brain-computer interfaces and artificial intelligence. Current approaches, however, remain constrained by the scarcity of high-quality stimulus-brain response pairs and the inherent semantic mismatch between neural representations and visual content. Inspired by perceptual variability and co-adaptive strategy of the biological systems, we propose a novel self-supervised architecture, named NeuroBridge, which integrates Cognitive Prior Augmentation (CPA) with Shared Semantic Projector (SSP) to promote effective cross-modality alignment. Specifically, CPA simulates perceptual variability by applying asymmetric, modality-specific transformations to both EEG signals and images, enhancing semantic diversity. Unlike previous approaches, SSP establishes a bidirectional alignment process through a co-adaptive strategy, which mutually aligns features from two modalities into a shared semantic space for effective cross-modal learning. NeuroBridge surpasses previous state-of-the-art methods under both intra-subject and inter-subject settings. In the intra-subject scenario, it achieves the improvements of 12.3% in top-1 accuracy and 10.2% in top-5 accuracy, reaching 63.2% and 89.9% respectively on a 200-way zero-shot retrieval task. Extensive experiments demonstrate the effectiveness, robustness, and scalability of the proposed framework for neural visual decoding.
>
---
#### [new 188] SFFR: Spatial-Frequency Feature Reconstruction for Multispectral Aerial Object Detection
- **分类: cs.CV**

- **简介: 该论文针对多光谱无人机目标检测任务，提出SFFR方法，利用KAN网络在空域和频域重建互补特征，通过FCEKAN和MSGKAN模块增强跨模态一致性与尺度鲁棒性，显著提升检测性能。**

- **链接: [http://arxiv.org/pdf/2511.06298v1](http://arxiv.org/pdf/2511.06298v1)**

> **作者:** Xin Zuo; Yuchen Qu; Haibo Zhan; Jifeng Shen; Wankou Yang
>
> **备注:** 11 pages,8 figures, accepted by IEEE TGRS
>
> **摘要:** Recent multispectral object detection methods have primarily focused on spatial-domain feature fusion based on CNNs or Transformers, while the potential of frequency-domain feature remains underexplored. In this work, we propose a novel Spatial and Frequency Feature Reconstruction method (SFFR) method, which leverages the spatial-frequency feature representation mechanisms of the Kolmogorov-Arnold Network (KAN) to reconstruct complementary representations in both spatial and frequency domains prior to feature fusion. The core components of SFFR are the proposed Frequency Component Exchange KAN (FCEKAN) module and Multi-Scale Gaussian KAN (MSGKAN) module. The FCEKAN introduces an innovative selective frequency component exchange strategy that effectively enhances the complementarity and consistency of cross-modal features based on the frequency feature of RGB and IR images. The MSGKAN module demonstrates excellent nonlinear feature modeling capability in the spatial domain. By leveraging multi-scale Gaussian basis functions, it effectively captures the feature variations caused by scale changes at different UAV flight altitudes, significantly enhancing the model's adaptability and robustness to scale variations. It is experimentally validated that our proposed FCEKAN and MSGKAN modules are complementary and can effectively capture the frequency and spatial semantic features respectively for better feature fusion. Extensive experiments on the SeaDroneSee, DroneVehicle and DVTOD datasets demonstrate the superior performance and significant advantages of the proposed method in UAV multispectral object perception task. Code will be available at https://github.com/qchenyu1027/SFFR.
>
---
#### [new 189] Polymap: generating high definition map based on rasterized polygons
- **分类: cs.CV**

- **简介: 该论文提出Polymap，将高精地图构建任务转化为栅格多边形实例分割问题，通过分割Transformer生成掩码，再用Potrace转为矢量元素，提升检测方法的泛化性，适用于自动标注系统。**

- **链接: [http://arxiv.org/pdf/2511.05944v1](http://arxiv.org/pdf/2511.05944v1)**

> **作者:** Shiyu Gao; Hao Jiang
>
> **摘要:** The perception of high-definition maps is an integral component of environmental perception in autonomous driving systems. Existing research have often focused on online construction of high-definition maps. For instance, the Maptr[9] series employ a detection-based method to output vectorized map instances parallelly in an end-to-end manner. However, despite their capability for real-time construction, detection-based methods are observed to lack robust generalizability[19], which hampers their applicability in auto-labeling systems. Therefore, aiming to improve the generalizability, we reinterpret road elements as rasterized polygons and design a concise framework based on instance segmentation. Initially, a segmentation-based transformer is employed to deliver instance masks in an end-to-end manner; succeeding this step, a Potrace-based[17] post-processing module is used to ultimately yield vectorized map elements. Quantitative results attained on the Nuscene[1] dataset substantiate the effectiveness and generaliz-ability of our method.
>
---
#### [new 190] Integrating Reweighted Least Squares with Plug-and-Play Diffusion Priors for Noisy Image Restoration
- **分类: cs.CV**

- **简介: 该论文针对非高斯噪声（如脉冲噪声）图像恢复问题，提出融合重加权最小二乘与扩散先验的Plug-and-Play框架，通过广义高斯尺度混合损失替代传统平方损失，提升去噪鲁棒性与恢复性能。**

- **链接: [http://arxiv.org/pdf/2511.06823v1](http://arxiv.org/pdf/2511.06823v1)**

> **作者:** Ji Li; Chao Wang
>
> **备注:** 12 pages
>
> **摘要:** Existing plug-and-play image restoration methods typically employ off-the-shelf Gaussian denoisers as proximal operators within classical optimization frameworks based on variable splitting. Recently, denoisers induced by generative priors have been successfully integrated into regularized optimization methods for image restoration under Gaussian noise. However, their application to non-Gaussian noise--such as impulse noise--remains largely unexplored. In this paper, we propose a plug-and-play image restoration framework based on generative diffusion priors for robust removal of general noise types, including impulse noise. Within the maximum a posteriori (MAP) estimation framework, the data fidelity term is adapted to the specific noise model. Departing from the conventional least-squares loss used for Gaussian noise, we introduce a generalized Gaussian scale mixture-based loss, which approximates a wide range of noise distributions and leads to an $\ell_q$-norm ($0<q\leq2$) fidelity term. This optimization problem is addressed using an iteratively reweighted least squares (IRLS) approach, wherein the proximal step involving the generative prior is efficiently performed via a diffusion-based denoiser. Experimental results on benchmark datasets demonstrate that the proposed method effectively removes non-Gaussian impulse noise and achieves superior restoration performance.
>
---
#### [new 191] GFix: Perceptually Enhanced Gaussian Splatting Video Compression
- **分类: cs.CV**

- **简介: 论文提出GFix，面向3D高斯泼溅视频压缩的感知增强任务，解决渲染伪影与压缩率低问题，通过单步扩散模型去噪并结合调制LoRA实现高效自适应，显著提升视觉质量与压缩效率。**

- **链接: [http://arxiv.org/pdf/2511.06953v1](http://arxiv.org/pdf/2511.06953v1)**

> **作者:** Siyue Teng; Ge Gao; Duolikun Danier; Yuxuan Jiang; Fan Zhang; Thomas Davis; Zoe Liu; David Bull
>
> **摘要:** 3D Gaussian Splatting (3DGS) enhances 3D scene reconstruction through explicit representation and fast rendering, demonstrating potential benefits for various low-level vision tasks, including video compression. However, existing 3DGS-based video codecs generally exhibit more noticeable visual artifacts and relatively low compression ratios. In this paper, we specifically target the perceptual enhancement of 3DGS-based video compression, based on the assumption that artifacts from 3DGS rendering and quantization resemble noisy latents sampled during diffusion training. Building on this premise, we propose a content-adaptive framework, GFix, comprising a streamlined, single-step diffusion model that serves as an off-the-shelf neural enhancer. Moreover, to increase compression efficiency, We propose a modulated LoRA scheme that freezes the low-rank decompositions and modulates the intermediate hidden states, thereby achieving efficient adaptation of the diffusion backbone with highly compressible updates. Experimental results show that GFix delivers strong perceptual quality enhancement, outperforming GSVC with up to 72.1% BD-rate savings in LPIPS and 21.4% in FID.
>
---
#### [new 192] Learning from the Right Patches: A Two-Stage Wavelet-Driven Masked Autoencoder for Histopathology Representation Learning
- **分类: cs.CV**

- **简介: 该论文提出WISE-MAE，一种面向组织病理学的两阶段小波引导掩码自编码器，解决传统MAE随机采样噪声干扰问题，通过低倍率小波筛选关键区域，再高倍率建模，提升无监督表征学习质量与下游分类性能。**

- **链接: [http://arxiv.org/pdf/2511.06958v1](http://arxiv.org/pdf/2511.06958v1)**

> **作者:** Raneen Younis; Louay Hamdi; Lukas Chavez; Zahra Ahmadi
>
> **摘要:** Whole-slide images are central to digital pathology, yet their extreme size and scarce annotations make self-supervised learning essential. Masked Autoencoders (MAEs) with Vision Transformer backbones have recently shown strong potential for histopathology representation learning. However, conventional random patch sampling during MAE pretraining often includes irrelevant or noisy regions, limiting the model's ability to capture meaningful tissue patterns. In this paper, we present a lightweight and domain-adapted framework that brings structure and biological relevance into MAE-based learning through a wavelet-informed patch selection strategy. WISE-MAE applies a two-step coarse-to-fine process: wavelet-based screening at low magnification to locate structurally rich regions, followed by high-resolution extraction for detailed modeling. This approach mirrors the diagnostic workflow of pathologists and improves the quality of learned representations. Evaluations across multiple cancer datasets, including lung, renal, and colorectal tissues, show that WISE-MAE achieves competitive representation quality and downstream classification performance while maintaining efficiency under weak supervision.
>
---
#### [new 193] MACMD: Multi-dilated Contextual Attention and Channel Mixer Decoding for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割中局部细节丢失与全局上下文融合不足的问题，提出MACMD解码器，结合多尺度空洞注意力与通道混合模块，增强编码器-解码器间信息交互，在保持高精度的同时提升效率。**

- **链接: [http://arxiv.org/pdf/2511.05803v1](http://arxiv.org/pdf/2511.05803v1)**

> **作者:** Lalit Maurya; Honghai Liu; Reyer Zwiggelaar
>
> **摘要:** Medical image segmentation faces challenges due to variations in anatomical structures. While convolutional neural networks (CNNs) effectively capture local features, they struggle with modeling long-range dependencies. Transformers mitigate this issue with self-attention mechanisms but lack the ability to preserve local contextual information. State-of-the-art models primarily follow an encoder-decoder architecture, achieving notable success. However, two key limitations remain: (1) Shallow layers, which are closer to the input, capture fine-grained details but suffer from information loss as data propagates through deeper layers. (2) Inefficient integration of local details and global context between the encoder and decoder stages. To address these challenges, we propose the MACMD-based decoder, which enhances attention mechanisms and facilitates channel mixing between encoder and decoder stages via skip connections. This design leverages hierarchical dilated convolutions, attention-driven modulation, and a cross channel-mixing module to capture long-range dependencies while preserving local contextual details, essential for precise medical image segmentation. We evaluated our approach using multiple transformer encoders on both binary and multi-organ segmentation tasks. The results demonstrate that our method outperforms state-of-the-art approaches in terms of Dice score and computational efficiency, highlighting its effectiveness in achieving accurate and robust segmentation performance. The code available at https://github.com/lalitmaurya47/MACMD
>
---
#### [new 194] PointCubeNet: 3D Part-level Reasoning with 3x3x3 Point Cloud Blocks
- **分类: cs.CV**

- **简介: PointCubeNet提出一种无监督3D部分级理解框架，通过3x3x3点云块实现局部语义推理，无需部分标注，结合伪标签与局部损失函数，首次实现无监督3D部件分析，提升整体理解能力。**

- **链接: [http://arxiv.org/pdf/2511.06744v1](http://arxiv.org/pdf/2511.06744v1)**

> **作者:** Da-Yeong Kim; Yeong-Jun Cho
>
> **摘要:** In this paper, we propose PointCubeNet, a novel multi-modal 3D understanding framework that achieves part-level reasoning without requiring any part annotations. PointCubeNet comprises global and local branches. The proposed local branch, structured into 3x3x3 local blocks, enables part-level analysis of point cloud sub-regions with the corresponding local text labels. Leveraging the proposed pseudo-labeling method and local loss function, PointCubeNet is effectively trained in an unsupervised manner. The experimental results demonstrate that understanding 3D object parts enhances the understanding of the overall 3D object. In addition, this is the first attempt to perform unsupervised 3D part-level reasoning and achieves reliable and meaningful results.
>
---
#### [new 195] Flexible Concept Bottleneck Model
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出柔性概念瓶颈模型（FCBM），解决传统CBM需重训才能适应新概念的问题。通过超网络生成权重与可学习稀疏max模块，实现无需重训的动态概念替换与选择，提升模型灵活性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.06678v1](http://arxiv.org/pdf/2511.06678v1)**

> **作者:** Xingbo Du; Qiantong Dou; Lei Fan; Rui Zhang
>
> **备注:** To appear in AAAI 2026
>
> **摘要:** Concept bottleneck models (CBMs) improve neural network interpretability by introducing an intermediate layer that maps human-understandable concepts to predictions. Recent work has explored the use of vision-language models (VLMs) to automate concept selection and annotation. However, existing VLM-based CBMs typically require full model retraining when new concepts are involved, which limits their adaptability and flexibility in real-world scenarios, especially considering the rapid evolution of vision-language foundation models. To address these issues, we propose Flexible Concept Bottleneck Model (FCBM), which supports dynamic concept adaptation, including complete replacement of the original concept set. Specifically, we design a hypernetwork that generates prediction weights based on concept embeddings, allowing seamless integration of new concepts without retraining the entire model. In addition, we introduce a modified sparsemax module with a learnable temperature parameter that dynamically selects the most relevant concepts, enabling the model to focus on the most informative features. Extensive experiments on five public benchmarks demonstrate that our method achieves accuracy comparable to state-of-the-art baselines with a similar number of effective concepts. Moreover, the model generalizes well to unseen concepts with just a single epoch of fine-tuning, demonstrating its strong adaptability and flexibility.
>
---
#### [new 196] Seq2Seq Models Reconstruct Visual Jigsaw Puzzles without Seeing Them
- **分类: cs.CV**

- **简介: 该论文提出用语言模型解决视觉拼图问题，无需视觉输入。通过将拼图块编码为离散序列，将拼图重组转化为Seq2Seq任务，基于Transformer的模型仅凭文本序列即可高精度还原布局，超越传统视觉方法。**

- **链接: [http://arxiv.org/pdf/2511.06315v1](http://arxiv.org/pdf/2511.06315v1)**

> **作者:** Gur Elkn; Ofir Itzhak Shahar; Ohad Ben-Shahar
>
> **摘要:** Jigsaw puzzles are primarily visual objects, whose algorithmic solutions have traditionally been framed from a visual perspective. In this work, however, we explore a fundamentally different approach: solving square jigsaw puzzles using language models, without access to raw visual input. By introducing a specialized tokenizer that converts each puzzle piece into a discrete sequence of tokens, we reframe puzzle reassembly as a sequence-to-sequence prediction task. Treated as "blind" solvers, encoder-decoder transformers accurately reconstruct the original layout by reasoning over token sequences alone. Despite being deliberately restricted from accessing visual input, our models achieve state-of-the-art results across multiple benchmarks, often outperforming vision-based methods. These findings highlight the surprising capability of language models to solve problems beyond their native domain, and suggest that unconventional approaches can inspire promising directions for puzzle-solving research.
>
---
#### [new 197] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对模糊医学图像分割任务，提出ATFM方法，通过数据分层推理、高斯截断表示和分割流匹配，解耦预测精度与多样性，提升结果的保真度与合理性，显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06857v1](http://arxiv.org/pdf/2511.06857v1)**

> **作者:** Fanding Li; Xiangyu Li; Xianghe Su; Xingyu Qiu; Suyu Dong; Wei Wang; Kuanquan Wang; Gongning Luo; Shuo Li
>
> **备注:** 13 pages, 10 figures, extended version of AAAI-26 paper
>
> **摘要:** A simultaneous enhancement of accuracy and diversity of predictions remains a challenge in ambiguous medical image segmentation (AMIS) due to the inherent trade-offs. While truncated diffusion probabilistic models (TDPMs) hold strong potential with a paradigm optimization, existing TDPMs suffer from entangled accuracy and diversity of predictions with insufficient fidelity and plausibility. To address the aforementioned challenges, we propose Ambiguity-aware Truncated Flow Matching (ATFM), which introduces a novel inference paradigm and dedicated model components. Firstly, we propose Data-Hierarchical Inference, a redefinition of AMIS-specific inference paradigm, which enhances accuracy and diversity at data-distribution and data-sample level, respectively, for an effective disentanglement. Secondly, Gaussian Truncation Representation (GTR) is introduced to enhance both fidelity of predictions and reliability of truncation distribution, by explicitly modeling it as a Gaussian distribution at $T_{\text{trunc}}$ instead of using sampling-based approximations.Thirdly, Segmentation Flow Matching (SFM) is proposed to enhance the plausibility of diverse predictions by extending semantic-aware flow transformation in Flow Matching (FM). Comprehensive evaluations on LIDC and ISIC3 datasets demonstrate that ATFM outperforms SOTA methods and simultaneously achieves a more efficient inference. ATFM improves GED and HM-IoU by up to $12\%$ and $7.3\%$ compared to advanced methods.
>
---
#### [new 198] RaLD: Generating High-Resolution 3D Radar Point Clouds with Latent Diffusion
- **分类: cs.CV**

- **简介: RaLD提出一种基于潜在扩散模型的雷达点云生成框架，解决毫米波雷达点云稀疏、分辨率低的问题，通过LiDAR自编码与频谱条件引导，高效生成高分辨率稠密3D点云。**

- **链接: [http://arxiv.org/pdf/2511.07067v1](http://arxiv.org/pdf/2511.07067v1)**

> **作者:** Ruijie Zhang; Bixin Zeng; Shengpeng Wang; Fuhui Zhou; Wei Wang
>
> **摘要:** Millimeter-wave radar offers a promising sensing modality for autonomous systems thanks to its robustness in adverse conditions and low cost. However, its utility is significantly limited by the sparsity and low resolution of radar point clouds, which poses challenges for tasks requiring dense and accurate 3D perception. Despite that recent efforts have shown great potential by exploring generative approaches to address this issue, they often rely on dense voxel representations that are inefficient and struggle to preserve structural detail. To fill this gap, we make the key observation that latent diffusion models (LDMs), though successful in other modalities, have not been effectively leveraged for radar-based 3D generation due to a lack of compatible representations and conditioning strategies. We introduce RaLD, a framework that bridges this gap by integrating scene-level frustum-based LiDAR autoencoding, order-invariant latent representations, and direct radar spectrum conditioning. These insights lead to a more compact and expressive generation process. Experiments show that RaLD produces dense and accurate 3D point clouds from raw radar spectrums, offering a promising solution for robust perception in challenging environments.
>
---
#### [new 199] DiffusionUavLoc: Visually Prompted Diffusion for Cross-View UAV Localization
- **分类: cs.CV**

- **简介: 论文提出DiffusionUavLoc，面向无GNSS环境下的跨视角无人机定位，通过几何渲染生成伪卫星图作为视觉提示，构建无文本的扩散模型，实现无人机与卫星图像的鲁棒匹配，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2511.06422v1](http://arxiv.org/pdf/2511.06422v1)**

> **作者:** Tao Liu; Kan Ren; Qian Chen
>
> **摘要:** With the rapid growth of the low-altitude economy, unmanned aerial vehicles (UAVs) have become key platforms for measurement and tracking in intelligent patrol systems. However, in GNSS-denied environments, localization schemes that rely solely on satellite signals are prone to failure. Cross-view image retrieval-based localization is a promising alternative, yet substantial geometric and appearance domain gaps exist between oblique UAV views and nadir satellite orthophotos. Moreover, conventional approaches often depend on complex network architectures, text prompts, or large amounts of annotation, which hinders generalization. To address these issues, we propose DiffusionUavLoc, a cross-view localization framework that is image-prompted, text-free, diffusion-centric, and employs a VAE for unified representation. We first use training-free geometric rendering to synthesize pseudo-satellite images from UAV imagery as structural prompts. We then design a text-free conditional diffusion model that fuses multimodal structural cues to learn features robust to viewpoint changes. At inference, descriptors are computed at a fixed time step t and compared using cosine similarity. On University-1652 and SUES-200, the method performs competitively for cross-view localization, especially for satellite-to-drone in University-1652.Our data and code will be published at the following URL: https://github.com/liutao23/DiffusionUavLoc.git.
>
---
#### [new 200] StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation
- **分类: cs.CV; cs.LG**

- **简介: StreamDiffusionV2提出一种无需训练的实时视频生成系统，解决视频扩散模型在直播场景下延迟高、帧间不一致、多GPU扩展难等问题，通过动态调度与缓存优化，实现低延迟、高帧率的交互式视频流生成。**

- **链接: [http://arxiv.org/pdf/2511.07399v1](http://arxiv.org/pdf/2511.07399v1)**

> **作者:** Tianrui Feng; Zhi Li; Shuo Yang; Haocheng Xi; Muyang Li; Xiuyu Li; Lvmin Zhang; Keting Yang; Kelly Peng; Song Han; Maneesh Agrawala; Kurt Keutzer; Akio Kodaira; Chenfeng Xu
>
> **备注:** Project Page: http://streamdiffusionv2.github.io
>
> **摘要:** Generative models are reshaping the live-streaming industry by redefining how content is created, styled, and delivered. Previous image-based streaming diffusion models have powered efficient and creative live streaming products but have hit limits on temporal consistency due to the foundation of image-based designs. Recent advances in video diffusion have markedly improved temporal consistency and sampling efficiency for offline generation. However, offline generation systems primarily optimize throughput by batching large workloads. In contrast, live online streaming operates under strict service-level objectives (SLOs): time-to-first-frame must be minimal, and every frame must meet a per-frame deadline with low jitter. Besides, scalable multi-GPU serving for real-time streams remains largely unresolved so far. To address this, we present StreamDiffusionV2, a training-free pipeline for interactive live streaming with video diffusion models. StreamDiffusionV2 integrates an SLO-aware batching scheduler and a block scheduler, together with a sink-token--guided rolling KV cache, a motion-aware noise controller, and other system-level optimizations. Moreover, we introduce a scalable pipeline orchestration that parallelizes the diffusion process across denoising steps and network layers, achieving near-linear FPS scaling without violating latency guarantees. The system scales seamlessly across heterogeneous GPU environments and supports flexible denoising steps (e.g., 1--4), enabling both ultra-low-latency and higher-quality modes. Without TensorRT or quantization, StreamDiffusionV2 renders the first frame within 0.5s and attains 58.28 FPS with a 14B-parameter model and 64.52 FPS with a 1.3B-parameter model on four H100 GPUs, making state-of-the-art generative live streaming practical and accessible--from individual creators to enterprise-scale platforms.
>
---
#### [new 201] Enhancing Diffusion Model Guidance through Calibration and Regularization
- **分类: cs.CV; cs.AI; cs.IT; cs.LG; eess.IV; math.IT; I.2.6; I.2.7; I.5.1**

- **简介: 该论文针对分类器引导扩散模型在去噪初期梯度消失问题，提出可微校准目标与无重训采样方法，通过校准与散度正则化提升生成质量，在ImageNet上实现SOTA FID 2.13。**

- **链接: [http://arxiv.org/pdf/2511.05844v1](http://arxiv.org/pdf/2511.05844v1)**

> **作者:** Seyed Alireza Javid; Amirhossein Bagheri; Nuria González-Prelcic
>
> **备注:** Accepted from NeurIPS 2025 Workshop on Structured Probabilistic Inference & Generative Modeling. Code available at https://github.com/ajavid34/guided-info-diffusion
>
> **摘要:** Classifier-guided diffusion models have emerged as a powerful approach for conditional image generation, but they suffer from overconfident predictions during early denoising steps, causing the guidance gradient to vanish. This paper introduces two complementary contributions to address this issue. First, we propose a differentiable calibration objective based on the Smooth Expected Calibration Error (Smooth ECE), which improves classifier calibration with minimal fine-tuning and yields measurable improvements in Frechet Inception Distance (FID). Second, we develop enhanced sampling guidance methods that operate on off-the-shelf classifiers without requiring retraining. These include tilted sampling with batch-level reweighting, adaptive entropy-regularized sampling to preserve diversity, and a novel f-divergence-based sampling strategy that strengthens class-consistent guidance while maintaining mode coverage. Experiments on ImageNet 128x128 demonstrate that our divergence-regularized guidance achieves an FID of 2.13 using a ResNet-101 classifier, improving upon existing classifier-guided diffusion methods while requiring no diffusion model retraining. The results show that principled calibration and divergence-aware sampling provide practical and effective improvements for classifier-guided diffusion.
>
---
#### [new 202] Adaptive 3D Reconstruction via Diffusion Priors and Forward Curvature-Matching Likelihood Updates
- **分类: cs.CV**

- **简介: 该论文针对单视图/多视图三维点云重建问题，提出融合扩散先验与自适应前向曲率匹配更新（FCM）的方法，无需重训练即可动态优化重建步长，提升质量与效率，支持多模态输入。**

- **链接: [http://arxiv.org/pdf/2511.06310v1](http://arxiv.org/pdf/2511.06310v1)**

> **作者:** Seunghyeok Shin; Dabin Kim; Hongki Lim
>
> **摘要:** Reconstructing high-quality point clouds from images remains challenging in computer vision. Existing generative-model-based approaches, particularly diffusion-model approaches that directly learn the posterior, may suffer from inflexibility -- they require conditioning signals during training, support only a fixed number of input views, and need complete retraining for different measurements. Recent diffusion-based methods have attempted to address this by combining prior models with likelihood updates, but they rely on heuristic fixed step sizes for the likelihood update that lead to slow convergence and suboptimal reconstruction quality. We advance this line of approach by integrating our novel Forward Curvature-Matching (FCM) update method with diffusion sampling. Our method dynamically determines optimal step sizes using only forward automatic differentiation and finite-difference curvature estimates, enabling precise optimization of the likelihood update. This formulation enables high-fidelity reconstruction from both single-view and multi-view inputs, and supports various input modalities through simple operator substitution -- all without retraining. Experiments on ShapeNet and CO3D datasets demonstrate that our method achieves superior reconstruction quality at matched or lower NFEs, yielding higher F-score and lower CD and EMD, validating its efficiency and adaptability for practical applications. Code is available at https://github.com/Seunghyeok0715/FCM
>
---
#### [new 203] DTTNet: Improving Video Shadow Detection via Dark-Aware Guidance and Tokenized Temporal Modeling
- **分类: cs.CV**

- **简介: 该论文针对视频阴影检测中阴影与暗背景混淆、动态变形难建模的问题，提出DTTNet，结合语言引导的暗区感知模块与分词时序建模，实现高精度实时阴影检测。**

- **链接: [http://arxiv.org/pdf/2511.06925v1](http://arxiv.org/pdf/2511.06925v1)**

> **作者:** Zhicheng Li; Kunyang Sun; Rui Yao; Hancheng Zhu; Fuyuan Hu; Jiaqi Zhao; Zhiwen Shao; Yong Zhou
>
> **摘要:** Video shadow detection confronts two entwined difficulties: distinguishing shadows from complex backgrounds and modeling dynamic shadow deformations under varying illumination. To address shadow-background ambiguity, we leverage linguistic priors through the proposed Vision-language Match Module (VMM) and a Dark-aware Semantic Block (DSB), extracting text-guided features to explicitly differentiate shadows from dark objects. Furthermore, we introduce adaptive mask reweighting to downweight penumbra regions during training and apply edge masks at the final decoder stage for better supervision. For temporal modeling of variable shadow shapes, we propose a Tokenized Temporal Block (TTB) that decouples spatiotemporal learning. TTB summarizes cross-frame shadow semantics into learnable temporal tokens, enabling efficient sequence encoding with minimal computation overhead. Comprehensive Experiments on multiple benchmark datasets demonstrate state-of-the-art accuracy and real-time inference efficiency. Codes are available at https://github.com/city-cheng/DTTNet.
>
---
#### [new 204] FoCLIP: A Feature-Space Misalignment Framework for CLIP-Based Image Manipulation and Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文提出FoCLIP框架，通过特征空间对齐误导CLIP图像质量评分，生成高CLIPscore但视觉失真的图像，并据此设计基于色彩通道敏感性的检测方法，准确率达91%，解决CLIP度量易被对抗攻击的问题。**

- **链接: [http://arxiv.org/pdf/2511.06947v1](http://arxiv.org/pdf/2511.06947v1)**

> **作者:** Yulin Chen; Zeyuan Wang; Tianyuan Yu; Yingmei Wei; Liang Bai
>
> **备注:** 15 page, 9 figures, published to PRCV
>
> **摘要:** The well-aligned attribute of CLIP-based models enables its effective application like CLIPscore as a widely adopted image quality assessment metric. However, such a CLIP-based metric is vulnerable for its delicate multimodal alignment. In this work, we propose \textbf{FoCLIP}, a feature-space misalignment framework for fooling CLIP-based image quality metric. Based on the stochastic gradient descent technique, FoCLIP integrates three key components to construct fooling examples: feature alignment as the core module to reduce image-text modality gaps, the score distribution balance module and pixel-guard regularization, which collectively optimize multimodal output equilibrium between CLIPscore performance and image quality. Such a design can be engineered to maximize the CLIPscore predictions across diverse input prompts, despite exhibiting either visual unrecognizability or semantic incongruence with the corresponding adversarial prompts from human perceptual perspectives. Experiments on ten artistic masterpiece prompts and ImageNet subsets demonstrate that optimized images can achieve significant improvement in CLIPscore while preserving high visual fidelity. In addition, we found that grayscale conversion induces significant feature degradation in fooling images, exhibiting noticeable CLIPscore reduction while preserving statistical consistency with original images. Inspired by this phenomenon, we propose a color channel sensitivity-driven tampering detection mechanism that achieves 91% accuracy on standard benchmarks. In conclusion, this work establishes a practical pathway for feature misalignment in CLIP-based multimodal systems and the corresponding defense method.
>
---
#### [new 205] Adaptive Morph-Patch Transformer for Arotic Vessel Segmentation
- **分类: cs.CV**

- **简介: 该论文针对主动脉血管分割任务，提出自适应形态补丁变换器（MPT），通过动态生成与血管形态对齐的补丁及语义聚类注意力机制，解决传统固定补丁破坏血管结构完整性的问题，显著提升分割精度。**

- **链接: [http://arxiv.org/pdf/2511.06897v1](http://arxiv.org/pdf/2511.06897v1)**

> **作者:** Zhenxi Zhang; Fuchen Zheng; Adnan Iltaf; Yifei Han; Zhenyu Cheng; Yue Du; Bin Li; Tianyong Liu; Shoujun Zhou
>
> **备注:** This is the preprint version of a paper accepted by AAAI 2026. The final version will appear in the AAAI Proceedings
>
> **摘要:** Accurate segmentation of aortic vascular structures is critical for diagnosing and treating cardiovascular diseases.Traditional Transformer-based models have shown promise in this domain by capturing long-range dependencies between vascular features. However, their reliance on fixed-size rectangular patches often influences the integrity of complex vascular structures, leading to suboptimal segmentation accuracy. To address this challenge, we propose the adaptive Morph Patch Transformer (MPT), a novel architecture specifically designed for aortic vascular segmentation. Specifically, MPT introduces an adaptive patch partitioning strategy that dynamically generates morphology-aware patches aligned with complex vascular structures. This strategy can preserve semantic integrity of complex vascular structures within individual patches. Moreover, a Semantic Clustering Attention (SCA) method is proposed to dynamically aggregate features from various patches with similar semantic characteristics. This method enhances the model's capability to segment vessels of varying sizes, preserving the integrity of vascular structures. Extensive experiments on three open-source dataset(AVT, AortaSeg24 and TBAD) demonstrate that MPT achieves state-of-the-art performance, with improvements in segmenting intricate vascular structures.
>
---
#### [new 206] Robust and High-Fidelity 3D Gaussian Splatting: Fusing Pose Priors and Geometry Constraints for Texture-Deficient Outdoor Scenes
- **分类: cs.CV; cs.GR**

- **简介: 该论文面向纹理稀疏的户外场景，提升3D高斯泼溅的位姿估计与场景重建质量。融合LiDAR-IMU先验位姿与法向约束，优化位姿稳定性与高斯形状一致性，显著提升重建精度与效率。**

- **链接: [http://arxiv.org/pdf/2511.06765v1](http://arxiv.org/pdf/2511.06765v1)**

> **作者:** Meijun Guo; Yongliang Shi; Caiyun Liu; Yixiao Feng; Ming Ma; Tinghai Yan; Weining Lu; Bin Liang
>
> **备注:** 7 pages, 3 figures. Accepted by IROS 2025
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a key rendering pipeline for digital asset creation due to its balance between efficiency and visual quality. To address the issues of unstable pose estimation and scene representation distortion caused by geometric texture inconsistency in large outdoor scenes with weak or repetitive textures, we approach the problem from two aspects: pose estimation and scene representation. For pose estimation, we leverage LiDAR-IMU Odometry to provide prior poses for cameras in large-scale environments. These prior pose constraints are incorporated into COLMAP's triangulation process, with pose optimization performed via bundle adjustment. Ensuring consistency between pixel data association and prior poses helps maintain both robustness and accuracy. For scene representation, we introduce normal vector constraints and effective rank regularization to enforce consistency in the direction and shape of Gaussian primitives. These constraints are jointly optimized with the existing photometric loss to enhance the map quality. We evaluate our approach using both public and self-collected datasets. In terms of pose optimization, our method requires only one-third of the time while maintaining accuracy and robustness across both datasets. In terms of scene representation, the results show that our method significantly outperforms conventional 3DGS pipelines. Notably, on self-collected datasets characterized by weak or repetitive textures, our approach demonstrates enhanced visualization capabilities and achieves superior overall performance. Codes and data will be publicly available at https://github.com/justinyeah/normal_shape.git.
>
---
#### [new 207] Med-SORA: Symptom to Organ Reasoning in Abdomen CT Images
- **分类: cs.CV**

- **简介: Med-SORA提出首个腹部CT症状-器官推理框架，解决传统模型依赖硬标签、忽略3D上下文的问题，通过RAG构建数据、软标签与2D-3D交叉注意力，实现多器官症状关联的精准推理。**

- **链接: [http://arxiv.org/pdf/2511.06752v1](http://arxiv.org/pdf/2511.06752v1)**

> **作者:** You-Kyoung Na; Yeong-Jun Cho
>
> **备注:** 9 pages
>
> **摘要:** Understanding symptom-image associations is crucial for clinical reasoning. However, existing medical multimodal models often rely on simple one-to-one hard labeling, oversimplifying clinical reality where symptoms relate to multiple organs. In addition, they mainly use single-slice 2D features without incorporating 3D information, limiting their ability to capture full anatomical context. In this study, we propose Med-SORA, a framework for symptom-to-organ reasoning in abdominal CT images. Med-SORA introduces RAG-based dataset construction, soft labeling with learnable organ anchors to capture one-to-many symptom-organ relationships, and a 2D-3D cross-attention architecture to fuse local and global image features. To our knowledge, this is the first work to address symptom-to-organ reasoning in medical multimodal learning. Experimental results show that Med-SORA outperforms existing medical multimodal models and enables accurate 3D clinical reasoning.
>
---
#### [new 208] Automatic Extraction of Road Networks by using Teacher-Student Adaptive Structural Deep Belief Network and Its Application to Landslide Disaster
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出基于师生自适应深度信念网络的RoadTracer模型，用于从航拍图像自动提取道路网络，解决复杂场景下检测精度低的问题，并应用于滑坡后应急道路识别，实现轻量化边缘部署。**

- **链接: [http://arxiv.org/pdf/2511.05567v1](http://arxiv.org/pdf/2511.05567v1)**

> **作者:** Shin Kamada; Takumi Ichimura
>
> **摘要:** An adaptive structural learning method of Restricted Boltzmann Machine (RBM) and Deep Belief Network (DBN) has been developed as one of prominent deep learning models. The neuron generation-annihilation algorithm in RBM and layer generation algorithm in DBN make an optimal network structure for given input during the learning. In this paper, our model is applied to an automatic recognition method of road network system, called RoadTracer. RoadTracer can generate a road map on the ground surface from aerial photograph data. A novel method of RoadTracer using the Teacher-Student based ensemble learning model of Adaptive DBN is proposed, since the road maps contain many complicated features so that a model with high representation power to detect should be required. The experimental results showed the detection accuracy of the proposed model was improved from 40.0\% to 89.0\% on average in the seven major cities among the test dataset. In addition, we challenged to apply our method to the detection of available roads when landslide by natural disaster is occurred, in order to rapidly obtain a way of transportation. For fast inference, a small size of the trained model was implemented on a small embedded edge device as lightweight deep learning. We reported the detection results for the satellite image before and after the rainfall disaster in Japan.
>
---
#### [new 209] Aerial Image Stitching Using IMU Data from a UAV
- **分类: cs.CV; cs.RO; cs.SY; eess.SY; math.DS**

- **简介: 该论文提出一种融合IMU数据与计算机视觉的无人机航拍图像拼接方法，解决传统特征匹配在大位移、姿态变化下易失败的问题，通过IMU估算位姿、校正畸变并计算单应矩阵，提升拼接精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.06841v1](http://arxiv.org/pdf/2511.06841v1)**

> **作者:** Selim Ahmet Iz; Mustafa Unel
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are widely used for aerial photography and remote sensing applications. One of the main challenges is to stitch together multiple images into a single high-resolution image that covers a large area. Featurebased image stitching algorithms are commonly used but can suffer from errors and ambiguities in feature detection and matching. To address this, several approaches have been proposed, including using bundle adjustment techniques or direct image alignment. In this paper, we present a novel method that uses a combination of IMU data and computer vision techniques for stitching images captured by a UAV. Our method involves several steps such as estimating the displacement and rotation of the UAV between consecutive images, correcting for perspective distortion, and computing a homography matrix. We then use a standard image stitching algorithm to align and blend the images together. Our proposed method leverages the additional information provided by the IMU data, corrects for various sources of distortion, and can be easily integrated into existing UAV workflows. Our experiments demonstrate the effectiveness and robustness of our method, outperforming some of the existing feature-based image stitching algorithms in terms of accuracy and reliability, particularly in challenging scenarios such as large displacements, rotations, and variations in camera pose.
>
---
#### [new 210] Google-MedGemma Based Abnormality Detection in Musculoskeletal radiographs
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出基于Google MedGemma的骨骼X光异常检测框架，利用其预训练视觉编码器与轻量MLP实现高精度二分类，突破传统自编码器方法，提升泛化能力与训练效率，助力医学影像自动化筛查。**

- **链接: [http://arxiv.org/pdf/2511.05600v1](http://arxiv.org/pdf/2511.05600v1)**

> **作者:** Soumyajit Maity; Pranjal Kamboj; Sneha Maity; Rajat Singh; Sankhadeep Chatterjee
>
> **备注:** Proceedings of ICICT 2026, London, Springer (Forthcoming, February 2026; Accepted for Publication)
>
> **摘要:** This paper proposes a MedGemma-based framework for automatic abnormality detection in musculoskeletal radiographs. Departing from conventional autoencoder and neural network pipelines, the proposed method leverages the MedGemma foundation model, incorporating a SigLIP-derived vision encoder pretrained on diverse medical imaging modalities. Preprocessed X-ray images are encoded into high-dimensional embeddings using the MedGemma vision backbone, which are subsequently passed through a lightweight multilayer perceptron for binary classification. Experimental assessment reveals that the MedGemma-driven classifier exhibits strong performance, exceeding conventional convolutional and autoencoder-based metrics. Additionally, the model leverages MedGemma's transfer learning capabilities, enhancing generalization and optimizing feature engineering. The integration of a modern medical foundation model not only enhances representation learning but also facilitates modular training strategies such as selective encoder block unfreezing for efficient domain adaptation. The findings suggest that MedGemma-powered classification systems can advance clinical radiograph triage by providing scalable and accurate abnormality detection, with potential for broader applications in automated medical image analysis. Keywords: Google MedGemma, MURA, Medical Image, Classification.
>
---
#### [new 211] LoopExpose: An Unsupervised Framework for Arbitrary-Length Exposure Correction
- **分类: cs.CV**

- **简介: LoopExpose提出一种无监督框架，用于任意长度曝光校正，通过多曝光融合生成伪标签并引入反馈循环与亮度排序损失，实现自增强优化，无需标注数据即可超越现有无监督方法。**

- **链接: [http://arxiv.org/pdf/2511.06066v1](http://arxiv.org/pdf/2511.06066v1)**

> **作者:** Ao Li; Chen Chen; Zhenyu Wang; Tao Huang; Fangfang Wu; Weisheng Dong
>
> **摘要:** Exposure correction is essential for enhancing image quality under challenging lighting conditions. While supervised learning has achieved significant progress in this area, it relies heavily on large-scale labeled datasets, which are difficult to obtain in practical scenarios. To address this limitation, we propose a pseudo label-based unsupervised method called LoopExpose for arbitrary-length exposure correction. A nested loop optimization strategy is proposed to address the exposure correction problem, where the correction model and pseudo-supervised information are jointly optimized in a two-level framework. Specifically, the upper-level trains a correction model using pseudo-labels generated through multi-exposure fusion at the lower level. A feedback mechanism is introduced where corrected images are fed back into the fusion process to refine the pseudo-labels, creating a self-reinforcing learning loop. Considering the dominant role of luminance calibration in exposure correction, a Luminance Ranking Loss is introduced to leverage the relative luminance ordering across the input sequence as a self-supervised constraint. Extensive experiments on different benchmark datasets demonstrate that LoopExpose achieves superior exposure correction and fusion performance, outperforming existing state-of-the-art unsupervised methods. Code is available at https://github.com/FALALAS/LoopExpose.
>
---
#### [new 212] Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文面向多模态虚假信息检测任务，针对图像信息不足问题，提出RETSIMD方法：将文本分段生成补充图像，结合互信息优化与图神经网络融合多模态特征，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2511.06284v1](http://arxiv.org/pdf/2511.06284v1)**

> **作者:** Bing Wang; Ximing Li; Yanjun Wang; Changchun Li; Lin Yuanbo Wu; Buyu Wang; Shengsheng Wang
>
> **备注:** Accepted by AAAI 2026. 13 pages, 6 figures. Code: https://github.com/wangbing1416/RETSIMD
>
> **摘要:** Multimodal Misinformation Detection (MMD) refers to the task of detecting social media posts involving misinformation, where the post often contains text and image modalities. However, by observing the MMD posts, we hold that the text modality may be much more informative than the image modality because the text generally describes the whole event/story of the current post but the image often presents partial scenes only. Our preliminary empirical results indicate that the image modality exactly contributes less to MMD. Upon this idea, we propose a new MMD method named RETSIMD. Specifically, we suppose that each text can be divided into several segments, and each text segment describes a partial scene that can be presented by an image. Accordingly, we split the text into a sequence of segments, and feed these segments into a pre-trained text-to-image generator to augment a sequence of images. We further incorporate two auxiliary objectives concerning text-image and image-label mutual information, and further post-train the generator over an auxiliary text-to-image generation benchmark dataset. Additionally, we propose a graph structure by defining three heuristic relationships between images, and use a graph neural network to generate the fused features. Extensive empirical results validate the effectiveness of RETSIMD.
>
---
#### [new 213] Efficient Online Continual Learning in Sensor-Based Human Activity Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PTRN-HAR，首次将预训练模型用于传感器人体活动识别的在线持续学习，通过对比学习预训练特征提取器并引入关系模块，显著降低计算与标注数据需求，提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2511.05566v1](http://arxiv.org/pdf/2511.05566v1)**

> **作者:** Yao Zhang; Souza Leite Clayton; Yu Xiao
>
> **备注:** 13 pages
>
> **摘要:** Machine learning models for sensor-based human activity recognition (HAR) are expected to adapt post-deployment to recognize new activities and different ways of performing existing ones. To address this need, Online Continual Learning (OCL) mechanisms have been proposed, allowing models to update their knowledge incrementally as new data become available while preserving previously acquired information. However, existing OCL approaches for sensor-based HAR are computationally intensive and require extensive labeled samples to represent new changes. Recently, pre-trained model-based (PTM-based) OCL approaches have shown significant improvements in performance and efficiency for computer vision applications. These methods achieve strong generalization capabilities by pre-training complex models on large datasets, followed by fine-tuning on downstream tasks for continual learning. However, applying PTM-based OCL approaches to sensor-based HAR poses significant challenges due to the inherent heterogeneity of HAR datasets and the scarcity of labeled data in post-deployment scenarios. This paper introduces PTRN-HAR, the first successful application of PTM-based OCL to sensor-based HAR. Unlike prior PTM-based OCL approaches, PTRN-HAR pre-trains the feature extractor using contrastive loss with a limited amount of data. This extractor is then frozen during the streaming stage. Furthermore, it replaces the conventional dense classification layer with a relation module network. Our design not only significantly reduces the resource consumption required for model training while maintaining high performance, but also improves data efficiency by reducing the amount of labeled data needed for effective continual learning, as demonstrated through experiments on three public datasets, outperforming the state-of-the-art. The code can be found here: https://anonymous.4open.science/r/PTRN-HAR-AF60/
>
---
#### [new 214] VMDT: Decoding the Trustworthiness of Video Foundation Models
- **分类: cs.CV; cs.LG**

- **简介: 论文提出VMDT，首个视频基础模型信任度评估平台，统一衡量T2V与V2T模型在安全、幻觉、公平、隐私和对抗鲁棒性五维度的表现，揭示当前模型信任短板，推动可信视频模型发展。**

- **链接: [http://arxiv.org/pdf/2511.05682v1](http://arxiv.org/pdf/2511.05682v1)**

> **作者:** Yujin Potter; Zhun Wang; Nicholas Crispino; Kyle Montgomery; Alexander Xiong; Ethan Y. Chang; Francesco Pinto; Yuqi Chen; Rahul Gupta; Morteza Ziyadi; Christos Christodoulopoulos; Bo Li; Chenguang Wang; Dawn Song
>
> **备注:** NeurIPS 2025 Datasets & Benchmarks
>
> **摘要:** As foundation models become more sophisticated, ensuring their trustworthiness becomes increasingly critical; yet, unlike text and image, the video modality still lacks comprehensive trustworthiness benchmarks. We introduce VMDT (Video-Modal DecodingTrust), the first unified platform for evaluating text-to-video (T2V) and video-to-text (V2T) models across five key trustworthiness dimensions: safety, hallucination, fairness, privacy, and adversarial robustness. Through our extensive evaluation of 7 T2V models and 19 V2T models using VMDT, we uncover several significant insights. For instance, all open-source T2V models evaluated fail to recognize harmful queries and often generate harmful videos, while exhibiting higher levels of unfairness compared to image modality models. In V2T models, unfairness and privacy risks rise with scale, whereas hallucination and adversarial robustness improve -- though overall performance remains low. Uniquely, safety shows no correlation with model size, implying that factors other than scale govern current safety levels. Our findings highlight the urgent need for developing more robust and trustworthy video foundation models, and VMDT provides a systematic framework for measuring and tracking progress toward this goal. The code is available at https://sunblaze-ucb.github.io/VMDT-page/.
>
---
#### [new 215] On Modality Incomplete Infrared-Visible Object Detection: An Architecture Compatibility Perspective
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对红外-可见光目标检测（IVOD）中模态缺失问题，提出Scarf-DETR，通过可插拔的Scarf Neck模块和伪模态丢弃策略，实现对单/双模态的自适应检测，在缺失主导模态时仍保持高性能。**

- **链接: [http://arxiv.org/pdf/2511.06406v1](http://arxiv.org/pdf/2511.06406v1)**

> **作者:** Shuo Yang; Yinghui Xing; Shizhou Zhang; Zhilong Niu
>
> **摘要:** Infrared and visible object detection (IVOD) is essential for numerous around-the-clock applications. Despite notable advancements, current IVOD models exhibit notable performance declines when confronted with incomplete modality data, particularly if the dominant modality is missing. In this paper, we take a thorough investigation on modality incomplete IVOD problem from an architecture compatibility perspective. Specifically, we propose a plug-and-play Scarf Neck module for DETR variants, which introduces a modality-agnostic deformable attention mechanism to enable the IVOD detector to flexibly adapt to any single or double modalities during training and inference. When training Scarf-DETR, we design a pseudo modality dropout strategy to fully utilize the multi-modality information, making the detector compatible and robust to both working modes of single and double modalities. Moreover, we introduce a comprehensive benchmark for the modality-incomplete IVOD task aimed at thoroughly assessing situations where the absent modality is either dominant or secondary. Our proposed Scarf-DETR not only performs excellently in missing modality scenarios but also achieves superior performances on the standard IVOD modality complete benchmarks. Our code will be available at https://github.com/YinghuiXing/Scarf-DETR.
>
---
#### [new 216] Active Learning for Animal Re-Identification with Ambiguity-Aware Sampling
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对动物重识别（Re-ID）任务，解决标注成本高与零样本性能差的问题，提出一种基于模糊感知采样的主动学习框架，通过聚类挖掘关键样本对，仅用0.033%标注即显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06658v1](http://arxiv.org/pdf/2511.06658v1)**

> **作者:** Depanshu Sani; Mehar Khurana; Saket Anand
>
> **备注:** In Proceedings of AAAI Conference on Artificial Intelligence 2026
>
> **摘要:** Animal Re-ID has recently gained substantial attention in the AI research community due to its high impact on biodiversity monitoring and unique research challenges arising from environmental factors. The subtle distinguishing patterns, handling new species and the inherent open-set nature make the problem even harder. To address these complexities, foundation models trained on labeled, large-scale and multi-species animal Re-ID datasets have recently been introduced to enable zero-shot Re-ID. However, our benchmarking reveals significant gaps in their zero-shot Re-ID performance for both known and unknown species. While this highlights the need for collecting labeled data in new domains, exhaustive annotation for Re-ID is laborious and requires domain expertise. Our analyses show that existing unsupervised (USL) and AL Re-ID methods underperform for animal Re-ID. To address these limitations, we introduce a novel AL Re-ID framework that leverages complementary clustering methods to uncover and target structurally ambiguous regions in the embedding space for mining pairs of samples that are both informative and broadly representative. Oracle feedback on these pairs, in the form of must-link and cannot-link constraints, facilitates a simple annotation interface, which naturally integrates with existing USL methods through our proposed constrained clustering refinement algorithm. Through extensive experiments, we demonstrate that, by utilizing only 0.033% of all annotations, our approach consistently outperforms existing foundational, USL and AL baselines. Specifically, we report an average improvement of 10.49%, 11.19% and 3.99% (mAP) on 13 wildlife datasets over foundational, USL and AL methods, respectively, while attaining state-of-the-art performance on each dataset. Furthermore, we also show an improvement of 11.09%, 8.2% and 2.06% for unknown individuals in an open-world setting.
>
---
#### [new 217] Generating an Image From 1,000 Words: Enhancing Text-to-Image With Structured Captions
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决短提示导致的生成控制不足问题。通过使用结构化长描述训练开源模型FIBO，提出DimFusion高效融合机制与TaBR评估协议，显著提升生成精确性与可控性。**

- **链接: [http://arxiv.org/pdf/2511.06876v1](http://arxiv.org/pdf/2511.06876v1)**

> **作者:** Eyal Gutflaish; Eliran Kachlon; Hezi Zisman; Tal Hacham; Nimrod Sarid; Alexander Visheratin; Saar Huberman; Gal Davidi; Guy Bukchin; Kfir Goldberg; Ron Mokady
>
> **摘要:** Text-to-image models have rapidly evolved from casual creative tools to professional-grade systems, achieving unprecedented levels of image quality and realism. Yet, most models are trained to map short prompts into detailed images, creating a gap between sparse textual input and rich visual outputs. This mismatch reduces controllability, as models often fill in missing details arbitrarily, biasing toward average user preferences and limiting precision for professional use. We address this limitation by training the first open-source text-to-image model on long structured captions, where every training sample is annotated with the same set of fine-grained attributes. This design maximizes expressive coverage and enables disentangled control over visual factors. To process long captions efficiently, we propose DimFusion, a fusion mechanism that integrates intermediate tokens from a lightweight LLM without increasing token length. We also introduce the Text-as-a-Bottleneck Reconstruction (TaBR) evaluation protocol. By assessing how well real images can be reconstructed through a captioning-generation loop, TaBR directly measures controllability and expressiveness, even for very long captions where existing evaluation methods fail. Finally, we demonstrate our contributions by training the large-scale model FIBO, achieving state-of-the-art prompt alignment among open-source models. Model weights are publicly available at https://huggingface.co/briaai/FIBO
>
---
#### [new 218] Lite VLA: Efficient Vision-Language-Action Control on CPU-Bound Edge Robots
- **分类: cs.RO; cs.AR; cs.CV; cs.SY; eess.SY**

- **简介: 该论文提出Lite VLA，首次在CPU受限的边缘机器人上部署轻量级视觉-语言模型，实现无需云端的实时感知与动作协同，解决资源受限环境下自主决策难题，支持服务、救灾等场景的高效边缘自治。**

- **链接: [http://arxiv.org/pdf/2511.05642v1](http://arxiv.org/pdf/2511.05642v1)**

> **作者:** Justin Williams; Kishor Datta Gupta; Roy George; Mrinmoy Sarkar
>
> **摘要:** The deployment of artificial intelligence models at the edge is increasingly critical for autonomous robots operating in GPS-denied environments where local, resource-efficient reasoning is essential. This work demonstrates the feasibility of deploying small Vision-Language Models (VLMs) on mobile robots to achieve real-time scene understanding and reasoning under strict computational constraints. Unlike prior approaches that separate perception from mobility, the proposed framework enables simultaneous movement and reasoning in dynamic environments using only on-board hardware. The system integrates a compact VLM with multimodal perception to perform contextual interpretation directly on embedded hardware, eliminating reliance on cloud connectivity. Experimental validation highlights the balance between computational efficiency, task accuracy, and system responsiveness. Implementation on a mobile robot confirms one of the first successful deployments of small VLMs for concurrent reasoning and mobility at the edge. This work establishes a foundation for scalable, assured autonomy in applications such as service robotics, disaster response, and defense operations.
>
---
#### [new 219] ArtReg: Visuo-Tactile based Pose Tracking and Manipulation of Unseen Articulated Objects
- **分类: cs.RO; cs.CV**

- **简介: 论文提出ArtReg方法，利用视觉-触觉点云融合与SE(3)李群卡尔曼滤波，实现对未知刚体与关节物体的无模型位姿追踪与闭环操控，解决机器人无先验知识下复杂对象交互难题。**

- **链接: [http://arxiv.org/pdf/2511.06378v1](http://arxiv.org/pdf/2511.06378v1)**

> **作者:** Prajval Kumar Murali; Mohsen Kaboli
>
> **备注:** Under review
>
> **摘要:** Robots operating in real-world environments frequently encounter unknown objects with complex structures and articulated components, such as doors, drawers, cabinets, and tools. The ability to perceive, track, and manipulate these objects without prior knowledge of their geometry or kinematic properties remains a fundamental challenge in robotics. In this work, we present a novel method for visuo-tactile-based tracking of unseen objects (single, multiple, or articulated) during robotic interaction without assuming any prior knowledge regarding object shape or dynamics. Our novel pose tracking approach termed ArtReg (stands for Articulated Registration) integrates visuo-tactile point clouds in an unscented Kalman Filter formulation in the SE(3) Lie Group for point cloud registration. ArtReg is used to detect possible articulated joints in objects using purposeful manipulation maneuvers such as pushing or hold-pulling with a two-robot team. Furthermore, we leverage ArtReg to develop a closed-loop controller for goal-driven manipulation of articulated objects to move the object into the desired pose configuration. We have extensively evaluated our approach on various types of unknown objects through real robot experiments. We also demonstrate the robustness of our method by evaluating objects with varying center of mass, low-light conditions, and with challenging visual backgrounds. Furthermore, we benchmarked our approach on a standard dataset of articulated objects and demonstrated improved performance in terms of pose accuracy compared to state-of-the-art methods. Our experiments indicate that robust and accurate pose tracking leveraging visuo-tactile information enables robots to perceive and interact with unseen complex articulated objects (with revolute or prismatic joints).
>
---
#### [new 220] SlotVLA: Towards Modeling of Object-Relation Representations in Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SlotVLA框架，面向机器人操作任务，解决传统模型视觉表征冗余、不可解释问题。通过LIBERO+数据集与槽注意力机制，构建对象及关系的紧凑表征，实现高效、可解释的多任务操控。**

- **链接: [http://arxiv.org/pdf/2511.06754v1](http://arxiv.org/pdf/2511.06754v1)**

> **作者:** Taisei Hanyu; Nhat Chung; Huy Le; Toan Nguyen; Yuki Ikebe; Anthony Gunderman; Duy Nguyen Ho Minh; Khoa Vo; Tung Kieu; Kashu Yamazaki; Chase Rainwater; Anh Nguyen; Ngan Le
>
> **备注:** under review
>
> **摘要:** Inspired by how humans reason over discrete objects and their relationships, we explore whether compact object-centric and object-relation representations can form a foundation for multitask robotic manipulation. Most existing robotic multitask models rely on dense embeddings that entangle both object and background cues, raising concerns about both efficiency and interpretability. In contrast, we study object-relation-centric representations as a pathway to more structured, efficient, and explainable visuomotor control. Our contributions are two-fold. First, we introduce LIBERO+, a fine-grained benchmark dataset designed to enable and evaluate object-relation reasoning in robotic manipulation. Unlike prior datasets, LIBERO+ provides object-centric annotations that enrich demonstrations with box- and mask-level labels as well as instance-level temporal tracking, supporting compact and interpretable visuomotor representations. Second, we propose SlotVLA, a slot-attention-based framework that captures both objects and their relations for action decoding. It uses a slot-based visual tokenizer to maintain consistent temporal object representations, a relation-centric decoder to produce task-relevant embeddings, and an LLM-driven module that translates these embeddings into executable actions. Experiments on LIBERO+ demonstrate that object-centric slot and object-relation slot representations drastically reduce the number of required visual tokens, while providing competitive generalization. Together, LIBERO+ and SlotVLA provide a compact, interpretable, and effective foundation for advancing object-relation-centric robotic manipulation.
>
---
#### [new 221] Oh That Looks Familiar: A Novel Similarity Measure for Spreadsheet Template Discovery
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出一种融合语义嵌入、数据类型与空间布局的混合相似度度量方法，用于无监督发现电子表格模板，解决传统方法忽略布局与类型模式的问题，在FUSTE数据集上实现完美聚类，支持自动化模板检索与数据清洗。**

- **链接: [http://arxiv.org/pdf/2511.06973v1](http://arxiv.org/pdf/2511.06973v1)**

> **作者:** Ananad Krishnakumar; Vengadesh Ravikumaran
>
> **备注:** 5 pages, 2 figures, Accepted for EuroIPS: AI for Tabular Data Workshop (2025)
>
> **摘要:** Traditional methods for identifying structurally similar spreadsheets fail to capture the spatial layouts and type patterns defining templates. To quantify spreadsheet similarity, we introduce a hybrid distance metric that combines semantic embeddings, data type information, and spatial positioning. In order to calculate spreadsheet similarity, our method converts spreadsheets into cell-level embeddings and then uses aggregation techniques like Chamfer and Hausdorff distances. Experiments across template families demonstrate superior unsupervised clustering performance compared to the graph-based Mondrian baseline, achieving perfect template reconstruction (Adjusted Rand Index of 1.00 versus 0.90) on the FUSTE dataset. Our approach facilitates large-scale automated template discovery, which in turn enables downstream applications such as retrieval-augmented generation over tabular collections, model training, and bulk data cleaning.
>
---
#### [new 222] Test-Time Iterative Error Correction for Efficient Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对高效扩散模型在部署时因近似误差导致生成质量下降的问题，提出测试时迭代误差校正（IEC）方法，无需重训练即可迭代优化输出，将误差从指数增长降为线性增长，显著提升生成质量。**

- **链接: [http://arxiv.org/pdf/2511.06250v1](http://arxiv.org/pdf/2511.06250v1)**

> **作者:** Yunshan Zhong; Yanwei Qi; Yuxin Zhang
>
> **摘要:** With the growing demand for high-quality image generation on resource-constrained devices, efficient diffusion models have received increasing attention. However, such models suffer from approximation errors introduced by efficiency techniques, which significantly degrade generation quality. Once deployed, these errors are difficult to correct, as modifying the model is typically infeasible in deployment environments. Through an analysis of error propagation across diffusion timesteps, we reveal that these approximation errors can accumulate exponentially, severely impairing output quality. Motivated by this insight, we propose Iterative Error Correction (IEC), a novel test-time method that mitigates inference-time errors by iteratively refining the model's output. IEC is theoretically proven to reduce error propagation from exponential to linear growth, without requiring any retraining or architectural changes. IEC can seamlessly integrate into the inference process of existing diffusion models, enabling a flexible trade-off between performance and efficiency. Extensive experiments show that IEC consistently improves generation quality across various datasets, efficiency techniques, and model architectures, establishing it as a practical and generalizable solution for test-time enhancement of efficient diffusion models.
>
---
#### [new 223] Robot Learning from a Physical World Model
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出PhysWorld框架，通过物理世界建模将生成视频转化为符合物理规律的机器人动作，解决视觉生成动作忽略物理导致的不准确问题，实现无需真实数据的零样本机器人操控。**

- **链接: [http://arxiv.org/pdf/2511.07416v1](http://arxiv.org/pdf/2511.07416v1)**

> **作者:** Jiageng Mao; Sicheng He; Hao-Ning Wu; Yang You; Shuyang Sun; Zhicheng Wang; Yanan Bao; Huizhong Chen; Leonidas Guibas; Vitor Guizilini; Howard Zhou; Yue Wang
>
> **备注:** Project page: https://pointscoder.github.io/PhysWorld_Web/
>
> **摘要:** We introduce PhysWorld, a framework that enables robot learning from video generation through physical world modeling. Recent video generation models can synthesize photorealistic visual demonstrations from language commands and images, offering a powerful yet underexplored source of training signals for robotics. However, directly retargeting pixel motions from generated videos to robots neglects physics, often resulting in inaccurate manipulations. PhysWorld addresses this limitation by coupling video generation with physical world reconstruction. Given a single image and a task command, our method generates task-conditioned videos and reconstructs the underlying physical world from the videos, and the generated video motions are grounded into physically accurate actions through object-centric residual reinforcement learning with the physical world model. This synergy transforms implicit visual guidance into physically executable robotic trajectories, eliminating the need for real robot data collection and enabling zero-shot generalizable robotic manipulation. Experiments on diverse real-world tasks demonstrate that PhysWorld substantially improves manipulation accuracy compared to previous approaches. Visit \href{https://pointscoder.github.io/PhysWorld_Web/}{the project webpage} for details.
>
---
#### [new 224] Referring Expressions as a Lens into Spatial Language Grounding in Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出使用指代表达理解任务评估视觉语言模型的空间推理能力，针对模糊检测、复杂空间关系与否定表达等挑战，分析不同模型在拓扑、方向等空间语义上的表现，揭示其短板与研究缺口。**

- **链接: [http://arxiv.org/pdf/2511.06146v1](http://arxiv.org/pdf/2511.06146v1)**

> **作者:** Akshar Tumu; Varad Shinde; Parisa Kordjamshidi
>
> **备注:** Accepted at IJCNLP-AACL 2025
>
> **摘要:** Spatial Reasoning is an important component of human cognition and is an area in which the latest Vision-language models (VLMs) show signs of difficulty. The current analysis works use image captioning tasks and visual question answering. In this work, we propose using the Referring Expression Comprehension task instead as a platform for the evaluation of spatial reasoning by VLMs. This platform provides the opportunity for a deeper analysis of spatial comprehension and grounding abilities when there is 1) ambiguity in object detection, 2) complex spatial expressions with a longer sentence structure and multiple spatial relations, and 3) expressions with negation ('not'). In our analysis, we use task-specific architectures as well as large VLMs and highlight their strengths and weaknesses in dealing with these specific situations. While all these models face challenges with the task at hand, the relative behaviors depend on the underlying models and the specific categories of spatial semantics (topological, directional, proximal, etc.). Our results highlight these challenges and behaviors and provide insight into research gaps and future directions.
>
---
#### [new 225] sMRI-based Brain Age Estimation in MCI using Persistent Homology
- **分类: q-bio.NC; cs.CV; eess.IV**

- **简介: 该论文利用持久同源性（Betti曲线）分析sMRI，构建脑龄预测模型，区分健康与病理性老化，解决阿尔茨海默病早期识别问题。基于ADNI数据，提取拓扑特征并关联临床指标，为认知衰退提供新型生物标志物框架。**

- **链接: [http://arxiv.org/pdf/2511.05520v1](http://arxiv.org/pdf/2511.05520v1)**

> **作者:** Debanjali Bhattacharya; Neelam Sinha
>
> **摘要:** In this study, we propose the use of persistent homology- specifically Betti curves for brain age prediction and for distinguishing between healthy and pathological aging. The proposed framework is applied to 100 structural MRI scans from the publicly available ADNI dataset. Our results indicate that Betti curve features, particularly those from dimension-1 (connected components) and dimension-2 (1D holes), effectively capture structural brain alterations associated with aging. Furthermore, clinical features are grouped into three categories based on their correlation, or lack thereof, with (i) predicted brain age and (ii) chronological age. The findings demonstrate that this approach successfully differentiates normal from pathological aging and provides a novel framework for understanding how structural brain changes relate to cognitive impairment. The proposed method serves as a foundation for developing potential biomarkers for early detection and monitoring of cognitive decline.
>
---
#### [new 226] PlanT 2.0: Exposing Biases and Structural Flaws in Closed-Loop Driving
- **分类: cs.RO; cs.CV**

- **简介: 论文提出PlanT 2.0，一种面向CARLA的物体中心驾驶规划模型，通过可控输入扰动揭示自动驾驶模型的偏差与结构缺陷，如过拟合、捷径学习等，呼吁转向数据-centric开发，并开源代码。**

- **链接: [http://arxiv.org/pdf/2511.07292v1](http://arxiv.org/pdf/2511.07292v1)**

> **作者:** Simon Gerstenecker; Andreas Geiger; Katrin Renz
>
> **摘要:** Most recent work in autonomous driving has prioritized benchmark performance and methodological innovation over in-depth analysis of model failures, biases, and shortcut learning. This has led to incremental improvements without a deep understanding of the current failures. While it is straightforward to look at situations where the model fails, it is hard to understand the underlying reason. This motivates us to conduct a systematic study, where inputs to the model are perturbed and the predictions observed. We introduce PlanT 2.0, a lightweight, object-centric planning transformer designed for autonomous driving research in CARLA. The object-level representation enables controlled analysis, as the input can be easily perturbed (e.g., by changing the location or adding or removing certain objects), in contrast to sensor-based models. To tackle the scenarios newly introduced by the challenging CARLA Leaderboard 2.0, we introduce multiple upgrades to PlanT, achieving state-of-the-art performance on Longest6 v2, Bench2Drive, and the CARLA validation routes. Our analysis exposes insightful failures, such as a lack of scene understanding caused by low obstacle diversity, rigid expert behaviors leading to exploitable shortcuts, and overfitting to a fixed set of expert trajectories. Based on these findings, we argue for a shift toward data-centric development, with a focus on richer, more robust, and less biased datasets. We open-source our code and model at https://github.com/autonomousvision/plant2.
>
---
#### [new 227] CAMP-HiVe: Cyclic Pair Merging based Efficient DNN Pruning with Hessian-Vector Approximation for Resource-Constrained Systems
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出CAMP-HiVe，一种基于Hessian向量积的循环权值对合并剪枝方法，用于资源受限系统的高效DNN压缩，解决传统剪枝计算开销大问题，在保持精度前提下显著降低模型复杂度。**

- **链接: [http://arxiv.org/pdf/2511.06265v1](http://arxiv.org/pdf/2511.06265v1)**

> **作者:** Mohammad Helal Uddin; Sai Krishna Ghanta; Liam Seymour; Sabur Baidya
>
> **摘要:** Deep learning algorithms are becoming an essential component of many artificial intelligence (AI) driven applications, many of which run on resource-constrained and energy-constrained systems. For efficient deployment of these algorithms, although different techniques for the compression of neural network models are proposed, neural pruning is one of the fastest and effective methods, which can provide a high compression gain with minimal cost. To harness enhanced performance gain with respect to model complexity, we propose a novel neural network pruning approach utilizing Hessian-vector products that approximate crucial curvature information in the loss function, which significantly reduces the computation demands. By employing a power iteration method, our algorithm effectively identifies and preserves the essential information, ensuring a balanced trade-off between model accuracy and computational efficiency. Herein, we introduce CAMP-HiVe, a cyclic pair merging-based pruning with Hessian Vector approximation by iteratively consolidating weight pairs, combining significant and less significant weights, thus effectively streamlining the model while preserving its performance. This dynamic, adaptive framework allows for real-time adjustment of weight significance, ensuring that only the most critical parameters are retained. Our experimental results demonstrate that our proposed method achieves significant reductions in computational requirements while maintaining high performance across different neural network architectures, e.g., ResNet18, ResNet56, and MobileNetv2, on standard benchmark datasets, e.g., CIFAR10, CIFAR-100, and ImageNet, and it outperforms the existing state-of-the-art neural pruning methods.
>
---
#### [new 228] Vision-Based System Identification of a Quadrotor
- **分类: cs.RO; cs.CV; cs.SY; eess.SY; math.DS**

- **简介: 该论文属于无人机建模与控制任务，旨在解决传统quadrotor模型中推力与阻力参数不确定性问题。通过机载视觉系统采集数据，采用灰箱建模与LQR控制，验证了视觉系统辨识模型的有效性与一致性。**

- **链接: [http://arxiv.org/pdf/2511.06839v1](http://arxiv.org/pdf/2511.06839v1)**

> **作者:** Selim Ahmet Iz; Mustafa Unel
>
> **摘要:** This paper explores the application of vision-based system identification techniques in quadrotor modeling and control. Through experiments and analysis, we address the complexities and limitations of quadrotor modeling, particularly in relation to thrust and drag coefficients. Grey-box modeling is employed to mitigate uncertainties, and the effectiveness of an onboard vision system is evaluated. An LQR controller is designed based on a system identification model using data from the onboard vision system. The results demonstrate consistent performance between the models, validating the efficacy of vision based system identification. This study highlights the potential of vision-based techniques in enhancing quadrotor modeling and control, contributing to improved performance and operational capabilities. Our findings provide insights into the usability and consistency of these techniques, paving the way for future research in quadrotor performance enhancement, fault detection, and decision-making processes.
>
---
#### [new 229] TabRAG: Tabular Document Retrieval via Structured Language Representations
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **简介: TabRAG面向表格密集文档的检索增强生成任务，解决传统解析方法对表格信息提取效果差的问题，提出基于结构化语言表示的解析管道，显著提升检索与生成性能。**

- **链接: [http://arxiv.org/pdf/2511.06582v1](http://arxiv.org/pdf/2511.06582v1)**

> **作者:** Jacob Si; Mike Qu; Michelle Lee; Yingzhen Li
>
> **备注:** NeurIPS 2025 AI4Tab
>
> **摘要:** Ingesting data for Retrieval-Augmented Generation (RAG) involves either fine-tuning the embedding model directly on the target corpus or parsing documents for embedding model encoding. The former, while accurate, incurs high computational hardware requirements, while the latter suffers from suboptimal performance when extracting tabular data. In this work, we address the latter by presenting TabRAG, a parsing-based RAG pipeline designed to tackle table-heavy documents via structured language representations. TabRAG outperforms existing popular parsing-based methods for generation and retrieval. Code is available at https://github.com/jacobyhsi/TabRAG.
>
---
#### [new 230] TauFlow: Dynamic Causal Constraint for Complexity-Adaptive Lightweight Segmentation
- **分类: eess.IV; cs.AI; cs.CV; 68U10, 68T45, 92C55, 68T07; I.4.6; I.2.10; J.3; I.2.6**

- **简介: TauFlow面向轻量级医学图像分割，解决边界与背景对比强烈及模型极简时精度骤降问题，提出ConvLTC动态调节特征更新、STDP自组织模块缓解编解码冲突，实现高效高精度分割。**

- **链接: [http://arxiv.org/pdf/2511.07057v1](http://arxiv.org/pdf/2511.07057v1)**

> **作者:** Zidong Chen; Fadratul Hafinaz Hassan
>
> **备注:** 42 pages and 9 figures
>
> **摘要:** Deploying lightweight medical image segmentation models on edge devices presents two major challenges: 1) efficiently handling the stark contrast between lesion boundaries and background regions, and 2) the sharp drop in accuracy that occurs when pursuing extremely lightweight designs (e.g., <0.5M parameters). To address these problems, this paper proposes TauFlow, a novel lightweight segmentation model. The core of TauFlow is a dynamic feature response strategy inspired by brain-like mechanisms. This is achieved through two key innovations: the Convolutional Long-Time Constant Cell (ConvLTC), which dynamically regulates the feature update rate to "slowly" process low-frequency backgrounds and "quickly" respond to high-frequency boundaries; and the STDP Self-Organizing Module, which significantly mitigates feature conflicts between the encoder and decoder, reducing the conflict rate from approximately 35%-40% to 8%-10%.
>
---
#### [new 231] Training-Free Adaptive Quantization for Variable Rate Image Coding for Machines
- **分类: eess.IV; cs.CV**

- **简介: 该论文面向机器视觉的可变率图像编码任务，解决传统方法需训练多模型的问题，提出一种无训练的自适应量化方法，通过熵与空间参数动态调整比特率，保留语义重要区域，实现单参数连续控速，提升编码效率。**

- **链接: [http://arxiv.org/pdf/2511.05836v1](http://arxiv.org/pdf/2511.05836v1)**

> **作者:** Yui Tatsumi; Ziyue Zeng; Hiroshi Watanabe
>
> **摘要:** Image Coding for Machines (ICM) has become increasingly important with the rapid integration of computer vision into real-world applications. However, most ICM frameworks utilize learned image compression (LIC) models that operate at a fixed rate and require separate training for each target bitrate, which may limit their practical applications. Existing variable rate LIC approaches mitigate this limitation but typically depend on training, increasing computational cost and deployment complexity. Moreover, variable rate control has not been thoroughly explored for ICM. To address these challenges, we propose a training-free, adaptive quantization step size control scheme that enables flexible bitrate adjustment. By leveraging both channel-wise entropy dependencies and spatial scale parameters predicted by the hyperprior network, the proposed method preserves semantically important regions while coarsely quantizing less critical areas. The bitrate can be continuously controlled through a single parameter. Experimental results demonstrate the effectiveness of our proposed method, achieving up to 11.07% BD-rate savings over the non-adaptive variable rate method.
>
---
#### [new 232] Non-Negative Stiefel Approximating Flow: Orthogonalish Matrix Optimization for Interpretable Embeddings
- **分类: stat.ML; cs.CV; cs.LG; stat.ME**

- **简介: 论文提出NSA-Flow，一种面向可解释嵌入的正交约束矩阵优化方法，融合稀疏分解与流形学习，在保持重建精度的同时实现稀疏、稳定且可解释的表示，适用于生物医学等高维数据降维任务。**

- **链接: [http://arxiv.org/pdf/2511.06425v1](http://arxiv.org/pdf/2511.06425v1)**

> **作者:** Brian B. Avants; Nicholas J. Tustison; James R Stone
>
> **摘要:** Interpretable representation learning is a central challenge in modern machine learning, particularly in high-dimensional settings such as neuroimaging, genomics, and text analysis. Current methods often struggle to balance the competing demands of interpretability and model flexibility, limiting their effectiveness in extracting meaningful insights from complex data. We introduce Non-negative Stiefel Approximating Flow (NSA-Flow), a general-purpose matrix estimation framework that unifies ideas from sparse matrix factorization, orthogonalization, and constrained manifold learning. NSA-Flow enforces structured sparsity through a continuous balance between reconstruction fidelity and column-wise decorrelation, parameterized by a single tunable weight. The method operates as a smooth flow near the Stiefel manifold with proximal updates for non-negativity and adaptive gradient control, yielding representations that are simultaneously sparse, stable, and interpretable. Unlike classical regularization schemes, NSA-Flow provides an intuitive geometric mechanism for manipulating sparsity at the level of global structure while simplifying latent features. We demonstrate that the NSA-Flow objective can be optimized smoothly and integrates seamlessly with existing pipelines for dimensionality reduction while improving interpretability and generalization in both simulated and real biomedical data. Empirical validation on the Golub leukemia dataset and in Alzheimer's disease demonstrate that the NSA-Flow constraints can maintain or improve performance over related methods with little additional methodological effort. NSA-Flow offers a scalable, general-purpose tool for interpretable ML, applicable across data science domains.
>
---
#### [new 233] MARAuder's Map: Motion-Aware Real-time Activity Recognition with Layout-Based Trajectories
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出MARAuder's Map，用于智能住宅中基于传感器的实时人体活动识别。通过将传感器数据映射到户型图生成轨迹图像，结合时空深度模型与时间嵌入、注意力机制，解决未分段数据下空间-时间建模难题，提升实时识别准确率。**

- **链接: [http://arxiv.org/pdf/2511.05773v1](http://arxiv.org/pdf/2511.05773v1)**

> **作者:** Zishuai Liu; Weihang You; Jin Lu; Fei Dou
>
> **摘要:** Ambient sensor-based human activity recognition (HAR) in smart homes remains challenging due to the need for real-time inference, spatially grounded reasoning, and context-aware temporal modeling. Existing approaches often rely on pre-segmented, within-activity data and overlook the physical layout of the environment, limiting their robustness in continuous, real-world deployments. In this paper, we propose MARAuder's Map, a novel framework for real-time activity recognition from raw, unsegmented sensor streams. Our method projects sensor activations onto the physical floorplan to generate trajectory-aware, image-like sequences that capture the spatial flow of human movement. These representations are processed by a hybrid deep learning model that jointly captures spatial structure and temporal dependencies. To enhance temporal awareness, we introduce a learnable time embedding module that encodes contextual cues such as hour-of-day and day-of-week. Additionally, an attention-based encoder selectively focuses on informative segments within each observation window, enabling accurate recognition even under cross-activity transitions and temporal ambiguity. Extensive experiments on multiple real-world smart home datasets demonstrate that our method outperforms strong baselines, offering a practical solution for real-time HAR in ambient sensor environments.
>
---
#### [new 234] Preparation of Fractal-Inspired Computational Architectures for Advanced Large Language Model Analysis
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出FractalNet，一种基于分形结构的神经网络架构自动生成方法，用于高效探索大语言模型的多样性。通过模板化层组合生成1200+变体，在CIFAR-10上验证其性能与计算效率，实现资源节约的自动化架构设计。**

- **链接: [http://arxiv.org/pdf/2511.07329v1](http://arxiv.org/pdf/2511.07329v1)**

> **作者:** Yash Mittal; Dmitry Ignatov; Radu Timofte
>
> **摘要:** It introduces FractalNet, a fractal-inspired computational architectures for advanced large language model analysis that mainly challenges model diversity on a large scale in an efficient manner. The new set-up involves a template-driven generator, runner, and evaluation framework that, through systematic permutations of convolutional, normalization, activation, and dropout layers, can create more than 1,200 variants of neural networks. Fractal templates allow for structural recursion and multi-column pathways, thus, models become deeper and wider in a balanced way. Training utilizes PyTorch, Automatic Mixed Precision (AMP), and gradient checkpointing and is carried out on the CIFAR-10 dataset for five epochs. The outcomes show that fractal-based architectures are capable of strong performance and are computationally efficient. The paper positions fractal design as a feasible and resource-efficient method of automated architecture exploration.
>
---
#### [new 235] Pinching Visuo-haptic Display: Investigating Cross-Modal Effects of Visual Textures on Electrostatic Cloth Tactile Sensations
- **分类: cs.HC; cs.CV; cs.MM; H.5.2; I.3.6; I.4.8**

- **简介: 该论文研究视觉纹理对静电布料触觉感知的跨模态影响，属于多模态人机交互任务。通过构建可视-触觉系统，发现视觉粗糙度会增强用户对摩擦力的感知，为虚拟材料触觉反馈提供设计依据。**

- **链接: [http://arxiv.org/pdf/2511.05952v1](http://arxiv.org/pdf/2511.05952v1)**

> **作者:** Takekazu Kitagishi; Chun-Wei Ooi; Yuichi Hiroi; Jun Rekimoto
>
> **备注:** 10 pages, 8 figures, 3 tables. Presented at ACM International Conference on Multimodal Interaction (ICMI) 2025
>
> **摘要:** This paper investigates how visual texture presentation influences tactile perception when interacting with electrostatic cloth displays. We propose a visuo-haptic system that allows users to pinch and rub virtual fabrics while feeling realistic frictional sensations modulated by electrostatic actuation. Through a user study, we examined the cross-modal effects between visual roughness and perceived tactile friction. The results demonstrate that visually rough textures amplify the perceived frictional force, even under identical electrostatic stimuli. These findings contribute to the understanding of multimodal texture perception and provide design insights for haptic feedback in virtual material interfaces.
>
---
#### [new 236] Verifying rich robustness properties for neural networks
- **分类: cs.LO; cs.AI; cs.CV**

- **简介: 该论文提出一种通用框架，通过简单语法指定神经网络的多样化鲁棒性属性，并新增网络层实现统一验证，兼顾输出置信度，兼容现有工具且误差有界，显著超越传统编码方法。**

- **链接: [http://arxiv.org/pdf/2511.07293v1](http://arxiv.org/pdf/2511.07293v1)**

> **作者:** Mohammad Afzal; S. Akshay; Ashutosh Gupta
>
> **摘要:** Robustness is a important problem in AI alignment and safety, with models such as neural networks being increasingly used in safety-critical systems. In the last decade, a large body of work has emerged on local robustness, i.e., checking if the decision of a neural network remains unchanged when the input is slightly perturbed. However, many of these approaches require specialized encoding and often ignore the confidence of a neural network on its output. In this paper, our goal is to build a generalized framework to specify and verify variants of robustness in neural network verification. We propose a specification framework using a simple grammar, which is flexible enough to capture most existing variants. This allows us to introduce new variants of robustness that take into account the confidence of the neural network in its outputs. Next, we develop a novel and powerful unified technique to verify all such variants in a homogeneous way, viz., by adding a few additional layers to the neural network. This enables us to use any state-of-the-art neural network verification tool, without having to tinker with the encoding within, while incurring an approximation error that we show is bounded. We perform an extensive experimental evaluation over a large suite of 8870 benchmarks having 138M parameters in a largest network, and show that we are able to capture a wide set of robustness variants and outperform direct encoding approaches by a significant margin.
>
---
#### [new 237] Omni-AVSR: Towards Unified Multimodal Speech Recognition with Large Language Models
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 论文提出Omni-AVSR，一个统一的多模态语音识别框架，整合ASR、VSR与AVSR任务，利用大语言模型与弹性推理，通过多粒度训练和LoRA适配，显著降低资源消耗并提升效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.07253v1](http://arxiv.org/pdf/2511.07253v1)**

> **作者:** Umberto Cappellazzo; Xubo Liu; Pingchuan Ma; Stavros Petridis; Maja Pantic
>
> **备注:** Project website: https://umbertocappellazzo.github.io/Omni-AVSR/
>
> **摘要:** Large language models (LLMs) have recently achieved impressive results in speech recognition across multiple modalities, including Auditory Speech Recognition (ASR), Visual Speech Recognition (VSR), and Audio-Visual Speech Recognition (AVSR). Despite this progress, current LLM-based approaches typically address each task independently, training separate models that raise computational and deployment resource use while missing potential cross-task synergies. They also rely on fixed-rate token compression, which restricts flexibility in balancing accuracy with efficiency. These limitations highlight the need for a unified framework that can support ASR, VSR, and AVSR while enabling elastic inference. To this end, we present Omni-AVSR, a unified audio-visual LLM that combines efficient multi-granularity training with parameter-efficient adaptation. Specifically, we adapt the matryoshka representation learning paradigm to efficiently train across multiple audio and visual granularities, reducing its inherent training resource use. Furthermore, we explore three LoRA-based strategies for adapting the backbone LLM, balancing shared and task-specific specialization. Experiments on LRS2 and LRS3 show that Omni-AVSR achieves comparable or superior accuracy to state-of-the-art baselines while training a single model at substantially lower training and deployment resource use. The model also remains robust under acoustic noise, and we analyze its scaling behavior as LLM size increases, providing insights into the trade-off between performance and efficiency.
>
---
#### [new 238] Towards a Humanized Social-Media Ecosystem: AI-Augmented HCI Design Patterns for Safety, Agency & Well-Being
- **分类: cs.HC; cs.AI; cs.CV**

- **简介: 该论文提出Human-Layer AI（HL-AI），一种用户可控的浏览器中间层，通过五种AI增强交互模式（如内容重写、信息完整性评估等），在不依赖平台的前提下提升社交媒体的安全性、自主性与福祉，解决算法驱动下的用户失控问题。**

- **链接: [http://arxiv.org/pdf/2511.05875v1](http://arxiv.org/pdf/2511.05875v1)**

> **作者:** Mohd Ruhul Ameen; Akif Islam
>
> **备注:** 6 pages, 5 tables, 7 figures, and 2 algorithm tables. Accepted at International Conference on Signal Processing, Information, Communication and Systems (SPICSCON 2025)
>
> **摘要:** Social platforms connect billions of people, yet their engagement-first algorithms often work on users rather than with them, amplifying stress, misinformation, and a loss of control. We propose Human-Layer AI (HL-AI)--user-owned, explainable intermediaries that sit in the browser between platform logic and the interface. HL-AI gives people practical, moment-to-moment control without requiring platform cooperation. We contribute a working Chrome/Edge prototype implementing five representative pattern frameworks--Context-Aware Post Rewriter, Post Integrity Meter, Granular Feed Curator, Micro-Withdrawal Agent, and Recovery Mode--alongside a unifying mathematical formulation balancing user utility, autonomy costs, and risk thresholds. Evaluation spans technical accuracy, usability, and behavioral outcomes. The result is a suite of humane controls that help users rewrite before harm, read with integrity cues, tune feeds with intention, pause compulsive loops, and seek shelter during harassment, all while preserving agency through explanations and override options. This prototype offers a practical path to retrofit today's feeds with safety, agency, and well-being, inviting rigorous cross-cultural user evaluation.
>
---
#### [new 239] Lightning Grasp: High Performance Procedural Grasp Synthesis with Contact Fields
- **分类: cs.RO; cs.AI; cs.CV; cs.DC; cs.GR**

- **简介: 论文提出Lightning Grasp，一种基于接触场的高效程序化抓取生成方法，解决灵巧手对不规则工具类物体的实时多样抓取合成难题，无需复杂能量函数与敏感初始化，速度远超现有方法。**

- **链接: [http://arxiv.org/pdf/2511.07418v1](http://arxiv.org/pdf/2511.07418v1)**

> **作者:** Zhao-Heng Yin; Pieter Abbeel
>
> **备注:** Code: https://github.com/zhaohengyin/lightning-grasp
>
> **摘要:** Despite years of research, real-time diverse grasp synthesis for dexterous hands remains an unsolved core challenge in robotics and computer graphics. We present Lightning Grasp, a novel high-performance procedural grasp synthesis algorithm that achieves orders-of-magnitude speedups over state-of-the-art approaches, while enabling unsupervised grasp generation for irregular, tool-like objects. The method avoids many limitations of prior approaches, such as the need for carefully tuned energy functions and sensitive initialization. This breakthrough is driven by a key insight: decoupling complex geometric computation from the search process via a simple, efficient data structure - the Contact Field. This abstraction collapses the problem complexity, enabling a procedural search at unprecedented speeds. We open-source our system to propel further innovation in robotic manipulation.
>
---
#### [new 240] Identity Card Presentation Attack Detection: A Systematic Review
- **分类: cs.CR; cs.CV**

- **简介: 该论文对2020–2025年AI驱动的身份证件伪造检测（PAD）进行系统综述，揭示模型因数据匮乏与“现实差距”导致泛化差，指出合成数据易过拟合，并提出未来研究框架以提升检测系统的鲁棒性与通用性。**

- **链接: [http://arxiv.org/pdf/2511.06056v1](http://arxiv.org/pdf/2511.06056v1)**

> **作者:** Esteban M. Ruiz; Juan E. Tapia; Reinel T. Soto; Christoph Busch
>
> **摘要:** Remote identity verification is essential for modern digital security; however, it remains highly vulnerable to sophisticated Presentation Attacks (PAs) that utilise forged or manipulated identity documents. Although Deep Learning (DL) has driven advances in Presentation Attack Detection (PAD), the field is fundamentally limited by a lack of data and the poor generalisation of models across various document types and new attack methods. This article presents a systematic literature review (SLR) conducted in accordance with the PRISMA methodology, aiming to analyse and synthesise the current state of AI-based PAD for identity documents from 2020 to 2025 comprehensively. Our analysis reveals a significant methodological evolution: a transition from standard Convolutional Neural Networks (CNNs) to specialised forensic micro-artefact analysis, and more recently, the adoption of large-scale Foundation Models (FMs), marking a substantial shift in the field. We identify a central paradox that hinders progress: a critical "Reality Gap" exists between models validated on extensive, private datasets and those assessed using limited public datasets, which typically consist of mock-ups or synthetic data. This gap limits the reproducibility of research results. Additionally, we highlight a "Synthetic Utility Gap," where synthetic data generation the primary academic response to data scarcity often fails to predict forensic utility. This can lead to model overfitting to generation artefacts instead of the actual attack. This review consolidates our findings, identifies critical research gaps, and provides a definitive reference framework that outlines a prescriptive roadmap for future research aimed at developing secure, robust, and globally generalizable PAD systems.
>
---
#### [new 241] Selective Diabetic Retinopathy Screening with Accuracy-Weighted Deep Ensembles and Entropy-Guided Abstention
- **分类: q-bio.QM; cs.AI; cs.CV**

- **简介: 该论文针对糖尿病视网膜病变（DR）筛查中误诊率高、不确定性缺失的问题，提出一种精度加权深度集成模型，结合熵引导弃权机制，在3.5万张眼底图像上实现99.44%准确率，提升诊断可靠性与可解释性。**

- **链接: [http://arxiv.org/pdf/2511.05529v1](http://arxiv.org/pdf/2511.05529v1)**

> **作者:** Jophy Lin
>
> **摘要:** Diabetic retinopathy (DR), a microvascular complication of diabetes and a leading cause of preventable blindness, is projected to affect more than 130 million individuals worldwide by 2030. Early identification is essential to reduce irreversible vision loss, yet current diagnostic workflows rely on methods such as fundus photography and expert review, which remain costly and resource-intensive. This, combined with DR's asymptomatic nature, results in its underdiagnosis rate of approximately 25 percent. Although convolutional neural networks (CNNs) have demonstrated strong performance in medical imaging tasks, limited interpretability and the absence of uncertainty quantification restrict clinical reliability. Therefore, in this study, a deep ensemble learning framework integrated with uncertainty estimation is introduced to improve robustness, transparency, and scalability in DR detection. The ensemble incorporates seven CNN architectures-ResNet-50, DenseNet-121, MobileNetV3 (Small and Large), and EfficientNet (B0, B2, B3)- whose outputs are fused through an accuracy-weighted majority voting strategy. A probability-weighted entropy metric quantifies prediction uncertainty, enabling low-confidence samples to be excluded or flagged for additional review. Training and validation on 35,000 EyePACS retinal fundus images produced an unfiltered accuracy of 93.70 percent (F1 = 0.9376). Uncertainty-filtering later was conducted to remove unconfident samples, resulting in maximum-accuracy of 99.44 percent (F1 = 0.9932). The framework shows that uncertainty-aware, accuracy-weighted ensembling improves reliability without hindering performance. With confidence-calibrated outputs and a tunable accuracy-coverage trade-off, it offers a generalizable paradigm for deploying trustworthy AI diagnostics in high-risk care.
>
---
#### [new 242] Achieving Effective Virtual Reality Interactions via Acoustic Gesture Recognition based on Large Language Models
- **分类: cs.HC; cs.AI; cs.CV**

- **简介: 该论文提出一种基于大语言模型（LLM）的声学手势识别框架，解决VR/AR中视觉手势识别成本高、隐私差、需大量标注数据的问题。通过差分CIR数据与LLM分类，实现零样本/小样本手势识别，无需重新训练。**

- **链接: [http://arxiv.org/pdf/2511.07085v1](http://arxiv.org/pdf/2511.07085v1)**

> **作者:** Xijie Zhang; Fengliang He; Hong-Ning Dai
>
> **备注:** 5 pages, 4 figures, 1 table, under review at ICASSP 2026
>
> **摘要:** Natural and efficient interaction remains a critical challenge for virtual reality and augmented reality (VR/AR) systems. Vision-based gesture recognition suffers from high computational cost, sensitivity to lighting conditions, and privacy leakage concerns. Acoustic sensing provides an attractive alternative: by emitting inaudible high-frequency signals and capturing their reflections, channel impulse response (CIR) encodes how gestures perturb the acoustic field in a low-cost and user-transparent manner. However, existing CIR-based gesture recognition methods often rely on extensive training of models on large labeled datasets, making them unsuitable for few-shot VR scenarios. In this work, we propose the first framework that leverages large language models (LLMs) for CIR-based gesture recognition in VR/AR systems. Despite LLMs' strengths, it is non-trivial to achieve few-shot and zero-shot learning of CIR gestures due to their inconspicuous features. To tackle this challenge, we collect differential CIR rather than original CIR data. Moreover, we construct a real-world dataset collected from 10 participants performing 15 gestures across three categories (digits, letters, and shapes), with 10 repetitions each. We then conduct extensive experiments on this dataset using an LLM-adopted classifier. Results show that our LLM-based framework achieves accuracy comparable to classical machine learning baselines, while requiring no domain-specific retraining.
>
---
#### [new 243] HarmoQ: Harmonized Post-Training Quantization for High-Fidelity Image
- **分类: eess.IV; cs.CV**

- **简介: 论文针对超分辨率模型的后训练量化，揭示权重与激活量化的耦合效应，提出HarmoQ统一框架，通过结构校准、尺度优化与边界自适应，显著提升低比特下的重建质量与效率。**

- **链接: [http://arxiv.org/pdf/2511.05868v1](http://arxiv.org/pdf/2511.05868v1)**

> **作者:** Hongjun Wang; Jiyuan Chen; Xuan Song; Yinqiang Zheng
>
> **摘要:** Post-training quantization offers an efficient pathway to deploy super-resolution models, yet existing methods treat weight and activation quantization independently, missing their critical interplay. Through controlled experiments on SwinIR, we uncover a striking asymmetry: weight quantization primarily degrades structural similarity, while activation quantization disproportionately affects pixel-level accuracy. This stems from their distinct roles--weights encode learned restoration priors for textures and edges, whereas activations carry input-specific intensity information. Building on this insight, we propose HarmoQ, a unified framework that harmonizes quantization across components through three synergistic steps: structural residual calibration proactively adjusts weights to compensate for activation-induced detail loss, harmonized scale optimization analytically balances quantization difficulty via closed-form solutions, and adaptive boundary refinement iteratively maintains this balance during optimization. Experiments show HarmoQ achieves substantial gains under aggressive compression, outperforming prior art by 0.46 dB on Set5 at 2-bit while delivering 3.2x speedup and 4x memory reduction on A100 GPUs. This work provides the first systematic analysis of weight-activation coupling in super-resolution quantization and establishes a principled solution for efficient high-quality image restoration.
>
---
#### [new 244] Cross-Modal Fine-Tuning of 3D Convolutional Foundation Models for ADHD Classification with Low-Rank Adaptation
- **分类: eess.IV; cs.CV; cs.LG; physics.med-ph**

- **简介: 该论文提出一种基于3D LoRA的跨模态微调方法，将CT预训练的3D卷积基础模型适配到MRI的ADHD分类任务，显著减少参数（113倍）并提升准确率，实现高效精准的儿童ADHD诊断。**

- **链接: [http://arxiv.org/pdf/2511.06163v1](http://arxiv.org/pdf/2511.06163v1)**

> **作者:** Jyun-Ping Kao; Shinyeong Rho; Shahar Lazarev; Hyun-Hae Cho; Fangxu Xing; Taehoon Shin; C. -C. Jay Kuo; Jonghye Woo
>
> **摘要:** Early diagnosis of attention-deficit/hyperactivity disorder (ADHD) in children plays a crucial role in improving outcomes in education and mental health. Diagnosing ADHD using neuroimaging data, however, remains challenging due to heterogeneous presentations and overlapping symptoms with other conditions. To address this, we propose a novel parameter-efficient transfer learning approach that adapts a large-scale 3D convolutional foundation model, pre-trained on CT images, to an MRI-based ADHD classification task. Our method introduces Low-Rank Adaptation (LoRA) in 3D by factorizing 3D convolutional kernels into 2D low-rank updates, dramatically reducing trainable parameters while achieving superior performance. In a five-fold cross-validated evaluation on a public diffusion MRI database, our 3D LoRA fine-tuning strategy achieved state-of-the-art results, with one model variant reaching 71.9% accuracy and another attaining an AUC of 0.716. Both variants use only 1.64 million trainable parameters (over 113x fewer than a fully fine-tuned foundation model). Our results represent one of the first successful cross-modal (CT-to-MRI) adaptations of a foundation model in neuroimaging, establishing a new benchmark for ADHD classification while greatly improving efficiency.
>
---
#### [new 245] Adaptive Sample-Level Framework Motivated by Distributionally Robust Optimization with Variance-Based Radius Assignment for Enhanced Neural Network Generalization Under Distribution Shift
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出一种无监督的自适应样本级DRO框架（Var-DRO），通过在线损失方差动态分配个体鲁棒性预算，解决传统DRO全局预算导致的保守性与资源错配问题，提升模型在分布偏移下的泛化性能。**

- **链接: [http://arxiv.org/pdf/2511.05568v1](http://arxiv.org/pdf/2511.05568v1)**

> **作者:** Aheer Sravon; Devdyuti Mazumder; Md. Ibrahim
>
> **备注:** Conference
>
> **摘要:** Distribution shifts and minority subpopulations frequently undermine the reliability of deep neural networks trained using Empirical Risk Minimization (ERM). Distributionally Robust Optimization (DRO) addresses this by optimizing for the worst-case risk within a neighborhood of the training distribution. However, conventional methods depend on a single, global robustness budget, which can lead to overly conservative models or a misallocation of robustness. We propose a variance-driven, adaptive, sample-level DRO (Var-DRO) framework that automatically identifies high-risk training samples and assigns a personalized robustness budget to each based on its online loss variance. Our formulation employs two-sided, KL-divergence-style bounds to constrain the ratio between adversarial and empirical weights for every sample. This results in a linear inner maximization problem over a convex polytope, which admits an efficient water-filling solution. To stabilize training, we introduce a warmup phase and a linear ramp schedule for the global cap on per-sample budgets, complemented by label smoothing for numerical robustness. Evaluated on CIFAR-10-C (corruptions), our method achieves the highest overall mean accuracy compared to ERM and KL-DRO. On Waterbirds, Var-DRO improves overall performance while matching or surpassing KL-DRO. On the original CIFAR-10 dataset, Var-DRO remains competitive, exhibiting the modest trade-off anticipated when prioritizing robustness. The proposed framework is unsupervised (requiring no group labels), straightforward to implement, theoretically sound, and computationally efficient.
>
---
#### [new 246] A Picture is Worth a Thousand (Correct) Captions: A Vision-Guided Judge-Corrector System for Multimodal Machine Translation
- **分类: cs.CL; cs.CV; cs.HC**

- **简介: 该论文面向英-印地语系多模态翻译任务，提出视觉引导的判断-修正系统，自动检测并修正训练数据中的翻译错误，结合LoRA微调模型，显著提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2511.07010v1](http://arxiv.org/pdf/2511.07010v1)**

> **作者:** Siddharth Betala; Kushan Raj; Vipul Betala; Rohan Saswade
>
> **备注:** Accepted at The 12th Workshop on Asian Translation, co-located with IJCLNLP-AACL 2025
>
> **摘要:** In this paper, we describe our system under the team name BLEU Monday for the English-to-Indic Multimodal Translation Task at WAT 2025. We participate in the text-only translation tasks for English-Hindi, English-Bengali, English-Malayalam, and English-Odia language pairs. We present a two-stage approach that addresses quality issues in the training data through automated error detection and correction, followed by parameter-efficient model fine-tuning. Our methodology introduces a vision-augmented judge-corrector pipeline that leverages multimodal language models to systematically identify and correct translation errors in the training data. The judge component classifies translations into three categories: correct, visually ambiguous (requiring image context), or mistranslated (poor translation quality). Identified errors are routed to specialized correctors: GPT-4o-mini regenerates captions requiring visual disambiguation, while IndicTrans2 retranslates cases with pure translation quality issues. This automated pipeline processes 28,928 training examples across four languages, correcting an average of 17.1% of captions per language. We then apply Low-Rank Adaptation (LoRA) to fine-tune the IndicTrans2 en-indic 200M distilled model on both original and corrected datasets. Training on corrected data yields consistent improvements, with BLEU score gains of +1.30 for English-Bengali on the evaluation set (42.00 -> 43.30) and +0.70 on the challenge set (44.90 -> 45.60), +0.60 for English-Odia on the evaluation set (41.00 -> 41.60), and +0.10 for English-Hindi on the challenge set (53.90 -> 54.00).
>
---
#### [new 247] ConnectomeBench: Can LLMs Proofread the Connectome?
- **分类: q-bio.NC; cs.AI; cs.CV; cs.LG**

- **简介: 论文提出ConnectomeBench基准，评估LLM在神经连接组数据校对中的三类任务：片段分类、分裂错误修正与合并错误检测，发现当前模型在前两项表现优异，有望辅助甚至替代人工校对。**

- **链接: [http://arxiv.org/pdf/2511.05542v1](http://arxiv.org/pdf/2511.05542v1)**

> **作者:** Jeff Brown; Andrew Kirjner Annika Vivekananthan; Ed Boyden
>
> **备注:** To appear in NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Connectomics - the mapping of neural connections in an organism's brain - currently requires extraordinary human effort to proofread the data collected from imaging and machine-learning assisted segmentation. With the growing excitement around using AI agents to automate important scientific tasks, we explore whether current AI systems can perform multiple tasks necessary for data proofreading. We introduce ConnectomeBench, a multimodal benchmark evaluating large language model (LLM) capabilities in three critical proofreading tasks: segment type identification, split error correction, and merge error detection. Using expert annotated data from two large open-source datasets - a cubic millimeter of mouse visual cortex and the complete Drosophila brain - we evaluate proprietary multimodal LLMs including Claude 3.7/4 Sonnet, o4-mini, GPT-4.1, GPT-4o, as well as open source models like InternVL-3 and NVLM. Our results demonstrate that current models achieve surprisingly high performance in segment identification (52-82% balanced accuracy vs. 20-25% chance) and binary/multiple choice split error correction (75-85% accuracy vs. 50% chance) while generally struggling on merge error identification tasks. Overall, while the best models still lag behind expert performance, they demonstrate promising capabilities that could eventually enable them to augment and potentially replace human proofreading in connectomics. Project page: https://github.com/jffbrwn2/ConnectomeBench and Dataset https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench/tree/main
>
---
#### [new 248] CAMP-VQA: Caption-Embedded Multimodal Perception for No-Reference Quality Assessment of Compressed Video
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: CAMP-VQA提出一种无参考视频质量评估方法，利用视觉-语言模型生成细粒度质量描述，融合元数据与时空特征，无需人工标注即可准确预测视频主观质量，显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.07290v1](http://arxiv.org/pdf/2511.07290v1)**

> **作者:** Xinyi Wang; Angeliki Katsenou; Junxiao Shen; David Bull
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** The prevalence of user-generated content (UGC) on platforms such as YouTube and TikTok has rendered no-reference (NR) perceptual video quality assessment (VQA) vital for optimizing video delivery. Nonetheless, the characteristics of non-professional acquisition and the subsequent transcoding of UGC video on sharing platforms present significant challenges for NR-VQA. Although NR-VQA models attempt to infer mean opinion scores (MOS), their modeling of subjective scores for compressed content remains limited due to the absence of fine-grained perceptual annotations of artifact types. To address these challenges, we propose CAMP-VQA, a novel NR-VQA framework that exploits the semantic understanding capabilities of large vision-language models. Our approach introduces a quality-aware prompting mechanism that integrates video metadata (e.g., resolution, frame rate, bitrate) with key fragments extracted from inter-frame variations to guide the BLIP-2 pretraining approach in generating fine-grained quality captions. A unified architecture has been designed to model perceptual quality across three dimensions: semantic alignment, temporal characteristics, and spatial characteristics. These multimodal features are extracted and fused, then regressed to video quality scores. Extensive experiments on a wide variety of UGC datasets demonstrate that our model consistently outperforms existing NR-VQA methods, achieving improved accuracy without the need for costly manual fine-grained annotations. Our method achieves the best performance in terms of average rank and linear correlation (SRCC: 0.928, PLCC: 0.938) compared to state-of-the-art methods. The source code and trained models, along with a user-friendly demo, are available at: https://github.com/xinyiW915/CAMP-VQA.
>
---
#### [new 249] Semi-distributed Cross-modal Air-Ground Relative Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出一种半分布式跨模态空地相对定位框架，解决传统多机器人SLAM耦合严重、带宽高问题。通过UGV融合多传感器进行局部优化，仅传输关键点与描述子，实现高精度、低带宽（<0.3 Mbps）的相对位姿估计。**

- **链接: [http://arxiv.org/pdf/2511.06749v1](http://arxiv.org/pdf/2511.06749v1)**

> **作者:** Weining Lu; Deer Bin; Lian Ma; Ming Ma; Zhihao Ma; Xiangyang Chen; Longfei Wang; Yixiao Feng; Zhouxian Jiang; Yongliang Shi; Bin Liang
>
> **备注:** 7 pages, 3 figures. Accepted by IROS 2025
>
> **摘要:** Efficient, accurate, and flexible relative localization is crucial in air-ground collaborative tasks. However, current approaches for robot relative localization are primarily realized in the form of distributed multi-robot SLAM systems with the same sensor configuration, which are tightly coupled with the state estimation of all robots, limiting both flexibility and accuracy. To this end, we fully leverage the high capacity of Unmanned Ground Vehicle (UGV) to integrate multiple sensors, enabling a semi-distributed cross-modal air-ground relative localization framework. In this work, both the UGV and the Unmanned Aerial Vehicle (UAV) independently perform SLAM while extracting deep learning-based keypoints and global descriptors, which decouples the relative localization from the state estimation of all agents. The UGV employs a local Bundle Adjustment (BA) with LiDAR, camera, and an IMU to rapidly obtain accurate relative pose estimates. The BA process adopts sparse keypoint optimization and is divided into two stages: First, optimizing camera poses interpolated from LiDAR-Inertial Odometry (LIO), followed by estimating the relative camera poses between the UGV and UAV. Additionally, we implement an incremental loop closure detection algorithm using deep learning-based descriptors to maintain and retrieve keyframes efficiently. Experimental results demonstrate that our method achieves outstanding performance in both accuracy and efficiency. Unlike traditional multi-robot SLAM approaches that transmit images or point clouds, our method only transmits keypoint pixels and their descriptors, effectively constraining the communication bandwidth under 0.3 Mbps. Codes and data will be publicly available on https://github.com/Ascbpiac/cross-model-relative-localization.git.
>
---
#### [new 250] EndoIR: Degradation-Agnostic All-in-One Endoscopic Image Restoration via Noise-Aware Routing Diffusion
- **分类: eess.IV; cs.AI; cs.CV; cs.RO**

- **简介: EndoIR提出一种去降解无关的内窥镜图像恢复框架，通过双域提示、双流扩散与噪声感知路由，单模型联合处理低光、烟雾、出血等复合退化，提升恢复效果与临床实用性。**

- **链接: [http://arxiv.org/pdf/2511.05873v1](http://arxiv.org/pdf/2511.05873v1)**

> **作者:** Tong Chen; Xinyu Ma; Long Bai; Wenyang Wang; Sun Yue; Luping Zhou
>
> **摘要:** Endoscopic images often suffer from diverse and co-occurring degradations such as low lighting, smoke, and bleeding, which obscure critical clinical details. Existing restoration methods are typically task-specific and often require prior knowledge of the degradation type, limiting their robustness in real-world clinical use. We propose EndoIR, an all-in-one, degradation-agnostic diffusion-based framework that restores multiple degradation types using a single model. EndoIR introduces a Dual-Domain Prompter that extracts joint spatial-frequency features, coupled with an adaptive embedding that encodes both shared and task-specific cues as conditioning for denoising. To mitigate feature confusion in conventional concatenation-based conditioning, we design a Dual-Stream Diffusion architecture that processes clean and degraded inputs separately, with a Rectified Fusion Block integrating them in a structured, degradation-aware manner. Furthermore, Noise-Aware Routing Block improves efficiency by dynamically selecting only noise-relevant features during denoising. Experiments on SegSTRONG-C and CEC datasets demonstrate that EndoIR achieves state-of-the-art performance across multiple degradation scenarios while using fewer parameters than strong baselines, and downstream segmentation experiments confirm its clinical utility.
>
---
#### [new 251] RRTS Dataset: A Benchmark Colonoscopy Dataset from Resource-Limited Settings for Computer-Aided Diagnosis Research
- **分类: eess.IV; cs.CV**

- **简介: 该论文构建了首个面向资源有限环境的结肠镜图像基准数据集BPD，包含2945张含真实伪影的图像，旨在提升AI在真实临床场景下的息肉检测与分割性能，并提供基线模型评估结果。**

- **链接: [http://arxiv.org/pdf/2511.06769v1](http://arxiv.org/pdf/2511.06769v1)**

> **作者:** Ridoy Chandra Shil; Ragib Abid; Tasnia Binte Mamun; Samiul Based Shuvo; Masfique Ahmed Bhuiyan; Jahid Ferdous
>
> **摘要:** Background and Objective: Colorectal cancer prevention relies on early detection of polyps during colonoscopy. Existing public datasets, such as CVC-ClinicDB and Kvasir-SEG, provide valuable benchmarks but are limited by small sample sizes, curated image selection, or lack of real-world artifacts. There remains a need for datasets that capture the complexity of clinical practice, particularly in resource-constrained settings. Methods: We introduce a dataset, BUET Polyp Dataset (BPD), of colonoscopy images collected using Olympus 170 and Pen- tax i-Scan series endoscopes under routine clinical conditions. The dataset contains images with corresponding expert-annotated binary masks, reflecting diverse challenges such as motion blur, specular highlights, stool artifacts, blood, and low-light frames. Annotations were manually reviewed by clinical experts to ensure quality. To demonstrate baseline performance, we provide bench- mark results for classification using VGG16, ResNet50, and InceptionV3, and for segmentation using UNet variants with VGG16, ResNet34, and InceptionV4 backbones. Results: The dataset comprises 1,288 images with polyps from 164 patients with corresponding ground-truth masks and 1,657 polyp-free images from 31 patients. Benchmarking experiments achieved up to 90.8% accuracy for binary classification (VGG16) and a maximum Dice score of 0.64 with InceptionV4-UNet for segmentation. Performance was lower compared to curated datasets, reflecting the real-world difficulty of images with artifacts and variable quality.
>
---
#### [new 252] Turbo-DDCM: Fast and Flexible Zero-Shot Diffusion-Based Image Compression
- **分类: eess.IV; cs.AI; cs.CV; eess.SP; stat.ML**

- **简介: 论文提出Turbo-DDCM，一种快速灵活的零样本扩散图像压缩方法，解决传统方法速度慢、计算开销大的问题。通过并行噪声向量组合与编码优化，大幅提升效率，并支持区域优先与PSNR控制两种灵活模式。**

- **链接: [http://arxiv.org/pdf/2511.06424v1](http://arxiv.org/pdf/2511.06424v1)**

> **作者:** Amit Vaisman; Guy Ohayon; Hila Manor; Michael Elad; Tomer Michaeli
>
> **备注:** Code is available at https://amitvaisman.github.io/turbo_ddcm/
>
> **摘要:** While zero-shot diffusion-based compression methods have seen significant progress in recent years, they remain notoriously slow and computationally demanding. This paper presents an efficient zero-shot diffusion-based compression method that runs substantially faster than existing methods, while maintaining performance that is on par with the state-of-the-art techniques. Our method builds upon the recently proposed Denoising Diffusion Codebook Models (DDCMs) compression scheme. Specifically, DDCM compresses an image by sequentially choosing the diffusion noise vectors from reproducible random codebooks, guiding the denoiser's output to reconstruct the target image. We modify this framework with Turbo-DDCM, which efficiently combines a large number of noise vectors at each denoising step, thereby significantly reducing the number of required denoising operations. This modification is also coupled with an improved encoding protocol. Furthermore, we introduce two flexible variants of Turbo-DDCM, a priority-aware variant that prioritizes user-specified regions and a distortion-controlled variant that compresses an image based on a target PSNR rather than a target BPP. Comprehensive experiments position Turbo-DDCM as a compelling, practical, and flexible image compression scheme.
>
---
#### [new 253] Task-Adaptive Low-Dose CT Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出一种任务自适应低剂量CT重建方法，通过冻结预训练分割网络作为正则项，提升重建图像的诊断关键结构保留能力，显著优于传统与联合训练方法，且可无缝集成至现有模型。**

- **链接: [http://arxiv.org/pdf/2511.07094v1](http://arxiv.org/pdf/2511.07094v1)**

> **作者:** Necati Sefercioglu; Mehmet Ozan Unal; Metin Ertas; Isa Yildirim
>
> **摘要:** Deep learning-based low-dose computed tomography reconstruction methods already achieve high performance on standard image quality metrics like peak signal-to-noise ratio and structural similarity index measure. Yet, they frequently fail to preserve the critical anatomical details needed for diagnostic tasks. This fundamental limitation hinders their clinical applicability despite their high metric scores. We propose a novel task-adaptive reconstruction framework that addresses this gap by incorporating a frozen pre-trained task network as a regularization term in the reconstruction loss function. Unlike existing joint-training approaches that simultaneously optimize both reconstruction and task networks, and risk diverging from satisfactory reconstructions, our method leverages a pre-trained task model to guide reconstruction training while still maintaining diagnostic quality. We validate our framework on a liver and liver tumor segmentation task. Our task-adaptive models achieve Dice scores up to 0.707, approaching the performance of full-dose scans (0.874), and substantially outperforming joint-training approaches (0.331) and traditional reconstruction methods (0.626). Critically, our framework can be integrated into any existing deep learning-based reconstruction model through simple loss function modification, enabling widespread adoption for task-adaptive optimization in clinical practice. Our codes are available at: https://github.com/itu-biai/task_adaptive_ct
>
---
#### [new 254] A Low-Rank Method for Vision Language Model Hallucination Mitigation in Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对自动驾驶中视觉语言模型的幻觉问题，提出一种无需外部参考的低秩自洽方法，通过分解句子嵌入矩阵，利用残差幅度排序候选描述，精准筛选低幻觉输出，提升准确率与推理效率。**

- **链接: [http://arxiv.org/pdf/2511.06496v1](http://arxiv.org/pdf/2511.06496v1)**

> **作者:** Keke Long; Jiacheng Guo; Tianyun Zhang; Hongkai Yu; Xiaopeng Li
>
> **摘要:** Vision Language Models (VLMs) are increasingly used in autonomous driving to help understand traffic scenes, but they sometimes produce hallucinations, which are false details not grounded in the visual input. Detecting and mitigating hallucinations is challenging when ground-truth references are unavailable and model internals are inaccessible. This paper proposes a novel self-contained low-rank approach to automatically rank multiple candidate captions generated by multiple VLMs based on their hallucination levels, using only the captions themselves without requiring external references or model access. By constructing a sentence-embedding matrix and decomposing it into a low-rank consensus component and a sparse residual, we use the residual magnitude to rank captions: selecting the one with the smallest residual as the most hallucination-free. Experiments on the NuScenes dataset demonstrate that our approach achieves 87% selection accuracy in identifying hallucination-free captions, representing a 19% improvement over the unfiltered baseline and a 6-10% improvement over multi-agent debate method. The sorting produced by sparse error magnitudes shows strong correlation with human judgments of hallucinations, validating our scoring mechanism. Additionally, our method, which can be easily parallelized, reduces inference time by 51-67% compared to debate approaches, making it practical for real-time autonomous driving applications.
>
---
#### [new 255] Hierarchical Spatial-Frequency Aggregation for Spectral Deconvolution Imaging
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对光谱反卷积成像中数据依赖算子导致重建困难的问题，提出HSFAUT框架，通过频域分解与空间-频率注意力机制，实现高效高保真光谱重建，显著提升性能并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2511.06751v1](http://arxiv.org/pdf/2511.06751v1)**

> **作者:** Tao Lv; Daoming Zhou; Chenglong Huang; Chongde Zi; Linsen Chen; Xun Cao
>
> **备注:** Under Review at TPAMI
>
> **摘要:** Computational spectral imaging (CSI) achieves real-time hyperspectral imaging through co-designed optics and algorithms, but typical CSI methods suffer from a bulky footprint and limited fidelity. Therefore, Spectral Deconvolution imaging (SDI) methods based on PSF engineering have been proposed to achieve high-fidelity compact CSI design recently. However, the composite convolution-integration operations of SDI render the normal-equation coefficient matrix scene-dependent, which hampers the efficient exploitation of imaging priors and poses challenges for accurate reconstruction. To tackle the inherent data-dependent operators in SDI, we introduce a Hierarchical Spatial-Spectral Aggregation Unfolding Framework (HSFAUF). By decomposing subproblems and projecting them into the frequency domain, HSFAUF transforms nonlinear processes into linear mappings, thereby enabling efficient solutions. Furthermore, to integrate spatial-spectral priors during iterative refinement, we propose a Spatial-Frequency Aggregation Transformer (SFAT), which explicitly aggregates information across spatial and frequency domains. By integrating SFAT into HSFAUF, we develop a Transformer-based deep unfolding method, \textbf{H}ierarchical \textbf{S}patial-\textbf{F}requency \textbf{A}ggregation \textbf{U}nfolding \textbf{T}ransformer (HSFAUT), to solve the inverse problem of SDI. Systematic simulated and real experiments show that HSFAUT surpasses SOTA methods with cheaper memory and computational costs, while exhibiting optimal performance on different SDI systems.
>
---
## 更新

#### [replaced 001] MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09199v3](http://arxiv.org/pdf/2412.09199v3)**

> **作者:** Qiwen Gu; Xufei Wang; Junqiao Zhao; Siyue Tao; Tiantian Feng; Ziqiao Wang; Guang Chen
>
> **备注:** 15 pages, 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Visual Place Recognition (VPR) enables robust localization through image retrieval based on learned descriptors. However, drastic appearance variations of images at the same place caused by viewpoint changes can lead to inconsistent supervision signals, thereby degrading descriptor learning. Existing methods either rely on manually defined cropping rules or labeled data for view differentiation, but they suffer from two major limitations: (1) reliance on labels or handcrafted rules restricts generalization capability; (2) even within the same view direction, occlusions can introduce feature ambiguity. To address these issues, we propose MutualVPR, a mutual learning framework that integrates unsupervised view self-classification and descriptor learning. We first group images by geographic coordinates, then iteratively refine the clusters using K-means to dynamically assign place categories without orientation labels. Specifically, we adopt a DINOv2-based encoder to initialize the clustering. During training, the encoder and clustering co-evolve, progressively separating drastic appearance variations of the same place and enabling consistent supervision. Furthermore, we find that capturing fine-grained image differences at a place enhances robustness. Experiments demonstrate that MutualVPR achieves state-of-the-art (SOTA) performance across multiple datasets, validating the effectiveness of our framework in improving view direction generalization, occlusion robustness.
>
---
#### [replaced 002] DNOI-4DRO: Deep 4D Radar Odometry with Differentiable Neural-Optimization Iterations
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12310v2](http://arxiv.org/pdf/2505.12310v2)**

> **作者:** Shouyi Lu; Huanyu Zhou; Guirong Zhuo; Xiao Tang
>
> **备注:** 9 pages,5 figures
>
> **摘要:** A novel learning-optimization-combined 4D radar odometry model, named DNOI-4DRO, is proposed in this paper. The proposed model seamlessly integrates traditional geometric optimization with end-to-end neural network training, leveraging an innovative differentiable neural-optimization iteration operator. In this framework, point-wise motion flow is first estimated using a neural network, followed by the construction of a cost function based on the relationship between point motion and pose in 3D space. The radar pose is then refined using Gauss-Newton updates. Additionally, we design a dual-stream 4D radar backbone that integrates multi-scale geometric features and clustering-based class-aware features to enhance the representation of sparse 4D radar point clouds. Extensive experiments on the VoD and Snail-Radar datasets demonstrate the superior performance of our model, which outperforms recent classical and learning-based approaches. Notably, our method even achieves results comparable to A-LOAM with mapping optimization using LiDAR point clouds as input. Our models and code will be publicly released.
>
---
#### [replaced 003] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05853v4](http://arxiv.org/pdf/2412.05853v4)**

> **作者:** Qing Wu; Hongjiang Wei; Jingyi Yu; Yuyao Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Ring artifacts are prevalent in 3D cone-beam computed tomography (CBCT) due to non-ideal responses of X-ray detectors, substantially affecting image quality and diagnostic reliability. Existing state-of-the-art (SOTA) ring artifact reduction (RAR) methods rely on supervised learning with large-scale paired CT datasets. While effective in-domain, supervised methods tend to struggle to fully capture the physical characteristics of ring artifacts, leading to pronounced performance drops in complex real-world acquisitions. Moreover, their scalability to 3D CBCT is limited by high memory demands. In this work, we propose Riner, a new unsupervised RAR method. Based on a theoretical analysis of ring artifact formation, we reformulate RAR as a multi-parameter inverse problem, where the non-ideal responses of X-ray detectors are parameterized as solvable physical variables. Using a new differentiable forward model, Riner can jointly learn the implicit neural representation of artifact-free images and estimate the physical parameters directly from CT measurements, without external training data. Additionally, Riner is memory-friendly due to its ray-based optimization, enhancing its usability in large-scale 3D CBCT. Experiments on both simulated and real-world datasets show Riner outperforms existing SOTA supervised methods.
>
---
#### [replaced 004] P3P Made Easy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.01312v3](http://arxiv.org/pdf/2508.01312v3)**

> **作者:** Seong Hun Lee; Patrick Vandewalle; Javier Civera
>
> **摘要:** We revisit the classical Perspective-Three-Point (P3P) problem, which aims to recover the absolute pose of a calibrated camera from three 2D-3D correspondences. It has long been known that P3P can be reduced to a quartic polynomial with analytically simple and computationally efficient coefficients. However, this elegant formulation has been largely overlooked in modern literature. Building on the theoretical foundation that traces back to Grunert's work in 1841, we propose a compact algebraic solver that achieves accuracy and runtime comparable to state-of-the-art methods. Our results show that this classical formulation remains highly competitive when implemented with modern insights, offering an excellent balance between simplicity, efficiency, and accuracy.
>
---
#### [replaced 005] Scalable Offline Metrics for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.08571v2](http://arxiv.org/pdf/2510.08571v2)**

> **作者:** Animikh Aich; Adwait Kulkarni; Eshed Ohn-Bar
>
> **备注:** Accepted at IROS 2025 (IEEE/RSJ International Conference on Intelligent Robots and Systems); typos corrected
>
> **摘要:** Real-world evaluation of perception-based planning models for robotic systems, such as autonomous vehicles, can be safely and inexpensively conducted offline, i.e. by computing model prediction error over a pre-collected validation dataset with ground-truth annotations. However, extrapolating from offline model performance to online settings remains a challenge. In these settings, seemingly minor errors can compound and result in test-time infractions or collisions. This relationship is understudied, particularly across diverse closed-loop metrics and complex urban maneuvers. In this work, we revisit this undervalued question in policy evaluation through an extensive set of experiments across diverse conditions and metrics. Based on analysis in simulation, we find an even worse correlation between offline and online settings than reported by prior studies, casting doubts on the validity of current evaluation practices and metrics for driving policies. Next, we bridge the gap between offline and online evaluation. We investigate an offline metric based on epistemic uncertainty, which aims to capture events that are likely to cause errors in closed-loop settings. The resulting metric achieves over 13% improvement in correlation compared to previous offline metrics. We further validate the generalization of our findings beyond the simulation environment in real-world settings, where even greater gains are observed.
>
---
#### [replaced 006] Improving Contactless Fingerprint Recognition with Robust 3D Feature Extraction and Graph Embedding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.08782v2](http://arxiv.org/pdf/2409.08782v2)**

> **作者:** Yuwei Jia; Siyang Zheng; Fei Feng; Zhe Cui; Fei Su
>
> **备注:** Oral presentation accepted at the 2025 IEEE International Joint Conference on Biometrics (IJCB) 2025, Osaka, Japan (9/8-9/11/2025)
>
> **摘要:** Contactless fingerprint has gained lots of attention in recent fingerprint studies. However, most existing contactless fingerprint algorithms treat contactless fingerprints as 2D plain fingerprints, and still utilize traditional contact-based 2D fingerprints recognition methods. This recognition approach lacks consideration of the modality difference between contactless and contact fingerprints, especially the intrinsic 3D features in contactless fingerprints. This paper proposes a novel contactless fingerprint recognition algorithm that captures the revealed 3D feature of contactless fingerprints rather than the plain 2D feature. The proposed method first recovers 3D features from the input contactless fingerprint, including the 3D shape model and 3D fingerprint feature (minutiae, orientation, etc.). Then, a novel 3D graph matching method is proposed according to the extracted 3D feature. Additionally, the proposed method is able to perform robust 3D feature extractions on various contactless fingerprints across multiple finger poses. The results of the experiments on contactless fingerprint databases show that the proposed method successfully improves the matching accuracy of contactless fingerprints. Exceptionally, our method performs stably across multiple poses of contactless fingerprints due to 3D embeddings, which is a great advantage compared to 2D-based previous contactless fingerprint recognition algorithms.
>
---
#### [replaced 007] MGSO: Monocular Real-time Photometric SLAM with Efficient 3D Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.13055v3](http://arxiv.org/pdf/2409.13055v3)**

> **作者:** Yan Song Hu; Nicolas Abboud; Muhammad Qasim Ali; Adam Srebrnjak Yang; Imad Elhajj; Daniel Asmar; Yuhao Chen; John S. Zelek
>
> **备注:** This is the pre-print version of a work that has been published in ICRA 2025 with doi: 10.1109/ICRA55743.2025.11127380. This version may no longer be accessible without notice. Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses. Please cite the official version
>
> **摘要:** Real-time SLAM with dense 3D mapping is computationally challenging, especially on resource-limited devices. The recent development of 3D Gaussian Splatting (3DGS) offers a promising approach for real-time dense 3D reconstruction. However, existing 3DGS-based SLAM systems struggle to balance hardware simplicity, speed, and map quality. Most systems excel in one or two of the aforementioned aspects but rarely achieve all. A key issue is the difficulty of initializing 3D Gaussians while concurrently conducting SLAM. To address these challenges, we present Monocular GSO (MGSO), a novel real-time SLAM system that integrates photometric SLAM with 3DGS. Photometric SLAM provides dense structured point clouds for 3DGS initialization, accelerating optimization and producing more efficient maps with fewer Gaussians. As a result, experiments show that our system generates reconstructions with a balance of quality, memory efficiency, and speed that outperforms the state-of-the-art. Furthermore, our system achieves all results using RGB inputs. We evaluate the Replica, TUM-RGBD, and EuRoC datasets against current live dense reconstruction systems. Not only do we surpass contemporary systems, but experiments also show that we maintain our performance on laptop hardware, making it a practical solution for robotics, A/R, and other real-time applications.
>
---
#### [replaced 008] Quantum Doubly Stochastic Transformers
- **分类: cs.LG; cs.AI; cs.CE; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.16275v2](http://arxiv.org/pdf/2504.16275v2)**

> **作者:** Jannis Born; Filip Skogh; Kahn Rhrissorrakrai; Filippo Utro; Nico Wagner; Aleksandros Sobczyk
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** At the core of the Transformer, the softmax normalizes the attention matrix to be right stochastic. Previous research has shown that this often de-stabilizes training and that enforcing the attention matrix to be doubly stochastic (through Sinkhorn's algorithm) consistently improves performance across different tasks, domains and Transformer flavors. However, Sinkhorn's algorithm is iterative, approximative, non-parametric and thus inflexible w.r.t. the obtained doubly stochastic matrix (DSM). Recently, it has been proven that DSMs can be obtained with a parametric quantum circuit, yielding a novel quantum inductive bias for DSMs with no known classical analogue. Motivated by this, we demonstrate the feasibility of a hybrid classical-quantum doubly stochastic Transformer (QDSFormer) that replaces the softmax in the self-attention layer with a variational quantum circuit. We study the expressive power of the circuit and find that it yields more diverse DSMs that better preserve information than classical operators. Across multiple small-scale object recognition tasks, we find that our QDSFormer consistently surpasses both a standard ViT and other doubly stochastic Transformers. Beyond the Sinkformer, this comparison includes a novel quantum-inspired doubly stochastic Transformer (based on QR decomposition) that can be of independent interest. Our QDSFormer also shows improved training stability and lower performance variation suggesting that it may mitigate the notoriously unstable training of ViTs on small-scale data.
>
---
#### [replaced 009] A Step Toward World Models: A Survey on Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2511.02097v2](http://arxiv.org/pdf/2511.02097v2)**

> **作者:** Peng-Fei Zhang; Ying Cheng; Xiaofan Sun; Shijie Wang; Fengling Li; Lei Zhu; Heng Tao Shen
>
> **备注:** 24 pages, 5 figures
>
> **摘要:** Autonomous agents are increasingly expected to operate in complex, dynamic, and uncertain environments, performing tasks such as manipulation, navigation, and decision-making. Achieving these capabilities requires agents to understand the underlying mechanisms and dynamics of the world, moving beyond reactive control or simple replication of observed states. This motivates the development of world models as internal representations that encode environmental states, capture dynamics, and support prediction, planning, and reasoning. Despite growing interest, the definition, scope, architectures, and essential capabilities of world models remain ambiguous. In this survey, we go beyond prescribing a fixed definition and limiting our scope to methods explicitly labeled as world models. Instead, we examine approaches that exhibit the core capabilities of world models through a review of methods in robotic manipulation. We analyze their roles across perception, prediction, and control, identify key challenges and solutions, and distill the core components, capabilities, and functions that a fully realized world model should possess. Building on this analysis, we aim to motivate further development toward generalizable and practical world models for robotics.
>
---
#### [replaced 010] UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.06165v2](http://arxiv.org/pdf/2509.06165v2)**

> **作者:** Huy Le; Nhat Chung; Tung Kieu; Jingkang Yang; Ngan Le
>
> **备注:** 11 pages, 7 figures. Accepted at WACV 2026
>
> **摘要:** Video Scene Graph Generation (VidSGG) aims to represent dynamic visual content by detecting objects and modeling their temporal interactions as structured graphs. Prior studies typically target either coarse-grained box-level or fine-grained panoptic pixel-level VidSGG, often requiring task-specific architectures and multi-stage training pipelines. In this paper, we present UNO (UNified Object-centric VidSGG), a single-stage, unified framework that jointly addresses both tasks within an end-to-end architecture. UNO is designed to minimize task-specific modifications and maximize parameter sharing, enabling generalization across different levels of visual granularity. The core of UNO is an extended slot attention mechanism that decomposes visual features into object and relation slots. To ensure robust temporal modeling, we introduce object temporal consistency learning, which enforces consistent object representations across frames without relying on explicit tracking modules. Additionally, a dynamic triplet prediction module links relation slots to corresponding object pairs, capturing evolving interactions over time. We evaluate UNO on standard box-level and pixel-level VidSGG benchmarks. Results demonstrate that UNO not only achieves competitive performance across both tasks but also offers improved efficiency through a unified, object-centric design.
>
---
#### [replaced 011] FedVLM: Scalable Personalized Vision-Language Models through Federated Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17088v2](http://arxiv.org/pdf/2507.17088v2)**

> **作者:** Arkajyoti Mitra; Afia Anjum; Paul Agbaje; Mert Pesé; Habeeb Olufowobi
>
> **备注:** Accepted to ECAI 2025 Main Track
>
> **摘要:** Vision-language models (VLMs) demonstrate impressive zero-shot and few-shot learning capabilities, making them essential for several downstream tasks. However, fine-tuning these models at scale remains challenging, particularly in federated environments where data is decentralized and non-iid across clients. Existing parameter-efficient tuning methods like LoRA (Low-Rank Adaptation) reduce computational overhead but struggle with heterogeneous client data, leading to suboptimal generalization. To address these challenges, we propose FedVLM, a federated LoRA fine-tuning framework that enables decentralized adaptation of VLMs while preserving model privacy and reducing reliance on centralized training. To further tackle data heterogeneity, we introduce personalized LoRA (pLoRA), which dynamically adapts LoRA parameters to each client's unique data distribution, significantly improving local adaptation while maintaining global model aggregation. Experiments on the RLAIF-V dataset show that pLoRA improves client-specific performance by 24.5% over standard LoRA, demonstrating superior adaptation in non-iid settings. FedVLM provides a scalable and efficient solution for fine-tuning VLMs in federated settings, advancing personalized adaptation in distributed learning scenarios.
>
---
#### [replaced 012] Diffusion Implicit Policy for Unpaired Scene-aware Motion Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02261v2](http://arxiv.org/pdf/2412.02261v2)**

> **作者:** Jingyu Gong; Chong Zhang; Fengqi Liu; Ke Fan; Qianyu Zhou; Xin Tan; Zhizhong Zhang; Yuan Xie
>
> **摘要:** Scene-aware motion synthesis has been widely researched recently due to its numerous applications. Prevailing methods rely heavily on paired motion-scene data, while it is difficult to generalize to diverse scenes when trained only on a few specific ones. Thus, we propose a unified framework, termed Diffusion Implicit Policy (DIP), for scene-aware motion synthesis, where paired motion-scene data are no longer necessary. In this paper, we disentangle human-scene interaction from motion synthesis during training, and then introduce an interaction-based implicit policy into motion diffusion during inference. Synthesized motion can be derived through iterative diffusion denoising and implicit policy optimization, thus motion naturalness and interaction plausibility can be maintained simultaneously. For long-term motion synthesis, we introduce motion blending in joint rotation power space. The proposed method is evaluated on synthesized scenes with ShapeNet furniture, and real scenes from PROX and Replica. Results show that our framework presents better motion naturalness and interaction plausibility than cutting-edge methods. This also indicates the feasibility of utilizing the DIP for motion synthesis in more general tasks and versatile scenes. Code will be publicly available at https://github.com/jingyugong/DIP.
>
---
#### [replaced 013] Physics-informed DeepCT: Sinogram Wavelet Decomposition Meets Masked Diffusion
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2501.09935v3](http://arxiv.org/pdf/2501.09935v3)**

> **作者:** Zekun Zhou; Tan Liu; Bing Yu; Yanru Gong; Liu Shi; Qiegen Liu
>
> **摘要:** Diffusion model shows remarkable potential on sparse-view computed tomography (SVCT) reconstruction. However, when a network is trained on a limited sample space, its generalization capability may be constrained, which degrades performance on unfamiliar data. For image generation tasks, this can lead to issues such as blurry details and inconsistencies between regions. To alleviate this problem, we propose a Sinogram-based Wavelet random decomposition And Random mask diffusion Model (SWARM) for SVCT reconstruction. Specifically, introducing a random mask strategy in the sinogram effectively expands the limited training sample space. This enables the model to learn a broader range of data distributions, enhancing its understanding and generalization of data uncertainty. In addition, applying a random training strategy to the high-frequency components of the sinogram wavelet enhances feature representation and improves the ability to capture details in different frequency bands, thereby improving performance and robustness. Two-stage iterative reconstruction method is adopted to ensure the global consistency of the reconstructed image while refining its details. Experimental results demonstrate that SWARM outperforms competing approaches in both quantitative and qualitative performance across various datasets.
>
---
#### [replaced 014] GUARD: Guideline Upholding Test through Adaptive Role-play and Jailbreak Diagnostics for LLMs
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.20325v2](http://arxiv.org/pdf/2508.20325v2)**

> **作者:** Haibo Jin; Ruoxi Chen; Peiyan Zhang; Andy Zhou; Haohan Wang
>
> **备注:** 54 pages
>
> **摘要:** As Large Language Models become increasingly integral to various domains, their potential to generate harmful responses has prompted significant societal and regulatory concerns. In response, governments have issued ethics guidelines to promote the development of trustworthy AI. However, these guidelines are typically high-level demands for developers and testers, leaving a gap in translating them into actionable testing questions to verify LLM compliance. To address this challenge, we introduce GUARD (\textbf{G}uideline \textbf{U}pholding Test through \textbf{A}daptive \textbf{R}ole-play and Jailbreak \textbf{D}iagnostics), a testing method designed to operationalize guidelines into specific guideline-violating questions that assess LLM adherence. To implement this, GUARD uses automated generation of guideline-violating questions based on government-issued guidelines, thereby testing whether responses comply with these guidelines. When responses directly violate guidelines, GUARD reports inconsistencies. Furthermore, for responses that do not directly violate guidelines, GUARD integrates the concept of ``jailbreaks'' to diagnostics, named GUARD-JD, which creates scenarios that provoke unethical or guideline-violating responses, effectively identifying potential scenarios that could bypass built-in safety mechanisms. Our method finally culminates in a compliance report, delineating the extent of adherence and highlighting any violations. We have empirically validated the effectiveness of GUARD on seven LLMs, including Vicuna-13B, LongChat-7B, Llama2-7B, Llama-3-8B, GPT-3.5, GPT-4, GPT-4o, and Claude-3.7, by testing compliance under three government-issued guidelines and conducting jailbreak diagnostics. Additionally, GUARD-JD can transfer jailbreak diagnostics to vision-language models, demonstrating its usage in promoting reliable LLM-based applications.
>
---
#### [replaced 015] Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21844v2](http://arxiv.org/pdf/2505.21844v2)**

> **作者:** Mehrdad Noori; David Osowiechi; Gustavo Adolfo Vargas Hakim; Ali Bahri; Moslem Yazdanpanah; Sahar Dastani; Farzad Beizaee; Ismail Ben Ayed; Christian Desrosiers
>
> **摘要:** Recently, test-time adaptation has attracted wide interest in the context of vision-language models for image classification. However, to the best of our knowledge, the problem is completely overlooked in dense prediction tasks such as Open-Vocabulary Semantic Segmentation (OVSS). In response, we propose a novel TTA method tailored to adapting VLMs for segmentation during test time. Unlike TTA methods for image classification, our Multi-Level and Multi-Prompt (MLMP) entropy minimization integrates features from intermediate vision-encoder layers and is performed with different text-prompt templates at both the global CLS token and local pixel-wise levels. Our approach could be used as plug-and-play for any segmentation network, does not require additional training data or labels, and remains effective even with a single test sample. Furthermore, we introduce a comprehensive OVSS TTA benchmark suite, which integrates a rigorous evaluation protocol, nine segmentation datasets, 15 common synthetic corruptions, and additional real and rendered domain shifts, \textbf{with a total of 87 distinct test scenarios}, establishing a standardized and comprehensive testbed for future TTA research in open-vocabulary segmentation. Our experiments on this suite demonstrate that our segmentation-tailored method consistently delivers significant gains over direct adoption of TTA classification baselines. Code and data are available at https://github.com/dosowiechi/MLMP.
>
---
#### [replaced 016] Controllable Hybrid Captioner for Improved Long-form Video Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.17047v4](http://arxiv.org/pdf/2507.17047v4)**

> **作者:** Kuleen Sasse; Efsun Sarioglu Kayi; Arun Reddy
>
> **摘要:** Video data, especially long-form video, is extremely dense and high-dimensional. Text-based summaries of video content offer a way to represent query-relevant content in a much more compact manner than raw video. In addition, textual representations are easily ingested by state-of-the-art large language models (LLMs), which enable reasoning over video content to answer complex natural language queries. To solve this issue, we rely on the progressive construction of a text-based memory by a video captioner operating on shorter chunks of the video, where spatio-temporal modeling is computationally feasible. We explore ways to improve the quality of the activity log comprised solely of short video captions. Because the video captions tend to be focused on human actions, and questions may pertain to other information in the scene, we seek to enrich the memory with static scene descriptions using Vision Language Models (VLMs). Our video understanding system relies on the LaViLa video captioner in combination with a LLM to answer questions about videos. We first explored different ways of partitioning the video into meaningful segments such that the textual descriptions more accurately reflect the structure of the video content. Furthermore, we incorporated static scene descriptions into the captioning pipeline using LLaVA VLM, resulting in a more detailed and complete caption log and expanding the space of questions that are answerable from the textual memory. Finally, we have successfully fine-tuned the LaViLa video captioner to produce both action and scene captions, significantly improving the efficiency of the captioning pipeline compared to using separate captioning models for the two tasks. Our model, controllable hybrid captioner, can alternate between different types of captions according to special input tokens that signals scene changes detected in the video.
>
---
#### [replaced 017] Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2511.04665v2](http://arxiv.org/pdf/2511.04665v2)**

> **作者:** Kaifeng Zhang; Shuo Sha; Hanxiao Jiang; Matthew Loper; Hyunjong Song; Guangyan Cai; Zhuo Xu; Xiaochen Hu; Changxi Zheng; Yunzhu Li
>
> **备注:** The first two authors contributed equally. Website: https://real2sim-eval.github.io/
>
> **摘要:** Robotic manipulation policies are advancing rapidly, but their direct evaluation in the real world remains costly, time-consuming, and difficult to reproduce, particularly for tasks involving deformable objects. Simulation provides a scalable and systematic alternative, yet existing simulators often fail to capture the coupled visual and physical complexity of soft-body interactions. We present a real-to-sim policy evaluation framework that constructs soft-body digital twins from real-world videos and renders robots, objects, and environments with photorealistic fidelity using 3D Gaussian Splatting. We validate our approach on representative deformable manipulation tasks, including plush toy packing, rope routing, and T-block pushing, demonstrating that simulated rollouts correlate strongly with real-world execution performance and reveal key behavioral patterns of learned policies. Our results suggest that combining physics-informed reconstruction with high-quality rendering enables reproducible, scalable, and accurate evaluation of robotic manipulation policies. Website: https://real2sim-eval.github.io/
>
---
#### [replaced 018] Enhancing Multimodal Medical Image Classification using Cross-Graph Modal Contrastive Learning
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.17494v5](http://arxiv.org/pdf/2410.17494v5)**

> **作者:** Jun-En Ding; Chien-Chin Hsu; Chi-Hsiang Chu; Shuqiang Wang; Feng Liu
>
> **摘要:** The classification of medical images is a pivotal aspect of disease diagnosis, often enhanced by deep learning techniques. However, traditional approaches typically focus on unimodal medical image data, neglecting the integration of diverse non-image patient data. This paper proposes a novel Cross-Graph Modal Contrastive Learning (CGMCL) framework for multimodal structured data from different data domains to improve medical image classification. The model effectively integrates both image and non-image data by constructing cross-modality graphs and leveraging contrastive learning to align multimodal features in a shared latent space. An inter-modality feature scaling module further optimizes the representation learning process by reducing the gap between heterogeneous modalities. The proposed approach is evaluated on two datasets: a Parkinson's disease (PD) dataset and a public melanoma dataset. Results demonstrate that CGMCL outperforms conventional unimodal methods in accuracy, interpretability, and early disease prediction. Additionally, the method shows superior performance in multi-class melanoma classification. The CGMCL framework provides valuable insights into medical image classification while offering improved disease interpretability and predictive capabilities.
>
---
#### [replaced 019] Evaluating Cell AI Foundation Models in Kidney Pathology with Human-in-the-Loop Enrichment
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2411.00078v2](http://arxiv.org/pdf/2411.00078v2)**

> **作者:** Junlin Guo; Siqi Lu; Can Cui; Ruining Deng; Tianyuan Yao; Zhewen Tao; Yizhe Lin; Marilyn Lionts; Quan Liu; Juming Xiong; Yu Wang; Shilin Zhao; Catie Chang; Mitchell Wilkes; Mengmeng Yin; Haichun Yang; Yuankai Huo
>
> **摘要:** Training AI foundation models has emerged as a promising large-scale learning approach for addressing real-world healthcare challenges, including digital pathology. While many of these models have been developed for tasks like disease diagnosis and tissue quantification using extensive and diverse training datasets, their readiness for deployment on some arguably simplest tasks, such as nuclei segmentation within a single organ (e.g., the kidney), remains uncertain. This paper seeks to answer this key question, "How good are we?", by thoroughly evaluating the performance of recent cell foundation models on a curated multi-center, multi-disease, and multi-species external testing dataset. Additionally, we tackle a more challenging question, "How can we improve?", by developing and assessing human-in-the-loop data enrichment strategies aimed at enhancing model performance while minimizing the reliance on pixel-level human annotation. To address the first question, we curated a multicenter, multidisease, and multispecies dataset consisting of 2,542 kidney whole slide images (WSIs). Three state-of-the-art (SOTA) cell foundation models-Cellpose, StarDist, and CellViT-were selected for evaluation. To tackle the second question, we explored data enrichment algorithms by distilling predictions from the different foundation models with a human-in-the-loop framework, aiming to further enhance foundation model performance with minimal human efforts. Our experimental results showed that all three foundation models improved over their baselines with model fine-tuning with enriched data. Interestingly, the baseline model with the highest F1 score does not yield the best segmentation outcomes after fine-tuning. This study establishes a benchmark for the development and deployment of cell vision foundation models tailored for real-world data applications.
>
---
#### [replaced 020] Free-T2M: Robust Text-to-Motion Generation for Humanoid Robots via Frequency-Domain
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.18232v2](http://arxiv.org/pdf/2501.18232v2)**

> **作者:** Wenshuo Chen; Haozhe Jia; Songning Lai; Lei Wang; Yuqi Lin; Hongru Xiao; Lijie Hu; Yutao Yue
>
> **摘要:** Enabling humanoid robots to synthesize complex, physically coherent motions from natural language commands is a cornerstone of autonomous robotics and human-robot interaction. While diffusion models have shown promise in this text-to-motion (T2M) task, they often generate semantically flawed or unstable motions, limiting their applicability to real-world robots. This paper reframes the T2M problem from a frequency-domain perspective, revealing that the generative process mirrors a hierarchical control paradigm. We identify two critical phases: a semantic planning stage, where low-frequency components establish the global motion trajectory, and a fine-grained execution stage, where high-frequency details refine the movement. To address the distinct challenges of each phase, we introduce Frequency enhanced text-to-motion (Free-T2M), a framework incorporating stage-specific frequency-domain consistency alignment. We design a frequency-domain temporal-adaptive module to modulate the alignment effects of different frequency bands. These designs enforce robustness in the foundational semantic plan and enhance the accuracy of detailed execution. Extensive experiments show our method dramatically improves motion quality and semantic correctness. Notably, when applied to the StableMoFusion baseline, Free-T2M reduces the FID from 0.152 to 0.060, establishing a new state-of-the-art within diffusion architectures. These findings underscore the critical role of frequency-domain insights for generating robust and reliable motions, paving the way for more intuitive natural language control of robots.
>
---
#### [replaced 021] MM-UNet: Morph Mamba U-shaped Convolutional Networks for Retinal Vessel Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2511.02193v2](http://arxiv.org/pdf/2511.02193v2)**

> **作者:** Jiawen Liu; Yuanbo Zeng; Jiaming Liang; Yizhen Yang; Yiheng Zhang; Enhui Cai; Xiaoqi Sheng; Hongmin Cai
>
> **备注:** This paper was accepted by IEEE BIBM 2025 conference
>
> **摘要:** Accurate detection of retinal vessels plays a critical role in reflecting a wide range of health status indicators in the clinical diagnosis of ocular diseases. Recently, advances in deep learning have led to a surge in retinal vessel segmentation methods, which have significantly contributed to the quantitative analysis of vascular morphology. However, retinal vasculature differs significantly from conventional segmentation targets in that it consists of extremely thin and branching structures, whose global morphology varies greatly across images. These characteristics continue to pose challenges to segmentation precision and robustness. To address these issues, we propose MM-UNet, a novel architecture tailored for efficient retinal vessel segmentation. The model incorporates Morph Mamba Convolution layers, which replace pointwise convolutions to enhance branching topological perception through morph, state-aware feature sampling. Additionally, Reverse Selective State Guidance modules integrate reverse guidance theory with state-space modeling to improve geometric boundary awareness and decoding efficiency. Extensive experiments conducted on two public retinal vessel segmentation datasets demonstrate the superior performance of the proposed method in segmentation accuracy. Compared to the existing approaches, MM-UNet achieves F1-score gains of 1.64 % on DRIVE and 1.25 % on STARE, demonstrating its effectiveness and advancement. The project code is public via https://github.com/liujiawen-jpg/MM-UNet.
>
---
#### [replaced 022] Continual Learning with Synthetic Boundary Experience Blending
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.23534v2](http://arxiv.org/pdf/2507.23534v2)**

> **作者:** Chih-Fan Hsu; Ming-Ching Chang; Wei-Chao Chen
>
> **摘要:** Continual learning (CL) seeks to mitigate catastrophic forgetting when models are trained with sequential tasks. A common approach, experience replay (ER), stores past exemplars but only sparsely approximates the data distribution, yielding fragile and oversimplified decision boundaries. We address this limitation by introducing synthetic boundary data (SBD), generated via differential privacy: inspired noise into latent features to create boundary-adjacent representations that implicitly regularize decision boundaries. Building on this idea, we propose Experience Blending (EB), a framework that jointly trains on exemplars and SBD through a dual-model aggregation strategy. EB has two components: (1) latent-space noise injection to synthesize boundary data, and (2) end-to-end training that jointly leverages exemplars and SBD. Unlike standard experience replay, SBD enriches the feature space near decision boundaries, leading to more stable and robust continual learning. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny ImageNet demonstrate consistent accuracy improvements of 10%, 6%, and 13%, respectively, over strong baselines.
>
---
#### [replaced 023] Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2511.04583v2](http://arxiv.org/pdf/2511.04583v2)**

> **作者:** Atsuyuki Miyai; Mashiro Toyooka; Takashi Otonari; Zaiying Zhao; Kiyoharu Aizawa
>
> **备注:** Issues, comments, and questions are all welcome in https://github.com/Agent4Science-UTokyo/Jr.AI-Scientist
>
> **摘要:** Understanding the current capabilities and risks of AI Scientist systems is essential for ensuring trustworthy and sustainable AI-driven scientific progress while preserving the integrity of the academic ecosystem. To this end, we develop Jr. AI Scientist, a state-of-the-art autonomous AI scientist system that mimics the core research workflow of a novice student researcher: Given the baseline paper from the human mentor, it analyzes its limitations, formulates novel hypotheses for improvement, and iteratively conducts experiments until improvements are realized, and writes a paper with the results. Unlike previous approaches that assume full automation or operate on small-scale code, Jr. AI Scientist follows a well-defined research workflow and leverages modern coding agents to handle complex, multi-file implementations, leading to scientifically valuable contributions. Through our experiments, the Jr. AI Scientist successfully generated new research papers that build upon real NeurIPS, IJCV, and ICLR works by proposing and implementing novel methods. For evaluation, we conducted automated assessments using AI Reviewers, author-led evaluations, and submissions to Agents4Science, a venue dedicated to AI-driven scientific contributions. The findings demonstrate that Jr. AI Scientist generates papers receiving higher review scores than existing fully automated systems. Nevertheless, we identify important limitations from both the author evaluation and the Agents4Science reviews, indicating the potential risks of directly applying current AI Scientist systems and key challenges for future research. Finally, we comprehensively report various risks identified during development. We believe this study clarifies the current role and limitations of AI Scientist systems, offering insights into the areas that still require human expertise and the risks that may emerge as these systems evolve.
>
---
#### [replaced 024] Reg-DPO: SFT-Regularized Direct Preference Optimization with GT-Pair for Improving Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2511.01450v3](http://arxiv.org/pdf/2511.01450v3)**

> **作者:** Jie Du; Xinyu Gong; Qingshan Tan; Wen Li; Yangming Cheng; Weitao Wang; Chenlu Zhan; Suhui Wu; Hao Zhang; Jun Zhang
>
> **备注:** The paper is withdrawn due to the need for further revision and verification of experimental results. A revised version will be resubmitted once the updates are completed
>
> **摘要:** Recent studies have identified Direct Preference Optimization (DPO) as an efficient and reward-free approach to improving video generation quality. However, existing methods largely follow image-domain paradigms and are mainly developed on small-scale models (approximately 2B parameters), limiting their ability to address the unique challenges of video tasks, such as costly data construction, unstable training, and heavy memory consumption. To overcome these limitations, we introduce a GT-Pair that automatically builds high-quality preference pairs by using real videos as positives and model-generated videos as negatives, eliminating the need for any external annotation. We further present Reg-DPO, which incorporates the SFT loss as a regularization term into the DPO loss to enhance training stability and generation fidelity. Additionally, by combining the FSDP framework with multiple memory optimization techniques, our approach achieves nearly three times higher training capacity than using FSDP alone. Extensive experiments on both I2V and T2V tasks across multiple datasets demonstrate that our method consistently outperforms existing approaches, delivering superior video generation quality.
>
---
#### [replaced 025] The Wisdom of a Crowd of Brains: A Universal Brain Encoder
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.12179v3](http://arxiv.org/pdf/2406.12179v3)**

> **作者:** Roman Beliy; Navve Wasserman; Amit Zalcher; Michal Irani
>
> **摘要:** Image-to-fMRI encoding is important for both neuroscience research and practical applications. However, such "Brain-Encoders" have been typically trained per-subject and per fMRI-dataset, thus restricted to very limited training data. In this paper we propose a Universal Brain-Encoder, which can be trained jointly on data from many different subjects/datasets/machines. What makes this possible is our new voxel-centric Encoder architecture, which learns a unique "voxel-embedding" per brain-voxel. Our Encoder trains to predict the response of each brain-voxel on every image, by directly computing the cross-attention between the brain-voxel embedding and multi-level deep image features. This voxel-centric architecture allows the functional role of each brain-voxel to naturally emerge from the voxel-image cross-attention. We show the power of this approach to (i) combine data from multiple different subjects (a "Crowd of Brains") to improve each individual brain-encoding, (ii) quick & effective Transfer-Learning across subjects, datasets, and machines (e.g., 3-Tesla, 7-Tesla), with few training examples, and (iii) use the learned voxel-embeddings as a powerful tool to explore brain functionality (e.g., what is encoded where in the brain).
>
---
#### [replaced 026] LangBridge: Interpreting Image as a Combination of Language Embeddings
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19404v3](http://arxiv.org/pdf/2503.19404v3)**

> **作者:** Jiaqi Liao; Yuwei Niu; Fanqing Meng; Hao Li; Changyao Tian; Yinuo Du; Yuwen Xiong; Dianqi Li; Xizhou Zhu; Li Yuan; Jifeng Dai; Yu Cheng
>
> **备注:** The code and weights are open-sourced. Project page: https://curryx-001.github.io/LangBridge.github.io/
>
> **摘要:** Recent years have witnessed remarkable advances in Large Vision-Language Models (LVLMs), which have achieved human-level performance across various complex vision-language tasks. Following LLaVA's paradigm, mainstream LVLMs typically employ a shallow MLP for visual-language alignment through a two-stage training process: pretraining for cross-modal alignment followed by instruction tuning. While this approach has proven effective, the underlying mechanisms of how MLPs bridge the modality gap remain poorly understood. Although some research has explored how LLMs process transformed visual tokens, few studies have investigated the fundamental alignment mechanism. Furthermore, the MLP adapter requires retraining whenever switching LLM backbones. To address these limitations, we first investigate the working principles of MLP adapters and discover that they learn to project visual embeddings into subspaces spanned by corresponding text embeddings progressively. Based on this insight, we propose LangBridge, a novel adapter that explicitly maps visual tokens to linear combinations of LLM vocabulary embeddings. This innovative design enables pretraining-free adapter transfer across different LLMs while maintaining performance. Our experimental results demonstrate that a LangBridge adapter pre-trained on Qwen2-0.5B can be directly applied to larger models such as LLaMA3-8B or Qwen2.5-14B while maintaining competitive performance. Overall, LangBridge enables interpretable vision-language alignment by grounding visual representations in LLM vocab embedding, while its plug-and-play design ensures efficient reuse across multiple LLMs with nearly no performance degradation. See our project page at https://curryx-001.github.io/LangBridge.github.io/
>
---
#### [replaced 027] LGM-Pose: A Lightweight Global Modeling Network for Real-time Human Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04561v2](http://arxiv.org/pdf/2506.04561v2)**

> **作者:** Biao Guo; Fangmin Guo; Guibo Luo; Xiaonan Luo; Feng Zhang
>
> **摘要:** Most of the current top-down multi-person pose estimation lightweight methods are based on multi-branch parallel pure CNN network architecture, which often struggle to capture the global context required for detecting semantically complex keypoints and are hindered by high latency due to their intricate and redundant structures. In this article, an approximate single-branch lightweight global modeling network (LGM-Pose) is proposed to address these challenges. In the network, a lightweight MobileViM Block is designed with a proposed Lightweight Attentional Representation Module (LARM), which integrates information within and between patches using the Non-Parametric Transformation Operation(NPT-Op) to extract global information. Additionally, a novel Shuffle-Integrated Fusion Module (SFusion) is introduced to effectively integrate multi-scale information, mitigating performance degradation often observed in single-branch structures. Experimental evaluations on the COCO and MPII datasets demonstrate that our approach not only reduces the number of parameters compared to existing mainstream lightweight methods but also achieves superior performance and faster processing speeds.
>
---
#### [replaced 028] Generative neural physics enables quantitative volumetric ultrasound of tissue mechanics
- **分类: cs.CV; 65N21, 92C55, 68T07**

- **链接: [http://arxiv.org/pdf/2508.12226v2](http://arxiv.org/pdf/2508.12226v2)**

> **作者:** Zhijun Zeng; Youjia Zheng; Chang Su; Qianhang Wu; Hao Hu; Zeyuan Dong; Shan Gao; Yang Lv; Rui Tang; Ligang Cui; Zhiyong Hou; Weijun Lin; Zuoqiang Shi; Yubing Li; He Sun
>
> **摘要:** Tissue mechanics--stiffness, density and impedance contrast--are broadly informative biomarkers across diseases, yet routine CT, MRI, and B-mode ultrasound rarely quantify them directly. While ultrasound tomography (UT) is intrinsically suited to in-vivo biomechanical assessment by capturing transmitted and reflected wavefields, efficient and accurate full-wave scattering models remain a bottleneck. Here, we introduce a generative neural physics framework that fuses generative models with physics-informed partial differential equation (PDE) solvers to produce rapid, high-fidelity 3D quantitative imaging of tissue mechanics. A compact neural surrogate for full-wave propagation is trained on limited cross-modality data, preserving physical accuracy while enabling efficient inversion. This enables, for the first time, accurate and efficient quantitative volumetric imaging of in vivo human breast and musculoskeletal tissues in under ten minutes, providing spatial maps of tissue mechanical properties not available from conventional reflection-mode or standard UT reconstructions. The resulting images reveal biomechanical features in bone, muscle, fat, and glandular tissues, maintaining structural resolution comparable to 3T MRI while providing substantially greater sensitivity to disease-related tissue mechanics.
>
---
#### [replaced 029] ColorBench: Can VLMs See and Understand the Colorful World? A Comprehensive Benchmark for Color Perception, Reasoning, and Robustness
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.10514v3](http://arxiv.org/pdf/2504.10514v3)**

> **作者:** Yijun Liang; Ming Li; Chenrui Fan; Ziyue Li; Dang Nguyen; Kwesi Cobbina; Shweta Bhardwaj; Jiuhai Chen; Fuxiao Liu; Tianyi Zhou
>
> **备注:** Accepted by NeurIPS2025. 36 pages, including references and appendix. Code is available at https://github.com/tianyi-lab/ColorBench
>
> **摘要:** Color plays an important role in human perception and usually provides critical clues in visual reasoning. However, it is unclear whether and how vision-language models (VLMs) can perceive, understand, and leverage color as humans. This paper introduces ColorBench, an innovative benchmark meticulously crafted to assess the capabilities of VLMs in color understanding, including color perception, reasoning, and robustness. By curating a suite of diverse test scenarios, with grounding in real applications, ColorBench evaluates how these models perceive colors, infer meanings from color-based cues, and maintain consistent performance under varying color transformations. Through an extensive evaluation of 32 VLMs with varying language models and vision encoders, our paper reveals some undiscovered findings: (i) The scaling law (larger models are better) still holds on ColorBench, while the language model plays a more important role than the vision encoder. (ii) However, the performance gaps across models are relatively small, indicating that color understanding has been largely neglected by existing VLMs. (iii) CoT reasoning improves color understanding accuracies and robustness, though they are vision-centric tasks. (iv) Color clues are indeed leveraged by VLMs on ColorBench but they can also mislead models in some tasks. These findings highlight the critical limitations of current VLMs and underscore the need to enhance color comprehension. Our ColorBenchcan serve as a foundational tool for advancing the study of human-level color understanding of multimodal AI.
>
---
#### [replaced 030] ACT-R: Adaptive Camera Trajectories for Single View 3D Reconstruction
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08239v3](http://arxiv.org/pdf/2505.08239v3)**

> **作者:** Yizhi Wang; Mingrui Zhao; Hao Zhang
>
> **备注:** 3DV 2026, Project Page: https://mingrui-zhao.github.io/ACT-R/
>
> **摘要:** We introduce the simple idea of adaptive view planning to multi-view synthesis, aiming to improve both occlusion revelation and 3D consistency for single-view 3D reconstruction. Instead of producing an unordered set of views independently or simultaneously, we generate a sequence of views, leveraging temporal consistency to enhance 3D coherence. More importantly, our view sequence is not determined by a pre-determined and fixed camera setup. Instead, we compute an adaptive camera trajectory (ACT), forming an orbit, which seeks to maximize the visibility of occluded regions of the 3D object to be reconstructed. Once the best orbit is found, we feed it to a video diffusion model to generate novel views around the orbit, which can then be passed to any multi-view 3D reconstruction model to obtain the final result. Our multi-view synthesis pipeline is quite efficient since it involves no run-time training/optimization, only forward inferences by applying pre-trained models for occlusion analysis and multi-view synthesis. Our method predicts camera trajectories that reveal occlusions effectively and produce consistent novel views, significantly improving 3D reconstruction over SOTA alternatives on the unseen GSO dataset. Project Page: https://mingrui-zhao.github.io/ACT-R/
>
---
#### [replaced 031] Improving Generalization in Deepfake Detection with Face Foundation Models and Metric Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.19730v2](http://arxiv.org/pdf/2508.19730v2)**

> **作者:** Stelios Mylonas; Symeon Papadopoulos
>
> **备注:** The authors did not manage to secure approval by the funder of this research on time
>
> **摘要:** The increasing realism and accessibility of deepfakes have raised critical concerns about media authenticity and information integrity. Despite recent advances, deepfake detection models often struggle to generalize beyond their training distributions, particularly when applied to media content found in the wild. In this work, we present a robust video deepfake detection framework with strong generalization that takes advantage of the rich facial representations learned by face foundation models. Our method is built on top of FSFM, a self-supervised model trained on real face data, and is further fine-tuned using an ensemble of deepfake datasets spanning both face-swapping and face-reenactment manipulations. To enhance discriminative power, we incorporate triplet loss variants during training, guiding the model to produce more separable embeddings between real and fake samples. Additionally, we explore attribution-based supervision schemes, where deepfakes are categorized by manipulation type or source dataset, to assess their impact on generalization. Extensive experiments across diverse evaluation benchmarks demonstrate the effectiveness of our approach, especially in challenging real-world scenarios.
>
---
#### [replaced 032] Survival Modeling from Whole Slide Images via Patch-Level Graph Clustering and Mixture Density Experts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16476v4](http://arxiv.org/pdf/2507.16476v4)**

> **作者:** Ardhendu Sekhar; Vasu Soni; Keshav Aske; Garima Jain; Pranav Jeevan; Amit Sethi
>
> **摘要:** We propose a modular framework for predicting cancer specific survival directly from whole slide pathology images (WSIs). The framework consists of four key stages designed to capture prognostic and morphological heterogeneity. First, a Quantile Based Patch Filtering module selects prognostically informative tissue regions through quantile thresholding. Second, Graph Regularized Patch Clustering models phenotype level variations using a k nearest neighbor graph that enforces spatial and morphological coherence. Third, Hierarchical Feature Aggregation learns both intra and inter cluster dependencies to represent multiscale tumor organization. Finally, an Expert Guided Mixture Density Model estimates complex survival distributions via Gaussian mixtures, enabling fine grained risk prediction. Evaluated on TCGA LUAD, TCGA KIRC, and TCGA BRCA cohorts, our model achieves concordance indices of 0.653 ,0.719 ,and 0.733 respectively, surpassing existing state of the art approaches in survival prediction from WSIs.
>
---
#### [replaced 033] FreeInsert: Disentangled Text-Guided Object Insertion in 3D Gaussian Scene without Spatial Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01322v4](http://arxiv.org/pdf/2505.01322v4)**

> **作者:** Chenxi Li; Weijie Wang; Qiang Li; Bruno Lepri; Nicu Sebe; Weizhi Nie
>
> **备注:** Accepted by ACMMM2025, Our project webpage: https://tjulcx.github.io/FreeInsert/
>
> **摘要:** Text-driven object insertion in 3D scenes is an emerging task that enables intuitive scene editing through natural language. However, existing 2D editing-based methods often rely on spatial priors such as 2D masks or 3D bounding boxes, and they struggle to ensure consistency of the inserted object. These limitations hinder flexibility and scalability in real-world applications. In this paper, we propose FreeInsert, a novel framework that leverages foundation models including MLLMs, LGMs, and diffusion models to disentangle object generation from spatial placement. This enables unsupervised and flexible object insertion in 3D scenes without spatial priors. FreeInsert starts with an MLLM-based parser that extracts structured semantics, including object types, spatial relationships, and attachment regions, from user instructions. These semantics guide both the reconstruction of the inserted object for 3D consistency and the learning of its degrees of freedom. We leverage the spatial reasoning capabilities of MLLMs to initialize object pose and scale. A hierarchical, spatially aware refinement stage further integrates spatial semantics and MLLM-inferred priors to enhance placement. Finally, the appearance of the object is improved using the inserted-object image to enhance visual fidelity. Experimental results demonstrate that FreeInsert achieves semantically coherent, spatially precise, and visually realistic 3D insertions without relying on spatial priors, offering a user-friendly and flexible editing experience.
>
---
#### [replaced 034] ChestGPT: Integrating Large Language Models and Vision Transformers for Disease Detection and Localization in Chest X-Rays
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03739v2](http://arxiv.org/pdf/2507.03739v2)**

> **作者:** Shehroz S. Khan; Petar Przulj; Ahmed Ashraf; Ali Abedi
>
> **备注:** 8 pages, 5 figures, 4 tables
>
> **摘要:** The global demand for radiologists is increasing rapidly due to a growing reliance on medical imaging services, while the supply of radiologists is not keeping pace. Advances in computer vision and image processing technologies present significant potential to address this gap by enhancing radiologists' capabilities and improving diagnostic accuracy. Large language models (LLMs), particularly generative pre-trained transformers (GPTs), have become the primary approach for understanding and generating textual data. In parallel, vision transformers (ViTs) have proven effective at converting visual data into a format that LLMs can process efficiently. In this paper, we present ChestGPT, a deep-learning framework that integrates the EVA ViT with the Llama 2 LLM to classify diseases and localize regions of interest in chest X-ray images. The ViT converts X-ray images into tokens, which are then fed, together with engineered prompts, into the LLM, enabling joint classification and localization of diseases. This approach incorporates transfer learning techniques to enhance both explainability and performance. The proposed method achieved strong global disease classification performance on the VinDr-CXR dataset, with an F1 score of 0.76, and successfully localized pathologies by generating bounding boxes around the regions of interest. We also outline several task-specific prompts, in addition to general-purpose prompts, for scenarios radiologists might encounter. Overall, this framework offers an assistive tool that can lighten radiologists' workload by providing preliminary findings and regions of interest to facilitate their diagnostic process.
>
---
#### [replaced 035] ViSurf: Visual Supervised-and-Reinforcement Fine-Tuning for Large Vision-and-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.10606v2](http://arxiv.org/pdf/2510.10606v2)**

> **作者:** Yuqi Liu; Liangyu Chen; Jiazhen Liu; Mingkang Zhu; Zhisheng Zhong; Bei Yu; Jiaya Jia
>
> **摘要:** Typical post-training paradigms for Large Vision-and-Language Models (LVLMs) include Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR). SFT leverages external guidance to inject new knowledge, whereas RLVR utilizes internal reinforcement to enhance reasoning capabilities and overall performance. However, our analysis reveals that SFT often leads to sub-optimal performance, while RLVR struggles with tasks that exceed the model's internal knowledge base. To address these limitations, we propose ViSurf (\textbf{Vi}sual \textbf{Su}pervised-and-\textbf{R}einforcement \textbf{F}ine-Tuning), a unified post-training paradigm that integrates the strengths of both SFT and RLVR within a single stage. We analyze the derivation of the SFT and RLVR objectives to establish the ViSurf objective, providing a unified perspective on these two paradigms. The core of ViSurf involves injecting ground-truth labels into the RLVR rollouts, thereby providing simultaneous external supervision and internal reinforcement. Furthermore, we introduce three novel reward control strategies to stabilize and optimize the training process. Extensive experiments across several diverse benchmarks demonstrate the effectiveness of ViSurf, outperforming both individual SFT, RLVR, and two-stage SFT \textrightarrow RLVR. In-depth analysis corroborates these findings, validating the derivation and design principles of ViSurf.
>
---
#### [replaced 036] Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21257v2](http://arxiv.org/pdf/2509.21257v2)**

> **作者:** Seyed Amir Kasaei; Mohammad Hossein Rohban
>
> **备注:** Accepted at GenProCC NeurIPS 2025 Workshop
>
> **摘要:** In language and vision-language models, hallucination is broadly understood as content generated from a model's prior knowledge or biases rather than from the given input. While this phenomenon has been studied in those domains, it has not been clearly framed for text-to-image (T2I) generative models. Existing evaluations mainly focus on alignment, checking whether prompt-specified elements appear, but overlook what the model generates beyond the prompt. We argue for defining hallucination in T2I as bias-driven deviations and propose a taxonomy with three categories: attribute, relation, and object hallucinations. This framing introduces an upper bound for evaluation and surfaces hidden biases, providing a foundation for richer assessment of T2I models.
>
---
#### [replaced 037] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.18094v4](http://arxiv.org/pdf/2509.18094v4)**

> **作者:** Ye Liu; Zongyang Ma; Junfu Pu; Zhongang Qi; Yang Wu; Ying Shan; Chang Wen Chen
>
> **备注:** NeurIPS 2025 Camera Ready. Project Page: https://polyu-chenlab.github.io/unipixel/
>
> **摘要:** Recent advances in Large Multi-modal Models (LMMs) have demonstrated their remarkable success as general-purpose multi-modal assistants, with particular focuses on holistic image- and video-language understanding. Conversely, less attention has been given to scaling fine-grained pixel-level understanding capabilities, where the models are expected to realize pixel-level alignment between visual signals and language semantics. Some previous studies have applied LMMs to related tasks such as region-level captioning and referring expression segmentation. However, these models are limited to performing either referring or segmentation tasks independently and fail to integrate these fine-grained perception capabilities into visual reasoning. To bridge this gap, we propose UniPixel, a large multi-modal model capable of flexibly comprehending visual prompt inputs and generating mask-grounded responses. Our model distinguishes itself by seamlessly integrating pixel-level perception with general visual understanding capabilities. Specifically, UniPixel processes visual prompts and generates relevant masks on demand, and performs subsequent reasoning conditioning on these intermediate pointers during inference, thereby enabling fine-grained pixel-level reasoning. The effectiveness of our approach has been verified on 10 benchmarks across a diverse set of tasks, including pixel-level referring/segmentation and object-centric understanding in images/videos. A novel PixelQA task that jointly requires referring, segmentation, and question answering is also designed to verify the flexibility of our method.
>
---
#### [replaced 038] Token Painter: Training-Free Text-Guided Image Inpainting via Mask Autoregressive Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23919v2](http://arxiv.org/pdf/2509.23919v2)**

> **作者:** Longtao Jiang; Jie Huang; Mingfei Han; Lei Chen; Yongqiang Yu; Feng Zhao; Xiaojun Chang; Zhihui Li
>
> **摘要:** Text-guided image inpainting aims to inpaint masked image regions based on a textual prompt while preserving the background. Although diffusion-based methods have become dominant, their property of modeling the entire image in latent space makes it challenging for the results to align well with prompt details and maintain a consistent background. To address these issues, we explore Mask AutoRegressive (MAR) models for this task. MAR naturally supports image inpainting by generating latent tokens corresponding to mask regions, enabling better local controllability without altering the background. However, directly applying MAR to this task makes the inpainting content either ignore the prompts or be disharmonious with the background context. Through analysis of the attention maps from the inpainting images, we identify the impact of background tokens on text tokens during the MAR generation, and leverage this to design \textbf{Token Painter}, a training-free text-guided image inpainting method based on MAR. Our approach introduces two key components: (1) Dual-Stream Encoder Information Fusion (DEIF), which fuses the semantic and context information from text and background in frequency domain to produce novel guidance tokens, allowing MAR to generate text-faithful inpainting content while keeping harmonious with background context. (2) Adaptive Decoder Attention Score Enhancing (ADAE), which adaptively enhances attention scores on guidance tokens and inpainting tokens to further enhance the alignment of prompt details and the content visual quality. Extensive experiments demonstrate that our training-free method outperforms prior state-of-the-art methods across almost all metrics. Codes: https://github.com/longtaojiang/Token-Painter.
>
---
#### [replaced 039] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.08009v2](http://arxiv.org/pdf/2506.08009v2)**

> **作者:** Xun Huang; Zhengqi Li; Guande He; Mingyuan Zhou; Eli Shechtman
>
> **备注:** NeurIPS 2025 spotlight. Project website: http://self-forcing.github.io/
>
> **摘要:** We introduce Self Forcing, a novel training paradigm for autoregressive video diffusion models. It addresses the longstanding issue of exposure bias, where models trained on ground-truth context must generate sequences conditioned on their own imperfect outputs during inference. Unlike prior methods that denoise future frames based on ground-truth context frames, Self Forcing conditions each frame's generation on previously self-generated outputs by performing autoregressive rollout with key-value (KV) caching during training. This strategy enables supervision through a holistic loss at the video level that directly evaluates the quality of the entire generated sequence, rather than relying solely on traditional frame-wise objectives. To ensure training efficiency, we employ a few-step diffusion model along with a stochastic gradient truncation strategy, effectively balancing computational cost and performance. We further introduce a rolling KV cache mechanism that enables efficient autoregressive video extrapolation. Extensive experiments demonstrate that our approach achieves real-time streaming video generation with sub-second latency on a single GPU, while matching or even surpassing the generation quality of significantly slower and non-causal diffusion models. Project website: http://self-forcing.github.io/
>
---
#### [replaced 040] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04090v2](http://arxiv.org/pdf/2508.04090v2)**

> **作者:** Yi-Ting Chen; Ting-Hsuan Liao; Pengsheng Guo; Alexander Schwing; Jia-Bin Huang
>
> **备注:** Accepted to ICCV 2025. Project website: https://consistent3dsr.github.io/
>
> **摘要:** We propose 3D Super Resolution (3DSR), a novel 3D Gaussian-splatting-based super-resolution framework that leverages off-the-shelf diffusion-based 2D super-resolution models. 3DSR encourages 3D consistency across views via the use of an explicit 3D Gaussian-splatting-based scene representation. This makes the proposed 3DSR different from prior work, such as image upsampling or the use of video super-resolution, which either don't consider 3D consistency or aim to incorporate 3D consistency implicitly. Notably, our method enhances visual quality without additional fine-tuning, ensuring spatial coherence within the reconstructed scene. We evaluate 3DSR on MipNeRF360 and LLFF data, demonstrating that it produces high-resolution results that are visually compelling, while maintaining structural consistency in 3D reconstructions.
>
---
#### [replaced 041] DeepEyesV2: Toward Agentic Multimodal Model
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2511.05271v2](http://arxiv.org/pdf/2511.05271v2)**

> **作者:** Jack Hong; Chenxiao Zhao; ChengLin Zhu; Weiheng Lu; Guohai Xu; Xing Yu
>
> **备注:** Homepage: https://visual-agent.github.io/
>
> **摘要:** Agentic multimodal models should not only comprehend text and images, but also actively invoke external tools, such as code execution environments and web search, and integrate these operations into reasoning. In this work, we introduce DeepEyesV2 and explore how to build an agentic multimodal model from the perspectives of data construction, training methods, and model evaluation. We observe that direct reinforcement learning alone fails to induce robust tool-use behavior. This phenomenon motivates a two-stage training pipeline: a cold-start stage to establish tool-use patterns, and reinforcement learning stage to further refine tool invocation. We curate a diverse, moderately challenging training dataset, specifically including examples where tool use is beneficial. We further introduce RealX-Bench, a comprehensive benchmark designed to evaluate real-world multimodal reasoning, which inherently requires the integration of multiple capabilities, including perception, search, and reasoning. We evaluate DeepEyesV2 on RealX-Bench and other representative benchmarks, demonstrating its effectiveness across real-world understanding, mathematical reasoning, and search-intensive tasks. Moreover, DeepEyesV2 exhibits task-adaptive tool invocation, tending to use image operations for perception tasks and numerical computations for reasoning tasks. Reinforcement learning further enables complex tool combinations and allows model to selectively invoke tools based on context. We hope our study can provide guidance for community in developing agentic multimodal models.
>
---
#### [replaced 042] Distilling Diversity and Control in Diffusion Models
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10637v4](http://arxiv.org/pdf/2503.10637v4)**

> **作者:** Rohit Gandikota; David Bau
>
> **备注:** Project Page: https://distillation.baulab.info/
>
> **摘要:** Distilled diffusion models generate images in far fewer timesteps but suffer from reduced sample diversity when generating multiple outputs from the same prompt. To understand this phenomenon, we first investigate whether distillation damages concept representations by examining if the required diversity is properly learned. Surprisingly, distilled models retain the base model's representational structure: control mechanisms like Concept Sliders and LoRAs transfer seamlessly without retraining, and SliderSpace analysis reveals distilled models possess variational directions needed for diversity yet fail to activate them. This redirects our investigation to understanding how the generation dynamics differ between base and distilled models. Using $\hat{\mathbf{x}}_{0}$ trajectory visualization, we discover distilled models commit to their final image structure almost immediately at the first timestep, while base models distribute structural decisions across many steps. To test whether this first-step commitment causes the diversity loss, we introduce diversity distillation, a hybrid approach using the base model for only the first critical timestep before switching to the distilled model. This single intervention restores sample diversity while maintaining computational efficiency. We provide both causal validation and theoretical support showing why the very first timestep concentrates the diversity bottleneck in distilled models. Our code and data are available at https://distillation.baulab.info/
>
---
#### [replaced 043] When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework
- **分类: cs.CV; cs.AI; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2507.13659v2](http://arxiv.org/pdf/2507.13659v2)**

> **作者:** Xiao Wang; Qian Zhu; Shujuan Wu; Bo Jiang; Shiliang Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Recent researchers have proposed using event cameras for person re-identification (ReID) due to their promising performance and better balance in terms of privacy protection, event camera-based person ReID has attracted significant attention. Currently, mainstream event-based person ReID algorithms primarily focus on fusing visible light and event stream, as well as preserving privacy. Although significant progress has been made, these methods are typically trained and evaluated on small-scale or simulated event camera datasets, making it difficult to assess their real identification performance and generalization ability. To address the issue of data scarcity, this paper introduces a large-scale RGB-event based person ReID dataset, called EvReID. The dataset contains 118,988 image pairs and covers 1200 pedestrian identities, with data collected across multiple seasons, scenes, and lighting conditions. We also evaluate 15 state-of-the-art person ReID algorithms, laying a solid foundation for future research in terms of both data and benchmarking. Based on our newly constructed dataset, this paper further proposes a pedestrian attribute-guided contrastive learning framework to enhance feature learning for person re-identification, termed TriPro-ReID. This framework not only effectively explores the visual features from both RGB frames and event streams, but also fully utilizes pedestrian attributes as mid-level semantic features. Extensive experiments on the EvReID dataset and MARS datasets fully validated the effectiveness of our proposed RGB-Event person ReID framework. The benchmark dataset and source code will be released on https://github.com/Event-AHU/Neuromorphic_ReID
>
---
#### [replaced 044] High-Frequency Semantics and Geometric Priors for End-to-End Detection Transformers in Challenging UAV Imagery
- **分类: cs.CV; I.2.10; I.4.8; I.5.1**

- **链接: [http://arxiv.org/pdf/2507.00825v3](http://arxiv.org/pdf/2507.00825v3)**

> **作者:** Hongxing Peng; Lide Chen; Hui Zhu; Yan Chen
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Object detection in Unmanned Aerial Vehicle (UAV) imagery is fundamentally challenged by a prevalence of small, densely packed, and occluded objects within cluttered backgrounds. Conventional detectors struggle with this domain, as they rely on hand-crafted components like pre-defined anchors and heuristic-based Non-Maximum Suppression (NMS), creating a well-known performance bottleneck in dense scenes. Even recent end-to-end frameworks have not been purpose-built to overcome these specific aerial challenges, resulting in a persistent performance gap. To bridge this gap, we introduce HEDS-DETR, a holistically enhanced real-time Detection Transformer tailored for aerial scenes. Our framework features three key innovations. First, we propose a novel High-Frequency Enhanced Semantics Network (HFESNet) backbone, which yields highly discriminative features by preserving critical high-frequency details alongside robust semantic context. Second, our Efficient Small Object Pyramid (ESOP) counteracts information loss by efficiently fusing high-resolution features, significantly boosting small object detection. Finally, we enhance decoder stability and localization precision with two synergistic components: Selective Query Recollection (SQR) and Geometry-Aware Positional Encoding (GAPE), which stabilize optimization and provide explicit spatial priors for dense object arrangements. On the VisDrone dataset, HEDS-DETR achieves a +3.8% AP and +5.1% AP50 gain over its baseline while reducing parameters by 4M and maintaining real-time speeds. This demonstrates a highly competitive accuracy-efficiency balance, especially for detecting dense and small objects in aerial scenes.
>
---
#### [replaced 045] DeepAndes: A Self-Supervised Vision Foundation Model for Multi-Spectral Remote Sensing Imagery of the Andes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20303v2](http://arxiv.org/pdf/2504.20303v2)**

> **作者:** Junlin Guo; James R. Zimmer-Dauphinee; Jordan M. Nieusma; Siqi Lu; Quan Liu; Ruining Deng; Can Cui; Jialin Yue; Yizhe Lin; Tianyuan Yao; Juming Xiong; Junchao Zhu; Chongyu Qu; Yuechen Yang; Mitchell Wilkes; Xiao Wang; Parker VanValkenburgh; Steven A. Wernke; Yuankai Huo
>
> **摘要:** By mapping sites at large scales using remotely sensed data, archaeologists can generate unique insights into long-term demographic trends, inter-regional social networks, and past adaptations to climate change. Remote sensing surveys complement field-based approaches, and their reach can be especially great when combined with deep learning and computer vision techniques. However, conventional supervised deep learning methods face challenges in annotating fine-grained archaeological features at scale. While recent vision foundation models have shown remarkable success in learning large-scale remote sensing data with minimal annotations, most off-the-shelf solutions are designed for RGB images rather than multi-spectral satellite imagery, such as the 8-band data used in our study. In this paper, we introduce DeepAndes, a transformer-based vision foundation model trained on three million multi-spectral satellite images, specifically tailored for Andean archaeology. DeepAndes incorporates a customized DINOv2 self-supervised learning algorithm optimized for 8-band multi-spectral imagery, marking the first foundation model designed explicitly for the Andes region. We evaluate its image understanding performance through imbalanced image classification, image instance retrieval, and pixel-level semantic segmentation tasks. Our experiments show that DeepAndes achieves superior F1 scores, mean average precision, and Dice scores in few-shot learning scenarios, significantly outperforming models trained from scratch or pre-trained on smaller datasets. This underscores the effectiveness of large-scale self-supervised pre-training in archaeological remote sensing. Codes will be available on https://github.com/geopacha/DeepAndes.
>
---
#### [replaced 046] Rethinking Robust Adversarial Concept Erasure in Diffusion Models
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2510.27285v2](http://arxiv.org/pdf/2510.27285v2)**

> **作者:** Qinghong Yin; Yu Tian; Heming Yang; Xiang Chen; Xianlin Zhang; Xueming Li; Yue Zhan
>
> **摘要:** Concept erasure aims to selectively unlearning undesirable content in diffusion models (DMs) to reduce the risk of sensitive content generation. As a novel paradigm in concept erasure, most existing methods employ adversarial training to identify and suppress target concepts, thus reducing the likelihood of sensitive outputs. However, these methods often neglect the specificity of adversarial training in DMs, resulting in only partial mitigation. In this work, we investigate and quantify this specificity from the perspective of concept space, i.e., can adversarial samples truly fit the target concept space? We observe that existing methods neglect the role of conceptual semantics when generating adversarial samples, resulting in ineffective fitting of concept spaces. This oversight leads to the following issues: 1) when there are few adversarial samples, they fail to comprehensively cover the object concept; 2) conversely, they will disrupt other target concept spaces. Motivated by the analysis of these findings, we introduce S-GRACE (Semantics-Guided Robust Adversarial Concept Erasure), which grace leveraging semantic guidance within the concept space to generate adversarial samples and perform erasure training. Experiments conducted with seven state-of-the-art methods and three adversarial prompt generation strategies across various DM unlearning scenarios demonstrate that S-GRACE significantly improves erasure performance 26%, better preserves non-target concepts, and reduces training time by 90%. Our code is available at https://github.com/Qhong-522/S-GRACE.
>
---
#### [replaced 047] SPIRIT: Short-term Prediction of solar IRradIance for zero-shot Transfer learning using Foundation Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.10307v2](http://arxiv.org/pdf/2502.10307v2)**

> **作者:** Aditya Mishra; Ravindra T; Srinivasan Iyengar; Shivkumar Kalyanaraman; Ponnurangam Kumaraguru
>
> **摘要:** Traditional solar forecasting models are based on several years of site-specific historical irradiance data, often spanning five or more years, which are unavailable for newer photovoltaic farms. As renewable energy is highly intermittent, building accurate solar irradiance forecasting systems is essential for efficient grid management and enabling the ongoing proliferation of solar energy, which is crucial to achieve the United Nations' net zero goals. In this work, we propose SPIRIT, a novel approach leveraging foundation models for solar irradiance forecasting, making it applicable to newer solar installations. Our approach outperforms state-of-the-art models in zero-shot transfer learning by about 70%, enabling effective performance at new locations without relying on any historical data. Further improvements in performance are achieved through fine-tuning, as more location-specific data becomes available. These findings are supported by statistical significance, further validating our approach. SPIRIT represents a pivotal step towards rapid, scalable, and adaptable solar forecasting solutions, advancing the integration of renewable energy into global power systems.
>
---
#### [replaced 048] MCE: Towards a General Framework for Handling Missing Modalities under Imbalanced Missing Rates
- **分类: cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2510.10534v2](http://arxiv.org/pdf/2510.10534v2)**

> **作者:** Binyu Zhao; Wei Zhang; Zhaonian Zou
>
> **备注:** This is the accepted version of an article that has been published in \textbf{Pattern Recognition}. The final version is available via the DOI, or for 50 days' free access via this Share Link: https://authors.elsevier.com/a/1m40D77nKsBm- (valid until December 28, 2025)
>
> **摘要:** Multi-modal learning has made significant advances across diverse pattern recognition applications. However, handling missing modalities, especially under imbalanced missing rates, remains a major challenge. This imbalance triggers a vicious cycle: modalities with higher missing rates receive fewer updates, leading to inconsistent learning progress and representational degradation that further diminishes their contribution. Existing methods typically focus on global dataset-level balancing, often overlooking critical sample-level variations in modality utility and the underlying issue of degraded feature quality. We propose Modality Capability Enhancement (MCE) to tackle these limitations. MCE includes two synergistic components: i) Learning Capability Enhancement (LCE), which introduces multi-level factors to dynamically balance modality-specific learning progress, and ii) Representation Capability Enhancement (RCE), which improves feature semantics and robustness through subset prediction and cross-modal completion tasks. Comprehensive evaluations on four multi-modal benchmarks show that MCE consistently outperforms state-of-the-art methods under various missing configurations. The final published version is now available at https://doi.org/10.1016/j.patcog.2025.112591. Our code is available at https://github.com/byzhaoAI/MCE.
>
---
#### [replaced 049] Descriptive Image-Text Matching with Graded Contextual Similarity
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.09997v3](http://arxiv.org/pdf/2505.09997v3)**

> **作者:** Jinhyun Jang; Jiyoung Lee; Kwanghoon Sohn
>
> **备注:** This version is incomplete and requires substantial revisions and extensions. We withdraw the paper and plan to submit a thoroughly revised version as a new submission
>
> **摘要:** Image-text matching aims to build correspondences between visual and textual data by learning their pairwise similarities. Most existing approaches have adopted sparse binary supervision, indicating whether a pair of images and sentences matches or not. However, such sparse supervision covers a limited subset of image-text relationships, neglecting their inherent many-to-many correspondences; an image can be described in numerous texts at different descriptive levels. Moreover, existing approaches overlook the implicit connections from general to specific descriptions, which form the underlying rationale for the many-to-many relationships between vision and language. In this work, we propose descriptive image-text matching, called DITM, to learn the graded contextual similarity between image and text by exploring the descriptive flexibility of language. We formulate the descriptiveness score of each sentence with cumulative term frequency-inverse document frequency (TF-IDF) to balance the pairwise similarity according to the keywords in the sentence. Our method leverages sentence descriptiveness to learn robust image-text matching in two key ways: (1) to refine the false negative labeling, dynamically relaxing the connectivity between positive and negative pairs, and (2) to build more precise matching, aligning a set of relevant sentences in a generic-to-specific order. By moving beyond rigid binary supervision, DITM enhances the discovery of both optimal matches and potential positive pairs. Extensive experiments on MS-COCO, Flickr30K, and CxC datasets demonstrate the effectiveness of our method in representing complex image-text relationships compared to state-of-the-art approaches. In addition, DITM enhances the hierarchical reasoning ability of the model, supported by the extensive analysis on HierarCaps benchmark.
>
---
#### [replaced 050] Consistent Story Generation: Unlocking the Potential of Zigzag Sampling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09612v5](http://arxiv.org/pdf/2506.09612v5)**

> **作者:** Mingxiao Li; Mang Ning; Marie-Francine Moens
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Text-to-image generation models have made significant progress in producing high-quality images from textual descriptions, yet they continue to struggle with maintaining subject consistency across multiple images, a fundamental requirement for visual storytelling. Existing methods attempt to address this by either fine-tuning models on large-scale story visualization datasets, which is resource-intensive, or by using training-free techniques that share information across generations, which still yield limited success. In this paper, we introduce a novel training-free sampling strategy called Zigzag Sampling with Asymmetric Prompts and Visual Sharing to enhance subject consistency in visual story generation. Our approach proposes a zigzag sampling mechanism that alternates between asymmetric prompting to retain subject characteristics, while a visual sharing module transfers visual cues across generated images to %further enforce consistency. Experimental results, based on both quantitative metrics and qualitative evaluations, demonstrate that our method significantly outperforms previous approaches in generating coherent and consistent visual stories. The code is available at https://github.com/Mingxiao-Li/Asymmetry-Zigzag-StoryDiffusion.
>
---
#### [replaced 051] PSDiffusion: Harmonized Multi-Layer Image Generation via Layout and Appearance Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11468v2](http://arxiv.org/pdf/2505.11468v2)**

> **作者:** Dingbang Huang; Wenbo Li; Yifei Zhao; Xinyu Pan; Chun Wang; Yanhong Zeng; Bo Dai
>
> **摘要:** Transparent image layer generation plays a significant role in digital art and design workflows. Existing methods typically decompose transparent layers from a single RGB image using a set of tools or generate multiple transparent layers sequentially. Despite some promising results, these methods often limit their ability to model global layout, physically plausible interactions, and visual effects such as shadows and reflections with high alpha quality due to limited shared global context among layers. To address this issue, we propose PSDiffusion, a unified diffusion framework that leverages image composition priors from pre-trained image diffusion model for simultaneous multi-layer text-to-image generation. Specifically, our method introduces a global layer interaction mechanism to generate layered images collaboratively, ensuring both individual layer quality and coherent spatial and visual relationships across layers. We include extensive experiments on benchmark datasets to demonstrate that PSDiffusion is able to outperform existing methods in generating multi-layer images with plausible structure and enhanced visual fidelity.
>
---
#### [replaced 052] HyCTAS: Multi-Objective Hybrid Convolution-Transformer Architecture Search for Real-Time Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.10413v3](http://arxiv.org/pdf/2403.10413v3)**

> **作者:** Hongyuan Yu; Cheng Wan; Xiyang Dai; Mengchen Liu; Dongdong Chen; Bin Xiao; Yan Huang; Yuan Lu; Liang Wang
>
> **备注:** 24 pages, 5 figures, published at Neurocomputing
>
> **摘要:** Real-time image segmentation demands architectures that preserve fine spatial detail while capturing global context under tight latency and memory budgets. Image segmentation is one of the most fundamental problems in computer vision and has drawn a lot of attention due to its vast applications in image understanding and autonomous driving. However, designing effective and efficient segmentation neural architectures is a labor-intensive process that may require numerous trials by human experts. In this paper, we address the challenge of integrating multi-head self-attention into high-resolution representation CNNs efficiently by leveraging architecture search. Manually replacing convolution layers with multi-head self-attention is non-trivial due to the costly overhead in memory to maintain high resolution. By contrast, we develop a multi-target multi-branch supernet method, which not only fully utilizes the advantages of high-resolution features but also finds the proper location for placing the multi-head self-attention module. Our search algorithm is optimized towards multiple objectives (e.g., latency and mIoU) and is capable of finding architectures on the approximate Pareto front with an arbitrary number of branches in a single search. We further present a series of models via the Hybrid Convolutional-Transformer Architecture Search (HyCTAS) method that searches for the best hybrid combination of lightweight convolution layers and memory-efficient self-attention layers between branches from different resolutions and fuses them at high resolution for both efficiency and effectiveness. On Cityscapes, ADE20K, and COCO, HyCTAS discovers competitive real-time models without ImageNet pretraining, delivering strong accuracy and latency trade-offs. Code and models are available at https://github.com/MarvinYu1995/HyCTAS.
>
---
#### [replaced 053] Evaluating BM3D and NBNet: A Comprehensive Study of Image Denoising Across Multiple Datasets
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.05697v2](http://arxiv.org/pdf/2408.05697v2)**

> **作者:** Ghazal Kaviani; Reza Marzban; Ghassan AlRegib
>
> **摘要:** This paper investigates image denoising, comparing traditional non-learning-based techniques, represented by Block-Matching 3D (BM3D), with modern learning-based methods, exemplified by NBNet. We assess these approaches across diverse datasets, including CURE-OR, CURE-TSR, SSID+, Set-12, and Chest-Xray, each presenting unique noise challenges. Our analysis employs seven Image Quality Assessment (IQA) metrics and examines the impact on object detection performance. We find that while BM3D excels in scenarios like blur challenges, NBNet is more effective in complex noise environments such as under-exposure and over-exposure. The study reveals the strengths and limitations of each method, providing insights into the effectiveness of different denoising strategies in varied real-world applications.
>
---
#### [replaced 054] VPN: Visual Prompt Navigation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.01766v4](http://arxiv.org/pdf/2508.01766v4)**

> **作者:** Shuo Feng; Zihan Wang; Yuchen Li; Rui Kong; Hengyi Cai; Shuaiqiang Wang; Gim Hee Lee; Piji Li; Shuqiang Jiang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** While natural language is commonly used to guide embodied agents, the inherent ambiguity and verbosity of language often hinder the effectiveness of language-guided navigation in complex environments. To this end, we propose Visual Prompt Navigation (VPN), a novel paradigm that guides agents to navigate using only user-provided visual prompts within 2D top-view maps. This visual prompt primarily focuses on marking the visual navigation trajectory on a top-down view of a scene, offering intuitive and spatially grounded guidance without relying on language instructions. It is more friendly for non-expert users and reduces interpretive ambiguity. We build VPN tasks in both discrete and continuous navigation settings, constructing two new datasets, R2R-VP and R2R-CE-VP, by extending existing R2R and R2R-CE episodes with corresponding visual prompts. Furthermore, we introduce VPNet, a dedicated baseline network to handle the VPN tasks, with two data augmentation strategies: view-level augmentation (altering initial headings and prompt orientations) and trajectory-level augmentation (incorporating diverse trajectories from large-scale 3D scenes), to enhance navigation performance. Extensive experiments evaluate how visual prompt forms, top-view map formats, and data augmentation strategies affect the performance of visual prompt navigation. The code is available at https://github.com/farlit/VPN.
>
---
#### [replaced 055] Rethinking Metrics and Diffusion Architecture for 3D Point Cloud Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2511.05308v2](http://arxiv.org/pdf/2511.05308v2)**

> **作者:** Matteo Bastico; David Ryckelynck; Laurent Corté; Yannick Tillier; Etienne Decencière
>
> **备注:** This paper has been accepted at International Conference on 3D Vision (3DV) 2026
>
> **摘要:** As 3D point clouds become a cornerstone of modern technology, the need for sophisticated generative models and reliable evaluation metrics has grown exponentially. In this work, we first expose that some commonly used metrics for evaluating generated point clouds, particularly those based on Chamfer Distance (CD), lack robustness against defects and fail to capture geometric fidelity and local shape consistency when used as quality indicators. We further show that introducing samples alignment prior to distance calculation and replacing CD with Density-Aware Chamfer Distance (DCD) are simple yet essential steps to ensure the consistency and robustness of point cloud generative model evaluation metrics. While existing metrics primarily focus on directly comparing 3D Euclidean coordinates, we present a novel metric, named Surface Normal Concordance (SNC), which approximates surface similarity by comparing estimated point normals. This new metric, when combined with traditional ones, provides a more comprehensive evaluation of the quality of generated samples. Finally, leveraging recent advancements in transformer-based models for point cloud analysis, such as serialized patch attention , we propose a new architecture for generating high-fidelity 3D structures, the Diffusion Point Transformer. We perform extensive experiments and comparisons on the ShapeNet dataset, showing that our model outperforms previous solutions, particularly in terms of quality of generated point clouds, achieving new state-of-the-art. Code available at https://github.com/matteo-bastico/DiffusionPointTransformer.
>
---
#### [replaced 056] InfiniBench: A Benchmark for Large Multi-Modal Models in Long-Form Movies and TV Shows
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.19875v5](http://arxiv.org/pdf/2406.19875v5)**

> **作者:** Kirolos Ataallah; Eslam Abdelrahman; Mahmoud Ahmed; Chenhui Gou; Khushbu Pahwa; Jian Ding; Mohamed Elhoseiny
>
> **备注:** Accepted for oral presentation at the EMNLP 2025 main conference
>
> **摘要:** Understanding long-form videos, such as movies and TV episodes ranging from tens of minutes to two hours, remains a significant challenge for multi-modal models. Existing benchmarks often fail to test the full range of cognitive skills needed to process these temporally rich and narratively complex inputs. Therefore, we introduce InfiniBench, a comprehensive benchmark designed to evaluate the capabilities of models in long video understanding rigorously. InfiniBench offers:(1) Over 1,000 hours of video content, with an average video length of 53 minutes. (2) The largest set of question-answer pairs for long video comprehension, totaling around 87.7 K. (3) Eight diverse skills that span both grounding-based (e.g., scene transitions, character actions) and reasoning-based (e.g., deep context understanding, multi-event linking). (4) Rich annotation formats, including both multiple-choice and open-ended questions. We conducted an in-depth evaluation across both commercial (GPT-4o, Gemini 2.0 Flash) and most recent open-source vision-language models such as Qwen2.5-VL, InternVL3.0). Results reveal that:(1) Models struggle across the board: Even the best model, GPT-4o, achieves only 47.1 % on grounding-based skills, with most models performing near or just above random chance. (2) Strong reliance on world knowledge: Models achieve surprisingly high scores using only metadata (e.g., video titles), highlighting a tendency to rely on pre-trained knowledge rather than actual visual or temporal understanding. (3) Multi-Modal Importance: When provided with full video and subtitle context, however, models show substantial improvements, confirming the critical role of multimodal input in video understanding. InfiniBench is publicly available at https://vision-cair.github.io/Infinibench
>
---
#### [replaced 057] LMSeg: An end-to-end geometric message-passing network on barycentric dual graphs for large-scale landscape mesh segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.04326v3](http://arxiv.org/pdf/2407.04326v3)**

> **作者:** Zexian Huang; Kourosh Khoshelham; Martin Tomko
>
> **摘要:** Semantic segmentation of large-scale 3D landscape meshes is critical for geospatial analysis in complex environments, yet existing approaches face persistent challenges of scalability, end-to-end trainability, and accurate segmentation of small and irregular objects. To address these issues, we introduce the BudjBim Wall (BBW) dataset, a large-scale annotated mesh dataset derived from high-resolution LiDAR scans of the UNESCO World Heritage-listed Budj Bim cultural landscape in Victoria, Australia. The BBW dataset captures historic dry-stone wall structures that are difficult to detect under vegetation occlusion, supporting research in underrepresented cultural heritage contexts. Building on this dataset, we propose LMSeg, a deep graph message-passing network for semantic segmentation of large-scale meshes. LMSeg employs a barycentric dual graph representation of mesh faces and introduces the Geometry Aggregation+ (GA+) module, a learnable softmax-based operator that adaptively combines neighborhood features and captures high-frequency geometric variations. A hierarchical-local dual pooling integrates hierarchical and local geometric aggregation to balance global context with fine-detail preservation. Experiments on three large-scale benchmarks (SUM, H3D, and BBW) show that LMSeg achieves 75.1% mIoU on SUM, 78.4% O.A. on H3D, and 62.4% mIoU on BBW, using only 2.4M lightweight parameters. In particular, LMSeg demonstrates accurate segmentation across both urban and natural scenes-capturing small-object classes such as vehicles and high vegetation in complex city environments, while also reliably detecting dry-stone walls in dense, occluded rural landscapes. Together, the BBW dataset and LMSeg provide a practical and extensible method for advancing 3D mesh segmentation in cultural heritage, environmental monitoring, and urban applications.
>
---
#### [replaced 058] DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.14376v3](http://arxiv.org/pdf/2510.14376v3)**

> **作者:** Dongnam Byun; Jungwon Park; Jungmin Ko; Changin Choi; Wonjong Rhee
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Recent progress in text-to-image (T2I) generative models has led to significant improvements in generating high-quality images aligned with text prompts. However, these models still struggle with prompts involving multiple objects, often resulting in object neglect or object mixing. Through extensive studies, we identify four problematic scenarios, Similar Shapes, Similar Textures, Dissimilar Background Biases, and Many Objects, where inter-object relationships frequently lead to such failures. Motivated by two key observations about CLIP embeddings, we propose DOS (Directional Object Separation), a method that modifies three types of CLIP text embeddings before passing them into text-to-image models. Experimental results show that DOS consistently improves the success rate of multi-object image generation and reduces object mixing. In human evaluations, DOS significantly outperforms four competing methods, receiving 26.24%-43.04% more votes across four benchmarks. These results highlight DOS as a practical and effective solution for improving multi-object image generation.
>
---
#### [replaced 059] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.08915v2](http://arxiv.org/pdf/2504.08915v2)**

> **作者:** Jiahuan Long; Tingsong Jiang; Wen Yao; Yizhe Xiong; Zhengqin Xu; Shuai Jia; Hanqing Liu; Chao Ma
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Vision foundation models (VFMs) have demonstrated remarkable capabilities in learning universal visual representations. However, adapting these models to downstream tasks conventionally requires parameter updates, with even parameter-efficient fine-tuning methods necessitating the modification of thousands to millions of weights. In this paper, we investigate the redundancies in the segment anything model (SAM) and then propose a novel parameter-free fine-tuning method. Unlike traditional fine-tuning methods that adjust parameters, our method emphasizes selecting, reusing, and enhancing pre-trained features, offering a new perspective on fine-tuning foundation models. Specifically, we introduce a channel selection algorithm based on the model's output difference to identify redundant and effective channels. By selectively replacing the redundant channels with more effective ones, we filter out less useful features and reuse more task-irrelevant features to downstream tasks, thereby enhancing the task-specific feature representation. Experiments on both out-of-domain and in-domain datasets demonstrate the efficiency and effectiveness of our method in different vision tasks (e.g., image segmentation, depth estimation and image classification). Notably, our approach can seamlessly integrate with existing fine-tuning strategies (e.g., LoRA, Adapter), further boosting the performance of already fine-tuned models. Moreover, since our channel selection involves only model inference, our method significantly reduces GPU memory overhead.
>
---
#### [replaced 060] Towards Visual Grounding: A Survey
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20206v2](http://arxiv.org/pdf/2412.20206v2)**

> **作者:** Linhui Xiao; Xiaoshan Yang; Xiangyuan Lan; Yaowei Wang; Changsheng Xu
>
> **备注:** Accepted by TPAMI 2025.We keep tracing related works at https://github.com/linhuixiao/Awesome-Visual-Grounding
>
> **摘要:** Visual Grounding, also known as Referring Expression Comprehension and Phrase Grounding, aims to ground the specific region(s) within the image(s) based on the given expression text. This task simulates the common referential relationships between visual and linguistic modalities, enabling machines to develop human-like multimodal comprehension capabilities. Consequently, it has extensive applications in various domains. However, since 2021, visual grounding has witnessed significant advancements, with emerging new concepts such as grounded pre-training, grounding multimodal LLMs, generalized visual grounding, and giga-pixel grounding, which have brought numerous new challenges. In this survey, we first examine the developmental history of visual grounding and provide an overview of essential background knowledge. We systematically track and summarize the advancements, and then meticulously define and organize the various settings to standardize future research and ensure a fair comparison. Additionally, we delve into numerous related datasets and applications, and highlight several advanced topics. Finally, we outline the challenges confronting visual grounding and propose valuable directions for future research, which may serve as inspiration for subsequent researchers. By extracting common technical details, this survey encompasses the representative work in each subtopic over the past decade. To the best of our knowledge, this paper represents the most comprehensive overview currently available in the field of visual grounding. This survey is designed to be suitable for both beginners and experienced researchers, serving as an invaluable resource for understanding key concepts and tracking the latest research developments. We keep tracing related work at https://github.com/linhuixiao/Awesome-Visual-Grounding.
>
---
#### [replaced 061] Distilling 3D distinctive local descriptors for 6D pose estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.15106v3](http://arxiv.org/pdf/2503.15106v3)**

> **作者:** Amir Hamza; Andrea Caraffa; Davide Boscaini; Fabio Poiesi
>
> **备注:** Project Website: https://tev-fbk.github.io/dGeDi/
>
> **摘要:** Three-dimensional local descriptors are crucial for encoding geometric surface properties, making them essential for various point cloud understanding tasks. Among these descriptors, GeDi has demonstrated strong zero-shot 6D pose estimation capabilities but remains computationally impractical for real-world applications due to its expensive inference process. Can we retain GeDi's effectiveness while significantly improving its efficiency? In this paper, we explore this question by introducing a knowledge distillation framework that trains an efficient student model to regress local descriptors from a GeDi teacher. Our key contributions include: an efficient large-scale training procedure that ensures robustness to occlusions and partial observations while operating under compute and storage constraints, and a novel loss formulation that handles weak supervision from non-distinctive teacher descriptors. We validate our approach on five BOP Benchmark datasets and demonstrate a significant reduction in inference time while maintaining competitive performance with existing methods, bringing zero-shot 6D pose estimation closer to real-time feasibility. Project Website: https://tev-fbk.github.io/dGeDi/
>
---
#### [replaced 062] Video CLIP Model for Multi-View Echocardiography Interpretation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.18800v3](http://arxiv.org/pdf/2504.18800v3)**

> **作者:** Ryo Takizawa; Satoshi Kodera; Tempei Kabayama; Ryo Matsuoka; Yuta Ando; Yuto Nakamura; Haruki Settai; Norihiko Takeda
>
> **摘要:** Echocardiography records ultrasound videos of the heart, enabling clinicians to assess cardiac function. Recent advances in large-scale vision-language models (VLMs) have spurred interest in automating echocardiographic interpretation. However, most existing medical VLMs rely on single-frame (image) inputs, which can reduce diagnostic accuracy for conditions identifiable only through cardiac motion. In addition, echocardiographic videos are captured from multiple views, each varying in suitability for detecting specific conditions. Leveraging multiple views may therefore improve diagnostic performance. We developed a video-language model that processes full video sequences from five standard views, trained on 60,747 echocardiographic video-report pairs. We evaluated the gains in retrieval performance from video input and multi-view support, including the contributions of various pretrained models. Code and model weights are available at https://github.com/UTcardiology/video-echo-clip
>
---
#### [replaced 063] OccLE: Label-Efficient 3D Semantic Occupancy Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20617v3](http://arxiv.org/pdf/2505.20617v3)**

> **作者:** Naiyu Fang; Zheyuan Zhou; Fayao Liu; Xulei Yang; Jiacheng Wei; Lemiao Qiu; Guosheng Lin
>
> **摘要:** 3D semantic occupancy prediction offers an intuitive and efficient scene understanding and has attracted significant interest in autonomous driving perception. Existing approaches either rely on full supervision, which demands costly voxel-level annotations, or on self-supervision, which provides limited guidance and yields suboptimal performance. To address these challenges, we propose OccLE, a Label-Efficient 3D Semantic Occupancy Prediction that takes images and LiDAR as inputs and maintains high performance with limited voxel annotations. Our intuition is to decouple the semantic and geometric learning tasks and then fuse the learned feature grids from both tasks for the final semantic occupancy prediction. Therefore, the semantic branch distills 2D foundation model to provide aligned pseudo labels for 2D and 3D semantic learning. The geometric branch integrates image and LiDAR inputs in cross-plane synergy based on their inherency, employing semi-supervision to enhance geometry learning. We fuse semantic-geometric feature grids through Dual Mamba and incorporate a scatter-accumulated projection to supervise unannotated prediction with aligned pseudo labels. Experiments show that OccLE achieves competitive performance with only 10\% of voxel annotations on the SemanticKITTI and Occ3D-nuScenes datasets. The code will be publicly released on https://github.com/NerdFNY/OccLE
>
---
#### [replaced 064] DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.04203v3](http://arxiv.org/pdf/2407.04203v3)**

> **作者:** Renqi Chen; Xinzhe Zheng; Haoyang Su; Kehan Wu
>
> **备注:** Accepted by AAAI-26 Main Technical Track
>
> **摘要:** Accurate segmentation of ultrasound images is essential for reliable medical diagnoses but is challenged by poor image quality and scarce labeled data. Prior approaches have relied on manually designed, complex network architectures to improve multi-scale feature extraction. However, such handcrafted models offer limited gains when prior knowledge is inadequate and are prone to overfitting on small datasets. In this paper, we introduce DeNAS-ViT, a data-efficient NAS-optimized Vision Transformer, the first method to leverage neural architecture search (NAS) for ultrasound image segmentation by automatically optimizing model architecture through token-level search. Specifically, we propose an efficient NAS module that performs multi-scale token search prior to the ViT's attention mechanism, effectively capturing both contextual and local features while minimizing computational costs. Given ultrasound's data scarcity and NAS's inherent data demands, we further develop a NAS-guided semi-supervised learning (SSL) framework. This approach integrates network independence and contrastive learning within a stage-wise optimization strategy, significantly enhancing model robustness under limited-data conditions. Extensive experiments on public datasets demonstrate that DeNAS-ViT achieves state-of-the-art performance, maintaining robustness with minimal labeled data. Moreover, we highlight DeNAS-ViT's generalization potential beyond ultrasound imaging, underscoring its broader applicability.
>
---
#### [replaced 065] Graph Flow Matching: Enhancing Image Generation with Neighbor-Aware Flow Fields
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24434v3](http://arxiv.org/pdf/2505.24434v3)**

> **作者:** Md Shahriar Rahim Siddiqui; Moshe Eliasof; Eldad Haber
>
> **备注:** The 40th Annual AAAI Conference on Artificial Intelligence
>
> **摘要:** Flow matching casts sample generation as learning a continuous-time velocity field that transports noise to data. Existing flow matching networks typically predict each point's velocity independently, considering only its location and time along its flow trajectory, and ignoring neighboring points. However, this pointwise approach may overlook correlations between points along the generation trajectory that could enhance velocity predictions, thereby improving downstream generation quality. To address this, we propose Graph Flow Matching (GFM), a lightweight enhancement that decomposes the learned velocity into a reaction term -- any standard flow matching network -- and a diffusion term that aggregates neighbor information via a graph neural module. This reaction-diffusion formulation retains the scalability of deep flow models while enriching velocity predictions with local context, all at minimal additional computational cost. Operating in the latent space of a pretrained variational autoencoder, GFM consistently improves Fr\'echet Inception Distance (FID) and recall across five image generation benchmarks (LSUN Church, LSUN Bedroom, FFHQ, AFHQ-Cat, and CelebA-HQ at $256\times256$), demonstrating its effectiveness as a modular enhancement to existing flow matching architectures.
>
---
#### [replaced 066] Not Only Consistency: Enhance Test-Time Adaptation with Spatio-temporal Inconsistency for Remote Physiological Measurement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07908v3](http://arxiv.org/pdf/2507.07908v3)**

> **作者:** Xiao Yang; Jiyao Wang; Yuxuan Fan; Can Liu; Houcheng Su; Weichen Guo; Zitong Yu; Dengbo He; Kaishun Wu
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Remote physiological measurement (RPM) has emerged as a promising non-invasive method for monitoring physiological signals using the non-contact device. Although various domain adaptation and generalization methods were proposed to promote the adaptability of deep-based RPM models in unseen deployment environments, considerations in aspects such as privacy concerns and real-time adaptation restrict their application in real-world deployment. Thus, we aim to propose a novel fully Test-Time Adaptation (TTA) strategy tailored for RPM tasks in this work. Specifically, based on prior knowledge in physiology and our observations, we noticed not only there is spatio-temporal consistency in the frequency domain of BVP signals, but also that inconsistency in the time domain was significant. Given this, by leveraging both consistency and inconsistency priors, we introduce an innovative expert knowledge-based self-supervised \textbf{C}onsistency-\textbf{i}n\textbf{C}onsistency-\textbf{i}ntegration (\textbf{CiCi}) framework to enhances model adaptation during inference. Besides, our approach further incorporates a gradient dynamic control mechanism to mitigate potential conflicts between priors, ensuring stable adaptation across instances. Through extensive experiments on five diverse datasets under the TTA protocol, our method consistently outperforms existing techniques, presenting state-of-the-art performance in real-time self-supervised adaptation without accessing source data. The code will be released later.
>
---
#### [replaced 067] DevFD: Developmental Face Forgery Detection by Learning Shared and Orthogonal LoRA Subspaces
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.19230v3](http://arxiv.org/pdf/2509.19230v3)**

> **作者:** Tianshuo Zhang; Li Gao; Siran Peng; Xiangyu Zhu; Zhen Lei
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** The rise of realistic digital face generation and manipulation poses significant social risks. The primary challenge lies in the rapid and diverse evolution of generation techniques, which often outstrip the detection capabilities of existing models. To defend against the ever-evolving new types of forgery, we need to enable our model to quickly adapt to new domains with limited computation and data while avoiding forgetting previously learned forgery types. In this work, we posit that genuine facial samples are abundant and relatively stable in acquisition methods, while forgery faces continuously evolve with the iteration of manipulation techniques. Given the practical infeasibility of exhaustively collecting all forgery variants, we frame face forgery detection as a continual learning problem and allow the model to develop as new forgery types emerge. Specifically, we employ a Developmental Mixture of Experts (MoE) architecture that uses LoRA models as its individual experts. These experts are organized into two groups: a Real-LoRA to learn and refine knowledge of real faces, and multiple Fake-LoRAs to capture incremental information from different forgery types. To prevent catastrophic forgetting, we ensure that the learning direction of Fake-LoRAs is orthogonal to the established subspace. Moreover, we integrate orthogonal gradients into the orthogonal loss of Fake-LoRAs, preventing gradient interference throughout the training process of each task. Experimental results under both the datasets and manipulation types incremental protocols demonstrate the effectiveness of our method.
>
---
#### [replaced 068] StereoDiff: Stereo-Diffusion Synergy for Video Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.20756v3](http://arxiv.org/pdf/2506.20756v3)**

> **作者:** Haodong Li; Chen Wang; Jiahui Lei; Kostas Daniilidis; Lingjie Liu
>
> **备注:** Work done in Nov 2024, during an internship at the University of Pennsylvania. Project page: https://stereodiff.github.io/
>
> **摘要:** Recent video depth estimation methods achieve great performance by following the paradigm of image depth estimation, i.e., typically fine-tuning pre-trained video diffusion models with massive data. However, we argue that video depth estimation is not a naive extension of image depth estimation. The temporal consistency requirements for dynamic and static regions in videos are fundamentally different. Consistent video depth in static regions, typically backgrounds, can be more effectively achieved via stereo matching across all frames, which provides much stronger global 3D cues. While the consistency for dynamic regions still should be learned from large-scale video depth data to ensure smooth transitions, due to the violation of triangulation constraints. Based on these insights, we introduce StereoDiff, a two-stage video depth estimator that synergizes stereo matching for mainly the static areas with video depth diffusion for maintaining consistent depth transitions in dynamic areas. We mathematically demonstrate how stereo matching and video depth diffusion offer complementary strengths through frequency domain analysis, highlighting the effectiveness of their synergy in capturing the advantages of both. Experimental results on zero-shot, real-world, dynamic video depth benchmarks, both indoor and outdoor, demonstrate StereoDiff's SoTA performance, showcasing its superior consistency and accuracy in video depth estimation.
>
---
#### [replaced 069] EraseFlow: Learning Concept Erasure Policies via GFlowNet-Driven Alignment
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2511.00804v3](http://arxiv.org/pdf/2511.00804v3)**

> **作者:** Abhiram Kusumba; Maitreya Patel; Kyle Min; Changhoon Kim; Chitta Baral; Yezhou Yang
>
> **备注:** NeurIPS'25 Spotlight | Project page: https://eraseflow.github.io/
>
> **摘要:** Erasing harmful or proprietary concepts from powerful text to image generators is an emerging safety requirement, yet current "concept erasure" techniques either collapse image quality, rely on brittle adversarial losses, or demand prohibitive retraining cycles. We trace these limitations to a myopic view of the denoising trajectories that govern diffusion based generation. We introduce EraseFlow, the first framework that casts concept unlearning as exploration in the space of denoising paths and optimizes it with GFlowNets equipped with the trajectory balance objective. By sampling entire trajectories rather than single end states, EraseFlow learns a stochastic policy that steers generation away from target concepts while preserving the model's prior. EraseFlow eliminates the need for carefully crafted reward models and by doing this, it generalizes effectively to unseen concepts and avoids hackable rewards while improving the performance. Extensive empirical results demonstrate that EraseFlow outperforms existing baselines and achieves an optimal trade off between performance and prior preservation.
>
---
#### [replaced 070] Bidirectional Image-Event Guided Fusion Framework for Low-Light Image Enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06120v2](http://arxiv.org/pdf/2506.06120v2)**

> **作者:** Zhanwen Liu; Huanna Song; Yang Wang; Nan Yang; Weiping Ding; Yisheng An
>
> **摘要:** Under extreme low-light conditions, frame-based cameras suffer from severe detail loss due to limited dynamic range. Recent studies have introduced event cameras for event-guided low-light image enhancement. However, existing approaches often overlook the flickering artifacts and structural discontinuities caused by dynamic illumination changes and event sparsity. To address these challenges, we propose BiLIE, a Bidirectional image-event guided fusion framework for Low-Light Image Enhancement, which achieves mutual guidance and complementary enhancement between the two modalities. First, to highlight edge details, we develop a Dynamic Adaptive Filtering Enhancement (DAFE) module that performs adaptive high-pass filtering on event representations to suppress flickering artifacts and preserve high-frequency information under varying illumination. Subsequently, we design a Bidirectional Guided Awareness Fusion (BGAF) mechanism, which achieves breakpoint-aware restoration from images to events and structure-aware enhancement from events to images through a two-stage attention mechanism, establishing cross-modal consistency, thereby producing a clear, smooth, and structurally intact fused representation. Moreover, recognizing that existing datasets exhibit insufficient ground-truth fidelity and color accuracy, we construct a high-quality low-light image-event dataset (RELIE) via a reliable ground truth refinement scheme. Extensive experiments demonstrate that our method outperforms existing approaches on both the RELIE and LIE datasets. Notably, on RELIE, BiLIE exceeds the state-of-the-art by 0.81dB in PSNR and shows significant advantages in edge restoration, color fidelity, and noise suppression.
>
---
#### [replaced 071] DA$^{2}$: Depth Anything in Any Direction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.26618v5](http://arxiv.org/pdf/2509.26618v5)**

> **作者:** Haodong Li; Wangguangdong Zheng; Jing He; Yuhao Liu; Xin Lin; Xin Yang; Ying-Cong Chen; Chunchao Guo
>
> **备注:** Work primarily done during an internship at Tencent Hunyuan. Project page: https://depth-any-in-any-dir.github.io/
>
> **摘要:** Panorama has a full FoV (360$^\circ\times$180$^\circ$), offering a more complete visual description than perspective images. Thanks to this characteristic, panoramic depth estimation is gaining increasing traction in 3D vision. However, due to the scarcity of panoramic data, previous methods are often restricted to in-domain settings, leading to poor zero-shot generalization. Furthermore, due to the spherical distortions inherent in panoramas, many approaches rely on perspective splitting (e.g., cubemaps), which leads to suboptimal efficiency. To address these challenges, we propose $\textbf{DA}$$^{\textbf{2}}$: $\textbf{D}$epth $\textbf{A}$nything in $\textbf{A}$ny $\textbf{D}$irection, an accurate, zero-shot generalizable, and fully end-to-end panoramic depth estimator. Specifically, for scaling up panoramic data, we introduce a data curation engine for generating high-quality panoramic depth data from perspective, and create $\sim$543K panoramic RGB-depth pairs, bringing the total to $\sim$607K. To further mitigate the spherical distortions, we present SphereViT, which explicitly leverages spherical coordinates to enforce the spherical geometric consistency in panoramic image features, yielding improved performance. A comprehensive benchmark on multiple datasets clearly demonstrates DA$^{2}$'s SoTA performance, with an average 38% improvement on AbsRel over the strongest zero-shot baseline. Surprisingly, DA$^{2}$ even outperforms prior in-domain methods, highlighting its superior zero-shot generalization. Moreover, as an end-to-end solution, DA$^{2}$ exhibits much higher efficiency over fusion-based approaches. Both the code and the curated panoramic data has be released. Project page: https://depth-any-in-any-dir.github.io/.
>
---
#### [replaced 072] GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2510.14270v3](http://arxiv.org/pdf/2510.14270v3)**

> **作者:** Alexander Valverde; Brian Xu; Yuyin Zhou; Meng Xu; Hongyun Wang
>
> **摘要:** Scene reconstruction has emerged as a central challenge in computer vision, with approaches such as Neural Radiance Fields (NeRF) and Gaussian Splatting achieving remarkable progress. While Gaussian Splatting demonstrates strong performance on large-scale datasets, it often struggles to capture fine details or maintain realism in regions with sparse coverage, largely due to the inherent limitations of sparse 3D training data. In this work, we propose GauSSmart, a hybrid method that effectively bridges 2D foundational models and 3D Gaussian Splatting reconstruction. Our approach integrates established 2D computer vision techniques, including convex filtering and semantic feature supervision from foundational models such as DINO, to enhance Gaussian-based scene reconstruction. By leveraging 2D segmentation priors and high-dimensional feature embeddings, our method guides the densification and refinement of Gaussian splats, improving coverage in underrepresented areas and preserving intricate structural details. We validate our approach across three datasets, where GauSSmart consistently outperforms existing Gaussian Splatting in the majority of evaluated scenes. Our results demonstrate the significant potential of hybrid 2D-3D approaches, highlighting how the thoughtful combination of 2D foundational models with 3D reconstruction pipelines can overcome the limitations inherent in either approach alone.
>
---
#### [replaced 073] Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14520v2](http://arxiv.org/pdf/2502.14520v2)**

> **作者:** Meng Wang; Fan Wu; Ruihui Li; Yunchuan Qin; Zhuo Tang; Kenli Li
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** 3D Semantic Scene Completion (SSC) provides comprehensive scene geometry and semantics for autonomous driving perception, which is crucial for enabling accurate and reliable decision-making. However, existing SSC methods are limited to capturing sparse information from the current frame or naively stacking multi-frame temporal features, thereby failing to acquire effective scene context. These approaches ignore critical motion dynamics and struggle to achieve temporal consistency. To address the above challenges, we propose a novel temporal SSC method FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance. By leveraging optical flow, FlowScene can integrate motion, different viewpoints, occlusions, and other contextual cues, thereby significantly improving the accuracy of 3D scene completion. Specifically, our framework introduces two key components: (1) a Flow-Guided Temporal Aggregation module that aligns and aggregates temporal features using optical flow, capturing motion-aware context and deformable structures; and (2) an Occlusion-Guided Voxel Refinement module that injects occlusion masks and temporally aggregated features into 3D voxel space, adaptively refining voxel representations for explicit geometric modeling. Experimental results demonstrate that FlowScene achieves state-of-the-art performance on the SemanticKITTI and SSCBench-KITTI-360 benchmarks.
>
---
#### [replaced 074] Role Bias in Diffusion Models: Diagnosing and Mitigating through Intermediate Decomposition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10037v3](http://arxiv.org/pdf/2503.10037v3)**

> **作者:** Sina Malakouti; Adriana Kovashka
>
> **摘要:** Text-to-image (T2I) diffusion models exhibit impressive photorealistic image generation capabilities, yet they struggle in compositional image generation. In this work, we introduce RoleBench, a benchmark focused on evaluating compositional generalization in action-based relations (e.g., "mouse chasing cat"). We show that state-of-the-art T2I models and compositional generation methods consistently default to frequent reversed relations (i.e., "cat chasing mouse"), a phenomenon we call role collapse. Related works attribute this to the model's architectural limitation or underrepresentation in the data. Our key insight reveals that while models fail on rare compositions when their inversions are common, they can successfully generate similar intermediate compositions (e.g., "mouse chasing boy"), suggesting that this limitation is also due to the presence of frequent counterparts rather than just the absence of rare compositions. Motivated by this, we hypothesize that directional decomposition can gradually mitigate role collapse. We test this via ReBind, a lightweight framework that teaches role bindings using carefully selected active/passive intermediate compositions. Experiments suggest that intermediate compositions through simple fine-tuning can significantly reduce role collapse, with humans preferring ReBind more than 78% compared to state-of-the-art methods. Our findings highlight the role of distributional asymmetries in compositional failures and offer a simple, effective path for improving generalization.
>
---
#### [replaced 075] Sekai: A Video Dataset towards World Exploration
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.15675v3](http://arxiv.org/pdf/2506.15675v3)**

> **作者:** Zhen Li; Chuanhao Li; Xiaofeng Mao; Shaoheng Lin; Ming Li; Shitian Zhao; Zhaopan Xu; Xinyue Li; Yukang Feng; Jianwen Sun; Zizhen Li; Fanrui Zhang; Jiaxin Ai; Zhixiang Wang; Yuwei Wu; Tong He; Jiangmiao Pang; Yu Qiao; Yunde Jia; Kaipeng Zhang
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Video generation techniques have made remarkable progress, promising to be the foundation of interactive world exploration. However, existing video generation datasets are not well-suited for world exploration training as they suffer from some limitations: limited locations, short duration, static scenes, and a lack of annotations about exploration and the world. In this paper, we introduce Sekai (meaning "world" in Japanese), a high-quality first-person view worldwide video dataset with rich annotations for world exploration. It consists of over 5,000 hours of walking or drone view (FPV and UVA) videos from over 100 countries and regions across 750 cities. We develop an efficient and effective toolbox to collect, pre-process and annotate videos with location, scene, weather, crowd density, captions, and camera trajectories. Comprehensive analyses and experiments demonstrate the dataset's scale, diversity, annotation quality, and effectiveness for training video generation models. We believe Sekai will benefit the area of video generation and world exploration, and motivate valuable applications. The project page is https://lixsp11.github.io/sekai-project/.
>
---
#### [replaced 076] Visual Structures Helps Visual Reasoning: Addressing the Binding Problem in VLMs
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.22146v4](http://arxiv.org/pdf/2506.22146v4)**

> **作者:** Amirmohammad Izadi; Mohammad Ali Banayeeanzade; Fatemeh Askari; Ali Rahimiakbar; Mohammad Mahdi Vahedi; Hosein Hasani; Mahdieh Soleymani Baghshah
>
> **备注:** Accepted to NeurIPS 2025 (Thirty-ninth Conference on Neural Information Processing Systems)
>
> **摘要:** Despite progress in Large Vision-Language Models (LVLMs), their capacity for visual reasoning is often limited by the binding problem: the failure to reliably associate perceptual features with their correct visual referents. This limitation underlies persistent errors in tasks such as counting, visual search, scene description, and spatial relationship understanding. A key factor is that current LVLMs process visual features largely in parallel, lacking mechanisms for spatially grounded, serial attention. This paper introduces Visual Input Structure for Enhanced Reasoning (VISER), a simple, effective method that augments visual inputs with low-level spatial structures and pairs them with a textual prompt that encourages sequential, spatially-aware parsing. We empirically demonstrate substantial performance improvements across core visual reasoning tasks, using only a single-query inference. Specifically, VISER improves GPT-4o performance on visual search, counting, and spatial relationship tasks by 25.0%, 26.8%, and 9.5%, respectively, and reduces edit distance error in scene description by 0.32 on 2D datasets. Furthermore, we find that the visual modification is essential for these gains; purely textual strategies, including Chain-of-Thought prompting, are insufficient and can even degrade performance. VISER underscores the importance of visual input design over purely linguistically based reasoning strategies and suggests that visual structuring is a powerful and general approach for enhancing compositional and spatial reasoning in LVLMs.
>
---
#### [replaced 077] Visual Hand Gesture Recognition with Deep Learning: A Comprehensive Review of Methods, Datasets, Challenges and Future Research Directions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04465v2](http://arxiv.org/pdf/2507.04465v2)**

> **作者:** Konstantinos Foteinos; Jorgen Cani; Manousos Linardakis; Panagiotis Radoglou-Grammatikis; Vasileios Argyriou; Panagiotis Sarigiannidis; Iraklis Varlamis; Georgios Th. Papadopoulos
>
> **摘要:** The rapid evolution of deep learning (DL) models and the ever-increasing size of available datasets have raised the interest of the research community in the always important field of visual hand gesture recognition (VHGR), and delivered a wide range of applications, such as sign language understanding and human-computer interaction using cameras. Despite the large volume of research works in the field, a structured and complete survey on VHGR is still missing, leaving researchers to navigate through hundreds of papers in order to find the right combination of data, model, and approach for each task. The current survey aims to fill this gap by presenting a comprehensive overview of this computer vision field. With a systematic research methodology that identifies the state-of-the-art works and a structured presentation of the various methods, datasets, and evaluation metrics, this review aims to constitute a useful guideline for researchers, helping them to choose the right strategy for handling a VHGR task. Starting with the methodology used to locate the related literature, the survey identifies and organizes the key VHGR approaches in a taxonomy-based format, and presents the various dimensions that affect the final method choice, such as input modality, task type, and application domain. The state-of-the-art techniques are grouped across three primary VHGR tasks: static gesture recognition, isolated dynamic gestures, and continuous gesture recognition. For each task, the architectural trends and learning strategies are listed. To support the experimental evaluation of future methods in the field, the study reviews commonly used datasets and presents the standard performance metrics. Our survey concludes by identifying the major challenges in VHGR, including both general computer vision issues and domain-specific obstacles, and outlines promising directions for future research.
>
---
#### [replaced 078] Forecasting When to Forecast: Accelerating Diffusion Models with Confidence-Gated Taylor
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.02240v3](http://arxiv.org/pdf/2508.02240v3)**

> **作者:** Xiaoliu Guan; Lielin Jiang; Hanqi Chen; Xu Zhang; Jiaxing Yan; Guanzhong Wang; Yi Liu; Zetao Zhang; Yu Wu
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Diffusion Transformers (DiTs) have demonstrated remarkable performance in visual generation tasks. However, their low inference speed limits their deployment in low-resource applications. Recent training-free approaches exploit the redundancy of features across timesteps by caching and reusing past representations to accelerate inference. Building on this idea, TaylorSeer instead uses cached features to predict future ones via Taylor expansion. However, its module-level prediction across all transformer blocks (e.g., attention or feedforward modules) requires storing fine-grained intermediate features, leading to notable memory and computation overhead. Moreover, it adopts a fixed caching schedule without considering the varying accuracy of predictions across timesteps, which can lead to degraded outputs when prediction fails. To address these limitations, we propose a novel approach to better leverage Taylor-based acceleration. First, we shift the Taylor prediction target from the module level to the last block level, significantly reducing the number of cached features. Furthermore, observing strong sequential dependencies among Transformer blocks, we propose to use the error between the Taylor-estimated and actual outputs of the first block as an indicator of prediction reliability. If the error is small, we trust the Taylor prediction for the last block; otherwise, we fall back to full computation, thereby enabling a dynamic caching mechanism. Empirical results show that our method achieves a better balance between speed and quality, achieving a 3.17x acceleration on FLUX, 2.36x on DiT, and 4.14x on Wan Video with negligible quality drop. The Project Page is \href{https://cg-taylor-acce.github.io/CG-Taylor/}{here.}
>
---
#### [replaced 079] Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18584v3](http://arxiv.org/pdf/2505.18584v3)**

> **作者:** Chaofan Gan; Yuanpeng Tu; Xi Chen; Tieyuan Chen; Yuxi Li; Mehrtash Harandi; Weiyao Lin
>
> **备注:** NeurIPS 2025, code: https://github.com/ganchaofan0000/DiTF
>
> **摘要:** Pre-trained stable diffusion models (SD) have shown great advances in visual correspondence. In this paper, we investigate the capabilities of Diffusion Transformers (DiTs) for accurate dense correspondence. Distinct from SD, DiTs exhibit a critical phenomenon in which very few feature activations exhibit significantly larger values than others, known as \textit{massive activations}, leading to uninformative representations and significant performance degradation for DiTs. The massive activations consistently concentrate at very few fixed dimensions across all image patch tokens, holding little local information. We analyze these dimension-concentrated massive activations and uncover that their concentration is inherently linked to the Adaptive Layer Normalization (AdaLN) in DiTs. Building on these findings, we propose the \textbf{Di}ffusion \textbf{T}ransformer \textbf{F}eature (DiTF), a training-free AdaLN-based framework that extracts semantically discriminative features from DiTs. Specifically, DiTF leverages AdaLN to adaptively localize and normalize massive activations through channel-wise modulation. Furthermore, a channel discard strategy is introduced to mitigate the adverse effects of massive activations. Experimental results demonstrate that our DiTF outperforms both DINO and SD-based models and establishes a new state-of-the-art performance for DiTs in different visual correspondence tasks (\eg, with +9.4\% on Spair-71k and +4.4\% on AP-10K-C.S.).
>
---
#### [replaced 080] PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20358v2](http://arxiv.org/pdf/2509.20358v2)**

> **作者:** Chen Wang; Chuhao Chen; Yiming Huang; Zhiyang Dou; Yuan Liu; Jiatao Gu; Lingjie Liu
>
> **备注:** NeurIPS 2025 Camera Ready Version
>
> **摘要:** Existing video generation models excel at producing photo-realistic videos from text or images, but often lack physical plausibility and 3D controllability. To overcome these limitations, we introduce PhysCtrl, a novel framework for physics-grounded image-to-video generation with physical parameters and force control. At its core is a generative physics network that learns the distribution of physical dynamics across four materials (elastic, sand, plasticine, and rigid) via a diffusion model conditioned on physics parameters and applied forces. We represent physical dynamics as 3D point trajectories and train on a large-scale synthetic dataset of 550K animations generated by physics simulators. We enhance the diffusion model with a novel spatiotemporal attention block that emulates particle interactions and incorporates physics-based constraints during training to enforce physical plausibility. Experiments show that PhysCtrl generates realistic, physics-grounded motion trajectories which, when used to drive image-to-video models, yield high-fidelity, controllable videos that outperform existing methods in both visual quality and physical plausibility. Project Page: https://cwchenwang.github.io/physctrl
>
---
#### [replaced 081] Explaining Bayesian Neural Networks
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2108.10346v2](http://arxiv.org/pdf/2108.10346v2)**

> **作者:** Kirill Bykov; Marina M. -C. Höhne; Adelaida Creosteanu; Klaus-Robert Müller; Frederick Klauschen; Shinichi Nakajima; Marius Kloft
>
> **备注:** 25 pages, 8 figures Accepted to Transactions on Machine Learning Research
>
> **摘要:** To advance the transparency of learning machines such as Deep Neural Networks (DNNs), the field of Explainable AI (XAI) was established to provide interpretations of DNNs' predictions. While different explanation techniques exist, a popular approach is given in the form of attribution maps, which illustrate, given a particular data point, the relevant patterns the model has used for making its prediction. Although Bayesian models such as Bayesian Neural Networks (BNNs) have a limited form of transparency built-in through their prior weight distribution, they lack explanations of their predictions for given instances. In this work, we take a step toward combining these two perspectives by examining how local attributions can be extended to BNNs. Within the Bayesian framework, network weights follow a probability distribution; hence, the standard point explanation extends naturally to an explanation distribution. Viewing explanations probabilistically, we aggregate and analyze multiple local attributions drawn from an approximate posterior to explore variability in explanation patterns. The diversity of explanations offers a way to further explore how predictive rationales may vary across posterior samples. Quantitative and qualitative experiments on toy and benchmark data, as well as on a real-world pathology dataset, illustrate that our framework enriches standard explanations with uncertainty information and may support the visualization of explanation stability.
>
---
#### [replaced 082] JailbreakZoo: Survey, Landscapes, and Horizons in Jailbreaking Large Language and Vision-Language Models
- **分类: cs.CL; cs.CR; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.01599v3](http://arxiv.org/pdf/2407.01599v3)**

> **作者:** Haibo Jin; Leyang Hu; Xinnuo Li; Peiyan Zhang; Chonghan Chen; Jun Zhuang; Haohan Wang
>
> **备注:** 45 pages
>
> **摘要:** The rapid evolution of artificial intelligence (AI) through developments in Large Language Models (LLMs) and Vision-Language Models (VLMs) has brought significant advancements across various technological domains. While these models enhance capabilities in natural language processing and visual interactive tasks, their growing adoption raises critical concerns regarding security and ethical alignment. This survey provides an extensive review of the emerging field of jailbreaking--deliberately circumventing the ethical and operational boundaries of LLMs and VLMs--and the consequent development of defense mechanisms. Our study categorizes jailbreaks into seven distinct types and elaborates on defense strategies that address these vulnerabilities. Through this comprehensive examination, we identify research gaps and propose directions for future studies to enhance the security frameworks of LLMs and VLMs. Our findings underscore the necessity for a unified perspective that integrates both jailbreak strategies and defensive solutions to foster a robust, secure, and reliable environment for the next generation of language models. More details can be found on our website: https://chonghan-chen.com/llm-jailbreak-zoo-survey/.
>
---
#### [replaced 083] FastVGGT: Training-Free Acceleration of Visual Geometry Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02560v2](http://arxiv.org/pdf/2509.02560v2)**

> **作者:** You Shen; Zhipeng Zhang; Yansong Qu; Xiawu Zheng; Jiayi Ji; Shengchuan Zhang; Liujuan Cao
>
> **摘要:** Foundation models for 3D vision have recently demonstrated remarkable capabilities in 3D perception. However, scaling these models to long-sequence image inputs remains a significant challenge due to inference-time inefficiency. In this work, we present a detailed analysis of VGGT, a state-of-the-art feed-forward visual geometry model and identify its primary bottleneck. Visualization further reveals a token collapse phenomenon in the attention maps. Motivated by these findings, we explore the potential of token merging in the feed-forward visual geometry model. Owing to the unique architectural and task-specific properties of 3D models, directly applying existing merging techniques proves challenging. To this end, we propose FastVGGT, which, for the first time, leverages token merging in the 3D domain through a training-free mechanism for accelerating VGGT. we devise a unique token partitioning strategy tailored to 3D architectures and tasks, effectively eliminating redundant computation while preserving VGGT's powerful reconstruction capacity. Extensive experiments on multiple 3D geometry benchmarks validate the effectiveness of our approach. Notably, with 1000 input images, FastVGGT achieves a 4x speedup over VGGT while mitigating error accumulation in long-sequence scenarios. These findings underscore the potential of token merging as a principled solution for scalable 3D vision systems. Code is available at: https://mystorm16.github.io/fastvggt/.
>
---
#### [replaced 084] MAROON: A Dataset for the Joint Characterization of Near-Field High-Resolution Radio-Frequency and Optical Depth Imaging Techniques
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.00527v4](http://arxiv.org/pdf/2411.00527v4)**

> **作者:** Vanessa Wirth; Johanna Bräunig; Nikolai Hofmann; Martin Vossiek; Tim Weyrich; Marc Stamminger
>
> **摘要:** Utilizing the complementary strengths of wavelength-specific range or depth sensors is crucial for robust computer-assisted tasks such as autonomous driving. Despite this, there is still little research done at the intersection of optical depth sensors and radars operating close range, where the target is decimeters away from the sensors. Together with a growing interest in high-resolution imaging radars operating in the near field, the question arises how these sensors behave in comparison to their traditional optical counterparts. In this work, we take on the unique challenge of jointly characterizing depth imagers from both, the optical and radio-frequency domain using a multimodal spatial calibration. We collect data from four depth imagers, with three optical sensors of varying operation principle and an imaging radar. We provide a comprehensive evaluation of their depth measurements with respect to distinct object materials, geometries, and object-to-sensor distances. Specifically, we reveal scattering effects of partially transmissive materials and investigate the response of radio-frequency signals. All object measurements will be made public in form of a multimodal dataset, called MAROON.
>
---
#### [replaced 085] LegalEval-Q: A New Benchmark for The Quality Evaluation of LLM-Generated Legal Text
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24826v2](http://arxiv.org/pdf/2505.24826v2)**

> **作者:** Li yunhan; Wu gengshen
>
> **备注:** 10 pages, 11 figures
>
> **摘要:** As large language models (LLMs) are increasingly used in legal applications, current evaluation benchmarks tend to focus mainly on factual accuracy while largely neglecting important linguistic quality aspects such as clarity, coherence, and terminology. To address this gap, we propose three steps: First, we develop a regression model to evaluate the quality of legal texts based on clarity, coherence, and terminology. Second, we create a specialized set of legal questions. Third, we analyze 49 LLMs using this evaluation framework. Our analysis identifies three key findings: First, model quality levels off at 14 billion parameters, with only a marginal improvement of $2.7\%$ noted at 72 billion parameters. Second, engineering choices such as quantization and context length have a negligible impact, as indicated by statistical significance thresholds above 0.016. Third, reasoning models consistently outperform base architectures. A significant outcome of our research is the release of a ranking list and Pareto analysis, which highlight the Qwen3 series as the optimal choice for cost-performance tradeoffs. This work not only establishes standardized evaluation protocols for legal LLMs but also uncovers fundamental limitations in current training data refinement approaches. Code and models are available at: https://github.com/lyxx3rd/LegalEval-Q.
>
---
#### [replaced 086] Predicting Video Slot Attention Queries from Random Slot-Feature Pairs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.01345v4](http://arxiv.org/pdf/2508.01345v4)**

> **作者:** Rongzhen Zhao; Jian Li; Juho Kannala; Joni Pajarinen
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Unsupervised video Object-Centric Learning (OCL) is promising as it enables object-level scene representation and dynamics modeling as we humans do. Mainstream video OCL methods adopt a recurrent architecture: An aggregator aggregates current video frame into object features, termed slots, under some queries; A transitioner transits current slots to queries for the next frame. This is an effective architecture but all existing implementations both (\textit{i1}) neglect to incorporate next frame features, the most informative source for query prediction, and (\textit{i2}) fail to learn transition dynamics, the knowledge essential for query prediction. To address these issues, we propose Random Slot-Feature pair for learning Query prediction (RandSF.Q): (\textit{t1}) We design a new transitioner to incorporate both slots and features, which provides more information for query prediction; (\textit{t2}) We train the transitioner to predict queries from slot-feature pairs randomly sampled from available recurrences, which drives it to learn transition dynamics. Experiments on scene representation demonstrate that our method surpass existing video OCL methods significantly, e.g., up to 10 points on object discovery, setting new state-of-the-art. Such superiority also benefits downstream tasks like dynamics modeling. Our core source code, model checkpoints and training logs are available on https://github.com/Genera1Z/RandSF.Q.
>
---
#### [replaced 087] Intelligent Sampling Consensus for Homography Estimation in Football Videos Using Featureless Unpaired Points
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2310.04912v2](http://arxiv.org/pdf/2310.04912v2)**

> **作者:** George Nousias; Konstantinos Delibasis; Ilias Maglogiannis
>
> **摘要:** Estimating the homography matrix between images captured under radically different camera poses and zoom factors is a complex challenge. Traditional methods rely on the Random Sample Consensus (RANSAC) algorithm, which requires pairs of homologous points, pre-matched based on local image feature vectors. Sampling consensus is a core step in many Artificial Intelligence (AI) algorithms that enable computer systems to recognize patterns in data. In this paper, we propose H-RANSAC, an algorithm for homography estimation that eliminates the need for feature vectors or explicit point pairing, while it optionally supports point labeling into two classes. H-RANSAC introduces a novel geometric (cheiral) criterion that intelligently rejects implausible point configurations at the beginning of each iteration, while leveraging concave quadrilaterals typically discarded by similar algorithms. A post-hoc criterion at the end of each iteration improves accuracy further. Analytical derivations of the expected maximum iterations are provided, considering success probabilities and outlier rates, enabling adaptive performance tuning. The algorithm is validated on a demanding task: estimating homography between video frames of football matches captured by 12 cameras with highly divergent viewpoints. Results show that H-RANSAC significantly outperforms state-of-the-art classical methods, combined with deep learning-based salient point detection, in terms of average reprojection error and success rates. The relevant implementation is available in https://github.com/gnousias/H-RANSAC.
>
---
#### [replaced 088] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.05664v3](http://arxiv.org/pdf/2410.05664v3)**

> **作者:** Saemi Moon; Minjong Lee; Sangdon Park; Dongwoo Kim
>
> **备注:** ICCV 2025
>
> **摘要:** As text-to-image diffusion models gain widespread commercial applications, there are increasing concerns about unethical or harmful use, including the unauthorized generation of copyrighted or sensitive content. Concept unlearning has emerged as a promising solution to these challenges by removing undesired and harmful information from the pre-trained model. However, the previous evaluations primarily focus on whether target concepts are removed while preserving image quality, neglecting the broader impacts such as unintended side effects. In this work, we propose Holistic Unlearning Benchmark (HUB), a comprehensive framework for evaluating unlearning methods across six key dimensions: faithfulness, alignment, pinpoint-ness, multilingual robustness, attack robustness, and efficiency. Our benchmark covers 33 target concepts, including 16,000 prompts per concept, spanning four categories: Celebrity, Style, Intellectual Property, and NSFW. Our investigation reveals that no single method excels across all evaluation criteria. By releasing our evaluation code and dataset, we hope to inspire further research in this area, leading to more reliable and effective unlearning methods.
>
---
#### [replaced 089] Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment
- **分类: cs.CV; cs.MM; eess.IV**

- **链接: [http://arxiv.org/pdf/2412.18933v2](http://arxiv.org/pdf/2412.18933v2)**

> **作者:** Yixiao Li; Xiaoyuan Yang; Weide Liu; Xin Jin; Xu Jia; Yukun Lai; Paul L Rosin; Haotao Liu; Wei Zhou
>
> **备注:** 15 pages, 10 figures, AAAI CONFERENCE ON ARTIFICIAL INTELLIGENCE(AAAI-26)
>
> **摘要:** As super-resolution (SR) techniques introduce unique distortions that fundamentally differ from those caused by traditional degradation processes (e.g., compression), there is an increasing demand for specialized video quality assessment (VQA) methods tailored to SR-generated content. One critical factor affecting perceived quality is temporal inconsistency, which refers to irregularities between consecutive frames. However, existing VQA approaches rarely quantify this phenomenon or explicitly investigate its relationship with human perception. Moreover, SR videos exhibit amplified inconsistency levels as a result of enhancement processes. In this paper, we propose \textit{Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment (TIG-SVQA)} that underscores the critical role of temporal inconsistency in guiding the quality assessment of SR videos. We first design a perception-oriented approach to quantify frame-wise temporal inconsistency. Based on this, we introduce the Inconsistency Highlighted Spatial Module, which localizes inconsistent regions at both coarse and fine scales. Inspired by the human visual system, we further develop an Inconsistency Guided Temporal Module that performs progressive temporal feature aggregation: (1) a consistency-aware fusion stage in which a visual memory capacity block adaptively determines the information load of each temporal segment based on inconsistency levels, and (2) an informative filtering stage for emphasizing quality-related features. Extensive experiments on both single-frame and multi-frame SR video scenarios demonstrate that our method significantly outperforms state-of-the-art VQA approaches. The code is publicly available at https://github.com/Lighting-YXLI/TIG-SVQA-main.
>
---
#### [replaced 090] EVLM: Self-Reflective Multimodal Reasoning for Cross-Dimensional Visual Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.10566v2](http://arxiv.org/pdf/2412.10566v2)**

> **作者:** Umar Khalid; Kashif Munir; Hasan Iqbal; Azib Farooq; Jing Hua; Nazanin Rahnavard; Chen Chen; Victor Zhu; Zhengping Ji
>
> **备注:** Technical Report
>
> **摘要:** Editing complex visual content from ambiguous or partially specified instructions remains a core challenge in vision-language modeling. Existing models can contextualize content but often fail to infer the underlying intent within a reference image or scene, leading to inconsistent or misaligned edits. We introduce the Editing Vision-Language Model (EVLM), a system that interprets ambiguous instructions in conjunction with reference visuals to produce precise, context-aware editing prompts. EVLM's key innovation is a reflective reasoning framework that translates subjective user intent into structured, actionable outputs by aligning with human-rated rationales through Reflection-Aware KL-Divergence Target Optimization (RKTO). By combining Chain-of-Thought (CoT) reasoning with RKTO alignment, EVLM captures fine-grained editing preferences without relying on binary supervision. Trained on a dataset of 30,000 CoT examples with human-annotated rationale quality, EVLM achieves substantial gains in alignment with human intent. Experiments across image, video, 3D, and 4D editing tasks show that EVLM generates coherent and high-quality instructions, providing a scalable foundation for multimodal editing and reasoning.
>
---
#### [replaced 091] Onboard Hyperspectral Super-Resolution with Deep Pushbroom Neural Network
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20765v2](http://arxiv.org/pdf/2507.20765v2)**

> **作者:** Davide Piccinini; Diego Valsesia; Enrico Magli
>
> **摘要:** Hyperspectral imagers on satellites obtain the fine spectral signatures essential for distinguishing one material from another at the expense of limited spatial resolution. Enhancing the latter is thus a desirable preprocessing step in order to further improve the detection capabilities offered by hyperspectral images on downstream tasks. At the same time, there is a growing interest towards deploying inference methods directly onboard of satellites, which calls for lightweight image super-resolution methods that can be run on the payload in real time. In this paper, we present a novel neural network design, called Deep Pushbroom Super-Resolution (DPSR) that matches the pushbroom acquisition of hyperspectral sensors by processing an image line by line in the along-track direction with a causal memory mechanism to exploit previously acquired lines. This design greatly limits memory requirements and computational complexity, achieving onboard real-time performance, i.e., the ability to super-resolve a line in the time it takes to acquire the next one, on low-power hardware. Experiments show that the quality of the super-resolved images is competitive or even outperforms state-of-the-art methods that are significantly more complex.
>
---
#### [replaced 092] Articulate That Object Part (ATOP): 3D Part Articulation via Text and Motion Personalization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07278v3](http://arxiv.org/pdf/2502.07278v3)**

> **作者:** Aditya Vora; Sauradip Nag; Kai Wang; Hao Zhang
>
> **备注:** Technical Report, 16 pages
>
> **摘要:** We present ATOP (Articulate That Object Part), a novel few-shot method based on motion personalization to articulate a static 3D object with respect to a part and its motion as prescribed in a text prompt. Given the scarcity of available datasets with motion attribute annotations, existing methods struggle to generalize well in this task. In our work, the text input allows us to tap into the power of modern-day diffusion models to generate plausible motion samples for the right object category and part. In turn, the input 3D object provides ``image prompting'' to personalize the generated motion to the very input object. Our method starts with a few-shot finetuning to inject articulation awareness to current diffusion models to learn a unique motion identifier associated with the target object part. Our finetuning is applied to a pre-trained diffusion model for controllable multi-view motion generation, trained with a small collection of reference motion frames demonstrating appropriate part motion. The resulting motion model can then be employed to realize plausible motion of the input 3D object from multiple views. At last, we transfer the personalized motion to the 3D space of the object via differentiable rendering to optimize part articulation parameters by a score distillation sampling loss. Experiments on PartNet-Mobility and ACD datasets demonstrate that our method can generate realistic motion samples with higher accuracy, leading to more generalizable 3D motion predictions compared to prior approaches in the few-shot setting.
>
---
#### [replaced 093] Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.18877v2](http://arxiv.org/pdf/2501.18877v2)**

> **作者:** Jaesin Ahn; Heechul Jung
>
> **备注:** NeurIPS 2025 accepted. Official code: https://github.com/amoeba04/des
>
> **摘要:** Diffusion models show remarkable image generation performance following text prompts, but risk generating sexual contents. Existing approaches, such as prompt filtering, concept removal, and even sexual contents mitigation methods, struggle to defend against adversarial attacks while maintaining benign image quality. In this paper, we propose a novel approach called Distorting Embedding Space (DES), a text encoder-based defense mechanism that effectively tackles these issues through innovative embedding space control. DES transforms unsafe embeddings, extracted from a text encoder using unsafe prompts, toward carefully calculated safe embedding regions to prevent unsafe contents generation, while reproducing the original safe embeddings. DES also neutralizes the ``nudity'' embedding, by aligning it with neutral embedding to enhance robustness against adversarial attacks. As a result, extensive experiments on explicit content mitigation and adaptive attack defense show that DES achieves state-of-the-art (SOTA) defense, with attack success rate (ASR) of 9.47% on FLUX.1, a recent popular model, and 0.52% on the widely adopted Stable Diffusion v1.5. These correspond to ASR reductions of 76.5% and 63.9% compared to previous SOTA methods, EraseAnything and AdvUnlearn, respectively. Furthermore, DES maintains benign image quality, achieving Frechet Inception Distance and CLIP score comparable to those of the original FLUX.1 and Stable Diffusion v1.5.
>
---
#### [replaced 094] Collaborating Vision, Depth, and Thermal Signals for Multi-Modal Tracking: Dataset and Algorithm
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.24741v2](http://arxiv.org/pdf/2509.24741v2)**

> **作者:** Xue-Feng Zhu; Tianyang Xu; Yifan Pan; Jinjie Gu; Xi Li; Jiwen Lu; Xiao-Jun Wu; Josef Kittler
>
> **摘要:** Existing multi-modal object tracking approaches primarily focus on dual-modal paradigms, such as RGB-Depth or RGB-Thermal, yet remain challenged in complex scenarios due to limited input modalities. To address this gap, this work introduces a novel multi-modal tracking task that leverages three complementary modalities, including visible RGB, Depth (D), and Thermal Infrared (TIR), aiming to enhance robustness in complex scenarios. To support this task, we construct a new multi-modal tracking dataset, coined RGBDT500, which consists of 500 videos with synchronised frames across the three modalities. Each frame provides spatially aligned RGB, depth, and thermal infrared images with precise object bounding box annotations. Furthermore, we propose a novel multi-modal tracker, dubbed RDTTrack. RDTTrack integrates tri-modal information for robust tracking by leveraging a pretrained RGB-only tracking model and prompt learning techniques. In specific, RDTTrack fuses thermal infrared and depth modalities under a proposed orthogonal projection constraint, then integrates them with RGB signals as prompts for the pre-trained foundation tracking model, effectively harmonising tri-modal complementary cues. The experimental results demonstrate the effectiveness and advantages of the proposed method, showing significant improvements over existing dual-modal approaches in terms of tracking accuracy and robustness in complex scenarios. The dataset and source code are publicly available at https://xuefeng-zhu5.github.io/RGBDT500.
>
---
#### [replaced 095] Grouped Discrete Representation for Object-Centric Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.02299v3](http://arxiv.org/pdf/2411.02299v3)**

> **作者:** Rongzhen Zhao; Vivienne Wang; Juho Kannala; Joni Pajarinen
>
> **备注:** Accepted to ECML-PKDD 2025
>
> **摘要:** Object-Centric Learning (OCL) aims to discover objects in images or videos by reconstructing the input. Representative methods achieve this by reconstructing the input as its Variational Autoencoder (VAE) discrete representations, which suppress (super-)pixel noise and enhance object separability. However, these methods treat features as indivisible units, overlooking their compositional attributes, and discretize features via scalar code indexes, losing attribute-level similarities and differences. We propose Grouped Discrete Representation (GDR) for OCL. For better generalization, features are decomposed into combinatorial attributes by organized channel grouping. For better convergence, features are quantized into discrete representations via tuple code indexes. Experiments demonstrate that GDR consistently improves both mainstream and state-of-the-art OCL methods across various datasets. Visualizations further highlight GDR's superior object separability and interpretability. The source code is available on https://github.com/Genera1Z/GroupedDiscreteRepresentation.
>
---
#### [replaced 096] A Feedback-Control Framework for Efficient Dataset Collection from In-Vehicle Data Streams
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2511.03239v2](http://arxiv.org/pdf/2511.03239v2)**

> **作者:** Philipp Reis; Philipp Rigoll; Christian Steinhauser; Jacob Langner; Eric Sax
>
> **备注:** 7 Pages, Submitted to IEEE Intelligent Vehicles Symposium 2026
>
> **摘要:** Modern AI systems are increasingly constrained not by model capacity but by the quality and diversity of their data. Despite growing emphasis on data-centric AI, most datasets are still gathered in an open-loop manner which accumulates redundant samples without feedback from the current coverage. This results in inefficient storage, costly labeling, and limited generalization. To address this, this paper introduces Feedback Control Data Collection (FCDC), a paradigm that formulates data collection as a closed-loop control problem. FCDC continuously approximates the state of the collected data distribution using an online probabilistic model and adaptively regulates sample retention using based on feedback signals such as likelihood and Mahalanobis distance. Through this feedback mechanism, the system dynamically balances exploration and exploitation, maintains dataset diversity, and prevents redundancy from accumulating over time. In addition to demonstrating the controllability of FCDC on a synthetic dataset that converges toward a uniform distribution under Gaussian input assumption, experiments on real data streams show that FCDC produces more balanced datasets by 25.9% while reducing data storage by 39.8%. These results demonstrate that data collection itself can be actively controlled, transforming collection from a passive pipeline stage into a self-regulating, feedback-driven process at the core of data-centric AI.
>
---
#### [replaced 097] Enhanced Partially Relevant Video Retrieval through Inter- and Intra-Sample Analysis with Coherence Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.19637v3](http://arxiv.org/pdf/2504.19637v3)**

> **作者:** Junlong Ren; Gangjian Zhang; Hao Wang; Yu Hu; Jian Shu; Hui Xiong
>
> **备注:** Upon further consideration, we have concluded that the current version requires revision and may not yet be ready for publication. We plan to conduct additional experiments and make necessary improvements to ensure the paper meets the standards for future submission
>
> **摘要:** Partially Relevant Video Retrieval (PRVR) aims to retrieve the target video that is partially relevant to the text query. The primary challenge in PRVR arises from the semantic asymmetry between textual and visual modalities, as videos often contain substantial content irrelevant to the query. Existing methods coarsely align paired videos and text queries to construct the semantic space, neglecting the critical cross-modal dual nature inherent in this task: inter-sample correlation and intra-sample redundancy. To this end, we propose a novel PRVR framework to systematically exploit these two characteristics. Our framework consists of three core modules. First, the Inter Correlation Enhancement (ICE) module captures inter-sample correlation by identifying semantically similar yet unpaired text queries and video moments, combining them to form pseudo-positive pairs for more robust semantic space construction. Second, the Intra Redundancy Mining (IRM) module mitigates intra-sample redundancy by mining redundant moment features and distinguishing them from query-relevant moments, encouraging the model to learn more discriminative representations. Finally, to reinforce these modules, we introduce the Temporal Coherence Prediction (TCP) module, which enhances temporal structure learning by training the model to predict the original temporal order of randomly shuffled video frames and moments. Extensive experiments demonstrate the superiority of our approach compared to prior methods, achieving state-of-the-art results.
>
---
#### [replaced 098] Multispectral-NeRF:a multispectral modeling approach based on neural radiance fields
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11169v2](http://arxiv.org/pdf/2509.11169v2)**

> **作者:** Hong Zhang; Fei Guo; Zihan Xie; Dizhao Yao
>
> **摘要:** 3D reconstruction technology generates three-dimensional representations of real-world objects, scenes, or environments using sensor data such as 2D images, with extensive applications in robotics, autonomous vehicles, and virtual reality systems. Traditional 3D reconstruction techniques based on 2D images typically relies on RGB spectral information. With advances in sensor technology, additional spectral bands beyond RGB have been increasingly incorporated into 3D reconstruction workflows. Existing methods that integrate these expanded spectral data often suffer from expensive scheme prices, low accuracy and poor geometric features. Three - dimensional reconstruction based on NeRF can effectively address the various issues in current multispectral 3D reconstruction methods, producing high - precision and high - quality reconstruction results. However, currently, NeRF and some improved models such as NeRFacto are trained on three - band data and cannot take into account the multi - band information. To address this problem, we propose Multispectral-NeRF, an enhanced neural architecture derived from NeRF that can effectively integrates multispectral information. Our technical contributions comprise threefold modifications: Expanding hidden layer dimensionality to accommodate 6-band spectral inputs; Redesigning residual functions to optimize spectral discrepancy calculations between reconstructed and reference images; Adapting data compression modules to address the increased bit-depth requirements of multispectral imagery. Experimental results confirm that Multispectral-NeRF successfully processes multi-band spectral features while accurately preserving the original scenes' spectral characteristics.
>
---
#### [replaced 099] X-Diffusion: Generating Detailed 3D MRI Volumes From a Single Image Using Cross-Sectional Diffusion Models
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2404.19604v4](http://arxiv.org/pdf/2404.19604v4)**

> **作者:** Emmanuelle Bourigault; Abdullah Hamdi; Amir Jamaludin
>
> **备注:** accepted at ICCV 2025 GAIA workshop https://era-ai-biomed.github.io/GAIA/ , project website: https://emmanuelleb985.github.io/XDiffusion/
>
> **摘要:** Magnetic Resonance Imaging (MRI) is a crucial diagnostic tool, but high-resolution scans are often slow and expensive due to extensive data acquisition requirements. Traditional MRI reconstruction methods aim to expedite this process by filling in missing frequency components in the K-space, performing 3D-to-3D reconstructions that demand full 3D scans. In contrast, we introduce X-Diffusion, a novel cross-sectional diffusion model that reconstructs detailed 3D MRI volumes from extremely sparse spatial-domain inputs, achieving 2D-to-3D reconstruction from as little as a single 2D MRI slice or few slices. A key aspect of X-Diffusion is that it models MRI data as holistic 3D volumes during the cross-sectional training and inference, unlike previous learning approaches that treat MRI scans as collections of 2D slices in standard planes (coronal, axial, sagittal). We evaluated X-Diffusion on brain tumor MRIs from the BRATS dataset and full-body MRIs from the UK Biobank dataset. Our results demonstrate that X-Diffusion not only surpasses state-of-the-art methods in quantitative accuracy (PSNR) on unseen data but also preserves critical anatomical features such as tumor profiles, spine curvature, and brain volume. Remarkably, the model generalizes beyond the training domain, successfully reconstructing knee MRIs despite being trained exclusively on brain data. Medical expert evaluations further confirm the clinical relevance and fidelity of the generated images. To our knowledge, X-Diffusion is the first method capable of producing detailed 3D MRIs from highly limited 2D input data, potentially accelerating MRI acquisition and reducing associated costs. The code is available on the project website https://emmanuelleb985.github.io/XDiffusion/ .
>
---
#### [replaced 100] STARS: Self-supervised Tuning for 3D Action Recognition in Skeleton Sequences
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.10935v2](http://arxiv.org/pdf/2407.10935v2)**

> **作者:** Soroush Mehraban; Mohammad Javad Rajabi; Andrea Iaboni; Babak Taati
>
> **摘要:** Self-supervised pretraining methods with masked prediction demonstrate remarkable within-dataset performance in skeleton-based action recognition. However, we show that, unlike contrastive learning approaches, they do not produce well-separated clusters. Additionally, these methods struggle with generalization in few-shot settings. To address these issues, we propose Self-supervised Tuning for 3D Action Recognition in Skeleton sequences (STARS). Specifically, STARS first uses a masked prediction stage using an encoder-decoder architecture. It then employs nearest-neighbor contrastive learning to partially tune the weights of the encoder, enhancing the formation of semantic clusters for different actions. By tuning the encoder for a few epochs, and without using hand-crafted data augmentations, STARS achieves state-of-the-art self-supervised results in various benchmarks, including NTU-60, NTU-120, and PKU-MMD. In addition, STARS exhibits significantly better results than masked prediction models in few-shot settings, where the model has not seen the actions throughout pretraining. Project page: https://soroushmehraban.github.io/stars/
>
---
#### [replaced 101] ImitDiff: Transferring Foundation-Model Priors for Distraction Robust Visuomotor Policy
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.09649v2](http://arxiv.org/pdf/2502.09649v2)**

> **作者:** Yuhang Dong; Haizhou Ge; Yupei Zeng; Jiangning Zhang; Beiwen Tian; Hongrui Zhu; Yufei Jia; Ruixiang Wang; Zhucun Xue; Guyue Zhou; Longhua Ma; Guanzhong Tian
>
> **摘要:** Visuomotor imitation learning policies enable robots to efficiently acquire manipulation skills from visual demonstrations. However, as scene complexity and visual distractions increase, policies that perform well in simple settings often experience substantial performance degradation. To address this challenge, we propose ImitDiff, a diffusion-based imitation learning policy guided by fine-grained semantics within a dual-resolution workflow. Leveraging pretrained priors of vision-language foundation models, our method transforms high-level instructions into pixel-level visual semantic masks. These masks guide a dual-resolution perception pipeline that captures both global context (e.g., overall layout) from low-resolution observation and fine-grained local features (e.g., geometric details) from high-resolution observation, enabling the policy to focus on task-relevant regions. Additionally, we introduce a consistency-driven diffusion transformer action head that bridges visual semantic conditions and real-time action generation. Extensive experiments demonstrate that ImitDiff outperforms state-of-the-art vision-language manipulation frameworks, as well as visuomotor imitation learning policies, particularly under increased scene complexity and visual distractions. Notably, ImitDiff exhibits strong generalization in zero-shot settings involving novel objects and visual distractions. Furthermore, our consistency-driven action head achieves an order-of-magnitude improvement in inference speed while maintaining competitive success rates.
>
---
#### [replaced 102] The Evolving Nature of Latent Spaces: From GANs to Diffusion
- **分类: cs.LG; cs.CV; cs.CY**

- **链接: [http://arxiv.org/pdf/2510.17383v2](http://arxiv.org/pdf/2510.17383v2)**

> **作者:** Ludovica Schaerf
>
> **备注:** Presented and published at Ethics and Aesthetics of Artificial Intelligence Conference (EA-AI'25)
>
> **摘要:** This paper examines the evolving nature of internal representations in generative visual models, focusing on the conceptual and technical shift from GANs and VAEs to diffusion-based architectures. Drawing on Beatrice Fazi's account of synthesis as the amalgamation of distributed representations, we propose a distinction between "synthesis in a strict sense", where a compact latent space wholly determines the generative process, and "synthesis in a broad sense," which characterizes models whose representational labor is distributed across layers. Through close readings of model architectures and a targeted experimental setup that intervenes in layerwise representations, we show how diffusion models fragment the burden of representation and thereby challenge assumptions of unified internal space. By situating these findings within media theoretical frameworks and critically engaging with metaphors such as the latent space and the Platonic Representation Hypothesis, we argue for a reorientation of how generative AI is understood: not as a direct synthesis of content, but as an emergent configuration of specialized processes.
>
---
#### [replaced 103] FedHUG: Federated Heterogeneous Unsupervised Generalization for Remote Physiological Measurements
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.12132v2](http://arxiv.org/pdf/2510.12132v2)**

> **作者:** Xiao Yang; Dengbo He; Jiyao Wang; Kaishun Wu
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Remote physiological measurement gained wide attention, while it requires collecting users' privacy-sensitive information, and existing contactless measurements still rely on labeled client data. This presents challenges when we want to further update real-world deployed models with numerous user data lacking labels. To resolve these challenges, we instantiate a new protocol called Federated Unsupervised Domain Generalization (FUDG) in this work. Subsequently, the \textbf{Fed}erated \textbf{H}eterogeneous \textbf{U}nsupervised \textbf{G}eneralization (\textbf{FedHUG}) framework is proposed and consists of: (1) Minimal Bias Aggregation module dynamically adjusts aggregation weights based on prior-driven bias evaluation to cope with heterogeneous non-IID features from multiple domains. (2) The Global Distribution-aware Learning Controller parameterizes the label distribution and dynamically manipulates client-specific training strategies, thereby mitigating the server-client label distribution skew and long-tail issue. The proposal shows superior performance across state-of-the-art techniques in estimation with either RGB video or mmWave radar. The code will be released.
>
---
#### [replaced 104] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21227v2](http://arxiv.org/pdf/2509.21227v2)**

> **作者:** Seyed Amir Kasaei; Ali Aghayari; Arash Marioriyad; Niki Sepasian; MohammadAmin Fazli; Mahdieh Soleymani Baghshah; Mohammad Hossein Rohban
>
> **备注:** Accepted at GenProCC NeurIPS 2025 Workshop
>
> **摘要:** Text-image generation has advanced rapidly, but assessing whether outputs truly capture the objects, attributes, and relations described in prompts remains a central challenge. Evaluation in this space relies heavily on automated metrics, yet these are often adopted by convention or popularity rather than validated against human judgment. Because evaluation and reported progress in the field depend directly on these metrics, it is critical to understand how well they reflect human preferences. To address this, we present a broad study of widely used metrics for compositional text-image evaluation. Our analysis goes beyond simple correlation, examining their behavior across diverse compositional challenges and comparing how different metric families align with human judgments. The results show that no single metric performs consistently across tasks: performance varies with the type of compositional problem. Notably, VQA-based metrics, though popular, are not uniformly superior, while certain embedding-based metrics prove stronger in specific cases. Image-only metrics, as expected, contribute little to compositional evaluation, as they are designed for perceptual quality rather than alignment. These findings underscore the importance of careful and transparent metric selection, both for trustworthy evaluation and for their use as reward models in generation. Project page is available at https://amirkasaei.com/eval-the-evals/ .
>
---
#### [replaced 105] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.01579v2](http://arxiv.org/pdf/2508.01579v2)**

> **作者:** Lingfeng He; De Cheng; Huaijie Wang; Nannan Wang
>
> **备注:** AAAI-2026 Poster
>
> **摘要:** Continual learning (CL) aims to equip models with the ability to learn from a stream of tasks without forgetting previous knowledge. With the progress of vision-language models like Contrastive Language-Image Pre-training (CLIP), their promise for CL has attracted increasing attention due to their strong generalizability. However, the potential of rich textual semantic priors in CLIP in addressing the stability-plasticity dilemma remains underexplored. During backbone training, most approaches transfer past knowledge without considering semantic relevance, leading to interference from unrelated tasks that disrupt the balance between stability and plasticity. Besides, while text-based classifiers provide strong generalization, they suffer from limited plasticity due to the inherent modality gap in CLIP. Visual classifiers help bridge this gap, but their prototypes lack rich and precise semantics. To address these challenges, we propose Semantic-Enriched Continual Adaptation (SECA), a unified framework that harnesses the anti-forgetting and structured nature of textual priors to guide semantic-aware knowledge transfer in the backbone and reinforce the semantic structure of the visual classifier. Specifically, a Semantic-Guided Adaptive Knowledge Transfer (SG-AKT) module is proposed to assess new images' relevance to diverse historical visual knowledge via textual cues, and aggregate relevant knowledge in an instance-adaptive manner as distillation signals. Moreover, a Semantic-Enhanced Visual Prototype Refinement (SE-VPR) module is introduced to refine visual prototypes using inter-class semantic relations captured in class-wise textual embeddings. Extensive experiments on multiple benchmarks validate the effectiveness of our approach.
>
---
#### [replaced 106] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10117v2](http://arxiv.org/pdf/2504.10117v2)**

> **作者:** Peizheng Li; Shuxiao Ding; You Zhou; Qingwen Zhang; Onat Inak; Larissa Triess; Niklas Hanselmann; Marius Cordts; Andreas Zell
>
> **摘要:** Open-world 3D semantic occupancy prediction aims to generate a voxelized 3D representation from sensor inputs while recognizing both known and unknown objects. Transferring open-vocabulary knowledge from vision-language models (VLMs) offers a promising direction but remains challenging. However, methods based on VLM-derived 2D pseudo-labels with traditional supervision are limited by a predefined label space and lack general prediction capabilities. Direct alignment with pretrained image embeddings, on the other hand, often fails to achieve reliable performance because of inconsistent image and text representations in VLMs. To address these challenges, we propose AGO, a novel 3D occupancy prediction framework with adaptive grounding to handle diverse open-world scenarios. AGO first encodes surrounding images and class prompts into 3D and text embeddings, respectively, leveraging similarity-based grounding training with 3D pseudo-labels. Additionally, a modality adapter maps 3D embeddings into a space aligned with VLM-derived image embeddings, reducing modality gaps. Experiments on Occ3D-nuScenes show that AGO improves unknown object prediction in zero-shot and few-shot transfer while achieving state-of-the-art closed-world self-supervised performance, surpassing prior methods by 4.09 mIoU. Code is available at: https://github.com/EdwardLeeLPZ/AGO.
>
---
#### [replaced 107] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Modeling
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.05743v3](http://arxiv.org/pdf/2502.05743v3)**

> **作者:** Xiao Li; Zekai Zhang; Xiang Li; Siyi Chen; Zhihui Zhu; Peng Wang; Qing Qu
>
> **备注:** First two authors contributed equally. Accepted at NeurIPS 2025
>
> **摘要:** Diffusion models, though originally designed for generative tasks, have demonstrated impressive self-supervised representation learning capabilities. A particularly intriguing phenomenon in these models is the emergence of unimodal representation dynamics, where the quality of learned features peaks at an intermediate noise level. In this work, we conduct a comprehensive theoretical and empirical investigation of this phenomenon. Leveraging the inherent low-dimensionality structure of image data, we theoretically demonstrate that the unimodal dynamic emerges when the diffusion model successfully captures the underlying data distribution. The unimodality arises from an interplay between denoising strength and class confidence across noise scales. Empirically, we further show that, in classification tasks, the presence of unimodal dynamics reliably reflects the generalization of the diffusion model: it emerges when the model generates novel images and gradually transitions to a monotonically decreasing curve as the model begins to memorize the training data.
>
---
#### [replaced 108] Temporal Zoom Networks: Distance Regression and Continuous Depth for Efficient Action Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2511.03943v2](http://arxiv.org/pdf/2511.03943v2)**

> **作者:** Ibne Farabi Shihab; Sanjeda Akter; Anuj Sharma
>
> **摘要:** Temporal action localization requires precise boundaries, yet most methods apply uniform computation despite varying boundary difficulty. We propose two complementary contributions. Boundary Distance Regression (BDR) replaces classification with signed-distance regression and zero-crossing extraction. Under idealized assumptions (i.i.d. Laplace noise, uniform stride, sufficient capacity), BDR approaches the Cramer-Rao lower bound, yielding variance on the order of (Delta t)^2 / T (appearing as O((Delta t)^2) for fixed-video inference). The variance ratio R = Var[b_BDR] / Var[b_cls] scales as O((Delta t)^2 / W) for plateau width W approx 2*kappa, with empirical scaling appearing stronger (O((Delta t)^2 / W^2)) due to amplification factors (see Section~4). Empirically, BDR reduces boundary variance by 3.3x to 16.7x (R = 0.06 to 0.30) via four amplification factors. BDR retrofits to existing methods with about 50 lines of code, improving mAP@0.7 by 1.8 to 3.1 percent (average +2.4). Adaptive Temporal Refinement (ATR) learns continuous depth allocation tau in [0,1] to adapt computation, avoiding discrete routing complexity. On THUMOS14, ATR achieves 56.5 percent mAP@0.7 at 151G FLOPs versus 53.6 percent at 198G for the Uniform-6 baseline (24 percent FLOPs reduction, 118 ms vs. 167 ms latency). Gains scale with boundary heterogeneity: THUMOS14 (+2.9), FineAction (+2.7), ActivityNet (+1.8). Training overhead (1.29x baseline) is mitigated via knowledge distillation, with students retaining 99.5 percent performance. Code will be released.
>
---
#### [replaced 109] SkinCaRe: A Multimodal Dermatology Dataset Annotated with Medical Caption and Chain-of-Thought Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.18004v2](http://arxiv.org/pdf/2405.18004v2)**

> **作者:** Yuhao Shen; Liyuan Sun; Yan Xu; Wenbin Liu; Shuping Zhang; Shawn Afvari; Zhongyi Han; Jiaoyan Song; Yongzhi Ji; Tao Lu; Xiaonan He; Xin Gao; Juexiao Zhou
>
> **摘要:** With the widespread application of artificial intelligence (AI), particularly deep learning (DL) and vision large language models (VLLMs), in skin disease diagnosis, the need for interpretability becomes crucial. However, existing dermatology datasets are limited in their inclusion of concept-level meta-labels, and none offer rich medical descriptions in natural language. This deficiency impedes the advancement of LLM-based methods in dermatologic diagnosis. To address this gap and provide a meticulously annotated dermatology dataset with comprehensive natural language descriptions, we introduce \textbf{SkinCaRe}, a comprehensive multimodal resource that unifies \textit{SkinCAP} and \textit{SkinCoT}. \textbf{SkinCAP} comprises 4,000 images sourced from the Fitzpatrick 17k skin disease dataset and the Diverse Dermatology Images dataset, annotated by board-certified dermatologists to provide extensive medical descriptions and captions. In addition, we introduce \textbf{SkinCoT}, a curated dataset pairing 3,041 dermatologic images with clinician-verified, hierarchical chain-of-thought (CoT) diagnoses. Each diagnostic narrative is rigorously evaluated against six quality criteria and iteratively refined until it meets a predefined standard of clinical accuracy and explanatory depth. Together, SkinCAP (captioning) and SkinCoT (reasoning), collectively referred to as SkinCaRe, encompass 7,041 expertly curated dermatologic cases and provide a unified and trustworthy resource for training multimodal models that both describe and explain dermatologic images. SkinCaRe is publicly available at https://huggingface.co/datasets/yuhos16/SkinCaRe.
>
---
#### [replaced 110] HumaniBench: A Human-Centric Framework for Large Multimodal Models Evaluation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11454v5](http://arxiv.org/pdf/2505.11454v5)**

> **作者:** Shaina Raza; Aravind Narayanan; Vahid Reza Khazaie; Ashmal Vayani; Mukund S. Chettiar; Amandeep Singh; Mubarak Shah; Deval Pandya
>
> **摘要:** Large multimodal models (LMMs) have achieved impressive performance on vision-language tasks such as visual question answering (VQA), image captioning, and visual grounding; however, they remain insufficiently evaluated for alignment with human-centered (HC) values such as fairness, ethics, and inclusivity. To address this gap, we introduce HumaniBench, a comprehensive benchmark comprising 32,000 real-world image-question pairs and an accompanying evaluation suite. Using a semi-automated annotation pipeline, each sample is rigorously validated by domain experts to ensure accuracy and ethical integrity. HumaniBench assesses LMMs across seven key alignment principles: fairness, ethics, empathy, inclusivity, reasoning, robustness, and multilinguality through a diverse set of open- and closed-ended VQA tasks. Grounded in AI ethics theory and real-world social contexts, these principles provide a holistic lens for examining human-aligned behavior. Benchmarking results reveal distinct behavioral patterns: certain model families excel in reasoning, fairness, and multilinguality, while others demonstrate greater robustness and grounding capability. However, most models still struggle to balance task accuracy with ethical and inclusive responses. Techniques such as chain-of-thought prompting and test-time scaling yield measurable alignment gains. As the first benchmark explicitly designed for HC evaluation, HumaniBench offers a rigorous testbed to diagnose limitations, quantify alignment trade-offs, and promote the responsible development of large multimodal models. All data and code are publicly released to ensure transparency and reproducibility. https://vectorinstitute.github.io/HumaniBench/
>
---
#### [replaced 111] Incomplete Multi-view Multi-label Classification via a Dual-level Contrastive Learning Framework
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18267v2](http://arxiv.org/pdf/2411.18267v2)**

> **作者:** Bingyan Nie; Wulin Xie; Jiang Long; Xiaohuan Lu
>
> **摘要:** Recently, multi-view and multi-label classification have become significant domains for comprehensive data analysis and exploration. However, incompleteness both in views and labels is still a real-world scenario for multi-view multi-label classification. In this paper, we seek to focus on double missing multi-view multi-label classification tasks and propose our dual-level contrastive learning framework to solve this issue. Different from the existing works, which couple consistent information and view-specific information in the same feature space, we decouple the two heterogeneous properties into different spaces and employ contrastive learning theory to fully disentangle the two properties. Specifically, our method first introduces a two-channel decoupling module that contains a shared representation and a view-proprietary representation to effectively extract consistency and complementarity information across all views. Second, to efficiently filter out high-quality consistent information from multi-view representations, two consistency objectives based on contrastive learning are conducted on the high-level features and the semantic labels, respectively. Extensive experiments on several widely used benchmark datasets demonstrate that the proposed method has more stable and superior classification performance.
>
---
#### [replaced 112] Multi-Scale Fusion for Object Representation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.01539v3](http://arxiv.org/pdf/2410.01539v3)**

> **作者:** Rongzhen Zhao; Vivienne Wang; Juho Kannala; Joni Pajarinen
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** Representing images or videos as object-level feature vectors, rather than pixel-level feature maps, facilitates advanced visual tasks. Object-Centric Learning (OCL) primarily achieves this by reconstructing the input under the guidance of Variational Autoencoder (VAE) intermediate representation to drive so-called \textit{slots} to aggregate as much object information as possible. However, existing VAE guidance does not explicitly address that objects can vary in pixel sizes while models typically excel at specific pattern scales. We propose \textit{Multi-Scale Fusion} (MSF) to enhance VAE guidance for OCL training. To ensure objects of all sizes fall within VAE's comfort zone, we adopt the \textit{image pyramid}, which produces intermediate representations at multiple scales; To foster scale-invariance/variance in object super-pixels, we devise \textit{inter}/\textit{intra-scale fusion}, which augments low-quality object super-pixels of one scale with corresponding high-quality super-pixels from another scale. On standard OCL benchmarks, our technique improves mainstream methods, including state-of-the-art diffusion-based ones. The source code is available on https://github.com/Genera1Z/MultiScaleFusion.
>
---
#### [replaced 113] Evaluating Reasoning Faithfulness in Medical Vision-Language Models using Multimodal Perturbations
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.11196v2](http://arxiv.org/pdf/2510.11196v2)**

> **作者:** Johannes Moll; Markus Graf; Tristan Lemke; Nicolas Lenhart; Daniel Truhn; Jean-Benoit Delbrouck; Jiazhen Pan; Daniel Rueckert; Lisa C. Adams; Keno K. Bressem
>
> **备注:** Accepted to ML4H 2025 Proceedings
>
> **摘要:** Vision-language models (VLMs) often produce chain-of-thought (CoT) explanations that sound plausible yet fail to reflect the underlying decision process, undermining trust in high-stakes clinical use. Existing evaluations rarely catch this misalignment, prioritizing answer accuracy or adherence to formats. We present a clinically grounded framework for chest X-ray visual question answering (VQA) that probes CoT faithfulness via controlled text and image modifications across three axes: clinical fidelity, causal attribution, and confidence calibration. In a reader study (n=4), evaluator-radiologist correlations fall within the observed inter-radiologist range for all axes, with strong alignment for attribution (Kendall's $\tau_b=0.670$), moderate alignment for fidelity ($\tau_b=0.387$), and weak alignment for confidence tone ($\tau_b=0.091$), which we report with caution. Benchmarking six VLMs shows that answer accuracy and explanation quality can be decoupled, acknowledging injected cues does not ensure grounding, and text cues shift explanations more than visual cues. While some open-source models match final answer accuracy, proprietary models score higher on attribution (25.0% vs. 1.4%) and often on fidelity (36.1% vs. 31.7%), highlighting deployment risks and the need to evaluate beyond final answer accuracy.
>
---
#### [replaced 114] F2RVLM: Boosting Fine-grained Fragment Retrieval for Multi-Modal Long-form Dialogue with Vision Language Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.17714v2](http://arxiv.org/pdf/2508.17714v2)**

> **作者:** Hanbo Bi; Zhiqiang Yuan; Zexi Jia; Jiapei Zhang; Chongyang Li; Peixiang Luo; Ying Deng; Xiaoyue Duan; Jinchao Zhang
>
> **摘要:** Traditional dialogue retrieval aims to select the most appropriate utterance or image from recent dialogue history. However, they often fail to meet users' actual needs for revisiting semantically coherent content scattered across long-form conversations. To fill this gap, we define the Fine-grained Fragment Retrieval (FFR) task, requiring models to locate query-relevant fragments, comprising both utterances and images, from multimodal long-form dialogues. As a foundation for FFR, we construct MLDR, the longest-turn multimodal dialogue retrieval dataset to date, averaging 25.45 turns per dialogue, with each naturally spanning three distinct topics. To evaluate generalization in real-world scenarios, we curate and annotate a WeChat-based test set comprising real-world multimodal dialogues with an average of 75.38 turns. Building on these resources, we explore existing generation-based Vision-Language Models (VLMs) on FFR and observe that they often retrieve incoherent utterance-image fragments. While optimized for generating responses from visual-textual inputs, these models lack explicit supervision to ensure semantic coherence within retrieved fragments. To this end, we propose F2RVLM, a generative retrieval model trained in a two-stage paradigm: (1) supervised fine-tuning to inject fragment-level retrieval knowledge, and (2) GRPO-based reinforcement learning with multi-objective rewards promoting semantic precision, relevance, and contextual coherence. To handle varying intra-fragment complexity, from locally dense to sparsely distributed, we introduce difficulty-aware curriculum sampling that ranks training instances by model-predicted difficulty and gradually exposes the model to harder samples. This boosts reasoning ability in long, multi-turn contexts. F2RVLM outperforms popular VLMs in both in-domain and real-domain settings, demonstrating superior retrieval performance.
>
---
#### [replaced 115] Effective Gaussian Management for High-fidelity Object Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.12742v2](http://arxiv.org/pdf/2509.12742v2)**

> **作者:** Jiateng Liu; Hao Gao; Jiu-Cheng Xie; Chi-Man Pun; Jian Xiong; Haolun Li; Junxin Chen; Feng Xu
>
> **摘要:** This paper presents an effective Gaussian management framework for high-fidelity scene reconstruction of appearance and geometry. Departing from recent Gaussian Splatting (GS) methods that rely on indiscriminate attribute assignment, our approach introduces a novel densification strategy called \emph{GauSep} that selectively activates Gaussian color or normal attributes. Together with a tailored rendering pipeline, termed \emph{Separate Rendering}, this strategy alleviates gradient conflicts arising from dual supervision and yields improved reconstruction quality. In addition, we develop \emph{GauRep}, an adaptive and integrated Gaussian representation that reduces redundancy both at the individual and global levels, effectively balancing model capacity and number of parameters. To provide reliable geometric supervision essential for effective management, we also introduce \emph{CoRe}, a novel surface reconstruction module that distills normal fields from the SDF branch to the Gaussian branch through a confidence mechanism. Notably, our management framework is model-agnostic and can be seamlessly incorporated into other architectures, simultaneously improving performance and reducing model size. Extensive experiments demonstrate that our approach achieves superior performance in reconstructing both appearance and geometry compared with state-of-the-art methods, while using significantly fewer parameters.
>
---
#### [replaced 116] Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22067v2](http://arxiv.org/pdf/2510.22067v2)**

> **作者:** Zheng Qi; Chao Shang; Evangelia Spiliopoulou; Nikolaos Pappas
>
> **摘要:** Vision language models (VLMs) often generate hallucination, i.e., content that cannot be substantiated by either textual or visual inputs. Prior work primarily attributes this to over-reliance on linguistic prior knowledge rather than visual inputs. Some methods attempt to mitigate hallucination by amplifying visual token attention proportionally to their attention scores. However, these methods overlook the visual attention sink problem, where attention is frequently misallocated to task-irrelevant visual regions, and neglect cross-modal fusion balance by enhancing only visual attention without adjusting attention to the user query. This can result in amplifying incorrect areas while failing to properly interpret the user query. To address these challenges, we propose a simple yet effective method called Gaze Shift-Guided Cross-modal Fusion Enhancement (GIFT). GIFT pre-computes a holistic visual saliency map by tracking positive changes in visual attention, or "gaze shifts", during user query comprehension, and leverages this map to amplify attention to both salient visual information and the user query at each decoding step. This reduces the impact of visual attention sink, as irrelevant tokens exhibit minimal shifts, while ensuring balanced cross-modal fusion for well-integrated representation. Extensive experiments show that GIFT effectively mitigates hallucination in VLMs across both generative and classification tasks, achieving up to 20.7% improvement over greedy decoding, while maintaining general vision-language performance with low computational overhead.
>
---
#### [replaced 117] Fine-grained Image Retrieval via Dual-Vision Adaptation
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.16273v3](http://arxiv.org/pdf/2506.16273v3)**

> **作者:** Xin Jiang; Meiqi Cao; Hao Tang; Fei Shen; Zechao Li
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Fine-Grained Image Retrieval~(FGIR) faces challenges in learning discriminative visual representations to retrieve images with similar fine-grained features. Current leading FGIR solutions typically follow two regimes: enforce pairwise similarity constraints in the semantic embedding space, or incorporate a localization sub-network to fine-tune the entire model. However, such two regimes tend to overfit the training data while forgetting the knowledge gained from large-scale pre-training, thus reducing their generalization ability. In this paper, we propose a Dual-Vision Adaptation (DVA) approach for FGIR, which guides the frozen pre-trained model to perform FGIR through collaborative sample and feature adaptation. Specifically, we design Object-Perceptual Adaptation, which modifies input samples to help the pre-trained model perceive critical objects and elements within objects that are helpful for category prediction. Meanwhile, we propose In-Context Adaptation, which introduces a small set of parameters for feature adaptation without modifying the pre-trained parameters. This makes the FGIR task using these adjusted features closer to the task solved during the pre-training. Additionally, to balance retrieval efficiency and performance, we propose Discrimination Perception Transfer to transfer the discriminative knowledge in the object-perceptual adaptation to the image encoder using the knowledge distillation mechanism. Extensive experiments show that DVA has fewer learnable parameters and performs well on three in-distribution and three out-of-distribution fine-grained datasets.
>
---
#### [replaced 118] SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.05007v4](http://arxiv.org/pdf/2411.05007v4)**

> **作者:** Muyang Li; Yujun Lin; Zhekai Zhang; Tianle Cai; Xiuyu Li; Junxian Guo; Enze Xie; Chenlin Meng; Jun-Yan Zhu; Song Han
>
> **备注:** ICLR 2025 Spotlight Quantization Library: https://github.com/mit-han-lab/deepcompressor Inference Engine: https://github.com/mit-han-lab/nunchaku Website: https://hanlab.mit.edu/projects/svdquant Demo: https://demo.nunchaku.tech/ Blog: https://hanlab.mit.edu/blog/svdquant
>
> **摘要:** Diffusion models can effectively generate high-quality images. However, as they scale, rising memory demands and higher latency pose substantial deployment challenges. In this work, we aim to accelerate diffusion models by quantizing their weights and activations to 4 bits. At such an aggressive level, both weights and activations are highly sensitive, where existing post-training quantization methods like smoothing become insufficient. To overcome this limitation, we propose SVDQuant, a new 4-bit quantization paradigm. Different from smoothing, which redistributes outliers between weights and activations, our approach absorbs these outliers using a low-rank branch. We first consolidate the outliers by shifting them from activations to weights. Then, we use a high-precision, low-rank branch to take in the weight outliers with Singular Value Decomposition (SVD), while a low-bit quantized branch handles the residuals. This process eases the quantization on both sides. However, naively running the low-rank branch independently incurs significant overhead due to extra data movement of activations, negating the quantization speedup. To address this, we co-design an inference engine Nunchaku that fuses the kernels of the low-rank branch into those of the low-bit branch to cut off redundant memory access. It can also seamlessly support off-the-shelf low-rank adapters (LoRAs) without re-quantization. Extensive experiments on SDXL, PixArt-$\Sigma$, and FLUX.1 validate the effectiveness of SVDQuant in preserving image quality. We reduce the memory usage for the 12B FLUX.1 models by 3.5$\times$, achieving 3.0$\times$ speedup over the 4-bit weight-only quantization (W4A16) baseline on the 16GB laptop 4090 GPU with INT4 precision. On the latest RTX 5090 desktop with Blackwell architecture, we achieve a 3.1$\times$ speedup compared to the W4A16 model using NVFP4 precision.
>
---
#### [replaced 119] TextDiffuser-RL: Efficient and Robust Text Layout Optimization for High-Fidelity Text-to-Image Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19291v3](http://arxiv.org/pdf/2505.19291v3)**

> **作者:** Kazi Mahathir Rahman; Showrin Rahman; Sharmin Sultana Srishty
>
> **备注:** 19 pages, 36 figures
>
> **摘要:** Text-embedded image generation plays a critical role in industries such as graphic design, advertising, and digital content creation. Text-to-Image generation methods leveraging diffusion models, such as TextDiffuser-2, have demonstrated promising results in producing images with embedded text. TextDiffuser-2 effectively generates bounding box layouts that guide the rendering of visual text, achieving high fidelity and coherence. However, existing approaches often rely on resource-intensive processes and are limited in their ability to run efficiently on both CPU and GPU platforms. To address these challenges, we propose a novel two-stage pipeline that integrates reinforcement learning (RL) for rapid and optimized text layout generation with a diffusion-based image synthesis model. Our RL-based approach significantly accelerates the bounding box prediction step while reducing overlaps, allowing the system to run efficiently on both CPUs and GPUs. Extensive evaluations demonstrate that our framework achieves comparable performance to TextDiffuser-2 in terms of text placement and image synthesis, while offering markedly faster runtime and increased flexibility. Our method produces high-quality images comparable to TextDiffuser-2, while being 42.29 times faster and requiring only 2 MB of CPU RAM for inference, unlike TextDiffuser-2's M1 model, which is not executable on CPU-only systems.
>
---
#### [replaced 120] VideoCAD: A Dataset and Model for Learning Long-Horizon 3D CAD UI Interactions from Video
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24838v2](http://arxiv.org/pdf/2505.24838v2)**

> **作者:** Brandon Man; Ghadi Nehme; Md Ferdous Alam; Faez Ahmed
>
> **摘要:** Computer-Aided Design (CAD) is a time-consuming and complex process, requiring precise, long-horizon user interactions with intricate 3D interfaces. While recent advances in AI-driven user interface (UI) agents show promise, most existing datasets and methods focus on short, low-complexity tasks in mobile or web applications, failing to capture the demands of professional engineering tools. In this work, we introduce VideoCAD, the first attempt to model UI interactions for precision engineering tasks. Specifically, VideoCAD is a large-scale synthetic dataset consisting of over 41K annotated video recordings of CAD operations, generated using an automated framework for collecting high-fidelity UI action data from human-made CAD designs. Compared to existing datasets, VideoCAD offers an order-of-magnitude increase in complexity for real-world engineering UI tasks, with time horizons up to 20x longer than those in other datasets. We show two important downstream applications of VideoCAD: (1) learning UI interactions from professional 3D CAD tools for precision tasks and (2) a visual question-answering (VQA) benchmark designed to evaluate multimodal large language models (LLMs) on spatial reasoning and video understanding. To learn the UI interactions, we propose VideoCADFormer, a state-of-the-art model for learning CAD interactions directly from video, which outperforms existing behavior cloning baselines. Both VideoCADFormer and the VQA benchmark derived from VideoCAD reveal key challenges in the current state of video-based UI understanding, including the need for precise action grounding, multi-modal and spatial reasoning, and long-horizon dependencies.
>
---
#### [replaced 121] LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.10040v2](http://arxiv.org/pdf/2501.10040v2)**

> **作者:** Wei Lu; Xue Yang; Si-Bao Chen
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Light-weight neural networks for remote sensing (RS) visual analysis must overcome two inherent redundancies: spatial redundancy from vast, homogeneous backgrounds, and channel redundancy, where extreme scale variations render a single feature space inefficient. Existing models, often designed for natural images, fail to address this dual challenge in RS scenarios. To bridge this gap, we propose LWGANet, a light-weight backbone engineered for RS-specific properties. LWGANet introduces two core innovations: a Top-K Global Feature Interaction (TGFI) module that mitigates spatial redundancy by focusing computation on salient regions, and a Light-Weight Grouped Attention (LWGA) module that resolves channel redundancy by partitioning channels into specialized, scale-specific pathways. By synergistically resolving these core inefficiencies, LWGANet achieves a superior trade-off between feature representation quality and computational cost. Extensive experiments on twelve diverse datasets across four major RS tasks--scene classification, oriented object detection, semantic segmentation, and change detection--demonstrate that LWGANet consistently outperforms state-of-the-art light-weight backbones in both accuracy and efficiency. Our work establishes a new, robust baseline for efficient visual analysis in RS images.
>
---
#### [replaced 122] PersonaAnimator: Personalized Motion Transfer from Unconstrained Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.19895v2](http://arxiv.org/pdf/2508.19895v2)**

> **作者:** Ziyun Qian; Runyu Xiao; Shuyuan Tu; Wei Xue; Dingkang Yang; Mingcheng Li; Dongliang Kou; Minghao Han; Zizhi Chen; Lihua Zhang
>
> **摘要:** Recent advances in motion generation show remarkable progress. However, several limitations remain: (1) Existing pose-guided character motion transfer methods merely replicate motion without learning its style characteristics, resulting in inexpressive characters. (2) Motion style transfer methods rely heavily on motion capture data, which is difficult to obtain. (3) Generated motions sometimes violate physical laws. To address these challenges, this paper pioneers a new task: Video-to-Video Motion Personalization. We propose a novel framework, PersonaAnimator, which learns personalized motion patterns directly from unconstrained videos. This enables personalized motion transfer. To support this task, we introduce PersonaVid, the first video-based personalized motion dataset. It contains 20 motion content categories and 120 motion style categories. We further propose a Physics-aware Motion Style Regularization mechanism to enforce physical plausibility in the generated motions. Extensive experiments show that PersonaAnimator outperforms state-of-the-art motion transfer methods and sets a new benchmark for the Video-to-Video Motion Personalization task.
>
---
#### [replaced 123] MLLM-For3D: Adapting Multimodal Large Language Model for 3D Reasoning Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18135v2](http://arxiv.org/pdf/2503.18135v2)**

> **作者:** Jiaxin Huang; Runnan Chen; Ziwen Li; Zhengqing Gao; Xiao He; Yandong Guo; Mingming Gong; Tongliang Liu
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Reasoning segmentation aims to segment target objects in complex scenes based on human intent and spatial reasoning. While recent multimodal large language models (MLLMs) have demonstrated impressive 2D image reasoning segmentation, adapting these capabilities to 3D scenes remains underexplored. In this paper, we introduce MLLM-For3D, a simple yet effective framework that transfers knowledge from 2D MLLMs to 3D scene understanding. Specifically, we utilize MLLMs to generate multi-view pseudo segmentation masks and corresponding text embeddings, then unproject 2D masks into 3D space and align them with the text embeddings. The primary challenge lies in the absence of 3D context and spatial consistency across multiple views, causing the model to hallucinate objects that do not exist and fail to target objects consistently. Training the 3D model with such irrelevant objects leads to performance degradation. To address this, we introduce a spatial consistency strategy to enforce that segmentation masks remain coherent in the 3D space, effectively capturing the geometry of the scene. Moreover, we develop a Token-for-Query approach for multimodal semantic alignment, enabling consistent identification of the same object across different views. Extensive evaluations on various challenging indoor scene benchmarks demonstrate that, even without any labeled 3D training data, MLLM-For3D outperforms existing 3D reasoning segmentation methods, effectively interpreting user intent, understanding 3D scenes, and reasoning about spatial relationships.
>
---
#### [replaced 124] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2412.20665v2](http://arxiv.org/pdf/2412.20665v2)**

> **作者:** Yuxuan Li; Xiang Li; Yunheng Li; Yicheng Zhang; Yimian Dai; Qibin Hou; Ming-Ming Cheng; Jian Yang
>
> **备注:** Accepted as Oral in AAAI 2026
>
> **摘要:** With the rapid advancement of remote sensing technology, high-resolution multi-modal imagery is now more widely accessible. Conventional Object detection models are trained on a single dataset, often restricted to a specific imaging modality and annotation format. However, such an approach overlooks the valuable shared knowledge across multi-modalities and limits the model's applicability in more versatile scenarios. This paper introduces a new task called Multi-Modal Datasets and Multi-Task Object Detection (M2Det) for remote sensing, designed to accurately detect horizontal or oriented objects from any sensor modality. This task poses challenges due to 1) the trade-offs involved in managing multi-modal modelling and 2) the complexities of multi-task optimization. To address these, we establish a benchmark dataset and propose a unified model, SM3Det (Single Model for Multi-Modal datasets and Multi-Task object Detection). SM3Det leverages a grid-level sparse MoE backbone to enable joint knowledge learning while preserving distinct feature representations for different modalities. Furthermore, it integrates a consistency and synchronization optimization strategy using dynamic learning rate adjustment, allowing it to effectively handle varying levels of learning difficulty across modalities and tasks. Extensive experiments demonstrate SM3Det's effectiveness and generalizability, consistently outperforming specialized models on individual datasets. The code is available at https://github.com/zcablii/SM3Det.
>
---
#### [replaced 125] Slot Attention with Re-Initialization and Self-Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.23755v3](http://arxiv.org/pdf/2507.23755v3)**

> **作者:** Rongzhen Zhao; Yi Zhao; Juho Kannala; Joni Pajarinen
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** Unlike popular solutions based on dense feature maps, Object-Centric Learning (OCL) represents visual scenes as sub-symbolic object-level feature vectors, termed slots, which are highly versatile for tasks involving visual modalities. OCL typically aggregates object superpixels into slots by iteratively applying competitive cross attention, known as Slot Attention, with the slots as the query. However, once initialized, these slots are reused naively, causing redundant slots to compete with informative ones for representing objects. This often results in objects being erroneously segmented into parts. Additionally, mainstream methods derive supervision signals solely from decoding slots into the input's reconstruction, overlooking potential supervision based on internal information. To address these issues, we propose Slot Attention with re-Initialization and self-Distillation (DIAS): $\emph{i)}$ We reduce redundancy in the aggregated slots and re-initialize extra aggregation to update the remaining slots; $\emph{ii)}$ We drive the bad attention map at the first aggregation iteration to approximate the good at the last iteration to enable self-distillation. Experiments demonstrate that DIAS achieves state-of-the-art on OCL tasks like object discovery and recognition, while also improving advanced visual prediction and reasoning. Our source code and model checkpoints are available on https://github.com/Genera1Z/DIAS.
>
---
#### [replaced 126] DCDB: Dynamic Conditional Dual Diffusion Bridge for Ill-posed Multi-Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.03044v2](http://arxiv.org/pdf/2509.03044v2)**

> **作者:** Chengjie Huang; Jiafeng Yan; Jing Li; Lu Bai
>
> **备注:** The article contains factual errors
>
> **摘要:** Conditional diffusion models have made impressive progress in the field of image processing, but the characteristics of constructing data distribution pathways make it difficult to exploit the intrinsic correlation between tasks in multi-task scenarios, which is even worse in ill-posed tasks with a lack of training data. In addition, traditional static condition control makes it difficult for networks to learn in multi-task scenarios with its dynamically evolving characteristics. To address these challenges, we propose a dynamic conditional double diffusion bridge training paradigm to build a general framework for ill-posed multi-tasks. Firstly, this paradigm decouples the diffusion and condition generation processes, avoiding the dependence of the diffusion model on supervised data in ill-posed tasks. Secondly, generated by the same noise schedule, dynamic conditions are used to gradually adjust their statistical characteristics, naturally embed time-related information, and reduce the difficulty of network learning. We analyze the learning objectives of the network under different conditional forms in the single-step denoising process and compare the changes in its attention weights in the network, demonstrating the superiority of our dynamic conditions. Taking dehazing and visible-infrared fusion as typical ill-posed multi-task scenarios, we achieve the best performance in multiple indicators on public datasets. The code has been publicly released at: https://anonymous.4open.science/r/DCDB-D3C2.
>
---
#### [replaced 127] Stack Transformer Based Spatial-Temporal Attention Model for Dynamic Sign Language and Fingerspelling Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16855v2](http://arxiv.org/pdf/2503.16855v2)**

> **作者:** Koki Hirooka; Abu Saleh Musa Miah; Tatsuya Murakami; Md. Al Mehedi Hasan; Yong Seok Hwang; Jungpil Shin
>
> **备注:** 15 pages, 12 figures. Submitted to IEEE Access (under review)
>
> **摘要:** Hand gesture-based Sign Language Recognition (SLR) serves as a crucial communication bridge between deaf and non-deaf individuals. While Graph Convolutional Networks (GCNs) are common, they are limited by their reliance on fixed skeletal graphs. To overcome this, we propose the Sequential Spatio-Temporal Attention Network (SSTAN), a novel Transformer-based architecture. Our model employs a hierarchical, stacked design that sequentially integrates Spatial Multi-Head Attention (MHA) to capture intra-frame joint relationships and Temporal MHA to model long-range inter-frame dependencies. This approach allows the model to efficiently learn complex spatio-temporal patterns without predefined graph structures. We validated our model through extensive experiments on diverse, large-scale datasets (WLASL, JSL, and KSL). A key finding is that our model, trained entirely from scratch, achieves state-of-the-art (SOTA) performance in the challenging fingerspelling categories (JSL and KSL). Furthermore, it establishes a new SOTA for skeleton-only methods on WLASL, outperforming several approaches that rely on complex self-supervised pre-training. These results demonstrate our model's high data efficiency and its effectiveness in capturing the intricate dynamics of sign language. The official implementation is available at our GitHub repository: \href{https://github.com/K-Hirooka-Aizu/skeleton-slr-transformer}{https://github.com/K-Hirooka-Aizu/skeleton-slr-transformer}.
>
---
#### [replaced 128] Environment-Driven Online LiDAR-Camera Extrinsic Calibration
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.00801v3](http://arxiv.org/pdf/2502.00801v3)**

> **作者:** Zhiwei Huang; Jiaqi Li; Hongbo Zhao; Xiao Ma; Ping Zhong; Xiaohu Zhou; Wei Ye; Rui Fan
>
> **摘要:** LiDAR-camera extrinsic calibration (LCEC) is crucial for multi-modal data fusion in autonomous robotic systems. Existing methods, whether target-based or target-free, typically rely on customized calibration targets or fixed scene types, which limit their applicability in real-world scenarios. To address these challenges, we present EdO-LCEC, the first environment-driven online calibration approach. Unlike traditional target-free methods, EdO-LCEC employs a generalizable scene discriminator to estimate the feature density of the application environment. Guided by this feature density, EdO-LCEC extracts LiDAR intensity and depth features from varying perspectives to achieve higher calibration accuracy. To overcome the challenges of cross-modal feature matching between LiDAR and camera, we introduce dual-path correspondence matching (DPCM), which leverages both structural and textural consistency for reliable 3D-2D correspondences. Furthermore, we formulate the calibration process as a joint optimization problem that integrates global constraints across multiple views and scenes, thereby enhancing overall accuracy. Extensive experiments on real-world datasets demonstrate that EdO-LCEC outperforms state-of-the-art methods, particularly in scenarios involving sparse point clouds or partially overlapping sensor views.
>
---
#### [replaced 129] FaceSleuth-R: Adaptive Orientation-Aware Attention for Robust Micro-Expression Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02695v3](http://arxiv.org/pdf/2506.02695v3)**

> **作者:** Linquan Wu; Tianxiang Jiang; Haoyu Yang; Wenhao Duan; Shaochao Lin; Zixuan Wang; Yini Fang; Jacky Keung
>
> **摘要:** Micro-expression recognition (MER) has achieved impressive accuracy in controlled laboratory settings. However, its real-world applicability faces a significant generalization cliff, severely hindering practical deployment due to poor performance on unseen data and susceptibility to domain shifts. Existing attention mechanisms often overfit to dataset-specific appearance cues or rely on fixed spatial priors, making them fragile in diverse environments. We posit that robust MER requires focusing on quasi-invariant motion orientations inherent to micro-expressions, rather than superficial pixel-level features. To this end, we introduce \textbf{FaceSleuth-R}, a framework centered on our novel \textbf{Single-Orientation Attention (SOA)} module. SOA is a lightweight, differentiable operator that enables the network to learn layer-specific optimal orientations, effectively guiding attention towards these robust motion cues. Through extensive experiments, we demonstrate that SOA consistently discovers a universal near-vertical motion prior across diverse datasets. More critically, FaceSleuth-R showcases superior generalization in rigorous Leave-One-Dataset-Out (LODO) protocols, significantly outperforming baselines and state-of-the-art methods when confronted with domain shifts. Furthermore, our approach establishes \textbf{state-of-the-art results} across several benchmarks. This work highlights adaptive orientation-aware attention as a key paradigm for developing truly generalized and high-performing MER systems.
>
---
#### [replaced 130] VinDr-CXR-VQA: A Visual Question Answering Dataset for Explainable Chest X-Ray Analysis with Multi-Task Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2511.00504v2](http://arxiv.org/pdf/2511.00504v2)**

> **作者:** Dang H. Nguyen; Hieu H. Pham; Hao T. Nguyen; Hieu H. Pham
>
> **备注:** ISBI submission. Contains 5 pages, 2 figures, and 6 tables. Code & data: https://huggingface.co/datasets/Dangindev/VinDR-CXR-VQA
>
> **摘要:** We present VinDr-CXR-VQA, a large-scale chest X-ray dataset for explainable Medical Visual Question Answering (Med-VQA) with spatial grounding. The dataset contains 17,597 question-answer pairs across 4,394 images, each annotated with radiologist-verified bounding boxes and clinical reasoning explanations. Our question taxonomy spans six diagnostic types-Where, What, Is there, How many, Which, and Yes/No-capturing diverse clinical intents. To improve reliability, we construct a balanced distribution of 41.7% positive and 58.3% negative samples, mitigating hallucinations in normal cases. Benchmarking with MedGemma-4B-it demonstrates improved performance (F1 = 0.624, +11.8% over baseline) while enabling lesion localization. VinDr-CXR-VQA aims to advance reproducible and clinically grounded Med-VQA research. The dataset and evaluation tools are publicly available at huggingface.co/datasets/Dangindev/VinDR-CXR-VQA.
>
---
#### [replaced 131] A Lightweight Complex-Valued Deformable CNN for High-Quality Computer-Generated Holography
- **分类: physics.optics; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14542v2](http://arxiv.org/pdf/2506.14542v2)**

> **作者:** Shuyang Xie; Jie Zhou; Bo Xu; Jun Wang; Renjing Xu
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Holographic displays have significant potential in virtual reality and augmented reality owing to their ability to provide all the depth cues. Deep learning-based methods play an important role in computer-generated holography (CGH). During the diffraction process, each pixel exerts an influence on the reconstructed image. However, previous works face challenges in capturing sufficient information to accurately model this process, primarily due to the inadequacy of their effective receptive field (ERF). Here, we designed complex-valued deformable convolution for integration into network, enabling dynamic adjustment of the convolution kernel's shape to increase flexibility of ERF for better feature extraction. This approach allows us to utilize a single model while achieving state-of-the-art performance in both simulated and optical experiment reconstructions, surpassing existing open-source models. Specifically, our method has a peak signal-to-noise ratio that is 2.04 dB, 5.31 dB, and 9.71 dB higher than that of CCNN-CGH, HoloNet, and Holo-encoder, respectively, when the resolution is 1920$\times$1072. The number of parameters of our model is only about one-eighth of that of CCNN-CGH.
>
---
#### [replaced 132] FreeBlend: Advancing Concept Blending with Staged Feedback-Driven Interpolation Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.05606v3](http://arxiv.org/pdf/2502.05606v3)**

> **作者:** Yufan Zhou; Haoyu Shen; Huan Wang
>
> **备注:** Webpage: https://petershen-csworld.github.io/FreeBlend
>
> **摘要:** Concept blending is a promising yet underexplored area in generative models. While recent approaches, such as embedding mixing and latent modification based on structural sketches, have been proposed, they often suffer from incompatible semantic information and discrepancies in shape and appearance. In this work, we introduce FreeBlend, an effective, training-free framework designed to address these challenges. To mitigate cross-modal loss and enhance feature detail, we leverage transferred image embeddings as conditional inputs. The framework employs a stepwise increasing interpolation strategy between latents, progressively adjusting the blending ratio to seamlessly integrate auxiliary features. Additionally, we introduce a feedback-driven mechanism that updates the auxiliary latents in reverse order, facilitating global blending and preventing rigid or unnatural outputs. Extensive experiments demonstrate that our method significantly improves both the semantic coherence and visual quality of blended images, yielding compelling and coherent results.
>
---
#### [replaced 133] RaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cues for 3D Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.19856v3](http://arxiv.org/pdf/2507.19856v3)**

> **作者:** Xiaokai Bai; Chenxu Zhou; Lianqing Zheng; Si-Yuan Cao; Jianan Liu; Xiaohan Zhang; Yiming Li; Zhengzhuang Zhang; Hui-liang Shen
>
> **摘要:** 4D millimeter-wave radar is a promising sensing modality for autonomous driving, yet effective 3D object detection from 4D radar and monocular images remains challenging. Existing fusion approaches either rely on instance proposals lacking global context or dense BEV grids constrained by rigid structures, lacking a flexible and adaptive representation for diverse scenes. To address this, we propose RaGS, the first framework that leverages 3D Gaussian Splatting (GS) to fuse 4D radar and monocular cues for 3D object detection. 3D GS models the scene as a continuous field of Gaussians, enabling dynamic resource allocation to foreground objects while maintaining flexibility and efficiency. Moreover, the velocity dimension of 4D radar provides motion cues that help anchor and refine the spatial distribution of Gaussians. Specifically, RaGS adopts a cascaded pipeline to construct and progressively refine the Gaussian field. It begins with Frustum-based Localization Initiation (FLI), which unprojects foreground pixels to initialize coarse Gaussian centers. Then, Iterative Multimodal Aggregation (IMA) explicitly exploits image semantics and implicitly integrates 4D radar velocity geometry to refine the Gaussians within regions of interest. Finally, Multi-level Gaussian Fusion (MGF) renders the Gaussian field into hierarchical BEV features for 3D object detection. By dynamically focusing on sparse and informative regions, RaGS achieves object-centric precision and comprehensive scene perception. Extensive experiments on View-of-Delft, TJ4DRadSet, and OmniHD-Scenes demonstrate its robustness and SOTA performance. Code will be released.
>
---
#### [replaced 134] X2Edit: Revisiting Arbitrary-Instruction Image Editing through Self-Constructed Data and Task-Aware Representation Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.07607v2](http://arxiv.org/pdf/2508.07607v2)**

> **作者:** Jian Ma; Xujie Zhu; Zihao Pan; Qirong Peng; Xu Guo; Chen Chen; Haonan Lu
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Existing open-source datasets for arbitrary-instruction image editing remain suboptimal, while a plug-and-play editing module compatible with community-prevalent generative models is notably absent. In this paper, we first introduce the X2Edit Dataset, a comprehensive dataset covering 14 diverse editing tasks, including subject-driven generation. We utilize the industry-leading unified image generation models and expert models to construct the data. Meanwhile, we design reasonable editing instructions with the VLM and implement various scoring mechanisms to filter the data. As a result, we construct 3.7 million high-quality data with balanced categories. Second, to better integrate seamlessly with community image generation models, we design task-aware MoE-LoRA training based on FLUX.1, with only 8\% of the parameters of the full model. To further improve the final performance, we utilize the internal representations of the diffusion model and define positive/negative samples based on image editing types to introduce contrastive learning. Extensive experiments demonstrate that the model's editing performance is competitive among many excellent models. Additionally, the constructed dataset exhibits substantial advantages over existing open-source datasets. The open-source code, checkpoints, and datasets for X2Edit can be found at the following link: https://github.com/OPPO-Mente-Lab/X2Edit.
>
---
#### [replaced 135] Bridging Weakly-Supervised Learning and VLM Distillation: Noisy Partial Label Learning for Efficient Downstream Adaptation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.03229v2](http://arxiv.org/pdf/2506.03229v2)**

> **作者:** Qian-Wei Wang; Yuqiu Xie; Letian Zhang; Zimo Liu; Shu-Tao Xia
>
> **摘要:** In the context of noisy partial label learning (NPLL), each training sample is associated with a set of candidate labels annotated by multiple noisy annotators. With the emergence of high-performance pre-trained vision-language models (VLMs) such as CLIP, LLaVA and GPT-4V, the direction of using these models to replace time-consuming manual annotation workflows and achieve ``manual-annotation-free" training for downstream tasks has become a highly promising research avenue. This paper focuses on learning from noisy partial labels annotated by pre-trained VLMs and proposes an innovative collaborative consistency regularization (Co-Reg) method. Unlike the symmetric noise primarily addressed in traditional noisy label learning, the noise generated by pre-trained models is instance-dependent, embodying the underlying patterns of the pre-trained models themselves, which significantly increases the learning difficulty for the model. To address this, we simultaneously train two neural networks that implement collaborative purification of training labels through a ``Co-Pseudo-Labeling" mechanism, while enforcing consistency regularization constraints in both the label space and feature representation space. Specifically, we construct multiple anti-overfitting mechanisms that efficiently mine latent information from noisy partially labeled samples including alternating optimization of contrastive feature representations and pseudo-labels, as well as maintaining prototypical class vectors in the shared feature space.
>
---
#### [replaced 136] Real-time Multi-view Omnidirectional Depth Estimation for Real Scenarios based on Teacher-Student Learning with Unlabeled Data
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.07843v2](http://arxiv.org/pdf/2409.07843v2)**

> **作者:** Ming Li; Xiong Yang; Chaofan Wu; Jiaheng Li; Pinzhi Wang; Xuejiao Hu; Sidan Du; Yang Li
>
> **摘要:** Omnidirectional depth estimation enables efficient 3D perception over a full 360-degree range. However, in real-world applications such as autonomous driving and robotics, achieving real-time performance and robust cross-scene generalization remains a significant challenge for existing algorithms. In this paper, we propose a real-time omnidirectional depth estimation method for edge computing platforms named Rt-OmniMVS, which introduces the Combined Spherical Sweeping method and implements the lightweight network structure to achieve real-time performance on edge computing platforms. To achieve high accuracy, robustness, and generalization in real-world environments, we introduce a teacher-student learning strategy. We leverage the high-precision stereo matching method as the teacher model to predict pseudo labels for unlabeled real-world data, and utilize data and model augmentation techniques for training to enhance performance of the student model Rt-OmniMVS. We also propose HexaMODE, an omnidirectional depth sensing system based on multi-view fisheye cameras and edge computation device. A large-scale hybrid dataset contains both unlabeled real-world data and synthetic data is collected for model training. Experiments on public datasets demonstrate that proposed method achieves results comparable to state-of-the-art approaches while consuming significantly less resource. The proposed system and algorithm also demonstrate high accuracy in various complex real-world scenarios, both indoors and outdoors, achieving an inference speed of 15 frames per second on edge computing platforms.
>
---
#### [replaced 137] Seg2Any: Open-set Segmentation-Mask-to-Image Generation with Precise Shape and Semantic Control
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00596v3](http://arxiv.org/pdf/2506.00596v3)**

> **作者:** Danfeng Li; Hui Zhang; Sheng Wang; Jiacheng Li; Zuxuan Wu
>
> **摘要:** Despite recent advances in diffusion models, top-tier text-to-image (T2I) models still struggle to achieve precise spatial layout control, i.e. accurately generating entities with specified attributes and locations. Segmentation-mask-to-image (S2I) generation has emerged as a promising solution by incorporating pixel-level spatial guidance and regional text prompts. However, existing S2I methods fail to simultaneously ensure semantic consistency and shape consistency. To address these challenges, we propose Seg2Any, a novel S2I framework built upon advanced multimodal diffusion transformers (e.g. FLUX). First, to achieve both semantic and shape consistency, we decouple segmentation mask conditions into regional semantic and high-frequency shape components. The regional semantic condition is introduced by a Semantic Alignment Attention Mask, ensuring that generated entities adhere to their assigned text prompts. The high-frequency shape condition, representing entity boundaries, is encoded as an Entity Contour Map and then introduced as an additional modality via multi-modal attention to guide image spatial structure. Second, to prevent attribute leakage across entities in multi-entity scenarios, we introduce an Attribute Isolation Attention Mask mechanism, which constrains each entity's image tokens to attend exclusively to themselves during image self-attention. To support open-set S2I generation, we construct SACap-1M, a large-scale dataset containing 1 million images with 5.9 million segmented entities and detailed regional captions, along with a SACap-Eval benchmark for comprehensive S2I evaluation. Extensive experiments demonstrate that Seg2Any achieves state-of-the-art performance on both open-set and closed-set S2I benchmarks, particularly in fine-grained spatial and attribute control of entities.
>
---
#### [replaced 138] MACS: Multi-source Audio-to-image Generation with Contextual Significance and Semantic Alignment
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.10287v2](http://arxiv.org/pdf/2503.10287v2)**

> **作者:** Hao Zhou; Xiaobao Guo; Yuzhe Zhu; Adams Wai-Kin Kong
>
> **备注:** Accepted at AAAI 2026. Code available at https://github.com/alxzzhou/MACS
>
> **摘要:** Propelled by the breakthrough in deep generative models, audio-to-image generation has emerged as a pivotal cross-modal task that converts complex auditory signals into rich visual representations. However, previous works only focus on single-source audio inputs for image generation, ignoring the multi-source characteristic in natural auditory scenes, thus limiting the performance in generating comprehensive visual content. To bridge this gap, we propose a method called MACS to conduct multi-source audio-to-image generation. To our best knowledge, this is the first work that explicitly separates multi-source audio to capture the rich audio components before image generation. MACS is a two-stage method. In the first stage, multi-source audio inputs are separated by a weakly supervised method, where the audio and text labels are semantically aligned by casting into a common space using the large pre-trained CLAP model. We introduce a ranking loss to consider the contextual significance of the separated audio signals. In the second stage, effective image generation is achieved by mapping the separated audio signals to the generation condition using only a trainable adapter and a MLP layer. We preprocess the LLP dataset as the first full multi-source audio-to-image generation benchmark. The experiments are conducted on multi-source, mixed-source, and single-source audio-to-image generation tasks. The proposed MACS outperforms the current state-of-the-art methods in 17 out of the 21 evaluation indexes on all tasks and delivers superior visual quality.
>
---
