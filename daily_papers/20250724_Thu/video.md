# 计算机视觉 cs.CV

- **最新发布 118 篇**

- **更新 71 篇**

## 最新发布

#### [new 001] Post-Disaster Affected Area Segmentation with a Vision Transformer (ViT)-based EVAP Model using Sentinel-2 and Formosat-5 Imagery
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像分割任务，旨在解决灾后受灾区域自动精确分割的问题。作者提出基于视觉变换器（ViT）的EVAP模型，结合Sentinel-2和福尔摩沙卫星五号影像，通过弱监督学习方法扩展标签并训练模型，提升了分割结果的平滑性与可靠性，适用于缺乏精确地面真值的灾害制图场景。**

- **链接: [http://arxiv.org/pdf/2507.16849v1](http://arxiv.org/pdf/2507.16849v1)**

> **作者:** Yi-Shan Chu; Hsuan-Cheng Wei
>
> **摘要:** We propose a vision transformer (ViT)-based deep learning framework to refine disaster-affected area segmentation from remote sensing imagery, aiming to support and enhance the Emergent Value Added Product (EVAP) developed by the Taiwan Space Agency (TASA). The process starts with a small set of manually annotated regions. We then apply principal component analysis (PCA)-based feature space analysis and construct a confidence index (CI) to expand these labels, producing a weakly supervised training set. These expanded labels are then used to train ViT-based encoder-decoder models with multi-band inputs from Sentinel-2 and Formosat-5 imagery. Our architecture supports multiple decoder variants and multi-stage loss strategies to improve performance under limited supervision. During the evaluation, model predictions are compared with higher-resolution EVAP output to assess spatial coherence and segmentation consistency. Case studies on the 2022 Poyang Lake drought and the 2023 Rhodes wildfire demonstrate that our framework improves the smoothness and reliability of segmentation results, offering a scalable approach for disaster mapping when accurate ground truth is unavailable.
>
---
#### [new 002] Controllable Hybrid Captioner for Improved Long-form Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解任务，旨在解决长视频内容难以高效处理的问题。通过构建基于文本的记忆系统，结合视频分段描述与静态场景分析，提升对复杂自然语言问题的回答能力。使用LaViLa与LLaVA模型融合动作与场景描述，并优化分段策略，增强文本摘要的准确性与完整性。**

- **链接: [http://arxiv.org/pdf/2507.17047v1](http://arxiv.org/pdf/2507.17047v1)**

> **作者:** Kuleen Sasse; Efsun Sarioglu Kayi; Arun Reddy
>
> **摘要:** Video data, especially long-form video, is extremely dense and high-dimensional. Text-based summaries of video content offer a way to represent query-relevant content in a much more compact manner than raw video. In addition, textual representations are easily ingested by state-of-the-art large language models (LLMs), which enable reasoning over video content to answer complex natural language queries. To solve this issue, we rely on the progressive construction of a text-based memory by a video captioner operating on shorter chunks of the video, where spatio-temporal modeling is computationally feasible. We explore ways to improve the quality of the activity log comprised solely of short video captions. Because the video captions tend to be focused on human actions, and questions may pertain to other information in the scene, we seek to enrich the memory with static scene descriptions using Vision Language Models (VLMs). Our video understanding system relies on the LaViLa video captioner in combination with a LLM to answer questions about videos. We first explored different ways of partitioning the video into meaningful segments such that the textual descriptions more accurately reflect the structure of the video content. Furthermore, we incorporated static scene descriptions into the captioning pipeline using LLaVA VLM, resulting in a more detailed and complete caption log and expanding the space of questions that are answerable from the textual memory. Finally, we have successfully fine-tuned the LaViLa video captioner to produce both action and scene captions, significantly improving the efficiency of the captioning pipeline compared to using separate captioning models for the two tasks. Our model, controllable hybrid captioner, can alternate between different types of captions according to special input tokens that signals scene changes detected in the video.
>
---
#### [new 003] UNICE: Training A Universal Image Contrast Enhancer
- **分类: cs.CV**

- **简介: 该论文属于图像对比度增强任务，旨在解决现有方法泛化能力差的问题。作者提出UNICE框架，利用HDR图像生成多曝光序列并融合，训练无需人工标注的通用增强模型，显著提升跨任务与跨数据集的对比度增强效果。**

- **链接: [http://arxiv.org/pdf/2507.17157v1](http://arxiv.org/pdf/2507.17157v1)**

> **作者:** Ruodai Cui; Lei Zhang
>
> **摘要:** Existing image contrast enhancement methods are typically designed for specific tasks such as under-/over-exposure correction, low-light and backlit image enhancement, etc. The learned models, however, exhibit poor generalization performance across different tasks, even across different datasets of a specific task. It is important to explore whether we can learn a universal and generalized model for various contrast enhancement tasks. In this work, we observe that the common key factor of these tasks lies in the need of exposure and contrast adjustment, which can be well-addressed if high-dynamic range (HDR) inputs are available. We hence collect 46,928 HDR raw images from public sources, and render 328,496 sRGB images to build multi-exposure sequences (MES) and the corresponding pseudo sRGB ground-truths via multi-exposure fusion. Consequently, we train a network to generate an MES from a single sRGB image, followed by training another network to fuse the generated MES into an enhanced image. Our proposed method, namely UNiversal Image Contrast Enhancer (UNICE), is free of costly human labeling. However, it demonstrates significantly stronger generalization performance than existing image contrast enhancement methods across and within different tasks, even outperforming manually created ground-truths in multiple no-reference image quality metrics. The dataset, code and model are available at https://github.com/BeyondHeaven/UNICE.
>
---
#### [new 004] DesignLab: Designing Slides Through Iterative Detection and Correction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于幻灯片设计优化任务，旨在解决非专业人士难以制作高质量幻灯片的问题。论文提出DesignLab框架，通过迭代检测与修正设计问题，提升自动生成效果。利用大模型分别担任评审与修改角色，模拟设计流程，最终实现更专业的幻灯片生成。**

- **链接: [http://arxiv.org/pdf/2507.17202v1](http://arxiv.org/pdf/2507.17202v1)**

> **作者:** Jooyeol Yun; Heng Wang; Yotaro Shimose; Jaegul Choo; Shingo Takamatsu
>
> **备注:** https://yeolj00.github.io/personal-projects/designlab
>
> **摘要:** Designing high-quality presentation slides can be challenging for non-experts due to the complexity involved in navigating various design choices. Numerous automated tools can suggest layouts and color schemes, yet often lack the ability to refine their own output, which is a key aspect in real-world workflows. We propose DesignLab, which separates the design process into two roles, the design reviewer, who identifies design-related issues, and the design contributor who corrects them. This decomposition enables an iterative loop where the reviewer continuously detects issues and the contributor corrects them, allowing a draft to be further polished with each iteration, reaching qualities that were unattainable. We fine-tune large language models for these roles and simulate intermediate drafts by introducing controlled perturbations, enabling the design reviewer learn design errors and the contributor learn how to fix them. Our experiments show that DesignLab outperforms existing design-generation methods, including a commercial tool, by embracing the iterative nature of designing which can result in polished, professional slides.
>
---
#### [new 005] Ultra3D: Efficient and High-Fidelity 3D Generation with Part Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D内容生成任务，旨在解决现有方法因注意力机制计算效率低导致生成速度慢的问题。工作包括：提出Ultra3D框架，使用VecSet生成粗略布局，减少标记数量；设计Part Attention机制，在局部区域内进行注意力计算，提升效率；构建部分标注流程，支持高效生成高分辨率3D模型。**

- **链接: [http://arxiv.org/pdf/2507.17745v1](http://arxiv.org/pdf/2507.17745v1)**

> **作者:** Yiwen Chen; Zhihao Li; Yikai Wang; Hu Zhang; Qin Li; Chi Zhang; Guosheng Lin
>
> **备注:** Project Page: https://buaacyw.github.io/ultra3d/
>
> **摘要:** Recent advances in sparse voxel representations have significantly improved the quality of 3D content generation, enabling high-resolution modeling with fine-grained geometry. However, existing frameworks suffer from severe computational inefficiencies due to the quadratic complexity of attention mechanisms in their two-stage diffusion pipelines. In this work, we propose Ultra3D, an efficient 3D generation framework that significantly accelerates sparse voxel modeling without compromising quality. Our method leverages the compact VecSet representation to efficiently generate a coarse object layout in the first stage, reducing token count and accelerating voxel coordinate prediction. To refine per-voxel latent features in the second stage, we introduce Part Attention, a geometry-aware localized attention mechanism that restricts attention computation within semantically consistent part regions. This design preserves structural continuity while avoiding unnecessary global attention, achieving up to 6.7x speed-up in latent generation. To support this mechanism, we construct a scalable part annotation pipeline that converts raw meshes into part-labeled sparse voxels. Extensive experiments demonstrate that Ultra3D supports high-resolution 3D generation at 1024 resolution and achieves state-of-the-art performance in both visual fidelity and user preference.
>
---
#### [new 006] PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决现有端到端模型依赖昂贵传感器、计算复杂的问题。作者提出PRIX，一种仅使用摄像头原始像素进行路径规划的高效架构，无需显式BEV表示或LiDAR。核心创新为上下文感知重校准Transformer（CaRT），提升视觉特征鲁棒性，实验证明其在主流数据集上性能领先，且更轻量、快速，适合实际部署。**

- **链接: [http://arxiv.org/pdf/2507.17596v1](http://arxiv.org/pdf/2507.17596v1)**

> **作者:** Maciej K. Wozniak; Lianhang Liu; Yixi Cai; Patric Jensfelt
>
> **备注:** under review
>
> **摘要:** While end-to-end autonomous driving models show promising results, their practical deployment is often hindered by large model sizes, a reliance on expensive LiDAR sensors and computationally intensive BEV feature representations. This limits their scalability, especially for mass-market vehicles equipped only with cameras. To address these challenges, we propose PRIX (Plan from Raw Pixels). Our novel and efficient end-to-end driving architecture operates using only camera data, without explicit BEV representation and forgoing the need for LiDAR. PRIX leverages a visual feature extractor coupled with a generative planning head to predict safe trajectories from raw pixel inputs directly. A core component of our architecture is the Context-aware Recalibration Transformer (CaRT), a novel module designed to effectively enhance multi-level visual features for more robust planning. We demonstrate through comprehensive experiments that PRIX achieves state-of-the-art performance on the NavSim and nuScenes benchmarks, matching the capabilities of larger, multimodal diffusion planners while being significantly more efficient in terms of inference speed and model size, making it a practical solution for real-world deployment. Our work is open-source and the code will be at https://maxiuw.github.io/prix.
>
---
#### [new 007] TransLPRNet: Lite Vision-Language Network for Single/Dual-line Chinese License Plate Recognition
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于车牌识别任务，旨在解决复杂环境下中英文车牌识别准确率低的问题。作者提出了TransLPRNet模型，结合轻量级视觉编码器和文本解码器，并构建了双行车牌数据集。此外，引入透视校正网络（PTN）提升识别稳定性与精度，实现了高速度和高准确率。**

- **链接: [http://arxiv.org/pdf/2507.17335v1](http://arxiv.org/pdf/2507.17335v1)**

> **作者:** Guangzhu Xu; Zhi Ke; Pengcheng Zuo; Bangjun Lei
>
> **摘要:** License plate recognition in open environments is widely applicable across various domains; however, the diversity of license plate types and imaging conditions presents significant challenges. To address the limitations encountered by CNN and CRNN-based approaches in license plate recognition, this paper proposes a unified solution that integrates a lightweight visual encoder with a text decoder, within a pre-training framework tailored for single and double-line Chinese license plates. To mitigate the scarcity of double-line license plate datasets, we constructed a single/double-line license plate dataset by synthesizing images, applying texture mapping onto real scenes, and blending them with authentic license plate images. Furthermore, to enhance the system's recognition accuracy, we introduce a perspective correction network (PTN) that employs license plate corner coordinate regression as an implicit variable, supervised by license plate view classification information. This network offers improved stability, interpretability, and low annotation costs. The proposed algorithm achieves an average recognition accuracy of 99.34% on the corrected CCPD test set under coarse localization disturbance. When evaluated under fine localization disturbance, the accuracy further improves to 99.58%. On the double-line license plate test set, it achieves an average recognition accuracy of 98.70%, with processing speeds reaching up to 167 frames per second, indicating strong practical applicability.
>
---
#### [new 008] Attention (as Discrete-Time Markov) Chains
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与自然语言处理任务，旨在分析视觉Transformer中的注意力机制。论文将注意力矩阵解释为离散时间马尔可夫链，提出间接注意力与TokenRank指标，解决注意力机制中语义相似区域识别与图像分割问题，实现了零样本分割与图像生成优化。**

- **链接: [http://arxiv.org/pdf/2507.17657v1](http://arxiv.org/pdf/2507.17657v1)**

> **作者:** Yotam Erel; Olaf Dünkel; Rishabh Dabral; Vladislav Golyanik; Christian Theobalt; Amit H. Bermano
>
> **备注:** Project page: https://yoterel.github.io/attention_chains/
>
> **摘要:** We introduce a new interpretation of the attention matrix as a discrete-time Markov chain. Our interpretation sheds light on common operations involving attention scores such as selection, summation, and averaging in a unified framework. It further extends them by considering indirect attention, propagated through the Markov chain, as opposed to previous studies that only model immediate effects. Our main observation is that tokens corresponding to semantically similar regions form a set of metastable states, where the attention clusters, while noisy attention scores tend to disperse. Metastable states and their prevalence can be easily computed through simple matrix multiplication and eigenanalysis, respectively. Using these lightweight tools, we demonstrate state-of-the-art zero-shot segmentation. Lastly, we define TokenRank -- the steady state vector of the Markov chain, which measures global token importance. We demonstrate that using it brings improvements in unconditional image generation. We believe our framework offers a fresh view of how tokens are being attended in modern visual transformers.
>
---
#### [new 009] Perceptual Classifiers: Detecting Generative Images using Perceptual Features
- **分类: cs.CV**

- **简介: 该论文属于图像生成内容检测任务，旨在解决区分真实图像与AI生成图像的问题。作者利用图像质量评估模型的感知特征，构建分类器以检测生成图像，并验证其跨生成模型的泛化能力和对图像退化的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.17240v1](http://arxiv.org/pdf/2507.17240v1)**

> **作者:** Krishna Srikar Durbha; Asvin Kumar Venkataramanan; Rajesh Sureddi; Alan C. Bovik
>
> **备注:** 8 pages, 6 figures, 3 tables, ICCV VQualA Workshop 2025
>
> **摘要:** Image Quality Assessment (IQA) models are employed in many practical image and video processing pipelines to reduce storage, minimize transmission costs, and improve the Quality of Experience (QoE) of millions of viewers. These models are sensitive to a diverse range of image distortions and can accurately predict image quality as judged by human viewers. Recent advancements in generative models have resulted in a significant influx of "GenAI" content on the internet. Existing methods for detecting GenAI content have progressed significantly with improved generalization performance on images from unseen generative models. Here, we leverage the capabilities of existing IQA models, which effectively capture the manifold of real images within a bandpass statistical space, to distinguish between real and AI-generated images. We investigate the generalization ability of these perceptual classifiers to the task of GenAI image detection and evaluate their robustness against various image degradations. Our results show that a two-layer network trained on the feature space of IQA models demonstrates state-of-the-art performance in detecting fake images across generative models, while maintaining significant robustness against image degradations.
>
---
#### [new 010] Finding Dori: Memorization in Text-to-Image Diffusion Models Is Less Local Than Assumed
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究文本到图像扩散模型中的记忆问题，探讨其潜在隐私与版权风险。现有方法假设记忆可被局部定位并剪枝消除，但论文发现该假设不成立。通过分析文本嵌入空间，作者发现记忆触发路径多样，表明剪枝策略无效。为解决此问题，他们提出一种对抗微调方法，逐步搜索并消除记忆触发点，提升模型鲁棒性，旨在构建更可信的生成模型。**

- **链接: [http://arxiv.org/pdf/2507.16880v1](http://arxiv.org/pdf/2507.16880v1)**

> **作者:** Antoni Kowalczuk; Dominik Hintersdorf; Lukas Struppek; Kristian Kersting; Adam Dziedzic; Franziska Boenisch
>
> **摘要:** Text-to-image diffusion models (DMs) have achieved remarkable success in image generation. However, concerns about data privacy and intellectual property remain due to their potential to inadvertently memorize and replicate training data. Recent mitigation efforts have focused on identifying and pruning weights responsible for triggering replication, based on the assumption that memorization can be localized. Our research assesses the robustness of these pruning-based approaches. We demonstrate that even after pruning, minor adjustments to text embeddings of input prompts are sufficient to re-trigger data replication, highlighting the fragility of these defenses. Furthermore, we challenge the fundamental assumption of memorization locality, by showing that replication can be triggered from diverse locations within the text embedding space, and follows different paths in the model. Our findings indicate that existing mitigation strategies are insufficient and underscore the need for methods that truly remove memorized content, rather than attempting to suppress its retrieval. As a first step in this direction, we introduce a novel adversarial fine-tuning method that iteratively searches for replication triggers and updates the model to increase robustness. Through our research, we provide fresh insights into the nature of memorization in text-to-image DMs and a foundation for building more trustworthy and compliant generative AI.
>
---
#### [new 011] DOOMGAN:High-Fidelity Dynamic Identity Obfuscation Ocular Generative Morphing
- **分类: cs.CV**

- **简介: 该论文属于生物特征安全任务，旨在解决可见光眼部数据的合成攻击问题。作者提出了DOOMGAN模型，通过关键点编码、注意力生成和动态损失加权，生成高保真眼部合成图像，提升了攻击成功率及眼部结构准确性，并发布了首个相关数据集。**

- **链接: [http://arxiv.org/pdf/2507.17158v1](http://arxiv.org/pdf/2507.17158v1)**

> **作者:** Bharath Krishnamurthy; Ajita Rattani
>
> **备注:** Accepted to IJCB 2025 (IEEE/IAPR International Joint Conference on Biometrics). 11 pages with references, 8-page main paper with 4 figures and 4 tables. Includes 6 pages of supplementary material with 3 additional figures and 3 tables. Code is available at the official lab repository: https://github.com/vcbsl/DOOMGAN and the author's repository: https://github.com/Bharath-K3/DOOMGAN
>
> **摘要:** Ocular biometrics in the visible spectrum have emerged as a prominent modality due to their high accuracy, resistance to spoofing, and non-invasive nature. However, morphing attacks, synthetic biometric traits created by blending features from multiple individuals, threaten biometric system integrity. While extensively studied for near-infrared iris and face biometrics, morphing in visible-spectrum ocular data remains underexplored. Simulating such attacks demands advanced generation models that handle uncontrolled conditions while preserving detailed ocular features like iris boundaries and periocular textures. To address this gap, we introduce DOOMGAN, that encompasses landmark-driven encoding of visible ocular anatomy, attention-guided generation for realistic morph synthesis, and dynamic weighting of multi-faceted losses for optimized convergence. DOOMGAN achieves over 20% higher attack success rates than baseline methods under stringent thresholds, along with 20% better elliptical iris structure generation and 30% improved gaze consistency. We also release the first comprehensive ocular morphing dataset to support further research in this domain.
>
---
#### [new 012] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts
- **分类: cs.CV**

- **简介: 该论文属于异常检测任务，旨在解决长尾分布数据下的在线异常检测问题。现有方法依赖类别标签，不适用于无标签的在线场景。作者提出了一种类无关的长尾异常检测框架，并将其扩展到在线学习设置，取得了优于现有方法的性能，适用于工业和医疗领域。**

- **链接: [http://arxiv.org/pdf/2507.16946v1](http://arxiv.org/pdf/2507.16946v1)**

> **作者:** Chiao-An Yang; Kuan-Chuan Peng; Raymond A. Yeh
>
> **备注:** This paper is accepted to ICCV 2025. The supplementary material is included. The long-tailed online anomaly detection dataset is available at https://doi.org/10.5281/zenodo.16283852
>
> **摘要:** Anomaly detection (AD) identifies the defect regions of a given image. Recent works have studied AD, focusing on learning AD without abnormal images, with long-tailed distributed training data, and using a unified model for all classes. In addition, online AD learning has also been explored. In this work, we expand in both directions to a realistic setting by considering the novel task of long-tailed online AD (LTOAD). We first identified that the offline state-of-the-art LTAD methods cannot be directly applied to the online setting. Specifically, LTAD is class-aware, requiring class labels that are not available in the online setting. To address this challenge, we propose a class-agnostic framework for LTAD and then adapt it to our online learning setting. Our method outperforms the SOTA baselines in most offline LTAD settings, including both the industrial manufacturing and the medical domain. In particular, we observe +4.63% image-AUROC on MVTec even compared to methods that have access to class labels and the number of classes. In the most challenging long-tailed online setting, we achieve +0.53% image-AUROC compared to baselines. Our LTOAD benchmark is released here: https://doi.org/10.5281/zenodo.16283852 .
>
---
#### [new 013] CartoonAlive: Towards Expressive Live2D Modeling from Single Portraits
- **分类: cs.CV**

- **简介: 该论文属于数字人建模任务，旨在解决从单张肖像图快速生成高质量、可交互的2D卡通风格Live2D模型的问题。作者提出CartoonAlive方法，利用3D人脸建模中的形状基思想，构建适用于Live2D的面部混合形状，并根据检测到的面部关键点推断混合权重，实现高效且富有表现力的Live2D模型生成。**

- **链接: [http://arxiv.org/pdf/2507.17327v1](http://arxiv.org/pdf/2507.17327v1)**

> **作者:** Chao He; Jianqiang Ren; Jianjing Xiang; Xiejie Shen
>
> **摘要:** With the rapid advancement of large foundation models, AIGC, cloud rendering, and real-time motion capture technologies, digital humans are now capable of achieving synchronized facial expressions and body movements, engaging in intelligent dialogues driven by natural language, and enabling the fast creation of personalized avatars. While current mainstream approaches to digital humans primarily focus on 3D models and 2D video-based representations, interactive 2D cartoon-style digital humans have received relatively less attention. Compared to 3D digital humans that require complex modeling and high rendering costs, and 2D video-based solutions that lack flexibility and real-time interactivity, 2D cartoon-style Live2D models offer a more efficient and expressive alternative. By simulating 3D-like motion through layered segmentation without the need for traditional 3D modeling, Live2D enables dynamic and real-time manipulation. In this technical report, we present CartoonAlive, an innovative method for generating high-quality Live2D digital humans from a single input portrait image. CartoonAlive leverages the shape basis concept commonly used in 3D face modeling to construct facial blendshapes suitable for Live2D. It then infers the corresponding blendshape weights based on facial keypoints detected from the input image. This approach allows for the rapid generation of a highly expressive and visually accurate Live2D model that closely resembles the input portrait, within less than half a minute. Our work provides a practical and scalable solution for creating interactive 2D cartoon characters, opening new possibilities in digital content creation and virtual character animation. The project homepage is https://human3daigc.github.io/CartoonAlive_webpage/.
>
---
#### [new 014] Unsupervised anomaly detection using Bayesian flow networks: application to brain FDG PET in the context of Alzheimer's disease
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决阿尔茨海默病相关的无监督异常检测问题。作者提出AnoBFN方法，基于贝叶斯流网络，实现高噪声环境下保持个体特异性的异常检测。在FDG PET图像上验证，效果优于VAE、GAN和扩散模型等现有方法。**

- **链接: [http://arxiv.org/pdf/2507.17486v1](http://arxiv.org/pdf/2507.17486v1)**

> **作者:** Hugues Roy; Reuben Dorent; Ninon Burgos
>
> **摘要:** Unsupervised anomaly detection (UAD) plays a crucial role in neuroimaging for identifying deviations from healthy subject data and thus facilitating the diagnosis of neurological disorders. In this work, we focus on Bayesian flow networks (BFNs), a novel class of generative models, which have not yet been applied to medical imaging or anomaly detection. BFNs combine the strength of diffusion frameworks and Bayesian inference. We introduce AnoBFN, an extension of BFNs for UAD, designed to: i) perform conditional image generation under high levels of spatially correlated noise, and ii) preserve subject specificity by incorporating a recursive feedback from the input image throughout the generative process. We evaluate AnoBFN on the challenging task of Alzheimer's disease-related anomaly detection in FDG PET images. Our approach outperforms other state-of-the-art methods based on VAEs (beta-VAE), GANs (f-AnoGAN), and diffusion models (AnoDDPM), demonstrating its effectiveness at detecting anomalies while reducing false positive rates.
>
---
#### [new 015] Robust Five-Class and binary Diabetic Retinopathy Classification Using Transfer Learning and Data Augmentation
- **分类: cs.CV; cs.LG; F.2.2; I.2.7**

- **简介: 该论文属于医学图像分类任务，旨在解决糖尿病视网膜病变的自动诊断问题。通过迁移学习和数据增强，构建了鲁棒的深度学习模型，实现二分类和五类严重程度分类，取得了优异性能。**

- **链接: [http://arxiv.org/pdf/2507.17121v1](http://arxiv.org/pdf/2507.17121v1)**

> **作者:** Faisal Ahmed; Mohammad Alfrad Nobel Bhuiyan
>
> **备注:** 9 pages, 1 Figure
>
> **摘要:** Diabetic retinopathy (DR) is a leading cause of vision loss worldwide, and early diagnosis through automated retinal image analysis can significantly reduce the risk of blindness. This paper presents a robust deep learning framework for both binary and five-class DR classification, leveraging transfer learning and extensive data augmentation to address the challenges of class imbalance and limited training data. We evaluate a range of pretrained convolutional neural network architectures, including variants of ResNet and EfficientNet, on the APTOS 2019 dataset. For binary classification, our proposed model achieves a state-of-the-art accuracy of 98.9%, with a precision of 98.6%, recall of 99.3%, F1-score of 98.9%, and an AUC of 99.4%. In the more challenging five-class severity classification task, our model obtains a competitive accuracy of 84.6% and an AUC of 94.1%, outperforming several existing approaches. Our findings also demonstrate that EfficientNet-B0 and ResNet34 offer optimal trade-offs between accuracy and computational efficiency across both tasks. These results underscore the effectiveness of combining class-balanced augmentation with transfer learning for high-performance DR diagnosis. The proposed framework provides a scalable and accurate solution for DR screening, with potential for deployment in real-world clinical environments.
>
---
#### [new 016] SIA: Enhancing Safety via Intent Awareness for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态安全任务，旨在解决视觉-语言模型（VLM）中因图文组合引发的潜在安全风险。现有方法难以检测隐性风险，作者提出SIA框架，通过视觉抽象、意图推理和响应优化三阶段提示工程，动态识别并缓解有害意图。实验表明其在多个安全基准上效果优于先前方法。**

- **链接: [http://arxiv.org/pdf/2507.16856v1](http://arxiv.org/pdf/2507.16856v1)**

> **作者:** Youngjin Na; Sangheon Jeong; Youngwan Lee
>
> **备注:** 5 pages, 6 figures
>
> **摘要:** As vision-language models (VLMs) are increasingly deployed in real-world applications, new safety risks arise from the subtle interplay between images and text. In particular, seemingly innocuous inputs can combine to reveal harmful intent, leading to unsafe model responses. Despite increasing attention to multimodal safety, previous approaches based on post hoc filtering or static refusal prompts struggle to detect such latent risks, especially when harmfulness emerges only from the combination of inputs. We propose SIA (Safety via Intent Awareness), a training-free prompt engineering framework that proactively detects and mitigates harmful intent in multimodal inputs. SIA employs a three-stage reasoning process: (1) visual abstraction via captioning, (2) intent inference through few-shot chain-of-thought prompting, and (3) intent-conditioned response refinement. Rather than relying on predefined rules or classifiers, SIA dynamically adapts to the implicit intent inferred from the image-text pair. Through extensive experiments on safety-critical benchmarks including SIUO, MM-SafetyBench, and HoliSafe, we demonstrate that SIA achieves substantial safety improvements, outperforming prior methods. Although SIA shows a minor reduction in general reasoning accuracy on MMStar, the corresponding safety gains highlight the value of intent-aware reasoning in aligning VLMs with human-centric values.
>
---
#### [new 017] Pixels, Patterns, but No Poetry: To See The World like Humans
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态人工智能任务，旨在解决当前多模态大语言模型（MLLMs）在人类水平感知上的不足。论文提出了一种新的感知导向基准“图灵视觉测试”（TET），通过四项合成图像任务评估MLLM的感知能力。实验表明，现有MLLM在这些任务上表现差，而微调视觉模块可提升性能，说明问题在于视觉模块的泛化能力，而非语言推理部分。**

- **链接: [http://arxiv.org/pdf/2507.16863v1](http://arxiv.org/pdf/2507.16863v1)**

> **作者:** Hongcheng Gao; Zihao Huang; Lin Xu; Jingyi Tang; Xinhao Li; Yue Liu; Haoyang Li; Taihang Hu; Minhua Lin; Xinlong Yang; Ge Wu; Balong Bi; Hongyu Chen; Wentao Zhang
>
> **摘要:** Achieving human-like perception and reasoning in Multimodal Large Language Models (MLLMs) remains a central challenge in artificial intelligence. While recent research has primarily focused on enhancing reasoning capabilities in MLLMs, a fundamental question persists: Can Multimodal Large Language Models truly perceive the world as humans do? This paper shifts focus from reasoning to perception. Rather than constructing benchmarks specifically for reasoning, we introduce the Turing Eye Test (TET), a challenging perception-oriented benchmark comprising four diagnostic tasks that evaluate MLLMs' performance on synthetic images that humans process intuitively. Our findings reveal that state-of-the-art MLLMs exhibit catastrophic failures on our perceptual tasks trivial for humans. Both in-context learning and training on language backbone-effective for previous benchmarks-fail to improve performance on our tasks, while fine-tuning the vision tower enables rapid adaptation, suggesting that our benchmark poses challenges for vision tower generalization rather than for the knowledge and reasoning capabilities of the language backbone-a key gap between current MLLMs and human perception. We release a representative subset of TET tasks in this version, and will introduce more diverse tasks and methods to enhance visual generalization in future work.
>
---
#### [new 018] HLFormer: Enhancing Partially Relevant Video Retrieval with Hyperbolic Learning
- **分类: cs.CV; cs.IR; cs.MM**

- **简介: 该论文属于部分相关视频检索任务，旨在解决视频与文本匹配中层次结构建模不足的问题。作者提出HLFormer，首次将双曲空间学习引入PRVR，通过混合空间编码和新损失函数，提升跨模态匹配效果。实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.17402v1](http://arxiv.org/pdf/2507.17402v1)**

> **作者:** Li Jun; Wang Jinpeng; Tan Chaolei; Lian Niu; Chen Long; Zhang Min; Wang Yaowei; Xia Shu-Tao; Chen Bin
>
> **备注:** Accepted by ICCV'25. 13 pages, 6 figures, 4 tables
>
> **摘要:** Partially Relevant Video Retrieval (PRVR) addresses the critical challenge of matching untrimmed videos with text queries describing only partial content. Existing methods suffer from geometric distortion in Euclidean space that sometimes misrepresents the intrinsic hierarchical structure of videos and overlooks certain hierarchical semantics, ultimately leading to suboptimal temporal modeling. To address this issue, we propose the first hyperbolic modeling framework for PRVR, namely HLFormer, which leverages hyperbolic space learning to compensate for the suboptimal hierarchical modeling capabilities of Euclidean space. Specifically, HLFormer integrates the Lorentz Attention Block and Euclidean Attention Block to encode video embeddings in hybrid spaces, using the Mean-Guided Adaptive Interaction Module to dynamically fuse features. Additionally, we introduce a Partial Order Preservation Loss to enforce "text < video" hierarchy through Lorentzian cone constraints. This approach further enhances cross-modal matching by reinforcing partial relevance between video content and text queries. Extensive experiments show that HLFormer outperforms state-of-the-art methods. Code is released at https://github.com/lijun2005/ICCV25-HLFormer.
>
---
#### [new 019] Multi-Scale PCB Defect Detection with YOLOv8 Network Improved via Pruning and Lightweight Network
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决高密度PCB缺陷检测中准确率与计算成本难以兼顾的问题。作者改进YOLOv8网络，通过优化骨干网络、增强特征融合、设计轻量检测头与新损失函数，并结合自适应剪枝，提升了检测精度与速度。**

- **链接: [http://arxiv.org/pdf/2507.17176v1](http://arxiv.org/pdf/2507.17176v1)**

> **作者:** Li Pingzhen; Xu Sheng; Chen Jing; Su Chengyue
>
> **摘要:** With the high density of printed circuit board (PCB) design and the high speed of production, the traditional PCB defect detection model is difficult to take into account the accuracy and computational cost, and cannot meet the requirements of high accuracy and real-time detection of tiny defects. Therefore, in this paper, a multi-scale PCB defect detection method is improved with YOLOv8 using a comprehensive strategy of tiny target sensitivity strategy, network lightweighting and adaptive pruning, which is able to improve the detection speed and accuracy by optimizing the backbone network, the neck network and the detection head, the loss function and the adaptive pruning rate. Firstly, a Ghost-HGNetv2 structure with fewer parameters is used in the backbone network, and multilevel features are used to extract image semantic features to discover accurate defects. Secondly, we integrate C2f-Faster with small number of parameters in the neck section to enhance the ability of multi-level feature fusion. Next, in the Head part, we design a new GCDetect detection head, which allows the prediction of bounding boxes and categories to share the weights of GroupConv, and uses a small number of grouping convolutions to accomplish the regression and classification tasks, which significantly reduces the number of parameters while maintaining the accuracy of detection. We also design the Inner-MPDIoU boundary loss function to improve the detection and localization of tiny targets. Finally, the model was pruned by an optimized adaptive pruning rate to further reduce the complexity of the model. Experimental results show that the model exhibits advantages in terms of accuracy and speed. On the publicly available PCB defect dataset, mAP0.5 reaches 99.32% and mAP0.5:0.9 reaches 75.18%, which is 10.13% higher compared to YOLOv8n.
>
---
#### [new 020] BetterCheck: Towards Safeguarding VLMs for Automotive Perception Systems
- **分类: cs.CV; I.4.m**

- **简介: 论文提出BetterCheck，旨在检测视觉语言模型（VLMs）在自动驾驶感知系统中的幻觉问题，提升安全性。属于自动驾驶与人工智能交叉任务，解决VLMs在交通场景中误检或漏检交通参与者的问题。工作包括评估三种VLMs在真实交通数据中的表现，并提出应对幻觉的检测策略。**

- **链接: [http://arxiv.org/pdf/2507.17722v1](http://arxiv.org/pdf/2507.17722v1)**

> **作者:** Malsha Ashani Mahawatta Dona; Beatriz Cabrero-Daniel; Yinan Yu; Christian Berger
>
> **备注:** Accepted in The IEEE International Conference on Intelligent Transportation Systems (ITSC)2025
>
> **摘要:** Large language models (LLMs) are growingly extended to process multimodal data such as text and video simultaneously. Their remarkable performance in understanding what is shown in images is surpassing specialized neural networks (NNs) such as Yolo that is supporting only a well-formed but very limited vocabulary, ie., objects that they are able to detect. When being non-restricted, LLMs and in particular state-of-the-art vision language models (VLMs) show impressive performance to describe even complex traffic situations. This is making them potentially suitable components for automotive perception systems to support the understanding of complex traffic situations or edge case situation. However, LLMs and VLMs are prone to hallucination, which mean to either potentially not seeing traffic agents such as vulnerable road users who are present in a situation, or to seeing traffic agents who are not there in reality. While the latter is unwanted making an ADAS or autonomous driving systems (ADS) to unnecessarily slow down, the former could lead to disastrous decisions from an ADS. In our work, we are systematically assessing the performance of 3 state-of-the-art VLMs on a diverse subset of traffic situations sampled from the Waymo Open Dataset to support safety guardrails for capturing such hallucinations in VLM-supported perception systems. We observe that both, proprietary and open VLMs exhibit remarkable image understanding capabilities even paying thorough attention to fine details sometimes difficult to spot for us humans. However, they are also still prone to making up elements in their descriptions to date requiring hallucination detection strategies such as BetterCheck that we propose in our work.
>
---
#### [new 021] Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于事件相机与语言理解的跨模任务，旨在解决动态场景中基于语言的物体定位问题。作者构建了大规模数据集Talk2Event，并提出EventRefer模型，融合多属性信息实现更精准的感知。**

- **链接: [http://arxiv.org/pdf/2507.17664v1](http://arxiv.org/pdf/2507.17664v1)**

> **作者:** Lingdong Kong; Dongyue Lu; Ao Liang; Rong Li; Yuhao Dong; Tianshuai Hu; Lai Xing Ng; Wei Tsang Ooi; Benoit R. Cottereau
>
> **备注:** Preprint; 42 pages, 17 figures, 16 tables; Project Page at https://talk2event.github.io
>
> **摘要:** Event cameras offer microsecond-level latency and robustness to motion blur, making them ideal for understanding dynamic environments. Yet, connecting these asynchronous streams to human language remains an open challenge. We introduce Talk2Event, the first large-scale benchmark for language-driven object grounding in event-based perception. Built from real-world driving data, we provide over 30,000 validated referring expressions, each enriched with four grounding attributes -- appearance, status, relation to viewer, and relation to other objects -- bridging spatial, temporal, and relational reasoning. To fully exploit these cues, we propose EventRefer, an attribute-aware grounding framework that dynamically fuses multi-attribute representations through a Mixture of Event-Attribute Experts (MoEE). Our method adapts to different modalities and scene dynamics, achieving consistent gains over state-of-the-art baselines in event-only, frame-only, and event-frame fusion settings. We hope our dataset and approach will establish a foundation for advancing multimodal, temporally-aware, and language-driven perception in real-world robotics and autonomy.
>
---
#### [new 022] ERMV: Editing 4D Robotic Multi-view images to enhance embodied agents
- **分类: cs.CV**

- **简介: 论文提出ERMV框架，用于编辑4D多视角机器人图像以增强具身智能体。任务是数据增强，解决多视角序列图像编辑中几何一致性、计算成本和语义完整性问题。工作包括设计EMA-Attn机制、Sparse STT模块和反馈干预机制，提升VLA模型的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.17462v1](http://arxiv.org/pdf/2507.17462v1)**

> **作者:** Chang Nie; Guangming Wang; Zhe Lie; Hesheng Wang
>
> **摘要:** Robot imitation learning relies on 4D multi-view sequential images. However, the high cost of data collection and the scarcity of high-quality data severely constrain the generalization and application of embodied intelligence policies like Vision-Language-Action (VLA) models. Data augmentation is a powerful strategy to overcome data scarcity, but methods for editing 4D multi-view sequential images for manipulation tasks are currently lacking. Thus, we propose ERMV (Editing Robotic Multi-View 4D data), a novel data augmentation framework that efficiently edits an entire multi-view sequence based on single-frame editing and robot state conditions. This task presents three core challenges: (1) maintaining geometric and appearance consistency across dynamic views and long time horizons; (2) expanding the working window with low computational costs; and (3) ensuring the semantic integrity of critical objects like the robot arm. ERMV addresses these challenges through a series of innovations. First, to ensure spatio-temporal consistency in motion blur, we introduce a novel Epipolar Motion-Aware Attention (EMA-Attn) mechanism that learns pixel shift caused by movement before applying geometric constraints. Second, to maximize the editing working window, ERMV pioneers a Sparse Spatio-Temporal (STT) module, which decouples the temporal and spatial views and remodels a single-frame multi-view problem through sparse sampling of the views to reduce computational demands. Third, to alleviate error accumulation, we incorporate a feedback intervention Mechanism, which uses a Multimodal Large Language Model (MLLM) to check editing inconsistencies and request targeted expert guidance only when necessary. Extensive experiments demonstrate that ERMV-augmented data significantly boosts the robustness and generalization of VLA models in both simulated and real-world environments.
>
---
#### [new 023] Exploring Active Learning for Label-Efficient Training of Semantic Neural Radiance Field
- **分类: cs.CV**

- **简介: 该论文属于语义神经辐射场（NeRF）训练任务，旨在解决其训练所需像素级语义标注成本过高的问题。作者探索了主动学习方法，设计了多种选择策略，并提出结合3D几何约束的新策略，以减少标注成本，实验表明相比随机采样可节省2倍以上的标注量。**

- **链接: [http://arxiv.org/pdf/2507.17351v1](http://arxiv.org/pdf/2507.17351v1)**

> **作者:** Yuzhe Zhu; Lile Cai; Kangkang Lu; Fayao Liu; Xulei Yang
>
> **备注:** Accepted to ICME 2025
>
> **摘要:** Neural Radiance Field (NeRF) models are implicit neural scene representation methods that offer unprecedented capabilities in novel view synthesis. Semantically-aware NeRFs not only capture the shape and radiance of a scene, but also encode semantic information of the scene. The training of semantically-aware NeRFs typically requires pixel-level class labels, which can be prohibitively expensive to collect. In this work, we explore active learning as a potential solution to alleviate the annotation burden. We investigate various design choices for active learning of semantically-aware NeRF, including selection granularity and selection strategies. We further propose a novel active learning strategy that takes into account 3D geometric constraints in sample selection. Our experiments demonstrate that active learning can effectively reduce the annotation cost of training semantically-aware NeRF, achieving more than 2X reduction in annotation cost compared to random sampling.
>
---
#### [new 024] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection
- **分类: cs.CV**

- **简介: 该论文属于开放词汇目标检测任务，旨在解决实时检测中模型效率与性能的平衡问题。作者提出Dynamic-DINO，通过引入MoE架构和动态推理机制，在小模型上实现高效、细粒度检测。**

- **链接: [http://arxiv.org/pdf/2507.17436v1](http://arxiv.org/pdf/2507.17436v1)**

> **作者:** Yehao Lu; Minghe Weng; Zekang Xiao; Rui Jiang; Wei Su; Guangcong Zheng; Ping Lu; Xi Li
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** The Mixture of Experts (MoE) architecture has excelled in Large Vision-Language Models (LVLMs), yet its potential in real-time open-vocabulary object detectors, which also leverage large-scale vision-language datasets but smaller models, remains unexplored. This work investigates this domain, revealing intriguing insights. In the shallow layers, experts tend to cooperate with diverse peers to expand the search space. While in the deeper layers, fixed collaborative structures emerge, where each expert maintains 2-3 fixed partners and distinct expert combinations are specialized in processing specific patterns. Concretely, we propose Dynamic-DINO, which extends Grounding DINO 1.5 Edge from a dense model to a dynamic inference framework via an efficient MoE-Tuning strategy. Additionally, we design a granularity decomposition mechanism to decompose the Feed-Forward Network (FFN) of base model into multiple smaller expert networks, expanding the subnet search space. To prevent performance degradation at the start of fine-tuning, we further propose a pre-trained weight allocation strategy for the experts, coupled with a specific router initialization. During inference, only the input-relevant experts are activated to form a compact subnet. Experiments show that, pretrained with merely 1.56M open-source data, Dynamic-DINO outperforms Grounding DINO 1.5 Edge, pretrained on the private Grounding20M dataset.
>
---
#### [new 025] AURA: A Multi-Modal Medical Agent for Understanding, Reasoning & Annotation
- **分类: cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出AURA，一种多模态医疗代理，旨在通过结合视觉与语言模型提升医学图像分析的可解释性。任务是医学图像分析，解决静态预测系统缺乏交互与解释的问题。工作包括构建具备分割、反事实生成与评估功能的模块化工具箱，推动AI向透明、适应性强的临床决策支持系统发展。**

- **链接: [http://arxiv.org/pdf/2507.16940v1](http://arxiv.org/pdf/2507.16940v1)**

> **作者:** Nima Fathi; Amar Kumar; Tal Arbel
>
> **备注:** 9 pages, 3 figures, International Conference on Medical Image Computing and Computer-Assisted Intervention
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have catalyzed a paradigm shift from static prediction systems to agentic AI agents capable of reasoning, interacting with tools, and adapting to complex tasks. While LLM-based agentic systems have shown promise across many domains, their application to medical imaging remains in its infancy. In this work, we introduce AURA, the first visual linguistic explainability agent designed specifically for comprehensive analysis, explanation, and evaluation of medical images. By enabling dynamic interactions, contextual explanations, and hypothesis testing, AURA represents a significant advancement toward more transparent, adaptable, and clinically aligned AI systems. We highlight the promise of agentic AI in transforming medical image analysis from static predictions to interactive decision support. Leveraging Qwen-32B, an LLM-based architecture, AURA integrates a modular toolbox comprising: (i) a segmentation suite with phase grounding, pathology segmentation, and anatomy segmentation to localize clinically meaningful regions; (ii) a counterfactual image-generation module that supports reasoning through image-level explanations; and (iii) a set of evaluation tools including pixel-wise difference-map analysis, classification, and advanced state-of-the-art components to assess diagnostic relevance and visual interpretability.
>
---
#### [new 026] Yume: An Interactive World Generation Model
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文属于交互式世界生成任务，旨在根据图像、文本或视频创建可交互的动态世界。论文提出Yume模型，通过键盘或神经信号实现探索与控制。主要工作包括设计包含相机运动量化、视频生成架构、高级采样器和模型加速的框架，提升生成质量与交互体验。**

- **链接: [http://arxiv.org/pdf/2507.17744v1](http://arxiv.org/pdf/2507.17744v1)**

> **作者:** Xiaofeng Mao; Shaoheng Lin; Zhen Li; Chuanhao Li; Wenshuo Peng; Tong He; Jiangmiao Pang; Mingmin Chi; Yu Qiao; Kaipeng Zhang
>
> **摘要:** Yume aims to use images, text, or videos to create an interactive, realistic, and dynamic world, which allows exploration and control using peripheral devices or neural signals. In this report, we present a preview version of \method, which creates a dynamic world from an input image and allows exploration of the world using keyboard actions. To achieve this high-fidelity and interactive video world generation, we introduce a well-designed framework, which consists of four main components, including camera motion quantization, video generation architecture, advanced sampler, and model acceleration. First, we quantize camera motions for stable training and user-friendly interaction using keyboard inputs. Then, we introduce the Masked Video Diffusion Transformer~(MVDT) with a memory module for infinite video generation in an autoregressive manner. After that, training-free Anti-Artifact Mechanism (AAM) and Time Travel Sampling based on Stochastic Differential Equations (TTS-SDE) are introduced to the sampler for better visual quality and more precise control. Moreover, we investigate model acceleration by synergistic optimization of adversarial distillation and caching mechanisms. We use the high-quality world exploration dataset \sekai to train \method, and it achieves remarkable results in diverse scenes and applications. All data, codebase, and model weights are available on https://github.com/stdstu12/YUME. Yume will update monthly to achieve its original goal. Project page: https://stdstu12.github.io/YUME-Project/.
>
---
#### [new 027] VLM-Guided Visual Place Recognition for Planet-Scale Geo-Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决全球尺度下单图像地理定位问题。传统方法在可扩展性和准确性上存在不足，作者提出结合视觉语言模型（VLM）与检索式视觉地点识别（VPR）的混合框架，利用VLM生成先验信息引导检索空间，并通过重排序机制提升定位精度，取得了优于现有方法的表现。**

- **链接: [http://arxiv.org/pdf/2507.17455v1](http://arxiv.org/pdf/2507.17455v1)**

> **作者:** Sania Waheed; Na Min An; Michael Milford; Sarvapali D. Ramchurn; Shoaib Ehsan
>
> **摘要:** Geo-localization from a single image at planet scale (essentially an advanced or extreme version of the kidnapped robot problem) is a fundamental and challenging task in applications such as navigation, autonomous driving and disaster response due to the vast diversity of locations, environmental conditions, and scene variations. Traditional retrieval-based methods for geo-localization struggle with scalability and perceptual aliasing, while classification-based approaches lack generalization and require extensive training data. Recent advances in vision-language models (VLMs) offer a promising alternative by leveraging contextual understanding and reasoning. However, while VLMs achieve high accuracy, they are often prone to hallucinations and lack interpretability, making them unreliable as standalone solutions. In this work, we propose a novel hybrid geo-localization framework that combines the strengths of VLMs with retrieval-based visual place recognition (VPR) methods. Our approach first leverages a VLM to generate a prior, effectively guiding and constraining the retrieval search space. We then employ a retrieval step, followed by a re-ranking mechanism that selects the most geographically plausible matches based on feature similarity and proximity to the initially estimated coordinates. We evaluate our approach on multiple geo-localization benchmarks and show that it consistently outperforms prior state-of-the-art methods, particularly at street (up to 4.51%) and city level (up to 13.52%). Our results demonstrate that VLM-generated geographic priors in combination with VPR lead to scalable, robust, and accurate geo-localization systems.
>
---
#### [new 028] SFUOD: Source-Free Unknown Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于无源未知目标检测任务，旨在解决目标域中存在未定义类别时的检测问题。作者提出SFUOD场景及CollaPAUL框架，通过跨域注意力机制和主轴伪标签方法，实现对已知和未知目标的同步检测，提升了无源域适应下的检测性能。**

- **链接: [http://arxiv.org/pdf/2507.17373v1](http://arxiv.org/pdf/2507.17373v1)**

> **作者:** Keon-Hee Park; Seun-An Choe; Gyeong-Moon Park
>
> **备注:** This paper has been accepted by ICCV 2025
>
> **摘要:** Source-free object detection adapts a detector pre-trained on a source domain to an unlabeled target domain without requiring access to labeled source data. While this setting is practical as it eliminates the need for the source dataset during domain adaptation, it operates under the restrictive assumption that only pre-defined objects from the source domain exist in the target domain. This closed-set setting prevents the detector from detecting undefined objects. To ease this assumption, we propose Source-Free Unknown Object Detection (SFUOD), a novel scenario which enables the detector to not only recognize known objects but also detect undefined objects as unknown objects. To this end, we propose CollaPAUL (Collaborative tuning and Principal Axis-based Unknown Labeling), a novel framework for SFUOD. Collaborative tuning enhances knowledge adaptation by integrating target-dependent knowledge from the auxiliary encoder with source-dependent knowledge from the pre-trained detector through a cross-domain attention mechanism. Additionally, principal axes-based unknown labeling assigns pseudo-labels to unknown objects by estimating objectness via principal axes projection and confidence scores from model predictions. The proposed CollaPAUL achieves state-of-the-art performances on SFUOD benchmarks, and extensive experiments validate its effectiveness.
>
---
#### [new 029] Unsupervised Exposure Correction
- **分类: cs.CV**

- **简介: 该论文属于图像处理任务，旨在解决曝光校正问题。现有方法依赖人工标注、泛化性差且影响下游任务。作者提出无监督曝光校正（UEC）方法，无需人工标注，使用模拟ISP管道数据训练，并构建大规模数据集。该方法提升泛化性，保留图像细节，参数量少，且改善低层次视觉任务效果。**

- **链接: [http://arxiv.org/pdf/2507.17252v1](http://arxiv.org/pdf/2507.17252v1)**

> **作者:** Ruodai Cui; Li Niu; Guosheng Hu
>
> **摘要:** Current exposure correction methods have three challenges, labor-intensive paired data annotation, limited generalizability, and performance degradation in low-level computer vision tasks. In this work, we introduce an innovative Unsupervised Exposure Correction (UEC) method that eliminates the need for manual annotations, offers improved generalizability, and enhances performance in low-level downstream tasks. Our model is trained using freely available paired data from an emulated Image Signal Processing (ISP) pipeline. This approach does not need expensive manual annotations, thereby minimizing individual style biases from the annotation and consequently improving its generalizability. Furthermore, we present a large-scale Radiometry Correction Dataset, specifically designed to emphasize exposure variations, to facilitate unsupervised learning. In addition, we develop a transformation function that preserves image details and outperforms state-of-the-art supervised methods [12], while utilizing only 0.01% of their parameters. Our work further investigates the broader impact of exposure correction on downstream tasks, including edge detection, demonstrating its effectiveness in mitigating the adverse effects of poor exposure on low-level features. The source code and dataset are publicly available at https://github.com/BeyondHeaven/uec_code.
>
---
#### [new 030] CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的特征匹配任务，旨在提升半稠密特征匹配的精度与效率。现有方法依赖全局搜索，限制了性能。CasP通过级联对应先验机制，分阶段进行区域选择性匹配，结合高阶特征降低计算成本，并提升了跨域泛化能力，适用于高实时性与鲁棒性需求的场景如SLAM。**

- **链接: [http://arxiv.org/pdf/2507.17312v1](http://arxiv.org/pdf/2507.17312v1)**

> **作者:** Peiqi Chen; Lei Yu; Yi Wan; Yingying Pei; Xinyi Liu; Yongxiang Yao; Yingying Zhang; Lixiang Ru; Liheng Zhong; Jingdong Chen; Ming Yang; Yongjun Zhang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Semi-dense feature matching methods have shown strong performance in challenging scenarios. However, the existing pipeline relies on a global search across the entire feature map to establish coarse matches, limiting further improvements in accuracy and efficiency. Motivated by this limitation, we propose a novel pipeline, CasP, which leverages cascaded correspondence priors for guidance. Specifically, the matching stage is decomposed into two progressive phases, bridged by a region-based selective cross-attention mechanism designed to enhance feature discriminability. In the second phase, one-to-one matches are determined by restricting the search range to the one-to-many prior areas identified in the first phase. Additionally, this pipeline benefits from incorporating high-level features, which helps reduce the computational costs of low-level feature extraction. The acceleration gains of CasP increase with higher resolution, and our lite model achieves a speedup of $\sim2.2\times$ at a resolution of 1152 compared to the most efficient method, ELoFTR. Furthermore, extensive experiments demonstrate its superiority in geometric estimation, particularly with impressive cross-domain generalization. These advantages highlight its potential for latency-sensitive and high-robustness applications, such as SLAM and UAV systems. Code is available at https://github.com/pq-chen/CasP.
>
---
#### [new 031] PointLAMA: Latent Attention meets Mamba for Efficient Point Cloud Pretraining
- **分类: cs.CV**

- **简介: 该论文属于点云预训练任务，旨在解决Mamba模型缺乏局部归纳偏置、难以捕捉细粒度几何结构的问题。论文提出PointLAMA框架，结合任务感知的点云序列化、融合Latent Attention与Mamba的编码器，以及基于Mamba的条件扩散机制，提升局部建模能力与表示学习效果。**

- **链接: [http://arxiv.org/pdf/2507.17296v1](http://arxiv.org/pdf/2507.17296v1)**

> **作者:** Xuanyu Lin; Xiaona Zeng; Xianwei Zheng; Xutao Li
>
> **摘要:** Mamba has recently gained widespread attention as a backbone model for point cloud modeling, leveraging a state-space architecture that enables efficient global sequence modeling with linear complexity. However, its lack of local inductive bias limits its capacity to capture fine-grained geometric structures in 3D data. To address this limitation, we propose \textbf{PointLAMA}, a point cloud pretraining framework that combines task-aware point cloud serialization, a hybrid encoder with integrated Latent Attention and Mamba blocks, and a conditional diffusion mechanism built upon the Mamba backbone. Specifically, the task-aware point cloud serialization employs Hilbert/Trans-Hilbert space-filling curves and axis-wise sorting to structurally align point tokens for classification and segmentation tasks, respectively. Our lightweight Latent Attention block features a Point-wise Multi-head Latent Attention (PMLA) module, which is specifically designed to align with the Mamba architecture by leveraging the shared latent space characteristics of PMLA and Mamba. This enables enhanced local context modeling while preserving overall efficiency. To further enhance representation learning, we incorporate a conditional diffusion mechanism during pretraining, which denoises perturbed feature sequences without relying on explicit point-wise reconstruction. Experimental results demonstrate that PointLAMA achieves competitive performance on multiple benchmark datasets with minimal parameter count and FLOPs, validating its effectiveness for efficient point cloud pretraining.
>
---
#### [new 032] CausalStep: A Benchmark for Explicit Stepwise Causal Reasoning in Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频因果推理任务，旨在解决现有视频推理基准无法严格评估因果和逐步推理的问题。作者构建了CausalStep基准，包含100个视频和1,852个问答对，采用分步问答协议和干扰项设计，引入7项诊断指标，评估模型的因果推理能力。实验表明当前模型与人类水平仍有差距。**

- **链接: [http://arxiv.org/pdf/2507.16878v1](http://arxiv.org/pdf/2507.16878v1)**

> **作者:** Xuchen Li; Xuzhao Li; Shiyu Hu; Kaiqi Huang; Wentao Zhang
>
> **备注:** Preprint, Under review
>
> **摘要:** Recent advances in large language models (LLMs) have improved reasoning in text and image domains, yet achieving robust video reasoning remains a significant challenge. Existing video benchmarks mainly assess shallow understanding and reasoning and allow models to exploit global context, failing to rigorously evaluate true causal and stepwise reasoning. We present CausalStep, a benchmark designed for explicit stepwise causal reasoning in videos. CausalStep segments videos into causally linked units and enforces a strict stepwise question-answer (QA) protocol, requiring sequential answers and preventing shortcut solutions. Each question includes carefully constructed distractors based on error type taxonomy to ensure diagnostic value. The benchmark features 100 videos across six categories and 1,852 multiple-choice QA pairs. We introduce seven diagnostic metrics for comprehensive evaluation, enabling precise diagnosis of causal reasoning capabilities. Experiments with leading proprietary and open-source models, as well as human baselines, reveal a significant gap between current models and human-level stepwise reasoning. CausalStep provides a rigorous benchmark to drive progress in robust and interpretable video reasoning.
>
---
#### [new 033] SRMambaV2: Biomimetic Attention for Sparse Point Cloud Upsampling in Autonomous Driving
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于自动驾驶中的稀疏点云上采样任务，旨在解决激光雷达点云因稀疏性和复杂结构导致的细节重建难题。论文提出SRMambaV2方法，设计仿生注意力机制与双分支网络，并引入渐进式自适应损失函数，以提升远距离稀疏区域的上采样精度与几何重建质量。**

- **链接: [http://arxiv.org/pdf/2507.17479v1](http://arxiv.org/pdf/2507.17479v1)**

> **作者:** Chuang Chen; Xiaolin Qin; Jing Hu; Wenyi Ge
>
> **摘要:** Upsampling LiDAR point clouds in autonomous driving scenarios remains a significant challenge due to the inherent sparsity and complex 3D structures of the data. Recent studies have attempted to address this problem by converting the complex 3D spatial scenes into 2D image super-resolution tasks. However, due to the sparse and blurry feature representation of range images, accurately reconstructing detailed and complex spatial topologies remains a major difficulty. To tackle this, we propose a novel sparse point cloud upsampling method named SRMambaV2, which enhances the upsampling accuracy in long-range sparse regions while preserving the overall geometric reconstruction quality. Specifically, inspired by human driver visual perception, we design a biomimetic 2D selective scanning self-attention (2DSSA) mechanism to model the feature distribution in distant sparse areas. Meanwhile, we introduce a dual-branch network architecture to enhance the representation of sparse features. In addition, we introduce a progressive adaptive loss (PAL) function to further refine the reconstruction of fine-grained details during the upsampling process. Experimental results demonstrate that SRMambaV2 achieves superior performance in both qualitative and quantitative evaluations, highlighting its effectiveness and practical value in automotive sparse point cloud upsampling tasks.
>
---
#### [new 034] Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶中的3D目标检测任务，旨在解决多模态（LiDAR与相机）特征融合时的跨模态对齐问题。论文提出了一种基于2D检测先验的深度校正方法（PGDC），结合几何融合策略（DAGF）与特征融合模块（SGDM），有效缓解了因标定误差和运动导致的投影错位问题，提升了检测性能。**

- **链接: [http://arxiv.org/pdf/2507.16861v1](http://arxiv.org/pdf/2507.16861v1)**

> **作者:** Xiang Li
>
> **摘要:** Integrating LiDAR and camera inputs into a unified Bird's-Eye-View (BEV) representation is crucial for enhancing 3D perception capabilities of autonomous vehicles. However, current methods are often affected by misalignment between camera and LiDAR features. This misalignment leads to inaccurate depth supervision in camera branch and erroneous fusion during cross-modal feature aggregation. The root cause of this misalignment lies in projection errors, stemming from minor extrinsic calibration inaccuracies and rolling shutter effect of LiDAR during vehicle motion. In this work, our key insight is that these projection errors are predominantly concentrated at object-background boundaries, which are readily identified by 2D detectors. Based on this, our main motivation is to utilize 2D object priors to pre-align cross-modal features before fusion. To address local misalignment, we propose Prior Guided Depth Calibration (PGDC), which leverages 2D priors to correct local misalignment and preserve correct cross-modal feature pairs. To resolve global misalignment, we introduce Discontinuity Aware Geometric Fusion (DAGF) to process calibrated results from PGDC, suppressing noise and explicitly enhancing sharp transitions at object-background boundaries. To effectively utilize these transition-aware depth representations, we incorporate Structural Guidance Depth Modulator (SGDM), using a gated attention mechanism to efficiently fuse aligned depth and image features. Our proposed method achieves state-of-the-art performance on nuScenes validation dataset, with its mAP and NDS reaching 71.5% and 73.6% respectively.
>
---
#### [new 035] VisionTrap: Unanswerable Questions On Visual Data
- **分类: cs.CV**

- **简介: 该论文属于视觉问答（VQA）任务，旨在解决模型对无法回答的视觉问题是否能正确识别并 abstain 的问题。论文构建了一个新数据集 VisionTrap，包含三类基于虚构或不合理图像的不可回答问题，测试模型是否会在不应回答时仍强行作答，强调评估模型认知边界的重要性。**

- **链接: [http://arxiv.org/pdf/2507.17262v1](http://arxiv.org/pdf/2507.17262v1)**

> **作者:** Asir Saadat; Syem Aziz; Shahriar Mahmud; Abdullah Ibne Masud Mahi; Sabbir Ahmed
>
> **摘要:** Visual Question Answering (VQA) has been a widely studied topic, with extensive research focusing on how VLMs respond to answerable questions based on real-world images. However, there has been limited exploration of how these models handle unanswerable questions, particularly in cases where they should abstain from providing a response. This research investigates VQA performance on unrealistically generated images or asking unanswerable questions, assessing whether models recognize the limitations of their knowledge or attempt to generate incorrect answers. We introduced a dataset, VisionTrap, comprising three categories of unanswerable questions across diverse image types: (1) hybrid entities that fuse objects and animals, (2) objects depicted in unconventional or impossible scenarios, and (3) fictional or non-existent figures. The questions posed are logically structured yet inherently unanswerable, testing whether models can correctly recognize their limitations. Our findings highlight the importance of incorporating such questions into VQA benchmarks to evaluate whether models tend to answer, even when they should abstain.
>
---
#### [new 036] Perspective-Invariant 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D目标检测任务，旨在解决非车载平台（如四足机器人、无人机）及跨平台场景下的检测问题。作者提出了Pi3DET数据集与跨平台适应框架，实现几何与特征级对齐，提升跨平台检测性能，推动通用3D感知系统发展。**

- **链接: [http://arxiv.org/pdf/2507.17665v1](http://arxiv.org/pdf/2507.17665v1)**

> **作者:** Ao Liang; Lingdong Kong; Dongyue Lu; Youquan Liu; Jian Fang; Huaici Zhao; Wei Tsang Ooi
>
> **备注:** ICCV 2025; 46 pages, 18 figures, 22 tables; Project Page at https://pi3det.github.io
>
> **摘要:** With the rise of robotics, LiDAR-based 3D object detection has garnered significant attention in both academia and industry. However, existing datasets and methods predominantly focus on vehicle-mounted platforms, leaving other autonomous platforms underexplored. To bridge this gap, we introduce Pi3DET, the first benchmark featuring LiDAR data and 3D bounding box annotations collected from multiple platforms: vehicle, quadruped, and drone, thereby facilitating research in 3D object detection for non-vehicle platforms as well as cross-platform 3D detection. Based on Pi3DET, we propose a novel cross-platform adaptation framework that transfers knowledge from the well-studied vehicle platform to other platforms. This framework achieves perspective-invariant 3D detection through robust alignment at both geometric and feature levels. Additionally, we establish a benchmark to evaluate the resilience and robustness of current 3D detectors in cross-platform scenarios, providing valuable insights for developing adaptive 3D perception systems. Extensive experiments validate the effectiveness of our approach on challenging cross-platform tasks, demonstrating substantial gains over existing adaptation methods. We hope this work paves the way for generalizable and unified 3D perception systems across diverse and complex environments. Our Pi3DET dataset, cross-platform benchmark suite, and annotation toolkit have been made publicly available.
>
---
#### [new 037] Boosting Ray Search Procedure of Hard-label Attacks with Transfer-based Priors
- **分类: cs.CV; cs.CR; cs.LG; I.2.6; I.5.1; G.1.6**

- **简介: 该论文属于对抗攻击任务，旨在解决硬标签黑盒攻击中查询效率低的问题。通过引入基于迁移的先验知识，论文改进了梯度估计方法，提升了攻击算法的效率，并在多个数据集上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.17577v1](http://arxiv.org/pdf/2507.17577v1)**

> **作者:** Chen Ma; Xinjie Xu; Shuyu Cheng; Qi Xuan
>
> **备注:** Published at ICLR 2025 (Spotlight paper)
>
> **摘要:** One of the most practical and challenging types of black-box adversarial attacks is the hard-label attack, where only the top-1 predicted label is available. One effective approach is to search for the optimal ray direction from the benign image that minimizes the $\ell_p$-norm distance to the adversarial region. The unique advantage of this approach is that it transforms the hard-label attack into a continuous optimization problem. The objective function value is the ray's radius, which can be obtained via binary search at a high query cost. Existing methods use a "sign trick" in gradient estimation to reduce the number of queries. In this paper, we theoretically analyze the quality of this gradient estimation and propose a novel prior-guided approach to improve ray search efficiency both theoretically and empirically. Specifically, we utilize the transfer-based priors from surrogate models, and our gradient estimators appropriately integrate them by approximating the projection of the true gradient onto the subspace spanned by these priors and random directions, in a query-efficient manner. We theoretically derive the expected cosine similarities between the obtained gradient estimators and the true gradient, and demonstrate the improvement achieved by incorporating priors. Extensive experiments on the ImageNet and CIFAR-10 datasets show that our approach significantly outperforms 11 state-of-the-art methods in terms of query efficiency.
>
---
#### [new 038] Exploring Active Learning for Semiconductor Defect Segmentation
- **分类: cs.CV**

- **简介: 该论文属于半导体缺陷分割任务，旨在减少深度学习模型所需的大量标注数据。论文提出结合对比预训练和考虑类别稀有性的主动学习方法，解决领域差异和类别不平衡问题，实验证明其性能达到先进水平。**

- **链接: [http://arxiv.org/pdf/2507.17359v1](http://arxiv.org/pdf/2507.17359v1)**

> **作者:** Lile Cai; Ramanpreet Singh Pahwa; Xun Xu; Jie Wang; Richard Chang; Lining Zhang; Chuan-Sheng Foo
>
> **备注:** accepted to ICIP 2022
>
> **摘要:** The development of X-Ray microscopy (XRM) technology has enabled non-destructive inspection of semiconductor structures for defect identification. Deep learning is widely used as the state-of-the-art approach to perform visual analysis tasks. However, deep learning based models require large amount of annotated data to train. This can be time-consuming and expensive to obtain especially for dense prediction tasks like semantic segmentation. In this work, we explore active learning (AL) as a potential solution to alleviate the annotation burden. We identify two unique challenges when applying AL on semiconductor XRM scans: large domain shift and severe class-imbalance. To address these challenges, we propose to perform contrastive pretraining on the unlabelled data to obtain the initialization weights for each AL cycle, and a rareness-aware acquisition function that favors the selection of samples containing rare classes. We evaluate our method on a semiconductor dataset that is compiled from XRM scans of high bandwidth memory structures composed of logic and memory dies, and demonstrate that our method achieves state-of-the-art performance.
>
---
#### [new 039] Reusing Attention for One-stage Lane Topology Understanding
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的车道拓扑理解任务，旨在解决现有方法因误差传播和计算开销大导致的效率低下问题。作者提出了一种单阶段架构，通过复用注意力机制，在不使用额外图网络的情况下同时预测交通元素、车道中心线及其拓扑关系，提升了准确率和推理速度，并实现了从使用标准地图模型到不使用标准地图模型的知识蒸馏。**

- **链接: [http://arxiv.org/pdf/2507.17617v1](http://arxiv.org/pdf/2507.17617v1)**

> **作者:** Yang Li; Zongzheng Zhang; Xuchong Qiu; Xinrun Li; Ziming Liu; Leichen Wang; Ruikai Li; Zhenxin Zhu; Huan-ang Gao; Xiaojian Lin; Zhiyong Cui; Hang Zhao; Hao Zhao
>
> **备注:** Accepted to IROS 2025, Project Page: https://github.com/Yang-Li-2000/one-stage.git
>
> **摘要:** Understanding lane toplogy relationships accurately is critical for safe autonomous driving. However, existing two-stage methods suffer from inefficiencies due to error propagations and increased computational overheads. To address these challenges, we propose a one-stage architecture that simultaneously predicts traffic elements, lane centerlines and topology relationship, improving both the accuracy and inference speed of lane topology understanding for autonomous driving. Our key innovation lies in reusing intermediate attention resources within distinct transformer decoders. This approach effectively leverages the inherent relational knowledge within the element detection module to enable the modeling of topology relationships among traffic elements and lanes without requiring additional computationally expensive graph networks. Furthermore, we are the first to demonstrate that knowledge can be distilled from models that utilize standard definition (SD) maps to those operates without using SD maps, enabling superior performance even in the absence of SD maps. Extensive experiments on the OpenLane-V2 dataset show that our approach outperforms baseline methods in both accuracy and efficiency, achieving superior results in lane detection, traffic element identification, and topology reasoning. Our code is available at https://github.com/Yang-Li-2000/one-stage.git.
>
---
#### [new 040] A Conditional Probability Framework for Compositional Zero-shot Learning
- **分类: cs.CV**

- **简介: 该论文属于组合零样本学习（CZSL）任务，旨在解决属性与物体组合的未见类别识别问题。传统方法忽略组合内部的语义依赖，而该文提出条件概率框架（CPF），建模属性-物体依赖关系，并利用文本描述和跨注意力机制优化特征学习，提升对未见组合的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.17377v1](http://arxiv.org/pdf/2507.17377v1)**

> **作者:** Peng Wu; Qiuxia Lai; Hao Fang; Guo-Sen Xie; Yilong Yin; Xiankai Lu; Wenguan Wang
>
> **摘要:** Compositional Zero-Shot Learning (CZSL) aims to recognize unseen combinations of known objects and attributes by leveraging knowledge from previously seen compositions. Traditional approaches primarily focus on disentangling attributes and objects, treating them as independent entities during learning. However, this assumption overlooks the semantic constraints and contextual dependencies inside a composition. For example, certain attributes naturally pair with specific objects (e.g., "striped" applies to "zebra" or "shirts" but not "sky" or "water"), while the same attribute can manifest differently depending on context (e.g., "young" in "young tree" vs. "young dog"). Thus, capturing attribute-object interdependence remains a fundamental yet long-ignored challenge in CZSL. In this paper, we adopt a Conditional Probability Framework (CPF) to explicitly model attribute-object dependencies. We decompose the probability of a composition into two components: the likelihood of an object and the conditional likelihood of its attribute. To enhance object feature learning, we incorporate textual descriptors to highlight semantically relevant image regions. These enhanced object features then guide attribute learning through a cross-attention mechanism, ensuring better contextual alignment. By jointly optimizing object likelihood and conditional attribute likelihood, our method effectively captures compositional dependencies and generalizes well to unseen compositions. Extensive experiments on multiple CZSL benchmarks demonstrate the superiority of our approach. Code is available at here.
>
---
#### [new 041] Vision Transformer attention alignment with human visual perception in aesthetic object evaluation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究视觉Transformer（ViT）注意力机制与人类视觉审美关注的一致性。通过眼动实验与模型分析，比较人类与ViT在评估手工艺品时的注意力分布差异，探索其在产品设计与美学评价中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.17616v1](http://arxiv.org/pdf/2507.17616v1)**

> **作者:** Miguel Carrasco; César González-Martín; José Aranda; Luis Oliveros
>
> **备注:** 25 pages, 15 figures
>
> **摘要:** Visual attention mechanisms play a crucial role in human perception and aesthetic evaluation. Recent advances in Vision Transformers (ViTs) have demonstrated remarkable capabilities in computer vision tasks, yet their alignment with human visual attention patterns remains underexplored, particularly in aesthetic contexts. This study investigates the correlation between human visual attention and ViT attention mechanisms when evaluating handcrafted objects. We conducted an eye-tracking experiment with 30 participants (9 female, 21 male, mean age 24.6 years) who viewed 20 artisanal objects comprising basketry bags and ginger jars. Using a Pupil Labs eye-tracker, we recorded gaze patterns and generated heat maps representing human visual attention. Simultaneously, we analyzed the same objects using a pre-trained ViT model with DINO (Self-DIstillation with NO Labels), extracting attention maps from each of the 12 attention heads. We compared human and ViT attention distributions using Kullback-Leibler divergence across varying Gaussian parameters (sigma=0.1 to 3.0). Statistical analysis revealed optimal correlation at sigma=2.4 +-0.03, with attention head #12 showing the strongest alignment with human visual patterns. Significant differences were found between attention heads, with heads #7 and #9 demonstrating the greatest divergence from human attention (p< 0.05, Tukey HSD test). Results indicate that while ViTs exhibit more global attention patterns compared to human focal attention, certain attention heads can approximate human visual behavior, particularly for specific object features like buckles in basketry items. These findings suggest potential applications of ViT attention mechanisms in product design and aesthetic evaluation, while highlighting fundamental differences in attention strategies between human perception and current AI models.
>
---
#### [new 042] Toward Scalable Video Narration: A Training-free Approach Using Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决视频内容生成与时间对齐的准确叙述问题。提出VideoNarrator方法，利用多模态大模型与视觉语言模型构建无需训练的流水线，提升叙述质量、减少幻觉并优化时间对齐，适用于总结、问答及广告等场景。**

- **链接: [http://arxiv.org/pdf/2507.17050v1](http://arxiv.org/pdf/2507.17050v1)**

> **作者:** Tz-Ying Wu; Tahani Trigui; Sharath Nittur Sridhar; Anand Bodas; Subarna Tripathi
>
> **备注:** Accepted to CVAM Workshop at ICCV 2025
>
> **摘要:** In this paper, we introduce VideoNarrator, a novel training-free pipeline designed to generate dense video captions that offer a structured snapshot of video content. These captions offer detailed narrations with precise timestamps, capturing the nuances present in each segment of the video. Despite advancements in multimodal large language models (MLLMs) for video comprehension, these models often struggle with temporally aligned narrations and tend to hallucinate, particularly in unfamiliar scenarios. VideoNarrator addresses these challenges by leveraging a flexible pipeline where off-the-shelf MLLMs and visual-language models (VLMs) can function as caption generators, context providers, or caption verifiers. Our experimental results demonstrate that the synergistic interaction of these components significantly enhances the quality and accuracy of video narrations, effectively reducing hallucinations and improving temporal alignment. This structured approach not only enhances video understanding but also facilitates downstream tasks such as video summarization and video question answering, and can be potentially extended for advertising and marketing applications.
>
---
#### [new 043] Probing Vision-Language Understanding through the Visual Entailment Task: promises and pitfalls
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言理解任务，旨在评估视觉蕴含（VE）任务作为多模态语言模型理解能力探测的有效性。论文通过不同实验设置，分析VE任务在零样本、少样本和微调下的表现，探讨模型对视觉信息的依赖与局限性，以及推理过程的可解释性。**

- **链接: [http://arxiv.org/pdf/2507.17467v1](http://arxiv.org/pdf/2507.17467v1)**

> **作者:** Elena Pitta; Tom Kouwenhoven; Tessa Verhoef
>
> **备注:** LUHME: 2nd Workshop on Language Understanding in the Human-Machine Era
>
> **摘要:** This study investigates the extent to which the Visual Entailment (VE) task serves as a reliable probe of vision-language understanding in multimodal language models, using the LLaMA 3.2 11B Vision model as a test case. Beyond reporting performance metrics, we aim to interpret what these results reveal about the underlying possibilities and limitations of the VE task. We conduct a series of experiments across zero-shot, few-shot, and fine-tuning settings, exploring how factors such as prompt design, the number and order of in-context examples and access to visual information might affect VE performance. To further probe the reasoning processes of the model, we used explanation-based evaluations. Results indicate that three-shot inference outperforms the zero-shot baselines. However, additional examples introduce more noise than they provide benefits. Additionally, the order of the labels in the prompt is a critical factor that influences the predictions. In the absence of visual information, the model has a strong tendency to hallucinate and imagine content, raising questions about the model's over-reliance on linguistic priors. Fine-tuning yields strong results, achieving an accuracy of 83.3% on the e-SNLI-VE dataset and outperforming the state-of-the-art OFA-X model. Additionally, the explanation evaluation demonstrates that the fine-tuned model provides semantically meaningful explanations similar to those of humans, with a BERTScore F1-score of 89.2%. We do, however, find comparable BERTScore results in experiments with limited vision, questioning the visual grounding of this task. Overall, our results highlight both the utility and limitations of VE as a diagnostic task for vision-language understanding and point to directions for refining multimodal evaluation methods.
>
---
#### [new 044] ReMeREC: Relation-aware and Multi-entity Referring Expression Comprehension
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多实体指代表达理解任务，旨在解决现有方法在复杂场景中忽略实体间关系导致的定位不准问题。论文构建了包含关系标注的ReMeX数据集，并提出ReMeREC框架，通过TMP模块动态识别实体，结合EIR模块建模实体间关系，提升多实体定位与关系预测性能。**

- **链接: [http://arxiv.org/pdf/2507.16877v1](http://arxiv.org/pdf/2507.16877v1)**

> **作者:** Yizhi Hu; Zezhao Tian; Xingqun Qi; Chen Su; Bingkun Yang; Junhui Yin; Muyi Sun; Man Zhang; Zhenan Sun
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Referring Expression Comprehension (REC) aims to localize specified entities or regions in an image based on natural language descriptions. While existing methods handle single-entity localization, they often ignore complex inter-entity relationships in multi-entity scenes, limiting their accuracy and reliability. Additionally, the lack of high-quality datasets with fine-grained, paired image-text-relation annotations hinders further progress. To address this challenge, we first construct a relation-aware, multi-entity REC dataset called ReMeX, which includes detailed relationship and textual annotations. We then propose ReMeREC, a novel framework that jointly leverages visual and textual cues to localize multiple entities while modeling their inter-relations. To address the semantic ambiguity caused by implicit entity boundaries in language, we introduce the Text-adaptive Multi-entity Perceptron (TMP), which dynamically infers both the quantity and span of entities from fine-grained textual cues, producing distinctive representations. Additionally, our Entity Inter-relationship Reasoner (EIR) enhances relational reasoning and global scene understanding. To further improve language comprehension for fine-grained prompts, we also construct a small-scale auxiliary dataset, EntityText, generated using large language models. Experiments on four benchmark datasets show that ReMeREC achieves state-of-the-art performance in multi-entity grounding and relation prediction, outperforming existing approaches by a large margin.
>
---
#### [new 045] Principled Multimodal Representation Learning
- **分类: cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于多模态表示学习任务，旨在解决传统方法依赖预设锚点和难以稳定对齐多模态数据的问题。作者提出PMRL框架，通过优化表示矩阵的主奇异值和引入对比正则化，实现多模态同时对齐并保持实例区分性，提升了多模态理解效果。**

- **链接: [http://arxiv.org/pdf/2507.17343v1](http://arxiv.org/pdf/2507.17343v1)**

> **作者:** Xiaohao Liu; Xiaobo Xia; See-Kiong Ng; Tat-Seng Chua
>
> **备注:** 32 pages, 9 figures, 10 tables
>
> **摘要:** Multimodal representation learning seeks to create a unified representation space by integrating diverse data modalities to improve multimodal understanding. Traditional methods often depend on pairwise contrastive learning, which relies on a predefined anchor modality, restricting alignment across all modalities. Recent advances have investigated the simultaneous alignment of multiple modalities, yet several challenges remain, such as limitations imposed by fixed anchor points and instability arising from optimizing the product of singular values. To address the challenges, in this paper, we propose Principled Multimodal Representation Learning (PMRL), a novel framework that achieves simultaneous alignment of multiple modalities without anchor dependency in a more stable manner. Specifically, grounded in the theoretical insight that full alignment corresponds to a rank-1 Gram matrix, PMRL optimizes the dominant singular value of the representation matrix to align modalities along a shared leading direction. We propose a softmax-based loss function that treats singular values as logits to prioritize the largest singular value. Besides, instance-wise contrastive regularization on the leading eigenvectors maintains inter-instance separability and prevents representation collapse. Extensive experiments across diverse tasks demonstrate PMRL's superiority compared to baseline methods. The source code will be publicly available.
>
---
#### [new 046] CLAMP: Contrastive Learning with Adaptive Multi-loss and Progressive Fusion for Multimodal Aspect-Based Sentiment Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态细粒度情感分析任务，旨在解决跨模态对齐噪声和细粒度表示不一致问题。作者提出了CLAMP框架，包含渐进注意力融合、多任务对比学习和自适应多损失聚合模块，提升文本与图像的对齐效果，优化情感分析性能。**

- **链接: [http://arxiv.org/pdf/2507.16854v1](http://arxiv.org/pdf/2507.16854v1)**

> **作者:** Xiaoqiang He
>
> **摘要:** Multimodal aspect-based sentiment analysis(MABSA) seeks to identify aspect terms within paired image-text data and determine their fine grained sentiment polarities, representing a fundamental task for improving the effectiveness of applications such as product review systems and public opinion monitoring. Existing methods face challenges such as cross modal alignment noise and insufficient consistency in fine-grained representations. While global modality alignment methods often overlook the connection between aspect terms and their corresponding local visual regions, bridging the representation gap between text and images remains a challenge. To address these limitations, this paper introduces an end to end Contrastive Learning framework with Adaptive Multi-loss and Progressive Attention Fusion(CLAMP). The framework is composed of three novel modules: Progressive Attention Fusion network, Multi-task Contrastive Learning, and Adaptive Multi-loss Aggregation. The Progressive Attention Fusion network enhances fine-grained alignment between textual features and image regions via hierarchical, multi-stage cross modal interactions, effectively suppressing irrelevant visual noise. Secondly, multi-task contrastive learning combines global modal contrast and local granularity alignment to enhance cross modal representation consistency. Adaptive Multi-loss Aggregation employs a dynamic uncertainty based weighting mechanism to calibrate loss contributions according to each task's uncertainty, thereby mitigating gradient interference. Evaluation on standard public benchmarks demonstrates that CLAMP consistently outperforms the vast majority of existing state of the art methods.
>
---
#### [new 047] CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17651v1](http://arxiv.org/pdf/2507.17651v1)**

> **作者:** Olaf Dünkel; Artur Jesslen; Jiahao Xie; Christian Theobalt; Christian Rupprecht; Adam Kortylewski
>
> **备注:** ICCV 2025. Project page: https://genintel.github.io/CNS
>
> **摘要:** An important challenge when using computer vision models in the real world is to evaluate their performance in potential out-of-distribution (OOD) scenarios. While simple synthetic corruptions are commonly applied to test OOD robustness, they often fail to capture nuisance shifts that occur in the real world. Recently, diffusion models have been applied to generate realistic images for benchmarking, but they are restricted to binary nuisance shifts. In this work, we introduce CNS-Bench, a Continuous Nuisance Shift Benchmark to quantify OOD robustness of image classifiers for continuous and realistic generative nuisance shifts. CNS-Bench allows generating a wide range of individual nuisance shifts in continuous severities by applying LoRA adapters to diffusion models. To address failure cases, we propose a filtering mechanism that outperforms previous methods, thereby enabling reliable benchmarking with generative models. With the proposed benchmark, we perform a large-scale study to evaluate the robustness of more than 40 classifiers under various nuisance shifts. Through carefully designed comparisons and analyses, we find that model rankings can change for varying shifts and shift scales, which cannot be captured when applying common binary shifts. Additionally, we show that evaluating the model performance on a continuous scale allows the identification of model failure points, providing a more nuanced understanding of model robustness. Project page including code and data: https://genintel.github.io/CNS.
>
---
#### [new 048] PARTE: Part-Guided Texturing for 3D Human Reconstruction from a Single Image
- **分类: cs.CV**

- **简介: 该论文属于3D人体重建任务，旨在解决单张图像中人体各部分纹理错位的问题。通过提出PARTE框架，利用3D人体部分信息作为指导，包含PartSegmenter和PartTexturer模块，提升纹理重建质量，实现高质量的3D人体重建。**

- **链接: [http://arxiv.org/pdf/2507.17332v1](http://arxiv.org/pdf/2507.17332v1)**

> **作者:** Hyeongjin Nam; Donghwan Kim; Gyeongsik Moon; Kyoung Mu Lee
>
> **备注:** Published at ICCV 2025, 22 pages including the supplementary material
>
> **摘要:** The misaligned human texture across different human parts is one of the main limitations of existing 3D human reconstruction methods. Each human part, such as a jacket or pants, should maintain a distinct texture without blending into others. The structural coherence of human parts serves as a crucial cue to infer human textures in the invisible regions of a single image. However, most existing 3D human reconstruction methods do not explicitly exploit such part segmentation priors, leading to misaligned textures in their reconstructions. In this regard, we present PARTE, which utilizes 3D human part information as a key guide to reconstruct 3D human textures. Our framework comprises two core components. First, to infer 3D human part information from a single image, we propose a 3D part segmentation module (PartSegmenter) that initially reconstructs a textureless human surface and predicts human part labels based on the textureless surface. Second, to incorporate part information into texture reconstruction, we introduce a part-guided texturing module (PartTexturer), which acquires prior knowledge from a pre-trained image generation network on texture alignment of human parts. Extensive experiments demonstrate that our framework achieves state-of-the-art quality in 3D human reconstruction. The project page is available at https://hygenie1228.github.io/PARTE/.
>
---
#### [new 049] Monocular Semantic Scene Completion via Masked Recurrent Networks
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于计算机视觉任务，旨在解决单目语义场景补全问题。通过提出一种新的两阶段框架MonoMRN，结合Masked Sparse Gated Recurrent Unit和距离注意力投影，提升复杂场景中可见和遮挡区域的预测性能，并增强模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.17661v1](http://arxiv.org/pdf/2507.17661v1)**

> **作者:** Xuzhi Wang; Xinran Wu; Song Wang; Lingdong Kong; Ziping Zhao
>
> **备注:** ICCV 2025; 15 pages, 10 figures, 6 tables; Code at https://github.com/alanWXZ/MonoMRN
>
> **摘要:** Monocular Semantic Scene Completion (MSSC) aims to predict the voxel-wise occupancy and semantic category from a single-view RGB image. Existing methods adopt a single-stage framework that aims to simultaneously achieve visible region segmentation and occluded region hallucination, while also being affected by inaccurate depth estimation. Such methods often achieve suboptimal performance, especially in complex scenes. We propose a novel two-stage framework that decomposes MSSC into coarse MSSC followed by the Masked Recurrent Network. Specifically, we propose the Masked Sparse Gated Recurrent Unit (MS-GRU) which concentrates on the occupied regions by the proposed mask updating mechanism, and a sparse GRU design is proposed to reduce the computation cost. Additionally, we propose the distance attention projection to reduce projection errors by assigning different attention scores according to the distance to the observed surface. Experimental results demonstrate that our proposed unified framework, MonoMRN, effectively supports both indoor and outdoor scenes and achieves state-of-the-art performance on the NYUv2 and SemanticKITTI datasets. Furthermore, we conduct robustness analysis under various disturbances, highlighting the role of the Masked Recurrent Network in enhancing the model's resilience to such challenges. The source code is publicly available.
>
---
#### [new 050] STQE: Spatial-Temporal Quality Enhancement for G-PCC Compressed Dynamic Point Clouds
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于动态点云压缩质量提升任务，旨在解决G-PCC压缩导致的视觉质量下降问题。论文提出STQE网络，结合时空相关性，通过运动补偿、时间注意力、空间特征聚合等模块提升压缩点云的属性质量，并设计联合损失函数缓解过拟合问题。**

- **链接: [http://arxiv.org/pdf/2507.17522v1](http://arxiv.org/pdf/2507.17522v1)**

> **作者:** Tian Guo; Hui Yuan; Xiaolong Mao; Shiqi Jiang; Raouf Hamzaoui; Sam Kwong
>
> **摘要:** Very few studies have addressed quality enhancement for compressed dynamic point clouds. In particular, the effective exploitation of spatial-temporal correlations between point cloud frames remains largely unexplored. Addressing this gap, we propose a spatial-temporal attribute quality enhancement (STQE) network that exploits both spatial and temporal correlations to improve the visual quality of G-PCC compressed dynamic point clouds. Our contributions include a recoloring-based motion compensation module that remaps reference attribute information to the current frame geometry to achieve precise inter-frame geometric alignment, a channel-aware temporal attention module that dynamically highlights relevant regions across bidirectional reference frames, a Gaussian-guided neighborhood feature aggregation module that efficiently captures spatial dependencies between geometry and color attributes, and a joint loss function based on the Pearson correlation coefficient, designed to alleviate over-smoothing effects typical of point-wise mean squared error optimization. When applied to the latest G-PCC test model, STQE achieved improvements of 0.855 dB, 0.682 dB, and 0.828 dB in delta PSNR, with Bj{\o}ntegaard Delta rate (BD-rate) reductions of -25.2%, -31.6%, and -32.5% for the Luma, Cb, and Cr components, respectively.
>
---
#### [new 051] The Early Bird Identifies the Worm: You Can't Beat a Head Start in Long-Term Body Re-ID (ECHO-BID)
- **分类: cs.CV**

- **简介: 该论文属于人体重识别任务，旨在解决长期视角下因衣物变化、遮挡等导致的身份识别难题。作者提出ECHO-BID模型，基于EVA-02 Large预训练架构，通过迁移学习优化，在多种复杂场景中实现了最优性能，尤其在遮挡和衣物变化情况下效果显著提升。**

- **链接: [http://arxiv.org/pdf/2507.17640v1](http://arxiv.org/pdf/2507.17640v1)**

> **作者:** Thomas M. Metz; Matthew Q. Hill; Alice J. O'Toole
>
> **摘要:** Person identification in unconstrained viewing environments presents significant challenges due to variations in distance, viewpoint, imaging conditions, and clothing. We introduce $\textbf{E}$va $\textbf{C}$lothes-Change from $\textbf{H}$idden $\textbf{O}$bjects - $\textbf{B}$ody $\textbf{ID}$entification (ECHO-BID), a class of long-term re-id models built on object-pretrained EVA-02 Large backbones. We compare ECHO-BID to 9 other models that vary systematically in backbone architecture, model size, scale of object classification pretraining, and transfer learning protocol. Models were evaluated on benchmark datasets across constrained, unconstrained, and occluded settings. ECHO-BID, with transfer learning on the most challenging clothes-change data, achieved state-of-the-art results on long-term re-id -- substantially outperforming other methods. ECHO-BID also surpassed other methods by a wide margin in occluded viewing scenarios. A combination of increased model size and Masked Image Modeling during pretraining underlie ECHO-BID's strong performance on long-term re-id. Notably, a smaller, but more challenging transfer learning dataset, generalized better across datasets than a larger, less challenging one. However, the larger dataset with an additional fine-tuning step proved best on the most difficult data. Selecting the correct pretrained backbone architecture and transfer learning protocols can drive substantial gains in long-term re-id performance.
>
---
#### [new 052] PIG-Nav: Key Insights for Pretrained Image Goal Navigation Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉导航任务，旨在提升预训练图像目标导航模型的泛化能力和零样本性能。论文提出了PIG-Nav，通过改进模型结构和引入辅助任务提升导航表现，并设计了新的数据预处理流程。在多个环境中实现了性能提升，同时减少了对标注数据的依赖。**

- **链接: [http://arxiv.org/pdf/2507.17220v1](http://arxiv.org/pdf/2507.17220v1)**

> **作者:** Jiansong Wan; Chengming Zhou; Jinkua Liu; Xiangge Huang; Xiaoyu Chen; Xiaohan Yi; Qisen Yang; Baiting Zhu; Xin-Qiang Cai; Lixing Liu; Rushuai Yang; Chuheng Zhang; Sherif Abdelfattah; Hayong Shin; Pushi Zhang; Li Zhao; Jiang Bian
>
> **摘要:** Recent studies have explored pretrained (foundation) models for vision-based robotic navigation, aiming to achieve generalizable navigation and positive transfer across diverse environments while enhancing zero-shot performance in unseen settings. In this work, we introduce PIG-Nav (Pretrained Image-Goal Navigation), a new approach that further investigates pretraining strategies for vision-based navigation models and contributes in two key areas. Model-wise, we identify two critical design choices that consistently improve the performance of pretrained navigation models: (1) integrating an early-fusion network structure to combine visual observations and goal images via appropriately pretrained Vision Transformer (ViT) image encoder, and (2) introducing suitable auxiliary tasks to enhance global navigation representation learning, thus further improving navigation performance. Dataset-wise, we propose a novel data preprocessing pipeline for efficiently labeling large-scale game video datasets for navigation model training. We demonstrate that augmenting existing open navigation datasets with diverse gameplay videos improves model performance. Our model achieves an average improvement of 22.6% in zero-shot settings and a 37.5% improvement in fine-tuning settings over existing visual navigation foundation models in two complex simulated environments and one real-world environment. These results advance the state-of-the-art in pretrained image-goal navigation models. Notably, our model maintains competitive performance while requiring significantly less fine-tuning data, highlighting its potential for real-world deployment with minimal labeled supervision.
>
---
#### [new 053] Bringing Balance to Hand Shape Classification: Mitigating Data Imbalance Through Generative Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于手形分类任务，旨在解决手语数据集小且类别不平衡导致模型训练效果差的问题。作者使用生成对抗网络（GAN）生成合成数据进行数据增强，比较了ReACGAN和SPADE两种方法，并提升了分类准确率5%。**

- **链接: [http://arxiv.org/pdf/2507.17008v1](http://arxiv.org/pdf/2507.17008v1)**

> **作者:** Gaston Gustavo Rios; Pedro Dal Bianco; Franco Ronchetti; Facundo Quiroga; Oscar Stanchi; Santiago Ponte Ahón; Waldo Hasperué
>
> **备注:** 23 pages, 8 figures, to be published in Applied Soft Computing
>
> **摘要:** Most sign language handshape datasets are severely limited and unbalanced, posing significant challenges to effective model training. In this paper, we explore the effectiveness of augmenting the training data of a handshape classifier by generating synthetic data. We use an EfficientNet classifier trained on the RWTH German sign language handshape dataset, which is small and heavily unbalanced, applying different strategies to combine generated and real images. We compare two Generative Adversarial Networks (GAN) architectures for data generation: ReACGAN, which uses label information to condition the data generation process through an auxiliary classifier, and SPADE, which utilizes spatially-adaptive normalization to condition the generation on pose information. ReACGAN allows for the generation of realistic images that align with specific handshape labels, while SPADE focuses on generating images with accurate spatial handshape configurations. Our proposed techniques improve the current state-of-the-art accuracy on the RWTH dataset by 5%, addressing the limitations of small and unbalanced datasets. Additionally, our method demonstrates the capability to generalize across different sign language datasets by leveraging pose-based generation trained on the extensive HaGRID dataset. We achieve comparable performance to single-source trained classifiers without the need for retraining the generator.
>
---
#### [new 054] Multi-modal Multi-task Pre-training for Improved Point Cloud Understanding
- **分类: cs.CV**

- **简介: 该论文属于点云理解任务，旨在解决现有方法依赖单一预训练任务导致信息利用不足的问题。作者提出了MMPT框架，结合三种预训练任务：token级重建、点级重建和多模态对比学习，提升点云表征能力，且无需3D标注，适用于大规模数据。**

- **链接: [http://arxiv.org/pdf/2507.17533v1](http://arxiv.org/pdf/2507.17533v1)**

> **作者:** Liwen Liu; Weidong Yang; Lipeng Ma; Ben Fei
>
> **摘要:** Recent advances in multi-modal pre-training methods have shown promising effectiveness in learning 3D representations by aligning multi-modal features between 3D shapes and their corresponding 2D counterparts. However, existing multi-modal pre-training frameworks primarily rely on a single pre-training task to gather multi-modal data in 3D applications. This limitation prevents the models from obtaining the abundant information provided by other relevant tasks, which can hinder their performance in downstream tasks, particularly in complex and diverse domains. In order to tackle this issue, we propose MMPT, a Multi-modal Multi-task Pre-training framework designed to enhance point cloud understanding. Specifically, three pre-training tasks are devised: (i) Token-level reconstruction (TLR) aims to recover masked point tokens, endowing the model with representative learning abilities. (ii) Point-level reconstruction (PLR) is integrated to predict the masked point positions directly, and the reconstructed point cloud can be considered as a transformed point cloud used in the subsequent task. (iii) Multi-modal contrastive learning (MCL) combines feature correspondences within and across modalities, thus assembling a rich learning signal from both 3D point cloud and 2D image modalities in a self-supervised manner. Moreover, this framework operates without requiring any 3D annotations, making it scalable for use with large datasets. The trained encoder can be effectively transferred to various downstream tasks. To demonstrate its effectiveness, we evaluated its performance compared to state-of-the-art methods in various discriminant and generative applications under widely-used benchmarks.
>
---
#### [new 055] IONext: Unlocking the Next Era of Inertial Odometry
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于惯性里程计任务，旨在提升定位精度与泛化能力。针对现有Transformer模型对局部运动敏感度低、CNN模型时序建模不足的问题，论文提出了IONext网络，结合Dual-wing Adaptive Dynamic Mixer和Spatio-Temporal Gating Unit模块，实现更优的全局与局部运动特征融合及时序建模。**

- **链接: [http://arxiv.org/pdf/2507.17089v1](http://arxiv.org/pdf/2507.17089v1)**

> **作者:** Shanshan Zhang; Siyue Wang; Tianshui Wen; Qi Zhang; Ziheng Zhou; Lingxiang Zheng; Yu Yang
>
> **摘要:** Researchers have increasingly adopted Transformer-based models for inertial odometry. While Transformers excel at modeling long-range dependencies, their limited sensitivity to local, fine-grained motion variations and lack of inherent inductive biases often hinder localization accuracy and generalization. Recent studies have shown that incorporating large-kernel convolutions and Transformer-inspired architectural designs into CNN can effectively expand the receptive field, thereby improving global motion perception. Motivated by these insights, we propose a novel CNN-based module called the Dual-wing Adaptive Dynamic Mixer (DADM), which adaptively captures both global motion patterns and local, fine-grained motion features from dynamic inputs. This module dynamically generates selective weights based on the input, enabling efficient multi-scale feature aggregation. To further improve temporal modeling, we introduce the Spatio-Temporal Gating Unit (STGU), which selectively extracts representative and task-relevant motion features in the temporal domain. This unit addresses the limitations of temporal modeling observed in existing CNN approaches. Built upon DADM and STGU, we present a new CNN-based inertial odometry backbone, named Next Era of Inertial Odometry (IONext). Extensive experiments on six public datasets demonstrate that IONext consistently outperforms state-of-the-art (SOTA) Transformer- and CNN-based methods. For instance, on the RNIN dataset, IONext reduces the average ATE by 10% and the average RTE by 12% compared to the representative model iMOT.
>
---
#### [new 056] HiProbe-VAD: Video Anomaly Detection via Hidden States Probing in Tuning-Free Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频异常检测任务，旨在发现视频中偏离正常模式的行为。现有方法计算量大且依赖大量标注数据。论文提出HiProbe-VAD，利用预训练多模态大语言模型的中间隐藏状态，通过动态层显著性探测机制提取敏感特征，实现无需微调的异常检测与解释。实验表明其性能优于现有方法，并具备跨模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.17394v1](http://arxiv.org/pdf/2507.17394v1)**

> **作者:** Zhaolin Cai; Fan Li; Ziwei Zheng; Yanjun Qin
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Video Anomaly Detection (VAD) aims to identify and locate deviations from normal patterns in video sequences. Traditional methods often struggle with substantial computational demands and a reliance on extensive labeled datasets, thereby restricting their practical applicability. To address these constraints, we propose HiProbe-VAD, a novel framework that leverages pre-trained Multimodal Large Language Models (MLLMs) for VAD without requiring fine-tuning. In this paper, we discover that the intermediate hidden states of MLLMs contain information-rich representations, exhibiting higher sensitivity and linear separability for anomalies compared to the output layer. To capitalize on this, we propose a Dynamic Layer Saliency Probing (DLSP) mechanism that intelligently identifies and extracts the most informative hidden states from the optimal intermediate layer during the MLLMs reasoning. Then a lightweight anomaly scorer and temporal localization module efficiently detects anomalies using these extracted hidden states and finally generate explanations. Experiments on the UCF-Crime and XD-Violence datasets demonstrate that HiProbe-VAD outperforms existing training-free and most traditional approaches. Furthermore, our framework exhibits remarkable cross-model generalization capabilities in different MLLMs without any tuning, unlocking the potential of pre-trained MLLMs for video anomaly detection and paving the way for more practical and scalable solutions.
>
---
#### [new 057] Divisive Decisions: Improving Salience-Based Training for Generalization in Binary Classification Tasks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于二分类任务，旨在提升模型泛化能力。现有方法仅利用真实类别激活图（CAM）与人类显著图对比训练，忽略错误类CAM。论文提出三种新方法，结合真实与错误类CAM，并设计新工具识别关键特征，验证了其在多个二分类任务上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.17000v1](http://arxiv.org/pdf/2507.17000v1)**

> **作者:** Jacob Piland; Chris Sweet; Adam Czajka
>
> **摘要:** Existing saliency-guided training approaches improve model generalization by incorporating a loss term that compares the model's class activation map (CAM) for a sample's true-class ({\it i.e.}, correct-label class) against a human reference saliency map. However, prior work has ignored the false-class CAM(s), that is the model's saliency obtained for incorrect-label class. We hypothesize that in binary tasks the true and false CAMs should diverge on the important classification features identified by humans (and reflected in human saliency maps). We use this hypothesis to motivate three new saliency-guided training methods incorporating both true- and false-class model's CAM into the training strategy and a novel post-hoc tool for identifying important features. We evaluate all introduced methods on several diverse binary close-set and open-set classification tasks, including synthetic face detection, biometric presentation attack detection, and classification of anomalies in chest X-ray scans, and find that the proposed methods improve generalization capabilities of deep learning models over traditional (true-class CAM only) saliency-guided training approaches. We offer source codes and model weights\footnote{GitHub repository link removed to preserve anonymity} to support reproducible research.
>
---
#### [new 058] Content-based 3D Image Retrieval and a ColBERT-inspired Re-ranking for Tumor Flagging and Staging
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文属于医学图像检索任务，旨在解决放射科医生在海量影像中快速获取相似病例的问题。论文提出了一种无需预分割数据的3D图像检索框架，并引入C-MIR方法，借鉴ColBERT的交互机制实现有效的三维医学图像重排序，提升了肿瘤标记与分期的效果。**

- **链接: [http://arxiv.org/pdf/2507.17412v1](http://arxiv.org/pdf/2507.17412v1)**

> **作者:** Farnaz Khun Jush; Steffen Vogler; Matthias Lenga
>
> **摘要:** The increasing volume of medical images poses challenges for radiologists in retrieving relevant cases. Content-based image retrieval (CBIR) systems offer potential for efficient access to similar cases, yet lack standardized evaluation and comprehensive studies. Building on prior studies for tumor characterization via CBIR, this study advances CBIR research for volumetric medical images through three key contributions: (1) a framework eliminating reliance on pre-segmented data and organ-specific datasets, aligning with large and unstructured image archiving systems, i.e. PACS in clinical practice; (2) introduction of C-MIR, a novel volumetric re-ranking method adapting ColBERT's contextualized late interaction mechanism for 3D medical imaging; (3) comprehensive evaluation across four tumor sites using three feature extractors and three database configurations. Our evaluations highlight the significant advantages of C-MIR. We demonstrate the successful adaptation of the late interaction principle to volumetric medical images, enabling effective context-aware re-ranking. A key finding is C-MIR's ability to effectively localize the region of interest, eliminating the need for pre-segmentation of datasets and offering a computationally efficient alternative to systems relying on expensive data enrichment steps. C-MIR demonstrates promising improvements in tumor flagging, achieving improved performance, particularly for colon and lung tumors (p<0.05). C-MIR also shows potential for improving tumor staging, warranting further exploration of its capabilities. Ultimately, our work seeks to bridge the gap between advanced retrieval techniques and their practical applications in healthcare, paving the way for improved diagnostic processes.
>
---
#### [new 059] DFDNet: Dynamic Frequency-Guided De-Flare Network
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像去雾任务，旨在解决夜间摄影中强光源导致的镜头光晕问题。现有方法难以处理大规模光晕及修复光晕造成的结构损伤。论文提出DFDNet网络，通过动态频率域引导和局部细节学习，分离光晕与内容信息，有效去除光晕并恢复细节。**

- **链接: [http://arxiv.org/pdf/2507.17489v1](http://arxiv.org/pdf/2507.17489v1)**

> **作者:** Minglong Xue; Aoxiang Ning; Shivakumara Palaiahnakote; Mingliang Zhou
>
> **摘要:** Strong light sources in nighttime photography frequently produce flares in images, significantly degrading visual quality and impacting the performance of downstream tasks. While some progress has been made, existing methods continue to struggle with removing large-scale flare artifacts and repairing structural damage in regions near the light source. We observe that these challenging flare artifacts exhibit more significant discrepancies from the reference images in the frequency domain compared to the spatial domain. Therefore, this paper presents a novel dynamic frequency-guided deflare network (DFDNet) that decouples content information from flare artifacts in the frequency domain, effectively removing large-scale flare artifacts. Specifically, DFDNet consists mainly of a global dynamic frequency-domain guidance (GDFG) module and a local detail guidance module (LDGM). The GDFG module guides the network to perceive the frequency characteristics of flare artifacts by dynamically optimizing global frequency domain features, effectively separating flare information from content information. Additionally, we design an LDGM via a contrastive learning strategy that aligns the local features of the light source with the reference image, reduces local detail damage from flare removal, and improves fine-grained image restoration. The experimental results demonstrate that the proposed method outperforms existing state-of-the-art methods in terms of performance. The code is available at \href{https://github.com/AXNing/DFDNet}{https://github.com/AXNing/DFDNet}.
>
---
#### [new 060] Hierarchical Fusion and Joint Aggregation: A Multi-Level Feature Representation Method for AIGC Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，旨在解决AI生成图像（AIGC）质量评估中低层次视觉特征与高层次语义理解之间的多维挑战。现有方法依赖单一视觉特征，难以捕捉复杂失真。论文提出一种多层次视觉表征方法，包括多级特征提取、层次融合与联合聚合，并设计了两个网络MGLF-Net和MPEF-Net，分别用于感知质量评估与图文对应性评估，实验表明该方法在相关任务上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.17182v1](http://arxiv.org/pdf/2507.17182v1)**

> **作者:** Linghe Meng; Jiarun Song
>
> **摘要:** The quality assessment of AI-generated content (AIGC) faces multi-dimensional challenges, that span from low-level visual perception to high-level semantic understanding. Existing methods generally rely on single-level visual features, limiting their ability to capture complex distortions in AIGC images. To address this limitation, a multi-level visual representation paradigm is proposed with three stages, namely multi-level feature extraction, hierarchical fusion, and joint aggregation. Based on this paradigm, two networks are developed. Specifically, the Multi-Level Global-Local Fusion Network (MGLF-Net) is designed for the perceptual quality assessment, extracting complementary local and global features via dual CNN and Transformer visual backbones. The Multi-Level Prompt-Embedded Fusion Network (MPEF-Net) targets Text-to-Image correspondence by embedding prompt semantics into the visual feature fusion process at each feature level. The fused multi-level features are then aggregated for final evaluation. Experiments on benchmarks demonstrate outstanding performance on both tasks, validating the effectiveness of the proposed multi-level visual assessment paradigm.
>
---
#### [new 061] EndoGen: Conditional Autoregressive Endoscopic Video Generation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于医学图像生成任务，旨在解决现有内镜视频生成方法缺乏动态上下文和条件引导的问题。作者提出了EndoGen框架，采用自回归模型结合时空网格-帧模式策略和语义感知令牌掩码机制，实现高质量、有条件引导的内镜视频生成，并提升了息肉分割效果。**

- **链接: [http://arxiv.org/pdf/2507.17388v1](http://arxiv.org/pdf/2507.17388v1)**

> **作者:** Xinyu Liu; Hengyu Liu; Cheng Wang; Tianming Liu; Yixuan Yuan
>
> **备注:** MICCAI 2025
>
> **摘要:** Endoscopic video generation is crucial for advancing medical imaging and enhancing diagnostic capabilities. However, prior efforts in this field have either focused on static images, lacking the dynamic context required for practical applications, or have relied on unconditional generation that fails to provide meaningful references for clinicians. Therefore, in this paper, we propose the first conditional endoscopic video generation framework, namely EndoGen. Specifically, we build an autoregressive model with a tailored Spatiotemporal Grid-Frame Patterning (SGP) strategy. It reformulates the learning of generating multiple frames as a grid-based image generation pattern, which effectively capitalizes the inherent global dependency modeling capabilities of autoregressive architectures. Furthermore, we propose a Semantic-Aware Token Masking (SAT) mechanism, which enhances the model's ability to produce rich and diverse content by selectively focusing on semantically meaningful regions during the generation process. Through extensive experiments, we demonstrate the effectiveness of our framework in generating high-quality, conditionally guided endoscopic content, and improves the performance of downstream task of polyp segmentation. Code released at https://www.github.com/CUHK-AIM-Group/EndoGen.
>
---
#### [new 062] Exploring Spatial Diversity for Region-based Active Learning
- **分类: cs.CV**

- **简介: 该论文属于语义分割任务，旨在通过区域主动学习减少标注成本。它提出了一种结合空间多样性与传统选择标准的统一框架，提升了主动学习效果，在Cityscapes和PASCAL VOC数据集上仅用5-9%的标注像素即达到全监督方法95%的性能。**

- **链接: [http://arxiv.org/pdf/2507.17367v1](http://arxiv.org/pdf/2507.17367v1)**

> **作者:** Lile Cai; Xun Xu; Lining Zhang; Chuan-Sheng Foo
>
> **备注:** published in IEEE Transactions on Image Processing, 2021
>
> **摘要:** State-of-the-art methods for semantic segmentation are based on deep neural networks trained on large-scale labeled datasets. Acquiring such datasets would incur large annotation costs, especially for dense pixel-level prediction tasks like semantic segmentation. We consider region-based active learning as a strategy to reduce annotation costs while maintaining high performance. In this setting, batches of informative image regions instead of entire images are selected for labeling. Importantly, we propose that enforcing local spatial diversity is beneficial for active learning in this case, and to incorporate spatial diversity along with the traditional active selection criterion, e.g., data sample uncertainty, in a unified optimization framework for region-based active learning. We apply this framework to the Cityscapes and PASCAL VOC datasets and demonstrate that the inclusion of spatial diversity effectively improves the performance of uncertainty-based and feature diversity-based active learning methods. Our framework achieves $95\%$ performance of fully supervised methods with only $5-9\%$ of the labeled pixels, outperforming all state-of-the-art region-based active learning methods for semantic segmentation.
>
---
#### [new 063] Dynamic Scoring with Enhanced Semantics for Training-Free Human-Object Interaction Detection
- **分类: cs.CV**

- **简介: 该论文属于人-物交互（HOI）检测任务，旨在解决依赖大量人工标注数据的问题。作者提出了一种无需训练的框架DYSCO，利用视觉-语言模型（VLMs）提升交互表示，通过多模态注册表和多头注意力机制，实现对罕见交互的有效识别。**

- **链接: [http://arxiv.org/pdf/2507.17456v1](http://arxiv.org/pdf/2507.17456v1)**

> **作者:** Francesco Tonini; Lorenzo Vaquero; Alessandro Conti; Cigdem Beyan; Elisa Ricci
>
> **备注:** Accepted to ACM Multimedia 2025
>
> **摘要:** Human-Object Interaction (HOI) detection aims to identify humans and objects within images and interpret their interactions. Existing HOI methods rely heavily on large datasets with manual annotations to learn interactions from visual cues. These annotations are labor-intensive to create, prone to inconsistency, and limit scalability to new domains and rare interactions. We argue that recent advances in Vision-Language Models (VLMs) offer untapped potential, particularly in enhancing interaction representation. While prior work has injected such potential and even proposed training-free methods, there remain key gaps. Consequently, we propose a novel training-free HOI detection framework for Dynamic Scoring with enhanced semantics (DYSCO) that effectively utilizes textual and visual interaction representations within a multimodal registry, enabling robust and nuanced interaction understanding. This registry incorporates a small set of visual cues and uses innovative interaction signatures to improve the semantic alignment of verbs, facilitating effective generalization to rare interactions. Additionally, we propose a unique multi-head attention mechanism that adaptively weights the contributions of the visual and textual features. Experimental results demonstrate that our DYSCO surpasses training-free state-of-the-art models and is competitive with training-based approaches, particularly excelling in rare interactions. Code is available at https://github.com/francescotonini/dysco.
>
---
#### [new 064] A Low-Cost Machine Learning Approach for Timber Diameter Estimation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉与机器学习任务，旨在解决木材加工行业中木材种类和厚度识别效率低、依赖人工的问题。作者使用YOLOv5算法，在公开数据集TimberSeg 1.0上进行微调，通过标准RGB图像实现木材直径的自动估计。实验表明该方法在真实工业环境下表现良好，具备低成本、轻量级和可扩展性，适用于中小型企业的木材初步分类和库存管理。**

- **链接: [http://arxiv.org/pdf/2507.17219v1](http://arxiv.org/pdf/2507.17219v1)**

> **作者:** Fatemeh Hasanzadeh Fard; Sanaz Hasanzadeh Fard; Mehdi Jonoobi
>
> **摘要:** The wood processing industry, particularly in facilities such as sawmills and MDF production lines, requires accurate and efficient identification of species and thickness of the wood. Although traditional methods rely heavily on expert human labor, they are slow, inconsistent, and prone to error, especially when processing large volumes. This study focuses on practical and cost-effective machine learning frameworks that automate the estimation of timber log diameter using standard RGB images captured under real-world working conditions. We employ the YOLOv5 object detection algorithm, fine-tuned on a public dataset (TimberSeg 1.0), to detect individual timber logs and estimate thickness through bounding-box dimensions. Unlike previous methods that require expensive sensors or controlled environments, this model is trained on images taken in typical industrial sheds during timber delivery. Experimental results show that the model achieves a mean Average Precision (mAP@0.5) of 0.64, demonstrating reliable log detection even with modest computing resources. This lightweight, scalable solution holds promise for practical integration into existing workflows, including on-site inventory management and preliminary sorting, particularly in small and medium-sized operations.
>
---
#### [new 065] URPO: A Unified Reward & Policy Optimization Framework for Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在解决传统分离式策略与奖励模型导致的复杂流程与性能瓶颈问题。论文提出URPO框架，通过统一奖励与策略优化，在单一模型中融合指令执行与奖励生成，简化训练流程并提升效果。实验表明其在多个评估指标上优于基线模型。**

- **链接: [http://arxiv.org/pdf/2507.17515v1](http://arxiv.org/pdf/2507.17515v1)**

> **作者:** Songshuo Lu; Hua Wang; Zhi Chen; Yaohua Tang
>
> **摘要:** Large-scale alignment pipelines typically pair a policy model with a separately trained reward model whose parameters remain frozen during reinforcement learning (RL). This separation creates a complex, resource-intensive pipeline and suffers from a performance ceiling due to a static reward signal. We propose a novel framework, Unified Reward & Policy Optimization (URPO), that unifies instruction-following ("player") and reward modeling ("referee") within a single model and a single training phase. Our method recasts all alignment data-including preference pairs, verifiable reasoning, and open-ended instructions-into a unified generative format optimized by a single Group-Relative Policy Optimization (GRPO) loop. This enables the model to learn from ground-truth preferences and verifiable logic while simultaneously generating its own rewards for open-ended tasks. Experiments on the Qwen2.5-7B model demonstrate URPO's superiority. Our unified model significantly outperforms a strong baseline using a separate generative reward model, boosting the instruction-following score on AlpacaEval from 42.24 to 44.84 and the composite reasoning average from 32.66 to 35.66. Furthermore, URPO cultivates a superior internal evaluator as a byproduct of training, achieving a RewardBench score of 85.15 and surpassing the dedicated reward model it replaces (83.55). By eliminating the need for a separate reward model and fostering a co-evolutionary dynamic between generation and evaluation, URPO presents a simpler, more efficient, and more effective path towards robustly aligned language models.
>
---
#### [new 066] RemixFusion: Residual-based Mixed Representation for Large-scale Online RGB-D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于三维场景重建任务，旨在解决大规模在线RGB-D重建中细节缺失和计算效率低的问题。论文提出RemixFusion，结合显式TSDF与隐式神经残差表示，并优化多帧位姿与梯度放大策略，提升重建质量与跟踪精度。**

- **链接: [http://arxiv.org/pdf/2507.17594v1](http://arxiv.org/pdf/2507.17594v1)**

> **作者:** Yuqing Lan; Chenyang Zhu; Shuaifeng Zhi; Jiazhao Zhang; Zhoufeng Wang; Renjiao Yi; Yijie Wang; Kai Xu
>
> **摘要:** The introduction of the neural implicit representation has notably propelled the advancement of online dense reconstruction techniques. Compared to traditional explicit representations, such as TSDF, it improves the mapping completeness and memory efficiency. However, the lack of reconstruction details and the time-consuming learning of neural representations hinder the widespread application of neural-based methods to large-scale online reconstruction. We introduce RemixFusion, a novel residual-based mixed representation for scene reconstruction and camera pose estimation dedicated to high-quality and large-scale online RGB-D reconstruction. In particular, we propose a residual-based map representation comprised of an explicit coarse TSDF grid and an implicit neural module that produces residuals representing fine-grained details to be added to the coarse grid. Such mixed representation allows for detail-rich reconstruction with bounded time and memory budget, contrasting with the overly-smoothed results by the purely implicit representations, thus paving the way for high-quality camera tracking. Furthermore, we extend the residual-based representation to handle multi-frame joint pose optimization via bundle adjustment (BA). In contrast to the existing methods, which optimize poses directly, we opt to optimize pose changes. Combined with a novel technique for adaptive gradient amplification, our method attains better optimization convergence and global optimality. Furthermore, we adopt a local moving volume to factorize the mixed scene representation with a divide-and-conquer design to facilitate efficient online learning in our residual-based framework. Extensive experiments demonstrate that our method surpasses all state-of-the-art ones, including those based either on explicit or implicit representations, in terms of the accuracy of both mapping and tracking on large-scale scenes.
>
---
#### [new 067] Illicit object detection in X-ray imaging using deep learning techniques: A comparative evaluation
- **分类: cs.CV**

- **简介: 该论文属于X光图像中的违禁物品检测任务，旨在解决因物体遮挡、设备差异和数据不足导致的检测难题。论文系统比较了多种深度学习方法，在六个数据集和十种模型上进行了综合评估，分析了检测性能和计算效率，以推动研究进展。**

- **链接: [http://arxiv.org/pdf/2507.17508v1](http://arxiv.org/pdf/2507.17508v1)**

> **作者:** Jorgen Cani; Christos Diou; Spyridon Evangelatos; Vasileios Argyriou; Panagiotis Radoglou-Grammatikis; Panagiotis Sarigiannidis; Iraklis Varlamis; Georgios Th. Papadopoulos
>
> **摘要:** Automated X-ray inspection is crucial for efficient and unobtrusive security screening in various public settings. However, challenges such as object occlusion, variations in the physical properties of items, diversity in X-ray scanning devices, and limited training data hinder accurate and reliable detection of illicit items. Despite the large body of research in the field, reported experimental evaluations are often incomplete, with frequently conflicting outcomes. To shed light on the research landscape and facilitate further research, a systematic, detailed, and thorough comparative evaluation of recent Deep Learning (DL)-based methods for X-ray object detection is conducted. For this, a comprehensive evaluation framework is developed, composed of: a) Six recent, large-scale, and widely used public datasets for X-ray illicit item detection (OPIXray, CLCXray, SIXray, EDS, HiXray, and PIDray), b) Ten different state-of-the-art object detection schemes covering all main categories in the literature, including generic Convolutional Neural Network (CNN), custom CNN, generic transformer, and hybrid CNN-transformer architectures, and c) Various detection (mAP50 and mAP50:95) and time/computational-complexity (inference time (ms), parameter size (M), and computational load (GFLOPS)) metrics. A thorough analysis of the results leads to critical observations and insights, emphasizing key aspects such as: a) Overall behavior of the object detection schemes, b) Object-level detection performance, c) Dataset-specific observations, and d) Time efficiency and computational complexity analysis. To support reproducibility of the reported experimental results, the evaluation code and model weights are made publicly available at https://github.com/jgenc/xray-comparative-evaluation.
>
---
#### [new 068] FedVLM: Scalable Personalized Vision-Language Models through Federated Learning
- **分类: cs.CV**

- **简介: 该论文属于联邦学习与视觉-语言模型（VLM）个性化适配任务，旨在解决在数据分布非独立同分布（non-iid）的联邦环境下，如何高效、隐私安全地对VLM进行个性化微调的问题。论文提出FedVLM框架，结合低秩适配（LoRA）与个性化策略（pLoRA），实现去中心化模型适应，提升各客户端在本地数据上的性能表现。**

- **链接: [http://arxiv.org/pdf/2507.17088v1](http://arxiv.org/pdf/2507.17088v1)**

> **作者:** Arkajyoti Mitra; Afia Anjum; Paul Agbaje; Mert Pesé; Habeeb Olufowobi
>
> **摘要:** Vision-language models (VLMs) demonstrate impressive zero-shot and few-shot learning capabilities, making them essential for several downstream tasks. However, fine-tuning these models at scale remains challenging, particularly in federated environments where data is decentralized and non-iid across clients. Existing parameter-efficient tuning methods like LoRA (Low-Rank Adaptation) reduce computational overhead but struggle with heterogeneous client data, leading to suboptimal generalization. To address these challenges, we propose FedVLM, a federated LoRA fine-tuning framework that enables decentralized adaptation of VLMs while preserving model privacy and reducing reliance on centralized training. To further tackle data heterogeneity, we introduce personalized LoRA (pLoRA), which dynamically adapts LoRA parameters to each client's unique data distribution, significantly improving local adaptation while maintaining global model aggregation. Experiments on the RLAIF-V dataset show that pLoRA improves client-specific performance by 24.5% over standard LoRA, demonstrating superior adaptation in non-iid settings. FedVLM provides a scalable and efficient solution for fine-tuning VLMs in federated settings, advancing personalized adaptation in distributed learning scenarios.
>
---
#### [new 069] Sparser2Sparse: Single-shot Sparser-to-Sparse Learning for Spatial Transcriptomics Imputation with Natural Image Co-learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于空间转录组学（ST）数据填补任务，旨在解决高分辨率ST数据获取成本高的问题。作者提出S2S-ST方法，通过自监督学习与自然图像协同训练，实现从稀疏ST数据中准确重建基因表达图谱。**

- **链接: [http://arxiv.org/pdf/2507.16886v1](http://arxiv.org/pdf/2507.16886v1)**

> **作者:** Yaoyu Fang; Jiahe Qian; Xinkun Wang; Lee A. Cooper; Bo Zhou
>
> **备注:** 16 pages, 5 figure, under review
>
> **摘要:** Spatial transcriptomics (ST) has revolutionized biomedical research by enabling high resolution gene expression profiling within tissues. However, the high cost and scarcity of high resolution ST data remain significant challenges. We present Single-shot Sparser-to-Sparse (S2S-ST), a novel framework for accurate ST imputation that requires only a single and low-cost sparsely sampled ST dataset alongside widely available natural images for co-training. Our approach integrates three key innovations: (1) a sparser-to-sparse self-supervised learning strategy that leverages intrinsic spatial patterns in ST data, (2) cross-domain co-learning with natural images to enhance feature representation, and (3) a Cascaded Data Consistent Imputation Network (CDCIN) that iteratively refines predictions while preserving sampled gene data fidelity. Extensive experiments on diverse tissue types, including breast cancer, liver, and lymphoid tissue, demonstrate that our method outperforms state-of-the-art approaches in imputation accuracy. By enabling robust ST reconstruction from sparse inputs, our framework significantly reduces reliance on costly high resolution data, facilitating potential broader adoption in biomedical research and clinical applications.
>
---
#### [new 070] Coarse-to-fine crack cue for robust crack detection
- **分类: cs.CV; cs.NE; eess.IV**

- **简介: 该论文属于计算机视觉中的裂缝检测任务，旨在提升模型在不同环境下的鲁棒性与泛化能力。现有方法忽略裂缝的细长结构特性，导致在复杂背景、阴影或光照变化下表现不佳。作者提出CrackCue方法，通过粗到细的策略生成裂缝线索，先用池化和重建网络去除背景，再通过原图与重建图的差值得到裂缝线索，增强对干扰的鲁棒性。该方法可插拔至多种裂缝检测网络，实验证明其提升了检测性能。**

- **链接: [http://arxiv.org/pdf/2507.16851v1](http://arxiv.org/pdf/2507.16851v1)**

> **作者:** Zelong Liu; Yuliang Gu; Zhichao Sun; Huachao Zhu; Xin Xiao; Bo Du; Laurent Najman; Yongchao Xu
>
> **摘要:** Crack detection is an important task in computer vision. Despite impressive in-dataset performance, deep learning-based methods still struggle in generalizing to unseen domains. The thin structure property of cracks is usually overlooked by previous methods. In this work, we introduce CrackCue, a novel method for robust crack detection based on coarse-to-fine crack cue generation. The core concept lies on leveraging the thin structure property to generate a robust crack cue, guiding the crack detection. Specifically, we first employ a simple max-pooling and upsampling operation on the crack image. This results in a coarse crack-free background, based on which a fine crack-free background can be obtained via a reconstruction network. The difference between the original image and fine crack-free background provides a fine crack cue. This fine cue embeds robust crack prior information which is unaffected by complex backgrounds, shadow, and varied lighting. As a plug-and-play method, we incorporate the proposed CrackCue into three advanced crack detection networks. Extensive experimental results demonstrate that the proposed CrackCue significantly improves the generalization ability and robustness of the baseline methods. The source code will be publicly available.
>
---
#### [new 071] VBCD: A Voxel-Based Framework for Personalized Dental Crown Design
- **分类: cs.CV**

- **简介: 该论文属于医疗设计自动化任务，旨在解决牙冠修复设计耗时问题。作者提出VBCD框架，基于体素生成初始牙冠，并通过精细化优化和新损失函数提升效果，结合FDI牙位编码系统提高精度。实验表明该方法优于现有技术，实现个性化牙冠设计的自动化。**

- **链接: [http://arxiv.org/pdf/2507.17205v1](http://arxiv.org/pdf/2507.17205v1)**

> **作者:** Linda Wei; Chang Liu; Wenran Zhang; Zengji Zhang; Shaoting Zhang; Hongsheng Li
>
> **摘要:** The design of restorative dental crowns from intraoral scans is labor-intensive for dental technicians. To address this challenge, we propose a novel voxel-based framework for automated dental crown design (VBCD). The VBCD framework generates an initial coarse dental crown from voxelized intraoral scans, followed by a fine-grained refiner incorporating distance-aware supervision to improve accuracy and quality. During the training stage, we employ the Curvature and Margin line Penalty Loss (CMPL) to enhance the alignment of the generated crown with the margin line. Additionally, a positional prompt based on the FDI tooth numbering system is introduced to further improve the accuracy of the generated dental crowns. Evaluation on a large-scale dataset of intraoral scans demonstrated that our approach outperforms existing methods, providing a robust solution for personalized dental crown design.
>
---
#### [new 072] See the Forest and the Trees: A Synergistic Reasoning Framework for Knowledge-Based Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文属于知识驱动视觉问答（KBVQA）任务，旨在解决现有模型推理能力受限于单一证据来源的问题。作者提出Synergos-VQA框架，融合整体、结构和因果三种证据流，实现多维度推理。实验表明该方法在多个基准上达到新SOTA，并展示出对不同模型的兼容性与优越性。**

- **链接: [http://arxiv.org/pdf/2507.17659v1](http://arxiv.org/pdf/2507.17659v1)**

> **作者:** Junjie Wang; Yunhan Tang; Yijie Wang; Zhihao Yuan; Huan Wang; Yangfan He; Bin Li
>
> **摘要:** Multimodal Large Language Models (MLLMs) have pushed the frontiers of Knowledge-Based Visual Question Answering (KBVQA), yet their reasoning is fundamentally bottlenecked by a reliance on uni-dimensional evidence. This "seeing only the trees, but not the forest" approach prevents robust, multi-faceted understanding. Inspired by the principle of seeing both the forest and trees, we propose Synergos-VQA, a novel synergistic reasoning framework. At its core, Synergos-VQA concurrently generates and fuses three complementary evidence streams at inference time: (1) Holistic Evidence to perceive the entire scene (the "forest"), (2) Structural Evidence from a prototype-driven module to identify key objects (the "trees"), and (3) Causal Evidence from a counterfactual probe to ensure the reasoning is robustly grounded. By synergistically fusing this multi-faceted evidence, our framework achieves a more comprehensive and reliable reasoning process. Extensive experiments show that Synergos-VQA decisively establishes a new state-of-the-art on three challenging benchmarks, including OK-VQA and A-OKVQA. Furthermore, our approach demonstrates strong plug-and-play capabilities, significantly boosting various open-source MLLMs and proving that superior methodological design can outperform sheer model scale.
>
---
#### [new 073] Physics-based Human Pose Estimation from a Single Moving RGB Camera
- **分类: cs.CV**

- **简介: 该论文属于人体姿态估计任务，旨在解决单目移动相机下非平坦场景中的人体运动跟踪问题。作者提出了MoviCam数据集和PhysDynPose方法，结合场景几何与物理约束，提升了动态相机和复杂场景下的姿态估计准确性。**

- **链接: [http://arxiv.org/pdf/2507.17406v1](http://arxiv.org/pdf/2507.17406v1)**

> **作者:** Ayce Idil Aytekin; Chuqiao Li; Diogo Luvizon; Rishabh Dabral; Martin Oswald; Marc Habermann; Christian Theobalt
>
> **摘要:** Most monocular and physics-based human pose tracking methods, while achieving state-of-the-art results, suffer from artifacts when the scene does not have a strictly flat ground plane or when the camera is moving. Moreover, these methods are often evaluated on in-the-wild real world videos without ground-truth data or on synthetic datasets, which fail to model the real world light transport, camera motion, and pose-induced appearance and geometry changes. To tackle these two problems, we introduce MoviCam, the first non-synthetic dataset containing ground-truth camera trajectories of a dynamically moving monocular RGB camera, scene geometry, and 3D human motion with human-scene contact labels. Additionally, we propose PhysDynPose, a physics-based method that incorporates scene geometry and physical constraints for more accurate human motion tracking in case of camera motion and non-flat scenes. More precisely, we use a state-of-the-art kinematics estimator to obtain the human pose and a robust SLAM method to capture the dynamic camera trajectory, enabling the recovery of the human pose in the world frame. We then refine the kinematic pose estimate using our scene-aware physics optimizer. From our new benchmark, we found that even state-of-the-art methods struggle with this inherently challenging setting, i.e. a moving camera and non-planar environments, while our method robustly estimates both human and camera poses in world coordinates.
>
---
#### [new 074] SDGOCC: Semantic and Depth-Guided Bird's-Eye View Transformation for 3D Multimodal Occupancy Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶中的3D多模态占据预测任务。旨在解决现有方法在深度估计和语义信息利用上的不足。提出了SDG-OCC网络，结合语义与深度引导的视图变换和主动蒸馏策略，提升预测精度与实时性。**

- **链接: [http://arxiv.org/pdf/2507.17083v1](http://arxiv.org/pdf/2507.17083v1)**

> **作者:** Zaipeng Duan; Chenxu Dang; Xuzhong Hu; Pei An; Junfeng Ding; Jie Zhan; Yunbiao Xu; Jie Ma
>
> **备注:** accepted by CVPR2025
>
> **摘要:** Multimodal 3D occupancy prediction has garnered significant attention for its potential in autonomous driving. However, most existing approaches are single-modality: camera-based methods lack depth information, while LiDAR-based methods struggle with occlusions. Current lightweight methods primarily rely on the Lift-Splat-Shoot (LSS) pipeline, which suffers from inaccurate depth estimation and fails to fully exploit the geometric and semantic information of 3D LiDAR points. Therefore, we propose a novel multimodal occupancy prediction network called SDG-OCC, which incorporates a joint semantic and depth-guided view transformation coupled with a fusion-to-occupancy-driven active distillation. The enhanced view transformation constructs accurate depth distributions by integrating pixel semantics and co-point depth through diffusion and bilinear discretization. The fusion-to-occupancy-driven active distillation extracts rich semantic information from multimodal data and selectively transfers knowledge to image features based on LiDAR-identified regions. Finally, for optimal performance, we introduce SDG-Fusion, which uses fusion alone, and SDG-KL, which integrates both fusion and distillation for faster inference. Our method achieves state-of-the-art (SOTA) performance with real-time processing on the Occ3D-nuScenes dataset and shows comparable performance on the more challenging SurroundOcc-nuScenes dataset, demonstrating its effectiveness and robustness. The code will be released at https://github.com/DzpLab/SDGOCC.
>
---
#### [new 075] HIPPO-Video: Simulating Watch Histories with Large Language Models for Personalized Video Highlighting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于个性化视频高亮任务，旨在解决现有数据集缺乏用户个性化行为建模的问题。作者提出HIPPO-Video数据集，利用大语言模型生成多样化的观看历史，并开发HiPHer方法预测个性化视频片段的重要性，提升了真实场景中的视频高亮效果。**

- **链接: [http://arxiv.org/pdf/2507.16873v1](http://arxiv.org/pdf/2507.16873v1)**

> **作者:** Jeongeun Lee; Youngjae Yu; Dongha Lee
>
> **备注:** Accepted to COLM2025
>
> **摘要:** The exponential growth of video content has made personalized video highlighting an essential task, as user preferences are highly variable and complex. Existing video datasets, however, often lack personalization, relying on isolated videos or simple text queries that fail to capture the intricacies of user behavior. In this work, we introduce HIPPO-Video, a novel dataset for personalized video highlighting, created using an LLM-based user simulator to generate realistic watch histories reflecting diverse user preferences. The dataset includes 2,040 (watch history, saliency score) pairs, covering 20,400 videos across 170 semantic categories. To validate our dataset, we propose HiPHer, a method that leverages these personalized watch histories to predict preference-conditioned segment-wise saliency scores. Through extensive experiments, we demonstrate that our method outperforms existing generic and query-based approaches, showcasing its potential for highly user-centric video highlighting in real-world scenarios.
>
---
#### [new 076] Dual-branch Prompting for Multimodal Machine Translation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态机器翻译（MMT）任务，旨在解决现有方法依赖成对图文输入且易受视觉噪声影响的问题。论文提出D2P-MMT框架，利用扩散模型生成重构图像，结合源文本进行双分支提示学习，并引入分布对齐损失提升模型鲁棒性与翻译性能。**

- **链接: [http://arxiv.org/pdf/2507.17588v1](http://arxiv.org/pdf/2507.17588v1)**

> **作者:** Jie Wang; Zhendong Yang; Liansong Zong; Xiaobo Zhang; Dexian Wang; Ji Zhang
>
> **摘要:** Multimodal Machine Translation (MMT) typically enhances text-only translation by incorporating aligned visual features. Despite the remarkable progress, state-of-the-art MMT approaches often rely on paired image-text inputs at inference and are sensitive to irrelevant visual noise, which limits their robustness and practical applicability. To address these issues, we propose D2P-MMT, a diffusion-based dual-branch prompting framework for robust vision-guided translation. Specifically, D2P-MMT requires only the source text and a reconstructed image generated by a pre-trained diffusion model, which naturally filters out distracting visual details while preserving semantic cues. During training, the model jointly learns from both authentic and reconstructed images using a dual-branch prompting strategy, encouraging rich cross-modal interactions. To bridge the modality gap and mitigate training-inference discrepancies, we introduce a distributional alignment loss that enforces consistency between the output distributions of the two branches. Extensive experiments on the Multi30K dataset demonstrate that D2P-MMT achieves superior translation performance compared to existing state-of-the-art approaches.
>
---
#### [new 077] Asymmetric Lesion Detection with Geometric Patterns and CNN-SVM Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决皮肤病变不对称性的自动检测与分类问题。作者提出一种结合几何模式分析与CNN-SVM的分类方法，利用对称性信息辅助非专家理解病变特征，并在实验中取得了高检测率与分类性能。**

- **链接: [http://arxiv.org/pdf/2507.17185v1](http://arxiv.org/pdf/2507.17185v1)**

> **作者:** M. A. Rasel; Sameem Abdul Kareem; Zhenli Kwan; Nik Aimee Azizah Faheem; Winn Hui Han; Rebecca Kai Jan Choong; Shin Shen Yong; Unaizah Obaidellah
>
> **备注:** Accepted version. Published in Computers in Biology and Medicine, Volume 179, 2024. DOI: 10.1016/j.compbiomed.2024.108851
>
> **摘要:** In dermoscopic images, which allow visualization of surface skin structures not visible to the naked eye, lesion shape offers vital insights into skin diseases. In clinically practiced methods, asymmetric lesion shape is one of the criteria for diagnosing melanoma. Initially, we labeled data for a non-annotated dataset with symmetrical information based on clinical assessments. Subsequently, we propose a supporting technique, a supervised learning image processing algorithm, to analyze the geometrical pattern of lesion shape, aiding non-experts in understanding the criteria of an asymmetric lesion. We then utilize a pre-trained convolutional neural network (CNN) to extract shape, color, and texture features from dermoscopic images for training a multiclass support vector machine (SVM) classifier, outperforming state-of-the-art methods from the literature. In the geometry-based experiment, we achieved a 99.00% detection rate for dermatological asymmetric lesions. In the CNN-based experiment, the best performance is found with 94% Kappa Score, 95% Macro F1-score, and 97% Weighted F1-score for classifying lesion shapes (Asymmetric, Half-Symmetric, and Symmetric).
>
---
#### [new 078] Swin-TUNA : A Novel PEFT Approach for Accurate Food Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于食品图像语义分割任务，旨在解决现有大规模Transformer模型（如FoodSAM）因参数多、计算需求高而难以部署的问题。论文提出Swin-TUNA方法，通过在Swin Transformer中引入多尺度可训练适配器，仅更新4%的参数，实现高效准确的食品图像分割。**

- **链接: [http://arxiv.org/pdf/2507.17347v1](http://arxiv.org/pdf/2507.17347v1)**

> **作者:** Haotian Chen; Zhiyong Xiao
>
> **摘要:** In the field of food image processing, efficient semantic segmentation techniques are crucial for industrial applications. However, existing large-scale Transformer-based models (such as FoodSAM) face challenges in meeting practical deploymentrequirements due to their massive parameter counts and high computational resource demands. This paper introduces TUNable Adapter module (Swin-TUNA), a Parameter Efficient Fine-Tuning (PEFT) method that integrates multiscale trainable adapters into the Swin Transformer architecture, achieving high-performance food image segmentation by updating only 4% of the parameters. The core innovation of Swin-TUNA lies in its hierarchical feature adaptation mechanism: it designs separable convolutions in depth and dimensional mappings of varying scales to address the differences in features between shallow and deep networks, combined with a dynamic balancing strategy for tasks-agnostic and task-specific features. Experiments demonstrate that this method achieves mIoU of 50.56% and 74.94% on the FoodSeg103 and UECFoodPix Complete datasets, respectively, surpassing the fully parameterized FoodSAM model while reducing the parameter count by 98.7% (to only 8.13M). Furthermore, Swin-TUNA exhibits faster convergence and stronger generalization capabilities in low-data scenarios, providing an efficient solution for assembling lightweight food image.
>
---
#### [new 079] ScSAM: Debiasing Morphology and Distributional Variability in Subcellular Semantic Segmentation
- **分类: cs.CV; cs.AI; cs.LG; I.4.6**

- **简介: 该论文属于亚细胞语义分割任务，旨在解决因亚细胞形态和分布差异大导致的模型特征学习偏差问题。作者提出ScSAM方法，融合SAM与MAE特征，并设计类提示编码器，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2507.17149v1](http://arxiv.org/pdf/2507.17149v1)**

> **作者:** Bo Fang; Jianan Fan; Dongnan Liu; Hang Chang; Gerald J. Shami; Filip Braet; Weidong Cai
>
> **备注:** Accepted by 28th European Conference on Artificial Intelligence (ECAI)
>
> **摘要:** The significant morphological and distributional variability among subcellular components poses a long-standing challenge for learning-based organelle segmentation models, significantly increasing the risk of biased feature learning. Existing methods often rely on single mapping relationships, overlooking feature diversity and thereby inducing biased training. Although the Segment Anything Model (SAM) provides rich feature representations, its application to subcellular scenarios is hindered by two key challenges: (1) The variability in subcellular morphology and distribution creates gaps in the label space, leading the model to learn spurious or biased features. (2) SAM focuses on global contextual understanding and often ignores fine-grained spatial details, making it challenging to capture subtle structural alterations and cope with skewed data distributions. To address these challenges, we introduce ScSAM, a method that enhances feature robustness by fusing pre-trained SAM with Masked Autoencoder (MAE)-guided cellular prior knowledge to alleviate training bias from data imbalance. Specifically, we design a feature alignment and fusion module to align pre-trained embeddings to the same feature space and efficiently combine different representations. Moreover, we present a cosine similarity matrix-based class prompt encoder to activate class-specific features to recognize subcellular categories. Extensive experiments on diverse subcellular image datasets demonstrate that ScSAM outperforms state-of-the-art methods.
>
---
#### [new 080] CAPRI-CT: Causal Analysis and Predictive Reasoning for Image Quality Optimization in Computed Tomography
- **分类: cs.CV**

- **简介: 该论文属于医学影像任务，旨在解决CT成像中图像质量优化与辐射剂量平衡的问题。作者提出了CAPRI-CT，一种结合因果推理和深度学习的框架，融合图像数据与扫描参数，预测图像质量指标（如SNR）并支持反事实推理，帮助优化扫描协议。**

- **链接: [http://arxiv.org/pdf/2507.17420v1](http://arxiv.org/pdf/2507.17420v1)**

> **作者:** Sneha George Gnanakalavathy; Hairil Abdul Razak; Robert Meertens; Jonathan E. Fieldsend; Xujiong Ye; Mohammed M. Abdelsamea
>
> **摘要:** In computed tomography (CT), achieving high image quality while minimizing radiation exposure remains a key clinical challenge. This paper presents CAPRI-CT, a novel causal-aware deep learning framework for Causal Analysis and Predictive Reasoning for Image Quality Optimization in CT imaging. CAPRI-CT integrates image data with acquisition metadata (such as tube voltage, tube current, and contrast agent types) to model the underlying causal relationships that influence image quality. An ensemble of Variational Autoencoders (VAEs) is employed to extract meaningful features and generate causal representations from observational data, including CT images and associated imaging parameters. These input features are fused to predict the Signal-to-Noise Ratio (SNR) and support counterfactual inference, enabling what-if simulations, such as changes in contrast agents (types and concentrations) or scan parameters. CAPRI-CT is trained and validated using an ensemble learning approach, achieving strong predictive performance. By facilitating both prediction and interpretability, CAPRI-CT provides actionable insights that could help radiologists and technicians design more efficient CT protocols without repeated physical scans. The source code and dataset are publicly available at https://github.com/SnehaGeorge22/capri-ct.
>
---
#### [new 081] Transformer Based Building Boundary Reconstruction using Attraction Field Maps
- **分类: cs.CV**

- **简介: 该论文属于建筑边界重建任务，旨在解决从卫星图像中自动提取高精度建筑轮廓的问题。现有方法依赖人工，效率低。作者提出Decoupled-PolyGCN模型，结合图卷积网络与吸引力场图，提升边界几何规则性与检测精度，实现更优的自动化空间制图。**

- **链接: [http://arxiv.org/pdf/2507.17038v1](http://arxiv.org/pdf/2507.17038v1)**

> **作者:** Muhammad Kamran; Mohammad Moein Sheikholeslami; Andreas Wichmann; Gunho Sohn
>
> **摘要:** In recent years, the number of remote satellites orbiting the Earth has grown significantly, streaming vast amounts of high-resolution visual data to support diverse applications across civil, public, and military domains. Among these applications, the generation and updating of spatial maps of the built environment have become critical due to the extensive coverage and detailed imagery provided by satellites. However, reconstructing spatial maps from satellite imagery is a complex computer vision task, requiring the creation of high-level object representations, such as primitives, to accurately capture the built environment. While the past decade has witnessed remarkable advancements in object detection and representation using visual data, primitives-based object representation remains a persistent challenge in computer vision. Consequently, high-quality spatial maps often rely on labor-intensive and manual processes. This paper introduces a novel deep learning methodology leveraging Graph Convolutional Networks (GCNs) to address these challenges in building footprint reconstruction. The proposed approach enhances performance by incorporating geometric regularity into building boundaries, integrating multi-scale and multi-resolution features, and embedding Attraction Field Maps into the network. These innovations provide a scalable and precise solution for automated building footprint extraction from a single satellite image, paving the way for impactful applications in urban planning, disaster management, and large-scale spatial analysis. Our model, Decoupled-PolyGCN, outperforms existing methods by 6% in AP and 10% in AR, demonstrating its ability to deliver accurate and regularized building footprints across diverse and challenging scenarios.
>
---
#### [new 082] A Comprehensive Evaluation Framework for the Study of the Effects of Facial Filters on Face Recognition Accuracy
- **分类: cs.CV**

- **简介: 该论文属于人脸识别任务，旨在研究面部滤镜对识别准确率的影响。论文构建了一个综合评估框架，包含控制数据集、滤镜筛选方法及实验设计，并通过跨文化案例分析展示了滤镜对识别性能的影响及恢复方法。**

- **链接: [http://arxiv.org/pdf/2507.17729v1](http://arxiv.org/pdf/2507.17729v1)**

> **作者:** Kagan Ozturk; Louisa Conwill; Jacob Gutierrez; Kevin Bowyer; Walter J. Scheirer
>
> **摘要:** Facial filters are now commonplace for social media users around the world. Previous work has demonstrated that facial filters can negatively impact automated face recognition performance. However, these studies focus on small numbers of hand-picked filters in particular styles. In order to more effectively incorporate the wide ranges of filters present on various social media applications, we introduce a framework that allows for larger-scale study of the impact of facial filters on automated recognition. This framework includes a controlled dataset of face images, a principled filter selection process that selects a representative range of filters for experimentation, and a set of experiments to evaluate the filters' impact on recognition. We demonstrate our framework with a case study of filters from the American applications Instagram and Snapchat and the Chinese applications Meitu and Pitu to uncover cross-cultural differences. Finally, we show how the filtering effect in a face embedding space can easily be detected and restored to improve face recognition performance.
>
---
#### [new 083] Vec2Face+ for Face Dataset Generation
- **分类: cs.CV**

- **简介: 该论文属于人脸识别任务，旨在解决合成高质量人脸训练数据的问题。现有方法在增加类内变化时忽略了身份一致性。论文提出了Vec2Face+生成模型，结合三种策略：生成分离身份、增加属性变化、姿态控制，生成了VFace10K及其扩展数据集，在多个测试集上超越了真实数据集CASIA-WebFace的表现。同时指出合成数据在孪生验证和模型偏见方面仍存在问题，需进一步研究。**

- **链接: [http://arxiv.org/pdf/2507.17192v1](http://arxiv.org/pdf/2507.17192v1)**

> **作者:** Haiyu Wu; Jaskirat Singh; Sicong Tian; Liang Zheng; Kevin W. Bowyer
>
> **摘要:** When synthesizing identities as face recognition training data, it is generally believed that large inter-class separability and intra-class attribute variation are essential for synthesizing a quality dataset. % This belief is generally correct, and this is what we aim for. However, when increasing intra-class variation, existing methods overlook the necessity of maintaining intra-class identity consistency. % To address this and generate high-quality face training data, we propose Vec2Face+, a generative model that creates images directly from image features and allows for continuous and easy control of face identities and attributes. Using Vec2Face+, we obtain datasets with proper inter-class separability and intra-class variation and identity consistency using three strategies: 1) we sample vectors sufficiently different from others to generate well-separated identities; 2) we propose an AttrOP algorithm for increasing general attribute variations; 3) we propose LoRA-based pose control for generating images with profile head poses, which is more efficient and identity-preserving than AttrOP. % Our system generates VFace10K, a synthetic face dataset with 10K identities, which allows an FR model to achieve state-of-the-art accuracy on seven real-world test sets. Scaling the size to 4M and 12M images, the corresponding VFace100K and VFace300K datasets yield higher accuracy than the real-world training dataset, CASIA-WebFace, on five real-world test sets. This is the first time a synthetic dataset beats the CASIA-WebFace in average accuracy. In addition, we find that only 1 out of 11 synthetic datasets outperforms random guessing (\emph{i.e., 50\%}) in twin verification and that models trained with synthetic identities are more biased than those trained with real identities. Both are important aspects for future investigation.
>
---
#### [new 084] Fully Automated SAM for Single-source Domain Generalization in Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决单源域泛化中依赖专家标注提示和提示质量影响分割效果的问题。论文提出FA-SAM框架，通过自动生成提示和融合图像与提示嵌入，实现全自动且鲁棒的跨域医学图像分割。**

- **链接: [http://arxiv.org/pdf/2507.17281v1](http://arxiv.org/pdf/2507.17281v1)**

> **作者:** Huanli Zhuo; Leilei Ma; Haifeng Zhao; Shiwei Zhou; Dengdi Sun; Yanping Fu
>
> **备注:** This manuscript has been accepted for presentation at the IEEE International Conference on Systems, Man, and Cybernetics (IEEE SMC 2025) and is copyrighted by IEEE
>
> **摘要:** Although SAM-based single-source domain generalization models for medical image segmentation can mitigate the impact of domain shift on the model in cross-domain scenarios, these models still face two major challenges. First, the segmentation of SAM is highly dependent on domain-specific expert-annotated prompts, which prevents SAM from achieving fully automated medical image segmentation and therefore limits its application in clinical settings. Second, providing poor prompts (such as bounding boxes that are too small or too large) to the SAM prompt encoder can mislead SAM into generating incorrect mask results. Therefore, we propose the FA-SAM, a single-source domain generalization framework for medical image segmentation that achieves fully automated SAM. FA-SAM introduces two key innovations: an Auto-prompted Generation Model (AGM) branch equipped with a Shallow Feature Uncertainty Modeling (SUFM) module, and an Image-Prompt Embedding Fusion (IPEF) module integrated into the SAM mask decoder. Specifically, AGM models the uncertainty distribution of shallow features through the SUFM module to generate bounding box prompts for the target domain, enabling fully automated segmentation with SAM. The IPEF module integrates multiscale information from SAM image embeddings and prompt embeddings to capture global and local details of the target object, enabling SAM to mitigate the impact of poor prompts. Extensive experiments on publicly available prostate and fundus vessel datasets validate the effectiveness of FA-SAM and highlight its potential to address the above challenges.
>
---
#### [new 085] Toward a Real-Time Framework for Accurate Monocular 3D Human Pose Estimation with Geometric Priors
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于单目3D人体姿态估计任务，旨在解决在无约束环境下实时准确估计3D人体姿态的问题。论文提出了一种结合实时2D关键点检测与几何感知的2D到3D提升方法，利用相机内参和人体解剖先验知识，通过自校准和生物力学约束生成训练数据，实现无需专用硬件的快速、个性化3D姿态估计。**

- **链接: [http://arxiv.org/pdf/2507.16850v1](http://arxiv.org/pdf/2507.16850v1)**

> **作者:** Mohamed Adjel
>
> **备注:** IEEE ICRA 2025 (workshop: Enhancing Human Mobility: From Computer Vision-Based Motion Tracking to Wearable Assistive Robot Control), May 2025, Atlanta (Georgia), United States
>
> **摘要:** Monocular 3D human pose estimation remains a challenging and ill-posed problem, particularly in real-time settings and unconstrained environments. While direct imageto-3D approaches require large annotated datasets and heavy models, 2D-to-3D lifting offers a more lightweight and flexible alternative-especially when enhanced with prior knowledge. In this work, we propose a framework that combines real-time 2D keypoint detection with geometry-aware 2D-to-3D lifting, explicitly leveraging known camera intrinsics and subject-specific anatomical priors. Our approach builds on recent advances in self-calibration and biomechanically-constrained inverse kinematics to generate large-scale, plausible 2D-3D training pairs from MoCap and synthetic datasets. We discuss how these ingredients can enable fast, personalized, and accurate 3D pose estimation from monocular images without requiring specialized hardware. This proposal aims to foster discussion on bridging data-driven learning and model-based priors to improve accuracy, interpretability, and deployability of 3D human motion capture on edge devices in the wild.
>
---
#### [new 086] From Scan to Action: Leveraging Realistic Scans for Embodied Scene Understanding
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于三维场景理解任务，旨在解决真实世界3D扫描数据因格式多样、工具不兼容等问题难以被有效利用的挑战。论文提出基于USD的统一标注整合方法，并通过LLM场景编辑与机器人仿真应用验证有效性，成功提升数据可用性与应用泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.17585v1](http://arxiv.org/pdf/2507.17585v1)**

> **作者:** Anna-Maria Halacheva; Jan-Nico Zaech; Sombit Dey; Luc Van Gool; Danda Pani Paudel
>
> **备注:** Accepted at the OpenSUN3D Workshop, CVPR 2025. This workshop paper is not included in the official CVPR proceedings
>
> **摘要:** Real-world 3D scene-level scans offer realism and can enable better real-world generalizability for downstream applications. However, challenges such as data volume, diverse annotation formats, and tool compatibility limit their use. This paper demonstrates a methodology to effectively leverage these scans and their annotations. We propose a unified annotation integration using USD, with application-specific USD flavors. We identify challenges in utilizing holistic real-world scan datasets and present mitigation strategies. The efficacy of our approach is demonstrated through two downstream applications: LLM-based scene editing, enabling effective LLM understanding and adaptation of the data (80% success), and robotic simulation, achieving an 87% success rate in policy learning.
>
---
#### [new 087] DeMo++: Motion Decoupling for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的运动预测与规划任务，旨在解决现有方法在建模轨迹时空演化上的不足。论文提出DeMo++框架，将运动估计解耦为整体运动意图和精细时空状态，并引入跨场景交互机制。结合Attention与Mamba模型，提升了轨迹预测与规划性能，在多个基准上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2507.17342v1](http://arxiv.org/pdf/2507.17342v1)**

> **作者:** Bozhou Zhang; Nan Song; Xiatian Zhu; Li Zhang
>
> **备注:** Journal extension of NeurIPS 2024. arXiv admin note: substantial text overlap with arXiv:2410.05982
>
> **摘要:** Motion forecasting and planning are tasked with estimating the trajectories of traffic agents and the ego vehicle, respectively, to ensure the safety and efficiency of autonomous driving systems in dynamically changing environments. State-of-the-art methods typically adopt a one-query-one-trajectory paradigm, where each query corresponds to a unique trajectory for predicting multi-mode trajectories. While this paradigm can produce diverse motion intentions, it often falls short in modeling the intricate spatiotemporal evolution of trajectories, which can lead to collisions or suboptimal outcomes. To overcome this limitation, we propose DeMo++, a framework that decouples motion estimation into two distinct components: holistic motion intentions to capture the diverse potential directions of movement, and fine spatiotemporal states to track the agent's dynamic progress within the scene and enable a self-refinement capability. Further, we introduce a cross-scene trajectory interaction mechanism to explore the relationships between motions in adjacent scenes. This allows DeMo++ to comprehensively model both the diversity of motion intentions and the spatiotemporal evolution of each trajectory. To effectively implement this framework, we developed a hybrid model combining Attention and Mamba. This architecture leverages the strengths of both mechanisms for efficient scene information aggregation and precise trajectory state sequence modeling. Extensive experiments demonstrate that DeMo++ achieves state-of-the-art performance across various benchmarks, including motion forecasting (Argoverse 2 and nuScenes), motion planning (nuPlan), and end-to-end planning (NAVSIM).
>
---
#### [new 088] An h-space Based Adversarial Attack for Protection Against Few-shot Personalization
- **分类: cs.CV**

- **简介: 该论文属于图像生成安全任务，旨在解决扩散模型在少样本个性化生成中的隐私风险问题。作者提出了基于h空间的对抗攻击方法HAAD及其高效变体HAAD-KV，通过生成破坏性的扰动来削弱模型生成能力，从而保护私有内容不被未经授权地修改。**

- **链接: [http://arxiv.org/pdf/2507.17554v1](http://arxiv.org/pdf/2507.17554v1)**

> **作者:** Xide Xu; Sandesh Kamath; Muhammad Atif Butt; Bogdan Raducanu
>
> **备注:** 32 pages, 15 figures. Accepted by ACM Multimedia 2025
>
> **摘要:** The versatility of diffusion models in generating customized images from few samples raises significant privacy concerns, particularly regarding unauthorized modifications of private content. This concerning issue has renewed the efforts in developing protection mechanisms based on adversarial attacks, which generate effective perturbations to poison diffusion models. Our work is motivated by the observation that these models exhibit a high degree of abstraction within their semantic latent space (`h-space'), which encodes critical high-level features for generating coherent and meaningful content. In this paper, we propose a novel anti-customization approach, called HAAD (h-space based Adversarial Attack for Diffusion models), that leverages adversarial attacks to craft perturbations based on the h-space that can efficiently degrade the image generation process. Building upon HAAD, we further introduce a more efficient variant, HAAD-KV, that constructs perturbations solely based on the KV parameters of the h-space. This strategy offers a stronger protection, that is computationally less expensive. Despite their simplicity, our methods outperform state-of-the-art adversarial attacks, highlighting their effectiveness.
>
---
#### [new 089] Temporal Point-Supervised Signal Reconstruction: A Human-Annotation-Free Framework for Weak Moving Target Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于弱小目标检测任务，旨在解决低空监视系统中弱移动目标因信号弱、背景复杂难以检测的问题。提出了一种无需人工标注的时序点监督框架TPS，通过时序信号重建网络TSRNet及动态多尺度注意力模块，实现像素级弱信号检测，并结合图优化提升准确性。**

- **链接: [http://arxiv.org/pdf/2507.17334v1](http://arxiv.org/pdf/2507.17334v1)**

> **作者:** Weihua Gao; Chunxu Ren; Wenlong Niu; Xiaodong Peng
>
> **摘要:** In low-altitude surveillance and early warning systems, detecting weak moving targets remains a significant challenge due to low signal energy, small spatial extent, and complex background clutter. Existing methods struggle with extracting robust features and suffer from the lack of reliable annotations. To address these limitations, we propose a novel Temporal Point-Supervised (TPS) framework that enables high-performance detection of weak targets without any manual annotations.Instead of conventional frame-based detection, our framework reformulates the task as a pixel-wise temporal signal modeling problem, where weak targets manifest as short-duration pulse-like responses. A Temporal Signal Reconstruction Network (TSRNet) is developed under the TPS paradigm to reconstruct these transient signals.TSRNet adopts an encoder-decoder architecture and integrates a Dynamic Multi-Scale Attention (DMSAttention) module to enhance its sensitivity to diverse temporal patterns. Additionally, a graph-based trajectory mining strategy is employed to suppress false alarms and ensure temporal consistency.Extensive experiments on a purpose-built low-SNR dataset demonstrate that our framework outperforms state-of-the-art methods while requiring no human annotations. It achieves strong detection performance and operates at over 1000 FPS, underscoring its potential for real-time deployment in practical scenarios.
>
---
#### [new 090] Learning-based Stage Verification System in Manual Assembly Scenarios
- **分类: cs.CV**

- **简介: 该论文属于工业监控任务，旨在解决仅使用少量视觉传感器时，手动装配过程中多目标、多状态的高精度监控问题。作者提出一种基于多机器学习模型的方法，通过融合相同时间戳的状态信息，实现平均超过92%的阶段识别准确率，并具备更强的错误检测与可视化能力。**

- **链接: [http://arxiv.org/pdf/2507.17304v1](http://arxiv.org/pdf/2507.17304v1)**

> **作者:** Xingjian Zhang; Yutong Duan; Zaishu Chen
>
> **摘要:** In the context of Industry 4.0, effective monitoring of multiple targets and states during assembly processes is crucial, particularly when constrained to using only visual sensors. Traditional methods often rely on either multiple sensor types or complex hardware setups to achieve high accuracy in monitoring, which can be cost-prohibitive and difficult to implement in dynamic industrial environments. This study presents a novel approach that leverages multiple machine learning models to achieve precise monitoring under the limitation of using a minimal number of visual sensors. By integrating state information from identical timestamps, our method detects and confirms the current stage of the assembly process with an average accuracy exceeding 92%. Furthermore, our approach surpasses conventional methods by offering enhanced error detection and visuali-zation capabilities, providing real-time, actionable guidance to operators. This not only improves the accuracy and efficiency of assembly monitoring but also re-duces dependency on expensive hardware solutions, making it a more practical choice for modern industrial applications.
>
---
#### [new 091] Accelerating Parallel Diffusion Model Serving with Residual Compression
- **分类: cs.CV**

- **简介: 论文提出CompactFusion，用于加速扩散模型并行推理的压缩框架。任务是解决多设备并行推理时通信开销大的问题。通过残差压缩传输激活差异，减少冗余数据，提升效率和生成质量，适用于多种扩散模型和并行设置。**

- **链接: [http://arxiv.org/pdf/2507.17511v1](http://arxiv.org/pdf/2507.17511v1)**

> **作者:** Jiajun Luo; Yicheng Xiao; Jianru Xu; Yangxiu You; Rongwei Lu; Chen Tang; Jingyan Jiang; Zhi Wang
>
> **摘要:** Diffusion models produce realistic images and videos but require substantial computational resources, necessitating multi-accelerator parallelism for real-time deployment. However, parallel inference introduces significant communication overhead from exchanging large activations between devices, limiting efficiency and scalability. We present CompactFusion, a compression framework that significantly reduces communication while preserving generation quality. Our key observation is that diffusion activations exhibit strong temporal redundancy-adjacent steps produce highly similar activations, saturating bandwidth with near-duplicate data carrying little new information. To address this inefficiency, we seek a more compact representation that encodes only the essential information. CompactFusion achieves this via Residual Compression that transmits only compressed residuals (step-wise activation differences). Based on empirical analysis and theoretical justification, we show that it effectively removes redundant data, enabling substantial data reduction while maintaining high fidelity. We also integrate lightweight error feedback to prevent error accumulation. CompactFusion establishes a new paradigm for parallel diffusion inference, delivering lower latency and significantly higher generation quality than prior methods. On 4xL20, it achieves 3.0x speedup while greatly improving fidelity. It also uniquely supports communication-heavy strategies like sequence parallelism on slow networks, achieving 6.7x speedup over prior overlap-based method. CompactFusion applies broadly across diffusion models and parallel settings, and integrates easily without requiring pipeline rework. Portable implementation demonstrated on xDiT is publicly available at https://github.com/Cobalt-27/CompactFusion
>
---
#### [new 092] MaskedCLIP: Bridging the Masked and CLIP Space for Semi-Supervised Medical Vision-Language Pre-training
- **分类: cs.CV**

- **简介: 该论文属于半监督视觉-语言预训练任务，旨在解决仅使用配对或未配对图像数据限制模型学习更丰富特征的问题。论文提出MaskedCLIP，结合掩码图像建模与对比语言-图像预训练，通过桥接掩码特征空间与CLIP特征空间，提升医学图像分析的特征泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.17239v1](http://arxiv.org/pdf/2507.17239v1)**

> **作者:** Lei Zhu; Jun Zhou; Rick Siow Mong Goh; Yong Liu
>
> **备注:** Accepted to MedAGI 2025 (Oral)
>
> **摘要:** Foundation models have recently gained tremendous popularity in medical image analysis. State-of-the-art methods leverage either paired image-text data via vision-language pre-training or unpaired image data via self-supervised pre-training to learn foundation models with generalizable image features to boost downstream task performance. However, learning foundation models exclusively on either paired or unpaired image data limits their ability to learn richer and more comprehensive image features. In this paper, we investigate a novel task termed semi-supervised vision-language pre-training, aiming to fully harness the potential of both paired and unpaired image data for foundation model learning. To this end, we propose MaskedCLIP, a synergistic masked image modeling and contrastive language-image pre-training framework for semi-supervised vision-language pre-training. The key challenge in combining paired and unpaired image data for learning a foundation model lies in the incompatible feature spaces derived from these two types of data. To address this issue, we propose to connect the masked feature space with the CLIP feature space with a bridge transformer. In this way, the more semantic specific CLIP features can benefit from the more general masked features for semantic feature extraction. We further propose a masked knowledge distillation loss to distill semantic knowledge of original image features in CLIP feature space back to the predicted masked image features in masked feature space. With this mutually interactive design, our framework effectively leverages both paired and unpaired image data to learn more generalizable image features for downstream tasks. Extensive experiments on retinal image analysis demonstrate the effectiveness and data efficiency of our method.
>
---
#### [new 093] PolarAnything: Diffusion-based Polarimetric Image Synthesis
- **分类: cs.CV**

- **简介: 该论文属于图像合成任务，旨在解决偏振图像获取困难的问题。现有方法依赖复杂3D资产，难以生成大规模真实图像。论文提出PolarAnything，基于扩散模型，从单张RGB图像合成高质量偏振图像，兼具真实感与物理准确性，支持后续三维重建等任务。**

- **链接: [http://arxiv.org/pdf/2507.17268v1](http://arxiv.org/pdf/2507.17268v1)**

> **作者:** Kailong Zhang; Youwei Lyu; Heng Guo; Si Li; Zhanyu Ma; Boxin Shi
>
> **备注:** 11 pages
>
> **摘要:** Polarization images facilitate image enhancement and 3D reconstruction tasks, but the limited accessibility of polarization cameras hinders their broader application. This gap drives the need for synthesizing photorealistic polarization images.The existing polarization simulator Mitsuba relies on a parametric polarization image formation model and requires extensive 3D assets covering shape and PBR materials, preventing it from generating large-scale photorealistic images. To address this problem, we propose PolarAnything, capable of synthesizing polarization images from a single RGB input with both photorealism and physical accuracy, eliminating the dependency on 3D asset collections. Drawing inspiration from the zero-shot performance of pretrained diffusion models, we introduce a diffusion-based generative framework with an effective representation strategy that preserves the fidelity of polarization properties. Experiments show that our model generates high-quality polarization images and supports downstream tasks like shape from polarization.
>
---
#### [new 094] Few-Shot Learning in Video and 3D Object Detection: A Survey
- **分类: cs.CV**

- **简介: 该论文属于视频与3D目标检测任务，旨在解决标注数据昂贵且耗时的问题。通过综述少样本学习（FSL）技术，探讨如何在仅少量标注示例下识别新类别。论文分析了视频中利用时空信息、3D检测中结合点云网络等方法，并总结了跨模态信息整合与泛化能力提升等核心问题。**

- **链接: [http://arxiv.org/pdf/2507.17079v1](http://arxiv.org/pdf/2507.17079v1)**

> **作者:** Md Meftahul Ferdaus; Kendall N. Niles; Joe Tom; Mahdi Abdelguerfi; Elias Ioup
>
> **备注:** Under review in ACM Computing Surveys
>
> **摘要:** Few-shot learning (FSL) enables object detection models to recognize novel classes given only a few annotated examples, thereby reducing expensive manual data labeling. This survey examines recent FSL advances for video and 3D object detection. For video, FSL is especially valuable since annotating objects across frames is more laborious than for static images. By propagating information across frames, techniques like tube proposals and temporal matching networks can detect new classes from a couple examples, efficiently leveraging spatiotemporal structure. FSL for 3D detection from LiDAR or depth data faces challenges like sparsity and lack of texture. Solutions integrate FSL with specialized point cloud networks and losses tailored for class imbalance. Few-shot 3D detection enables practical autonomous driving deployment by minimizing costly 3D annotation needs. Core issues in both domains include balancing generalization and overfitting, integrating prototype matching, and handling data modality properties. In summary, FSL shows promise for reducing annotation requirements and enabling real-world video, 3D, and other applications by efficiently leveraging information across feature, temporal, and data modalities. By comprehensively surveying recent advancements, this paper illuminates FSL's potential to minimize supervision needs and enable deployment across video, 3D, and other real-world applications.
>
---
#### [new 095] InvRGB+L: Inverse Rendering of Complex Scenes with Unified Color and LiDAR Reflectance Modeling
- **分类: cs.CV**

- **简介: 该论文属于逆向渲染任务，旨在从单个RGB+LiDAR序列重建复杂场景。它解决了传统方法因依赖RGB导致的材质估计不佳问题，通过引入物理基础的LiDAR着色模型和RGB-LiDAR材质一致性损失，实现了更优的场景重建与动态编辑能力。**

- **链接: [http://arxiv.org/pdf/2507.17613v1](http://arxiv.org/pdf/2507.17613v1)**

> **作者:** Xiaoxue Chen; Bhargav Chandaka; Chih-Hao Lin; Ya-Qin Zhang; David Forsyth; Hao Zhao; Shenlong Wang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** We present InvRGB+L, a novel inverse rendering model that reconstructs large, relightable, and dynamic scenes from a single RGB+LiDAR sequence. Conventional inverse graphics methods rely primarily on RGB observations and use LiDAR mainly for geometric information, often resulting in suboptimal material estimates due to visible light interference. We find that LiDAR's intensity values-captured with active illumination in a different spectral range-offer complementary cues for robust material estimation under variable lighting. Inspired by this, InvRGB+L leverages LiDAR intensity cues to overcome challenges inherent in RGB-centric inverse graphics through two key innovations: (1) a novel physics-based LiDAR shading model and (2) RGB-LiDAR material consistency losses. The model produces novel-view RGB and LiDAR renderings of urban and indoor scenes and supports relighting, night simulations, and dynamic object insertions, achieving results that surpass current state-of-the-art methods in both scene-level urban inverse rendering and LiDAR simulation.
>
---
#### [new 096] MyGO: Make your Goals Obvious, Avoiding Semantic Confusion in Prostate Cancer Lesion Region Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决前列腺癌病灶区域语义混淆问题。作者提出像素锚点模块和自注意力选择策略，增强模型对全局上下文的理解，结合焦点损失缓解类别不平衡，显著提升病灶检测精度。**

- **链接: [http://arxiv.org/pdf/2507.17269v1](http://arxiv.org/pdf/2507.17269v1)**

> **作者:** Zhengcheng Lin; Zuobin Ying; Zhenyu Li; Zhenyu Liu; Jian Lu; Weiping Ding
>
> **摘要:** Early diagnosis and accurate identification of lesion location and progression in prostate cancer (PCa) are critical for assisting clinicians in formulating effective treatment strategies. However, due to the high semantic homogeneity between lesion and non-lesion areas, existing medical image segmentation methods often struggle to accurately comprehend lesion semantics, resulting in the problem of semantic confusion. To address this challenge, we propose a novel Pixel Anchor Module, which guides the model to discover a sparse set of feature anchors that serve to capture and interpret global contextual information. This mechanism enhances the model's nonlinear representation capacity and improves segmentation accuracy within lesion regions. Moreover, we design a self-attention-based Top_k selection strategy to further refine the identification of these feature anchors, and incorporate a focal loss function to mitigate class imbalance, thereby facilitating more precise semantic interpretation across diverse regions. Our method achieves state-of-the-art performance on the PI-CAI dataset, demonstrating 69.73% IoU and 74.32% Dice scores, and significantly improving prostate cancer lesion detection.
>
---
#### [new 097] A Versatile Pathology Co-pilot via Reasoning Enhanced Multimodal Large Language Model
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析与多模态大语言模型结合的任务，旨在解决当前模型推理能力弱、依赖昂贵标注的问题。作者提出了SmartPath-R1，通过多任务强化学习与混合专家机制，实现病理图像的分类、检测、分割及问答等多层级分析，提升了模型的推理能力与应用广度。**

- **链接: [http://arxiv.org/pdf/2507.17303v1](http://arxiv.org/pdf/2507.17303v1)**

> **作者:** Zhe Xu; Ziyi Liu; Junlin Hou; Jiabo Ma; Cheng Jin; Yihui Wang; Zhixuan Chen; Zhengyu Zhang; Zhengrui Guo; Fengtao Zhou; Yingxue Xu; Xi Wang; Ronald Cheong Kin Chan; Li Liang; Hao Chen
>
> **摘要:** Multimodal large language models (MLLMs) have emerged as powerful tools for computational pathology, offering unprecedented opportunities to integrate pathological images with language context for comprehensive diagnostic analysis. These models hold particular promise for automating complex tasks that traditionally require expert interpretation of pathologists. However, current MLLM approaches in pathology demonstrate significantly constrained reasoning capabilities, primarily due to their reliance on expensive chain-of-thought annotations. Additionally, existing methods remain limited to simplex application of visual question answering (VQA) at region-of-interest (ROI) level, failing to address the full spectrum of diagnostic needs such as ROI classification, detection, segmentation, whole-slide-image (WSI) classification and VQA in clinical practice. In this study, we present SmartPath-R1, a versatile MLLM capable of simultaneously addressing both ROI-level and WSI-level tasks while demonstrating robust pathological reasoning capability. Our framework combines scale-dependent supervised fine-tuning and task-aware reinforcement fine-tuning, which circumvents the requirement for chain-of-thought supervision by leveraging the intrinsic knowledge within MLLM. Furthermore, SmartPath-R1 integrates multiscale and multitask analysis through a mixture-of-experts mechanism, enabling dynamic processing for diverse tasks. We curate a large-scale dataset comprising 2.3M ROI samples and 188K WSI samples for training and evaluation. Extensive experiments across 72 tasks validate the effectiveness and superiority of the proposed approach. This work represents a significant step toward developing versatile, reasoning-enhanced AI systems for precision pathology.
>
---
#### [new 098] A tissue and cell-level annotated H&E and PD-L1 histopathology image dataset in non-small cell lung cancer
- **分类: q-bio.QM; cs.CV; eess.IV**

- **简介: 该论文旨在解决非小细胞肺癌（NSCLC）组织病理图像中免疫微环境分析缺乏标注数据的问题。论文任务包括组织分割、细胞核检测和PD-L1阳性细胞检测。作者构建了IGNITE数据集，包含887个标注区域，涵盖多中心、多染色和多扫描仪数据，是首个提供H&E和PD-L1 IHC标注的公开NSCLC数据集。**

- **链接: [http://arxiv.org/pdf/2507.16855v1](http://arxiv.org/pdf/2507.16855v1)**

> **作者:** Joey Spronck; Leander van Eekelen; Dominique van Midden; Joep Bogaerts; Leslie Tessier; Valerie Dechering; Muradije Demirel-Andishmand; Gabriel Silva de Souza; Roland Nemeth; Enrico Munari; Giuseppe Bogina; Ilaria Girolami; Albino Eccher; Balazs Acs; Ceren Boyaci; Natalie Klubickova; Monika Looijen-Salamon; Shoko Vos; Francesco Ciompi
>
> **备注:** Our dataset is available at 'https://zenodo.org/records/15674785' and our code is available at 'https://github.com/DIAGNijmegen/ignite-data-toolkit'
>
> **摘要:** The tumor immune microenvironment (TIME) in non-small cell lung cancer (NSCLC) histopathology contains morphological and molecular characteristics predictive of immunotherapy response. Computational quantification of TIME characteristics, such as cell detection and tissue segmentation, can support biomarker development. However, currently available digital pathology datasets of NSCLC for the development of cell detection or tissue segmentation algorithms are limited in scope, lack annotations of clinically prevalent metastatic sites, and forgo molecular information such as PD-L1 immunohistochemistry (IHC). To fill this gap, we introduce the IGNITE data toolkit, a multi-stain, multi-centric, and multi-scanner dataset of annotated NSCLC whole-slide images. We publicly release 887 fully annotated regions of interest from 155 unique patients across three complementary tasks: (i) multi-class semantic segmentation of tissue compartments in H&E-stained slides, with 16 classes spanning primary and metastatic NSCLC, (ii) nuclei detection, and (iii) PD-L1 positive tumor cell detection in PD-L1 IHC slides. To the best of our knowledge, this is the first public NSCLC dataset with manual annotations of H&E in metastatic sites and PD-L1 IHC.
>
---
#### [new 099] Controllable Video Generation: A Survey
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决现有文本生成视频模型控制性不足的问题。论文系统综述了可控视频生成方法，重点分析了如何通过非文本条件（如姿态、深度图等）增强生成控制，提升视频生成的灵活性与实用性。**

- **链接: [http://arxiv.org/pdf/2507.16869v1](http://arxiv.org/pdf/2507.16869v1)**

> **作者:** Yue Ma; Kunyu Feng; Zhongyuan Hu; Xinyu Wang; Yucheng Wang; Mingzhe Zheng; Xuanhua He; Chenyang Zhu; Hongyu Liu; Yingqing He; Zeyu Wang; Zhifeng Li; Xiu Li; Wei Liu; Dan Xu; Linfeng Zhang; Qifeng Chen
>
> **备注:** project page: https://github.com/mayuelala/Awesome-Controllable-Video-Generation
>
> **摘要:** With the rapid development of AI-generated content (AIGC), video generation has emerged as one of its most dynamic and impactful subfields. In particular, the advancement of video generation foundation models has led to growing demand for controllable video generation methods that can more accurately reflect user intent. Most existing foundation models are designed for text-to-video generation, where text prompts alone are often insufficient to express complex, multi-modal, and fine-grained user requirements. This limitation makes it challenging for users to generate videos with precise control using current models. To address this issue, recent research has explored the integration of additional non-textual conditions, such as camera motion, depth maps, and human pose, to extend pretrained video generation models and enable more controllable video synthesis. These approaches aim to enhance the flexibility and practical applicability of AIGC-driven video generation systems. In this survey, we provide a systematic review of controllable video generation, covering both theoretical foundations and recent advances in the field. We begin by introducing the key concepts and commonly used open-source video generation models. We then focus on control mechanisms in video diffusion models, analyzing how different types of conditions can be incorporated into the denoising process to guide generation. Finally, we categorize existing methods based on the types of control signals they leverage, including single-condition generation, multi-condition generation, and universal controllable generation. For a complete list of the literature on controllable video generation reviewed, please visit our curated repository at https://github.com/mayuelala/Awesome-Controllable-Video-Generation.
>
---
#### [new 100] A Hybrid CNN-VSSM model for Multi-View, Multi-Task Mammography Analysis: Robust Diagnosis with Attention-Based Fusion
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决乳腺X线摄影筛查中早期准确诊断乳腺癌的问题。现有方法多依赖单视角输入或单任务输出，限制了临床应用。论文提出了一种结合卷积神经网络（CNN）和视觉状态空间模型（VSSM）的多视角、多任务深度学习框架，并引入注意力融合模块，提升诊断鲁棒性与可解释性。实验表明该方法在BI-RADS分类任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.16955v1](http://arxiv.org/pdf/2507.16955v1)**

> **作者:** Yalda Zafari; Roaa Elalfy; Mohamed Mabrok; Somaya Al-Maadeed; Tamer Khattab; Essam A. Rashed
>
> **摘要:** Early and accurate interpretation of screening mammograms is essential for effective breast cancer detection, yet it remains a complex challenge due to subtle imaging findings and diagnostic ambiguity. Many existing AI approaches fall short by focusing on single view inputs or single-task outputs, limiting their clinical utility. To address these limitations, we propose a novel multi-view, multitask hybrid deep learning framework that processes all four standard mammography views and jointly predicts diagnostic labels and BI-RADS scores for each breast. Our architecture integrates a hybrid CNN VSSM backbone, combining convolutional encoders for rich local feature extraction with Visual State Space Models (VSSMs) to capture global contextual dependencies. To improve robustness and interpretability, we incorporate a gated attention-based fusion module that dynamically weights information across views, effectively handling cases with missing data. We conduct extensive experiments across diagnostic tasks of varying complexity, benchmarking our proposed hybrid models against baseline CNN architectures and VSSM models in both single task and multi task learning settings. Across all tasks, the hybrid models consistently outperform the baselines. In the binary BI-RADS 1 vs. 5 classification task, the shared hybrid model achieves an AUC of 0.9967 and an F1 score of 0.9830. For the more challenging ternary classification, it attains an F1 score of 0.7790, while in the five-class BI-RADS task, the best F1 score reaches 0.4904. These results highlight the effectiveness of the proposed hybrid framework and underscore both the potential and limitations of multitask learning for improving diagnostic performance and enabling clinically meaningful mammography analysis.
>
---
#### [new 101] MCM: Mamba-based Cardiac Motion Tracking using Sequential Images in MRI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决心脏磁共振图像中心肌运动跟踪不连续、不一致的问题。作者提出了一种基于Mamba模型的网络（MCM），利用双向扫描机制和运动解码器，实现更平滑、时间一致的心脏运动估计，提升了运动跟踪的准确性与连贯性。**

- **链接: [http://arxiv.org/pdf/2507.17678v1](http://arxiv.org/pdf/2507.17678v1)**

> **作者:** Jiahui Yin; Xinxing Cheng; Jinming Duan; Yan Pang; Declan O'Regan; Hadrien Reynaud; Qingjie Meng
>
> **备注:** Medical Image Computing and Computer-Assisted Intervention (MICCAI), Reconstruction and Imaging Motion Estimation Workshop (RIME), 2025
>
> **摘要:** Myocardial motion tracking is important for assessing cardiac function and diagnosing cardiovascular diseases, for which cine cardiac magnetic resonance (CMR) has been established as the gold standard imaging modality. Many existing methods learn motion from single image pairs consisting of a reference frame and a randomly selected target frame from the cardiac cycle. However, these methods overlook the continuous nature of cardiac motion and often yield inconsistent and non-smooth motion estimations. In this work, we propose a novel Mamba-based cardiac motion tracking network (MCM) that explicitly incorporates target image sequence from the cardiac cycle to achieve smooth and temporally consistent motion tracking. By developing a bi-directional Mamba block equipped with a bi-directional scanning mechanism, our method facilitates the estimation of plausible deformation fields. With our proposed motion decoder that integrates motion information from frames adjacent to the target frame, our method further enhances temporal coherence. Moreover, by taking advantage of Mamba's structured state-space formulation, the proposed method learns the continuous dynamics of the myocardium from sequential images without increasing computational complexity. We evaluate the proposed method on two public datasets. The experimental results demonstrate that the proposed method quantitatively and qualitatively outperforms both conventional and state-of-the-art learning-based cardiac motion tracking methods. The code is available at https://github.com/yjh-0104/MCM.
>
---
#### [new 102] Constructing Ophthalmic MLLM for Positioning-diagnosis Collaboration Through Clinical Cognitive Chain Reasoning
- **分类: cs.AI; cs.CV; eess.IV**

- **简介: 该论文属于医学图像分析与多模态语言模型结合任务，旨在解决眼科诊断中跨模态理解不足、标注碎片化和推理逻辑不一致问题。作者构建了眼科专用模型FundusExpert及配套数据集FundusGen，通过临床认知链实现定位-诊断协同推理，提升模型准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2507.17539v1](http://arxiv.org/pdf/2507.17539v1)**

> **作者:** Xinyao Liu; Diping Song
>
> **摘要:** Multimodal large language models (MLLMs) demonstrate significant potential in the field of medical diagnosis. However, they face critical challenges in specialized domains such as ophthalmology, particularly the fragmentation of annotation granularity and inconsistencies in clinical reasoning logic, which hinder precise cross-modal understanding. This paper introduces FundusExpert, an ophthalmology-specific MLLM with integrated positioning-diagnosis reasoning capabilities, along with FundusGen, a dataset constructed through the intelligent Fundus-Engine system. Fundus-Engine automates localization and leverages MLLM-based semantic expansion to integrate global disease classification, local object detection, and fine-grained feature analysis within a single fundus image. Additionally, by constructing a clinically aligned cognitive chain, it guides the model to generate interpretable reasoning paths. FundusExpert, fine-tuned with instruction data from FundusGen, achieves the best performance in ophthalmic question-answering tasks, surpassing the average accuracy of the 40B MedRegA by 26.6%. It also excels in zero-shot report generation tasks, achieving a clinical consistency of 77.0%, significantly outperforming GPT-4o's 47.6%. Furthermore, we reveal a scaling law between data quality and model capability ($L \propto N^{0.068}$), demonstrating that the cognitive alignment annotations in FundusGen enhance data utilization efficiency. By integrating region-level localization with diagnostic reasoning chains, our work develops a scalable, clinically-aligned MLLM and explores a pathway toward bridging the visual-language gap in specific MLLMs. Our project can be found at https://github.com/MeteorElf/FundusExpert.
>
---
#### [new 103] Large Learning Rates Simultaneously Achieve Robustness to Spurious Correlations and Compressibility
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文属于机器学习任务，旨在解决模型在面对虚假相关性时缺乏鲁棒性以及模型压缩困难的问题。作者发现，使用大学习率可同时提升模型对虚假相关性的鲁棒性和压缩性，并展现出良好的表示特性，如特征不变性、类间分离和激活稀疏性。**

- **链接: [http://arxiv.org/pdf/2507.17748v1](http://arxiv.org/pdf/2507.17748v1)**

> **作者:** Melih Barsbey; Lucas Prieto; Stefanos Zafeiriou; Tolga Birdal
>
> **备注:** Accepted at ICCV 2025, 23 pages
>
> **摘要:** Robustness and resource-efficiency are two highly desirable properties for modern machine learning models. However, achieving them jointly remains a challenge. In this paper, we position high learning rates as a facilitator for simultaneously achieving robustness to spurious correlations and network compressibility. We demonstrate that large learning rates also produce desirable representation properties such as invariant feature utilization, class separation, and activation sparsity. Importantly, our findings indicate that large learning rates compare favorably to other hyperparameters and regularization methods, in consistently satisfying these properties in tandem. In addition to demonstrating the positive effect of large learning rates across diverse spurious correlation datasets, models, and optimizers, we also present strong evidence that the previously documented success of large learning rates in standard classification tasks is likely due to its effect on addressing hidden/rare spurious correlations in the training dataset.
>
---
#### [new 104] Joint Asymmetric Loss for Learning with Noisy Labels
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于学习噪声标签任务，旨在解决标签噪声导致模型性能下降的问题。作者提出了一种新的非对称损失函数AMSE，并将其引入到APL框架中，构建了联合非对称损失（JAL），以提升模型在噪声标签下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.17692v1](http://arxiv.org/pdf/2507.17692v1)**

> **作者:** Jialiang Wang; Xianming Liu; Xiong Zhou; Gangfeng Hu; Deming Zhai; Junjun Jiang; Xiangyang Ji
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Learning with noisy labels is a crucial task for training accurate deep neural networks. To mitigate label noise, prior studies have proposed various robust loss functions, particularly symmetric losses. Nevertheless, symmetric losses usually suffer from the underfitting issue due to the overly strict constraint. To address this problem, the Active Passive Loss (APL) jointly optimizes an active and a passive loss to mutually enhance the overall fitting ability. Within APL, symmetric losses have been successfully extended, yielding advanced robust loss functions. Despite these advancements, emerging theoretical analyses indicate that asymmetric losses, a new class of robust loss functions, possess superior properties compared to symmetric losses. However, existing asymmetric losses are not compatible with advanced optimization frameworks such as APL, limiting their potential and applicability. Motivated by this theoretical gap and the prospect of asymmetric losses, we extend the asymmetric loss to the more complex passive loss scenario and propose the Asymetric Mean Square Error (AMSE), a novel asymmetric loss. We rigorously establish the necessary and sufficient condition under which AMSE satisfies the asymmetric condition. By substituting the traditional symmetric passive loss in APL with our proposed AMSE, we introduce a novel robust loss framework termed Joint Asymmetric Loss (JAL). Extensive experiments demonstrate the effectiveness of our method in mitigating label noise. Code available at: https://github.com/cswjl/joint-asymmetric-loss
>
---
#### [new 105] On the Interaction of Compressibility and Adversarial Robustness
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文研究神经网络中压缩性与对抗鲁棒性之间的关系。它分析了不同形式的压缩（如神经元级稀疏性和谱压缩性）如何影响模型对对抗扰动的鲁棒性，揭示了压缩性可能引入敏感方向，使模型易受攻击。论文通过理论分析和实验验证，表明压缩与鲁棒性之间存在根本张力，并为设计高效且安全的模型提供了新思路。**

- **链接: [http://arxiv.org/pdf/2507.17725v1](http://arxiv.org/pdf/2507.17725v1)**

> **作者:** Melih Barsbey; Antônio H. Ribeiro; Umut Şimşekli; Tolga Birdal
>
> **摘要:** Modern neural networks are expected to simultaneously satisfy a host of desirable properties: accurate fitting to training data, generalization to unseen inputs, parameter and computational efficiency, and robustness to adversarial perturbations. While compressibility and robustness have each been studied extensively, a unified understanding of their interaction still remains elusive. In this work, we develop a principled framework to analyze how different forms of compressibility - such as neuron-level sparsity and spectral compressibility - affect adversarial robustness. We show that these forms of compression can induce a small number of highly sensitive directions in the representation space, which adversaries can exploit to construct effective perturbations. Our analysis yields a simple yet instructive robustness bound, revealing how neuron and spectral compressibility impact $L_\infty$ and $L_2$ robustness via their effects on the learned representations. Crucially, the vulnerabilities we identify arise irrespective of how compression is achieved - whether via regularization, architectural bias, or implicit learning dynamics. Through empirical evaluations across synthetic and realistic tasks, we confirm our theoretical predictions, and further demonstrate that these vulnerabilities persist under adversarial training and transfer learning, and contribute to the emergence of universal adversarial perturbations. Our findings show a fundamental tension between structured compressibility and robustness, and suggest new pathways for designing models that are both efficient and secure.
>
---
#### [new 106] DNT: a Deeply Normalized Transformer that can be trained by Momentum SGD
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于自然语言处理与深度学习优化任务，旨在解决Transformer模型难以使用动量SGD优化器训练的问题。论文提出DNT模型，通过深度归一化技术平衡梯度分布，使其适用于动量SGD训练，同时保持与AdamW相当的性能。**

- **链接: [http://arxiv.org/pdf/2507.17501v1](http://arxiv.org/pdf/2507.17501v1)**

> **作者:** Xianbiao Qi; Marco Chen; Wenjie Xiao; Jiaquan Ye; Yelin He; Chun-Guang Li; Zhouchen Lin
>
> **备注:** We have introduced a novel architecture, Deeply Normalized Transformer (DNT), which enables efficient training with vanilla momentum SGDW (mSGDW), achieving performance on par with AdamW-optimized Transformers
>
> **摘要:** Transformers have become the de facto backbone of modern deep learning, yet their training typically demands an advanced optimizer with adaptive learning rate like AdamW, rather than a momentum SGDW (mSGDW). Previous works show that it is mainly due to a heavy-tailed distribution of the gradients. In this paper, we introduce a Deeply Normalized Transformer (DNT), which is meticulously engineered to overcome this limitation enabling seamless training with vanilla mSGDW while yielding comparable performance to the Transformers trained via AdamW. To be specific, in DNT, we strategically integrate normalization techniques at proper positions in the Transformers to effectively modulate the Jacobian matrices of each layer, balance the influence of weights, activations, and their interactions, and thus enable the distributions of gradients concentrated. We provide both theoretical justifications of the normalization technique used in our DNT and extensive empirical evaluation on two popular Transformer architectures to validate that: a) DNT outperforms its counterparts (\ie, ViT and GPT), and b) DNT can be effectively trained with vanilla mSGDW.
>
---
#### [new 107] VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings
- **分类: cs.IR; cs.AI; cs.CV**

- **简介: 该论文属于多模态推荐任务，旨在解决现有视觉-语言模型在电商推荐中的对象级对齐弱、文本表示模糊和领域不匹配问题。论文提出VL-CLIP框架，结合视觉定位和大语言模型增强的CLIP嵌入，提升推荐准确性和效果。**

- **链接: [http://arxiv.org/pdf/2507.17080v1](http://arxiv.org/pdf/2507.17080v1)**

> **作者:** Ramin Giahi; Kehui Yao; Sriram Kollipara; Kai Zhao; Vahid Mirjalili; Jianpeng Xu; Topojoy Biswas; Evren Korpeoglu; Kannan Achan
>
> **备注:** Accepted at RecSys 2025; DOI:https://doi.org/10.1145/3705328.3748064
>
> **摘要:** Multimodal learning plays a critical role in e-commerce recommendation platforms today, enabling accurate recommendations and product understanding. However, existing vision-language models, such as CLIP, face key challenges in e-commerce recommendation systems: 1) Weak object-level alignment, where global image embeddings fail to capture fine-grained product attributes, leading to suboptimal retrieval performance; 2) Ambiguous textual representations, where product descriptions often lack contextual clarity, affecting cross-modal matching; and 3) Domain mismatch, as generic vision-language models may not generalize well to e-commerce-specific data. To address these limitations, we propose a framework, VL-CLIP, that enhances CLIP embeddings by integrating Visual Grounding for fine-grained visual understanding and an LLM-based agent for generating enriched text embeddings. Visual Grounding refines image representations by localizing key products, while the LLM agent enhances textual features by disambiguating product descriptions. Our approach significantly improves retrieval accuracy, multimodal retrieval effectiveness, and recommendation quality across tens of millions of items on one of the largest e-commerce platforms in the U.S., increasing CTR by 18.6%, ATC by 15.5%, and GMV by 4.0%. Additional experimental results show that our framework outperforms vision-language models, including CLIP, FashionCLIP, and GCL, in both precision and semantic alignment, demonstrating the potential of combining object-aware visual grounding and LLM-enhanced text representation for robust multimodal recommendations.
>
---
#### [new 108] Audio-Vision Contrastive Learning for Phonological Class Recognition
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于语音分类任务，旨在解决基于发音器官运动与语音信号的音系类别识别问题。作者提出了一种多模态深度学习框架，结合实时磁共振成像与语音信号，通过对比学习实现音视频融合，显著提升了分类性能。**

- **链接: [http://arxiv.org/pdf/2507.17682v1](http://arxiv.org/pdf/2507.17682v1)**

> **作者:** Daiqi Liu; Tomás Arias-Vergara; Jana Hutter; Andreas Maier; Paula Andrea Pérez-Toro
>
> **备注:** conference to TSD 2025
>
> **摘要:** Accurate classification of articulatory-phonological features plays a vital role in understanding human speech production and developing robust speech technologies, particularly in clinical contexts where targeted phonemic analysis and therapy can improve disease diagnosis accuracy and personalized rehabilitation. In this work, we propose a multimodal deep learning framework that combines real-time magnetic resonance imaging (rtMRI) and speech signals to classify three key articulatory dimensions: manner of articulation, place of articulation, and voicing. We perform classification on 15 phonological classes derived from the aforementioned articulatory dimensions and evaluate the system with four audio/vision configurations: unimodal rtMRI, unimodal audio signals, multimodal middle fusion, and contrastive learning-based audio-vision fusion. Experimental results on the USC-TIMIT dataset show that our contrastive learning-based approach achieves state-of-the-art performance, with an average F1-score of 0.81, representing an absolute increase of 0.23 over the unimodal baseline. The results confirm the effectiveness of contrastive representation learning for multimodal articulatory analysis. Our code and processed dataset will be made publicly available at https://github.com/DaE-plz/AC_Contrastive_Phonology to support future research.
>
---
#### [new 109] StreamME: Simplify 3D Gaussian Avatar within Live Stream
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于3D avatar重建任务，旨在实时从直播视频流中快速重建头像。论文提出StreamME方法，基于3D高斯点绘，实现无需预存数据的即时训练，提升面部表情适应速度，并通过简化点云策略优化效率，适用于VR、会议系统及动画等应用。**

- **链接: [http://arxiv.org/pdf/2507.17029v1](http://arxiv.org/pdf/2507.17029v1)**

> **作者:** Luchuan Song; Yang Zhou; Zhan Xu; Yi Zhou; Deepali Aneja; Chenliang Xu
>
> **备注:** 12 pages, 15 Figures
>
> **摘要:** We propose StreamME, a method focuses on fast 3D avatar reconstruction. The StreamME synchronously records and reconstructs a head avatar from live video streams without any pre-cached data, enabling seamless integration of the reconstructed appearance into downstream applications. This exceptionally fast training strategy, which we refer to as on-the-fly training, is central to our approach. Our method is built upon 3D Gaussian Splatting (3DGS), eliminating the reliance on MLPs in deformable 3DGS and relying solely on geometry, which significantly improves the adaptation speed to facial expression. To further ensure high efficiency in on-the-fly training, we introduced a simplification strategy based on primary points, which distributes the point clouds more sparsely across the facial surface, optimizing points number while maintaining rendering quality. Leveraging the on-the-fly training capabilities, our method protects the facial privacy and reduces communication bandwidth in VR system or online conference. Additionally, it can be directly applied to downstream application such as animation, toonify, and relighting. Please refer to our project page for more details: https://songluchuan.github.io/StreamME/.
>
---
#### [new 110] Dataset Distillation as Data Compression: A Rate-Utility Perspective
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于数据压缩与机器学习交叉任务，旨在解决大规模数据集带来的计算与存储压力。作者提出一种联合优化速率与效用的方法，通过合成数据的潜在编码与轻量解码器，实现高效的数据集蒸馏，显著提升压缩率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.17221v1](http://arxiv.org/pdf/2507.17221v1)**

> **作者:** Youneng Bao; Yiping Liu; Zhuo Chen; Yongsheng Liang; Mu Li; Kede Ma
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Driven by the ``scale-is-everything'' paradigm, modern machine learning increasingly demands ever-larger datasets and models, yielding prohibitive computational and storage requirements. Dataset distillation mitigates this by compressing an original dataset into a small set of synthetic samples, while preserving its full utility. Yet, existing methods either maximize performance under fixed storage budgets or pursue suitable synthetic data representations for redundancy removal, without jointly optimizing both objectives. In this work, we propose a joint rate-utility optimization method for dataset distillation. We parameterize synthetic samples as optimizable latent codes decoded by extremely lightweight networks. We estimate the Shannon entropy of quantized latents as the rate measure and plug any existing distillation loss as the utility measure, trading them off via a Lagrange multiplier. To enable fair, cross-method comparisons, we introduce bits per class (bpc), a precise storage metric that accounts for sample, label, and decoder parameter costs. On CIFAR-10, CIFAR-100, and ImageNet-128, our method achieves up to $170\times$ greater compression than standard distillation at comparable accuracy. Across diverse bpc budgets, distillation losses, and backbone architectures, our approach consistently establishes better rate-utility trade-offs.
>
---
#### [new 111] Mammo-Mamba: A Hybrid State-Space and Transformer Architecture with Sequential Mixture of Experts for Multi-View Mammography
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决乳腺癌早期检测中多视角乳腺X线图像分类效率与精度不足的问题。论文提出Mammo-Mamba模型，结合状态空间模型、Transformer注意力与专家机制，提升特征学习与计算效率，在CBIS-DDSM数据集上验证了性能优势。**

- **链接: [http://arxiv.org/pdf/2507.17662v1](http://arxiv.org/pdf/2507.17662v1)**

> **作者:** Farnoush Bayatmakou; Reza Taleei; Nicole Simone; Arash Mohammadi
>
> **摘要:** Breast cancer (BC) remains one of the leading causes of cancer-related mortality among women, despite recent advances in Computer-Aided Diagnosis (CAD) systems. Accurate and efficient interpretation of multi-view mammograms is essential for early detection, driving a surge of interest in Artificial Intelligence (AI)-powered CAD models. While state-of-the-art multi-view mammogram classification models are largely based on Transformer architectures, their computational complexity scales quadratically with the number of image patches, highlighting the need for more efficient alternatives. To address this challenge, we propose Mammo-Mamba, a novel framework that integrates Selective State-Space Models (SSMs), transformer-based attention, and expert-driven feature refinement into a unified architecture. Mammo-Mamba extends the MambaVision backbone by introducing the Sequential Mixture of Experts (SeqMoE) mechanism through its customized SecMamba block. The SecMamba is a modified MambaVision block that enhances representation learning in high-resolution mammographic images by enabling content-adaptive feature refinement. These blocks are integrated into the deeper stages of MambaVision, allowing the model to progressively adjust feature emphasis through dynamic expert gating, effectively mitigating the limitations of traditional Transformer models. Evaluated on the CBIS-DDSM benchmark dataset, Mammo-Mamba achieves superior classification performance across all key metrics while maintaining computational efficiency.
>
---
#### [new 112] Assessing Medical Training Skills via Eye and Head Movements
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于医疗技能评估任务，旨在通过眼动和头动数据客观评估临床培训效果。研究采集24名受试者在模拟接生训练中的生理数据，提取瞳孔反应率、注视时长、角速度等特征，发现头部特征在区分训练与非训练者上表现最佳（F1=0.85），表明眼动追踪技术可作为临床技能评估的辅助工具。**

- **链接: [http://arxiv.org/pdf/2507.16819v1](http://arxiv.org/pdf/2507.16819v1)**

> **作者:** Kayhan Latifzadeh; Luis A. Leiva; Klen Čopič Pucihar; Matjaž Kljun; Iztok Devetak; Lili Steblovnik
>
> **摘要:** We examined eye and head movements to gain insights into skill development in clinical settings. A total of 24 practitioners participated in simulated baby delivery training sessions. We calculated key metrics, including pupillary response rate, fixation duration, or angular velocity. Our findings indicate that eye and head tracking can effectively differentiate between trained and untrained practitioners, particularly during labor tasks. For example, head-related features achieved an F1 score of 0.85 and AUC of 0.86, whereas pupil-related features achieved F1 score of 0.77 and AUC of 0.85. The results lay the groundwork for computational models that support implicit skill assessment and training in clinical settings by using commodity eye-tracking glasses as a complementary device to more traditional evaluation methods such as subjective scores.
>
---
#### [new 113] Weak Links in LinkedIn: Enhancing Fake Profile Detection in the Age of LLMs
- **分类: cs.SI; cs.CV; cs.CY**

- **简介: 该论文属于虚假用户资料检测任务，旨在解决大型语言模型生成的逼真虚假LinkedIn资料难以被现有系统识别的问题。作者通过评估现有检测器效果，发现其对GPT生成资料检测率低，提出使用GPT辅助对抗训练增强检测器，并验证了结合数值与文本嵌入的方法效果最佳。**

- **链接: [http://arxiv.org/pdf/2507.16860v1](http://arxiv.org/pdf/2507.16860v1)**

> **作者:** Apoorva Gulati; Rajesh Kumar; Vinti Agarwal; Aditya Sharma
>
> **备注:** 10 pages, 3 figures, 1 table, accepted for publication at ASONAM 2025. https://sites.google.com/view/weaklinksinlinkedin/home
>
> **摘要:** Large Language Models (LLMs) have made it easier to create realistic fake profiles on platforms like LinkedIn. This poses a significant risk for text-based fake profile detectors. In this study, we evaluate the robustness of existing detectors against LLM-generated profiles. While highly effective in detecting manually created fake profiles (False Accept Rate: 6-7%), the existing detectors fail to identify GPT-generated profiles (False Accept Rate: 42-52%). We propose GPT-assisted adversarial training as a countermeasure, restoring the False Accept Rate to between 1-7% without impacting the False Reject Rates (0.5-2%). Ablation studies revealed that detectors trained on combined numerical and textual embeddings exhibit the highest robustness, followed by those using numerical-only embeddings, and lastly those using textual-only embeddings. Complementary analysis on the ability of prompt-based GPT-4Turbo and human evaluators affirms the need for robust automated detectors such as the one proposed in this study.
>
---
#### [new 114] SADA: Stability-guided Adaptive Diffusion Acceleration
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于生成模型加速任务，旨在解决扩散模型计算成本高的问题。论文提出SADA方法，通过稳定性引导的自适应稀疏策略，统一了步级和令牌级的稀疏决策，提升了采样速度并保持了生成质量。**

- **链接: [http://arxiv.org/pdf/2507.17135v1](http://arxiv.org/pdf/2507.17135v1)**

> **作者:** Ting Jiang; Yixiao Wang; Hancheng Ye; Zishan Shao; Jingwei Sun; Jingyang Zhang; Zekai Chen; Jianyi Zhang; Yiran Chen; Hai Li
>
> **备注:** Accepted and published by ICML 2025. Code is available at: https://github.com/Ting-Justin-Jiang/sada-icml
>
> **摘要:** Diffusion models have achieved remarkable success in generative tasks but suffer from high computational costs due to their iterative sampling process and quadratic attention costs. Existing training-free acceleration strategies that reduce per-step computation cost, while effectively reducing sampling time, demonstrate low faithfulness compared to the original baseline. We hypothesize that this fidelity gap arises because (a) different prompts correspond to varying denoising trajectory, and (b) such methods do not consider the underlying ODE formulation and its numerical solution. In this paper, we propose Stability-guided Adaptive Diffusion Acceleration (SADA), a novel paradigm that unifies step-wise and token-wise sparsity decisions via a single stability criterion to accelerate sampling of ODE-based generative models (Diffusion and Flow-matching). For (a), SADA adaptively allocates sparsity based on the sampling trajectory. For (b), SADA introduces principled approximation schemes that leverage the precise gradient information from the numerical ODE solver. Comprehensive evaluations on SD-2, SDXL, and Flux using both EDM and DPM++ solvers reveal consistent $\ge 1.8\times$ speedups with minimal fidelity degradation (LPIPS $\leq 0.10$ and FID $\leq 4.5$) compared to unmodified baselines, significantly outperforming prior methods. Moreover, SADA adapts seamlessly to other pipelines and modalities: It accelerates ControlNet without any modifications and speeds up MusicLDM by $1.8\times$ with $\sim 0.01$ spectrogram LPIPS.
>
---
#### [new 115] Explainable AI for Collaborative Assessment of 2D/3D Registration Quality
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于医学图像处理与人工智能交叉任务，旨在解决手术中2D/3D图像配准质量评估问题。现有算法虽先进但可能出错，仅靠可视化难以检测误差，影响手术安全。作者提出首个用于配准质量评估的可解释AI框架，并结合人类操作员进行多条件对比实验，评估AI与人类协作效果，探索可解释性对信任与决策的影响。**

- **链接: [http://arxiv.org/pdf/2507.17597v1](http://arxiv.org/pdf/2507.17597v1)**

> **作者:** Sue Min Cho; Alexander Do; Russell H. Taylor; Mathias Unberath
>
> **摘要:** As surgery embraces digital transformation--integrating sophisticated imaging, advanced algorithms, and robotics to support and automate complex sub-tasks--human judgment of system correctness remains a vital safeguard for patient safety. This shift introduces new "operator-type" roles tasked with verifying complex algorithmic outputs, particularly at critical junctures of the procedure, such as the intermediary check before drilling or implant placement. A prime example is 2D/3D registration, a key enabler of image-based surgical navigation that aligns intraoperative 2D images with preoperative 3D data. Although registration algorithms have advanced significantly, they occasionally yield inaccurate results. Because even small misalignments can lead to revision surgery or irreversible surgical errors, there is a critical need for robust quality assurance. Current visualization-based strategies alone have been found insufficient to enable humans to reliably detect 2D/3D registration misalignments. In response, we propose the first artificial intelligence (AI) framework trained specifically for 2D/3D registration quality verification, augmented by explainability features that clarify the model's decision-making. Our explainable AI (XAI) approach aims to enhance informed decision-making for human operators by providing a second opinion together with a rationale behind it. Through algorithm-centric and human-centered evaluations, we systematically compare four conditions: AI-only, human-only, human-AI, and human-XAI. Our findings reveal that while explainability features modestly improve user trust and willingness to override AI errors, they do not exceed the standalone AI in aggregate performance. Nevertheless, future work extending both the algorithmic design and the human-XAI collaboration elements holds promise for more robust quality assurance of 2D/3D registration.
>
---
#### [new 116] Harmonization in Magnetic Resonance Imaging: A Survey of Acquisition, Image-level, and Feature-level Methods
- **分类: eess.IV; cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像处理任务，旨在解决多中心MRI数据因设备或协议差异导致的“批次效应”问题。论文系统综述了图像采集、重建、图像级和特征级的谐和方法，重点分析了深度学习方法的应用，并总结了挑战与未来方向，以提升数据可比性和模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.16962v1](http://arxiv.org/pdf/2507.16962v1)**

> **作者:** Qinqin Yang; Firoozeh Shomal-Zadeh; Ali Gholipour
>
> **备注:** 20 pages, 6 figures, 2 tables
>
> **摘要:** Modern medical imaging technologies have greatly advanced neuroscience research and clinical diagnostics. However, imaging data collected across different scanners, acquisition protocols, or imaging sites often exhibit substantial heterogeneity, known as "batch effects" or "site effects". These non-biological sources of variability can obscure true biological signals, reduce reproducibility and statistical power, and severely impair the generalizability of learning-based models across datasets. Image harmonization aims to eliminate or mitigate such site-related biases while preserving meaningful biological information, thereby improving data comparability and consistency. This review provides a comprehensive overview of key concepts, methodological advances, publicly available datasets, current challenges, and future directions in the field of medical image harmonization, with a focus on magnetic resonance imaging (MRI). We systematically cover the full imaging pipeline, and categorize harmonization approaches into prospective acquisition and reconstruction strategies, retrospective image-level and feature-level methods, and traveling-subject-based techniques. Rather than providing an exhaustive survey, we focus on representative methods, with particular emphasis on deep learning-based approaches. Finally, we summarize the major challenges that remain and outline promising avenues for future research.
>
---
#### [new 117] CA-Cut: Crop-Aligned Cutout for Data Augmentation to Learn More Robust Under-Canopy Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于农业机器人视觉导航任务，旨在解决复杂农田环境下模型泛化能力不足的问题。作者提出了一种新的数据增强方法CA-Cut，在训练中对作物行附近的区域进行随机遮挡，以提升模型在遮挡情况下的语义关键点预测鲁棒性。实验表明该方法显著减少了预测误差，并增强了跨环境的适应能力。**

- **链接: [http://arxiv.org/pdf/2507.17727v1](http://arxiv.org/pdf/2507.17727v1)**

> **作者:** Robel Mamo; Taeyeong Choi
>
> **备注:** Accepted for publication at the 12th European Conference on Mobile Robots (ECMR 2025)
>
> **摘要:** State-of-the-art visual under-canopy navigation methods are designed with deep learning-based perception models to distinguish traversable space from crop rows. While these models have demonstrated successful performance, they require large amounts of training data to ensure reliability in real-world field deployment. However, data collection is costly, demanding significant human resources for in-field sampling and annotation. To address this challenge, various data augmentation techniques are commonly employed during model training, such as color jittering, Gaussian blur, and horizontal flip, to diversify training data and enhance model robustness. In this paper, we hypothesize that utilizing only these augmentation techniques may lead to suboptimal performance, particularly in complex under-canopy environments with frequent occlusions, debris, and non-uniform spacing of crops. Instead, we propose a novel augmentation method, so-called Crop-Aligned Cutout (CA-Cut) which masks random regions out in input images that are spatially distributed around crop rows on the sides to encourage trained models to capture high-level contextual features even when fine-grained information is obstructed. Our extensive experiments with a public cornfield dataset demonstrate that masking-based augmentations are effective for simulating occlusions and significantly improving robustness in semantic keypoint predictions for visual navigation. In particular, we show that biasing the mask distribution toward crop rows in CA-Cut is critical for enhancing both prediction accuracy and generalizability across diverse environments achieving up to a 36.9% reduction in prediction error. In addition, we conduct ablation studies to determine the number of masks, the size of each mask, and the spatial distribution of masks to maximize overall performance.
>
---
#### [new 118] InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉-语言-动作（VLA）建模任务，旨在解决现有模型在多模态推理与动作生成间的权衡问题。论文提出InstructVLA模型，通过VLA-IT训练方法，结合多模态数据与专家混合机制，提升任务理解与操作性能，实现更强的泛化与推理能力。**

- **链接: [http://arxiv.org/pdf/2507.17520v1](http://arxiv.org/pdf/2507.17520v1)**

> **作者:** Shuai Yang; Hao Li; Yilun Chen; Bin Wang; Yang Tian; Tai Wang; Hanqing Wang; Feng Zhao; Yiyi Liao; Jiangmiao Pang
>
> **备注:** 38 pages
>
> **摘要:** To operate effectively in the real world, robots must integrate multimodal reasoning with precise action generation. However, existing vision-language-action (VLA) models often sacrifice one for the other, narrow their abilities to task-specific manipulation data, and suffer catastrophic forgetting of pre-trained vision-language capabilities. To bridge this gap, we introduce InstructVLA, an end-to-end VLA model that preserves the flexible reasoning of large vision-language models (VLMs) while delivering leading manipulation performance. InstructVLA introduces a novel training paradigm, Vision-Language-Action Instruction Tuning (VLA-IT), which employs multimodal training with mixture-of-experts adaptation to jointly optimize textual reasoning and action generation on both standard VLM corpora and a curated 650K-sample VLA-IT dataset. On in-domain SimplerEnv tasks, InstructVLA achieves 30.5% improvement over SpatialVLA. To evaluate generalization, we introduce SimplerEnv-Instruct, an 80-task benchmark requiring closed-loop control and high-level instruction understanding, where it outperforms a fine-tuned OpenVLA by 92% and an action expert aided by GPT-4o by 29%. Additionally, InstructVLA surpasses baseline VLMs on multimodal tasks and exhibits inference-time scaling by leveraging textual reasoning to boost manipulation performance in both simulated and real-world settings. These results demonstrate InstructVLA's potential for bridging intuitive and steerable human-robot interaction with efficient policy learning.
>
---
## 更新

#### [replaced 001] Improving the Reasoning of Multi-Image Grounding in MLLMs via Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00748v2](http://arxiv.org/pdf/2507.00748v2)**

> **作者:** Bob Zhang; Haoran Li; Tao Zhang; Cilin Yan; Jiayin Cai; Yanbin Hao
>
> **备注:** 10 pages
>
> **摘要:** Recently, Multimodal Large Language Models (MLLMs) excel at visual grounding in single-image scenarios with textual references. However, their performance degrades when handling real-world applications that involve complex multi-image compositions and multi-modal instructions, revealing limitations in cross-image reasoning and generalization. To address these challenges, we adopt a Reinforcement Learning (RL) based post-training strategy to improve the reasoning of MLLMs in multi-image grounding tasks. Our approach begins with synthesizing high-quality chain-of-thought (CoT) data for cold-start initialization, followed by supervised fine-tuning (SFT) using low-rank adaptation (LoRA). The cold-start training stage enables the model to identify correct solutions. Subsequently, we perform rejection sampling using the merged SFT model to curate high-quality RL data and leverage rule-based RL to guide the model toward optimal reasoning paths. Extensive experimental results demonstrate the effectiveness of our approach, yielding improvements of +9.04% on MIG-Bench, +6.37% on MC-Bench, and +4.98% on several out-of-domain reasoning grounding benchmarks compared to the SFT baseline. Furthermore, our method exhibits strong generalization in multi-image perception, with gains of +3.1% and +2.4% over the base model on BLINK and MMIU benchmarks, respectively.
>
---
#### [replaced 002] NVS-SQA: Exploring Self-Supervised Quality Representation Learning for Neurally Synthesized Scenes without References
- **分类: cs.CV; cs.AI; cs.HC; cs.MM; eess.IV**

- **链接: [http://arxiv.org/pdf/2501.06488v2](http://arxiv.org/pdf/2501.06488v2)**

> **作者:** Qiang Qu; Yiran Shen; Xiaoming Chen; Yuk Ying Chung; Weidong Cai; Tongliang Liu
>
> **摘要:** Neural View Synthesis (NVS), such as NeRF and 3D Gaussian Splatting, effectively creates photorealistic scenes from sparse viewpoints, typically evaluated by quality assessment methods like PSNR, SSIM, and LPIPS. However, these full-reference methods, which compare synthesized views to reference views, may not fully capture the perceptual quality of neurally synthesized scenes (NSS), particularly due to the limited availability of dense reference views. Furthermore, the challenges in acquiring human perceptual labels hinder the creation of extensive labeled datasets, risking model overfitting and reduced generalizability. To address these issues, we propose NVS-SQA, a NSS quality assessment method to learn no-reference quality representations through self-supervision without reliance on human labels. Traditional self-supervised learning predominantly relies on the "same instance, similar representation" assumption and extensive datasets. However, given that these conditions do not apply in NSS quality assessment, we employ heuristic cues and quality scores as learning objectives, along with a specialized contrastive pair preparation process to improve the effectiveness and efficiency of learning. The results show that NVS-SQA outperforms 17 no-reference methods by a large margin (i.e., on average 109.5% in SRCC, 98.6% in PLCC, and 91.5% in KRCC over the second best) and even exceeds 16 full-reference methods across all evaluation metrics (i.e., 22.9% in SRCC, 19.1% in PLCC, and 18.6% in KRCC over the second best).
>
---
#### [replaced 003] MCA-LLaVA: Manhattan Causal Attention for Reducing Hallucination in Large Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09184v2](http://arxiv.org/pdf/2507.09184v2)**

> **作者:** Qiyan Zhao; Xiaofeng Zhang; Yiheng Li; Yun Xing; Xiaosong Yuan; Feilong Tang; Sinan Fan; Xuhang Chen; Xuyao Zhang; Dahan Wang
>
> **备注:** Accepted in ACM MM 2025
>
> **摘要:** Hallucinations pose a significant challenge in Large Vision Language Models (LVLMs), with misalignment between multimodal features identified as a key contributing factor. This paper reveals the negative impact of the long-term decay in Rotary Position Encoding (RoPE), used for positional modeling in LVLMs, on multimodal alignment. Concretely, under long-term decay, instruction tokens exhibit uneven perception of image tokens located at different positions within the two-dimensional space: prioritizing image tokens from the bottom-right region since in the one-dimensional sequence, these tokens are positionally closer to the instruction tokens. This biased perception leads to insufficient image-instruction interaction and suboptimal multimodal alignment. We refer to this phenomenon as image alignment bias. To enhance instruction's perception of image tokens at different spatial locations, we propose MCA-LLaVA, based on Manhattan distance, which extends the long-term decay to a two-dimensional, multi-directional spatial decay. MCA-LLaVA integrates the one-dimensional sequence order and two-dimensional spatial position of image tokens for positional modeling, mitigating hallucinations by alleviating image alignment bias. Experimental results of MCA-LLaVA across various hallucination and general benchmarks demonstrate its effectiveness and generality. The code can be accessed in https://github.com/ErikZ719/MCA-LLaVA.
>
---
#### [replaced 004] EndoControlMag: Robust Endoscopic Vascular Motion Magnification with Periodic Reference Resetting and Hierarchical Tissue-aware Dual-Mask Contro
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15292v3](http://arxiv.org/pdf/2507.15292v3)**

> **作者:** An Wang; Rulin Zhou; Mengya Xu; Yiru Ye; Longfei Gou; Yiting Chang; Hao Chen; Chwee Ming Lim; Jiankun Wang; Hongliang Ren
>
> **摘要:** Visualizing subtle vascular motions in endoscopic surgery is crucial for surgical precision and decision-making, yet remains challenging due to the complex and dynamic nature of surgical scenes. To address this, we introduce EndoControlMag, a training-free, Lagrangian-based framework with mask-conditioned vascular motion magnification tailored to endoscopic environments. Our approach features two key modules: a Periodic Reference Resetting (PRR) scheme that divides videos into short overlapping clips with dynamically updated reference frames to prevent error accumulation while maintaining temporal coherence, and a Hierarchical Tissue-aware Magnification (HTM) framework with dual-mode mask dilation. HTM first tracks vessel cores using a pretrained visual tracking model to maintain accurate localization despite occlusions and view changes. It then applies one of two adaptive softening strategies to surrounding tissues: motion-based softening that modulates magnification strength proportional to observed tissue displacement, or distance-based exponential decay that simulates biomechanical force attenuation. This dual-mode approach accommodates diverse surgical scenarios-motion-based softening excels with complex tissue deformations while distance-based softening provides stability during unreliable optical flow conditions. We evaluate EndoControlMag on our EndoVMM24 dataset spanning four different surgery types and various challenging scenarios, including occlusions, instrument disturbance, view changes, and vessel deformations. Quantitative metrics, visual assessments, and expert surgeon evaluations demonstrate that EndoControlMag significantly outperforms existing methods in both magnification accuracy and visual quality while maintaining robustness across challenging surgical conditions. The code, dataset, and video results are available at https://szupc.github.io/EndoControlMag/.
>
---
#### [replaced 005] Qffusion: Controllable Portrait Video Editing via Quadrant-Grid Attention Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.06438v2](http://arxiv.org/pdf/2501.06438v2)**

> **作者:** Maomao Li; Lijian Lin; Yunfei Liu; Ye Zhu; Yu Li
>
> **备注:** 19 pages
>
> **摘要:** This paper presents Qffusion, a dual-frame-guided framework for portrait video editing. Specifically, we consider a design principle of ``animation for editing'', and train Qffusion as a general animation framework from two still reference images while we can use it for portrait video editing easily by applying modified start and end frames as references during inference. Leveraging the powerful generative power of Stable Diffusion, we propose a Quadrant-grid Arrangement (QGA) scheme for latent re-arrangement, which arranges the latent codes of two reference images and that of four facial conditions into a four-grid fashion, separately. Then, we fuse features of these two modalities and use self-attention for both appearance and temporal learning, where representations at different times are jointly modeled under QGA. Our Qffusion can achieve stable video editing without additional networks or complex training stages, where only the input format of Stable Diffusion is modified. Further, we propose a Quadrant-grid Propagation (QGP) inference strategy, which enjoys a unique advantage on stable arbitrary-length video generation by processing reference and condition frames recursively. Through extensive experiments, Qffusion consistently outperforms state-of-the-art techniques on portrait video editing. Project page: https://qffusion.github.io/page/.
>
---
#### [replaced 006] Parasite: A Steganography-based Backdoor Attack Framework for Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.05815v2](http://arxiv.org/pdf/2504.05815v2)**

> **作者:** Jiahao Chen; Yu Pan; Yi Du; Chunkai Wu; Lin Wang
>
> **摘要:** Recently, the diffusion model has gained significant attention as one of the most successful image generation models, which can generate high-quality images by iteratively sampling noise. However, recent studies have shown that diffusion models are vulnerable to backdoor attacks, allowing attackers to enter input data containing triggers to activate the backdoor and generate their desired output. Existing backdoor attack methods primarily focused on target noise-to-image and text-to-image tasks, with limited work on backdoor attacks in image-to-image tasks. Furthermore, traditional backdoor attacks often rely on a single, conspicuous trigger to generate a fixed target image, lacking concealability and flexibility. To address these limitations, we propose a novel backdoor attack method called "Parasite" for image-to-image tasks in diffusion models, which not only is the first to leverage steganography for triggers hiding, but also allows attackers to embed the target content as a backdoor trigger to achieve a more flexible attack. "Parasite" as a novel attack method effectively bypasses existing detection frameworks to execute backdoor attacks. In our experiments, "Parasite" achieved a 0 percent backdoor detection rate against the mainstream defense frameworks. In addition, in the ablation study, we discuss the influence of different hiding coefficients on the attack results. You can find our code at https://anonymous.4open.science/r/Parasite-1715/.
>
---
#### [replaced 007] InceptionMamba: An Efficient Hybrid Network with Large Band Convolution and Bottleneck Mamba
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.08735v3](http://arxiv.org/pdf/2506.08735v3)**

> **作者:** Yuhang Wang; Jun Li; Zhijian Wu; Jifeng Shen; Jianhua Xu; Wankou Yang
>
> **摘要:** Within the family of convolutional neural networks, InceptionNeXt has shown excellent competitiveness in image classification and a number of downstream tasks. Built on parallel one-dimensional strip convolutions, however, it suffers from limited ability of capturing spatial dependencies along different dimensions and fails to fully explore spatial modeling in local neighborhood. Besides, inherent locality constraints of convolution operations are detrimental to effective global context modeling. To overcome these limitations, we propose a novel backbone architecture termed InceptionMamba in this study. More specifically, the traditional one-dimensional strip convolutions are replaced by orthogonal band convolutions in our InceptionMamba to achieve cohesive spatial modeling. Furthermore, global contextual modeling can be achieved via a bottleneck Mamba module, facilitating enhanced cross-channel information fusion and enlarged receptive field. Extensive evaluations on classification and various downstream tasks demonstrate that the proposed InceptionMamba achieves state-of-the-art performance with superior parameter and computational efficiency. The source code will be available at https://github.com/Wake1021/InceptionMamba.
>
---
#### [replaced 008] Vascular Segmentation of Functional Ultrasound Images using Deep Learning
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.22365v2](http://arxiv.org/pdf/2410.22365v2)**

> **作者:** Hana Sebia; Thomas Guyet; Mickaël Pereira; Marco Valdebenito; Hugues Berry; Benjamin Vidal
>
> **摘要:** Segmentation of medical images is a fundamental task with numerous applications. While MRI, CT, and PET modalities have significantly benefited from deep learning segmentation techniques, more recent modalities, like functional ultrasound (fUS), have seen limited progress. fUS is a non invasive imaging method that measures changes in cerebral blood volume (CBV) with high spatio-temporal resolution. However, distinguishing arterioles from venules in fUS is challenging due to opposing blood flow directions within the same pixel. Ultrasound localization microscopy (ULM) can enhance resolution by tracking microbubble contrast agents but is invasive, and lacks dynamic CBV quantification. In this paper, we introduce the first deep learning-based segmentation tool for fUS images, capable of differentiating signals from different vascular compartments, based on ULM automatic annotation and enabling dynamic CBV quantification. We evaluate various UNet architectures on fUS images of rat brains, achieving competitive segmentation performance, with 90% accuracy, a 71% F1 score, and an IoU of 0.59, using only 100 temporal frames from a fUS stack. These results are comparable to those from tubular structure segmentation in other imaging modalities. Additionally, models trained on resting-state data generalize well to images captured during visual stimulation, highlighting robustness. This work offers a non-invasive, cost-effective alternative to ULM, enhancing fUS data interpretation and improving understanding of vessel function. Our pipeline shows high linear correlation coefficients between signals from predicted and actual compartments in both cortical and deeper regions, showcasing its ability to accurately capture blood flow dynamics.
>
---
#### [replaced 009] Mapping of Weed Management Methods in Orchards using Sentinel-2 and PlanetScope Data
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.19991v2](http://arxiv.org/pdf/2504.19991v2)**

> **作者:** Ioannis Kontogiorgakis; Iason Tsardanidis; Dimitrios Bormpoudakis; Ilias Tsoumas; Dimitra A. Loka; Christos Noulas; Alexandros Tsitouras; Charalampos Kontoes
>
> **备注:** Accepted for 2025 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025)
>
> **摘要:** Effective weed management is crucial for improving agricultural productivity, as weeds compete with crops for vital resources like nutrients and water. Accurate maps of weed management methods are essential for policymakers to assess farmer practices, evaluate impacts on vegetation health, biodiversity, and climate, as well as ensure compliance with policies and subsidies. However, monitoring weed management methods is challenging as they commonly rely on ground-based field surveys, which are often costly, time-consuming and subject to delays. In order to tackle this problem, we leverage earth observation data and Machine Learning (ML). Specifically, we developed separate ML models using Sentinel-2 and PlanetScope satellite time series data, respectively, to classify four distinct weed management methods (Mowing, Tillage, Chemical-spraying, and No practice) in orchards. The findings demonstrate the potential of ML-driven remote sensing to enhance the efficiency and accuracy of weed management mapping in orchards.
>
---
#### [replaced 010] Temporally Consistent Dynamic Scene Graphs: An End-to-End Approach for Action Tracklet Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.02808v2](http://arxiv.org/pdf/2412.02808v2)**

> **作者:** Raphael Ruschel; Md Awsafur Rahman; Hardik Prajapati; Suya You; B. S. Manjuanth
>
> **摘要:** Understanding video content is pivotal for advancing real-world applications like activity recognition, autonomous systems, and human-computer interaction. While scene graphs are adept at capturing spatial relationships between objects in individual frames, extending these representations to capture dynamic interactions across video sequences remains a significant challenge. To address this, we present TCDSG, Temporally Consistent Dynamic Scene Graphs, an innovative end-to-end framework that detects, tracks, and links subject-object relationships across time, generating action tracklets, temporally consistent sequences of entities and their interactions. Our approach leverages a novel bipartite matching mechanism, enhanced by adaptive decoder queries and feedback loops, ensuring temporal coherence and robust tracking over extended sequences. This method not only establishes a new benchmark by achieving over 60% improvement in temporal recall@k on the Action Genome, OpenPVSG, and MEVA datasets but also pioneers the augmentation of MEVA with persistent object ID annotations for comprehensive tracklet generation. By seamlessly integrating spatial and temporal dynamics, our work sets a new standard in multi-frame video analysis, opening new avenues for high-impact applications in surveillance, autonomous navigation, and beyond.
>
---
#### [replaced 011] PoemTale Diffusion: Minimising Information Loss in Poem to Image Generation with Multi-Stage Prompt Refinement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13708v2](http://arxiv.org/pdf/2507.13708v2)**

> **作者:** Sofia Jamil; Bollampalli Areen Reddy; Raghvendra Kumar; Sriparna Saha; Koustava Goswami; K. J. Joseph
>
> **备注:** ECAI 2025
>
> **摘要:** Recent advancements in text-to-image diffusion models have achieved remarkable success in generating realistic and diverse visual content. A critical factor in this process is the model's ability to accurately interpret textual prompts. However, these models often struggle with creative expressions, particularly those involving complex, abstract, or highly descriptive language. In this work, we introduce a novel training-free approach tailored to improve image generation for a unique form of creative language: poetic verse, which frequently features layered, abstract, and dual meanings. Our proposed PoemTale Diffusion approach aims to minimise the information that is lost during poetic text-to-image conversion by integrating a multi stage prompt refinement loop into Language Models to enhance the interpretability of poetic texts. To support this, we adapt existing state-of-the-art diffusion models by modifying their self-attention mechanisms with a consistent self-attention technique to generate multiple consistent images, which are then collectively used to convey the poem's meaning. Moreover, to encourage research in the field of poetry, we introduce the P4I (PoemForImage) dataset, consisting of 1111 poems sourced from multiple online and offline resources. We engaged a panel of poetry experts for qualitative assessments. The results from both human and quantitative evaluations validate the efficacy of our method and contribute a novel perspective to poem-to-image generation with enhanced information capture in the generated images.
>
---
#### [replaced 012] DMS-Net:Dual-Modal Multi-Scale Siamese Network for Binocular Fundus Image Classification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.18046v2](http://arxiv.org/pdf/2504.18046v2)**

> **作者:** Guohao Huo; Zibo Lin; Zitong Wang; Ruiting Dai; Hao Tang
>
> **摘要:** Ophthalmic diseases pose a significant global health challenge, yet traditional diagnosis methods and existing single-eye deep learning approaches often fail to account for binocular pathological correlations. To address this, we propose DMS-Net, a dual-modal multi-scale Siamese network for binocular fundus image classification. Our framework leverages weight-shared Siamese ResNet-152 backbones to extract deep semantic features from paired fundus images. To tackle challenges such as lesion boundary ambiguity and scattered pathological distributions, we introduce a Multi-Scale Context-Aware Module (MSCAM) that integrates adaptive pooling and attention mechanisms for multi-resolution feature aggregation. Additionally, a Dual-Modal Feature Fusion (DMFF) module enhances cross-modal interaction through spatial-semantic recalibration and bidirectional attention, effectively combining global context and local edge features. Evaluated on the ODIR-5K dataset, DMS-Net achieves state-of-the-art performance with 82.9% accuracy, 84.5% recall, and 83.2% Cohen's kappa, demonstrating superior capability in detecting symmetric pathologies and advancing clinical decision-making for ocular diseases.
>
---
#### [replaced 013] How Well Does GPT-4o Understand Vision? Evaluating Multimodal Foundation Models on Standard Computer Vision Tasks
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.01955v2](http://arxiv.org/pdf/2507.01955v2)**

> **作者:** Rahul Ramachandran; Ali Garjani; Roman Bachmann; Andrei Atanov; Oğuzhan Fatih Kar; Amir Zamir
>
> **备注:** Project page at https://fm-vision-evals.epfl.ch/
>
> **摘要:** Multimodal foundation models, such as GPT-4o, have recently made remarkable progress, but it is not clear where exactly these models stand in terms of understanding vision. In this paper, we benchmark the performance of popular multimodal foundation models (GPT-4o, o4-mini, Gemini 1.5 Pro and Gemini 2.0 Flash, Claude 3.5 Sonnet, Qwen2-VL, Llama 3.2) on standard computer vision tasks (semantic segmentation, object detection, image classification, depth and surface normal prediction) using established datasets (e.g., COCO, ImageNet and its variants, etc). The main challenges to performing this are: 1) most models are trained to output text and cannot natively express versatile domains, such as segments or 3D geometry, and 2) many leading models are proprietary and accessible only at an API level, i.e., there is no weight access to adapt them. We address these challenges by translating standard vision tasks into equivalent text-promptable and API-compatible tasks via prompt chaining to create a standardized benchmarking framework. We observe that 1) the models are not close to the state-of-the-art specialist models at any task. However, 2) they are respectable generalists; this is remarkable as they are presumably trained on primarily image-text-based tasks. 3) They perform semantic tasks notably better than geometric ones. 4) While the prompt-chaining techniques affect performance, better models exhibit less sensitivity to prompt variations. 5) GPT-4o performs the best among non-reasoning models, securing the top position in 4 out of 6 tasks, 6) reasoning models, e.g. o3, show improvements in geometric tasks, and 7) a preliminary analysis of models with native image generation, like the latest GPT-4o, shows they exhibit quirks like hallucinations and spatial misalignments.
>
---
#### [replaced 014] APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation
- **分类: cs.NE; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.14270v2](http://arxiv.org/pdf/2507.14270v2)**

> **作者:** Ravin Kumar
>
> **备注:** 10 pages, 2 figures, 1 table, and GitHub repository for the source code
>
> **摘要:** We propose the APTx Neuron, a novel, unified neural computation unit that integrates non-linear activation and linear transformation into a single trainable expression. The APTx Neuron is derived from the APTx activation function, thereby eliminating the need for separate activation layers and making the architecture both computationally efficient and elegant. The proposed neuron follows the functional form $y = \sum_{i=1}^{n} ((\alpha_i + \tanh(\beta_i x_i)) \cdot \gamma_i x_i) + \delta$, where all parameters $\alpha_i$, $\beta_i$, $\gamma_i$, and $\delta$ are trainable. We validate our APTx Neuron-based architecture on the MNIST dataset, achieving up to 96.69% test accuracy in just 20 epochs using approximately 332K trainable parameters. The results highlight the superior expressiveness and computational efficiency of the APTx Neuron compared to traditional neurons, pointing toward a new paradigm in unified neuron design and the architectures built upon it.
>
---
#### [replaced 015] Application of YOLOv8 in monocular downward multiple Car Target detection
- **分类: cs.CV; cs.AI; I.4.8; I.2.10**

- **链接: [http://arxiv.org/pdf/2505.10016v2](http://arxiv.org/pdf/2505.10016v2)**

> **作者:** Shijie Lyu
>
> **备注:** This submission included authors who did not consent to the submission. The paper is being withdrawn until authorship issues are resolved
>
> **摘要:** Autonomous driving technology is progressively transforming traditional car driving methods, marking a significant milestone in modern transportation. Object detection serves as a cornerstone of autonomous systems, playing a vital role in enhancing driving safety, enabling autonomous functionality, improving traffic efficiency, and facilitating effective emergency responses. However, current technologies such as radar for environmental perception, cameras for road perception, and vehicle sensor networks face notable challenges, including high costs, vulnerability to weather and lighting conditions, and limited resolution.To address these limitations, this paper presents an improved autonomous target detection network based on YOLOv8. By integrating structural reparameterization technology, a bidirectional pyramid structure network model, and a novel detection pipeline into the YOLOv8 framework, the proposed approach achieves highly efficient and precise detection of multi-scale, small, and remote objects. Experimental results demonstrate that the enhanced model can effectively detect both large and small objects with a detection accuracy of 65%, showcasing significant advancements over traditional methods.This improved model holds substantial potential for real-world applications and is well-suited for autonomous driving competitions, such as the Formula Student Autonomous China (FSAC), particularly excelling in scenarios involving single-target and small-object detection.
>
---
#### [replaced 016] LEGO Co-builder: Exploring Fine-Grained Vision-Language Modeling for Multimodal LEGO Assembly Assistants
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05515v2](http://arxiv.org/pdf/2507.05515v2)**

> **作者:** Haochen Huang; Jiahuan Pei; Mohammad Aliannejadi; Xin Sun; Moonisa Ahsan; Chuang Yu; Zhaochun Ren; Pablo Cesar; Junxiao Wang
>
> **备注:** This version has been anonymized for double-blind review
>
> **摘要:** Vision-language models (VLMs) are facing the challenges of understanding and following multimodal assembly instructions, particularly when fine-grained spatial reasoning and precise object state detection are required. In this work, we explore LEGO Co-builder, a hybrid benchmark combining real-world LEGO assembly logic with programmatically generated multimodal scenes. The dataset captures stepwise visual states and procedural instructions, allowing controlled evaluation of instruction-following, object detection, and state detection. We introduce a unified framework and assess leading VLMs such as GPT-4o, Gemini, and Qwen-VL, under zero-shot and fine-tuned settings. Our results reveal that even advanced models like GPT-4o struggle with fine-grained assembly tasks, with a maximum F1 score of just 40.54\% on state detection, highlighting gaps in fine-grained visual understanding. We release the benchmark, codebase, and generation pipeline to support future research on multimodal assembly assistants grounded in real-world workflows.
>
---
#### [replaced 017] Human-Activity AGV Quality Assessment: A Benchmark Dataset and an Objective Evaluation Metric
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16619v3](http://arxiv.org/pdf/2411.16619v3)**

> **作者:** Zhichao Zhang; Wei Sun; Xinyue Li; Yunhao Li; Qihang Ge; Jun Jia; Zicheng Zhang; Zhongpeng Ji; Fengyu Sun; Shangling Jui; Xiongkuo Min; Guangtao Zhai
>
> **备注:** Accepted by ACMMM 2025
>
> **摘要:** AI-driven video generation techniques have made significant progress in recent years. However, AI-generated videos (AGVs) involving human activities often exhibit substantial visual and semantic distortions, hindering the practical application of video generation technologies in real-world scenarios. To address this challenge, we conduct a pioneering study on human activity AGV quality assessment, focusing on visual quality evaluation and the identification of semantic distortions. First, we construct the AI-Generated Human activity Video Quality Assessment (Human-AGVQA) dataset, consisting of 6,000 AGVs derived from 15 popular text-to-video (T2V) models using 400 text prompts that describe diverse human activities. We conduct a subjective study to evaluate the human appearance quality, action continuity quality, and overall video quality of AGVs, and identify semantic issues of human body parts. Based on Human-AGVQA, we benchmark the performance of T2V models and analyze their strengths and weaknesses in generating different categories of human activities. Second, we develop an objective evaluation metric, named AI-Generated Human activity Video Quality metric (GHVQ), to automatically analyze the quality of human activity AGVs. GHVQ systematically extracts human-focused quality features, AI-generated content-aware quality features, and temporal continuity features, making it a comprehensive and explainable quality metric for human activity AGVs. The extensive experimental results show that GHVQ outperforms existing quality metrics on the Human-AGVQA dataset by a large margin, demonstrating its efficacy in assessing the quality of human activity AGVs. The Human-AGVQA dataset and GHVQ metric will be released at https://github.com/zczhang-sjtu/GHVQ.git.
>
---
#### [replaced 018] BoxFusion: Reconstruction-Free Open-Vocabulary 3D Object Detection via Real-Time Multi-View Box Fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.15610v2](http://arxiv.org/pdf/2506.15610v2)**

> **作者:** Yuqing Lan; Chenyang Zhu; Zhirui Gao; Jiazhao Zhang; Yihan Cao; Renjiao Yi; Yijie Wang; Kai Xu
>
> **备注:** Project page: https://lanlan96.github.io/BoxFusion/
>
> **摘要:** Open-vocabulary 3D object detection has gained significant interest due to its critical applications in autonomous driving and embodied AI. Existing detection methods, whether offline or online, typically rely on dense point cloud reconstruction, which imposes substantial computational overhead and memory constraints, hindering real-time deployment in downstream tasks. To address this, we propose a novel reconstruction-free online framework tailored for memory-efficient and real-time 3D detection. Specifically, given streaming posed RGB-D video input, we leverage Cubify Anything as a pre-trained visual foundation model (VFM) for single-view 3D object detection by bounding boxes, coupled with CLIP to capture open-vocabulary semantics of detected objects. To fuse all detected bounding boxes across different views into a unified one, we employ an association module for correspondences of multi-views and an optimization module to fuse the 3D bounding boxes of the same instance predicted in multi-views. The association module utilizes 3D Non-Maximum Suppression (NMS) and a box correspondence matching module, while the optimization module uses an IoU-guided efficient random optimization technique based on particle filtering to enforce multi-view consistency of the 3D bounding boxes while minimizing computational complexity. Extensive experiments on ScanNetV2 and CA-1M datasets demonstrate that our method achieves state-of-the-art performance among online methods. Benefiting from this novel reconstruction-free paradigm for 3D object detection, our method exhibits great generalization abilities in various scenarios, enabling real-time perception even in environments exceeding 1000 square meters.
>
---
#### [replaced 019] Diffusion-Guided Knowledge Distillation for Weakly-Supervised Low-Light Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07578v2](http://arxiv.org/pdf/2507.07578v2)**

> **作者:** Chunyan Wang; Dong Zhang; Jinhui Tang
>
> **备注:** Accepted by ACM Multimedia
>
> **摘要:** Weakly-supervised semantic segmentation aims to assign category labels to each pixel using weak annotations, significantly reducing manual annotation costs. Although existing methods have achieved remarkable progress in well-lit scenarios, their performance significantly degrades in low-light environments due to two fundamental limitations: severe image quality degradation (e.g., low contrast, noise, and color distortion) and the inherent constraints of weak supervision. These factors collectively lead to unreliable class activation maps and semantically ambiguous pseudo-labels, ultimately compromising the model's ability to learn discriminative feature representations. To address these problems, we propose Diffusion-Guided Knowledge Distillation for Weakly-Supervised Low-light Semantic Segmentation (DGKD-WLSS), a novel framework that synergistically combines Diffusion-Guided Knowledge Distillation (DGKD) with Depth-Guided Feature Fusion (DGF2). DGKD aligns normal-light and low-light features via diffusion-based denoising and knowledge distillation, while DGF2 integrates depth maps as illumination-invariant geometric priors to enhance structural feature learning. Extensive experiments demonstrate the effectiveness of DGKD-WLSS, which achieves state-of-the-art performance in weakly supervised semantic segmentation tasks under low-light conditions. The source codes have been released at:https://github.com/ChunyanWang1/DGKD-WLSS.
>
---
#### [replaced 020] AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.23956v3](http://arxiv.org/pdf/2503.23956v3)**

> **作者:** Kai Huang; Hao Zou; Bochen Wang; Ye Xi; Zhen Xie; Hao Wang
>
> **摘要:** Recent advancements in Large Visual Language Models (LVLMs) have gained significant attention due to their remarkable reasoning capabilities and proficiency in generalization. However, processing a large number of visual tokens and generating long-context outputs impose substantial computational overhead, leading to excessive demands for key-value (KV) cache. To address this critical bottleneck, we propose AirCache, a novel KV cache compression method aimed at accelerating LVLMs inference. This work systematically investigates the correlations between visual and textual tokens within the attention mechanisms of LVLMs. Our empirical analysis reveals considerable redundancy in cached visual tokens, wherein strategically eliminating these tokens preserves model performance while significantly accelerating context generation. Inspired by these findings, we introduce an elite observation window for assessing the importance of visual components in the KV cache, focusing on stable inter-modal relevancy modeling with enhanced multi-perspective consistency. Additionally, we develop an adaptive layer-wise budget allocation strategy that capitalizes on the strength and skewness of token importance distribution, showcasing superior efficiency compared to uniform allocation. Comprehensive evaluations across multiple LVLMs and benchmarks demonstrate that our method achieves comparable performance to the full cache while retaining only 10% of visual KV cache, thereby reducing decoding latency by 29% to 66% across various batch size and prompt length of inputs. Notably, as cache retention rates decrease, our method exhibits increasing performance advantages over existing approaches.
>
---
#### [replaced 021] UniCUE: Unified Recognition and Generation Framework for Chinese Cued Speech Video-to-Speech Generation
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.04134v2](http://arxiv.org/pdf/2506.04134v2)**

> **作者:** Jinting Wang; Shan Yang; Li Liu
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Cued Speech (CS) enhances lipreading through hand coding, providing precise speech perception support for the hearing-impaired. CS Video-to-Speech generation (CSV2S) task aims to convert the CS visual expressions (CS videos) of hearing-impaired individuals into comprehensible speech signals. Direct generation of speech from CS video (called single CSV2S) yields poor performance due to insufficient CS data. Current research mostly focuses on CS Recognition (CSR), which convert video content into linguistic text. Based on this, one straightforward way of CSV2S is to combine CSR with a Text-to-Speech system. This combined architecture relies on text as an intermediate medium for stepwise cross-modal alignment, which may lead to error propagation and temporal misalignment between speech and video dynamics. To address these challenges, we propose a novel approach that directly generates speech from CS videos without relying on intermediate text. Building upon this, we propose UniCUE, the first unified framework for CSV2S, whose core innovation lies in the integration of the CSR task that provides fine-grained visual-semantic information to facilitate speech generation from CS videos. More precisely, (1) a novel fine-grained semantic alignment pool to ensure precise mapping between visual features and speech contents; (2) a VisioPhonetic adapter to bridge cross-task representations, ensuring seamless compatibility between two distinct tasks (i.e., CSV2S and CSR); (3) a pose-aware visual processor is introduced to enhance fine-grained spatiotemporal correlations between lip and hand movements in CS video. Experiments on our new established Chinese CS dataset show that our UniCUE achieves state-of-the-art performance across various metrics.
>
---
#### [replaced 022] Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22334v2](http://arxiv.org/pdf/2505.22334v2)**

> **作者:** Lai Wei; Yuting Li; Kaipeng Zheng; Chen Wang; Yue Wang; Linghe Kong; Lichao Sun; Weiran Huang
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated impressive chain-of-thought reasoning capabilities, with reinforcement learning (RL) playing a crucial role in this progress. While "aha moment" patterns--where models exhibit self-correction through reflection--are often attributed to emergent properties from RL, we first demonstrate that these patterns exist in multimodal LLMs (MLLMs) prior to RL training but may not necessarily correlate with improved reasoning performance. Building on these insights, we present a comprehensive study on enhancing multimodal reasoning through a two-stage approach: (1) supervised fine-tuning (SFT) as a cold start with structured chain-of-thought reasoning patterns, followed by (2) reinforcement learning via GRPO to further refine these capabilities. Our extensive experiments show that this combined approach consistently outperforms both SFT-only and RL-only methods across challenging multimodal reasoning benchmarks. The resulting models achieve state-of-the-art performance among open-source MLLMs at both 3B and 7B scales, with our 7B model showing substantial improvements over base models (e.g., 66.3 %$\rightarrow$73.4 % on MathVista, 62.9 %$\rightarrow$70.4 % on We-Math) and our 3B model achieving performance competitive with several 7B models. Overall, this work provides practical guidance for building advanced multimodal reasoning models. Our code is available at https://github.com/waltonfuture/RL-with-Cold-Start.
>
---
#### [replaced 023] Att-Adapter: A Robust and Precise Domain-Specific Multi-Attributes T2I Diffusion Adapter via Conditional Variational Autoencoder
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.11937v3](http://arxiv.org/pdf/2503.11937v3)**

> **作者:** Wonwoong Cho; Yan-Ying Chen; Matthew Klenk; David I. Inouye; Yanxia Zhang
>
> **备注:** ICCV'25, The project page is available at https://tri-mac.github.io/att-adapter/
>
> **摘要:** Text-to-Image (T2I) Diffusion Models have achieved remarkable performance in generating high quality images. However, enabling precise control of continuous attributes, especially multiple attributes simultaneously, in a new domain (e.g., numeric values like eye openness or car width) with text-only guidance remains a significant challenge. To address this, we introduce the Attribute (Att) Adapter, a novel plug-and-play module designed to enable fine-grained, multi-attributes control in pretrained diffusion models. Our approach learns a single control adapter from a set of sample images that can be unpaired and contain multiple visual attributes. The Att-Adapter leverages the decoupled cross attention module to naturally harmonize the multiple domain attributes with text conditioning. We further introduce Conditional Variational Autoencoder (CVAE) to the Att-Adapter to mitigate overfitting, matching the diverse nature of the visual world. Evaluations on two public datasets show that Att-Adapter outperforms all LoRA-based baselines in controlling continuous attributes. Additionally, our method enables a broader control range and also improves disentanglement across multiple attributes, surpassing StyleGAN-based techniques. Notably, Att-Adapter is flexible, requiring no paired synthetic data for training, and is easily scalable to multiple attributes within a single model.
>
---
#### [replaced 024] Transformer-Based Auxiliary Loss for Face Recognition Across Age Variations
- **分类: cs.CV; I.5.2**

- **链接: [http://arxiv.org/pdf/2412.02198v3](http://arxiv.org/pdf/2412.02198v3)**

> **作者:** Pritesh Prakash; S Umamaheswaran
>
> **备注:** Face Recognition for Age-variant Datasets
>
> **摘要:** Aging presents a significant challenge in face recognition, as changes in skin texture and tone can alter facial features over time, making it particularly difficult to compare images of the same individual taken years apart, such as in long-term identification scenarios. Transformer networks have the strength to preserve sequential spatial relationships caused by aging effect. This paper presents a technique for loss evaluation that uses a transformer network as an additive loss in the face recognition domain. The standard metric loss function typically takes the final embedding of the main CNN backbone as its input. Here, we employ a transformer-metric loss, a combined approach that integrates both transformer-loss and metric-loss. This research intends to analyze the transformer behavior on the convolution output when the CNN outcome is arranged in a sequential vector. These sequential vectors have the potential to overcome the texture or regional structure referred to as wrinkles or sagging skin affected by aging. The transformer encoder takes input from the contextual vectors obtained from the final convolution layer of the network. The learned features can be more age-invariant, complementing the discriminative power of the standard metric loss embedding. With this technique, we use transformer loss with various base metric-loss functions to evaluate the effect of the combined loss functions. We observe that such a configuration allows the network to achieve SoTA results in LFW and age-variant datasets (CA-LFW and AgeDB). This research expands the role of transformers in the machine vision domain and opens new possibilities for exploring transformers as a loss function.
>
---
#### [replaced 025] BadHMP: Backdoor Attack against Human Motion Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.19638v2](http://arxiv.org/pdf/2409.19638v2)**

> **作者:** Chaohui Xu; Si Wang; Chip-Hong Chang
>
> **摘要:** Precise future human motion prediction over sub-second horizons from past observations is crucial for various safety-critical applications. To date, only a few studies have examined the vulnerability of skeleton-based neural networks to evasion and backdoor attacks. In this paper, we propose BadHMP, a novel backdoor attack that targets specifically human motion prediction tasks. Our approach involves generating poisoned training samples by embedding a localized backdoor trigger in one limb of the skeleton, causing selected joints to follow predefined motion in historical time steps. Subsequently, the future sequences are globally modified that all the joints move following the target trajectories. Our carefully designed backdoor triggers and targets guarantee the smoothness and naturalness of the poisoned samples, making them stealthy enough to evade detection by the model trainer while keeping the poisoned model unobtrusive in terms of prediction fidelity to untainted sequences. The target sequences can be successfully activated by the designed input sequences even with a low poisoned sample injection ratio. Experimental results on two datasets (Human3.6M and CMU-Mocap) and two network architectures (LTD and HRI) demonstrate the high-fidelity, effectiveness, and stealthiness of BadHMP. Robustness of our attack against fine-tuning defense is also verified.
>
---
#### [replaced 026] Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.01630v2](http://arxiv.org/pdf/2507.01630v2)**

> **作者:** Yuxiao Wang; Yu Lei; Zhenao Wei; Weiying Xue; Xinyu Jiang; Nan Zhuang; Qi Liu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** The task of Human-Object conTact (HOT) detection involves identifying the specific areas of the human body that are touching objects. Nevertheless, current models are restricted to just one type of image, often leading to too much segmentation in areas with little interaction, and struggling to maintain category consistency within specific regions. To tackle this issue, a HOT framework, termed \textbf{P3HOT}, is proposed, which blends \textbf{P}rompt guidance and human \textbf{P}roximal \textbf{P}erception. To begin with, we utilize a semantic-driven prompt mechanism to direct the network's attention towards the relevant regions based on the correlation between image and text. Then a human proximal perception mechanism is employed to dynamically perceive key depth range around the human, using learnable parameters to effectively eliminate regions where interactions are not expected. Calculating depth resolves the uncertainty of the overlap between humans and objects in a 2D perspective, providing a quasi-3D viewpoint. Moreover, a Regional Joint Loss (RJLoss) has been created as a new loss to inhibit abnormal categories in the same area. A new evaluation metric called ``AD-Acc.'' is introduced to address the shortcomings of existing methods in addressing negative samples. Comprehensive experimental results demonstrate that our approach achieves state-of-the-art performance in four metrics across two benchmark datasets. Specifically, our model achieves an improvement of \textbf{0.7}$\uparrow$, \textbf{2.0}$\uparrow$, \textbf{1.6}$\uparrow$, and \textbf{11.0}$\uparrow$ in SC-Acc., mIoU, wIoU, and AD-Acc. metrics, respectively, on the HOT-Annotated dataset. The sources code are available at https://github.com/YuxiaoWang-AI/P3HOT.
>
---
#### [replaced 027] Flexible Coded Distributed Convolution Computing for Enhanced Straggler Resilience and Numerical Stability in Distributed CNNs
- **分类: cs.DC; cs.AI; cs.CV; cs.IT; cs.LG; math.IT**

- **链接: [http://arxiv.org/pdf/2411.01579v2](http://arxiv.org/pdf/2411.01579v2)**

> **作者:** Shuo Tan; Rui Liu; Xuesong Han; XianLei Long; Kai Wan; Linqi Song; Yong Li
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Deploying Convolutional Neural Networks (CNNs) on resource-constrained devices necessitates efficient management of computational resources, often via distributed environments susceptible to latency from straggler nodes. This paper introduces the Flexible Coded Distributed Convolution Computing (FCDCC) framework to enhance straggler resilience and numerical stability in distributed CNNs. We extend Coded Distributed Computing (CDC) with Circulant and Rotation Matrix Embedding (CRME) which was originally proposed for matrix multiplication to high-dimensional tensor convolution. For the proposed scheme, referred to as the Numerically Stable Coded Tensor Convolution (NSCTC) scheme, we also propose two new coded partitioning schemes: Adaptive-Padding Coded Partitioning (APCP) for the input tensor and Kernel-Channel Coded Partitioning (KCCP) for the filter tensor. These strategies enable linear decomposition of tensor convolutions and encoding them into CDC subtasks, combining model parallelism with coded redundancy for robust and efficient execution. Theoretical analysis identifies an optimal trade-off between communication and storage costs. Empirical results validate the framework's effectiveness in computational efficiency, straggler resilience, and scalability across various CNN architectures.
>
---
#### [replaced 028] Rethinking Range-View LiDAR Segmentation in Adverse Weather
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08979v2](http://arxiv.org/pdf/2506.08979v2)**

> **作者:** Longyu Yang; Lu Zhang; Jun Liu; Yap-Peng Tan; Heng Tao Shen; Xiaofeng Zhu; Ping Hu
>
> **摘要:** LiDAR segmentation has emerged as an important task to enrich scene perception and understanding. Range-view-based methods have gained popularity due to their high computational efficiency and compatibility with real-time deployment. However, their generalized performance under adverse weather conditions remains underexplored, limiting their reliability in real-world environments. In this work, we identify and analyze the unique challenges that affect the generalization of range-view LiDAR segmentation in severe weather. To address these challenges, we propose a modular and lightweight framework that enhances robustness without altering the core architecture of existing models. Our method reformulates the initial stem block of standard range-view networks into two branches to process geometric attributes and reflectance intensity separately. Specifically, a Geometric Abnormality Suppression (GAS) module reduces the influence of weather-induced spatial noise, and a Reflectance Distortion Calibration (RDC) module corrects reflectance distortions through memory-guided adaptive instance normalization. The processed features are then fused and passed to the original segmentation pipeline. Extensive experiments on different benchmarks and baseline models demonstrate that our approach significantly improves generalization to adverse weather with minimal inference overhead, offering a practical and effective solution for real-world LiDAR segmentation.
>
---
#### [replaced 029] Infinite Video Understanding
- **分类: cs.CV; cs.AI; cs.IR; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.09068v2](http://arxiv.org/pdf/2507.09068v2)**

> **作者:** Dell Zhang; Xiangyu Chen; Jixiang Luo; Mengxi Jia; Changzhi Sun; Ruilong Ren; Jingren Liu; Hao Sun; Xuelong Li
>
> **摘要:** The rapid advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have ushered in remarkable progress in video understanding. However, a fundamental challenge persists: effectively processing and comprehending video content that extends beyond minutes or hours. While recent efforts like Video-XL-2 have demonstrated novel architectural solutions for extreme efficiency, and advancements in positional encoding such as HoPE and VideoRoPE++ aim to improve spatio-temporal understanding over extensive contexts, current state-of-the-art models still encounter significant computational and memory constraints when faced with the sheer volume of visual tokens from lengthy sequences. Furthermore, maintaining temporal coherence, tracking complex events, and preserving fine-grained details over extended periods remain formidable hurdles, despite progress in agentic reasoning systems like Deep Video Discovery. This position paper posits that a logical, albeit ambitious, next frontier for multimedia research is Infinite Video Understanding -- the capability for models to continuously process, understand, and reason about video data of arbitrary, potentially never-ending duration. We argue that framing Infinite Video Understanding as a blue-sky research objective provides a vital north star for the multimedia, and the wider AI, research communities, driving innovation in areas such as streaming architectures, persistent memory mechanisms, hierarchical and adaptive representations, event-centric reasoning, and novel evaluation paradigms. Drawing inspiration from recent work on long/ultra-long video understanding and several closely related fields, we outline the core challenges and key research directions towards achieving this transformative capability.
>
---
#### [replaced 030] Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography
- **分类: eess.IV; cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2409.10876v4](http://arxiv.org/pdf/2409.10876v4)**

> **作者:** Tianao Li; Manxiu Cui; Cheng Ma; Emma Alexander
>
> **备注:** Accepted to IEEE/CVF International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Photoacoustic computed tomography (PACT) is a non-invasive imaging modality, similar to ultrasound, with wide-ranging medical applications. Conventional PACT images are degraded by wavefront distortion caused by the heterogeneous speed of sound (SOS) in tissue. Accounting for these effects can improve image quality and provide medically useful information, but measuring the SOS directly is burdensome and the existing joint reconstruction method is computationally expensive. Traditional supervised learning techniques are currently inaccessible in this data-starved domain. In this work, we introduce an efficient, self-supervised joint reconstruction method that recovers SOS and high-quality images for ring array PACT systems. To solve this semi-blind inverse problem, we parametrize the SOS using either a pixel grid or a neural field (NF) and update it directly by backpropagating the gradients through a differentiable imaging forward model. Our method removes SOS aberrations more accurately and 35x faster than the current SOTA. We demonstrate the success of our method quantitatively in simulation and qualitatively on experimentally-collected and in vivo data. Our code and synthetic numerical phantoms are available on our project page: https://lukeli0425.github.io/Coord-SoS-PACT/.
>
---
#### [replaced 031] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18079v3](http://arxiv.org/pdf/2505.18079v3)**

> **作者:** Xiaoyi Zhang; Zhaoyang Jia; Zongyu Guo; Jiahao Li; Bin Li; Houqiang Li; Yan Lu
>
> **备注:** V3 draft. Under review
>
> **摘要:** Long-form video understanding presents significant challenges due to extensive temporal-spatial complexity and the difficulty of question answering under such extended contexts. While Large Language Models (LLMs) have demonstrated considerable advancements in video analysis capabilities and long context handling, they continue to exhibit limitations when processing information-dense hour-long videos. To overcome such limitations, we propose the Deep Video Discovery agent to leverage an agentic search strategy over segmented video clips. Different from previous video agents manually designing a rigid workflow, our approach emphasizes the autonomous nature of agents. By providing a set of search-centric tools on multi-granular video database, our DVD agent leverages the advanced reasoning capability of LLM to plan on its current observation state, strategically selects tools, formulates appropriate parameters for actions, and iteratively refines its internal reasoning in light of the gathered information. We perform comprehensive evaluation on multiple long video understanding benchmarks that demonstrates the advantage of the entire system design. Our DVD agent achieves SOTA performance, significantly surpassing prior works by a large margin on the challenging LVBench dataset. Comprehensive ablation studies and in-depth tool analyses are also provided, yielding insights to further advance intelligent agents tailored for long-form video understanding tasks. The code has been released in https://github.com/microsoft/DeepVideoDiscovery.
>
---
#### [replaced 032] Rethinking Occlusion in FER: A Semantic-Aware Perspective and Go Beyond
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15401v2](http://arxiv.org/pdf/2507.15401v2)**

> **作者:** Huiyu Zhai; Xingxing Yang; Yalan Ye; Chenyang Li; Bin Fan; Changze Li
>
> **摘要:** Facial expression recognition (FER) is a challenging task due to pervasive occlusion and dataset biases. Especially when facial information is partially occluded, existing FER models struggle to extract effective facial features, leading to inaccurate classifications. In response, we present ORSANet, which introduces the following three key contributions: First, we introduce auxiliary multi-modal semantic guidance to disambiguate facial occlusion and learn high-level semantic knowledge, which is two-fold: 1) we introduce semantic segmentation maps as dense semantics prior to generate semantics-enhanced facial representations; 2) we introduce facial landmarks as sparse geometric prior to mitigate intrinsic noises in FER, such as identity and gender biases. Second, to facilitate the effective incorporation of these two multi-modal priors, we customize a Multi-scale Cross-interaction Module (MCM) to adaptively fuse the landmark feature and semantics-enhanced representations within different scales. Third, we design a Dynamic Adversarial Repulsion Enhancement Loss (DARELoss) that dynamically adjusts the margins of ambiguous classes, further enhancing the model's ability to distinguish similar expressions. We further construct the first occlusion-oriented FER dataset to facilitate specialized robustness analysis on various real-world occlusion conditions, dubbed Occlu-FER. Extensive experiments on both public benchmarks and Occlu-FER demonstrate that our proposed ORSANet achieves SOTA recognition performance. Code is publicly available at https://github.com/Wenyuzhy/ORSANet-master.
>
---
#### [replaced 033] MRI-CORE: A Foundation Model for Magnetic Resonance Imaging
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.12186v2](http://arxiv.org/pdf/2506.12186v2)**

> **作者:** Haoyu Dong; Yuwen Chen; Hanxue Gu; Nicholas Konz; Yaqian Chen; Qihang Li; Maciej A. Mazurowski
>
> **备注:** 36 pages, under review
>
> **摘要:** The widespread use of Magnetic Resonance Imaging (MRI) in combination with deep learning shows promise for many high-impact automated diagnostic and prognostic tools. However, training new models requires large amounts of labeled data, a challenge due to high cost of precise annotations and data privacy. To address this issue, we introduce the MRI-CORE, a vision foundation model trained using more than 6 million slices from over 110 thousand MRI volumes across 18 body locations. Our experiments show notable improvements in performance over state-of-the-art methods in 13 data-restricted segmentation tasks, as well as in image classification, and zero-shot segmentation, showing the strong potential of MRI-CORE to enable data-efficient development of artificial intelligence models. We also present data on which strategies yield most useful foundation models and a novel analysis relating similarity between pre-training and downstream task data with transfer learning performance. Our model is publicly available with a permissive license.
>
---
#### [replaced 034] Visual-Language Model Knowledge Distillation Method for Image Quality Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15680v3](http://arxiv.org/pdf/2507.15680v3)**

> **作者:** Yongkang Hou; Jiarun Song
>
> **摘要:** Image Quality Assessment (IQA) is a core task in computer vision. Multimodal methods based on vision-language models, such as CLIP, have demonstrated exceptional generalization capabilities in IQA tasks. To address the issues of excessive parameter burden and insufficient ability to identify local distorted features in CLIP for IQA, this study proposes a visual-language model knowledge distillation method aimed at guiding the training of models with architectural advantages using CLIP's IQA knowledge. First, quality-graded prompt templates were designed to guide CLIP to output quality scores. Then, CLIP is fine-tuned to enhance its capabilities in IQA tasks. Finally, a modality-adaptive knowledge distillation strategy is proposed to achieve guidance from the CLIP teacher model to the student model. Our experiments were conducted on multiple IQA datasets, and the results show that the proposed method significantly reduces model complexity while outperforming existing IQA methods, demonstrating strong potential for practical deployment.
>
---
#### [replaced 035] Latent Diffusion Models with Masked AutoEncoders
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09984v2](http://arxiv.org/pdf/2507.09984v2)**

> **作者:** Junho Lee; Jeongwoo Shin; Hyungwook Choi; Joonseok Lee
>
> **摘要:** In spite of the remarkable potential of Latent Diffusion Models (LDMs) in image generation, the desired properties and optimal design of the autoencoders have been underexplored. In this work, we analyze the role of autoencoders in LDMs and identify three key properties: latent smoothness, perceptual compression quality, and reconstruction quality. We demonstrate that existing autoencoders fail to simultaneously satisfy all three properties, and propose Variational Masked AutoEncoders (VMAEs), taking advantage of the hierarchical features maintained by Masked AutoEncoders. We integrate VMAEs into the LDM framework, introducing Latent Diffusion Models with Masked AutoEncoders (LDMAEs).
>
---
#### [replaced 036] RALAD: Bridging the Real-to-Sim Domain Gap in Autonomous Driving with Retrieval-Augmented Learning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.12296v3](http://arxiv.org/pdf/2501.12296v3)**

> **作者:** Jiacheng Zuo; Haibo Hu; Zikang Zhou; Yufei Cui; Ziquan Liu; Jianping Wang; Nan Guan; Jin Wang; Chun Jason Xue
>
> **摘要:** In the pursuit of robust autonomous driving systems, models trained on real-world datasets often struggle to adapt to new environments, particularly when confronted with corner cases such as extreme weather conditions. Collecting these corner cases in the real world is non-trivial, which necessitates the use of simulators for validation. However,the high computational cost and the domain gap in data distribution have hindered the seamless transition between real and simulated driving scenarios. To tackle this challenge, we propose Retrieval-Augmented Learning for Autonomous Driving (RALAD), a novel framework designed to bridge the real-to-sim gap at a low cost. RALAD features three primary designs, including (1) domain adaptation via an enhanced Optimal Transport (OT) method that accounts for both individual and grouped image distances, (2) a simple and unified framework that can be applied to various models, and (3) efficient fine-tuning techniques that freeze the computationally expensive layers while maintaining robustness. Experimental results demonstrate that RALAD compensates for the performance degradation in simulated environments while maintaining accuracy in real-world scenarios across three different models. Taking Cross View as an example, the mIOU and mAP metrics in real-world scenarios remain stable before and after RALAD fine-tuning, while in simulated environments,the mIOU and mAP metrics are improved by 10.30% and 12.29%, respectively. Moreover, the re-training cost of our approach is reduced by approximately 88.1%. Our code is available at https://github.com/JiachengZuo/RALAD.git.
>
---
#### [replaced 037] SFNet: A Spatial-Frequency Domain Deep Learning Network for Efficient Alzheimer's Disease Diagnosis
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16267v2](http://arxiv.org/pdf/2507.16267v2)**

> **作者:** Xinyue Yang; Meiliang Liu; Yunfang Xu; Xiaoxiao Yang; Zhengye Si; Zijin Li; Zhiwen Zhao
>
> **摘要:** Alzheimer's disease (AD) is a progressive neurodegenerative disorder that predominantly affects the elderly population and currently has no cure. Magnetic Resonance Imaging (MRI), as a non-invasive imaging technique, is essential for the early diagnosis of AD. MRI inherently contains both spatial and frequency information, as raw signals are acquired in the frequency domain and reconstructed into spatial images via the Fourier transform. However, most existing AD diagnostic models extract features from a single domain, limiting their capacity to fully capture the complex neuroimaging characteristics of the disease. While some studies have combined spatial and frequency information, they are mostly confined to 2D MRI, leaving the potential of dual-domain analysis in 3D MRI unexplored. To overcome this limitation, we propose Spatio-Frequency Network (SFNet), the first end-to-end deep learning framework that simultaneously leverages spatial and frequency domain information to enhance 3D MRI-based AD diagnosis. SFNet integrates an enhanced dense convolutional network to extract local spatial features and a global frequency module to capture global frequency-domain representations. Additionally, a novel multi-scale attention module is proposed to further refine spatial feature extraction. Experiments on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset demonstrate that SFNet outperforms existing baselines and reduces computational overhead in classifying cognitively normal (CN) and AD, achieving an accuracy of 95.1%.
>
---
#### [replaced 038] Spatial Frequency Modulation for Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.11893v2](http://arxiv.org/pdf/2507.11893v2)**

> **作者:** Linwei Chen; Ying Fu; Lin Gu; Dezhi Zheng; Jifeng Dai
>
> **备注:** Accept by TPAMI 2025
>
> **摘要:** High spatial frequency information, including fine details like textures, significantly contributes to the accuracy of semantic segmentation. However, according to the Nyquist-Shannon Sampling Theorem, high-frequency components are vulnerable to aliasing or distortion when propagating through downsampling layers such as strided-convolution. Here, we propose a novel Spatial Frequency Modulation (SFM) that modulates high-frequency features to a lower frequency before downsampling and then demodulates them back during upsampling. Specifically, we implement modulation through adaptive resampling (ARS) and design a lightweight add-on that can densely sample the high-frequency areas to scale up the signal, thereby lowering its frequency in accordance with the Frequency Scaling Property. We also propose Multi-Scale Adaptive Upsampling (MSAU) to demodulate the modulated feature and recover high-frequency information through non-uniform upsampling This module further improves segmentation by explicitly exploiting information interaction between densely and sparsely resampled areas at multiple scales. Both modules can seamlessly integrate with various architectures, extending from convolutional neural networks to transformers. Feature visualization and analysis confirm that our method effectively alleviates aliasing while successfully retaining details after demodulation. Finally, we validate the broad applicability and effectiveness of SFM by extending it to image classification, adversarial robustness, instance segmentation, and panoptic segmentation tasks. The code is available at https://github.com/Linwei-Chen/SFM.
>
---
#### [replaced 039] JEDI: The Force of Jensen-Shannon Divergence in Disentangling Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.19166v2](http://arxiv.org/pdf/2505.19166v2)**

> **作者:** Eric Tillmann Bill; Enis Simsar; Thomas Hofmann
>
> **摘要:** We introduce JEDI, a test-time adaptation method that enhances subject separation and compositional alignment in diffusion models without requiring retraining or external supervision. JEDI operates by minimizing semantic entanglement in attention maps using a novel Jensen-Shannon divergence based objective. To improve efficiency, we leverage adversarial optimization, reducing the number of updating steps required. JEDI is model-agnostic and applicable to architectures such as Stable Diffusion 1.5 and 3.5, consistently improving prompt alignment and disentanglement in complex scenes. Additionally, JEDI provides a lightweight, CLIP-free disentanglement score derived from internal attention distributions, offering a principled benchmark for compositional alignment under test-time conditions. Code and results are available at https://ericbill21.github.io/JEDI/.
>
---
#### [replaced 040] A novel approach to navigate the taxonomic hierarchy to address the Open-World Scenarios in Medicinal Plant Classification
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.17289v3](http://arxiv.org/pdf/2502.17289v3)**

> **作者:** Soumen Sinha; Tanisha Rana; Rahul Roy
>
> **备注:** We want to do some modifications and add more experiments
>
> **摘要:** In this article, we propose a novel approach for plant hierarchical taxonomy classification by posing the problem as an open class problem. It is observed that existing methods for medicinal plant classification often fail to perform hierarchical classification and accurately identifying unknown species, limiting their effectiveness in comprehensive plant taxonomy classification. Thus we address the problem of unknown species classification by assigning it best hierarchical labels. We propose a novel method, which integrates DenseNet121, Multi-Scale Self-Attention (MSSA) and cascaded classifiers for hierarchical classification. The approach systematically categorizes medicinal plants at multiple taxonomic levels, from phylum to species, ensuring detailed and precise classification. Using multi scale space attention, the model captures both local and global contextual information from the images, improving the distinction between similar species and the identification of new ones. It uses attention scores to focus on important features across multiple scales. The proposed method provides a solution for hierarchical classification, showcasing superior performance in identifying both known and unknown species. The model was tested on two state-of-art datasets with and without background artifacts and so that it can be deployed to tackle real word application. We used unknown species for testing our model. For unknown species the model achieved an average accuracy of 83.36%, 78.30%, 60.34% and 43.32% for predicting correct phylum, class, order and family respectively. Our proposed model size is almost four times less than the existing state of the art methods making it easily deploy able in real world application.
>
---
#### [replaced 041] FE-UNet: Frequency Domain Enhanced U-Net for Low-Frequency Information-Rich Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.03829v2](http://arxiv.org/pdf/2502.03829v2)**

> **作者:** Guohao Huo; Ruiting Dai; Ling Shao; Jinliang Liu; Hao Tang
>
> **摘要:** In deep-sea exploration and surgical robotics scenarios, environmental lighting and device resolution limitations often cause high-frequency feature attenuation. Addressing the differences in frequency band sensitivity between CNNs and the human visual system (mid-frequency sensitivity with low-frequency sensitivity surpassing high-frequency), we experimentally quantified the CNN contrast sensitivity function and proposed a wavelet adaptive spectrum fusion (WASF) method inspired by biological vision mechanisms to balance cross-frequency image features. Furthermore, we designed a perception frequency block (PFB) that integrates WASF to enhance frequency-domain feature extraction. Based on this, we developed the FE-UNet model, which employs a SAM2 backbone network and incorporates fine-tuned Hiera-Large modules to ensure segmentation accuracy while improving generalization capability. Experiments demonstrate that FE-UNet achieves state-of-the-art performance in cross-domain tasks such as marine organism segmentation and polyp segmentation, showcasing robust adaptability and significant application potential.
>
---
#### [replaced 042] MoDA: Multi-modal Diffusion Architecture for Talking Head Generation
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03256v2](http://arxiv.org/pdf/2507.03256v2)**

> **作者:** Xinyang Li; Gen Li; Zhihui Lin; Yichen Qian; GongXin Yao; Weinan Jia; Aowen Wang; Weihua Chen; Fan Wang
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Talking head generation with arbitrary identities and speech audio remains a crucial problem in the realm of the virtual metaverse. Recently, diffusion models have become a popular generative technique in this field with their strong generation capabilities. However, several challenges remain for diffusion-based methods: 1) inefficient inference and visual artifacts caused by the implicit latent space of Variational Auto-Encoders (VAE), which complicates the diffusion process; 2) a lack of authentic facial expressions and head movements due to inadequate multi-modal information fusion. In this paper, MoDA handles these challenges by: 1) defining a joint parameter space that bridges motion generation and neural rendering, and leveraging flow matching to simplify diffusion learning; 2) introducing a multi-modal diffusion architecture to model the interaction among noisy motion, audio, and auxiliary conditions, enhancing overall facial expressiveness. In addition, a coarse-to-fine fusion strategy is employed to progressively integrate different modalities, ensuring effective feature fusion. Experimental results demonstrate that MoDA improves video diversity, realism, and efficiency, making it suitable for real-world applications. Project Page: https://lixinyyang.github.io/MoDA.github.io/
>
---
#### [replaced 043] Fine-Grained Alignment and Noise Refinement for Compositional Text-to-Image Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.06506v2](http://arxiv.org/pdf/2503.06506v2)**

> **作者:** Amir Mohammad Izadi; Seyed Mohammad Hadi Hosseini; Soroush Vafaie Tabar; Ali Abdollahi; Armin Saghafian; Mahdieh Soleymani Baghshah
>
> **摘要:** Text-to-image generative models have made significant advancements in recent years; however, accurately capturing intricate details in textual prompts-such as entity missing, attribute binding errors, and incorrect relationships remains a formidable challenge. In response, we present an innovative, training-free method that directly addresses these challenges by incorporating tailored objectives to account for textual constraints. Unlike layout-based approaches that enforce rigid structures and limit diversity, our proposed approach offers a more flexible arrangement of the scene by imposing just the extracted constraints from the text, without any unnecessary additions. These constraints are formulated as losses-entity missing, entity mixing, attribute binding, and spatial relationships-integrated into a unified loss that is applied in the first generation stage. Furthermore, we introduce a feedback-driven system for fine-grained initial noise refinement. This system integrates a verifier that evaluates the generated image, identifies inconsistencies, and provides corrective feedback. Leveraging this feedback, our refinement method first targets the unmet constraints by refining the faulty attention maps caused by initial noise, through the optimization of selective losses associated with these constraints. Subsequently, our unified loss function is reapplied to proceed the second generation phase. Experimental results demonstrate that our method, relying solely on our proposed objective functions, significantly enhances compositionality, achieving a 24% improvement in human evaluation and a 25% gain in spatial relationships. Furthermore, our fine-grained noise refinement proves effective, boosting performance by up to 5%. Code is available at \href{https://github.com/hadi-hosseini/noise-refinement}{https://github.com/hadi-hosseini/noise-refinement}.
>
---
#### [replaced 044] TaoAvatar: Real-Time Lifelike Full-Body Talking Avatars for Augmented Reality via 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.17032v2](http://arxiv.org/pdf/2503.17032v2)**

> **作者:** Jianchuan Chen; Jingchuan Hu; Gaige Wang; Zhonghua Jiang; Tiansong Zhou; Zhiwen Chen; Chengfei Lv
>
> **备注:** Accepted by CVPR 2025 (Highlight), project page: https://PixelAI-Team.github.io/TaoAvatar
>
> **摘要:** Realistic 3D full-body talking avatars hold great potential in AR, with applications ranging from e-commerce live streaming to holographic communication. Despite advances in 3D Gaussian Splatting (3DGS) for lifelike avatar creation, existing methods struggle with fine-grained control of facial expressions and body movements in full-body talking tasks. Additionally, they often lack sufficient details and cannot run in real-time on mobile devices. We present TaoAvatar, a high-fidelity, lightweight, 3DGS-based full-body talking avatar driven by various signals. Our approach starts by creating a personalized clothed human parametric template that binds Gaussians to represent appearances. We then pre-train a StyleUnet-based network to handle complex pose-dependent non-rigid deformation, which can capture high-frequency appearance details but is too resource-intensive for mobile devices. To overcome this, we "bake" the non-rigid deformations into a lightweight MLP-based network using a distillation technique and develop blend shapes to compensate for details. Extensive experiments show that TaoAvatar achieves state-of-the-art rendering quality while running in real-time across various devices, maintaining 90 FPS on high-definition stereo devices such as the Apple Vision Pro.
>
---
#### [replaced 045] Optimizing against Infeasible Inclusions from Data for Semantic Segmentation through Morphology
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.14672v4](http://arxiv.org/pdf/2408.14672v4)**

> **作者:** Shamik Basu; Luc Van Gool; Christos Sakaridis
>
> **摘要:** State-of-the-art semantic segmentation models are typically optimized in a data-driven fashion, minimizing solely per-pixel or per-segment classification objectives on their training data. This purely data-driven paradigm often leads to absurd segmentations, especially when the domain of input images is shifted from the one encountered during training. For instance, state-of-the-art models may assign the label "road" to a segment that is included by another segment that is respectively labeled as "sky". However, the ground truth of the existing dataset at hand dictates that such inclusion is not feasible. Our method, Infeasible Semantic Inclusions (InSeIn), first extracts explicit inclusion constraints that govern spatial class relations from the semantic segmentation training set at hand in an offline, data-driven fashion, and then enforces a morphological yet differentiable loss that penalizes violations of these constraints during training to promote prediction feasibility. InSeIn is a light-weight plug-and-play method, constitutes a novel step towards minimizing infeasible semantic inclusions in the predictions of learned segmentation models, and yields consistent and significant performance improvements over diverse state-of-the-art networks across the ADE20K, Cityscapes, and ACDC datasets. https://github.com/SHAMIK-97/InSeIn/tree/main
>
---
#### [replaced 046] DeepShade: Enable Shade Simulation by Text-conditioned Image Generation
- **分类: cs.CV; cs.CY; 68T45, 68U10, 62H35; I.2.10; I.4.8; I.5.1**

- **链接: [http://arxiv.org/pdf/2507.12103v2](http://arxiv.org/pdf/2507.12103v2)**

> **作者:** Longchao Da; Xiangrui Liu; Mithun Shivakoti; Thirulogasankar Pranav Kutralingam; Yezhou Yang; Hua Wei
>
> **备注:** 7pages, 4 figures. Accepted to IJCAI 2025
>
> **摘要:** Heatwaves pose a significant threat to public health, especially as global warming intensifies. However, current routing systems (e.g., online maps) fail to incorporate shade information due to the difficulty of estimating shades directly from noisy satellite imagery and the limited availability of training data for generative models. In this paper, we address these challenges through two main contributions. First, we build an extensive dataset covering diverse longitude-latitude regions, varying levels of building density, and different urban layouts. Leveraging Blender-based 3D simulations alongside building outlines, we capture building shadows under various solar zenith angles throughout the year and at different times of day. These simulated shadows are aligned with satellite images, providing a rich resource for learning shade patterns. Second, we propose the DeepShade, a diffusion-based model designed to learn and synthesize shade variations over time. It emphasizes the nuance of edge features by jointly considering RGB with the Canny edge layer, and incorporates contrastive learning to capture the temporal change rules of shade. Then, by conditioning on textual descriptions of known conditions (e.g., time of day, solar angles), our framework provides improved performance in generating shade images. We demonstrate the utility of our approach by using our shade predictions to calculate shade ratios for real-world route planning in Tempe, Arizona. We believe this work will benefit society by providing a reference for urban planning in extreme heat weather and its potential practical applications in the environment.
>
---
#### [replaced 047] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07986v3](http://arxiv.org/pdf/2506.07986v3)**

> **作者:** Zhengyao Lv; Tianlin Pan; Chenyang Si; Zhaoxi Chen; Wangmeng Zuo; Ziwei Liu; Kwan-Yee K. Wong
>
> **备注:** Accepted by ICCV 2025; Project Page: https://vchitect.github.io/TACA/
>
> **摘要:** Multimodal Diffusion Transformers (MM-DiTs) have achieved remarkable progress in text-driven visual generation. However, even state-of-the-art MM-DiT models like FLUX struggle with achieving precise alignment between text prompts and generated content. We identify two key issues in the attention mechanism of MM-DiT, namely 1) the suppression of cross-modal attention due to token imbalance between visual and textual modalities and 2) the lack of timestep-aware attention weighting, which hinder the alignment. To address these issues, we propose \textbf{Temperature-Adjusted Cross-modal Attention (TACA)}, a parameter-efficient method that dynamically rebalances multimodal interactions through temperature scaling and timestep-dependent adjustment. When combined with LoRA fine-tuning, TACA significantly enhances text-image alignment on the T2I-CompBench benchmark with minimal computational overhead. We tested TACA on state-of-the-art models like FLUX and SD3.5, demonstrating its ability to improve image-text alignment in terms of object appearance, attribute binding, and spatial relationships. Our findings highlight the importance of balancing cross-modal attention in improving semantic fidelity in text-to-image diffusion models. Our codes are publicly available at \href{https://github.com/Vchitect/TACA}
>
---
#### [replaced 048] Context Diffusion: In-Context Aware Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.03584v2](http://arxiv.org/pdf/2312.03584v2)**

> **作者:** Ivona Najdenkoska; Animesh Sinha; Abhimanyu Dubey; Dhruv Mahajan; Vignesh Ramanathan; Filip Radenovic
>
> **摘要:** We propose Context Diffusion, a diffusion-based framework that enables image generation models to learn from visual examples presented in context. Recent work tackles such in-context learning for image generation, where a query image is provided alongside context examples and text prompts. However, the quality and context fidelity of the generated images deteriorate when the prompt is not present, demonstrating that these models cannot truly learn from the visual context. To address this, we propose a novel framework that separates the encoding of the visual context and the preservation of the desired image layout. This results in the ability to learn from the visual context and prompts, but also from either of them. Furthermore, we enable our model to handle few-shot settings, to effectively address diverse in-context learning scenarios. Our experiments and human evaluation demonstrate that Context Diffusion excels in both in-domain and out-of-domain tasks, resulting in an overall enhancement in image quality and context fidelity compared to counterpart models.
>
---
#### [replaced 049] ORL-LDM: Offline Reinforcement Learning Guided Latent Diffusion Model Super-Resolution Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.10027v2](http://arxiv.org/pdf/2505.10027v2)**

> **作者:** Shijie Lyu
>
> **备注:** This submission included authors who did not consent to the submission. The paper is being withdrawn until authorship issues are resolved
>
> **摘要:** With the rapid advancement of remote sensing technology, super-resolution image reconstruction is of great research and practical significance. Existing deep learning methods have made progress but still face limitations in handling complex scenes and preserving image details. This paper proposes a reinforcement learning-based latent diffusion model (LDM) fine-tuning method for remote sensing image super-resolution. The method constructs a reinforcement learning environment with states, actions, and rewards, optimizing decision objectives through proximal policy optimization (PPO) during the reverse denoising process of the LDM model. Experiments on the RESISC45 dataset show significant improvements over the baseline model in PSNR, SSIM, and LPIPS, with PSNR increasing by 3-4dB, SSIM improving by 0.08-0.11, and LPIPS reducing by 0.06-0.10, particularly in structured and complex natural scenes. The results demonstrate the method's effectiveness in enhancing super-resolution quality and adaptability across scenes.
>
---
#### [replaced 050] SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition
- **分类: cs.NE; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.15986v2](http://arxiv.org/pdf/2503.15986v2)**

> **作者:** Zeqi Zheng; Yanchen Huang; Yingchao Yu; Zizheng Zhu; Junfeng Tang; Zhaofei Yu; Yaochu Jin
>
> **备注:** Accepted by ICCV 2025. The first two authors contributed equally
>
> **摘要:** Spiking Neural Networks (SNNs) based on Transformers have garnered significant attention due to their superior performance and high energy efficiency. However, the spiking attention modules of most existing Transformer-based SNNs are adapted from those of analog Transformers, failing to fully address the issue of over-allocating attention to irrelevant contexts. To fix this fundamental yet overlooked issue, we propose a Lateral Inhibition-inspired Spiking Transformer (SpiLiFormer). It emulates the brain's lateral inhibition mechanism, guiding the model to enhance attention to relevant tokens while suppressing attention to irrelevant ones. Our model achieves state-of-the-art (SOTA) performance across multiple datasets, including CIFAR-10 (+0.45%), CIFAR-100 (+0.48%), CIFAR10-DVS (+2.70%), N-Caltech101 (+1.94%), and ImageNet-1K (+1.6%). Notably, on the ImageNet-1K dataset, SpiLiFormer (69.9M parameters, 4 time steps, 384 resolution) outperforms E-SpikeFormer (173.0M parameters, 8 time steps, 384 resolution), a SOTA spiking Transformer, by 0.46% using only 39% of the parameters and half the time steps. The code and model checkpoints are publicly available at https://github.com/KirinZheng/SpiLiFormer.
>
---
#### [replaced 051] Fractal Signatures: Securing AI-Generated Pollock-Style Art via Intrinsic Watermarking and Blockchain
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.20519v4](http://arxiv.org/pdf/2410.20519v4)**

> **作者:** Yiquan Wang
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** The digital art market faces unprecedented challenges in authenticity verification and copyright protection. This study introduces an integrated framework to address these issues by combining neural style transfer, fractal analysis, and blockchain technology. We generate abstract artworks inspired by Jackson Pollock, using their inherent mathematical complexity to create robust, imperceptible watermarks. Our method embeds these watermarks, derived from fractal and turbulence features, directly into the artwork's structure. This approach is then secured by linking the watermark to NFT metadata, ensuring immutable proof of ownership. Rigorous testing shows our feature-based watermarking achieves a 76.2% average detection rate against common attacks, significantly outperforming traditional methods (27.8-44.0%). This work offers a practical solution for digital artists and collectors, enhancing security and trust in the digital art ecosystem.
>
---
#### [replaced 052] Cross-domain Multi-step Thinking: Zero-shot Fine-grained Traffic Sign Recognition in the Wild
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2409.01534v2](http://arxiv.org/pdf/2409.01534v2)**

> **作者:** Yaozong Gan; Guang Li; Ren Togo; Keisuke Maeda; Takahiro Ogawa; Miki Haseyama
>
> **备注:** Published by Knowledge-Based Systems
>
> **摘要:** In this study, we propose Cross-domain Multi-step Thinking (CdMT) to improve zero-shot fine-grained traffic sign recognition (TSR) performance in the wild. Zero-shot fine-grained TSR in the wild is challenging due to the cross-domain problem between clean template traffic signs and real-world counterparts, and existing approaches particularly struggle with cross-country TSR scenarios, where traffic signs typically differ between countries. The proposed CdMT framework tackles these challenges by leveraging the multi-step reasoning capabilities of large multimodal models (LMMs). We introduce context, characteristic, and differential descriptions to design multiple thinking processes for LMMs. Context descriptions, which are enhanced by center coordinate prompt optimization, enable the precise localization of target traffic signs in complex road images and filter irrelevant responses via novel prior traffic sign hypotheses. Characteristic descriptions, which are derived from in-context learning with template traffic signs, bridge cross-domain gaps and enhance fine-grained TSR. Differential descriptions refine the multimodal reasoning ability of LMMs by distinguishing subtle differences among similar signs. CdMT is independent of training data and requires only simple and uniform instructions, enabling it to achieve cross-country TSR. We conducted extensive experiments on three benchmark datasets and two real-world datasets from different countries. The proposed CdMT framework achieved superior performance compared with other state-of-the-art methods on all five datasets, with recognition accuracies of 0.93, 0.89, 0.97, 0.89, and 0.85 on the GTSRB, BTSD, TT-100K, Sapporo, and Yokohama datasets, respectively.
>
---
#### [replaced 053] The BabyView dataset: High-resolution egocentric videos of infants' and young children's everyday experiences
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.10447v2](http://arxiv.org/pdf/2406.10447v2)**

> **作者:** Bria Long; Robert Z. Sparks; Violet Xiang; Stefan Stojanov; Zi Yin; Grace E. Keene; Alvin W. M. Tan; Steven Y. Feng; Chengxu Zhuang; Virginia A. Marchman; Daniel L. K. Yamins; Michael C. Frank
>
> **备注:** 9 pages, 3 figures, 4 tables and Appendix. Published in the Proceedings of the 8th Annual Conference on Cognitive Computational Neuroscience
>
> **摘要:** Human children far exceed modern machine learning algorithms in their sample efficiency, achieving high performance in key domains with much less data than current models. This ''data gap'' is a key challenge both for building intelligent artificial systems and for understanding human development. Egocentric video capturing children's experience--their ''training data''--is a key ingredient for comparison of humans and models and for the development of algorithmic innovations to bridge this gap. Yet there are few such datasets available, and extant data are low-resolution, have limited metadata, and importantly, represent only a small set of children's experiences. Here, we provide the first release of a large developmental egocentric video dataset--the BabyView dataset--recorded using a high-resolution camera with a large vertical field-of-view and gyroscope/accelerometer data. This 868 hour dataset includes egocentric videos from children spanning 6 months to 3 years of age in longitudinal, at-home contexts. We provide gold-standard annotations for the evaluation of speech transcription, speaker diarization, and human pose estimation, and evaluate models in each of these domains. We train self-supervised language and vision models and evaluate their transfer to out-of-distribution tasks, including syntactic structure learning, object recognition, depth estimation, and image segmentation. Although performance in each domain scales with dataset size, overall performance is relatively lower than when models are trained on curated datasets, especially in the visual domain. Our dataset stands as an open challenge for robust, human-like AI systems: how can such systems achieve human-levels of success on the same scale and distribution of training data as humans?
>
---
#### [replaced 054] Text2Stereo: Repurposing Stable Diffusion for Stereo Generation with Consistency Rewards
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05367v2](http://arxiv.org/pdf/2506.05367v2)**

> **作者:** Aakash Garg; Libing Zeng; Andrii Tsarov; Nima Khademi Kalantari
>
> **摘要:** In this paper, we propose a novel diffusion-based approach to generate stereo images given a text prompt. Since stereo image datasets with large baselines are scarce, training a diffusion model from scratch is not feasible. Therefore, we propose leveraging the strong priors learned by Stable Diffusion and fine-tuning it on stereo image datasets to adapt it to the task of stereo generation. To improve stereo consistency and text-to-image alignment, we further tune the model using prompt alignment and our proposed stereo consistency reward functions. Comprehensive experiments demonstrate the superiority of our approach in generating high-quality stereo images across diverse scenarios, outperforming existing methods.
>
---
#### [replaced 055] Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.13926v2](http://arxiv.org/pdf/2501.13926v2)**

> **作者:** Ziyu Guo; Renrui Zhang; Chengzhuo Tong; Zhizheng Zhao; Rui Huang; Haoquan Zhang; Manyuan Zhang; Jiaming Liu; Shanghang Zhang; Peng Gao; Hongsheng Li; Pheng-Ann Heng
>
> **备注:** Journal Version. Code and models are released at https://github.com/ZiyuGuo99/Image-Generation-CoT
>
> **摘要:** Chain-of-Thought (CoT) reasoning has been extensively explored in large models to tackle complex understanding tasks. However, it still remains an open question whether such strategies can be applied to verifying and reinforcing image generation scenarios. In this paper, we provide the first comprehensive investigation of the potential of CoT reasoning to enhance autoregressive image generation. We focus on three techniques: scaling test-time computation for verification, aligning model preferences with Direct Preference Optimization (DPO), and integrating these techniques for complementary effects. Our results demonstrate that these approaches can be effectively adapted and combined to significantly improve image generation performance. Furthermore, given the pivotal role of reward models in our findings, we propose the Potential Assessment Reward Model (PARM) and PARM++, specialized for autoregressive image generation. PARM adaptively assesses each generation step through a potential assessment approach, merging the strengths of existing reward models, and PARM++ further introduces a reflection mechanism to self-correct the generated unsatisfactory image, which is the first to incorporate reflection in autoregressive image generation. Using our investigated reasoning strategies, we enhance a baseline model, Show-o, to achieve superior results, with a significant +24% improvement on the GenEval benchmark, surpassing Stable Diffusion 3 by +15%. We hope our study provides unique insights and paves a new path for integrating CoT reasoning with autoregressive image generation. Code and models are released at https://github.com/ZiyuGuo99/Image-Generation-CoT
>
---
#### [replaced 056] OpenVLThinker: Complex Vision-Language Reasoning via Iterative SFT-RL Cycles
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.17352v2](http://arxiv.org/pdf/2503.17352v2)**

> **作者:** Yihe Deng; Hritik Bansal; Fan Yin; Nanyun Peng; Wei Wang; Kai-Wei Chang
>
> **备注:** 23 pages, 11 figures, 8 tables
>
> **摘要:** We introduce OpenVLThinker, one of the first open-source large vision-language models (LVLMs) to exhibit sophisticated chain-of-thought reasoning, achieving notable performance gains on challenging visual reasoning tasks. While text-based reasoning models (e.g., Deepseek R1) show promising results in text-only tasks, distilling their reasoning into LVLMs via supervised fine-tuning (SFT) often results in performance degradation due to imprecise visual grounding. Conversely, purely reinforcement learning (RL)-based methods face a large search space, hindering the emergence of reflective behaviors in smaller models (e.g., 7B LVLMs). Surprisingly, alternating between SFT and RL ultimately results in significant performance improvements after a few iterations. Our analysis reveals that the base model rarely exhibits reasoning behaviors initially, but SFT effectively surfaces these latent actions and narrows the RL search space, accelerating the development of reasoning capabilities. Each subsequent RL stage further refines the model's reasoning skills, producing higher-quality SFT data for continued self-improvement. OpenVLThinker-7B consistently advances performance across six benchmarks demanding mathematical and general reasoning, notably improving MathVista by 3.8%, EMMA by 2.4%, and HallusionBench by 1.6%. Beyond demonstrating the synergy between SFT and RL for complex reasoning tasks, our findings provide early evidence towards achieving R1-style reasoning in multimodal contexts. The code, model and data are held at https://github.com/yihedeng9/OpenVLThinker.
>
---
#### [replaced 057] AuroraLong: Bringing RNNs Back to Efficient Open-Ended Video Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.02591v3](http://arxiv.org/pdf/2507.02591v3)**

> **作者:** Weili Xu; Enxin Song; Wenhao Chai; Xuexiang Wen; Tian Ye; Gaoang Wang
>
> **备注:** ICCV 2025 Camera Ready
>
> **摘要:** The challenge of long video understanding lies in its high computational complexity and prohibitive memory cost, since the memory and computation required by transformer-based LLMs scale quadratically with input sequence length. We propose AuroraLong to address this challenge by replacing the LLM component in MLLMs with a linear RNN language model that handles input sequence of arbitrary length with constant-size hidden states. To further increase throughput and efficiency, we combine visual token merge with linear RNN models by reordering the visual tokens by their sizes in ascending order. Despite having only 2B parameters and being trained exclusively on public data, AuroraLong achieves performance comparable to Transformer-based models of similar size trained on private datasets across multiple video benchmarks. This demonstrates the potential of efficient, linear RNNs to democratize long video understanding by lowering its computational entry barrier. To our best knowledge, we are the first to use a linear RNN based LLM backbone in a LLaVA-like model for open-ended video understanding.
>
---
#### [replaced 058] Advanced U-Net Architectures with CNN Backbones for Automated Lung Cancer Detection and Segmentation in Chest CT Images
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.09898v2](http://arxiv.org/pdf/2507.09898v2)**

> **作者:** Alireza Golkarieh; Kiana Kiashemshaki; Sajjad Rezvani Boroujeni; Nasibeh Asadi Isakan
>
> **备注:** This manuscript has 20 pages and 10 figures. It is submitted to the Journal 'Scientific Reports'
>
> **摘要:** This study investigates the effectiveness of U-Net architectures integrated with various convolutional neural network (CNN) backbones for automated lung cancer detection and segmentation in chest CT images, addressing the critical need for accurate diagnostic tools in clinical settings. A balanced dataset of 832 chest CT images (416 cancerous and 416 non-cancerous) was preprocessed using Contrast Limited Adaptive Histogram Equalization (CLAHE) and resized to 128x128 pixels. U-Net models were developed with three CNN backbones: ResNet50, VGG16, and Xception, to segment lung regions. After segmentation, CNN-based classifiers and hybrid models combining CNN feature extraction with traditional machine learning classifiers (Support Vector Machine, Random Forest, and Gradient Boosting) were evaluated using 5-fold cross-validation. Metrics included accuracy, precision, recall, F1-score, Dice coefficient, and ROC-AUC. U-Net with ResNet50 achieved the best performance for cancerous lungs (Dice: 0.9495, Accuracy: 0.9735), while U-Net with VGG16 performed best for non-cancerous segmentation (Dice: 0.9532, Accuracy: 0.9513). For classification, the CNN model using U-Net with Xception achieved 99.1 percent accuracy, 99.74 percent recall, and 99.42 percent F1-score. The hybrid CNN-SVM-Xception model achieved 96.7 percent accuracy and 97.88 percent F1-score. Compared to prior methods, our framework consistently outperformed existing models. In conclusion, combining U-Net with advanced CNN backbones provides a powerful method for both segmentation and classification of lung cancer in CT scans, supporting early diagnosis and clinical decision-making.
>
---
#### [replaced 059] Feature-Enhanced TResNet for Fine-Grained Food Image Classification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.12828v2](http://arxiv.org/pdf/2507.12828v2)**

> **作者:** Lulu Liu; Zhiyong Xiao
>
> **摘要:** Food is not only essential to human health but also serves as a medium for cultural identity and emotional connection. In the context of precision nutrition, accurately identifying and classifying food images is critical for dietary monitoring, nutrient estimation, and personalized health management. However, fine-grained food classification remains challenging due to the subtle visual differences among similar dishes. To address this, we propose Feature-Enhanced TResNet (FE-TResNet), a novel deep learning model designed to improve the accuracy of food image recognition in fine-grained scenarios. Built on the TResNet architecture, FE-TResNet integrates a Style-based Recalibration Module (StyleRM) and Deep Channel-wise Attention (DCA) to enhance feature extraction and emphasize subtle distinctions between food items. Evaluated on two benchmark Chinese food datasets-ChineseFoodNet and CNFOOD-241-FE-TResNet achieved high classification accuracies of 81.37% and 80.29%, respectively. These results demonstrate its effectiveness and highlight its potential as a key enabler for intelligent dietary assessment and personalized recommendations in precision nutrition systems.
>
---
#### [replaced 060] A Deep Learning Approach for Augmenting Perceptional Understanding of Histopathology Images
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06894v3](http://arxiv.org/pdf/2503.06894v3)**

> **作者:** Xiaoqian Hu
>
> **备注:** Accepted by International Conference on Semantic & Natural Language Processing (SNLP 2025)
>
> **摘要:** In Recent Years, Digital Technologies Have Made Significant Strides In Augmenting-Human-Health, Cognition, And Perception, Particularly Within The Field Of Computational-Pathology. This Paper Presents A Novel Approach To Enhancing The Analysis Of Histopathology Images By Leveraging A Mult-modal-Model That Combines Vision Transformers (Vit) With Gpt-2 For Image Captioning. The Model Is Fine-Tuned On The Specialized Arch-Dataset, Which Includes Dense Image Captions Derived From Clinical And Academic Resources, To Capture The Complexities Of Pathology Images Such As Tissue Morphologies, Staining Variations, And Pathological Conditions. By Generating Accurate, Contextually Captions, The Model Augments The Cognitive Capabilities Of Healthcare Professionals, Enabling More Efficient Disease Classification, Segmentation, And Detection. The Model Enhances The Perception Of Subtle Pathological Features In Images That Might Otherwise Go Unnoticed, Thereby Improving Diagnostic Accuracy. Our Approach Demonstrates The Potential For Digital Technologies To Augment Human Cognitive Abilities In Medical Image Analysis, Providing Steps Toward More Personalized And Accurate Healthcare Outcomes.
>
---
#### [replaced 061] Frequency-Dynamic Attention Modulation for Dense Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.12006v2](http://arxiv.org/pdf/2507.12006v2)**

> **作者:** Linwei Chen; Lin Gu; Ying Fu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Vision Transformers (ViTs) have significantly advanced computer vision, demonstrating strong performance across various tasks. However, the attention mechanism in ViTs makes each layer function as a low-pass filter, and the stacked-layer architecture in existing transformers suffers from frequency vanishing. This leads to the loss of critical details and textures. We propose a novel, circuit-theory-inspired strategy called Frequency-Dynamic Attention Modulation (FDAM), which can be easily plugged into ViTs. FDAM directly modulates the overall frequency response of ViTs and consists of two techniques: Attention Inversion (AttInv) and Frequency Dynamic Scaling (FreqScale). Since circuit theory uses low-pass filters as fundamental elements, we introduce AttInv, a method that generates complementary high-pass filtering by inverting the low-pass filter in the attention matrix, and dynamically combining the two. We further design FreqScale to weight different frequency components for fine-grained adjustments to the target response function. Through feature similarity analysis and effective rank evaluation, we demonstrate that our approach avoids representation collapse, leading to consistent performance improvements across various models, including SegFormer, DeiT, and MaskDINO. These improvements are evident in tasks such as semantic segmentation, object detection, and instance segmentation. Additionally, we apply our method to remote sensing detection, achieving state-of-the-art results in single-scale settings. The code is available at https://github.com/Linwei-Chen/FDAM.
>
---
#### [replaced 062] RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01709v3](http://arxiv.org/pdf/2505.01709v3)**

> **作者:** Kaidong Zhang; Rongtao Xu; Pengzhen Ren; Junfan Lin; Hefeng Wu; Liang Lin; Xiaodan Liang
>
> **备注:** project page: https://abliao.github.io/RoBridge/
>
> **摘要:** Operating robots in open-ended scenarios with diverse tasks is a crucial research and application direction in robotics. While recent progress in natural language processing and large multimodal models has enhanced robots' ability to understand complex instructions, robot manipulation still faces the procedural skill dilemma and the declarative skill dilemma in open environments. Existing methods often compromise cognitive and executive capabilities. To address these challenges, in this paper, we propose RoBridge, a hierarchical intelligent architecture for general robotic manipulation. It consists of a high-level cognitive planner (HCP) based on a large-scale pre-trained vision-language model (VLM), an invariant operable representation (IOR) serving as a symbolic bridge, and a generalist embodied agent (GEA). RoBridge maintains the declarative skill of VLM and unleashes the procedural skill of reinforcement learning, effectively bridging the gap between cognition and execution. RoBridge demonstrates significant performance improvements over existing baselines, achieving a 75% success rate on new tasks and an 83% average success rate in sim-to-real generalization using only five real-world data samples per task. This work represents a significant step towards integrating cognitive reasoning with physical execution in robotic systems, offering a new paradigm for general robotic manipulation.
>
---
#### [replaced 063] ROADWork Dataset: Learning to Recognize, Observe, Analyze and Drive Through Work Zones
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.07661v2](http://arxiv.org/pdf/2406.07661v2)**

> **作者:** Anurag Ghosh; Shen Zheng; Robert Tamburo; Khiem Vuong; Juan Alvarez-Padilla; Hailiang Zhu; Michael Cardei; Nicholas Dunn; Christoph Mertz; Srinivasa G. Narasimhan
>
> **备注:** ICCV 2025 Accepted Paper
>
> **摘要:** Perceiving and autonomously navigating through work zones is a challenging and underexplored problem. Open datasets for this long-tailed scenario are scarce. We propose the ROADWork dataset to learn to recognize, observe, analyze, and drive through work zones. State-of-the-art foundation models fail when applied to work zones. Fine-tuning models on our dataset significantly improves perception and navigation in work zones. With ROADWork dataset, we discover new work zone images with higher precision (+32.5%) at a much higher rate (12.8$\times$) around the world. Open-vocabulary methods fail too, whereas fine-tuned detectors improve performance (+32.2 AP). Vision-Language Models (VLMs) struggle to describe work zones, but fine-tuning substantially improves performance (+36.7 SPICE). Beyond fine-tuning, we show the value of simple techniques. Video label propagation provides additional gains (+2.6 AP) for instance segmentation. While reading work zone signs, composing a detector and text spotter via crop-scaling improves performance +14.2% 1-NED). Composing work zone detections to provide context further reduces hallucinations (+3.9 SPICE) in VLMs. We predict navigational goals and compute drivable paths from work zone videos. Incorporating road work semantics ensures 53.6% goals have angular error (AE) < 0.5 (+9.9 %) and 75.3% pathways have AE < 0.5 (+8.1 %).
>
---
#### [replaced 064] GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.14456v3](http://arxiv.org/pdf/2507.14456v3)**

> **作者:** Chi Wan; Yixin Cui; Jiatong Du; Shuo Yang; Yulong Bai; Yanjun Huang
>
> **摘要:** End-to-end autonomous driving requires adaptive and robust handling of complex and diverse traffic environments. However, prevalent single-mode planning methods attempt to learn an overall policy while struggling to acquire diversified driving skills to handle diverse scenarios. Therefore, this paper proposes GEMINUS, a Mixture-of-Experts end-to-end autonomous driving framework featuring a Global Expert, a Scene-Adaptive Experts Group, and equipped with a Dual-aware Router. Specifically, the Global Expert is trained on the overall dataset, possessing robust performance. The Scene-Adaptive Experts are trained on corresponding scene subsets, achieving adaptive performance. The Dual-aware Router simultaneously considers scenario-level features and routing uncertainty to dynamically activate expert modules. Through the effective coupling of the Global Expert and the Scene-Adaptive Experts Group via the Dual-aware Router, GEMINUS achieves adaptive and robust performance in diverse scenarios. GEMINUS outperforms existing methods in the Bench2Drive closed-loop benchmark and achieves state-of-the-art performance in Driving Score and Success Rate, even with only monocular vision input. Furthermore, ablation studies demonstrate significant improvements over the original single-expert baseline: 7.67% in Driving Score, 22.06% in Success Rate, and 19.41% in MultiAbility-Mean. The code will be available at https://github.com/newbrains1/GEMINUS.
>
---
#### [replaced 065] Emerging Frameworks for Objective Task-based Evaluation of Quantitative Medical Imaging Methods
- **分类: physics.med-ph; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.04591v2](http://arxiv.org/pdf/2507.04591v2)**

> **作者:** Yan Liu; Huitian Xia; Nancy A. Obuchowski; Richard Laforest; Arman Rahmim; Barry A. Siegel; Abhinav K. Jha
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Quantitative imaging (QI) is demonstrating strong promise across multiple clinical applications. For clinical translation of QI methods, objective evaluation on clinically relevant tasks is essential. To address this need, multiple evaluation strategies are being developed. In this paper, based on previous literature, we outline four emerging frameworks to perform evaluation studies of QI methods. We first discuss the use of virtual imaging trials (VITs) to evaluate QI methods. Next, we outline a no-gold-standard evaluation framework to clinically evaluate QI methods without ground truth. Third, a framework to evaluate QI methods for joint detection and quantification tasks is outlined. Finally, we outline a framework to evaluate QI methods that output multi-dimensional parameters, such as radiomic features. We review these frameworks, discussing their utilities and limitations. Further, we examine future research areas in evaluation of QI methods. Given the recent advancements in PET, including long axial field-of-view scanners and the development of artificial-intelligence algorithms, we present these frameworks in the context of PET.
>
---
#### [replaced 066] SurgXBench: Explainable Vision-Language Model Benchmark for Surgery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10764v3](http://arxiv.org/pdf/2505.10764v3)**

> **作者:** Jiajun Cheng; Xianwu Zhao; Sainan Liu; Xiaofan Yu; Ravi Prakash; Patrick J. Codd; Jonathan Elliott Katz; Shan Lin
>
> **摘要:** Innovations in digital intelligence are transforming robotic surgery with more informed decision-making. Real-time awareness of surgical instrument presence and actions (e.g., cutting tissue) is essential for such systems. Yet, despite decades of research, most machine learning models for this task are trained on small datasets and still struggle to generalize. Recently, vision-Language Models (VLMs) have brought transformative advances in reasoning across visual and textual modalities. Their unprecedented generalization capabilities suggest great potential for advancing intelligent robotic surgery. However, surgical VLMs remain under-explored, and existing models show limited performance, highlighting the need for benchmark studies to assess their capabilities and limitations and to inform future development. To this end, we benchmark the zero-shot performance of several advanced VLMs on two public robotic-assisted laparoscopic datasets for instrument and action classification. Beyond standard evaluation, we integrate explainable AI to visualize VLM attention and uncover causal explanations behind their predictions. This provides a previously underexplored perspective in this field for evaluating the reliability of model predictions. We also propose several explainability analysis-based metrics to complement standard evaluations. Our analysis reveals that surgical VLMs, despite domain-specific training, often rely on weak contextual cues rather than clinically relevant visual evidence, highlighting the need for stronger visual and reasoning supervision in surgical applications.
>
---
#### [replaced 067] Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2502.20650v4](http://arxiv.org/pdf/2502.20650v4)**

> **作者:** Yu Pan; Jiahao Chen; Bingrong Dai; Lin Wang; Yi Du; Jiao Liu
>
> **摘要:** In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.
>
---
#### [replaced 068] Monitoring digestate application on agricultural crops using Sentinel-2 Satellite imagery
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.19996v2](http://arxiv.org/pdf/2504.19996v2)**

> **作者:** Andreas Kalogeras; Dimitrios Bormpoudakis; Iason Tsardanidis; Dimitra A. Loka; Charalampos Kontoes
>
> **备注:** Accepted for 2025 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025)
>
> **摘要:** The widespread use of Exogenous Organic Matter in agriculture necessitates monitoring to assess its effects on soil and crop health. This study evaluates optical Sentinel-2 satellite imagery for detecting digestate application, a practice that enhances soil fertility but poses environmental risks like microplastic contamination and nitrogen losses. In the first instance, Sentinel-2 satellite image time series (SITS) analysis of specific indices (EOMI, NDVI, EVI) was used to characterize EOM's spectral behavior after application on the soils of four different crop types in Thessaly, Greece. Furthermore, Machine Learning (ML) models (namely Random Forest, k-NN, Gradient Boosting and a Feed-Forward Neural Network), were used to investigate digestate presence detection, achieving F1-scores up to 0.85. The findings highlight the potential of combining remote sensing and ML for scalable and cost-effective monitoring of EOM applications, supporting precision agriculture and sustainability.
>
---
#### [replaced 069] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.14811v2](http://arxiv.org/pdf/2507.14811v2)**

> **作者:** Jiaji Zhang; Ruichao Sun; Hailiang Zhao; Jiaju Wu; Peng Chen; Hao Li; Yuying Liu; Xinkui Zhao; Kingsum Chow; Gang Xiong; Shuiguang Deng
>
> **摘要:** Diffusion models have demonstrated exceptional generative capabilities but are computationally intensive, posing significant challenges for deployment in resource-constrained or latency-sensitive environments. Quantization offers an effective means to reduce model size and computational cost, with post-training quantization (PTQ) being particularly appealing due to its compatibility with pre-trained models without requiring retraining or training data. However, existing PTQ methods for diffusion models often rely on architecture-specific heuristics that limit their generalizability and hinder integration with industrial deployment pipelines. To address these limitations, we propose SegQuant, a unified quantization framework that adaptively combines complementary techniques to enhance cross-model versatility. SegQuant consists of a segment-aware, graph-based quantization strategy (SegLinear) that captures structural semantics and spatial heterogeneity, along with a dual-scale quantization scheme (DualScale) that preserves polarity-asymmetric activations, which is crucial for maintaining visual fidelity in generated outputs. SegQuant is broadly applicable beyond Transformer-based diffusion models, achieving strong performance while ensuring seamless compatibility with mainstream deployment tools.
>
---
#### [replaced 070] EarthCrafter: Scalable 3D Earth Generation via Dual-Sparse Latent Diffusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.16535v2](http://arxiv.org/pdf/2507.16535v2)**

> **作者:** Shang Liu; Chenjie Cao; Chaohui Yu; Wen Qian; Jing Wang; Fan Wang
>
> **备注:** Models and codes will be released at this https URL: https://github.com/whiteinblue/EarthCrafter
>
> **摘要:** Despite the remarkable developments achieved by recent 3D generation works, scaling these methods to geographic extents, such as modeling thousands of square kilometers of Earth's surface, remains an open challenge. We address this through a dual innovation in data infrastructure and model architecture. First, we introduce Aerial-Earth3D, the largest 3D aerial dataset to date, consisting of 50k curated scenes (each measuring 600m x 600m) captured across the U.S. mainland, comprising 45M multi-view Google Earth frames. Each scene provides pose-annotated multi-view images, depth maps, normals, semantic segmentation, and camera poses, with explicit quality control to ensure terrain diversity. Building on this foundation, we propose EarthCrafter, a tailored framework for large-scale 3D Earth generation via sparse-decoupled latent diffusion. Our architecture separates structural and textural generation: 1) Dual sparse 3D-VAEs compress high-resolution geometric voxels and textural 2D Gaussian Splats (2DGS) into compact latent spaces, largely alleviating the costly computation suffering from vast geographic scales while preserving critical information. 2) We propose condition-aware flow matching models trained on mixed inputs (semantics, images, or neither) to flexibly model latent geometry and texture features independently. Extensive experiments demonstrate that EarthCrafter performs substantially better in extremely large-scale generation. The framework further supports versatile applications, from semantic-guided urban layout generation to unconditional terrain synthesis, while maintaining geographic plausibility through our rich data priors from Aerial-Earth3D. Our project page is available at https://whiteinblue.github.io/earthcrafter/
>
---
#### [replaced 071] RGBX-DiffusionDet: A Framework for Multi-Modal RGB-X Object Detection Using DiffusionDet
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02586v3](http://arxiv.org/pdf/2505.02586v3)**

> **作者:** Eliraz Orfaig; Inna Stainvas; Igal Bilik
>
> **摘要:** This work introduces RGBX-DiffusionDet, an object detection framework extending the DiffusionDet model to fuse the heterogeneous 2D data (X) with RGB imagery via an adaptive multimodal encoder. To enable cross-modal interaction, we design the dynamic channel reduction within a convolutional block attention module (DCR-CBAM), which facilitates cross-talk between subnetworks by dynamically highlighting salient channel features. Furthermore, the dynamic multi-level aggregation block (DMLAB) is proposed to refine spatial feature representations through adaptive multiscale fusion. Finally, novel regularization losses that enforce channel saliency and spatial selectivity are introduced, leading to compact and discriminative feature embeddings. Extensive experiments using RGB-Depth (KITTI), a novel annotated RGB-Polarimetric dataset, and RGB-Infrared (M$^3$FD) benchmark dataset were conducted. We demonstrate consistent superiority of the proposed approach over the baseline RGB-only DiffusionDet. The modular architecture maintains the original decoding complexity, ensuring efficiency. These results establish the proposed RGBX-DiffusionDet as a flexible multimodal object detection approach, providing new insights into integrating diverse 2D sensing modalities into diffusion-based detection pipelines.
>
---
