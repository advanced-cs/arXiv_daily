# 计算机视觉 cs.CV

- **最新发布 125 篇**

- **更新 86 篇**

## 最新发布

#### [new 001] Towards Affordable Tumor Segmentation and Visualization for 3D Breast MRI Using SAM2
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决3D乳腺MRI中肿瘤分割与可视化的高成本与低效率问题。研究使用SAM2模型，在仅需单一切片标注的情况下，通过三种切片跟踪策略进行3D分割，发现中心向外传播效果最佳，展示了该模型在资源有限环境中的潜力。**

- **链接: [http://arxiv.org/pdf/2507.23272v1](http://arxiv.org/pdf/2507.23272v1)**

> **作者:** Solha Kang; Eugene Kim; Joris Vankerschaver; Utku Ozbulak
>
> **备注:** Accepted for publication in the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2nd Deep Breast Workshop on AI and Imaging for Diagnostic and Treatment Challenges in Breast Care (DeepBreath), 2025
>
> **摘要:** Breast MRI provides high-resolution volumetric imaging critical for tumor assessment and treatment planning, yet manual interpretation of 3D scans remains labor-intensive and subjective. While AI-powered tools hold promise for accelerating medical image analysis, adoption of commercial medical AI products remains limited in low- and middle-income countries due to high license costs, proprietary software, and infrastructure demands. In this work, we investigate whether the Segment Anything Model 2 (SAM2) can be adapted for low-cost, minimal-input 3D tumor segmentation in breast MRI. Using a single bounding box annotation on one slice, we propagate segmentation predictions across the 3D volume using three different slice-wise tracking strategies: top-to-bottom, bottom-to-top, and center-outward. We evaluate these strategies across a large cohort of patients and find that center-outward propagation yields the most consistent and accurate segmentations. Despite being a zero-shot model not trained for volumetric medical data, SAM2 achieves strong segmentation performance under minimal supervision. We further analyze how segmentation performance relates to tumor size, location, and shape, identifying key failure modes. Our results suggest that general-purpose foundation models such as SAM2 can support 3D medical image analysis with minimal supervision, offering an accessible and affordable alternative for resource-constrained settings.
>
---
#### [new 002] A Novel Dataset for Flood Detection Robust to Seasonal Changes in Satellite Imagery
- **分类: cs.CV; I.4.6; I.2.10; I.5.4**

- **简介: 该论文属于遥感图像语义分割任务，旨在解决现有数据集在洪水检测中季节变化适应性不足的问题。作者构建了一个包含多地、多时相的卫星图像数据集，并评估了现有模型性能，指出需发展多模态和时序学习方法。**

- **链接: [http://arxiv.org/pdf/2507.23193v1](http://arxiv.org/pdf/2507.23193v1)**

> **作者:** Youngsun Jang; Dongyoun Kim; Chulwoo Pack; Kwanghee Won
>
> **备注:** 8 pages, 2 figures. Presented at ACM RACS 2024 (Pompei, Italy, Nov 5-8, 2024)
>
> **摘要:** This study introduces a novel dataset for segmenting flooded areas in satellite images. After reviewing 77 existing benchmarks utilizing satellite imagery, we identified a shortage of suitable datasets for this specific task. To fill this gap, we collected satellite imagery of the 2019 Midwestern USA floods from Planet Explorer by Planet Labs (Image \c{opyright} 2024 Planet Labs PBC). The dataset consists of 10 satellite images per location, each containing both flooded and non-flooded areas. We selected ten locations from each of the five states: Iowa, Kansas, Montana, Nebraska, and South Dakota. The dataset ensures uniform resolution and resizing during data processing. For evaluating semantic segmentation performance, we tested state-of-the-art models in computer vision and remote sensing on our dataset. Additionally, we conducted an ablation study varying window sizes to capture temporal characteristics. Overall, the models demonstrated modest results, suggesting a requirement for future multimodal and temporal learning strategies. The dataset will be publicly available on <https://github.com/youngsunjang/SDSU_MidWest_Flood_2019>.
>
---
#### [new 003] IN45023 Neural Network Design Patterns in Computer Vision Seminar Report, Summer 2025
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在分析关键设计模式的演变。论文通过研究六篇重要论文，探讨了图像识别基础架构（如ResNet、ViT）、生成模型（如GANs、LDMs）及自监督学习方法（如DINO、MAE），总结了其贡献与影响。**

- **链接: [http://arxiv.org/pdf/2507.23357v1](http://arxiv.org/pdf/2507.23357v1)**

> **作者:** Radu-Andrei Bourceanu; Neil De La Fuente; Jan Grimm; Andrei Jardan; Andriy Manucharyan; Cornelius Weiss; Roman Pflugfelder
>
> **摘要:** This report analyzes the evolution of key design patterns in computer vision by examining six influential papers. The analy- sis begins with foundational architectures for image recognition. We review ResNet, which introduced residual connections to overcome the vanishing gradient problem and enable effective training of significantly deeper convolutional networks. Subsequently, we examine the Vision Transformer (ViT), which established a new paradigm by applying the Transformer ar- chitecture to sequences of image patches, demonstrating the efficacy of attention-based models for large-scale image recogni- tion. Building on these visual representation backbones, we investigate generative models. Generative Adversarial Networks (GANs) are analyzed for their novel adversarial training process, which challenges a generator against a discriminator to learn complex data distributions. Then, Latent Diffusion Models (LDMs) are covered, which improve upon prior generative methods by performing a sequential denoising process in a perceptually compressed latent space. LDMs achieve high-fidelity synthesis with greater computational efficiency, representing the current state-of-the-art for image generation. Finally, we explore self-supervised learning techniques that reduce dependency on labeled data. DINO is a self-distillation framework in which a student network learns to match the output of a momentum-updated teacher, yielding features with strong k-NN classification performance. We conclude with Masked Autoencoders (MAE), which utilize an asymmetric encoder-decoder design to reconstruct heavily masked inputs, providing a highly scalable and effective method for pre-training large-scale vision models.
>
---
#### [new 004] NeRF Is a Valuable Assistant for 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于三维场景表示任务，旨在解决3D高斯泼溅（3DGS）在初始化敏感、空间感知弱等问题。论文提出NeRF-GS框架，联合优化NeRF和3DGS，通过共享空间信息提升3DGS性能，实现更优三维场景重建。**

- **链接: [http://arxiv.org/pdf/2507.23374v1](http://arxiv.org/pdf/2507.23374v1)**

> **作者:** Shuangkang Fang; I-Chao Shen; Takeo Igarashi; Yufeng Wang; ZeSheng Wang; Yi Yang; Wenrui Ding; Shuchang Zhou
>
> **备注:** Accepted by ICCV
>
> **摘要:** We introduce NeRF-GS, a novel framework that jointly optimizes Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). This framework leverages the inherent continuous spatial representation of NeRF to mitigate several limitations of 3DGS, including sensitivity to Gaussian initialization, limited spatial awareness, and weak inter-Gaussian correlations, thereby enhancing its performance. In NeRF-GS, we revisit the design of 3DGS and progressively align its spatial features with NeRF, enabling both representations to be optimized within the same scene through shared 3D spatial information. We further address the formal distinctions between the two approaches by optimizing residual vectors for both implicit features and Gaussian positions to enhance the personalized capabilities of 3DGS. Experimental results on benchmark datasets show that NeRF-GS surpasses existing methods and achieves state-of-the-art performance. This outcome confirms that NeRF and 3DGS are complementary rather than competing, offering new insights into hybrid approaches that combine 3DGS and NeRF for efficient 3D scene representation.
>
---
#### [new 005] SUB: Benchmarking CBM Generalization via Synthetic Attribute Substitutions
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像生成与可解释AI任务，旨在解决概念瓶颈模型（CBMs）在分布偏移下难以准确识别概念的问题。作者构建了SUB基准数据集，包含38,400张合成图像，通过新提出的Tied Diffusion Guidance方法控制图像属性，用于评估CBMs的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.23784v1](http://arxiv.org/pdf/2507.23784v1)**

> **作者:** Jessica Bader; Leander Girrbach; Stephan Alaniz; Zeynep Akata
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Concept Bottleneck Models (CBMs) and other concept-based interpretable models show great promise for making AI applications more transparent, which is essential in fields like medicine. Despite their success, we demonstrate that CBMs struggle to reliably identify the correct concepts under distribution shifts. To assess the robustness of CBMs to concept variations, we introduce SUB: a fine-grained image and concept benchmark containing 38,400 synthetic images based on the CUB dataset. To create SUB, we select a CUB subset of 33 bird classes and 45 concepts to generate images which substitute a specific concept, such as wing color or belly pattern. We introduce a novel Tied Diffusion Guidance (TDG) method to precisely control generated images, where noise sharing for two parallel denoising processes ensures that both the correct bird class and the correct attribute are generated. This novel benchmark enables rigorous evaluation of CBMs and similar interpretable models, contributing to the development of more robust methods. Our code is available at https://github.com/ExplainableML/sub and the dataset at http://huggingface.co/datasets/Jessica-bader/SUB.
>
---
#### [new 006] Forgetting of task-specific knowledge in model merging-based continual learning
- **分类: cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决模型合并中任务特定知识遗忘的问题。通过视觉实验，论文发现线性模型合并能保留或增强共享知识，但任务特定知识会快速退化，并指出增量训练模型合并优于并行训练。**

- **链接: [http://arxiv.org/pdf/2507.23311v1](http://arxiv.org/pdf/2507.23311v1)**

> **作者:** Timm Hess; Gido M van de Ven; Tinne Tuytelaars
>
> **摘要:** This paper investigates the linear merging of models in the context of continual learning (CL). Using controlled visual cues in computer vision experiments, we demonstrate that merging largely preserves or enhances shared knowledge, while unshared task-specific knowledge rapidly degrades. We further find that merging models from an incremental training process consistently outperforms merging models trained in parallel.
>
---
#### [new 007] Machine learning and machine learned prediction in chest X-ray images
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在通过机器学习预测胸部X光图像中的疾病。研究使用了5824张X光图像，比较了CNN和DenseNet-121两种算法的性能。结果表明两者均表现良好，但DenseNet-121在关注图像关键区域方面更优。**

- **链接: [http://arxiv.org/pdf/2507.23455v1](http://arxiv.org/pdf/2507.23455v1)**

> **作者:** Shereiff Garrett; Abhinav Adhikari; Sarina Gautam; DaShawn Marquis Morris; Chandra Mani Adhikari
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Machine learning and artificial intelligence are fast-growing fields of research in which data is used to train algorithms, learn patterns, and make predictions. This approach helps to solve seemingly intricate problems with significant accuracy without explicit programming by recognizing complex relationships in data. Taking an example of 5824 chest X-ray images, we implement two machine learning algorithms, namely, a baseline convolutional neural network (CNN) and a DenseNet-121, and present our analysis in making machine-learned predictions in predicting patients with ailments. Both baseline CNN and DenseNet-121 perform very well in the binary classification problem presented in this work. Gradient-weighted class activation mapping shows that DenseNet-121 correctly focuses on essential parts of the input chest X-ray images in its decision-making more than the baseline CNN.
>
---
#### [new 008] PriorFusion: Unified Integration of Priors for Robust Road Perception in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的道路感知任务，旨在解决复杂环境中道路元素识别不准确的问题。论文提出了PriorFusion框架，统一融合语义、几何和生成先验，通过实例感知注意力机制和扩散模型生成准确的道路元素预测。**

- **链接: [http://arxiv.org/pdf/2507.23309v1](http://arxiv.org/pdf/2507.23309v1)**

> **作者:** Xuewei Tang; Mengmeng Yang; Tuopu Wen; Peijin Jia; Le Cui; Mingshang Luo; Kehua Sheng; Bo Zhang; Diange Yang; Kun Jiang
>
> **摘要:** With the growing interest in autonomous driving, there is an increasing demand for accurate and reliable road perception technologies. In complex environments without high-definition map support, autonomous vehicles must independently interpret their surroundings to ensure safe and robust decision-making. However, these scenarios pose significant challenges due to the large number, complex geometries, and frequent occlusions of road elements. A key limitation of existing approaches lies in their insufficient exploitation of the structured priors inherently present in road elements, resulting in irregular, inaccurate predictions. To address this, we propose PriorFusion, a unified framework that effectively integrates semantic, geometric, and generative priors to enhance road element perception. We introduce an instance-aware attention mechanism guided by shape-prior features, then construct a data-driven shape template space that encodes low-dimensional representations of road elements, enabling clustering to generate anchor points as reference priors. We design a diffusion-based framework that leverages these prior anchors to generate accurate and complete predictions. Experiments on large-scale autonomous driving datasets demonstrate that our method significantly improves perception accuracy, particularly under challenging conditions. Visualization results further confirm that our approach produces more accurate, regular, and coherent predictions of road elements.
>
---
#### [new 009] Toward Safe, Trustworthy and Realistic Augmented Reality User Experience
- **分类: cs.CV**

- **简介: 论文致力于提升增强现实（AR）用户体验的安全性与可信度，解决虚拟内容可能遮挡关键信息或操纵用户感知的问题。研究提出了ViDDAR和VIM-Sense系统，利用视觉-语言模型和多模态推理检测恶意内容，并提出未来三个方向：内容质量评估、多模态攻击检测和轻量化模型适配。**

- **链接: [http://arxiv.org/pdf/2507.23226v1](http://arxiv.org/pdf/2507.23226v1)**

> **作者:** Yanming Xiu
>
> **备注:** 2 pages, 4 figures
>
> **摘要:** As augmented reality (AR) becomes increasingly integrated into everyday life, ensuring the safety and trustworthiness of its virtual content is critical. Our research addresses the risks of task-detrimental AR content, particularly that which obstructs critical information or subtly manipulates user perception. We developed two systems, ViDDAR and VIM-Sense, to detect such attacks using vision-language models (VLMs) and multimodal reasoning modules. Building on this foundation, we propose three future directions: automated, perceptually aligned quality assessment of virtual content; detection of multimodal attacks; and adaptation of VLMs for efficient and user-centered deployment on AR devices. Overall, our work aims to establish a scalable, human-aligned framework for safeguarding AR experiences and seeks feedback on perceptual modeling, multimodal AR content implementation, and lightweight model adaptation.
>
---
#### [new 010] Stable-Sim2Real: Exploring Simulation of Real-Captured 3D Data with Two-Stage Depth Diffusion
- **分类: cs.CV**

- **简介: 该论文属于3D数据仿真任务，旨在解决仿真与真实3D数据之间的差异问题。作者提出了Stable-Sim2Real方法，利用两阶段深度扩散模型，先生成粗略深度，再通过第二阶段优化局部细节，提升仿真数据的真实感。实验表明该方法在真实3D视觉任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.23483v1](http://arxiv.org/pdf/2507.23483v1)**

> **作者:** Mutian Xu; Chongjie Ye; Haolin Liu; Yushuang Wu; Jiahao Chang; Xiaoguang Han
>
> **备注:** ICCV 2025 (Highlight). Project page: https://mutianxu.github.io/stable-sim2real/
>
> **摘要:** 3D data simulation aims to bridge the gap between simulated and real-captured 3D data, which is a fundamental problem for real-world 3D visual tasks. Most 3D data simulation methods inject predefined physical priors but struggle to capture the full complexity of real data. An optimal approach involves learning an implicit mapping from synthetic to realistic data in a data-driven manner, but progress in this solution has met stagnation in recent studies. This work explores a new solution path of data-driven 3D simulation, called Stable-Sim2Real, based on a novel two-stage depth diffusion model. The initial stage finetunes Stable-Diffusion to generate the residual between the real and synthetic paired depth, producing a stable but coarse depth, where some local regions may deviate from realistic patterns. To enhance this, both the synthetic and initial output depth are fed into a second-stage diffusion, where diffusion loss is adjusted to prioritize these distinct areas identified by a 3D discriminator. We provide a new benchmark scheme to evaluate 3D data simulation methods. Extensive experiments show that training the network with the 3D simulated data derived from our method significantly enhances performance in real-world 3D visual tasks. Moreover, the evaluation demonstrates the high similarity between our 3D simulated data and real-captured patterns. Project page: https://mutianxu.github.io/stable-sim2real/.
>
---
#### [new 011] Mamba-based Efficient Spatio-Frequency Motion Perception for Video Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 该论文属于视频伪装物体检测任务，旨在解决因前景与背景高度相似导致的空间外观特征难以有效区分的问题。论文提出了一种基于Mamba的双域运动感知模型（Vcamba），融合空间与频率特征，提升检测准确性与完整性。**

- **链接: [http://arxiv.org/pdf/2507.23601v1](http://arxiv.org/pdf/2507.23601v1)**

> **作者:** Xin Li; Keren Fu; Qijun Zhao
>
> **备注:** 11 pages, 11 figures
>
> **摘要:** Existing video camouflaged object detection (VCOD) methods primarily rely on spatial appearance features to perceive motion cues for breaking camouflage. However, the high similarity between foreground and background in VCOD results in limited discriminability of spatial appearance features (e.g., color and texture), restricting detection accuracy and completeness. Recent studies demonstrate that frequency features can not only enhance feature representation to compensate for appearance limitations but also perceive motion through dynamic variations in frequency energy. Furthermore, the emerging state space model called Mamba, enables efficient perception of motion cues in frame sequences due to its linear-time long-sequence modeling capability. Motivated by this, we propose a novel visual camouflage Mamba (Vcamba) based on spatio-frequency motion perception that integrates frequency and spatial features for efficient and accurate VCOD. Specifically, we propose a receptive field visual state space (RFVSS) module to extract multi-scale spatial features after sequence modeling. For frequency learning, we introduce an adaptive frequency component enhancement (AFE) module with a novel frequency-domain sequential scanning strategy to maintain semantic consistency. Then we propose a space-based long-range motion perception (SLMP) module and a frequency-based long-range motion perception (FLMP) module to model spatio-temporal and frequency-temporal sequences in spatial and frequency phase domains. Finally, the space and frequency motion fusion module (SFMF) integrates dual-domain features for unified motion representation. Experimental results show that our Vcamba outperforms state-of-the-art methods across 6 evaluation metrics on 2 datasets with lower computation cost, confirming the superiority of Vcamba. Our code is available at: https://github.com/BoydeLi/Vcamba.
>
---
#### [new 012] Adjustable Spatio-Spectral Hyperspectral Image Compression Network
- **分类: cs.CV**

- **简介: 该论文属于遥感图像压缩任务，旨在解决高效存储超光谱图像（HSI）的问题。现有方法未充分研究空间与光谱冗余的联合压缩效果。为此，作者提出了可调空间-光谱压缩网络HyCASS，包含六个模块，支持在不同压缩比下灵活压缩空间和光谱信息。**

- **链接: [http://arxiv.org/pdf/2507.23447v1](http://arxiv.org/pdf/2507.23447v1)**

> **作者:** Martin Hermann Paul Fuchs; Behnood Rasti; Begüm Demir
>
> **摘要:** With the rapid growth of hyperspectral data archives in remote sensing (RS), the need for efficient storage has become essential, driving significant attention toward learning-based hyperspectral image (HSI) compression. However, a comprehensive investigation of the individual and joint effects of spectral and spatial compression on learning-based HSI compression has not been thoroughly examined yet. Conducting such an analysis is crucial for understanding how the exploitation of spectral, spatial, and joint spatio-spectral redundancies affects HSI compression. To address this issue, we propose Adjustable Spatio-Spectral Hyperspectral Image Compression Network (HyCASS), a learning-based model designed for adjustable HSI compression in both spectral and spatial dimensions. HyCASS consists of six main modules: 1) spectral encoder; 2) spatial encoder; 3) compression ratio (CR) adapter encoder; 4) CR adapter decoder; 5) spatial decoder; and 6) spectral decoder module. The modules employ convolutional layers and transformer blocks to capture both short-range and long-range redundancies. Experimental results on two HSI benchmark datasets demonstrate the effectiveness of our proposed adjustable model compared to existing learning-based compression models. Based on our results, we establish a guideline for effectively balancing spectral and spatial compression across different CRs, taking into account the spatial resolution of the HSIs. Our code and pre-trained model weights are publicly available at https://git.tu-berlin.de/rsim/hycass .
>
---
#### [new 013] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection
- **分类: cs.CV**

- **简介: 该论文属于单目三维目标检测任务，旨在解决开放场景下新环境和新类别物体检测的挑战。作者提出3D-MOOD框架，通过设计3D边界框头和规范化图像空间，实现2D到3D的提升和跨数据集训练，取得最优性能。**

- **链接: [http://arxiv.org/pdf/2507.23567v1](http://arxiv.org/pdf/2507.23567v1)**

> **作者:** Yung-Hsu Yang; Luigi Piccinelli; Mattia Segu; Siyuan Li; Rui Huang; Yuqian Fu; Marc Pollefeys; Hermann Blum; Zuria Bauer
>
> **备注:** ICCV 2025
>
> **摘要:** Monocular 3D object detection is valuable for various applications such as robotics and AR/VR. Existing methods are confined to closed-set settings, where the training and testing sets consist of the same scenes and/or object categories. However, real-world applications often introduce new environments and novel object categories, posing a challenge to these methods. In this paper, we address monocular 3D object detection in an open-set setting and introduce the first end-to-end 3D Monocular Open-set Object Detector (3D-MOOD). We propose to lift the open-set 2D detection into 3D space through our designed 3D bounding box head, enabling end-to-end joint training for both 2D and 3D tasks to yield better overall performance. We condition the object queries with geometry prior and overcome the generalization for 3D estimation across diverse scenes. To further improve performance, we design the canonical image space for more efficient cross-dataset training. We evaluate 3D-MOOD on both closed-set settings (Omni3D) and open-set settings (Omni3D to Argoverse 2, ScanNet), and achieve new state-of-the-art results. Code and models are available at royyang0714.github.io/3D-MOOD.
>
---
#### [new 014] Adversarial-Guided Diffusion for Multimodal LLM Attacks
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型（MLLM）攻击任务，旨在生成对抗图像以欺骗模型生成目标回复，同时保持图像干净。论文提出“对抗引导扩散”（AGD）方法，在扩散模型的噪声中注入目标语义，增强攻击效果并提升防御鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.23202v1](http://arxiv.org/pdf/2507.23202v1)**

> **作者:** Chengwei Xia; Fan Ma; Ruijie Quan; Kun Zhan; Yi Yang
>
> **摘要:** This paper addresses the challenge of generating adversarial image using a diffusion model to deceive multimodal large language models (MLLMs) into generating the targeted responses, while avoiding significant distortion of the clean image. To address the above challenges, we propose an adversarial-guided diffusion (AGD) approach for adversarial attack MLLMs. We introduce adversarial-guided noise to ensure attack efficacy. A key observation in our design is that, unlike most traditional adversarial attacks which embed high-frequency perturbations directly into the clean image, AGD injects target semantics into the noise component of the reverse diffusion. Since the added noise in a diffusion model spans the entire frequency spectrum, the adversarial signal embedded within it also inherits this full-spectrum property. Importantly, during reverse diffusion, the adversarial image is formed as a linear combination of the clean image and the noise. Thus, when applying defenses such as a simple low-pass filtering, which act independently on each component, the adversarial image within the noise component is less likely to be suppressed, as it is not confined to the high-frequency band. This makes AGD inherently robust to variety defenses. Extensive experiments demonstrate that our AGD outperforms state-of-the-art methods in attack performance as well as in model robustness to some defenses.
>
---
#### [new 015] The Cow of Rembrandt - Analyzing Artistic Prompt Interpretation in Text-to-Image Models
- **分类: cs.CV**

- **简介: 该论文研究文本到图像扩散模型如何内部表示艺术创作中的内容与风格概念。通过交叉注意力热图，分析不同提示词对图像区域的影响，探索模型在无监督情况下是否具备内容与风格分离的能力。属于计算机视觉与生成模型领域的艺术生成机制分析任务。**

- **链接: [http://arxiv.org/pdf/2507.23313v1](http://arxiv.org/pdf/2507.23313v1)**

> **作者:** Alfio Ferrara; Sergio Picascia; Elisabetta Rocchetti
>
> **备注:** to be published in: Applications of AI in the Analysis of Cultural and Artistic Heritage, organized within the 35th IEEE International Workshop on Machine Learning for Signal Processing (MLSP) 2025
>
> **摘要:** Text-to-image diffusion models have demonstrated remarkable capabilities in generating artistic content by learning from billions of images, including popular artworks. However, the fundamental question of how these models internally represent concepts, such as content and style in paintings, remains unexplored. Traditional computer vision assumes content and style are orthogonal, but diffusion models receive no explicit guidance about this distinction during training. In this work, we investigate how transformer-based text-to-image diffusion models encode content and style concepts when generating artworks. We leverage cross-attention heatmaps to attribute pixels in generated images to specific prompt tokens, enabling us to isolate image regions influenced by content-describing versus style-describing tokens. Our findings reveal that diffusion models demonstrate varying degrees of content-style separation depending on the specific artistic prompt and style requested. In many cases, content tokens primarily influence object-related regions while style tokens affect background and texture areas, suggesting an emergent understanding of the content-style distinction. These insights contribute to our understanding of how large-scale generative models internally represent complex artistic concepts without explicit supervision. We share the code and dataset, together with an exploratory tool for visualizing attention maps at https://github.com/umilISLab/artistic-prompt-interpretation.
>
---
#### [new 016] RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决开放世界场景中缺乏大规模推理型可抓取区域预测数据的问题。作者构建了包含273k图像和26k推理指令的大规模基准数据集RAGNet，并提出基于可抓取区域感知的框架AffordanceNet，提升机器人根据语言指令在复杂环境中抓取物体的能力。**

- **链接: [http://arxiv.org/pdf/2507.23734v1](http://arxiv.org/pdf/2507.23734v1)**

> **作者:** Dongming Wu; Yanping Fu; Saike Huang; Yingfei Liu; Fan Jia; Nian Liu; Feng Dai; Tiancai Wang; Rao Muhammad Anwer; Fahad Shahbaz Khan; Jianbing Shen
>
> **备注:** Accepted by ICCV 2025. The code is at https://github.com/wudongming97/AffordanceNet
>
> **摘要:** General robotic grasping systems require accurate object affordance perception in diverse open-world scenarios following human instructions. However, current studies suffer from the problem of lacking reasoning-based large-scale affordance prediction data, leading to considerable concern about open-world effectiveness. To address this limitation, we build a large-scale grasping-oriented affordance segmentation benchmark with human-like instructions, named RAGNet. It contains 273k images, 180 categories, and 26k reasoning instructions. The images cover diverse embodied data domains, such as wild, robot, ego-centric, and even simulation data. They are carefully annotated with an affordance map, while the difficulty of language instructions is largely increased by removing their category name and only providing functional descriptions. Furthermore, we propose a comprehensive affordance-based grasping framework, named AffordanceNet, which consists of a VLM pre-trained on our massive affordance data and a grasping network that conditions an affordance map to grasp the target. Extensive experiments on affordance segmentation benchmarks and real-robot manipulation tasks show that our model has a powerful open-world generalization ability. Our data and code is available at https://github.com/wudongming97/AffordanceNet.
>
---
#### [new 017] ST-SAM: SAM-Driven Self-Training Framework for Semi-Supervised Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 论文提出ST-SAM，用于半监督伪装目标检测（SSCOD），旨在减少对像素级标注的依赖。现有方法存在预测偏差和计算开销大问题。ST-SAM采用自训练策略，结合Segment Anything Model，动态生成高置信度伪标签，并转化为包含领域知识的混合提示，提升检测效果。实验表明其性能优越，仅需1%标注数据。**

- **链接: [http://arxiv.org/pdf/2507.23307v1](http://arxiv.org/pdf/2507.23307v1)**

> **作者:** Xihang Hu; Fuming Sun; Jiazhe Liu; Feilong Xu; Xiaoli Zhang
>
> **备注:** 10 pages, 6 figures, ACM MM 2025
>
> **摘要:** Semi-supervised Camouflaged Object Detection (SSCOD) aims to reduce reliance on costly pixel-level annotations by leveraging limited annotated data and abundant unlabeled data. However, existing SSCOD methods based on Teacher-Student frameworks suffer from severe prediction bias and error propagation under scarce supervision, while their multi-network architectures incur high computational overhead and limited scalability. To overcome these limitations, we propose ST-SAM, a highly annotation-efficient yet concise framework that breaks away from conventional SSCOD constraints. Specifically, ST-SAM employs Self-Training strategy that dynamically filters and expands high-confidence pseudo-labels to enhance a single-model architecture, thereby fundamentally circumventing inter-model prediction bias. Furthermore, by transforming pseudo-labels into hybrid prompts containing domain-specific knowledge, ST-SAM effectively harnesses the Segment Anything Model's potential for specialized tasks to mitigate error accumulation in self-training. Experiments on COD benchmark datasets demonstrate that ST-SAM achieves state-of-the-art performance with only 1\% labeled data, outperforming existing SSCOD methods and even matching fully supervised methods. Remarkably, ST-SAM requires training only a single network, without relying on specific models or loss functions. This work establishes a new paradigm for annotation-efficient SSCOD. Codes will be available at https://github.com/hu-xh/ST-SAM.
>
---
#### [new 018] Single Image Rain Streak Removal Using Harris Corner Loss and R-CBAM Network
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像去雨任务，旨在去除单张雨天图像中的雨 streak 同时保留细节。作者提出一种新网络，引入Harris角点损失防止边界模糊，并设计R-CBAM模块增强特征表达，提升去雨效果。实验表明方法在Rain100L/H数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.23185v1](http://arxiv.org/pdf/2507.23185v1)**

> **作者:** Jongwook Si; Sungyoung Kim
>
> **备注:** 21 pages
>
> **摘要:** The problem of single-image rain streak removal goes beyond simple noise suppression, requiring the simultaneous preservation of fine structural details and overall visual quality. In this study, we propose a novel image restoration network that effectively constrains the restoration process by introducing a Corner Loss, which prevents the loss of object boundaries and detailed texture information during restoration. Furthermore, we propose a Residual Convolutional Block Attention Module (R-CBAM) Block into the encoder and decoder to dynamically adjust the importance of features in both spatial and channel dimensions, enabling the network to focus more effectively on regions heavily affected by rain streaks. Quantitative evaluations conducted on the Rain100L and Rain100H datasets demonstrate that the proposed method significantly outperforms previous approaches, achieving a PSNR of 33.29 dB on Rain100L and 26.16 dB on Rain100H.
>
---
#### [new 019] Recovering Diagnostic Value: Super-Resolution-Aided Echocardiographic Classification in Resource-Constrained Imaging
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决资源受限环境下超声心动图图像质量差影响诊断的问题。作者使用超分辨率技术提升低质量图像，并结合分类模型进行心脏视图和时相分类实验，验证了超分辨率在提升诊断准确性上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.23027v1](http://arxiv.org/pdf/2507.23027v1)**

> **作者:** Krishan Agyakari Raja Babu; Om Prabhu; Annu; Mohanasankar Sivaprakasam
>
> **备注:** Accepted at the MICCAI Workshop on "Medical Image Computing in Resource Constrained Settings & Knowledge Interchange (MIRASOL)" 2025
>
> **摘要:** Automated cardiac interpretation in resource-constrained settings (RCS) is often hindered by poor-quality echocardiographic imaging, limiting the effectiveness of downstream diagnostic models. While super-resolution (SR) techniques have shown promise in enhancing magnetic resonance imaging (MRI) and computed tomography (CT) scans, their application to echocardiography-a widely accessible but noise-prone modality-remains underexplored. In this work, we investigate the potential of deep learning-based SR to improve classification accuracy on low-quality 2D echocardiograms. Using the publicly available CAMUS dataset, we stratify samples by image quality and evaluate two clinically relevant tasks of varying complexity: a relatively simple Two-Chamber vs. Four-Chamber (2CH vs. 4CH) view classification and a more complex End-Diastole vs. End-Systole (ED vs. ES) phase classification. We apply two widely used SR models-Super-Resolution Generative Adversarial Network (SRGAN) and Super-Resolution Residual Network (SRResNet), to enhance poor-quality images and observe significant gains in performance metric-particularly with SRResNet, which also offers computational efficiency. Our findings demonstrate that SR can effectively recover diagnostic value in degraded echo scans, making it a viable tool for AI-assisted care in RCS, achieving more with less.
>
---
#### [new 020] CHECK-MAT: Checking Hand-Written Mathematical Answers for the Russian Unified State Exam
- **分类: cs.CV; cs.AI; cs.LG; 68T07, 97D50; I.2.7; I.4; K.3.1**

- **简介: 该论文属于AI评估手写数学解答任务，旨在解决自动化批改数学试卷的问题。作者构建了EGE-Math基准，包含122份俄罗斯高考数学手写解答及专家评分，评估多个视觉-语言模型的评分能力，揭示其在数学推理和评分标准对齐方面的不足。**

- **链接: [http://arxiv.org/pdf/2507.22958v1](http://arxiv.org/pdf/2507.22958v1)**

> **作者:** Ruslan Khrulev
>
> **备注:** 15 pages, 3 figures, 10 tables. Code is available at: https://github.com/Karifannaa/Auto-check-EGE-math
>
> **摘要:** This paper introduces a novel benchmark, EGE-Math Solutions Assessment Benchmark, for evaluating Vision-Language Models (VLMs) on their ability to assess hand-written mathematical solutions. Unlike existing benchmarks that focus on problem solving, our approach centres on understanding student solutions, identifying mistakes, and assigning grades according to fixed criteria. We compile 122 scanned solutions from the Russian Unified State Exam (EGE) together with official expert grades, and evaluate seven modern VLMs from Google, OpenAI, Arcee AI, and Alibaba Cloud in three inference modes. The results reveal current limitations in mathematical reasoning and human-rubric alignment, opening new research avenues in AI-assisted assessment. You can find code in https://github.com/Karifannaa/Auto-check-EGE-math
>
---
#### [new 021] Beyond Linear Bottlenecks: Spline-Based Knowledge Distillation for Culturally Diverse Art Style Classification
- **分类: cs.CV**

- **简介: 该论文属于艺术风格分类任务，旨在解决标注数据稀缺及风格特征复杂交互的问题。通过改进双教师知识蒸馏框架，引入Kolmogorov-Arnold Networks替代传统MLP，以更好建模非线性特征关联。实验表明其方法在WikiArt和Pandora18k数据集上提升了Top-1准确率。**

- **链接: [http://arxiv.org/pdf/2507.23436v1](http://arxiv.org/pdf/2507.23436v1)**

> **作者:** Abdellah Zakaria Sellam; Salah Eddine Bekhouche; Cosimo Distante; Abdelmalik Taleb-Ahmed
>
> **摘要:** Art style classification remains a formidable challenge in computational aesthetics due to the scarcity of expertly labeled datasets and the intricate, often nonlinear interplay of stylistic elements. While recent dual-teacher self-supervised frameworks reduce reliance on labeled data, their linear projection layers and localized focus struggle to model global compositional context and complex style-feature interactions. We enhance the dual-teacher knowledge distillation framework to address these limitations by replacing conventional MLP projection and prediction heads with Kolmogorov-Arnold Networks (KANs). Our approach retains complementary guidance from two teacher networks, one emphasizing localized texture and brushstroke patterns, the other capturing broader stylistic hierarchies while leveraging KANs' spline-based activations to model nonlinear feature correlations with mathematical precision. Experiments on WikiArt and Pandora18k demonstrate that our approach outperforms the base dual teacher architecture in Top-1 accuracy. Our findings highlight the importance of KANs in disentangling complex style manifolds, leading to better linear probe accuracy than MLP projections.
>
---
#### [new 022] Learning Semantic-Aware Threshold for Multi-Label Image Recognition with Partial Labels
- **分类: cs.CV**

- **简介: 该论文属于多标签图像识别任务，旨在解决部分标签数据下伪标签生成不准确的问题。作者提出了语义感知阈值学习算法（SATL），通过动态计算类别相关的分数分布和阈值，并引入差异排序损失增强判别能力，提升了标签识别的性能。**

- **链接: [http://arxiv.org/pdf/2507.23263v1](http://arxiv.org/pdf/2507.23263v1)**

> **作者:** Haoxian Ruan; Zhihua Xu; Zhijing Yang; Guang Ma; Jieming Xie; Changxiang Fan; Tianshui Chen
>
> **备注:** 15 pages, 13 figures, publish to ESWA (Expert Systems With Applications)
>
> **摘要:** Multi-label image recognition with partial labels (MLR-PL) is designed to train models using a mix of known and unknown labels. Traditional methods rely on semantic or feature correlations to create pseudo-labels for unidentified labels using pre-set thresholds. This approach often overlooks the varying score distributions across categories, resulting in inaccurate and incomplete pseudo-labels, thereby affecting performance. In our study, we introduce the Semantic-Aware Threshold Learning (SATL) algorithm. This innovative approach calculates the score distribution for both positive and negative samples within each category and determines category-specific thresholds based on these distributions. These distributions and thresholds are dynamically updated throughout the learning process. Additionally, we implement a differential ranking loss to establish a significant gap between the score distributions of positive and negative samples, enhancing the discrimination of the thresholds. Comprehensive experiments and analysis on large-scale multi-label datasets, such as Microsoft COCO and VG-200, demonstrate that our method significantly improves performance in scenarios with limited labels.
>
---
#### [new 023] Short-LVLM: Compressing and Accelerating Large Vision-Language Models by Pruning Redundant Layers
- **分类: cs.CV**

- **简介: 该论文属于模型压缩任务，旨在解决大型视觉-语言模型（LVLMs）因参数量大、计算成本高而限制应用的问题。作者发现直接应用NLP中的层剪枝方法无效，因此提出新框架Short-LVLM，通过利用关键视觉-语言标记和减少层间特征差异，实现高效压缩，兼顾性能与效率，且无需训练、适用于不同模型。**

- **链接: [http://arxiv.org/pdf/2507.23362v1](http://arxiv.org/pdf/2507.23362v1)**

> **作者:** Ji Ma; Wei Suo; Peng Wang; Yanning Zhang
>
> **备注:** Accepted By ACM MM 25
>
> **摘要:** Although large vision-language models (LVLMs) have demonstrated impressive capabilities in multi-modal understanding and reasoning, their practical applications are still limited by massive model parameters and high computational costs. Recent efforts from natural language processing (NLP) have shown the effectiveness of layer pruning, offering a plausible training-free compression solution. However, due to the modality divergence between vision and language, it is unclear whether these NLP techniques are still effective in LVLMs. In this paper, we empirically prove that directly applying these layer pruning methods to LVLMs is ineffective. Through extensive experiments, we find that non-essential vision-language (VL) tokens and inter-layer feature gaps pose critical challenges to pruning layers in LVLMs. Based on these insights, we propose a novel framework Short-LVLM (SVL) that can utilize important VL tokens and mitigate the layer-wise feature gaps. Notably, Short-LVLM not only achieves a superior trade-off between performance and efficiency but also exhibits several potential advantages, i.e., training-free, model-agnostic, and highly compatible. The code for this work is publicly available at https://github.com/ASGO-MM/Short-LVLM.
>
---
#### [new 024] Vocabulary-free Fine-grained Visual Recognition via Enriched Contextually Grounded Vision-Language Model
- **分类: cs.CV**

- **简介: 该论文属于细粒度视觉识别任务，旨在解决传统方法依赖固定词汇和封闭类别限制可扩展性的问题。作者提出了一种无需训练的方法E-FineR，结合大语言模型与视觉语言模型，实现开放集识别，并提升可解释性与适应性。**

- **链接: [http://arxiv.org/pdf/2507.23070v1](http://arxiv.org/pdf/2507.23070v1)**

> **作者:** Dmitry Demidov; Zaigham Zaheer; Omkar Thawakar; Salman Khan; Fahad Shahbaz Khan
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Fine-grained image classification, the task of distinguishing between visually similar subcategories within a broader category (e.g., bird species, car models, flower types), is a challenging computer vision problem. Traditional approaches rely heavily on fixed vocabularies and closed-set classification paradigms, limiting their scalability and adaptability in real-world settings where novel classes frequently emerge. Recent research has demonstrated that combining large language models (LLMs) with vision-language models (VLMs) makes open-set recognition possible without the need for predefined class labels. However, the existing methods are often limited in harnessing the power of LLMs at the classification phase, and also rely heavily on the guessed class names provided by an LLM without thorough analysis and refinement. To address these bottlenecks, we propose our training-free method, Enriched-FineR (or E-FineR for short), which demonstrates state-of-the-art results in fine-grained visual recognition while also offering greater interpretability, highlighting its strong potential in real-world scenarios and new domains where expert annotations are difficult to obtain. Additionally, we demonstrate the application of our proposed approach to zero-shot and few-shot classification, where it demonstrated performance on par with the existing SOTA while being training-free and not requiring human interventions. Overall, our vocabulary-free framework supports the shift in image classification from rigid label prediction to flexible, language-driven understanding, enabling scalable and generalizable systems for real-world applications. Well-documented code is available on https://github.com/demidovd98/e-finer.
>
---
#### [new 025] FFGAF-SNN: The Forward-Forward Based Gradient Approximation Free Training Framework for Spiking Neural Networks
- **分类: cs.CV**

- **简介: 该论文属于神经网络训练方法研究，旨在解决脉冲神经网络（SNN）因非可导性导致的训练难题。作者提出了一种无需梯度近似的前向训练框架FFGAF-SNN，并引入类别感知的复杂度自适应机制，优化损失函数，提升分类准确率和计算效率。**

- **链接: [http://arxiv.org/pdf/2507.23643v1](http://arxiv.org/pdf/2507.23643v1)**

> **作者:** Changqing Xu; Ziqiang Yang; Yi Liu; Xinfang Liao; Guiqi Mo; Hao Zeng; Yintang Yang
>
> **摘要:** Spiking Neural Networks (SNNs) offer a biologically plausible framework for energy-efficient neuromorphic computing. However, it is a challenge to train SNNs due to their non-differentiability, efficiently. Existing gradient approximation approaches frequently sacrifice accuracy and face deployment limitations on edge devices due to the substantial computational requirements of backpropagation. To address these challenges, we propose a Forward-Forward (FF) based gradient approximation-free training framework for Spiking Neural Networks, which treats spiking activations as black-box modules, thereby eliminating the need for gradient approximation while significantly reducing computational complexity. Furthermore, we introduce a class-aware complexity adaptation mechanism that dynamically optimizes the loss function based on inter-class difficulty metrics, enabling efficient allocation of network resources across different categories. Experimental results demonstrate that our proposed training framework achieves test accuracies of 99.58%, 92.13%, and 75.64% on the MNIST, Fashion-MNIST, and CIFAR-10 datasets, respectively, surpassing all existing FF-based SNN approaches. Additionally, our proposed method exhibits significant advantages in terms of memory access and computational power consumption.
>
---
#### [new 026] Consistent Point Matching
- **分类: cs.CV; cs.DC; cs.LG**

- **简介: 该论文属于医学图像配准任务，旨在解决跨模态医疗图像中解剖位置匹配不准确的问题。作者在点匹配算法中引入一致性启发式方法，提高了匹配鲁棒性与精度，无需训练数据或机器学习模型，适用于CT与MRI图像的高精度导航与病灶追踪。**

- **链接: [http://arxiv.org/pdf/2507.23609v1](http://arxiv.org/pdf/2507.23609v1)**

> **作者:** Halid Ziya Yerebakan; Gerardo Hermosillo Valadez
>
> **摘要:** This study demonstrates that incorporating a consistency heuristic into the point-matching algorithm \cite{yerebakan2023hierarchical} improves robustness in matching anatomical locations across pairs of medical images. We validated our approach on diverse longitudinal internal and public datasets spanning CT and MRI modalities. Notably, it surpasses state-of-the-art results on the Deep Lesion Tracking dataset. Additionally, we show that the method effectively addresses landmark localization. The algorithm operates efficiently on standard CPU hardware and allows configurable trade-offs between speed and robustness. The method enables high-precision navigation between medical images without requiring a machine learning model or training data.
>
---
#### [new 027] Early Goal-Guided Multi-Scale Fusion for Real-Time Vision-Language Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO; I.2.6; I.2.9; I.2.10; C.3.3**

- **简介: 该论文属于自动驾驶任务，旨在提升车辆在复杂场景中的实时导航能力与安全性。论文提出了NovaDrive，一种单分支视觉-语言架构，融合图像、高精地图、LiDAR和文本路径点信息，通过跨注意力机制与平滑损失优化路径，减少碰撞并提高行驶效率。**

- **链接: [http://arxiv.org/pdf/2507.23042v1](http://arxiv.org/pdf/2507.23042v1)**

> **作者:** Santosh Patapati; Trisanth Srinivasan
>
> **备注:** 6 pages
>
> **摘要:** Autonomous vehicles must react in milliseconds while reasoning about road geometry and traffic intent to navigate complex situations. We introduce NovaDrive, a single-branch vision-language architecture that processes front-camera images, HD-map tiles, LiDAR depth, and textual waypoints in a single branch. A lightweight, two-stage cross-attention block first aligns waypoint tokens with the HD map, then refines attention over fine-grained image and depth patches. Coupled with a novel smoothness loss that discourages abrupt steering and speed changes, this design eliminates the need for recurrent memory. We fine-tune the top 15 layers of an 11B LLaMA-3.2 vision-language backbone, enabling real-time inference. On the nuScenes / Waymo subset of the MD-NEX Outdoor benchmark, NovaDrive raises success rate to 84% (+4%), boosts path-efficiency (SPL) to 0.66 (+0.11), and reduces collision frequency from 2.6% to 1.2% (-1.4%) relative to the previous state-of-the-art. Our ablations confirm that waypoint tokens, partial VLM fine-tuning, and the cross-attention fusion each contribute the most to these gains. Beyond safety, NovaDrive's shorter routes (resulting from the novel smoothness loss) translate to lower fuel or battery usage, pointing toward leaner, more easily updated driving stacks. NovaDrive can be extended to other embodied-AI domains as well.
>
---
#### [new 028] Adaptively Distilled ControlNet: Accelerated Training and Superior Sampling for Medical Image Synthesis
- **分类: cs.CV**

- **简介: 论文提出“Adaptively Distilled ControlNet”，用于医学图像合成任务，解决隐私和标注限制导致的模型性能问题。通过双模型蒸馏训练，提升生成图像的病变区域准确性，采样时仅用学生模型，保障隐私。在两个医学数据集上验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2507.23652v1](http://arxiv.org/pdf/2507.23652v1)**

> **作者:** Kunpeng Qiu; Zhiying Zhou; Yongxin Guo
>
> **备注:** Accepted by MICCAI2025
>
> **摘要:** Medical image annotation is constrained by privacy concerns and labor-intensive labeling, significantly limiting the performance and generalization of segmentation models. While mask-controllable diffusion models excel in synthesis, they struggle with precise lesion-mask alignment. We propose \textbf{Adaptively Distilled ControlNet}, a task-agnostic framework that accelerates training and optimization through dual-model distillation. Specifically, during training, a teacher model, conditioned on mask-image pairs, regularizes a mask-only student model via predicted noise alignment in parameter space, further enhanced by adaptive regularization based on lesion-background ratios. During sampling, only the student model is used, enabling privacy-preserving medical image generation. Comprehensive evaluations on two distinct medical datasets demonstrate state-of-the-art performance: TransUNet improves mDice/mIoU by 2.4%/4.2% on KiTS19, while SANet achieves 2.6%/3.5% gains on Polyps, highlighting its effectiveness and superiority. Code is available at GitHub.
>
---
#### [new 029] Learning Semantic Directions for Feature Augmentation in Domain-Generalized Medical Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决跨中心/跨域数据导致的模型性能下降问题。作者提出一种领域泛化框架，通过学习语义方向和特征扰动，增强模型对不同成像条件的鲁棒性，同时保持解剖结构一致性，从而提升分割的稳定性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.23326v1](http://arxiv.org/pdf/2507.23326v1)**

> **作者:** Yingkai Wang; Yaoyao Zhu; Xiuding Cai; Yuhao Xiao; Haotian Wu; Yu Yao
>
> **摘要:** Medical image segmentation plays a crucial role in clinical workflows, but domain shift often leads to performance degradation when models are applied to unseen clinical domains. This challenge arises due to variations in imaging conditions, scanner types, and acquisition protocols, limiting the practical deployment of segmentation models. Unlike natural images, medical images typically exhibit consistent anatomical structures across patients, with domain-specific variations mainly caused by imaging conditions. This unique characteristic makes medical image segmentation particularly challenging. To address this challenge, we propose a domain generalization framework tailored for medical image segmentation. Our approach improves robustness to domain-specific variations by introducing implicit feature perturbations guided by domain statistics. Specifically, we employ a learnable semantic direction selector and a covariance-based semantic intensity sampler to modulate domain-variant features while preserving task-relevant anatomical consistency. Furthermore, we design an adaptive consistency constraint that is selectively applied only when feature adjustment leads to degraded segmentation performance. This constraint encourages the adjusted features to align with the original predictions, thereby stabilizing feature selection and improving the reliability of the segmentation. Extensive experiments on two public multi-center benchmarks show that our framework consistently outperforms existing domain generalization approaches, achieving robust and generalizable segmentation performance across diverse clinical domains.
>
---
#### [new 030] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在从单视角图像生成高保真3D高斯化身。现有方法依赖2D扩散模型生成多视角图像，但存在不一致和模糊问题。作者提出MoGA，利用生成式3D化身模型作为先验，结合2D扩散模型生成的视图进行模型反演，确保3D一致性并提升细节真实性。**

- **链接: [http://arxiv.org/pdf/2507.23597v1](http://arxiv.org/pdf/2507.23597v1)**

> **作者:** Zijian Dong; Longteng Duan; Jie Song; Michael J. Black; Andreas Geiger
>
> **备注:** ICCV 2025 (Highlight), Project Page: https://zj-dong.github.io/MoGA/
>
> **摘要:** We present MoGA, a novel method to reconstruct high-fidelity 3D Gaussian avatars from a single-view image. The main challenge lies in inferring unseen appearance and geometric details while ensuring 3D consistency and realism. Most previous methods rely on 2D diffusion models to synthesize unseen views; however, these generated views are sparse and inconsistent, resulting in unrealistic 3D artifacts and blurred appearance. To address these limitations, we leverage a generative avatar model, that can generate diverse 3D avatars by sampling deformed Gaussians from a learned prior distribution. Due to the limited amount of 3D training data such a 3D model alone cannot capture all image details of unseen identities. Consequently, we integrate it as a prior, ensuring 3D consistency by projecting input images into its latent space and enforcing additional 3D appearance and geometric constraints. Our novel approach formulates Gaussian avatar creation as a model inversion process by fitting the generative avatar to synthetic views from 2D diffusion models. The generative avatar provides a meaningful initialization for model fitting, enforces 3D regularization, and helps in refining pose estimation. Experiments show that our method surpasses state-of-the-art techniques and generalizes well to real-world scenarios. Our Gaussian avatars are also inherently animatable
>
---
#### [new 031] DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching
- **分类: cs.CV**

- **简介: 该论文属于非刚性形状匹配任务，旨在解决现有方法依赖先验模型假设导致适用性受限的问题。作者提出DiffuMatch，首次将网络内正则化和功能映射训练完全替换为数据驱动方法，通过谱域扩散模型生成功能映射，实现零样本下更准确的匹配。**

- **链接: [http://arxiv.org/pdf/2507.23715v1](http://arxiv.org/pdf/2507.23715v1)**

> **作者:** Emery Pierson; Lei Li; Angela Dai; Maks Ovsjanikov
>
> **备注:** Presented at ICCV 2025
>
> **摘要:** Deep functional maps have recently emerged as a powerful tool for solving non-rigid shape correspondence tasks. Methods that use this approach combine the power and flexibility of the functional map framework, with data-driven learning for improved accuracy and generality. However, most existing methods in this area restrict the learning aspect only to the feature functions and still rely on axiomatic modeling for formulating the training loss or for functional map regularization inside the networks. This limits both the accuracy and the applicability of the resulting approaches only to scenarios where assumptions of the axiomatic models hold. In this work, we show, for the first time, that both in-network regularization and functional map training can be replaced with data-driven methods. For this, we first train a generative model of functional maps in the spectral domain using score-based generative modeling, built from a large collection of high-quality maps. We then exploit the resulting model to promote the structural properties of ground truth functional maps on new shape collections. Remarkably, we demonstrate that the learned models are category-agnostic, and can fully replace commonly used strategies such as enforcing Laplacian commutativity or orthogonality of functional maps. Our key technical contribution is a novel distillation strategy from diffusion models in the spectral domain. Experiments demonstrate that our learned regularization leads to better results than axiomatic approaches for zero-shot non-rigid shape matching. Our code is available at: https://github.com/daidedou/diffumatch/
>
---
#### [new 032] Generalized Reinforcement Learning for Retriever-Specific Query Rewriter with Unstructured Real-World Documents
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于检索增强生成（RAG）系统中的查询优化任务，旨在解决在多样、非结构化现实文档中有效查询构建的挑战。论文提出了RL-QR，一种基于强化学习的检索器专用查询重写框架，无需人工标注数据，适用于文本和多模态数据库。通过合成场景-问题对并采用GRPO训练策略，提升了检索性能，尤其在多模态和词法检索中取得显著改进。**

- **链接: [http://arxiv.org/pdf/2507.23242v1](http://arxiv.org/pdf/2507.23242v1)**

> **作者:** Sungguk Cha; DongWook Kim; Taeseung Hahn; Mintae Kim; Youngsub Han; Byoung-Ki Jeon
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems rely heavily on effective query formulation to unlock external knowledge, yet optimizing queries for diverse, unstructured real-world documents remains a challenge. We introduce \textbf{RL-QR}, a reinforcement learning framework for retriever-specific query rewriting that eliminates the need for human-annotated datasets and extends applicability to both text-only and multi-modal databases. By synthesizing scenario-question pairs and leveraging Generalized Reward Policy Optimization (GRPO), RL-QR trains query rewriters tailored to specific retrievers, enhancing retrieval performance across varied domains. Experiments on industrial in-house data demonstrate significant improvements, with $\text{RL-QR}_{\text{multi-modal}}$ achieving an 11\% relative gain in NDCG@3 for multi-modal RAG and $\text{RL-QR}_{\text{lexical}}$ yielding a 9\% gain for lexical retrievers. However, challenges persist with semantic and hybrid retrievers, where rewriters failed to improve performance, likely due to training misalignments. Our findings highlight RL-QR's potential to revolutionize query optimization for RAG systems, offering a scalable, annotation-free solution for real-world retrieval tasks, while identifying avenues for further refinement in semantic retrieval contexts.
>
---
#### [new 033] LED Benchmark: Diagnosing Structural Layout Errors for Document Layout Analysis
- **分类: cs.CV**

- **简介: 该论文属于文档布局分析任务，旨在解决现有评估指标无法有效检测结构错误的问题。作者提出了LED基准，定义八种结构错误类型，并构建了包含合成数据的LED-Dataset，用于评估模型的结构鲁棒性。实验表明，LED能更全面地反映模型在结构理解上的优劣。**

- **链接: [http://arxiv.org/pdf/2507.23295v1](http://arxiv.org/pdf/2507.23295v1)**

> **作者:** Inbum Heo; Taewook Hwang; Jeesu Jung; Sangkeun Jung
>
> **摘要:** Recent advancements in Document Layout Analysis through Large Language Models and Multimodal Models have significantly improved layout detection. However, despite these improvements, challenges remain in addressing critical structural errors, such as region merging, splitting, and missing content. Conventional evaluation metrics like IoU and mAP, which focus primarily on spatial overlap, are insufficient for detecting these errors. To address this limitation, we propose Layout Error Detection (LED), a novel benchmark designed to evaluate the structural robustness of document layout predictions. LED defines eight standardized error types, and formulates three complementary tasks: error existence detection, error type classification, and element-wise error type classification. Furthermore, we construct LED-Dataset, a synthetic dataset generated by injecting realistic structural errors based on empirical distributions from DLA models. Experimental results across a range of LMMs reveal that LED effectively differentiates structural understanding capabilities, exposing modality biases and performance trade-offs not visible through traditional metrics.
>
---
#### [new 034] Honey Adulteration Detection using Hyperspectral Imaging and Machine Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于食品检测任务，旨在解决蜂蜜掺假问题。通过构建机器学习系统，利用高光谱成像技术对蜂蜜的植物来源进行分类，并检测糖浆掺假及其浓度。系统采用LDA提取特征，结合KNN模型实现分类，验证准确率达96.39%，为化学检测提供了高效替代方案。**

- **链接: [http://arxiv.org/pdf/2507.23416v1](http://arxiv.org/pdf/2507.23416v1)**

> **作者:** Mokhtar A. Al-Awadhi; Ratnadeep R. Deshmukh
>
> **摘要:** This paper aims to develop a machine learning-based system for automatically detecting honey adulteration with sugar syrup, based on honey hyperspectral imaging data. First, the floral source of a honey sample is classified by a botanical origin identification subsystem. Then, the sugar syrup adulteration is identified, and its concentration is quantified by an adulteration detection subsystem. Both subsystems consist of two steps. The first step involves extracting relevant features from the honey sample using Linear Discriminant Analysis (LDA). In the second step, we utilize the K-Nearest Neighbors (KNN) model to classify the honey botanical origin in the first subsystem and identify the adulteration level in the second subsystem. We assess the proposed system performance on a public honey hyperspectral image dataset. The result indicates that the proposed system can detect adulteration in honey with an overall cross-validation accuracy of 96.39%, making it an appropriate alternative to the current chemical-based detection methods.
>
---
#### [new 035] Modeling Human Gaze Behavior with Diffusion Models for Unified Scanpath Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉注意力建模任务，旨在解决现有模型生成平均化眼动路径、缺乏个体差异性的问题。作者提出ScanDiff，结合扩散模型与视觉Transformer，生成多样化、逼真的眼动轨迹，并引入文本条件控制以适应不同视觉搜索目标，提升了预测的多样性和准确性。**

- **链接: [http://arxiv.org/pdf/2507.23021v1](http://arxiv.org/pdf/2507.23021v1)**

> **作者:** Giuseppe Cartella; Vittorio Cuculo; Alessandro D'Amelio; Marcella Cornia; Giuseppe Boccignone; Rita Cucchiara
>
> **备注:** Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Predicting human gaze scanpaths is crucial for understanding visual attention, with applications in human-computer interaction, autonomous systems, and cognitive robotics. While deep learning models have advanced scanpath prediction, most existing approaches generate averaged behaviors, failing to capture the variability of human visual exploration. In this work, we present ScanDiff, a novel architecture that combines diffusion models with Vision Transformers to generate diverse and realistic scanpaths. Our method explicitly models scanpath variability by leveraging the stochastic nature of diffusion models, producing a wide range of plausible gaze trajectories. Additionally, we introduce textual conditioning to enable task-driven scanpath generation, allowing the model to adapt to different visual search objectives. Experiments on benchmark datasets show that ScanDiff surpasses state-of-the-art methods in both free-viewing and task-driven scenarios, producing more diverse and accurate scanpaths. These results highlight its ability to better capture the complexity of human visual behavior, pushing forward gaze prediction research. Source code and models are publicly available at https://aimagelab.github.io/ScanDiff.
>
---
#### [new 036] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis
- **分类: cs.CV**

- **简介: 该论文属于视频到4D生成任务，旨在从单个视频生成高质量动态3D内容。由于直接建模4D扩散成本高且复杂，作者提出一种新框架，使用无需实例拟合的VAE编码高斯点云及其时间变化，并结合扩散模型与时间感知变压器，实现从视频生成高质量3D动画。**

- **链接: [http://arxiv.org/pdf/2507.23785v1](http://arxiv.org/pdf/2507.23785v1)**

> **作者:** Bowen Zhang; Sicheng Xu; Chuxin Wang; Jiaolong Yang; Feng Zhao; Dong Chen; Baining Guo
>
> **备注:** ICCV 2025. Project page: https://gvfdiffusion.github.io/
>
> **摘要:** In this paper, we present a novel framework for video-to-4D generation that creates high-quality dynamic 3D content from single video inputs. Direct 4D diffusion modeling is extremely challenging due to costly data construction and the high-dimensional nature of jointly representing 3D shape, appearance, and motion. We address these challenges by introducing a Direct 4DMesh-to-GS Variation Field VAE that directly encodes canonical Gaussian Splats (GS) and their temporal variations from 3D animation data without per-instance fitting, and compresses high-dimensional animations into a compact latent space. Building upon this efficient representation, we train a Gaussian Variation Field diffusion model with temporal-aware Diffusion Transformer conditioned on input videos and canonical GS. Trained on carefully-curated animatable 3D objects from the Objaverse dataset, our model demonstrates superior generation quality compared to existing methods. It also exhibits remarkable generalization to in-the-wild video inputs despite being trained exclusively on synthetic data, paving the way for generating high-quality animated 3D content. Project page: https://gvfdiffusion.github.io/.
>
---
#### [new 037] FastPoint: Accelerating 3D Point Cloud Model Inference via Sample Point Distance Prediction
- **分类: cs.CV**

- **简介: 该论文属于3D点云处理任务，旨在解决大规模不规则点云推理效率低的问题。作者提出FastPoint方法，通过预测采样点间距离趋势，加速最远点采样和邻域搜索，实现模型推理速度提升2.55倍，且不损失精度。**

- **链接: [http://arxiv.org/pdf/2507.23480v1](http://arxiv.org/pdf/2507.23480v1)**

> **作者:** Donghyun Lee; Dawoon Jeong; Jae W. Lee; Hongil Yoon
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Deep neural networks have revolutionized 3D point cloud processing, yet efficiently handling large and irregular point clouds remains challenging. To tackle this problem, we introduce FastPoint, a novel software-based acceleration technique that leverages the predictable distance trend between sampled points during farthest point sampling. By predicting the distance curve, we can efficiently identify subsequent sample points without exhaustively computing all pairwise distances. Our proposal substantially accelerates farthest point sampling and neighbor search operations while preserving sampling quality and model performance. By integrating FastPoint into state-of-the-art 3D point cloud models, we achieve 2.55x end-to-end speedup on NVIDIA RTX 3090 GPU without sacrificing accuracy.
>
---
#### [new 038] PixNerd: Pixel Neural Field Diffusion
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型依赖预训练VAE导致的误差累积和解码伪影问题。论文提出PixNerd，采用基于神经场的块解码方法，实现单阶段、端到端的高效图像生成，在ImageNet上取得优异FID表现，并扩展至文本生成图像任务。**

- **链接: [http://arxiv.org/pdf/2507.23268v1](http://arxiv.org/pdf/2507.23268v1)**

> **作者:** Shuai Wang; Ziteng Gao; Chenhui Zhu; Weilin Huang; Limin Wang
>
> **备注:** a single-scale, single-stage, efficient, end-to-end pixel space diffusion model
>
> **摘要:** The current success of diffusion transformers heavily depends on the compressed latent space shaped by the pre-trained variational autoencoder(VAE). However, this two-stage training paradigm inevitably introduces accumulated errors and decoding artifacts. To address the aforementioned problems, researchers return to pixel space at the cost of complicated cascade pipelines and increased token complexity. In contrast to their efforts, we propose to model the patch-wise decoding with neural field and present a single-scale, single-stage, efficient, end-to-end solution, coined as pixel neural field diffusion~(PixelNerd). Thanks to the efficient neural field representation in PixNerd, we directly achieved 2.15 FID on ImageNet $256\times256$ and 2.84 FID on ImageNet $512\times512$ without any complex cascade pipeline or VAE. We also extend our PixNerd framework to text-to-image applications. Our PixNerd-XXL/16 achieved a competitive 0.73 overall score on the GenEval benchmark and 80.9 overall score on the DPG benchmark.
>
---
#### [new 039] Adaptive Time-step Training for Enhancing Spike-Based Neural Radiance Fields
- **分类: cs.CV; cs.NE**

- **简介: 该论文属于3D重建与渲染任务，旨在解决NeRF模型计算量大、能耗高的问题。作者提出了一种基于脉冲神经网络（SNN）的NeRF框架PATA，通过自适应调整训练与推理的时间步长，在保持渲染质量的同时显著降低计算资源消耗，提升能效。**

- **链接: [http://arxiv.org/pdf/2507.23033v1](http://arxiv.org/pdf/2507.23033v1)**

> **作者:** Ranxi Lin; Canming Yao; Jiayi Li; Weihang Liu; Xin Lou; Pingqiang Zhou
>
> **摘要:** Neural Radiance Fields (NeRF)-based models have achieved remarkable success in 3D reconstruction and rendering tasks. However, during both training and inference, these models rely heavily on dense point sampling along rays from multiple viewpoints, resulting in a surge in floating-point operations and severely limiting their use in resource-constrained scenarios like edge computing. Spiking Neural Networks (SNNs), which communicate via binary spikes over discrete time steps, offer a promising alternative due to their energy-efficient nature. Given the inherent variability in scene scale and texture complexity in neural rendering and the prevailing practice of training separate models per scene, we propose a spike-based NeRF framework with a dynamic time step training strategy, termed Pretrain-Adaptive Time-step Adjustment (PATA). This approach automatically explores the trade-off between rendering quality and time step length during training. Consequently, it enables scene-adaptive inference with variable time steps and reduces the additional consumption of computational resources in the inference process. Anchoring to the established Instant-NGP architecture, we evaluate our method across diverse datasets. The experimental results show that PATA can preserve rendering fidelity while reducing inference time steps by 64\% and running power by 61.55\%.
>
---
#### [new 040] Ambiguity-Guided Learnable Distribution Calibration for Semi-Supervised Few-Shot Class-Incremental Learning
- **分类: cs.CV**

- **简介: 该论文属于半监督小样本类增量学习任务，旨在解决新旧类别样本分布偏差问题。作者提出ALDC方法，利用基础类样本动态校正新类特征分布，提升模型在广义半监督场景下的性能，实验证明其效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.23237v1](http://arxiv.org/pdf/2507.23237v1)**

> **作者:** Fan Lyu; Linglan Zhao; Chengyan Liu; Yinying Mei; Zhang Zhang; Jian Zhang; Fuyuan Hu; Liang Wang
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** Few-Shot Class-Incremental Learning (FSCIL) focuses on models learning new concepts from limited data while retaining knowledge of previous classes. Recently, many studies have started to leverage unlabeled samples to assist models in learning from few-shot samples, giving rise to the field of Semi-supervised Few-shot Class-Incremental Learning (Semi-FSCIL). However, these studies often assume that the source of unlabeled data is only confined to novel classes of the current session, which presents a narrow perspective and cannot align well with practical scenarios. To better reflect real-world scenarios, we redefine Semi-FSCIL as Generalized Semi-FSCIL (GSemi-FSCIL) by incorporating both base and all the ever-seen novel classes in the unlabeled set. This change in the composition of unlabeled samples poses a new challenge for existing methods, as they struggle to distinguish between unlabeled samples from base and novel classes. To address this issue, we propose an Ambiguity-guided Learnable Distribution Calibration (ALDC) strategy. ALDC dynamically uses abundant base samples to correct biased feature distributions for few-shot novel classes. Experiments on three benchmark datasets show that our method outperforms existing works, setting new state-of-the-art results.
>
---
#### [new 041] CST Anti-UAV: A Thermal Infrared Benchmark for Tiny UAV Tracking in Complex Scenes
- **分类: cs.CV**

- **简介: 该论文属于目标跟踪任务，旨在解决复杂场景中微型无人机（UAV）跟踪的难题。现有数据集缺乏多样性和精细标注，限制了实际应用。作者构建了CST Anti-UAV热红外数据集，包含220个视频序列和24万标注框，并提供细粒度属性标注。实验表明，当前方法在该数据集上表现较差，凸显其挑战性和研究必要性。**

- **链接: [http://arxiv.org/pdf/2507.23473v1](http://arxiv.org/pdf/2507.23473v1)**

> **作者:** Bin Xie; Congxuan Zhang; Fagan Wang; Peng Liu; Feng Lu; Zhen Chen; Weiming Hu
>
> **备注:** Accepted by ICCVW2025
>
> **摘要:** The widespread application of Unmanned Aerial Vehicles (UAVs) has raised serious public safety and privacy concerns, making UAV perception crucial for anti-UAV tasks. However, existing UAV tracking datasets predominantly feature conspicuous objects and lack diversity in scene complexity and attribute representation, limiting their applicability to real-world scenarios. To overcome these limitations, we present the CST Anti-UAV, a new thermal infrared dataset specifically designed for Single Object Tracking (SOT) in Complex Scenes with Tiny UAVs (CST). It contains 220 video sequences with over 240k high-quality bounding box annotations, highlighting two key properties: a significant number of tiny-sized UAV targets and the diverse and complex scenes. To the best of our knowledge, CST Anti-UAV is the first dataset to incorporate complete manual frame-level attribute annotations, enabling precise evaluations under varied challenges. To conduct an in-depth performance analysis for CST Anti-UAV, we evaluate 20 existing SOT methods on the proposed dataset. Experimental results demonstrate that tracking tiny UAVs in complex environments remains a challenge, as the state-of-the-art method achieves only 35.92% state accuracy, much lower than the 67.69% observed on the Anti-UAV410 dataset. These findings underscore the limitations of existing benchmarks and the need for further advancements in UAV tracking research. The CST Anti-UAV benchmark is about to be publicly released, which not only fosters the development of more robust SOT methods but also drives innovation in anti-UAV systems.
>
---
#### [new 042] I2V-GS: Infrastructure-to-Vehicle View Transformation with Gaussian Splatting for Autonomous Driving Data Generation
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶数据生成任务，旨在解决车载数据采集成本高、效率低的问题。论文提出I2V-GS方法，通过基础设施视角到车载视角的高斯溅射转换，实现高质量驾驶数据合成，并引入新数据集RoadSight。**

- **链接: [http://arxiv.org/pdf/2507.23683v1](http://arxiv.org/pdf/2507.23683v1)**

> **作者:** Jialei Chen; Wuhao Xu; Sipeng He; Baoru Huang; Dongchun Ren
>
> **摘要:** Vast and high-quality data are essential for end-to-end autonomous driving systems. However, current driving data is mainly collected by vehicles, which is expensive and inefficient. A potential solution lies in synthesizing data from real-world images. Recent advancements in 3D reconstruction demonstrate photorealistic novel view synthesis, highlighting the potential of generating driving data from images captured on the road. This paper introduces a novel method, I2V-GS, to transfer the Infrastructure view To the Vehicle view with Gaussian Splatting. Reconstruction from sparse infrastructure viewpoints and rendering under large view transformations is a challenging problem. We adopt the adaptive depth warp to generate dense training views. To further expand the range of views, we employ a cascade strategy to inpaint warped images, which also ensures inpainting content is consistent across views. To further ensure the reliability of the diffusion model, we utilize the cross-view information to perform a confidenceguided optimization. Moreover, we introduce RoadSight, a multi-modality, multi-view dataset from real scenarios in infrastructure views. To our knowledge, I2V-GS is the first framework to generate autonomous driving datasets with infrastructure-vehicle view transformation. Experimental results demonstrate that I2V-GS significantly improves synthesis quality under vehicle view, outperforming StreetGaussian in NTA-Iou, NTL-Iou, and FID by 45.7%, 34.2%, and 14.9%, respectively.
>
---
#### [new 043] OmniTraj: Pre-Training on Heterogeneous Data for Adaptive and Zero-Shot Human Trajectory Prediction
- **分类: cs.CV**

- **简介: 该论文属于人类轨迹预测任务，旨在解决现有预训练模型在未见数据集上零样本迁移能力差的问题。作者提出OmniTraj模型，通过显式条件化时间元数据，实现对不同时态设置的泛化。该模型在多数据集上表现优异，提升了零样本迁移性能。**

- **链接: [http://arxiv.org/pdf/2507.23657v1](http://arxiv.org/pdf/2507.23657v1)**

> **作者:** Yang Gao; Po-Chien Luan; Kaouther Messaoud; Lan Feng; Alexandre Alahi
>
> **摘要:** While large-scale pre-training has advanced human trajectory prediction, a critical challenge remains: zero-shot transfer to unseen dataset with varying temporal dynamics. State-of-the-art pre-trained models often require fine-tuning to adapt to new datasets with different frame rates or observation horizons, limiting their scalability and practical utility. In this work, we systematically investigate this limitation and propose a robust solution. We first demonstrate that existing data-aware discrete models struggle when transferred to new scenarios with shifted temporal setups. We then isolate the temporal generalization from dataset shift, revealing that a simple, explicit conditioning mechanism for temporal metadata is a highly effective solution. Based on this insight, we present OmniTraj, a Transformer-based model pre-trained on a large-scale, heterogeneous dataset. Our experiments show that explicitly conditioning on the frame rate enables OmniTraj to achieve state-of-the-art zero-shot transfer performance, reducing prediction error by over 70\% in challenging cross-setup scenarios. After fine-tuning, OmniTraj achieves state-of-the-art results on four datasets, including NBA, JTA, WorldPose, and ETH-UCY. The code is publicly available: https://github.com/vita-epfl/omnitraj
>
---
#### [new 044] UniLDiff: Unlocking the Power of Diffusion Priors for All-in-One Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决多种图像退化问题。作者提出UniLDiff框架，结合扩散模型与退化感知模块，提升修复效果。工作包括设计DAFF模块与DAEM模块，优化多退化场景下的图像复原。**

- **链接: [http://arxiv.org/pdf/2507.23685v1](http://arxiv.org/pdf/2507.23685v1)**

> **作者:** Zihan Cheng; Liangtai Zhou; Dian Chen; Ni Tang; Xiaotong Luo; Yanyun Qu
>
> **摘要:** All-in-One Image Restoration (AiOIR) has emerged as a promising yet challenging research direction. To address its core challenges, we propose a novel unified image restoration framework based on latent diffusion models (LDMs). Our approach structurally integrates low-quality visual priors into the diffusion process, unlocking the powerful generative capacity of diffusion models for diverse degradations. Specifically, we design a Degradation-Aware Feature Fusion (DAFF) module to enable adaptive handling of diverse degradation types. Furthermore, to mitigate detail loss caused by the high compression and iterative sampling of LDMs, we design a Detail-Aware Expert Module (DAEM) in the decoder to enhance texture and fine-structure recovery. Extensive experiments across multi-task and mixed degradation settings demonstrate that our method consistently achieves state-of-the-art performance, highlighting the practical potential of diffusion priors for unified image restoration. Our code will be released.
>
---
#### [new 045] MamV2XCalib: V2X-based Target-less Infrastructure Camera Calibration with State Space Model
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的车路协同（V2X）感知任务，旨在解决基础设施摄像头的大规模精确标定问题。现有方法依赖人工或特定标定物，成本高且效率低。论文提出MamV2XCalib方法，利用车载LiDAR与路侧摄像头协同标定，无需标定物或人工干预。通过多尺度特征、4D相关体与Mamba模型，提升标定鲁棒性与精度，适用于复杂V2X场景。**

- **链接: [http://arxiv.org/pdf/2507.23595v1](http://arxiv.org/pdf/2507.23595v1)**

> **作者:** Yaoye Zhu; Zhe Wang; Yan Wang
>
> **备注:** ICCV25 poster
>
> **摘要:** As cooperative systems that leverage roadside cameras to assist autonomous vehicle perception become increasingly widespread, large-scale precise calibration of infrastructure cameras has become a critical issue. Traditional manual calibration methods are often time-consuming, labor-intensive, and may require road closures. This paper proposes MamV2XCalib, the first V2X-based infrastructure camera calibration method with the assistance of vehicle-side LiDAR. MamV2XCalib only requires autonomous vehicles equipped with LiDAR to drive near the cameras to be calibrated in the infrastructure, without the need for specific reference objects or manual intervention. We also introduce a new targetless LiDAR-camera calibration method, which combines multi-scale features and a 4D correlation volume to estimate the correlation between vehicle-side point clouds and roadside images. We model the temporal information and estimate the rotation angles with Mamba, effectively addressing calibration failures in V2X scenarios caused by defects in the vehicle-side data (such as occlusions) and large differences in viewpoint. We evaluate MamV2XCalib on the V2X-Seq and TUMTraf-V2X real-world datasets, demonstrating the effectiveness and robustness of our V2X-based automatic calibration approach. Compared to previous LiDAR-camera methods designed for calibration on one car, our approach achieves better and more stable calibration performance in V2X scenarios with fewer parameters. The code is available at https://github.com/zhuyaoye/MamV2XCalib.
>
---
#### [new 046] 3D-R1: Enhancing Reasoning in 3D VLMs for Unified Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于3D视觉-语言模型（3D VLM）任务，旨在提升模型在3D场景理解中的推理与泛化能力。针对现有模型因空间数据质量低和视角固定导致的局限性，论文提出3D-R1，构建高质量合成数据集Scene-30K，并引入强化学习与动态视角选择策略，显著提升了模型表现。**

- **链接: [http://arxiv.org/pdf/2507.23478v1](http://arxiv.org/pdf/2507.23478v1)**

> **作者:** Ting Huang; Zeyu Zhang; Hao Tang
>
> **摘要:** Large vision-language models (VLMs) have made significant strides in 2D visual understanding tasks, sparking interest in extending these capabilities to 3D scene understanding. However, current 3D VLMs often struggle with robust reasoning and generalization due to limitations in high-quality spatial data and the static nature of viewpoint assumptions. To address these challenges, we propose 3D-R1, a foundation model that enhances the reasoning capabilities of 3D VLMs. Specifically, we first construct a high-quality synthetic dataset with CoT, named Scene-30K, leveraging existing 3D-VL datasets and a data engine based on Gemini 2.5 Pro. It serves as cold-start initialization data for 3D-R1. Moreover, we leverage RLHF policy such as GRPO in the reinforcement learning training process to enhance reasoning capabilities and introduce three reward functions: a perception reward, a semantic similarity reward and a format reward to maintain detection accuracy and answer semantic precision. Furthermore, we introduce a dynamic view selection strategy that adaptively chooses the most informative perspectives for 3D scene understanding. Extensive experiments demonstrate that 3D-R1 delivers an average improvement of 10% across various 3D scene benchmarks, highlighting its effectiveness in enhancing reasoning and generalization in 3D scene understanding. Code: https://github.com/AIGeeksGroup/3D-R1. Website: https://aigeeksgroup.github.io/3D-R1.
>
---
#### [new 047] Details Matter for Indoor Open-vocabulary 3D Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于室内开放词汇3D实例分割（OV-3DIS）任务，旨在解决无需固定类别预定义、基于视觉语言模型的3D实例识别与分割问题。作者提出了一种新方法，结合3D跟踪生成提案、Alpha-CLIP分类及SMS评分优化，提升了检测精度与性能，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2507.23134v1](http://arxiv.org/pdf/2507.23134v1)**

> **作者:** Sanghun Jung; Jingjing Zheng; Ke Zhang; Nan Qiao; Albert Y. C. Chen; Lu Xia; Chi Liu; Yuyin Sun; Xiao Zeng; Hsiang-Wei Huang; Byron Boots; Min Sun; Cheng-Hao Kuo
>
> **备注:** ICCV 2025
>
> **摘要:** Unlike closed-vocabulary 3D instance segmentation that is often trained end-to-end, open-vocabulary 3D instance segmentation (OV-3DIS) often leverages vision-language models (VLMs) to generate 3D instance proposals and classify them. While various concepts have been proposed from existing research, we observe that these individual concepts are not mutually exclusive but complementary. In this paper, we propose a new state-of-the-art solution for OV-3DIS by carefully designing a recipe to combine the concepts together and refining them to address key challenges. Our solution follows the two-stage scheme: 3D proposal generation and instance classification. We employ robust 3D tracking-based proposal aggregation to generate 3D proposals and remove overlapped or partial proposals by iterative merging/removal. For the classification stage, we replace the standard CLIP model with Alpha-CLIP, which incorporates object masks as an alpha channel to reduce background noise and obtain object-centric representation. Additionally, we introduce the standardized maximum similarity (SMS) score to normalize text-to-proposal similarity, effectively filtering out false positives and boosting precision. Our framework achieves state-of-the-art performance on ScanNet200 and S3DIS across all AP and AR metrics, even surpassing an end-to-end closed-vocabulary method.
>
---
#### [new 048] Explainable Image Classification with Reduced Overconfidence for Tissue Characterisation
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决模型预测与像素归因的过度自信问题。作者提出一种结合风险估计的像素归因方法，通过生成像素级归因分布与风险评估，提升分类解释性。实验验证了方法在pCLE数据和ImageNet上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.23709v1](http://arxiv.org/pdf/2507.23709v1)**

> **作者:** Alfie Roddan; Chi Xu; Serine Ajlouni; Irini Kakaletri; Patra Charalampaki; Stamatia Giannarou
>
> **摘要:** The deployment of Machine Learning models intraoperatively for tissue characterisation can assist decision making and guide safe tumour resections. For image classification models, pixel attribution methods are popular to infer explainability. However, overconfidence in deep learning model's predictions translates to overconfidence in pixel attribution. In this paper, we propose the first approach which incorporates risk estimation into a pixel attribution method for improved image classification explainability. The proposed method iteratively applies a classification model with a pixel attribution method to create a volume of PA maps. This volume is used for the first time, to generate a pixel-wise distribution of PA values. We introduce a method to generate an enhanced PA map by estimating the expectation values of the pixel-wise distributions. In addition, the coefficient of variation (CV) is used to estimate pixel-wise risk of this enhanced PA map. Hence, the proposed method not only provides an improved PA map but also produces an estimation of risk on the output PA values. Performance evaluation on probe-based Confocal Laser Endomicroscopy (pCLE) data and ImageNet verifies that our improved explainability method outperforms the state-of-the-art.
>
---
#### [new 049] SAMSA: Segment Anything Model Enhanced with Spectral Angles for Hyperspectral Interactive Medical Image Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在解决高光谱成像（HSI）在数据限制和硬件差异下的分割难题。论文提出了SAMSA方法，结合RGB基础模型与光谱分析，通过用户点击引导分割和光谱相似性计算，实现跨光谱特性的高效融合，提升了分割精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.23673v1](http://arxiv.org/pdf/2507.23673v1)**

> **作者:** Alfie Roddan; Tobias Czempiel; Chi Xu; Daniel S. Elson; Stamatia Giannarou
>
> **摘要:** Hyperspectral imaging (HSI) provides rich spectral information for medical imaging, yet encounters significant challenges due to data limitations and hardware variations. We introduce SAMSA, a novel interactive segmentation framework that combines an RGB foundation model with spectral analysis. SAMSA efficiently utilizes user clicks to guide both RGB segmentation and spectral similarity computations. The method addresses key limitations in HSI segmentation through a unique spectral feature fusion strategy that operates independently of spectral band count and resolution. Performance evaluation on publicly available datasets has shown 81.0% 1-click and 93.4% 5-click DICE on a neurosurgical and 81.1% 1-click and 89.2% 5-click DICE on an intraoperative porcine hyperspectral dataset. Experimental results demonstrate SAMSA's effectiveness in few-shot and zero-shot learning scenarios and using minimal training examples. Our approach enables seamless integration of datasets with different spectral characteristics, providing a flexible framework for hyperspectral medical image analysis.
>
---
#### [new 050] X-NeMo: Expressive Neural Motion Reenactment via Disentangled Latent Attention
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决跨个体的面部动作迁移问题。现有方法存在身份泄露和难以捕捉细微表情的问题。论文提出X-NeMo，通过解耦潜在注意力机制，实现从驱动视频中提取1D身份无关的运动描述，并结合GAN监督与增强策略，提升动画表现力与身份保留能力。**

- **链接: [http://arxiv.org/pdf/2507.23143v1](http://arxiv.org/pdf/2507.23143v1)**

> **作者:** Xiaochen Zhao; Hongyi Xu; Guoxian Song; You Xie; Chenxu Zhang; Xiu Li; Linjie Luo; Jinli Suo; Yebin Liu
>
> **备注:** ICLR 2025, code is available at https://github.com/bytedance/x-nemo-inference
>
> **摘要:** We propose X-NeMo, a novel zero-shot diffusion-based portrait animation pipeline that animates a static portrait using facial movements from a driving video of a different individual. Our work first identifies the root causes of the key issues in prior approaches, such as identity leakage and difficulty in capturing subtle and extreme expressions. To address these challenges, we introduce a fully end-to-end training framework that distills a 1D identity-agnostic latent motion descriptor from driving image, effectively controlling motion through cross-attention during image generation. Our implicit motion descriptor captures expressive facial motion in fine detail, learned end-to-end from a diverse video dataset without reliance on pretrained motion detectors. We further enhance expressiveness and disentangle motion latents from identity cues by supervising their learning with a dual GAN decoder, alongside spatial and color augmentations. By embedding the driving motion into a 1D latent vector and controlling motion via cross-attention rather than additive spatial guidance, our design eliminates the transmission of spatial-aligned structural clues from the driving condition to the diffusion backbone, substantially mitigating identity leakage. Extensive experiments demonstrate that X-NeMo surpasses state-of-the-art baselines, producing highly expressive animations with superior identity resemblance. Our code and models are available for research.
>
---
#### [new 051] iLRM: An Iterative Large 3D Reconstruction Model
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决现有方法在多视角图像处理中计算成本高、扩展性差的问题。作者提出iLRM模型，通过解耦场景表示与输入视图、采用两阶段注意力机制、逐层注入高分辨率信息，实现高效、高质量的3D高斯表示重建。**

- **链接: [http://arxiv.org/pdf/2507.23277v1](http://arxiv.org/pdf/2507.23277v1)**

> **作者:** Gyeongjin Kang; Seungtae Nam; Xiangyu Sun; Sameh Khamis; Abdelrahman Mohamed; Eunbyung Park
>
> **备注:** Project page: https://gynjn.github.io/iLRM/
>
> **摘要:** Feed-forward 3D modeling has emerged as a promising approach for rapid and high-quality 3D reconstruction. In particular, directly generating explicit 3D representations, such as 3D Gaussian splatting, has attracted significant attention due to its fast and high-quality rendering, as well as numerous applications. However, many state-of-the-art methods, primarily based on transformer architectures, suffer from severe scalability issues because they rely on full attention across image tokens from multiple input views, resulting in prohibitive computational costs as the number of views or image resolution increases. Toward a scalable and efficient feed-forward 3D reconstruction, we introduce an iterative Large 3D Reconstruction Model (iLRM) that generates 3D Gaussian representations through an iterative refinement mechanism, guided by three core principles: (1) decoupling the scene representation from input-view images to enable compact 3D representations; (2) decomposing fully-attentional multi-view interactions into a two-stage attention scheme to reduce computational costs; and (3) injecting high-resolution information at every layer to achieve high-fidelity reconstruction. Experimental results on widely used datasets, such as RE10K and DL3DV, demonstrate that iLRM outperforms existing methods in both reconstruction quality and speed. Notably, iLRM exhibits superior scalability, delivering significantly higher reconstruction quality under comparable computational cost by efficiently leveraging a larger number of input views.
>
---
#### [new 052] Enhanced Velocity Field Modeling for Gaussian Video Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D视频重建任务，旨在解决复杂运动和尺度变化下高斯变形场建模的过拟合与动态内容缺失问题。论文提出FlowGaussian-VR，包含基于光流的速度场优化和动态区域自适应稠密化策略，提升了视频重建质量与视觉效果。**

- **链接: [http://arxiv.org/pdf/2507.23704v1](http://arxiv.org/pdf/2507.23704v1)**

> **作者:** Zhenyang Li; Xiaoyang Bai; Tongchen Zhang; Pengfei Shen; Weiwei Xu; Yifan Peng
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** High-fidelity 3D video reconstruction is essential for enabling real-time rendering of dynamic scenes with realistic motion in virtual and augmented reality (VR/AR). The deformation field paradigm of 3D Gaussian splatting has achieved near-photorealistic results in video reconstruction due to the great representation capability of deep deformation networks. However, in videos with complex motion and significant scale variations, deformation networks often overfit to irregular Gaussian trajectories, leading to suboptimal visual quality. Moreover, the gradient-based densification strategy designed for static scene reconstruction proves inadequate to address the absence of dynamic content. In light of these challenges, we propose a flow-empowered velocity field modeling scheme tailored for Gaussian video reconstruction, dubbed FlowGaussian-VR. It consists of two core components: a velocity field rendering (VFR) pipeline which enables optical flow-based optimization, and a flow-assisted adaptive densification (FAD) strategy that adjusts the number and size of Gaussians in dynamic regions. We validate our model's effectiveness on multi-view dynamic reconstruction and novel view synthesis with multiple real-world datasets containing challenging motion scenarios, demonstrating not only notable visual improvements (over 2.5 dB gain in PSNR) and less blurry artifacts in dynamic textures, but also regularized and trackable per-Gaussian trajectories.
>
---
#### [new 053] CNN-based solution for mango classification in agricultural environments
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于农业环境中的水果分类任务，旨在解决芒果自动分级问题。作者设计了一种基于CNN的系统，采用ResNet-18进行分类，结合级联检测器实现检测，最终通过MatLab界面展示结果，提升农业质量管控效率。**

- **链接: [http://arxiv.org/pdf/2507.23174v1](http://arxiv.org/pdf/2507.23174v1)**

> **作者:** Beatriz Díaz Peón; Jorge Torres Gómez; Ariel Fajardo Márquez
>
> **摘要:** This article exemplifies the design of a fruit detection and classification system using Convolutional Neural Networks (CNN). The goal is to develop a system that automatically assesses fruit quality for farm inventory management. Specifically, a method for mango fruit classification was developed using image processing, ensuring both accuracy and efficiency. Resnet-18 was selected as the preliminary architecture for classification, while a cascade detector was used for detection, balancing execution speed and computational resource consumption. Detection and classification results were displayed through a graphical interface developed in MatLab App Designer, streamlining system interaction. The integration of convolutional neural networks and cascade detectors proffers a reliable solution for fruit classification and detection, with potential applications in agricultural quality control.
>
---
#### [new 054] Training-free Geometric Image Editing on Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决复杂几何变换下保持场景连贯性的问题。作者提出了一种无需训练的扩散模型FreeFine，通过解耦物体变换、源区域修复与目标区域优化的流程，提升了编辑精度与图像质量，尤其适用于复杂变换场景。**

- **链接: [http://arxiv.org/pdf/2507.23300v1](http://arxiv.org/pdf/2507.23300v1)**

> **作者:** Hanshen Zhu; Zhen Zhu; Kaile Zhang; Yiming Gong; Yuliang Liu; Xiang Bai
>
> **备注:** 24 pages, 22 figures, ICCV
>
> **摘要:** We tackle the task of geometric image editing, where an object within an image is repositioned, reoriented, or reshaped while preserving overall scene coherence. Previous diffusion-based editing methods often attempt to handle all relevant subtasks in a single step, proving difficult when transformations become large or structurally complex. We address this by proposing a decoupled pipeline that separates object transformation, source region inpainting, and target region refinement. Both inpainting and refinement are implemented using a training-free diffusion approach, FreeFine. In experiments on our new GeoBench benchmark, which contains both 2D and 3D editing scenarios, FreeFine outperforms state-of-the-art alternatives in image fidelity, and edit precision, especially under demanding transformations. Code and benchmark are available at: https://github.com/CIawevy/FreeFine
>
---
#### [new 055] Phi-Ground Tech Report: Advancing Perception in GUI Grounding
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于多模态人机交互任务，旨在提升计算机使用代理（CUAs）在图形用户界面（GUI）操作中的感知与执行能力。论文聚焦解决GUI基础操作（如点击、输入）的准确性问题，提出了Phi-Ground模型家族，在多个基准测试中达到最优性能，尤其在ScreenSpot-pro和UI-Vision上表现突出。**

- **链接: [http://arxiv.org/pdf/2507.23779v1](http://arxiv.org/pdf/2507.23779v1)**

> **作者:** Miaosen Zhang; Ziqiang Xu; Jialiang Zhu; Qi Dai; Kai Qiu; Yifan Yang; Chong Luo; Tianyi Chen; Justin Wagle; Tim Franklin; Baining Guo
>
> **摘要:** With the development of multimodal reasoning models, Computer Use Agents (CUAs), akin to Jarvis from \textit{"Iron Man"}, are becoming a reality. GUI grounding is a core component for CUAs to execute actual actions, similar to mechanical control in robotics, and it directly leads to the success or failure of the system. It determines actions such as clicking and typing, as well as related parameters like the coordinates for clicks. Current end-to-end grounding models still achieve less than 65\% accuracy on challenging benchmarks like ScreenSpot-pro and UI-Vision, indicating they are far from being ready for deployment. % , as a single misclick can result in unacceptable consequences. In this work, we conduct an empirical study on the training of grounding models, examining details from data collection to model training. Ultimately, we developed the \textbf{Phi-Ground} model family, which achieves state-of-the-art performance across all five grounding benchmarks for models under $10B$ parameters in agent settings. In the end-to-end model setting, our model still achieves SOTA results with scores of \textit{\textbf{43.2}} on ScreenSpot-pro and \textit{\textbf{27.2}} on UI-Vision. We believe that the various details discussed in this paper, along with our successes and failures, not only clarify the construction of grounding models but also benefit other perception tasks. Project homepage: \href{https://zhangmiaosen2000.github.io/Phi-Ground/}{https://zhangmiaosen2000.github.io/Phi-Ground/}
>
---
#### [new 056] Automated Mapping the Pathways of Cranial Nerve II, III, V, and VII/VIII: A Multi-Parametric Multi-Stage Diffusion Tractography Atlas
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决颅神经路径自动映射的问题。通过多参数多阶段扩散纤维束成像技术，构建了首个全面的颅神经图谱，可自动识别8条与5对颅神经相关的纤维束，提升了术前对神经与周围组织关系的理解。**

- **链接: [http://arxiv.org/pdf/2507.23245v1](http://arxiv.org/pdf/2507.23245v1)**

> **作者:** Lei Xie; Jiahao Huang; Jiawei Zhang; Jianzhong He; Yiang Pan; Guoqiang Xie; Mengjun Li; Qingrun Zeng; Mingchu Li; Yuanjing Feng
>
> **摘要:** Cranial nerves (CNs) play a crucial role in various essential functions of the human brain, and mapping their pathways from diffusion MRI (dMRI) provides valuable preoperative insights into the spatial relationships between individual CNs and key tissues. However, mapping a comprehensive and detailed CN atlas is challenging because of the unique anatomical structures of each CN pair and the complexity of the skull base environment.In this work, we present what we believe to be the first study to develop a comprehensive diffusion tractography atlas for automated mapping of CN pathways in the human brain. The CN atlas is generated by fiber clustering by using the streamlines generated by multi-parametric fiber tractography for each pair of CNs. Instead of disposable clustering, we explore a new strategy of multi-stage fiber clustering for multiple analysis of approximately 1,000,000 streamlines generated from the 50 subjects from the Human Connectome Project (HCP). Quantitative and visual experiments demonstrate that our CN atlas achieves high spatial correspondence with expert manual annotations on multiple acquisition sites, including the HCP dataset, the Multi-shell Diffusion MRI (MDM) dataset and two clinical cases of pituitary adenoma patients. The proposed CN atlas can automatically identify 8 fiber bundles associated with 5 pairs of CNs, including the optic nerve CN II, oculomotor nerve CN III, trigeminal nerve CN V and facial-vestibulocochlear nerve CN VII/VIII, and its robustness is demonstrated experimentally. This work contributes to the field of diffusion imaging by facilitating more efficient and automated mapping the pathways of multiple pairs of CNs, thereby enhancing the analysis and understanding of complex brain structures through visualization of their spatial relationships with nearby anatomy.
>
---
#### [new 057] Half-Physics: Enabling Kinematic 3D Human Model with Physical Interactions
- **分类: cs.CV**

- **简介: 该论文属于3D人体建模与物理交互任务，旨在解决现有模型（如SMPL-X）无法与环境进行物理交互的问题。论文提出“Half-Physics”方法，将3D运动转换为物理仿真，保留原有姿态控制的同时实现真实物理交互，避免穿透和不真实动力学表现，且无需训练，适用于任意人体形态和动作，实时运行。**

- **链接: [http://arxiv.org/pdf/2507.23778v1](http://arxiv.org/pdf/2507.23778v1)**

> **作者:** Li Siyao; Yao Feng; Omid Tehari; Chen Change Loy; Michael J. Black
>
> **摘要:** While current general-purpose 3D human models (e.g., SMPL-X) efficiently represent accurate human shape and pose, they lacks the ability to physically interact with the environment due to the kinematic nature. As a result, kinematic-based interaction models often suffer from issues such as interpenetration and unrealistic object dynamics. To address this limitation, we introduce a novel approach that embeds SMPL-X into a tangible entity capable of dynamic physical interactions with its surroundings. Specifically, we propose a "half-physics" mechanism that transforms 3D kinematic motion into a physics simulation. Our approach maintains kinematic control over inherent SMPL-X poses while ensuring physically plausible interactions with scenes and objects, effectively eliminating penetration and unrealistic object dynamics. Unlike reinforcement learning-based methods, which demand extensive and complex training, our half-physics method is learning-free and generalizes to any body shape and motion; meanwhile, it operates in real time. Moreover, it preserves the fidelity of the original kinematic motion while seamlessly integrating physical interactions
>
---
#### [new 058] FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的道路场景理解任务，旨在提升车道线拓扑结构的检测与推理性能。针对现有方法在时间信息利用、姿态估计鲁棒性等方面的不足，论文提出FASTopoWM框架，融合快慢系统与潜在世界模型，增强时空一致性与鲁棒性，取得了更好的检测与中心线感知效果。**

- **链接: [http://arxiv.org/pdf/2507.23325v1](http://arxiv.org/pdf/2507.23325v1)**

> **作者:** Yiming Yang; Hongbin Lin; Yueru Luo; Suzhong Fu; Chao Zheng; Xinrui Yan; Shuqi Mei; Kun Tang; Shuguang Cui; Zhen Li
>
> **摘要:** Lane segment topology reasoning provides comprehensive bird's-eye view (BEV) road scene understanding, which can serve as a key perception module in planning-oriented end-to-end autonomous driving systems. Existing lane topology reasoning methods often fall short in effectively leveraging temporal information to enhance detection and reasoning performance. Recently, stream-based temporal propagation method has demonstrated promising results by incorporating temporal cues at both the query and BEV levels. However, it remains limited by over-reliance on historical queries, vulnerability to pose estimation failures, and insufficient temporal propagation. To overcome these limitations, we propose FASTopoWM, a novel fast-slow lane segment topology reasoning framework augmented with latent world models. To reduce the impact of pose estimation failures, this unified framework enables parallel supervision of both historical and newly initialized queries, facilitating mutual reinforcement between the fast and slow systems. Furthermore, we introduce latent query and BEV world models conditioned on the action latent to propagate the state representations from past observations to the current timestep. This design substantially improves the performance of temporal perception within the slow pipeline. Extensive experiments on the OpenLane-V2 benchmark demonstrate that FASTopoWM outperforms state-of-the-art methods in both lane segment detection (37.4% v.s. 33.6% on mAP) and centerline perception (46.3% v.s. 41.5% on OLS).
>
---
#### [new 059] UniLiP: Adapting CLIP for Unified Multimodal Understanding, Generation and Editing
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决CLIP模型在生成和编辑任务中的局限性。作者提出了UniLiP，通过两阶段训练和自蒸馏策略，使CLIP具备图像重建能力，同时保持其理解性能。此外，采用双条件架构连接MLLM和扩散模型，提升了生成与编辑效果。**

- **链接: [http://arxiv.org/pdf/2507.23278v1](http://arxiv.org/pdf/2507.23278v1)**

> **作者:** Hao Tang; Chenwei Xie; Xiaoyi Bao; Tingyu Weng; Pandeng Li; Yun Zheng; Liwei Wang
>
> **摘要:** In this paper, we propose UniLIP, which extends CLIP to reconstruction, generation and editing, thereby building a unified tokenizer upon its exceptional comprehension capabilities. Previous CLIP-based unified methods often require additional diffusion decoders or quantization to support reconstruction and generation tasks, leading to inconsistent reconstruction or degradation of original comprehension performance.In contrast, we introduce a two-stage training scheme and a self-distillation strategy that progressively integrates reconstruction capabilities into CLIP, allowing it to maintain original comprehension performance while achieving effective image reconstruction. Furthermore, we propose a dual-condition architecture to connect the MLLM and diffusion transformer, using both learnable queries and the last layer multimodal hidden states as joint conditions. This method not only enables the utilization of the MLLM's strong reasoning capabilities in generation tasks, but also maximizes the exploitation of the rich information in UniLIP features during editing tasks. In text-to-image generation tasks, UniLIP obtains scores of 0.87 and 0.53 on GenEval and WISE benchmark respectively, surpassing all previous unified models of similar scale. In image editing, UniLIP also achieves a score of 3.62 on the ImgEdit Benchmark, surpassing recent state-of-the-art models such as BAGEL and UniWorld-V1. UniLIP effectively expand the application scope of CLIP, enabling continuous CLIP features to not only serve as the optimal choice for understanding tasks but also achieve highly competitive performance in generation and editing tasks.
>
---
#### [new 060] Slot Attention with Re-Initialization and Self-Distillation
- **分类: cs.CV**

- **简介: 该论文属于对象中心学习（OCL）任务，旨在解决视觉场景中对象表示的冗余与分割错误问题。作者提出DIAS方法，通过槽位重新初始化和自蒸馏优化注意力机制，提升了对象发现、识别及视觉推理性能。**

- **链接: [http://arxiv.org/pdf/2507.23755v1](http://arxiv.org/pdf/2507.23755v1)**

> **作者:** Rongzhen Zhao; Yi Zhao; Juho Kannala; Joni Pajarinen
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Unlike popular solutions based on dense feature maps, Object-Centric Learning (OCL) represents visual scenes as sub-symbolic object-level feature vectors, termed slots, which are highly versatile for tasks involving visual modalities. OCL typically aggregates object superpixels into slots by iteratively applying competitive cross attention, known as Slot Attention, with the slots as the query. However, once initialized, these slots are reused naively, causing redundant slots to compete with informative ones for representing objects. This often results in objects being erroneously segmented into parts. Additionally, mainstream methods derive supervision signals solely from decoding slots into the input's reconstruction, overlooking potential supervision based on internal information. To address these issues, we propose Slot Attention with re-Initialization and self-Distillation (DIAS): $\emph{i)}$ We reduce redundancy in the aggregated slots and re-initialize extra aggregation to update the remaining slots; $\emph{ii)}$ We drive the bad attention map at the first aggregation iteration to approximate the good at the last iteration to enable self-distillation. Experiments demonstrate that DIAS achieves state-of-the-art on OCL tasks like object discovery and recognition, while also improving advanced visual prediction and reasoning. Our code is available on https://github.com/Genera1Z/DIAS.
>
---
#### [new 061] The Impact of Image Resolution on Face Detection: A Comparative Analysis of MTCNN, YOLOv XI and YOLOv XII models
- **分类: cs.CV; 68T45, 68T07; I.4.8; I.4.9; I.5.4**

- **简介: 该论文属于计算机视觉任务，旨在解决低分辨率图像中人脸检测性能下降的问题。作者评估了MTCNN、YOLOv11和YOLOv12三种模型在不同分辨率下的检测精度、召回率及推理速度，分析其在WIDER FACE数据集上的表现，为不同场景选择合适模型提供依据。**

- **链接: [http://arxiv.org/pdf/2507.23341v1](http://arxiv.org/pdf/2507.23341v1)**

> **作者:** Ahmet Can Ömercikoğlu; Mustafa Mansur Yönügül; Pakize Erdoğmuş
>
> **备注:** 6 pages, 5 figures, 4 tables
>
> **摘要:** Face detection is a crucial component in many AI-driven applications such as surveillance, biometric authentication, and human-computer interaction. However, real-world conditions like low-resolution imagery present significant challenges that degrade detection performance. In this study, we systematically investigate the impact of input resolution on the accuracy and robustness of three prominent deep learning-based face detectors: YOLOv11, YOLOv12, and MTCNN. Using the WIDER FACE dataset, we conduct extensive evaluations across multiple image resolutions (160x160, 320x320, and 640x640) and assess each model's performance using metrics such as precision, recall, mAP50, mAP50-95, and inference time. Results indicate that YOLOv11 outperforms YOLOv12 and MTCNN in terms of detection accuracy, especially at higher resolutions, while YOLOv12 exhibits slightly better recall. MTCNN, although competitive in landmark localization, lags in real-time inference speed. Our findings provide actionable insights for selecting resolution-aware face detection models suitable for varying operational constraints.
>
---
#### [new 062] Hyperbolic Cycle Alignment for Infrared-Visible Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于图像融合任务，旨在解决跨模态图像（如红外与可见光图像）因模态差异导致的配准不准问题。作者提出了一种基于双曲空间的图像配准方法Hy-CycleAlign，通过在非欧空间中构建前向与反向注册网络，形成闭环结构，提升对跨模态图像的对齐和融合效果。**

- **链接: [http://arxiv.org/pdf/2507.23508v1](http://arxiv.org/pdf/2507.23508v1)**

> **作者:** Timing Li; Bing Cao; Jiahe Feng; Haifang Cao; Qinghau Hu; Pengfei Zhu
>
> **摘要:** Image fusion synthesizes complementary information from multiple sources, mitigating the inherent limitations of unimodal imaging systems. Accurate image registration is essential for effective multi-source data fusion. However, existing registration methods, often based on image translation in Euclidean space, fail to handle cross-modal misalignment effectively, resulting in suboptimal alignment and fusion quality. To overcome this limitation, we explore image alignment in non-Euclidean space and propose a Hyperbolic Cycle Alignment Network (Hy-CycleAlign). To the best of our knowledge, Hy-CycleAlign is the first image registration method based on hyperbolic space. It introduces a dual-path cross-modal cyclic registration framework, in which a forward registration network aligns cross-modal inputs, while a backward registration network reconstructs the original image, forming a closed-loop registration structure with geometric consistency. Additionally, we design a Hyperbolic Hierarchy Contrastive Alignment (H$^{2}$CA) module, which maps images into hyperbolic space and imposes registration constraints, effectively reducing interference caused by modality discrepancies. We further analyze image registration in both Euclidean and hyperbolic spaces, demonstrating that hyperbolic space enables more sensitive and effective multi-modal image registration. Extensive experiments on misaligned multi-modal images demonstrate that our method significantly outperforms existing approaches in both image alignment and fusion. Our code will be publicly available.
>
---
#### [new 063] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决视觉语言动作（VLA）模型因视觉token过多导致的计算效率低下问题。作者提出FastDriveVLA框架，通过基于重建的视觉token剪枝方法ReconPruner，优先保留前景信息，提升推理效率，并在nuScenes数据集上验证有效性。**

- **链接: [http://arxiv.org/pdf/2507.23318v1](http://arxiv.org/pdf/2507.23318v1)**

> **作者:** Jiajun Cao; Qizhe Zhang; Peidong Jia; Xuhui Zhao; Bo Lan; Xiaoan Zhang; Xiaobao Wei; Sixiang Chen; Zhuo Li; Yang Wang; Liyun Li; Xianming Liu; Ming Lu; Shanghang Zhang
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated significant potential in complex scene understanding and action reasoning, leading to their increasing adoption in end-to-end autonomous driving systems. However, the long visual tokens of VLA models greatly increase computational costs. Current visual token pruning methods in Vision-Language Models (VLM) rely on either visual token similarity or visual-text attention, but both have shown poor performance in autonomous driving scenarios. Given that human drivers concentrate on relevant foreground areas while driving, we assert that retaining visual tokens containing this foreground information is essential for effective decision-making. Inspired by this, we propose FastDriveVLA, a novel reconstruction-based vision token pruning framework designed specifically for autonomous driving. FastDriveVLA includes a plug-and-play visual token pruner called ReconPruner, which prioritizes foreground information through MAE-style pixel reconstruction. A novel adversarial foreground-background reconstruction strategy is designed to train ReconPruner for the visual encoder of VLA models. Once trained, ReconPruner can be seamlessly applied to different VLA models with the same visual encoder without retraining. To train ReconPruner, we also introduce a large-scale dataset called nuScenes-FG, consisting of 241K image-mask pairs with annotated foreground regions. Our approach achieves state-of-the-art results on the nuScenes closed-loop planning benchmark across different pruning ratios.
>
---
#### [new 064] SeqAffordSplat: Scene-level Sequential Affordance Reasoning on 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决现有方法在复杂、多对象场景中长期交互推理能力不足的问题。作者提出了SeqAffordSplat基准和SeqSplatNet框架，通过语言模型生成指令与分割标记，并融合几何与语义信息，实现从单步交互到复杂序列任务的提升。**

- **链接: [http://arxiv.org/pdf/2507.23772v1](http://arxiv.org/pdf/2507.23772v1)**

> **作者:** Di Li; Jie Feng; Jiahao Chen; Weisheng Dong; Guanbin Li; Yuhui Zheng; Mingtao Feng; Guangming Shi
>
> **摘要:** 3D affordance reasoning, the task of associating human instructions with the functional regions of 3D objects, is a critical capability for embodied agents. Current methods based on 3D Gaussian Splatting (3DGS) are fundamentally limited to single-object, single-step interactions, a paradigm that falls short of addressing the long-horizon, multi-object tasks required for complex real-world applications. To bridge this gap, we introduce the novel task of Sequential 3D Gaussian Affordance Reasoning and establish SeqAffordSplat, a large-scale benchmark featuring 1800+ scenes to support research on long-horizon affordance understanding in complex 3DGS environments. We then propose SeqSplatNet, an end-to-end framework that directly maps an instruction to a sequence of 3D affordance masks. SeqSplatNet employs a large language model that autoregressively generates text interleaved with special segmentation tokens, guiding a conditional decoder to produce the corresponding 3D mask. To handle complex scene geometry, we introduce a pre-training strategy, Conditional Geometric Reconstruction, where the model learns to reconstruct complete affordance region masks from known geometric observations, thereby building a robust geometric prior. Furthermore, to resolve semantic ambiguities, we design a feature injection mechanism that lifts rich semantic features from 2D Vision Foundation Models (VFM) and fuses them into the 3D decoder at multiple scales. Extensive experiments demonstrate that our method sets a new state-of-the-art on our challenging benchmark, effectively advancing affordance reasoning from single-step interactions to complex, sequential tasks at the scene level.
>
---
#### [new 065] Contrastive Learning-Driven Traffic Sign Perception: Multi-Modal Fusion of Text and Vision
- **分类: cs.CV**

- **简介: 该论文属于交通标志识别任务，旨在解决数据长尾分布和小目标多尺度特征提取问题。论文提出了一种结合开放词汇检测与跨模态学习的两阶段框架，包含NanoVerse YOLO检测模型与TSR-MCL分类模型，提升了识别性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.23331v1](http://arxiv.org/pdf/2507.23331v1)**

> **作者:** Qiang Lu; Waikit Xiu; Xiying Li; Shenyu Hu; Shengbo Sun
>
> **备注:** 11pages, 5 figures
>
> **摘要:** Traffic sign recognition, as a core component of autonomous driving perception systems, directly influences vehicle environmental awareness and driving safety. Current technologies face two significant challenges: first, the traffic sign dataset exhibits a pronounced long-tail distribution, resulting in a substantial decline in recognition performance of traditional convolutional networks when processing low-frequency and out-of-distribution classes; second, traffic signs in real-world scenarios are predominantly small targets with significant scale variations, making it difficult to extract multi-scale features.To overcome these issues, we propose a novel two-stage framework combining open-vocabulary detection and cross-modal learning. For traffic sign detection, our NanoVerse YOLO model integrates a reparameterizable vision-language path aggregation network (RepVL-PAN) and an SPD-Conv module to specifically enhance feature extraction for small, multi-scale targets. For traffic sign classification, we designed a Traffic Sign Recognition Multimodal Contrastive Learning model (TSR-MCL). By contrasting visual features from a Vision Transformer with semantic features from a rule-based BERT, TSR-MCL learns robust, frequency-independent representations, effectively mitigating class confusion caused by data imbalance. On the TT100K dataset, our method achieves a state-of-the-art 78.4% mAP in the long-tail detection task for all-class recognition. The model also obtains 91.8% accuracy and 88.9% recall, significantly outperforming mainstream algorithms and demonstrating superior accuracy and generalization in complex, open-world scenarios.
>
---
#### [new 066] Efficient Masked Attention Transformer for Few-Shot Classification and Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于少样本分类与分割（FS-CS）任务，旨在用少量标注样本同时完成多标签分类与多类分割。现有方法在小物体上表现不佳，为此论文提出EMAT模型，包含高效注意力机制、可学习下采样与参数效率优化，显著提升小物体精度，且参数更少。此外，还提出更贴近实际的新评估设置。**

- **链接: [http://arxiv.org/pdf/2507.23642v1](http://arxiv.org/pdf/2507.23642v1)**

> **作者:** Dustin Carrión-Ojeda; Stefan Roth; Simone Schaub-Meyer
>
> **备注:** Accepted for GCPR 2025. Project page: https://visinf.github.io/emat
>
> **摘要:** Few-shot classification and segmentation (FS-CS) focuses on jointly performing multi-label classification and multi-class segmentation using few annotated examples. Although the current state of the art (SOTA) achieves high accuracy in both tasks, it struggles with small objects. To overcome this, we propose the Efficient Masked Attention Transformer (EMAT), which improves classification and segmentation accuracy, especially for small objects. EMAT introduces three modifications: a novel memory-efficient masked attention mechanism, a learnable downscaling strategy, and parameter-efficiency enhancements. EMAT outperforms all FS-CS methods on the PASCAL-5$^i$ and COCO-20$^i$ datasets, using at least four times fewer trainable parameters. Moreover, as the current FS-CS evaluation setting discards available annotations, despite their costly collection, we introduce two novel evaluation settings that consider these annotations to better reflect practical scenarios.
>
---
#### [new 067] AGA: An adaptive group alignment framework for structured medical cross-modal representation learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学跨模态表示学习任务，旨在解决现有方法忽略临床报告结构、依赖大量负样本的问题。作者提出AGA框架，通过自适应分组机制和阈值门控模块，实现图像与文本的细粒度对齐，并引入实例感知的对齐损失，提升跨模态检索与分类性能。**

- **链接: [http://arxiv.org/pdf/2507.23402v1](http://arxiv.org/pdf/2507.23402v1)**

> **作者:** Wei Li; Xun Gong; Jiao Li; Xiaobin Sun
>
> **摘要:** Learning medical visual representations from paired images and reports is a promising direction in representation learning. However, current vision-language pretraining methods in the medical domain often simplify clinical reports into single entities or fragmented tokens, ignoring their inherent structure. In addition, contrastive learning frameworks typically depend on large quantities of hard negative samples, which is impractical for small-scale medical datasets. To tackle these challenges, we propose Adaptive Grouped Alignment (AGA), a new framework that captures structured semantics from paired medical images and reports. AGA introduces a bidirectional grouping mechanism based on a sparse similarity matrix. For each image-report pair, we compute fine-grained similarities between text tokens and image patches. Each token selects its top-matching patches to form a visual group, and each patch selects its most related tokens to form a language group. To enable adaptive grouping, we design two threshold gating modules, called Language Grouped Threshold Gate and Vision Grouped Threshold Gate, which learn grouping thresholds dynamically. Group representations are computed as weighted averages based on similarity scores. To align each token with its group representation, we introduce an Instance Aware Group Alignment loss that operates within each image-text pair, removing the need for external negatives. Finally, a Bidirectional Cross-modal Grouped Alignment module is applied to enhance fine-grained alignment between visual and linguistic group representations. Extensive experiments on public and private datasets show that our method achieves strong performance on image-text retrieval and classification tasks under both fine-tuning and zero-shot settings.
>
---
#### [new 068] Out-of-Distribution Detection in Medical Imaging via Diffusion Trajectories
- **分类: cs.CV**

- **简介: 该论文属于医学影像中的无监督异常检测任务，旨在解决罕见病灶识别中数据不平衡和计算成本高的问题。作者提出一种基于扩散轨迹的重构无关方法，利用预训练Stein得分扩散模型，通过估计轨迹曲率实现快速准确的异常评分，显著提升检测性能并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2507.23411v1](http://arxiv.org/pdf/2507.23411v1)**

> **作者:** Lemar Abdi; Francisco Caetano; Amaan Valiuddin; Christiaan Viviers; Hamdi Joudeh; Fons van der Sommen
>
> **备注:** Accepted at Uncertainty for Safe Utilization of Machine Learning in Medical Imaging, MICCAI 2025
>
> **摘要:** In medical imaging, unsupervised out-of-distribution (OOD) detection offers an attractive approach for identifying pathological cases with extremely low incidence rates. In contrast to supervised methods, OOD-based approaches function without labels and are inherently robust to data imbalances. Current generative approaches often rely on likelihood estimation or reconstruction error, but these methods can be computationally expensive, unreliable, and require retraining if the inlier data changes. These limitations hinder their ability to distinguish nominal from anomalous inputs efficiently, consistently, and robustly. We propose a reconstruction-free OOD detection method that leverages the forward diffusion trajectories of a Stein score-based denoising diffusion model (SBDDM). By capturing trajectory curvature via the estimated Stein score, our approach enables accurate anomaly scoring with only five diffusion steps. A single SBDDM pre-trained on a large, semantically aligned medical dataset generalizes effectively across multiple Near-OOD and Far-OOD benchmarks, achieving state-of-the-art performance while drastically reducing computational cost during inference. Compared to existing methods, SBDDM achieves a relative improvement of up to 10.43% and 18.10% for Near-OOD and Far-OOD detection, making it a practical building block for real-time, reliable computer-aided diagnosis.
>
---
#### [new 069] Impact of Hyperparameter Optimization on the Accuracy of Lightweight Deep Learning Models for Real-Time Image Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像分类任务，旨在研究超参数优化对轻量级深度学习模型在实时应用中的准确性和部署效率的影响。论文通过系统实验分析了多种高效模型在不同超参数设置下的表现，并评估其在准确率、推理速度等方面的平衡，为实时图像分类提供优化指导。**

- **链接: [http://arxiv.org/pdf/2507.23315v1](http://arxiv.org/pdf/2507.23315v1)**

> **作者:** Vineet Kumar Rakesh; Soumya Mazumdar; Tapas Samanta; Sarbajit Pal; Amitabha Das
>
> **备注:** 13 pages, 4 figures, 4 tables. Includes ablation study and evaluation on 7 lightweight deep learning models. Code and logs available at https://github.com/VineetKumarRakesh/lcnn-opt
>
> **摘要:** Lightweight convolutional and transformer-based models have become vital for real-time image classification in resource-constrained applications, such as embedded systems and edge devices. This work analyzes the influence of hyperparameter adjustment on the accuracy and convergence behavior of seven efficient deep learning architectures: EfficientNetV2-S, ConvNeXt-T, MobileViT v2 (XXS/XS/S), MobileNetV3-L, TinyViT-21M, and RepVGG-A2. All models are trained on the ImageNet-1K dataset under consistent training settings, with an emphasis on real-time practicality. An comprehensive ablation study is undertaken to separate the effect of critical hyperparameters, including learning rate schedules, batch sizes, input resolution, data augmentation, regularization approaches, and optimizer choice. To assess appropriateness for real-time applications, each model is assessed not only in terms of Top-1 and Top-5 classification accuracy, but also in terms of inference time, parameter count, model size, and frames-per-second (FPS) on a GPU-accelerated edge deployment simulation. Results demonstrate that cosine learning rate decay and adjustable batch size may greatly boost both accuracy and convergence speed, while keeping low latency and memory cost. Notably, RepVGG-A2 achieves over 80% Top-1 accuracy with efficient inference performance, offering a compelling balance between accuracy and deployment cost for VGG-style models. The results give practical guidance for constructing resource-efficient deep learning models appropriate for real-time image processing pipelines. All code and training logs are publicly accessible at https://github.com/VineetKumarRakesh/lcnn-opt.
>
---
#### [new 070] A Deep Dive into Generic Object Tracking: A Survey
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的通用目标跟踪任务，旨在解决复杂场景下目标跟踪面临的遮挡、外观变化和干扰物等问题。论文系统综述了包括Siamese网络、判别式方法和基于Transformer的最新方法在内的多种跟踪范式，提出了新的分类体系，并通过可视化和表格对比分析了各类方法的设计原理、创新点与局限性，同时总结了主要评估基准，突出了基于Transformer方法在时空建模上的优势。**

- **链接: [http://arxiv.org/pdf/2507.23251v1](http://arxiv.org/pdf/2507.23251v1)**

> **作者:** Fereshteh Aghaee Meibodi; Shadi Alijani; Homayoun Najjaran
>
> **备注:** 55 pages, 29 figures, 9 tables
>
> **摘要:** Generic object tracking remains an important yet challenging task in computer vision due to complex spatio-temporal dynamics, especially in the presence of occlusions, similar distractors, and appearance variations. Over the past two decades, a wide range of tracking paradigms, including Siamese-based trackers, discriminative trackers, and, more recently, prominent transformer-based approaches, have been introduced to address these challenges. While a few existing survey papers in this field have either concentrated on a single category or widely covered multiple ones to capture progress, our paper presents a comprehensive review of all three categories, with particular emphasis on the rapidly evolving transformer-based methods. We analyze the core design principles, innovations, and limitations of each approach through both qualitative and quantitative comparisons. Our study introduces a novel categorization and offers a unified visual and tabular comparison of representative methods. Additionally, we organize existing trackers from multiple perspectives and summarize the major evaluation benchmarks, highlighting the fast-paced advancements in transformer-based tracking driven by their robust spatio-temporal modeling capabilities.
>
---
#### [new 071] Multi-Modal Motion Retrieval by Learning a Fine-Grained Joint Embedding Space
- **分类: cs.CV**

- **简介: 该论文属于多模态动作检索任务，旨在解决现有方法在交互方式和跨模态对齐上的不足。作者提出了一种细粒度联合嵌入空间框架，融合文本、音频、视频和动作模态，首次引入音频提升检索性能。通过序列级对比学习实现跨模态对齐，并构建了新的多模态动作检索数据集。实验表明其方法在多个子任务上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.23188v1](http://arxiv.org/pdf/2507.23188v1)**

> **作者:** Shiyao Yu; Zi-An Wang; Kangning Yin; Zheng Tian; Mingyuan Zhang; Weixin Si; Shihao Zou
>
> **备注:** Accepted by IEEE TMM 2025
>
> **摘要:** Motion retrieval is crucial for motion acquisition, offering superior precision, realism, controllability, and editability compared to motion generation. Existing approaches leverage contrastive learning to construct a unified embedding space for motion retrieval from text or visual modality. However, these methods lack a more intuitive and user-friendly interaction mode and often overlook the sequential representation of most modalities for improved retrieval performance. To address these limitations, we propose a framework that aligns four modalities -- text, audio, video, and motion -- within a fine-grained joint embedding space, incorporating audio for the first time in motion retrieval to enhance user immersion and convenience. This fine-grained space is achieved through a sequence-level contrastive learning approach, which captures critical details across modalities for better alignment. To evaluate our framework, we augment existing text-motion datasets with synthetic but diverse audio recordings, creating two multi-modal motion retrieval datasets. Experimental results demonstrate superior performance over state-of-the-art methods across multiple sub-tasks, including an 10.16% improvement in R@10 for text-to-motion retrieval and a 25.43% improvement in R@1 for video-to-motion retrieval on the HumanML3D dataset. Furthermore, our results show that our 4-modal framework significantly outperforms its 3-modal counterpart, underscoring the potential of multi-modal motion retrieval for advancing motion acquisition.
>
---
#### [new 072] Multi-Prompt Progressive Alignment for Multi-Source Unsupervised Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于多源无监督域适应（MS-UDA）任务，旨在解决现有方法在一次性对齐多源与目标域时因伪标签噪声导致的误差传播问题。作者提出MP²A方法，通过渐进式对齐策略，先从高置信度样本训练，逐步引入困难样本，从而提升模型鲁棒性和域不变特征学习效果。**

- **链接: [http://arxiv.org/pdf/2507.23373v1](http://arxiv.org/pdf/2507.23373v1)**

> **作者:** Haoran Chen; Zexiao Wang; Haidong Cao; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Large Vision-Language Models like CLIP have become a powerful foundation for Unsupervised Domain Adaptation due to their strong zero-shot generalization. State-of-the-art methods typically leverage CLIP to generate pseudo-labels for the target domain, then fine-tune the model to learn domain-invariant features. However, these methods attempt to align source and target domains using all pseudo-labeled data simultaneously. This one-shot alignment struggles with noisy, hard-to-classify samples, leading to error propagation and suboptimal feature learning. The problem is even more amplified in the multi-source scenario, where diverse domain gaps and varying noise levels across multiple source domains further destabilize the alignment process. To address this issue, in this work, we propose a progressive alignment strategy for adapting CLIP to unlabeled downstream task. Our method begins by training the model on a high-confidence subset of target samples, allowing it to first learn a well-aligned representation from the most reliable data. As training progresses, it gradually incorporates more challenging samples, guiding the model to refine its understanding without being overwhelmed by initial label noise. This progressive approach effectively mitigates confirmation bias and promotes a more robust convergence, allowing for the learning of genuinely domain-invariant features. We name our approach MP^2A and test it on three popular UDA benchmarks, namely ImageCLEF, Office-Home, and the most challenging DomainNet. Experiments showcase that MP^2A achieves state-of-the-art performance when compared with most recent CLIP-based MS-UDA approaches, demonstrating the effectiveness of our approach.
>
---
#### [new 073] Beyond Gloss: A Hand-Centric Framework for Gloss-Free Sign Language Translation
- **分类: cs.CV**

- **简介: 该论文属于手语翻译任务，旨在解决视觉与语言间模态差异及手部细节建模难题。论文提出“BeyondGloss”框架，利用视频大模型生成手部运动细粒度描述，并通过对比对齐模块与知识蒸馏提升手部表征，最终在两个基准数据集上取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2507.23575v1](http://arxiv.org/pdf/2507.23575v1)**

> **作者:** Sobhan Asasi; Mohamed Ilyas Lakhal; Ozge Mercanoglu Sincan; Richard Bowden
>
> **备注:** Accepted at BMVC 2025
>
> **摘要:** Sign Language Translation (SLT) is a challenging task that requires bridging the modality gap between visual and linguistic information while capturing subtle variations in hand shapes and movements. To address these challenges, we introduce \textbf{BeyondGloss}, a novel gloss-free SLT framework that leverages the spatio-temporal reasoning capabilities of Video Large Language Models (VideoLLMs). Since existing VideoLLMs struggle to model long videos in detail, we propose a novel approach to generate fine-grained, temporally-aware textual descriptions of hand motion. A contrastive alignment module aligns these descriptions with video features during pre-training, encouraging the model to focus on hand-centric temporal dynamics and distinguish signs more effectively. To further enrich hand-specific representations, we distill fine-grained features from HaMeR. Additionally, we apply a contrastive loss between sign video representations and target language embeddings to reduce the modality gap in pre-training. \textbf{BeyondGloss} achieves state-of-the-art performance on the Phoenix14T and CSL-Daily benchmarks, demonstrating the effectiveness of the proposed framework. We will release the code upon acceptance of the paper.
>
---
#### [new 074] YOLO-ROC: A High-Precision and Ultra-Lightweight Model for Real-Time Road Damage Detection
- **分类: cs.CV**

- **简介: 该论文属于道路损伤检测任务，旨在解决现有模型对小目标检测效果差、计算量大的问题。作者提出了YOLO-ROC模型，引入BMS-SPPF模块增强多尺度特征提取，并采用通道压缩策略降低计算复杂度，从而提升检测精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.23225v1](http://arxiv.org/pdf/2507.23225v1)**

> **作者:** Zicheng Lin; Weichao Pan
>
> **摘要:** Road damage detection is a critical task for ensuring traffic safety and maintaining infrastructure integrity. While deep learning-based detection methods are now widely adopted, they still face two core challenges: first, the inadequate multi-scale feature extraction capabilities of existing networks for diverse targets like cracks and potholes, leading to high miss rates for small-scale damage; and second, the substantial parameter counts and computational demands of mainstream models, which hinder their deployment for efficient, real-time detection in practical applications. To address these issues, this paper proposes a high-precision and lightweight model, YOLO - Road Orthogonal Compact (YOLO-ROC). We designed a Bidirectional Multi-scale Spatial Pyramid Pooling Fast (BMS-SPPF) module to enhance multi-scale feature extraction and implemented a hierarchical channel compression strategy to reduce computational complexity. The BMS-SPPF module leverages a bidirectional spatial-channel attention mechanism to improve the detection of small targets. Concurrently, the channel compression strategy reduces the parameter count from 3.01M to 0.89M and GFLOPs from 8.1 to 2.6. Experiments on the RDD2022_China_Drone dataset demonstrate that YOLO-ROC achieves a mAP50 of 67.6%, surpassing the baseline YOLOv8n by 2.11%. Notably, the mAP50 for the small-target D40 category improved by 16.8%, and the final model size is only 2.0 MB. Furthermore, the model exhibits excellent generalization performance on the RDD2022_China_Motorbike dataset.
>
---
#### [new 075] I Am Big, You Are Little; I Am Right, You Are Wrong
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分类任务，旨在分析不同模型的决策机制。通过提取“最小足够像素集”衡量模型对图像关键区域的关注程度，发现不同架构模型在关注区域的位置和大小上存在显著差异，且错误分类图像需要更多像素。**

- **链接: [http://arxiv.org/pdf/2507.23509v1](http://arxiv.org/pdf/2507.23509v1)**

> **作者:** David A. Kelly; Akchunya Chanchal; Nathan Blake
>
> **备注:** 10 pages, International Conference on Computer Vision, ICCV 2025
>
> **摘要:** Machine learning for image classification is an active and rapidly developing field. With the proliferation of classifiers of different sizes and different architectures, the problem of choosing the right model becomes more and more important. While we can assess a model's classification accuracy statistically, our understanding of the way these models work is unfortunately limited. In order to gain insight into the decision-making process of different vision models, we propose using minimal sufficient pixels sets to gauge a model's `concentration': the pixels that capture the essence of an image through the lens of the model. By comparing position, overlap, and size of sets of pixels, we identify that different architectures have statistically different concentration, in both size and position. In particular, ConvNext and EVA models differ markedly from the others. We also identify that images which are misclassified are associated with larger pixels sets than correct classifications.
>
---
#### [new 076] Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues
- **分类: cs.CV**

- **简介: 该论文属于逆向渲染任务，旨在从多视角图像中联合重建几何、反射属性和光照条件。传统方法依赖标定或中间线索，而本文提出一种无需光度立体线索的神经多视角自校准方法，通过隐式神经场和阴影感知体积渲染，直接从原始图像优化所有场景参数，提升了形状和光照估计的准确性。**

- **链接: [http://arxiv.org/pdf/2507.23162v1](http://arxiv.org/pdf/2507.23162v1)**

> **作者:** Xu Cao; Takafumi Taketomi
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** We propose a neural inverse rendering approach that jointly reconstructs geometry, spatially varying reflectance, and lighting conditions from multi-view images captured under varying directional lighting. Unlike prior multi-view photometric stereo methods that require light calibration or intermediate cues such as per-view normal maps, our method jointly optimizes all scene parameters from raw images in a single stage. We represent both geometry and reflectance as neural implicit fields and apply shadow-aware volume rendering. A spatial network first predicts the signed distance and a reflectance latent code for each scene point. A reflectance network then estimates reflectance values conditioned on the latent code and angularly encoded surface normal, view, and light directions. The proposed method outperforms state-of-the-art normal-guided approaches in shape and lighting estimation accuracy, generalizes to view-unaligned multi-light images, and handles objects with challenging geometry and reflectance.
>
---
#### [new 077] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于城市场景三维重建任务，旨在解决大规模场景的高效重建与实时渲染问题。论文提出了一种基于3D高斯点绘的框架，通过场景划分、可见性图像选择、细节层次控制、外观变换及增强模块，实现高效、高质的城市级场景重建。**

- **链接: [http://arxiv.org/pdf/2507.23006v1](http://arxiv.org/pdf/2507.23006v1)**

> **作者:** Zhensheng Yuan; Haozhi Huang; Zhen Xiong; Di Wang; Guanghua Yang
>
> **摘要:** We present a framework that enables fast reconstruction and real-time rendering of urban-scale scenes while maintaining robustness against appearance variations across multi-view captures. Our approach begins with scene partitioning for parallel training, employing a visibility-based image selection strategy to optimize training efficiency. A controllable level-of-detail (LOD) strategy explicitly regulates Gaussian density under a user-defined budget, enabling efficient training and rendering while maintaining high visual fidelity. The appearance transformation module mitigates the negative effects of appearance inconsistencies across images while enabling flexible adjustments. Additionally, we utilize enhancement modules, such as depth regularization, scale regularization, and antialiasing, to improve reconstruction fidelity. Experimental results demonstrate that our method effectively reconstructs urban-scale scenes and outperforms previous approaches in both efficiency and quality. The source code is available at: https://yzslab.github.io/REUrbanGS.
>
---
#### [new 078] Confidence-aware agglomeration classification and segmentation of 2D microscopic food crystal images
- **分类: cs.CV**

- **简介: 该论文属于图像分类与分割任务，旨在解决食品晶体显微图像中的团聚现象识别难题。由于手动标注困难且透明水桥不易识别，论文提出一种结合分类与分割的监督模型，并设计后处理模块提升准确率，有效预测团聚区域及其尺寸分布。**

- **链接: [http://arxiv.org/pdf/2507.23206v1](http://arxiv.org/pdf/2507.23206v1)**

> **作者:** Xiaoyu Ji; Ali Shakouri; Fengqing Zhu
>
> **摘要:** Food crystal agglomeration is a phenomenon occurs during crystallization which traps water between crystals and affects food product quality. Manual annotation of agglomeration in 2D microscopic images is particularly difficult due to the transparency of water bonding and the limited perspective focusing on a single slide of the imaged sample. To address this challenge, we first propose a supervised baseline model to generate segmentation pseudo-labels for the coarsely labeled classification dataset. Next, an instance classification model that simultaneously performs pixel-wise segmentation is trained. Both models are used in the inference stage to combine their respective strengths in classification and segmentation. To preserve crystal properties, a post processing module is designed and included to both steps. Our method improves true positive agglomeration classification accuracy and size distribution predictions compared to other existing methods. Given the variability in confidence levels of manual annotations, our proposed method is evaluated under two confidence levels and successfully classifies potential agglomerated instances.
>
---
#### [new 079] DivControl: Knowledge Diversion for Controllable Image Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决现有方法在多条件控制生成中泛化性差、适配成本高的问题。作者提出DivControl框架，通过知识分流实现解耦表征学习，结合动态门控和对齐损失，提升可控性与零样本迁移能力。**

- **链接: [http://arxiv.org/pdf/2507.23620v1](http://arxiv.org/pdf/2507.23620v1)**

> **作者:** Yucheng Xie; Fu Feng; Ruixiao Shi; Jing Wang; Yong Rui; Xin Geng
>
> **摘要:** Diffusion models have advanced from text-to-image (T2I) to image-to-image (I2I) generation by incorporating structured inputs such as depth maps, enabling fine-grained spatial control. However, existing methods either train separate models for each condition or rely on unified architectures with entangled representations, resulting in poor generalization and high adaptation costs for novel conditions. To this end, we propose DivControl, a decomposable pretraining framework for unified controllable generation and efficient adaptation. DivControl factorizes ControlNet via SVD into basic components-pairs of singular vectors-which are disentangled into condition-agnostic learngenes and condition-specific tailors through knowledge diversion during multi-condition training. Knowledge diversion is implemented via a dynamic gate that performs soft routing over tailors based on the semantics of condition instructions, enabling zero-shot generalization and parameter-efficient adaptation to novel conditions. To further improve condition fidelity and training efficiency, we introduce a representation alignment loss that aligns condition embeddings with early diffusion features. Extensive experiments demonstrate that DivControl achieves state-of-the-art controllability with 36.4$\times$ less training cost, while simultaneously improving average performance on basic conditions. It also delivers strong zero-shot and few-shot performance on unseen conditions, demonstrating superior scalability, modularity, and transferability.
>
---
#### [new 080] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval
- **分类: cs.CV**

- **简介: 论文属于文本-视频检索任务，旨在解决多模态检索中候选先验偏差问题。作者提出BLiM框架，结合双向似然估计与候选先验归一化（CPN），提升检索相关性。方法在多个基准上取得显著提升，并展现CPN在多模任务中的广泛适用性。**

- **链接: [http://arxiv.org/pdf/2507.23284v1](http://arxiv.org/pdf/2507.23284v1)**

> **作者:** Dohwan Ko; Ji Soo Lee; Minhyuk Choi; Zihang Meng; Hyunwoo J. Kim
>
> **备注:** ICCV 2025 Highlight
>
> **摘要:** Text-Video Retrieval aims to find the most relevant text (or video) candidate given a video (or text) query from large-scale online databases. Recent work leverages multi-modal large language models (MLLMs) to improve retrieval, especially for long or complex query-candidate pairs. However, we observe that the naive application of MLLMs, i.e., retrieval based on candidate likelihood, introduces candidate prior bias, favoring candidates with inherently higher priors over those more relevant to the query. To this end, we propose a novel retrieval framework, Bidirectional Likelihood Estimation with MLLM (BLiM), which leverages both query and candidate likelihoods by training the model to generate text from a given video as well as video features from a given text. Furthermore, we introduce Candidate Prior Normalization (CPN), a simple yet effective training-free score calibration module designed to mitigate candidate prior bias in candidate likelihood. On four Text-Video Retrieval benchmarks, our BLiM equipped with CPN outperforms previous state-of-the-art models by 6.4 R@1 on average, effectively alleviating candidate prior bias and emphasizing query-candidate relevance. Our in-depth analysis across various multi-modal tasks beyond retrieval highlights the broad applicability of CPN which enhances visual understanding by reducing reliance on textual priors. Code is available at https://github.com/mlvlab/BLiM.
>
---
#### [new 081] Online Estimation of Table-Top Grown Strawberry Mass in Field Conditions with Occlusions
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于计算机视觉与农业自动化任务，旨在解决田间条件下桌面种植草莓因遮挡和姿态变化导致的质量估计难题。作者提出了一种结合RGB-D传感与深度学习的在线质量估计算法，采用YOLOv8-Seg实例分割、CycleGAN修复遮挡区域、倾角校正及多项式回归模型，实现了高精度非接触式草莓质量估计。**

- **链接: [http://arxiv.org/pdf/2507.23487v1](http://arxiv.org/pdf/2507.23487v1)**

> **作者:** Jinshan Zhen; Yuanyue Ge; Tianxiao Zhu; Hui Zhao; Ya Xiong
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Accurate mass estimation of table-top grown strawberries under field conditions remains challenging due to frequent occlusions and pose variations. This study proposes a vision-based pipeline integrating RGB-D sensing and deep learning to enable non-destructive, real-time and online mass estimation. The method employed YOLOv8-Seg for instance segmentation, Cycle-consistent generative adversarial network (CycleGAN) for occluded region completion, and tilt-angle correction to refine frontal projection area calculations. A polynomial regression model then mapped the geometric features to mass. Experiments demonstrated mean mass estimation errors of 8.11% for isolated strawberries and 10.47% for occluded cases. CycleGAN outperformed large mask inpainting (LaMa) model in occlusion recovery, achieving superior pixel area ratios (PAR) (mean: 0.978 vs. 1.112) and higher intersection over union (IoU) scores (92.3% vs. 47.7% in the [0.9-1] range). This approach addresses critical limitations of traditional methods, offering a robust solution for automated harvesting and yield monitoring with complex occlusion patterns.
>
---
#### [new 082] VMatcher: State-Space Semi-Dense Local Feature Matching
- **分类: cs.CV**

- **简介: 该论文属于图像匹配任务，旨在解决现有方法计算成本高的问题。作者提出了VMatcher，结合Mamba的高效状态空间模型与Transformer的注意力机制，实现快速且性能优越的半稠密特征匹配，适用于实时应用。**

- **链接: [http://arxiv.org/pdf/2507.23371v1](http://arxiv.org/pdf/2507.23371v1)**

> **作者:** Ali Youssef
>
> **摘要:** This paper introduces VMatcher, a hybrid Mamba-Transformer network for semi-dense feature matching between image pairs. Learning-based feature matching methods, whether detector-based or detector-free, achieve state-of-the-art performance but depend heavily on the Transformer's attention mechanism, which, while effective, incurs high computational costs due to its quadratic complexity. In contrast, Mamba introduces a Selective State-Space Model (SSM) that achieves comparable or superior performance with linear complexity, offering significant efficiency gains. VMatcher leverages a hybrid approach, integrating Mamba's highly efficient long-sequence processing with the Transformer's attention mechanism. Multiple VMatcher configurations are proposed, including hierarchical architectures, demonstrating their effectiveness in setting new benchmarks efficiently while ensuring robustness and practicality for real-time applications where rapid inference is crucial. Source Code is available at: https://github.com/ayoussf/VMatcher
>
---
#### [new 083] Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，旨在保护隐私的同时实现精准定位。为解决隐私泄露问题，论文提出高斯点特征场（GSFFs），结合显式几何模型与隐式特征场，利用对比学习与聚类方法学习鲁棒特征，并通过特征或分割图对齐实现定位，取得最优性能。**

- **链接: [http://arxiv.org/pdf/2507.23569v1](http://arxiv.org/pdf/2507.23569v1)**

> **作者:** Maxime Pietrantoni; Gabriela Csurka; Torsten Sattler
>
> **备注:** CVPR 2025
>
> **摘要:** Visual localization is the task of estimating a camera pose in a known environment. In this paper, we utilize 3D Gaussian Splatting (3DGS)-based representations for accurate and privacy-preserving visual localization. We propose Gaussian Splatting Feature Fields (GSFFs), a scene representation for visual localization that combines an explicit geometry model (3DGS) with an implicit feature field. We leverage the dense geometric information and differentiable rasterization algorithm from 3DGS to learn robust feature representations grounded in 3D. In particular, we align a 3D scale-aware feature field and a 2D feature encoder in a common embedding space through a contrastive framework. Using a 3D structure-informed clustering procedure, we further regularize the representation learning and seamlessly convert the features to segmentations, which can be used for privacy-preserving visual localization. Pose refinement, which involves aligning either feature maps or segmentations from a query image with those rendered from the GSFFs scene representation, is used to achieve localization. The resulting privacy- and non-privacy-preserving localization pipelines, evaluated on multiple real-world datasets, show state-of-the-art performances.
>
---
#### [new 084] Seeing More with Less: Video Capsule Endoscopy with Multi-Task Learning
- **分类: cs.CV**

- **简介: 该论文属于医疗AI任务，旨在解决胶囊内镜电池寿命短的问题。通过多任务学习，将胃肠道定位与异常检测结合，实现低参数（仅100万）高效模型。在Galar数据集上表现优于现有方法，定位准确率93.63%，异常检测87.48%。**

- **链接: [http://arxiv.org/pdf/2507.23479v1](http://arxiv.org/pdf/2507.23479v1)**

> **作者:** Julia Werner; Oliver Bause; Julius Oexle; Maxime Le Floch; Franz Brinkmann; Jochen Hampe; Oliver Bringmann
>
> **备注:** Accepted at Applications of Medical AI (AMAI workshop) at MICCAI 2025 (submitted version)
>
> **摘要:** Video capsule endoscopy has become increasingly important for investigating the small intestine within the gastrointestinal tract. However, a persistent challenge remains the short battery lifetime of such compact sensor edge devices. Integrating artificial intelligence can help overcome this limitation by enabling intelligent real-time decision- making, thereby reducing the energy consumption and prolonging the battery life. However, this remains challenging due to data sparsity and the limited resources of the device restricting the overall model size. In this work, we introduce a multi-task neural network that combines the functionalities of precise self-localization within the gastrointestinal tract with the ability to detect anomalies in the small intestine within a single model. Throughout the development process, we consistently restricted the total number of parameters to ensure the feasibility to deploy such model in a small capsule. We report the first multi-task results using the recently published Galar dataset, integrating established multi-task methods and Viterbi decoding for subsequent time-series analysis. This outperforms current single-task models and represents a significant ad- vance in AI-based approaches in this field. Our model achieves an accu- racy of 93.63% on the localization task and an accuracy of 87.48% on the anomaly detection task. The approach requires only 1 million parameters while surpassing the current baselines.
>
---
#### [new 085] MonoFusion: Sparse-View 4D Reconstruction via Monocular Fusion
- **分类: cs.CV**

- **简介: 该论文属于动态场景重建任务，旨在解决从稀疏视角视频中重建动态场景的问题。现有方法依赖密集多视角相机，成本高昂且场景受限。论文提出MonoFusion，通过融合单目重建结果，实现多视角一致性重建，适用于如修复自行车或跳舞等动态场景。**

- **链接: [http://arxiv.org/pdf/2507.23782v1](http://arxiv.org/pdf/2507.23782v1)**

> **作者:** Zihan Wang; Jeff Tan; Tarasha Khurana; Neehar Peri; Deva Ramanan
>
> **备注:** ICCV 2025. Project Page: https://imnotprepared.github.io/research/25_DSR/
>
> **摘要:** We address the problem of dynamic scene reconstruction from sparse-view videos. Prior work often requires dense multi-view captures with hundreds of calibrated cameras (e.g. Panoptic Studio). Such multi-view setups are prohibitively expensive to build and cannot capture diverse scenes in-the-wild. In contrast, we aim to reconstruct dynamic human behaviors, such as repairing a bike or dancing, from a small set of sparse-view cameras with complete scene coverage (e.g. four equidistant inward-facing static cameras). We find that dense multi-view reconstruction methods struggle to adapt to this sparse-view setup due to limited overlap between viewpoints. To address these limitations, we carefully align independent monocular reconstructions of each camera to produce time- and view-consistent dynamic scene reconstructions. Extensive experiments on PanopticStudio and Ego-Exo4D demonstrate that our method achieves higher quality reconstructions than prior art, particularly when rendering novel views. Code, data, and data-processing scripts are available on https://github.com/ImNotPrepared/MonoFusion.
>
---
#### [new 086] Reference-Guided Diffusion Inpainting For Multimodal Counterfactual Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决安全关键领域（如自动驾驶和医学影像）中合成数据生成的真实性与可控性问题。论文提出了MObI和AnydoorMed两种方法，分别用于自动驾驶和医学影像中的多模态图像修复，实现基于参考图像和3D框的物体插入与异常结构修复，保持多模态一致性与细节真实性。**

- **链接: [http://arxiv.org/pdf/2507.23058v1](http://arxiv.org/pdf/2507.23058v1)**

> **作者:** Alexandru Buburuzan
>
> **备注:** A dissertation submitted to The University of Manchester for the degree of Bachelor of Science in Artificial Intelligence
>
> **摘要:** Safety-critical applications, such as autonomous driving and medical image analysis, require extensive multimodal data for rigorous testing. Synthetic data methods are gaining prominence due to the cost and complexity of gathering real-world data, but they demand a high degree of realism and controllability to be useful. This work introduces two novel methods for synthetic data generation in autonomous driving and medical image analysis, namely MObI and AnydoorMed, respectively. MObI is a first-of-its-kind framework for Multimodal Object Inpainting that leverages a diffusion model to produce realistic and controllable object inpaintings across perceptual modalities, demonstrated simultaneously for camera and lidar. Given a single reference RGB image, MObI enables seamless object insertion into existing multimodal scenes at a specified 3D location, guided by a bounding box, while maintaining semantic consistency and multimodal coherence. Unlike traditional inpainting methods that rely solely on edit masks, this approach uses 3D bounding box conditioning to ensure accurate spatial positioning and realistic scaling. AnydoorMed extends this paradigm to the medical imaging domain, focusing on reference-guided inpainting for mammography scans. It leverages a diffusion-based model to inpaint anomalies with impressive detail preservation, maintaining the reference anomaly's structural integrity while semantically blending it with the surrounding tissue. Together, these methods demonstrate that foundation models for reference-guided inpainting in natural images can be readily adapted to diverse perceptual modalities, paving the way for the next generation of systems capable of constructing highly realistic, controllable and multimodal counterfactual scenarios.
>
---
#### [new 087] DA-Occ: Efficient 3D Voxel Occupancy Prediction via Directional 2D for Geometric Structure Preservation
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的3D占用预测任务，旨在解决现有方法在准确率与推理速度间难以平衡的问题。作者提出DA-Occ方法，通过方向性2D策略保留3D几何结构信息，并引入方向注意力机制提升效率。实验表明该方法在精度与速度上取得良好平衡，适用于边缘设备部署。**

- **链接: [http://arxiv.org/pdf/2507.23599v1](http://arxiv.org/pdf/2507.23599v1)**

> **作者:** Yuchen Zhou; Yan Luo; Xiangang Wang; Xingjian Gu; Mingzhou Lu
>
> **摘要:** Efficient and high-accuracy 3D occupancy prediction is crucial for ensuring the performance of autonomous driving (AD) systems. However, many current methods focus on high accuracy at the expense of real-time processing needs. To address this challenge of balancing accuracy and inference speed, we propose a directional pure 2D approach. Our method involves slicing 3D voxel features to preserve complete vertical geometric information. This strategy compensates for the loss of height cues in Bird's-Eye View (BEV) representations, thereby maintaining the integrity of the 3D geometric structure. By employing a directional attention mechanism, we efficiently extract geometric features from different orientations, striking a balance between accuracy and computational efficiency. Experimental results highlight the significant advantages of our approach for autonomous driving. On the Occ3D-nuScenes, the proposed method achieves an mIoU of 39.3% and an inference speed of 27.7 FPS, effectively balancing accuracy and efficiency. In simulations on edge devices, the inference speed reaches 14.8 FPS, further demonstrating the method's applicability for real-time deployment in resource-constrained environments.
>
---
#### [new 088] Medical Image De-Identification Benchmark Challenge
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于医学图像去标识化任务，旨在解决共享医疗图像时保护患者隐私与保留研究元数据的问题。论文设计并实施了MIDI-B挑战赛，提供标准化平台评估去标识化工具效果，采用合成敏感信息的真实图像数据进行测试，最终展示了多种方法的性能表现与实践经验。**

- **链接: [http://arxiv.org/pdf/2507.23608v1](http://arxiv.org/pdf/2507.23608v1)**

> **作者:** Linmin Pei; Granger Sutton; Michael Rutherford; Ulrike Wagner; Tracy Nolan; Kirk Smith; Phillip Farmer; Peter Gu; Ambar Rana; Kailing Chen; Thomas Ferleman; Brian Park; Ye Wu; Jordan Kojouharov; Gargi Singh; Jon Lemon; Tyler Willis; Milos Vukadinovic; Grant Duffy; Bryan He; David Ouyang; Marco Pereanez; Daniel Samber; Derek A. Smith; Christopher Cannistraci; Zahi Fayad; David S. Mendelson; Michele Bufano; Elmar Kotter; Hamideh Haghiri; Rajesh Baidya; Stefan Dvoretskii; Klaus H. Maier-Hein; Marco Nolden; Christopher Ablett; Silvia Siggillino; Sandeep Kaushik; Hongzhu Jiang; Sihan Xie; Zhiyu Wan; Alex Michie; Simon J Doran; Angeline Aurelia Waly; Felix A. Nathaniel Liang; Humam Arshad Mustagfirin; Michelle Grace Felicia; Kuo Po Chih; Rahul Krish; Ghulam Rasool; Nidhal Bouaynaya; Nikolas Koutsoubis; Kyle Naddeo; Kartik Pandit; Tony O'Sullivan; Raj Krish; Qinyan Pan; Scott Gustafson; Benjamin Kopchick; Laura Opsahl-Ong; Andrea Olvera-Morales; Jonathan Pinney; Kathryn Johnson; Theresa Do; Juergen Klenk; Maria Diaz; Arti Singh; Rong Chai; David A. Clunie; Fred Prior; Keyvan Farahani
>
> **备注:** 19 pages
>
> **摘要:** The de-identification (deID) of protected health information (PHI) and personally identifiable information (PII) is a fundamental requirement for sharing medical images, particularly through public repositories, to ensure compliance with patient privacy laws. In addition, preservation of non-PHI metadata to inform and enable downstream development of imaging artificial intelligence (AI) is an important consideration in biomedical research. The goal of MIDI-B was to provide a standardized platform for benchmarking of DICOM image deID tools based on a set of rules conformant to the HIPAA Safe Harbor regulation, the DICOM Attribute Confidentiality Profiles, and best practices in preservation of research-critical metadata, as defined by The Cancer Imaging Archive (TCIA). The challenge employed a large, diverse, multi-center, and multi-modality set of real de-identified radiology images with synthetic PHI/PII inserted. The MIDI-B Challenge consisted of three phases: training, validation, and test. Eighty individuals registered for the challenge. In the training phase, we encouraged participants to tune their algorithms using their in-house or public data. The validation and test phases utilized the DICOM images containing synthetic identifiers (of 216 and 322 subjects, respectively). Ten teams successfully completed the test phase of the challenge. To measure success of a rule-based approach to image deID, scores were computed as the percentage of correct actions from the total number of required actions. The scores ranged from 97.91% to 99.93%. Participants employed a variety of open-source and proprietary tools with customized configurations, large language models, and optical character recognition (OCR). In this paper we provide a comprehensive report on the MIDI-B Challenge's design, implementation, results, and lessons learned.
>
---
#### [new 089] Who is a Better Talker: Subjective and Objective Quality Assessment for AI-Generated Talking Heads
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于AI生成数字人质量评估任务，旨在解决AI生成对话头像的质量问题。论文构建了包含10,457个样本的THQA-10K数据集，招募志愿者进行主观评分，并提出一种新的客观质量评估方法，实现了当前最优性能。**

- **链接: [http://arxiv.org/pdf/2507.23343v1](http://arxiv.org/pdf/2507.23343v1)**

> **作者:** Yingjie Zhou; Jiezhang Cao; Zicheng Zhang; Farong Wen; Yanwei Jiang; Jun Jia; Xiaohong Liu; Xiongkuo Min; Guangtao Zhai
>
> **摘要:** Speech-driven methods for portraits are figuratively known as "Talkers" because of their capability to synthesize speaking mouth shapes and facial movements. Especially with the rapid development of the Text-to-Image (T2I) models, AI-Generated Talking Heads (AGTHs) have gradually become an emerging digital human media. However, challenges persist regarding the quality of these talkers and AGTHs they generate, and comprehensive studies addressing these issues remain limited. To address this gap, this paper presents the largest AGTH quality assessment dataset THQA-10K to date, which selects 12 prominent T2I models and 14 advanced talkers to generate AGTHs for 14 prompts. After excluding instances where AGTH generation is unsuccessful, the THQA-10K dataset contains 10,457 AGTHs. Then, volunteers are recruited to subjectively rate the AGTHs and give the corresponding distortion categories. In our analysis for subjective experimental results, we evaluate the performance of talkers in terms of generalizability and quality, and also expose the distortions of existing AGTHs. Finally, an objective quality assessment method based on the first frame, Y-T slice and tone-lip consistency is proposed. Experimental results show that this method can achieve state-of-the-art (SOTA) performance in AGTH quality assessment. The work is released at https://github.com/zyj-2000/Talker.
>
---
#### [new 090] Mitigating Resolution-Drift in Federated Learning: Case of Keypoint Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于联邦学习中的人体姿态估计任务，旨在解决因客户端分辨率差异导致的“分辨率漂移”问题。作者提出了基于热图知识蒸馏的分辨率自适应联邦学习方法（RAF），通过多分辨率蒸馏提升模型在不同分辨率下的鲁棒性，有效缓解分辨率异构性带来的性能下降。**

- **链接: [http://arxiv.org/pdf/2507.23461v1](http://arxiv.org/pdf/2507.23461v1)**

> **作者:** Taeheon Lim; Joohyung Lee; Kyungjae Lee; Jungchan Cho
>
> **摘要:** The Federated Learning (FL) approach enables effective learning across distributed systems, while preserving user data privacy. To date, research has primarily focused on addressing statistical heterogeneity and communication efficiency, through which FL has achieved success in classification tasks. However, its application to non-classification tasks, such as human pose estimation, remains underexplored. This paper identifies and investigates a critical issue termed ``resolution-drift,'' where performance degrades significantly due to resolution variability across clients. Unlike class-level heterogeneity, resolution drift highlights the importance of resolution as another axis of not independent or identically distributed (non-IID) data. To address this issue, we present resolution-adaptive federated learning (RAF), a method that leverages heatmap-based knowledge distillation. Through multi-resolution knowledge distillation between higher-resolution outputs (teachers) and lower-resolution outputs (students), our approach enhances resolution robustness without overfitting. Extensive experiments and theoretical analysis demonstrate that RAF not only effectively mitigates resolution drift and achieves significant performance improvements, but also can be integrated seamlessly into existing FL frameworks. Furthermore, although this paper focuses on human pose estimation, our t-SNE analysis reveals distinct characteristics between classification and high-resolution representation tasks, supporting the generalizability of RAF to other tasks that rely on preserving spatial detail.
>
---
#### [new 091] MagicRoad: Semantic-Aware 3D Road Surface Reconstruction via Obstacle Inpainting
- **分类: cs.CV**

- **简介: 论文属于三维重建任务，旨在解决复杂城市环境中道路表面重建受遮挡和光照变化影响的问题。提出了MagicRoad框架，采用语义感知的2D高斯点云和色彩增强方法，实现高效、鲁棒的大规模道路建模。**

- **链接: [http://arxiv.org/pdf/2507.23340v1](http://arxiv.org/pdf/2507.23340v1)**

> **作者:** Xingyue Peng; Yuandong Lyu; Lang Zhang; Jian Zhu; Songtao Wang; Jiaxin Deng; Songxin Lu; Weiliang Ma; Dangen She; Peng Jia; XianPeng Lang
>
> **摘要:** Road surface reconstruction is essential for autonomous driving, supporting centimeter-accurate lane perception and high-definition mapping in complex urban environments.While recent methods based on mesh rendering or 3D Gaussian splatting (3DGS) achieve promising results under clean and static conditions, they remain vulnerable to occlusions from dynamic agents, visual clutter from static obstacles, and appearance degradation caused by lighting and weather changes. We present a robust reconstruction framework that integrates occlusion-aware 2D Gaussian surfels with semantic-guided color enhancement to recover clean, consistent road surfaces. Our method leverages a planar-adapted Gaussian representation for efficient large-scale modeling, employs segmentation-guided video inpainting to remove both dynamic and static foreground objects, and enhances color coherence via semantic-aware correction in HSV space. Extensive experiments on urban-scale datasets demonstrate that our framework produces visually coherent and geometrically faithful reconstructions, significantly outperforming prior methods under real-world conditions.
>
---
#### [new 092] ART: Adaptive Relation Tuning for Generalized Relation Prediction
- **分类: cs.CV; cs.AI**

- **简介: 论文属于视觉关系检测（VRD）任务，旨在解决现有模型泛化能力差、难以识别新关系的问题。作者提出ART框架，通过指令调优和自适应采样提升视觉语言模型在关系分类中的表现，实现了对未见关系的推理能力，并验证了其在复杂场景分割中的实用性。**

- **链接: [http://arxiv.org/pdf/2507.23543v1](http://arxiv.org/pdf/2507.23543v1)**

> **作者:** Gopika Sudhakaran; Hikaru Shindo; Patrick Schramowski; Simone Schaub-Meyer; Kristian Kersting; Stefan Roth
>
> **备注:** Accepted for publication in ICCV 2025
>
> **摘要:** Visual relation detection (VRD) is the task of identifying the relationships between objects in a scene. VRD models trained solely on relation detection data struggle to generalize beyond the relations on which they are trained. While prompt tuning has been used to adapt vision-language models (VLMs) for VRD, it uses handcrafted prompts and struggles with novel or complex relations. We argue that instruction tuning offers a more effective solution by fine-tuning VLMs on diverse instructional data. We thus introduce ART, an Adaptive Relation Tuning framework that adapts VLMs for VRD through instruction tuning and strategic instance selection. By converting VRD datasets into an instruction tuning format and employing an adaptive sampling algorithm, ART directs the VLM to focus on informative relations while maintaining generalizability. Specifically, we focus on the relation classification, where subject-object boxes are given and the model predicts the predicate between them. We tune on a held-in set and evaluate across multiple held-out datasets of varying complexity. Our approach strongly improves over its baselines and can infer unseen relation concepts, a capability absent in mainstream VRD methods. We demonstrate ART's practical value by using the predicted relations for segmenting complex scenes.
>
---
#### [new 093] Towards Measuring and Modeling Geometric Structures in Time Series Forecasting via Image Modality
- **分类: cs.CV**

- **简介: 该论文属于时间序列预测任务，旨在解决现有数值指标无法评估时间序列几何结构的问题。作者提出了TGSI指标，通过图像模态衡量几何结构，并设计了可融入训练的SATL损失函数，结合一阶差分、频域和感知特征，提升模型对结构的建模能力。**

- **链接: [http://arxiv.org/pdf/2507.23253v1](http://arxiv.org/pdf/2507.23253v1)**

> **作者:** Mingyang Yu; Xiahui Guo; Peng chen; Zhenkai Li; Yang Shu
>
> **摘要:** Time Series forecasting is critical in diverse domains such as weather forecasting, financial investment, and traffic management. While traditional numerical metrics like mean squared error (MSE) can quantify point-wise accuracy, they fail to evaluate the geometric structure of time series data, which is essential to understand temporal dynamics. To address this issue, we propose the time series Geometric Structure Index (TGSI), a novel evaluation metric that transforms time series into images to leverage their inherent two-dimensional geometric representations. However, since the image transformation process is non-differentiable, TGSI cannot be directly integrated as a training loss. We further introduce the Shape-Aware Temporal Loss (SATL), a multi-component loss function operating in the time series modality to bridge this gap and enhance structure modeling during training. SATL combines three components: a first-order difference loss that measures structural consistency through the MSE between first-order differences, a frequency domain loss that captures essential periodic patterns using the Fast Fourier Transform while minimizing noise, and a perceptual feature loss that measures geometric structure difference in time-series by aligning temporal features with geometric structure features through a pre-trained temporal feature extractor and time-series image autoencoder. Experiments across multiple datasets demonstrate that models trained with SATL achieve superior performance in both MSE and the proposed TGSI metrics compared to baseline methods, without additional computational cost during inference.
>
---
#### [new 094] Vision-Language Fusion for Real-Time Autonomous Driving: Goal-Centered Cross-Attention of Camera, HD-Map, & Waypoints
- **分类: cs.CV; cs.AI; cs.LG; cs.RO; I.4.8; I.2.10; I.2.6; C.3.3; I.4.9**

- **简介: 该论文属于自动驾驶任务，旨在解决复杂环境中几何精度与语义理解分离的问题。论文提出XYZ-Drive模型，通过视觉-语言融合与目标中心注意力机制，结合摄像头、高精地图和路径点，实现高效实时驾驶决策，提升了成功率与安全性。**

- **链接: [http://arxiv.org/pdf/2507.23064v1](http://arxiv.org/pdf/2507.23064v1)**

> **作者:** Santosh Patapati; Trisanth Srinivasan; Murari Ambati
>
> **备注:** 5 pages
>
> **摘要:** Autonomous cars need geometric accuracy and semantic understanding to navigate complex environments, yet most stacks handle them separately. We present XYZ-Drive, a single vision-language model that reads a front-camera frame, a 25m $\times$ 25m overhead map, and the next waypoint, then outputs steering and speed. A lightweight goal-centered cross-attention layer lets waypoint tokens highlight relevant image and map patches, supporting both action and textual explanations, before the fused tokens enter a partially fine-tuned LLaMA-3.2 11B model. On the MD-NEX Outdoor-Driving benchmark XYZ-Drive attains 95% success and 0.80 Success weighted by Path Length (SPL), surpassing PhysNav-DG by 15%. and halving collisions, all while significantly improving efficiency by using only a single branch. Sixteen ablations explain the gains. Removing any modality (vision, waypoint, map) drops success by up to 11%, confirming their complementary roles and rich connections. Replacing goal-centered attention with simple concatenation cuts 3% in performance, showing query-based fusion injects map knowledge more effectively. Keeping the transformer frozen loses 5%, showing the importance of fine-tuning when applying VLMs for specific tasks such as autonomous driving. Coarsening map resolution from 10 cm to 40 cm blurs lane edges and raises crash rate. Overall, these results demonstrate that early, token-level fusion of intent and map layout enables accurate, transparent, real-time driving.
>
---
#### [new 095] UniEmo: Unifying Emotional Understanding and Generation with Learnable Expert Queries
- **分类: cs.CV**

- **简介: 论文提出UniEmo框架，统一情感理解和生成任务。通过层次化情感理解链与可学习专家查询，提取多尺度情感特征，并融合至扩散模型生成情感图像。引入情感相关系数与条件损失提升生成质量，结合数据过滤算法实现双向反馈优化。解决了情感理解与生成分离的问题，提升了模型整体性能。**

- **链接: [http://arxiv.org/pdf/2507.23372v1](http://arxiv.org/pdf/2507.23372v1)**

> **作者:** Yijie Zhu; Lingsen Zhang; Zitong Yu; Rui Shao; Tao Tan; Liqiang Nie
>
> **摘要:** Emotional understanding and generation are often treated as separate tasks, yet they are inherently complementary and can mutually enhance each other. In this paper, we propose the UniEmo, a unified framework that seamlessly integrates these two tasks. The key challenge lies in the abstract nature of emotions, necessitating the extraction of visual representations beneficial for both tasks. To address this, we propose a hierarchical emotional understanding chain with learnable expert queries that progressively extracts multi-scale emotional features, thereby serving as a foundational step for unification. Simultaneously, we fuse these expert queries and emotional representations to guide the diffusion model in generating emotion-evoking images. To enhance the diversity and fidelity of the generated emotional images, we further introduce the emotional correlation coefficient and emotional condition loss into the fusion process. This step facilitates fusion and alignment for emotional generation guided by the understanding. In turn, we demonstrate that joint training allows the generation component to provide implicit feedback to the understanding part. Furthermore, we propose a novel data filtering algorithm to select high-quality and diverse emotional images generated by the well-trained model, which explicitly feedback into the understanding part. Together, these generation-driven dual feedback processes enhance the model's understanding capacity. Extensive experiments show that UniEmo significantly outperforms state-of-the-art methods in both emotional understanding and generation tasks. The code for the proposed method is available at https://github.com/JiuTian-VL/UniEmo.
>
---
#### [new 096] Towards High-Resolution Alignment and Super-Resolution of Multi-Sensor Satellite Imagery
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决多传感器卫星影像分辨率差异导致的数据融合难题。作者提出了一种对齐和融合Landsat与Sentinel卫星影像的初步框架，利用HLS10数据作为参考提升HLS30影像的分辨率，从而提高遥感应用的图像质量。**

- **链接: [http://arxiv.org/pdf/2507.23150v1](http://arxiv.org/pdf/2507.23150v1)**

> **作者:** Philip Wootaek Shin; Vishal Gaur; Rahul Ramachandran; Manil Maskey; Jack Sampson; Vijaykrishnan Narayanan; Sujit Roy
>
> **摘要:** High-resolution satellite imagery is essential for geospatial analysis, yet differences in spatial resolution across satellite sensors present challenges for data fusion and downstream applications. Super-resolution techniques can help bridge this gap, but existing methods rely on artificially downscaled images rather than real sensor data and are not well suited for heterogeneous satellite sensors with differing spectral, temporal characteristics. In this work, we develop a preliminary framework to align and Harmonized Landsat Sentinel 30m(HLS 30) imagery using Harmonized Landsat Sentinel 10m(HLS10) as a reference from the HLS dataset. Our approach aims to bridge the resolution gap between these sensors and improve the quality of super-resolved Landsat imagery. Quantitative and qualitative evaluations demonstrate the effectiveness of our method, showing its potential for enhancing satellite-based sensing applications. This study provides insights into the feasibility of heterogeneous satellite image super-resolution and highlights key considerations for future advancements in the field.
>
---
#### [new 097] A Unified Perception-Language-Action Framework for Adaptive Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出了一种统一的感知-语言-行动（PLA）框架，用于自动驾驶，融合多传感器数据与大语言模型（GPT-4.1），提升系统在复杂环境中的适应性、可解释性与安全性。属于自动驾驶任务，旨在解决现有系统在开放世界中适应性差、泛化能力弱及语义理解不足的问题。**

- **链接: [http://arxiv.org/pdf/2507.23540v1](http://arxiv.org/pdf/2507.23540v1)**

> **作者:** Yi Zhang; Erik Leo Haß; Kuo-Yi Chao; Nenad Petrovic; Yinglei Song; Chengdong Wu; Alois Knoll
>
> **摘要:** Autonomous driving systems face significant challenges in achieving human-like adaptability, robustness, and interpretability in complex, open-world environments. These challenges stem from fragmented architectures, limited generalization to novel scenarios, and insufficient semantic extraction from perception. To address these limitations, we propose a unified Perception-Language-Action (PLA) framework that integrates multi-sensor fusion (cameras, LiDAR, radar) with a large language model (LLM)-augmented Vision-Language-Action (VLA) architecture, specifically a GPT-4.1-powered reasoning core. This framework unifies low-level sensory processing with high-level contextual reasoning, tightly coupling perception with natural language-based semantic understanding and decision-making to enable context-aware, explainable, and safety-bounded autonomous driving. Evaluations on an urban intersection scenario with a construction zone demonstrate superior performance in trajectory tracking, speed prediction, and adaptive planning. The results highlight the potential of language-augmented cognitive frameworks for advancing the safety, interpretability, and scalability of autonomous driving systems.
>
---
#### [new 098] DepMicroDiff: Diffusion-Based Dependency-Aware Multimodal Imputation for Microbiome Data
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于微生物组数据填补任务，旨在解决数据稀疏性和噪声导致的填补不准确问题。作者提出了DepMicroDiff方法，结合扩散模型与依赖感知Transformer，并利用预训练和患者元数据编码提升填补效果，在多个癌症数据集中表现出色。**

- **链接: [http://arxiv.org/pdf/2507.23676v1](http://arxiv.org/pdf/2507.23676v1)**

> **作者:** Rabeya Tus Sadia; Qiang Cheng
>
> **摘要:** Microbiome data analysis is essential for understanding host health and disease, yet its inherent sparsity and noise pose major challenges for accurate imputation, hindering downstream tasks such as biomarker discovery. Existing imputation methods, including recent diffusion-based models, often fail to capture the complex interdependencies between microbial taxa and overlook contextual metadata that can inform imputation. We introduce DepMicroDiff, a novel framework that combines diffusion-based generative modeling with a Dependency-Aware Transformer (DAT) to explicitly capture both mutual pairwise dependencies and autoregressive relationships. DepMicroDiff is further enhanced by VAE-based pretraining across diverse cancer datasets and conditioning on patient metadata encoded via a large language model (LLM). Experiments on TCGA microbiome datasets show that DepMicroDiff substantially outperforms state-of-the-art baselines, achieving higher Pearson correlation (up to 0.712), cosine similarity (up to 0.812), and lower RMSE and MAE across multiple cancer types, demonstrating its robustness and generalizability for microbiome imputation.
>
---
#### [new 099] Accessibility Scout: Personalized Accessibility Scans of Built Environments
- **分类: cs.HC; cs.AI; cs.CV; cs.MA**

- **简介: 该论文属于图像分析与辅助技术任务，旨在解决传统无障碍环境评估劳动强度大、个性化不足的问题。作者开发了基于大语言模型的系统Accessibility Scout，通过照片识别环境中的无障碍问题，并根据用户需求提供个性化评估，结合了技术准确性与人性化反馈。**

- **链接: [http://arxiv.org/pdf/2507.23190v1](http://arxiv.org/pdf/2507.23190v1)**

> **作者:** William Huang; Xia Su; Jon E. Froehlich; Yang Zhang
>
> **备注:** 18 pages, 16 figures. Presented at ACM UIST 2025
>
> **摘要:** Assessing the accessibility of unfamiliar built environments is critical for people with disabilities. However, manual assessments, performed by users or their personal health professionals, are laborious and unscalable, while automatic machine learning methods often neglect an individual user's unique needs. Recent advances in Large Language Models (LLMs) enable novel approaches to this problem, balancing personalization with scalability to enable more adaptive and context-aware assessments of accessibility. We present Accessibility Scout, an LLM-based accessibility scanning system that identifies accessibility concerns from photos of built environments. With use, Accessibility Scout becomes an increasingly capable "accessibility scout", tailoring accessibility scans to an individual's mobility level, preferences, and specific environmental interests through collaborative Human-AI assessments. We present findings from three studies: a formative study with six participants to inform the design of Accessibility Scout, a technical evaluation of 500 images of built environments, and a user study with 10 participants of varying mobility. Results from our technical evaluation and user study show that Accessibility Scout can generate personalized accessibility scans that extend beyond traditional ADA considerations. Finally, we conclude with a discussion on the implications of our work and future steps for building more scalable and personalized accessibility assessments of the physical world.
>
---
#### [new 100] EH-Benchmark Ophthalmic Hallucination Benchmark and Agent-Driven Top-Down Traceable Reasoning Workflow
- **分类: cs.CL; cs.CV; cs.MA**

- **简介: 该论文属于医学自然语言处理任务，旨在解决医疗大语言模型在眼科诊断中的幻觉问题。作者构建了EH-Benchmark评估框架，分类幻觉类型，并提出基于智能体的三阶段推理流程，以提升模型的准确性与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.22929v1](http://arxiv.org/pdf/2507.22929v1)**

> **作者:** Xiaoyu Pan; Yang Bai; Ke Zou; Yang Zhou; Jun Zhou; Huazhu Fu; Yih-Chung Tham; Yong Liu
>
> **备注:** 9 figures, 5 tables. submit/6621751
>
> **摘要:** Medical Large Language Models (MLLMs) play a crucial role in ophthalmic diagnosis, holding significant potential to address vision-threatening diseases. However, their accuracy is constrained by hallucinations stemming from limited ophthalmic knowledge, insufficient visual localization and reasoning capabilities, and a scarcity of multimodal ophthalmic data, which collectively impede precise lesion detection and disease diagnosis. Furthermore, existing medical benchmarks fail to effectively evaluate various types of hallucinations or provide actionable solutions to mitigate them. To address the above challenges, we introduce EH-Benchmark, a novel ophthalmology benchmark designed to evaluate hallucinations in MLLMs. We categorize MLLMs' hallucinations based on specific tasks and error types into two primary classes: Visual Understanding and Logical Composition, each comprising multiple subclasses. Given that MLLMs predominantly rely on language-based reasoning rather than visual processing, we propose an agent-centric, three-phase framework, including the Knowledge-Level Retrieval stage, the Task-Level Case Studies stage, and the Result-Level Validation stage. Experimental results show that our multi-agent framework significantly mitigates both types of hallucinations, enhancing accuracy, interpretability, and reliability. Our project is available at https://github.com/ppxy1/EH-Benchmark.
>
---
#### [new 101] Automated Label Placement on Maps via Large Language Models
- **分类: cs.HC; cs.CV; cs.LG**

- **简介: 该论文属于自动地图标注任务，旨在解决标签放置依赖人工、难以扩展的问题。作者提出新方法，将标签放置视为数据编辑问题，利用大语言模型结合检索增强生成，实现上下文感知的空间标注。构建了首个真实地图标注数据集MAPLE，并验证了模型在不同地标类型上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22952v1](http://arxiv.org/pdf/2507.22952v1)**

> **作者:** Harry Shomer; Jiejun Xu
>
> **备注:** Workshop on AI for Data Editing (AI4DE) at KDD 2025
>
> **摘要:** Label placement is a critical aspect of map design, serving as a form of spatial annotation that directly impacts clarity and interpretability. Despite its importance, label placement remains largely manual and difficult to scale, as existing automated systems struggle to integrate cartographic conventions, adapt to context, or interpret labeling instructions. In this work, we introduce a new paradigm for automatic label placement (ALP) that formulates the task as a data editing problem and leverages large language models (LLMs) for context-aware spatial annotation. To support this direction, we curate MAPLE, the first known benchmarking dataset for evaluating ALP on real-world maps, encompassing diverse landmark types and label placement annotations from open-source data. Our method retrieves labeling guidelines relevant to each landmark type leveraging retrieval-augmented generation (RAG), integrates them into prompts, and employs instruction-tuned LLMs to generate ideal label coordinates. We evaluate four open-source LLMs on MAPLE, analyzing both overall performance and generalization across different types of landmarks. This includes both zero-shot and instruction-tuned performance. Our results demonstrate that LLMs, when guided by structured prompts and domain-specific retrieval, can learn to perform accurate spatial edits, aligning the generated outputs with expert cartographic standards. Overall, our work presents a scalable framework for AI-assisted map finishing and demonstrates the potential of foundation models in structured data editing tasks. The code and data can be found at https://github.com/HarryShomer/MAPLE.
>
---
#### [new 102] MPCC: A Novel Benchmark for Multimodal Planning with Complex Constraints in Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.CV; I.2.8; I.2.10**

- **简介: 该论文提出MPCC基准，评估多模态大语言模型（MLLM）在复杂约束下的多模态规划能力。任务是多模态规划，解决现有基准无法评估真实场景规划能力和缺乏跨模态约束的问题。工作包括设计含复杂约束的现实任务，并验证MLLM在多约束场景下的表现与挑战。**

- **链接: [http://arxiv.org/pdf/2507.23382v1](http://arxiv.org/pdf/2507.23382v1)**

> **作者:** Yiyan Ji; Haoran Chen; Qiguang Chen; Chengyue Wu; Libo Qin; Wanxiang Che
>
> **备注:** Accepted to ACM Multimedia 2025
>
> **摘要:** Multimodal planning capabilities refer to the ability to predict, reason, and design steps for task execution with multimodal context, which is essential for complex reasoning and decision-making across multiple steps. However, current benchmarks face two key challenges: (1) they cannot directly assess multimodal real-world planning capabilities, and (2) they lack constraints or implicit constraints across modalities. To address these issues, we introduce Multimodal Planning with Complex Constraints (MPCC), the first benchmark to systematically evaluate MLLMs' ability to handle multimodal constraints in planning. To address the first challenge, MPCC focuses on three real-world tasks: Flight Planning, Calendar Planning, and Meeting Planning. To solve the second challenge, we introduce complex constraints (e.g. budget, temporal, and spatial) in these tasks, with graded difficulty levels (EASY, MEDIUM, HARD) to separate constraint complexity from search space expansion. Experiments on 13 advanced MLLMs reveal significant challenges: closed-source models achieve only 21.3% feasible plans, while open-source models average below 11%. Additionally, we observe that MLLMs are highly sensitive to constraint complexity and that traditional multimodal prompting strategies fail in multi-constraint scenarios. Our work formalizes multimodal constraints in planning, provides a rigorous evaluation framework, and highlights the need for advancements in constraint-aware reasoning for real-world MLLM applications.
>
---
#### [new 103] Pixel Embedding Method for Tubular Neurite Segmentation
- **分类: eess.IV; cs.CV; q-bio.NC**

- **简介: 该论文属于神经图像分割任务，旨在解决神经元形态复杂和纤维遮挡导致的分割难题。论文提出了一种像素嵌入方法，结合新设计的损失函数，实现了神经元结构的端到端分割与重建，并提出了新的拓扑评估指标，有效提升了分割准确性。**

- **链接: [http://arxiv.org/pdf/2507.23359v1](http://arxiv.org/pdf/2507.23359v1)**

> **作者:** Huayu Fu; Jiamin Li; Haozhi Qu; Xiaolin Hu; Zengcai Guo
>
> **摘要:** Automatic segmentation of neuronal topology is critical for handling large scale neuroimaging data, as it can greatly accelerate neuron annotation and analysis. However, the intricate morphology of neuronal branches and the occlusions among fibers pose significant challenges for deep learning based segmentation. To address these issues, we propose an improved framework: First, we introduce a deep network that outputs pixel level embedding vectors and design a corresponding loss function, enabling the learned features to effectively distinguish different neuronal connections within occluded regions. Second, building on this model, we develop an end to end pipeline that directly maps raw neuronal images to SWC formatted neuron structure trees. Finally, recognizing that existing evaluation metrics fail to fully capture segmentation accuracy, we propose a novel topological assessment metric to more appropriately quantify the quality of neuron segmentation and reconstruction. Experiments on our fMOST imaging dataset demonstrate that, compared to several classical methods, our approach significantly reduces the error rate in neuronal topology reconstruction.
>
---
#### [new 104] EMedNeXt: An Enhanced Brain Tumor Segmentation Framework for Sub-Saharan Africa using MedNeXt V2 with Deep Supervision
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决非洲资源匮乏地区脑肿瘤分割困难的问题。作者提出了EMedNeXt框架，基于MedNeXt V2并引入深监督和优化后处理，提升低质量MRI图像的肿瘤分割效果，以应对数据量小、图像质量和专家资源不足的挑战。**

- **链接: [http://arxiv.org/pdf/2507.23256v1](http://arxiv.org/pdf/2507.23256v1)**

> **作者:** Ahmed Jaheen; Abdelrahman Elsayed; Damir Kim; Daniil Tikhonov; Matheus Scatolin; Mohor Banerjee; Qiankun Ji; Mostafa Salem; Hu Wang; Sarim Hashmi; Mohammad Yaqub
>
> **备注:** Submitted to the BraTS-Lighthouse 2025 Challenge (MICCAI 2025)
>
> **摘要:** Brain cancer affects millions worldwide, and in nearly every clinical setting, doctors rely on magnetic resonance imaging (MRI) to diagnose and monitor gliomas. However, the current standard for tumor quantification through manual segmentation of multi-parametric MRI is time-consuming, requires expert radiologists, and is often infeasible in under-resourced healthcare systems. This problem is especially pronounced in low-income regions, where MRI scanners are of lower quality and radiology expertise is scarce, leading to incorrect segmentation and quantification. In addition, the number of acquired MRI scans in Africa is typically small. To address these challenges, the BraTS-Lighthouse 2025 Challenge focuses on robust tumor segmentation in sub-Saharan Africa (SSA), where resource constraints and image quality degradation introduce significant shifts. In this study, we present EMedNeXt -- an enhanced brain tumor segmentation framework based on MedNeXt V2 with deep supervision and optimized post-processing pipelines tailored for SSA. EMedNeXt introduces three key contributions: a larger region of interest, an improved nnU-Net v2-based architectural skeleton, and a robust model ensembling system. Evaluated on the hidden validation set, our solution achieved an average LesionWise DSC of 0.897 with an average LesionWise NSD of 0.541 and 0.84 at a tolerance of 0.5 mm and 1.0 mm, respectively.
>
---
#### [new 105] Investigating the Invertibility of Multimodal Latent Spaces: Limitations of Optimization-Based Methods
- **分类: cs.LG; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于多模态AI模型任务，旨在探索多模态潜在空间的可逆性问题。作者提出基于优化的方法，尝试从输出反推输入，发现在文本-图像和文本-音频模型中，尽管优化能实现文本对齐，但反向映射在感知质量和语义解释上表现混乱，表明当前多模态潜在空间不支持稳健的可逆映射。**

- **链接: [http://arxiv.org/pdf/2507.23010v1](http://arxiv.org/pdf/2507.23010v1)**

> **作者:** Siwoo Park
>
> **摘要:** This paper investigates the inverse capabilities and broader utility of multimodal latent spaces within task-specific AI (Artificial Intelligence) models. While these models excel at their designed forward tasks (e.g., text-to-image generation, audio-to-text transcription), their potential for inverse mappings remains largely unexplored. We propose an optimization-based framework to infer input characteristics from desired outputs, applying it bidirectionally across Text-Image (BLIP, Flux.1-dev) and Text-Audio (Whisper-Large-V3, Chatterbox-TTS) modalities. Our central hypothesis posits that while optimization can guide models towards inverse tasks, their multimodal latent spaces will not consistently support semantically meaningful and perceptually coherent inverse mappings. Experimental results consistently validate this hypothesis. We demonstrate that while optimization can force models to produce outputs that align textually with targets (e.g., a text-to-image model generating an image that an image captioning model describes correctly, or an ASR model transcribing optimized audio accurately), the perceptual quality of these inversions is chaotic and incoherent. Furthermore, when attempting to infer the original semantic input from generative models, the reconstructed latent space embeddings frequently lack semantic interpretability, aligning with nonsensical vocabulary tokens. These findings highlight a critical limitation. multimodal latent spaces, primarily optimized for specific forward tasks, do not inherently possess the structure required for robust and interpretable inverse mappings. Our work underscores the need for further research into developing truly semantically rich and invertible multimodal latent spaces.
>
---
#### [new 106] FuseTen: A Generative Model for Daily 10 m Land Surface Temperature Estimation from Spatio-Temporal Satellite Observations
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于遥感数据融合任务，旨在解决卫星观测中空间与时间分辨率之间的权衡问题。作者提出FuseTen，一种基于生成模型的方法，融合Sentinel-2、Landsat 8和Terra MODIS数据，生成每日10米分辨率的地表温度（LST）估计，提升了精度与视觉质量。**

- **链接: [http://arxiv.org/pdf/2507.23154v1](http://arxiv.org/pdf/2507.23154v1)**

> **作者:** Sofiane Bouaziz; Adel Hafiane; Raphael Canals; Rachid Nedjai
>
> **备注:** Accepted in the 2025 International Conference on Machine Intelligence for GeoAnalytics and Remote Sensing (MIGARS)
>
> **摘要:** Urban heatwaves, droughts, and land degradation are pressing and growing challenges in the context of climate change. A valuable approach to studying them requires accurate spatio-temporal information on land surface conditions. One of the most important variables for assessing and understanding these phenomena is Land Surface Temperature (LST), which is derived from satellites and provides essential information about the thermal state of the Earth's surface. However, satellite platforms inherently face a trade-off between spatial and temporal resolutions. To bridge this gap, we propose FuseTen, a novel generative framework that produces daily LST observations at a fine 10 m spatial resolution by fusing spatio-temporal observations derived from Sentinel-2, Landsat 8, and Terra MODIS. FuseTen employs a generative architecture trained using an averaging-based supervision strategy grounded in physical principles. It incorporates attention and normalization modules within the fusion process and uses a PatchGAN discriminator to enforce realism. Experiments across multiple dates show that FuseTen outperforms linear baselines, with an average 32.06% improvement in quantitative metrics and 31.42% in visual fidelity. To the best of our knowledge, this is the first non-linear method to generate daily LST estimates at such fine spatial resolution.
>
---
#### [new 107] MRpro - open PyTorch-based MR reconstruction and processing package
- **分类: eess.IV; cs.CV; physics.med-ph**

- **简介: 该论文介绍了MRpro，一个基于PyTorch的开源磁共振（MR）图像重建与处理工具包。论文属于医学影像处理任务，旨在解决MR图像重建中的数据格式不统一、算法实现复杂及可重复性差等问题。作者提供了统一的数据结构、优化算法、深度学习模块，并支持多种MR成像应用，以促进协作开发与研究可重复性。**

- **链接: [http://arxiv.org/pdf/2507.23129v1](http://arxiv.org/pdf/2507.23129v1)**

> **作者:** Felix Frederik Zimmermann; Patrick Schuenke; Christoph S. Aigner; Bill A. Bernhardt; Mara Guastini; Johannes Hammacher; Noah Jaitner; Andreas Kofler; Leonid Lunin; Stefan Martin; Catarina Redshaw Kranich; Jakob Schattenfroh; David Schote; Yanglei Wu; Christoph Kolbitsch
>
> **备注:** Submitted to Magnetic Resonance in Medicine
>
> **摘要:** We introduce MRpro, an open-source image reconstruction package built upon PyTorch and open data formats. The framework comprises three main areas. First, it provides unified data structures for the consistent manipulation of MR datasets and their associated metadata (e.g., k-space trajectories). Second, it offers a library of composable operators, proximable functionals, and optimization algorithms, including a unified Fourier operator for all common trajectories and an extended phase graph simulation for quantitative MR. These components are used to create ready-to-use implementations of key reconstruction algorithms. Third, for deep learning, MRpro includes essential building blocks such as data consistency layers, differentiable optimization layers, and state-of-the-art backbone networks and integrates public datasets to facilitate reproducibility. MRpro is developed as a collaborative project supported by automated quality control. We demonstrate the versatility of MRpro across multiple applications, including Cartesian, radial, and spiral acquisitions; motion-corrected reconstruction; cardiac MR fingerprinting; learned spatially adaptive regularization weights; model-based learned image reconstruction and quantitative parameter estimation. MRpro offers an extensible framework for MR image reconstruction. With reproducibility and maintainability at its core, it facilitates collaborative development and provides a foundation for future MR imaging research.
>
---
#### [new 108] Smart Video Capsule Endoscopy: Raw Image-Based Localization for Enhanced GI Tract Investigation
- **分类: eess.IV; cs.AR; cs.CV**

- **简介: 该论文属于医疗AI任务，旨在解决低功耗设备上实现高效图像分类的问题。针对胶囊内镜在小肠检查中受限于电池寿命的问题，提出直接在Bayer图像上进行轻量级CNN分类（仅63,000参数）并结合维特比解码提升定位精度。在定制的PULPissimo芯片上实现，每张图像仅耗能5.31μJ，相比传统方法节能89.9%。**

- **链接: [http://arxiv.org/pdf/2507.23398v1](http://arxiv.org/pdf/2507.23398v1)**

> **作者:** Oliver Bause; Julia Werner; Paul Palomero Bernardo; Oliver Bringmann
>
> **备注:** Accepted at the 32nd International Conference on Neural Information Processing - ICONIP 2025
>
> **摘要:** For many real-world applications involving low-power sensor edge devices deep neural networks used for image classification might not be suitable. This is due to their typically large model size and require- ment of operations often exceeding the capabilities of such resource lim- ited devices. Furthermore, camera sensors usually capture images with a Bayer color filter applied, which are subsequently converted to RGB images that are commonly used for neural network training. However, on resource-constrained devices, such conversions demands their share of energy and optimally should be skipped if possible. This work ad- dresses the need for hardware-suitable AI targeting sensor edge devices by means of the Video Capsule Endoscopy, an important medical proce- dure for the investigation of the small intestine, which is strongly limited by its battery lifetime. Accurate organ classification is performed with a final accuracy of 93.06% evaluated directly on Bayer images involv- ing a CNN with only 63,000 parameters and time-series analysis in the form of Viterbi decoding. Finally, the process of capturing images with a camera and raw image processing is demonstrated with a customized PULPissimo System-on-Chip with a RISC-V core and an ultra-low power hardware accelerator providing an energy-efficient AI-based image clas- sification approach requiring just 5.31 {\mu}J per image. As a result, it is possible to save an average of 89.9% of energy before entering the small intestine compared to classic video capsules.
>
---
#### [new 109] iLearnRobot: An Interactive Learning-Based Multi-Modal Robot with Continuous Improvement
- **分类: cs.HC; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于机器人交互学习任务，旨在解决机器人部署后难以适应新场景的问题。论文提出iLearnRobot系统，基于多模态大语言模型，通过与非专家用户的自然对话进行持续学习，结合问题链和双模态检索模块，提升机器人适应性和性能，实现无缝用户体验。**

- **链接: [http://arxiv.org/pdf/2507.22896v1](http://arxiv.org/pdf/2507.22896v1)**

> **作者:** Kohou Wang; ZhaoXiang Liu; Lin Bai; Kun Fan; Xiang Liu; Huan Hu; Kai Wang; Shiguo Lian
>
> **备注:** 17 pages, 12 figures
>
> **摘要:** It is crucial that robots' performance can be improved after deployment, as they are inherently likely to encounter novel scenarios never seen before. This paper presents an innovative solution: an interactive learning-based robot system powered by a Multi-modal Large Language Model(MLLM). A key feature of our system is its ability to learn from natural dialogues with non-expert users. We also propose chain of question to clarify the exact intent of the question before providing an answer and dual-modality retrieval modules to leverage these interaction events to avoid repeating same mistakes, ensuring a seamless user experience before model updates, which is in contrast to current mainstream MLLM-based robotic systems. Our system marks a novel approach in robotics by integrating interactive learning, paving the way for superior adaptability and performance in diverse environments. We demonstrate the effectiveness and improvement of our method through experiments, both quantitively and qualitatively.
>
---
#### [new 110] JPEG Processing Neural Operator for Backward-Compatible Coding
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决现有JPEG压缩算法在保持图像质量与兼容性方面的不足。作者提出了JPNeO，一种兼容JPEG格式的新型压缩算法，通过在编码和解码阶段引入神经算子，提升色度保留能力和重建质量，同时减少内存和参数开销。**

- **链接: [http://arxiv.org/pdf/2507.23521v1](http://arxiv.org/pdf/2507.23521v1)**

> **作者:** Woo Kyoung Han; Yongjun Lee; Byeonghun Lee; Sang Hyun Park; Sunghoon Im; Kyong Hwan Jin
>
> **摘要:** Despite significant advances in learning-based lossy compression algorithms, standardizing codecs remains a critical challenge. In this paper, we present the JPEG Processing Neural Operator (JPNeO), a next-generation JPEG algorithm that maintains full backward compatibility with the current JPEG format. Our JPNeO improves chroma component preservation and enhances reconstruction fidelity compared to existing artifact removal methods by incorporating neural operators in both the encoding and decoding stages. JPNeO achieves practical benefits in terms of reduced memory usage and parameter count. We further validate our hypothesis about the existence of a space with high mutual information through empirical evidence. In summary, the JPNeO functions as a high-performance out-of-the-box image compression pipeline without changing source coding's protocol. Our source code is available at https://github.com/WooKyoungHan/JPNeO.
>
---
#### [new 111] Continual Learning with Synthetic Boundary Experience Blending
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于持续学习任务，旨在缓解模型在顺序学习多任务时的灾难性遗忘问题。通过引入合成边界数据作为隐式正则化，提升决策边界稳定性。论文提出“经验融合”框架，结合存储样本与合成数据进行端到端训练，实验表明其在多个数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.23534v1](http://arxiv.org/pdf/2507.23534v1)**

> **作者:** Chih-Fan Hsu; Ming-Ching Chang; Wei-Chao Chen
>
> **摘要:** Continual learning (CL) aims to address catastrophic forgetting in models trained sequentially on multiple tasks. While experience replay has shown promise, its effectiveness is often limited by the sparse distribution of stored key samples, leading to overly simplified decision boundaries. We hypothesize that introducing synthetic data near the decision boundary (Synthetic Boundary Data, or SBD) during training serves as an implicit regularizer, improving boundary stability and mitigating forgetting. To validate this hypothesis, we propose a novel training framework, {\bf Experience Blending}, which integrates knowledge from both stored key samples and synthetic, boundary-adjacent data. Experience blending consists of two core components: (1) a multivariate Differential Privacy (DP) noise mechanism that injects batch-wise noise into low-dimensional feature representations, generating SBD; and (2) an end-to-end training strategy that jointly leverages both stored key samples and SBD. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny ImageNet demonstrate that our method outperforms nine CL baselines, achieving accuracy improvements of 10%, 6%, and 13%, respectively.
>
---
#### [new 112] Causal Identification of Sufficient, Contrastive and Complete Feature Sets in Image Classification
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于图像分类解释任务，旨在解决现有解释方法缺乏形式化定义与可计算性问题。工作包括：提出因果解释及其形式化性质，引入对比性与完全性因果解释，实现无需模型内部信息的高效黑盒算法，验证其在ResNet50上的有效性与效率。**

- **链接: [http://arxiv.org/pdf/2507.23497v1](http://arxiv.org/pdf/2507.23497v1)**

> **作者:** David A Kelly; Hana Chockler
>
> **备注:** 13 pages, 13 figures, appendix included
>
> **摘要:** Existing algorithms for explaining the outputs of image classifiers are based on a variety of approaches and produce explanations that lack formal rigor. On the other hand, logic-based explanations are formally and rigorously defined but their computability relies on strict assumptions about the model that do not hold on image classifiers. In this paper, we show that causal explanations, in addition to being formally and rigorously defined, enjoy the same formal properties as logic-based ones, while still lending themselves to black-box algorithms and being a natural fit for image classifiers. We prove formal properties of causal explanations and introduce contrastive causal explanations for image classifiers. Moreover, we augment the definition of explanation with confidence awareness and introduce complete causal explanations: explanations that are classified with exactly the same confidence as the original image. We implement our definitions, and our experimental results demonstrate that different models have different patterns of sufficiency, contrastiveness, and completeness. Our algorithms are efficiently computable, taking on average 6s per image on a ResNet50 model to compute all types of explanations, and are totally black-box, needing no knowledge of the model, no access to model internals, no access to gradient, nor requiring any properties, such as monotonicity, of the model.
>
---
#### [new 113] Towards Field-Ready AI-based Malaria Diagnosis: A Continual Learning Approach
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决疟疾诊断中AI模型跨站点泛化能力差的问题。作者采用持续学习方法，通过多站点血液涂片数据集评估四种策略，提升模型适应新环境的能力，从而增强现场部署的可行性。**

- **链接: [http://arxiv.org/pdf/2507.23648v1](http://arxiv.org/pdf/2507.23648v1)**

> **作者:** Louise Guillon; Soheib Biga; Yendoube E. Kantchire; Mouhamadou Lamine Sane; Grégoire Pasquier; Kossi Yakpa; Stéphane E. Sossou; Marc Thellier; Laurent Bonnardot; Laurence Lachaud; Renaud Piarroux; Ameyo M. Dorkenoo
>
> **备注:** MICCAI 2025 AMAI Workshop, Accepted, Submitted Manuscript Version
>
> **摘要:** Malaria remains a major global health challenge, particularly in low-resource settings where access to expert microscopy may be limited. Deep learning-based computer-aided diagnosis (CAD) systems have been developed and demonstrate promising performance on thin blood smear images. However, their clinical deployment may be hindered by limited generalization across sites with varying conditions. Yet very few practical solutions have been proposed. In this work, we investigate continual learning (CL) as a strategy to enhance the robustness of malaria CAD models to domain shifts. We frame the problem as a domain-incremental learning scenario, where a YOLO-based object detector must adapt to new acquisition sites while retaining performance on previously seen domains. We evaluate four CL strategies, two rehearsal-based and two regularization-based methods, on real-life conditions thanks to a multi-site clinical dataset of thin blood smear images. Our results suggest that CL, and rehearsal-based methods in particular, can significantly improve performance. These findings highlight the potential of continual learning to support the development of deployable, field-ready CAD tools for malaria.
>
---
#### [new 114] LesionGen: A Concept-Guided Diffusion Model for Dermatology Image Synthesis
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决皮肤疾病数据集不足的问题。作者提出LesionGen，基于扩散模型，利用专家标注的结构化描述生成多样化的皮肤病变图像，提升分类模型性能，尤其在少数子群体中表现更好。**

- **链接: [http://arxiv.org/pdf/2507.23001v1](http://arxiv.org/pdf/2507.23001v1)**

> **作者:** Jamil Fayyad; Nourhan Bayasi; Ziyang Yu; Homayoun Najjaran
>
> **备注:** Accepted at the MICCAI 2025 ISIC Workshop
>
> **摘要:** Deep learning models for skin disease classification require large, diverse, and well-annotated datasets. However, such resources are often limited due to privacy concerns, high annotation costs, and insufficient demographic representation. While text-to-image diffusion probabilistic models (T2I-DPMs) offer promise for medical data synthesis, their use in dermatology remains underexplored, largely due to the scarcity of rich textual descriptions in existing skin image datasets. In this work, we introduce LesionGen, a clinically informed T2I-DPM framework for dermatology image synthesis. Unlike prior methods that rely on simplistic disease labels, LesionGen is trained on structured, concept-rich dermatological captions derived from expert annotations and pseudo-generated, concept-guided reports. By fine-tuning a pretrained diffusion model on these high-quality image-caption pairs, we enable the generation of realistic and diverse skin lesion images conditioned on meaningful dermatological descriptions. Our results demonstrate that models trained solely on our synthetic dataset achieve classification accuracy comparable to those trained on real images, with notable gains in worst-case subgroup performance. Code and data are available here.
>
---
#### [new 115] Rethink Domain Generalization in Heterogeneous Sequence MRI Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决跨中心和跨序列磁共振成像（MRI）数据中胰腺分割的域泛化问题。作者提出了PancreasDG数据集，包含563例多中心MRI扫描，并揭示了域泛化研究中的关键问题。他们还提出了一种半监督方法，在跨序列分割中显著优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.23110v1](http://arxiv.org/pdf/2507.23110v1)**

> **作者:** Zheyuan Zhang; Linkai Peng; Wanying Dou; Cuiling Sun; Halil Ertugrul Aktas; Andrea M. Bejar; Elif Keles; Gorkem Durak; Ulas Bagci
>
> **摘要:** Clinical magnetic-resonance (MR) protocols generate many T1 and T2 sequences whose appearance differs more than the acquisition sites that produce them. Existing domain-generalization benchmarks focus almost on cross-center shifts and overlook this dominant source of variability. Pancreas segmentation remains a major challenge in abdominal imaging: the gland is small, irregularly, surrounded by organs and fat, and often suffers from low T1 contrast. State-of-the-art deep networks that already achieve >90% Dice on the liver or kidneys still miss 20-30% of the pancreas. The organ is also systematically under-represented in public cross-domain benchmarks, despite its clinical importance in early cancer detection, surgery, and diabetes research. To close this gap, we present PancreasDG, a large-scale multi-center 3D MRI pancreas segmentation dataset for investigating domain generalization in medical imaging. The dataset comprises 563 MRI scans from six institutions, spanning both venous phase and out-of-phase sequences, enabling study of both cross-center and cross-sequence variations with pixel-accurate pancreas masks created by a double-blind, two-pass protocol. Through comprehensive analysis, we reveal three insights: (i) limited sampling introduces significant variance that may be mistaken for distribution shifts, (ii) cross-center performance correlates with source domain performance for identical sequences, and (iii) cross-sequence shifts require specialized solutions. We also propose a semi-supervised approach that leverages anatomical invariances, significantly outperforming state-of-the-art domain generalization techniques with 61.63% Dice score improvements and 87.00% on two test centers for cross-sequence segmentation. PancreasDG sets a new benchmark for domain generalization in medical imaging. Dataset, code, and models will be available at https://pancreasdg.netlify.app.
>
---
#### [new 116] Planning for Cooler Cities: A Multimodal AI Framework for Predicting and Mitigating Urban Heat Stress through Urban Landscape Transformation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于城市气候适应任务，旨在解决城市热应激问题。通过提出多模态深度学习框架GSM-UTCI，预测高分辨率热环境指数，并模拟城市景观改造的降温效果，为城市规划提供高效决策支持。**

- **链接: [http://arxiv.org/pdf/2507.23000v1](http://arxiv.org/pdf/2507.23000v1)**

> **作者:** Shengao Yi; Xiaojiang Li; Wei Tu; Tianhong Zhao
>
> **摘要:** As extreme heat events intensify due to climate change and urbanization, cities face increasing challenges in mitigating outdoor heat stress. While traditional physical models such as SOLWEIG and ENVI-met provide detailed assessments of human-perceived heat exposure, their computational demands limit scalability for city-wide planning. In this study, we propose GSM-UTCI, a multimodal deep learning framework designed to predict daytime average Universal Thermal Climate Index (UTCI) at 1-meter hyperlocal resolution. The model fuses surface morphology (nDSM), high-resolution land cover data, and hourly meteorological conditions using a feature-wise linear modulation (FiLM) architecture that dynamically conditions spatial features on atmospheric context. Trained on SOLWEIG-derived UTCI maps, GSM-UTCI achieves near-physical accuracy, with an R2 of 0.9151 and a mean absolute error (MAE) of 0.41{\deg}C, while reducing inference time from hours to under five minutes for an entire city. To demonstrate its planning relevance, we apply GSM-UTCI to simulate systematic landscape transformation scenarios in Philadelphia, replacing bare earth, grass, and impervious surfaces with tree canopy. Results show spatially heterogeneous but consistently strong cooling effects, with impervious-to-tree conversion producing the highest aggregated benefit (-4.18{\deg}C average change in UTCI across 270.7 km2). Tract-level bivariate analysis further reveals strong alignment between thermal reduction potential and land cover proportions. These findings underscore the utility of GSM-UTCI as a scalable, fine-grained decision support tool for urban climate adaptation, enabling scenario-based evaluation of greening strategies across diverse urban environments.
>
---
#### [new 117] Consensus-Driven Active Model Selection
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于模型选择任务，旨在解决在众多现有模型中选择最佳模型时标注验证数据代价高昂的问题。论文提出了CODA方法，通过建模分类器、类别与数据点间的关系，利用模型间共识与分歧引导标注过程，并使用贝叶斯推理更新对最佳模型的判断，显著减少标注工作量。**

- **链接: [http://arxiv.org/pdf/2507.23771v1](http://arxiv.org/pdf/2507.23771v1)**

> **作者:** Justin Kay; Grant Van Horn; Subhransu Maji; Daniel Sheldon; Sara Beery
>
> **备注:** ICCV 2025 Highlight. 16 pages, 8 figures
>
> **摘要:** The widespread availability of off-the-shelf machine learning models poses a challenge: which model, of the many available candidates, should be chosen for a given data analysis task? This question of model selection is traditionally answered by collecting and annotating a validation dataset -- a costly and time-intensive process. We propose a method for active model selection, using predictions from candidate models to prioritize the labeling of test data points that efficiently differentiate the best candidate. Our method, CODA, performs consensus-driven active model selection by modeling relationships between classifiers, categories, and data points within a probabilistic framework. The framework uses the consensus and disagreement between models in the candidate pool to guide the label acquisition process, and Bayesian inference to update beliefs about which model is best as more information is collected. We validate our approach by curating a collection of 26 benchmark tasks capturing a range of model selection scenarios. CODA outperforms existing methods for active model selection significantly, reducing the annotation effort required to discover the best model by upwards of 70% compared to the previous state-of-the-art. Code and data are available at https://github.com/justinkay/coda.
>
---
#### [new 118] XSpecMesh: Quality-Preserving Auto-Regressive Mesh Generation Acceleration via Multi-Head Speculative Decoding
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文属于3D网格生成任务，旨在解决自回归模型生成高质量网格时推理速度慢的问题。论文提出XSpecMesh方法，通过多头推测解码并行预测多个标记，并设计验证与重采样策略确保质量，结合知识蒸馏提升推测成功率，在不损失质量的前提下实现1.7倍加速。**

- **链接: [http://arxiv.org/pdf/2507.23777v1](http://arxiv.org/pdf/2507.23777v1)**

> **作者:** Dian Chen; Yansong Qu; Xinyang Li; Ming Li; Shengchuan Zhang
>
> **摘要:** Current auto-regressive models can generate high-quality, topologically precise meshes; however, they necessitate thousands-or even tens of thousands-of next-token predictions during inference, resulting in substantial latency. We introduce XSpecMesh, a quality-preserving acceleration method for auto-regressive mesh generation models. XSpecMesh employs a lightweight, multi-head speculative decoding scheme to predict multiple tokens in parallel within a single forward pass, thereby accelerating inference. We further propose a verification and resampling strategy: the backbone model verifies each predicted token and resamples any tokens that do not meet the quality criteria. In addition, we propose a distillation strategy that trains the lightweight decoding heads by distilling from the backbone model, encouraging their prediction distributions to align and improving the success rate of speculative predictions. Extensive experiments demonstrate that our method achieves a 1.7x speedup without sacrificing generation quality. Our code will be released.
>
---
#### [new 119] Topology Optimization in Medical Image Segmentation with Fast Euler Characteristic
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决现有深度学习方法在满足拓扑约束（如连续边界、闭合表面）方面的不足。论文提出基于欧拉特征快速优化拓扑结构的方法，通过计算拓扑误差并使用拓扑违例图进行分割结果修正，提升了分割的拓扑正确性，同时保持像素级精度。**

- **链接: [http://arxiv.org/pdf/2507.23763v1](http://arxiv.org/pdf/2507.23763v1)**

> **作者:** Liu Li; Qiang Ma; Cheng Ouyang; Johannes C. Paetzold; Daniel Rueckert; Bernhard Kainz
>
> **摘要:** Deep learning-based medical image segmentation techniques have shown promising results when evaluated based on conventional metrics such as the Dice score or Intersection-over-Union. However, these fully automatic methods often fail to meet clinically acceptable accuracy, especially when topological constraints should be observed, e.g., continuous boundaries or closed surfaces. In medical image segmentation, the correctness of a segmentation in terms of the required topological genus sometimes is even more important than the pixel-wise accuracy. Existing topology-aware approaches commonly estimate and constrain the topological structure via the concept of persistent homology (PH). However, these methods are difficult to implement for high dimensional data due to their polynomial computational complexity. To overcome this problem, we propose a novel and fast approach for topology-aware segmentation based on the Euler Characteristic ($\chi$). First, we propose a fast formulation for $\chi$ computation in both 2D and 3D. The scalar $\chi$ error between the prediction and ground-truth serves as the topological evaluation metric. Then we estimate the spatial topology correctness of any segmentation network via a so-called topological violation map, i.e., a detailed map that highlights regions with $\chi$ errors. Finally, the segmentation results from the arbitrary network are refined based on the topological violation maps by a topology-aware correction network. Our experiments are conducted on both 2D and 3D datasets and show that our method can significantly improve topological correctness while preserving pixel-wise segmentation accuracy.
>
---
#### [new 120] LLM-Based Identification of Infostealer Infection Vectors from Screenshots: The Case of Aurora
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文属于网络安全任务，旨在解决从感染截图中快速识别信息窃取程序（如Aurora）的感染途径问题。研究利用LLM（如gpt-4o-mini）分析截图，提取威胁指标，识别恶意URL、文件等，进而追踪攻击活动，提升威胁情报能力。**

- **链接: [http://arxiv.org/pdf/2507.23611v1](http://arxiv.org/pdf/2507.23611v1)**

> **作者:** Estelle Ruellan; Eric Clay; Nicholas Ascoli
>
> **摘要:** Infostealers exfiltrate credentials, session cookies, and sensitive data from infected systems. With over 29 million stealer logs reported in 2024, manual analysis and mitigation at scale are virtually unfeasible/unpractical. While most research focuses on proactive malware detection, a significant gap remains in leveraging reactive analysis of stealer logs and their associated artifacts. Specifically, infection artifacts such as screenshots, image captured at the point of compromise, are largely overlooked by the current literature. This paper introduces a novel approach leveraging Large Language Models (LLMs), more specifically gpt-4o-mini, to analyze infection screenshots to extract potential Indicators of Compromise (IoCs), map infection vectors, and track campaigns. Focusing on the Aurora infostealer, we demonstrate how LLMs can process screenshots to identify infection vectors, such as malicious URLs, installer files, and exploited software themes. Our method extracted 337 actionable URLs and 246 relevant files from 1000 screenshots, revealing key malware distribution methods and social engineering tactics. By correlating extracted filenames, URLs, and infection themes, we identified three distinct malware campaigns, demonstrating the potential of LLM-driven analysis for uncovering infection workflows and enhancing threat intelligence. By shifting malware analysis from traditional log-based detection methods to a reactive, artifact-driven approach that leverages infection screenshots, this research presents a scalable method for identifying infection vectors and enabling early intervention.
>
---
#### [new 121] H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决机器人操作中缺乏高质量演示数据的问题。论文提出H-RDT模型，利用人类操作数据预训练，再通过跨形态微调提升双手机器人的操作能力，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.23523v1](http://arxiv.org/pdf/2507.23523v1)**

> **作者:** Hongzhe Bi; Lingxuan Wu; Tianwei Lin; Hengkai Tan; Zhizhong Su; Hang Su; Jun Zhu
>
> **摘要:** Imitation learning for robotic manipulation faces a fundamental challenge: the scarcity of large-scale, high-quality robot demonstration data. Recent robotic foundation models often pre-train on cross-embodiment robot datasets to increase data scale, while they face significant limitations as the diverse morphologies and action spaces across different robot embodiments make unified training challenging. In this paper, we present H-RDT (Human to Robotics Diffusion Transformer), a novel approach that leverages human manipulation data to enhance robot manipulation capabilities. Our key insight is that large-scale egocentric human manipulation videos with paired 3D hand pose annotations provide rich behavioral priors that capture natural manipulation strategies and can benefit robotic policy learning. We introduce a two-stage training paradigm: (1) pre-training on large-scale egocentric human manipulation data, and (2) cross-embodiment fine-tuning on robot-specific data with modular action encoders and decoders. Built on a diffusion transformer architecture with 2B parameters, H-RDT uses flow matching to model complex action distributions. Extensive evaluations encompassing both simulation and real-world experiments, single-task and multitask scenarios, as well as few-shot learning and robustness assessments, demonstrate that H-RDT outperforms training from scratch and existing state-of-the-art methods, including Pi0 and RDT, achieving significant improvements of 13.9% and 40.5% over training from scratch in simulation and real-world experiments, respectively. The results validate our core hypothesis that human manipulation data can serve as a powerful foundation for learning bimanual robotic manipulation policies.
>
---
#### [new 122] Learning Arbitrary-Scale RAW Image Downscaling with Wavelet-based Recurrent Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像处理任务，旨在解决RAW图像在任意尺度下采时细节模糊和伪影问题。提出了基于小波变换的循环重建框架，包含低频下采模组和高频预测模组，并设计能量最大化损失函数。构建了Real-NIRD数据集，结合公开数据集实现任意尺度下采基准测试，实验表明方法优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.23219v1](http://arxiv.org/pdf/2507.23219v1)**

> **作者:** Yang Ren; Hai Jiang; Wei Li; Menglong Yang; Heng Zhang; Zehua Sheng; Qingsheng Ye; Shuaicheng Liu
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Image downscaling is critical for efficient storage and transmission of high-resolution (HR) images. Existing learning-based methods focus on performing downscaling within the sRGB domain, which typically suffers from blurred details and unexpected artifacts. RAW images, with their unprocessed photonic information, offer greater flexibility but lack specialized downscaling frameworks. In this paper, we propose a wavelet-based recurrent reconstruction framework that leverages the information lossless attribute of wavelet transformation to fulfill the arbitrary-scale RAW image downscaling in a coarse-to-fine manner, in which the Low-Frequency Arbitrary-Scale Downscaling Module (LASDM) and the High-Frequency Prediction Module (HFPM) are proposed to preserve structural and textural integrity of the reconstructed low-resolution (LR) RAW images, alongside an energy-maximization loss to align high-frequency energy between HR and LR domain. Furthermore, we introduce the Realistic Non-Integer RAW Downscaling (Real-NIRD) dataset, featuring a non-integer downscaling factor of 1.3$\times$, and incorporate it with publicly available datasets with integer factors (2$\times$, 3$\times$, 4$\times$) for comprehensive benchmarking arbitrary-scale image downscaling purposes. Extensive experiments demonstrate that our method outperforms existing state-of-the-art competitors both quantitatively and visually. The code and dataset will be released at https://github.com/RenYangSCU/ASRD.
>
---
#### [new 123] GSFusion:Globally Optimized LiDAR-Inertial-Visual Mapping for Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于三维重建与SLAM任务，旨在解决3D高斯点绘制（3DGS）在计算负载高、弱纹理/光照环境失效及短程操作的局限性。作者提出GSFusion，融合LiDAR、惯性与视觉信息，通过全局位姿优化、像素感知初始化和有界Sigmoid约束，实现高精度、高效的实时地图构建。**

- **链接: [http://arxiv.org/pdf/2507.23273v1](http://arxiv.org/pdf/2507.23273v1)**

> **作者:** Jaeseok Park; Chanoh Park; Minsu Kim; Soohwan Kim
>
> **摘要:** While 3D Gaussian Splatting (3DGS) has revolutionized photorealistic mapping, conventional approaches based on camera sensor, even RGB-D, suffer from fundamental limitations such as high computational load, failure in environments with poor texture or illumination, and short operational ranges. LiDAR emerges as a robust alternative, but its integration with 3DGS introduces new challenges, such as the need for exceptional global alignment for photorealistic quality and prolonged optimization times caused by sparse data. To address these challenges, we propose GSFusion, an online LiDAR-Inertial-Visual mapping system that ensures high-precision map consistency through a surfel-to-surfel constraint in the global pose-graph optimization. To handle sparse data, our system employs a pixel-aware Gaussian initialization strategy for efficient representation and a bounded sigmoid constraint to prevent uncontrolled Gaussian growth. Experiments on public and our datasets demonstrate our system outperforms existing 3DGS SLAM systems in terms of rendering quality and map-building efficiency.
>
---
#### [new 124] Noise-Coded Illumination for Forensic and Photometric Video Analysis
- **分类: cs.GR; cs.CR; cs.CV**

- **简介: 论文提出了一种通过在场景照明中编码微弱噪声信号来对抗视频伪造的技术。属于视频取证任务，旨在解决伪造视频难以鉴别问题。工作是利用编码照明生成时间水印，使伪造视频面临更大挑战。**

- **链接: [http://arxiv.org/pdf/2507.23002v1](http://arxiv.org/pdf/2507.23002v1)**

> **作者:** Peter F. Michael; Zekun Hao; Serge Belongie; Abe Davis
>
> **备注:** ACM Transactions on Graphics (2025), presented at SIGGRAPH 2025
>
> **摘要:** The proliferation of advanced tools for manipulating video has led to an arms race, pitting those who wish to sow disinformation against those who want to detect and expose it. Unfortunately, time favors the ill-intentioned in this race, with fake videos growing increasingly difficult to distinguish from real ones. At the root of this trend is a fundamental advantage held by those manipulating media: equal access to a distribution of what we consider authentic (i.e., "natural") video. In this paper, we show how coding very subtle, noise-like modulations into the illumination of a scene can help combat this advantage by creating an information asymmetry that favors verification. Our approach effectively adds a temporal watermark to any video recorded under coded illumination. However, rather than encoding a specific message, this watermark encodes an image of the unmanipulated scene as it would appear lit only by the coded illumination. We show that even when an adversary knows that our technique is being used, creating a plausible coded fake video amounts to solving a second, more difficult version of the original adversarial content creation problem at an information disadvantage. This is a promising avenue for protecting high-stakes settings like public events and interviews, where the content on display is a likely target for manipulation, and while the illumination can be controlled, the cameras capturing video cannot.
>
---
#### [new 125] User Experience Estimation in Human-Robot Interaction Via Multi-Instance Learning of Multimodal Social Signals
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文属于人机交互中的用户体验（UX）评估任务，旨在解决如何准确估计用户在与机器人互动中的多维度体验问题。论文通过融合多模态社交信号（如面部表情和语音），构建了一个基于Transformer的模型，并采用多实例学习框架捕捉交互的短期与长期模式，提升了UX估计的准确性，实验表明其效果优于人类评估者。**

- **链接: [http://arxiv.org/pdf/2507.23544v1](http://arxiv.org/pdf/2507.23544v1)**

> **作者:** Ryo Miyoshi; Yuki Okafuji; Takuya Iwamoto; Junya Nakanishi; Jun Baba
>
> **备注:** This paper has been accepted for presentation at IEEE/RSJ International Conference on Intelligent Robots and Systems 2025 (IROS 2025)
>
> **摘要:** In recent years, the demand for social robots has grown, requiring them to adapt their behaviors based on users' states. Accurately assessing user experience (UX) in human-robot interaction (HRI) is crucial for achieving this adaptability. UX is a multi-faceted measure encompassing aspects such as sentiment and engagement, yet existing methods often focus on these individually. This study proposes a UX estimation method for HRI by leveraging multimodal social signals. We construct a UX dataset and develop a Transformer-based model that utilizes facial expressions and voice for estimation. Unlike conventional models that rely on momentary observations, our approach captures both short- and long-term interaction patterns using a multi-instance learning framework. This enables the model to capture temporal dynamics in UX, providing a more holistic representation. Experimental results demonstrate that our method outperforms third-party human evaluators in UX estimation.
>
---
## 更新

#### [replaced 001] Sparse Reconstruction of Optical Doppler Tomography with Alternative State Space Model and Attention
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2404.17484v3](http://arxiv.org/pdf/2404.17484v3)**

> **作者:** Zhenghong Li; Jiaxiang Ren; Wensheng Cheng; Yanzuo Liu; Congwu Du; Yingtian Pan; Haibin Ling
>
> **备注:** MICCAI25, 10 pages, 3 figures
>
> **摘要:** Optical coherence Doppler tomography (ODT) is an emerging blood flow imaging technique. The fundamental unit of ODT is the 1D depth-resolved trace named raw A-scans (or A-line). A 2D ODT image (B-scan) is formed by reconstructing a cross-sectional flow image via Doppler phase-subtraction of raw A-scans along B-line. To obtain a high-fidelity B-scan, densely sampled A-scans are required currently, leading to prolonged scanning time and increased storage demands. Addressing this issue, we propose a novel sparse ODT reconstruction framework with an Alternative State Space Attention Network (ASSAN) that effectively reduces raw A-scans needed. Inspired by the distinct distributions of information along A-line and B-line, ASSAN applies 1D State Space Model (SSM) to each A-line to learn the intra-A-scan representation, while using 1D gated self-attention along B-line to capture the inter-A-scan features. In addition, an effective feedforward network based on sequential 1D convolutions along different axes is employed to enhance the local feature. In validation experiments on real animal data, ASSAN shows clear effectiveness in the reconstruction in comparison with state-of-the-art reconstruction methods.
>
---
#### [replaced 002] VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22607v2](http://arxiv.org/pdf/2507.22607v2)**

> **作者:** Ruifeng Yuan; Chenghao Xiao; Sicong Leng; Jianyu Wang; Long Li; Weiwen Xu; Hou Pong Chan; Deli Zhao; Tingyang Xu; Zhongyu Wei; Hao Zhang; Yu Rong
>
> **备注:** 21 pages, 5 figures, 6 tables. Work in progress
>
> **摘要:** Reinforcement learning has proven its effectiveness in enhancing the reasoning capabilities of large language models. Recent research efforts have progressively extended this paradigm to multimodal reasoning tasks. Due to the inherent complexity and diversity of multimodal tasks, especially in semantic content and problem formulations, existing models often exhibit unstable performance across various domains and difficulty levels. To address these limitations, we propose VL-Cogito, an advanced multimodal reasoning model trained via a novel multi-stage Progressive Curriculum Reinforcement Learning (PCuRL) framework. PCuRL systematically guides the model through tasks of gradually increasing difficulty, substantially improving its reasoning abilities across diverse multimodal contexts. The framework introduces two key innovations: (1) an online difficulty soft weighting mechanism, dynamically adjusting training difficulty across successive RL training stages; and (2) a dynamic length reward mechanism, which encourages the model to adaptively regulate its reasoning path length according to task complexity, thus balancing reasoning efficiency with correctness. Experimental evaluations demonstrate that VL-Cogito consistently matches or surpasses existing reasoning-oriented models across mainstream multimodal benchmarks spanning mathematics, science, logic, and general understanding, validating the effectiveness of our approach.
>
---
#### [replaced 003] GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19480v3](http://arxiv.org/pdf/2503.19480v3)**

> **作者:** Shijie Ma; Yuying Ge; Teng Wang; Yuxin Guo; Yixiao Ge; Ying Shan
>
> **备注:** ICCV 2025. Project released at: https://mashijie1028.github.io/GenHancer/
>
> **摘要:** The synergy between generative and discriminative models receives growing attention. While discriminative Contrastive Language-Image Pre-Training (CLIP) excels in high-level semantics, it struggles with perceiving fine-grained visual details. Generally, to enhance representations, generative models take CLIP's visual features as conditions for reconstruction. However, the underlying principle remains underexplored. In this work, we empirically found that visually perfect generations are not always optimal for representation enhancement. The essence lies in effectively extracting fine-grained knowledge from generative models while mitigating irrelevant information. To explore critical factors, we delve into three aspects: (1) Conditioning mechanisms: We found that even a small number of local tokens can drastically reduce the difficulty of reconstruction, leading to collapsed training. We thus conclude that utilizing only global visual tokens as conditions is the most effective strategy. (2) Denoising configurations: We observed that end-to-end training introduces extraneous information. To address this, we propose a two-stage training strategy to prioritize learning useful visual knowledge. Additionally, we demonstrate that lightweight denoisers can yield remarkable improvements. (3) Generation paradigms: We explore both continuous and discrete denoisers with desirable outcomes, validating the versatility of our method. Through our in-depth explorations, we have finally arrived at an effective method, namely GenHancer, which consistently outperforms prior arts on the MMVP-VLM benchmark, e.g., 6.0% on OpenAICLIP. The enhanced CLIP can be further plugged into multimodal large language models for better vision-centric performance. All the models and codes are made publicly available.
>
---
#### [replaced 004] Continual-MEGA: A Large-scale Benchmark for Generalizable Continual Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00956v2](http://arxiv.org/pdf/2506.00956v2)**

> **作者:** Geonu Lee; Yujeong Oh; Geonhui Jang; Soyoung Lee; Jeonghyo Song; Sungmin Cha; YoungJoon Yoo
>
> **摘要:** In this paper, we introduce a new benchmark for continual learning in anomaly detection, aimed at better reflecting real-world deployment scenarios. Our benchmark, Continual-MEGA, includes a large and diverse dataset that significantly expands existing evaluation settings by combining carefully curated existing datasets with our newly proposed dataset, ContinualAD. In addition to standard continual learning with expanded quantity, we propose a novel scenario that measures zero-shot generalization to unseen classes, those not observed during continual adaptation. This setting poses a new problem setting that continual adaptation also enhances zero-shot performance. We also present a unified baseline algorithm that improves robustness in few-shot detection and maintains strong generalization. Through extensive evaluations, we report three key findings: (1) existing methods show substantial room for improvement, particularly in pixel-level defect localization; (2) our proposed method consistently outperforms prior approaches; and (3) the newly introduced ContinualAD dataset enhances the performance of strong anomaly detection models. We release the benchmark and code in https://github.com/Continual-Mega/Continual-Mega.
>
---
#### [replaced 005] Estimating Scene Flow in Robot Surroundings with Distributed Miniaturized Time-of-Flight Sensors
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02439v2](http://arxiv.org/pdf/2504.02439v2)**

> **作者:** Jack Sander; Giammarco Caroleo; Alessandro Albini; Perla Maiolino
>
> **备注:** 7 pages, 5 figures, 2 tables, 1 algorithm, IEEE RO-MAN 2025 accepted paper
>
> **摘要:** Tracking motions of humans or objects in the surroundings of the robot is essential to improve safe robot motions and reactions. In this work, we present an approach for scene flow estimation from low-density and noisy point clouds acquired from miniaturized Time of Flight (ToF) sensors distributed on the robot body. The proposed method clusters points from consecutive frames and applies Iterative Closest Point (ICP) to estimate a dense motion flow, with additional steps introduced to mitigate the impact of sensor noise and low-density data points. Specifically, we employ a fitness-based classification to distinguish between stationary and moving points and an inlier removal strategy to refine geometric correspondences. The proposed approach is validated in an experimental setup where 24 ToF are used to estimate the velocity of an object moving at different controlled speeds. Experimental results show that the method consistently approximates the direction of the motion and its magnitude with an error which is in line with sensor noise.
>
---
#### [replaced 006] Advancing Vision-based Human Action Recognition: Exploring Vision-Language CLIP Model for Generalisation in Domain-Independent Tasks
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.18675v2](http://arxiv.org/pdf/2507.18675v2)**

> **作者:** Utkarsh Shandilya; Marsha Mariya Kappan; Sanyam Jain; Vijeta Sharma
>
> **摘要:** Human action recognition plays a critical role in healthcare and medicine, supporting applications such as patient behavior monitoring, fall detection, surgical robot supervision, and procedural skill assessment. While traditional models like CNNs and RNNs have achieved moderate success, they often struggle to generalize across diverse and complex actions. Recent advancements in vision-language models, especially the transformer-based CLIP model, offer promising capabilities for generalizing action recognition from video data. In this work, we evaluate CLIP on the UCF-101 dataset and systematically analyze its performance under three masking strategies: (1) percentage-based and shape-based black masking at 10%, 30%, and 50%, (2) feature-specific masking to suppress bias-inducing elements, and (3) isolation masking that retains only class-specific regions. Our results reveal that CLIP exhibits inconsistent behavior and frequent misclassifications, particularly when essential visual cues are obscured. To overcome these limitations, we propose incorporating class-specific noise, learned via a custom loss function, to reinforce attention to class-defining features. This enhancement improves classification accuracy and model confidence while reducing bias. We conclude with a discussion on the challenges of applying such models in clinical domains and outline directions for future work to improve generalizability across domain-independent healthcare scenarios.
>
---
#### [replaced 007] SDFit: 3D Object Pose and Shape by Fitting a Morphable SDF to a Single Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.16178v3](http://arxiv.org/pdf/2409.16178v3)**

> **作者:** Dimitrije Antić; Georgios Paschalidis; Shashank Tripathi; Theo Gevers; Sai Kumar Dwivedi; Dimitrios Tzionas
>
> **备注:** ICCV'25 Camera Ready; 12 pages, 11 figures, 5 tables
>
> **摘要:** Recovering 3D object pose and shape from a single image is a challenging and ill-posed problem. This is due to strong (self-)occlusions, depth ambiguities, the vast intra- and inter-class shape variance, and the lack of 3D ground truth for natural images. Existing deep-network methods are trained on synthetic datasets to predict 3D shapes, so they often struggle generalizing to real-world images. Moreover, they lack an explicit feedback loop for refining noisy estimates, and primarily focus on geometry without directly considering pixel alignment. To tackle these limitations, we develop a novel render-and-compare optimization framework, called SDFit. This has three key innovations: First, it uses a learned category-specific and morphable signed-distance-function (mSDF) model, and fits this to an image by iteratively refining both 3D pose and shape. The mSDF robustifies inference by constraining the search on the manifold of valid shapes, while allowing for arbitrary shape topologies. Second, SDFit retrieves an initial 3D shape that likely matches the image, by exploiting foundational models for efficient look-up into 3D shape databases. Third, SDFit initializes pose by establishing rich 2D-3D correspondences between the image and the mSDF through foundational features. We evaluate SDFit on three image datasets, i.e., Pix3D, Pascal3D+, and COMIC. SDFit performs on par with SotA feed-forward networks for unoccluded images and common poses, but is uniquely robust to occlusions and uncommon poses. Moreover, it requires no retraining for unseen images. Thus, SDFit contributes new insights for generalizing in the wild. Code is available at https://anticdimi.github.io/sdfit.
>
---
#### [replaced 008] WeakSupCon: Weakly Supervised Contrastive Learning for Encoder Pre-training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.04165v2](http://arxiv.org/pdf/2503.04165v2)**

> **作者:** Bodong Zhang; Hamid Manoochehri; Xiwen Li; Beatrice S. Knudsen; Tolga Tasdizen
>
> **备注:** Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025 workshop on Efficient Medical AI
>
> **摘要:** Weakly supervised multiple instance learning (MIL) is a challenging task given that only bag-level labels are provided, while each bag typically contains multiple instances. This topic has been extensively studied in histopathological image analysis, where labels are usually available only at the whole slide image (WSI) level, while each WSI could be divided into thousands of small image patches for training. The dominant MIL approaches focus on feature aggregation and take fixed patch features as inputs. However, weakly supervised feature representation learning in MIL settings is always neglected. Those features used to be generated by self-supervised learning methods that do not utilize weak labels, or by foundation encoders pre-trained on other large datasets. In this paper, we propose a novel weakly supervised feature representation learning method called Weakly Supervised Contrastive Learning (WeakSupCon) that utilizes bag-level labels. In our method, we employ multi-task learning and define distinct contrastive losses for samples with different bag labels. Our experiments demonstrate that the features generated using WeakSupCon with limited computing resources significantly enhance MIL classification performance compared to self-supervised approaches across three datasets. Our WeakSupCon code is available at github.com/BzhangURU/Paper_WeakSupCon
>
---
#### [replaced 009] Understanding implementation pitfalls of distance-based metrics for image segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.02630v2](http://arxiv.org/pdf/2410.02630v2)**

> **作者:** Gasper Podobnik; Tomaz Vrtovec
>
> **摘要:** Distance-based metrics, such as the Hausdorff distance (HD), are widely used to validate segmentation performance in (bio)medical imaging. However, their implementation is complex, and critical differences across open-source tools remain largely unrecognized by the community. These discrepancies undermine benchmarking efforts, introduce bias in biomarker calculations, and potentially distort medical device development and clinical commissioning. In this study, we systematically dissect 11 open-source tools that implement distance-based metric computation by performing both a conceptual analysis of their computational steps and an empirical analysis on representative two- and three-dimensional image datasets. Alarmingly, we observed deviations in HD exceeding 100 mm and identified multiple statistically significant differences between tools - demonstrating that statistically significant improvements on the same set of segmentations can be achieved simply by selecting a particular implementation. These findings cast doubts on the validity of prior comparisons of results across studies without accounting for the differences in metric implementations. To address this, we provide practical recommendations for tool selection; additionally, our conceptual analysis informs about the future evolution of implementing open-source tools.
>
---
#### [replaced 010] Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.03222v5](http://arxiv.org/pdf/2503.03222v5)**

> **作者:** Zhumei Wang; Zechen Hu; Ruoxi Guo; Huaijin Pi; Ziyong Feng; Sida Peng; Xiaowei Zhou; Mingtao Pei; Siyuan Huang
>
> **备注:** Project page: https://wangzhumei.github.io/mocap-2-to-3/
>
> **摘要:** Recovering absolute human motion from monocular inputs is challenging due to two main issues. First, existing methods depend on 3D training data collected from limited environments, constraining out-of-distribution generalization. The second issue is the difficulty of estimating metric-scale poses from monocular input. To address these challenges, we introduce Mocap-2-to-3, a novel framework that performs multi-view lifting from monocular input by leveraging 2D data pre-training, enabling the reconstruction of metrically accurate 3D motions with absolute positions. To leverage abundant 2D data, we decompose complex 3D motion into multi-view syntheses. We first pretrain a single-view diffusion model on extensive 2D datasets, then fine-tune a multi-view model using public 3D data to enable view-consistent motion generation from monocular input, allowing the model to acquire action priors and diversity through 2D data. Furthermore, to recover absolute poses, we propose a novel human motion representation that decouples the learning of local pose and global movements, while encoding geometric priors of the ground to accelerate convergence. This enables progressive recovery of motion in absolute space during inference. Experimental results on in-the-wild benchmarks demonstrate that our method surpasses state-of-the-art approaches in both camera-space motion realism and world-grounded human positioning, while exhibiting superior generalization capability. Our code will be made publicly available.
>
---
#### [replaced 011] DisTime: Distribution-based Time Representation for Video Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24329v2](http://arxiv.org/pdf/2505.24329v2)**

> **作者:** Yingsen Zeng; Zepeng Huang; Yujie Zhong; Chengjian Feng; Jie Hu; Lin Ma; Yang Liu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Despite advances in general video understanding, Video Large Language Models (Video-LLMs) face challenges in precise temporal localization due to discrete time representations and limited temporally aware datasets. Existing methods for temporal expression either conflate time with text-based numerical values, add a series of dedicated temporal tokens, or regress time using specialized temporal grounding heads. To address these issues, we introduce DisTime, a lightweight framework designed to enhance temporal comprehension in Video-LLMs. DisTime employs a learnable token to create a continuous temporal embedding space and incorporates a Distribution-based Time Decoder that generates temporal probability distributions, effectively mitigating boundary ambiguities and maintaining temporal continuity. Additionally, the Distribution-based Time Encoder re-encodes timestamps to provide time markers for Video-LLMs. To overcome temporal granularity limitations in existing datasets, we propose an automated annotation paradigm that combines the captioning capabilities of Video-LLMs with the localization expertise of dedicated temporal models. This leads to the creation of InternVid-TG, a substantial dataset with 1.25M temporally grounded events across 179k videos, surpassing ActivityNet-Caption by 55 times. Extensive experiments demonstrate that DisTime achieves state-of-the-art performance across benchmarks in three time-sensitive tasks while maintaining competitive performance in Video QA tasks. Code and data are released at https://github.com/josephzpng/DisTime.
>
---
#### [replaced 012] Acknowledging Focus Ambiguity in Visual Questions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.02201v2](http://arxiv.org/pdf/2501.02201v2)**

> **作者:** Chongyan Chen; Yu-Yun Tseng; Zhuoheng Li; Anush Venkatesh; Danna Gurari
>
> **摘要:** No published work on visual question answering (VQA) accounts for ambiguity regarding where the content described in the question is located in the image. To fill this gap, we introduce VQ-FocusAmbiguity, the first VQA dataset that visually grounds each plausible image region a question could refer to when arriving at valid answers. We next analyze and compare our dataset to existing datasets to reveal its unique properties. Finally, we benchmark modern models for two novel tasks related to acknowledging focus ambiguity: recognizing whether a visual question has focus ambiguity and locating all plausible focus regions within the image. Results show that the dataset is challenging for modern models. To facilitate future progress on these tasks, we publicly share the dataset with an evaluation server at https://vizwiz.org/tasks-and-datasets/focus-ambiguity-in-visual-questions.
>
---
#### [replaced 013] Beyond the Encoder: Joint Encoder-Decoder Contrastive Pre-Training Improves Dense Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.17526v2](http://arxiv.org/pdf/2503.17526v2)**

> **作者:** Sébastien Quetin; Tapotosh Ghosh; Farhad Maleki
>
> **摘要:** Contrastive learning methods in self-supervised settings have primarily focused on pre-training encoders, while decoders are typically introduced and trained separately for downstream dense prediction tasks. However, this conventional approach overlooks the potential benefits of jointly pre-training both encoder and decoder. In this paper, we propose DeCon, an efficient encoder-decoder self-supervised learning (SSL) framework that supports joint contrastive pre-training. We first extend existing SSL architectures to accommodate diverse decoders and their corresponding contrastive losses. Then, we introduce a weighted encoder-decoder contrastive loss with non-competing objectives to enable the joint pre-training of encoder-decoder architectures. By adapting an established contrastive SSL framework for dense prediction tasks, DeCon achieves new state-of-the-art results: on COCO object detection and instance segmentation when pre-trained on COCO dataset; across almost all dense downstream benchmark tasks when pre-trained on COCO+ and ImageNet-1K. Our results demonstrate that joint pre-training enhances the representation power of the encoder and improves performance in dense prediction tasks. This gain persists across heterogeneous decoder architectures, various encoder architectures, and in out-of-domain limited-data scenarios.
>
---
#### [replaced 014] Insights into Closed-form IPM-GAN Discriminator Guidance for Diffusion Modeling
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2306.01654v2](http://arxiv.org/pdf/2306.01654v2)**

> **作者:** Aadithya Srikanth; Siddarth Asokan; Nishanth Shetty; Chandra Sekhar Seelamantula
>
> **摘要:** Diffusion models are a state-of-the-art generative modeling framework that transform noise to images via Langevin sampling, guided by the score, which is the gradient of the logarithm of the data distribution. Recent works have shown empirically that the generation quality can be improved when guided by classifier network, which is typically the discriminator trained in a generative adversarial network (GAN) setting. In this paper, we propose a theoretical framework to analyze the effect of the GAN discriminator on Langevin-based sampling, and show that the IPM-GAN optimization can be seen as one of smoothed score-matching, wherein the scores of the data and the generator distributions are convolved with the kernel function associated with the IPM. The proposed approach serves to unify score-based training and optimization of IPM-GANs. Based on these insights, we demonstrate that closed-form kernel-based discriminator guidance, results in improvements (in terms of CLIP-FID and KID metrics) when applied atop baseline diffusion models. We demonstrate these results on the denoising diffusion implicit model (DDIM) and latent diffusion model (LDM) settings on various standard datasets. We also show that the proposed approach can be combined with existing accelerated-diffusion techniques to improve latent-space image generation.
>
---
#### [replaced 015] Revisiting the Evaluation Bias Introduced by Frame Sampling Strategies in Surgical Video Segmentation Using SAM2
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.20934v3](http://arxiv.org/pdf/2502.20934v3)**

> **作者:** Utku Ozbulak; Seyed Amir Mousavi; Francesca Tozzi; Niki Rashidian; Wouter Willaert; Wesley De Neve; Joris Vankerschaver
>
> **备注:** Accepted for publication in the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) Workshop on Fairness of AI in Medical Imaging (FAIMI), 2025
>
> **摘要:** Real-time video segmentation is a promising opportunity for AI-assisted surgery, offering intraoperative guidance by identifying tools and anatomical structures. Despite growing interest in surgical video segmentation, annotation protocols vary widely across datasets -- some provide dense, frame-by-frame labels, while others rely on sparse annotations sampled at low frame rates such as 1 FPS. In this study, we investigate how such inconsistencies in annotation density and frame rate sampling influence the evaluation of zero-shot segmentation models, using SAM2 as a case study for cholecystectomy procedures. Surprisingly, we find that under conventional sparse evaluation settings, lower frame rates can appear to outperform higher ones due to a smoothing effect that conceals temporal inconsistencies. However, when assessed under real-time streaming conditions, higher frame rates yield superior segmentation stability, particularly for dynamic objects like surgical graspers. To understand how these differences align with human perception, we conducted a survey among surgeons, nurses, and machine learning engineers and found that participants consistently preferred high-FPS segmentation overlays, reinforcing the importance of evaluating every frame in real-time applications rather than relying on sparse sampling strategies. Our findings highlight the risk of evaluation bias that is introduced by inconsistent dataset protocols and bring attention to the need for temporally fair benchmarking in surgical video AI.
>
---
#### [replaced 016] Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application
- **分类: cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06989v2](http://arxiv.org/pdf/2503.06989v2)**

> **作者:** Wenzhuo Xu; Zhipeng Wei; Xiongtao Sun; Zonghao Ying; Deyue Zhang; Dongdong Yang; Xiangzheng Zhang; Quanchen Zou
>
> **摘要:** Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.
>
---
#### [replaced 017] Controlling diverse robots by inferring Jacobian fields with deep networks
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.08722v2](http://arxiv.org/pdf/2407.08722v2)**

> **作者:** Sizhe Lester Li; Annan Zhang; Boyuan Chen; Hanna Matusik; Chao Liu; Daniela Rus; Vincent Sitzmann
>
> **备注:** Project Page: https://sizhe-li.github.io/publication/neural_jacobian_field
>
> **摘要:** Mirroring the complex structures and diverse functions of natural organisms is a long-standing challenge in robotics. Modern fabrication techniques have greatly expanded the feasible hardware, but using these systems requires control software to translate the desired motions into actuator commands. Conventional robots can easily be modeled as rigid links connected by joints, but it remains an open challenge to model and control biologically inspired robots that are often soft or made of several materials, lack sensing capabilities, and may change their material properties with use. Here, we introduce a method that uses deep neural networks to map a video stream of a robot to its visuomotor Jacobian field (the sensitivity of all 3D points to the robot's actuators). Our method enables the control of robots from only a single camera, makes no assumptions about the robots' materials, actuation, or sensing, and is trained without expert intervention by observing the execution of random commands. We demonstrate our method on a diverse set of robot manipulators that vary in actuation, materials, fabrication, and cost. Our approach achieves accurate closed-loop control and recovers the causal dynamic structure of each robot. Because it enables robot control using a generic camera as the only sensor, we anticipate that our work will broaden the design space of robotic systems and serve as a starting point for lowering the barrier to robotic automation.
>
---
#### [replaced 018] Indian Sign Language Detection for Real-Time Translation using Machine Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20414v2](http://arxiv.org/pdf/2507.20414v2)**

> **作者:** Rajat Singhal; Jatin Gupta; Akhil Sharma; Anushka Gupta; Navya Sharma
>
> **备注:** 7 pages, 6 figures, 2 tables. Published in Proceedings of the 6th International Conference on Recent Advances in Information Technology (RAIT), 2025, IEEE
>
> **摘要:** Gestural language is used by deaf & mute communities to communicate through hand gestures & body movements that rely on visual-spatial patterns known as sign languages. Sign languages, which rely on visual-spatial patterns of hand gestures & body movements, are the primary mode of communication for deaf & mute communities worldwide. Effective communication is fundamental to human interaction, yet individuals in these communities often face significant barriers due to a scarcity of skilled interpreters & accessible translation technologies. This research specifically addresses these challenges within the Indian context by focusing on Indian Sign Language (ISL). By leveraging machine learning, this study aims to bridge the critical communication gap for the deaf & hard-of-hearing population in India, where technological solutions for ISL are less developed compared to other global sign languages. We propose a robust, real-time ISL detection & translation system built upon a Convolutional Neural Network (CNN). Our model is trained on a comprehensive ISL dataset & demonstrates exceptional performance, achieving a classification accuracy of 99.95%. This high precision underscores the model's capability to discern the nuanced visual features of different signs. The system's effectiveness is rigorously evaluated using key performance metrics, including accuracy, F1 score, precision & recall, ensuring its reliability for real-world applications. For real-time implementation, the framework integrates MediaPipe for precise hand tracking & motion detection, enabling seamless translation of dynamic gestures. This paper provides a detailed account of the model's architecture, the data preprocessing pipeline & the classification methodology. The research elaborates the model architecture, preprocessing & classification methodologies for enhancing communication in deaf & mute communities.
>
---
#### [replaced 019] Sparfels: Fast Reconstruction from Sparse Unposed Imagery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02178v4](http://arxiv.org/pdf/2505.02178v4)**

> **作者:** Shubhendu Jena; Amine Ouasfi; Mae Younes; Adnane Boukhayma
>
> **备注:** ICCV 2025. Project page : https://shubhendu-jena.github.io/Sparfels-web/
>
> **摘要:** We present a method for Sparse view reconstruction with surface element splatting that runs within 3 minutes on a consumer grade GPU. While few methods address sparse radiance field learning from noisy or unposed sparse cameras, shape recovery remains relatively underexplored in this setting. Several radiance and shape learning test-time optimization methods address the sparse posed setting by learning data priors or using combinations of external monocular geometry priors. Differently, we propose an efficient and simple pipeline harnessing a single recent 3D foundation model. We leverage its various task heads, notably point maps and camera initializations to instantiate a bundle adjusting 2D Gaussian Splatting (2DGS) model, and image correspondences to guide camera optimization midst 2DGS training. Key to our contribution is a novel formulation of splatted color variance along rays, which can be computed efficiently. Reducing this moment in training leads to more accurate shape reconstructions. We demonstrate state-of-the-art performances in the sparse uncalibrated setting in reconstruction and novel view benchmarks based on established multi-view datasets.
>
---
#### [replaced 020] Learning from Rendering: Realistic and Controllable Extreme Rainy Image Synthesis for Autonomous Driving Simulation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16421v2](http://arxiv.org/pdf/2502.16421v2)**

> **作者:** Kaibin Zhou; Kaifeng Huang; Hao Deng; Zelin Tao; Ziniu Liu; Lin Zhang; Shengjie Zhao
>
> **摘要:** Autonomous driving simulators provide an effective and low-cost alternative for evaluating or enhancing visual perception models. However, the reliability of evaluation depends on the diversity and realism of the generated scenes. Extreme weather conditions, particularly extreme rainfalls, are rare and costly to capture in real-world settings. While simulated environments can help address this limitation, existing rainy image synthesizers often suffer from poor controllability over illumination and limited realism, which significantly undermines the effectiveness of the model evaluation. To that end, we propose a learning-from-rendering rainy image synthesizer, which combines the benefits of the realism of rendering-based methods and the controllability of learning-based methods. To validate the effectiveness of our extreme rainy image synthesizer on semantic segmentation task, we require a continuous set of well-labeled extreme rainy images. By integrating the proposed synthesizer with the CARLA driving simulator, we develop CARLARain an extreme rainy street scene simulator which can obtain paired rainy-clean images and labels under complex illumination conditions. Qualitative and quantitative experiments validate that CARLARain can effectively improve the accuracy of semantic segmentation models in extreme rainy scenes, with the models' accuracy (mIoU) improved by 5% - 8% on the synthetic dataset and significantly enhanced in real extreme rainy scenarios under complex illuminations. Our source code and datasets are available at https://github.com/kb824999404/CARLARain/.
>
---
#### [replaced 021] Accenture-NVS1: A Novel View Synthesis Dataset
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.18711v2](http://arxiv.org/pdf/2503.18711v2)**

> **作者:** Thomas Sugg; Kyle O'Brien; Lekh Poudel; Alex Dumouchelle; Michelle Jou; Marc Bosch; Deva Ramanan; Srinivasa Narasimhan; Shubham Tulsiani
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** This paper introduces ACC-NVS1, a specialized dataset designed for research on Novel View Synthesis specifically for airborne and ground imagery. Data for ACC-NVS1 was collected in Austin, TX and Pittsburgh, PA in 2023 and 2024. The collection encompasses six diverse real-world scenes captured from both airborne and ground cameras, resulting in a total of 148,000 images. ACC-NVS1 addresses challenges such as varying altitudes and transient objects. This dataset is intended to supplement existing datasets, providing additional resources for comprehensive research, rather than serving as a benchmark.
>
---
#### [replaced 022] LLaVA-MORE: A Comparative Study of LLMs and Visual Backbones for Enhanced Visual Instruction Tuning
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.15621v2](http://arxiv.org/pdf/2503.15621v2)**

> **作者:** Federico Cocchi; Nicholas Moratelli; Davide Caffagni; Sara Sarto; Lorenzo Baraldi; Marcella Cornia; Rita Cucchiara
>
> **备注:** ICCV 2025 Workshop on What is Next in Multimodal Foundation Models
>
> **摘要:** Recent progress in Multimodal Large Language Models (MLLMs) has highlighted the critical roles of both the visual backbone and the underlying language model. While prior work has primarily focused on scaling these components to billions of parameters, the trade-offs between model size, architecture, and performance remain underexplored. Additionally, inconsistencies in training data and evaluation protocols have hindered direct comparisons, making it difficult to derive optimal design choices. In this paper, we introduce LLaVA-MORE, a new family of MLLMs that integrates recent language models with diverse visual backbones. To ensure fair comparisons, we employ a unified training protocol applied consistently across all architectures. Our analysis systematically explores both small- and medium-scale LLMs -- including Phi-4, LLaMA-3.1, and Gemma-2 -- to evaluate multimodal reasoning, generation, and instruction following, while examining the relationship between model size and performance. Beyond evaluating the LLM impact on final results, we conduct a comprehensive study of various visual encoders, ranging from CLIP-based architectures to alternatives such as DINOv2, SigLIP, and SigLIP2. Additional experiments investigate the effects of increased image resolution and variations in pre-training datasets. Overall, our results provide insights into the design of more effective MLLMs, offering a reproducible evaluation framework that facilitates direct comparisons and can guide future model development. Our source code and trained models are publicly available at: https://github.com/aimagelab/LLaVA-MORE.
>
---
#### [replaced 023] Color as the Impetus: Transforming Few-Shot Learner
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.22136v2](http://arxiv.org/pdf/2507.22136v2)**

> **作者:** Chaofei Qi; Zhitai Liu; Jianbin Qiu
>
> **摘要:** Humans possess innate meta-learning capabilities, partly attributable to their exceptional color perception. In this paper, we pioneer an innovative viewpoint on few-shot learning by simulating human color perception mechanisms. We propose the ColorSense Learner, a bio-inspired meta-learning framework that capitalizes on inter-channel feature extraction and interactive learning. By strategically emphasizing distinct color information across different channels, our approach effectively filters irrelevant features while capturing discriminative characteristics. Color information represents the most intuitive visual feature, yet conventional meta-learning methods have predominantly neglected this aspect, focusing instead on abstract feature differentiation across categories. Our framework bridges the gap via synergistic color-channel interactions, enabling better intra-class commonality extraction and larger inter-class differences. Furthermore, we introduce a meta-distiller based on knowledge distillation, ColorSense Distiller, which incorporates prior teacher knowledge to augment the student network's meta-learning capacity. We've conducted comprehensive coarse/fine-grained and cross-domain experiments on eleven few-shot benchmarks for validation. Numerous experiments reveal that our methods have extremely strong generalization ability, robustness, and transferability, and effortless handle few-shot classification from the perspective of color perception.
>
---
#### [replaced 024] Step1X-Edit: A Practical Framework for General Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.17761v5](http://arxiv.org/pdf/2504.17761v5)**

> **作者:** Shiyu Liu; Yucheng Han; Peng Xing; Fukun Yin; Rui Wang; Wei Cheng; Jiaqi Liao; Yingming Wang; Honghao Fu; Chunrui Han; Guopeng Li; Yuang Peng; Quan Sun; Jingwei Wu; Yan Cai; Zheng Ge; Ranchen Ming; Lei Xia; Xianfang Zeng; Yibo Zhu; Binxing Jiao; Xiangyu Zhang; Gang Yu; Daxin Jiang
>
> **备注:** code: https://github.com/stepfun-ai/Step1X-Edit
>
> **摘要:** In recent years, image editing models have witnessed remarkable and rapid development. The recent unveiling of cutting-edge multimodal models such as GPT-4o and Gemini2 Flash has introduced highly promising image editing capabilities. These models demonstrate an impressive aptitude for fulfilling a vast majority of user-driven editing requirements, marking a significant advancement in the field of image manipulation. However, there is still a large gap between the open-source algorithm with these closed-source models. Thus, in this paper, we aim to release a state-of-the-art image editing model, called Step1X-Edit, which can provide comparable performance against the closed-source models like GPT-4o and Gemini2 Flash. More specifically, we adopt the Multimodal LLM to process the reference image and the user's editing instruction. A latent embedding has been extracted and integrated with a diffusion image decoder to obtain the target image. To train the model, we build a data generation pipeline to produce a high-quality dataset. For evaluation, we develop the GEdit-Bench, a novel benchmark rooted in real-world user instructions. Experimental results on GEdit-Bench demonstrate that Step1X-Edit outperforms existing open-source baselines by a substantial margin and approaches the performance of leading proprietary models, thereby making significant contributions to the field of image editing.
>
---
#### [replaced 025] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.11098v3](http://arxiv.org/pdf/2411.11098v3)**

> **作者:** Xi Fang; Jiankun Wang; Xiaochen Cai; Shangqian Chen; Shuwen Yang; Haoyi Tao; Nan Wang; Lin Yao; Linfeng Zhang; Guolin Ke
>
> **摘要:** In recent decades, chemistry publications and patents have increased rapidly. A significant portion of key information is embedded in molecular structure figures, complicating large-scale literature searches and limiting the application of large language models in fields such as biology, chemistry, and pharmaceuticals. The automatic extraction of precise chemical structures is of critical importance. However, the presence of numerous Markush structures in real-world documents, along with variations in molecular image quality, drawing styles, and noise, significantly limits the performance of existing optical chemical structure recognition (OCSR) methods. We present MolParser, a novel end-to-end OCSR method that efficiently and accurately recognizes chemical structures from real-world documents, including difficult Markush structure. We use a extended SMILES encoding rule to annotate our training dataset. Under this rule, we build MolParser-7M, the largest annotated molecular image dataset to our knowledge. While utilizing a large amount of synthetic data, we employed active learning methods to incorporate substantial in-the-wild data, specifically samples cropped from real patents and scientific literature, into the training process. We trained an end-to-end molecular image captioning model, MolParser, using a curriculum learning approach. MolParser significantly outperforms classical and learning-based methods across most scenarios, with potential for broader downstream applications. The dataset is publicly available in huggingface.
>
---
#### [replaced 026] VisNumBench: Evaluating Number Sense of Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14939v2](http://arxiv.org/pdf/2503.14939v2)**

> **作者:** Tengjin Weng; Jingyi Wang; Wenhao Jiang; Zhong Ming
>
> **备注:** accepted by ICCV 2025
>
> **摘要:** Can Multimodal Large Language Models (MLLMs) develop an intuitive number sense similar to humans? Targeting this problem, we introduce Visual Number Benchmark (VisNumBench) to evaluate the number sense abilities of MLLMs across a wide range of visual numerical tasks. VisNumBench consists of about 1,900 multiple-choice question-answer pairs derived from both synthetic and real-world visual data, covering seven visual numerical attributes and four types of visual numerical estimation tasks. Our experiments on VisNumBench led to the following key findings: (i) The 17 MLLMs we tested, including open-source models such as Qwen2.5-VL and InternVL2.5, as well as proprietary models like GPT-4o and Gemini 2.0 Flash, perform significantly below human levels in number sense-related tasks. (ii) Multimodal mathematical models and multimodal chain-of-thought (CoT) models did not exhibit significant improvements in number sense abilities. (iii) Stronger MLLMs with larger parameter sizes and broader general abilities demonstrate modest gains in number sense abilities. We believe VisNumBench will serve as a valuable resource for the research community, encouraging further advancements in enhancing MLLMs' number sense abilities. Code and dataset are available at https://wwwtttjjj.github.io/VisNumBench/.
>
---
#### [replaced 027] Learning 3D Scene Analogies with Neural Contextual Scene Maps
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.15897v2](http://arxiv.org/pdf/2503.15897v2)**

> **作者:** Junho Kim; Gwangtak Bae; Eun Sun Lee; Young Min Kim
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Understanding scene contexts is crucial for machines to perform tasks and adapt prior knowledge in unseen or noisy 3D environments. As data-driven learning is intractable to comprehensively encapsulate diverse ranges of layouts and open spaces, we propose teaching machines to identify relational commonalities in 3D spaces. Instead of focusing on point-wise or object-wise representations, we introduce 3D scene analogies, which are smooth maps between 3D scene regions that align spatial relationships. Unlike well-studied single instance-level maps, these scene-level maps smoothly link large scene regions, potentially enabling unique applications in trajectory transfer in AR/VR, long demonstration transfer for imitation learning, and context-aware object rearrangement. To find 3D scene analogies, we propose neural contextual scene maps, which extract descriptor fields summarizing semantic and geometric contexts, and holistically align them in a coarse-to-fine manner for map estimation. This approach reduces reliance on individual feature points, making it robust to input noise or shape variations. Experiments demonstrate the effectiveness of our approach in identifying scene analogies and transferring trajectories or object placements in diverse indoor scenes, indicating its potential for robotics and AR/VR applications. Project page including the code is available through this link: https://82magnolia.github.io/3d_scene_analogies/.
>
---
#### [replaced 028] KineDepth: Utilizing Robot Kinematics for Online Metric Depth Estimation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.19490v2](http://arxiv.org/pdf/2409.19490v2)**

> **作者:** Soofiyan Atar; Yuheng Zhi; Florian Richter; Michael Yip
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Depth perception is essential for a robot's spatial and geometric understanding of its environment, with many tasks traditionally relying on hardware-based depth sensors like RGB-D or stereo cameras. However, these sensors face practical limitations, including issues with transparent and reflective objects, high costs, calibration complexity, spatial and energy constraints, and increased failure rates in compound systems. While monocular depth estimation methods offer a cost-effective and simpler alternative, their adoption in robotics is limited due to their output of relative rather than metric depth, which is crucial for robotics applications. In this paper, we propose a method that utilizes a single calibrated camera, enabling the robot to act as a "measuring stick" to convert relative depth estimates into metric depth in real-time as tasks are performed. Our approach employs an LSTM-based metric depth regressor, trained online and refined through probabilistic filtering, to accurately restore the metric depth across the monocular depth map, particularly in areas proximal to the robot's motion. Experiments with real robots demonstrate that our method significantly outperforms current state-of-the-art monocular metric depth estimation techniques, achieving a 22.1% reduction in depth error and a 52% increase in success rate for a downstream task.
>
---
#### [replaced 029] Balancing Task-invariant Interaction and Task-specific Adaptation for Unified Image Fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.05164v2](http://arxiv.org/pdf/2504.05164v2)**

> **作者:** Xingyu Hu; Junjun Jiang; Chenyang Wang; Kui Jiang; Xianming Liu; Jiayi Ma
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Unified image fusion aims to integrate complementary information from multi-source images, enhancing image quality through a unified framework applicable to diverse fusion tasks. While treating all fusion tasks as a unified problem facilitates task-invariant knowledge sharing, it often overlooks task-specific characteristics, thereby limiting the overall performance. Existing general image fusion methods incorporate explicit task identification to enable adaptation to different fusion tasks. However, this dependence during inference restricts the model's generalization to unseen fusion tasks. To address these issues, we propose a novel unified image fusion framework named "TITA", which dynamically balances both Task-invariant Interaction and Task-specific Adaptation. For task-invariant interaction, we introduce the Interaction-enhanced Pixel Attention (IPA) module to enhance pixel-wise interactions for better multi-source complementary information extraction. For task-specific adaptation, the Operation-based Adaptive Fusion (OAF) module dynamically adjusts operation weights based on task properties. Additionally, we incorporate the Fast Adaptive Multitask Optimization (FAMO) strategy to mitigate the impact of gradient conflicts across tasks during joint training. Extensive experiments demonstrate that TITA not only achieves competitive performance compared to specialized methods across three image fusion scenarios but also exhibits strong generalization to unseen fusion tasks. The source codes are released at https://github.com/huxingyuabc/TITA.
>
---
#### [replaced 030] Tile and Slide : A New Framework for Scaling NeRF from Local to Global 3D Earth Observation
- **分类: cs.CV; cs.AI; cs.GR; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.01631v2](http://arxiv.org/pdf/2507.01631v2)**

> **作者:** Camille Billouard; Dawa Derksen; Alexandre Constantin; Bruno Vallet
>
> **备注:** Accepted at ICCV 2025 Workshop 3D-VAST (From street to space: 3D Vision Across Altitudes). Our code will be made public after the conference at https://github.com/Ellimac0/Snake-NeRF
>
> **摘要:** Neural Radiance Fields (NeRF) have recently emerged as a paradigm for 3D reconstruction from multiview satellite imagery. However, state-of-the-art NeRF methods are typically constrained to small scenes due to the memory footprint during training, which we study in this paper. Previous work on large-scale NeRFs palliate this by dividing the scene into NeRFs. This paper introduces Snake-NeRF, a framework that scales to large scenes. Our out-of-core method eliminates the need to load all images and networks simultaneously, and operates on a single device. We achieve this by dividing the region of interest into NeRFs that 3D tile without overlap. Importantly, we crop the images with overlap to ensure each NeRFs is trained with all the necessary pixels. We introduce a novel $2\times 2$ 3D tile progression strategy and segmented sampler, which together prevent 3D reconstruction errors along the tile edges. Our experiments conclude that large satellite images can effectively be processed with linear time complexity, on a single GPU, and without compromise in quality.
>
---
#### [replaced 031] ZIP: Scalable Crowd Counting via Zero-Inflated Poisson Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19955v3](http://arxiv.org/pdf/2506.19955v3)**

> **作者:** Yiming Ma; Victor Sanchez; Tanaya Guha
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** Most crowd counting methods directly regress blockwise density maps using Mean Squared Error (MSE) losses. This practice has two key limitations: (1) it fails to account for the extreme spatial sparsity of annotations - over 95% of 8x8 blocks are empty across standard benchmarks, so supervision signals in informative regions are diluted by the predominant zeros; (2) MSE corresponds to a Gaussian error model that poorly matches discrete, non-negative count data. To address these issues, we introduce ZIP, a scalable crowd counting framework that models blockwise counts with a Zero-Inflated Poisson likelihood: a zero-inflation term learns the probability a block is structurally empty (handling excess zeros), while the Poisson component captures expected counts when people are present (respecting discreteness). We provide a generalization analysis showing a tighter risk bound for ZIP than MSE-based losses and DMCount provided that the training resolution is moderately large. To assess the scalability of ZIP, we instantiate it on backbones spanning over 100x in parameters/compute. Experiments on ShanghaiTech A & B, UCF-QNRF, and NWPU-Crowd demonstrate that ZIP consistently surpasses state-of-the-art methods across all model scales.
>
---
#### [replaced 032] SinBasis Networks: Matrix-Equivalent Feature Extraction for Wave-Like Optical Spectrograms
- **分类: cs.LG; cs.AI; cs.CV; physics.optics**

- **链接: [http://arxiv.org/pdf/2505.06275v2](http://arxiv.org/pdf/2505.06275v2)**

> **作者:** Yuzhou Zhu; Zheng Zhang; Ruyi Zhang; Liang Zhou
>
> **摘要:** Wave-like images-from attosecond streaking spectrograms to optical spectra, audio mel-spectrograms and periodic video frames-encode critical harmonic structures that elude conventional feature extractors. We propose a unified, matrix-equivalent framework that reinterprets convolution and attention as linear transforms on flattened inputs, revealing filter weights as basis vectors spanning latent feature subspaces. To infuse spectral priors we apply elementwise $\sin(\cdot)$ mappings to each weight matrix. Embedding these transforms into CNN, ViT and Capsule architectures yields Sin-Basis Networks with heightened sensitivity to periodic motifs and built-in invariance to spatial shifts. Experiments on a diverse collection of wave-like image datasets-including 80,000 synthetic attosecond streaking spectrograms, thousands of Raman, photoluminescence and FTIR spectra, mel-spectrograms from AudioSet and cycle-pattern frames from Kinetics-demonstrate substantial gains in reconstruction accuracy, translational robustness and zero-shot cross-domain transfer. Theoretical analysis via matrix isomorphism and Mercer-kernel truncation quantifies how sinusoidal reparametrization enriches expressivity while preserving stability in data-scarce regimes. Sin-Basis Networks thus offer a lightweight, physics-informed approach to deep learning across all wave-form imaging modalities.
>
---
#### [replaced 033] KAN or MLP? Point Cloud Shows the Way Forward
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13593v2](http://arxiv.org/pdf/2504.13593v2)**

> **作者:** Yan Shi; Qingdong He; Yijun Liu; Xiaoyu Liu; Jingyong Su
>
> **摘要:** Multi-Layer Perceptrons (MLPs) have become one of the fundamental architectural component in point cloud analysis due to its effective feature learning mechanism. However, when processing complex geometric structures in point clouds, MLPs' fixed activation functions struggle to efficiently capture local geometric features, while suffering from poor parameter efficiency and high model redundancy. In this paper, we propose PointKAN, which applies Kolmogorov-Arnold Networks (KANs) to point cloud analysis tasks to investigate their efficacy in hierarchical feature representation. First, we introduce a Geometric Affine Module (GAM) to transform local features, improving the model's robustness to geometric variations. Next, in the Local Feature Processing (LFP), a parallel structure extracts both group-level features and global context, providing a rich representation of both fine details and overall structure. Finally, these features are combined and processed in the Global Feature Processing (GFP). By repeating these operations, the receptive field gradually expands, enabling the model to capture complete geometric information of the point cloud. To overcome the high parameter counts and computational inefficiency of standard KANs, we develop Efficient-KANs in the PointKAN-elite variant, which significantly reduces parameters while maintaining accuracy. Experimental results demonstrate that PointKAN outperforms PointMLP on benchmark datasets such as ModelNet40, ScanObjectNN, and ShapeNetPart, with particularly strong performance in Few-shot Learning task. Additionally, PointKAN achieves substantial reductions in parameter counts and computational complexity (FLOPs). This work highlights the potential of KANs-based architectures in 3D vision and opens new avenues for research in point cloud understanding.
>
---
#### [replaced 034] Adapt before Continual Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03956v3](http://arxiv.org/pdf/2506.03956v3)**

> **作者:** Aojun Lu; Tao Feng; Hangjie Yuan; Chunhui Ding; Yanan Sun
>
> **摘要:** Continual Learning (CL) seeks to enable neural networks to incrementally acquire new knowledge (plasticity) while retaining existing knowledge (stability). Although pre-trained models (PTMs) have provided a strong foundation for CL, existing approaches face a fundamental challenge in balancing these two competing objectives. Current methods typically address stability by freezing the PTM backbone, which severely limits the model's plasticity, particularly when incoming data distribution diverges largely from the pre-training data. Alternatively, sequentially fine-tuning the entire PTM can adapt to new knowledge but often leads to catastrophic forgetting, highlighting the critical stability-plasticity trade-off in PTM-based CL. To address this limitation, we propose Adapting PTMs before the core CL} process (ACL), a novel framework that introduces a plug-and-play adaptation phase prior to learning each new task. During this phase, ACL refines the PTM backbone by aligning embeddings with their original class prototypes while distancing them from irrelevant classes. This mechanism theoretically and empirically demonstrates desirable balance between stability and plasticity, significantly improving CL performance across benchmarks and integrated methods. Code is available at https://github.com/byyx666/ACL_code.
>
---
#### [replaced 035] TARS: MinMax Token-Adaptive Preference Strategy for Hallucination Reduction in MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.21584v2](http://arxiv.org/pdf/2507.21584v2)**

> **作者:** Kejia Zhang; Keda Tao; Zhiming Luo; Chang Liu; Jiasheng Tang; Huan Wang
>
> **摘要:** Multimodal large language models (MLLMs) enable vision-language reasoning, yet often generate plausible outputs that are factually incorrect or visually ungrounded, thereby compromising their reliability. Direct preference optimization (DPO) is a common strategy for correcting hallucinations by aligning model outputs with human preferences. Existing DPO strategies typically treat hallucination-related preferences as fixed targets, relying on static supervision signals during training. This approach tends to overfit to superficial linguistic cues in preference data, leading to distributional rigidity and spurious correlations that impair grounding in causally relevant visual information. To overcome this limitation, we propose TARS, a token-adaptive preference strategy that reformulates DPO as a min-max optimization problem. TARS maximizes token-level distributional shifts under semantic constraints to simulate alignment uncertainty, and simultaneously minimizes the expected preference loss under these controlled perturbations. This joint objective preserves causal grounding while mitigating overfitting to preference patterns, thereby reducing hallucinations in multimodal reasoning. We evaluate TARS on multiple hallucination benchmarks and find consistently strong performance. Using only 4.8k preference samples and no expert feedback, TARS reduces hallucination rates from 26.4% to 13.2% and decreases cognition value from 2.5 to 0.4. It outperforms standard DPO and matches GPT-4o on several key metrics.
>
---
#### [replaced 036] A Lightweight Optimization Framework for Estimating 3D Brain Tumor Infiltration
- **分类: physics.med-ph; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.13811v2](http://arxiv.org/pdf/2412.13811v2)**

> **作者:** Jonas Weidner; Michal Balcerak; Ivan Ezhov; André Datchev; Laurin Lux; Lucas Zimmer; Daniel Rueckert; Björn Menze; Benedikt Wiestler
>
> **摘要:** Glioblastoma, the most aggressive primary brain tumor, poses a severe clinical challenge due to its diffuse microscopic infiltration, which remains largely undetected on standard MRI. As a result, current radiotherapy planning employs a uniform 15 mm margin around the resection cavity, failing to capture patient-specific tumor spread. Tumor growth modeling offers a promising approach to reveal this hidden infiltration. However, methods based on partial differential equations or physics-informed neural networks tend to be computationally intensive or overly constrained, limiting their clinical adaptability to individual patients. In this work, we propose a lightweight, rapid, and robust optimization framework that estimates the 3D tumor concentration by fitting it to MRI tumor segmentations while enforcing a smooth concentration landscape. This approach achieves superior tumor recurrence prediction on 192 brain tumor patients across two public datasets, outperforming state-of-the-art baselines while reducing runtime from 30 minutes to less than one minute. Furthermore, we demonstrate the framework's versatility and adaptability by showing its ability to seamlessly integrate additional imaging modalities or physical constraints.
>
---
#### [replaced 037] Vector-Quantized Vision Foundation Models for Object-Centric Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20263v4](http://arxiv.org/pdf/2502.20263v4)**

> **作者:** Rongzhen Zhao; Vivienne Wang; Juho Kannala; Joni Pajarinen
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Perceiving visual scenes as objects and background--like humans do--Object-Centric Learning (OCL) aggregates image or video feature maps into object-level feature vectors, termed \textit{slots}. OCL's self-supervision of reconstructing the input from these aggregated slots struggles with complex object textures, thus Vision Foundation Model (VFM) representations are used as the aggregation input and reconstruction target. However, existing methods leverage VFM representations in diverse ways and often fail to fully exploit their potential. In response, we propose a clean architecture--Vector-Quantized VFMs for OCL (VQ-VFM-OCL, or VVO)--that unifies mainstream OCL methods. The key to our unification is simple yet effective, just shared quantizing the same VFM representation as the reconstruction target. Through mathematical modeling and statistical verification, we further analyze why VFM representations facilitate OCL aggregation and how their shared quantization as reconstruction targets strengthens OCL supervision. Experiments show that across different VFMs, aggregators and decoders, our VVO consistently outperforms baselines in object discovery and recognition, as well as downstream visual prediction and reasoning. The implementation and model checkpoints are available on https://github.com/Genera1Z/VQ-VFM-OCL.
>
---
#### [replaced 038] DeepShade: Enable Shade Simulation by Text-conditioned Image Generation
- **分类: cs.CV; cs.CY; 68T45, 68U10, 62H35; I.2.10; I.4.8; I.5.1**

- **链接: [http://arxiv.org/pdf/2507.12103v3](http://arxiv.org/pdf/2507.12103v3)**

> **作者:** Longchao Da; Xiangrui Liu; Mithun Shivakoti; Thirulogasankar Pranav Kutralingam; Yezhou Yang; Hua Wei
>
> **备注:** 7pages, 4 figures
>
> **摘要:** Heatwaves pose a significant threat to public health, especially as global warming intensifies. However, current routing systems (e.g., online maps) fail to incorporate shade information due to the difficulty of estimating shades directly from noisy satellite imagery and the limited availability of training data for generative models. In this paper, we address these challenges through two main contributions. First, we build an extensive dataset covering diverse longitude-latitude regions, varying levels of building density, and different urban layouts. Leveraging Blender-based 3D simulations alongside building outlines, we capture building shadows under various solar zenith angles throughout the year and at different times of day. These simulated shadows are aligned with satellite images, providing a rich resource for learning shade patterns. Second, we propose the DeepShade, a diffusion-based model designed to learn and synthesize shade variations over time. It emphasizes the nuance of edge features by jointly considering RGB with the Canny edge layer, and incorporates contrastive learning to capture the temporal change rules of shade. Then, by conditioning on textual descriptions of known conditions (e.g., time of day, solar angles), our framework provides improved performance in generating shade images. We demonstrate the utility of our approach by using our shade predictions to calculate shade ratios for real-world route planning in Tempe, Arizona. We believe this work will benefit society by providing a reference for urban planning in extreme heat weather and its potential practical applications in the environment.
>
---
#### [replaced 039] FovEx: Human-Inspired Explanations for Vision Transformers and Convolutional Neural Networks
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.02123v3](http://arxiv.org/pdf/2408.02123v3)**

> **作者:** Mahadev Prasad Panda; Matteo Tiezzi; Martina Vilas; Gemma Roig; Bjoern M. Eskofier; Dario Zanca
>
> **备注:** Accepted in the International Journal of Computer Vision (Springer Nature)
>
> **摘要:** Explainability in artificial intelligence (XAI) remains a crucial aspect for fostering trust and understanding in machine learning models. Current visual explanation techniques, such as gradient-based or class-activation-based methods, often exhibit a strong dependence on specific model architectures. Conversely, perturbation-based methods, despite being model-agnostic, are computationally expensive as they require evaluating models on a large number of forward passes. In this work, we introduce Foveation-based Explanations (FovEx), a novel XAI method inspired by human vision. FovEx seamlessly integrates biologically inspired perturbations by iteratively creating foveated renderings of the image and combines them with gradient-based visual explorations to determine locations of interest efficiently. These locations are selected to maximize the performance of the model to be explained with respect to the downstream task and then combined to generate an attribution map. We provide a thorough evaluation with qualitative and quantitative assessments on established benchmarks. Our method achieves state-of-the-art performance on both transformers (on 4 out of 5 metrics) and convolutional models (on 3 out of 5 metrics), demonstrating its versatility among various architectures. Furthermore, we show the alignment between the explanation map produced by FovEx and human gaze patterns (+14\% in NSS compared to RISE, +203\% in NSS compared to GradCAM). This comparison enhances our confidence in FovEx's ability to close the interpretation gap between humans and machines.
>
---
#### [replaced 040] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.22886v2](http://arxiv.org/pdf/2507.22886v2)**

> **作者:** Kaining Ying; Henghui Ding; Guangquan Jie; Yu-Gang Jiang
>
> **备注:** ICCV 2025, Project Page: https://henghuiding.com/OmniAVS/
>
> **摘要:** Referring audio-visual segmentation (RAVS) has recently seen significant advancements, yet challenges remain in integrating multimodal information and deeply understanding and reasoning about audiovisual content. To extend the boundaries of RAVS and facilitate future research in this field, we propose Omnimodal Referring Audio-Visual Segmentation (OmniAVS), a new dataset containing 2,104 videos and 61,095 multimodal referring expressions. OmniAVS stands out with three key innovations: (1) 8 types of multimodal expressions that flexibly combine text, speech, sound, and visual cues; (2) an emphasis on understanding audio content beyond just detecting their presence; and (3) the inclusion of complex reasoning and world knowledge in expressions. Furthermore, we introduce Omnimodal Instructed Segmentation Assistant (OISA), to address the challenges of multimodal reasoning and fine-grained understanding of audiovisual content in OmniAVS. OISA uses MLLM to comprehend complex cues and perform reasoning-based segmentation. Extensive experiments show that OISA outperforms existing methods on OmniAVS and achieves competitive results on other related tasks.
>
---
#### [replaced 041] Prompt-Based Exemplar Super-Compression and Regeneration for Class-Incremental Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.18266v3](http://arxiv.org/pdf/2311.18266v3)**

> **作者:** Ruxiao Duan; Jieneng Chen; Adam Kortylewski; Alan Yuille; Yaoyao Liu
>
> **备注:** BMVC 2025. Code: https://github.com/KerryDRX/PESCR
>
> **摘要:** Replay-based methods in class-incremental learning (CIL) have attained remarkable success. Despite their effectiveness, the inherent memory restriction results in saving a limited number of exemplars with poor diversity. In this paper, we introduce PESCR, a novel approach that substantially increases the quantity and enhances the diversity of exemplars based on a pre-trained general-purpose diffusion model, without fine-tuning it on target datasets or storing it in the memory buffer. Images are compressed into visual and textual prompts, which are saved instead of the original images, decreasing memory consumption by a factor of 24. In subsequent phases, diverse exemplars are regenerated by the diffusion model. We further propose partial compression and diffusion-based data augmentation to minimize the domain gap between generated exemplars and real images. PESCR significantly improves CIL performance across multiple benchmarks, e.g., 3.2% above the previous state-of-the-art on ImageNet-100.
>
---
#### [replaced 042] ClaraVid: A Holistic Scene Reconstruction Benchmark From Aerial Perspective With Delentropy-Based Complexity Profiling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.17856v2](http://arxiv.org/pdf/2503.17856v2)**

> **作者:** Radu Beche; Sergiu Nedevschi
>
> **备注:** Accepted ICCV 2025
>
> **摘要:** The development of aerial holistic scene understanding algorithms is hindered by the scarcity of comprehensive datasets that enable both semantic and geometric reconstruction. While synthetic datasets offer an alternative, existing options exhibit task-specific limitations, unrealistic scene compositions, and rendering artifacts that compromise real-world applicability. We introduce ClaraVid, a synthetic aerial dataset specifically designed to overcome these limitations. Comprising 16,917 high-resolution images captured at 4032x3024 from multiple viewpoints across diverse landscapes, ClaraVid provides dense depth maps, panoptic segmentation, sparse point clouds, and dynamic object masks, while mitigating common rendering artifacts. To further advance neural reconstruction, we introduce the Delentropic Scene Profile (DSP), a novel complexity metric derived from differential entropy analysis, designed to quantitatively assess scene difficulty and inform reconstruction tasks. Utilizing DSP, we systematically benchmark neural reconstruction methods, uncovering a consistent, measurable correlation between scene complexity and reconstruction accuracy. Empirical results indicate that higher delentropy strongly correlates with increased reconstruction errors, validating DSP as a reliable complexity prior. The data and code are available on the project page at https://rdbch.github.io/claravid/
>
---
#### [replaced 043] Exemplar Med-DETR: Toward Generalized and Robust Lesion Detection in Mammogram Images and beyond
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.19621v2](http://arxiv.org/pdf/2507.19621v2)**

> **作者:** Sheethal Bhat; Bogdan Georgescu; Adarsh Bhandary Panambur; Mathias Zinnen; Tri-Thien Nguyen; Awais Mansoor; Karim Khalifa Elbarbary; Siming Bayer; Florin-Cristian Ghesu; Sasa Grbic; Andreas Maier
>
> **备注:** I am asking for a withdrawal of the paper as I did not have institutional approval to release this paper right now
>
> **摘要:** Detecting abnormalities in medical images poses unique challenges due to differences in feature representations and the intricate relationship between anatomical structures and abnormalities. This is especially evident in mammography, where dense breast tissue can obscure lesions, complicating radiological interpretation. Despite leveraging anatomical and semantic context, existing detection methods struggle to learn effective class-specific features, limiting their applicability across different tasks and imaging modalities. In this work, we introduce Exemplar Med-DETR, a novel multi-modal contrastive detector that enables feature-based detection. It employs cross-attention with inherently derived, intuitive class-specific exemplar features and is trained with an iterative strategy. We achieve state-of-the-art performance across three distinct imaging modalities from four public datasets. On Vietnamese dense breast mammograms, we attain an mAP of 0.7 for mass detection and 0.55 for calcifications, yielding an absolute improvement of 16 percentage points. Additionally, a radiologist-supported evaluation of 100 mammograms from an out-of-distribution Chinese cohort demonstrates a twofold gain in lesion detection performance. For chest X-rays and angiography, we achieve an mAP of 0.25 for mass and 0.37 for stenosis detection, improving results by 4 and 7 percentage points, respectively. These results highlight the potential of our approach to advance robust and generalizable detection systems for medical imaging.
>
---
#### [replaced 044] Divided Attention: Unsupervised Multi-Object Discovery with Contextually Separated Slots
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2304.01430v3](http://arxiv.org/pdf/2304.01430v3)**

> **作者:** Dong Lao; Zhengyang Hu; Francesco Locatello; Yanchao Yang; Stefano Soatto
>
> **摘要:** We investigate the emergence of objects in visual perception in the absence of any semantic annotation. The resulting model has received no supervision, does not use any pre-trained features, and yet it can segment the domain of an image into multiple independently moving regions. The resulting motion segmentation method can handle an unknown and varying number of objects in real-time. The core multi-modal conditional encoder-decoder architecture has one modality (optical flow) feed the encoder to produce a collection of latent codes (slots), and the other modality (color image) conditions the decoder to generate the first modality (flow) from the slots. The training criterion is designed to foster 'information separation' among the slots, while the architecture explicitly allocates activations to individual slots, leading to a method we call Divided Attention (DivA). At test time, DivA handles a different number of objects and different image resolution than seen at training, and is invariant to permutations of the slots. DivA achieves state-of-the-art performance while tripling the runtime speed of comparable methods, up to 104 FPS, and reduces the performance gap from supervised methods to 12% or less. Objects bootstrapped by DivA can then be used to prime static classifiers via contrastive learning. On fewer than 5,000 video clips, training DINO on DivA's object proposals narrows the performance gap to ImageNet-based training by up to 30.2% compared to training directly on the video frames.
>
---
#### [replaced 045] MVG4D: Image Matrix-Based Multi-View and Motion Generation for 4D Content Creation from a Single Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18371v2](http://arxiv.org/pdf/2507.18371v2)**

> **作者:** DongFu Yin; Xiaotian Chen; Fei Richard Yu; Xuanchen Li; Xinhao Zhang
>
> **摘要:** Advances in generative modeling have significantly enhanced digital content creation, extending from 2D images to complex 3D and 4D scenes. Despite substantial progress, producing high-fidelity and temporally consistent dynamic 4D content remains a challenge. In this paper, we propose MVG4D, a novel framework that generates dynamic 4D content from a single still image by combining multi-view synthesis with 4D Gaussian Splatting (4D GS). At its core, MVG4D employs an image matrix module that synthesizes temporally coherent and spatially diverse multi-view images, providing rich supervisory signals for downstream 3D and 4D reconstruction. These multi-view images are used to optimize a 3D Gaussian point cloud, which is further extended into the temporal domain via a lightweight deformation network. Our method effectively enhances temporal consistency, geometric fidelity, and visual realism, addressing key challenges in motion discontinuity and background degradation that affect prior 4D GS-based methods. Extensive experiments on the Objaverse dataset demonstrate that MVG4D outperforms state-of-the-art baselines in CLIP-I, PSNR, FVD, and time efficiency. Notably, it reduces flickering artifacts and sharpens structural details across views and time, enabling more immersive AR/VR experiences. MVG4D sets a new direction for efficient and controllable 4D generation from minimal inputs.
>
---
#### [replaced 046] Pruning All-Rounder: Rethinking and Improving Inference Efficiency for Large Vision Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.06458v2](http://arxiv.org/pdf/2412.06458v2)**

> **作者:** Wei Suo; Ji Ma; Mengyang Sun; Lin Yuanbo Wu; Peng Wang; Yanning Zhang
>
> **备注:** Accepted by ICCV 25
>
> **摘要:** Although Large Vision-Language Models (LVLMs) have achieved impressive results, their high computational costs pose a significant barrier to wide application. To enhance inference efficiency, most existing approaches can be categorized as parameter-dependent or token-dependent strategies to reduce computational demands. However, parameter-dependent methods require retraining LVLMs to recover performance while token-dependent strategies struggle to consistently select the most relevant tokens. In this paper, we systematically analyze the above challenges and provide a series of valuable insights for inference acceleration. Based on these findings, we propose a novel framework, the Pruning All-Rounder (PAR). Different from previous works, PAR develops a meta-router to adaptively organize pruning flows across both tokens and layers. With a self-supervised learning manner, our method achieves a superior balance between performance and efficiency. Notably, PAR is highly flexible, offering multiple pruning versions to address a range of acceleration scenarios. The code for this work is publicly available at https://github.com/ASGO-MM/Pruning-All-Rounder.
>
---
#### [replaced 047] Other Vehicle Trajectories Are Also Needed: A Driving World Model Unifies Ego-Other Vehicle Trajectories in Video Latent Space
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.09215v3](http://arxiv.org/pdf/2503.09215v3)**

> **作者:** Jian Zhu; Zhengyu Jia; Tian Gao; Jiaxin Deng; Shidi Li; Lang Zhang; Fu Liu; Peng Jia; Xianpeng Lang
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Advanced end-to-end autonomous driving systems predict other vehicles' motions and plan ego vehicle's trajectory. The world model that can foresee the outcome of the trajectory has been used to evaluate the autonomous driving system. However, existing world models predominantly emphasize the trajectory of the ego vehicle and leave other vehicles uncontrollable. This limitation hinders their ability to realistically simulate the interaction between the ego vehicle and the driving scenario. In this paper, we propose a driving World Model named EOT-WM, unifying Ego-Other vehicle Trajectories in videos for driving simulation. Specifically, it remains a challenge to match multiple trajectories in the BEV space with each vehicle in the video to control the video generation. We first project ego-other vehicle trajectories in the BEV space into the image coordinate for vehicle-trajectory match via pixel positions. Then, trajectory videos are encoded by the Spatial-Temporal Variational Auto Encoder to align with driving video latents spatially and temporally in the unified visual space. A trajectory-injected diffusion Transformer is further designed to denoise the noisy video latents for video generation with the guidance of ego-other vehicle trajectories. In addition, we propose a metric based on control latent similarity to evaluate the controllability of trajectories. Extensive experiments are conducted on the nuScenes dataset, and the proposed model outperforms the state-of-the-art method by 30% in FID and 55% in FVD. The model can also predict unseen driving scenes with self-produced trajectories.
>
---
#### [replaced 048] When Words Smile: Generating Diverse Emotional Facial Expressions from Text
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02508v3](http://arxiv.org/pdf/2412.02508v3)**

> **作者:** Haidong Xu; Meishan Zhang; Hao Ju; Zhedong Zheng; Erik Cambria; Min Zhang; Hao Fei
>
> **备注:** 19 pages. Resources: https://github.com/WalkerMitty/EmoAva
>
> **摘要:** Enabling digital humans to express rich emotions has significant applications in dialogue systems, gaming, and other interactive scenarios. While recent advances in talking head synthesis have achieved impressive results in lip synchronization, they tend to overlook the rich and dynamic nature of facial expressions. To fill this critical gap, we introduce an end-to-end text-to-expression model that explicitly focuses on emotional dynamics. Our model learns expressive facial variations in a continuous latent space and generates expressions that are diverse, fluid, and emotionally coherent. To support this task, we introduce EmoAva, a large-scale and high-quality dataset containing 15,000 text-3D expression pairs. Extensive experiments on both existing datasets and EmoAva demonstrate that our method significantly outperforms baselines across multiple evaluation metrics, marking a significant advancement in the field.
>
---
#### [replaced 049] An Inversion-based Measure of Memorization for Diffusion Models
- **分类: cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.05846v3](http://arxiv.org/pdf/2405.05846v3)**

> **作者:** Zhe Ma; Qingming Li; Xuhong Zhang; Tianyu Du; Ruixiao Lin; Zonghui Wang; Shouling Ji; Wenzhi Chen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** The past few years have witnessed substantial advances in image generation powered by diffusion models. However, it was shown that diffusion models are susceptible to training data memorization, raising significant concerns regarding copyright infringement and privacy invasion. This study delves into a rigorous analysis of memorization in diffusion models. We introduce InvMM, an inversion-based measure of memorization, which is based on inverting a sensitive latent noise distribution accounting for the replication of an image. For accurate estimation of the measure, we propose an adaptive algorithm that balances the normality and sensitivity of the noise distribution. Comprehensive experiments across four datasets, conducted on both unconditional and text-guided diffusion models, demonstrate that InvMM provides a reliable and complete quantification of memorization. Notably, InvMM is commensurable between samples, reveals the true extent of memorization from an adversarial standpoint and implies how memorization differs from membership. In practice, it serves as an auditing tool for developers to reliably assess the risk of memorization, thereby contributing to the enhancement of trustworthiness and privacy-preserving capabilities of diffusion models.
>
---
#### [replaced 050] Dual-Stream Global-Local Feature Collaborative Representation Network for Scene Classification of Mining Area
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20216v2](http://arxiv.org/pdf/2507.20216v2)**

> **作者:** Shuqi Fan; Haoyi Wang; Xianju Li
>
> **备注:** Accepted to IJCNN 2025
>
> **摘要:** Scene classification of mining areas provides accurate foundational data for geological environment monitoring and resource development planning. This study fuses multi-source data to construct a multi-modal mine land cover scene classification dataset. A significant challenge in mining area classification lies in the complex spatial layout and multi-scale characteristics. By extracting global and local features, it becomes possible to comprehensively reflect the spatial distribution, thereby enabling a more accurate capture of the holistic characteristics of mining scenes. We propose a dual-branch fusion model utilizing collaborative representation to decompose global features into a set of key semantic vectors. This model comprises three key components:(1) Multi-scale Global Transformer Branch: It leverages adjacent large-scale features to generate global channel attention features for small-scale features, effectively capturing the multi-scale feature relationships. (2) Local Enhancement Collaborative Representation Branch: It refines the attention weights by leveraging local features and reconstructed key semantic sets, ensuring that the local context and detailed characteristics of the mining area are effectively integrated. This enhances the model's sensitivity to fine-grained spatial variations. (3) Dual-Branch Deep Feature Fusion Module: It fuses the complementary features of the two branches to incorporate more scene information. This fusion strengthens the model's ability to distinguish and classify complex mining landscapes. Finally, this study employs multi-loss computation to ensure a balanced integration of the modules. The overall accuracy of this model is 83.63%, which outperforms other comparative models. Additionally, it achieves the best performance across all other evaluation metrics.
>
---
#### [replaced 051] Robust Adverse Weather Removal via Spectral-based Spatial Grouping
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.22498v2](http://arxiv.org/pdf/2507.22498v2)**

> **作者:** Yuhwan Jeong; Yunseo Yang; Youngho Yoon; Kuk-Jin Yoon
>
> **备注:** accepted by ICCV25
>
> **摘要:** Adverse weather conditions cause diverse and complex degradation patterns, driving the development of All-in-One (AiO) models. However, recent AiO solutions still struggle to capture diverse degradations, since global filtering methods like direct operations on the frequency domain fail to handle highly variable and localized distortions. To address these issue, we propose Spectral-based Spatial Grouping Transformer (SSGformer), a novel approach that leverages spectral decomposition and group-wise attention for multi-weather image restoration. SSGformer decomposes images into high-frequency edge features using conventional edge detection and low-frequency information via Singular Value Decomposition. We utilize multi-head linear attention to effectively model the relationship between these features. The fused features are integrated with the input to generate a grouping-mask that clusters regions based on the spatial similarity and image texture. To fully leverage this mask, we introduce a group-wise attention mechanism, enabling robust adverse weather removal and ensuring consistent performance across diverse weather conditions. We also propose a Spatial Grouping Transformer Block that uses both channel attention and spatial attention, effectively balancing feature-wise relationships and spatial dependencies. Extensive experiments show the superiority of our approach, validating its effectiveness in handling the varied and intricate adverse weather degradations.
>
---
#### [replaced 052] Recovering Partially Corrupted Objects via Sketch-Guided Bidirectional Feature Interaction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07047v2](http://arxiv.org/pdf/2503.07047v2)**

> **作者:** Yongle Zhang; Yimin Liu; Yan Huang; Qiang Wu
>
> **备注:** 13 pages. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Text-guided diffusion models have achieved remarkable success in object inpainting by providing high-level semantic guidance through text prompts. However, they often lack precise pixel-level spatial control, especially in scenarios involving partially corrupted objects where critical uncorrupted cues remain. To overcome this limitation, sketch-guided methods have been introduced, using either indirect gradient modulation or direct sketch injection to improve structural control. Yet, existing approaches typically establish a one-way mapping from the sketch to the masked regions only, neglecting the contextual information from unmasked object areas. This leads to a disconnection between the sketch and the uncorrupted content, thereby causing sketch-guided inconsistency and structural mismatch. To tackle this challenge, we propose a sketch-guided bidirectional feature interaction framework built upon a pretrained Stable Diffusion model. Our bidirectional interaction features two complementary directions, context-to-sketch and sketch-to-inpainting, that enable fine-grained spatial control for partially corrupted object inpainting. In the context-to-sketch direction, multi-scale latents from uncorrupted object regions are propagated to the sketch branch to generate a visual mask that adapts the sketch features to the visible context and denoising progress. In the sketch-to-inpainting direction, a sketch-conditional affine transformation modulates the influence of sketch guidance based on the learned visual mask, ensuring consistency with uncorrupted object content. This interaction is applied at multiple scales within the encoder of the diffusion U-Net, enabling the model to restore object structures with enhanced spatial fidelity. Extensive experiments on two newly constructed benchmark datasets demonstrate that our approach outperforms state-of-the-art methods.
>
---
#### [replaced 053] LiteGS: A High-performance Framework to Train 3DGS in Subminutes via System and Algorithm Codesign
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.01199v2](http://arxiv.org/pdf/2503.01199v2)**

> **作者:** Kaimin Liao; Hua Wang; Zhi Chen; Luchao Wang; Yaohua Tang
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as promising alternative in 3D representation. However, it still suffers from high training cost. This paper introduces LiteGS, a high performance framework that systematically optimizes the 3DGS training pipeline from multiple aspects. At the low-level computation layer, we design a ``warp-based raster'' associated with two hardware-aware optimizations to significantly reduce gradient reduction overhead. At the mid-level data management layer, we introduce dynamic spatial sorting based on Morton coding to enable a performant ``Cluster-Cull-Compact'' pipeline and improve data locality, therefore reducing cache misses. At the top-level algorithm layer, we establish a new robust densification criterion based on the variance of the opacity gradient, paired with a more stable opacity control mechanism, to achieve more precise parameter growth. Experimental results demonstrate that LiteGS accelerates the original 3DGS training by up to 13.4x with comparable or superior quality and surpasses the current SOTA in lightweight models by up to 1.4x speedup. For high-quality reconstruction tasks, LiteGS sets a new accuracy record and decreases the training time by an order of magnitude.
>
---
#### [replaced 054] LidaRefer: Context-aware Outdoor 3D Visual Grounding for Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.04351v2](http://arxiv.org/pdf/2411.04351v2)**

> **作者:** Yeong-Seung Baek; Heung-Seon Oh
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** 3D visual grounding (VG) aims to locate objects or regions within 3D scenes guided by natural language descriptions. While indoor 3D VG has advanced, outdoor 3D VG remains underexplored due to two challenges: (1) large-scale outdoor LiDAR scenes are dominated by background points and contain limited foreground information, making cross-modal alignment and contextual understanding more difficult; and (2) most outdoor datasets lack spatial annotations for referential non-target objects, which hinders explicit learning of referential context. To this end, we propose LidaRefer, a context-aware 3D VG framework for outdoor scenes. LidaRefer incorporates an object-centric feature selection strategy to focus on semantically relevant visual features while reducing computational overhead. Then, its transformer-based encoder-decoder architecture excels at establishing fine-grained cross-modal alignment between refined visual features and word-level text features, and capturing comprehensive global context. Additionally, we present Discriminative-Supportive Collaborative localization (DiSCo), a novel supervision strategy that explicitly models spatial relationships between target, contextual, and ambiguous objects for accurate target identification. To enable this without manual labeling, we introduce a pseudo-labeling approach that retrieves 3D localization labels for referential non-target objects. LidaRefer achieves state-of-the-art performance on Talk2Car-3D dataset under various evaluation settings.
>
---
#### [replaced 055] PatchTraj: Unified Time-Frequency Representation Learning via Dynamic Patches for Trajectory Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.19119v3](http://arxiv.org/pdf/2507.19119v3)**

> **作者:** Yanghong Liu; Xingping Dong; Ming Li; Weixing Zhang; Yidong Lou
>
> **摘要:** Pedestrian trajectory prediction is crucial for autonomous driving and robotics. While existing point-based and grid-based methods expose two main limitations: insufficiently modeling human motion dynamics, as they fail to balance local motion details with long-range spatiotemporal dependencies, and the time representations lack interaction with their frequency components in jointly modeling trajectory sequences. To address these challenges, we propose PatchTraj, a dynamic patch-based framework that integrates time-frequency joint modeling for trajectory prediction. Specifically, we decompose the trajectory into raw time sequences and frequency components, and employ dynamic patch partitioning to perform multi-scale segmentation, capturing hierarchical motion patterns. Each patch undergoes adaptive embedding with scale-aware feature extraction, followed by hierarchical feature aggregation to model both fine-grained and long-range dependencies. The outputs of the two branches are further enhanced via cross-modal attention, facilitating complementary fusion of temporal and spectral cues. The resulting enhanced embeddings exhibit strong expressive power, enabling accurate predictions even when using a vanilla Transformer architecture. Extensive experiments on ETH-UCY, SDD, NBA, and JRDB datasets demonstrate that our method achieves state-of-the-art performance. Notably, on the egocentric JRDB dataset, PatchTraj attains significant relative improvements of 26.7% in ADE and 17.4% in FDE, underscoring its substantial potential in embodied intelligence.
>
---
#### [replaced 056] Generalizable Image Repair for Robust Visual Control
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05911v2](http://arxiv.org/pdf/2503.05911v2)**

> **作者:** Carson Sobolewski; Zhenjiang Mao; Kshitij Maruti Vejre; Ivan Ruchkin
>
> **备注:** 8 pages, 4 figures, 2 tables, 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Vision-based control relies on accurate perception to achieve robustness. However, image distribution changes caused by sensor noise, adverse weather, and dynamic lighting can degrade perception, leading to suboptimal control decisions. Existing approaches, including domain adaptation and adversarial training, improve robustness but struggle to generalize to unseen corruptions while introducing computational overhead. To address this challenge, we propose a real-time image repair module that restores corrupted images before they are used by the controller. Our method leverages generative adversarial models, specifically CycleGAN and pix2pix, for image repair. CycleGAN enables unpaired image-to-image translation to adapt to novel corruptions, while pix2pix exploits paired image data when available to improve the quality. To ensure alignment with control performance, we introduce a control-focused loss function that prioritizes perceptual consistency in repaired images. We evaluated our method in a simulated autonomous racing environment with various visual corruptions. The results show that our approach significantly improves performance compared to baselines, mitigating distribution shift and enhancing controller reliability.
>
---
#### [replaced 057] PLMP -- Point-Line Minimal Problems for Projective SfM
- **分类: cs.CV; math.AG**

- **链接: [http://arxiv.org/pdf/2503.04351v2](http://arxiv.org/pdf/2503.04351v2)**

> **作者:** Kim Kiehn; Albin Ahlbäck; Kathlén Kohn
>
> **摘要:** We completely classify all minimal problems for Structure-from-Motion (SfM) where arrangements of points and lines are fully observed by multiple uncalibrated pinhole cameras. We find 291 minimal problems, 73 of which have unique solutions and can thus be solved linearly. Two of the linear problems allow an arbitrary number of views, while all other minimal problems have at most 9 cameras. All minimal problems have at most 7 points and at most 12 lines. We compute the number of solutions of each minimal problem, as this gives a measurement of the problem's intrinsic difficulty, and find that these number are relatively low (e.g., when comparing with minimal problems for calibrated cameras). Finally, by exploring stabilizer subgroups of subarrangements, we develop a geometric and systematic way to 1) factorize minimal problems into smaller problems, 2) identify minimal problems in underconstrained problems, and 3) formally prove non-minimality.
>
---
#### [replaced 058] VRM: Knowledge Distillation via Virtual Relation Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20760v3](http://arxiv.org/pdf/2502.20760v3)**

> **作者:** Weijia Zhang; Fei Xie; Weidong Cai; Chao Ma
>
> **备注:** Accepted by ICCV 2025 (Highlight)
>
> **摘要:** Knowledge distillation (KD) aims to transfer the knowledge of a more capable yet cumbersome teacher model to a lightweight student model. In recent years, relation-based KD methods have fallen behind, as their instance-matching counterparts dominate in performance. In this paper, we revive relational KD by identifying and tackling several key issues in relation-based methods, including their susceptibility to overfitting and spurious responses. Specifically, we transfer novelly constructed affinity graphs that compactly encapsulate a wealth of beneficial inter-sample, inter-class, and inter-view correlations by exploiting virtual views and relations as a new kind of knowledge. As a result, the student has access to richer guidance signals and stronger regularisation throughout the distillation process. To further mitigate the adverse impact of spurious responses, we prune the affinity graphs by dynamically detaching redundant and unreliable edges. Extensive experiments on CIFAR-100, ImageNet, and MS-COCO datasets demonstrate the superior performance of the proposed virtual relation matching (VRM) method, where it consistently sets new state-of-the-art records over a range of models, architectures, tasks, and set-ups. For instance, VRM for the first time hits 74.0% accuracy for ResNet50-to-MobileNetV2 distillation on ImageNet, and improves DeiT-T by 14.44% on CIFAR-100 with a ResNet56 teacher.
>
---
#### [replaced 059] Detecting Visual Information Manipulation Attacks in Augmented Reality: A Multimodal Semantic Reasoning Approach
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20356v2](http://arxiv.org/pdf/2507.20356v2)**

> **作者:** Yanming Xiu; Maria Gorlatova
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** The virtual content in augmented reality (AR) can introduce misleading or harmful information, leading to semantic misunderstandings or user errors. In this work, we focus on visual information manipulation (VIM) attacks in AR where virtual content changes the meaning of real-world scenes in subtle but impactful ways. We introduce a taxonomy that categorizes these attacks into three formats: character, phrase, and pattern manipulation, and three purposes: information replacement, information obfuscation, and extra wrong information. Based on the taxonomy, we construct a dataset, AR-VIM. It consists of 452 raw-AR video pairs spanning 202 different scenes, each simulating a real-world AR scenario. To detect such attacks, we propose a multimodal semantic reasoning framework, VIM-Sense. It combines the language and visual understanding capabilities of vision-language models (VLMs) with optical character recognition (OCR)-based textual analysis. VIM-Sense achieves an attack detection accuracy of 88.94% on AR-VIM, consistently outperforming vision-only and text-only baselines. The system reaches an average attack detection latency of 7.07 seconds in a simulated video processing framework and 7.17 seconds in a real-world evaluation conducted on a mobile Android AR application.
>
---
#### [replaced 060] ModalTune: Fine-Tuning Slide-Level Foundation Models with Multi-Modal Information for Multi-task Learning in Digital Pathology
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.17564v2](http://arxiv.org/pdf/2503.17564v2)**

> **作者:** Vishwesh Ramanathan; Tony Xu; Pushpak Pati; Faruk Ahmed; Maged Goubran; Anne L. Martel
>
> **摘要:** Prediction tasks in digital pathology are challenging due to the massive size of whole-slide images (WSIs) and the weak nature of training signals. Advances in computing, data availability, and self-supervised learning (SSL) have paved the way for slide-level foundation models (SLFMs) that can improve prediction tasks in low-data regimes. However, current methods under-utilize shared information between tasks and modalities. To overcome this challenge, we propose ModalTune, a novel fine-tuning framework which introduces the Modal Adapter to integrate new modalities without modifying SLFM weights. Additionally, we use large-language models (LLMs) to encode labels as text, capturing semantic relationships across multiple tasks and cancer types in a single training recipe. ModalTune achieves state-of-the-art (SOTA) results against both uni-modal and multi-modal models across four cancer types, jointly improving survival and cancer subtype prediction while remaining competitive in pan-cancer settings. Additionally, we show ModalTune is generalizable to two out-of-distribution (OOD) datasets. To our knowledge, this is the first unified fine-tuning framework for multi-modal, multi-task, and pan-cancer modeling in digital pathology.
>
---
#### [replaced 061] Diffusion Beats Autoregressive in Data-Constrained Settings
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.15857v4](http://arxiv.org/pdf/2507.15857v4)**

> **作者:** Mihir Prabhudesai; Mengning Wu; Amir Zadeh; Katerina Fragkiadaki; Deepak Pathak
>
> **备注:** Project Webpage: https://diffusion-scaling.github.io
>
> **摘要:** Autoregressive (AR) models have long dominated the landscape of large language models, driving progress across a wide range of tasks. Recently, diffusion-based language models have emerged as a promising alternative, though their advantages over AR models remain underexplored. In this paper, we systematically study masked diffusion models in data-constrained settings-where training involves repeated passes over limited data-and find that they significantly outperform AR models when compute is abundant but data is scarce. Diffusion models make better use of repeated data, achieving lower validation loss and superior downstream performance. We interpret this advantage as implicit data augmentation: masked diffusion exposes the model to a diverse distribution of token orderings and prediction tasks, unlike AR's fixed left-to-right factorization. We find new scaling laws for diffusion models and derive a closed-form expression for the critical compute threshold at which diffusion begins to outperform AR. These results suggest that when data, not compute, is the bottleneck, diffusion models offer a compelling alternative to the standard AR paradigm. Our code is available at: https://diffusion-scaling.github.io.
>
---
#### [replaced 062] Learning to Align and Refine: A Foundation-to-Diffusion Framework for Occlusion-Robust Two-Hand Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.17788v2](http://arxiv.org/pdf/2503.17788v2)**

> **作者:** Gaoge Han; Yongkang Cheng; Zhe Chen; Shaoli Huang; Tongliang Liu
>
> **摘要:** Two-hand reconstruction from monocular images faces persistent challenges due to complex and dynamic hand postures and occlusions, causing significant difficulty in achieving plausible interaction alignment. Existing approaches struggle with such alignment issues, often resulting in misalignment and penetration artifacts. To tackle this, we propose a dual-stage Foundation-to-Diffusion framework that precisely align 2D prior guidance from vision foundation models and diffusion-based generative 3D interaction refinement to achieve occlusion-robust two-hand reconstruction. First, we introduce a lightweight fusion alignment encoder that aligns fused multimodal 2D priors like key points, segmentation maps, and depth cues from vision foundation models during training. This provides robust structured guidance, further enabling efficient inference without heavy foundation model encoders at test time while maintaining high reconstruction accuracy. Second, we implement a two-hand diffusion model explicitly trained to convert interpenetrated 3D poses into plausible, penetration-free counterparts. Through collision gradient-guided denoising, the model rectifies artifacts while preserving natural spatial relationships between hands. Extensive evaluations demonstrate that our method achieves state-of-the-art performance on InterHand2.6M, HIC, and FreiHAND datasets, significantly advancing occlusion handling and interaction robustness. Our code will be publicly released.
>
---
#### [replaced 063] Comparative Performance of Finetuned ImageNet Pre-trained Models for Electronic Component Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19330v2](http://arxiv.org/pdf/2506.19330v2)**

> **作者:** Yidi Shao; Longfei Zhou; Fangshuo Tang; Xinyi Shi; Dalang Chen; Shengtao Xia
>
> **备注:** Due to issues related to author order and some problems in the current version regarding methodology, we would like to withdraw the preprint to avoid potential conflicts
>
> **摘要:** Electronic component classification and detection are crucial in manufacturing industries, significantly reducing labor costs and promoting technological and industrial development. Pre-trained models, especially those trained on ImageNet, are highly effective in image classification, allowing researchers to achieve excellent results even with limited data. This paper compares the performance of twelve ImageNet pre-trained models in classifying electronic components. Our findings show that all models tested delivered respectable accuracies. MobileNet-V2 recorded the highest at 99.95%, while EfficientNet-B0 had the lowest at 92.26%. These results underscore the substantial benefits of using ImageNet pre-trained models in image classification tasks and confirm the practical applicability of these methods in the electronics manufacturing sector.
>
---
#### [replaced 064] EgoOops: A Dataset for Mistake Action Detection from Egocentric Videos referring to Procedural Texts
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.05343v3](http://arxiv.org/pdf/2410.05343v3)**

> **作者:** Yuto Haneji; Taichi Nishimura; Hirotaka Kameko; Keisuke Shirai; Tomoya Yoshida; Keiya Kajimura; Koki Yamamoto; Taiyu Cui; Tomohiro Nishimoto; Shinsuke Mori
>
> **备注:** Main 8 pages, supplementary 6 pages
>
> **摘要:** Mistake action detection is crucial for developing intelligent archives that detect workers' errors and provide feedback. Existing studies have focused on visually apparent mistakes in free-style activities, resulting in video-only approaches to mistake detection. However, in text-following activities, models cannot determine the correctness of some actions without referring to the texts. Additionally, current mistake datasets rarely use procedural texts for video recording except for cooking. To fill these gaps, this paper proposes the EgoOops dataset, where egocentric videos record erroneous activities when following procedural texts across diverse domains. It features three types of annotations: video-text alignment, mistake labels, and descriptions for mistakes. We also propose a mistake detection approach, combining video-text alignment and mistake label classification to leverage the texts. Our experimental results show that incorporating procedural texts is essential for mistake detection. Data is available through https://y-haneji.github.io/EgoOops-project-page/.
>
---
#### [replaced 065] EaqVLA: Encoding-aligned Quantization for Vision-Language-Action Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21567v2](http://arxiv.org/pdf/2505.21567v2)**

> **作者:** Feng Jiang; Zihao Zheng; Xiuping Cui; Maoliang Li; JIayu Chen; Xiang Chen
>
> **备注:** There is an error in this paper, and as the author, I request retraction
>
> **摘要:** With the development of Embodied Artificial intelligence, the end-to-end control policy such as Vision-Language-Action (VLA) model has become the mainstream. Existing VLA models faces expensive computing/storage cost, which need to be optimized. Quantization is considered as the most effective method which can not only reduce the memory cost but also achieve computation acceleration. However, we find the token alignment of VLA models hinders the application of existing quantization methods. To address this, we proposed an optimized framework called EaqVLA, which apply encoding-aligned quantization to VLA models. Specifically, we propose an complete analysis method to find the misalignment in various granularity. Based on the analysis results, we propose a mixed precision quantization with the awareness of encoding alignment. Experiments shows that the porposed EaqVLA achieves better quantization performance (with the minimal quantization loss for end-to-end action control and xxx times acceleration) than existing quantization methods.
>
---
#### [replaced 066] BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12620v4](http://arxiv.org/pdf/2505.12620v4)**

> **作者:** Haiquan Wen; Yiwei He; Zhenglin Huang; Tianxiao Li; Zihan Yu; Xingru Huang; Lu Qi; Baoyuan Wu; Xiangtai Li; Guangliang Cheng
>
> **摘要:** Advances in AI generative models facilitate super-realistic video synthesis, amplifying misinformation risks via social media and eroding trust in digital content. Several research works have explored new deepfake detection methods on AI-generated images to alleviate these risks. However, with the fast development of video generation models, such as Sora and WanX, there is currently a lack of large-scale, high-quality AI-generated video datasets for forgery detection. In addition, existing detection approaches predominantly treat the task as binary classification, lacking explainability in model decision-making and failing to provide actionable insights or guidance for the public. To address these challenges, we propose \textbf{GenBuster-200K}, a large-scale AI-generated video dataset featuring 200K high-resolution video clips, diverse latest generative techniques, and real-world scenes. We further introduce \textbf{BusterX}, a novel AI-generated video detection and explanation framework leveraging multimodal large language model (MLLM) and reinforcement learning for authenticity determination and explainable rationale. To our knowledge, GenBuster-200K is the {\it \textbf{first}} large-scale, high-quality AI-generated video dataset that incorporates the latest generative techniques for real-world scenarios. BusterX is the {\it \textbf{first}} framework to integrate MLLM with reinforcement learning for explainable AI-generated video detection. Extensive comparisons with state-of-the-art methods and ablation studies validate the effectiveness and generalizability of BusterX. The code, models, and datasets will be released.
>
---
#### [replaced 067] Priority-Aware Clinical Pathology Hierarchy Training for Multiple Instance Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20469v2](http://arxiv.org/pdf/2507.20469v2)**

> **作者:** Sungrae Hong; Kyungeun Kim; Juhyeon Kim; Sol Lee; Jisu Shin; Chanjae Song; Mun Yong Yi
>
> **备注:** 10 pages, 4 figures, Accepted for oral presentation by The 2nd MICCAI Student Board (MSB) EMERGE Workshop
>
> **摘要:** Multiple Instance Learning (MIL) is increasingly being used as a support tool within clinical settings for pathological diagnosis decisions, achieving high performance and removing the annotation burden. However, existing approaches for clinical MIL tasks have not adequately addressed the priority issues that exist in relation to pathological symptoms and diagnostic classes, causing MIL models to ignore priority among classes. To overcome this clinical limitation of MIL, we propose a new method that addresses priority issues using two hierarchies: vertical inter-hierarchy and horizontal intra-hierarchy. The proposed method aligns MIL predictions across each hierarchical level and employs an implicit feature re-usability during training to facilitate clinically more serious classes within the same level. Experiments with real-world patient data show that the proposed method effectively reduces misdiagnosis and prioritizes more important symptoms in multiclass scenarios. Further analysis verifies the efficacy of the proposed components and qualitatively confirms the MIL predictions against challenging cases with multiple symptoms.
>
---
#### [replaced 068] DHCP: Detecting Hallucinations by Cross-modal Attention Pattern in Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.18659v2](http://arxiv.org/pdf/2411.18659v2)**

> **作者:** Yudong Zhang; Ruobing Xie; Xingwu Sun; Yiqing Huang; Jiansheng Chen; Zhanhui Kang; Di Wang; Yu Wang
>
> **备注:** Accepted by ACM Multimedia 2025
>
> **摘要:** Large vision-language models (LVLMs) have demonstrated exceptional performance on complex multimodal tasks. However, they continue to suffer from significant hallucination issues, including object, attribute, and relational hallucinations. To accurately detect these hallucinations, we investigated the variations in cross-modal attention patterns between hallucination and non-hallucination states. Leveraging these distinctions, we developed a lightweight detector capable of identifying hallucinations. Our proposed method, Detecting Hallucinations by Cross-modal Attention Patterns (DHCP), is straightforward and does not require additional LVLM training or extra LVLM inference steps. Experimental results show that DHCP achieves remarkable performance in hallucination detection. By offering novel insights into the identification and analysis of hallucinations in LVLMs, DHCP contributes to advancing the reliability and trustworthiness of these models. The code is available at https://github.com/btzyd/DHCP.
>
---
#### [replaced 069] DeepForest: Sensing Into Self-Occluding Volumes of Vegetation With Aerial Imaging
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.02171v3](http://arxiv.org/pdf/2502.02171v3)**

> **作者:** Mohamed Youssef; Jian Peng; Oliver Bimber
>
> **摘要:** Access to below-canopy volumetric vegetation data is crucial for understanding ecosystem dynamics. We address the long-standing limitation of remote sensing to penetrate deep into dense canopy layers. LiDAR and radar are currently considered the primary options for measuring 3D vegetation structures, while cameras can only extract the reflectance and depth of top layers. Using conventional, high-resolution aerial images, our approach allows sensing deep into self-occluding vegetation volumes, such as forests. It is similar in spirit to the imaging process of wide-field microscopy, but can handle much larger scales and strong occlusion. We scan focal stacks by synthetic-aperture imaging with drones and reduce out-of-focus signal contributions using pre-trained 3D convolutional neural networks with mean squared error (MSE) as the loss function. The resulting volumetric reflectance stacks contain low-frequency representations of the vegetation volume. Combining multiple reflectance stacks from various spectral channels provides insights into plant health, growth, and environmental conditions throughout the entire vegetation volume. Compared with simulated ground truth, our correction leads to ~x7 average improvements (min: ~x2, max: ~x12) for forest densities of 220 trees/ha - 1680 trees/ha. In our field experiment, we achieved an MSE of 0.05 when comparing with the top-vegetation layer that was measured with classical multispectral aerial imaging.
>
---
#### [replaced 070] HRVVS: A High-resolution Video Vasculature Segmentation Network via Hierarchical Autoregressive Residual Priors
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.22530v2](http://arxiv.org/pdf/2507.22530v2)**

> **作者:** Xincheng Yao; Yijun Yang; Kangwei Guo; Ruiqiang Xiao; Haipeng Zhou; Haisu Tao; Jian Yang; Lei Zhu
>
> **备注:** Accepted by MICCAI 2025
>
> **摘要:** The segmentation of the hepatic vasculature in surgical videos holds substantial clinical significance in the context of hepatectomy procedures. However, owing to the dearth of an appropriate dataset and the inherently complex task characteristics, few researches have been reported in this domain. To address this issue, we first introduce a high quality frame-by-frame annotated hepatic vasculature dataset containing 35 long hepatectomy videos and 11442 high-resolution frames. On this basis, we propose a novel high-resolution video vasculature segmentation network, dubbed as HRVVS. We innovatively embed a pretrained visual autoregressive modeling (VAR) model into different layers of the hierarchical encoder as prior information to reduce the information degradation generated during the downsampling process. In addition, we designed a dynamic memory decoder on a multi-view segmentation network to minimize the transmission of redundant information while preserving more details between frames. Extensive experiments on surgical video datasets demonstrate that our proposed HRVVS significantly outperforms the state-of-the-art methods. The source code and dataset will be publicly available at \{https://github.com/scott-yjyang/HRVVS}.
>
---
#### [replaced 071] Ultra3D: Efficient and High-Fidelity 3D Generation with Part Attention
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.17745v3](http://arxiv.org/pdf/2507.17745v3)**

> **作者:** Yiwen Chen; Zhihao Li; Yikai Wang; Hu Zhang; Qin Li; Chi Zhang; Guosheng Lin
>
> **备注:** Project Page: https://buaacyw.github.io/ultra3d/
>
> **摘要:** Recent advances in sparse voxel representations have significantly improved the quality of 3D content generation, enabling high-resolution modeling with fine-grained geometry. However, existing frameworks suffer from severe computational inefficiencies due to the quadratic complexity of attention mechanisms in their two-stage diffusion pipelines. In this work, we propose Ultra3D, an efficient 3D generation framework that significantly accelerates sparse voxel modeling without compromising quality. Our method leverages the compact VecSet representation to efficiently generate a coarse object layout in the first stage, reducing token count and accelerating voxel coordinate prediction. To refine per-voxel latent features in the second stage, we introduce Part Attention, a geometry-aware localized attention mechanism that restricts attention computation within semantically consistent part regions. This design preserves structural continuity while avoiding unnecessary global attention, achieving up to 6.7x speed-up in latent generation. To support this mechanism, we construct a scalable part annotation pipeline that converts raw meshes into part-labeled sparse voxels. Extensive experiments demonstrate that Ultra3D supports high-resolution 3D generation at 1024 resolution and achieves state-of-the-art performance in both visual fidelity and user preference.
>
---
#### [replaced 072] Butter: Frequency Consistency and Hierarchical Fusion for Autonomous Driving Object Detection
- **分类: cs.CV; I.4.8; I.2.10; H.5.1; I.2.6**

- **链接: [http://arxiv.org/pdf/2507.13373v2](http://arxiv.org/pdf/2507.13373v2)**

> **作者:** Xiaojian Lin; Wenxin Zhang; Yuchu Jiang; Wangyu Wu; Yiran Guo; Kangxu Wang; Zongzheng Zhang; Guijin Wang; Lei Jin; Hao Zhao
>
> **备注:** 10 pages, 6 figures. Supplementary material: 8 pages, 7 figures. Accepted at ACM Multimedia 2025
>
> **摘要:** Hierarchical feature representations play a pivotal role in computer vision, particularly in object detection for autonomous driving. Multi-level semantic understanding is crucial for accurately identifying pedestrians, vehicles, and traffic signs in dynamic environments. However, existing architectures, such as YOLO and DETR, struggle to maintain feature consistency across different scales while balancing detection precision and computational efficiency. To address these challenges, we propose Butter, a novel object detection framework designed to enhance hierarchical feature representations for improving detection robustness. Specifically, Butter introduces two key innovations: Frequency-Adaptive Feature Consistency Enhancement (FAFCE) Component, which refines multi-scale feature consistency by leveraging adaptive frequency filtering to enhance structural and boundary precision, and Progressive Hierarchical Feature Fusion Network (PHFFNet) Module, which progressively integrates multi-level features to mitigate semantic gaps and strengthen hierarchical feature learning. Through extensive experiments on BDD100K, KITTI, and Cityscapes, Butter demonstrates superior feature representation capabilities, leading to notable improvements in detection accuracy while reducing model complexity. By focusing on hierarchical feature refinement and integration, Butter provides an advanced approach to object detection that achieves a balance between accuracy, deployability, and computational efficiency in real-time autonomous driving scenarios. Our model and implementation are publicly available at https://github.com/Aveiro-Lin/Butter, facilitating further research and validation within the autonomous driving community.
>
---
#### [replaced 073] BusterX++: Towards Unified Cross-Modal AI-Generated Content Detection and Explanation with MLLM
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.14632v2](http://arxiv.org/pdf/2507.14632v2)**

> **作者:** Haiquan Wen; Tianxiao Li; Zhenglin Huang; Yiwei He; Guangliang Cheng
>
> **摘要:** Recent advances in generative AI have dramatically improved image and video synthesis capabilities, significantly increasing the risk of misinformation through sophisticated fake content. In response, detection methods have evolved from traditional approaches to multimodal large language models (MLLMs), offering enhanced transparency and interpretability in identifying synthetic media. However, current detection systems remain fundamentally limited by their single-modality design. These approaches analyze images or videos separately, making them ineffective against synthetic content that combines multiple media formats. To address these challenges, we introduce \textbf{BusterX++}, a novel framework designed specifically for cross-modal detection and explanation of synthetic media. Our approach incorporates an advanced reinforcement learning (RL) post-training strategy that eliminates cold-start. Through Multi-stage Training, Thinking Reward, and Hybrid Reasoning, BusterX++ achieves stable and substantial performance improvements. To enable comprehensive evaluation, we also present \textbf{GenBuster++}, a cross-modal benchmark leveraging state-of-the-art image and video generation techniques. This benchmark comprises 4,000 images and video clips, meticulously curated by human experts using a novel filtering methodology to ensure high quality, diversity, and real-world applicability. Extensive experiments demonstrate the effectiveness and generalizability of our approach.
>
---
#### [replaced 074] HER2 Expression Prediction with Flexible Multi-Modal Inputs via Dynamic Bidirectional Reconstruction
- **分类: cs.MM; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.10006v2](http://arxiv.org/pdf/2506.10006v2)**

> **作者:** Jie Qin; Wei Yang; Yan Su; Yiran Zhu; Weizhen Li; Yunyue Pan; Chengchang Pan; Honggang Qi
>
> **备注:** 8 pages,6 figures,3 tables,accepted by the 33rd ACM International Conference on Multimedia(ACM MM 2025)
>
> **摘要:** In breast cancer HER2 assessment, clinical evaluation relies on combined H&E and IHC images, yet acquiring both modalities is often hindered by clinical constraints and cost. We propose an adaptive bimodal prediction framework that flexibly supports single- or dual-modality inputs through two core innovations: a dynamic branch selector activating modality completion or joint inference based on input availability, and a cross-modal GAN (CM-GAN) enabling feature-space reconstruction of missing modalities. This design dramatically improves H&E-only accuracy from 71.44% to 94.25%, achieves 95.09% with full dual-modality inputs, and maintains 90.28% reliability under single-modality conditions. The "dual-modality preferred, single-modality compatible" architecture delivers near-dual-modality accuracy without mandatory synchronized acquisition, offering a cost-effective solution for resource-limited regions and significantly improving HER2 assessment accessibility.
>
---
#### [replaced 075] Exploiting Scale-Variant Attention for Segmenting Small Medical Objects
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.07720v5](http://arxiv.org/pdf/2407.07720v5)**

> **作者:** Wei Dai; Rui Liu; Zixuan Wu; Tianyi Wu; Min Wang; Junxian Zhou; Yixuan Yuan; Jun Liu
>
> **备注:** 14 pages, 9 figures, under review
>
> **摘要:** Early detection and accurate diagnosis can predict the risk of malignant disease transformation, thereby increasing the probability of effective treatment. Identifying mild syndrome with small pathological regions serves as an ominous warning and is fundamental in the early diagnosis of diseases. While deep learning algorithms, particularly convolutional neural networks (CNNs), have shown promise in segmenting medical objects, analyzing small areas in medical images remains challenging. This difficulty arises due to information losses and compression defects from convolution and pooling operations in CNNs, which become more pronounced as the network deepens, especially for small medical objects. To address these challenges, we propose a novel scale-variant attention-based network (SvANet) for accurately segmenting small-scale objects in medical images. The SvANet consists of scale-variant attention, cross-scale guidance, Monte Carlo attention, and vision transformer, which incorporates cross-scale features and alleviates compression artifacts for enhancing the discrimination of small medical objects. Quantitative experimental results demonstrate the superior performance of SvANet, achieving 96.12%, 96.11%, 89.79%, 84.15%, 80.25%, 73.05%, and 72.58% in mean Dice coefficient for segmenting kidney tumors, skin lesions, hepatic tumors, polyps, surgical excision cells, retinal vasculatures, and sperms, which occupy less than 1% of the image areas in KiTS23, ISIC 2018, ATLAS, PolypGen, TissueNet, FIVES, and SpermHealth datasets, respectively.
>
---
#### [replaced 076] Collaborative Perceiver: Elevating Vision-based 3D Object Detection via Local Density-Aware Spatial Occupancy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.21358v3](http://arxiv.org/pdf/2507.21358v3)**

> **作者:** Jicheng Yuan; Manh Nguyen Duc; Qian Liu; Manfred Hauswirth; Danh Le Phuoc
>
> **备注:** The manuscript has been accepted by ICONIP2025
>
> **摘要:** Vision-based bird's-eye-view (BEV) 3D object detection has advanced significantly in autonomous driving by offering cost-effectiveness and rich contextual information. However, existing methods often construct BEV representations by collapsing extracted object features, neglecting intrinsic environmental contexts, such as roads and pavements. This hinders detectors from comprehensively perceiving the characteristics of the physical world. To alleviate this, we introduce a multi-task learning framework, Collaborative Perceiver (CoP), that leverages spatial occupancy as auxiliary information to mine consistent structural and conceptual similarities shared between 3D object detection and occupancy prediction tasks, bridging gaps in spatial representations and feature refinement. To this end, we first propose a pipeline to generate dense occupancy ground truths incorporating local density information (LDO) for reconstructing detailed environmental information. Next, we employ a voxel-height-guided sampling (VHS) strategy to distill fine-grained local features according to distinct object properties. Furthermore, we develop a global-local collaborative feature fusion (CFF) module that seamlessly integrates complementary knowledge between both tasks, thus composing more robust BEV representations. Extensive experiments on the nuScenes benchmark demonstrate that CoP outperforms existing vision-based frameworks, achieving 49.5\% mAP and 59.2\% NDS on the test set. Code and supplementary materials are available at this link https://github.com/jichengyuan/Collaborative-Perceiver.
>
---
#### [replaced 077] RoCo-Sim: Enhancing Roadside Collaborative Perception through Foreground Simulation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10410v3](http://arxiv.org/pdf/2503.10410v3)**

> **作者:** Yuwen Du; Anning Hu; Zichen Chao; Yifan Lu; Junhao Ge; Genjia Liu; Weitao Wu; Lanjun Wang; Siheng Chen
>
> **摘要:** Roadside Collaborative Perception refers to a system where multiple roadside units collaborate to pool their perceptual data, assisting vehicles in enhancing their environmental awareness. Existing roadside perception methods concentrate on model design but overlook data issues like calibration errors, sparse information, and multi-view consistency, leading to poor performance on recent published datasets. To significantly enhance roadside collaborative perception and address critical data issues, we present the first simulation framework RoCo-Sim for road-side collaborative perception. RoCo-Sim is capable of generating diverse, multi-view consistent simulated roadside data through dynamic foreground editing and full-scene style transfer of a single image. RoCo-Sim consists of four components: (1) Camera Extrinsic Optimization ensures accurate 3D to 2D projection for roadside cameras; (2) A novel Multi-View Occlusion-Aware Sampler (MOAS) determines the placement of diverse digital assets within 3D space; (3) DepthSAM innovatively models foreground-background relationships from single-frame fixed-view images, ensuring multi-view consistency of foreground; and (4) Scalable Post-Processing Toolkit generates more realistic and enriched scenes through style transfer and other enhancements. RoCo-Sim significantly improves roadside 3D object detection, outperforming SOTA methods by 83.74 on Rcooper-Intersection and 83.12 on TUMTraf-V2X for AP70. RoCo-Sim fills a critical gap in roadside perception simulation. Code and pre-trained models will be released soon: https://github.com/duyuwen-duen/RoCo-Sim
>
---
#### [replaced 078] Meta CLIP 2: A Worldwide Scaling Recipe
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22062v2](http://arxiv.org/pdf/2507.22062v2)**

> **作者:** Yung-Sung Chuang; Yang Li; Dong Wang; Ching-Feng Yeh; Kehan Lyu; Ramya Raghavendra; James Glass; Lifei Huang; Jason Weston; Luke Zettlemoyer; Xinlei Chen; Zhuang Liu; Saining Xie; Wen-tau Yih; Shang-Wen Li; Hu Xu
>
> **备注:** 10 pages
>
> **摘要:** Contrastive Language-Image Pretraining (CLIP) is a popular foundation model, supporting from zero-shot classification, retrieval to encoders for multimodal large language models (MLLMs). Although CLIP is successfully trained on billion-scale image-text pairs from the English world, scaling CLIP's training further to learning from the worldwide web data is still challenging: (1) no curation method is available to handle data points from non-English world; (2) the English performance from existing multilingual CLIP is worse than its English-only counterpart, i.e., "curse of multilinguality" that is common in LLMs. Here, we present Meta CLIP 2, the first recipe training CLIP from scratch on worldwide web-scale image-text pairs. To generalize our findings, we conduct rigorous ablations with minimal changes that are necessary to address the above challenges and present a recipe enabling mutual benefits from English and non-English world data. In zero-shot ImageNet classification, Meta CLIP 2 ViT-H/14 surpasses its English-only counterpart by 0.8% and mSigLIP by 0.7%, and surprisingly sets new state-of-the-art without system-level confounding factors (e.g., translation, bespoke architecture changes) on multilingual benchmarks, such as CVQA with 57.4%, Babel-ImageNet with 50.2% and XM3600 with 64.3% on image-to-text retrieval.
>
---
#### [replaced 079] Uncovering Cultural Representation Disparities in Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14729v3](http://arxiv.org/pdf/2505.14729v3)**

> **作者:** Ram Mohan Rao Kadiyala; Siddhant Gupta; Jebish Purbey; Srishti Yadav; Suman Debnath; Alejandro Salamanca; Desmond Elliott
>
> **备注:** 28 pages, 36 figures
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated impressive capabilities across a range of tasks, yet concerns about their potential biases exist. This work investigates the extent to which prominent VLMs exhibit cultural biases by evaluating their performance on an image-based country identification task at a country level. Utilizing the geographically diverse Country211 dataset, we probe several large vision language models (VLMs) under various prompting strategies: open-ended questions, multiple-choice questions (MCQs) including challenging setups like multilingual and adversarial settings. Our analysis aims to uncover disparities in model accuracy across different countries and question formats, providing insights into how training data distribution and evaluation methodologies might influence cultural biases in VLMs. The findings highlight significant variations in performance, suggesting that while VLMs possess considerable visual understanding, they inherit biases from their pre-training data and scale that impact their ability to generalize uniformly across diverse global contexts.
>
---
#### [replaced 080] EP-Diffuser: An Efficient Diffusion Model for Traffic Scene Generation and Prediction via Polynomial Representations
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.05422v3](http://arxiv.org/pdf/2504.05422v3)**

> **作者:** Yue Yao; Mohamed-Khalil Bouzidi; Daniel Goehring; Joerg Reichardt
>
> **摘要:** As the prediction horizon increases, predicting the future evolution of traffic scenes becomes increasingly difficult due to the multi-modal nature of agent motion. Most state-of-the-art (SotA) prediction models primarily focus on forecasting the most likely future. However, for the safe operation of autonomous vehicles, it is equally important to cover the distribution for plausible motion alternatives. To address this, we introduce EP-Diffuser, a novel parameter-efficient diffusion-based generative model designed to capture the distribution of possible traffic scene evolutions. Conditioned on road layout and agent history, our model acts as a predictor and generates diverse, plausible scene continuations. We benchmark EP-Diffuser against two SotA models in terms of accuracy and plausibility of predictions on the Argoverse 2 dataset. Despite its significantly smaller model size, our approach achieves both highly accurate and plausible traffic scene predictions. We further evaluate model generalization ability in an out-of-distribution (OoD) test setting using Waymo Open dataset and show superior robustness of our approach. The code and model checkpoints are available at: https://github.com/continental/EP-Diffuser.
>
---
#### [replaced 081] Multi-Task Label Discovery via Hierarchical Task Tokens for Partially Annotated Dense Predictions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18823v2](http://arxiv.org/pdf/2411.18823v2)**

> **作者:** Jingdong Zhang; Hanrong Ye; Xin Li; Wenping Wang; Dan Xu
>
> **摘要:** In recent years, simultaneous learning of multiple dense prediction tasks with partially annotated label data has emerged as an important research area. Previous works primarily focus on leveraging cross-task relations or conducting adversarial training for extra regularization, which achieve promising performance improvements, while still suffering from the lack of direct pixel-wise supervision and extra training of heavy mapping networks. To effectively tackle this challenge, we propose a novel approach to optimize a set of compact learnable hierarchical task tokens, including global and fine-grained ones, to discover consistent pixel-wise supervision signals in both feature and prediction levels. Specifically, the global task tokens are designed for effective cross-task feature interactions in a global context. Then, a group of fine-grained task-specific spatial tokens for each task is learned from the corresponding global task tokens. It is embedded to have dense interactions with each task-specific feature map. The learned global and local fine-grained task tokens are further used to discover pseudo task-specific dense labels at different levels of granularity, and they can be utilized to directly supervise the learning of the multi-task dense prediction framework. Extensive experimental results on challenging NYUD-v2, Cityscapes, and PASCAL Context datasets demonstrate significant improvements over existing state-of-the-art methods for partially annotated multi-task dense prediction.
>
---
#### [replaced 082] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22351v3](http://arxiv.org/pdf/2503.22351v3)**

> **作者:** Byeongjun Kwon; Munchurl Kim
>
> **备注:** ICCV 2025 (camera-ready version). [Project page](https://kaist-viclab.github.io/One-Look-is-Enough_site)
>
> **摘要:** Zero-shot depth estimation (DE) models exhibit strong generalization performance as they are trained on large-scale datasets. However, existing models struggle with high-resolution images due to the discrepancy in image resolutions of training (with smaller resolutions) and inference (for high resolutions). Processing them at full resolution leads to decreased estimation accuracy on depth with tremendous memory consumption, while downsampling to the training resolution results in blurred edges in the estimated depth images. Prevailing high-resolution depth estimation methods adopt a patch-based approach, which introduces depth discontinuity issues when reassembling the estimated depth patches, resulting in test-time inefficiency. Additionally, to obtain fine-grained depth details, these methods rely on synthetic datasets due to the real-world sparse ground truth depth, leading to poor generalizability. To tackle these limitations, we propose Patch Refine Once (PRO), an efficient and generalizable tile-based framework. Our PRO consists of two key components: (i) Grouped Patch Consistency Training that enhances test-time efficiency while mitigating the depth discontinuity problem by jointly processing four overlapping patches and enforcing a consistency loss on their overlapping regions within a single backpropagation step, and (ii) Bias Free Masking that prevents the DE models from overfitting to dataset-specific biases, enabling better generalization to real-world datasets even after training on synthetic data. Zero-shot evaluations on Booster, ETH3D, Middlebury 2014, and NuScenes demonstrate that our PRO can be seamlessly integrated into existing depth estimation models.
>
---
#### [replaced 083] Human-in-the-Loop Local Corrections of 3D Scene Layouts via Infilling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11806v2](http://arxiv.org/pdf/2503.11806v2)**

> **作者:** Christopher Xie; Armen Avetisyan; Henry Howard-Jenkins; Yawar Siddiqui; Julian Straub; Richard Newcombe; Vasileios Balntas; Jakob Engel
>
> **备注:** Project page: https://www.projectaria.com/scenescript/
>
> **摘要:** We present a novel human-in-the-loop approach to estimate 3D scene layout that uses human feedback from an egocentric standpoint. We study this approach through introduction of a novel local correction task, where users identify local errors and prompt a model to automatically correct them. Building on SceneScript, a state-of-the-art framework for 3D scene layout estimation that leverages structured language, we propose a solution that structures this problem as "infilling", a task studied in natural language processing. We train a multi-task version of SceneScript that maintains performance on global predictions while significantly improving its local correction ability. We integrate this into a human-in-the-loop system, enabling a user to iteratively refine scene layout estimates via a low-friction "one-click fix'' workflow. Our system enables the final refined layout to diverge from the training distribution, allowing for more accurate modelling of complex layouts.
>
---
#### [replaced 084] LIDAR: Lightweight Adaptive Cue-Aware Fusion Vision Mamba for Multimodal Segmentation of Structural Cracks
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.22477v2](http://arxiv.org/pdf/2507.22477v2)**

> **作者:** Hui Liu; Chen Jia; Fan Shi; Xu Cheng; Mengfei Shi; Xia Xie; Shengyong Chen
>
> **备注:** This paper has been accepted by ACM MM 2025
>
> **摘要:** Achieving pixel-level segmentation with low computational cost using multimodal data remains a key challenge in crack segmentation tasks. Existing methods lack the capability for adaptive perception and efficient interactive fusion of cross-modal features. To address these challenges, we propose a Lightweight Adaptive Cue-Aware Vision Mamba network (LIDAR), which efficiently perceives and integrates morphological and textural cues from different modalities under multimodal crack scenarios, generating clear pixel-level crack segmentation maps. Specifically, LIDAR is composed of a Lightweight Adaptive Cue-Aware Visual State Space module (LacaVSS) and a Lightweight Dual Domain Dynamic Collaborative Fusion module (LD3CF). LacaVSS adaptively models crack cues through the proposed mask-guided Efficient Dynamic Guided Scanning Strategy (EDG-SS), while LD3CF leverages an Adaptive Frequency Domain Perceptron (AFDP) and a dual-pooling fusion strategy to effectively capture spatial and frequency-domain cues across modalities. Moreover, we design a Lightweight Dynamically Modulated Multi-Kernel convolution (LDMK) to perceive complex morphological structures with minimal computational overhead, replacing most convolutional operations in LIDAR. Experiments on three datasets demonstrate that our method outperforms other state-of-the-art (SOTA) methods. On the light-field depth dataset, our method achieves 0.8204 in F1 and 0.8465 in mIoU with only 5.35M parameters. Code and datasets are available at https://github.com/Karl1109/LIDAR-Mamba.
>
---
#### [replaced 085] Mitigating Hallucination of Large Vision-Language Models via Dynamic Logits Calibration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21509v2](http://arxiv.org/pdf/2506.21509v2)**

> **作者:** Jiahe Chen; Jiaying He; Qian Shao; Qiyuan Chen; Jiahe Ying; Hongxia Xu; Jintai Chen; Jianwei Zheng; Jian Wu
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated significant advancements in multimodal understanding, yet they are frequently hampered by hallucination-the generation of text that contradicts visual input. Existing training-free decoding strategies exhibit critical limitations, including the use of static constraints that do not adapt to semantic drift during generation, inefficiency stemming from the need for multiple forward passes, and degradation of detail due to overly rigid intervention rules. To overcome these challenges, this paper introduces Dynamic Logits Calibration (DLC), a novel training-free decoding framework designed to dynamically align text generation with visual evidence at inference time. At the decoding phase, DLC step-wise employs CLIP to assess the semantic alignment between the input image and the generated text sequence. Then, the Relative Visual Advantage (RVA) of candidate tokens is evaluated against a dynamically updated contextual baseline, adaptively adjusting output logits to favor tokens that are visually grounded. Furthermore, an adaptive weighting mechanism, informed by a real-time context alignment score, carefully balances the visual guidance while ensuring the overall quality of the textual output. Extensive experiments conducted across diverse benchmarks and various LVLM architectures (such as LLaVA, InstructBLIP, and MiniGPT-4) demonstrate that DLC significantly reduces hallucinations, outperforming current methods while maintaining high inference efficiency by avoiding multiple forward passes. Overall, we present an effective and efficient decoding-time solution to mitigate hallucinations, thereby enhancing the reliability of LVLMs for more practices. Code will be released on Github.
>
---
#### [replaced 086] YOLO-FireAD: Efficient Fire Detection via Attention-Guided Inverted Residual Learning and Dual-Pooling Feature Preservation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20884v2](http://arxiv.org/pdf/2505.20884v2)**

> **作者:** Weichao Pan; Bohan Xu; Xu Wang; Chengze Lv; Shuoyang Wang; Zhenke Duan
>
> **备注:** 2025 International Conference on Intelligent Computing (ICIC 2025)
>
> **摘要:** Fire detection in dynamic environments faces continuous challenges, including the interference of illumination changes, many false detections or missed detections, and it is difficult to achieve both efficiency and accuracy. To address the problem of feature extraction limitation and information loss in the existing YOLO-based models, this study propose You Only Look Once for Fire Detection with Attention-guided Inverted Residual and Dual-pooling Downscale Fusion (YOLO-FireAD) with two core innovations: (1) Attention-guided Inverted Residual Block (AIR) integrates hybrid channel-spatial attention with inverted residuals to adaptively enhance fire features and suppress environmental noise; (2) Dual Pool Downscale Fusion Block (DPDF) preserves multi-scale fire patterns through learnable fusion of max-average pooling outputs, mitigating small-fire detection failures. Extensive evaluation on two public datasets shows the efficient performance of our model. Our proposed model keeps the sum amount of parameters (1.45M, 51.8% lower than YOLOv8n) (4.6G, 43.2% lower than YOLOv8n), and mAP75 is higher than the mainstream real-time object detection models YOLOv8n, YOL-Ov9t, YOLOv10n, YOLO11n, YOLOv12n and other YOLOv8 variants 1.3-5.5%. For more details, please visit our repository: https://github.com/JEFfersusu/YOLO-FireAD
>
---
