# 计算机视觉 cs.CV

- **最新发布 74 篇**

- **更新 51 篇**

## 最新发布

#### [new 001] Data-Agnostic Augmentations for Unknown Variations: Out-of-Distribution Generalisation in MRI Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学影像分割任务，旨在解决模型因训练与测试数据分布差异导致的性能下降问题。针对传统数据增强方法在真实场景中的不足，提出了无需特定分布偏移假设的MixUp和傅里叶增强策略，通过提升特征可分性与紧凑性增强MRI分割模型的分布外泛化能力，并验证了其在心脏/前列腺MRI分割中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.10223v1](http://arxiv.org/pdf/2505.10223v1)**

> **作者:** Puru Vaish; Felix Meister; Tobias Heimann; Christoph Brune; Jelmer M. Wolterink
>
> **备注:** Accepted at MIDL 2025
>
> **摘要:** Medical image segmentation models are often trained on curated datasets, leading to performance degradation when deployed in real-world clinical settings due to mismatches between training and test distributions. While data augmentation techniques are widely used to address these challenges, traditional visually consistent augmentation strategies lack the robustness needed for diverse real-world scenarios. In this work, we systematically evaluate alternative augmentation strategies, focusing on MixUp and Auxiliary Fourier Augmentation. These methods mitigate the effects of multiple variations without explicitly targeting specific sources of distribution shifts. We demonstrate how these techniques significantly improve out-of-distribution generalization and robustness to imaging variations across a wide range of transformations in cardiac cine MRI and prostate MRI segmentation. We quantitatively find that these augmentation methods enhance learned feature representations by promoting separability and compactness. Additionally, we highlight how their integration into nnU-Net training pipelines provides an easy-to-implement, effective solution for enhancing the reliability of medical segmentation models in real-world applications.
>
---
#### [new 002] Exploring Implicit Visual Misunderstandings in Multimodal Large Language Models through Attention Analysis
- **分类: cs.CV**

- **简介: 该论文属于多模态模型评估任务，旨在解决模型正确回答却未真正理解视觉输入的隐式视觉误解（IVM）问题。通过解耦注意力模块分析视觉依赖偏差，提出"注意力准确度"指标及量化基准，直接评估视觉理解能力，并验证其在单模态场景的泛化性。**

- **链接: [http://arxiv.org/pdf/2505.10541v1](http://arxiv.org/pdf/2505.10541v1)**

> **作者:** Pengfei Wang; Guohai Xu; Weinong Wang; Junjie Yang; Jie Lou; Yunhua Xue
>
> **摘要:** Recent advancements have enhanced the capability of Multimodal Large Language Models (MLLMs) to comprehend multi-image information. However, existing benchmarks primarily evaluate answer correctness, overlooking whether models genuinely comprehend the visual input. To address this, we define implicit visual misunderstanding (IVM), where MLLMs provide correct answers without fully comprehending the visual input. Through our analysis, we decouple the visual and textual modalities within the causal attention module, revealing that attention distribution increasingly converges on the image associated with the correct answer as the network layers deepen. This insight leads to the introduction of a scale-agnostic metric, \textit{attention accuracy}, and a novel benchmark for quantifying IVMs. Attention accuracy directly evaluates the model's visual understanding via internal mechanisms, remaining robust to positional biases for more reliable assessments. Furthermore, we extend our approach to finer granularities and demonstrate its effectiveness in unimodal scenarios, underscoring its versatility and generalizability.
>
---
#### [new 003] Multi-Source Collaborative Style Augmentation and Domain-Invariant Learning for Federated Domain Generalization
- **分类: cs.CV**

- **简介: 该论文针对联邦领域泛化任务，解决分散数据源中风格增强空间受限问题。提出MCSAD方法，通过多源协作式风格增强扩展数据风格空间，并采用跨域特征对齐和类关系集成蒸馏实现领域不变学习，提升模型在未知目标域的泛化性能。**

- **链接: [http://arxiv.org/pdf/2505.10152v1](http://arxiv.org/pdf/2505.10152v1)**

> **作者:** Yikang Wei
>
> **备注:** IJCAI 2025
>
> **摘要:** Federated domain generalization aims to learn a generalizable model from multiple decentralized source domains for deploying on the unseen target domain. The style augmentation methods have achieved great progress on domain generalization. However, the existing style augmentation methods either explore the data styles within isolated source domain or interpolate the style information across existing source domains under the data decentralization scenario, which leads to limited style space. To address this issue, we propose a Multi-source Collaborative Style Augmentation and Domain-invariant learning method (MCSAD) for federated domain generalization. Specifically, we propose a multi-source collaborative style augmentation module to generate data in the broader style space. Furthermore, we conduct domain-invariant learning between the original data and augmented data by cross-domain feature alignment within the same class and classes relation ensemble distillation between different classes to learn a domain-invariant model. By alternatively conducting collaborative style augmentation and domain-invariant learning, the model can generalize well on unseen target domain. Extensive experiments on multiple domain generalization datasets indicate that our method significantly outperforms the state-of-the-art federated domain generalization methods.
>
---
#### [new 004] MambaControl: Anatomy Graph-Enhanced Mamba ControlNet with Fourier Refinement for Diffusion-Based Disease Trajectory Prediction
- **分类: cs.CV**

- **简介: 该论文属于医学图像疾病轨迹预测任务，旨在解决现有方法在长期依赖建模和结构一致性上的不足。提出MambaControl框架，结合Mamba状态空间模型与图结构控制网络，利用傅里叶增强的谱图表征捕捉时空动态，提升阿尔茨海默病预测的解剖保真度和预测精度。**

- **链接: [http://arxiv.org/pdf/2505.09965v1](http://arxiv.org/pdf/2505.09965v1)**

> **作者:** Hao Yang; Tao Tan; Shuai Tan; Weiqin Yang; Kunyan Cai; Calvin Chen; Yue Sun
>
> **摘要:** Modelling disease progression in precision medicine requires capturing complex spatio-temporal dynamics while preserving anatomical integrity. Existing methods often struggle with longitudinal dependencies and structural consistency in progressive disorders. To address these limitations, we introduce MambaControl, a novel framework that integrates selective state-space modelling with diffusion processes for high-fidelity prediction of medical image trajectories. To better capture subtle structural changes over time while maintaining anatomical consistency, MambaControl combines Mamba-based long-range modelling with graph-guided anatomical control to more effectively represent anatomical correlations. Furthermore, we introduce Fourier-enhanced spectral graph representations to capture spatial coherence and multiscale detail, enabling MambaControl to achieve state-of-the-art performance in Alzheimer's disease prediction. Quantitative and regional evaluations demonstrate improved progression prediction quality and anatomical fidelity, highlighting its potential for personalised prognosis and clinical decision support.
>
---
#### [new 005] APCoTTA: Continual Test-Time Adaptation for Semantic Segmentation of Airborne LiDAR Point Clouds
- **分类: cs.CV**

- **简介: 该论文研究机载激光点云语义分割的持续测试时自适应（CTTA）任务，解决域偏移导致的模型性能下降问题。提出了APCoTTA方法：动态选择低置信度层训练缓解灾难性遗忘，熵一致性损失减少误差累积，参数插值平衡新旧知识。建立了两个CTTA基准数据集，实验显示mIoU提升9%-14%。**

- **链接: [http://arxiv.org/pdf/2505.09971v1](http://arxiv.org/pdf/2505.09971v1)**

> **作者:** Yuan Gao; Shaobo Xia; Sheng Nie; Cheng Wang; Xiaohuan Xi; Bisheng Yang
>
> **备注:** 18 pages,12 figures
>
> **摘要:** Airborne laser scanning (ALS) point cloud segmentation is a fundamental task for large-scale 3D scene understanding. In real-world applications, models are typically fixed after training. However, domain shifts caused by changes in the environment, sensor types, or sensor degradation often lead to a decline in model performance. Continuous Test-Time Adaptation (CTTA) offers a solution by adapting a source-pretrained model to evolving, unlabeled target domains. Despite its potential, research on ALS point clouds remains limited, facing challenges such as the absence of standardized datasets and the risk of catastrophic forgetting and error accumulation during prolonged adaptation. To tackle these challenges, we propose APCoTTA, the first CTTA method tailored for ALS point cloud semantic segmentation. We propose a dynamic trainable layer selection module. This module utilizes gradient information to select low-confidence layers for training, and the remaining layers are kept frozen, mitigating catastrophic forgetting. To further reduce error accumulation, we propose an entropy-based consistency loss. By losing such samples based on entropy, we apply consistency loss only to the reliable samples, enhancing model stability. In addition, we propose a random parameter interpolation mechanism, which randomly blends parameters from the selected trainable layers with those of the source model. This approach helps balance target adaptation and source knowledge retention, further alleviating forgetting. Finally, we construct two benchmarks, ISPRSC and H3DC, to address the lack of CTTA benchmarks for ALS point cloud segmentation. Experimental results demonstrate that APCoTTA achieves the best performance on two benchmarks, with mIoU improvements of approximately 9% and 14% over direct inference. The new benchmarks and code are available at https://github.com/Gaoyuan2/APCoTTA.
>
---
#### [new 006] From Air to Wear: Personalized 3D Digital Fashion with AR/VR Immersive 3D Sketching
- **分类: cs.CV**

- **简介: 该论文属于3D服装生成任务，旨在降低AR/VR环境下数字时装设计的技术门槛。针对普通用户难以使用专业工具和数据匮乏的问题，提出基于条件扩散模型和共享潜在空间编码的框架，通过3D草图生成个性化服装，并构建KO3DClothes数据集。系统采用自适应课程学习策略处理自由手绘输入，实验验证其生成质量和易用性优势。**

- **链接: [http://arxiv.org/pdf/2505.09998v1](http://arxiv.org/pdf/2505.09998v1)**

> **作者:** Ying Zang; Yuanqi Hu; Xinyu Chen; Yuxia Xu; Suhui Wang; Chunan Yu; Lanyun Zhu; Deyi Ji; Xin Xu; Tianrun Chen
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** In the era of immersive consumer electronics, such as AR/VR headsets and smart devices, people increasingly seek ways to express their identity through virtual fashion. However, existing 3D garment design tools remain inaccessible to everyday users due to steep technical barriers and limited data. In this work, we introduce a 3D sketch-driven 3D garment generation framework that empowers ordinary users - even those without design experience - to create high-quality digital clothing through simple 3D sketches in AR/VR environments. By combining a conditional diffusion model, a sketch encoder trained in a shared latent space, and an adaptive curriculum learning strategy, our system interprets imprecise, free-hand input and produces realistic, personalized garments. To address the scarcity of training data, we also introduce KO3DClothes, a new dataset of paired 3D garments and user-created sketches. Extensive experiments and user studies confirm that our method significantly outperforms existing baselines in both fidelity and usability, demonstrating its promise for democratized fashion design on next-generation consumer platforms.
>
---
#### [new 007] Does Feasibility Matter? Understanding the Impact of Feasibility on Synthetic Training Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究合成训练数据可行性（图像属性是否现实）对CLIP分类器的影响，属于数据增强优化任务。通过VariReal生成可控可行/不可行图像，实验发现可行性对微调模型性能影响微弱（<0.3%准确率差异），且混合数据无明显副作用，挑战了传统需过滤不可行数据的假设。**

- **链接: [http://arxiv.org/pdf/2505.10551v1](http://arxiv.org/pdf/2505.10551v1)**

> **作者:** Yiwen Liu; Jessica Bader; Jae Myung Kim
>
> **备注:** CVPRW 2025
>
> **摘要:** With the development of photorealistic diffusion models, models trained in part or fully on synthetic data achieve progressively better results. However, diffusion models still routinely generate images that would not exist in reality, such as a dog floating above the ground or with unrealistic texture artifacts. We define the concept of feasibility as whether attributes in a synthetic image could realistically exist in the real-world domain; synthetic images containing attributes that violate this criterion are considered infeasible. Intuitively, infeasible images are typically considered out-of-distribution; thus, training on such images is expected to hinder a model's ability to generalize to real-world data, and they should therefore be excluded from the training set whenever possible. However, does feasibility really matter? In this paper, we investigate whether enforcing feasibility is necessary when generating synthetic training data for CLIP-based classifiers, focusing on three target attributes: background, color, and texture. We introduce VariReal, a pipeline that minimally edits a given source image to include feasible or infeasible attributes given by the textual prompt generated by a large language model. Our experiments show that feasibility minimally affects LoRA-fine-tuned CLIP performance, with mostly less than 0.3% difference in top-1 accuracy across three fine-grained datasets. Also, the attribute matters on whether the feasible/infeasible images adversarially influence the classification performance. Finally, mixing feasible and infeasible images in training datasets does not significantly impact performance compared to using purely feasible or infeasible datasets.
>
---
#### [new 008] ToonifyGB: StyleGAN-based Gaussian Blendshapes for 3D Stylized Head Avatars
- **分类: cs.CV**

- **简介: 该论文提出ToonifyGB框架，用于生成可动画的3D风格化头部虚拟形象。针对现有方法固定分辨率预处理导致的视频不稳定问题，采用两阶段方案：改进StyleGAN生成稳定风格化视频，再通过高斯混合形状建模中性头部与表情基，实现高保真实时渲染。在Arcane和Pixar风格数据集验证了有效性。**

- **链接: [http://arxiv.org/pdf/2505.10072v1](http://arxiv.org/pdf/2505.10072v1)**

> **作者:** Rui-Yang Ju; Sheng-Yen Huang; Yi-Ping Hung
>
> **摘要:** The introduction of 3D Gaussian blendshapes has enabled the real-time reconstruction of animatable head avatars from monocular video. Toonify, a StyleGAN-based framework, has become widely used for facial image stylization. To extend Toonify for synthesizing diverse stylized 3D head avatars using Gaussian blendshapes, we propose an efficient two-stage framework, ToonifyGB. In Stage 1 (stylized video generation), we employ an improved StyleGAN to generate the stylized video from the input video frames, which addresses the limitation of cropping aligned faces at a fixed resolution as preprocessing for normal StyleGAN. This process provides a more stable video, which enables Gaussian blendshapes to better capture the high-frequency details of the video frames, and efficiently generate high-quality animation in the next stage. In Stage 2 (Gaussian blendshapes synthesis), we learn a stylized neutral head model and a set of expression blendshapes from the generated video. By combining the neutral head model with expression blendshapes, ToonifyGB can efficiently render stylized avatars with arbitrary expressions. We validate the effectiveness of ToonifyGB on the benchmark dataset using two styles: Arcane and Pixar.
>
---
#### [new 009] High Quality Underwater Image Compression with Adaptive Correction and Codebook-based Augmentation
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究水下图像高效压缩任务，解决现有算法未充分利用水下图像特性（如光照、色调差异）导致性能不佳的问题。提出HQUIC方法：通过ALTC模块自适应预测衰减系数与全局光照以校正差异，利用码本提取共性物体增强主模型，并动态加权多尺度频率保留关键信息，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09986v1](http://arxiv.org/pdf/2505.09986v1)**

> **作者:** Yimin Zhou; Yichong Xia; Sicheng Pan; Bin Chen; Baoyi An; Haoqian Wang; Zhi Wang; Yaowei Wang; Zikun Zhou
>
> **摘要:** With the increasing exploration and exploitation of the underwater world, underwater images have become a critical medium for human interaction with marine environments, driving extensive research into their efficient transmission and storage. However, contemporary underwater image compression algorithms fail to fully leverage the unique characteristics distinguishing underwater scenes from terrestrial images, resulting in suboptimal performance. To address this limitation, we introduce HQUIC, designed to exploit underwater-image-specific features for enhanced compression efficiency. HQUIC employs an ALTC module to adaptively predict the attenuation coefficients and global light information of the images, which effectively mitigates the issues caused by the differences in lighting and tone existing in underwater images. Subsequently, HQUIC employs a codebook as an auxiliary branch to extract the common objects within underwater images and enhances the performance of the main branch. Furthermore, HQUIC dynamically weights multi-scale frequency components, prioritizing information critical for distortion quality while discarding redundant details. Extensive evaluations on diverse underwater datasets demonstrate that HQUIC outperforms state-of-the-art compression methods.
>
---
#### [new 010] MTVCrafter: 4D Motion Tokenization for Open-World Human Image Animation
- **分类: cs.CV**

- **简介: 该论文研究人体图像动画生成任务，解决现有方法依赖2D姿势导致泛化差、丢失3D信息的问题。提出MTVCrafter框架：通过4DMoT将3D运动序列量化为4D令牌，结合MV-DiT模型利用运动注意力和位置编码生成动画，实现开放世界中多样化角色的灵活控制，实验指标(FID-VID 6.98)优于次优方法65%。**

- **链接: [http://arxiv.org/pdf/2505.10238v1](http://arxiv.org/pdf/2505.10238v1)**

> **作者:** Yanbo Ding
>
> **摘要:** Human image animation has gained increasing attention and developed rapidly due to its broad applications in digital humans. However, existing methods rely largely on 2D-rendered pose images for motion guidance, which limits generalization and discards essential 3D information for open-world animation. To tackle this problem, we propose MTVCrafter (Motion Tokenization Video Crafter), the first framework that directly models raw 3D motion sequences (i.e., 4D motion) for human image animation. Specifically, we introduce 4DMoT (4D motion tokenizer) to quantize 3D motion sequences into 4D motion tokens. Compared to 2D-rendered pose images, 4D motion tokens offer more robust spatio-temporal cues and avoid strict pixel-level alignment between pose image and character, enabling more flexible and disentangled control. Then, we introduce MV-DiT (Motion-aware Video DiT). By designing unique motion attention with 4D positional encodings, MV-DiT can effectively leverage motion tokens as 4D compact yet expressive context for human image animation in the complex 3D world. Hence, it marks a significant step forward in this field and opens a new direction for pose-guided human video generation. Experiments show that our MTVCrafter achieves state-of-the-art results with an FID-VID of 6.98, surpassing the second-best by 65%. Powered by robust motion tokens, MTVCrafter also generalizes well to diverse open-world characters (single/multiple, full/half-body) across various styles and scenarios. Our video demos and code are provided in the supplementary material and at this anonymous GitHub link: https://anonymous.4open.science/r/MTVCrafter-1B13.
>
---
#### [new 011] Depth Anything with Any Prior
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在融合不完整度量深度信息与相对预测结构，生成高精度稠密深度图。通过两阶段框架：先对齐多种度量先验缩小域差距，再构建条件化模型隐式融合互补数据并优化噪声，在零样本场景下实现深度补全、超分辨率等任务，性能超越专用方法且支持效率-精度灵活调整。**

- **链接: [http://arxiv.org/pdf/2505.10565v1](http://arxiv.org/pdf/2505.10565v1)**

> **作者:** Zehan Wang; Siyu Chen; Lihe Yang; Jialei Wang; Ziang Zhang; Hengshuang Zhao; Zhou Zhao
>
> **备注:** Home page: https://prior-depth-anything.github.io/
>
> **摘要:** This work presents Prior Depth Anything, a framework that combines incomplete but precise metric information in depth measurement with relative but complete geometric structures in depth prediction, generating accurate, dense, and detailed metric depth maps for any scene. To this end, we design a coarse-to-fine pipeline to progressively integrate the two complementary depth sources. First, we introduce pixel-level metric alignment and distance-aware weighting to pre-fill diverse metric priors by explicitly using depth prediction. It effectively narrows the domain gap between prior patterns, enhancing generalization across varying scenarios. Second, we develop a conditioned monocular depth estimation (MDE) model to refine the inherent noise of depth priors. By conditioning on the normalized pre-filled prior and prediction, the model further implicitly merges the two complementary depth sources. Our model showcases impressive zero-shot generalization across depth completion, super-resolution, and inpainting over 7 real-world datasets, matching or even surpassing previous task-specific methods. More importantly, it performs well on challenging, unseen mixed priors and enables test-time improvements by switching prediction models, providing a flexible accuracy-efficiency trade-off while evolving with advancements in MDE models.
>
---
#### [new 012] End-to-End Vision Tokenizer Tuning
- **分类: cs.CV**

- **简介: 该论文提出端到端视觉分词器调优方法ETT，解决现有视觉分词器与下游任务（如图像生成、问答）优化脱节导致的表征瓶颈问题。通过联合优化分词器与自回归任务，结合重构和描述目标，提升多模态任务性能2-6%，同时保持原有重建能力，无需修改基础模型结构。**

- **链接: [http://arxiv.org/pdf/2505.10562v1](http://arxiv.org/pdf/2505.10562v1)**

> **作者:** Wenxuan Wang; Fan Zhang; Yufeng Cui; Haiwen Diao; Zhuoyan Luo; Huchuan Lu; Jing Liu; Xinlong Wang
>
> **摘要:** Existing vision tokenization isolates the optimization of vision tokenizers from downstream training, implicitly assuming the visual tokens can generalize well across various tasks, e.g., image generation and visual question answering. The vision tokenizer optimized for low-level reconstruction is agnostic to downstream tasks requiring varied representations and semantics. This decoupled paradigm introduces a critical misalignment: The loss of the vision tokenization can be the representation bottleneck for target tasks. For example, errors in tokenizing text in a given image lead to poor results when recognizing or generating them. To address this, we propose ETT, an end-to-end vision tokenizer tuning approach that enables joint optimization between vision tokenization and target autoregressive tasks. Unlike prior autoregressive models that use only discrete indices from a frozen vision tokenizer, ETT leverages the visual embeddings of the tokenizer codebook, and optimizes the vision tokenizers end-to-end with both reconstruction and caption objectives. ETT can be seamlessly integrated into existing training pipelines with minimal architecture modifications. Our ETT is simple to implement and integrate, without the need to adjust the original codebooks or architectures of the employed large language models. Extensive experiments demonstrate that our proposed end-to-end vision tokenizer tuning unlocks significant performance gains, i.e., 2-6% for multimodal understanding and visual generation tasks compared to frozen tokenizer baselines, while preserving the original reconstruction capability. We hope this very simple and strong method can empower multimodal foundation models besides image generation and understanding.
>
---
#### [new 013] Learned Lightweight Smartphone ISP with Unpaired Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究智能手机图像信号处理（ISP），提出一种无需配对数据的轻量级深度学习模型。传统方法依赖对齐的RAW-RGB数据训练，成本高昂。作者通过对抗训练结合多判别器结构，利用预训练网络特征保持图像结构，从非配对数据中学习色彩与纹理特征，在移动端实现高质量RAW转RGB，性能接近配对训练方法。**

- **链接: [http://arxiv.org/pdf/2505.10420v1](http://arxiv.org/pdf/2505.10420v1)**

> **作者:** Andrei Arhire; Radu Timofte
>
> **备注:** Accepted at CVPRW 2025
>
> **摘要:** The Image Signal Processor (ISP) is a fundamental component in modern smartphone cameras responsible for conversion of RAW sensor image data to RGB images with a strong focus on perceptual quality. Recent work highlights the potential of deep learning approaches and their ability to capture details with a quality increasingly close to that of professional cameras. A difficult and costly step when developing a learned ISP is the acquisition of pixel-wise aligned paired data that maps the raw captured by a smartphone camera sensor to high-quality reference images. In this work, we address this challenge by proposing a novel training method for a learnable ISP that eliminates the need for direct correspondences between raw images and ground-truth data with matching content. Our unpaired approach employs a multi-term loss function guided by adversarial training with multiple discriminators processing feature maps from pre-trained networks to maintain content structure while learning color and texture characteristics from the target RGB dataset. Using lightweight neural network architectures suitable for mobile devices as backbones, we evaluated our method on the Zurich RAW to RGB and Fujifilm UltraISP datasets. Compared to paired training methods, our unpaired learning strategy shows strong potential and achieves high fidelity across multiple evaluation metrics. The code and pre-trained models are available at https://github.com/AndreiiArhire/Learned-Lightweight-Smartphone-ISP-with-Unpaired-Data .
>
---
#### [new 014] Why 1 + 1 < 1 in Visual Token Pruning: Beyond Naive Integration via Multi-Objective Balanced Covering
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉令牌剪枝任务，解决现有方法因静态策略忽视多目标动态权衡导致的性能不稳定问题。通过理论分析提出误差界与最优目标平衡条件，设计多目标平衡覆盖方法（MoB），将剪枝转化为预算分配问题，实现高效自适应剪枝。实验验证其在多模态大模型中显著降低计算量且保持性能。**

- **链接: [http://arxiv.org/pdf/2505.10118v1](http://arxiv.org/pdf/2505.10118v1)**

> **作者:** Yangfu Li; Hongjian Zhan; Tianyi Chen; Qi Liu; Yue Lu
>
> **备注:** 31 pages,9 figures,conference
>
> **摘要:** Existing visual token pruning methods target prompt alignment and visual preservation with static strategies, overlooking the varying relative importance of these objectives across tasks, which leads to inconsistent performance. To address this, we derive the first closed-form error bound for visual token pruning based on the Hausdorff distance, uniformly characterizing the contributions of both objectives. Moreover, leveraging $\epsilon$-covering theory, we reveal an intrinsic trade-off between these objectives and quantify their optimal attainment levels under a fixed budget. To practically handle this trade-off, we propose Multi-Objective Balanced Covering (MoB), which reformulates visual token pruning as a bi-objective covering problem. In this framework, the attainment trade-off reduces to budget allocation via greedy radius trading. MoB offers a provable performance bound and linear scalability with respect to the number of input visual tokens, enabling adaptation to challenging pruning scenarios. Extensive experiments show that MoB preserves 96.4% of performance for LLaVA-1.5-7B using only 11.1% of the original visual tokens and accelerates LLaVA-Next-7B by 1.3-1.5$\times$ with negligible performance loss. Additionally, evaluations on Qwen2-VL and Video-LLaVA confirm that MoB integrates seamlessly into advanced MLLMs and diverse vision-language tasks.
>
---
#### [new 015] AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AdaptCLIP方法，用于通用视觉异常检测任务，解决跨域检测时依赖提示模板、复杂交互或微调的问题。通过交替学习视觉与文本表征，结合上下文和对齐残差特征的对比学习，仅在CLIP模型输入/输出端添加三个轻量适配器，实现无需目标域训练即可完成零/少样本跨域异常检测，在12个工业与医学基准数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2505.09926v1](http://arxiv.org/pdf/2505.09926v1)**

> **作者:** Bin-Bin Gao; Yue Zhu; Jiangtao Yan; Yuezhi Cai; Weixi Zhang; Meng Wang; Jun Liu; Yong Liu; Lei Wang; Chengjie Wang
>
> **备注:** 27 pages, 15 figures, 22 tables
>
> **摘要:** Universal visual anomaly detection aims to identify anomalies from novel or unseen vision domains without additional fine-tuning, which is critical in open scenarios. Recent studies have demonstrated that pre-trained vision-language models like CLIP exhibit strong generalization with just zero or a few normal images. However, existing methods struggle with designing prompt templates, complex token interactions, or requiring additional fine-tuning, resulting in limited flexibility. In this work, we present a simple yet effective method called AdaptCLIP based on two key insights. First, adaptive visual and textual representations should be learned alternately rather than jointly. Second, comparative learning between query and normal image prompt should incorporate both contextual and aligned residual features, rather than relying solely on residual features. AdaptCLIP treats CLIP models as a foundational service, adding only three simple adapters, visual adapter, textual adapter, and prompt-query adapter, at its input or output ends. AdaptCLIP supports zero-/few-shot generalization across domains and possesses a training-free manner on target domains once trained on a base dataset. AdaptCLIP achieves state-of-the-art performance on 12 anomaly detection benchmarks from industrial and medical domains, significantly outperforming existing competitive methods. We will make the code and model of AdaptCLIP available at https://github.com/gaobb/AdaptCLIP.
>
---
#### [new 016] MMRL++: Parameter-Efficient and Interaction-Aware Representation Learning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于多模态表示学习任务，旨在解决视觉-语言模型在少样本场景下过拟合和泛化能力差的问题。提出MMRL++方法，通过共享模态无关表示空间，在高层编码器插入可学习的表示令牌以增强跨模态交互，保留底层通用知识，并设计正则化与解耦推理策略，减少参数的同时提升任务适应与泛化平衡，实验证明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.10088v1](http://arxiv.org/pdf/2505.10088v1)**

> **作者:** Yuncheng Guo; Xiaodong Gu
>
> **备注:** Due to the limitation "The abstract field cannot be longer than 1,920 characters", the abstract appearing here is slightly shorter than that in the PDF file
>
> **摘要:** Large-scale pre-trained Vision-Language Models (VLMs) have significantly advanced transfer learning across diverse tasks. However, adapting these models with limited few-shot data often leads to overfitting, undermining their ability to generalize to new tasks. To address this, we propose Multi-Modal Representation Learning (MMRL), which introduces a shared, learnable, modality-agnostic representation space. MMRL generates space tokens projected into both text and image encoders as representation tokens, enabling more effective cross-modal interactions. Unlike prior methods that mainly optimize class token features, MMRL inserts representation tokens into higher encoder layers--where task-specific features are more prominent--while preserving general knowledge in the lower layers. During training, both class and representation features are jointly optimized: a trainable projection layer is applied to representation tokens for task adaptation, while the projection layer for class token remains frozen to retain pre-trained knowledge. To further promote generalization, we introduce a regularization term aligning class and text features with the frozen VLM's zero-shot features. At inference, a decoupling strategy uses both class and representation features for base tasks, but only class features for novel tasks due to their stronger generalization. Building upon this, we propose MMRL++, a parameter-efficient and interaction-aware extension that significantly reduces trainable parameters and enhances intra-modal interactions--particularly across the layers of representation tokens--allowing gradient sharing and instance-specific information to propagate more effectively through the network. Extensive experiments on 15 datasets demonstrate that MMRL and MMRL++ consistently outperform state-of-the-art methods, achieving a strong balance between task-specific adaptation and generalization.
>
---
#### [new 017] UniEval: Unified Holistic Evaluation for Unified Multimodal Understanding and Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出UniEval框架，用于统一评估多模态理解和生成模型。针对现有评估方法分散、依赖额外数据/模型、指标单一等问题，设计了包含多样化任务标签的UniBench基准及UniScore评测指标，无需外部资源即可实现高效评估。实验表明其挑战性强且贴近人工评分，揭示了现有模型的不足。**

- **链接: [http://arxiv.org/pdf/2505.10483v1](http://arxiv.org/pdf/2505.10483v1)**

> **作者:** Yi Li; Haonan Wang; Qixiang Zhang; Boyu Xiao; Chenchang Hu; Hualiang Wang; Xiaomeng Li
>
> **备注:** UniEval is the first evaluation framework designed for unified multimodal models, including a holistic benchmark UniBench and the UniScore metric
>
> **摘要:** The emergence of unified multimodal understanding and generation models is rapidly attracting attention because of their ability to enhance instruction-following capabilities while minimizing model redundancy. However, there is a lack of a unified evaluation framework for these models, which would enable an elegant, simplified, and overall evaluation. Current models conduct evaluations on multiple task-specific benchmarks, but there are significant limitations, such as the lack of overall results, errors from extra evaluation models, reliance on extensive labeled images, benchmarks that lack diversity, and metrics with limited capacity for instruction-following evaluation. To tackle these challenges, we introduce UniEval, the first evaluation framework designed for unified multimodal models without extra models, images, or annotations. This facilitates a simplified and unified evaluation process. The UniEval framework contains a holistic benchmark, UniBench (supports both unified and visual generation models), along with the corresponding UniScore metric. UniBench includes 81 fine-grained tags contributing to high diversity. Experimental results indicate that UniBench is more challenging than existing benchmarks, and UniScore aligns closely with human evaluations, surpassing current metrics. Moreover, we extensively evaluated SoTA unified and visual generation models, uncovering new insights into Univeral's unique values.
>
---
#### [new 018] CSPENet: Contour-Aware and Saliency Priors Embedding Network for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决现有方法在弱光目标定位及密集杂波下轮廓感知不足的问题。提出CSPENet模型，通过轮廓收敛先验提取、双分支特征融合及注意力增强模块，提升目标定位精度和轮廓细节表征能力，实验验证其在多个数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09943v1](http://arxiv.org/pdf/2505.09943v1)**

> **作者:** Jiakun Deng; Kexuan Li; Xingye Cui; Jiaxuan Li; Chang Long; Tian Pu; Zhenming Peng
>
> **摘要:** Infrared small target detection (ISTD) plays a critical role in a wide range of civilian and military applications. Existing methods suffer from deficiencies in the localization of dim targets and the perception of contour information under dense clutter environments, severely limiting their detection performance. To tackle these issues, we propose a contour-aware and saliency priors embedding network (CSPENet) for ISTD. We first design a surround-convergent prior extraction module (SCPEM) that effectively captures the intrinsic characteristic of target contour pixel gradients converging toward their center. This module concurrently extracts two collaborative priors: a boosted saliency prior for accurate target localization and multi-scale structural priors for comprehensively enriching contour detail representation. Building upon this, we propose a dual-branch priors embedding architecture (DBPEA) that establishes differentiated feature fusion pathways, embedding these two priors at optimal network positions to achieve performance enhancement. Finally, we develop an attention-guided feature enhancement module (AGFEM) to refine feature representations and improve saliency estimation accuracy. Experimental results on public datasets NUDT-SIRST, IRSTD-1k, and NUAA-SIRST demonstrate that our CSPENet outperforms other state-of-the-art methods in detection performance. The code is available at https://github.com/IDIP2025/CSPENet.
>
---
#### [new 019] Exploring the Deep Fusion of Large Language Models and Diffusion Transformers for Text-to-Image Synthesis
- **分类: cs.CV**

- **简介: 该论文属于多模态文本到图像合成任务，旨在解决LLM与扩散Transformer融合方法中设计细节不透明、潜力评估不足的问题。通过系统性的对照实验分析设计选择，公开可复现的大规模训练方案，填补现有研究在方法论比较与优化策略上的空白，为多模态生成提供实践指导。**

- **链接: [http://arxiv.org/pdf/2505.10046v1](http://arxiv.org/pdf/2505.10046v1)**

> **作者:** Bingda Tang; Boyang Zheng; Xichen Pan; Sayak Paul; Saining Xie
>
> **摘要:** This paper does not describe a new method; instead, it provides a thorough exploration of an important yet understudied design space related to recent advances in text-to-image synthesis -- specifically, the deep fusion of large language models (LLMs) and diffusion transformers (DiTs) for multi-modal generation. Previous studies mainly focused on overall system performance rather than detailed comparisons with alternative methods, and key design details and training recipes were often left undisclosed. These gaps create uncertainty about the real potential of this approach. To fill these gaps, we conduct an empirical study on text-to-image generation, performing controlled comparisons with established baselines, analyzing important design choices, and providing a clear, reproducible recipe for training at scale. We hope this work offers meaningful data points and practical guidelines for future research in multi-modal generation.
>
---
#### [new 020] Descriptive Image-Text Matching with Graded Contextual Similarity
- **分类: cs.CV**

- **简介: 该论文研究图像文本匹配任务，旨在解决现有方法因二元监督忽视多对多对应及描述层次性的问题。提出DITM方法，利用句子描述性评分（基于TF-IDF）学习分级上下文相似性，动态调整正负样本关联并按通用到具体顺序对齐文本，提升匹配精度及潜在正样本挖掘能力，实验验证其在多数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2505.09997v1](http://arxiv.org/pdf/2505.09997v1)**

> **作者:** Jinhyun Jang; Jiyeong Lee; Kwanghoon Sohn
>
> **摘要:** Image-text matching aims to build correspondences between visual and textual data by learning their pairwise similarities. Most existing approaches have adopted sparse binary supervision, indicating whether a pair of images and sentences matches or not. However, such sparse supervision covers a limited subset of image-text relationships, neglecting their inherent many-to-many correspondences; an image can be described in numerous texts at different descriptive levels. Moreover, existing approaches overlook the implicit connections from general to specific descriptions, which form the underlying rationale for the many-to-many relationships between vision and language. In this work, we propose descriptive image-text matching, called DITM, to learn the graded contextual similarity between image and text by exploring the descriptive flexibility of language. We formulate the descriptiveness score of each sentence with cumulative term frequency-inverse document frequency (TF-IDF) to balance the pairwise similarity according to the keywords in the sentence. Our method leverages sentence descriptiveness to learn robust image-text matching in two key ways: (1) to refine the false negative labeling, dynamically relaxing the connectivity between positive and negative pairs, and (2) to build more precise matching, aligning a set of relevant sentences in a generic-to-specific order. By moving beyond rigid binary supervision, DITM enhances the discovery of both optimal matches and potential positive pairs. Extensive experiments on MS-COCO, Flickr30K, and CxC datasets demonstrate the effectiveness of our method in representing complex image-text relationships compared to state-of-the-art approaches. In addition, DITM enhances the hierarchical reasoning ability of the model, supported by the extensive analysis on HierarCaps benchmark.
>
---
#### [new 021] Mission Balance: Generating Under-represented Class Samples using Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于计算机辅助手术领域，针对手术视频数据集中类别不平衡问题，提出一种两阶段文本条件扩散模型，通过分离时空建模生成高保真视频样本，并结合拒绝采样策略增强数据集。实验表明合成视频有效提升了动作识别和事件预测任务性能。**

- **链接: [http://arxiv.org/pdf/2505.09858v1](http://arxiv.org/pdf/2505.09858v1)**

> **作者:** Danush Kumar Venkatesh; Isabel Funke; Micha Pfeiffer; Fiona Kolbinger; Hanna Maria Schmeiser; Juergen Weitz; Marius Distler; Stefanie Speidel
>
> **备注:** Early accept at MICCAI 2025
>
> **摘要:** Computer-assisted interventions can improve intra-operative guidance, particularly through deep learning methods that harness the spatiotemporal information in surgical videos. However, the severe data imbalance often found in surgical video datasets hinders the development of high-performing models. In this work, we aim to overcome the data imbalance by synthesizing surgical videos. We propose a unique two-stage, text-conditioned diffusion-based method to generate high-fidelity surgical videos for under-represented classes. Our approach conditions the generation process on text prompts and decouples spatial and temporal modeling by utilizing a 2D latent diffusion model to capture spatial content and then integrating temporal attention layers to ensure temporal consistency. Furthermore, we introduce a rejection sampling strategy to select the most suitable synthetic samples, effectively augmenting existing datasets to address class imbalance. We evaluate our method on two downstream tasks-surgical action recognition and intra-operative event prediction-demonstrating that incorporating synthetic videos from our approach substantially enhances model performance. We open-source our implementation at https://gitlab.com/nct_tso_public/surgvgen.
>
---
#### [new 022] Logos as a Well-Tempered Pre-train for Sign Language Recognition
- **分类: cs.CV**

- **简介: 该论文针对孤立手语识别（ISLR）中跨语言数据稀缺及相似手势语义歧义问题，提出最大的俄罗斯手语数据集Logos，包含标注的相似手势组。通过预训练模型作为通用编码器，结合多分类头联合训练提升低资源任务准确率，在WLASL等数据集取得最优结果。**

- **链接: [http://arxiv.org/pdf/2505.10481v1](http://arxiv.org/pdf/2505.10481v1)**

> **作者:** Ilya Ovodov; Petr Surovtsev; Karina Kvanchiani; Alexander Kapitanov; Alexander Nagaev
>
> **摘要:** This paper examines two aspects of the isolated sign language recognition (ISLR) task. First, despite the availability of a number of datasets, the amount of data for most individual sign languages is limited. It poses the challenge of cross-language ISLR model training, including transfer learning. Second, similar signs can have different semantic meanings. It leads to ambiguity in dataset labeling and raises the question of the best policy for annotating such signs. To address these issues, this study presents Logos, a novel Russian Sign Language (RSL) dataset, the most extensive ISLR dataset by the number of signers and one of the largest available datasets while also the largest RSL dataset in size and vocabulary. It is shown that a model, pre-trained on the Logos dataset can be used as a universal encoder for other language SLR tasks, including few-shot learning. We explore cross-language transfer learning approaches and find that joint training using multiple classification heads benefits accuracy for the target lowresource datasets the most. The key feature of the Logos dataset is explicitly annotated visually similar sign groups. We show that explicitly labeling visually similar signs improves trained model quality as a visual encoder for downstream tasks. Based on the proposed contributions, we outperform current state-of-the-art results for the WLASL dataset and get competitive results for the AUTSL dataset, with a single stream model processing solely RGB video. The source code, dataset, and pre-trained models are publicly available.
>
---
#### [new 023] On the Interplay of Human-AI Alignment,Fairness, and Performance Trade-offs in Medical Imaging
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究医学影像中AI的公平性、人机协同与性能平衡问题，属于医疗AI系统优化任务。针对模型偏见导致的跨群体公平差距，提出通过校准人机协作策略（融入专家知识）减少不公平性并提升泛化能力，同时避免过度对齐引发的性能下降，旨在构建兼顾公平、鲁棒与效率的医疗AI。**

- **链接: [http://arxiv.org/pdf/2505.10231v1](http://arxiv.org/pdf/2505.10231v1)**

> **作者:** Haozhe Luo; Ziyu Zhou; Zixin Shu; Aurélie Pahud de Mortanges; Robert Berke; Mauricio Reyes
>
> **摘要:** Deep neural networks excel in medical imaging but remain prone to biases, leading to fairness gaps across demographic groups. We provide the first systematic exploration of Human-AI alignment and fairness in this domain. Our results show that incorporating human insights consistently reduces fairness gaps and enhances out-of-domain generalization, though excessive alignment can introduce performance trade-offs, emphasizing the need for calibrated strategies. These findings highlight Human-AI alignment as a promising approach for developing fair, robust, and generalizable medical AI systems, striking a balance between expert guidance and automated efficiency. Our code is available at https://github.com/Roypic/Aligner.
>
---
#### [new 024] ORL-LDM: Offline Reinforcement Learning Guided Latent Diffusion Model Super-Resolution Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对遥感图像超分辨率重建任务，提出基于强化学习的潜在扩散模型微调方法，解决现有技术处理复杂场景时细节保留不足的问题。通过构建强化学习环境，在LDM反向去噪过程中使用PPO算法优化决策，在RESISC45数据集上显著提升了PSNR、SSIM和LPIPS指标。**

- **链接: [http://arxiv.org/pdf/2505.10027v1](http://arxiv.org/pdf/2505.10027v1)**

> **作者:** Shijie Lyu
>
> **备注:** Accepted by the 4th International Conference on Computing Innovation and Applied Physics (CONF-CIAP 2025), and will be published in EAI Community Research Series-CORE or Theoretical and Natural Science (TNS)
>
> **摘要:** With the rapid advancement of remote sensing technology, super-resolution image reconstruction is of great research and practical significance. Existing deep learning methods have made progress but still face limitations in handling complex scenes and preserving image details. This paper proposes a reinforcement learning-based latent diffusion model (LDM) fine-tuning method for remote sensing image super-resolution. The method constructs a reinforcement learning environment with states, actions, and rewards, optimizing decision objectives through proximal policy optimization (PPO) during the reverse denoising process of the LDM model. Experiments on the RESISC45 dataset show significant improvements over the baseline model in PSNR, SSIM, and LPIPS, with PSNR increasing by 3-4dB, SSIM improving by 0.08-0.11, and LPIPS reducing by 0.06-0.10, particularly in structured and complex natural scenes. The results demonstrate the method's effectiveness in enhancing super-resolution quality and adaptability across scenes.
>
---
#### [new 025] Vision language models have difficulty recognizing virtual objects
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态场景理解评估任务，旨在解决视觉语言模型（VLMs）对图像虚拟物体的空间推理能力不足问题。研究者提出通过引入未实际呈现的虚拟对象（如"树上的风筝"）测试VLMs场景表征更新与空间关系推理能力，并对主流模型展开系统评测，验证其存在明显缺陷。**

- **链接: [http://arxiv.org/pdf/2505.10453v1](http://arxiv.org/pdf/2505.10453v1)**

> **作者:** Tyler Tran; Sangeet Khemlani; J. G. Trafton
>
> **摘要:** Vision language models (VLMs) are AI systems paired with both language and vision encoders to process multimodal input. They are capable of performing complex semantic tasks such as automatic captioning, but it remains an open question about how well they comprehend the visuospatial properties of scenes depicted in the images they process. We argue that descriptions of virtual objects -- objects that are not visually represented in an image -- can help test scene comprehension in these AI systems. For example, an image that depicts a person standing under a tree can be paired with the following prompt: imagine that a kite is stuck in the tree. VLMs that comprehend the scene should update their representations and reason sensibly about the spatial relations between all three objects. We describe systematic evaluations of state-of-the-art VLMs and show that their ability to process virtual objects is inadequate.
>
---
#### [new 026] MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting
- **分类: cs.CV**

- **简介: 该论文属于海洋雾检测与预测任务，旨在解决现有数据集单一区域/卫星导致的泛化性差问题。研究者构建了首个多区域、多卫星数据集MFogHub，整合15个地区和6颗卫星的6.8万+高分辨率样本，通过16个基线模型验证其能揭示区域/卫星差异对模型性能的影响，推动全球海洋雾监测及机理研究。**

- **链接: [http://arxiv.org/pdf/2505.10281v1](http://arxiv.org/pdf/2505.10281v1)**

> **作者:** Mengqiu Xu; Kaixin Chen; Heng Guo; Yixiang Huang; Ming Wu; Zhenwei Shi; Chuang Zhang; Jun Guo
>
> **摘要:** Deep learning approaches for marine fog detection and forecasting have outperformed traditional methods, demonstrating significant scientific and practical importance. However, the limited availability of open-source datasets remains a major challenge. Existing datasets, often focused on a single region or satellite, restrict the ability to evaluate model performance across diverse conditions and hinder the exploration of intrinsic marine fog characteristics. To address these limitations, we introduce \textbf{MFogHub}, the first multi-regional and multi-satellite dataset to integrate annotated marine fog observations from 15 coastal fog-prone regions and six geostationary satellites, comprising over 68,000 high-resolution samples. By encompassing diverse regions and satellite perspectives, MFogHub facilitates rigorous evaluation of both detection and forecasting methods under varying conditions. Extensive experiments with 16 baseline models demonstrate that MFogHub can reveal generalization fluctuations due to regional and satellite discrepancy, while also serving as a valuable resource for the development of targeted and scalable fog prediction techniques. Through MFogHub, we aim to advance both the practical monitoring and scientific understanding of marine fog dynamics on a global scale. The dataset and code are at \href{https://github.com/kaka0910/MFogHub}{https://github.com/kaka0910/MFogHub}.
>
---
#### [new 027] Dyadic Mamba: Long-term Dyadic Human Motion Synthesis
- **分类: cs.CV**

- **简介: 该论文研究文本驱动的长时双人动作生成任务，解决传统Transformer方法因位置编码限制难以生成长序列的问题。提出Dyadic Mamba模型，采用状态空间模型构建简单架构实现跨动作序列信息交互，无需复杂注意力机制。在短序列基准表现持平，长序列显著优于Transformer，并建立新长时动作评估标准。**

- **链接: [http://arxiv.org/pdf/2505.09827v1](http://arxiv.org/pdf/2505.09827v1)**

> **作者:** Julian Tanke; Takashi Shibuya; Kengo Uchida; Koichi Saito; Yuki Mitsufuji
>
> **备注:** CVPR 2025 HuMoGen Workshop
>
> **摘要:** Generating realistic dyadic human motion from text descriptions presents significant challenges, particularly for extended interactions that exceed typical training sequence lengths. While recent transformer-based approaches have shown promising results for short-term dyadic motion synthesis, they struggle with longer sequences due to inherent limitations in positional encoding schemes. In this paper, we introduce Dyadic Mamba, a novel approach that leverages State-Space Models (SSMs) to generate high-quality dyadic human motion of arbitrary length. Our method employs a simple yet effective architecture that facilitates information flow between individual motion sequences through concatenation, eliminating the need for complex cross-attention mechanisms. We demonstrate that Dyadic Mamba achieves competitive performance on standard short-term benchmarks while significantly outperforming transformer-based approaches on longer sequences. Additionally, we propose a new benchmark for evaluating long-term motion synthesis quality, providing a standardized framework for future research. Our results demonstrate that SSM-based architectures offer a promising direction for addressing the challenging task of long-term dyadic human motion synthesis from text descriptions.
>
---
#### [new 028] Advances in Radiance Field for Dynamic Scene: From Neural Field to Gaussian Field
- **分类: cs.CV**

- **简介: 该论文为动态场景重建的综述研究，系统归类分析200多篇辐射场方法，解决4D动态场景建模的复杂性问题。任务聚焦从神经场到高斯场的技术演进，比较运动表示、重建技术及时间一致性策略，总结挑战与未来方向，为领域提供系统参考。**

- **链接: [http://arxiv.org/pdf/2505.10049v1](http://arxiv.org/pdf/2505.10049v1)**

> **作者:** Jinlong Fan; Xuepu Zeng; Jing Zhang; Mingming Gong; Yuxiang Yang; Dacheng Tao
>
> **摘要:** Dynamic scene representation and reconstruction have undergone transformative advances in recent years, catalyzed by breakthroughs in neural radiance fields and 3D Gaussian splatting techniques. While initially developed for static environments, these methodologies have rapidly evolved to address the complexities inherent in 4D dynamic scenes through an expansive body of research. Coupled with innovations in differentiable volumetric rendering, these approaches have significantly enhanced the quality of motion representation and dynamic scene reconstruction, thereby garnering substantial attention from the computer vision and graphics communities. This survey presents a systematic analysis of over 200 papers focused on dynamic scene representation using radiance field, spanning the spectrum from implicit neural representations to explicit Gaussian primitives. We categorize and evaluate these works through multiple critical lenses: motion representation paradigms, reconstruction techniques for varied scene dynamics, auxiliary information integration strategies, and regularization approaches that ensure temporal consistency and physical plausibility. We organize diverse methodological approaches under a unified representational framework, concluding with a critical examination of persistent challenges and promising research directions. By providing this comprehensive overview, we aim to establish a definitive reference for researchers entering this rapidly evolving field while offering experienced practitioners a systematic understanding of both conceptual principles and practical frontiers in dynamic scene reconstruction.
>
---
#### [new 029] MorphGuard: Morph Specific Margin Loss for Enhancing Robustness to Face Morphing Attacks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10497v1](http://arxiv.org/pdf/2505.10497v1)**

> **作者:** Iurii Medvedev; Nuno Goncalves
>
> **摘要:** Face recognition has evolved significantly with the advancement of deep learning techniques, enabling its widespread adoption in various applications requiring secure authentication. However, this progress has also increased its exposure to presentation attacks, including face morphing, which poses a serious security threat by allowing one identity to impersonate another. Therefore, modern face recognition systems must be robust against such attacks. In this work, we propose a novel approach for training deep networks for face recognition with enhanced robustness to face morphing attacks. Our method modifies the classification task by introducing a dual-branch classification strategy that effectively handles the ambiguity in the labeling of face morphs. This adaptation allows the model to incorporate morph images into the training process, improving its ability to distinguish them from bona fide samples. Our strategy has been validated on public benchmarks, demonstrating its effectiveness in enhancing robustness against face morphing attacks. Furthermore, our approach is universally applicable and can be integrated into existing face recognition training pipelines to improve classification-based recognition methods.
>
---
#### [new 030] MIPHEI-ViT: Multiplex Immunofluorescence Prediction from H&E Images using ViT Foundation Models
- **分类: cs.CV; q-bio.TO; 68T07 (Primary), 92C55 (Secondary); I.4.9; I.2.10; I.5.4; J.3**

- **简介: 该论文属于医学图像分析任务，旨在通过H&E病理图像预测多标记免疫荧光（mIF）信号，以解决mIF临床应用成本高的问题。研究者提出MIPHEI模型，结合U-Net架构与视觉Transformer编码器，利用结直肠癌数据集训练验证，在多个细胞标记预测中优于现有方法，实现基于常规染色图像的细胞类型分析。**

- **链接: [http://arxiv.org/pdf/2505.10294v1](http://arxiv.org/pdf/2505.10294v1)**

> **作者:** Guillaume Balezo; Roger Trullo; Albert Pla Planas; Etienne Decenciere; Thomas Walter
>
> **摘要:** Histopathological analysis is a cornerstone of cancer diagnosis, with Hematoxylin and Eosin (H&E) staining routinely acquired for every patient to visualize cell morphology and tissue architecture. On the other hand, multiplex immunofluorescence (mIF) enables more precise cell type identification via proteomic markers, but has yet to achieve widespread clinical adoption due to cost and logistical constraints. To bridge this gap, we introduce MIPHEI (Multiplex Immunofluorescence Prediction from H&E), a U-Net-inspired architecture that integrates state-of-the-art ViT foundation models as encoders to predict mIF signals from H&E images. MIPHEI targets a comprehensive panel of markers spanning nuclear content, immune lineages (T cells, B cells, myeloid), epithelium, stroma, vasculature, and proliferation. We train our model using the publicly available ORION dataset of restained H&E and mIF images from colorectal cancer tissue, and validate it on two independent datasets. MIPHEI achieves accurate cell-type classification from H&E alone, with F1 scores of 0.88 for Pan-CK, 0.57 for CD3e, 0.56 for SMA, 0.36 for CD68, and 0.30 for CD20, substantially outperforming both a state-of-the-art baseline and a random classifier for most markers. Our results indicate that our model effectively captures the complex relationships between nuclear morphologies in their tissue context, as visible in H&E images and molecular markers defining specific cell types. MIPHEI offers a promising step toward enabling cell-type-aware analysis of large-scale H&E datasets, in view of uncovering relationships between spatial cellular organization and patient outcomes.
>
---
#### [new 031] Inferring Driving Maps by Deep Learning-based Trail Map Extraction
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究自动驾驶中的高精地图自动生成任务，旨在解决传统在线制图方法存在的时效性差、传感器遮挡等问题。通过整合车辆轨迹数据，提出基于Transformer的离线地图构建方法，支持持续更新并提升对未知环境的泛化能力，实验验证其优于现有在线方法。**

- **链接: [http://arxiv.org/pdf/2505.10258v1](http://arxiv.org/pdf/2505.10258v1)**

> **作者:** Michael Hubbertz; Pascal Colling; Qi Han; Tobias Meisen
>
> **备注:** This paper was accepted at the CVPR WAD 2025 Workshop
>
> **摘要:** High-definition (HD) maps offer extensive and accurate environmental information about the driving scene, making them a crucial and essential element for planning within autonomous driving systems. To avoid extensive efforts from manual labeling, methods for automating the map creation have emerged. Recent trends have moved from offline mapping to online mapping, ensuring availability and actuality of the utilized maps. While the performance has increased in recent years, online mapping still faces challenges regarding temporal consistency, sensor occlusion, runtime, and generalization. We propose a novel offline mapping approach that integrates trails - informal routes used by drivers - into the map creation process. Our method aggregates trail data from the ego vehicle and other traffic participants to construct a comprehensive global map using transformer-based deep learning models. Unlike traditional offline mapping, our approach enables continuous updates while remaining sensor-agnostic, facilitating efficient data transfer. Our method demonstrates superior performance compared to state-of-the-art online mapping approaches, achieving improved generalization to previously unseen environments and sensor configurations. We validate our approach on two benchmark datasets, highlighting its robustness and applicability in autonomous driving systems.
>
---
#### [new 032] VRU-CIPI: Crossing Intention Prediction at Intersections for Improving Vulnerable Road Users Safety
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于行人行为预测任务，旨在解决弱势道路用户（VRU）在路口过街意图误判引发的安全隐患。提出VRU-CIPI框架，结合门控循环单元（GRU）和Transformer多头自注意力机制，捕捉VRU运动时序特征及空间上下文依赖，实现96.45%的高精度实时预测，并通过车路协同系统提前触发信号预警，提升路口交互安全。**

- **链接: [http://arxiv.org/pdf/2505.09935v1](http://arxiv.org/pdf/2505.09935v1)**

> **作者:** Ahmed S. Abdelrahman; Mohamed Abdel-Aty; Quoc Dai Tran
>
> **摘要:** Understanding and predicting human behavior in-thewild, particularly at urban intersections, remains crucial for enhancing interaction safety between road users. Among the most critical behaviors are crossing intentions of Vulnerable Road Users (VRUs), where misinterpretation may result in dangerous conflicts with oncoming vehicles. In this work, we propose the VRU-CIPI framework with a sequential attention-based model designed to predict VRU crossing intentions at intersections. VRU-CIPI employs Gated Recurrent Unit (GRU) to capture temporal dynamics in VRU movements, combined with a multi-head Transformer self-attention mechanism to encode contextual and spatial dependencies critical for predicting crossing direction. Evaluated on UCF-VRU dataset, our proposed achieves state-of-the-art performance with an accuracy of 96.45% and achieving real-time inference speed reaching 33 frames per second. Furthermore, by integrating with Infrastructure-to-Vehicles (I2V) communication, our approach can proactively enhance intersection safety through timely activation of crossing signals and providing early warnings to connected vehicles, ensuring smoother and safer interactions for all road users.
>
---
#### [new 033] BoundarySeg:An Embarrassingly Simple Method To Boost Medical Image Segmentation Performance for Low Data Regimes
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割任务，解决标注数据稀缺时半监督方法依赖未标注数据的问题。提出BoundarySeg多任务框架，通过联合训练器官分割与边界预测任务，利用预测一致性增强监督，在低数据条件下实现与先进半监督方法相当的性能，无需额外未标注数据。**

- **链接: [http://arxiv.org/pdf/2505.09829v1](http://arxiv.org/pdf/2505.09829v1)**

> **作者:** Tushar Kataria; Shireen Y. Elhabian
>
> **摘要:** Obtaining large-scale medical data, annotated or unannotated, is challenging due to stringent privacy regulations and data protection policies. In addition, annotating medical images requires that domain experts manually delineate anatomical structures, making the process both time-consuming and costly. As a result, semi-supervised methods have gained popularity for reducing annotation costs. However, the performance of semi-supervised methods is heavily dependent on the availability of unannotated data, and their effectiveness declines when such data are scarce or absent. To overcome this limitation, we propose a simple, yet effective and computationally efficient approach for medical image segmentation that leverages only existing annotations. We propose BoundarySeg , a multi-task framework that incorporates organ boundary prediction as an auxiliary task to full organ segmentation, leveraging consistency between the two task predictions to provide additional supervision. This strategy improves segmentation accuracy, especially in low data regimes, allowing our method to achieve performance comparable to or exceeding state-of-the-art semi supervised approaches all without relying on unannotated data or increasing computational demands. Code will be released upon acceptance.
>
---
#### [new 034] MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态数学推理任务，旨在解决现有模型因缺乏数学图表细节而推理受限的问题。通过代码监督实现视觉与代码跨模态对齐，开发了FigCodifier模型和ImgCode-8.6M数据集，并合成MM-MathInstruct-3M指令数据。最终训练的MathCoder-VL模型在MathVista几何任务上超越GPT-4o和Claude 3.5，达到开源SOTA。**

- **链接: [http://arxiv.org/pdf/2505.10557v1](http://arxiv.org/pdf/2505.10557v1)**

> **作者:** Ke Wang; Junting Pan; Linda Wei; Aojun Zhou; Weikang Shi; Zimu Lu; Han Xiao; Yunqiao Yang; Houxing Ren; Mingjie Zhan; Hongsheng Li
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Natural language image-caption datasets, widely used for training Large Multimodal Models, mainly focus on natural scenarios and overlook the intricate details of mathematical figures that are critical for problem-solving, hindering the advancement of current LMMs in multimodal mathematical reasoning. To this end, we propose leveraging code as supervision for cross-modal alignment, since code inherently encodes all information needed to generate corresponding figures, establishing a precise connection between the two modalities. Specifically, we co-develop our image-to-code model and dataset with model-in-the-loop approach, resulting in an image-to-code model, FigCodifier and ImgCode-8.6M dataset, the largest image-code dataset to date. Furthermore, we utilize FigCodifier to synthesize novel mathematical figures and then construct MM-MathInstruct-3M, a high-quality multimodal math instruction fine-tuning dataset. Finally, we present MathCoder-VL, trained with ImgCode-8.6M for cross-modal alignment and subsequently fine-tuned on MM-MathInstruct-3M for multimodal math problem solving. Our model achieves a new open-source SOTA across all six metrics. Notably, it surpasses GPT-4o and Claude 3.5 Sonnet in the geometry problem-solving subset of MathVista, achieving improvements of 8.9% and 9.2%. The dataset and models will be released at https://github.com/mathllm/MathCoder.
>
---
#### [new 035] Enhancing Multi-Image Question Answering via Submodular Subset Selection
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多图像问答任务中模型扩展性差、检索性能低的问题，提出基于子模函数（如GraphCut）的预选方法，从大规模图像集中筛选语义相关子集，优化现有检索框架（MIRAGE），并通过锚点查询和数据增强提升效果。**

- **链接: [http://arxiv.org/pdf/2505.10533v1](http://arxiv.org/pdf/2505.10533v1)**

> **作者:** Aaryan Sharma; Shivansh Gupta; Samar Agarwal; Vishak Prasad C.; Ganesh Ramakrishnan
>
> **摘要:** Large multimodal models (LMMs) have achieved high performance in vision-language tasks involving single image but they struggle when presented with a collection of multiple images (Multiple Image Question Answering scenario). These tasks, which involve reasoning over large number of images, present issues in scalability (with increasing number of images) and retrieval performance. In this work, we propose an enhancement for retriever framework introduced in MIRAGE model using submodular subset selection techniques. Our method leverages query-aware submodular functions, such as GraphCut, to pre-select a subset of semantically relevant images before main retrieval component. We demonstrate that using anchor-based queries and augmenting the data improves submodular-retriever pipeline effectiveness, particularly in large haystack sizes.
>
---
#### [new 036] DeepSeqCoco: A Robust Mobile Friendly Deep Learning Model for Detection of Diseases in Cocos nucifera
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像识别任务，旨在解决椰子树病害人工检测效率低、难扩展的问题。研究者提出了DeepSeqCoco深度学习模型，通过优化器混合策略（SGD-Adam）平衡精度与计算成本，实现99.5%准确率，训练和预测时间分别减少18%和85%，为农业提供高效AI监测方案。**

- **链接: [http://arxiv.org/pdf/2505.10030v1](http://arxiv.org/pdf/2505.10030v1)**

> **作者:** Miit Daga; Dhriti Parikh; Swarna Priya Ramu
>
> **备注:** This paper is accepted for publication in IEEE Access journal and is currently pending revisions before publication
>
> **摘要:** Coconut tree diseases are a serious risk to agricultural yield, particularly in developing countries where conventional farming practices restrict early diagnosis and intervention. Current disease identification methods are manual, labor-intensive, and non-scalable. In response to these limitations, we come up with DeepSeqCoco, a deep learning based model for accurate and automatic disease identification from coconut tree images. The model was tested under various optimizer settings, such as SGD, Adam, and hybrid configurations, to identify the optimal balance between accuracy, minimization of loss, and computational cost. Results from experiments indicate that DeepSeqCoco can achieve as much as 99.5% accuracy (achieving up to 5% higher accuracy than existing models) with the hybrid SGD-Adam showing the lowest validation loss of 2.81%. It also shows a drop of up to 18% in training time and up to 85% in prediction time for input images. The results point out the promise of the model to improve precision agriculture through an AI-based, scalable, and efficient disease monitoring system.
>
---
#### [new 037] ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization
- **分类: cs.CV**

- **简介: 该论文研究单目图像3D人体网格恢复任务，针对现有概率方法预测与2D观测错位、鲁棒性差的问题，提出ADHMR框架。通过训练无监督评估模型HMR-Scorer构建偏好数据集，采用直接偏好优化对齐扩散模型预测，并利用该评估模型进行数据清洗，提升模型对齐能力和性能，实验验证其超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.10250v1](http://arxiv.org/pdf/2505.10250v1)**

> **作者:** Wenhao Shen; Wanqi Yin; Xiaofeng Yang; Cheng Chen; Chaoyue Song; Zhongang Cai; Lei Yang; Hao Wang; Guosheng Lin
>
> **备注:** Accepted by ICML 2025. Code: https://github.com/shenwenhao01/ADHMR
>
> **摘要:** Human mesh recovery (HMR) from a single image is inherently ill-posed due to depth ambiguity and occlusions. Probabilistic methods have tried to solve this by generating numerous plausible 3D human mesh predictions, but they often exhibit misalignment with 2D image observations and weak robustness to in-the-wild images. To address these issues, we propose ADHMR, a framework that Aligns a Diffusion-based HMR model in a preference optimization manner. First, we train a human mesh prediction assessment model, HMR-Scorer, capable of evaluating predictions even for in-the-wild images without 3D annotations. We then use HMR-Scorer to create a preference dataset, where each input image has a pair of winner and loser mesh predictions. This dataset is used to finetune the base model using direct preference optimization. Moreover, HMR-Scorer also helps improve existing HMR models by data cleaning, even with fewer training samples. Extensive experiments show that ADHMR outperforms current state-of-the-art methods. Code is available at: https://github.com/shenwenhao01/ADHMR.
>
---
#### [new 038] IMITATE: Image Registration with Context for unknown time frame recovery
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究图像配准任务，旨在解决放疗中4D-CT扫描因呼吸不规则导致的3D重建伪影问题。提出基于条件U-Net的新方法，无需固定参考图像，通过上下文建模实现未知呼吸相位图像估计，成功应用于胸腹肿瘤运动插值，在临床数据上实现了无伪影实时重建。**

- **链接: [http://arxiv.org/pdf/2505.10124v1](http://arxiv.org/pdf/2505.10124v1)**

> **作者:** Ziad Kheil; Lucas Robinet; Laurent Risser; Soleakhena Ken
>
> **备注:** IEEE ISBI 2025
>
> **摘要:** In this paper, we formulate a novel image registration formalism dedicated to the estimation of unknown condition-related images, based on two or more known images and their associated conditions. We show how to practically model this formalism by using a new conditional U-Net architecture, which fully takes into account the conditional information and does not need any fixed image. Our formalism is then applied to image moving tumors for radiotherapy treatment at different breathing amplitude using 4D-CT (3D+t) scans in thoracoabdominal regions. This driving application is particularly complex as it requires to stitch a collection of sequential 2D slices into several 3D volumes at different organ positions. Movement interpolation with standard methods then generates well known reconstruction artefacts in the assembled volumes due to irregular patient breathing, hysteresis and poor correlation of breathing signal to internal motion. Results obtained on 4D-CT clinical data showcase artefact-free volumes achieved through real-time latencies. The code is publicly available at https://github.com/Kheil-Z/IMITATE .
>
---
#### [new 039] VolE: A Point-cloud Framework for Food 3D Reconstruction and Volume Estimation
- **分类: cs.CV**

- **简介: 该论文属于食品3D重建与体积估计任务，旨在解决现有方法依赖专用硬件、深度信息或参考物的问题。提出VolE框架：利用移动设备AR自由拍摄图像生成3D点云模型，通过视频分割生成食物掩膜，无需参考物或深度传感器。构建新数据集验证方案，实验显示其平均绝对百分比误差仅2.22%，优于现有技术。**

- **链接: [http://arxiv.org/pdf/2505.10205v1](http://arxiv.org/pdf/2505.10205v1)**

> **作者:** Umair Haroon; Ahmad AlMughrabi; Thanasis Zoumpekas; Ricardo Marques; Petia Radeva
>
> **摘要:** Accurate food volume estimation is crucial for medical nutrition management and health monitoring applications, but current food volume estimation methods are often limited by mononuclear data, leveraging single-purpose hardware such as 3D scanners, gathering sensor-oriented information such as depth information, or relying on camera calibration using a reference object. In this paper, we present VolE, a novel framework that leverages mobile device-driven 3D reconstruction to estimate food volume. VolE captures images and camera locations in free motion to generate precise 3D models, thanks to AR-capable mobile devices. To achieve real-world measurement, VolE is a reference- and depth-free framework that leverages food video segmentation for food mask generation. We also introduce a new food dataset encompassing the challenging scenarios absent in the previous benchmarks. Our experiments demonstrate that VolE outperforms the existing volume estimation techniques across multiple datasets by achieving 2.22 % MAPE, highlighting its superior performance in food volume estimation.
>
---
#### [new 040] PointArena: Probing Multimodal Grounding Through Language-Guided Pointing
- **分类: cs.CV**

- **简介: 该论文属于多模态模型评估任务，旨在解决现有基准局限于对象定位的问题。作者构建了PointArena评估平台，包含数据集、交互评测场和机器人系统，测试语言引导的视觉指向能力。通过评估主流模型发现，专用训练显著提升性能，精确指向对连接抽象推理与现实行动至关重要。**

- **链接: [http://arxiv.org/pdf/2505.09990v1](http://arxiv.org/pdf/2505.09990v1)**

> **作者:** Long Cheng; Jiafei Duan; Yi Ru Wang; Haoquan Fang; Boyang Li; Yushan Huang; Elvis Wang; Ainaz Eftekhar; Jason Lee; Wentao Yuan; Rose Hendrix; Noah A. Smith; Fei Xia; Dieter Fox; Ranjay Krishna
>
> **备注:** 10 Pages, Dataset and code:https://pointarena.github.io/
>
> **摘要:** Pointing serves as a fundamental and intuitive mechanism for grounding language within visual contexts, with applications spanning robotics, assistive technologies, and interactive AI systems. While recent multimodal models have started to support pointing capabilities, existing benchmarks typically focus only on referential object localization tasks. We introduce PointArena, a comprehensive platform for evaluating multimodal pointing across diverse reasoning scenarios. PointArena comprises three components: (1) Point-Bench, a curated dataset containing approximately 1,000 pointing tasks across five reasoning categories; (2) Point-Battle, an interactive, web-based arena facilitating blind, pairwise model comparisons, which has already gathered over 4,500 anonymized votes; and (3) Point-Act, a real-world robotic manipulation system allowing users to directly evaluate multimodal model pointing capabilities in practical settings. We conducted extensive evaluations of both state-of-the-art open-source and proprietary multimodal models. Results indicate that Molmo-72B consistently outperforms other models, though proprietary models increasingly demonstrate comparable performance. Additionally, we find that supervised training specifically targeting pointing tasks significantly enhances model performance. Across our multi-stage evaluation pipeline, we also observe strong correlations, underscoring the critical role of precise pointing capabilities in enabling multimodal models to effectively bridge abstract reasoning with concrete, real-world actions. Project page: https://pointarena.github.io/
>
---
#### [new 041] StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation
- **分类: cs.CV; cs.CL; I.2.10; I.2.7**

- **简介: 该论文属于视觉叙事任务，旨在解决角色一致性差和指代幻觉问题。通过构建StoryReasoning数据集（含结构化场景分析和视觉关联故事），结合跨帧目标重识别、链式推理及视觉实体关联方法，提出Qwen Storyteller模型，将幻觉率降低12.3%。**

- **链接: [http://arxiv.org/pdf/2505.10292v1](http://arxiv.org/pdf/2505.10292v1)**

> **作者:** Daniel A. P. Oliveira; David Martins de Matos
>
> **备注:** 31 pages, 14 figures
>
> **摘要:** Visual storytelling systems struggle to maintain character identity across frames and link actions to appropriate subjects, frequently leading to referential hallucinations. These issues can be addressed through grounding of characters, objects, and other entities on the visual elements. We propose StoryReasoning, a dataset containing 4,178 stories derived from 52,016 movie images, with both structured scene analyses and grounded stories. Each story maintains character and object consistency across frames while explicitly modeling multi-frame relationships through structured tabular representations. Our approach features cross-frame object re-identification using visual similarity and face recognition, chain-of-thought reasoning for explicit narrative modeling, and a grounding scheme that links textual elements to visual entities across multiple frames. We establish baseline performance by fine-tuning Qwen2.5-VL 7B, creating Qwen Storyteller, which performs end-to-end object detection, re-identification, and landmark detection while maintaining consistent object references throughout the story. Evaluation demonstrates a reduction from 4.06 to 3.56 (-12.3%) hallucinations on average per story when compared to a non-fine-tuned model.
>
---
#### [new 042] Modeling Saliency Dataset Bias
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对图像显着性预测中的数据集偏差问题，提出通过少量可解释参数（多尺度、中心偏差等）适配不同数据集，提升跨数据集泛化能力。研究揭示了60%性能差距源于数据集特有偏差，新模型仅用20个参数调整即可填补75%差距，在MIT/Tuebingen基准中达到最优，并解析了空间显着性机制。**

- **链接: [http://arxiv.org/pdf/2505.10169v1](http://arxiv.org/pdf/2505.10169v1)**

> **作者:** Matthias Kümmerer; Harneet Khanuja; Matthias Bethge
>
> **摘要:** Recent advances in image-based saliency prediction are approaching gold standard performance levels on existing benchmarks. Despite this success, we show that predicting fixations across multiple saliency datasets remains challenging due to dataset bias. We find a significant performance drop (around 40%) when models trained on one dataset are applied to another. Surprisingly, increasing dataset diversity does not resolve this inter-dataset gap, with close to 60% attributed to dataset-specific biases. To address this remaining generalization gap, we propose a novel architecture extending a mostly dataset-agnostic encoder-decoder structure with fewer than 20 dataset-specific parameters that govern interpretable mechanisms such as multi-scale structure, center bias, and fixation spread. Adapting only these parameters to new data accounts for more than 75% of the generalization gap, with a large fraction of the improvement achieved with as few as 50 samples. Our model sets a new state-of-the-art on all three datasets of the MIT/Tuebingen Saliency Benchmark (MIT300, CAT2000, and COCO-Freeview), even when purely generalizing from unrelated datasets, but with a substantial boost when adapting to the respective training datasets. The model also provides valuable insights into spatial saliency properties, revealing complex multi-scale effects that combine both absolute and relative sizes.
>
---
#### [new 043] SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出SpikeVideoFormer，一种基于脉冲神经网络的视频Transformer，用于视频分类、姿态跟踪和语义分割。针对现有SNN模型在视频任务中时空特征利用不足、计算复杂度高的问题，设计了线性时间复杂度的脉冲驱动汉明注意力机制，通过优化时空注意力结构实现高效视频处理。模型在多项任务中达到SNN的SOTA性能，效率超过ANN方法5-16倍。**

- **链接: [http://arxiv.org/pdf/2505.10352v1](http://arxiv.org/pdf/2505.10352v1)**

> **作者:** Shihao Zou; Qingfeng Li; Wei Ji; Jingjing Li; Yongkui Yang; Guoqi Li; Chao Dong
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Spiking Neural Networks (SNNs) have shown competitive performance to Artificial Neural Networks (ANNs) in various vision tasks, while offering superior energy efficiency. However, existing SNN-based Transformers primarily focus on single-image tasks, emphasizing spatial features while not effectively leveraging SNNs' efficiency in video-based vision tasks. In this paper, we introduce SpikeVideoFormer, an efficient spike-driven video Transformer, featuring linear temporal complexity $\mathcal{O}(T)$. Specifically, we design a spike-driven Hamming attention (SDHA) which provides a theoretically guided adaptation from traditional real-valued attention to spike-driven attention. Building on SDHA, we further analyze various spike-driven space-time attention designs and identify an optimal scheme that delivers appealing performance for video tasks, while maintaining only linear temporal complexity. The generalization ability and efficiency of our model are demonstrated across diverse downstream video tasks, including classification, human pose tracking, and semantic segmentation. Empirical results show our method achieves state-of-the-art (SOTA) performance compared to existing SNN approaches, with over 15\% improvement on the latter two tasks. Additionally, it matches the performance of recent ANN-based methods while offering significant efficiency gains, achieving $\times 16$, $\times 10$ and $\times 5$ improvements on the three tasks. https://github.com/JimmyZou/SpikeVideoFormer
>
---
#### [new 044] 3D-Fixup: Advancing Photo Editing with 3D Priors
- **分类: cs.CV**

- **简介: 该论文属于3D感知图像编辑任务，旨在解决单图3D编辑中物体平移/旋转的难题。提出3D-Fixup框架，通过视频数据生成训练对，结合扩散模型与Image-to-3D模型的3D指导，将2D信息投影至3D空间，实现高质量身份保持的复杂编辑。**

- **链接: [http://arxiv.org/pdf/2505.10566v1](http://arxiv.org/pdf/2505.10566v1)**

> **作者:** Yen-Chi Cheng; Krishna Kumar Singh; Jae Shin Yoon; Alex Schwing; Liangyan Gui; Matheus Gadelha; Paul Guerrero; Nanxuan Zhao
>
> **备注:** SIGGRAPH 2025. Project page: https://3dfixup.github.io/
>
> **摘要:** Despite significant advances in modeling image priors via diffusion models, 3D-aware image editing remains challenging, in part because the object is only specified via a single image. To tackle this challenge, we propose 3D-Fixup, a new framework for editing 2D images guided by learned 3D priors. The framework supports difficult editing situations such as object translation and 3D rotation. To achieve this, we leverage a training-based approach that harnesses the generative power of diffusion models. As video data naturally encodes real-world physical dynamics, we turn to video data for generating training data pairs, i.e., a source and a target frame. Rather than relying solely on a single trained model to infer transformations between source and target frames, we incorporate 3D guidance from an Image-to-3D model, which bridges this challenging task by explicitly projecting 2D information into 3D space. We design a data generation pipeline to ensure high-quality 3D guidance throughout training. Results show that by integrating these 3D priors, 3D-Fixup effectively supports complex, identity coherent 3D-aware edits, achieving high-quality results and advancing the application of diffusion models in realistic image manipulation. The code is provided at https://3dfixup.github.io/
>
---
#### [new 045] A Computational Pipeline for Advanced Analysis of 4D Flow MRI in the Left Atrium
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决左心房4D血流MRI分析中因低流速、分辨率不足及缺乏统一框架导致的参数提取难题。作者开发了首个开源计算框架，支持跨中心多协议数据的高精度自动化分割（Dice>0.9），并首次系统性评估能量、涡度等血流参数作为疾病预后生物标志物的潜力。**

- **链接: [http://arxiv.org/pdf/2505.09746v1](http://arxiv.org/pdf/2505.09746v1)**

> **作者:** Xabier Morales; Ayah Elsayed; Debbie Zhao; Filip Loncaric; Ainhoa Aguado; Mireia Masias; Gina Quill; Marc Ramos; Ada Doltra; Ana Garcia; Marta Sitges; David Marlevi; Alistair Young; Martyn Nash; Bart Bijnens; Oscar Camara
>
> **摘要:** The left atrium (LA) plays a pivotal role in modulating left ventricular filling, but our comprehension of its hemodynamics is significantly limited by the constraints of conventional ultrasound analysis. 4D flow magnetic resonance imaging (4D Flow MRI) holds promise for enhancing our understanding of atrial hemodynamics. However, the low velocities within the LA and the limited spatial resolution of 4D Flow MRI make analyzing this chamber challenging. Furthermore, the absence of dedicated computational frameworks, combined with diverse acquisition protocols and vendors, complicates gathering large cohorts for studying the prognostic value of hemodynamic parameters provided by 4D Flow MRI. In this study, we introduce the first open-source computational framework tailored for the analysis of 4D Flow MRI in the LA, enabling comprehensive qualitative and quantitative analysis of advanced hemodynamic parameters. Our framework proves robust to data from different centers of varying quality, producing high-accuracy automated segmentations (Dice $>$ 0.9 and Hausdorff 95 $<$ 3 mm), even with limited training data. Additionally, we conducted the first comprehensive assessment of energy, vorticity, and pressure parameters in the LA across a spectrum of disorders to investigate their potential as prognostic biomarkers.
>
---
#### [new 046] Sage Deer: A Super-Aligned Driving Generalist Is Your Copilot
- **分类: cs.CV**

- **简介: 该论文研究智能驾驶座舱的个性化适配问题，属于自动驾驶人机交互领域。提出SAGE DeeR智能体，通过超级对齐（适应用户偏好）、通用多模态推理（分析生理、场景等数据）及自我激发思维链，解决用户需求匹配难题，并构建基准测试评估性能。**

- **链接: [http://arxiv.org/pdf/2505.10257v1](http://arxiv.org/pdf/2505.10257v1)**

> **作者:** Hao Lu; Jiaqi Tang; Jiyao Wang; Yunfan LU; Xu Cao; Qingyong Hu; Yin Wang; Yuting Zhang; Tianxin Xie; Yunpeng Zhang; Yong Chen; Jiayu. Gao; Bin Huang; Dengbo He; Shuiguang Deng; Hao Chen; Ying-Cong Chen
>
> **摘要:** The intelligent driving cockpit, an important part of intelligent driving, needs to match different users' comfort, interaction, and safety needs. This paper aims to build a Super-Aligned and GEneralist DRiving agent, SAGE DeeR. Sage Deer achieves three highlights: (1) Super alignment: It achieves different reactions according to different people's preferences and biases. (2) Generalist: It can understand the multi-view and multi-mode inputs to reason the user's physiological indicators, facial emotions, hand movements, body movements, driving scenarios, and behavioral decisions. (3) Self-Eliciting: It can elicit implicit thought chains in the language space to further increase generalist and super-aligned abilities. Besides, we collected multiple data sets and built a large-scale benchmark. This benchmark measures the deer's perceptual decision-making ability and the super alignment's accuracy.
>
---
#### [new 047] TKFNet: Learning Texture Key Factor Driven Feature for Facial Expression Recognition
- **分类: cs.CV**

- **简介: 该论文针对自然场景下面部表情识别（FER）任务，解决因表情特征细微、局部及面部外观复杂变化导致的识别难题。提出TKFNet框架，通过挖掘纹理关键驱动因素（TKDF）捕捉眉、眼、嘴部微纹理变化，设计纹理感知特征提取器（TAFE）和双上下文信息过滤模块（DCIF），结合注意力机制优化特征表达，在主流数据集上实现最优性能。**

- **链接: [http://arxiv.org/pdf/2505.09967v1](http://arxiv.org/pdf/2505.09967v1)**

> **作者:** Liqian Deng
>
> **摘要:** Facial expression recognition (FER) in the wild remains a challenging task due to the subtle and localized nature of expression-related features, as well as the complex variations in facial appearance. In this paper, we introduce a novel framework that explicitly focuses on Texture Key Driver Factors (TKDF), localized texture regions that exhibit strong discriminative power across emotional categories. By carefully observing facial image patterns, we identify that certain texture cues, such as micro-changes in skin around the brows, eyes, and mouth, serve as primary indicators of emotional dynamics. To effectively capture and leverage these cues, we propose a FER architecture comprising a Texture-Aware Feature Extractor (TAFE) and Dual Contextual Information Filtering (DCIF). TAFE employs a ResNet-based backbone enhanced with multi-branch attention to extract fine-grained texture representations, while DCIF refines these features by filtering context through adaptive pooling and attention mechanisms. Experimental results on RAF-DB and KDEF datasets demonstrate that our method achieves state-of-the-art performance, verifying the effectiveness and robustness of incorporating TKDFs into FER pipelines.
>
---
#### [new 048] HandReader: Advanced Techniques for Efficient Fingerspelling Recognition
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于手语指拼识别任务，旨在提升动态手部动作识别的准确率。针对现有方法处理时序信息不足的问题，提出三种架构：基于RGB视频的TSAM模块、基于关键点的TPE编码器及融合两者的联合模型，并在多个数据集取得SOTA结果。同时发布了首个俄语指拼数据集Znaki及预训练模型。**

- **链接: [http://arxiv.org/pdf/2505.10267v1](http://arxiv.org/pdf/2505.10267v1)**

> **作者:** Pavel Korotaev; Petr Surovtsev; Alexander Kapitanov; Karina Kvanchiani; Aleksandr Nagaev
>
> **备注:** https://github.com/ai-forever/handreader
>
> **摘要:** Fingerspelling is a significant component of Sign Language (SL), allowing the interpretation of proper names, characterized by fast hand movements during signing. Although previous works on fingerspelling recognition have focused on processing the temporal dimension of videos, there remains room for improving the accuracy of these approaches. This paper introduces HandReader, a group of three architectures designed to address the fingerspelling recognition task. HandReader$_{RGB}$ employs the novel Temporal Shift-Adaptive Module (TSAM) to process RGB features from videos of varying lengths while preserving important sequential information. HandReader$_{KP}$ is built on the proposed Temporal Pose Encoder (TPE) operated on keypoints as tensors. Such keypoints composition in a batch allows the encoder to pass them through 2D and 3D convolution layers, utilizing temporal and spatial information and accumulating keypoints coordinates. We also introduce HandReader_RGB+KP - architecture with a joint encoder to benefit from RGB and keypoint modalities. Each HandReader model possesses distinct advantages and achieves state-of-the-art results on the ChicagoFSWild and ChicagoFSWild+ datasets. Moreover, the models demonstrate high performance on the first open dataset for Russian fingerspelling, Znaki, presented in this paper. The Znaki dataset and HandReader pre-trained models are publicly available.
>
---
#### [new 049] Large-Scale Gaussian Splatting SLAM
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉SLAM任务，旨在解决现有方法依赖RGBD传感器且无法适应大规模户外场景的问题。提出LSG-SLAM系统，采用立体相机实现多模态位姿估计，通过特征对齐约束和连续高斯子地图处理大范围场景，结合环路检测与结构优化提升重建质量，在EuRoc/KITTI数据集上超越传统及神经方法。**

- **链接: [http://arxiv.org/pdf/2505.09915v1](http://arxiv.org/pdf/2505.09915v1)**

> **作者:** Zhe Xin; Chenyang Wu; Penghui Huang; Yanyong Zhang; Yinian Mao; Guoquan Huang
>
> **摘要:** The recently developed Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have shown encouraging and impressive results for visual SLAM. However, most representative methods require RGBD sensors and are only available for indoor environments. The robustness of reconstruction in large-scale outdoor scenarios remains unexplored. This paper introduces a large-scale 3DGS-based visual SLAM with stereo cameras, termed LSG-SLAM. The proposed LSG-SLAM employs a multi-modality strategy to estimate prior poses under large view changes. In tracking, we introduce feature-alignment warping constraints to alleviate the adverse effects of appearance similarity in rendering losses. For the scalability of large-scale scenarios, we introduce continuous Gaussian Splatting submaps to tackle unbounded scenes with limited memory. Loops are detected between GS submaps by place recognition and the relative pose between looped keyframes is optimized utilizing rendering and feature warping losses. After the global optimization of camera poses and Gaussian points, a structure refinement module enhances the reconstruction quality. With extensive evaluations on the EuRoc and KITTI datasets, LSG-SLAM achieves superior performance over existing Neural, 3DGS-based, and even traditional approaches. Project page: https://lsg-slam.github.io.
>
---
#### [new 050] MSCI: Addressing CLIP's Inherent Limitations for Compositional Zero-Shot Learning
- **分类: cs.CV**

- **简介: 该论文针对组合零样本学习任务，解决CLIP模型因架构限制难以捕捉细粒度特征的问题。提出多阶段跨模态交互模型(MSCI)，通过自适应聚合器提取CLIP视觉编码器的低层局部特征与高层全局特征，分阶段融合至文本表示，动态调整注意力权重增强细粒度感知能力，实验验证了模型优越性。**

- **链接: [http://arxiv.org/pdf/2505.10289v1](http://arxiv.org/pdf/2505.10289v1)**

> **作者:** Yue Wang; Shuai Xu; Xuelin Zhu; Yicong Li
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Compositional Zero-Shot Learning (CZSL) aims to recognize unseen state-object combinations by leveraging known combinations. Existing studies basically rely on the cross-modal alignment capabilities of CLIP but tend to overlook its limitations in capturing fine-grained local features, which arise from its architectural and training paradigm. To address this issue, we propose a Multi-Stage Cross-modal Interaction (MSCI) model that effectively explores and utilizes intermediate-layer information from CLIP's visual encoder. Specifically, we design two self-adaptive aggregators to extract local information from low-level visual features and integrate global information from high-level visual features, respectively. These key information are progressively incorporated into textual representations through a stage-by-stage interaction mechanism, significantly enhancing the model's perception capability for fine-grained local visual information. Additionally, MSCI dynamically adjusts the attention weights between global and local visual information based on different combinations, as well as different elements within the same combination, allowing it to flexibly adapt to diverse scenarios. Experiments on three widely used datasets fully validate the effectiveness and superiority of the proposed model. Data and code are available at https://github.com/ltpwy/MSCI.
>
---
#### [new 051] Few-Shot Learning of Visual Compositional Concepts through Probabilistic Schema Induction
- **分类: cs.CV**

- **简介: 论文研究小样本视觉组合概念学习任务，解决传统模型缺乏结构化表示及类比推理的问题。提出概率模式归纳（PSI）模型，利用深度学习进行类比映射，权衡对象与关系相似性，放大关键关系。实验表明PSI性能接近人类，优于对照组，验证了结构化表示与类比映射的重要性。**

- **链接: [http://arxiv.org/pdf/2505.09859v1](http://arxiv.org/pdf/2505.09859v1)**

> **作者:** Andrew Jun Lee; Taylor Webb; Trevor Bihl; Keith Holyoak; Hongjing Lu
>
> **备注:** Lee, A. J., Webb, T., Bihl, T., Holyoak, K. J., & Lu, H. (2025). Few-shot learning of visual compositional concepts through probabilistic schema induction. In A. Ruggeri, D. Barner, C. Walker, & N. Bramley (Eds.), Proceedings of the 47th Annual Conference of the Cognitive Science Society. Cognitive Science Society
>
> **摘要:** The ability to learn new visual concepts from limited examples is a hallmark of human cognition. While traditional category learning models represent each example as an unstructured feature vector, compositional concept learning is thought to depend on (1) structured representations of examples (e.g., directed graphs consisting of objects and their relations) and (2) the identification of shared relational structure across examples through analogical mapping. Here, we introduce Probabilistic Schema Induction (PSI), a prototype model that employs deep learning to perform analogical mapping over structured representations of only a handful of examples, forming a compositional concept called a schema. In doing so, PSI relies on a novel conception of similarity that weighs object-level similarity and relational similarity, as well as a mechanism for amplifying relations relevant to classification, analogous to selective attention parameters in traditional models. We show that PSI produces human-like learning performance and outperforms two controls: a prototype model that uses unstructured feature vectors extracted from a deep learning model, and a variant of PSI with weaker structured representations. Notably, we find that PSI's human-like performance is driven by an adaptive strategy that increases relational similarity over object-level similarity and upweights the contribution of relations that distinguish classes. These findings suggest that structured representations and analogical mapping are critical to modeling rapid human-like learning of compositional visual concepts, and demonstrate how deep learning can be leveraged to create psychological models.
>
---
#### [new 052] Non-Registration Change Detection: A Novel Change Detection Task and Benchmark Dataset
- **分类: cs.CV; eess.IV**

- **简介: 论文提出非配准变化检测任务（NRCD），解决遥感图像未精确配准时传统方法失效的问题。通过定义8种现实场景，构建图像转换方案生成非配准数据集，验证现有先进方法在该任务中性能严重下降，并开源了相关数据集和代码。**

- **链接: [http://arxiv.org/pdf/2505.09939v1](http://arxiv.org/pdf/2505.09939v1)**

> **作者:** Zhe Shan; Lei Zhou; Liu Mao; Shaofan Chen; Chuanqiu Ren; Xia Xie
>
> **备注:** Accepted to IGARSS 2025
>
> **摘要:** In this study, we propose a novel remote sensing change detection task, non-registration change detection, to address the increasing number of emergencies such as natural disasters, anthropogenic accidents, and military strikes. First, in light of the limited discourse on the issue of non-registration change detection, we systematically propose eight scenarios that could arise in the real world and potentially contribute to the occurrence of non-registration problems. Second, we develop distinct image transformation schemes tailored to various scenarios to convert the available registration change detection dataset into a non-registration version. Finally, we demonstrate that non-registration change detection can cause catastrophic damage to the state-of-the-art methods. Our code and dataset are available at https://github.com/ShanZard/NRCD.
>
---
#### [new 053] Consistent Quantity-Quality Control across Scenes for Deployment-Aware Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建优化任务，旨在解决高斯泼溅(3DGS)技术中存储/计算成本与渲染质量难以灵活适配实际部署需求的问题。提出ControlGS方法，通过单次训练实现跨场景可调节的数量-质量平衡控制，在保持高性能的同时支持无级调节，适用于多样化硬件部署场景。**

- **链接: [http://arxiv.org/pdf/2505.10473v1](http://arxiv.org/pdf/2505.10473v1)**

> **作者:** Fengdi Zhang; Hongkun Cao; Ruqi Huang
>
> **摘要:** To reduce storage and computational costs, 3D Gaussian splatting (3DGS) seeks to minimize the number of Gaussians used while preserving high rendering quality, introducing an inherent trade-off between Gaussian quantity and rendering quality. Existing methods strive for better quantity-quality performance, but lack the ability for users to intuitively adjust this trade-off to suit practical needs such as model deployment under diverse hardware and communication constraints. Here, we present ControlGS, a 3DGS optimization method that achieves semantically meaningful and cross-scene consistent quantity-quality control while maintaining strong quantity-quality performance. Through a single training run using a fixed setup and a user-specified hyperparameter reflecting quantity-quality preference, ControlGS can automatically find desirable quantity-quality trade-off points across diverse scenes, from compact objects to large outdoor scenes. It also outperforms baselines by achieving higher rendering quality with fewer Gaussians, and supports a broad adjustment range with stepless control over the trade-off.
>
---
#### [new 054] CheXGenBench: A Unified Benchmark For Fidelity, Privacy and Utility of Synthetic Chest Radiographs
- **分类: cs.CV**

- **简介: 该论文提出CheXGenBench，用于系统评估合成胸部X光片的真实性、隐私性和临床价值，解决现有医学生成模型评测标准不一致、实用性不足的问题。通过标准化数据划分和20多项指标，对11种主流模型进行多维度分析，并发布数据集SynthCheX-75K及开源框架，推动医学AI生成模型的客观评估与应用。**

- **链接: [http://arxiv.org/pdf/2505.10496v1](http://arxiv.org/pdf/2505.10496v1)**

> **作者:** Raman Dutt; Pedro Sanchez; Yongchen Yao; Steven McDonagh; Sotirios A. Tsaftaris; Timothy Hospedales
>
> **摘要:** We introduce CheXGenBench, a rigorous and multifaceted evaluation framework for synthetic chest radiograph generation that simultaneously assesses fidelity, privacy risks, and clinical utility across state-of-the-art text-to-image generative models. Despite rapid advancements in generative AI for real-world imagery, medical domain evaluations have been hindered by methodological inconsistencies, outdated architectural comparisons, and disconnected assessment criteria that rarely address the practical clinical value of synthetic samples. CheXGenBench overcomes these limitations through standardised data partitioning and a unified evaluation protocol comprising over 20 quantitative metrics that systematically analyse generation quality, potential privacy vulnerabilities, and downstream clinical applicability across 11 leading text-to-image architectures. Our results reveal critical inefficiencies in the existing evaluation protocols, particularly in assessing generative fidelity, leading to inconsistent and uninformative comparisons. Our framework establishes a standardised benchmark for the medical AI community, enabling objective and reproducible comparisons while facilitating seamless integration of both existing and future generative models. Additionally, we release a high-quality, synthetic dataset, SynthCheX-75K, comprising 75K radiographs generated by the top-performing model (Sana 0.6B) in our benchmark to support further research in this critical domain. Through CheXGenBench, we establish a new state-of-the-art and release our framework, models, and SynthCheX-75K dataset at https://raman1121.github.io/CheXGenBench/
>
---
#### [new 055] Application of YOLOv8 in monocular downward multiple Car Target detection
- **分类: cs.CV; cs.AI; I.4.8; I.2.10**

- **简介: 该论文属于自动驾驶中的目标检测任务，旨在解决现有方法成本高、易受环境干扰及分辨率低的问题。基于YOLOv8，结合结构重参数化、双向金字塔网络和新检测流程，提升了多尺度及小目标检测精度，实验精度达65%，适用于FSAC竞赛等场景。**

- **链接: [http://arxiv.org/pdf/2505.10016v1](http://arxiv.org/pdf/2505.10016v1)**

> **作者:** Shijie Lyu
>
> **备注:** Accepted by the 5th International Conference on Signal Processing and Machine Learning (CONF-SPML 2025), to appear in Applied and Computational Engineering
>
> **摘要:** Autonomous driving technology is progressively transforming traditional car driving methods, marking a significant milestone in modern transportation. Object detection serves as a cornerstone of autonomous systems, playing a vital role in enhancing driving safety, enabling autonomous functionality, improving traffic efficiency, and facilitating effective emergency responses. However, current technologies such as radar for environmental perception, cameras for road perception, and vehicle sensor networks face notable challenges, including high costs, vulnerability to weather and lighting conditions, and limited resolution.To address these limitations, this paper presents an improved autonomous target detection network based on YOLOv8. By integrating structural reparameterization technology, a bidirectional pyramid structure network model, and a novel detection pipeline into the YOLOv8 framework, the proposed approach achieves highly efficient and precise detection of multi-scale, small, and remote objects. Experimental results demonstrate that the enhanced model can effectively detect both large and small objects with a detection accuracy of 65%, showcasing significant advancements over traditional methods.This improved model holds substantial potential for real-world applications and is well-suited for autonomous driving competitions, such as the Formula Student Autonomous China (FSAC), particularly excelling in scenarios involving single-target and small-object detection.
>
---
#### [new 056] A Unified and Scalable Membership Inference Method for Visual Self-supervised Encoder via Part-aware Capability
- **分类: cs.CV**

- **简介: 该论文研究自监督视觉模型的成员推理攻击，解决黑盒场景下攻击方法统一性问题。提出PartCrop方法，通过裁剪图像局部区域触发模型特征响应差异，实现跨不同训练范式的成员推断，验证其有效性并提出防御策略及可扩展改进版本。**

- **链接: [http://arxiv.org/pdf/2505.10351v1](http://arxiv.org/pdf/2505.10351v1)**

> **作者:** Jie Zhu; Jirong Zha; Ding Li; Leye Wang
>
> **备注:** An extension of our ACM CCS2024 conference paper (arXiv:2404.02462). We show the impacts of scaling from both data and model aspects on membership inference for self-supervised visual encoders
>
> **摘要:** Self-supervised learning shows promise in harnessing extensive unlabeled data, but it also confronts significant privacy concerns, especially in vision. In this paper, we perform membership inference on visual self-supervised models in a more realistic setting: self-supervised training method and details are unknown for an adversary when attacking as he usually faces a black-box system in practice. In this setting, considering that self-supervised model could be trained by completely different self-supervised paradigms, e.g., masked image modeling and contrastive learning, with complex training details, we propose a unified membership inference method called PartCrop. It is motivated by the shared part-aware capability among models and stronger part response on the training data. Specifically, PartCrop crops parts of objects in an image to query responses within the image in representation space. We conduct extensive attacks on self-supervised models with different training protocols and structures using three widely used image datasets. The results verify the effectiveness and generalization of PartCrop. Moreover, to defend against PartCrop, we evaluate two common approaches, i.e., early stop and differential privacy, and propose a tailored method called shrinking crop scale range. The defense experiments indicate that all of them are effective. Finally, besides prototype testing on toy visual encoders and small-scale image datasets, we quantitatively study the impacts of scaling from both data and model aspects in a realistic scenario and propose a scalable PartCrop-v2 by introducing two structural improvements to PartCrop. Our code is at https://github.com/JiePKU/PartCrop.
>
---
#### [new 057] PsOCR: Benchmarking Large Multimodal Models for Optical Character Recognition in Low-resource Pashto Language
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于低资源语言的OCR任务，解决普什图语因连写字体和数据集匮乏导致的识别难题。研究者构建了包含百万合成图像的PsOCR数据集，覆盖多样字体和版式，并评估了11个大模型性能。Gemini表现最佳，Qwen-7B是开源最优，为普什图及类似文字研究奠定基础。**

- **链接: [http://arxiv.org/pdf/2505.10055v1](http://arxiv.org/pdf/2505.10055v1)**

> **作者:** Ijazul Haq; Yingjie Zhang; Irfan Ali Khan
>
> **摘要:** This paper evaluates the performance of Large Multimodal Models (LMMs) on Optical Character Recognition (OCR) in the low-resource Pashto language. Natural Language Processing (NLP) in Pashto faces several challenges due to the cursive nature of its script and a scarcity of structured datasets. To address this, we developed a synthetic Pashto OCR dataset, PsOCR, consisting of one million images annotated with bounding boxes at word, line, and document levels, suitable for training and evaluating models based on different architectures, including Convolutional Neural Networks (CNNs) and Transformers. PsOCR covers variations across 1,000 unique font families, colors, image sizes, and layouts. A benchmark subset of 10K images was selected to evaluate the performance of several LMMs, including seven open-source models: DeepSeek's Janus, InternVL, MiniCPM, Florence, and Qwen (3B and 7B), and four closed-source models: GPT-4o, Gemini, Claude, and Grok. Experimental results demonstrate that Gemini achieves the best performance among all models, whereas among open-source models, Qwen-7B stands out. This work provides an insightful assessment of the capabilities and limitations of current LMMs for OCR tasks in Pashto and establishes a foundation for further research not only in Pashto OCR but also for other similar scripts such as Arabic, Persian, and Urdu. PsOCR is available at https://github.com/zirak-ai/PashtoOCR.
>
---
#### [new 058] DDFP: Data-dependent Frequency Prompt for Source Free Domain Adaptation of Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割的无源域适应（SFDA）任务，解决现有方法在图像风格迁移、伪标签质量和训练效率上的不足。提出预适应机制生成高质量伪标签，设计数据依赖频率提示改善跨域风格转换，并采用分层微调策略提升模型性能，实验证明其在跨模态分割任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09927v1](http://arxiv.org/pdf/2505.09927v1)**

> **作者:** Siqi Yin; Shaolei Liu; Manning Wang
>
> **摘要:** Domain adaptation addresses the challenge of model performance degradation caused by domain gaps. In the typical setup for unsupervised domain adaptation, labeled data from a source domain and unlabeled data from a target domain are used to train a target model. However, access to labeled source domain data, particularly in medical datasets, can be restricted due to privacy policies. As a result, research has increasingly shifted to source-free domain adaptation (SFDA), which requires only a pretrained model from the source domain and unlabeled data from the target domain data for adaptation. Existing SFDA methods often rely on domain-specific image style translation and self-supervision techniques to bridge the domain gap and train the target domain model. However, the quality of domain-specific style-translated images and pseudo-labels produced by these methods still leaves room for improvement. Moreover, training the entire model during adaptation can be inefficient under limited supervision. In this paper, we propose a novel SFDA framework to address these challenges. Specifically, to effectively mitigate the impact of domain gap in the initial training phase, we introduce preadaptation to generate a preadapted model, which serves as an initialization of target model and allows for the generation of high-quality enhanced pseudo-labels without introducing extra parameters. Additionally, we propose a data-dependent frequency prompt to more effectively translate target domain images into a source-like style. To further enhance adaptation, we employ a style-related layer fine-tuning strategy, specifically designed for SFDA, to train the target model using the prompted target domain images and pseudo-labels. Extensive experiments on cross-modality abdominal and cardiac SFDA segmentation tasks demonstrate that our proposed method outperforms existing state-of-the-art methods.
>
---
#### [new 059] Generative diffusion model surrogates for mechanistic agent-based biological models
- **分类: q-bio.QM; cs.CV; cs.ET; cs.PF**

- **简介: 该论文属于生成模型替代计算密集型生物模型的任务，旨在解决细胞Potts模型（CPM）模拟血管生成时计算成本过高的问题。通过训练去噪扩散概率模型（DDPM）作为代理模型，结合图像分类器筛选参数特征，实现了比原生代码快22倍的模拟速度，并能预测2万步后的模型状态。**

- **链接: [http://arxiv.org/pdf/2505.09630v1](http://arxiv.org/pdf/2505.09630v1)**

> **作者:** Tien Comlekoglu; J. Quetzalcóatl Toledo-Marín; Douglas W. DeSimone; Shayn M. Peirce; Geoffrey Fox; James A. Glazier
>
> **摘要:** Mechanistic, multicellular, agent-based models are commonly used to investigate tissue, organ, and organism-scale biology at single-cell resolution. The Cellular-Potts Model (CPM) is a powerful and popular framework for developing and interrogating these models. CPMs become computationally expensive at large space- and time- scales making application and investigation of developed models difficult. Surrogate models may allow for the accelerated evaluation of CPMs of complex biological systems. However, the stochastic nature of these models means each set of parameters may give rise to different model configurations, complicating surrogate model development. In this work, we leverage denoising diffusion probabilistic models to train a generative AI surrogate of a CPM used to investigate \textit{in vitro} vasculogenesis. We describe the use of an image classifier to learn the characteristics that define unique areas of a 2-dimensional parameter space. We then apply this classifier to aid in surrogate model selection and verification. Our CPM model surrogate generates model configurations 20,000 timesteps ahead of a reference configuration and demonstrates approximately a 22x reduction in computational time as compared to native code execution. Our work represents a step towards the implementation of DDPMs to develop digital twins of stochastic biological systems.
>
---
#### [new 060] VRSplat: Fast and Robust Gaussian Splatting for Virtual Reality
- **分类: cs.GR; cs.CV**

- **简介: 该论文属虚拟现实中的新型视图合成任务，旨在解决3D高斯溅射（3DGS）在VR中存在的时序伪影、投影失真及低帧率问题。提出VRSplat框架，整合Mini-Splatting、StopThePop及优化投影方法，设计高效中心凹光栅化器并优化高斯参数，通过用户验证实现72+ FPS且消除视觉干扰，提升VR渲染质量与性能。**

- **链接: [http://arxiv.org/pdf/2505.10144v1](http://arxiv.org/pdf/2505.10144v1)**

> **作者:** Xuechang Tu; Lukas Radl; Michael Steiner; Markus Steinberger; Bernhard Kerbl; Fernando de la Torre
>
> **备注:** I3D'25 (PACMCGIT); Project Page: https://cekavis.site/VRSplat/
>
> **摘要:** 3D Gaussian Splatting (3DGS) has rapidly become a leading technique for novel-view synthesis, providing exceptional performance through efficient software-based GPU rasterization. Its versatility enables real-time applications, including on mobile and lower-powered devices. However, 3DGS faces key challenges in virtual reality (VR): (1) temporal artifacts, such as popping during head movements, (2) projection-based distortions that result in disturbing and view-inconsistent floaters, and (3) reduced framerates when rendering large numbers of Gaussians, falling below the critical threshold for VR. Compared to desktop environments, these issues are drastically amplified by large field-of-view, constant head movements, and high resolution of head-mounted displays (HMDs). In this work, we introduce VRSplat: we combine and extend several recent advancements in 3DGS to address challenges of VR holistically. We show how the ideas of Mini-Splatting, StopThePop, and Optimal Projection can complement each other, by modifying the individual techniques and core 3DGS rasterizer. Additionally, we propose an efficient foveated rasterizer that handles focus and peripheral areas in a single GPU launch, avoiding redundant computations and improving GPU utilization. Our method also incorporates a fine-tuning step that optimizes Gaussian parameters based on StopThePop depth evaluations and Optimal Projection. We validate our method through a controlled user study with 25 participants, showing a strong preference for VRSplat over other configurations of Mini-Splatting. VRSplat is the first, systematically evaluated 3DGS approach capable of supporting modern VR applications, achieving 72+ FPS while eliminating popping and stereo-disrupting floaters.
>
---
#### [new 061] FlowDreamer: A RGB-D World Model with Flow-based Motion Representations for Robot Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究机器人操作的视觉世界模型，旨在提升RGB-D帧的未来预测能力。针对现有方法隐式处理动态预测的问题，提出FlowDreamer：通过U-Net预测显式3D场景流，结合扩散模型生成未来帧。实验表明其语义相似性、像素质量和任务成功率优于基线模型。**

- **链接: [http://arxiv.org/pdf/2505.10075v1](http://arxiv.org/pdf/2505.10075v1)**

> **作者:** Jun Guo; Xiaojian Ma; Yikai Wang; Min Yang; Huaping Liu; Qing Li
>
> **备注:** Project page: see https://sharinka0715.github.io/FlowDreamer/
>
> **摘要:** This paper investigates training better visual world models for robot manipulation, i.e., models that can predict future visual observations by conditioning on past frames and robot actions. Specifically, we consider world models that operate on RGB-D frames (RGB-D world models). As opposed to canonical approaches that handle dynamics prediction mostly implicitly and reconcile it with visual rendering in a single model, we introduce FlowDreamer, which adopts 3D scene flow as explicit motion representations. FlowDreamer first predicts 3D scene flow from past frame and action conditions with a U-Net, and then a diffusion model will predict the future frame utilizing the scene flow. FlowDreamer is trained end-to-end despite its modularized nature. We conduct experiments on 4 different benchmarks, covering both video prediction and visual planning tasks. The results demonstrate that FlowDreamer achieves better performance compared to other baseline RGB-D world models by 7% on semantic similarity, 11% on pixel quality, and 6% on success rate in various robot manipulation domains.
>
---
#### [new 062] Multi-contrast laser endoscopy for in vivo gastrointestinal imaging
- **分类: eess.IV; cs.CV; physics.med-ph; physics.optics**

- **简介: 该论文属于医学影像技术领域，旨在解决传统白光内窥镜在胃肠道疾病检测中组织对比度不足的问题。研究者开发了多对比激光内窥镜（MLE），整合光谱、相干和方向可调照明，通过多模态成像增强组织显影能力，在临床结肠镜检查中验证其对比度提升效果（3倍对比度、5倍色差优化），实现了无创精准检测。**

- **链接: [http://arxiv.org/pdf/2505.10492v1](http://arxiv.org/pdf/2505.10492v1)**

> **作者:** Taylor L. Bobrow; Mayank Golhar; Suchapa Arayakarnkul; Anthony A. Song; Saowanee Ngamruengphong; Nicholas J. Durr
>
> **摘要:** White light endoscopy is the clinical gold standard for detecting diseases in the gastrointestinal tract. Most applications involve identifying visual abnormalities in tissue color, texture, and shape. Unfortunately, the contrast of these features is often subtle, causing many clinically relevant cases to go undetected. To overcome this challenge, we introduce Multi-contrast Laser Endoscopy (MLE): a platform for widefield clinical imaging with rapidly tunable spectral, coherent, and directional illumination. We demonstrate three capabilities of MLE: enhancing tissue chromophore contrast with multispectral diffuse reflectance, quantifying blood flow using laser speckle contrast imaging, and characterizing mucosal topography using photometric stereo. We validate MLE with benchtop models, then demonstrate MLE in vivo during clinical colonoscopies. MLE images from 31 polyps demonstrate an approximate three-fold improvement in contrast and a five-fold improvement in color difference compared to white light and narrow band imaging. With the ability to reveal multiple complementary types of tissue contrast while seamlessly integrating into the clinical environment, MLE shows promise as an investigative tool to improve gastrointestinal imaging.
>
---
#### [new 063] Multi-Token Prediction Needs Registers
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于语言模型优化任务，旨在提升多令牌预测在微调等场景的泛化性。提出MuToR方法：通过插入可学习的寄存器令牌预测多步目标，保持与原模型兼容性，仅少量参数即可支持扩展预测范围，并在语言/视觉任务的监督微调、PEFT及预训练中验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.10518v1](http://arxiv.org/pdf/2505.10518v1)**

> **作者:** Anastasios Gerontopoulos; Spyros Gidaris; Nikos Komodakis
>
> **摘要:** Multi-token prediction has emerged as a promising objective for improving language model pretraining, but its benefits have not consistently generalized to other settings such as fine-tuning. In this paper, we propose MuToR, a simple and effective approach to multi-token prediction that interleaves learnable register tokens into the input sequence, each tasked with predicting future targets. Compared to existing methods, MuToR offers several key advantages: it introduces only a negligible number of additional parameters, requires no architectural changes--ensuring compatibility with off-the-shelf pretrained language models--and remains aligned with the next-token pretraining objective, making it especially well-suited for supervised fine-tuning. Moreover, it naturally supports scalable prediction horizons. We demonstrate the effectiveness and versatility of MuToR across a range of use cases, including supervised fine-tuning, parameter-efficient fine-tuning (PEFT), and pretraining, on challenging generative tasks in both language and vision domains. Our code will be available at: https://github.com/nasosger/MuToR.
>
---
#### [new 064] SEAL: Searching Expandable Architectures for Incremental Learning
- **分类: cs.LG; cs.AI; cs.CV; 68T07**

- **简介: 该论文针对数据增量学习任务，解决模型在持续学习中平衡新任务适应与旧知识保留的问题。提出SEAL框架，通过动态架构搜索实现必要扩展与交叉蒸馏训练，在降低资源消耗的同时减少遗忘，较现有方法实现更高精度与更小模型规模。**

- **链接: [http://arxiv.org/pdf/2505.10457v1](http://arxiv.org/pdf/2505.10457v1)**

> **作者:** Matteo Gambella; Vicente Javier Castro Solar; Manuel Roveri
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Incremental learning is a machine learning paradigm where a model learns from a sequential stream of tasks. This setting poses a key challenge: balancing plasticity (learning new tasks) and stability (preserving past knowledge). Neural Architecture Search (NAS), a branch of AutoML, automates the design of the architecture of Deep Neural Networks and has shown success in static settings. However, existing NAS-based approaches to incremental learning often rely on expanding the model at every task, making them impractical in resource-constrained environments. In this work, we introduce SEAL, a NAS-based framework tailored for data-incremental learning, a scenario where disjoint data samples arrive sequentially and are not stored for future access. SEAL adapts the model structure dynamically by expanding it only when necessary, based on a capacity estimation metric. Stability is preserved through cross-distillation training after each expansion step. The NAS component jointly searches for both the architecture and the optimal expansion policy. Experiments across multiple benchmarks demonstrate that SEAL effectively reduces forgetting and enhances accuracy while maintaining a lower model size compared to prior methods. These results highlight the promise of combining NAS and selective expansion for efficient, adaptive learning in incremental scenarios.
>
---
#### [new 065] HWA-UNETR: Hierarchical Window Aggregate UNETR for 3D Multimodal Gastric Lesion Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对胃癌多模态医学图像分割任务，解决数据稀缺和模态未对齐问题。提出首个开源胃癌多模态MRI数据集GCM 2025，并开发HWA-UNETR模型，通过可学习窗口聚合层和三向融合机制实现跨模态特征对齐及长程依赖建模，Dice分数提升1.68%。**

- **链接: [http://arxiv.org/pdf/2505.10464v1](http://arxiv.org/pdf/2505.10464v1)**

> **作者:** Jiaming Liang; Lihuan Dai; Xiaoqi Sheng; Xiangguang Chen; Chun Yao; Guihua Tao; Qibin Leng; Honming Cai; Xi Zhong
>
> **备注:** This work has been provisionally accepted for MICCAI 2025
>
> **摘要:** Multimodal medical image segmentation faces significant challenges in the context of gastric cancer lesion analysis. This clinical context is defined by the scarcity of independent multimodal datasets and the imperative to amalgamate inherently misaligned modalities. As a result, algorithms are constrained to train on approximate data and depend on application migration, leading to substantial resource expenditure and a potential decline in analysis accuracy. To address those challenges, we have made two major contributions: First, we publicly disseminate the GCM 2025 dataset, which serves as the first large-scale, open-source collection of gastric cancer multimodal MRI scans, featuring professionally annotated FS-T2W, CE-T1W, and ADC images from 500 patients. Second, we introduce HWA-UNETR, a novel 3D segmentation framework that employs an original HWA block with learnable window aggregation layers to establish dynamic feature correspondences between different modalities' anatomical structures, and leverages the innovative tri-orientated fusion mamba mechanism for context modeling and capturing long-range spatial dependencies. Extensive experiments on our GCM 2025 dataset and the publicly BraTS 2021 dataset validate the performance of our framework, demonstrating that the new approach surpasses existing methods by up to 1.68\% in the Dice score while maintaining solid robustness. The dataset and code are public via https://github.com/JeMing-creater/HWA-UNETR.
>
---
#### [new 066] SOS: A Shuffle Order Strategy for Data Augmentation in Industrial Human Activity Recognition
- **分类: cs.HC; cs.CV**

- **简介: 该论文针对工业人类活动识别（HAR）任务中数据质量低、异构性强的问题，提出基于随机序列重排的数据增强策略。通过注意力自编码器和条件GAN生成数据，并采用打乱时序依赖性的方法强制模型关注瞬时特征，提升分类效果（准确率0.70±0.03），增强对复杂场景的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.10312v1](http://arxiv.org/pdf/2505.10312v1)**

> **作者:** Anh Tuan Ha; Hoang Khang Phan; Thai Minh Tien Ngo; Anh Phan Truong; Nhat Tan Le
>
> **摘要:** In the realm of Human Activity Recognition (HAR), obtaining high quality and variance data is still a persistent challenge due to high costs and the inherent variability of real-world activities. This study introduces a generation dataset by deep learning approaches (Attention Autoencoder and conditional Generative Adversarial Networks). Another problem that data heterogeneity is a critical challenge, one of the solutions is to shuffle the data to homogenize the distribution. Experimental results demonstrate that the random sequence strategy significantly improves classification performance, achieving an accuracy of up to 0.70 $\pm$ 0.03 and a macro F1 score of 0.64 $\pm$ 0.01. For that, disrupting temporal dependencies through random sequence reordering compels the model to focus on instantaneous recognition, thereby improving robustness against activity transitions. This approach not only broadens the effective training dataset but also offers promising avenues for enhancing HAR systems in complex, real-world scenarios.
>
---
#### [new 067] Visual Feedback of Pattern Separability Improves Myoelectric Decoding Performance of Upper Limb Prostheses
- **分类: cs.HC; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于人机交互与肌电假肢控制优化任务，旨在解决上肢假肢用户因肌电信号模式混杂导致的分类可靠性问题。研究者开发了"Reviewer"系统，通过3D可视化界面实时投影肌电信号至分类空间，提供直观反馈以优化用户动作模式与解码器边界的协同适应。实验表明，该方法相比传统训练显著提升了任务完成率、路径效率和操作精度，降低了试错调整依赖。**

- **链接: [http://arxiv.org/pdf/2505.09819v1](http://arxiv.org/pdf/2505.09819v1)**

> **作者:** Ruichen Yang; György M. Lévay; Christopher L. Hunt; Dániel Czeiner; Megan C. Hodgson; Damini Agarwal; Rahul R. Kaliki; Nitish V. Thakor
>
> **摘要:** State-of-the-art upper limb myoelectric prostheses often use pattern recognition (PR) control systems that translate electromyography (EMG) signals into desired movements. As prosthesis movement complexity increases, users often struggle to produce sufficiently distinct EMG patterns for reliable classification. Existing training typically involves heuristic, trial-and-error user adjustments to static decoder boundaries. Goal: We introduce the Reviewer, a 3D visual interface projecting EMG signals directly into the decoder's classification space, providing intuitive, real-time insight into PR algorithm behavior. This structured feedback reduces cognitive load and fosters mutual, data-driven adaptation between user-generated EMG patterns and decoder boundaries. Methods: A 10-session study with 12 able-bodied participants compared PR performance after motor-based training and updating using the Reviewer versus conventional virtual arm visualization. Performance was assessed using a Fitts law task that involved the aperture of the cursor and the control of orientation. Results: Participants trained with the Reviewer achieved higher completion rates, reduced overshoot, and improved path efficiency and throughput compared to the standard visualization group. Significance: The Reviewer introduces decoder-informed motor training, facilitating immediate and consistent PR-based myoelectric control improvements. By iteratively refining control through real-time feedback, this approach reduces reliance on trial-and-error recalibration, enabling a more adaptive, self-correcting training framework. Conclusion: The 3D visual feedback significantly improves PR control in novice operators through structured training, enabling feedback-driven adaptation and reducing reliance on extensive heuristic adjustments.
>
---
#### [new 068] RainPro-8: An Efficient Deep Learning Model to Estimate Rainfall Probabilities Over 8 Hours
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出RainPro-8深度学习模型，专注于欧洲地区8小时高分辨率降水概率预测任务。针对现有雷达模型预测时效短、数据源单一的问题，融合雷达、卫星及数值气象数据，通过紧凑架构捕捉长程关联并生成概率图，实现高效精准的降雨预报，性能超越传统气象模型和现有深度学习方法。**

- **链接: [http://arxiv.org/pdf/2505.10271v1](http://arxiv.org/pdf/2505.10271v1)**

> **作者:** Rafael Pablos Sarabia; Joachim Nyborg; Morten Birk; Jeppe Liborius Sjørup; Anders Lillevang Vesterholt; Ira Assent
>
> **摘要:** We present a deep learning model for high-resolution probabilistic precipitation forecasting over an 8-hour horizon in Europe, overcoming the limitations of radar-only deep learning models with short forecast lead times. Our model efficiently integrates multiple data sources - including radar, satellite, and physics-based numerical weather prediction (NWP) - while capturing long-range interactions, resulting in accurate forecasts with robust uncertainty quantification through consistent probabilistic maps. Featuring a compact architecture, it enables more efficient training and faster inference than existing models. Extensive experiments demonstrate that our model surpasses current operational NWP systems, extrapolation-based methods, and deep-learning nowcasting models, setting a new standard for high-resolution precipitation forecasting in Europe, ensuring a balance between accuracy, interpretability, and computational efficiency.
>
---
#### [new 069] Style Customization of Text-to-Vector Generation with Image Diffusion Priors
- **分类: cs.GR; cs.CV**

- **简介: 该论文研究文本到矢量图生成中的风格定制任务，解决现有方法难以平衡结构规则性与风格一致性的问题。提出两阶段流程：先训练路径级扩散模型确保结构，再蒸馏图像扩散先验实现高效风格适配，生成高质量定制风格SVG。**

- **链接: [http://arxiv.org/pdf/2505.10558v1](http://arxiv.org/pdf/2505.10558v1)**

> **作者:** Peiying Zhang; Nanxuan Zhao; Jing Liao
>
> **备注:** Accepted by SIGGRAPH 2025 (Conference Paper). Project page: https://customsvg.github.io
>
> **摘要:** Scalable Vector Graphics (SVGs) are highly favored by designers due to their resolution independence and well-organized layer structure. Although existing text-to-vector (T2V) generation methods can create SVGs from text prompts, they often overlook an important need in practical applications: style customization, which is vital for producing a collection of vector graphics with consistent visual appearance and coherent aesthetics. Extending existing T2V methods for style customization poses certain challenges. Optimization-based T2V models can utilize the priors of text-to-image (T2I) models for customization, but struggle with maintaining structural regularity. On the other hand, feed-forward T2V models can ensure structural regularity, yet they encounter difficulties in disentangling content and style due to limited SVG training data. To address these challenges, we propose a novel two-stage style customization pipeline for SVG generation, making use of the advantages of both feed-forward T2V models and T2I image priors. In the first stage, we train a T2V diffusion model with a path-level representation to ensure the structural regularity of SVGs while preserving diverse expressive capabilities. In the second stage, we customize the T2V diffusion model to different styles by distilling customized T2I models. By integrating these techniques, our pipeline can generate high-quality and diverse SVGs in custom styles based on text prompts in an efficient feed-forward manner. The effectiveness of our method has been validated through extensive experiments. The project page is https://customsvg.github.io.
>
---
#### [new 070] Visual Fidelity Index for Generative Semantic Communications with Critical Information Embedding
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对生成式语义通信（Gen-SemCom）中视觉细节丢失及缺乏评估指标的问题，提出混合传输框架：通过语义过滤提取文本提示与关键特征，结合扩散模型重建高保真图像；设计GVIF指标量化视觉信息保真度，并开发自适应信道优化系统。属于6G通信中图像生成与传输优化任务，实验验证了系统性能优于基准方案。**

- **链接: [http://arxiv.org/pdf/2505.10405v1](http://arxiv.org/pdf/2505.10405v1)**

> **作者:** Jianhao Huang; Qunsong Zeng; Kaibin Huang
>
> **摘要:** Generative semantic communication (Gen-SemCom) with large artificial intelligence (AI) model promises a transformative paradigm for 6G networks, which reduces communication costs by transmitting low-dimensional prompts rather than raw data. However, purely prompt-driven generation loses fine-grained visual details. Additionally, there is a lack of systematic metrics to evaluate the performance of Gen-SemCom systems. To address these issues, we develop a hybrid Gen-SemCom system with a critical information embedding (CIE) framework, where both text prompts and semantically critical features are extracted for transmissions. First, a novel approach of semantic filtering is proposed to select and transmit the semantically critical features of images relevant to semantic label. By integrating the text prompt and critical features, the receiver reconstructs high-fidelity images using a diffusion-based generative model. Next, we propose the generative visual information fidelity (GVIF) metric to evaluate the visual quality of the generated image. By characterizing the statistical models of image features, the GVIF metric quantifies the mutual information between the distorted features and their original counterparts. By maximizing the GVIF metric, we design a channel-adaptive Gen-SemCom system that adaptively control the volume of features and compression rate according to the channel state. Experimental results validate the GVIF metric's sensitivity to visual fidelity, correlating with both the PSNR and critical information volume. In addition, the optimized system achieves superior performance over benchmarking schemes in terms of higher PSNR and lower FID scores.
>
---
#### [new 071] PIF: Anomaly detection via preference embedding
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文属于异常检测任务，旨在解决结构化模式中的异常识别问题。提出PIF方法，结合自适应隔离与偏好嵌入技术，通过高维空间嵌入数据并利用PI-Forest树模型计算异常分数。实验验证其优于现有方法，尤其在测量复杂距离与隔离异常点方面表现更优。**

- **链接: [http://arxiv.org/pdf/2505.10441v1](http://arxiv.org/pdf/2505.10441v1)**

> **作者:** Filippo Leveni; Luca Magri; Giacomo Boracchi; Cesare Alippi
>
> **备注:** Accepted at International Conference on Pattern Recognition (ICPR 2020)
>
> **摘要:** We address the problem of detecting anomalies with respect to structured patterns. To this end, we conceive a novel anomaly detection method called PIF, that combines the advantages of adaptive isolation methods with the flexibility of preference embedding. Specifically, we propose to embed the data in a high dimensional space where an efficient tree-based method, PI-Forest, is employed to compute an anomaly score. Experiments on synthetic and real datasets demonstrate that PIF favorably compares with state-of-the-art anomaly detection techniques, and confirm that PI-Forest is better at measuring arbitrary distances and isolate points in the preference space.
>
---
#### [new 072] MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文针对视觉语言模型（VLM）推测解码加速任务，解决小语言模型无法处理视觉输入且预测不匹配的问题。提出MASSV方法，通过轻量级投影器连接视觉编码器，并利用目标模型自蒸馏对齐预测，提升推理速度1.46倍，兼容现有VLM架构。**

- **链接: [http://arxiv.org/pdf/2505.10526v1](http://arxiv.org/pdf/2505.10526v1)**

> **作者:** Mugilan Ganesan; Shane Segal; Ankur Aggarwal; Nish Sinnadurai; Sean Lie; Vithursan Thangarasa
>
> **备注:** Main paper: 11 pp., 4 figs., 3 tabs.; Supplementary: 2 pp
>
> **摘要:** Speculative decoding significantly accelerates language model inference by enabling a lightweight draft model to propose multiple tokens that a larger target model verifies simultaneously. However, applying this technique to vision-language models (VLMs) presents two fundamental challenges: small language models that could serve as efficient drafters lack the architectural components to process visual inputs, and their token predictions fail to match those of VLM target models that consider visual context. We introduce Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models (MASSV), which transforms existing small language models into effective multimodal drafters through a two-phase approach. MASSV first connects the target VLM's vision encoder to the draft model via a lightweight trainable projector, then applies self-distilled visual instruction tuning using responses generated by the target VLM to align token predictions. Comprehensive experiments across the Qwen2.5-VL and Gemma3 model families demonstrate that MASSV increases accepted length by up to 30% and delivers end-to-end inference speedups of up to 1.46x on visually-grounded tasks. MASSV provides a scalable, architecture-compatible method for accelerating both current and future VLMs.
>
---
#### [new 073] ImplicitStainer: Data-Efficient Medical Image Translation for Virtual Antibody-based Tissue Staining Using Local Implicit Functions
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像转换任务，旨在解决传统虚拟染色方法（如GAN/扩散模型）因独立处理图像块导致数据需求高的问题。提出ImplicitStainer，利用局部隐函数实现像素级预测，在数据有限时生成高质量免疫组化染色图像，降低对数据量和多样性的依赖，并通过实验验证其优于现有15种模型。**

- **链接: [http://arxiv.org/pdf/2505.09831v1](http://arxiv.org/pdf/2505.09831v1)**

> **作者:** Tushar Kataria; Beatrice Knudsen; Shireen Y. Elhabian
>
> **摘要:** Hematoxylin and eosin (H&E) staining is a gold standard for microscopic diagnosis in pathology. However, H&E staining does not capture all the diagnostic information that may be needed. To obtain additional molecular information, immunohistochemical (IHC) stains highlight proteins that mark specific cell types, such as CD3 for T-cells or CK8/18 for epithelial cells. While IHC stains are vital for prognosis and treatment guidance, they are typically only available at specialized centers and time consuming to acquire, leading to treatment delays for patients. Virtual staining, enabled by deep learning-based image translation models, provides a promising alternative by computationally generating IHC stains from H&E stained images. Although many GAN and diffusion based image to image (I2I) translation methods have been used for virtual staining, these models treat image patches as independent data points, which results in increased and more diverse data requirements for effective generation. We present ImplicitStainer, a novel approach that leverages local implicit functions to improve image translation, specifically virtual staining performance, by focusing on pixel-level predictions. This method enhances robustness to variations in dataset sizes, delivering high-quality results even with limited data. We validate our approach on two datasets using a comprehensive set of metrics and benchmark it against over fifteen state-of-the-art GAN- and diffusion based models. Full Code and models trained will be released publicly via Github upon acceptance.
>
---
#### [new 074] Ordered-subsets Multi-diffusion Model for Sparse-view CT Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于稀疏视图CT重建任务，旨在解决传统扩散模型因数据冗余导致的细节缺失和噪声问题。提出有序子集多扩散模型（OSMM），将投影数据分块独立学习，结合全局约束提升细节重建，并通过无监督框架增强鲁棒性，实验显示其图像质量和抗噪性更优。**

- **链接: [http://arxiv.org/pdf/2505.09985v1](http://arxiv.org/pdf/2505.09985v1)**

> **作者:** Pengfei Yu; Bin Huang; Minghui Zhang; Weiwen Wu; Shaoyu Wang; Qiegen Liu
>
> **摘要:** Score-based diffusion models have shown significant promise in the field of sparse-view CT reconstruction. However, the projection dataset is large and riddled with redundancy. Consequently, applying the diffusion model to unprocessed data results in lower learning effectiveness and higher learning difficulty, frequently leading to reconstructed images that lack fine details. To address these issues, we propose the ordered-subsets multi-diffusion model (OSMM) for sparse-view CT reconstruction. The OSMM innovatively divides the CT projection data into equal subsets and employs multi-subsets diffusion model (MSDM) to learn from each subset independently. This targeted learning approach reduces complexity and enhances the reconstruction of fine details. Furthermore, the integration of one-whole diffusion model (OWDM) with complete sinogram data acts as a global information constraint, which can reduce the possibility of generating erroneous or inconsistent sinogram information. Moreover, the OSMM's unsupervised learning framework provides strong robustness and generalizability, adapting seamlessly to varying sparsity levels of CT sinograms. This ensures consistent and reliable performance across different clinical scenarios. Experimental results demonstrate that OSMM outperforms traditional diffusion models in terms of image quality and noise resilience, offering a powerful and versatile solution for advanced CT imaging in sparse-view scenarios.
>
---
## 更新

#### [replaced 001] Highly Efficient 3D Human Pose Tracking from Events with Spiking Spatiotemporal Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2303.09681v5](http://arxiv.org/pdf/2303.09681v5)**

> **作者:** Shihao Zou; Yuxuan Mu; Wei Ji; Zi-An Wang; Xinxin Zuo; Sen Wang; Weixin Si; Li Cheng
>
> **备注:** Accepted by IEEE TCSVT
>
> **摘要:** Event camera, as an asynchronous vision sensor capturing scene dynamics, presents new opportunities for highly efficient 3D human pose tracking. Existing approaches typically adopt modern-day Artificial Neural Networks (ANNs), such as CNNs or Transformer, where sparse events are converted into dense images or paired with additional gray-scale images as input. Such practices, however, ignore the inherent sparsity of events, resulting in redundant computations, increased energy consumption, and potentially degraded performance. Motivated by these observations, we introduce the first sparse Spiking Neural Networks (SNNs) framework for 3D human pose tracking based solely on events. Our approach eliminates the need to convert sparse data to dense formats or incorporate additional images, thereby fully exploiting the innate sparsity of input events. Central to our framework is a novel Spiking Spatiotemporal Transformer, which enables bi-directional spatiotemporal fusion of spike pose features and provides a guaranteed similarity measurement between binary spike features in spiking attention. Moreover, we have constructed a large-scale synthetic dataset, SynEventHPD, that features a broad and diverse set of 3D human motions, as well as much longer hours of event streams. Empirical experiments demonstrate the superiority of our approach over existing state-of-the-art (SOTA) ANN-based methods, requiring only 19.1% FLOPs and 3.6% energy cost. Furthermore, our approach outperforms existing SNN-based benchmarks in this task, highlighting the effectiveness of our proposed SNN framework. The dataset will be released upon acceptance, and code can be found at https://github.com/JimmyZou/HumanPoseTracking_SNN.
>
---
#### [replaced 002] Latent Action Pretraining from Videos
- **分类: cs.RO; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.11758v2](http://arxiv.org/pdf/2410.11758v2)**

> **作者:** Seonghyeon Ye; Joel Jang; Byeongguk Jeon; Sejune Joo; Jianwei Yang; Baolin Peng; Ajay Mandlekar; Reuben Tan; Yu-Wei Chao; Bill Yuchen Lin; Lars Liden; Kimin Lee; Jianfeng Gao; Luke Zettlemoyer; Dieter Fox; Minjoon Seo
>
> **备注:** ICLR 2025 Website: https://latentactionpretraining.github.io
>
> **摘要:** We introduce Latent Action Pretraining for general Action models (LAPA), an unsupervised method for pretraining Vision-Language-Action (VLA) models without ground-truth robot action labels. Existing Vision-Language-Action models require action labels typically collected by human teleoperators during pretraining, which significantly limits possible data sources and scale. In this work, we propose a method to learn from internet-scale videos that do not have robot action labels. We first train an action quantization model leveraging VQ-VAE-based objective to learn discrete latent actions between image frames, then pretrain a latent VLA model to predict these latent actions from observations and task descriptions, and finally finetune the VLA on small-scale robot manipulation data to map from latent to robot actions. Experimental results demonstrate that our method significantly outperforms existing techniques that train robot manipulation policies from large-scale videos. Furthermore, it outperforms the state-of-the-art VLA model trained with robotic action labels on real-world manipulation tasks that require language conditioning, generalization to unseen objects, and semantic generalization to unseen instructions. Training only on human manipulation videos also shows positive transfer, opening up the potential for leveraging web-scale data for robotics foundation model.
>
---
#### [replaced 003] Exploring Convolutional Neural Networks for Rice Grain Classification: An Explainable AI Approach
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05513v2](http://arxiv.org/pdf/2505.05513v2)**

> **作者:** Muhammad Junaid Asif; Hamza Khan; Rabia Tehseen; Syed Tahir Hussain Rizvi; Mujtaba Asad; Shazia Saqib; Rana Fayyaz Ahmad
>
> **摘要:** Rice is an essential staple food worldwide that is important in promoting international trade, economic growth, and nutrition. Asian countries such as China, India, Pakistan, Thailand, Vietnam, and Indonesia are notable for their significant contribution to the cultivation and utilization of rice. These nations are also known for cultivating different rice grains, including short and long grains. These sizes are further classified as basmati, jasmine, kainat saila, ipsala, arborio, etc., catering to diverse culinary preferences and cultural traditions. For both local and international trade, inspecting and maintaining the quality of rice grains to satisfy customers and preserve a country's reputation is necessary. Manual quality check and classification is quite a laborious and time-consuming process. It is also highly prone to mistakes. Therefore, an automatic solution must be proposed for the effective and efficient classification of different varieties of rice grains. This research paper presents an automatic framework based on a convolutional neural network (CNN) for classifying different varieties of rice grains. We evaluated the proposed model based on performance metrics such as accuracy, recall, precision, and F1-Score. The CNN model underwent rigorous training and validation, achieving a remarkable accuracy rate and a perfect area under each class's Receiver Operating Characteristic (ROC) curve. The confusion matrix analysis confirmed the model's effectiveness in distinguishing between the different rice varieties, indicating minimal misclassifications. Additionally, the integration of explainability techniques such as LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) provided valuable insights into the model's decision-making process, revealing how specific features of the rice grains influenced classification outcomes.
>
---
#### [replaced 004] Cyclic 2.5D Perceptual Loss for Cross-Modal 3D Medical Image Synthesis: T1w MRI to Tau PET
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.12632v2](http://arxiv.org/pdf/2406.12632v2)**

> **作者:** Junho Moon; Symac Kim; Haejun Chung; Ikbeom Jang
>
> **摘要:** There is a demand for medical image synthesis or translation to generate synthetic images of missing modalities from available data. This need stems from challenges such as restricted access to high-cost imaging devices, government regulations, or failure to follow up with patients or study participants. In medical imaging, preserving high-level semantic features is often more critical than achieving pixel-level accuracy. Perceptual loss functions are widely employed to train medical image synthesis or translation models, as they quantify differences in high-level image features using a pre-trained feature extraction network. While 3D and 2.5D perceptual losses are used in 3D medical image synthesis, they face challenges, such as the lack of pre-trained 3D models or difficulties in balancing loss reduction across different planes. In this work, we focus on synthesizing 3D tau PET images from 3D T1-weighted MR images. We propose a cyclic 2.5D perceptual loss that sequentially computes the 2D average perceptual loss for each of the axial, coronal, and sagittal planes over epochs, with the cycle duration gradually decreasing. Additionally, we process tau PET images using by-manufacturer standardization to enhance the preservation of high-SUVR regions indicative of tau pathology and mitigate SUVR variability caused by inter-manufacturer differences. We combine the proposed loss with SSIM and MSE losses and demonstrate its effectiveness in improving both quantitative and qualitative performance across various generative models, including U-Net, UNETR, SwinUNETR, CycleGAN, and Pix2Pix.
>
---
#### [replaced 005] Hierarchical World Models as Visual Whole-Body Humanoid Controllers
- **分类: cs.LG; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.18418v3](http://arxiv.org/pdf/2405.18418v3)**

> **作者:** Nicklas Hansen; Jyothir S V; Vlad Sobal; Yann LeCun; Xiaolong Wang; Hao Su
>
> **备注:** Code and videos at https://nicklashansen.com/rlpuppeteer
>
> **摘要:** Whole-body control for humanoids is challenging due to the high-dimensional nature of the problem, coupled with the inherent instability of a bipedal morphology. Learning from visual observations further exacerbates this difficulty. In this work, we explore highly data-driven approaches to visual whole-body humanoid control based on reinforcement learning, without any simplifying assumptions, reward design, or skill primitives. Specifically, we propose a hierarchical world model in which a high-level agent generates commands based on visual observations for a low-level agent to execute, both of which are trained with rewards. Our approach produces highly performant control policies in 8 tasks with a simulated 56-DoF humanoid, while synthesizing motions that are broadly preferred by humans.
>
---
#### [replaced 006] Illegal Waste Detection in Remote Sensing Images: A Case Study
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.06607v3](http://arxiv.org/pdf/2502.06607v3)**

> **作者:** Federico Gibellini; Piero Fraternali; Giacomo Boracchi; Luca Morandini; Thomas Martinoli; Andrea Diecidue; Simona Malegori
>
> **摘要:** Environmental crime is the third largest criminal activity worldwide, with significant revenues coming from illegal management of solid waste. Thanks to the increasing availability and the decreasing cost of Very High Resolution Remote Sensing (VHR RS) images, the fight against environmental crime can nowadays rely on modern image-analysis tools to support photo-interpretation for scanning vast territories in search of illegal waste disposal sites. This paper illustrates a semi-automatic waste detection pipeline, developed in collaboration with a regional environmental protection agency, for detecting candidate illegal dumping sites in VHR RS images. To optimize the effectiveness of the waste detector, extensive experiments evaluate such design choices as the network architecture, the ground resolution and geographic span of the input images, as well as the pretraining procedures. The best model attains remarkable performance, achieving 92.02% F1-Score and 94.56% Accuracy. A generalization study assesses the performance variation when the detector processes images from a territory substantially different from the one used during training, incurring only a moderate performance loss, i.e., 6.5% decrease in the F1-Score. Finally, an exercise in which photo interpreters compare the territory scanning effort with and without the support of the waste detector assesses the concrete benefit of using a computer-aided image analysis tool in a professional environment protection agency. Results show that a reduction up to 30% of the time spent for waste site detection can be attained.
>
---
#### [replaced 007] EndoMamba: An Efficient Foundation Model for Endoscopic Videos via Hierarchical Pre-training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.19090v2](http://arxiv.org/pdf/2502.19090v2)**

> **作者:** Qingyao Tian; Huai Liao; Xinyan Huang; Bingyu Yang; Dongdong Lei; Sebastien Ourselin; Hongbin Liu
>
> **摘要:** Endoscopic video-based tasks, such as visual navigation and surgical phase recognition, play a crucial role in minimally invasive surgeries by providing real-time assistance. While recent video foundation models have shown promise, their applications are hindered by (1) computational inefficiencies and (2) suboptimal performance caused by limited data for pre-training in endoscopy. To address these issues, we present EndoMamba, a foundation model designed for real-time inference while learning generalized spatiotemporal representations. First, to mitigate computational inefficiencies, we propose the EndoMamba backbone, optimized for real-time inference. Inspired by recent advancements in state space models, EndoMamba integrates Bidirectional Mamba blocks for spatial modeling within individual frames and vanilla Mamba blocks for past-to-present reasoning across the temporal domain. This design enables both strong spatiotemporal modeling and efficient inference in online video streams. Second, we propose a self-supervised hierarchical pre-training diagram to enhance EndoMamba's representation learning using endoscopic videos and incorporating general video domain knowledge. Specifically, our approach combines masked reconstruction with auxiliary supervision, leveraging low-level reconstruction to capture spatial-temporal structures and high-level alignment to transfer broader knowledge from a pretrained general-video domain foundation model. Extensive experiments on four downstream tasks--classification, segmentation, surgical phase recognition, and localization--demonstrate that EndoMamba outperforms existing foundation models and task-specific methods while maintaining real-time inference speed. The source code is available at https://github.com/TianCuteQY/EndoMamba.
>
---
#### [replaced 008] Efficient Quantum Convolutional Neural Networks for Image Classification: Overcoming Hardware Constraints
- **分类: quant-ph; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05957v2](http://arxiv.org/pdf/2505.05957v2)**

> **作者:** Peter Röseler; Oliver Schaudt; Helmut Berg; Christian Bauckhage; Matthias Koch
>
> **摘要:** While classical convolutional neural networks (CNNs) have revolutionized image classification, the emergence of quantum computing presents new opportunities for enhancing neural network architectures. Quantum CNNs (QCNNs) leverage quantum mechanical properties and hold potential to outperform classical approaches. However, their implementation on current noisy intermediate-scale quantum (NISQ) devices remains challenging due to hardware limitations. In our research, we address this challenge by introducing an encoding scheme that significantly reduces the input dimensionality. We demonstrate that a primitive QCNN architecture with 49 qubits is sufficient to directly process $28\times 28$ pixel MNIST images, eliminating the need for classical dimensionality reduction pre-processing. Additionally, we propose an automated framework based on expressibility, entanglement, and complexity characteristics to identify the building blocks of QCNNs, parameterized quantum circuits (PQCs). Our approach demonstrates advantages in accuracy and convergence speed with a similar parameter count compared to both hybrid QCNNs and classical CNNs. We validated our experiments on IBM's Heron r2 quantum processor, achieving $96.08\%$ classification accuracy, surpassing the $71.74\%$ benchmark of traditional approaches under identical training conditions. These results represent one of the first implementations of image classifications on real quantum hardware and validate the potential of quantum computing in this area.
>
---
#### [replaced 009] Estimating the Diameter at Breast Height of Trees in a Forest With a Single 360 Camera
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03093v2](http://arxiv.org/pdf/2505.03093v2)**

> **作者:** Siming He; Zachary Osman; Fernando Cladera; Dexter Ong; Nitant Rai; Patrick Corey Green; Vijay Kumar; Pratik Chaudhari
>
> **摘要:** Forest inventories rely on accurate measurements of the diameter at breast height (DBH) for ecological monitoring, resource management, and carbon accounting. While LiDAR-based techniques can achieve centimeter-level precision, they are cost-prohibitive and operationally complex. We present a low-cost alternative that only needs a consumer-grade 360 video camera. Our semi-automated pipeline comprises of (i) a dense point cloud reconstruction using Structure from Motion (SfM) photogrammetry software called Agisoft Metashape, (ii) semantic trunk segmentation by projecting Grounded Segment Anything (SAM) masks onto the 3D cloud, and (iii) a robust RANSAC-based technique to estimate cross section shape and DBH. We introduce an interactive visualization tool for inspecting segmented trees and their estimated DBH. On 61 acquisitions of 43 trees under a variety of conditions, our method attains median absolute relative errors of 5-9% with respect to "ground-truth" manual measurements. This is only 2-4% higher than LiDAR-based estimates, while employing a single 360 camera that costs orders of magnitude less, requires minimal setup, and is widely available.
>
---
#### [replaced 010] Unsupervised Video Highlight Detection by Learning from Audio and Visual Recurrence
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.13933v2](http://arxiv.org/pdf/2407.13933v2)**

> **作者:** Zahidul Islam; Sujoy Paul; Mrigank Rochan
>
> **备注:** Accepted to the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
>
> **摘要:** With the exponential growth of video content, the need for automated video highlight detection to extract key moments or highlights from lengthy videos has become increasingly pressing. This technology has the potential to enhance user experiences by allowing quick access to relevant content across diverse domains. Existing methods typically rely either on expensive manually labeled frame-level annotations, or on a large external dataset of videos for weak supervision through category information. To overcome this, we focus on unsupervised video highlight detection, eliminating the need for manual annotations. We propose a novel unsupervised approach which capitalizes on the premise that significant moments tend to recur across multiple videos of the similar category in both audio and visual modalities. Surprisingly, audio remains under-explored, especially in unsupervised algorithms, despite its potential to detect key moments. Through a clustering technique, we identify pseudo-categories of videos and compute audio pseudo-highlight scores for each video by measuring the similarities of audio features among audio clips of all the videos within each pseudo-category. Similarly, we also compute visual pseudo-highlight scores for each video using visual features. Then, we combine audio and visual pseudo-highlights to create the audio-visual pseudo ground-truth highlight of each video for training an audio-visual highlight detection network. Extensive experiments and ablation studies on three benchmarks showcase the superior performance of our method over prior work.
>
---
#### [replaced 011] Saliency-Motion Guided Trunk-Collateral Network for Unsupervised Video Object Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.05904v2](http://arxiv.org/pdf/2504.05904v2)**

> **作者:** Xiangyu Zheng; Wanyun Li; Songcheng He; Jianping Fan; Xiaoqiang Li; We Zhang
>
> **摘要:** Recent mainstream unsupervised video object segmentation (UVOS) motion-appearance approaches use either the bi-encoder structure to separately encode motion and appearance features, or the uni-encoder structure for joint encoding. However, these methods fail to properly balance the motion-appearance relationship. Consequently, even with complex fusion modules for motion-appearance integration, the extracted suboptimal features degrade the models' overall performance. Moreover, the quality of optical flow varies across scenarios, making it insufficient to rely solely on optical flow to achieve high-quality segmentation results. To address these challenges, we propose the Saliency-Motion guided Trunk-Collateral Network (SMTC-Net), which better balances the motion-appearance relationship and incorporates model's intrinsic saliency information to enhance segmentation performance. Specifically, considering that optical flow maps are derived from RGB images, they share both commonalities and differences. Accordingly, we propose a novel Trunk-Collateral structure for motion-appearance UVOS. The shared trunk backbone captures the motion-appearance commonality, while the collateral branch learns the uniqueness of motion features. Furthermore, an Intrinsic Saliency guided Refinement Module (ISRM) is devised to efficiently leverage the model's intrinsic saliency information to refine high-level features, and provide pixel-level guidance for motion-appearance fusion, thereby enhancing performance without additional input. Experimental results show that SMTC-Net achieved state-of-the-art performance on three UVOS datasets ( 89.2% J&F on DAVIS-16, 76% J on YouTube-Objects, 86.4% J on FBMS ) and four standard video salient object detection (VSOD) benchmarks with the notable increase, demonstrating its effectiveness and superiority over previous methods.
>
---
#### [replaced 012] SeagrassFinder: Deep Learning for Eelgrass Detection and Coverage Estimation in the Wild
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.16147v2](http://arxiv.org/pdf/2412.16147v2)**

> **作者:** Jannik Elsäßer; Laura Weihl; Veronika Cheplygina; Lisbeth Tangaa Nielsen
>
> **摘要:** Seagrass meadows play a crucial role in marine ecosystems, providing benefits such as carbon sequestration, water quality improvement, and habitat provision. Monitoring the distribution and abundance of seagrass is essential for environmental impact assessments and conservation efforts. However, the current manual methods of analyzing underwater video data to assess seagrass coverage are time-consuming and subjective. This work explores the use of deep learning models to automate the process of seagrass detection and coverage estimation from underwater video data. We create a new dataset of over 8,300 annotated underwater images, and subsequently evaluate several deep learning architectures, including ResNet, InceptionNetV3, DenseNet, and Vision Transformer for the task of binary classification on the presence and absence of seagrass by transfer learning. The results demonstrate that deep learning models, particularly Vision Transformers, can achieve high performance in predicting eelgrass presence, with AUROC scores exceeding 0.95 on the final test dataset. The application of underwater image enhancement further improved the models' prediction capabilities. Furthermore, we introduce a novel approach for estimating seagrass coverage from video data, showing promising preliminary results that align with expert manual labels, and indicating potential for consistent and scalable monitoring. The proposed methodology allows for the efficient processing of large volumes of video data, enabling the acquisition of much more detailed information on seagrass distributions in comparison to current manual methods. This information is crucial for environmental impact assessments and monitoring programs, as seagrasses are important indicators of coastal ecosystem health. This project demonstrates the value that deep learning can bring to the field of marine ecology and environmental monitoring.
>
---
#### [replaced 013] Behind Maya: Building a Multilingual Vision Language Model
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08910v2](http://arxiv.org/pdf/2505.08910v2)**

> **作者:** Nahid Alam; Karthik Reddy Kanjula; Surya Guthikonda; Timothy Chung; Bala Krishna S Vegesna; Abhipsha Das; Anthony Susevski; Ryan Sze-Yin Chan; S M Iftekhar Uddin; Shayekh Bin Islam; Roshan Santhosh; Snegha A; Drishti Sharma; Chen Liu; Isha Chaturvedi; Genta Indra Winata; Ashvanth. S; Snehanshu Mukherjee; Alham Fikri Aji
>
> **备注:** Accepted at VLMs4ALL CVPR 2025 Workshop; corrected workshop name spelling
>
> **摘要:** In recent times, we have seen a rapid development of large Vision-Language Models (VLMs). They have shown impressive results on academic benchmarks, primarily in widely spoken languages but lack performance on low-resource languages and varied cultural contexts. To address these limitations, we introduce Maya, an open-source Multilingual VLM. Our contributions are: 1) a multilingual image-text pretraining dataset in eight languages, based on the LLaVA pretraining dataset; and 2) a multilingual image-text model supporting these languages, enhancing cultural and linguistic comprehension in vision-language tasks. Code available at https://github.com/nahidalam/maya.
>
---
#### [replaced 014] StyleMorpheus: A Style-Based 3D-Aware Morphable Face Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11792v2](http://arxiv.org/pdf/2503.11792v2)**

> **作者:** Peizhi Yan; Rabab K. Ward; Dan Wang; Qiang Tang; Shan Du
>
> **备注:** 13 pages, work was completed in 2023
>
> **摘要:** For 3D face modeling, the recently developed 3D-aware neural rendering methods are able to render photorealistic face images with arbitrary viewing directions. The training of the parametric controllable 3D-aware face models, however, still relies on a large-scale dataset that is lab-collected. To address this issue, this paper introduces "StyleMorpheus", the first style-based neural 3D Morphable Face Model (3DMM) that is trained on in-the-wild images. It inherits 3DMM's disentangled controllability (over face identity, expression, and appearance) but without the need for accurately reconstructed explicit 3D shapes. StyleMorpheus employs an auto-encoder structure. The encoder aims at learning a representative disentangled parametric code space and the decoder improves the disentanglement using shape and appearance-related style codes in the different sub-modules of the network. Furthermore, we fine-tune the decoder through style-based generative adversarial learning to achieve photorealistic 3D rendering quality. The proposed style-based design enables StyleMorpheus to achieve state-of-the-art 3D-aware face reconstruction results, while also allowing disentangled control of the reconstructed face. Our model achieves real-time rendering speed, allowing its use in virtual reality applications. We also demonstrate the capability of the proposed style-based design in face editing applications such as style mixing and color editing. Project homepage: https://github.com/ubc-3d-vision-lab/StyleMorpheus.
>
---
#### [replaced 015] CreativeSynth: Cross-Art-Attention for Artistic Image Synthesis with Multimodal Diffusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2401.14066v3](http://arxiv.org/pdf/2401.14066v3)**

> **作者:** Nisha Huang; Weiming Dong; Yuxin Zhang; Fan Tang; Ronghui Li; Chongyang Ma; Xiu Li; Tong-Yee Lee; Changsheng Xu
>
> **摘要:** Although remarkable progress has been made in image style transfer, style is just one of the components of artistic paintings. Directly transferring extracted style features to natural images often results in outputs with obvious synthetic traces. This is because key painting attributes including layout, perspective, shape, and semantics often cannot be conveyed and expressed through style transfer. Large-scale pretrained text-to-image generation models have demonstrated their capability to synthesize a vast amount of high-quality images. However, even with extensive textual descriptions, it is challenging to fully express the unique visual properties and details of paintings. Moreover, generic models often disrupt the overall artistic effect when modifying specific areas, making it more complicated to achieve a unified aesthetic in artworks. Our main novel idea is to integrate multimodal semantic information as a synthesis guide into artworks, rather than transferring style to the real world. We also aim to reduce the disruption to the harmony of artworks while simplifying the guidance conditions. Specifically, we propose an innovative multi-task unified framework called CreativeSynth, based on the diffusion model with the ability to coordinate multimodal inputs. CreativeSynth combines multimodal features with customized attention mechanisms to seamlessly integrate real-world semantic content into the art domain through Cross-Art-Attention for aesthetic maintenance and semantic fusion. We demonstrate the results of our method across a wide range of different art categories, proving that CreativeSynth bridges the gap between generative models and artistic expression. Code and results are available at https://github.com/haha-lisa/CreativeSynth.
>
---
#### [replaced 016] WildFireCan-MMD: A Multimodal Dataset for Classification of User-Generated Content During Wildfires in Canada
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.13231v2](http://arxiv.org/pdf/2504.13231v2)**

> **作者:** Braeden Sherritt; Isar Nejadgholi; Marzieh Amini
>
> **摘要:** Rapid information access is vital during wildfires, yet traditional data sources are slow and costly. Social media offers real-time updates, but extracting relevant insights remains a challenge. We present WildFireCan-MMD, a new multimodal dataset of X posts from recent Canadian wildfires, annotated across twelve key themes. Evaluating both vision-language models and custom-trained classifiers, we show that while zero-shot prompting offers quick deployment, even simple trained models outperform them when labelled data is available. Our best-performing transformer-based fine-tuned model reaches 83% f-score, outperforming gpt4 by 23%. As a use case, we demonstrate how this model can be used to uncover trends during wildfires. Our findings highlight the enduring importance of tailored datasets and task-specific training. Importantly, such datasets should be localized, as disaster response requirements vary across regions and contexts.
>
---
#### [replaced 017] Measuring Student Behavioral Engagement using Histogram of Actions
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2307.09420v2](http://arxiv.org/pdf/2307.09420v2)**

> **作者:** Ahmed Abdelkawy; Aly Farag; Islam Alkabbany; Asem Ali; Chris Foreman; Thomas Tretter; Nicholas Hindy
>
> **摘要:** In this paper, we propose a novel technique for measuring behavioral engagement through students' actions recognition. The proposed approach recognizes student actions then predicts the student behavioral engagement level. For student action recognition, we use human skeletons to model student postures and upper body movements. To learn the dynamics of student upper body, a 3D-CNN model is used. The trained 3D-CNN model is used to recognize actions within every 2minute video segment then these actions are used to build a histogram of actions which encodes the student actions and their frequencies. This histogram is utilized as an input to SVM classifier to classify whether the student is engaged or disengaged. To evaluate the proposed framework, we build a dataset consisting of 1414 2-minute video segments annotated with 13 actions and 112 video segments annotated with two engagement levels. Experimental results indicate that student actions can be recognized with top 1 accuracy 83.63% and the proposed framework can capture the average engagement of the class.
>
---
#### [replaced 018] Teaching Humans Subtle Differences with DIFFusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.08046v2](http://arxiv.org/pdf/2504.08046v2)**

> **作者:** Mia Chiquier; Orr Avrech; Yossi Gandelsman; Berthy Feng; Katherine Bouman; Carl Vondrick
>
> **摘要:** Scientific expertise often requires recognizing subtle visual differences that remain challenging to articulate even for domain experts. We present a system that leverages generative models to automatically discover and visualize minimal discriminative features between categories while preserving instance identity. Our method generates counterfactual visualizations with subtle, targeted transformations between classes, performing well even in domains where data is sparse, examples are unpaired, and category boundaries resist verbal description. Experiments across six domains, including black hole simulations, butterfly taxonomy, and medical imaging, demonstrate accurate transitions with limited training data, highlighting both established discriminative features and novel subtle distinctions that measurably improved category differentiation. User studies confirm our generated counterfactuals significantly outperform traditional approaches in teaching humans to correctly differentiate between fine-grained classes, showing the potential of generative models to advance visual learning and scientific research.
>
---
#### [replaced 019] A Trust-Guided Approach to MR Image Reconstruction with Side Information
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.03021v2](http://arxiv.org/pdf/2501.03021v2)**

> **作者:** Arda Atalık; Sumit Chopra; Daniel K. Sodickson
>
> **备注:** 27 pages, 9 figures
>
> **摘要:** Reducing MRI scan times can improve patient care and lower healthcare costs. Many acceleration methods are designed to reconstruct diagnostic-quality images from sparse k-space data, via an ill-posed or ill-conditioned linear inverse problem (LIP). To address the resulting ambiguities, it is crucial to incorporate prior knowledge into the optimization problem, e.g., in the form of regularization. Another form of prior knowledge less commonly used in medical imaging is the readily available auxiliary data (a.k.a. side information) obtained from sources other than the current acquisition. In this paper, we present the Trust- Guided Variational Network (TGVN), an end-to-end deep learning framework that effectively and reliably integrates side information into LIPs. We demonstrate its effectiveness in multi-coil, multi-contrast MRI reconstruction, where incomplete or low-SNR measurements from one contrast are used as side information to reconstruct high-quality images of another contrast from heavily under-sampled data. TGVN is robust across different contrasts, anatomies, and field strengths. Compared to baselines utilizing side information, TGVN achieves superior image quality while preserving subtle pathological features even at challenging acceleration levels, drastically speeding up acquisition while minimizing hallucinations. Source code and dataset splits are available on github.com/sodicksonlab/TGVN.
>
---
#### [replaced 020] DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.14347v3](http://arxiv.org/pdf/2411.14347v3)**

> **作者:** Tianhe Ren; Yihao Chen; Qing Jiang; Zhaoyang Zeng; Yuda Xiong; Wenlong Liu; Zhengyu Ma; Junyi Shen; Yuan Gao; Xiaoke Jiang; Xingyu Chen; Zhuheng Song; Yuhong Zhang; Hongjie Huang; Han Gao; Shilong Liu; Hao Zhang; Feng Li; Kent Yu; Lei Zhang
>
> **备注:** Technical Report
>
> **摘要:** In this paper, we introduce DINO-X, which is a unified object-centric vision model developed by IDEA Research with the best open-world object detection performance to date. DINO-X employs the same Transformer-based encoder-decoder architecture as Grounding DINO 1.5 to pursue an object-level representation for open-world object understanding. To make long-tailed object detection easy, DINO-X extends its input options to support text prompt, visual prompt, and customized prompt. With such flexible prompt options, we develop a universal object prompt to support prompt-free open-world detection, making it possible to detect anything in an image without requiring users to provide any prompt. To enhance the model's core grounding capability, we have constructed a large-scale dataset with over 100 million high-quality grounding samples, referred to as Grounding-100M, for advancing the model's open-vocabulary detection performance. Pre-training on such a large-scale grounding dataset leads to a foundational object-level representation, which enables DINO-X to integrate multiple perception heads to simultaneously support multiple object perception and understanding tasks, including detection, segmentation, pose estimation, object captioning, object-based QA, etc. Experimental results demonstrate the superior performance of DINO-X. Specifically, the DINO-X Pro model achieves 56.0 AP, 59.8 AP, and 52.4 AP on the COCO, LVIS-minival, and LVIS-val zero-shot object detection benchmarks, respectively. Notably, it scores 63.3 AP and 56.5 AP on the rare classes of LVIS-minival and LVIS-val benchmarks, improving the previous SOTA performance by 5.8 AP and 5.0 AP. Such a result underscores its significantly improved capacity for recognizing long-tailed objects.
>
---
#### [replaced 021] A Deep Learning-Driven Inhalation Injury Grading Assistant Using Bronchoscopy Images
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.08517v2](http://arxiv.org/pdf/2505.08517v2)**

> **作者:** Yifan Li; Alan W Pang; Jo Woon Chong
>
> **摘要:** Inhalation injuries present a challenge in clinical diagnosis and grading due to Conventional grading methods such as the Abbreviated Injury Score (AIS) being subjective and lacking robust correlation with clinical parameters like mechanical ventilation duration and patient mortality. This study introduces a novel deep learning-based diagnosis assistant tool for grading inhalation injuries using bronchoscopy images to overcome subjective variability and enhance consistency in severity assessment. Our approach leverages data augmentation techniques, including graphic transformations, Contrastive Unpaired Translation (CUT), and CycleGAN, to address the scarcity of medical imaging data. We evaluate the classification performance of two deep learning models, GoogLeNet and Vision Transformer (ViT), across a dataset significantly expanded through these augmentation methods. The results demonstrate GoogLeNet combined with CUT as the most effective configuration for grading inhalation injuries through bronchoscopy images and achieves a classification accuracy of 97.8%. The histograms and frequency analysis evaluations reveal variations caused by the augmentation CUT with distribution changes in the histogram and texture details of the frequency spectrum. PCA visualizations underscore the CUT substantially enhances class separability in the feature space. Moreover, Grad-CAM analyses provide insight into the decision-making process; mean intensity for CUT heatmaps is 119.6, which significantly exceeds 98.8 of the original datasets. Our proposed tool leverages mechanical ventilation periods as a novel grading standard, providing comprehensive diagnostic support.
>
---
#### [replaced 022] Video-R1: Reinforcing Video Reasoning in MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21776v3](http://arxiv.org/pdf/2503.21776v3)**

> **作者:** Kaituo Feng; Kaixiong Gong; Bohao Li; Zonghao Guo; Yibing Wang; Tianshuo Peng; Junfei Wu; Xiaoying Zhang; Benyou Wang; Xiangyu Yue
>
> **备注:** Project page: https://github.com/tulerfeng/Video-R1
>
> **摘要:** Inspired by DeepSeek-R1's success in eliciting reasoning abilities through rule-based reinforcement learning (RL), we introduce Video-R1 as the first attempt to systematically explore the R1 paradigm for incentivizing video reasoning within multimodal large language models (MLLMs). However, directly applying RL training with the GRPO algorithm to video reasoning presents two primary challenges: (i) a lack of temporal modeling for video reasoning, and (ii) the scarcity of high-quality video-reasoning data. To address these issues, we first propose the T-GRPO algorithm, which encourages models to utilize temporal information in videos for reasoning. Additionally, instead of relying solely on video data, we incorporate high-quality image-reasoning data into the training process. We have constructed two datasets: Video-R1-CoT-165k for SFT cold start and Video-R1-260k for RL training, both comprising image and video data. Experimental results demonstrate that Video-R1 achieves significant improvements on video reasoning benchmarks such as VideoMMMU and VSI-Bench, as well as on general video benchmarks including MVBench and TempCompass, etc. Notably, Video-R1-7B attains a 37.1% accuracy on video spatial reasoning benchmark VSI-bench, surpassing the commercial proprietary model GPT-4o. All code, models, and data are released in: https://github.com/tulerfeng/Video-R1.
>
---
#### [replaced 023] Scaling Laws for Black box Adversarial Attacks
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16782v2](http://arxiv.org/pdf/2411.16782v2)**

> **作者:** Chuan Liu; Huanran Chen; Yichi Zhang; Yinpeng Dong; Jun Zhu
>
> **摘要:** Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.
>
---
#### [replaced 024] PEP-GS: Perceptually-Enhanced Precise Structured 3D Gaussians for View-Adaptive Rendering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.05731v3](http://arxiv.org/pdf/2411.05731v3)**

> **作者:** Junxi Jin; Xiulai Li; Haiping Huang; Lianjun Liu; Yujie Sun; Logan Liu
>
> **摘要:** Recently, 3D Gaussian Splatting (3D-GS) has achieved significant success in real-time, high-quality 3D scene rendering. However, it faces several challenges, including Gaussian redundancy, limited ability to capture view-dependent effects, and difficulties in handling complex lighting and specular reflections. Additionally, methods that use spherical harmonics for color representation often struggle to effectively capture anisotropic components, especially when modeling view-dependent colors under complex lighting conditions, leading to insufficient contrast and unnatural color saturation. To address these limitations, we introduce PEP-GS, a perceptually-enhanced framework that dynamically predicts Gaussian attributes, including opacity, color, and covariance. We replace traditional spherical harmonics with a Hierarchical Granular-Structural Attention mechanism, which enables more accurate modeling of complex view-dependent color effects. By employing a stable and interpretable framework for opacity and covariance estimation, PEP-GS avoids the removal of essential Gaussians prematurely, ensuring a more accurate scene representation. Furthermore, perceptual optimization is applied to the final rendered images, enhancing perceptual consistency across different views and ensuring high-quality renderings with improved texture fidelity and fine-scale detail preservation. Experimental results demonstrate that PEP-GS outperforms state-of-the-art methods, particularly in challenging scenarios involving view-dependent effects and fine-scale details.
>
---
#### [replaced 025] Leveraging Multi-Modal Information to Enhance Dataset Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08605v2](http://arxiv.org/pdf/2505.08605v2)**

> **作者:** Zhe Li; Hadrien Reynaud; Bernhard Kainz
>
> **备注:** 10 pages
>
> **摘要:** Dataset distillation aims to create a compact and highly representative synthetic dataset that preserves the knowledge of a larger real dataset. While existing methods primarily focus on optimizing visual representations, incorporating additional modalities and refining object-level information can significantly improve the quality of distilled datasets. In this work, we introduce two key enhancements to dataset distillation: caption-guided supervision and object-centric masking. To integrate textual information, we propose two strategies for leveraging caption features: the feature concatenation, where caption embeddings are fused with visual features at the classification stage, and caption matching, which introduces a caption-based alignment loss during training to ensure semantic coherence between real and synthetic data. Additionally, we apply segmentation masks to isolate target objects and remove background distractions, introducing two loss functions designed for object-centric learning: masked feature alignment loss and masked gradient matching loss. Comprehensive evaluations demonstrate that integrating caption-based guidance and object-centric masking enhances dataset distillation, leading to synthetic datasets that achieve superior performance on downstream tasks.
>
---
#### [replaced 026] Generative Pre-trained Autoregressive Diffusion Transformer
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07344v2](http://arxiv.org/pdf/2505.07344v2)**

> **作者:** Yuan Zhang; Jiacheng Jiang; Guoqing Ma; Zhiying Lu; Haoyang Huang; Jianlong Yuan; Nan Duan
>
> **摘要:** In this work, we present GPDiT, a Generative Pre-trained Autoregressive Diffusion Transformer that unifies the strengths of diffusion and autoregressive modeling for long-range video synthesis, within a continuous latent space. Instead of predicting discrete tokens, GPDiT autoregressively predicts future latent frames using a diffusion loss, enabling natural modeling of motion dynamics and semantic consistency across frames. This continuous autoregressive framework not only enhances generation quality but also endows the model with representation capabilities. Additionally, we introduce a lightweight causal attention variant and a parameter-free rotation-based time-conditioning mechanism, improving both the training and inference efficiency. Extensive experiments demonstrate that GPDiT achieves strong performance in video generation quality, video representation ability, and few-shot learning tasks, highlighting its potential as an effective framework for video modeling in continuous space.
>
---
#### [replaced 027] Examining the Source of Defects from a Mechanical Perspective for 3D Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.05901v2](http://arxiv.org/pdf/2505.05901v2)**

> **作者:** Hanzhe Liang; Aoran Wang; Jie Zhou; Xin Jin; Can Gao; Jinbao Wang
>
> **备注:** 26 pages
>
> **摘要:** In this paper, we explore a novel approach to 3D anomaly detection (AD) that goes beyond merely identifying anomalies based on structural characteristics. Our primary perspective is that most anomalies arise from unpredictable defective forces originating from both internal and external sources. To address these anomalies, we seek out opposing forces that can help correct them. Therefore, we introduce the Mechanics Complementary Model-based Framework for the 3D-AD task (MC4AD), which generates internal and external corrective forces for each point. We first propose a Diverse Anomaly-Generation (DA-Gen) module designed to simulate various types of anomalies. Next, we present the Corrective Force Prediction Network (CFP-Net), which uses complementary representations for point-level analysis to simulate the different contributions from internal and external corrective forces. To ensure the corrective forces are constrained effectively, we have developed a combined loss function that includes a new symmetric loss and an overall loss. Notably, we implement a Hierarchical Quality Control (HQC) strategy based on a three-way decision process and contribute a dataset titled Anomaly-IntraVariance, which incorporates intraclass variance to evaluate our model. As a result, the proposed MC4AD has been proven effective through theory and experimentation. The experimental results demonstrate that our approach yields nine state-of-the-art performances, achieving optimal results with minimal parameters and the fastest inference speed across five existing datasets, in addition to the proposed Anomaly-IntraVariance dataset. The source is available at https://github.com/hzzzzzhappy/MC4AD
>
---
#### [replaced 028] UniCAD: Efficient and Extendable Architecture for Multi-Task Computer-Aided Diagnosis System
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.09178v2](http://arxiv.org/pdf/2505.09178v2)**

> **作者:** Yitao Zhu; Yuan Yin; Zhenrong Shen; Zihao Zhao; Haiyu Song; Sheng Wang; Dinggang Shen; Qian Wang
>
> **备注:** 14 pages
>
> **摘要:** The growing complexity and scale of visual model pre-training have made developing and deploying multi-task computer-aided diagnosis (CAD) systems increasingly challenging and resource-intensive. Furthermore, the medical imaging community lacks an open-source CAD platform to enable the rapid creation of efficient and extendable diagnostic models. To address these issues, we propose UniCAD, a unified architecture that leverages the robust capabilities of pre-trained vision foundation models to seamlessly handle both 2D and 3D medical images while requiring only minimal task-specific parameters. UniCAD introduces two key innovations: (1) Efficiency: A low-rank adaptation strategy is employed to adapt a pre-trained visual model to the medical image domain, achieving performance on par with fully fine-tuned counterparts while introducing only 0.17% trainable parameters. (2) Plug-and-Play: A modular architecture that combines a frozen foundation model with multiple plug-and-play experts, enabling diverse tasks and seamless functionality expansion. Building on this unified CAD architecture, we establish an open-source platform where researchers can share and access lightweight CAD experts, fostering a more equitable and efficient research ecosystem. Comprehensive experiments across 12 diverse medical datasets demonstrate that UniCAD consistently outperforms existing methods in both accuracy and deployment efficiency. The source code and project page are available at https://mii-laboratory.github.io/UniCAD/.
>
---
#### [replaced 029] BiECVC: Gated Diversification of Bidirectional Contexts for Learned Video Compression
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.09193v2](http://arxiv.org/pdf/2505.09193v2)**

> **作者:** Wei Jiang; Junru Li; Kai Zhang; Li Zhang
>
> **备注:** The first learned video codec that surpasses VTM 13.2 RA across all standard test datasets. Code will be available at https://github.com/JiangWeibeta/ECVC
>
> **摘要:** Recent forward prediction-based learned video compression (LVC) methods have achieved impressive results, even surpassing VVC reference software VTM under the Low Delay B (LDB) configuration. In contrast, learned bidirectional video compression (BVC) remains underexplored and still lags behind its forward-only counterparts. This performance gap is mainly due to the limited ability to extract diverse and accurate contexts: most existing BVCs primarily exploit temporal motion while neglecting non-local correlations across frames. Moreover, they lack the adaptability to dynamically suppress harmful contexts arising from fast motion or occlusion. To tackle these challenges, we propose BiECVC, a BVC framework that incorporates diversified local and non-local context modeling along with adaptive context gating. For local context enhancement, BiECVC reuses high-quality features from lower layers and aligns them using decoded motion vectors without introducing extra motion overhead. To model non-local dependencies efficiently, we adopt a linear attention mechanism that balances performance and complexity. To further mitigate the impact of inaccurate context prediction, we introduce Bidirectional Context Gating, inspired by data-dependent decay in recent autoregressive language models, to dynamically filter contextual information based on conditional coding results. Extensive experiments demonstrate that BiECVC achieves state-of-the-art performance, reducing the bit-rate by 13.4% and 15.7% compared to VTM 13.2 under the Random Access (RA) configuration with intra periods of 32 and 64, respectively. To our knowledge, BiECVC is the first learned video codec to surpass VTM 13.2 RA across all standard test datasets. Code will be available at https://github.com/JiangWeibeta/ECVC.
>
---
#### [replaced 030] Single View Garment Reconstruction Using Diffusion Mapping Via Pattern Coordinates
- **分类: cs.GR; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.08353v2](http://arxiv.org/pdf/2504.08353v2)**

> **作者:** Ren Li; Cong Cao; Corentin Dumery; Yingxuan You; Hao Li; Pascal Fua
>
> **备注:** SIGGRAPH 2025
>
> **摘要:** Reconstructing 3D clothed humans from images is fundamental to applications like virtual try-on, avatar creation, and mixed reality. While recent advances have enhanced human body recovery, accurate reconstruction of garment geometry -- especially for loose-fitting clothing -- remains an open challenge. We present a novel method for high-fidelity 3D garment reconstruction from single images that bridges 2D and 3D representations. Our approach combines Implicit Sewing Patterns (ISP) with a generative diffusion model to learn rich garment shape priors in a 2D UV space. A key innovation is our mapping model that establishes correspondences between 2D image pixels, UV pattern coordinates, and 3D geometry, enabling joint optimization of both 3D garment meshes and the corresponding 2D patterns by aligning learned priors with image observations. Despite training exclusively on synthetically simulated cloth data, our method generalizes effectively to real-world images, outperforming existing approaches on both tight- and loose-fitting garments. The reconstructed garments maintain physical plausibility while capturing fine geometric details, enabling downstream applications including garment retargeting and texture manipulation.
>
---
#### [replaced 031] A portable diagnosis model for Keratoconus using a smartphone
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.08616v2](http://arxiv.org/pdf/2505.08616v2)**

> **作者:** Yifan Li; Peter Ho; Jo Woon Chong
>
> **摘要:** Keratoconus (KC) is a corneal disorder that results in blurry and distorted vision. Traditional diagnostic tools, while effective, are often bulky, costly, and require professional operation. In this paper, we present a portable and innovative methodology for diagnosing. Our proposed approach first captures the image reflected on the eye's cornea when a smartphone screen-generated Placido disc sheds its light on an eye, then utilizes a two-stage diagnosis for identifying the KC cornea and pinpointing the location of the KC on the cornea. The first stage estimates the height and width of the Placido disc extracted from the captured image to identify whether it has KC. In this KC identification, k-means clustering is implemented to discern statistical characteristics, such as height and width values of extracted Placido discs, from non-KC (control) and KC-affected groups. The second stage involves the creation of a distance matrix, providing a precise localization of KC on the cornea, which is critical for efficient treatment planning. The analysis of these distance matrices, paired with a logistic regression model and robust statistical analysis, reveals a clear distinction between control and KC groups. The logistic regression model, which classifies small areas on the cornea as either control or KC-affected based on the corresponding inter-disc distances in the distance matrix, reported a classification accuracy of 96.94%, which indicates that we can effectively pinpoint the protrusion caused by KC. This comprehensive, smartphone-based method is expected to detect KC and streamline timely treatment.
>
---
#### [replaced 032] S2-Track: A Simple yet Strong Approach for End-to-End 3D Multi-Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.02147v2](http://arxiv.org/pdf/2406.02147v2)**

> **作者:** Tao Tang; Lijun Zhou; Pengkun Hao; Zihang He; Kalok Ho; Shuo Gu; Zhihui Hao; Haiyang Sun; Kun Zhan; Peng Jia; XianPeng Lang; Xiaodan Liang
>
> **摘要:** 3D multiple object tracking (MOT) plays a crucial role in autonomous driving perception. Recent end-to-end query-based trackers simultaneously detect and track objects, which have shown promising potential for the 3D MOT task. However, existing methods are still in the early stages of development and lack systematic improvements, failing to track objects in certain complex scenarios, like occlusions and the small size of target object's situations. In this paper, we first summarize the current end-to-end 3D MOT framework by decomposing it into three constituent parts: query initialization, query propagation, and query matching. Then we propose corresponding improvements, which lead to a strong yet simple tracker: S2-Track. Specifically, for query initialization, we present 2D-Prompted Query Initialization, which leverages predicted 2D object and depth information to prompt an initial estimate of the object's 3D location. For query propagation, we introduce an Uncertainty-aware Probabilistic Decoder to capture the uncertainty of complex environment in object prediction with probabilistic attention. For query matching, we propose a Hierarchical Query Denoising strategy to enhance training robustness and convergence. As a result, our S2-Track achieves state-of-the-art performance on nuScenes benchmark, i.e., 66.3% AMOTA on test split, surpassing the previous best end-to-end solution by a significant margin of 8.9% AMOTA. We achieve 1st place on the nuScenes tracking task leaderboard.
>
---
#### [replaced 033] CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.16028v2](http://arxiv.org/pdf/2412.16028v2)**

> **作者:** Jungho Lee; Suhwan Cho; Taeoh Kim; Ho-Deok Jang; Minhyeok Lee; Geonho Cha; Dongyoon Wee; Dogyoon Lee; Sangyoun Lee
>
> **备注:** CVPR 2025, Project Page: https://Jho-Yonsei.github.io/CoCoGaussian/
>
> **摘要:** 3D Gaussian Splatting (3DGS) has attracted significant attention for its high-quality novel view rendering, inspiring research to address real-world challenges. While conventional methods depend on sharp images for accurate scene reconstruction, real-world scenarios are often affected by defocus blur due to finite depth of field, making it essential to account for realistic 3D scene representation. In this study, we propose CoCoGaussian, a Circle of Confusion-aware Gaussian Splatting that enables precise 3D scene representation using only defocused images. CoCoGaussian addresses the challenge of defocus blur by modeling the Circle of Confusion (CoC) through a physically grounded approach based on the principles of photographic defocus. Exploiting 3D Gaussians, we compute the CoC diameter from depth and learnable aperture information, generating multiple Gaussians to precisely capture the CoC shape. Furthermore, we introduce a learnable scaling factor to enhance robustness and provide more flexibility in handling unreliable depth in scenes with reflective or refractive surfaces. Experiments on both synthetic and real-world datasets demonstrate that CoCoGaussian achieves state-of-the-art performance across multiple benchmarks.
>
---
#### [replaced 034] TactileNet: Bridging the Accessibility Gap with AI-Generated Tactile Graphics for Individuals with Vision Impairment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.04722v2](http://arxiv.org/pdf/2504.04722v2)**

> **作者:** Adnan Khan; Alireza Choubineh; Mai A. Shaaban; Abbas Akkasi; Majid Komeili
>
> **摘要:** Tactile graphics are essential for providing access to visual information for the 43 million people globally living with vision loss. Traditional methods for creating these graphics are labor-intensive and cannot meet growing demand. We introduce TactileNet, the first comprehensive dataset and AI-driven framework for generating embossing-ready 2D tactile templates using text-to-image Stable Diffusion (SD) models. By integrating Low-Rank Adaptation (LoRA) and DreamBooth, our method fine-tunes SD models to produce high-fidelity, guideline-compliant graphics while reducing computational costs. Quantitative evaluations with tactile experts show 92.86% adherence to accessibility standards. Structural fidelity analysis revealed near-human design similarity, with an SSIM of 0.538 between generated graphics and expert-designed tactile images. Notably, our method preserves object silhouettes better than human designs (SSIM = 0.259 vs. 0.215 for binary masks), addressing a key limitation of manual tactile abstraction. The framework scales to 32,000 images (7,050 high-quality) across 66 classes, with prompt editing enabling customizable outputs (e.g., adding or removing details). By automating the 2D template generation step-compatible with standard embossing workflows-TactileNet accelerates production while preserving design flexibility. This work demonstrates how AI can augment (not replace) human expertise to bridge the accessibility gap in education and beyond. Code, data, and models will be publicly released to foster further research.
>
---
#### [replaced 035] HaHeAE: Learning Generalisable Joint Representations of Human Hand and Head Movements in Extended Reality
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.16430v2](http://arxiv.org/pdf/2410.16430v2)**

> **作者:** Zhiming Hu; Guanhua Zhang; Zheming Yin; Daniel Haeufle; Syn Schmitt; Andreas Bulling
>
> **备注:** Link: https://zhiminghu.net/hu25_haheae
>
> **摘要:** Human hand and head movements are the most pervasive input modalities in extended reality (XR) and are significant for a wide range of applications. However, prior works on hand and head modelling in XR only explored a single modality or focused on specific applications. We present HaHeAE - a novel self-supervised method for learning generalisable joint representations of hand and head movements in XR. At the core of our method is an autoencoder (AE) that uses a graph convolutional network-based semantic encoder and a diffusion-based stochastic encoder to learn the joint semantic and stochastic representations of hand-head movements. It also features a diffusion-based decoder to reconstruct the original signals. Through extensive evaluations on three public XR datasets, we show that our method 1) significantly outperforms commonly used self-supervised methods by up to 74.0% in terms of reconstruction quality and is generalisable across users, activities, and XR environments, 2) enables new applications, including interpretable hand-head cluster identification and variable hand-head movement generation, and 3) can serve as an effective feature extractor for downstream tasks. Together, these results demonstrate the effectiveness of our method and underline the potential of self-supervised methods for jointly modelling hand-head behaviours in extended reality.
>
---
#### [replaced 036] UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08787v2](http://arxiv.org/pdf/2505.08787v2)**

> **作者:** Hanjung Kim; Jaehyun Kang; Hyolim Kang; Meedeum Cho; Seon Joo Kim; Youngwoon Lee
>
> **备注:** Project Page: https://kimhanjung.github.io/UniSkill/
>
> **摘要:** Mimicry is a fundamental learning mechanism in humans, enabling individuals to learn new tasks by observing and imitating experts. However, applying this ability to robots presents significant challenges due to the inherent differences between human and robot embodiments in both their visual appearance and physical capabilities. While previous methods bridge this gap using cross-embodiment datasets with shared scenes and tasks, collecting such aligned data between humans and robots at scale is not trivial. In this paper, we propose UniSkill, a novel framework that learns embodiment-agnostic skill representations from large-scale cross-embodiment video data without any labels, enabling skills extracted from human video prompts to effectively transfer to robot policies trained only on robot data. Our experiments in both simulation and real-world environments show that our cross-embodiment skills successfully guide robots in selecting appropriate actions, even with unseen video prompts. The project website can be found at: https://kimhanjung.github.io/UniSkill.
>
---
#### [replaced 037] Learned Image Compression with Dictionary-based Entropy Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.00496v2](http://arxiv.org/pdf/2504.00496v2)**

> **作者:** Jingbo Lu; Leheng Zhang; Xingyu Zhou; Mu Li; Wen Li; Shuhang Gu
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Learned image compression methods have attracted great research interest and exhibited superior rate-distortion performance to the best classical image compression standards of the present. The entropy model plays a key role in learned image compression, which estimates the probability distribution of the latent representation for further entropy coding. Most existing methods employed hyper-prior and auto-regressive architectures to form their entropy models. However, they only aimed to explore the internal dependencies of latent representation while neglecting the importance of extracting prior from training data. In this work, we propose a novel entropy model named Dictionary-based Cross Attention Entropy model, which introduces a learnable dictionary to summarize the typical structures occurring in the training dataset to enhance the entropy model. Extensive experimental results have demonstrated that the proposed model strikes a better balance between performance and latency, achieving state-of-the-art results on various benchmark datasets.
>
---
#### [replaced 038] Towards user-centered interactive medical image segmentation in VR with an assistive AI agent
- **分类: cs.HC; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07214v2](http://arxiv.org/pdf/2505.07214v2)**

> **作者:** Pascal Spiegler; Arash Harirpoush; Yiming Xiao
>
> **摘要:** Crucial in disease analysis and surgical planning, manual segmentation of volumetric medical scans (e.g. MRI, CT) is laborious, error-prone, and challenging to master, while fully automatic algorithms can benefit from user feedback. Therefore, with the complementary power of the latest radiological AI foundation models and virtual reality (VR)'s intuitive data interaction, we propose SAMIRA, a novel conversational AI agent that assists users with localizing, segmenting, and visualizing 3D medical concepts in VR. Through speech-based interaction, the agent helps users understand radiological features, locate clinical targets, and generate segmentation masks that can be refined with just a few point prompts. The system also supports true-to-scale 3D visualization of segmented pathology to enhance patient-specific anatomical understanding. Furthermore, to determine the optimal interaction paradigm under near-far attention-switching for refining segmentation masks in an immersive, human-in-the-loop workflow, we compare VR controller pointing, head pointing, and eye tracking as input modes. With a user study, evaluations demonstrated a high usability score (SUS=90.0 $\pm$ 9.0), low overall task load, as well as strong support for the proposed VR system's guidance, training potential, and integration of AI in radiological segmentation tasks.
>
---
#### [replaced 039] Towards Scalable IoT Deployment for Visual Anomaly Detection via Efficient Compression
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07119v2](http://arxiv.org/pdf/2505.07119v2)**

> **作者:** Arianna Stropeni; Francesco Borsatti; Manuel Barusco; Davide Dalle Pezze; Marco Fabris; Gian Antonio Susto
>
> **摘要:** Visual Anomaly Detection (VAD) is a key task in industrial settings, where minimizing operational costs is essential. Deploying deep learning models within Internet of Things (IoT) environments introduces specific challenges due to limited computational power and bandwidth of edge devices. This study investigates how to perform VAD effectively under such constraints by leveraging compact, efficient processing strategies. We evaluate several data compression techniques, examining the tradeoff between system latency and detection accuracy. Experiments on the MVTec AD benchmark demonstrate that significant compression can be achieved with minimal loss in anomaly detection performance compared to uncompressed data. Current results show up to 80% reduction in end-to-end inference time, including edge processing, transmission, and server computation.
>
---
#### [replaced 040] CoGenAV: Versatile Audio-Visual Representation Learning via Contrastive-Generative Synchronization
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.03186v2](http://arxiv.org/pdf/2505.03186v2)**

> **作者:** Detao Bai; Zhiheng Ma; Xihan Wei; Liefeng Bo
>
> **摘要:** The inherent synchronization between a speaker's lip movements, voice, and the underlying linguistic content offers a rich source of information for improving speech processing tasks, especially in challenging conditions where traditional audio-only systems falter. We introduce CoGenAV, a powerful and data-efficient model designed to learn versatile audio-visual representations applicable across a wide range of speech and audio-visual tasks. CoGenAV is trained by optimizing a dual objective derived from natural audio-visual synchrony, contrastive feature alignment and generative text prediction, using only 223 hours of labeled data from the LRS2 dataset. This contrastive-generative synchronization strategy effectively captures fundamental cross-modal correlations. We showcase the effectiveness and versatility of the learned CoGenAV representations on multiple benchmarks. When utilized for Audio-Visual Speech Recognition (AVSR) on LRS2, these representations contribute to achieving a state-of-the-art Word Error Rate (WER) of 1.27. They also enable strong performance in Visual Speech Recognition (VSR) with a WER of 20.5 on LRS2, and significantly improve performance in noisy environments by over 70%. Furthermore, CoGenAV representations benefit speech reconstruction tasks, boosting performance in Speech Enhancement and Separation, and achieve competitive results in audio-visual synchronization tasks like Active Speaker Detection (ASD). Our model will be open-sourced to facilitate further development and collaboration within both academia and industry.
>
---
#### [replaced 041] Translating Electrocardiograms to Cardiac Magnetic Resonance Imaging Useful for Cardiac Assessment and Disease Screening: A Multi-Center Study AI for ECG to CMR Translation Study
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.13602v2](http://arxiv.org/pdf/2411.13602v2)**

> **作者:** Zhengyao Ding; Ziyu Li; Yujian Hu; Youyao Xu; Chengchen Zhao; Yiheng Mao; Haitao Li; Zhikang Li; Qian Li; Jing Wang; Yue Chen; Mengjia Chen; Longbo Wang; Xuesen Chu; Weichao Pan; Ziyi Liu; Fei Wu; Hongkun Zhang; Ting Chen; Zhengxing Huang
>
> **备注:** 27 pages, 11 figures
>
> **摘要:** Cardiovascular diseases (CVDs) are the leading cause of global mortality, necessitating accessible and accurate diagnostic tools. While cardiac magnetic resonance imaging (CMR) provides gold-standard insights into cardiac structure and function, its clinical utility is limited by high cost and complexity. In contrast, electrocardiography (ECG) is inexpensive and widely available but lacks the granularity of CMR. We propose CardioNets, a deep learning framework that translates 12-lead ECG signals into CMR-level functional parameters and synthetic images, enabling scalable cardiac assessment. CardioNets integrates cross-modal contrastive learning and generative pretraining, aligning ECG with CMR-derived cardiac phenotypes and synthesizing high-resolution CMR images via a masked autoregressive model. Trained on 159,819 samples from five cohorts, including the UK Biobank (n=42,483) and MIMIC-IV-ECG (n=164,550), and externally validated on independent clinical datasets (n=3,767), CardioNets achieved strong performance across disease screening and phenotype estimation tasks. In the UK Biobank, it improved cardiac phenotype regression R2 by 24.8% and cardiomyopathy AUC by up to 39.3% over baseline models. In MIMIC, it increased AUC for pulmonary hypertension detection by 5.6%. Generated CMR images showed 36.6% higher SSIM and 8.7% higher PSNR than prior approaches. In a reader study, ECG-only CardioNets achieved 13.9% higher accuracy than human physicians using both ECG and real CMR. These results suggest that CardioNets offers a promising, low-cost alternative to CMR for large-scale CVD screening, particularly in resource-limited settings. Future efforts will focus on clinical deployment and regulatory validation of ECG-based synthetic imaging.
>
---
#### [replaced 042] IntrinsicEdit: Precise generative image manipulation in intrinsic space
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08889v2](http://arxiv.org/pdf/2505.08889v2)**

> **作者:** Linjie Lyu; Valentin Deschaintre; Yannick Hold-Geoffroy; Miloš Hašan; Jae Shin Yoon; Thomas Leimkühler; Christian Theobalt; Iliyan Georgiev
>
> **备注:** SIGGRAPH 2025 Journal track
>
> **摘要:** Generative diffusion models have advanced image editing with high-quality results and intuitive interfaces such as prompts and semantic drawing. However, these interfaces lack precise control, and the associated methods typically specialize on a single editing task. We introduce a versatile, generative workflow that operates in an intrinsic-image latent space, enabling semantic, local manipulation with pixel precision for a range of editing operations. Building atop the RGB-X diffusion framework, we address key challenges of identity preservation and intrinsic-channel entanglement. By incorporating exact diffusion inversion and disentangled channel manipulation, we enable precise, efficient editing with automatic resolution of global illumination effects -- all without additional data collection or model fine-tuning. We demonstrate state-of-the-art performance across a variety of tasks on complex images, including color and texture adjustments, object insertion and removal, global relighting, and their combinations.
>
---
#### [replaced 043] OSMLoc: Single Image-Based Visual Localization in OpenStreetMap with Fused Geometric and Semantic Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.08665v2](http://arxiv.org/pdf/2411.08665v2)**

> **作者:** Youqi Liao; Xieyuanli Chen; Shuhao Kang; Jianping Li; Zhen Dong; Hongchao Fan; Bisheng Yang
>
> **备注:** 16 pages, technical report
>
> **摘要:** OpenStreetMap (OSM), a rich and versatile source of volunteered geographic information (VGI), facilitates human self-localization and scene understanding by integrating nearby visual observations with vectorized map data. However, the disparity in modalities and perspectives poses a major challenge for effectively matching camera imagery with compact map representations, thereby limiting the full potential of VGI data in real-world localization applications. Inspired by the fact that the human brain relies on the fusion of geometric and semantic understanding for spatial localization tasks, we propose the OSMLoc in this paper. OSMLoc is a brain-inspired visual localization approach based on first-person-view images against the OSM maps. It integrates semantic and geometric guidance to significantly improve accuracy, robustness, and generalization capability. First, we equip the OSMLoc with the visual foundational model to extract powerful image features. Second, a geometry-guided depth distribution adapter is proposed to bridge the monocular depth estimation and camera-to-BEV transform. Thirdly, the semantic embeddings from the OSM data are utilized as auxiliary guidance for image-to-OSM feature matching. To validate the proposed OSMLoc, we collect a worldwide cross-area and cross-condition (CC) benchmark for extensive evaluation. Experiments on the MGL dataset, CC validation benchmark, and KITTI dataset have demonstrated the superiority of our method. Code, pre-trained models, CC validation benchmark, and additional results are available at: https://github.com/WHU-USI3DV/OSMLoc.
>
---
#### [replaced 044] SMURF: Continuous Dynamics for Motion-Deblurring Radiance Fields
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.07547v2](http://arxiv.org/pdf/2403.07547v2)**

> **作者:** Jungho Lee; Dogyoon Lee; Minhyeok Lee; Donghyung Kim; Sangyoun Lee
>
> **备注:** CVPRW 2025, Neural Fields Beyond Conventional Cameras, Project Page: https://jho-yonsei.github.io/SMURF/
>
> **摘要:** Neural radiance fields (NeRF) has attracted considerable attention for their exceptional ability in synthesizing novel views with high fidelity. However, the presence of motion blur, resulting from slight camera movements during extended shutter exposures, poses a significant challenge, potentially compromising the quality of the reconstructed 3D scenes. To effectively handle this issue, we propose sequential motion understanding radiance fields (SMURF), a novel approach that models continuous camera motion and leverages the explicit volumetric representation method for robustness to motion-blurred input images. The core idea of the SMURF is continuous motion blurring kernel (CMBK), a module designed to model a continuous camera movements for processing blurry inputs. Our model is evaluated against benchmark datasets and demonstrates state-of-the-art performance both quantitatively and qualitatively.
>
---
#### [replaced 045] CryoSAMU: Enhancing 3D Cryo-EM Density Maps of Protein Structures at Intermediate Resolution with Structure-Aware Multimodal U-Nets
- **分类: cs.CV; cs.AI; cs.LG; q-bio.BM**

- **链接: [http://arxiv.org/pdf/2503.20291v2](http://arxiv.org/pdf/2503.20291v2)**

> **作者:** Chenwei Zhang; Khanh Dao Duc
>
> **备注:** 19 pages, 6 main figures, 2 supplementary figures, 3 main tables, 4 supplementary tables
>
> **摘要:** Enhancing cryogenic electron microscopy (cryo-EM) 3D density maps at intermediate resolution (4-8 {\AA}) is crucial in protein structure determination. Recent advances in deep learning have led to the development of automated approaches for enhancing experimental cryo-EM density maps. Yet, these methods are not optimized for intermediate-resolution maps and rely on map density features alone. To address this, we propose CryoSAMU, a novel method designed to enhance 3D cryo-EM density maps of protein structures using structure-aware multimodal U-Nets and trained on curated intermediate-resolution density maps. We comprehensively evaluate CryoSAMU across various metrics and demonstrate its competitive performance compared to state-of-the-art methods. Notably, CryoSAMU achieves significantly faster processing speed, showing promise for future practical applications. Our code is available at https://github.com/chenwei-zhang/CryoSAMU.
>
---
#### [replaced 046] Improving Fine-Grained Control via Aggregation of Multiple Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.01262v3](http://arxiv.org/pdf/2410.01262v3)**

> **作者:** Conghan Yue; Zhengwei Peng; Shiyan Du; Zhi Ji; Chuangjian Cai; Le Wan; Dongyu Zhang
>
> **摘要:** While many diffusion models perform well when controlling for particular aspect among style, character, and interaction, they struggle with fine-grained control due to dataset limitations and intricate model architecture design. This paper first introduces a novel training-free algorithm in fine-grained generation, Aggregation of Multiple Diffusion Models (AMDM), which integrates features from multiple diffusion models into a specified model to activate specific features and enable fine-grained control. Experimental results demonstrate that AMDM significantly improves fine-grained control without training, validating its effectiveness. Additionally, it reveals that diffusion models initially focus on features such as position, attributes, and style, with later stages improving generation quality and consistency. AMDM offers a new perspective for tackling the challenges of fine-grained conditional control generation in diffusion models: We can fully utilize existing or develop new conditional diffusion models that control specific aspects, and then aggregate them using AMDM algorithm. This eliminates the need for constructing complex datasets, designing intricate model architectures, and incurring high training costs. Code is available at: https://github.com/Hammour-steak/AMDM.
>
---
#### [replaced 047] A Sliding Layer Merging Method for Efficient Depth-Wise Pruning in LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.19159v3](http://arxiv.org/pdf/2502.19159v3)**

> **作者:** Xuan Ding; Rui Sun; Yunjian Zhang; Xiu Yan; Yueqi Zhou; Kaihao Huang; Suzhong Fu; Angelica I Aviles-Rivero; Chuanlong Xie; Yao Zhu
>
> **摘要:** Compared to width-wise pruning, depth-wise pruning can significantly accelerate inference in resource-constrained scenarios. However, treating the entire Transformer layer as the minimum pruning unit may degrade model performance by indiscriminately discarding the entire information of the layer. This paper reveals the ``Patch-like'' feature relationship between layers in large language models by analyzing the correlation of the outputs of different layers in the reproducing kernel Hilbert space. Building on this observation, we propose a sliding layer merging method that dynamically selects and fuses consecutive layers from top to bottom according to a pre-defined similarity threshold, thereby simplifying the model structure while maintaining its performance. Extensive experiments on LLMs with various architectures and different parameter scales show that our method outperforms existing pruning techniques in both zero-shot inference performance and retraining recovery quality after pruning. In particular, in the experiment with 35% pruning on the Vicuna-7B model, our method achieved a 1.654% improvement in average performance on zero-shot tasks compared to the existing method. Moreover, we further reveal the potential of combining depth pruning with width pruning to enhance the pruning effect. Our codes are available at https://github.com/920927/SLM-a-sliding-layer-merging-method.
>
---
#### [replaced 048] HCMA: Hierarchical Cross-model Alignment for Grounded Text-to-Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06512v3](http://arxiv.org/pdf/2505.06512v3)**

> **作者:** Hang Wang; Zhi-Qi Cheng; Chenhao Lin; Chao Shen; Lei Zhang
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Text-to-image synthesis has progressed to the point where models can generate visually compelling images from natural language prompts. Yet, existing methods often fail to reconcile high-level semantic fidelity with explicit spatial control, particularly in scenes involving multiple objects, nuanced relations, or complex layouts. To bridge this gap, we propose a Hierarchical Cross-Modal Alignment (HCMA) framework for grounded text-to-image generation. HCMA integrates two alignment modules into each diffusion sampling step: a global module that continuously aligns latent representations with textual descriptions to ensure scene-level coherence, and a local module that employs bounding-box layouts to anchor objects at specified locations, enabling fine-grained spatial control. Extensive experiments on the MS-COCO 2014 validation set show that HCMA surpasses state-of-the-art baselines, achieving a 0.69 improvement in Frechet Inception Distance (FID) and a 0.0295 gain in CLIP Score. These results demonstrate HCMA's effectiveness in faithfully capturing intricate textual semantics while adhering to user-defined spatial constraints, offering a robust solution for semantically grounded image generation. Our code is available at https://github.com/hwang-cs-ime/HCMA.
>
---
#### [replaced 049] Charm: The Missing Piece in ViT fine-tuning for Image Aesthetic Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02522v2](http://arxiv.org/pdf/2504.02522v2)**

> **作者:** Fatemeh Behrad; Tinne Tuytelaars; Johan Wagemans
>
> **备注:** CVPR 2025
>
> **摘要:** The capacity of Vision transformers (ViTs) to handle variable-sized inputs is often constrained by computational complexity and batch processing limitations. Consequently, ViTs are typically trained on small, fixed-size images obtained through downscaling or cropping. While reducing computational burden, these methods result in significant information loss, negatively affecting tasks like image aesthetic assessment. We introduce Charm, a novel tokenization approach that preserves Composition, High-resolution, Aspect Ratio, and Multi-scale information simultaneously. Charm prioritizes high-resolution details in specific regions while downscaling others, enabling shorter fixed-size input sequences for ViTs while incorporating essential information. Charm is designed to be compatible with pre-trained ViTs and their learned positional embeddings. By providing multiscale input and introducing variety to input tokens, Charm improves ViT performance and generalizability for image aesthetic assessment. We avoid cropping or changing the aspect ratio to further preserve information. Extensive experiments demonstrate significant performance improvements on various image aesthetic and quality assessment datasets (up to 8.1 %) using a lightweight ViT backbone. Code and pre-trained models are available at https://github.com/FBehrad/Charm.
>
---
#### [replaced 050] An unsupervised method for MRI recovery: Deep image prior with structured sparsity
- **分类: eess.IV; cs.CV; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2501.01482v2](http://arxiv.org/pdf/2501.01482v2)**

> **作者:** Muhammad Ahmad Sultan; Chong Chen; Yingmin Liu; Katarzyna Gil; Karolina Zareba; Rizwan Ahmad
>
> **备注:** Magn Reson Mater Phy (2025)
>
> **摘要:** Objective: To propose and validate an unsupervised MRI reconstruction method that does not require fully sampled k-space data. Materials and Methods: The proposed method, deep image prior with structured sparsity (DISCUS), extends the deep image prior (DIP) by introducing group sparsity to frame-specific code vectors, enabling the discovery of a low-dimensional manifold for capturing temporal variations. \discus was validated using four studies: (I) simulation of a dynamic Shepp-Logan phantom to demonstrate its manifold discovery capabilities, (II) comparison with compressed sensing and DIP-based methods using simulated single-shot late gadolinium enhancement (LGE) image series from six distinct digital cardiac phantoms in terms of normalized mean square error (NMSE) and structural similarity index measure (SSIM), (III) evaluation on retrospectively undersampled single-shot LGE data from eight patients, and (IV) evaluation on prospectively undersampled single-shot LGE data from eight patients, assessed via blind scoring from two expert readers. Results: DISCUS outperformed competing methods, demonstrating superior reconstruction quality in terms of NMSE and SSIM (Studies I--III) and expert reader scoring (Study IV). Discussion: An unsupervised image reconstruction method is presented and validated on simulated and measured data. These developments can benefit applications where acquiring fully sampled data is challenging.
>
---
#### [replaced 051] Pose Priors from Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.03689v2](http://arxiv.org/pdf/2405.03689v2)**

> **作者:** Sanjay Subramanian; Evonne Ng; Lea Müller; Dan Klein; Shiry Ginosar; Trevor Darrell
>
> **备注:** CVPR 2025
>
> **摘要:** Language is often used to describe physical interaction, yet most 3D human pose estimation methods overlook this rich source of information. We bridge this gap by leveraging large multimodal models (LMMs) as priors for reconstructing contact poses, offering a scalable alternative to traditional methods that rely on human annotations or motion capture data. Our approach extracts contact-relevant descriptors from an LMM and translates them into tractable losses to constrain 3D human pose optimization. Despite its simplicity, our method produces compelling reconstructions for both two-person interactions and self-contact scenarios, accurately capturing the semantics of physical and social interactions. Our results demonstrate that LMMs can serve as powerful tools for contact prediction and pose estimation, offering an alternative to costly manual human annotations or motion capture data. Our code is publicly available at https://prosepose.github.io.
>
---
