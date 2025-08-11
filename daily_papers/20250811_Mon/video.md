# 计算机视觉 cs.CV

- **最新发布 113 篇**

- **更新 63 篇**

## 最新发布

#### [new 001] InstantEdit: Text-Guided Few-Step Image Editing with Piecewise Rectified Flow
- **分类: cs.CV**

- **简介: 论文提出基于RectifiedFlow的快速文本引导少步图像编辑方法InstantEdit，解决如何在保持关键内容前提下高效遵循文本指令，通过PerRFI逆向策略、Inversion Latent Injection及Disentangled Prompt Guidance等创新实现高质量编辑。**

- **链接: [http://arxiv.org/pdf/2508.06033v1](http://arxiv.org/pdf/2508.06033v1)**

> **作者:** Yiming Gong; Zhen Zhu; Minjia Zhang
>
> **备注:** ICCV 2025
>
> **摘要:** We propose a fast text-guided image editing method called InstantEdit based on the RectifiedFlow framework, which is structured as a few-step editing process that preserves critical content while following closely to textual instructions. Our approach leverages the straight sampling trajectories of RectifiedFlow by introducing a specialized inversion strategy called PerRFI. To maintain consistent while editable results for RectifiedFlow model, we further propose a novel regeneration method, Inversion Latent Injection, which effectively reuses latent information obtained during inversion to facilitate more coherent and detailed regeneration. Additionally, we propose a Disentangled Prompt Guidance technique to balance editability with detail preservation, and integrate a Canny-conditioned ControlNet to incorporate structural cues and suppress artifacts. Evaluation on the PIE image editing dataset demonstrates that InstantEdit is not only fast but also achieves better qualitative and quantitative results compared to state-of-the-art few-step editing methods.
>
---
#### [new 002] ExploreGS: Explorable 3D Scene Reconstruction with Virtual Camera Samplings and Diffusion Priors
- **分类: cs.CV**

- **简介: 论文提出一种基于3DGS的三维场景重建方法，通过虚拟相机采样与扩散先验优化，解决非训练视角下的伪影和缺失区域问题，提升任意视角下的高质量渲染。**

- **链接: [http://arxiv.org/pdf/2508.06014v1](http://arxiv.org/pdf/2508.06014v1)**

> **作者:** Minsu Kim; Subin Jeon; In Cho; Mijin Yoo; Seon Joo Kim
>
> **备注:** 10 pages, 6 Figures, ICCV 2025
>
> **摘要:** Recent advances in novel view synthesis (NVS) have enabled real-time rendering with 3D Gaussian Splatting (3DGS). However, existing methods struggle with artifacts and missing regions when rendering from viewpoints that deviate from the training trajectory, limiting seamless scene exploration. To address this, we propose a 3DGS-based pipeline that generates additional training views to enhance reconstruction. We introduce an information-gain-driven virtual camera placement strategy to maximize scene coverage, followed by video diffusion priors to refine rendered results. Fine-tuning 3D Gaussians with these enhanced views significantly improves reconstruction quality. To evaluate our method, we present Wild-Explore, a benchmark designed for challenging scene exploration. Experiments demonstrate that our approach outperforms existing 3DGS-based methods, enabling high-quality, artifact-free rendering from arbitrary viewpoints. https://exploregs.github.io
>
---
#### [new 003] Roll Your Eyes: Gaze Redirection via Explicit 3D Eyeball Rotation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于3D Gaussian Splatting的三维眼球旋转框架，实现高保真眼神引导，通过显式建模眼球结构与自适应变形模块提升图像质量与精度。**

- **链接: [http://arxiv.org/pdf/2508.06136v1](http://arxiv.org/pdf/2508.06136v1)**

> **作者:** YoungChan Choi; HengFei Wang; YiHua Cheng; Boeun Kim; Hyung Jin Chang; YoungGeun Choi; Sang-Il Choi
>
> **备注:** 9 pages, 5 figures, ACM Multimeida 2025 accepted
>
> **摘要:** We propose a novel 3D gaze redirection framework that leverages an explicit 3D eyeball structure. Existing gaze redirection methods are typically based on neural radiance fields, which employ implicit neural representations via volume rendering. Unlike these NeRF-based approaches, where the rotation and translation of 3D representations are not explicitly modeled, we introduce a dedicated 3D eyeball structure to represent the eyeballs with 3D Gaussian Splatting (3DGS). Our method generates photorealistic images that faithfully reproduce the desired gaze direction by explicitly rotating and translating the 3D eyeball structure. In addition, we propose an adaptive deformation module that enables the replication of subtle muscle movements around the eyes. Through experiments conducted on the ETH-XGaze dataset, we demonstrate that our framework is capable of generating diverse novel gaze images, achieving superior image quality and gaze estimation accuracy compared to previous state-of-the-art methods.
>
---
#### [new 004] An Interpretable Multi-Plane Fusion Framework With Kolmogorov-Arnold Network Guided Attention Enhancement for Alzheimer's Disease Diagnosis
- **分类: cs.CV**

- **简介: 本文提出一种可解释的多平面融合框架MPF-KANSC，通过整合冠状、矢状和轴位MRI特征，并利用Kolmogorov-Arnold网络指导的注意力机制，提升阿尔茨海默病早期诊断精度，揭示亚脑区对称性变化。**

- **链接: [http://arxiv.org/pdf/2508.06157v1](http://arxiv.org/pdf/2508.06157v1)**

> **作者:** Xiaoxiao Yang; Meiliang Liu; Yunfang Xu; Zijin Li; Zhengye Si; Xinyue Yang; Zhiwen Zhao
>
> **摘要:** Alzheimer's disease (AD) is a progressive neurodegenerative disorder that severely impairs cognitive function and quality of life. Timely intervention in AD relies heavily on early and precise diagnosis, which remains challenging due to the complex and subtle structural changes in the brain. Most existing deep learning methods focus only on a single plane of structural magnetic resonance imaging (sMRI) and struggle to accurately capture the complex and nonlinear relationships among pathological regions of the brain, thus limiting their ability to precisely identify atrophic features. To overcome these limitations, we propose an innovative framework, MPF-KANSC, which integrates multi-plane fusion (MPF) for combining features from the coronal, sagittal, and axial planes, and a Kolmogorov-Arnold Network-guided spatial-channel attention mechanism (KANSC) to more effectively learn and represent sMRI atrophy features. Specifically, the proposed model enables parallel feature extraction from multiple anatomical planes, thus capturing more comprehensive structural information. The KANSC attention mechanism further leverages a more flexible and accurate nonlinear function approximation technique, facilitating precise identification and localization of disease-related abnormalities. Experiments on the ADNI dataset confirm that the proposed MPF-KANSC achieves superior performance in AD diagnosis. Moreover, our findings provide new evidence of right-lateralized asymmetry in subcortical structural changes during AD progression, highlighting the model's promising interpretability.
>
---
#### [new 005] Temporal Cluster Assignment for Efficient Real-Time Video Segmentation
- **分类: cs.CV**

- **简介: 论文提出基于时间冗余的Temporal Cluster Assignment（TCA）方法，解决视频分割中窗口注意力导致的高计算成本问题，通过利用帧间一致性优化token聚类，提升实时分割效率与精度。**

- **链接: [http://arxiv.org/pdf/2508.05851v1](http://arxiv.org/pdf/2508.05851v1)**

> **作者:** Ka-Wai Yung; Felix J. S. Bragman; Jialang Xu; Imanol Luengo; Danail Stoyanov; Evangelos B. Mazomenos
>
> **摘要:** Vision Transformers have substantially advanced the capabilities of segmentation models across both image and video domains. Among them, the Swin Transformer stands out for its ability to capture hierarchical, multi-scale representations, making it a popular backbone for segmentation in videos. However, despite its window-attention scheme, it still incurs a high computational cost, especially in larger variants commonly used for dense prediction in videos. This remains a major bottleneck for real-time, resource-constrained applications. Whilst token reduction methods have been proposed to alleviate this, the window-based attention mechanism of Swin requires a fixed number of tokens per window, limiting the applicability of conventional pruning techniques. Meanwhile, training-free token clustering approaches have shown promise in image segmentation while maintaining window consistency. Nevertheless, they fail to exploit temporal redundancy, missing a key opportunity to further optimize video segmentation performance. We introduce Temporal Cluster Assignment (TCA), a lightweight and effective, fine-tuning-free strategy that enhances token clustering by leveraging temporal coherence across frames. Instead of indiscriminately dropping redundant tokens, TCA refines token clusters using temporal correlations, thereby retaining fine-grained details while significantly reducing computation. Extensive evaluations on YouTube-VIS 2019, YouTube-VIS 2021, OVIS, and a private surgical video dataset show that TCA consistently boosts the accuracy-speed trade-off of existing clustering-based methods. Our results demonstrate that TCA generalizes competently across both natural and domain-specific videos.
>
---
#### [new 006] Deepfake Detection that Generalizes Across Benchmarks
- **分类: cs.CV**

- **简介: 论文提出基于CLIP模型的深度伪造检测方法，通过参数高效微调和特征约束提升跨基准泛化能力，解决现有检测器对新技术泛化差的问题。**

- **链接: [http://arxiv.org/pdf/2508.06248v1](http://arxiv.org/pdf/2508.06248v1)**

> **作者:** Andrii Yermakov; Jan Cech; Jiri Matas; Mario Fritz
>
> **摘要:** The generalization of deepfake detectors to unseen manipulation techniques remains a challenge for practical deployment. Although many approaches adapt foundation models by introducing significant architectural complexity, this work demonstrates that robust generalization is achievable through a parameter-efficient adaptation of a pre-trained CLIP vision encoder. The proposed method, LNCLIP-DF, fine-tunes only the Layer Normalization parameters (0.03% of the total) and enhances generalization by enforcing a hyperspherical feature manifold using L2 normalization and latent space augmentations. We conducted an extensive evaluation on 13 benchmark datasets spanning from 2019 to 2025. The proposed method achieves state-of-the-art performance, outperforming more complex, recent approaches in average cross-dataset AUROC. Our analysis yields two primary findings for the field: 1) training on paired real-fake data from the same source video is essential for mitigating shortcut learning and improving generalization, and 2) detection difficulty on academic datasets has not strictly increased over time, with models trained on older, diverse datasets showing strong generalization capabilities. This work delivers a computationally efficient and reproducible method, proving that state-of-the-art generalization is attainable by making targeted, minimal changes to a pre-trained CLIP model. The code will be made publicly available upon acceptance.
>
---
#### [new 007] Towards Unified Image Deblurring using a Mixture-of-Experts Decoder
- **分类: cs.CV**

- **简介: 该论文提出一种统一图像去模糊方法，通过混合专家解码器动态适应不同模糊类型，解决传统方法泛化不足的问题，实现高效精准恢复。**

- **链接: [http://arxiv.org/pdf/2508.06228v1](http://arxiv.org/pdf/2508.06228v1)**

> **作者:** Daniel Feijoo; Paula Garrido-Mellado; Jaesung Rim; Alvaro Garcia; Marcos V. Conde
>
> **备注:** Preprint. Under review
>
> **摘要:** Image deblurring, removing blurring artifacts from images, is a fundamental task in computational photography and low-level computer vision. Existing approaches focus on specialized solutions tailored to particular blur types, thus, these solutions lack generalization. This limitation in current methods implies requiring multiple models to cover several blur types, which is not practical in many real scenarios. In this paper, we introduce the first all-in-one deblurring method capable of efficiently restoring images affected by diverse blur degradations, including global motion, local motion, blur in low-light conditions, and defocus blur. We propose a mixture-of-experts (MoE) decoding module, which dynamically routes image features based on the recognized blur degradation, enabling precise and efficient restoration in an end-to-end manner. Our unified approach not only achieves performance comparable to dedicated task-specific models, but also demonstrates remarkable robustness and generalization capabilities on unseen blur degradation scenarios.
>
---
#### [new 008] EvoMakeup: High-Fidelity and Controllable Makeup Editing with MakeupQuad
- **分类: cs.CV**

- **简介: 论文提出EvoMakeup框架，解决面部化妆编辑中细节粗糙、身份保真度不足的问题，通过MakeupQuad数据集与多阶段蒸馏提升模型质量，实现高保真、可控的多任务化妆编辑。**

- **链接: [http://arxiv.org/pdf/2508.05994v1](http://arxiv.org/pdf/2508.05994v1)**

> **作者:** Huadong Wu; Yi Fu; Yunhao Li; Yuan Gao; Kang Du
>
> **摘要:** Facial makeup editing aims to realistically transfer makeup from a reference to a target face. Existing methods often produce low-quality results with coarse makeup details and struggle to preserve both identity and makeup fidelity, mainly due to the lack of structured paired data -- where source and result share identity, and reference and result share identical makeup. To address this, we introduce MakeupQuad, a large-scale, high-quality dataset with non-makeup faces, references, edited results, and textual makeup descriptions. Building on this, we propose EvoMakeup, a unified training framework that mitigates image degradation during multi-stage distillation, enabling iterative improvement of both data and model quality. Although trained solely on synthetic data, EvoMakeup generalizes well and outperforms prior methods on real-world benchmarks. It supports high-fidelity, controllable, multi-task makeup editing -- including full-face and partial reference-based editing, as well as text-driven makeup editing -- within a single model. Experimental results demonstrate that our method achieves superior makeup fidelity and identity preservation, effectively balancing both aspects. Code and dataset will be released upon acceptance.
>
---
#### [new 009] SAM Encoder Breach by Adversarial Simplicial Complex Triggers Downstream Model Failures
- **分类: cs.CV; I.4.9**

- **简介: 论文提出VeSCA方法，通过分析SAM编码器的单纯形复杂度识别共享脆弱区域，生成可转移对抗样本以评估下游模型风险，提升12.7%性能。任务为对抗攻击防御，解决SAM漏洞导致下游模型失效问题。**

- **链接: [http://arxiv.org/pdf/2508.06127v1](http://arxiv.org/pdf/2508.06127v1)**

> **作者:** Yi Qin; Rui Wang; Tao Huang; Tong Xiao; Liping Jing
>
> **备注:** 8 pages,recived by ICCV2025
>
> **摘要:** While the Segment Anything Model (SAM) transforms interactive segmentation with zero-shot abilities, its inherent vulnerabilities present a single-point risk, potentially leading to the failure of numerous downstream applications. Proactively evaluating these transferable vulnerabilities is thus imperative. Prior adversarial attacks on SAM often present limited transferability due to insufficient exploration of common weakness across domains. To address this, we propose Vertex-Refining Simplicial Complex Attack (VeSCA), a novel method that leverages only the encoder of SAM for generating transferable adversarial examples. Specifically, it achieves this by explicitly characterizing the shared vulnerable regions between SAM and downstream models through a parametric simplicial complex. Our goal is to identify such complexes within adversarially potent regions by iterative vertex-wise refinement. A lightweight domain re-adaptation strategy is introduced to bridge domain divergence using minimal reference data during the initialization of simplicial complex. Ultimately, VeSCA generates consistently transferable adversarial examples through random simplicial complex sampling. Extensive experiments demonstrate that VeSCA achieves performance improved by 12.7% compared to state-of-the-art methods across three downstream model categories across five domain-specific datasets. Our findings further highlight the downstream model risks posed by SAM's vulnerabilities and emphasize the urgency of developing more robust foundation models.
>
---
#### [new 010] Depth Jitter: Seeing through the Depth
- **分类: cs.CV; cs.RO**

- **简介: 论文提出Depth-Jitter技术，通过自适应深度偏移模拟自然深度变化，提升模型在真实深度条件下的稳定性和泛化能力，解决传统方法忽略深度感知导致的鲁棒性不足问题。**

- **链接: [http://arxiv.org/pdf/2508.06227v1](http://arxiv.org/pdf/2508.06227v1)**

> **作者:** Md Sazidur Rahman; David Cabecinhas; Ricard Marxer
>
> **摘要:** Depth information is essential in computer vision, particularly in underwater imaging, robotics, and autonomous navigation. However, conventional augmentation techniques overlook depth aware transformations, limiting model robustness in real world depth variations. In this paper, we introduce Depth-Jitter, a novel depth-based augmentation technique that simulates natural depth variations to improve generalization. Our approach applies adaptive depth offsetting, guided by depth variance thresholds, to generate synthetic depth perturbations while preserving structural integrity. We evaluate Depth-Jitter on two benchmark datasets, FathomNet and UTDAC2020 demonstrating its impact on model stability under diverse depth conditions. Extensive experiments compare Depth-Jitter against traditional augmentation strategies such as ColorJitter, analyzing performance across varying learning rates, encoders, and loss functions. While Depth-Jitter does not always outperform conventional methods in absolute performance, it consistently enhances model stability and generalization in depth-sensitive environments. These findings highlight the potential of depth-aware augmentation for real-world applications and provide a foundation for further research into depth-based learning strategies. The proposed technique is publicly available to support advancements in depth-aware augmentation. The code is publicly available on \href{https://github.com/mim-team/Depth-Jitter}{github}.
>
---
#### [new 011] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 论文提出首个多模态大语言模型安全评估框架SDEval，通过文本、图像及文本-图像动态生成新样本，解决现有基准数据过时与污染问题，提升安全评估效果。**

- **链接: [http://arxiv.org/pdf/2508.06142v1](http://arxiv.org/pdf/2508.06142v1)**

> **作者:** Hanqing Wang; Yuan Tian; Mingyu Liu; Zhenhao Zhang; Xiangyang Zhu
>
> **摘要:** In the rapidly evolving landscape of Multimodal Large Language Models (MLLMs), the safety concerns of their outputs have earned significant attention. Although numerous datasets have been proposed, they may become outdated with MLLM advancements and are susceptible to data contamination issues. To address these problems, we propose \textbf{SDEval}, the \textit{first} safety dynamic evaluation framework to controllably adjust the distribution and complexity of safety benchmarks. Specifically, SDEval mainly adopts three dynamic strategies: text, image, and text-image dynamics to generate new samples from original benchmarks. We first explore the individual effects of text and image dynamics on model safety. Then, we find that injecting text dynamics into images can further impact safety, and conversely, injecting image dynamics into text also leads to safety risks. SDEval is general enough to be applied to various existing safety and even capability benchmarks. Experiments across safety benchmarks, MLLMGuard and VLSBench, and capability benchmarks, MMBench and MMVet, show that SDEval significantly influences safety evaluation, mitigates data contamination, and exposes safety limitations of MLLMs. Code is available at https://github.com/hq-King/SDEval
>
---
#### [new 012] SIFThinker: Spatially-Aware Image Focus for Visual Reasoning
- **分类: cs.CV; cs.AI; I.2.10**

- **简介: 该论文提出SIFThinker框架，针对多模态大语言模型在复杂视觉任务中的空间理解与细粒度感知不足问题，通过结合空间线索与自然语言，设计逆向扩展策略生成图像文本链，并引入GRPO-SIF强化训练，提升模型对视觉焦点的动态修正能力。**

- **链接: [http://arxiv.org/pdf/2508.06259v1](http://arxiv.org/pdf/2508.06259v1)**

> **作者:** Zhangquan Chen; Ruihui Zhao; Chuwei Luo; Mingze Sun; Xinlei Yu; Yangyang Kang; Ruqi Huang
>
> **备注:** 15 pages, 13 figures
>
> **摘要:** Current multimodal large language models (MLLMs) still face significant challenges in complex visual tasks (e.g., spatial understanding, fine-grained perception). Prior methods have tried to incorporate visual reasoning, however, they fail to leverage attention correction with spatial cues to iteratively refine their focus on prompt-relevant regions. In this paper, we introduce SIFThinker, a spatially-aware "think-with-images" framework that mimics human visual perception. Specifically, SIFThinker enables attention correcting and image region focusing by interleaving depth-enhanced bounding boxes and natural language. Our contributions are twofold: First, we introduce a reverse-expansion-forward-inference strategy that facilitates the generation of interleaved image-text chains of thought for process-level supervision, which in turn leads to the construction of the SIF-50K dataset. Besides, we propose GRPO-SIF, a reinforced training paradigm that integrates depth-informed visual grounding into a unified reasoning pipeline, teaching the model to dynamically correct and focus on prompt-relevant regions. Extensive experiments demonstrate that SIFThinker outperforms state-of-the-art methods in spatial understanding and fine-grained visual perception, while maintaining strong general capabilities, highlighting the effectiveness of our method.
>
---
#### [new 013] Boosting Adversarial Transferability via Residual Perturbation Attack
- **分类: cs.CV; cs.CR; cs.LG**

- **简介: 论文提出Residual Perturbation Attack（ResPA）方法，通过残差梯度引导对抗样本向损失函数平坦区域转移，提升对抗样本的转移性。任务为提升对抗样本转移能力，解决传统方法忽视扰动方向影响的问题，工作包括动态参考梯度计算与全局方向捕捉。**

- **链接: [http://arxiv.org/pdf/2508.05689v1](http://arxiv.org/pdf/2508.05689v1)**

> **作者:** Jinjia Peng; Zeze Tao; Huibing Wang; Meng Wang; Yang Wang
>
> **备注:** Accepted to ieee/cvf international conference on computer vision (ICCV2025)
>
> **摘要:** Deep neural networks are susceptible to adversarial examples while suffering from incorrect predictions via imperceptible perturbations. Transfer-based attacks create adversarial examples for surrogate models and transfer these examples to target models under black-box scenarios. Recent studies reveal that adversarial examples in flat loss landscapes exhibit superior transferability to alleviate overfitting on surrogate models. However, the prior arts overlook the influence of perturbation directions, resulting in limited transferability. In this paper, we propose a novel attack method, named Residual Perturbation Attack (ResPA), relying on the residual gradient as the perturbation direction to guide the adversarial examples toward the flat regions of the loss function. Specifically, ResPA conducts an exponential moving average on the input gradients to obtain the first moment as the reference gradient, which encompasses the direction of historical gradients. Instead of heavily relying on the local flatness that stems from the current gradients as the perturbation direction, ResPA further considers the residual between the current gradient and the reference gradient to capture the changes in the global perturbation direction. The experimental results demonstrate the better transferability of ResPA than the existing typical transfer-based attack methods, while the transferability can be further improved by combining ResPA with the current input transformation methods. The code is available at https://github.com/ZezeTao/ResPA.
>
---
#### [new 014] Synthetic Data-Driven Multi-Architecture Framework for Automated Polyp Segmentation Through Integrated Detection and Mask Generation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于合成数据的多架构框架，用于自动分割肠息肉，解决数据不足和标注复杂问题，结合Faster R-CNN与SAM实现检测与分割，并评估FPN和U-Net等模型性能。**

- **链接: [http://arxiv.org/pdf/2508.06170v1](http://arxiv.org/pdf/2508.06170v1)**

> **作者:** Ojonugwa Oluwafemi Ejiga Peter; Akingbola Oluwapemiisin; Amalahu Chetachi; Adeniran Opeyemi; Fahmi Khalifa; Md Mahmudur Rahman
>
> **摘要:** Colonoscopy is a vital tool for the early diagnosis of colorectal cancer, which is one of the main causes of cancer-related mortality globally; hence, it is deemed an essential technique for the prevention and early detection of colorectal cancer. The research introduces a unique multidirectional architectural framework to automate polyp detection within colonoscopy images while helping resolve limited healthcare dataset sizes and annotation complexities. The research implements a comprehensive system that delivers synthetic data generation through Stable Diffusion enhancements together with detection and segmentation algorithms. This detection approach combines Faster R-CNN for initial object localization while the Segment Anything Model (SAM) refines the segmentation masks. The faster R-CNN detection algorithm achieved a recall of 93.08% combined with a precision of 88.97% and an F1 score of 90.98%.SAM is then used to generate the image mask. The research evaluated five state-of-the-art segmentation models that included U-Net, PSPNet, FPN, LinkNet, and MANet using ResNet34 as a base model. The results demonstrate the superior performance of FPN with the highest scores of PSNR (7.205893) and SSIM (0.492381), while UNet excels in recall (84.85%) and LinkNet shows balanced performance in IoU (64.20%) and Dice score (77.53%).
>
---
#### [new 015] Distribution-Specific Learning for Joint Salient and Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 本文提出一种联合检测显著与伪装对象的方法，解决传统联合学习导致的性能下降问题。通过引入任务特定参数解耦分布特性，设计SBSS采样策略平衡训练集，训练JoNet网络实现同时检测显著与伪装目标。**

- **链接: [http://arxiv.org/pdf/2508.06063v1](http://arxiv.org/pdf/2508.06063v1)**

> **作者:** Chao Hao; Zitong Yu; Xin Liu; Yuhao Wang; Weicheng Xie; Jingang Shi; Huanjing Yue; Jingyu Yang
>
> **摘要:** Salient object detection (SOD) and camouflaged object detection (COD) are two closely related but distinct computer vision tasks. Although both are class-agnostic segmentation tasks that map from RGB space to binary space, the former aims to identify the most salient objects in the image, while the latter focuses on detecting perfectly camouflaged objects that blend into the background in the image. These two tasks exhibit strong contradictory attributes. Previous works have mostly believed that joint learning of these two tasks would confuse the network, reducing its performance on both tasks. However, here we present an opposite perspective: with the correct approach to learning, the network can simultaneously possess the capability to find both salient and camouflaged objects, allowing both tasks to benefit from joint learning. We propose SCJoint, a joint learning scheme for SOD and COD tasks, assuming that the decoding processes of SOD and COD have different distribution characteristics. The key to our method is to learn the respective means and variances of the decoding processes for both tasks by inserting a minimal amount of task-specific learnable parameters within a fully shared network structure, thereby decoupling the contradictory attributes of the two tasks at a minimal cost. Furthermore, we propose a saliency-based sampling strategy (SBSS) to sample the training set of the SOD task to balance the training set sizes of the two tasks. In addition, SBSS improves the training set quality and shortens the training time. Based on the proposed SCJoint and SBSS, we train a powerful generalist network, named JoNet, which has the ability to simultaneously capture both ``salient" and ``camouflaged". Extensive experiments demonstrate the competitive performance and effectiveness of our proposed method. The code is available at https://github.com/linuxsino/JoNet.
>
---
#### [new 016] LoRA in LoRA: Towards Parameter-Efficient Architecture Expansion for Continual Visual Instruction Tuning
- **分类: cs.CV; cs.AI**

- **简介: 论文提出LiLoRA方法，针对CVIT任务解决灾难性遗忘与参数膨胀问题，通过共享LoRA矩阵和低秩分解提升参数效率。**

- **链接: [http://arxiv.org/pdf/2508.06202v1](http://arxiv.org/pdf/2508.06202v1)**

> **作者:** Chang Che; Ziqi Wang; Pengwan Yang; Qi Wang; Hui Ma; Zenglin Shi
>
> **摘要:** Continual Visual Instruction Tuning (CVIT) enables Multimodal Large Language Models (MLLMs) to incrementally learn new tasks over time. However, this process is challenged by catastrophic forgetting, where performance on previously learned tasks deteriorates as the model adapts to new ones. A common approach to mitigate forgetting is architecture expansion, which introduces task-specific modules to prevent interference. Yet, existing methods often expand entire layers for each task, leading to significant parameter overhead and poor scalability. To overcome these issues, we introduce LoRA in LoRA (LiLoRA), a highly efficient architecture expansion method tailored for CVIT in MLLMs. LiLoRA shares the LoRA matrix A across tasks to reduce redundancy, applies an additional low-rank decomposition to matrix B to minimize task-specific parameters, and incorporates a cosine-regularized stability loss to preserve consistency in shared representations over time. Extensive experiments on a diverse CVIT benchmark show that LiLoRA consistently achieves superior performance in sequential task learning while significantly improving parameter efficiency compared to existing approaches.
>
---
#### [new 017] NEP: Autoregressive Image Editing via Next Editing Token Prediction
- **分类: cs.CV**

- **简介: 论文提出基于自回归的Next Editing Token Prediction（NEP）方法，解决文本引导图像编辑中生成全图导致的计算成本高和非编辑区域污染问题，通过预测特定区域的编辑token实现精准修复。**

- **链接: [http://arxiv.org/pdf/2508.06044v1](http://arxiv.org/pdf/2508.06044v1)**

> **作者:** Huimin Wu; Xiaojian Ma; Haozhe Zhao; Yanpeng Zhao; Qing Li
>
> **备注:** The project page is: https://nep-bigai.github.io/
>
> **摘要:** Text-guided image editing involves modifying a source image based on a language instruction and, typically, requires changes to only small local regions. However, existing approaches generate the entire target image rather than selectively regenerate only the intended editing areas. This results in (1) unnecessary computational costs and (2) a bias toward reconstructing non-editing regions, which compromises the quality of the intended edits. To resolve these limitations, we propose to formulate image editing as Next Editing-token Prediction (NEP) based on autoregressive image generation, where only regions that need to be edited are regenerated, thus avoiding unintended modification to the non-editing areas. To enable any-region editing, we propose to pre-train an any-order autoregressive text-to-image (T2I) model. Once trained, it is capable of zero-shot image editing and can be easily adapted to NEP for image editing, which achieves a new state-of-the-art on widely used image editing benchmarks. Moreover, our model naturally supports test-time scaling (TTS) through iteratively refining its generation in a zero-shot manner. The project page is: https://nep-bigai.github.io/
>
---
#### [new 018] PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 论文提出PASG框架，解决机器人操作中高阶语义与低阶几何特征碎片化问题，通过几何特征聚合自动提取基本体并结合VLM实现语义锚定，建立空间语义推理基准，实现细粒度语义-具身理解。**

- **链接: [http://arxiv.org/pdf/2508.05976v1](http://arxiv.org/pdf/2508.05976v1)**

> **作者:** Zhihao Zhu; Yifan Zheng; Siyu Pan; Yaohui Jin; Yao Mu
>
> **备注:** Accepted to ICCV 2025. 8 pages main paper, 8 figures, plus supplementary material
>
> **摘要:** The fragmentation between high-level task semantics and low-level geometric features remains a persistent challenge in robotic manipulation. While vision-language models (VLMs) have shown promise in generating affordance-aware visual representations, the lack of semantic grounding in canonical spaces and reliance on manual annotations severely limit their ability to capture dynamic semantic-affordance relationships. To address these, we propose Primitive-Aware Semantic Grounding (PASG), a closed-loop framework that introduces: (1) Automatic primitive extraction through geometric feature aggregation, enabling cross-category detection of keypoints and axes; (2) VLM-driven semantic anchoring that dynamically couples geometric primitives with functional affordances and task-relevant description; (3) A spatial-semantic reasoning benchmark and a fine-tuned VLM (Qwen2.5VL-PA). We demonstrate PASG's effectiveness in practical robotic manipulation tasks across diverse scenarios, achieving performance comparable to manual annotations. PASG achieves a finer-grained semantic-affordance understanding of objects, establishing a unified paradigm for bridging geometric primitives with task semantics in robotic manipulation.
>
---
#### [new 019] FedX: Explanation-Guided Pruning for Communication-Efficient Federated Learning in Remote Sensing
- **分类: cs.CV**

- **简介: 论文提出FedX方法，通过解释引导剪枝减少联邦学习中模型通信开销，适用于遥感图像分类任务，提升全局模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.06256v1](http://arxiv.org/pdf/2508.06256v1)**

> **作者:** Barış Büyüktaş; Jonas Klotz; Begüm Demir
>
> **摘要:** Federated learning (FL) enables the collaborative training of deep neural networks across decentralized data archives (i.e., clients), where each client stores data locally and only shares model updates with a central server. This makes FL a suitable learning paradigm for remote sensing (RS) image classification tasks, where data centralization may be restricted due to legal and privacy constraints. However, a key challenge in applying FL to RS tasks is the communication overhead caused by the frequent exchange of large model updates between clients and the central server. To address this issue, in this paper we propose a novel strategy (denoted as FedX) that uses explanation-guided pruning to reduce communication overhead by minimizing the size of the transmitted models without compromising performance. FedX leverages backpropagation-based explanation methods to estimate the task-specific importance of model components and prunes the least relevant ones at the central server. The resulting sparse global model is then sent to clients, substantially reducing communication overhead. We evaluate FedX on multi-label scene classification using the BigEarthNet-S2 dataset and single-label scene classification using the EuroSAT dataset. Experimental results show the success of FedX in significantly reducing the number of shared model parameters while enhancing the generalization capability of the global model, compared to both unpruned model and state-of-the-art pruning methods. The code of FedX will be available at https://git.tu-berlin.de/rsim/FedX.
>
---
#### [new 020] Optimization-Free Style Transfer for 3D Gaussian Splats
- **分类: cs.CV**

- **简介: 论文提出无需训练或优化的3D Gaussian splats风格迁移方法，通过构建图结构和表面建模实现风格化，支持任意风格图像应用，速度达2分钟/场景，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.05813v1](http://arxiv.org/pdf/2508.05813v1)**

> **作者:** Raphael Du Sablon; David Hart
>
> **摘要:** The task of style transfer for 3D Gaussian splats has been explored in many previous works, but these require reconstructing or fine-tuning the splat while incorporating style information or optimizing a feature extraction network on the splat representation. We propose a reconstruction- and optimization-free approach to stylizing 3D Gaussian splats. This is done by generating a graph structure across the implicit surface of the splat representation. A feed-forward, surface-based stylization method is then used and interpolated back to the individual splats in the scene. This allows for any style image and 3D Gaussian splat to be used without any additional training or optimization. This also allows for fast stylization of splats, achieving speeds under 2 minutes even on consumer-grade hardware. We demonstrate the quality results this approach achieves and compare to other 3D Gaussian splat style transfer methods. Code is publicly available at https://github.com/davidmhart/FastSplatStyler.
>
---
#### [new 021] HOLODECK 2.0: Vision-Language-Guided 3D World Generation with Editing
- **分类: cs.CV; cs.GR**

- **简介: 论文提出HOLODECK 2.0框架，通过视觉-语言模型生成3D场景，解决传统方法依赖人工、缺乏编辑能力的问题，支持多风格生成与交互编辑，提升开放场景生成效率。**

- **链接: [http://arxiv.org/pdf/2508.05899v1](http://arxiv.org/pdf/2508.05899v1)**

> **作者:** Zixuan Bian; Ruohan Ren; Yue Yang; Chris Callison-Burch
>
> **摘要:** 3D scene generation plays a crucial role in gaming, artistic creation, virtual reality and many other domains. However, current 3D scene design still relies heavily on extensive manual effort from creators, and existing automated methods struggle to generate open-domain scenes or support flexible editing. As a result, generating 3D worlds directly from text has garnered increasing attention. In this paper, we introduce HOLODECK 2.0, an advanced vision-language-guided framework for 3D world generation with support for interactive scene editing based on human feedback. HOLODECK 2.0 can generate diverse and stylistically rich 3D scenes (e.g., realistic, cartoon, anime, and cyberpunk styles) that exhibit high semantic fidelity to fine-grained input descriptions, suitable for both indoor and open-domain environments. HOLODECK 2.0 leverages vision-language models (VLMs) to identify and parse the objects required in a scene and generates corresponding high-quality assets via state-of-the-art 3D generative models. It then iteratively applies spatial constraints derived from the VLMs to achieve semantically coherent and physically plausible layouts. Human evaluations and CLIP-based assessments demonstrate that HOLODECK 2.0 effectively generates high-quality scenes closely aligned with detailed textual descriptions, consistently outperforming baselines across indoor and open-domain scenarios. Additionally, we provide editing capabilities that flexibly adapt to human feedback, supporting layout refinement and style-consistent object edits. Finally, we present a practical application of HOLODECK 2.0 in procedural game modeling, generating visually rich and immersive environments, potentially boosting efficiency.
>
---
#### [new 022] ECMF: Enhanced Cross-Modal Fusion for Multimodal Emotion Recognition in MER-SEMI Challenge
- **分类: cs.CV; cs.AI; cs.CY**

- **简介: 论文提出基于多模态特征融合的EMR-SEMI挑战解决方案，通过预训练模型提取视觉、文本等模态特征，结合自注意力机制与残差连接实现动态权重分配，优化噪声标签处理，取得87.49%的F-score成绩。**

- **链接: [http://arxiv.org/pdf/2508.05991v1](http://arxiv.org/pdf/2508.05991v1)**

> **作者:** Juewen Hu; Yexin Li; Jiulin Li; Shuo Chen; Pring Wong
>
> **摘要:** Emotion recognition plays a vital role in enhancing human-computer interaction. In this study, we tackle the MER-SEMI challenge of the MER2025 competition by proposing a novel multimodal emotion recognition framework. To address the issue of data scarcity, we leverage large-scale pre-trained models to extract informative features from visual, audio, and textual modalities. Specifically, for the visual modality, we design a dual-branch visual encoder that captures both global frame-level features and localized facial representations. For the textual modality, we introduce a context-enriched method that employs large language models to enrich emotional cues within the input text. To effectively integrate these multimodal features, we propose a fusion strategy comprising two key components, i.e., self-attention mechanisms for dynamic modality weighting, and residual connections to preserve original representations. Beyond architectural design, we further refine noisy labels in the training set by a multi-source labeling strategy. Our approach achieves a substantial performance improvement over the official baseline on the MER2025-SEMI dataset, attaining a weighted F-score of 87.49% compared to 78.63%, thereby validating the effectiveness of the proposed framework.
>
---
#### [new 023] WGAST: Weakly-Supervised Generative Network for Daily 10 m Land Surface Temperature Estimation via Spatio-Temporal Fusion
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出基于弱监督生成网络（WGAST）的每日10米地表温度（LST）估计方法，通过时空融合解决遥感数据分辨率矛盾，提升精度并抗云遮挡，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.06485v1](http://arxiv.org/pdf/2508.06485v1)**

> **作者:** Sofiane Bouaziz; Adel Hafiane; Raphael Canals; Rachid Nedjai
>
> **备注:** Submitted to IEEE Transactions on Geoscience and Remote Sensing (TGRS)
>
> **摘要:** Urbanization, climate change, and agricultural stress are increasing the demand for precise and timely environmental monitoring. Land Surface Temperature (LST) is a key variable in this context and is retrieved from remote sensing satellites. However, these systems face a trade-off between spatial and temporal resolution. While spatio-temporal fusion methods offer promising solutions, few have addressed the estimation of daily LST at 10 m resolution. In this study, we present WGAST, a Weakly-Supervised Generative Network for Daily 10 m LST Estimation via Spatio-Temporal Fusion of Terra MODIS, Landsat 8, and Sentinel-2. WGAST is the first end-to-end deep learning framework designed for this task. It adopts a conditional generative adversarial architecture, with a generator composed of four stages: feature extraction, fusion, LST reconstruction, and noise suppression. The first stage employs a set of encoders to extract multi-level latent representations from the inputs, which are then fused in the second stage using cosine similarity, normalization, and temporal attention mechanisms. The third stage decodes the fused features into high-resolution LST, followed by a Gaussian filter to suppress high-frequency noise. Training follows a weakly supervised strategy based on physical averaging principles and reinforced by a PatchGAN discriminator. Experiments demonstrate that WGAST outperforms existing methods in both quantitative and qualitative evaluations. Compared to the best-performing baseline, on average, WGAST reduces RMSE by 17.18% and improves SSIM by 11.00%. Furthermore, WGAST is robust to cloud-induced LST and effectively captures fine-scale thermal patterns, as validated against 33 ground-based sensors. The code is available at https://github.com/Sofianebouaziz1/WGAST.git.
>
---
#### [new 024] Towards MR-Based Trochleoplasty Planning
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于MRI的 trochleoplasty 规划方法，通过超分辨率重建和Wavelet Diffusion Model生成高精度3D形态，无需CT，提升手术效果。**

- **链接: [http://arxiv.org/pdf/2508.06076v1](http://arxiv.org/pdf/2508.06076v1)**

> **作者:** Michael Wehrli; Alicia Durrer; Paul Friedrich; Sidaty El Hadramy; Edwin Li; Luana Brahaj; Carol C. Hasler; Philippe C. Cattin
>
> **备注:** Accepted at MICCAI COLAS Workshop 2025. Code: https://wehrlimi.github.io/sr-3d-planning/
>
> **摘要:** To treat Trochlear Dysplasia (TD), current approaches rely mainly on low-resolution clinical Magnetic Resonance (MR) scans and surgical intuition. The surgeries are planned based on surgeons experience, have limited adoption of minimally invasive techniques, and lead to inconsistent outcomes. We propose a pipeline that generates super-resolved, patient-specific 3D pseudo-healthy target morphologies from conventional clinical MR scans. First, we compute an isotropic super-resolved MR volume using an Implicit Neural Representation (INR). Next, we segment femur, tibia, patella, and fibula with a multi-label custom-trained network. Finally, we train a Wavelet Diffusion Model (WDM) to generate pseudo-healthy target morphologies of the trochlear region. In contrast to prior work producing pseudo-healthy low-resolution 3D MR images, our approach enables the generation of sub-millimeter resolved 3D shapes compatible for pre- and intraoperative use. These can serve as preoperative blueprints for reshaping the femoral groove while preserving the native patella articulation. Furthermore, and in contrast to other work, we do not require a CT for our pipeline - reducing the amount of radiation. We evaluated our approach on 25 TD patients and could show that our target morphologies significantly improve the sulcus angle (SA) and trochlear groove depth (TGD). The code and interactive visualization are available at https://wehrlimi.github.io/sr-3d-planning/.
>
---
#### [new 025] Text Embedded Swin-UMamba for DeepLesion Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出将文本嵌入Swin-UMamba架构，用于CT病变分割，结合影像与文本信息提升分割精度，取得82% Dice分数和6.58 Hausdorff距离，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.06453v1](http://arxiv.org/pdf/2508.06453v1)**

> **作者:** Ruida Cheng; Tejas Sudharshan Mathai; Pritam Mukherjee; Benjamin Hou; Qingqing Zhu; Zhiyong Lu; Matthew McAuliffe; Ronald M. Summers
>
> **摘要:** Segmentation of lesions on CT enables automatic measurement for clinical assessment of chronic diseases (e.g., lymphoma). Integrating large language models (LLMs) into the lesion segmentation workflow offers the potential to combine imaging features with descriptions of lesion characteristics from the radiology reports. In this study, we investigate the feasibility of integrating text into the Swin-UMamba architecture for the task of lesion segmentation. The publicly available ULS23 DeepLesion dataset was used along with short-form descriptions of the findings from the reports. On the test dataset, a high Dice Score of 82% and low Hausdorff distance of 6.58 (pixels) was obtained for lesion segmentation. The proposed Text-Swin-UMamba model outperformed prior approaches: 37% improvement over the LLM-driven LanGuideMedSeg model (p < 0.001),and surpassed the purely image-based xLSTM-UNet and nnUNet models by 1.74% and 0.22%, respectively. The dataset and code can be accessed at https://github.com/ruida/LLM-Swin-UMamba
>
---
#### [new 026] SPARSE Data, Rich Results: Few-Shot Semi-Supervised Learning via Class-Conditioned Image Translation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于GAN的半监督学习框架，针对小样本医学影像分类，通过类条件图像翻译和伪标签提升性能，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.06429v1](http://arxiv.org/pdf/2508.06429v1)**

> **作者:** Guido Manni; Clemente Lauretti; Loredana Zollo; Paolo Soda
>
> **摘要:** Deep learning has revolutionized medical imaging, but its effectiveness is severely limited by insufficient labeled training data. This paper introduces a novel GAN-based semi-supervised learning framework specifically designed for low labeled-data regimes, evaluated across settings with 5 to 50 labeled samples per class. Our approach integrates three specialized neural networks -- a generator for class-conditioned image translation, a discriminator for authenticity assessment and classification, and a dedicated classifier -- within a three-phase training framework. The method alternates between supervised training on limited labeled data and unsupervised learning that leverages abundant unlabeled images through image-to-image translation rather than generation from noise. We employ ensemble-based pseudo-labeling that combines confidence-weighted predictions from the discriminator and classifier with temporal consistency through exponential moving averaging, enabling reliable label estimation for unlabeled data. Comprehensive evaluation across eleven MedMNIST datasets demonstrates that our approach achieves statistically significant improvements over six state-of-the-art GAN-based semi-supervised methods, with particularly strong performance in the extreme 5-shot setting where the scarcity of labeled data is most challenging. The framework maintains its superiority across all evaluated settings (5, 10, 20, and 50 shots per class). Our approach offers a practical solution for medical imaging applications where annotation costs are prohibitive, enabling robust classification performance even with minimal labeled data. Code is available at https://github.com/GuidoManni/SPARSE.
>
---
#### [new 027] LightSwitch: Multi-view Relighting with Material-guided Diffusion
- **分类: cs.CV**

- **简介: 论文提出LightSwitch框架，通过结合多视角与材料属性，解决传统重照明方法在利用物体内在特征和多视角数据时的不足，提升重照明质量并加速处理。**

- **链接: [http://arxiv.org/pdf/2508.06494v1](http://arxiv.org/pdf/2508.06494v1)**

> **作者:** Yehonathan Litman; Fernando De la Torre; Shubham Tulsiani
>
> **备注:** ICCV 2025, Project page & Code: https://yehonathanlitman.github.io/light_switch/
>
> **摘要:** Recent approaches for 3D relighting have shown promise in integrating 2D image relighting generative priors to alter the appearance of a 3D representation while preserving the underlying structure. Nevertheless, generative priors used for 2D relighting that directly relight from an input image do not take advantage of intrinsic properties of the subject that can be inferred or cannot consider multi-view data at scale, leading to subpar relighting. In this paper, we propose Lightswitch, a novel finetuned material-relighting diffusion framework that efficiently relights an arbitrary number of input images to a target lighting condition while incorporating cues from inferred intrinsic properties. By using multi-view and material information cues together with a scalable denoising scheme, our method consistently and efficiently relights dense multi-view data of objects with diverse material compositions. We show that our 2D relighting prediction quality exceeds previous state-of-the-art relighting priors that directly relight from images. We further demonstrate that LightSwitch matches or outperforms state-of-the-art diffusion inverse rendering methods in relighting synthetic and real objects in as little as 2 minutes.
>
---
#### [new 028] MA-CBP: A Criminal Behavior Prediction Framework Based on Multi-Agent Asynchronous Collaboration
- **分类: cs.CV**

- **简介: 论文提出基于多智能体异步协作的犯罪行为预测框架MA-CBP，解决传统方法难以捕捉高阶语义与实时性问题，通过融合视频帧级语义与历史因果关系，实现犯罪行为早期预警。**

- **链接: [http://arxiv.org/pdf/2508.06189v1](http://arxiv.org/pdf/2508.06189v1)**

> **作者:** Cheng Liu; Daou Zhang; Tingxu Liu; Yuhan Wang; Jinyang Chen; Yuexuan Li; Xinying Xiao; Chenbo Xin; Ziru Wang; Weichao Wu
>
> **摘要:** With the acceleration of urbanization, criminal behavior in public scenes poses an increasingly serious threat to social security. Traditional anomaly detection methods based on feature recognition struggle to capture high-level behavioral semantics from historical information, while generative approaches based on Large Language Models (LLMs) often fail to meet real-time requirements. To address these challenges, we propose MA-CBP, a criminal behavior prediction framework based on multi-agent asynchronous collaboration. This framework transforms real-time video streams into frame-level semantic descriptions, constructs causally consistent historical summaries, and fuses adjacent image frames to perform joint reasoning over long- and short-term contexts. The resulting behavioral decisions include key elements such as event subjects, locations, and causes, enabling early warning of potential criminal activity. In addition, we construct a high-quality criminal behavior dataset that provides multi-scale language supervision, including frame-level, summary-level, and event-level semantic annotations. Experimental results demonstrate that our method achieves superior performance on multiple datasets and offers a promising solution for risk warning in urban public safety scenarios.
>
---
#### [new 029] Bifrost-1: Bridging Multimodal LLMs and Diffusion Models with Patch-level CLIP Latents
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出一种基于patch-level CLIP嵌入的框架，将预训练多模态LLMs与扩散模型融合，解决直接训练或桥梁方法成本高的问题，通过轻量适配器实现高效可控图像生成。**

- **链接: [http://arxiv.org/pdf/2508.05954v1](http://arxiv.org/pdf/2508.05954v1)**

> **作者:** Han Lin; Jaemin Cho; Amir Zadeh; Chuan Li; Mohit Bansal
>
> **备注:** Project Page: https://bifrost-1.github.io
>
> **摘要:** There is growing interest in integrating high-fidelity visual synthesis capabilities into large language models (LLMs) without compromising their strong reasoning capabilities. Existing methods that directly train LLMs or bridge LLMs and diffusion models usually suffer from costly training since the backbone LLMs have not seen image representations during pretraining. We present Bifrost-1, a unified framework that bridges pretrained multimodal LLMs (MLLMs) and diffusion models using patch-level CLIP image embeddings as latent variables, which are natively aligned with the MLLM's CLIP visual encoder. These patch-level image embeddings are integrated into the diffusion model with a lightweight adaptation of its ControlNet. To retain the original multimodal reasoning capabilities of MLLMs, we equip the MLLM with a visual generation branch initialized from the original MLLM parameters when predicting the patch-level image embeddings. By seamlessly integrating pretrained MLLMs and diffusion models with patch-level CLIP latents, our framework enables high-fidelity controllable image generation with significant training efficiency. Our experiments demonstrate that Bifrost-1 achieves comparable or better performance than previous methods in terms of visual fidelity and multimodal understanding, with substantially lower compute during training. We also provide comprehensive ablation studies showing the effectiveness of our design choices.
>
---
#### [new 030] Learning 3D Texture-Aware Representations for Parsing Diverse Human Clothing and Body Parts
- **分类: cs.CV**

- **简介: 该论文提出Spectrum网络，通过3D纹理生成器和I2Tx模型实现人体解析，解决传统方法对细粒度服装分类和多样化目标的不足，提升开放词汇分割性能。**

- **链接: [http://arxiv.org/pdf/2508.06032v1](http://arxiv.org/pdf/2508.06032v1)**

> **作者:** Kiran Chhatre; Christopher Peters; Srikrishna Karanam
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** Existing methods for human parsing into body parts and clothing often use fixed mask categories with broad labels that obscure fine-grained clothing types. Recent open-vocabulary segmentation approaches leverage pretrained text-to-image (T2I) diffusion model features for strong zero-shot transfer, but typically group entire humans into a single person category, failing to distinguish diverse clothing or detailed body parts. To address this, we propose Spectrum, a unified network for part-level pixel parsing (body parts and clothing) and instance-level grouping. While diffusion-based open-vocabulary models generalize well across tasks, their internal representations are not specialized for detailed human parsing. We observe that, unlike diffusion models with broad representations, image-driven 3D texture generators maintain faithful correspondence to input images, enabling stronger representations for parsing diverse clothing and body parts. Spectrum introduces a novel repurposing of an Image-to-Texture (I2Tx) diffusion model -- obtained by fine-tuning a T2I model on 3D human texture maps -- for improved alignment with body parts and clothing. From an input image, we extract human-part internal features via the I2Tx diffusion model and generate semantically valid masks aligned to diverse clothing categories through prompt-guided grounding. Once trained, Spectrum produces semantic segmentation maps for every visible body part and clothing category, ignoring standalone garments or irrelevant objects, for any number of humans in the scene. We conduct extensive cross-dataset experiments -- separately assessing body parts, clothing parts, unseen clothing categories, and full-body masks -- and demonstrate that Spectrum consistently outperforms baseline methods in prompt-based segmentation.
>
---
#### [new 031] Enhancing Construction Site Analysis and Understanding with 3D Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 论文提出使用三维分割技术提升施工现场分析，针对复杂动态环境优化SAM与Mask3D性能，对比其适应性并构建定制化流程以解决现有基准缺失问题。**

- **链接: [http://arxiv.org/pdf/2508.05922v1](http://arxiv.org/pdf/2508.05922v1)**

> **作者:** Sri Ramana Saketh Vasanthawada; Pengkun Liu; Pingbo Tang
>
> **摘要:** Monitoring construction progress is crucial yet resource-intensive, prompting the exploration of computer-vision-based methodologies for enhanced efficiency and scalability. Traditional data acquisition methods, primarily focusing on indoor environments, falter in construction site's complex, cluttered, and dynamically changing conditions. This paper critically evaluates the application of two advanced 3D segmentation methods, Segment Anything Model (SAM) and Mask3D, in challenging outdoor and indoor conditions. Trained initially on indoor datasets, both models' adaptability and performance are assessed in real-world construction settings, highlighting the gap in current segmentation approaches due to the absence of benchmarks for outdoor scenarios. Through a comparative analysis, this study not only showcases the relative effectiveness of SAM and Mask3D but also addresses the critical need for tailored segmentation workflows capable of extracting actionable insights from construction site data, thereby advancing the field towards more automated and precise monitoring techniques.
>
---
#### [new 032] Robust Image Stitching with Optimal Plane
- **分类: cs.CV**

- **简介: 该论文提出RopStitch，通过双分支架构融合内容先验与细粒度特征，利用虚拟最优平面缓解内容对齐与结构保留冲突，提升图像拼接的鲁棒性与自然性。**

- **链接: [http://arxiv.org/pdf/2508.05903v1](http://arxiv.org/pdf/2508.05903v1)**

> **作者:** Lang Nie; Yuan Mei; Kang Liao; Yunqiu Xu; Chunyu Lin; Bin Xiao
>
> **备注:** * Equal contribution
>
> **摘要:** We present \textit{RopStitch}, an unsupervised deep image stitching framework with both robustness and naturalness. To ensure the robustness of \textit{RopStitch}, we propose to incorporate the universal prior of content perception into the image stitching model by a dual-branch architecture. It separately captures coarse and fine features and integrates them to achieve highly generalizable performance across diverse unseen real-world scenes. Concretely, the dual-branch model consists of a pretrained branch to capture semantically invariant representations and a learnable branch to extract fine-grained discriminative features, which are then merged into a whole by a controllable factor at the correlation level. Besides, considering that content alignment and structural preservation are often contradictory to each other, we propose a concept of virtual optimal planes to relieve this conflict. To this end, we model this problem as a process of estimating homography decomposition coefficients, and design an iterative coefficient predictor and minimal semantic distortion constraint to identify the optimal plane. This scheme is finally incorporated into \textit{RopStitch} by warping both views onto the optimal plane bidirectionally. Extensive experiments across various datasets demonstrate that \textit{RopStitch} significantly outperforms existing methods, particularly in scene robustness and content naturalness. The code is available at {\color{red}https://github.com/MmelodYy/RopStitch}.
>
---
#### [new 033] Text as Any-Modality for Zero-Shot Classification by Consistent Prompt Tuning
- **分类: cs.CV**

- **简介: 论文提出基于文本的多模态零样本分类方法TaAM-CPT，通过一致提示调优实现跨模态泛化，无需模态特定数据即可支持无限模态扩展。**

- **链接: [http://arxiv.org/pdf/2508.06382v1](http://arxiv.org/pdf/2508.06382v1)**

> **作者:** Xiangyu Wu; Feng Yu; Yang Yang; Jianfeng Lu
>
> **备注:** Accepted for publication at ACMMM 2025
>
> **摘要:** The integration of prompt tuning with multimodal learning has shown significant generalization abilities for various downstream tasks. Despite advancements, existing methods heavily depend on massive modality-specific labeled data (e.g., video, audio, and image), or are customized for a single modality. In this study, we present Text as Any-Modality by Consistent Prompt Tuning (TaAM-CPT), a scalable approach for constructing a general representation model toward unlimited modalities using solely text data. TaAM-CPT comprises modality prompt pools, text construction, and modality-aligned text encoders from pre-trained models, which allows for extending new modalities by simply adding prompt pools and modality-aligned text encoders. To harmonize the learning across different modalities, TaAM-CPT designs intra- and inter-modal learning objectives, which can capture category details within modalities while maintaining semantic consistency across different modalities. Benefiting from its scalable architecture and pre-trained models, TaAM-CPT can be seamlessly extended to accommodate unlimited modalities. Remarkably, without any modality-specific labeled data, TaAM-CPT achieves leading results on diverse datasets spanning various modalities, including video classification, image classification, and audio classification. The code is available at https://github.com/Jinx630/TaAM-CPT.
>
---
#### [new 034] MZEN: Multi-Zoom Enhanced NeRF for 3-D Reconstruction with Unknown Camera Poses
- **分类: cs.CV**

- **简介: 论文提出MZEN框架，针对未知姿态下的3D重建难题，通过多焦距图像增强和姿态优化，提升工业检测中微米级细节还原能力，显著提升PSNR和SSIM指标。**

- **链接: [http://arxiv.org/pdf/2508.05819v1](http://arxiv.org/pdf/2508.05819v1)**

> **作者:** Jong-Ik Park; Carlee Joe-Wong; Gary K. Fedder
>
> **摘要:** Neural Radiance Fields (NeRF) methods excel at 3D reconstruction from multiple 2D images, even those taken with unknown camera poses. However, they still miss the fine-detailed structures that matter in industrial inspection, e.g., detecting sub-micron defects on a production line or analyzing chips with Scanning Electron Microscopy (SEM). In these scenarios, the sensor resolution is fixed and compute budgets are tight, so the only way to expose fine structure is to add zoom-in images; yet, this breaks the multi-view consistency that pose-free NeRF training relies on. We propose Multi-Zoom Enhanced NeRF (MZEN), the first NeRF framework that natively handles multi-zoom image sets. MZEN (i) augments the pin-hole camera model with an explicit, learnable zoom scalar that scales the focal length, and (ii) introduces a novel pose strategy: wide-field images are solved first to establish a global metric frame, and zoom-in images are then pose-primed to the nearest wide-field counterpart via a zoom-consistent crop-and-match procedure before joint refinement. Across eight forward-facing scenes$\unicode{x2013}$synthetic TCAD models, real SEM of micro-structures, and BLEFF objects$\unicode{x2013}$MZEN consistently outperforms pose-free baselines and even high-resolution variants, boosting PSNR by up to $28 \%$, SSIM by $10 \%$, and reducing LPIPS by up to $222 \%$. MZEN, therefore, extends NeRF to real-world factory settings, preserving global accuracy while capturing the micron-level details essential for industrial inspection.
>
---
#### [new 035] DSConv: Dynamic Splitting Convolution for Pansharpening
- **分类: cs.CV**

- **简介: 论文提出DSConv，通过动态分割卷积与注意力机制提升Pansharpening中不同位置特征提取，解决传统方法效率与效果不足问题，实现更优图像融合。**

- **链接: [http://arxiv.org/pdf/2508.06147v1](http://arxiv.org/pdf/2508.06147v1)**

> **作者:** Xuanyu Liu; Bonan An
>
> **摘要:** Aiming to obtain a high-resolution image, pansharpening involves the fusion of a multi-spectral image (MS) and a panchromatic image (PAN), the low-level vision task remaining significant and challenging in contemporary research. Most existing approaches rely predominantly on standard convolutions, few making the effort to adaptive convolutions, which are effective owing to the inter-pixel correlations of remote sensing images. In this paper, we propose a novel strategy for dynamically splitting convolution kernels in conjunction with attention, selecting positions of interest, and splitting the original convolution kernel into multiple smaller kernels, named DSConv. The proposed DSConv more effectively extracts features of different positions within the receptive field, enhancing the network's generalization, optimization, and feature representation capabilities. Furthermore, we innovate and enrich concepts of dynamic splitting convolution and provide a novel network architecture for pansharpening capable of achieving the tasks more efficiently, building upon this methodology. Adequate fair experiments illustrate the effectiveness and the state-of-the-art performance attained by DSConv.Comprehensive and rigorous discussions proved the superiority and optimal usage conditions of DSConv.
>
---
#### [new 036] An Implemention of Two-Phase Image Segmentation using the Split Bregman Method
- **分类: cs.CV; math.OC**

- **简介: 论文提出使用分裂Bregman方法实现两阶段图像分割，改进Chan-Vese模型能量以高效求解。**

- **链接: [http://arxiv.org/pdf/2508.06351v1](http://arxiv.org/pdf/2508.06351v1)**

> **作者:** Olakunle S. Abawonse; Günay Doğan
>
> **备注:** 15 pages
>
> **摘要:** In this paper, we describe an implementation of the two-phase image segmentation algorithm proposed by Goldstein, Bresson, Osher in \cite{gold:bre}. This algorithm partitions the domain of a given 2d image into foreground and background regions, and each pixel of the image is assigned membership to one of these two regions. The underlying assumption for the segmentation model is that the pixel values of the input image can be summarized by two distinct average values, and that the region boundaries are smooth. Accordingly, the model is defined as an energy in which the variable is a region membership function to assign pixels to either region, originally proposed by Chan and Vese in \cite{chan:vese}. This energy is the sum of image data terms in the regions and a length penalty for region boundaries. Goldstein, Bresson, Osher modify the energy of Chan-Vese in \cite{gold:bre} so that their new energy can be minimized efficiently using the split Bregman method to produce an equivalent two-phase segmentation. We provide a detailed implementation of this method \cite{gold:bre}, and document its performance with several images over a range of algorithm parameters.
>
---
#### [new 037] ViPro-2: Unsupervised State Estimation via Integrated Dynamics for Guiding Video Prediction
- **分类: cs.CV**

- **简介: 论文提出一种无监督视频预测方法，通过整合动力学避免依赖初始符号状态，解决状态估计问题，扩展了Orbits数据集的3D版本。**

- **链接: [http://arxiv.org/pdf/2508.06335v1](http://arxiv.org/pdf/2508.06335v1)**

> **作者:** Patrick Takenaka; Johannes Maucher; Marco F. Huber
>
> **备注:** Published in 2025 International Joint Conference on Neural Networks (IJCNN)
>
> **摘要:** Predicting future video frames is a challenging task with many downstream applications. Previous work has shown that procedural knowledge enables deep models for complex dynamical settings, however their model ViPro assumed a given ground truth initial symbolic state. We show that this approach led to the model learning a shortcut that does not actually connect the observed environment with the predicted symbolic state, resulting in the inability to estimate states given an observation if previous states are noisy. In this work, we add several improvements to ViPro that enables the model to correctly infer states from observations without providing a full ground truth state in the beginning. We show that this is possible in an unsupervised manner, and extend the original Orbits dataset with a 3D variant to close the gap to real world scenarios.
>
---
#### [new 038] Interpretable Rheumatoid Arthritis Scoring via Anatomy-aware Multiple Instance Learning
- **分类: cs.CV**

- **简介: 论文提出基于解剖结构的多实例学习方法，用于可解释的RA SvdH评分预测，解决传统手动评分效率低的问题，通过区域提取与注意力机制提升预测精度，达到与经验医生相当的性能。**

- **链接: [http://arxiv.org/pdf/2508.06218v1](http://arxiv.org/pdf/2508.06218v1)**

> **作者:** Zhiyan Bo; Laura C. Coates; Bartlomiej W. Papiez
>
> **备注:** Accepted by MICCAI AMAI Workshop 2025
>
> **摘要:** The Sharp/van der Heijde (SvdH) score has been widely used in clinical trials to quantify radiographic damage in Rheumatoid Arthritis (RA), but its complexity has limited its adoption in routine clinical practice. To address the inefficiency of manual scoring, this work proposes a two-stage pipeline for interpretable image-level SvdH score prediction using dual-hand radiographs. Our approach extracts disease-relevant image regions and integrates them using attention-based multiple instance learning to generate image-level features for prediction. We propose two region extraction schemes: 1) sampling image tiles most likely to contain abnormalities, and 2) cropping patches containing disease-relevant joints. With Scheme 2, our best individual score prediction model achieved a Pearson's correlation coefficient (PCC) of 0.943 and a root mean squared error (RMSE) of 15.73. Ensemble learning further boosted prediction accuracy, yielding a PCC of 0.945 and RMSE of 15.57, achieving state-of-the-art performance that is comparable to that of experienced radiologists (PCC = 0.97, RMSE = 18.75). Finally, our pipeline effectively identified and made decisions based on anatomical structures which clinicians consider relevant to RA progression.
>
---
#### [new 039] TEFormer: Texture-Aware and Edge-Guided Transformer for Semantic Segmentation of Urban Remote Sensing Images
- **分类: cs.CV**

- **简介: 论文提出TEFormer，通过纹理感知模块和边缘引导解码器提升城市遥感图像语义分割性能，解决纹理相似性高、边缘复杂导致的分类歧义问题。**

- **链接: [http://arxiv.org/pdf/2508.06224v1](http://arxiv.org/pdf/2508.06224v1)**

> **作者:** Guoyu Zhou; Jing Zhang; Yi Yan; Hui Zhang; Li Zhuo
>
> **备注:** Submitted to GRSL
>
> **摘要:** Semantic segmentation of urban remote sensing images (URSIs) is crucial for applications such as urban planning and environmental monitoring. However, geospatial objects often exhibit subtle texture differences and similar spatial structures, which can easily lead to semantic ambiguity and misclassification. Moreover, challenges such as irregular object shapes, blurred boundaries, and overlapping spatial distributions of semantic objects contribute to complex and diverse edge morphologies, further complicating accurate segmentation. To tackle these issues, we propose a texture-aware and edge-guided Transformer (TEFormer) that integrates texture awareness and edge-guidance mechanisms for semantic segmentation of URSIs. In the encoder, a texture-aware module (TaM) is designed to capture fine-grained texture differences between visually similar categories to enhance semantic discrimination. Then, an edge-guided tri-branch decoder (Eg3Head) is constructed to preserve local edges and details for multiscale context-awareness. Finally, an edge-guided feature fusion module (EgFFM) is to fuse contextual and detail information with edge information to realize refined semantic segmentation. Extensive experiments show that TEFormer achieves mIoU of 88.57%, 81.46%, and 53.55% on the Potsdam, Vaihingen, and LoveDA datasets, respectively, shows the effectiveness in URSI semantic segmentation.
>
---
#### [new 040] Fourier-VLM: Compressing Vision Tokens in the Frequency Domain for Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出在频域压缩视觉特征的方法，通过DCT与FFT降低计算开销，提升视觉语言模型的推理效率。**

- **链接: [http://arxiv.org/pdf/2508.06038v1](http://arxiv.org/pdf/2508.06038v1)**

> **作者:** Huanyu Wang; Jushi Kai; Haoli Bai; Lu Hou; Bo Jiang; Ziwei He; Zhouhan Lin
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Vision-Language Models (VLMs) typically replace the predefined image placeholder token (<image>) in textual instructions with visual features from an image encoder, forming the input to a backbone Large Language Model (LLM). However, the large number of vision tokens significantly increases the context length, leading to high computational overhead and inference latency. While previous efforts mitigate this by selecting only important visual features or leveraging learnable queries to reduce token count, they often compromise performance or introduce substantial extra costs. In response, we propose Fourier-VLM, a simple yet efficient method that compresses visual representations in the frequency domain. Our approach is motivated by the observation that vision features output from the vision encoder exhibit concentrated energy in low-frequency components. Leveraging this, we apply a low-pass filter to the vision features using a two-dimentional Discrete Cosine Transform (DCT). Notably, the DCT is efficiently computed via the Fast Fourier Transform (FFT) operator with a time complexity of $\mathcal{O}(n\log n)$, minimizing the extra computational cost while introducing no additional parameters. Extensive experiments across various image-based benchmarks demonstrate that Fourier-VLM achieves competitive performance with strong generalizability across both LLaVA and Qwen-VL architectures. Crucially, it reduce inference FLOPs by up to 83.8% and boots generation speed by 31.2% compared to LLaVA-v1.5, highlighting the superior efficiency and practicality.
>
---
#### [new 041] Lightweight Quad Bayer HybridEVS Demosaicing via State Space Augmented Cross-Attention
- **分类: cs.CV**

- **简介: 论文提出TSANet，针对HybridEVS传感器中事件像素缺乏颜色信息导致的aliasing问题，通过状态空间增强的交叉注意力机制实现轻量级去马赛克，分阶段处理补全与色度解码，提升性能并降低资源消耗。**

- **链接: [http://arxiv.org/pdf/2508.06058v1](http://arxiv.org/pdf/2508.06058v1)**

> **作者:** Shiyang Zhou; Haijin Zeng; Yunfan Lu; Yongyong Chen; Jie Liu; Jingyong Su
>
> **摘要:** Event cameras like the Hybrid Event-based Vision Sensor (HybridEVS) camera capture brightness changes as asynchronous "events" instead of frames, offering advanced application on mobile photography. However, challenges arise from combining a Quad Bayer Color Filter Array (CFA) sensor with event pixels lacking color information, resulting in aliasing and artifacts on the demosaicing process before downstream application. Current methods struggle to address these issues, especially on resource-limited mobile devices. In response, we introduce \textbf{TSANet}, a lightweight \textbf{T}wo-stage network via \textbf{S}tate space augmented cross-\textbf{A}ttention, which can handle event pixels inpainting and demosaicing separately, leveraging the benefits of dividing complex tasks into manageable subtasks. Furthermore, we introduce a lightweight Cross-Swin State Block that uniquely utilizes positional prior for demosaicing and enhances global dependencies through the state space model with linear complexity. In summary, TSANet demonstrates excellent demosaicing performance on both simulated and real data of HybridEVS while maintaining a lightweight model, averaging better results than the previous state-of-the-art method DemosaicFormer across seven diverse datasets in both PSNR and SSIM, while respectively reducing parameter and computation costs by $1.86\times$ and $3.29\times$. Our approach presents new possibilities for efficient image demosaicing on mobile devices. Code is available in the supplementary materials.
>
---
#### [new 042] More Is Better: A MoE-Based Emotion Recognition Framework with Human Preference Alignment
- **分类: cs.CV**

- **简介: 论文提出基于MoE的情感识别框架，通过半监督学习与人类偏好对齐，整合多模态数据提升准确率并纠正预测偏差。**

- **链接: [http://arxiv.org/pdf/2508.06036v1](http://arxiv.org/pdf/2508.06036v1)**

> **作者:** Jun Xie; Yingjian Zhu; Feng Chen; Zhenghao Zhang; Xiaohui Fan; Hongzhu Yi; Xinming Wang; Chen Yu; Yue Bi; Zhaoran Zhao; Xiongjun Guan; Zhepeng Wang
>
> **摘要:** In this paper, we present our solution for the semi-supervised learning track (MER-SEMI) in MER2025. We propose a comprehensive framework, grounded in the principle that "more is better," to construct a robust Mixture of Experts (MoE) emotion recognition system. Our approach integrates a diverse range of input modalities as independent experts, including novel signals such as knowledge from large Vision-Language Models (VLMs) and temporal Action Unit (AU) information. To effectively utilize unlabeled data, we introduce a consensus-based pseudo-labeling strategy, generating high-quality labels from the agreement between a baseline model and Gemini, which are then used in a two-stage training paradigm. Finally, we employ a multi-expert voting ensemble combined with a rule-based re-ranking process to correct prediction bias and better align the outputs with human preferences. Evaluated on the MER2025-SEMI challenge dataset, our method achieves an F1-score of 0.8772 on the test set, ranking 2nd in the track. Our code is available at https://github.com/zhuyjan/MER2025-MRAC25.
>
---
#### [new 043] XAG-Net: A Cross-Slice Attention and Skip Gating Network for 2.5D Femur MRI Segmentation
- **分类: cs.CV**

- **简介: 论文提出XAG-Net，一种2.5D U-Net架构，通过跨切片注意力（CSA）和跳过注意力门控（AG）提升骨盆MRI分割精度，解决传统2D/3D方法在复杂结构建模上的不足，实现高效准确分割。**

- **链接: [http://arxiv.org/pdf/2508.06258v1](http://arxiv.org/pdf/2508.06258v1)**

> **作者:** Byunghyun Ko; Anning Tian; Jeongkyu Lee
>
> **备注:** Accepted at the 2025 International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA). This is the preprint version of the paper
>
> **摘要:** Accurate segmentation of femur structures from Magnetic Resonance Imaging (MRI) is critical for orthopedic diagnosis and surgical planning but remains challenging due to the limitations of existing 2D and 3D deep learning-based segmentation approaches. In this study, we propose XAG-Net, a novel 2.5D U-Net-based architecture that incorporates pixel-wise cross-slice attention (CSA) and skip attention gating (AG) mechanisms to enhance inter-slice contextual modeling and intra-slice feature refinement. Unlike previous CSA-based models, XAG-Net applies pixel-wise softmax attention across adjacent slices at each spatial location for fine-grained inter-slice modeling. Extensive evaluations demonstrate that XAG-Net surpasses baseline 2D, 2.5D, and 3D U-Net models in femur segmentation accuracy while maintaining computational efficiency. Ablation studies further validate the critical role of the CSA and AG modules, establishing XAG-Net as a promising framework for efficient and accurate femur MRI segmentation.
>
---
#### [new 044] VISTA: Vision-Language Imitation of Situational Thinking and Attention for Human-Like Driver Focus in Dynamic Environments
- **分类: cs.CV; I.5.4**

- **简介: 论文提出一种视觉-语言框架，通过结合自然语言和RGB图像，预测动态环境下驾驶员注意力分配，解决传统静态图像方法不足的问题，提升注意力转移检测与可解释性，为自动驾驶提供新方向。**

- **链接: [http://arxiv.org/pdf/2508.05852v1](http://arxiv.org/pdf/2508.05852v1)**

> **作者:** Kaiser Hamid; Khandakar Ashrafi Akbar; Nade Liang
>
> **摘要:** Driver visual attention prediction is a critical task in autonomous driving and human-computer interaction (HCI) research. Most prior studies focus on estimating attention allocation at a single moment in time, typically using static RGB images such as driving scene pictures. In this work, we propose a vision-language framework that models the changing landscape of drivers' gaze through natural language, using few-shot and zero-shot learning on single RGB images. We curate and refine high-quality captions from the BDD-A dataset using human-in-the-loop feedback, then fine-tune LLaVA to align visual perception with attention-centric scene understanding. Our approach integrates both low-level cues and top-down context (e.g., route semantics, risk anticipation), enabling language-based descriptions of gaze behavior. We evaluate performance across training regimes (few shot, and one-shot) and introduce domain-specific metrics for semantic alignment and response diversity. Results show that our fine-tuned model outperforms general-purpose VLMs in attention shift detection and interpretability. To our knowledge, this is among the first attempts to generate driver visual attention allocation and shifting predictions in natural language, offering a new direction for explainable AI in autonomous driving. Our approach provides a foundation for downstream tasks such as behavior forecasting, human-AI teaming, and multi-agent coordination.
>
---
#### [new 045] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning
- **分类: cs.CV**

- **简介: 论文提出SC-Captioner框架，通过强化学习实现图像描述的自修正，利用场景图解析分解对象/属性/关系，结合差异计算与奖励机制提升准确性，同时改进评估指标并构建RefinedCaps数据集，显著优于直接优化方法。**

- **链接: [http://arxiv.org/pdf/2508.06125v1](http://arxiv.org/pdf/2508.06125v1)**

> **作者:** Lin Zhang; Xianfang Zeng; Kangcong Li; Gang Yu; Tao Chen
>
> **备注:** ICCV 2025
>
> **摘要:** We propose SC-Captioner, a reinforcement learning framework that enables the self-correcting capability of image caption models. Our crucial technique lies in the design of the reward function to incentivize accurate caption corrections. Specifically, the predicted and reference captions are decomposed into object, attribute, and relation sets using scene-graph parsing algorithms. We calculate the set difference between sets of initial and self-corrected captions to identify added and removed elements. These elements are matched against the reference sets to calculate correctness bonuses for accurate refinements and mistake punishments for wrong additions and removals, thereby forming the final reward. For image caption quality assessment, we propose a set of metrics refined from CAPTURE that alleviate its incomplete precision evaluation and inefficient relation matching problems. Furthermore, we collect a fine-grained annotated image caption dataset, RefinedCaps, consisting of 6.5K diverse images from COCO dataset. Experiments show that applying SC-Captioner on large visual-language models can generate better image captions across various scenarios, significantly outperforming the direct preference optimization training strategy.
>
---
#### [new 046] UnGuide: Learning to Forget with LoRA-Guided Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 论文提出UnGuide方法，通过LoRA指导的扩散模型实现可控遗忘，解决预训练模型生成有害内容问题。核心在于动态调整指导尺度，结合CFG控制遗忘过程，既保留模型表达力又精准移除指定概念，优于现有LoRA方法。**

- **链接: [http://arxiv.org/pdf/2508.05755v1](http://arxiv.org/pdf/2508.05755v1)**

> **作者:** Agnieszka Polowczyk; Alicja Polowczyk; Dawid Malarz; Artur Kasymov; Marcin Mazur; Jacek Tabor; Przemysław Spurek
>
> **摘要:** Recent advances in large-scale text-to-image diffusion models have heightened concerns about their potential misuse, especially in generating harmful or misleading content. This underscores the urgent need for effective machine unlearning, i.e., removing specific knowledge or concepts from pretrained models without compromising overall performance. One possible approach is Low-Rank Adaptation (LoRA), which offers an efficient means to fine-tune models for targeted unlearning. However, LoRA often inadvertently alters unrelated content, leading to diminished image fidelity and realism. To address this limitation, we introduce UnGuide -- a novel approach which incorporates UnGuidance, a dynamic inference mechanism that leverages Classifier-Free Guidance (CFG) to exert precise control over the unlearning process. UnGuide modulates the guidance scale based on the stability of a few first steps of denoising processes, enabling selective unlearning by LoRA adapter. For prompts containing the erased concept, the LoRA module predominates and is counterbalanced by the base model; for unrelated prompts, the base model governs generation, preserving content fidelity. Empirical results demonstrate that UnGuide achieves controlled concept removal and retains the expressive power of diffusion models, outperforming existing LoRA-based methods in both object erasure and explicit content removal tasks.
>
---
#### [new 047] TRUST: Leveraging Text Robustness for Unsupervised Domain Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 论文提出TRUST，通过语言模态增强视觉模型在复杂域转移（如地理）中的鲁棒性，利用伪标签与CLIP相似度估计不确定性，结合多模态对比学习优化特征对齐，实现更有效的无监督领域自适应。**

- **链接: [http://arxiv.org/pdf/2508.06452v1](http://arxiv.org/pdf/2508.06452v1)**

> **作者:** Mattia Litrico; Mario Valerio Giuffrida; Sebastiano Battiato; Devis Tuia
>
> **摘要:** Recent unsupervised domain adaptation (UDA) methods have shown great success in addressing classical domain shifts (e.g., synthetic-to-real), but they still suffer under complex shifts (e.g. geographical shift), where both the background and object appearances differ significantly across domains. Prior works showed that the language modality can help in the adaptation process, exhibiting more robustness to such complex shifts. In this paper, we introduce TRUST, a novel UDA approach that exploits the robustness of the language modality to guide the adaptation of a vision model. TRUST generates pseudo-labels for target samples from their captions and introduces a novel uncertainty estimation strategy that uses normalised CLIP similarity scores to estimate the uncertainty of the generated pseudo-labels. Such estimated uncertainty is then used to reweight the classification loss, mitigating the adverse effects of wrong pseudo-labels obtained from low-quality captions. To further increase the robustness of the vision model, we propose a multimodal soft-contrastive learning loss that aligns the vision and language feature spaces, by leveraging captions to guide the contrastive training of the vision model on target images. In our contrastive loss, each pair of images acts as both a positive and a negative pair and their feature representations are attracted and repulsed with a strength proportional to the similarity of their captions. This solution avoids the need for hardly determining positive and negative pairs, which is critical in the UDA setting. Our approach outperforms previous methods, setting the new state-of-the-art on classical (DomainNet) and complex (GeoNet) domain shifts. The code will be available upon acceptance.
>
---
#### [new 048] VQAThinker: Exploring Generalizable and Explainable Video Quality Assessment via Reinforcement Learning
- **分类: cs.CV**

- **简介: 论文提出VQAThinker，通过强化学习与大模态模型，结合三种奖励机制解决VQA中泛化与解释性问题，提升性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.06051v1](http://arxiv.org/pdf/2508.06051v1)**

> **作者:** Linhan Cao; Wei Sun; Weixia Zhang; Xiangyang Zhu; Jun Jia; Kaiwei Zhang; Dandan Zhu; Guangtao Zhai; Xiongkuo Min
>
> **摘要:** Video quality assessment (VQA) aims to objectively quantify perceptual quality degradation in alignment with human visual perception. Despite recent advances, existing VQA models still suffer from two critical limitations: \textit{poor generalization to out-of-distribution (OOD) videos} and \textit{limited explainability}, which restrict their applicability in real-world scenarios. To address these challenges, we propose \textbf{VQAThinker}, a reasoning-based VQA framework that leverages large multimodal models (LMMs) with reinforcement learning to jointly model video quality understanding and scoring, emulating human perceptual decision-making. Specifically, we adopt group relative policy optimization (GRPO), a rule-guided reinforcement learning algorithm that enables reasoning over video quality under score-level supervision, and introduce three VQA-specific rewards: (1) a \textbf{bell-shaped regression reward} that increases rapidly as the prediction error decreases and becomes progressively less sensitive near the ground truth; (2) a \textbf{pairwise ranking reward} that guides the model to correctly determine the relative quality between video pairs; and (3) a \textbf{temporal consistency reward} that encourages the model to prefer temporally coherent videos over their perturbed counterparts. Extensive experiments demonstrate that VQAThinker achieves state-of-the-art performance on both in-domain and OOD VQA benchmarks, showing strong generalization for video quality scoring. Furthermore, evaluations on video quality understanding tasks validate its superiority in distortion attribution and quality description compared to existing explainable VQA models and LMMs. These findings demonstrate that reinforcement learning offers an effective pathway toward building generalizable and explainable VQA models solely with score-level supervision.
>
---
#### [new 049] ETTA: Efficient Test-Time Adaptation for Vision-Language Models through Dynamic Embedding Updates
- **分类: cs.CV**

- **简介: 论文提出ETTA，通过递归更新模块动态整合所有测试样本，提升视觉-语言模型在分布偏移下的泛化能力，兼顾效率与准确性。**

- **链接: [http://arxiv.org/pdf/2508.05898v1](http://arxiv.org/pdf/2508.05898v1)**

> **作者:** Hamidreza Dastmalchi; Aijun An; Ali cheraghian
>
> **备注:** BMVC2025
>
> **摘要:** Pretrained vision-language models (VLMs) like CLIP show strong zero-shot performance but struggle with generalization under distribution shifts. Test-Time Adaptation (TTA) addresses this by adapting VLMs to unlabeled test data in new domains. While some TTA methods rely on prompt-tuning, training-free cache-based approaches are preferred for efficiency. However, current cache-based TTA models store only a limited set of high-confidence samples, restricting the decision boundary to these samples and ignoring the influence of other incoming test data. To address this, we propose Efficient Test-Time Adaptation (ETTA), introducing a Recursive Updating module that integrates all incoming test samples, progressively refining the decision boundary. This strategy mimics an unbounded cache, dynamically updating contextual embeddings for improved accuracy with minimal memory and computational overhead. ETTA also includes an Adaptive Ensemble module to reduce prompt dependency in image-to-text scores by dynamically selecting optimal prompts for each class. Furthermore, ETTA adaptively combines scores from both modules based on confidence levels, leveraging their complementary strengths. Extensive experiments on two benchmarks confirm that ETTA surpasses the state-of-the-art TTA models in computational complexity and accuracy, setting a new standard for effective, efficient test-time adaptation. The code has been released at https://github.com/hamidreza-dastmalchi/ETTA.
>
---
#### [new 050] UW-3DGS: Underwater 3D Reconstruction with Physics-Aware Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **简介: 论文提出UW-3DGS框架，通过物理感知的不确定性剪枝和可学习的水下图像形成模块，解决水下3D重建中的光衰减和浑浊问题，提升几何和颜色精度。**

- **链接: [http://arxiv.org/pdf/2508.06169v1](http://arxiv.org/pdf/2508.06169v1)**

> **作者:** Wenpeng Xing; Jie Chen; Zaifeng Yang; Changting Lin; Jianfeng Dong; Chaochao Chen; Xun Zhou; Meng Han
>
> **摘要:** Underwater 3D scene reconstruction faces severe challenges from light absorption, scattering, and turbidity, which degrade geometry and color fidelity in traditional methods like Neural Radiance Fields (NeRF). While NeRF extensions such as SeaThru-NeRF incorporate physics-based models, their MLP reliance limits efficiency and spatial resolution in hazy environments. We introduce UW-3DGS, a novel framework adapting 3D Gaussian Splatting (3DGS) for robust underwater reconstruction. Key innovations include: (1) a plug-and-play learnable underwater image formation module using voxel-based regression for spatially varying attenuation and backscatter; and (2) a Physics-Aware Uncertainty Pruning (PAUP) branch that adaptively removes noisy floating Gaussians via uncertainty scoring, ensuring artifact-free geometry. The pipeline operates in training and rendering stages. During training, noisy Gaussians are optimized end-to-end with underwater parameters, guided by PAUP pruning and scattering modeling. In rendering, refined Gaussians produce clean Unattenuated Radiance Images (URIs) free from media effects, while learned physics enable realistic Underwater Images (UWIs) with accurate light transport. Experiments on SeaThru-NeRF and UWBundle datasets show superior performance, achieving PSNR of 27.604, SSIM of 0.868, and LPIPS of 0.104 on SeaThru-NeRF, with ~65% reduction in floating artifacts.
>
---
#### [new 051] Learning Representations of Satellite Images with Evaluations on Synoptic Weather Events
- **分类: cs.CV; physics.ao-ph**

- **简介: 该论文研究卫星图像表示学习，旨在通过天气事件分类评估不同算法（PCA、CAE、PT）的性能，发现CAE在所有任务中表现最优，高分辨率数据提升效果显著，但需改进解释性。**

- **链接: [http://arxiv.org/pdf/2508.06122v1](http://arxiv.org/pdf/2508.06122v1)**

> **作者:** Ting-Shuo Yo; Shih-Hao Su; Chien-Ming Wu; Wei-Ting Chen; Jung-Lien Chu; Chiao-Wei Chang; Hung-Chi Kuo
>
> **备注:** 37 pages, 6 figures, 3 tables
>
> **摘要:** This study applied representation learning algorithms to satellite images and evaluated the learned latent spaces with classifications of various weather events. The algorithms investigated include the classical linear transformation, i.e., principal component analysis (PCA), state-of-the-art deep learning method, i.e., convolutional autoencoder (CAE), and a residual network pre-trained with large image datasets (PT). The experiment results indicated that the latent space learned by CAE consistently showed higher threat scores for all classification tasks. The classifications with PCA yielded high hit rates but also high false-alarm rates. In addition, the PT performed exceptionally well at recognizing tropical cyclones but was inferior in other tasks. Further experiments suggested that representations learned from higher-resolution datasets are superior in all classification tasks for deep-learning algorithms, i.e., CAE and PT. We also found that smaller latent space sizes had minor impact on the classification task's hit rate. Still, a latent space dimension smaller than 128 caused a significantly higher false alarm rate. Though the CAE can learn latent spaces effectively and efficiently, the interpretation of the learned representation lacks direct connections to physical attributions. Therefore, developing a physics-informed version of CAE can be a promising outlook for the current work.
>
---
#### [new 052] Can Diffusion Models Bridge the Domain Gap in Cardiac MR Imaging?
- **分类: cs.CV**

- **简介: 论文探讨扩散模型在心脏MRI领域迁移中的应用，解决因设备/协议差异导致的域偏移问题，通过生成合成影像提升分割性能。**

- **链接: [http://arxiv.org/pdf/2508.06327v1](http://arxiv.org/pdf/2508.06327v1)**

> **作者:** Xin Ci Wong; Duygu Sarikaya; Kieran Zucker; Marc De Kamps; Nishant Ravikumar
>
> **备注:** ICONIP 2025
>
> **摘要:** Magnetic resonance (MR) imaging, including cardiac MR, is prone to domain shift due to variations in imaging devices and acquisition protocols. This challenge limits the deployment of trained AI models in real-world scenarios, where performance degrades on unseen domains. Traditional solutions involve increasing the size of the dataset through ad-hoc image augmentation or additional online training/transfer learning, which have several limitations. Synthetic data offers a promising alternative, but anatomical/structural consistency constraints limit the effectiveness of generative models in creating image-label pairs. To address this, we propose a diffusion model (DM) trained on a source domain that generates synthetic cardiac MR images that resemble a given reference. The synthetic data maintains spatial and structural fidelity, ensuring similarity to the source domain and compatibility with the segmentation mask. We assess the utility of our generative approach in multi-centre cardiac MR segmentation, using the 2D nnU-Net, 3D nnU-Net and vanilla U-Net segmentation networks. We explore domain generalisation, where, domain-invariant segmentation models are trained on synthetic source domain data, and domain adaptation, where, we shift target domain data towards the source domain using the DM. Both strategies significantly improved segmentation performance on data from an unseen target domain, in terms of surface-based metrics (Welch's t-test, p < 0.01), compared to training segmentation models on real data alone. The proposed method ameliorates the need for transfer learning or online training to address domain shift challenges in cardiac MR image analysis, especially useful in data-scarce settings.
>
---
#### [new 053] A 3DGS-Diffusion Self-Supervised Framework for Normal Estimation from a Single Image
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种基于3D Gaussians splatting的自监督框架SINGAD，解决单图像法向量估计中多视角几何不一致与数据依赖问题，通过物理光-表面建模与可微渲染重投影策略，将几何误差转化为法向量优化信号，无需密集法向量标注。**

- **链接: [http://arxiv.org/pdf/2508.05950v1](http://arxiv.org/pdf/2508.05950v1)**

> **作者:** Yanxing Liang; Yinghui Wang; Jinlong Yang; Wei Li
>
> **摘要:** The lack of spatial dimensional information remains a challenge in normal estimation from a single image. Recent diffusion-based methods have demonstrated significant potential in 2D-to-3D implicit mapping, they rely on data-driven statistical priors and miss the explicit modeling of light-surface interaction, leading to multi-view normal direction conflicts. Moreover, the discrete sampling mechanism of diffusion models causes gradient discontinuity in differentiable rendering reconstruction modules, preventing 3D geometric errors from being backpropagated to the normal generation network, thereby forcing existing methods to depend on dense normal annotations. This paper proposes SINGAD, a novel Self-supervised framework from a single Image for Normal estimation via 3D GAussian splatting guided Diffusion. By integrating physics-driven light-interaction modeling and a differentiable rendering-based reprojection strategy, our framework directly converts 3D geometric errors into normal optimization signals, solving the challenges of multi-view geometric inconsistency and data dependency. Specifically, the framework constructs a light-interaction-driven 3DGS reparameterization model to generate multi-scale geometric features consistent with light transport principles, ensuring multi-view normal consistency. A cross-domain feature fusion module is designed within a conditional diffusion model, embedding geometric priors to constrain normal generation while maintaining accurate geometric error propagation. Furthermore, a differentiable 3D reprojection loss strategy is introduced for self-supervised optimization that minimizes geometric error between the reconstructed and input image, eliminating dependence on annotated normal datasets. Quantitative evaluations on the Google Scanned Objects dataset demonstrate that our method outperforms state-of-the-art approaches across multiple metrics.
>
---
#### [new 054] Generalized Few-Shot Out-of-Distribution Detection
- **分类: cs.CV**

- **简介: 论文提出基于通用知识模型（GKM）的Few-shot OOD检测框架GOOD，通过理论推导Generality-Specificity平衡（GS-balance）和动态知识嵌入（KDE）提升泛化能力，实验证明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.05732v1](http://arxiv.org/pdf/2508.05732v1)**

> **作者:** Pinxuan Li; Bing Cao; Changqing Zhang; Qinghua Hu
>
> **摘要:** Few-shot Out-of-Distribution (OOD) detection has emerged as a critical research direction in machine learning for practical deployment. Most existing Few-shot OOD detection methods suffer from insufficient generalization capability for the open world. Due to the few-shot learning paradigm, the OOD detection ability is often overfit to the limited training data itself, thus degrading the performance on generalized data and performing inconsistently across different scenarios. To address this challenge, we proposed a Generalized Few-shot OOD Detection (GOOD) framework, which empowers the general knowledge of the OOD detection model with an auxiliary General Knowledge Model (GKM), instead of directly learning from few-shot data. We proceed to reveal the few-shot OOD detection from a generalization perspective and theoretically derive the Generality-Specificity balance (GS-balance) for OOD detection, which provably reduces the upper bound of generalization error with a general knowledge model. Accordingly, we propose a Knowledge Dynamic Embedding (KDE) mechanism to adaptively modulate the guidance of general knowledge. KDE dynamically aligns the output distributions of the OOD detection model to the general knowledge model based on the Generalized Belief (G-Belief) of GKM, thereby boosting the GS-balance. Experiments on real-world OOD benchmarks demonstrate our superiority. Codes will be available.
>
---
#### [new 055] Mixture of Experts Guided by Gaussian Splatters Matters: A new Approach to Weakly-Supervised Video Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于高斯溅射的混合专家模型（GS-MoE）解决弱监督视频异常检测问题，通过时空特征融合与专家协同，提升复杂场景下异常识别性能。**

- **链接: [http://arxiv.org/pdf/2508.06318v1](http://arxiv.org/pdf/2508.06318v1)**

> **作者:** Giacomo D'Amicantonio; Snehashis Majhi; Quan Kong; Lorenzo Garattoni; Gianpiero Francesca; François Bremond; Egor Bondarev
>
> **摘要:** Video Anomaly Detection (VAD) is a challenging task due to the variability of anomalous events and the limited availability of labeled data. Under the Weakly-Supervised VAD (WSVAD) paradigm, only video-level labels are provided during training, while predictions are made at the frame level. Although state-of-the-art models perform well on simple anomalies (e.g., explosions), they struggle with complex real-world events (e.g., shoplifting). This difficulty stems from two key issues: (1) the inability of current models to address the diversity of anomaly types, as they process all categories with a shared model, overlooking category-specific features; and (2) the weak supervision signal, which lacks precise temporal information, limiting the ability to capture nuanced anomalous patterns blended with normal events. To address these challenges, we propose Gaussian Splatting-guided Mixture of Experts (GS-MoE), a novel framework that employs a set of expert models, each specialized in capturing specific anomaly types. These experts are guided by a temporal Gaussian splatting loss, enabling the model to leverage temporal consistency and enhance weak supervision. The Gaussian splatting approach encourages a more precise and comprehensive representation of anomalies by focusing on temporal segments most likely to contain abnormal events. The predictions from these specialized experts are integrated through a mixture-of-experts mechanism to model complex relationships across diverse anomaly patterns. Our approach achieves state-of-the-art performance, with a 91.58% AUC on the UCF-Crime dataset, and demonstrates superior results on XD-Violence and MSAD datasets. By leveraging category-specific expertise and temporal guidance, GS-MoE sets a new benchmark for VAD under weak supervision.
>
---
#### [new 056] Graph-based Robot Localization Using a Graph Neural Network with a Floor Camera and a Feature Rich Industrial Floor
- **分类: cs.CV; cs.RO**

- **简介: 论文提出基于图神经网络的机器人定位方法，利用地板特征图表示，实现高精度（0.64cm）和高效定位，解决复杂环境下的kidnapped robot问题。**

- **链接: [http://arxiv.org/pdf/2508.06177v1](http://arxiv.org/pdf/2508.06177v1)**

> **作者:** Dominik Brämer; Diana Kleingarn; Oliver Urbann
>
> **备注:** Accepted at 28th RoboCup International Symposium, Salvador, Brasil
>
> **摘要:** Accurate localization represents a fundamental challenge in robotic navigation. Traditional methodologies, such as Lidar or QR-code based systems, suffer from inherent scalability and adaptability con straints, particularly in complex environments. In this work, we propose an innovative localization framework that harnesses flooring characteris tics by employing graph-based representations and Graph Convolutional Networks (GCNs). Our method uses graphs to represent floor features, which helps localize the robot more accurately (0.64cm error) and more efficiently than comparing individual image features. Additionally, this approach successfully addresses the kidnapped robot problem in every frame without requiring complex filtering processes. These advancements open up new possibilities for robotic navigation in diverse environments.
>
---
#### [new 057] SynSeg: Feature Synergy for Multi-Category Contrastive Learning in Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文提出SynSeg，针对开放词汇语义分割中类别监督不足和特征构造问题，通过多类别对比学习（MCCL）与特征重构框架FSS，提升模型在弱监督下的语义定位与区分能力，实验表明其在VOC等基准上优于SOTA。**

- **链接: [http://arxiv.org/pdf/2508.06115v1](http://arxiv.org/pdf/2508.06115v1)**

> **作者:** Weichen Zhang; Kebin Liu; Fan Dang; Zhui Zhu; Xikai Sun; Yunhao Liu
>
> **摘要:** Semantic segmentation in open-vocabulary scenarios presents significant challenges due to the wide range and granularity of semantic categories. Existing weakly-supervised methods often rely on category-specific supervision and ill-suited feature construction methods for contrastive learning, leading to semantic misalignment and poor performance. In this work, we propose a novel weakly-supervised approach, SynSeg, to address the challenges. SynSeg performs Multi-Category Contrastive Learning (MCCL) as a stronger training signal with a new feature reconstruction framework named Feature Synergy Structure (FSS). Specifically, MCCL strategy robustly combines both intra- and inter-category alignment and separation in order to make the model learn the knowledge of correlations from different categories within the same image. Moreover, FSS reconstructs discriminative features for contrastive learning through prior fusion and semantic-activation-map enhancement, effectively avoiding the foreground bias introduced by the visual encoder. In general, SynSeg effectively improves the abilities in semantic localization and discrimination under weak supervision. Extensive experiments on benchmarks demonstrate that our method outperforms state-of-the-art (SOTA) performance. For instance, SynSeg achieves higher accuracy than SOTA baselines by 4.5\% on VOC, 8.9\% on Context, 2.6\% on Object and 2.0\% on City.
>
---
#### [new 058] Few-Shot Deployment of Pretrained MRI Transformers in Brain Imaging Tasks
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于MAE的少样本部署框架，针对脑成像任务中标注数据稀缺问题，通过预训练MRI变换器获取可迁移特征，并设计MAE-FUnet提升分割性能，实现高效稳定的应用。**

- **链接: [http://arxiv.org/pdf/2508.05783v1](http://arxiv.org/pdf/2508.05783v1)**

> **作者:** Mengyu Li; Guoyao Shen; Chad W. Farris; Xin Zhang
>
> **备注:** 30 pages, 8 figures, 7 tables
>
> **摘要:** Machine learning using transformers has shown great potential in medical imaging, but its real-world applicability remains limited due to the scarcity of annotated data. In this study, we propose a practical framework for the few-shot deployment of pretrained MRI transformers in diverse brain imaging tasks. By utilizing the Masked Autoencoder (MAE) pretraining strategy on a large-scale, multi-cohort brain MRI dataset comprising over 31 million slices, we obtain highly transferable latent representations that generalize well across tasks and datasets. For high-level tasks such as classification, a frozen MAE encoder combined with a lightweight linear head achieves state-of-the-art accuracy in MRI sequence identification with minimal supervision. For low-level tasks such as segmentation, we propose MAE-FUnet, a hybrid architecture that fuses multiscale CNN features with pretrained MAE embeddings. This model consistently outperforms other strong baselines in both skull stripping and multi-class anatomical segmentation under data-limited conditions. With extensive quantitative and qualitative evaluations, our framework demonstrates efficiency, stability, and scalability, suggesting its suitability for low-resource clinical environments and broader neuroimaging applications.
>
---
#### [new 059] A Semantic Segmentation Algorithm for Pleural Effusion Based on DBIF-AUNet
- **分类: cs.CV; 68T45, 92C55; I.4.6; I.5.4; J.3**

- **简介: 论文提出DBIF-AUNet用于pleural effusion CT语义分割，解决灰度相似、边缘模糊及形态变化问题，通过双分支交互融合注意力机制与层次化损失函数提升分割精度，实验显示优于U-Net++和Swin-UNet。**

- **链接: [http://arxiv.org/pdf/2508.06191v1](http://arxiv.org/pdf/2508.06191v1)**

> **作者:** Ruixiang Tang; Jianglong Qin; Mingda Zhang; Yan Song; Yi Wu; Wei Wu
>
> **备注:** 12 pages, 6 figures, 2 tables
>
> **摘要:** Pleural effusion semantic segmentation can significantly enhance the accuracy and timeliness of clinical diagnosis and treatment by precisely identifying disease severity and lesion areas. Currently, semantic segmentation of pleural effusion CT images faces multiple challenges. These include similar gray levels between effusion and surrounding tissues, blurred edges, and variable morphology. Existing methods often struggle with diverse image variations and complex edges, primarily because direct feature concatenation causes semantic gaps. To address these challenges, we propose the Dual-Branch Interactive Fusion Attention model (DBIF-AUNet). This model constructs a densely nested skip-connection network and innovatively refines the Dual-Domain Feature Disentanglement module (DDFD). The DDFD module orthogonally decouples the functions of dual-domain modules to achieve multi-scale feature complementarity and enhance characteristics at different levels. Concurrently, we design a Branch Interaction Attention Fusion module (BIAF) that works synergistically with the DDFD. This module dynamically weights and fuses global, local, and frequency band features, thereby improving segmentation robustness. Furthermore, we implement a nested deep supervision mechanism with hierarchical adaptive hybrid loss to effectively address class imbalance. Through validation on 1,622 pleural effusion CT images from Southwest Hospital, DBIF-AUNet achieved IoU and Dice scores of 80.1% and 89.0% respectively. These results outperform state-of-the-art medical image segmentation models U-Net++ and Swin-UNet by 5.7%/2.7% and 2.2%/1.5% respectively, demonstrating significant optimization in segmentation accuracy for complex pleural effusion CT images.
>
---
#### [new 060] Improved Sub-Visible Particle Classification in Flow Imaging Microscopy via Generative AI-Based Image Synthesis
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出基于生成式AI的扩散模型解决亚可见粒子分类中的数据不足与类别不平衡问题，通过合成高保真图像增强训练集，提升多类分类性能，并开源模型与代码。**

- **链接: [http://arxiv.org/pdf/2508.06021v1](http://arxiv.org/pdf/2508.06021v1)**

> **作者:** Utku Ozbulak; Michaela Cohrs; Hristo L. Svilenov; Joris Vankerschaver; Wesley De Neve
>
> **摘要:** Sub-visible particle analysis using flow imaging microscopy combined with deep learning has proven effective in identifying particle types, enabling the distinction of harmless components such as silicone oil from protein particles. However, the scarcity of available data and severe imbalance between particle types within datasets remain substantial hurdles when applying multi-class classifiers to such problems, often forcing researchers to rely on less effective methods. The aforementioned issue is particularly challenging for particle types that appear unintentionally and in lower numbers, such as silicone oil and air bubbles, as opposed to protein particles, where obtaining large numbers of images through controlled settings is comparatively straightforward. In this work, we develop a state-of-the-art diffusion model to address data imbalance by generating high-fidelity images that can augment training datasets, enabling the effective training of multi-class deep neural networks. We validate this approach by demonstrating that the generated samples closely resemble real particle images in terms of visual quality and structure. To assess the effectiveness of using diffusion-generated images in training datasets, we conduct large-scale experiments on a validation dataset comprising 500,000 protein particle images and demonstrate that this approach improves classification performance with no negligible downside. Finally, to promote open research and reproducibility, we publicly release both our diffusion models and the trained multi-class deep neural network classifiers, along with a straightforward interface for easy integration into future studies, at https://github.com/utkuozbulak/svp-generative-ai.
>
---
#### [new 061] Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment
- **分类: cs.CV**

- **简介: 论文探讨在不微调下部署扩散模型的计算优化策略，解决减少消噪步骤 vs 降低每步推理成本的权衡问题。提出PostDiff框架，通过输入与模块级冗余减少提升效率，实验表明降低每步成本更优。**

- **链接: [http://arxiv.org/pdf/2508.06160v1](http://arxiv.org/pdf/2508.06160v1)**

> **作者:** Zhenbang Du; Yonggan Fu; Lifu Wang; Jiayi Qian; Xiao Luo; Yingyan; Lin
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Diffusion models have shown remarkable success across generative tasks, yet their high computational demands challenge deployment on resource-limited platforms. This paper investigates a critical question for compute-optimal diffusion model deployment: Under a post-training setting without fine-tuning, is it more effective to reduce the number of denoising steps or to use a cheaper per-step inference? Intuitively, reducing the number of denoising steps increases the variability of the distributions across steps, making the model more sensitive to compression. In contrast, keeping more denoising steps makes the differences smaller, preserving redundancy, and making post-training compression more feasible. To systematically examine this, we propose PostDiff, a training-free framework for accelerating pre-trained diffusion models by reducing redundancy at both the input level and module level in a post-training manner. At the input level, we propose a mixed-resolution denoising scheme based on the insight that reducing generation resolution in early denoising steps can enhance low-frequency components and improve final generation fidelity. At the module level, we employ a hybrid module caching strategy to reuse computations across denoising steps. Extensive experiments and ablation studies demonstrate that (1) PostDiff can significantly improve the fidelity-efficiency trade-off of state-of-the-art diffusion models, and (2) to boost efficiency while maintaining decent generation fidelity, reducing per-step inference cost is often more effective than reducing the number of denoising steps. Our code is available at https://github.com/GATECH-EIC/PostDiff.
>
---
#### [new 062] AGI for the Earth, the path, possibilities and how to evaluate intelligence of models that work with Earth Observation Data?
- **分类: cs.CV; cs.LG**

- **简介: 论文探讨AGI与地球观测数据的关联，指出现有基准无法有效评估模型泛化能力，提出需构建全面任务集以提升地球观测模型智能评估标准。**

- **链接: [http://arxiv.org/pdf/2508.06057v1](http://arxiv.org/pdf/2508.06057v1)**

> **作者:** Mojtaba Valipour; Kelly Zheng; James Lowman; Spencer Szabados; Mike Gartner; Bobby Braswell
>
> **备注:** Accepted in IGARSS 2025!
>
> **摘要:** Artificial General Intelligence (AGI) is closer than ever to becoming a reality, sparking widespread enthusiasm in the research community to collect and work with various modalities, including text, image, video, and audio. Despite recent efforts, satellite spectral imagery, as an additional modality, has yet to receive the attention it deserves. This area presents unique challenges, but also holds great promise in advancing the capabilities of AGI in understanding the natural world. In this paper, we argue why Earth Observation data is useful for an intelligent model, and then we review existing benchmarks and highlight their limitations in evaluating the generalization ability of foundation models in this domain. This paper emphasizes the need for a more comprehensive benchmark to evaluate earth observation models. To facilitate this, we propose a comprehensive set of tasks that a benchmark should encompass to effectively assess a model's ability to understand and interact with Earth observation data.
>
---
#### [new 063] Feature-Space Oversampling for Addressing Class Imbalance in SAR Ship Classification
- **分类: cs.CV**

- **简介: 论文针对SAR船分类中长尾分布问题，提出基于特征空间的过采样方法M2m_f和M2m_u，通过实验验证其有效性，提升分类性能。**

- **链接: [http://arxiv.org/pdf/2508.06420v1](http://arxiv.org/pdf/2508.06420v1)**

> **作者:** Ch Muhammad Awais; Marco Reggiannini; Davide Moroni; Oktay Karakus
>
> **备注:** Accepted and presented at IGARSS
>
> **摘要:** SAR ship classification faces the challenge of long-tailed datasets, which complicates the classification of underrepresented classes. Oversampling methods have proven effective in addressing class imbalance in optical data. In this paper, we evaluated the effect of oversampling in the feature space for SAR ship classification. We propose two novel algorithms inspired by the Major-to-minor (M2m) method M2m$_f$, M2m$_u$. The algorithms are tested on two public datasets, OpenSARShip (6 classes) and FuSARShip (9 classes), using three state-of-the-art models as feature extractors: ViT, VGG16, and ResNet50. Additionally, we also analyzed the impact of oversampling methods on different class sizes. The results demonstrated the effectiveness of our novel methods over the original M2m and baselines, with an average F1-score increase of 8.82% for FuSARShip and 4.44% for OpenSARShip.
>
---
#### [new 064] Q-CLIP: Unleashing the Power of Vision-Language Models for Video Quality Assessment through Unified Cross-Modal Adaptation
- **分类: cs.CV**

- **简介: 论文提出Q-CLIP，通过跨模态适配器和质量提示提升视频质量评估（VQA）性能，解决传统方法资源消耗大、泛化差的问题。**

- **链接: [http://arxiv.org/pdf/2508.06092v1](http://arxiv.org/pdf/2508.06092v1)**

> **作者:** Yachun Mi; Yu Li; Yanting Li; Shixin Sun; Chen Hui; Tong Zhang; Yuanyuan Liu; Chenyue Song; Shaohui Liu
>
> **摘要:** Accurate and efficient Video Quality Assessment (VQA) has long been a key research challenge. Current mainstream VQA methods typically improve performance by pretraining on large-scale classification datasets (e.g., ImageNet, Kinetics-400), followed by fine-tuning on VQA datasets. However, this strategy presents two significant challenges: (1) merely transferring semantic knowledge learned from pretraining is insufficient for VQA, as video quality depends on multiple factors (e.g., semantics, distortion, motion, aesthetics); (2) pretraining on large-scale datasets demands enormous computational resources, often dozens or even hundreds of times greater than training directly on VQA datasets. Recently, Vision-Language Models (VLMs) have shown remarkable generalization capabilities across a wide range of visual tasks, and have begun to demonstrate promising potential in quality assessment. In this work, we propose Q-CLIP, the first fully VLMs-based framework for VQA. Q-CLIP enhances both visual and textual representations through a Shared Cross-Modal Adapter (SCMA), which contains only a minimal number of trainable parameters and is the only component that requires training. This design significantly reduces computational cost. In addition, we introduce a set of five learnable quality-level prompts to guide the VLMs in perceiving subtle quality variations, thereby further enhancing the model's sensitivity to video quality. Furthermore, we investigate the impact of different frame sampling strategies on VQA performance, and find that frame-difference-based sampling leads to better generalization performance across datasets. Extensive experiments demonstrate that Q-CLIP exhibits excellent performance on several VQA datasets.
>
---
#### [new 065] MotionSwap
- **分类: cs.CV**

- **简介: 论文提出改进SimSwap，通过自/交叉注意力、动态损失权重和余弦调度提升面部交换质量，实验显示性能提升。**

- **链接: [http://arxiv.org/pdf/2508.06430v1](http://arxiv.org/pdf/2508.06430v1)**

> **作者:** Om Patil; Jinesh Modi; Suryabha Mukhopadhyay; Meghaditya Giri; Chhavi Malhotra
>
> **备注:** 8 pages, 7 figures, 5 tables. This is a student research submission from BITS Pilani, Hyderabad Campus. Our implementation enhances SimSwap with attention modules and dynamic training strategies
>
> **摘要:** Face swapping technology has gained significant attention in both academic research and commercial applications. This paper presents our implementation and enhancement of SimSwap, an efficient framework for high fidelity face swapping. We introduce several improvements to the original model, including the integration of self and cross-attention mechanisms in the generator architecture, dynamic loss weighting, and cosine annealing learning rate scheduling. These enhancements lead to significant improvements in identity preservation, attribute consistency, and overall visual quality. Our experimental results, spanning 400,000 training iterations, demonstrate progressive improvements in generator and discriminator performance. The enhanced model achieves better identity similarity, lower FID scores, and visibly superior qualitative results compared to the baseline. Ablation studies confirm the importance of each architectural and training improvement. We conclude by identifying key future directions, such as integrating StyleGAN3, improving lip synchronization, incorporating 3D facial modeling, and introducing temporal consistency for video-based applications.
>
---
#### [new 066] Fast Motion Estimation and Context-Aware Refinement for Efficient Bayer-Domain Video Vision
- **分类: cs.CV**

- **简介: 该论文提出一种高效视频计算机视觉系统，通过移除图像信号处理器、采用快速块匹配运动估计并引入上下文感知精炼网络，解决时间冗余与前端计算开销问题，实现加速同时保持性能。**

- **链接: [http://arxiv.org/pdf/2508.05990v1](http://arxiv.org/pdf/2508.05990v1)**

> **作者:** Haichao Wang; Xinyue Xi; Jiangtao Wen; Yuxing Han
>
> **摘要:** The efficiency of video computer vision system remains a challenging task due to the high temporal redundancy inside a video. Existing works have been proposed for efficient vision computer vision. However, they do not fully reduce the temporal redundancy and neglect the front end computation overhead. In this paper, we propose an efficient video computer vision system. First, image signal processor is removed and Bayer-format data is directly fed into video computer vision models, thus saving the front end computation. Second, instead of optical flow models and video codecs, a fast block matching-based motion estimation algorithm is proposed specifically for efficient video computer vision, with a MV refinement module. To correct the error, context-aware block refinement network is introduced to refine regions with large error. To further balance the accuracy and efficiency, a frame selection strategy is employed. Experiments on multiple video computer vision tasks demonstrate that our method achieves significant acceleration with slight performance loss.
>
---
#### [new 067] DreamVE: Unified Instruction-based Image and Video Editing
- **分类: cs.CV**

- **简介: 论文提出基于指令的统一图像与视频编辑模型DreamVE，解决视频编辑因数据不足而受限的问题，通过两阶段训练（图像→视频）结合拼贴与生成模型数据合成，提升编辑效率与泛化能力，并采用T2V模型优化引导机制。**

- **链接: [http://arxiv.org/pdf/2508.06080v1](http://arxiv.org/pdf/2508.06080v1)**

> **作者:** Bin Xia; Jiyang Liu; Yuechen Zhang; Bohao Peng; Ruihang Chu; Yitong Wang; Xinglong Wu; Bei Yu; Jiaya Jia
>
> **摘要:** Instruction-based editing holds vast potential due to its simple and efficient interactive editing format. However, instruction-based editing, particularly for video, has been constrained by limited training data, hindering its practical application. To this end, we introduce DreamVE, a unified model for instruction-based image and video editing. Specifically, We propose a two-stage training strategy: first image editing, then video editing. This offers two main benefits: (1) Image data scales more easily, and models are more efficient to train, providing useful priors for faster and better video editing training. (2) Unifying image and video generation is natural and aligns with current trends. Moreover, we present comprehensive training data synthesis pipelines, including collage-based and generative model-based data synthesis. The collage-based data synthesis combines foreground objects and backgrounds to generate diverse editing data, such as object manipulation, background changes, and text modifications. It can easily generate billions of accurate, consistent, realistic, and diverse editing pairs. We pretrain DreamVE on extensive collage-based data to achieve strong performance in key editing types and enhance generalization and transfer capabilities. However, collage-based data lacks some attribute editing cases, leading to a relative drop in performance. In contrast, the generative model-based pipeline, despite being hard to scale up, offers flexibility in handling attribute editing cases. Therefore, we use generative model-based data to further fine-tune DreamVE. Besides, we design an efficient and powerful editing framework for DreamVE. We build on the SOTA T2V model and use a token concatenation with early drop approach to inject source image guidance, ensuring strong consistency and editability. The codes and models will be released.
>
---
#### [new 068] MathReal: We Keep It Real! A Real Scene Benchmark for Evaluating Math Reasoning in Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 论文提出MathReal数据集，用于评估多模态大语言模型在真实场景中的数学推理能力，解决现有基准缺乏真实图像的问题，通过分类和实验分析模型性能。**

- **链接: [http://arxiv.org/pdf/2508.06009v1](http://arxiv.org/pdf/2508.06009v1)**

> **作者:** Jun Feng; Zixin Wang; Zhentao Zhang; Yue Guo; Zhihan Zhou; Xiuyi Chen; Zhenyang Li; Dawei Yin
>
> **备注:** 29 pages, 16 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in visual mathematical reasoning across various existing benchmarks. However, these benchmarks are predominantly based on clean or processed multimodal inputs, without incorporating the images provided by real-world Kindergarten through 12th grade (K-12) educational users. To address this gap, we introduce MathReal, a meticulously curated dataset comprising 2,000 mathematical questions with images captured by handheld mobile devices in authentic scenarios. Each question is an image, containing the question text and visual element. We systematically classify the real images into three primary categories: image quality degradation, perspective variation, and irrelevant content interference, which are further delineated into 14 subcategories. Additionally, MathReal spans five core knowledge and ability categories, which encompass three question types and are divided into three difficulty levels. To comprehensively evaluate the multimodal mathematical reasoning abilities of state-of-the-art MLLMs in real-world scenarios, we design six experimental settings that enable a systematic analysis of their performance. Through extensive experimentation, we find that the problem-solving abilities of existing MLLMs are significantly challenged in realistic educational contexts. Based on this, we conduct a thorough analysis of their performance and error patterns, providing insights into their recognition, comprehension, and reasoning capabilities, and outlining directions for future improvements. Data and code: https://github.com/junfeng0288/MathReal.
>
---
#### [new 069] VISTAR:A User-Centric and Role-Driven Benchmark for Text-to-Image Evaluation
- **分类: cs.CV**

- **简介: 论文提出用户驱动的文本到图像评估基准VISTAR，通过多维度指标与HWPQ方案解决现有评估不足，结合德尔菲研究构建2845个验证数据集，提升抽象语义评估精度，支持领域化部署。**

- **链接: [http://arxiv.org/pdf/2508.06152v1](http://arxiv.org/pdf/2508.06152v1)**

> **作者:** Kaiyuan Jiang; Ruoxi Sun; Ying Cao; Yuqi Xu; Xinran Zhang; Junyan Guo; ChengSheng Deng
>
> **备注:** 17 pages,8 figures
>
> **摘要:** We present VISTAR, a user-centric, multi-dimensional benchmark for text-to-image (T2I) evaluation that addresses the limitations of existing metrics. VISTAR introduces a two-tier hybrid paradigm: it employs deterministic, scriptable metrics for physically quantifiable attributes (e.g., text rendering, lighting) and a novel Hierarchical Weighted P/N Questioning (HWPQ) scheme that uses constrained vision-language models to assess abstract semantics (e.g., style fusion, cultural fidelity). Grounded in a Delphi study with 120 experts, we defined seven user roles and nine evaluation angles to construct the benchmark, which comprises 2,845 prompts validated by over 15,000 human pairwise comparisons. Our metrics achieve high human alignment (>75%), with the HWPQ scheme reaching 85.9% accuracy on abstract semantics, significantly outperforming VQA baselines. Comprehensive evaluation of state-of-the-art models reveals no universal champion, as role-weighted scores reorder rankings and provide actionable guidance for domain-specific deployment. All resources are publicly released to foster reproducible T2I assessment.
>
---
#### [new 070] Effective Training Data Synthesis for Improving MLLM Chart Understanding
- **分类: cs.CV; cs.CL**

- **简介: 论文提出五步数据合成方法，生成ECD数据集以提升MLLM图表理解能力，解决现有合成图表与真实图表相似性不足的问题。**

- **链接: [http://arxiv.org/pdf/2508.06492v1](http://arxiv.org/pdf/2508.06492v1)**

> **作者:** Yuwei Yang; Zeyu Zhang; Yunzhong Hou; Zhuowan Li; Gaowen Liu; Ali Payani; Yuan-Sen Ting; Liang Zheng
>
> **备注:** Accepted by ICCV 2025 (poster). 26 pages, 17 figures
>
> **摘要:** Being able to effectively read scientific plots, or chart understanding, is a central part toward building effective agents for science. However, existing multimodal large language models (MLLMs), especially open-source ones, are still falling behind with a typical success rate of 30%-50% on challenging benchmarks. Previous studies on fine-tuning MLLMs with synthetic charts are often restricted by their inadequate similarity to the real charts, which could compromise model training and performance on complex real-world charts. In this study, we show that modularizing chart generation and diversifying visual details improves chart understanding capabilities. In particular, we design a five-step data synthesis pipeline, where we separate data and function creation for single plot generation, condition the generation of later subplots on earlier ones for multi-subplot figures, visually diversify the generated figures, filter out low quality data, and finally generate the question-answer (QA) pairs with GPT-4o. This approach allows us to streamline the generation of fine-tuning datasets and introduce the effective chart dataset (ECD), which contains 10k+ chart images and 300k+ QA pairs, covering 25 topics and featuring 250+ chart type combinations with high visual complexity. We show that ECD consistently improves the performance of various MLLMs on a range of real-world and synthetic test sets. Code, data and models are available at: https://github.com/yuweiyang-anu/ECD.
>
---
#### [new 071] Uncertainty-quantified Rollout Policy Adaptation for Unlabelled Cross-domain Temporal Grounding
- **分类: cs.CV**

- **简介: 论文提出跨域视频时间定位任务，解决无标注域适应难题，通过URPA结合GRPO生成候选预测并加权训练，降低标注需求与计算开销，支持实时部署。**

- **链接: [http://arxiv.org/pdf/2508.06317v1](http://arxiv.org/pdf/2508.06317v1)**

> **作者:** Jian Hu; Zixu Cheng; Shaogang Gong; Isabel Guan; Jianye Hao; Jun Wang; Kun Shao
>
> **摘要:** Video Temporal Grounding (TG) aims to temporally locate video segments matching a natural language description (a query) in a long video. While Vision-Language Models (VLMs) are effective at holistic semantic matching, they often struggle with fine-grained temporal localisation. Recently, Group Relative Policy Optimisation (GRPO) reformulates the inference process as a reinforcement learning task, enabling fine-grained grounding and achieving strong in-domain performance. However, GRPO relies on labelled data, making it unsuitable in unlabelled domains. Moreover, because videos are large and expensive to store and process, performing full-scale adaptation introduces prohibitive latency and computational overhead, making it impractical for real-time deployment. To overcome both problems, we introduce a Data-Efficient Unlabelled Cross-domain Temporal Grounding method, from which a model is first trained on a labelled source domain, then adapted to a target domain using only a small number of unlabelled videos from the target domain. This approach eliminates the need for target annotation and keeps both computational and storage overhead low enough to run in real time. Specifically, we introduce. Uncertainty-quantified Rollout Policy Adaptation (URPA) for cross-domain knowledge transfer in learning video temporal grounding without target labels. URPA generates multiple candidate predictions using GRPO rollouts, averages them to form a pseudo label, and estimates confidence from the variance across these rollouts. This confidence then weights the training rewards, guiding the model to focus on reliable supervision. Experiments on three datasets across six cross-domain settings show that URPA generalises well using only a few unlabelled target videos. Codes will be released once published.
>
---
#### [new 072] TSMS-SAM2: Multi-scale Temporal Sampling Augmentation and Memory-Splitting Pruning for Promptable Video Object Segmentation and Tracking in Surgical Scenarios
- **分类: cs.CV**

- **简介: 该论文提出TSMS-SAM2框架，针对手术视频中运动快速和内存冗余问题，通过多时序尺度采样增强和内存分割剪枝提升视频对象分割与跟踪性能，实现高精度分割（Dice分数达95.24）。**

- **链接: [http://arxiv.org/pdf/2508.05829v1](http://arxiv.org/pdf/2508.05829v1)**

> **作者:** Guoping Xu; Hua-Chieh Shao; You Zhang
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** Promptable video object segmentation and tracking (VOST) has seen significant advances with the emergence of foundation models like Segment Anything Model 2 (SAM2); however, their application in surgical video analysis remains challenging due to complex motion dynamics and the redundancy of memory that impedes effective learning. In this work, we propose TSMS-SAM2, a novel framework that enhances promptable VOST in surgical videos by addressing challenges of rapid object motion and memory redundancy in SAM2. TSMS-SAM2 introduces two key strategies: multi-temporal-scale video sampling augmentation to improve robustness against motion variability, and a memory splitting and pruning mechanism that organizes and filters past frame features for more efficient and accurate segmentation. Evaluated on EndoVis2017 and EndoVis2018 datasets, TSMS-SAM2 achieved the highest mean Dice scores of 95.24 and 86.73, respectively, outperforming prior SAM-based and task-specific methods. Extensive ablation studies confirm the effectiveness of multiscale temporal augmentation and memory splitting, highlighting the framework's potential for robust, efficient segmentation in complex surgical scenarios. Our source code will be available at https://github.com/apple1986/TSMS-SAM2.
>
---
#### [new 073] FMCE-Net++: Feature Map Convergence Evaluation and Training
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FMCE-Net++框架，通过引入预训练FMCE-Net作为辅助头，结合特征收敛损失优化模型性能，解决DNN可解释性问题，实验证明能有效提升模型表现。**

- **链接: [http://arxiv.org/pdf/2508.06109v1](http://arxiv.org/pdf/2508.06109v1)**

> **作者:** Zhibo Zhu; Renyu Huang; Lei He
>
> **摘要:** Deep Neural Networks (DNNs) face interpretability challenges due to their opaque internal representations. While Feature Map Convergence Evaluation (FMCE) quantifies module-level convergence via Feature Map Convergence Scores (FMCS), it lacks experimental validation and closed-loop integration. To address this limitation, we propose FMCE-Net++, a novel training framework that integrates a pretrained, frozen FMCE-Net as an auxiliary head. This module generates FMCS predictions, which, combined with task labels, jointly supervise backbone optimization through a Representation Auxiliary Loss. The RAL dynamically balances the primary classification loss and feature convergence optimization via a tunable \Representation Abstraction Factor. Extensive experiments conducted on MNIST, CIFAR-10, FashionMNIST, and CIFAR-100 demonstrate that FMCE-Net++ consistently enhances model performance without architectural modifications or additional data. Key experimental outcomes include accuracy gains of $+1.16$ pp (ResNet-50/CIFAR-10) and $+1.08$ pp (ShuffleNet v2/CIFAR-100), validating that FMCE-Net++ can effectively elevate state-of-the-art performance ceilings.
>
---
#### [new 074] Multi-view Gaze Target Estimation
- **分类: cs.CV**

- **简介: 论文提出多视角注视目标估计方法，解决单视角方法的遮挡、模糊及出视图问题，通过融合多视角信息与三模块（HIA、UGS、ESA）提升精度，并构建多视角数据集。**

- **链接: [http://arxiv.org/pdf/2508.05857v1](http://arxiv.org/pdf/2508.05857v1)**

> **作者:** Qiaomu Miao; Vivek Raju Golani; Jingyi Xu; Progga Paromita Dutta; Minh Hoai; Dimitris Samaras
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** This paper presents a method that utilizes multiple camera views for the gaze target estimation (GTE) task. The approach integrates information from different camera views to improve accuracy and expand applicability, addressing limitations in existing single-view methods that face challenges such as face occlusion, target ambiguity, and out-of-view targets. Our method processes a pair of camera views as input, incorporating a Head Information Aggregation (HIA) module for leveraging head information from both views for more accurate gaze estimation, an Uncertainty-based Gaze Selection (UGS) for identifying the most reliable gaze output, and an Epipolar-based Scene Attention (ESA) module for cross-view background information sharing. This approach significantly outperforms single-view baselines, especially when the second camera provides a clear view of the person's face. Additionally, our method can estimate the gaze target in the first view using the image of the person in the second view only, a capability not possessed by single-view GTE methods. Furthermore, the paper introduces a multi-view dataset for developing and evaluating multi-view GTE methods. Data and code are available at https://www3.cs.stonybrook.edu/~cvl/multiview_gte.html
>
---
#### [new 075] SwiftVideo: A Unified Framework for Few-Step Video Generation through Trajectory-Distribution Alignment
- **分类: cs.CV**

- **简介: 论文提出SwiftVideo框架，通过轨迹-分布对齐解决少步视频生成中的性能与质量矛盾，结合连续时间一致性蒸馏与双视角对齐，减少推理步骤并提升生成质量。**

- **链接: [http://arxiv.org/pdf/2508.06082v1](http://arxiv.org/pdf/2508.06082v1)**

> **作者:** Yanxiao Sun; Jiafu Wu; Yun Cao; Chengming Xu; Yabiao Wang; Weijian Cao; Donghao Luo; Chengjie Wang; Yanwei Fu
>
> **摘要:** Diffusion-based or flow-based models have achieved significant progress in video synthesis but require multiple iterative sampling steps, which incurs substantial computational overhead. While many distillation methods that are solely based on trajectory-preserving or distribution-matching have been developed to accelerate video generation models, these approaches often suffer from performance breakdown or increased artifacts under few-step settings. To address these limitations, we propose \textbf{\emph{SwiftVideo}}, a unified and stable distillation framework that combines the advantages of trajectory-preserving and distribution-matching strategies. Our approach introduces continuous-time consistency distillation to ensure precise preservation of ODE trajectories. Subsequently, we propose a dual-perspective alignment that includes distribution alignment between synthetic and real data along with trajectory alignment across different inference steps. Our method maintains high-quality video generation while substantially reducing the number of inference steps. Quantitative evaluations on the OpenVid-1M benchmark demonstrate that our method significantly outperforms existing approaches in few-step video generation.
>
---
#### [new 076] MCA: 2D-3D Retrieval with Noisy Labels via Multi-level Adaptive Correction and Alignment
- **分类: cs.CV**

- **简介: 该论文提出MCA框架，解决2D-3D跨模态检索中噪声标签问题，通过多级自适应纠正与对齐提升性能。**

- **链接: [http://arxiv.org/pdf/2508.06104v1](http://arxiv.org/pdf/2508.06104v1)**

> **作者:** Gui Zou; Chaofan Gan; Chern Hong Lim; Supavadee Aramvith; Weiyao Lin
>
> **备注:** ICMEW 2025
>
> **摘要:** With the increasing availability of 2D and 3D data, significant advancements have been made in the field of cross-modal retrieval. Nevertheless, the existence of imperfect annotations presents considerable challenges, demanding robust solutions for 2D-3D cross-modal retrieval in the presence of noisy label conditions. Existing methods generally address the issue of noise by dividing samples independently within each modality, making them susceptible to overfitting on corrupted labels. To address these issues, we propose a robust 2D-3D \textbf{M}ulti-level cross-modal adaptive \textbf{C}orrection and \textbf{A}lignment framework (MCA). Specifically, we introduce a Multimodal Joint label Correction (MJC) mechanism that leverages multimodal historical self-predictions to jointly model the modality prediction consistency, enabling reliable label refinement. Additionally, we propose a Multi-level Adaptive Alignment (MAA) strategy to effectively enhance cross-modal feature semantics and discrimination across different levels. Extensive experiments demonstrate the superiority of our method, MCA, which achieves state-of-the-art performance on both conventional and realistic noisy 3D benchmarks, highlighting its generality and effectiveness.
>
---
#### [new 077] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation
- **分类: cs.CV**

- **简介: 论文针对3D重建中稀疏视角生成效率低的问题，提出FVGen框架，通过对抗生成网络蒸馏多步骤VDMs为少步骤模型，实现快速新颖视角合成，采样速度提升90%。**

- **链接: [http://arxiv.org/pdf/2508.06392v1](http://arxiv.org/pdf/2508.06392v1)**

> **作者:** Wenbin Teng; Gonglin Chen; Haiwei Chen; Yajie Zhao
>
> **摘要:** Recent progress in 3D reconstruction has enabled realistic 3D models from dense image captures, yet challenges persist with sparse views, often leading to artifacts in unseen areas. Recent works leverage Video Diffusion Models (VDMs) to generate dense observations, filling the gaps when only sparse views are available for 3D reconstruction tasks. A significant limitation of these methods is their slow sampling speed when using VDMs. In this paper, we present FVGen, a novel framework that addresses this challenge by enabling fast novel view synthesis using VDMs in as few as four sampling steps. We propose a novel video diffusion model distillation method that distills a multi-step denoising teacher model into a few-step denoising student model using Generative Adversarial Networks (GANs) and softened reverse KL-divergence minimization. Extensive experiments on real-world datasets show that, compared to previous works, our framework generates the same number of novel views with similar (or even better) visual quality while reducing sampling time by more than 90%. FVGen significantly improves time efficiency for downstream reconstruction tasks, particularly when working with sparse input views (more than 2) where pre-trained VDMs need to be run multiple times to achieve better spatial coverage.
>
---
#### [new 078] Text-guided Visual Prompt DINO for Generic Segmentation
- **分类: cs.CV**

- **简介: 论文提出Prompt-DINO框架，通过早期融合、顺序对齐和生成数据引擎解决开放世界分割中的语义模糊和词汇限制问题，提升性能并减少标签噪声。**

- **链接: [http://arxiv.org/pdf/2508.06146v1](http://arxiv.org/pdf/2508.06146v1)**

> **作者:** Yuchen Guan; Chong Sun; Canmiao Fu; Zhipeng Huang; Chun Yuan; Chen Li
>
> **摘要:** Recent advancements in multimodal vision models have highlighted limitations in late-stage feature fusion and suboptimal query selection for hybrid prompts open-world segmentation, alongside constraints from caption-derived vocabularies. To address these challenges, we propose Prompt-DINO, a text-guided visual Prompt DINO framework featuring three key innovations. First, we introduce an early fusion mechanism that unifies text/visual prompts and backbone features at the initial encoding stage, enabling deeper cross-modal interactions to resolve semantic ambiguities. Second, we design order-aligned query selection for DETR-based architectures, explicitly optimizing the structural alignment between text and visual queries during decoding to enhance semantic-spatial consistency. Third, we develop a generative data engine powered by the Recognize Anything via Prompting (RAP) model, which synthesizes 0.5B diverse training instances through a dual-path cross-verification pipeline, reducing label noise by 80.5% compared to conventional approaches. Extensive experiments demonstrate that Prompt-DINO achieves state-of-the-art performance on open-world detection benchmarks while significantly expanding semantic coverage beyond fixed-vocabulary constraints. Our work establishes a new paradigm for scalable multimodal detection and data generation in open-world scenarios. Data&Code are available at https://github.com/WeChatCV/WeVisionOne.
>
---
#### [new 079] Mask & Match: Learning to Recognize Handwritten Math with Self-Supervised Attention
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种自监督学习框架，用于识别手写数学表达式（HMER），解决数据稀缺与结构复杂问题，通过自监督注意力机制学习关键符号区域，提升识别性能。**

- **链接: [http://arxiv.org/pdf/2508.06107v1](http://arxiv.org/pdf/2508.06107v1)**

> **作者:** Shree Mitra; Ritabrata Chakraborty; Nilkanta Sahu
>
> **摘要:** Recognizing handwritten mathematical expressions (HMER) is a challenging task due to the inherent two-dimensional structure, varying symbol scales, and complex spatial relationships among symbols. In this paper, we present a self-supervised learning (SSL) framework for HMER that eliminates the need for expensive labeled data. Our approach begins by pretraining an image encoder using a combination of global and local contrastive loss, enabling the model to learn both holistic and fine-grained representations. A key contribution of this work is a novel self-supervised attention network, which is trained using a progressive spatial masking strategy. This attention mechanism is designed to learn semantically meaningful focus regions, such as operators, exponents, and nested mathematical notation, without requiring any supervision. The progressive masking curriculum encourages the network to become increasingly robust to missing or occluded visual information, ultimately improving structural understanding. Our complete pipeline consists of (1) self-supervised pretraining of the encoder, (2) self-supervised attention learning, and (3) supervised fine-tuning with a transformer decoder to generate LATEX sequences. Extensive experiments on CROHME benchmarks demonstrate that our method outperforms existing SSL and fully supervised baselines, validating the effectiveness of our progressive attention mechanism in enhancing HMER performance. Our codebase can be found here.
>
---
#### [new 080] AdaptInfer: Adaptive Token Pruning for Vision-Language Model Inference with Dynamical Text Guidance
- **分类: cs.CV**

- **简介: 论文提出AdaptInfer框架，针对VLM推理中视觉token剪枝问题，通过动态文本引导机制与跨模态注意力分析，实现高效轻量化的剪枝调度，降低CUDA延迟并提升准确率。**

- **链接: [http://arxiv.org/pdf/2508.06084v1](http://arxiv.org/pdf/2508.06084v1)**

> **作者:** Weichen Zhang; Zhui Zhu; Ningbo Li; Kebin Liu; Yunhao Liu
>
> **摘要:** Vision-language models (VLMs) have achieved impressive performance on multimodal reasoning tasks such as visual question answering (VQA), but their inference cost remains a significant challenge due to the large number of vision tokens processed during the prefill stage. Existing pruning methods often rely on directly using the attention patterns or static text prompt guidance, failing to exploit the dynamic internal signals generated during inference. To address these issues, we propose AdaptInfer, a plug-and-play framework for adaptive vision token pruning in VLMs. First, we introduce a fine-grained, dynamic text-guided pruning mechanism that reuses layer-wise text-to-text attention maps to construct soft priors over text-token importance, allowing more informed scoring of vision tokens at each stage. Second, we perform an offline analysis of cross-modal attention shifts and identify consistent inflection locations in inference, which inspire us to propose a more principled and efficient pruning schedule. Our method is lightweight and plug-and-play, also generalizable across multi-modal tasks. Experimental results have verified the effectiveness of the proposed method. For example, it reduces CUDA latency by 61.3\% while maintaining an average accuracy of 92.9\% on vanilla LLaVA-1.5-7B. Under the same token budget, AdaptInfer surpasses SOTA in accuracy.
>
---
#### [new 081] AnomalyMoE: Towards a Language-free Generalist Model for Unified Visual Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文提出AnomalyMoE模型，解决跨模态视觉异常检测专用性问题，通过分层专家网络与EIR/ESB模块提升泛化能力，实现统一检测。**

- **链接: [http://arxiv.org/pdf/2508.06203v1](http://arxiv.org/pdf/2508.06203v1)**

> **作者:** Zhaopeng Gu; Bingke Zhu; Guibo Zhu; Yingying Chen; Wei Ge; Ming Tang; Jinqiao Wang
>
> **摘要:** Anomaly detection is a critical task across numerous domains and modalities, yet existing methods are often highly specialized, limiting their generalizability. These specialized models, tailored for specific anomaly types like textural defects or logical errors, typically exhibit limited performance when deployed outside their designated contexts. To overcome this limitation, we propose AnomalyMoE, a novel and universal anomaly detection framework based on a Mixture-of-Experts (MoE) architecture. Our key insight is to decompose the complex anomaly detection problem into three distinct semantic hierarchies: local structural anomalies, component-level semantic anomalies, and global logical anomalies. AnomalyMoE correspondingly employs three dedicated expert networks at the patch, component, and global levels, and is specialized in reconstructing features and identifying deviations at its designated semantic level. This hierarchical design allows a single model to concurrently understand and detect a wide spectrum of anomalies. Furthermore, we introduce an Expert Information Repulsion (EIR) module to promote expert diversity and an Expert Selection Balancing (ESB) module to ensure the comprehensive utilization of all experts. Experiments on 8 challenging datasets spanning industrial imaging, 3D point clouds, medical imaging, video surveillance, and logical anomaly detection demonstrate that AnomalyMoE establishes new state-of-the-art performance, significantly outperforming specialized methods in their respective domains.
>
---
#### [new 082] E-React: Towards Emotionally Controlled Synthesis of Human Reactions
- **分类: cs.CV**

- **简介: 论文提出一种基于半监督情感先验的演员-反应扩散模型，解决情绪影响下的反应生成问题，通过有限数据学习情绪表示并生成自然反应，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.06093v1](http://arxiv.org/pdf/2508.06093v1)**

> **作者:** Chen Zhu; Buzhen Huang; Zijing Wu; Binghui Zuo; Yangang Wang
>
> **摘要:** Emotion serves as an essential component in daily human interactions. Existing human motion generation frameworks do not consider the impact of emotions, which reduces naturalness and limits their application in interactive tasks, such as human reaction synthesis. In this work, we introduce a novel task: generating diverse reaction motions in response to different emotional cues. However, learning emotion representation from limited motion data and incorporating it into a motion generation framework remains a challenging problem. To address the above obstacles, we introduce a semi-supervised emotion prior in an actor-reactor diffusion model to facilitate emotion-driven reaction synthesis. Specifically, based on the observation that motion clips within a short sequence tend to share the same emotion, we first devise a semi-supervised learning framework to train an emotion prior. With this prior, we further train an actor-reactor diffusion model to generate reactions by considering both spatial interaction and emotional response. Finally, given a motion sequence of an actor, our approach can generate realistic reactions under various emotional conditions. Experimental results demonstrate that our model outperforms existing reaction generation methods. The code and data will be made publicly available at https://ereact.github.io/
>
---
#### [new 083] LV-Net: Anatomy-aware lateral ventricle shape modeling with a case study on Alzheimer's disease, the Australian Imaging Biomarkers and Lifestyle flagship study of ageing
- **分类: cs.CV; cs.GR**

- **简介: 论文提出LV-Net框架，通过结合解剖关系的联合LV-海马体模板网格，解决侧脑室形状建模中分割误差和形状变异问题，应用于阿尔茨海默病研究，提升形态统计准确性。**

- **链接: [http://arxiv.org/pdf/2508.06055v1](http://arxiv.org/pdf/2508.06055v1)**

> **作者:** Wonjung Park; Suhyun Ahn; Jinah Park
>
> **摘要:** Lateral ventricle (LV) shape analysis holds promise as a biomarker for neurological diseases; however, challenges remain due to substantial shape variability across individuals and segmentation difficulties arising from limited MRI resolution. We introduce LV-Net, a novel framework for producing individualized 3D LV meshes from brain MRI by deforming an anatomy-aware joint LV-hippocampus template mesh. By incorporating anatomical relationships embedded within the joint template, LV-Net reduces boundary segmentation artifacts and improves reconstruction robustness. In addition, by classifying the vertices of the template mesh based on their anatomical adjacency, our method enhances point correspondence across subjects, leading to more accurate LV shape statistics. We demonstrate that LV-Net achieves superior reconstruction accuracy, even in the presence of segmentation imperfections, and delivers more reliable shape descriptors across diverse datasets. Finally, we apply LV-Net to Alzheimer's disease analysis, identifying LV subregions that show significantly associations with the disease relative to cognitively normal controls. The codes for LV shape modeling are available at https://github.com/PWonjung/LV_Shape_Modeling.
>
---
#### [new 084] Street View Sociability: Interpretable Analysis of Urban Social Behavior Across 15 Cities
- **分类: cs.CV; cs.SI**

- **简介: 论文利用街景图像分析城市社交行为，应用社会学理论验证社交互动与环境因素的关系，提出街景图像可作为研究工具。**

- **链接: [http://arxiv.org/pdf/2508.06342v1](http://arxiv.org/pdf/2508.06342v1)**

> **作者:** Kieran Elrod; Katherine Flanigan; Mario Bergés
>
> **摘要:** Designing socially active streets has long been a goal of urban planning, yet existing quantitative research largely measures pedestrian volume rather than the quality of social interactions. We hypothesize that street view imagery -- an inexpensive data source with global coverage -- contains latent social information that can be extracted and interpreted through established social science theory. As a proof of concept, we analyzed 2,998 street view images from 15 cities using a multimodal large language model guided by Mehta's taxonomy of passive, fleeting, and enduring sociability -- one illustrative example of a theory grounded in urban design that could be substituted or complemented by other sociological frameworks. We then used linear regression models, controlling for factors like weather, time of day, and pedestrian counts, to test whether the inferred sociability measures correlate with city-level place attachment scores from the World Values Survey and with environmental predictors (e.g., green, sky, and water view indices) derived from individual street view images. Results aligned with long-standing urban planning theory: the sky view index was associated with all three sociability types, the green view index predicted enduring sociability, and place attachment was positively associated with fleeting sociability. These results provide preliminary evidence that street view images can be used to infer relationships between specific types of social interactions and built environment variables. Further research could establish street view imagery as a scalable, privacy-preserving tool for studying urban sociability, enabling cross-cultural theory testing and evidence-based design of socially vibrant cities.
>
---
#### [new 085] GMF-Drive: Gated Mamba Fusion with Spatial-Aware BEV Representation for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 论文提出GMF-Drive框架，解决自动驾驶中Transformer融合的高计算复杂度和空间先验不足问题，通过几何增强LiDAR表示与高效SSM融合，提升BEV场景建模能力。**

- **链接: [http://arxiv.org/pdf/2508.06113v1](http://arxiv.org/pdf/2508.06113v1)**

> **作者:** Jian Wang; Chaokang Jiang; Haitao Xu
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Diffusion-based models are redefining the state-of-the-art in end-to-end autonomous driving, yet their performance is increasingly hampered by a reliance on transformer-based fusion. These architectures face fundamental limitations: quadratic computational complexity restricts the use of high-resolution features, and a lack of spatial priors prevents them from effectively modeling the inherent structure of Bird's Eye View (BEV) representations. This paper introduces GMF-Drive (Gated Mamba Fusion for Driving), an end-to-end framework that overcomes these challenges through two principled innovations. First, we supersede the information-limited histogram-based LiDAR representation with a geometrically-augmented pillar format encoding shape descriptors and statistical features, preserving critical 3D geometric details. Second, we propose a novel hierarchical gated mamba fusion (GM-Fusion) architecture that substitutes an expensive transformer with a highly efficient, spatially-aware state-space model (SSM). Our core BEV-SSM leverages directional sequencing and adaptive fusion mechanisms to capture long-range dependencies with linear complexity, while explicitly respecting the unique spatial properties of the driving scene. Extensive experiments on the challenging NAVSIM benchmark demonstrate that GMF-Drive achieves a new state-of-the-art performance, significantly outperforming DiffusionDrive. Comprehensive ablation studies validate the efficacy of each component, demonstrating that task-specific SSMs can surpass a general-purpose transformer in both performance and efficiency for autonomous driving.
>
---
#### [new 086] DiffCap: Diffusion-based Real-time Human Motion Capture using Sparse IMUs and a Monocular Camera
- **分类: cs.CV**

- **简介: 该论文提出一种基于扩散模型的实时人体运动捕捉方法，融合稀疏IMU与单目相机数据，解决多模态信号融合与实时性问题，实现高效姿态估计。**

- **链接: [http://arxiv.org/pdf/2508.06139v1](http://arxiv.org/pdf/2508.06139v1)**

> **作者:** Shaohua Pan; Xinyu Yi; Yan Zhou; Weihua Jian; Yuan Zhang; Pengfei Wan; Feng Xu
>
> **摘要:** Combining sparse IMUs and a monocular camera is a new promising setting to perform real-time human motion capture. This paper proposes a diffusion-based solution to learn human motion priors and fuse the two modalities of signals together seamlessly in a unified framework. By delicately considering the characteristics of the two signals, the sequential visual information is considered as a whole and transformed into a condition embedding, while the inertial measurement is concatenated with the noisy body pose frame by frame to construct a sequential input for the diffusion model. Firstly, we observe that the visual information may be unavailable in some frames due to occlusions or subjects moving out of the camera view. Thus incorporating the sequential visual features as a whole to get a single feature embedding is robust to the occasional degenerations of visual information in those frames. On the other hand, the IMU measurements are robust to occlusions and always stable when signal transmission has no problem. So incorporating them frame-wisely could better explore the temporal information for the system. Experiments have demonstrated the effectiveness of the system design and its state-of-the-art performance in pose estimation compared with the previous works. Our codes are available for research at https://shaohua-pan.github.io/diffcap-page.
>
---
#### [new 087] Can Large Models Fool the Eye? A New Turing Test for Biological Animation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出BioMotion Arena框架，通过视觉动画评估大模型与多模态模型的生物运动表现，解决现有基准缺乏直观反馈问题，揭示模型在生成生物运动上的差距。**

- **链接: [http://arxiv.org/pdf/2508.06072v1](http://arxiv.org/pdf/2508.06072v1)**

> **作者:** Zijian Chen; Lirong Deng; Zhengyu Chen; Kaiwei Zhang; Qi Jia; Yuan Tian; Yucheng Zhu; Guangtao Zhai
>
> **备注:** 24 pages, 10 figures
>
> **摘要:** Evaluating the abilities of large models and manifesting their gaps are challenging. Current benchmarks adopt either ground-truth-based score-form evaluation on static datasets or indistinct textual chatbot-style human preferences collection, which may not provide users with immediate, intuitive, and perceptible feedback on performance differences. In this paper, we introduce BioMotion Arena, a novel framework for evaluating large language models (LLMs) and multimodal large language models (MLLMs) via visual animation. Our methodology draws inspiration from the inherent visual perception of motion patterns characteristic of living organisms that utilizes point-light source imaging to amplify the performance discrepancies between models. Specifically, we employ a pairwise comparison evaluation and collect more than 45k votes for 53 mainstream LLMs and MLLMs on 90 biological motion variants. Data analyses show that the crowd-sourced human votes are in good agreement with those of expert raters, demonstrating the superiority of our BioMotion Arena in offering discriminative feedback. We also find that over 90\% of evaluated models, including the cutting-edge open-source InternVL3 and proprietary Claude-4 series, fail to produce fundamental humanoid point-light groups, much less smooth and biologically plausible motions. This enables BioMotion Arena to serve as a challenging benchmark for performance visualization and a flexible evaluation framework without restrictions on ground-truth.
>
---
#### [new 088] PA-HOI: A Physics-Aware Human and Object Interaction Dataset
- **分类: cs.CV**

- **简介: 论文提出PA-HOI数据集，聚焦物理属性对人类-物体交互的影响，解决现有数据忽略物体物理特性导致的研究盲区，通过562个运动序列验证物理感知能力。**

- **链接: [http://arxiv.org/pdf/2508.06205v1](http://arxiv.org/pdf/2508.06205v1)**

> **作者:** Ruiyan Wang; Lin Zuo; Zonghao Lin; Qiang Wang; Zhengxue Cheng; Rong Xie; Jun Ling; Li Song
>
> **摘要:** The Human-Object Interaction (HOI) task explores the dynamic interactions between humans and objects in physical environments, providing essential biomechanical and cognitive-behavioral foundations for fields such as robotics, virtual reality, and human-computer interaction. However, existing HOI data sets focus on details of affordance, often neglecting the influence of physical properties of objects on human long-term motion. To bridge this gap, we introduce the PA-HOI Motion Capture dataset, which highlights the impact of objects' physical attributes on human motion dynamics, including human posture, moving velocity, and other motion characteristics. The dataset comprises 562 motion sequences of human-object interactions, with each sequence performed by subjects of different genders interacting with 35 3D objects that vary in size, shape, and weight. This dataset stands out by significantly extending the scope of existing ones for understanding how the physical attributes of different objects influence human posture, speed, motion scale, and interacting strategies. We further demonstrate the applicability of the PA-HOI dataset by integrating it with existing motion generation methods, validating its capacity to transfer realistic physical awareness.
>
---
#### [new 089] Improving Masked Style Transfer using Blended Partial Convolution
- **分类: cs.CV**

- **简介: 论文提出一种基于部分卷积的风格迁移方法，解决遮罩技术在捕捉兴趣区域风格时的不足，通过融合技术优化区域选择缺陷，提升视觉与量化效果。**

- **链接: [http://arxiv.org/pdf/2508.05769v1](http://arxiv.org/pdf/2508.05769v1)**

> **作者:** Seyed Hadi Seyed; Ayberk Cansever; David Hart
>
> **摘要:** Artistic style transfer has long been possible with the advancements of convolution- and transformer-based neural networks. Most algorithms apply the artistic style transfer to the whole image, but individual users may only need to apply a style transfer to a specific region in the image. The standard practice is to simply mask the image after the stylization. This work shows that this approach tends to improperly capture the style features in the region of interest. We propose a partial-convolution-based style transfer network that accurately applies the style features exclusively to the region of interest. Additionally, we present network-internal blending techniques that account for imperfections in the region selection. We show that this visually and quantitatively improves stylization using examples from the SA-1B dataset. Code is publicly available at https://github.com/davidmhart/StyleTransferMasked.
>
---
#### [new 090] AnimateScene: Camera-controllable Animation in Any Scene
- **分类: cs.CV**

- **简介: 论文提出AnimateScene，解决3D场景与4D动画融合中的位置放置、光照风格对齐及相机轨迹重建问题，通过模块化设计实现动态场景视频生成。**

- **链接: [http://arxiv.org/pdf/2508.05982v1](http://arxiv.org/pdf/2508.05982v1)**

> **作者:** Qingyang Liu; Bingjie Gao; Weiheng Huang; Jun Zhang; Zhongqian Sun; Yang Wei; Zelin Peng; Qianli Ma; Shuai Yang; Zhaohe Liao; Haonan Zhao; Li Niu
>
> **摘要:** 3D scene reconstruction and 4D human animation have seen rapid progress and broad adoption in recent years. However, seamlessly integrating reconstructed scenes with 4D human animation to produce visually engaging results remains challenging. One key difficulty lies in placing the human at the correct location and scale within the scene while avoiding unrealistic interpenetration. Another challenge is that the human and the background may exhibit different lighting and style, leading to unrealistic composites. In addition, appealing character motion videos are often accompanied by camera movements, which means that the viewpoints need to be reconstructed along a specified trajectory. We present AnimateScene, which addresses the above issues in a unified framework. First, we design an accurate placement module that automatically determines a plausible 3D position for the human and prevents any interpenetration within the scene during motion. Second, we propose a training-free style alignment method that adapts the 4D human representation to match the background's lighting and style, achieving coherent visual integration. Finally, we design a joint post-reconstruction method for both the 4D human and the 3D scene that allows camera trajectories to be inserted, enabling the final rendered video to feature visually appealing camera movements. Extensive experiments show that AnimateScene generates dynamic scene videos with high geometric detail and spatiotemporal coherence across various camera and action combinations.
>
---
#### [new 091] ETA: Energy-based Test-time Adaptation for Depth Completion
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 本文提出基于能量的测试时间适应方法（ETA）用于深度补全任务，解决模型在新环境下因分布偏移导致的预测误差。通过对抗性扰动探索数据空间，训练能量模型评估深度预测的分布性，优化模型参数以匹配源分布，实现跨环境泛化性能提升。**

- **链接: [http://arxiv.org/pdf/2508.05989v1](http://arxiv.org/pdf/2508.05989v1)**

> **作者:** Younjoon Chung; Hyoungseob Park; Patrick Rim; Xiaoran Zhang; Jihe He; Ziyao Zeng; Safa Cicek; Byung-Woo Hong; James S. Duncan; Alex Wong
>
> **摘要:** We propose a method for test-time adaptation of pretrained depth completion models. Depth completion models, trained on some ``source'' data, often predict erroneous outputs when transferred to ``target'' data captured in novel environmental conditions due to a covariate shift. The crux of our method lies in quantifying the likelihood of depth predictions belonging to the source data distribution. The challenge is in the lack of access to out-of-distribution (target) data prior to deployment. Hence, rather than making assumptions regarding the target distribution, we utilize adversarial perturbations as a mechanism to explore the data space. This enables us to train an energy model that scores local regions of depth predictions as in- or out-of-distribution. We update the parameters of pretrained depth completion models at test time to minimize energy, effectively aligning test-time predictions to those of the source distribution. We call our method ``Energy-based Test-time Adaptation'', or ETA for short. We evaluate our method across three indoor and three outdoor datasets, where ETA improve over the previous state-of-the-art method by an average of 6.94% for outdoors and 10.23% for indoors. Project Page: https://fuzzythecat.github.io/eta.
>
---
#### [new 092] Aligning Effective Tokens with Video Anomaly in Large Language Models
- **分类: cs.CV**

- **简介: 论文提出VA-GPT，针对视频异常理解中空间时间稀疏导致的冗余问题，通过SETS和TETG模块提升模型性能，构建数据集与评估基准。**

- **链接: [http://arxiv.org/pdf/2508.06350v1](http://arxiv.org/pdf/2508.06350v1)**

> **作者:** Yingxian Chen; Jiahui Liu; Ruifan Di; Yanwei Li; Chirui Chang; Shizhen Zhao; Wilton W. T. Fok; Xiaojuan Qi; Yik-Chung Wu
>
> **摘要:** Understanding abnormal events in videos is a vital and challenging task that has garnered significant attention in a wide range of applications. Although current video understanding Multi-modal Large Language Models (MLLMs) are capable of analyzing general videos, they often struggle to handle anomalies due to the spatial and temporal sparsity of abnormal events, where the redundant information always leads to suboptimal outcomes. To address these challenges, exploiting the representation and generalization capabilities of Vison Language Models (VLMs) and Large Language Models (LLMs), we propose VA-GPT, a novel MLLM designed for summarizing and localizing abnormal events in various videos. Our approach efficiently aligns effective tokens between visual encoders and LLMs through two key proposed modules: Spatial Effective Token Selection (SETS) and Temporal Effective Token Generation (TETG). These modules enable our model to effectively capture and analyze both spatial and temporal information associated with abnormal events, resulting in more accurate responses and interactions. Furthermore, we construct an instruction-following dataset specifically for fine-tuning video-anomaly-aware MLLMs, and introduce a cross-domain evaluation benchmark based on XD-Violence dataset. Our proposed method outperforms existing state-of-the-art methods on various benchmarks.
>
---
#### [new 093] Are you In or Out (of gallery)? Wisdom from the Same-Identity Crowd
- **分类: cs.CV; cs.AI**

- **简介: 论文解决one-to-many面部识别中In/Out-gallery问题，通过利用同一身份的额外注册图像生成特征向量训练分类器，有效应对模糊、低分辨率等降质场景，跨人群准确率相似，新方法通过先进损失函数提升效果。**

- **链接: [http://arxiv.org/pdf/2508.06357v1](http://arxiv.org/pdf/2508.06357v1)**

> **作者:** Aman Bhatta; Maria Dhakal; Michael C. King; Kevin W. Bowyer
>
> **摘要:** A central problem in one-to-many facial identification is that the person in the probe image may or may not have enrolled image(s) in the gallery; that is, may be In-gallery or Out-of-gallery. Past approaches to detect when a rank-one result is Out-of-gallery have mostly focused on finding a suitable threshold on the similarity score. We take a new approach, using the additional enrolled images of the identity with the rank-one result to predict if the rank-one result is In-gallery / Out-of-gallery. Given a gallery of identities and images, we generate In-gallery and Out-of-gallery training data by extracting the ranks of additional enrolled images corresponding to the rank-one identity. We then train a classifier to utilize this feature vector to predict whether a rank-one result is In-gallery or Out-of-gallery. Using two different datasets and four different matchers, we present experimental results showing that our approach is viable for mugshot quality probe images, and also, importantly, for probes degraded by blur, reduced resolution, atmospheric turbulence and sunglasses. We also analyze results across demographic groups, and show that In-gallery / Out-of-gallery classification accuracy is similar across demographics. Our approach has the potential to provide an objective estimate of whether a one-to-many facial identification is Out-of-gallery, and thereby to reduce false positive identifications, wrongful arrests, and wasted investigative time. Interestingly, comparing the results of older deep CNN-based face matchers with newer ones suggests that the effectiveness of our Out-of-gallery detection approach emerges only with matchers trained using advanced margin-based loss functions.
>
---
#### [new 094] Neural Field Representations of Mobile Computational Photography
- **分类: cs.CV**

- **简介: 论文提出利用神经场模型，直接从移动设备拍摄的原始数据中重建复杂场景，无需预处理、标注或机器学习先验，通过随机梯度下降优化。**

- **链接: [http://arxiv.org/pdf/2508.05907v1](http://arxiv.org/pdf/2508.05907v1)**

> **作者:** Ilya Chugunov
>
> **备注:** PhD thesis
>
> **摘要:** Over the past two decades, mobile imaging has experienced a profound transformation, with cell phones rapidly eclipsing all other forms of digital photography in popularity. Today's cell phones are equipped with a diverse range of imaging technologies - laser depth ranging, multi-focal camera arrays, and split-pixel sensors - alongside non-visual sensors such as gyroscopes, accelerometers, and magnetometers. This, combined with on-board integrated chips for image and signal processing, makes the cell phone a versatile pocket-sized computational imaging platform. Parallel to this, we have seen in recent years how neural fields - small neural networks trained to map continuous spatial input coordinates to output signals - enable the reconstruction of complex scenes without explicit data representations such as pixel arrays or point clouds. In this thesis, I demonstrate how carefully designed neural field models can compactly represent complex geometry and lighting effects. Enabling applications such as depth estimation, layer separation, and image stitching directly from collected in-the-wild mobile photography data. These methods outperform state-of-the-art approaches without relying on complex pre-processing steps, labeled ground truth data, or machine learning priors. Instead, they leverage well-constructed, self-regularized models that tackle challenging inverse problems through stochastic gradient descent, fitting directly to raw measurements from a smartphone.
>
---
#### [new 095] CLIPin: A Non-contrastive Plug-in to CLIP for Multimodal Semantic Alignment
- **分类: cs.CV; cs.AI**

- **简介: 论文提出CLIPin作为非对比插件，增强CLIP多模态语义对齐能力，解决弱监督数据语义松散及医学数据多样性不足问题，通过共享预投影器平衡参数使用。**

- **链接: [http://arxiv.org/pdf/2508.06434v1](http://arxiv.org/pdf/2508.06434v1)**

> **作者:** Shengzhu Yang; Jiawei Du; Shuai Lu; Weihang Zhang; Ningli Wang; Huiqi Li
>
> **摘要:** Large-scale natural image-text datasets, especially those automatically collected from the web, often suffer from loose semantic alignment due to weak supervision, while medical datasets tend to have high cross-modal correlation but low content diversity. These properties pose a common challenge for contrastive language-image pretraining (CLIP): they hinder the model's ability to learn robust and generalizable representations. In this work, we propose CLIPin, a unified non-contrastive plug-in that can be seamlessly integrated into CLIP-style architectures to improve multimodal semantic alignment, providing stronger supervision and enhancing alignment robustness. Furthermore, two shared pre-projectors are designed for image and text modalities respectively to facilitate the integration of contrastive and non-contrastive learning in a parameter-compromise manner. Extensive experiments on diverse downstream tasks demonstrate the effectiveness and generality of CLIPin as a plug-and-play component compatible with various contrastive frameworks. Code is available at https://github.com/T6Yang/CLIPin.
>
---
#### [new 096] UGD-IML: A Unified Generative Diffusion-based Framework for Constrained and Unconstrained Image Manipulation Localization
- **分类: cs.CV**

- **简介: 论文提出基于扩散模型的统一框架UGD-IML，解决传统IML需大量标注数据及CIML流程复杂的问题，通过参数共享与类嵌入机制实现IML与CIML无缝切换，提升在有限数据下的性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.06101v1](http://arxiv.org/pdf/2508.06101v1)**

> **作者:** Yachun Mi; Xingyang He; Shixin Sun; Yu Li; Yanting Li; Zhixuan Li; Jian Jin; Chen Hui; Shaohui Liu
>
> **摘要:** In the digital age, advanced image editing tools pose a serious threat to the integrity of visual content, making image forgery detection and localization a key research focus. Most existing Image Manipulation Localization (IML) methods rely on discriminative learning and require large, high-quality annotated datasets. However, current datasets lack sufficient scale and diversity, limiting model performance in real-world scenarios. To overcome this, recent studies have explored Constrained IML (CIML), which generates pixel-level annotations through algorithmic supervision. However, existing CIML approaches often depend on complex multi-stage pipelines, making the annotation process inefficient. In this work, we propose a novel generative framework based on diffusion models, named UGD-IML, which for the first time unifies both IML and CIML tasks within a single framework. By learning the underlying data distribution, generative diffusion models inherently reduce the reliance on large-scale labeled datasets, allowing our approach to perform effectively even under limited data conditions. In addition, by leveraging a class embedding mechanism and a parameter-sharing design, our model seamlessly switches between IML and CIML modes without extra components or training overhead. Furthermore, the end-to-end design enables our model to avoid cumbersome steps in the data annotation process. Extensive experimental results on multiple datasets demonstrate that UGD-IML outperforms the SOTA methods by an average of 9.66 and 4.36 in terms of F1 metrics for IML and CIML tasks, respectively. Moreover, the proposed method also excels in uncertainty estimation, visualization and robustness.
>
---
#### [new 097] MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss
- **分类: cs.CV**

- **简介: 论文提出MAISI-v2框架，通过直连流和区域特定对比损失解决3D医学图像合成中通用性差、推理慢及条件不一致问题，实现33倍加速与高精度生成。**

- **链接: [http://arxiv.org/pdf/2508.05772v1](http://arxiv.org/pdf/2508.05772v1)**

> **作者:** Can Zhao; Pengfei Guo; Dong Yang; Yucheng Tang; Yufan He; Benjamin Simon; Mason Belue; Stephanie Harmon; Baris Turkbey; Daguang Xu
>
> **摘要:** Medical image synthesis is an important topic for both clinical and research applications. Recently, diffusion models have become a leading approach in this area. Despite their strengths, many existing methods struggle with (1) limited generalizability that only work for specific body regions or voxel spacings, (2) slow inference, which is a common issue for diffusion models, and (3) weak alignment with input conditions, which is a critical issue for medical imaging. MAISI, a previously proposed framework, addresses generalizability issues but still suffers from slow inference and limited condition consistency. In this work, we present MAISI-v2, the first accelerated 3D medical image synthesis framework that integrates rectified flow to enable fast and high quality generation. To further enhance condition fidelity, we introduce a novel region-specific contrastive loss to enhance the sensitivity to region of interest. Our experiments show that MAISI-v2 can achieve SOTA image quality with $33 \times$ acceleration for latent diffusion model. We also conducted a downstream segmentation experiment to show that the synthetic images can be used for data augmentation. We release our code, training details, model weights, and a GUI demo to facilitate reproducibility and promote further development within the community.
>
---
#### [new 098] A Classification-Aware Super-Resolution Framework for Ship Targets in SAR Imagery
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 本文提出一种分类感知的超分辨率框架，针对SAR图像中的船只目标，通过优化损失函数同时提升图像质量和分类准确率，解决传统方法忽略分类与超分辨率关系的问题。**

- **链接: [http://arxiv.org/pdf/2508.06407v1](http://arxiv.org/pdf/2508.06407v1)**

> **作者:** Ch Muhammad Awais; Marco Reggiannini; Davide Moroni; Oktay Karakus
>
> **摘要:** High-resolution imagery plays a critical role in improving the performance of visual recognition tasks such as classification, detection, and segmentation. In many domains, including remote sensing and surveillance, low-resolution images can limit the accuracy of automated analysis. To address this, super-resolution (SR) techniques have been widely adopted to attempt to reconstruct high-resolution images from low-resolution inputs. Related traditional approaches focus solely on enhancing image quality based on pixel-level metrics, leaving the relationship between super-resolved image fidelity and downstream classification performance largely underexplored. This raises a key question: can integrating classification objectives directly into the super-resolution process further improve classification accuracy? In this paper, we try to respond to this question by investigating the relationship between super-resolution and classification through the deployment of a specialised algorithmic strategy. We propose a novel methodology that increases the resolution of synthetic aperture radar imagery by optimising loss functions that account for both image quality and classification performance. Our approach improves image quality, as measured by scientifically ascertained image quality indicators, while also enhancing classification accuracy.
>
---
#### [new 099] Advanced Deep Learning Techniques for Accurate Lung Cancer Detection and Classification
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出基于DenseNet201的深度学习方法，针对肺癌检测中的假阳性与数据不平衡问题，通过Focal Loss、数据增强及正则化提升准确率至98.95%。任务为肺癌分类与检测，核心解决数据不平衡与过拟合难题。**

- **链接: [http://arxiv.org/pdf/2508.06287v1](http://arxiv.org/pdf/2508.06287v1)**

> **作者:** Mobarak Abumohsen; Enrique Costa-Montenegro; Silvia García-Méndez; Amani Yousef Owda; Majdi Owda
>
> **摘要:** Lung cancer (LC) ranks among the most frequently diagnosed cancers and is one of the most common causes of death for men and women worldwide. Computed Tomography (CT) images are the most preferred diagnosis method because of their low cost and their faster processing times. Many researchers have proposed various ways of identifying lung cancer using CT images. However, such techniques suffer from significant false positives, leading to low accuracy. The fundamental reason results from employing a small and imbalanced dataset. This paper introduces an innovative approach for LC detection and classification from CT images based on the DenseNet201 model. Our approach comprises several advanced methods such as Focal Loss, data augmentation, and regularization to overcome the imbalanced data issue and overfitting challenge. The findings show the appropriateness of the proposal, attaining a promising performance of 98.95% accuracy.
>
---
#### [new 100] KnapFormer: An Online Load Balancer for Efficient Diffusion Transformers Training
- **分类: cs.DC; cs.CV**

- **简介: 论文提出KnapFormer框架，通过全局背包问题分配tokens解决扩散变压器分布式训练中的token不平衡与通信开销问题，实现2-3倍速度提升。**

- **链接: [http://arxiv.org/pdf/2508.06001v1](http://arxiv.org/pdf/2508.06001v1)**

> **作者:** Kai Zhang; Peng Wang; Sai Bi; Jianming Zhang; Yuanjun Xiong
>
> **备注:** Code is available at https://github.com/Kai-46/KnapFormer/
>
> **摘要:** We present KnapFormer, an efficient and versatile framework to combine workload balancing and sequence parallelism in distributed training of Diffusion Transformers (DiT). KnapFormer builds on the insight that strong synergy exists between sequence parallelism and the need to address the significant token imbalance across ranks. This imbalance arises from variable-length text inputs and varying visual token counts in mixed-resolution and image-video joint training. KnapFormer redistributes tokens by first gathering sequence length metadata across all ranks in a balancing group and solving a global knapsack problem. The solver aims to minimize the variances of total workload per-GPU, while accounting for the effect of sequence parallelism. By integrating DeepSpeed-Ulysees-based sequence parallelism in the load-balancing decision process and utilizing a simple semi-empirical workload model, KnapFormers achieves minimal communication overhead and less than 1% workload discrepancy in real-world training workloads with sequence length varying from a few hundred to tens of thousands. It eliminates straggler effects and achieves 2x to 3x speedup when training state-of-the-art diffusion models like FLUX on mixed-resolution and image-video joint data corpora. We open-source the KnapFormer implementation at https://github.com/Kai-46/KnapFormer/
>
---
#### [new 101] Fine-Tuning Vision-Language Models for Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports
- **分类: cs.IR; cs.AI; cs.CL; cs.CV; cs.LG; I.2.7; I.7.2; J.1**

- **简介: 该论文提出微调VLM用于马来西亚审计财务报告中表格的Markdown转换，解决旋转布局、多级标题等问题，通过LoRA优化并评估TEDS指标，实现高精度转换。**

- **链接: [http://arxiv.org/pdf/2508.05669v1](http://arxiv.org/pdf/2508.05669v1)**

> **作者:** Jin Khye Tan; En Jun Choong; Ethan Jeremiah Chitty; Yan Pheng Choo; John Hsin Yang Wong; Chern Eu Cheah
>
> **备注:** 28 pages, 14 figures, 5 tables. Evaluation code (LLM-as-a-judge and Markdown TEDS) is available at https://github.com/jinkhye/MyFinMarkdown. The development dataset and evaluation benchmark are available on Hugging Face at https://huggingface.co/datasets/jinkhye/MyFinMarkdown-sample and https://huggingface.co/datasets/jinkhye/MyFinMarkdown-bench respectively
>
> **摘要:** Accurately extracting and representing the structure of tabular data from financial documents remains a critical challenge in document understanding, particularly for regulatory and analytical use cases. This study addresses the complexity of converting financial tables from Malaysian audited financial reports into Markdown format, a task complicated by rotated layouts, multi-level headers, and implicit structural cues. We propose a fine-tuned vision-language model (VLM), based on Qwen2.5-VL-7B, optimized for high-fidelity Markdown generation from document images. Our approach includes a curated dataset of 2,152 image-text pairs with augmentations and a supervised fine-tuning strategy using LoRA. To assess performance, we evaluated our model on 100 out-of-sample tables using a dual framework: a criteria-based LLM-as-a-judge for fine-grained accuracy and our novel Markdown Tree-Edit-Distance-based Similarity (TEDS) metric for holistic structural fidelity. Our model achieves a 92.20% overall accuracy on the criteria-based assessment and a 96.53% Markdown TEDS score. This performance significantly surpasses its Qwen2.5-VL-7B base model, larger-scale VLMs, and specialized reasoning-enabled models. Compared to these self-hosted alternatives, it also significantly reduces inference time. Furthermore, its accuracy exceeds that of widely used proprietary models such as OpenAI's GPT-4o and Gemini 2.5 Flash. These results demonstrate that domain-specific fine-tuning provides an effective and efficient method to bridge the gap between unstructured financial documents and downstream automation, rivalling much larger and more general models without their computational overhead.
>
---
#### [new 102] Multivariate Fields of Experts
- **分类: eess.IV; cs.CV; cs.LG; eess.SP**

- **简介: 该论文提出多变量专家场框架，用于图像先验学习，解决逆向问题，通过Moreau envelopes构造多变量势函数，提升性能、速度与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.06490v1](http://arxiv.org/pdf/2508.06490v1)**

> **作者:** Stanislas Ducotterd; Michael Unser
>
> **摘要:** We introduce the multivariate fields of experts, a new framework for the learning of image priors. Our model generalizes existing fields of experts methods by incorporating multivariate potential functions constructed via Moreau envelopes of the $\ell_\infty$-norm. We demonstrate the effectiveness of our proposal across a range of inverse problems that include image denoising, deblurring, compressed-sensing magnetic-resonance imaging, and computed tomography. The proposed approach outperforms comparable univariate models and achieves performance close to that of deep-learning-based regularizers while being significantly faster, requiring fewer parameters, and being trained on substantially fewer data. In addition, our model retains a relatively high level of interpretability due to its structured design.
>
---
#### [new 103] Neural Field-Based 3D Surface Reconstruction of Microstructures from Multi-Detector Signals in Scanning Electron Microscopy
- **分类: eess.IV; cs.CV; physics.ins-det**

- **简介: 论文提出基于神经场的3D微结构重建方法，解决传统方法的离散表示、校准和阴影误差问题，通过多探测器数据融合与自校准实现高保真重建。**

- **链接: [http://arxiv.org/pdf/2508.04728v1](http://arxiv.org/pdf/2508.04728v1)**

> **作者:** Shuo Chen; Yijin Li; Xi Zheng; Guofeng Zhang
>
> **摘要:** The scanning electron microscope (SEM) is a widely used imaging device in scientific research and industrial applications. Conventional two-dimensional (2D) SEM images do not directly reveal the three-dimensional (3D) topography of micro samples, motivating the development of SEM 3D surface reconstruction methods. However, reconstruction of complex microstructures remains challenging for existing methods due to the limitations of discrete 3D representations, the need for calibration with reference samples, and shadow-induced gradient errors. Here, we introduce NFH-SEM, a neural field-based hybrid SEM 3D reconstruction method that takes multi-view, multi-detector 2D SEM images as input and fuses geometric and photometric information into a continuous neural field representation. NFH-SEM eliminates the manual calibration procedures through end-to-end self-calibration and automatically disentangles shadows from SEM images during training, enabling accurate reconstruction of intricate microstructures. We validate the effectiveness of NFH-SEM on real and simulated datasets. Our experiments show high-fidelity reconstructions of diverse, challenging samples, including two-photon lithography microstructures, peach pollen, and silicon carbide particle surfaces, demonstrating precise detail and broad applicability.
>
---
#### [new 104] Anti-Tamper Protection for Unauthorized Individual Image Generation
- **分类: cs.CR; cs.CV**

- **简介: 论文提出Anti-Tamper Perturbation（ATP）方法，针对未经授权的个体图像生成中伪造攻击，通过频域下的保护与授权扰动机制，抵御净化技术绕过保护，保障肖像隐私。**

- **链接: [http://arxiv.org/pdf/2508.06325v1](http://arxiv.org/pdf/2508.06325v1)**

> **作者:** Zelin Li; Ruohan Zong; Yifan Liu; Ruichen Yao; Yaokun Liu; Yang Zhang; Dong Wang
>
> **备注:** 22 pages ,22 figures, Paper has been accepted by ICCV'2025
>
> **摘要:** With the advancement of personalized image generation technologies, concerns about forgery attacks that infringe on portrait rights and privacy are growing. To address these concerns, protection perturbation algorithms have been developed to disrupt forgery generation. However, the protection algorithms would become ineffective when forgery attackers apply purification techniques to bypass the protection. To address this issue, we present a novel approach, Anti-Tamper Perturbation (ATP). ATP introduces a tamper-proof mechanism within the perturbation. It consists of protection and authorization perturbations, where the protection perturbation defends against forgery attacks, while the authorization perturbation detects purification-based tampering. Both protection and authorization perturbations are applied in the frequency domain under the guidance of a mask, ensuring that the protection perturbation does not disrupt the authorization perturbation. This design also enables the authorization perturbation to be distributed across all image pixels, preserving its sensitivity to purification-based tampering. ATP demonstrates its effectiveness in defending forgery attacks across various attack settings through extensive experiments, providing a robust solution for protecting individuals' portrait rights and privacy. Our code is available at: https://github.com/Seeyn/Anti-Tamper-Perturbation .
>
---
#### [new 105] Transformer-Based Explainable Deep Learning for Breast Cancer Detection in Mammography: The MammoFormer Framework
- **分类: eess.IV; cs.CV**

- **简介: 论文提出基于Transformer的可解释深度学习框架MammoFormer，解决乳腺癌X光检测中CNN局限性问题，通过融合多特征增强与可解释AI，提升检测精度并提供诊断解释，实现临床可部署。**

- **链接: [http://arxiv.org/pdf/2508.06137v1](http://arxiv.org/pdf/2508.06137v1)**

> **作者:** Ojonugwa Oluwafemi Ejiga Peter; Daniel Emakporuena; Bamidele Dayo Tunde; Maryam Abdulkarim; Abdullahi Bn Umar
>
> **摘要:** Breast cancer detection through mammography interpretation remains difficult because of the minimal nature of abnormalities that experts need to identify alongside the variable interpretations between readers. The potential of CNNs for medical image analysis faces two limitations: they fail to process both local information and wide contextual data adequately, and do not provide explainable AI (XAI) operations that doctors need to accept them in clinics. The researcher developed the MammoFormer framework, which unites transformer-based architecture with multi-feature enhancement components and XAI functionalities within one framework. Seven different architectures consisting of CNNs, Vision Transformer, Swin Transformer, and ConvNext were tested alongside four enhancement techniques, including original images, negative transformation, adaptive histogram equalization, and histogram of oriented gradients. The MammoFormer framework addresses critical clinical adoption barriers of AI mammography systems through: (1) systematic optimization of transformer architectures via architecture-specific feature enhancement, achieving up to 13% performance improvement, (2) comprehensive explainable AI integration providing multi-perspective diagnostic interpretability, and (3) a clinically deployable ensemble system combining CNN reliability with transformer global context modeling. The combination of transformer models with suitable feature enhancements enables them to achieve equal or better results than CNN approaches. ViT achieves 98.3% accuracy alongside AHE while Swin Transformer gains a 13.0% advantage through HOG enhancements
>
---
#### [new 106] Improving Diagnostic Accuracy for Oral Cancer with inpainting Synthesis Lesions Generated Using Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 论文提出利用扩散模型生成合成口腔癌病变，解决数据不足问题，提升诊断准确性，分类准确率达0.97，检测达0.85。**

- **链接: [http://arxiv.org/pdf/2508.06151v1](http://arxiv.org/pdf/2508.06151v1)**

> **作者:** Yong Oh Lee; JeeEun Kim; Jung Woo Lee
>
> **摘要:** In oral cancer diagnostics, the limited availability of annotated datasets frequently constrains the performance of diagnostic models, particularly due to the variability and insufficiency of training data. To address these challenges, this study proposed a novel approach to enhance diagnostic accuracy by synthesizing realistic oral cancer lesions using an inpainting technique with a fine-tuned diffusion model. We compiled a comprehensive dataset from multiple sources, featuring a variety of oral cancer images. Our method generated synthetic lesions that exhibit a high degree of visual fidelity to actual lesions, thereby significantly enhancing the performance of diagnostic algorithms. The results show that our classification model achieved a diagnostic accuracy of 0.97 in differentiating between cancerous and non-cancerous tissues, while our detection model accurately identified lesion locations with 0.85 accuracy. This method validates the potential for synthetic image generation in medical diagnostics and paves the way for further research into extending these methods to other types of cancer diagnostics.
>
---
#### [new 107] Universally Unfiltered and Unseen:Input-Agnostic Multimodal Jailbreaks against Text-to-Image Model Safeguards
- **分类: cs.CR; cs.CV; cs.MM**

- **简介: 论文提出一种多模态攻击方法U3-Attack，针对文本到图像模型的安全防护，通过优化图像背景和敏感词改写，突破提示过滤与安全检查器限制，提升攻击成功率。**

- **链接: [http://arxiv.org/pdf/2508.05658v1](http://arxiv.org/pdf/2508.05658v1)**

> **作者:** Song Yan; Hui Wei; Jinlong Fei; Guoliang Yang; Zhengyu Zhao; Zheng Wamg
>
> **备注:** ACM MM 2025
>
> **摘要:** Various (text) prompt filters and (image) safety checkers have been implemented to mitigate the misuse of Text-to-Image (T2I) models in creating Not-Safe-For-Work (NSFW) content.In order to expose potential security vulnerabilities of such safeguards, multimodal jailbreaks have been studied.However, existing jailbreaks are limited to prompt-specific and image-specific perturbations, which suffer from poor scalability and time-consuming optimization.To address these limitations, we propose Universally Unfiltered and Unseen (U3)-Attack, a multimodal jailbreak attack method against T2I safeguards.Specifically, U3-Attack optimizes an adversarial patch on the image background to universally bypass safety checkers and optimizes a safe paraphrase set from a sensitive word to universally bypass prompt filters while eliminating redundant computations.Extensive experimental results demonstrate the superiority of our U3-Attack on both open-source and commercial T2I models.For example, on the commercial Runway-inpainting model with both prompt filter and safety checker, our U3-Attack achieves $~4\times$ higher success rates than the state-of-the-art multimodal jailbreak attack, MMA-Diffusion.Content Warning: This paper includes examples of NSFW content.
>
---
#### [new 108] Affordance-R1: Reinforcement Learning for Generalizable Affordance Reasoning in Multimodal Large Language Model
- **分类: cs.RO; cs.CV**

- **简介: 论文提出Affordance-R1框架，通过强化学习结合认知CoT引导的GRPO，解决多模态大语言模型在跨域泛化和推理能力不足的问题，设计多奖励函数与ReasonAff数据集，实现零样本泛化与测试时推理。**

- **链接: [http://arxiv.org/pdf/2508.06206v1](http://arxiv.org/pdf/2508.06206v1)**

> **作者:** Hanqing Wang; Shaoyang Wang; Yiming Zhong; Zemin Yang; Jiamin Wang; Zhiqing Cui; Jiahao Yuan; Yifan Han; Mingyu Liu; Yuexin Ma
>
> **摘要:** Affordance grounding focuses on predicting the specific regions of objects that are associated with the actions to be performed by robots. It plays a vital role in the fields of human-robot interaction, human-object interaction, embodied manipulation, and embodied perception. Existing models often neglect the affordance shared among different objects because they lack the Chain-of-Thought(CoT) reasoning abilities, limiting their out-of-domain (OOD) generalization and explicit reasoning capabilities. To address these challenges, we propose Affordance-R1, the first unified affordance grounding framework that integrates cognitive CoT guided Group Relative Policy Optimization (GRPO) within a reinforcement learning paradigm. Specifically, we designed a sophisticated affordance function, which contains format, perception, and cognition rewards to effectively guide optimization directions. Furthermore, we constructed a high-quality affordance-centric reasoning dataset, ReasonAff, to support training. Trained exclusively via reinforcement learning with GRPO and without explicit reasoning data, Affordance-R1 achieves robust zero-shot generalization and exhibits emergent test-time reasoning capabilities. Comprehensive experiments demonstrate that our model outperforms well-established methods and exhibits open-world generalization. To the best of our knowledge, Affordance-R1 is the first to integrate GRPO-based RL with reasoning into affordance reasoning. The code of our method and our dataset is released on https://github.com/hq-King/Affordance-R1.
>
---
#### [new 109] Integrating Vision Foundation Models with Reinforcement Learning for Enhanced Object Interaction
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY; 68T07, 68T40, 90C40, 93E35; I.2.6; I.2.9; I.2.10**

- **简介: 论文提出融合视觉基础模型与强化学习，提升机器人在模拟环境中的物体交互能力，通过SAM+YOLOv5+PPO代理实现感知与交互优化，显著提高成功率和导航效率。**

- **链接: [http://arxiv.org/pdf/2508.05838v1](http://arxiv.org/pdf/2508.05838v1)**

> **作者:** Ahmad Farooq; Kamran Iqbal
>
> **备注:** Published in the Proceedings of the 2025 3rd International Conference on Robotics, Control and Vision Engineering (RCVE'25). 6 pages, 3 figures, 1 table
>
> **摘要:** This paper presents a novel approach that integrates vision foundation models with reinforcement learning to enhance object interaction capabilities in simulated environments. By combining the Segment Anything Model (SAM) and YOLOv5 with a Proximal Policy Optimization (PPO) agent operating in the AI2-THOR simulation environment, we enable the agent to perceive and interact with objects more effectively. Our comprehensive experiments, conducted across four diverse indoor kitchen settings, demonstrate significant improvements in object interaction success rates and navigation efficiency compared to a baseline agent without advanced perception. The results show a 68% increase in average cumulative reward, a 52.5% improvement in object interaction success rate, and a 33% increase in navigation efficiency. These findings highlight the potential of integrating foundation models with reinforcement learning for complex robotic tasks, paving the way for more sophisticated and capable autonomous agents.
>
---
#### [new 110] ThematicPlane: Bridging Tacit User Intent and Latent Spaces for Image Generation
- **分类: cs.HC; cs.AI; cs.CL; cs.CV; H.5.2; I.2.7**

- **简介: 论文提出ThematicPlane系统，通过交互式主题平面连接隐性创意意图与系统控制，解决非专家图像生成中意图对齐难题，促进迭代创作。**

- **链接: [http://arxiv.org/pdf/2508.06065v1](http://arxiv.org/pdf/2508.06065v1)**

> **作者:** Daniel Lee; Nikhil Sharma; Donghoon Shin; DaEun Choi; Harsh Sharma; Jeonghwan Kim; Heng Ji
>
> **摘要:** Generative AI has made image creation more accessible, yet aligning outputs with nuanced creative intent remains challenging, particularly for non-experts. Existing tools often require users to externalize ideas through prompts or references, limiting fluid exploration. We introduce ThematicPlane, a system that enables users to navigate and manipulate high-level semantic concepts (e.g., mood, style, or narrative tone) within an interactive thematic design plane. This interface bridges the gap between tacit creative intent and system control. In our exploratory study (N=6), participants engaged in divergent and convergent creative modes, often embracing unexpected results as inspiration or iteration cues. While they grounded their exploration in familiar themes, differing expectations of how themes mapped to outputs revealed a need for more explainable controls. Overall, ThematicPlane fosters expressive, iterative workflows and highlights new directions for intuitive, semantics-driven interaction in generative design tools.
>
---
#### [new 111] Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文探讨一般主义机器人策略的短路学习机制，分析数据集多样性与碎片化对泛化能力的影响，提出优化数据采集和增强策略以提升泛化性能。**

- **链接: [http://arxiv.org/pdf/2508.06426v1](http://arxiv.org/pdf/2508.06426v1)**

> **作者:** Youguang Xing; Xu Luo; Junlin Xie; Lianli Gao; Hengtao Shen; Jingkuan Song
>
> **备注:** CoRL 2025
>
> **摘要:** Generalist robot policies trained on large-scale datasets such as Open X-Embodiment (OXE) demonstrate strong performance across a wide range of tasks. However, they often struggle to generalize beyond the distribution of their training data. In this paper, we investigate the underlying cause of this limited generalization capability. We identify shortcut learning -- the reliance on task-irrelevant features -- as a key impediment to generalization. Through comprehensive theoretical and empirical analysis, we uncover two primary contributors to shortcut learning: (1) limited diversity within individual sub-datasets, and (2) significant distributional disparities across sub-datasets, leading to dataset fragmentation. These issues arise from the inherent structure of large-scale datasets like OXE, which are typically composed of multiple sub-datasets collected independently across varied environments and embodiments. Our findings provide critical insights into dataset collection strategies that can reduce shortcut learning and enhance the generalization ability of generalist robot policies. Moreover, in scenarios where acquiring new large-scale data is impractical, we demonstrate that carefully selected robotic data augmentation strategies can effectively reduce shortcut learning in existing offline datasets, thereby improving generalization capabilities of generalist robot policies, e.g., $\pi_0$, in both simulation and real-world environments. More information at https://lucky-light-sun.github.io/proj/shortcut-learning-in-grps/.
>
---
#### [new 112] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields
- **分类: cs.LG; cs.AI; cs.CV; cs.DC**

- **简介: 论文提出FedMeNF，通过隐私保护的损失函数解决联邦元学习中的数据泄露问题，实现高效神经场建模，适用于资源受限设备，提升小样本场景下的性能与隐私性。**

- **链接: [http://arxiv.org/pdf/2508.06301v1](http://arxiv.org/pdf/2508.06301v1)**

> **作者:** Junhyeog Yun; Minui Hong; Gunhee Kim
>
> **备注:** ICCV 2025
>
> **摘要:** Neural fields provide a memory-efficient representation of data, which can effectively handle diverse modalities and large-scale data. However, learning to map neural fields often requires large amounts of training data and computations, which can be limited to resource-constrained edge devices. One approach to tackle this limitation is to leverage Federated Meta-Learning (FML), but traditional FML approaches suffer from privacy leakage. To address these issues, we introduce a novel FML approach called FedMeNF. FedMeNF utilizes a new privacy-preserving loss function that regulates privacy leakage in the local meta-optimization. This enables the local meta-learner to optimize quickly and efficiently without retaining the client's private data. Our experiments demonstrate that FedMeNF achieves fast optimization speed and robust reconstruction performance, even with few-shot or non-IID data across diverse data modalities, while preserving client data privacy.
>
---
#### [new 113] Clinically-guided Data Synthesis for Laryngeal Lesion Detection
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于临床指导的合成数据生成方法，解决喉部病变检测中数据稀缺问题，通过LDM和ControlNet生成高质量标注数据，提升检测准确率。**

- **链接: [http://arxiv.org/pdf/2508.06182v1](http://arxiv.org/pdf/2508.06182v1)**

> **作者:** Chiara Baldini; Kaisar Kushibar; Richard Osuala; Simone Balocco; Oliver Diaz; Karim Lekadir; Leonardo S. Mattos
>
> **摘要:** Although computer-aided diagnosis (CADx) and detection (CADe) systems have made significant progress in various medical domains, their application is still limited in specialized fields such as otorhinolaryngology. In the latter, current assessment methods heavily depend on operator expertise, and the high heterogeneity of lesions complicates diagnosis, with biopsy persisting as the gold standard despite its substantial costs and risks. A critical bottleneck for specialized endoscopic CADx/e systems is the lack of well-annotated datasets with sufficient variability for real-world generalization. This study introduces a novel approach that exploits a Latent Diffusion Model (LDM) coupled with a ControlNet adapter to generate laryngeal endoscopic image-annotation pairs, guided by clinical observations. The method addresses data scarcity by conditioning the diffusion process to produce realistic, high-quality, and clinically relevant image features that capture diverse anatomical conditions. The proposed approach can be leveraged to expand training datasets for CADx/e models, empowering the assessment process in laryngology. Indeed, during a downstream task of detection, the addition of only 10% synthetic data improved the detection rate of laryngeal lesions by 9% when the model was internally tested and 22.1% on out-of-domain external data. Additionally, the realism of the generated images was evaluated by asking 5 expert otorhinolaryngologists with varying expertise to rate their confidence in distinguishing synthetic from real images. This work has the potential to accelerate the development of automated tools for laryngeal disease diagnosis, offering a solution to data scarcity and demonstrating the applicability of synthetic data in real-world scenarios.
>
---
## 更新

#### [replaced 001] TD3Net: A Temporal Densely Connected Multi-Dilated Convolutional Network for Lipreading
- **分类: cs.CV; I.4.8; I.5.4; I.2.10**

- **链接: [http://arxiv.org/pdf/2506.16073v2](http://arxiv.org/pdf/2506.16073v2)**

> **作者:** Byung Hoon Lee; Wooseok Shin; Sung Won Han
>
> **备注:** Accepted for publication in Journal of Visual Communication and Image Representation. DOI: https://doi.org/10.1016/j.jvcir.2025.104540
>
> **摘要:** The word-level lipreading approach typically employs a two-stage framework with separate frontend and backend architectures to model dynamic lip movements. Each component has been extensively studied, and in the backend architecture, temporal convolutional networks (TCNs) have been widely adopted in state-of-the-art methods. Recently, dense skip connections have been introduced in TCNs to mitigate the limited density of the receptive field, thereby improving the modeling of complex temporal representations. However, their performance remains constrained owing to potential information loss regarding the continuous nature of lip movements, caused by blind spots in the receptive field. To address this limitation, we propose TD3Net, a temporal densely connected multi-dilated convolutional network that combines dense skip connections and multi-dilated temporal convolutions as the backend architecture. TD3Net covers a wide and dense receptive field without blind spots by applying different dilation factors to skip-connected features. Experimental results on a word-level lipreading task using two large publicly available datasets, Lip Reading in the Wild (LRW) and LRW-1000, indicate that the proposed method achieves performance comparable to state-of-the-art methods. It achieved higher accuracy with fewer parameters and lower floating-point operations compared to existing TCN-based backend architectures. Moreover, visualization results suggest that our approach effectively utilizes diverse temporal features while preserving temporal continuity, presenting notable advantages in lipreading systems. The code is available at our GitHub repository: https://github.com/Leebh-kor/TD3Net-A-Temporal-Densely-Connected-Multi-dilated-Convolutional-Network-for-Lipreading
>
---
#### [replaced 002] FLUX-Text: A Simple and Advanced Diffusion Transformer Baseline for Scene Text Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03329v3](http://arxiv.org/pdf/2505.03329v3)**

> **作者:** Rui Lan; Yancheng Bai; Xu Duan; Mingxing Li; Dongyang Jin; Ryan Xu; Lei Sun; Xiangxiang Chu
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Scene text editing aims to modify or add texts on images while ensuring text fidelity and overall visual quality consistent with the background. Recent methods are primarily built on UNet-based diffusion models, which have improved scene text editing results, but still struggle with complex glyph structures, especially for non-Latin ones (\eg, Chinese, Korean, Japanese). To address these issues, we present \textbf{FLUX-Text}, a simple and advanced multilingual scene text editing DiT method. Specifically, our FLUX-Text enhances glyph understanding and generation through lightweight Visual and Text Embedding Modules, while preserving the original generative capability of FLUX. We further propose a Regional Text Perceptual Loss tailored for text regions, along with a matching two-stage training strategy to better balance text editing and overall image quality. Benefiting from the DiT-based architecture and lightweight feature injection modules, FLUX-Text can be trained with only $0.1$M training examples, a \textbf{97\%} reduction compared to $2.9$M required by popular methods. Extensive experiments on multiple public datasets, including English and Chinese benchmarks, demonstrate that our method surpasses other methods in visual quality and text fidelity. All the code is available at https://github.com/AMAP-ML/FluxText.
>
---
#### [replaced 003] Improved DDIM Sampling with Moment Matching Gaussian Mixtures
- **分类: cs.CV; cs.AI; cs.LG; I.2, I.4**

- **链接: [http://arxiv.org/pdf/2311.04938v4](http://arxiv.org/pdf/2311.04938v4)**

> **作者:** Prasad Gabbur
>
> **备注:** 33 pages, 11 figures; Extension to Rectified Flow Matching with implicit SDE solvers
>
> **摘要:** We propose using a Gaussian Mixture Model (GMM) as reverse transition operator (kernel) within the Denoising Diffusion Implicit Models (DDIM) framework, which is one of the most widely used approaches for accelerated sampling from pre-trained Denoising Diffusion Probabilistic Models (DDPM). Specifically we match the first and second order central moments of the DDPM forward marginals by constraining the parameters of the GMM. We see that moment matching is sufficient to obtain samples with equal or better quality than the original DDIM with Gaussian kernels. We provide experimental results with unconditional models trained on CelebAHQ and FFHQ, class-conditional models trained on ImageNet, and text-to-image generation using Stable Diffusion v2.1 on COYO700M datasets respectively. Our results suggest that using the GMM kernel leads to significant improvements in the quality of the generated samples when the number of sampling steps is small, as measured by FID and IS metrics. For example on ImageNet 256x256, using 10 sampling steps, we achieve a FID of 6.94 and IS of 207.85 with a GMM kernel compared to 10.15 and 196.73 respectively with a Gaussian kernel. Further, we derive novel SDE samplers for rectified flow matching models and experiment with the proposed approach. We see improvements using both 1-rectified flow and 2-rectified flow models.
>
---
#### [replaced 004] DAVSP: Safety Alignment for Large Vision-Language Models via Deep Aligned Visual Safety Prompt
- **分类: cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09353v2](http://arxiv.org/pdf/2506.09353v2)**

> **作者:** Yitong Zhang; Jia Li; Liyi Cai; Ge Li
>
> **备注:** 16 pages
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved impressive progress across various applications but remain vulnerable to malicious queries that exploit the visual modality. Existing alignment approaches typically fail to resist malicious queries while preserving utility on benign ones effectively. To address these challenges, we propose Deep Aligned Visual Safety Prompt (DAVSP), which is built upon two key innovations. First, we introduce the Visual Safety Prompt, which appends a trainable padding region around the input image. It preserves visual features and expands the optimization space. Second, we propose Deep Alignment, a novel approach to train the visual safety prompt through supervision in the model's activation space. It enhances the inherent ability of LVLMs to perceive malicious queries, achieving deeper alignment than prior works. Extensive experiments across five benchmarks on two representative LVLMs demonstrate that DAVSP effectively resists malicious queries while preserving benign input utility. Furthermore, DAVSP exhibits great cross-model generation ability. Ablation studies further reveal that both the Visual Safety Prompt and Deep Alignment are essential components, jointly contributing to its overall effectiveness. The code is publicly available at https://github.com/zhangyitonggg/DAVSP.
>
---
#### [replaced 005] MBA-SLAM: Motion Blur Aware Gaussian Splatting SLAM
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.08279v2](http://arxiv.org/pdf/2411.08279v2)**

> **作者:** Peng Wang; Lingzhe Zhao; Yin Zhang; Shiyu Zhao; Peidong Liu
>
> **备注:** Accepted to TPAMI; Deblur Gaussian Splatting SLAM
>
> **摘要:** Emerging 3D scene representations, such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have demonstrated their effectiveness in Simultaneous Localization and Mapping (SLAM) for photo-realistic rendering, particularly when using high-quality video sequences as input. However, existing methods struggle with motion-blurred frames, which are common in real-world scenarios like low-light or long-exposure conditions. This often results in a significant reduction in both camera localization accuracy and map reconstruction quality. To address this challenge, we propose a dense visual deblur SLAM pipeline (i.e. MBA-SLAM) to handle severe motion-blurred inputs and enhance image deblurring. Our approach integrates an efficient motion blur-aware tracker with either neural radiance fields or Gaussian Splatting based mapper. By accurately modeling the physical image formation process of motion-blurred images, our method simultaneously learns 3D scene representation and estimates the cameras' local trajectory during exposure time, enabling proactive compensation for motion blur caused by camera movement. In our experiments, we demonstrate that MBA-SLAM surpasses previous state-of-the-art methods in both camera localization and map reconstruction, showcasing superior performance across a range of datasets, including synthetic and real datasets featuring sharp images as well as those affected by motion blur, highlighting the versatility and robustness of our approach. Code is available at https://github.com/WU-CVGL/MBA-SLAM.
>
---
#### [replaced 006] Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.08566v2](http://arxiv.org/pdf/2409.08566v2)**

> **作者:** Hyewon Park; Hyejin Park; Jueun Ko; Dongbo Min
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Continual Test Time Adaptation (CTTA) has emerged as a critical approach for bridging the domain gap between the controlled training environments and the real-world scenarios, enhancing model adaptability and robustness. Existing CTTA methods, typically categorized into Full-Tuning (FT) and Efficient-Tuning (ET), struggle with effectively addressing domain shifts. To overcome these challenges, we propose Hybrid-TTA, a holistic approach that dynamically selects instance-wise tuning method for optimal adaptation. Our approach introduces the Dynamic Domain Shift Detection (DDSD) strategy, which identifies domain shifts by leveraging temporal correlations in input sequences and dynamically switches between FT and ET to adapt to varying domain shifts effectively. Additionally, the Masked Image Modeling based Adaptation (MIMA) framework is integrated to ensure domain-agnostic robustness with minimal computational overhead. Our Hybrid-TTA achieves a notable 1.6%p improvement in mIoU on the Cityscapes-to-ACDC benchmark dataset, surpassing previous state-of-the-art methods and offering a robust solution for real-world continual adaptation challenges.
>
---
#### [replaced 007] TurboTrain: Towards Efficient and Balanced Multi-Task Learning for Multi-Agent Perception and Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04682v2](http://arxiv.org/pdf/2508.04682v2)**

> **作者:** Zewei Zhou; Seth Z. Zhao; Tianhui Cai; Zhiyu Huang; Bolei Zhou; Jiaqi Ma
>
> **备注:** ICCV 2025
>
> **摘要:** End-to-end training of multi-agent systems offers significant advantages in improving multi-task performance. However, training such models remains challenging and requires extensive manual design and monitoring. In this work, we introduce TurboTrain, a novel and efficient training framework for multi-agent perception and prediction. TurboTrain comprises two key components: a multi-agent spatiotemporal pretraining scheme based on masked reconstruction learning and a balanced multi-task learning strategy based on gradient conflict suppression. By streamlining the training process, our framework eliminates the need for manually designing and tuning complex multi-stage training pipelines, substantially reducing training time and improving performance. We evaluate TurboTrain on a real-world cooperative driving dataset, V2XPnP-Seq, and demonstrate that it further improves the performance of state-of-the-art multi-agent perception and prediction models. Our results highlight that pretraining effectively captures spatiotemporal multi-agent features and significantly benefits downstream tasks. Moreover, the proposed balanced multi-task learning strategy enhances detection and prediction.
>
---
#### [replaced 008] InterAct-Video: Reasoning-Rich Video QA for Urban Traffic
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.14743v2](http://arxiv.org/pdf/2507.14743v2)**

> **作者:** Joseph Raj Vishal; Rutuja Patil; Manas Srinivas Gowda; Katha Naik; Yezhou Yang; Bharatesh Chakravarthi
>
> **摘要:** Traffic monitoring is crucial for urban mobility, road safety, and intelligent transportation systems (ITS). Deep learning has advanced video-based traffic monitoring through video question answering (VideoQA) models, enabling structured insight extraction from traffic videos. However, existing VideoQA models struggle with the complexity of real-world traffic scenes, where multiple concurrent events unfold across spatiotemporal dimensions. To address these challenges, this paper introduces \textbf{InterAct VideoQA}, a curated dataset designed to benchmark and enhance VideoQA models for traffic monitoring tasks. The InterAct VideoQA dataset comprises 8 hours of real-world traffic footage collected from diverse intersections, segmented into 10-second video clips, with over 25,000 question-answer (QA) pairs covering spatiotemporal dynamics, vehicle interactions, incident detection, and other critical traffic attributes. State-of-the-art VideoQA models are evaluated on InterAct VideoQA, exposing challenges in reasoning over fine-grained spatiotemporal dependencies within complex traffic scenarios. Additionally, fine-tuning these models on InterAct VideoQA yields notable performance improvements, demonstrating the necessity of domain-specific datasets for VideoQA. InterAct VideoQA is publicly available as a benchmark dataset to facilitate future research in real-world deployable VideoQA models for intelligent transportation systems. GitHub Repo: https://github.com/joe-rabbit/InterAct_VideoQA
>
---
#### [replaced 009] M$^2$IV: Towards Efficient and Fine-grained Multimodal In-Context Learning via Representation Engineering
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04633v2](http://arxiv.org/pdf/2504.04633v2)**

> **作者:** Yanshu Li; Yi Cao; Hongyang He; Qisen Cheng; Xiang Fu; Xi Xiao; Tianyang Wang; Ruixiang Tang
>
> **备注:** COLM 2025, 30 pages, 10 figures, 16 tables
>
> **摘要:** Multimodal in-context learning (ICL) equips Large Vision-language Models (LVLMs) with the ability to adapt to new tasks via multiple user-provided demonstrations, without requiring any model parameter updates. However, its effectiveness is constrained by the token-intensive nature of multimodal inputs and the complexity of cross-modal few-shot reasoning, which together hinder LVLMs from extracting useful patterns from demonstrations. To address these challenges, we propose \textbf{M$^2$IV}, a novel representation engineering approach that replaces explicit token-level demonstrations with a set of learnable Multimodal In-context Vectors directly injected into the residual streams of LVLMs. By analyzing the distinct roles of multi-head attention (MHA) and multi-layer perceptrons (MLP) in the ICL process, we design a training strategy that enables M$^2$IV to perform fine-grained semantic distillation and robust cross-modal representation learning. M$^2$IV not only improves performance across diverse tasks and LVLMs but also significantly reduces token overhead, enabling graceful scaling to many-shot scenarios. To further enhance usability, we introduce \textbf{VLibrary}, a repository that stores trained M$^2$IVs for flexible retrieval and injection. With VLibrary, users can steer pre-trained LVLMs in a customized manner that meets diverse requirements. Extensive experiments demonstrate that M$^2$IV consistently outperforms vanilla ICL and prior representation engineering baselines, achieving an average accuracy gain of 3.74\% with substantial improvements in overall efficiency.
>
---
#### [replaced 010] ATM: Improving Model Merging by Alternating Tuning and Merging
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.03055v4](http://arxiv.org/pdf/2411.03055v4)**

> **作者:** Luca Zhou; Daniele Solombrino; Donato Crisostomi; Maria Sofia Bucarelli; Fabrizio Silvestri; Emanuele Rodolà
>
> **备注:** Main paper: 9 Pages, 4 figures, 1 table
>
> **摘要:** Model merging has emerged as a cost-efficient approximation to multitask learning. Among merging strategies, task arithmetic is notable for its simplicity and effectiveness. In this work, we provide a theoretical motivation for task vectors by highlighting that, under single-epoch full-batch gradient descent, they are equivalent to multitask gradients. This insight leads us to reinterpret model merging as a single step in an iterative procedure that Alternates between Tuning and Merging (ATM). We propose two applications of ATM: (1) as an alternative to multitask learning in scenarios where data sharing is restricted (e.g., federated settings), and (2) as a lightweight refinement step to improve existing model merging methods using a small validation set. Experiments across diverse vision tasks demonstrate the effectiveness of ATM.
>
---
#### [replaced 011] Building Age Estimation: A New Multi-Modal Benchmark Dataset and Community Challenge
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13818v3](http://arxiv.org/pdf/2502.13818v3)**

> **作者:** Nikolaos Dionelis; Alessandra Feliciotti; Mattia Marconcini; Devis Peressutti; Nika Oman Kadunc; JaeWan Park; Hagai Raja Sinulingga; Steve Andreas Immanuel; Ba Tran; Caroline Arnold; Nicolas Longépé
>
> **备注:** 15 pages, 20 figures, 1 table, Submitted
>
> **摘要:** Estimating the construction year of buildings is critical for advancing sustainability, as older structures often lack energy-efficient features. Sustainable urban planning relies on accurate building age data to reduce energy consumption and mitigate climate change. In this work, we introduce MapYourCity, a novel multi-modal benchmark dataset comprising top-view Very High Resolution (VHR) imagery, multi-spectral Earth Observation (EO) data from the Copernicus Sentinel-2 constellation, and co-localized street-view images across various European cities. Each building is labeled with its construction epoch, and the task is formulated as a seven-class classification problem covering periods from 1900 to the present. To advance research in EO generalization and multi-modal learning, we organized a community-driven data challenge in 2024, hosted by ESA $\Phi$-lab, which ran for four months and attracted wide participation. This paper presents the Top-4 performing models from the challenge and their evaluation results. We assess model generalization on cities excluded from training to prevent data leakage, and evaluate performance under missing modality scenarios, particularly when street-view data is unavailable. Results demonstrate that building age estimation is both feasible and effective, even in previously unseen cities and when relying solely on top-view satellite imagery (i.e. with VHR and Sentinel-2 images). The new MapYourCity dataset thus provides a valuable resource for developing scalable, real-world solutions in sustainable urban analytics.
>
---
#### [replaced 012] Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.09688v2](http://arxiv.org/pdf/2501.09688v2)**

> **作者:** Jiho Choi; Seonho Lee; Minhyun Lee; Seungho Lee; Hyunjung Shim
>
> **备注:** CVPR 2025
>
> **摘要:** Open-Vocabulary Part Segmentation (OVPS) is an emerging field for recognizing fine-grained parts in unseen categories. We identify two primary challenges in OVPS: (1) the difficulty in aligning part-level image-text correspondence, and (2) the lack of structural understanding in segmenting object parts. To address these issues, we propose PartCATSeg, a novel framework that integrates object-aware part-level cost aggregation, compositional loss, and structural guidance from DINO. Our approach employs a disentangled cost aggregation strategy that handles object and part-level costs separately, enhancing the precision of part-level segmentation. We also introduce a compositional loss to better capture part-object relationships, compensating for the limited part annotations. Additionally, structural guidance from DINO features improves boundary delineation and inter-part understanding. Extensive experiments on Pascal-Part-116, ADE20K-Part-234, and PartImageNet datasets demonstrate that our method significantly outperforms state-of-the-art approaches, setting a new baseline for robust generalization to unseen part categories.
>
---
#### [replaced 013] MambaEviScrib: Mamba and Evidence-Guided Consistency Enhance CNN Robustness for Scribble-Based Weakly Supervised Ultrasound Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.19370v3](http://arxiv.org/pdf/2409.19370v3)**

> **作者:** Xiaoxiang Han; Xinyu Li; Jiang Shang; Yiman Liu; Keyan Chen; Shugong Xu; Qiaohong Liu; Qi Zhang
>
> **备注:** Accepted by Information Fusion
>
> **摘要:** Segmenting anatomical structures and lesions from ultrasound images contributes to disease assessment. Weakly supervised learning (WSL) based on sparse annotation has achieved encouraging performance and demonstrated the potential to reduce annotation costs. This study attempts to introduce scribble-based WSL into ultrasound image segmentation tasks. However, ultrasound images often suffer from poor contrast and unclear edges, coupled with insufficient supervison signals for edges, posing challenges to edge prediction. Uncertainty modeling has been proven to facilitate models in dealing with these issues. Nevertheless, existing uncertainty estimation paradigms are not robust enough and often filter out predictions near decision boundaries, resulting in unstable edge predictions. Therefore, we propose leveraging predictions near decision boundaries effectively. Specifically, we introduce Dempster-Shafer Theory (DST) of evidence to design an Evidence-Guided Consistency strategy. This strategy utilizes high-evidence predictions, which are more likely to occur near high-density regions, to guide the optimization of low-evidence predictions that may appear near decision boundaries. Furthermore, the diverse sizes and locations of lesions in ultrasound images pose a challenge for CNNs with local receptive fields, as they struggle to model global information. Therefore, we introduce Visual Mamba based on structured state space sequence models, which achieves long-range dependency with linear computational complexity, and we construct a novel hybrid CNN-Mamba framework. During training, the collaboration between the CNN branch and the Mamba branch in the proposed framework draws inspiration from each other based on the EGC strategy. Experiments demonstrate the competitiveness of the proposed method. Dataset and code will be available on https://github.com/GtLinyer/MambaEviScrib.
>
---
#### [replaced 014] COIN: Confidence Score-Guided Distillation for Annotation-Free Cell Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11439v4](http://arxiv.org/pdf/2503.11439v4)**

> **作者:** Sanghyun Jo; Seo Jin Lee; Seungwoo Lee; Seohyung Hong; Hyungseok Seo; Kyungsu Kim
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Cell instance segmentation (CIS) is crucial for identifying individual cell morphologies in histopathological images, providing valuable insights for biological and medical research. While unsupervised CIS (UCIS) models aim to reduce the heavy reliance on labor-intensive image annotations, they fail to accurately capture cell boundaries, causing missed detections and poor performance. Recognizing the absence of error-free instances as a key limitation, we present COIN (COnfidence score-guided INstance distillation), a novel annotation-free framework with three key steps: (1) Increasing the sensitivity for the presence of error-free instances via unsupervised semantic segmentation with optimal transport, leveraging its ability to discriminate spatially minor instances, (2) Instance-level confidence scoring to measure the consistency between model prediction and refined mask and identify highly confident instances, offering an alternative to ground truth annotations, and (3) Progressive expansion of confidence with recursive self-distillation. Extensive experiments across six datasets show COIN outperforming existing UCIS methods, even surpassing semi- and weakly-supervised approaches across all metrics on the MoNuSeg and TNBC datasets. The code is available at https://github.com/shjo-april/COIN.
>
---
#### [replaced 015] Event2Vec: Processing neuromorphic events directly by representations in vector space
- **分类: cs.CV; cs.NE**

- **链接: [http://arxiv.org/pdf/2504.15371v2](http://arxiv.org/pdf/2504.15371v2)**

> **作者:** Wei Fang; Priyadarshini Panda
>
> **摘要:** The neuromorphic event cameras have overwhelming advantages in temporal resolution, power efficiency, and dynamic range compared to traditional cameras. However, the event cameras output asynchronous, sparse, and irregular events, which are not compatible with mainstream computer vision and deep learning methods. Various methods have been proposed to solve this issue but at the cost of long preprocessing procedures, losing temporal resolutions, or being incompatible with massively parallel computation. Inspired by the great success of the word to vector, we summarize the similarities between words and events, then propose the first event to vector (event2vec) representation. We validate event2vec on classifying the ASL-DVS dataset, showing impressive parameter efficiency, accuracy, and speed than previous graph/image/voxel-based representations. Beyond task performance, the most attractive advantage of event2vec is that it aligns events to the domain of natural language processing, showing the promising prospect of integrating events into large language and multimodal models. Our codes, models, and training logs are available at https://github.com/fangwei123456/event2vec.
>
---
#### [replaced 016] MOR-VIT: Efficient Vision Transformer with Mixture-of-Recursions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.21761v2](http://arxiv.org/pdf/2507.21761v2)**

> **作者:** YiZhou Li
>
> **备注:** 20 pages,9 figuers
>
> **摘要:** Vision Transformers (ViTs) have achieved remarkable success in image recognition, yet standard ViT architectures are hampered by substantial parameter redundancy and high computational cost, limiting their practical deployment. While recent efforts on efficient ViTs primarily focus on static model compression or token-level sparsification, they remain constrained by fixed computational depth for all tokens. In this work, we present MoR-ViT, a novel vision transformer framework that, for the first time, incorporates a token-level dynamic recursion mechanism inspired by the Mixture-of-Recursions (MoR) paradigm. This approach enables each token to adaptively determine its processing depth, yielding a flexible and input-dependent allocation of computational resources. Extensive experiments on ImageNet-1K and transfer benchmarks demonstrate that MoR-ViT not only achieves state-of-the-art accuracy with up to 70% parameter reduction and 2.5x inference acceleration, but also outperforms leading efficient ViT baselines such as DynamicViT and TinyViT under comparable conditions. These results establish dynamic recursion as an effective strategy for efficient vision transformers and open new avenues for scalable and deployable deep learning models in real-world scenarios.
>
---
#### [replaced 017] MoDA: Multi-modal Diffusion Architecture for Talking Head Generation
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03256v3](http://arxiv.org/pdf/2507.03256v3)**

> **作者:** Xinyang Li; Gen Li; Zhihui Lin; Yichen Qian; GongXin Yao; Weinan Jia; Aowen Wang; Weihua Chen; Fan Wang
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Talking head generation with arbitrary identities and speech audio remains a crucial problem in the realm of the virtual metaverse. Recently, diffusion models have become a popular generative technique in this field with their strong generation capabilities. However, several challenges remain for diffusion-based methods: 1) inefficient inference and visual artifacts caused by the implicit latent space of Variational Auto-Encoders (VAE), which complicates the diffusion process; 2) a lack of authentic facial expressions and head movements due to inadequate multi-modal information fusion. In this paper, MoDA handles these challenges by: 1) defining a joint parameter space that bridges motion generation and neural rendering, and leveraging flow matching to simplify diffusion learning; 2) introducing a multi-modal diffusion architecture to model the interaction among noisy motion, audio, and auxiliary conditions, enhancing overall facial expressiveness. In addition, a coarse-to-fine fusion strategy is employed to progressively integrate different modalities, ensuring effective feature fusion. Experimental results demonstrate that MoDA improves video diversity, realism, and efficiency, making it suitable for real-world applications. Project Page: https://lixinyyang.github.io/MoDA.github.io/
>
---
#### [replaced 018] Generative Video Bi-flow
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2503.06364v2](http://arxiv.org/pdf/2503.06364v2)**

> **作者:** Chen Liu; Tobias Ritschel
>
> **备注:** ICCV 2025. Project Page at https://ryushinn.github.io/ode-video
>
> **摘要:** We propose a novel generative video model to robustly learn temporal change as a neural Ordinary Differential Equation (ODE) flow with a bilinear objective which combines two aspects: The first is to map from the past into future video frames directly. Previous work has mapped the noise to new frames, a more computationally expensive process. Unfortunately, starting from the previous frame, instead of noise, is more prone to drifting errors. Hence, second, we additionally learn how to remove the accumulated errors as the joint objective by adding noise during training. We demonstrate unconditional video generation in a streaming manner for various video datasets, all at competitive quality compared to a conditional diffusion baseline but with higher speed, i.e., fewer ODE solver steps.
>
---
#### [replaced 019] Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Free
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.03687v2](http://arxiv.org/pdf/2502.03687v2)**

> **作者:** Gian Mario Favero; Parham Saremi; Emily Kaczmarek; Brennan Nichyporuk; Tal Arbel
>
> **备注:** Accepted for publication at MIDL 2025
>
> **摘要:** Discriminative classifiers have become a foundational tool in deep learning for medical imaging, excelling at learning separable features of complex data distributions. However, these models often need careful design, augmentation, and training techniques to ensure safe and reliable deployment. Recently, diffusion models have become synonymous with generative modeling in 2D. These models showcase robustness across a range of tasks including natural image classification, where classification is performed by comparing reconstruction errors across images generated for each possible conditioning input. This work presents the first exploration of the potential of class conditional diffusion models for 2D medical image classification. First, we develop a novel majority voting scheme shown to improve the performance of medical diffusion classifiers. Next, extensive experiments on the CheXpert and ISIC Melanoma skin cancer datasets demonstrate that foundation and trained-from-scratch diffusion models achieve competitive performance against SOTA discriminative classifiers without the need for explicit supervision. In addition, we show that diffusion classifiers are intrinsically explainable, and can be used to quantify the uncertainty of their predictions, increasing their trustworthiness and reliability in safety-critical, clinical contexts. Further information is available on our project page: https://faverogian.github.io/med-diffusion-classifier.github.io/.
>
---
#### [replaced 020] Localized Gaussians as Self-Attention Weights for Point Clouds Correspondence
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2409.13291v2](http://arxiv.org/pdf/2409.13291v2)**

> **作者:** Alessandro Riva; Alessandro Raganato; Simone Melzi
>
> **摘要:** Current data-driven methodologies for point cloud matching demand extensive training time and computational resources, presenting significant challenges for model deployment and application. In the point cloud matching task, recent advancements with an encoder-only Transformer architecture have revealed the emergence of semantically meaningful patterns in the attention heads, particularly resembling Gaussian functions centered on each point of the input shape. In this work, we further investigate this phenomenon by integrating these patterns as fixed attention weights within the attention heads of the Transformer architecture. We evaluate two variants: one utilizing predetermined variance values for the Gaussians, and another where the variance values are treated as learnable parameters. Additionally we analyze the performances on noisy data and explore a possible way to improve robustness to noise. Our findings demonstrate that fixing the attention weights not only accelerates the training process but also enhances the stability of the optimization. Furthermore, we conducted an ablation study to identify the specific layers where the infused information is most impactful and to understand the reliance of the network on this information.
>
---
#### [replaced 021] Your other Left! Vision-Language Models Fail to Identify Relative Positions in Medical Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00549v2](http://arxiv.org/pdf/2508.00549v2)**

> **作者:** Daniel Wolf; Heiko Hillenhagen; Billurvan Taskin; Alex Bäuerle; Meinrad Beer; Michael Götz; Timo Ropinski
>
> **备注:** Accepted at the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025
>
> **摘要:** Clinical decision-making relies heavily on understanding relative positions of anatomical structures and anomalies. Therefore, for Vision-Language Models (VLMs) to be applicable in clinical practice, the ability to accurately determine relative positions on medical images is a fundamental prerequisite. Despite its importance, this capability remains highly underexplored. To address this gap, we evaluate the ability of state-of-the-art VLMs, GPT-4o, Llama3.2, Pixtral, and JanusPro, and find that all models fail at this fundamental task. Inspired by successful approaches in computer vision, we investigate whether visual prompts, such as alphanumeric or colored markers placed on anatomical structures, can enhance performance. While these markers provide moderate improvements, results remain significantly lower on medical images compared to observations made on natural images. Our evaluations suggest that, in medical imaging, VLMs rely more on prior anatomical knowledge than on actual image content for answering relative position questions, often leading to incorrect conclusions. To facilitate further research in this area, we introduce the MIRP , Medical Imaging Relative Positioning, benchmark dataset, designed to systematically evaluate the capability to identify relative positions in medical images.
>
---
#### [replaced 022] SPA++: Generalized Graph Spectral Alignment for Versatile Domain Adaptation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.05182v2](http://arxiv.org/pdf/2508.05182v2)**

> **作者:** Zhiqing Xiao; Haobo Wang; Xu Lu; Wentao Ye; Gang Chen; Junbo Zhao
>
> **备注:** The article has been accepted by Frontiers of Computer Science (FCS), with the DOI: {10.1007/s11704-025-50328-w}. It is an extended journal version of the conference paper arXiv:2310.17594
>
> **摘要:** Domain Adaptation (DA) aims to transfer knowledge from a labeled source domain to an unlabeled or sparsely labeled target domain under domain shifts. Most prior works focus on capturing the inter-domain transferability but largely overlook rich intra-domain structures, which empirically results in even worse discriminability. To tackle this tradeoff, we propose a generalized graph SPectral Alignment framework, SPA++. Its core is briefly condensed as follows: (1)-by casting the DA problem to graph primitives, it composes a coarse graph alignment mechanism with a novel spectral regularizer toward aligning the domain graphs in eigenspaces; (2)-we further develop a fine-grained neighbor-aware propagation mechanism for enhanced discriminability in the target domain; (3)-by incorporating data augmentation and consistency regularization, SPA++ can adapt to complex scenarios including most DA settings and even challenging distribution scenarios. Furthermore, we also provide theoretical analysis to support our method, including the generalization bound of graph-based DA and the role of spectral alignment and smoothing consistency. Extensive experiments on benchmark datasets demonstrate that SPA++ consistently outperforms existing cutting-edge methods, achieving superior robustness and adaptability across various challenging adaptation scenarios.
>
---
#### [replaced 023] Direct Robot Configuration Space Construction using Convolutional Encoder-Decoders
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2303.05653v2](http://arxiv.org/pdf/2303.05653v2)**

> **作者:** Christopher Benka; Judah Goldfeder; Carl Gross; Riya Gupta; Hod Lipson
>
> **备注:** 8 pages, 7 figures, 4 tables; Appeared at the ICML 2025 Workshop on Building Physically Plausible World Models
>
> **摘要:** Intelligent robots must be able to perform safe and efficient motion planning in their environments. Central to modern motion planning is the configuration space. Configuration spaces define the set of configurations of a robot that result in collisions with obstacles in the workspace, $\text{C}_{\text{clsn}}$, and the set of configurations that do not, $\text{C}_{\text{free}}$. Modern approaches to motion planning first compute the configuration space and then perform motion planning using the calculated configuration space. Real-time motion planning requires accurate and efficient construction of configuration spaces. We are the first to apply a convolutional encoder-decoder framework for calculating highly accurate approximations to configuration spaces, essentially learning how the robot and physical world interact. Our model achieves an average 97.5% F1-score for predicting $\text{C}_{\text{free}}$ and $\text{C}_{\text{clsn}}$ for 2-D robotic workspaces with a dual-arm robot. Our method limits undetected collisions to less than 2.5% on robotic workspaces that involve translation, rotation, and removal of obstacles. Our model learns highly transferable features between robotic workspaces, requiring little to no fine-tuning to adapt to new transformations of obstacles in the workspace.
>
---
#### [replaced 024] A Study of Gender Classification Techniques Based on Iris Images: A Deep Survey and Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.05246v2](http://arxiv.org/pdf/2508.05246v2)**

> **作者:** Basna Mohammed Salih Hasan; Ramadhan J. Mstafa
>
> **备注:** 13 Pages, 8 Figures, 1 Table
>
> **摘要:** Gender classification is attractive in a range of applications, including surveillance and monitoring, corporate profiling, and human-computer interaction. Individuals' identities may be gleaned from information about their gender, which is a kind of soft biometric. Over the years, several methods for determining a person's gender have been devised. Some of the most well-known ones are based on physical characteristics like face, fingerprint, palmprint, DNA, ears, gait, and iris. On the other hand, facial features account for the vast majority of gender classification methods. Also, the iris is a significant biometric trait because the iris, according to research, remains basically constant during an individual's life. Besides that, the iris is externally visible and is non-invasive to the user, which is important for practical applications. Furthermore, there are already high-quality methods for segmenting and encoding iris images, and the current methods facilitate selecting and extracting attribute vectors from iris textures. This study discusses several approaches to determining gender. The previous works of literature are briefly reviewed. Additionally, there are a variety of methodologies for different steps of gender classification. This study provides researchers with knowledge and analysis of the existing gender classification approaches. Also, it will assist researchers who are interested in this specific area, as well as highlight the gaps and challenges in the field, and finally provide suggestions and future paths for improvement.
>
---
#### [replaced 025] CARE: Enhancing Safety of Visual Navigation through Collision Avoidance via Repulsive Estimation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03834v4](http://arxiv.org/pdf/2506.03834v4)**

> **作者:** Joonkyung Kim; Joonyeol Sim; Woojun Kim; Katia Sycara; Changjoo Nam
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** We propose CARE (Collision Avoidance via Repulsive Estimation) to improve the robustness of learning-based visual navigation methods. Recently, visual navigation models, particularly foundation models, have demonstrated promising performance by generating viable trajectories using only RGB images. However, these policies can generalize poorly to environments containing out-of-distribution (OOD) scenes characterized by unseen objects or different camera setups (e.g., variations in field of view, camera pose, or focal length). Without fine-tuning, such models could produce trajectories that lead to collisions, necessitating substantial efforts in data collection and additional training. To address this limitation, we introduce CARE, an attachable module that enhances the safety of visual navigation without requiring additional range sensors or fine-tuning of pretrained models. CARE can be integrated seamlessly into any RGB-based navigation model that generates local robot trajectories. It dynamically adjusts trajectories produced by a pretrained model using repulsive force vectors computed from depth images estimated directly from RGB inputs. We evaluate CARE by integrating it with state-of-the-art visual navigation models across diverse robot platforms. Real-world experiments show that CARE significantly reduces collisions (up to 100%) without compromising navigation performance in goal-conditioned navigation, and further improves collision-free travel distance (up to 10.7x) in exploration tasks. Project page: https://airlab-sogang.github.io/CARE/
>
---
#### [replaced 026] Can Large Pretrained Depth Estimation Models Help With Image Dehazing?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00698v2](http://arxiv.org/pdf/2508.00698v2)**

> **作者:** Hongfei Zhang; Kun Zhou; Ruizheng Wu; Jiangbo Lu
>
> **摘要:** Image dehazing remains a challenging problem due to the spatially varying nature of haze in real-world scenes. While existing methods have demonstrated the promise of large-scale pretrained models for image dehazing, their architecture-specific designs hinder adaptability across diverse scenarios with different accuracy and efficiency requirements. In this work, we systematically investigate the generalization capability of pretrained depth representations-learned from millions of diverse images-for image dehazing. Our empirical analysis reveals that the learned deep depth features maintain remarkable consistency across varying haze levels. Building on this insight, we propose a plug-and-play RGB-D fusion module that seamlessly integrates with diverse dehazing architectures. Extensive experiments across multiple benchmarks validate both the effectiveness and broad applicability of our approach.
>
---
#### [replaced 027] Vision-Language Model-Based Semantic-Guided Imaging Biomarker for Lung Nodule Malignancy Prediction
- **分类: cs.CV; cs.AI; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2504.21344v2](http://arxiv.org/pdf/2504.21344v2)**

> **作者:** Luoting Zhuang; Seyed Mohammad Hossein Tabatabaei; Ramin Salehi-Rad; Linh M. Tran; Denise R. Aberle; Ashley E. Prosper; William Hsu
>
> **摘要:** Machine learning models have utilized semantic features, deep features, or both to assess lung nodule malignancy. However, their reliance on manual annotation during inference, limited interpretability, and sensitivity to imaging variations hinder their application in real-world clinical settings. Thus, this research aims to integrate semantic features derived from radiologists' assessments of nodules, guiding the model to learn clinically relevant, robust, and explainable imaging features for predicting lung cancer. We obtained 938 low-dose CT scans from the National Lung Screening Trial (NLST) with 1,246 nodules and semantic features. Additionally, the Lung Image Database Consortium dataset contains 1,018 CT scans, with 2,625 lesions annotated for nodule characteristics. Three external datasets were obtained from UCLA Health, the LUNGx Challenge, and the Duke Lung Cancer Screening. We fine-tuned a pretrained Contrastive Language-Image Pretraining (CLIP) model with a parameter-efficient fine-tuning approach to align imaging and semantic text features and predict the one-year lung cancer diagnosis. Our model outperformed state-of-the-art (SOTA) models in the NLST test set with an AUROC of 0.901 and AUPRC of 0.776. It also showed robust results in external datasets. Using CLIP, we also obtained predictions on semantic features through zero-shot inference, such as nodule margin (AUROC: 0.812), nodule consistency (0.812), and pleural attachment (0.840). Our approach surpasses the SOTA models in predicting lung cancer across datasets collected from diverse clinical settings, providing explainable outputs, aiding clinicians in comprehending the underlying meaning of model predictions. This approach also prevents the model from learning shortcuts and generalizes across clinical settings. The code is available at https://github.com/luotingzhuang/CLIP_nodule.
>
---
#### [replaced 028] CDI: Blind Image Restoration Fidelity Evaluation based on Consistency with Degraded Image
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.14264v2](http://arxiv.org/pdf/2501.14264v2)**

> **作者:** Xiaojun Tang; Jingru Wang; Guangwei Huang; Guannan Chen; Rui Zheng; Lian Huai; Yuyu Liu; Xingqun Jiang
>
> **摘要:** Recent advancements in Blind Image Restoration (BIR) methods, based on Generative Adversarial Networks and Diffusion Models, have significantly improved visual quality. However, they present significant challenges for Image Quality Assessment (IQA), as the existing Full-Reference IQA methods often rate images with high perceptual quality poorly. In this paper, we reassess the Solution Non-Uniqueness and Degradation Indeterminacy issues of BIR, and propose constructing a specific BIR IQA system. In stead of directly comparing a restored image with a reference image, the BIR IQA evaluates fidelity by calculating the Consistency with Degraded Image (CDI). Specifically, we propose a wavelet domain Reference Guided CDI algorithm, which can acquire the consistency with a degraded image for various types without requiring knowledge of degradation parameters. The supported degradation types include down sampling, blur, noise, JPEG and complex combined degradations etc. In addition, we propose a Reference Agnostic CDI, enabling BIR fidelity evaluation without reference images. Finally, in order to validate the rationality of CDI, we create a new Degraded Images Switch Display Comparison Dataset (DISDCD) for subjective evaluation of BIR fidelity. Experiments conducted on DISDCD verify that CDI is markedly superior to common Full Reference IQA methods for BIR fidelity evaluation. The source code and the DISDCD dataset will be publicly available shortly.
>
---
#### [replaced 029] Advancing Welding Defect Detection in Maritime Operations via Adapt-WeldNet and Defect Detection Interpretability Analysis
- **分类: cs.CV; cs.AI; cs.CE; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.00381v2](http://arxiv.org/pdf/2508.00381v2)**

> **作者:** Kamal Basha S; Athira Nambiar
>
> **摘要:** Weld defect detection is crucial for ensuring the safety and reliability of piping systems in the oil and gas industry, especially in challenging marine and offshore environments. Traditional non-destructive testing (NDT) methods often fail to detect subtle or internal defects, leading to potential failures and costly downtime. Furthermore, existing neural network-based approaches for defect classification frequently rely on arbitrarily selected pretrained architectures and lack interpretability, raising safety concerns for deployment. To address these challenges, this paper introduces ``Adapt-WeldNet", an adaptive framework for welding defect detection that systematically evaluates various pre-trained architectures, transfer learning strategies, and adaptive optimizers to identify the best-performing model and hyperparameters, optimizing defect detection and providing actionable insights. Additionally, a novel Defect Detection Interpretability Analysis (DDIA) framework is proposed to enhance system transparency. DDIA employs Explainable AI (XAI) techniques, such as Grad-CAM and LIME, alongside domain-specific evaluations validated by certified ASNT NDE Level II professionals. Incorporating a Human-in-the-Loop (HITL) approach and aligning with the principles of Trustworthy AI, DDIA ensures the reliability, fairness, and accountability of the defect detection system, fostering confidence in automated decisions through expert validation. By improving both performance and interpretability, this work enhances trust, safety, and reliability in welding defect detection systems, supporting critical operations in offshore and marine environments.
>
---
#### [replaced 030] ShadowMamba: State-Space Model with Boundary-Region Selective Scan for Shadow Removal
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.03260v4](http://arxiv.org/pdf/2411.03260v4)**

> **作者:** Xiujin Zhu; Chee-Onn Chow; Joon Huang Chuah
>
> **摘要:** Image shadow removal is a typical low-level vision task. Shadows cause local brightness shifts, which reduce the performance of downstream vision tasks. Currently, Transformer-based shadow removal methods suffer from quadratic computational complexity due to the self-attention mechanism. To improve efficiency, many approaches use local attention, but this limits the ability to model global information and weakens the perception of brightness changes between regions. Recently, Mamba has shown strong performance in vision tasks by enabling global modeling with linear complexity. However, existing scanning strategies are not suitable for shadow removal, as they ignore the semantic continuity of shadow boundaries and internal regions. To address this, this paper proposes a boundary-region selective scanning mechanism that captures local details while enhancing semantic continuity between them, effectively improving shadow removal performance. In addition, a shadow mask denoising method is introduced to support the scanning mechanism and improve data quality. Based on these techniques, this paper presents a model called ShadowMamba, the first Mamba-based model designed for shadow removal. Experimental results show that the proposed method outperforms existing mainstream approaches on the AISTD, ISTD, and SRD datasets, and also offers clear advantages in parameter efficiency and computational complexity. Code is available at: https://github.com/ZHUXIUJINChris/ShadowMamba
>
---
#### [replaced 031] SAR Strikes Back: A New Hope for RSVQA
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08131v2](http://arxiv.org/pdf/2501.08131v2)**

> **作者:** Lucrezia Tosato; Flora Weissgerber; Laurent Wendling; Sylvain Lobry
>
> **备注:** Accepted at IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 13 pages, 6 figures
>
> **摘要:** Remote Sensing Visual Question Answering (RSVQA) is a task that extracts information from satellite images to answer questions in natural language, aiding image interpretation. While several methods exist for optical images with varying spectral bands and resolutions, only recently have high-resolution Synthetic Aperture Radar (SAR) images been explored. SAR's ability to operate in all weather conditions and capture electromagnetic features makes it a promising modality, yet no study has compared SAR and optical imagery in RSVQA or proposed effective fusion strategies. This work investigates how to integrate SAR data into RSVQA and how to best combine it with optical images. We present a dataset that enables SAR-based RSVQA and explore two pipelines for the task. The first is an end-to-end model, while the second is a two-stage framework: SAR information is first extracted and translated into text, which is then processed by a language model to produce the final answer. Our results show that the two-stage model performs better, improving accuracy by nearly 10% over the end-to-end approach. We also evaluate fusion strategies for combining SAR and optical data. A decision-level fusion yields the best results, with an F1-micro score of 75.00%, F1-average of 81.21%, and overall accuracy of 75.49% on the proposed dataset. SAR proves especially beneficial for questions related to specific land cover types, such as water areas, demonstrating its value as a complementary modality to optical imagery.
>
---
#### [replaced 032] Soft Dice Confidence: A Near-Optimal Confidence Estimator for Selective Prediction in Semantic Segmentation
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2402.10665v4](http://arxiv.org/pdf/2402.10665v4)**

> **作者:** Bruno Laboissiere Camargos Borges; Bruno Machado Pacheco; Danilo Silva
>
> **备注:** 42 pages, 9 figures
>
> **摘要:** In semantic segmentation, even state-of-the-art deep learning models fall short of the performance required in certain high-stakes applications such as medical image analysis. In these cases, performance can be improved by allowing a model to abstain from making predictions when confidence is low, an approach known as selective prediction. While well-known in the classification literature, selective prediction has been underexplored in the context of semantic segmentation. This paper tackles the problem by focusing on image-level abstention, which involves producing a single confidence estimate for the entire image, in contrast to previous approaches that focus on pixel-level uncertainty. Assuming the Dice coefficient as the evaluation metric for segmentation, two main contributions are provided in this paper: (i) In the case of known marginal posterior probabilities, we derive the optimal confidence estimator, which is observed to be intractable for typical image sizes. Then, an approximation computable in linear time, named Soft Dice Confidence (SDC), is proposed and proven to be tightly bounded to the optimal estimator. (ii) When only an estimate of the marginal posterior probabilities are known, we propose a plug-in version of the SDC and show it outperforms all previous methods, including those requiring additional tuning data. These findings are supported by experimental results on both synthetic data and real-world data from six medical imaging tasks, including out-of-distribution scenarios, positioning the SDC as a reliable and efficient tool for selective prediction in semantic segmentation.
>
---
#### [replaced 033] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10143v5](http://arxiv.org/pdf/2504.10143v5)**

> **作者:** Yichao Cai; Yuhang Liu; Erdun Gao; Tianjiao Jiang; Zhen Zhang; Anton van den Hengel; Javen Qinfeng Shi
>
> **摘要:** Multimodal representation learning, exemplified by multimodal contrastive learning (MMCL) using image-text pairs, aims to learn powerful representations by aligning cues across modalities. This approach relies on the core assumption that the exemplar image-text pairs constitute two representations of an identical concept. However, recent research has revealed that real-world datasets often exhibit cross-modal misalignment. There are two distinct viewpoints on how to address this issue: one suggests mitigating the misalignment, and the other leveraging it. We seek here to reconcile these seemingly opposing perspectives, and to provide a practical guide for practitioners. Using latent variable models we thus formalize cross-modal misalignment by introducing two specific mechanisms: Selection bias, where some semantic variables are absent in the text, and perturbation bias, where semantic variables are altered -- both leading to misalignment in data pairs. Our theoretical analysis demonstrates that, under mild assumptions, the representations learned by MMCL capture exactly the information related to the subset of the semantic variables invariant to selection and perturbation biases. This provides a unified perspective for understanding misalignment. Based on this, we further offer actionable insights into how misalignment should inform the design of real-world ML systems. We validate our theoretical findings via extensive empirical studies on both synthetic data and real image-text datasets, shedding light on the nuanced impact of cross-modal misalignment on multimodal representation learning.
>
---
#### [replaced 034] INS-MMBench: A Comprehensive Benchmark for Evaluating LVLMs' Performance in Insurance
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.09105v2](http://arxiv.org/pdf/2406.09105v2)**

> **作者:** Chenwei Lin; Hanjia Lyu; Xian Xu; Jiebo Luo
>
> **备注:** To appear in the International Conference on Computer Vision, ICCV 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) and Multimodal Large Language Models (MLLMs) have demonstrated outstanding performance in various general multimodal applications and have shown increasing promise in specialized domains. However, their potential in the insurance domain-characterized by diverse application scenarios and rich multimodal data-remains largely underexplored. To date, there is no systematic review of multimodal tasks, nor a benchmark specifically designed to assess the capabilities of LVLMs in insurance. This gap hinders the development of LVLMs within the insurance industry. This study systematically reviews and categorizes multimodal tasks for 4 representative types of insurance: auto, property, health, and agricultural. We introduce INS-MMBench, the first hierarchical benchmark tailored for the insurance domain. INS-MMBench encompasses 22 fundamental tasks, 12 meta-tasks and 5 scenario tasks, enabling a comprehensive and progressive assessment from basic capabilities to real-world use cases. We benchmark 11 leading LVLMs, including closed-source models such as GPT-4o and open-source models like LLaVA. Our evaluation validates the effectiveness of INS-MMBench and offers detailed insights into the strengths and limitations of current LVLMs on a variety of insurance-related multimodal tasks. We hope that INS-MMBench will accelerate the integration of LVLMs into the insurance industry and foster interdisciplinary research. Our dataset and evaluation code are available at https://github.com/FDU-INS/INS-MMBench.
>
---
#### [replaced 035] MPG-SAM 2: Adapting SAM 2 with Mask Priors and Global Context for Referring Video Object Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.13667v5](http://arxiv.org/pdf/2501.13667v5)**

> **作者:** Fu Rong; Meng Lan; Qian Zhang; Lefei Zhang
>
> **备注:** ICCV 2025
>
> **摘要:** Referring video object segmentation (RVOS) aims to segment objects in a video according to textual descriptions, which requires the integration of multimodal information and temporal dynamics perception. The Segment Anything Model 2 (SAM 2) has shown great effectiveness across various video segmentation tasks. However, its application to offline RVOS is challenged by the translation of the text into effective prompts and a lack of global context awareness. In this paper, we propose a novel RVOS framework, termed MPG-SAM 2, to address these challenges. Specifically, MPG-SAM 2 employs a unified multimodal encoder to jointly encode video and textual features, generating semantically aligned video and text embeddings, along with multimodal class tokens. A mask prior generator utilizes the video embeddings and class tokens to create pseudo masks of target objects and global context. These masks are fed into the prompt encoder as dense prompts along with multimodal class tokens as sparse prompts to generate accurate prompts for SAM 2. To provide the online SAM 2 with a global view, we introduce a hierarchical global-historical aggregator, which allows SAM 2 to aggregate global and historical information of target objects at both pixel and object levels, enhancing the target representation and temporal consistency. Extensive experiments on several RVOS benchmarks demonstrate the superiority of MPG-SAM 2 and the effectiveness of our proposed modules. The code is available at https://github.com/rongfu-dsb/MPG-SAM2.
>
---
#### [replaced 036] Can Multimodal Large Language Models Understand Spatial Relations?
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.19015v2](http://arxiv.org/pdf/2505.19015v2)**

> **作者:** Jingping Liu; Ziyan Liu; Zhedong Cen; Yan Zhou; Yinan Zou; Weiyan Zhang; Haiyun Jiang; Tong Ruan
>
> **备注:** 13 pages, 7 figures, published to ACL 2025
>
> **摘要:** Spatial relation reasoning is a crucial task for multimodal large language models (MLLMs) to understand the objective world. However, current benchmarks have issues like relying on bounding boxes, ignoring perspective substitutions, or allowing questions to be answered using only the model's prior knowledge without image understanding. To address these issues, we introduce SpatialMQA, a human-annotated spatial relation reasoning benchmark based on COCO2017, which enables MLLMs to focus more on understanding images in the objective world. To ensure data quality, we design a well-tailored annotation procedure, resulting in SpatialMQA consisting of 5,392 samples. Based on this benchmark, a series of closed- and open-source MLLMs are implemented and the results indicate that the current state-of-the-art MLLM achieves only 48.14% accuracy, far below the human-level accuracy of 98.40%. Extensive experimental analyses are also conducted, suggesting the future research directions. The benchmark and codes are available at https://github.com/ziyan-xiaoyu/SpatialMQA.git.
>
---
#### [replaced 037] Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05741v2](http://arxiv.org/pdf/2505.05741v2)**

> **作者:** Zhangchi Hu; Peixi Wu; Jie Chen; Huyue Zhu; Yijun Wang; Yansong Peng; Hebei Li; Xiaoyan Sun
>
> **备注:** Accepted by ACM Multimedia 2025
>
> **摘要:** Tiny object detection plays a vital role in drone surveillance, remote sensing, and autonomous systems, enabling the identification of small targets across vast landscapes. However, existing methods suffer from inefficient feature leverage and high computational costs due to redundant feature processing and rigid query allocation. To address these challenges, we propose Dome-DETR, a novel framework with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection. To reduce feature redundancies, we introduce a lightweight Density-Focal Extractor (DeFE) to produce clustered compact foreground masks. Leveraging these masks, we incorporate Masked Window Attention Sparsification (MWAS) to focus computational resources on the most informative regions via sparse attention. Besides, we propose Progressive Adaptive Query Initialization (PAQI), which adaptively modulates query density across spatial areas for better query allocation. Extensive experiments demonstrate that Dome-DETR achieves state-of-the-art performance (+3.3 AP on AI-TOD-V2 and +2.5 AP on VisDrone) while maintaining low computational complexity and a compact model size. Code is available at https://github.com/RicePasteM/Dome-DETR.
>
---
#### [replaced 038] MESAHA-Net: Multi-Encoders based Self-Adaptive Hard Attention Network with Maximum Intensity Projections for Lung Nodule Segmentation in CT Scan
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2304.01576v2](http://arxiv.org/pdf/2304.01576v2)**

> **作者:** Muhammad Usman; Azka Rehman; Abd Ur Rehman; Abdullah Shahid; Tariq Mahmood Khan; Imran Razzak; Minyoung Chung; Yeong Gil Shin
>
> **摘要:** Accurate lung nodule segmentation is crucial for early-stage lung cancer diagnosis, as it can substantially enhance patient survival rates. Computed tomography (CT) images are widely employed for early diagnosis in lung nodule analysis. However, the heterogeneity of lung nodules, size diversity, and the complexity of the surrounding environment pose challenges for developing robust nodule segmentation methods. In this study, we propose an efficient end-to-end framework, the multi-encoder-based self-adaptive hard attention network (MESAHA-Net), for precise lung nodule segmentation in CT scans. MESAHA-Net comprises three encoding paths, an attention block, and a decoder block, facilitating the integration of three types of inputs: CT slice patches, forward and backward maximum intensity projection (MIP) images, and region of interest (ROI) masks encompassing the nodule. By employing a novel adaptive hard attention mechanism, MESAHA-Net iteratively performs slice-by-slice 2D segmentation of lung nodules, focusing on the nodule region in each slice to generate 3D volumetric segmentation of lung nodules. The proposed framework has been comprehensively evaluated on the LIDC-IDRI dataset, the largest publicly available dataset for lung nodule segmentation. The results demonstrate that our approach is highly robust for various lung nodule types, outperforming previous state-of-the-art techniques in terms of segmentation accuracy and computational complexity, rendering it suitable for real-time clinical implementation.
>
---
#### [replaced 039] End-to-End Fine-Tuning of 3D Texture Generation using Differentiable Rewards
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18331v3](http://arxiv.org/pdf/2506.18331v3)**

> **作者:** AmirHossein Zamani; Tianhao Xie; Amir G. Aghdam; Tiberiu Popa; Eugene Belilovsky
>
> **摘要:** While recent 3D generative models can produce high-quality texture images, they often fail to capture human preferences or meet task-specific requirements. Moreover, a core challenge in the 3D texture generation domain is that most existing approaches rely on repeated calls to 2D text-to-image generative models, which lack an inherent understanding of the 3D structure of the input 3D mesh object. To alleviate these issues, we propose an end-to-end differentiable, reinforcement-learning-free framework that embeds human feedback, expressed as differentiable reward functions, directly into the 3D texture synthesis pipeline. By back-propagating preference signals through both geometric and appearance modules of the proposed framework, our method generates textures that respect the 3D geometry structure and align with desired criteria. To demonstrate its versatility, we introduce three novel geometry-aware reward functions, which offer a more controllable and interpretable pathway for creating high-quality 3D content from natural language. By conducting qualitative, quantitative, and user-preference evaluations against state-of-the-art methods, we demonstrate that our proposed strategy consistently outperforms existing approaches. We will make our implementation code publicly available upon acceptance of the paper.
>
---
#### [replaced 040] A Calibration Tool for Refractive Underwater Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.18018v2](http://arxiv.org/pdf/2405.18018v2)**

> **作者:** Felix Seegräber; Mengkun She; Felix Woelk; Kevin Köser
>
> **备注:** 9 pages, 5 figures, the paper is accepted to the ICCV 2025 Workshop CVAUI-AAMVEM
>
> **摘要:** Many underwater applications rely on vision sensors and require proper camera calibration, i.e. knowing the incoming light ray for each pixel in the image. While for the ideal pinhole camera model all viewing rays intersect in a single 3D point, underwater cameras suffer from - possibly multiple - refractions of light rays at the interfaces of water, glass and air. These changes of direction depend on the position and orientation of the camera inside the water-proof housing, as well as on the shape and properties of the optical window, the port, itself. In recent years explicit models for underwater vision behind common ports such as flat or dome port have been proposed, but the underwater community is still lacking a calibration tool which can determine port parameters through refractive calibration. With this work we provide the first open source implementation of an underwater refractive camera calibration toolbox. It allows end-to-end calibration of underwater vision systems, including camera, stereo and housing calibration for systems with dome or flat ports. The implementation is verified using rendered datasets and real-world experiments.
>
---
#### [replaced 041] Trustworthy Pedestrian Trajectory Prediction via Pattern-Aware Interaction Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13397v2](http://arxiv.org/pdf/2507.13397v2)**

> **作者:** Kaiyuan Zhai; Juan Chen; Chao Wang; Zeyi Xu; Guoming Tang
>
> **摘要:** Accurate and reliable pedestrian trajectory prediction is critical for the safety and robustness of intelligent applications, yet achieving trustworthy prediction remains highly challenging due to the complexity of interactions among pedestrians. Previous methods often adopt black-box modeling of pedestrian interactions, treating all neighbors uniformly. Despite their strong performance, such opaque modeling limits the reliability of predictions in safety-critical real-world deployments. To address this issue, we propose InSyn (Interaction-Synchronization Network), a novel Transformer-based model that explicitly captures diverse interaction patterns (e.g., walking in sync or conflicting) while effectively modeling direction-sensitive social behaviors. Additionally, we introduce a training strategy, termed Seq-Start of Seq (SSOS), designed to alleviate the common issue of initial-step divergence in numerical time-series prediction. Experiments on the ETH and UCY datasets demonstrate that our model not only outperforms recent black-box baselines in prediction accuracy, especially under high-density scenarios, but also provides stronger interpretability, achieving a favorable trade-off between reliability and accuracy. Furthermore, the SSOS strategy proves to be effective in improving sequential prediction performance, reducing the initial-step prediction error by approximately 6.58%.
>
---
#### [replaced 042] Survival Modeling from Whole Slide Images via Patch-Level Graph Clustering and Mixture Density Experts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16476v2](http://arxiv.org/pdf/2507.16476v2)**

> **作者:** Ardhendu Sekhar; Vasu Soni; Keshav Aske; Garima Jain; Pranav Jeevan; Amit Sethi
>
> **摘要:** We introduce a modular framework for predicting cancer-specific survival from whole slide pathology images (WSIs) that significantly improves upon the state-of-the-art accuracy. Our method integrating four key components. Firstly, to tackle large size of WSIs, we use dynamic patch selection via quantile-based thresholding for isolating prognostically informative tissue regions. Secondly, we use graph-guided k-means clustering to capture phenotype-level heterogeneity through spatial and morphological coherence. Thirdly, we use attention mechanisms that model both intra- and inter-cluster relationships to contextualize local features within global spatial relations between various types of tissue compartments. Finally, we use an expert-guided mixture density modeling for estimating complex survival distributions using Gaussian mixture models. The proposed model achieves a concordance index of $0.712 \pm 0.028$ and Brier score of $0.254 \pm 0.018$ on TCGA-KIRC (renal cancer), and a concordance index of $0.645 \pm 0.017$ and Brier score of $0.281 \pm 0.031$ on TCGA-LUAD (lung adenocarcinoma). These results are significantly better than the state-of-art and demonstrate predictive potential of the proposed method across diverse cancer types.
>
---
#### [replaced 043] Extending Foundational Monocular Depth Estimators to Fisheye Cameras with Calibration Tokens
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.04928v2](http://arxiv.org/pdf/2508.04928v2)**

> **作者:** Suchisrit Gangopadhyay; Jung-Hee Kim; Xien Chen; Patrick Rim; Hyoungseob Park; Alex Wong
>
> **摘要:** We propose a method to extend foundational monocular depth estimators (FMDEs), trained on perspective images, to fisheye images. Despite being trained on tens of millions of images, FMDEs are susceptible to the covariate shift introduced by changes in camera calibration (intrinsic, distortion) parameters, leading to erroneous depth estimates. Our method aligns the distribution of latent embeddings encoding fisheye images to those of perspective images, enabling the reuse of FMDEs for fisheye cameras without retraining or finetuning. To this end, we introduce a set of Calibration Tokens as a light-weight adaptation mechanism that modulates the latent embeddings for alignment. By exploiting the already expressive latent space of FMDEs, we posit that modulating their embeddings avoids the negative impact of artifacts and loss introduced in conventional recalibration or map projection to a canonical reference frame in the image space. Our method is self-supervised and does not require fisheye images but leverages publicly available large-scale perspective image datasets. This is done by recalibrating perspective images to fisheye images, and enforcing consistency between their estimates during training. We evaluate our approach with several FMDEs, on both indoors and outdoors, where we consistently improve over state-of-the-art methods using a single set of tokens for both. Code available at: https://github.com/JungHeeKim29/calibration-token.
>
---
#### [replaced 044] CPT-Interp: Continuous sPatial and Temporal Motion Modeling for 4D Medical Image Interpolation
- **分类: cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2405.15385v2](http://arxiv.org/pdf/2405.15385v2)**

> **作者:** Xia Li; Runzhao Yang; Xiangtai Li; Antony Lomax; Ye Zhang; Joachim Buhmann
>
> **备注:** This paper has been merged into the new version of arXiv:2405.00430
>
> **摘要:** Motion information from 4D medical imaging offers critical insights into dynamic changes in patient anatomy for clinical assessments and radiotherapy planning and, thereby, enhances the capabilities of 3D image analysis. However, inherent physical and technical constraints of imaging hardware often necessitate a compromise between temporal resolution and image quality. Frame interpolation emerges as a pivotal solution to this challenge. Previous methods often suffer from discretion when they estimate the intermediate motion and execute the forward warping. In this study, we draw inspiration from fluid mechanics to propose a novel approach for continuously modeling patient anatomic motion using implicit neural representation. It ensures both spatial and temporal continuity, effectively bridging Eulerian and Lagrangian specifications together to naturally facilitate continuous frame interpolation. Our experiments across multiple datasets underscore the method's superior accuracy and speed. Furthermore, as a case-specific optimization (training-free) approach, it circumvents the need for extensive datasets and addresses model generalization issues.
>
---
#### [replaced 045] WildSAT: Learning Satellite Image Representations from Wildlife Observations
- **分类: cs.CV; cs.LG; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2412.14428v2](http://arxiv.org/pdf/2412.14428v2)**

> **作者:** Rangel Daroya; Elijah Cole; Oisin Mac Aodha; Grant Van Horn; Subhransu Maji
>
> **摘要:** Species distributions encode valuable ecological and environmental information, yet their potential for guiding representation learning in remote sensing remains underexplored. We introduce WildSAT, which pairs satellite images with millions of geo-tagged wildlife observations readily-available on citizen science platforms. WildSAT employs a contrastive learning approach that jointly leverages satellite images, species occurrence maps, and textual habitat descriptions to train or fine-tune models. This approach significantly improves performance on diverse satellite image recognition tasks, outperforming both ImageNet-pretrained models and satellite-specific baselines. Additionally, by aligning visual and textual information, WildSAT enables zero-shot retrieval, allowing users to search geographic locations based on textual descriptions. WildSAT surpasses recent cross-modal learning methods, including approaches that align satellite images with ground imagery or wildlife photos, demonstrating the advantages of our approach. Finally, we analyze the impact of key design choices and highlight the broad applicability of WildSAT to remote sensing and biodiversity monitoring.
>
---
#### [replaced 046] ART: Adaptive Relation Tuning for Generalized Relation Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.23543v2](http://arxiv.org/pdf/2507.23543v2)**

> **作者:** Gopika Sudhakaran; Hikaru Shindo; Patrick Schramowski; Simone Schaub-Meyer; Kristian Kersting; Stefan Roth
>
> **备注:** Accepted for publication in ICCV 2025
>
> **摘要:** Visual relation detection (VRD) is the task of identifying the relationships between objects in a scene. VRD models trained solely on relation detection data struggle to generalize beyond the relations on which they are trained. While prompt tuning has been used to adapt vision-language models (VLMs) for VRD, it uses handcrafted prompts and struggles with novel or complex relations. We argue that instruction tuning offers a more effective solution by fine-tuning VLMs on diverse instructional data. We thus introduce ART, an Adaptive Relation Tuning framework that adapts VLMs for VRD through instruction tuning and strategic instance selection. By converting VRD datasets into an instruction tuning format and employing an adaptive sampling algorithm, ART directs the VLM to focus on informative relations while maintaining generalizability. Specifically, we focus on the relation classification, where subject-object boxes are given and the model predicts the predicate between them. We tune on a held-in set and evaluate across multiple held-out datasets of varying complexity. Our approach strongly improves over its baselines and can infer unseen relation concepts, a capability absent in mainstream VRD methods. We demonstrate ART's practical value by using the predicted relations for segmenting complex scenes.
>
---
#### [replaced 047] CMIC: Content-Adaptive Mamba for Learned Image Compression
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.02192v3](http://arxiv.org/pdf/2508.02192v3)**

> **作者:** Yunuo Chen; Zezheng Lyu; Bing He; Hongwei Hu; Qi Wang; Yuan Tian; Li Song; Wenjun Zhang; Guo Lu
>
> **摘要:** Recent Learned image compression (LIC) leverages Mamba-style state-space models (SSMs) for global receptive fields with linear complexity. However, vanilla Mamba is content-agnostic, relying on fixed and predefined selective scans, which restricts its ability to dynamically and fully exploit content dependencies. We introduce Content-Adaptive Mamba (CAM), a dynamic SSM that addresses two critical limitations. First, it employs content-aware token reorganization, clustering and reordering tokens based on content similarity to prioritize proximity in feature space over Euclidean space. Second, it integrates global priors into SSM via a prompt dictionary, effectively mitigating the strict causality and long-range decay in the token interactions of Mamba. These innovations enable CAM to better capture global dependencies while preserving computational efficiency. Leveraging CAM, our Content-Adaptive Mamba-based LIC model (CMIC) achieves state-of-the-art rate-distortion performance, surpassing VTM-21.0 by -15.91\%, -21.34\%, and -17.58\% BD-rate on Kodak, Tecnick, and CLIC benchmarks, respectively.
>
---
#### [replaced 048] Rethinking the Bias of Foundation Model under Long-tailed Distribution
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2501.15955v3](http://arxiv.org/pdf/2501.15955v3)**

> **作者:** Jiahao Chen; Bin Qin; Jiangmeng Li; Hao Chen; Bing Su
>
> **备注:** Published as a conference paper in ICML 2025
>
> **摘要:** Long-tailed learning has garnered increasing attention due to its practical significance. Among the various approaches, the fine-tuning paradigm has gained considerable interest with the advent of foundation models. However, most existing methods primarily focus on leveraging knowledge from these models, overlooking the inherent biases introduced by the imbalanced training data they rely on. In this paper, we examine how such imbalances from pre-training affect long-tailed downstream tasks. Specifically, we find the imbalance biases inherited in foundation models on downstream task as parameter imbalance and data imbalance. During fine-tuning, we observe that parameter imbalance plays a more critical role, while data imbalance can be mitigated using existing re-balancing strategies. Moreover, we find that parameter imbalance cannot be effectively addressed by current re-balancing techniques, such as adjusting the logits, during training, unlike data imbalance. To tackle both imbalances simultaneously, we build our method on causal learning and view the incomplete semantic factor as the confounder, which brings spurious correlations between input samples and labels. To resolve the negative effects of this, we propose a novel backdoor adjustment method that learns the true causal effect between input samples and labels, rather than merely fitting the correlations in the data. Notably, we achieve an average performance increase of about $1.67\%$ on each dataset. Code is available: https://github.com/JiahaoChen1/Pre-train-Imbalance
>
---
#### [replaced 049] POMATO: Marrying Pointmap Matching with Temporal Motion for Dynamic 3D Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.05692v2](http://arxiv.org/pdf/2504.05692v2)**

> **作者:** Songyan Zhang; Yongtao Ge; Jinyuan Tian; Guangkai Xu; Hao Chen; Chen Lv; Chunhua Shen
>
> **备注:** code: https://github.com/wyddmw/POMATO
>
> **摘要:** 3D reconstruction in dynamic scenes primarily relies on the combination of geometry estimation and matching modules where the latter task is pivotal for distinguishing dynamic regions which can help to mitigate the interference introduced by camera and object motion. Furthermore, the matching module explicitly models object motion, enabling the tracking of specific targets and advancing motion understanding in complex scenarios. Recently, the proposed representation of pointmap in DUSt3R suggests a potential solution to unify both geometry estimation and matching in 3D space, but it still struggles with ambiguous matching in dynamic regions, which may hamper further improvement. In this work, we present POMATO, a unified framework for dynamic 3D reconstruction by marrying pointmap matching with temporal motion. Specifically, our method first learns an explicit matching relationship by mapping RGB pixels from both dynamic and static regions across different views to 3D pointmaps within a unified coordinate system. Furthermore, we introduce a temporal motion module for dynamic motions that ensures scale consistency across different frames and enhances performance in tasks requiring both precise geometry and reliable matching, most notably 3D point tracking. We show the effectiveness of the proposed pointmap matching and temporal fusion paradigm by demonstrating the remarkable performance across multiple downstream tasks, including video depth estimation, 3D point tracking, and pose estimation. Code and models are publicly available at https://github.com/wyddmw/POMATO.
>
---
#### [replaced 050] Two-stage deep learning framework for the restoration of incomplete-ring PET images
- **分类: cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2504.00816v4](http://arxiv.org/pdf/2504.00816v4)**

> **作者:** Yeqi Fang; Rong Zhou
>
> **备注:** 20 pages, 7 figures
>
> **摘要:** Positron Emission Tomography (PET) is an important molecular imaging tool widely used in medicine. Traditional PET systems rely on complete detector rings for full angular coverage and reliable data collection. However, incomplete-ring PET scanners have emerged due to hardware failures, cost constraints, or specific clinical needs. Standard reconstruction algorithms often suffer from performance degradation with these systems because of reduced data completeness and geometric inconsistencies. We present a two-stage deep-learning framework that, without incorporating any time-of-flight (TOF) information, restores high-quality images from data with about 50% missing coincidences - double the loss levels previously addressed by CNN-based methods. The pipeline operates in two stages: a projection-domain Attention U-Net first predicts the missing sections of the sinogram by leveraging spatial context from neighbouring slices, after which the completed data are reconstructed with OSEM algorithm and passed to a U-Net-diffusion module that removes residual artefacts while reinstating high-frequency detail. Using 206 brain volumes from a public dataset, the result shows that our model successfully preserves most anatomical structures and tracer distribution features with PSNR of 30.92 dB and SSIM of 0.9708. We also achieve higher inference speed, thus providing an effective solution for incomplete-ring PET imaging.
>
---
#### [replaced 051] Crop Pest Classification Using Deep Learning Techniques: A Review
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.01494v3](http://arxiv.org/pdf/2507.01494v3)**

> **作者:** Muhammad Hassam Ejaz; Muhammad Bilal; Usman Habib; Muhammad Attique; Tae-Sun Chung
>
> **备注:** This version adds co-authors who were unintentionally left out of the prior submission. Additionally, Table 1 has been reformatted for clarity, and several typographical errors have been corrected
>
> **摘要:** Insect pests continue to bring a serious threat to crop yields around the world, and traditional methods for monitoring them are often slow, manual, and difficult to scale. In recent years, deep learning has emerged as a powerful solution, with techniques like convolutional neural networks (CNNs), vision transformers (ViTs), and hybrid models gaining popularity for automating pest detection. This review looks at 37 carefully selected studies published between 2018 and 2025, all focused on AI-based pest classification. The selected research is organized by crop type, pest species, model architecture, dataset usage, and key technical challenges. The early studies relied heavily on CNNs but latest work is shifting toward hybrid and transformer-based models that deliver higher accuracy and better contextual understanding. Still, challenges like imbalanced datasets, difficulty in detecting small pests, limited generalizability, and deployment on edge devices remain significant hurdles. Overall, this review offers a structured overview of the field, highlights useful datasets, and outlines the key challenges and future directions for AI-based pest monitoring systems.
>
---
#### [replaced 052] Can Test-Time Scaling Improve World Foundation Model?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.24320v2](http://arxiv.org/pdf/2503.24320v2)**

> **作者:** Wenyan Cong; Hanqing Zhu; Peihao Wang; Bangya Liu; Dejia Xu; Kevin Wang; David Z. Pan; Yan Wang; Zhiwen Fan; Zhangyang Wang
>
> **备注:** Accepted by COLM2025
>
> **摘要:** World foundation models, which simulate the physical world by predicting future states from current observations and inputs, have become central to many applications in physical intelligence, including autonomous driving and robotics. However, these models require substantial computational resources for pretraining and are further constrained by available data during post-training. As such, scaling computation at test time emerges as both a critical and practical alternative to traditional model enlargement or re-training. In this work, we introduce SWIFT, a test-time scaling framework tailored for WFMs. SWIFT integrates our extensible WFM evaluation toolkit with process-level inference strategies, including fast tokenization, probability-based Top-K pruning, and efficient beam search. Empirical results on the COSMOS model demonstrate that test-time scaling exists even in a compute-optimal way. Our findings reveal that test-time scaling laws hold for WFMs and that SWIFT provides a scalable and effective pathway for improving WFM inference without retraining or increasing model size. Project page: https://scalingwfm.github.io/.
>
---
#### [replaced 053] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.09082v2](http://arxiv.org/pdf/2506.09082v2)**

> **作者:** Zheda Mai; Arpita Chowdhury; Zihe Wang; Sooyoung Jeon; Lemeng Wang; Jiacheng Hou; Wei-Lun Chao
>
> **备注:** First two authors contribute equally
>
> **摘要:** The rise of vision foundation models (VFMs) calls for systematic evaluation. A common approach pairs VFMs with large language models (LLMs) as general-purpose heads, followed by evaluation on broad Visual Question Answering (VQA) benchmarks. However, this protocol has two key blind spots: (i) the instruction tuning data may not align with VQA test distributions, meaning a wrong prediction can stem from such data mismatch rather than a VFM' visual shortcomings; (ii) VQA benchmarks often require multiple visual abilities, making it hard to tell whether errors stem from lacking all required abilities or just a single critical one. To address these gaps, we introduce AVA-Bench, the first benchmark that explicitly disentangles 14 Atomic Visual Abilities (AVAs) -- foundational skills like localization, depth estimation, and spatial understanding that collectively support complex visual reasoning tasks. By decoupling AVAs and matching training and test distributions within each, AVA-Bench pinpoints exactly where a VFM excels or falters. Applying AVA-Bench to leading VFMs thus reveals distinctive "ability fingerprints," turning VFM selection from educated guesswork into principled engineering. Notably, we find that a 0.5B LLM yields similar VFM rankings as a 7B LLM while cutting GPU hours by 8x, enabling more efficient evaluation. By offering a comprehensive and transparent benchmark, we hope AVA-Bench lays the foundation for the next generation of VFMs.
>
---
#### [replaced 054] Embodied Intelligence for 3D Understanding: A Survey on 3D Scene Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00342v2](http://arxiv.org/pdf/2502.00342v2)**

> **作者:** Zechuan Li; Hongshan Yu; Yihao Ding; Yan Li; Yong He; Naveed Akhtar
>
> **备注:** This is a submitted version of a paper accepted by Information Fusion
>
> **摘要:** 3D Scene Question Answering (3D SQA) represents an interdisciplinary task that integrates 3D visual perception and natural language processing, empowering intelligent agents to comprehend and interact with complex 3D environments. Recent advances in large multimodal modelling have driven the creation of diverse datasets and spurred the development of instruction-tuning and zero-shot methods for 3D SQA. However, this rapid progress introduces challenges, particularly in achieving unified analysis and comparison across datasets and baselines. In this survey, we provide the first comprehensive and systematic review of 3D SQA. We organize existing work from three perspectives: datasets, methodologies, and evaluation metrics. Beyond basic categorization, we identify shared architectural patterns across methods. Our survey further synthesizes core limitations and discusses how current trends, such as instruction tuning, multimodal alignment, and zero-shot, can shape future developments. Finally, we propose a range of promising research directions covering dataset construction, task generalization, interaction modeling, and unified evaluation protocols. This work aims to serve as a foundation for future research and foster progress toward more generalizable and intelligent 3D SQA systems.
>
---
#### [replaced 055] MetaOcc: Spatio-Temporal Fusion of Surround-View 4D Radar and Camera for 3D Occupancy Prediction with Dual Training Strategies
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.15384v3](http://arxiv.org/pdf/2501.15384v3)**

> **作者:** Long Yang; Lianqing Zheng; Wenjin Ai; Minghao Liu; Sen Li; Qunshu Lin; Shengyu Yan; Jie Bai; Zhixiong Ma; Tao Huang; Xichan Zhu
>
> **摘要:** Robust 3D occupancy prediction is essential for autonomous driving, particularly under adverse weather conditions where traditional vision-only systems struggle. While the fusion of surround-view 4D radar and cameras offers a promising low-cost solution, effectively extracting and integrating features from these heterogeneous sensors remains challenging. This paper introduces MetaOcc, a novel multi-modal framework for omnidirectional 3D occupancy prediction that leverages both multi-view 4D radar and images. To address the limitations of directly applying LiDAR-oriented encoders to sparse radar data, we propose a Radar Height Self-Attention module that enhances vertical spatial reasoning and feature extraction. Additionally, a Hierarchical Multi-scale Multi-modal Fusion strategy is developed to perform adaptive local-global fusion across modalities and time, mitigating spatio-temporal misalignments and enriching fused feature representations. To reduce reliance on expensive point cloud annotations, we further propose a pseudo-label generation pipeline based on an open-set segmentor. This enables a semi-supervised strategy that achieves 90% of the fully supervised performance using only 50% of the ground truth labels, offering an effective trade-off between annotation cost and accuracy. Extensive experiments demonstrate that MetaOcc under full supervision achieves state-of-the-art performance, outperforming previous methods by +0.47 SC IoU and +4.02 mIoU on the OmniHD-Scenes dataset, and by +1.16 SC IoU and +1.24 mIoU on the SurroundOcc-nuScenes dataset. These results demonstrate the scalability and robustness of MetaOcc across sensor domains and training conditions, paving the way for practical deployment in real-world autonomous systems. Code and data are available at https://github.com/LucasYang567/MetaOcc.
>
---
#### [replaced 056] DanceGRPO: Unleashing GRPO on Visual Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07818v3](http://arxiv.org/pdf/2505.07818v3)**

> **作者:** Zeyue Xue; Jie Wu; Yu Gao; Fangyuan Kong; Lingting Zhu; Mengzhao Chen; Zhiheng Liu; Wei Liu; Qiushan Guo; Weilin Huang; Ping Luo
>
> **备注:** Project Page: https://dancegrpo.github.io/
>
> **摘要:** Recent advances in generative AI have revolutionized visual content creation, yet aligning model outputs with human preferences remains a critical challenge. While Reinforcement Learning (RL) has emerged as a promising approach for fine-tuning generative models, existing methods like DDPO and DPOK face fundamental limitations - particularly their inability to maintain stable optimization when scaling to large and diverse prompt sets, severely restricting their practical utility. This paper presents DanceGRPO, a framework that addresses these limitations through an innovative adaptation of Group Relative Policy Optimization (GRPO) for visual generation tasks. Our key insight is that GRPO's inherent stability mechanisms uniquely position it to overcome the optimization challenges that plague prior RL-based approaches on visual generation. DanceGRPO establishes several significant advances: First, it demonstrates consistent and stable policy optimization across multiple modern generative paradigms, including both diffusion models and rectified flows. Second, it maintains robust performance when scaling to complex, real-world scenarios encompassing three key tasks and four foundation models. Third, it shows remarkable versatility in optimizing for diverse human preferences as captured by five distinct reward models assessing image/video aesthetics, text-image alignment, video motion quality, and binary feedback. Our comprehensive experiments reveal that DanceGRPO outperforms baseline methods by up to 181\% across multiple established benchmarks, including HPS-v2.1, CLIP Score, VideoAlign, and GenEval. Our results establish DanceGRPO as a robust and versatile solution for scaling Reinforcement Learning from Human Feedback (RLHF) tasks in visual generation, offering new insights into harmonizing reinforcement learning and visual synthesis.
>
---
#### [replaced 057] CANVAS: Commonsense-Aware Navigation System for Intuitive Human-Robot Interaction
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.01273v3](http://arxiv.org/pdf/2410.01273v3)**

> **作者:** Suhwan Choi; Yongjun Cho; Minchan Kim; Jaeyoon Jung; Myunchul Joe; Yubeen Park; Minseo Kim; Sungwoong Kim; Sungjae Lee; Hwiseong Park; Jiwan Chung; Youngjae Yu
>
> **备注:** Accepted to ICRA 2025, project page https://worv-ai.github.io/canvas
>
> **摘要:** Real-life robot navigation involves more than just reaching a destination; it requires optimizing movements while addressing scenario-specific goals. An intuitive way for humans to express these goals is through abstract cues like verbal commands or rough sketches. Such human guidance may lack details or be noisy. Nonetheless, we expect robots to navigate as intended. For robots to interpret and execute these abstract instructions in line with human expectations, they must share a common understanding of basic navigation concepts with humans. To this end, we introduce CANVAS, a novel framework that combines visual and linguistic instructions for commonsense-aware navigation. Its success is driven by imitation learning, enabling the robot to learn from human navigation behavior. We present COMMAND, a comprehensive dataset with human-annotated navigation results, spanning over 48 hours and 219 km, designed to train commonsense-aware navigation systems in simulated environments. Our experiments show that CANVAS outperforms the strong rule-based system ROS NavStack across all environments, demonstrating superior performance with noisy instructions. Notably, in the orchard environment, where ROS NavStack records a 0% total success rate, CANVAS achieves a total success rate of 67%. CANVAS also closely aligns with human demonstrations and commonsense constraints, even in unseen environments. Furthermore, real-world deployment of CANVAS showcases impressive Sim2Real transfer with a total success rate of 69%, highlighting the potential of learning from human demonstrations in simulated environments for real-world applications.
>
---
#### [replaced 058] TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.18350v2](http://arxiv.org/pdf/2411.18350v2)**

> **作者:** Riza Velioglu; Petra Bevandic; Robin Chan; Barbara Hammer
>
> **备注:** Accepted at BMVC'25
>
> **摘要:** This paper introduces Virtual Try-Off (VTOFF), a novel task generating standardized garment images from single photos of clothed individuals. Unlike Virtual Try-On (VTON), which digitally dresses models, VTOFF extracts canonical garment images, demanding precise reconstruction of shape, texture, and complex patterns, enabling robust evaluation of generative model fidelity. We propose TryOffDiff, adapting Stable Diffusion with SigLIP-based visual conditioning to deliver high-fidelity reconstructions. Experiments on VITON-HD and Dress Code datasets show that TryOffDiff outperforms adapted pose transfer and VTON baselines. We observe that traditional metrics such as SSIM inadequately reflect reconstruction quality, prompting our use of DISTS for reliable assessment. Our findings highlight VTOFF's potential to improve e-commerce product imagery, advance generative model evaluation, and guide future research on high-fidelity reconstruction. Demo, code, and models are available at: https://rizavelioglu.github.io/tryoffdiff
>
---
#### [replaced 059] Neural-Driven Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05397v2](http://arxiv.org/pdf/2507.05397v2)**

> **作者:** Pengfei Zhou; Jie Xia; Xiaopeng Peng; Wangbo Zhao; Zilong Ye; Zekai Li; Suorong Yang; Jiadong Pan; Yuanxiang Chen; Ziqiao Wang; Kai Wang; Qian Zheng; Xiaojun Chang; Gang Pan; Shurong Dong; Kaipeng Zhang; Yang You
>
> **备注:** 22 pages, 14 figures
>
> **摘要:** Traditional image editing typically relies on manual prompting, making it labor-intensive and inaccessible to individuals with limited motor control or language abilities. Leveraging recent advances in brain-computer interfaces (BCIs) and generative models, we propose LoongX, a hands-free image editing approach driven by multimodal neurophysiological signals. LoongX utilizes state-of-the-art diffusion models trained on a comprehensive dataset of 23,928 image editing pairs, each paired with synchronized electroencephalography (EEG), functional near-infrared spectroscopy (fNIRS), photoplethysmography (PPG), and head motion signals that capture user intent. To effectively address the heterogeneity of these signals, LoongX integrates two key modules. The cross-scale state space (CS3) module encodes informative modality-specific features. The dynamic gated fusion (DGF) module further aggregates these features into a unified latent space, which is then aligned with edit semantics via fine-tuning on a diffusion transformer (DiT). Additionally, we pre-train the encoders using contrastive learning to align cognitive states with semantic intentions from embedded natural language. Extensive experiments demonstrate that LoongX achieves performance comparable to text-driven methods (CLIP-I: 0.6605 vs. 0.6558; DINO: 0.4812 vs. 0.4636) and outperforms them when neural signals are combined with speech (CLIP-T: 0.2588 vs. 0.2549). These results highlight the promise of neural-driven generative models in enabling accessible, intuitive image editing and open new directions for cognitive-driven creative technologies. Datasets and code will be released to support future work and foster progress in this emerging area.
>
---
#### [replaced 060] A dataset of primary nasopharyngeal carcinoma MRI with multi-modalities segmentation
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.03253v3](http://arxiv.org/pdf/2404.03253v3)**

> **作者:** Yin Li; Qi Chen; Kai Wang; Meige Li; Liping Si; Yingwei Guo; Yu Xiong; Qixing Wang; Yang Qin; Ling Xu; Patrick van der Smagt; Jun Tang; Nutan Chen
>
> **备注:** This preprint has been submitted to and accepted in principle for publication in Scientific Data without significant changes
>
> **摘要:** Multi-modality magnetic resonance imaging(MRI) data facilitate the early diagnosis, tumor segmentation, and disease staging in the management of nasopharyngeal carcinoma (NPC). The lack of publicly available, comprehensive datasets limits advancements in diagnosis, treatment planning, and the development of machine learning algorithms for NPC. Addressing this critical need, we introduce the first comprehensive NPC MRI dataset, encompassing MR axial imaging of 277 primary NPC patients. This dataset includes T1-weighted, T2-weighted, and contrast-enhanced T1-weighted sequences, totaling 831 scans. In addition to the corresponding clinical data, manually annotated and labeled segmentations by experienced radiologists offer high-quality data resources from untreated primary NPC.
>
---
#### [replaced 061] RetinexDual: Retinex-based Dual Nature Approach for Generalized Ultra-High-Definition Image Restoration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04797v2](http://arxiv.org/pdf/2508.04797v2)**

> **作者:** Mohab Kishawy; Ali Abdellatif Hussein; Jun Chen
>
> **摘要:** Advancements in image sensing have elevated the importance of Ultra-High-Definition Image Restoration (UHD IR). Traditional methods, such as extreme downsampling or transformation from the spatial to the frequency domain, encounter significant drawbacks: downsampling induces irreversible information loss in UHD images, while our frequency analysis reveals that pure frequency-domain approaches are ineffective for spatially confined image artifacts, primarily due to the loss of degradation locality. To overcome these limitations, we present RetinexDual, a novel Retinex theory-based framework designed for generalized UHD IR tasks. RetinexDual leverages two complementary sub-networks: the Scale-Attentive maMBA (SAMBA) and the Frequency Illumination Adaptor (FIA). SAMBA, responsible for correcting the reflectance component, utilizes a coarse-to-fine mechanism to overcome the causal modeling of mamba, which effectively reduces artifacts and restores intricate details. On the other hand, FIA ensures precise correction of color and illumination distortions by operating in the frequency domain and leveraging the global context provided by it. Evaluating RetinexDual on four UHD IR tasks, namely deraining, deblurring, dehazing, and Low-Light Image Enhancement (LLIE), shows that it outperforms recent methods qualitatively and quantitatively. Ablation studies demonstrate the importance of employing distinct designs for each branch in RetinexDual, as well as the effectiveness of its various components.
>
---
#### [replaced 062] X-VFL: A New Vertical Federated Learning Framework with Cross Completion and Decision Subspace Alignment
- **分类: cs.LG; cs.CV; cs.DC; math.OC**

- **链接: [http://arxiv.org/pdf/2508.05568v2](http://arxiv.org/pdf/2508.05568v2)**

> **作者:** Qinghua Yao; Xiangrui Xu; Zhize Li
>
> **备注:** 20 pages
>
> **摘要:** Vertical Federated Learning (VFL) enables collaborative learning by integrating disjoint feature subsets from multiple clients/parties. However, VFL typically faces two key challenges: i) the requirement for perfectly aligned data samples across all clients (missing features are not allowed); ii) the requirement for joint collaborative inference/prediction involving all clients (it does not support locally independent inference on a single client). To address these challenges, we propose X-VFL, a new VFL framework designed to deal with the non-aligned data samples with (partially) missing features and to support locally independent inference of new data samples for each client. In particular, we design two novel modules in X-VFL: Cross Completion (XCom) and Decision Subspace Alignment (DS-Align). XCom can complete/reconstruct missing features for non-aligned data samples by leveraging information from other clients. DS-Align aligns local features with completed and global features across all clients within the decision subspace, thus enabling locally independent inference at each client. Moreover, we provide convergence theorems for different algorithms used in training X-VFL, showing an $O(1/\sqrt{T})$ convergence rate for SGD-type algorithms and an $O(1/T)$ rate for PAGE-type algorithms, where $T$ denotes the number of training update steps. Extensive experiments on real-world datasets demonstrate that X-VFL significantly outperforms existing methods, e.g., achieving a 15% improvement in accuracy on the image CIFAR-10 dataset and a 43% improvement on the medical MIMIC-III dataset. These results validate the practical effectiveness and superiority of X-VFL, particularly in scenarios involving partially missing features and locally independent inference.
>
---
#### [replaced 063] CF3: Compact and Fast 3D Feature Fields
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.05254v2](http://arxiv.org/pdf/2508.05254v2)**

> **作者:** Hyunjoon Lee; Joonkyu Min; Jaesik Park
>
> **备注:** ICCV 2025
>
> **摘要:** 3D Gaussian Splatting (3DGS) has begun incorporating rich information from 2D foundation models. However, most approaches rely on a bottom-up optimization process that treats raw 2D features as ground truth, incurring increased computational costs. We propose a top-down pipeline for constructing compact and fast 3D Gaussian feature fields, namely, CF3. We first perform a fast weighted fusion of multi-view 2D features with pre-trained Gaussians. This approach enables training a per-Gaussian autoencoder directly on the lifted features, instead of training autoencoders in the 2D domain. As a result, the autoencoder better aligns with the feature distribution. More importantly, we introduce an adaptive sparsification method that optimizes the Gaussian attributes of the feature field while pruning and merging the redundant Gaussians, constructing an efficient representation with preserved geometric details. Our approach achieves a competitive 3D feature field using as little as 5% of the Gaussians compared to Feature-3DGS.
>
---
