# 计算机视觉 cs.CV

- **最新发布 118 篇**

- **更新 77 篇**

## 最新发布

#### [new 001] DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration
- **分类: cs.CV**

- **简介: 该论文针对离线签名验证任务，解决传统方法依赖整体特征、忽略细粒度差异的问题。提出DetailSemNet模型，通过局部结构匹配与细节语义融合机制，增强细节表达与判别性，显著提升准确率与可解释性，并在跨数据集测试中表现优异。**

- **链接: [https://arxiv.org/pdf/2511.16364v1](https://arxiv.org/pdf/2511.16364v1)**

> **作者:** Meng-Cheng Shih; Tsai-Ling Huang; Yu-Heng Shih; Hong-Han Shuai; Hsuan-Tung Liu; Yi-Ren Yeh; Ching-Chun Huang
>
> **摘要:** Offline signature verification (OSV) is a frequently utilized technology in forensics. This paper proposes a new model, DetailSemNet, for OSV. Unlike previous methods that rely on holistic features for pair comparisons, our approach underscores the significance of fine-grained differences for robust OSV. We propose to match local structures between two signature images, significantly boosting verification accuracy. Furthermore, we observe that without specific architectural modifications, transformer-based backbones might naturally obscure local details, adversely impacting OSV performance. To address this, we introduce a Detail Semantics Integrator, leveraging feature disentanglement and re-entanglement. This integrator is specifically designed to enhance intricate details while simultaneously expanding discriminative semantics, thereby augmenting the efficacy of local structural matching. We evaluate our method against leading benchmarks in offline signature verification. Our model consistently outperforms recent methods, achieving state-of-the-art results with clear margins. The emphasis on local structure matching not only improves performance but also enhances the model's interpretability, supporting our findings. Additionally, our model demonstrates remarkable generalization capabilities in cross-dataset testing scenarios. The combination of generalizability and interpretability significantly bolsters the potential of DetailSemNet for real-world applications.
>
---
#### [new 002] RB-FT: Rationale-Bootstrapped Fine-Tuning for Video Classification
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型在小样本领域视频分类任务中的表现不佳问题，提出一种两阶段自提升方法（RB-FT）。通过引导模型生成视频的详细推理理由，利用自生成的推理作为中间监督信号进行微调，再进行常规监督微调，有效缩小语义差距，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.15923v1](https://arxiv.org/pdf/2511.15923v1)**

> **作者:** Meilong Xu; Di Fu; Jiaxing Zhang; Gong Yu; Jiayu Zheng; Xiaoling Hu; Dongdi Zhao; Feiyang Li; Chao Chen; Yong Cao
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** Vision Language Models (VLMs) are becoming increasingly integral to multimedia understanding; however, they often struggle with domain-specific video classification tasks, particularly in cases with limited data. This stems from a critical \textit{rationale gap}, where sparse domain data is insufficient to bridge the semantic distance between complex spatio-temporal content and abstract classification labels. We propose a two-stage self-improvement paradigm to bridge this gap without new annotations. First, we prompt the VLMs to generate detailed textual rationales for each video, compelling them to articulate the domain-specific logic. The VLM is then fine-tuned on these self-generated rationales, utilizing this intermediate supervision to align its representations with the nuances of the target domain. Second, conventional supervised fine-tuning (SFT) is performed on the task labels, achieving markedly higher effectiveness as a result of the model's pre-acquired domain reasoning. Extensive experiments on diverse datasets demonstrate that our method significantly outperforms direct SFT, validating self-generated rationale as an effective, annotation-efficient paradigm for adapting VLMs to domain-specific video analysis.
>
---
#### [new 003] Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective
- **分类: cs.CV**

- **简介: 该论文针对开放词汇语义分割任务，聚焦于提升CLIP模型在像素级多模态对齐中的性能。针对其因无关令牌干扰导致注意力分散的问题，提出无需训练的ReFocusing CLIP方法，通过重分配注意力聚焦目标区域，增强对齐精度，显著提升分割效果并保持高效推理。**

- **链接: [https://arxiv.org/pdf/2511.16170v1](https://arxiv.org/pdf/2511.16170v1)**

> **作者:** Jiahao Li; Yang Lu; Yachao Zhang; Yong Xie; Fangyong Wang; Yuan Xie; Yanyun Qu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) employs pixel-level vision-language alignment to associate category-related prompts with corresponding pixels. A key challenge is enhancing the multimodal dense prediction capability, specifically this pixel-level multimodal alignment. Although existing methods achieve promising results by leveraging CLIP's vision-language alignment, they rarely investigate the performance boundaries of CLIP for dense prediction from an interpretability mechanisms perspective. In this work, we systematically investigate CLIP's internal mechanisms and identify a critical phenomenon: analogous to human distraction, CLIP diverts significant attention resources from target regions to irrelevant tokens. Our analysis reveals that these tokens arise from dimension-specific over-activation; filtering them enhances CLIP's dense prediction performance. Consequently, we propose ReFocusing CLIP (RF-CLIP), a training-free approach that emulates human distraction-refocusing behavior to redirect attention from distraction tokens back to target regions, thereby refining CLIP's multimodal alignment granularity. Our method achieves SOTA performance on eight benchmarks while maintaining high inference efficiency.
>
---
#### [new 004] TetraSDF: Precise Mesh Extraction with Multi-resolution Tetrahedral Grid
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对神经SDF网格提取中精度不足的问题，提出TetraSDF框架。通过多分辨率四面体位置编码保持CPWA结构，实现精确的解析网格提取，克服了采样误差与仅适用于简单MLP的局限，显著提升重建精度与网格一致性，兼具高效性。**

- **链接: [https://arxiv.org/pdf/2511.16273v1](https://arxiv.org/pdf/2511.16273v1)**

> **作者:** Seonghun Oh; Youngjung Uh; Jin-Hwa Kim
>
> **摘要:** Extracting meshes that exactly match the zero-level set of neural signed distance functions (SDFs) remains challenging. Sampling-based methods introduce discretization error, while continuous piecewise affine (CPWA) analytic approaches apply only to plain ReLU MLPs. We present TetraSDF, a precise analytic meshing framework for SDFs represented by a ReLU MLP composed with a multi-resolution tetrahedral positional encoder. The encoder's barycentric interpolation preserves global CPWA structure, enabling us to track ReLU linear regions within an encoder-induced polyhedral complex. A fixed analytic input preconditioner derived from the encoder's metric further reduces directional bias and stabilizes training. Across multiple benchmarks, TetraSDF matches or surpasses existing grid-based encoders in SDF reconstruction accuracy, and its analytic extractor produces highly self-consistent meshes that remain faithful to the learned isosurfaces, all with practical runtime and memory efficiency.
>
---
#### [new 005] Improving Long-Tailed Object Detection with Balanced Group Softmax and Metric Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对长尾分布下的2D目标检测任务，解决因类别不平衡导致模型偏向常见类、稀有类检测性能差的问题。提出改进的平衡分组Softmax（BAGS）框架，并引入度量学习与k-NN分类，提升稀有类检测精度，实现24.5% mAP，刷新了LVISv1数据集上的最佳性能。**

- **链接: [https://arxiv.org/pdf/2511.16619v1](https://arxiv.org/pdf/2511.16619v1)**

> **作者:** Satyam Gaba
>
> **备注:** 8 pages, 7 figures, International Conference on Semantic Computing
>
> **摘要:** Object detection has been widely explored for class-balanced datasets such as COCO. However, real-world scenarios introduce the challenge of long-tailed distributions, where numerous categories contain only a few instances. This inherent class imbalance biases detection models towards the more frequent classes, degrading performance on rare categories. In this paper, we tackle the problem of long-tailed 2D object detection using the LVISv1 dataset, which consists of 1,203 categories and 164,000 images. We employ a two-stage Faster R-CNN architecture and propose enhancements to the Balanced Group Softmax (BAGS) framework to mitigate class imbalance. Our approach achieves a new state-of-the-art performance with a mean Average Precision (mAP) of 24.5%, surpassing the previous benchmark of 24.0%. Additionally, we hypothesize that tail class features may form smaller, denser clusters within the feature space of head classes, making classification challenging for regression-based classifiers. To address this issue, we explore metric learning to produce feature embeddings that are both well-separated across classes and tightly clustered within each class. For inference, we utilize a k-Nearest Neighbors (k-NN) approach to improve classification performance, particularly for rare classes. Our results demonstrate the effectiveness of these methods in advancing long-tailed object detection.
>
---
#### [new 006] LLMs-based Augmentation for Domain Adaptation in Long-tailed Food Datasets
- **分类: cs.CV**

- **简介: 该论文针对长尾分布、域偏移及细粒度食物识别难题，提出基于大语言模型（LLMs）的增强框架。通过LLMs生成食物标题与配料，将图文映射至共享嵌入空间以对齐跨域特征，提升模型在真实场景下的识别性能。**

- **链接: [https://arxiv.org/pdf/2511.16037v1](https://arxiv.org/pdf/2511.16037v1)**

> **作者:** Qing Wang; Chong-Wah Ngo; Ee-Peng Lim; Qianru Sun
>
> **摘要:** Training a model for food recognition is challenging because the training samples, which are typically crawled from the Internet, are visually different from the pictures captured by users in the free-living environment. In addition to this domain-shift problem, the real-world food datasets tend to be long-tailed distributed and some dishes of different categories exhibit subtle variations that are difficult to distinguish visually. In this paper, we present a framework empowered with large language models (LLMs) to address these challenges in food recognition. We first leverage LLMs to parse food images to generate food titles and ingredients. Then, we project the generated texts and food images from different domains to a shared embedding space to maximize the pair similarities. Finally, we take the aligned features of both modalities for recognition. With this simple framework, we show that our proposed approach can outperform the existing approaches tailored for long-tailed data distribution, domain adaptation, and fine-grained classification, respectively, on two food datasets.
>
---
#### [new 007] Box6D : Zero-shot Category-level 6D Pose Estimation of Warehouse Boxes
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Box6D，一种面向仓库存储箱的零样本类别级6D姿态估计方法。针对现有方法在复杂场景下精度或效率不足的问题，利用单张RGB-D图像快速估算箱体尺寸，结合类别级CAD模板进行姿态估计，并通过深度滤波与早停策略提升效率与准确性，显著降低推理时间并保持高精度。**

- **链接: [https://arxiv.org/pdf/2511.15884v1](https://arxiv.org/pdf/2511.15884v1)**

> **作者:** Yintao Ma; Sajjad Pakdamansavoji; Amir Rasouli; Tongtong Cao
>
> **摘要:** Accurate and efficient 6D pose estimation of novel objects under clutter and occlusion is critical for robotic manipulation across warehouse automation, bin picking, logistics, and e-commerce fulfillment. There are three main approaches in this domain; Model-based methods assume an exact CAD model at inference but require high-resolution meshes and transfer poorly to new environments; Model-free methods that rely on a few reference images or videos are more flexible, however often fail under challenging conditions; Category-level approaches aim to balance flexibility and accuracy but many are overly general and ignore environment and object priors, limiting their practicality in industrial settings. To this end, we propose Box6d, a category-level 6D pose estimation method tailored for storage boxes in the warehouse context. From a single RGB-D observation, Box6D infers the dimensions of the boxes via a fast binary search and estimates poses using a category CAD template rather than instance-specific models. Suing a depth-based plausibility filter and early-stopping strategy, Box6D then rejects implausible hypotheses, lowering computational cost. We conduct evaluations on real-world storage scenarios and public benchmarks, and show that our approach delivers competitive or superior 6D pose precision while reducing inference time by approximately 76%.
>
---
#### [new 008] PartUV: Part-Based UV Unwrapping of 3D Meshes
- **分类: cs.CV; cs.CG; cs.GR**

- **简介: 该论文提出PartUV，一种基于部件的3D网格UV展开方法，旨在解决AI生成网格因噪声和病态导致的展开碎片化问题。通过结合语义部件分解与几何启发式策略，生成更少、对齐部件的低失真图表，提升展开质量与效率，适用于复杂网格的高效参数化与多图块打包。**

- **链接: [https://arxiv.org/pdf/2511.16659v1](https://arxiv.org/pdf/2511.16659v1)**

> **作者:** Zhaoning Wang; Xinyue Wei; Ruoxi Shi; Xiaoshuai Zhang; Hao Su; Minghua Liu
>
> **备注:** project page: https://www.zhaoningwang.com/PartUV
>
> **摘要:** UV unwrapping flattens 3D surfaces to 2D with minimal distortion, often requiring the complex surface to be decomposed into multiple charts. Although extensively studied, existing UV unwrapping methods frequently struggle with AI-generated meshes, which are typically noisy, bumpy, and poorly conditioned. These methods often produce highly fragmented charts and suboptimal boundaries, introducing artifacts and hindering downstream tasks. We introduce PartUV, a part-based UV unwrapping pipeline that generates significantly fewer, part-aligned charts while maintaining low distortion. Built on top of a recent learning-based part decomposition method PartField, PartUV combines high-level semantic part decomposition with novel geometric heuristics in a top-down recursive framework. It ensures each chart's distortion remains below a user-specified threshold while minimizing the total number of charts. The pipeline integrates and extends parameterization and packing algorithms, incorporates dedicated handling of non-manifold and degenerate meshes, and is extensively parallelized for efficiency. Evaluated across four diverse datasets, including man-made, CAD, AI-generated, and Common Shapes, PartUV outperforms existing tools and recent neural methods in chart count and seam length, achieves comparable distortion, exhibits high success rates on challenging meshes, and enables new applications like part-specific multi-tiles packing. Our project page is at https://www.zhaoningwang.com/PartUV.
>
---
#### [new 009] Arbitrary-Resolution and Arbitrary-Scale Face Super-Resolution with Implicit Representation Networks
- **分类: cs.CV**

- **简介: 该论文针对人脸超分辨率（FSR）任务，解决现有方法固定放大倍数、对输入尺寸敏感的问题。提出ARASFSR框架，利用隐式表示网络，结合局部坐标与尺度比，实现任意分辨率和任意放大倍数的超分；引入频率估计与全局坐标调制模块，增强纹理细节与结构适应性，显著提升性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.16341v1](https://arxiv.org/pdf/2511.16341v1)**

> **作者:** Yi Ting Tsai; Yu Wei Chen; Hong-Han Shuai; Ching-Chun Huang
>
> **摘要:** Face super-resolution (FSR) is a critical technique for enhancing low-resolution facial images and has significant implications for face-related tasks. However, existing FSR methods are limited by fixed up-sampling scales and sensitivity to input size variations. To address these limitations, this paper introduces an Arbitrary-Resolution and Arbitrary-Scale FSR method with implicit representation networks (ARASFSR), featuring three novel designs. First, ARASFSR employs 2D deep features, local relative coordinates, and up-sampling scale ratios to predict RGB values for each target pixel, allowing super-resolution at any up-sampling scale. Second, a local frequency estimation module captures high-frequency facial texture information to reduce the spectral bias effect. Lastly, a global coordinate modulation module guides FSR to leverage prior facial structure knowledge and achieve resolution adaptation effectively. Quantitative and qualitative evaluations demonstrate the robustness of ARASFSR over existing state-of-the-art methods while super-resolving facial images across various input sizes and up-sampling scales.
>
---
#### [new 010] Late-decoupled 3D Hierarchical Semantic Segmentation with Semantic Prototype Discrimination based Bi-branch Supervision
- **分类: cs.CV**

- **简介: 该论文针对3D场景的分层语义分割任务，解决多层级优化冲突与类别不平衡问题。提出晚解耦框架与基于语义原型的双分支监督机制，通过多解码器结构与相互监督增强特征判别性，有效提升分割性能，并可作为通用模块增强现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16650v1](https://arxiv.org/pdf/2511.16650v1)**

> **作者:** Shuyu Cao; Chongshou Li; Jie Xu; Tianrui Li; Na Zhao
>
> **摘要:** 3D hierarchical semantic segmentation (3DHS) is crucial for embodied intelligence applications that demand a multi-grained and multi-hierarchy understanding of 3D scenes. Despite the progress, previous 3DHS methods have overlooked following two challenges: I) multi-label learning with a parameter-sharing model can lead to multi-hierarchy conflicts in cross-hierarchy optimization, and II) the class imbalance issue is inevitable across multiple hierarchies of 3D scenes, which makes the model performance become dominated by major classes. To address these issues, we propose a novel framework with a primary 3DHS branch and an auxiliary discrimination branch. Specifically, to alleviate the multi-hierarchy conflicts, we propose a late-decoupled 3DHS framework which employs multiple decoders with the coarse-to-fine hierarchical guidance and consistency. The late-decoupled architecture can mitigate the underfitting and overfitting conflicts among multiple hierarchies and can also constrain the class imbalance problem in each individual hierarchy. Moreover, we introduce a 3DHS-oriented semantic prototype based bi-branch supervision mechanism, which additionally learns class-wise discriminative point cloud features and performs mutual supervision between the auxiliary and 3DHS branches, to enhance the class-imbalance segmentation. Extensive experiments on multiple datasets and backbones demonstrate that our approach achieves state-of-the-art 3DHS performance, and its core components can also be used as a plug-and-play enhancement to improve previous methods.
>
---
#### [new 011] NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出NutriScreener，一种用于儿童营养不良筛查的多姿态图注意力网络。针对现有方法效率低、泛化性差的问题，结合视觉嵌入与知识检索，实现从图像中准确预测体征指标。在多数据集上验证，显著提升检测召回率与测量精度，适用于低资源环境。**

- **链接: [https://arxiv.org/pdf/2511.16566v1](https://arxiv.org/pdf/2511.16566v1)**

> **作者:** Misaal Khan; Mayank Vatsa; Kuldeep Singh; Richa Singh
>
> **备注:** Accepted in AAAI 2026 Special Track on AI for Social Impact
>
> **摘要:** Child malnutrition remains a global crisis, yet existing screening methods are laborious and poorly scalable, hindering early intervention. In this work, we present NutriScreener, a retrieval-augmented, multi-pose graph attention network that combines CLIP-based visual embeddings, class-boosted knowledge retrieval, and context awareness to enable robust malnutrition detection and anthropometric prediction from children's images, simultaneously addressing generalizability and class imbalance. In a clinical study, doctors rated it 4.3/5 for accuracy and 4.6/5 for efficiency, confirming its deployment readiness in low-resource settings. Trained and tested on 2,141 children from AnthroVision and additionally evaluated on diverse cross-continent populations, including ARAN and an in-house collected CampusPose dataset, it achieves 0.79 recall, 0.82 AUC, and significantly lower anthropometric RMSEs, demonstrating reliable measurement in unconstrained pediatric settings. Cross-dataset results show up to 25% recall gain and up to 3.5 cm RMSE reduction using demographically matched knowledge bases. NutriScreener offers a scalable and accurate solution for early malnutrition detection in low-resource environments.
>
---
#### [new 012] Erase to Retain: Low Rank Adaptation Guided Selective Unlearning in Medical Segmentation Networks
- **分类: cs.CV**

- **简介: 该论文针对医疗图像分割中的选择性遗忘问题，提出Erase to Retain框架。通过低秩适配（LoRA）引导的师生蒸馏，实现对特定病变或类别知识的可控删除，同时保留整体解剖结构理解与泛化能力，无需全量重训练。**

- **链接: [https://arxiv.org/pdf/2511.16574v1](https://arxiv.org/pdf/2511.16574v1)**

> **作者:** Nirjhor Datta; Md. Golam Rabiul Alam
>
> **摘要:** The ability to selectively remove knowledge from medical segmentation networks is increasingly important for privacy compliance, ethical deployment, and continual dataset revision. We introduce Erase to Retain, a controllable unlearning framework for medical image segmentation that achieves targeted forgetting without full retraining. Our method uses a teacher-student distillation paradigm with Low-Rank Adaptation (LoRA) constrained subspace updates, enabling the student network to erase lesion-specific or class-specific representations in low-rank decoder spaces while preserving global anatomical understanding. During the strong unlearning phase, LoRA modules are adversarially optimized to contradict the teacher's confident predictions on a designated forget subset, enforcing semantic removal. This is followed by a gentle restoration phase that recovers generalization on retained data through head-only supervised refinement. For ISIC segmentation, the student reduces forget-set IoU from 0.875 to 0.509 while maintaining competitive performance on the retain and validation splits (0.647 to 0.677 IoU). On the cross-domain CHASE dataset, Erase to Retain consistently lowers forget-set IoU while preserving utility on retain and validation sets. For ISIC classification, our method decreases accuracy on the forget subset from 87.0 percent to 64.1 percent while improving retain accuracy from 83.9 percent to 90.6 percent. These results demonstrate that LoRA-based subspace unlearning provides a practical pathway toward responsible, controllable, and reversible unlearning in medical image analysis, enabling models to forget sensitive samples or structures while preserving performance where it matters most.
>
---
#### [new 013] Generative AI for Enhanced Wildfire Detection: Bridging the Synthetic-Real Domain Gap
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对火灾烟雾检测中真实标注数据稀缺的问题，利用生成式AI合成高质量、带标注的烟雾数据，并通过风格迁移、GAN和图像抠图等技术提升合成数据的真实性，结合无监督域适应方法缩小合成与真实数据的差异，从而提升烟雾检测模型的准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.16617v1](https://arxiv.org/pdf/2511.16617v1)**

> **作者:** Satyam Gaba
>
> **备注:** 8 pages, 16 figures
>
> **摘要:** The early detection of wildfires is a critical environmental challenge, with timely identification of smoke plumes being key to mitigating large-scale damage. While deep neural networks have proven highly effective for localization tasks, the scarcity of large, annotated datasets for smoke detection limits their potential. In response, we leverage generative AI techniques to address this data limitation by synthesizing a comprehensive, annotated smoke dataset. We then explore unsupervised domain adaptation methods for smoke plume segmentation, analyzing their effectiveness in closing the gap between synthetic and real-world data. To further refine performance, we integrate advanced generative approaches such as style transfer, Generative Adversarial Networks (GANs), and image matting. These methods aim to enhance the realism of synthetic data and bridge the domain disparity, paving the way for more accurate and scalable wildfire detection models.
>
---
#### [new 014] Layer-wise Noise Guided Selective Wavelet Reconstruction for Robust Medical Image Segmentation
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对医学图像分割在分布偏移和扰动下的鲁棒性问题，提出层噪声引导的可选小波重建方法（LNG-SWR）。通过多层注入噪声学习频域先验，引导特征避开敏感方向，并选择性重构频率成分以增强结构与边界稳定性。该方法无需额外训练成本，兼容主流模型，显著提升抗攻击能力且不损失干净性能。**

- **链接: [https://arxiv.org/pdf/2511.16162v1](https://arxiv.org/pdf/2511.16162v1)**

> **作者:** Yuting Lu; Ziliang Wang; Weixin Xu; Wei Zhang; Yongqiang Zhao; Yang Yu; Xiaohong Zhang
>
> **摘要:** Clinical deployment requires segmentation models to stay stable under distribution shifts and perturbations. The mainstream solution is adversarial training (AT) to improve robustness; however, AT often brings a clean--robustness trade-off and high training/tuning cost, which limits scalability and maintainability in medical imaging. We propose \emph{Layer-wise Noise-Guided Selective Wavelet Reconstruction (LNG-SWR)}. During training, we inject small, zero-mean noise at multiple layers to learn a frequency-bias prior that steers representations away from noise-sensitive directions. We then apply prior-guided selective wavelet reconstruction on the input/feature branch to achieve frequency adaptation: suppress noise-sensitive bands, enhance directional structures and shape cues, and stabilize boundary responses while maintaining spectral consistency. The framework is backbone-agnostic and adds low additional inference overhead. It can serve as a plug-in enhancement to AT and also improves robustness without AT. On CT and ultrasound datasets, under a unified protocol with PGD-$L_{\infty}/L_{2}$ and SSAH, LNG-SWR delivers consistent gains on clean Dice/IoU and significantly reduces the performance drop under strong attacks; combining LNG-SWR with AT yields additive gains. When combined with adversarial training, robustness improves further without sacrificing clean accuracy, indicating an engineering-friendly and scalable path to robust segmentation. These results indicate that LNG-SWR provides a simple, effective, and engineering-friendly path to robust medical image segmentation in both adversarial and standard training regimes.
>
---
#### [new 015] TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出TimeViper，一种用于长视频理解的混合Mamba-Transformer视觉语言模型。针对长视频处理中效率与上下文建模难题，通过混合架构提升效率，并发现视觉信息向文本传递导致冗余，提出TransV模块压缩视觉令牌，实现超长视频（>10,000帧）高效理解，实验验证其性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.16595v1](https://arxiv.org/pdf/2511.16595v1)**

> **作者:** Boshen Xu; Zihan Xiao; Jiaze Li; Jianzhong Ju; Zhenbo Luo; Jian Luan; Qin Jin
>
> **备注:** Project page: https://xuboshen.github.io/TimeViper
>
> **摘要:** We introduce TimeViper, a hybrid vision-language model designed to tackle challenges of long video understanding. Processing long videos demands both an efficient model architecture and an effective mechanism for handling extended temporal contexts. To this end, TimeViper adopts a hybrid Mamba-Transformer backbone that combines the efficiency of state-space models with the expressivity of attention mechanisms. Through this hybrid design, we reveal the vision-to-text information aggregation phenomenon, where information progressively flows from vision tokens to text tokens across increasing LLM depth, resulting in severe vision token redundancy. Motivated by this observation, we propose TransV, a token information transfer module that transfers and compresses vision tokens into instruction tokens while maintaining multimodal understanding capabilities. This design enables TimeViper to process hour-long videos exceeding 10,000 frames. Extensive experiments across multiple benchmarks demonstrate that TimeViper competes with state-of-the-art models while extending frame numbers. We further analyze attention behaviors of both Mamba and Transformer layers, offering new insights into hybrid model interpretability. This work represents an initial step towards developing, interpreting, and compressing hybrid Mamba-Transformer architectures.
>
---
#### [new 016] NoPo-Avatar: Generalizable and Animatable Avatars from Sparse Inputs without Human Poses
- **分类: cs.CV**

- **简介: 该论文针对从单张或稀疏图像重建可动画化3D人体形象的任务，提出NoPo-Avatar方法，无需依赖人体姿态输入。通过消除对测试时姿态的依赖，提升在噪声姿态下的鲁棒性，在真实场景中表现优于现有方法，同时在理想条件下保持竞争力。**

- **链接: [https://arxiv.org/pdf/2511.16673v1](https://arxiv.org/pdf/2511.16673v1)**

> **作者:** Jing Wen; Alexander G. Schwing; Shenlong Wang
>
> **备注:** NeurIPS'25; project page: https://wenj.github.io/NoPo-Avatar/
>
> **摘要:** We tackle the task of recovering an animatable 3D human avatar from a single or a sparse set of images. For this task, beyond a set of images, many prior state-of-the-art methods use accurate "ground-truth" camera poses and human poses as input to guide reconstruction at test-time. We show that pose-dependent reconstruction degrades results significantly if pose estimates are noisy. To overcome this, we introduce NoPo-Avatar, which reconstructs avatars solely from images, without any pose input. By removing the dependence of test-time reconstruction on human poses, NoPo-Avatar is not affected by noisy human pose estimates, making it more widely applicable. Experiments on challenging THuman2.0, XHuman, and HuGe100K data show that NoPo-Avatar outperforms existing baselines in practical settings (without ground-truth poses) and delivers comparable results in lab settings (with ground-truth poses).
>
---
#### [new 017] BioBench: A Blueprint to Move Beyond ImageNet for Scientific ML Benchmarks
- **分类: cs.CV**

- **简介: 该论文针对科学图像识别中ImageNet基准失效的问题，提出BioBench生态视觉基准。它整合9项应用任务、4个生物界、6种成像方式，共310万张图像，提供统一接口与评估指标，有效提升模型性能预测准确性，为科学领域AI基准建设提供范本。**

- **链接: [https://arxiv.org/pdf/2511.16315v1](https://arxiv.org/pdf/2511.16315v1)**

> **作者:** Samuel Stevens
>
> **备注:** Accepted at the 3rd Imageomics Workshop at NeurIPS 2025
>
> **摘要:** ImageNet-1K linear-probe transfer accuracy remains the default proxy for visual representation quality, yet it no longer predicts performance on scientific imagery. Across 46 modern vision model checkpoints, ImageNet top-1 accuracy explains only 34% of variance on ecology tasks and mis-ranks 30% of models above 75% accuracy. We present BioBench, an open ecology vision benchmark that captures what ImageNet misses. BioBench unifies 9 publicly released, application-driven tasks, 4 taxonomic kingdoms, and 6 acquisition modalities (drone RGB, web video, micrographs, in-situ and specimen photos, camera-trap frames), totaling 3.1M images. A single Python API downloads data, fits lightweight classifiers to frozen backbones, and reports class-balanced macro-F1 (plus domain metrics for FishNet and FungiCLEF); ViT-L models evaluate in 6 hours on an A6000 GPU. BioBench provides new signal for computer vision in ecology and a template recipe for building reliable AI-for-science benchmarks in any domain. Code and predictions are available at https://github.com/samuelstevens/biobench and results at https://samuelstevens.me/biobench.
>
---
#### [new 018] CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement
- **分类: cs.CV**

- **简介: 该论文针对组合零样本学习（CZSL）任务，解决现有方法因依赖全局语义表示导致属性与对象难以完全解耦的问题。提出CAMS模型，通过门控交叉注意力提取细粒度语义特征，并在多空间中实现属性与对象的解耦，提升对未见组合的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.16378v1](https://arxiv.org/pdf/2511.16378v1)**

> **作者:** Pan Yang; Cheng Deng; Jing Yang; Han Zhao; Yun Liu; Yuling Chen; Xiaoli Ruan; Yanping Chen
>
> **摘要:** Compositional zero-shot learning (CZSL) aims to learn the concepts of attributes and objects in seen compositions and to recognize their unseen compositions. Most Contrastive Language-Image Pre-training (CLIP)-based CZSL methods focus on disentangling attributes and objects by leveraging the global semantic representation obtained from the image encoder. However, this representation has limited representational capacity and do not allow for complete disentanglement of the two. To this end, we propose CAMS, which aims to extract semantic features from visual features and perform semantic disentanglement in multidimensional spaces, thereby improving generalization over unseen attribute-object compositions. Specifically, CAMS designs a Gated Cross-Attention that captures fine-grained semantic features from the high-level image encoding blocks of CLIP through a set of latent units, while adaptively suppressing background and other irrelevant information. Subsequently, it conducts Multi-Space Disentanglement to achieve disentanglement of attribute and object semantics. Experiments on three popular benchmarks (MIT-States, UT-Zappos, and C-GQA) demonstrate that CAMS achieves state-of-the-art performance in both closed-world and open-world settings. The code is available at https://github.com/ybyangjing/CAMS.
>
---
#### [new 019] UniDGF: A Unified Detection-to-Generation Framework for Hierarchical Object Visual Recognition
- **分类: cs.CV**

- **简介: 该论文提出UniDGF框架，针对大尺度电商场景下细粒度视觉识别任务，解决传统方法依赖全局相似性、难以区分细粒度类别和属性的问题。通过检测引导的生成机制，联合完成目标检测、类别与属性识别，实现从粗到细的语义序列生成，显著提升识别精度与推理一致性。**

- **链接: [https://arxiv.org/pdf/2511.15984v1](https://arxiv.org/pdf/2511.15984v1)**

> **作者:** Xinyu Nan; Lingtao Mao; Huangyu Dai; Zexin Zheng; Xinyu Sun; Zihan Liang; Ben Chen; Yuqing Ding; Chenyi Lei; Wenwu Ou; Han Li
>
> **摘要:** Achieving visual semantic understanding requires a unified framework that simultaneously handles object detection, category prediction, and attribute recognition. However, current advanced approaches rely on global similarity and struggle to capture fine-grained category distinctions and category-specific attribute diversity, especially in large-scale e-commerce scenarios. To overcome these challenges, we introduce a detection-guided generative framework that predicts hierarchical category and attribute tokens. For each detected object, we extract refined ROI-level features and employ a BART-based generator to produce semantic tokens in a coarse-to-fine sequence covering category hierarchies and property-value pairs, with support for property-conditioned attribute recognition. Experiments on both large-scale proprietary e-commerce datasets and open-source datasets demonstrate that our approach significantly outperforms existing similarity-based pipelines and multi-stage classification systems, achieving stronger fine-grained recognition and more coherent unified inference.
>
---
#### [new 020] SurvAgent: Hierarchical CoT-Enhanced Case Banking and Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对癌症生存预测中模型缺乏透明性的问题，提出SurvAgent系统。通过层次化思维链增强的多模态病例库构建与基于二分法的多专家推理，实现病理与基因数据融合、兴趣区域有效探索及历史经验学习，提升预测可解释性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.16635v1](https://arxiv.org/pdf/2511.16635v1)**

> **作者:** Guolin Huang; Wenting Chen; Jiaqi Yang; Xinheng Lyu; Xiaoling Luo; Sen Yang; Xiaohan Xing; Linlin Shen
>
> **备注:** 20 pages
>
> **摘要:** Survival analysis is critical for cancer prognosis and treatment planning, yet existing methods lack the transparency essential for clinical adoption. While recent pathology agents have demonstrated explainability in diagnostic tasks, they face three limitations for survival prediction: inability to integrate multimodal data, ineffective region-of-interest exploration, and failure to leverage experiential learning from historical cases. We introduce SurvAgent, the first hierarchical chain-of-thought (CoT)-enhanced multi-agent system for multimodal survival prediction. SurvAgent consists of two stages: (1) WSI-Gene CoT-Enhanced Case Bank Construction employs hierarchical analysis through Low-Magnification Screening, Cross-Modal Similarity-Aware Patch Mining, and Confidence-Aware Patch Mining for pathology images, while Gene-Stratified analysis processes six functional gene categories. Both generate structured reports with CoT reasoning, storing complete analytical processes for experiential learning. (2) Dichotomy-Based Multi-Expert Agent Inference retrieves similar cases via RAG and integrates multimodal reports with expert predictions through progressive interval refinement. Extensive experiments on five TCGA cohorts demonstrate SurvAgent's superority over conventional methods, proprietary MLLMs, and medical agents, establishing a new paradigm for explainable AI-driven survival prediction in precision oncology.
>
---
#### [new 021] StreetView-Waste: A Multi-Task Dataset for Urban Waste Management
- **分类: cs.CV**

- **简介: 该论文提出StreetView-Waste数据集，针对城市垃圾管理中垃圾箱溢出监测难题，聚焦垃圾箱检测、跟踪与溢出分割任务。通过构建多任务数据集并引入几何先验与启发式方法，显著提升跟踪精度与分割性能，为智慧城市建设中的实时感知提供有效基准。**

- **链接: [https://arxiv.org/pdf/2511.16440v1](https://arxiv.org/pdf/2511.16440v1)**

> **作者:** Diogo J. Paulo; João Martins; Hugo Proença; João C. Neves
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Urban waste management remains a critical challenge for the development of smart cities. Despite the growing number of litter detection datasets, the problem of monitoring overflowing waste containers, particularly from images captured by garbage trucks, has received little attention. While existing datasets are valuable, they often lack annotations for specific container tracking or are captured in static, decontextualized environments, limiting their utility for real-world logistics. To address this gap, we present StreetView-Waste, a comprehensive dataset of urban scenes featuring litter and waste containers. The dataset supports three key evaluation tasks: (1) waste container detection, (2) waste container tracking, and (3) waste overflow segmentation. Alongside the dataset, we provide baselines for each task by benchmarking state-of-the-art models in object detection, tracking, and segmentation. Additionally, we enhance baseline performance by proposing two complementary strategies: a heuristic-based method for improved waste container tracking and a model-agnostic framework that leverages geometric priors to refine litter segmentation. Our experimental results show that while fine-tuned object detectors achieve reasonable performance in detecting waste containers, baseline tracking methods struggle to accurately estimate their number; however, our proposed heuristics reduce the mean absolute counting error by 79.6%. Similarly, while segmenting amorphous litter is challenging, our geometry-aware strategy improves segmentation mAP@0.5 by 27% on lightweight models, demonstrating the value of multimodal inputs for this task. Ultimately, StreetView-Waste provides a challenging benchmark to encourage research into real-world perception systems for urban waste management.
>
---
#### [new 022] TriDiff-4D: Fast 4D Generation through Diffusion-based Triplane Re-posing
- **分类: cs.CV**

- **简介: 该论文提出TriDiff-4D，解决文本驱动高保真、可控4D角色生成中时序不一致、计算成本高等问题。通过扩散模型实现三平面重定位，采用自回归策略高效生成任意长度4D序列，支持骨骼驱动动画，显著提升生成速度与质量。**

- **链接: [https://arxiv.org/pdf/2511.16662v1](https://arxiv.org/pdf/2511.16662v1)**

> **作者:** Eddie Pokming Sheung; Qihao Liu; Wufei Ma; Prakhar Kaushik; Jianwen Xie; Alan Yuille
>
> **备注:** 8 pages, 10 figures, Under review at a conference
>
> **摘要:** With the increasing demand for 3D animation, generating high-fidelity, controllable 4D avatars from textual descriptions remains a significant challenge. Despite notable efforts in 4D generative modeling, existing methods exhibit fundamental limitations that impede their broader applicability, including temporal and geometric inconsistencies, perceptual artifacts, motion irregularities, high computational costs, and limited control over dynamics. To address these challenges, we propose TriDiff-4D, a novel 4D generative pipeline that employs diffusion-based triplane re-posing to produce high-quality, temporally coherent 4D avatars. Our model adopts an auto-regressive strategy to generate 4D sequences of arbitrary length, synthesizing each 3D frame with a single diffusion process. By explicitly learning 3D structure and motion priors from large-scale 3D and motion datasets, TriDiff-4D enables skeleton-driven 4D generation that excels in temporal consistency, motion accuracy, computational efficiency, and visual fidelity. Specifically, TriDiff-4D first generates a canonical 3D avatar and a corresponding motion sequence from a text prompt, then uses a second diffusion model to animate the avatar according to the motion sequence, supporting arbitrarily long 4D generation. Experimental results demonstrate that TriDiff-4D significantly outperforms existing methods, reducing generation time from hours to seconds by eliminating the optimization process, while substantially improving the generation of complex motions with high-fidelity appearance and accurate 3D geometry.
>
---
#### [new 023] POMA-3D: The Point Map Way to 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文提出POMA-3D，首个基于点图的自监督3D表征模型。针对3D表示学习中预训练先验稀缺与数据有限的问题，利用点图编码3D坐标，结合2D基础模型先验，设计视图-场景对齐与联合嵌入预测架构，构建ScenePoint数据集，实现仅用几何输入的多种3D理解任务。**

- **链接: [https://arxiv.org/pdf/2511.16567v1](https://arxiv.org/pdf/2511.16567v1)**

> **作者:** Ye Mao; Weixun Luo; Ranran Huang; Junpeng Jing; Krystian Mikolajczyk
>
> **备注:** 11 pages, 6 tables, 5 figures
>
> **摘要:** In this paper, we introduce POMA-3D, the first self-supervised 3D representation model learned from point maps. Point maps encode explicit 3D coordinates on a structured 2D grid, preserving global 3D geometry while remaining compatible with the input format of 2D foundation models. To transfer rich 2D priors into POMA-3D, a view-to-scene alignment strategy is designed. Moreover, as point maps are view-dependent with respect to a canonical space, we introduce POMA-JEPA, a joint embedding-predictive architecture that enforces geometrically consistent point map features across multiple views. Additionally, we introduce ScenePoint, a point map dataset constructed from 6.5K room-level RGB-D scenes and 1M 2D image scenes to facilitate large-scale POMA-3D pretraining. Experiments show that POMA-3D serves as a strong backbone for both specialist and generalist 3D understanding. It benefits diverse tasks, including 3D question answering, embodied navigation, scene retrieval, and embodied localization, all achieved using only geometric inputs (i.e., 3D coordinates). Overall, our POMA-3D explores a point map way to 3D scene understanding, addressing the scarcity of pretrained priors and limited data in 3D representation learning. Project Page: https://matchlab-imperial.github.io/poma3d/
>
---
#### [new 024] Contrastive vision-language learning with paraphrasing and negation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视觉-语言模型在面对文本否定和改写时性能不稳定的问题，提出SemCLIP方法。通过引入大模型生成的原始、改写与否定文本三元组，设计新型对比损失函数，增强模型对语义变换的鲁棒性。实验表明，该方法显著提升对否定文本的区分能力，并在多个任务上优于基线模型。**

- **链接: [https://arxiv.org/pdf/2511.16527v1](https://arxiv.org/pdf/2511.16527v1)**

> **作者:** Kwun Ho Ngan; Saman Sadeghi Afgeh; Joe Townsend; Artur d'Avila Garcez
>
> **摘要:** Contrastive vision-language models continue to be the dominant approach for image and text retrieval. Contrastive Language-Image Pre-training (CLIP) trains two neural networks in contrastive manner to align their image and text embeddings in a shared latent space. Recent results evaluating CLIP on negated or paraphrased text have shown mixed performance because negation changes meaning radically with minimal lexical changes, while paraphrasing can create very different textual expressions with the same intended meaning. This poses a significant challenge for improving the evaluation results and alignment of vision-language models. To address this challenge, this paper evaluates the combination of paraphrasing and negation, proposes a new CLIP contrastive loss function accounting for both paraphrasing and negation, and applies LLM-generated training triples consisting of original, paraphrased and negated textual captions to CLIP-like training models. The approach, called SemCLIP, is shown to move paraphrased captions towards the original image embeddings while pushing negated captions further away in embedding space. Empirically, SemCLIP is shown to be capable of preserving CLIP's performance while increasing considerably the distances to negated captions. On the CC-Neg benchmark using an original over negation image-retrieval accuracy metric, SemCLIP improves accuracy from 68.1% to 78.1%. Although results are mixed when compared with CLIP on the Sugarcrepe++ benchmark, SemCLIP's performance is generally better than the models trained with negated captions. This robustness to negation extends to downstream zero-shot classification tasks where SemCLIP pre-trained on Sugarcrepe++ performs better than CLIP on all tested downstream tasks. These results indicate that SemCLIP can achieve significant robustness to semantic transformations.
>
---
#### [new 025] Supervised Contrastive Learning for Few-Shot AI-Generated Image Detection and Attribution
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对生成式AI图像检测与溯源难题，提出一种两阶段监督对比学习框架。通过在部分生成模型上训练提取判别性特征，再用少量样本的k-NN分类器实现少样本泛化，有效提升对未知生成模型图像的检测与溯源性能。**

- **链接: [https://arxiv.org/pdf/2511.16541v1](https://arxiv.org/pdf/2511.16541v1)**

> **作者:** Jaime Álvarez Urueña; David Camacho; Javier Huertas Tato
>
> **备注:** 17 pages, 6 figures, 6 tables
>
> **摘要:** The rapid advancement of generative artificial intelligence has enabled the creation of synthetic images that are increasingly indistinguishable from authentic content, posing significant challenges for digital media integrity. This problem is compounded by the accelerated release cycle of novel generative models, which renders traditional detection approaches (reliant on periodic retraining) computationally infeasible and operationally impractical. This work proposes a novel two-stage detection framework designed to address the generalization challenge inherent in synthetic image detection. The first stage employs a vision deep learning model trained via supervised contrastive learning to extract discriminative embeddings from input imagery. Critically, this model was trained on a strategically partitioned subset of available generators, with specific architectures withheld from training to rigorously ablate cross-generator generalization capabilities. The second stage utilizes a k-nearest neighbors (k-NN) classifier operating on the learned embedding space, trained in a few-shot learning paradigm incorporating limited samples from previously unseen test generators. With merely 150 images per class in the few-shot learning regime, which are easily obtainable from current generation models, the proposed framework achieves an average detection accuracy of 91.3\%, representing a 5.2 percentage point improvement over existing approaches . For the source attribution task, the proposed approach obtains improvements of of 14.70\% and 4.27\% in AUC and OSCR respectively on an open set classification context, marking a significant advancement toward robust, scalable forensic attribution systems capable of adapting to the evolving generative AI landscape without requiring exhaustive retraining protocols.
>
---
#### [new 026] Graph Neural Networks for Surgical Scene Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对腹腔镜胆囊切除术中解剖结构分割难题，提出基于视觉变压器与图神经网络结合的分割方法，通过静态和动态图结构建模空间关系，提升对细小、稀有及关键结构的识别精度，显著改善分割性能与解剖一致性。**

- **链接: [https://arxiv.org/pdf/2511.16430v1](https://arxiv.org/pdf/2511.16430v1)**

> **作者:** Yihan Li; Nikhil Churamani; Maria Robu; Imanol Luengo; Danail Stoyanov
>
> **备注:** 12 pages, 4 figures, 3 tables
>
> **摘要:** Purpose: Accurate identification of hepatocystic anatomy is critical to preventing surgical complications during laparoscopic cholecystectomy. Deep learning models often struggle with occlusions, long-range dependencies, and capturing the fine-scale geometry of rare structures. This work addresses these challenges by introducing graph-based segmentation approaches that enhance spatial and semantic understanding in surgical scene analyses. Methods: We propose two segmentation models integrating Vision Transformer (ViT) feature encoders with Graph Neural Networks (GNNs) to explicitly model spatial relationships between anatomical regions. (1) A static k Nearest Neighbours (k-NN) graph with a Graph Convolutional Network with Initial Residual and Identity Mapping (GCNII) enables stable long-range information propagation. (2) A dynamic Differentiable Graph Generator (DGG) with a Graph Attention Network (GAT) supports adaptive topology learning. Both models are evaluated on the Endoscapes-Seg50 and CholecSeg8k benchmarks. Results: The proposed approaches achieve up to 7-8% improvement in Mean Intersection over Union (mIoU) and 6% improvement in Mean Dice (mDice) scores over state-of-the-art baselines. It produces anatomically coherent predictions, particularly on thin, rare and safety-critical structures. Conclusion: The proposed graph-based segmentation methods enhance both performance and anatomical consistency in surgical scene segmentation. By combining ViT-based global context with graph-based relational reasoning, the models improve interpretability and reliability, paving the way for safer laparoscopic and robot-assisted surgery through a precise identification of critical anatomical features.
>
---
#### [new 027] Towards a Safer and Sustainable Manufacturing Process: Material classification in Laser Cutting Using Deep Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于材料分类任务，旨在解决激光切割中因烟尘危害和材料识别不准带来的安全与效率问题。通过构建基于深度学习的卷积神经网络，利用激光散斑图实现不同材料的高精度实时识别，即使激光颜色改变也能保持稳定性能，显著提升切割过程的安全性与可持续性。**

- **链接: [https://arxiv.org/pdf/2511.16026v1](https://arxiv.org/pdf/2511.16026v1)**

> **作者:** Mohamed Abdallah Salem; Hamdy Ahmed Ashur; Ahmed Elshinnawy
>
> **摘要:** Laser cutting is a widely adopted technology in material processing across various industries, but it generates a significant amount of dust, smoke, and aerosols during operation, posing a risk to both the environment and workers' health. Speckle sensing has emerged as a promising method to monitor the cutting process and identify material types in real-time. This paper proposes a material classification technique using a speckle pattern of the material's surface based on deep learning to monitor and control the laser cutting process. The proposed method involves training a convolutional neural network (CNN) on a dataset of laser speckle patterns to recognize distinct material types for safe and efficient cutting. Previous methods for material classification using speckle sensing may face issues when the color of the laser used to produce the speckle pattern is changed. Experiments conducted in this study demonstrate that the proposed method achieves high accuracy in material classification, even when the laser color is changed. The model achieved an accuracy of 98.30 % on the training set and 96.88% on the validation set. Furthermore, the model was evaluated on a set of 3000 new images for 30 different materials, achieving an F1-score of 0.9643. The proposed method provides a robust and accurate solution for material-aware laser cutting using speckle sensing.
>
---
#### [new 028] How Noise Benefits AI-generated Image Detection
- **分类: cs.CV**

- **简介: 该论文针对AI生成图像检测中泛化能力弱的问题，提出PiN-CLIP方法。通过在特征空间注入正向激励噪声，抑制训练中的虚假捷径，增强稳定伪造特征，提升模型对多类生成模型的鲁棒性与泛化能力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16136v1](https://arxiv.org/pdf/2511.16136v1)**

> **作者:** Jiazhen Yan; Ziqiang Li; Fan Wang; Kai Zeng; Zhangjie Fu
>
> **摘要:** The rapid advancement of generative models has made real and synthetic images increasingly indistinguishable. Although extensive efforts have been devoted to detecting AI-generated images, out-of-distribution generalization remains a persistent challenge. We trace this weakness to spurious shortcuts exploited during training and we also observe that small feature-space perturbations can mitigate shortcut dominance. To address this problem in a more controllable manner, we propose the Positive-Incentive Noise for CLIP (PiN-CLIP), which jointly trains a noise generator and a detection network under a variational positive-incentive principle. Specifically, we construct positive-incentive noise in the feature space via cross-attention fusion of visual and categorical semantic features. During optimization, the noise is injected into the feature space to fine-tune the visual encoder, suppressing shortcut-sensitive directions while amplifying stable forensic cues, thereby enabling the extraction of more robust and generalized artifact representations. Comparative experiments are conducted on an open-world dataset comprising synthetic images generated by 42 distinct generative models. Our method achieves new state-of-the-art performance, with notable improvements of 5.4 in average accuracy over existing approaches.
>
---
#### [new 029] NaTex: Seamless Texture Generation as Latent Color Diffusion
- **分类: cs.CV**

- **简介: 该论文提出NaTex，一种直接在3D空间生成纹理颜色的框架。针对传统多视图扩散模型在遮挡处理、纹理对齐和跨视图一致性上的缺陷，提出基于潜在颜色扩散的3D纹理生成方法，通过几何感知的点云VAE与多控制扩散变压器实现精准纹理重建与生成，具备强泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.16317v1](https://arxiv.org/pdf/2511.16317v1)**

> **作者:** Zeqiang Lai; Yunfei Zhao; Zibo Zhao; Xin Yang; Xin Huang; Jingwei Huang; Xiangyu Yue; Chunchao Guo
>
> **备注:** Technical Report
>
> **摘要:** We present NaTex, a native texture generation framework that predicts texture color directly in 3D space. In contrast to previous approaches that rely on baking 2D multi-view images synthesized by geometry-conditioned Multi-View Diffusion models (MVDs), NaTex avoids several inherent limitations of the MVD pipeline. These include difficulties in handling occluded regions that require inpainting, achieving precise mesh-texture alignment along boundaries, and maintaining cross-view consistency and coherence in both content and color intensity. NaTex features a novel paradigm that addresses the aforementioned issues by viewing texture as a dense color point cloud. Driven by this idea, we propose latent color diffusion, which comprises a geometry-awared color point cloud VAE and a multi-control diffusion transformer (DiT), entirely trained from scratch using 3D data, for texture reconstruction and generation. To enable precise alignment, we introduce native geometry control that conditions the DiT on direct 3D spatial information via positional embeddings and geometry latents. We co-design the VAE-DiT architecture, where the geometry latents are extracted via a dedicated geometry branch tightly coupled with the color VAE, providing fine-grained surface guidance that maintains strong correspondence with the texture. With these designs, NaTex demonstrates strong performance, significantly outperforming previous methods in texture coherence and alignment. Moreover, NaTex also exhibits strong generalization capabilities, either training-free or with simple tuning, for various downstream applications, e.g., material generation, texture refinement, and part segmentation and texturing.
>
---
#### [new 030] SAM2S: Segment Anything in Surgical Videos via Semantic Long-term Tracking
- **分类: cs.CV; eess.IV; q-bio.TO**

- **简介: 该论文针对手术视频中交互式视频目标分割任务，解决现有模型在长时跟踪与跨数据集泛化上的不足。提出SAM2S，通过可训练的多样记忆机制、时序语义学习和抗歧义训练，在自建大型手术视频标注数据集SA-SV上实现高效实时分割，显著提升精度与零样本泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.16618v1](https://arxiv.org/pdf/2511.16618v1)**

> **作者:** Haofeng Liu; Ziyue Wang; Sudhanshu Mishra; Mingqi Gao; Guanyi Qin; Chang Han Low; Alex Y. W. Kong; Yueming Jin
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Surgical video segmentation is crucial for computer-assisted surgery, enabling precise localization and tracking of instruments and tissues. Interactive Video Object Segmentation (iVOS) models such as Segment Anything Model 2 (SAM2) provide prompt-based flexibility beyond methods with predefined categories, but face challenges in surgical scenarios due to the domain gap and limited long-term tracking. To address these limitations, we construct SA-SV, the largest surgical iVOS benchmark with instance-level spatio-temporal annotations (masklets) spanning eight procedure types (61k frames, 1.6k masklets), enabling comprehensive development and evaluation for long-term tracking and zero-shot generalization. Building on SA-SV, we propose SAM2S, a foundation model enhancing \textbf{SAM2} for \textbf{S}urgical iVOS through: (1) DiveMem, a trainable diverse memory mechanism for robust long-term tracking; (2) temporal semantic learning for instrument understanding; and (3) ambiguity-resilient learning to mitigate annotation inconsistencies across multi-source datasets. Extensive experiments demonstrate that fine-tuning on SA-SV enables substantial performance gains, with SAM2 improving by 12.99 average $\mathcal{J}$\&$\mathcal{F}$ over vanilla SAM2. SAM2S further advances performance to 80.42 average $\mathcal{J}$\&$\mathcal{F}$, surpassing vanilla and fine-tuned SAM2 by 17.10 and 4.11 points respectively, while maintaining 68 FPS real-time inference and strong zero-shot generalization. Code and dataset will be released at https://jinlab-imvr.github.io/SAM2S.
>
---
#### [new 031] T2T-VICL: Unlocking the Boundaries of Cross-Task Visual In-Context Learning via Implicit Text-Driven VLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究跨任务视觉上下文学习（Cross-task VICL）问题，旨在探索统一视觉语言模型在不同视觉任务间迁移学习的潜力。提出T2T-VICL框架，通过生成与选择隐式文本提示，构建首个跨任务VICL数据集，并设计结合感知评分与传统指标的推理机制，实现多场景下优异性能，突破了VLMs在跨任务学习中的边界。**

- **链接: [https://arxiv.org/pdf/2511.16107v1](https://arxiv.org/pdf/2511.16107v1)**

> **作者:** Shao-Jun Xia; Huixin Zhang; Zhengzhong Tu
>
> **摘要:** In large language models (LLM), in-context learning (ICL) refers to performing new tasks by conditioning on small demonstrations provided in the input context. Recent advances in visual in-context learning (VICL) demonstrate promising capabilities for solving downstream tasks by unified vision-language models (VLMs). When the visual prompt and the target images originate from different visual tasks, can VLMs still enable VICL? In the paper, we propose a fully collaborative pipeline, i.e. T2T-VICL, for VLMs to investigate the potential of cross-task VICL. Fundamentally, we design a mechanism to generate and select text prompts that best implicitly describe the differences between two distinct low-level vision tasks, and construct the first cross-task VICL dataset. Building upon this, we propose a novel inference framework that combines perceptual score-based reasoning with traditional evaluation metrics to perform cross-task VICL. Our approach achieves top-tier results across nine cross-task scenarios and second-tier performance in ten additional scenarios, unlocking the boundaries of cross-task VICL within VLMs.
>
---
#### [new 032] BoxingVI: A Multi-Modal Benchmark for Boxing Action Recognition and Localization
- **分类: cs.CV**

- **简介: 该论文提出BoxingVI，一个用于拳击动作识别与定位的多模态基准数据集。针对拳击动作复杂、环境多变导致的分析难题，构建了6,915个标注精确的拳击片段，涵盖六类拳法，支持实时视觉分析与智能教练系统研究。**

- **链接: [https://arxiv.org/pdf/2511.16524v1](https://arxiv.org/pdf/2511.16524v1)**

> **作者:** Rahul Kumar; Vipul Baghel; Sudhanshu Singh; Bikash Kumar Badatya; Shivam Yadav; Babji Srinivasan; Ravi Hegde
>
> **摘要:** Accurate analysis of combat sports using computer vision has gained traction in recent years, yet the development of robust datasets remains a major bottleneck due to the dynamic, unstructured nature of actions and variations in recording environments. In this work, we present a comprehensive, well-annotated video dataset tailored for punch detection and classification in boxing. The dataset comprises 6,915 high-quality punch clips categorized into six distinct punch types, extracted from 20 publicly available YouTube sparring sessions and involving 18 different athletes. Each clip is manually segmented and labeled to ensure precise temporal boundaries and class consistency, capturing a wide range of motion styles, camera angles, and athlete physiques. This dataset is specifically curated to support research in real-time vision-based action recognition, especially in low-resource and unconstrained environments. By providing a rich benchmark with diverse punch examples, this contribution aims to accelerate progress in movement analysis, automated coaching, and performance assessment within boxing and related domains.
>
---
#### [new 033] Multi-Order Matching Network for Alignment-Free Depth Super-Resolution
- **分类: cs.CV**

- **简介: 该论文针对非对齐RGB-D数据下的深度超分辨率任务，提出多阶匹配网络（MOMNet）。通过多阶匹配与聚合机制，自适应融合错位RGB信息，实现无需严格对齐的高质量深度重建，显著提升真实场景下的鲁棒性与性能。**

- **链接: [https://arxiv.org/pdf/2511.16361v1](https://arxiv.org/pdf/2511.16361v1)**

> **作者:** Zhengxue Wang; Zhiqiang Yan; Yuan Wu; Guangwei Gao; Xiang Li; Jian Yang
>
> **摘要:** Recent guided depth super-resolution methods are premised on the assumption of strictly spatial alignment between depth and RGB, achieving high-quality depth reconstruction. However, in real-world scenarios, the acquisition of strictly aligned RGB-D is hindered by inherent hardware limitations (e.g., physically separate RGB-D sensors) and unavoidable calibration drift induced by mechanical vibrations or temperature variations. Consequently, existing approaches often suffer inevitable performance degradation when applied to misaligned real-world scenes. In this paper, we propose the Multi-Order Matching Network (MOMNet), a novel alignment-free framework that adaptively retrieves and selects the most relevant information from misaligned RGB. Specifically, our method begins with a multi-order matching mechanism, which jointly performs zero-order, first-order, and second-order matching to comprehensively identify RGB information consistent with depth across multi-order feature spaces. To effectively integrate the retrieved RGB and depth, we further introduce a multi-order aggregation composed of multiple structure detectors. This strategy uses multi-order priors as prompts to facilitate the selective feature transfer from RGB to depth. Extensive experiments demonstrate that MOMNet achieves state-of-the-art performance and exhibits outstanding robustness.
>
---
#### [new 034] Degradation-Aware Hierarchical Termination for Blind Quality Enhancement of Compressed Video
- **分类: cs.CV**

- **简介: 该论文针对盲质量增强压缩视频（Blind QECV）任务，解决现有方法依赖已知量化参数及忽视空间细节与计算差异的问题。提出预训练的退化表征学习模块以提取多尺度退化特征，并设计分层终止机制，根据压缩等级动态调整处理深度，提升性能并降低推理时间。**

- **链接: [https://arxiv.org/pdf/2511.16137v1](https://arxiv.org/pdf/2511.16137v1)**

> **作者:** Li Yu; Yingbo Zhao; Shiyu Wu; Siyue Yu; Moncef Gabbouj; Qingshan Liu
>
> **摘要:** Existing studies on Quality Enhancement for Compressed Video (QECV) predominantly rely on known Quantization Parameters (QPs), employing distinct enhancement models per QP setting, termed non-blind methods. However, in real-world scenarios involving transcoding or transmission, QPs may be partially or entirely unknown, limiting the applicability of such approaches and motivating the development of blind QECV techniques. Current blind methods generate degradation vectors via classification models with cross-entropy loss, using them as channel attention to guide artifact removal. However, these vectors capture only global degradation information and lack spatial details, hindering adaptation to varying artifact patterns at different spatial positions. To address these limitations, we propose a pretrained Degradation Representation Learning (DRL) module that decouples and extracts high-dimensional, multiscale degradation representations from video content to guide the artifact removal. Additionally, both blind and non-blind methods typically employ uniform architectures across QPs, hence, overlooking the varying computational demands inherent to different compression levels. We thus introduce a hierarchical termination mechanism that dynamically adjusts the number of artifact reduction stages based on the compression level. Experimental results demonstrate that the proposed approach significantly enhances performance, achieving a PSNR improvement of 110% (from 0.31 dB to 0.65 dB) over a competing state-of-the-art blind method at QP = 22. Furthermore, the proposed hierarchical termination mechanism reduces the average inference time at QP = 22 by half compared to QP = 42.
>
---
#### [new 035] Decoupling Complexity from Scale in Latent Diffusion Model
- **分类: cs.CV**

- **简介: 该论文针对视觉生成任务中尺度与内容复杂度耦合的问题，提出DCS-LDM模型。通过构建层级化、尺度无关的潜在空间，将复杂度与尺度解耦，支持任意分辨率和帧率生成，并实现渐进式粗到细生成，提升生成灵活性与效率。**

- **链接: [https://arxiv.org/pdf/2511.16117v1](https://arxiv.org/pdf/2511.16117v1)**

> **作者:** Tianxiong Zhong; Xingye Tian; Xuebo Wang; Boyuan Jiang; Xin Tao; Pengfei Wan
>
> **备注:** 15 pages, 16 figures
>
> **摘要:** Existing latent diffusion models typically couple scale with content complexity, using more latent tokens to represent higher-resolution images or higher-frame rate videos. However, the latent capacity required to represent visual data primarily depends on content complexity, with scale serving only as an upper bound. Motivated by this observation, we propose DCS-LDM, a novel paradigm for visual generation that decouples information complexity from scale. DCS-LDM constructs a hierarchical, scale-independent latent space that models sample complexity through multi-level tokens and supports decoding to arbitrary resolutions and frame rates within a fixed latent representation. This latent space enables DCS-LDM to achieve a flexible computation-quality tradeoff. Furthermore, by decomposing structural and detailed information across levels, DCS-LDM supports a progressive coarse-to-fine generation paradigm. Experimental results show that DCS-LDM delivers performance comparable to state-of-the-art methods while offering flexible generation across diverse scales and visual qualities.
>
---
#### [new 036] Physically Realistic Sequence-Level Adversarial Clothing for Robust Human-Detection Evasion
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对深度学习人体检测系统易受对抗攻击的问题，提出一种序列级物理可实现的服装对抗样本生成方法。通过在UV空间参数化服装纹理，结合物理仿真模拟运动与光照变化，优化控制点使检测置信度在长时间视频中持续降低。实验验证了其在数字与物理场景下的强鲁棒性与跨模型迁移能力。**

- **链接: [https://arxiv.org/pdf/2511.16020v1](https://arxiv.org/pdf/2511.16020v1)**

> **作者:** Dingkun Zhou; Patrick P. K. Chan; Hengxu Wu; Shikang Zheng; Ruiqi Huang; Yuanjie Zhao
>
> **摘要:** Deep neural networks used for human detection are highly vulnerable to adversarial manipulation, creating safety and privacy risks in real surveillance environments. Wearable attacks offer a realistic threat model, yet existing approaches usually optimize textures frame by frame and therefore fail to maintain concealment across long video sequences with motion, pose changes, and garment deformation. In this work, a sequence-level optimization framework is introduced to generate natural, printable adversarial textures for shirts, trousers, and hats that remain effective throughout entire walking videos in both digital and physical settings. Product images are first mapped to UV space and converted into a compact palette and control-point parameterization, with ICC locking to keep all colors printable. A physically based human-garment pipeline is then employed to simulate motion, multi-angle camera viewpoints, cloth dynamics, and illumination variation. An expectation-over-transformation objective with temporal weighting is used to optimize the control points so that detection confidence is minimized across whole sequences. Extensive experiments demonstrate strong and stable concealment, high robustness to viewpoint changes, and superior cross-model transferability. Physical garments produced with sublimation printing achieve reliable suppression under indoor and outdoor recordings, confirming real-world feasibility.
>
---
#### [new 037] SwiTrack: Tri-State Switch for Cross-Modal Object Tracking
- **分类: cs.CV**

- **简介: 该论文针对跨模态目标跟踪（CMOT）任务，解决模态切换时特征提取不充分与目标漂移问题。提出SwiTrack框架，通过视觉编码器、NIR门控适配器和轨迹预测模块实现三流协同，结合动态模板重建与相似性对齐损失，提升跨模态一致性与追踪鲁棒性，在多个基准上达到实时性能并显著超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16227v1](https://arxiv.org/pdf/2511.16227v1)**

> **作者:** Boyue Xu; Ruichao Hou; Tongwei Ren; Dongming Zhou; Gangshan Wu; Jinde Cao
>
> **摘要:** Cross-modal object tracking (CMOT) is an emerging task that maintains target consistency while the video stream switches between different modalities, with only one modality available in each frame, mostly focusing on RGB-Near Infrared (RGB-NIR) tracking. Existing methods typically connect parallel RGB and NIR branches to a shared backbone, which limits the comprehensive extraction of distinctive modality-specific features and fails to address the issue of object drift, especially in the presence of unreliable inputs. In this paper, we propose SwiTrack, a novel state-switching framework that redefines CMOT through the deployment of three specialized streams. Specifically, RGB frames are processed by the visual encoder, while NIR frames undergo refinement via a NIR gated adapter coupled with the visual encoder to progressively calibrate shared latent space features, thereby yielding more robust cross-modal representations. For invalid modalities, a consistency trajectory prediction module leverages spatio-temporal cues to estimate target movement, ensuring robust tracking and mitigating drift. Additionally, we incorporate dynamic template reconstruction to iteratively update template features and employ a similarity alignment loss to reinforce feature consistency. Experimental results on the latest benchmarks demonstrate that our tracker achieves state-of-the-art performance, boosting precision rate and success rate gains by 7.2\% and 4.3\%, respectively, while maintaining real-time tracking at 65 frames per second. Code and models are available at https://github.com/xuboyue1999/SwiTrack.git.
>
---
#### [new 038] VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人任务中因处理连续视觉流导致的高计算开销问题，提出VLA-Pruner。该方法基于双层重要性准则（语义与动作层面），结合时间连续性，实现高效视觉标记剪枝，兼顾语义理解与动作执行，显著提升推理效率与性能。**

- **链接: [https://arxiv.org/pdf/2511.16449v1](https://arxiv.org/pdf/2511.16449v1)**

> **作者:** Ziyan Liu; Yeqiu Chen; Hongyi Cai; Tao Lin; Shuo Yang; Zheng Liu; Bo Zhao
>
> **摘要:** Vision-Language-Action (VLA) models have shown great promise for embodied AI, yet the heavy computational cost of processing continuous visual streams severely limits their real-time deployment. Token pruning (keeping salient visual tokens and dropping redundant ones) has emerged as an effective approach for accelerating Vision-Language Models (VLMs), offering a solution for efficient VLA. However, these VLM-specific token pruning methods select tokens based solely on semantic salience metrics (e.g., prefill attention), while overlooking the VLA's intrinsic dual-system nature of high-level semantic understanding and low-level action execution. Consequently, these methods bias token retention toward semantic cues, discard critical information for action generation, and significantly degrade VLA performance. To bridge this gap, we propose VLA-Pruner, a versatile plug-and-play VLA-specific token prune method that aligns with the dual-system nature of VLA models and exploits the temporal continuity in robot manipulation. Specifically, VLA-Pruner adopts a dual-level importance criterion for visual token retention: vision-language prefill attention for semantic-level relevance and action decode attention, estimated via temporal smoothing, for action-level importance. Based on this criterion, VLA-Pruner proposes a novel dual-level token selection strategy that adaptively preserves a compact, informative set of visual tokens for both semantic understanding and action execution under given compute budget. Experiments show that VLA-Pruner achieves state-of-the-art performance across multiple VLA architectures and diverse robotic tasks.
>
---
#### [new 039] Unsupervised Image Classification with Adaptive Nearest Neighbor Selection and Cluster Ensembles
- **分类: cs.CV**

- **简介: 该论文聚焦于无监督图像分类任务，旨在提升聚类性能。针对现有方法忽略表示学习的问题，提出ICCE框架：通过自适应近邻选择与聚类集成，生成共识伪标签，并训练分类器。在多个基准上取得领先效果，首次实现无监督ImageNet分类超70%准确率。**

- **链接: [https://arxiv.org/pdf/2511.16213v1](https://arxiv.org/pdf/2511.16213v1)**

> **作者:** Melih Baydar; Emre Akbas
>
> **摘要:** Unsupervised image classification, or image clustering, aims to group unlabeled images into semantically meaningful categories. Early methods integrated representation learning and clustering within an iterative framework. However, the rise of foundational models have recently shifted focus solely to clustering, bypassing the representation learning step. In this work, we build upon a recent multi-head clustering approach by introducing adaptive nearest neighbor selection and cluster ensembling strategies to improve clustering performance. Our method, "Image Clustering through Cluster Ensembles" (ICCE), begins with a clustering stage, where we train multiple clustering heads on a frozen backbone, producing diverse image clusterings. We then employ a cluster ensembling technique to consolidate these potentially conflicting results into a unified consensus clustering. Finally, we train an image classifier using the consensus clustering result as pseudo-labels. ICCE achieves state-of-the-art performance on ten image classification benchmarks, achieving 99.3% accuracy on CIFAR10, 89% on CIFAR100, and 70.4% on ImageNet datasets, narrowing the performance gap with supervised methods. To the best of our knowledge, ICCE is the first fully unsupervised image classification method to exceed 70% accuracy on ImageNet.
>
---
#### [new 040] CuriGS: Curriculum-Guided Gaussian Splatting for Sparse View Synthesis
- **分类: cs.CV**

- **简介: 该论文针对稀疏视图3D重建中因视角少导致的过拟合与监督不足问题，提出CuriGS框架。通过引入基于教师视角生成的多级扰动学生视图，采用渐进式训练策略，结合深度相关性与多信号评估，动态优化并扩充训练视图，显著提升重建质量与几何一致性。**

- **链接: [https://arxiv.org/pdf/2511.16030v1](https://arxiv.org/pdf/2511.16030v1)**

> **作者:** Zijian Wu; Mingfeng Jiang; Zidian Lin; Ying Song; Hanjie Ma; Qun Wu; Dongping Zhang; Guiyang Pu
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently emerged as an efficient, high-fidelity representation for real-time scene reconstruction and rendering. However, extending 3DGS to sparse-view settings remains challenging because of supervision scarcity and overfitting caused by limited viewpoint coverage. In this paper, we present CuriGS, a curriculum-guided framework for sparse-view 3D reconstruction using 3DGS. CuriGS addresses the core challenge of sparse-view synthesis by introducing student views: pseudo-views sampled around ground-truth poses (teacher). For each teacher, we generate multiple groups of student views with different perturbation levels. During training, we follow a curriculum schedule that gradually unlocks higher perturbation level, randomly sampling candidate students from the active level to assist training. Each sampled student is regularized via depth-correlation and co-regularization, and evaluated using a multi-signal metric that combines SSIM, LPIPS, and an image-quality measure. For every teacher and perturbation level, we periodically retain the best-performing students and promote those that satisfy a predefined quality threshold to the training set, resulting in a stable augmentation of sparse training views. Experimental results show that CuriGS outperforms state-of-the-art baselines in both rendering fidelity and geometric consistency across various synthetic and real sparse-view scenes. Project page: https://zijian1026.github.io/CuriGS/
>
---
#### [new 041] Aerial View River Landform Video segmentation: A Weakly Supervised Context-aware Temporal Consistency Distillation Approach
- **分类: cs.CV**

- **简介: 该论文针对无人机遥感下河流地貌视频分割任务，解决标注数据稀缺与时间一致性不足问题。提出弱监督上下文感知时序一致性蒸馏方法，通过教师-学生架构与关键帧更新，仅用30%标注数据即提升mIoU与时间一致性，实现稳定地形定位。**

- **链接: [https://arxiv.org/pdf/2511.16343v1](https://arxiv.org/pdf/2511.16343v1)**

> **作者:** Chi-Han Chen; Chieh-Ming Chen; Wen-Huang Cheng; Ching-Chun Huang
>
> **摘要:** The study of terrain and landform classification through UAV remote sensing diverges significantly from ground vehicle patrol tasks. Besides grappling with the complexity of data annotation and ensuring temporal consistency, it also confronts the scarcity of relevant data and the limitations imposed by the effective range of many technologies. This research substantiates that, in aerial positioning tasks, both the mean Intersection over Union (mIoU) and temporal consistency (TC) metrics are of paramount importance. It is demonstrated that fully labeled data is not the optimal choice, as selecting only key data lacks the enhancement in TC, leading to failures. Hence, a teacher-student architecture, coupled with key frame selection and key frame updating algorithms, is proposed. This framework successfully performs weakly supervised learning and TC knowledge distillation, overcoming the deficiencies of traditional TC training in aerial tasks. The experimental results reveal that our method utilizing merely 30\% of labeled data, concurrently elevates mIoU and temporal consistency ensuring stable localization of terrain objects. Result demo : https://gitlab.com/prophet.ai.inc/drone-based-riverbed-inspection
>
---
#### [new 042] Building temporally coherent 3D maps with VGGT for memory-efficient Semantic SLAM
- **分类: cs.CV**

- **简介: 该论文针对语义SLAM中内存占用高、时空一致性差的问题，提出基于VGGT的高效框架。通过滑动窗口处理图像流，结合时间戳与实例身份信息，实现低内存消耗的实时3D语义地图构建，提升环境变化检测与上下文理解能力，适用于辅助导航等实际场景。**

- **链接: [https://arxiv.org/pdf/2511.16282v1](https://arxiv.org/pdf/2511.16282v1)**

> **作者:** Gergely Dinya; Péter Halász; András Lőrincz; Kristóf Karacs; Anna Gelencsér-Horváth
>
> **摘要:** We present a fast, spatio-temporal scene understanding framework based on Vision Gated Generative Transformers (VGGT). The proposed pipeline is designed to enable efficient, close to real-time performance, supporting applications including assistive navigation. To achieve continuous updates of the 3D scene representation, we process the image flow with a sliding window, aligning submaps, thereby overcoming VGGT's high memory demands. We exploit the VGGT tracking head to aggregate 2D semantic instance masks into 3D objects. To allow for temporal consistency and richer contextual reasoning the system stores timestamps and instance-level identities, thereby enabling the detection of changes in the environment. We evaluate the approach on well-known benchmarks and custom datasets specifically designed for assistive navigation scenarios. The results demonstrate the applicability of the framework to real-world scenarios.
>
---
#### [new 043] FastSurfer-CC: A robust, accurate, and comprehensive framework for corpus callosum morphometry
- **分类: cs.CV**

- **简介: 该论文提出FastSurfer-CC，一个全自动、高精度的胼胝体形态测量框架。针对现有工具缺乏全面自动化分析流程的问题，该方法实现胼胝体与穹窿分割、解剖定位、厚度分析及八项形态指标提取，显著提升准确性并揭示疾病差异。**

- **链接: [https://arxiv.org/pdf/2511.16471v1](https://arxiv.org/pdf/2511.16471v1)**

> **作者:** Clemens Pollak; Kersten Diers; Santiago Estrada; David Kügler; Martin Reuter
>
> **摘要:** The corpus callosum, the largest commissural structure in the human brain, is a central focus in research on aging and neurological diseases. It is also a critical target for interventions such as deep brain stimulation and serves as an important biomarker in clinical trials, including those investigating remyelination therapies. Despite extensive research on corpus callosum segmentation, few publicly available tools provide a comprehensive and automated analysis pipeline. To address this gap, we present FastSurfer-CC, an efficient and fully automated framework for corpus callosum morphometry. FastSurfer-CC automatically identifies mid-sagittal slices, segments the corpus callosum and fornix, localizes the anterior and posterior commissures to standardize head positioning, generates thickness profiles and subdivisions, and extracts eight shape metrics for statistical analysis. We demonstrate that FastSurfer-CC outperforms existing specialized tools across the individual tasks. Moreover, our method reveals statistically significant differences between Huntington's disease patients and healthy controls that are not detected by the current state-of-the-art.
>
---
#### [new 044] Video-as-Answer: Predict and Generate Next Video Event with Joint-GRPO
- **分类: cs.CV**

- **简介: 该论文提出视频作为答案的下一事件预测任务（VNEP），旨在用视频生成替代文本回答，解决传统方法难以表达复杂动作的问题。提出VANS模型与联合强化学习算法Joint-GRPO，协同视觉语言模型与视频扩散模型，实现精准且一致的视频生成。构建了VANS-Data-100K数据集，实验证明其在预测与生成上均达到先进水平。**

- **链接: [https://arxiv.org/pdf/2511.16669v1](https://arxiv.org/pdf/2511.16669v1)**

> **作者:** Junhao Cheng; Liang Hou; Xin Tao; Jing Liao
>
> **备注:** Project page: https://video-as-answer.github.io/
>
> **摘要:** While language models have become impactful in many real-world applications, video generation remains largely confined to entertainment. Motivated by video's inherent capacity to demonstrate physical-world information that is difficult to convey through language alone (e.g., imagine teaching someone to tie a tie using only text), we identify an underutilized opportunity to extend video as a new answer modality for Next-Event Prediction (NEP), formalized as Video-Next-Event Prediction (VNEP). While the established NEP task takes a video with a procedural or predictive question as input to predict the next event in text, VNEP requires dynamic video responses. This shift from telling to showing unlocks more intuitive and customized answers for procedural learning and creative exploration. However, this task remains challenging for existing models, as it demands an understanding of multimodal input, instruction-conditioned reasoning, and the generation of video with visual and semantic consistency. To address this, we introduce VANS, a model that leverages reinforcement learning to align a Vision-Language Model (VLM) with a Video Diffusion Model (VDM) for VNEP. The core of VANS is our proposed Joint-GRPO that orchestrates the VLM and VDM to function as a unit. Driven by a shared reward on their respective output, it optimizes the VLM to produce captions that are both accurate and friendly to visualize, while guiding the VDM to generate videos that are faithful to these captions and the input visual context. To enable this learning, we craft VANS-Data-100K, a dedicated dataset for the VNEP task. Experiments on procedural and predictive benchmarks demonstrate that VANS achieves state-of-the-art performance in both video event prediction and visualization. Codes are released in https://github.com/KlingTeam/VANS.
>
---
#### [new 045] Lite Any Stereo: Efficient Zero-Shot Stereo Matching
- **分类: cs.CV**

- **简介: 该论文提出Lite Any Stereo，解决高效立体匹配中零样本泛化能力弱的问题。通过紧凑骨干网络与混合代价聚合模块，结合三阶段训练策略，实现超轻量模型在真实场景下卓越的零样本性能，精度媲美甚至超越复杂模型，计算成本不足1%。**

- **链接: [https://arxiv.org/pdf/2511.16555v1](https://arxiv.org/pdf/2511.16555v1)**

> **作者:** Junpeng Jing; Weixun Luo; Ye Mao; Krystian Mikolajczyk
>
> **摘要:** Recent advances in stereo matching have focused on accuracy, often at the cost of significantly increased model size. Traditionally, the community has regarded efficient models as incapable of zero-shot ability due to their limited capacity. In this paper, we introduce Lite Any Stereo, a stereo depth estimation framework that achieves strong zero-shot generalization while remaining highly efficient. To this end, we design a compact yet expressive backbone to ensure scalability, along with a carefully crafted hybrid cost aggregation module. We further propose a three-stage training strategy on million-scale data to effectively bridge the sim-to-real gap. Together, these components demonstrate that an ultra-light model can deliver strong generalization, ranking 1st across four widely used real-world benchmarks. Remarkably, our model attains accuracy comparable to or exceeding state-of-the-art non-prior-based accurate methods while requiring less than 1% computational cost, setting a new standard for efficient stereo matching.
>
---
#### [new 046] A Spatial Semantics and Continuity Perception Attention for Remote Sensing Water Body Change Detection
- **分类: cs.CV**

- **简介: 该论文针对高分辨率遥感水体变化检测（WBCD）中数据稀缺与深度特征语义结构利用不足的问题，提出HSRW-CD高分辨率数据集及SSCP注意力模块。SSCP融合多语义空间、结构感知与通道自注意机制，提升水体定位与变化识别精度，可无缝集成至现有模型，显著增强检测性能。**

- **链接: [https://arxiv.org/pdf/2511.16143v1](https://arxiv.org/pdf/2511.16143v1)**

> **作者:** Quanqing Ma; Jiaen Chen; Peng Wang; Yao Zheng; Qingzhan Zhao; Yuchen Zheng
>
> **摘要:** Remote sensing Water Body Change Detection (WBCD) aims to detect water body surface changes from bi-temporal images of the same geographic area. Recently, the scarcity of high spatial resolution datasets for WBCD restricts its application in urban and rural regions, which require more accurate positioning. Meanwhile, previous deep learning-based methods fail to comprehensively exploit the spatial semantic and structural information in deep features in the change detection networks. To resolve these concerns, we first propose a new dataset, HSRW-CD, with a spatial resolution higher than 3 meters for WBCD. Specifically, it contains a large number of image pairs, widely covering various water body types. Besides, a Spatial Semantics and Continuity Perception (SSCP) attention module is designed to fully leverage both the spatial semantics and structure of deep features in the WBCD networks, significantly improving the discrimination capability for water body. The proposed SSCP has three components: the Multi-Semantic spatial Attention (MSA), the Structural Relation-aware Global Attention (SRGA), and the Channel-wise Self-Attention (CSA). The MSA enhances the spatial semantics of water body features and provides precise spatial semantic priors for the CSA. Then, the SRGA further extracts spatial structure to learn the spatial continuity of the water body. Finally, the CSA utilizes the spatial semantic and structural priors from the MSA and SRGA to compute the similarity across channels. Specifically designed as a plug-and-play module for water body deep features, the proposed SSCP allows integration into existing WBCD models. Numerous experiments conducted on the proposed HSRW-CD and Water-CD datasets validate the effectiveness and generalization of the SSCP. The code of this work and the HSRW-CD dataset will be accessed at https://github.com/QingMa1/SSCP.
>
---
#### [new 047] Boosting Medical Visual Understanding From Multi-Granular Language Learning
- **分类: cs.CV**

- **简介: 该论文针对医疗图像理解中多标签、跨粒度标注的挑战，提出多粒度语言学习（MGLL）框架。通过整合多粒度文本描述与软标签监督，增强视觉-语言对齐，提升模型在复杂医学场景下的理解能力。**

- **链接: [https://arxiv.org/pdf/2511.15943v1](https://arxiv.org/pdf/2511.15943v1)**

> **作者:** Zihan Li; Yiqing Wang; Sina Farsiu; Paul Kinahan
>
> **备注:** Preprint. 40 pages
>
> **摘要:** Recent advances in image-text pretraining have significantly enhanced visual understanding by aligning visual and textual representations. Contrastive Language-Image Pretraining (CLIP) has played a pivotal role in multimodal learning. However, its focus on single-label, single-granularity alignment limits its effectiveness in complex domains such as medical imaging, where images often correspond to multiple high-level labels (e.g., disease categories) across different annotation granularities (e.g., diagnostic description, clinical explanation). To address this, we propose Multi-Granular Language Learning (MGLL), a contrastive learning framework designed to improve both multi-label and cross-granularity alignment. MGLL leverages structured multi-label supervision, integrates textual descriptions across granularities, and introduces soft-label supervision with point-wise constraints to enhance alignment. MGLL employs smooth Kullback-Leibler (KL) divergence to ensure cross-granularity consistency while maintaining computational efficiency as a plug-and-play module for vision-language models. Pretrained on our constructed large-scale multi-granular datasets and evaluated across multiple datasets, MGLL outperforms other state-of-the-art methods in downstream tasks. The code is available at \href{https://github.com/HUANGLIZI/MGLL}{https://github.com/HUANGLIZI/MGLL}.
>
---
#### [new 048] YOWO: You Only Walk Once to Jointly Map An Indoor Scene and Register Ceiling-mounted Cameras
- **分类: cs.CV**

- **简介: 该论文提出YOWO框架，解决室内场景联合建图与吊装摄像头（CMC）注册难题。通过移动设备采集多视角数据，同步获取自车轨迹与CMC视频，利用时间对齐实现CMC相对位姿与全局场景的联合优化，实现高效、精准的协同建图与摄像头注册。**

- **链接: [https://arxiv.org/pdf/2511.16521v1](https://arxiv.org/pdf/2511.16521v1)**

> **作者:** Fan Yang; Sosuke Yamao; Ikuo Kusajima; Atsunori Moteki; Shoichi Masui; Shan Jiang
>
> **摘要:** Using ceiling-mounted cameras (CMCs) for indoor visual capturing opens up a wide range of applications. However, registering CMCs to the target scene layout presents a challenging task. While manual registration with specialized tools is inefficient and costly, automatic registration with visual localization may yield poor results when visual ambiguity exists. To alleviate these issues, we propose a novel solution for jointly mapping an indoor scene and registering CMCs to the scene layout. Our approach involves equipping a mobile agent with a head-mounted RGB-D camera to traverse the entire scene once and synchronize CMCs to capture this mobile agent. The egocentric videos generate world-coordinate agent trajectories and the scene layout, while the videos of CMCs provide pseudo-scale agent trajectories and CMC relative poses. By correlating all the trajectories with their corresponding timestamps, the CMC relative poses can be aligned to the world-coordinate scene layout. Based on this initialization, a factor graph is customized to enable the joint optimization of ego-camera poses, scene layout, and CMC poses. We also develop a new dataset, setting the first benchmark for collaborative scene mapping and CMC registration (https://sites.google.com/view/yowo/home). Experimental results indicate that our method not only effectively accomplishes two tasks within a unified framework, but also jointly enhances their performance. We thus provide a reliable tool to facilitate downstream position-aware applications.
>
---
#### [new 049] Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling
- **分类: cs.CV**

- **简介: 该论文针对视觉基础模型因下采样导致的像素级应用受限问题，提出无需训练的轻量级测试时优化框架Upsample Anything。通过学习各向异性高斯核，实现跨架构、跨模态的边缘感知特征上采样，有效恢复高分辨率输出，显著提升语义分割、深度估计等任务性能。**

- **链接: [https://arxiv.org/pdf/2511.16301v1](https://arxiv.org/pdf/2511.16301v1)**

> **作者:** Minseok Seo; Mark Hamilton; Changick Kim
>
> **备注:** 15 pages, 12 figures
>
> **摘要:** We present \textbf{Upsample Anything}, a lightweight test-time optimization (TTO) framework that restores low-resolution features to high-resolution, pixel-wise outputs without any training. Although Vision Foundation Models demonstrate strong generalization across diverse downstream tasks, their representations are typically downsampled by 14x/16x (e.g., ViT), which limits their direct use in pixel-level applications. Existing feature upsampling approaches depend on dataset-specific retraining or heavy implicit optimization, restricting scalability and generalization. Upsample Anything addresses these issues through a simple per-image optimization that learns an anisotropic Gaussian kernel combining spatial and range cues, effectively bridging Gaussian Splatting and Joint Bilateral Upsampling. The learned kernel acts as a universal, edge-aware operator that transfers seamlessly across architectures and modalities, enabling precise high-resolution reconstruction of features, depth, or probability maps. It runs in only $\approx0.419 \text{s}$ per 224x224 image and achieves state-of-the-art performance on semantic segmentation, depth estimation, and both depth and probability map upsampling.
>
---
#### [new 050] Domain-Shared Learning and Gradual Alignment for Unsupervised Domain Adaptation Visible-Infrared Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对真实场景下可见光-红外行人重识别（VI-ReID）的泛化难题，提出无监督域自适应方法UDA-VI-ReID。针对域间与域内模态差异，设计两阶段模型DSLGA：先通过共享学习缓解域间差异，再通过渐进对齐解决模态内差异，显著提升跨域性能。**

- **链接: [https://arxiv.org/pdf/2511.16184v1](https://arxiv.org/pdf/2511.16184v1)**

> **作者:** Nianchang Huang; Yi Xu; Ruida Xi; Ruida Xi; Qiang Zhang
>
> **摘要:** Recently, Visible-Infrared person Re-Identification (VI-ReID) has achieved remarkable performance on public datasets. However, due to the discrepancies between public datasets and real-world data, most existing VI-ReID algorithms struggle in real-life applications. To address this, we take the initiative to investigate Unsupervised Domain Adaptation Visible-Infrared person Re-Identification (UDA-VI-ReID), aiming to transfer the knowledge learned from the public data to real-world data without compromising accuracy and requiring the annotation of new samples. Specifically, we first analyze two basic challenges in UDA-VI-ReID, i.e., inter-domain modality discrepancies and intra-domain modality discrepancies. Then, we design a novel two-stage model, i.e., Domain-Shared Learning and Gradual Alignment (DSLGA), to handle these discrepancies. In the first pre-training stage, DSLGA introduces a Domain-Shared Learning Strategy (DSLS) to mitigate ineffective pre-training caused by inter-domain modality discrepancies via exploiting shared information between the source and target domains. While, in the second fine-tuning stage, DSLGA designs a Gradual Alignment Strategy (GAS) to handle the cross-modality alignment challenges between visible and infrared data caused by the large intra-domain modality discrepancies through a cluster-to-holistic alignment way. Finally, a new UDA-VI-ReID testing method i.e., CMDA-XD, is constructed for training and testing different UDA-VI-ReID models. A large amount of experiments demonstrate that our method significantly outperforms existing domain adaptation methods for VI-ReID and even some supervised methods under various settings.
>
---
#### [new 051] Enhancing Multi-Camera Gymnast Tracking Through Domain Knowledge Integration
- **分类: cs.CV**

- **简介: 该论文针对多摄像头体操运动员跟踪难题，提出融合体操领域知识的追踪方法。针对场馆摄像机数量有限及视图遮挡导致检测不全的问题，利用体操动作中重心常在垂直平面内的先验知识，通过射线-平面相交生成共面3D轨迹候选，结合级联数据关联策略提升跟踪鲁棒性，在世界体操锦标赛中成功应用。**

- **链接: [https://arxiv.org/pdf/2511.16532v1](https://arxiv.org/pdf/2511.16532v1)**

> **作者:** Fan Yang; Shigeyuki Odashima; Shoichi Masui; Ikuo Kusajima; Sosuke Yamao; Shan Jiang
>
> **摘要:** We present a robust multi-camera gymnast tracking, which has been applied at international gymnastics championships for gymnastics judging. Despite considerable progress in multi-camera tracking algorithms, tracking gymnasts presents unique challenges: (i) due to space restrictions, only a limited number of cameras can be installed in the gymnastics stadium; and (ii) due to variations in lighting, background, uniforms, and occlusions, multi-camera gymnast detection may fail in certain views and only provide valid detections from two opposing views. These factors complicate the accurate determination of a gymnast's 3D trajectory using conventional multi-camera triangulation. To alleviate this issue, we incorporate gymnastics domain knowledge into our tracking solution. Given that a gymnast's 3D center typically lies within a predefined vertical plane during \revised{much of their} performance, we can apply a ray-plane intersection to generate coplanar 3D trajectory candidates for opposing-view detections. More specifically, we propose a novel cascaded data association (DA) paradigm that employs triangulation to generate 3D trajectory candidates when cross-view detections are sufficient, and resort to the ray-plane intersection when they are insufficient. Consequently, coplanar candidates are used to compensate for uncertain trajectories, thereby minimizing tracking failures. The robustness of our method is validated through extensive experimentation, demonstrating its superiority over existing methods in challenging scenarios. Furthermore, our gymnastics judging system, equipped with this tracking method, has been successfully applied to recent Gymnastics World Championships, earning significant recognition from the International Gymnastics Federation.
>
---
#### [new 052] ChangeDINO: DINOv3-Driven Building Change Detection in Optical Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文针对光学遥感影像中的建筑变化检测任务，解决现有方法依赖标注、忽视非变化区域语义信息的问题。提出ChangeDINO，一个基于DINOv3的端到端多尺度孪生框架，融合语义与上下文特征，利用空间谱差异变压器和可学习形态模块提升检测精度与边界清晰度。**

- **链接: [https://arxiv.org/pdf/2511.16322v1](https://arxiv.org/pdf/2511.16322v1)**

> **作者:** Ching-Heng Cheng; Chih-Chung Hsu
>
> **摘要:** Remote sensing change detection (RSCD) aims to identify surface changes from co-registered bi-temporal images. However, many deep learning-based RSCD methods rely solely on change-map annotations and underuse the semantic information in non-changing regions, which limits robustness under illumination variation, off-nadir views, and scarce labels. This article introduces ChangeDINO, an end-to-end multiscale Siamese framework for optical building change detection. The model fuses a lightweight backbone stream with features transferred from a frozen DINOv3, yielding semantic- and context-rich pyramids even on small datasets. A spatial-spectral differential transformer decoder then exploits multi-scale absolute differences as change priors to highlight true building changes and suppress irrelevant responses. Finally, a learnable morphology module refines the upsampled logits to recover clean boundaries. Experiments on four public benchmarks show that ChangeDINO consistently outperforms recent state-of-the-art methods in IoU and F1, and ablation studies confirm the effectiveness of each component. The source code is available at https://github.com/chingheng0808/ChangeDINO.
>
---
#### [new 053] TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming
- **分类: cs.CV**

- **简介: 该论文针对3D高斯扩散模型推理效率低的问题，提出TRIM方法。通过时空剪枝与实例掩码去噪，实现轨迹缩减和冗余背景去除，显著提升生成速度与可扩展性，同时保持高质量输出。**

- **链接: [https://arxiv.org/pdf/2511.16642v1](https://arxiv.org/pdf/2511.16642v1)**

> **作者:** Zeyuan Yin; Xiaoming Liu
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recent advances in 3D Gaussian diffusion models suffer from time-intensive denoising and post-denoising processing due to the massive number of Gaussian primitives, resulting in slow generation and limited scalability along sampling trajectories. To improve the efficiency of 3D diffusion models, we propose $\textbf{TRIM}$ ($\textbf{T}$rajectory $\textbf{R}$eduction and $\textbf{I}$nstance $\textbf{M}$ask denoising), a post-training approach that incorporates both temporal and spatial trimming strategies, to accelerate inference without compromising output quality while supporting the inference-time scaling for Gaussian diffusion models. Instead of scaling denoising trajectories in a costly end-to-end manner, we develop a lightweight selector model to evaluate latent Gaussian primitives derived from multiple sampled noises, enabling early trajectory reduction by selecting candidates with high-quality potential. Furthermore, we introduce instance mask denoising to prune learnable Gaussian primitives by filtering out redundant background regions, reducing inference computation at each denoising step. Extensive experiments and analysis demonstrate that TRIM significantly improves both the efficiency and quality of 3D generation. Source code is available at $\href{https://github.com/zeyuanyin/TRIM}{link}$.
>
---
#### [new 054] Explainable AI for Diabetic Retinopathy Detection Using Deep Learning with Attention Mechanisms and Fuzzy Logic-Based Interpretability
- **分类: cs.CV**

- **简介: 该论文针对精准农业中的杂草检测任务，提出一种融合CNN、ViT、GNN的混合深度学习框架，结合GAN数据增强与自监督对比预训练，提升模型在复杂田间条件下的鲁棒性与泛化能力。实验表明模型在多基准数据集上达到99.33%的高精度，具备局部、全局及关系特征建模能力，并支持边缘设备实时部署，实现可解释、可持续的智能除草。**

- **链接: [https://arxiv.org/pdf/2511.16294v1](https://arxiv.org/pdf/2511.16294v1)**

> **作者:** Abishek Karthik; Pandiyaraju V; Sreya Mynampati
>
> **摘要:** The task of weed detection is an essential element of precision agriculture since accurate species identification allows a farmer to selectively apply herbicides and fits into sustainable agriculture crop management. This paper proposes a hybrid deep learning framework recipe for weed detection that utilizes Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and Graph Neural Networks (GNNs) to build robustness to multiple field conditions. A Generative Adversarial Network (GAN)-based augmentation method was imposed to balance class distributions and better generalize the model. Further, a self-supervised contrastive pre-training method helps to learn more features from limited annotated data. Experimental results yield superior results with 99.33% accuracy, precision, recall, and F1-score on multi-benchmark datasets. The proposed model architecture enables local, global, and relational feature representations and offers high interpretability and adaptability. Practically, the framework allows real-time, efficient deployment of edge devices for automated weed detecting, reducing over-reliance on herbicides and providing scalable, sustainable precision-farming options.
>
---
#### [new 055] EvoVLA: Self-Evolving Vision-Language-Action Model
- **分类: cs.CV**

- **简介: 该论文针对长时程机器人操作中VLA模型的阶段幻觉问题，提出EvoVLA框架。通过阶段对齐奖励、基于位姿的探索和长时记忆机制，提升任务完成真实性和样本效率，显著降低幻觉率并实现高效仿真到现实的迁移。**

- **链接: [https://arxiv.org/pdf/2511.16166v1](https://arxiv.org/pdf/2511.16166v1)**

> **作者:** Zeting Liu; Zida Yang; Zeyu Zhang; Hao Tang
>
> **摘要:** Long-horizon robotic manipulation remains challenging for Vision-Language-Action (VLA) models despite recent progress in zero-shot generalization and simulation-to-real-world transfer. Current VLA models suffer from stage hallucination, where agents exploit coarse evaluation signals to shortcut multi-step tasks, reporting high progress without truly completing them. We present EvoVLA, a self-supervised VLA framework that addresses this issue through three complementary components: Stage-Aligned Reward (SAR), which uses triplet contrastive learning with Gemini-generated hard negatives to prevent visual shortcuts; Pose-Based Object Exploration (POE), which grounds curiosity in relative object-gripper pose instead of raw pixels; and Long-Horizon Memory, which uses selective context retention and gated fusion to stabilize intrinsic shaping during extended rollouts. Extensive evaluations on Discoverse-L, a long-horizon manipulation benchmark with three multi-stage tasks, show that EvoVLA improves average task success by 10.2 percentage points over the strongest baseline (OpenVLA-OFT), reaching 69.2 percent. EvoVLA also achieves one-and-a-half times better sample efficiency and reduces stage hallucination from 38.5 percent to 14.8 percent. Real-world deployment on physical robots reaches an average success rate of 54.6 percent across four manipulation tasks, outperforming OpenVLA-OFT by 11 points, demonstrating effective sim-to-real transfer and strong generalization. Code: https://github.com/AIGeeksGroup/EvoVLA. Website: https://aigeeksgroup.github.io/EvoVLA.
>
---
#### [new 056] Optimizing 3D Gaussian Splattering for Mobile GPUs
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对移动GPU上的3D场景重建任务，解决3D高斯溅射（3DGS）在移动端效率低的问题。提出Texture3dgs，通过优化2D纹理缓存的排序算法与数据布局，显著提升计算效率，实现排序和整体重建分别加速4.1×和1.7×，并减少1.6×内存占用。**

- **链接: [https://arxiv.org/pdf/2511.16298v1](https://arxiv.org/pdf/2511.16298v1)**

> **作者:** Md Musfiqur Rahman Sanim; Zhihao Shu; Bahram Afsharmanesh; AmirAli Mirian; Jiexiong Guan; Wei Niu; Bin Ren; Gagan Agrawal
>
> **摘要:** Image-based 3D scene reconstruction, which transforms multi-view images into a structured 3D representation of the surrounding environment, is a common task across many modern applications. 3D Gaussian Splatting (3DGS) is a new paradigm to address this problem and offers considerable efficiency as compared to the previous methods. Motivated by this, and considering various benefits of mobile device deployment (data privacy, operating without internet connectivity, and potentially faster responses), this paper develops Texture3dgs, an optimized mapping of 3DGS for a mobile GPU. A critical challenge in this area turns out to be optimizing for the two-dimensional (2D) texture cache, which needs to be exploited for faster executions on mobile GPUs. As a sorting method dominates the computations in 3DGS on mobile platforms, the core of Texture3dgs is a novel sorting algorithm where the processing, data movement, and placement are highly optimized for 2D memory. The properties of this algorithm are analyzed in view of a cost model for the texture cache. In addition, we accelerate other steps of the 3DGS algorithm through improved variable layout design and other optimizations. End-to-end evaluation shows that Texture3dgs delivers up to 4.1$\times$ and 1.7$\times$ speedup for the sorting and overall 3D scene reconstruction, respectively -- while also reducing memory usage by up to 1.6$\times$ -- demonstrating the effectiveness of our design for efficient mobile 3D scene reconstruction.
>
---
#### [new 057] Physics-Informed Machine Learning for Efficient Sim-to-Real Data Augmentation in Micro-Object Pose Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对微物体姿态估计中真实图像数据稀缺问题，提出一种物理信息驱动的生成模型。通过融合波光学渲染与深度对齐的GAN，高效生成高保真显微图像，显著提升合成数据质量与泛化能力，实现接近真实数据训练效果的实时姿态估计。**

- **链接: [https://arxiv.org/pdf/2511.16494v1](https://arxiv.org/pdf/2511.16494v1)**

> **作者:** Zongcai Tan; Lan Wei; Dandan Zhang
>
> **摘要:** Precise pose estimation of optical microrobots is essential for enabling high-precision object tracking and autonomous biological studies. However, current methods rely heavily on large, high-quality microscope image datasets, which are difficult and costly to acquire due to the complexity of microrobot fabrication and the labour-intensive labelling. Digital twin systems offer a promising path for sim-to-real data augmentation, yet existing techniques struggle to replicate complex optical microscopy phenomena, such as diffraction artifacts and depth-dependent imaging.This work proposes a novel physics-informed deep generative learning framework that, for the first time, integrates wave optics-based physical rendering and depth alignment into a generative adversarial network (GAN), to synthesise high-fidelity microscope images for microrobot pose estimation efficiently. Our method improves the structural similarity index (SSIM) by 35.6% compared to purely AI-driven methods, while maintaining real-time rendering speeds (0.022 s/frame).The pose estimator (CNN backbone) trained on our synthetic data achieves 93.9%/91.9% (pitch/roll) accuracy, just 5.0%/5.4% (pitch/roll) below that of an estimator trained exclusively on real data. Furthermore, our framework generalises to unseen poses, enabling data augmentation and robust pose estimation for novel microrobot configurations without additional training data.
>
---
#### [new 058] LEGO-SLAM: Language-Embedded Gaussian Optimization SLAM
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LEGO-SLAM，面向3DGS-based SLAM系统，解决现有方法缺乏开放词汇语义理解、内存占用高及适应性差的问题。通过自适应编码器将语言嵌入压缩至16维，实现实时开放词汇建图，支持语言引导删减与回环检测，显著降低冗余点云，提升效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.16144v1](https://arxiv.org/pdf/2511.16144v1)**

> **作者:** Sibaek Lee; Seongbo Ha; Kyeongsu Kang; Joonyeol Choi; Seungjun Tak; Hyeonwoo Yu
>
> **备注:** 18 pages
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) have enabled Simultaneous Localization and Mapping (SLAM) systems to build photorealistic maps. However, these maps lack the open-vocabulary semantic understanding required for advanced robotic interaction. Integrating language features into SLAM remains a significant challenge, as storing high-dimensional features demands excessive memory and rendering overhead, while existing methods with static models lack adaptability for novel environments. To address these limitations, we propose LEGO-SLAM (Language-Embedded Gaussian Optimization SLAM), the first framework to achieve real-time, open-vocabulary mapping within a 3DGS-based SLAM system. At the core of our method is a scene-adaptive encoder-decoder that distills high-dimensional language embeddings into a compact 16-dimensional feature space. This design reduces the memory per Gaussian and accelerates rendering, enabling real-time performance. Unlike static approaches, our encoder adapts online to unseen scenes. These compact features also enable a language-guided pruning strategy that identifies semantic redundancy, reducing the map's Gaussian count by over 60\% while maintaining rendering quality. Furthermore, we introduce a language-based loop detection approach that reuses these mapping features, eliminating the need for a separate detection model. Extensive experiments demonstrate that LEGO-SLAM achieves competitive mapping quality and tracking accuracy, all while providing open-vocabulary capabilities at 15 FPS.
>
---
#### [new 059] Mem-MLP: Real-Time 3D Human Motion Generation from Sparse Inputs
- **分类: cs.CV**

- **简介: 该论文针对AR/VR中全身体感追踪不完整的问题，提出Mem-MLP方法，通过稀疏传感器输入实时生成流畅3D人体运动。利用带记忆块的MLP架构，结合历史数据增强时序一致性，并采用多任务学习提升精度。实验表明其在移动HMD上达72 FPS，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16264v1](https://arxiv.org/pdf/2511.16264v1)**

> **作者:** Sinan Mutlu; Georgios F. Angelis; Savas Ozkan; Paul Wisbey; Anastasios Drosou; Mete Ozay
>
> **摘要:** Realistic and smooth full-body tracking is crucial for immersive AR/VR applications. Existing systems primarily track head and hands via Head Mounted Devices (HMDs) and controllers, making the 3D full-body reconstruction in-complete. One potential approach is to generate the full-body motions from sparse inputs collected from limited sensors using a Neural Network (NN) model. In this paper, we propose a novel method based on a multi-layer perceptron (MLP) backbone that is enhanced with residual connections and a novel NN-component called Memory-Block. In particular, Memory-Block represents missing sensor data with trainable code-vectors, which are combined with the sparse signals from previous time instances to improve the temporal consistency. Furthermore, we formulate our solution as a multi-task learning problem, allowing our MLP-backbone to learn robust representations that boost accuracy. Our experiments show that our method outperforms state-of-the-art baselines by substantially reducing prediction errors. Moreover, it achieves 72 FPS on mobile HMDs that ultimately improves the accuracy-running time tradeoff.
>
---
#### [new 060] WWE-UIE: A Wavelet & White Balance Efficient Network for Underwater Image Enhancement
- **分类: cs.CV**

- **简介: 该论文针对水下图像增强任务，解决颜色失真与低可见性问题。提出WWE-UIE网络，融合自适应白平衡、小波增强块与梯度感知模块，提升恢复质量同时大幅降低计算开销，实现轻量化实时增强。**

- **链接: [https://arxiv.org/pdf/2511.16321v1](https://arxiv.org/pdf/2511.16321v1)**

> **作者:** Ching-Heng Cheng; Jen-Wei Lee; Chia-Ming Lee; Chih-Chung Hsu
>
> **摘要:** Underwater Image Enhancement (UIE) aims to restore visibility and correct color distortions caused by wavelength-dependent absorption and scattering. Recent hybrid approaches, which couple domain priors with modern deep neural architectures, have achieved strong performance but incur high computational cost, limiting their practicality in real-time scenarios. In this work, we propose WWE-UIE, a compact and efficient enhancement network that integrates three interpretable priors. First, adaptive white balance alleviates the strong wavelength-dependent color attenuation, particularly the dominance of blue-green tones. Second, a wavelet-based enhancement block (WEB) performs multi-band decomposition, enabling the network to capture both global structures and fine textures, which are critical for underwater restoration. Third, a gradient-aware module (SGFB) leverages Sobel operators with learnable gating to explicitly preserve edge structures degraded by scattering. Extensive experiments on benchmark datasets demonstrate that WWE-UIE achieves competitive restoration quality with substantially fewer parameters and FLOPs, enabling real-time inference on resource-limited platforms. Ablation studies and visualizations further validate the contribution of each component. The source code is available at https://github.com/chingheng0808/WWE-UIE.
>
---
#### [new 061] Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Mantis，一种新型视觉-语言-动作模型，解决现有模型因直接预测高维视觉状态导致的训练成本高与信息瓶颈问题。通过解耦视觉前瞻预测与主干网络，利用元查询与扩散Transformer实现隐式动作学习，提升指令遵循、泛化与推理能力。在LIBERO基准上达96.7%成功率，优于主流模型。**

- **链接: [https://arxiv.org/pdf/2511.16175v1](https://arxiv.org/pdf/2511.16175v1)**

> **作者:** Yi Yang; Xueqi Li; Yiyang Chen; Jin Song; Yihan Wang; Zipeng Xiao; Jiadi Su; You Qiaoben; Pengfei Liu; Zhijie Deng
>
> **摘要:** Recent advances in Vision-Language-Action (VLA) models demonstrate that visual signals can effectively complement sparse action supervisions. However, letting VLA directly predict high-dimensional visual states can distribute model capacity and incur prohibitive training cost, while compressing visual states into more compact supervisory signals inevitably incurs information bottlenecks. Moreover, existing methods often suffer from poor comprehension and reasoning capabilities due to the neglect of language supervision. This paper introduces Mantis, a novel framework featuring a Disentangled Visual Foresight (DVF) to tackle these issues. Specifically, Mantis decouples visual foresight prediction from the backbone with the combination of meta queries and a diffusion Transformer (DiT) head. With the current visual state provided to the DiT via a residual connection, a simple next-state prediction objective enables the meta queries to automatically capture the latent actions that delineate the visual trajectory, and hence boost the learning of explicit actions. The disentanglement reduces the burden of the VLA backbone, enabling it to maintain comprehension and reasoning capabilities through language supervision. Empirically, pretrained on human manipulation videos, robot demonstrations, and image-text pairs, Mantis achieves a 96.7% success rate on LIBERO benchmark after fine-tuning, surpassing powerful baselines while exhibiting high convergence speed. Real-world evaluations show that Mantis outperforms $π_{0.5}$, a leading open-source VLA model, particularly in instruction-following capability, generalization to unseen instructions, and reasoning ability. Code and weights are released to support the open-source community.
>
---
#### [new 062] CRISTAL: Real-time Camera Registration in Static LiDAR Scans using Neural Rendering
- **分类: cs.CV**

- **简介: 该论文提出CRISTAL，一种基于神经渲染的实时相机定位方法，用于在静态彩色LiDAR点云中精确追踪相机。针对传统视觉方法存在的漂移、尺度模糊等问题，通过合成视图与真实图像匹配，利用神经渲染缩小域差距，实现无漂移、带量纲的实时定位，优于现有SLAM系统。**

- **链接: [https://arxiv.org/pdf/2511.16349v1](https://arxiv.org/pdf/2511.16349v1)**

> **作者:** Joni Vanherck; Steven Moonen; Brent Zoomers; Kobe Werner; Jeroen Put; Lode Jorissen; Nick Michiels
>
> **摘要:** Accurate camera localization is crucial for robotics and Extended Reality (XR), enabling reliable navigation and alignment of virtual and real content. Existing visual methods often suffer from drift, scale ambiguity, and depend on fiducials or loop closure. This work introduces a real-time method for localizing a camera within a pre-captured, highly accurate colored LiDAR point cloud. By rendering synthetic views from this cloud, 2D-3D correspondences are established between live frames and the point cloud. A neural rendering technique narrows the domain gap between synthetic and real images, reducing occlusion and background artifacts to improve feature matching. The result is drift-free camera tracking with correct metric scale in the global LiDAR coordinate system. Two real-time variants are presented: Online Render and Match, and Prebuild and Localize. We demonstrate improved results on the ScanNet++ dataset and outperform existing SLAM pipelines.
>
---
#### [new 063] InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer
- **分类: cs.CV**

- **简介: 该论文针对CLIP在开放词汇语义分割中微调时易过拟合、破坏预训练模态对齐的问题，提出InfoCLIP。通过信息论视角，利用互信息优化，压缩并迁移预训练模型的像素-文本对齐知识，增强细粒度语义表征，提升分割性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.15967v1](https://arxiv.org/pdf/2511.15967v1)**

> **作者:** Muyao Yuan; Yuanhong Zhang; Weizhan Zhang; Lan Ma; Yuan Gao; Jiangyong Ying; Yudeng Xin
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Recently, the strong generalization ability of CLIP has facilitated open-vocabulary semantic segmentation, which labels pixels using arbitrary text. However, existing methods that fine-tune CLIP for segmentation on limited seen categories often lead to overfitting and degrade the pretrained vision-language alignment. To stabilize modality alignment during fine-tuning, we propose InfoCLIP, which leverages an information-theoretic perspective to transfer alignment knowledge from pretrained CLIP to the segmentation task. Specifically, this transfer is guided by two novel objectives grounded in mutual information. First, we compress the pixel-text modality alignment from pretrained CLIP to reduce noise arising from its coarse-grained local semantic representations learned under image-text supervision. Second, we maximize the mutual information between the alignment knowledge of pretrained CLIP and the fine-tuned model to transfer compact local semantic relations suited for the segmentation task. Extensive evaluations across various benchmarks validate the effectiveness of InfoCLIP in enhancing CLIP fine-tuning for open-vocabulary semantic segmentation, demonstrating its adaptability and superiority in asymmetric transfer.
>
---
#### [new 064] Real-Time 3D Object Detection with Inference-Aligned Learning
- **分类: cs.CV**

- **简介: 该论文针对实时3D目标检测中训练与推理不一致的问题，提出SR3D框架。通过空间优先的最优传输分配和排名感知自蒸馏，增强模型对空间可靠性和预测排序的感知，有效缩小训练-推理差距，在保持实时性的同时提升检测精度。**

- **链接: [https://arxiv.org/pdf/2511.16140v1](https://arxiv.org/pdf/2511.16140v1)**

> **作者:** Chenyu Zhao; Xianwei Zheng; Zimin Xia; Linwei Yue; Nan Xue
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Real-time 3D object detection from point clouds is essential for dynamic scene understanding in applications such as augmented reality, robotics and navigation. We introduce a novel Spatial-prioritized and Rank-aware 3D object detection (SR3D) framework for indoor point clouds, to bridge the gap between how detectors are trained and how they are evaluated. This gap stems from the lack of spatial reliability and ranking awareness during training, which conflicts with the ranking-based prediction selection used as inference. Such a training-inference gap hampers the model's ability to learn representations aligned with inference-time behavior. To address the limitation, SR3D consists of two components tailored to the spatial nature of point clouds during training: a novel spatial-prioritized optimal transport assignment that dynamically emphasizes well-located and spatially reliable samples, and a rank-aware adaptive self-distillation scheme that adaptively injects ranking perception via a self-distillation paradigm. Extensive experiments on ScanNet V2 and SUN RGB-D show that SR3D effectively bridges the training-inference gap and significantly outperforms prior methods in accuracy while maintaining real-time speed.
>
---
#### [new 065] Investigating Optical Flow Computation: From Local Methods to a Multiresolution Horn-Schunck Implementation with Bilinear Interpolation
- **分类: cs.CV**

- **简介: 该论文研究光学流计算任务，旨在提升运动估计的准确性与收敛性。针对局部（如Lucas-Kanade）与全局（如Horn-Schunck）方法的局限，提出多分辨率Horn-Schunck算法，结合双线性插值与延拓策略，有效改善复杂图像条件下的运动估计性能。**

- **链接: [https://arxiv.org/pdf/2511.16535v1](https://arxiv.org/pdf/2511.16535v1)**

> **作者:** Haytham Ziani
>
> **摘要:** This paper presents an applied analysis of local and global methods, with a focus on the Horn-Schunck algorithm for optical flow computation. We explore the theoretical and practical aspects of local approaches, such as the Lucas-Kanade method, and global techniques such as Horn-Schunck. Additionally, we implement a multiresolution version of the Horn-Schunck algorithm, using bilinear interpolation and prolongation to improve accuracy and convergence. The study investigates the effectiveness of these combined strategies in estimating motion between frames, particularly under varying image conditions.
>
---
#### [new 066] Beyond Visual Cues: Leveraging General Semantics as Support for Few-Shot Segmentation
- **分类: cs.CV**

- **简介: 该论文研究少样本图像分割任务，针对现有方法依赖支持图像视觉线索导致引导不准确的问题，提出利用语言描述构建无偏元引导。通过多属性增强与跨模态对齐模块，融合大语言模型生成的语义信息，提升对未见类别的分割性能，实现新SOTA。**

- **链接: [https://arxiv.org/pdf/2511.16435v1](https://arxiv.org/pdf/2511.16435v1)**

> **作者:** Jin Wang; Bingfeng Zhang; Jian Pang; Mengyu Liu; Honglong Chen; Weifeng Liu
>
> **摘要:** Few-shot segmentation (FSS) aims to segment novel classes under the guidance of limited support samples by a meta-learning paradigm. Existing methods mainly mine references from support images as meta guidance. However, due to intra-class variations among visual representations, the meta information extracted from support images cannot produce accurate guidance to segment untrained classes. In this paper, we argue that the references from support images may not be essential, the key to the support role is to provide unbiased meta guidance for both trained and untrained classes. We then introduce a Language-Driven Attribute Generalization (LDAG) architecture to utilize inherent target property language descriptions to build robust support strategy. Specifically, to obtain an unbiased support representation, we design a Multi-attribute Enhancement (MaE) module, which produces multiple detailed attribute descriptions of the target class through Large Language Models (LLMs), and then builds refined visual-text prior guidance utilizing multi-modal matching. Meanwhile, due to text-vision modal shift, attribute text struggles to promote visual feature representation, we design a Multi-modal Attribute Alignment (MaA) to achieve cross-modal interaction between attribute texts and visual feature. Experiments show that our proposed method outperforms existing approaches by a clear margin and achieves the new state-of-the art performance. The code will be released.
>
---
#### [new 067] When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉-语言-动作模型在具身环境中的多模态对抗鲁棒性问题。针对现有研究忽视跨模态错位的缺陷，提出VLA-Fool框架，涵盖文本、视觉及跨模态对齐攻击，并设计语义引导的自动提示生成方法。实验表明微小扰动即可导致行为严重偏离，揭示了多模态对齐的脆弱性。**

- **链接: [https://arxiv.org/pdf/2511.16203v1](https://arxiv.org/pdf/2511.16203v1)**

> **作者:** Yuping Yan; Yuhan Xie; Yinxin Zhang; Lingjuan Lyu; Yaochu Jin
>
> **摘要:** Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.
>
---
#### [new 068] V-ReasonBench: Toward Unified Reasoning Benchmark Suite for Video Generation Models
- **分类: cs.CV**

- **简介: 该论文提出V-ReasonBench，一个评估视频生成模型推理能力的统一基准。针对现有方法缺乏系统性评估的问题，构建涵盖结构化、空间、模式和物理推理的多样化任务，实现可复现、可扩展的评测，揭示模型在不同推理维度的表现差异及幻觉行为。**

- **链接: [https://arxiv.org/pdf/2511.16668v1](https://arxiv.org/pdf/2511.16668v1)**

> **作者:** Yang Luo; Xuanlei Zhao; Baijiong Lin; Lingting Zhu; Liyao Tang; Yuqi Liu; Ying-Cong Chen; Shengju Qian; Xin Wang; Yang You
>
> **备注:** Project Page: https://oahzxl.github.io/VReasonBench
>
> **摘要:** Recent progress in generative video models, such as Veo-3, has shown surprising zero-shot reasoning abilities, creating a growing need for systematic and reliable evaluation. We introduce V-ReasonBench, a benchmark designed to assess video reasoning across four key dimensions: structured problem-solving, spatial cognition, pattern-based inference, and physical dynamics. The benchmark is built from both synthetic and real-world image sequences and provides a diverse set of answer-verifiable tasks that are reproducible, scalable, and unambiguous. Evaluations of six state-of-the-art video models reveal clear dimension-wise differences, with strong variation in structured, spatial, pattern-based, and physical reasoning. We further compare video models with strong image models, analyze common hallucination behaviors, and study how video duration affects Chain-of-Frames reasoning. Overall, V-ReasonBench offers a unified and reproducible framework for measuring video reasoning and aims to support the development of models with more reliable, human-aligned reasoning skills.
>
---
#### [new 069] VideoSeg-R1:Reasoning Video Object Segmentation via Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文提出VideoSeg-R1，首个将强化学习用于视频推理分割的方法。针对传统方法依赖监督微调、泛化性差且缺乏显式推理的问题，设计分阶段框架：通过文本引导采样模拟人类注意力，生成空间线索与推理链，并结合SAM2和XMem实现分割与传播。引入难度感知机制自适应控制推理长度，提升效率与精度，在多个基准上达到领先性能。**

- **链接: [https://arxiv.org/pdf/2511.16077v1](https://arxiv.org/pdf/2511.16077v1)**

> **作者:** Zishan Xu; Yifu Guo; Yuquan Lu; Fengyu Yang; Junxin Li
>
> **摘要:** Traditional video reasoning segmentation methods rely on supervised fine-tuning, which limits generalization to out-of-distribution scenarios and lacks explicit reasoning. To address this, we propose \textbf{VideoSeg-R1}, the first framework to introduce reinforcement learning into video reasoning segmentation. It adopts a decoupled architecture that formulates the task as joint referring image segmentation and video mask propagation. It comprises three stages: (1) A hierarchical text-guided frame sampler to emulate human attention; (2) A reasoning model that produces spatial cues along with explicit reasoning chains; and (3) A segmentation-propagation stage using SAM2 and XMem. A task difficulty-aware mechanism adaptively controls reasoning length for better efficiency and accuracy. Extensive evaluations on multiple benchmarks demonstrate that VideoSeg-R1 achieves state-of-the-art performance in complex video reasoning and segmentation tasks. The code will be publicly available at https://github.com/euyis1019/VideoSeg-R1.
>
---
#### [new 070] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation
- **分类: cs.CV**

- **简介: 该论文针对多物体图像生成中9自由度姿态（位置、大小、方向）的精确控制难题，提出SceneDesigner方法。通过引入CNOCS表示与分支网络，结合新数据集ObjectPose9D及两阶段强化学习训练策略，实现高效稳定生成。引入解耦采样与个性化权重，显著提升可控性与生成质量。**

- **链接: [https://arxiv.org/pdf/2511.16666v1](https://arxiv.org/pdf/2511.16666v1)**

> **作者:** Zhenyuan Qin; Xincheng Shuai; Henghui Ding
>
> **备注:** NeurIPS 2025 (Spotlight), Project Page: https://henghuiding.com/SceneDesigner/
>
> **摘要:** Controllable image generation has attracted increasing attention in recent years, enabling users to manipulate visual content such as identity and style. However, achieving simultaneous control over the 9D poses (location, size, and orientation) of multiple objects remains an open challenge. Despite recent progress, existing methods often suffer from limited controllability and degraded quality, falling short of comprehensive multi-object 9D pose control. To address these limitations, we propose SceneDesigner, a method for accurate and flexible multi-object 9-DoF pose manipulation. SceneDesigner incorporates a branched network to the pre-trained base model and leverages a new representation, CNOCS map, which encodes 9D pose information from the camera view. This representation exhibits strong geometric interpretation properties, leading to more efficient and stable training. To support training, we construct a new dataset, ObjectPose9D, which aggregates images from diverse sources along with 9D pose annotations. To further address data imbalance issues, particularly performance degradation on low-frequency poses, we introduce a two-stage training strategy with reinforcement learning, where the second stage fine-tunes the model using a reward-based objective on rebalanced data. At inference time, we propose Disentangled Object Sampling, a technique that mitigates insufficient object generation and concept confusion in complex multi-object scenes. Moreover, by integrating user-specific personalization weights, SceneDesigner enables customized pose control for reference subjects. Extensive qualitative and quantitative experiments demonstrate that SceneDesigner significantly outperforms existing approaches in both controllability and quality. Code is publicly available at https://github.com/FudanCVL/SceneDesigner.
>
---
#### [new 071] Sparse Autoencoders are Topic Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文将稀疏自编码器（SAE）视为主题模型，提出SAE-TM框架，通过在嵌入空间建模主题分布，实现可复用的主题原子学习与灵活主题合并。解决了SAE解释性不足的问题，提升了文本与图像数据的主题一致性与多样性，支持跨模态主题分析。**

- **链接: [https://arxiv.org/pdf/2511.16309v1](https://arxiv.org/pdf/2511.16309v1)**

> **作者:** Leander Girrbach; Zeynep Akata
>
> **摘要:** Sparse autoencoders (SAEs) are used to analyze embeddings, but their role and practical value are debated. We propose a new perspective on SAEs by demonstrating that they can be naturally understood as topic models. We extend Latent Dirichlet Allocation to embedding spaces and derive the SAE objective as a maximum a posteriori estimator under this model. This view implies SAE features are thematic components rather than steerable directions. Based on this, we introduce SAE-TM, a topic modeling framework that: (1) trains an SAE to learn reusable topic atoms, (2) interprets them as word distributions on downstream data, and (3) merges them into any number of topics without retraining. SAE-TM yields more coherent topics than strong baselines on text and image datasets while maintaining diversity. Finally, we analyze thematic structure in image datasets and trace topic changes over time in Japanese woodblock prints. Our work positions SAEs as effective tools for large-scale thematic analysis across modalities. Code and data will be released upon publication.
>
---
#### [new 072] SAM 3D: 3Dfy Anything in Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SAM 3D，一种从单张图像重建3D物体的生成模型，解决自然场景中因遮挡和杂乱导致的3D重建难题。通过人机协同标注构建大规模视觉引导数据集，结合合成预训练与真实对齐的多阶段训练，突破3D数据瓶颈，在真实场景下显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16624v1](https://arxiv.org/pdf/2511.16624v1)**

> **作者:** SAM 3D Team; Xingyu Chen; Fu-Jen Chu; Pierre Gleize; Kevin J Liang; Alexander Sax; Hao Tang; Weiyao Wang; Michelle Guo; Thibaut Hardin; Xiang Li; Aohan Lin; Jiawei Liu; Ziqi Ma; Anushka Sagar; Bowen Song; Xiaodong Wang; Jianing Yang; Bowen Zhang; Piotr Dollár; Georgia Gkioxari; Matt Feiszli; Jitendra Malik
>
> **备注:** Website: https://ai.meta.com/sam3d/
>
> **摘要:** We present SAM 3D, a generative model for visually grounded 3D object reconstruction, predicting geometry, texture, and layout from a single image. SAM 3D excels in natural images, where occlusion and scene clutter are common and visual recognition cues from context play a larger role. We achieve this with a human- and model-in-the-loop pipeline for annotating object shape, texture, and pose, providing visually grounded 3D reconstruction data at unprecedented scale. We learn from this data in a modern, multi-stage training framework that combines synthetic pretraining with real-world alignment, breaking the 3D "data barrier". We obtain significant gains over recent work, with at least a 5:1 win rate in human preference tests on real-world objects and scenes. We will release our code and model weights, an online demo, and a new challenging benchmark for in-the-wild 3D object reconstruction.
>
---
#### [new 073] Progressive Supernet Training for Efficient Visual Autoregressive Modeling
- **分类: cs.CV**

- **简介: 该论文针对视觉自回归模型（VAR）生成时内存占用高的问题，提出VARiant方法。通过等距采样构建多尺度子网，共享权重并采用渐进式训练策略，在保持生成质量的同时显著降低内存消耗与推理延迟，实现单模型下灵活的深度调整与高效部署。**

- **链接: [https://arxiv.org/pdf/2511.16546v1](https://arxiv.org/pdf/2511.16546v1)**

> **作者:** Xiaoyue Chen; Yuling Shi; Kaiyuan Li; Huandong Wang; Yong Li; Xiaodong Gu; Xinlei Chen; Mingbao Lin
>
> **备注:** Submitted to CVPR 2025. 10 pages, 7 figures
>
> **摘要:** Visual Auto-Regressive (VAR) models significantly reduce inference steps through the "next-scale" prediction paradigm. However, progressive multi-scale generation incurs substantial memory overhead due to cumulative KV caching, limiting practical deployment. We observe a scale-depth asymmetric dependency in VAR: early scales exhibit extreme sensitivity to network depth, while later scales remain robust to depth reduction. Inspired by this, we propose VARiant: by equidistant sampling, we select multiple subnets ranging from 16 to 2 layers from the original 30-layer VAR-d30 network. Early scales are processed by the full network, while later scales utilize subnet. Subnet and the full network share weights, enabling flexible depth adjustment within a single model. However, weight sharing between subnet and the entire network can lead to optimization conflicts. To address this, we propose a progressive training strategy that breaks through the Pareto frontier of generation quality for both subnets and the full network under fixed-ratio training, achieving joint optimality. Experiments on ImageNet demonstrate that, compared to the pretrained VAR-d30 (FID 1.95), VARiant-d16 and VARiant-d8 achieve nearly equivalent quality (FID 2.05/2.12) while reducing memory consumption by 40-65%. VARiant-d2 achieves 3.5 times speedup and 80% memory reduction at moderate quality cost (FID 2.97). In terms of deployment, VARiant's single-model architecture supports zero-cost runtime depth switching and provides flexible deployment options from high quality to extreme efficiency, catering to diverse application scenarios.
>
---
#### [new 074] Reasoning Guided Embeddings: Leveraging MLLM Reasoning for Improved Multimodal Retrieval
- **分类: cs.CV**

- **简介: 该论文针对多模态检索任务，解决现有嵌入方法忽视多模态大模型推理能力的问题。提出推理引导嵌入（RGE），通过显式引入模型的生成式推理过程并结合对比学习，增强嵌入的上下文条件感知能力，显著提升多模态表示质量，在MMEB上性能提升4.9%。**

- **链接: [https://arxiv.org/pdf/2511.16150v1](https://arxiv.org/pdf/2511.16150v1)**

> **作者:** Chunxu Liu; Jiyuan Yang; Ruopeng Gao; Yuhan Zhu; Feng Zhu; Rui Zhao; Limin Wang
>
> **摘要:** Multimodal embeddings are widely used in downstream tasks such as multimodal retrieval, enabling alignment of interleaved modalities in a shared representation space. While recent studies show that Multimodal Large Language Models (MLLMs) can serve as strong embedding extractors, existing approaches treat embedding extraction as a direct encoding step, overlooking the fact that MLLMs possess the generative capability for reasoning that could be leveraged to enhance representation quality. In this work, we explore how to explicitly incorporate reasoning into the embedding process. To this end, we propose Reasoning Guided Embeddings (RGE), which preserves the generative rationale process of MLLMs and couples it with contrastive training. Our method first enables the model to perform structured rationale generation conditioned on the instruction, and then extracts representations after reasoning has unfolded. This simple design enhances the context-conditional inference signals within the embedding, leading to improved multimodal representation quality. Experiments on the MMEB benchmark show that reasoning-guided conditioning improves multimodal retrieval performance by 4.9% over the non-reasoning baseline, confirming that explicit reasoning can effectively enhance embedding quality.
>
---
#### [new 075] An Image Is Worth Ten Thousand Words: Verbose-Text Induction Attacks on VLMs
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型（VLMs）生成冗长低效文本的问题，提出一种两阶段的隐性对抗攻击方法（VTIA）。通过强化学习搜索恶意提示，并优化图像扰动以匹配提示特征，诱导模型生成超长输出，显著提升攻击效果与可控性。**

- **链接: [https://arxiv.org/pdf/2511.16163v1](https://arxiv.org/pdf/2511.16163v1)**

> **作者:** Zhi Luo; Zenghui Yuan; Wenqi Wei; Daizong Liu; Pan Zhou
>
> **摘要:** With the remarkable success of Vision-Language Models (VLMs) on multimodal tasks, concerns regarding their deployment efficiency have become increasingly prominent. In particular, the number of tokens consumed during the generation process has emerged as a key evaluation metric.Prior studies have shown that specific inputs can induce VLMs to generate lengthy outputs with low information density, which significantly increases energy consumption, latency, and token costs. However, existing methods simply delay the occurrence of the EOS token to implicitly prolong output, and fail to directly maximize the output token length as an explicit optimization objective, lacking stability and controllability.To address these limitations, this paper proposes a novel verbose-text induction attack (VTIA) to inject imperceptible adversarial perturbations into benign images via a two-stage framework, which identifies the most malicious prompt embeddings for optimizing and maximizing the output token of the perturbed images.Specifically, we first perform adversarial prompt search, employing reinforcement learning strategies to automatically identify adversarial prompts capable of inducing the LLM component within VLMs to produce verbose outputs. We then conduct vision-aligned perturbation optimization to craft adversarial examples on input images, maximizing the similarity between the perturbed image's visual embeddings and those of the adversarial prompt, thereby constructing malicious images that trigger verbose text generation. Comprehensive experiments on four popular VLMs demonstrate that our method achieves significant advantages in terms of effectiveness, efficiency, and generalization capability.
>
---
#### [new 076] Dataset Distillation for Pre-Trained Self-Supervised Vision Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究预训练自监督视觉模型下的数据集蒸馏任务，旨在生成少量合成图像，使在线性探测器上训练时能逼近真实数据性能。提出Linear Gradient Matching方法，通过匹配预训练特征提取器产生的梯度来优化合成数据，实现跨模型泛化与高效细粒度分类，提升模型可解释性。**

- **链接: [https://arxiv.org/pdf/2511.16674v1](https://arxiv.org/pdf/2511.16674v1)**

> **作者:** George Cazenavette; Antonio Torralba; Vincent Sitzmann
>
> **备注:** Accepted at NeurIPS 2025. Project page: https://linear-gradient-matching.github.io/ Code: https://github.com/GeorgeCazenavette/linear-gradient-matching
>
> **摘要:** The task of dataset distillation aims to find a small set of synthetic images such that training a model on them reproduces the performance of the same model trained on a much larger dataset of real samples. Existing distillation methods focus on synthesizing datasets that enable training randomly initialized models. In contrast, state-of-the-art vision approaches are increasingly building on large, pre-trained self-supervised models rather than training from scratch. In this paper, we investigate the problem of distilling datasets that enable us to optimally train linear probes on top of such large, pre-trained vision models. We introduce a method of dataset distillation for this task called Linear Gradient Matching that optimizes the synthetic images such that, when passed through a pre-trained feature extractor, they induce gradients in the linear classifier similar to those produced by the real data. Our method yields synthetic data that outperform all real-image baselines and, remarkably, generalize across pre-trained vision models, enabling us, for instance, to train a linear CLIP probe that performs competitively using a dataset distilled via a DINO backbone. Further, we show that our distilled datasets are exceptionally effective for fine-grained classification and provide a valuable tool for model interpretability, predicting, among other things, how similar two models' embedding spaces are under the platonic representation hypothesis or whether a model is sensitive to spurious correlations in adversarial datasets.
>
---
#### [new 077] AMS-KV: Adaptive KV Caching in Multi-Scale Visual Autoregressive Transformers
- **分类: cs.CV**

- **简介: 该论文针对多尺度视觉自回归模型中的KV缓存效率问题，提出AMS-KV缓存策略。通过分析尺度间相似性，自适应地缓存关键尺度的KV，显著降低内存占用与计算延迟，提升模型可扩展性，解决了高分辨率图像生成中缓存膨胀导致的性能瓶颈。**

- **链接: [https://arxiv.org/pdf/2511.16047v1](https://arxiv.org/pdf/2511.16047v1)**

> **作者:** Boxun Xu; Yu Wang; Zihu Wang; Peng Li
>
> **摘要:** Visual autoregressive modeling (VAR) via next-scale prediction has emerged as a scalable image generation paradigm. While Key and Value (KV) caching in large language models (LLMs) has been extensively studied, next-scale prediction presents unique challenges, and KV caching design for next-scale based VAR transformers remains largely unexplored. A major bottleneck is the excessive KV memory growth with the increasing number of scales-severely limiting scalability. Our systematic investigation reveals that: (1) Attending to tokens from local scales significantly contributes to generation quality (2) Allocating a small amount of memory for the coarsest scales, termed as condensed scales, stabilizes multi-scale image generation (3) Strong KV similarity across finer scales is predominantly observed in cache-efficient layers, whereas cache-demanding layers exhibit weaker inter-scale similarity. Based on the observations, we introduce AMS-KV, a scale-adaptive KV caching policy for next-scale prediction in VAR models. AMS-KV prioritizes storing KVs from condensed and local scales, preserving the most relevant tokens to maintain generation quality. It further optimizes KV cache utilization and computational efficiency identifying cache-demanding layers through inter-scale similarity analysis. Compared to the vanilla next-scale prediction-based VAR models, AMS-KV reduces KV cache usage by up to 84.83% and self-attention latency by 60.48%. Moreover, when the baseline VAR-d30 model encounters out-of-memory failures at a batch size of 128, AMS-KV enables stable scaling to a batch size of 256 with improved throughput.
>
---
#### [new 078] Fairness in Multi-modal Medical Diagnosis with Demonstration Selection
- **分类: cs.CV; cs.CY; cs.LG**

- **简介: 该论文针对多模态医疗诊断中因数据偏倚导致的公平性问题，提出无需微调的公平感知示范选择方法FADS。通过聚类实现人口统计学平衡且语义相关示范的选取，有效降低性别、种族等偏差，提升模型公平性与准确性，为大规模医疗AI提供高效可扩展的公平性解决方案。**

- **链接: [https://arxiv.org/pdf/2511.15986v1](https://arxiv.org/pdf/2511.15986v1)**

> **作者:** Dawei Li; Zijian Gu; Peng Wang; Chuhan Song; Zhen Tan; Mohan Zhang; Tianlong Chen; Yu Tian; Song Wang
>
> **备注:** 10 pages (including 2 pages of references), 4 figures. This work explores fairness in multi-modal medical image reasoning using in-context learning
>
> **摘要:** Multimodal large language models (MLLMs) have shown strong potential for medical image reasoning, yet fairness across demographic groups remains a major concern. Existing debiasing methods often rely on large labeled datasets or fine-tuning, which are impractical for foundation-scale models. We explore In-Context Learning (ICL) as a lightweight, tuning-free alternative for improving fairness. Through systematic analysis, we find that conventional demonstration selection (DS) strategies fail to ensure fairness due to demographic imbalance in selected exemplars. To address this, we propose Fairness-Aware Demonstration Selection (FADS), which builds demographically balanced and semantically relevant demonstrations via clustering-based sampling. Experiments on multiple medical imaging benchmarks show that FADS consistently reduces gender-, race-, and ethnicity-related disparities while maintaining strong accuracy, offering an efficient and scalable path toward fair medical image reasoning. These results highlight the potential of fairness-aware in-context learning as a scalable and data-efficient solution for equitable medical image reasoning.
>
---
#### [new 079] Adaptive Guided Upsampling for Low-light Image Enhancement
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文针对低光照图像增强任务，解决传统引导上采样因噪声高、亮度低导致特征传递效果差的问题。提出自适应引导上采样（AGU）方法，通过多参数优化学习低光与明亮图像间的特征关联，实现噪声抑制与锐化同步提升，可实时生成高质量图像，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16623v1](https://arxiv.org/pdf/2511.16623v1)**

> **作者:** Angela Vivian Dcosta; Chunbo Song; Rafael Radkowski
>
> **备注:** 18 pages, 12 figures
>
> **摘要:** We introduce Adaptive Guided Upsampling (AGU), an efficient method for upscaling low-light images capable of optimizing multiple image quality characteristics at the same time, such as reducing noise and increasing sharpness. It is based on a guided image method, which transfers image characteristics from a guidance image to the target image. Using state-of-the-art guided methods, low-light images lack sufficient characteristics for this purpose due to their high noise level and low brightness, rendering suboptimal/not significantly improved images in the process. We solve this problem with multi-parameter optimization, learning the association between multiple low-light and bright image characteristics. Our proposed machine learning method learns these characteristics from a few sample images-pairs. AGU can render high-quality images in real time using low-quality, low-resolution input; our experiments demonstrate that it is superior to state-of-the-art methods in the addressed low-light use case.
>
---
#### [new 080] Exploiting Inter-Sample Information for Long-tailed Out-of-Distribution Detection
- **分类: cs.CV**

- **简介: 该论文针对长尾分布下出域检测（OOD）中误报率高、尾部类别识别差的问题，提出基于图结构的跨样本关系建模方法。通过预训练特征初始化图并引入高斯化与图卷积网络优化，提升尾部类别的识别准确率和整体OOD检测性能，在多个基准上超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16015v1](https://arxiv.org/pdf/2511.16015v1)**

> **作者:** Nimeshika Udayangani; Hadi M. Dolatabadi; Sarah Erfani; Christopher Leckie
>
> **摘要:** Detecting out-of-distribution (OOD) data is essential for safe deployment of deep neural networks (DNNs). This problem becomes particularly challenging in the presence of long-tailed in-distribution (ID) datasets, often leading to high false positive rates (FPR) and low tail-class ID classification accuracy. In this paper, we demonstrate that exploiting inter-sample relationships using a graph-based representation can significantly improve OOD detection in long-tailed recognition of vision datasets. To this end, we use the feature space of a pre-trained model to initialize our graph structure. We account for the differences between the activation layer distribution of the pre-training vs. training data, and actively introduce Gaussianization to alleviate any deviations from a standard normal distribution in the activation layers of the pre-trained model. We then refine this initial graph representation using graph convolutional networks (GCNs) to arrive at a feature space suitable for long-tailed OOD detection. This leads us to address the inferior performance observed in ID tail-classes within existing OOD detection methods. Experiments over three benchmarks CIFAR10-LT, CIFAR100-LT, and ImageNet-LT demonstrate that our method outperforms the state-of-the-art approaches by a large margin in terms of FPR and tail-class ID classification accuracy.
>
---
#### [new 081] Clustered Error Correction with Grouped 4D Gaussian Splatting
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对动态场景的4D Gaussian Splatting重建难题，提出基于椭圆误差聚类与分组4DGS的方法。通过分类渲染误差并针对性修正，提升动态区域的密度与一致性，显著改善时序连贯性与视觉质量，实现更精准的动态对象建模。**

- **链接: [https://arxiv.org/pdf/2511.16112v1](https://arxiv.org/pdf/2511.16112v1)**

> **作者:** Taeho Kang; Jaeyeon Park; Kyungjin Lee; Youngki Lee
>
> **备注:** 16 pages, 8 figures, SIGGRAPH Asia Conference Papers 2025
>
> **摘要:** Existing 4D Gaussian Splatting (4DGS) methods struggle to accurately reconstruct dynamic scenes, often failing to resolve ambiguous pixel correspondences and inadequate densification in dynamic regions. We address these issues by introducing a novel method composed of two key components: (1) Elliptical Error Clustering and Error Correcting Splat Addition that pinpoints dynamic areas to improve and initialize fitting splats, and (2) Grouped 4D Gaussian Splatting that improves consistency of mapping between splats and represented dynamic objects. Specifically, we classify rendering errors into missing-color and occlusion types, then apply targeted corrections via backprojection or foreground splitting guided by cross-view color consistency. Evaluations on Neural 3D Video and Technicolor datasets demonstrate that our approach significantly improves temporal consistency and achieves state-of-the-art perceptual rendering quality, improving 0.39dB of PSNR on the Technicolor Light Field dataset. Our visualization shows improved alignment between splats and dynamic objects, and the error correction method's capability to identify errors and properly initialize new splats. Our implementation details and source code are available at https://github.com/tho-kn/cem-4dgs.
>
---
#### [new 082] Can MLLMs Read the Room? A Multimodal Benchmark for Assessing Deception in Multi-Party Social Interactions
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型在社交互动中识别欺骗能力不足的问题，提出MIDA任务与多模态数据集，构建基准评估12个模型。发现主流模型难以基于多模态线索判断真伪。提出SoCoT与DSEM框架，提升模型对社会认知的推理能力，推动更智能的社交理解AI发展。**

- **链接: [https://arxiv.org/pdf/2511.16221v1](https://arxiv.org/pdf/2511.16221v1)**

> **作者:** Caixin Kang; Yifei Huang; Liangyang Ouyang; Mingfang Zhang; Ruicong Liu; Yoichi Sato
>
> **摘要:** Despite their advanced reasoning capabilities, state-of-the-art Multimodal Large Language Models (MLLMs) demonstrably lack a core component of human intelligence: the ability to `read the room' and assess deception in complex social interactions. To rigorously quantify this failure, we introduce a new task, Multimodal Interactive Deception Assessment (MIDA), and present a novel multimodal dataset providing synchronized video and text with verifiable ground-truth labels for every statement. We establish a comprehensive benchmark evaluating 12 state-of-the-art open- and closed-source MLLMs, revealing a significant performance gap: even powerful models like GPT-4o struggle to distinguish truth from falsehood reliably. Our analysis of failure modes indicates that these models fail to effectively ground language in multimodal social cues and lack the ability to model what others know, believe, or intend, highlighting the urgent need for novel approaches to building more perceptive and trustworthy AI systems. To take a step forward, we design a Social Chain-of-Thought (SoCoT) reasoning pipeline and a Dynamic Social Epistemic Memory (DSEM) module. Our framework yields performance improvement on this challenging task, demonstrating a promising new path toward building MLLMs capable of genuine human-like social reasoning.
>
---
#### [new 083] Acquisition Time-Informed Breast Tumor Segmentation from Dynamic Contrast-Enhanced MRI
- **分类: cs.CV**

- **简介: 该论文针对动态对比增强MRI（DCE-MRI）中乳腺肿瘤自动分割难题，提出一种基于图像采集时间的时序信息融合方法。通过FiLM层利用采集时间调制模型特征，提升分割精度与跨数据集泛化能力，有效应对不同扫描协议和个体差异带来的图像变异问题。**

- **链接: [https://arxiv.org/pdf/2511.16498v1](https://arxiv.org/pdf/2511.16498v1)**

> **作者:** Rui Wang; Yuexi Du; John Lewin; R. Todd Constable; Nicha C. Dvornek
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI) plays an important role in breast cancer screening, tumor assessment, and treatment planning and monitoring. The dynamic changes in contrast in different tissues help to highlight the tumor in post-contrast images. However, varying acquisition protocols and individual factors result in large variation in the appearance of tissues, even for images acquired in the same phase (e.g., first post-contrast phase), making automated tumor segmentation challenging. Here, we propose a tumor segmentation method that leverages knowledge of the image acquisition time to modulate model features according to the specific acquisition sequence. We incorporate the acquisition times using feature-wise linear modulation (FiLM) layers, a lightweight method for incorporating temporal information that also allows for capitalizing on the full, variables number of images acquired per imaging study. We trained baseline and different configurations for the time-modulated models with varying backbone architectures on a large public multisite breast DCE-MRI dataset. Evaluation on in-domain images and a public out-of-domain dataset showed that incorporating knowledge of phase acquisition time improved tumor segmentation performance and model generalization.
>
---
#### [new 084] Solving Spatial Supersensing Without Spatial Supersensing
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视频世界模型中的空间超感知任务，批判性评估Cambrian-S提出的两个基准（VSR、VSC）及推理方法。发现其性能依赖数据集捷径而非真实空间理解：无感知基线可近乎完美解决VSR；VSC因“不重复场景”假设被简单扰动击穿。结论：当前基准无法可靠衡量空间超感知，现有方法通过捷径提升性能而非真正具备空间认知。**

- **链接: [https://arxiv.org/pdf/2511.16655v1](https://arxiv.org/pdf/2511.16655v1)**

> **作者:** Vishaal Udandarao; Shyamgopal Karthik; Surabhi S. Nath; Andreas Hochlehnert; Matthias Bethge; Ameya Prabhu
>
> **备注:** Tech Report
>
> **摘要:** Cambrian-S aims to take the first steps towards improving video world models with spatial supersensing by introducing (i) two benchmarks, VSI-Super-Recall (VSR) and VSI-Super-Counting (VSC), and (ii) bespoke predictive sensing inference strategies tailored to each benchmark. In this work, we conduct a critical analysis of Cambrian-S across both these fronts. First, we introduce a simple baseline, NoSense, which discards almost all temporal structure and uses only a bag-of-words SigLIP model, yet near-perfectly solves VSR, achieving 95% accuracy even on 4-hour videos. This shows benchmarks like VSR can be nearly solved without spatial cognition, world modeling or spatial supersensing. Second, we hypothesize that the tailored inference methods proposed by Cambrian-S likely exploit shortcut heuristics in the benchmark. We illustrate this with a simple sanity check on the VSC benchmark, called VSC-Repeat: We concatenate each video with itself 1-5 times, which does not change the number of unique objects. However, this simple perturbation entirely collapses the mean relative accuracy of Cambrian-S from 42% to 0%. A system that performs spatial supersensing and integrates information across experiences should recognize views of the same scene and keep object-count predictions unchanged; instead, Cambrian-S inference algorithm relies largely on a shortcut in the VSC benchmark that rooms are never revisited. Taken together, our findings suggest that (i) current VSI-Super benchmarks do not yet reliably measure spatial supersensing, and (ii) predictive-sensing inference recipes used by Cambrian-S improve performance by inadvertently exploiting shortcuts rather than from robust spatial supersensing. We include the response from the Cambrian-S authors (in Appendix A) to provide a balanced perspective alongside our claims. We release our code at: https://github.com/bethgelab/supersanity
>
---
#### [new 085] LiSTAR: Ray-Centric World Models for 4D LiDAR Sequences in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出LiSTAR，一种基于射线中心的4D LiDAR生成模型，旨在解决自动驾驶中高保真、可控4D点云合成难题。针对传感器球面几何与时空稀疏性，提出HCS表示与射线中心注意力机制，实现高效动态建模与高质量生成，显著提升重建、预测与条件生成性能。**

- **链接: [https://arxiv.org/pdf/2511.16049v1](https://arxiv.org/pdf/2511.16049v1)**

> **作者:** Pei Liu; Songtao Wang; Lang Zhang; Xingyue Peng; Yuandong Lyu; Jiaxin Deng; Songxin Lu; Weiliang Ma; Xueyang Zhang; Yifei Zhan; XianPeng Lang; Jun Ma
>
> **摘要:** Synthesizing high-fidelity and controllable 4D LiDAR data is crucial for creating scalable simulation environments for autonomous driving. This task is inherently challenging due to the sensor's unique spherical geometry, the temporal sparsity of point clouds, and the complexity of dynamic scenes. To address these challenges, we present LiSTAR, a novel generative world model that operates directly on the sensor's native geometry. LiSTAR introduces a Hybrid-Cylindrical-Spherical (HCS) representation to preserve data fidelity by mitigating quantization artifacts common in Cartesian grids. To capture complex dynamics from sparse temporal data, it utilizes a Spatio-Temporal Attention with Ray-Centric Transformer (START) that explicitly models feature evolution along individual sensor rays for robust temporal coherence. Furthermore, for controllable synthesis, we propose a novel 4D point cloud-aligned voxel layout for conditioning and a corresponding discrete Masked Generative START (MaskSTART) framework, which learns a compact, tokenized representation of the scene, enabling efficient, high-resolution, and layout-guided compositional generation. Comprehensive experiments validate LiSTAR's state-of-the-art performance across 4D LiDAR reconstruction, prediction, and conditional generation, with substantial quantitative gains: reducing generation MMD by a massive 76%, improving reconstruction IoU by 32%, and lowering prediction L1 Med by 50%. This level of performance provides a powerful new foundation for creating realistic and controllable autonomous systems simulations. Project link: https://ocean-luna.github.io/LiSTAR.gitub.io.
>
---
#### [new 086] Thinking-while-Generating: Interleaving Textual Reasoning throughout Visual Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Thinking-while-Generating（TwiG）框架，解决视觉生成中推理与生成分离的问题。通过在生成过程中动态交织文本推理，实现对局部区域的引导与已有内容的反思，提升输出的语义丰富性与上下文一致性。研究对比了零样本提示、微调和强化学习三种策略，验证了交错推理的有效性。**

- **链接: [https://arxiv.org/pdf/2511.16671v1](https://arxiv.org/pdf/2511.16671v1)**

> **作者:** Ziyu Guo; Renrui Zhang; Hongyu Li; Manyuan Zhang; Xinyan Chen; Sifan Wang; Yan Feng; Peng Pei; Pheng-Ann Heng
>
> **备注:** Project Page: https://think-while-gen.github.io Code: https://github.com/ZiyuGuo99/Thinking-while-Generating
>
> **摘要:** Recent advances in visual generation have increasingly explored the integration of reasoning capabilities. They incorporate textual reasoning, i.e., think, either before (as pre-planning) or after (as post-refinement) the generation process, yet they lack on-the-fly multimodal interaction during the generation itself. In this preliminary study, we introduce Thinking-while-Generating (TwiG), the first interleaved framework that enables co-evolving textual reasoning throughout the visual generation process. As visual content is progressively generating, textual reasoning is interleaved to both guide upcoming local regions and reflect on previously synthesized ones. This dynamic interplay produces more context-aware and semantically rich visual outputs. To unveil the potential of this framework, we investigate three candidate strategies, zero-shot prompting, supervised fine-tuning (SFT) on our curated TwiG-50K dataset, and reinforcement learning (RL) via a customized TwiG-GRPO strategy, each offering unique insights into the dynamics of interleaved reasoning. We hope this work inspires further research into interleaving textual reasoning for enhanced visual generation. Code will be released at: https://github.com/ZiyuGuo99/Thinking-while-Generating.
>
---
#### [new 087] Crossmodal learning for Crop Canopy Trait Estimation
- **分类: cs.CV**

- **简介: 该论文针对卫星影像空间分辨率低、难以支持精细化农田管理的问题，提出跨模态学习方法，将高分辨率无人机影像的细节信息迁移至卫星影像，提升其在作物性状估计中的表现。通过训练模型建立卫星与无人机图像间的细粒度对应关系，生成类无人机视觉表征，在产量和氮素预测等任务上优于原始卫星数据。**

- **链接: [https://arxiv.org/pdf/2511.16031v1](https://arxiv.org/pdf/2511.16031v1)**

> **作者:** Timilehin T. Ayanlade; Anirudha Powadi; Talukder Z. Jubery; Baskar Ganapathysubramanian; Soumik Sarkar
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Recent advances in plant phenotyping have driven widespread adoption of multi sensor platforms for collecting crop canopy reflectance data. This includes the collection of heterogeneous data across multiple platforms, with Unmanned Aerial Vehicles (UAV) seeing significant usage due to their high performance in crop monitoring, forecasting, and prediction tasks. Similarly, satellite missions have been shown to be effective for agriculturally relevant tasks. In contrast to UAVs, such missions are bound to the limitation of spatial resolution, which hinders their effectiveness for modern farming systems focused on micro-plot management. In this work, we propose a cross modal learning strategy that enriches high-resolution satellite imagery with UAV level visual detail for crop canopy trait estimation. Using a dataset of approximately co registered satellite UAV image pairs collected from replicated plots of 84 hybrid maize varieties across five distinct locations in the U.S. Corn Belt, we train a model that learns fine grained spectral spatial correspondences between sensing modalities. Results show that the generated UAV-like representations from satellite inputs consistently outperform real satellite imagery on multiple downstream tasks, including yield and nitrogen prediction, demonstrating the potential of cross-modal correspondence learning to bridge the gap between satellite and UAV sensing in agricultural monitoring.
>
---
#### [new 088] Click2Graph: Interactive Panoptic Video Scene Graphs from a Single Click
- **分类: cs.CV**

- **简介: 该论文提出Click2Graph，首个交互式全景视频场景图生成框架。针对现有VSG系统缺乏用户引导、分割模型缺少语义推理的问题，通过单次点击/框实现目标跟踪、交互对象发现与三元组预测，融合动态交互发现与联合语义分类，实现可控、可解释的视频场景理解。**

- **链接: [https://arxiv.org/pdf/2511.15948v1](https://arxiv.org/pdf/2511.15948v1)**

> **作者:** Raphael Ruschel; Hardikkumar Prajapati; Awsafur Rahman; B. S. Manjunath
>
> **摘要:** State-of-the-art Video Scene Graph Generation (VSGG) systems provide structured visual understanding but operate as closed, feed-forward pipelines with no ability to incorporate human guidance. In contrast, promptable segmentation models such as SAM2 enable precise user interaction but lack semantic or relational reasoning. We introduce Click2Graph, the first interactive framework for Panoptic Video Scene Graph Generation (PVSG) that unifies visual prompting with spatial, temporal, and semantic understanding. From a single user cue, such as a click or bounding box, Click2Graph segments and tracks the subject across time, autonomously discovers interacting objects, and predicts <subject, object, predicate> triplets to form a temporally consistent scene graph. Our framework introduces two key components: a Dynamic Interaction Discovery Module that generates subject-conditioned object prompts, and a Semantic Classification Head that performs joint entity and predicate reasoning. Experiments on the OpenPVSG benchmark demonstrate that Click2Graph establishes a strong foundation for user-guided PVSG, showing how human prompting can be combined with panoptic grounding and relational inference to enable controllable and interpretable video scene understanding.
>
---
#### [new 089] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
- **分类: cs.CV**

- **简介: 该论文提出EvoLMM，一种无需标注数据或人工奖励的自进化大型多模态模型框架。通过协同的提问者与求解者代理，利用连续自奖励机制提升模型在图像-文本推理任务中的能力，显著改善了多模态数学推理性能。**

- **链接: [https://arxiv.org/pdf/2511.16672v1](https://arxiv.org/pdf/2511.16672v1)**

> **作者:** Omkat Thawakar; Shravan Venkatraman; Ritesh Thawkar; Abdelrahman Shaker; Hisham Cholakkal; Rao Muhammad Anwer; Salman Khan; Fahad Khan
>
> **备注:** 9 Pages, 6 Figures, 4 Tables
>
> **摘要:** Recent advances in large multimodal models (LMMs) have enabled impressive reasoning and perception abilities, yet most existing training pipelines still depend on human-curated data or externally verified reward models, limiting their autonomy and scalability. In this work, we strive to improve LMM reasoning capabilities in a purely unsupervised fashion (without any annotated data or reward distillation). To this end, we propose a self-evolving framework, named EvoLMM, that instantiates two cooperative agents from a single backbone model: a Proposer, which generates diverse, image-grounded questions, and a Solver, which solves them through internal consistency, where learning proceeds through a continuous self-rewarding process. This dynamic feedback encourages both the generation of informative queries and the refinement of structured reasoning without relying on ground-truth or human judgments. When using the popular Qwen2.5-VL as the base model, our EvoLMM yields consistent gains upto $\sim$3\% on multimodal math-reasoning benchmarks, including ChartQA, MathVista, and MathVision, using only raw training images. We hope our simple yet effective approach will serve as a solid baseline easing future research in self-improving LMMs in a fully-unsupervised fashion. Our code and models are available at https://github.com/mbzuai-oryx/EvoLMM.
>
---
#### [new 090] UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment
- **分类: cs.CV**

- **简介: 该论文针对图像驱动的虚拟试穿任务，旨在解决文本与图像间语义鸿沟及复杂场景数据稀缺问题。提出UniFit框架，利用多模态大模型引导语义对齐，通过可学习查询和两阶段渐进训练，实现通用、高精度的虚拟试穿，支持多服装、跨模特等复杂任务。**

- **链接: [https://arxiv.org/pdf/2511.15831v1](https://arxiv.org/pdf/2511.15831v1)**

> **作者:** Wei Zhang; Yeying Jin; Xin Li; Yan Zhang; Xiaofeng Cong; Cong Wang; Fengcai Qiao; zhichao Lian
>
> **备注:** accepted to AAAI-2026
>
> **摘要:** Image-based virtual try-on (VTON) aims to synthesize photorealistic images of a person wearing specified garments. Despite significant progress, building a universal VTON framework that can flexibly handle diverse and complex tasks remains a major challenge. Recent methods explore multi-task VTON frameworks guided by textual instructions, yet they still face two key limitations: (1) semantic gap between text instructions and reference images, and (2) data scarcity in complex scenarios. To address these challenges, we propose UniFit, a universal VTON framework driven by a Multimodal Large Language Model (MLLM). Specifically, we introduce an MLLM-Guided Semantic Alignment Module (MGSA), which integrates multimodal inputs using an MLLM and a set of learnable queries. By imposing a semantic alignment loss, MGSA captures cross-modal semantic relationships and provides coherent and explicit semantic guidance for the generative process, thereby reducing the semantic gap. Moreover, by devising a two-stage progressive training strategy with a self-synthesis pipeline, UniFit is able to learn complex tasks from limited data. Extensive experiments show that UniFit not only supports a wide range of VTON tasks, including multi-garment and model-to-model try-on, but also achieves state-of-the-art performance. The source code and pretrained models are available at https://github.com/zwplus/UniFit.
>
---
#### [new 091] Automated Interpretable 2D Video Extraction from 3D Echocardiography
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决3D超声心动图中标准2D视图自动提取问题。通过结合深度学习分类器与解剖学规则，从3D数据中自动化生成符合临床习惯的2D视频，提升诊断效率与准确性，验证显示高精度及临床可用性。**

- **链接: [https://arxiv.org/pdf/2511.15946v1](https://arxiv.org/pdf/2511.15946v1)**

> **作者:** Milos Vukadinovic; Hirotaka Ieki; Yuki Sahasi; David Ouyang; Bryan He
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Although the heart has complex three-dimensional (3D) anatomy, conventional medical imaging with cardiac ultrasound relies on a series of 2D videos showing individual cardiac structures. 3D echocardiography is a developing modality that now offers adequate image quality for clinical use, with potential to streamline acquisition and improve assessment of off-axis features. We propose an automated method to select standard 2D views from 3D cardiac ultrasound volumes, allowing physicians to interpret the data in their usual format while benefiting from the speed and usability of 3D scanning. Applying a deep learning view classifier and downstream heuristics based on anatomical landmarks together with heuristics provided by cardiologists, we reconstruct standard echocardiography views. This approach was validated by three cardiologists in blinded evaluation (96\% accuracy in 1,600 videos from 2 hospitals). The downstream 2D videos were also validated in their ability to detect cardiac abnormalities using AI echocardiography models (EchoPrime and PanEcho) as well as ability to generate clinical-grade measurements of cardiac anatomy (EchoNet-Measurement). We demonstrated that the extracted 2D videos preserve spatial calibration and diagnostic features, allowing clinicians to obtain accurate real-world interpretations from 3D volumes. We release the code and a dataset of 29 3D echocardiography videos https://github.com/echonet/3d-echo .
>
---
#### [new 092] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion
- **分类: cs.CV**

- **简介: 该论文针对点云补全任务，解决现有方法在细节保留与结构一致性间的矛盾。提出Simba框架，将变换回归转为分布学习，结合扩散模型与对称先验，提升泛化性与鲁棒性，并采用分层Mamba架构实现高保真上采样。**

- **链接: [https://arxiv.org/pdf/2511.16161v1](https://arxiv.org/pdf/2511.16161v1)**

> **作者:** Lirui Zhang; Zhengkai Zhao; Zhi Zuo; Pan Gao; Jie Qin
>
> **备注:** Accepted for publication at the 40th AAAI Conference on Artificial Intelligence (AAAI-26)
>
> **摘要:** Point cloud completion is a fundamental task in 3D vision. A persistent challenge in this field is simultaneously preserving fine-grained details present in the input while ensuring the global structural integrity of the completed shape. While recent works leveraging local symmetry transformations via direct regression have significantly improved the preservation of geometric structure details, these methods suffer from two major limitations: (1) These regression-based methods are prone to overfitting which tend to memorize instant-specific transformations instead of learning a generalizable geometric prior. (2) Their reliance on point-wise transformation regression lead to high sensitivity to input noise, severely degrading their robustness and generalization. To address these challenges, we introduce Simba, a novel framework that reformulates point-wise transformation regression as a distribution learning problem. Our approach integrates symmetry priors with the powerful generative capabilities of diffusion models, avoiding instance-specific memorization while capturing robust geometric structures. Additionally, we introduce a hierarchical Mamba-based architecture to achieve high-fidelity upsampling. Extensive experiments across the PCN, ShapeNet, and KITTI benchmarks validate our method's state-of-the-art (SOTA) performance.
>
---
#### [new 093] End-to-End Motion Capture from Rigid Body Markers with Geodesic Loss
- **分类: cs.CV; cs.HC**

- **简介: 该论文针对标记式动作捕捉中密集标记导致的设置繁琐、识别模糊问题，提出基于刚体标记（RBM）的稀疏6-DoF数据新范式。通过构建端到端深度学习模型，结合测地线损失，实现对SMPL人体参数的高效高精度估计，显著降低计算开销，适用于实时图形、VR与生物力学应用。**

- **链接: [https://arxiv.org/pdf/2511.16418v1](https://arxiv.org/pdf/2511.16418v1)**

> **作者:** Hai Lan; Zongyan Li; Jianmin Hu; Jialing Yang; Houde Dai
>
> **备注:** The source code is available in : https://github.com/wer010/GLRBM-Mocap
>
> **摘要:** Marker-based optical motion capture (MoCap), while long regarded as the gold standard for accuracy, faces practical challenges, such as time-consuming preparation and marker identification ambiguity, due to its reliance on dense marker configurations, which fundamentally limit its scalability. To address this, we introduce a novel fundamental unit for MoCap, the Rigid Body Marker (RBM), which provides unambiguous 6-DoF data and drastically simplifies setup. Leveraging this new data modality, we develop a deep-learning-based regression model that directly estimates SMPL parameters under a geodesic loss. This end-to-end approach matches the performance of optimization-based methods while requiring over an order of magnitude less computation. Trained on synthesized data from the AMASS dataset, our end-to-end model achieves state-of-the-art accuracy in body pose estimation. Real-world data captured using a Vicon optical tracking system further demonstrates the practical viability of our approach. Overall, the results show that combining sparse 6-DoF RBM with a manifold-aware geodesic loss yields a practical and high-fidelity solution for real-time MoCap in graphics, virtual reality, and biomechanics.
>
---
#### [new 094] SpectralTrain: A Universal Framework for Hyperspectral Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对高光谱图像分类中训练成本高的问题，提出SpectralTrain框架。通过结合课程学习与基于PCA的光谱降采样，实现高效训练，显著降低计算开销，且兼容多种模型架构。实验验证其在多个数据集上具强泛化性，训练速度提升2-7倍，同时保持较高精度。**

- **链接: [https://arxiv.org/pdf/2511.16084v1](https://arxiv.org/pdf/2511.16084v1)**

> **作者:** Meihua Zhou; Liping Yu; Jiawei Cai; Wai Kin Fung; Ruiguo Hu; Jiarui Zhao; Wenzhuo Liu; Nan Wan
>
> **摘要:** Hyperspectral image (HSI) classification typically involves large-scale data and computationally intensive training, which limits the practical deployment of deep learning models in real-world remote sensing tasks. This study introduces SpectralTrain, a universal, architecture-agnostic training framework that enhances learning efficiency by integrating curriculum learning (CL) with principal component analysis (PCA)-based spectral downsampling. By gradually introducing spectral complexity while preserving essential information, SpectralTrain enables efficient learning of spectral -- spatial patterns at significantly reduced computational costs. The framework is independent of specific architectures, optimizers, or loss functions and is compatible with both classical and state-of-the-art (SOTA) models. Extensive experiments on three benchmark datasets -- Indian Pines, Salinas-A, and the newly introduced CloudPatch-7 -- demonstrate strong generalization across spatial scales, spectral characteristics, and application domains. The results indicate consistent reductions in training time by 2-7x speedups with small-to-moderate accuracy deltas depending on backbone. Its application to cloud classification further reveals potential in climate-related remote sensing, emphasizing training strategy optimization as an effective complement to architectural design in HSI models. Code is available at https://github.com/mh-zhou/SpectralTrain.
>
---
#### [new 095] LLaVA$^3$: Representing 3D Scenes like a Cubist Painter to Boost 3D Scene Understanding of VLMs
- **分类: cs.CV**

- **简介: 该论文针对3D场景理解中3D训练数据稀缺的问题，提出LLaVA³方法。通过多视角2D图像生成类立体派的全景视觉表征，增强视觉语言模型对3D场景的理解能力，无需微调即可在3D VQA和语言定位任务上超越现有2D基线方法。**

- **链接: [https://arxiv.org/pdf/2511.16454v1](https://arxiv.org/pdf/2511.16454v1)**

> **作者:** Doriand Petit; Steve Bourgeois; Vincent Gay-Bellile; Florian Chabot; Loïc Barthe
>
> **备注:** Accepted at AAAI'26
>
> **摘要:** Developing a multi-modal language model capable of understanding 3D scenes remains challenging due to the limited availability of 3D training data, in contrast to the abundance of 2D datasets used for vision-language models (VLM). As an alternative, we introduce LLaVA$^3$ (pronounced LLaVA-Cube), a novel method that improves the 3D scene understanding capabilities of VLM using only multi-view 2D images and without any fine-tuning. Inspired by Cubist painters, who represented multiple viewpoints of a 3D object within a single picture, we propose to describe the 3D scene for the VLM through omnidirectional visual representations of each object. These representations are derived from an intermediate multi-view 3D reconstruction of the scene. Extensive experiments on 3D VQA and 3D language grounding show that our approach outperforms previous 2D-based VLM solutions.
>
---
#### [new 096] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文针对真实世界图像超分辨率任务，解决现有方法在复杂退化下适应性差、知识共享不足的问题。提出基于低秩分解的稀疏专家混合（MoR）架构，将每层秩视为独立专家，结合退化感知路由与动态负载均衡，实现高效精准的单步超分辨率重建。**

- **链接: [https://arxiv.org/pdf/2511.16024v1](https://arxiv.org/pdf/2511.16024v1)**

> **作者:** Xiao He; Zhijun Tu; Kun Cheng; Mingrui Zhu; Jie Hu; Nannan Wang; Xinbo Gao
>
> **备注:** 16 pages, Accepted by AAAI 2026
>
> **摘要:** The demonstrated success of sparsely-gated Mixture-of-Experts (MoE) architectures, exemplified by models such as DeepSeek and Grok, has motivated researchers to investigate their adaptation to diverse domains. In real-world image super-resolution (Real-ISR), existing approaches mainly rely on fine-tuning pre-trained diffusion models through Low-Rank Adaptation (LoRA) module to reconstruct high-resolution (HR) images. However, these dense Real-ISR models are limited in their ability to adaptively capture the heterogeneous characteristics of complex real-world degraded samples or enable knowledge sharing between inputs under equivalent computational budgets. To address this, we investigate the integration of sparse MoE into Real-ISR and propose a Mixture-of-Ranks (MoR) architecture for single-step image super-resolution. We introduce a fine-grained expert partitioning strategy that treats each rank in LoRA as an independent expert. This design enables flexible knowledge recombination while isolating fixed-position ranks as shared experts to preserve common-sense features and minimize routing redundancy. Furthermore, we develop a degradation estimation module leveraging CLIP embeddings and predefined positive-negative text pairs to compute relative degradation scores, dynamically guiding expert activation. To better accommodate varying sample complexities, we incorporate zero-expert slots and propose a degradation-aware load-balancing loss, which dynamically adjusts the number of active experts based on degradation severity, ensuring optimal computational resource allocation. Comprehensive experiments validate our framework's effectiveness and state-of-the-art performance.
>
---
#### [new 097] PrIntMesh: Precise Intersection Surfaces for 3D Organ Mesh Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出PrIntMesh，一种用于3D器官网格重建的模板驱动框架。针对现有深度学习方法独立处理器官子结构导致解剖不一致的问题，通过联合变形保持内部边界与拓扑一致性，实现精确的交界面重建。在心脏、海马体和肺部数据上验证了其高精度与数据高效性。**

- **链接: [https://arxiv.org/pdf/2511.16186v1](https://arxiv.org/pdf/2511.16186v1)**

> **作者:** Deniz Sayin Mercadier; Hieu Le; Yihong Chen; Jiancheng Yang; Udaranga Wickramasinghe; Pascal Fua
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Human organs are composed of interconnected substructures whose geometry and spatial relationships constrain one another. Yet, most deep-learning approaches treat these parts independently, producing anatomically implausible reconstructions. We introduce PrIntMesh, a template-based, topology-preserving framework that reconstructs organs as unified systems. Starting from a connected template, PrIntMesh jointly deforms all substructures to match patient-specific anatomy, while explicitly preserving internal boundaries and enforcing smooth, artifact-free surfaces. We demonstrate its effectiveness on the heart, hippocampus, and lungs, achieving high geometric accuracy, correct topology, and robust performance even with limited or noisy training data. Compared to voxel- and surface-based methods, PrIntMesh better reconstructs shared interfaces, maintains structural consistency, and provides a data-efficient solution suitable for clinical use.
>
---
#### [new 098] VTinker: Guided Flow Upsampling and Texture Mapping for High-Resolution Video Frame Interpolation
- **分类: cs.CV**

- **简介: 该论文针对高分辨率视频帧插值中运动估计精度不足导致的模糊与伪影问题，提出VTinker框架。通过引导式光流上采样增强边缘清晰度，并引入纹理映射生成中间代理以减少像素级伪影，有效提升插值质量。**

- **链接: [https://arxiv.org/pdf/2511.16124v1](https://arxiv.org/pdf/2511.16124v1)**

> **作者:** Chenyang Wu; Jiayi Fu; Chun-Le Guo; Shuhao Han; Chongyi Li
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Due to large pixel movement and high computational cost, estimating the motion of high-resolution frames is challenging. Thus, most flow-based Video Frame Interpolation (VFI) methods first predict bidirectional flows at low resolution and then use high-magnification upsampling (e.g., bilinear) to obtain the high-resolution ones. However, this kind of upsampling strategy may cause blur or mosaic at the flows' edges. Additionally, the motion of fine pixels at high resolution cannot be adequately captured in motion estimation at low resolution, which leads to the misalignment of task-oriented flows. With such inaccurate flows, input frames are warped and combined pixel-by-pixel, resulting in ghosting and discontinuities in the interpolated frame. In this study, we propose a novel VFI pipeline, VTinker, which consists of two core components: guided flow upsampling (GFU) and Texture Mapping. After motion estimation at low resolution, GFU introduces input frames as guidance to alleviate the blurring details in bilinear upsampling flows, which makes flows' edges clearer. Subsequently, to avoid pixel-level ghosting and discontinuities, Texture Mapping generates an initial interpolated frame, referred to as the intermediate proxy. The proxy serves as a cue for selecting clear texture blocks from the input frames, which are then mapped onto the proxy to facilitate producing the final interpolated frame via a reconstruction module. Extensive experiments demonstrate that VTinker achieves state-of-the-art performance in VFI. Codes are available at: https://github.com/Wucy0519/VTinker.
>
---
#### [new 099] EfficientSAM3: Progressive Hierarchical Distillation for Video Concept Segmentation from SAM1, 2, and 3
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视频概念分割任务，解决SAM3模型计算量大、无法在设备端部署的问题。提出EfficientSAM3，通过渐进式分层蒸馏（PHD）将SAM3能力迁移至轻量级学生模型，涵盖编码器、时序记忆和端到端微调三阶段，实现高效且高保真的一体化视频概念分割与跟踪。**

- **链接: [https://arxiv.org/pdf/2511.15833v1](https://arxiv.org/pdf/2511.15833v1)**

> **作者:** Chengxi Zeng; Yuxuan Jiang; Aaron Zhang
>
> **备注:** Github: https://github.com/SimonZeng7108/efficientsam3
>
> **摘要:** The Segment Anything Model 3 (SAM3) advances visual understanding with Promptable Concept Segmentation (PCS) across images and videos, but its unified architecture (shared vision backbone, DETR-style detector, dense-memory tracker) remains prohibitive for on-device use. We present EfficientSAM3, a family of efficient models built on Progressive Hierarchical Distillation (PHD) that transfers capability from SAM3 to lightweight students in three stages: (1) Encoder Distillation aligns image features via prompt-in-the-loop training on SA-1B; (2) Temporal Memory Distillation replaces dense memory with a compact Perceiver-based module trained on SA-V to compress and retrieve spatiotemporal features efficiently; and (3) End-to-End Fine-Tuning refines the full pipeline on the official SAM3 PCS data to preserve concept-level performance. PHD yields a spectrum of student variants using RepViT, TinyViT, and EfficientViT backbones, enabling on-device concept segmentation and tracking while maintaining high fidelity to teacher behavior. We benchmark on popular VOS datasets, and compare with varies of releated work, achieing strong performance-efficiency trade-offs.
>
---
#### [new 100] WALDO: Where Unseen Model-based 6D Pose Estimation Meets Occlusion
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对未见过物体在遮挡下的6D姿态估计问题，提出WALDO方法。通过动态采样、多假设推理、迭代优化和遮挡增强训练，提升精度与鲁棒性，并设计新的评估指标。在ICBIN和BOP上均显著优于现有方法，且推理速度更快。**

- **链接: [https://arxiv.org/pdf/2511.15874v1](https://arxiv.org/pdf/2511.15874v1)**

> **作者:** Sajjad Pakdamansavoji; Yintao Ma; Amir Rasouli; Tongtong Cao
>
> **摘要:** Accurate 6D object pose estimation is vital for robotics, augmented reality, and scene understanding. For seen objects, high accuracy is often attainable via per-object fine-tuning but generalizing to unseen objects remains a challenge. To address this problem, past arts assume access to CAD models at test time and typically follow a multi-stage pipeline to estimate poses: detect and segment the object, propose an initial pose, and then refine it. Under occlusion, however, the early-stage of such pipelines are prone to errors, which can propagate through the sequential processing, and consequently degrade the performance. To remedy this shortcoming, we propose four novel extensions to model-based 6D pose estimation methods: (i) a dynamic non-uniform dense sampling strategy that focuses computation on visible regions, reducing occlusion-induced errors; (ii) a multi-hypothesis inference mechanism that retains several confidence-ranked pose candidates, mitigating brittle single-path failures; (iii) iterative refinement to progressively improve pose accuracy; and (iv) series of occlusion-focused training augmentations that strengthen robustness and generalization. Furthermore, we propose a new weighted by visibility metric for evaluation under occlusion to minimize the bias in the existing protocols. Via extensive empirical evaluations, we show that our proposed approach achieves more than 5% improvement in accuracy on ICBIN and more than 2% on BOP dataset benchmarks, while achieving approximately 3 times faster inference.
>
---
#### [new 101] EOGS++: Earth Observation Gaussian Splatting with Internal Camera Refinement and Direct Panchromatic Rendering
- **分类: cs.CV**

- **简介: 该论文针对卫星影像三维重建任务，解决传统方法依赖预处理和外部优化的问题。提出EOGS++，直接处理高分辨率全色图像，融合光流引导的相机位姿优化与改进的后处理，显著提升重建质量与几何精度，实现更高效、更准确的地球观测三维重建。**

- **链接: [https://arxiv.org/pdf/2511.16542v1](https://arxiv.org/pdf/2511.16542v1)**

> **作者:** Pierrick Bournez; Luca Savant Aira; Thibaud Ehret; Gabriele Facciolo
>
> **备注:** 8 pages, ISPRS
>
> **摘要:** Recently, 3D Gaussian Splatting has been introduced as a compelling alternative to NeRF for Earth observation, offering com- petitive reconstruction quality with significantly reduced training times. In this work, we extend the Earth Observation Gaussian Splatting (EOGS) framework to propose EOGS++, a novel method tailored for satellite imagery that directly operates on raw high-resolution panchromatic data without requiring external preprocessing. Furthermore, leveraging optical flow techniques we embed bundle adjustment directly within the training process, avoiding reliance on external optimization tools while improving camera pose estimation. We also introduce several improvements to the original implementation, including early stopping and TSDF post-processing, all contributing to sharper reconstructions and better geometric accuracy. Experiments on the IARPA 2016 and DFC2019 datasets demonstrate that EOGS++ achieves state-of-the-art performance in terms of reconstruction quality and effi- ciency, outperforming the original EOGS method and other NeRF-based methods while maintaining the computational advantages of Gaussian Splatting. Our model demonstrates an improvement from 1.33 to 1.19 mean MAE errors on buildings compared to the original EOGS models
>
---
#### [new 102] Teacher-Guided One-Shot Pruning via Context-Aware Knowledge Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对深度神经网络压缩中的计算开销问题，提出一种教师引导的一次性剪枝方法。通过融合上下文感知的知识蒸馏与重要性评分，实现高效全局剪枝，无需迭代训练，显著降低计算成本，同时保持高精度，在多个图像分类数据集上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16653v1](https://arxiv.org/pdf/2511.16653v1)**

> **作者:** Md. Samiul Alim; Sharjil Khan; Amrijit Biswas; Fuad Rahman; Shafin Rahman; Nabeel Mohammed
>
> **备注:** Accepted at 2025 IEEE International Conference on Big Data (IEEE BigData 2025)
>
> **摘要:** Unstructured pruning remains a powerful strategy for compressing deep neural networks, yet it often demands iterative train-prune-retrain cycles, resulting in significant computational overhead. To address this challenge, we introduce a novel teacher-guided pruning framework that tightly integrates Knowledge Distillation (KD) with importance score estimation. Unlike prior approaches that apply KD as a post-pruning recovery step, our method leverages gradient signals informed by the teacher during importance score calculation to identify and retain parameters most critical for both task performance and knowledge transfer. Our method facilitates a one-shot global pruning strategy that efficiently eliminates redundant weights while preserving essential representations. After pruning, we employ sparsity-aware retraining with and without KD to recover accuracy without reactivating pruned connections. Comprehensive experiments across multiple image classification benchmarks, including CIFAR-10, CIFAR-100, and TinyImageNet, demonstrate that our method consistently achieves high sparsity levels with minimal performance degradation. Notably, our approach outperforms state-of-the-art baselines such as EPG and EPSD at high sparsity levels, while offering a more computationally efficient alternative to iterative pruning schemes like COLT. The proposed framework offers a computation-efficient, performance-preserving solution well suited for deployment in resource-constrained environments.
>
---
#### [new 103] CylinderDepth: Cylindrical Spatial Attention for Multi-View Consistent Self-Supervised Surround Depth Estimation
- **分类: cs.CV**

- **简介: 该论文针对多视角自监督环视深度估计中图像间深度不一致的问题，提出CylinderDepth方法。通过将各图像的3D点投影到共享单位圆柱面，构建跨图像的空间位置映射，并设计非学习的几何引导空间注意力机制，聚合圆柱面上邻近像素特征，实现更一致的深度预测。**

- **链接: [https://arxiv.org/pdf/2511.16428v1](https://arxiv.org/pdf/2511.16428v1)**

> **作者:** Samer Abualhanud; Christian Grannemann; Max Mehltretter
>
> **摘要:** Self-supervised surround-view depth estimation enables dense, low-cost 3D perception with a 360° field of view from multiple minimally overlapping images. Yet, most existing methods suffer from depth estimates that are inconsistent between overlapping images. Addressing this limitation, we propose a novel geometry-guided method for calibrated, time-synchronized multi-camera rigs that predicts dense, metric, and cross-view-consistent depth. Given the intrinsic and relative orientation parameters, a first depth map is predicted per image and the so-derived 3D points from all images are projected onto a shared unit cylinder, establishing neighborhood relations across different images. This produces a 2D position map for every image, where each pixel is assigned its projected position on the cylinder. Based on these position maps, we apply an explicit, non-learned spatial attention that aggregates features among pixels across images according to their distances on the cylinder, to predict a final depth map per image. Evaluated on the DDAD and nuScenes datasets, our approach improves the consistency of depth estimates across images and the overall depth compared to state-of-the-art methods.
>
---
#### [new 104] Automatic Uncertainty-Aware Synthetic Data Bootstrapping for Historical Map Segmentation
- **分类: cs.CV**

- **简介: 该论文针对历史地图语义分割中标注数据稀缺问题，提出一种自动化的不确定性感知合成数据生成方法。通过风格迁移与噪声模拟，生成大量逼真且多样化的合成历史地图，提升模型在小样本场景下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.15875v1](https://arxiv.org/pdf/2511.15875v1)**

> **作者:** Lukas Arzoumanidis; Julius Knechtel; Jan-Henrik Haunert; Youness Dehbi
>
> **摘要:** The automated analysis of historical documents, particularly maps, has drastically benefited from advances in deep learning and its success across various computer vision applications. However, most deep learning-based methods heavily rely on large amounts of annotated training data, which are typically unavailable for historical maps, especially for those belonging to specific, homogeneous cartographic domains, also known as corpora. Creating high-quality training data suitable for machine learning often takes a significant amount of time and involves extensive manual effort. While synthetic training data can alleviate the scarcity of real-world samples, it often lacks the affinity (realism) and diversity (variation) necessary for effective learning. By transferring the cartographic style of an original historical map corpus onto vector data, we bootstrap an effectively unlimited number of synthetic historical maps suitable for tasks such as land-cover interpretation of a homogeneous historical map corpus. We propose an automatic deep generative approach and a alternative manual stochastic degradation technique to emulate the visual uncertainty and noise, also known as data-dependent uncertainty, commonly observed in historical map scans. To quantitatively evaluate the effectiveness and applicability of our approach, the generated training datasets were employed for domain-adaptive semantic segmentation on a homogeneous map corpus using a Self-Constructing Graph Convolutional Network, enabling a comprehensive assessment of the impact of our data bootstrapping methods.
>
---
#### [new 105] Externally Validated Multi-Task Learning via Consistency Regularization Using Differentiable BI-RADS Features for Breast Ultrasound Tumor Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对乳腺超声肿瘤分割中的多任务学习泛化问题，提出基于可微BI-RADS特征的一致性正则化方法，有效缓解任务间干扰。在多个外部数据集上验证，显著提升分割性能，实现更优的跨中心泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.15968v1](https://arxiv.org/pdf/2511.15968v1)**

> **作者:** Jingru Zhang; Saed Moradi; Ashirbani Saha
>
> **摘要:** Multi-task learning can suffer from destructive task interference, where jointly trained models underperform single-task baselines and limit generalization. To improve generalization performance in breast ultrasound-based tumor segmentation via multi-task learning, we propose a novel consistency regularization approach that mitigates destructive interference between segmentation and classification. The consistency regularization approach is composed of differentiable BI-RADS-inspired morphological features. We validated this approach by training all models on the BrEaST dataset (Poland) and evaluating them on three external datasets: UDIAT (Spain), BUSI (Egypt), and BUS-UCLM (Spain). Our comprehensive analysis demonstrates statistically significant (p<0.001) improvements in generalization for segmentation task of the proposed multi-task approach vs. the baseline one: UDIAT, BUSI, BUS-UCLM (Dice coefficient=0.81 vs 0.59, 0.66 vs 0.56, 0.69 vs 0.49, resp.). The proposed approach also achieves state-of-the-art segmentation performance under rigorous external validation on the UDIAT dataset.
>
---
#### [new 106] Video2Layout: Recall and Reconstruct Metric-Grounded Cognitive Map for Spatial Reasoning
- **分类: cs.CV**

- **简介: 该论文提出Video2Layout，旨在解决多模态大模型在空间推理中因离散栅格表示导致的细粒度空间理解不足问题。通过连续边界坐标重建度量基准空间布局，提升量化空间计算能力，并构建QVS-Bench评估基准，验证了方法在空间推理上的优越性。**

- **链接: [https://arxiv.org/pdf/2511.16160v1](https://arxiv.org/pdf/2511.16160v1)**

> **作者:** Yibin Huang; Wang Xu; Wanyue Zhang; Helu Zhi; Jingjing Huang; Yangbin Xu; Yangang Sun; Conghui Zhu; Tiejun Zhao
>
> **摘要:** Spatial intelligence is a critical frontier for Multimodal Large Language Models (MLLMs), empowering them to comprehend the physical world. Drawing inspiration from human perception mechanisms, existing studies attempt to construct a coherent spatial understanding via grid-based cognitive maps from multi-frame visual inputs. However, current grid-based map methods rely on discretized raster representations, which limit the model's ability in fine-grained spatial reasoning. To overcome this limitation, we propose Video2Layout, a framework for reconstructing metric-grounded spatial layouts from video. The framework employs continuous object boundary coordinates to quantify inter-object physical distances and object size. This empowers the model with quantitative spatial computation capabilities, effectively alleviating the inherent ambiguity when describing spatial relationships in natural language. Specifically, our method comprises two core stages. First, in supervised fine-tuning stage, we construct a high-quality dataset from the AI2THOR simulator, which enables the model to learn the mapping from visual inputs to precise boundary coordinates. Subsequently, a reinforcement fine-tuning stage further enhances the model's real-world generalization capabilities. To systematically evaluate the correlation between cognitive map accuracy and image quantity, as well as how the quantity of image inputs affects spatial reasoning accuracy, we introduce QVS-Bench, a diagnostic benchmark designed to analyze the relevant mechanisms. Evaluated on QVS-Bench and mainstream spatial reasoning benchmarks, our model, V2LO-7B achieves an average improvement of 4.92% over the model trained on grid maps, validating the superiority of our method. Our code is available at https://github.com/ybrrraway/Video2Layout.
>
---
#### [new 107] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文针对扩散变压器（DiT）参数量大、计算成本高的问题，提出一种可插拔的结构化剪枝框架PPCL。通过分析层间冗余并设计师生交替蒸馏机制，实现深度与宽度联合剪枝，无需重新训练即可支持多种剪枝率。实验表明，模型参数减少50%，性能损失低于3%，显著提升资源受限环境下的部署可行性。**

- **链接: [https://arxiv.org/pdf/2511.16156v1](https://arxiv.org/pdf/2511.16156v1)**

> **作者:** Jian Ma; Qirong Peng; Xujie Zhu; Peixing Xie; Chen Chen; Haonan Lu
>
> **备注:** https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning
>
> **摘要:** Diffusion Transformers (DiTs) have shown exceptional performance in image generation, yet their large parameter counts incur high computational costs, impeding deployment in resource-constrained settings. To address this, we propose Pluggable Pruning with Contiguous Layer Distillation (PPCL), a flexible structured pruning framework specifically designed for DiT architectures. First, we identify redundant layer intervals through a linear probing mechanism combined with the first-order differential trend analysis of similarity metrics. Subsequently, we propose a plug-and-play teacher-student alternating distillation scheme tailored to integrate depth-wise and width-wise pruning within a single training phase. This distillation framework enables flexible knowledge transfer across diverse pruning ratios, eliminating the need for per-configuration retraining. Extensive experiments on multiple Multi-Modal Diffusion Transformer architecture models demonstrate that PPCL achieves a 50\% reduction in parameter count compared to the full model, with less than 3\% degradation in key objective metrics. Notably, our method maintains high-quality image generation capabilities while achieving higher compression ratios, rendering it well-suited for resource-constrained environments. The open-source code, checkpoints for PPCL can be found at the following link: https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning.
>
---
#### [new 108] Learning to Think Fast and Slow for Visual Language Models
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型在复杂任务中计算效率低的问题，提出双模式思考机制（DualMindVLM）。通过基于输出长度的标签划分快慢思维模式，并使用GRPO强化学习训练，使模型根据任务难度自动切换思考模式，显著提升推理效率与性能。**

- **链接: [https://arxiv.org/pdf/2511.16670v1](https://arxiv.org/pdf/2511.16670v1)**

> **作者:** Chenyu Lin; Cheng Chi; Jinlin Wu; Sharon Li; Kaiyang Zhou
>
> **摘要:** When confronted with complex problems, we tend to think slowly; conversely, for simple questions, we think quickly. Such a two-system thinking mechanism allows us to efficiently allocate cognitive resources, enabling quick decision-making for straightforward issues while reserving deeper analytical thinking for more intricate challenges. However, existing reasoning-oriented visual language models (VLMs), whether trained with explicit chain-of-thought annotations or rule-based RL rewards, mainly pursue lengthy, detailed reasoning chains, which often lead to excessive computational costs. In this work, we propose a simple RL approach, which enables VLMs to automatically switch between fast and slow thinking modes depending on task difficulty. The approach consists of two stages: in the first stage, we label data as either requiring fast thinking or slow thinking based on the model output length, which is inspired by the observation that pre-trained VLMs typically produce answers of varying lengths for different types of questions; in the second stage, we train the model using GRPO along with the thinking mode labels to develop dual-mode thinking. Despite its simplicity, our model, named DualMindVLM, significantly outperforms the base model and achieves performance on par with state-of-the-art visual reasoning models, while maintaining exceptionally high token efficiency.
>
---
#### [new 109] Flow and Depth Assisted Video Prediction with Latent Transformer
- **分类: cs.CV**

- **简介: 该论文研究视频预测任务，针对遮挡场景下预测不准的问题，提出基于潜在变换器的模型，融合点流与深度图信息。通过在合成与真实数据集上的实验验证，证明引入几何与运动先验可提升遮挡场景下的预测精度，尤其改善背景运动建模。**

- **链接: [https://arxiv.org/pdf/2511.16484v1](https://arxiv.org/pdf/2511.16484v1)**

> **作者:** Eliyas Suleyman; Paul Henderson; Eksan Firkat; Nicolas Pugeault
>
> **摘要:** Video prediction is a fundamental task for various downstream applications, including robotics and world modeling. Although general video prediction models have achieved remarkable performance in standard scenarios, occlusion is still an inherent challenge in video prediction. We hypothesize that providing explicit information about motion (via point-flow) and geometric structure (via depth-maps) will enable video prediction models to perform better in situations with occlusion and the background motion. To investigate this, we present the first systematic study dedicated to occluded video prediction. We use a standard multi-object latent transformer architecture to predict future frames, but modify this to incorporate information from depth and point-flow. We evaluate this model in a controlled setting on both synthetic and real-world datasets with not only appearance-based metrics but also Wasserstein distances on object masks, which can effectively measure the motion distribution of the prediction. We find that when the prediction model is assisted with point flow and depth, it performs better in occluded scenarios and predicts more accurate background motion compared to models without the help of these modalities.
>
---
#### [new 110] Rad-GS: Radar-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments
- **分类: cs.CV**

- **简介: 该论文提出Rad-GS，一种面向室外大尺度环境的雷达-视觉融合4D SLAM系统，利用3D高斯作为可微分空间表示。通过融合雷达点云与多普勒信息，实现动态物体掩码，减少渲染伪影；借助非同步图像全局优化高斯表示，提升纹理一致性和新视角合成质量；结合全局八叉树与针对性高斯管理，降低噪声与内存消耗，实现了千米级场景的鲁棒重建。**

- **链接: [https://arxiv.org/pdf/2511.16091v1](https://arxiv.org/pdf/2511.16091v1)**

> **作者:** Renxiang Xiao; Wei Liu; Yuanfan Zhang; Yushuai Chen; Jinming Chen; Zilu Wang; Liang Hu
>
> **摘要:** We present Rad-GS, a 4D radar-camera SLAM system designed for kilometer-scale outdoor environments, utilizing 3D Gaussian as a differentiable spatial representation. Rad-GS combines the advantages of raw radar point cloud with Doppler information and geometrically enhanced point cloud to guide dynamic object masking in synchronized images, thereby alleviating rendering artifacts and improving localization accuracy. Additionally, unsynchronized image frames are leveraged to globally refine the 3D Gaussian representation, enhancing texture consistency and novel view synthesis fidelity. Furthermore, the global octree structure coupled with a targeted Gaussian primitive management strategy further suppresses noise and significantly reduces memory consumption in large-scale environments. Extensive experiments and ablation studies demonstrate that Rad-GS achieves performance comparable to traditional 3D Gaussian methods based on camera or LiDAR inputs, highlighting the feasibility of robust outdoor mapping using 4D mmWave radar. Real-world reconstruction at kilometer scale validates the potential of Rad-GS for large-scale scene reconstruction.
>
---
#### [new 111] Green Resilience of Cyber-Physical Systems: Doctoral Dissertation
- **分类: cs.SE; cs.AI; cs.CV; cs.RO**

- **简介: 该论文研究在线协作人工智能系统（OL-CAIS）的绿色韧性问题，旨在平衡系统在扰动后的恢复能力与能耗。通过构建三态模型与GResilience框架，提出多目标优化、博弈决策与强化学习策略，实现高效绿色恢复，并量化韧性与绿色度。实验验证了方法有效性，揭示了灾难性遗忘现象并提出缓解措施。**

- **链接: [https://arxiv.org/pdf/2511.16593v1](https://arxiv.org/pdf/2511.16593v1)**

> **作者:** Diaeddin Rimawi
>
> **摘要:** Cyber-physical systems (CPS) combine computational and physical components. Online Collaborative AI System (OL-CAIS) is a type of CPS that learn online in collaboration with humans to achieve a common goal, which makes it vulnerable to disruptive events that degrade performance. Decision-makers must therefore restore performance while limiting energy impact, creating a trade-off between resilience and greenness. This research addresses how to balance these two properties in OL-CAIS. It aims to model resilience for automatic state detection, develop agent-based policies that optimize the greenness-resilience trade-off, and understand catastrophic forgetting to maintain performance consistency. We model OL-CAIS behavior through three operational states: steady, disruptive, and final. To support recovery during disruptions, we introduce the GResilience framework, which provides recovery strategies through multi-objective optimization (one-agent), game-theoretic decision-making (two-agent), and reinforcement learning (RL-agent). We also design a measurement framework to quantify resilience and greenness. Empirical evaluation uses real and simulated experiments with a collaborative robot learning object classification from human demonstrations. Results show that the resilience model captures performance transitions during disruptions, and that GResilience policies improve green recovery by shortening recovery time, stabilizing performance, and reducing human dependency. RL-agent policies achieve the strongest results, although with a marginal increase in CO2 emissions. We also observe catastrophic forgetting after repeated disruptions, while our policies help maintain steadiness. A comparison with containerized execution shows that containerization cuts CO2 emissions by half. Overall, this research provides models, metrics, and policies that ensure the green recovery of OL-CAIS.
>
---
#### [new 112] FOOTPASS: A Multi-Modal Multi-Agent Tactical Context Dataset for Play-by-Play Action Spotting in Soccer Broadcast Videos
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出FOOTPASS数据集，面向足球比赛全时长的逐事件动作定位任务。针对现有方法难以自动构建可靠逐事件数据的问题，融合多模态视觉信息与足球战术先验知识，实现基于计算机视觉与战术规律的球员中心动作识别，推动自动化体育数据分析发展。**

- **链接: [https://arxiv.org/pdf/2511.16183v1](https://arxiv.org/pdf/2511.16183v1)**

> **作者:** Jeremie Ochin; Raphael Chekroun; Bogdan Stanciulescu; Sotiris Manitsaris
>
> **摘要:** Soccer video understanding has motivated the creation of datasets for tasks such as temporal action localization, spatiotemporal action detection (STAD), or multiobject tracking (MOT). The annotation of structured sequences of events (who does what, when, and where) used for soccer analytics requires a holistic approach that integrates both STAD and MOT. However, current action recognition methods remain insufficient for constructing reliable play-by-play data and are typically used to assist rather than fully automate annotation. Parallel research has advanced tactical modeling, trajectory forecasting, and performance analysis, all grounded in game-state and play-by-play data. This motivates leveraging tactical knowledge as a prior to support computer-vision-based predictions, enabling more automated and reliable extraction of play-by-play data. We introduce Footovision Play-by-Play Action Spotting in Soccer Dataset (FOOTPASS), the first benchmark for play-by-play action spotting over entire soccer matches in a multi-modal, multi-agent tactical context. It enables the development of methods for player-centric action spotting that exploit both outputs from computer-vision tasks (e.g., tracking, identification) and prior knowledge of soccer, including its tactical regularities over long time horizons, to generate reliable play-by-play data streams. These streams form an essential input for data-driven sports analytics.
>
---
#### [new 113] MiMo-Embodied: X-Embodied Foundation Model Technical Report
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出MiMo-Embodied，首个跨具身基础模型，融合自动驾驶与具身智能任务。通过多阶段学习与数据优化，在17项具身AI与12项自动驾驶基准上均达领先性能，验证两领域间正向迁移与协同增强。**

- **链接: [https://arxiv.org/pdf/2511.16518v1](https://arxiv.org/pdf/2511.16518v1)**

> **作者:** Xiaoshuai Hao; Lei Zhou; Zhijian Huang; Zhiwen Hou; Yingbo Tang; Lingfeng Zhang; Guang Li; Zheng Lu; Shuhuai Ren; Xianhui Meng; Yuchen Zhang; Jing Wu; Jinghui Lu; Chenxu Dang; Jiayi Guan; Jianhua Wu; Zhiyi Hou; Hanbing Li; Shumeng Xia; Mingliang Zhou; Yinan Zheng; Zihao Yue; Shuhao Gu; Hao Tian; Yuannan Shen; Jianwei Cui; Wen Zhang; Shaoqing Xu; Bing Wang; Haiyang Sun; Zeyu Zhu; Yuncheng Jiang; Zibin Guo; Chuhong Gong; Chaofan Zhang; Wenbo Ding; Kun Ma; Guang Chen; Rui Cai; Diyun Xiang; Heng Qu; Fuli Luo; Hangjun Ye; Long Chen
>
> **备注:** Code: https://github.com/XiaomiMiMo/MiMo-Embodied Model: https://huggingface.co/XiaomiMiMo/MiMo-Embodied-7B
>
> **摘要:** We open-source MiMo-Embodied, the first cross-embodied foundation model to successfully integrate and achieve state-of-the-art performance in both Autonomous Driving and Embodied AI. MiMo-Embodied sets new records across 17 embodied AI benchmarks in Task Planning, Affordance Prediction and Spatial Understanding, while also excelling in 12 autonomous driving benchmarks across Environmental Perception, Status Prediction, and Driving Planning. Across these tasks, MiMo-Embodied significantly outperforms existing open-source, closed-source, and specialized baselines. Our results indicate that through multi-stage learning, curated data construction, and CoT/RL fine-tuning, these two domains exhibit strong positive transfer and mutually reinforce one another. We provide a detailed analysis of our model design and training methodologies to facilitate further research. Code and models are available at https://github.com/XiaomiMiMo/MiMo-Embodied.
>
---
#### [new 114] How Modality Shapes Perception and Reasoning: A Study of Error Propagation in ARC-AGI
- **分类: cs.AI; cs.CV; cs.MA**

- **简介: 该论文研究多模态对感知与推理的影响，聚焦ARC-AGI任务中错误传播问题。通过对比九种文本与图像模态，发现模态影响特征感知精度；提出两阶段推理框架，结合文本坐标精确性与图像形状保真性，提升执行准确率，实现跨模态验证以优化指令与执行。**

- **链接: [https://arxiv.org/pdf/2511.15717v1](https://arxiv.org/pdf/2511.15717v1)**

> **作者:** Bo Wen; Chen Wang; Erhan Bilal
>
> **摘要:** ARC-AGI and ARC-AGI-2 measure generalization-through-composition on small color-quantized grids, and their prize competitions make progress on these harder held-out tasks a meaningful proxy for systematic generalization. Recent instruction-first systems translate grids into concise natural-language or DSL rules executed in generate-execute-select loops, yet we lack a principled account of how encodings shape model perception and how to separate instruction errors from execution errors. We hypothesize that modality imposes perceptual bottlenecks -- text flattens 2D structure into 1D tokens while images preserve layout but can introduce patch-size aliasing -- thereby shaping which grid features are reliably perceived. To test this, we isolate perception from reasoning across nine text and image modalities using a weighted set-disagreement metric and a two-stage reasoning pipeline, finding that structured text yields precise coordinates on sparse features, images capture 2D shapes yet are resolution-sensitive, and combining them improves execution (about 8 perception points; about 0.20 median similarity). Overall, aligning representations with transformer inductive biases and enabling cross-validation between text and image yields more accurate instructions and more reliable execution without changing the underlying model.
>
---
#### [new 115] UniUltra: Interactive Parameter-Efficient SAM2 for Universal Ultrasound Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对超声图像分割任务，解决SAM2在超声域性能下降及部署资源消耗大的问题。提出UniUltra框架，通过上下文-边缘混合适配器实现参数高效微调，并结合深度监督知识蒸馏压缩模型，显著降低参数量，提升临床适用性。**

- **链接: [https://arxiv.org/pdf/2511.15771v1](https://arxiv.org/pdf/2511.15771v1)**

> **作者:** Yue Li; Qing Xu; Yixuan Zhang; Xiangjian He; Qian Zhang; Yuan Yao; Fiseha B. Tesem; Xin Chen; Ruili Wang; Zhen Chen; Chang Wen Chen
>
> **摘要:** The Segment Anything Model 2 (SAM2) demonstrates remarkable universal segmentation capabilities on natural images. However, its performance on ultrasound images is significantly degraded due to domain disparities. This limitation raises two critical challenges: how to efficiently adapt SAM2 to ultrasound imaging while maintaining parameter efficiency, and how to deploy the adapted model effectively in resource-constrained clinical environments. To address these issues, we propose UniUltra for universal ultrasound segmentation. Specifically, we first introduce a novel context-edge hybrid adapter (CH-Adapter) that enhances fine-grained perception across diverse ultrasound imaging modalities while achieving parameter-efficient fine-tuning. To further improve clinical applicability, we develop a deep-supervised knowledge distillation (DSKD) technique that transfers knowledge from the large image encoder of the fine-tuned SAM2 to a super lightweight encoder, substantially reducing computational requirements without compromising performance. Extensive experiments demonstrate that UniUltra outperforms state-of-the-arts with superior generalization capabilities. Notably, our framework achieves competitive performance using only 8.91% of SAM2's parameters during fine-tuning, and the final compressed model reduces the parameter count by 94.08% compared to the original SAM2, making it highly suitable for practical clinical deployment. The source code is available at https://github.com/xq141839/UniUltra.
>
---
#### [new 116] How Robot Dogs See the Unseeable
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究机器人视觉中的遮挡问题，提出通过模仿动物“侧视”运动实现合成孔径成像。利用机器人移动时相机的运动轨迹形成宽合成孔径，计算融合图像以实现极浅景深，使背景清晰而遮挡物模糊。该方法无需额外传感器，实时高效，可提升复杂场景下的视觉理解能力，尤其增强大模型在遮挡环境中的推理性能。**

- **链接: [https://arxiv.org/pdf/2511.16262v1](https://arxiv.org/pdf/2511.16262v1)**

> **作者:** Oliver Bimber; Karl Dietrich von Ellenrieder; Michael Haller; Rakesh John Amala Arokia Nathan; Gianni Lunardi; Marco Camurri; Mohamed Youssef; Santos Miguel Orozco Soto; Jeremy E. Niven
>
> **摘要:** Peering, a side-to-side motion used by animals to estimate distance through motion parallax, offers a powerful bio-inspired strategy to overcome a fundamental limitation in robotic vision: partial occlusion. Conventional robot cameras, with their small apertures and large depth of field, render both foreground obstacles and background objects in sharp focus, causing occluders to obscure critical scene information. This work establishes a formal connection between animal peering and synthetic aperture (SA) sensing from optical imaging. By having a robot execute a peering motion, its camera describes a wide synthetic aperture. Computational integration of the captured images synthesizes an image with an extremely shallow depth of field, effectively blurring out occluding elements while bringing the background into sharp focus. This efficient, wavelength-independent technique enables real-time, high-resolution perception across various spectral bands. We demonstrate that this approach not only restores basic scene understanding but also empowers advanced visual reasoning in large multimodal models, which fail with conventionally occluded imagery. Unlike feature-dependent multi-view 3D vision methods or active sensors like LiDAR, SA sensing via peering is robust to occlusion, computationally efficient, and immediately deployable on any mobile robot. This research bridges animal behavior and robotics, suggesting that peering motions for synthetic aperture sensing are a key to advanced scene understanding in complex, cluttered environments.
>
---
#### [new 117] Weakly Supervised Segmentation and Classification of Alpha-Synuclein Aggregates in Brightfield Midbrain Images
- **分类: eess.IV; cs.CV; q-bio.QM**

- **简介: 该论文针对帕金森病中α-突触核蛋白聚集物的自动分析任务，解决免疫组化染色变异导致的标注困难问题。提出弱监督分割与分类框架，基于ResNet50实现对中脑全切片图像中聚集体（如路易小体、神经纤维）的精准分割与分类，平衡准确率达80%，推动对病理空间分布及细胞互作的研究。**

- **链接: [https://arxiv.org/pdf/2511.16268v1](https://arxiv.org/pdf/2511.16268v1)**

> **作者:** Erwan Dereure; Robin Louiset; Laura Parkkinen; David A Menassa; David Holcman
>
> **摘要:** Parkinson's disease (PD) is a neurodegenerative disorder associated with the accumulation of misfolded alpha-synuclein aggregates, forming Lewy bodies and neuritic shape used for pathology diagnostics. Automatic analysis of immunohistochemistry histopathological images with Deep Learning provides a promising tool for better understanding the spatial organization of these aggregates. In this study, we develop an automated image processing pipeline to segment and classify these aggregates in whole-slide images (WSIs) of midbrain tissue from PD and incidental Lewy Body Disease (iLBD) cases based on weakly supervised segmentation, robust to immunohistochemical labelling variability, with a ResNet50 classifier. Our approach allows to differentiate between major aggregate morphologies, including Lewy bodies and neurites with a balanced accuracy of $80\%$. This framework paves the way for large-scale characterization of the spatial distribution and heterogeneity of alpha-synuclein aggregates in brightfield immunohistochemical tissue, and for investigating their poorly understood relationships with surrounding cells such as microglia and astrocytes.
>
---
#### [new 118] Arctic-Extract Technical Report
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出Arctic-Extract模型，用于从扫描或数字生成的业务文档中提取结构化数据（问答、实体、表格）。针对资源受限设备部署难题，模型仅6.6 GiB，可在A10 GPU上处理高达125页的长文档，兼具高性能与低资源消耗，显著提升文档理解在边缘设备上的可行性。**

- **链接: [https://arxiv.org/pdf/2511.16470v1](https://arxiv.org/pdf/2511.16470v1)**

> **作者:** Mateusz Chiliński; Julita Ołtusek; Wojciech Jaśkowski
>
> **摘要:** Arctic-Extract is a state-of-the-art model designed for extracting structural data (question answering, entities and tables) from scanned or digital-born business documents. Despite its SoTA capabilities, the model is deployable on resource-constrained hardware, weighting only 6.6 GiB, making it suitable for deployment on devices with limited resources, such as A10 GPUs with 24 GB of memory. Arctic-Extract can process up to 125 A4 pages on those GPUs, making suitable for long document processing. This paper highlights Arctic-Extract's training protocols and evaluation results, demonstrating its strong performance in document understanding.
>
---
## 更新

#### [replaced 001] RAPID: Robust and Agile Planner Using Inverse Reinforcement Learning for Vision-Based Drone Navigation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2502.02054v2](https://arxiv.org/pdf/2502.02054v2)**

> **作者:** Minwoo Kim; Geunsik Bae; Jinwoo Lee; Woojae Shin; Changseung Kim; Myong-Yol Choi; Heejung Shin; Hyondong Oh
>
> **备注:** 18 pages, 11 figures, 58 references, and appendix is included
>
> **摘要:** This paper introduces a learning-based visual planner for agile drone flight in cluttered environments. The proposed planner generates collision-free waypoints in milliseconds, enabling drones to perform agile maneuvers in complex environments without building separate perception, mapping, and planning modules. Learning-based methods, such as behavior cloning (BC) and reinforcement learning (RL), demonstrate promising performance in visual navigation but still face inherent limitations. BC is susceptible to compounding errors due to limited expert imitation, while RL struggles with reward function design and sample inefficiency. To address these limitations, this paper proposes an inverse reinforcement learning (IRL)-based framework for high-speed visual navigation. By leveraging IRL, it is possible to reduce the number of interactions with simulation environments and improve capability to deal with high-dimensional spaces while preserving the robustness of RL policies. A motion primitive-based path planning algorithm collects an expert dataset with privileged map data from diverse environments, ensuring comprehensive scenario coverage. By leveraging both the acquired expert and learner dataset gathered from the agent's interactions with the simulation environments, a robust reward function and policy are learned across diverse states. While the proposed method is trained in a simulation environment only, it can be directly applied to real-world scenarios without additional training or tuning. The performance of the proposed method is validated in both simulation and real-world environments, including forests and various structures. The trained policy achieves an average speed of 7 m/s and a maximum speed of 8.8 m/s in real flight experiments. To the best of our knowledge, this is the first work to successfully apply an IRL framework for high-speed visual navigation of drones.
>
---
#### [replaced 002] Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.14993v2](https://arxiv.org/pdf/2511.14993v2)**

> **作者:** Vladimir Arkhipkin; Vladimir Korviakov; Nikolai Gerasimenko; Denis Parkhomenko; Viacheslav Vasilev; Alexey Letunovskiy; Nikolai Vaulin; Maria Kovaleva; Ivan Kirillov; Lev Novitskiy; Denis Koposov; Nikita Kiselev; Alexander Varlamov; Dmitrii Mikhailov; Vladimir Polovnikov; Andrey Shutkin; Julia Agafonova; Ilya Vasiliev; Anastasiia Kargapoltseva; Anna Dmitrienko; Anastasia Maltseva; Anna Averchenkova; Olga Kim; Tatiana Nikulina; Denis Dimitrov
>
> **备注:** Website: https://kandinskylab.ai/
>
> **摘要:** This report introduces Kandinsky 5.0, a family of state-of-the-art foundation models for high-resolution image and 10-second video synthesis. The framework comprises three core line-up of models: Kandinsky 5.0 Image Lite - a line-up of 6B parameter image generation models, Kandinsky 5.0 Video Lite - a fast and lightweight 2B parameter text-to-video and image-to-video models, and Kandinsky 5.0 Video Pro - 19B parameter models that achieves superior video generation quality. We provide a comprehensive review of the data curation lifecycle - including collection, processing, filtering and clustering - for the multi-stage training pipeline that involves extensive pre-training and incorporates quality-enhancement techniques such as self-supervised fine-tuning (SFT) and reinforcement learning (RL)-based post-training. We also present novel architectural, training, and inference optimizations that enable Kandinsky 5.0 to achieve high generation speeds and state-of-the-art performance across various tasks, as demonstrated by human evaluation. As a large-scale, publicly available generative framework, Kandinsky 5.0 leverages the full potential of its pre-training and subsequent stages to be adapted for a wide range of generative applications. We hope that this report, together with the release of our open-source code and training checkpoints, will substantially advance the development and accessibility of high-quality generative models for the research community.
>
---
#### [replaced 003] Phased One-Step Adversarial Equilibrium for Video Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.21019v2](https://arxiv.org/pdf/2508.21019v2)**

> **作者:** Jiaxiang Cheng; Bing Ma; Xuhua Ren; Hongyi Henry Jin; Kai Yu; Peng Zhang; Wenyue Li; Yuan Zhou; Tianxiang Zheng; Qinglin Lu
>
> **备注:** Accepted in AAAI 2026. Renamed from POSE to V-PAE to avoid ambiguity. Project Page: https://v-pae.github.io/
>
> **摘要:** Video diffusion generation suffers from critical sampling efficiency bottlenecks, particularly for large-scale models and long contexts. Existing video acceleration methods, adapted from image-based techniques, lack a single-step distillation ability for large-scale video models and task generalization for conditional downstream tasks. To bridge this gap, we propose the Video Phased Adversarial Equilibrium (V-PAE), a distillation framework that enables high-quality, single-step video generation from large-scale video models. Our approach employs a two-phase process. (i) Stability priming is a warm-up process to align the distributions of real and generated videos. It improves the stability of single-step adversarial distillation in the following process. (ii) Unified adversarial equilibrium is a flexible self-adversarial process that reuses generator parameters for the discriminator backbone. It achieves a co-evolutionary adversarial equilibrium in the Gaussian noise space. For the conditional tasks, we primarily preserve video-image subject consistency, which is caused by semantic degradation and conditional frame collapse during the distillation training in image-to-video (I2V) generation. Comprehensive experiments on VBench-I2V demonstrate that V-PAE outperforms existing acceleration methods by an average of 5.8% in the overall quality score, including semantic alignment, temporal coherence, and frame quality. In addition, our approach reduces the diffusion latency of the large-scale video model (e.g., Wan2.1-I2V-14B) by 100 times, while preserving competitive performance.
>
---
#### [replaced 004] Conan: Progressive Learning to Reason Like a Detective over Multi-Scale Visual Evidence
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.20470v2](https://arxiv.org/pdf/2510.20470v2)**

> **作者:** Kun Ouyang; Yuanxin Liu; Linli Yao; Yishuo Cai; Hao Zhou; Jie Zhou; Fandong Meng; Xu Sun
>
> **摘要:** Video reasoning, which requires multi-step deduction across frames, remains a major challenge for multimodal large language models (MLLMs). While reinforcement learning (RL)-based methods enhance reasoning capabilities, they often rely on text-only chains that yield ungrounded or hallucinated conclusions. Conversely, frame-retrieval approaches introduce visual grounding, yet still struggle with inaccurate evidence localization. To address these limitations, we present Conan, a framework for evidence-grounded multi-step video reasoning. Conan identifies context and evidence frames, reasons over cross-frame clues, and adaptively decides when to conclude or explore further. To achieve this, we 1) construct Conan-91K, a large-scale dataset of automatically generated reasoning traces that include frame identification, evidence reasoning, and action decision, and 2) design a multi-stage progressive cold-start strategy combined with an Identification-Reasoning-Action (AIR) RLVR training framework to progressively incentivize multi-step visual reasoning. Extensive experiments on six multi-step reasoning benchmarks demonstrate that Conan surpasses the baseline Qwen2.5-VL-7B-Instruct by an average of over 10% in accuracy, achieving state-of-the-art performance. Furthermore, Conan generalizes effectively to long video understanding tasks, validating its strong scalability and robustness.
>
---
#### [replaced 005] Rep-GLS: Report-Guided Generalized Label Smoothing for Robust Disease Detection
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02495v3](https://arxiv.org/pdf/2508.02495v3)**

> **作者:** Kunyu Zhang; Fukang Ge; Binyang Wang; Yingke Chen; Kazuma Kobayashi; Lin Gu; Jinhao Bi; Yingying Zhu
>
> **摘要:** Unlike nature image classification where groundtruth label is explicit and of no doubt, physicians commonly interpret medical image conditioned on certainty like using phrase "probable" or "likely". Existing medical image datasets either simply overlooked the nuance and polarise into binary label. Here, we propose a novel framework that leverages a Large Language Model (LLM) to directly mine medical reports to utilise the uncertainty relevant expression for supervision signal. At first, we collect uncertainty keywords from medical reports. Then, we use Qwen-3 4B to identify the textual uncertainty and map them into an adaptive Generalized Label Smoothing (GLS) rate. This rate allows our model to treat uncertain labels not as errors, but as informative signals, effectively incorporating expert skepticism into the training process. We establish a new clinical expert uncertainty-aware benchmark to rigorously evaluate this problem. Experiments demonstrate that our approach significantly outperforms state-of-the-art methods in medical disease detection. The curated uncertainty words database, code, and benchmark will be made publicly available upon acceptance.
>
---
#### [replaced 006] Enhancing efficiency in paediatric brain tumour segmentation using a pathologically diverse single-center clinical dataset
- **分类: cs.CV; physics.med-ph**

- **链接: [https://arxiv.org/pdf/2507.22152v2](https://arxiv.org/pdf/2507.22152v2)**

> **作者:** A. Piffer; J. A. Buchner; A. G. Gennari; P. Grehten; S. Sirin; E. Ross; I. Ezhov; M. Rosier; J. C. Peeken; M. Piraud; B. Menze; A. Guerreiro Stücklin; A. Jakab; F. Kofler
>
> **备注:** A. Jakab and F. Kofler have shared last authorship
>
> **摘要:** Background Brain tumours are the most common solid malignancies in children, encompassing diverse histological, molecular subtypes and imaging features and outcomes. Paediatric brain tumours (PBTs), including high- and low-grade gliomas (HGG, LGG), medulloblastomas (MB), ependymomas, and rarer forms, pose diagnostic and therapeutic challenges. Deep learning (DL)-based segmentation offers promising tools for tumour delineation, yet its performance across heterogeneous PBT subtypes and MRI protocols remains uncertain. Methods A retrospective single-centre cohort of 174 paediatric patients with HGG, LGG, medulloblastomas (MB), ependymomas, and other rarer subtypes was used. MRI sequences included T1, T1 post-contrast (T1-C), T2, and FLAIR. Manual annotations were provided for four tumour subregions: whole tumour (WT), T2-hyperintensity (T2H), enhancing tumour (ET), and cystic component (CC). A 3D nnU-Net model was trained and tested (121/53 split), with segmentation performance assessed using the Dice similarity coefficient (DSC) and compared against intra- and inter-rater variability. Results The model achieved robust performance for WT and T2H (mean DSC: 0.85), comparable to human annotator variability (mean DSC: 0.86). ET segmentation was moderately accurate (mean DSC: 0.75), while CC performance was poor. Segmentation accuracy varied by tumour type, MRI sequence combination, and location. Notably, T1, T1-C, and T2 alone produced results nearly equivalent to the full protocol. Conclusions DL is feasible for PBTs, particularly for T2H and WT. Challenges remain for ET and CC segmentation, highlighting the need for further refinement. These findings support the potential for protocol simplification and automation to enhance volumetric assessment and streamline paediatric neuro-oncology workflows.
>
---
#### [replaced 007] Active Measurement: Efficient Estimation at Scale
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.01372v2](https://arxiv.org/pdf/2507.01372v2)**

> **作者:** Max Hamilton; Jinlin Lai; Wenlong Zhao; Subhransu Maji; Daniel Sheldon
>
> **备注:** NeurIPS 2025
>
> **摘要:** AI has the potential to transform scientific discovery by analyzing vast datasets with little human effort. However, current workflows often do not provide the accuracy or statistical guarantees that are needed. We introduce active measurement, a human-in-the-loop AI framework for scientific measurement. An AI model is used to predict measurements for individual units, which are then sampled for human labeling using importance sampling. With each new set of human labels, the AI model is improved and an unbiased Monte Carlo estimate of the total measurement is refined. Active measurement can provide precise estimates even with an imperfect AI model, and requires little human effort when the AI model is very accurate. We derive novel estimators, weighting schemes, and confidence intervals, and show that active measurement reduces estimation error compared to alternatives in several measurement tasks.
>
---
#### [replaced 008] OWT: A Foundational Organ-Wise Tokenization Framework for Medical Imaging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.04899v2](https://arxiv.org/pdf/2505.04899v2)**

> **作者:** Sifan Song; Siyeop Yoon; Pengfei Jin; Sekeun Kim; Matthew Tivnan; Yujin Oh; Runqi Meng; Ling Chen; Zhiliang Lyu; Dufan Wu; Ning Guo; Xiang Li; Quanzheng Li
>
> **摘要:** Recent advances in representation learning often rely on holistic embeddings that entangle multiple semantic components, limiting interpretability and generalization. These issues are especially critical in medical imaging, where downstream tasks depend on anatomically interpretable features. To address these limitations, we propose an Organ-Wise Tokenization (OWT) framework with a Token Group-based Reconstruction (TGR) training paradigm. Unlike conventional approaches, OWT explicitly disentangles an image into separable token groups, each corresponding to a distinct organ or semantic entity. Our design ensures each token group encapsulates organ-specific information, boosting interpretability, generalization, and efficiency while enabling fine-grained control for targeted clinical applications. Experiments on CT and MRI datasets demonstrate OWT's power: it not only achieves strong performance on standard tasks like image reconstruction and segmentation, but also unlocks novel, high-impact clinical capabilities including organ-specific tumor identification, organ-level retrieval and semantic-level generation, without requiring any additional training. These findings underscore the potential of OWT as a foundational framework for semantically disentangled representation learning, offering broad scalability and a new perspective on how representations can be leveraged.
>
---
#### [replaced 009] Efficient Architectures for High Resolution Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.02584v2](https://arxiv.org/pdf/2501.02584v2)**

> **作者:** Miguel Carvalho; Bruno Martins
>
> **备注:** Accepted at COLING 2025
>
> **摘要:** Vision-Language Models (VLMs) have recently experienced significant advancements. However, challenges persist in the accurate recognition of fine details within high resolution images, which limits performance in multiple tasks. This work introduces Pheye, a novel architecture that efficiently processes high-resolution images while training fewer parameters than similarly sized VLMs. Notably, Pheye achieves a high efficiency while maintaining strong performance, particularly in tasks that demand fine-grained image understanding and/or the handling of scene-text.
>
---
#### [replaced 010] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.22444v2](https://arxiv.org/pdf/2505.22444v2)**

> **作者:** Liyao Tang; Zhe Chen; Dacheng Tao
>
> **备注:** Neurips 2025; available at https://github.com/LiyaoTang/GEM
>
> **摘要:** The emergence of large-scale pre-trained point cloud models has significantly advanced 3D scene understanding, but adapting these models to specific downstream tasks typically demands full fine-tuning, incurring high computational and storage costs. Parameter-efficient fine-tuning (PEFT) techniques, successful in natural language processing and 2D vision tasks, would underperform when naively applied to 3D point cloud models due to significant geometric and spatial distribution shifts. Existing PEFT methods commonly treat points as orderless tokens, neglecting important local spatial structures and global geometric contexts in 3D modeling. To bridge this gap, we introduce the Geometric Encoding Mixer (GEM), a novel geometry-aware PEFT module specifically designed for 3D point cloud transformers. GEM explicitly integrates fine-grained local positional encodings with a lightweight latent attention mechanism to capture comprehensive global context, thereby effectively addressing the spatial and geometric distribution mismatch. Extensive experiments demonstrate that GEM achieves performance comparable to or sometimes even exceeding full fine-tuning, while only updating 1.6% of the model's parameters, fewer than other PEFT methods. With significantly reduced training time and memory requirements, our approach thus sets a new benchmark for efficient, scalable, and geometry-aware fine-tuning of large-scale 3D point cloud models. Code is available at https://github.com/LiyaoTang/GEM.
>
---
#### [replaced 011] Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models
- **分类: cs.CR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.09201v3](https://arxiv.org/pdf/2508.09201v3)**

> **作者:** Shuang Liang; Zhihao Xu; Jialing Tao; Hui Xue; Xiting Wang
>
> **备注:** 16 pages; Previously this version appeared as arXiv:2510.15430 which was submitted as a new work by accident
>
> **摘要:** Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.
>
---
#### [replaced 012] A Decade of You Only Look Once (YOLO) for Object Detection: A Review
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.18586v3](https://arxiv.org/pdf/2504.18586v3)**

> **作者:** Leo Thomas Ramos; Angel D. Sappa
>
> **摘要:** This review marks the tenth anniversary of You Only Look Once (YOLO), one of the most influential frameworks in real-time object detection. Over the past decade, YOLO has evolved from a streamlined detector into a diverse family of architectures characterized by efficient design, modular scalability, and cross-domain adaptability. The paper presents a technical overview of the main versions (from YOLOv1 to YOLOv13), highlights key architectural trends, and surveys the principal application areas in which YOLO has been adopted. It also addresses evaluation practices, ethical considerations, and potential future directions for the framework's continued development. The analysis aims to provide a comprehensive and critical perspective on YOLO's trajectory and ongoing transformation.
>
---
#### [replaced 013] Seeing Beyond Haze: Generative Nighttime Image Dehazing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.08073v2](https://arxiv.org/pdf/2503.08073v2)**

> **作者:** Beibei Lin; Stephen Lin; Robby Tan
>
> **摘要:** Nighttime image dehazing is particularly challenging when dense haze and intense glow severely degrade or entirely obscure background information. Existing methods often struggle due to insufficient background priors and limited generative capability, both of which are highly important under such conditions. In this paper, we introduce BeyondHaze, a generative nighttime dehazing method that not only reduces haze and glow effects but also reconstructs plausible background structures in regions where visual cues are heavily degraded. Our approach is built on two main ideas: obtaining strong background priors by adapting image diffusion models to nighttime dehazing, and enhancing generative ability in haze- and glow-obscured areas through guided training. Task-specific nighttime dehazing knowledge is distilled into an image diffusion model while preserving its capacity to generate clean images. The diffusion model is further trained on tailored image pairs to improve its ability to recover background details that are suppressed by haze effects. Since generative models may introduce hallucinated content, we design our framework to allow user control over the generative level, enabling a balance between visual realism and fidelity. Experiments on real-world nighttime images demonstrate that BeyondHaze substantially improves visibility and scene detail under dense haze.
>
---
#### [replaced 014] TC-Light: Temporally Coherent Generative Rendering for Realistic World Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.18904v3](https://arxiv.org/pdf/2506.18904v3)**

> **作者:** Yang Liu; Chuanchen Luo; Zimo Tang; Yingyan Li; Yuran Yang; Yuanyong Ning; Lue Fan; Junran Peng; Zhaoxiang Zhang
>
> **备注:** Project Page: https://dekuliutesla.github.io/tclight/ Code: https://github.com/Linketic/TC-Light
>
> **摘要:** Illumination and texture editing are critical dimensions for world-to-world transfer, which is valuable for applications including sim2real and real2real visual data scaling up for embodied AI. Existing techniques generatively re-render the input video to realize the transfer, such as video relighting models and conditioned world generation models. Nevertheless, these models are predominantly limited to the domain of training data (e.g., portrait) or fall into the bottleneck of temporal consistency and computation efficiency, especially when the input video involves complex dynamics and long durations. In this paper, we propose TC-Light, a novel generative renderer to overcome these problems. Starting from the video preliminarily relighted by an inflated video relighting model, it optimizes appearance embedding in the first stage to align global illumination. Then it optimizes the proposed canonical video representation, i.e., Unique Video Tensor (UVT), to align fine-grained texture and lighting in the second stage. To comprehensively evaluate performance, we also establish a long and highly dynamic video benchmark. Extensive experiments show that our method enables physically plausible re-rendering results with superior temporal coherence and low computation cost. The code and video demos are available at https://dekuliutesla.github.io/tclight/.
>
---
#### [replaced 015] Fusion of Multi-scale Heterogeneous Pathology Foundation Models for Whole Slide Image Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.27237v2](https://arxiv.org/pdf/2510.27237v2)**

> **作者:** Zhidong Yang; Xiuhui Shi; Wei Ba; Zhigang Song; Haijing Luan; Taiyuan Hu; Senlin Lin; Jiguang Wang; Shaohua Kevin Zhou; Rui Yan
>
> **备注:** 22 pages, 9 figures
>
> **摘要:** Whole slide image (WSI) analysis has emerged as an increasingly essential technique in computational pathology. Recent advances in the pathology foundation models (FMs) have demonstrated significant advantages in deriving meaningful patch-level or slide-level multi-scale features from WSIs. However, current pathology FMs have exhibited substantial heterogeneity caused by diverse private training datasets and different network architectures. This heterogeneity introduces performance variability when we utilize the features from different FMs in the downstream tasks. To fully explore the advantages of multiple FMs effectively, in this work, we propose a novel framework for the fusion of multi-scale heterogeneous pathology FMs, called FuseCPath, yielding a model with a superior ensemble performance. The main contributions of our framework can be summarized as follows: (i) To guarantee the representativeness of the training patches, we propose a multi-view clustering-based method to filter out the discriminative patches via multiple FMs' embeddings. (ii) To effectively fuse the patch-level FMs, we devise a cluster-level re-embedding strategy to online capture patch-level local features. (iii) To effectively fuse the slide-level FMs, we devise a collaborative distillation strategy to explore the connections between slide-level FMs. Extensive experiments demonstrate that the proposed FuseCPath achieves state-of-the-art performance across multiple tasks on diverse datasets.
>
---
#### [replaced 016] Unsupervised Discovery of Long-Term Spatiotemporal Periodic Workflows in Human Activities
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14945v2](https://arxiv.org/pdf/2511.14945v2)**

> **作者:** Fan Yang; Quanting Xie; Atsunori Moteki; Shoichi Masui; Shan Jiang; Kanji Uchino; Yonatan Bisk; Graham Neubig
>
> **备注:** accepted to WACV 2026
>
> **摘要:** Periodic human activities with implicit workflows are common in manufacturing, sports, and daily life. While short-term periodic activities -- characterized by simple structures and high-contrast patterns -- have been widely studied, long-term periodic workflows with low-contrast patterns remain largely underexplored. To bridge this gap, we introduce the first benchmark comprising 580 multimodal human activity sequences featuring long-term periodic workflows. The benchmark supports three evaluation tasks aligned with real-world applications: unsupervised periodic workflow detection, task completion tracking, and procedural anomaly detection. We also propose a lightweight, training-free baseline for modeling diverse periodic workflow patterns. Experiments show that: (i) our benchmark presents significant challenges to both unsupervised periodic detection methods and zero-shot approaches based on powerful large language models (LLMs); (ii) our baseline outperforms competing methods by a substantial margin in all evaluation tasks; and (iii) in real-world applications, our baseline demonstrates deployment advantages on par with traditional supervised workflow detection approaches, eliminating the need for annotation and retraining. Our project page is https://sites.google.com/view/periodicworkflow.
>
---
#### [replaced 017] Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding
- **分类: cs.CV; cs.CL**

- **链接: [https://arxiv.org/pdf/2509.21223v2](https://arxiv.org/pdf/2509.21223v2)**

> **作者:** Muxin Pu; Mei Kuan Lim; Chun Yong Chong; Chen Change Loy
>
> **摘要:** Pre-training has proven effective for learning transferable features in sign language understanding (SLU) tasks. Recently, skeleton-based methods have gained increasing attention because they can robustly handle variations in subjects and backgrounds without being affected by appearance or environmental factors. Current SLU methods continue to face three key limitations: 1) weak semantic grounding, as models often capture low-level motion patterns from skeletal data but struggle to relate them to linguistic meaning; 2) imbalance between local details and global context, with models either focusing too narrowly on fine-grained cues or overlooking them for broader context; and 3) inefficient cross-modal learning, as constructing semantically aligned representations across modalities remains difficult. To address these, we propose Sigma, a unified skeleton-based SLU framework featuring: 1) a sign-aware early fusion mechanism that facilitates deep interaction between visual and textual modalities, enriching visual features with linguistic context; 2) a hierarchical alignment learning strategy that jointly maximises agreements across different levels of paired features from different modalities, effectively capturing both fine-grained details and high-level semantic relationships; and 3) a unified pre-training framework that combines contrastive learning, text matching and language modelling to promote semantic consistency and generalisation. Sigma achieves new state-of-the-art results on isolated sign language recognition, continuous sign language recognition, and gloss-free sign language translation on multiple benchmarks spanning different sign and spoken languages, demonstrating the impact of semantically informative pre-training and the effectiveness of skeletal data as a stand-alone solution for SLU.
>
---
#### [replaced 018] Multimodal Evaluation of Russian-language Architectures
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15552v2](https://arxiv.org/pdf/2511.15552v2)**

> **作者:** Artem Chervyakov; Ulyana Isaeva; Anton Emelyanov; Artem Safin; Maria Tikhonova; Alexander Kharitonov; Yulia Lyakh; Petr Surovtsev; Denis Shevelev; Vildan Saburov; Vasily Konovalov; Elisei Rykov; Ivan Sviridov; Amina Miftakhova; Ilseyar Alimova; Alexander Panchenko; Alexander Kapitanov; Alena Fenogenova
>
> **摘要:** Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, we introduce Mera Multi, an open multimodal evaluation framework for Russian-spoken architectures. The benchmark is instruction-based and encompasses default text, image, audio, and video modalities, comprising 18 newly constructed evaluation tasks for both general-purpose models and modality-specific architectures (image-to-text, video-to-text, and audio-to-text). Our contributions include: (i) a universal taxonomy of multimodal abilities; (ii) 18 datasets created entirely from scratch with attention to Russian cultural and linguistic specificity, unified prompts, and metrics; (iii) baseline results for both closed-source and open-source models; (iv) a methodology for preventing benchmark leakage, including watermarking and licenses for private sets. While our current focus is on Russian, the proposed benchmark provides a replicable methodology for constructing multimodal benchmarks in typologically diverse languages, particularly within the Slavic language family.
>
---
#### [replaced 019] VisPlay: Self-Evolving Vision-Language Models from Images
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.15661v2](https://arxiv.org/pdf/2511.15661v2)**

> **作者:** Yicheng He; Chengsong Huang; Zongxia Li; Jiaxin Huang; Yonghui Yang
>
> **摘要:** Reinforcement learning (RL) provides a principled framework for improving Vision-Language Models (VLMs) on complex reasoning tasks. However, existing RL approaches often rely on human-annotated labels or task-specific heuristics to define verifiable rewards, both of which are costly and difficult to scale. We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data. Starting from a single base VLM, VisPlay assigns the model into two interacting roles: an Image-Conditioned Questioner that formulates challenging yet answerable visual questions, and a Multimodal Reasoner that generates silver responses. These roles are jointly trained with Group Relative Policy Optimization (GRPO), which incorporates diversity and difficulty rewards to balance the complexity of generated questions with the quality of the silver answers. VisPlay scales efficiently across two model families. When trained on Qwen2.5-VL and MiMo-VL, VisPlay achieves consistent improvements in visual reasoning, compositional generalization, and hallucination reduction across eight benchmarks, including MM-Vet and MMMU, demonstrating a scalable path toward self-evolving multimodal intelligence. The project page is available at https://bruno686.github.io/VisPlay/
>
---
#### [replaced 020] Structural-Spectral Graph Convolution with Evidential Edge Learning for Hyperspectral Image Clustering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.09920v4](https://arxiv.org/pdf/2506.09920v4)**

> **作者:** Jianhan Qi; Yuheng Jia; Hui Liu; Junhui Hou
>
> **摘要:** Hyperspectral image (HSI) clustering groups pixels into clusters without labeled data, which is an important yet challenging task. For large-scale HSIs, most methods rely on superpixel segmentation and perform superpixel-level clustering based on graph neural networks (GNNs). However, existing GNNs cannot fully exploit the spectral information of the input HSI, and the inaccurate superpixel topological graph may lead to the confusion of different class semantics during information aggregation. To address these challenges, we first propose a structural-spectral graph convolutional operator (SSGCO) tailored for graph-structured HSI superpixels to improve their representation quality through the co-extraction of spatial and spectral features. Second, we propose an evidence-guided adaptive edge learning (EGAEL) module that adaptively predicts and refines edge weights in the superpixel topological graph. We integrate the proposed method into a contrastive learning framework to achieve clustering, where representation learning and clustering are simultaneously conducted. Experiments demonstrate that the proposed method improves clustering accuracy by 2.61%, 6.06%, 4.96% and 3.15% over the best compared methods on four HSI datasets. Our code is available at https://github.com/jhqi/SSGCO-EGAEL.
>
---
#### [replaced 021] LSAP: Rethinking Inversion Fidelity, Perception and Editability in GAN Latent Space
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2209.12746v3](https://arxiv.org/pdf/2209.12746v3)**

> **作者:** Xuekun Zhao; Pu Cao; Xiaoya Yang; Mingjian Zhang; Lu Yang; Qing Song
>
> **备注:** under review
>
> **摘要:** As research on image inversion advances, the process is generally divided into two stages. The first step is Image Embedding, involves using an encoder or optimization procedure to embed an image and obtain its corresponding latent code. The second stage, referred to as Result Refinement, further improves the inversion and editing outcomes. Although this refinement stage substantially enhances reconstruction fidelity, perception and editability remain largely unchanged and are highly dependent on the latent codes derived from the first stage. Therefore, a key challenge lies in obtaining latent codes that preserve reconstruction fidelity while simultaneously improving perception and editability. In this work, we first reveal that these two properties are closely related to the degree of alignment (or disalignment) between the inverted latent codes and the synthetic distribution. Based on this insight, we propose the \textbf{ Latent Space Alignment Inversion Paradigm (LSAP)}, which integrates both an evaluation metric and a unified inversion solution. Specifically, we introduce the \textbf{Normalized Style Space ($\mathcal{S^N}$ space)} and \textbf{Normalized Style Space Cosine Distance (NSCD)} to quantify the disalignment of inversion methods. Moreover, our paradigm can be optimized for both encoder-based and optimization-based embeddings, providing a consistent alignment framework. Extensive experiments across various domains demonstrate that NSCD effectively captures perceptual and editable characteristics, and that our alignment paradigm achieves state-of-the-art performance in both stages of inversion.
>
---
#### [replaced 022] BrainRotViT: Transformer-ResNet Hybrid for Explainable Modeling of Brain Aging from 3D sMRI
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.15188v2](https://arxiv.org/pdf/2511.15188v2)**

> **作者:** Wasif Jalal; Md Nafiu Rahman; Atif Hasan Rahman; M. Sohel Rahman
>
> **摘要:** Accurate brain age estimation from structural MRI is a valuable biomarker for studying aging and neurodegeneration. Traditional regression and CNN-based methods face limitations such as manual feature engineering, limited receptive fields, and overfitting on heterogeneous data. Pure transformer models, while effective, require large datasets and high computational cost. We propose Brain ResNet over trained Vision Transformer (BrainRotViT), a hybrid architecture that combines the global context modeling of vision transformers (ViT) with the local refinement of residual CNNs. A ViT encoder is first trained on an auxiliary age and sex classification task to learn slice-level features. The frozen encoder is then applied to all sagittal slices to generate a 2D matrix of embedding vectors, which is fed into a residual CNN regressor that incorporates subject sex at the final fully-connected layer to estimate continuous brain age. Our method achieves an MAE of 3.34 years (Pearson $r=0.98$, Spearman $ρ=0.97$, $R^2=0.95$) on validation across 11 MRI datasets encompassing more than 130 acquisition sites, outperforming baseline and state-of-the-art models. It also generalizes well across 4 independent cohorts with MAEs between 3.77 and 5.04 years. Analyses on the brain age gap (the difference between the predicted age and actual age) show that aging patterns are associated with Alzheimer's disease, cognitive impairment, and autism spectrum disorder. Model attention maps highlight aging-associated regions of the brain, notably the cerebellar vermis, precentral and postcentral gyri, temporal lobes, and medial superior frontal gyrus. Our results demonstrate that this method provides an efficient, interpretable, and generalizable framework for brain-age prediction, bridging the gap between CNN- and transformer-based approaches while opening new avenues for aging and neurodegeneration research.
>
---
#### [replaced 023] Spatial-and-Frequency-aware Restoration method for Images based on Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2401.17629v2](https://arxiv.org/pdf/2401.17629v2)**

> **作者:** Kyungsung Lee; Donggyu Lee; Myungjoo Kang
>
> **摘要:** Diffusion models have recently emerged as a promising framework for Image Restoration (IR), owing to their ability to produce high-quality reconstructions and their compatibility with established methods. Existing methods for solving noisy inverse problems in IR, considers the pixel-wise data-fidelity. In this paper, we propose SaFaRI, a spatial-and-frequency-aware diffusion model for IR with Gaussian noise. Our model encourages images to preserve data-fidelity in both the spatial and frequency domains, resulting in enhanced reconstruction quality. We comprehensively evaluate the performance of our model on a variety of noisy inverse problems, including inpainting, denoising, and super-resolution. Our thorough evaluation demonstrates that SaFaRI achieves state-of-the-art performance on both the ImageNet datasets and FFHQ datasets, outperforming existing zero-shot IR methods in terms of LPIPS and FID metrics.
>
---
#### [replaced 024] CAIRe: Cultural Attribution of Images by Retrieval-Augmented Evaluation
- **分类: cs.CV; cs.CL**

- **链接: [https://arxiv.org/pdf/2506.09109v2](https://arxiv.org/pdf/2506.09109v2)**

> **作者:** Arnav Yayavaram; Siddharth Yayavaram; Simran Khanuja; Michael Saxon; Graham Neubig
>
> **备注:** Preprint, under review
>
> **摘要:** As text-to-image models become increasingly prevalent, ensuring their equitable performance across diverse cultural contexts is critical. Efforts to mitigate cross-cultural biases have been hampered by trade-offs, including a loss in performance, factual inaccuracies, or offensive outputs. Despite widespread recognition of these challenges, an inability to reliably measure these biases has stalled progress. To address this gap, we introduce CAIRe, an evaluation metric that assesses the degree of cultural relevance of an image, given a user-defined set of labels. Our framework grounds entities and concepts in the image to a knowledge base and uses factual information to give independent graded judgments for each culture label. On a manually curated dataset of culturally salient but rare items built using language models, CAIRe surpasses all baselines by 22% F1 points. Additionally, we construct two datasets for culturally universal concepts, one comprising T2I-generated outputs and another retrieved from naturally occurring data. CAIRe achieves Pearson's correlations of 0.56 and 0.66 with human ratings on these sets, based on a 5-point Likert scale of cultural relevance. This demonstrates its strong alignment with human judgment across diverse image sources.
>
---
#### [replaced 025] MHR: Momentum Human Rig
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15586v2](https://arxiv.org/pdf/2511.15586v2)**

> **作者:** Aaron Ferguson; Ahmed A. A. Osman; Berta Bescos; Carsten Stoll; Chris Twigg; Christoph Lassner; David Otte; Eric Vignola; Fabian Prada; Federica Bogo; Igor Santesteban; Javier Romero; Jenna Zarate; Jeongseok Lee; Jinhyung Park; Jinlong Yang; John Doublestein; Kishore Venkateshan; Kris Kitani; Ladislav Kavan; Marco Dal Farra; Matthew Hu; Matthew Cioffi; Michael Fabris; Michael Ranieri; Mohammad Modarres; Petr Kadlecek; Rawal Khirodkar; Rinat Abdrashitov; Romain Prévost; Roman Rajbhandari; Ronald Mallet; Russel Pearsall; Sandy Kao; Sanjeev Kumar; Scott Parrish; Shoou-I Yu; Shunsuke Saito; Takaaki Shiratori; Te-Li Wang; Tony Tung; Yichen Xu; Yuan Dong; Yuhua Chen; Yuanlu Xu; Yuting Ye; Zhongshi Jiang
>
> **摘要:** We present MHR, a parametric human body model that combines the decoupled skeleton/shape paradigm of ATLAS with a flexible, modern rig and pose corrective system inspired by the Momentum library. Our model enables expressive, anatomically plausible human animation, supporting non-linear pose correctives, and is designed for robust integration in AR/VR and graphics pipelines.
>
---
#### [replaced 026] DiffuSyn Bench: Evaluating Vision-Language Models on Real-World Complexities with Diffusion-Generated Synthetic Benchmarks
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2406.04470v3](https://arxiv.org/pdf/2406.04470v3)**

> **作者:** Haokun Zhou; Yipeng Hong
>
> **摘要:** This study assesses the ability of Large Vision-Language Models (LVLMs) to differentiate between AI-generated and human-generated images. It introduces a new automated benchmark construction method for this evaluation. The experiment compared common LVLMs with human participants using a mixed dataset of AI and human-created images. Results showed that LVLMs could distinguish between the image types to some extent but exhibited a rightward bias, and perform significantly worse compared to humans. To build on these findings, we developed an automated benchmark construction process using AI. This process involved topic retrieval, narrative script generation, error embedding, and image generation, creating a diverse set of text-image pairs with intentional errors. We validated our method through constructing two caparable benchmarks. This study highlights the strengths and weaknesses of LVLMs in real-world understanding and advances benchmark construction techniques, providing a scalable and automatic approach for AI model evaluation.
>
---
#### [replaced 027] Enhancing Video Large Language Models with Structured Multi-Video Collaborative Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.13161v2](https://arxiv.org/pdf/2509.13161v2)**

> **作者:** Zhihao He; Tianyao He; Yun Xu; Tieyuan Chen; Huabin Liu; Chaofan Gan; Zuxuan Wu; Weiyao Lin
>
> **摘要:** Despite the prosperity of the video language model, the current pursuit of comprehensive video reasoning is thwarted by the inherent spatio-temporal incompleteness within individual videos, resulting in hallucinations and inaccuracies. A promising solution is to augment the reasoning performance with multiple related videos. However, video tokens are numerous and contain redundant information, so directly feeding the relevant video data into a large language model to enhance responses could be counterproductive. To address this challenge, we propose a multi-video collaborative framework for video language models. For efficient and flexible video representation, we establish a Video Structuring Module to represent the video's knowledge as a spatio-temporal graph. Based on the structured video representation, we design the Graph Fusion Module to fuse the structured knowledge and valuable information from related videos into the augmented graph node tokens. Finally, we construct an elaborate multi-video structured prompt to integrate the graph, visual, and textual tokens as the input to the large language model. Extensive experiments substantiate the effectiveness of our framework, showcasing its potential as a promising avenue for advancing video language models. Code will be open-sourced at https://github.com/ziHoHe/SMV-CR.
>
---
#### [replaced 028] Self-Supervised Discriminative Feature Learning for Deep Multi-View Clustering
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2103.15069v3](https://arxiv.org/pdf/2103.15069v3)**

> **作者:** Jie Xu; Yazhou Ren; Huayi Tang; Zhimeng Yang; Lili Pan; Yang Yang; Xiaorong Pu; Philip S. Yu; Lifang He
>
> **摘要:** Multi-view clustering is an important research topic due to its capability to utilize complementary information from multiple views. However, there are few methods to consider the negative impact caused by certain views with unclear clustering structures, resulting in poor multi-view clustering performance. To address this drawback, we propose self-supervised discriminative feature learning for deep multi-view clustering (SDMVC). Concretely, deep autoencoders are applied to learn embedded features for each view independently. To leverage the multi-view complementary information, we concatenate all views' embedded features to form the global features, which can overcome the negative impact of some views' unclear clustering structures. In a self-supervised manner, pseudo-labels are obtained to build a unified target distribution to perform multi-view discriminative feature learning. During this process, global discriminative information can be mined to supervise all views to learn more discriminative features, which in turn are used to update the target distribution. Besides, this unified target distribution can make SDMVC learn consistent cluster assignments, which accomplishes the clustering consistency of multiple views while preserving their features' diversity. Experiments on various types of multi-view datasets show that SDMVC outperforms 14 competitors including classic and state-of-the-art methods. The code is available at https://github.com/SubmissionsIn/SDMVC.
>
---
#### [replaced 029] VividFace: High-Quality and Efficient One-Step Diffusion For Video Face Enhancement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23584v2](https://arxiv.org/pdf/2509.23584v2)**

> **作者:** Shulian Zhang; Yong Guo; Long Peng; Ziyang Wang; Ye Chen; Wenbo Li; Xiao Zhang; Yulun Zhang; Jian Chen
>
> **摘要:** Video Face Enhancement (VFE) aims to restore high-quality facial regions from degraded video sequences, enabling a wide range of practical applications. Despite substantial progress in the field, current methods that primarily rely on video super-resolution and generative frameworks continue to face three fundamental challenges: (1) computational inefficiency caused by iterative multi-step denoising in diffusion models; (2) faithfully modeling intricate facial textures while preserving temporal consistency; and (3) limited model generalization due to the lack of high-quality face video training data. To address these challenges, we propose VividFace, a novel and efficient one-step diffusion framework for VFE. Built upon the pretrained WANX video generation model, VividFace reformulates the traditional multi-step diffusion process as a single-step flow matching paradigm that directly maps degraded inputs to high-quality outputs with significantly reduced inference time. To enhance facial detail recovery, we introduce a Joint Latent-Pixel Face-Focused Training strategy that constructs spatiotemporally aligned facial masks to guide optimization toward critical facial regions in both latent and pixel spaces. Furthermore, we develop an MLLM-driven automated filtering pipeline that produces MLLM-Face90, a meticulously curated high-quality face video dataset, ensuring models learn from photorealistic facial textures. Extensive experiments demonstrate that VividFace achieves superior performance in perceptual quality, identity preservation, and temporal consistency across both synthetic and real-world benchmarks. We will publicly release our code, models, and dataset to support future research.
>
---
#### [replaced 030] VIDSTAMP: A Temporally-Aware Watermark for Ownership and Integrity in Video Diffusion Models
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.01406v2](https://arxiv.org/pdf/2505.01406v2)**

> **作者:** Mohammadreza Teymoorianfard; Siddarth Sitaraman; Shiqing Ma; Amir Houmansadr
>
> **摘要:** Video diffusion models can generate realistic and temporally consistent videos. This raises concerns about provenance, ownership, and integrity. Watermarking can help address these issues by embedding metadata directly into the content. To work well, a watermark needs enough capacity for meaningful metadata. It must also stay imperceptible and remain robust to common video manipulations. Existing methods struggle with limited capacity, extra inference cost, or reduced visual quality. We introduce VidStamp, a watermarking framework that embeds frame-level messages through the decoder of a latent video diffusion model. The decoder is fine-tuned in two stages. The first stage uses static image datasets to encourage spatial message separation. The second stage uses synthesized video sequences to restore temporal consistency. This approach enables high-capacity watermarks with minimal perceptual impact. VidStamp also supports dynamic watermarking through a control signal that selects message templates during inference. This adds flexibility and creates a second channel for communication. We evaluate VidStamp on Stable Video Diffusion (I2V), OpenSora, and Wan (T2V). The system embeds 48 bits per frame while preserving visual quality and staying robust to common distortions. Compared with VideoSeal, VideoShield, and RivaGAN, it achieves lower log P-values and stronger detectability. Its frame-wise watermarking design also enables precise temporal tamper localization, with an accuracy of 0.96, which exceeds the VideoShield baseline. Code: https://github.com/SPIN-UMass/VidStamp
>
---
#### [replaced 031] From Play to Replay: Composed Video Retrieval for Temporally Fine-Grained Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.05274v2](https://arxiv.org/pdf/2506.05274v2)**

> **作者:** Animesh Gupta; Jay Parmar; Ishan Rajendrakumar Dave; Mubarak Shah
>
> **摘要:** Composed Video Retrieval (CoVR) retrieves a target video given a query video and a modification text describing the intended change. Existing CoVR benchmarks emphasize appearance shifts or coarse event changes and therefore do not test the ability to capture subtle, fast-paced temporal differences. We introduce TF-CoVR, the first large-scale benchmark dedicated to temporally fine-grained CoVR. TF-CoVR focuses on gymnastics and diving, and provides 180K triplets drawn from FineGym and FineDiving datasets. Previous CoVR benchmarks, focusing on temporal aspect, link each query to a single target segment taken from the same video, limiting practical usefulness. In TF-CoVR, we instead construct each <query, modification> pair by prompting an LLM with the label differences between clips drawn from different videos; every pair is thus associated with multiple valid target videos (3.9 on average), reflecting real-world tasks such as sports-highlight generation. To model these temporal dynamics, we propose TF-CoVR-Base, a concise two-stage training framework: (i) pre-train a video encoder on fine-grained action classification to obtain temporally discriminative embeddings; (ii) align the composed query with candidate videos using contrastive learning. We conduct the first comprehensive study of image, video, and general multimodal embedding (GME) models on temporally fine-grained composed retrieval in both zero-shot and fine-tuning regimes. On TF-CoVR, TF-CoVR-Base improves zero-shot mAP@50 from 5.92 (LanguageBind) to 7.51, and after fine-tuning raises the state-of-the-art from 19.83 to 27.22.
>
---
#### [replaced 032] Segmenting Collision Sound Sources in Egocentric Videos
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2511.13863v2](https://arxiv.org/pdf/2511.13863v2)**

> **作者:** Kranti Kumar Parida; Omar Emara; Hazel Doughty; Dima Damen
>
> **备注:** Webpage: https://krantiparida.github.io/projects/cs3.html
>
> **摘要:** Humans excel at multisensory perception and can often recognise object properties from the sound of their interactions. Inspired by this, we propose the novel task of Collision Sound Source Segmentation (CS3), where we aim to segment the objects responsible for a collision sound in visual input (i.e. video frames from the collision clip), conditioned on the audio. This task presents unique challenges. Unlike isolated sound events, a collision sound arises from interactions between two objects, and the acoustic signature of the collision depends on both. We focus on egocentric video, where sounds are often clear, but the visual scene is cluttered, objects are small, and interactions are brief. To address these challenges, we propose a weakly-supervised method for audio-conditioned segmentation, utilising foundation models (CLIP and SAM2). We also incorporate egocentric cues, i.e. objects in hands, to find acting objects that can potentially be collision sound sources. Our approach outperforms competitive baselines by $3\times$ and $4.7\times$ in mIoU on two benchmarks we introduce for the CS3 task: EPIC-CS3 and Ego4D-CS3.
>
---
#### [replaced 033] CD-DPE: Dual-Prompt Expert Network based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14014v2](https://arxiv.org/pdf/2511.14014v2)**

> **作者:** Xianming Gu; Lihui Wang; Ying Cao; Zeyu Deng; Yingfeng Ou; Guodong Hu; Yi Chen
>
> **备注:** This paper has been accepted by AAAI, but due to the final camera-ready version not being finalized, there are still some expression errors. It will be re-published after correction
>
> **摘要:** Multi-contrast magnetic resonance imaging (MRI) super-resolution intends to reconstruct high-resolution (HR) images from low-resolution (LR) scans by leveraging structural information present in HR reference images acquired with different contrasts. This technique enhances anatomical detail and soft tissue differentiation, which is vital for early diagnosis and clinical decision-making. However, inherent contrasts disparities between modalities pose fundamental challenges in effectively utilizing reference image textures to guide target image reconstruction, often resulting in suboptimal feature integration. To address this issue, we propose a dual-prompt expert network based on a convolutional dictionary feature decoupling (CD-DPE) strategy for multi-contrast MRI super-resolution. Specifically, we introduce an iterative convolutional dictionary feature decoupling module (CD-FDM) to separate features into cross-contrast and intra-contrast components, thereby reducing redundancy and interference. To fully integrate these features, a novel dual-prompt feature fusion expert module (DP-FFEM) is proposed. This module uses a frequency prompt to guide the selection of relevant reference features for incorporation into the target image, while an adaptive routing prompt determines the optimal method for fusing reference and target features to enhance reconstruction quality. Extensive experiments on public multi-contrast MRI datasets demonstrate that CD-DPE outperforms state-of-the-art methods in reconstructing fine details. Additionally, experiments on unseen datasets demonstrated that CD-DPE exhibits strong generalization capabilities.
>
---
#### [replaced 034] Simple Lines, Big Ideas: Towards Interpretable Assessment of Human Creativity from Drawings
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12880v2](https://arxiv.org/pdf/2511.12880v2)**

> **作者:** Zihao Lin; Zhenshan Shi; Sasa Zhao; Hanwei Zhu; Lingyu Zhu; Baoliang Chen; Lei Mo
>
> **备注:** We updated the version, expanding related work (acknowledging Nath et al., 2025, Pencils to Pixels: A Systematic Study of Creative Drawings) and clarifying how our model builds upon the content-style framework
>
> **摘要:** Assessing human creativity through visual outputs, such as drawings, plays a critical role in fields including psychology, education, and cognitive science. However, current assessment practices still rely heavily on expert-based subjective scoring, which is both labor-intensive and inherently subjective. In this paper, we propose a data-driven framework for automatic and interpretable creativity assessment from drawings. Motivated by the cognitive evidence proposed in [6] that creativity can emerge from both what is drawn (content) and how it is drawn (style), we reinterpret the creativity score as a function of these two complementary dimensions. Specifically, we first augment an existing creativity-labeled dataset with additional annotations targeting content categories. Based on the enriched dataset, we further propose a conditional model predicting content, style, and ratings simultaneously. In particular, the conditional learning mechanism that enables the model to adapt its visual feature extraction by dynamically tuning it to creativity-relevant signals conditioned on the drawing's stylistic and semantic cues. Experimental results demonstrate that our model achieves state-of-the-art performance compared to existing regression-based approaches and offers interpretable visualizations that align well with human judgments. The code and annotations will be made publicly available at https://github.com/WonderOfU9/CSCA_PRCV_2025
>
---
#### [replaced 035] Localized Region Guidance for Class Activation Mapping in WSSS
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.12496v2](https://arxiv.org/pdf/2509.12496v2)**

> **作者:** Ali Torabi; Sanjog Gaihre; MD Mahbubur Rahman; Yaqoob Majeed
>
> **摘要:** Weakly Supervised Semantic Segmentation (WSSS) addresses the challenge of training segmentation models using only image-level annotations. Existing WSSS methods struggle with precise object boundary localization and focus only on the most discriminative regions. To address these challenges, we propose IG-CAM (Instance-Guided Class Activation Mapping), a novel approach that leverages instance-level cues and influence functions to generate high-quality, boundary-aware localization maps. Our method introduces three key innovations: (1) Instance-Guided Refinement using object proposals to guide CAM generation, ensuring complete object coverage; (2) Influence Function Integration that captures the relationship between training samples and model predictions; and (3) Multi-Scale Boundary Enhancement with progressive refinement strategies. IG-CAM achieves state-of-the-art performance on PASCAL VOC 2012 with 82.3% mIoU before post-processing, improving to 86.6% after CRF refinement, significantly outperforming previous WSSS methods. Extensive ablation studies validate each component's contribution, establishing IG-CAM as a new benchmark for weakly supervised semantic segmentation.
>
---
#### [replaced 036] UINO-FSS: Unifying Representation Learning and Few-shot Segmentation via Hierarchical Distillation and Mamba-HyperCorrelation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.15669v4](https://arxiv.org/pdf/2504.15669v4)**

> **作者:** Wei Zhuo; Zhiyue Tang; Wufeng Xue; Hao Ding; Junkai Ji; Linlin Shen
>
> **摘要:** Few-shot semantic segmentation has attracted growing interest for its ability to generalize to novel object categories using only a few annotated samples. To address data scarcity, recent methods incorporate multiple foundation models to improve feature transferability and segmentation performance. However, they often rely on dual-branch architectures that combine pre-trained encoders to leverage complementary strengths, a design that limits flexibility and efficiency. This raises a fundamental question: can we build a unified model that integrates knowledge from different foundation architectures? Achieving this is, however, challenging due to the misalignment between class-agnostic segmentation capabilities and fine-grained discriminative representations. To this end, we present UINO-FSS, a novel framework built on the key observation that early-stage DINOv2 features exhibit distribution consistency with SAM's output embeddings. This consistency enables the integration of both models' knowledge into a single-encoder architecture via coarse-to-fine multimodal distillation. In particular, our segmenter consists of three core components: a bottleneck adapter for embedding alignment, a meta-visual prompt generator that leverages dense similarity volumes and semantic embeddings, and a mask decoder. Using hierarchical cross-model distillation, we effectively transfer SAM's knowledge into the segmenter, further enhanced by Mamba-based 4D correlation mining on support-query pairs. Extensive experiments on PASCAL-5$^i$ and COCO-20$^i$ show that UINO-FSS achieves new state-of-the-art results under the 1-shot setting, with mIoU of 80.6 (+3.8%) on PASCAL-5$^i$ and 64.5 (+4.1%) on COCO-20$^i$, demonstrating the effectiveness of our unified approach.
>
---
#### [replaced 037] Label-Efficient Cross-Modality Generalization for Liver Segmentation in Multi-Phase MRI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04705v3](https://arxiv.org/pdf/2510.04705v3)**

> **作者:** Quang-Khai Bui-Tran; Minh-Toan Dinh; Thanh-Huy Nguyen; Ba-Thinh Lam; Mai-Anh Vu; Ulas Bagci
>
> **备注:** Accepted at MICCAI 2025 Workshop
>
> **摘要:** Accurate liver segmentation in multi-phase MRI is vital for liver fibrosis assessment, yet labeled data is often scarce and unevenly distributed across imaging modalities and vendor systems. We propose a label-efficient segmentation approach that promotes cross-modality generalization under real-world conditions, where GED4 hepatobiliary-phase annotations are limited, non-contrast sequences (T1WI, T2WI, DWI) are unlabeled, and spatial misalignment and missing phases are common. Our method integrates a foundation-scale 3D segmentation backbone adapted via fine-tuning, co-training with cross pseudo supervision to leverage unlabeled volumes, and a standardized preprocessing pipeline. Without requiring spatial registration, the model learns to generalize across MRI phases and vendors, demonstrating robust segmentation performance in both labeled and unlabeled domains. Our results exhibit the effectiveness of our proposed label-efficient baseline for liver segmentation in multi-phase, multi-vendor MRI and highlight the potential of combining foundation model adaptation with co-training for real-world clinical imaging tasks.
>
---
#### [replaced 038] SpeeDe3DGS: Speedy Deformable 3D Gaussian Splatting with Temporal Pruning and Motion Grouping
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.07917v3](https://arxiv.org/pdf/2506.07917v3)**

> **作者:** Allen Tu; Haiyang Ying; Alex Hanson; Yonghan Lee; Tom Goldstein; Matthias Zwicker
>
> **备注:** Project Page: https://speede3dgs.github.io/
>
> **摘要:** Dynamic extensions of 3D Gaussian Splatting (3DGS) achieve high-quality reconstructions through neural motion fields, but per-Gaussian neural inference makes these models computationally expensive. Building on DeformableGS, we introduce Speedy Deformable 3D Gaussian Splatting (SpeeDe3DGS), which bridges this efficiency-fidelity gap through three complementary modules: Temporal Sensitivity Pruning (TSP) removes low-impact Gaussians via temporally aggregated sensitivity analysis, Temporal Sensitivity Sampling (TSS) perturbs timestamps to suppress floaters and improve temporal coherence, and GroupFlow distills the learned deformation field into shared SE(3) transformations for efficient groupwise motion. On the 50 dynamic scenes in MonoDyGauBench, integrating TSP and TSS into DeformableGS accelerates rendering by 6.78$\times$ on average while maintaining neural-field fidelity and using 10$\times$ fewer primitives. Adding GroupFlow culminates in 13.71$\times$ faster rendering and 2.53$\times$ shorter training, surpassing all baselines in speed while preserving superior image quality.
>
---
#### [replaced 039] Shape and Texture Recognition in Large Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.23062v4](https://arxiv.org/pdf/2503.23062v4)**

> **作者:** Sagi Eppel; Mor Bismut; Alona Faktor-Strugatski
>
> **摘要:** Shapes and textures are the basic building blocks of visual perception. The ability to identify shapes regardless of orientation, texture, or context, and to recognize textures and materials independently of their associated objects, is essential for a general visual understanding of the world. This work introduces the Large Shape and Textures dataset (LAS&T), a giant collection of highly diverse shapes and textures, created by unsupervised extraction of patterns from natural images. This dataset is used to benchmark how effectively leading Large Vision-Language Models (VLM) recognize and represent shapes, textures, and materials in 2D and 3D scenes. For shape recognition, we test the models' ability to match images of identical shapes that differ in orientation, texture, color, or environment. Our results show that the shape recognition capabilities of the LVLMs remain significantly below human performance. VLMs rely predominantly on high-level and semantic features and struggle with abstract shapes lacking class associations. For texture and material recognition, we evaluated the models' ability to identify images with identical textures and materials across different objects and environments. Interestingly, leading LVLMs approach human-level performance in recognizing materials in 3D scenes, yet substantially underperform humans when identifying simpler, more abstract 2D textures and shapes. These results are consistent across a wide range of leading LVLMs (GPT/Gemini/Qwen) and foundation vision models (DINO/CLIP), exposing major deficiencies in the ability of leading models to extract and represent low-level visual features. In contrast, humans and simple nets trained directly for these tasks achieve high accuracy. The LAS&T dataset, featuring over 700,000 images for 2D/3D shape, texture, and material recognition and retrieval is freely available.
>
---
#### [replaced 040] One Model for All: Unified Try-On and Try-Off in Any Pose via LLM-Inspired Bidirectional Tweedie Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.04559v2](https://arxiv.org/pdf/2508.04559v2)**

> **作者:** Jinxi Liu; Zijian He; Guangrun Wang; Guanbin Li; Liang Lin
>
> **摘要:** Recent diffusion-based approaches have made significant advances in image-based virtual try-on, enabling more realistic and end-to-end garment synthesis. However, most existing methods remain constrained by their reliance on exhibition garments and segmentation masks, as well as their limited ability to handle flexible pose variations. These limitations reduce their practicality in real-world scenarios - for instance, users cannot easily transfer garments worn by one person onto another, and the generated try-on results are typically restricted to the same pose as the reference image. In this paper, we introduce OMFA (One Model For All), a unified diffusion framework for both virtual try-on and try-off that operates without the need for exhibition garments and supports arbitrary poses. OMFA is inspired by language modeling, where generation is guided by conditioning prompts. However, our framework differs fundamentally from LLMs in two key aspects. First, it employs a bidirectional modeling paradigm that symmetrically allows prompting either from the garment to generate try-on results or from the dressed person to recover the try-off garment. Second, it strictly adheres to Tweedie's formula, enabling faithful estimation of the underlying data distribution during the denoising process. Instead of imposing lower body constraints, OMFA is an entirely mask-free framework that requires only a single portrait and a target garment as input, and is designed to support flexible outfit combinations and cross-person garment transfer, making it better aligned with practical usage scenarios. Additionally, by leveraging SMPL-X-based pose conditioning, OMFA supports multi-view and arbitrary-pose try-on from just one image. Extensive experiments demonstrate that OMFA achieves state-of-the-art results on both try-on and try-off tasks, providing a practical solution for virtual garment synthesis.
>
---
#### [replaced 041] FunnyNodules: A Customizable Medical Dataset Tailored for Evaluating Explainable AI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15481v2](https://arxiv.org/pdf/2511.15481v2)**

> **作者:** Luisa Gallée; Yiheng Xiong; Meinrad Beer; Michael Götz
>
> **摘要:** Densely annotated medical image datasets that capture not only diagnostic labels but also the underlying reasoning behind these diagnoses are scarce. Such reasoning-related annotations are essential for developing and evaluating explainable AI (xAI) models that reason similarly to radiologists: making correct predictions for the right reasons. To address this gap, we introduce FunnyNodules, a fully parameterized synthetic dataset designed for systematic analysis of attribute-based reasoning in medical AI models. The dataset generates abstract, lung nodule-like shapes with controllable visual attributes such as roundness, margin sharpness, and spiculation. Target class is derived from a predefined attribute combination, allowing full control over the decision rule that links attributes to the diagnostic class. We demonstrate how FunnyNodules can be used in model-agnostic evaluations to assess whether models learn correct attribute-target relations, to interpret over- or underperformance in attribute prediction, and to analyze attention alignment with attribute-specific regions of interest. The framework is fully customizable, supporting variations in dataset complexity, target definitions, class balance, and beyond. With complete ground truth information, FunnyNodules provides a versatile foundation for developing, benchmarking, and conducting in-depth analyses of explainable AI methods in medical image analysis.
>
---
#### [replaced 042] Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06741v3](https://arxiv.org/pdf/2511.06741v3)**

> **作者:** Wenbo Huang; Jinghui Zhang; Zhenghao Chen; Guang Li; Lei Zhang; Yang Cao; Fang Dong; Takahiro Ogawa; Miki Haseyama
>
> **备注:** Accepted by AAAI 2026 Oral
>
> **摘要:** Wide-angle videos in few-shot action recognition (FSAR) effectively express actions within specific scenarios. However, without a global understanding of both subjects and background, recognizing actions in such samples remains challenging because of the background distractions. Receptance Weighted Key Value (RWKV), which learns interaction between various dimensions, shows promise for global modeling. While directly applying RWKV to wide-angle FSAR may fail to highlight subjects due to excessive background information. Additionally, temporal relation degraded by frames with similar backgrounds is difficult to reconstruct, further impacting performance. Therefore, we design the CompOund SegmenTation and Temporal REconstructing RWKV (Otter). Specifically, the Compound Segmentation Module~(CSM) is devised to segment and emphasize key patches in each frame, effectively highlighting subjects against background information. The Temporal Reconstruction Module (TRM) is incorporated into the temporal-enhanced prototype construction to enable bidirectional scanning, allowing better reconstruct temporal relation. Furthermore, a regular prototype is combined with the temporal-enhanced prototype to simultaneously enhance subject emphasis and temporal modeling, improving wide-angle FSAR performance. Extensive experiments on benchmarks such as SSv2, Kinetics, UCF101, and HMDB51 demonstrate that Otter achieves state-of-the-art performance. Extra evaluation on the VideoBadminton dataset further validates the superiority of Otter in wide-angle FSAR.
>
---
#### [replaced 043] IOR: Inversed Objects Replay for Incremental Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.04829v5](https://arxiv.org/pdf/2406.04829v5)**

> **作者:** Zijia An; Boyu Diao; Libo Huang; Ruiqi Liu; Zhulin An; Yongjun Xu
>
> **摘要:** Existing Incremental Object Detection (IOD) methods partially alleviate catastrophic forgetting when incrementally detecting new objects in real-world scenarios. However, many of these methods rely on the assumption that unlabeled old-class objects may co-occur with labeled new-class objects in the incremental data. When unlabeled old-class objects are absent, the performance of existing methods tends to degrade. The absence can be mitigated by generating old-class samples, but it incurs high costs. This paper argues that previous generation-based IOD suffers from redundancy, both in the use of generative models, which require additional training and storage, and in the overproduction of generated samples, many of which do not contribute significantly to performance improvements. To eliminate the redundancy, we propose Inversed Objects Replay (IOR). Specifically, we generate old-class samples by inversing the original detectors, thus eliminating the necessity of training and storing additional generative models. We propose augmented replay to reuse the objects in generated samples, reducing redundant generations. Moreover, we propose high-value knowledge distillation focusing on the positions of old-class objects overwhelmed by the background, which transfers the knowledge to the incremental detector. Extensive experiments conducted on MS COCO 2017 demonstrate that our method can efficiently improve detection performance in IOD scenarios with the absence of old-class objects. The code is available at https://github.com/JiaJia075/IOR.
>
---
#### [replaced 044] MagicFace: High-Fidelity Facial Expression Editing with Action-Unit Control
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.02260v3](https://arxiv.org/pdf/2501.02260v3)**

> **作者:** Mengting Wei; Tuomas Varanka; Xingxun Jiang; Huai-Qian Khor; Guoying Zhao
>
> **摘要:** We address the problem of facial expression editing by controling the relative variation of facial action-unit (AU) from the same person. This enables us to edit this specific person's expression in a fine-grained, continuous and interpretable manner, while preserving their identity, pose, background and detailed facial attributes. Key to our model, which we dub MagicFace, is a diffusion model conditioned on AU variations and an ID encoder to preserve facial details of high consistency. Specifically, to preserve the facial details with the input identity, we leverage the power of pretrained Stable-Diffusion models and design an ID encoder to merge appearance features through self-attention. To keep background and pose consistency, we introduce an efficient Attribute Controller by explicitly informing the model of current background and pose of the target. By injecting AU variations into a denoising UNet, our model can animate arbitrary identities with various AU combinations, yielding superior results in high-fidelity expression editing compared to other facial expression editing works. Code is publicly available at https://github.com/weimengting/MagicFace.
>
---
#### [replaced 045] Introducing DEFORMISE: A deep learning framework for dementia diagnosis in the elderly using optimized MRI slice selection
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2407.17324v3](https://arxiv.org/pdf/2407.17324v3)**

> **作者:** Nikolaos Ntampakis; Konstantinos Diamantaras; Ioanna Chouvarda; Vasileios Argyriou; Panagiotis Sarigianndis
>
> **摘要:** Dementia, a debilitating neurological condition affecting millions worldwide, presents significant diagnostic challenges. In this work, we introduce DEFORMISE, a novel DEep learning Framework for dementia diagnOsis of eldeRly patients using 3D brain Magnetic resonance Imaging (MRI) scans with Optimized Slice sElection. Our approach features a unique technique for selectively processing MRI slices, focusing on the most relevant brain regions and excluding less informative sections. This methodology is complemented by a confidence-based classification committee composed of three novel deep learning models. Tested on the Open OASIS datasets, our method achieved an impressive accuracy of 94.12%, surpassing existing methodologies. Furthermore, validation on the ADNI dataset confirmed the robustness and generalizability of our approach. The use of explainable AI (XAI) techniques and comprehensive ablation studies further substantiate the effectiveness of our techniques, providing insights into the decision-making process and the importance of our methodology. This research offers a significant advancement in dementia diagnosis, providing a highly accurate and efficient tool for clinical applications.
>
---
#### [replaced 046] Body-Hand Modality Expertized Networks with Cross-attention for Fine-grained Skeleton Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.14960v3](https://arxiv.org/pdf/2503.14960v3)**

> **作者:** Seungyeon Cho; Tae-Kyun Kim
>
> **备注:** 7 figures, 8 pages
>
> **摘要:** Skeleton-based Human Action Recognition (HAR) is a vital technology in robotics and human-robot interaction. However, most existing methods concentrate primarily on full-body movements and often overlook subtle hand motions that are critical for distinguishing fine-grained actions. Recent work leverages a unified graph representation that combines body, hand, and foot keypoints to capture detailed body dynamics. Yet, these models often blur fine hand details due to the disparity between body and hand action characteristics and the loss of subtle features during the spatial-pooling. In this paper, we propose BHaRNet (Body-Hand action Recognition Network), a novel framework that augments a typical body-expert model with a hand-expert model. Our model jointly trains both streams with an ensemble loss that fosters cooperative specialization, functioning in a manner reminiscent of a Mixture-of-Experts (MoE). Moreover, cross-attention is employed via an expertized branch method and a pooling-attention module to enable feature-level interactions and selectively fuse complementary information. Inspired by MMNet, we also demonstrate the applicability of our approach to multi-modal tasks by leveraging RGB information, where body features guide RGB learning to capture richer contextual cues. Experiments on large-scale benchmarks (NTU RGB+D 60, NTU RGB+D 120, PKU-MMD, and Northwestern-UCLA) demonstrate that BHaRNet achieves SOTA accuracies -- improving from 86.4\% to 93.0\% in hand-intensive actions -- while maintaining fewer GFLOPs and parameters than the relevant unified methods.
>
---
#### [replaced 047] Orion: A Unified Visual Agent for Multimodal Perception, Advanced Visual Reasoning and Execution
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.14210v2](https://arxiv.org/pdf/2511.14210v2)**

> **作者:** N Dinesh Reddy; Dylan Snyder; Lona Kiragu; Mirajul Mohin; Shahrear Bin Amin; Sudeep Pillai
>
> **摘要:** We introduce Orion, a visual agent that integrates vision-based reasoning with tool-augmented execution to achieve powerful, precise, multi-step visual intelligence across images, video, and documents. Unlike traditional vision-language models that generate descriptive outputs, Orion orchestrates a suite of specialized computer vision tools, including object detection, keypoint localization, panoptic segmentation, Optical Character Recognition (OCR), and geometric analysis, to execute complex multi-step visual workflows. The system achieves competitive performance across MMMU, MMBench, DocVQA, and MMLongBench while extending monolithic VLM capabilities to production-grade visual intelligence. Through its agentic, tool-augmented approach, Orion enables autonomous visual reasoning that bridges neural perception with symbolic execution, marking the transition from passive visual understanding to active, tool-driven visual intelligence. Try Orion for free at: https://chat.vlm.run Learn more at: https://www.vlm.run/orion
>
---
#### [replaced 048] CoT-Saliency: Unified Chain-of-Thought Reasoning for Heterogeneous Saliency Tasks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00396v2](https://arxiv.org/pdf/2511.00396v2)**

> **作者:** Long Li; Shuichen Ji; Ziyang Luo; Nian Liu; Dingwen Zhang; Junwei Han
>
> **备注:** The entire article has undergone significant changes. Many statements in the current version are not precise, and the experimental results have also changed. Considering that we cannot quickly update it in the short term, in order to prevent misleading researchers, we have decided to retract this article. We sincerely appreciate your understanding and cooperation
>
> **摘要:** We present the first unified framework that jointly handles three operationally heterogeneous saliency tasks, eg, SOD, CoSOD, and SIS, by casting each as a Chain-of-Thought (CoT) reasoning process in a Vision-Language Model (VLM) to bridge task heterogeneity. CoT training follows a two-stage paradigm: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). To enhance CoT quality in RL, we propose Confidence-Guided Policy Optimization (CGPO), a lightweight single-sample algorithm that leverages the discrepancy between reward and model confidence as a per-sample advantage signal. This design naturally focuses updates on informative responses while eliminating group sampling, thereby addressing GRPO's key limitations: confidence-agnostic learning, signal dilution, and prohibitive computational overhead. We also introduce an "output-to-reasoning" strategy to construct high-fidelity SFT data that ensures logical consistency with ground-truth masks. Experiments show our model matches or outperforms specialized SOTA methods and strong closed-source VLMs across all tasks, especially achieving an S-measure of 0.899 on CoCA for CoSOD, surpassing the prior best by 8.0 percentage points, despite using far less training data.
>
---
#### [replaced 049] System Filter-Based Common Components Modeling for Cross-Subject EEG Decoding
- **分类: q-bio.NC; cs.CV; eess.SY**

- **链接: [https://arxiv.org/pdf/2507.05268v2](https://arxiv.org/pdf/2507.05268v2)**

> **作者:** Xiaoyuan Li; Xinru Xue; Bohan Zhang; Ye Sun; Shoushuo Xi; Gang Liu
>
> **备注:** 12 pages, 11 figures
>
> **摘要:** Brain-computer interface (BCI) technology enables direct communication between the brain and external devices through electroencephalography (EEG) signals. However, existing decoding models often mix common and personalized components, leading to interference from individual variability that limits cross-subject decoding performance. To address this issue, this paper proposes a system filter that extends the concept of signal filtering to the system level. The method expands a system into its spectral representation, selectively removes unnecessary components, and reconstructs the system from the retained target components, thereby achieving explicit system-level decomposition and filtering. We further integrate the system filter into a Cross-Subject Decoding framework based on the System Filter (CSD-SF) and evaluate it on the four-class motor imagery (MI) task of the BCIC IV 2a dataset. Personalized models are transformed into relation spectrums, and statistical testing across subjects is used to remove personalized components. The remaining stable relations, representing common components across subjects, are then used to construct a common model for cross-subject decoding. Experimental results show an average improvement of 3.28% in decoding accuracy over baseline methods, demonstrating that the proposed system filter effectively isolates stable common components and enhances model robustness and generalizability in cross-subject EEG decoding.
>
---
#### [replaced 050] HAWAII: Hierarchical Visual Knowledge Transfer for Efficient Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2506.19072v2](https://arxiv.org/pdf/2506.19072v2)**

> **作者:** Yimu Wang; Mozhgan Nasr Azadani; Sean Sedwards; Krzysztof Czarnecki
>
> **备注:** NeurIPS 2025
>
> **摘要:** Improving the visual understanding ability of vision-language models (VLMs) is crucial for enhancing their performance across various tasks. While using multiple pretrained visual experts has shown great promise, it often incurs significant computational costs during training and inference. To address this challenge, we propose HAWAII, a novel framework that distills knowledge from multiple visual experts into a single vision encoder, enabling it to inherit the complementary strengths of several experts with minimal computational overhead. To mitigate conflicts among different teachers and switch between different teacher-specific knowledge, instead of using a fixed set of adapters for multiple teachers, we propose to use teacher-specific Low-Rank Adaptation (LoRA) adapters with a corresponding router. Each adapter is aligned with a specific teacher, avoiding noisy guidance during distillation. To enable efficient knowledge distillation, we propose fine-grained and coarse-grained distillation. At the fine-grained level, token importance scores are employed to emphasize the most informative tokens from each teacher adaptively. At the coarse-grained level, we summarize the knowledge from multiple teachers and transfer it to the student using a set of general-knowledge LoRA adapters with a router. Extensive experiments on various vision-language tasks demonstrate the superiority of HAWAII compared to popular open-source VLMs. The code is available at https://github.com/yimuwangcs/wise-hawaii.
>
---
#### [replaced 051] CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.16356v3](https://arxiv.org/pdf/2503.16356v3)**

> **作者:** Yunzhi Yao; Jizhan Fang; Jia-Chen Gu; Ningyu Zhang; Shumin Deng; Huajun Chen; Nanyun Peng
>
> **备注:** EMNLP 2025
>
> **摘要:** Knowledge Editing (KE) enables the modification of outdated or incorrect information in large language models (LLMs). While existing KE methods can update isolated facts, they often fail to generalize these updates to multi-hop reasoning tasks that rely on the modified knowledge. Through an analysis of reasoning circuits -- the neural pathways LLMs use for knowledge-based inference, we find that current layer-localized KE approaches (e.g., MEMIT, WISE), which edit only single or a few model layers, inadequately integrate updated knowledge into these reasoning pathways. To address this limitation, we present CaKE (Circuit-aware Knowledge Editing), a novel method that enhances the effective integration of updated knowledge in LLMs. By only leveraging a few curated data samples guided by our circuit-based analysis, CaKE stimulates the model to develop appropriate reasoning circuits for newly incorporated knowledge. Experiments show that CaKE enables more accurate and consistent use of edited knowledge across related reasoning tasks, achieving an average improvement of 20% in multi-hop reasoning accuracy on the MQuAKE dataset while requiring less memory than existing KE methods. We release the code and data in https://github.com/zjunlp/CaKE.
>
---
#### [replaced 052] Towards Metric-Aware Multi-Person Mesh Recovery by Jointly Optimizing Human Crowd in Camera Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13282v2](https://arxiv.org/pdf/2511.13282v2)**

> **作者:** Kaiwen Wang; Kaili Zheng; Yiming Shi; Chenyi Guo; Ji Wu
>
> **摘要:** Multi-person human mesh recovery from a single image is a challenging task, hindered by the scarcity of in-the-wild training data. Prevailing in-the-wild human mesh pseudo-ground-truth (pGT) generation pipelines are single-person-centric, where each human is processed individually without joint optimization. This oversight leads to a lack of scene-level consistency, producing individuals with conflicting depths and scales within the same image. To address this, we introduce Depth-conditioned Translation Optimization (DTO), a novel optimization-based method that jointly refines the camera-space translations of all individuals in a crowd. By leveraging anthropometric priors on human height and depth cues from a monocular depth estimator, DTO solves for a scene-consistent placement of all subjects within a principled Maximum a posteriori (MAP) framework. Applying DTO to the 4D-Humans dataset, we construct DTO-Humans, a new large-scale pGT dataset of 0.56M high-quality, scene-consistent multi-person images, featuring dense crowds with an average of 4.8 persons per image. Furthermore, we propose Metric-Aware HMR, an end-to-end network that directly estimates human mesh and camera parameters in metric scale. This is enabled by a camera branch and a relative metric loss that enforces plausible relative scales. Extensive experiments demonstrate that our method achieves state-of-the-art performance on relative depth reasoning and human mesh recovery. Code is available at: https://github.com/gouba2333/MA-HMR.
>
---
#### [replaced 053] Zero-Shot Video Translation via Token Warping
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2402.12099v3](https://arxiv.org/pdf/2402.12099v3)**

> **作者:** Haiming Zhu; Yangyang Xu; Jun Yu; Shengfeng He
>
> **摘要:** With the revolution of generative AI, video-related tasks have been widely studied. However, current state-of-the-art video models still lag behind image models in visual quality and user control over generated content. In this paper, we introduce TokenWarping, a novel framework for temporally coherent video translation. Existing diffusion-based video editing approaches rely solely on key and value patches in self-attention to ensure temporal consistency, often sacrificing the preservation of local and structural regions. Critically, these methods overlook the significance of the query patches in achieving accurate feature aggregation and temporal coherence. In contrast, TokenWarping leverages complementary token priors by constructing temporal correlations across different frames. Our method begins by extracting optical flows from source videos. During the denoising process of the diffusion model, these optical flows are used to warp the previous frame's query, key, and value patches, aligning them with the current frame's patches. By directly warping the query patches, we enhance feature aggregation in self-attention, while warping the key and value patches ensures temporal consistency across frames. This token warping imposes explicit constraints on the self-attention layer outputs, effectively ensuring temporally coherent translation. Our framework does not require any additional training or fine-tuning and can be seamlessly integrated with existing text-to-image editing methods. We conduct extensive experiments on various video translation tasks, demonstrating that TokenWarping surpasses state-of-the-art methods both qualitatively and quantitatively. Video demonstrations are available in supplementary materials.
>
---
#### [replaced 054] One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image
- **分类: cs.CL; cs.CR; cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2504.02132v3](https://arxiv.org/pdf/2504.02132v3)**

> **作者:** Ezzeldin Shereen; Dan Ristea; Shae McFadden; Burak Hasircioglu; Vasilios Mavroudis; Chris Hicks
>
> **摘要:** Retrieval-augmented generation (RAG) is instrumental for inhibiting hallucinations in large language models (LLMs) through the use of a factual knowledge base (KB). Although PDF documents are prominent sources of knowledge, text-based RAG pipelines are ineffective at capturing their rich multi-modal information. In contrast, visual document RAG (VD-RAG) uses screenshots of document pages as the KB, which has been shown to achieve state-of-the-art results. However, by introducing the image modality, VD-RAG introduces new attack vectors for adversaries to disrupt the system by injecting malicious documents into the KB. In this paper, we demonstrate the vulnerability of VD-RAG to poisoning attacks targeting both retrieval and generation. We define two attack objectives and demonstrate that both can be realized by injecting only a single adversarial image into the KB. Firstly, we introduce a targeted attack against one or a group of queries with the goal of spreading targeted disinformation. Secondly, we present a universal attack that, for any potential user query, influences the response to cause a denial-of-service in the VD-RAG system. We investigate the two attack objectives under both white-box and black-box assumptions, employing a multi-objective gradient-based optimization approach as well as prompting state-of-the-art generative models. Using two visual document datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (vision language models), we show VD-RAG is vulnerable to poisoning attacks in both the targeted and universal settings, yet demonstrating robustness to black-box attacks in the universal setting.
>
---
#### [replaced 055] Learning from Dense Events: Towards Fast Spiking Neural Networks Training via Event Dataset Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12095v2](https://arxiv.org/pdf/2511.12095v2)**

> **作者:** Shuhan Ye; Yi Yu; Qixin Zhang; Chenqi Kong; Qiangqiang Wu; Kun Wang; Xudong Jiang
>
> **摘要:** Event cameras sense brightness changes and output binary asynchronous event streams, attracting increasing attention. Their bio-inspired dynamics align well with spiking neural networks (SNNs), offering a promising energy-efficient alternative to conventional vision systems. However, SNNs remain costly to train due to temporal coding, which limits their practical deployment. To alleviate the high training cost of SNNs, we introduce \textbf{PACE} (Phase-Aligned Condensation for Events), the first dataset distillation framework to SNNs and event-based vision. PACE distills a large training dataset into a compact synthetic one that enables fast SNN training, which is achieved by two core modules: \textbf{ST-DSM} and \textbf{PEQ-N}. ST-DSM uses residual membrane potentials to densify spike-based features (SDR) and to perform fine-grained spatiotemporal matching of amplitude and phase (ST-SM), while PEQ-N provides a plug-and-play straight through probabilistic integer quantizer compatible with standard event-frame pipelines. Across DVS-Gesture, CIFAR10-DVS, and N-MNIST datasets, PACE outperforms existing coreset selection and dataset distillation baselines, with particularly strong gains on dynamic event streams and at low or moderate IPC. Specifically, on N-MNIST, it achieves \(84.4\%\) accuracy, about \(85\%\) of the full training set performance, while reducing training time by more than \(50\times\) and storage cost by \(6000\times\), yielding compact surrogates that enable minute-scale SNN training and efficient edge deployment.
>
---
#### [replaced 056] Training and Inference within 1 Second -- Tackle Cross-Sensor Degradation of Real-World Pansharpening with Efficient Residual Feature Tailoring
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.07369v2](https://arxiv.org/pdf/2508.07369v2)**

> **作者:** Tianyu Xin; Jin-Liang Xiao; Zeyu Xia; Shan Yin; Liang-Jian Deng
>
> **摘要:** Deep learning methods for pansharpening have advanced rapidly, yet models pretrained on data from a specific sensor often generalize poorly to data from other sensors. Existing methods to tackle such cross-sensor degradation include retraining model or zero-shot methods, but they are highly time-consuming or even need extra training data. To address these challenges, our method first performs modular decomposition on deep learning-based pansharpening models, revealing a general yet critical interface where high-dimensional fused features begin mapping to the channel space of the final image. % may need revisement A Feature Tailor is then integrated at this interface to address cross-sensor degradation at the feature level, and is trained efficiently with physics-aware unsupervised losses. Moreover, our method operates in a patch-wise manner, training on partial patches and performing parallel inference on all patches to boost efficiency. Our method offers two key advantages: (1) $\textit{Improved Generalization Ability}$: it significantly enhance performance in cross-sensor cases. (2) $\textit{Low Generalization Cost}$: it achieves sub-second training and inference, requiring only partial test inputs and no external data, whereas prior methods often take minutes or even hours. Experiments on the real-world data from multiple datasets demonstrate that our method achieves state-of-the-art quality and efficiency in tackling cross-sensor degradation. For example, training and inference of $512\times512\times8$ image within $\textit{0.2 seconds}$ and $4000\times4000\times8$ image within $\textit{3 seconds}$ at the fastest setting on a commonly used RTX 3090 GPU, which is over 100 times faster than zero-shot methods.
>
---
#### [replaced 057] Human Motion Unlearning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.18674v2](https://arxiv.org/pdf/2503.18674v2)**

> **作者:** Edoardo De Matteis; Matteo Migliarini; Alessio Sampieri; Indro Spinelli; Fabio Galasso
>
> **摘要:** We introduce the task of human motion unlearning to prevent the synthesis of toxic animations while preserving the general text-to-motion generative performance. Unlearning toxic motions is challenging as those can be generated from explicit text prompts and from implicit toxic combinations of safe motions (e.g., "kicking" is "loading and swinging a leg"). We propose the first motion unlearning benchmark by filtering toxic motions from the large and recent text-to-motion datasets of HumanML3D and Motion-X. We propose baselines, by adapting state-of-the-art image unlearning techniques to process spatio-temporal signals. Finally, we propose a novel motion unlearning model based on Latent Code Replacement, which we dub LCR. LCR is training-free and suitable to the discrete latent spaces of state-of-the-art text-to-motion diffusion models. LCR is simple and consistently outperforms baselines qualitatively and quantitatively. Project page: https://www.pinlab.org/hmu.
>
---
#### [replaced 058] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10241v2](https://arxiv.org/pdf/2511.10241v2)**

> **作者:** Jinxuan Li; Yi Zhang; Jian-Fang Hu; Chaolei Tan; Tianming Liang; Beihao Xia
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Spatio-Temporal Video Grounding (STVG) aims to localize a spatio-temporal tube that corresponds to a given language query in an untrimmed video. This is a challenging task since it involves complex vision-language understanding and spatiotemporal reasoning. Recent works have explored weakly-supervised setting in STVG to eliminate reliance on fine-grained annotations like bounding boxes or temporal stamps. However, they typically follow a simple late-fusion manner, which generates tubes independent of the text description, often resulting in failed target identification and inconsistent target tracking. To address this limitation, we propose a Tube-conditioned Reconstruction with Mutual Constraints (\textbf{TubeRMC}) framework that generates text-conditioned candidate tubes with pre-trained visual grounding models and further refine them via tube-conditioned reconstruction with spatio-temporal constraints. Specifically, we design three reconstruction strategies from temporal, spatial, and spatio-temporal perspectives to comprehensively capture rich tube-text correspondences. Each strategy is equipped with a Tube-conditioned Reconstructor, utilizing spatio-temporal tubes as condition to reconstruct the key clues in the query. We further introduce mutual constraints between spatial and temporal proposals to enhance their quality for reconstruction. TubeRMC outperforms existing methods on two public benchmarks VidSTG and HCSTVG. Further visualization shows that TubeRMC effectively mitigates both target identification errors and inconsistent tracking.
>
---
#### [replaced 059] FLUX-Text: A Simple and Advanced Diffusion Transformer Baseline for Scene Text Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.03329v4](https://arxiv.org/pdf/2505.03329v4)**

> **作者:** Rui Lan; Yancheng Bai; Xu Duan; Mingxing Li; Dongyang Jin; Ryan Xu; Dong Nie; Lei Sun; Xiangxiang Chu
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Scene text editing aims to modify or add texts on images while ensuring text fidelity and overall visual quality consistent with the background. Recent methods are primarily built on UNet-based diffusion models, which have improved scene text editing results, but still struggle with complex glyph structures, especially for non-Latin ones (\eg, Chinese, Korean, Japanese). To address these issues, we present \textbf{FLUX-Text}, a simple and advanced multilingual scene text editing DiT method. Specifically, our FLUX-Text enhances glyph understanding and generation through lightweight Visual and Text Embedding Modules, while preserving the original generative capability of FLUX. We further propose a Regional Text Perceptual Loss tailored for text regions, along with a matching two-stage training strategy to better balance text editing and overall image quality. Benefiting from the DiT-based architecture and lightweight feature injection modules, FLUX-Text can be trained with only $0.1$M training examples, a \textbf{97\%} reduction compared to $2.9$M required by popular methods. Extensive experiments on multiple public datasets, including English and Chinese benchmarks, demonstrate that our method surpasses other methods in visual quality and text fidelity. All the code is available at https://github.com/AMAP-ML/FluxText.
>
---
#### [replaced 060] Event Stream Filtering via Probability Flux Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.07503v3](https://arxiv.org/pdf/2504.07503v3)**

> **作者:** Jinze Chen; Wei Zhai; Yang Cao; Bin Li; Zheng-Jun Zha
>
> **摘要:** Event cameras asynchronously capture brightness changes with microsecond latency, offering exceptional temporal precision but suffering from severe noise and signal inconsistencies. Unlike conventional signals, events carry state information through polarities and process information through inter-event time intervals. However, existing event filters often ignore the latter, producing outputs that are sparser than the raw input and limiting the reconstruction of continuous irradiance dynamics. We propose the Event Density Flow Filter (EDFilter), a framework that models event generation as threshold-crossing probability fluxes arising from the stochastic diffusion of irradiance trajectories. EDFilter performs nonparametric, kernel-based estimation of probability flux and reconstructs the continuous event density flow using an O(1) recursive solver, enabling real-time processing. The Rotary Event Dataset (RED), featuring microsecond-resolution ground-truth irradiance flow under controlled illumination is also presented for event quality evaluation. Experiments demonstrate that EDFilter achieves high-fidelity, physically interpretable event denoising and motion reconstruction.
>
---
#### [replaced 061] LightFusion: A Light-weighted, Double Fusion Framework for Unified Multimodal Understanding and Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.22946v4](https://arxiv.org/pdf/2510.22946v4)**

> **作者:** Zeyu Wang; Zilong Chen; Chenhui Gou; Feng Li; Chaorui Deng; Deyao Zhu; Kunchang Li; Weihao Yu; Haoqin Tu; Haoqi Fan; Cihang Xie
>
> **备注:** Preprint. Work in progress
>
> **摘要:** Unified multimodal models have recently shown remarkable gains in both capability and versatility, yet most leading systems are still trained from scratch and require substantial computational resources. In this paper, we show that competitive performance can be obtained far more efficiently by strategically fusing publicly available models specialized for either generation or understanding. Our key design is to retain the original blocks while additionally interleaving multimodal self-attention blocks throughout the networks. This double fusion mechanism (1) effectively enables rich multi-modal fusion while largely preserving the original strengths of the base models, and (2) catalyzes synergistic fusion of high-level semantic representations from the understanding encoder with low-level spatial signals from the generation encoder. By training with only ~ 35B tokens, this approach achieves strong results across multiple benchmarks: 0.91 on GenEval for compositional text-to-image generation, 82.16 on DPG-Bench for complex text-to-image generation, 6.06 on GEditBench, and 3.77 on ImgEdit-Bench for image editing. By fully releasing the entire suite of code, model weights, and datasets, we hope to support future research on unified multimodal modeling.
>
---
#### [replaced 062] CleverDistiller: Simple and Spatially Consistent Cross-modal Distillation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2503.09878v3](https://arxiv.org/pdf/2503.09878v3)**

> **作者:** Hariprasath Govindarajan; Maciej K. Wozniak; Marvin Klingner; Camille Maurice; B Ravi Kiran; Senthil Yogamani
>
> **备注:** Accepted to BMVC 2025
>
> **摘要:** Vision foundation models (VFMs) such as DINO have led to a paradigm shift in 2D camera-based perception towards extracting generalized features to support many downstream tasks. Recent works introduce self-supervised cross-modal knowledge distillation (KD) as a way to transfer these powerful generalization capabilities into 3D LiDAR-based models. However, they either rely on highly complex distillation losses, pseudo-semantic maps, or limit KD to features useful for semantic segmentation only. In this work, we propose CleverDistiller, a self-supervised, cross-modal 2D-to-3D KD framework introducing a set of simple yet effective design choices: Unlike contrastive approaches relying on complex loss design choices, our method employs a direct feature similarity loss in combination with a multi layer perceptron (MLP) projection head to allow the 3D network to learn complex semantic dependencies throughout the projection. Crucially, our approach does not depend on pseudo-semantic maps, allowing for direct knowledge transfer from a VFM without explicit semantic supervision. Additionally, we introduce the auxiliary self-supervised spatial task of occupancy prediction to enhance the semantic knowledge, obtained from a VFM through KD, with 3D spatial reasoning capabilities. Experiments on standard autonomous driving benchmarks for 2D-to-3D KD demonstrate that CleverDistiller achieves state-of-the-art performance in both semantic segmentation and 3D object detection (3DOD) by up to 10% mIoU, especially when fine tuning on really low data amounts, showing the effectiveness of our simple yet powerful KD strategy
>
---
#### [replaced 063] DuetMatch: Harmonizing Semi-Supervised Brain MRI Segmentation via Decoupled Branch Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.16146v2](https://arxiv.org/pdf/2510.16146v2)**

> **作者:** Thanh-Huy Nguyen; Hoang-Thien Nguyen; Vi Vu; Ba-Thinh Lam; Phat Huynh; Tianyang Wang; Xingjian Li; Ulas Bagci; Min Xu
>
> **备注:** Published in Computerized Medical Imaging and Graphics (CMIG)
>
> **摘要:** The limited availability of annotated data in medical imaging makes semi-supervised learning increasingly appealing for its ability to learn from imperfect supervision. Recently, teacher-student frameworks have gained popularity for their training benefits and robust performance. However, jointly optimizing the entire network can hinder convergence and stability, especially in challenging scenarios. To address this for medical image segmentation, we propose DuetMatch, a novel dual-branch semi-supervised framework with asynchronous optimization, where each branch optimizes either the encoder or decoder while keeping the other frozen. To improve consistency under noisy conditions, we introduce Decoupled Dropout Perturbation, enforcing regularization across branches. We also design Pair-wise CutMix Cross-Guidance to enhance model diversity by exchanging pseudo-labels through augmented input pairs. To mitigate confirmation bias from noisy pseudo-labels, we propose Consistency Matching, refining labels using stable predictions from frozen teacher models. Extensive experiments on benchmark brain MRI segmentation datasets, including ISLES2022 and BraTS, show that DuetMatch consistently outperforms state-of-the-art methods, demonstrating its effectiveness and robustness across diverse semi-supervised segmentation scenarios.
>
---
#### [replaced 064] Beyond Patches: Mining Interpretable Part-Prototypes for Explainable AI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.12197v3](https://arxiv.org/pdf/2504.12197v3)**

> **作者:** Mahdi Alehdaghi; Rajarshi Bhattacharya; Pourya Shamsolmoali; Rafael M. O. Cruz; Maguelonne Heritier; Eric Granger
>
> **摘要:** As AI systems grow more capable, it becomes increasingly important that their decisions remain understandable and aligned with human expectations. A key challenge is the limited interpretability of deep models. Post-hoc methods like GradCAM offer heatmaps but provide limited conceptual insight, while prototype-based approaches offer example-based explanations but often rely on rigid region selection and lack semantic consistency. To address these limitations, we propose PCMNet, a part-prototypical concept mining network that learns human-comprehensible prototypes from meaningful image regions without additional supervision. By clustering these prototypes into concept groups and extracting concept activation vectors, PCMNet provides structured, concept-level explanations and enhances robustness to occlusion and challenging conditions, which are both critical for building reliable and aligned AI systems. Experiments across multiple image classification benchmarks show that PCMNet outperforms state-of-the-art methods in interpretability, stability, and robustness. This work contributes to AI alignment by enhancing transparency, controllability, and trustworthiness in AI systems. Our code is available at: https://github.com/alehdaghi/PCMNet.
>
---
#### [replaced 065] Structure-Aware Correspondence Learning for Relative Pose Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.18671v2](https://arxiv.org/pdf/2503.18671v2)**

> **作者:** Yihan Chen; Wenfei Yang; Huan Ren; Shifeng Zhang; Tianzhu Zhang; Feng Wu
>
> **备注:** CVPR2025
>
> **摘要:** Relative pose estimation provides a promising way for achieving object-agnostic pose estimation. Despite the success of existing 3D correspondence-based methods, the reliance on explicit feature matching suffers from small overlaps in visible regions and unreliable feature estimation for invisible regions. Inspired by humans' ability to assemble two object parts that have small or no overlapping regions by considering object structure, we propose a novel Structure-Aware Correspondence Learning method for Relative Pose Estimation, which consists of two key modules. First, a structure-aware keypoint extraction module is designed to locate a set of kepoints that can represent the structure of objects with different shapes and appearance, under the guidance of a keypoint based image reconstruction loss. Second, a structure-aware correspondence estimation module is designed to model the intra-image and inter-image relationships between keypoints to extract structure-aware features for correspondence estimation. By jointly leveraging these two modules, the proposed method can naturally estimate 3D-3D correspondences for unseen objects without explicit feature matching for precise relative pose estimation. Experimental results on the CO3D, Objaverse and LineMOD datasets demonstrate that the proposed method significantly outperforms prior methods, i.e., with 5.7°reduction in mean angular error on the CO3D dataset.
>
---
#### [replaced 066] End-to-End 4D Heart Mesh Recovery Across Full-Stack and Sparse Cardiac MRI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.12090v2](https://arxiv.org/pdf/2509.12090v2)**

> **作者:** Yihong Chen; Jiancheng Yang; Deniz Sayin Mercadier; Hieu Le; Juerg Schwitter; Pascal Fua
>
> **摘要:** Reconstructing cardiac motion from CMR sequences is critical for diagnosis, prognosis, and intervention. Existing methods rely on complete CMR stacks to infer full heart motion, limiting their applicability during intervention when only sparse observations are available. We present TetHeart, the first end-to-end framework for unified 4D heart mesh recovery from both offline full-stack and intra-procedural sparse-slice observations. Our method leverages deformable tetrahedra to capture shape and motion in a coherent space shared across cardiac structures. Before a procedure, it initializes detailed, patient-specific heart meshes from high-quality full stacks, which can then be updated using whatever slices can be obtained in real-time, down to a single one during the procedure. TetHeart incorporates several key innovations: (i) an attentive slice-adaptive 2D-3D feature assembly mechanism that integrates information from arbitrary numbers of slices at any position; (ii) a distillation strategy to ensure accurate reconstruction under extreme sparsity; and (iii) a weakly supervised motion learning scheme requiring annotations only at keyframes, such as the end-diastolic and end-systolic phases. Trained and validated on three large public datasets and evaluated zero-shot on additional private interventional and public datasets without retraining, TetHeart achieves state-of-the-art accuracy and strong generalization in both pre- and intra-procedural settings.
>
---
#### [replaced 067] Linear time small coresets for k-mean clustering of segments with applications
- **分类: cs.LG; cs.CG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12564v2](https://arxiv.org/pdf/2511.12564v2)**

> **作者:** David Denisov; Shlomi Dolev; Dan Felmdan; Michael Segal
>
> **备注:** First published in WALCOM 2026 by Springer Nature
>
> **摘要:** We study the $k$-means problem for a set $\mathcal{S} \subseteq \mathbb{R}^d$ of $n$ segments, aiming to find $k$ centers $X \subseteq \mathbb{R}^d$ that minimize $D(\mathcal{S},X) := \sum_{S \in \mathcal{S}} \min_{x \in X} D(S,x)$, where $D(S,x) := \int_{p \in S} |p - x| dp$ measures the total distance from each point along a segment to a center. Variants of this problem include handling outliers, employing alternative distance functions such as M-estimators, weighting distances to achieve balanced clustering, or enforcing unique cluster assignments. For any $\varepsilon > 0$, an $\varepsilon$-coreset is a weighted subset $C \subseteq \mathbb{R}^d$ that approximates $D(\mathcal{S},X)$ within a factor of $1 \pm \varepsilon$ for any set of $k$ centers, enabling efficient streaming, distributed, or parallel computation. We propose the first coreset construction that provably handles arbitrary input segments. For constant $k$ and $\varepsilon$, it produces a coreset of size $O(\log^2 n)$ computable in $O(nd)$ time. Experiments, including a real-time video tracking application, demonstrate substantial speedups with minimal loss in clustering accuracy, confirming both the practical efficiency and theoretical guarantees of our method.
>
---
#### [replaced 068] Unsupervised learning of spatially varying regularization for diffeomorphic image registration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.17982v2](https://arxiv.org/pdf/2412.17982v2)**

> **作者:** Junyu Chen; Shuwen Wei; Yihao Liu; Zhangxing Bian; Yufan He; Aaron Carass; Harrison Bai; Yong Du
>
> **备注:** Accepted to Medical Image Analysis ((c) MedIA). Code available at http://bit.ly/3BrXGxz
>
> **摘要:** Spatially varying regularization accommodates the deformation variations that may be necessary for different anatomical regions during deformable image registration. Historically, optimization-based registration models have harnessed spatially varying regularization to address anatomical subtleties. However, most modern deep learning-based models tend to gravitate towards spatially invariant regularization, wherein a homogenous regularization strength is applied across the entire image, potentially disregarding localized variations. In this paper, we propose a hierarchical probabilistic model that integrates a prior distribution on the deformation regularization strength, enabling the end-to-end learning of a spatially varying deformation regularizer directly from the data. The proposed method is straightforward to implement and easily integrates with various registration network architectures. Additionally, automatic tuning of hyperparameters is achieved through Bayesian optimization, allowing efficient identification of optimal hyperparameters for any given registration task. Comprehensive evaluations on publicly available datasets demonstrate that the proposed method significantly improves registration performance and enhances the interpretability of deep learning-based registration, all while maintaining smooth deformations.
>
---
#### [replaced 069] Context-Aware Multimodal Representation Learning for Spatio-Temporally Explicit Environmental Modelling
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11706v3](https://arxiv.org/pdf/2511.11706v3)**

> **作者:** Julia Peters; Karin Mora; Miguel D. Mahecha; Chaonan Ji; David Montero; Clemens Mosig; Guido Kraemer
>
> **备注:** 10 pages (incliding 2 pages of references), 7 figures
>
> **摘要:** Earth observation (EO) foundation models have emerged as an effective approach to derive latent representations of the Earth system from various remote sensing sensors. These models produce embeddings that can be used as analysis-ready datasets, enabling the modelling of ecosystem dynamics without extensive sensor-specific preprocessing. However, existing models typically operate at fixed spatial or temporal scales, limiting their use for ecological analyses that require both fine spatial detail and high temporal fidelity. To overcome these limitations, we propose a representation learning framework that integrates different EO modalities into a unified feature space at high spatio-temporal resolution. We introduce the framework using Sentinel-1 and Sentinel-2 data as representative modalities. Our approach produces a latent space at native 10 m resolution and the temporal frequency of cloud-free Sentinel-2 acquisitions. Each sensor is first modeled independently to capture its sensor-specific characteristics. Their representations are then combined into a shared model. This two-stage design enables modality-specific optimisation and easy extension to new sensors, retaining pretrained encoders while retraining only fusion layers. This enables the model to capture complementary remote sensing data and to preserve coherence across space and time. Qualitative analyses reveal that the learned embeddings exhibit high spatial and semantic consistency across heterogeneous landscapes. Quantitative evaluation in modelling Gross Primary Production reveals that they encode ecologically meaningful patterns and retain sufficient temporal fidelity to support fine-scale analyses. Overall, the proposed framework provides a flexible, analysis-ready representation learning approach for environmental applications requiring diverse spatial and temporal resolutions.
>
---
#### [replaced 070] Adaptive Query Prompting for Multi-Domain Landmark Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2404.01194v2](https://arxiv.org/pdf/2404.01194v2)**

> **作者:** Yuhui Li; Qiusen Wei; Guoheng Huang; Xiaochen Yuan; Xuhang Chen; Guo Zhong; Jianwen Huang; Jiajie Huang
>
> **摘要:** Medical landmark detection is crucial in various medical imaging modalities and procedures. Although deep learning-based methods have achieve promising performance, they are mostly designed for specific anatomical regions or tasks. In this work, we propose a universal model for multi-domain landmark detection by leveraging transformer architecture and developing a prompting component, named as Adaptive Query Prompting (AQP). Instead of embedding additional modules in the backbone network, we design a separate module to generate prompts that can be effectively extended to any other transformer network. In our proposed AQP, prompts are learnable parameters maintained in a memory space called prompt pool. The central idea is to keep the backbone frozen and then optimize prompts to instruct the model inference process. Furthermore, we employ a lightweight decoder to decode landmarks from the extracted features, namely Light-MLD. Thanks to the lightweight nature of the decoder and AQP, we can handle multiple datasets by sharing the backbone encoder and then only perform partial parameter tuning without incurring much additional cost. It has the potential to be extended to more landmark detection tasks. We conduct experiments on three widely used X-ray datasets for different medical landmark detection tasks. Our proposed Light-MLD coupled with AQP achieves SOTA performance on many metrics even without the use of elaborate structural designs or complex frameworks.
>
---
#### [replaced 071] LEARNER: Contrastive Pretraining for Learning Fine-Grained Patient Progression from Coarse Inter-Patient Labels
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.01144v2](https://arxiv.org/pdf/2411.01144v2)**

> **作者:** Jana Armouti; Nikhil Madaan; Rohan Panda; Tom Fox; Laura Hutchins; Amita Krishnan; Ricardo Rodriguez; Bennett DeBoisblanc; Deva Ramanan; John Galeotti; Gautam Gare
>
> **备注:** Under review at ISBI 2026 conference
>
> **摘要:** Predicting whether a treatment leads to meaningful improvement is a central challenge in personalized medicine, particularly when disease progression manifests as subtle visual changes over time. While data-driven deep learning (DL) offers a promising route to automate such predictions, acquiring large-scale longitudinal data for each individual patient remains impractical. To address this limitation, we explore whether inter-patient variability can serve as a proxy for learning intra-patient progression. We propose LEARNER, a contrastive pretraining framework that leverages coarsely labeled inter-patient data to learn fine-grained, patient-specific representations. Using lung ultrasound (LUS) and brain MRI datasets, we demonstrate that contrastive objectives trained on coarse inter-patient differences enable models to capture subtle intra-patient changes associated with treatment response. Across both modalities, our approach improves downstream classification accuracy and F1-score compared to standard MSE pretraining, highlighting the potential of inter-patient contrastive learning for individualized outcome prediction.
>
---
#### [replaced 072] Medverse: A Universal Model for Full-Resolution 3D Medical Image Segmentation, Transformation and Enhancement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.09232v3](https://arxiv.org/pdf/2509.09232v3)**

> **作者:** Jiesi Hu; Jianfeng Cao; Yanwu Yang; Chenfei Ye; Yixuan Zhang; Hanyang Peng; Ting Ma
>
> **摘要:** In-context learning (ICL) offers a promising paradigm for universal medical image analysis, enabling models to perform diverse image processing tasks without retraining. However, current ICL models for medical imaging remain limited in two critical aspects: they cannot simultaneously achieve high-fidelity predictions and global anatomical understanding, and there is no unified model trained across diverse medical imaging tasks (e.g., segmentation and enhancement) and anatomical regions. As a result, the full potential of ICL in medical imaging remains underexplored. Thus, we present \textbf{Medverse}, a universal ICL model for 3D medical imaging, trained on 22 datasets covering diverse tasks in universal image segmentation, transformation, and enhancement across multiple organs, imaging modalities, and clinical centers. Medverse employs a next-scale autoregressive in-context learning framework that progressively refines predictions from coarse to fine, generating consistent, full-resolution volumetric outputs and enabling multi-scale anatomical awareness. We further propose a blockwise cross-attention module that facilitates long-range interactions between context and target inputs while preserving computational efficiency through spatial sparsity. Medverse is extensively evaluated on a broad collection of held-out datasets covering previously unseen clinical centers, organs, species, and imaging modalities. Results demonstrate that Medverse substantially outperforms existing ICL baselines and establishes a novel paradigm for in-context learning. Code and model weights will be made publicly available. Our model are publicly available at https://github.com/jiesihu/Medverse.
>
---
#### [replaced 073] vMFCoOp: Towards Equilibrium on a Unified Hyperspherical Manifold for Prompting Biomedical VLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09540v3](https://arxiv.org/pdf/2511.09540v3)**

> **作者:** Minye Shao; Sihan Guo; Xinrun Li; Xingyu Miao; Haoran Duan; Yang Long
>
> **备注:** Accepted as an Oral Presentation at AAAI 2026 Main Technical Track (this version is not peer-reviewed; it is the extended version)
>
> **摘要:** Recent advances in context optimization (CoOp) guided by large language model (LLM)-distilled medical semantic priors offer a scalable alternative to manual prompt engineering and full fine-tuning for adapting biomedical CLIP-based vision-language models (VLMs). However, prompt learning in this context is challenged by semantic misalignment between LLMs and CLIP variants due to divergent training corpora and model architectures; it further lacks scalability across continuously evolving families of foundation models. More critically, pairwise multimodal alignment via conventional Euclidean-space optimization lacks the capacity to model unified representations or apply localized geometric constraints, which tends to amplify modality gaps in complex biomedical imaging and destabilize few-shot adaptation. In this work, we propose vMFCoOp, a framework that inversely estimates von Mises-Fisher (vMF) distributions on a shared Hyperspherical Manifold, aligning semantic biases between arbitrary LLMs and CLIP backbones via Unified Semantic Anchors to achieve robust biomedical prompting and superior few-shot classification. Grounded in three complementary constraints, vMFCoOp demonstrates consistent improvements across 14 medical datasets, 12 medical imaging modalities, and 13 anatomical regions, outperforming state-of-the-art methods in accuracy, generalization, and clinical applicability. This work aims to continuously expand to encompass more downstream applications, and the corresponding resources are intended to be shared through https://github.com/VinyehShaw/UniEqui.
>
---
#### [replaced 074] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [https://arxiv.org/pdf/2505.17534v3](https://arxiv.org/pdf/2505.17534v3)**

> **作者:** Jingjing Jiang; Chongjie Si; Jun Luo; Hanwang Zhang; Chao Ma
>
> **备注:** NeurIPS 2025
>
> **摘要:** This paper presents a pioneering exploration of reinforcement learning (RL) via group relative policy optimization for unified multimodal large language models (ULMs), aimed at simultaneously reinforcing generation and understanding capabilities. Through systematic pilot studies, we uncover the significant potential of ULMs to enable the synergistic co-evolution of dual capabilities within a shared policy optimization framework. Building on this insight, we introduce CoRL, a co-reinforcement learning framework comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement. With the proposed CoRL, our resulting model, ULM-R1, achieves average improvements of 7% on three text-to-image generation datasets and 23% on nine multimodal understanding benchmarks. These results demonstrate the effectiveness of CoRL and highlight the substantial benefit of reinforcement learning in facilitating cross-task synergy and optimization for ULMs. Code is available at https://github.com/mm-vl/ULM-R1.
>
---
#### [replaced 075] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.15580v2](https://arxiv.org/pdf/2511.15580v2)**

> **作者:** Sifan Zhou; Yichao Cao; Jiahao Nie; Yuqian Fu; Ziyu Zhao; Xiaobo Lu; Shuo Wang
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** 3D single object tracking (SOT) in LiDAR point clouds is a critical task in computer vision and autonomous driving. Despite great success having been achieved, the inherent sparsity of point clouds introduces a dual-redundancy challenge that limits existing trackers: (1) vast spatial redundancy from background noise impairs accuracy, and (2) informational redundancy within the foreground hinders efficiency. To tackle these issues, we propose CompTrack, a novel end-to-end framework that systematically eliminates both forms of redundancy in point clouds. First, CompTrack incorporates a Spatial Foreground Predictor (SFP) module to filter out irrelevant background noise based on information entropy, addressing spatial redundancy. Subsequently, its core is an Information Bottleneck-guided Dynamic Token Compression (IB-DTC) module that eliminates the informational redundancy within the foreground. Theoretically grounded in low-rank approximation, this module leverages an online SVD analysis to adaptively compress the redundant foreground into a compact and highly informative set of proxy tokens. Extensive experiments on KITTI, nuScenes and Waymo datasets demonstrate that CompTrack achieves top-performing tracking performance with superior efficiency, running at a real-time 90 FPS on a single RTX 3090 GPU.
>
---
#### [replaced 076] RoMa v2: Harder Better Faster Denser Feature Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15706v2](https://arxiv.org/pdf/2511.15706v2)**

> **作者:** Johan Edstedt; David Nordström; Yushan Zhang; Georg Bökman; Jonathan Astermark; Viktor Larsson; Anders Heyden; Fredrik Kahl; Mårten Wadenbäck; Michael Felsberg
>
> **备注:** Added acknowledgements, and some minor fixes
>
> **摘要:** Dense feature matching aims to estimate all correspondences between two images of a 3D scene and has recently been established as the gold-standard due to its high accuracy and robustness. However, existing dense matchers still fail or perform poorly for many hard real-world scenarios, and high-precision models are often slow, limiting their applicability. In this paper, we attack these weaknesses on a wide front through a series of systematic improvements that together yield a significantly better model. In particular, we construct a novel matching architecture and loss, which, combined with a curated diverse training distribution, enables our model to solve many complex matching tasks. We further make training faster through a decoupled two-stage matching-then-refinement pipeline, and at the same time, significantly reduce refinement memory usage through a custom CUDA kernel. Finally, we leverage the recent DINOv3 foundation model along with multiple other insights to make the model more robust and unbiased. In our extensive set of experiments we show that the resulting novel matcher sets a new state-of-the-art, being significantly more accurate than its predecessors. Code is available at https://github.com/Parskatt/romav2
>
---
#### [replaced 077] DINO in the Room: Leveraging 2D Foundation Models for 3D Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.18944v2](https://arxiv.org/pdf/2503.18944v2)**

> **作者:** Karim Abou Zeid; Kadir Yilmaz; Daan de Geus; Alexander Hermans; David Adrian; Timm Linder; Bastian Leibe
>
> **备注:** Accepted to 3DV 2026. Project page at https://vision.rwth-aachen.de/ditr
>
> **摘要:** Vision foundation models (VFMs) trained on large-scale image datasets provide high-quality features that have significantly advanced 2D visual recognition. However, their potential in 3D scene segmentation remains largely untapped, despite the common availability of 2D images alongside 3D point cloud datasets. While significant research has been dedicated to 2D-3D fusion, recent state-of-the-art 3D methods predominantly focus on 3D data, leaving the integration of VFMs into 3D models underexplored. In this work, we challenge this trend by introducing DITR, a generally applicable approach that extracts 2D foundation model features, projects them to 3D, and finally injects them into a 3D point cloud segmentation model. DITR achieves state-of-the-art results on both indoor and outdoor 3D semantic segmentation benchmarks. To enable the use of VFMs even when images are unavailable during inference, we additionally propose to pretrain 3D models by distilling 2D foundation models. By initializing the 3D backbone with knowledge distilled from 2D VFMs, we create a strong basis for downstream 3D segmentation tasks, ultimately boosting performance across various datasets.
>
---
