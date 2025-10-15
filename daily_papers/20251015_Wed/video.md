# 计算机视觉 cs.CV

- **最新发布 102 篇**

- **更新 86 篇**

## 最新发布

#### [new 001] Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究多模态推理任务，旨在解决现有方法依赖人工标注、推理慢的问题。提出在隐空间中融合视觉与文本的交错隐式推理（IVT-LR），通过渐进式训练提升准确性和推理效率。**

- **链接: [http://arxiv.org/pdf/2510.12603v1](http://arxiv.org/pdf/2510.12603v1)**

> **作者:** Chao Chen; Zhixin Ma; Yongqi Li; Yupeng Hu; Yinwei Wei; Wenjie Li; Liqiang Nie
>
> **摘要:** Multimodal reasoning aims to enhance the capabilities of MLLMs by incorporating intermediate reasoning steps before reaching the final answer. It has evolved from text-only reasoning to the integration of visual information, enabling the thought process to be conveyed through both images and text. Despite its effectiveness, current multimodal reasoning methods depend on explicit reasoning steps that require labor-intensive vision-text annotations and inherently introduce significant inference latency. To address these issues, we introduce multimodal latent reasoning with the advantages of multimodal representation, reduced annotation, and inference efficiency. To facilicate it, we propose Interleaved Vision-Text Latent Reasoning (IVT-LR), which injects both visual and textual information in the reasoning process within the latent space. Specifically, IVT-LR represents each reasoning step by combining two implicit parts: latent text (the hidden states from the previous step) and latent vision (a set of selected image embeddings). We further introduce a progressive multi-stage training strategy to enable MLLMs to perform the above multimodal latent reasoning steps. Experiments on M3CoT and ScienceQA demonstrate that our IVT-LR method achieves an average performance increase of 5.45% in accuracy, while simultaneously achieving a speed increase of over 5 times compared to existing approaches. Code available at https://github.com/FYYDCC/IVT-LR.
>
---
#### [new 002] Data or Language Supervision: What Makes CLIP Better than DINO?
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文探究CLIP优于DINO的原因，区分是语言监督还是数据量的影响。通过控制变量训练，发现CLIP更擅高级语义，适合文本任务；DINO侧重低级特征，略胜于视觉任务。**

- **链接: [http://arxiv.org/pdf/2510.11835v1](http://arxiv.org/pdf/2510.11835v1)**

> **作者:** Yiming Liu; Yuhui Zhang; Dhruba Ghosh; Ludwig Schmidt; Serena Yeung-Levy
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** CLIP outperforms self-supervised models like DINO as vision encoders for vision-language models (VLMs), but it remains unclear whether this advantage stems from CLIP's language supervision or its much larger training data. To disentangle these factors, we pre-train CLIP and DINO under controlled settings -- using the same architecture, dataset, and training configuration -- achieving similar ImageNet accuracy. Embedding analysis shows that CLIP captures high-level semantics (e.g., object categories, text), while DINO is more responsive to low-level features like colors and styles. When integrated into VLMs and evaluated on 20 VQA benchmarks, CLIP excels at text-intensive tasks, while DINO slightly outperforms on vision-centric ones. Variants of language supervision (e.g., sigmoid loss, pre-trained language encoders) yield limited gains. Our findings provide scientific insights into vision encoder design and its impact on VLM performance.
>
---
#### [new 003] Deep Attention-guided Adaptive Subsampling
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究3D医学影像与超声视频分类任务，旨在解决因数据冗余导致的模型计算复杂问题。提出一种可集成于任意网络的注意力引导自适应子采样方法，实现输入自适应的动态采样，兼顾性能与效率。**

- **链接: [http://arxiv.org/pdf/2510.12376v1](http://arxiv.org/pdf/2510.12376v1)**

> **作者:** Sharath M Shankaranarayana; Soumava Kumar Roy; Prasad Sudhakar; Chandan Aladahalli
>
> **摘要:** Although deep neural networks have provided impressive gains in performance, these improvements often come at the cost of increased computational complexity and expense. In many cases, such as 3D volume or video classification tasks, not all slices or frames are necessary due to inherent redundancies. To address this issue, we propose a novel learnable subsampling framework that can be integrated into any neural network architecture. Subsampling, being a nondifferentiable operation, poses significant challenges for direct adaptation into deep learning models. While some works, have proposed solutions using the Gumbel-max trick to overcome the problem of non-differentiability, they fall short in a crucial aspect: they are only task-adaptive and not inputadaptive. Once the sampling mechanism is learned, it remains static and does not adjust to different inputs, making it unsuitable for real-world applications. To this end, we propose an attention-guided sampling module that adapts to inputs even during inference. This dynamic adaptation results in performance gains and reduces complexity in deep neural network models. We demonstrate the effectiveness of our method on 3D medical imaging datasets from MedMNIST3D as well as two ultrasound video datasets for classification tasks, one of them being a challenging in-house dataset collected under real-world clinical conditions.
>
---
#### [new 004] UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出UniGS，面向多模态渲染的统一三维重建任务，解决高保真RGB、深度、法线与语义联合重建问题。通过几何感知的高斯点阵化框架，实现解析梯度优化与可微剪枝，提升精度与效率。**

- **链接: [http://arxiv.org/pdf/2510.12174v1](http://arxiv.org/pdf/2510.12174v1)**

> **作者:** Yusen Xie; Zhenmin Huang; Jianhao Jiao; Dimitrios Kanoulas; Jun Ma
>
> **摘要:** In this paper, we propose UniGS, a unified map representation and differentiable framework for high-fidelity multimodal 3D reconstruction based on 3D Gaussian Splatting. Our framework integrates a CUDA-accelerated rasterization pipeline capable of rendering photo-realistic RGB images, geometrically accurate depth maps, consistent surface normals, and semantic logits simultaneously. We redesign the rasterization to render depth via differentiable ray-ellipsoid intersection rather than using Gaussian centers, enabling effective optimization of rotation and scale attribute through analytic depth gradients. Furthermore, we derive the analytic gradient formulation for surface normal rendering, ensuring geometric consistency among reconstructed 3D scenes. To improve computational and storage efficiency, we introduce a learnable attribute that enables differentiable pruning of Gaussians with minimal contribution during training. Quantitative and qualitative experiments demonstrate state-of-the-art reconstruction accuracy across all modalities, validating the efficacy of our geometry-aware paradigm. Source code and multimodal viewer will be available on GitHub.
>
---
#### [new 005] Unlocking Zero-Shot Plant Segmentation with Pl@ntNet Intelligence
- **分类: cs.CV**

- **简介: 该论文研究零样本植物分割任务，旨在解决农业图像中标注数据稀缺的问题。作者利用Plantnet的植物分类能力生成粗略掩码，结合SAM模型精细分割，无需新标注数据，在多种复杂场景下提升了分割性能。**

- **链接: [http://arxiv.org/pdf/2510.12579v1](http://arxiv.org/pdf/2510.12579v1)**

> **作者:** Simon Ravé; Jean-Christophe Lombardo; Pejman Rasti; Alexis Joly; David Rousseau
>
> **摘要:** We present a zero-shot segmentation approach for agricultural imagery that leverages Plantnet, a large-scale plant classification model, in conjunction with its DinoV2 backbone and the Segment Anything Model (SAM). Rather than collecting and annotating new datasets, our method exploits Plantnet's specialized plant representations to identify plant regions and produce coarse segmentation masks. These masks are then refined by SAM to yield detailed segmentations. We evaluate on four publicly available datasets of various complexity in terms of contrast including some where the limited size of the training data and complex field conditions often hinder purely supervised methods. Our results show consistent performance gains when using Plantnet-fine-tuned DinoV2 over the base DinoV2 model, as measured by the Jaccard Index (IoU). These findings highlight the potential of combining foundation models with specialized plant-centric models to alleviate the annotation bottleneck and enable effective segmentation in diverse agricultural scenarios.
>
---
#### [new 006] DIANet: A Phase-Aware Dual-Stream Network for Micro-Expression Recognition via Dynamic Images
- **分类: cs.CV**

- **简介: 该论文针对微表情识别任务，解决传统动态图像方法忽略时相信息的问题。提出DIANet双流网络，分别建模 onset-to-apex 和 apex-to-offset 阶段，并通过交叉注意力融合特征，提升识别性能。**

- **链接: [http://arxiv.org/pdf/2510.12219v1](http://arxiv.org/pdf/2510.12219v1)**

> **作者:** Vu Tram Anh Khuong; Luu Tu Nguyen; Thi Bich Phuong Man; Thanh Ha Le; Thi Duyen Ngo
>
> **摘要:** Micro-expressions are brief, involuntary facial movements that typically last less than half a second and often reveal genuine emotions. Accurately recognizing these subtle expressions is critical for applications in psychology, security, and behavioral analysis. However, micro-expression recognition (MER) remains a challenging task due to the subtle and transient nature of facial cues and the limited availability of annotated data. While dynamic image (DI) representations have been introduced to summarize temporal motion into a single frame, conventional DI-based methods often overlook the distinct characteristics of different temporal phases within a micro-expression. To address this issue, this paper proposes a novel dual-stream framework, DIANet, which leverages phase-aware dynamic images - one encoding the onset-to-apex phase and the other capturing the apex-to-offset phase. Each stream is processed by a dedicated convolutional neural network, and a cross-attention fusion module is employed to adaptively integrate features from both streams based on their contextual relevance. Extensive experiments conducted on three benchmark MER datasets (CASME-II, SAMM, and MMEW) demonstrate that the proposed method consistently outperforms conventional single-phase DI-based approaches. The results highlight the importance of modeling temporal phase information explicitly and suggest a promising direction for advancing MER.
>
---
#### [new 007] SPORTS: Simultaneous Panoptic Odometry, Rendering, Tracking and Segmentation for Urban Scenes Understanding
- **分类: cs.CV**

- **简介: 该论文提出SPORTS框架，面向城市场景的综合理解，解决动态物体干扰、数据稀疏等问题。通过联合优化视频全景分割、视觉里程计与场景渲染，实现同步定位、分割、跟踪与新视角合成，提升感知与重建精度。**

- **链接: [http://arxiv.org/pdf/2510.12749v1](http://arxiv.org/pdf/2510.12749v1)**

> **作者:** Zhiliu Yang; Jinyu Dai; Jianyuan Zhang; Zhu Yang
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** The scene perception, understanding, and simulation are fundamental techniques for embodied-AI agents, while existing solutions are still prone to segmentation deficiency, dynamic objects' interference, sensor data sparsity, and view-limitation problems. This paper proposes a novel framework, named SPORTS, for holistic scene understanding via tightly integrating Video Panoptic Segmentation (VPS), Visual Odometry (VO), and Scene Rendering (SR) tasks into an iterative and unified perspective. Firstly, VPS designs an adaptive attention-based geometric fusion mechanism to align cross-frame features via enrolling the pose, depth, and optical flow modality, which automatically adjust feature maps for different decoding stages. And a post-matching strategy is integrated to improve identities tracking. In VO, panoptic segmentation results from VPS are combined with the optical flow map to improve the confidence estimation of dynamic objects, which enhances the accuracy of the camera pose estimation and completeness of the depth map generation via the learning-based paradigm. Furthermore, the point-based rendering of SR is beneficial from VO, transforming sparse point clouds into neural fields to synthesize high-fidelity RGB views and twin panoptic views. Extensive experiments on three public datasets demonstrate that our attention-based feature fusion outperforms most existing state-of-the-art methods on the odometry, tracking, segmentation, and novel view synthesis tasks.
>
---
#### [new 008] AngularFuse: A Closer Look at Angle-based Perception for Spatial-Sensitive Multi-Modality Image Fusion
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文研究可见光-红外图像融合，旨在提升融合图像质量。针对现有无监督方法参考图像细节不足、梯度损失忽略方向的问题，提出AngularFuse框架，引入互补掩码模块、改进参考图像生成及角度感知损失，兼顾梯度幅度与方向，显著提升融合效果。**

- **链接: [http://arxiv.org/pdf/2510.12260v1](http://arxiv.org/pdf/2510.12260v1)**

> **作者:** Xiaopeng Liu; Yupei Lin; Sen Zhang; Xiao Wang; Yukai Shi; Liang Lin
>
> **备注:** For the first time, angle-based perception was introduced into the multi-modality image fusion task
>
> **摘要:** Visible-infrared image fusion is crucial in key applications such as autonomous driving and nighttime surveillance. Its main goal is to integrate multimodal information to produce enhanced images that are better suited for downstream tasks. Although deep learning based fusion methods have made significant progress, mainstream unsupervised approaches still face serious challenges in practical applications. Existing methods mostly rely on manually designed loss functions to guide the fusion process. However, these loss functions have obvious limitations. On one hand, the reference images constructed by existing methods often lack details and have uneven brightness. On the other hand, the widely used gradient losses focus only on gradient magnitude. To address these challenges, this paper proposes an angle-based perception framework for spatial-sensitive image fusion (AngularFuse). At first, we design a cross-modal complementary mask module to force the network to learn complementary information between modalities. Then, a fine-grained reference image synthesis strategy is introduced. By combining Laplacian edge enhancement with adaptive histogram equalization, reference images with richer details and more balanced brightness are generated. Last but not least, we introduce an angle-aware loss, which for the first time constrains both gradient magnitude and direction simultaneously in the gradient domain. AngularFuse ensures that the fused images preserve both texture intensity and correct edge orientation. Comprehensive experiments on the MSRS, RoadScene, and M3FD public datasets show that AngularFuse outperforms existing mainstream methods with clear margin. Visual comparisons further confirm that our method produces sharper and more detailed results in challenging scenes, demonstrating superior fusion capability.
>
---
#### [new 009] A Text-Image Fusion Method with Data Augmentation Capabilities for Referring Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究指代表达的医学图像分割，解决传统数据增强破坏图文空间对齐的问题。提出早期融合框架和文本到视觉的轻量生成器，在增强前融合图文特征，保持空间一致性，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2510.12482v1](http://arxiv.org/pdf/2510.12482v1)**

> **作者:** Shurong Chai; Rahul Kumar JAIN; Rui Xu; Shaocong Mo; Ruibo Hou; Shiyu Teng; Jiaqing Liu; Lanfen Lin; Yen-Wei Chen
>
> **摘要:** Deep learning relies heavily on data augmentation to mitigate limited data, especially in medical imaging. Recent multimodal learning integrates text and images for segmentation, known as referring or text-guided image segmentation. However, common augmentations like rotation and flipping disrupt spatial alignment between image and text, weakening performance. To address this, we propose an early fusion framework that combines text and visual features before augmentation, preserving spatial consistency. We also design a lightweight generator that projects text embeddings into visual space, bridging semantic gaps. Visualization of generated pseudo-images shows accurate region localization. Our method is evaluated on three medical imaging tasks and four segmentation frameworks, achieving state-of-the-art results. Code is publicly available on GitHub: https://github.com/11yxk/MedSeg_EarlyFusion.
>
---
#### [new 010] Class-aware Domain Knowledge Fusion and Fission for Continual Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文研究持续测试时适应（CTTA）任务，旨在解决模型在未知域间切换时的灾难性遗忘与知识干扰问题。提出类感知的知识融合与裂变方法（KFF），通过动态分离和合并领域知识，提升新旧知识兼容性与学习效率。**

- **链接: [http://arxiv.org/pdf/2510.12150v1](http://arxiv.org/pdf/2510.12150v1)**

> **作者:** Jiahuan Zhou; Chao Zhu; Zhenyu Cui; Zichen Liu; Xu Zou; Gang Hua
>
> **摘要:** Continual Test-Time Adaptation (CTTA) aims to quickly fine-tune the model during the test phase so that it can adapt to multiple unknown downstream domain distributions without pre-acquiring downstream domain data. To this end, existing advanced CTTA methods mainly reduce the catastrophic forgetting of historical knowledge caused by irregular switching of downstream domain data by restoring the initial model or reusing historical models. However, these methods are usually accompanied by serious insufficient learning of new knowledge and interference from potentially harmful historical knowledge, resulting in severe performance degradation. To this end, we propose a class-aware domain Knowledge Fusion and Fission method for continual test-time adaptation, called KFF, which adaptively expands and merges class-aware domain knowledge in old and new domains according to the test-time data from different domains, where discriminative historical knowledge can be dynamically accumulated. Specifically, considering the huge domain gap within streaming data, a domain Knowledge FIssion (KFI) module is designed to adaptively separate new domain knowledge from a paired class-aware domain prompt pool, alleviating the impact of negative knowledge brought by old domains that are distinct from the current domain. Besides, to avoid the cumulative computation and storage overheads from continuously fissioning new knowledge, a domain Knowledge FUsion (KFU) module is further designed to merge the fissioned new knowledge into the existing knowledge pool with minimal cost, where a greedy knowledge dynamic merging strategy is designed to improve the compatibility of new and old knowledge while keeping the computational efficiency. Extensive experiments on the ImageNet-C dataset verify the effectiveness of our proposed method against other methods.
>
---
#### [new 011] DRL: Discriminative Representation Learning with Parallel Adapters for Class Incremental Learning
- **分类: cs.CV; 68T05, 68T07; I.2.6; I.5.4**

- **简介: 该论文研究无回放的类增量学习，旨在解决模型复杂度高、表示迁移不平滑及阶段优化与全局推理不一致问题。作者提出DRL框架，采用并行适配器和解耦锚点监督，实现高效、判别性强且跨阶段一致的表示学习。**

- **链接: [http://arxiv.org/pdf/2510.12107v1](http://arxiv.org/pdf/2510.12107v1)**

> **作者:** Jiawei Zhan; Jun Liu; Jinlong Peng; Xiaochen Chen; Bin-Bin Gao; Yong Liu; Chengjie Wang
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** With the excellent representation capabilities of Pre-Trained Models (PTMs), remarkable progress has been made in non-rehearsal Class-Incremental Learning (CIL) research. However, it remains an extremely challenging task due to three conundrums: increasingly large model complexity, non-smooth representation shift during incremental learning and inconsistency between stage-wise sub-problem optimization and global inference. In this work, we propose the Discriminative Representation Learning (DRL) framework to specifically address these challenges. To conduct incremental learning effectively and yet efficiently, the DRL's network, called Incremental Parallel Adapter (IPA) network, is built upon a PTM and increasingly augments the model by learning a lightweight adapter with a small amount of parameter learning overhead in each incremental stage. The adapter is responsible for adapting the model to new classes, it can inherit and propagate the representation capability from the current model through parallel connection between them by a transfer gate. As a result, this design guarantees a smooth representation shift between different incremental stages. Furthermore, to alleviate inconsistency and enable comparable feature representations across incremental stages, we design the Decoupled Anchor Supervision (DAS). It decouples constraints of positive and negative samples by respectively comparing them with the virtual anchor. This decoupling promotes discriminative representation learning and aligns the feature spaces learned at different stages, thereby narrowing the gap between stage-wise local optimization over a subset of data and global inference across all classes. Extensive experiments on six benchmarks reveal that our DRL consistently outperforms other state-of-the-art methods throughout the entire CIL period while maintaining high efficiency in both training and inference phases.
>
---
#### [new 012] HoneyBee: Data Recipes for Vision-Language Reasoners
- **分类: cs.CV; cs.LG**

- **简介: 该论文聚焦视觉-语言推理任务，旨在提升模型的推理能力。通过系统研究数据构建策略，提出HoneyBee数据集和测试时扩展方法，显著提升性能并降低计算成本。**

- **链接: [http://arxiv.org/pdf/2510.12225v1](http://arxiv.org/pdf/2510.12225v1)**

> **作者:** Hritik Bansal; Devandra Singh Sachan; Kai-Wei Chang; Aditya Grover; Gargi Ghosh; Wen-tau Yih; Ramakanth Pasunuru
>
> **备注:** 32 pages
>
> **摘要:** Recent advances in vision-language models (VLMs) have made them highly effective at reasoning tasks. However, the principles underlying the construction of performant VL reasoning training datasets remain poorly understood. In this work, we introduce several data curation approaches and study their impacts on VL reasoning capabilities by carefully controlling training and evaluation setups. We analyze the effects of context (image and question pair) sources, implement targeted data interventions, and explore scaling up images, questions, and chain-of-thought (CoT) solutions. Our findings reveal that (a) context source strategies significantly affect VLM performance, (b) interventions such as auxiliary signals from image captions and the inclusion of text-only reasoning yield substantial gains, and (c) scaling all data dimensions (e.g., unique questions per image and unique CoTs per image-question pair) consistently improves reasoning capability. Motivated by these insights, we introduce HoneyBee, a large-scale, high-quality CoT reasoning dataset with 2.5M examples consisting 350K image-question pairs. VLMs trained with HoneyBee outperform state-of-the-art models across model sizes. For instance, a HoneyBee-trained VLM with 3B parameters outperforms the SOTA model and the base model by 7.8% and 24.8%, respectively, on MathVerse. Furthermore, we propose a test-time scaling strategy that reduces decoding cost by 73% without sacrificing accuracy. Overall, this work presents improved strategies for VL reasoning dataset curation research.
>
---
#### [new 013] Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos through Spatio-Temporal Modeling
- **分类: cs.CV**

- **简介: 该论文研究第一人称装配视频中的步骤识别任务，旨在解决因物体遮挡导致的识别延迟问题。作者提出STORM-PSR双流框架，结合空间与时空特征，提升遮挡下的步骤完成判断准确性，显著降低预测延迟。**

- **链接: [http://arxiv.org/pdf/2510.12385v1](http://arxiv.org/pdf/2510.12385v1)**

> **作者:** Tim J. Schoonbeek; Shao-Hsuan Hung; Dan Lehman; Hans Onvlee; Jacek Kustra; Peter H. N. de With; Fons van der Sommen
>
> **备注:** 26 pages, 7 figures and 5 tables in the main paper and one figure and table in the appendix. To be published in Computer Vision and Image Understanding
>
> **摘要:** Procedure step recognition (PSR) aims to identify all correctly completed steps and their sequential order in videos of procedural tasks. The existing state-of-the-art models rely solely on detecting assembly object states in individual video frames. By neglecting temporal features, model robustness and accuracy are limited, especially when objects are partially occluded. To overcome these limitations, we propose Spatio-Temporal Occlusion-Resilient Modeling for Procedure Step Recognition (STORM-PSR), a dual-stream framework for PSR that leverages both spatial and temporal features. The assembly state detection stream operates effectively with unobstructed views of the object, while the spatio-temporal stream captures both spatial and temporal features to recognize step completions even under partial occlusion. This stream includes a spatial encoder, pre-trained using a novel weakly supervised approach to capture meaningful spatial representations, and a transformer-based temporal encoder that learns how these spatial features relate over time. STORM-PSR is evaluated on the MECCANO and IndustReal datasets, reducing the average delay between actual and predicted assembly step completions by 11.2% and 26.1%, respectively, compared to prior methods. We demonstrate that this reduction in delay is driven by the spatio-temporal stream, which does not rely on unobstructed views of the object to infer completed steps. The code for STORM-PSR, along with the newly annotated MECCANO labels, is made publicly available at https://timschoonbeek.github.io/stormpsr .
>
---
#### [new 014] UniFusion: Vision-Language Model as Unified Encoder in Image Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出UniFusion，旨在解决图像生成中图文分离编码限制跨模态推理的问题。通过冻结大视觉语言模型作为统一编码器，结合层注意力池化与重写注入机制，实现高效图文融合生成与编辑，提升对齐性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.12789v1](http://arxiv.org/pdf/2510.12789v1)**

> **作者:** Kevin Li; Manuel Brack; Sudeep Katakol; Hareesh Ravi; Ajinkya Kale
>
> **备注:** Project page at https://thekevinli.github.io/unifusion/
>
> **摘要:** Although recent advances in visual generation have been remarkable, most existing architectures still depend on distinct encoders for images and text. This separation constrains diffusion models' ability to perform cross-modal reasoning and knowledge transfer. Prior attempts to bridge this gap often use the last layer information from VLM, employ multiple visual encoders, or train large unified models jointly for text and image generation, which demands substantial computational resources and large-scale data, limiting its accessibility.We present UniFusion, a diffusion-based generative model conditioned on a frozen large vision-language model (VLM) that serves as a unified multimodal encoder. At the core of UniFusion is the Layerwise Attention Pooling (LAP) mechanism that extracts both high level semantics and low level details from text and visual tokens of a frozen VLM to condition a diffusion generative model. We demonstrate that LAP outperforms other shallow fusion architectures on text-image alignment for generation and faithful transfer of visual information from VLM to the diffusion model which is key for editing. We propose VLM-Enabled Rewriting Injection with Flexibile Inference (VERIFI), which conditions a diffusion transformer (DiT) only on the text tokens generated by the VLM during in-model prompt rewriting. VERIFI combines the alignment of the conditioning distribution with the VLM's reasoning capabilities for increased capabilities and flexibility at inference. In addition, finetuning on editing task not only improves text-image alignment for generation, indicative of cross-modality knowledge transfer, but also exhibits tremendous generalization capabilities. Our model when trained on single image editing, zero-shot generalizes to multiple image references further motivating the unified encoder design of UniFusion.
>
---
#### [new 015] Detect Anything via Next Point Prediction
- **分类: cs.CV**

- **简介: 该论文聚焦开放世界物体检测任务，旨在解决MLLM在检测中召回率低、预测重复等问题。作者提出Rex-Omni，通过量化坐标表示、构建高质量数据引擎和两阶段训练，实现零样本下媲美YOLO等模型的性能，并支持多种视觉语言任务。**

- **链接: [http://arxiv.org/pdf/2510.12798v1](http://arxiv.org/pdf/2510.12798v1)**

> **作者:** Qing Jiang; Junan Huo; Xingyu Chen; Yuda Xiong; Zhaoyang Zeng; Yihao Chen; Tianhe Ren; Junzhi Yu; Lei Zhang
>
> **备注:** homepage: https://rex-omni.github.io/
>
> **摘要:** Object detection has long been dominated by traditional coordinate regression-based models, such as YOLO, DETR, and Grounding DINO. Although recent efforts have attempted to leverage MLLMs to tackle this task, they face challenges like low recall rate, duplicate predictions, coordinate misalignment, etc. In this work, we bridge this gap and propose Rex-Omni, a 3B-scale MLLM that achieves state-of-the-art object perception performance. On benchmarks like COCO and LVIS, Rex-Omni attains performance comparable to or exceeding regression-based models (e.g., DINO, Grounding DINO) in a zero-shot setting. This is enabled by three key designs: 1) Task Formulation: we use special tokens to represent quantized coordinates from 0 to 999, reducing the model's learning difficulty and improving token efficiency for coordinate prediction; 2) Data Engines: we construct multiple data engines to generate high-quality grounding, referring, and pointing data, providing semantically rich supervision for training; \3) Training Pipelines: we employ a two-stage training process, combining supervised fine-tuning on 22 million data with GRPO-based reinforcement post-training. This RL post-training leverages geometry-aware rewards to effectively bridge the discrete-to-continuous coordinate prediction gap, improve box accuracy, and mitigate undesirable behaviors like duplicate predictions that stem from the teacher-guided nature of the initial SFT stage. Beyond conventional detection, Rex-Omni's inherent language understanding enables versatile capabilities such as object referring, pointing, visual prompting, GUI grounding, spatial referring, OCR and key-pointing, all systematically evaluated on dedicated benchmarks. We believe that Rex-Omni paves the way for more versatile and language-aware visual perception systems.
>
---
#### [new 016] DPL: Spatial-Conditioned Diffusion Prototype Enhancement for One-Shot Medical Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割中单样本学习的原型表征问题，提出扩散原型学习（DPL）框架。通过扩散模型生成多样化且语义一致的原型变体，结合空间感知条件机制与保守融合策略，提升模型泛化能力，在腹部MRI和CT数据上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2510.12159v1](http://arxiv.org/pdf/2510.12159v1)**

> **作者:** Ziyuan Gao; Philippe Morel
>
> **备注:** Accepted at IVCNZ 2025. To be published in IEEE proceedings
>
> **摘要:** One-shot medical image segmentation faces fundamental challenges in prototype representation due to limited annotated data and significant anatomical variability across patients. Traditional prototype-based methods rely on deterministic averaging of support features, creating brittle representations that fail to capture intra-class diversity essential for robust generalization. This work introduces Diffusion Prototype Learning (DPL), a novel framework that reformulates prototype construction through diffusion-based feature space exploration. DPL models one-shot prototypes as learnable probability distributions, enabling controlled generation of diverse yet semantically coherent prototype variants from minimal labeled data. The framework operates through three core innovations: (1) a diffusion-based prototype enhancement module that transforms single support prototypes into diverse variant sets via forward-reverse diffusion processes, (2) a spatial-aware conditioning mechanism that leverages geometric properties derived from prototype feature statistics, and (3) a conservative fusion strategy that preserves prototype fidelity while maximizing representational diversity. DPL ensures training-inference consistency by using the same diffusion enhancement and fusion pipeline in both phases. This process generates enhanced prototypes that serve as the final representations for similarity calculations, while the diffusion process itself acts as a regularizer. Extensive experiments on abdominal MRI and CT datasets demonstrate significant improvements respectively, establishing new state-of-the-art performance in one-shot medical image segmentation.
>
---
#### [new 017] Unconditional Human Motion and Shape Generation via Balanced Score-Based Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究无条件人体运动与形状生成任务，旨在解决现有方法依赖过参数化特征和辅助损失的问题。作者提出基于分数的扩散模型，通过特征空间归一化和解析推导的权重分配，在不使用复杂设计的情况下实现高质量生成，并同步输出运动与形状。**

- **链接: [http://arxiv.org/pdf/2510.12537v1](http://arxiv.org/pdf/2510.12537v1)**

> **作者:** David Björkstrand; Tiesheng Wang; Lars Bretzner; Josephine Sullivan
>
> **摘要:** Recent work has explored a range of model families for human motion generation, including Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and diffusion-based models. Despite their differences, many methods rely on over-parameterized input features and auxiliary losses to improve empirical results. These strategies should not be strictly necessary for diffusion models to match the human motion distribution. We show that on par with state-of-the-art results in unconditional human motion generation are achievable with a score-based diffusion model using only careful feature-space normalization and analytically derived weightings for the standard L2 score-matching loss, while generating both motion and shape directly, thereby avoiding slow post hoc shape recovery from joints. We build the method step by step, with a clear theoretical motivation for each component, and provide targeted ablations demonstrating the effectiveness of each proposed addition in isolation.
>
---
#### [new 018] TerraCodec: Compressing Earth Observations
- **分类: cs.CV**

- **简介: 该论文聚焦地球观测图像压缩任务，解决现有方法在多光谱时序数据上压缩效率低、忽略时间冗余的问题。作者提出TerraCodec，包括面向多光谱图像的高效变体和利用时间依赖的Temporal Transformer模型，并引入Latent Repacking实现灵活码率压缩，在压缩性能与云修复任务上均取得更好效果。**

- **链接: [http://arxiv.org/pdf/2510.12670v1](http://arxiv.org/pdf/2510.12670v1)**

> **作者:** Julen Costa-Watanabe; Isabelle Wittmann; Benedikt Blumenstiel; Konrad Schindler
>
> **摘要:** Earth observation (EO) satellites produce massive streams of multispectral image time series, posing pressing challenges for storage and transmission. Yet, learned EO compression remains fragmented, lacking publicly available pretrained models and misaligned with advances in compression for natural imagery. Image codecs overlook temporal redundancy, while video codecs rely on motion priors that fail to capture the radiometric evolution of largely static scenes. We introduce TerraCodec (TEC), a family of learned codecs tailored to EO. TEC includes efficient image-based variants adapted to multispectral inputs, as well as a Temporal Transformer model (TEC-TT) that leverages dependencies across time. To overcome the fixed-rate setting of today's neural codecs, we present Latent Repacking, a novel method for training flexible-rate transformer models that operate on varying rate-distortion settings. Trained on Sentinel-2 data, TerraCodec outperforms classical codecs, achieving 3-10x stronger compression at equivalent image quality. Beyond compression, TEC-TT enables zero-shot cloud inpainting, surpassing state-of-the-art methods on the AllClear benchmark. Our results establish bespoke, learned compression algorithms as a promising direction for Earth observation. Code and model weights will be released under a permissive license.
>
---
#### [new 019] BIGFix: Bidirectional Image Generation with Token Fixing
- **分类: cs.CV**

- **简介: 该论文研究图像与视频生成任务，旨在解决并行多标记预测导致的结构不一致问题。作者提出BIGFix方法，通过注入随机标记训练模型，实现生成过程中的标记自修正，在保持高效推理的同时显著提升生成质量。**

- **链接: [http://arxiv.org/pdf/2510.12231v1](http://arxiv.org/pdf/2510.12231v1)**

> **作者:** Victor Besnier; David Hurych; Andrei Bursuc; Eduardo Valle
>
> **摘要:** Recent advances in image and video generation have raised significant interest from both academia and industry. A key challenge in this field is improving inference efficiency, as model size and the number of inference steps directly impact the commercial viability of generative models while also posing fundamental scientific challenges. A promising direction involves combining auto-regressive sequential token modeling with multi-token prediction per step, reducing inference time by up to an order of magnitude. However, predicting multiple tokens in parallel can introduce structural inconsistencies due to token incompatibilities, as capturing complex joint dependencies during training remains challenging. Traditionally, once tokens are sampled, there is no mechanism to backtrack and refine erroneous predictions. We propose a method for self-correcting image generation by iteratively refining sampled tokens. We achieve this with a novel training scheme that injects random tokens in the context, improving robustness and enabling token fixing during sampling. Our method preserves the efficiency benefits of parallel token prediction while significantly enhancing generation quality. We evaluate our approach on image generation using the ImageNet-256 and CIFAR-10 datasets, as well as on video generation with UCF-101 and NuScenes, demonstrating substantial improvements across both modalities.
>
---
#### [new 020] BEEP3D: Box-Supervised End-to-End Pseudo-Mask Generation for 3D Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文研究3D实例分割，旨在降低对密集点标注的依赖。提出BEEP3D方法，利用框级标注，通过学生-教师框架端到端生成伪掩码，引入查询优化与一致性损失，提升分割精度与训练效率。**

- **链接: [http://arxiv.org/pdf/2510.12182v1](http://arxiv.org/pdf/2510.12182v1)**

> **作者:** Youngju Yoo; Seho Kim; Changick Kim
>
> **摘要:** 3D instance segmentation is crucial for understanding complex 3D environments, yet fully supervised methods require dense point-level annotations, resulting in substantial annotation costs and labor overhead. To mitigate this, box-level annotations have been explored as a weaker but more scalable form of supervision. However, box annotations inherently introduce ambiguity in overlapping regions, making accurate point-to-instance assignment challenging. Recent methods address this ambiguity by generating pseudo-masks through training a dedicated pseudo-labeler in an additional training stage. However, such two-stage pipelines often increase overall training time and complexity, hinder end-to-end optimization. To overcome these challenges, we propose BEEP3D-Box-supervised End-to-End Pseudo-mask generation for 3D instance segmentation. BEEP3D adopts a student-teacher framework, where the teacher model serves as a pseudo-labeler and is updated by the student model via an Exponential Moving Average. To better guide the teacher model to generate precise pseudo-masks, we introduce an instance center-based query refinement that enhances position query localization and leverages features near instance centers. Additionally, we design two novel losses-query consistency loss and masked feature consistency loss-to align semantic and geometric signals between predictions and pseudo-masks. Extensive experiments on ScanNetV2 and S3DIS datasets demonstrate that BEEP3D achieves competitive or superior performance compared to state-of-the-art weakly supervised methods while remaining computationally efficient.
>
---
#### [new 021] Beyond Seeing: Evaluating Multimodal LLMs on Tool-Enabled Image Perception, Transformation, and Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出IRIS基准，旨在评估多模态大模型在“与图像共思考”范式下的视觉感知、图像操作与推理能力，解决现有基准仅静态理解图像的问题。构建了1204个开放性任务，揭示当前模型在工具协同与视觉推理上的不足。**

- **链接: [http://arxiv.org/pdf/2510.12712v1](http://arxiv.org/pdf/2510.12712v1)**

> **作者:** Xingang Guo; Utkarsh Tyagi; Advait Gosai; Paula Vergara; Ernesto Gabriel Hernández Montoya; Chen Bo Calvin Zhang; Bin Hu; Yunzhong He; Bing Liu; Rakshith Sharma Srinivasa
>
> **摘要:** Multimodal Large Language Models (MLLMs) are increasingly applied in real-world scenarios where user-provided images are often imperfect, requiring active image manipulations such as cropping, editing, or enhancement to uncover salient visual cues. Beyond static visual perception, MLLMs must also think with images: dynamically transforming visual content and integrating it with other tools to solve complex tasks. However, this shift from treating vision as passive context to a manipulable cognitive workspace remains underexplored. Most existing benchmarks still follow a think about images paradigm, where images are regarded as static inputs. To address this gap, we introduce IRIS, an Interactive Reasoning with Images and Systems that evaluates MLLMs' ability to perceive, transform, and reason across complex visual-textual tasks under the think with images paradigm. IRIS comprises 1,204 challenging, open-ended vision tasks (603 single-turn, 601 multi-turn) spanning across five diverse domains, each paired with detailed rubrics to enable systematic evaluation. Our evaluation shows that current MLLMs struggle with tasks requiring effective integration of vision and general-purpose tools. Even the strongest model (GPT-5-think) reaches only 18.68% pass rate. We further observe divergent tool-use behaviors, with OpenAI models benefiting from diverse image manipulations while Gemini-2.5-pro shows no improvement. By introducing the first benchmark centered on think with images, IRIS offers critical insights for advancing visual intelligence in MLLMs.
>
---
#### [new 022] Prompt-Guided Spatial Understanding with RGB-D Transformers for Fine-Grained Object Relation Reasoning
- **分类: cs.CV**

- **简介: 该论文针对大规模3D环境中细粒度物体关系推理任务，解决因遮挡和杂乱导致的空间理解难题。提出一种将边界框坐标嵌入提示的RGB-D Transformer框架，通过提示引导增强空间推理，在四项任务上进行微调，提升了模型在真实工业场景中的空间认知能力。**

- **链接: [http://arxiv.org/pdf/2510.11996v1](http://arxiv.org/pdf/2510.11996v1)**

> **作者:** Tanner Muturi; Blessing Agyei Kyem; Joshua Kofi Asamoah; Neema Jakisa Owor; Richard Dyzinela; Andrews Danyo; Yaw Adu-Gyamfi; Armstrong Aboah
>
> **备注:** The paper was accepted at ICCV Conference 2025
>
> **摘要:** Spatial reasoning in large-scale 3D environments such as warehouses remains a significant challenge for vision-language systems due to scene clutter, occlusions, and the need for precise spatial understanding. Existing models often struggle with generalization in such settings, as they rely heavily on local appearance and lack explicit spatial grounding. In this work, we introduce a dedicated spatial reasoning framework for the Physical AI Spatial Intelligence Warehouse dataset introduced in the Track 3 2025 AI City Challenge. Our approach enhances spatial comprehension by embedding mask dimensions in the form of bounding box coordinates directly into the input prompts, enabling the model to reason over object geometry and layout. We fine-tune the framework across four question categories namely: Distance Estimation, Object Counting, Multi-choice Grounding, and Spatial Relation Inference using task-specific supervision. To further improve consistency with the evaluation system, normalized answers are appended to the GPT response within the training set. Our comprehensive pipeline achieves a final score of 73.0606, placing 4th overall on the public leaderboard. These results demonstrate the effectiveness of structured prompt enrichment and targeted optimization in advancing spatial reasoning for real-world industrial environments.
>
---
#### [new 023] MammoDINO: Anatomically Aware Self-Supervision for Mammographic Images
- **分类: cs.CV; cs.AI; 1.2**

- **简介: 该论文提出MammoDINO，一种面向乳腺X光图像的自监督学习框架。针对医学图像数据少、领域偏差大问题，引入乳腺组织感知增强和跨切片对比学习，在140万图像上预训练，提升多种乳腺癌筛查任务性能，支持无标注、多用途CAD工具开发。**

- **链接: [http://arxiv.org/pdf/2510.11883v1](http://arxiv.org/pdf/2510.11883v1)**

> **作者:** Sicheng Zhou; Lei Wu; Cao Xiao; Parminder Bhatia; Taha Kass-Hout
>
> **备注:** 5 pages
>
> **摘要:** Self-supervised learning (SSL) has transformed vision encoder training in general domains but remains underutilized in medical imaging due to limited data and domain specific biases. We present MammoDINO, a novel SSL framework for mammography, pretrained on 1.4 million mammographic images. To capture clinically meaningful features, we introduce a breast tissue aware data augmentation sampler for both image-level and patch-level supervision and a cross-slice contrastive learning objective that leverages 3D digital breast tomosynthesis (DBT) structure into 2D pretraining. MammoDINO achieves state-of-the-art performance on multiple breast cancer screening tasks and generalizes well across five benchmark datasets. It offers a scalable, annotation-free foundation for multipurpose computer-aided diagnosis (CAD) tools for mammogram, helping reduce radiologists' workload and improve diagnostic efficiency in breast cancer screening.
>
---
#### [new 024] LayerSync: Self-aligning Intermediate Layers
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出LayerSync，一种领域无关的扩散模型中间层自对齐方法，旨在提升生成质量和训练效率。通过利用模型自身语义丰富的中间表示作为内在引导，无需外部监督或额外数据，实现即插即用的正则化，显著加速训练并提升多模态生成性能。**

- **链接: [http://arxiv.org/pdf/2510.12581v1](http://arxiv.org/pdf/2510.12581v1)**

> **作者:** Yasaman Haghighi; Bastien van Delft; Mariam Hassan; Alexandre Alahi
>
> **摘要:** We propose LayerSync, a domain-agnostic approach for improving the generation quality and the training efficiency of diffusion models. Prior studies have highlighted the connection between the quality of generation and the representations learned by diffusion models, showing that external guidance on model intermediate representations accelerates training. We reconceptualize this paradigm by regularizing diffusion models with their own intermediate representations. Building on the observation that representation quality varies across diffusion model layers, we show that the most semantically rich representations can act as an intrinsic guidance for weaker ones, reducing the need for external supervision. Our approach, LayerSync, is a self-sufficient, plug-and-play regularizer term with no overhead on diffusion model training and generalizes beyond the visual domain to other modalities. LayerSync requires no pretrained models nor additional data. We extensively evaluate the method on image generation and demonstrate its applicability to other domains such as audio, video, and motion generation. We show that it consistently improves the generation quality and the training efficiency. For example, we speed up the training of flow-based transformer by over 8.75x on ImageNet dataset and improved the generation quality by 23.6%. The code is available at https://github.com/vita-epfl/LayerSync.
>
---
#### [new 025] BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring
- **分类: cs.CV**

- **简介: 该论文针对相机运动导致的图像模糊，提出双阶段3D高斯点阵化（BSGS）框架，用于高质量3D场景重建。通过相机位姿优化与全局刚性变换两阶段设计，结合梯度聚合与时空优化策略，有效缓解模糊引起的失真与噪声问题。**

- **链接: [http://arxiv.org/pdf/2510.12493v1](http://arxiv.org/pdf/2510.12493v1)**

> **作者:** An Zhao; Piaopiao Yu; Zhe Zhu; Mingqiang Wei
>
> **摘要:** 3D Gaussian Splatting has exhibited remarkable capabilities in 3D scene reconstruction.However, reconstructing high-quality 3D scenes from motion-blurred images caused by camera motion poses a significant challenge.The performance of existing 3DGS-based deblurring methods are limited due to their inherent mechanisms, such as extreme dependence on the accuracy of camera poses and inability to effectively control erroneous Gaussian primitives densification caused by motion blur.To solve these problems, we introduce a novel framework, Bi-Stage 3D Gaussian Splatting, to accurately reconstruct 3D scenes from motion-blurred images.BSGS contains two stages. First, Camera Pose Refinement roughly optimizes camera poses to reduce motion-induced distortions. Second, with fixed rough camera poses, Global RigidTransformation further corrects motion-induced blur distortions.To alleviate multi-subframe gradient conflicts, we propose a subframe gradient aggregation strategy to optimize both stages.Furthermore, a space-time bi-stage optimization strategy is introduced to dynamically adjust primitive densification thresholds and prevent premature noisy Gaussian generation in blurred regions. Comprehensive experiments verify the effectiveness of our proposed deblurring method and show its superiority over the state of the arts.
>
---
#### [new 026] Ivan-ISTD: Rethinking Cross-domain Heteroscedastic Noise Perturbations in Infrared Small Target Detection
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究红外小目标检测中的跨域异方差噪声问题，提出Ivan-ISTD框架，通过小波引导的跨域合成和真实噪声不变性学习，提升模型在动态退化场景下的鲁棒性，并构建新基准进行验证。**

- **链接: [http://arxiv.org/pdf/2510.12241v1](http://arxiv.org/pdf/2510.12241v1)**

> **作者:** Yuehui Li; Yahao Lu; Haoyuan Wu; Sen Zhang; Liang Lin; Yukai Shi
>
> **备注:** In infrared small target detection, noise from different sensors can cause significant interference to performance. We propose a new dataset and a wavelet-guided Invariance learning framework(Ivan-ISTD) to emphasize this issue
>
> **摘要:** In the multimedia domain, Infrared Small Target Detection (ISTD) plays a important role in drone-based multi-modality sensing. To address the dual challenges of cross-domain shift and heteroscedastic noise perturbations in ISTD, we propose a doubly wavelet-guided Invariance learning framework(Ivan-ISTD). In the first stage, we generate training samples aligned with the target domain using Wavelet-guided Cross-domain Synthesis. This wavelet-guided alignment machine accurately separates the target background through multi-frequency wavelet filtering. In the second stage, we introduce Real-domain Noise Invariance Learning, which extracts real noise characteristics from the target domain to build a dynamic noise library. The model learns noise invariance through self-supervised loss, thereby overcoming the limitations of distribution bias in traditional artificial noise modeling. Finally, we create the Dynamic-ISTD Benchmark, a cross-domain dynamic degradation dataset that simulates the distribution shifts encountered in real-world applications. Additionally, we validate the versatility of our method using other real-world datasets. Experimental results demonstrate that our approach outperforms existing state-of-the-art methods in terms of many quantitative metrics. In particular, Ivan-ISTD demonstrates excellent robustness in cross-domain scenarios. The code for this work can be found at: https://github.com/nanjin1/Ivan-ISTD.
>
---
#### [new 027] Hierarchical Reasoning with Vision-Language Models for Incident Reports from Dashcam Videos
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶中罕见事故场景的理解难题，提出一种基于视觉-语言模型的分层推理框架，用于从行车记录视频生成可读性强、事实准确的事故报告，在2COOOL挑战赛中取得优异成绩。**

- **链接: [http://arxiv.org/pdf/2510.12190v1](http://arxiv.org/pdf/2510.12190v1)**

> **作者:** Shingo Yokoi; Kento Sasaki; Yu Yamaguchi
>
> **备注:** 2nd Place Winner, ICCV 2025 2COOOL Competition
>
> **摘要:** Recent advances in end-to-end (E2E) autonomous driving have been enabled by training on diverse large-scale driving datasets, yet autonomous driving models still struggle in out-of-distribution (OOD) scenarios. The COOOL benchmark targets this gap by encouraging hazard understanding beyond closed taxonomies, and the 2COOOL challenge extends it to generating human-interpretable incident reports. We present a hierarchical reasoning framework for incident report generation from dashcam videos that integrates frame-level captioning, incident frame detection, and fine-grained reasoning within vision-language models (VLMs). We further improve factual accuracy and readability through model ensembling and a Blind A/B Scoring selection protocol. On the official 2COOOL open leaderboard, our method ranks 2nd among 29 teams and achieves the best CIDEr-D score, producing accurate and coherent incident narratives. These results indicate that hierarchical reasoning with VLMs is a promising direction for accident analysis and for broader understanding of safety-critical traffic events. The implementation and code are available at https://github.com/riron1206/kaggle-2COOOL-2nd-Place-Solution.
>
---
#### [new 028] DeepMMSearch-R1: Empowering Multimodal LLMs in Multimodal Web Search
- **分类: cs.CV; cs.IR**

- **简介: 该论文聚焦多模态网页搜索任务，旨在解决现有方法搜索僵化、查询低效的问题。作者提出DeepMMSearch-R1，首个支持按需多轮图文搜索的多模态大模型，通过两阶段训练提升动态查询与自我修正能力。**

- **链接: [http://arxiv.org/pdf/2510.12801v1](http://arxiv.org/pdf/2510.12801v1)**

> **作者:** Kartik Narayan; Yang Xu; Tian Cao; Kavya Nerella; Vishal M. Patel; Navid Shiee; Peter Grasch; Chao Jia; Yinfei Yang; Zhe Gan
>
> **摘要:** Multimodal Large Language Models (MLLMs) in real-world applications require access to external knowledge sources and must remain responsive to the dynamic and ever-changing real-world information in order to address information-seeking and knowledge-intensive user queries. Existing approaches, such as retrieval augmented generation (RAG) methods, search agents, and search equipped MLLMs, often suffer from rigid pipelines, excessive search calls, and poorly constructed search queries, which result in inefficiencies and suboptimal outcomes. To address these limitations, we present DeepMMSearch-R1, the first multimodal LLM capable of performing on-demand, multi-turn web searches and dynamically crafting queries for both image and text search tools. Specifically, DeepMMSearch-R1 can initiate web searches based on relevant crops of the input image making the image search more effective, and can iteratively adapt text search queries based on retrieved information, thereby enabling self-reflection and self-correction. Our approach relies on a two-stage training pipeline: a cold start supervised finetuning phase followed by an online reinforcement learning optimization. For training, we introduce DeepMMSearchVQA, a novel multimodal VQA dataset created through an automated pipeline intermixed with real-world information from web search tools. This dataset contains diverse, multi-hop queries that integrate textual and visual information, teaching the model when to search, what to search for, which search tool to use and how to reason over the retrieved information. We conduct extensive experiments across a range of knowledge-intensive benchmarks to demonstrate the superiority of our approach. Finally, we analyze the results and provide insights that are valuable for advancing multimodal web-search.
>
---
#### [new 029] Low-Field Magnetic Resonance Image Quality Enhancement using a Conditional Flow Matching Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究低场磁共振图像质量增强，旨在提升低信噪比图像的诊断质量。提出基于条件流匹配（CFM）的框架，通过学习噪声到目标数据的连续流，将低场图像转化为高场-like图像，显著减少参数量并实现先进性能。**

- **链接: [http://arxiv.org/pdf/2510.12408v1](http://arxiv.org/pdf/2510.12408v1)**

> **作者:** Huu Tien Nguyen; Ahmed Karam Eldaly
>
> **摘要:** This paper introduces a novel framework for image quality transfer based on conditional flow matching (CFM). Unlike conventional generative models that rely on iterative sampling or adversarial objectives, CFM learns a continuous flow between a noise distribution and target data distributions through the direct regression of an optimal velocity field. We evaluate this approach in the context of low-field magnetic resonance imaging (LF-MRI), a rapidly emerging modality that offers affordable and portable scanning but suffers from inherently low signal-to-noise ratio and reduced diagnostic quality. Our framework is designed to reconstruct high-field-like MR images from their corresponding low-field inputs, thereby bridging the quality gap without requiring expensive infrastructure. Experiments demonstrate that CFM not only achieves state-of-the-art performance, but also generalizes robustly to both in-distribution and out-of-distribution data. Importantly, it does so while utilizing significantly fewer parameters than competing deep learning methods. These results underline the potential of CFM as a powerful and scalable tool for MRI reconstruction, particularly in resource-limited clinical environments.
>
---
#### [new 030] ViCO: A Training Strategy towards Semantic Aware Dynamic High-Resolution
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型因视觉token增多导致推理成本高的问题，提出ViCO训练策略。通过多MLP连接器和视觉分辨率路由（ViR），根据图像语义复杂度动态调整视觉token数量，兼顾效率与性能。**

- **链接: [http://arxiv.org/pdf/2510.12793v1](http://arxiv.org/pdf/2510.12793v1)**

> **作者:** Long Cui; Weiyun Wang; Jie Shao; Zichen Wen; Gen Luo; Linfeng Zhang; Yanting Zhang; Yu Qiao; Wenhai Wang
>
> **摘要:** Existing Multimodal Large Language Models (MLLMs) suffer from increased inference costs due to the additional vision tokens introduced by image inputs. In this work, we propose Visual Consistency Learning (ViCO), a novel training algorithm that enables the model to represent images of varying semantic complexities using different numbers of vision tokens. The key idea behind our method is to employ multiple MLP connectors, each with a different image compression ratio, to downsample the vision tokens based on the semantic complexity of the image. During training, we minimize the KL divergence between the responses conditioned on different MLP connectors. At inference time, we introduce an image router, termed Visual Resolution Router (ViR), that automatically selects the appropriate compression rate for each image patch. Compared with existing dynamic high-resolution strategies, which adjust the number of visual tokens based on image resolutions, our method dynamically adapts the number of visual tokens according to semantic complexity. Experimental results demonstrate that our method can reduce the number of vision tokens by up to 50% while maintaining the model's perception, reasoning, and OCR capabilities. We hope this work will contribute to the development of more efficient MLLMs. The code and models will be released to facilitate future research.
>
---
#### [new 031] CurriFlow: Curriculum-Guided Depth Fusion with Optical Flow-Based Temporal Alignment for 3D Semantic Scene Completion
- **分类: cs.CV**

- **简介: 该论文研究3D语义场景补全（SSC），旨在从单目图像恢复完整三维几何与语义。针对现有方法在运动推理、遮挡处理和深度监督噪声上的不足，提出CurriFlow框架，结合光流对齐与课程学习融合策略，提升时序一致性与几何鲁棒性，在SemanticKITTI上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2510.12362v1](http://arxiv.org/pdf/2510.12362v1)**

> **作者:** Jinzhou Lin; Jie Zhou; Wenhao Xu; Rongtao Xu; Changwei Wang; Shunpeng Chen; Kexue Fu; Yihua Shao; Li Guo; Shibiao Xu
>
> **摘要:** Semantic Scene Completion (SSC) aims to infer complete 3D geometry and semantics from monocular images, serving as a crucial capability for camera-based perception in autonomous driving. However, existing SSC methods relying on temporal stacking or depth projection often lack explicit motion reasoning and struggle with occlusions and noisy depth supervision. We propose CurriFlow, a novel semantic occupancy prediction framework that integrates optical flow-based temporal alignment with curriculum-guided depth fusion. CurriFlow employs a multi-level fusion strategy to align segmentation, visual, and depth features across frames using pre-trained optical flow, thereby improving temporal consistency and dynamic object understanding. To enhance geometric robustness, a curriculum learning mechanism progressively transitions from sparse yet accurate LiDAR depth to dense but noisy stereo depth during training, ensuring stable optimization and seamless adaptation to real-world deployment. Furthermore, semantic priors from the Segment Anything Model (SAM) provide category-agnostic supervision, strengthening voxel-level semantic learning and spatial consistency. Experiments on the SemanticKITTI benchmark demonstrate that CurriFlow achieves state-of-the-art performance with a mean IoU of 16.9, validating the effectiveness of our motion-guided and curriculum-aware design for camera-based 3D semantic scene completion.
>
---
#### [new 032] SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models
- **分类: cs.CV; cs.CL; I.4.0**

- **简介: 该论文针对统一多模态模型中视觉理解与生成能力不匹配的问题，提出SRUM框架，利用模型自身的理解模块作为奖励信号，通过全局-局部双奖励机制实现生成模块的自我优化，提升图文生成质量与推理能力。**

- **链接: [http://arxiv.org/pdf/2510.12784v1](http://arxiv.org/pdf/2510.12784v1)**

> **作者:** Weiyang Jin; Yuwei Niu; Jiaqi Liao; Chengqi Duan; Aoxue Li; Shenghua Gao; Xihui Liu
>
> **备注:** 20 pages, 8 figures, webpage can be seen in https://waynejin0918.github.io/srum_web/
>
> **摘要:** Recently, remarkable progress has been made in Unified Multimodal Models (UMMs), which integrate vision-language generation and understanding capabilities within a single framework. However, a significant gap exists where a model's strong visual understanding often fails to transfer to its visual generation. A model might correctly understand an image based on user instructions, yet be unable to generate a faithful image from text prompts. This phenomenon directly raises a compelling question: Can a model achieve self-improvement by using its understanding module to reward its generation module? To bridge this gap and achieve self-improvement, we introduce SRUM, a self-rewarding post-training framework that can be directly applied to existing UMMs of various designs. SRUM creates a feedback loop where the model's own understanding module acts as an internal ``evaluator'', providing corrective signals to improve its generation module, without requiring additional human-labeled data. To ensure this feedback is comprehensive, we designed a global-local dual reward system. To tackle the inherent structural complexity of images, this system offers multi-scale guidance: a \textbf{global reward} ensures the correctness of the overall visual semantics and layout, while a \textbf{local reward} refines fine-grained, object-level fidelity. SRUM leads to powerful capabilities and shows strong generalization, boosting performance on T2I-CompBench from 82.18 to \textbf{88.37} and on T2I-ReasonBench from 43.82 to \textbf{46.75}. Overall, our work establishes a powerful new paradigm for enabling a UMMs' understanding module to guide and enhance its own generation via self-rewarding.
>
---
#### [new 033] ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation
- **分类: cs.CV**

- **简介: 该论文针对检索增强图像生成中私有图像数据的 unauthorized 使用问题，提出ImageSentinel框架。通过生成带随机字符密钥的视觉一致哨兵图像，结合视觉-语言模型实现数据保护与验证，有效检测未授权使用，同时不影响合法生成质量。**

- **链接: [http://arxiv.org/pdf/2510.12119v1](http://arxiv.org/pdf/2510.12119v1)**

> **作者:** Ziyuan Luo; Yangyi Zhao; Ka Chun Cheung; Simon See; Renjie Wan
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** The widespread adoption of Retrieval-Augmented Image Generation (RAIG) has raised significant concerns about the unauthorized use of private image datasets. While these systems have shown remarkable capabilities in enhancing generation quality through reference images, protecting visual datasets from unauthorized use in such systems remains a challenging problem. Traditional digital watermarking approaches face limitations in RAIG systems, as the complex feature extraction and recombination processes fail to preserve watermark signals during generation. To address these challenges, we propose ImageSentinel, a novel framework for protecting visual datasets in RAIG. Our framework synthesizes sentinel images that maintain visual consistency with the original dataset. These sentinels enable protection verification through randomly generated character sequences that serve as retrieval keys. To ensure seamless integration, we leverage vision-language models to generate the sentinel images. Experimental results demonstrate that ImageSentinel effectively detects unauthorized dataset usage while preserving generation quality for authorized applications. Code is available at https://github.com/luo-ziyuan/ImageSentinel.
>
---
#### [new 034] E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization
- **分类: cs.CV**

- **简介: 该论文研究事件相机下的自运动与光流联合估计，解决无监督下因数据关联弱导致的病态问题。提出E-MoFlow框架，通过隐式时空与几何正则化，实现6-DoF自运动和光流的联合优化，无需显式深度或光滑性先验，性能达SOTA。**

- **链接: [http://arxiv.org/pdf/2510.12753v1](http://arxiv.org/pdf/2510.12753v1)**

> **作者:** Wenpu Li; Bangyan Liao; Yi Zhou; Qi Xu; Pian Wan; Peidong Liu
>
> **备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems(NeurIPS 2025)
>
> **摘要:** The estimation of optical flow and 6-DoF ego-motion, two fundamental tasks in 3D vision, has typically been addressed independently. For neuromorphic vision (e.g., event cameras), however, the lack of robust data association makes solving the two problems separately an ill-posed challenge, especially in the absence of supervision via ground truth. Existing works mitigate this ill-posedness by either enforcing the smoothness of the flow field via an explicit variational regularizer or leveraging explicit structure-and-motion priors in the parametrization to improve event alignment. The former notably introduces bias in results and computational overhead, while the latter, which parametrizes the optical flow in terms of the scene depth and the camera motion, often converges to suboptimal local minima. To address these issues, we propose an unsupervised framework that jointly optimizes egomotion and optical flow via implicit spatial-temporal and geometric regularization. First, by modeling camera's egomotion as a continuous spline and optical flow as an implicit neural representation, our method inherently embeds spatial-temporal coherence through inductive biases. Second, we incorporate structure-and-motion priors through differential geometric constraints, bypassing explicit depth estimation while maintaining rigorous geometric consistency. As a result, our framework (called E-MoFlow) unifies egomotion and optical flow estimation via implicit regularization under a fully unsupervised paradigm. Experiments demonstrate its versatility to general 6-DoF motion scenarios, achieving state-of-the-art performance among unsupervised methods and competitive even with supervised approaches.
>
---
#### [new 035] Towards General Urban Monitoring with Vision-Language Models: A Review, Evaluation, and a Research Agenda
- **分类: cs.CV**

- **简介: 该论文综述并评估视觉-语言模型（VLM）在城市公共设施监测中的零样本应用，旨在让机器“像市民一样看”。研究通过系统分析32项文献，探讨任务类型、模型架构、数据集及评估方法，提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2510.12400v1](http://arxiv.org/pdf/2510.12400v1)**

> **作者:** André Torneiro; Diogo Monteiro; Paulo Novais; Pedro Rangel Henriques; Nuno F. Rodrigues
>
> **备注:** 44 pages
>
> **摘要:** Urban monitoring of public infrastructure (such as waste bins, road signs, vegetation, sidewalks, and construction sites) poses significant challenges due to the diversity of objects, environments, and contextual conditions involved. Current state-of-the-art approaches typically rely on a combination of IoT sensors and manual inspections, which are costly, difficult to scale, and often misaligned with citizens' perception formed through direct visual observation. This raises a critical question: Can machines now "see" like citizens and infer informed opinions about the condition of urban infrastructure? Vision-Language Models (VLMs), which integrate visual understanding with natural language reasoning, have recently demonstrated impressive capabilities in processing complex visual information, turning them into a promising technology to address this challenge. This systematic review investigates the role of VLMs in urban monitoring, with particular emphasis on zero-shot applications. Following the PRISMA methodology, we analyzed 32 peer-reviewed studies published between 2021 and 2025 to address four core research questions: (1) What urban monitoring tasks have been effectively addressed using VLMs? (2) Which VLM architectures and frameworks are most commonly used and demonstrate superior performance? (3) What datasets and resources support this emerging field? (4) How are VLM-based applications evaluated, and what performance levels have been reported?
>
---
#### [new 036] Advancing End-to-End Pixel Space Generative Modeling via Self-supervised Pre-training
- **分类: cs.CV**

- **简介: 该论文研究像素空间生成模型，旨在解决其训练难、性能低的问题。提出两阶段自监督预训练框架，先对编码器预训练，再与解码器联合微调，显著提升扩散与一致性模型在ImageNet上的生成质量与效率。**

- **链接: [http://arxiv.org/pdf/2510.12586v1](http://arxiv.org/pdf/2510.12586v1)**

> **作者:** Jiachen Lei; Keli Liu; Julius Berner; Haiming Yu; Hongkai Zheng; Jiahong Wu; Xiangxiang Chu
>
> **摘要:** Pixel-space generative models are often more difficult to train and generally underperform compared to their latent-space counterparts, leaving a persistent performance and efficiency gap. In this paper, we introduce a novel two-stage training framework that closes this gap for pixel-space diffusion and consistency models. In the first stage, we pre-train encoders to capture meaningful semantics from clean images while aligning them with points along the same deterministic sampling trajectory, which evolves points from the prior to the data distribution. In the second stage, we integrate the encoder with a randomly initialized decoder and fine-tune the complete model end-to-end for both diffusion and consistency models. Our training framework demonstrates strong empirical performance on ImageNet dataset. Specifically, our diffusion model reaches an FID of 2.04 on ImageNet-256 and 2.35 on ImageNet-512 with 75 number of function evaluations (NFE), surpassing prior pixel-space methods by a large margin in both generation quality and efficiency while rivaling leading VAE-based models at comparable training cost. Furthermore, on ImageNet-256, our consistency model achieves an impressive FID of 8.82 in a single sampling step, significantly surpassing its latent-space counterpart. To the best of our knowledge, this marks the first successful training of a consistency model directly on high-resolution images without relying on pre-trained VAEs or diffusion models.
>
---
#### [new 037] G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior
- **分类: cs.CV**

- **简介: 该论文聚焦3D场景重建，旨在解决生成先验方法中几何监督不足和多视角不一致的问题。作者提出G4Splat，利用平面结构提供度量深度监督，并在生成过程中引入几何引导，提升可见性估计、视图选择与补全一致性，显著改善观测与未观测区域的重建质量。**

- **链接: [http://arxiv.org/pdf/2510.12099v1](http://arxiv.org/pdf/2510.12099v1)**

> **作者:** Junfeng Ni; Yixin Chen; Zhifei Yang; Yu Liu; Ruijie Lu; Song-Chun Zhu; Siyuan Huang
>
> **备注:** Project page: https://dali-jack.github.io/g4splat-web/
>
> **摘要:** Despite recent advances in leveraging generative prior from pre-trained diffusion models for 3D scene reconstruction, existing methods still face two critical limitations. First, due to the lack of reliable geometric supervision, they struggle to produce high-quality reconstructions even in observed regions, let alone in unobserved areas. Second, they lack effective mechanisms to mitigate multi-view inconsistencies in the generated images, leading to severe shape-appearance ambiguities and degraded scene geometry. In this paper, we identify accurate geometry as the fundamental prerequisite for effectively exploiting generative models to enhance 3D scene reconstruction. We first propose to leverage the prevalence of planar structures to derive accurate metric-scale depth maps, providing reliable supervision in both observed and unobserved regions. Furthermore, we incorporate this geometry guidance throughout the generative pipeline to improve visibility mask estimation, guide novel view selection, and enhance multi-view consistency when inpainting with video diffusion models, resulting in accurate and consistent scene completion. Extensive experiments on Replica, ScanNet++, and DeepBlending show that our method consistently outperforms existing baselines in both geometry and appearance reconstruction, particularly for unobserved regions. Moreover, our method naturally supports single-view inputs and unposed videos, with strong generalizability in both indoor and outdoor scenarios with practical real-world applicability. The project page is available at https://dali-jack.github.io/g4splat-web/.
>
---
#### [new 038] Efficient Real-World Deblurring using Single Images: AIM 2025 Challenge Report
- **分类: cs.CV**

- **简介: 该论文针对真实场景单图像去模糊任务，旨在解决高效去模糊问题。基于RSBlur新测试集，要求模型参数少于500万、计算量低于200 GMACs。论文综述了挑战赛情况，分析了4支队伍的方案，最高达31.1298 dB PSNR。**

- **链接: [http://arxiv.org/pdf/2510.12788v1](http://arxiv.org/pdf/2510.12788v1)**

> **作者:** Daniel Feijoo; Paula Garrido-Mellado; Marcos V. Conde; Jaesung Rim; Alvaro Garcia; Sunghyun Cho; Radu Timofte
>
> **备注:** ICCV 2025 - AIM Workshop
>
> **摘要:** This paper reviews the AIM 2025 Efficient Real-World Deblurring using Single Images Challenge, which aims to advance in efficient real-blur restoration. The challenge is based on a new test set based on the well known RSBlur dataset. Pairs of blur and degraded images in this dataset are captured using a double-camera system. Participant were tasked with developing solutions to effectively deblur these type of images while fulfilling strict efficiency constraints: fewer than 5 million model parameters and a computational budget under 200 GMACs. A total of 71 participants registered, with 4 teams finally submitting valid solutions. The top-performing approach achieved a PSNR of 31.1298 dB, showcasing the potential of efficient methods in this domain. This paper provides a comprehensive overview of the challenge, compares the proposed solutions, and serves as a valuable reference for researchers in efficient real-world image deblurring.
>
---
#### [new 039] CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究多模态大模型的知识蒸馏任务，旨在解决学生模型视觉感知能力弱的问题。作者发现主因是视觉注意力不对齐，提出CompoDistill框架，通过显式对齐师生视觉注意力，提升学生模型在组合推理任务中的表现。**

- **链接: [http://arxiv.org/pdf/2510.12184v1](http://arxiv.org/pdf/2510.12184v1)**

> **作者:** Jiwan Kim; Kibum Kim; Sangwoo Seo; Chanyoung Park
>
> **备注:** Preprint. Under Review
>
> **摘要:** Recently, efficient Multimodal Large Language Models (MLLMs) have gained significant attention as a solution to their high computational complexity, making them more practical for real-world applications. In this regard, the knowledge distillation (KD) approach has emerged as a promising alternative, which transfers the rich visual and linguistic knowledge from a larger model (teacher) to a smaller model (student). However, we observe that existing KD methods struggle to effectively distill the teacher MLLM's rich visual perception abilities to the student, a challenge that has been largely overlooked in previous studies. Through a systematic analysis, we identify visual attention misalignment between student and teacher as the main cause of this issue. Based on this insight, we propose CompoDistill, a novel KD framework that explicitly aligns the student's visual attention with that of the teacher to enhance the student's visual perception abilities. Our extensive experiments show that CompoDistill significantly improves performance on compositional reasoning tasks that require visual perception abilities while maintaining strong performance on visual question answering tasks, as done in existing studies. Furthermore, CompoDistill demonstrates effectiveness with a more advanced backbone, highlighting its generalizability.
>
---
#### [new 040] EReLiFM: Evidential Reliability-Aware Residual Flow Meta-Learning for Open-Set Domain Generalization under Noisy Labels
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文研究开放集域泛化下的噪声标签问题，提出EReLiFM方法。通过证据性损失聚类提升标签可靠性，结合残差流匹配建模域与类别差异，实现更鲁棒的知识迁移。**

- **链接: [http://arxiv.org/pdf/2510.12687v1](http://arxiv.org/pdf/2510.12687v1)**

> **作者:** Kunyu Peng; Di Wen; Kailun Yang; Jia Fu; Yufan Chen; Ruiping Liu; Jiamin Wu; Junwei Zheng; M. Saquib Sarfraz; Luc Van Gool; Danda Pani Paudel; Rainer Stiefelhagen
>
> **备注:** The source code is available at https://github.com/KPeng9510/ERELIFM
>
> **摘要:** Open-Set Domain Generalization (OSDG) aims to enable deep learning models to recognize unseen categories in new domains, which is crucial for real-world applications. Label noise hinders open-set domain generalization by corrupting source-domain knowledge, making it harder to recognize known classes and reject unseen ones. While existing methods address OSDG under Noisy Labels (OSDG-NL) using hyperbolic prototype-guided meta-learning, they struggle to bridge domain gaps, especially with limited clean labeled data. In this paper, we propose Evidential Reliability-Aware Residual Flow Meta-Learning (EReLiFM). We first introduce an unsupervised two-stage evidential loss clustering method to promote label reliability awareness. Then, we propose a residual flow matching mechanism that models structured domain- and category-conditioned residuals, enabling diverse and uncertainty-aware transfer paths beyond interpolation-based augmentation. During this meta-learning process, the model is optimized such that the update direction on the clean set maximizes the loss decrease on the noisy set, using pseudo labels derived from the most confident predicted class for supervision. Experimental results show that EReLiFM outperforms existing methods on OSDG-NL, achieving state-of-the-art performance. The source code is available at https://github.com/KPeng9510/ERELIFM.
>
---
#### [new 041] VideoLucy: Deep Memory Backtracking for Long Video Understanding
- **分类: cs.CV**

- **简介: 该论文针对长视频理解中时序建模不足和关键信息丢失问题，提出VideoLucy框架，采用分层记忆结构与回溯机制，实现从粗到细的深度记忆挖掘，并构建新基准EgoMem。方法在开源模型上超越现有技术，甚至优于GPT-4o。**

- **链接: [http://arxiv.org/pdf/2510.12422v1](http://arxiv.org/pdf/2510.12422v1)**

> **作者:** Jialong Zuo; Yongtai Deng; Lingdong Kong; Jingkang Yang; Rui Jin; Yiwei Zhang; Nong Sang; Liang Pan; Ziwei Liu; Changxin Gao
>
> **备注:** NeurIPS-2025 Accepted Paper
>
> **摘要:** Recent studies have shown that agent-based systems leveraging large language models (LLMs) for key information retrieval and integration have emerged as a promising approach for long video understanding. However, these systems face two major challenges. First, they typically perform modeling and reasoning on individual frames, struggling to capture the temporal context of consecutive frames. Second, to reduce the cost of dense frame-level captioning, they adopt sparse frame sampling, which risks discarding crucial information. To overcome these limitations, we propose VideoLucy, a deep memory backtracking framework for long video understanding. Inspired by the human recollection process from coarse to fine, VideoLucy employs a hierarchical memory structure with progressive granularity. This structure explicitly defines the detail level and temporal scope of memory at different hierarchical depths. Through an agent-based iterative backtracking mechanism, VideoLucy systematically mines video-wide, question-relevant deep memories until sufficient information is gathered to provide a confident answer. This design enables effective temporal understanding of consecutive frames while preserving critical details. In addition, we introduce EgoMem, a new benchmark for long video understanding. EgoMem is designed to comprehensively evaluate a model's ability to understand complex events that unfold over time and capture fine-grained details in extremely long videos. Extensive experiments demonstrate the superiority of VideoLucy. Built on open-source models, VideoLucy significantly outperforms state-of-the-art methods on multiple long video understanding benchmarks, achieving performance even surpassing the latest proprietary models such as GPT-4o. Our code and dataset will be made publicly at https://videolucy.github.io
>
---
#### [new 042] What If : Understanding Motion Through Sparse Interactions
- **分类: cs.CV**

- **简介: 该论文研究物理场景中稀疏交互下的运动建模，提出Flow Poke Transformer（FPT），直接预测局部运动分布，解决传统方法难以捕捉多模态动态与不确定性的难题，实现可解释的运动预测，并在多种下游任务中展现优越性能。**

- **链接: [http://arxiv.org/pdf/2510.12777v1](http://arxiv.org/pdf/2510.12777v1)**

> **作者:** Stefan Andreas Baumann; Nick Stracke; Timy Phan; Björn Ommer
>
> **备注:** Project page and code: https://compvis.github.io/flow-poke-transformer
>
> **摘要:** Understanding the dynamics of a physical scene involves reasoning about the diverse ways it can potentially change, especially as a result of local interactions. We present the Flow Poke Transformer (FPT), a novel framework for directly predicting the distribution of local motion, conditioned on sparse interactions termed "pokes". Unlike traditional methods that typically only enable dense sampling of a single realization of scene dynamics, FPT provides an interpretable directly accessible representation of multi-modal scene motion, its dependency on physical interactions and the inherent uncertainties of scene dynamics. We also evaluate our model on several downstream tasks to enable comparisons with prior methods and highlight the flexibility of our approach. On dense face motion generation, our generic pre-trained model surpasses specialized baselines. FPT can be fine-tuned in strongly out-of-distribution tasks such as synthetic datasets to enable significant improvements over in-domain methods in articulated object motion estimation. Additionally, predicting explicit motion distributions directly enables our method to achieve competitive performance on tasks like moving part segmentation from pokes which further demonstrates the versatility of our FPT. Code and models are publicly available at https://compvis.github.io/flow-poke-transformer.
>
---
#### [new 043] MS-GAGA: Metric-Selective Guided Adversarial Generation Attack
- **分类: cs.CV**

- **简介: 该论文针对黑盒环境下深度伪造检测器的攻击任务，提出MS-GAGA框架。通过双流生成和度量感知选择，提升对抗样本的可迁移性与视觉不可察觉性，有效增强了对未见模型的攻击成功率。**

- **链接: [http://arxiv.org/pdf/2510.12468v1](http://arxiv.org/pdf/2510.12468v1)**

> **作者:** Dion J. X. Ho; Gabriel Lee Jun Rong; Niharika Shrivastava; Harshavardhan Abichandani; Pai Chet Ng; Xiaoxiao Miao
>
> **摘要:** We present MS-GAGA (Metric-Selective Guided Adversarial Generation Attack), a two-stage framework for crafting transferable and visually imperceptible adversarial examples against deepfake detectors in black-box settings. In Stage 1, a dual-stream attack module generates adversarial candidates: MNTD-PGD applies enhanced gradient calculations optimized for small perturbation budgets, while SG-PGD focuses perturbations on visually salient regions. This complementary design expands the adversarial search space and improves transferability across unseen models. In Stage 2, a metric-aware selection module evaluates candidates based on both their success against black-box models and their structural similarity (SSIM) to the original image. By jointly optimizing transferability and imperceptibility, MS-GAGA achieves up to 27% higher misclassification rates on unseen detectors compared to state-of-the-art attacks.
>
---
#### [new 044] Voronoi-Assisted Diffusion for Computing Unsigned Distance Fields from Unoriented Points
- **分类: cs.CV**

- **简介: 该论文研究从无向点云计算无符号距离场（UDF）的任务，旨在解决现有神经方法的不稳定性与高成本问题。提出无需网络的Voronoi辅助扩散方法，通过Voronoi引导法线对齐与扩散积分高效稳定生成UDF。**

- **链接: [http://arxiv.org/pdf/2510.12524v1](http://arxiv.org/pdf/2510.12524v1)**

> **作者:** Jiayi Kong; Chen Zong; Junkai Deng; Xuhui Chen; Fei Hou; Shiqing Xin; Junhui Hou; Chen Qian; Ying He
>
> **摘要:** Unsigned Distance Fields (UDFs) provide a flexible representation for 3D shapes with arbitrary topology, including open and closed surfaces, orientable and non-orientable geometries, and non-manifold structures. While recent neural approaches have shown promise in learning UDFs, they often suffer from numerical instability, high computational cost, and limited controllability. We present a lightweight, network-free method, Voronoi-Assisted Diffusion (VAD), for computing UDFs directly from unoriented point clouds. Our approach begins by assigning bi-directional normals to input points, guided by two Voronoi-based geometric criteria encoded in an energy function for optimal alignment. The aligned normals are then diffused to form an approximate UDF gradient field, which is subsequently integrated to recover the final UDF. Experiments demonstrate that VAD robustly handles watertight and open surfaces, as well as complex non-manifold and non-orientable geometries, while remaining computationally efficient and stable.
>
---
#### [new 045] Local Background Features Matter in Out-of-Distribution Detection
- **分类: cs.CV**

- **简介: 该论文研究分布外检测任务，旨在缓解模型对OOD样本的过置信问题。作者提出利用ID图像的局部背景特征作为伪OOD特征训练模型，通过优化降低其L2范数，有效提升OOD检测性能，无需额外数据收集。**

- **链接: [http://arxiv.org/pdf/2510.12259v1](http://arxiv.org/pdf/2510.12259v1)**

> **作者:** Jinlun Ye; Zhuohao Sun; Yiqiao Qiu; Qiu Li; Zhijun Tan; Ruixuan Wang
>
> **摘要:** Out-of-distribution (OOD) detection is crucial when deploying deep neural networks in the real world to ensure the reliability and safety of their applications. One main challenge in OOD detection is that neural network models often produce overconfident predictions on OOD data. While some methods using auxiliary OOD datasets or generating fake OOD images have shown promising OOD detection performance, they are limited by the high costs of data collection and training. In this study, we propose a novel and effective OOD detection method that utilizes local background features as fake OOD features for model training. Inspired by the observation that OOD images generally share similar background regions with ID images, the background features are extracted from ID images as simulated OOD visual representations during training based on the local invariance of convolution. Through being optimized to reduce the $L_2$-norm of these background features, the neural networks are able to alleviate the overconfidence issue on OOD data. Extensive experiments on multiple standard OOD detection benchmarks confirm the effectiveness of our method and its wide combinatorial compatibility with existing post-hoc methods, with new state-of-the-art performance achieved from our method.
>
---
#### [new 046] Efficient Perceptual Image Super Resolution: AIM 2025 Study and Benchmark
- **分类: cs.CV**

- **简介: 该论文研究高效感知图像超分辨率（EPSR），旨在提升感知质量的同时满足5M参数和2000 GFLOPs的效率约束。提出新基准和测试集，评估方法在无真值4K图像下的性能，实现了优于Real-ESRGAN的效果，建立了现代高效感知超分基线。**

- **链接: [http://arxiv.org/pdf/2510.12765v1](http://arxiv.org/pdf/2510.12765v1)**

> **作者:** Bruno Longarela; Marcos V. Conde; Alvaro Garcia; Radu Timofte
>
> **备注:** ICCV 2025 - AIM Workshop
>
> **摘要:** This paper presents a comprehensive study and benchmark on Efficient Perceptual Super-Resolution (EPSR). While significant progress has been made in efficient PSNR-oriented super resolution, approaches focusing on perceptual quality metrics remain relatively inefficient. Motivated by this gap, we aim to replicate or improve the perceptual results of Real-ESRGAN while meeting strict efficiency constraints: a maximum of 5M parameters and 2000 GFLOPs, calculated for an input size of 960x540 pixels. The proposed solutions were evaluated on a novel dataset consisting of 500 test images of 4K resolution, each degraded using multiple degradation types, without providing the original high-quality counterparts. This design aims to reflect realistic deployment conditions and serves as a diverse and challenging benchmark. The top-performing approach manages to outperform Real-ESRGAN across all benchmark datasets, demonstrating the potential of efficient methods in the perceptual domain. This paper establishes the modern baselines for efficient perceptual super resolution.
>
---
#### [new 047] PAGS: Priority-Adaptive Gaussian Splatting for Dynamic Driving Scenes
- **分类: cs.CV**

- **简介: 该论文针对动态驾驶场景三维重建中效率与精度难平衡的问题，提出优先级自适应高斯点阵化（PAGS）方法。通过语义引导剪枝与优先渲染机制，在保障关键物体精度的同时显著提升计算效率。**

- **链接: [http://arxiv.org/pdf/2510.12282v1](http://arxiv.org/pdf/2510.12282v1)**

> **作者:** Ying A; Wenzhang Sun; Chang Zeng; Chunfeng Wang; Hao Li; Jianxun Cui
>
> **摘要:** Reconstructing dynamic 3D urban scenes is crucial for autonomous driving, yet current methods face a stark trade-off between fidelity and computational cost. This inefficiency stems from their semantically agnostic design, which allocates resources uniformly, treating static backgrounds and safety-critical objects with equal importance. To address this, we introduce Priority-Adaptive Gaussian Splatting (PAGS), a framework that injects task-aware semantic priorities directly into the 3D reconstruction and rendering pipeline. PAGS introduces two core contributions: (1) Semantically-Guided Pruning and Regularization strategy, which employs a hybrid importance metric to aggressively simplify non-critical scene elements while preserving fine-grained details on objects vital for navigation. (2) Priority-Driven Rendering pipeline, which employs a priority-based depth pre-pass to aggressively cull occluded primitives and accelerate the final shading computations. Extensive experiments on the Waymo and KITTI datasets demonstrate that PAGS achieves exceptional reconstruction quality, particularly on safety-critical objects, while significantly reducing training time and boosting rendering speeds to over 350 FPS.
>
---
#### [new 048] Vision Language Models Map Logos to Text via Semantic Entanglement in the Visual Projector
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型（VLM）在无文本标志中误生成品牌名称的“标志幻觉”问题。通过构建数据集与扰动实验，发现幻觉源于视觉投影器中的语义纠缠，并提出通过子空间消融缓解该问题，提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2510.12287v1](http://arxiv.org/pdf/2510.12287v1)**

> **作者:** Sifan Li; Hongkai Chen; Yujun Cai; Qingwen Ye; Liyang Chen; Junsong Yuan; Yiwei Wang
>
> **摘要:** Vision Language Models (VLMs) have achieved impressive progress in multimodal reasoning; yet, they remain vulnerable to hallucinations, where outputs are not grounded in visual evidence. In this paper, we investigate a previously overlooked setting: logo hallucination, where models generate brand names or textual content despite logos containing no visible words. Using curated splits of pure symbols, hybrids, and text-bearing logos, as well as the challenging Hard-60 subset, we systematically measure hallucination across leading VLMs. We further probe robustness through nine structured perturbations and show that hallucinations persist even under strong distortions, with occlusion exposing the sharpest weaknesses. Embedding-level analysis with open-weight LLaVA demonstrates that hallucination is tied to a small subset of projector dimensions, and targeted ablation substantially reduces errors while preserving OCR accuracy. Together, these findings reveal that VLMs often rely on symbolic priors rather than genuine glyph perception, particularly for iconic circular logos, and that projector subspaces play a decisive role in this failure mode. Our work contributes both a novel diagnostic lens and actionable mitigation insights, highlighting projector disentanglement and OCR-guided decoding as promising directions for building more trustworthy multimodal systems.
>
---
#### [new 049] FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution
- **分类: cs.CV**

- **简介: 该论文研究视频超分辨率（VSR）任务，旨在解决扩散模型在实时性、计算量和高分辨率泛化上的瓶颈。作者提出FlashVSR，通过蒸馏训练、稀疏注意力和轻量解码器实现高效流式单步超分，达到实时性能。**

- **链接: [http://arxiv.org/pdf/2510.12747v1](http://arxiv.org/pdf/2510.12747v1)**

> **作者:** Junhao Zhuang; Shi Guo; Xin Cai; Xiaohui Li; Yihao Liu; Chun Yuan; Tianfan Xue
>
> **备注:** Project page with code: https://zhuang2002.github.io/FlashVSR
>
> **摘要:** Diffusion models have recently advanced video restoration, but applying them to real-world video super-resolution (VSR) remains challenging due to high latency, prohibitive computation, and poor generalization to ultra-high resolutions. Our goal in this work is to make diffusion-based VSR practical by achieving efficiency, scalability, and real-time performance. To this end, we propose FlashVSR, the first diffusion-based one-step streaming framework towards real-time VSR. FlashVSR runs at approximately 17 FPS for 768x1408 videos on a single A100 GPU by combining three complementary innovations: (i) a train-friendly three-stage distillation pipeline that enables streaming super-resolution, (ii) locality-constrained sparse attention that cuts redundant computation while bridging the train-test resolution gap, and (iii) a tiny conditional decoder that accelerates reconstruction without sacrificing quality. To support large-scale training, we also construct VSR-120K, a new dataset with 120k videos and 180k images. Extensive experiments show that FlashVSR scales reliably to ultra-high resolutions and achieves state-of-the-art performance with up to 12x speedup over prior one-step diffusion VSR models. We will release the code, pretrained models, and dataset to foster future research in efficient diffusion-based VSR.
>
---
#### [new 050] Vectorized Video Representation with Easy Editing via Hierarchical Spatio-Temporally Consistent Proxy Embedding
- **分类: cs.CV**

- **简介: 该论文提出一种基于分层时空一致代理节点的视频表示方法，旨在解决传统像素级跟踪在运动建模中的不稳定性问题。通过解耦形状与纹理编码，实现鲁棒的对象表达与精细可控的视频编辑，支持高质量重建与复杂编辑任务。**

- **链接: [http://arxiv.org/pdf/2510.12256v1](http://arxiv.org/pdf/2510.12256v1)**

> **作者:** Ye Chen; Liming Tan; Yupeng Zhu; Yuanbin Wang; Bingbing Ni
>
> **摘要:** Current video representations heavily rely on unstable and over-grained priors for motion and appearance modelling, \emph{i.e.}, pixel-level matching and tracking. A tracking error of just a few pixels would lead to the collapse of the visual object representation, not to mention occlusions and large motion frequently occurring in videos. To overcome the above mentioned vulnerability, this work proposes spatio-temporally consistent proxy nodes to represent dynamically changing objects/scenes in the video. On the one hand, the hierarchical proxy nodes have the ability to stably express the multi-scale structure of visual objects, so they are not affected by accumulated tracking error, long-term motion, occlusion, and viewpoint variation. On the other hand, the dynamic representation update mechanism of the proxy nodes adequately leverages spatio-temporal priors of the video to mitigate the impact of inaccurate trackers, thereby effectively handling drastic changes in scenes and objects. Additionally, the decoupled encoding manner of the shape and texture representations across different visual objects in the video facilitates controllable and fine-grained appearance editing capability. Extensive experiments demonstrate that the proposed representation achieves high video reconstruction accuracy with fewer parameters and supports complex video processing tasks, including video in-painting and keyframe-based temporally consistent video editing.
>
---
#### [new 051] The Impact of Synthetic Data on Object Detection Model Performance: A Comparative Analysis with Real-World Data
- **分类: cs.CV**

- **简介: 该论文研究合成数据对物体检测模型性能的影响，解决仓库物流中因真实数据不足导致的模型训练难题。通过NVIDIA Omniverse生成合成数据，对比不同数据组合下的检测效果，验证合成数据的有效性。**

- **链接: [http://arxiv.org/pdf/2510.12208v1](http://arxiv.org/pdf/2510.12208v1)**

> **作者:** Muammer Bay; Timo von Marcard; Dren Fazlija
>
> **备注:** 18 pages, 12 figures, 2 tables. Code: https://github.com/MuammerBay/omniverse-replicator-sim2real-analysis ; Data: https://doi.org/10.5281/zenodo.17308406
>
> **摘要:** Recent advances in generative AI, particularly in computer vision (CV), offer new opportunities to optimize workflows across industries, including logistics and manufacturing. However, many AI applications are limited by a lack of expertise and resources, which forces a reliance on general-purpose models. Success with these models often requires domain-specific data for fine-tuning, which can be costly and inefficient. Thus, using synthetic data for fine-tuning is a popular, cost-effective alternative to gathering real-world data. This work investigates the impact of synthetic data on the performance of object detection models, compared to models trained on real-world data only, specifically within the domain of warehouse logistics. To this end, we examined the impact of synthetic data generated using the NVIDIA Omniverse Replicator tool on the effectiveness of object detection models in real-world scenarios. It comprises experiments focused on pallet detection in a warehouse setting, utilizing both real and various synthetic dataset generation strategies. Our findings provide valuable insights into the practical applications of synthetic image data in computer vision, suggesting that a balanced integration of synthetic and real data can lead to robust and efficient object detection models.
>
---
#### [new 052] VQArt-Bench: A semantically rich VQA Benchmark for Art and Cultural Heritage
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对艺术与文化遗产领域的视觉问答（VQA）任务，提出新基准VQArt-Bench，旨在解决现有数据集语义浅、多样性不足的问题。作者构建多智能体流水线生成富含语义、多样且经验证的问题，评估显示当前多模态大模型在深层视觉理解方面存在显著局限。**

- **链接: [http://arxiv.org/pdf/2510.12750v1](http://arxiv.org/pdf/2510.12750v1)**

> **作者:** A. Alfarano; L. Venturoli; D. Negueruela del Castillo
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated significant capabilities in joint visual and linguistic tasks. However, existing Visual Question Answering (VQA) benchmarks often fail to evaluate deep semantic understanding, particularly in complex domains like visual art analysis. Confined to simple syntactic structures and surface-level attributes, these questions fail to capture the diversity and depth of human visual inquiry. This limitation incentivizes models to exploit statistical shortcuts rather than engage in visual reasoning. To address this gap, we introduce VQArt-Bench, a new, large-scale VQA benchmark for the cultural heritage domain. This benchmark is constructed using a novel multi-agent pipeline where specialized agents collaborate to generate nuanced, validated, and linguistically diverse questions. The resulting benchmark is structured along relevant visual understanding dimensions that probe a model's ability to interpret symbolic meaning, narratives, and complex visual relationships. Our evaluation of 14 state-of-the-art MLLMs on this benchmark reveals significant limitations in current models, including a surprising weakness in simple counting tasks and a clear performance gap between proprietary and open-source models.
>
---
#### [new 053] Hybrid Gaussian Splatting for Novel Urban View Synthesis
- **分类: cs.CV**

- **简介: 该论文针对城市街景新视角合成任务，旨在从车载多视角图像生成不同行驶路径下的新视图。提出两阶段混合方法：先用高斯点阵进行3D场景重建与渲染，再通过单步扩散模型增强图像质量，最终在挑战赛中取得第二名。**

- **链接: [http://arxiv.org/pdf/2510.12308v1](http://arxiv.org/pdf/2510.12308v1)**

> **作者:** Mohamed Omran; Farhad Zanjani; Davide Abati; Jens Petersen; Amirhossein Habibian
>
> **备注:** ICCV 2025 RealADSim Workshop
>
> **摘要:** This paper describes the Qualcomm AI Research solution to the RealADSim-NVS challenge, hosted at the RealADSim Workshop at ICCV 2025. The challenge concerns novel view synthesis in street scenes, and participants are required to generate, starting from car-centric frames captured during some training traversals, renders of the same urban environment as viewed from a different traversal (e.g. different street lane or car direction). Our solution is inspired by hybrid methods in scene generation and generative simulators merging gaussian splatting and diffusion models, and it is composed of two stages: First, we fit a 3D reconstruction of the scene and render novel views as seen from the target cameras. Then, we enhance the resulting frames with a dedicated single-step diffusion model. We discuss specific choices made in the initialization of gaussian primitives as well as the finetuning of the enhancer model and its training data curation. We report the performance of our model design and we ablate its components in terms of novel view quality as measured by PSNR, SSIM and LPIPS. On the public leaderboard reporting test results, our proposal reaches an aggregated score of 0.432, achieving the second place overall.
>
---
#### [new 054] MCOP: Multi-UAV Collaborative Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文研究多无人机协同占据预测任务，旨在解决现有BEV方法语义几何信息不全及遮挡场景性能差的问题。提出包含空间感知编码、跨智能体特征融合、高度感知压缩与双掩码引导的框架，在自建数据集上实现了更优精度与更低通信开销。**

- **链接: [http://arxiv.org/pdf/2510.12679v1](http://arxiv.org/pdf/2510.12679v1)**

> **作者:** Zefu Lin; Wenbo Chen; Xiaojuan Jin; Yuran Yang; Lue Fan; Yixin Zhang; Yufeng Zhang; Zhaoxiang Zhang
>
> **摘要:** Unmanned Aerial Vehicle (UAV) swarm systems necessitate efficient collaborative perception mechanisms for diverse operational scenarios. Current Bird's Eye View (BEV)-based approaches exhibit two main limitations: bounding-box representations fail to capture complete semantic and geometric information of the scene, and their performance significantly degrades when encountering undefined or occluded objects. To address these limitations, we propose a novel multi-UAV collaborative occupancy prediction framework. Our framework effectively preserves 3D spatial structures and semantics through integrating a Spatial-Aware Feature Encoder and Cross-Agent Feature Integration. To enhance efficiency, we further introduce Altitude-Aware Feature Reduction to compactly represent scene information, along with a Dual-Mask Perceptual Guidance mechanism to adaptively select features and reduce communication overhead. Due to the absence of suitable benchmark datasets, we extend three datasets for evaluation: two virtual datasets (Air-to-Pred-Occ and UAV3D-Occ) and one real-world dataset (GauUScene-Occ). Experiments results demonstrate that our method achieves state-of-the-art accuracy, significantly outperforming existing collaborative methods while reducing communication overhead to only a fraction of previous approaches.
>
---
#### [new 055] Enhancing the Quality of 3D Lunar Maps Using JAXA's Kaguya Imagery
- **分类: cs.CV; cs.LG**

- **简介: 该论文旨在提升月球3D地图质量，解决Kaguya影像压缩导致的视差噪声问题。通过分析噪声模式，提出去噪方法，有效降低高程误差，提升地形数据可靠性，服务于长距离月球探测任务。**

- **链接: [http://arxiv.org/pdf/2510.11817v1](http://arxiv.org/pdf/2510.11817v1)**

> **作者:** Yumi Iwashita; Haakon Moe; Yang Cheng; Adnan Ansar; Georgios Georgakis; Adrian Stoica; Kazuto Nakashima; Ryo Kurazume; Jim Torresen
>
> **备注:** Presented at IEEE SMC 2025
>
> **摘要:** As global efforts to explore the Moon intensify, the need for high-quality 3D lunar maps becomes increasingly critical-particularly for long-distance missions such as NASA's Endurance mission concept, in which a rover aims to traverse 2,000 km across the South Pole-Aitken basin. Kaguya TC (Terrain Camera) images, though globally available at 10 m/pixel, suffer from altitude inaccuracies caused by stereo matching errors and JPEG-based compression artifacts. This paper presents a method to improve the quality of 3D maps generated from Kaguya TC images, focusing on mitigating the effects of compression-induced noise in disparity maps. We analyze the compression behavior of Kaguya TC imagery, and identify systematic disparity noise patterns, especially in darker regions. In this paper, we propose an approach to enhance 3D map quality by reducing residual noise in disparity images derived from compressed images. Our experimental results show that the proposed approach effectively reduces elevation noise, enhancing the safety and reliability of terrain data for future lunar missions.
>
---
#### [new 056] Evaluating the Explainability of Vision Transformers in Medical Imaging
- **分类: cs.CV**

- **简介: 该论文研究医学图像中视觉Transformer（ViT）的可解释性问题，比较不同ViT模型与解释方法（Grad-CAM、Gradient Attention Rollout）在细胞和超声图像分类中的表现，发现DINO结合Grad-CAM能生成更准确、局部化的解释，提升模型透明度与临床可信度。**

- **链接: [http://arxiv.org/pdf/2510.12021v1](http://arxiv.org/pdf/2510.12021v1)**

> **作者:** Leili Barekatain; Ben Glocker
>
> **备注:** Accepted at Workshop on Interpretability of Machine Intelligence in Medical Image Computing at MICCAI 2025
>
> **摘要:** Understanding model decisions is crucial in medical imaging, where interpretability directly impacts clinical trust and adoption. Vision Transformers (ViTs) have demonstrated state-of-the-art performance in diagnostic imaging; however, their complex attention mechanisms pose challenges to explainability. This study evaluates the explainability of different Vision Transformer architectures and pre-training strategies - ViT, DeiT, DINO, and Swin Transformer - using Gradient Attention Rollout and Grad-CAM. We conduct both quantitative and qualitative analyses on two medical imaging tasks: peripheral blood cell classification and breast ultrasound image classification. Our findings indicate that DINO combined with Grad-CAM offers the most faithful and localized explanations across datasets. Grad-CAM consistently produces class-discriminative and spatially precise heatmaps, while Gradient Attention Rollout yields more scattered activations. Even in misclassification cases, DINO with Grad-CAM highlights clinically relevant morphological features that appear to have misled the model. By improving model transparency, this research supports the reliable and explainable integration of ViTs into critical medical diagnostic workflows.
>
---
#### [new 057] SpineBench: Benchmarking Multimodal LLMs for Spinal Pathology Analysis
- **分类: cs.CV**

- **简介: 该论文提出SpineBench，首个面向脊柱疾病的多模态大模型评测基准，解决现有医学评测忽略脊柱视觉分析的问题。构建含6.4万问答对的脊柱VQA数据集，涵盖11种疾病诊断与病灶定位，评估12个主流MLLM，揭示其在脊柱任务中的不足。**

- **链接: [http://arxiv.org/pdf/2510.12267v1](http://arxiv.org/pdf/2510.12267v1)**

> **作者:** Chenghanyu Zhang; Zekun Li; Peipei Li; Xing Cui; Shuhan Xia; Weixiang Yan; Yiqiao Zhang; Qianyu Zhuang
>
> **备注:** Proceedings of the 33rd ACM International Conference on Multimedia,ACMMM 2025 Dataset Track
>
> **摘要:** With the increasing integration of Multimodal Large Language Models (MLLMs) into the medical field, comprehensive evaluation of their performance in various medical domains becomes critical. However, existing benchmarks primarily assess general medical tasks, inadequately capturing performance in nuanced areas like the spine, which relies heavily on visual input. To address this, we introduce SpineBench, a comprehensive Visual Question Answering (VQA) benchmark designed for fine-grained analysis and evaluation of MLLMs in the spinal domain. SpineBench comprises 64,878 QA pairs from 40,263 spine images, covering 11 spinal diseases through two critical clinical tasks: spinal disease diagnosis and spinal lesion localization, both in multiple-choice format. SpineBench is built by integrating and standardizing image-label pairs from open-source spinal disease datasets, and samples challenging hard negative options for each VQA pair based on visual similarity (similar but not the same disease), simulating real-world challenging scenarios. We evaluate 12 leading MLLMs on SpineBench. The results reveal that these models exhibit poor performance in spinal tasks, highlighting limitations of current MLLM in the spine domain and guiding future improvements in spinal medicine applications. SpineBench is publicly available at https://zhangchenghanyu.github.io/SpineBench.github.io/.
>
---
#### [new 058] PET Head Motion Estimation Using Supervised Deep Learning with Attention
- **分类: cs.CV**

- **简介: 该论文针对PET脑成像中头动导致的图像伪影问题，提出一种基于监督深度学习与交叉注意力的头动估计方法DL-HMC++，利用带标签的真实头动数据训练模型，实现从1秒3D PET数据中准确预测刚性头动，提升运动校正效果，适用于多设备与示踪剂，推动无硬件追踪的临床应用。**

- **链接: [http://arxiv.org/pdf/2510.12758v1](http://arxiv.org/pdf/2510.12758v1)**

> **作者:** Zhuotong Cai; Tianyi Zeng; Jiazhen Zhang; Eléonore V. Lieffrig; Kathryn Fontaine; Chenyu You; Enette Mae Revilla; James S. Duncan; Jingmin Xin; Yihuan Lu; John A. Onofrey
>
> **备注:** Accepted for publication in IEEE Transactions on Medical Imaging (TMI), 2025. This is the accepted manuscript version
>
> **摘要:** Head movement poses a significant challenge in brain positron emission tomography (PET) imaging, resulting in image artifacts and tracer uptake quantification inaccuracies. Effective head motion estimation and correction are crucial for precise quantitative image analysis and accurate diagnosis of neurological disorders. Hardware-based motion tracking (HMT) has limited applicability in real-world clinical practice. To overcome this limitation, we propose a deep-learning head motion correction approach with cross-attention (DL-HMC++) to predict rigid head motion from one-second 3D PET raw data. DL-HMC++ is trained in a supervised manner by leveraging existing dynamic PET scans with gold-standard motion measurements from external HMT. We evaluate DL-HMC++ on two PET scanners (HRRT and mCT) and four radiotracers (18F-FDG, 18F-FPEB, 11C-UCB-J, and 11C-LSN3172176) to demonstrate the effectiveness and generalization of the approach in large cohort PET studies. Quantitative and qualitative results demonstrate that DL-HMC++ consistently outperforms state-of-the-art data-driven motion estimation methods, producing motion-free images with clear delineation of brain structures and reduced motion artifacts that are indistinguishable from gold-standard HMT. Brain region of interest standard uptake value analysis exhibits average difference ratios between DL-HMC++ and gold-standard HMT to be 1.2 plus-minus 0.5% for HRRT and 0.5 plus-minus 0.2% for mCT. DL-HMC++ demonstrates the potential for data-driven PET head motion correction to remove the burden of HMT, making motion correction accessible to clinical populations beyond research settings. The code is available at https://github.com/maxxxxxxcai/DL-HMC-TMI.
>
---
#### [new 059] Task-Specific Dual-Model Framework for Comprehensive Traffic Safety Video Description and Analysis
- **分类: cs.CV**

- **简介: 该论文针对交通安全隐患分析任务，提出一种双模型框架，分别优化视频描述与视觉问答。利用VideoLLaMA和Qwen2.5-VL的互补优势，通过任务分离训练提升性能，在WTS数据集上取得良好效果。**

- **链接: [http://arxiv.org/pdf/2510.11907v1](http://arxiv.org/pdf/2510.11907v1)**

> **作者:** Blessing Agyei Kyem; Neema Jakisa Owor; Andrews Danyo; Joshua Kofi Asamoah; Eugene Denteh; Tanner Muturi; Anthony Dontoh; Yaw Adu-Gyamfi; Armstrong Aboah
>
> **备注:** This paper was accepted at ICCV 2025
>
> **摘要:** Traffic safety analysis requires complex video understanding to capture fine-grained behavioral patterns and generate comprehensive descriptions for accident prevention. In this work, we present a unique dual-model framework that strategically utilizes the complementary strengths of VideoLLaMA and Qwen2.5-VL through task-specific optimization to address this issue. The core insight behind our approach is that separating training for captioning and visual question answering (VQA) tasks minimizes task interference and allows each model to specialize more effectively. Experimental results demonstrate that VideoLLaMA is particularly effective in temporal reasoning, achieving a CIDEr score of 1.1001, while Qwen2.5-VL excels in visual understanding with a VQA accuracy of 60.80\%. Through extensive experiments on the WTS dataset, our method achieves an S2 score of 45.7572 in the 2025 AI City Challenge Track 2, placing 10th on the challenge leaderboard. Ablation studies validate that our separate training strategy outperforms joint training by 8.6\% in VQA accuracy while maintaining captioning quality.
>
---
#### [new 060] MetaCaptioner: Towards Generalist Visual Captioning with Open-source Suites
- **分类: cs.CV**

- **简介: 该论文研究通用视觉描述生成任务，旨在缩小开源模型与商业模型间的性能差距。作者提出多代理协作流程CapFlow，低成本生成高质量图文描述数据，并训练出高性能开源模型MetaCaptioner，在多个领域达到媲美GPT-4的描述能力。**

- **链接: [http://arxiv.org/pdf/2510.12126v1](http://arxiv.org/pdf/2510.12126v1)**

> **作者:** Zhenxin Lei; Zhangwei Gao; Changyao Tian; Erfei Cui; Guanzhou Chen; Danni Yang; Yuchen Duan; Zhaokai Wang; Wenhao Li; Weiyun Wang; Xiangyu Zhao; Jiayi Ji; Yu Qiao; Wenhai Wang; Gen Luo
>
> **摘要:** Generalist visual captioning goes beyond a simple appearance description task, but requires integrating a series of visual cues into a caption and handling various visual domains. In this task, current open-source models present a large performance gap with commercial ones, which limits various applications such as data synthesis. To bridge the gap, this paper proposes CapFlow, a novel multi-agent collaboration workflow. CapFlow demonstrates for the first time that, by capitalizing on open-source models, it is possible to achieve caption quality on par with GPT-4.1 in various domains with an 89.5% reduction in costs. By leveraging CapFlow as the data synthesizer, we produce high-quality visual captions from image and video domains at scale, and obtain a generalist visual captioner via fine-tuning, namely MetaCaptioner. Through extensive experiments, we show that MetaCaptioner not only achieves comparable captioning capabilities with commercial models but also reaches top-tier multimodal performance in the open-source community. We hope CapFlow and MetaCaptioner can benefit future multimodal research by providing a strong and cost-effective visual captioning solution.
>
---
#### [new 061] DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦自动驾驶中的视觉-语言-动作（VLA）模型，解决因动作标注稀疏导致的“监督不足”问题。提出DriveVLA-W0，引入世界模型预测未来图像，提供密集自监督信号，并结合轻量动作专家实现高效推理，显著提升性能并增强数据扩展能力。**

- **链接: [http://arxiv.org/pdf/2510.12796v1](http://arxiv.org/pdf/2510.12796v1)**

> **作者:** Yingyan Li; Shuyao Shang; Weisong Liu; Bing Zhan; Haochen Wang; Yuqi Wang; Yuntao Chen; Xiaoman Wang; Yasong An; Chufeng Tang; Lu Hou; Lue Fan; Zhaoxiang Zhang
>
> **摘要:** Scaling Vision-Language-Action (VLA) models on large-scale data offers a promising path to achieving a more generalized driving intelligence. However, VLA models are limited by a ``supervision deficit'': the vast model capacity is supervised by sparse, low-dimensional actions, leaving much of their representational power underutilized. To remedy this, we propose \textbf{DriveVLA-W0}, a training paradigm that employs world modeling to predict future images. This task generates a dense, self-supervised signal that compels the model to learn the underlying dynamics of the driving environment. We showcase the paradigm's versatility by instantiating it for two dominant VLA archetypes: an autoregressive world model for VLAs that use discrete visual tokens, and a diffusion world model for those operating on continuous visual features. Building on the rich representations learned from world modeling, we introduce a lightweight action expert to address the inference latency for real-time deployment. Extensive experiments on the NAVSIM v1/v2 benchmark and a 680x larger in-house dataset demonstrate that DriveVLA-W0 significantly outperforms BEV and VLA baselines. Crucially, it amplifies the data scaling law, showing that performance gains accelerate as the training dataset size increases.
>
---
#### [new 062] Learning Human Motion with Temporally Conditional Mamba
- **分类: cs.CV**

- **简介: 该论文研究基于时变输入信号的人体运动生成任务，旨在解决现有方法在时间对齐上的不足。作者提出时序条件Mamba（TCM），将条件信息融入Mamba模块的循环动态中，提升运动生成的时间对齐性与真实感。**

- **链接: [http://arxiv.org/pdf/2510.12573v1](http://arxiv.org/pdf/2510.12573v1)**

> **作者:** Quang Nguyen; Tri Le; Baoru Huang; Minh Nhat Vu; Ngan Le; Thieu Vo; Anh Nguyen
>
> **备注:** 10 pages
>
> **摘要:** Learning human motion based on a time-dependent input signal presents a challenging yet impactful task with various applications. The goal of this task is to generate or estimate human movement that consistently reflects the temporal patterns of conditioning inputs. Existing methods typically rely on cross-attention mechanisms to fuse the condition with motion. However, this approach primarily captures global interactions and struggles to maintain step-by-step temporal alignment. To address this limitation, we introduce Temporally Conditional Mamba, a new mamba-based model for human motion generation. Our approach integrates conditional information into the recurrent dynamics of the Mamba block, enabling better temporally aligned motion. To validate the effectiveness of our method, we evaluate it on a variety of human motion tasks. Extensive experiments demonstrate that our model significantly improves temporal alignment, motion realism, and condition consistency over state-of-the-art approaches. Our project page is available at https://zquang2202.github.io/TCM.
>
---
#### [new 063] A Review of Longitudinal Radiology Report Generation: Dataset Composition, Methods, and Performance Evaluation
- **分类: cs.CV**

- **简介: 该论文综述了纵向放射学报告生成任务，旨在解决现有模型忽视历史影像信息的问题。作者系统梳理了数据集构建、模型架构与评估方法，强调纵向信息的重要性，并指出当前研究的局限与未来方向。**

- **链接: [http://arxiv.org/pdf/2510.12444v1](http://arxiv.org/pdf/2510.12444v1)**

> **作者:** Shaoyang Zhou; Yingshu Li; Yunyi Liu; Lingqiao Liu; Lei Wang; Luping Zhou
>
> **摘要:** Chest Xray imaging is a widely used diagnostic tool in modern medicine, and its high utilization creates substantial workloads for radiologists. To alleviate this burden, vision language models are increasingly applied to automate Chest Xray radiology report generation (CXRRRG), aiming for clinically accurate descriptions while reducing manual effort. Conventional approaches, however, typically rely on single images, failing to capture the longitudinal context necessary for producing clinically faithful comparison statements. Recently, growing attention has been directed toward incorporating longitudinal data into CXR RRG, enabling models to leverage historical studies in ways that mirror radiologists diagnostic workflows. Nevertheless, existing surveys primarily address single image CXRRRG and offer limited guidance for longitudinal settings, leaving researchers without a systematic framework for model design. To address this gap, this survey provides the first comprehensive review of longitudinal radiology report generation (LRRG). Specifically, we examine dataset construction strategies, report generation architectures alongside longitudinally tailored designs, and evaluation protocols encompassing both longitudinal specific measures and widely used benchmarks. We further summarize LRRG methods performance, alongside analyses of different ablation studies, which collectively highlight the critical role of longitudinal information and architectural design choices in improving model performance. Finally, we summarize five major limitations of current research and outline promising directions for future development, aiming to lay a foundation for advancing this emerging field.
>
---
#### [new 064] Multiplicative Loss for Enhancing Semantic Segmentation in Medical and Cellular Images
- **分类: cs.CV**

- **简介: 该论文针对医学和细胞图像语义分割中数据稀缺导致训练不稳定的问题，提出两种新损失函数：乘性损失及其置信度自适应变体，通过动态调节梯度，提升小样本下的分割性能。**

- **链接: [http://arxiv.org/pdf/2510.12258v1](http://arxiv.org/pdf/2510.12258v1)**

> **作者:** Yuto Yokoi; Kazuhiro Hotta
>
> **备注:** Accepted by ICCV2025 Workshop "Third Workshop on Computer Vision for Automated Medical Diagnosis"
>
> **摘要:** We propose two novel loss functions, Multiplicative Loss and Confidence-Adaptive Multiplicative Loss, for semantic segmentation in medical and cellular images. Although Cross Entropy and Dice Loss are widely used, their additive combination is sensitive to hyperparameters and often performs suboptimally, especially with limited data. Medical images suffer from data scarcity due to privacy, ethics, and costly annotations, requiring robust and efficient training objectives. Our Multiplicative Loss combines Cross Entropy and Dice losses multiplicatively, dynamically modulating gradients based on prediction confidence. This reduces penalties for confident correct predictions and amplifies gradients for incorrect overconfident ones, stabilizing optimization. Building on this, Confidence-Adaptive Multiplicative Loss applies a confidence-driven exponential scaling inspired by Focal Loss, integrating predicted probabilities and Dice coefficients to emphasize difficult samples. This enhances learning under extreme data scarcity by strengthening gradients when confidence is low. Experiments on cellular and medical segmentation benchmarks show our framework consistently outperforms tuned additive and existing loss functions, offering a simple, effective, and hyperparameter-free mechanism for robust segmentation under challenging data limitations.
>
---
#### [new 065] Hybrid Explanation-Guided Learning for Transformer-Based Chest X-Ray Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究胸部X光诊断任务，旨在解决Transformer模型易学偏见、泛化差的问题。提出混合解释引导学习（H-EGL）框架，结合自监督与人工引导约束，提升注意力对齐和模型性能。**

- **链接: [http://arxiv.org/pdf/2510.12704v1](http://arxiv.org/pdf/2510.12704v1)**

> **作者:** Shelley Zixin Shu; Haozhe Luo; Alexander Poellinger; Mauricio Reyes
>
> **备注:** Accepted by iMIMIC at MICCAI 2025
>
> **摘要:** Transformer-based deep learning models have demonstrated exceptional performance in medical imaging by leveraging attention mechanisms for feature representation and interpretability. However, these models are prone to learning spurious correlations, leading to biases and limited generalization. While human-AI attention alignment can mitigate these issues, it often depends on costly manual supervision. In this work, we propose a Hybrid Explanation-Guided Learning (H-EGL) framework that combines self-supervised and human-guided constraints to enhance attention alignment and improve generalization. The self-supervised component of H-EGL leverages class-distinctive attention without relying on restrictive priors, promoting robustness and flexibility. We validate our approach on chest X-ray classification using the Vision Transformer (ViT), where H-EGL outperforms two state-of-the-art Explanation-Guided Learning (EGL) methods, demonstrating superior classification accuracy and generalization capability. Additionally, it produces attention maps that are better aligned with human expertise.
>
---
#### [new 066] Zero-Shot CFC: Fast Real-World Image Denoising based on Cross-Frequency Consistency
- **分类: cs.CV**

- **简介: 该论文研究真实图像去噪任务，解决现有零样本方法训练慢、依赖噪声假设的问题。提出基于跨频一致性（ZSCFC）的轻量方法，利用纹理在频域的一致性设计损失函数，实现高效单图去噪。**

- **链接: [http://arxiv.org/pdf/2510.12646v1](http://arxiv.org/pdf/2510.12646v1)**

> **作者:** Yanlin Jiang; Yuchen Liu; Mingren Liu
>
> **备注:** The British Machine Vision Conference
>
> **摘要:** Zero-shot denoisers address the dataset dependency of deep-learning-based denoisers, enabling the denoising of unseen single images. Nonetheless, existing zero-shot methods suffer from long training times and rely on the assumption of noise independence and a zero-mean property, limiting their effectiveness in real-world denoising scenarios where noise characteristics are more complicated. This paper proposes an efficient and effective method for real-world denoising, the Zero-Shot denoiser based on Cross-Frequency Consistency (ZSCFC), which enables training and denoising with a single noisy image and does not rely on assumptions about noise distribution. Specifically, image textures exhibit position similarity and content consistency across different frequency bands, while noise does not. Based on this property, we developed cross-frequency consistency loss and an ultralight network to realize image denoising. Experiments on various real-world image datasets demonstrate that our ZSCFC outperforms other state-of-the-art zero-shot methods in terms of computational efficiency and denoising performance.
>
---
#### [new 067] State Space Prompting via Gathering and Spreading Spatio-Temporal Information for Video Understanding
- **分类: cs.CV**

- **简介: 该论文研究视频理解任务，旨在解决预训练状态空间模型中视觉提示无法有效捕获时空上下文信息的问题。作者提出状态空间提示（SSP）方法，通过帧内聚合与帧间传播模块，增强关键时空信息的传递，提升性能并减少微调参数。**

- **链接: [http://arxiv.org/pdf/2510.12160v1](http://arxiv.org/pdf/2510.12160v1)**

> **作者:** Jiahuan Zhou; Kai Zhu; Zhenyu Cui; Zichen Liu; Xu Zou; Gang Hua
>
> **摘要:** Recently, pre-trained state space models have shown great potential for video classification, which sequentially compresses visual tokens in videos with linear complexity, thereby improving the processing efficiency of video data while maintaining high performance. To apply powerful pre-trained models to downstream tasks, prompt learning is proposed to achieve efficient downstream task adaptation with only a small number of fine-tuned parameters. However, the sequentially compressed visual prompt tokens fail to capture the spatial and temporal contextual information in the video, thus limiting the effective propagation of spatial information within a video frame and temporal information between frames in the state compression model and the extraction of discriminative information. To tackle the above issue, we proposed a State Space Prompting (SSP) method for video understanding, which combines intra-frame and inter-frame prompts to aggregate and propagate key spatiotemporal information in the video. Specifically, an Intra-Frame Gathering (IFG) module is designed to aggregate spatial key information within each frame. Besides, an Inter-Frame Spreading (IFS) module is designed to spread discriminative spatio-temporal information across different frames. By adaptively balancing and compressing key spatio-temporal information within and between frames, our SSP effectively propagates discriminative information in videos in a complementary manner. Extensive experiments on four video benchmark datasets verify that our SSP significantly outperforms existing SOTA methods by 2.76% on average while reducing the overhead of fine-tuning parameters.
>
---
#### [new 068] Personalized Federated Fine-Tuning of Vision Foundation Models for Healthcare
- **分类: cs.CV; cs.DC**

- **简介: 该论文研究医疗视觉基础模型的个性化联邦微调。针对数据隐私导致的跨机构数据共享难题，提出一种基于正交LoRA适配器的方法，解耦通用与客户端特有知识，在保护隐私的同时提升各客户端模型性能。**

- **链接: [http://arxiv.org/pdf/2510.12741v1](http://arxiv.org/pdf/2510.12741v1)**

> **作者:** Adam Tupper; Christian Gagné
>
> **备注:** Accepted to the Symposium on Model Accountability, Sustainability and Healthcare (SMASH) 2025
>
> **摘要:** Foundation models open up new possibilities for the use of AI in healthcare. However, even when pre-trained on health data, they still need to be fine-tuned for specific downstream tasks. Furthermore, although foundation models reduce the amount of training data required to achieve good performance, obtaining sufficient data is still a challenge. This is due, in part, to restrictions on sharing and aggregating data from different sources to protect patients' privacy. One possible solution to this is to fine-tune foundation models via federated learning across multiple participating clients (i.e., hospitals, clinics, etc.). In this work, we propose a new personalized federated fine-tuning method that learns orthogonal LoRA adapters to disentangle general and client-specific knowledge, enabling each client to fully exploit both their own data and the data of others. Our preliminary results on real-world federated medical imaging tasks demonstrate that our approach is competitive against current federated fine-tuning methods.
>
---
#### [new 069] Dual Learning with Dynamic Knowledge Distillation and Soft Alignment for Partially Relevant Video Retrieval
- **分类: cs.CV**

- **简介: 该论文研究部分相关视频检索（PRVR）任务，解决真实场景中长视频与文本仅部分匹配的问题。提出双学习框架DL-DKD++，通过动态知识蒸馏和软对齐机制，利用大模型指导轻量学生网络，提升细粒度跨模态对齐能力。**

- **链接: [http://arxiv.org/pdf/2510.12283v1](http://arxiv.org/pdf/2510.12283v1)**

> **作者:** Jianfeng Dong; Lei Huang; Daizong Liu; Xianke Chen; Xun Yang; Changting Lin; Xun Wang; Meng Wang
>
> **摘要:** Almost all previous text-to-video retrieval works ideally assume that videos are pre-trimmed with short durations containing solely text-related content. However, in practice, videos are typically untrimmed in long durations with much more complicated background content. Therefore, in this paper, we focus on the more practical yet challenging task of Partially Relevant Video Retrieval (PRVR), which aims to retrieve partially relevant untrimmed videos with the given query. To tackle this task, we propose a novel framework that distills generalization knowledge from a powerful large-scale vision-language pre-trained model and transfers it to a lightweight, task-specific PRVR network. Specifically, we introduce a Dual Learning framework with Dynamic Knowledge Distillation (DL-DKD++), where a large teacher model provides supervision to a compact dual-branch student network. The student model comprises two branches: an inheritance branch that absorbs transferable knowledge from the teacher, and an exploration branch that learns task-specific information from the PRVR dataset to address domain gaps. To further enhance learning, we incorporate a dynamic soft-target construction mechanism. By replacing rigid hard-target supervision with adaptive soft targets that evolve during training, our method enables the model to better capture the fine-grained, partial relevance between videos and queries. Experiment results demonstrate that our proposed model achieves state-of-the-art performance on TVR, ActivityNet, and Charades-STA datasets for PRVR. The code is available at https://github.com/HuiGuanLab/DL-DKD.
>
---
#### [new 070] VIDMP3: Video Editing by Representing Motion with Pose and Position Priors
- **分类: cs.CV**

- **简介: 该论文研究运动保持的视频编辑任务，旨在解决结构可变编辑中的时序不一致、身份漂移等问题。提出VidMP3方法，利用姿态和位置先验学习通用运动表示，实现语义与结构灵活编辑的同时保留原始运动。**

- **链接: [http://arxiv.org/pdf/2510.12069v1](http://arxiv.org/pdf/2510.12069v1)**

> **作者:** Sandeep Mishra; Oindrila Saha; Alan C. Bovik
>
> **摘要:** Motion-preserved video editing is crucial for creators, particularly in scenarios that demand flexibility in both the structure and semantics of swapped objects. Despite its potential, this area remains underexplored. Existing diffusion-based editing methods excel in structure-preserving tasks, using dense guidance signals to ensure content integrity. While some recent methods attempt to address structure-variable editing, they often suffer from issues such as temporal inconsistency, subject identity drift, and the need for human intervention. To address these challenges, we introduce VidMP3, a novel approach that leverages pose and position priors to learn a generalized motion representation from source videos. Our method enables the generation of new videos that maintain the original motion while allowing for structural and semantic flexibility. Both qualitative and quantitative evaluations demonstrate the superiority of our approach over existing methods. The code will be made publicly available at https://github.com/sandeep-sm/VidMP3.
>
---
#### [new 071] FedHUG: Federated Heterogeneous Unsupervised Generalization for Remote Physiological Measurements
- **分类: cs.CV**

- **简介: 该论文研究远程生理测量中的无监督联邦学习，解决多域异构数据下标签缺失与分布偏移问题。提出FedHUG框架，包含动态聚合与分布感知控制模块，实现无需标签的跨域泛化。**

- **链接: [http://arxiv.org/pdf/2510.12132v1](http://arxiv.org/pdf/2510.12132v1)**

> **作者:** Xiao Yang; Jiyao Wang
>
> **摘要:** Remote physiological measurement gained wide attention, while it requires collecting users' privacy-sensitive information, and existing contactless measurements still rely on labeled client data. This presents challenges when we want to further update real-world deployed models with numerous user data lacking labels. To resolve these challenges, we instantiate a new protocol called Federated Unsupervised Domain Generalization (FUDG) in this work. Subsequently, the \textbf{Fed}erated \textbf{H}eterogeneous \textbf{U}nsupervised \textbf{G}eneralization (\textbf{FedHUG}) framework is proposed and consists of: (1) Minimal Bias Aggregation module dynamically adjusts aggregation weights based on prior-driven bias evaluation to cope with heterogeneous non-IID features from multiple domains. (2) The Global Distribution-aware Learning Controller parameterizes the label distribution and dynamically manipulates client-specific training strategies, thereby mitigating the server-client label distribution skew and long-tail issue. The proposal shows superior performance across state-of-the-art techniques in estimation with either RGB video or mmWave radar. The code will be released.
>
---
#### [new 072] On the Use of Hierarchical Vision Foundation Models for Low-Cost Human Mesh Recovery and Pose Estimation
- **分类: cs.CV**

- **简介: 该论文研究人体网格恢复与姿态估计，旨在提升轻量模型的效率与精度。作者基于层次化视觉基础模型的早期阶段构建轻量编码器，提出高效HMR方法，在降低计算成本的同时保持优异性能。**

- **链接: [http://arxiv.org/pdf/2510.12660v1](http://arxiv.org/pdf/2510.12660v1)**

> **作者:** Shuhei Tarashima; Yushan Wang; Norio Tagawa
>
> **备注:** Accepted at ICCVW 2025
>
> **摘要:** In this work, we aim to develop simple and efficient models for human mesh recovery (HMR) and its predecessor task, human pose estimation (HPE). State-of-the-art HMR methods, such as HMR2.0 and its successors, rely on large, non-hierarchical vision transformers as encoders, which are inherited from the corresponding HPE models like ViTPose. To establish baselines across varying computational budgets, we first construct three lightweight HMR2.0 variants by adapting the corresponding ViTPose models. In addition, we propose leveraging the early stages of hierarchical vision foundation models (VFMs), including Swin Transformer, GroupMixFormer, and VMamba, as encoders. This design is motivated by the observation that intermediate stages of hierarchical VFMs produce feature maps with resolutions comparable to or higher than those of non-hierarchical counterparts. We conduct a comprehensive evaluation of 27 hierarchical-VFM-based HMR and HPE models, demonstrating that using only the first two or three stages achieves performance on par with full-stage models. Moreover, we show that the resulting truncated models exhibit better trade-offs between accuracy and computational efficiency compared to existing lightweight alternatives.
>
---
#### [new 073] CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations
- **分类: cs.CV; cs.AI; cs.LG; math.AT; stat.ML**

- **简介: 该论文提出CuMPerLay，一种可微分的立方体多参数持久性向量化层，旨在将拓扑特征融入深度学习。针对多滤波结构复杂和难向量化问题，设计可学习单参数分解方法，提升图像分类与分割性能，尤其适用于小样本场景。**

- **链接: [http://arxiv.org/pdf/2510.12795v1](http://arxiv.org/pdf/2510.12795v1)**

> **作者:** Caner Korkmaz; Brighton Nuwagira; Barış Coşkunuzer; Tolga Birdal
>
> **备注:** Appears at ICCV 2025
>
> **摘要:** We present CuMPerLay, a novel differentiable vectorization layer that enables the integration of Cubical Multiparameter Persistence (CMP) into deep learning pipelines. While CMP presents a natural and powerful way to topologically work with images, its use is hindered by the complexity of multifiltration structures as well as the vectorization of CMP. In face of these challenges, we introduce a new algorithm for vectorizing MP homologies of cubical complexes. Our CuMPerLay decomposes the CMP into a combination of individual, learnable single-parameter persistence, where the bifiltration functions are jointly learned. Thanks to the differentiability, its robust topological feature vectors can be seamlessly used within state-of-the-art architectures such as Swin Transformers. We establish theoretical guarantees for the stability of our vectorization under generalized Wasserstein metrics. Our experiments on benchmark medical imaging and computer vision datasets show the benefit CuMPerLay on classification and segmentation performance, particularly in limited-data scenarios. Overall, CuMPerLay offers a promising direction for integrating global structural information into deep networks for structured image analysis.
>
---
#### [new 074] MMOT: The First Challenging Benchmark for Drone-based Multispectral Multi-Object Tracking
- **分类: cs.CV**

- **简介: 该论文针对无人机多目标跟踪中RGB模态性能受限的问题，提出首个面向无人机的多光谱多目标跟踪基准MMOT，包含大规模、复杂场景和精确方向标注，并设计了融合光谱特征与方向感知的跟踪方法，显著提升小目标和密集场景下的跟踪性能。**

- **链接: [http://arxiv.org/pdf/2510.12565v1](http://arxiv.org/pdf/2510.12565v1)**

> **作者:** Tianhao Li; Tingfa Xu; Ying Wang; Haolin Qin; Xu Lin; Jianan Li
>
> **摘要:** Drone-based multi-object tracking is essential yet highly challenging due to small targets, severe occlusions, and cluttered backgrounds. Existing RGB-based tracking algorithms heavily depend on spatial appearance cues such as color and texture, which often degrade in aerial views, compromising reliability. Multispectral imagery, capturing pixel-level spectral reflectance, provides crucial cues that enhance object discriminability under degraded spatial conditions. However, the lack of dedicated multispectral UAV datasets has hindered progress in this domain. To bridge this gap, we introduce MMOT, the first challenging benchmark for drone-based multispectral multi-object tracking. It features three key characteristics: (i) Large Scale - 125 video sequences with over 488.8K annotations across eight categories; (ii) Comprehensive Challenges - covering diverse conditions such as extreme small targets, high-density scenarios, severe occlusions, and complex motion; and (iii) Precise Oriented Annotations - enabling accurate localization and reduced ambiguity under aerial perspectives. To better extract spectral features and leverage oriented annotations, we further present a multispectral and orientation-aware MOT scheme adapting existing methods, featuring: (i) a lightweight Spectral 3D-Stem integrating spectral features while preserving compatibility with RGB pretraining; (ii) an orientation-aware Kalman filter for precise state estimation; and (iii) an end-to-end orientation-adaptive transformer. Extensive experiments across representative trackers consistently show that multispectral input markedly improves tracking performance over RGB baselines, particularly for small and densely packed objects. We believe our work will advance drone-based multispectral multi-object tracking research. Our MMOT, code, and benchmarks are publicly available at https://github.com/Annzstbl/MMOT.
>
---
#### [new 075] APGNet: Adaptive Prior-Guided for Underwater Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 该论文研究水下伪装物体检测任务，旨在解决水下图像退化与生物伪装导致的检测难题。提出APGNet网络，结合多尺度增强、扩展感受野模块及自适应先验引导机制，提升特征表达与定位精度，在两个公开数据集上优于15种现有方法。**

- **链接: [http://arxiv.org/pdf/2510.12056v1](http://arxiv.org/pdf/2510.12056v1)**

> **作者:** Xinxin Huang; Han Sun; Junmin Cai; Ningzhong Liu; Huiyu Zhou
>
> **备注:** 6 pages. accepted by ACM MM Asia 2025
>
> **摘要:** Detecting camouflaged objects in underwater environments is crucial for marine ecological research and resource exploration. However, existing methods face two key challenges: underwater image degradation, including low contrast and color distortion, and the natural camouflage of marine organisms. Traditional image enhancement techniques struggle to restore critical features in degraded images, while camouflaged object detection (COD) methods developed for terrestrial scenes often fail to adapt to underwater environments due to the lack of consideration for underwater optical characteristics. To address these issues, we propose APGNet, an Adaptive Prior-Guided Network, which integrates a Siamese architecture with a novel prior-guided mechanism to enhance robustness and detection accuracy. First, we employ the Multi-Scale Retinex with Color Restoration (MSRCR) algorithm for data augmentation, generating illumination-invariant images to mitigate degradation effects. Second, we design an Extended Receptive Field (ERF) module combined with a Multi-Scale Progressive Decoder (MPD) to capture multi-scale contextual information and refine feature representations. Furthermore, we propose an adaptive prior-guided mechanism that hierarchically fuses position and boundary priors by embedding spatial attention in high-level features for coarse localization and using deformable convolution to refine contours in low-level features. Extensive experimental results on two public MAS datasets demonstrate that our proposed method APGNet outperforms 15 state-of-art methods under widely used evaluation metrics.
>
---
#### [new 076] CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文研究自动驾驶决策控制任务，旨在解决纯模仿学习泛化性差、强化学习样本效率低的问题。提出CoIRL-AD框架，通过隐空间中双策略竞争机制，实现模仿与强化学习的协同训练，提升安全性与长尾场景表现。**

- **链接: [http://arxiv.org/pdf/2510.12560v1](http://arxiv.org/pdf/2510.12560v1)**

> **作者:** Xiaoji Zheng; Ziyuan Yang; Yanhao Chen; Yuhang Peng; Yuanrong Tang; Gengyuan Liu; Bokui Chen; Jiangtao Gong
>
> **备注:** 18 pages, 17 figures
>
> **摘要:** End-to-end autonomous driving models trained solely with imitation learning (IL) often suffer from poor generalization. In contrast, reinforcement learning (RL) promotes exploration through reward maximization but faces challenges such as sample inefficiency and unstable convergence. A natural solution is to combine IL and RL. Moving beyond the conventional two-stage paradigm (IL pretraining followed by RL fine-tuning), we propose CoIRL-AD, a competitive dual-policy framework that enables IL and RL agents to interact during training. CoIRL-AD introduces a competition-based mechanism that facilitates knowledge exchange while preventing gradient conflicts. Experiments on the nuScenes dataset show an 18% reduction in collision rate compared to baselines, along with stronger generalization and improved performance on long-tail scenarios. Code is available at: https://github.com/SEU-zxj/CoIRL-AD.
>
---
#### [new 077] MVP4D: Multi-View Portrait Video Diffusion for Animatable 4D Avatars
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文聚焦于生成可动画的4D数字人像，旨在解决单图生成 avatar 在多视角下失真问题。提出 MVP4D 方法，基于扩散模型生成多视角一致的动态人像视频，并蒸馏为可实时渲染的4D avatar，提升 realism 与时空一致性。**

- **链接: [http://arxiv.org/pdf/2510.12785v1](http://arxiv.org/pdf/2510.12785v1)**

> **作者:** Felix Taubner; Ruihang Zhang; Mathieu Tuli; Sherwin Bahmani; David B. Lindell
>
> **备注:** 18 pages, 12 figures
>
> **摘要:** Digital human avatars aim to simulate the dynamic appearance of humans in virtual environments, enabling immersive experiences across gaming, film, virtual reality, and more. However, the conventional process for creating and animating photorealistic human avatars is expensive and time-consuming, requiring large camera capture rigs and significant manual effort from professional 3D artists. With the advent of capable image and video generation models, recent methods enable automatic rendering of realistic animated avatars from a single casually captured reference image of a target subject. While these techniques significantly lower barriers to avatar creation and offer compelling realism, they lack constraints provided by multi-view information or an explicit 3D representation. So, image quality and realism degrade when rendered from viewpoints that deviate strongly from the reference image. Here, we build a video model that generates animatable multi-view videos of digital humans based on a single reference image and target expressions. Our model, MVP4D, is based on a state-of-the-art pre-trained video diffusion model and generates hundreds of frames simultaneously from viewpoints varying by up to 360 degrees around a target subject. We show how to distill the outputs of this model into a 4D avatar that can be rendered in real-time. Our approach significantly improves the realism, temporal consistency, and 3D consistency of generated avatars compared to previous methods.
>
---
#### [new 078] WaterFlow: Explicit Physics-Prior Rectified Flow for Underwater Saliency Mask Generation
- **分类: cs.CV**

- **简介: 该论文针对水下显著性目标检测（USOD）中图像退化与域差异问题，提出WaterFlow框架。其将水下成像物理先验显式引入网络训练，并结合时间维度建模，提升检测性能，在USOD10K上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.12605v1](http://arxiv.org/pdf/2510.12605v1)**

> **作者:** Runting Li; Shijie Lian; Hua Li; Yutong Li; Wenhui Wu; Sam Kwong
>
> **摘要:** Underwater Salient Object Detection (USOD) faces significant challenges, including underwater image quality degradation and domain gaps. Existing methods tend to ignore the physical principles of underwater imaging or simply treat degradation phenomena in underwater images as interference factors that must be eliminated, failing to fully exploit the valuable information they contain. We propose WaterFlow, a rectified flow-based framework for underwater salient object detection that innovatively incorporates underwater physical imaging information as explicit priors directly into the network training process and introduces temporal dimension modeling, significantly enhancing the model's capability for salient object identification. On the USOD10K dataset, WaterFlow achieves a 0.072 gain in S_m, demonstrating the effectiveness and superiority of our method. The code will be published after the acceptance.
>
---
#### [new 079] PanoTPS-Net: Panoramic Room Layout Estimation via Thin Plate Spline Transformation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究单张全景图的3D房间布局估计任务，旨在解决 cuboid 与非 cuboid 房间的通用布局预测问题。提出 PanoTPS-Net 模型，结合 CNN 与薄板样条（TPS）变换，通过两阶段网络实现布局参数学习与参考布局形变，提升了精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.11992v1](http://arxiv.org/pdf/2510.11992v1)**

> **作者:** Hatem Ibrahem; Ahmed Salem; Qinmin Vivian Hu; Guanghui Wang
>
> **摘要:** Accurately estimating the 3D layout of rooms is a crucial task in computer vision, with potential applications in robotics, augmented reality, and interior design. This paper proposes a novel model, PanoTPS-Net, to estimate room layout from a single panorama image. Leveraging a Convolutional Neural Network (CNN) and incorporating a Thin Plate Spline (TPS) spatial transformation, the architecture of PanoTPS-Net is divided into two stages: First, a convolutional neural network extracts the high-level features from the input images, allowing the network to learn the spatial parameters of the TPS transformation. Second, the TPS spatial transformation layer is generated to warp a reference layout to the required layout based on the predicted parameters. This unique combination empowers the model to properly predict room layouts while also generalizing effectively to both cuboid and non-cuboid layouts. Extensive experiments on publicly available datasets and comparisons with state-of-the-art methods demonstrate the effectiveness of the proposed method. The results underscore the model's accuracy in room layout estimation and emphasize the compatibility between the TPS transformation and panorama images. The robustness of the model in handling both cuboid and non-cuboid room layout estimation is evident with a 3DIoU value of 85.49, 86.16, 81.76, and 91.98 on PanoContext, Stanford-2D3D, Matterport3DLayout, and ZInD datasets, respectively. The source code is available at: https://github.com/HatemHosam/PanoTPS_Net.
>
---
#### [new 080] AnyUp: Universal Feature Upsampling
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出AnyUp，解决现有特征上采样方法无法泛化到不同特征提取器的问题。它设计了一种无需训练、适用于任意视觉特征和分辨率的通用上采样架构，在保持语义的同时提升质量，可广泛用于下游任务。**

- **链接: [http://arxiv.org/pdf/2510.12764v1](http://arxiv.org/pdf/2510.12764v1)**

> **作者:** Thomas Wimmer; Prune Truong; Marie-Julie Rakotosaona; Michael Oechsle; Federico Tombari; Bernt Schiele; Jan Eric Lenssen
>
> **备注:** Project Website: https://wimmerth.github.io/anyup/
>
> **摘要:** We introduce AnyUp, a method for feature upsampling that can be applied to any vision feature at any resolution, without encoder-specific training. Existing learning-based upsamplers for features like DINO or CLIP need to be re-trained for every feature extractor and thus do not generalize to different feature types at inference time. In this work, we propose an inference-time feature-agnostic upsampling architecture to alleviate this limitation and improve upsampling quality. In our experiments, AnyUp sets a new state of the art for upsampled features, generalizes to different feature types, and preserves feature semantics while being efficient and easy to apply to a wide range of downstream tasks.
>
---
#### [new 081] Hardware-aware Coding Function Design for Compressive Single-Photon 3D Cameras
- **分类: cs.CV**

- **简介: 该论文针对单光子3D成像中硬件限制导致压缩直方图性能下降的问题，提出一种硬件感知的编码函数设计方法。通过联合优化光照与编码矩阵，在满足带宽、峰值功率等约束下提升成像性能，尤其在峰值功率受限时表现更优。**

- **链接: [http://arxiv.org/pdf/2510.12123v1](http://arxiv.org/pdf/2510.12123v1)**

> **作者:** David Parra; Felipe Gutierrez-Barragan; Trevor Seets; Andreas Velten
>
> **备注:** IEEE TPAMI Special Issue
>
> **摘要:** Single-photon cameras are becoming increasingly popular in time-of-flight 3D imaging because they can time-tag individual photons with extreme resolution. However, their performance is susceptible to hardware limitations, such as system bandwidth, maximum laser power, sensor data rates, and in-sensor memory and compute resources. Compressive histograms were recently introduced as a solution to the challenge of data rates through an online in-sensor compression of photon timestamp data. Although compressive histograms work within limited in-sensor memory and computational resources, they underperform when subjected to real-world illumination hardware constraints. To address this, we present a constrained optimization approach for designing practical coding functions for compressive single-photon 3D imaging. Using gradient descent, we jointly optimize an illumination and coding matrix (i.e., the coding functions) that adheres to hardware constraints. We show through extensive simulations that our coding functions consistently outperform traditional coding designs under both bandwidth and peak power constraints. This advantage is particularly pronounced in systems constrained by peak power. Finally, we show that our approach adapts to arbitrary parameterized impulse responses by evaluating it on a real-world system with a non-ideal impulse response function.
>
---
#### [new 082] Playmate2: Training-Free Multi-Character Audio-Driven Animation via Diffusion Transformer with Reward Feedback
- **分类: cs.CV**

- **简介: 该论文研究音频驱动的多人物说话视频生成，旨在解决长视频时序连贯性、唇同步精度及多角色协同动画问题。提出基于扩散Transformer的框架，结合LoRA训练、奖励反馈与无需训练的Mask-CFG方法，实现高质量、无需专门训练的多角色动画生成。**

- **链接: [http://arxiv.org/pdf/2510.12089v1](http://arxiv.org/pdf/2510.12089v1)**

> **作者:** Xingpei Ma; Shenneng Huang; Jiaran Cai; Yuansheng Guan; Shen Zheng; Hanfeng Zhao; Qiang Zhang; Shunsi Zhang
>
> **摘要:** Recent advances in diffusion models have significantly improved audio-driven human video generation, surpassing traditional methods in both quality and controllability. However, existing approaches still face challenges in lip-sync accuracy, temporal coherence for long video generation, and multi-character animation. In this work, we propose a diffusion transformer (DiT)-based framework for generating lifelike talking videos of arbitrary length, and introduce a training-free method for multi-character audio-driven animation. First, we employ a LoRA-based training strategy combined with a position shift inference approach, which enables efficient long video generation while preserving the capabilities of the foundation model. Moreover, we combine partial parameter updates with reward feedback to enhance both lip synchronization and natural body motion. Finally, we propose a training-free approach, Mask Classifier-Free Guidance (Mask-CFG), for multi-character animation, which requires no specialized datasets or model modifications and supports audio-driven animation for three or more characters. Experimental results demonstrate that our method outperforms existing state-of-the-art approaches, achieving high-quality, temporally coherent, and multi-character audio-driven video generation in a simple, efficient, and cost-effective manner.
>
---
#### [new 083] Scene Coordinate Reconstruction Priors
- **分类: cs.CV**

- **简介: 该论文研究场景坐标回归（SCR）在3D视觉中的应用，旨在解决因多视图约束不足导致的模型退化问题。作者提出引入深度分布和点云扩散等重建先验，提升场景表示的几何合理性，从而改善位姿估计与新视角合成等任务性能。**

- **链接: [http://arxiv.org/pdf/2510.12387v1](http://arxiv.org/pdf/2510.12387v1)**

> **作者:** Wenjing Bian; Axel Barroso-Laguna; Tommaso Cavallari; Victor Adrian Prisacariu; Eric Brachmann
>
> **备注:** ICCV 2025, Project page: https://nianticspatial.github.io/scr-priors/
>
> **摘要:** Scene coordinate regression (SCR) models have proven to be powerful implicit scene representations for 3D vision, enabling visual relocalization and structure-from-motion. SCR models are trained specifically for one scene. If training images imply insufficient multi-view constraints SCR models degenerate. We present a probabilistic reinterpretation of training SCR models, which allows us to infuse high-level reconstruction priors. We investigate multiple such priors, ranging from simple priors over the distribution of reconstructed depth values to learned priors over plausible scene coordinate configurations. For the latter, we train a 3D point cloud diffusion model on a large corpus of indoor scans. Our priors push predicted 3D scene points towards plausible geometry at each training step to increase their likelihood. On three indoor datasets our priors help learning better scene representations, resulting in more coherent scene point clouds, higher registration rates and better camera poses, with a positive effect on down-stream tasks such as novel view synthesis and camera relocalization.
>
---
#### [new 084] Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文针对单目4D动态场景重建中因遮挡和视角不足导致的运动漂移与合成质量下降问题，提出一种不确定性感知的动态高斯点阵方法（USplat4D），通过建模时变高斯不确定性并构建时空图优化，提升重建稳定性与新视角合成效果。**

- **链接: [http://arxiv.org/pdf/2510.12768v1](http://arxiv.org/pdf/2510.12768v1)**

> **作者:** Fengzhi Guo; Chih-Chuan Hsu; Sihao Ding; Cheng Zhang
>
> **备注:** Project page: https://tamu-visual-ai.github.io/usplat4d/
>
> **摘要:** Reconstructing dynamic 3D scenes from monocular input is fundamentally under-constrained, with ambiguities arising from occlusion and extreme novel views. While dynamic Gaussian Splatting offers an efficient representation, vanilla models optimize all Gaussian primitives uniformly, ignoring whether they are well or poorly observed. This limitation leads to motion drifts under occlusion and degraded synthesis when extrapolating to unseen views. We argue that uncertainty matters: Gaussians with recurring observations across views and time act as reliable anchors to guide motion, whereas those with limited visibility are treated as less reliable. To this end, we introduce USplat4D, a novel Uncertainty-aware dynamic Gaussian Splatting framework that propagates reliable motion cues to enhance 4D reconstruction. Our key insight is to estimate time-varying per-Gaussian uncertainty and leverages it to construct a spatio-temporal graph for uncertainty-aware optimization. Experiments on diverse real and synthetic datasets show that explicitly modeling uncertainty consistently improves dynamic Gaussian Splatting models, yielding more stable geometry under occlusion and high-quality synthesis at extreme viewpoints.
>
---
#### [new 085] IL3D: A Large-Scale Indoor Layout Dataset for LLM-Driven 3D Scene Generation
- **分类: cs.CV**

- **简介: 该论文提出IL3D，一个面向大语言模型驱动的3D场景生成的大规模室内布局数据集。旨在解决高质量、多样化训练数据缺乏的问题，支持多模态学习与室内场景理解，推动具身智能研究。**

- **链接: [http://arxiv.org/pdf/2510.12095v1](http://arxiv.org/pdf/2510.12095v1)**

> **作者:** Wenxu Zhou; Kaixuan Nie; Hang Du; Dong Yin; Wei Huang; Siqiang Guo; Xiaobo Zhang; Pengbo Hu
>
> **备注:** 9 pages main paper; 15 pages references and appendix
>
> **摘要:** In this study, we present IL3D, a large-scale dataset meticulously designed for large language model (LLM)-driven 3D scene generation, addressing the pressing demand for diverse, high-quality training data in indoor layout design. Comprising 27,816 indoor layouts across 18 prevalent room types and a library of 29,215 high-fidelity 3D object assets, IL3D is enriched with instance-level natural language annotations to support robust multimodal learning for vision-language tasks. We establish rigorous benchmarks to evaluate LLM-driven scene generation. Experimental results show that supervised fine-tuning (SFT) of LLMs on IL3D significantly improves generalization and surpasses the performance of SFT on other datasets. IL3D offers flexible multimodal data export capabilities, including point clouds, 3D bounding boxes, multiview images, depth maps, normal maps, and semantic masks, enabling seamless adaptation to various visual tasks. As a versatile and robust resource, IL3D significantly advances research in 3D scene generation and embodied intelligence, by providing high-fidelity scene data to support environment perception tasks of embodied agents.
>
---
#### [new 086] An Adaptive Edge-Guided Dual-Network Framework for Fast QR Code Motion Deblurring
- **分类: cs.CV**

- **简介: 该论文研究QR码运动去模糊任务，旨在提升解码成功率。针对不同程度模糊，提出边缘引导的双网络框架：严重模糊用EG-Restormer，轻微模糊用轻量LENet，并自适应切换，兼顾性能与效率。**

- **链接: [http://arxiv.org/pdf/2510.12098v1](http://arxiv.org/pdf/2510.12098v1)**

> **作者:** Jianping Li; Dongyang Guo; Wenjie Li; Wei Zhao
>
> **摘要:** Unlike general image deblurring that prioritizes perceptual quality, QR code deblurring focuses on ensuring successful decoding. QR codes are characterized by highly structured patterns with sharp edges, a robust prior for restoration. Yet existing deep learning methods rarely exploit these priors explicitly. To address this gap, we propose the Edge-Guided Attention Block (EGAB), which embeds explicit edge priors into a Transformer architecture. Based on EGAB, we develop Edge-Guided Restormer (EG-Restormer), an effective network that significantly boosts the decoding rate of severely blurred QR codes. For mildly blurred inputs, we design the Lightweight and Efficient Network (LENet) for fast deblurring. We further integrate these two networks into an Adaptive Dual-network (ADNet), which dynamically selects the suitable network based on input blur severity, making it ideal for resource-constrained mobile devices. Extensive experiments show that our EG-Restormer and ADNet achieve state-of-the-art performance with a competitive speed. Project page: https://github.com/leejianping/ADNet
>
---
#### [new 087] Self-Supervised Selective-Guided Diffusion Model for Old-Photo Face Restoration
- **分类: cs.CV**

- **简介: 该论文研究老照片人脸修复，针对现有方法在局部伪影和颜色恢复上的不足，提出自监督选择性引导扩散模型（SSDiff），利用伪参考脸实现结构与颜色分阶段恢复，并构建新数据集VintageFace，提升了修复质量与可控性。**

- **链接: [http://arxiv.org/pdf/2510.12114v1](http://arxiv.org/pdf/2510.12114v1)**

> **作者:** Wenjie Li; Xiangyi Wang; Heng Guo; Guangwei Gao; Zhanyu Ma
>
> **摘要:** Old-photo face restoration poses significant challenges due to compounded degradations such as breakage, fading, and severe blur. Existing pre-trained diffusion-guided methods either rely on explicit degradation priors or global statistical guidance, which struggle with localized artifacts or face color. We propose Self-Supervised Selective-Guided Diffusion (SSDiff), which leverages pseudo-reference faces generated by a pre-trained diffusion model under weak guidance. These pseudo-labels exhibit structurally aligned contours and natural colors, enabling region-specific restoration via staged supervision: structural guidance applied throughout the denoising process and color refinement in later steps, aligned with the coarse-to-fine nature of diffusion. By incorporating face parsing maps and scratch masks, our method selectively restores breakage regions while avoiding identity mismatch. We further construct VintageFace, a 300-image benchmark of real old face photos with varying degradation levels. SSDiff outperforms existing GAN-based and diffusion-based methods in perceptual quality, fidelity, and regional controllability. Code link: https://github.com/PRIS-CV/SSDiff.
>
---
#### [new 088] A Review on Domain Adaption and Generative Adversarial Networks(GANs)
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在解决标注数据稀缺问题。通过综述域适应方法，探讨如何将源域训练的模型有效应用于目标域，提升跨域图像分类性能。**

- **链接: [http://arxiv.org/pdf/2510.12075v1](http://arxiv.org/pdf/2510.12075v1)**

> **作者:** Aashish Dhawan; Divyanshu Mudgal
>
> **摘要:** The major challenge in today's computer vision scenario is the availability of good quality labeled data. In a field of study like image classification, where data is of utmost importance, we need to find more reliable methods which can overcome the scarcity of data to produce results comparable to previous benchmark results. In most cases, obtaining labeled data is very difficult because of the high cost of human labor and in some cases impossible. The purpose of this paper is to discuss Domain Adaptation and various methods to implement it. The main idea is to use a model trained on a particular dataset to predict on data from a different domain of the same kind, for example - a model trained on paintings of airplanes predicting on real images of airplanes
>
---
#### [new 089] Audio-Guided Visual Perception for Audio-Visual Navigation
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 该论文研究音频-视觉导航任务，解决现有方法在新声音或新环境中泛化能力差的问题。提出AGVP框架，通过音频引导视觉注意力，实现跨模态对齐，提升导航效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.11760v1](http://arxiv.org/pdf/2510.11760v1)**

> **作者:** Yi Wang; Yinfeng Yu; Fuchun Sun; Liejun Wang; Wendong Zheng
>
> **备注:** Main paper (6 pages). Accepted for publication by International Conference on Virtual Reality and Visualization 2025 (ICVRV 2025)
>
> **摘要:** Audio-Visual Embodied Navigation aims to enable agents to autonomously navigate to sound sources in unknown 3D environments using auditory cues. While current AVN methods excel on in-distribution sound sources, they exhibit poor cross-source generalization: navigation success rates plummet and search paths become excessively long when agents encounter unheard sounds or unseen environments. This limitation stems from the lack of explicit alignment mechanisms between auditory signals and corresponding visual regions. Policies tend to memorize spurious \enquote{acoustic fingerprint-scenario} correlations during training, leading to blind exploration when exposed to novel sound sources. To address this, we propose the AGVP framework, which transforms sound from policy-memorable acoustic fingerprint cues into spatial guidance. The framework first extracts global auditory context via audio self-attention, then uses this context as queries to guide visual feature attention, highlighting sound-source-related regions at the feature level. Subsequent temporal modeling and policy optimization are then performed. This design, centered on interpretable cross-modal alignment and region reweighting, reduces dependency on specific acoustic fingerprints. Experimental results demonstrate that AGVP improves both navigation efficiency and robustness while achieving superior cross-scenario generalization on previously unheard sounds.
>
---
#### [new 090] Your VAR Model is Secretly an Efficient and Explainable Generative Classifier
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究生成式分类任务，旨在解决扩散模型计算成本高、可解释性差的问题。作者提出基于视觉自回归模型（VAR）的生成分类器A-VARC⁺，兼具高效推理、良好准确率、可解释性和抗灾难性遗忘能力。**

- **链接: [http://arxiv.org/pdf/2510.12060v1](http://arxiv.org/pdf/2510.12060v1)**

> **作者:** Yi-Chung Chen; David I. Inouye; Jing Gao
>
> **摘要:** Generative classifiers, which leverage conditional generative models for classification, have recently demonstrated desirable properties such as robustness to distribution shifts. However, recent progress in this area has been largely driven by diffusion-based models, whose substantial computational cost severely limits scalability. This exclusive focus on diffusion-based methods has also constrained our understanding of generative classifiers. In this work, we propose a novel generative classifier built on recent advances in visual autoregressive (VAR) modeling, which offers a new perspective for studying generative classifiers. To further enhance its performance, we introduce the Adaptive VAR Classifier$^+$ (A-VARC$^+$), which achieves a superior trade-off between accuracy and inference speed, thereby significantly improving practical applicability. Moreover, we show that the VAR-based method exhibits fundamentally different properties from diffusion-based methods. In particular, due to its tractable likelihood, the VAR-based classifier enables visual explainability via token-wise mutual information and demonstrates inherent resistance to catastrophic forgetting in class-incremental learning tasks.
>
---
#### [new 091] VISaGE: Understanding Visual Generics and Exceptions
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究视觉语言模型（VLM）在处理典型与异常图像时的概念理解能力，旨在揭示模型在语义先验与语用先验间的权衡问题。作者构建了新数据集VISaGE，通过实验发现，当图像与文本不一致时，模型概念理解显著下降。**

- **链接: [http://arxiv.org/pdf/2510.12548v1](http://arxiv.org/pdf/2510.12548v1)**

> **作者:** Stella Frank; Emily Allaway
>
> **备注:** EMNLP 2025
>
> **摘要:** While Vision Language Models (VLMs) learn conceptual representations, in the form of generalized knowledge, during training, they are typically used to analyze individual instances. When evaluation instances are atypical, this paradigm results in tension between two priors in the model. The first is a pragmatic prior that the textual and visual input are both relevant, arising from VLM finetuning on congruent inputs; the second is a semantic prior that the conceptual representation is generally true for instances of the category. In order to understand how VLMs trade off these priors, we introduce a new evaluation dataset, VISaGE, consisting of both typical and exceptional images. In carefully balanced experiments, we show that conceptual understanding degrades when the assumption of congruency underlying the pragmatic prior is violated with incongruent images. This effect is stronger than the effect of the semantic prior when querying about individual instances.
>
---
#### [new 092] DiffEM: Learning from Corrupted Data with Diffusion Models via Expectation Maximization
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究从含噪数据训练扩散模型的问题，提出DiffEM方法。通过期望最大化框架交替重建干净数据并优化模型，实现对污染数据的有效学习，适用于图像重建等逆问题。**

- **链接: [http://arxiv.org/pdf/2510.12691v1](http://arxiv.org/pdf/2510.12691v1)**

> **作者:** Danial Hosseintabar; Fan Chen; Giannis Daras; Antonio Torralba; Constantinos Daskalakis
>
> **摘要:** Diffusion models have emerged as powerful generative priors for high-dimensional inverse problems, yet learning them when only corrupted or noisy observations are available remains challenging. In this work, we propose a new method for training diffusion models with Expectation-Maximization (EM) from corrupted data. Our proposed method, DiffEM, utilizes conditional diffusion models to reconstruct clean data from observations in the E-step, and then uses the reconstructed data to refine the conditional diffusion model in the M-step. Theoretically, we provide monotonic convergence guarantees for the DiffEM iteration, assuming appropriate statistical conditions. We demonstrate the effectiveness of our approach through experiments on various image reconstruction tasks.
>
---
#### [new 093] SAIL-Embedding Technical Report: Omni-modal Embedding Foundation Model
- **分类: cs.IR; cs.CV**

- **简介: 该论文提出SAIL-Embedding，一种面向多模态检索与推荐的统一嵌入模型。针对模态支持有限、训练不稳定和工业场景适配差的问题，设计了多阶段训练策略与专用架构，在跨模态检索和推荐任务中显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.12709v1](http://arxiv.org/pdf/2510.12709v1)**

> **作者:** Lin Lin; Jiefeng Long; Zhihe Wan; Yuchi Wang; Dingkang Yang; Shuang Yang; Yueyang Yao; Xu Chen; Zirui Guo; Shengqiang Li; Weiran Li; Hanyu Li; Yaling Mou; Yan Qiu; Haiyang Yu; Xiao Liang; Hongsheng Li; Chao Feng
>
> **备注:** Technical Report
>
> **摘要:** Multimodal embedding models aim to yield informative unified representations that empower diverse cross-modal tasks. Despite promising developments in the evolution from CLIP-based dual-tower architectures to large vision-language models, prior works still face unavoidable challenges in real-world applications and business scenarios, such as the limited modality support, unstable training mechanisms, and industrial domain gaps. In this work, we introduce SAIL-Embedding, an omni-modal embedding foundation model that addresses these issues through tailored training strategies and architectural design. In the optimization procedure, we propose a multi-stage training scheme to boost the multifaceted effectiveness of representation learning. Specifically, the content-aware progressive training aims to enhance the model's adaptability to diverse downstream tasks and master enriched cross-modal proficiency. The collaboration-aware recommendation enhancement training further adapts multimodal representations for recommendation scenarios by distilling knowledge from sequence-to-item and ID-to-item embeddings while mining user historical interests. Concurrently, we develop the stochastic specialization and dataset-driven pattern matching to strengthen model training flexibility and generalizability. Experimental results show that SAIL-Embedding achieves SOTA performance compared to other methods in different retrieval tasks. In online experiments across various real-world scenarios integrated with our model, we observe a significant increase in Lifetime (LT), which is a crucial indicator for the recommendation experience. For instance, the model delivers the 7-day LT gain of +0.158% and the 14-day LT gain of +0.144% in the Douyin-Selected scenario. For the Douyin feed rank model, the match features produced by SAIL-Embedding yield a +0.08% AUC gain.
>
---
#### [new 094] MAPS: Masked Attribution-based Probing of Strategies- A computational framework to align human and model explanations
- **分类: q-bio.NC; cs.CV**

- **简介: 该论文提出MAPS框架，旨在评估人工神经网络解释方法是否能反映人类视觉策略。通过将归因图转化为解释掩码图像，结合少量行为数据，验证模型解释与人类行为的一致性，解决了传统方法需大量实验的问题，实现了高效、可扩展的人类与模型解释对齐。**

- **链接: [http://arxiv.org/pdf/2510.12141v1](http://arxiv.org/pdf/2510.12141v1)**

> **作者:** Sabine Muzellec; Yousif Kashef Alghetaa; Simon Kornblith; Kohitij Kar
>
> **摘要:** Human core object recognition depends on the selective use of visual information, but the strategies guiding these choices are difficult to measure directly. We present MAPS (Masked Attribution-based Probing of Strategies), a behaviorally validated computational tool that tests whether explanations derived from artificial neural networks (ANNs) can also explain human vision. MAPS converts attribution maps into explanation-masked images (EMIs) and compares image-by-image human accuracies on these minimal images with limited pixel budgets with accuracies on the full stimuli. MAPS provides a principled way to evaluate and choose among competing ANN interpretability methods. In silico, EMI-based behavioral similarity between models reliably recovers the ground-truth similarity computed from their attribution maps, establishing which explanation methods best capture the model's strategy. When applied to humans and macaques, MAPS identifies ANN-explanation combinations whose explanations align most closely with biological vision, achieving the behavioral validity of Bubble masks while requiring far fewer behavioral trials. Because it needs only access to model attributions and a modest set of behavioral data on the original images, MAPS avoids exhaustive psychophysics while offering a scalable tool for adjudicating explanations and linking human behavior, neural activity, and model decisions under a common standard.
>
---
#### [new 095] A Function Centric Perspective On Flat and Sharp Minima
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文探讨平坦与尖锐极小值对模型泛化的意义，挑战“尖锐意味着差泛化”的观点。研究表明，在正则化下，尖锐极小值常伴随更好性能，提出应从函数复杂性而非单纯平坦性理解损失景观，倡导以函数为中心重新审视优化解的几何性质。**

- **链接: [http://arxiv.org/pdf/2510.12451v1](http://arxiv.org/pdf/2510.12451v1)**

> **作者:** Israel Mason-Williams; Gabryel Mason-Williams; Helen Yannakoudakis
>
> **备注:** 26 pages, 26 tables, 63 figures, pre-print
>
> **摘要:** Flat minima are widely believed to correlate with improved generalisation in deep neural networks. However, this connection has proven more nuanced in recent studies, with both theoretical counterexamples and empirical exceptions emerging in the literature. In this paper, we revisit the role of sharpness in model performance, proposing that sharpness is better understood as a function-dependent property rather than a reliable indicator of poor generalisation. We conduct extensive empirical studies, from single-objective optimisation to modern image classification tasks, showing that sharper minima often emerge when models are regularised (e.g., via SAM, weight decay, or data augmentation), and that these sharp minima can coincide with better generalisation, calibration, robustness, and functional consistency. Across a range of models and datasets, we find that baselines without regularisation tend to converge to flatter minima yet often perform worse across all safety metrics. Our findings demonstrate that function complexity, rather than flatness alone, governs the geometry of solutions, and that sharper minima can reflect more appropriate inductive biases (especially under regularisation), calling for a function-centric reappraisal of loss landscape geometry.
>
---
#### [new 096] MosaicDiff: Training-free Structural Pruning for Diffusion Model Acceleration Reflecting Pretraining Dynamics
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对扩散模型推理加速任务，提出无需训练的结构剪枝方法MosaicDiff。通过分析预训练动态，设计轨迹感知的自适应剪枝策略，兼顾不同学习阶段特性，在DiT和SDXL上实现高效采样加速且保持生成质量。**

- **链接: [http://arxiv.org/pdf/2510.11962v1](http://arxiv.org/pdf/2510.11962v1)**

> **作者:** Bowei Guo; Shengkun Tang; Cong Zeng; Zhiqiang Shen
>
> **备注:** International Conference on Computer Vision, ICCV 2025
>
> **摘要:** Diffusion models are renowned for their generative capabilities, yet their pretraining processes exhibit distinct phases of learning speed that have been entirely overlooked in prior post-training acceleration efforts in the community. In this study, we introduce a novel framework called MosaicDiff that aligns diffusion pretraining dynamics with post-training sampling acceleration via trajectory-aware structural pruning. Our approach leverages the observation that the middle, fast-learning stage of diffusion pretraining requires more conservative pruning to preserve critical model features, while the early and later, slow-learning stages benefit from a more aggressive pruning strategy. This adaptive pruning mechanism is the first to explicitly mirror the inherent learning speed variations of diffusion pretraining, thereby harmonizing the model's inner training dynamics with its accelerated sampling process. Extensive experiments on DiT and SDXL demonstrate that our method achieves significant speed-ups in sampling without compromising output quality, outperforming previous state-of-the-art methods by large margins, also providing a new viewpoint for more efficient and robust training-free diffusion acceleration.
>
---
#### [new 097] Fast Visuomotor Policy for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对高频率、资源受限的机器人操作任务，提出一种名为Energy Policy的快速视觉运动策略框架。通过能量得分学习目标和高效MLP架构，实现单次前向传播的多模态动作预测，在降低计算开销的同时保持高性能。**

- **链接: [http://arxiv.org/pdf/2510.12483v1](http://arxiv.org/pdf/2510.12483v1)**

> **作者:** Jingkai Jia; Tong Yang; Xueyao Chen; Chenhuan Liu; Wenqiang Zhang
>
> **摘要:** We present a fast and effective policy framework for robotic manipulation, named Energy Policy, designed for high-frequency robotic tasks and resource-constrained systems. Unlike existing robotic policies, Energy Policy natively predicts multimodal actions in a single forward pass, enabling high-precision manipulation at high speed. The framework is built upon two core components. First, we adopt the energy score as the learning objective to facilitate multimodal action modeling. Second, we introduce an energy MLP to implement the proposed objective while keeping the architecture simple and efficient. We conduct comprehensive experiments in both simulated environments and real-world robotic tasks to evaluate the effectiveness of Energy Policy. The results show that Energy Policy matches or surpasses the performance of state-of-the-art manipulation methods while significantly reducing computational overhead. Notably, on the MimicGen benchmark, Energy Policy achieves superior performance with at a faster inference compared to existing approaches.
>
---
#### [new 098] Omni-Captioner: Data Pipeline, Models, and Benchmark for Omni Detailed Perception
- **分类: cs.CL; cs.CV; cs.MM; cs.SD**

- **简介: 该论文研究多模态细粒度感知任务，旨在解决现有模型在生成细节时易产生幻觉的问题。作者提出Omni-Detective数据生成 pipeline，训练Audio-Captioner和Omni-Captioner模型，并构建新基准Omni-Cloze，实现更准确、可靠的细粒度音频-视觉描述。**

- **链接: [http://arxiv.org/pdf/2510.12720v1](http://arxiv.org/pdf/2510.12720v1)**

> **作者:** Ziyang Ma; Ruiyang Xu; Zhenghao Xing; Yunfei Chu; Yuxuan Wang; Jinzheng He; Jin Xu; Pheng-Ann Heng; Kai Yu; Junyang Lin; Eng Siong Chng; Xie Chen
>
> **备注:** https://github.com/ddlBoJack/Omni-Captioner
>
> **摘要:** Fine-grained perception of multimodal information is critical for advancing human-AI interaction. With recent progress in audio-visual technologies, Omni Language Models (OLMs), capable of processing audio and video signals in parallel, have emerged as a promising paradigm for achieving richer understanding and reasoning. However, their capacity to capture and describe fine-grained details remains limited explored. In this work, we present a systematic and comprehensive investigation of omni detailed perception from the perspectives of the data pipeline, models, and benchmark. We first identify an inherent "co-growth" between detail and hallucination in current OLMs. To address this, we propose Omni-Detective, an agentic data generation pipeline integrating tool-calling, to autonomously produce highly detailed yet minimally hallucinatory multimodal data. Based on the data generated with Omni-Detective, we train two captioning models: Audio-Captioner for audio-only detailed perception, and Omni-Captioner for audio-visual detailed perception. Under the cascade evaluation protocol, Audio-Captioner achieves the best performance on MMAU and MMAR among all open-source models, surpassing Gemini 2.5 Flash and delivering performance comparable to Gemini 2.5 Pro. On existing detailed captioning benchmarks, Omni-Captioner sets a new state-of-the-art on VDC and achieves the best trade-off between detail and hallucination on the video-SALMONN 2 testset. Given the absence of a dedicated benchmark for omni detailed perception, we design Omni-Cloze, a novel cloze-style evaluation for detailed audio, visual, and audio-visual captioning that ensures stable, efficient, and reliable assessment. Experimental results and analysis demonstrate the effectiveness of Omni-Detective in generating high-quality detailed captions, as well as the superiority of Omni-Cloze in evaluating such detailed captions.
>
---
#### [new 099] GS-Verse: Mesh-based Gaussian Splatting for Physics-aware Interaction in Virtual Reality
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出GS-Verse，旨在解决虚拟现实中3D内容物理交互中视觉保真度与物理准确性不足的问题。通过将网格与高斯点阵结合，实现高保真、可变形的交互，并支持物理引擎无关的灵活部署，提升操作的真实感与一致性。**

- **链接: [http://arxiv.org/pdf/2510.11878v1](http://arxiv.org/pdf/2510.11878v1)**

> **作者:** Anastasiya Pechko; Piotr Borycki; Joanna Waczyńska; Daniel Barczyk; Agata Szymańska; Sławomir Tadeja; Przemysław Spurek
>
> **摘要:** As the demand for immersive 3D content grows, the need for intuitive and efficient interaction methods becomes paramount. Current techniques for physically manipulating 3D content within Virtual Reality (VR) often face significant limitations, including reliance on engineering-intensive processes and simplified geometric representations, such as tetrahedral cages, which can compromise visual fidelity and physical accuracy. In this paper, we introduce \our{} (\textbf{G}aussian \textbf{S}platting for \textbf{V}irtual \textbf{E}nvironment \textbf{R}endering and \textbf{S}cene \textbf{E}diting), a novel method designed to overcome these challenges by directly integrating an object's mesh with a Gaussian Splatting (GS) representation. Our approach enables more precise surface approximation, leading to highly realistic deformations and interactions. By leveraging existing 3D mesh assets, \our{} facilitates seamless content reuse and simplifies the development workflow. Moreover, our system is designed to be physics-engine-agnostic, granting developers robust deployment flexibility. This versatile architecture delivers a highly realistic, adaptable, and intuitive approach to interactive 3D manipulation. We rigorously validate our method against the current state-of-the-art technique that couples VR with GS in a comparative user study involving 18 participants. Specifically, we demonstrate that our approach is statistically significantly better for physics-aware stretching manipulation and is also more consistent in other physics-based manipulations like twisting and shaking. Further evaluation across various interactions and scenes confirms that our method consistently delivers high and reliable performance, showing its potential as a plausible alternative to existing methods.
>
---
#### [new 100] Tensor Completion via Monotone Inclusion: Generalized Low-Rank Priors Meet Deep Denoisers
- **分类: math.OC; cs.CV; 65K10, 68T07, 94A08**

- **简介: 该论文研究张量补全任务，旨在解决高维数据缺失问题。提出一种基于单调包含的新框架，结合广义低秩先验与深度伪压缩去噪器，克服了现有方法依赖不实假设的问题，并设计GTCTV-DPC算法，理论证明其收敛性，实验验证了优越性能。**

- **链接: [http://arxiv.org/pdf/2510.12425v1](http://arxiv.org/pdf/2510.12425v1)**

> **作者:** Peng Chen; Deliang Wei; Jiale Yao; Fang Li
>
> **备注:** 22 pages, 5 figures
>
> **摘要:** Missing entries in multi dimensional data pose significant challenges for downstream analysis across diverse real world applications. These data are naturally modeled as tensors, and recent completion methods integrating global low rank priors with plug and play denoisers have demonstrated strong empirical performance. However, these approaches often rely on empirical convergence alone or unrealistic assumptions, such as deep denoisers acting as proximal operators of implicit regularizers, which generally does not hold. To address these limitations, we propose a novel tensor completion framework grounded in the monotone inclusion paradigm, which unifies generalized low rank priors with deep pseudo contractive denoisers and extends beyond traditional convex optimization. Building on the Davis Yin splitting scheme, we develop the GTCTV DPC algorithm and rigorously establish its global convergence. Extensive experiments demonstrate that GTCTV DPC consistently outperforms existing methods in both quantitative metrics and visual quality, particularly at low sampling rates.
>
---
#### [new 101] SeeingSounds: Learning Audio-to-Visual Alignment via Text
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 该论文研究音频到图像生成任务，旨在无需配对音视频数据下实现可控生成。提出SeeingSounds框架，通过文本桥接音频与视觉模态，利用冻结的扩散模型和轻量适配器，实现高效跨模态对齐与细粒度控制。**

- **链接: [http://arxiv.org/pdf/2510.11738v1](http://arxiv.org/pdf/2510.11738v1)**

> **作者:** Simone Carnemolla; Matteo Pennisi; Chiara Russo; Simone Palazzo; Daniela Giordano; Concetto Spampinato
>
> **备注:** accepted to ACM Multimedia Asia 2025
>
> **摘要:** We introduce SeeingSounds, a lightweight and modular framework for audio-to-image generation that leverages the interplay between audio, language, and vision-without requiring any paired audio-visual data or training on visual generative models. Rather than treating audio as a substitute for text or relying solely on audio-to-text mappings, our method performs dual alignment: audio is projected into a semantic language space via a frozen language encoder, and, contextually grounded into the visual domain using a vision-language model. This approach, inspired by cognitive neuroscience, reflects the natural cross-modal associations observed in human perception. The model operates on frozen diffusion backbones and trains only lightweight adapters, enabling efficient and scalable learning. Moreover, it supports fine-grained and interpretable control through procedural text prompt generation, where audio transformations (e.g., volume or pitch shifts) translate into descriptive prompts (e.g., "a distant thunder") that guide visual outputs. Extensive experiments across standard benchmarks confirm that SeeingSounds outperforms existing methods in both zero-shot and supervised settings, establishing a new state of the art in controllable audio-to-visual generation.
>
---
#### [new 102] Gaussian Semantic Field for One-shot LiDAR Global Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究LiDAR全局定位任务，旨在解决语义地标重复导致的误匹配问题。作者提出高斯语义场模型，用连续函数表达语义分布，构建轻量三层场景图，提升单次定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.12101v1](http://arxiv.org/pdf/2510.12101v1)**

> **作者:** Pengyu Yin; Shenghai Yuan; Haozhi Cao; Xingyu Ji; Ruofei Bai; Siyu Chen; Lihua Xie
>
> **摘要:** We present a one-shot LiDAR global localization algorithm featuring semantic disambiguation ability based on a lightweight tri-layered scene graph. While landmark semantic registration-based methods have shown promising performance improvements in global localization compared with geometric-only methods, landmarks can be repetitive and misleading for correspondence establishment. We propose to mitigate this problem by modeling semantic distributions with continuous functions learned from a population of Gaussian processes. Compared with discrete semantic labels, the continuous functions capture finer-grained geo-semantic information and also provide more detailed metric information for correspondence establishment. We insert this continuous function as the middle layer between the object layer and the metric-semantic layer, forming a tri-layered 3D scene graph, serving as a light-weight yet performant backend for one-shot localization. We term our global localization pipeline Outram-GSF (Gaussian semantic field) and conduct a wide range of experiments on publicly available data sets, validating the superior performance against the current state-of-the-art.
>
---
## 更新

#### [replaced 001] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07392v3](http://arxiv.org/pdf/2503.07392v3)**

> **作者:** Ouxiang Li; Yuan Wang; Xinting Hu; Houcheng Jiang; Tao Liang; Yanbin Hao; Guojun Ma; Fuli Feng
>
> **备注:** This version has been temporarily withdrawn for procedural review purposes. The withdrawal is unrelated to the technical content of the paper
>
> **摘要:** Erasing concepts from large-scale text-to-image (T2I) diffusion models has become increasingly crucial due to the growing concerns over copyright infringement, offensive content, and privacy violations. In scalable applications, fine-tuning-based methods are time-consuming to precisely erase multiple target concepts, while real-time editing-based methods often degrade the generation quality of non-target concepts due to conflicting optimization objectives. To address this dilemma, we introduce SPEED, an efficient concept erasure approach that directly edits model parameters. SPEED searches for a null space, a model editing space where parameter updates do not affect non-target concepts, to achieve scalable and precise erasure. To facilitate accurate null space optimization, we incorporate three complementary strategies: Influence-based Prior Filtering (IPF) to selectively retain the most affected non-target concepts, Directed Prior Augmentation (DPA) to enrich the filtered retain set with semantically consistent variations, and Invariant Equality Constraints (IEC) to preserve key invariants during the T2I generation process. Extensive evaluations across multiple concept erasure tasks demonstrate that SPEED consistently outperforms existing methods in non-target preservation while achieving efficient and high-fidelity concept erasure, successfully erasing 100 concepts within only 5 seconds. Our code and models are available at: https://github.com/Ouxiang-Li/SPEED.
>
---
#### [replaced 002] Calibration and Uncertainty for multiRater Volume Assessment in multiorgan Segmentation (CURVAS) challenge results
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08685v2](http://arxiv.org/pdf/2505.08685v2)**

> **作者:** Meritxell Riera-Marin; Sikha O K; Julia Rodriguez-Comas; Matthias Stefan May; Zhaohong Pan; Xiang Zhou; Xiaokun Liang; Franciskus Xaverius Erick; Andrea Prenner; Cedric Hemon; Valentin Boussot; Jean-Louis Dillenseger; Jean-Claude Nunes; Abdul Qayyum; Moona Mazher; Steven A Niederer; Kaisar Kushibar; Carlos Martin-Isla; Petia Radeva; Karim Lekadir; Theodore Barfoot; Luis C. Garcia Peraza Herrera; Ben Glocker; Tom Vercauteren; Lucas Gago; Justin Englemann; Joy-Marie Kleiss; Anton Aubanell; Andreu Antolin; Javier Garcia-Lopez; Miguel A. Gonzalez Ballester; Adrian Galdran
>
> **备注:** This challenge was hosted in MICCAI 2024
>
> **摘要:** Deep learning (DL) has become the dominant approach for medical image segmentation, yet ensuring the reliability and clinical applicability of these models requires addressing key challenges such as annotation variability, calibration, and uncertainty estimation. This is why we created the Calibration and Uncertainty for multiRater Volume Assessment in multiorgan Segmentation (CURVAS), which highlights the critical role of multiple annotators in establishing a more comprehensive ground truth, emphasizing that segmentation is inherently subjective and that leveraging inter-annotator variability is essential for robust model evaluation. Seven teams participated in the challenge, submitting a variety of DL models evaluated using metrics such as Dice Similarity Coefficient (DSC), Expected Calibration Error (ECE), and Continuous Ranked Probability Score (CRPS). By incorporating consensus and dissensus ground truth, we assess how DL models handle uncertainty and whether their confidence estimates align with true segmentation performance. Our findings reinforce the importance of well-calibrated models, as better calibration is strongly correlated with the quality of the results. Furthermore, we demonstrate that segmentation models trained on diverse datasets and enriched with pre-trained knowledge exhibit greater robustness, particularly in cases deviating from standard anatomical structures. Notably, the best-performing models achieved high DSC and well-calibrated uncertainty estimates. This work underscores the need for multi-annotator ground truth, thorough calibration assessments, and uncertainty-aware evaluations to develop trustworthy and clinically reliable DL-based medical image segmentation models.
>
---
#### [replaced 003] Exploring Facial Biomarkers for Depression through Temporal Analysis of Action Units
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.13753v2](http://arxiv.org/pdf/2407.13753v2)**

> **作者:** Aditya Parikh; Misha Sadeghi; Robert Richer; Lydia Helene Rupp; Lena Schindler-Gmelch; Marie Keinert; Malin Hager; Klara Capito; Farnaz Rahimi; Bernhard Egger; Matthias Berking; Bjoern M. Eskofier
>
> **备注:** Updated Authors
>
> **摘要:** Depression is characterized by persistent sadness and loss of interest, significantly impairing daily functioning and now a widespread mental disorder. Traditional diagnostic methods rely on subjective assessments, necessitating objective approaches for accurate diagnosis. Our study investigates the use of facial action units (AUs) and emotions as biomarkers for depression. We analyzed facial expressions from video data of participants classified with or without depression. Our methodology involved detailed feature extraction, mean intensity comparisons of key AUs, and the application of time series classification models. Furthermore, we employed Principal Component Analysis (PCA) and various clustering algorithms to explore the variability in emotional expression patterns. Results indicate significant differences in the intensities of AUs associated with sadness and happiness between the groups, highlighting the potential of facial analysis in depression assessment.
>
---
#### [replaced 004] Online Topological Localization for Navigation Assistance in Bronchoscopy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.09144v2](http://arxiv.org/pdf/2510.09144v2)**

> **作者:** Clara Tomasini; Luis Riazuelo; Ana C. Murillo
>
> **摘要:** Video bronchoscopy is a fundamental procedure in respiratory medicine, where medical experts navigate through the bronchial tree of a patient to diagnose or operate the patient. Surgeons need to determine the position of the scope as they go through the airway until they reach the area of interest. This task is very challenging for practitioners due to the complex bronchial tree structure and varying doctor experience and training. Navigation assistance to locate the bronchoscope during the procedure can improve its outcome. Currently used techniques for navigational guidance commonly rely on previous CT scans of the patient to obtain a 3D model of the airway, followed by tracking of the scope with additional sensors or image registration. These methods obtain accurate locations but imply additional setup, scans and training. Accurate metric localization is not always required, and a topological localization with regard to a generic airway model can often suffice to assist the surgeon with navigation. We present an image-based bronchoscopy topological localization pipeline to provide navigation assistance during the procedure, with no need of patient CT scan. Our approach is trained only on phantom data, eliminating the high cost of real data labeling, and presents good generalization capabilities. The results obtained surpass existing methods, particularly on real data test sequences.
>
---
#### [replaced 005] REACT3D: Recovering Articulations for Interactive Physical 3D Scenes
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.11340v2](http://arxiv.org/pdf/2510.11340v2)**

> **作者:** Zhao Huang; Boyang Sun; Alexandros Delitzas; Jiaqi Chen; Marc Pollefeys
>
> **备注:** 8 pages
>
> **摘要:** Interactive 3D scenes are increasingly vital for embodied intelligence, yet existing datasets remain limited due to the labor-intensive process of annotating part segmentation, kinematic types, and motion trajectories. We present REACT3D, a scalable zero-shot framework that converts static 3D scenes into simulation-ready interactive replicas with consistent geometry, enabling direct use in diverse downstream tasks. Our contributions include: (i) openable-object detection and segmentation to extract candidate movable parts from static scenes, (ii) articulation estimation that infers joint types and motion parameters, (iii) hidden-geometry completion followed by interactive object assembly, and (iv) interactive scene integration in widely supported formats to ensure compatibility with standard simulation platforms. We achieve state-of-the-art performance on detection/segmentation and articulation metrics across diverse indoor scenes, demonstrating the effectiveness of our framework and providing a practical foundation for scalable interactive scene generation, thereby lowering the barrier to large-scale research on articulated scene understanding. Our project page is https://react3d.github.io/
>
---
#### [replaced 006] GarmageNet: A Multimodal Generative Framework for Sewing Pattern Design and Generic Garment Modeling
- **分类: cs.GR; cs.CV; I.3.5; I.2.10**

- **链接: [http://arxiv.org/pdf/2504.01483v4](http://arxiv.org/pdf/2504.01483v4)**

> **作者:** Siran Li; Ruiyang Liu; Chen Liu; Zhendong Wang; Gaofeng He; Yong-Lu Li; Xiaogang Jin; Huamin Wang
>
> **备注:** 23 pages,20 figures
>
> **摘要:** Realistic digital garment modeling remains a labor-intensive task due to the intricate process of translating 2D sewing patterns into high-fidelity, simulation-ready 3D garments. We introduce GarmageNet, a unified generative framework that automates the creation of 2D sewing patterns, the construction of sewing relationships, and the synthesis of 3D garment initializations compatible with physics-based simulation. Central to our approach is Garmage, a novel garment representation that encodes each panel as a structured geometry image, effectively bridging the semantic and geometric gap between 2D structural patterns and 3D garment geometries. Followed by GarmageNet, a latent diffusion transformer to synthesize panel-wise geometry images and GarmageJigsaw, a neural module for predicting point-to-point sewing connections along panel contours. To support training and evaluation, we build GarmageSet, a large-scale dataset comprising 14,801 professionally designed garments with detailed structural and style annotations. Our method demonstrates versatility and efficacy across multiple application scenarios, including scalable garment generation from multi-modal design concepts (text prompts, sketches, photographs), automatic modeling from raw flat sewing patterns, pattern recovery from unstructured point clouds, and progressive garment editing using conventional instructions, laying the foundation for fully automated, production-ready pipelines in digital fashion. Project page: https://style3d.github.io/garmagenet/.
>
---
#### [replaced 007] FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.11190v2](http://arxiv.org/pdf/2510.11190v2)**

> **作者:** Shengming Yuan; Xinyu Lyu; Shuailong Wang; Beitao Chen; Jingkuan Song; Lianli Gao
>
> **备注:** 19 pages, 11 figures. Accepted by the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Multimodal large language models (MLLMs) face an inherent trade-off between faithfulness and creativity, as different tasks require varying degrees of associative reasoning. However, existing methods lack the flexibility to modulate this reasoning strength, limiting MLLMs' adaptability across factual and creative scenarios. To bridge this gap, we propose equipping MLLMs with mechanisms that enable flexible control over associative reasoning. We begin by investigating the internal mechanisms underlying associative behavior in MLLMs and find that: (1) middle layers play a pivotal role in shaping model's associative tendencies, (2) modifying representations in these layers effectively regulates associative reasoning strength, and (3) hallucinations can be exploited to derive steering vectors that guide this modulation. Building on these findings, we introduce Flexible Association Control (FlexAC), a lightweight and training-free framework for modulating associative behavior in MLLMs. FlexAC first induces hallucination-guided intermediate representations to encode associative directions. Then, it selects high-association instances to construct effective associative steering vectors, whose strengths are adaptively calibrated to balance creative guidance with output stability. Finally, recognizing the multi-dimensional nature of associative reasoning, FlexAC incorporates task-specific associative vectors derived from a forward pass on a few target-domain samples, enabling models to follow diverse associative directions and better adapt to creative tasks. Notably, our method achieves up to a 5.8x improvement in creativity on Creation-MMBench and a 29% reduction in hallucination rate on CHAIR, surpassing existing baselines and demonstrating its effectiveness in enabling flexible control over associative reasoning in MLLMs. Our code is available at https://github.com/ylhz/FlexAC.
>
---
#### [replaced 008] DSM: Constructing a Diverse Semantic Map for 3D Visual Grounding
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.08307v2](http://arxiv.org/pdf/2504.08307v2)**

> **作者:** Qinghongbing Xie; Zijian Liang; Fuhao Li; Long Zeng
>
> **备注:** 8 pages, 6 figures, Project Page: https://binicey.github.io/DSM
>
> **摘要:** Effective scene representation is critical for the visual grounding ability of representations, yet existing methods for 3D Visual Grounding are often constrained. They either only focus on geometric and visual cues, or, like traditional 3D scene graphs, lack the multi-dimensional attributes needed for complex reasoning. To bridge this gap, we introduce the Diverse Semantic Map (DSM) framework, a novel scene representation framework that enriches robust geometric models with a spectrum of VLM-derived semantics, including appearance, physical properties, and affordances. The DSM is first constructed online by fusing multi-view observations within a temporal sliding window, creating a persistent and comprehensive world model. Building on this foundation, we propose DSM-Grounding, a new paradigm that shifts grounding from free-form VLM queries to a structured reasoning process over the semantic-rich map, markedly improving accuracy and interpretability. Extensive evaluations validate our approach's superiority. On the ScanRefer benchmark, DSM-Grounding achieves a state-of-the-art 59.06% overall accuracy of IoU@0.5, surpassing others by 10%. In semantic segmentation, our DSM attains a 67.93% F-mIoU, outperforming all baselines, including privileged ones. Furthermore, successful deployment on physical robots for complex navigation and grasping tasks confirms the framework's practical utility in real-world scenarios.
>
---
#### [replaced 009] Joint Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for Self Supervised Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12477v2](http://arxiv.org/pdf/2505.12477v2)**

> **作者:** Hugues Van Assel; Mark Ibrahim; Tommaso Biancalani; Aviv Regev; Randall Balestriero
>
> **备注:** 33 pages, 9 figures
>
> **摘要:** Reconstruction and joint embedding have emerged as two leading paradigms in Self Supervised Learning (SSL). Reconstruction methods focus on recovering the original sample from a different view in input space. On the other hand, joint embedding methods align the representations of different views in latent space. Both approaches offer compelling advantages, yet practitioners lack clear guidelines for choosing between them. In this work, we unveil the core mechanisms that distinguish each paradigm. By leveraging closed form solutions for both approaches, we precisely characterize how the view generation process, e.g. data augmentation, impacts the learned representations. We then demonstrate that, unlike supervised learning, both SSL paradigms require a minimal alignment between augmentations and irrelevant features to achieve asymptotic optimality with increasing sample size. Our findings indicate that in scenarios where these irrelevant features have a large magnitude, joint embedding methods are preferable because they impose a strictly weaker alignment condition compared to reconstruction based methods. These results not only clarify the trade offs between the two paradigms but also substantiate the empirical success of joint embedding approaches on real world challenging datasets.
>
---
#### [replaced 010] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12434v4](http://arxiv.org/pdf/2505.12434v4)**

> **作者:** Qi Wang; Yanrui Yu; Ye Yuan; Rui Mao; Tianfei Zhou
>
> **备注:** Accepted by NeurIPS 2025. Code: https://github.com/QiWang98/VideoRFT
>
> **摘要:** Reinforcement fine-tuning (RFT) has shown great promise in achieving humanlevel reasoning capabilities of Large Language Models (LLMs), and has recently been extended to MLLMs. Nevertheless, reasoning about videos, which is a fundamental aspect of human intelligence, remains a persistent challenge due to the complex logic, temporal and causal structures inherent in video data. To fill this gap, we propose VideoRFT, a novel approach that extends the RFT paradigm to cultivate human-like video reasoning capabilities in MLLMs. VideoRFT follows the standard two-stage scheme in RFT: supervised fine-tuning (SFT) with chain-of-thought (CoT) annotations, followed by reinforcement learning (RL) to improve generalization. A central challenge to achieve this in the video domain lies in the scarcity of large-scale, high-quality video CoT datasets. We address this by building a multi-expert-driven, cognition-inspired CoT curation pipeline. First, we devise a cognition-inspired prompting strategy to elicit a reasoning LLM to generate preliminary CoTs based solely on rich, structured, and literal representations of video content. Subsequently, these CoTs are revised by a MLLM conditioned on the actual video, ensuring visual consistency and reducing visual hallucinations. This pipeline results in two new datasets, i.e.VideoRFT-CoT-102K for SFT and VideoRFT-RL-310K for RL. To further strengthen the RL phase, we introduce a novel semantic-consistency reward that explicitly promotes the alignment between textual reasoning and visual evidence. This reward encourages the model to produce coherent, context-aware reasoning outputs grounded in visual input. Extensive experiments show that VideoRFT achieves state-of-the-art performance on six video reasoning benchmarks.
>
---
#### [replaced 011] Semi-Unsupervised Microscopy Segmentation with Fuzzy Logic and Spatial Statistics for Cross-Domain Analysis Using a GUI
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.15979v2](http://arxiv.org/pdf/2508.15979v2)**

> **作者:** Surajit Das; Pavel Zun
>
> **摘要:** Brightfield microscopy of unstained live cells is challenging due to low contrast, dynamic morphology, uneven illumination, and lack of labels. Deep learning achieved SOTA performance on stained, high-contrast images but needs large labeled datasets, expensive hardware, and fails under uneven illumination. This study presents a low-cost, lightweight, annotation-free segmentation method by introducing one-time calibration-assisted unsupervised framework adaptable across imaging modalities and image type. The framework determines background via spatial standard deviation from the local mean. Uncertain pixels are resolved using fuzzy logic, cumulative squared shift of nodal intensity, statistical features, followed by post-segmentation denoising calibration which is saved as a profile for reuse until noise pattern or object type substantially change. The program runs as a script or graphical interface for non-programmers. The method was rigorously evaluated using \textit{IoU}, \textit{F1-score}, and other metrics, with statistical significance confirmed via Wilcoxon signed-rank tests. On unstained brightfield myoblast (C2C12) images, it outperformed \textit{Cellpose 3.0} and \textit{StarDist}, improving IoU by up to 48\% (average IoU = 0.43, F1 = 0.60). In phase-contrast microscopy, it achieved a mean IoU of 0.69 and an F1-score of 0.81 on the \textit{LIVECell} dataset ($n = 3178$), with substantial expert agreement ($\kappa > 0.75$) confirming cross-modality robustness. Successful segmentation of laser-affected polymer surfaces further confirmed cross-domain robustness. By introducing the \textit{Homogeneous Image Plane} concept, this work provides a new theoretical foundation for training-free, annotation-free segmentation. The framework operates efficiently on CPU, avoids cell staining, and is practical for live-cell imaging and biomedical applications.
>
---
#### [replaced 012] Logarithmic Mathematical Morphology: theory and applications
- **分类: eess.IV; cs.CV; cs.NA; math.FA; math.NA**

- **链接: [http://arxiv.org/pdf/2309.02007v2](http://arxiv.org/pdf/2309.02007v2)**

> **作者:** Guillaume Noyel
>
> **摘要:** In Mathematical Morphology for grey-level functions, an image is analysed by another image named the structuring function. This structuring function is translated over the image domain and summed to the image. However, in an image presenting lighting variations, the amplitude of the structuring function should vary according to the image intensity. Such a property is not verified in Mathematical Morphology for grey level functions, when the structuring function is summed to the image with the usual additive law. In order to address this issue, a new framework is defined with an additive law for which the amplitude of the structuring function varies according to the image amplitude. This additive law is chosen within the Logarithmic Image Processing framework and models the lighting variations with a physical cause such as a change of light intensity. The new framework is named Logarithmic Mathematical Morphology (LMM) and allows the definition of operators which are robust to such lighting variations.
>
---
#### [replaced 013] Enhancing Representations through Heterogeneous Self-Supervised Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2310.05108v4](http://arxiv.org/pdf/2310.05108v4)**

> **作者:** Zhong-Yu Li; Bo-Wen Yin; Yongxiang Liu; Li Liu; Ming-Ming Cheng
>
> **摘要:** Incorporating heterogeneous representations from different architectures has facilitated various vision tasks, e.g., some hybrid networks combine transformers and convolutions. However, complementarity between such heterogeneous architectures has not been well exploited in self-supervised learning. Thus, we propose Heterogeneous Self-Supervised Learning (HSSL), which enforces a base model to learn from an auxiliary head whose architecture is heterogeneous from the base model. In this process, HSSL endows the base model with new characteristics in a representation learning way without structural changes. To comprehensively understand the HSSL, we conduct experiments on various heterogeneous pairs containing a base model and an auxiliary head. We discover that the representation quality of the base model moves up as their architecture discrepancy grows. This observation motivates us to propose a search strategy that quickly determines the most suitable auxiliary head for a specific base model to learn and several simple but effective methods to enlarge the model discrepancy. The HSSL is compatible with various self-supervised methods, achieving superior performances on various downstream tasks, including image classification, semantic segmentation, instance segmentation, and object detection. The codes are available at https://github.com/NK-JittorCV/Self-Supervised/.
>
---
#### [replaced 014] Tracing Back the Malicious Clients in Poisoning Attacks to Federated Learning
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2407.07221v2](http://arxiv.org/pdf/2407.07221v2)**

> **作者:** Yuqi Jia; Minghong Fang; Hongbin Liu; Jinghuai Zhang; Neil Zhenqiang Gong
>
> **备注:** Conference on Neural Information Processing Systems (NeurIPS) 2025
>
> **摘要:** Poisoning attacks compromise the training phase of federated learning (FL) such that the learned global model misclassifies attacker-chosen inputs called target inputs. Existing defenses mainly focus on protecting the training phase of FL such that the learnt global model is poison free. However, these defenses often achieve limited effectiveness when the clients' local training data is highly non-iid or the number of malicious clients is large, as confirmed in our experiments. In this work, we propose FLForensics, the first poison-forensics method for FL. FLForensics complements existing training-phase defenses. In particular, when training-phase defenses fail and a poisoned global model is deployed, FLForensics aims to trace back the malicious clients that performed the poisoning attack after a misclassified target input is identified. We theoretically show that FLForensics can accurately distinguish between benign and malicious clients under a formal definition of poisoning attack. Moreover, we empirically show the effectiveness of FLForensics at tracing back both existing and adaptive poisoning attacks on five benchmark datasets.
>
---
#### [replaced 015] Efficient Fine-Tuning of DINOv3 Pretrained on Natural Images for Atypical Mitotic Figure Classification (MIDOG 2025 Task 2 Winner)
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.21041v3](http://arxiv.org/pdf/2508.21041v3)**

> **作者:** Guillaume Balezo; Hana Feki; Raphaël Bourgade; Lily Monnier; Matthieu Blons; Alice Blondel; Etienne Decencière; Albert Pla Planas; Thomas Walter
>
> **备注:** 4 pages. Challenge report for MIDOG 2025 (Task 2: Atypical Mitotic Figure Classification)
>
> **摘要:** Atypical mitotic figures (AMFs) represent abnormal cell division associated with poor prognosis. Yet their detection remains difficult due to low prevalence, subtle morphology, and inter-observer variability. The MIDOG 2025 challenge introduces a benchmark for AMF classification across multiple domains. In this work, we fine-tuned the recently published DINOv3-H+ vision transformer, pretrained on natural images, using low-rank adaptation (LoRA), training only ~1.3M parameters in combination with extensive augmentation and a domain-weighted Focal Loss to handle domain heterogeneity. Despite the domain gap, our fine-tuned DINOv3 transfers effectively to histopathology, reaching first place on the final test set. These results highlight the advantages of DINOv3 pretraining and underline the efficiency and robustness of our fine-tuning strategy, yielding state-of-the-art results for the atypical mitosis classification challenge in MIDOG 2025.
>
---
#### [replaced 016] TTT3R: 3D Reconstruction as Test-Time Training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.26645v2](http://arxiv.org/pdf/2509.26645v2)**

> **作者:** Xingyu Chen; Yue Chen; Yuliang Xiu; Andreas Geiger; Anpei Chen
>
> **备注:** Page: https://rover-xingyu.github.io/TTT3R/ Code: https://github.com/Inception3D/TTT3R
>
> **摘要:** Modern Recurrent Neural Networks have become a competitive architecture for 3D reconstruction due to their linear-time complexity. However, their performance degrades significantly when applied beyond the training context length, revealing limited length generalization. In this work, we revisit the 3D reconstruction foundation models from a Test-Time Training perspective, framing their designs as an online learning problem. Building on this perspective, we leverage the alignment confidence between the memory state and incoming observations to derive a closed-form learning rate for memory updates, to balance between retaining historical information and adapting to new observations. This training-free intervention, termed TTT3R, substantially improves length generalization, achieving a $2\times$ improvement in global pose estimation over baselines, while operating at 20 FPS with just 6 GB of GPU memory to process thousands of images. Code available in https://rover-xingyu.github.io/TTT3R
>
---
#### [replaced 017] AndesVL Technical Report: An Efficient Mobile-side Multimodal Large Language Model
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.11496v2](http://arxiv.org/pdf/2510.11496v2)**

> **作者:** Zhiwei Jin; Xiaohui Song; Nan Wang; Yafei Liu; Chao Li; Xin Li; Ruichen Wang; Zhihao Li; Qi Qi; Long Cheng; Dongze Hao; Quanlong Zheng; Yanhao Zhang; Haobo Ji; Jian Ma; Zhitong Zheng; Zhenyi Lin; Haolin Deng; Xin Zou; Xiaojie Yin; Ruilin Wang; Liankai Cai; Haijing Liu; Yuqing Qiu; Ke Chen; Zixian Li; Chi Xie; Huafei Li; Chenxing Li; Chuangchuang Wang; Kai Tang; Zhiguang Zhu; Kai Tang; Wenmei Gao; Rui Wang; Jun Wu; Chao Liu; Qin Xie; Chen Chen; Haonan Lu
>
> **备注:** Tech report of OPPO AndesVL Team
>
> **摘要:** In recent years, while cloud-based MLLMs such as QwenVL, InternVL, GPT-4o, Gemini, and Claude Sonnet have demonstrated outstanding performance with enormous model sizes reaching hundreds of billions of parameters, they significantly surpass the limitations in memory, power consumption, and computing capacity of edge devices such as mobile phones. This paper introduces AndesVL, a suite of mobile-side MLLMs with 0.6B to 4B parameters based on Qwen3's LLM and various visual encoders. We comprehensively outline the model architectures, training pipeline, and training data of AndesVL, which achieves first-tier performance across a wide range of open-source benchmarks, including fields such as text-rich image understanding, reasoning and math, multi-image comprehension, general VQA, hallucination mitigation, multilingual understanding, and GUI-related tasks when compared with state-of-the-art models of a similar scale. Furthermore, we introduce a 1+N LoRA architecture alongside a Quantization-Aware LoRA Fine-Tuning (QALFT) framework to facilitate efficient task adaptation and model compression during mobile-side deployment of AndesVL. Moreover, utilizing our cache eviction algorithm -- OKV -- along with customized speculative decoding and compression strategies, we achieve a 6.7x peak decoding speedup ratio, up to 30.9% memory reduction, and 1.8 bits-per-weight when deploying AndesVL-4B on MediaTek Dimensity 9500 chips. We release all models on https://huggingface.co/OPPOer.
>
---
#### [replaced 018] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02631v2](http://arxiv.org/pdf/2502.02631v2)**

> **作者:** Zechun Liu; Changsheng Zhao; Hanxian Huang; Sijia Chen; Jing Zhang; Jiawei Zhao; Scott Roy; Lisa Jin; Yunyang Xiong; Yangyang Shi; Lin Xiao; Yuandong Tian; Bilge Soran; Raghuraman Krishnamoorthi; Tijmen Blankevoort; Vikas Chandra
>
> **备注:** NeurIPS 2025. Model weights are available at https://huggingface.co/collections/facebook/mobilellm-6722be18cb86c20ebe113e95
>
> **摘要:** The optimal bit-width for achieving the best trade-off between quantized model size and accuracy has been a subject of ongoing debate. While some advocate for 4-bit quantization, others propose that 1.58-bit offers superior results. However, the lack of a cohesive framework for different bits has left such conclusions relatively tenuous. We present ParetoQ, the first unified framework that facilitates rigorous comparisons across 1-bit, 1.58-bit, 2-bit, 3-bit, and 4-bit quantization settings. Our findings reveal a notable learning transition between 2 and 3 bits: For 3-bits and above, the fine-tuned models stay close to their original pre-trained distributions, whereas for learning 2-bit networks or below, the representations change drastically. By optimizing training schemes and refining quantization functions, ParetoQ surpasses all previous methods tailored to specific bit widths. Remarkably, our ParetoQ ternary 600M-parameter model even outperforms the previous SoTA ternary 3B-parameter model in accuracy, using only one-fifth of the parameters. Extensive experimentation shows that ternary, 2-bit, and 3-bit quantization maintains comparable performance in the size-accuracy trade-off and generally exceeds 4-bit and binary quantization. Considering hardware constraints, 2-bit quantization offers promising potential for memory reduction and speedup.
>
---
#### [replaced 019] UrbanTwin: Building High-Fidelity Digital Twins for Sim2Real LiDAR Perception and Evaluation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02903v2](http://arxiv.org/pdf/2509.02903v2)**

> **作者:** Muhammad Shahbaz; Shaurya Agarwal
>
> **摘要:** LiDAR-based perception in intelligent transportation systems (ITS) relies on deep neural networks trained with large-scale labeled datasets. However, creating such datasets is expensive, time-consuming, and labor-intensive, limiting the scalability of perception systems. Sim2Real learning offers a scalable alternative, but its success depends on the simulation's fidelity to real-world environments, dynamics, and sensors. This tutorial introduces a reproducible workflow for building high-fidelity digital twins (HiFi DTs) to generate realistic synthetic datasets. We outline practical steps for modeling static geometry, road infrastructure, and dynamic traffic using open-source resources such as satellite imagery, OpenStreetMap, and sensor specifications. The resulting environments support scalable and cost-effective data generation for robust Sim2Real learning. Using this workflow, we have released three synthetic LiDAR datasets, namely UT-LUMPI, UT-V2X-Real, and UT-TUMTraf-I, which closely replicate real locations and outperform real-data-trained baselines in perception tasks. This guide enables broader adoption of HiFi DTs in ITS research and deployment.
>
---
#### [replaced 020] Image Quality Assessment for Embodied AI
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.16815v2](http://arxiv.org/pdf/2505.16815v2)**

> **作者:** Chunyi Li; Jiaohao Xiao; Jianbo Zhang; Farong Wen; Zicheng Zhang; Yuan Tian; Xiangyang Zhu; Xiaohong Liu; Zhengxue Cheng; Weisi Lin; Guangtao Zhai
>
> **摘要:** Embodied AI has developed rapidly in recent years, but it is still mainly deployed in laboratories, with various distortions in the Real-world limiting its application. Traditionally, Image Quality Assessment (IQA) methods are applied to predict human preferences for distorted images; however, there is no IQA method to assess the usability of an image in embodied tasks, namely, the perceptual quality for robots. To provide accurate and reliable quality indicators for future embodied scenarios, we first propose the topic: IQA for Embodied AI. Specifically, we (1) based on the Mertonian system and meta-cognitive theory, constructed a perception-cognition-decision-execution pipeline and defined a comprehensive subjective score collection process; (2) established the Embodied-IQA database, containing over 36k reference/distorted image pairs, with more than 5m fine-grained annotations provided by Vision Language Models/Vision Language Action-models/Real-world robots; (3) trained and validated the performance of mainstream IQA methods on Embodied-IQA, demonstrating the need to develop more accurate quality indicators for Embodied AI. We sincerely hope that through evaluation, we can promote the application of Embodied AI under complex distortions in the Real-world. Project page: https://github.com/lcysyzxdxc/EmbodiedIQA
>
---
#### [replaced 021] Optimally Deep Networks -- Adapting Model Depth to Datasets for Superior Efficiency
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.10764v2](http://arxiv.org/pdf/2510.10764v2)**

> **作者:** Shaharyar Ahmed Khan Tareen; Filza Khan Tareen
>
> **备注:** 6 pages, 3 figures, 1 table
>
> **摘要:** Deep neural networks (DNNs) have provided brilliant performance across various tasks. However, this success often comes at the cost of unnecessarily large model sizes, high computational demands, and substantial memory footprints. Typically, powerful architectures are trained at full depths but not all datasets or tasks require such high model capacity. Training very deep architectures on relatively low-complexity datasets frequently leads to wasted computation, unnecessary energy consumption, and excessive memory usage, which in turn makes deployment of models on resource-constrained devices impractical. To address this problem, we introduce Optimally Deep Networks (ODNs), which provide a balance between model depth and task complexity. Specifically, we propose a NAS like training strategy called progressive depth expansion, which begins by training deep networks at shallower depths and incrementally increases their depth as the earlier blocks converge, continuing this process until the target accuracy is reached. ODNs use only the optimal depth for the given datasets, removing redundant layers. This cuts down future training and inference costs, lowers the memory footprint, enhances computational efficiency, and facilitates deployment on edge devices. Empirical results show that the optimal depths of ResNet-18 and ResNet-34 for MNIST and SVHN, achieve up to 98.64 % and 96.44 % reduction in memory footprint, while maintaining a competitive accuracy of 99.31 % and 96.08 %, respectively.
>
---
#### [replaced 022] Visual Affordance Prediction: Survey and Reproducibility
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.05074v2](http://arxiv.org/pdf/2505.05074v2)**

> **作者:** Tommaso Apicella; Alessio Xompero; Andrea Cavallaro
>
> **备注:** 18 pages, 3 figures, 13 tables. Project website at https://apicis.github.io/aff-survey/
>
> **摘要:** Affordances are the potential actions an agent can perform on an object, as observed by a camera. Visual affordance prediction is formulated differently for tasks such as grasping detection, affordance classification, affordance segmentation, and hand pose estimation. This diversity in formulations leads to inconsistent definitions that prevent fair comparisons between methods. In this paper, we propose a unified formulation of visual affordance prediction by accounting for the complete information on the objects of interest and the interaction of the agent with the objects to accomplish a task. This unified formulation allows us to comprehensively and systematically review disparate visual affordance works, highlighting strengths and limitations of both methods and datasets. We also discuss reproducibility issues, such as the unavailability of methods implementation and experimental setups details, making benchmarks for visual affordance prediction unfair and unreliable. To favour transparency, we introduce the Affordance Sheet, a document that details the solution, datasets, and validation of a method, supporting future reproducibility and fairness in the community.
>
---
#### [replaced 023] Modular Embedding Recomposition for Incremental Learning
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.16463v2](http://arxiv.org/pdf/2508.16463v2)**

> **作者:** Aniello Panariello; Emanuele Frascaroli; Pietro Buzzega; Lorenzo Bonicelli; Angelo Porrello; Simone Calderara
>
> **备注:** Accepted to the 36th British Machine Vision Conference (BMVC 2025), Sheffield, UK
>
> **摘要:** The advent of pre-trained Vision-Language Models (VLMs) has significantly transformed Continual Learning (CL), mainly due to their zero-shot classification abilities. Such proficiency makes VLMs well-suited for real-world applications, enabling robust performance on novel unseen classes without requiring adaptation. However, fine-tuning remains essential when downstream tasks deviate significantly from the pre-training domain. Prior CL approaches primarily focus on preserving the zero-shot capabilities of VLMs during incremental fine-tuning on a downstream task. We take a step further by devising an approach that transforms preservation into enhancement of the zero-shot capabilities of VLMs. Our approach, named MoDular Embedding Recomposition (MoDER), introduces a modular framework that trains multiple textual experts, each specialized in a single seen class, and stores them in a foundational hub. At inference time, for each unseen class, we query the hub and compose the retrieved experts to synthesize a refined prototype that improves classification. We show the effectiveness of our method across two popular zero-shot incremental protocols, Class-IL and MTIL, comprising a total of 14 datasets. The codebase is available at https://github.com/aimagelab/mammoth.
>
---
#### [replaced 024] Finding Dori: Memorization in Text-to-Image Diffusion Models Is Not Local
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16880v2](http://arxiv.org/pdf/2507.16880v2)**

> **作者:** Antoni Kowalczuk; Dominik Hintersdorf; Lukas Struppek; Kristian Kersting; Adam Dziedzic; Franziska Boenisch
>
> **摘要:** Text-to-image diffusion models (DMs) have achieved remarkable success in image generation. However, concerns about data privacy and intellectual property remain due to their potential to inadvertently memorize and replicate training data. Recent mitigation efforts have focused on identifying and pruning weights responsible for triggering verbatim training data replication, based on the assumption that memorization can be localized. We challenge this assumption and demonstrate that, even after such pruning, small perturbations to the text embeddings of previously mitigated prompts can re-trigger data replication, revealing the fragility of such defenses. Our further analysis then provides multiple indications that memorization is indeed not inherently local: (1) replication triggers for memorized images are distributed throughout text embedding space; (2) embeddings yielding the same replicated image produce divergent model activations; and (3) different pruning methods identify inconsistent sets of memorization-related weights for the same image. Finally, we show that bypassing the locality assumption enables more robust mitigation through adversarial fine-tuning. These findings provide new insights into the nature of memorization in text-to-image DMs and inform the development of more reliable mitigations against DM memorization.
>
---
#### [replaced 025] BAAF: A benchmark attention adaptive framework for medical ultrasound image segmentation tasks
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.00919v3](http://arxiv.org/pdf/2310.00919v3)**

> **作者:** Gongping Chen; Lei Zhao; Xiaotao Yin; Liang Cui; Jianxun Zhang; Yu Dai; Ningning Liu
>
> **备注:** 10 pages, 11 figures
>
> **摘要:** The AI-based assisted diagnosis programs have been widely investigated on medical ultrasound images. Complex scenario of ultrasound image, in which the coupled interference of internal and external factors is severe, brings a unique challenge for localize the object region automatically and precisely in ultrasound images. In this study, we seek to propose a more general and robust Benchmark Attention Adaptive Framework (BAAF) to assist doctors segment or diagnose lesions and tissues in ultrasound images more quickly and accurately. Different from existing attention schemes, the BAAF consists of a parallel hybrid attention module (PHAM) and an adaptive calibration mechanism (ACM). Specifically, BAAF first coarsely calibrates the input features from the channel and spatial dimensions, and then adaptively selects more robust lesion or tissue characterizations from the coarse-calibrated feature maps. The design of BAAF further optimizes the "what" and "where" focus and selection problems in CNNs and seeks to improve the segmentation accuracy of lesions or tissues in medical ultrasound images. The method is evaluated on four medical ultrasound segmentation tasks, and the adequate experimental results demonstrate the remarkable performance improvement over existing state-of-the-art methods. In addition, the comparison with existing attention mechanisms also demonstrates the superiority of BAAF. This work provides the possibility for automated medical ultrasound assisted diagnosis and reduces reliance on human accuracy and precision.
>
---
#### [replaced 026] Macro-from-Micro Planning for High-Quality and Parallelized Autoregressive Long Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03334v3](http://arxiv.org/pdf/2508.03334v3)**

> **作者:** Xunzhi Xiang; Yabo Chen; Guiyu Zhang; Zhongyu Wang; Zhe Gao; Quanming Xiang; Gonghu Shang; Junqi Liu; Haibin Huang; Yang Gao; Chi Zhang; Qi Fan; Xuelong Li
>
> **摘要:** Current autoregressive diffusion models excel at video generation but are generally limited to short temporal durations. Our theoretical analysis indicates that the autoregressive modeling typically suffers from temporal drift caused by error accumulation and hinders parallelization in long video synthesis. To address these limitations, we propose a novel planning-then-populating framework centered on Macro-from-Micro Planning (MMPL) for long video generation. MMPL sketches a global storyline for the entire video through two hierarchical stages: Micro Planning and Macro Planning. Specifically, Micro Planning predicts a sparse set of future keyframes within each short video segment, offering motion and appearance priors to guide high-quality video segment generation. Macro Planning extends the in-segment keyframes planning across the entire video through an autoregressive chain of micro plans, ensuring long-term consistency across video segments. Subsequently, MMPL-based Content Populating generates all intermediate frames in parallel across segments, enabling efficient parallelization of autoregressive generation. The parallelization is further optimized by Adaptive Workload Scheduling for balanced GPU execution and accelerated autoregressive video generation. Extensive experiments confirm that our method outperforms existing long video generation models in quality and stability. Generated videos and comparison results are in our project page.
>
---
#### [replaced 027] CoRGI: Verified Chain-of-Thought Reasoning with Post-hoc Visual Grounding
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00378v3](http://arxiv.org/pdf/2508.00378v3)**

> **作者:** Shixin Yi; Lin Shang
>
> **备注:** The paper is not yet mature and needs further improvement
>
> **摘要:** Multimodal reasoning with vision-language models (VLMs) often suffers from hallucinations, as models tend to generate explanations after only a superficial inspection of the image. We present \textbf{CoRGI}(\textbf{C}hain \textbf{o}f \textbf{R}easoning with \textbf{G}rounded \textbf{I}nsights), a framework that enhances reasoning reliability through post-hoc verification of chain-of-thought outputs. Given a VLM-generated rationale, CoRGI decomposes it into step-wise statements, grounds each step in visual evidence, and filters or corrects unsupported claims before producing the final answer. Experiments on five challenging benchmark-VCR, ScienceQA, MMMU, MathVista, and HallusionBenc-demonstrate that CoRGI consistently improves both answer accuracy and explanation faithfulness across multiple VLM backbones, including Qwen-2.5VL, LLaVA-1.6, and Gemma3-12B. Beyond quantitative gains, qualitative analyses further illustrate how the verification process reduces hallucination and strengthens interpretability, suggesting that post-hoc visual grounding is a promising direction for building more trustworthy and transparent multimodal reasoning systems.
>
---
#### [replaced 028] Uncertainty-Supervised Interpretable and Robust Evidential Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.17098v2](http://arxiv.org/pdf/2509.17098v2)**

> **作者:** Yuzhu Li; An Sui; Fuping Wu; Xiahai Zhuang
>
> **摘要:** Uncertainty estimation has been widely studied in medical image segmentation as a tool to provide reliability, particularly in deep learning approaches. However, previous methods generally lack effective supervision in uncertainty estimation, leading to low interpretability and robustness of the predictions. In this work, we propose a self-supervised approach to guide the learning of uncertainty. Specifically, we introduce three principles about the relationships between the uncertainty and the image gradients around boundaries and noise. Based on these principles, two uncertainty supervision losses are designed. These losses enhance the alignment between model predictions and human interpretation. Accordingly, we introduce novel quantitative metrics for evaluating the interpretability and robustness of uncertainty. Experimental results demonstrate that compared to state-of-the-art approaches, the proposed method can achieve competitive segmentation performance and superior results in out-of-distribution (OOD) scenarios while significantly improving the interpretability and robustness of uncertainty estimation. Code is available via https://github.com/suiannaius/SURE.
>
---
#### [replaced 029] BenthiCat: An opti-acoustic dataset for advancing benthic classification and habitat mapping
- **分类: cs.CV; cs.LG; I.2.6; I.4.6; I.5.1; I.5.4**

- **链接: [http://arxiv.org/pdf/2510.04876v2](http://arxiv.org/pdf/2510.04876v2)**

> **作者:** Hayat Rajani; Valerio Franchi; Borja Martinez-Clavel Valles; Raimon Ramos; Rafael Garcia; Nuno Gracias
>
> **备注:** Article under review by IJRR
>
> **摘要:** Benthic habitat mapping is fundamental for understanding marine ecosystems, guiding conservation efforts, and supporting sustainable resource management. Yet, the scarcity of large, annotated datasets limits the development and benchmarking of machine learning models in this domain. This paper introduces a thorough multi-modal dataset, comprising about a million side-scan sonar (SSS) tiles collected along the coast of Catalonia (Spain), complemented by bathymetric maps and a set of co-registered optical images from targeted surveys using an autonomous underwater vehicle (AUV). Approximately \num{36000} of the SSS tiles have been manually annotated with segmentation masks to enable supervised fine-tuning of classification models. All the raw sensor data, together with mosaics, are also released to support further exploration and algorithm development. To address challenges in multi-sensor data fusion for AUVs, we spatially associate optical images with corresponding SSS tiles, facilitating self-supervised, cross-modal representation learning. Accompanying open-source preprocessing and annotation tools are provided to enhance accessibility and encourage research. This resource aims to establish a standardized benchmark for underwater habitat mapping, promoting advancements in autonomous seafloor classification and multi-sensor integration.
>
---
#### [replaced 030] Benchmarking foundation models for hyperspectral image classification: Application to cereal crop type mapping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.11576v2](http://arxiv.org/pdf/2510.11576v2)**

> **作者:** Walid Elbarz; Mohamed Bourriz; Hicham Hajji; Hamd Ait Abdelali; François Bourzeix
>
> **备注:** currently being reviewed for WHISPERS conference ( Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing )
>
> **摘要:** Foundation models are transforming Earth observation, but their potential for hyperspectral crop mapping remains underexplored. This study benchmarks three foundation models for cereal crop mapping using hyperspectral imagery: HyperSigma, DOFA, and Vision Transformers pre-trained on the SpectralEarth dataset (a large multitemporal hyperspectral archive). Models were fine-tuned on manually labeled data from a training region and evaluated on an independent test region. Performance was measured with overall accuracy (OA), average accuracy (AA), and F1-score. HyperSigma achieved an OA of 34.5% (+/- 1.8%), DOFA reached 62.6% (+/- 3.5%), and the SpectralEarth model achieved an OA of 93.5% (+/- 0.8%). A compact SpectralEarth variant trained from scratch achieved 91%, highlighting the importance of model architecture for strong generalization across geographic regions and sensor platforms. These results provide a systematic evaluation of foundation models for operational hyperspectral crop mapping and outline directions for future model development.
>
---
#### [replaced 031] OST-Bench: Evaluating the Capabilities of MLLMs in Online Spatio-temporal Scene Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07984v2](http://arxiv.org/pdf/2507.07984v2)**

> **作者:** Jingli Lin; Chenming Zhu; Runsen Xu; Xiaohan Mao; Xihui Liu; Tai Wang; Jiangmiao Pang
>
> **备注:** 30 pages, a benchmark designed to evaluate Online Spatio-Temporal understanding from the perspective of an agent actively exploring a scene. Project Page: https://rbler1234.github.io/OSTBench.github.io/
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have shown remarkable capabilities in integrating vision and language for complex reasoning. While most existing benchmarks evaluate models under offline settings with a fixed set of pre-recorded inputs, we introduce OST-Bench, a benchmark designed to evaluate Online Spatio-Temporal understanding from the perspective of an agent actively exploring a scene. The Online aspect emphasizes the need to process and reason over incrementally acquired observations, while the Spatio-Temporal component requires integrating current visual inputs with historical memory to support dynamic spatial reasoning. OST-Bench better reflects the challenges of real-world embodied perception. Built on an efficient data collection pipeline, OST-Bench consists of 1.4k scenes and 10k question-answer pairs collected from ScanNet, Matterport3D, and ARKitScenes. We evaluate several leading MLLMs on OST-Bench and observe that they fall short on tasks requiring complex spatio-temporal reasoning. Under the online setting, their accuracy declines as the exploration horizon extends and the memory grows. Through further experimental analysis, we identify common error patterns across models and find that both complex clue-based spatial reasoning demands and long-term memory retrieval requirements significantly drop model performance along two separate axes, highlighting the core challenges that must be addressed to improve online embodied reasoning. To foster further research and development in the field, our codes, dataset, and benchmark are available. Our project page is: https://rbler1234.github.io/OSTBench.github.io/
>
---
#### [replaced 032] EgoBrain: Synergizing Minds and Eyes For Human Action Understanding
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01353v2](http://arxiv.org/pdf/2506.01353v2)**

> **作者:** Nie Lin; Yansen Wang; Dongqi Han; Weibang Jiang; Jingyuan Li; Ryosuke Furuta; Yoichi Sato; Dongsheng Li
>
> **备注:** 22 pages, 12 figures
>
> **摘要:** The integration of brain-computer interfaces (BCIs), in particular electroencephalography (EEG), with artificial intelligence (AI) has shown tremendous promise in decoding human cognition and behavior from neural signals. In particular, the rise of multimodal AI models have brought new possibilities that have never been imagined before. Here, we present EgoBrain --the world's first large-scale, temporally aligned multimodal dataset that synchronizes egocentric vision and EEG of human brain over extended periods of time, establishing a new paradigm for human-centered behavior analysis. This dataset comprises 61 hours of synchronized 32-channel EEG recordings and first-person video from 40 participants engaged in 29 categories of daily activities. We then developed a muiltimodal learning framework to fuse EEG and vision for action understanding, validated across both cross-subject and cross-environment challenges, achieving an action recognition accuracy of 66.70%. EgoBrain paves the way for a unified framework for brain-computer interface with multiple modalities. All data, tools, and acquisition protocols are openly shared to foster open science in cognitive computing.
>
---
#### [replaced 033] Normalize Filters! Classical Wisdom for Deep Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04401v2](http://arxiv.org/pdf/2506.04401v2)**

> **作者:** Gustavo Perez; Stella X. Yu
>
> **摘要:** Classical image filters, such as those for averaging or differencing, are carefully normalized to ensure consistency, interpretability, and to avoid artifacts like intensity shifts, halos, or ringing. In contrast, convolutional filters learned end-to-end in deep networks lack such constraints. Although they may resemble wavelets and blob/edge detectors, they are not normalized in the same or any way. Consequently, when images undergo atmospheric transfer, their responses become distorted, leading to incorrect outcomes. We address this limitation by proposing filter normalization, followed by learnable scaling and shifting, akin to batch normalization. This simple yet effective modification ensures that the filters are atmosphere-equivariant, enabling co-domain symmetry. By integrating classical filtering principles into deep learning (applicable to both convolutional neural networks and convolution-dependent vision transformers), our method achieves significant improvements on artificial and natural intensity variation benchmarks. Our ResNet34 could even outperform CLIP by a large margin. Our analysis reveals that unnormalized filters degrade performance, whereas filter normalization regularizes learning, promotes diversity, and improves robustness and generalization.
>
---
#### [replaced 034] FlagEval Findings Report: A Preliminary Evaluation of Large Reasoning Models on Automatically Verifiable Textual and Visual Questions
- **分类: cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.17177v2](http://arxiv.org/pdf/2509.17177v2)**

> **作者:** Bowen Qin; Chen Yue; Fang Yin; Hui Wang; JG Yao; Jiakang Liu; Jing-Shu Zheng; Miguel Hu Chen; Richeng Xuan; Shibei Meng; Shiqi Zhou; Teng Dai; Tong-Shuai Ren; Wei Cui; Xi Yang; Xialin Du; Xiaojing Xu; Xue Sun; Xuejing Li; Yaming Liu; Yesheng Liu; Ying Liu; Yonghua Lin; Yu Zhao; Yunduo Zhang; Yuwen Luo; Zheqi He; Zhiyuan He; Zhongyuan Wang
>
> **备注:** Project homepage: https://flageval-baai.github.io/LRM-Eval/ This work will also be presented at NeurIPS 2025 Workshop on Foundations of Reasoning in Language Models (FoRLM)
>
> **摘要:** We conduct a moderate-scale contamination-free (to some extent) evaluation of current large reasoning models (LRMs) with some preliminary findings. We also release ROME, our evaluation benchmark for vision language models intended to test reasoning from visual clues. We attach links to the benchmark, evaluation data, and other updates on this website: https://flageval-baai.github.io/LRM-Eval/
>
---
#### [replaced 035] CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.10845v3](http://arxiv.org/pdf/2408.10845v3)**

> **作者:** Hidehisa Arai; Keita Miwa; Kento Sasaki; Yu Yamaguchi; Kohei Watanabe; Shunsuke Aoki; Issei Yamamoto
>
> **备注:** WACV 2025, Project Page: https://turingmotors.github.io/covla-ad/
>
> **摘要:** Autonomous driving, particularly navigating complex and unanticipated scenarios, demands sophisticated reasoning and planning capabilities. While Multi-modal Large Language Models (MLLMs) offer a promising avenue for this, their use has been largely confined to understanding complex environmental contexts or generating high-level driving commands, with few studies extending their application to end-to-end path planning. A major research bottleneck is the lack of large-scale annotated datasets encompassing vision, language, and action. To address this issue, we propose CoVLA (Comprehensive Vision-Language-Action) Dataset, an extensive dataset comprising real-world driving videos spanning more than 80 hours. This dataset leverages a novel, scalable approach based on automated data processing and a caption generation pipeline to generate accurate driving trajectories paired with detailed natural language descriptions of driving environments and maneuvers. This approach utilizes raw in-vehicle sensor data, allowing it to surpass existing datasets in scale and annotation richness. Using CoVLA, we investigate the driving capabilities of MLLMs that can handle vision, language, and action in a variety of driving scenarios. Our results illustrate the strong proficiency of our model in generating coherent language and action outputs, emphasizing the potential of Vision-Language-Action (VLA) models in the field of autonomous driving. This dataset establishes a framework for robust, interpretable, and data-driven autonomous driving systems by providing a comprehensive platform for training and evaluating VLA models, contributing to safer and more reliable self-driving vehicles. The dataset is released for academic purpose.
>
---
#### [replaced 036] Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.07214v4](http://arxiv.org/pdf/2404.07214v4)**

> **作者:** Akash Ghosh; Arkadeep Acharya; Sriparna Saha; Vinija Jain; Aman Chadha
>
> **备注:** One of the first survey on Visual Language Models
>
> **摘要:** The advent of Large Language Models (LLMs) has significantly reshaped the trajectory of the AI revolution. Nevertheless, these LLMs exhibit a notable limitation, as they are primarily adept at processing textual information. To address this constraint, researchers have endeavored to integrate visual capabilities with LLMs, resulting in the emergence of Vision-Language Models (VLMs). These advanced models are instrumental in tackling more intricate tasks such as image captioning and visual question answering. In our comprehensive survey paper, we delve into the key advancements within the realm of VLMs. Our classification organizes VLMs into three distinct categories: models dedicated to vision-language understanding, models that process multimodal inputs to generate unimodal (textual) outputs and models that both accept and produce multimodal inputs and outputs.This classification is based on their respective capabilities and functionalities in processing and generating various modalities of data.We meticulously dissect each model, offering an extensive analysis of its foundational architecture, training data sources, as well as its strengths and limitations wherever possible, providing readers with a comprehensive understanding of its essential components. We also analyzed the performance of VLMs in various benchmark datasets. By doing so, we aim to offer a nuanced understanding of the diverse landscape of VLMs. Additionally, we underscore potential avenues for future research in this dynamic domain, anticipating further breakthroughs and advancements.
>
---
#### [replaced 037] Boosting Generic Semi-Supervised Medical Image Segmentation via Diverse Teaching and Label Propagation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.08549v2](http://arxiv.org/pdf/2508.08549v2)**

> **作者:** Wei Li; Pengcheng Zhou; Linye Ma; Wenyi Zhao; Huihua Yang
>
> **备注:** Under Review
>
> **摘要:** Both limited annotation and domain shift are significant challenges frequently encountered in medical image segmentation, leading to derivative scenarios like semi-supervised medical (SSMIS), semi-supervised medical domain generalization (Semi-MDG) and unsupervised medical domain adaptation (UMDA). Conventional methods are generally tailored to specific tasks in isolation, the error accumulation hinders the effective utilization of unlabeled data and limits further improvements, resulting in suboptimal performance when these issues occur. In this paper, we aim to develop a generic framework that masters all three tasks. We found that the key to solving the problem lies in how to generate reliable pseudo labels for the unlabeled data in the presence of domain shift with labeled data and increasing the diversity of the model. To tackle this issue, we employ a Diverse Teaching and Label Propagation Network (DTLP-Net) to boosting the Generic Semi-Supervised Medical Image Segmentation. Our DTLP-Net involves a single student model and two diverse teacher models, which can generate reliable pseudo-labels for the student model. The first teacher model decouple the training process with labeled and unlabeled data, The second teacher is momentum-updated periodically, thus generating reliable yet divers pseudo-labels. To fully utilize the information within the data, we adopt inter-sample and intra-sample data augmentation to learn the global and local knowledge. In addition, to further capture the voxel-level correlations, we propose label propagation to enhance the model robust. We evaluate our proposed framework on five benchmark datasets for SSMIS, UMDA, and Semi-MDG tasks. The results showcase notable improvements compared to state-of-the-art methods across all five settings, indicating the potential of our framework to tackle more challenging SSL scenarios.
>
---
#### [replaced 038] Extremely low-bitrate Image Compression Semantically Disentangled by LMMs from a Human Perception Perspective
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.00399v4](http://arxiv.org/pdf/2503.00399v4)**

> **作者:** Juan Song; Lijie Yang; Mingtao Feng
>
> **摘要:** It remains a significant challenge to compress images at extremely low bitrate while achieving both semantic consistency and high perceptual quality. Inspired by human progressive perception mechanism, we propose a Semantically Disentangled Image Compression framework (SEDIC) in this paper. Initially, an extremely compressed reference image is obtained through a learned image encoder. Then we leverage LMMs to extract essential semantic components, including overall descriptions, object detailed description, and semantic segmentation masks. We propose a training-free Object Restoration model with Attention Guidance (ORAG) built on pre-trained ControlNet to restore object details conditioned by object-level text descriptions and semantic masks. Based on the proposed ORAG, we design a multistage semantic image decoder to progressively restore the details object by object, starting from the extremely compressed reference image, ultimately generating high-quality and high-fidelity reconstructions. Experimental results demonstrate that SEDIC significantly outperforms state-of-the-art approaches, achieving superior perceptual quality and semantic consistency at extremely low-bitrates ($\le$ 0.05 bpp).
>
---
#### [replaced 039] Robust Real-Time Endoscopic Stereo Matching under Fuzzy Tissue Boundaries
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.00731v2](http://arxiv.org/pdf/2503.00731v2)**

> **作者:** Yang Ding; Can Han; Sijia Du; Yaqi Wang; Dahong Qian
>
> **摘要:** Real-time acquisition of accurate scene depth is essential for automated robotic minimally invasive surgery. Stereo matching with binocular endoscopy can provide this depth information. However, existing stereo matching methods, designed primarily for natural images, often struggle with endoscopic images due to fuzzy tissue boundaries and typically fail to meet real-time requirements for high-resolution endoscopic image inputs. To address these challenges, we propose \textbf{RRESM}, a real-time stereo matching method tailored for endoscopic images. Our approach integrates a 3D Mamba Coordinate Attention module that enhances cost aggregation through position-sensitive attention maps and long-range spatial dependency modeling via the Mamba block, generating a robust cost volume without substantial computational overhead. Additionally, we introduce a High-Frequency Disparity Optimization module that refines disparity predictions near tissue boundaries by amplifying high-frequency details in the wavelet domain. Evaluations on the SCARED and SERV-CT datasets demonstrate state-of-the-art matching accuracy with a real-time inference speed of 42 FPS. The code is available at https://github.com/Sonne-Ding/RRESM.
>
---
#### [replaced 040] HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10177v2](http://arxiv.org/pdf/2510.10177v2)**

> **作者:** Yulin Wang; Mengting Hu; Hongli Li; Chen Luo
>
> **备注:** International Conference on Computer Vision, ICCV 2025 (Highlight) https://iccv.thecvf.com/virtual/2025/poster/338
>
> **摘要:** In pose estimation for seen objects, a prevalent pipeline involves using neural networks to predict dense 3D coordinates of the object surface on 2D images, which are then used to establish dense 2D-3D correspondences. However, current methods primarily focus on more efficient encoding techniques to improve the precision of predicted 3D coordinates on the object's front surface, overlooking the potential benefits of incorporating the back surface and interior of the object. To better utilize the full surface and interior of the object, this study predicts 3D coordinates of both the object's front and back surfaces and densely samples 3D coordinates between them. This process creates ultra-dense 2D-3D correspondences, effectively enhancing pose estimation accuracy based on the Perspective-n-Point (PnP) algorithm. Additionally, we propose Hierarchical Continuous Coordinate Encoding (HCCE) to provide a more accurate and efficient representation of front and back surface coordinates. Experimental results show that, compared to existing state-of-the-art (SOTA) methods on the BOP website, the proposed approach outperforms across seven classic BOP core datasets. Code is available at https://github.com/WangYuLin-SEU/HCCEPose.
>
---
#### [replaced 041] Denoised Diffusion for Object-Focused Image Augmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.08955v2](http://arxiv.org/pdf/2510.08955v2)**

> **作者:** Nisha Pillai; Aditi Virupakshaiah; Harrison W. Smith; Amanda J. Ashworth; Prasanna Gowda; Phillip R. Owens; Adam R. Rivers; Bindu Nanduri; Mahalingam Ramkumar
>
> **摘要:** Modern agricultural operations increasingly rely on integrated monitoring systems that combine multiple data sources for farm optimization. Aerial drone-based animal health monitoring serves as a key component but faces limited data availability, compounded by scene-specific issues such as small, occluded, or partially visible animals. Transfer learning approaches often fail to address this limitation due to the unavailability of large datasets that reflect specific farm conditions, including variations in animal breeds, environments, and behaviors. Therefore, there is a need for developing a problem-specific, animal-focused data augmentation strategy tailored to these unique challenges. To address this gap, we propose an object-focused data augmentation framework designed explicitly for animal health monitoring in constrained data settings. Our approach segments animals from backgrounds and augments them through transformations and diffusion-based synthesis to create realistic, diverse scenes that enhance animal detection and monitoring performance. Our initial experiments demonstrate that our augmented dataset yields superior performance compared to our baseline models on the animal detection task. By generating domain-specific data, our method empowers real-time animal health monitoring solutions even in data-scarce scenarios, bridging the gap between limited data and practical applicability.
>
---
#### [replaced 042] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15877v2](http://arxiv.org/pdf/2505.15877v2)**

> **作者:** Siting Li; Xiang Gao; Simon Shaolei Du
>
> **备注:** NeurIPS 2025; 27 pages, 6 figures
>
> **摘要:** While an image is worth more than a thousand words, only a few provide crucial information for a given task and thus should be focused on. In light of this, ideal text-to-image (T2I) retrievers should prioritize specific visual attributes relevant to queries. To evaluate current retrievers on handling attribute-focused queries, we build COCO-Facet, a COCO-based benchmark with 9,112 queries about diverse attributes of interest. We find that CLIP-like retrievers, which are widely adopted due to their efficiency and zero-shot ability, have poor and imbalanced performance, possibly because their image embeddings focus on global semantics and subjects while leaving out other details. Notably, we reveal that even recent Multimodal Large Language Model (MLLM)-based, stronger retrievers with a larger output dimension struggle with this limitation. Hence, we hypothesize that retrieving with general image embeddings is suboptimal for performing such queries. As a solution, we propose to use promptable image embeddings enabled by these multimodal retrievers, which boost performance by highlighting required attributes. Our pipeline for deriving such embeddings generalizes across query types, image pools, and base retriever architectures. To enhance real-world applicability, we offer two acceleration strategies: Pre-processing promptable embeddings and using linear approximations. We show that the former yields a 15% improvement in Recall@5 when prompts are predefined, while the latter achieves an 8% improvement when prompts are only available during inference.
>
---
#### [replaced 043] GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13731v3](http://arxiv.org/pdf/2505.13731v3)**

> **作者:** Pengyue Jia; Seongheon Park; Song Gao; Xiangyu Zhao; Sharon Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Worldwide image geolocalization-the task of predicting GPS coordinates from images taken anywhere on Earth-poses a fundamental challenge due to the vast diversity in visual content across regions. While recent approaches adopt a two-stage pipeline of retrieving candidates and selecting the best match, they typically rely on simplistic similarity heuristics and point-wise supervision, failing to model spatial relationships among candidates. In this paper, we propose GeoRanker, a distance-aware ranking framework that leverages large vision-language models to jointly encode query-candidate interactions and predict geographic proximity. In addition, we introduce a multi-order distance loss that ranks both absolute and relative distances, enabling the model to reason over structured spatial relationships. To support this, we curate GeoRanking, the first dataset explicitly designed for geographic ranking tasks with multimodal candidate information. GeoRanker achieves state-of-the-art results on two well-established benchmarks (IM2GPS3K and YFCC4K), significantly outperforming current best methods.
>
---
#### [replaced 044] Hands-Free Heritage: Automated 3D Scanning for Cultural Heritage Digitization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.04781v2](http://arxiv.org/pdf/2510.04781v2)**

> **作者:** Javed Ahmad; Federico Dassiè; Selene Frascella; Gabriele Marchello; Ferdinando Cannella; Arianna Traviglia
>
> **备注:** The author has decided to withdraw this version to verify and update authorization details for certain image materials obtained from a collaborating institution. The issue is administrative and does not affect the technical content of the work. A revised version will be submitted once the verification process is complete
>
> **摘要:** High-fidelity 3D scanning is essential for preserving cultural heritage artefacts, supporting documentation, analysis, and long-term conservation. However, conventional methods typically require specialized expertise and manual intervention to maintain optimal scanning conditions and coverage. We present an automated two-robot scanning system that eliminates the need for handheld or semi-automatic workflows by combining coordinated robotic manipulation with high-resolution 3D scanning. Our system parameterizes the scanning space into distinct regions, enabling coordinated motion planning between a scanner-equipped robot and a tray-handling robot. Optimized trajectory planning and waypoint distribution ensure comprehensive surface coverage, minimize occlusions, and balance reconstruction accuracy with system efficiency. Experimental results show that our approach achieves significantly lower Chamfer Distance and higher F-score compared to baseline methods, offering superior geometric accuracy, improved digitization efficiency, and reduced reliance on expert operators.
>
---
#### [replaced 045] Pathology-CoT: Learning Visual Chain-of-Thought Agent from Expert Whole Slide Image Diagnosis Behavior
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.04587v2](http://arxiv.org/pdf/2510.04587v2)**

> **作者:** Sheng Wang; Ruiming Wu; Charles Herndon; Yihang Liu; Shunsuke Koga; Jeanne Shen; Zhi Huang
>
> **摘要:** Diagnosing a whole-slide image is an interactive, multi-stage process of changing magnification and moving between fields. Although recent pathology foundation models demonstrated superior performances, practical agentic systems that decide what field to examine next, adjust magnification, and deliver explainable diagnoses are still lacking. Such limitation is largely bottlenecked by data: scalable, clinically aligned supervision of expert viewing behavior that is tacit and experience-based, not documented in textbooks or internet, and therefore absent from LLM training. Here we introduce a framework designed to address this challenge through three key breakthroughs. First, the AI Session Recorder seamlessly integrates with standard whole-slide image viewers to unobtrusively record routine navigation and convert the viewer logs into standardized behavioral commands and bounding boxes. Second, a lightweight human-in-the-loop review turns AI-drafted rationales for behavioral commands into the Pathology-CoT dataset, a form of paired "where to look" and "why it matters", enabling six-fold faster labeling compared to manual constructing such Chain-of-Thought dataset. Using this behavioral data, we build Pathology-o3, a two-stage agent that first proposes important ROIs and then performs behavior-guided reasoning. On the gastrointestinal lymph-node metastasis detection task, our method achieved 100 recall on the internal validation from Stanford Medicine and 97.6 recall on an independent external validation from Sweden, exceeding the state-of-the-art OpenAI o3 model and generalizing across backbones. To our knowledge, Pathology-CoT constitutes one of the first behavior-grounded agentic systems in pathology. Turning everyday viewer logs into scalable, expert-validated supervision, our framework makes agentic pathology practical and establishes a path to human-aligned, upgradeable clinical AI.
>
---
#### [replaced 046] Learning Adaptive and Temporally Causal Video Tokenization in a 1D Latent Space
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17011v2](http://arxiv.org/pdf/2505.17011v2)**

> **作者:** Yan Li; Changyao Tian; Renqiu Xia; Ning Liao; Weiwei Guo; Junchi Yan; Hongsheng Li; Jifeng Dai; Hao Li; Xue Yang
>
> **备注:** Code: https://github.com/VisionXLab/AdapTok
>
> **摘要:** We propose AdapTok, an adaptive temporal causal video tokenizer that can flexibly allocate tokens for different frames based on video content. AdapTok is equipped with a block-wise masking strategy that randomly drops tail tokens of each block during training, and a block causal scorer to predict the reconstruction quality of video frames using different numbers of tokens. During inference, an adaptive token allocation strategy based on integer linear programming is further proposed to adjust token usage given predicted scores. Such design allows for sample-wise, content-aware, and temporally dynamic token allocation under a controllable overall budget. Extensive experiments for video reconstruction and generation on UCF-101 and Kinetics-600 demonstrate the effectiveness of our approach. Without additional image data, AdapTok consistently improves reconstruction quality and generation performance under different token budgets, allowing for more scalable and token-efficient generative video modeling.
>
---
#### [replaced 047] FOCUS on Contamination: A Geospatial Deep Learning Framework with a Noise-Aware Loss for Surface Water PFAS Prediction
- **分类: cs.CV; cs.AI; cs.CY; cs.LG; I.2.1; I.2.10; I.4.6; I.4.9; I.4.10; J.2**

- **链接: [http://arxiv.org/pdf/2502.14894v3](http://arxiv.org/pdf/2502.14894v3)**

> **作者:** Jowaria Khan; Alexa Friedman; Sydney Evans; Rachel Klein; Runzi Wang; Katherine E. Manz; Kaley Beins; David Q. Andrews; Elizabeth Bondi-Kelly
>
> **摘要:** Per- and polyfluoroalkyl substances (PFAS), chemicals found in products like non-stick cookware, are unfortunately persistent environmental pollutants with severe health risks. Accurately mapping PFAS contamination is crucial for guiding targeted remediation efforts and protecting public and environmental health, yet detection across large regions remains challenging due to the cost of testing and the difficulty of simulating their spread. In this work, we introduce FOCUS, a geospatial deep learning framework with a label noise-aware loss function, to predict PFAS contamination in surface water over large regions. By integrating hydrological flow data, land cover information, and proximity to known PFAS sources, our approach leverages both spatial and environmental context to improve prediction accuracy. We evaluate the performance of our approach through extensive ablation studies, robustness analysis, real-world validation, and comparative analyses against baselines like sparse segmentation, as well as existing scientific methods, including Kriging and pollutant transport simulations. Results and expert feedback highlight our framework's potential for scalable PFAS monitoring.
>
---
#### [replaced 048] How to Train Your Metamorphic Deep Neural Network
- **分类: cs.NE; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05510v2](http://arxiv.org/pdf/2505.05510v2)**

> **作者:** Thomas Sommariva; Simone Calderara; Angelo Porrello
>
> **备注:** Accepted with an Honorable Mention Award at ICIAP 2025 (Rome, Italy). 14 pages, 7 figures
>
> **摘要:** Neural Metamorphosis (NeuMeta) is a recent paradigm for generating neural networks of varying width and depth. Based on Implicit Neural Representation (INR), NeuMeta learns a continuous weight manifold, enabling the direct generation of compressed models, including those with configurations not seen during training. While promising, the original formulation of NeuMeta proves effective only for the final layers of the undelying model, limiting its broader applicability. In this work, we propose a training algorithm that extends the capabilities of NeuMeta to enable full-network metamorphosis with minimal accuracy degradation. Our approach follows a structured recipe comprising block-wise incremental training, INR initialization, and strategies for replacing batch normalization. The resulting metamorphic networks maintain competitive accuracy across a wide range of compression ratios, offering a scalable solution for adaptable and efficient deployment of deep models. The code is available at: https://github.com/TSommariva/HTTY_NeuMeta.
>
---
#### [replaced 049] VR-Thinker: Boosting Video Reward Models through Thinking-with-Image Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.10518v2](http://arxiv.org/pdf/2510.10518v2)**

> **作者:** Qunzhong Wang; Jie Liu; Jiajun Liang; Yilei Jiang; Yuanxing Zhang; Jinyuan Chen; Yaozhi Zheng; Xintao Wang; Pengfei Wan; Xiangyu Yue; Jiaheng Liu
>
> **摘要:** Recent advancements in multimodal reward models (RMs) have substantially improved post-training for visual generative models. However, current RMs face inherent limitations: (1) visual inputs consume large context budgets, forcing fewer frames and causing loss of fine-grained details; and (2) all visual information is packed into the initial prompt, exacerbating hallucination and forgetting during chain-of-thought reasoning. To overcome these issues, we introduce VideoReward Thinker (VR-Thinker), a thinking-with-image framework that equips the RM with visual reasoning operations (e.g., select frame) and a configurable visual memory window. This allows the RM to actively acquire and update visual evidence within context limits, improving reasoning fidelity and reliability. We activate visual reasoning via a reinforcement fine-tuning pipeline: (i) Cold Start with curated visual chain-of-thought data to distill basic reasoning skills and operation formatting; (ii) select samples whose per-dimension and overall judgments are all correct, then conduct Rejection sampling Fine-Tuning on these high-quality traces to further enhance reasoning; and (iii) apply Group Relative Policy Optimization (GRPO) to strengthen reasoning. Our approach delivers state-of-the-art accuracy among open-source models on video preference benchmarks, especially for longer videos: a 7B VR-Thinker achieves 80.5% on VideoGen Reward, 82.3% on GenAI-Bench, and 75.6% on MJ-Bench-Video. These results validate the effectiveness and promise of thinking-with-image multimodal reward modeling.
>
---
#### [replaced 050] OmniLens: Towards Universal Lens Aberration Correction via LensLib-to-Specific Domain Adaptation
- **分类: physics.optics; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2409.05809v2](http://arxiv.org/pdf/2409.05809v2)**

> **作者:** Qi Jiang; Yao Gao; Shaohua Gao; Zhonghua Yi; Xiaolong Qian; Hao Shi; Kailun Yang; Lei Sun; Kaiwei Wang
>
> **备注:** The code and data will be available at https://github.com/zju-jiangqi/OmniLens
>
> **摘要:** Emerging universal Computational Aberration Correction (CAC) paradigms provide an inspiring solution to light-weight and high-quality imaging with a universal model trained on a lens library (LensLib) to address arbitrary lens aberrations blindly. However, the limited coverage of existing LensLibs leads to poor generalization of the trained models to unseen lenses, whose fine-tuning pipeline is also confined to the lens-descriptions-known case. In this work, we introduce OmniLens, a flexible solution to universal CAC via (i) establishing a convincing LensLib with comprehensive coverage for pre-training a robust base model, and (ii) adapting the model to any specific lens designs with unknown lens descriptions via fast LensLib-to-specific domain adaptation. To achieve these, an Evolution-based Automatic Optical Design (EAOD) pipeline is proposed to generate a rich variety of lens samples with realistic aberration behaviors. Then, we design an unsupervised regularization term for efficient domain adaptation on a few easily accessible real-captured images based on the statistical observation of dark channel priors in degradation induced by lens aberrations. Extensive experiments demonstrate that the LensLib generated by EAOD effectively develops a universal CAC model with strong generalization capabilities, which can also improve the non-blind lens-specific methods by 0.35-1.81dB in PSNR. Additionally, the proposed domain adaptation method significantly improves the base model, especially in severe aberration cases (at most 2.59dB in PSNR). The code and data will be available at https://github.com/zju-jiangqi/OmniLens.
>
---
#### [replaced 051] mmWave Radar-Based Non-Line-of-Sight Pedestrian Localization at T-Junctions Utilizing Road Layout Extraction via Camera
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.02348v2](http://arxiv.org/pdf/2508.02348v2)**

> **作者:** Byeonggyu Park; Hee-Yeun Kim; Byonghyok Choi; Hansang Cho; Byungkwan Kim; Soomok Lee; Mingu Jeon; Seong-Woo Kim
>
> **摘要:** Pedestrians Localization in Non-Line-of-Sight (NLoS) regions within urban environments poses a significant challenge for autonomous driving systems. While mmWave radar has demonstrated potential for detecting objects in such scenarios, the 2D radar point cloud (PCD) data is susceptible to distortions caused by multipath reflections, making accurate spatial inference difficult. Additionally, although camera images provide high-resolution visual information, they lack depth perception and cannot directly observe objects in NLoS regions. In this paper, we propose a novel framework that interprets radar PCD through road layout inferred from camera for localization of NLoS pedestrians. The proposed method leverages visual information from the camera to interpret 2D radar PCD, enabling spatial scene reconstruction. The effectiveness of the proposed approach is validated through experiments conducted using a radar-camera system mounted on a real vehicle. The localization performance is evaluated using a dataset collected in outdoor NLoS driving environments, demonstrating the practical applicability of the method.
>
---
#### [replaced 052] Towards Robust and Realible Multimodal Misinformation Recognition with Incomplete Modality
- **分类: cs.MM; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.05839v3](http://arxiv.org/pdf/2510.05839v3)**

> **作者:** Hengyang Zhou; Yiwei Wei; Jian Yang; Zhenyu Zhang
>
> **摘要:** Multimodal Misinformation Recognition has become an urgent task with the emergence of huge multimodal fake content on social media platforms. Previous studies mainly focus on complex feature extraction and fusion to learn discriminative information from multimodal content. However, in real-world applications, multimedia news may naturally lose some information during dissemination, resulting in modality incompleteness, which is detrimental to the generalization and robustness of existing models. To this end, we propose a novel generic and robust multimodal fusion strategy, termed Multi-expert Modality-incomplete Learning Network (MMLNet), which is simple yet effective. It consists of three key steps: (1) Multi-Expert Collaborative Reasoning to compensate for missing modalities by dynamically leveraging complementary information through multiple experts. (2) Incomplete Modality Adapters compensates for the missing information by leveraging the new feature distribution. (3) Modality Missing Learning leveraging an label-aware adaptive weighting strategy to learn a robust representation with contrastive learning. We evaluate MMLNet on three real-world benchmarks across two languages, demonstrating superior performance compared to state-of-the-art methods while maintaining relative simplicity. By ensuring the accuracy of misinformation recognition in incomplete modality scenarios caused by information propagation, MMLNet effectively curbs the spread of malicious misinformation. Code is publicly available at https://github.com/zhyhome/MMLNet.
>
---
#### [replaced 053] Massive Activations are the Key to Local Detail Synthesis in Diffusion Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.11538v2](http://arxiv.org/pdf/2510.11538v2)**

> **作者:** Chaofan Gan; Zicheng Zhao; Yuanpeng Tu; Xi Chen; Ziran Qin; Tieyuan Chen; Mehrtash Harandi; Weiyao Lin
>
> **摘要:** Diffusion Transformers (DiTs) have recently emerged as a powerful backbone for visual generation. Recent observations reveal \emph{Massive Activations} (MAs) in their internal feature maps, yet their function remains poorly understood. In this work, we systematically investigate these activations to elucidate their role in visual generation. We found that these massive activations occur across all spatial tokens, and their distribution is modulated by the input timestep embeddings. Importantly, our investigations further demonstrate that these massive activations play a key role in local detail synthesis, while having minimal impact on the overall semantic content of output. Building on these insights, we propose \textbf{D}etail \textbf{G}uidance (\textbf{DG}), a MAs-driven, training-free self-guidance strategy to explicitly enhance local detail fidelity for DiTs. Specifically, DG constructs a degraded ``detail-deficient'' model by disrupting MAs and leverages it to guide the original network toward higher-quality detail synthesis. Our DG can seamlessly integrate with Classifier-Free Guidance (CFG), enabling further refinements of fine-grained details. Extensive experiments demonstrate that our DG consistently improves fine-grained detail quality across various pre-trained DiTs (\eg, SD3, SD3.5, and Flux).
>
---
#### [replaced 054] SPADE: Spatial Transcriptomics and Pathology Alignment Using a Mixture of Data Experts for an Expressive Latent Space
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.21857v2](http://arxiv.org/pdf/2506.21857v2)**

> **作者:** Ekaterina Redekop; Mara Pleasure; Zichen Wang; Kimberly Flores; Anthony Sisk; William Speier; Corey W. Arnold
>
> **摘要:** The rapid growth of digital pathology and advances in self-supervised deep learning have enabled the development of foundational models for various pathology tasks across diverse diseases. While multimodal approaches integrating diverse data sources have emerged, a critical gap remains in the comprehensive integration of whole-slide images (WSIs) with spatial transcriptomics (ST), which is crucial for capturing critical molecular heterogeneity beyond standard hematoxylin & eosin (H&E) staining. We introduce SPADE, a foundation model that integrates histopathology with ST data to guide image representation learning within a unified framework, in effect creating an ST-informed latent space. SPADE leverages a mixture-of-data experts technique, where experts are created via two-stage imaging feature-space clustering using contrastive learning to learn representations of co-registered WSI patches and gene expression profiles. Pre-trained on the comprehensive HEST-1k dataset, SPADE is evaluated on 20 downstream tasks, demonstrating significantly superior few-shot performance compared to baseline models, highlighting the benefits of integrating morphological and molecular information into one latent space. Code and pretrained weights are available at https://github.com/uclabair/SPADE.
>
---
#### [replaced 055] Funny-Valen-Tine: Planning Solution Distribution Enhances Machine Abstract Reasoning Ability
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.02688v3](http://arxiv.org/pdf/2407.02688v3)**

> **作者:** Ruizhuo Song; Beiming Yuan
>
> **备注:** 14 pages, 20 figures, 3 tables
>
> **摘要:** Visual abstract reasoning is core to image processing. We present Valen, a unified probability-highlighting baseline that excels on both RPM (progression) and Bongard-Logo (clustering) tasks. Analysing its internals, we find solvers implicitly treat each task as a distribution where primary samples fit and auxiliaries do not; hence the learning target is jointly shaped by both sets, not by correct solutions alone. To close the gap we first introduce Tine, an adversarial adapter that nudges Valen toward correct-solution density, but adversarial training is unstable. We therefore replace it with Funny, a fast Gaussian-mixture model that directly estimates the correct-solution density without adversarial games, and extend the same paradigm to SBR for progressive-pattern planning. Extensive experiments show explicit distribution planning is the key to stronger, interpretable abstract reasoning. Codes are available in: https://github.com/Yuanbeiming/Funny-Valen-Tine-Planning-Solution-Distribution-Enhances-Machine-Abstract-Reasoning-Ability
>
---
#### [replaced 056] EvolveNav: Empowering LLM-Based Vision-Language Navigation via Self-Improving Embodied Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01551v3](http://arxiv.org/pdf/2506.01551v3)**

> **作者:** Bingqian Lin; Yunshuang Nie; Khun Loun Zai; Ziming Wei; Mingfei Han; Rongtao Xu; Minzhe Niu; Jianhua Han; Hanwang Zhang; Liang Lin; Bokui Chen; Cewu Lu; Xiaodan Liang
>
> **摘要:** Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for enhancing vision-language navigation (VLN) performance, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches predominantly adopt straightforward input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. To address these issues, we propose EvolveNav, a novel sElf-improving embodied reasoning paradigm that realizes adaptable and generalizable navigational reasoning for boosting LLM-based vision-language Navigation. Specifically, EvolveNav involves a two-stage training process: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with curated formalized CoT labels to first activate the model's navigational reasoning capabilities, and simultaneously increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also designed to encourage the model to learn correct reasoning patterns by contrasting with wrong ones. Experimental results under both task-specific and cross-task training paradigms demonstrate the consistent superiority of EvolveNav over previous LLM-based VLN approaches on various popular benchmarks, including R2R, REVERIE, CVDN, and SOON. Code is available at https://github.com/expectorlin/EvolveNav.
>
---
#### [replaced 057] StegOT: Trade-offs in Steganography via Optimal Transport
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.11178v2](http://arxiv.org/pdf/2509.11178v2)**

> **作者:** Chengde Lin; Xuezhu Gong; Shuxue Ding; Mingzhe Yang; Xijun Lu; Chengjun Mo
>
> **备注:** Accepted by IEEE International Conference on Multimedia and Expo (ICME 2025)
>
> **摘要:** Image hiding is often referred to as steganography, which aims to hide a secret image in a cover image of the same resolution. Many steganography models are based on genera-tive adversarial networks (GANs) and variational autoencoders (VAEs). However, most existing models suffer from mode collapse. Mode collapse will lead to an information imbalance between the cover and secret images in the stego image and further affect the subsequent extraction. To address these challenges, this paper proposes StegOT, an autoencoder-based steganography model incorporating optimal transport theory. We designed the multiple channel optimal transport (MCOT) module to transform the feature distribution, which exhibits multiple peaks, into a single peak to achieve the trade-off of information. Experiments demonstrate that we not only achieve a trade-off between the cover and secret images but also enhance the quality of both the stego and recovery images. The source code will be released on https://github.com/Rss1124/StegOT.
>
---
#### [replaced 058] IWR-Bench: Can LVLMs reconstruct interactive webpage from a user interaction video?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.24709v2](http://arxiv.org/pdf/2509.24709v2)**

> **作者:** Yang Chen; Minghao Liu; Yufan Shen; Yunwen Li; Tianyuan Huang; Xinyu Fang; Tianyu Zheng; Wenxuan Huang; Cheng Yang; Daocheng Fu; Jianbiao Mei; Rong Wu; Yunfei Zhao; Licheng Wen; Xuemeng Yang; Song Mao; Qunshu Lin; Zhi Yu; Yongliang Shen; Yu Qiao; Botian Shi
>
> **摘要:** The webpage-to-code task requires models to understand visual representations of webpages and generate corresponding code. However, existing benchmarks primarily focus on static screenshot-to-code tasks, thereby overlooking the dynamic interactions fundamental to real-world web applications. To address this limitation, this paper introduces IWR-Bench, a novel benchmark for evaluating the capabilities of Large Vision-Language Models (LVLMs) in interactive webpage reconstruction from video. IWR-Bench comprises 113 meticulously curated tasks from 100 real-world websites, with 1,001 actions and featuring diverse interaction complexities (e.g., web games), visual styles, and domains. Aligning with standard web development practices, each task includes not only user interaction videos but also all crawled static assets (e.g., images, videos). This benchmark evaluates models on two fundamental challenges: comprehensive multi-modal reasoning to infer interaction logic from video and assets, and advanced code generation to translate this logic into functional code. An agent-as-a-judge framework with a comprehensive metric system automatically assesses the functional correctness and visual fidelity of generated webpages. Extensive experiments on 28 LVLMs reveal a significant challenge: the best model achieves an overall score of only 36.35%, as functional correctness (24.39% IFS) lags significantly behind visual fidelity (64.25% VFS). These results highlight critical limitations in current models' ability to reason about temporal dynamics and synthesize event-driven logic, establishing IWR-Bench as a challenging frontier for vision-language research. The benchmark and evaluation code will be made publicly available at https://github.com/L-O-I/IWR-Bench.
>
---
#### [replaced 059] NinA: Normalizing Flows in Action. Training VLA Models with Normalizing Flows
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.16845v2](http://arxiv.org/pdf/2508.16845v2)**

> **作者:** Denis Tarasov; Alexander Nikulin; Ilya Zisman; Albina Klepach; Nikita Lyubaykin; Andrei Polubarov; Alexander Derevyagin; Vladislav Kurenkov
>
> **备注:** https://github.com/dunnolab/NinA/
>
> **摘要:** Recent advances in Vision-Language-Action (VLA) models have established a two-component architecture, where a pre-trained Vision-Language Model (VLM) encodes visual observations and task descriptions, and an action decoder maps these representations to continuous actions. Diffusion models have been widely adopted as action decoders due to their ability to model complex, multimodal action distributions. However, they require multiple iterative denoising steps at inference time or downstream techniques to speed up sampling, limiting their practicality in real-world settings where high-frequency control is crucial. In this work, we present NinA (Normalizing Flows in Action), a fast and expressive alternative to diffusion-based decoders for VLAs. NinA replaces the diffusion action decoder with a Normalizing Flow (NF) that enables one-shot sampling through an invertible transformation, significantly reducing inference time. We integrate NinA into the FLOWER VLA architecture and fine-tune on the LIBERO benchmark. Our experiments show that NinA matches the performance of its diffusion-based counterpart under the same training regime, while achieving substantially faster inference. These results suggest that NinA offers a promising path toward efficient, high-frequency VLA control without compromising performance.
>
---
#### [replaced 060] Levarging Learning Bias for Noisy Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.07441v2](http://arxiv.org/pdf/2508.07441v2)**

> **作者:** Yuxin Zhang; Yunkang Cao; Yuqi Cheng; Yihan Sun; Weiming Shen
>
> **摘要:** This paper addresses the challenge of fully unsupervised image anomaly detection (FUIAD), where training data may contain unlabeled anomalies. Conventional methods assume anomaly-free training data, but real-world contamination leads models to absorb anomalies as normal, degrading detection performance. To mitigate this, we propose a two-stage framework that systematically exploits inherent learning bias in models. The learning bias stems from: (1) the statistical dominance of normal samples, driving models to prioritize learning stable normal patterns over sparse anomalies, and (2) feature-space divergence, where normal data exhibit high intra-class consistency while anomalies display high diversity, leading to unstable model responses. Leveraging the learning bias, stage 1 partitions the training set into subsets, trains sub-models, and aggregates cross-model anomaly scores to filter a purified dataset. Stage 2 trains the final detector on this dataset. Experiments on the Real-IAD benchmark demonstrate superior anomaly detection and localization performance under different noise conditions. Ablation studies further validate the framework's contamination resilience, emphasizing the critical role of learning bias exploitation. The model-agnostic design ensures compatibility with diverse unsupervised backbones, offering a practical solution for real-world scenarios with imperfect training data. Code is available at https://github.com/hustzhangyuxin/LLBNAD.
>
---
#### [replaced 061] Reframing Image Difference Captioning with BLIP2IDC and Synthetic Augmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.15939v2](http://arxiv.org/pdf/2412.15939v2)**

> **作者:** Gautier Evennou; Antoine Chaffin; Vivien Chappelier; Ewa Kijak
>
> **备注:** This paper has been accepted for the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025; Code released at https://github.com/gautierevn/BLIP2IDC
>
> **摘要:** The rise of the generative models quality during the past years enabled the generation of edited variations of images at an important scale. To counter the harmful effects of such technology, the Image Difference Captioning (IDC) task aims to describe the differences between two images. While this task is successfully handled for simple 3D rendered images, it struggles on real-world images. The reason is twofold: the training data-scarcity, and the difficulty to capture fine-grained differences between complex images. To address those issues, we propose in this paper a simple yet effective framework to both adapt existing image captioning models to the IDC task and augment IDC datasets. We introduce BLIP2IDC, an adaptation of BLIP2 to the IDC task at low computational cost, and show it outperforms two-streams approaches by a significant margin on real-world IDC datasets. We also propose to use synthetic augmentation to improve the performance of IDC models in an agnostic fashion. We show that our synthetic augmentation strategy provides high quality data, leading to a challenging new dataset well-suited for IDC named Syned1.
>
---
#### [replaced 062] Probabilistic Temporal Masked Attention for Cross-view Online Action Detection
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.17025v2](http://arxiv.org/pdf/2508.17025v2)**

> **作者:** Liping Xie; Yang Tan; Shicheng Jing; Huimin Lu; Kanjian Zhang
>
> **备注:** 12 pages, 6 figures, accepted at IEEE Transactions on Multimedia (TMM), in press
>
> **摘要:** As a critical task in video sequence classification within computer vision, Online Action Detection (OAD) has garnered significant attention. The sensitivity of mainstream OAD models to varying video viewpoints often hampers their generalization when confronted with unseen sources. To address this limitation, we propose a novel Probabilistic Temporal Masked Attention (PTMA) model, which leverages probabilistic modeling to derive latent compressed representations of video frames in a cross-view setting. The PTMA model incorporates a GRU-based temporal masked attention (TMA) cell, which leverages these representations to effectively query the input video sequence, thereby enhancing information interaction and facilitating autoregressive frame-level video analysis. Additionally, multi-view information can be integrated into the probabilistic modeling to facilitate the extraction of view-invariant features. Experiments conducted under three evaluation protocols: cross-subject (cs), cross-view (cv), and cross-subject-view (csv) show that PTMA achieves state-of-the-art performance on the DAHLIA, IKEA ASM, and Breakfast datasets.
>
---
#### [replaced 063] AsynFusion: Towards Asynchronous Latent Consistency Models for Decoupled Whole-Body Audio-Driven Avatars
- **分类: cs.SD; cs.AI; cs.CV; cs.GR; eess.AS; 68T10**

- **链接: [http://arxiv.org/pdf/2505.15058v2](http://arxiv.org/pdf/2505.15058v2)**

> **作者:** Tianbao Zhang; Jian Zhao; Yuer Li; Zheng Zhu; Ping Hu; Zhaoxin Fan; Wenjun Wu; Xuelong Li
>
> **备注:** 15pages, conference
>
> **摘要:** Whole-body audio-driven avatar pose and expression generation is a critical task for creating lifelike digital humans and enhancing the capabilities of interactive virtual agents, with wide-ranging applications in virtual reality, digital entertainment, and remote communication. Existing approaches often generate audio-driven facial expressions and gestures independently, which introduces a significant limitation: the lack of seamless coordination between facial and gestural elements, resulting in less natural and cohesive animations. To address this limitation, we propose AsynFusion, a novel framework that leverages diffusion transformers to achieve harmonious expression and gesture synthesis. The proposed method is built upon a dual-branch DiT architecture, which enables the parallel generation of facial expressions and gestures. Within the model, we introduce a Cooperative Synchronization Module to facilitate bidirectional feature interaction between the two modalities, and an Asynchronous LCM Sampling strategy to reduce computational overhead while maintaining high-quality outputs. Extensive experiments demonstrate that AsynFusion achieves state-of-the-art performance in generating real-time, synchronized whole-body animations, consistently outperforming existing methods in both quantitative and qualitative evaluations.
>
---
#### [replaced 064] Constructing a Real-World Benchmark for Early Wildfire Detection with the New PYRONEAR-2025 Dataset
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.05349v3](http://arxiv.org/pdf/2402.05349v3)**

> **作者:** Mateo Lostanlen; Nicolas Isla; Jose Guillen; Renzo Zanca; Felix Veith; Cristian Buc; Valentin Barriere
>
> **备注:** Preprint of ongoing work
>
> **摘要:** Early wildfire detection (EWD) is of the utmost importance to enable rapid response efforts, and thus minimize the negative impacts of wildfire spreads. To this end, we present PYRONEAR-2025, a new dataset composed of both images and videos, allowing for the training and evaluation of smoke plume detection models, including sequential models. The data is sourced from: (i) web-scraped videos of wildfires from public networks of cameras for wildfire detection in-the-wild, (ii) videos from our in-house network of cameras, and (iii) a small portion of synthetic and real images. This dataset includes around 150,000 manual annotations on 50,000 images, covering 640 wildfires, PYRONEAR-2025 surpasses existing datasets in size and diversity. It includes data from France, Spain, Chile and the United States. Finally, it is composed of both images and videos, allowing for the training and evaluation of smoke plume detection models, including sequential models. We ran cross-dataset experiments using a lightweight state-of-the-art object detection model, as the ones used in-real-life, and found out the proposed dataset is particularly challenging, with F1 score of around 70\%, but more stable than existing datasets. Finally, its use in concordance with other public datasets helps to reach higher results overall. Last but not least, the video part of the dataset can be used to train a lightweight sequential model, improving global recall while maintaining precision for earlier detections. [We make both our code and data available online](https://github.com/joseg20/wildfires2025).
>
---
#### [replaced 065] Generate, Transduct, Adapt: Iterative Transduction with VLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.06031v2](http://arxiv.org/pdf/2501.06031v2)**

> **作者:** Oindrila Saha; Logan Lawrence; Grant Van Horn; Subhransu Maji
>
> **备注:** Code is released at https://github.com/cvl-umass/GTA-CLIP
>
> **摘要:** Transductive zero-shot learning with vision-language models leverages image-image similarities within the dataset to achieve better classification accuracy compared to the inductive setting. However, there is little work that explores the structure of the language space in this context. We propose GTA-CLIP, a novel technique that incorporates supervision from language models for joint transduction in language and vision spaces. Our approach is iterative and consists of three steps: (i) incrementally exploring the attribute space by querying language models, (ii) an attribute-augmented transductive inference procedure, and (iii) fine-tuning the language and vision encoders based on inferred labels within the dataset. Through experiments with CLIP encoders, we demonstrate that GTA-CLIP, yields an average performance improvement of 8.6% and 3.7% across 12 datasets and 3 encoders, over CLIP and transductive CLIP respectively in the zero-shot setting. We also observe similar improvements in a few-shot setting. We present ablation studies that demonstrate the value of each step and visualize how the vision and language spaces evolve over iterations driven by the transductive learning. Code is released at https://github.com/cvl-umass/GTA-CLIP
>
---
#### [replaced 066] DarkIR: Robust Low-Light Image Restoration
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2412.13443v3](http://arxiv.org/pdf/2412.13443v3)**

> **作者:** Daniel Feijoo; Juan C. Benito; Alvaro Garcia; Marcos V. Conde
>
> **备注:** CVPR 2025
>
> **摘要:** Photography during night or in dark conditions typically suffers from noise, low light and blurring issues due to the dim environment and the common use of long exposure. Although Deblurring and Low-light Image Enhancement (LLIE) are related under these conditions, most approaches in image restoration solve these tasks separately. In this paper, we present an efficient and robust neural network for multi-task low-light image restoration. Instead of following the current tendency of Transformer-based models, we propose new attention mechanisms to enhance the receptive field of efficient CNNs. Our method reduces the computational costs in terms of parameters and MAC operations compared to previous methods. Our model, DarkIR, achieves new state-of-the-art results on the popular LOLBlur, LOLv2 and Real-LOLBlur datasets, being able to generalize on real-world night and dark images. Code and models at https://github.com/cidautai/DarkIR
>
---
#### [replaced 067] KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09823v2](http://arxiv.org/pdf/2508.09823v2)**

> **作者:** Valentin Boussot; Jean-Louis Dillenseger
>
> **备注:** https://github.com/vboussot/KonfAI
>
> **摘要:** KonfAI is a modular, extensible, and fully configurable deep learning framework specifically designed for medical imaging tasks. It enables users to define complete training, inference, and evaluation workflows through structured YAML configuration files, without modifying the underlying code. This declarative approach enhances reproducibility, transparency, and experimental traceability while reducing development time. Beyond the capabilities of standard pipelines, KonfAI provides native abstractions for advanced strategies including patch-based learning, test-time augmentation, model ensembling, and direct access to intermediate feature representations for deep supervision. It also supports complex multi-model training setups such as generative adversarial architectures. Thanks to its modular and extensible architecture, KonfAI can easily accommodate custom models, loss functions, and data processing components. The framework has been successfully applied to segmentation, registration, and image synthesis tasks, and has contributed to top-ranking results in several international medical imaging challenges. KonfAI is open source and available at https://github.com/vboussot/KonfAI.
>
---
#### [replaced 068] GeoVLM-R1: Reinforcement Fine-Tuning for Improved Remote Sensing Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.25026v3](http://arxiv.org/pdf/2509.25026v3)**

> **作者:** Mustansar Fiaz; Hiyam Debary; Paolo Fraccaro; Danda Paudel; Luc Van Gool; Fahad Khan; Salman Khan
>
> **备注:** Tables 6 and Figures 8. https://mustansarfiaz.github.io/GeoVLM-R1/
>
> **摘要:** Recent advances in reinforcement learning (RL) have delivered strong reasoning capabilities in natural image domains, yet their potential for Earth Observation (EO) remains largely unexplored. EO tasks introduce unique challenges, spanning referred object detection, image or region captioning, change detection, grounding, and temporal analysis, that demand task aware reasoning. We propose a novel post training framework that incorporates task aware rewards to enable effective adaptation of reasoning based RL models to diverse EO tasks. This training strategy enhances reasoning capabilities for remote sensing images, stabilizes optimization, and improves robustness. Extensive experiments across multiple EO benchmarks show consistent performance gains over state of the art generic and specialized vision language models. Code and models will be released publicly at https://mustansarfiaz.github.io/GeoVLM-R1/ .
>
---
#### [replaced 069] InternScenes: A Large-scale Simulatable Indoor Scene Dataset with Realistic Layouts
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.10813v2](http://arxiv.org/pdf/2509.10813v2)**

> **作者:** Weipeng Zhong; Peizhou Cao; Yichen Jin; Li Luo; Wenzhe Cai; Jingli Lin; Hanqing Wang; Zhaoyang Lyu; Tai Wang; Bo Dai; Xudong Xu; Jiangmiao Pang
>
> **摘要:** The advancement of Embodied AI heavily relies on large-scale, simulatable 3D scene datasets characterized by scene diversity and realistic layouts. However, existing datasets typically suffer from limitations in data scale or diversity, sanitized layouts lacking small items, and severe object collisions. To address these shortcomings, we introduce \textbf{InternScenes}, a novel large-scale simulatable indoor scene dataset comprising approximately 40,000 diverse scenes by integrating three disparate scene sources, real-world scans, procedurally generated scenes, and designer-created scenes, including 1.96M 3D objects and covering 15 common scene types and 288 object classes. We particularly preserve massive small items in the scenes, resulting in realistic and complex layouts with an average of 41.5 objects per region. Our comprehensive data processing pipeline ensures simulatability by creating real-to-sim replicas for real-world scans, enhances interactivity by incorporating interactive objects into these scenes, and resolves object collisions by physical simulations. We demonstrate the value of InternScenes with two benchmark applications: scene layout generation and point-goal navigation. Both show the new challenges posed by the complex and realistic layouts. More importantly, InternScenes paves the way for scaling up the model training for both tasks, making the generation and navigation in such complex scenes possible. We commit to open-sourcing the data, models, and benchmarks to benefit the whole community.
>
---
#### [replaced 070] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.07447v2](http://arxiv.org/pdf/2509.07447v2)**

> **作者:** Taiying Peng; Jiacheng Hua; Miao Liu; Feng Lu
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** The emergence of advanced multimodal large language models (MLLMs) has significantly enhanced AI assistants' ability to process complex information across modalities. Recently, egocentric videos, by directly capturing user focus, actions, and context in an unified coordinate, offer an exciting opportunity to enable proactive and personalized AI user experiences with MLLMs. However, existing benchmarks overlook the crucial role of gaze as an indicator of user intent. To address this gap, we introduce EgoGazeVQA, an egocentric gaze-guided video question answering benchmark that leverages gaze information to improve the understanding of longer daily-life videos. EgoGazeVQA consists of gaze-based QA pairs generated by MLLMs and refined by human annotators. Our experiments reveal that existing MLLMs struggle to accurately interpret user intentions. In contrast, our gaze-guided intent prompting methods significantly enhance performance by integrating spatial, temporal, and intent-related cues. We further conduct experiments on gaze-related fine-tuning and analyze how gaze estimation accuracy impacts prompting effectiveness. These results underscore the value of gaze for more personalized and effective AI assistants in egocentric settings. Project page: https://taiyi98.github.io/projects/EgoGazeVQA
>
---
#### [replaced 071] J-RAS: Enhancing Medical Image Segmentation via Retrieval-Augmented Joint Training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.09953v2](http://arxiv.org/pdf/2510.09953v2)**

> **作者:** Salma J. Ahmed; Emad A. Mohammed; Azam Asilian Bidgoli
>
> **摘要:** Image segmentation, the process of dividing images into meaningful regions, is critical in medical applications for accurate diagnosis, treatment planning, and disease monitoring. Although manual segmentation by healthcare professionals produces precise outcomes, it is time-consuming, costly, and prone to variability due to differences in human expertise. Artificial intelligence (AI)-based methods have been developed to address these limitations by automating segmentation tasks; however, they often require large, annotated datasets that are rarely available in practice and frequently struggle to generalize across diverse imaging conditions due to inter-patient variability and rare pathological cases. In this paper, we propose Joint Retrieval Augmented Segmentation (J-RAS), a joint training method for guided image segmentation that integrates a segmentation model with a retrieval model. Both models are jointly optimized, enabling the segmentation model to leverage retrieved image-mask pairs to enrich its anatomical understanding, while the retrieval model learns segmentation-relevant features beyond simple visual similarity. This joint optimization ensures that retrieval actively contributes meaningful contextual cues to guide boundary delineation, thereby enhancing the overall segmentation performance. We validate J-RAS across multiple segmentation backbones, including U-Net, TransUNet, SAM, and SegFormer, on two benchmark datasets: ACDC and M&Ms, demonstrating consistent improvements. For example, on the ACDC dataset, SegFormer without J-RAS achieves a mean Dice score of 0.8708$\pm$0.042 and a mean Hausdorff Distance (HD) of 1.8130$\pm$2.49, whereas with J-RAS, the performance improves substantially to a mean Dice score of 0.9115$\pm$0.031 and a mean HD of 1.1489$\pm$0.30. These results highlight the method's effectiveness and its generalizability across architectures and datasets.
>
---
#### [replaced 072] CryoFastAR: Fast Cryo-EM Ab Initio Reconstruction Made Easy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05864v2](http://arxiv.org/pdf/2506.05864v2)**

> **作者:** Jiakai Zhang; Shouchen Zhou; Haizhao Dai; Xinhang Liu; Peihao Wang; Zhiwen Fan; Yuan Pei; Jingyi Yu
>
> **摘要:** Pose estimation from unordered images is fundamental for 3D reconstruction, robotics, and scientific imaging. Recent geometric foundation models, such as DUSt3R, enable end-to-end dense 3D reconstruction but remain underexplored in scientific imaging fields like cryo-electron microscopy (cryo-EM) for near-atomic protein reconstruction. In cryo-EM, pose estimation and 3D reconstruction from unordered particle images still depend on time-consuming iterative optimization, primarily due to challenges such as low signal-to-noise ratios (SNR) and distortions from the contrast transfer function (CTF). We introduce CryoFastAR, the first geometric foundation model that can directly predict poses from Cryo-EM noisy images for Fast ab initio Reconstruction. By integrating multi-view features and training on large-scale simulated cryo-EM data with realistic noise and CTF modulations, CryoFastAR enhances pose estimation accuracy and generalization. To enhance training stability, we propose a progressive training strategy that first allows the model to extract essential features under simpler conditions before gradually increasing difficulty to improve robustness. Experiments show that CryoFastAR achieves comparable quality while significantly accelerating inference over traditional iterative approaches on both synthetic and real datasets.
>
---
#### [replaced 073] Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06486v2](http://arxiv.org/pdf/2504.06486v2)**

> **作者:** Samuel Stevens; S M Rayeed; Jenna Kline
>
> **备注:** 5 pages (main text), 3 figures. Accepted at the Imageomics Workshop at NeurIPS 2025
>
> **摘要:** The practical application of AI tools for specific computer vision tasks relies on the "small-data regime" of hundreds to thousands of labeled samples. This small-data regime is vital for applications requiring expensive expert annotations, such as ecological monitoring, medical diagnostics or industrial quality control. We find, however, that computer vision research has ignored the small data regime as evaluations increasingly focus on zero- and few-shot learning. We use the Natural World Tasks (NeWT) benchmark to compare multi-modal large language models (MLLMs) and vision-only methods across varying training set sizes. MLLMs exhibit early performance plateaus, while vision-only methods improve throughout the small-data regime, with performance gaps widening beyond 10 training examples. We provide the first comprehensive comparison between these approaches in small-data contexts and advocate for explicit small-data evaluations in AI research to better bridge theoretical advances with practical deployments.
>
---
#### [replaced 074] Prompt-guided Representation Disentanglement for Action Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.21783v3](http://arxiv.org/pdf/2509.21783v3)**

> **作者:** Tianci Wu; Guangming Zhu; Jiang Lu; Siyuan Wang; Ning Wang; Nuoye Xiong; Zhang Liang
>
> **摘要:** Action recognition is a fundamental task in video understanding. Existing methods typically extract unified features to process all actions in one video, which makes it challenging to model the interactions between different objects in multi-action scenarios. To alleviate this issue, we explore disentangling any specified actions from complex scenes as an effective solution. In this paper, we propose Prompt-guided Disentangled Representation for Action Recognition (ProDA), a novel framework that disentangles any specified actions from a multi-action scene. ProDA leverages Spatio-temporal Scene Graphs (SSGs) and introduces Dynamic Prompt Module (DPM) to guide a Graph Parsing Neural Network (GPNN) in generating action-specific representations. Furthermore, we design a video-adapted GPNN that aggregates information using dynamic weights. Experiments in video action recognition demonstrate the effectiveness of our approach when compared with the state-of-the-art methods. Our code can be found in https://github.com/iamsnaping/ProDA.git
>
---
#### [replaced 075] GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset
- **分类: cs.CV; I.4.6; I.2.10**

- **链接: [http://arxiv.org/pdf/2507.14697v2](http://arxiv.org/pdf/2507.14697v2)**

> **作者:** Zhiwei Zhang; Zi Ye; Yibin Wen; Shuai Yuan; Haohuan Fu; Jianxi Huang; Juepeng Zheng
>
> **备注:** 38 pages, 18 figures, submitted to NeurIPS 2025
>
> **摘要:** Agricultural parcels serve as basic units for conducting agricultural practices and applications, which is vital for land ownership registration, food security assessment, soil erosion monitoring, etc. However, existing agriculture parcel extraction studies only focus on mid-resolution mapping or regular plain farmlands while lacking representation of complex terraced terrains due to the demands of precision agriculture.In this paper, we introduce a more fine-grained terraced parcel dataset named GTPBD (Global Terraced Parcel and Boundary Dataset), which is the first fine-grained dataset covering major worldwide terraced regions with more than 200,000 complex terraced parcels with manual annotation. GTPBD comprises 47,537 high-resolution images with three-level labels, including pixel-level boundary labels, mask labels, and parcel labels. It covers seven major geographic zones in China and transcontinental climatic regions around the world.Compared to the existing datasets, the GTPBD dataset brings considerable challenges due to the: (1) terrain diversity; (2) complex and irregular parcel objects; and (3) multiple domain styles. Our proposed GTPBD dataset is suitable for four different tasks, including semantic segmentation, edge detection, terraced parcel extraction, and unsupervised domain adaptation (UDA) tasks.Accordingly, we benchmark the GTPBD dataset on eight semantic segmentation methods, four edge extraction methods, three parcel extraction methods, and five UDA methods, along with a multi-dimensional evaluation framework integrating pixel-level and object-level metrics. GTPBD fills a critical gap in terraced remote sensing research, providing a basic infrastructure for fine-grained agricultural terrain analysis and cross-scenario knowledge transfer.
>
---
#### [replaced 076] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.19764v2](http://arxiv.org/pdf/2503.19764v2)**

> **作者:** Christina Kassab; Sacha Morin; Martin Büchner; Matías Mattamala; Kumaraditya Gupta; Abhinav Valada; Liam Paull; Maurice Fallon
>
> **备注:** NeurIPS 2025
>
> **摘要:** 3D scene understanding has been transformed by open-vocabulary language models that enable interaction via natural language. However, at present the evaluation of these representations is limited to datasets with closed-set semantics that do not capture the richness of language. This work presents OpenLex3D, a dedicated benchmark for evaluating 3D open-vocabulary scene representations. OpenLex3D provides entirely new label annotations for scenes from Replica, ScanNet++, and HM3D, which capture real-world linguistic variability by introducing synonymical object categories and additional nuanced descriptions. Our label sets provide 13 times more labels per scene than the original datasets. By introducing an open-set 3D semantic segmentation task and an object retrieval task, we evaluate various existing 3D open-vocabulary methods on OpenLex3D, showcasing failure cases, and avenues for improvement. Our experiments provide insights on feature precision, segmentation, and downstream capabilities. The benchmark is publicly available at: https://openlex3d.github.io/.
>
---
#### [replaced 077] Capturing More: Learning Multi-Domain Representations for Robust Online Handwriting Verification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.01427v2](http://arxiv.org/pdf/2508.01427v2)**

> **作者:** Peirong Zhang; Kai Ding; Lianwen Jin
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** In this paper, we propose SPECTRUM, a temporal-frequency synergistic model that unlocks the untapped potential of multi-domain representation learning for online handwriting verification (OHV). SPECTRUM comprises three core components: (1) a multi-scale interactor that finely combines temporal and frequency features through dual-modal sequence interaction and multi-scale aggregation, (2) a self-gated fusion module that dynamically integrates global temporal and frequency features via self-driven balancing. These two components work synergistically to achieve micro-to-macro spectral-temporal integration. (3) A multi-domain distance-based verifier then utilizes both temporal and frequency representations to improve discrimination between genuine and forged handwriting, surpassing conventional temporal-only approaches. Extensive experiments demonstrate SPECTRUM's superior performance over existing OHV methods, underscoring the effectiveness of temporal-frequency multi-domain learning. Furthermore, we reveal that incorporating multiple handwritten biometrics fundamentally enhances the discriminative power of handwriting representations and facilitates verification. These findings not only validate the efficacy of multi-domain learning in OHV but also pave the way for future research in multi-domain approaches across both feature and biometric domains. Code is publicly available at https://github.com/NiceRingNode/SPECTRUM.
>
---
#### [replaced 078] Algorithmic Implementation: An Introduction to a Low-Cost, GUI-Based, Semi-Unsupervised Microscopy Segmentation Framework
- **分类: q-bio.QM; cs.CV; eess.IV; q-bio.CB**

- **链接: [http://arxiv.org/pdf/2509.11354v2](http://arxiv.org/pdf/2509.11354v2)**

> **作者:** Surajit Das; Pavel Zun
>
> **摘要:** This article presents a novel microscopy image analysis framework designed for low-budget labs equipped with a standard CPU desktop. The Python-based program enables cytometric analysis of live, unstained cells in culture through an advanced computer vision and machine learning pipeline. Crucially, the framework operates on label-free data, requiring no manually annotated training data or training phase. It is accessible via a user-friendly, cross-platform GUI that requires no programming skills, while also providing a scripting interface for programmatic control and integration by developers. The end-to-end workflow performs semantic and instance segmentation, feature extraction, analysis, evaluation, and automated report generation. Its modular architecture supports easy maintenance and flexible integration while supporting both single-image and batch processing. Validated on several unstained cell types from the public dataset of livecells, the framework demonstrates superior accuracy and reproducibility compared to contemporary tools like Cellpose and StarDist. Its competitive segmentation speed on a CPU-based platform highlights its significant potential for basic research and clinical application-particularly in cell transplantation for personalised medicine and muscle regeneration therapies. The access to the application is available for reproducibility.
>
---
#### [replaced 079] UVE: Are MLLMs Unified Evaluators for AI-Generated Videos?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09949v3](http://arxiv.org/pdf/2503.09949v3)**

> **作者:** Yuanxin Liu; Rui Zhu; Shuhuai Ren; Jiacong Wang; Haoyuan Guo; Xu Sun; Lu Jiang
>
> **摘要:** With the rapid growth of video generative models (VGMs), it is essential to develop reliable and comprehensive automatic metrics for AI-generated videos (AIGVs). Existing methods either use off-the-shelf models optimized for other tasks or rely on human assessment data to train specialized evaluators. These approaches are constrained to specific evaluation aspects and are difficult to scale with the increasing demands for finer-grained and more comprehensive evaluations. To address this issue, this work investigates the feasibility of using multimodal large language models (MLLMs) as a unified evaluator for AIGVs, leveraging their strong visual perception and language understanding capabilities. To evaluate the performance of automatic metrics in unified AIGV evaluation, we introduce a benchmark called UVE-Bench. UVE-Bench collects videos generated by state-of-the-art VGMs and provides pairwise human preference annotations across 15 evaluation aspects. Using UVE-Bench, we extensively evaluate 18 MLLMs. Our empirical results suggest that while advanced MLLMs (e.g., Qwen2VL-72B and InternVL2.5-78B) still lag behind human evaluators, they demonstrate promising ability in unified AIGV evaluation, significantly surpassing existing specialized evaluation methods. Additionally, we conduct an in-depth analysis of key design choices that impact the performance of MLLM-driven evaluators, offering valuable insights for future research on AIGV evaluation.
>
---
#### [replaced 080] STRIDE-QA: Visual Question Answering Dataset for Spatiotemporal Reasoning in Urban Driving Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10427v2](http://arxiv.org/pdf/2508.10427v2)**

> **作者:** Keishi Ishihara; Kento Sasaki; Tsubasa Takahashi; Daiki Shiono; Yu Yamaguchi
>
> **备注:** Project Page: https://turingmotors.github.io/stride-qa/
>
> **摘要:** Vision-Language Models (VLMs) have been applied to autonomous driving to support decision-making in complex real-world scenarios. However, their training on static, web-sourced image-text pairs fundamentally limits the precise spatiotemporal reasoning required to understand and predict dynamic traffic scenes. We address this critical gap with STRIDE-QA, a large-scale visual question answering (VQA) dataset for physically grounded reasoning from an ego-centric perspective. Constructed from 100 hours of multi-sensor driving data in Tokyo, capturing diverse and challenging conditions, STRIDE-QA is the largest VQA dataset for spatiotemporal reasoning in urban driving, offering 16 million QA pairs over 285K frames. Grounded by dense, automatically generated annotations including 3D bounding boxes, segmentation masks, and multi-object tracks, the dataset uniquely supports both object-centric and ego-centric reasoning through three novel QA tasks that require spatial localization and temporal prediction. Our benchmarks demonstrate that existing VLMs struggle significantly, achieving near-zero scores on prediction consistency. In contrast, VLMs fine-tuned on STRIDE-QA exhibit dramatic performance gains, achieving 55% success in spatial localization and 28% consistency in future motion prediction, compared to near-zero scores from general-purpose VLMs. Therefore, STRIDE-QA establishes a comprehensive foundation for developing more reliable VLMs for safety-critical autonomous systems.
>
---
#### [replaced 081] Human-MME: A Holistic Evaluation Benchmark for Human-Centric Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.26165v2](http://arxiv.org/pdf/2509.26165v2)**

> **作者:** Yuansen Liu; Haiming Tang; Jinlong Peng; Jiangning Zhang; Xiaozhong Ji; Qingdong He; Wenbin Wu; Donghao Luo; Zhenye Gan; Junwei Zhu; Yunhang Shen; Chaoyou Fu; Chengjie Wang; Xiaobin Hu; Shuicheng Yan
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated significant advances in visual understanding tasks. However, their capacity to comprehend human-centric scenes has rarely been explored, primarily due to the absence of comprehensive evaluation benchmarks that take into account both the human-oriented granular level and higher-dimensional causal reasoning ability. Such high-quality evaluation benchmarks face tough obstacles, given the physical complexity of the human body and the difficulty of annotating granular structures. In this paper, we propose Human-MME, a curated benchmark designed to provide a more holistic evaluation of MLLMs in human-centric scene understanding. Compared with other existing benchmarks, our work provides three key features: 1. Diversity in human scene, spanning 4 primary visual domains with 15 secondary domains and 43 sub-fields to ensure broad scenario coverage. 2. Progressive and diverse evaluation dimensions, evaluating the human-based activities progressively from the human-oriented granular perception to the higher-dimensional reasoning, consisting of eight dimensions with 19,945 real-world image question pairs and an evaluation suite. 3. High-quality annotations with rich data paradigms, constructing the automated annotation pipeline and human-annotation platform, supporting rigorous manual labeling to facilitate precise and reliable model assessment. Our benchmark extends the single-target understanding to the multi-person and multi-image mutual understanding by constructing the choice, short-answer, grounding, ranking and judgment question components, and complex questions of their combination. The extensive experiments on 17 state-of-the-art MLLMs effectively expose the limitations and guide future MLLMs research toward better human-centric image understanding. All data and code are available at https://github.com/Yuan-Hou/Human-MME.
>
---
#### [replaced 082] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.24099v2](http://arxiv.org/pdf/2509.24099v2)**

> **作者:** Prerit Gupta; Shourya Verma; Ananth Grama; Aniket Bera
>
> **备注:** Under review at ICLR 2026
>
> **摘要:** Generating realistic, context-aware two-person motion conditioned on diverse modalities remains a central challenge in computer graphics, animation, and human-computer interaction. We introduce DualFlow, a unified and efficient framework for multi-modal two-person motion generation. DualFlow conditions 3D motion synthesis on diverse inputs, including text, music, and prior motion sequences. Leveraging rectified flow, it achieves deterministic straight-line sampling paths between noise and data, reducing inference time and mitigating error accumulation common in diffusion-based models. To enhance semantic grounding, DualFlow employs a Retrieval-Augmented Generation (RAG) module that retrieves motion exemplars using music features and LLM-based text decompositions of spatial relations, body movements, and rhythmic patterns. We use contrastive objective that further strengthens alignment with conditioning signals and introduce synchronization loss that improves inter-person coordination. Extensive evaluations across text-to-motion, music-to-motion, and multi-modal interactive benchmarks show consistent gains in motion quality, responsiveness, and efficiency. DualFlow produces temporally coherent and rhythmically synchronized motions, setting state-of-the-art in multi-modal human motion generation.
>
---
#### [replaced 083] SAIP-Net: Enhancing Remote Sensing Image Segmentation via Spectral Adaptive Information Propagation
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2504.16564v2](http://arxiv.org/pdf/2504.16564v2)**

> **作者:** Zhongtao Wang; Xizhe Cao; Yisong Chen; Guoping Wang
>
> **摘要:** Semantic segmentation of remote sensing imagery demands precise spatial boundaries and robust intra-class consistency, challenging conventional hierarchical models. To address limitations arising from spatial domain feature fusion and insufficient receptive fields, this paper introduces SAIP-Net, a novel frequency-aware segmentation framework that leverages Spectral Adaptive Information Propagation. SAIP-Net employs adaptive frequency filtering and multi-scale receptive field enhancement to effectively suppress intra-class feature inconsistencies and sharpen boundary lines. Comprehensive experiments demonstrate significant performance improvements over state-of-the-art methods, highlighting the effectiveness of spectral-adaptive strategies combined with expanded receptive fields for remote sensing image segmentation.
>
---
#### [replaced 084] Contrast Sensitivity in Multimodal Large Language Models: A Psychophysics-Inspired Evaluation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10367v2](http://arxiv.org/pdf/2508.10367v2)**

> **作者:** Pablo Hernández-Cámara; Alexandra Gomez-Villa; Jose Manuel Jaén-Lorites; Jorge Vila-Tomás; Valero Laparra; Jesus Malo
>
> **摘要:** Understanding how Multimodal Large Language Models (MLLMs) process low-level visual features is critical for evaluating their perceptual abilities and has not been systematically characterized. Inspired by human psychophysics, we introduce a behavioural method for estimating the Contrast Sensitivity Function (CSF) in MLLMs by treating them as end-to-end observers. Models are queried with structured prompts while viewing noise-based stimuli filtered at specific spatial frequencies. Psychometric functions are derived from the binary verbal responses, and contrast thresholds (and CSFs) are obtained without relying on internal activations or classifier-based proxies. Our results reveal that some models resemble human CSFs in shape or scale, but none capture both. We also find that CSF estimates are highly sensitive to prompt phrasing, indicating limited linguistic robustness. Finally, we show that CSFs predict model performance under frequency-filtered and adversarial conditions. These findings highlight systematic differences in frequency tuning across MLLMs and establish CSF estimation as a scalable diagnostic tool for multimodal perception.
>
---
#### [replaced 085] LongLive: Real-time Interactive Long Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.22622v2](http://arxiv.org/pdf/2509.22622v2)**

> **作者:** Shuai Yang; Wei Huang; Ruihang Chu; Yicheng Xiao; Yuyang Zhao; Xianbang Wang; Muyang Li; Enze Xie; Yingcong Chen; Yao Lu; Song Han; Yukang Chen
>
> **备注:** Code, model, and demos are available at https://github.com/NVlabs/LongLive
>
> **摘要:** We present LongLive, a frame-level autoregressive (AR) framework for real-time and interactive long video generation. Long video generation presents challenges in both efficiency and quality. Diffusion and Diffusion-Forcing models can produce high-quality videos but suffer from low efficiency due to bidirectional attention. Causal attention AR models support KV caching for faster inference, but often degrade in quality on long videos due to memory challenges during long-video training. In addition, beyond static prompt-based generation, interactive capabilities, such as streaming prompt inputs, are critical for dynamic content creation, enabling users to guide narratives in real time. This interactive requirement significantly increases complexity, especially in ensuring visual consistency and semantic coherence during prompt transitions. To address these challenges, LongLive adopts a causal, frame-level AR design that integrates a KV-recache mechanism that refreshes cached states with new prompts for smooth, adherent switches; streaming long tuning to enable long video training and to align training and inference (train-long-test-long); and short window attention paired with a frame-level attention sink, shorten as frame sink, preserving long-range consistency while enabling faster generation. With these key designs, LongLive fine-tunes a 1.3B-parameter short-clip model to minute-long generation in just 32 GPU-days. At inference, LongLive sustains 20.7 FPS on a single NVIDIA H100, achieves strong performance on VBench in both short and long videos. LongLive supports up to 240-second videos on a single H100 GPU. LongLive further supports INT8-quantized inference with only marginal quality loss.
>
---
#### [replaced 086] TreeDiffusion: Hierarchical Generative Clustering for Conditional Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.16910v2](http://arxiv.org/pdf/2410.16910v2)**

> **作者:** Jorge da Silva Gonçalves; Laura Manduchi; Moritz Vandenhirtz; Julia E. Vogt
>
> **备注:** 31 pages, accepted to ECML PKDD 2025
>
> **摘要:** Generative modeling and clustering are conventionally distinct tasks in machine learning. Variational Autoencoders (VAEs) have been widely explored for their ability to integrate both, providing a framework for generative clustering. However, while VAEs can learn meaningful cluster representations in latent space, they often struggle to generate high-quality samples. This paper addresses this problem by introducing TreeDiffusion, a deep generative model that conditions diffusion models on learned latent hierarchical cluster representations from a VAE to obtain high-quality, cluster-specific generations. Our approach consists of two steps: first, a VAE-based clustering model learns a hierarchical latent representation of the data. Second, a cluster-aware diffusion model generates realistic images conditioned on the learned hierarchical structure. We systematically compare the generative capabilities of our approach with those of alternative conditioning strategies. Empirically, we demonstrate that conditioning diffusion models on hierarchical cluster representations improves the generative performance on real-world datasets compared to other approaches. Moreover, a key strength of our method lies in its ability to generate images that are both representative and specific to each cluster, enabling more detailed visualization of the learned latent structure. Our approach addresses the generative limitations of VAE-based clustering approaches by leveraging their learned structure, thereby advancing the field of generative clustering.
>
---
