# 计算机视觉 cs.CV

- **最新发布 118 篇**

- **更新 62 篇**

## 最新发布

#### [new 001] Dynam3D: Dynamic Layered 3D Tokens Empower VLM for Vision-and-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉与语言导航任务，解决现有模型在3D几何理解、长期记忆和动态环境适应上的不足，提出Dynam3D模型。通过分层动态3D表征将2D特征投影至3D空间，构建多层级语义表达，实现在线更新与定位，提升导航性能并在多个基准中刷新纪录。**

- **链接: [http://arxiv.org/pdf/2505.11383v1](http://arxiv.org/pdf/2505.11383v1)**

> **作者:** Zihan Wang; Seungjun Lee; Gim Hee Lee
>
> **摘要:** Vision-and-Language Navigation (VLN) is a core task where embodied agents leverage their spatial mobility to navigate in 3D environments toward designated destinations based on natural language instructions. Recently, video-language large models (Video-VLMs) with strong generalization capabilities and rich commonsense knowledge have shown remarkable performance when applied to VLN tasks. However, these models still encounter the following challenges when applied to real-world 3D navigation: 1) Insufficient understanding of 3D geometry and spatial semantics; 2) Limited capacity for large-scale exploration and long-term environmental memory; 3) Poor adaptability to dynamic and changing environments.To address these limitations, we propose Dynam3D, a dynamic layered 3D representation model that leverages language-aligned, generalizable, and hierarchical 3D representations as visual input to train 3D-VLM in navigation action prediction. Given posed RGB-D images, our Dynam3D projects 2D CLIP features into 3D space and constructs multi-level 3D patch-instance-zone representations for 3D geometric and semantic understanding with a dynamic and layer-wise update strategy. Our Dynam3D is capable of online encoding and localization of 3D instances, and dynamically updates them in changing environments to provide large-scale exploration and long-term memory capabilities for navigation. By leveraging large-scale 3D-language pretraining and task-specific adaptation, our Dynam3D sets new state-of-the-art performance on VLN benchmarks including R2R-CE, REVERIE-CE and NavRAG-CE under monocular settings. Furthermore, experiments for pre-exploration, lifelong memory, and real-world robot validate the effectiveness of practical deployment.
>
---
#### [new 002] Relative Drawing Identification Complexity is Invariant to Modality in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉-语言模型中概念识别复杂度是否跨模态一致，属于多模态表征分析任务。通过比较图像位图和笔画坐标两种模态在Quick, Draw!数据集上的教学效率，发现图像模态教学更高效，但概念复杂度排序跨模态呈现显著一致性，表明概念简繁具有独立于表征模态的内在属性。**

- **链接: [http://arxiv.org/pdf/2505.10583v1](http://arxiv.org/pdf/2505.10583v1)**

> **作者:** Diogo Freitas; Brigt Håvardstun; Cèsar Ferri; Darío Garigliotti; Jan Arne Telle; José Hernández-Orallo
>
> **备注:** 54 pages (42 pages of appendix)
>
> **摘要:** Large language models have become multimodal, and many of them are said to integrate their modalities using common representations. If this were true, a drawing of a car as an image, for instance, should map to the similar area in the latent space as a textual description of the strokes that conform the drawing. To explore this in a black-box access regime to these models, we propose the use of machine teaching, a theory that studies the minimal set of examples a teacher needs to choose so that the learner captures the concept. In this paper we evaluate the complexity of teaching visual-language models a subset of objects in the Quick, Draw! dataset using two presentations: raw images as bitmaps and trace coordinates in TikZ format. The results indicate that image-based representations generally require fewer segments and achieve higher accuracy than coordinate-based representations. But, surprisingly, the teaching size usually ranks concepts similarly across both modalities, even when controlling for (a human proxy of) concept priors, suggesting that the simplicity of concepts may be an inherent property that transcends modality representations.
>
---
#### [new 003] NeuSEditor: From Multi-View Images to Text-Guided Neural Surface Edits
- **分类: cs.CV**

- **简介: 该论文提出NeuSEditor，属于神经隐式表面编辑任务，解决多视图图像生成场景中文本引导编辑时身份失真与几何不一致问题。通过身份保留架构分离场景要素，结合几何感知蒸馏损失提升质量，无需持续数据更新，效果优于PDS等方法。**

- **链接: [http://arxiv.org/pdf/2505.10827v1](http://arxiv.org/pdf/2505.10827v1)**

> **作者:** Nail Ibrahimli; Julian F. P. Kooij; Liangliang Nan
>
> **摘要:** Implicit surface representations are valued for their compactness and continuity, but they pose significant challenges for editing. Despite recent advancements, existing methods often fail to preserve identity and maintain geometric consistency during editing. To address these challenges, we present NeuSEditor, a novel method for text-guided editing of neural implicit surfaces derived from multi-view images. NeuSEditor introduces an identity-preserving architecture that efficiently separates scenes into foreground and background, enabling precise modifications without altering the scene-specific elements. Our geometry-aware distillation loss significantly enhances rendering and geometric quality. Our method simplifies the editing workflow by eliminating the need for continuous dataset updates and source prompting. NeuSEditor outperforms recent state-of-the-art methods like PDS and InstructNeRF2NeRF, delivering superior quantitative and qualitative results. For more visual results, visit: neuseditor.github.io.
>
---
#### [new 004] HumaniBench: A Human-Centric Framework for Large Multimodal Models Evaluation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出HumaniBench评估框架，针对多模态模型的人类价值对齐问题。通过构建含32K图像问答对的基准数据集，评估模型在公平、伦理、同理心等7项人本原则的表现，发现主流模型在稳健性、视觉定位等方面存在缺陷，为提升模型社会责任感提供诊断工具。**

- **链接: [http://arxiv.org/pdf/2505.11454v1](http://arxiv.org/pdf/2505.11454v1)**

> **作者:** Shaina Raza; Aravind Narayanan; Vahid Reza Khazaie; Ashmal Vayani; Mukund S. Chettiar; Amandeep Singh; Mubarak Shah; Deval Pandya
>
> **摘要:** Large multimodal models (LMMs) now excel on many vision language benchmarks, however, they still struggle with human centered criteria such as fairness, ethics, empathy, and inclusivity, key to aligning with human values. We introduce HumaniBench, a holistic benchmark of 32K real-world image question pairs, annotated via a scalable GPT4o assisted pipeline and exhaustively verified by domain experts. HumaniBench evaluates seven Human Centered AI (HCAI) principles: fairness, ethics, understanding, reasoning, language inclusivity, empathy, and robustness, across seven diverse tasks, including open and closed ended visual question answering (VQA), multilingual QA, visual grounding, empathetic captioning, and robustness tests. Benchmarking 15 state of the art LMMs (open and closed source) reveals that proprietary models generally lead, though robustness and visual grounding remain weak points. Some open-source models also struggle to balance accuracy with adherence to human-aligned principles. HumaniBench is the first benchmark purpose built around HCAI principles. It provides a rigorous testbed for diagnosing alignment gaps and guiding LMMs toward behavior that is both accurate and socially responsible. Dataset, annotation prompts, and evaluation code are available at: https://vectorinstitute.github.io/HumaniBench
>
---
#### [new 005] GeoMM: On Geodesic Perspective for Multi-modal Learning
- **分类: cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决传统距离度量在非线性语义空间中难以区分高相似但语义差异样本的问题。提出首次引入测地线距离，通过构建图结构计算最短路径获取样本间关联，结合分层聚类与动态更新策略优化计算效率，实验证明其能有效挖掘复杂关系并提升模型性能。**

- **链接: [http://arxiv.org/pdf/2505.11216v1](http://arxiv.org/pdf/2505.11216v1)**

> **作者:** Shibin Mei; Hang Wang; Bingbing Ni
>
> **备注:** 15 pages, 3 figures, accepted by CVPR2025
>
> **摘要:** Geodesic distance serves as a reliable means of measuring distance in nonlinear spaces, and such nonlinear manifolds are prevalent in the current multimodal learning. In these scenarios, some samples may exhibit high similarity, yet they convey different semantics, making traditional distance metrics inadequate for distinguishing between positive and negative samples. This paper introduces geodesic distance as a novel distance metric in multi-modal learning for the first time, to mine correlations between samples, aiming to address the limitations of common distance metric. Our approach incorporates a comprehensive series of strategies to adapt geodesic distance for the current multimodal learning. Specifically, we construct a graph structure to represent the adjacency relationships among samples by thresholding distances between them and then apply the shortest-path algorithm to obtain geodesic distance within this graph. To facilitate efficient computation, we further propose a hierarchical graph structure through clustering and combined with incremental update strategies for dynamic status updates. Extensive experiments across various downstream tasks validate the effectiveness of our proposed method, demonstrating its capability to capture complex relationships between samples and improve the performance of multimodal learning models.
>
---
#### [new 006] GIE-Bench: Towards Grounded Evaluation for Text-Guided Image Editing
- **分类: cs.CV**

- **简介: 该论文属于文本引导图像编辑的评估任务，旨在解决现有评估方法（如CLIP相似度）精度不足的问题。提出GIE-Bench基准，通过功能正确性（自动多选题验证）和内容保持性（对象感知掩码评分）双维度评估模型，构建含1000+样本的数据集，验证指标有效性并分析模型性能权衡。**

- **链接: [http://arxiv.org/pdf/2505.11493v1](http://arxiv.org/pdf/2505.11493v1)**

> **作者:** Yusu Qian; Jiasen Lu; Tsu-Jui Fu; Xinze Wang; Chen Chen; Yinfei Yang; Wenze Hu; Zhe Gan
>
> **摘要:** Editing images using natural language instructions has become a natural and expressive way to modify visual content; yet, evaluating the performance of such models remains challenging. Existing evaluation approaches often rely on image-text similarity metrics like CLIP, which lack precision. In this work, we introduce a new benchmark designed to evaluate text-guided image editing models in a more grounded manner, along two critical dimensions: (i) functional correctness, assessed via automatically generated multiple-choice questions that verify whether the intended change was successfully applied; and (ii) image content preservation, which ensures that non-targeted regions of the image remain visually consistent using an object-aware masking technique and preservation scoring. The benchmark includes over 1000 high-quality editing examples across 20 diverse content categories, each annotated with detailed editing instructions, evaluation questions, and spatial object masks. We conduct a large-scale study comparing GPT-Image-1, the latest flagship in the text-guided image editing space, against several state-of-the-art editing models, and validate our automatic metrics against human ratings. Results show that GPT-Image-1 leads in instruction-following accuracy, but often over-modifies irrelevant image regions, highlighting a key trade-off in the current model behavior. GIE-Bench provides a scalable, reproducible framework for advancing more accurate evaluation of text-guided image editing.
>
---
#### [new 007] CUBIC: Concept Embeddings for Unsupervised Bias Identification using VLMs
- **分类: cs.CV; cs.AI; 68T10; I.2.4; I.5.2**

- **简介: 该论文提出CUBIC方法，用于无监督识别视觉模型中的潜在偏差。针对现有基于概念方法需人工标注和预定义偏见的局限，通过视觉语言模型的图像-文本潜在空间和线性分类器探针，量化超类标签潜在表征受概念影响的程度，对比决策边界法向量自动发现影响预测的可解释偏差概念。**

- **链接: [http://arxiv.org/pdf/2505.11060v1](http://arxiv.org/pdf/2505.11060v1)**

> **作者:** David Méndez; Gianpaolo Bontempo; Elisa Ficarra; Roberto Confalonieri; Natalia Díaz-Rodríguez
>
> **备注:** 8 pages, 3 figures, 5 tables. Accepted at IJCNN 2025; to appear in IEEE Xplore
>
> **摘要:** Deep vision models often rely on biases learned from spurious correlations in datasets. To identify these biases, methods that interpret high-level, human-understandable concepts are more effective than those relying primarily on low-level features like heatmaps. A major challenge for these concept-based methods is the lack of image annotations indicating potentially bias-inducing concepts, since creating such annotations requires detailed labeling for each dataset and concept, which is highly labor-intensive. We present CUBIC (Concept embeddings for Unsupervised Bias IdentifiCation), a novel method that automatically discovers interpretable concepts that may bias classifier behavior. Unlike existing approaches, CUBIC does not rely on predefined bias candidates or examples of model failures tied to specific biases, as such information is not always available. Instead, it leverages image-text latent space and linear classifier probes to examine how the latent representation of a superclass label$\unicode{x2014}$shared by all instances in the dataset$\unicode{x2014}$is influenced by the presence of a given concept. By measuring these shifts against the normal vector to the classifier's decision boundary, CUBIC identifies concepts that significantly influence model predictions. Our experiments demonstrate that CUBIC effectively uncovers previously unknown biases using Vision-Language Models (VLMs) without requiring the samples in the dataset where the classifier underperforms or prior knowledge of potential biases.
>
---
#### [new 008] QVGen: Pushing the Limit of Quantized Video Generative Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成模型的高效量化任务，旨在解决视频扩散模型因高计算需求难以部署的问题。提出QVGen框架，通过量化感知训练和辅助模块减少低比特（如4位）量化误差，并设计秩衰减策略逐步消除模块以保持性能。实验表明其4/3位模型在质量与指标上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11497v1](http://arxiv.org/pdf/2505.11497v1)**

> **作者:** Yushi Huang; Ruihao Gong; Jing Liu; Yifu Ding; Chengtao Lv; Haotong Qin; Jun Zhang
>
> **备注:** Our code will be released upon acceptance
>
> **摘要:** Video diffusion models (DMs) have enabled high-quality video synthesis. Yet, their substantial computational and memory demands pose serious challenges to real-world deployment, even on high-end GPUs. As a commonly adopted solution, quantization has proven notable success in reducing cost for image DMs, while its direct application to video DMs remains ineffective. In this paper, we present QVGen, a novel quantization-aware training (QAT) framework tailored for high-performance and inference-efficient video DMs under extremely low-bit quantization (e.g., 4-bit or below). We begin with a theoretical analysis demonstrating that reducing the gradient norm is essential to facilitate convergence for QAT. To this end, we introduce auxiliary modules ($\Phi$) to mitigate large quantization errors, leading to significantly enhanced convergence. To eliminate the inference overhead of $\Phi$, we propose a rank-decay strategy that progressively eliminates $\Phi$. Specifically, we repeatedly employ singular value decomposition (SVD) and a proposed rank-based regularization $\mathbf{\gamma}$ to identify and decay low-contributing components. This strategy retains performance while zeroing out inference overhead. Extensive experiments across $4$ state-of-the-art (SOTA) video DMs, with parameter sizes ranging from $1.3$B $\sim14$B, show that QVGen is the first to reach full-precision comparable quality under 4-bit settings. Moreover, it significantly outperforms existing methods. For instance, our 3-bit CogVideoX-2B achieves improvements of $+25.28$ in Dynamic Degree and $+8.43$ in Scene Consistency on VBench.
>
---
#### [new 009] Unsupervised Detection of Distribution Shift in Inverse Problems using Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于分布偏移检测任务，旨在解决逆问题中因训练与测试数据分布差异导致的扩散模型性能下降问题。提出一种无监督度量方法，仅利用间接测量数据和扩散模型分数函数估计KL散度，无需干净测试图像。通过理论证明与实验验证，该方法可量化分布偏移并提升重建质量。**

- **链接: [http://arxiv.org/pdf/2505.11482v1](http://arxiv.org/pdf/2505.11482v1)**

> **作者:** Shirin Shoushtari; Edward P. Chandler; Yuanhao Wang; M. Salman Asif; Ulugbek S. Kamilov
>
> **摘要:** Diffusion models are widely used as priors in imaging inverse problems. However, their performance often degrades under distribution shifts between the training and test-time images. Existing methods for identifying and quantifying distribution shifts typically require access to clean test images, which are almost never available while solving inverse problems (at test time). We propose a fully unsupervised metric for estimating distribution shifts using only indirect (corrupted) measurements and score functions from diffusion models trained on different datasets. We theoretically show that this metric estimates the KL divergence between the training and test image distributions. Empirically, we show that our score-based metric, using only corrupted measurements, closely approximates the KL divergence computed from clean images. Motivated by this result, we show that aligning the out-of-distribution score with the in-distribution score -- using only corrupted measurements -- reduces the KL divergence and leads to improved reconstruction quality across multiple inverse problems.
>
---
#### [new 010] Aquarius: A Family of Industry-Level Video Generation Models for Marketing Scenarios
- **分类: cs.CV**

- **简介: 该论文属于工业级视频生成任务，旨在解决营销场景中高效生成高保真、多格式长视频的挑战。提出Aquarius框架，包含分布式数据处理、多规模模型架构、高性能训练设施、并行推理加速及多营销应用，支持大规模集群和数百亿参数模型，提升生成效率与多样性。**

- **链接: [http://arxiv.org/pdf/2505.10584v1](http://arxiv.org/pdf/2505.10584v1)**

> **作者:** Huafeng Shi; Jianzhong Liang; Rongchang Xie; Xian Wu; Cheng Chen; Chang Liu
>
> **摘要:** This report introduces Aquarius, a family of industry-level video generation models for marketing scenarios designed for thousands-xPU clusters and models with hundreds of billions of parameters. Leveraging efficient engineering architecture and algorithmic innovation, Aquarius demonstrates exceptional performance in high-fidelity, multi-aspect-ratio, and long-duration video synthesis. By disclosing the framework's design details, we aim to demystify industrial-scale video generation systems and catalyze advancements in the generative video community. The Aquarius framework consists of five components: Distributed Graph and Video Data Processing Pipeline: Manages tens of thousands of CPUs and thousands of xPUs via automated task distribution, enabling efficient video data processing. Additionally, we are about to open-source the entire data processing framework named "Aquarius-Datapipe". Model Architectures for Different Scales: Include a Single-DiT architecture for 2B models and a Multimodal-DiT architecture for 13.4B models, supporting multi-aspect ratios, multi-resolution, and multi-duration video generation. High-Performance infrastructure designed for video generation model training: Incorporating hybrid parallelism and fine-grained memory optimization strategies, this infrastructure achieves 36% MFU at large scale. Multi-xPU Parallel Inference Acceleration: Utilizes diffusion cache and attention optimization to achieve a 2.35x inference speedup. Multiple marketing-scenarios applications: Including image-to-video, text-to-video (avatar), video inpainting and video personalization, among others. More downstream applications and multi-dimensional evaluation metrics will be added in the upcoming version updates.
>
---
#### [new 011] MutualNeRF: Improve the Performance of NeRF under Limited Samples with Mutual Information Theory
- **分类: cs.CV**

- **简介: 该论文属于3D场景合成任务，旨在解决NeRF在有限样本下性能不足的问题。提出MutualNeRF框架，基于互信息理论统一衡量图像间宏/微观相关性：稀疏视图时通过最小化互信息选择信息增量视角，少样本合成时最大化推断图像与真值的互信息，并引入正则项提升效果。实验验证其在低样本场景下优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11386v1](http://arxiv.org/pdf/2505.11386v1)**

> **作者:** Zifan Wang; Jingwei Li; Yitang Li; Yunze Liu
>
> **摘要:** This paper introduces MutualNeRF, a framework enhancing Neural Radiance Field (NeRF) performance under limited samples using Mutual Information Theory. While NeRF excels in 3D scene synthesis, challenges arise with limited data and existing methods that aim to introduce prior knowledge lack theoretical support in a unified framework. We introduce a simple but theoretically robust concept, Mutual Information, as a metric to uniformly measure the correlation between images, considering both macro (semantic) and micro (pixel) levels. For sparse view sampling, we strategically select additional viewpoints containing more non-overlapping scene information by minimizing mutual information without knowing ground truth images beforehand. Our framework employs a greedy algorithm, offering a near-optimal solution. For few-shot view synthesis, we maximize the mutual information between inferred images and ground truth, expecting inferred images to gain more relevant information from known images. This is achieved by incorporating efficient, plug-and-play regularization terms. Experiments under limited samples show consistent improvement over state-of-the-art baselines in different settings, affirming the efficacy of our framework.
>
---
#### [new 012] CLIP Embeddings for AI-Generated Image Detection: A Few-Shot Study with Lightweight Classifier
- **分类: cs.CV; cs.AI; I.2.10**

- **简介: 该论文研究基于CLIP嵌入的AI生成图像检测，属图像鉴真任务。旨在解决现有视觉语言模型在生成图分类中因预训练标签缺失导致性能受限的问题。提出冻结CLIP提取特征+轻量分类器微调的方案，在CIFAKE基准达95%准确率，少量数据下获85%效果，但发现广角照片和油画等特定类型存在显著分类挑战。**

- **链接: [http://arxiv.org/pdf/2505.10664v1](http://arxiv.org/pdf/2505.10664v1)**

> **作者:** Ziyang Ou
>
> **备注:** 8 pages, 5 figures, not submitted to any conference
>
> **摘要:** Verifying the authenticity of AI-generated images presents a growing challenge on social media platforms these days. While vision-language models (VLMs) like CLIP outdo in multimodal representation, their capacity for AI-generated image classification is underexplored due to the absence of such labels during the pre-training process. This work investigates whether CLIP embeddings inherently contain information indicative of AI generation. A proposed pipeline extracts visual embeddings using a frozen CLIP model, feeds its embeddings to lightweight networks, and fine-tunes only the final classifier. Experiments on the public CIFAKE benchmark show the performance reaches 95% accuracy without language reasoning. Few-shot adaptation to curated custom with 20% of the data results in performance to 85%. A closed-source baseline (Gemini-2.0) has the best zero-shot accuracy yet fails on specific styles. Notably, some specific image types, such as wide-angle photographs and oil paintings, pose significant challenges to classification. These results indicate previously unexplored difficulties in classifying certain types of AI-generated images, revealing new and more specific questions in this domain that are worth further investigation.
>
---
#### [new 013] A High-Performance Thermal Infrared Object Detection Framework with Centralized Regulation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对热红外图像目标检测中局部-全局信息融合不足的问题，提出CRT-YOLO框架。通过集中式特征调节机制整合高效多尺度注意力模块（EMA）和全局调控特征金字塔网络（CFP），增强长程依赖捕捉能力。实验表明模型在基准数据集上显著优于传统方法，消融研究验证了模块有效性。**

- **链接: [http://arxiv.org/pdf/2505.10825v1](http://arxiv.org/pdf/2505.10825v1)**

> **作者:** Jinke Li; Yue Wu; Xiaoyan Yang
>
> **备注:** This manuscript has been accepted for publication in the International Journal for Housing Science and Its Applications (IJHSA), 2025
>
> **摘要:** Thermal Infrared (TIR) technology involves the use of sensors to detect and measure infrared radiation emitted by objects, and it is widely utilized across a broad spectrum of applications. The advancements in object detection methods utilizing TIR images have sparked significant research interest. However, most traditional methods lack the capability to effectively extract and fuse local-global information, which is crucial for TIR-domain feature attention. In this study, we present a novel and efficient thermal infrared object detection framework, known as CRT-YOLO, that is based on centralized feature regulation, enabling the establishment of global-range interaction on TIR information. Our proposed model integrates efficient multi-scale attention (EMA) modules, which adeptly capture long-range dependencies while incurring minimal computational overhead. Additionally, it leverages the Centralized Feature Pyramid (CFP) network, which offers global regulation of TIR features. Extensive experiments conducted on two benchmark datasets demonstrate that our CRT-YOLO model significantly outperforms conventional methods for TIR image object detection. Furthermore, the ablation study provides compelling evidence of the effectiveness of our proposed modules, reinforcing the potential impact of our approach on advancing the field of thermal infrared object detection.
>
---
#### [new 014] Entropy-Driven Genetic Optimization for Deep-Feature-Guided Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文针对低光照图像增强任务，解决传统方法忽视语义特征的问题，提出一种无监督优化框架。基于NSGA-II算法优化亮度、对比度和伽马参数，结合深度特征保持语义一致性，利用GPU加速和局部搜索提升效果，实现视觉质量与语义保真平衡，在无配对数据场景下取得优异指标。**

- **链接: [http://arxiv.org/pdf/2505.11246v1](http://arxiv.org/pdf/2505.11246v1)**

> **作者:** Nirjhor Datta; Afroza Akther; M. Sohel Rahman
>
> **摘要:** Image enhancement methods often prioritize pixel level information, overlooking the semantic features. We propose a novel, unsupervised, fuzzy-inspired image enhancement framework guided by NSGA-II algorithm that optimizes image brightness, contrast, and gamma parameters to achieve a balance between visual quality and semantic fidelity. Central to our proposed method is the use of a pre trained deep neural network as a feature extractor. To find the best enhancement settings, we use a GPU-accelerated NSGA-II algorithm that balances multiple objectives, namely, increasing image entropy, improving perceptual similarity, and maintaining appropriate brightness. We further improve the results by applying a local search phase to fine-tune the top candidates from the genetic algorithm. Our approach operates entirely without paired training data making it broadly applicable across domains with limited or noisy labels. Quantitatively, our model achieves excellent performance with average BRISQUE and NIQE scores of 19.82 and 3.652, respectively, in all unpaired datasets. Qualitatively, enhanced images by our model exhibit significantly improved visibility in shadowed regions, natural balance of contrast and also preserve the richer fine detail without introducing noticable artifacts. This work opens new directions for unsupervised image enhancement where semantic consistency is critical.
>
---
#### [new 015] Human-Aligned Bench: Fine-Grained Assessment of Reasoning Ability in MLLMs vs. Humans
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Human-Aligned Bench基准，用于细粒度评估多模态大语言模型(MLLMs)与人类在推理任务中的对齐程度。通过构建包含9,794个双语多模态问题的数据集（涵盖视觉推理、定义判断等四类），结合人类正确率与易错选项分析，揭示当前MLLMs与人类表现的显著差异，为模型优化提供方向。**

- **链接: [http://arxiv.org/pdf/2505.11141v1](http://arxiv.org/pdf/2505.11141v1)**

> **作者:** Yansheng Qiu; Li Xiao; Zhaopan Xu; Pengfei Zhou; Zheng Wang; Kaipeng Zhang
>
> **摘要:** The goal of achieving Artificial General Intelligence (AGI) is to imitate humans and surpass them. Models such as OpenAI's o1, o3, and DeepSeek's R1 have demonstrated that large language models (LLMs) with human-like reasoning capabilities exhibit exceptional performance and are being gradually integrated into multimodal large language models (MLLMs). However, whether these models possess capabilities comparable to humans in handling reasoning tasks remains unclear at present. In this paper, we propose Human-Aligned Bench, a benchmark for fine-grained alignment of multimodal reasoning with human performance. Specifically, we collected 9,794 multimodal questions that solely rely on contextual reasoning, including bilingual (Chinese and English) multimodal questions and pure text-based questions, encompassing four question types: visual reasoning, definition judgment, analogical reasoning, and logical judgment. More importantly, each question is accompanied by human success rates and options that humans are prone to choosing incorrectly. Extensive experiments on the Human-Aligned Bench reveal notable differences between the performance of current MLLMs in multimodal reasoning and human performance. The findings on our benchmark provide insights into the development of the next-generation models.
>
---
#### [new 016] Towards Cross-modal Retrieval in Chinese Cultural Heritage Documents: Dataset and Solution
- **分类: cs.CV**

- **简介: 该论文针对中文文化遗产跨模态检索任务，解决该领域专用数据集缺失及细粒度图文对齐难题。提出CulTi数据集（5,726丝绸/敦煌壁画图文对），并设计基于中文CLIP的无训练策略LACLIP，通过加权相似度计算增强局部对齐能力，显著提升跨模态检索效果。**

- **链接: [http://arxiv.org/pdf/2505.10921v1](http://arxiv.org/pdf/2505.10921v1)**

> **作者:** Junyi Yuan; Jian Zhang; Fangyu Wu; Dongming Lu; Huanda Lu; Qiufeng Wang
>
> **摘要:** China has a long and rich history, encompassing a vast cultural heritage that includes diverse multimodal information, such as silk patterns, Dunhuang murals, and their associated historical narratives. Cross-modal retrieval plays a pivotal role in understanding and interpreting Chinese cultural heritage by bridging visual and textual modalities to enable accurate text-to-image and image-to-text retrieval. However, despite the growing interest in multimodal research, there is a lack of specialized datasets dedicated to Chinese cultural heritage, limiting the development and evaluation of cross-modal learning models in this domain. To address this gap, we propose a multimodal dataset named CulTi, which contains 5,726 image-text pairs extracted from two series of professional documents, respectively related to ancient Chinese silk and Dunhuang murals. Compared to existing general-domain multimodal datasets, CulTi presents a challenge for cross-modal retrieval: the difficulty of local alignment between intricate decorative motifs and specialized textual descriptions. To address this challenge, we propose LACLIP, a training-free local alignment strategy built upon a fine-tuned Chinese-CLIP. LACLIP enhances the alignment of global textual descriptions with local visual regions by computing weighted similarity scores during inference. Experimental results on CulTi demonstrate that LACLIP significantly outperforms existing models in cross-modal retrieval, particularly in handling fine-grained semantic associations within Chinese cultural heritage.
>
---
#### [new 017] MAVOS-DD: Multilingual Audio-Video Open-Set Deepfake Detection Benchmark
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文提出首个多语言音视频开放集深度伪造检测基准MAVOS-DD，解决现有检测器在训练未覆盖的生成模型/语言场景下性能骤降的问题。构建了包含8种语言、7种生成模型的250小时数据集，采用模型/语言隔离的数据划分方式验证开放集检测能力。实验表明主流检测器在开放集场景中效果显著下降，相关数据与代码已开源。**

- **链接: [http://arxiv.org/pdf/2505.11109v1](http://arxiv.org/pdf/2505.11109v1)**

> **作者:** Florinel-Alin Croitoru; Vlad Hondru; Marius Popescu; Radu Tudor Ionescu; Fahad Shahbaz Khan; Mubarak Shah
>
> **备注:** 15 pages
>
> **摘要:** We present the first large-scale open-set benchmark for multilingual audio-video deepfake detection. Our dataset comprises over 250 hours of real and fake videos across eight languages, with 60% of data being generated. For each language, the fake videos are generated with seven distinct deepfake generation models, selected based on the quality of the generated content. We organize the training, validation and test splits such that only a subset of the chosen generative models and languages are available during training, thus creating several challenging open-set evaluation setups. We perform experiments with various pre-trained and fine-tuned deepfake detectors proposed in recent literature. Our results show that state-of-the-art detectors are not currently able to maintain their performance levels when tested in our open-set scenarios. We publicly release our data and code at: https://huggingface.co/datasets/unibuc-cs/MAVOS-DD.
>
---
#### [new 018] Mitigate Language Priors in Large Vision-Language Models by Cross-Images Contrastive Decoding
- **分类: cs.CV**

- **简介: 该论文针对大视觉语言模型（LVLMs）因语言先验导致生成与视觉不一致的幻觉问题，提出跨图像对比解码方法（CICD）。通过对比多图特征识别并消除有害语言先验，保持文本流畅性，在图像描述任务中显著降低幻觉。属于多模态模型优化任务。**

- **链接: [http://arxiv.org/pdf/2505.10634v1](http://arxiv.org/pdf/2505.10634v1)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng
>
> **摘要:** Language priors constitute one of the primary causes of hallucinations in Large Vision-Language Models (LVLMs), driving the models to generate linguistically plausible yet visually inconsistent content. The language priors in LVLMs originate from the linguistic knowledge inherited from their pre-trained Large Language Model (LLM) backbone. Consequently, this characteristic is an intrinsic property of the model that remains independent of visual inputs. Inspired by the finding that language priors are consistent across images, we propose Cross-Image Contrastive Decoding (CICD), a simple yet effective training-free method to alleviate language priors in LVLMs. CICD first identifies essential and detrimental priors, and then employs contrastive decoding to eliminate the detrimental ones. This approach simultaneously prevents LVLMs from generating hallucinated content while maintaining textual fluency and coherence. Furthermore, the limited information overlap between images helps prevent visual information loss during contrastive decoding. We validate the effectiveness of CICD on four benchmarks with six LVLMs. Our experiments demonstrate that CICD performs remarkably well in mitigating language priors, especially in the image captioning task, where such priors are most pronounced. Code will be released once accepted.
>
---
#### [new 019] AW-GATCN: Adaptive Weighted Graph Attention Convolutional Network for Event Camera Data Joint Denoising and Object Recognition
- **分类: cs.CV**

- **简介: 该论文提出自适应加权图注意力网络(AW-GATCN)，解决事件相机数据中联合去噪与物体识别问题。针对噪声冗余和时空信息保留的挑战，通过自适应事件分割、多因素边权重机制和图去噪策略，在保持关键结构特征的同时有效降噪，实验显示其在四个数据集上识别准确率和降噪性能均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11232v1](http://arxiv.org/pdf/2505.11232v1)**

> **作者:** Haiyu Li; Charith Abhayaratne
>
> **摘要:** Event cameras, which capture brightness changes with high temporal resolution, inherently generate a significant amount of redundant and noisy data beyond essential object structures. The primary challenge in event-based object recognition lies in effectively removing this noise without losing critical spatial-temporal information. To address this, we propose an Adaptive Graph-based Noisy Data Removal framework for Event-based Object Recognition. Specifically, our approach integrates adaptive event segmentation based on normalized density analysis, a multifactorial edge-weighting mechanism, and adaptive graph-based denoising strategies. These innovations significantly enhance the integration of spatiotemporal information, effectively filtering noise while preserving critical structural features for robust recognition. Experimental evaluations on four challenging datasets demonstrate that our method achieves superior recognition accuracies of 83.77%, 76.79%, 99.30%, and 96.89%, surpassing existing graph-based methods by up to 8.79%, and improving noise reduction performance by up to 19.57%, with an additional accuracy gain of 6.26% compared to traditional Euclidean-based techniques.
>
---
#### [new 020] Visual Anomaly Detection under Complex View-Illumination Interplay: A Large-Scale Benchmark
- **分类: cs.CV**

- **简介: 该论文针对视觉异常检测（VAD）在复杂视角-光照耦合变化下的鲁棒性问题，构建了首个大规模评测基准M2AD，包含12种视角和10种光照组合的11.9万图像，提出跨配置信息融合与单图抗干扰双评测协议，揭示了现有方法在此场景下的显著性能缺陷。**

- **链接: [http://arxiv.org/pdf/2505.10996v1](http://arxiv.org/pdf/2505.10996v1)**

> **作者:** Yunkang Cao; Yuqi Cheng; Xiaohao Xu; Yiheng Zhang; Yihan Sun; Yuxiang Tan; Yuxin Zhang; Xiaonan Huang; Weiming Shen
>
> **备注:** Homgepage: https://hustcyq.github.io/M2AD/. Yunkang Cao and Yuqi Cheng contribute equally to this work
>
> **摘要:** The practical deployment of Visual Anomaly Detection (VAD) systems is hindered by their sensitivity to real-world imaging variations, particularly the complex interplay between viewpoint and illumination which drastically alters defect visibility. Current benchmarks largely overlook this critical challenge. We introduce Multi-View Multi-Illumination Anomaly Detection (M2AD), a new large-scale benchmark comprising 119,880 high-resolution images designed explicitly to probe VAD robustness under such interacting conditions. By systematically capturing 999 specimens across 10 categories using 12 synchronized views and 10 illumination settings (120 configurations total), M2AD enables rigorous evaluation. We establish two evaluation protocols: M2AD-Synergy tests the ability to fuse information across diverse configurations, and M2AD-Invariant measures single-image robustness against realistic view-illumination effects. Our extensive benchmarking shows that state-of-the-art VAD methods struggle significantly on M2AD, demonstrating the profound challenge posed by view-illumination interplay. This benchmark serves as an essential tool for developing and validating VAD methods capable of overcoming real-world complexities. Our full dataset and test suite will be released at https://hustcyq.github.io/M2AD to facilitate the field.
>
---
#### [new 021] Robust Emotion Recognition via Bi-Level Self-Supervised Continual Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于情感识别任务，针对生理信号（如脑电）的跨主体差异和噪声标签问题，提出双层自监督持续学习框架SSOCL。通过动态内存缓冲区迭代优化样本保留与伪标签分配，结合快速适应和聚类模块，解决连续无标记数据流下的模型泛化问题，实验验证其跨主体性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.10575v1](http://arxiv.org/pdf/2505.10575v1)**

> **作者:** Adnan Ahmad; Bahareh Nakisa; Mohammad Naim Rastgoo
>
> **摘要:** Emotion recognition through physiological signals such as electroencephalogram (EEG) has become an essential aspect of affective computing and provides an objective way to capture human emotions. However, physiological data characterized by cross-subject variability and noisy labels hinder the performance of emotion recognition models. Existing domain adaptation and continual learning methods struggle to address these issues, especially under realistic conditions where data is continuously streamed and unlabeled. To overcome these limitations, we propose a novel bi-level self-supervised continual learning framework, SSOCL, based on a dynamic memory buffer. This bi-level architecture iteratively refines the dynamic buffer and pseudo-label assignments to effectively retain representative samples, enabling generalization from continuous, unlabeled physiological data streams for emotion recognition. The assigned pseudo-labels are subsequently leveraged for accurate emotion prediction. Key components of the framework, including a fast adaptation module and a cluster-mapping module, enable robust learning and effective handling of evolving data streams. Experimental validation on two mainstream EEG tasks demonstrates the framework's ability to adapt to continuous data streams while maintaining strong generalization across subjects, outperforming existing approaches.
>
---
#### [new 022] RefPose: Leveraging Reference Geometric Correspondences for Accurate 6D Pose Estimation of Unseen Objects
- **分类: cs.CV**

- **简介: 该论文研究6D姿态估计任务，解决未见物体因缺乏先验知识导致的位姿估计难题。提出RefPose方法：通过参考图像建立几何对应关系，分阶段预测初始姿态并利用渲染比较策略迭代优化，结合注意力机制增强跨图像关联。相比依赖预定义模型的方法，能动态适应新物体形状，在BOP数据集实现SOTA精度。**

- **链接: [http://arxiv.org/pdf/2505.10841v1](http://arxiv.org/pdf/2505.10841v1)**

> **作者:** Jaeguk Kim; Jaewoo Park; Keuntek Lee; Nam Ik Cho
>
> **备注:** Accepted at CVPR 2025
>
> **摘要:** Estimating the 6D pose of unseen objects from monocular RGB images remains a challenging problem, especially due to the lack of prior object-specific knowledge. To tackle this issue, we propose RefPose, an innovative approach to object pose estimation that leverages a reference image and geometric correspondence as guidance. RefPose first predicts an initial pose by using object templates to render the reference image and establish the geometric correspondence needed for the refinement stage. During the refinement stage, RefPose estimates the geometric correspondence of the query based on the generated references and iteratively refines the pose through a render-and-compare approach. To enhance this estimation, we introduce a correlation volume-guided attention mechanism that effectively captures correlations between the query and reference images. Unlike traditional methods that depend on pre-defined object models, RefPose dynamically adapts to new object shapes by leveraging a reference image and geometric correspondence. This results in robust performance across previously unseen objects. Extensive evaluation on the BOP benchmark datasets shows that RefPose achieves state-of-the-art results while maintaining a competitive runtime.
>
---
#### [new 023] Deepfake Forensic Analysis: Source Dataset Attribution and Legal Implications of Synthetic Media Manipulation
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测和溯源任务，旨在解决GAN生成图像的来源追踪及法律合规问题。提出基于频谱、颜色和局部特征的框架，通过监督分类器实现高精度数据集归属分析，并探讨版权、隐私等法律影响。**

- **链接: [http://arxiv.org/pdf/2505.11110v1](http://arxiv.org/pdf/2505.11110v1)**

> **作者:** Massimiliano Cassia; Luca Guarnera; Mirko Casu; Ignazio Zangara; Sebastiano Battiato
>
> **摘要:** Synthetic media generated by Generative Adversarial Networks (GANs) pose significant challenges in verifying authenticity and tracing dataset origins, raising critical concerns in copyright enforcement, privacy protection, and legal compliance. This paper introduces a novel forensic framework for identifying the training dataset (e.g., CelebA or FFHQ) of GAN-generated images through interpretable feature analysis. By integrating spectral transforms (Fourier/DCT), color distribution metrics, and local feature descriptors (SIFT), our pipeline extracts discriminative statistical signatures embedded in synthetic outputs. Supervised classifiers (Random Forest, SVM, XGBoost) achieve 98-99% accuracy in binary classification (real vs. synthetic) and multi-class dataset attribution across diverse GAN architectures (StyleGAN, AttGAN, GDWCT, StarGAN, and StyleGAN2). Experimental results highlight the dominance of frequency-domain features (DCT/FFT) in capturing dataset-specific artifacts, such as upsampling patterns and spectral irregularities, while color histograms reveal implicit regularization strategies in GAN training. We further examine legal and ethical implications, showing how dataset attribution can address copyright infringement, unauthorized use of personal data, and regulatory compliance under frameworks like GDPR and California's AB 602. Our framework advances accountability and governance in generative modeling, with applications in digital forensics, content moderation, and intellectual property litigation.
>
---
#### [new 024] MIRAGE: A Multi-modal Benchmark for Spatial Perception, Reasoning, and Intelligence
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出多模态基准MIRAGE，针对空间感知与推理任务，解决现有模型在物体属性识别和空间关系推理的不足。通过设计包含计数、关系及联合任务的复杂场景，评估模型细粒度认知能力，揭示当前技术缺陷，推动时空推理框架发展。**

- **链接: [http://arxiv.org/pdf/2505.10604v1](http://arxiv.org/pdf/2505.10604v1)**

> **作者:** Chonghan Liu; Haoran Wang; Felix Henry; Pu Miao; Yajie Zhang; Yu Zhao; Peiran Wu
>
> **摘要:** Spatial perception and reasoning are core components of human cognition, encompassing object recognition, spatial relational understanding, and dynamic reasoning. Despite progress in computer vision, existing benchmarks reveal significant gaps in models' abilities to accurately recognize object attributes and reason about spatial relationships, both essential for dynamic reasoning. To address these limitations, we propose MIRAGE, a multi-modal benchmark designed to evaluate models' capabilities in Counting (object attribute recognition), Relation (spatial relational reasoning), and Counting with Relation. Through diverse and complex scenarios requiring fine-grained recognition and reasoning, MIRAGE highlights critical limitations in state-of-the-art models, underscoring the need for improved representations and reasoning frameworks. By targeting these foundational abilities, MIRAGE provides a pathway toward spatiotemporal reasoning in future research.
>
---
#### [new 025] GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding
- **分类: cs.CV**

- **简介: 该论文研究无约束场景下的3D视线估计任务，解决因缺少眼部特写、姿态变化导致的2D-3D映射歧义问题。提出GA3CE方法，通过3D姿态/物体位置构建空间上下文，采用自我中心对齐降低复杂度，并设计方向-距离分解编码强化空间关系建模，在单帧场景下将角度误差降低13%-37%。**

- **链接: [http://arxiv.org/pdf/2505.10671v1](http://arxiv.org/pdf/2505.10671v1)**

> **作者:** Yuki Kawana; Shintaro Shiba; Quan Kong; Norimasa Kobori
>
> **备注:** Accepted to CVPR2025. Project page: https://woven-visionai.github.io/ga3ce-project/
>
> **摘要:** We propose a novel 3D gaze estimation approach that learns spatial relationships between the subject and objects in the scene, and outputs 3D gaze direction. Our method targets unconstrained settings, including cases where close-up views of the subject's eyes are unavailable, such as when the subject is distant or facing away. Previous approaches typically rely on either 2D appearance alone or incorporate limited spatial cues using depth maps in the non-learnable post-processing step. Estimating 3D gaze direction from 2D observations in these scenarios is challenging; variations in subject pose, scene layout, and gaze direction, combined with differing camera poses, yield diverse 2D appearances and 3D gaze directions even when targeting the same 3D scene. To address this issue, we propose GA3CE: Gaze-Aware 3D Context Encoding. Our method represents subject and scene using 3D poses and object positions, treating them as 3D context to learn spatial relationships in 3D space. Inspired by human vision, we align this context in an egocentric space, significantly reducing spatial complexity. Furthermore, we propose D$^3$ (direction-distance-decomposed) positional encoding to better capture the spatial relationship between 3D context and gaze direction in direction and distance space. Experiments demonstrate substantial improvements, reducing mean angle error by 13%-37% compared to leading baselines on benchmark datasets in single-frame settings.
>
---
#### [new 026] CleanPatrick: A Benchmark for Image Data Cleaning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出首个大规模图像数据清理基准CleanPatrick，解决现有基准依赖合成噪声或小规模人工研究的问题。基于皮肤科数据集构建真实场景标注，识别离题/重复/标签错误样本，结合理论模型与专家验证生成高质量真值。通过标准化评估框架对比多种算法，发现自监督方法在重复检测占优，传统方法在有限预算下有效检测离题，标签纠错仍具挑战。发布数据集与框架以推动数据可靠性研究。**

- **链接: [http://arxiv.org/pdf/2505.11034v1](http://arxiv.org/pdf/2505.11034v1)**

> **作者:** Fabian Gröger; Simone Lionetti; Philippe Gottfrois; Alvaro Gonzalez-Jimenez; Ludovic Amruthalingam; Elisabeth Victoria Goessinger; Hanna Lindemann; Marie Bargiela; Marie Hofbauer; Omar Badri; Philipp Tschandl; Arash Koochek; Matthew Groh; Alexander A. Navarini; Marc Pouly
>
> **摘要:** Robust machine learning depends on clean data, yet current image data cleaning benchmarks rely on synthetic noise or narrow human studies, limiting comparison and real-world relevance. We introduce CleanPatrick, the first large-scale benchmark for data cleaning in the image domain, built upon the publicly available Fitzpatrick17k dermatology dataset. We collect 496,377 binary annotations from 933 medical crowd workers, identify off-topic samples (4%), near-duplicates (21%), and label errors (22%), and employ an aggregation model inspired by item-response theory followed by expert review to derive high-quality ground truth. CleanPatrick formalizes issue detection as a ranking task and adopts typical ranking metrics mirroring real audit workflows. Benchmarking classical anomaly detectors, perceptual hashing, SSIM, Confident Learning, NoiseRank, and SelfClean, we find that, on CleanPatrick, self-supervised representations excel at near-duplicate detection, classical methods achieve competitive off-topic detection under constrained review budgets, and label-error detection remains an open challenge for fine-grained medical classification. By releasing both the dataset and the evaluation framework, CleanPatrick enables a systematic comparison of image-cleaning strategies and paves the way for more reliable data-centric artificial intelligence.
>
---
#### [new 027] PSDiffusion: Harmonized Multi-Layer Image Generation via Layout and Appearance Alignment
- **分类: cs.CV**

- **简介: 该论文研究多透明层图像生成任务，解决现有方法无法协调层间布局、物理接触及视觉效果的不足。提出PSDiffusion框架，通过全局层交互机制同步生成多层RGBA图像，单次前馈确保各层质量与全局空间、视觉关联，提升生成协同性。**

- **链接: [http://arxiv.org/pdf/2505.11468v1](http://arxiv.org/pdf/2505.11468v1)**

> **作者:** Dingbang Huang; Wenbo Li; Yifei Zhao; Xinyu Pan; Yanhong Zeng; Bo Dai
>
> **备注:** Project Page: https://github.com/dingbang777/PSDiffusion/
>
> **摘要:** Diffusion models have made remarkable advancements in generating high-quality images from textual descriptions. Recent works like LayerDiffuse have extended the previous single-layer, unified image generation paradigm to transparent image layer generation. However, existing multi-layer generation methods fail to handle the interactions among multiple layers such as rational global layout, physics-plausible contacts and visual effects like shadows and reflections while maintaining high alpha quality. To solve this problem, we propose PSDiffusion, a unified diffusion framework for simultaneous multi-layer text-to-image generation. Our model can automatically generate multi-layer images with one RGB background and multiple RGBA foregrounds through a single feed-forward process. Unlike existing methods that combine multiple tools for post-decomposition or generate layers sequentially and separately, our method introduces a global-layer interactive mechanism that generates layered-images concurrently and collaboratively, ensuring not only high quality and completeness for each layer, but also spatial and visual interactions among layers for global coherence.
>
---
#### [new 028] Are Spatial-Temporal Graph Convolution Networks for Human Action Recognition Over-Parameterized?
- **分类: cs.CV**

- **简介: 该论文针对骨架动作识别任务，研究ST-GCN网络过参数化问题。通过彩票假设实验验证冗余参数存在，提出稀疏生成器构建轻量级网络，并融合多级稀疏结构提升性能。在保持精度的前提下，参数减少95%仅损失1%准确率，多级模型用66%参数实现精度提升>1%。**

- **链接: [http://arxiv.org/pdf/2505.10679v1](http://arxiv.org/pdf/2505.10679v1)**

> **作者:** Jianyang Xie; Yitian Zhao; Yanda Meng; He Zhao; Anh Nguyen; Yalin Zheng
>
> **摘要:** Spatial-temporal graph convolutional networks (ST-GCNs) showcase impressive performance in skeleton-based human action recognition (HAR). However, despite the development of numerous models, their recognition performance does not differ significantly after aligning the input settings. With this observation, we hypothesize that ST-GCNs are over-parameterized for HAR, a conjecture subsequently confirmed through experiments employing the lottery ticket hypothesis. Additionally, a novel sparse ST-GCNs generator is proposed, which trains a sparse architecture from a randomly initialized dense network while maintaining comparable performance levels to the dense components. Moreover, we generate multi-level sparsity ST-GCNs by integrating sparse structures at various sparsity levels and demonstrate that the assembled model yields a significant enhancement in HAR performance. Thorough experiments on four datasets, including NTU-RGB+D 60(120), Kinetics-400, and FineGYM, demonstrate that the proposed sparse ST-GCNs can achieve comparable performance to their dense components. Even with 95% fewer parameters, the sparse ST-GCNs exhibit a degradation of <1% in top-1 accuracy. Meanwhile, the multi-level sparsity ST-GCNs, which require only 66% of the parameters of the dense ST-GCNs, demonstrate an improvement of >1% in top-1 accuracy. The code is available at https://github.com/davelailai/Sparse-ST-GCN.
>
---
#### [new 029] From Embeddings to Accuracy: Comparing Foundation Models for Radiographic Classification
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于医学影像分类任务，旨在评估不同预训练基础模型生成的特征在X光片多分类中的效果。研究比较了六种模型的嵌入性能，使用轻量级适配器进行分类，发现MedImageInsight结合SVM准确率最高（93.8%），同时验证了计算效率与分类公平性。**

- **链接: [http://arxiv.org/pdf/2505.10823v1](http://arxiv.org/pdf/2505.10823v1)**

> **作者:** Xue Li; Jameson Merkow; Noel C. F. Codella; Alberto Santamaria-Pang; Naiteek Sangani; Alexander Ersoy; Christopher Burt; John W. Garrett; Richard J. Bruce; Joshua D. Warner; Tyler Bradshaw; Ivan Tarapov; Matthew P. Lungren; Alan B. McMillan
>
> **备注:** 11 pages, 5 figures, 4 tables
>
> **摘要:** Foundation models, pretrained on extensive datasets, have significantly advanced machine learning by providing robust and transferable embeddings applicable to various domains, including medical imaging diagnostics. This study evaluates the utility of embeddings derived from both general-purpose and medical domain-specific foundation models for training lightweight adapter models in multi-class radiography classification, focusing specifically on tube placement assessment. A dataset comprising 8842 radiographs classified into seven distinct categories was employed to extract embeddings using six foundation models: DenseNet121, BiomedCLIP, Med-Flamingo, MedImageInsight, Rad-DINO, and CXR-Foundation. Adapter models were subsequently trained using classical machine learning algorithms. Among these combinations, MedImageInsight embeddings paired with an support vector machine adapter yielded the highest mean area under the curve (mAUC) at 93.8%, followed closely by Rad-DINO (91.1%) and CXR-Foundation (89.0%). In comparison, BiomedCLIP and DenseNet121 exhibited moderate performance with mAUC scores of 83.0% and 81.8%, respectively, whereas Med-Flamingo delivered the lowest performance at 75.1%. Notably, most adapter models demonstrated computational efficiency, achieving training within one minute and inference within seconds on CPU, underscoring their practicality for clinical applications. Furthermore, fairness analyses on adapters trained on MedImageInsight-derived embeddings indicated minimal disparities, with gender differences in performance within 2% and standard deviations across age groups not exceeding 3%. These findings confirm that foundation model embeddings-especially those from MedImageInsight-facilitate accurate, computationally efficient, and equitable diagnostic classification using lightweight adapters for radiographic image analysis.
>
---
#### [new 030] Completely Weakly Supervised Class-Incremental Learning for Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究完全弱监督的类增量语义分割，旨在仅用图像级标签训练基础/新增类别的分割模型。针对传统方法依赖像素标注、现有弱监督方法不完整的问题，提出基于定位器与基础模型不确定性的伪标签生成方法，并设计样本引导的数据增强缓解灾难性遗忘，在VOC等数据集上超越部分弱监督方法。**

- **链接: [http://arxiv.org/pdf/2505.10781v1](http://arxiv.org/pdf/2505.10781v1)**

> **作者:** David Minkwan Kim; Soeun Lee; Byeongkeun Kang
>
> **备注:** 8 pages
>
> **摘要:** This work addresses the task of completely weakly supervised class-incremental learning for semantic segmentation to learn segmentation for both base and additional novel classes using only image-level labels. While class-incremental semantic segmentation (CISS) is crucial for handling diverse and newly emerging objects in the real world, traditional CISS methods require expensive pixel-level annotations for training. To overcome this limitation, partially weakly-supervised approaches have recently been proposed. However, to the best of our knowledge, this is the first work to introduce a completely weakly-supervised method for CISS. To achieve this, we propose to generate robust pseudo-labels by combining pseudo-labels from a localizer and a sequence of foundation models based on their uncertainty. Moreover, to mitigate catastrophic forgetting, we introduce an exemplar-guided data augmentation method that generates diverse images containing both previous and novel classes with guidance. Finally, we conduct experiments in three common experimental settings: 15-5 VOC, 10-10 VOC, and COCO-to-VOC, and in two scenarios: disjoint and overlap. The experimental results demonstrate that our completely weakly supervised method outperforms even partially weakly supervised methods in the 15-5 VOC and 10-10 VOC settings while achieving competitive accuracy in the COCO-to-VOC setting.
>
---
#### [new 031] MTevent: A Multi-Task Event Camera Dataset for 6D Pose Estimation and Moving Object Detection
- **分类: cs.CV**

- **简介: 该论文提出MTevent数据集，针对高速动态环境中6D姿态估计与移动物体检测任务，解决RGB相机因运动模糊和延迟导致的感知瓶颈。通过立体事件相机和RGB相机采集75个含极端视角、光照变化及遮挡的复杂场景数据，首次整合高速运动、长距离感知与真实物体交互，为事件相机研究提供基准，验证RGB方法在动态场景下的局限性（平均召回率0.22）。**

- **链接: [http://arxiv.org/pdf/2505.11282v1](http://arxiv.org/pdf/2505.11282v1)**

> **作者:** Shrutarv Awasthi; Anas Gouda; Sven Franke; Jérôme Rutinowski; Frank Hoffmann; Moritz Roidl
>
> **备注:** accepted to CVPR 2025 Workshop on Event-based Vision
>
> **摘要:** Mobile robots are reaching unprecedented speeds, with platforms like Unitree B2, and Fraunhofer O3dyn achieving maximum speeds between 5 and 10 m/s. However, effectively utilizing such speeds remains a challenge due to the limitations of RGB cameras, which suffer from motion blur and fail to provide real-time responsiveness. Event cameras, with their asynchronous operation, and low-latency sensing, offer a promising alternative for high-speed robotic perception. In this work, we introduce MTevent, a dataset designed for 6D pose estimation and moving object detection in highly dynamic environments with large detection distances. Our setup consists of a stereo-event camera and an RGB camera, capturing 75 scenes, each on average 16 seconds, and featuring 16 unique objects under challenging conditions such as extreme viewing angles, varying lighting, and occlusions. MTevent is the first dataset to combine high-speed motion, long-range perception, and real-world object interactions, making it a valuable resource for advancing event-based vision in robotics. To establish a baseline, we evaluate the task of 6D pose estimation using NVIDIA's FoundationPose on RGB images, achieving an Average Recall of 0.22 with ground-truth masks, highlighting the limitations of RGB-based approaches in such dynamic settings. With MTevent, we provide a novel resource to improve perception models and foster further research in high-speed robotic vision. The dataset is available for download https://huggingface.co/datasets/anas-gouda/MTevent
>
---
#### [new 032] IMAGE-ALCHEMY: Advancing subject fidelity in personalised text-to-image generation
- **分类: cs.CV**

- **简介: 该论文属于个性化文本到图像生成任务，旨在解决现有方法在生成新主体时存在的遗忘、过拟合和计算量大问题。提出两阶段方法：先通过原模型生成通用场景，再结合LoRA微调与分割驱动流程精准插入个性化主体，既保留模型生成能力又提升主体保真度，DINO相似性达0.789超越现有方案。**

- **链接: [http://arxiv.org/pdf/2505.10743v1](http://arxiv.org/pdf/2505.10743v1)**

> **作者:** Amritanshu Tiwari; Cherish Puniani; Kaustubh Sharma; Ojasva Nema
>
> **备注:** 8 pages
>
> **摘要:** Recent advances in text-to-image diffusion models, particularly Stable Diffusion, have enabled the generation of highly detailed and semantically rich images. However, personalizing these models to represent novel subjects based on a few reference images remains challenging. This often leads to catastrophic forgetting, overfitting, or large computational overhead.We propose a two-stage pipeline that addresses these limitations by leveraging LoRA-based fine-tuning on the attention weights within the U-Net of the Stable Diffusion XL (SDXL) model. First, we use the unmodified SDXL to generate a generic scene by replacing the subject with its class label. Then, we selectively insert the personalized subject through a segmentation-driven image-to-image (Img2Img) pipeline that uses the trained LoRA weights.This framework isolates the subject encoding from the overall composition, thus preserving SDXL's broader generative capabilities while integrating the new subject in a high-fidelity manner. Our method achieves a DINO similarity score of 0.789 on SDXL, outperforming existing personalized text-to-image approaches.
>
---
#### [new 033] SurgPose: Generalisable Surgical Instrument Pose Estimation using Zero-Shot Learning and Stereo Vision
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于机器人辅助手术中的6自由度器械姿态估计任务，旨在解决传统方法因遮挡、反光及泛化性差的问题。通过融合零样本RGB-D模型与立体视觉，改进深度估计和分割模块，实现在未见工具上的高精度姿态估计，提升方法通用性。**

- **链接: [http://arxiv.org/pdf/2505.11439v1](http://arxiv.org/pdf/2505.11439v1)**

> **作者:** Utsav Rai; Haozheng Xu; Stamatia Giannarou
>
> **备注:** To be published in 2025 International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Accurate pose estimation of surgical tools in Robot-assisted Minimally Invasive Surgery (RMIS) is essential for surgical navigation and robot control. While traditional marker-based methods offer accuracy, they face challenges with occlusions, reflections, and tool-specific designs. Similarly, supervised learning methods require extensive training on annotated datasets, limiting their adaptability to new tools. Despite their success in other domains, zero-shot pose estimation models remain unexplored in RMIS for pose estimation of surgical instruments, creating a gap in generalising to unseen surgical tools. This paper presents a novel 6 Degrees of Freedom (DoF) pose estimation pipeline for surgical instruments, leveraging state-of-the-art zero-shot RGB-D models like the FoundationPose and SAM-6D. We advanced these models by incorporating vision-based depth estimation using the RAFT-Stereo method, for robust depth estimation in reflective and textureless environments. Additionally, we enhanced SAM-6D by replacing its instance segmentation module, Segment Anything Model (SAM), with a fine-tuned Mask R-CNN, significantly boosting segmentation accuracy in occluded and complex conditions. Extensive validation reveals that our enhanced SAM-6D surpasses FoundationPose in zero-shot pose estimation of unseen surgical instruments, setting a new benchmark for zero-shot RGB-D pose estimation in RMIS. This work enhances the generalisability of pose estimation for unseen objects and pioneers the application of RGB-D zero-shot methods in RMIS.
>
---
#### [new 034] Efficient Malicious UAV Detection Using Autoencoder-TSMamba Integration
- **分类: cs.CV; cs.CR**

- **简介: 该论文研究恶意无人机检测，属于分类任务。针对下一代网络中无人机威胁（如监控、数据窃取），提出集成自编码器（AE）与Tri-orientated Spatial Mamba架构，通过残差特征提取和ResNet分类器降低复杂度并提升准确率，实验显示召回率达99.8%，优于基准且适合大规模部署。**

- **链接: [http://arxiv.org/pdf/2505.10585v1](http://arxiv.org/pdf/2505.10585v1)**

> **作者:** Azim Akhtarshenas; Ramin Toosi; David López-Pérez; Tohid Alizadeh; Alireza Hosseini
>
> **备注:** 12 pages, 6 figures and 3 tables, accepted in IbPRIA 2025, https://www.ibpria.org/2025/?page=dates
>
> **摘要:** Malicious Unmanned Aerial Vehicles (UAVs) present a significant threat to next-generation networks (NGNs), posing risks such as unauthorized surveillance, data theft, and the delivery of hazardous materials. This paper proposes an integrated (AE)-classifier system to detect malicious UAVs. The proposed AE, based on a 4-layer Tri-orientated Spatial Mamba (TSMamba) architecture, effectively captures complex spatial relationships crucial for identifying malicious UAV activities. The first phase involves generating residual values through the AE, which are subsequently processed by a ResNet-based classifier. This classifier leverages the residual values to achieve lower complexity and higher accuracy. Our experiments demonstrate significant improvements in both binary and multi-class classification scenarios, achieving up to 99.8 % recall compared to 96.7 % in the benchmark. Additionally, our method reduces computational complexity, making it more suitable for large-scale deployment. These results highlight the robustness and scalability of our approach, offering an effective solution for malicious UAV detection in NGN environments.
>
---
#### [new 035] Benchmarking performance, explainability, and evaluation strategies of vision-language models for surgery: Challenges and opportunities
- **分类: cs.CV**

- **简介: 该论文属于医疗AI评估任务，旨在解决通用视觉语言模型(VLMs)在外科领域的应用问题。通过多手术数据集测试，发现现有VLMs在语言-视觉区域关联存在不足，评估了其性能、可解释性及评估策略的适用性。**

- **链接: [http://arxiv.org/pdf/2505.10764v1](http://arxiv.org/pdf/2505.10764v1)**

> **作者:** Jiajun Cheng; Xianwu Zhao; Shan Lin
>
> **摘要:** Minimally invasive surgery (MIS) presents significant visual and technical challenges, including surgical instrument classification and understanding surgical action involving instruments, verbs, and anatomical targets. While many machine learning-based methods have been developed for surgical understanding, they typically rely on procedure- and task-specific models trained on small, manually annotated datasets. In contrast, the recent success of vision-language models (VLMs) trained on large volumes of raw image-text pairs has demonstrated strong adaptability to diverse visual data and a range of downstream tasks. This opens meaningful research questions: how well do these general-purpose VLMs perform in the surgical domain? In this work, we explore those questions by benchmarking several VLMs across diverse surgical datasets, including general laparoscopic procedures and endoscopic submucosal dissection, to assess their current capabilities and limitations. Our benchmark reveals key gaps in the models' ability to consistently link language to the correct regions in surgical scenes.
>
---
#### [new 036] Imputation-free and Alignment-free: Incomplete Multi-view Clustering Driven by Consensus Semantic Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对不完整多视图聚类任务，解决缺失数据导致的原型偏移和跨视图语义不一致问题。现有方法存在两局限：未构建共享语义空间，且过度依赖一致性导致不可靠填补。作者提出无需填补和对齐的FreeCSL框架，通过共识原型学习构建共享语义空间，结合启发式图聚类增强视图内簇结构，提升聚类效果。**

- **链接: [http://arxiv.org/pdf/2505.11182v1](http://arxiv.org/pdf/2505.11182v1)**

> **作者:** Yuzhuo Dai; Jiaqi Jin; Zhibin Dong; Siwei Wang; Xinwang Liu; En Zhu; Xihong Yang; Xinbiao Gan; Yu Feng
>
> **备注:** The paper has been accepted by the 42nd CVPR 2025. The main text has 9 pages, including 8 figures and 4 tables. The appendix has 8 pages, with 10 figures and 6 tables. The reference list has 3 pages
>
> **摘要:** In incomplete multi-view clustering (IMVC), missing data induce prototype shifts within views and semantic inconsistencies across views. A feasible solution is to explore cross-view consistency in paired complete observations, further imputing and aligning the similarity relationships inherently shared across views. Nevertheless, existing methods are constrained by two-tiered limitations: (1) Neither instance- nor cluster-level consistency learning construct a semantic space shared across views to learn consensus semantics. The former enforces cross-view instances alignment, and wrongly regards unpaired observations with semantic consistency as negative pairs; the latter focuses on cross-view cluster counterparts while coarsely handling fine-grained intra-cluster relationships within views. (2) Excessive reliance on consistency results in unreliable imputation and alignment without incorporating view-specific cluster information. Thus, we propose an IMVC framework, imputation- and alignment-free for consensus semantics learning (FreeCSL). To bridge semantic gaps across all observations, we learn consensus prototypes from available data to discover a shared space, where semantically similar observations are pulled closer for consensus semantics learning. To capture semantic relationships within specific views, we design a heuristic graph clustering based on modularity to recover cluster structure with intra-cluster compactness and inter-cluster separation for cluster semantics enhancement. Extensive experiments demonstrate, compared to state-of-the-art competitors, FreeCSL achieves more confident and robust assignments on IMVC task.
>
---
#### [new 037] Rethinking the Mean Teacher Strategy from the Perspective of Self-paced Learning
- **分类: cs.CV**

- **简介: 该论文针对半监督医学图像分割任务，旨在提升模型利用未标注数据的能力。通过将传统均值教师策略重新解释为自定进度学习，提出双师生框架(DTSL)，结合跨架构模型一致性及Jensen-Shannon共识标签生成器，动态调控学习节奏，实验证明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11018v1](http://arxiv.org/pdf/2505.11018v1)**

> **作者:** Pengchen Zhang; Alan J. X. Guo; Sipin Luo; Zhe Han; Lin Guo
>
> **摘要:** Semi-supervised medical image segmentation has attracted significant attention due to its potential to reduce manual annotation costs. The mean teacher (MT) strategy, commonly understood as introducing smoothed, temporally lagged consistency regularization, has demonstrated strong performance across various tasks in this field. In this work, we reinterpret the MT strategy on supervised data as a form of self-paced learning, regulated by the output agreement between the temporally lagged teacher model and the ground truth labels. This idea is further extended to incorporate agreement between a temporally lagged model and a cross-architectural model, which offers greater flexibility in regulating the learning pace and enables application to unlabeled data. Specifically, we propose dual teacher-student learning (DTSL), a framework that introduces two groups of teacher-student models with different architectures. The output agreement between the cross-group teacher and student models is used as pseudo-labels, generated via a Jensen-Shannon divergence-based consensus label generator (CLG). Extensive experiments on popular datasets demonstrate that the proposed method consistently outperforms existing state-of-the-art approaches. Ablation studies further validate the effectiveness of the proposed modules.
>
---
#### [new 038] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Pretraining
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉-语言预训练中假阴性样本导致的对比学习干扰问题，提出FALCON方法。属于跨模态表示学习任务，通过动态调度负样本挖掘策略，在构建训练批次时自适应平衡硬负样本与假阴性的比例，提升跨模态对齐效果。实验证明其在主流框架和下游任务中有效缓解假负面影响。**

- **链接: [http://arxiv.org/pdf/2505.11192v1](http://arxiv.org/pdf/2505.11192v1)**

> **作者:** Myunsoo Kim; Seong-Woong Shim; Byung-Jun Lee
>
> **摘要:** False negatives pose a critical challenge in vision-language pretraining (VLP) due to the many-to-many correspondence between images and texts in large-scale datasets. These false negatives introduce conflicting supervision signals that degrade the learned embedding space and diminish the effectiveness of hard negative sampling. In this paper, we propose FALCON (False-negative Aware Learning of COntrastive Negatives), a learning-based mini-batch construction strategy that adaptively balances the trade-off between hard and false negatives during VLP. Rather than relying on fixed heuristics, FALCON employs a negative mining scheduler that dynamically selects negative samples of appropriate hardness for each anchor instance during mini-batch construction, guided by a proxy for cross-modal alignment improvement. Experimental results demonstrate that FALCON significantly improves performance across two widely adopted VLP frameworks (ALBEF, BLIP-2) and a broad range of downstream tasks and evaluation settings, underscoring its effectiveness and robustness in mitigating the impact of false negatives.
>
---
#### [new 039] Dynamic Base model Shift for Delta Compression
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究模型压缩任务，解决传统delta压缩方法因固定预训练基模型导致高压缩率下性能下降的问题。提出动态基模型迁移(DBMS)方法，通过调整基模型和压缩参数，在极高压缩率下保持微调模型性能，适用于多模态模型且兼容其他压缩技术。**

- **链接: [http://arxiv.org/pdf/2505.11344v1](http://arxiv.org/pdf/2505.11344v1)**

> **作者:** Chenyu Huang; Peng Ye; Shenghe Zheng; Xiaohui Wang; Lei Bai; Tao Chen; Wanli Ouyang
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** Transformer-based models with the pretrain-finetune paradigm bring about significant progress, along with the heavy storage and deployment costs of finetuned models on multiple tasks. Delta compression attempts to lower the costs by reducing the redundancy of delta parameters (i.e., the difference between the finetuned and pre-trained model weights) through pruning or quantization. However, existing methods by default employ the pretrained model as the base model and compress the delta parameters for every task, which may causes significant performance degradation, especially when the compression rate is extremely high. To tackle this issue, we investigate the impact of different base models on the performance of delta compression and find that the pre-trained base model can hardly be optimal. To this end, we propose Dynamic Base Model Shift (DBMS), which dynamically adapts the base model to the target task before performing delta compression. Specifically, we adjust two parameters, which respectively determine the magnitude of the base model shift and the overall scale of delta compression, to boost the compression performance on each task. Through low-cost learning of these two parameters, our DBMS can maintain most of the finetuned model's performance even under an extremely high compression ratio setting, significantly surpassing existing methods. Moreover, our DBMS is orthogonal and can be integrated with a variety of other methods, and it has been evaluated across different types of models including language, vision transformer, and multi-modal models.
>
---
#### [new 040] PhiNet v2: A Mask-Free Brain-Inspired Vision Foundation Model from Video
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于自监督学习的计算机视觉任务，旨在构建更接近生物视觉处理的模型。针对现有方法依赖静态图像和强数据增强的问题，提出PhiNet v2：基于Transformer处理连续视频输入，通过变分推理学习鲁棒表征，无需强增强即达到SOTA性能，推进了生物启发的视觉系统研究。**

- **链接: [http://arxiv.org/pdf/2505.11129v1](http://arxiv.org/pdf/2505.11129v1)**

> **作者:** Makoto Yamada; Kian Ming A. Chai; Ayoub Rhim; Satoki Ishikawa; Mohammad Sabokrou; Yao-Hung Hubert Tsai
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2405.14650
>
> **摘要:** Recent advances in self-supervised learning (SSL) have revolutionized computer vision through innovative architectures and learning objectives, yet they have not fully leveraged insights from biological visual processing systems. Recently, a brain-inspired SSL model named PhiNet was proposed; it is based on a ResNet backbone and operates on static image inputs with strong augmentation. In this paper, we introduce PhiNet v2, a novel Transformer-based architecture that processes temporal visual input (that is, sequences of images) without relying on strong augmentation. Our model leverages variational inference to learn robust visual representations from continuous input streams, similar to human visual processing. Through extensive experimentation, we demonstrate that PhiNet v2 achieves competitive performance compared to state-of-the-art vision foundation models, while maintaining the ability to learn from sequential input without strong data augmentation. This work represents a significant step toward more biologically plausible computer vision systems that process visual information in a manner more closely aligned with human cognitive processes.
>
---
#### [new 041] Unifying Segment Anything in Microscopy with Multimodal Large Language Model
- **分类: cs.CV; 68T99**

- **简介: 该论文属于生物医学图像分割任务，旨在解决显微镜图像跨域分割中模型泛化不足的问题。通过结合多模态大语言模型（MLLMs）向SAM注入视觉-语言知识，提出VLSA模块增强语义对齐，并设计SBR优化边界感知，显著提升了跨域数据集的性能（Dice提升6.79%-7.71%），实现了通用显微分割的统一框架。**

- **链接: [http://arxiv.org/pdf/2505.10769v1](http://arxiv.org/pdf/2505.10769v1)**

> **作者:** Manyu Li; Ruian He; Zixian Zhang; Weimin Tan; Bo Yan
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Accurate segmentation of regions of interest in biomedical images holds substantial value in image analysis. Although several foundation models for biomedical segmentation have currently achieved excellent performance on certain datasets, they typically demonstrate sub-optimal performance on unseen domain data. We owe the deficiency to lack of vision-language knowledge before segmentation. Multimodal Large Language Models (MLLMs) bring outstanding understanding and reasoning capabilities to multimodal tasks, which inspires us to leverage MLLMs to inject Vision-Language Knowledge (VLK), thereby enabling vision models to demonstrate superior generalization capabilities on cross-domain datasets. In this paper, we propose using MLLMs to guide SAM in learning microscopy crose-domain data, unifying Segment Anything in Microscopy, named uLLSAM. Specifically, we propose the Vision-Language Semantic Alignment (VLSA) module, which injects VLK into Segment Anything Model (SAM). We find that after SAM receives global VLK prompts, its performance improves significantly, but there are deficiencies in boundary contour perception. Therefore, we further propose Semantic Boundary Regularization (SBR) to prompt SAM. Our method achieves performance improvements of 7.71% in Dice and 12.10% in SA across 9 in-domain microscopy datasets, achieving state-of-the-art performance. Our method also demonstrates improvements of 6.79% in Dice and 10.08% in SA across 10 out-ofdomain datasets, exhibiting strong generalization capabilities. Code is available at https://github.com/ieellee/uLLSAM.
>
---
#### [new 042] Super-Resolution Generative Adversarial Networks based Video Enhancement
- **分类: cs.CV; cs.AI; eess.IV; I.4.3**

- **简介: 该论文研究视频超分辨率任务，解决传统单图SRGAN缺乏时间连续性的问题。通过扩展SRGAN结构，引入3D非局部模块捕捉时空关联，结合分块训练和数据退化技术提升模型泛化能力，并设计两种模型权衡性能与效率。相比单图方法，其方案增强了时间连贯性，减少伪影，适用于流媒体等视频增强场景。**

- **链接: [http://arxiv.org/pdf/2505.10589v1](http://arxiv.org/pdf/2505.10589v1)**

> **作者:** Kağan ÇETİN
>
> **摘要:** This study introduces an enhanced approach to video super-resolution by extending ordinary Single-Image Super-Resolution (SISR) Super-Resolution Generative Adversarial Network (SRGAN) structure to handle spatio-temporal data. While SRGAN has proven effective for single-image enhancement, its design does not account for the temporal continuity required in video processing. To address this, a modified framework that incorporates 3D Non-Local Blocks is proposed, which is enabling the model to capture relationships across both spatial and temporal dimensions. An experimental training pipeline is developed, based on patch-wise learning and advanced data degradation techniques, to simulate real-world video conditions and learn from both local and global structures and details. This helps the model generalize better and maintain stability across varying video content while maintaining the general structure besides the pixel-wise correctness. Two model variants-one larger and one more lightweight-are presented to explore the trade-offs between performance and efficiency. The results demonstrate improved temporal coherence, sharper textures, and fewer visual artifacts compared to traditional single-image methods. This work contributes to the development of practical, learning-based solutions for video enhancement tasks, with potential applications in streaming, gaming, and digital restoration.
>
---
#### [new 043] Improving Object Detection Performance through YOLOv8: A Comprehensive Training and Evaluation Study
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于目标检测与图像分割任务，旨在提升面部皱纹检测的精度。研究基于YOLOv8模型，通过系统化训练与评估，优化其分割性能，解决现有方法在复杂面部特征中检测效果不足的问题。核心工作包括模型训练调参及多维度性能验证。**

- **链接: [http://arxiv.org/pdf/2505.11424v1](http://arxiv.org/pdf/2505.11424v1)**

> **作者:** Rana Poureskandar; Shiva Razzagzadeh
>
> **摘要:** This study evaluated the performance of a YOLOv8-based segmentation model for detecting and segmenting wrinkles in facial images.
>
---
#### [new 044] Advancing Multiple Instance Learning with Continual Learning for Whole Slide Imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像持续学习任务，针对多示例学习(MIL)模型在动态数据中适应性差、内存效率低的问题，提出注意力知识蒸馏(AKD)和伪袋记忆池(PMP)方法，抑制注意力层遗忘并优化存储，提升WSI诊断模型的准确性和内存效率。**

- **链接: [http://arxiv.org/pdf/2505.10649v1](http://arxiv.org/pdf/2505.10649v1)**

> **作者:** Xianrui Li; Yufei Cui; Jun Li; Antoni B. Chan
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Advances in medical imaging and deep learning have propelled progress in whole slide image (WSI) analysis, with multiple instance learning (MIL) showing promise for efficient and accurate diagnostics. However, conventional MIL models often lack adaptability to evolving datasets, as they rely on static training that cannot incorporate new information without extensive retraining. Applying continual learning (CL) to MIL models is a possible solution, but often sees limited improvements. In this paper, we analyze CL in the context of attention MIL models and find that the model forgetting is mainly concentrated in the attention layers of the MIL model. Using the results of this analysis we propose two components for improving CL on MIL: Attention Knowledge Distillation (AKD) and the Pseudo-Bag Memory Pool (PMP). AKD mitigates catastrophic forgetting by focusing on retaining attention layer knowledge between learning sessions, while PMP reduces the memory footprint by selectively storing only the most informative patches, or ``pseudo-bags'' from WSIs. Experimental evaluations demonstrate that our method significantly improves both accuracy and memory efficiency on diverse WSI datasets, outperforming current state-of-the-art CL methods. This work provides a foundation for CL in large-scale, weakly annotated clinical datasets, paving the way for more adaptable and resilient diagnostic models.
>
---
#### [new 045] Hybrid-Emba3D: Geometry-Aware and Cross-Path Feature Hybrid Enhanced State Space Model for Point Cloud Classification
- **分类: cs.CV**

- **简介: 该论文属于点云分类任务，旨在解决现有方法在局部几何特征提取与模型复杂度平衡上的矛盾。针对Mamba架构的单向依赖与点云无序性冲突，提出双向Hybrid-Emba3D模型，通过几何特征耦合和跨路径特征混合增强局部判别力及长程建模，在ModelNet40达到95.99%准确率。**

- **链接: [http://arxiv.org/pdf/2505.11099v1](http://arxiv.org/pdf/2505.11099v1)**

> **作者:** Bin Liu; Chunyang Wang; Xuelian Liu; Guan Xi; Ge Zhang; Ziteng Yao; Mengxue Dong
>
> **摘要:** The point cloud classification tasks face the dual challenge of efficiently extracting local geometric features while maintaining model complexity. The Mamba architecture utilizes the linear complexity advantage of state space models (SSMs) to overcome the computational bottleneck of Transformers while balancing global modeling capabilities. However, the inherent contradiction between its unidirectional dependency and the unordered nature of point clouds impedes modeling spatial correlation in local neighborhoods, thus constraining geometric feature extraction. This paper proposes Hybrid-Emba3D, a bidirectional Mamba model enhanced by geometry-feature coupling and cross-path feature hybridization. The Local geometric pooling with geometry-feature coupling mechanism significantly enhances local feature discriminative power via coordinated propagation and dynamic aggregation of geometric information between local center points and their neighborhoods, without introducing additional parameters. The designed Collaborative feature enhancer adopts dual-path hybridization, effectively handling local mutations and sparse key signals, breaking through the limitations of traditional SSM long-range modeling. Experimental results demonstrate that the proposed model achieves a new SOTA classification accuracy of 95.99% on ModelNet40 with only 0.03M additional.
>
---
#### [new 046] PoseBench3D: A Cross-Dataset Analysis Framework for 3D Human Pose Estimation
- **分类: cs.CV**

- **简介: 该论文针对3D人体姿态估计模型泛化能力不足的问题，提出统一评估框架PoseBench3D。通过整合四个主流数据集并标准化测试流程，系统分析18种方法在跨数据集场景下的性能（使用MPJPE/PA-MPJPE指标），探究预处理与数据参数对模型泛化的影响，为领域提供可扩展的公平比较基准。**

- **链接: [http://arxiv.org/pdf/2505.10888v1](http://arxiv.org/pdf/2505.10888v1)**

> **作者:** Saad Manzur; Bryan Vela; Brandon Vela; Aditya Agrawal; Lan-Anh Dang-Vu; David Li; Wayne Hayes
>
> **备注:** https://github.com/bryanjvela/PoseLab3D/tree/submission_branch
>
> **摘要:** Reliable three-dimensional human pose estimation is becoming increasingly important for real-world applications, yet much of prior work has focused solely on the performance within a single dataset. In practice, however, systems must adapt to diverse viewpoints, environments, and camera setups -- conditions that differ significantly from those encountered during training, which is often the case in real-world scenarios. To address these challenges, we present a standardized testing environment in which each method is evaluated on a variety of datasets, ensuring consistent and fair cross-dataset comparisons -- allowing for the analysis of methods on previously unseen data. Therefore, we propose PoseBench3D, a unified framework designed to systematically re-evaluate prior and future models across four of the most widely used datasets for human pose estimation -- with the framework able to support novel and future datasets as the field progresses. Through a unified interface, our framework provides datasets in a pre-configured yet easily modifiable format, ensuring compatibility with diverse model architectures. We re-evaluated the work of 18 methods, either trained or gathered from existing literature, and reported results using both Mean Per Joint Position Error (MPJPE) and Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE) metrics, yielding more than 100 novel cross-dataset evaluation results. Additionally, we analyze performance differences resulting from various pre-processing techniques and dataset preparation parameters -- offering further insight into model generalization capabilities.
>
---
#### [new 047] Breaking the Batch Barrier (B3) of Contrastive Learning via Smart Batch Mining
- **分类: cs.CV**

- **简介: 该论文属于对比学习领域，旨在解决训练批次质量依赖性问题。传统方法依赖大批次提供足够负样本，但影响效率。作者提出B3策略，通过预训练模型构建相似性图谱并聚类，生成高密度负样本的小批次，在MMEB基准上以更小批次（64）实现SOTA，性能提升1.3-2.9点。**

- **链接: [http://arxiv.org/pdf/2505.11293v1](http://arxiv.org/pdf/2505.11293v1)**

> **作者:** Raghuveer Thirukovalluru; Rui Meng; Ye Liu; Karthikeyan K; Mingyi Su; Ping Nie; Semih Yavuz; Yingbo Zhou; Wenhu Chen; Bhuwan Dhingra
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Contrastive learning (CL) is a prevalent technique for training embedding models, which pulls semantically similar examples (positives) closer in the representation space while pushing dissimilar ones (negatives) further apart. A key source of negatives are 'in-batch' examples, i.e., positives from other examples in the batch. Effectiveness of such models is hence strongly influenced by the size and quality of training batches. In this work, we propose 'Breaking the Batch Barrier' (B3), a novel batch construction strategy designed to curate high-quality batches for CL. Our approach begins by using a pretrained teacher embedding model to rank all examples in the dataset, from which a sparse similarity graph is constructed. A community detection algorithm is then applied to this graph to identify clusters of examples that serve as strong negatives for one another. The clusters are then used to construct batches that are rich in in-batch negatives. Empirical results on the MMEB multimodal embedding benchmark (36 tasks) demonstrate that our method sets a new state of the art, outperforming previous best methods by +1.3 and +2.9 points at the 7B and 2B model scales, respectively. Notably, models trained with B3 surpass existing state-of-the-art results even with a batch size as small as 64, which is 4-16x smaller than that required by other methods.
>
---
#### [new 048] Pseudo-Label Quality Decoupling and Correction for Semi-Supervised Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文针对半监督实例分割（SSIS）中伪标签质量耦合导致的性能不稳定问题，提出PL-DC框架。任务通过解耦实例级类别/掩码双阈值过滤、动态校正类别混淆及像素级掩码不确定性加权，独立优化分类与分组质量。实验在COCO和Cityscapes上实现SOTA，1%标注数据提升11.6 mAP。**

- **链接: [http://arxiv.org/pdf/2505.11075v1](http://arxiv.org/pdf/2505.11075v1)**

> **作者:** Jianghang Lin; Yilin Lu; Yunhang Shen; Chaoyang Zhu; Shengchuan Zhang; Liujuan Cao; Rongrong Ji
>
> **摘要:** Semi-Supervised Instance Segmentation (SSIS) involves classifying and grouping image pixels into distinct object instances using limited labeled data. This learning paradigm usually faces a significant challenge of unstable performance caused by noisy pseudo-labels of instance categories and pixel masks. We find that the prevalent practice of filtering instance pseudo-labels assessing both class and mask quality with a single score threshold, frequently leads to compromises in the trade-off between the qualities of class and mask labels. In this paper, we introduce a novel Pseudo-Label Quality Decoupling and Correction (PL-DC) framework for SSIS to tackle the above challenges. Firstly, at the instance level, a decoupled dual-threshold filtering mechanism is designed to decouple class and mask quality estimations for instance-level pseudo-labels, thereby independently controlling pixel classifying and grouping qualities. Secondly, at the category level, we introduce a dynamic instance category correction module to dynamically correct the pseudo-labels of instance categories, effectively alleviating category confusion. Lastly, we introduce a pixel-level mask uncertainty-aware mechanism at the pixel level to re-weight the mask loss for different pixels, thereby reducing the impact of noise introduced by pixel-level mask pseudo-labels. Extensive experiments on the COCO and Cityscapes datasets demonstrate that the proposed PL-DC achieves significant performance improvements, setting new state-of-the-art results for SSIS. Notably, our PL-DC shows substantial gains even with minimal labeled data, achieving an improvement of +11.6 mAP with just 1% COCO labeled data and +15.5 mAP with 5% Cityscapes labeled data. The code will be public.
>
---
#### [new 049] EmotionHallucer: Evaluating Emotion Hallucinations in Multimodal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态大语言模型（MLLMs）的幻觉评估任务，旨在解决模型在情绪理解中生成无关/错误内容的问题。作者提出首个情绪幻觉基准EmotionHallucer，结合心理学知识与多模态感知，通过对抗性QA框架评估38个模型，发现普遍存在情绪幻觉且闭源模型更优，最终提出PEP-MEK框架提升检测效果。**

- **链接: [http://arxiv.org/pdf/2505.11405v1](http://arxiv.org/pdf/2505.11405v1)**

> **作者:** Bohao Xing; Xin Liu; Guoying Zhao; Chengyu Liu; Xiaolan Fu; Heikki Kälviäinen
>
> **摘要:** Emotion understanding is a critical yet challenging task. Recent advances in Multimodal Large Language Models (MLLMs) have significantly enhanced their capabilities in this area. However, MLLMs often suffer from hallucinations, generating irrelevant or nonsensical content. To the best of our knowledge, despite the importance of this issue, there has been no dedicated effort to evaluate emotion-related hallucinations in MLLMs. In this work, we introduce EmotionHallucer, the first benchmark for detecting and analyzing emotion hallucinations in MLLMs. Unlike humans, whose emotion understanding stems from the interplay of biology and social learning, MLLMs rely solely on data-driven learning and lack innate emotional instincts. Fortunately, emotion psychology provides a solid foundation of knowledge about human emotions. Building on this, we assess emotion hallucinations from two dimensions: emotion psychology knowledge and real-world multimodal perception. To support robust evaluation, we utilize an adversarial binary question-answer (QA) framework, which employs carefully crafted basic and hallucinated pairs to assess the emotion hallucination tendencies of MLLMs. By evaluating 38 LLMs and MLLMs on EmotionHallucer, we reveal that: i) most current models exhibit substantial issues with emotion hallucinations; ii) closed-source models outperform open-source ones in detecting emotion hallucinations, and reasoning capability provides additional advantages; iii) existing models perform better in emotion psychology knowledge than in multimodal emotion perception. As a byproduct, these findings inspire us to propose the PEP-MEK framework, which yields an average improvement of 9.90% in emotion hallucination detection across selected models. Resources will be available at https://github.com/xxtars/EmotionHallucer.
>
---
#### [new 050] Multi-view dense image matching with similarity learning and geometry priors
- **分类: cs.CV**

- **简介: 该论文研究多视角密集图像匹配与三维重建任务，旨在提升传统方法在跨视角泛化能力。提出MV-DeepSimNets网络，通过融合极线几何先验与深度相似性学习，自动提取几何感知特征，结合平面扫描投影构建正则化代价体，无需多视角训练数据即实现高精度表面重建，适用于航空/卫星影像的跨分辨率场景，已集成至MicMac软件。**

- **链接: [http://arxiv.org/pdf/2505.11264v1](http://arxiv.org/pdf/2505.11264v1)**

> **作者:** Mohamed Ali Chebbi; Ewelina Rupnik; Paul Lopes; Marc Pierrot-Deseilligny
>
> **摘要:** We introduce MV-DeepSimNets, a comprehensive suite of deep neural networks designed for multi-view similarity learning, leveraging epipolar geometry for training. Our approach incorporates an online geometry prior to characterize pixel relationships, either along the epipolar line or through homography rectification. This enables the generation of geometry-aware features from native images, which are then projected across candidate depth hypotheses using plane sweeping. Our method geometric preconditioning effectively adapts epipolar-based features for enhanced multi-view reconstruction, without requiring the laborious multi-view training dataset creation. By aggregating learned similarities, we construct and regularize the cost volume, leading to improved multi-view surface reconstruction over traditional dense matching approaches. MV-DeepSimNets demonstrates superior performance against leading similarity learning networks and end-to-end regression models, especially in terms of generalization capabilities across both aerial and satellite imagery with varied ground sampling distances. Our pipeline is integrated into MicMac software and can be readily adopted in standard multi-resolution image matching pipelines.
>
---
#### [new 051] Automated Detection of Salvin's Albatrosses: Improving Deep Learning Tools for Aerial Wildlife Surveys
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在提升无人机航拍图像中Salvin信天翁的自动识别精度。针对现有通用鸟类检测模型在密集群落监测中的性能不足，研究者评估了BirdDetector模型的零样本检测能力，并通过目标域数据微调和增强图像预处理方法，显著提高了该濒危物种的种群统计准确性。**

- **链接: [http://arxiv.org/pdf/2505.10737v1](http://arxiv.org/pdf/2505.10737v1)**

> **作者:** Mitchell Rogers; Theo Thompson; Isla Duporge; Johannes Fischer; Klemens Pütz; Thomas Mattern; Bing Xue; Mengjie Zhang
>
> **备注:** Accepted to the CV4Animals workshop at CVPR 2025
>
> **摘要:** Recent advancements in deep learning and aerial imaging have transformed wildlife monitoring, enabling researchers to survey wildlife populations at unprecedented scales. Unmanned Aerial Vehicles (UAVs) provide a cost-effective means of capturing high-resolution imagery, particularly for monitoring densely populated seabird colonies. In this study, we assess the performance of a general-purpose avian detection model, BirdDetector, in estimating the breeding population of Salvin's albatross (Thalassarche salvini) on the Bounty Islands, New Zealand. Using drone-derived imagery, we evaluate the model's effectiveness in both zero-shot and fine-tuned settings, incorporating enhanced inference techniques and stronger augmentation methods. Our findings indicate that while applying the model in a zero-shot setting offers a strong baseline, fine-tuning with annotations from the target domain and stronger image augmentation leads to marked improvements in detection accuracy. These results highlight the potential of leveraging pre-trained deep-learning models for species-specific monitoring in remote and challenging environments.
>
---
#### [new 052] Diffusion-NPO: Negative Preference Optimization for Better Preference Aligned Generation of Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对扩散模型生成结果与人类偏好不一致的问题，提出负偏好优化方法（Diffusion-NPO），属于生成模型偏好对齐任务。通过训练专门处理负面/无条件输出的模型，增强分类器无关引导（CFG）效果，无需新策略或数据，仅改进现有技术即可提升SD1.5、SDXL等模型的生成质量。**

- **链接: [http://arxiv.org/pdf/2505.11245v1](http://arxiv.org/pdf/2505.11245v1)**

> **作者:** Fu-Yun Wang; Yunhao Shui; Jingtan Piao; Keqiang Sun; Hongsheng Li
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** Diffusion models have made substantial advances in image generation, yet models trained on large, unfiltered datasets often yield outputs misaligned with human preferences. Numerous methods have been proposed to fine-tune pre-trained diffusion models, achieving notable improvements in aligning generated outputs with human preferences. However, we argue that existing preference alignment methods neglect the critical role of handling unconditional/negative-conditional outputs, leading to a diminished capacity to avoid generating undesirable outcomes. This oversight limits the efficacy of classifier-free guidance~(CFG), which relies on the contrast between conditional generation and unconditional/negative-conditional generation to optimize output quality. In response, we propose a straightforward but versatile effective approach that involves training a model specifically attuned to negative preferences. This method does not require new training strategies or datasets but rather involves minor modifications to existing techniques. Our approach integrates seamlessly with models such as SD1.5, SDXL, video diffusion models and models that have undergone preference optimization, consistently enhancing their alignment with human preferences.
>
---
#### [new 053] Towards Robust and Controllable Text-to-Motion via Masked Autoregressive Diffusion
- **分类: cs.CV; cs.MM; I.3.8**

- **简介: 该论文研究文本生成3D人体运动的任务，旨在解决现有方法泛化性差、控制不灵活的问题。提出MoMADiff框架，通过结合掩码建模与扩散过程，在连续空间实现细粒度帧控制，支持用户指定关键帧调控时空特征，在运动质量、文本对齐和关键帧匹配上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11013v1](http://arxiv.org/pdf/2505.11013v1)**

> **作者:** Zongye Zhang; Bohan Kong; Qingjie Liu; Yunhong Wang
>
> **备注:** 10 pages, 6 figures, 5 tables
>
> **摘要:** Generating 3D human motion from text descriptions remains challenging due to the diverse and complex nature of human motion. While existing methods excel within the training distribution, they often struggle with out-of-distribution motions, limiting their applicability in real-world scenarios. Existing VQVAE-based methods often fail to represent novel motions faithfully using discrete tokens, which hampers their ability to generalize beyond seen data. Meanwhile, diffusion-based methods operating on continuous representations often lack fine-grained control over individual frames. To address these challenges, we propose a robust motion generation framework MoMADiff, which combines masked modeling with diffusion processes to generate motion using frame-level continuous representations. Our model supports flexible user-provided keyframe specification, enabling precise control over both spatial and temporal aspects of motion synthesis. MoMADiff demonstrates strong generalization capability on novel text-to-motion datasets with sparse keyframes as motion prompts. Extensive experiments on two held-out datasets and two standard benchmarks show that our method consistently outperforms state-of-the-art models in motion quality, instruction fidelity, and keyframe adherence.
>
---
#### [new 054] Temporally-Grounded Language Generation: A Benchmark for Real-Time Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对实时视觉语言模型(VLMs)，提出时间同步语言生成任务(TGLG)，解决传统离线VLMs在动态视频流中生成语义准确且时间精准语句的难题。通过构建体育/第一视角数据集和TRACE评估指标，开发了时间同步交织模型VLM-TSI，验证了方法的有效性但揭示了任务挑战性，推动实时VLMs研究。**

- **链接: [http://arxiv.org/pdf/2505.11326v1](http://arxiv.org/pdf/2505.11326v1)**

> **作者:** Keunwoo Peter Yu; Joyce Chai
>
> **备注:** 18 pages
>
> **摘要:** Vision-language models (VLMs) have shown remarkable progress in offline tasks such as image captioning and video question answering. However, real-time interactive environments impose new demands on VLMs, requiring them to generate utterances that are not only semantically accurate but also precisely timed. We identify two core capabilities necessary for such settings -- $\textit{perceptual updating}$ and $\textit{contingency awareness}$ -- and propose a new benchmark task, $\textbf{Temporally-Grounded Language Generation (TGLG)}$, to evaluate them. TGLG requires models to generate utterances in response to streaming video such that both content and timing align with dynamic visual input. To support this benchmark, we curate evaluation datasets from sports broadcasting and egocentric human interaction domains, and introduce a new metric, $\textbf{TRACE}$, to evaluate TGLG by jointly measuring semantic similarity and temporal alignment. Finally, we present $\textbf{Vision-Language Model with Time-Synchronized Interleaving (VLM-TSI)}$, a model that interleaves visual and linguistic tokens in a time-synchronized manner, enabling real-time language generation without relying on turn-based assumptions. Experimental results show that VLM-TSI significantly outperforms a strong baseline, yet overall performance remains modest -- highlighting the difficulty of TGLG and motivating further research in real-time VLMs. Code and data available $\href{https://github.com/yukw777/tglg}{here}$.
>
---
#### [new 055] MARRS: Masked Autoregressive Unit-based Reaction Synthesis
- **分类: cs.CV**

- **简介: 该论文研究人类动作-反应合成任务，解决现有自回归模型因向量量化导致信息丢失及手部精细动作生成不足的问题，提出MARRS框架，通过单元化编码、条件融合与自适应调制实现协调的连续动作生成。**

- **链接: [http://arxiv.org/pdf/2505.11334v1](http://arxiv.org/pdf/2505.11334v1)**

> **作者:** Y. B. Wang; S Wang; J. N. Zhang; J. F. Wu; Q. D. He; C. C. Fu; C. J. Wang; Y. Liu
>
> **摘要:** This work aims at a challenging task: human action-reaction synthesis, i.e., generating human reactions based on the action sequence of the other as conditions. Currently, autoregressive modeling approaches have achieved remarkable performance in motion generation tasks, e.g. text-to-motion. However, vector quantization (VQ) accompanying autoregressive generation has inherent disadvantages, including loss of quantization information, low codebook utilization, etc. Moreover, unlike text-to-motion, which focuses solely on the movement of body joints, human action-reaction synthesis also encompasses fine-grained hand movements. In this work, we propose MARRS, a novel framework designed to generate coordinated and fine-grained reaction motions in continuous representations. Initially, we present the Unit-distinguished Motion Variational AutoEncoder (UD-VAE), which segments the entire body into distinct body and hand units, encoding them independently. Subsequently, we propose Action-Conditioned Fusion (ACF), which involves randomly masking a subset of reactive tokens and extracting specific information about the body and hands from the active tokens. Furthermore, we introduce Adaptive Unit Modulation (AUM) to facilitate interaction between body and hand units by using the information from one unit to adaptively modulate the other. Finally, for the diffusion model, we employ a compact MLP as a noise predictor for each distinct body unit and incorporate the diffusion loss to model the probability distribution of each token. Quantitative and qualitative results demonstrate that our method achieves superior performance. The code will be released upon acceptance.
>
---
#### [new 056] HSRMamba: Efficient Wavelet Stripe State Space Model for Hyperspectral Image Super-Resolution
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对单高光谱图像超分辨率任务，解决现有方法因单向扫描导致生成伪影及模态冲突的问题。提出HSRMamba模型，采用条带扫描降低伪影，结合小波分解协调高低频特征，在保持高效计算的同时提升重建性能，达到SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.11062v1](http://arxiv.org/pdf/2505.11062v1)**

> **作者:** Baisong Li; Xingwang Wang; Haixiao Xu
>
> **摘要:** Single hyperspectral image super-resolution (SHSR) aims to restore high-resolution images from low-resolution hyperspectral images. Recently, the Visual Mamba model has achieved an impressive balance between performance and computational efficiency. However, due to its 1D scanning paradigm, the model may suffer from potential artifacts during image generation. To address this issue, we propose HSRMamba. While maintaining the computational efficiency of Visual Mamba, we introduce a strip-based scanning scheme to effectively reduce artifacts from global unidirectional scanning. Additionally, HSRMamba uses wavelet decomposition to alleviate modal conflicts between high-frequency spatial features and low-frequency spectral features, further improving super-resolution performance. Extensive experiments show that HSRMamba not only excels in reducing computational load and model size but also outperforms existing methods, achieving state-of-the-art results.
>
---
#### [new 057] CROC: Evaluating and Training T2I Metrics with Pseudo- and Human-Labeled Contrastive Robustness Checks
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成（T2I）的评估指标研究，旨在解决现有评价指标鲁棒性差且人工验证成本高的问题。提出CROC框架，通过合成百万级对比样本（CROC$^{syn}$）自动量化指标缺陷，并训练出SOTA指标CROCScore，同时构建人工标注基准（CROC$^{hum}$）验证复杂场景，揭示现有指标在否定语义、身体部位等场景的显著缺陷。**

- **链接: [http://arxiv.org/pdf/2505.11314v1](http://arxiv.org/pdf/2505.11314v1)**

> **作者:** Christoph Leiter; Yuki M. Asano; Margret Keuper; Steffen Eger
>
> **备注:** preprint
>
> **摘要:** The assessment of evaluation metrics (meta-evaluation) is crucial for determining the suitability of existing metrics in text-to-image (T2I) generation tasks. Human-based meta-evaluation is costly and time-intensive, and automated alternatives are scarce. We address this gap and propose CROC: a scalable framework for automated Contrastive Robustness Checks that systematically probes and quantifies metric robustness by synthesizing contrastive test cases across a comprehensive taxonomy of image properties. With CROC, we generate a pseudo-labeled dataset (CROC$^{syn}$) of over one million contrastive prompt-image pairs to enable a fine-grained comparison of evaluation metrics. We also use the dataset to train CROCScore, a new metric that achieves state-of-the-art performance among open-source methods, demonstrating an additional key application of our framework. To complement this dataset, we introduce a human-supervised benchmark (CROC$^{hum}$) targeting especially challenging categories. Our results highlight robustness issues in existing metrics: for example, many fail on prompts involving negation, and all tested open-source metrics fail on at least 25% of cases involving correct identification of body parts.
>
---
#### [new 058] M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection
- **分类: cs.CV**

- **简介: 该论文属于多源遥感数据融合目标检测任务，旨在解决单源光学/SAR图像在复杂环境中检测性能受限及缺乏标准化数据集的问题。提出了首个多分辨率、多极化、多场景的M4-SAR数据集（含11.2万对齐图像对），开发了统一评测工具和E2E-OSDet融合框架，实验表明融合后mAP提升5.7%。**

- **链接: [http://arxiv.org/pdf/2505.10931v1](http://arxiv.org/pdf/2505.10931v1)**

> **作者:** Chao Wang; Wei Lu; Xiang Li; Jian Yang; Lei Luo
>
> **摘要:** Single-source remote sensing object detection using optical or SAR images struggles in complex environments. Optical images offer rich textural details but are often affected by low-light, cloud-obscured, or low-resolution conditions, reducing the detection performance. SAR images are robust to weather, but suffer from speckle noise and limited semantic expressiveness. Optical and SAR images provide complementary advantages, and fusing them can significantly improve the detection accuracy. However, progress in this field is hindered by the lack of large-scale, standardized datasets. To address these challenges, we propose the first comprehensive dataset for optical-SAR fusion object detection, named Multi-resolution, Multi-polarization, Multi-scene, Multi-source SAR dataset (M4-SAR). It contains 112,184 precisely aligned image pairs and nearly one million labeled instances with arbitrary orientations, spanning six key categories. To enable standardized evaluation, we develop a unified benchmarking toolkit that integrates six state-of-the-art multi-source fusion methods. Furthermore, we propose E2E-OSDet, a novel end-to-end multi-source fusion detection framework that mitigates cross-domain discrepancies and establishes a robust baseline for future studies. Extensive experiments on M4-SAR demonstrate that fusing optical and SAR data can improve $mAP$ by 5.7\% over single-source inputs, with particularly significant gains in complex environments. The dataset and code are publicly available at https://github.com/wchao0601/M4-SAR.
>
---
#### [new 059] GaussianFormer3D: Multi-Modal Gaussian-based Semantic Occupancy Prediction with 3D Deformable Attention
- **分类: cs.CV**

- **简介: 该论文研究自动驾驶中的3D语义占用预测任务，旨在解决现有密集网格方法内存消耗大、效率低的问题。提出基于多模态（LiDAR-相机）的高斯表征框架，结合体素初始化策略和可变形注意力机制，在降低资源消耗的同时保持高精度。**

- **链接: [http://arxiv.org/pdf/2505.10685v1](http://arxiv.org/pdf/2505.10685v1)**

> **作者:** Lingjun Zhao; Sizhe Wei; James Hays; Lu Gan
>
> **摘要:** 3D semantic occupancy prediction is critical for achieving safe and reliable autonomous driving. Compared to camera-only perception systems, multi-modal pipelines, especially LiDAR-camera fusion methods, can produce more accurate and detailed predictions. Although most existing works utilize a dense grid-based representation, in which the entire 3D space is uniformly divided into discrete voxels, the emergence of 3D Gaussians provides a compact and continuous object-centric representation. In this work, we propose a multi-modal Gaussian-based semantic occupancy prediction framework utilizing 3D deformable attention, named as GaussianFormer3D. We introduce a voxel-to-Gaussian initialization strategy to provide 3D Gaussians with geometry priors from LiDAR data, and design a LiDAR-guided 3D deformable attention mechanism for refining 3D Gaussians with LiDAR-camera fusion features in a lifted 3D space. We conducted extensive experiments on both on-road and off-road datasets, demonstrating that our GaussianFormer3D achieves high prediction accuracy that is comparable to state-of-the-art multi-modal fusion-based methods with reduced memory consumption and improved efficiency.
>
---
#### [new 060] Patient-Specific Dynamic Digital-Physical Twin for Coronary Intervention Training: An Integrated Mixed Reality Approach
- **分类: cs.CV; cs.HC; 92C50; I.3.8; I.6.8**

- **简介: 该论文属于医疗培训技术开发，旨在解决冠状动脉介入训练中动态心脏模型缺失问题。通过整合4D-CTA数据、数字孪生和混合现实技术，构建了患者特异性动态心脏模型，结合透明硅胶物理模型与虚拟系统，实现了80.9%的形态一致性及精准导丝运动模拟，为手术培训提供动态视觉-触觉反馈环境。**

- **链接: [http://arxiv.org/pdf/2505.10902v1](http://arxiv.org/pdf/2505.10902v1)**

> **作者:** Shuo Wang; Tong Ren; Nan Cheng; Rong Wang; Li Zhang
>
> **备注:** 34 pages, 24 figures
>
> **摘要:** Background and Objective: Precise preoperative planning and effective physician training for coronary interventions are increasingly important. Despite advances in medical imaging technologies, transforming static or limited dynamic imaging data into comprehensive dynamic cardiac models remains challenging. Existing training systems lack accurate simulation of cardiac physiological dynamics. This study develops a comprehensive dynamic cardiac model research framework based on 4D-CTA, integrating digital twin technology, computer vision, and physical model manufacturing to provide precise, personalized tools for interventional cardiology. Methods: Using 4D-CTA data from a 60-year-old female with three-vessel coronary stenosis, we segmented cardiac chambers and coronary arteries, constructed dynamic models, and implemented skeletal skinning weight computation to simulate vessel deformation across 20 cardiac phases. Transparent vascular physical models were manufactured using medical-grade silicone. We developed cardiac output analysis and virtual angiography systems, implemented guidewire 3D reconstruction using binocular stereo vision, and evaluated the system through angiography validation and CABG training applications. Results: Morphological consistency between virtual and real angiography reached 80.9%. Dice similarity coefficients for guidewire motion ranged from 0.741-0.812, with mean trajectory errors below 1.1 mm. The transparent model demonstrated advantages in CABG training, allowing direct visualization while simulating beating heart challenges. Conclusion: Our patient-specific digital-physical twin approach effectively reproduces both anatomical structures and dynamic characteristics of coronary vasculature, offering a dynamic environment with visual and tactile feedback valuable for education and clinical planning.
>
---
#### [new 061] SRMamba: Mamba for Super-Resolution of LiDAR Point Clouds
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出SRMamba方法，属于LiDAR点云超分辨率任务，旨在解决稀疏场景下新视角的三维结构恢复问题。通过Hough投票投影和空洞补偿消除图像缺陷，结合状态空间模型与多向扫描机制增强三维空间特征提取，并设计自适应U-Net支持多光束点云重建，在主流数据集验证了性能优势。**

- **链接: [http://arxiv.org/pdf/2505.10601v1](http://arxiv.org/pdf/2505.10601v1)**

> **作者:** Chuang Chen; Wenyi Ge
>
> **摘要:** In recent years, range-view-based LiDAR point cloud super-resolution techniques attract significant attention as a low-cost method for generating higher-resolution point cloud data. However, due to the sparsity and irregular structure of LiDAR point clouds, the point cloud super-resolution problem remains a challenging topic, especially for point cloud upsampling under novel views. In this paper, we propose SRMamba, a novel method for super-resolution of LiDAR point clouds in sparse scenes, addressing the key challenge of recovering the 3D spatial structure of point clouds from novel views. Specifically, we implement projection technique based on Hough Voting and Hole Compensation strategy to eliminate horizontally linear holes in range image. To improve the establishment of long-distance dependencies and to focus on potential geometric features in vertical 3D space, we employ Visual State Space model and Multi-Directional Scanning mechanism to mitigate the loss of 3D spatial structural information due to the range image. Additionally, an asymmetric U-Net network adapts to the input characteristics of LiDARs with different beam counts, enabling super-resolution reconstruction for multi-beam point clouds. We conduct a series of experiments on multiple challenging public LiDAR datasets (SemanticKITTI and nuScenes), and SRMamba demonstrates significant superiority over other algorithms in both qualitative and quantitative evaluations.
>
---
#### [new 062] CheX-DS: Improving Chest X-ray Image Classification with Ensemble Learning Based on DenseNet and Swin Transformer
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对胸片多标签分类任务，解决现有CNN方法忽略全局特征及数据不平衡问题。提出CheX-DS模型，通过集成DenseNet（提取局部特征）和Swin Transformer（捕捉全局关联），结合加权交叉熵与不对称损失函数优化长尾数据分类，在NIH ChestX-ray14数据集实现83.76%平均AUC，性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11168v1](http://arxiv.org/pdf/2505.11168v1)**

> **作者:** Xinran Li; Yu Liu; Xiujuan Xu; Xiaowei Zhao
>
> **备注:** BIBM
>
> **摘要:** The automatic diagnosis of chest diseases is a popular and challenging task. Most current methods are based on convolutional neural networks (CNNs), which focus on local features while neglecting global features. Recently, self-attention mechanisms have been introduced into the field of computer vision, demonstrating superior performance. Therefore, this paper proposes an effective model, CheX-DS, for classifying long-tail multi-label data in the medical field of chest X-rays. The model is based on the excellent CNN model DenseNet for medical imaging and the newly popular Swin Transformer model, utilizing ensemble deep learning techniques to combine the two models and leverage the advantages of both CNNs and Transformers. The loss function of CheX-DS combines weighted binary cross-entropy loss with asymmetric loss, effectively addressing the issue of data imbalance. The NIH ChestX-ray14 dataset is selected to evaluate the model's effectiveness. The model outperforms previous studies with an excellent average AUC score of 83.76\%, demonstrating its superior performance.
>
---
#### [new 063] CompAlign: Improving Compositional Text-to-Image Generation with a Complex Benchmark and Fine-Grained Feedback
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于组合文本到图像生成任务，旨在解决现有模型对多对象、属性和空间关系描述不准确的问题。提出了包含900复杂提示的基准CompAlign，开发了基于多模态大模型的可解释评估框架CompQuest（分解提示并提供细粒度反馈），并通过反馈信号改进扩散模型的对齐能力。实验表明模型在复杂3D空间关系任务中表现差且开源/闭源模型差距大，经对齐优化后生成准确性显著提升。**

- **链接: [http://arxiv.org/pdf/2505.11178v1](http://arxiv.org/pdf/2505.11178v1)**

> **作者:** Yixin Wan; Kai-Wei Chang
>
> **摘要:** State-of-the-art T2I models are capable of generating high-resolution images given textual prompts. However, they still struggle with accurately depicting compositional scenes that specify multiple objects, attributes, and spatial relations. We present CompAlign, a challenging benchmark with an emphasis on assessing the depiction of 3D-spatial relationships, for evaluating and improving models on compositional image generation. CompAlign consists of 900 complex multi-subject image generation prompts that combine numerical and 3D-spatial relationships with varied attribute bindings. Our benchmark is remarkably challenging, incorporating generation tasks with 3+ generation subjects with complex 3D-spatial relationships. Additionally, we propose CompQuest, an interpretable and accurate evaluation framework that decomposes complex prompts into atomic sub-questions, then utilizes a MLLM to provide fine-grained binary feedback on the correctness of each aspect of generation elements in model-generated images. This enables precise quantification of alignment between generated images and compositional prompts. Furthermore, we propose an alignment framework that uses CompQuest's feedback as preference signals to improve diffusion models' compositional image generation abilities. Using adjustable per-image preferences, our method is easily scalable and flexible for different tasks. Evaluation of 9 T2I models reveals that: (1) models remarkable struggle more with compositional tasks with more complex 3D-spatial configurations, and (2) a noticeable performance gap exists between open-source accessible models and closed-source commercial models. Further empirical study on using CompAlign for model alignment yield promising results: post-alignment diffusion models achieve remarkable improvements in compositional accuracy, especially on complex generation tasks, outperforming previous approaches.
>
---
#### [new 064] A Convolution-Based Gait Asymmetry Metric for Inter-Limb Synergistic Coordination
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于步态分析任务，旨在改进传统基于肌电/加速度的对称性评估方法。通过线性时不变系统建模肢体协同运动，提出卷积度量量化步态不对称性，并在5例对称/非对称步态样本中验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.10869v1](http://arxiv.org/pdf/2505.10869v1)**

> **作者:** Go Fukino; Kanta Tachibana
>
> **备注:** 7 pages, 13 figures, 3 tables
>
> **摘要:** This study focuses on the velocity patterns of various body parts during walking and proposes a method for evaluating gait symmetry. Traditional motion analysis studies have assessed gait symmetry based on differences in electromyographic (EMG) signals or acceleration between the left and right sides. In contrast, this paper models intersegmental coordination using an LTI system and proposes a dissimilarity metric to evaluate symmetry. The method was tested on five subjects with both symmetric and asymmetric gait.
>
---
#### [new 065] Learning Dense Hand Contact Estimation from Imbalanced Data
- **分类: cs.CV**

- **简介: 该论文研究密集手部接触估计任务，解决数据类别不平衡（非接触样本主导）和空间分布不均（接触集中于指尖）问题。提出平衡接触采样策略均衡样本分布，并设计顶点级类平衡损失函数，根据接触频率动态调整各顶点损失权重，有效提升模型对全手区域接触的泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.11152v1](http://arxiv.org/pdf/2505.11152v1)**

> **作者:** Daniel Sungho Jung; Kyoung Mu Lee
>
> **备注:** Project page: http://haco-release.github.io
>
> **摘要:** Hands are essential to human interaction, and understanding contact between hands and the world can promote comprehensive understanding of their function. Recently, there have been growing number of hand interaction datasets that cover interaction with object, other hand, scene, and body. Despite the significance of the task and increasing high-quality data, how to effectively learn dense hand contact estimation remains largely underexplored. There are two major challenges for learning dense hand contact estimation. First, there exists class imbalance issue from hand contact datasets where majority of samples are not in contact. Second, hand contact datasets contain spatial imbalance issue with most of hand contact exhibited in finger tips, resulting in challenges for generalization towards contacts in other hand regions. To tackle these issues, we present a framework that learns dense HAnd COntact estimation (HACO) from imbalanced data. To resolve the class imbalance issue, we introduce balanced contact sampling, which builds and samples from multiple sampling groups that fairly represent diverse contact statistics for both contact and non-contact samples. Moreover, to address the spatial imbalance issue, we propose vertex-level class-balanced (VCB) loss, which incorporates spatially varying contact distribution by separately reweighting loss contribution of each vertex based on its contact frequency across dataset. As a result, we effectively learn to predict dense hand contact estimation with large-scale hand contact data without suffering from class and spatial imbalance issue. The codes will be released.
>
---
#### [new 066] Towards Self-Improvement of Diffusion Models via Group Preference Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像扩散模型偏好优化中DPO方法对数据敏感且依赖人工标注的问题，提出群体偏好优化（GPO）方法。通过将两两比较扩展为群体比较并引入奖励标准化，实现模型自增强，无需外部数据即可提升生成质量（如计数准确率提升20%），属于生成模型自优化任务。**

- **链接: [http://arxiv.org/pdf/2505.11070v1](http://arxiv.org/pdf/2505.11070v1)**

> **作者:** Renjie Chen; Wenfeng Lin; Yichen Zhang; Jiangchuan Wei; Boyuan Liu; Chao Feng; Jiao Ran; Mingyu Guo
>
> **摘要:** Aligning text-to-image (T2I) diffusion models with Direct Preference Optimization (DPO) has shown notable improvements in generation quality. However, applying DPO to T2I faces two challenges: the sensitivity of DPO to preference pairs and the labor-intensive process of collecting and annotating high-quality data. In this work, we demonstrate that preference pairs with marginal differences can degrade DPO performance. Since DPO relies exclusively on relative ranking while disregarding the absolute difference of pairs, it may misclassify losing samples as wins, or vice versa. We empirically show that extending the DPO from pairwise to groupwise and incorporating reward standardization for reweighting leads to performance gains without explicit data selection. Furthermore, we propose Group Preference Optimization (GPO), an effective self-improvement method that enhances performance by leveraging the model's own capabilities without requiring external data. Extensive experiments demonstrate that GPO is effective across various diffusion models and tasks. Specifically, combining with widely used computer vision models, such as YOLO and OCR, the GPO improves the accurate counting and text rendering capabilities of the Stable Diffusion 3.5 Medium by 20 percentage points. Notably, as a plug-and-play method, no extra overhead is introduced during inference.
>
---
#### [new 067] Bias and Generalizability of Foundation Models across Datasets in Breast Mammography
- **分类: cs.CV**

- **简介: 该论文研究乳腺钼靶影像基础模型的公平性与泛化性，属于医学图像分类任务。针对现有模型因数据异质性和虚假相关性导致的性能偏差问题，通过多源数据集（含欠发达地区数据）验证发现：模态预训练虽提升性能，但跨域泛化不足，数据聚合无法完全消除密度/年龄等亚组偏差。对比实验表明公平性优化比域适应策略更稳定，强调需将公平性评估融入模型开发。**

- **链接: [http://arxiv.org/pdf/2505.10579v1](http://arxiv.org/pdf/2505.10579v1)**

> **作者:** Germani Elodie; Selin Türk Ilayda; Zeineddine Fatima; Mourad Charbel; Albarqouni Shadi
>
> **备注:** Accepted at the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2025
>
> **摘要:** Over the past decades, computer-aided diagnosis tools for breast cancer have been developed to enhance screening procedures, yet their clinical adoption remains challenged by data variability and inherent biases. Although foundation models (FMs) have recently demonstrated impressive generalizability and transfer learning capabilities by leveraging vast and diverse datasets, their performance can be undermined by spurious correlations that arise from variations in image quality, labeling uncertainty, and sensitive patient attributes. In this work, we explore the fairness and bias of FMs for breast mammography classification by leveraging a large pool of datasets from diverse sources-including data from underrepresented regions and an in-house dataset. Our extensive experiments show that while modality-specific pre-training of FMs enhances performance, classifiers trained on features from individual datasets fail to generalize across domains. Aggregating datasets improves overall performance, yet does not fully mitigate biases, leading to significant disparities across under-represented subgroups such as extreme breast densities and age groups. Furthermore, while domain-adaptation strategies can reduce these disparities, they often incur a performance trade-off. In contrast, fairness-aware techniques yield more stable and equitable performance across subgroups. These findings underscore the necessity of incorporating rigorous fairness evaluations and mitigation strategies into FM-based models to foster inclusive and generalizable AI.
>
---
#### [new 068] MoCLIP: Motion-Aware Fine-Tuning and Distillation of CLIP for Human Motion Generation
- **分类: cs.CV**

- **简介: 该论文研究文本到人体运动生成任务，解决现有CLIP模型因图像训练导致的运动动态理解不足问题。提出MoCLIP框架，通过在CLIP基础上增加运动编码头，结合对比学习与tethering损失微调模型，提升运动保真度并保持与现有方法的兼容性，实验验证了其在运动对齐指标上的提升效果。**

- **链接: [http://arxiv.org/pdf/2505.10810v1](http://arxiv.org/pdf/2505.10810v1)**

> **作者:** Gabriel Maldonado; Armin Danesh Pazho; Ghazal Alinezhad Noghre; Vinit Katariya; Hamed Tabkhi
>
> **备注:** 11 pages, 5 figures, 2 tables. Presented at the CVPR 2025 Human Motion Generation (HuMoGen) Workshop. Introduces MoCLIP, a CLIP-based fine-tuning strategy for motion generation, with results on HumanML3D dataset and ablation studies
>
> **摘要:** Human motion generation is essential for fields such as animation, robotics, and virtual reality, requiring models that effectively capture motion dynamics from text descriptions. Existing approaches often rely on Contrastive Language-Image Pretraining (CLIP)-based text encoders, but their training on text-image pairs constrains their ability to understand temporal and kinematic structures inherent in motion and motion generation. This work introduces MoCLIP, a fine-tuned CLIP model with an additional motion encoding head, trained on motion sequences using contrastive learning and tethering loss. By explicitly incorporating motion-aware representations, MoCLIP enhances motion fidelity while remaining compatible with existing CLIP-based pipelines and seamlessly integrating into various CLIP-based methods. Experiments demonstrate that MoCLIP improves Top-1, Top-2, and Top-3 accuracy while maintaining competitive FID, leading to improved text-to-motion alignment results. These results highlight MoCLIP's versatility and effectiveness, establishing it as a robust framework for enhancing motion generation.
>
---
#### [new 069] A Light and Smart Wearable Platform with Multimodal Foundation Model for Enhanced Spatial Reasoning in People with Blindness and Low Vision
- **分类: cs.CV**

- **简介: 该论文属于辅助技术任务，解决盲人及低视力者因视觉线索缺失导致的导航与物体定位困难。提出新型多模态大语言模型，通过空间推理增强和眼镜附件硬件，结合视觉语言模型实时解析环境数据，提升空间感知能力。使用VizWiz及自制数据集验证，显著提高导航精度和用户体验。**

- **链接: [http://arxiv.org/pdf/2505.10875v1](http://arxiv.org/pdf/2505.10875v1)**

> **作者:** Alexey Magay; Dhurba Tripathi; Yu Hao; Yi Fang
>
> **备注:** Project website and code: https://dktpt44.github.io/LV-GPT/
>
> **摘要:** People with blindness and low vision (pBLV) face significant challenges, struggling to navigate environments and locate objects due to limited visual cues. Spatial reasoning is crucial for these individuals, as it enables them to understand and interpret the spatial relationships in their surroundings, enhancing their ability to navigate and interact more safely and independently. Current multi-modal large language (MLLM) models for low vision people lack the spatial reasoning capabilities needed to effectively assist in these tasks. Moreover, there is a notable absence of lightweight, easy-to-use systems that allow pBLV to effectively perceive and interact with their surrounding environment. In this paper, we propose a novel spatial enhanced multi-modal large language model based approach for visually impaired individuals. By fine-tuning the MLLM to incorporate spatial reasoning capabilities, our method significantly improves the understanding of environmental context, which is critical for navigation and object recognition. The innovation extends to a hardware component, designed as an attachment for glasses, ensuring increased accessibility and ease of use. This integration leverages advanced VLMs to interpret visual data and provide real-time, spatially aware feedback to the user. Our approach aims to bridge the gap between advanced machine learning models and practical, user-friendly assistive devices, offering a robust solution for visually impaired users to navigate their surroundings more effectively and independently. The paper includes an in-depth evaluation using the VizWiz dataset, demonstrating substantial improvements in accuracy and user experience. Additionally, we design a comprehensive dataset to evaluate our method's effectiveness in realworld situations, demonstrating substantial improvements in accuracy and user experience.
>
---
#### [new 070] DDAE++: Enhancing Diffusion Models Towards Unified Generative and Discriminative Learning
- **分类: cs.CV**

- **简介: 该论文属于生成与判别联合学习的扩散模型优化任务，旨在解决扩散模型生成预训练中特征质量不足及无法自我提升的问题。提出自调节机制，利用去噪网络内部语义指导解码层，形成语义瓶颈以提升生成质量与表征能力，在不损失生成性能的前提下整合对比自蒸馏等判别技术，使模型在生成和识别任务上均超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.10999v1](http://arxiv.org/pdf/2505.10999v1)**

> **作者:** Weilai Xiang; Hongyu Yang; Di Huang; Yunhong Wang
>
> **摘要:** While diffusion models have gained prominence in image synthesis, their generative pre-training has been shown to yield discriminative representations, paving the way towards unified visual generation and understanding. However, two key questions remain: 1) Can these representations be leveraged to improve the training of diffusion models themselves, rather than solely benefiting downstream tasks? 2) Can the feature quality be enhanced to rival or even surpass modern self-supervised learners, without compromising generative capability? This work addresses these questions by introducing self-conditioning, a straightforward yet effective mechanism that internally leverages the rich semantics inherent in denoising network to guide its own decoding layers, forming a tighter bottleneck that condenses high-level semantics to improve generation. Results are compelling: our method boosts both generation FID and recognition accuracy with 1% computational overhead and generalizes across diverse diffusion architectures. Crucially, self-conditioning facilitates an effective integration of discriminative techniques, such as contrastive self-distillation, directly into diffusion models without sacrificing generation quality. Extensive experiments on pixel-space and latent-space datasets show that in linear evaluations, our enhanced diffusion models, particularly UViT and DiT, serve as strong representation learners, surpassing various self-supervised models.
>
---
#### [new 071] Equal is Not Always Fair: A New Perspective on Hyperspectral Representation Non-Uniformity
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对高光谱图像处理中的非均匀性表征问题（如光谱依赖、空间连续性冲突），提出FairHyp框架，通过三个专用模块分别解决空间、特征和光谱维度的非均匀性，在分类/去噪等任务中超越现有方法，重新定义了高维视觉任务的结构公平性需求。**

- **链接: [http://arxiv.org/pdf/2505.11267v1](http://arxiv.org/pdf/2505.11267v1)**

> **作者:** Wuzhou Quan; Mingqiang Wei; Jinhui Tang
>
> **摘要:** Hyperspectral image (HSI) representation is fundamentally challenged by pervasive non-uniformity, where spectral dependencies, spatial continuity, and feature efficiency exhibit complex and often conflicting behaviors. Most existing models rely on a unified processing paradigm that assumes homogeneity across dimensions, leading to suboptimal performance and biased representations. To address this, we propose FairHyp, a fairness-directed framework that explicitly disentangles and resolves the threefold non-uniformity through cooperative yet specialized modules. We introduce a Runge-Kutta-inspired spatial variability adapter to restore spatial coherence under resolution discrepancies, a multi-receptive field convolution module with sparse-aware refinement to enhance discriminative features while respecting inherent sparsity, and a spectral-context state space model that captures stable and long-range spectral dependencies via bidirectional Mamba scanning and statistical aggregation. Unlike one-size-fits-all solutions, FairHyp achieves dimension-specific adaptation while preserving global consistency and mutual reinforcement. This design is grounded in the view that non-uniformity arises from the intrinsic structure of HSI representations, rather than any particular task setting. To validate this, we apply FairHyp across four representative tasks including classification, denoising, super-resolution, and inpaintin, demonstrating its effectiveness in modeling a shared structural flaw. Extensive experiments show that FairHyp consistently outperforms state-of-the-art methods under varied imaging conditions. Our findings redefine fairness as a structural necessity in HSI modeling and offer a new paradigm for balancing adaptability, efficiency, and fidelity in high-dimensional vision tasks.
>
---
#### [new 072] ARFC-WAHNet: Adaptive Receptive Field Convolution and Wavelet-Attentive Hierarchical Network for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文针对红外小目标检测任务，解决复杂场景下小目标特征提取不足和背景干扰问题。提出ARFC-WAHNet网络，通过自适应卷积、小波增强降采样、高低层特征融合和全局注意力机制四个创新模块，提升目标特征表达与噪声抑制能力，在多个数据集上验证了检测精度和鲁棒性优势。**

- **链接: [http://arxiv.org/pdf/2505.10595v1](http://arxiv.org/pdf/2505.10595v1)**

> **作者:** Xingye Cui; Junhai Luo; Jiakun Deng; Kexuan Li; Xiangyu Qiu; Zhenming Peng
>
> **摘要:** Infrared small target detection (ISTD) is critical in both civilian and military applications. However, the limited texture and structural information in infrared images makes accurate detection particularly challenging. Although recent deep learning-based methods have improved performance, their use of conventional convolution kernels limits adaptability to complex scenes and diverse targets. Moreover, pooling operations often cause feature loss and insufficient exploitation of image information. To address these issues, we propose an adaptive receptive field convolution and wavelet-attentive hierarchical network for infrared small target detection (ARFC-WAHNet). This network incorporates a multi-receptive field feature interaction convolution (MRFFIConv) module to adaptively extract discriminative features by integrating multiple convolutional branches with a gated unit. A wavelet frequency enhancement downsampling (WFED) module leverages Haar wavelet transform and frequency-domain reconstruction to enhance target features and suppress background noise. Additionally, we introduce a high-low feature fusion (HLFF) module for integrating low-level details with high-level semantics, and a global median enhancement attention (GMEA) module to improve feature diversity and expressiveness via global attention. Experiments on public datasets SIRST, NUDT-SIRST, and IRSTD-1k demonstrate that ARFC-WAHNet outperforms recent state-of-the-art methods in both detection accuracy and robustness, particularly under complex backgrounds. The code is available at https://github.com/Leaf2001/ARFC-WAHNet.
>
---
#### [new 073] Classifying Shelf Life Quality of Pineapples by Combining Audio and Visual Features
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于多模态分类任务，旨在通过非破坏性方法解决菠萝保质期质量分级问题。研究构建了融合多视角音频（敲击声）和视觉（多角度图像）特征的分类模型，创建了包含500个样本的PQC500数据集，并改进对比学习框架训练跨模态分类器。实验表明音频主导采样策略使模型准确率达84%，优于单模态方法。**

- **链接: [http://arxiv.org/pdf/2505.11020v1](http://arxiv.org/pdf/2505.11020v1)**

> **作者:** Yi-Lu Jiang; Wen-Chang Chang; Ching-Lin Wang; Kung-Liang Hsu; Chih-Yi Chiu
>
> **摘要:** Determining the shelf life quality of pineapples using non-destructive methods is a crucial step to reduce waste and increase income. In this paper, a multimodal and multiview classification model was constructed to classify pineapples into four quality levels based on audio and visual characteristics. For research purposes, we compiled and released the PQC500 dataset consisting of 500 pineapples with two modalities: one was tapping pineapples to record sounds by multiple microphones and the other was taking pictures by multiple cameras at different locations, providing multimodal and multi-view audiovisual features. We modified the contrastive audiovisual masked autoencoder to train the cross-modal-based classification model by abundant combinations of audio and visual pairs. In addition, we proposed to sample a compact size of training data for efficient computation. The experiments were evaluated under various data and model configurations, and the results demonstrated that the proposed cross-modal model trained using audio-major sampling can yield 84% accuracy, outperforming the unimodal models of only audio and only visual by 6% and 18%, respectively.
>
---
#### [new 074] Face Consistency Benchmark for GenAI Video
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于AI视频生成评估任务，旨在解决生成视频中角色外观/属性一致性不足的问题。提出Face Consistency Benchmark（FCB）框架，通过标准化评估指标量化现有模型缺陷，推动改进生成模型的角色连贯性，为提升视频生成可靠性提供基准方法。**

- **链接: [http://arxiv.org/pdf/2505.11425v1](http://arxiv.org/pdf/2505.11425v1)**

> **作者:** Michal Podstawski; Malgorzata Kudelska; Haohong Wang
>
> **摘要:** Video generation driven by artificial intelligence has advanced significantly, enabling the creation of dynamic and realistic content. However, maintaining character consistency across video sequences remains a major challenge, with current models struggling to ensure coherence in appearance and attributes. This paper introduces the Face Consistency Benchmark (FCB), a framework for evaluating and comparing the consistency of characters in AI-generated videos. By providing standardized metrics, the benchmark highlights gaps in existing solutions and promotes the development of more reliable approaches. This work represents a crucial step toward improving character consistency in AI video generation technologies.
>
---
#### [new 075] EA-3DGS: Efficient and Adaptive 3D Gaussians with Highly Enhanced Quality for outdoor scenes
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建与实时渲染任务，旨在解决户外场景中3D高斯点云方法存在的内存占用高、几何结构表达不足及训练效率低的问题。提出基于自适应四面体网格初始化高斯点、高效剪枝与结构感知致密化策略，结合参数量化优化存储，实现高质量实时渲染。**

- **链接: [http://arxiv.org/pdf/2505.10787v1](http://arxiv.org/pdf/2505.10787v1)**

> **作者:** Jianlin Guo; Haihong Xiao; Wenxiong Kang
>
> **摘要:** Efficient scene representations are essential for many real-world applications, especially those involving spatial measurement. Although current NeRF-based methods have achieved impressive results in reconstructing building-scale scenes, they still suffer from slow training and inference speeds due to time-consuming stochastic sampling. Recently, 3D Gaussian Splatting (3DGS) has demonstrated excellent performance with its high-quality rendering and real-time speed, especially for objects and small-scale scenes. However, in outdoor scenes, its point-based explicit representation lacks an effective adjustment mechanism, and the millions of Gaussian points required often lead to memory constraints during training. To address these challenges, we propose EA-3DGS, a high-quality real-time rendering method designed for outdoor scenes. First, we introduce a mesh structure to regulate the initialization of Gaussian components by leveraging an adaptive tetrahedral mesh that partitions the grid and initializes Gaussian components on each face, effectively capturing geometric structures in low-texture regions. Second, we propose an efficient Gaussian pruning strategy that evaluates each 3D Gaussian's contribution to the view and prunes accordingly. To retain geometry-critical Gaussian points, we also present a structure-aware densification strategy that densifies Gaussian points in low-curvature regions. Additionally, we employ vector quantization for parameter quantization of Gaussian components, significantly reducing disk space requirements with only a minimal impact on rendering quality. Extensive experiments on 13 scenes, including eight from four public datasets (MatrixCity-Aerial, Mill-19, Tanks \& Temples, WHU) and five self-collected scenes acquired through UAV photogrammetry measurement from SCUT-CA and plateau regions, further demonstrate the superiority of our method.
>
---
#### [new 076] One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究文本-图像扩散模型的概念擦除任务，解决现有方法依赖文本提示导致效果与可用性失衡的问题。提出首个文本-图像协同擦除框架（Co-Erasing），通过联合文本描述和诱导生成的不良图像进行负向指导，并设计视觉特征精炼策略，在高效移除目标概念的同时减少对其他良性概念的影响，实现效果与可用性的最优平衡。**

- **链接: [http://arxiv.org/pdf/2505.11131v1](http://arxiv.org/pdf/2505.11131v1)**

> **作者:** Feiran Li; Qianqian Xu; Shilong Bao; Zhiyong Yang; Xiaochun Cao; Qingming Huang
>
> **备注:** This paper has been accepeted to ICML 2025. Not Final Version
>
> **摘要:** Concept erasing has recently emerged as an effective paradigm to prevent text-to-image diffusion models from generating visually undesirable or even harmful content. However, current removal methods heavily rely on manually crafted text prompts, making it challenging to achieve a high erasure (efficacy) while minimizing the impact on other benign concepts (usability). In this paper, we attribute the limitations to the inherent gap between the text and image modalities, which makes it hard to transfer the intricately entangled concept knowledge from text prompts to the image generation process. To address this, we propose a novel solution by directly integrating visual supervision into the erasure process, introducing the first text-image Collaborative Concept Erasing (Co-Erasing) framework. Specifically, Co-Erasing describes the concept jointly by text prompts and the corresponding undesirable images induced by the prompts, and then reduces the generating probability of the target concept through negative guidance. This approach effectively bypasses the knowledge gap between text and image, significantly enhancing erasure efficacy. Additionally, we design a text-guided image concept refinement strategy that directs the model to focus on visual features most relevant to the specified text concept, minimizing disruption to other benign concepts. Finally, comprehensive experiments suggest that Co-Erasing outperforms state-of-the-art erasure approaches significantly with a better trade-off between efficacy and usability. Codes are available at https://github.com/Ferry-Li/Co-Erasing.
>
---
#### [new 077] Redundancy-Aware Pretraining of Vision-Language Foundation Models in Remote Sensing
- **分类: cs.CV**

- **简介: 该论文针对遥感领域视觉语言模型（VLM）预训练中多描述文本冗余导致效率低的问题，提出加权特征聚合（WFA）策略。通过非参数唯一性（基于BLEU评分）和注意力机制动态计算描述权重，提取互补信息并抑制冗余，提升文本-图像检索的下游任务性能，同时提供不同场景下的技术选择指导。**

- **链接: [http://arxiv.org/pdf/2505.11121v1](http://arxiv.org/pdf/2505.11121v1)**

> **作者:** Mathis Jürgen Adler; Leonard Hackel; Gencer Sumbul; Begüm Demir
>
> **备注:** Accepted at IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2025. Our code is available at https://git.tu-berlin.de/rsim/redundacy-aware-rs-vlm
>
> **摘要:** The development of foundation models through pretraining of vision-language models (VLMs) has recently attracted great attention in remote sensing (RS). VLM pretraining aims to learn image and language alignments from a large number of image-text pairs. Each pretraining image is often associated with multiple captions containing redundant information due to repeated or semantically similar phrases, resulting in increased pretraining and inference time. To overcome this, we introduce a weighted feature aggregation (WFA) strategy for VLM pretraining in RS. Our strategy aims to extract and exploit complementary information from multiple captions per image while reducing redundancies through feature aggregation with importance weighting. To calculate adaptive importance weights for different captions of each image, we propose two techniques: (i) non-parametric uniqueness and (ii) learning-based attention. In the first technique, importance weights are calculated based on the bilingual evaluation understudy (BLEU) scores of the captions to emphasize unique sentences and reduce the influence of repetitive ones. In the second technique, importance weights are learned through an attention mechanism instead of relying on hand-crafted features. The effectiveness of the proposed WFA strategy with the two techniques is analyzed in terms of downstream performance on text-to-image retrieval in RS. Experimental results show that the proposed strategy enables efficient and effective pretraining of VLMs in RS. Based on the experimental analysis, we derive guidelines for selecting appropriate techniques depending on downstream task requirements and resource constraints. The code of this work is publicly available at https://git.tu-berlin.de/rsim/redundacy-aware-rs-vlm.
>
---
#### [new 078] VISTA: Enhancing Vision-Text Alignment in MLLMs via Cross-Modal Mutual Information Maximization
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（MLLMs）中视觉-文本对齐偏差问题，提出VISTA方法。通过信息论分析揭示传统交叉熵损失导致视觉信息弱化的缺陷，设计显式互信息最大化目标增强跨模态对齐，无需额外模块或数据即提升视觉理解能力，在VQAv2等十余个基准测试中显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2505.10917v1](http://arxiv.org/pdf/2505.10917v1)**

> **作者:** Mingxiao Li; Na Su; Fang Qu; Zhizhou Zhong; Ziyang Chen; Zhaopeng Tu; Xiaolong Li
>
> **摘要:** Current multimodal large language models (MLLMs) face a critical challenge in modality alignment, often exhibiting a bias towards textual information at the expense of other modalities like vision. This paper conducts a systematic information-theoretic analysis of the widely used cross-entropy loss in MLLMs, uncovering its implicit alignment objective. Our theoretical investigation reveals that this implicit objective has inherent limitations, leading to a degradation of cross-modal alignment as text sequence length increases, thereby hindering effective multimodal information fusion. To overcome these drawbacks, we propose Vision-Text Alignment (VISTA), a novel approach guided by our theoretical insights. VISTA introduces an explicit alignment objective designed to maximize cross-modal mutual information, preventing the degradation of visual alignment. Notably, VISTA enhances the visual understanding capabilities of existing MLLMs without requiring any additional trainable modules or extra training data, making it both efficient and practical. Our method significantly outperforms baseline models across more than a dozen benchmark datasets, including VQAv2, MMStar, and MME, paving the way for new directions in MLLM modal alignment research.
>
---
#### [new 079] ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization
- **分类: cs.CV**

- **简介: 该论文属于多领域假图像检测与定位（FIDL）任务，旨在解决领域孤立问题。针对现有各领域（如Deepfake、AIGC等）数据集、模型和评估标准不统一的问题，提出首个全领域统一基准框架ForensicHub，通过模块化架构整合10个基线模型、6种骨干网络及跨领域数据集，并提供8项关键研究洞见，推动领域协同发展。**

- **链接: [http://arxiv.org/pdf/2505.11003v1](http://arxiv.org/pdf/2505.11003v1)**

> **作者:** Bo Du; Xuekang Zhu; Xiaochen Ma; Chenfan Qu; Kaiwen Feng; Zhe Yang; Chi-Man Pun; Jian Liu; Jizhe Zhou
>
> **备注:** Technical report. Code available at: https://github.com/scu-zjz/ForensicHub
>
> **摘要:** The field of Fake Image Detection and Localization (FIDL) is highly fragmented, encompassing four domains: deepfake detection (Deepfake), image manipulation detection and localization (IMDL), artificial intelligence-generated image detection (AIGC), and document image manipulation localization (Doc). Although individual benchmarks exist in some domains, a unified benchmark for all domains in FIDL remains blank. The absence of a unified benchmark results in significant domain silos, where each domain independently constructs its datasets, models, and evaluation protocols without interoperability, preventing cross-domain comparisons and hindering the development of the entire FIDL field. To close the domain silo barrier, we propose ForensicHub, the first unified benchmark & codebase for all-domain fake image detection and localization. Considering drastic variations on dataset, model, and evaluation configurations across all domains, as well as the scarcity of open-sourced baseline models and the lack of individual benchmarks in some domains, ForensicHub: i) proposes a modular and configuration-driven architecture that decomposes forensic pipelines into interchangeable components across datasets, transforms, models, and evaluators, allowing flexible composition across all domains; ii) fully implements 10 baseline models, 6 backbones, 2 new benchmarks for AIGC and Doc, and integrates 2 existing benchmarks of DeepfakeBench and IMDLBenCo through an adapter-based design; iii) conducts indepth analysis based on the ForensicHub, offering 8 key actionable insights into FIDL model architecture, dataset characteristics, and evaluation standards. ForensicHub represents a significant leap forward in breaking the domain silos in the FIDL field and inspiring future breakthroughs.
>
---
#### [new 080] Artifacts of Idiosyncracy in Global Street View Data
- **分类: cs.CV**

- **简介: 该论文研究全球街景数据偏差问题，属于数据质量评估任务。针对城市布局等特质导致密集采样仍存在系统性偏差的现象，作者提出定量分析框架评估28城数据分布，结合阿姆斯特丹案例访谈揭示采集过程对城市表征的影响，旨在从源头识别和修正数据偏差。**

- **链接: [http://arxiv.org/pdf/2505.11046v1](http://arxiv.org/pdf/2505.11046v1)**

> **作者:** Tim Alpherts; Sennay Ghebreab; Nanne van Noord
>
> **备注:** Published at FAccT '25
>
> **摘要:** Street view data is increasingly being used in computer vision applications in recent years. Machine learning datasets are collected for these applications using simple sampling techniques. These datasets are assumed to be a systematic representation of cities, especially when densely sampled. Prior works however, show that there are clear gaps in coverage, with certain cities or regions being covered poorly or not at all. Here we demonstrate that a cities' idiosyncracies, such as city layout, may lead to biases in street view data for 28 cities across the globe, even when they are densely covered. We quantitatively uncover biases in the distribution of coverage of street view data and propose a method for evaluation of such distributions to get better insight in idiosyncracies in a cities' coverage. In addition, we perform a case study of Amsterdam with semi-structured interviews, showing how idiosyncracies of the collection process impact representation of cities and regions and allowing us to address biases at their source.
>
---
#### [new 081] SynRailObs: A Synthetic Dataset for Obstacle Detection in Railway Scenarios
- **分类: cs.CV**

- **简介: 该论文属于铁路障碍物检测任务，旨在解决现有数据集规模不足、标注粗糙的问题。提出高保真合成数据集SynRailObs，利用扩散模型生成复杂天气、地理特征及罕见障碍物，并通过真实场景实验验证其有效性。模型展现跨距离、环境的稳定性能及零样本能力，推动铁路安全研究。**

- **链接: [http://arxiv.org/pdf/2505.10784v1](http://arxiv.org/pdf/2505.10784v1)**

> **作者:** Qiushi Guo; Jason Rambach
>
> **摘要:** Detecting potential obstacles in railway environments is critical for preventing serious accidents. Identifying a broad range of obstacle categories under complex conditions requires large-scale datasets with precisely annotated, high-quality images. However, existing publicly available datasets fail to meet these requirements, thereby hindering progress in railway safety research. To address this gap, we introduce SynRailObs, a high-fidelity synthetic dataset designed to represent a diverse range of weather conditions and geographical features. Furthermore, diffusion models are employed to generate rare and difficult-to-capture obstacles that are typically challenging to obtain in real-world scenarios. To evaluate the effectiveness of SynRailObs, we perform experiments in real-world railway environments, testing on both ballasted and ballastless tracks across various weather conditions. The results demonstrate that SynRailObs holds substantial potential for advancing obstacle detection in railway safety applications. Models trained on this dataset show consistent performance across different distances and environmental conditions. Moreover, the model trained on SynRailObs exhibits zero-shot capabilities, which are essential for applications in security-sensitive domains. The data is available in https://www.kaggle.com/datasets/qiushi910/synrailobs.
>
---
#### [new 082] WildDoc: How Far Are We from Achieving Comprehensive and Robust Document Understanding in the Wild?
- **分类: cs.CV**

- **简介: 该论文属于文档理解评估任务，旨在解决现有基准（如DocVQA）因依赖扫描/数字文档而无法反映真实环境复杂性的问题。作者提出了WildDoc基准，通过人工拍摄真实场景文档（含多条件变体）评估多模态大模型的鲁棒性，发现现有模型在自然环境中性能显著下降。**

- **链接: [http://arxiv.org/pdf/2505.11015v1](http://arxiv.org/pdf/2505.11015v1)**

> **作者:** An-Lan Wang; Jingqun Tang; Liao Lei; Hao Feng; Qi Liu; Xiang Fei; Jinghui Lu; Han Wang; Weiwei Liu; Hao Liu; Yuliang Liu; Xiang Bai; Can Huang
>
> **摘要:** The rapid advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced capabilities in Document Understanding. However, prevailing benchmarks like DocVQA and ChartQA predominantly comprise \textit{scanned or digital} documents, inadequately reflecting the intricate challenges posed by diverse real-world scenarios, such as variable illumination and physical distortions. This paper introduces WildDoc, the inaugural benchmark designed specifically for assessing document understanding in natural environments. WildDoc incorporates a diverse set of manually captured document images reflecting real-world conditions and leverages document sources from established benchmarks to facilitate comprehensive comparisons with digital or scanned documents. Further, to rigorously evaluate model robustness, each document is captured four times under different conditions. Evaluations of state-of-the-art MLLMs on WildDoc expose substantial performance declines and underscore the models' inadequate robustness compared to traditional benchmarks, highlighting the unique challenges posed by real-world document understanding. Our project homepage is available at https://bytedance.github.io/WildDoc.
>
---
#### [new 083] Mapping Semantic Segmentation to Point Clouds Using Structure from Motion for Forest Analysis
- **分类: cs.CV**

- **简介: 该论文属于3D点云语义分割任务，旨在解决森林环境中缺乏基于SfM的带语义标注点云数据集的问题。作者提出新方法：通过森林模拟器生成带语义标签的合成图像，改进开源SfM算法实现语义信息保留，最终生成兼具几何与语义的点云数据，为深度学习模型提供训练资源。**

- **链接: [http://arxiv.org/pdf/2505.10751v1](http://arxiv.org/pdf/2505.10751v1)**

> **作者:** Francisco Raverta Capua; Pablo De Cristoforis
>
> **备注:** Work in progress, accepted in Novel Approaches for Precision Agriculture and Forestry with Autonomous Robots, ICRA 2025 Workshop - May 23, 2025 - Atlanta, GA
>
> **摘要:** Although the use of remote sensing technologies for monitoring forested environments has gained increasing attention, publicly available point cloud datasets remain scarce due to the high costs, sensor requirements, and time-intensive nature of their acquisition. Moreover, as far as we are aware, there are no public annotated datasets generated through Structure From Motion (SfM) algorithms applied to imagery, which may be due to the lack of SfM algorithms that can map semantic segmentation information into an accurate point cloud, especially in a challenging environment like forests. In this work, we present a novel pipeline for generating semantically segmented point clouds of forest environments. Using a custom-built forest simulator, we generate realistic RGB images of diverse forest scenes along with their corresponding semantic segmentation masks. These labeled images are then processed using modified open-source SfM software capable of preserving semantic information during 3D reconstruction. The resulting point clouds provide both geometric and semantic detail, offering a valuable resource for training and evaluating deep learning models aimed at segmenting real forest point clouds obtained via SfM.
>
---
#### [new 084] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出首个长上下文视觉语言模型（LCVLM）评测基准MMLongBench，覆盖5类任务、13,331样本及多类型图像，通过标准化输入长度（8K-128K）评估46个模型，揭示现有模型在长上下文任务中的不足，为改进模型提供诊断基础。**

- **链接: [http://arxiv.org/pdf/2505.10610v1](http://arxiv.org/pdf/2505.10610v1)**

> **作者:** Zhaowei Wang; Wenhao Yu; Xiyu Ren; Jipeng Zhang; Yu Zhao; Rohit Saxena; Liang Cheng; Ginny Wong; Simon See; Pasquale Minervini; Yangqiu Song; Mark Steedman
>
> **备注:** Work in progress
>
> **摘要:** The rapid extension of context windows in large vision-language models has given rise to long-context vision-language models (LCVLMs), which are capable of handling hundreds of images with interleaved text tokens in a single forward pass. In this work, we introduce MMLongBench, the first benchmark covering a diverse set of long-context vision-language tasks, to evaluate LCVLMs effectively and thoroughly. MMLongBench is composed of 13,331 examples spanning five different categories of downstream tasks, such as Visual RAG and Many-Shot ICL. It also provides broad coverage of image types, including various natural and synthetic images. To assess the robustness of the models to different input lengths, all examples are delivered at five standardized input lengths (8K-128K tokens) via a cross-modal tokenization scheme that combines vision patches and text tokens. Through a thorough benchmarking of 46 closed-source and open-source LCVLMs, we provide a comprehensive analysis of the current models' vision-language long-context ability. Our results show that: i) performance on a single task is a weak proxy for overall long-context capability; ii) both closed-source and open-source models face challenges in long-context vision-language tasks, indicating substantial room for future improvement; iii) models with stronger reasoning ability tend to exhibit better long-context performance. By offering wide task coverage, various image types, and rigorous length control, MMLongBench provides the missing foundation for diagnosing and advancing the next generation of LCVLMs.
>
---
#### [new 085] Patho-R1: A Multimodal Reinforcement Learning-Based Pathology Expert Reasoner
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Patho-R1，一种多模态强化学习病理推理模型，解决现有病理视觉语言模型诊断准确性低、推理逻辑薄弱的问题。通过构建高质量推理数据集，采用三阶段训练（知识注入、思维链微调、强化学习优化），提升病理图像与文本的多模态对齐能力，在分类、检索等任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.11404v1](http://arxiv.org/pdf/2505.11404v1)**

> **作者:** Wenchuan Zhang; Penghao Zhang; Jingru Guo; Tao Cheng; Jie Chen; Shuwan Zhang; Zhang Zhang; Yuhao Yi; Hong Bu
>
> **摘要:** Recent advances in vision language models (VLMs) have enabled broad progress in the general medical field. However, pathology still remains a more challenging subdomain, with current pathology specific VLMs exhibiting limitations in both diagnostic accuracy and reasoning plausibility. Such shortcomings are largely attributable to the nature of current pathology datasets, which are primarily composed of image description pairs that lack the depth and structured diagnostic paradigms employed by real world pathologists. In this study, we leverage pathology textbooks and real world pathology experts to construct high-quality, reasoning-oriented datasets. Building on this, we introduce Patho-R1, a multimodal RL-based pathology Reasoner, trained through a three-stage pipeline: (1) continued pretraining on 3.5 million image-text pairs for knowledge infusion; (2) supervised fine-tuning on 500k high-quality Chain-of-Thought samples for reasoning incentivizing; (3) reinforcement learning using Group Relative Policy Optimization and Decoupled Clip and Dynamic sAmpling Policy Optimization strategies for multimodal reasoning quality refinement. To further assess the alignment quality of our dataset, we propose PathoCLIP, trained on the same figure-caption corpus used for continued pretraining. Comprehensive experimental results demonstrate that both PathoCLIP and Patho-R1 achieve robust performance across a wide range of pathology-related tasks, including zero-shot classification, cross-modal retrieval, Visual Question Answering, and Multiple Choice Question. Our project is available at the Patho-R1 repository: https://github.com/Wenchuan-Zhang/Patho-R1.
>
---
#### [new 086] DRAGON: A Large-Scale Dataset of Realistic Images Generated by Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于合成图像检测任务，旨在解决现有数据集覆盖范围有限且易过时的问题。作者构建了DRAGON数据集，包含25个扩散模型的多样化图像，通过大语言模型优化提示生成高真实性图像，并提供多尺寸版本及测试集，支持检测方法开发与性能评估。**

- **链接: [http://arxiv.org/pdf/2505.11257v1](http://arxiv.org/pdf/2505.11257v1)**

> **作者:** Giulia Bertazzini; Daniele Baracchi; Dasara Shullani; Isao Echizen; Alessandro Piva
>
> **摘要:** The remarkable ease of use of diffusion models for image generation has led to a proliferation of synthetic content online. While these models are often employed for legitimate purposes, they are also used to generate fake images that support misinformation and hate speech. Consequently, it is crucial to develop robust tools capable of detecting whether an image has been generated by such models. Many current detection methods, however, require large volumes of sample images for training. Unfortunately, due to the rapid evolution of the field, existing datasets often cover only a limited range of models and quickly become outdated. In this work, we introduce DRAGON, a comprehensive dataset comprising images from 25 diffusion models, spanning both recent advancements and older, well-established architectures. The dataset contains a broad variety of images representing diverse subjects. To enhance image realism, we propose a simple yet effective pipeline that leverages a large language model to expand input prompts, thereby generating more diverse and higher-quality outputs, as evidenced by improvements in standard quality metrics. The dataset is provided in multiple sizes (ranging from extra-small to extra-large) to accomodate different research scenarios. DRAGON is designed to support the forensic community in developing and evaluating detection and attribution techniques for synthetic content. Additionally, the dataset is accompanied by a dedicated test set, intended to serve as a benchmark for assessing the performance of newly developed methods.
>
---
#### [new 087] DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在提升扩散模型的效率。针对扩散Transformer（DiT）计算成本高、自注意力冗余的问题，提出基于卷积网络（ConvNet）的DiCo模型。通过设计紧凑通道注意力机制减少通道冗余，增强特征多样性，在ImageNet上实现更高生成质量与速度（如DiCo-XL比DiT-XL快2.7倍），且无需额外监督。**

- **链接: [http://arxiv.org/pdf/2505.11196v1](http://arxiv.org/pdf/2505.11196v1)**

> **作者:** Yuang Ai; Qihang Fan; Xuefeng Hu; Zhenheng Yang; Ran He; Huaibo Huang
>
> **备注:** 27 pages, 29 figures, 9 tables
>
> **摘要:** Diffusion Transformer (DiT), a promising diffusion model for visual generation, demonstrates impressive performance but incurs significant computational overhead. Intriguingly, analysis of pre-trained DiT models reveals that global self-attention is often redundant, predominantly capturing local patterns-highlighting the potential for more efficient alternatives. In this paper, we revisit convolution as an alternative building block for constructing efficient and expressive diffusion models. However, naively replacing self-attention with convolution typically results in degraded performance. Our investigations attribute this performance gap to the higher channel redundancy in ConvNets compared to Transformers. To resolve this, we introduce a compact channel attention mechanism that promotes the activation of more diverse channels, thereby enhancing feature diversity. This leads to Diffusion ConvNet (DiCo), a family of diffusion models built entirely from standard ConvNet modules, offering strong generative performance with significant efficiency gains. On class-conditional ImageNet benchmarks, DiCo outperforms previous diffusion models in both image quality and generation speed. Notably, DiCo-XL achieves an FID of 2.05 at 256x256 resolution and 2.53 at 512x512, with a 2.7x and 3.1x speedup over DiT-XL/2, respectively. Furthermore, our largest model, DiCo-H, scaled to 1B parameters, reaches an FID of 1.90 on ImageNet 256x256-without any additional supervision during training. Code: https://github.com/shallowdream204/DiCo.
>
---
#### [new 088] What's Inside Your Diffusion Model? A Score-Based Riemannian Metric to Explore the Data Manifold
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成模型几何分析任务，旨在揭示扩散模型学习的数据流形几何结构。针对现有方法难以刻画流形内在几何的问题，作者提出基于分数函数的黎曼度量，通过环境空间张量建立保持切向距离、压缩垂直距离的几何体系，开发高效测地线算法用于图像插值/外推。实验验证其在感知指标(FID/LPIPS)上优于基线，证实了扩散模型隐含的几何表达能力。**

- **链接: [http://arxiv.org/pdf/2505.11128v1](http://arxiv.org/pdf/2505.11128v1)**

> **作者:** Simone Azeglio; Arianna Di Bernardo
>
> **摘要:** Recent advances in diffusion models have demonstrated their remarkable ability to capture complex image distributions, but the geometric properties of the learned data manifold remain poorly understood. We address this gap by introducing a score-based Riemannian metric that leverages the Stein score function from diffusion models to characterize the intrinsic geometry of the data manifold without requiring explicit parameterization. Our approach defines a metric tensor in the ambient space that stretches distances perpendicular to the manifold while preserving them along tangential directions, effectively creating a geometry where geodesics naturally follow the manifold's contours. We develop efficient algorithms for computing these geodesics and demonstrate their utility for both interpolation between data points and extrapolation beyond the observed data distribution. Through experiments on synthetic data with known geometry, Rotated MNIST, and complex natural images via Stable Diffusion, we show that our score-based geodesics capture meaningful transformations that respect the underlying data distribution. Our method consistently outperforms baseline approaches on perceptual metrics (LPIPS) and distribution-level metrics (FID, KID), producing smoother, more realistic image transitions. These results reveal the implicit geometric structure learned by diffusion models and provide a principled way to navigate the manifold of natural images through the lens of Riemannian geometry.
>
---
#### [new 089] A Fourier Space Perspective on Diffusion Models
- **分类: stat.ML; cs.CV; cs.LG; stat.ME**

- **简介: 该论文研究扩散模型的傅里叶空间特性，属于生成模型优化任务。针对标准DDPM前向过程导致高频成分信噪比衰减过快、生成质量下降的问题，作者通过理论分析和实验验证，提出均衡频率噪声化的替代前向过程，在高频主导数据中提升生成效果，标准数据集表现持平。**

- **链接: [http://arxiv.org/pdf/2505.11278v1](http://arxiv.org/pdf/2505.11278v1)**

> **作者:** Fabian Falck; Teodora Pandeva; Kiarash Zahirnia; Rachel Lawrence; Richard Turner; Edward Meeds; Javier Zazo; Sushrut Karmalkar
>
> **摘要:** Diffusion models are state-of-the-art generative models on data modalities such as images, audio, proteins and materials. These modalities share the property of exponentially decaying variance and magnitude in the Fourier domain. Under the standard Denoising Diffusion Probabilistic Models (DDPM) forward process of additive white noise, this property results in high-frequency components being corrupted faster and earlier in terms of their Signal-to-Noise Ratio (SNR) than low-frequency ones. The reverse process then generates low-frequency information before high-frequency details. In this work, we study the inductive bias of the forward process of diffusion models in Fourier space. We theoretically analyse and empirically demonstrate that the faster noising of high-frequency components in DDPM results in violations of the normality assumption in the reverse process. Our experiments show that this leads to degraded generation quality of high-frequency components. We then study an alternate forward process in Fourier space which corrupts all frequencies at the same rate, removing the typical frequency hierarchy during generation, and demonstrate marked performance improvements on datasets where high frequencies are primary, while performing on par with DDPM on standard imaging benchmarks.
>
---
#### [new 090] MOSAIC: A Multi-View 2.5D Organ Slice Selector with Cross-Attentional Reasoning for Anatomically-Aware CT Localization in Medical Organ Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，解决CT多器官分割中计算量大、无关切片多的问题。提出基于视觉语言模型的2.5D多视角切片选择框架，通过交叉注意力机制过滤无关切片，并设计SLC指标评估定位精度，显著降低下游分割成本。**

- **链接: [http://arxiv.org/pdf/2505.10672v1](http://arxiv.org/pdf/2505.10672v1)**

> **作者:** Hania Ghouse; Muzammil Behzad
>
> **摘要:** Efficient and accurate multi-organ segmentation from abdominal CT volumes is a fundamental challenge in medical image analysis. Existing 3D segmentation approaches are computationally and memory intensive, often processing entire volumes that contain many anatomically irrelevant slices. Meanwhile, 2D methods suffer from class imbalance and lack cross-view contextual awareness. To address these limitations, we propose a novel, anatomically-aware slice selector pipeline that reduces input volume prior to segmentation. Our unified framework introduces a vision-language model (VLM) for cross-view organ presence detection using fused tri-slice (2.5D) representations from axial, sagittal, and coronal planes. Our proposed model acts as an "expert" in anatomical localization, reasoning over multi-view representations to selectively retain slices with high structural relevance. This enables spatially consistent filtering across orientations while preserving contextual cues. More importantly, since standard segmentation metrics such as Dice or IoU fail to measure the spatial precision of such slice selection, we introduce a novel metric, Slice Localization Concordance (SLC), which jointly captures anatomical coverage and spatial alignment with organ-centric reference slices. Unlike segmentation-specific metrics, SLC provides a model-agnostic evaluation of localization fidelity. Our model offers substantial improvement gains against several baselines across all organs, demonstrating both accurate and reliable organ-focused slice filtering. These results show that our method enables efficient and spatially consistent organ filtering, thereby significantly reducing downstream segmentation cost while maintaining high anatomical fidelity.
>
---
#### [new 091] ROIsGAN: A Region Guided Generative Adversarial Framework for Murine Hippocampal Subregion Segmentation
- **分类: eess.IV; cs.CV; cs.LG; q-bio.NC**

- **简介: 该论文属于医学图像分割任务，旨在解决小鼠海马体亚区（DG/CA1/CA3）在免疫组化图像中自动化分割的空白。作者提出ROIsGAN框架，通过区域引导对抗网络结合Dice与交叉熵损失优化边界识别，并创建了包含多种染色模态的基准数据集。实验显示模型在复杂染色条件下Dice和IoU指标提升1-11%。**

- **链接: [http://arxiv.org/pdf/2505.10687v1](http://arxiv.org/pdf/2505.10687v1)**

> **作者:** Sayed Mehedi Azim; Brian Corbett; Iman Dehzangi
>
> **摘要:** The hippocampus, a critical brain structure involved in memory processing and various neurodegenerative and psychiatric disorders, comprises three key subregions: the dentate gyrus (DG), Cornu Ammonis 1 (CA1), and Cornu Ammonis 3 (CA3). Accurate segmentation of these subregions from histological tissue images is essential for advancing our understanding of disease mechanisms, developmental dynamics, and therapeutic interventions. However, no existing methods address the automated segmentation of hippocampal subregions from tissue images, particularly from immunohistochemistry (IHC) images. To bridge this gap, we introduce a novel set of four comprehensive murine hippocampal IHC datasets featuring distinct staining modalities: cFos, NeuN, and multiplexed stains combining cFos, NeuN, and either {\Delta}FosB or GAD67, capturing structural, neuronal activity, and plasticity associated information. Additionally, we propose ROIsGAN, a region-guided U-Net-based generative adversarial network tailored for hippocampal subregion segmentation. By leveraging adversarial learning, ROIsGAN enhances boundary delineation and structural detail refinement through a novel region-guided discriminator loss combining Dice and binary cross-entropy loss. Evaluated across DG, CA1, and CA3 subregions, ROIsGAN consistently outperforms conventional segmentation models, achieving performance gains ranging from 1-10% in Dice score and up to 11% in Intersection over Union (IoU), particularly under challenging staining conditions. Our work establishes foundational datasets and methods for automated hippocampal segmentation, enabling scalable, high-precision analysis of tissue images in neuroscience research. Our generated datasets, proposed model as a standalone tool, and its corresponding source code are publicly available at: https://github.com/MehediAzim/ROIsGAN
>
---
#### [new 092] DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人衣物操作任务，旨在解决灵巧双手操作衣物时模拟不真实、数据收集低效及泛化性差的问题。研究者开发了DexGarmentLab环境，包含15种任务的3D模拟资产，提出基于单演示自动生成轨迹的数据集构建方法，并设计分层策略HALO，通过可迁移抓取点识别和通用轨迹生成提升跨衣物形状/变形的泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.11032v1](http://arxiv.org/pdf/2505.11032v1)**

> **作者:** Yuran Wang; Ruihai Wu; Yue Chen; Jiarui Wang; Jiaqi Liang; Ziyu Zhu; Haoran Geng; Jitendra Malik; Pieter Abbeel; Hao Dong
>
> **摘要:** Garment manipulation is a critical challenge due to the diversity in garment categories, geometries, and deformations. Despite this, humans can effortlessly handle garments, thanks to the dexterity of our hands. However, existing research in the field has struggled to replicate this level of dexterity, primarily hindered by the lack of realistic simulations of dexterous garment manipulation. Therefore, we propose DexGarmentLab, the first environment specifically designed for dexterous (especially bimanual) garment manipulation, which features large-scale high-quality 3D assets for 15 task scenarios, and refines simulation techniques tailored for garment modeling to reduce the sim-to-real gap. Previous data collection typically relies on teleoperation or training expert reinforcement learning (RL) policies, which are labor-intensive and inefficient. In this paper, we leverage garment structural correspondence to automatically generate a dataset with diverse trajectories using only a single expert demonstration, significantly reducing manual intervention. However, even extensive demonstrations cannot cover the infinite states of garments, which necessitates the exploration of new algorithms. To improve generalization across diverse garment shapes and deformations, we propose a Hierarchical gArment-manipuLation pOlicy (HALO). It first identifies transferable affordance points to accurately locate the manipulation area, then generates generalizable trajectories to complete the task. Through extensive experiments and detailed analysis of our method and baseline, we demonstrate that HALO consistently outperforms existing methods, successfully generalizing to previously unseen instances even with significant variations in shape and deformation where others fail. Our project page is available at: https://wayrise.github.io/DexGarmentLab/.
>
---
#### [new 093] From Fibers to Cells: Fourier-Based Registration Enables Virtual Cresyl Violet Staining From 3D Polarized Light Imaging
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于跨模态图像配准与虚拟染色任务，旨在解决脑组织切片中纤维与细胞结构图像因染色失真难以对齐的问题。通过结合傅里叶配准和深度学习，直接从3D偏振光成像生成细胞级对齐的虚拟尼氏染色，实现无物理染色的多模态关联分析。**

- **链接: [http://arxiv.org/pdf/2505.11394v1](http://arxiv.org/pdf/2505.11394v1)**

> **作者:** Alexander Oberstrass; Esteban Vaca; Eric Upschulte; Meiqi Niu; Nicola Palomero-Gallagher; David Graessel; Christian Schiffer; Markus Axer; Katrin Amunts; Timo Dickscheid
>
> **摘要:** Comprehensive assessment of the various aspects of the brain's microstructure requires the use of complementary imaging techniques. This includes measuring the spatial distribution of cell bodies (cytoarchitecture) and nerve fibers (myeloarchitecture). The gold standard for cytoarchitectonic analysis is light microscopic imaging of cell-body stained tissue sections. To reveal the 3D orientations of nerve fibers, 3D Polarized Light Imaging (3D-PLI) has been introduced as a reliable technique providing a resolution in the micrometer range while allowing processing of series of complete brain sections. 3D-PLI acquisition is label-free and allows subsequent staining of sections after measurement. By post-staining for cell bodies, a direct link between fiber- and cytoarchitecture can potentially be established within the same section. However, inevitable distortions introduced during the staining process make a nonlinear and cross-modal registration necessary in order to study the detailed relationships between cells and fibers in the images. In addition, the complexity of processing histological sections for post-staining only allows for a limited number of samples. In this work, we take advantage of deep learning methods for image-to-image translation to generate a virtual staining of 3D-PLI that is spatially aligned at the cellular level. In a supervised setting, we build on a unique dataset of brain sections, to which Cresyl violet staining has been applied after 3D-PLI measurement. To ensure high correspondence between both modalities, we address the misalignment of training data using Fourier-based registration methods. In this way, registration can be efficiently calculated during training for local image patches of target and predicted staining. We demonstrate that the proposed method enables prediction of a Cresyl violet staining from 3D-PLI, matching individual cell instances.
>
---
#### [new 094] Preference Isolation Forest for Structure-based Anomaly Detection
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文针对基于结构的异常检测任务，旨在识别不符合低维流形模式的异常样本。提出偏好隔离森林(PIF)框架，结合自适应隔离方法与偏好嵌入技术，通过将数据映射到高维偏好空间实现异常定位，并设计了三种隔离方法：通用型Voronoi-iForest、基于局部敏感哈希的RuzHash-iForest，以及利用局部先验提升效能的Sliding-PIF。**

- **链接: [http://arxiv.org/pdf/2505.10876v1](http://arxiv.org/pdf/2505.10876v1)**

> **作者:** Filippo Leveni; Luca Magri; Cesare Alippi; Giacomo Boracchi
>
> **备注:** Submitted to Pattern Recognition
>
> **摘要:** We address the problem of detecting anomalies as samples that do not conform to structured patterns represented by low-dimensional manifolds. To this end, we conceive a general anomaly detection framework called Preference Isolation Forest (PIF), that combines the benefits of adaptive isolation-based methods with the flexibility of preference embedding. The key intuition is to embed the data into a high-dimensional preference space by fitting low-dimensional manifolds, and to identify anomalies as isolated points. We propose three isolation approaches to identify anomalies: $i$) Voronoi-iForest, the most general solution, $ii$) RuzHash-iForest, that avoids explicit computation of distances via Local Sensitive Hashing, and $iii$) Sliding-PIF, that leverages a locality prior to improve efficiency and effectiveness.
>
---
#### [new 095] ExploreGS: a vision-based low overhead framework for 3D scene reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在降低无人机重建成本并提升实时性。针对传统激光雷达方案昂贵、计算量大问题，提出ExploreGS视觉框架，利用RGB图像和BoW模型实现低开销实时处理，集成场景探索与3D高斯重建，验证了在受限设备的高效性与质量。**

- **链接: [http://arxiv.org/pdf/2505.10578v1](http://arxiv.org/pdf/2505.10578v1)**

> **作者:** Yunji Feng; Chengpu Yu; Fengrui Ran; Zhi Yang; Yinni Liu
>
> **摘要:** This paper proposes a low-overhead, vision-based 3D scene reconstruction framework for drones, named ExploreGS. By using RGB images, ExploreGS replaces traditional lidar-based point cloud acquisition process with a vision model, achieving a high-quality reconstruction at a lower cost. The framework integrates scene exploration and model reconstruction, and leverags a Bag-of-Words(BoW) model to enable real-time processing capabilities, therefore, the 3D Gaussian Splatting (3DGS) training can be executed on-board. Comprehensive experiments in both simulation and real-world environments demonstrate the efficiency and applicability of the ExploreGS framework on resource-constrained devices, while maintaining reconstruction quality comparable to state-of-the-art methods.
>
---
#### [new 096] Maximizing Asynchronicity in Event-based Neural Networks
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于事件相机数据处理任务，旨在解决异步稀疏事件流难以适配传统同步机器学习框架的问题。提出了EVA框架，通过借鉴语言模型的线性注意力与自监督学习，将事件逐次编码为高表达力、强泛化的同步表示，在识别和检测任务中性能超越现有方法，最高达47.7 mAP。**

- **链接: [http://arxiv.org/pdf/2505.11165v1](http://arxiv.org/pdf/2505.11165v1)**

> **作者:** Haiqing Hao; Nikola Zubić; Weihua He; Zhipeng Sui; Davide Scaramuzza; Wenhui Wang
>
> **备注:** 18 pages, 5 figures, 9 tables
>
> **摘要:** Event cameras deliver visual data with high temporal resolution, low latency, and minimal redundancy, yet their asynchronous, sparse sequential nature challenges standard tensor-based machine learning (ML). While the recent asynchronous-to-synchronous (A2S) paradigm aims to bridge this gap by asynchronously encoding events into learned representations for ML pipelines, existing A2S approaches often sacrifice representation expressivity and generalizability compared to dense, synchronous methods. This paper introduces EVA (EVent Asynchronous representation learning), a novel A2S framework to generate highly expressive and generalizable event-by-event representations. Inspired by the analogy between events and language, EVA uniquely adapts advances from language modeling in linear attention and self-supervised learning for its construction. In demonstration, EVA outperforms prior A2S methods on recognition tasks (DVS128-Gesture and N-Cars), and represents the first A2S framework to successfully master demanding detection tasks, achieving a remarkable 47.7 mAP on the Gen1 dataset. These results underscore EVA's transformative potential for advancing real-time event-based vision applications.
>
---
#### [new 097] TartanGround: A Large-Scale Dataset for Ground Robot Perception and Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出TartanGround数据集，针对地面机器人感知与导航任务，解决现有数据缺乏多样性和泛化性问题。通过仿真环境采集多模态数据（RGB、深度、LiDAR等），包含910条轨迹、150万样本，模拟不同机器人运动模式。验证显示现有方法跨场景性能不足，该数据集支持SLAM、占据预测等任务，旨在提升模型鲁棒性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.10696v1](http://arxiv.org/pdf/2505.10696v1)**

> **作者:** Manthan Patel; Fan Yang; Yuheng Qiu; Cesar Cadena; Sebastian Scherer; Marco Hutter; Wenshan Wang
>
> **备注:** Under review for IEEE conference
>
> **摘要:** We present TartanGround, a large-scale, multi-modal dataset to advance the perception and autonomy of ground robots operating in diverse environments. This dataset, collected in various photorealistic simulation environments includes multiple RGB stereo cameras for 360-degree coverage, along with depth, optical flow, stereo disparity, LiDAR point clouds, ground truth poses, semantic segmented images, and occupancy maps with semantic labels. Data is collected using an integrated automatic pipeline, which generates trajectories mimicking the motion patterns of various ground robot platforms, including wheeled and legged robots. We collect 910 trajectories across 70 environments, resulting in 1.5 million samples. Evaluations on occupancy prediction and SLAM tasks reveal that state-of-the-art methods trained on existing datasets struggle to generalize across diverse scenes. TartanGround can serve as a testbed for training and evaluation of a broad range of learning-based tasks, including occupancy prediction, SLAM, neural scene representation, perception-based navigation, and more, enabling advancements in robotic perception and autonomy towards achieving robust models generalizable to more diverse scenarios. The dataset and codebase for data collection will be made publicly available upon acceptance. Webpage: https://tartanair.org/tartanground
>
---
#### [new 098] GRNN:Recurrent Neural Network based on Ghost Features for Video Super-Resolution
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对视频超分辨率任务，解决传统卷积神经网络计算成本高、特征冗余问题。提出结合Ghost模块（减少冗余特征）和循环神经网络（缓解梯度消失），利用相邻帧及历史信息建模时序，在提升PSNR/SSIM指标的同时保留纹理细节。**

- **链接: [http://arxiv.org/pdf/2505.10577v1](http://arxiv.org/pdf/2505.10577v1)**

> **作者:** Yutong Guo
>
> **备注:** Accepted by 2023 IEEE International Conference on Multimedia and Expo (ICME 2023)
>
> **摘要:** Modern video super-resolution (VSR) systems based on convolutional neural networks (CNNs) require huge computational costs. The problem of feature redundancy is present in most models in many domains, but is rarely discussed in VSR. We experimentally observe that many features in VSR models are also similar to each other, so we propose to use "Ghost features" to reduce this redundancy. We also analyze the so-called "gradient disappearance" phenomenon generated by the conventional recurrent convolutional network (RNN) model, and combine the Ghost module with RNN to complete the modeling on time series. The current frame is used as input to the model together with the next frame, the output of the previous frame and the hidden state. Extensive experiments on several benchmark models and datasets show that the PSNR and SSIM of our proposed modality are improved to some extent. Some texture details in the video are also better preserved.
>
---
#### [new 099] MultiLink: Multi-class Structure Recovery via Agglomerative Clustering and Model Selection
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究多类结构恢复任务，解决含噪声数据中同时拟合多种参数模型（如平面、圆柱）的鲁棒性问题。提出MultiLink算法，通过凝聚聚类与在线模型选择合并类簇，提升速度、降低阈值敏感性并减少采样偏差，优于现有方法。实验验证了其在多类及单类问题中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.10874v1](http://arxiv.org/pdf/2505.10874v1)**

> **作者:** Luca Magri; Filippo Leveni; Giacomo Boracchi
>
> **备注:** Accepted at Computer Vision and Pattern Recognition (CVPR 2021)
>
> **摘要:** We address the problem of recovering multiple structures of different classes in a dataset contaminated by noise and outliers. In particular, we consider geometric structures defined by a mixture of underlying parametric models (e.g. planes and cylinders, homographies and fundamental matrices), and we tackle the robust fitting problem by preference analysis and clustering. We present a new algorithm, termed MultiLink, that simultaneously deals with multiple classes of models. MultiLink combines on-the-fly model fitting and model selection in a novel linkage scheme that determines whether two clusters are to be merged. The resulting method features many practical advantages with respect to methods based on preference analysis, being faster, less sensitive to the inlier threshold, and able to compensate limitations deriving from hypotheses sampling. Experiments on several public datasets demonstrate that Multi-Link favourably compares with state of the art alternatives, both in multi-class and single-class problems. Code is publicly made available for download.
>
---
#### [new 100] Visual Planning: Let's Think Only with Images
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出视觉规划任务，解决现有模型依赖文本进行空间推理的问题。通过纯视觉表示（图像序列）执行规划，开发强化学习框架VPRL训练视觉模型，在FrozenLake等导航任务中超越文本推理方法，验证了图像推理的有效性。**

- **链接: [http://arxiv.org/pdf/2505.11409v1](http://arxiv.org/pdf/2505.11409v1)**

> **作者:** Yi Xu; Chengzu Li; Han Zhou; Xingchen Wan; Caiqi Zhang; Anna Korhonen; Ivan Vulić
>
> **备注:** 10 pages, 6 figures, 1 table (26 pages, 12 figures, 8 tables including references and appendices)
>
> **摘要:** Recent advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have substantially enhanced machine reasoning across diverse tasks. However, these models predominantly rely on pure text as the medium for both expressing and structuring reasoning, even when visual information is present. In this work, we argue that language may not always be the most natural or effective modality for reasoning, particularly in tasks involving spatial and geometrical information. Motivated by this, we propose a new paradigm, Visual Planning, which enables planning through purely visual representations, independent of text. In this paradigm, planning is executed via sequences of images that encode step-by-step inference in the visual domain, akin to how humans sketch or visualize future actions. We introduce a novel reinforcement learning framework, Visual Planning via Reinforcement Learning (VPRL), empowered by GRPO for post-training large vision models, leading to substantial improvements in planning in a selection of representative visual navigation tasks, FrozenLake, Maze, and MiniBehavior. Our visual planning paradigm outperforms all other planning variants that conduct reasoning in the text-only space. Our results establish Visual Planning as a viable and promising alternative to language-based reasoning, opening new avenues for tasks that benefit from intuitive, image-based inference.
>
---
#### [new 101] Adaptive Spatial Transcriptomics Interpolation via Cross-modal Cross-slice Modeling
- **分类: eess.IV; cs.CV; q-bio.QM**

- **简介: 该论文提出C2-STi模型，解决空间转录组学(ST)切片缺失的插值问题。通过跨模态跨切片建模，设计三个模块：距离感知结构调节捕获切片形变，金字塔基因共表达建模多尺度关联，跨模态对齐整合H&E图像特征，实现任意位置ST切片生成，提升3D空间基因表达分析的可行性。**

- **链接: [http://arxiv.org/pdf/2505.10729v1](http://arxiv.org/pdf/2505.10729v1)**

> **作者:** NingFeng Que; Xiaofei Wang; Jingjing Chen; Yixuan Jiang; Chao Li
>
> **备注:** Early accepted by MICCAI 2025
>
> **摘要:** Spatial transcriptomics (ST) is a promising technique that characterizes the spatial gene profiling patterns within the tissue context. Comprehensive ST analysis depends on consecutive slices for 3D spatial insights, whereas the missing intermediate tissue sections and high costs limit the practical feasibility of generating multi-slice ST. In this paper, we propose C2-STi, the first attempt for interpolating missing ST slices at arbitrary intermediate positions between adjacent ST slices. Despite intuitive, effective ST interpolation presents significant challenges, including 1) limited continuity across heterogeneous tissue sections, 2) complex intrinsic correlation across genes, and 3) intricate cellular structures and biological semantics within each tissue section. To mitigate these challenges, in C2-STi, we design 1) a distance-aware local structural modulation module to adaptively capture cross-slice deformations and enhance positional correlations between ST slices, 2) a pyramid gene co-expression correlation module to capture multi-scale biological associations among genes, and 3) a cross-modal alignment module that integrates the ST-paired hematoxylin and eosin (H&E)-stained images to filter and align the essential cellular features across ST and H\&E images. Extensive experiments on the public dataset demonstrate our superiority over state-of-the-art approaches on both single-slice and multi-slice ST interpolation. Codes are available at https://github.com/XiaofeiWang2018/C2-STi.
>
---
#### [new 102] A probabilistic framework for dynamic quantization
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于神经网络压缩任务，旨在解决静态量化方法无法自适应调整参数导致性能下降的问题。作者提出了动态量化概率框架，通过轻量级代理模型分析预激活分布，实现输入自适应的量化参数调整，在计算机视觉任务中验证了该方法在保持精度的同时显著降低计算开销。**

- **链接: [http://arxiv.org/pdf/2505.10689v1](http://arxiv.org/pdf/2505.10689v1)**

> **作者:** Gabriele Santini; Francesco Paissan; Elisabetta Farella
>
> **摘要:** We propose a probabilistic framework for dynamic quantization of neural networks that allows for a computationally efficient input-adaptive rescaling of the quantization parameters. Our framework applies a probabilistic model to the network's pre-activations through a lightweight surrogate, enabling the adaptive adjustment of the quantization parameters on a per-input basis without significant memory overhead. We validate our approach on a set of popular computer vision tasks and models, observing only a negligible loss in performance. Our method strikes the best performance and computational overhead tradeoff compared to standard quantization strategies.
>
---
#### [new 103] Pretrained hybrid transformer for generalizable cardiac substructures segmentation from contrast and non-contrast CTs in lung and breast cancers
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决AI模型在跨临床场景（不同CT对比度、患者体位）下心脏亚结构分割性能下降问题。研究者提出混合Transformer卷积网络（HTN），通过预训练和平衡数据策略，使用有限标注数据实现了对肺癌/乳腺癌患者CT图像中心脏结构的鲁棒分割，在几何精度和放疗剂量指标上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.10855v1](http://arxiv.org/pdf/2505.10855v1)**

> **作者:** Aneesh Rangnekar; Nikhil Mankuzhy; Jonas Willmann; Chloe Choi; Abraham Wu; Maria Thor; Andreas Rimner; Harini Veeraraghavan
>
> **摘要:** AI automated segmentations for radiation treatment planning (RTP) can deteriorate when applied in clinical cases with different characteristics than training dataset. Hence, we refined a pretrained transformer into a hybrid transformer convolutional network (HTN) to segment cardiac substructures lung and breast cancer patients acquired with varying imaging contrasts and patient scan positions. Cohort I, consisting of 56 contrast-enhanced (CECT) and 124 non-contrast CT (NCCT) scans from patients with non-small cell lung cancers acquired in supine position, was used to create oracle with all 180 training cases and balanced (CECT: 32, NCCT: 32 training) HTN models. Models were evaluated on a held-out validation set of 60 cohort I patients and 66 patients with breast cancer from cohort II acquired in supine (n=45) and prone (n=21) positions. Accuracy was measured using DSC, HD95, and dose metrics. Publicly available TotalSegmentator served as the benchmark. The oracle and balanced models were similarly accurate (DSC Cohort I: 0.80 \pm 0.10 versus 0.81 \pm 0.10; Cohort II: 0.77 \pm 0.13 versus 0.80 \pm 0.12), outperforming TotalSegmentator. The balanced model, using half the training cases as oracle, produced similar dose metrics as manual delineations for all cardiac substructures. This model was robust to CT contrast in 6 out of 8 substructures and patient scan position variations in 5 out of 8 substructures and showed low correlations of accuracy to patient size and age. A HTN demonstrated robustly accurate (geometric and dose metrics) cardiac substructures segmentation from CTs with varying imaging and patient characteristics, one key requirement for clinical use. Moreover, the model combining pretraining with balanced distribution of NCCT and CECT scans was able to provide reliably accurate segmentations under varied conditions with far fewer labeled datasets compared to an oracle model.
>
---
#### [new 104] Towards Robust Spiking Neural Networks:Mitigating Heterogeneous Training Vulnerability via Dominant Eigencomponent Projection
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对脉冲神经网络（SNNs）在直接编码和BPTT训练中暴露的异质数据脆弱性问题，提出无超参数方法DEP。通过正交投影消除梯度主导成分，降低Hessian谱半径，防止网络陷入尖锐极小值，显著提升SNN抗数据污染能力和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.11134v1](http://arxiv.org/pdf/2505.11134v1)**

> **作者:** Desong Zhang; Jia Hu; Geyong Min
>
> **摘要:** Spiking Neural Networks (SNNs) process information via discrete spikes, enabling them to operate at remarkably low energy levels. However, our experimental observations reveal a striking vulnerability when SNNs are trained using the mainstream method--direct encoding combined with backpropagation through time (BPTT): even a single backward pass on data drawn from a slightly different distribution can lead to catastrophic network collapse. Our theoretical analysis attributes this vulnerability to the repeated inputs inherent in direct encoding and the gradient accumulation characteristic of BPTT, which together produce an exceptional large Hessian spectral radius. To address this challenge, we develop a hyperparameter-free method called Dominant Eigencomponent Projection (DEP). By orthogonally projecting gradients to precisely remove their dominant components, DEP effectively reduces the Hessian spectral radius, thereby preventing SNNs from settling into sharp minima. Extensive experiments demonstrate that DEP not only mitigates the vulnerability of SNNs to heterogeneous data poisoning, but also significantly enhances overall robustness compared to key baselines, providing strong support for safer and more reliable SNN deployment.
>
---
#### [new 105] Exploiting Radiance Fields for Grasp Generation on Novel Synthetic Views
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究机器人抓取生成任务，解决多视角图像采集耗时及可达性问题。利用辐射场合成新视图补充稀疏真实图像，提升抓取覆盖率和准确性。实验证明合成视图可增加有效抓取位姿，未来拟扩展至单图像构建辐射场。**

- **链接: [http://arxiv.org/pdf/2505.11467v1](http://arxiv.org/pdf/2505.11467v1)**

> **作者:** Abhishek Kashyap; Henrik Andreasson; Todor Stoyanov
>
> **备注:** 6 pages
>
> **摘要:** Vision based robot manipulation uses cameras to capture one or more images of a scene containing the objects to be manipulated. Taking multiple images can help if any object is occluded from one viewpoint but more visible from another viewpoint. However, the camera has to be moved to a sequence of suitable positions for capturing multiple images, which requires time and may not always be possible, due to reachability constraints. So while additional images can produce more accurate grasp poses due to the extra information available, the time-cost goes up with the number of additional views sampled. Scene representations like Gaussian Splatting are capable of rendering accurate photorealistic virtual images from user-specified novel viewpoints. In this work, we show initial results which indicate that novel view synthesis can provide additional context in generating grasp poses. Our experiments on the Graspnet-1billion dataset show that novel views contributed force-closure grasps in addition to the force-closure grasps obtained from sparsely sampled real views while also improving grasp coverage. In the future we hope this work can be extended to improve grasp extraction from radiance fields constructed with a single input image, using for example diffusion models or generalizable radiance fields.
>
---
#### [new 106] Hashing for Structure-based Anomaly Detection
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文属于异常检测任务，旨在高效识别不符合低维流形结构的数据样本。针对高维空间距离计算成本高的问题，提出基于局部敏感哈希的隔离检测方法，在偏好空间中定位孤立点作为异常，以更低计算量实现先进性能。通过哈希技术优化了检测效率，代码已开源。**

- **链接: [http://arxiv.org/pdf/2505.10873v1](http://arxiv.org/pdf/2505.10873v1)**

> **作者:** Filippo Leveni; Luca Magri; Cesare Alippi; Giacomo Boracchi
>
> **备注:** Accepted at International Conference on Image Analysis and Processing (ICIAP 2023)
>
> **摘要:** We focus on the problem of identifying samples in a set that do not conform to structured patterns represented by low-dimensional manifolds. An effective way to solve this problem is to embed data in a high dimensional space, called Preference Space, where anomalies can be identified as the most isolated points. In this work, we employ Locality Sensitive Hashing to avoid explicit computation of distances in high dimensions and thus improve Anomaly Detection efficiency. Specifically, we present an isolation-based anomaly detection technique designed to work in the Preference Space which achieves state-of-the-art performance at a lower computational cost. Code is publicly available at https://github.com/ineveLoppiliF/Hashing-for-Structure-based-Anomaly-Detection.
>
---
#### [new 107] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI models in Sound Localization
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **简介: 该论文研究多模态AI的声源定位任务，解决视听冲突下模型模态偏差问题。通过对比实验发现AI过度依赖视觉，性能低于人类；提出利用3D仿真数据微调模型，使其在冲突场景中表现提升并呈现类似人类的左右定位偏好，揭示了感官输入质量与架构对多模态表征的影响。**

- **链接: [http://arxiv.org/pdf/2505.11217v1](http://arxiv.org/pdf/2505.11217v1)**

> **作者:** Yanhao Jia; Ji Xie; S Jivaganesh; Hao Li; Xu Wu; Mengmi Zhang
>
> **备注:** 16 pages, 14 figures
>
> **摘要:** Imagine hearing a dog bark and turning toward the sound only to see a parked car, while the real, silent dog sits elsewhere. Such sensory conflicts test perception, yet humans reliably resolve them by prioritizing sound over misleading visuals. Despite advances in multimodal AI integrating vision and audio, little is known about how these systems handle cross-modal conflicts or whether they favor one modality. In this study, we systematically examine modality bias and conflict resolution in AI sound localization. We assess leading multimodal models and benchmark them against human performance in psychophysics experiments across six audiovisual conditions, including congruent, conflicting, and absent cues. Humans consistently outperform AI, demonstrating superior resilience to conflicting or missing visuals by relying on auditory information. In contrast, AI models often default to visual input, degrading performance to near chance levels. To address this, we finetune a state-of-the-art model using a stereo audio-image dataset generated via 3D simulations. Even with limited training data, the refined model surpasses existing benchmarks. Notably, it also mirrors human-like horizontal localization bias favoring left-right precision-likely due to the stereo audio structure reflecting human ear placement. These findings underscore how sensory input quality and system architecture shape multimodal representation accuracy.
>
---
#### [new 108] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?
- **分类: cs.CL; cs.AI; cs.CR; cs.CV; cs.SE**

- **简介: 该论文为系统化综述，研究计算机使用代理（CUAs）的安全威胁。通过文献分析，定义CUA安全分析框架，分类现有威胁，提出防御策略，总结评估指标，为未来研究提供基础。**

- **链接: [http://arxiv.org/pdf/2505.10924v1](http://arxiv.org/pdf/2505.10924v1)**

> **作者:** Ada Chen; Yongjiang Wu; Junyuan Zhang; Shu Yang; Jen-tse Huang; Kun Wang; Wenxuan Wang; Shuai Wang
>
> **摘要:** Recently, AI-driven interactions with computing devices have advanced from basic prototype tools to sophisticated, LLM-based systems that emulate human-like operations in graphical user interfaces. We are now witnessing the emergence of \emph{Computer-Using Agents} (CUAs), capable of autonomously performing tasks such as navigating desktop applications, web pages, and mobile apps. However, as these agents grow in capability, they also introduce novel safety and security risks. Vulnerabilities in LLM-driven reasoning, with the added complexity of integrating multiple software components and multimodal inputs, further complicate the security landscape. In this paper, we present a systematization of knowledge on the safety and security threats of CUAs. We conduct a comprehensive literature review and distill our findings along four research objectives: \textit{\textbf{(i)}} define the CUA that suits safety analysis; \textit{\textbf{(ii)} } categorize current safety threats among CUAs; \textit{\textbf{(iii)}} propose a comprehensive taxonomy of existing defensive strategies; \textit{\textbf{(iv)}} summarize prevailing benchmarks, datasets, and evaluation metrics used to assess the safety and performance of CUAs. Building on these insights, our work provides future researchers with a structured foundation for exploring unexplored vulnerabilities and offers practitioners actionable guidance in designing and deploying secure Computer-Using Agents.
>
---
#### [new 109] Predicting Risk of Pulmonary Fibrosis Formation in PASC Patients
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决长新冠（PASC）患者肺部纤维化早期预测难题。通过结合深度学习和放射组学，构建多中心胸部CT分析框架，利用CNN与可解释特征提取实现肺纤维化分类（82.2%准确率），并验证Grad-CAM和放射组学特征对临床评估的指导价值。**

- **链接: [http://arxiv.org/pdf/2505.10691v1](http://arxiv.org/pdf/2505.10691v1)**

> **作者:** Wanying Dou; Gorkem Durak; Koushik Biswas; Ziliang Hong; Andrea Mia Bejar; Elif Keles; Kaan Akin; Sukru Mehmet Erturk; Alpay Medetalibeyoglu; Marc Sala; Alexander Misharin; Hatice Savas; Mary Salvatore; Sachin Jambawalikar; Drew Torigian; Jayaram K. Udupa; Ulas Bagci
>
> **摘要:** While the acute phase of the COVID-19 pandemic has subsided, its long-term effects persist through Post-Acute Sequelae of COVID-19 (PASC), commonly known as Long COVID. There remains substantial uncertainty regarding both its duration and optimal management strategies. PASC manifests as a diverse array of persistent or newly emerging symptoms--ranging from fatigue, dyspnea, and neurologic impairments (e.g., brain fog), to cardiovascular, pulmonary, and musculoskeletal abnormalities--that extend beyond the acute infection phase. This heterogeneous presentation poses substantial challenges for clinical assessment, diagnosis, and treatment planning. In this paper, we focus on imaging findings that may suggest fibrotic damage in the lungs, a critical manifestation characterized by scarring of lung tissue, which can potentially affect long-term respiratory function in patients with PASC. This study introduces a novel multi-center chest CT analysis framework that combines deep learning and radiomics for fibrosis prediction. Our approach leverages convolutional neural networks (CNNs) and interpretable feature extraction, achieving 82.2% accuracy and 85.5% AUC in classification tasks. We demonstrate the effectiveness of Grad-CAM visualization and radiomics-based feature analysis in providing clinically relevant insights for PASC-related lung fibrosis prediction. Our findings highlight the potential of deep learning-driven computational methods for early detection and risk assessment of PASC-related lung fibrosis--presented for the first time in the literature.
>
---
#### [new 110] GrowSplat: Constructing Temporal Digital Twins of Plants with Gaussian Splats
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出GrowSplat框架，属于4D植物生长建模任务，旨在解决复杂植物形态的动态重建难题。通过融合3D高斯点云与两阶段配准（粗对齐+ICP精修），从多视角数据构建离散时间步的植物数字孪生，并在真实生态数据中验证了红杉和藜麦的时序重建效果。**

- **链接: [http://arxiv.org/pdf/2505.10923v1](http://arxiv.org/pdf/2505.10923v1)**

> **作者:** Simeon Adebola; Shuangyu Xie; Chung Min Kim; Justin Kerr; Bart M. van Marrewijk; Mieke van Vlaardingen; Tim van Daalen; Robert van Loo; Jose Luis Susa Rincon; Eugen Solowjow; Rick van de Zedde; Ken Goldberg
>
> **摘要:** Accurate temporal reconstructions of plant growth are essential for plant phenotyping and breeding, yet remain challenging due to complex geometries, occlusions, and non-rigid deformations of plants. We present a novel framework for building temporal digital twins of plants by combining 3D Gaussian Splatting with a robust sample alignment pipeline. Our method begins by reconstructing Gaussian Splats from multi-view camera data, then leverages a two-stage registration approach: coarse alignment through feature-based matching and Fast Global Registration, followed by fine alignment with Iterative Closest Point. This pipeline yields a consistent 4D model of plant development in discrete time steps. We evaluate the approach on data from the Netherlands Plant Eco-phenotyping Center, demonstrating detailed temporal reconstructions of Sequoia and Quinoa species. Videos and Images can be seen at https://berkeleyautomation.github.io/GrowSplat/
>
---
#### [new 111] Multimodal Event Detection: Current Approaches and Defining the New Playground through LLMs and VLMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究社交媒体多模态事件检测任务，解决传统单模态系统应对多模态数据传播的不足。通过对比单模态模型、多模态融合模型及生成模型（如GPT-4o），发现多模态方法优于单模态，但生成模型因无法准确生成事件类别，精度落后于监督方法。同时揭示生成模型擅长处理网络语言现象而监督方法难以应对的特性。**

- **链接: [http://arxiv.org/pdf/2505.10836v1](http://arxiv.org/pdf/2505.10836v1)**

> **作者:** Abhishek Dey; Aabha Bothera; Samhita Sarikonda; Rishav Aryan; Sanjay Kumar Podishetty; Akshay Havalgi; Gaurav Singh; Saurabh Srivastava
>
> **备注:** Accepted at NLDB 2025
>
> **摘要:** In this paper, we study the challenges of detecting events on social media, where traditional unimodal systems struggle due to the rapid and multimodal nature of data dissemination. We employ a range of models, including unimodal ModernBERT and ConvNeXt-V2, multimodal fusion techniques, and advanced generative models like GPT-4o, and LLaVA. Additionally, we also study the effect of providing multimodal generative models (such as GPT-4o) with a single modality to assess their efficacy. Our results indicate that while multimodal approaches notably outperform unimodal counterparts, generative approaches despite having a large number of parameters, lag behind supervised methods in precision. Furthermore, we also found that they lag behind instruction-tuned models because of their inability to generate event classes correctly. During our error analysis, we discovered that common social media issues such as leet speak, text elongation, etc. are effectively handled by generative approaches but are hard to tackle using supervised approaches.
>
---
#### [new 112] Textured mesh Quality Assessment using Geometry and Color Field Similarity
- **分类: cs.GR; cs.CV; cs.MM**

- **简介: 该论文针对纹理网格质量评估（TMQA）任务，解决现有方法评估不准确、鲁棒性差的问题。提出基于几何与颜色场相似性的FMQM方法，利用符号距离场和新颜色场提取四个视觉感知特征，在多个数据集上优于现有方法，且计算高效，适用于3D图形应用。**

- **链接: [http://arxiv.org/pdf/2505.10824v1](http://arxiv.org/pdf/2505.10824v1)**

> **作者:** Kaifa Yang; Qi Yang; Zhu Li; Yiling Xu
>
> **备注:** 15 pages main content, 4 pages supplementary material. Submitted to IEEE Transactions on Visualization and Computer Graphics (IEEE TVCG) for review
>
> **摘要:** Textured mesh quality assessment (TMQA) is critical for various 3D mesh applications. However, existing TMQA methods often struggle to provide accurate and robust evaluations. Motivated by the effectiveness of fields in representing both 3D geometry and color information, we propose a novel point-based TMQA method called field mesh quality metric (FMQM). FMQM utilizes signed distance fields and a newly proposed color field named nearest surface point color field to realize effective mesh feature description. Four features related to visual perception are extracted from the geometry and color fields: geometry similarity, geometry gradient similarity, space color distribution similarity, and space color gradient similarity. Experimental results on three benchmark datasets demonstrate that FMQM outperforms state-of-the-art (SOTA) TMQA metrics. Furthermore, FMQM exhibits low computational complexity, making it a practical and efficient solution for real-world applications in 3D graphics and visualization. Our code is publicly available at: https://github.com/yyyykf/FMQM.
>
---
#### [new 113] Generative Models in Computational Pathology: A Comprehensive Survey on Methods, Applications, and Challenges
- **分类: eess.IV; cs.CV**

- **简介: 该论文为综述类研究，系统总结计算病理学中生成模型的方法、应用及挑战。任务为梳理生成模型（如GAN、扩散模型）在病理图像/文本生成、多模态合成等领域的进展，解决数据效率、合成质量、临床适用性等问题。通过分析150+文献，归纳技术演化路径，评估数据集与方法，指出高保真图像生成、临床解释性不足等局限，并探讨伦理法律风险及未来研究方向。**

- **链接: [http://arxiv.org/pdf/2505.10993v1](http://arxiv.org/pdf/2505.10993v1)**

> **作者:** Yuan Zhang; Xinfeng Zhang; Xiaoming Qi Xinyu Wu; Feng Chen; Guanyu Yang; Huazhu Fu
>
> **备注:** 18 pages,9 figures
>
> **摘要:** Generative modeling has emerged as a promising direction in computational pathology, offering capabilities such as data-efficient learning, synthetic data augmentation, and multimodal representation across diverse diagnostic tasks. This review provides a comprehensive synthesis of recent progress in the field, organized into four key domains: image generation, text generation, multimodal image-text generation, and other generative applications, including spatial simulation and molecular inference. By analyzing over 150 representative studies, we trace the evolution of generative architectures from early generative adversarial networks to recent advances in diffusion models and foundation models with generative capabilities. We further examine the datasets and evaluation protocols commonly used in this domain and highlight ongoing limitations, including challenges in generating high-fidelity whole slide images, clinical interpretability, and concerns related to the ethical and legal implications of synthetic data. The review concludes with a discussion of open challenges and prospective research directions, with an emphasis on developing unified, multimodal, and clinically deployable generative systems. This work aims to provide a foundational reference for researchers and practitioners developing and applying generative models in computational pathology.
>
---
#### [new 114] Diffusion Model in Hyperspectral Image Processing and Analysis: A Review
- **分类: eess.IV; cs.CV**

- **简介: 该论文为综述类研究，聚焦扩散模型在高光谱图像处理中的应用。针对高维数据冗余、噪声干扰等挑战，传统方法存在局限。文章系统回顾了扩散模型在降维、去噪、分类及异常检测等任务中的优势，对比性能并总结挑战，证明其可提升分析精度与效率，为领域提供新思路。**

- **链接: [http://arxiv.org/pdf/2505.11158v1](http://arxiv.org/pdf/2505.11158v1)**

> **作者:** Xing Hu; Xiangcheng Liu; Qianqian Duan; Danfeng Hong; Dawei Zhang
>
> **备注:** 33 pages,20 figures
>
> **摘要:** Hyperspectral image processing and analysis has important application value in remote sensing, agriculture and environmental monitoring, but its high dimensionality, data redundancy and noise interference etc. bring great challenges to the analysis. Traditional models have limitations in dealing with these complex data, and it is difficult to meet the increasing demand for analysis. In recent years, Diffusion Model, as an emerging generative model, has shown unique advantages in hyperspectral image processing. By simulating the diffusion process of data in time, the Diffusion Model can effectively process high-dimensional data, generate high-quality samples, and perform well in denoising and data enhancement. In this paper, we review the recent research advances in diffusion modeling for hyperspectral image processing and analysis, and discuss its applications in tasks such as high-dimensional data processing, noise removal, classification, and anomaly detection. The performance of diffusion-based models on image processing is compared and the challenges are summarized. It is shown that the diffusion model can significantly improve the accuracy and efficiency of hyperspectral image analysis, providing a new direction for future research.
>
---
#### [new 115] Open-Source Multi-Viewpoint Surgical Telerobotics
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于医疗机器人领域，旨在通过多视角增强手术遥操作的可视化与机器感知。针对传统单视角限制，提出开源多视角系统，集成多传感器与升级控制逻辑，提升手术协作、共享自主及3D场景重建能力，推动临床转化研究。**

- **链接: [http://arxiv.org/pdf/2505.11142v1](http://arxiv.org/pdf/2505.11142v1)**

> **作者:** Guido Caccianiga; Yarden Sharon; Bernard Javot; Senya Polikovsky; Gökce Ergün; Ivan Capobianco; André L. Mihaljevic; Anton Deguet; Katherine J. Kuchenbecker
>
> **备注:** 2 pages, 2 figures, ICRA-RAMI workshop long abstract
>
> **摘要:** As robots for minimally invasive surgery (MIS) gradually become more accessible and modular, we believe there is a great opportunity to rethink and expand the visualization and control paradigms that have characterized surgical teleoperation since its inception. We conjecture that introducing one or more additional adjustable viewpoints in the abdominal cavity would not only unlock novel visualization and collaboration strategies for surgeons but also substantially boost the robustness of machine perception toward shared autonomy. Immediate advantages include controlling a second viewpoint and teleoperating surgical tools from a different perspective, which would allow collaborating surgeons to adjust their views independently and still maneuver their robotic instruments intuitively. Furthermore, we believe that capturing synchronized multi-view 3D measurements of the patient's anatomy would unlock advanced scene representations. Accurate real-time intraoperative 3D perception will allow algorithmic assistants to directly control one or more robotic instruments and/or robotic cameras. Toward these goals, we are building a synchronized multi-viewpoint, multi-sensor robotic surgery system by integrating high-performance vision components and upgrading the da Vinci Research Kit control logic. This short paper reports a functional summary of our setup and elaborates on its potential impacts in research and future clinical practice. By fully open-sourcing our system, we will enable the research community to reproduce our setup, improve it, and develop powerful algorithms, effectively boosting clinical translation of cutting-edge research.
>
---
#### [new 116] CTP: A hybrid CNN-Transformer-PINN model for ocean front forecasting
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于海洋锋预测任务，旨在解决现有模型在多步预测中空间连续性差、物理一致性弱的问题。提出CTP混合模型，整合CNN、Transformer和物理约束神经网络，通过局部编码、时序注意力及物理规则增强预测效果。实验显示其在南海和黑潮区域(1993-2020)单/多步预测中精度、F1分数及时序稳定性均超越基线模型。**

- **链接: [http://arxiv.org/pdf/2505.10894v1](http://arxiv.org/pdf/2505.10894v1)**

> **作者:** Yishuo Wang; Feng Zhou; Muping Zhou; Qicheng Meng; Zhijun Hu; Yi Wang
>
> **摘要:** This paper proposes CTP, a novel deep learning framework that integrates convolutional neural network(CNN), Transformer architectures, and physics-informed neural network(PINN) for ocean front prediction. Ocean fronts, as dynamic interfaces between distinct water masses, play critical roles in marine biogeochemical and physical processes. Existing methods such as LSTM, ConvLSTM, and AttentionConv often struggle to maintain spatial continuity and physical consistency over multi-step forecasts. CTP addresses these challenges by combining localized spatial encoding, long-range temporal attention, and physical constraint enforcement. Experimental results across south China sea(SCS) and Kuroshio(KUR) regions from 1993 to 2020 demonstrate that CTP achieves state-of-the-art(SOTA) performance in both single-step and multi-step predictions, significantly outperforming baseline models in accuracy, $F_1$ score, and temporal stability.
>
---
#### [new 117] Assessing the Performance of Analog Training for Transfer Learning
- **分类: cs.LG; cs.AI; cs.AR; cs.CV; cs.DC; cs.NE**

- **简介: 该论文属于迁移学习任务，旨在解决模拟内存计算中因设备非线性、不对称等特性导致的训练效果差问题。提出新算法c-TTv2，在Swin-ViT模型和CIFAR100数据集上验证其性能，并测试算法对设备噪声、对称性偏差等参数的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.11067v1](http://arxiv.org/pdf/2505.11067v1)**

> **作者:** Omobayode Fagbohungbe; Corey Lammie; Malte J. Rasch; Takashi Ando; Tayfun Gokmen; Vijay Narayanan
>
> **摘要:** Analog in-memory computing is a next-generation computing paradigm that promises fast, parallel, and energy-efficient deep learning training and transfer learning (TL). However, achieving this promise has remained elusive due to a lack of suitable training algorithms. Analog memory devices exhibit asymmetric and non-linear switching behavior in addition to device-to-device variation, meaning that most, if not all, of the current off-the-shelf training algorithms cannot achieve good training outcomes. Also, recently introduced algorithms have enjoyed limited attention, as they require bi-directionally switching devices of unrealistically high symmetry and precision and are highly sensitive. A new algorithm chopped TTv2 (c-TTv2), has been introduced, which leverages the chopped technique to address many of the challenges mentioned above. In this paper, we assess the performance of the c-TTv2 algorithm for analog TL using a Swin-ViT model on a subset of the CIFAR100 dataset. We also investigate the robustness of our algorithm to changes in some device specifications, including weight transfer noise, symmetry point skew, and symmetry point variability
>
---
#### [new 118] Planar Velocity Estimation for Fast-Moving Mobile Robots Using Event-Based Optical Flow
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究移动机器人平面速度估计任务，解决传统轮式里程计依赖地面摩擦假设、环境适应性差的问题。提出基于事件相机光流与平面运动学的融合方法，利用事件相机微秒延迟和高动态范围特性克服运动模糊。实验表明，在高速场景(32m/s)下横向误差降低38.3%，验证了实际部署潜力。**

- **链接: [http://arxiv.org/pdf/2505.11116v1](http://arxiv.org/pdf/2505.11116v1)**

> **作者:** Liam Boyle; Jonas Kühne; Nicolas Baumann; Niklas Bastuck; Michele Magno
>
> **摘要:** Accurate velocity estimation is critical in mobile robotics, particularly for driver assistance systems and autonomous driving. Wheel odometry fused with Inertial Measurement Unit (IMU) data is a widely used method for velocity estimation; however, it typically requires strong assumptions, such as non-slip steering, or complex vehicle dynamics models that do not hold under varying environmental conditions like slippery surfaces. We introduce an approach to velocity estimation that is decoupled from wheel-to-surface traction assumptions by leveraging planar kinematics in combination with optical flow from event cameras pointed perpendicularly at the ground. The asynchronous micro-second latency and high dynamic range of event cameras make them highly robust to motion blur, a common challenge in vision-based perception techniques for autonomous driving. The proposed method is evaluated through in-field experiments on a 1:10 scale autonomous racing platform and compared to precise motion capture data, demonstrating not only performance on par with the state-of-the-art Event-VIO method but also a 38.3 % improvement in lateral error. Qualitative experiments at highway speeds of up to 32 m/s further confirm the effectiveness of our approach, indicating significant potential for real-world deployment.
>
---
## 更新

#### [replaced 001] SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04893v4](http://arxiv.org/pdf/2504.04893v4)**

> **作者:** Justus Westerhoff; Erblina Purelku; Jakob Hackstein; Jonas Loos; Leo Pinetzki; Lorenz Hufe
>
> **备注:** Accepted at CVPR 2025 Workshop EVAL-FoMo-2
>
> **摘要:** Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper along with the code for evaluations at www.bliss.berlin/research/scam.
>
---
#### [replaced 002] Visual Watermarking in the Era of Diffusion Models: Advances and Challenges
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08197v2](http://arxiv.org/pdf/2505.08197v2)**

> **作者:** Junxian Duan; Jiyang Guan; Wenkui Yang; Ran He
>
> **摘要:** As generative artificial intelligence technologies like Stable Diffusion advance, visual content becomes more vulnerable to misuse, raising concerns about copyright infringement. Visual watermarks serve as effective protection mechanisms, asserting ownership and deterring unauthorized use. Traditional deepfake detection methods often rely on passive techniques that struggle with sophisticated manipulations. In contrast, diffusion models enhance detection accuracy by allowing for the effective learning of features, enabling the embedding of imperceptible and robust watermarks. We analyze the strengths and challenges of watermark techniques related to diffusion models, focusing on their robustness and application in watermark generation. By exploring the integration of advanced diffusion models and watermarking security, we aim to advance the discourse on preserving watermark robustness against evolving forgery threats. It emphasizes the critical importance of developing innovative solutions to protect digital content and ensure the preservation of ownership rights in the era of generative AI.
>
---
#### [replaced 003] Structured Preference Optimization for Vision-Language Long-Horizon Task Planning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20742v3](http://arxiv.org/pdf/2502.20742v3)**

> **作者:** Xiwen Liang; Min Lin; Weiqi Ruan; Rongtao Xu; Yuecheng Liu; Jiaqi Chen; Bingqian Lin; Yuzheng Zhuang; Xiaodan Liang
>
> **备注:** 18 pages
>
> **摘要:** Existing methods for vision-language task planning excel in short-horizon tasks but often fall short in complex, long-horizon planning within dynamic environments. These challenges primarily arise from the difficulty of effectively training models to produce high-quality reasoning processes for long-horizon tasks. To address this, we propose Structured Preference Optimization (SPO), which aims to enhance reasoning and action selection in long-horizon task planning through structured preference evaluation and optimized training strategies. Specifically, SPO introduces: 1) Preference-Based Scoring and Optimization, which systematically evaluates reasoning chains based on task relevance, visual grounding, and historical consistency; and 2) Curriculum-Guided Training, where the model progressively adapts from simple to complex tasks, improving its generalization ability in long-horizon scenarios and enhancing reasoning robustness. To advance research in vision-language long-horizon task planning, we introduce ExtendaBench, a comprehensive benchmark covering 1,509 tasks across VirtualHome and Habitat 2.0, categorized into ultra-short, short, medium, and long tasks. Experimental results demonstrate that SPO significantly improves reasoning quality and final decision accuracy, outperforming prior methods on long-horizon tasks and underscoring the effectiveness of preference-driven optimization in vision-language task planning. Specifically, SPO achieves a +5.98% GCR and +4.68% SR improvement in VirtualHome and a +3.30% GCR and +2.11% SR improvement in Habitat over the best-performing baselines.
>
---
#### [replaced 004] An Enhanced YOLOv8 Model for Real-Time and Accurate Pothole Detection and Measurement
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.04207v2](http://arxiv.org/pdf/2505.04207v2)**

> **作者:** Mustafa Yurdakul; Şakir Tasdemir
>
> **摘要:** Potholes cause vehicle damage and traffic accidents, creating serious safety and economic problems. Therefore, early and accurate detection of potholes is crucial. Existing detection methods are usually only based on 2D RGB images and cannot accurately analyze the physical characteristics of potholes. In this paper, a publicly available dataset of RGB-D images (PothRGBD) is created and an improved YOLOv8-based model is proposed for both pothole detection and pothole physical features analysis. The Intel RealSense D415 depth camera was used to collect RGB and depth data from the road surfaces, resulting in a PothRGBD dataset of 1000 images. The data was labeled in YOLO format suitable for segmentation. A novel YOLO model is proposed based on the YOLOv8n-seg architecture, which is structurally improved with Dynamic Snake Convolution (DSConv), Simple Attention Module (SimAM) and Gaussian Error Linear Unit (GELU). The proposed model segmented potholes with irregular edge structure more accurately, and performed perimeter and depth measurements on depth maps with high accuracy. The standard YOLOv8n-seg model achieved 91.9% precision, 85.2% recall and 91.9% mAP@50. With the proposed model, the values increased to 93.7%, 90.4% and 93.8% respectively. Thus, an improvement of 1.96% in precision, 6.13% in recall and 2.07% in mAP was achieved. The proposed model performs pothole detection as well as perimeter and depth measurement with high accuracy and is suitable for real-time applications due to its low model complexity. In this way, a lightweight and effective model that can be used in deep learning-based intelligent transportation solutions has been acquired.
>
---
#### [replaced 005] EmoFace: Emotion-Content Disentangled Speech-Driven 3D Talking Face Animation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.11518v3](http://arxiv.org/pdf/2408.11518v3)**

> **作者:** Yihong Lin; Liang Peng; Zhaoxin Fan; Xianjia Wu; Jianqiao Hu; Xiandong Li; Wenxiong Kang; Songju Lei
>
> **摘要:** The creation of increasingly vivid 3D talking face has become a hot topic in recent years. Currently, most speech-driven works focus on lip synchronisation but neglect to effectively capture the correlations between emotions and facial motions. To address this problem, we propose a two-stream network called EmoFace, which consists of an emotion branch and a content branch. EmoFace employs a novel Mesh Attention mechanism to analyse and fuse the emotion features and content features. Particularly, a newly designed spatio-temporal graph-based convolution, SpiralConv3D, is used in Mesh Attention to learn potential temporal and spatial feature dependencies between mesh vertices. In addition, to the best of our knowledge, it is the first time to introduce a new self-growing training scheme with intermediate supervision to dynamically adjust the ratio of groundtruth adopted in the 3D face animation task. Comprehensive quantitative and qualitative evaluations on our high-quality 3D emotional facial animation dataset, 3D-RAVDESS ($4.8863\times 10^{-5}$mm for LVE and $0.9509\times 10^{-5}$mm for EVE), together with the public dataset VOCASET ($2.8669\times 10^{-5}$mm for LVE and $0.4664\times 10^{-5}$mm for EVE), demonstrate that our approach achieves state-of-the-art performance.
>
---
#### [replaced 006] Rethinking Weight-Averaged Model-merging
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.09263v4](http://arxiv.org/pdf/2411.09263v4)**

> **作者:** Hu Wang; Congbo Ma; Ibrahim Almakky; Ian Reid; Gustavo Carneiro; Mohammad Yaqub
>
> **摘要:** Model-merging has emerged as a powerful approach in deep learning, capable of enhancing model performance without any training. However, the underlying mechanisms that explain its effectiveness remain largely unexplored. In this paper, we investigate this technique from three novel perspectives to empirically provide deeper insights into why and how weight-averaged model-merging~\cite{wortsman2022soups} works: (1) we examine the intrinsic patterns captured by the learning of the model weights, and we are the first to connect that these weights encode structured with why weight-averaged model merging can work; (2) we investigate averaging on weights versus averaging on features, providing analyses from the view of diverse architecture comparisons on multiple datasets; and (3) we explore the impact on model-merging prediction stability in terms of changing the parameter magnitude, revealing insights into the way of weight averaging works as regularization by showing the robustness across different parameter scales. The code is available at https://github.com/billhhh/Rethink-Merge.
>
---
#### [replaced 007] Empowering Agentic Video Analytics Systems with Video Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.00254v3](http://arxiv.org/pdf/2505.00254v3)**

> **作者:** Yuxuan Yan; Shiqi Jiang; Ting Cao; Yifan Yang; Qianqian Yang; Yuanchao Shu; Yuqing Yang; Lili Qiu
>
> **备注:** 15 pages, AVAS, add latency breakdown
>
> **摘要:** AI-driven video analytics has become increasingly pivotal across diverse domains. However, existing systems are often constrained to specific, predefined tasks, limiting their adaptability in open-ended analytical scenarios. The recent emergence of Video-Language Models (VLMs) as transformative technologies offers significant potential for enabling open-ended video understanding, reasoning, and analytics. Nevertheless, their limited context windows present challenges when processing ultra-long video content, which is prevalent in real-world applications. To address this, we introduce AVAS, a VLM-powered system designed for open-ended, advanced video analytics. AVAS incorporates two key innovations: (1) the near real-time construction of Event Knowledge Graphs (EKGs) for efficient indexing of long or continuous video streams, and (2) an agentic retrieval-generation mechanism that leverages EKGs to handle complex and diverse queries. Comprehensive evaluations on public benchmarks, LVBench and VideoMME-Long, demonstrate that AVAS achieves state-of-the-art performance, attaining 62.3% and 64.1% accuracy, respectively, significantly surpassing existing VLM and video Retrieval-Augmented Generation (RAG) systems. Furthermore, to evaluate video analytics in ultra-long and open-world video scenarios, we introduce a new benchmark, AVAS-100. This benchmark comprises 8 videos, each exceeding 10 hours in duration, along with 120 manually annotated, diverse, and complex question-answer pairs. On AVAS-100, AVAS achieves top-tier performance with an accuracy of 75.8%.
>
---
#### [replaced 008] Question-Answering Dense Video Events
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2409.04388v5](http://arxiv.org/pdf/2409.04388v5)**

> **作者:** Hangyu Qin; Junbin Xiao; Angela Yao
>
> **备注:** Accepted to SIGIR'25
>
> **摘要:** This paper presents question-answering on dense video events, a novel task that answers and grounds dense-event questions in long videos, thus challenging MLLMs to faithfully comprehend and reason about multiple events over extended periods of time. To facilitate the study, we construct DeVE-QA -- a dataset featuring 78K questions about 26K events on 10.6K long videos. Our benchmarking shows that state-of-the-art MLLMs struggle on DeVE-QA. For improvement, we propose DeVi, a novel training-free MLLM approach that highlights a hierarchical captioning module, a temporal event memory module, and a self-consistency checking module to respectively detect, contextualize and memorize, and ground dense-events in long videos for question answering. Extensive experiments show that DeVi is superior at answering dense-event questions and grounding relevant video moments. Compared with existing MLLMs, it achieves a notable increase of 4.8% and 2.1% for G(round)QA accuracy on DeVE-QA and NExT-GQA, respectively. Data and code are available at https://github.com/QHUni/DeVE-QA.
>
---
#### [replaced 009] NeRF-To-Real Tester: Neural Radiance Fields as Test Image Generators for Vision of Autonomous Systems
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.16141v2](http://arxiv.org/pdf/2412.16141v2)**

> **作者:** Laura Weihl; Bilal Wehbe; Andrzej Wąsowski
>
> **摘要:** Autonomous inspection of infrastructure on land and in water is a quickly growing market, with applications including surveying constructions, monitoring plants, and tracking environmental changes in on- and off-shore wind energy farms. For Autonomous Underwater Vehicles and Unmanned Aerial Vehicles overfitting of controllers to simulation conditions fundamentally leads to poor performance in the operation environment. There is a pressing need for more diverse and realistic test data that accurately represents the challenges faced by these systems. We address the challenge of generating perception test data for autonomous systems by leveraging Neural Radiance Fields to generate realistic and diverse test images, and integrating them into a metamorphic testing framework for vision components such as vSLAM and object detection. Our tool, N2R-Tester, allows training models of custom scenes and rendering test images from perturbed positions. An experimental evaluation of N2R-Tester on eight different vision components in AUVs and UAVs demonstrates the efficacy and versatility of the approach.
>
---
#### [replaced 010] Learning Robust Anymodal Segmentor with Unimodal and Cross-modal Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17141v2](http://arxiv.org/pdf/2411.17141v2)**

> **作者:** Xu Zheng; Haiwei Xue; Jialei Chen; Yibo Yan; Lutao Jiang; Yuanhuiyi Lyu; Kailun Yang; Linfeng Zhang; Xuming Hu
>
> **备注:** Preprint
>
> **摘要:** Simultaneously using multimodal inputs from multiple sensors to train segmentors is intuitively advantageous but practically challenging. A key challenge is unimodal bias, where multimodal segmentors over rely on certain modalities, causing performance drops when others are missing, common in real world applications. To this end, we develop the first framework for learning robust segmentor that can handle any combinations of visual modalities. Specifically, we first introduce a parallel multimodal learning strategy for learning a strong teacher. The cross-modal and unimodal distillation is then achieved in the multi scale representation space by transferring the feature level knowledge from multimodal to anymodal segmentors, aiming at addressing the unimodal bias and avoiding over-reliance on specific modalities. Moreover, a prediction level modality agnostic semantic distillation is proposed to achieve semantic knowledge transferring for segmentation. Extensive experiments on both synthetic and real-world multi-sensor benchmarks demonstrate that our method achieves superior performance.
>
---
#### [replaced 011] Efficient and Comprehensive Feature Extraction in Large Vision-Language Model for Pathology Analysis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.09521v3](http://arxiv.org/pdf/2412.09521v3)**

> **作者:** Shengxuming Zhang; Weihan Li; Tianhong Gao; Jiacong Hu; Haoming Luo; Xiuming Zhang; Jing Zhang; Mingli Song; Zunlei Feng
>
> **摘要:** Pathological diagnosis is vital for determining disease characteristics, guiding treatment, and assessing prognosis, relying heavily on detailed, multi-scale analysis of high-resolution whole slide images (WSI). However, existing large vision-language models (LVLMs) are limited by input resolution constraints, hindering their efficiency and accuracy in pathology image analysis. To overcome these issues, we propose two innovative strategies: the mixed task-guided feature enhancement, which directs feature extraction toward lesion-related details across scales, and the prompt-guided detail feature completion, which integrates coarse- and fine-grained features from WSI based on specific prompts without compromising inference speed. Leveraging a comprehensive dataset of 490K samples from diverse pathology tasks, we trained the pathology-specialized LVLM, OmniPath. Extensive experiments demonstrate that this model significantly outperforms existing methods in diagnostic accuracy and efficiency, providing an interactive, clinically aligned approach for auxiliary diagnosis in a wide range of pathology applications.
>
---
#### [replaced 012] RGB-Event Fusion with Self-Attention for Collision Prediction
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04258v2](http://arxiv.org/pdf/2505.04258v2)**

> **作者:** Pietro Bonazzi; Christian Vogt; Michael Jost; Haotong Qin; Lyes Khacef; Federico Paredes-Valles; Michele Magno
>
> **备注:** arXiv admin note: text overlap with arXiv:2504.10400
>
> **摘要:** Ensuring robust and real-time obstacle avoidance is critical for the safe operation of autonomous robots in dynamic, real-world environments. This paper proposes a neural network framework for predicting the time and collision position of an unmanned aerial vehicle with a dynamic object, using RGB and event-based vision sensors. The proposed architecture consists of two separate encoder branches, one for each modality, followed by fusion by self-attention to improve prediction accuracy. To facilitate benchmarking, we leverage the ABCD [8] dataset collected that enables detailed comparisons of single-modality and fusion-based approaches. At the same prediction throughput of 50Hz, the experimental results show that the fusion-based model offers an improvement in prediction accuracy over single-modality approaches of 1% on average and 10% for distances beyond 0.5m, but comes at the cost of +71% in memory and + 105% in FLOPs. Notably, the event-based model outperforms the RGB model by 4% for position and 26% for time error at a similar computational cost, making it a competitive alternative. Additionally, we evaluate quantized versions of the event-based models, applying 1- to 8-bit quantization to assess the trade-offs between predictive performance and computational efficiency. These findings highlight the trade-offs of multi-modal perception using RGB and event-based cameras in robotic applications.
>
---
#### [replaced 013] Descriptive Image-Text Matching with Graded Contextual Similarity
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.09997v2](http://arxiv.org/pdf/2505.09997v2)**

> **作者:** Jinhyun Jang; Jiyoung Lee; Kwanghoon Sohn
>
> **摘要:** Image-text matching aims to build correspondences between visual and textual data by learning their pairwise similarities. Most existing approaches have adopted sparse binary supervision, indicating whether a pair of images and sentences matches or not. However, such sparse supervision covers a limited subset of image-text relationships, neglecting their inherent many-to-many correspondences; an image can be described in numerous texts at different descriptive levels. Moreover, existing approaches overlook the implicit connections from general to specific descriptions, which form the underlying rationale for the many-to-many relationships between vision and language. In this work, we propose descriptive image-text matching, called DITM, to learn the graded contextual similarity between image and text by exploring the descriptive flexibility of language. We formulate the descriptiveness score of each sentence with cumulative term frequency-inverse document frequency (TF-IDF) to balance the pairwise similarity according to the keywords in the sentence. Our method leverages sentence descriptiveness to learn robust image-text matching in two key ways: (1) to refine the false negative labeling, dynamically relaxing the connectivity between positive and negative pairs, and (2) to build more precise matching, aligning a set of relevant sentences in a generic-to-specific order. By moving beyond rigid binary supervision, DITM enhances the discovery of both optimal matches and potential positive pairs. Extensive experiments on MS-COCO, Flickr30K, and CxC datasets demonstrate the effectiveness of our method in representing complex image-text relationships compared to state-of-the-art approaches. In addition, DITM enhances the hierarchical reasoning ability of the model, supported by the extensive analysis on HierarCaps benchmark.
>
---
#### [replaced 014] Ophora: A Large-Scale Data-Driven Text-Guided Ophthalmic Surgical Video Generation Model
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07449v3](http://arxiv.org/pdf/2505.07449v3)**

> **作者:** Wei Li; Ming Hu; Guoan Wang; Lihao Liu; Kaijin Zhou; Junzhi Ning; Xin Guo; Zongyuan Ge; Lixu Gu; Junjun He
>
> **备注:** Early accepted in MICCAI25
>
> **摘要:** In ophthalmic surgery, developing an AI system capable of interpreting surgical videos and predicting subsequent operations requires numerous ophthalmic surgical videos with high-quality annotations, which are difficult to collect due to privacy concerns and labor consumption. Text-guided video generation (T2V) emerges as a promising solution to overcome this issue by generating ophthalmic surgical videos based on surgeon instructions. In this paper, we present Ophora, a pioneering model that can generate ophthalmic surgical videos following natural language instructions. To construct Ophora, we first propose a Comprehensive Data Curation pipeline to convert narrative ophthalmic surgical videos into a large-scale, high-quality dataset comprising over 160K video-instruction pairs, Ophora-160K. Then, we propose a Progressive Video-Instruction Tuning scheme to transfer rich spatial-temporal knowledge from a T2V model pre-trained on natural video-text datasets for privacy-preserved ophthalmic surgical video generation based on Ophora-160K. Experiments on video quality evaluation via quantitative analysis and ophthalmologist feedback demonstrate that Ophora can generate realistic and reliable ophthalmic surgical videos based on surgeon instructions. We also validate the capability of Ophora for empowering downstream tasks of ophthalmic surgical workflow understanding. Code is available at https://github.com/mar-cry/Ophora.
>
---
#### [replaced 015] Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.04877v2](http://arxiv.org/pdf/2503.04877v2)**

> **作者:** Albert Wilcox; Mohamed Ghanem; Masoud Moghani; Pierre Barroso; Benjamin Joffe; Animesh Garg
>
> **备注:** Videos, code, and data: https://pairlab.github.io/Adapt3R
>
> **摘要:** Imitation Learning can train robots to perform complex and diverse manipulation tasks, but learned policies are brittle with observations outside of the training distribution. 3D scene representations that incorporate observations from calibrated RGBD cameras have been proposed as a way to mitigate this, but in our evaluations with unseen embodiments and camera viewpoints they show only modest improvement. To address those challenges, we propose Adapt3R, a general-purpose 3D observation encoder which synthesizes data from calibrated RGBD cameras into a vector that can be used as conditioning for arbitrary IL algorithms. The key idea is to use a pretrained 2D backbone to extract semantic information, using 3D only as a medium to localize this information with respect to the end-effector. We show across 93 simulated and 6 real tasks that when trained end-to-end with a variety of IL algorithms, Adapt3R maintains these algorithms' learning capacity while enabling zero-shot transfer to novel embodiments and camera poses.
>
---
#### [replaced 016] Discriminating image representations with principal distortions
- **分类: q-bio.NC; cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.15433v2](http://arxiv.org/pdf/2410.15433v2)**

> **作者:** Jenelle Feather; David Lipshutz; Sarah E. Harvey; Alex H. Williams; Eero P. Simoncelli
>
> **摘要:** Image representations (artificial or biological) are often compared in terms of their global geometric structure; however, representations with similar global structure can have strikingly different local geometries. Here, we propose a framework for comparing a set of image representations in terms of their local geometries. We quantify the local geometry of a representation using the Fisher information matrix, a standard statistical tool for characterizing the sensitivity to local stimulus distortions, and use this as a substrate for a metric on the local geometry in the vicinity of a base image. This metric may then be used to optimally differentiate a set of models, by finding a pair of "principal distortions" that maximize the variance of the models under this metric. As an example, we use this framework to compare a set of simple models of the early visual system, identifying a novel set of image distortions that allow immediate comparison of the models by visual inspection. In a second example, we apply our method to a set of deep neural network models and reveal differences in the local geometry that arise due to architecture and training types. These examples demonstrate how our framework can be used to probe for informative differences in local sensitivities between complex models, and suggest how it could be used to compare model representations with human perception.
>
---
#### [replaced 017] Learning to Deblur Polarized Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.18134v2](http://arxiv.org/pdf/2402.18134v2)**

> **作者:** Chu Zhou; Minggui Teng; Xinyu Zhou; Chao Xu; Imari Sato; Boxin Shi
>
> **备注:** This version has been accepted for publication in IJCV. This arXiv version corresponds to the final accepted manuscript
>
> **摘要:** A polarization camera can capture four linear polarized images with different polarizer angles in a single shot, which is useful in polarization-based vision applications since the degree of linear polarization (DoLP) and the angle of linear polarization (AoLP) can be directly computed from the captured polarized images. However, since the on-chip micro-polarizers block part of the light so that the sensor often requires a longer exposure time, the captured polarized images are prone to motion blur caused by camera shakes, leading to noticeable degradation in the computed DoLP and AoLP. Deblurring methods for conventional images often show degraded performance when handling the polarized images since they only focus on deblurring without considering the polarization constraints. In this paper, we propose a polarized image deblurring pipeline to solve the problem in a polarization-aware manner by adopting a divide-and-conquer strategy to explicitly decompose the problem into two less ill-posed sub-problems, and design a two-stage neural network to handle the two sub-problems respectively. Experimental results show that our method achieves state-of-the-art performance on both synthetic and real-world images, and can improve the performance of polarization-based vision applications such as image dehazing and reflection removal.
>
---
#### [replaced 018] TwinTURBO: Semi-Supervised Fine-Tuning of Foundation Models via Mutual Information Decompositions for Downstream Task and Latent Spaces
- **分类: cs.LG; cs.CV; cs.IT; math.IT; stat.ML**

- **链接: [http://arxiv.org/pdf/2503.07851v2](http://arxiv.org/pdf/2503.07851v2)**

> **作者:** Guillaume Quétant; Pavlo Molchanov; Slava Voloshynovskiy
>
> **摘要:** We present a semi-supervised fine-tuning framework for foundation models that utilises mutual information decomposition to address the challenges of training for a limited amount of labelled data. Our approach derives two distinct lower bounds: i) for the downstream task space, such as classification, optimised using conditional and marginal cross-entropy alongside Kullback-Leibler divergence, and ii) for the latent space representation, regularised and aligned using a contrastive-like decomposition. This fine-tuning strategy retains the pre-trained structure of the foundation model, modifying only a specialised projector module comprising a small transformer and a token aggregation technique. Experiments on several datasets demonstrate significant improvements in classification tasks under extremely low-labelled conditions by effectively leveraging unlabelled data.
>
---
#### [replaced 019] IMPACT: A Generic Semantic Loss for Multimodal Medical Image Registration
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.24121v3](http://arxiv.org/pdf/2503.24121v3)**

> **作者:** Valentin Boussot; Cédric Hémon; Jean-Claude Nunes; Jason Dowling; Simon Rouzé; Caroline Lafond; Anaïs Barateau; Jean-Louis Dillenseger
>
> **备注:** Submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). This is a preprint version and has not been peer-reviewed
>
> **摘要:** Image registration is fundamental in medical imaging, enabling precise alignment of anatomical structures for diagnosis, treatment planning, image-guided interventions, and longitudinal monitoring. This work introduces IMPACT (Image Metric with Pretrained model-Agnostic Comparison for Transmodality registration), a novel similarity metric designed for robust multimodal image registration. Rather than relying on raw intensities, handcrafted descriptors, or task-specific training, IMPACT defines a semantic similarity measure based on the comparison of deep features extracted from large-scale pretrained segmentation models. By leveraging representations from models such as TotalSegmentator, Segment Anything (SAM), and other foundation networks, IMPACT provides a task-agnostic, training-free solution that generalizes across imaging modalities. These features, originally trained for segmentation, offer strong spatial correspondence and semantic alignment capabilities, making them naturally suited for registration. The method integrates seamlessly into both algorithmic (Elastix) and learning-based (VoxelMorph) frameworks, leveraging the strengths of each. IMPACT was evaluated on five challenging 3D registration tasks involving thoracic CT/CBCT and pelvic MR/CT datasets. Quantitative metrics, including Target Registration Error and Dice Similarity Coefficient, demonstrated consistent improvements in anatomical alignment over baseline methods. Qualitative analyses further highlighted the robustness of the proposed metric in the presence of noise, artifacts, and modality variations. With its versatility, efficiency, and strong performance across diverse tasks, IMPACT offers a powerful solution for advancing multimodal image registration in both clinical and research settings.
>
---
#### [replaced 020] Understanding Galaxy Morphology Evolution Through Cosmic Time via Redshift Conditioned Diffusion Models
- **分类: astro-ph.GA; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18440v2](http://arxiv.org/pdf/2411.18440v2)**

> **作者:** Andrew Lizarraga; Eric Hanchen Jiang; Jacob Nowack; Yun Qi Li; Ying Nian Wu; Bernie Boscoe; Tuan Do
>
> **摘要:** Redshift measures the distance to galaxies and underlies our understanding of the origin of the Universe and galaxy evolution. Spectroscopic redshift is the gold-standard method for measuring redshift, but it requires about $1000$ times more telescope time than broad-band imaging. That extra cost limits sky coverage and sample size and puts large spectroscopic surveys out of reach. Photometric redshift methods rely on imaging in multiple color filters and template fitting, yet they ignore the wealth of information carried by galaxy shape and structure. We demonstrate that a diffusion model conditioned on continuous redshift learns this missing joint structure, reproduces known morphology-$z$ correlations. We verify on the HyperSuprime-Cam survey, that the model captures redshift-dependent trends in ellipticity, semi-major axis, S\'ersic index, and isophotal area that these generated images correlate closely with true redshifts on test data. To our knowledge this is the first study to establish a direct link between galaxy morphology and redshift. Our approach offers a simple and effective path to redshift estimation from imaging data and will help unlock the full potential of upcoming wide-field surveys.
>
---
#### [replaced 021] Evaluating Vision-Language Models as Evaluators in Path Planning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.18711v4](http://arxiv.org/pdf/2411.18711v4)**

> **作者:** Mohamed Aghzal; Xiang Yue; Erion Plaku; Ziyu Yao
>
> **备注:** Accepted to the 2025 IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)
>
> **摘要:** Despite their promise to perform complex reasoning, large language models (LLMs) have been shown to have limited effectiveness in end-to-end planning. This has inspired an intriguing question: if these models cannot plan well, can they still contribute to the planning framework as a helpful plan evaluator? In this work, we generalize this question to consider LLMs augmented with visual understanding, i.e., Vision-Language Models (VLMs). We introduce PathEval, a novel benchmark evaluating VLMs as plan evaluators in complex path-planning scenarios. Succeeding in the benchmark requires a VLM to be able to abstract traits of optimal paths from the scenario description, demonstrate precise low-level perception on each path, and integrate this information to decide the better path. Our analysis of state-of-the-art VLMs reveals that these models face significant challenges on the benchmark. We observe that the VLMs can precisely abstract given scenarios to identify the desired traits and exhibit mixed performance in integrating the provided information. Yet, their vision component presents a critical bottleneck, with models struggling to perceive low-level details about a path. Our experimental results show that this issue cannot be trivially addressed via end-to-end fine-tuning; rather, task-specific discriminative adaptation of these vision encoders is needed for these VLMs to become effective path evaluators.
>
---
#### [replaced 022] MTVCrafter: 4D Motion Tokenization for Open-World Human Image Animation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10238v2](http://arxiv.org/pdf/2505.10238v2)**

> **作者:** Yanbo Ding; Xirui Hu; Zhizhi Guo; Yali Wang
>
> **摘要:** Human image animation has gained increasing attention and developed rapidly due to its broad applications in digital humans. However, existing methods rely largely on 2D-rendered pose images for motion guidance, which limits generalization and discards essential 3D information for open-world animation. To tackle this problem, we propose MTVCrafter (Motion Tokenization Video Crafter), the first framework that directly models raw 3D motion sequences (i.e., 4D motion) for human image animation. Specifically, we introduce 4DMoT (4D motion tokenizer) to quantize 3D motion sequences into 4D motion tokens. Compared to 2D-rendered pose images, 4D motion tokens offer more robust spatio-temporal cues and avoid strict pixel-level alignment between pose image and character, enabling more flexible and disentangled control. Then, we introduce MV-DiT (Motion-aware Video DiT). By designing unique motion attention with 4D positional encodings, MV-DiT can effectively leverage motion tokens as 4D compact yet expressive context for human image animation in the complex 3D world. Hence, it marks a significant step forward in this field and opens a new direction for pose-guided human video generation. Experiments show that our MTVCrafter achieves state-of-the-art results with an FID-VID of 6.98, surpassing the second-best by 65%. Powered by robust motion tokens, MTVCrafter also generalizes well to diverse open-world characters (single/multiple, full/half-body) across various styles and scenarios. Our video demos and code are on: https://github.com/DINGYANB/MTVCrafter.
>
---
#### [replaced 023] Leveraging Automatic CAD Annotations for Supervised Learning in 3D Scene Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13580v4](http://arxiv.org/pdf/2504.13580v4)**

> **作者:** Yuchen Rao; Stefan Ainetter; Sinisa Stekovic; Vincent Lepetit; Friedrich Fraundorfer
>
> **备注:** Project page: https://stefan-ainetter.github.io/SCANnotatepp; CVPR'25 Workshop
>
> **摘要:** High-level 3D scene understanding is essential in many applications. However, the challenges of generating accurate 3D annotations make development of deep learning models difficult. We turn to recent advancements in automatic retrieval of synthetic CAD models, and show that data generated by such methods can be used as high-quality ground truth for training supervised deep learning models. More exactly, we employ a pipeline akin to the one previously used to automatically annotate objects in ScanNet scenes with their 9D poses and CAD models. This time, we apply it to the recent ScanNet++ v1 dataset, which previously lacked such annotations. Our findings demonstrate that it is not only possible to train deep learning models on these automatically-obtained annotations but that the resulting models outperform those trained on manually annotated data. We validate this on two distinct tasks: point cloud completion and single-view CAD model retrieval and alignment. Our results underscore the potential of automatic 3D annotations to enhance model performance while significantly reducing annotation costs. To support future research in 3D scene understanding, we will release our annotations, which we call SCANnotate++, along with our trained models.
>
---
#### [replaced 024] DARTer: Dynamic Adaptive Representation Tracker for Nighttime UAV Tracking
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.00752v2](http://arxiv.org/pdf/2505.00752v2)**

> **作者:** Xuzhao Li; Xuchen Li; Shiyu Hu
>
> **备注:** Preprint, Under review
>
> **摘要:** Nighttime UAV tracking presents significant challenges due to extreme illumination variations and viewpoint changes, which severely degrade tracking performance. Existing approaches either rely on light enhancers with high computational costs or introduce redundant domain adaptation mechanisms, failing to fully utilize the dynamic features in varying perspectives. To address these issues, we propose \textbf{DARTer} (\textbf{D}ynamic \textbf{A}daptive \textbf{R}epresentation \textbf{T}racker), an end-to-end tracking framework designed for nighttime UAV scenarios. DARTer leverages a Dynamic Feature Blender (DFB) to effectively fuse multi-perspective nighttime features from static and dynamic templates, enhancing representation robustness. Meanwhile, a Dynamic Feature Activator (DFA) adaptively activates Vision Transformer layers based on extracted features, significantly improving efficiency by reducing redundant computations. Our model eliminates the need for complex multi-task loss functions, enabling a streamlined training process. Extensive experiments on multiple nighttime UAV tracking benchmarks demonstrate the superiority of DARTer over state-of-the-art trackers. These results confirm that DARTer effectively balances tracking accuracy and efficiency, making it a promising solution for real-world nighttime UAV tracking applications.
>
---
#### [replaced 025] L-WISE: Boosting Human Visual Category Learning Through Model-Based Image Selection and Enhancement
- **分类: cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2412.09765v4](http://arxiv.org/pdf/2412.09765v4)**

> **作者:** Morgan B. Talbot; Gabriel Kreiman; James J. DiCarlo; Guy Gaziv
>
> **摘要:** The currently leading artificial neural network models of the visual ventral stream - which are derived from a combination of performance optimization and robustification methods - have demonstrated a remarkable degree of behavioral alignment with humans on visual categorization tasks. We show that image perturbations generated by these models can enhance the ability of humans to accurately report the ground truth class. Furthermore, we find that the same models can also be used out-of-the-box to predict the proportion of correct human responses to individual images, providing a simple, human-aligned estimator of the relative difficulty of each image. Motivated by these observations, we propose to augment visual learning in humans in a way that improves human categorization accuracy at test time. Our learning augmentation approach consists of (i) selecting images based on their model-estimated recognition difficulty, and (ii) applying image perturbations that aid recognition for novice learners. We find that combining these model-based strategies leads to categorization accuracy gains of 33-72% relative to control subjects without these interventions, on unmodified, randomly selected held-out test images. Beyond the accuracy gain, the training time for the augmented learning group was also shortened by 20-23%, despite both groups completing the same number of training trials. We demonstrate the efficacy of our approach in a fine-grained categorization task with natural images, as well as two tasks in clinically relevant image domains - histology and dermoscopy - where visual learning is notoriously challenging. To the best of our knowledge, our work is the first application of artificial neural networks to increase visual learning performance in humans by enhancing category-specific image features.
>
---
#### [replaced 026] Just Functioning as a Hook for Two-Stage Referring Multi-Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07516v2](http://arxiv.org/pdf/2503.07516v2)**

> **作者:** Weize Li; Yunhao Du; Qixiang Yin; Zhicheng Zhao; Fei Su; Daqi Liu
>
> **摘要:** Referring Multi-Object Tracking (RMOT) aims to localize target trajectories specified by natural language expressions in videos. Existing RMOT methods mainly follow two paradigms: one-stage strategies and two-stage ones. The former jointly trains tracking with referring but suffers from substantial computational overhead. Although the latter improves efficiency, it overlooks the inherent contextual aggregation capabilities of pre-trained visual backbones and takes a detour. Meanwhile, its fixed dual-tower architecture restricts compatibility with other visual / text backbones. To address these limitations, we propose JustHook, a novel hook-like framework for two-stage RMOT, which introduces two core components: (1) a Visual Feature Hook (VFH), enabling JustHook to extract context-rich local features directly from the original visual backbone like a hook; (2) a Parallel Combined Decoder (PCD), which transforms the passive cosine similarity measurement between independent modalities into active contrastive learning within the combined feature space. The proposed JustHook not only leverages the capabilities of pre-trained models but also breaks free from the constraints of inherent modality alignment, achieving strong scalability. Extensive experiments on Refer-KITTI and Refer-KITTI-V2 demonstrate that JustHook outperforms state-of-the-art methods across diverse encoder combinations, achieving a notable 7.77\% HOTA improvement on Refer-KITTI-V2. Code will be made available soon.
>
---
#### [replaced 027] Two-Stage Random Alternation Framework for One-Shot Pansharpening
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.06576v2](http://arxiv.org/pdf/2505.06576v2)**

> **作者:** Haorui Chen; Zeyu Ren; Jiaxuan Ren; Ran Ran; Jinliang Shao; Jie Huang; Liangjian Deng
>
> **摘要:** Deep learning has substantially advanced pansharpening, achieving impressive fusion quality. However, a prevalent limitation is that conventional deep learning models, which typically rely on training datasets, often exhibit suboptimal generalization to unseen real-world image pairs. This restricts their practical utility when faced with real-world scenarios not included in the training datasets. To overcome this, we introduce a two-stage random alternating framework (TRA-PAN) that performs instance-specific optimization for any given Multispectral(MS)/Panchromatic(PAN) pair, ensuring robust and high-quality fusion. TRA-PAN effectively integrates strong supervision constraints from reduced-resolution images with the physical characteristics of the full-resolution images. The first stage introduces a pre-training procedure, which includes Degradation-Aware Modeling (DAM) to capture spectral degradation mappings, alongside a warm-up procedure designed to reduce training time and mitigate the adverse effects of reduced-resolution data. The second stage employs Random Alternation Optimization (RAO), randomly alternating between reduced- and full-resolution images to refine the fusion model progressively. This adaptive, per-instance optimization strategy, operating in a one-shot manner for each MS/PAN pair, yields superior high-resolution multispectral images. Experimental results demonstrate that TRA-PAN outperforms state-of-the-art (SOTA) methods in quantitative metrics and visual quality in real-world scenarios, underscoring its enhanced practical applicability and robustness.
>
---
#### [replaced 028] From Image to Video, what do we need in multimodal LLMs?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.11865v2](http://arxiv.org/pdf/2404.11865v2)**

> **作者:** Suyuan Huang; Haoxin Zhang; Linqing Zhong; Honggu Chen; Yan Gao; Yao Hu; Zengchang Qin
>
> **摘要:** Covering from Image LLMs to the more complex Video LLMs, the Multimodal Large Language Models (MLLMs) have demonstrated profound capabilities in comprehending cross-modal information as numerous studies have illustrated. Previous methods delve into designing comprehensive Video LLMs through integrating video foundation models with primitive LLMs. Despite its effectiveness, such paradigm renders Video LLM's structure verbose and typically requires substantial video data for pre-training. Crucially, it neglects leveraging the foundational contributions of ready-made Image LLMs. In this paper, we introduce RED-VILLM, a Resource-Efficient Development pipeline which builds robust Video LLMs through leveraging the prior knowledge of Image LLMs. Specifically, since a video is naturally a combination of images along the temporal dimension, we devise a temporal adaptation plug-and-play structure, endowing the backbone Image LLM with the capability to grasp temporal information. Moreover, through applying this pipeline, we achieve the first Video LLM within the Chinese-speaking community. Extensive experiments demonstrate that Video LLMs developed through our approach surpass conventional Video LLMs, requiring minimal instructional data and training resources. Our approach highlights the potential for a more cost-effective and scalable advancement in multimodal models.
>
---
#### [replaced 029] Are We Truly Forgetting? A Critical Re-examination of Machine Unlearning Evaluation Protocols
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06991v2](http://arxiv.org/pdf/2503.06991v2)**

> **作者:** Yongwoo Kim; Sungmin Cha; Donghyun Kim
>
> **摘要:** Machine unlearning is a process to remove specific data points from a trained model while maintaining the performance on retain data, addressing privacy or legal requirements. Despite its importance, existing unlearning evaluations tend to focus on logit-based metrics (i.e., accuracy) under small-scale scenarios. We observe that this could lead to a false sense of security in unlearning approaches under real-world scenarios. In this paper, we conduct a new comprehensive evaluation that employs representation-based evaluations of the unlearned model under large-scale scenarios to verify whether the unlearning approaches genuinely eliminate the targeted forget data from the model's representation perspective. Our analysis reveals that current state-of-the-art unlearning approaches either completely degrade the representational quality of the unlearned model or merely modify the classifier (i.e., the last layer), thereby achieving superior logit-based evaluation metrics while maintaining significant representational similarity to the original model. Furthermore, we introduce a rigorous unlearning evaluation setup, in which the forgetting classes exhibit semantic similarity to downstream task classes, necessitating that feature representations diverge significantly from those of the original model, thus enabling a more rigorous evaluation from a representation perspective. We hope our benchmark serves as a standardized protocol for evaluating unlearning algorithms under realistic conditions.
>
---
#### [replaced 030] HaHeAE: Learning Generalisable Joint Representations of Human Hand and Head Movements in Extended Reality
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.16430v3](http://arxiv.org/pdf/2410.16430v3)**

> **作者:** Zhiming Hu; Guanhua Zhang; Zheming Yin; Daniel Haeufle; Syn Schmitt; Andreas Bulling
>
> **备注:** Link: https://zhiminghu.net/hu25_haheae
>
> **摘要:** Human hand and head movements are the most pervasive input modalities in extended reality (XR) and are significant for a wide range of applications. However, prior works on hand and head modelling in XR only explored a single modality or focused on specific applications. We present HaHeAE - a novel self-supervised method for learning generalisable joint representations of hand and head movements in XR. At the core of our method is an autoencoder (AE) that uses a graph convolutional network-based semantic encoder and a diffusion-based stochastic encoder to learn the joint semantic and stochastic representations of hand-head movements. It also features a diffusion-based decoder to reconstruct the original signals. Through extensive evaluations on three public XR datasets, we show that our method 1) significantly outperforms commonly used self-supervised methods by up to 74.0% in terms of reconstruction quality and is generalisable across users, activities, and XR environments, 2) enables new applications, including interpretable hand-head cluster identification and variable hand-head movement generation, and 3) can serve as an effective feature extractor for downstream tasks. Together, these results demonstrate the effectiveness of our method and underline the potential of self-supervised methods for jointly modelling hand-head behaviours in extended reality.
>
---
#### [replaced 031] HiFlow: Training-free High-Resolution Image Generation with Flow-Aligned Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06232v2](http://arxiv.org/pdf/2504.06232v2)**

> **作者:** Jiazi Bu; Pengyang Ling; Yujie Zhou; Pan Zhang; Tong Wu; Xiaoyi Dong; Yuhang Zang; Yuhang Cao; Dahua Lin; Jiaqi Wang
>
> **备注:** Project Page: https://bujiazi.github.io/hiflow.github.io/
>
> **摘要:** Text-to-image (T2I) diffusion/flow models have drawn considerable attention recently due to their remarkable ability to deliver flexible visual creations. Still, high-resolution image synthesis presents formidable challenges due to the scarcity and complexity of high-resolution content. Recent approaches have investigated training-free strategies to enable high-resolution image synthesis with pre-trained models. However, these techniques often struggle with generating high-quality visuals and tend to exhibit artifacts or low-fidelity details, as they typically rely solely on the endpoint of the low-resolution sampling trajectory while neglecting intermediate states that are critical for preserving structure and synthesizing finer detail. To this end, we present HiFlow, a training-free and model-agnostic framework to unlock the resolution potential of pre-trained flow models. Specifically, HiFlow establishes a virtual reference flow within the high-resolution space that effectively captures the characteristics of low-resolution flow information, offering guidance for high-resolution generation through three key aspects: initialization alignment for low-frequency consistency, direction alignment for structure preservation, and acceleration alignment for detail fidelity. By leveraging such flow-aligned guidance, HiFlow substantially elevates the quality of high-resolution image synthesis of T2I models and demonstrates versatility across their personalized variants. Extensive experiments validate HiFlow's capability in achieving superior high-resolution image quality over state-of-the-art methods.
>
---
#### [replaced 032] Espresso: High Compression For Rich Extraction From Videos for Your Vision-Language Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.04729v3](http://arxiv.org/pdf/2412.04729v3)**

> **作者:** Keunwoo Peter Yu; Achal Dave; Rares Ambrus; Jean Mercat
>
> **备注:** 16 pages
>
> **摘要:** Recent advances in vision-language models (VLMs) have shown great promise in connecting images and text, but extending these models to long videos remains challenging due to the rapid growth in token counts. Models that compress videos by local aggregation in time or space have become popular for handling long-form inputs; however, these pooling-based projectors sacrifice the benefits of fixed-length representations that are crucial for streaming and efficient video understanding. We introduce $\texttt{Espresso}$, a new architecture that separately compresses spatial and temporal features into fixed-length sequences. $\texttt{Espresso}$ enables efficient video encoding while maintaining strong long-form reasoning capabilities. Experiments show that fixed-length compression combined with segment-wise processing offers a scalable and competitive alternative to pooling-based approaches. Our results demonstrate that fixed-length projectors, when properly designed and trained, remain a viable foundation for video-language modeling.
>
---
#### [replaced 033] Towards Low-Latency Event-based Obstacle Avoidance on a FPGA-Drone
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10400v2](http://arxiv.org/pdf/2504.10400v2)**

> **作者:** Pietro Bonazzi; Christian Vogt; Michael Jost; Lyes Khacef; Federico Paredes-Vallés; Michele Magno
>
> **摘要:** This work quantitatively evaluates the performance of event-based vision systems (EVS) against conventional RGB-based models for action prediction in collision avoidance on an FPGA accelerator. Our experiments demonstrate that the EVS model achieves a significantly higher effective frame rate (1 kHz) and lower temporal (-20 ms) and spatial prediction errors (-20 mm) compared to the RGB-based model, particularly when tested on out-of-distribution data. The EVS model also exhibits superior robustness in selecting optimal evasion maneuvers. In particular, in distinguishing between movement and stationary states, it achieves a 59 percentage point advantage in precision (78% vs. 19%) and a substantially higher F1 score (0.73 vs. 0.06), highlighting the susceptibility of the RGB model to overfitting. Further analysis in different combinations of spatial classes confirms the consistent performance of the EVS model in both test data sets. Finally, we evaluated the system end-to-end and achieved a latency of approximately 2.14 ms, with event aggregation (1 ms) and inference on the processing unit (0.94 ms) accounting for the largest components. These results underscore the advantages of event-based vision for real-time collision avoidance and demonstrate its potential for deployment in resource-constrained environments.
>
---
#### [replaced 034] Normalized Matching Transformer
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.17715v2](http://arxiv.org/pdf/2503.17715v2)**

> **作者:** Abtin Pourhadi; Paul Swoboda
>
> **摘要:** We present a new state of the art approach for sparse keypoint matching between pairs of images. Our method consists of a fully deep learning based approach combining a visual backbone coupled with a SplineCNN graph neural network for feature processing and a normalized transformer decoder for decoding keypoint correspondences together with the Sinkhorn algorithm. Our method is trained using a contrastive and a hyperspherical loss for better feature representations. We additionally use data augmentation during training. This comparatively simple architecture combining extensive normalization and advanced losses outperforms current state of the art approaches on PascalVOC and SPair-71k datasets by $5.1\%$ and $2.2\%$ respectively compared to BBGM, ASAR, COMMON and GMTR while training for at least $1.7x$ fewer epochs.
>
---
#### [replaced 035] Disentangling CLIP for Multi-Object Perception
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02977v3](http://arxiv.org/pdf/2502.02977v3)**

> **作者:** Samyak Rawlekar; Yujun Cai; Yiwei Wang; Ming-Hsuan Yang; Narendra Ahuja
>
> **摘要:** Vision-language models like CLIP excel at recognizing the single, prominent object in a scene. However, they struggle in complex scenes containing multiple objects. We identify a fundamental reason behind this limitation: VLMs features space exhibits significant semantic entanglement, where features of one class contain substantial information about other unrelated classes, a phenomenon we term mutual feature information (MFI). This entanglement becomes evident during class-specific queries, as unrelated objects are activated alongside the queried class. To address this limitation, we propose DCLIP, a framework that disentangles CLIP features using two complementary objectives: a novel MFI Loss that orthogonalizes the text (class) features to reduce inter-class similarity, and the Asymmetric Loss (ASL) that aligns image features with the disentangled text features. Our experiment demonstrates that DCLIP reduces inter-class feature similarity by 30\% compared to CLIP, leading to significant performance gains on multi-label recognition (MLR) and zero-shot semantic segmentation (ZS3). In MLR, DCLIP outperforms SOTA approaches on VOC2007 and COCO-14 while using 75\% fewer parameters, and surpasses SOTA ZS3 methods by 3.4 mIoU on VOC2012 and 2.8 mIoU on COCO-17. These results establish feature disentanglement as a critical factor for effective multi-object perception in vision-language models.
>
---
#### [replaced 036] CoMP: Continual Multimodal Pre-training for Vision Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18931v2](http://arxiv.org/pdf/2503.18931v2)**

> **作者:** Yitong Chen; Lingchen Meng; Wujian Peng; Zuxuan Wu; Yu-Gang Jiang
>
> **备注:** Code is available in https://github.com/SliMM-X/CoMP-MM
>
> **摘要:** Pre-trained Vision Foundation Models (VFMs) provide strong visual representations for a wide range of applications. In this paper, we continually pre-train prevailing VFMs in a multimodal manner such that they can effortlessly process visual inputs of varying sizes and produce visual representations that are more aligned with language representations, regardless of their original pre-training process. To this end, we introduce CoMP, a carefully designed multimodal pre-training pipeline. CoMP uses a Continual Rotary Position Embedding to accommodate visual inputs with different resolutions, and an Alignment Loss between visual and textual features for better cross-modal alignment. After continual pre-training, leading VFMs like DINOv2, SigLIP and AIMv2 achieve remarkable improvements not only in multimodal understanding tasks but also in generic classification and segmentation tasks. Remarkably, CoMP-AIMv2 achieves scores of 64.9 on ChartQA with a 0.5B LLM, while maintaining an 87.3% accuracy on ImageNet-1K and a 51.8 mIoU on ADE20K under frozen chunk evaluation.
>
---
#### [replaced 037] SynCL: A Synergistic Training Strategy with Instance-Aware Contrastive Learning for End-to-End Multi-Camera 3D Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.06780v3](http://arxiv.org/pdf/2411.06780v3)**

> **作者:** Shubo Lin; Yutong Kou; Zirui Wu; Shaoru Wang; Bing Li; Weiming Hu; Jin Gao
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** While existing query-based 3D end-to-end visual trackers integrate detection and tracking via the tracking-by-attention paradigm, these two chicken-and-egg tasks encounter optimization difficulties when sharing the same parameters. Our findings reveal that these difficulties arise due to two inherent constraints on the self-attention mechanism, i.e., over-deduplication for object queries and self-centric attention for track queries. In contrast, removing the self-attention mechanism not only minimally impacts regression predictions of the tracker, but also tends to generate more latent candidate boxes. Based on these analyses, we present SynCL, a novel plug-and-play synergistic training strategy designed to co-facilitate multi-task learning for detection and tracking. Specifically, we propose a Task-specific Hybrid Matching module for a weight-shared cross-attention-based decoder that matches the targets of track queries with multiple object queries to exploit promising candidates overlooked by the self-attention mechanism. To flexibly select optimal candidates for the one-to-many matching, we also design a Dynamic Query Filtering module controlled by model training status. Moreover, we introduce Instance-aware Contrastive Learning to break through the barrier of self-centric attention for track queries, effectively bridging the gap between detection and tracking. Without additional inference costs, SynCL consistently delivers improvements in various benchmarks and achieves state-of-the-art performance with $58.9\%$ AMOTA on the nuScenes dataset. Code and raw results will be publicly available.
>
---
#### [replaced 038] Positional Encoder Graph Quantile Neural Networks for Geographic Data
- **分类: stat.ML; cs.AI; cs.CV; cs.LG; cs.SI**

- **链接: [http://arxiv.org/pdf/2409.18865v2](http://arxiv.org/pdf/2409.18865v2)**

> **作者:** William E. R. de Amorim; Scott A. Sisson; T. Rodrigues; David J. Nott; Guilherme S. Rodrigues
>
> **备注:** 12 main text pages, 4 figures
>
> **摘要:** Positional Encoder Graph Neural Networks (PE-GNNs) are among the most effective models for learning from continuous spatial data. However, their predictive distributions are often poorly calibrated, limiting their utility in applications that require reliable uncertainty quantification. We propose the Positional Encoder Graph Quantile Neural Network (PE-GQNN), a novel framework that combines PE-GNNs with Quantile Neural Networks, partially monotonic neural blocks, and post-hoc recalibration techniques. The PE-GQNN enables flexible and robust conditional density estimation with minimal assumptions about the target distribution, and it extends naturally to tasks beyond spatial data. Empirical results on benchmark datasets show that the PE-GQNN outperforms existing methods in both predictive accuracy and uncertainty quantification, without incurring additional computational cost. We also provide theoretical insights and identify important special cases arising from our formulation, including the PE-GNN.
>
---
#### [replaced 039] FastVLM: Efficient Vision Encoding for Vision Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.13303v2](http://arxiv.org/pdf/2412.13303v2)**

> **作者:** Pavan Kumar Anasosalu Vasu; Fartash Faghri; Chun-Liang Li; Cem Koc; Nate True; Albert Antony; Gokul Santhanam; James Gabriel; Peter Grasch; Oncel Tuzel; Hadi Pouransari
>
> **备注:** CVPR 2025
>
> **摘要:** Scaling the input image resolution is essential for enhancing the performance of Vision Language Models (VLMs), particularly in text-rich image understanding tasks. However, popular visual encoders such as ViTs become inefficient at high resolutions due to the large number of tokens and high encoding latency caused by stacked self-attention layers. At different operational resolutions, the vision encoder of a VLM can be optimized along two axes: reducing encoding latency and minimizing the number of visual tokens passed to the LLM, thereby lowering overall latency. Based on a comprehensive efficiency analysis of the interplay between image resolution, vision latency, token count, and LLM size, we introduce FastVLM, a model that achieves an optimized trade-off between latency, model size and accuracy. FastVLM incorporates FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images. Unlike previous methods, FastVLM achieves the optimal balance between visual token count and image resolution solely by scaling the input image, eliminating the need for additional token pruning and simplifying the model design. In the LLaVA-1.5 setup, FastVLM achieves 3.2$\times$ improvement in time-to-first-token (TTFT) while maintaining similar performance on VLM benchmarks compared to prior works. Compared to LLaVa-OneVision at the highest resolution (1152$\times$1152), FastVLM achieves better performance on key benchmarks like SeedBench, MMMU and DocVQA, using the same 0.5B LLM, but with 85$\times$ faster TTFT and a vision encoder that is 3.4$\times$ smaller. Code and models are available at https://github.com/apple/ml-fastvlm.
>
---
#### [replaced 040] VideoHallu: Evaluating and Mitigating Multi-modal Hallucinations on Synthetic Video Understanding
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.01481v2](http://arxiv.org/pdf/2505.01481v2)**

> **作者:** Zongxia Li; Xiyang Wu; Guangyao Shi; Yubin Qin; Hongyang Du; Tianyi Zhou; Dinesh Manocha; Jordan Lee Boyd-Graber
>
> **摘要:** Synthetic video generation has gained significant attention for its realism and broad applications, but remains prone to violations of common sense and physical laws. This highlights the need for reliable abnormality detectors that understand such principles and are robust to hallucinations. To address this, we introduce VideoHallu, a benchmark of over 3,000 video QA pairs built from synthetic videos generated by models like Veo2, Sora, and Kling, paired with expert-crafted counterintuitive QA to evaluate the critical thinking abilities of Multi-modal Large Language Models (MLLMs) on abnormalities that are perceptually obvious to humans but often hallucinated due to language priors. VideoHallu evaluates MLLMs' abnormality detection abilities with examples across alignment, consistency, commonsense, and physics. We benchmark SOTA MLLMs, including GPT-4o, Gemini-2.5-Pro, Qwen2.5-VL, Video-R1, and VideoChat-R1. We observe that these models perform well on many real-world benchmarks like MVBench and MovieChat, but still struggle with basic physics-based and commonsense reasoning in synthetic videos. We further show that post-training with Group Relative Policy Optimization (GRPO), using curriculum learning on datasets combining video QA with counterintuitive commonsense and physics reasoning over real and synthetic videos, improves MLLMs' abnormality detection and critical thinking, demonstrating the value of targeted training for improving their understanding of commonsense and physical laws.
>
---
#### [replaced 041] In-Model Merging for Enhancing the Robustness of Medical Imaging Classification Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20516v2](http://arxiv.org/pdf/2502.20516v2)**

> **作者:** Hu Wang; Ibrahim Almakky; Congbo Ma; Numan Saeed; Mohammad Yaqub
>
> **摘要:** Model merging is an effective strategy to merge multiple models for enhancing model performances, and more efficient than ensemble learning as it will not introduce extra computation into inference. However, limited research explores if the merging process can occur within one model and enhance the model's robustness, which is particularly critical in the medical image domain. In the paper, we are the first to propose in-model merging (InMerge), a novel approach that enhances the model's robustness by selectively merging similar convolutional kernels in the deep layers of a single convolutional neural network (CNN) during the training process for classification. We also analytically reveal important characteristics that affect how in-model merging should be performed, serving as an insightful reference for the community. We demonstrate the feasibility and effectiveness of this technique for different CNN architectures on 4 prevalent datasets. The proposed InMerge-trained model surpasses the typically-trained model by a substantial margin. The code will be made public.
>
---
#### [replaced 042] reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2407.03653v5](http://arxiv.org/pdf/2407.03653v5)**

> **作者:** Kai Norman Clasen; Leonard Hackel; Tom Burgert; Gencer Sumbul; Begüm Demir; Volker Markl
>
> **备注:** Accepted at IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2025. Our code is available at https://github.com/rsim-tu-berlin/bigearthnet-pipeline
>
> **摘要:** This paper presents refined BigEarthNet (reBEN) that is a large-scale, multi-modal remote sensing dataset constructed to support deep learning (DL) studies for remote sensing image analysis. The reBEN dataset consists of 549,488 pairs of Sentinel-1 and Sentinel-2 image patches. To construct reBEN, we initially consider the Sentinel-1 and Sentinel-2 tiles used to construct the BigEarthNet dataset and then divide them into patches of size 1200 m x 1200 m. We apply atmospheric correction to the Sentinel-2 patches using the latest version of the sen2cor tool, resulting in higher-quality patches compared to those present in BigEarthNet. Each patch is then associated with a pixel-level reference map and scene-level multi-labels. This makes reBEN suitable for pixel- and scene-based learning tasks. The labels are derived from the most recent CORINE Land Cover (CLC) map of 2018 by utilizing the 19-class nomenclature as in BigEarthNet. The use of the most recent CLC map results in overcoming the label noise present in BigEarthNet. Furthermore, we introduce a new geographical-based split assignment algorithm that significantly reduces the spatial correlation among the train, validation, and test sets with respect to those present in BigEarthNet. This increases the reliability of the evaluation of DL models. To minimize the DL model training time, we introduce software tools that convert the reBEN dataset into a DL-optimized data format. In our experiments, we show the potential of reBEN for multi-modal multi-label image classification problems by considering several state-of-the-art DL models. The pre-trained model weights, associated code, and complete dataset are available at https://bigearth.net.
>
---
#### [replaced 043] Fast and Robust Localization for Humanoid Soccer Robot via Iterative Landmark Matching
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11020v2](http://arxiv.org/pdf/2503.11020v2)**

> **作者:** Ruochen Hou; Mingzhang Zhu; Hyunwoo Nam; Gabriel I. Fernandez; Dennis W. Hong
>
> **摘要:** Accurate robot localization is essential for effective operation. Monte Carlo Localization (MCL) is commonly used with known maps but is computationally expensive due to landmark matching for each particle. Humanoid robots face additional challenges, including sensor noise from locomotion vibrations and a limited field of view (FOV) due to camera placement. This paper proposes a fast and robust localization method via iterative landmark matching (ILM) for humanoid robots. The iterative matching process improves the accuracy of the landmark association so that it does not need MCL to match landmarks to particles. Pose estimation with the outlier removal process enhances its robustness to measurement noise and faulty detections. Furthermore, an additional filter can be utilized to fuse inertial data from the inertial measurement unit (IMU) and pose data from localization. We compared ILM with Iterative Closest Point (ICP), which shows that ILM method is more robust towards the error in the initial guess and easier to get a correct matching. We also compared ILM with the Augmented Monte Carlo Localization (aMCL), which shows that ILM method is much faster than aMCL and even more accurate. The proposed method's effectiveness is thoroughly evaluated through experiments and validated on the humanoid robot ARTEMIS during RoboCup 2024 adult-sized soccer competition.
>
---
#### [replaced 044] Resolving the Ambiguity of Complete-to-Partial Point Cloud Registration for Image-Guided Liver Surgery with Patches-to-Partial Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.19328v2](http://arxiv.org/pdf/2412.19328v2)**

> **作者:** Zixin Yang; Jon S. Heiselman; Cheng Han; Kelly Merrell; Richard Simon; Cristian. A. Linte
>
> **摘要:** In image-guided liver surgery, the initial rigid alignment between preoperative and intraoperative data, often represented as point clouds, is crucial for providing sub-surface information from preoperative CT/MRI images to the surgeon during the procedure. Currently, this alignment is typically performed using semi-automatic methods, which, while effective to some extent, are prone to errors that demand manual correction. Point cloud correspondence-based registration methods are promising to serve as a fully automatic solution. However, they may struggle in scenarios with limited intraoperative surface visibility, a common challenge in liver surgery, particularly in laparoscopic procedures, which we refer to as complete-to-partial ambiguity. We first illustrate this ambiguity by evaluating the performance of state-of-the-art learning-based point cloud registration methods on our carefully constructed in silico and in vitro datasets. Then, we propose a patches-to-partial matching strategy as a plug-and-play module to resolve the ambiguity, which can be seamlessly integrated into learning-based registration methods without disrupting their end-to-end structure. It has proven effective and efficient in improving registration performance for cases with limited intraoperative visibility. The constructed benchmark and the proposed module establish a solid foundation for advancing applications of point cloud correspondence-based registration methods in image-guided liver surgery.
>
---
#### [replaced 045] RefRef: A Synthetic Dataset and Benchmark for Reconstructing Refractive and Reflective Objects
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05848v2](http://arxiv.org/pdf/2505.05848v2)**

> **作者:** Yue Yin; Enze Tao; Weijian Deng; Dylan Campbell
>
> **摘要:** Modern 3D reconstruction and novel view synthesis approaches have demonstrated strong performance on scenes with opaque Lambertian objects. However, most assume straight light paths and therefore cannot properly handle refractive and reflective materials. Moreover, datasets specialized for these effects are limited, stymieing efforts to evaluate performance and develop suitable techniques. In this work, we introduce a synthetic RefRef dataset and benchmark for reconstructing scenes with refractive and reflective objects from posed images. Our dataset has 50 such objects of varying complexity, from single-material convex shapes to multi-material non-convex shapes, each placed in three different background types, resulting in 150 scenes. We also propose an oracle method that, given the object geometry and refractive indices, calculates accurate light paths for neural rendering, and an approach based on this that avoids these assumptions. We benchmark these against several state-of-the-art methods and show that all methods lag significantly behind the oracle, highlighting the challenges of the task and dataset.
>
---
#### [replaced 046] A Review on Discriminative Self-supervised Learning Methods in Computer Vision
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.04969v2](http://arxiv.org/pdf/2405.04969v2)**

> **作者:** Nikolaos Giakoumoglou; Tania Stathaki; Athanasios Gkelias
>
> **备注:** Preprint. 97 pages, 12 figures, 16 tables
>
> **摘要:** Self-supervised learning (SSL) has rapidly emerged as a transformative approach in computer vision, enabling the extraction of rich feature representations from vast amounts of unlabeled data and reducing reliance on costly manual annotations. This review presents a comprehensive analysis of discriminative SSL methods, which focus on learning representations by solving pretext tasks that do not require human labels. The paper systematically categorizes discriminative SSL approaches into five main groups: contrastive methods, clustering methods, self-distillation methods, knowledge distillation methods, and feature decorrelation methods. For each category, the review details the underlying principles, architectural components, loss functions, and representative algorithms, highlighting their unique mechanisms and contributions to the field. Extensive comparative evaluations are provided, including linear and semi-supervised protocols on standard benchmarks such as ImageNet, as well as transfer learning performance across diverse downstream tasks. The review also discusses theoretical foundations, scalability, efficiency, and practical challenges, such as computational demands and accessibility. By synthesizing recent advancements and identifying key trends, open challenges, and future research directions, this work serves as a valuable resource for researchers and practitioners aiming to leverage discriminative SSL for robust and generalizable computer vision models.
>
---
#### [replaced 047] INSIGHT: Enhancing Autonomous Driving Safety through Vision-Language Models on Context-Aware Hazard Detection and Edge Case Evaluation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.00262v3](http://arxiv.org/pdf/2502.00262v3)**

> **作者:** Dianwei Chen; Zifan Zhang; Yuchen Liu; Xianfeng Terry Yang
>
> **摘要:** Autonomous driving systems face significant challenges in handling unpredictable edge-case scenarios, such as adversarial pedestrian movements, dangerous vehicle maneuvers, and sudden environmental changes. Current end-to-end driving models struggle with generalization to these rare events due to limitations in traditional detection and prediction approaches. To address this, we propose INSIGHT (Integration of Semantic and Visual Inputs for Generalized Hazard Tracking), a hierarchical vision-language model (VLM) framework designed to enhance hazard detection and edge-case evaluation. By using multimodal data fusion, our approach integrates semantic and visual representations, enabling precise interpretation of driving scenarios and accurate forecasting of potential dangers. Through supervised fine-tuning of VLMs, we optimize spatial hazard localization using attention-based mechanisms and coordinate regression techniques. Experimental results on the BDD100K dataset demonstrate a substantial improvement in hazard prediction straightforwardness and accuracy over existing models, achieving a notable increase in generalization performance. This advancement enhances the robustness and safety of autonomous driving systems, ensuring improved situational awareness and potential decision-making in complex real-world scenarios.
>
---
#### [replaced 048] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13837v2](http://arxiv.org/pdf/2504.13837v2)**

> **作者:** Yang Yue; Zhiqi Chen; Rui Lu; Andrew Zhao; Zhaokai Wang; Yang Yue; Shiji Song; Gao Huang
>
> **备注:** 30 pages, 27 figures
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.
>
---
#### [replaced 049] Communication-Efficient Federated Learning Based on Explanation-Guided Pruning for Remote Sensing Image Classification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.11493v2](http://arxiv.org/pdf/2501.11493v2)**

> **作者:** Jonas Klotz; Barış Büyüktaş; Begüm Demir
>
> **备注:** Accepted at the IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2025
>
> **摘要:** Federated learning (FL) is a decentralized machine learning paradigm in which multiple clients collaboratively train a global model by exchanging only model updates with the central server without sharing the local data of the clients. Due to the large volume of model updates required to be transmitted between clients and the central server, most FL systems are associated with high transfer costs (i.e., communication overhead). This issue is more critical for operational applications in remote sensing (RS), especially when large-scale RS data is processed and analyzed through FL systems with restricted communication bandwidth. To address this issue, we introduce an explanation-guided pruning strategy for communication-efficient FL in the context of RS image classification. Our pruning strategy is defined based on the layer-wise relevance propagation (LRP) driven explanations to: 1) efficiently and effectively identify the most relevant and informative model parameters (to be exchanged between clients and the central server); and 2) eliminate the non-informative ones to minimize the volume of model updates. The experimental results on the BigEarthNet-S2 dataset demonstrate that our strategy effectively reduces the number of shared model updates, while increasing the generalization ability of the global model. The code of this work is publicly available at https://git.tu-berlin.de/rsim/FL-LRP.
>
---
#### [replaced 050] Visual Feedback of Pattern Separability Improves Myoelectric Decoding Performance of Upper Limb Prostheses
- **分类: cs.HC; cs.CV; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.09819v2](http://arxiv.org/pdf/2505.09819v2)**

> **作者:** Ruichen Yang; György M. Lévay; Christopher L. Hunt; Dániel Czeiner; Megan C. Hodgson; Damini Agarwal; Rahul R. Kaliki; Nitish V. Thakor
>
> **摘要:** State-of-the-art upper limb myoelectric prostheses often use pattern recognition (PR) control systems that translate electromyography (EMG) signals into desired movements. As prosthesis movement complexity increases, users often struggle to produce sufficiently distinct EMG patterns for reliable classification. Existing training typically involves heuristic, trial-and-error user adjustments to static decoder boundaries. Goal: We introduce the Reviewer, a 3D visual interface projecting EMG signals directly into the decoder's classification space, providing intuitive, real-time insight into PR algorithm behavior. This structured feedback reduces cognitive load and fosters mutual, data-driven adaptation between user-generated EMG patterns and decoder boundaries. Methods: A 10-session study with 12 able-bodied participants compared PR performance after motor-based training and updating using the Reviewer versus conventional virtual arm visualization. Performance was assessed using a Fitts law task that involved the aperture of the cursor and the control of orientation. Results: Participants trained with the Reviewer achieved higher completion rates, reduced overshoot, and improved path efficiency and throughput compared to the standard visualization group. Significance: The Reviewer introduces decoder-informed motor training, facilitating immediate and consistent PR-based myoelectric control improvements. By iteratively refining control through real-time feedback, this approach reduces reliance on trial-and-error recalibration, enabling a more adaptive, self-correcting training framework. Conclusion: The 3D visual feedback significantly improves PR control in novice operators through structured training, enabling feedback-driven adaptation and reducing reliance on extensive heuristic adjustments.
>
---
#### [replaced 051] VIN-NBV: A View Introspection Network for Next-Best-View Selection for Resource-Efficient 3D Reconstruction
- **分类: cs.CV; cs.RO; I.2.10; I.2.9**

- **链接: [http://arxiv.org/pdf/2505.06219v2](http://arxiv.org/pdf/2505.06219v2)**

> **作者:** Noah Frahm; Dongxu Zhao; Andrea Dunn Beltran; Ron Alterovitz; Jan-Michael Frahm; Junier Oliva; Roni Sengupta
>
> **备注:** The paper has not gone through legal review. We will update it with new version once the review is complete
>
> **摘要:** Next Best View (NBV) algorithms aim to acquire an optimal set of images using minimal resources, time, or number of captures to enable efficient 3D reconstruction of a scene. Existing approaches often rely on prior scene knowledge or additional image captures and often develop policies that maximize coverage. Yet, for many real scenes with complex geometry and self-occlusions, coverage maximization does not lead to better reconstruction quality directly. In this paper, we propose the View Introspection Network (VIN), which is trained to predict the reconstruction quality improvement of views directly, and the VIN-NBV policy. A greedy sequential sampling-based policy, where at each acquisition step, we sample multiple query views and choose the one with the highest VIN predicted improvement score. We design the VIN to perform 3D-aware featurization of the reconstruction built from prior acquisitions, and for each query view create a feature that can be decoded into an improvement score. We then train the VIN using imitation learning to predict the reconstruction improvement score. We show that VIN-NBV improves reconstruction quality by ~30% over a coverage maximization baseline when operating with constraints on the number of acquisitions or the time in motion.
>
---
#### [replaced 052] UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08787v3](http://arxiv.org/pdf/2505.08787v3)**

> **作者:** Hanjung Kim; Jaehyun Kang; Hyolim Kang; Meedeum Cho; Seon Joo Kim; Youngwoon Lee
>
> **备注:** Project Page: https://kimhanjung.github.io/UniSkill/
>
> **摘要:** Mimicry is a fundamental learning mechanism in humans, enabling individuals to learn new tasks by observing and imitating experts. However, applying this ability to robots presents significant challenges due to the inherent differences between human and robot embodiments in both their visual appearance and physical capabilities. While previous methods bridge this gap using cross-embodiment datasets with shared scenes and tasks, collecting such aligned data between humans and robots at scale is not trivial. In this paper, we propose UniSkill, a novel framework that learns embodiment-agnostic skill representations from large-scale cross-embodiment video data without any labels, enabling skills extracted from human video prompts to effectively transfer to robot policies trained only on robot data. Our experiments in both simulation and real-world environments show that our cross-embodiment skills successfully guide robots in selecting appropriate actions, even with unseen video prompts. The project website can be found at: https://kimhanjung.github.io/UniSkill.
>
---
#### [replaced 053] FreeA: Human-object Interaction Detection using Free Annotation Labels
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.01840v2](http://arxiv.org/pdf/2403.01840v2)**

> **作者:** Qi Liu; Yuxiao Wang; Xinyu Jiang; Wolin Liang; Zhenao Wei; Yu Lei; Nan Zhuang; Weiying Xue
>
> **摘要:** Recent human-object interaction (HOI) detection methods depend on extensively annotated image datasets, which require a significant amount of manpower. In this paper, we propose a novel self-adaptive, language-driven HOI detection method, termed FreeA. This method leverages the adaptability of the text-image model to generate latent HOI labels without requiring manual annotation. Specifically, FreeA aligns image features of human-object pairs with HOI text templates and employs a knowledge-based masking technique to decrease improbable interactions. Furthermore, FreeA implements a proposed method for matching interaction correlations to increase the probability of actions associated with a particular action, thereby improving the generated HOI labels. Experiments on two benchmark datasets showcase that FreeA achieves state-of-the-art performance among weakly supervised HOI competitors. Our proposal gets +\textbf{13.29} (\textbf{159\%$\uparrow$}) mAP and +\textbf{17.30} (\textbf{98\%$\uparrow$}) mAP than the newest ``Weakly'' supervised model, and +\textbf{7.19} (\textbf{28\%$\uparrow$}) mAP and +\textbf{14.69} (\textbf{34\%$\uparrow$}) mAP than the latest ``Weakly+'' supervised model, respectively, on HICO-DET and V-COCO datasets, more accurate in localizing and classifying the interactive actions. The source code will be made public.
>
---
#### [replaced 054] A Plasticity-Aware Method for Continual Self-Supervised Learning in Remote Sensing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.24088v2](http://arxiv.org/pdf/2503.24088v2)**

> **作者:** Lars Möllenbrok; Behnood Rasti; Begüm Demir
>
> **备注:** We found the reported results of the compared method to be misleading
>
> **摘要:** Continual self-supervised learning (CSSL) methods have gained increasing attention in remote sensing (RS) due to their capability to learn new tasks sequentially from continuous streams of unlabeled data. Existing CSSL methods, while learning new tasks, focus on preventing catastrophic forgetting. To this end, most of them use regularization strategies to retain knowledge of previous tasks. This reduces the model's ability to adapt to the data of new tasks (i.e., learning plasticity), which can degrade performance. To address this problem, in this paper, we propose a novel CSSL method that aims to learn tasks sequentially, while achieving high learning plasticity. To this end, the proposed method uses a knowledge distillation strategy with an integrated decoupling mechanism. The decoupling is achieved by first dividing the feature dimensions into task-common and task-specific parts. Then, the task-common features are forced to be correlated to ensure memory stability while the task-specific features are forced to be de-correlated facilitating the learning of new features. Experimental results show the effectiveness of the proposed method compared to CaSSLe, which is a widely used CSSL framework, with improvements of up to 1.12% in average accuracy and 2.33% in intransigence in a task-incremental scenario, and 1.24% in average accuracy and 2.01% in intransigence in a class-incremental scenario.
>
---
#### [replaced 055] A-I-RAVEN and I-RAVEN-Mesh: Two New Benchmarks for Abstract Visual Reasoning
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.11061v2](http://arxiv.org/pdf/2406.11061v2)**

> **作者:** Mikołaj Małkiński; Jacek Mańdziuk
>
> **备注:** Accepted to the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)
>
> **摘要:** We study generalization and knowledge reuse capabilities of deep neural networks in the domain of abstract visual reasoning (AVR), employing Raven's Progressive Matrices (RPMs), a recognized benchmark task for assessing AVR abilities. Two knowledge transfer scenarios referring to the I-RAVEN dataset are investigated. Firstly, inspired by generalization assessment capabilities of the PGM dataset and popularity of I-RAVEN, we introduce Attributeless-I-RAVEN (A-I-RAVEN), a benchmark with 10 generalization regimes that allow to systematically test generalization of abstract rules applied to held-out attributes at various levels of complexity (primary and extended regimes). In contrast to PGM, A-I-RAVEN features compositionality, a variety of figure configurations, and does not require substantial computational resources. Secondly, we construct I-RAVEN-Mesh, a dataset that enriches RPMs with a novel component structure comprising line-based patterns, facilitating assessment of progressive knowledge acquisition in transfer learning setting. We evaluate 13 strong models from the AVR literature on the introduced datasets, revealing their specific shortcomings in generalization and knowledge transfer.
>
---
#### [replaced 056] Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2403.11083v2](http://arxiv.org/pdf/2403.11083v2)**

> **作者:** Xiaohao Xu; Yunkang Cao; Huaxin Zhang; Nong Sang; Xiaonan Huang
>
> **备注:** Best Student Paper Award at IEEE International Conference on Computer Supported Cooperative Work in Design, 2025
>
> **摘要:** Anomaly detection is vital in various industrial scenarios, including the identification of unusual patterns in production lines and the detection of manufacturing defects for quality control. Existing techniques tend to be specialized in individual scenarios and lack generalization capacities. In this study, our objective is to develop a generic anomaly detection model that can be applied in multiple scenarios. To achieve this, we custom-build generic visual language foundation models that possess extensive knowledge and robust reasoning abilities as anomaly detectors and reasoners. Specifically, we introduce a multi-modal prompting strategy that incorporates domain knowledge from experts as conditions to guide the models. Our approach considers diverse prompt types, including task descriptions, class context, normality rules, and reference images. In addition, we unify the input representation of multi-modality into a 2D image format, enabling multi-modal anomaly detection and reasoning. Our preliminary studies demonstrate that combining visual and language prompts as conditions for customizing the models enhances anomaly detection performance. The customized models showcase the ability to detect anomalies across different data modalities such as images, point clouds, and videos. Qualitative case studies further highlight the anomaly detection and reasoning capabilities, particularly for multi-object scenes and temporal data. Our code is publicly available at https://github.com/Xiaohao-Xu/Customizable-VLM
>
---
#### [replaced 057] Self-Supervised Representation Learning for Nerve Fiber Distribution Patterns in 3D-PLI
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.17207v2](http://arxiv.org/pdf/2401.17207v2)**

> **作者:** Alexander Oberstrass; Sascha E. A. Muenzing; Meiqi Niu; Nicola Palomero-Gallagher; Christian Schiffer; Markus Axer; Katrin Amunts; Timo Dickscheid
>
> **备注:** Journal version
>
> **摘要:** A comprehensive understanding of the organizational principles in the human brain requires, among other factors, well-quantifiable descriptors of nerve fiber architecture. Three-dimensional polarized light imaging (3D-PLI) is a microscopic imaging technique that enables insights into the fine-grained organization of myelinated nerve fibers with high resolution. Descriptors characterizing the fiber architecture observed in 3D-PLI would enable downstream analysis tasks such as multimodal correlation studies, clustering, and mapping. However, best practices for observer-independent characterization of fiber architecture in 3D-PLI are not yet available. To this end, we propose the application of a fully data-driven approach to characterize nerve fiber architecture in 3D-PLI images using self-supervised representation learning. We introduce a 3D-Context Contrastive Learning (CL-3D) objective that utilizes the spatial neighborhood of texture examples across histological brain sections of a 3D reconstructed volume to sample positive pairs for contrastive learning. We combine this sampling strategy with specifically designed image augmentations to gain robustness to typical variations in 3D-PLI parameter maps. The approach is demonstrated for the 3D reconstructed occipital lobe of a vervet monkey brain. We show that extracted features are highly sensitive to different configurations of nerve fibers, yet robust to variations between consecutive brain sections arising from histological processing. We demonstrate their practical applicability for retrieving clusters of homogeneous fiber architecture, performing classification with minimal annotations, and query-based retrieval of characteristic components of fiber architecture such as U-fibers.
>
---
#### [replaced 058] From Pixels to Perception: Interpretable Predictions via Instance-wise Grouped Feature Selection
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.06003v2](http://arxiv.org/pdf/2505.06003v2)**

> **作者:** Moritz Vandenhirtz; Julia E. Vogt
>
> **备注:** International Conference on Machine Learning
>
> **摘要:** Understanding the decision-making process of machine learning models provides valuable insights into the task, the data, and the reasons behind a model's failures. In this work, we propose a method that performs inherently interpretable predictions through the instance-wise sparsification of input images. To align the sparsification with human perception, we learn the masking in the space of semantically meaningful pixel regions rather than on pixel-level. Additionally, we introduce an explicit way to dynamically determine the required level of sparsity for each instance. We show empirically on semi-synthetic and natural image datasets that our inherently interpretable classifier produces more meaningful, human-understandable predictions than state-of-the-art benchmarks.
>
---
#### [replaced 059] Words in Motion: Extracting Interpretable Control Vectors for Motion Transformers
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.11624v5](http://arxiv.org/pdf/2406.11624v5)**

> **作者:** Omer Sahin Tas; Royden Wagner
>
> **备注:** ICLR 2025 final version. Our implementation is available at https://github.com/kit-mrt/future-motion
>
> **摘要:** Transformer-based models generate hidden states that are difficult to interpret. In this work, we analyze hidden states and modify them at inference, with a focus on motion forecasting. We use linear probing to analyze whether interpretable features are embedded in hidden states. Our experiments reveal high probing accuracy, indicating latent space regularities with functionally important directions. Building on this, we use the directions between hidden states with opposing features to fit control vectors. At inference, we add our control vectors to hidden states and evaluate their impact on predictions. Remarkably, such modifications preserve the feasibility of predictions. We further refine our control vectors using sparse autoencoders (SAEs). This leads to more linear changes in predictions when scaling control vectors. Our approach enables mechanistic interpretation as well as zero-shot generalization to unseen dataset characteristics with negligible computational overhead.
>
---
#### [replaced 060] Inspiring the Next Generation of Segment Anything Models: Comprehensively Evaluate SAM and SAM 2 with Diverse Prompts Towards Context-Dependent Concepts under Different Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01240v2](http://arxiv.org/pdf/2412.01240v2)**

> **作者:** Xiaoqi Zhao; Youwei Pang; Shijie Chang; Yuan Zhao; Lihe Zhang; Huchuan Lu; Georges El Fakhri; Xiaofeng Liu
>
> **摘要:** As a foundational model, SAM has significantly influenced multiple fields within computer vision, and its upgraded version, SAM 2, enhances capabilities in video segmentation, poised to make a substantial impact once again. While SAMs (SAM and SAM 2) have demonstrated excellent performance in segmenting context-independent concepts like people, cars, and roads, they overlook more challenging context-dependent (CD) concepts, such as visual saliency, camouflage, product defects, and medical lesions. CD concepts rely heavily on global and local contextual information, making them susceptible to shifts in different contexts, which requires strong discriminative capabilities from the model. The lack of comprehensive evaluation of SAMs limits understanding of their performance boundaries, which may hinder the design of future models. In this paper, we conduct a thorough quantitative evaluation of SAMs on 11 CD concepts across 2D and 3D images and videos in various visual modalities within natural, medical, and industrial scenes. We develop a unified evaluation framework for SAM and SAM 2 that supports manual, automatic, and intermediate self-prompting, aided by our specific prompt generation and interaction strategies. We further explore the potential of SAM 2 for in-context learning and introduce prompt robustness testing to simulate real-world imperfect prompts. Finally, we analyze the benefits and limitations of SAMs in understanding CD concepts and discuss their future development in segmentation tasks. This work aims to provide valuable insights to guide future research in both context-independent and context-dependent concepts segmentation, potentially informing the development of the next version -- SAM 3.
>
---
#### [replaced 061] GLDiTalker: Speech-Driven 3D Facial Animation with Graph Latent Diffusion Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.01826v4](http://arxiv.org/pdf/2408.01826v4)**

> **作者:** Yihong Lin; Zhaoxin Fan; Xianjia Wu; Lingyu Xiong; Liang Peng; Xiandong Li; Wenxiong Kang; Songju Lei; Huang Xu
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Speech-driven talking head generation is a critical yet challenging task with applications in augmented reality and virtual human modeling. While recent approaches using autoregressive and diffusion-based models have achieved notable progress, they often suffer from modality inconsistencies, particularly misalignment between audio and mesh, leading to reduced motion diversity and lip-sync accuracy. To address this, we propose GLDiTalker, a novel speech-driven 3D facial animation model based on a Graph Latent Diffusion Transformer. GLDiTalker resolves modality misalignment by diffusing signals within a quantized spatiotemporal latent space. It employs a two-stage training pipeline: the Graph-Enhanced Quantized Space Learning Stage ensures lip-sync accuracy, while the Space-Time Powered Latent Diffusion Stage enhances motion diversity. Together, these stages enable GLDiTalker to generate realistic, temporally stable 3D facial animations. Extensive evaluations on standard benchmarks demonstrate that GLDiTalker outperforms existing methods, achieving superior results in both lip-sync accuracy and motion diversity.
>
---
#### [replaced 062] V-MAGE: A Game Evaluation Framework for Assessing Vision-Centric Capabilities in Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06148v2](http://arxiv.org/pdf/2504.06148v2)**

> **作者:** Xiangxi Zheng; Linjie Li; Zhengyuan Yang; Ping Yu; Alex Jinpeng Wang; Rui Yan; Yuan Yao; Lijuan Wang
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in visual-text processing. However, existing static image-text benchmarks are insufficient for evaluating their dynamic perception and interactive reasoning abilities. We introduce Vision-centric Multiple Abilities Game Evaluation(V-MAGE), a novel game-based evaluation framework designed to systematically assess MLLMs' visual reasoning in interactive, continuous-space environments. V-MAGE features five distinct video games comprising over 30 carefully constructed evaluation scenarios. These scenarios are set in free-form, visually complex environments that require models to interpret dynamic game states and make decisions based solely on visual input, thereby closely reflecting the conditions encountered by human players. To ensure robust and interpretable comparisons across models, V-MAGE employs a dynamic Elo-based ranking system that accounts for varying difficulty levels and task diversity. Benchmarking state-of-the-art MLLMs against human baselines reveals that while leading models approach human-level performance in simple tasks, their performance drops significantly in complex scenarios requiring advanced reasoning and task orchestration. This persistent performance gap highlights fundamental limitations in current MLLMs' ability to perform real-time, vision-grounded interactions. Through extensive analyses, we demonstrate the utility of V-MAGE in uncovering these limitations and providing actionable insights for improving the visual and reasoning capabilities of MLLMs in dynamic, interactive settings. Code is publicly available at https://github.com/CSU-JPG/V-MAGE.
>
---
