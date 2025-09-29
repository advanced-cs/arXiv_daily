# 计算机视觉 cs.CV

- **最新发布 190 篇**

- **更新 109 篇**

## 最新发布

#### [new 001] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DynaNav，用于高效视觉导航任务。针对现有模型计算开销大、可解释性差的问题，设计动态特征与层选择机制，结合稀疏操作和早退出机制，显著降低计算成本并提升性能。**

- **链接: [http://arxiv.org/pdf/2509.21930v1](http://arxiv.org/pdf/2509.21930v1)**

> **作者:** Jiahui Wang; Changhao Chen
>
> **备注:** Accepted as a poster in NeurIPS 2025
>
> **摘要:** Visual navigation is essential for robotics and embodied AI. However, existing foundation models, particularly those with transformer decoders, suffer from high computational overhead and lack interpretability, limiting their deployment in resource-tight scenarios. To address this, we propose DynaNav, a Dynamic Visual Navigation framework that adapts feature and layer selection based on scene complexity. It employs a trainable hard feature selector for sparse operations, enhancing efficiency and interpretability. Additionally, we integrate feature selection into an early-exit mechanism, with Bayesian Optimization determining optimal exit thresholds to reduce computational cost. Extensive experiments in real-world-based datasets and simulated environments demonstrate the effectiveness of DynaNav. Compared to ViNT, DynaNav achieves a 2.26x reduction in FLOPs, 42.3% lower inference time, and 32.8% lower memory usage, while improving navigation performance across four public datasets.
>
---
#### [new 002] Lightweight Structured Multimodal Reasoning for Clinical Scene Understanding in Robotics
- **分类: cs.CV; cs.AI; cs.HC; cs.RO**

- **简介: 该论文提出一种轻量级多模态框架，用于机器人临床场景理解。针对现有视觉-语言模型在时序推理和结构化输出的不足，结合Qwen2.5-VL与SmolAgent，实现动态工具调用与可解释推理，提升医疗机器人安全性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.22014v1](http://arxiv.org/pdf/2509.22014v1)**

> **作者:** Saurav Jha; Stefan K. Ehrlich
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Healthcare robotics requires robust multimodal perception and reasoning to ensure safety in dynamic clinical environments. Current Vision-Language Models (VLMs) demonstrate strong general-purpose capabilities but remain limited in temporal reasoning, uncertainty estimation, and structured outputs needed for robotic planning. We present a lightweight agentic multimodal framework for video-based scene understanding. Combining the Qwen2.5-VL-3B-Instruct model with a SmolAgent-based orchestration layer, it supports chain-of-thought reasoning, speech-vision fusion, and dynamic tool invocation. The framework generates structured scene graphs and leverages a hybrid retrieval module for interpretable and adaptive reasoning. Evaluations on the Video-MME benchmark and a custom clinical dataset show competitive accuracy and improved robustness compared to state-of-the-art VLMs, demonstrating its potential for applications in robot-assisted surgery, patient monitoring, and decision support.
>
---
#### [new 003] TUN3D: Towards Real-World Scene Understanding from Unposed Images
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出TUN3D，用于从无姿态图像中联合估计室内场景布局和3D物体检测。针对缺乏深度信息的问题，其采用稀疏卷积结构，在三个基准上取得SOTA性能，无需深度监督或相机姿态。**

- **链接: [http://arxiv.org/pdf/2509.21388v1](http://arxiv.org/pdf/2509.21388v1)**

> **作者:** Anton Konushin; Nikita Drozdov; Bulat Gabdullin; Alexey Zakharov; Anna Vorontsova; Danila Rukhovich; Maksim Kolodiazhnyi
>
> **摘要:** Layout estimation and 3D object detection are two fundamental tasks in indoor scene understanding. When combined, they enable the creation of a compact yet semantically rich spatial representation of a scene. Existing approaches typically rely on point cloud input, which poses a major limitation since most consumer cameras lack depth sensors and visual-only data remains far more common. We address this issue with TUN3D, the first method that tackles joint layout estimation and 3D object detection in real scans, given multi-view images as input, and does not require ground-truth camera poses or depth supervision. Our approach builds on a lightweight sparse-convolutional backbone and employs two dedicated heads: one for 3D object detection and one for layout estimation, leveraging a novel and effective parametric wall representation. Extensive experiments show that TUN3D achieves state-of-the-art performance across three challenging scene understanding benchmarks: (i) using ground-truth point clouds, (ii) using posed images, and (iii) using unposed images. While performing on par with specialized 3D object detection methods, TUN3D significantly advances layout estimation, setting a new benchmark in holistic indoor scene understanding. Code is available at https://github.com/col14m/tun3d .
>
---
#### [new 004] Large Material Gaussian Model for Relightable 3D Generation
- **分类: cs.CV**

- **简介: 该论文提出Large Material Gaussian Model (MGM)，旨在解决3D生成中材质属性建模不足的问题。通过多视角材质扩散模型与高斯表示，生成带PBR材质（如反照率、粗糙度、金属度）的高质量3D内容，支持动态光照渲染，提升真实感与实用性。**

- **链接: [http://arxiv.org/pdf/2509.22112v1](http://arxiv.org/pdf/2509.22112v1)**

> **作者:** Jingrui Ye; Lingting Zhu; Runze Zhang; Zeyu Hu; Yingda Yin; Lanjiong Li; Lequan Yu; Qingmin Liao
>
> **摘要:** The increasing demand for 3D assets across various industries necessitates efficient and automated methods for 3D content creation. Leveraging 3D Gaussian Splatting, recent large reconstruction models (LRMs) have demonstrated the ability to efficiently achieve high-quality 3D rendering by integrating multiview diffusion for generation and scalable transformers for reconstruction. However, existing models fail to produce the material properties of assets, which is crucial for realistic rendering in diverse lighting environments. In this paper, we introduce the Large Material Gaussian Model (MGM), a novel framework designed to generate high-quality 3D content with Physically Based Rendering (PBR) materials, ie, albedo, roughness, and metallic properties, rather than merely producing RGB textures with uncontrolled light baking. Specifically, we first fine-tune a new multiview material diffusion model conditioned on input depth and normal maps. Utilizing the generated multiview PBR images, we explore a Gaussian material representation that not only aligns with 2D Gaussian Splatting but also models each channel of the PBR materials. The reconstructed point clouds can then be rendered to acquire PBR attributes, enabling dynamic relighting by applying various ambient light maps. Extensive experiments demonstrate that the materials produced by our method not only exhibit greater visual appeal compared to baseline methods but also enhance material modeling, thereby enabling practical downstream rendering applications.
>
---
#### [new 005] Effectiveness of Large Multimodal Models in Detecting Disinformation: Experimental Results
- **分类: cs.CV**

- **简介: 该论文研究了大型多模态模型（GPT-4o）在检测虚假信息中的有效性，旨在解决多模态环境下文本与图像结合的虚假信息识别问题。工作包括优化提示设计、构建分析框架、定义评估标准，并在多个数据集上进行性能分析与稳定性测试。**

- **链接: [http://arxiv.org/pdf/2509.22377v1](http://arxiv.org/pdf/2509.22377v1)**

> **作者:** Yasmina Kheddache; Marc Lalonde
>
> **备注:** 9 pages
>
> **摘要:** The proliferation of disinformation, particularly in multimodal contexts combining text and images, presents a significant challenge across digital platforms. This study investigates the potential of large multimodal models (LMMs) in detecting and mitigating false information. We propose to approach multimodal disinformation detection by leveraging the advanced capabilities of the GPT-4o model. Our contributions include: (1) the development of an optimized prompt incorporating advanced prompt engineering techniques to ensure precise and consistent evaluations; (2) the implementation of a structured framework for multimodal analysis, including a preprocessing methodology for images and text to comply with the model's token limitations; (3) the definition of six specific evaluation criteria that enable a fine-grained classification of content, complemented by a self-assessment mechanism based on confidence levels; (4) a comprehensive performance analysis of the model across multiple heterogeneous datasets Gossipcop, Politifact, Fakeddit, MMFakeBench, and AMMEBA highlighting GPT-4o's strengths and limitations in disinformation detection; (5) an investigation of prediction variability through repeated testing, evaluating the stability and reliability of the model's classifications; and (6) the introduction of confidence-level and variability-based evaluation methods. These contributions provide a robust and reproducible methodological framework for automated multimodal disinformation analysis.
>
---
#### [new 006] Dynamic Novel View Synthesis in High Dynamic Range
- **分类: cs.CV**

- **简介: 该论文提出HDR-4DGS，解决动态场景下从LDR图像生成HDR新视角的问题。通过引入动态色调映射模块，联合建模时空辐射变化与LDR-HDR转换，实现时序辐射一致性与高保真HDR渲染。**

- **链接: [http://arxiv.org/pdf/2509.21853v1](http://arxiv.org/pdf/2509.21853v1)**

> **作者:** Kaixuan Zhang; Zhipeng Xiong; Minxian Li; Mingwu Ren; Jiankang Deng; Xiatian Zhu
>
> **摘要:** High Dynamic Range Novel View Synthesis (HDR NVS) seeks to learn an HDR 3D model from Low Dynamic Range (LDR) training images captured under conventional imaging conditions. Current methods primarily focus on static scenes, implicitly assuming all scene elements remain stationary and non-living. However, real-world scenarios frequently feature dynamic elements, such as moving objects, varying lighting conditions, and other temporal events, thereby presenting a significantly more challenging scenario. To address this gap, we propose a more realistic problem named HDR Dynamic Novel View Synthesis (HDR DNVS), where the additional dimension ``Dynamic'' emphasizes the necessity of jointly modeling temporal radiance variations alongside sophisticated 3D translation between LDR and HDR. To tackle this complex, intertwined challenge, we introduce HDR-4DGS, a Gaussian Splatting-based architecture featured with an innovative dynamic tone-mapping module that explicitly connects HDR and LDR domains, maintaining temporal radiance coherence by dynamically adapting tone-mapping functions according to the evolving radiance distributions across the temporal dimension. As a result, HDR-4DGS achieves both temporal radiance consistency and spatially accurate color translation, enabling photorealistic HDR renderings from arbitrary viewpoints and time instances. Extensive experiments demonstrate that HDR-4DGS surpasses existing state-of-the-art methods in both quantitative performance and visual fidelity. Source code will be released.
>
---
#### [new 007] Coreset selection based on Intra-class diversity
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于深度学习中的数据集子集选择任务，旨在解决随机采样在不平衡数据集中无法保留类内多样性的不足。提出了一种基于类内多样性聚类的轻量级核心集选择方法，并在医学图像分类中验证了其优于随机采样的性能。**

- **链接: [http://arxiv.org/pdf/2509.21380v1](http://arxiv.org/pdf/2509.21380v1)**

> **作者:** Imran Ashraf; Mukhtar Ullah; Muhammad Faisal Nadeem; Muhammad Nouman Noor
>
> **摘要:** Deep Learning models have transformed various domains, including the healthcare sector, particularly biomedical image classification by learning intricate features and enabling accurate diagnostics pertaining to complex diseases. Recent studies have adopted two different approaches to train DL models: training from scratch and transfer learning. Both approaches demand substantial computational time and resources due to the involvement of massive datasets in model training. These computational demands are further increased due to the design-space exploration required for selecting optimal hyperparameters, which typically necessitates several training rounds. With the growing sizes of datasets, exploring solutions to this problem has recently gained the research community's attention. A plausible solution is to select a subset of the dataset for training and hyperparameter search. This subset, referred to as the corset, must be a representative set of the original dataset. A straightforward approach to selecting the coreset could be employing random sampling, albeit at the cost of compromising the representativeness of the original dataset. A critical limitation of random sampling is the bias towards the dominant classes in an imbalanced dataset. Even if the dataset has inter-class balance, this random sampling will not capture intra-class diversity. This study addresses this issue by introducing an intelligent, lightweight mechanism for coreset selection. Specifically, it proposes a method to extract intra-class diversity, forming per-class clusters that are utilized for the final sampling. We demonstrate the efficacy of the proposed methodology by conducting extensive classification experiments on a well-known biomedical imaging dataset. Results demonstrate that the proposed scheme outperforms the random sampling approach on several performance metrics for uniform conditions.
>
---
#### [new 008] Text Adversarial Attacks with Dynamic Outputs
- **分类: cs.CV**

- **简介: 该论文研究文本对抗攻击任务，旨在解决动态输出场景下攻击方法不足的问题。提出TDOA方法，通过聚类模型训练和最远标签策略提升攻击效果，并扩展至生成式任务，实验表明其在静态和动态场景中均表现优异。**

- **链接: [http://arxiv.org/pdf/2509.22393v1](http://arxiv.org/pdf/2509.22393v1)**

> **作者:** Wenqiang Wang; Siyuan Liang; Xiao Yan; Xiaochun Cao
>
> **摘要:** Text adversarial attack methods are typically designed for static scenarios with fixed numbers of output labels and a predefined label space, relying on extensive querying of the victim model (query-based attacks) or the surrogate model (transfer-based attacks). To address this gap, we introduce the Textual Dynamic Outputs Attack (TDOA) method, which employs a clustering-based surrogate model training approach to convert the dynamic-output scenario into a static single-output scenario. To improve attack effectiveness, we propose the farthest-label targeted attack strategy, which selects adversarial vectors that deviate most from the model's coarse-grained labels, thereby maximizing disruption. We extensively evaluate TDOA on four datasets and eight victim models (e.g., ChatGPT-4o, ChatGPT-4.1), showing its effectiveness in crafting adversarial examples and its strong potential to compromise large language models with limited access. With a single query per text, TDOA achieves a maximum attack success rate of 50.81\%. Additionally, we find that TDOA also achieves state-of-the-art performance in conventional static output scenarios, reaching a maximum ASR of 82.68\%. Meanwhile, by conceptualizing translation tasks as classification problems with unbounded output spaces, we extend the TDOA framework to generative settings, surpassing prior results by up to 0.64 RDBLEU and 0.62 RDchrF.
>
---
#### [new 009] SpecXNet: A Dual-Domain Convolutional Network for Robust Deepfake Detection
- **分类: cs.CV**

- **简介: 该论文提出SpecXNet，用于鲁棒的深度伪造检测。针对现有方法仅依赖单一空间或频域特征导致泛化能力差的问题，设计了双域特征耦合器（DDFC）和双傅里叶注意力（DFA）模块，融合局部纹理与全局频谱信息，提升检测性能与泛化性。**

- **链接: [http://arxiv.org/pdf/2509.22070v1](http://arxiv.org/pdf/2509.22070v1)**

> **作者:** Inzamamul Alam; Md Tanvir Islam; Simon S. Woo
>
> **备注:** ACM MM Accepted
>
> **摘要:** The increasing realism of content generated by GANs and diffusion models has made deepfake detection significantly more challenging. Existing approaches often focus solely on spatial or frequency-domain features, limiting their generalization to unseen manipulations. We propose the Spectral Cross-Attentional Network (SpecXNet), a dual-domain architecture for robust deepfake detection. The core \textbf{Dual-Domain Feature Coupler (DDFC)} decomposes features into a local spatial branch for capturing texture-level anomalies and a global spectral branch that employs Fast Fourier Transform to model periodic inconsistencies. This dual-domain formulation allows SpecXNet to jointly exploit localized detail and global structural coherence, which are critical for distinguishing authentic from manipulated images. We also introduce the \textbf{Dual Fourier Attention (DFA)} module, which dynamically fuses spatial and spectral features in a content-aware manner. Built atop a modified XceptionNet backbone, we embed the DDFC and DFA modules within a separable convolution block. Extensive experiments on multiple deepfake benchmarks show that SpecXNet achieves state-of-the-art accuracy, particularly under cross-dataset and unseen manipulation scenarios, while maintaining real-time feasibility. Our results highlight the effectiveness of unified spatial-spectral learning for robust and generalizable deepfake detection. To ensure reproducibility, we released the full code on \href{https://github.com/inzamamulDU/SpecXNet}{\textcolor{blue}{\textbf{GitHub}}}.
>
---
#### [new 010] Color Names in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文研究视觉-语言模型（VLMs）的颜色命名能力，属于自然语言处理与计算机视觉交叉任务。旨在探究VLMs是否像人类一样命名颜色，通过系统评估五种代表性模型在957个颜色样本上的表现，揭示其在典型与非典型颜色上的差异，并分析语言架构和训练数据对颜色命名的影响。**

- **链接: [http://arxiv.org/pdf/2509.22524v1](http://arxiv.org/pdf/2509.22524v1)**

> **作者:** Alexandra Gomez-Villa; Pablo Hernández-Cámara; Muhammad Atif Butt; Valero Laparra; Jesus Malo; Javier Vazquez-Corral
>
> **摘要:** Color serves as a fundamental dimension of human visual perception and a primary means of communicating about objects and scenes. As vision-language models (VLMs) become increasingly prevalent, understanding whether they name colors like humans is crucial for effective human-AI interaction. We present the first systematic evaluation of color naming capabilities across VLMs, replicating classic color naming methodologies using 957 color samples across five representative models. Our results show that while VLMs achieve high accuracy on prototypical colors from classical studies, performance drops significantly on expanded, non-prototypical color sets. We identify 21 common color terms that consistently emerge across all models, revealing two distinct approaches: constrained models using predominantly basic terms versus expansive models employing systematic lightness modifiers. Cross-linguistic analysis across nine languages demonstrates severe training imbalances favoring English and Chinese, with hue serving as the primary driver of color naming decisions. Finally, ablation studies reveal that language model architecture significantly influences color naming independent of visual processing capabilities.
>
---
#### [new 011] MORPH: Shape-agnostic PDE Foundation Models
- **分类: cs.CV; cs.AI; cs.LG; physics.comp-ph**

- **简介: 该论文提出MORPH，一种面向偏微分方程（PDE）的形状无关基础模型。它基于卷积视觉Transformer，处理多维时空数据，结合局部交互、跨场注意力和轴向注意力，实现高效建模。通过预训练与微调，在多种预测任务中表现优异，推动科学机器学习的数据效率与可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.21670v1](http://arxiv.org/pdf/2509.21670v1)**

> **作者:** Mahindra Singh Rautela; Alexander Most; Siddharth Mansingh; Bradley C. Love; Ayan Biswas; Diane Oyen; Earl Lawrence
>
> **摘要:** We introduce MORPH, a shape-agnostic, autoregressive foundation model for partial differential equations (PDEs). MORPH is built on a convolutional vision transformer backbone that seamlessly handles heterogeneous spatiotemporal datasets of varying data dimensionality (1D--3D) at different resolutions, multiple fields with mixed scalar and vector components. The architecture combines (i) component-wise convolution, which jointly processes scalar and vector channels to capture local interactions, (ii) inter-field cross-attention, which models and selectively propagates information between different physical fields, (iii) axial attentions, which factorizes full spatiotemporal self-attention along individual spatial and temporal axes to reduce computational burden while retaining expressivity. We pretrain multiple model variants on a diverse collection of heterogeneous PDE datasets and evaluate transfer to a range of downstream prediction tasks. Using both full-model fine-tuning and parameter-efficient low-rank adapters (LoRA), MORPH outperforms models trained from scratch in both zero-shot and full-shot generalization. Across extensive evaluations, MORPH matches or surpasses strong baselines and recent state-of-the-art models. Collectively, these capabilities present a flexible and powerful backbone for learning from heterogeneous and multimodal nature of scientific observations, charting a path toward scalable and data-efficient scientific machine learning.
>
---
#### [new 012] JaiLIP: Jailbreaking Vision-Language Models via Loss Guided Image Perturbation
- **分类: cs.CV**

- **简介: 该论文提出JaiLIP方法，用于通过图像扰动破解视觉-语言模型（VLMs），解决现有攻击方法效果不稳定、扰动明显的问题。方法结合MSE损失与有害输出损失生成不可察觉的对抗图像，并在毒性文本和交通领域验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.21401v1](http://arxiv.org/pdf/2509.21401v1)**

> **作者:** Md Jueal Mia; M. Hadi Amini
>
> **摘要:** Vision-Language Models (VLMs) have remarkable abilities in generating multimodal reasoning tasks. However, potential misuse or safety alignment concerns of VLMs have increased significantly due to different categories of attack vectors. Among various attack vectors, recent studies have demonstrated that image-based perturbations are particularly effective in generating harmful outputs. In the literature, many existing techniques have been proposed to jailbreak VLMs, leading to unstable performance and visible perturbations. In this study, we propose Jailbreaking with Loss-guided Image Perturbation (JaiLIP), a jailbreaking attack in the image space that minimizes a joint objective combining the mean squared error (MSE) loss between clean and adversarial image with the models harmful-output loss. We evaluate our proposed method on VLMs using standard toxicity metrics from Perspective API and Detoxify. Experimental results demonstrate that our method generates highly effective and imperceptible adversarial images, outperforming existing methods in producing toxicity. Moreover, we have evaluated our method in the transportation domain to demonstrate the attacks practicality beyond toxic text generation in specific domain. Our findings emphasize the practical challenges of image-based jailbreak attacks and the need for efficient defense mechanisms for VLMs.
>
---
#### [new 013] WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出WAVE，一种基于多模态大模型的统一音频-视觉嵌入方法。旨在解决动态模态（如音视频）表示不足的问题，通过层次特征融合和联合多任务训练，实现任意模态间检索与指令感知嵌入生成，提升跨模态应用效果。**

- **链接: [http://arxiv.org/pdf/2509.21990v1](http://arxiv.org/pdf/2509.21990v1)**

> **作者:** Changli Tang; Qinfan Xiao; Ke Mei; Tianyi Wang; Fengyun Rao; Chao Zhang
>
> **摘要:** While embeddings from multimodal large language models (LLMs) excel as general-purpose representations, their application to dynamic modalities like audio and video remains underexplored. We introduce WAVE (\textbf{u}nified \& \textbf{v}ersatile \textbf{a}udio-\textbf{v}isual \textbf{e}mbeddings), the first LLM-based embedding that creates a unified representation space for text, audio, and video modalities. WAVE employs a novel hierarchical feature fusion strategy and a joint multi-modal, multi-task training approach to enable two key capabilities: any-to-any cross-modal retrieval and the generation of prompt-aware embeddings tailored to user instructions. Experimentally, WAVE sets a new state-of-the-art on the MMEB-v2 video benchmark and achieves superior results in audio and video-to-audio retrieval. Its prompt-aware nature also yields remarkable performance in multimodal question answering, significantly outperforming existing embedding models. Ablation studies validate our joint training strategy, demonstrating improved performance across all modalities. With a newly introduced benchmark for versatile audio-visual learning, WAVE opens up broad possibilities for cross-modal, any-to-any applications. Our code, checkpoints, and data will be released.
>
---
#### [new 014] Category Discovery: An Open-World Perspective
- **分类: cs.CV**

- **简介: 该论文综述了开放世界下的类别发现（CD）任务，旨在利用已标记数据自动对未标记的未知类别数据进行分类。论文提出了NCD和GCD两种基础设定及多种扩展场景，分析了方法的核心组件，并总结了关键结论与未来方向。**

- **链接: [http://arxiv.org/pdf/2509.22542v1](http://arxiv.org/pdf/2509.22542v1)**

> **作者:** Zhenqi He; Yuanpei Liu; Kai Han
>
> **摘要:** Category discovery (CD) is an emerging open-world learning task, which aims at automatically categorizing unlabelled data containing instances from unseen classes, given some labelled data from seen classes. This task has attracted significant attention over the years and leads to a rich body of literature trying to address the problem from different perspectives. In this survey, we provide a comprehensive review of the literature, and offer detailed analysis and in-depth discussion on different methods. Firstly, we introduce a taxonomy for the literature by considering two base settings, namely novel category discovery (NCD) and generalized category discovery (GCD), and several derived settings that are designed to address the extra challenges in different real-world application scenarios, including continual category discovery, skewed data distribution, federated category discovery, etc. Secondly, for each setting, we offer a detailed analysis of the methods encompassing three fundamental components, representation learning, label assignment, and estimation of class number. Thirdly, we benchmark all the methods and distill key insights showing that large-scale pretrained backbones, hierarchical and auxiliary cues, and curriculum-style training are all beneficial for category discovery, while challenges remain in the design of label assignment, the estimation of class numbers, and scaling to complex multi-object scenarios.Finally, we discuss the key insights from the literature so far and point out promising future research directions. We compile a living survey of the category discovery literature at \href{https://github.com/Visual-AI/Category-Discovery}{https://github.com/Visual-AI/Category-Discovery}.
>
---
#### [new 015] LongScape: Advancing Long-Horizon Embodied World Models with Context-Aware MoE
- **分类: cs.CV**

- **简介: 该论文提出LongScape，用于解决长时序视频生成中的时间不一致和视觉漂移问题。通过结合分块扩散与自回归生成，并引入上下文感知的专家混合框架，实现了高质量、稳定的长时序动作生成。**

- **链接: [http://arxiv.org/pdf/2509.21790v1](http://arxiv.org/pdf/2509.21790v1)**

> **作者:** Yu Shang; Lei Jin; Yiding Ma; Xin Zhang; Chen Gao; Wei Wu; Yong Li
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** Video-based world models hold significant potential for generating high-quality embodied manipulation data. However, current video generation methods struggle to achieve stable long-horizon generation: classical diffusion-based approaches often suffer from temporal inconsistency and visual drift over multiple rollouts, while autoregressive methods tend to compromise on visual detail. To solve this, we introduce LongScape, a hybrid framework that adaptively combines intra-chunk diffusion denoising with inter-chunk autoregressive causal generation. Our core innovation is an action-guided, variable-length chunking mechanism that partitions video based on the semantic context of robotic actions. This ensures each chunk represents a complete, coherent action, enabling the model to flexibly generate diverse dynamics. We further introduce a Context-aware Mixture-of-Experts (CMoE) framework that adaptively activates specialized experts for each chunk during generation, guaranteeing high visual quality and seamless chunk transitions. Extensive experimental results demonstrate that our method achieves stable and consistent long-horizon generation over extended rollouts. Our code is available at: https://github.com/tsinghua-fib-lab/Longscape.
>
---
#### [new 016] Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究音频-视觉导航任务，旨在解决多模态信息融合不足导致的导航效率问题。提出DMTF-AVN方法，通过多目标架构与改进Transformer机制实现高效跨模态融合，在多个指标上达到最优。**

- **链接: [http://arxiv.org/pdf/2509.21377v1](http://arxiv.org/pdf/2509.21377v1)**

> **作者:** Yinfeng Yu; Hailong Zhang; Meiling Zhu
>
> **备注:** Main paper (8 pages). Accepted for publication by ECAI( European Conference on Artificial Intelligence) 2025
>
> **摘要:** Audiovisual embodied navigation enables robots to locate audio sources by dynamically integrating visual observations from onboard sensors with the auditory signals emitted by the target. The core challenge lies in effectively leveraging multimodal cues to guide navigation. While prior works have explored basic fusion of visual and audio data, they often overlook deeper perceptual context. To address this, we propose the Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation (DMTF-AVN). Our approach uses a multi-target architecture coupled with a refined Transformer mechanism to filter and selectively fuse cross-modal information. Extensive experiments on the Replica and Matterport3D datasets demonstrate that DMTF-AVN achieves state-of-the-art performance, outperforming existing methods in success rate (SR), path efficiency (SPL), and scene adaptation (SNA). Furthermore, the model exhibits strong scalability and generalizability, paving the way for advanced multimodal fusion strategies in robotic navigation. The code and videos are available at https://github.com/zzzmmm-svg/DMTF.
>
---
#### [new 017] Deepfakes: we need to re-think the concept of "real" images
- **分类: cs.CV**

- **简介: 该论文属于图像真实性研究领域，旨在解决“真实图像”定义模糊的问题。作者指出当前伪造图像检测依赖过时数据集，而真实图像生成技术已变化巨大，呼吁重新定义“真实图像”并建立新基准数据集。**

- **链接: [http://arxiv.org/pdf/2509.21864v1](http://arxiv.org/pdf/2509.21864v1)**

> **作者:** Janis Keuper; Margret Keuper
>
> **摘要:** The wide availability and low usability barrier of modern image generation models has triggered the reasonable fear of criminal misconduct and negative social implications. The machine learning community has been engaging this problem with an extensive series of publications proposing algorithmic solutions for the detection of "fake", e.g. entirely generated or partially manipulated images. While there is undoubtedly some progress towards technical solutions of the problem, we argue that current and prior work is focusing too much on generative algorithms and "fake" data-samples, neglecting a clear definition and data collection of "real" images. The fundamental question "what is a real image?" might appear to be quite philosophical, but our analysis shows that the development and evaluation of basically all current "fake"-detection methods is relying on only a few, quite old low-resolution datasets of "real" images like ImageNet. However, the technology for the acquisition of "real" images, aka taking photos, has drastically evolved over the last decade: Today, over 90% of all photographs are produced by smartphones which typically use algorithms to compute an image from multiple inputs (over time) from multiple sensors. Based on the fact that these image formation algorithms are typically neural network architectures which are closely related to "fake"-image generators, we state the position that today, we need to re-think the concept of "real" images. The purpose of this position paper is to raise the awareness of the current shortcomings in this active field of research and to trigger an open discussion whether the detection of "fake" images is a sound objective at all. At the very least, we need a clear technical definition of "real" images and new benchmark datasets.
>
---
#### [new 018] RAU: Reference-based Anatomical Understanding with Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出RAU框架，利用视觉语言模型（VLM）结合参考图像进行医学图像的解剖理解任务，解决标注数据稀缺问题。通过相对空间推理和SAM2整合，实现定位与分割，提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.22404v1](http://arxiv.org/pdf/2509.22404v1)**

> **作者:** Yiwei Li; Yikang Liu; Jiaqi Guo; Lin Zhao; Zheyuan Zhang; Xiao Chen; Boris Mailhe; Ankush Mukherjee; Terrence Chen; Shanhui Sun
>
> **摘要:** Anatomical understanding through deep learning is critical for automatic report generation, intra-operative navigation, and organ localization in medical imaging; however, its progress is constrained by the scarcity of expert-labeled data. A promising remedy is to leverage an annotated reference image to guide the interpretation of an unlabeled target. Although recent vision-language models (VLMs) exhibit non-trivial visual reasoning, their reference-based understanding and fine-grained localization remain limited. We introduce RAU, a framework for reference-based anatomical understanding with VLMs. We first show that a VLM learns to identify anatomical regions through relative spatial reasoning between reference and target images, trained on a moderately sized dataset. We validate this capability through visual question answering (VQA) and bounding box prediction. Next, we demonstrate that the VLM-derived spatial cues can be seamlessly integrated with the fine-grained segmentation capability of SAM2, enabling localization and pixel-level segmentation of small anatomical regions, such as vessel segments. Across two in-distribution and two out-of-distribution datasets, RAU consistently outperforms a SAM2 fine-tuning baseline using the same memory setup, yielding more accurate segmentations and more reliable localization. More importantly, its strong generalization ability makes it scalable to out-of-distribution datasets, a property crucial for medical image applications. To the best of our knowledge, RAU is the first to explore the capability of VLMs for reference-based identification, localization, and segmentation of anatomical structures in medical images. Its promising performance highlights the potential of VLM-driven approaches for anatomical understanding in automated clinical workflows.
>
---
#### [new 019] UniMapGen: A Generative Framework for Large-Scale Map Construction from Multi-modal Data
- **分类: cs.CV**

- **简介: 该论文提出UniMapGen，用于大规模地图构建任务。针对传统方法成本高、效率低及卫星数据缺陷问题，设计了支持多模态输入的生成框架，创新性地采用离散序列表示车道线，并引入状态更新策略以提升地图全局一致性与连续性。**

- **链接: [http://arxiv.org/pdf/2509.22262v1](http://arxiv.org/pdf/2509.22262v1)**

> **作者:** Yujian Yuan; Changjie Wu; Xinyuan Chang; Sijin Wang; Hang Zhang; Shiyi Liang; Shuang Zeng; Mu Xu
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Large-scale map construction is foundational for critical applications such as autonomous driving and navigation systems. Traditional large-scale map construction approaches mainly rely on costly and inefficient special data collection vehicles and labor-intensive annotation processes. While existing satellite-based methods have demonstrated promising potential in enhancing the efficiency and coverage of map construction, they exhibit two major limitations: (1) inherent drawbacks of satellite data (e.g., occlusions, outdatedness) and (2) inefficient vectorization from perception-based methods, resulting in discontinuous and rough roads that require extensive post-processing. This paper presents a novel generative framework, UniMapGen, for large-scale map construction, offering three key innovations: (1) representing lane lines as \textbf{discrete sequence} and establishing an iterative strategy to generate more complete and smooth map vectors than traditional perception-based methods. (2) proposing a flexible architecture that supports \textbf{multi-modal} inputs, enabling dynamic selection among BEV, PV, and text prompt, to overcome the drawbacks of satellite data. (3) developing a \textbf{state update} strategy for global continuity and consistency of the constructed large-scale map. UniMapGen achieves state-of-the-art performance on the OpenSatMap dataset. Furthermore, UniMapGen can infer occluded roads and predict roads missing from dataset annotations. Our code will be released.
>
---
#### [new 020] Skeleton Sparsification and Densification Scale-Spaces
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出一种骨架尺度空间方法，通过稀疏化与稠密化处理解决传统中轴线对噪声敏感的问题。工作包括理论构建和应用验证，用于形状简化、压缩及3D打印刚度增强等任务。**

- **链接: [http://arxiv.org/pdf/2509.21398v1](http://arxiv.org/pdf/2509.21398v1)**

> **作者:** Julia Gierke; Pascal Peter
>
> **摘要:** The Hamilton-Jacobi skeleton, also known as the medial axis, is a powerful shape descriptor that represents binary objects in terms of the centres of maximal inscribed discs. Despite its broad applicability, the medial axis suffers from sensitivity to noise: minor boundary variations can lead to disproportionately large and undesirable expansions of the skeleton. Classical pruning methods mitigate this shortcoming by systematically removing extraneous skeletal branches. This sequential simplification of skeletons resembles the principle of sparsification scale-spaces that embed images into a family of reconstructions from increasingly sparse pixel representations. We combine both worlds by introducing skeletonisation scale-spaces: They leverage sparsification of the medial axis to achieve hierarchical simplification of shapes. Unlike conventional pruning, our framework inherently satisfies key scale-space properties such as hierarchical architecture, controllable simplification, and equivariance to geometric transformations. We provide a rigorous theoretical foundation in both continuous and discrete formulations and extend the concept further with densification. This allows inverse progression from coarse to fine scales and can even reach beyond the original skeleton to produce overcomplete shape representations with relevancy for practical applications. Through proof-of-concept experiments, we demonstrate the effectiveness of our framework for practical tasks including robust skeletonisation, shape compression, and stiffness enhancement for additive manufacturing.
>
---
#### [new 021] In silico Deep Learning Protocols for Label-Free Super-Resolution Microscopy: A Comparative Study of Network Architectures and SNR Dependence
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算显微成像任务，旨在解决低成本实现超分辨显微的问题。研究对比了两种深度学习模型（O-Net和Theta-Net）在不同信噪比下对非荧光显微图像的超分辨性能，发现其性能受信噪比影响显著，具有互补性。**

- **链接: [http://arxiv.org/pdf/2509.21376v1](http://arxiv.org/pdf/2509.21376v1)**

> **作者:** Shiraz S Kaderuppan; Jonathan Mar; Andrew Irvine; Anurag Sharma; Muhammad Ramadan Saifuddin; Wai Leong Eugene Wong; Wai Lok Woo
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** The field of optical microscopy spans across numerous industries and research domains, ranging from education to healthcare, quality inspection and analysis. Nonetheless, a key limitation often cited by optical microscopists refers to the limit of its lateral resolution (typically defined as ~200nm), with potential circumventions involving either costly external modules (e.g. confocal scan heads, etc) and/or specialized techniques [e.g. super-resolution (SR) fluorescent microscopy]. Addressing these challenges in a normal (non-specialist) context thus remains an aspect outside the scope of most microscope users & facilities. This study thus seeks to evaluate an alternative & economical approach to achieving SR optical microscopy, involving non-fluorescent phase-modulated microscopical modalities such as Zernike phase contrast (PCM) and differential interference contrast (DIC) microscopy. Two in silico deep neural network (DNN) architectures which we developed previously (termed O-Net and Theta-Net) are assessed on their abilities to resolve a custom-fabricated test target containing nanoscale features calibrated via atomic force microscopy (AFM). The results of our study demonstrate that although both O-Net and Theta-Net seemingly performed well when super-resolving these images, they were complementary (rather than competing) approaches to be considered for image SR, particularly under different image signal-to-noise ratios (SNRs). High image SNRs favoured the application of O-Net models, while low SNRs inclined preferentially towards Theta-Net models. These findings demonstrate the importance of model architectures (in conjunction with the source image SNR) on model performance and the SR quality of the generated images where DNN models are utilized for non-fluorescent optical nanoscopy, even where the same training dataset & number of epochs are being used.
>
---
#### [new 022] Taming Flow-based I2V Models for Creative Video Editing
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出IF-V2V，一种无需反演的视频编辑方法，旨在解决现有图像到视频（I2V）模型在视频编辑中计算开销大、依赖特定设计的问题。通过向量场修正和结构运动保持初始化，实现高效、高质量的视频编辑。**

- **链接: [http://arxiv.org/pdf/2509.21917v1](http://arxiv.org/pdf/2509.21917v1)**

> **作者:** Xianghao Kong; Hansheng Chen; Yuwei Guo; Lvmin Zhang; Gordon Wetzstein; Maneesh Agrawala; Anyi Rao
>
> **摘要:** Although image editing techniques have advanced significantly, video editing, which aims to manipulate videos according to user intent, remains an emerging challenge. Most existing image-conditioned video editing methods either require inversion with model-specific design or need extensive optimization, limiting their capability of leveraging up-to-date image-to-video (I2V) models to transfer the editing capability of image editing models to the video domain. To this end, we propose IF-V2V, an Inversion-Free method that can adapt off-the-shelf flow-matching-based I2V models for video editing without significant computational overhead. To circumvent inversion, we devise Vector Field Rectification with Sample Deviation to incorporate information from the source video into the denoising process by introducing a deviation term into the denoising vector field. To further ensure consistency with the source video in a model-agnostic way, we introduce Structure-and-Motion-Preserving Initialization to generate motion-aware temporally correlated noise with structural information embedded. We also present a Deviation Caching mechanism to minimize the additional computational cost for denoising vector rectification without significantly impacting editing quality. Evaluations demonstrate that our method achieves superior editing quality and consistency over existing approaches, offering a lightweight plug-and-play solution to realize visual creativity.
>
---
#### [new 023] PSTTS: A Plug-and-Play Token Selector for Efficient Event-based Spatio-temporal Representation Learning
- **分类: cs.CV**

- **简介: 该论文针对事件相机数据的时空表征学习任务，提出PSTTS模块，解决事件帧序列中空间稀疏性和时间冗余导致的计算开销大问题。通过两阶段的冗余token剔除，提升效率并保持精度。**

- **链接: [http://arxiv.org/pdf/2509.22481v1](http://arxiv.org/pdf/2509.22481v1)**

> **作者:** Xiangmo Zhao; Nan Yang; Yang Wang; Zhanwen Liu
>
> **摘要:** Mainstream event-based spatio-temporal representation learning methods typically process event streams by converting them into sequences of event frames, achieving remarkable performance. However, they neglect the high spatial sparsity and inter-frame motion redundancy inherent in event frame sequences, leading to significant computational overhead. Existing token sparsification methods for RGB videos rely on unreliable intermediate token representations and neglect the influence of event noise, making them ineffective for direct application to event data. In this paper, we propose Progressive Spatio-Temporal Token Selection (PSTTS), a Plug-and-Play module for event data without introducing any additional parameters. PSTTS exploits the spatio-temporal distribution characteristics embedded in raw event data to effectively identify and discard spatio-temporal redundant tokens, achieving an optimal trade-off between accuracy and efficiency. Specifically, PSTTS consists of two stages, Spatial Token Purification and Temporal Token Selection. Spatial Token Purification discards noise and non-event regions by assessing the spatio-temporal consistency of events within each event frame to prevent interference with subsequent temporal redundancy evaluation. Temporal Token Selection evaluates the motion pattern similarity between adjacent event frames, precisely identifying and removing redundant temporal information. We apply PSTTS to four representative backbones UniformerV2, VideoSwin, EVMamba, and ExACT on the HARDVS, DailyDVS-200, and SeACT datasets. Experimental results demonstrate that PSTTS achieves significant efficiency improvements. Specifically, PSTTS reduces FLOPs by 29-43.6% and increases FPS by 21.6-41.3% on the DailyDVS-200 dataset, while maintaining task accuracy. Our code will be available.
>
---
#### [new 024] X-Streamer: Unified Human World Modeling with Audiovisual Interaction
- **分类: cs.CV**

- **简介: 该论文提出X-Streamer，一种统一的人机交互建模框架，用于构建能进行文本、语音和视频无限交互的数字人。其核心是Thinker-Actor双变压器架构，实现从静态肖像到实时音视频交互的转换，解决多模态对齐与长期稳定性问题。**

- **链接: [http://arxiv.org/pdf/2509.21574v1](http://arxiv.org/pdf/2509.21574v1)**

> **作者:** You Xie; Tianpei Gu; Zenan Li; Chenxu Zhang; Guoxian Song; Xiaochen Zhao; Chao Liang; Jianwen Jiang; Hongyi Xu; Linjie Luo
>
> **备注:** Project Page at https://byteaigc.github.io/X-Streamer
>
> **摘要:** We introduce X-Streamer, an end-to-end multimodal human world modeling framework for building digital human agents capable of infinite interactions across text, speech, and video within a single unified architecture. Starting from a single portrait, X-Streamer enables real-time, open-ended video calls driven by streaming multimodal inputs. At its core is a Thinker-Actor dual-transformer architecture that unifies multimodal understanding and generation, turning a static portrait into persistent and intelligent audiovisual interactions. The Thinker module perceives and reasons over streaming user inputs, while its hidden states are translated by the Actor into synchronized multimodal streams in real time. Concretely, the Thinker leverages a pretrained large language-speech model, while the Actor employs a chunk-wise autoregressive diffusion model that cross-attends to the Thinker's hidden states to produce time-aligned multimodal responses with interleaved discrete text and audio tokens and continuous video latents. To ensure long-horizon stability, we design inter- and intra-chunk attentions with time-aligned multimodal positional embeddings for fine-grained cross-modality alignment and context retention, further reinforced by chunk-wise diffusion forcing and global identity referencing. X-Streamer runs in real time on two A100 GPUs, sustaining hours-long consistent video chat experiences from arbitrary portraits and paving the way toward unified world modeling of interactive digital humans.
>
---
#### [new 025] GS-2M: Gaussian Splatting for Joint Mesh Reconstruction and Material Decomposition
- **分类: cs.CV**

- **简介: 该论文提出GS-2M方法，基于3D高斯泼溅技术，联合解决网格重建与材质分解问题。针对反光表面重建困难和依赖外部先验的问题，通过优化深度与法线属性，并引入粗糙度监督策略，提升了重建精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.22276v1](http://arxiv.org/pdf/2509.22276v1)**

> **作者:** Dinh Minh Nguyen; Malte Avenhaus; Thomas Lindemeier
>
> **备注:** 13 pages, 10 figures
>
> **摘要:** We propose a unified solution for mesh reconstruction and material decomposition from multi-view images based on 3D Gaussian Splatting, referred to as GS-2M. Previous works handle these tasks separately and struggle to reconstruct highly reflective surfaces, often relying on priors from external models to enhance the decomposition results. Conversely, our method addresses these two problems by jointly optimizing attributes relevant to the quality of rendered depth and normals, maintaining geometric details while being resilient to reflective surfaces. Although contemporary works effectively solve these tasks together, they often employ sophisticated neural components to learn scene properties, which hinders their performance at scale. To further eliminate these neural components, we propose a novel roughness supervision strategy based on multi-view photometric variation. When combined with a carefully designed loss and optimization process, our unified framework produces reconstruction results comparable to state-of-the-art methods, delivering triangle meshes and their associated material components for downstream tasks. We validate the effectiveness of our approach with widely used datasets from previous works and qualitative comparisons with state-of-the-art surface reconstruction methods.
>
---
#### [new 026] REFINE-CONTROL: A Semi-supervised Distillation Method For Conditional Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对条件图像生成任务，旨在解决模型资源消耗大和标注数据稀缺的问题。提出Refine-Control，一种半监督蒸馏框架，通过三级知识融合损失提升学生模型性能，并利用未标注数据增强泛化能力，降低计算成本。**

- **链接: [http://arxiv.org/pdf/2509.22139v1](http://arxiv.org/pdf/2509.22139v1)**

> **作者:** Yicheng Jiang; Jin Yuan; Hua Yuan; Yao Zhang; Yong Rui
>
> **备注:** 5 pages,17 figures
>
> **摘要:** Conditional image generation models have achieved remarkable results by leveraging text-based control to generate customized images. However, the high resource demands of these models and the scarcity of well-annotated data have hindered their deployment on edge devices, leading to enormous costs and privacy concerns, especially when user data is sent to a third party. To overcome these challenges, we propose Refine-Control, a semi-supervised distillation framework. Specifically, we improve the performance of the student model by introducing a tri-level knowledge fusion loss to transfer different levels of knowledge. To enhance generalization and alleviate dataset scarcity, we introduce a semi-supervised distillation method utilizing both labeled and unlabeled data. Our experiments reveal that Refine-Control achieves significant reductions in computational cost and latency, while maintaining high-fidelity generation capabilities and controllability, as quantified by comparative metrics.
>
---
#### [new 027] UrbanFeel: A Comprehensive Benchmark for Temporal and Perceptual Understanding of City Scenes through Human Perspective
- **分类: cs.CV**

- **简介: 该论文提出UrbanFeel，一个用于评估多模态大语言模型在城市场景理解与感知能力的综合基准。针对现有基准缺乏对时间演化和主观感知的研究，构建了14.3K视觉问题，并通过多维度评估发现模型在时序推理上表现较弱，而在主观感知上接近或超越人类水平。**

- **链接: [http://arxiv.org/pdf/2509.22228v1](http://arxiv.org/pdf/2509.22228v1)**

> **作者:** Jun He; Yi Lin; Zilong Huang; Jiacong Yin; Junyan Ye; Yuchuan Zhou; Weijia Li; Xiang Zhang
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Urban development impacts over half of the global population, making human-centered understanding of its structural and perceptual changes essential for sustainable development. While Multimodal Large Language Models (MLLMs) have shown remarkable capabilities across various domains, existing benchmarks that explore their performance in urban environments remain limited, lacking systematic exploration of temporal evolution and subjective perception of urban environment that aligns with human perception. To address these limitations, we propose UrbanFeel, a comprehensive benchmark designed to evaluate the performance of MLLMs in urban development understanding and subjective environmental perception. UrbanFeel comprises 14.3K carefully constructed visual questions spanning three cognitively progressive dimensions: Static Scene Perception, Temporal Change Understanding, and Subjective Environmental Perception. We collect multi-temporal single-view and panoramic street-view images from 11 representative cities worldwide, and generate high-quality question-answer pairs through a hybrid pipeline of spatial clustering, rule-based generation, model-assisted prompting, and manual annotation. Through extensive evaluation of 20 state-of-the-art MLLMs, we observe that Gemini-2.5 Pro achieves the best overall performance, with its accuracy approaching human expert levels and narrowing the average gap to just 1.5\%. Most models perform well on tasks grounded in scene understanding. In particular, some models even surpass human annotators in pixel-level change detection. However, performance drops notably in tasks requiring temporal reasoning over urban development. Additionally, in the subjective perception dimension, several models reach human-level or even higher consistency in evaluating dimension such as beautiful and safety.
>
---
#### [new 028] CubistMerge: Spatial-Preserving Token Merging For Diverse ViT Backbones
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出CubistMerge，一种用于视觉Transformer（ViT）的令牌合并方法，旨在在减少令牌数量的同时保持空间结构。针对现有方法破坏空间架构的问题，设计了2D缩减策略、空间感知合并算法和新特征表示方式，在多种任务中实现高效且高性能的模型加速。**

- **链接: [http://arxiv.org/pdf/2509.21764v1](http://arxiv.org/pdf/2509.21764v1)**

> **作者:** Wenyi Gong; Mieszko Lis
>
> **摘要:** Many modern ViT backbones adopt spatial architectural designs, such as window attention, decomposed relative positional embeddings in SAM, and RoPE in DINOv3. Such architectures impose new challenges on token reduction, as the vast majority of existing methods fail to preserve the spatial structure these architectures depend on. In this paper, we introduce a simple yet effective token merging method that maintains spatial integrity, enabling seamless compatibility with spatial architectures. We reconcile two seemingly conflicting requirements: (i)exploiting the uneven information distribution across the spatial layout while (ii)preserving the spatial structure post-merging. Our approach employs (i)a 2D reduction strategy to enforce structured token layouts, (ii)a spatial-aware merging algorithm that maintains relative token positions, and (iii)a novel max-magnitude-per-dimension token representation that preserves salient features. Our method demonstrates strong performance both off-the-shelf and with fine-tuning, achieving state-of-the-art results on spatial and non-spatial architectures across various vision tasks. Specifically, we achieve 1.25x speedup on SAM-H with only 0.7% mIOU drop evaluated on COCO off-the-shelf, and 1.15x speedup on DeiT-B with no top-1 accuracy drop on ImageNet within just one epoch of fine-tuning.
>
---
#### [new 029] Polysemous Language Gaussian Splatting via Matching-based Mask Lifting
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MUSplat，用于开放词汇的3D高斯点云语义理解。针对现有方法依赖训练、语义单一和视图不一致的问题，设计无需训练的框架，通过2D分割与VLM结合，实现多粒度语义提升与鲁棒表示。**

- **链接: [http://arxiv.org/pdf/2509.22225v1](http://arxiv.org/pdf/2509.22225v1)**

> **作者:** Jiayu Ding; Xinpeng Liu; Zhiyi Pan; Shiqiang Long; Ge Li
>
> **摘要:** Lifting 2D open-vocabulary understanding into 3D Gaussian Splatting (3DGS) scenes is a critical challenge. However, mainstream methods suffer from three key flaws: (i) their reliance on costly per-scene retraining prevents plug-and-play application; (ii) their restrictive monosemous design fails to represent complex, multi-concept semantics; and (iii) their vulnerability to cross-view semantic inconsistencies corrupts the final semantic representation. To overcome these limitations, we introduce MUSplat, a training-free framework that abandons feature optimization entirely. Leveraging a pre-trained 2D segmentation model, our pipeline generates and lifts multi-granularity 2D masks into 3D, where we estimate a foreground probability for each Gaussian point to form initial object groups. We then optimize the ambiguous boundaries of these initial groups using semantic entropy and geometric opacity. Subsequently, by interpreting the object's appearance across its most representative viewpoints, a Vision-Language Model (VLM) distills robust textual features that reconciles visual inconsistencies, enabling open-vocabulary querying via semantic matching. By eliminating the costly per-scene training process, MUSplat reduces scene adaptation time from hours to mere minutes. On benchmark tasks for open-vocabulary 3D object selection and semantic segmentation, MUSplat outperforms established training-based frameworks while simultaneously addressing their monosemous limitations.
>
---
#### [new 030] Learning GUI Grounding with Spatial Reasoning from Visual Feedback
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对GUI定位任务中坐标预测不准确的问题，提出将GUI定位重构为交互式搜索任务。通过引入光标移动与视觉反馈机制，结合强化学习训练模型GUI-Cursor，提升了高分辨率复杂界面下的定位精度，在多个数据集上取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.21552v1](http://arxiv.org/pdf/2509.21552v1)**

> **作者:** Yu Zhao; Wei-Ning Chen; Huseyin Atahan Inan; Samuel Kessler; Lu Wang; Lukas Wutschitz; Fangkai Yang; Chaoyun Zhang; Pasquale Minervini; Saravan Rajmohan; Robert Sim
>
> **摘要:** Graphical User Interface (GUI) grounding is commonly framed as a coordinate prediction task -- given a natural language instruction, generate on-screen coordinates for actions such as clicks and keystrokes. However, recent Vision Language Models (VLMs) often fail to predict accurate numeric coordinates when processing high-resolution GUI images with complex layouts. To address this issue, we reframe GUI grounding as an \emph{interactive search task}, where the VLM generates actions to move a cursor in the GUI to locate UI elements. At each step, the model determines the target object, evaluates the spatial relations between the cursor and the target, and moves the cursor closer to the target conditioned on the movement history. In this interactive process, the rendered cursor provides visual feedback to help the model align its predictions with the corresponding on-screen locations. We train our GUI grounding model, GUI-Cursor, using multi-step online reinforcement learning with a dense trajectory-based reward function. Our experimental results show that GUI-Cursor, based on Qwen2.5-VL-7B, improves the GUI grounding accuracy and achieves state-of-the-art results on ScreenSpot-v2 ($88.8\% \rightarrow 93.9\%$) and ScreenSpot-Pro ($26.8\% \rightarrow 56.5\%$). Moreover, we observe that GUI-Cursor learns to solve the problem within two steps for 95\% of instances and can adaptively conduct more steps on more difficult examples.
>
---
#### [new 031] A Mutual Learning Method for Salient Object Detection with intertwined Multi-Supervision--Revised
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对显著性目标检测任务，旨在解决预测不完整和边界不准确的问题。提出一种结合显著性检测、前景轮廓检测和边缘检测的多监督互学习方法，并设计了互学习模块（MLM），在多个数据集上取得了SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.21363v1](http://arxiv.org/pdf/2509.21363v1)**

> **作者:** Runmin Wu; Mengyang Feng; Wenlong Guan; Dong Wang; Huchuan Lu; Errui Ding
>
> **备注:** 11 pages
>
> **摘要:** Though deep learning techniques have made great progress in salient object detection recently, the predicted saliency maps still suffer from incomplete predictions due to the internal complexity of objects and inaccurate boundaries caused by strides in convolution and pooling operations. To alleviate these issues, we propose to train saliency detection networks by exploiting the supervision from not only salient object detection, but also foreground contour detection and edge detection. First, we leverage salient object detection and foreground contour detection tasks in an intertwined manner to generate saliency maps with uniform highlight. Second, the foreground contour and edge detection tasks guide each other simultaneously, thereby leading to precise foreground contour prediction and reducing the local noises for edge prediction. In addition, we develop a novel mutual learning module (MLM) which serves as the building block of our method. Each MLM consists of multiple network branches trained in a mutual learning manner, which improves the performance by a large margin. Extensive experiments on seven challenging datasets demonstrate that the proposed method has delivered state-of-the-art results in both salient object detection and edge detection.
>
---
#### [new 032] Closing the Safety Gap: Surgical Concept Erasure in Visual Autoregressive Models
- **分类: cs.CV**

- **简介: 该论文针对视觉自回归（VAR）模型在文本到图像生成中的安全性问题，提出VARE和S-VARE方法，通过精准消除有害概念，在保证生成质量的同时提升模型安全性。**

- **链接: [http://arxiv.org/pdf/2509.22400v1](http://arxiv.org/pdf/2509.22400v1)**

> **作者:** Xinhao Zhong; Yimin Zhou; Zhiqi Zhang; Junhao Li; Yi Sun; Bin Chen; Shu-Tao Xia; Ke Xu
>
> **摘要:** The rapid progress of visual autoregressive (VAR) models has brought new opportunities for text-to-image generation, but also heightened safety concerns. Existing concept erasure techniques, primarily designed for diffusion models, fail to generalize to VARs due to their next-scale token prediction paradigm. In this paper, we first propose a novel VAR Erasure framework VARE that enables stable concept erasure in VAR models by leveraging auxiliary visual tokens to reduce fine-tuning intensity. Building upon this, we introduce S-VARE, a novel and effective concept erasure method designed for VAR, which incorporates a filtered cross entropy loss to precisely identify and minimally adjust unsafe visual tokens, along with a preservation loss to maintain semantic fidelity, addressing the issues such as language drift and reduced diversity introduce by na\"ive fine-tuning. Extensive experiments demonstrate that our approach achieves surgical concept erasure while preserving generation quality, thereby closing the safety gap in autoregressive text-to-image generation by earlier methods.
>
---
#### [new 033] MultiMat: Multimodal Program Synthesis for Procedural Materials using Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文提出MultiMat，一种基于多模态大模型的程序合成框架，用于生成过程材质节点图。针对现有方法仅用文本表示、忽视视觉特性的问题，MultiMat结合视觉与文本信息，提升生成效率与质量，实现更自然的过程材质建模。**

- **链接: [http://arxiv.org/pdf/2509.22151v1](http://arxiv.org/pdf/2509.22151v1)**

> **作者:** Jonas Belouadi; Tamy Boubekeur; Adrien Kaiser
>
> **备注:** Submitted to ICLR 2026
>
> **摘要:** Material node graphs are programs that generate the 2D channels of procedural materials, including geometry such as roughness and displacement maps, and reflectance such as albedo and conductivity maps. They are essential in computer graphics for representing the appearance of virtual 3D objects parametrically and at arbitrary resolution. In particular, their directed acyclic graph structures and intermediate states provide an intuitive understanding and workflow for interactive appearance modeling. Creating such graphs is a challenging task and typically requires professional training. While recent neural program synthesis approaches attempt to simplify this process, they solely represent graphs as textual programs, failing to capture the inherently visual-spatial nature of node graphs that makes them accessible to humans. To address this gap, we present MultiMat, a multimodal program synthesis framework that leverages large multimodal models to process both visual and textual graph representations for improved generation of procedural material graphs. We train our models on a new dataset of production-quality procedural materials and combine them with a constrained tree search inference algorithm that ensures syntactic validity while efficiently navigating the program space. Our experimental results show that our multimodal program synthesis method is more efficient in both unconditional and conditional graph synthesis with higher visual quality and fidelity than text-only baselines, establishing new state-of-the-art performance.
>
---
#### [new 034] Random Direct Preference Optimization for Radiography Report Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对放射科报告生成（RRG）任务，旨在提升其临床实用性。提出一种无需奖励模型或人工标注的随机直接偏好优化（Random DPO）框架，通过对比采样增强模型性能，实验表明可提升5%的临床指标。**

- **链接: [http://arxiv.org/pdf/2509.21351v1](http://arxiv.org/pdf/2509.21351v1)**

> **作者:** Valentin Samokhin; Boris Shirokikh; Mikhail Goncharov; Dmitriy Umerenkov; Maksim Bobrin; Ivan Oseledets; Dmitry Dylov; Mikhail Belyaev
>
> **摘要:** Radiography Report Generation (RRG) has gained significant attention in medical image analysis as a promising tool for alleviating the growing workload of radiologists. However, despite numerous advancements, existing methods have yet to achieve the quality required for deployment in real-world clinical settings. Meanwhile, large Visual Language Models (VLMs) have demonstrated remarkable progress in the general domain by adopting training strategies originally designed for Large Language Models (LLMs), such as alignment techniques. In this paper, we introduce a model-agnostic framework to enhance RRG accuracy using Direct Preference Optimization (DPO). Our approach leverages random contrastive sampling to construct training pairs, eliminating the need for reward models or human preference annotations. Experiments on supplementing three state-of-the-art models with our Random DPO show that our method improves clinical performance metrics by up to 5%, without requiring any additional training data.
>
---
#### [new 035] No Alignment Needed for Generation: Learning Linearly Separable Representations in Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对扩散模型中的表示学习问题，提出无需外部编码器对齐的LSEP正则化方法，直接提升中间层表示的线性可分性。实验表明，该方法在生成任务中提高了训练效率和生成质量，适用于SiT等架构，在ImageNet数据集上取得了优异结果。**

- **链接: [http://arxiv.org/pdf/2509.21565v1](http://arxiv.org/pdf/2509.21565v1)**

> **作者:** Junno Yun; Yaşar Utku Alçalar; Mehmet Akçakaya
>
> **摘要:** Efficient training strategies for large-scale diffusion models have recently emphasized the importance of improving discriminative feature representations in these models. A central line of work in this direction is representation alignment with features obtained from powerful external encoders, which improves the representation quality as assessed through linear probing. Alignment-based approaches show promise but depend on large pretrained encoders, which are computationally expensive to obtain. In this work, we propose an alternative regularization for training, based on promoting the Linear SEParability (LSEP) of intermediate layer representations. LSEP eliminates the need for an auxiliary encoder and representation alignment, while incorporating linear probing directly into the network's learning dynamics rather than treating it as a simple post-hoc evaluation tool. Our results demonstrate substantial improvements in both training efficiency and generation quality on flow-based transformer architectures such as SiTs, achieving an FID of 1.46 on $256 \times 256$ ImageNet dataset.
>
---
#### [new 036] Enhancing Vehicle Detection under Adverse Weather Conditions with Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文针对无人机在北欧恶劣天气下车辆检测任务，解决因积雪导致的可见度低和领域偏移问题。提出一种sideload-CL-adaptation框架，利用对比学习预训练轻量模型，并将其适配到YOLO11n中，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.21916v1](http://arxiv.org/pdf/2509.21916v1)**

> **作者:** Boying Li; Chang Liu; Petter Kyösti; Mattias Öhman; Devashish Singha Roy; Sofia Plazzi; Hamam Mokayed; Olle Hagner
>
> **摘要:** Aside from common challenges in remote sensing like small, sparse targets and computation cost limitations, detecting vehicles from UAV images in the Nordic regions faces strong visibility challenges and domain shifts caused by diverse levels of snow coverage. Although annotated data are expensive, unannotated data is cheaper to obtain by simply flying the drones. In this work, we proposed a sideload-CL-adaptation framework that enables the use of unannotated data to improve vehicle detection using lightweight models. Specifically, we propose to train a CNN-based representation extractor through contrastive learning on the unannotated data in the pretraining stage, and then sideload it to a frozen YOLO11n backbone in the fine-tuning stage. To find a robust sideload-CL-adaptation, we conducted extensive experiments to compare various fusion methods and granularity. Our proposed sideload-CL-adaptation model improves the detection performance by 3.8% to 9.5% in terms of mAP50 on the NVD dataset.
>
---
#### [new 037] A Tale of Two Experts: Cooperative Learning for Source-Free Unsupervised Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文研究源无关无监督域适应（SFUDA）任务，旨在无需源域数据的情况下将模型迁移到目标域。针对现有方法的不足，提出EXCL方法，通过双专家框架与检索增强交互优化，有效利用源模型和视觉-语言模型的协同学习，提升目标域适应性能。**

- **链接: [http://arxiv.org/pdf/2509.22229v1](http://arxiv.org/pdf/2509.22229v1)**

> **作者:** Jiaping Yu; Muli Yang; Jiapeng Ji; Jiexi Yan; Cheng Deng
>
> **摘要:** Source-Free Unsupervised Domain Adaptation (SFUDA) addresses the realistic challenge of adapting a source-trained model to a target domain without access to the source data, driven by concerns over privacy and cost. Existing SFUDA methods either exploit only the source model's predictions or fine-tune large multimodal models, yet both neglect complementary insights and the latent structure of target data. In this paper, we propose the Experts Cooperative Learning (EXCL). EXCL contains the Dual Experts framework and Retrieval-Augmentation-Interaction optimization pipeline. The Dual Experts framework places a frozen source-domain model (augmented with Conv-Adapter) and a pretrained vision-language model (with a trainable text prompt) on equal footing to mine consensus knowledge from unlabeled target samples. To effectively train these plug-in modules under purely unsupervised conditions, we introduce Retrieval-Augmented-Interaction(RAIN), a three-stage pipeline that (1) collaboratively retrieves pseudo-source and complex target samples, (2) separately fine-tunes each expert on its respective sample set, and (3) enforces learning object consistency via a shared learning result. Extensive experiments on four benchmark datasets demonstrate that our approach matches state-of-the-art performance.
>
---
#### [new 038] Downscaling climate projections to 1 km with single-image super resolution
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于气候数据降尺度任务，旨在解决气候预测空间分辨率低的问题。作者利用单图像超分辨率模型，将12.5 km分辨率的气候预测提升至1 km，并通过气候指标评估方法验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.21399v1](http://arxiv.org/pdf/2509.21399v1)**

> **作者:** Petr Košťál; Pavel Kordík; Ondřej Podsztavek
>
> **摘要:** High-resolution climate projections are essential for local decision-making. However, available climate projections have low spatial resolution (e.g. 12.5 km), which limits their usability. We address this limitation by leveraging single-image super-resolution models to statistically downscale climate projections to 1-km resolution. Since high-resolution climate projections are unavailable for training, we train models on a high-resolution observational gridded data set and apply them to low-resolution climate projections. We propose a climate indicator-based assessment using observed climate indices computed at weather station locations to evaluate the downscaled climate projections without ground-truth high-resolution climate projections. Experiments on daily mean temperature demonstrate that single-image super-resolution models can downscale climate projections without increasing the error of climate indicators compared to low-resolution climate projections.
>
---
#### [new 039] MIRG-RL: Multi-Image Reasoning and Grounding with Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文提出MIRG-RL，用于多图像推理与定位任务。针对现有模型缺乏跨图推理能力和奖励建模不足的问题，设计了两阶段训练框架和双奖励函数的图像感知强化学习策略，并构建了轻量级数据集，实验在跨图推理任务上达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.21788v1](http://arxiv.org/pdf/2509.21788v1)**

> **作者:** Lihao Zheng; Jiawei Chen; Xintian Shen; Hao Ma; Tao Wei
>
> **摘要:** Multi-image reasoning and grounding require understanding complex cross-image relationships at both object levels and image levels. Current Large Visual Language Models (LVLMs) face two critical challenges: the lack of cross-image reasoning capabilities and insufficient cross-image reference reward modeling. To address these issues, we propose a unified framework - Multi-Image Reasoning and Grounding with Reinforcement Learning (MIRG-RL). Specifically, our two-stage training paradigm combines supervised fine-tuning with annotated trajectories and image-aware reinforcement learning optimization, progressively developing multi-image reasoning capabilities. Furthermore, we innovatively propose a method for constructing the trajectory data, which integrates object-level and image-level annotation information, and use this method to generate a lightweight reasoning-enhanced dataset. To effectively resolve cross-image ambiguities, we design an image-aware RL policy with dual reward functions for objects and images. Experiments demonstrate that MIRG-RL achieves state-of-the-art (SOTA) performance in multi-image grounding benchmarks, attaining 64.82% on cross-image reasoning tasks - exceeding the previous best method by 1%. The code and dataset have been released at https://github.com/ZEUS2035/MIRG-RL.
>
---
#### [new 040] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出JanusVLN，针对视觉-语言导航任务，解决现有方法因显式语义记忆导致的空间信息丢失和计算冗余问题。通过构建双隐式神经记忆，分别建模空间几何与视觉语义信息，实现高效增量更新，提升导航成功率。**

- **链接: [http://arxiv.org/pdf/2509.22548v1](http://arxiv.org/pdf/2509.22548v1)**

> **作者:** Shuang Zeng; Dekang Qi; Xinyuan Chang; Feng Xiong; Shichao Xie; Xiaolong Wu; Shiyi Liang; Mu Xu; Xing Wei
>
> **备注:** Project page: https://miv-xjtu.github.io/JanusVLN.github.io/
>
> **摘要:** Vision-and-Language Navigation requires an embodied agent to navigate through unseen environments, guided by natural language instructions and a continuous video stream. Recent advances in VLN have been driven by the powerful semantic understanding of Multimodal Large Language Models. However, these methods typically rely on explicit semantic memory, such as building textual cognitive maps or storing historical visual frames. This type of method suffers from spatial information loss, computational redundancy, and memory bloat, which impede efficient navigation. Inspired by the implicit scene representation in human navigation, analogous to the left brain's semantic understanding and the right brain's spatial cognition, we propose JanusVLN, a novel VLN framework featuring a dual implicit neural memory that models spatial-geometric and visual-semantic memory as separate, compact, and fixed-size neural representations. This framework first extends the MLLM to incorporate 3D prior knowledge from the spatial-geometric encoder, thereby enhancing the spatial reasoning capabilities of models based solely on RGB input. Then, the historical key-value caches from the spatial-geometric and visual-semantic encoders are constructed into a dual implicit memory. By retaining only the KVs of tokens in the initial and sliding window, redundant computation is avoided, enabling efficient incremental updates. Extensive experiments demonstrate that JanusVLN outperforms over 20 recent methods to achieve SOTA performance. For example, the success rate improves by 10.5-35.5 compared to methods using multiple data types as input and by 3.6-10.8 compared to methods using more RGB training data. This indicates that the proposed dual implicit neural memory, as a novel paradigm, explores promising new directions for future VLN research. Ours project page: https://miv-xjtu.github.io/JanusVLN.github.io/.
>
---
#### [new 041] FailureAtlas:Mapping the Failure Landscape of T2I Models via Active Exploration
- **分类: cs.CV**

- **简介: 该论文提出FailureAtlas，一种通过主动探索映射文本到图像模型失败模式的框架。针对静态基准难以全面诊断模型系统性失败的问题，该工作设计了一种可扩展的方法，自动发现大量未知错误场景，并揭示其与训练数据稀缺性的关联。**

- **链接: [http://arxiv.org/pdf/2509.21995v1](http://arxiv.org/pdf/2509.21995v1)**

> **作者:** Muxi Chen; Zhaohua Zhang; Chenchen Zhao; Mingyang Chen; Wenyu Jiang; Tianwen Jiang; Jianhuan Zhuo; Yu Tang; Qiuyong Xiao; Jihong Zhang; Qiang Xu
>
> **摘要:** Static benchmarks have provided a valuable foundation for comparing Text-to-Image (T2I) models. However, their passive design offers limited diagnostic power, struggling to uncover the full landscape of systematic failures or isolate their root causes. We argue for a complementary paradigm: active exploration. We introduce FailureAtlas, the first framework designed to autonomously explore and map the vast failure landscape of T2I models at scale. FailureAtlas frames error discovery as a structured search for minimal, failure-inducing concepts. While it is a computationally explosive problem, we make it tractable with novel acceleration techniques. When applied to Stable Diffusion models, our method uncovers hundreds of thousands of previously unknown error slices (over 247,000 in SD1.5 alone) and provides the first large-scale evidence linking these failures to data scarcity in the training set. By providing a principled and scalable engine for deep model auditing, FailureAtlas establishes a new, diagnostic-first methodology to guide the development of more robust generative AI. The code is available at https://github.com/cure-lab/FailureAtlas
>
---
#### [new 042] Syncphony: Synchronized Audio-to-Video Generation with Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文提出Syncphony，用于音频到视频生成任务，旨在解决现有方法在时间同步上的不足。通过引入Motion-aware Loss和Audio Sync Guidance，提升了视频与音频的同步精度和视觉质量，并提出了CycleSync评估指标。**

- **链接: [http://arxiv.org/pdf/2509.21893v1](http://arxiv.org/pdf/2509.21893v1)**

> **作者:** Jibin Song; Mingi Kwon; Jaeseok Jeong; Youngjung Uh
>
> **备注:** Project page: https://jibin86.github.io/syncphony_project_page
>
> **摘要:** Text-to-video and image-to-video generation have made rapid progress in visual quality, but they remain limited in controlling the precise timing of motion. In contrast, audio provides temporal cues aligned with video motion, making it a promising condition for temporally controlled video generation. However, existing audio-to-video (A2V) models struggle with fine-grained synchronization due to indirect conditioning mechanisms or limited temporal modeling capacity. We present Syncphony, which generates 380x640 resolution, 24fps videos synchronized with diverse audio inputs. Our approach builds upon a pre-trained video backbone and incorporates two key components to improve synchronization: (1) Motion-aware Loss, which emphasizes learning at high-motion regions; (2) Audio Sync Guidance, which guides the full model using a visually aligned off-sync model without audio layers to better exploit audio cues at inference while maintaining visual quality. To evaluate synchronization, we propose CycleSync, a video-to-audio-based metric that measures the amount of motion cues in the generated video to reconstruct the original audio. Experiments on AVSync15 and The Greatest Hits datasets demonstrate that Syncphony outperforms existing methods in both synchronization accuracy and visual quality. Project page is available at: https://jibin86.github.io/syncphony_project_page
>
---
#### [new 043] DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints
- **分类: cs.CV**

- **简介: 该论文提出DualFocus，用于深度从聚焦（DFF）任务，旨在解决复杂场景中因纹理或深度突变导致的聚焦线索模糊问题。通过引入空间-焦距双变分约束，提升深度估计的鲁棒性与精度。**

- **链接: [http://arxiv.org/pdf/2509.21992v1](http://arxiv.org/pdf/2509.21992v1)**

> **作者:** Sungmin Woo; Sangyoun Lee
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Depth-from-Focus (DFF) enables precise depth estimation by analyzing focus cues across a stack of images captured at varying focal lengths. While recent learning-based approaches have advanced this field, they often struggle in complex scenes with fine textures or abrupt depth changes, where focus cues may become ambiguous or misleading. We present DualFocus, a novel DFF framework that leverages the focal stack's unique gradient patterns induced by focus variation, jointly modeling focus changes over spatial and focal dimensions. Our approach introduces a variational formulation with dual constraints tailored to DFF: spatial constraints exploit gradient pattern changes across focus levels to distinguish true depth edges from texture artifacts, while focal constraints enforce unimodal, monotonic focus probabilities aligned with physical focus behavior. These inductive biases improve robustness and accuracy in challenging regions. Comprehensive experiments on four public datasets demonstrate that DualFocus consistently outperforms state-of-the-art methods in both depth accuracy and perceptual quality.
>
---
#### [new 044] Multimodal Prompt Decoupling Attack on the Safety Filters in Text-to-Image Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究文本到图像模型的安全性问题，针对现有文本攻击方法易被过滤的局限，提出多模态提示解耦攻击（MPDA）。通过分离并重构有害语义，结合图像和语言模型生成NSFW内容，绕过安全过滤器。**

- **链接: [http://arxiv.org/pdf/2509.21360v1](http://arxiv.org/pdf/2509.21360v1)**

> **作者:** Xingkai Peng; Jun Jiang; Meng Tong; Shuai Li; Weiming Zhang; Nenghai Yu; Kejiang Chen
>
> **摘要:** Text-to-image (T2I) models have been widely applied in generating high-fidelity images across various domains. However, these models may also be abused to produce Not-Safe-for-Work (NSFW) content via jailbreak attacks. Existing jailbreak methods primarily manipulate the textual prompt, leaving potential vulnerabilities in image-based inputs largely unexplored. Moreover, text-based methods face challenges in bypassing the model's safety filters. In response to these limitations, we propose the Multimodal Prompt Decoupling Attack (MPDA), which utilizes image modality to separate the harmful semantic components of the original unsafe prompt. MPDA follows three core steps: firstly, a large language model (LLM) decouples unsafe prompts into pseudo-safe prompts and harmful prompts. The former are seemingly harmless sub-prompts that can bypass filters, while the latter are sub-prompts with unsafe semantics that trigger filters. Subsequently, the LLM rewrites the harmful prompts into natural adversarial prompts to bypass safety filters, which guide the T2I model to modify the base image into an NSFW output. Finally, to ensure semantic consistency between the generated NSFW images and the original unsafe prompts, the visual language model generates image captions, providing a new pathway to guide the LLM in iterative rewriting and refining the generated content.
>
---
#### [new 045] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation
- **分类: cs.CV**

- **简介: 该论文提出EAGLE，一种用于解释多模态大语言模型（MLLMs）自回归生成过程的轻量级框架。旨在解决生成token对视觉模态依赖性不明确的问题，通过统一充分性和必要性指标，实现高效、可靠的可视化归因与模态分析，提升模型可解释性。**

- **链接: [http://arxiv.org/pdf/2509.22496v1](http://arxiv.org/pdf/2509.22496v1)**

> **作者:** Ruoyu Chen; Xiaoqing Guo; Kangwei Liu; Siyuan Liang; Shiming Liu; Qunli Zhang; Hua Zhang; Xiaochun Cao
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in aligning visual inputs with natural language outputs. Yet, the extent to which generated tokens depend on visual modalities remains poorly understood, limiting interpretability and reliability. In this work, we present EAGLE, a lightweight black-box framework for explaining autoregressive token generation in MLLMs. EAGLE attributes any selected tokens to compact perceptual regions while quantifying the relative influence of language priors and perceptual evidence. The framework introduces an objective function that unifies sufficiency (insight score) and indispensability (necessity score), optimized via greedy search over sparsified image regions for faithful and efficient attribution. Beyond spatial attribution, EAGLE performs modality-aware analysis that disentangles what tokens rely on, providing fine-grained interpretability of model decisions. Extensive experiments across open-source MLLMs show that EAGLE consistently outperforms existing methods in faithfulness, localization, and hallucination diagnosis, while requiring substantially less GPU memory. These results highlight its effectiveness and practicality for advancing the interpretability of MLLMs. The code is available at https://github.com/RuoyuChen10/EAGLE.
>
---
#### [new 046] Customizing Visual Emotion Evaluation for MLLMs: An Open-vocabulary, Multifaceted, and Scalable Approach
- **分类: cs.CV**

- **简介: 该论文针对多模态大语言模型（MLLMs）在视觉情感理解任务中的评估问题，提出了一种开放词汇、多维度且可扩展的情感判断方法。通过构建自动化的情感陈述判断任务，解决了现有评估方法的局限性，并系统评估了主流MLLMs的情感理解能力。**

- **链接: [http://arxiv.org/pdf/2509.21950v1](http://arxiv.org/pdf/2509.21950v1)**

> **作者:** Daiqing Wu; Dongbao Yang; Sicheng Zhao; Can Ma; Yu Zhou
>
> **摘要:** Recently, Multimodal Large Language Models (MLLMs) have achieved exceptional performance across diverse tasks, continually surpassing previous expectations regarding their capabilities. Nevertheless, their proficiency in perceiving emotions from images remains debated, with studies yielding divergent results in zero-shot scenarios. We argue that this inconsistency stems partly from constraints in existing evaluation methods, including the oversight of plausible responses, limited emotional taxonomies, neglect of contextual factors, and labor-intensive annotations. To facilitate customized visual emotion evaluation for MLLMs, we propose an Emotion Statement Judgment task that overcomes these constraints. Complementing this task, we devise an automated pipeline that efficiently constructs emotion-centric statements with minimal human effort. Through systematically evaluating prevailing MLLMs, our study showcases their stronger performance in emotion interpretation and context-based emotion judgment, while revealing relative limitations in comprehending perception subjectivity. When compared to humans, even top-performing MLLMs like GPT4o demonstrate remarkable performance gaps, underscoring key areas for future improvement. By developing a fundamental evaluation framework and conducting a comprehensive MLLM assessment, we hope this work contributes to advancing emotional intelligence in MLLMs. Project page: https://github.com/wdqqdw/MVEI.
>
---
#### [new 047] Improving Autism Detection with Multimodal Behavioral Analysis
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于自闭症检测任务，旨在解决现有方法在眼神特征表现差和泛化能力不足的问题。研究者构建了一个大规模平衡数据集，通过多模态行为分析（面部表情、语音、头部运动、HRV和眼动），提出新眼动特征，并采用晚融合策略，提升了分类准确率至74%。**

- **链接: [http://arxiv.org/pdf/2509.21352v1](http://arxiv.org/pdf/2509.21352v1)**

> **作者:** William Saakyan; Matthias Norden; Lola Eversmann; Simon Kirsch; Muyu Lin; Simon Guendelman; Isabel Dziobek; Hanna Drimalla
>
> **摘要:** Due to the complex and resource-intensive nature of diagnosing Autism Spectrum Condition (ASC), several computer-aided diagnostic support methods have been proposed to detect autism by analyzing behavioral cues in patient video data. While these models show promising results on some datasets, they struggle with poor gaze feature performance and lack of real-world generalizability. To tackle these challenges, we analyze a standardized video dataset comprising 168 participants with ASC (46% female) and 157 non-autistic participants (46% female), making it, to our knowledge, the largest and most balanced dataset available. We conduct a multimodal analysis of facial expressions, voice prosody, head motion, heart rate variability (HRV), and gaze behavior. To address the limitations of prior gaze models, we introduce novel statistical descriptors that quantify variability in eye gaze angles, improving gaze-based classification accuracy from 64% to 69% and aligning computational findings with clinical research on gaze aversion in ASC. Using late fusion, we achieve a classification accuracy of 74%, demonstrating the effectiveness of integrating behavioral markers across multiple modalities. Our findings highlight the potential for scalable, video-based screening tools to support autism assessment.
>
---
#### [new 048] HiGS: History-Guided Sampling for Plug-and-Play Enhancement of Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出HiGS方法，用于提升扩散模型的采样质量与效率。针对低采样步数或低引导尺度下生成图像细节不足的问题，HiGS通过引入历史预测信息，无需额外训练即可增强输出真实性，实验表明其在多种模型中均有效，并在少步骤下取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.22300v1](http://arxiv.org/pdf/2509.22300v1)**

> **作者:** Seyedmorteza Sadat; Farnood Salehi; Romann M. Weber
>
> **摘要:** While diffusion models have made remarkable progress in image generation, their outputs can still appear unrealistic and lack fine details, especially when using fewer number of neural function evaluations (NFEs) or lower guidance scales. To address this issue, we propose a novel momentum-based sampling technique, termed history-guided sampling (HiGS), which enhances quality and efficiency of diffusion sampling by integrating recent model predictions into each inference step. Specifically, HiGS leverages the difference between the current prediction and a weighted average of past predictions to steer the sampling process toward more realistic outputs with better details and structure. Our approach introduces practically no additional computation and integrates seamlessly into existing diffusion frameworks, requiring neither extra training nor fine-tuning. Extensive experiments show that HiGS consistently improves image quality across diverse models and architectures and under varying sampling budgets and guidance scales. Moreover, using a pretrained SiT model, HiGS achieves a new state-of-the-art FID of 1.61 for unguided ImageNet generation at 256$\times$256 with only 30 sampling steps (instead of the standard 250). We thus present HiGS as a plug-and-play enhancement to standard diffusion sampling that enables faster generation with higher fidelity.
>
---
#### [new 049] SRHand: Super-Resolving Hand Images and 3D Shapes via View/Pose-aware Neural Image Representations and Explicit 3D Meshes
- **分类: cs.CV**

- **简介: 该论文提出SRHand，用于从低分辨率图像中重建高精度的手部3D几何和纹理。针对现有方法在低分辨率输入下表现差的问题，结合隐式图像表示与显式3D网格，实现多视角与姿态一致性，提升细节重建效果。**

- **链接: [http://arxiv.org/pdf/2509.21859v1](http://arxiv.org/pdf/2509.21859v1)**

> **作者:** Minje Kim; Tae-Kyun Kim
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Reconstructing detailed hand avatars plays a crucial role in various applications. While prior works have focused on capturing high-fidelity hand geometry, they heavily rely on high-resolution multi-view image inputs and struggle to generalize on low-resolution images. Multi-view image super-resolution methods have been proposed to enforce 3D view consistency. These methods, however, are limited to static objects/scenes with fixed resolutions and are not applicable to articulated deformable hands. In this paper, we propose SRHand (Super-Resolution Hand), the method for reconstructing detailed 3D geometry as well as textured images of hands from low-resolution images. SRHand leverages the advantages of implicit image representation with explicit hand meshes. Specifically, we introduce a geometric-aware implicit image function (GIIF) that learns detailed hand prior by upsampling the coarse input images. By jointly optimizing the implicit image function and explicit 3D hand shapes, our method preserves multi-view and pose consistency among upsampled hand images, and achieves fine-detailed 3D reconstruction (wrinkles, nails). In experiments using the InterHand2.6M and Goliath datasets, our method significantly outperforms state-of-the-art image upsampling methods adapted to hand datasets, and 3D hand reconstruction methods, quantitatively and qualitatively. Project page: https://yunminjin2.github.io/projects/srhand
>
---
#### [new 050] Unsupervised Defect Detection for Surgical Instruments
- **分类: cs.CV**

- **简介: 该论文针对手术器械的缺陷检测任务，旨在解决现有方法在纹理背景误检、微小缺陷敏感度低及领域适配不足的问题。提出结合背景掩码、块级分析与高效领域适配的方法，提升检测可靠性。**

- **链接: [http://arxiv.org/pdf/2509.21561v1](http://arxiv.org/pdf/2509.21561v1)**

> **作者:** Joseph Huang; Yichi Zhang; Jingxi Yu; Wei Chen; Seunghyun Hwang; Qiang Qiu; Amy R. Reibman; Edward J. Delp; Fengqing Zhu
>
> **摘要:** Ensuring the safety of surgical instruments requires reliable detection of visual defects. However, manual inspection is prone to error, and existing automated defect detection methods, typically trained on natural/industrial images, fail to transfer effectively to the surgical domain. We demonstrate that simply applying or fine-tuning these approaches leads to issues: false positive detections arising from textured backgrounds, poor sensitivity to small, subtle defects, and inadequate capture of instrument-specific features due to domain shift. To address these challenges, we propose a versatile method that adapts unsupervised defect detection methods specifically for surgical instruments. By integrating background masking, a patch-based analysis strategy, and efficient domain adaptation, our method overcomes these limitations, enabling the reliable detection of fine-grained defects in surgical instrument imagery.
>
---
#### [new 051] Incorporating Scene Context and Semantic Labels for Enhanced Group-level Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文聚焦于群体情感识别任务，旨在解决现有方法忽视场景上下文和语义信息的问题。提出融合视觉场景上下文与标签引导的语义信息的框架，通过多模块编码与交互提升群体情感识别性能。**

- **链接: [http://arxiv.org/pdf/2509.21747v1](http://arxiv.org/pdf/2509.21747v1)**

> **作者:** Qing Zhu; Wangdong Guo; Qirong Mao; Xiaohua Huang; Xiuyan Shao; Wenming Zheng
>
> **备注:** 10 pages, 5figures, submitted to IEEE Transactions on Human-Machine Systems
>
> **摘要:** Group-level emotion recognition (GER) aims to identify holistic emotions within a scene involving multiple individuals. Current existed methods underestimate the importance of visual scene contextual information in modeling individual relationships. Furthermore, they overlook the crucial role of semantic information from emotional labels for complete understanding of emotions. To address this limitation, we propose a novel framework that incorporates visual scene context and label-guided semantic information to improve GER performance. It involves the visual context encoding module that leverages multi-scale scene information to diversely encode individual relationships. Complementarily, the emotion semantic encoding module utilizes group-level emotion labels to prompt a large language model to generate nuanced emotion lexicons. These lexicons, in conjunction with the emotion labels, are then subsequently refined into comprehensive semantic representations through the utilization of a structured emotion tree. Finally, similarity-aware interaction is proposed to align and integrate visual and semantic information, thereby generating enhanced group-level emotion representations and subsequently improving the performance of GER. Experiments on three widely adopted GER datasets demonstrate that our proposed method achieves competitive performance compared to state-of-the-art methods.
>
---
#### [new 052] PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data
- **分类: cs.CV**

- **简介: 该论文提出PartSAM，用于3D物体部件分割任务。针对现有方法依赖2D模型导致泛化性差的问题，PartSAM直接在大规模3D数据上训练，采用双分支编码器和自动生成的500万标注数据，实现了高精度、可扩展的开放世界3D部件分割。**

- **链接: [http://arxiv.org/pdf/2509.21965v1](http://arxiv.org/pdf/2509.21965v1)**

> **作者:** Zhe Zhu; Le Wan; Rui Xu; Yiheng Zhang; Honghua Chen; Zhiyang Dou; Cheng Lin; Yuan Liu; Mingqiang Wei
>
> **摘要:** Segmenting 3D objects into parts is a long-standing challenge in computer vision. To overcome taxonomy constraints and generalize to unseen 3D objects, recent works turn to open-world part segmentation. These approaches typically transfer supervision from 2D foundation models, such as SAM, by lifting multi-view masks into 3D. However, this indirect paradigm fails to capture intrinsic geometry, leading to surface-only understanding, uncontrolled decomposition, and limited generalization. We present PartSAM, the first promptable part segmentation model trained natively on large-scale 3D data. Following the design philosophy of SAM, PartSAM employs an encoder-decoder architecture in which a triplane-based dual-branch encoder produces spatially structured tokens for scalable part-aware representation learning. To enable large-scale supervision, we further introduce a model-in-the-loop annotation pipeline that curates over five million 3D shape-part pairs from online assets, providing diverse and fine-grained labels. This combination of scalable architecture and diverse 3D data yields emergent open-world capabilities: with a single prompt, PartSAM achieves highly accurate part identification, and in a Segment-Every-Part mode, it automatically decomposes shapes into both surface and internal structures. Extensive experiments show that PartSAM outperforms state-of-the-art methods by large margins across multiple benchmarks, marking a decisive step toward foundation models for 3D part understanding. Our code and model will be released soon.
>
---
#### [new 053] Johnson-Lindenstrauss Lemma Guided Network for Efficient 3D Medical Segmentation
- **分类: cs.CV**

- **简介: 该论文提出VeloxSeg，用于高效3D医学图像分割。针对轻量方法在复杂结构和多模态下的表现不足，设计了双流CNN-Transformer架构，结合PWA与JLC模块，并引入SDKT提升表征能力，显著提升了效率与分割性能。**

- **链接: [http://arxiv.org/pdf/2509.22307v1](http://arxiv.org/pdf/2509.22307v1)**

> **作者:** Jinpeng Lu; Linghan Cai; Yinda Chen; Guo Tang; Songhan Jiang; Haoyuan Shi; Zhiwei Xiong
>
> **摘要:** Lightweight 3D medical image segmentation remains constrained by a fundamental "efficiency / robustness conflict", particularly when processing complex anatomical structures and heterogeneous modalities. In this paper, we study how to redesign the framework based on the characteristics of high-dimensional 3D images, and explore data synergy to overcome the fragile representation of lightweight methods. Our approach, VeloxSeg, begins with a deployable and extensible dual-stream CNN-Transformer architecture composed of Paired Window Attention (PWA) and Johnson-Lindenstrauss lemma-guided convolution (JLC). For each 3D image, we invoke a "glance-and-focus" principle, where PWA rapidly retrieves multi-scale information, and JLC ensures robust local feature extraction with minimal parameters, significantly enhancing the model's ability to operate with low computational budget. Followed by an extension of the dual-stream architecture that incorporates modal interaction into the multi-scale image-retrieval process, VeloxSeg efficiently models heterogeneous modalities. Finally, Spatially Decoupled Knowledge Transfer (SDKT) via Gram matrices injects the texture prior extracted by a self-supervised network into the segmentation network, yielding stronger representations than baselines at no extra inference cost. Experimental results on multimodal benchmarks show that VeloxSeg achieves a 26% Dice improvement, alongside increasing GPU throughput by 11x and CPU by 48x. Codes are available at https://github.com/JinPLu/VeloxSeg.
>
---
#### [new 054] Towards Faithful Reasoning in Remote Sensing: A Perceptually-Grounded GeoSpatial Chain-of-Thought for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对遥感领域视觉-语言模型（VLMs）在复杂分析任务中推理能力不足的问题，提出了一种可验证的多步骤Geo-CoT框架，并通过SFT和GRPO训练策略构建了RSThinker模型，显著提升了分析性能。**

- **链接: [http://arxiv.org/pdf/2509.22221v1](http://arxiv.org/pdf/2509.22221v1)**

> **作者:** Jiaqi Liu; Lang Sun; Ronghao Fu; Bo Yang
>
> **摘要:** Vision-Language Models (VLMs) in remote sensing often fail at complex analytical tasks, a limitation stemming from their end-to-end training paradigm that bypasses crucial reasoning steps and leads to unverifiable outputs. To address this limitation, we introduce the Perceptually-Grounded Geospatial Chain-of-Thought (Geo-CoT), a framework that models remote sensing analysis as a verifiable, multi-step process. We instill this analytical process through a two-stage alignment strategy, leveraging Geo-CoT380k, the first large-scale dataset of structured Geo-CoT rationales. This strategy first employs supervised fine-tuning (SFT) to instill the foundational cognitive architecture, then leverages Group Reward Policy Optimization (GRPO) to refine the model's reasoning policy towards factual correctness. The resulting model, RSThinker, outputs both a final answer and its justifying, verifiable analytical trace. This capability yields dominant performance, significantly outperforming state-of-the-art models across a comprehensive range of tasks. The public release of our Geo-CoT380k dataset and RSThinker model upon publication serves as a concrete pathway from opaque perception towards structured, verifiable reasoning for Earth Observation.
>
---
#### [new 055] Gradient-based multi-focus image fusion with focus-aware saliency enhancement
- **分类: cs.CV**

- **简介: 该论文研究多焦点图像融合任务，旨在解决现有方法在保留清晰焦界面上的不足。提出基于梯度域的模型和显著性增强方法，有效提升边界细节和聚焦区域质量，实验验证了方法的优越性。**

- **链接: [http://arxiv.org/pdf/2509.22392v1](http://arxiv.org/pdf/2509.22392v1)**

> **作者:** Haoyu Li; XiaoSong Li
>
> **备注:** iCIG 2025
>
> **摘要:** Multi-focus image fusion (MFIF) aims to yield an all-focused image from multiple partially focused inputs, which is crucial in applications cover sur-veillance, microscopy, and computational photography. However, existing methods struggle to preserve sharp focus-defocus boundaries, often resulting in blurred transitions and focused details loss. To solve this problem, we propose a MFIF method based on significant boundary enhancement, which generates high-quality fused boundaries while effectively detecting focus in-formation. Particularly, we propose a gradient-domain-based model that can obtain initial fusion results with complete boundaries and effectively pre-serve the boundary details. Additionally, we introduce Tenengrad gradient detection to extract salient features from both the source images and the ini-tial fused image, generating the corresponding saliency maps. For boundary refinement, we develop a focus metric based on gradient and complementary information, integrating the salient features with the complementary infor-mation across images to emphasize focused regions and produce a high-quality initial decision result. Extensive experiments on four public datasets demonstrate that our method consistently outperforms 12 state-of-the-art methods in both subjective and objective evaluations. We have realized codes in https://github.com/Lihyua/GICI
>
---
#### [new 056] CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦图像描述生成任务，旨在解决监督微调方法导致模型泛化能力差、描述单一的问题。提出CapRL框架，利用强化学习通过语言模型回答问题的准确性作为奖励，提升图像描述的多样性和实用性。**

- **链接: [http://arxiv.org/pdf/2509.22647v1](http://arxiv.org/pdf/2509.22647v1)**

> **作者:** Long Xing; Xiaoyi Dong; Yuhang Zang; Yuhang Cao; Jianze Liang; Qidong Huang; Jiaqi Wang; Feng Wu; Dahua Lin
>
> **备注:** Code is available at https://github.com/InternLM/CapRL
>
> **摘要:** Image captioning is a fundamental task that bridges the visual and linguistic domains, playing a critical role in pre-training Large Vision-Language Models (LVLMs). Current state-of-the-art captioning models are typically trained with Supervised Fine-Tuning (SFT), a paradigm that relies on expensive, non-scalable data annotated by humans or proprietary models. This approach often leads to models that memorize specific ground-truth answers, limiting their generality and ability to generate diverse, creative descriptions. To overcome the limitation of SFT, we propose applying the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm to the open-ended task of image captioning. A primary challenge, however, is designing an objective reward function for the inherently subjective nature of what constitutes a "good" caption. We introduce Captioning Reinforcement Learning (CapRL), a novel training framework that redefines caption quality through its utility: a high-quality caption should enable a non-visual language model to accurately answer questions about the corresponding image. CapRL employs a decoupled two-stage pipeline where an LVLM generates a caption, and the objective reward is derived from the accuracy of a separate, vision-free LLM answering Multiple-Choice Questions based solely on that caption. As the first study to apply RLVR to the subjective image captioning task, we demonstrate that CapRL significantly enhances multiple settings. Pretraining on the CapRL-5M caption dataset annotated by CapRL-3B results in substantial gains across 12 benchmarks. Moreover, within the Prism Framework for caption quality evaluation, CapRL achieves performance comparable to Qwen2.5-VL-72B, while exceeding the baseline by an average margin of 8.4%. Code is available here: https://github.com/InternLM/CapRL.
>
---
#### [new 057] UML-CoT: Structured Reasoning and Planning with Unified Modeling Language for Robotic Room Cleaning
- **分类: cs.CV; I.2.6; I.2.7; I.2.8; I.4.8; I.5.4**

- **简介: 该论文提出UML-CoT，利用统一建模语言（UML）改进机器人房间清洁任务中的结构化推理与规划。针对传统CoT在可解释性和执行性上的不足，UML-CoT通过类图和活动图实现更高效、可执行的行动规划。**

- **链接: [http://arxiv.org/pdf/2509.22628v1](http://arxiv.org/pdf/2509.22628v1)**

> **作者:** Hongyu Chen; Guangrun Wang
>
> **摘要:** Chain-of-Thought (CoT) prompting improves reasoning in large language models (LLMs), but its reliance on unstructured text limits interpretability and executability in embodied tasks. Prior work has explored structured CoTs using scene or logic graphs, yet these remain fundamentally limited: they model only low-order relations, lack constructs like inheritance or behavioral abstraction, and provide no standardized semantics for sequential or conditional planning. We propose UML-CoT, a structured reasoning and planning framework that leverages Unified Modeling Language (UML) to generate symbolic CoTs and executable action plans. UML class diagrams capture compositional object semantics, while activity diagrams model procedural control flow. Our three-stage training pipeline combines supervised fine-tuning with Group Relative Policy Optimization (GRPO), including reward learning from answer-only data. We evaluate UML-CoT on MRoom-30k, a new benchmark of cluttered room-cleaning scenarios. UML-CoT outperforms unstructured CoTs in interpretability, planning coherence, and execution success, highlighting UML as a more expressive and actionable structured reasoning formalism.
>
---
#### [new 058] Integrating Background Knowledge in Medical Semantic Segmentation with Logic Tensor Networks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学语义分割任务，旨在通过引入逻辑张量网络（LTNs）将医学背景知识整合到模型中，以提升分割性能。作者在SwinUNETR框架中应用LTNs，利用一阶逻辑规则约束分割结果的形状与区域关系，并在海马体MRI分割任务上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2509.22399v1](http://arxiv.org/pdf/2509.22399v1)**

> **作者:** Luca Bergamin; Giovanna Maria Dimitri; Fabio Aiolli
>
> **备注:** Accepted at TAIM@IJCNN 2025
>
> **摘要:** Semantic segmentation is a fundamental task in medical image analysis, aiding medical decision-making by helping radiologists distinguish objects in an image. Research in this field has been driven by deep learning applications, which have the potential to scale these systems even in the presence of noise and artifacts. However, these systems are not yet perfected. We argue that performance can be improved by incorporating common medical knowledge into the segmentation model's loss function. To this end, we introduce Logic Tensor Networks (LTNs) to encode medical background knowledge using first-order logic (FOL) rules. The encoded rules span from constraints on the shape of the produced segmentation, to relationships between different segmented areas. We apply LTNs in an end-to-end framework with a SwinUNETR for semantic segmentation. We evaluate our method on the task of segmenting the hippocampus in brain MRI scans. Our experiments show that LTNs improve the baseline segmentation performance, especially when training data is scarce. Despite being in its preliminary stages, we argue that neurosymbolic methods are general enough to be adapted and applied to other medical semantic segmentation tasks.
>
---
#### [new 059] No-Reference Image Contrast Assessment with Customized EfficientNet-B0
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究无参考图像对比度质量评估任务，旨在提升模型在真实场景下对对比度失真的感知评估能力。工作内容包括定制并微调EfficientNet-B0、ResNet18和MobileNetV2，引入对比度感知回归头，并在两个基准数据集上训练评估。结果表明，改进的EfficientNet-B0性能最优，达到SOTA水平。**

- **链接: [http://arxiv.org/pdf/2509.21967v1](http://arxiv.org/pdf/2509.21967v1)**

> **作者:** Javad Hassannataj Joloudari; Bita Mesbahzadeh; Omid Zare; Emrah Arslan; Roohallah Alizadehsani; Hossein Moosaei
>
> **备注:** 32 pages, 9 tables, 6 figures
>
> **摘要:** Image contrast was a fundamental factor in visual perception and played a vital role in overall image quality. However, most no reference image quality assessment NR IQA models struggled to accurately evaluate contrast distortions under diverse real world conditions. In this study, we proposed a deep learning based framework for blind contrast quality assessment by customizing and fine-tuning three pre trained architectures, EfficientNet B0, ResNet18, and MobileNetV2, for perceptual Mean Opinion Score, along with an additional model built on a Siamese network, which indicated a limited ability to capture perceptual contrast distortions. Each model is modified with a contrast-aware regression head and trained end to end using targeted data augmentations on two benchmark datasets, CID2013 and CCID2014, containing synthetic and authentic contrast distortions. Performance is evaluated using Pearson Linear Correlation Coefficient and Spearman Rank Order Correlation Coefficient, which assess the alignment between predicted and human rated scores. Among these three models, our customized EfficientNet B0 model achieved state-of-the-art performance with PLCC = 0.9286 and SRCC = 0.9178 on CCID2014 and PLCC = 0.9581 and SRCC = 0.9369 on CID2013, surpassing traditional methods and outperforming other deep baselines. These results highlighted the models robustness and effectiveness in capturing perceptual contrast distortion. Overall, the proposed method demonstrated that contrast aware adaptation of lightweight pre trained networks can yield a high performing, scalable solution for no reference contrast quality assessment suitable for real time and resource constrained applications.
>
---
#### [new 060] VLCE: A Knowledge-Enhanced Framework for Image Description in Disaster Assessment
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出VLCE，一种用于灾害评估图像描述的多模态框架。针对传统方法效率低、信息量少的问题，VLCE结合CNN-LSTM和ViT模型，并引入外部语义知识，提升描述的准确性和信息量，在RescueNet和xBD数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.21609v1](http://arxiv.org/pdf/2509.21609v1)**

> **作者:** Md. Mahfuzur Rahman; Kishor Datta Gupta; Marufa Kamal; Fahad Rahman; Sunzida Siddique; Ahmed Rafi Hasan; Mohd Ariful Haque; Roy George
>
> **备注:** 29 pages, 40 figures, 3 algorithms
>
> **摘要:** Immediate damage assessment is essential after natural catastrophes; yet, conventional hand evaluation techniques are sluggish and perilous. Although satellite and unmanned aerial vehicle (UAV) photos offer extensive perspectives of impacted regions, current computer vision methodologies generally yield just classification labels or segmentation masks, so constraining their capacity to deliver a thorough situational comprehension. We introduce the Vision Language Caption Enhancer (VLCE), a multimodal system designed to produce comprehensive, contextually-informed explanations of disaster imagery. VLCE employs a dual-architecture approach: a CNN-LSTM model with a ResNet50 backbone pretrained on EuroSat satellite imagery for the xBD dataset, and a Vision Transformer (ViT) model pretrained on UAV pictures for the RescueNet dataset. Both systems utilize external semantic knowledge from ConceptNet and WordNet to expand vocabulary coverage and improve description accuracy. We assess VLCE in comparison to leading vision-language models (LLaVA and QwenVL) utilizing CLIPScore for semantic alignment and InfoMetIC for caption informativeness. Experimental findings indicate that VLCE markedly surpasses baseline models, attaining a maximum of 95.33% on InfoMetIC while preserving competitive semantic alignment. Our dual-architecture system demonstrates significant potential for improving disaster damage assessment by automating the production of actionable, information-dense descriptions from satellite and drone photos.
>
---
#### [new 061] U-MAN: U-Net with Multi-scale Adaptive KAN Network for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割任务，旨在解决传统U-Net在保留细节和边界上的不足。提出了U-MAN模型，通过引入PAGF和MAN模块，增强特征融合与多尺度处理能力，提升了分割精度和边界定义能力。**

- **链接: [http://arxiv.org/pdf/2509.22444v1](http://arxiv.org/pdf/2509.22444v1)**

> **作者:** Bohan Huang; Qianyun Bao; Haoyuan Ma
>
> **备注:** 5 pages
>
> **摘要:** Medical image segmentation faces significant challenges in preserving fine-grained details and precise boundaries due to complex anatomical structures and pathological regions. These challenges primarily stem from two key limitations of conventional U-Net architectures: (1) their simple skip connections ignore the encoder-decoder semantic gap between various features, and (2) they lack the capability for multi-scale feature extraction in deep layers. To address these challenges, we propose the U-Net with Multi-scale Adaptive KAN (U-MAN), a novel architecture that enhances the emerging Kolmogorov-Arnold Network (KAN) with two specialized modules: Progressive Attention-Guided Feature Fusion (PAGF) and the Multi-scale Adaptive KAN (MAN). Our PAGF module replaces the simple skip connection, using attention to fuse features from the encoder and decoder. The MAN module enables the network to adaptively process features at multiple scales, improving its ability to segment objects of various sizes. Experiments on three public datasets (BUSI, GLAS, and CVC) show that U-MAN outperforms state-of-the-art methods, particularly in defining accurate boundaries and preserving fine details.
>
---
#### [new 062] MDF-MLLM: Deep Fusion Through Cross-Modal Feature Alignment for Contextually Aware Fundoscopic Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MDF-MLLM模型，用于眼底图像分类任务。针对现有模型难以捕捉视网膜疾病关键细节的问题，通过多尺度图像特征与文本信息的深度融合，显著提升了分类准确率（达94%），适用于临床辅助诊断系统。**

- **链接: [http://arxiv.org/pdf/2509.21358v1](http://arxiv.org/pdf/2509.21358v1)**

> **作者:** Jason Jordan; Mohammadreza Akbari Lor; Peter Koulen; Mei-Ling Shyu; Shu-Ching Chen
>
> **备注:** Word count: 5157, Table count: 2, Figure count: 5
>
> **摘要:** This study aimed to enhance disease classification accuracy from retinal fundus images by integrating fine-grained image features and global textual context using a novel multimodal deep learning architecture. Existing multimodal large language models (MLLMs) often struggle to capture low-level spatial details critical for diagnosing retinal diseases such as glaucoma, diabetic retinopathy, and retinitis pigmentosa. This model development and validation study was conducted on 1,305 fundus image-text pairs compiled from three public datasets (FIVES, HRF, and StoneRounds), covering acquired and inherited retinal diseases, and evaluated using classification accuracy and F1-score. The MDF-MLLM integrates skip features from four U-Net encoder layers into cross-attention blocks within a LLaMA 3.2 11B MLLM. Vision features are patch-wise projected and fused using scaled cross-attention and FiLM-based U-Net modulation. Baseline MLLM achieved 60% accuracy on the dual-type disease classification task. MDF-MLLM, with both U-Net and MLLM components fully fine-tuned during training, achieved a significantly higher accuracy of 94%, representing a 56% improvement. Recall and F1-scores improved by as much as 67% and 35% over baseline, respectively. Ablation studies confirmed that the multi-depth fusion approach contributed to substantial gains in spatial reasoning and classification, particularly for inherited diseases with rich clinical text. MDF-MLLM presents a generalizable, interpretable, and modular framework for fundus image classification, outperforming traditional MLLM baselines through multi-scale feature fusion. The architecture holds promise for real-world deployment in clinical decision support systems. Future work will explore synchronized training techniques, a larger pool of diseases for more generalizability, and extending the model for segmentation tasks.
>
---
#### [new 063] SingRef6D: Monocular Novel Object Pose Estimation with a Single RGB Reference
- **分类: cs.CV**

- **简介: 该论文提出SingRef6D，用于单目RGB图像的6D姿态估计任务。针对深度传感器依赖性强、低光照和无纹理场景匹配性能差的问题，提出了基于Depth-Anything v2的改进深度预测方法和深度感知匹配机制，提升了在复杂条件下的鲁棒性和精度。**

- **链接: [http://arxiv.org/pdf/2509.21927v1](http://arxiv.org/pdf/2509.21927v1)**

> **作者:** Jiahui Wang; Haiyue Zhu; Haoren Guo; Abdullah Al Mamun; Cheng Xiang; Tong Heng Lee
>
> **备注:** Accepted as a poster in NeurIPS 2025
>
> **摘要:** Recent 6D pose estimation methods demonstrate notable performance but still face some practical limitations. For instance, many of them rely heavily on sensor depth, which may fail with challenging surface conditions, such as transparent or highly reflective materials. In the meantime, RGB-based solutions provide less robust matching performance in low-light and texture-less scenes due to the lack of geometry information. Motivated by these, we propose SingRef6D, a lightweight pipeline requiring only a single RGB image as a reference, eliminating the need for costly depth sensors, multi-view image acquisition, or training view synthesis models and neural fields. This enables SingRef6D to remain robust and capable even under resource-limited settings where depth or dense templates are unavailable. Our framework incorporates two key innovations. First, we propose a token-scaler-based fine-tuning mechanism with a novel optimization loss on top of Depth-Anything v2 to enhance its ability to predict accurate depth, even for challenging surfaces. Our results show a 14.41% improvement (in $\delta_{1.05}$) on REAL275 depth prediction compared to Depth-Anything v2 (with fine-tuned head). Second, benefiting from depth availability, we introduce a depth-aware matching process that effectively integrates spatial relationships within LoFTR, enabling our system to handle matching for challenging materials and lighting conditions. Evaluations of pose estimation on the REAL275, ClearPose, and Toyota-Light datasets show that our approach surpasses state-of-the-art methods, achieving a 6.1% improvement in average recall.
>
---
#### [new 064] EgoInstruct: An Egocentric Video Dataset of Face-to-face Instructional Interactions with Multi-modal LLM Benchmarking
- **分类: cs.CV**

- **简介: 该论文提出了EgoInstruct数据集，用于研究面对面教学场景中的指令交互。针对现有数据和方法不足的问题，构建了包含多模态标注的数据集，并对多模态大模型（MLLMs）进行了基准测试，验证其在无需微调情况下优于传统模型，推动教学交互理解任务的发展。**

- **链接: [http://arxiv.org/pdf/2509.22019v1](http://arxiv.org/pdf/2509.22019v1)**

> **作者:** Yuki Sakai; Ryosuke Furuta; Juichun Yen; Yoichi Sato
>
> **备注:** Accepted to the I-HFM Workshop at ICCV 2025
>
> **摘要:** Analyzing instructional interactions between an instructor and a learner who are co-present in the same physical space is a critical problem for educational support and skill transfer. Yet such face-to-face instructional scenes have not been systematically studied in computer vision. We identify two key reasons: i) the lack of suitable datasets and ii) limited analytical techniques. To address this gap, we present a new egocentric video dataset of face-to-face instruction and provide ground-truth annotations for two fundamental tasks that serve as a first step toward a comprehensive understanding of instructional interactions: procedural step segmentation and conversation-state classification. Using this dataset, we benchmark multimodal large language models (MLLMs) against conventional task-specific models. Since face-to-face instruction involves multiple modalities (speech content and prosody, gaze and body motion, and visual context), effective understanding requires methods that handle verbal and nonverbal communication in an integrated manner. Accordingly, we evaluate recently introduced MLLMs that jointly process images, audio, and text. This evaluation quantifies the extent to which current machine learning models understand face-to-face instructional scenes. In experiments, MLLMs outperform specialized baselines even without task-specific fine-tuning, suggesting their promise for holistic understanding of instructional interactions.
>
---
#### [new 065] Automated Prompt Generation for Creative and Counterfactual Text-to-image Synthesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦文本到图像生成中的反事实可控性问题，提出自动提示工程框架，通过改进提示生成反常规模的创意图像。构建首个反事实尺寸数据集，提升生成效果与可控性。**

- **链接: [http://arxiv.org/pdf/2509.21375v1](http://arxiv.org/pdf/2509.21375v1)**

> **作者:** Aleksa Jelaca; Ying Jiao; Chang Tian; Marie-Francine Moens
>
> **备注:** text-to-image generation, automatic prompt, DPO, Counterfactual
>
> **摘要:** Text-to-image generation has advanced rapidly with large-scale multimodal training, yet fine-grained controllability remains a critical challenge. Counterfactual controllability, defined as the capacity to deliberately generate images that contradict common-sense patterns, remains a major challenge but plays a crucial role in enabling creativity and exploratory applications. In this work, we address this gap with a focus on counterfactual size (e.g., generating a tiny walrus beside a giant button) and propose an automatic prompt engineering framework that adapts base prompts into revised prompts for counterfactual images. The framework comprises three components: an image evaluator that guides dataset construction by identifying successful image generations, a supervised prompt rewriter that produces revised prompts, and a DPO-trained ranker that selects the optimal revised prompt. We construct the first counterfactual size text-image dataset and enhance the image evaluator by extending Grounded SAM with refinements, achieving a 114 percent improvement over its backbone. Experiments demonstrate that our method outperforms state-of-the-art baselines and ChatGPT-4o, establishing a foundation for future research on counterfactual controllability.
>
---
#### [new 066] High-Quality Sound Separation Across Diverse Categories via Visually-Guided Generative Modeling
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出DAVIS，一种基于扩散模型和流匹配的视听音源分离框架，旨在解决传统方法在分离多类别声音时质量受限的问题。通过生成式学习直接合成目标频谱，提升分离效果，并在标准数据集上验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2509.22063v1](http://arxiv.org/pdf/2509.22063v1)**

> **作者:** Chao Huang; Susan Liang; Yapeng Tian; Anurag Kumar; Chenliang Xu
>
> **备注:** Accepted to IJCV
>
> **摘要:** We propose DAVIS, a Diffusion-based Audio-VIsual Separation framework that solves the audio-visual sound source separation task through generative learning. Existing methods typically frame sound separation as a mask-based regression problem, achieving significant progress. However, they face limitations in capturing the complex data distribution required for high-quality separation of sounds from diverse categories. In contrast, DAVIS circumvents these issues by leveraging potent generative modeling paradigms, specifically Denoising Diffusion Probabilistic Models (DDPM) and the more recent Flow Matching (FM), integrated within a specialized Separation U-Net architecture. Our framework operates by synthesizing the desired separated sound spectrograms directly from a noise distribution, conditioned concurrently on the mixed audio input and associated visual information. The inherent nature of its generative objective makes DAVIS particularly adept at producing high-quality sound separations for diverse sound categories. We present comparative evaluations of DAVIS, encompassing both its DDPM and Flow Matching variants, against leading methods on the standard AVE and MUSIC datasets. The results affirm that both variants surpass existing approaches in separation quality, highlighting the efficacy of our generative framework for tackling the audio-visual source separation task.
>
---
#### [new 067] FlashEdit: Decoupling Speed, Structure, and Semantics for Precise Image Editing
- **分类: cs.CV**

- **简介: 该论文提出FlashEdit，用于文本引导的图像编辑任务。旨在解决扩散模型编辑速度慢的问题。通过OSIE、BG-Shield和SSCA三项创新，实现了高保真、实时的图像编辑，背景一致性好且结构完整，速度提升150倍。**

- **链接: [http://arxiv.org/pdf/2509.22244v1](http://arxiv.org/pdf/2509.22244v1)**

> **作者:** Junyi Wu; Zhiteng Li; Haotong Qin; Xiaohong Liu; Linghe Kong; Yulun Zhang; Xiaokang Yang
>
> **备注:** Our code will be made publicly available at https://github.com/JunyiWuCode/FlashEdit
>
> **摘要:** Text-guided image editing with diffusion models has achieved remarkable quality but suffers from prohibitive latency, hindering real-world applications. We introduce FlashEdit, a novel framework designed to enable high-fidelity, real-time image editing. Its efficiency stems from three key innovations: (1) a One-Step Inversion-and-Editing (OSIE) pipeline that bypasses costly iterative processes; (2) a Background Shield (BG-Shield) technique that guarantees background preservation by selectively modifying features only within the edit region; and (3) a Sparsified Spatial Cross-Attention (SSCA) mechanism that ensures precise, localized edits by suppressing semantic leakage to the background. Extensive experiments demonstrate that FlashEdit maintains superior background consistency and structural integrity, while performing edits in under 0.2 seconds, which is an over 150$\times$ speedup compared to prior multi-step methods. Our code will be made publicly available at https://github.com/JunyiWuCode/FlashEdit.
>
---
#### [new 068] Hierarchical Representation Matching for CLIP-based Class-Incremental Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对基于CLIP的类增量学习任务，提出HERMAN方法。通过利用大语言模型生成具有层次语义的文本描述符，并匹配不同层级的视觉特征，增强模型在增量学习中的判别能力和防止灾难性遗忘。**

- **链接: [http://arxiv.org/pdf/2509.22645v1](http://arxiv.org/pdf/2509.22645v1)**

> **作者:** Zhen-Hao Wen; Yan Wang; Ji Feng; Han-Jia Ye; De-Chuan Zhan; Da-Wei Zhou
>
> **摘要:** Class-Incremental Learning (CIL) aims to endow models with the ability to continuously adapt to evolving data streams. Recent advances in pre-trained vision-language models (e.g., CLIP) provide a powerful foundation for this task. However, existing approaches often rely on simplistic templates, such as "a photo of a [CLASS]", which overlook the hierarchical nature of visual concepts. For example, recognizing "cat" versus "car" depends on coarse-grained cues, while distinguishing "cat" from "lion" requires fine-grained details. Similarly, the current feature mapping in CLIP relies solely on the representation from the last layer, neglecting the hierarchical information contained in earlier layers. In this work, we introduce HiErarchical Representation MAtchiNg (HERMAN) for CLIP-based CIL. Our approach leverages LLMs to recursively generate discriminative textual descriptors, thereby augmenting the semantic space with explicit hierarchical cues. These descriptors are matched to different levels of the semantic hierarchy and adaptively routed based on task-specific requirements, enabling precise discrimination while alleviating catastrophic forgetting in incremental tasks. Extensive experiments on multiple benchmarks demonstrate that our method consistently achieves state-of-the-art performance.
>
---
#### [new 069] Motion-Aware Transformer for Multi-Object Tracking
- **分类: cs.CV**

- **简介: 该论文针对视频中的多目标跟踪（MOT）任务，旨在解决目标运动复杂、检测与跟踪查询冲突导致的关联精度下降问题。提出Motion-Aware Transformer（MATR），通过显式预测目标运动提前更新跟踪查询，减少冲突，提升检测和关联性能。**

- **链接: [http://arxiv.org/pdf/2509.21715v1](http://arxiv.org/pdf/2509.21715v1)**

> **作者:** Xu Yang; Gady Agam
>
> **摘要:** Multi-object tracking (MOT) in videos remains challenging due to complex object motions and crowded scenes. Recent DETR-based frameworks offer end-to-end solutions but typically process detection and tracking queries jointly within a single Transformer Decoder layer, leading to conflicts and degraded association accuracy. We introduce the Motion-Aware Transformer (MATR), which explicitly predicts object movements across frames to update track queries in advance. By reducing query collisions, MATR enables more consistent training and improves both detection and association. Extensive experiments on DanceTrack, SportsMOT, and BDD100k show that MATR delivers significant gains across standard metrics. On DanceTrack, MATR improves HOTA by more than 9 points over MOTR without additional data and reaches a new state-of-the-art score of 71.3 with supplementary data. MATR also achieves state-of-the-art results on SportsMOT (72.2 HOTA) and BDD100k (54.7 mTETA, 41.6 mHOTA) without relying on external datasets. These results demonstrate that explicitly modeling motion within end-to-end Transformers offers a simple yet highly effective approach to advancing multi-object tracking.
>
---
#### [new 070] Training-Free Multimodal Deepfake Detection via Graph Reasoning
- **分类: cs.CV; cs.CY**

- **简介: 该论文针对多模态深度伪造检测（MDD）任务，提出GASP-ICL框架，无需训练即可利用大视觉语言模型（LVLMs）。通过构建语义对齐的样本集和图结构评分机制，有效捕捉跨模态线索，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.21774v1](http://arxiv.org/pdf/2509.21774v1)**

> **作者:** Yuxin Liu; Fei Wang; Kun Li; Yiqi Nie; Junjie Chen; Yanyan Wei; Zhangling Duan; Zhaohong Jia
>
> **摘要:** Multimodal deepfake detection (MDD) aims to uncover manipulations across visual, textual, and auditory modalities, thereby reinforcing the reliability of modern information systems. Although large vision-language models (LVLMs) exhibit strong multimodal reasoning, their effectiveness in MDD is limited by challenges in capturing subtle forgery cues, resolving cross-modal inconsistencies, and performing task-aligned retrieval. To this end, we propose Guided Adaptive Scorer and Propagation In-Context Learning (GASP-ICL), a training-free framework for MDD. GASP-ICL employs a pipeline to preserve semantic relevance while injecting task-aware knowledge into LVLMs. We leverage an MDD-adapted feature extractor to retrieve aligned image-text pairs and build a candidate set. We further design the Graph-Structured Taylor Adaptive Scorer (GSTAS) to capture cross-sample relations and propagate query-aligned signals, producing discriminative exemplars. This enables precise selection of semantically aligned, task-relevant demonstrations, enhancing LVLMs for robust MDD. Experiments on four forgery types show that GASP-ICL surpasses strong baselines, delivering gains without LVLM fine-tuning.
>
---
#### [new 071] Residual Vector Quantization For Communication-Efficient Multi-Agent Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究多智能体协作感知任务，旨在解决通信带宽限制问题。提出ReVQom方法，通过残差向量量化压缩特征，实现高效通信，在低带宽下保持感知精度，推动V2X应用落地。**

- **链接: [http://arxiv.org/pdf/2509.21464v1](http://arxiv.org/pdf/2509.21464v1)**

> **作者:** Dereje Shenkut; B. V. K Vijaya Kumar
>
> **备注:** 5 pages
>
> **摘要:** Multi-agent collaborative perception (CP) improves scene understanding by sharing information across connected agents such as autonomous vehicles, unmanned aerial vehicles, and robots. Communication bandwidth, however, constrains scalability. We present ReVQom, a learned feature codec that preserves spatial identity while compressing intermediate features. ReVQom is an end-to-end method that compresses feature dimensions via a simple bottleneck network followed by multi-stage residual vector quantization (RVQ). This allows only per-pixel code indices to be transmitted, reducing payloads from 8192 bits per pixel (bpp) of uncompressed 32-bit float features to 6-30 bpp per agent with minimal accuracy loss. On DAIR-V2X real-world CP dataset, ReVQom achieves 273x compression at 30 bpp to 1365x compression at 6 bpp. At 18 bpp (455x), ReVQom matches or outperforms raw-feature CP, and at 6-12 bpp it enables ultra-low-bandwidth operation with graceful degradation. ReVQom allows efficient and accurate multi-agent collaborative perception with a step toward practical V2X deployment.
>
---
#### [new 072] Training-Free Synthetic Data Generation with Dual IP-Adapter Guidance
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对少样本图像分类任务，提出了一种无需训练的合成数据生成方法DIPSY。利用双IP-Adapter引导，通过图像到图像翻译生成高判别性的合成图像，无需微调或外部工具，提升了分类性能。**

- **链接: [http://arxiv.org/pdf/2509.22635v1](http://arxiv.org/pdf/2509.22635v1)**

> **作者:** Luc Boudier; Loris Manganelli; Eleftherios Tsonis; Nicolas Dufour; Vicky Kalogeiton
>
> **备注:** BMVC 2025. Project page: https://www.lix.polytechnique.fr/vista/projects/2025_bmvc_dipsy/
>
> **摘要:** Few-shot image classification remains challenging due to the limited availability of labeled examples. Recent approaches have explored generating synthetic training data using text-to-image diffusion models, but often require extensive model fine-tuning or external information sources. We present a novel training-free approach, called DIPSY, that leverages IP-Adapter for image-to-image translation to generate highly discriminative synthetic images using only the available few-shot examples. DIPSY introduces three key innovations: (1) an extended classifier-free guidance scheme that enables independent control over positive and negative image conditioning; (2) a class similarity-based sampling strategy that identifies effective contrastive examples; and (3) a simple yet effective pipeline that requires no model fine-tuning or external captioning and filtering. Experiments across ten benchmark datasets demonstrate that our approach achieves state-of-the-art or comparable performance, while eliminating the need for generative model adaptation or reliance on external tools for caption generation and image filtering. Our results highlight the effectiveness of leveraging dual image prompting with positive-negative guidance for generating class-discriminative features, particularly for fine-grained classification tasks.
>
---
#### [new 073] HyCoVAD: A Hybrid SSL-LLM Model for Complex Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文提出HyCoVAD，一种用于复杂视频异常检测的混合SSL-LLM模型。针对传统方法难以捕捉多实体间语义关系的问题，结合SSL提取时空特征与LLM进行语义验证，提升了复杂场景下异常检测性能。**

- **链接: [http://arxiv.org/pdf/2509.22544v1](http://arxiv.org/pdf/2509.22544v1)**

> **作者:** Mohammad Mahdi Hemmatyar; Mahdi Jafari; Mohammad Amin Yousefi; Mohammad Reza Nemati; Mobin Azadani; Hamid Reza Rastad; Amirmohammad Akbari
>
> **摘要:** Video anomaly detection (VAD) is crucial for intelligent surveillance, but a significant challenge lies in identifying complex anomalies, which are events defined by intricate relationships and temporal dependencies among multiple entities rather than by isolated actions. While self-supervised learning (SSL) methods effectively model low-level spatiotemporal patterns, they often struggle to grasp the semantic meaning of these interactions. Conversely, large language models (LLMs) offer powerful contextual reasoning but are computationally expensive for frame-by-frame analysis and lack fine-grained spatial localization. We introduce HyCoVAD, Hybrid Complex Video Anomaly Detection, a hybrid SSL-LLM model that combines a multi-task SSL temporal analyzer with LLM validator. The SSL module is built upon an nnFormer backbone which is a transformer-based model for image segmentation. It is trained with multiple proxy tasks, learns from video frames to identify those suspected of anomaly. The selected frames are then forwarded to the LLM, which enriches the analysis with semantic context by applying structured, rule-based reasoning to validate the presence of anomalies. Experiments on the challenging ComplexVAD dataset show that HyCoVAD achieves a 72.5% frame-level AUC, outperforming existing baselines by 12.5% while reducing LLM computation. We release our interaction anomaly taxonomy, adaptive thresholding protocol, and code to facilitate future research in complex VAD scenarios.
>
---
#### [new 074] Overview of ExpertLifeCLEF 2018: how far automated identification systems are from the best experts?
- **分类: cs.CV**

- **简介: 该论文属于生物识别领域，旨在比较自动识别系统与人类专家的识别能力。通过LifeCLEF 2018挑战赛，评估了19个深度学习系统与9位植物学专家的表现，发现当前先进模型性能已接近顶尖人类专家水平。**

- **链接: [http://arxiv.org/pdf/2509.21419v1](http://arxiv.org/pdf/2509.21419v1)**

> **作者:** Herve Goeau; Pierre Bonnet; Alexis Joly
>
> **备注:** 11 pages, 2 figures, CLEF 2018 Conference and Labs of the Evaluation Forum, September 10 to 14, 2018, Avignon, France
>
> **摘要:** Automated identification of plants and animals has improved considerably in the last few years, in particular thanks to the recent advances in deep learning. The next big question is how far such automated systems are from the human expertise. Indeed, even the best experts are sometimes confused and/or disagree between each others when validating visual or audio observations of living organism. A picture actually contains only a partial information that is usually not sufficient to determine the right species with certainty. Quantifying this uncertainty and comparing it to the performance of automated systems is of high interest for both computer scientists and expert naturalists. The LifeCLEF 2018 ExpertCLEF challenge presented in this paper was designed to allow this comparison between human experts and automated systems. In total, 19 deep-learning systems implemented by 4 different research teams were evaluated with regard to 9 expert botanists of the French flora. The main outcome of this work is that the performance of state-of-the-art deep learning models is now close to the most advanced human expertise. This paper presents more precisely the resources and assessments of the challenge, summarizes the approaches and systems employed by the participating research groups, and provides an analysis of the main outcomes.
>
---
#### [new 075] PANICL: Mitigating Over-Reliance on Single Prompt in Visual In-Context Learning
- **分类: cs.CV**

- **简介: 该论文针对视觉上下文学习（VICL）中过度依赖单一示例的问题，提出PANICL框架。通过利用多个上下文对平滑分配分数，在无需额外训练的情况下提升模型性能与鲁棒性，适用于分割、检测等多种视觉任务。**

- **链接: [http://arxiv.org/pdf/2509.21926v1](http://arxiv.org/pdf/2509.21926v1)**

> **作者:** Jiahao Zhang; Bowen Wang; Hong Liu; Yuta Nakashima; Hajime Nagahara
>
> **备注:** 21 pages, 12 figures
>
> **摘要:** Visual In-Context Learning (VICL) uses input-output image pairs, referred to as in-context pairs (or examples), as prompts alongside query images to guide models in performing diverse vision tasks. However, VICL often suffers from over-reliance on a single in-context pair, which can lead to biased and unstable predictions. We introduce PAtch-based $k$-Nearest neighbor visual In-Context Learning (PANICL), a general training-free framework that mitigates this issue by leveraging multiple in-context pairs. PANICL smooths assignment scores across pairs, reducing bias without requiring additional training. Extensive experiments on a variety of tasks, including foreground segmentation, single object detection, colorization, multi-object segmentation, and keypoint detection, demonstrate consistent improvements over strong baselines. Moreover, PANICL exhibits strong robustness to domain shifts, including dataset-level shift (e.g., from COCO to Pascal) and label-space shift (e.g., FSS-1000), and generalizes well to other VICL models such as SegGPT, Painter, and LVM, highlighting its versatility and broad applicability.
>
---
#### [new 076] Temporal vs. Spatial: Comparing DINOv3 and V-JEPA2 Feature Representations for Video Action Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频动作分析任务，旨在比较DINOv3和V-JEPA2两种自监督学习架构的特征表示性能。通过在UCF Sports数据集上的多维度评估，揭示了空间与时间建模方法在分类、聚类和可靠性方面的优劣，为模型选择提供依据。**

- **链接: [http://arxiv.org/pdf/2509.21595v1](http://arxiv.org/pdf/2509.21595v1)**

> **作者:** Sai Varun Kodathala; Rakesh Vunnam
>
> **摘要:** This study presents a comprehensive comparative analysis of two prominent self-supervised learning architectures for video action recognition: DINOv3, which processes frames independently through spatial feature extraction, and V-JEPA2, which employs joint temporal modeling across video sequences. We evaluate both approaches on the UCF Sports dataset, examining feature quality through multiple dimensions including classification accuracy, clustering performance, intra-class consistency, and inter-class discrimination. Our analysis reveals fundamental architectural trade-offs: DINOv3 achieves superior clustering performance (Silhouette score: 0.31 vs 0.21) and demonstrates exceptional discrimination capability (6.16x separation ratio) particularly for pose-identifiable actions, while V-JEPA2 exhibits consistent reliability across all action types with significantly lower performance variance (0.094 vs 0.288). Through action-specific evaluation, we identify that DINOv3's spatial processing architecture excels at static pose recognition but shows degraded performance on motion-dependent actions, whereas V-JEPA2's temporal modeling provides balanced representation quality across diverse action categories. These findings contribute to the understanding of architectural design choices in video analysis systems and provide empirical guidance for selecting appropriate feature extraction methods based on task requirements and reliability constraints.
>
---
#### [new 077] MAJORScore: A Novel Metric for Evaluating Multimodal Relevance via Joint Representation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MAJORScore，用于评估多模态相关性。针对现有方法仅适用于两模态的问题，通过多模态联合表征首次实现N模态（N≥3）的相关性评分，实验表明其在一致性与不一致性模态上均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.21365v1](http://arxiv.org/pdf/2509.21365v1)**

> **作者:** Zhicheng Du; Qingyang Shi; Jiasheng Lu; Yingshan Liang; Xinyu Zhang; Yiran Wang; Peiwu Qin
>
> **摘要:** The multimodal relevance metric is usually borrowed from the embedding ability of pretrained contrastive learning models for bimodal data, which is used to evaluate the correlation between cross-modal data (e.g., CLIP). However, the commonly used evaluation metrics are only suitable for the associated analysis between two modalities, which greatly limits the evaluation of multimodal similarity. Herein, we propose MAJORScore, a brand-new evaluation metric for the relevance of multiple modalities (N modalities, N>=3) via multimodal joint representation for the first time. The ability of multimodal joint representation to integrate multiple modalities into the same latent space can accurately represent different modalities at one scale, providing support for fair relevance scoring. Extensive experiments have shown that MAJORScore increases by 26.03%-64.29% for consistent modality and decreases by 13.28%-20.54% for inconsistence compared to existing methods. MAJORScore serves as a more reliable metric for evaluating similarity on large-scale multimodal datasets and multimodal model performance evaluation.
>
---
#### [new 078] TDEdit: A Unified Diffusion Framework for Text-Drag Guided Image Manipulation
- **分类: cs.CV**

- **简介: 该论文提出TDEdit，一个结合文本和拖拽控制的统一扩散框架，用于图像编辑。旨在解决文本编辑缺乏空间控制、拖拽编辑缺乏纹理指导的问题，引入点云确定性拖拽和拖拽-文本引导去噪技术，实现高质量的联合编辑。**

- **链接: [http://arxiv.org/pdf/2509.21905v1](http://arxiv.org/pdf/2509.21905v1)**

> **作者:** Qihang Wang; Yaxiong Wang; Lechao Cheng; Zhun Zhong
>
> **摘要:** This paper explores image editing under the joint control of text and drag interactions. While recent advances in text-driven and drag-driven editing have achieved remarkable progress, they suffer from complementary limitations: text-driven methods excel in texture manipulation but lack precise spatial control, whereas drag-driven approaches primarily modify shape and structure without fine-grained texture guidance. To address these limitations, we propose a unified diffusion-based framework for joint drag-text image editing, integrating the strengths of both paradigms. Our framework introduces two key innovations: (1) Point-Cloud Deterministic Drag, which enhances latent-space layout control through 3D feature mapping, and (2) Drag-Text Guided Denoising, dynamically balancing the influence of drag and text conditions during denoising. Notably, our model supports flexible editing modes - operating with text-only, drag-only, or combined conditions - while maintaining strong performance in each setting. Extensive quantitative and qualitative experiments demonstrate that our method not only achieves high-fidelity joint editing but also matches or surpasses the performance of specialized text-only or drag-only approaches, establishing a versatile and generalizable solution for controllable image manipulation. Code will be made publicly available to reproduce all results presented in this work.
>
---
#### [new 079] Beyond Classification Accuracy: Neural-MedBench and the Need for Deeper Reasoning Benchmarks
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Neural-MedBench，一个针对神经医学的多模态临床推理基准，旨在超越分类准确率，评估模型在诊断、病灶识别和推理生成中的深度推理能力，揭示当前VLM在高风险医疗任务中的不足。**

- **链接: [http://arxiv.org/pdf/2509.22258v1](http://arxiv.org/pdf/2509.22258v1)**

> **作者:** Miao Jing; Mengting Jia; Junling Lin; Zhongxia Shen; Lijun Wang; Yuanyuan Peng; Huan Gao; Mingkun Xu; Shangyang Li
>
> **备注:** 23 pages, 12 figures
>
> **摘要:** Recent advances in vision-language models (VLMs) have achieved remarkable performance on standard medical benchmarks, yet their true clinical reasoning ability remains unclear. Existing datasets predominantly emphasize classification accuracy, creating an evaluation illusion in which models appear proficient while still failing at high-stakes diagnostic reasoning. We introduce Neural-MedBench, a compact yet reasoning-intensive benchmark specifically designed to probe the limits of multimodal clinical reasoning in neurology. Neural-MedBench integrates multi-sequence MRI scans, structured electronic health records, and clinical notes, and encompasses three core task families: differential diagnosis, lesion recognition, and rationale generation. To ensure reliable evaluation, we develop a hybrid scoring pipeline that combines LLM-based graders, clinician validation, and semantic similarity metrics. Through systematic evaluation of state-of-the-art VLMs, including GPT-4o, Claude-4, and MedGemma, we observe a sharp performance drop compared to conventional datasets. Error analysis shows that reasoning failures, rather than perceptual errors, dominate model shortcomings. Our findings highlight the necessity of a Two-Axis Evaluation Framework: breadth-oriented large datasets for statistical generalization, and depth-oriented, compact benchmarks such as Neural-MedBench for reasoning fidelity. We release Neural-MedBench at https://neuromedbench.github.io/ as an open and extensible diagnostic testbed, which guides the expansion of future benchmarks and enables rigorous yet cost-effective assessment of clinically trustworthy AI.
>
---
#### [new 080] HierLight-YOLO: A Hierarchical and Lightweight Object Detection Network for UAV Photography
- **分类: cs.CV**

- **简介: 该论文针对无人机摄影中小目标实时检测问题，提出HierLight-YOLO模型。通过HEPAN多尺度特征融合、IRDCB和LDown轻量化模块，提升小目标检测精度与效率，在VisDrone2019数据集上取得先进性能。**

- **链接: [http://arxiv.org/pdf/2509.22365v1](http://arxiv.org/pdf/2509.22365v1)**

> **作者:** Defan Chen; Yaohua Hu; Luchan Zhang
>
> **摘要:** The real-time detection of small objects in complex scenes, such as the unmanned aerial vehicle (UAV) photography captured by drones, has dual challenges of detecting small targets (<32 pixels) and maintaining real-time efficiency on resource-constrained platforms. While YOLO-series detectors have achieved remarkable success in real-time large object detection, they suffer from significantly higher false negative rates for drone-based detection where small objects dominate, compared to large object scenarios. This paper proposes HierLight-YOLO, a hierarchical feature fusion and lightweight model that enhances the real-time detection of small objects, based on the YOLOv8 architecture. We propose the Hierarchical Extended Path Aggregation Network (HEPAN), a multi-scale feature fusion method through hierarchical cross-level connections, enhancing the small object detection accuracy. HierLight-YOLO includes two innovative lightweight modules: Inverted Residual Depthwise Convolution Block (IRDCB) and Lightweight Downsample (LDown) module, which significantly reduce the model's parameters and computational complexity without sacrificing detection capabilities. Small object detection head is designed to further enhance spatial resolution and feature fusion to tackle the tiny object (4 pixels) detection. Comparison experiments and ablation studies on the VisDrone2019 benchmark demonstrate state-of-the-art performance of HierLight-YOLO.
>
---
#### [new 081] SemanticControl: A Training-Free Approach for Handling Loosely Aligned Visual Conditions in ControlNet
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对ControlNet在处理与文本提示不精确对齐的视觉条件时效果不佳的问题，提出了一种无需训练的方法SemanticControl。通过利用语义相关但结构不匹配的视觉条件，自适应调整视觉和文本引导，提升了生成效果。**

- **链接: [http://arxiv.org/pdf/2509.21938v1](http://arxiv.org/pdf/2509.21938v1)**

> **作者:** Woosung Joung; Daewon Chae; Jinkyu Kim
>
> **备注:** BMVC 2025
>
> **摘要:** ControlNet has enabled detailed spatial control in text-to-image diffusion models by incorporating additional visual conditions such as depth or edge maps. However, its effectiveness heavily depends on the availability of visual conditions that are precisely aligned with the generation goal specified by text prompt-a requirement that often fails in practice, especially for uncommon or imaginative scenes. For example, generating an image of a cat cooking in a specific pose may be infeasible due to the lack of suitable visual conditions. In contrast, structurally similar cues can often be found in more common settings-for instance, poses of humans cooking are widely available and can serve as rough visual guides. Unfortunately, existing ControlNet models struggle to use such loosely aligned visual conditions, often resulting in low text fidelity or visual artifacts. To address this limitation, we propose SemanticControl, a training-free method for effectively leveraging misaligned but semantically relevant visual conditions. Our approach adaptively suppresses the influence of the visual condition where it conflicts with the prompt, while strengthening guidance from the text. The key idea is to first run an auxiliary denoising process using a surrogate prompt aligned with the visual condition (e.g., "a human playing guitar" for a human pose condition) to extract informative attention masks, and then utilize these masks during the denoising of the actual target prompt (e.g., cat playing guitar). Experimental results demonstrate that our method improves performance under loosely aligned conditions across various conditions, including depth maps, edge maps, and human skeletons, outperforming existing baselines. Our code is available at https://mung3477.github.io/semantic-control.
>
---
#### [new 082] MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MinerU2.5，一种用于高效高分辨率文档解析的视觉-语言模型。它通过解耦全局布局分析与局部内容识别，实现计算效率与识别精度的平衡，在多个基准上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.22186v1](http://arxiv.org/pdf/2509.22186v1)**

> **作者:** Junbo Niu; Zheng Liu; Zhuangcheng Gu; Bin Wang; Linke Ouyang; Zhiyuan Zhao; Tao Chu; Tianyao He; Fan Wu; Qintong Zhang; Zhenjiang Jin; Guang Liang; Rui Zhang; Wenzheng Zhang; Yuan Qu; Zhifei Ren; Yuefeng Sun; Yuanhong Zheng; Dongsheng Ma; Zirui Tang; Boyu Niu; Ziyang Miao; Hejun Dong; Siyi Qian; Junyuan Zhang; Jingzhou Chen; Fangdong Wang; Xiaomeng Zhao; Liqun Wei; Wei Li; Shasha Wang; Ruiliang Xu; Yuanyuan Cao; Lu Chen; Qianqian Wu; Huaiyu Gu; Lindong Lu; Keming Wang; Dechen Lin; Guanlin Shen; Xuanhe Zhou; Linfeng Zhang; Yuhang Zang; Xiaoyi Dong; Jiaqi Wang; Bo Zhang; Lei Bai; Pei Chu; Weijia Li; Jiang Wu; Lijun Wu; Zhenxiang Li; Guangyu Wang; Zhongying Tu; Chao Xu; Kai Chen; Yu Qiao; Bowen Zhou; Dahua Lin; Wentao Zhang; Conghui He
>
> **备注:** Technical Report; GitHub Repo: https://github.com/opendatalab/MinerU; Hugging Face Model: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B; Hugging Face Demo: https://huggingface.co/spaces/opendatalab/MinerU
>
> **摘要:** We introduce MinerU2.5, a 1.2B-parameter document parsing vision-language model that achieves state-of-the-art recognition accuracy while maintaining exceptional computational efficiency. Our approach employs a coarse-to-fine, two-stage parsing strategy that decouples global layout analysis from local content recognition. In the first stage, the model performs efficient layout analysis on downsampled images to identify structural elements, circumventing the computational overhead of processing high-resolution inputs. In the second stage, guided by the global layout, it performs targeted content recognition on native-resolution crops extracted from the original image, preserving fine-grained details in dense text, complex formulas, and tables. To support this strategy, we developed a comprehensive data engine that generates diverse, large-scale training corpora for both pretraining and fine-tuning. Ultimately, MinerU2.5 demonstrates strong document parsing ability, achieving state-of-the-art performance on multiple benchmarks, surpassing both general-purpose and domain-specific models across various recognition tasks, while maintaining significantly lower computational overhead.
>
---
#### [new 083] Phrase-grounded Fact-checking for Automatically Generated Chest X-ray Reports
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种短语级事实核查模型，用于检测自动生成的胸部X光报告中的错误。针对视觉语言模型生成报告时存在的事实错误和幻觉问题，构建了一个合成数据集，并训练了一个多标签跨模态对比回归网络，实现了高精度的事实验证与定位。**

- **链接: [http://arxiv.org/pdf/2509.21356v1](http://arxiv.org/pdf/2509.21356v1)**

> **作者:** Razi Mahmood; Diego Machado-Reyes; Joy Wu; Parisa Kaviani; Ken C. L. Wong; Niharika D'Souza; Mannudeep Kalra; Ge Wang; Pingkun Yan; Tanveer Syeda-Mahmood
>
> **备注:** In proceedings MICCAI 2025
>
> **摘要:** With the emergence of large-scale vision language models (VLM), it is now possible to produce realistic-looking radiology reports for chest X-ray images. However, their clinical translation has been hampered by the factual errors and hallucinations in the produced descriptions during inference. In this paper, we present a novel phrase-grounded fact-checking model (FC model) that detects errors in findings and their indicated locations in automatically generated chest radiology reports. Specifically, we simulate the errors in reports through a large synthetic dataset derived by perturbing findings and their locations in ground truth reports to form real and fake findings-location pairs with images. A new multi-label cross-modal contrastive regression network is then trained on this dataset. We present results demonstrating the robustness of our method in terms of accuracy of finding veracity prediction and localization on multiple X-ray datasets. We also show its effectiveness for error detection in reports of SOTA report generators on multiple datasets achieving a concordance correlation coefficient of 0.997 with ground truth-based verification, thus pointing to its utility during clinical inference in radiology workflows.
>
---
#### [new 084] Joint graph entropy knowledge distillation for point cloud classification and robustness against corruptions
- **分类: cs.CV; cs.IR**

- **简介: 该论文针对3D点云分类任务，提出JGEKD方法，解决类别间相关性丢失及抗干扰问题。通过联合图熵知识蒸馏实现类间关系建模与鲁棒性提升，在多个数据集上取得良好效果。**

- **链接: [http://arxiv.org/pdf/2509.22150v1](http://arxiv.org/pdf/2509.22150v1)**

> **作者:** Zhiqiang Tian; Weigang Li; Junwei Hu; Chunhua Deng
>
> **摘要:** Classification tasks in 3D point clouds often assume that class events \replaced{are }{follow }independent and identically distributed (IID), although this assumption destroys the correlation between classes. This \replaced{study }{paper }proposes a classification strategy, \textbf{J}oint \textbf{G}raph \textbf{E}ntropy \textbf{K}nowledge \textbf{D}istillation (JGEKD), suitable for non-independent and identically distributed 3D point cloud data, \replaced{which }{the strategy } achieves knowledge transfer of class correlations through knowledge distillation by constructing a loss function based on joint graph entropy. First\deleted{ly}, we employ joint graphs to capture add{the }hidden relationships between classes\replaced{ and}{,} implement knowledge distillation to train our model by calculating the entropy of add{add }graph.\replaced{ Subsequently}{ Then}, to handle 3D point clouds \deleted{that is }invariant to spatial transformations, we construct \replaced{S}{s}iamese structures and develop two frameworks, self-knowledge distillation and teacher-knowledge distillation, to facilitate information transfer between different transformation forms of the same data. \replaced{In addition}{ Additionally}, we use the above framework to achieve knowledge transfer between point clouds and their corrupted forms, and increase the robustness against corruption of model. Extensive experiments on ScanObject, ModelNet40, ScanntV2\_cls and ModelNet-C demonstrate that the proposed strategy can achieve competitive results.
>
---
#### [new 085] MoWM: Mixture-of-World-Models for Embodied Planning via Latent-to-Pixel Feature Modulation
- **分类: cs.CV**

- **简介: 该论文提出MoWM，用于具身行动规划任务。针对视觉冗余和细节缺失的问题，融合潜空间与像素空间模型，利用潜空间引导像素特征提取，提升动作解码精度与泛化能力，在CALVIN基准上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.21797v1](http://arxiv.org/pdf/2509.21797v1)**

> **作者:** Yu Shang; Yangcheng Yu; Xin Zhang; Xin Jin; Haisheng Su; Wei Wu; Yong Li
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Embodied action planning is a core challenge in robotics, requiring models to generate precise actions from visual observations and language instructions. While video generation world models are promising, their reliance on pixel-level reconstruction often introduces visual redundancies that hinder action decoding and generalization. Latent world models offer a compact, motion-aware representation, but overlook the fine-grained details critical for precise manipulation. To overcome these limitations, we propose MoWM, a mixture-of-world-model framework that fuses representations from hybrid world models for embodied action planning. Our approach uses motion-aware representations from a latent model as a high-level prior, which guides the extraction of fine-grained visual features from the pixel space model. This design allows MoWM to highlight the informative visual details needed for action decoding. Extensive evaluations on the CALVIN benchmark demonstrate that our method achieves state-of-the-art task success rates and superior generalization. We also provide a comprehensive analysis of the strengths of each feature space, offering valuable insights for future research in embodied planning. The code is available at: https://github.com/tsinghua-fib-lab/MoWM.
>
---
#### [new 086] GPT-4 for Occlusion Order Recovery
- **分类: cs.CV; I.4.5**

- **简介: 该论文研究遮挡顺序恢复任务，旨在解决视觉模型难以准确判断物体遮挡关系的问题。提出利用预训练的GPT-4模型，通过设计提示生成遮挡预测，无需标注数据即可实现零样本推理，提升图像理解能力。**

- **链接: [http://arxiv.org/pdf/2509.22383v1](http://arxiv.org/pdf/2509.22383v1)**

> **作者:** Kaziwa Saleh; Zhyar Rzgar K Rostam; Sándor Szénási; Zoltán Vámossy
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Occlusion remains a significant challenge for current vision models to robustly interpret complex and dense real-world images and scenes. To address this limitation and to enable accurate prediction of the occlusion order relationship between objects, we propose leveraging the advanced capability of a pre-trained GPT-4 model to deduce the order. By providing a specifically designed prompt along with the input image, GPT-4 can analyze the image and generate order predictions. The response can then be parsed to construct an occlusion matrix which can be utilized in assisting with other occlusion handling tasks and image understanding. We report the results of evaluating the model on COCOA and InstaOrder datasets. The results show that by using semantic context, visual patterns, and commonsense knowledge, the model can produce more accurate order predictions. Unlike baseline methods, the model can reason about occlusion relationships in a zero-shot fashion, which requires no annotated training data and can easily be integrated into occlusion handling frameworks.
>
---
#### [new 087] QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models
- **分类: cs.CV**

- **简介: 该论文提出QuadGPT，首个端到端自回归四边形网格生成框架。针对现有方法依赖三角转四边形导致拓扑质量差的问题，创新性地引入统一的标记化方法和强化学习微调策略tDPO，显著提升了几何与拓扑质量。**

- **链接: [http://arxiv.org/pdf/2509.21420v1](http://arxiv.org/pdf/2509.21420v1)**

> **作者:** Jian Liu; Chunshi Wang; Song Guo; Haohan Weng; Zhen Zhou; Zhiqi Li; Jiaao Yu; Yiling Zhu; Jing Xu; Biwen Lei; Zhuo Chen; Chunchao Guo
>
> **摘要:** The generation of quadrilateral-dominant meshes is a cornerstone of professional 3D content creation. However, existing generative models generate quad meshes by first generating triangle meshes and then merging triangles into quadrilaterals with some specific rules, which typically produces quad meshes with poor topology. In this paper, we introduce QuadGPT, the first autoregressive framework for generating quadrilateral meshes in an end-to-end manner. QuadGPT formulates this as a sequence prediction paradigm, distinguished by two key innovations: a unified tokenization method to handle mixed topologies of triangles and quadrilaterals, and a specialized Reinforcement Learning fine-tuning method tDPO for better generation quality. Extensive experiments demonstrate that QuadGPT significantly surpasses previous triangle-to-quad conversion pipelines in both geometric accuracy and topological quality. Our work establishes a new benchmark for native quad-mesh generation and showcases the power of combining large-scale autoregressive models with topology-aware RL refinement for creating structured 3D assets.
>
---
#### [new 088] SPARK: Synergistic Policy And Reward Co-Evolving Framework
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出SPARK框架，用于大模型的强化学习优化。针对RLHF成本高、RLVR监督浪费的问题，SPARK通过共进化策略，利用生成式奖励模型复用数据，提升政策与奖励模型协同训练效率和效果。**

- **链接: [http://arxiv.org/pdf/2509.22624v1](http://arxiv.org/pdf/2509.22624v1)**

> **作者:** Ziyu Liu; Yuhang Zang; Shengyuan Ding; Yuhang Cao; Xiaoyi Dong; Haodong Duan; Dahua Lin; Jiaqi Wang
>
> **备注:** Project:https://github.com/InternLM/Spark
>
> **摘要:** Recent Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) increasingly use Reinforcement Learning (RL) for post-pretraining, such as RL with Verifiable Rewards (RLVR) for objective tasks and RL from Human Feedback (RLHF) for subjective tasks. However, RLHF incurs high costs and potential reward-policy mismatch due to reliance on human preferences, while RLVR still wastes supervision by discarding rollouts and correctness signals after each update. To address these challenges, we introduce the Synergistic Policy And Reward Co-Evolving Framework (SPARK), an efficient, on-policy, and stable method that builds on RLVR. Instead of discarding rollouts and correctness data, SPARK recycles this valuable information to simultaneously train the model itself as a generative reward model. This auxiliary training uses a mix of objectives, such as pointwise reward score, pairwise comparison, and evaluation conditioned on further-reflection responses, to teach the model to evaluate and improve its own responses. Our process eliminates the need for a separate reward model and costly human preference data. SPARK creates a positive co-evolving feedback loop: improved reward accuracy yields better policy gradients, which in turn produce higher-quality rollouts that further refine the reward model. Our unified framework supports test-time scaling via self-reflection without external reward models and their associated costs. We show that SPARK achieves significant performance gains on multiple LLM and LVLM models and multiple reasoning, reward models, and general benchmarks. For example, SPARK-VL-7B achieves an average 9.7% gain on 7 reasoning benchmarks, 12.1% on 2 reward benchmarks, and 1.5% on 8 general benchmarks over the baselines, demonstrating robustness and broad generalization.
>
---
#### [new 089] StableDub: Taming Diffusion Prior for Generalized and Efficient Visual Dubbing
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出StableDub，针对视觉配音任务中唇部习惯不符和遮挡处理不足的问题，设计了唇习感知建模与遮挡鲁棒生成框架，结合Mamba-Transformer结构提升效率，在音唇同步、视频质量等方面表现优异。**

- **链接: [http://arxiv.org/pdf/2509.21887v1](http://arxiv.org/pdf/2509.21887v1)**

> **作者:** Liyang Chen; Tianze Zhou; Xu He; Boshi Tang; Zhiyong Wu; Yang Huang; Yang Wu; Zhongqian Sun; Wei Yang; Helen Meng
>
> **摘要:** The visual dubbing task aims to generate mouth movements synchronized with the driving audio, which has seen significant progress in recent years. However, two critical deficiencies hinder their wide application: (1) Audio-only driving paradigms inadequately capture speaker-specific lip habits, which fail to generate lip movements similar to the target avatar; (2) Conventional blind-inpainting approaches frequently produce visual artifacts when handling obstructions (e.g., microphones, hands), limiting practical deployment. In this paper, we propose StableDub, a novel and concise framework integrating lip-habit-aware modeling with occlusion-robust synthesis. Specifically, building upon the Stable-Diffusion backbone, we develop a lip-habit-modulated mechanism that jointly models phonemic audio-visual synchronization and speaker-specific orofacial dynamics. To achieve plausible lip geometries and object appearances under occlusion, we introduce the occlusion-aware training strategy by explicitly exposing the occlusion objects to the inpainting process. By incorporating the proposed designs, the model eliminates the necessity for cost-intensive priors in previous methods, thereby exhibiting superior training efficiency on the computationally intensive diffusion-based backbone. To further optimize training efficiency from the perspective of model architecture, we introduce a hybrid Mamba-Transformer architecture, which demonstrates the enhanced applicability in low-resource research scenarios. Extensive experimental results demonstrate that StableDub achieves superior performance in lip habit resemblance and occlusion robustness. Our method also surpasses other methods in audio-lip sync, video quality, and resolution consistency. We expand the applicability of visual dubbing methods from comprehensive aspects, and demo videos can be found at https://stabledub.github.io.
>
---
#### [new 090] Group Critical-token Policy Optimization for Autoregressive Image Generation
- **分类: cs.CV**

- **简介: 该论文针对自回归图像生成中的强化学习优化问题，提出GCPO方法。通过识别关键图像token（基于因果依赖、熵梯度和多样性），并进行动态加权优化，提升了RLVR的训练效果。实验表明，在使用30% token的情况下优于全token方法。**

- **链接: [http://arxiv.org/pdf/2509.22485v1](http://arxiv.org/pdf/2509.22485v1)**

> **作者:** Guohui Zhang; Hu Yu; Xiaoxiao Ma; JingHao Zhang; Yaning Pan; Mingde Yao; Jie Xiao; Linjiang Huang; Feng Zhao
>
> **备注:** Code is available at https://github.com/zghhui/GCPO
>
> **摘要:** Recent studies have extended Reinforcement Learning with Verifiable Rewards (RLVR) to autoregressive (AR) visual generation and achieved promising progress. However, existing methods typically apply uniform optimization across all image tokens, while the varying contributions of different image tokens for RLVR's training remain unexplored. In fact, the key obstacle lies in how to identify more critical image tokens during AR generation and implement effective token-wise optimization for them. To tackle this challenge, we propose $\textbf{G}$roup $\textbf{C}$ritical-token $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{GCPO}$), which facilitates effective policy optimization on critical tokens. We identify the critical tokens in RLVR-based AR generation from three perspectives, specifically: $\textbf{(1)}$ Causal dependency: early tokens fundamentally determine the later tokens and final image effect due to unidirectional dependency; $\textbf{(2)}$ Entropy-induced spatial structure: tokens with high entropy gradients correspond to image structure and bridges distinct visual regions; $\textbf{(3)}$ RLVR-focused token diversity: tokens with low visual similarity across a group of sampled images contribute to richer token-level diversity. For these identified critical tokens, we further introduce a dynamic token-wise advantage weight to encourage exploration, based on confidence divergence between the policy model and reference model. By leveraging 30\% of the image tokens, GCPO achieves better performance than GRPO with full tokens. Extensive experiments on multiple text-to-image benchmarks for both AR models and unified multimodal models demonstrate the effectiveness of GCPO for AR visual generation.
>
---
#### [new 091] Gender Stereotypes in Professional Roles Among Saudis: An Analytical Study of AI-Generated Images Using Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于社会与AI交叉研究任务，旨在探究AI生成图像是否强化沙特职业中的性别刻板印象和文化偏差。研究分析了三种AI模型生成的1006张沙特职业图像，评估其在性别、服饰、背景等方面的偏见，揭示当前AI输出存在显著性别失衡和文化不准确性。**

- **链接: [http://arxiv.org/pdf/2509.21466v1](http://arxiv.org/pdf/2509.21466v1)**

> **作者:** Khaloud S. AlKhalifah; Malak Mashaabi; Hend Al-Khalifa
>
> **摘要:** This study investigates the extent to which contemporary Text-to-Image artificial intelligence (AI) models perpetuate gender stereotypes and cultural inaccuracies when generating depictions of professionals in Saudi Arabia. We analyzed 1,006 images produced by ImageFX, DALL-E V3, and Grok for 56 diverse Saudi professions using neutral prompts. Two trained Saudi annotators evaluated each image on five dimensions: perceived gender, clothing and appearance, background and setting, activities and interactions, and age. A third senior researcher adjudicated whenever the two primary raters disagreed, yielding 10,100 individual judgements. The results reveal a strong gender imbalance, with ImageFX outputs being 85\% male, Grok 86.6\% male, and DALL-E V3 96\% male, indicating that DALL-E V3 exhibited the strongest overall gender stereotyping. This imbalance was most evident in leadership and technical roles. Moreover, cultural inaccuracies in clothing, settings, and depicted activities were frequently observed across all three models. Counter-stereotypical images often arise from cultural misinterpretations rather than genuinely progressive portrayals. We conclude that current models mirror societal biases embedded in their training data, generated by humans, offering only a limited reflection of the Saudi labour market's gender dynamics and cultural nuances. These findings underscore the urgent need for more diverse training data, fairer algorithms, and culturally sensitive evaluation frameworks to ensure equitable and authentic visual outputs.
>
---
#### [new 092] Exposing Hallucinations To Suppress Them: VLMs Representation Editing With Generative Anchors
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（VLMs）在视觉-语言任务中易产生幻觉的问题，提出一种无需训练的自监督方法。通过文本到图像生成构建正负锚点，编辑解码器隐藏状态以抑制幻觉，有效减少对象、属性和关系层面的不实描述，且保持描述丰富性。**

- **链接: [http://arxiv.org/pdf/2509.21997v1](http://arxiv.org/pdf/2509.21997v1)**

> **作者:** Youxu Shi; Suorong Yang; Dong Liu
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable success across diverse vision-language tasks, yet they remain highly susceptible to hallucinations, producing content that is fluent but inconsistent with visual evidence. Such hallucinations, spanning objects, attributes, and relations, persist even in larger models, while existing mitigation approaches often require additional finetuning, handcrafted priors, or trade-offs that compromise informativeness and scalability. To address this limitation, we propose a training-free, self-supervised method for hallucination mitigation. Our approach introduces a novel hallucination amplification mechanism: a caption is projected into the visual space via a text-to-image model to reveal implicit hallucination signals, serving as a negative anchor, while the original image provides a positive anchor. Leveraging these dual anchors, we edit decoder hidden states by pulling representations toward faithful semantics and pushing them away from hallucination directions. This correction requires no human priors or additional training costs, ensuring both effectiveness and efficiency. Extensive experiments across multiple benchmarks show that our method significantly reduces hallucinations at the object, attribute, and relation levels while largely preserving recall and caption richness, e.g., achieving a hallucination reduction by over 5% using LLaVA-v1.5-7B on CHAIR. Furthermore, results on diverse architectures, including LLaVA-NEXT-7B, Cambrian-8B, and InstructBLIP-7B, validate strong cross-architecture generalization. More importantly, when applied to hallucination-free captions, our method introduces almost no side effects, underscoring its robustness and practical plug-and-play applicability. The implementation will be publicly available.
>
---
#### [new 093] SAEmnesia: Erasing Concepts in Diffusion Models with Sparse Autoencoders
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SAEmnesia，一种用于文本到图像扩散模型的概念遗忘方法。针对概念表征分散的问题，通过监督稀疏自编码器训练实现概念-神经元的一一映射，提高可解释性和遗忘效率，在UnlearnCanvas基准上取得显著提升。**

- **链接: [http://arxiv.org/pdf/2509.21379v1](http://arxiv.org/pdf/2509.21379v1)**

> **作者:** Enrico Cassano; Riccardo Renzulli; Marco Nurisso; Mirko Zaffaroni; Alan Perotti; Marco Grangetto
>
> **摘要:** Effective concept unlearning in text-to-image diffusion models requires precise localization of concept representations within the model's latent space. While sparse autoencoders successfully reduce neuron polysemanticity (i.e., multiple concepts per neuron) compared to the original network, individual concept representations can still be distributed across multiple latent features, requiring extensive search procedures for concept unlearning. We introduce SAEmnesia, a supervised sparse autoencoder training method that promotes one-to-one concept-neuron mappings through systematic concept labeling, mitigating feature splitting and promoting feature centralization. Our approach learns specialized neurons with significantly stronger concept associations compared to unsupervised baselines. The only computational overhead introduced by SAEmnesia is limited to cross-entropy computation during training. At inference time, this interpretable representation reduces hyperparameter search by 96.67% with respect to current approaches. On the UnlearnCanvas benchmark, SAEmnesia achieves a 9.22% improvement over the state-of-the-art. In sequential unlearning tasks, we demonstrate superior scalability with a 28.4% improvement in unlearning accuracy for 9-object removal.
>
---
#### [new 094] A Comprehensive Evaluation of Transformer-Based Question Answering Models and RAG-Enhanced Design
- **分类: cs.CV**

- **简介: 该论文聚焦多跳问答任务，旨在解决Transformer模型在多文档推理中的不足。研究评估了多种检索策略，提出融合稠密嵌入与词法重叠的混合方法，并优化检索流程。实验表明，该方法在HotpotQA数据集上显著提升了准确率和效率。**

- **链接: [http://arxiv.org/pdf/2509.21845v1](http://arxiv.org/pdf/2509.21845v1)**

> **作者:** Zichen Zhang; Kunlong Zhang; Hongwei Ruan; Yiming Luo
>
> **摘要:** Transformer-based models have advanced the field of question answering, but multi-hop reasoning, where answers require combining evidence across multiple passages, remains difficult. This paper presents a comprehensive evaluation of retrieval strategies for multi-hop question answering within a retrieval-augmented generation framework. We compare cosine similarity, maximal marginal relevance, and a hybrid method that integrates dense embeddings with lexical overlap and re-ranking. To further improve retrieval, we adapt the EfficientRAG pipeline for query optimization, introducing token labeling and iterative refinement while maintaining efficiency. Experiments on the HotpotQA dataset show that the hybrid approach substantially outperforms baseline methods, achieving a relative improvement of 50 percent in exact match and 47 percent in F1 score compared to cosine similarity. Error analysis reveals that hybrid retrieval improves entity recall and evidence complementarity, while remaining limited in handling distractors and temporal reasoning. Overall, the results suggest that hybrid retrieval-augmented generation provides a practical zero-shot solution for multi-hop question answering, balancing accuracy, efficiency, and interpretability.
>
---
#### [new 095] SpikeMatch: Semi-Supervised Learning with Temporal Dynamics of Spiking Neural Networks
- **分类: cs.CV**

- **简介: 该论文提出SpikeMatch，首个利用脉冲神经网络（SNN）时序动态的半监督学习框架。通过共训练和伪标签策略，解决SNN在有限标签下特征学习不足的问题，在多个基准上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.22581v1](http://arxiv.org/pdf/2509.22581v1)**

> **作者:** Jini Yang; Beomseok Oh; Seungryong Kim; Sunok Kim
>
> **摘要:** Spiking neural networks (SNNs) have recently been attracting significant attention for their biological plausibility and energy efficiency, but semi-supervised learning (SSL) methods for SNN-based models remain underexplored compared to those for artificial neural networks (ANNs). In this paper, we introduce SpikeMatch, the first SSL framework for SNNs that leverages the temporal dynamics through the leakage factor of SNNs for diverse pseudo-labeling within a co-training framework. By utilizing agreement among multiple predictions from a single SNN, SpikeMatch generates reliable pseudo-labels from weakly-augmented unlabeled samples to train on strongly-augmented ones, effectively mitigating confirmation bias by capturing discriminative features with limited labels. Experiments show that SpikeMatch outperforms existing SSL methods adapted to SNN backbones across various standard benchmarks.
>
---
#### [new 096] Rate-Distortion Optimized Communication for Collaborative Perception
- **分类: cs.CV**

- **简介: 该论文研究多智能体协作感知中的通信优化问题，旨在减少通信开销的同时保持任务性能。提出基于率失真理论的RDcomm框架，通过任务熵离散编码和互信息驱动的消息选择，实现高效通信。实验在3D目标检测和BEV分割任务中验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2509.21994v1](http://arxiv.org/pdf/2509.21994v1)**

> **作者:** Genjia Liu; Anning Hu; Yue Hu; Wenjun Zhang; Siheng Chen
>
> **摘要:** Collaborative perception emphasizes enhancing environmental understanding by enabling multiple agents to share visual information with limited bandwidth resources. While prior work has explored the empirical trade-off between task performance and communication volume, a significant gap remains in the theoretical foundation. To fill this gap, we draw on information theory and introduce a pragmatic rate-distortion theory for multi-agent collaboration, specifically formulated to analyze performance-communication trade-off in goal-oriented multi-agent systems. This theory concretizes two key conditions for designing optimal communication strategies: supplying pragmatically relevant information and transmitting redundancy-less messages. Guided by these two conditions, we propose RDcomm, a communication-efficient collaborative perception framework that introduces two key innovations: i) task entropy discrete coding, which assigns features with task-relevant codeword-lengths to maximize the efficiency in supplying pragmatic information; ii) mutual-information-driven message selection, which utilizes mutual information neural estimation to approach the optimal redundancy-less condition. Experiments on 3D object detection and BEV segmentation demonstrate that RDcomm achieves state-of-the-art accuracy on DAIR-V2X and OPV2V, while reducing communication volume by up to 108 times. The code will be released.
>
---
#### [new 097] Jailbreaking on Text-to-Video Models via Scene Splitting Strategy
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究文本到视频模型的安全漏洞，提出SceneSplit方法，通过分割有害叙事为多个安全场景，绕过安全过滤机制，成功生成有害视频。实验表明其攻击成功率显著高于基线。**

- **链接: [http://arxiv.org/pdf/2509.22292v1](http://arxiv.org/pdf/2509.22292v1)**

> **作者:** Wonjun Lee; Haon Park; Doehyeon Lee; Bumsub Ham; Suhyun Kim
>
> **摘要:** Along with the rapid advancement of numerous Text-to-Video (T2V) models, growing concerns have emerged regarding their safety risks. While recent studies have explored vulnerabilities in models like LLMs, VLMs, and Text-to-Image (T2I) models through jailbreak attacks, T2V models remain largely unexplored, leaving a significant safety gap. To address this gap, we introduce SceneSplit, a novel black-box jailbreak method that works by fragmenting a harmful narrative into multiple scenes, each individually benign. This approach manipulates the generative output space, the abstract set of all potential video outputs for a given prompt, using the combination of scenes as a powerful constraint to guide the final outcome. While each scene individually corresponds to a wide and safe space where most outcomes are benign, their sequential combination collectively restricts this space, narrowing it to an unsafe region and significantly increasing the likelihood of generating a harmful video. This core mechanism is further enhanced through iterative scene manipulation, which bypasses the safety filter within this constrained unsafe region. Additionally, a strategy library that reuses successful attack patterns further improves the attack's overall effectiveness and robustness. To validate our method, we evaluate SceneSplit across 11 safety categories on T2V models. Our results show that it achieves a high average Attack Success Rate (ASR) of 77.2% on Luma Ray2, 84.1% on Hailuo, and 78.2% on Veo2, significantly outperforming the existing baseline. Through this work, we demonstrate that current T2V safety mechanisms are vulnerable to attacks that exploit narrative structure, providing new insights for understanding and improving the safety of T2V models.
>
---
#### [new 098] A Data-driven Typology of Vision Models from Integrated Representational Metrics
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于多表示相似性度量的数据驱动方法，用于对视觉模型进行分类。任务是分析不同模型的表征特性，解决如何区分模型家族共享与独特计算策略的问题。工作包括融合多种相似性指标，揭示模型结构和训练目标对表征的影响。**

- **链接: [http://arxiv.org/pdf/2509.21628v1](http://arxiv.org/pdf/2509.21628v1)**

> **作者:** Jialin Wu; Shreya Saha; Yiqing Bo; Meenakshi Khosla
>
> **摘要:** Large vision models differ widely in architecture and training paradigm, yet we lack principled methods to determine which aspects of their representations are shared across families and which reflect distinctive computational strategies. We leverage a suite of representational similarity metrics, each capturing a different facet-geometry, unit tuning, or linear decodability-and assess family separability using multiple complementary measures. Metrics preserving geometry or tuning (e.g., RSA, Soft Matching) yield strong family discrimination, whereas flexible mappings such as Linear Predictivity show weaker separation. These findings indicate that geometry and tuning carry family-specific signatures, while linearly decodable information is more broadly shared. To integrate these complementary facets, we adapt Similarity Network Fusion (SNF), a method inspired by multi-omics integration. SNF achieves substantially sharper family separation than any individual metric and produces robust composite signatures. Clustering of the fused similarity matrix recovers both expected and surprising patterns: supervised ResNets and ViTs form distinct clusters, yet all self-supervised models group together across architectural boundaries. Hybrid architectures (ConvNeXt, Swin) cluster with masked autoencoders, suggesting convergence between architectural modernization and reconstruction-based training. This biology-inspired framework provides a principled typology of vision models, showing that emergent computational strategies-shaped jointly by architecture and training objective-define representational structure beyond surface design categories.
>
---
#### [new 099] KV-Efficient VLA: A Method of Speed up Vision Language Model with RNN-Gated Chunked KV Cache
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出KV-Efficient VLA，针对视觉-语言-动作模型在长序列推理中的KV缓存效率问题，通过分块KV缓存和递归门控机制实现高效压缩，提升推理速度并减少内存占用。**

- **链接: [http://arxiv.org/pdf/2509.21354v1](http://arxiv.org/pdf/2509.21354v1)**

> **作者:** Wanshun Xu; Long Zhuang
>
> **摘要:** Vision-Language-Action (VLA) models promise unified robotic perception and control, yet their scalability is constrained by the quadratic cost of attention and the unbounded growth of key-value (KV) memory during long-horizon inference. While recent methods improve generalization through scaling backbone architectures, they often neglect the inference inefficiencies critical to real-time deployment. In this work, we present KV-Efficient VLA, a model-agnostic memory compression framework that addresses these limitations by introducing a lightweight, training-friendly mechanism to selectively retain high-utility context. Our method partitions the KV cache into fixed size chunks and employs a recurrent gating module to summarize and filter historical context according to learned utility scores. This design preserves recent fine-grained detail while aggressively pruning stale, low-relevance memory, all while maintaining causality. Theoretically, KV-Efficient VLA yields up to 1.21x inference speedup and 36% KV memory reduction, with minimal impact on task success. Our method integrates seamlessly into existing autoregressive and hybrid VLA stacks, enabling scalable inference without modifying training pipelines or downstream control logic.
>
---
#### [new 100] X-CoT: Explainable Text-to-Video Retrieval via LLM-based Chain-of-Thought Reasoning
- **分类: cs.CV**

- **简介: 该论文提出X-CoT，一种基于大语言模型思维链（CoT）的可解释文本到视频检索框架。针对传统嵌入模型在数据质量和可解释性上的不足，X-CoT通过引入语义推理和配对比较，提升了检索性能并提供详细解释，用于评估模型与数据质量。**

- **链接: [http://arxiv.org/pdf/2509.21559v1](http://arxiv.org/pdf/2509.21559v1)**

> **作者:** Prasanna Reddy Pulakurthi; Jiamian Wang; Majid Rabbani; Sohail Dianat; Raghuveer Rao; Zhiqiang Tao
>
> **备注:** 12 pages, 7 figures. Accepted at EMNLP 2025 (Main Conference)
>
> **摘要:** Prevalent text-to-video retrieval systems mainly adopt embedding models for feature extraction and compute cosine similarities for ranking. However, this design presents two limitations. Low-quality text-video data pairs could compromise the retrieval, yet are hard to identify and examine. Cosine similarity alone provides no explanation for the ranking results, limiting the interpretability. We ask that can we interpret the ranking results, so as to assess the retrieval models and examine the text-video data? This work proposes X-CoT, an explainable retrieval framework upon LLM CoT reasoning in place of the embedding model-based similarity ranking. We first expand the existing benchmarks with additional video annotations to support semantic understanding and reduce data bias. We also devise a retrieval CoT consisting of pairwise comparison steps, yielding detailed reasoning and complete ranking. X-CoT empirically improves the retrieval performance and produces detailed rationales. It also facilitates the model behavior and data quality analysis. Code and data are available at: https://github.com/PrasannaPulakurthi/X-CoT.
>
---
#### [new 101] Reasoning-Enhanced Domain-Adaptive Pretraining of Multimodal Large Language Models for Short Video Content Moderation
- **分类: cs.CV**

- **简介: 该论文针对短视频内容审核任务，旨在解决现有方法泛化能力差、依赖大量标注数据的问题。提出一种增强推理能力的多模态大语言模型预训练方法，引入Caption、VQA和CoT三个任务，提升模型对视频细节、规则理解和推理能力。**

- **链接: [http://arxiv.org/pdf/2509.21486v1](http://arxiv.org/pdf/2509.21486v1)**

> **作者:** Zixuan Wang; Yu Sun; Hongwei Wang; Baoyu Jing; Xiang Shen; Xin Dong; Zhuolin Hao; Hongyu Xiong; Yang Song
>
> **摘要:** Short video platforms are evolving rapidly, making the identification of inappropriate content increasingly critical. Existing approaches typically train separate and small classification models for each type of issue, which requires extensive human-labeled data and lacks cross-issue generalization. We propose a reasoning-enhanced multimodal large language model (MLLM) pretraining paradigm for unified inappropriate content detection. To address the distribution gap between short video content and the original pretraining data of MLLMs, as well as the complex issue definitions, we introduce three targeted pretraining tasks: (1) \textit{Caption}, to enhance the MLLM's perception of video details; (2) \textit{Visual Question Answering (VQA)}, to deepen the MLLM's understanding of issue definitions and annotation guidelines; (3) \textit{Chain-of-Thought (CoT)}, to enhance the MLLM's reasoning capability. Experimental results show that our pretraining approach significantly improves the MLLM's performance in both zero-shot and supervised fine-tuning (SFT) settings. In addition, our pretrained model demonstrates strong generalization capabilities to emergent, previously unseen issues.
>
---
#### [new 102] On the Status of Foundation Models for SAR Imagery
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究了基础模型在SAR图像目标识别中的可行性，旨在解决SAR领域标注数据少、特征提取难的问题。通过自监督微调现有视觉基础模型（如DINOv2），提出AFRL-DINOv2s，显著优于当前最优模型SARATR-X，并分析了不同模型结构与下游任务的适配性。**

- **链接: [http://arxiv.org/pdf/2509.21722v1](http://arxiv.org/pdf/2509.21722v1)**

> **作者:** Nathan Inkawhich
>
> **摘要:** In this work we investigate the viability of foundational AI/ML models for Synthetic Aperture Radar (SAR) object recognition tasks. We are inspired by the tremendous progress being made in the wider community, particularly in the natural image domain where frontier labs are training huge models on web-scale datasets with unprecedented computing budgets. It has become clear that these models, often trained with Self-Supervised Learning (SSL), will transform how we develop AI/ML solutions for object recognition tasks - they can be adapted downstream with very limited labeled data, they are more robust to many forms of distribution shift, and their features are highly transferable out-of-the-box. For these reasons and more, we are motivated to apply this technology to the SAR domain. In our experiments we first run tests with today's most powerful visual foundational models, including DINOv2, DINOv3 and PE-Core and observe their shortcomings at extracting semantically-interesting discriminative SAR target features when used off-the-shelf. We then show that Self-Supervised finetuning of publicly available SSL models with SAR data is a viable path forward by training several AFRL-DINOv2s and setting a new state-of-the-art for SAR foundation models, significantly outperforming today's best SAR-domain model SARATR-X. Our experiments further analyze the performance trade-off of using different backbones with different downstream task-adaptation recipes, and we monitor each model's ability to overcome challenges within the downstream environments (e.g., extended operating conditions and low amounts of labeled data). We hope this work will inform and inspire future SAR foundation model builders, because despite our positive results, we still have a long way to go.
>
---
#### [new 103] Assessing the Alignment of Popular CNNs to the Brain for Valence Appraisal
- **分类: cs.CV**

- **简介: 该论文研究卷积神经网络（CNN）在情感评估任务中与人脑的对应关系。通过行为和fMRI数据，分析发现CNN难以反映高阶认知处理，并提出Object2Brain框架，结合GradCAM和目标检测，探讨不同物体类别对模型-人脑相关性的影响。**

- **链接: [http://arxiv.org/pdf/2509.21384v1](http://arxiv.org/pdf/2509.21384v1)**

> **作者:** Laurent Mertens; Elahe' Yargholi; Laura Van Hove; Hans Op de Beeck; Jan Van den Stock; Joost Vennekens
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Convolutional Neural Networks (CNNs) are a popular type of computer model that have proven their worth in many computer vision tasks. Moreover, they form an interesting study object for the field of psychology, with shown correspondences between the workings of CNNs and the human brain. However, these correspondences have so far mostly been studied in the context of general visual perception. In contrast, this paper explores to what extent this correspondence also holds for a more complex brain process, namely social cognition. To this end, we assess the alignment between popular CNN architectures and both human behavioral and fMRI data for image valence appraisal through a correlation analysis. We show that for this task CNNs struggle to go beyond simple visual processing, and do not seem to reflect higher-order brain processing. Furthermore, we present Object2Brain, a novel framework that combines GradCAM and object detection at the CNN-filter level with the aforementioned correlation analysis to study the influence of different object classes on the CNN-to-human correlations. Despite similar correlation trends, different CNN architectures are shown to display different object class sensitivities.
>
---
#### [new 104] CircuitSense: A Hierarchical Circuit System Benchmark Bridging Visual Comprehension and Symbolic Reasoning in Engineering Design Process
- **分类: cs.CV**

- **简介: 该论文提出CircuitSense，一个评估电路理解的基准，聚焦视觉解析与符号推理在工程设计中的结合。通过8006+问题和分层生成方法，揭示当前多模态模型在从视觉到数学推理任务中的显著性能差距，强调符号推理对工程能力的关键作用。**

- **链接: [http://arxiv.org/pdf/2509.22339v1](http://arxiv.org/pdf/2509.22339v1)**

> **作者:** Arman Akbari; Jian Gao; Yifei Zou; Mei Yang; Jinru Duan; Dmitrii Torbunov; Yanzhi Wang; Yihui Ren; Xuan Zhang
>
> **摘要:** Engineering design operates through hierarchical abstraction from system specifications to component implementations, requiring visual understanding coupled with mathematical reasoning at each level. While Multi-modal Large Language Models (MLLMs) excel at natural image tasks, their ability to extract mathematical models from technical diagrams remains unexplored. We present \textbf{CircuitSense}, a comprehensive benchmark evaluating circuit understanding across this hierarchy through 8,006+ problems spanning component-level schematics to system-level block diagrams. Our benchmark uniquely examines the complete engineering workflow: Perception, Analysis, and Design, with a particular emphasis on the critical but underexplored capability of deriving symbolic equations from visual inputs. We introduce a hierarchical synthetic generation pipeline consisting of a grid-based schematic generator and a block diagram generator with auto-derived symbolic equation labels. Comprehensive evaluation of six state-of-the-art MLLMs, including both closed-source and open-source models, reveals fundamental limitations in visual-to-mathematical reasoning. Closed-source models achieve over 85\% accuracy on perception tasks involving component recognition and topology identification, yet their performance on symbolic derivation and analytical reasoning falls below 19\%, exposing a critical gap between visual parsing and symbolic reasoning. Models with stronger symbolic reasoning capabilities consistently achieve higher design task accuracy, confirming the fundamental role of mathematical understanding in circuit synthesis and establishing symbolic reasoning as the key metric for engineering competence.
>
---
#### [new 105] SSVIF: Self-Supervised Segmentation-Oriented Visible and Infrared Image Fusion
- **分类: cs.CV**

- **简介: 该论文提出SSVIF，用于可见光与红外图像融合任务。针对应用导向方法依赖标注数据的问题，设计自监督框架，通过特征与像素级分割一致性学习语义特征，无需标注即可有效提升分割性能。**

- **链接: [http://arxiv.org/pdf/2509.22450v1](http://arxiv.org/pdf/2509.22450v1)**

> **作者:** Zixian Zhao; Xingchen Zhang
>
> **摘要:** Visible and infrared image fusion (VIF) has gained significant attention in recent years due to its wide application in tasks such as scene segmentation and object detection. VIF methods can be broadly classified into traditional VIF methods and application-oriented VIF methods. Traditional methods focus solely on improving the quality of fused images, while application-oriented VIF methods additionally consider the performance of downstream tasks on fused images by introducing task-specific loss terms during training. However, compared to traditional methods, application-oriented VIF methods require datasets labeled for downstream tasks (e.g., semantic segmentation or object detection), making data acquisition labor-intensive and time-consuming. To address this issue, we propose a self-supervised training framework for segmentation-oriented VIF methods (SSVIF). Leveraging the consistency between feature-level fusion-based segmentation and pixel-level fusion-based segmentation, we introduce a novel self-supervised task-cross-segmentation consistency-that enables the fusion model to learn high-level semantic features without the supervision of segmentation labels. Additionally, we design a two-stage training strategy and a dynamic weight adjustment method for effective joint learning within our self-supervised framework. Extensive experiments on public datasets demonstrate the effectiveness of our proposed SSVIF. Remarkably, although trained only on unlabeled visible-infrared image pairs, our SSVIF outperforms traditional VIF methods and rivals supervised segmentation-oriented ones. Our code will be released upon acceptance.
>
---
#### [new 106] $γ$-Quant: Towards Learnable Quantization for Low-bit Pattern Recognition
- **分类: cs.CV**

- **简介: 该论文针对低带宽和能耗受限场景下的模式识别任务，提出$\gamma$-Quant方法，通过学习非线性量化策略，在仅使用4位数据时即可达到与12位原始数据相当的性能，应用于目标检测和人体活动识别。**

- **链接: [http://arxiv.org/pdf/2509.22448v1](http://arxiv.org/pdf/2509.22448v1)**

> **作者:** Mishal Fatima; Shashank Agnihotri; Marius Bock; Kanchana Vaishnavi Gandikota; Kristof Van Laerhoven; Michael Moeller; Margret Keuper
>
> **备注:** Accepted at DAGM GCPR 2025
>
> **摘要:** Most pattern recognition models are developed on pre-proce\-ssed data. In computer vision, for instance, RGB images processed through image signal processing (ISP) pipelines designed to cater to human perception are the most frequent input to image analysis networks. However, many modern vision tasks operate without a human in the loop, raising the question of whether such pre-processing is optimal for automated analysis. Similarly, human activity recognition (HAR) on body-worn sensor data commonly takes normalized floating-point data arising from a high-bit analog-to-digital converter (ADC) as an input, despite such an approach being highly inefficient in terms of data transmission, significantly affecting the battery life of wearable devices. In this work, we target low-bandwidth and energy-constrained settings where sensors are limited to low-bit-depth capture. We propose $\gamma$-Quant, i.e.~the task-specific learning of a non-linear quantization for pattern recognition. We exemplify our approach on raw-image object detection as well as HAR of wearable data, and demonstrate that raw data with a learnable quantization using as few as 4-bits can perform on par with the use of raw 12-bit data. All code to reproduce our experiments is publicly available via https://github.com/Mishalfatima/Gamma-Quant
>
---
#### [new 107] MS-YOLO: Infrared Object Detection for Edge Deployment via MobileNetV4 and SlideLoss
- **分类: cs.CV**

- **简介: 该论文针对红外图像中目标检测任务，旨在解决计算开销大、类别不平衡等问题。提出了MS-YOLO模型，采用MobileNetV4降低计算量，并引入SlideLoss提升检测精度，适用于边缘设备部署。**

- **链接: [http://arxiv.org/pdf/2509.21696v1](http://arxiv.org/pdf/2509.21696v1)**

> **作者:** Jiali Zhang; Thomas S. White; Haoliang Zhang; Wenqing Hu; Donald C. Wunsch II; Jian Liu
>
> **备注:** Accepted by the International Joint Conference on Neural Networks (IJCNN) 2025. Keywords: Infrared Object Detection, MobileNetV4, SlideLoss, YOLO Model
>
> **摘要:** Infrared imaging has emerged as a robust solution for urban object detection under low-light and adverse weather conditions, offering significant advantages over traditional visible-light cameras. However, challenges such as class imbalance, thermal noise, and computational constraints can significantly hinder model performance in practical settings. To address these issues, we evaluate multiple YOLO variants on the FLIR ADAS V2 dataset, ultimately selecting YOLOv8 as our baseline due to its balanced accuracy and efficiency. Building on this foundation, we present \texttt{MS-YOLO} (\textbf{M}obileNetv4 and \textbf{S}lideLoss based on YOLO), which replaces YOLOv8's CSPDarknet backbone with the more efficient MobileNetV4, reducing computational overhead by \textbf{1.5%} while sustaining high accuracy. In addition, we introduce \emph{SlideLoss}, a novel loss function that dynamically emphasizes under-represented and occluded samples, boosting precision without sacrificing recall. Experiments on the FLIR ADAS V2 benchmark show that \texttt{MS-YOLO} attains competitive mAP and superior precision while operating at only \textbf{6.7 GFLOPs}. These results demonstrate that \texttt{MS-YOLO} effectively addresses the dual challenge of maintaining high detection quality while minimizing computational costs, making it well-suited for real-time edge deployment in urban environments.
>
---
#### [new 108] Learning Human-Perceived Fakeness in AI-Generated Videos via Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦于AI生成视频中人类感知的虚假痕迹检测任务，旨在解决如何识别并定位视频中的伪造线索。为此，作者构建了DeeptraceReward数据集，并训练多模态语言模型作为奖励模型，以模仿人类判断与定位能力，提升视频生成的可信度评估。**

- **链接: [http://arxiv.org/pdf/2509.22646v1](http://arxiv.org/pdf/2509.22646v1)**

> **作者:** Xingyu Fu; Siyi Liu; Yinuo Xu; Pan Lu; Guangqiuse Hu; Tianbo Yang; Taran Anantasagar; Christopher Shen; Yikai Mao; Yuanzhe Liu; Keyush Shah; Chung Un Lee; Yejin Choi; James Zou; Dan Roth; Chris Callison-Burch
>
> **备注:** Project Page: https://deeptracereward.github.io/
>
> **摘要:** Can humans identify AI-generated (fake) videos and provide grounded reasons? While video generation models have advanced rapidly, a critical dimension -- whether humans can detect deepfake traces within a generated video, i.e., spatiotemporal grounded visual artifacts that reveal a video as machine generated -- has been largely overlooked. We introduce DeeptraceReward, the first fine-grained, spatially- and temporally- aware benchmark that annotates human-perceived fake traces for video generation reward. The dataset comprises 4.3K detailed annotations across 3.3K high-quality generated videos. Each annotation provides a natural-language explanation, pinpoints a bounding-box region containing the perceived trace, and marks precise onset and offset timestamps. We consolidate these annotations into 9 major categories of deepfake traces that lead humans to identify a video as AI-generated, and train multimodal language models (LMs) as reward models to mimic human judgments and localizations. On DeeptraceReward, our 7B reward model outperforms GPT-5 by 34.7% on average across fake clue identification, grounding, and explanation. Interestingly, we observe a consistent difficulty gradient: binary fake v.s. real classification is substantially easier than fine-grained deepfake trace detection; within the latter, performance degrades from natural language explanations (easiest), to spatial grounding, to temporal labeling (hardest). By foregrounding human-perceived deepfake traces, DeeptraceReward provides a rigorous testbed and training signal for socially aware and trustworthy video generation.
>
---
#### [new 109] UniVid: Unifying Vision Tasks with Pre-trained Video Generation Models
- **分类: cs.CV**

- **简介: 该论文提出UniVid，旨在解决视觉任务统一与扩展性问题。通过微调视频扩散模型，无需任务特定调整即可处理多种图像和视频任务，实现了跨模态和跨数据源的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.21760v1](http://arxiv.org/pdf/2509.21760v1)**

> **作者:** Lan Chen; Yuchao Gu; Qi Mao
>
> **摘要:** Large language models, trained on extensive corpora, successfully unify diverse linguistic tasks within a single generative framework. Inspired by this, recent works like Large Vision Model (LVM) extend this paradigm to vision by organizing tasks into sequential visual sentences, where visual prompts serve as the context to guide outputs. However, such modeling requires task-specific pre-training across modalities and sources, which is costly and limits scalability to unseen tasks. Given that pre-trained video generation models inherently capture temporal sequence dependencies, we explore a more unified and scalable alternative: can a pre-trained video generation model adapt to diverse image and video tasks? To answer this, we propose UniVid, a framework that fine-tunes a video diffusion transformer to handle various vision tasks without task-specific modifications. Tasks are represented as visual sentences, where the context sequence defines both the task and the expected output modality. We evaluate the generalization of UniVid from two perspectives: (1) cross-modal inference with contexts composed of both images and videos, extending beyond LVM's uni-modal setting; (2) cross-source tasks from natural to annotated data, without multi-source pre-training. Despite being trained solely on natural video data, UniVid generalizes well in both settings. Notably, understanding and generation tasks can easily switch by simply reversing the visual sentence order in this paradigm. These findings highlight the potential of pre-trained video generation models to serve as a scalable and unified foundation for vision modeling. Our code will be released at https://github.com/CUC-MIPG/UniVid.
>
---
#### [new 110] DyME: Dynamic Multi-Concept Erasure in Diffusion Models with Bi-Level Orthogonal LoRA Adaptation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出DyME，一种面向文本到图像扩散模型的动态多概念擦除框架。针对现有方法在多冲突概念擦除中的不足，DyME采用模块化LoRA适配器与双层正交约束，实现按需组合、高效精准的多概念抑制，提升擦除效果并减少副作用。**

- **链接: [http://arxiv.org/pdf/2509.21433v1](http://arxiv.org/pdf/2509.21433v1)**

> **作者:** Jiaqi Liu; Lan Zhang; Xiaoyong Yuan
>
> **摘要:** Text-to-image diffusion models (DMs) inadvertently reproduce copyrighted styles and protected visual concepts, raising legal and ethical concerns. Concept erasure has emerged as a safeguard, aiming to selectively suppress such concepts through fine-tuning. However, existing methods do not scale to practical settings where providers must erase multiple and possibly conflicting concepts. The core bottleneck is their reliance on static erasure: a single checkpoint is fine-tuned to remove all target concepts, regardless of the actual erasure needs at inference. This rigid design mismatches real-world usage, where requests vary per generation, leading to degraded erasure success and reduced fidelity for non-target content. We propose DyME, an on-demand erasure framework that trains lightweight, concept-specific LoRA adapters and dynamically composes only those needed at inference. This modular design enables flexible multi-concept erasure, but naive composition causes interference among adapters, especially when many or semantically related concepts are suppressed. To overcome this, we introduce bi-level orthogonality constraints at both the feature and parameter levels, disentangling representation shifts and enforcing orthogonal adapter subspaces. We further develop ErasureBench-H, a new hierarchical benchmark with brand-series-character structure, enabling principled evaluation across semantic granularities and erasure set sizes. Experiments on ErasureBench-H and standard datasets (e.g., CIFAR-100, Imagenette) demonstrate that DyME consistently outperforms state-of-the-art baselines, achieving higher multi-concept erasure fidelity with minimal collateral degradation.
>
---
#### [new 111] Pedestrian Attribute Recognition via Hierarchical Cross-Modality HyperGraph Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对行人属性识别任务，旨在解决现有方法未能充分利用属性知识和上下文信息的问题。提出构建多模态知识图谱，并引入跨模态超图学习框架，以增强视觉与属性间的关系建模，提升识别准确性。**

- **链接: [http://arxiv.org/pdf/2509.22331v1](http://arxiv.org/pdf/2509.22331v1)**

> **作者:** Xiao Wang; Shujuan Wu; Xiaoxia Cheng; Changwei Bi; Jin Tang; Bin Luo
>
> **备注:** The First Work that Exploits Multi-modal Knowledge Graph for Pedestrian Attribute Recognition
>
> **摘要:** Current Pedestrian Attribute Recognition (PAR) algorithms typically focus on mapping visual features to semantic labels or attempt to enhance learning by fusing visual and attribute information. However, these methods fail to fully exploit attribute knowledge and contextual information for more accurate recognition. Although recent works have started to consider using attribute text as additional input to enhance the association between visual and semantic information, these methods are still in their infancy. To address the above challenges, this paper proposes the construction of a multi-modal knowledge graph, which is utilized to mine the relationships between local visual features and text, as well as the relationships between attributes and extensive visual context samples. Specifically, we propose an effective multi-modal knowledge graph construction method that fully considers the relationships among attributes and the relationships between attributes and vision tokens. To effectively model these relationships, this paper introduces a knowledge graph-guided cross-modal hypergraph learning framework to enhance the standard pedestrian attribute recognition framework. Comprehensive experiments on multiple PAR benchmark datasets have thoroughly demonstrated the effectiveness of our proposed knowledge graph for the PAR task, establishing a strong foundation for knowledge-guided pedestrian attribute recognition. The source code of this paper will be released on https://github.com/Event-AHU/OpenPAR
>
---
#### [new 112] Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation
- **分类: cs.CV**

- **简介: 该论文提出一种用于主题驱动生成中检测视觉不一致性的方法，通过解耦预训练扩散模型的视觉和语义特征，构建对比架构，并设计新的VSM度量，实现对不一致性的同时量化与定位。**

- **链接: [http://arxiv.org/pdf/2509.21989v1](http://arxiv.org/pdf/2509.21989v1)**

> **作者:** Abdelrahman Eldesokey; Aleksandar Cvejic; Bernard Ghanem; Peter Wonka
>
> **备注:** NeurIPS 2025 (Spotlight). Project Page: https://abdo-eldesokey.github.io/mind-the-glitch/
>
> **摘要:** We propose a novel approach for disentangling visual and semantic features from the backbones of pre-trained diffusion models, enabling visual correspondence in a manner analogous to the well-established semantic correspondence. While diffusion model backbones are known to encode semantically rich features, they must also contain visual features to support their image synthesis capabilities. However, isolating these visual features is challenging due to the absence of annotated datasets. To address this, we introduce an automated pipeline that constructs image pairs with annotated semantic and visual correspondences based on existing subject-driven image generation datasets, and design a contrastive architecture to separate the two feature types. Leveraging the disentangled representations, we propose a new metric, Visual Semantic Matching (VSM), that quantifies visual inconsistencies in subject-driven image generation. Empirical results show that our approach outperforms global feature-based metrics such as CLIP, DINO, and vision--language models in quantifying visual inconsistencies while also enabling spatial localization of inconsistent regions. To our knowledge, this is the first method that supports both quantification and localization of inconsistencies in subject-driven generation, offering a valuable tool for advancing this task. Project Page:https://abdo-eldesokey.github.io/mind-the-glitch/
>
---
#### [new 113] Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究了基于2D高斯溅射（2DGS）的视觉-语言对齐方法，旨在解决传统RGB图像传输能耗高和序列长度过长的问题。论文提出了高效的2DGS表示与优化流程，并适配CLIP模型，实现高效压缩与语义对齐，验证了其在零样本任务中的可行性。**

- **链接: [http://arxiv.org/pdf/2509.22615v1](http://arxiv.org/pdf/2509.22615v1)**

> **作者:** Yasmine Omri; Connor Ding; Tsachy Weissman; Thierry Tambe
>
> **摘要:** Modern vision language pipelines are driven by RGB vision encoders trained on massive image text corpora. While these pipelines have enabled impressive zero shot capabilities and strong transfer across tasks, they still inherit two structural inefficiencies from the pixel domain: (i) transmitting dense RGB images from edge devices to the cloud is energy intensive and costly, and (ii) patch based tokenization explodes sequence length, stressing attention budgets and context limits. We explore 2D Gaussian Splatting (2DGS) as an alternative visual substrate for alignment: a compact, spatially adaptive representation that parameterizes images by a set of colored anisotropic Gaussians. We develop a scalable 2DGS pipeline with structured initialization, luminance aware pruning, and batched CUDA kernels, achieving over 90x faster fitting and about 97% GPU utilization compared to prior implementations. We further adapt contrastive language image pretraining (CLIP) to 2DGS by reusing a frozen RGB-based transformer backbone with a lightweight splat aware input stem and a perceiver resampler, training only about 7% of the total parameters. On large DataComp subsets, GS encoders yield meaningful zero shot ImageNet-1K performance while compressing inputs 3 to 20x relative to pixels. While accuracy currently trails RGB encoders, our results establish 2DGS as a viable multimodal substrate, pinpoint architectural bottlenecks, and open a path toward representations that are both semantically powerful and transmission efficient for edge cloud learning.
>
---
#### [new 114] UISim: An Interactive Image-Based UI Simulator for Dynamic Mobile Environments
- **分类: cs.CV; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文提出UISim，一种基于图像的移动UI模拟器，用于动态环境下的UI测试与AI代理训练。针对现有方法依赖物理设备或静态分析的问题，UISim通过两阶段方法预测并生成视觉一致的UI状态，提升UI开发效率和AI训练效果。**

- **链接: [http://arxiv.org/pdf/2509.21733v1](http://arxiv.org/pdf/2509.21733v1)**

> **作者:** Jiannan Xiang; Yun Zhu; Lei Shu; Maria Wang; Lijun Yu; Gabriel Barcik; James Lyon; Srinivas Sunkara; Jindong Chen
>
> **摘要:** Developing and testing user interfaces (UIs) and training AI agents to interact with them are challenging due to the dynamic and diverse nature of real-world mobile environments. Existing methods often rely on cumbersome physical devices or limited static analysis of screenshots, which hinders scalable testing and the development of intelligent UI agents. We introduce UISim, a novel image-based UI simulator that offers a dynamic and interactive platform for exploring mobile phone environments purely from screen images. Our system employs a two-stage method: given an initial phone screen image and a user action, it first predicts the abstract layout of the next UI state, then synthesizes a new, visually consistent image based on this predicted layout. This approach enables the realistic simulation of UI transitions. UISim provides immediate practical benefits for UI testing, rapid prototyping, and synthetic data generation. Furthermore, its interactive capabilities pave the way for advanced applications, such as UI navigation task planning for AI agents. Our experimental results show that UISim outperforms end-to-end UI generation baselines in generating realistic and coherent subsequent UI states, highlighting its fidelity and potential to streamline UI development and enhance AI agent training.
>
---
#### [new 115] Bézier Meets Diffusion: Robust Generation Across Domains for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割中的跨域适应问题，提出“Bézier Meets Diffusion”框架。通过贝塞尔曲线风格迁移和条件扩散模型，结合不确定性引导的训练方法，有效减少域间差异，生成高质量标注目标域图像，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2509.22476v1](http://arxiv.org/pdf/2509.22476v1)**

> **作者:** Chen Li; Meilong Xu; Xiaoling Hu; Weimin Lyu; Chao Chen
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Training robust learning algorithms across different medical imaging modalities is challenging due to the large domain gap. Unsupervised domain adaptation (UDA) mitigates this problem by using annotated images from the source domain and unlabeled images from the target domain to train the deep models. Existing approaches often rely on GAN-based style transfer, but these methods struggle to capture cross-domain mappings in regions with high variability. In this paper, we propose a unified framework, B\'ezier Meets Diffusion, for cross-domain image generation. First, we introduce a B\'ezier-curve-based style transfer strategy that effectively reduces the domain gap between source and target domains. The transferred source images enable the training of a more robust segmentation model across domains. Thereafter, using pseudo-labels generated by this segmentation model on the target domain, we train a conditional diffusion model (CDM) to synthesize high-quality, labeled target-domain images. To mitigate the impact of noisy pseudo-labels, we further develop an uncertainty-guided score matching method that improves the robustness of CDM training. Extensive experiments on public datasets demonstrate that our approach generates realistic labeled images, significantly augmenting the target domain and improving segmentation performance.
>
---
#### [new 116] The LongiMam model for improved breast cancer risk prediction using longitudinal mammograms
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出LongiMam模型，用于改进乳腺癌风险预测。任务是基于纵向乳腺X光图像的深度学习建模，解决现有模型未能充分利用历史影像数据的问题。工作包括设计结合卷积和循环神经网络的端到端模型，并在真实筛查数据中验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.21383v1](http://arxiv.org/pdf/2509.21383v1)**

> **作者:** Manel Rakez; Thomas Louis; Julien Guillaumin; Foucauld Chamming's; Pierre Fillard; Brice Amadeo; Virginie Rondeau
>
> **摘要:** Risk-adapted breast cancer screening requires robust models that leverage longitudinal imaging data. Most current deep learning models use single or limited prior mammograms and lack adaptation for real-world settings marked by imbalanced outcome distribution and heterogeneous follow-up. We developed LongiMam, an end-to-end deep learning model that integrates both current and up to four prior mammograms. LongiMam combines a convolutional and a recurrent neural network to capture spatial and temporal patterns predictive of breast cancer. The model was trained and evaluated using a large, population-based screening dataset with disproportionate case-to-control ratio typical of clinical screening. Across several scenarios that varied in the number and composition of prior exams, LongiMam consistently improved prediction when prior mammograms were included. The addition of prior and current visits outperformed single-visit models, while priors alone performed less well, highlighting the importance of combining historical and recent information. Subgroup analyses confirmed the model's efficacy across key risk groups, including women with dense breasts and those aged 55 years or older. Moreover, the model performed best in women with observed changes in mammographic density over time. These findings demonstrate that longitudinal modeling enhances breast cancer prediction and support the use of repeated mammograms to refine risk stratification in screening programs. LongiMam is publicly available as open-source software.
>
---
#### [new 117] ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ERGO，针对视觉-语言模型处理高分辨率图像计算成本高的问题，设计了一种“粗到细”的推理流程，通过多模态上下文指导观察区域选择，在保证精度的同时显著提升效率。**

- **链接: [http://arxiv.org/pdf/2509.21991v1](http://arxiv.org/pdf/2509.21991v1)**

> **作者:** Jewon Lee; Wooksu Shin; Seungmin Yang; Ki-Ung Song; DongUk Lim; Jaeyeon Kim; Tae-Ho Kim; Bo-Kyeong Kim
>
> **摘要:** Efficient processing of high-resolution images is crucial for real-world vision-language applications. However, existing Large Vision-Language Models (LVLMs) incur substantial computational overhead due to the large number of vision tokens. With the advent of "thinking with images" models, reasoning now extends beyond text to the visual domain. This capability motivates our two-stage "coarse-to-fine" reasoning pipeline: first, a downsampled image is analyzed to identify task-relevant regions; then, only these regions are cropped at full resolution and processed in a subsequent reasoning stage. This approach reduces computational cost while preserving fine-grained visual details where necessary. A major challenge lies in inferring which regions are truly relevant to a given query. Recent related methods often fail in the first stage after input-image downsampling, due to perception-driven reasoning, where clear visual information is required for effective reasoning. To address this issue, we propose ERGO (Efficient Reasoning & Guided Observation) that performs reasoning-driven perception-leveraging multimodal context to determine where to focus. Our model can account for perceptual uncertainty, expanding the cropped region to cover visually ambiguous areas for answering questions. To this end, we develop simple yet effective reward components in a reinforcement learning framework for coarse-to-fine perception. Across multiple datasets, our approach delivers higher accuracy than the original model and competitive methods, with greater efficiency. For instance, ERGO surpasses Qwen2.5-VL-7B on the V* benchmark by 4.7 points while using only 23% of the vision tokens, achieving a 3x inference speedup. The code and models can be found at: https://github.com/nota-github/ERGO.
>
---
#### [new 118] From Bias to Balance: Exploring and Mitigating Spatial Bias in LVLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究了大视觉-语言模型（LVLMs）中的空间偏差问题，发现语言模型组件的位置嵌入设计不平衡导致视觉信息整合不均。提出了一种简单有效的平衡位置分配机制BaPA，无需微调即可提升模型的空间鲁棒性和多模态性能。**

- **链接: [http://arxiv.org/pdf/2509.21984v1](http://arxiv.org/pdf/2509.21984v1)**

> **作者:** Yingjie Zhu; Xuefeng Bai; Kehai Chen; Yang Xiang; Weili Guan; Jun Yu; Min Zhang
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved remarkable success across a wide range of multimodal tasks, yet their robustness to spatial variations remains insufficiently understood. In this work, we present a systematic study of the spatial bias of LVLMs, focusing on how models respond when identical key visual information is placed at different locations within an image. Through a carefully designed probing dataset, we demonstrate that current LVLMs often produce inconsistent outputs under such spatial shifts, revealing a fundamental limitation in their spatial-semantic understanding. Further analysis shows that this phenomenon originates not from the vision encoder, which reliably perceives and interprets visual content across positions, but from the unbalanced design of position embeddings in the language model component. In particular, the widely adopted position embedding strategies, such as RoPE, introduce imbalance during cross-modal interaction, leading image tokens at different positions to exert unequal influence on semantic understanding. To mitigate this issue, we introduce Balanced Position Assignment (BaPA), a simple yet effective mechanism that assigns identical position embeddings to all image tokens, promoting a more balanced integration of visual information. Extensive experiments show that BaPA enhances the spatial robustness of LVLMs without retraining and further boosts their performance across diverse multimodal benchmarks when combined with lightweight fine-tuning. Further analysis of information flow reveals that BaPA yields balanced attention, enabling more holistic visual understanding.
>
---
#### [new 119] KG-SAM: Injecting Anatomical Knowledge into Segment Anything Models via Conditional Random Fields
- **分类: cs.CV**

- **简介: 该论文提出KG-SAM，用于医学图像分割任务。针对SAM在医学应用中的边界模糊、解剖关系建模不足等问题，引入知识图谱、CRF和不确定性模块，提升分割精度与可靠性，在前列腺和腹部影像中取得显著效果。**

- **链接: [http://arxiv.org/pdf/2509.21750v1](http://arxiv.org/pdf/2509.21750v1)**

> **作者:** Yu Li; Da Chang; Xi Xiao
>
> **摘要:** While the Segment Anything Model (SAM) has achieved remarkable success in image segmentation, its direct application to medical imaging remains hindered by fundamental challenges, including ambiguous boundaries, insufficient modeling of anatomical relationships, and the absence of uncertainty quantification. To address these limitations, we introduce KG-SAM, a knowledge-guided framework that synergistically integrates anatomical priors with boundary refinement and uncertainty estimation. Specifically, KG-SAM incorporates (i) a medical knowledge graph to encode fine-grained anatomical relationships, (ii) an energy-based Conditional Random Field (CRF) to enforce anatomically consistent predictions, and (iii) an uncertainty-aware fusion module to enhance reliability in high-stakes clinical scenarios. Extensive experiments across multi-center medical datasets demonstrate the effectiveness of our approach: KG-SAM achieves an average Dice score of 82.69% on prostate segmentation and delivers substantial gains in abdominal segmentation, reaching 78.05% on MRI and 79.68% on CT. These results establish KG-SAM as a robust and generalizable framework for advancing medical image segmentation.
>
---
#### [new 120] mmHSense: Multi-Modal and Distributed mmWave ISAC Datasets for Human Sensing
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出了mmHSense，一套用于毫米波ISAC系统的人体感知开源数据集，支持手势识别、身份识别等任务。工作包括搭建测试平台、设计实验，并验证了参数高效微调方法在降低计算复杂度上的有效性。**

- **链接: [http://arxiv.org/pdf/2509.21396v1](http://arxiv.org/pdf/2509.21396v1)**

> **作者:** Nabeel Nisar Bhat; Maksim Karnaukh; Stein Vandenbroeke; Wouter Lemoine; Jakob Struye; Jesus Omar Lacruz; Siddhartha Kumar; Mohammad Hossein Moghaddam; Joerg Widmer; Rafael Berkvens; Jeroen Famaey
>
> **摘要:** This article presents mmHSense, a set of open labeled mmWave datasets to support human sensing research within Integrated Sensing and Communication (ISAC) systems. The datasets can be used to explore mmWave ISAC for various end applications such as gesture recognition, person identification, pose estimation, and localization. Moreover, the datasets can be used to develop and advance signal processing and deep learning research on mmWave ISAC. This article describes the testbed, experimental settings, and signal features for each dataset. Furthermore, the utility of the datasets is demonstrated through validation on a specific downstream task. In addition, we demonstrate the use of parameter-efficient fine-tuning to adapt ISAC models to different tasks, significantly reducing computational complexity while maintaining performance on prior tasks.
>
---
#### [new 121] RefAM: Attention Magnets for Zero-Shot Referral Segmentation
- **分类: cs.CV**

- **简介: 该论文提出RefAM，一种无需训练的零样本指代表达分割方法。通过利用扩散模型中的注意力分数，结合注意力重分布策略和全局注意力处理，实现了更准确的图像和视频分割，解决了传统方法依赖微调或复杂架构的问题。**

- **链接: [http://arxiv.org/pdf/2509.22650v1](http://arxiv.org/pdf/2509.22650v1)**

> **作者:** Anna Kukleva; Enis Simsar; Alessio Tonioni; Muhammad Ferjad Naeem; Federico Tombari; Jan Eric Lenssen; Bernt Schiele
>
> **备注:** Project Page: https://refam-diffusion.github.io/
>
> **摘要:** Most existing approaches to referring segmentation achieve strong performance only through fine-tuning or by composing multiple pre-trained models, often at the cost of additional training and architectural modifications. Meanwhile, large-scale generative diffusion models encode rich semantic information, making them attractive as general-purpose feature extractors. In this work, we introduce a new method that directly exploits features, attention scores, from diffusion transformers for downstream tasks, requiring neither architectural modifications nor additional training. To systematically evaluate these features, we extend benchmarks with vision-language grounding tasks spanning both images and videos. Our key insight is that stop words act as attention magnets: they accumulate surplus attention and can be filtered to reduce noise. Moreover, we identify global attention sinks (GAS) emerging in deeper layers and show that they can be safely suppressed or redirected onto auxiliary tokens, leading to sharper and more accurate grounding maps. We further propose an attention redistribution strategy, where appended stop words partition background activations into smaller clusters, yielding sharper and more localized heatmaps. Building on these findings, we develop RefAM, a simple training-free grounding framework that combines cross-attention maps, GAS handling, and redistribution. Across zero-shot referring image and video segmentation benchmarks, our approach consistently outperforms prior methods, establishing a new state of the art without fine-tuning or additional components.
>
---
#### [new 122] Explaining multimodal LLMs via intra-modal token interactions
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态大语言模型（MLLMs）的可解释性问题，提出通过模态内交互提升解释质量。视觉分支采用多尺度解释聚合（MSEA），文本分支引入激活排序相关性（ARC），有效缓解局部依赖与虚假激活问题，提升解释的准确性与连贯性。**

- **链接: [http://arxiv.org/pdf/2509.22415v1](http://arxiv.org/pdf/2509.22415v1)**

> **作者:** Jiawei Liang; Ruoyu Chen; Xianghao Jiao; Siyuan Liang; Shiming Liu; Qunli Zhang; Zheng Hu; Xiaochun Cao
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable success across diverse vision-language tasks, yet their internal decision-making mechanisms remain insufficiently understood. Existing interpretability research has primarily focused on cross-modal attribution, identifying which image regions the model attends to during output generation. However, these approaches often overlook intra-modal dependencies. In the visual modality, attributing importance to isolated image patches ignores spatial context due to limited receptive fields, resulting in fragmented and noisy explanations. In the textual modality, reliance on preceding tokens introduces spurious activations. Failing to effectively mitigate these interference compromises attribution fidelity. To address these limitations, we propose enhancing interpretability by leveraging intra-modal interaction. For the visual branch, we introduce \textit{Multi-Scale Explanation Aggregation} (MSEA), which aggregates attributions over multi-scale inputs to dynamically adjust receptive fields, producing more holistic and spatially coherent visual explanations. For the textual branch, we propose \textit{Activation Ranking Correlation} (ARC), which measures the relevance of contextual tokens to the current token via alignment of their top-$k$ prediction rankings. ARC leverages this relevance to suppress spurious activations from irrelevant contexts while preserving semantically coherent ones. Extensive experiments across state-of-the-art MLLMs and benchmark datasets demonstrate that our approach consistently outperforms existing interpretability methods, yielding more faithful and fine-grained explanations of model behavior.
>
---
#### [new 123] Benchmarking and Mitigate Psychological Sycophancy in Medical Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦医疗视觉问答任务，旨在解决医学视觉语言模型（VLMs）在面对用户社会暗示时表现出的趋同性偏差问题。研究构建了一个临床基准数据集，评估不同模型的趋同性，并提出VIPER方法过滤非证据信息，提升模型基于证据的回答能力。**

- **链接: [http://arxiv.org/pdf/2509.21979v1](http://arxiv.org/pdf/2509.21979v1)**

> **作者:** Zikun Guo; Xinyue Xu; Pei Xiang; Shu Yang; Xin Han; Di Wang; Lijie Hu
>
> **备注:** 19figures, 37pages
>
> **摘要:** Vision language models(VLMs) are increasingly integrated into clinical workflows, but they often exhibit sycophantic behavior prioritizing alignment with user phrasing social cues or perceived authority over evidence based reasoning. This study evaluate clinical sycophancy in medical visual question answering through a novel clinically grounded benchmark. We propose a medical sycophancy dataset construct from PathVQA, SLAKE, and VQA-RAD stratified by different type organ system and modality. Using psychologically motivated pressure templates including various sycophancy. In our adversarial experiments on various VLMs, we found that these models are generally vulnerable, exhibiting significant variations in the occurrence of adversarial responses, with weak correlations to the model accuracy or size. Imitation and expert provided corrections were found to be the most effective triggers, suggesting that the models possess a bias mechanism independent of visual evidence. To address this, we propose Visual Information Purification for Evidence based Response (VIPER) a lightweight mitigation strategy that filters non evidentiary content for example social pressures and then generates constrained evidence first answers. This framework reduces sycophancy by an average amount outperforming baselines while maintaining interpretability. Our benchmark analysis and mitigation framework lay the groundwork for robust deployment of medical VLMs in real world clinician interactions emphasizing the need for evidence anchored defenses.
>
---
#### [new 124] NIFTY: a Non-Local Image Flow Matching for Texture Synthesis
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出NIFTY，用于基于示例的纹理合成任务。针对传统块匹配方法存在的初始化差和视觉伪影问题，结合扩散模型与非局部块匹配技术，构建无需训练神经网络的非参数流匹配模型，有效提升了合成效果。**

- **链接: [http://arxiv.org/pdf/2509.22318v1](http://arxiv.org/pdf/2509.22318v1)**

> **作者:** Pierrick Chatillon; Julien Rabin; David Tschumperlé
>
> **摘要:** This paper addresses the problem of exemplar-based texture synthesis. We introduce NIFTY, a hybrid framework that combines recent insights on diffusion models trained with convolutional neural networks, and classical patch-based texture optimization techniques. NIFTY is a non-parametric flow-matching model built on non-local patch matching, which avoids the need for neural network training while alleviating common shortcomings of patch-based methods, such as poor initialization or visual artifacts. Experimental results demonstrate the effectiveness of the proposed approach compared to representative methods from the literature. Code is available at https://github.com/PierrickCh/Nifty.git
>
---
#### [new 125] DiTraj: training-free trajectory control for video diffusion transformer
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DiTraj，一种无需训练的视频扩散Transformer轨迹控制方法。针对现有方法依赖大量训练资源或不适用于DiT的问题，通过前景-背景分离引导和改进时空位置编码，提升轨迹可控性与视频质量。**

- **链接: [http://arxiv.org/pdf/2509.21839v1](http://arxiv.org/pdf/2509.21839v1)**

> **作者:** Cheng Lei; Jiayu Zhang; Yue Ma; Xinyu Wang; Long Chen; Liang Tang; Yiqiang Yan; Fei Su; Zhicheng Zhao
>
> **摘要:** Diffusion Transformers (DiT)-based video generation models with 3D full attention exhibit strong generative capabilities. Trajectory control represents a user-friendly task in the field of controllable video generation. However, existing methods either require substantial training resources or are specifically designed for U-Net, do not take advantage of the superior performance of DiT. To address these issues, we propose DiTraj, a simple but effective training-free framework for trajectory control in text-to-video generation, tailored for DiT. Specifically, first, to inject the object's trajectory, we propose foreground-background separation guidance: we use the Large Language Model (LLM) to convert user-provided prompts into foreground and background prompts, which respectively guide the generation of foreground and background regions in the video. Then, we analyze 3D full attention and explore the tight correlation between inter-token attention scores and position embedding. Based on this, we propose inter-frame Spatial-Temporal Decoupled 3D-RoPE (STD-RoPE). By modifying only foreground tokens' position embedding, STD-RoPE eliminates their cross-frame spatial discrepancies, strengthening cross-frame attention among them and thus enhancing trajectory control. Additionally, we achieve 3D-aware trajectory control by regulating the density of position embedding. Extensive experiments demonstrate that our method outperforms previous methods in both video quality and trajectory controllability.
>
---
#### [new 126] MultiCrafter: High-Fidelity Multi-Subject Generation via Spatially Disentangled Attention and Identity-Aware Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文提出MultiCrafter，用于高质量多主体图像生成。针对现有方法存在的属性泄露和与人类偏好不一致的问题，通过空间解耦注意力机制和混合专家架构提升主体保真度，并设计在线强化学习框架对齐人类偏好。**

- **链接: [http://arxiv.org/pdf/2509.21953v1](http://arxiv.org/pdf/2509.21953v1)**

> **作者:** Tao Wu; Yibo Jiang; Yehao Lu; Zhizhong Wang; Zeyi Huang; Zequn Qin; Xi Li
>
> **备注:** Project Page: https://wutao-cs.github.io/MultiCrafter/
>
> **摘要:** Multi-subject image generation aims to synthesize user-provided subjects in a single image while preserving subject fidelity, ensuring prompt consistency, and aligning with human aesthetic preferences. However, existing methods, particularly those built on the In-Context-Learning paradigm, are limited by their reliance on simple reconstruction-based objectives, leading to both severe attribute leakage that compromises subject fidelity and failing to align with nuanced human preferences. To address this, we propose MultiCrafter, a framework that ensures high-fidelity, preference-aligned generation. First, we find that the root cause of attribute leakage is a significant entanglement of attention between different subjects during the generation process. Therefore, we introduce explicit positional supervision to explicitly separate attention regions for each subject, effectively mitigating attribute leakage. To enable the model to accurately plan the attention region of different subjects in diverse scenarios, we employ a Mixture-of-Experts architecture to enhance the model's capacity, allowing different experts to focus on different scenarios. Finally, we design a novel online reinforcement learning framework to align the model with human preferences, featuring a scoring mechanism to accurately assess multi-subject fidelity and a more stable training strategy tailored for the MoE architecture. Experiments validate that our framework significantly improves subject fidelity while aligning with human preferences better.
>
---
#### [new 127] What Happens Next? Anticipating Future Motion by Generating Point Trajectories
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究从单张图像预测物体未来运动轨迹的任务，旨在解决传统方法在动态建模和不确定性处理上的不足。提出生成密集轨迹网格的方法，优于现有模型，在模拟和真实数据中表现良好。**

- **链接: [http://arxiv.org/pdf/2509.21592v1](http://arxiv.org/pdf/2509.21592v1)**

> **作者:** Gabrijel Boduljak; Laurynas Karazija; Iro Laina; Christian Rupprecht; Andrea Vedaldi
>
> **摘要:** We consider the problem of forecasting motion from a single image, i.e., predicting how objects in the world are likely to move, without the ability to observe other parameters such as the object velocities or the forces applied to them. We formulate this task as conditional generation of dense trajectory grids with a model that closely follows the architecture of modern video generators but outputs motion trajectories instead of pixels. This approach captures scene-wide dynamics and uncertainty, yielding more accurate and diverse predictions than prior regressors and generators. We extensively evaluate our method on simulated data, demonstrate its effectiveness on downstream applications such as robotics, and show promising accuracy on real-world intuitive physics datasets. Although recent state-of-the-art video generators are often regarded as world models, we show that they struggle with forecasting motion from a single image, even in simple physical scenarios such as falling blocks or mechanical object interactions, despite fine-tuning on such data. We show that this limitation arises from the overhead of generating pixels rather than directly modeling motion.
>
---
#### [new 128] Safety Assessment of Scaffolding on Construction Site using AI
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于安全评估任务，旨在解决脚手架人工检查耗时且易出错的问题。研究开发了一个基于云的AI平台，通过点云数据分析，自动检测脚手架结构变化，提升施工现场安全性。**

- **链接: [http://arxiv.org/pdf/2509.21368v1](http://arxiv.org/pdf/2509.21368v1)**

> **作者:** Sameer Prabhu; Amit Patwardhan; Ramin Karim
>
> **摘要:** In the construction industry, safety assessment is vital to ensure both the reliability of assets and the safety of workers. Scaffolding, a key structural support asset requires regular inspection to detect and identify alterations from the design rules that may compromise the integrity and stability. At present, inspections are primarily visual and are conducted by site manager or accredited personnel to identify deviations. However, visual inspection is time-intensive and can be susceptible to human errors, which can lead to unsafe conditions. This paper explores the use of Artificial Intelligence (AI) and digitization to enhance the accuracy of scaffolding inspection and contribute to the safety improvement. A cloud-based AI platform is developed to process and analyse the point cloud data of scaffolding structure. The proposed system detects structural modifications through comparison and evaluation of certified reference data with the recent point cloud data. This approach may enable automated monitoring of scaffolding, reducing the time and effort required for manual inspections while enhancing the safety on a construction site.
>
---
#### [new 129] LucidFlux: Caption-Free Universal Image Restoration via a Large-Scale Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文提出LucidFlux，一种无需文字描述的通用图像修复框架。针对传统方法易模糊、失真或漂移的问题，利用大规模扩散变换器（Flux.1），设计轻量条件分支和自适应调制策略，实现结构保留与纹理恢复，提升真实场景下的修复效果。**

- **链接: [http://arxiv.org/pdf/2509.22414v1](http://arxiv.org/pdf/2509.22414v1)**

> **作者:** Song Fei; Tian Ye; Lujia Wang; Lei Zhu
>
> **备注:** Project Page: https://w2genai-lab.github.io/LucidFlux
>
> **摘要:** Universal image restoration (UIR) aims to recover images degraded by unknown mixtures while preserving semantics -- conditions under which discriminative restorers and UNet-based diffusion priors often oversmooth, hallucinate, or drift. We present LucidFlux, a caption-free UIR framework that adapts a large diffusion transformer (Flux.1) without image captions. LucidFlux introduces a lightweight dual-branch conditioner that injects signals from the degraded input and a lightly restored proxy to respectively anchor geometry and suppress artifacts. Then, a timestep- and layer-adaptive modulation schedule is designed to route these cues across the backbone's hierarchy, in order to yield coarse-to-fine and context-aware updates that protect the global structure while recovering texture. After that, to avoid the latency and instability of text prompts or MLLM captions, we enforce caption-free semantic alignment via SigLIP features extracted from the proxy. A scalable curation pipeline further filters large-scale data for structure-rich supervision. Across synthetic and in-the-wild benchmarks, LucidFlux consistently outperforms strong open-source and commercial baselines, and ablation studies verify the necessity of each component. LucidFlux shows that, for large DiTs, when, where, and what to condition on -- rather than adding parameters or relying on text prompts -- is the governing lever for robust and caption-free universal image restoration in the wild.
>
---
#### [new 130] LABELING COPILOT: A Deep Research Agent for Automated Data Curation in Computer Vision
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Labeling Copilot，一个用于计算机视觉领域自动数据整理的深度研究代理。针对高质量数据集构建中质量、多样性和成本的平衡问题，设计了三大核心模块：校准发现、可控合成和共识标注，实现了高效、可扩展的数据筛选与标注流程。**

- **链接: [http://arxiv.org/pdf/2509.22631v1](http://arxiv.org/pdf/2509.22631v1)**

> **作者:** Debargha Ganguly; Sumit Kumar; Ishwar Balappanawar; Weicong Chen; Shashank Kambhatla; Srinivasan Iyengar; Shivkumar Kalyanaraman; Ponnurangam Kumaraguru; Vipin Chaudhary
>
> **摘要:** Curating high-quality, domain-specific datasets is a major bottleneck for deploying robust vision systems, requiring complex trade-offs between data quality, diversity, and cost when researching vast, unlabeled data lakes. We introduce Labeling Copilot, the first data curation deep research agent for computer vision. A central orchestrator agent, powered by a large multimodal language model, uses multi-step reasoning to execute specialized tools across three core capabilities: (1) Calibrated Discovery sources relevant, in-distribution data from large repositories; (2) Controllable Synthesis generates novel data for rare scenarios with robust filtering; and (3) Consensus Annotation produces accurate labels by orchestrating multiple foundation models via a novel consensus mechanism incorporating non-maximum suppression and voting. Our large-scale validation proves the effectiveness of Labeling Copilot's components. The Consensus Annotation module excels at object discovery: on the dense COCO dataset, it averages 14.2 candidate proposals per image-nearly double the 7.4 ground-truth objects-achieving a final annotation mAP of 37.1%. On the web-scale Open Images dataset, it navigated extreme class imbalance to discover 903 new bounding box categories, expanding its capability to over 1500 total. Concurrently, our Calibrated Discovery tool, tested at a 10-million sample scale, features an active learning strategy that is up to 40x more computationally efficient than alternatives with equivalent sample efficiency. These experiments validate that an agentic workflow with optimized, scalable tools provides a robust foundation for curating industrial-scale datasets.
>
---
#### [new 131] DragGANSpace: Latent Space Exploration and Control for GANs
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出DragGANSpace，结合StyleGAN、DragGAN和PCA，提升GAN生成图像的潜在空间效率与可控性。通过PCA降维优化潜空间结构，在保持视觉质量的同时提高优化效率，并实现跨模型对齐与直观编辑。**

- **链接: [http://arxiv.org/pdf/2509.22169v1](http://arxiv.org/pdf/2509.22169v1)**

> **作者:** Kirsten Odendaal; Neela Kaushik; Spencer Halverson
>
> **备注:** 6 pages with 7 figures and 3 tables
>
> **摘要:** This work integrates StyleGAN, DragGAN and Principal Component Analysis (PCA) to enhance the latent space efficiency and controllability of GAN-generated images. Style-GAN provides a structured latent space, DragGAN enables intuitive image manipulation, and PCA reduces dimensionality and facilitates cross-model alignment for more streamlined and interpretable exploration of latent spaces. We apply our techniques to the Animal Faces High Quality (AFHQ) dataset, and find that our approach of integrating PCA-based dimensionality reduction with the Drag-GAN framework for image manipulation retains performance while improving optimization efficiency. Notably, introducing PCA into the latent W+ layers of DragGAN can consistently reduce the total optimization time while maintaining good visual quality and even boosting the Structural Similarity Index Measure (SSIM) of the optimized image, particularly in shallower latent spaces (W+ layers = 3). We also demonstrate capability for aligning images generated by two StyleGAN models trained on similar but distinct data domains (AFHQ-Dog and AFHQ-Cat), and show that we can control the latent space of these aligned images to manipulate the images in an intuitive and interpretable manner. Our findings highlight the possibility for efficient and interpretable latent space control for a wide range of image synthesis and editing applications.
>
---
#### [new 132] FreqDebias: Towards Generalizable Deepfake Detection via Consistency-Driven Frequency Debiasing
- **分类: cs.CV**

- **简介: 该论文针对深度伪造检测任务中模型因频率域偏差（谱偏）导致的泛化性差问题，提出FreqDebias框架。通过Forgery Mixup数据增强和双一致性正则化策略，有效缓解频域依赖，提升跨领域检测性能。**

- **链接: [http://arxiv.org/pdf/2509.22412v1](http://arxiv.org/pdf/2509.22412v1)**

> **作者:** Hossein Kashiani; Niloufar Alipour Talemi; Fatemeh Afghah
>
> **备注:** Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)
>
> **摘要:** Deepfake detectors often struggle to generalize to novel forgery types due to biases learned from limited training data. In this paper, we identify a new type of model bias in the frequency domain, termed spectral bias, where detectors overly rely on specific frequency bands, restricting their ability to generalize across unseen forgeries. To address this, we propose FreqDebias, a frequency debiasing framework that mitigates spectral bias through two complementary strategies. First, we introduce a novel Forgery Mixup (Fo-Mixup) augmentation, which dynamically diversifies frequency characteristics of training samples. Second, we incorporate a dual consistency regularization (CR), which enforces both local consistency using class activation maps (CAMs) and global consistency through a von Mises-Fisher (vMF) distribution on a hyperspherical embedding space. This dual CR mitigates over-reliance on certain frequency components by promoting consistent representation learning under both local and global supervision. Extensive experiments show that FreqDebias significantly enhances cross-domain generalization and outperforms state-of-the-art methods in both cross-domain and in-domain settings.
>
---
#### [new 133] LG-CD: Enhancing Language-Guided Change Detection through SAM2 Adaptation
- **分类: cs.CV**

- **简介: 该论文提出LG-CD模型，用于遥感变化检测任务。针对现有方法忽视多模态语义信息的问题，利用文本提示引导注意力，并结合SAM2和跨模态融合机制，提升变化检测的准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.21894v1](http://arxiv.org/pdf/2509.21894v1)**

> **作者:** Yixiao Liu; Yizhou Yang; Jinwen Li; Jun Tao; Ruoyu Li; Xiangkun Wang; Min Zhu; Junlong Cheng
>
> **备注:** *Corresponding authors: Min Zhu (min.zhu@scu.edu.cn) and Junlong Cheng (jlcheng@scu.edu.cn)
>
> **摘要:** Remote Sensing Change Detection (RSCD) typically identifies changes in land cover or surface conditions by analyzing multi-temporal images. Currently, most deep learning-based methods primarily focus on learning unimodal visual information, while neglecting the rich semantic information provided by multimodal data such as text. To address this limitation, we propose a novel Language-Guided Change Detection model (LG-CD). This model leverages natural language prompts to direct the network's attention to regions of interest, significantly improving the accuracy and robustness of change detection. Specifically, LG-CD utilizes a visual foundational model (SAM2) as a feature extractor to capture multi-scale pyramid features from high-resolution to low-resolution across bi-temporal remote sensing images. Subsequently, multi-layer adapters are employed to fine-tune the model for downstream tasks, ensuring its effectiveness in remote sensing change detection. Additionally, we design a Text Fusion Attention Module (TFAM) to align visual and textual information, enabling the model to focus on target change regions using text prompts. Finally, a Vision-Semantic Fusion Decoder (V-SFD) is implemented, which deeply integrates visual and semantic information through a cross-attention mechanism to produce highly accurate change detection masks. Our experiments on three datasets (LEVIR-CD, WHU-CD, and SYSU-CD) demonstrate that LG-CD consistently outperforms state-of-the-art change detection methods. Furthermore, our approach provides new insights into achieving generalized change detection by leveraging multimodal information.
>
---
#### [new 134] Unlocking the Essence of Beauty: Advanced Aesthetic Reasoning with Relative-Absolute Policy Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦图像美学评估任务，旨在解决多模态大模型因美学数据稀缺和主观性导致的评分与解释不准确问题。提出Aes-R1框架，结合AesCoT数据生成和RAPO强化学习算法，提升模型的美学评分与可解释性推理能力。**

- **链接: [http://arxiv.org/pdf/2509.21871v1](http://arxiv.org/pdf/2509.21871v1)**

> **作者:** Boyang Liu; Yifan Hu; Senjie Jin; Shihan Dou; Gonglei Shi; Jie Shao; Tao Gui; Xuanjing Huang
>
> **摘要:** Multimodal large language models (MLLMs) are well suited to image aesthetic assessment, as they can capture high-level aesthetic features leveraging their cross-modal understanding capacity. However, the scarcity of multimodal aesthetic reasoning data and the inherently subjective nature of aesthetic judgment make it difficult for MLLMs to generate accurate aesthetic judgments with interpretable rationales. To this end, we propose Aes-R1, a comprehensive aesthetic reasoning framework with reinforcement learning (RL). Concretely, Aes-R1 integrates a pipeline, AesCoT, to construct and filter high-quality chain-of-thought aesthetic reasoning data used for cold-start. After teaching the model to generate structured explanations prior to scoring, we then employ the Relative-Absolute Policy Optimization (RAPO), a novel RL algorithm that jointly optimizes absolute score regression and relative ranking order, improving both per-image accuracy and cross-image preference judgments. Aes-R1 enables MLLMs to generate grounded explanations alongside faithful scores, thereby enhancing aesthetic scoring and reasoning in a unified framework. Extensive experiments demonstrate that Aes-R1 improves the backbone's average PLCC/SRCC by 47.9%/34.8%, surpassing state-of-the-art baselines of similar size. More ablation studies validate Aes-R1's robust generalization under limited supervision and in out-of-distribution scenarios.
>
---
#### [new 135] Spatial Reasoning in Foundation Models: Benchmarking Object-Centric Spatial Understanding
- **分类: cs.CV**

- **简介: 该论文聚焦于基础模型的空间推理能力，旨在解决现有模型对物体布局和关系理解不足的问题。通过构建合成数据集，系统评估了多种视觉模型和视觉语言模型在空间定位、推理及检索任务中的表现，揭示了定位精度与空间理解之间的差距，强调需发展具备空间感知的基础模型。**

- **链接: [http://arxiv.org/pdf/2509.21922v1](http://arxiv.org/pdf/2509.21922v1)**

> **作者:** Vahid Mirjalili; Ramin Giahi; Sriram Kollipara; Akshay Kekuda; Kehui Yao; Kai Zhao; Jianpeng Xu; Kaushiki Nag; Sinduja Subramaniam; Topojoy Biswas; Evren Korpeoglu; Kannan Achan
>
> **备注:** 4 pages, NeurIPS Workshop SpaVLE
>
> **摘要:** Spatial understanding is a critical capability for vision foundation models. While recent advances in large vision models or vision-language models (VLMs) have expanded recognition capabilities, most benchmarks emphasize localization accuracy rather than whether models capture how objects are arranged and related within a scene. This gap is consequential; effective scene understanding requires not only identifying objects, but reasoning about their relative positions, groupings, and depth. In this paper, we present a systematic benchmark for object-centric spatial reasoning in foundation models. Using a controlled synthetic dataset, we evaluate state-of-the-art vision models (e.g., GroundingDINO, Florence-2, OWLv2) and large VLMs (e.g., InternVL, LLaVA, GPT-4o) across three tasks: spatial localization, spatial reasoning, and downstream retrieval tasks. We find a stable trade-off: detectors such as GroundingDINO and OWLv2 deliver precise boxes with limited relational reasoning, while VLMs like SmolVLM and GPT-4o provide coarse layout cues and fluent captions but struggle with fine-grained spatial context. Our study highlights the gap between localization and true spatial understanding, and pointing toward the need for spatially-aware foundation models in the community.
>
---
#### [new 136] DeLiVR: Differential Spatiotemporal Lie Bias for Efficient Video Deraining
- **分类: cs.CV**

- **简介: 该论文提出DeLiVR，用于视频去雨任务。针对视频中雨痕、模糊及帧间不一致问题，引入基于李群的时空偏差机制，通过旋转约束和差分位移提升对齐与特征聚合效率，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2509.21719v1](http://arxiv.org/pdf/2509.21719v1)**

> **作者:** Shuning Sun; Jialang Lu; Xiang Chen; Jichao Wang; Dianjie Lu; Guijuan Zhang; Guangwei Gao; Zhuoran Zheng
>
> **摘要:** Videos captured in the wild often suffer from rain streaks, blur, and noise. In addition, even slight changes in camera pose can amplify cross-frame mismatches and temporal artifacts. Existing methods rely on optical flow or heuristic alignment, which are computationally expensive and less robust. To address these challenges, Lie groups provide a principled way to represent continuous geometric transformations, making them well-suited for enforcing spatial and temporal consistency in video modeling. Building on this insight, we propose DeLiVR, an efficient video deraining method that injects spatiotemporal Lie-group differential biases directly into attention scores of the network. Specifically, the method introduces two complementary components. First, a rotation-bounded Lie relative bias predicts the in-plane angle of each frame using a compact prediction module, where normalized coordinates are rotated and compared with base coordinates to achieve geometry-consistent alignment before feature aggregation. Second, a differential group displacement computes angular differences between adjacent frames to estimate a velocity. This bias computation combines temporal decay and attention masks to focus on inter-frame relationships while precisely matching the direction of rain streaks. Extensive experimental results demonstrate the effectiveness of our method on publicly available benchmarks.
>
---
#### [new 137] Debugging Concept Bottleneck Models through Removal and Retraining
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究概念瓶颈模型（CBM）的调试方法，旨在解决模型与专家推理不一致的问题。提出“移除-重训练”框架和CBDebug方法，通过去除不良概念并利用反馈生成辅助标签，减少模型对错误概念的依赖，提升模型可靠性与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.21385v1](http://arxiv.org/pdf/2509.21385v1)**

> **作者:** Eric Enouen; Sainyam Galhotra
>
> **摘要:** Concept Bottleneck Models (CBMs) use a set of human-interpretable concepts to predict the final task label, enabling domain experts to not only validate the CBM's predictions, but also intervene on incorrect concepts at test time. However, these interventions fail to address systemic misalignment between the CBM and the expert's reasoning, such as when the model learns shortcuts from biased data. To address this, we present a general interpretable debugging framework for CBMs that follows a two-step process of Removal and Retraining. In the Removal step, experts use concept explanations to identify and remove any undesired concepts. In the Retraining step, we introduce CBDebug, a novel method that leverages the interpretability of CBMs as a bridge for converting concept-level user feedback into sample-level auxiliary labels. These labels are then used to apply supervised bias mitigation and targeted augmentation, reducing the model's reliance on undesired concepts. We evaluate our framework with both real and automated expert feedback, and find that CBDebug significantly outperforms prior retraining methods across multiple CBM architectures (PIP-Net, Post-hoc CBM) and benchmarks with known spurious correlations.
>
---
#### [new 138] LFA-Net: A Lightweight Network with LiteFusion Attention for Retinal Vessel Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LFA-Net，用于视网膜血管分割任务。针对现有模型计算成本高、小血管分割效果差的问题，设计了轻量级的LiteFusion注意力模块，结合残差连接与视觉状态空间动态，在保证性能的同时降低参数量和计算开销，并在多个数据集上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.21738v1](http://arxiv.org/pdf/2509.21738v1)**

> **作者:** Mehwish Mehmood; Ivor Spence; Muhammad Fahim
>
> **摘要:** Lightweight retinal vessel segmentation is important for the early diagnosis of vision-threatening and systemic diseases, especially in a real-world clinical environment with limited computational resources. Although segmentation methods based on deep learning are improving, existing models are still facing challenges of small vessel segmentation and high computational costs. To address these challenges, we proposed a new vascular segmentation network, LFA-Net, which incorporates a newly designed attention module, LiteFusion-Attention. This attention module incorporates residual learning connections, Vision Mamba-inspired dynamics, and modulation-based attention, enabling the model to capture local and global context efficiently and in a lightweight manner. LFA-Net offers high performance with 0.11 million parameters, 0.42 MB memory size, and 4.46 GFLOPs, which make it ideal for resource-constrained environments. We validated our proposed model on DRIVE, STARE, and CHASE_DB with outstanding performance in terms of dice scores of 83.28, 87.44, and 84.50% and Jaccard indices of 72.85, 79.31, and 74.70%, respectively. The code of LFA-Net is available online https://github.com/Mehwish4593/LFA-Net.
>
---
#### [new 139] VideoJudge: Bootstrapping Enables Scalable Supervision of MLLM-as-a-Judge for Video Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VideoJudge，一种用于视频理解模型评估的MLLM裁判。针对现有评估方法不足，通过生成器与评估器的交互训练，实现更符合人类判断的自动评估，在多个基准上优于大模型基线。**

- **链接: [http://arxiv.org/pdf/2509.21451v1](http://arxiv.org/pdf/2509.21451v1)**

> **作者:** Abdul Waheed; Zhen Wu; Dareen Alharthi; Seungone Kim; Bhiksha Raj
>
> **备注:** Work in progress
>
> **摘要:** Precisely evaluating video understanding models remains challenging: commonly used metrics such as BLEU, ROUGE, and BERTScore fail to capture the fineness of human judgment, while obtaining such judgments through manual evaluation is costly. Recent work has explored using large language models (LLMs) or multimodal LLMs (MLLMs) as evaluators, but their extension to video understanding remains relatively unexplored. In this work, we introduce VideoJudge, a 3B and 7B-sized MLLM judge specialized to evaluate outputs from video understanding models (\textit{i.e.}, text responses conditioned on videos). To train VideoJudge, our recipe builds on the interplay between a generator and an evaluator: the generator is prompted to produce responses conditioned on a target rating, and responses not matching the evaluator's rating are discarded. Across three out of four meta-evaluation benchmarks, VideoJudge-7B outperforms larger MLLM judge baselines such as Qwen2.5-VL (32B and 72B). Notably, we find that LLM judges (Qwen3) models perform worse than MLLM judges (Qwen2.5-VL) and long chain-of-thought reasoning does not improve performance, indicating that providing video inputs is crucial for evaluation of video understanding tasks.
>
---
#### [new 140] RAPID^3: Tri-Level Reinforced Acceleration Policies for Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文提出RAPID^3，旨在加速扩散Transformer（DiT）的采样过程。针对现有加速方法在图像生成质量与速度间的权衡不足，设计了三重强化策略（Step-Skip、Cache-Reuse、Sparse-Attention），通过在线训练策略头实现图像级自适应加速，无需修改基础模型。**

- **链接: [http://arxiv.org/pdf/2509.22323v1](http://arxiv.org/pdf/2509.22323v1)**

> **作者:** Wangbo Zhao; Yizeng Han; Zhiwei Tang; Jiasheng Tang; Pengfei Zhou; Kai Wang; Bohan Zhuang; Zhangyang Wang; Fan Wang; Yang You
>
> **摘要:** Diffusion Transformers (DiTs) excel at visual generation yet remain hampered by slow sampling. Existing training-free accelerators - step reduction, feature caching, and sparse attention - enhance inference speed but typically rely on a uniform heuristic or a manually designed adaptive strategy for all images, leaving quality on the table. Alternatively, dynamic neural networks offer per-image adaptive acceleration, but their high fine-tuning costs limit broader applicability. To address these limitations, we introduce RAPID3: Tri-Level Reinforced Acceleration Policies for Diffusion Transformers, a framework that delivers image-wise acceleration with zero updates to the base generator. Specifically, three lightweight policy heads - Step-Skip, Cache-Reuse, and Sparse-Attention - observe the current denoising state and independently decide their corresponding speed-up at each timestep. All policy parameters are trained online via Group Relative Policy Optimization (GRPO) while the generator remains frozen. Meanwhile, an adversarially learned discriminator augments the reward signal, discouraging reward hacking by boosting returns only when generated samples stay close to the original model's distribution. Across state-of-the-art DiT backbones, including Stable Diffusion 3 and FLUX, RAPID3 achieves nearly 3x faster sampling with competitive generation quality.
>
---
#### [new 141] Scale-Wise VAR is Secretly Discrete Diffusion
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究视觉生成任务，探讨如何提升AR变压器模型VAR的性能。提出VAR在特定条件下等价于离散扩散，并基于此建立SRDD框架，融合扩散优势，提高生成效率和质量。**

- **链接: [http://arxiv.org/pdf/2509.22636v1](http://arxiv.org/pdf/2509.22636v1)**

> **作者:** Amandeep Kumar; Nithin Gopalakrishnan Nair; Vishal M. Patel
>
> **备注:** Technical Reports
>
> **摘要:** Autoregressive (AR) transformers have emerged as a powerful paradigm for visual generation, largely due to their scalability, computational efficiency and unified architecture with language and vision. Among them, next scale prediction Visual Autoregressive Generation (VAR) has recently demonstrated remarkable performance, even surpassing diffusion-based models. In this work, we revisit VAR and uncover a theoretical insight: when equipped with a Markovian attention mask, VAR is mathematically equivalent to a discrete diffusion. We term this reinterpretation as Scalable Visual Refinement with Discrete Diffusion (SRDD), establishing a principled bridge between AR transformers and diffusion models. Leveraging this new perspective, we show how one can directly import the advantages of diffusion such as iterative refinement and reduce architectural inefficiencies into VAR, yielding faster convergence, lower inference cost, and improved zero-shot reconstruction. Across multiple datasets, we show that the diffusion based perspective of VAR leads to consistent gains in efficiency and generation.
>
---
#### [new 142] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MesaTask，一个基于LLM的框架，用于任务驱动的桌面场景生成。针对传统方法在布局合理性与任务对齐上的不足，构建了包含10,700个合成场景的数据集MesaTask-10K，并引入空间推理链以提升生成效果。**

- **链接: [http://arxiv.org/pdf/2509.22281v1](http://arxiv.org/pdf/2509.22281v1)**

> **作者:** Jinkun Hao; Naifu Liang; Zhen Luo; Xudong Xu; Weipeng Zhong; Ran Yi; Yichen Jin; Zhaoyang Lyu; Feng Zheng; Lizhuang Ma; Jiangmiao Pang
>
> **备注:** Accepted by NeurIPS 2025; Project page: https://mesatask.github.io/
>
> **摘要:** The ability of robots to interpret human instructions and execute manipulation tasks necessitates the availability of task-relevant tabletop scenes for training. However, traditional methods for creating these scenes rely on time-consuming manual layout design or purely randomized layouts, which are limited in terms of plausibility or alignment with the tasks. In this paper, we formulate a novel task, namely task-oriented tabletop scene generation, which poses significant challenges due to the substantial gap between high-level task instructions and the tabletop scenes. To support research on such a challenging task, we introduce MesaTask-10K, a large-scale dataset comprising approximately 10,700 synthetic tabletop scenes with manually crafted layouts that ensure realistic layouts and intricate inter-object relations. To bridge the gap between tasks and scenes, we propose a Spatial Reasoning Chain that decomposes the generation process into object inference, spatial interrelation reasoning, and scene graph construction for the final 3D layout. We present MesaTask, an LLM-based framework that utilizes this reasoning chain and is further enhanced with DPO algorithms to generate physically plausible tabletop scenes that align well with given task descriptions. Exhaustive experiments demonstrate the superior performance of MesaTask compared to baselines in generating task-conforming tabletop scenes with realistic layouts. Project page is at https://mesatask.github.io/
>
---
#### [new 143] FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction
- **分类: cs.CV**

- **简介: 该论文提出FantasyWorld，旨在解决视频生成与3D建模空间一致性不足的问题。通过引入几何分支与跨分支监督，实现视频与3D场联合建模，提升多视角一致性和下游任务表现。**

- **链接: [http://arxiv.org/pdf/2509.21657v1](http://arxiv.org/pdf/2509.21657v1)**

> **作者:** Yixiang Dai; Fan Jiang; Chiyu Wang; Mu Xu; Yonggang Qi
>
> **摘要:** High-quality 3D world models are pivotal for embodied intelligence and Artificial General Intelligence (AGI), underpinning applications such as AR/VR content creation and robotic navigation. Despite the established strong imaginative priors, current video foundation models lack explicit 3D grounding capabilities, thus being limited in both spatial consistency and their utility for downstream 3D reasoning tasks. In this work, we present FantasyWorld, a geometry-enhanced framework that augments frozen video foundation models with a trainable geometric branch, enabling joint modeling of video latents and an implicit 3D field in a single forward pass. Our approach introduces cross-branch supervision, where geometry cues guide video generation and video priors regularize 3D prediction, thus yielding consistent and generalizable 3D-aware video representations. Notably, the resulting latents from the geometric branch can potentially serve as versatile representations for downstream 3D tasks such as novel view synthesis and navigation, without requiring per-scene optimization or fine-tuning. Extensive experiments show that FantasyWorld effectively bridges video imagination and 3D perception, outperforming recent geometry-consistent baselines in multi-view coherence and style consistency. Ablation studies further confirm that these gains stem from the unified backbone and cross-branch information exchange.
>
---
#### [new 144] CCNeXt: An Effective Self-Supervised Stereo Depth Estimation Approach
- **分类: cs.CV**

- **简介: 该论文提出CCNeXt，一种高效的自监督立体深度估计方法。针对计算受限场景下深度估计的问题，设计了带有窗口化对极交叉注意力模块的CNN架构，改进解码器，在KITTI数据集上取得SOTA性能并显著提升速度。**

- **链接: [http://arxiv.org/pdf/2509.22627v1](http://arxiv.org/pdf/2509.22627v1)**

> **作者:** Alexandre Lopes; Roberto Souza; Helio Pedrini
>
> **摘要:** Depth Estimation plays a crucial role in recent applications in robotics, autonomous vehicles, and augmented reality. These scenarios commonly operate under constraints imposed by computational power. Stereo image pairs offer an effective solution for depth estimation since it only needs to estimate the disparity of pixels in image pairs to determine the depth in a known rectified system. Due to the difficulty in acquiring reliable ground-truth depth data across diverse scenarios, self-supervised techniques emerge as a solution, particularly when large unlabeled datasets are available. We propose a novel self-supervised convolutional approach that outperforms existing state-of-the-art Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) while balancing computational cost. The proposed CCNeXt architecture employs a modern CNN feature extractor with a novel windowed epipolar cross-attention module in the encoder, complemented by a comprehensive redesign of the depth estimation decoder. Our experiments demonstrate that CCNeXt achieves competitive metrics on the KITTI Eigen Split test data while being 10.18$\times$ faster than the current best model and achieves state-of-the-art results in all metrics in the KITTI Eigen Split Improved Ground Truth and Driving Stereo datasets when compared to recently proposed techniques. To ensure complete reproducibility, our project is accessible at \href{https://github.com/alelopes/CCNext}{\texttt{https://github.com/alelopes/CCNext}}.
>
---
#### [new 145] Enhancing Contrastive Learning for Geolocalization by Discovering Hard Negatives on Semivariograms
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对图像地理定位任务，解决对比学习中因忽略空间依赖性导致的假负例和难负例区分问题。提出结合半变异函数的空间正则化对比学习方法，通过建模视觉与地理距离关系，提升细粒度定位性能。**

- **链接: [http://arxiv.org/pdf/2509.21573v1](http://arxiv.org/pdf/2509.21573v1)**

> **作者:** Boyi Chen; Zhangyu Wang; Fabian Deuser; Johann Maximilian Zollner; Martin Werner
>
> **摘要:** Accurate and robust image-based geo-localization at a global scale is challenging due to diverse environments, visually ambiguous scenes, and the lack of distinctive landmarks in many regions. While contrastive learning methods show promising performance by aligning features between street-view images and corresponding locations, they neglect the underlying spatial dependency in the geographic space. As a result, they fail to address the issue of false negatives -- image pairs that are both visually and geographically similar but labeled as negatives, and struggle to effectively distinguish hard negatives, which are visually similar but geographically distant. To address this issue, we propose a novel spatially regularized contrastive learning strategy that integrates a semivariogram, which is a geostatistical tool for modeling how spatial correlation changes with distance. We fit the semivariogram by relating the distance of images in feature space to their geographical distance, capturing the expected visual content in a spatial correlation. With the fitted semivariogram, we define the expected visual dissimilarity at a given spatial distance as reference to identify hard negatives and false negatives. We integrate this strategy into GeoCLIP and evaluate it on the OSV5M dataset, demonstrating that explicitly modeling spatial priors improves image-based geo-localization performance, particularly at finer granularity.
>
---
#### [new 146] Do Sparse Subnetworks Exhibit Cognitively Aligned Attention? Effects of Pruning on Saliency Map Fidelity, Sparsity, and Concept Coherence
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究神经网络剪枝对模型可解释性的影响，属于模型压缩与可解释性交叉任务。通过分析剪枝后ResNet-18的显著图和概念表示，发现轻度至中度剪枝可提升注意力聚焦与语义一致性，而重度剪枝则损害可解释性。**

- **链接: [http://arxiv.org/pdf/2509.21387v1](http://arxiv.org/pdf/2509.21387v1)**

> **作者:** Sanish Suwal; Dipkamal Bhusal; Michael Clifford; Nidhi Rastogi
>
> **备注:** 4 pages
>
> **摘要:** Prior works have shown that neural networks can be heavily pruned while preserving performance, but the impact of pruning on model interpretability remains unclear. In this work, we investigate how magnitude-based pruning followed by fine-tuning affects both low-level saliency maps and high-level concept representations. Using a ResNet-18 trained on ImageNette, we compare post-hoc explanations from Vanilla Gradients (VG) and Integrated Gradients (IG) across pruning levels, evaluating sparsity and faithfulness. We further apply CRAFT-based concept extraction to track changes in semantic coherence of learned concepts. Our results show that light-to-moderate pruning improves saliency-map focus and faithfulness while retaining distinct, semantically meaningful concepts. In contrast, aggressive pruning merges heterogeneous features, reducing saliency map sparsity and concept coherence despite maintaining accuracy. These findings suggest that while pruning can shape internal representations toward more human-aligned attention patterns, excessive pruning undermines interpretability.
>
---
#### [new 147] Resolving Ambiguity in Gaze-Facilitated Visual Assistant Interaction Paradigm
- **分类: cs.CV**

- **简介: 该论文针对基于眼动的视觉助手交互中出现的歧义问题，提出GLARIFY方法。通过引入时空眼动信息、构建数据集和设计热力图模块，提升视觉语言模型对用户注意力的理解，从而优化多模态交互效果。**

- **链接: [http://arxiv.org/pdf/2509.21980v1](http://arxiv.org/pdf/2509.21980v1)**

> **作者:** Zeyu Wang; Baiyu Chen; Kun Yan; Hongjing Piao; Hao Xue; Flora D. Salim; Yuanchun Shi; Yuntao Wang
>
> **摘要:** With the rise in popularity of smart glasses, users' attention has been integrated into Vision-Language Models (VLMs) to streamline multi-modal querying in daily scenarios. However, leveraging gaze data to model users' attention may introduce ambiguity challenges: (1) users' verbal questions become ambiguous by using pronouns or skipping context, (2) humans' gaze patterns can be noisy and exhibit complex spatiotemporal relationships with their spoken questions. Previous works only consider single image as visual modality input, failing to capture the dynamic nature of the user's attention. In this work, we introduce GLARIFY, a novel method to leverage spatiotemporal gaze information to enhance the model's effectiveness in real-world applications. Initially, we analyzed hundreds of querying samples with the gaze modality to demonstrate the noisy nature of users' gaze patterns. We then utilized GPT-4o to design an automatic data synthesis pipeline to generate the GLARIFY-Ambi dataset, which includes a dedicated chain-of-thought (CoT) process to handle noisy gaze patterns. Finally, we designed a heatmap module to incorporate gaze information into cutting-edge VLMs while preserving their pretrained knowledge. We evaluated GLARIFY using a hold-out test set. Experiments demonstrate that GLARIFY significantly outperforms baselines. By robustly aligning VLMs with human attention, GLARIFY paves the way for a usable and intuitive interaction paradigm with a visual assistant.
>
---
#### [new 148] Rule-Based Reinforcement Learning for Document Image Classification with Vision Language Models
- **分类: cs.CV**

- **简介: 该论文研究了基于规则的强化学习在文档图像分类任务中的应用，旨在提升模型对分布外数据的泛化能力。通过实验分析了三种场景下的表现，验证了强化学习在文档分析任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.22283v1](http://arxiv.org/pdf/2509.22283v1)**

> **作者:** Michael Jungo; Andreas Fischer
>
> **备注:** Code available at https://github.com/jungomi/vision-finetune
>
> **摘要:** Rule-based reinforcement learning has been gaining popularity ever since DeepSeek-R1 has demonstrated its success through simple verifiable rewards. In the domain of document analysis, reinforcement learning is not as prevalent, even though many downstream tasks may benefit from the emerging properties of reinforcement learning, particularly the enhanced reason capabilities. We study the effects of rule-based reinforcement learning with the task of Document Image Classification which is one of the most commonly studied downstream tasks in document analysis. We find that reinforcement learning tends to have better generalisation capabilities to out-of-distritbution data, which we examine in three different scenarios, namely out-of-distribution images, unseen classes and different modalities. Our code is available at https://github.com/jungomi/vision-finetune.
>
---
#### [new 149] ShipwreckFinder: A QGIS Tool for Shipwreck Detection in Multibeam Sonar Data
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出ShipwreckFinder，一个QGIS插件，用于从多波束声呐数据中自动检测沉船。针对传统人工分析耗时且依赖专家的问题，该工具结合深度学习和合成数据生成，实现了高效的沉船分割与定位。**

- **链接: [http://arxiv.org/pdf/2509.21386v1](http://arxiv.org/pdf/2509.21386v1)**

> **作者:** Anja Sheppard; Tyler Smithline; Andrew Scheffer; David Smith; Advaith V. Sethuraman; Ryan Bird; Sabrina Lin; Katherine A. Skinner
>
> **备注:** Accepted to OCEANS 2025 Great Lakes
>
> **摘要:** In this paper, we introduce ShipwreckFinder, an open-source QGIS plugin that detects shipwrecks from multibeam sonar data. Shipwrecks are an important historical marker of maritime history, and can be discovered through manual inspection of bathymetric data. However, this is a time-consuming process and often requires expert analysis. Our proposed tool allows users to automatically preprocess bathymetry data, perform deep learning inference, threshold model outputs, and produce either pixel-wise segmentation masks or bounding boxes of predicted shipwrecks. The backbone of this open-source tool is a deep learning model, which is trained on a variety of shipwreck data from the Great Lakes and the coasts of Ireland. Additionally, we employ synthetic data generation in order to increase the size and diversity of our dataset. We demonstrate superior segmentation performance with our open-source tool and training pipeline as compared to a deep learning-based ArcGIS toolkit and a more classical inverse sinkhole detection method. The open-source tool can be found at https://github.com/umfieldrobotics/ShipwreckFinderQGISPlugin.
>
---
#### [new 150] Multi-View Crowd Counting With Self-Supervised Learning
- **分类: cs.CV**

- **简介: 该论文提出SSLCounter，一种基于自监督学习（SSL）的多视角人群计数（MVC）方法。针对传统方法依赖大量标注数据的问题，利用神经体积渲染技术提升数据效率，在少数据下仍能取得优异性能。**

- **链接: [http://arxiv.org/pdf/2509.21918v1](http://arxiv.org/pdf/2509.21918v1)**

> **作者:** Hong Mo; Xiong Zhang; Tengfei Shi; Zhongbo Wu
>
> **摘要:** Multi-view counting (MVC) methods have attracted significant research attention and stimulated remarkable progress in recent years. Despite their success, most MVC methods have focused on improving performance by following the fully supervised learning (FSL) paradigm, which often requires large amounts of annotated data. In this work, we propose SSLCounter, a novel self-supervised learning (SSL) framework for MVC that leverages neural volumetric rendering to alleviate the reliance on large-scale annotated datasets. SSLCounter learns an implicit representation w.r.t. the scene, enabling the reconstruction of continuous geometry shape and the complex, view-dependent appearance of their 2D projections via differential neural rendering. Owing to its inherent flexibility, the key idea of our method can be seamlessly integrated into exsiting frameworks. Notably, extensive experiments demonstrate that SSLCounter not only demonstrates state-of-the-art performances but also delivers competitive performance with only using 70% proportion of training data, showcasing its superior data efficiency across multiple MVC benchmarks.
>
---
#### [new 151] Prompt-guided Representation Disentanglement for Action Recognition
- **分类: cs.CV**

- **简介: 该论文针对视频动作识别任务，旨在解决多动作场景中对象间交互建模困难的问题。提出ProDA框架，通过动态提示模块和图解析网络生成动作特定表征，实现复杂场景中指定动作的解耦表示。**

- **链接: [http://arxiv.org/pdf/2509.21783v1](http://arxiv.org/pdf/2509.21783v1)**

> **作者:** Tianci Wu; Guangming Zhu; Jiang Lu; Siyuan Wang; Ning Wang; Nuoye Xiong; Zhang Liang
>
> **摘要:** Action recognition is a fundamental task in video understanding. Existing methods typically extract unified features to process all actions in one video, which makes it challenging to model the interactions between different objects in multi-action scenarios. To alleviate this issue, we explore disentangling any specified actions from complex scenes as an effective solution. In this paper, we propose Prompt-guided Disentangled Representation for Action Recognition (ProDA), a novel framework that disentangles any specified actions from a multi-action scene. ProDA leverages Spatio-temporal Scene Graphs (SSGs) and introduces Dynamic Prompt Module (DPM) to guide a Graph Parsing Neural Network (GPNN) in generating action-specific representations. Furthermore, we design a video-adapted GPNN that aggregates information using dynamic weights. Experiments in video action recognition demonstrate the effectiveness of our approach when compared with the state-of-the-art methods. Our code can be found in https://github.com/iamsnaping/ProDA.git
>
---
#### [new 152] CoFFT: Chain of Foresight-Focus Thought for Visual Language Models
- **分类: cs.CV**

- **简介: 该论文提出CoFFT方法，用于提升视觉语言模型的视觉推理能力。针对模型在复杂图像中易受干扰的问题，通过模拟人类视觉认知，采用三阶段迭代机制（多样本生成、双前瞻解码、视觉焦点调整），无需训练即可增强模型对关键视觉区域的聚焦与推理精度。**

- **链接: [http://arxiv.org/pdf/2509.22010v1](http://arxiv.org/pdf/2509.22010v1)**

> **作者:** Xinyu Zhang; Yuxuan Dong; Lingling Zhang; Chengyou Jia; Zhuohang Dang; Basura Fernando; Jun Liu; Mike Zheng Shou
>
> **摘要:** Despite significant advances in Vision Language Models (VLMs), they remain constrained by the complexity and redundancy of visual input. When images contain large amounts of irrelevant information, VLMs are susceptible to interference, thus generating excessive task-irrelevant reasoning processes or even hallucinations. This limitation stems from their inability to discover and process the required regions during reasoning precisely. To address this limitation, we present the Chain of Foresight-Focus Thought (CoFFT), a novel training-free approach that enhances VLMs' visual reasoning by emulating human visual cognition. Each Foresight-Focus Thought consists of three stages: (1) Diverse Sample Generation: generates diverse reasoning samples to explore potential reasoning paths, where each sample contains several reasoning steps; (2) Dual Foresight Decoding: rigorously evaluates these samples based on both visual focus and reasoning progression, adding the first step of optimal sample to the reasoning process; (3) Visual Focus Adjustment: precisely adjust visual focus toward regions most beneficial for future reasoning, before returning to stage (1) to generate subsequent reasoning samples until reaching the final answer. These stages function iteratively, creating an interdependent cycle where reasoning guides visual focus and visual focus informs subsequent reasoning. Empirical results across multiple benchmarks using Qwen2.5-VL, InternVL-2.5, and Llava-Next demonstrate consistent performance improvements of 3.1-5.8\% with controllable increasing computational overhead.
>
---
#### [new 153] Geo-R1: Improving Few-Shot Geospatial Referring Expression Understanding with Reinforcement Fine-Tuning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对遥感图像中少样本地理指代表达理解任务，提出Geo-R1方法。通过强化微调，使模型先生成可解释的推理链再定位目标，提升在数据稀缺场景下的泛化性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.21976v1](http://arxiv.org/pdf/2509.21976v1)**

> **作者:** Zilun Zhang; Zian Guan; Tiancheng Zhao; Haozhan Shen; Tianyu Li; Yuxiang Cai; Zhonggen Su; Zhaojun Liu; Jianwei Yin; Xiang Li
>
> **摘要:** Referring expression understanding in remote sensing poses unique challenges, as it requires reasoning over complex object-context relationships. While supervised fine-tuning (SFT) on multimodal large language models achieves strong performance with massive labeled datasets, they struggle in data-scarce scenarios, leading to poor generalization. To address this limitation, we propose Geo-R1, a reasoning-centric reinforcement fine-tuning (RFT) paradigm for few-shot geospatial referring. Geo-R1 enforces the model to first generate explicit, interpretable reasoning chains that decompose referring expressions, and then leverage these rationales to localize target objects. This "reason first, then act" process enables the model to make more effective use of limited annotations, enhances generalization, and provides interpretability. We validate Geo-R1 on three carefully designed few-shot geospatial referring benchmarks, where our model consistently and substantially outperforms SFT baselines. It also demonstrates strong cross-dataset generalization, highlighting its robustness. Code and data will be released at http://geo-r1.github.io.
>
---
#### [new 154] Self-Supervised Point Cloud Completion based on Multi-View Augmentations of Single Partial Point Cloud
- **分类: cs.CV**

- **简介: 该论文研究点云补全任务，旨在从部分观测重建完整形状。针对现有方法依赖真实标签、数据限制及自监督信号弱的问题，提出基于单部分点云多视角增强的自监督方法，并引入Mamba提升生成质量，实验表明效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.22132v1](http://arxiv.org/pdf/2509.22132v1)**

> **作者:** Jingjing Lu; Huilong Pi; Yunchuan Qin; Zhuo Tang; Ruihui Li
>
> **摘要:** Point cloud completion aims to reconstruct complete shapes from partial observations. Although current methods have achieved remarkable performance, they still have some limitations: Supervised methods heavily rely on ground truth, which limits their generalization to real-world datasets due to the synthetic-to-real domain gap. Unsupervised methods require complete point clouds to compose unpaired training data, and weakly-supervised methods need multi-view observations of the object. Existing self-supervised methods frequently produce unsatisfactory predictions due to the limited capabilities of their self-supervised signals. To overcome these challenges, we propose a novel self-supervised point cloud completion method. We design a set of novel self-supervised signals based on multi-view augmentations of the single partial point cloud. Additionally, to enhance the model's learning ability, we first incorporate Mamba into self-supervised point cloud completion task, encouraging the model to generate point clouds with better quality. Experiments on synthetic and real-world datasets demonstrate that our method achieves state-of-the-art results.
>
---
#### [new 155] Drag4D: Align Your Motion with Text-Driven 3D Scene Generation
- **分类: cs.CV**

- **简介: 该论文提出Drag4D，一个结合文本驱动3D场景生成与物体运动控制的交互框架。任务是实现用户可控的3D物体运动与高质量背景对齐。工作包括：改进3D背景生成、3D对象融合及运动动画，解决空间与时间一致性问题。**

- **链接: [http://arxiv.org/pdf/2509.21888v1](http://arxiv.org/pdf/2509.21888v1)**

> **作者:** Minjun Kang; Inkyu Shin; Taeyeop Lee; In So Kweon; Kuk-Jin Yoon
>
> **备注:** version 1
>
> **摘要:** We introduce Drag4D, an interactive framework that integrates object motion control within text-driven 3D scene generation. This framework enables users to define 3D trajectories for the 3D objects generated from a single image, seamlessly integrating them into a high-quality 3D background. Our Drag4D pipeline consists of three stages. First, we enhance text-to-3D background generation by applying 2D Gaussian Splatting with panoramic images and inpainted novel views, resulting in dense and visually complete 3D reconstructions. In the second stage, given a reference image of the target object, we introduce a 3D copy-and-paste approach: the target instance is extracted in a full 3D mesh using an off-the-shelf image-to-3D model and seamlessly composited into the generated 3D scene. The object mesh is then positioned within the 3D scene via our physics-aware object position learning, ensuring precise spatial alignment. Lastly, the spatially aligned object is temporally animated along a user-defined 3D trajectory. To mitigate motion hallucination and ensure view-consistent temporal alignment, we develop a part-augmented, motion-conditioned video diffusion model that processes multiview image pairs together with their projected 2D trajectories. We demonstrate the effectiveness of our unified architecture through evaluations at each stage and in the final results, showcasing the harmonized alignment of user-controlled object motion within a high-quality 3D background.
>
---
#### [new 156] LongLive: Real-time Interactive Long Video Generation
- **分类: cs.CV**

- **简介: 该论文提出LongLive，一种用于实时交互式长视频生成的帧级自回归框架。针对长视频生成中的效率和质量挑战，设计了KV-recache机制、流式长调优和帧级注意力池，实现了高质量、高效率的分钟级视频生成。**

- **链接: [http://arxiv.org/pdf/2509.22622v1](http://arxiv.org/pdf/2509.22622v1)**

> **作者:** Shuai Yang; Wei Huang; Ruihang Chu; Yicheng Xiao; Yuyang Zhao; Xianbang Wang; Muyang Li; Enze Xie; Yingcong Chen; Yao Lu; Song Han; Yukang Chen
>
> **备注:** Code, model, and demos are available at https://github.com/NVlabs/LongLive
>
> **摘要:** We present LongLive, a frame-level autoregressive (AR) framework for real-time and interactive long video generation. Long video generation presents challenges in both efficiency and quality. Diffusion and Diffusion-Forcing models can produce high-quality videos but suffer from low efficiency due to bidirectional attention. Causal attention AR models support KV caching for faster inference, but often degrade in quality on long videos due to memory challenges during long-video training. In addition, beyond static prompt-based generation, interactive capabilities, such as streaming prompt inputs, are critical for dynamic content creation, enabling users to guide narratives in real time. This interactive requirement significantly increases complexity, especially in ensuring visual consistency and semantic coherence during prompt transitions. To address these challenges, LongLive adopts a causal, frame-level AR design that integrates a KV-recache mechanism that refreshes cached states with new prompts for smooth, adherent switches; streaming long tuning to enable long video training and to align training and inference (train-long-test-long); and short window attention paired with a frame-level attention sink, shorten as frame sink, preserving long-range consistency while enabling faster generation. With these key designs, LongLive fine-tunes a 1.3B-parameter short-clip model to minute-long generation in just 32 GPU-days. At inference, LongLive sustains 20.7 FPS on a single NVIDIA H100, achieves strong performance on VBench in both short and long videos. LongLive supports up to 240-second videos on a single H100 GPU. LongLive further supports INT8-quantized inference with only marginal quality loss.
>
---
#### [new 157] Large AI Model-Enabled Generative Semantic Communications for Image Transmission
- **分类: cs.CV; cs.AI; cs.IT; math.IT**

- **简介: 该论文提出一种基于大AI模型的生成式语义通信系统，用于图像传输。针对现有方法忽视图像区域重要性差异的问题，通过区分关键与非关键区域进行差异化处理，并采用轻量化部署策略提升资源利用率。实验表明其在语义保真度和视觉质量上优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.21394v1](http://arxiv.org/pdf/2509.21394v1)**

> **作者:** Qiyu Ma; Wanli Ni; Zhijin Qin
>
> **备注:** Accepted to the IEEE GLOBECOM 2025
>
> **摘要:** The rapid development of generative artificial intelligence (AI) has introduced significant opportunities for enhancing the efficiency and accuracy of image transmission within semantic communication systems. Despite these advancements, existing methodologies often neglect the difference in importance of different regions of the image, potentially compromising the reconstruction quality of visually critical content. To address this issue, we introduce an innovative generative semantic communication system that refines semantic granularity by segmenting images into key and non-key regions. Key regions, which contain essential visual information, are processed using an image oriented semantic encoder, while non-key regions are efficiently compressed through an image-to-text modeling approach. Additionally, to mitigate the substantial storage and computational demands posed by large AI models, the proposed system employs a lightweight deployment strategy incorporating model quantization and low-rank adaptation fine-tuning techniques, significantly boosting resource utilization without sacrificing performance. Simulation results demonstrate that the proposed system outperforms traditional methods in terms of both semantic fidelity and visual quality, thereby affirming its effectiveness for image transmission tasks.
>
---
#### [new 158] EfficientDepth: A Fast and Detail-Preserving Monocular Depth Estimation Model
- **分类: cs.CV**

- **简介: 该论文提出EfficientDepth，用于单目深度估计任务。针对现有方法在几何一致性、细节保留和效率上的不足，结合Transformer与轻量解码器，引入双模密度头和LPIPS损失，提升深度图质量与计算效率。**

- **链接: [http://arxiv.org/pdf/2509.22527v1](http://arxiv.org/pdf/2509.22527v1)**

> **作者:** Andrii Litvynchuk; Ivan Livinsky; Anand Ravi; Nima Kalantari; Andrii Tsarov
>
> **备注:** 12 pages, 7 figures, 5 tables
>
> **摘要:** Monocular depth estimation (MDE) plays a pivotal role in various computer vision applications, such as robotics, augmented reality, and autonomous driving. Despite recent advancements, existing methods often fail to meet key requirements for 3D reconstruction and view synthesis, including geometric consistency, fine details, robustness to real-world challenges like reflective surfaces, and efficiency for edge devices. To address these challenges, we introduce a novel MDE system, called EfficientDepth, which combines a transformer architecture with a lightweight convolutional decoder, as well as a bimodal density head that allows the network to estimate detailed depth maps. We train our model on a combination of labeled synthetic and real images, as well as pseudo-labeled real images, generated using a high-performing MDE method. Furthermore, we employ a multi-stage optimization strategy to improve training efficiency and produce models that emphasize geometric consistency and fine detail. Finally, in addition to commonly used objectives, we introduce a loss function based on LPIPS to encourage the network to produce detailed depth maps. Experimental results demonstrate that EfficientDepth achieves performance comparable to or better than existing state-of-the-art models, with significantly reduced computational resources.
>
---
#### [new 159] DeHate: A Stable Diffusion-based Multimodal Approach to Mitigate Hate Speech in Images
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出DeHate，一种基于Stable Diffusion的多模态方法，用于识别并模糊图像中的仇恨内容。通过水印增强扩散技术和DAAM模块生成仇恨注意力图，解决在线图像中仇恨言论的问题，并发布相关数据集和共享任务。**

- **链接: [http://arxiv.org/pdf/2509.21787v1](http://arxiv.org/pdf/2509.21787v1)**

> **作者:** Dwip Dalal; Gautam Vashishtha; Anku Ranui; Aishwarya Reganti; Parth Patwa; Mohd Sarique; Chandan Gupta; Keshav Nath; Viswanatha Reddy; Vinija Jain; Aman Chadha; Amitava Das; Amit Sheth; Asif Ekbal
>
> **备注:** Defactify 3 workshop at AAAI 2024
>
> **摘要:** The rise in harmful online content not only distorts public discourse but also poses significant challenges to maintaining a healthy digital environment. In response to this, we introduce a multimodal dataset uniquely crafted for identifying hate in digital content. Central to our methodology is the innovative application of watermarked, stability-enhanced, stable diffusion techniques combined with the Digital Attention Analysis Module (DAAM). This combination is instrumental in pinpointing the hateful elements within images, thereby generating detailed hate attention maps, which are used to blur these regions from the image, thereby removing the hateful sections of the image. We release this data set as a part of the dehate shared task. This paper also describes the details of the shared task. Furthermore, we present DeHater, a vision-language model designed for multimodal dehatification tasks. Our approach sets a new standard in AI-driven image hate detection given textual prompts, contributing to the development of more ethical AI applications in social media.
>
---
#### [new 160] Perception-Consistency Multimodal Large Language Models Reasoning via Caption-Regularized Policy Optimization
- **分类: cs.MM; cs.CV; 68T07, 68T45; I.2.6; I.2.7; I.2.10**

- **简介: 该论文针对多模态大语言模型中视觉感知与推理不一致的问题，提出CapPO方法。通过引入基于描述的感知一致性正则化和KL加权优势估计，有效减少感知错误对推理的影响，提升数学和通用推理任务性能。**

- **链接: [http://arxiv.org/pdf/2509.21854v1](http://arxiv.org/pdf/2509.21854v1)**

> **作者:** Songjun Tu; Qichao Zhang; Jingbo Sun; Yuqian Fu; Linjing Li; Xiangyuan Lan; Dongmei Jiang; Yaowei Wang; Dongbin Zhao
>
> **备注:** 12pages, 11 figures
>
> **摘要:** While multimodal large language models excel at tasks that integrate visual perception with symbolic reasoning, their performance is often undermined by a critical vulnerability: perception-induced errors that propagate through the reasoning chain. Current reinforcement learning (RL) fine-tuning methods, while enhancing reasoning abilities, largely fail to address the underlying misalignment between visual grounding and the subsequent reasoning process. To address this challenge, we propose \textbf{Caption-Regularized Policy Optimization (CapPO)}, a novel RL framework that explicitly enforces perceptual consistency during policy optimization. CapPO integrates two key mechanisms: (1) a caption-based consistency regularization, which minimizes the divergence between responses conditioned on raw images and those conditioned on captions, thereby anchoring reasoning to semantically faithful visual content; and (2) a KL-weighted advantage estimation scheme, which adaptively scales reinforcement signals to strengthen perceptually consistent trajectories while suppressing spurious correlations. Extensive experiments on five math-focused and five general reasoning benchmarks demonstrate that CapPO achieves competitive performance, yielding gains of +6.0% accuracy on math-related tasks and +2.4% on general reasoning tasks over the base Qwen2.5-VL-7B model. Moreover, ablation studies further confirm the effectiveness of each component, while error analysis reveals that CapPO significantly reduces perception-related mistakes compared with baselines. Overall, CapPO provides a simple yet effective framework for improving multimodal reasoning.
>
---
#### [new 161] Visual Multi-Agent System: Mitigating Hallucination Snowballing via Visual Flow
- **分类: cs.MA; cs.CV**

- **简介: 该论文研究多智能体视觉语言系统中的“幻觉雪球效应”问题，提出ViF方法，通过选择视觉中继令牌和注意力重分配，减轻因过度依赖文本传递导致的视觉信息失真，有效提升多智能体系统的任务表现。**

- **链接: [http://arxiv.org/pdf/2509.21789v1](http://arxiv.org/pdf/2509.21789v1)**

> **作者:** Xinlei Yu; Chengming Xu; Guibin Zhang; Yongbo He; Zhangquan Chen; Zhucun Xue; Jiangning Zhang; Yue Liao; Xiaobin Hu; Yu-Gang Jiang; Shuicheng Yan
>
> **摘要:** Multi-Agent System (MAS) powered by Visual Language Models (VLMs) enables challenging tasks but suffers from a novel failure term, multi-agent visual hallucination snowballing, where hallucinations are seeded in a single agent and amplified by following ones due to the over-reliance on textual flow to relay visual information. Through turn-, layer-, and token-wise attention analyses, we provide detailed insights into the essence of hallucination snowballing regarding the reduction of visual attention allocation. It leads us to identify a subset of vision tokens with a unimodal attention peak in middle layers that best preserve visual evidence but gradually diminish in deeper agent turns, resulting in the visual hallucination snowballing in MAS. Thus, we propose ViF, a lightweight, plug-and-play mitigation paradigm that relays inter-agent messages with Visual Flow powered by the selected visual relay tokens and applies attention reallocation to amplify this pattern. The experiment results demonstrate that our method markedly reduces hallucination snowballing, consistently improving the performance across eight benchmarks based on four common MAS structures and ten base models. The source code will be available at: https://github.com/YU-deep/ViF.git.
>
---
#### [new 162] Enriching Knowledge Distillation with Intra-Class Contrastive Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于知识蒸馏任务，旨在解决学生模型未能充分利用教师模型软标签中类内多样性的问题。提出在教师训练中引入带边界损失的类内对比学习，以增强软标签的类内信息，提升学生模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.22053v1](http://arxiv.org/pdf/2509.22053v1)**

> **作者:** Hua Yuan; Ning Xu; Xin Geng; Yong Rui
>
> **摘要:** Since the advent of knowledge distillation, much research has focused on how the soft labels generated by the teacher model can be utilized effectively. Existing studies points out that the implicit knowledge within soft labels originates from the multi-view structure present in the data. Feature variations within samples of the same class allow the student model to generalize better by learning diverse representations. However, in existing distillation methods, teacher models predominantly adhere to ground-truth labels as targets, without considering the diverse representations within the same class. Therefore, we propose incorporating an intra-class contrastive loss during teacher training to enrich the intra-class information contained in soft labels. In practice, we find that intra-class loss causes instability in training and slows convergence. To mitigate these issues, margin loss is integrated into intra-class contrastive learning to improve the training stability and convergence speed. Simultaneously, we theoretically analyze the impact of this loss on the intra-class distances and inter-class distances. It has been proved that the intra-class contrastive loss can enrich the intra-class diversity. Experimental results demonstrate the effectiveness of the proposed method.
>
---
#### [new 163] ControlHair: Physically-based Video Diffusion for Controllable Dynamic Hair Rendering
- **分类: cs.GR; cs.CV; I.3; I.2; I.4**

- **简介: 该论文提出ControlHair，一个结合物理模拟与视频扩散模型的框架，用于可控动态头发渲染。任务是生成具有精确头发动态的视频，解决现有方法缺乏细粒度控制的问题。工作包括设计三阶段流水线，并在自建数据集上验证效果。**

- **链接: [http://arxiv.org/pdf/2509.21541v1](http://arxiv.org/pdf/2509.21541v1)**

> **作者:** Weikai Lin; Haoxiang Li; Yuhao Zhu
>
> **备注:** 9 pages,Project website: https://ctrlhair-arxiv.netlify.app/
>
> **摘要:** Hair simulation and rendering are challenging due to complex strand dynamics, diverse material properties, and intricate light-hair interactions. Recent video diffusion models can generate high-quality videos, but they lack fine-grained control over hair dynamics. We present ControlHair, a hybrid framework that integrates a physics simulator with conditional video diffusion to enable controllable dynamic hair rendering. ControlHair adopts a three-stage pipeline: it first encodes physics parameters (e.g., hair stiffness, wind) into per-frame geometry using a simulator, then extracts per-frame control signals, and finally feeds control signals into a video diffusion model to generate videos with desired hair dynamics. This cascaded design decouples physics reasoning from video generation, supports diverse physics, and makes training the video diffusion model easy. Trained on a curated 10K video dataset, ControlHair outperforms text- and pose-conditioned baselines, delivering precisely controlled hair dynamics. We further demonstrate three use cases of ControlHair: dynamic hairstyle try-on, bullet-time effects, and cinemagraphic. ControlHair introduces the first physics-informed video diffusion framework for controllable dynamics. We provide a teaser video and experimental results on our website.
>
---
#### [new 164] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics
- **分类: eess.IV; cs.CV; cs.LG; stat.AP; stat.ML**

- **简介: 该论文提出COMPASS，一种用于医学图像分割的鲁棒特征符合预测框架。针对分割模型下游指标（如器官大小）的不确定性量化问题，通过利用深度网络的表示空间进行高效校准，生成更紧致的置信区间，提升临床决策可靠性。**

- **链接: [http://arxiv.org/pdf/2509.22240v1](http://arxiv.org/pdf/2509.22240v1)**

> **作者:** Matt Y. Cheung; Ashok Veeraraghavan; Guha Balakrishnan
>
> **摘要:** In clinical applications, the utility of segmentation models is often based on the accuracy of derived downstream metrics such as organ size, rather than by the pixel-level accuracy of the segmentation masks themselves. Thus, uncertainty quantification for such metrics is crucial for decision-making. Conformal prediction (CP) is a popular framework to derive such principled uncertainty guarantees, but applying CP naively to the final scalar metric is inefficient because it treats the complex, non-linear segmentation-to-metric pipeline as a black box. We introduce COMPASS, a practical framework that generates efficient, metric-based CP intervals for image segmentation models by leveraging the inductive biases of their underlying deep neural networks. COMPASS performs calibration directly in the model's representation space by perturbing intermediate features along low-dimensional subspaces maximally sensitive to the target metric. We prove that COMPASS achieves valid marginal coverage under exchangeability and nestedness assumptions. Empirically, we demonstrate that COMPASS produces significantly tighter intervals than traditional CP baselines on four medical image segmentation tasks for area estimation of skin lesions and anatomical structures. Furthermore, we show that leveraging learned internal features to estimate importance weights allows COMPASS to also recover target coverage under covariate shifts. COMPASS paves the way for practical, metric-based uncertainty quantification for medical image segmentation.
>
---
#### [new 165] Rigidity-Aware 3D Gaussian Deformation from a Single Image
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文提出DeformSplat，旨在从单张图像中重建物体变形。针对现有方法依赖多视角视频的局限，提出了Gaussian-to-Pixel匹配和刚性区域分割技术，有效引导3D高斯变形，实现了更广泛适用的变形重建。**

- **链接: [http://arxiv.org/pdf/2509.22222v1](http://arxiv.org/pdf/2509.22222v1)**

> **作者:** Jinhyeok Kim; Jaehun Bang; Seunghyun Seo; Kyungdon Joo
>
> **备注:** 10 pages, 11 figures, conference
>
> **摘要:** Reconstructing object deformation from a single image remains a significant challenge in computer vision and graphics. Existing methods typically rely on multi-view video to recover deformation, limiting their applicability under constrained scenarios. To address this, we propose DeformSplat, a novel framework that effectively guides 3D Gaussian deformation from only a single image. Our method introduces two main technical contributions. First, we present Gaussian-to-Pixel Matching which bridges the domain gap between 3D Gaussian representations and 2D pixel observations. This enables robust deformation guidance from sparse visual cues. Second, we propose Rigid Part Segmentation consisting of initialization and refinement. This segmentation explicitly identifies rigid regions, crucial for maintaining geometric coherence during deformation. By combining these two techniques, our approach can reconstruct consistent deformations from a single image. Extensive experiments demonstrate that our approach significantly outperforms existing methods and naturally extends to various applications,such as frame interpolation and interactive object manipulation.
>
---
#### [new 166] Cross-Modal Retrieval with Cauchy-Schwarz Divergence
- **分类: cs.IR; cs.AI; cs.CV; cs.MM**

- **简介: 该论文研究跨模态检索任务，旨在解决现有方法在多模态对齐中的稳定性差、超参数敏感等问题。提出基于Cauchy-Schwarz散度（CS/GCS）的无超参数方法，实现双模态和三模态数据的有效对齐与检索。**

- **链接: [http://arxiv.org/pdf/2509.21339v1](http://arxiv.org/pdf/2509.21339v1)**

> **作者:** Jiahao Zhang; Wenzhe Yin; Shujian Yu
>
> **备注:** Accepted by ACMMM-25
>
> **摘要:** Effective cross-modal retrieval requires robust alignment of heterogeneous data types. Most existing methods focus on bi-modal retrieval tasks and rely on distributional alignment techniques such as Kullback-Leibler divergence, Maximum Mean Discrepancy, and correlation alignment. However, these methods often suffer from critical limitations, including numerical instability, sensitivity to hyperparameters, and their inability to capture the full structure of the underlying distributions. In this paper, we introduce the Cauchy-Schwarz (CS) divergence, a hyperparameter-free measure that improves both training stability and retrieval performance. We further propose a novel Generalized CS (GCS) divergence inspired by H\"older's inequality. This extension enables direct alignment of three or more modalities within a unified mathematical framework through a bidirectional circular comparison scheme, eliminating the need for exhaustive pairwise comparisons. Extensive experiments on six benchmark datasets demonstrate the effectiveness of our method in both bi-modal and tri-modal retrieval tasks. The code of our CS/GCS divergence is publicly available at https://github.com/JiahaoZhang666/CSD.
>
---
#### [new 167] Deep Learning-Based Cross-Anatomy CT Synthesis Using Adapted nnResU-Net with Anatomical Feature Prioritized Loss
- **分类: eess.IV; cs.AI; cs.CV; I.2; J.3**

- **简介: 该论文研究基于深度学习的跨模态CT图像合成任务，旨在解决MR/CBCT到CT图像转换中的解剖结构重建问题。作者采用改进的nnResUNet网络和AFP损失函数，结合多中心数据集进行训练，提升了骨结构和病灶区域的重建质量。**

- **链接: [http://arxiv.org/pdf/2509.22394v1](http://arxiv.org/pdf/2509.22394v1)**

> **作者:** Javier Sequeiro González; Arthur Longuefosse; Miguel Díaz Benito; Álvaro García Martín; Fabien Baldacci
>
> **摘要:** We present a patch-based 3D nnUNet adaptation for MR to CT and CBCT to CT image translation using the multicenter SynthRAD2025 dataset, covering head and neck (HN), thorax (TH), and abdomen (AB) regions. Our approach leverages two main network configurations: a standard UNet and a residual UNet, both adapted from nnUNet for image synthesis. The Anatomical Feature-Prioritized (AFP) loss was introduced, which compares multilayer features extracted from a compact segmentation network trained on TotalSegmentator labels, enhancing reconstruction of clinically relevant structures. Input volumes were normalized per-case using zscore normalization for MRIs, and clipping plus dataset level zscore normalization for CBCT and CT. Training used 3D patches tailored to each anatomical region without additional data augmentation. Models were trained for 1000 and 1500 epochs, with AFP fine-tuning performed for 500 epochs using a combined L1+AFP objective. During inference, overlapping patches were aggregated via mean averaging with step size of 0.3, and postprocessing included reverse zscore normalization. Both network configurations were applied across all regions, allowing consistent model design while capturing local adaptations through residual learning and AFP loss. Qualitative and quantitative evaluation revealed that residual networks combined with AFP yielded sharper reconstructions and improved anatomical fidelity, particularly for bone structures in MR to CT and lesions in CBCT to CT, while L1only networks achieved slightly better intensity-based metrics. This methodology provides a stable solution for cross modality medical image synthesis, demonstrating the effectiveness of combining the automatic nnUNet pipeline with residual learning and anatomically guided feature losses.
>
---
#### [new 168] Patch-Based Diffusion for Data-Efficient, Radiologist-Preferred MRI Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文研究MRI重建任务，旨在解决数据稀缺下高质量MRI重建的问题。提出PaDIS-MRI方法，基于小数据集训练补丁扩散模型，在图像质量和诊断性能上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.21531v1](http://arxiv.org/pdf/2509.21531v1)**

> **作者:** Rohan Sanda; Asad Aali; Andrew Johnston; Eduardo Reis; Jonathan Singh; Gordon Wetzstein; Sara Fridovich-Keil
>
> **备注:** Code is available at: https://github.com/voilalab/PaDIS-MRI
>
> **摘要:** Magnetic resonance imaging (MRI) requires long acquisition times, raising costs, reducing accessibility, and making scans more susceptible to motion artifacts. Diffusion probabilistic models that learn data-driven priors can potentially assist in reducing acquisition time. However, they typically require large training datasets that can be prohibitively expensive to collect. Patch-based diffusion models have shown promise in learning effective data-driven priors over small real-valued datasets, but have not yet demonstrated clinical value in MRI. We extend the Patch-based Diffusion Inverse Solver (PaDIS) to complex-valued, multi-coil MRI reconstruction, and compare it against a state-of-the-art whole-image diffusion baseline (FastMRI-EDM) for 7x undersampled MRI reconstruction on the FastMRI brain dataset. We show that PaDIS-MRI models trained on small datasets of as few as 25 k-space images outperform FastMRI-EDM on image quality metrics (PSNR, SSIM, NRMSE), pixel-level uncertainty, cross-contrast generalization, and robustness to severe k-space undersampling. In a blinded study with three radiologists, PaDIS-MRI reconstructions were chosen as diagnostically superior in 91.7% of cases, compared to baselines (i) FastMRI-EDM and (ii) classical convex reconstruction with wavelet sparsity. These findings highlight the potential of patch-based diffusion priors for high-fidelity MRI reconstruction in data-scarce clinical settings where diagnostic confidence matters.
>
---
#### [new 169] DistillKac: Few-Step Image Generation via Damped Wave Equations
- **分类: cs.LG; cs.AI; cs.CV; math.PR; stat.ML**

- **简介: 该论文提出DistillKac，一种基于阻尼波方程的快速图像生成方法。不同于扩散模型，其通过有限速度传播概率质量，提升了数值稳定性，并采用端点蒸馏策略，在少量步骤内生成高质量图像。**

- **链接: [http://arxiv.org/pdf/2509.21513v1](http://arxiv.org/pdf/2509.21513v1)**

> **作者:** Weiqiao Han; Chenlin Meng; Christopher D. Manning; Stefano Ermon
>
> **摘要:** We present DistillKac, a fast image generator that uses the damped wave equation and its stochastic Kac representation to move probability mass at finite speed. In contrast to diffusion models whose reverse time velocities can become stiff and implicitly allow unbounded propagation speed, Kac dynamics enforce finite speed transport and yield globally bounded kinetic energy. Building on this structure, we introduce classifier-free guidance in velocity space that preserves square integrability under mild conditions. We then propose endpoint only distillation that trains a student to match a frozen teacher over long intervals. We prove a stability result that promotes supervision at the endpoints to closeness along the entire path. Experiments demonstrate DistillKac delivers high quality samples with very few function evaluations while retaining the numerical stability benefits of finite speed probability flows.
>
---
#### [new 170] VISION: Prompting Ocean Vertical Velocity Reconstruction from Incomplete Observations
- **分类: cs.LG; cs.CV; physics.ao-ph**

- **简介: 该论文聚焦于从不完整的海洋观测中重建垂直速度场的问题，构建了高分辨率的KD48基准数据集，并提出了基于动态提示的VISION模型，有效应对数据缺失挑战，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.21477v1](http://arxiv.org/pdf/2509.21477v1)**

> **作者:** Yuan Gao; Hao Wu; Qingsong Wen; Kun Wang; Xian Wu; Xiaomeng Huang
>
> **摘要:** Reconstructing subsurface ocean dynamics, such as vertical velocity fields, from incomplete surface observations poses a critical challenge in Earth science, a field long hampered by the lack of standardized, analysis-ready benchmarks. To systematically address this issue and catalyze research, we first build and release KD48, a high-resolution ocean dynamics benchmark derived from petascale simulations and curated with expert-driven denoising. Building on this benchmark, we introduce VISION, a novel reconstruction paradigm based on Dynamic Prompting designed to tackle the core problem of missing data in real-world observations. The essence of VISION lies in its ability to generate a visual prompt on-the-fly from any available subset of observations, which encodes both data availability and the ocean's physical state. More importantly, we design a State-conditioned Prompting module that efficiently injects this prompt into a universal backbone, endowed with geometry- and scale-aware operators, to guide its adaptive adjustment of computational strategies. This mechanism enables VISION to precisely handle the challenges posed by varying input combinations. Extensive experiments on the KD48 benchmark demonstrate that VISION not only substantially outperforms state-of-the-art models but also exhibits strong generalization under extreme data missing scenarios. By providing a high-quality benchmark and a robust model, our work establishes a solid infrastructure for ocean science research under data uncertainty. Our codes are available at: https://github.com/YuanGao-YG/VISION.
>
---
#### [new 171] Guidance Watermarking for Diffusion Models
- **分类: cs.CR; cs.CV**

- **简介: 该论文提出一种针对扩散模型的水印嵌入方法，通过在扩散过程中利用水印解码器的梯度引导生成带水印的图像。无需重新训练模型即可增强抗攻击性，并保持生成图像的质量和多样性。**

- **链接: [http://arxiv.org/pdf/2509.22126v1](http://arxiv.org/pdf/2509.22126v1)**

> **作者:** Enoal Gesny; Eva Giboulot; Teddy Furon; Vivien Chappelier
>
> **摘要:** This paper introduces a novel watermarking method for diffusion models. It is based on guiding the diffusion process using the gradient computed from any off-the-shelf watermark decoder. The gradient computation encompasses different image augmentations, increasing robustness to attacks against which the decoder was not originally robust, without retraining or fine-tuning. Our method effectively convert any \textit{post-hoc} watermarking scheme into an in-generation embedding along the diffusion process. We show that this approach is complementary to watermarking techniques modifying the variational autoencoder at the end of the diffusion process. We validate the methods on different diffusion models and detectors. The watermarking guidance does not significantly alter the generated image for a given seed and prompt, preserving both the diversity and quality of generation.
>
---
#### [new 172] Are Hallucinations Bad Estimations?
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; stat.ML**

- **简介: 该论文研究生成模型中的幻觉问题，将其形式化为估计与合理原因的脱节。指出即使最优估计器也会产生幻觉，并通过理论和实验验证了这一现象，揭示了损失最小化与人类可接受输出之间的结构性错位。**

- **链接: [http://arxiv.org/pdf/2509.21473v1](http://arxiv.org/pdf/2509.21473v1)**

> **作者:** Hude Liu; Jerry Yao-Chieh Hu; Jennifer Yuntong Zhang; Zhao Song; Han Liu
>
> **备注:** Code is available at https://github.com/MAGICS-LAB/hallucination
>
> **摘要:** We formalize hallucinations in generative models as failures to link an estimate to any plausible cause. Under this interpretation, we show that even loss-minimizing optimal estimators still hallucinate. We confirm this with a general high probability lower bound on hallucinate rate for generic data distributions. This reframes hallucination as structural misalignment between loss minimization and human-acceptable outputs, and hence estimation errors induced by miscalibration. Experiments on coin aggregation, open-ended QA, and text-to-image support our theory.
>
---
#### [new 173] VoiceAssistant-Eval: Benchmarking AI Assistants across Listening, Speaking, and Viewing
- **分类: cs.CL; cs.AI; cs.CV; cs.HC; cs.SD**

- **简介: 该论文提出了VoiceAssistant-Eval，一个用于评估语音助手在听、说、视三方面能力的综合基准，包含10,497个任务示例。通过评估21个开源模型和GPT-4o-Audio，揭示了当前模型在音频理解、多模态处理等方面的不足，为下一代AI助手的研发提供了指导框架。**

- **链接: [http://arxiv.org/pdf/2509.22651v1](http://arxiv.org/pdf/2509.22651v1)**

> **作者:** Ke Wang; Houxing Ren; Zimu Lu; Mingjie Zhan; Hongsheng Li
>
> **摘要:** The growing capabilities of large language models and multimodal systems have spurred interest in voice-first AI assistants, yet existing benchmarks are inadequate for evaluating the full range of these systems' capabilities. We introduce VoiceAssistant-Eval, a comprehensive benchmark designed to assess AI assistants across listening, speaking, and viewing. VoiceAssistant-Eval comprises 10,497 curated examples spanning 13 task categories. These tasks include natural sounds, music, and spoken dialogue for listening; multi-turn dialogue, role-play imitation, and various scenarios for speaking; and highly heterogeneous images for viewing. To demonstrate its utility, we evaluate 21 open-source models and GPT-4o-Audio, measuring the quality of the response content and speech, as well as their consistency. The results reveal three key findings: (1) proprietary models do not universally outperform open-source models; (2) most models excel at speaking tasks but lag in audio understanding; and (3) well-designed smaller models can rival much larger ones. Notably, the mid-sized Step-Audio-2-mini (7B) achieves more than double the listening accuracy of LLaMA-Omni2-32B-Bilingual. However, challenges remain: multimodal (audio plus visual) input and role-play voice imitation tasks are difficult for current models, and significant gaps persist in robustness and safety alignment. VoiceAssistant-Eval identifies these gaps and establishes a rigorous framework for evaluating and guiding the development of next-generation AI assistants. Code and data will be released at https://mathllm.github.io/VoiceAssistantEval/ .
>
---
#### [new 174] RoboView-Bias: Benchmarking Visual Bias in Embodied Agents for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboView-Bias，首个系统量化机器人操作中视觉偏见的基准。针对现有基准缺乏对视觉偏见的系统评估问题，构建了2,127个任务实例，并验证了视觉因素对决策的影响，提出了缓解策略。**

- **链接: [http://arxiv.org/pdf/2509.22356v1](http://arxiv.org/pdf/2509.22356v1)**

> **作者:** Enguang Liu; Siyuan Liang; Liming Lu; Xiyu Zeng; Xiaochun Cao; Aishan Liu; Shuchao Pang
>
> **摘要:** The safety and reliability of embodied agents rely on accurate and unbiased visual perception. However, existing benchmarks mainly emphasize generalization and robustness under perturbations, while systematic quantification of visual bias remains scarce. This gap limits a deeper understanding of how perception influences decision-making stability. To address this issue, we propose RoboView-Bias, the first benchmark specifically designed to systematically quantify visual bias in robotic manipulation, following a principle of factor isolation. Leveraging a structured variant-generation framework and a perceptual-fairness validation protocol, we create 2,127 task instances that enable robust measurement of biases induced by individual visual factors and their interactions. Using this benchmark, we systematically evaluate three representative embodied agents across two prevailing paradigms and report three key findings: (i) all agents exhibit significant visual biases, with camera viewpoint being the most critical factor; (ii) agents achieve their highest success rates on highly saturated colors, indicating inherited visual preferences from underlying VLMs; and (iii) visual biases show strong, asymmetric coupling, with viewpoint strongly amplifying color-related bias. Finally, we demonstrate that a mitigation strategy based on a semantic grounding layer substantially reduces visual bias by approximately 54.5\% on MOKA. Our results highlight that systematic analysis of visual bias is a prerequisite for developing safe and reliable general-purpose embodied agents.
>
---
#### [new 175] Clinical Uncertainty Impacts Machine Learning Evaluations
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器学习评估任务，旨在解决临床数据标注不确定性影响模型评价的问题。作者提出使用概率度量方法处理标注不确定性，并呼吁公开原始标注以改进性能评估。**

- **链接: [http://arxiv.org/pdf/2509.22242v1](http://arxiv.org/pdf/2509.22242v1)**

> **作者:** Simone Lionetti; Fabian Gröger; Philippe Gottfrois; Alvaro Gonzalez-Jimenez; Ludovic Amruthalingam; Alexander A. Navarini; Marc Pouly
>
> **摘要:** Clinical dataset labels are rarely certain as annotators disagree and confidence is not uniform across cases. Typical aggregation procedures, such as majority voting, obscure this variability. In simple experiments on medical imaging benchmarks, accounting for the confidence in binary labels significantly impacts model rankings. We therefore argue that machine-learning evaluations should explicitly account for annotation uncertainty using probabilistic metrics that directly operate on distributions. These metrics can be applied independently of the annotations' generating process, whether modeled by simple counting, subjective confidence ratings, or probabilistic response models. They are also computationally lightweight, as closed-form expressions have linear-time implementations once examples are sorted by model score. We thus urge the community to release raw annotations for datasets and to adopt uncertainty-aware evaluation so that performance estimates may better reflect clinical data.
>
---
#### [new 176] SlimDiff: Training-Free, Activation-Guided Hands-free Slimming of Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出SlimDiff，一种无需训练的扩散模型结构压缩方法。针对扩散模型计算量大、参数多的问题，通过激活引导的动态剪枝，减少注意力与前馈维度，在不损失生成质量的前提下实现高效压缩，显著提升推理速度并降低参数量。**

- **链接: [http://arxiv.org/pdf/2509.21498v1](http://arxiv.org/pdf/2509.21498v1)**

> **作者:** Arani Roy; Shristi Das Biswas; Kaushik Roy
>
> **摘要:** Diffusion models (DMs), lauded for their generative performance, are computationally prohibitive due to their billion-scale parameters and iterative denoising dynamics. Existing efficiency techniques, such as quantization, timestep reduction, or pruning, offer savings in compute, memory, or runtime but are strictly bottlenecked by reliance on fine-tuning or retraining to recover performance. In this work, we introduce SlimDiff, an automated activation-informed structural compression framework that reduces both attention and feedforward dimensionalities in DMs, while being entirely gradient-free. SlimDiff reframes DM compression as a spectral approximation task, where activation covariances across denoising timesteps define low-rank subspaces that guide dynamic pruning under a fixed compression budget. This activation-aware formulation mitigates error accumulation across timesteps by applying module-wise decompositions over functional weight groups: query--key interactions, value--output couplings, and feedforward projections, rather than isolated matrix factorizations, while adaptively allocating sparsity across modules to respect the non-uniform geometry of diffusion trajectories. SlimDiff achieves up to 35\% acceleration and $\sim$100M parameter reduction over baselines, with generation quality on par with uncompressed models without any backpropagation. Crucially, our approach requires only about 500 calibration samples, over 70$\times$ fewer than prior methods. To our knowledge, this is the first closed-form, activation-guided structural compression of DMs that is entirely training-free, providing both theoretical clarity and practical efficiency.
>
---
#### [new 177] Aerial Path Planning for Urban Geometry and Texture Co-Capture
- **分类: cs.GR; cs.CV**

- **简介: 该论文研究城市场景的空中路径规划任务，旨在解决几何与纹理协同重建中纹理质量低的问题。提出了一种基于有限先验知识的路径规划框架和纹理质量评估方法，通过多目标优化生成高质量图像集，实现低成本高效重建。**

- **链接: [http://arxiv.org/pdf/2509.22227v1](http://arxiv.org/pdf/2509.22227v1)**

> **作者:** Weidan Xiong; Bochuan Zeng; Ziyu Hu; Jianwei Guo; Ke Xie; Hui Huang
>
> **备注:** ACM TOG and SIGGRAPH Asia 2025 (Patent Protected); Project page: https://vcc.tech/research/2025/DroneTex
>
> **摘要:** Recent advances in image acquisition and scene reconstruction have enabled the generation of high-quality structural urban scene geometry, given sufficient site information. However, current capture techniques often overlook the crucial importance of texture quality, resulting in noticeable visual artifacts in the textured models. In this work, we introduce the urban geometry and texture co-capture problem under limited prior knowledge before a site visit. The only inputs are a 2D building contour map of the target area and a safe flying altitude above the buildings. We propose an innovative aerial path planning framework designed to co-capture images for reconstructing both structured geometry and high-fidelity textures. To evaluate and guide view planning, we introduce a comprehensive texture quality assessment system, including two novel metrics tailored for building facades. Firstly, our method generates high-quality vertical dipping views and horizontal planar views to effectively capture both geometric and textural details. A multi-objective optimization strategy is then proposed to jointly maximize texture fidelity, improve geometric accuracy, and minimize the cost associated with aerial views. Furthermore, we present a sequential path planning algorithm that accounts for texture consistency during image capture. Extensive experiments on large-scale synthetic and real-world urban datasets demonstrate that our approach effectively produces image sets suitable for concurrent geometric and texture reconstruction, enabling the creation of realistic, textured scene proxies at low operational cost.
>
---
#### [new 178] TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出TRiCo，一种用于半监督学习的三元博弈协同训练框架。通过引入教师、两个学生和对抗生成器，解决伪标签不可靠、视图交互静态等问题，提升模型鲁棒性，在低标注数据下实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.21526v1](http://arxiv.org/pdf/2509.21526v1)**

> **作者:** Hongyang He; Xinyuan Song; Yangfan He; Zeyu Zhang; Yanshu Li; Haochen You; Lifan Sun; Wenqiao Zhang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** We introduce TRiCo, a novel triadic game-theoretic co-training framework that rethinks the structure of semi-supervised learning by incorporating a teacher, two students, and an adversarial generator into a unified training paradigm. Unlike existing co-training or teacher-student approaches, TRiCo formulates SSL as a structured interaction among three roles: (i) two student classifiers trained on frozen, complementary representations, (ii) a meta-learned teacher that adaptively regulates pseudo-label selection and loss balancing via validation-based feedback, and (iii) a non-parametric generator that perturbs embeddings to uncover decision boundary weaknesses. Pseudo-labels are selected based on mutual information rather than confidence, providing a more robust measure of epistemic uncertainty. This triadic interaction is formalized as a Stackelberg game, where the teacher leads strategy optimization and students follow under adversarial perturbations. By addressing key limitations in existing SSL frameworks, such as static view interactions, unreliable pseudo-labels, and lack of hard sample modeling, TRiCo provides a principled and generalizable solution. Extensive experiments on CIFAR-10, SVHN, STL-10, and ImageNet demonstrate that TRiCo consistently achieves state-of-the-art performance in low-label regimes, while remaining architecture-agnostic and compatible with frozen vision backbones.
>
---
#### [new 179] Adaptive Dual-Mode Distillation with Incentive Schemes for Scalable, Heterogeneous Federated Learning on Non-IID Data
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究联邦学习中的统计异质性和模型异质性问题，提出DL-SH、DL-MH和I-DL-MH方法，提升非IID数据下的模型性能与通信效率，并设计激励机制促进客户端参与。**

- **链接: [http://arxiv.org/pdf/2509.22507v1](http://arxiv.org/pdf/2509.22507v1)**

> **作者:** Zahid Iqbal
>
> **摘要:** Federated Learning (FL) has emerged as a promising decentralized learning (DL) approach that enables the use of distributed data without compromising user privacy. However, FL poses several key challenges. First, it is frequently assumed that every client can train the same machine learning models, however, not all clients are able to meet this assumption because of differences in their business needs and computational resources. Second, statistical heterogeneity (a.k.a. non-IID data) poses a major challenge in FL, which can lead to lower global model performance. Third, while addressing these challenges, there is a need for a cost-effective incentive mechanism to encourage clients to participate in FL training. In response to these challenges, we propose several methodologies: DL-SH, which facilitates efficient, privacy-preserving, and communication-efficient learning in the context of statistical heterogeneity; DL-MH, designed to manage fully heterogeneous models while tackling statistical disparities; and I-DL-MH, an incentive-based extension of DL-MH that promotes client engagement in federated learning training by providing incentives within this complex federated learning framework. Comprehensive experiments were carried out to assess the performance and scalability of the proposed approaches across a range of complex experimental settings. This involved utilizing various model architectures, in diverse data distributions, including IID and several non-IID scenarios, as well as multiple datasets. Experimental results demonstrate that the proposed approaches significantly enhance accuracy and decrease communication costs while effectively addressing statistical heterogeneity and model heterogeneity in comparison to existing state-of-the-art approaches and baselines, with DL-SH improving global model accuracy by 153%, and I-DL-MH achieving a 225% improvement under non-IID conditions.
>
---
#### [new 180] Comparative Analysis of GAN and Diffusion for MRI-to-CT translation
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文研究MRI到CT图像翻译任务，旨在解决生成高质量合成CT（sCT）的问题。比较了cGAN（Pix2Pix）和cDDPM（Palette）两种模型性能，并提出新评估指标SIMOS。实验表明，多通道输入和cDDPM架构效果更优。**

- **链接: [http://arxiv.org/pdf/2509.22049v1](http://arxiv.org/pdf/2509.22049v1)**

> **作者:** Emily Honey; Anders Helbo; Jens Petersen
>
> **摘要:** Computed tomography (CT) is essential for treatment and diagnostics; In case CT are missing or otherwise difficult to obtain, methods for generating synthetic CT (sCT) images from magnetic resonance imaging (MRI) images are sought after. Therefore, it is valuable to establish a reference for what strategies are most effective for MRI-to-CT translation. In this paper, we compare the performance of two frequently used architectures for MRI-to-CT translation: a conditional generative adversarial network (cGAN) and a conditional denoising diffusion probabilistic model (cDDPM). We chose well-established implementations to represent each architecture: Pix2Pix for cGAN, and Palette for cDDPM. We separate the classical 3D translation problem into a sequence of 2D translations on the transverse plane, to investigate the viability of a strategy that reduces the computational cost. We also investigate the impact of conditioning the generative process on a single MRI image/slice and on multiple MRI slices. The performance is assessed using a thorough evaluation protocol, including a novel slice-wise metric Similarity Of Slices (SIMOS), which measures the continuity between transverse slices when compiling the sCTs into 3D format. Our comparative analysis revealed that MRI-to-CT generative models benefit from multi-channel conditional input and using cDDPM as an architecture.
>
---
#### [new 181] Pixel Motion Diffusion is What We Need for Robot Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DAWN，一个基于扩散模型的统一框架，用于语言条件下的机器人操作任务。它通过结构化像素运动表示，连接高层意图与低层动作，实现端到端控制。实验在CALVIN和MetaWorld上展示了其优越的多任务性能和现实迁移能力。**

- **链接: [http://arxiv.org/pdf/2509.22652v1](http://arxiv.org/pdf/2509.22652v1)**

> **作者:** E-Ro Nguyen; Yichi Zhang; Kanchana Ranasinghe; Xiang Li; Michael S. Ryoo
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** We present DAWN (Diffusion is All We Need for robot control), a unified diffusion-based framework for language-conditioned robotic manipulation that bridges high-level motion intent and low-level robot action via structured pixel motion representation. In DAWN, both the high-level and low-level controllers are modeled as diffusion processes, yielding a fully trainable, end-to-end system with interpretable intermediate motion abstractions. DAWN achieves state-of-the-art results on the challenging CALVIN benchmark, demonstrating strong multi-task performance, and further validates its effectiveness on MetaWorld. Despite the substantial domain gap between simulation and reality and limited real-world data, we demonstrate reliable real-world transfer with only minimal finetuning, illustrating the practical viability of diffusion-based motion abstractions for robotic control. Our results show the effectiveness of combining diffusion modeling with motion-centric representations as a strong baseline for scalable and robust robot learning. Project page: https://nero1342.github.io/DAWN/
>
---
#### [new 182] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出JointDiff，一种统一生成连续轨迹与离散事件的扩散框架，用于多智能体轨迹生成任务。针对连续与离散过程分离的问题，设计了同步建模方法，并在体育领域验证其效果，实现可控生成和新基准数据集。**

- **链接: [http://arxiv.org/pdf/2509.22522v1](http://arxiv.org/pdf/2509.22522v1)**

> **作者:** Guillem Capellera; Luis Ferraz; Antonio Rubio; Alexandre Alahi; Antonio Agudo
>
> **摘要:** Generative models often treat continuous data and discrete events as separate processes, creating a gap in modeling complex systems where they interact synchronously. To bridge this gap, we introduce JointDiff, a novel diffusion framework designed to unify these two processes by simultaneously generating continuous spatio-temporal data and synchronous discrete events. We demonstrate its efficacy in the sports domain by simultaneously modeling multi-agent trajectories and key possession events. This joint modeling is validated with non-controllable generation and two novel controllable generation scenarios: weak-possessor-guidance, which offers flexible semantic control over game dynamics through a simple list of intended ball possessors, and text-guidance, which enables fine-grained, language-driven generation. To enable the conditioning with these guidance signals, we introduce CrossGuid, an effective conditioning operation for multi-agent domains. We also share a new unified sports benchmark enhanced with textual descriptions for soccer and football datasets. JointDiff achieves state-of-the-art performance, demonstrating that joint modeling is crucial for building realistic and controllable generative models for interactive systems.
>
---
#### [new 183] See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出See, Point, Fly（SPF），一种无需训练的视觉-语言导航框架，用于无人机在复杂环境中根据自然语言指令自主导航。其核心是将动作预测转化为2D空间定位任务，并结合3D位移控制无人机，实现了高效、通用的闭环导航。**

- **链接: [http://arxiv.org/pdf/2509.22653v1](http://arxiv.org/pdf/2509.22653v1)**

> **作者:** Chih Yao Hu; Yang-Sen Lin; Yuna Lee; Chih-Hai Su; Jie-Ying Lee; Shr-Ruei Tsai; Chin-Yang Lin; Kuan-Wen Chen; Tsung-Wei Ke; Yu-Lun Liu
>
> **备注:** CoRL 2025. Project page: https://spf-web.pages.dev
>
> **摘要:** We present See, Point, Fly (SPF), a training-free aerial vision-and-language navigation (AVLN) framework built atop vision-language models (VLMs). SPF is capable of navigating to any goal based on any type of free-form instructions in any kind of environment. In contrast to existing VLM-based approaches that treat action prediction as a text generation task, our key insight is to consider action prediction for AVLN as a 2D spatial grounding task. SPF harnesses VLMs to decompose vague language instructions into iterative annotation of 2D waypoints on the input image. Along with the predicted traveling distance, SPF transforms predicted 2D waypoints into 3D displacement vectors as action commands for UAVs. Moreover, SPF also adaptively adjusts the traveling distance to facilitate more efficient navigation. Notably, SPF performs navigation in a closed-loop control manner, enabling UAVs to follow dynamic targets in dynamic environments. SPF sets a new state of the art in DRL simulation benchmark, outperforming the previous best method by an absolute margin of 63%. In extensive real-world evaluations, SPF outperforms strong baselines by a large margin. We also conduct comprehensive ablation studies to highlight the effectiveness of our design choice. Lastly, SPF shows remarkable generalization to different VLMs. Project page: https://spf-web.pages.dev
>
---
#### [new 184] SGAligner++: Cross-Modal Language-Aided 3D Scene Graph Alignment
- **分类: cs.GR; cs.CV; cs.RO**

- **简介: 该论文提出SGAligner++，用于3D场景图对齐任务。针对现有方法依赖单一模态数据、处理噪声和低重叠场景效果差的问题，设计了一个跨模态语言辅助框架，通过联合嵌入空间提升对齐精度与鲁棒性，在真实数据集上表现优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.20401v1](http://arxiv.org/pdf/2509.20401v1)**

> **作者:** Binod Singh; Sayan Deb Sarkar; Iro Armeni
>
> **摘要:** Aligning 3D scene graphs is a crucial initial step for several applications in robot navigation and embodied perception. Current methods in 3D scene graph alignment often rely on single-modality point cloud data and struggle with incomplete or noisy input. We introduce SGAligner++, a cross-modal, language-aided framework for 3D scene graph alignment. Our method addresses the challenge of aligning partially overlapping scene observations across heterogeneous modalities by learning a unified joint embedding space, enabling accurate alignment even under low-overlap conditions and sensor noise. By employing lightweight unimodal encoders and attention-based fusion, SGAligner++ enhances scene understanding for tasks such as visual localization, 3D reconstruction, and navigation, while ensuring scalability and minimal computational overhead. Extensive evaluations on real-world datasets demonstrate that SGAligner++ outperforms state-of-the-art methods by up to 40% on noisy real-world reconstructions, while enabling cross-modal generalization.
>
---
#### [new 185] MINT-RVAE: Multi-Cues Intention Prediction of Human-Robot Interaction using Human Pose and Emotion Information from RGB-only Camera Data
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MINT-RVAE，用于基于RGB相机数据的人机交互意图预测任务。旨在解决真实场景中类别不平衡问题，通过合成序列生成和新损失函数提升模型性能与泛化能力，实现帧级精度的意图检测。**

- **链接: [http://arxiv.org/pdf/2509.22573v1](http://arxiv.org/pdf/2509.22573v1)**

> **作者:** Farida Mohsen; Ali Safa
>
> **摘要:** Efficiently detecting human intent to interact with ubiquitous robots is crucial for effective human-robot interaction (HRI) and collaboration. Over the past decade, deep learning has gained traction in this field, with most existing approaches relying on multimodal inputs, such as RGB combined with depth (RGB-D), to classify time-sequence windows of sensory data as interactive or non-interactive. In contrast, we propose a novel RGB-only pipeline for predicting human interaction intent with frame-level precision, enabling faster robot responses and improved service quality. A key challenge in intent prediction is the class imbalance inherent in real-world HRI datasets, which can hinder the model's training and generalization. To address this, we introduce MINT-RVAE, a synthetic sequence generation method, along with new loss functions and training strategies that enhance generalization on out-of-sample data. Our approach achieves state-of-the-art performance (AUROC: 0.95) outperforming prior works (AUROC: 0.90-0.912), while requiring only RGB input and supporting precise frame onset prediction. Finally, to support future research, we openly release our new dataset with frame-level labeling of human interaction intent.
>
---
#### [new 186] Language-in-the-Loop Culvert Inspection on the Erie Canal
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VISION系统，用于运河涵洞的自主检测。针对人工检测困难的问题，结合视觉-语言模型与路径规划，实现自动识别、定位并拍摄关键区域，提升检测效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.21370v1](http://arxiv.org/pdf/2509.21370v1)**

> **作者:** Yashom Dighe; Yash Turkar; Karthik Dantu
>
> **备注:** First two authors contributed equally
>
> **摘要:** Culverts on canals such as the Erie Canal, built originally in 1825, require frequent inspections to ensure safe operation. Human inspection of culverts is challenging due to age, geometry, poor illumination, weather, and lack of easy access. We introduce VISION, an end-to-end, language-in-the-loop autonomy system that couples a web-scale vision-language model (VLM) with constrained viewpoint planning for autonomous inspection of culverts. Brief prompts to the VLM solicit open-vocabulary ROI proposals with rationales and confidences, stereo depth is fused to recover scale, and a planner -- aware of culvert constraints -- commands repositioning moves to capture targeted close-ups. Deployed on a quadruped in a culvert under the Erie Canal, VISION closes the see, decide, move, re-image loop on-board and produces high-resolution images for detailed reporting without domain-specific fine-tuning. In an external evaluation by New York Canal Corporation personnel, initial ROI proposals achieved 61.4\% agreement with subject-matter experts, and final post-re-imaging assessments reached 80\%, indicating that VISION converts tentative hypotheses into grounded, expert-aligned findings.
>
---
#### [new 187] Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 该论文针对智能体在长期稀疏奖励任务中的探索-利用平衡问题，提出SPEAR方法。通过渐进式课程学习和自模仿学习，结合内在奖励与经验回放，稳定训练并提升LLM的工具使用能力。**

- **链接: [http://arxiv.org/pdf/2509.22601v1](http://arxiv.org/pdf/2509.22601v1)**

> **作者:** Yulei Qin; Xiaoyu Tan; Zhengbao He; Gang Li; Haojia Lin; Zongyi Li; Zihan Xu; Yuchen Shi; Siqi Cai; Renting Rui; Shaofei Cai; Yuzheng Cai; Xuan Zhang; Sheng Ye; Ke Li; Xing Sun
>
> **备注:** 26 pages, 11 figures
>
> **摘要:** Reinforcement learning (RL) is the dominant paradigm for sharpening strategic tool use capabilities of LLMs on long-horizon, sparsely-rewarded agent tasks, yet it faces a fundamental challenge of exploration-exploitation trade-off. Existing studies stimulate exploration through the lens of policy entropy, but such mechanical entropy maximization is prone to RL training instability due to the multi-turn distribution shifting. In this paper, we target the progressive exploration-exploitation balance under the guidance of the agent own experiences without succumbing to either entropy collapsing or runaway divergence. We propose SPEAR, a curriculum-based self-imitation learning (SIL) recipe for training agentic LLMs. It extends the vanilla SIL framework, where a replay buffer stores self-generated promising trajectories for off-policy update, by gradually steering the policy evolution within a well-balanced range of entropy across stages. Specifically, our approach incorporates a curriculum to manage the exploration process, utilizing intrinsic rewards to foster skill-level exploration and facilitating action-level exploration through SIL. At first, the auxiliary tool call reward plays a critical role in the accumulation of tool-use skills, enabling broad exposure to the unfamiliar distributions of the environment feedback with an upward entropy trend. As training progresses, self-imitation gets strengthened to exploit existing successful patterns from replayed experiences for comparative action-level exploration, accelerating solution iteration without unbounded entropy growth. To further stabilize training, we recalibrate the advantages of experiences in the replay buffer to address the potential policy drift. Reugularizations such as the clipping of tokens with high covariance between probability and advantage are introduced to the trajectory-level entropy control to curb over-confidence.
>
---
#### [new 188] WoW: Towards a World omniscient World model Through Embodied Interaction
- **分类: cs.RO; cs.CV; cs.MM**

- **简介: 该论文提出WoW，一个通过200万条机器人交互轨迹训练的140亿参数生成式世界模型，旨在解决视频模型缺乏物理因果理解的问题。通过SOPHIA机制约束生成结果，并构建逆动力学模型实现从想象到行动的闭环，最终在物理一致性与因果推理任务中取得SOTA表现。**

- **链接: [http://arxiv.org/pdf/2509.22642v1](http://arxiv.org/pdf/2509.22642v1)**

> **作者:** Xiaowei Chi; Peidong Jia; Chun-Kai Fan; Xiaozhu Ju; Weishi Mi; Kevin Zhang; Zhiyuan Qin; Wanxin Tian; Kuangzhi Ge; Hao Li; Zezhong Qian; Anthony Chen; Qiang Zhou; Yueru Jia; Jiaming Liu; Yong Dai; Qingpo Wuwu; Chengyu Bai; Yu-Kai Wang; Ying Li; Lizhang Chen; Yong Bao; Zhiyuan Jiang; Jiacheng Zhu; Kai Tang; Ruichuan An; Yulin Luo; Qiuxuan Feng; Siyuan Zhou; Chi-min Chan; Chengkai Hou; Wei Xue; Sirui Han; Yike Guo; Shanghang Zhang; Jian Tang
>
> **摘要:** Humans develop an understanding of intuitive physics through active interaction with the world. This approach is in stark contrast to current video models, such as Sora, which rely on passive observation and therefore struggle with grasping physical causality. This observation leads to our central hypothesis: authentic physical intuition of the world model must be grounded in extensive, causally rich interactions with the real world. To test this hypothesis, we present WoW, a 14-billion-parameter generative world model trained on 2 million robot interaction trajectories. Our findings reveal that the model's understanding of physics is a probabilistic distribution of plausible outcomes, leading to stochastic instabilities and physical hallucinations. Furthermore, we demonstrate that this emergent capability can be actively constrained toward physical realism by SOPHIA, where vision-language model agents evaluate the DiT-generated output and guide its refinement by iteratively evolving the language instructions. In addition, a co-trained Inverse Dynamics Model translates these refined plans into executable robotic actions, thus closing the imagination-to-action loop. We establish WoWBench, a new benchmark focused on physical consistency and causal reasoning in video, where WoW achieves state-of-the-art performance in both human and autonomous evaluation, demonstrating strong ability in physical causality, collision dynamics, and object permanence. Our work provides systematic evidence that large-scale, real-world interaction is a cornerstone for developing physical intuition in AI. Models, data, and benchmarks will be open-sourced.
>
---
#### [new 189] Closing the Oracle Gap: Increment Vector Transformation for Class Incremental Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对类增量学习（CIL）任务，旨在缩小与全数据训练模型的性能差距。提出IVT框架，通过维持低损失连接路径缓解灾难性遗忘，提升模型在新旧任务上的稳定性与准确性。**

- **链接: [http://arxiv.org/pdf/2509.21898v1](http://arxiv.org/pdf/2509.21898v1)**

> **作者:** Zihuan Qiu; Yi Xu; Fanman Meng; Runtong Zhang; Linfeng Xu; Qingbo Wu; Hongliang Li
>
> **摘要:** Class Incremental Learning (CIL) aims to sequentially acquire knowledge of new classes without forgetting previously learned ones. Despite recent progress, current CIL methods still exhibit significant performance gaps compared to their oracle counterparts-models trained with full access to historical data. Inspired by recent insights on Linear Mode Connectivity (LMC), we revisit the geometric properties of oracle solutions in CIL and uncover a fundamental observation: these oracle solutions typically maintain low-loss linear connections to the optimum of previous tasks. Motivated by this finding, we propose Increment Vector Transformation (IVT), a novel plug-and-play framework designed to mitigate catastrophic forgetting during training. Rather than directly following CIL updates, IVT periodically teleports the model parameters to transformed solutions that preserve linear connectivity to previous task optimum. By maintaining low-loss along these connecting paths, IVT effectively ensures stable performance on previously learned tasks. The transformation is efficiently approximated using diagonal Fisher Information Matrices, making IVT suitable for both exemplar-free and exemplar-based scenarios, and compatible with various initialization strategies. Extensive experiments on CIFAR-100, FGVCAircraft, ImageNet-Subset, and ImageNet-Full demonstrate that IVT consistently enhances the performance of strong CIL baselines. Specifically, on CIFAR-100, IVT improves the last accuracy of the PASS baseline by +5.12% and reduces forgetting by 2.54%. For the CLIP-pre-trained SLCA baseline on FGVCAircraft, IVT yields gains of +14.93% in average accuracy and +21.95% in last accuracy. The code will be released.
>
---
#### [new 190] Activation Function Design Sustains Plasticity in Continual Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究持续学习中的塑料性丧失问题，提出通过设计激活函数来维持模型适应能力。论文分析了激活函数的特性，提出了两种新的非线性函数，并在监督学习和强化学习任务中验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.22562v1](http://arxiv.org/pdf/2509.22562v1)**

> **作者:** Lute Lillo; Nick Cheney
>
> **摘要:** In independent, identically distributed (i.i.d.) training regimes, activation functions have been benchmarked extensively, and their differences often shrink once model size and optimization are tuned. In continual learning, however, the picture is different: beyond catastrophic forgetting, models can progressively lose the ability to adapt (referred to as loss of plasticity) and the role of the non-linearity in this failure mode remains underexplored. We show that activation choice is a primary, architecture-agnostic lever for mitigating plasticity loss. Building on a property-level analysis of negative-branch shape and saturation behavior, we introduce two drop-in nonlinearities (Smooth-Leaky and Randomized Smooth-Leaky) and evaluate them in two complementary settings: (i) supervised class-incremental benchmarks and (ii) reinforcement learning with non-stationary MuJoCo environments designed to induce controlled distribution and dynamics shifts. We also provide a simple stress protocol and diagnostics that link the shape of the activation to the adaptation under change. The takeaway is straightforward: thoughtful activation design offers a lightweight, domain-general way to sustain plasticity in continual learning without extra capacity or task-specific tuning.
>
---
## 更新

#### [replaced 001] Automated Facility Enumeration for Building Compliance Checking using Door Detection and Large Language Models
- **分类: cs.CV; cs.AI; cs.ET**

- **链接: [http://arxiv.org/pdf/2509.17283v2](http://arxiv.org/pdf/2509.17283v2)**

> **作者:** Licheng Zhang; Bach Le; Naveed Akhtar; Tuan Ngo
>
> **备注:** Author name correction in the second version (same content as the first version)
>
> **摘要:** Building compliance checking (BCC) is a critical process for ensuring that constructed facilities meet regulatory standards. A core component of BCC is the accurate enumeration of facility types and their spatial distribution. Despite its importance, this problem has been largely overlooked in the literature, posing a significant challenge for BCC and leaving a critical gap in existing workflows. Performing this task manually is time-consuming and labor-intensive. Recent advances in large language models (LLMs) offer new opportunities to enhance automation by combining visual recognition with reasoning capabilities. In this paper, we introduce a new task for BCC: automated facility enumeration, which involves validating the quantity of each facility type against statutory requirements. To address it, we propose a novel method that integrates door detection with LLM-based reasoning. We are the first to apply LLMs to this task and further enhance their performance through a Chain-of-Thought (CoT) pipeline. Our approach generalizes well across diverse datasets and facility types. Experiments on both real-world and synthetic floor plan data demonstrate the effectiveness and robustness of our method.
>
---
#### [replaced 002] Deep Learning for Clouds and Cloud Shadow Segmentation in Methane Satellite and Airborne Imaging Spectroscopy
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.19665v2](http://arxiv.org/pdf/2509.19665v2)**

> **作者:** Manuel Perez-Carrasco; Maya Nasr; Sebastien Roche; Chris Chan Miller; Zhan Zhang; Core Francisco Park; Eleanor Walker; Cecilia Garraffo; Douglas Finkbeiner; Ritesh Gautam; Steven Wofsy
>
> **摘要:** Effective cloud and cloud shadow detection is a critical prerequisite for accurate retrieval of concentrations of atmospheric methane or other trace gases in hyperspectral remote sensing. This challenge is especially pertinent for MethaneSAT and for its airborne companion mission, MethaneAIR. In this study, we use machine learning methods to address the cloud and cloud shadow detection problem for sensors with these high spatial resolutions instruments. Cloud and cloud shadows in remote sensing data need to be effectively screened out as they bias methane retrievals in remote sensing imagery and impact the quantification of emissions. We deploy and evaluate conventional techniques including Iterative Logistic Regression (ILR) and Multilayer Perceptron (MLP), with advanced deep learning architectures, namely UNet and a Spectral Channel Attention Network (SCAN) method. Our results show that conventional methods struggle with spatial coherence and boundary definition, affecting the detection of clouds and cloud shadows. Deep learning models substantially improve detection quality: UNet performs best in preserving spatial structure, while SCAN excels at capturing fine boundary details. Notably, SCAN surpasses UNet on MethaneSAT data, underscoring the benefits of incorporating spectral attention for satellite specific features. This in depth assessment of various disparate machine learning techniques demonstrates the strengths and effectiveness of advanced deep learning architectures in providing robust, scalable solutions for clouds and cloud shadow screening towards enhancing methane emission quantification capacity of existing and next generation hyperspectral missions. Our data and code is publicly available at https://doi.org/10.7910/DVN/IKLZOJ
>
---
#### [replaced 003] ReSpace: Text-Driven 3D Indoor Scene Synthesis and Editing with Preference Alignment
- **分类: cs.CV; I.2.10; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.02459v3](http://arxiv.org/pdf/2506.02459v3)**

> **作者:** Martin JJ. Bucher; Iro Armeni
>
> **备注:** 22 pages, 17 figures (incl. appendix)
>
> **摘要:** Scene synthesis and editing has emerged as a promising direction in computer graphics. Current trained approaches for 3D indoor scenes either oversimplify object semantics through one-hot class encodings (e.g., 'chair' or 'table'), require masked diffusion for editing, ignore room boundaries, or rely on floor plan renderings that fail to capture complex layouts. LLM-based methods enable richer semantics via natural language (e.g., 'modern studio with light wood furniture'), but lack editing functionality, are limited to rectangular layouts, or rely on weak spatial reasoning from implicit world models. We introduce ReSpace, a generative framework for text-driven 3D indoor scene synthesis and editing using autoregressive language models. Our approach features a compact structured scene representation with explicit room boundaries that enables asset-agnostic deployment and frames scene editing as a next-token prediction task. We leverage a dual-stage training approach combining supervised fine-tuning and preference alignment, enabling a specially trained language model for object addition that accounts for user instructions, spatial geometry, object semantics, and scene-level composition. For scene editing, we employ a zero-shot LLM to handle object removal and prompts for addition. We further introduce a voxelization-based evaluation capturing fine-grained geometry beyond 3D bounding boxes. Experimental results surpass state-of-the-art on addition and achieve superior human-perceived quality on full scene synthesis.
>
---
#### [replaced 004] DanceText: A Training-Free Layered Framework for Controllable Multilingual Text Transformation in Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14108v2](http://arxiv.org/pdf/2504.14108v2)**

> **作者:** Zhenyu Yu; Mohd Yamani Idna Idris; Hua Wang; Pei Wang; Rizwan Qureshi; Shaina Raza; Aman Chadha; Yong Xiang; Zhixiang Chen
>
> **摘要:** We present DanceText, a training-free framework for multilingual text editing in images, designed to support complex geometric transformations and achieve seamless foreground-background integration. While diffusion-based generative models have shown promise in text-guided image synthesis, they often lack controllability and fail to preserve layout consistency under non-trivial manipulations such as rotation, translation, scaling, and warping. To address these limitations, DanceText introduces a layered editing strategy that separates text from the background, allowing geometric transformations to be performed in a modular and controllable manner. A depth-aware module is further proposed to align appearance and perspective between the transformed text and the reconstructed background, enhancing photorealism and spatial consistency. Importantly, DanceText adopts a fully training-free design by integrating pretrained modules, allowing flexible deployment without task-specific fine-tuning. Extensive experiments on the AnyWord-3M benchmark demonstrate that our method achieves superior performance in visual quality, especially under large-scale and complex transformation scenarios. Code is avaible at https://github.com/YuZhenyuLindy/DanceText.git.
>
---
#### [replaced 005] Diverse Subset Selection via Norm-Based Sampling and Orthogonality
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.01086v2](http://arxiv.org/pdf/2406.01086v2)**

> **作者:** Noga Bar; Raja Giryes
>
> **摘要:** Large annotated datasets are crucial for the success of deep neural networks, but labeling data can be prohibitively expensive in domains such as medical imaging. This work tackles the subset selection problem: selecting a small set of the most informative examples from a large unlabeled pool for annotation. We propose a simple and effective method that combines feature norms, randomization, and orthogonality (via the Gram-Schmidt process) to select diverse and informative samples. Feature norms serve as a proxy for informativeness, while randomization and orthogonalization reduce redundancy and encourage coverage of the feature space. Extensive experiments on image and text benchmarks, including CIFAR-10/100, Tiny ImageNet, ImageNet, OrganAMNIST, and Yelp, show that our method consistently improves subset selection performance, both as a standalone approach and when integrated with existing techniques.
>
---
#### [replaced 006] TAPTRv3: Spatial and Temporal Context Foster Robust Tracking of Any Point in Long Video
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18671v2](http://arxiv.org/pdf/2411.18671v2)**

> **作者:** Jinyuan Qu; Hongyang Li; Shilong Liu; Tianhe Ren; Zhaoyang Zeng; Lei Zhang
>
> **摘要:** In this paper, built upon TAPTRv2, we present TAPTRv3. TAPTRv2 is a simple yet effective DETR-like point tracking framework that works fine in regular videos but tends to fail in long videos. TAPTRv3 improves TAPTRv2 by addressing its shortcomings in querying high-quality features from long videos, where the target tracking points normally undergo increasing variation over time. In TAPTRv3, we propose to utilize both spatial and temporal context to bring better feature querying along the spatial and temporal dimensions for more robust tracking in long videos. For better spatial feature querying, we identify that off-the-shelf attention mechanisms struggle with point-level tasks and present Context-aware Cross-Attention (CCA). CCA introduces spatial context into the attention mechanism to enhance the quality of attention scores when querying image features. For better temporal feature querying, we introduce Visibility-aware Long-Temporal Attention (VLTA), which conducts temporal attention over past frames while considering their corresponding visibilities. This effectively addresses the feature drifting problem in TAPTRv2 caused by its RNN-like long-term modeling. TAPTRv3 surpasses TAPTRv2 by a large margin on most of the challenging datasets and obtains state-of-the-art performance. Even when compared with methods trained on large-scale extra internal data, TAPTRv3 still demonstrates superiority.
>
---
#### [replaced 007] Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.22376v3](http://arxiv.org/pdf/2410.22376v3)**

> **作者:** Dongmin Park; Sebin Kim; Taehong Moon; Minkyu Kim; Kangwook Lee; Jaewoong Cho
>
> **备注:** ICLR 2025 (spotlight)
>
> **摘要:** State-of-the-art text-to-image (T2I) diffusion models often struggle to generate rare compositions of concepts, e.g., objects with unusual attributes. In this paper, we show that the compositional generation power of diffusion models on such rare concepts can be significantly enhanced by the Large Language Model (LLM) guidance. We start with empirical and theoretical analysis, demonstrating that exposing frequent concepts relevant to the target rare concepts during the diffusion sampling process yields more accurate concept composition. Based on this, we propose a training-free approach, R2F, that plans and executes the overall rare-to-frequent concept guidance throughout the diffusion inference by leveraging the abundant semantic knowledge in LLMs. Our framework is flexible across any pre-trained diffusion models and LLMs, and can be seamlessly integrated with the region-guided diffusion approaches. Extensive experiments on three datasets, including our newly proposed benchmark, RareBench, containing various prompts with rare compositions of concepts, R2F significantly surpasses existing models including SD3.0 and FLUX by up to 28.1%p in T2I alignment. Code is available at https://github.com/krafton-ai/Rare-to-Frequent.
>
---
#### [replaced 008] Investigating Redundancy in Multimodal Large Language Models with Multiple Vision Encoders
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.03262v2](http://arxiv.org/pdf/2507.03262v2)**

> **作者:** Yizhou Wang; Song Mao; Yang Chen; Yufan Shen; Yinqiao Yan; Pinlong Cai; Ding Wang; Guohang Yan; Zhi Yu; Xuming Hu; Botian Shi
>
> **摘要:** Recent multimodal large language models (MLLMs) increasingly integrate multiple vision encoders to improve performance on various benchmarks, assuming that diverse pretraining objectives yield complementary visual signals. However, we show this assumption often fails in practice. Through systematic encoder masking across representative multi encoder MLLMs, we find that performance typically degrades gracefully and sometimes even improves when selected encoders are masked, revealing pervasive encoder redundancy. To quantify this effect, we introduce two principled metrics: the Conditional Utilization Rate (CUR), which measures an encoders marginal contribution in the presence of others, and the Information Gap (IG), which captures heterogeneity in encoder utility within a model. Using these tools, we observe (i) strong specialization on tasks like OCR and Chart, where a single encoder can dominate with a CUR greater than 90%, (ii) high redundancy on general VQA and knowledge-based tasks, where encoders are largely interchangeable, (iii) instances of detrimental encoders with negative CUR. Notably, masking specific encoders can yield up to 16% higher accuracy on a specific task category and 3.6% overall performance boost compared to the full model.Furthermore, single and dual encoder variants recover over 90% of baseline on most non OCR tasks. Our analysis challenges the more encoders are better heuristic in MLLMs and provides actionable diagnostics for developing more efficient and effective multimodal architectures.
>
---
#### [replaced 009] Multimodal Recurrent Ensembles for Predicting Brain Responses to Naturalistic Movies (Algonauts 2025)
- **分类: q-bio.NC; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.17897v3](http://arxiv.org/pdf/2507.17897v3)**

> **作者:** Semih Eren; Deniz Kucukahmetler; Nico Scherf
>
> **备注:** 8 pages, 2 figures, 1 table. Invited report, CCN 2025 Algonauts Project session (3rd-place team). Code: https://github.com/erensemih/Algonauts2025_ModalityRNN v3: Added equal contribution footnote to author list
>
> **摘要:** Accurately predicting distributed cortical responses to naturalistic stimuli requires models that integrate visual, auditory and semantic information over time. We present a hierarchical multimodal recurrent ensemble that maps pretrained video, audio, and language embeddings to fMRI time series recorded while four subjects watched almost 80 hours of movies provided by the Algonauts 2025 challenge. Modality-specific bidirectional RNNs encode temporal dynamics; their hidden states are fused and passed to a second recurrent layer, and lightweight subject-specific heads output responses for 1000 cortical parcels. Training relies on a composite MSE-correlation loss and a curriculum that gradually shifts emphasis from early sensory to late association regions. Averaging 100 model variants further boosts robustness. The resulting system ranked third on the competition leaderboard, achieving an overall Pearson r = 0.2094 and the highest single-parcel peak score (mean r = 0.63) among all participants, with particularly strong gains for the most challenging subject (Subject 5). The approach establishes a simple, extensible baseline for future multimodal brain-encoding benchmarks.
>
---
#### [replaced 010] NarrLV: Towards a Comprehensive Narrative-Centric Evaluation for Long Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11245v3](http://arxiv.org/pdf/2507.11245v3)**

> **作者:** X. Feng; H. Yu; M. Wu; S. Hu; J. Chen; C. Zhu; J. Wu; X. Chu; K. Huang
>
> **备注:** Project Page: https://amap-ml.github.io/NarrLV-Website/
>
> **摘要:** With the rapid development of foundation video generation technologies, long video generation models have exhibited promising research potential thanks to expanded content creation space. Recent studies reveal that the goal of long video generation tasks is not only to extend video duration but also to accurately express richer narrative content within longer videos. However, due to the lack of evaluation benchmarks specifically designed for long video generation models, the current assessment of these models primarily relies on benchmarks with simple narrative prompts (e.g., VBench). To the best of our knowledge, our proposed NarrLV is the first benchmark to comprehensively evaluate the Narrative expression capabilities of Long Video generation models. Inspired by film narrative theory, (i) we first introduce the basic narrative unit maintaining continuous visual presentation in videos as Temporal Narrative Atom (TNA), and use its count to quantitatively measure narrative richness. Guided by three key film narrative elements influencing TNA changes, we construct an automatic prompt generation pipeline capable of producing evaluation prompts with a flexibly expandable number of TNAs. (ii) Then, based on the three progressive levels of narrative content expression, we design an effective evaluation metric using the MLLM-based question generation and answering framework. (iii) Finally, we conduct extensive evaluations on existing long video generation models and the foundation generation models. Experimental results demonstrate that our metric aligns closely with human judgments. The derived evaluation outcomes reveal the detailed capability boundaries of current video generation models in narrative content expression.
>
---
#### [replaced 011] Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17659v3](http://arxiv.org/pdf/2505.17659v3)**

> **作者:** Xiaolong Tang; Meina Kan; Shiguang Shan; Xilin Chen
>
> **摘要:** Safe and feasible trajectory planning is critical for real-world autonomous driving systems. However, existing learning-based planners rely heavily on expert demonstrations, which not only lack explicit safety awareness but also risk inheriting undesirable behaviors such as speeding from suboptimal human driving data. Inspired by the success of large language models, we propose Plan-R1, a two-stage trajectory planning framework that decouples principle alignment from behavior learning. In the first stage, a general trajectory predictor is pre-trained on expert data to capture diverse, human-like driving behaviors. In the second stage, the model is fine-tuned with rule-based rewards using Group Relative Policy Optimization (GRPO), explicitly aligning ego planning with principles such as safety, comfort, and traffic rule compliance. This two-stage paradigm retains human-like behaviors while enhancing safety awareness and discarding undesirable patterns from demonstrations. Furthermore, we identify a key limitation of directly applying GRPO to planning: group-wise normalization erases cross-group scale differences, causing rare, high-variance safety-violation groups to have similar advantages as abundant low-variance safe groups, thereby suppressing optimization for safety-critical objectives. To address this, we propose Variance-Decoupled GRPO (VD-GRPO), which replaces normalization with centering and fixed scaling to preserve absolute reward magnitudes, ensuring that safety-critical objectives remain dominant throughout training. Experiments on the nuPlan benchmark demonstrate that Plan-R1 significantly improves planning safety and feasibility, achieving state-of-the-art performance, particularly in realistic reactive settings. Our code is available at https://github.com/XiaolongTang23/Plan-R1.
>
---
#### [replaced 012] Surgical Vision World Model
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.02904v2](http://arxiv.org/pdf/2503.02904v2)**

> **作者:** Saurabh Koju; Saurav Bastola; Prashant Shrestha; Sanskar Amgain; Yash Raj Shrestha; Rudra P. K. Poudel; Binod Bhattarai
>
> **备注:** This paper has been accepted at the Data Engineering in Medical Imaging Workshop, MICCAI 2025
>
> **摘要:** Realistic and interactive surgical simulation has the potential to facilitate crucial applications, such as medical professional training and autonomous surgical agent training. In the natural visual domain, world models have enabled action-controlled data generation, demonstrating the potential to train autonomous agents in interactive simulated environments when large-scale real data acquisition is infeasible. However, such works in the surgical domain have been limited to simplified computer simulations, and lack realism. Furthermore, existing literature in world models has predominantly dealt with action-labeled data, limiting their applicability to real-world surgical data, where obtaining action annotation is prohibitively expensive. Inspired by the recent success of Genie in leveraging unlabeled video game data to infer latent actions and enable action-controlled data generation, we propose the first surgical vision world model. The proposed model can generate action-controllable surgical data and the architecture design is verified with extensive experiments on the unlabeled SurgToolLoc-2022 dataset. Codes and implementation details are available at https://github.com/bhattarailab/Surgical-Vision-World-Model
>
---
#### [replaced 013] Image Recognition with Online Lightweight Vision Transformer: A Survey
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03113v3](http://arxiv.org/pdf/2505.03113v3)**

> **作者:** Zherui Zhang; Rongtao Xu; Jie Zhou; Changwei Wang; Xingtian Pei; Wenhao Xu; Jiguang Zhang; Li Guo; Longxiang Gao; Wenbo Xu; Shibiao Xu
>
> **摘要:** The Transformer architecture has achieved significant success in natural language processing, motivating its adaptation to computer vision tasks. Unlike convolutional neural networks, vision transformers inherently capture long-range dependencies and enable parallel processing, yet lack inductive biases and efficiency benefits, facing significant computational and memory challenges that limit its real-world applicability. This paper surveys various online strategies for generating lightweight vision transformers for image recognition, focusing on three key areas: Efficient Component Design, Dynamic Network, and Knowledge Distillation. We evaluate the relevant exploration for each topic on the ImageNet-1K benchmark, analyzing trade-offs among precision, parameters, throughput, and more to highlight their respective advantages, disadvantages, and flexibility. Finally, we propose future research directions and potential challenges in the lightweighting of vision transformers with the aim of inspiring further exploration and providing practical guidance for the community. Project Page: https://github.com/ajxklo/Lightweight-VIT
>
---
#### [replaced 014] Think With Videos For Agentic Long-Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10821v3](http://arxiv.org/pdf/2506.10821v3)**

> **作者:** Huaying Yuan; Zheng Liu; Junjie Zhou; Hongjin Qian; Yan Shu; Nicu Sebe; Ji-Rong Wen; Zhicheng Dou
>
> **摘要:** Long-video understanding~(LVU) is a challenging problem in computer vision. Existing methods either downsample frames for single-pass reasoning, sacrificing fine-grained details, or depend on textual reasoning over task-agnostic representations, hindering task-specific perception and exploration. In this paper, we propose VideoExplorer, a framework grounded in the principle of ``thinking with video'', which naturally intertwines planning, temporal grounding, and scalable perception into a coherent reasoning process. Rather than reasoning over a static context, VideoExplorer iteratively formulates sub-questions, locates relevant moments, and performs task-oriented, temporally scalable video understanding until reaching the final answer, enabling faithful, efficient, and interpretable reasoning. To address the lack of LVU training resources, we construct a long-video reasoning dataset using difficulty-adaptive sampling to ensure high-quality trajectories on complex tasks. Building on this dataset, we design a two-stage training pipeline: supervised trajectory initialization followed by trajectory-level preference optimization, encouraging adaptive temporal grounding and iterative information integration guided by downstream rewards. Extensive evaluations on popular long-video understanding and reasoning benchmarks demonstrate VideoExplorer's significant advantage over existing baselines, highlighting its robustness, adaptability, and efficiency. Our code is made publicly available in this repository(https://github.com/yhy-2000/VideoDeepResearch).
>
---
#### [replaced 015] SIU3R: Simultaneous Scene Understanding and 3D Reconstruction Beyond Feature Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.02705v2](http://arxiv.org/pdf/2507.02705v2)**

> **作者:** Qi Xu; Dongxu Wei; Lingzhe Zhao; Wenpu Li; Zhangchi Huang; Shunping Ji; Peidong Liu
>
> **备注:** Accepted to NeurIPS'25 (Spotlight). Project page: https://insomniaaac.github.io/siu3r/
>
> **摘要:** Simultaneous understanding and 3D reconstruction plays an important role in developing end-to-end embodied intelligent systems. To achieve this, recent approaches resort to 2D-to-3D feature alignment paradigm, which leads to limited 3D understanding capability and potential semantic information loss. In light of this, we propose SIU3R, the first alignment-free framework for generalizable simultaneous understanding and 3D reconstruction from unposed images. Specifically, SIU3R bridges reconstruction and understanding tasks via pixel-aligned 3D representation, and unifies multiple understanding (segmentation) tasks into a set of unified learnable queries, enabling native 3D understanding without the need of alignment with 2D models. To encourage collaboration between the two tasks with shared representation, we further conduct in-depth analyses of their mutual benefits, and propose two lightweight modules to facilitate their interaction. Extensive experiments demonstrate that our method achieves state-of-the-art performance not only on the individual tasks of 3D reconstruction and understanding, but also on the task of simultaneous understanding and 3D reconstruction, highlighting the advantages of our alignment-free framework and the effectiveness of the mutual benefit designs. Project page: https://insomniaaac.github.io/siu3r/
>
---
#### [replaced 016] Intentional Gesture: Deliver Your Intentions with Gestures for Speech
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2505.15197v2](http://arxiv.org/pdf/2505.15197v2)**

> **作者:** Pinxin Liu; Haiyang Liu; Luchuan Song; Jason J. Corso; Chenliang Xu
>
> **摘要:** When humans speak, gestures help convey communicative intentions, such as adding emphasis or describing concepts. However, current co-speech gesture generation methods rely solely on superficial linguistic cues (e.g. speech audio or text transcripts), neglecting to understand and leverage the communicative intention that underpins human gestures. This results in outputs that are rhythmically synchronized with speech but are semantically shallow. To address this gap, we introduce Intentional-Gesture, a novel framework that casts gesture generation as an intention-reasoning task grounded in high-level communicative functions. First, we curate the InG dataset by augmenting BEAT-2 with gesture-intention annotations (i.e., text sentences summarizing intentions), which are automatically annotated using large vision-language models. Next, we introduce the Intentional Gesture Motion Tokenizer to leverage these intention annotations. It injects high-level communicative functions (e.g., intentions) into tokenized motion representations to enable intention-aware gesture synthesis that are both temporally aligned and semantically meaningful, achieving new state-of-the-art performance on the BEAT-2 benchmark. Our framework offers a modular foundation for expressive gesture generation in digital humans and embodied AI. Project Page: https://andypinxinliu.github.io/Intentional-Gesture
>
---
#### [replaced 017] pFedMMA: Personalized Federated Fine-Tuning with Multi-Modal Adapter for Vision-Language Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.05394v2](http://arxiv.org/pdf/2507.05394v2)**

> **作者:** Sajjad Ghiasvand; Mahnoosh Alizadeh; Ramtin Pedarsani
>
> **摘要:** Vision-Language Models (VLMs) like CLIP have demonstrated remarkable generalization in zero- and few-shot settings, but adapting them efficiently to decentralized, heterogeneous data remains a challenge. While prompt tuning has emerged as a popular parameter-efficient approach in personalized federated learning, existing methods often sacrifice generalization in favor of personalization, struggling particularly on unseen classes or domains. In this work, we propose pFedMMA, the first personalized federated learning framework that leverages multi-modal adapters for vision-language tasks. Each adapter contains modality-specific up- and down-projection layers alongside a globally shared projection that aligns cross-modal features. Our optimization strategy allows clients to locally adapt to personalized data distributions while collaboratively training the shared projection to improve global generalization. This design is also communication-efficient, as only the shared component is exchanged during communication rounds. Through extensive experiments across eleven datasets, including domain- and label-shift scenarios, we show that pFedMMA achieves state-of-the-art trade-offs between personalization and generalization, outperforming recent federated prompt tuning methods.
>
---
#### [replaced 018] Metric-Guided Conformal Bounds for Probabilistic Image Reconstruction
- **分类: cs.LG; cs.CV; eess.IV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2404.15274v4](http://arxiv.org/pdf/2404.15274v4)**

> **作者:** Matt Y Cheung; Tucker J Netherton; Laurence E Court; Ashok Veeraraghavan; Guha Balakrishnan
>
> **备注:** Accepted as Long Oral at UNSURE @ MICCAI 2025. 11 pages, 4 figures, 1 table, 2 algorithms. Code available at https://github.com/matthewyccheung/conformal-metric. Previously titled "Metric-guided Image Reconstruction Bounds via Conformal Prediction"
>
> **摘要:** Modern deep learning reconstruction algorithms generate impressively realistic scans from sparse inputs, but can often produce significant inaccuracies. This makes it difficult to provide statistically guaranteed claims about the true state of a subject from scans reconstructed by these algorithms. In this study, we propose a framework for computing provably valid prediction bounds on claims derived from probabilistic black-box image reconstruction algorithms. The key insights behind our framework are to represent reconstructed scans with a derived clinical metric of interest, and to calibrate bounds on the ground truth metric with conformal prediction (CP) using a prior calibration dataset. These bounds convey interpretable feedback about the subject's state, and can also be used to retrieve nearest-neighbor reconstructed scans for visual inspection. We demonstrate the utility of this framework on sparse-view computed tomography (CT) for fat mass quantification and radiotherapy planning tasks. Results show that our framework produces bounds with better semantical interpretation than conventional pixel-based bounding approaches. Furthermore, we can flag dangerous outlier reconstructions that look plausible but have statistically unlikely metric values.
>
---
#### [replaced 019] GLEAM: Learning to Match and Explain in Cross-View Geo-Localization
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.07450v2](http://arxiv.org/pdf/2509.07450v2)**

> **作者:** Xudong Lu; Zhi Zheng; Yi Wan; Yongxiang Yao; Annan Wang; Renrui Zhang; Panwang Xia; Qiong Wu; Qingyun Li; Weifeng Lin; Xiangyu Zhao; Peifeng Ma; Xue Yang; Hongsheng Li
>
> **备注:** 18 pages
>
> **摘要:** Cross-View Geo-Localization (CVGL) focuses on identifying correspondences between images captured from distinct perspectives of the same geographical location. However, existing CVGL approaches are typically restricted to a single view or modality, and their direct visual matching strategy lacks interpretability: they only determine whether two images correspond, without explaining the rationale behind the match. In this paper, we present GLEAM-C, a foundational CVGL model that unifies multiple views and modalities-including UAV imagery, street maps, panoramic views, and ground photographs-by aligning them exclusively with satellite imagery. Our framework enhances training efficiency through optimized implementation while achieving accuracy comparable to prior modality-specific CVGL models through a two-phase training strategy. Moreover, to address the lack of interpretability in traditional CVGL methods, we leverage the reasoning capabilities of multimodal large language models (MLLMs) to propose a new task, GLEAM-X, which combines cross-view correspondence prediction with explainable reasoning. To support this task, we construct a bilingual benchmark using GPT-4o and Doubao-1.5-Thinking-Vision-Pro to generate training and testing data. The test set is further refined through detailed human revision, enabling systematic evaluation of explainable cross-view reasoning and advancing transparency and scalability in geo-localization. Together, GLEAM-C and GLEAM-X form a comprehensive CVGL pipeline that integrates multi-modal, multi-view alignment with interpretable correspondence analysis, unifying accurate cross-view matching with explainable reasoning and advancing Geo-Localization by enabling models to better Explain And Match. Code and datasets used in this work will be made publicly accessible at https://github.com/Lucky-Lance/GLEAM.
>
---
#### [replaced 020] Recent Advancements in Microscopy Image Enhancement using Deep Learning: A Survey
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.15363v2](http://arxiv.org/pdf/2509.15363v2)**

> **作者:** Debasish Dutta; Neeharika Sonowal; Risheraj Barauh; Deepjyoti Chetia; Sanjib Kr Kalita
>
> **备注:** 7 pages, 3 figures and 1 table. 2024 IEEE International Conference on Computer Vision and Machine Intelligence (CVMI). IEEE, 2024
>
> **摘要:** Microscopy image enhancement plays a pivotal role in understanding the details of biological cells and materials at microscopic scales. In recent years, there has been a significant rise in the advancement of microscopy image enhancement, specifically with the help of deep learning methods. This survey paper aims to provide a snapshot of this rapidly growing state-of-the-art method, focusing on its evolution, applications, challenges, and future directions. The core discussions take place around the key domains of microscopy image enhancement of super-resolution, reconstruction, and denoising, with each domain explored in terms of its current trends and their practical utility of deep learning.
>
---
#### [replaced 021] VidCRAFT3: Camera, Object, and Lighting Control for Image-to-Video Generation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.07531v4](http://arxiv.org/pdf/2502.07531v4)**

> **作者:** Sixiao Zheng; Zimian Peng; Yanpeng Zhou; Yi Zhu; Hang Xu; Xiangru Huang; Yanwei Fu
>
> **摘要:** Controllable image-to-video (I2V) generation transforms a reference image into a coherent video guided by user-specified control signals. In content creation workflows, precise and simultaneous control over camera motion, object motion, and lighting direction enhances both accuracy and flexibility. However, existing approaches typically treat these control signals separately, largely due to the scarcity of datasets with high-quality joint annotations and mismatched control spaces across modalities. We present VidCRAFT3, a unified and flexible I2V framework that supports both independent and joint control over camera motion, object motion, and lighting direction by integrating three core components. Image2Cloud reconstructs a 3D point cloud from the reference image to enable precise camera motion control. ObjMotionNet encodes sparse object trajectories into multi-scale optical flow features to guide object motion. The Spatial Triple-Attention Transformer integrates lighting direction embeddings via parallel cross-attention. To address the scarcity of jointly annotated data, we curate the VideoLightingDirection (VLD) dataset of synthetic static-scene video clips with per-frame lighting-direction labels, and adopt a three-stage training strategy that enables robust learning without fully joint annotations. Extensive experiments show that VidCRAFT3 outperforms existing methods in control precision and visual coherence. Code and data will be released. Project page: https://sixiaozheng.github.io/VidCRAFT3/.
>
---
#### [replaced 022] Large Pre-Training Datasets Don't Always Guarantee Robustness after Fine-Tuning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.21582v3](http://arxiv.org/pdf/2410.21582v3)**

> **作者:** Jaedong Hwang; Brian Cheung; Zhang-Wei Hong; Akhilan Boopathy; Pulkit Agrawal; Ila Fiete
>
> **摘要:** Large-scale pretrained models are widely leveraged as foundations for learning new specialized tasks via fine-tuning, with the goal of maintaining the general performance of the model while allowing it to gain new skills. A valuable goal for all such models is robustness: the ability to perform well on out-of-distribution (OOD) tasks. We assess whether fine-tuning preserves the overall robustness of the pretrained model, and observed that models pretrained on large datasets exhibited strong catastrophic forgetting and loss of OOD generalization. To systematically assess robustness preservation in fine-tuned models, we propose the Robustness Inheritance Benchmark (ImageNet-RIB). The benchmark, which can be applied to any pretrained model, consists of a set of related but distinct OOD (downstream) tasks and involves fine-tuning on one of the OOD tasks in the set then testing on the rest. We find that though continual learning methods help, fine-tuning reduces robustness across pretrained models. Surprisingly, models pretrained on the largest and most diverse datasets (e.g., LAION-2B) exhibit both larger robustness losses and lower absolute robustness after fine-tuning on small datasets, relative to models pretrained on smaller datasets. These findings suggest that starting with the strongest foundation model is not necessarily the best approach for performance on specialist tasks. https://jd730.github.io/projects/ImageNet-RIB
>
---
#### [replaced 023] Physics-Guided Motion Loss for Video Generation Model
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.02244v2](http://arxiv.org/pdf/2506.02244v2)**

> **作者:** Bowen Xue; Giuseppe Claudio Guarnera; Shuang Zhao; Zahra Montazeri
>
> **摘要:** Current video diffusion models generate visually compelling content but often violate basic laws of physics, producing subtle artifacts like rubber-sheet deformations and inconsistent object motion. We introduce a frequency-domain physics prior that improves motion plausibility without modifying model architectures. Our method decomposes common rigid motions (translation, rotation, scaling) into lightweight spectral losses, requiring only 2.7% of frequency coefficients while preserving 97%+ of spectral energy. Applied to Open-Sora, MVDIT, and Hunyuan, our approach improves both motion accuracy and action recognition by ~11% on average on OpenVID-1M (relative), while maintaining visual quality. User studies show 74--83% preference for our physics-enhanced videos. It also reduces warping error by 22--37% (depending on the backbone) and improves temporal consistency scores. These results indicate that simple, global spectral cues are an effective drop-in regularizer for physically plausible motion in video diffusion.
>
---
#### [replaced 024] Dual Branch VideoMamba with Gated Class Token Fusion for Violence Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.03162v2](http://arxiv.org/pdf/2506.03162v2)**

> **作者:** Damith Chamalke Senadeera; Xiaoyun Yang; Shibo Li; Muhammad Awais; Dimitrios Kollias; Gregory Slabaugh
>
> **摘要:** The rapid proliferation of surveillance cameras has increased the demand for automated violence detection. While CNNs and Transformers have shown success in extracting spatio-temporal features, they struggle with long-term dependencies and computational efficiency. We propose Dual Branch VideoMamba with Gated Class Token Fusion (GCTF), an efficient architecture combining a dual-branch design and a state-space model (SSM) backbone where one branch captures spatial features, while the other focuses on temporal dynamics. The model performs continuous fusion via a gating mechanism between the branches to enhance the model's ability to detect violent activities even in challenging surveillance scenarios. We also present a new benchmark by merging RWF-2000, RLVS, SURV and VioPeru datasets in video violence detection, ensuring strict separation between training and testing sets. Experimental results demonstrate that our model achieves state-of-the-art performance on this benchmark and also on DVD dataset which is another novel dataset on video violence detection, offering an optimal balance between accuracy and computational efficiency, demonstrating the promise of SSMs for scalable, near real-time surveillance violence detection.
>
---
#### [replaced 025] Reasoning to Edit: Hypothetical Instruction-Based Image Editing with Visual Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01908v2](http://arxiv.org/pdf/2507.01908v2)**

> **作者:** Qingdong He; Xueqin Chen; Chaoyi Wang; Yanjie Pan; Xiaobin Hu; Zhenye Gan; Yabiao Wang; Chengjie Wang; Xiangtai Li; Jiangning Zhang
>
> **摘要:** Instruction-based image editing (IIE) has advanced rapidly with the success of diffusion models. However, existing efforts primarily focus on simple and explicit instructions to execute editing operations such as adding, deleting, moving, or swapping objects. They struggle to handle more complex implicit hypothetical instructions that require deeper reasoning to infer plausible visual changes and user intent. Additionally, current datasets provide limited support for training and evaluating reasoning-aware editing capabilities. Architecturally, these methods also lack mechanisms for fine-grained detail extraction that support such reasoning. To address these limitations, we propose Reason50K, a large-scale dataset specifically curated for training and evaluating hypothetical instruction reasoning image editing, along with ReasonBrain, a novel framework designed to reason over and execute implicit hypothetical instructions across diverse scenarios. Reason50K includes over 50K samples spanning four key reasoning scenarios: Physical, Temporal, Causal, and Story reasoning. ReasonBrain leverages Multimodal Large Language Models (MLLMs) for editing guidance generation and a diffusion model for image synthesis, incorporating a Fine-grained Reasoning Cue Extraction (FRCE) module to capture detailed visual and textual semantics essential for supporting instruction reasoning. To mitigate the semantic loss, we further introduce a Cross-Modal Enhancer (CME) that enables rich interactions between the fine-grained cues and MLLM-derived features. Extensive experiments demonstrate that ReasonBrain consistently outperforms state-of-the-art baselines on reasoning scenarios while exhibiting strong zero-shot generalization to conventional IIE tasks. Our dataset and code will be released publicly.
>
---
#### [replaced 026] DynamicControl: Adaptive Condition Selection for Improved Text-to-Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03255v3](http://arxiv.org/pdf/2412.03255v3)**

> **作者:** Qingdong He; Jinlong Peng; Pengcheng Xu; Boyuan Jiang; Xiaobin Hu; Donghao Luo; Yong Liu; Yabiao Wang; Chengjie Wang; Xiangtai Li; Jiangning Zhang
>
> **摘要:** To enhance the controllability of text-to-image diffusion models, current ControlNet-like models have explored various control signals to dictate image attributes. However, existing methods either handle conditions inefficiently or use a fixed number of conditions, which does not fully address the complexity of multiple conditions and their potential conflicts. This underscores the need for innovative approaches to manage multiple conditions effectively for more reliable and detailed image synthesis. To address this issue, we propose a novel framework, DynamicControl, which supports dynamic combinations of diverse control signals, allowing adaptive selection of different numbers and types of conditions. Our approach begins with a double-cycle controller that generates an initial real score sorting for all input conditions by leveraging pre-trained conditional generation models and discriminative models. This controller evaluates the similarity between extracted conditions and input conditions, as well as the pixel-level similarity with the source image. Then, we integrate a Multimodal Large Language Model (MLLM) to build an efficient condition evaluator. This evaluator optimizes the ordering of conditions based on the double-cycle controller's score ranking. Our method jointly optimizes MLLMs and diffusion models, utilizing MLLMs' reasoning capabilities to facilitate multi-condition text-to-image (T2I) tasks. The final sorted conditions are fed into a parallel multi-control adapter, which learns feature maps from dynamic visual conditions and integrates them to modulate ControlNet, thereby enhancing control over generated images. Through both quantitative and qualitative comparisons, DynamicControl demonstrates its superiority over existing methods in terms of controllability, generation quality and composability under various conditional controls.
>
---
#### [replaced 027] MOSS-ChatV: Reinforcement Learning with Process Reasoning Reward for Video Temporal Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.21113v2](http://arxiv.org/pdf/2509.21113v2)**

> **作者:** Sicheng Tao; Jungang Li; Yibo Yan; Junyan Zhang; Yubo Gao; Hanqian Li; ShuHang Xun; Yuxuan Fan; Hong Chen; Jianxiang He; Xuming Hu
>
> **摘要:** Video reasoning has emerged as a critical capability for multimodal large language models (MLLMs), requiring models to move beyond static perception toward coherent understanding of temporal dynamics in complex scenes. Yet existing MLLMs often exhibit process inconsistency, where intermediate reasoning drifts from video dynamics even when the final answer is correct, undermining interpretability and robustness. To address this issue, we introduce MOSS-ChatV, a reinforcement learning framework with a Dynamic Time Warping (DTW)-based process reward. This rule-based reward aligns reasoning traces with temporally grounded references, enabling efficient process supervision without auxiliary reward models. We further identify dynamic state prediction as a key measure of video reasoning and construct MOSS-Video, a benchmark with annotated reasoning traces, where the training split is used to fine-tune MOSS-ChatV and the held-out split is reserved for evaluation. MOSS-ChatV achieves 87.2\% on MOSS-Video (test) and improves performance on general video benchmarks such as MVBench and MMVU. The framework consistently yields gains across different architectures, including Qwen2.5-VL and Phi-2, confirming its broad applicability. Evaluations with GPT-4o-as-judge further show that MOSS-ChatV produces more consistent and stable reasoning traces.
>
---
#### [replaced 028] Multi-View Hypercomplex Learning for Breast Cancer Screening
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2204.05798v4](http://arxiv.org/pdf/2204.05798v4)**

> **作者:** Eleonora Lopez; Eleonora Grassucci; Danilo Comminiello
>
> **备注:** This paper has been submitted to Expert Systems with Applications
>
> **摘要:** Radiologists interpret mammography exams by jointly analyzing all four views, as correlations among them are crucial for accurate diagnosis. Recent methods employ dedicated fusion blocks to capture such dependencies, but these are often hindered by view dominance, training instability, and computational overhead. To address these challenges, we introduce multi-view hypercomplex learning, a novel learning paradigm for multi-view breast cancer classification based on parameterized hypercomplex neural networks (PHNNs). Thanks to hypercomplex algebra, our models intrinsically capture both intra- and inter-view relations. We propose PHResNets for two-view exams and two complementary four-view architectures: PHYBOnet, optimized for efficiency, and PHYSEnet, optimized for accuracy. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art multi-view models, while also generalizing across radiographic modalities and tasks such as disease classification from chest X-rays and multimodal brain tumor segmentation. Full code and pretrained models are available at https://github.com/ispamm/PHBreast.
>
---
#### [replaced 029] RelMap: Enhancing Online Map Construction with Class-Aware Spatial Relation and Semantic Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.21567v2](http://arxiv.org/pdf/2507.21567v2)**

> **作者:** Tianhui Cai; Yun Zhang; Zewei Zhou; Zhiyu Huang; Jiaqi Ma
>
> **摘要:** Online high-definition (HD) map construction is crucial for scaling autonomous driving systems. While Transformer-based methods have become prevalent in online HD map construction, most existing approaches overlook the inherent spatial dependencies and semantic relationships among map elements, which constrains their accuracy and generalization capabilities. To address this, we propose RelMap, an end-to-end framework that explicitly models both spatial relations and semantic priors to enhance online HD map construction. Specifically, we introduce a Class-aware Spatial Relation Prior, which explicitly encodes relative positional dependencies between map elements using a learnable class-aware relation encoder. Additionally, we design a Mixture-of-Experts-based Semantic Prior, which routes features to class-specific experts based on predicted class probabilities, refining instance feature decoding. RelMap is compatible with both single-frame and temporal perception backbones, achieving state-of-the-art performance on both the nuScenes and Argoverse 2 datasets.
>
---
#### [replaced 030] CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.19223v3](http://arxiv.org/pdf/2504.19223v3)**

> **作者:** Alexander Baumann; Leonardo Ayala; Silvia Seidlitz; Jan Sellner; Alexander Studier-Fischer; Berkin Özdemir; Lena Maier-Hein; Slobodan Ilic
>
> **摘要:** Spectral imaging offers promising applications across diverse domains, including medicine and urban scene understanding, and is already established as a critical modality in remote sensing. However, variability in channel dimensionality and captured wavelengths among spectral cameras impede the development of AI-driven methodologies, leading to camera-specific models with limited generalizability and inadequate cross-camera applicability. To address this bottleneck, we introduce CARL, a model for Camera-Agnostic Representation Learning across RGB, multispectral, and hyperspectral imaging modalities. To enable the conversion of a spectral image with any channel dimensionality to a camera-agnostic representation, we introduce a novel spectral encoder, featuring a self-attention-cross-attention mechanism, to distill salient spectral information into learned spectral representations. Spatio-spectral pre-training is achieved with a novel feature-based self-supervision strategy tailored to CARL. Large-scale experiments across the domains of medical imaging, autonomous driving, and satellite imaging demonstrate our model's unique robustness to spectral heterogeneity, outperforming on datasets with simulated and real-world cross-camera spectral variations. The scalability and versatility of the proposed approach position our model as a backbone for future spectral foundation models.
>
---
#### [replaced 031] Structure before the Machine: Input Space is the Prerequisite for Concepts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.08543v2](http://arxiv.org/pdf/2506.08543v2)**

> **作者:** Bowei Tian; Xuntao Lyu; Meng Liu; Hongyi Wang; Ang Li
>
> **备注:** arXiv admin note: text overlap with arXiv:2503.22720
>
> **摘要:** High-level representations have become a central focus in enhancing AI transparency and control, shifting attention from individual neurons or circuits to structured semantic directions that align with human-interpretable concepts. Motivated by the Linear Representation Hypothesis (LRH), we propose the Input-Space Linearity Hypothesis (ISLH), which posits that concept-aligned directions originate in the input space and are selectively amplified with increasing depth. We then introduce the Spectral Principal Path (SPP) framework, which formalizes how deep networks progressively distill linear representations along a small set of dominant spectral directions. Building on this framework, we further demonstrate the multimodal robustness of these representations in Vision-Language Models (VLMs). By bridging theoretical insights with empirical validation, this work advances a structured theory of representation formation in deep networks, paving the way for improving AI robustness, fairness, and transparency.
>
---
#### [replaced 032] Leveraging Model Guidance to Extract Training Data from Personalized Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.03039v3](http://arxiv.org/pdf/2410.03039v3)**

> **作者:** Xiaoyu Wu; Jiaru Zhang; Zhiwei Steven Wu
>
> **备注:** Accepted at the International Conference on Machine Learning (ICML) 2025
>
> **摘要:** Diffusion Models (DMs) have become powerful image generation tools, especially for few-shot fine-tuning where a pretrained DM is fine-tuned on a small image set to capture specific styles or objects. Many people upload these personalized checkpoints online, fostering communities such as Civitai and HuggingFace. However, model owners may overlook the data leakage risks when releasing fine-tuned checkpoints. Moreover, concerns regarding copyright violations arise when unauthorized data is used during fine-tuning. In this paper, we ask: "Can training data be extracted from these fine-tuned DMs shared online?" A successful extraction would present not only data leakage threats but also offer tangible evidence of copyright infringement. To answer this, we propose FineXtract, a framework for extracting fine-tuning data. Our method approximates fine-tuning as a gradual shift in the model's learned distribution -- from the original pretrained DM toward the fine-tuning data. By extrapolating the models before and after fine-tuning, we guide the generation toward high-probability regions within the fine-tuned data distribution. We then apply a clustering algorithm to extract the most probable images from those generated using this extrapolated guidance. Experiments on DMs fine-tuned with datasets including WikiArt, DreamBooth, and real-world checkpoints posted online validate the effectiveness of our method, extracting about 20% of fine-tuning data in most cases. The code is available https://github.com/Nicholas0228/FineXtract.
>
---
#### [replaced 033] Group Evidence Matters: Tiling-based Semantic Gating for Dense Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.10779v2](http://arxiv.org/pdf/2509.10779v2)**

> **作者:** Yilun Xiao
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Dense small objects in UAV imagery are often missed due to long-range viewpoints, occlusion, and clutter[cite: 5]. This paper presents a detector-agnostic post-processing framework that converts overlap-induced redundancy into group evidence[cite: 6]. Overlapping tiling first recovers low-confidence candidates[cite: 7]. A Spatial Gate (DBSCAN on box centroids) and a Semantic Gate (DBSCAN on ResNet-18 embeddings) then validates group evidence[cite: 7]. Validated groups receive controlled confidence reweighting before class-aware NMS fusion[cite: 8]. Experiments on VisDrone show a recall increase from 0.685 to 0.778 (+0.093) and a precision adjustment from 0.801 to 0.595, yielding F1=0.669[cite: 9]. Post-processing latency averages 0.095 s per image[cite: 10]. These results indicate recall-first, precision-trade-off behavior that benefits recall-sensitive applications such as far-field counting and monitoring[cite: 10]. Ablation confirms that tiling exposes missed objects, spatial clustering stabilizes geometry, semantic clustering enforces appearance coherence, and reweighting provides calibrated integration with the baseline[cite: 11]. The framework requires no retraining and integrates with modern detectors[cite: 12]. Future work will reduce semantic gating cost and extend the approach with temporal cues[cite: 13].
>
---
#### [replaced 034] iTACO: Interactable Digital Twins of Articulated Objects from Casually Captured RGBD Videos
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.08334v2](http://arxiv.org/pdf/2506.08334v2)**

> **作者:** Weikun Peng; Jun Lv; Cewu Lu; Manolis Savva
>
> **备注:** Project website can be found at https://3dlg-hcvc.github.io/video2articulation/
>
> **摘要:** Articulated objects are prevalent in daily life. Interactable digital twins of such objects have numerous applications in embodied AI and robotics. Unfortunately, current methods to digitize articulated real-world objects require carefully captured data, preventing practical, scalable, and generalizable acquisition. We focus on motion analysis and part-level segmentation of an articulated object from a casually captured RGBD video shot with a hand-held camera. A casually captured video of an interaction with an articulated object is easy to obtain at scale using smartphones. However, this setting is challenging due to simultaneous object and camera motion and significant occlusions as the person interacts with the object. To tackle these challenges, we introduce iTACO: a coarse-to-fine framework that infers joint parameters and segments movable parts of the object from a dynamic RGBD video. To evaluate our method under this new setting, we build a dataset of 784 videos containing 284 objects across 11 categories that is 20$\times$ larger than available in prior work. We then compare our approach with existing methods that also take video as input. Our experiments show that iTACO outperforms existing articulated object digital twin methods on both synthetic and real casually captured RGBD videos.
>
---
#### [replaced 035] STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.20470v3](http://arxiv.org/pdf/2405.20470v3)**

> **作者:** Jiuhong Xiao; Ning Zhang; Daniel Tortei; Giuseppe Loianno
>
> **备注:** 8 pages, 7 figures. Accepted for IEEE Robotics and Automation Letters
>
> **摘要:** Accurate geo-localization of Unmanned Aerial Vehicles (UAVs) is crucial for outdoor applications including search and rescue operations, power line inspections, and environmental monitoring. The vulnerability of Global Navigation Satellite Systems (GNSS) signals to interference and spoofing necessitates the development of additional robust localization methods for autonomous navigation. Visual Geo-localization (VG), leveraging onboard cameras and reference satellite maps, offers a promising solution for absolute localization. Specifically, Thermal Geo-localization (TG), which relies on image-based matching between thermal imagery with satellite databases, stands out by utilizing infrared cameras for effective nighttime localization. However, the efficiency and effectiveness of current TG approaches, are hindered by dense sampling on satellite maps and geometric noises in thermal query images. To overcome these challenges, we introduce STHN, a novel UAV thermal geo-localization approach that employs a coarse-to-fine deep homography estimation method. This method attains reliable thermal geo-localization within a 512-meter radius of the UAV's last known location even with a challenging 11% size ratio between thermal and satellite images, despite the presence of indistinct textures and self-similar patterns. We further show how our research significantly enhances UAV thermal geo-localization performance and robustness against geometric noises under low-visibility conditions in the wild. The code is made publicly available.
>
---
#### [replaced 036] MS-GS: Multi-Appearance Sparse-View 3D Gaussian Splatting in the Wild
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.15548v3](http://arxiv.org/pdf/2509.15548v3)**

> **作者:** Deming Li; Kaiwen Jiang; Yutao Tang; Ravi Ramamoorthi; Rama Chellappa; Cheng Peng
>
> **备注:** fixed typos
>
> **摘要:** In-the-wild photo collections often contain limited volumes of imagery and exhibit multiple appearances, e.g., taken at different times of day or seasons, posing significant challenges to scene reconstruction and novel view synthesis. Although recent adaptations of Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) have improved in these areas, they tend to oversmooth and are prone to overfitting. In this paper, we present MS-GS, a novel framework designed with Multi-appearance capabilities in Sparse-view scenarios using 3DGS. To address the lack of support due to sparse initializations, our approach is built on the geometric priors elicited from monocular depth estimations. The key lies in extracting and utilizing local semantic regions with a Structure-from-Motion (SfM) points anchored algorithm for reliable alignment and geometry cues. Then, to introduce multi-view constraints, we propose a series of geometry-guided supervision at virtual views in a fine-grained and coarse scheme to encourage 3D consistency and reduce overfitting. We also introduce a dataset and an in-the-wild experiment setting to set up more realistic benchmarks. We demonstrate that MS-GS achieves photorealistic renderings under various challenging sparse-view and multi-appearance conditions and outperforms existing approaches significantly across different datasets.
>
---
#### [replaced 037] HiPerformer: A High-Performance Global-Local Segmentation Model with Modular Hierarchical Fusion Strategy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20280v2](http://arxiv.org/pdf/2509.20280v2)**

> **作者:** Dayu Tan; Zhenpeng Xu; Yansen Su; Xin Peng; Chunhou Zheng; Weimin Zhong
>
> **摘要:** Both local details and global context are crucial in medical image segmentation, and effectively integrating them is essential for achieving high accuracy. However, existing mainstream methods based on CNN-Transformer hybrid architectures typically employ simple feature fusion techniques such as serial stacking, endpoint concatenation, or pointwise addition, which struggle to address the inconsistencies between features and are prone to information conflict and loss. To address the aforementioned challenges, we innovatively propose HiPerformer. The encoder of HiPerformer employs a novel modular hierarchical architecture that dynamically fuses multi-source features in parallel, enabling layer-wise deep integration of heterogeneous information. The modular hierarchical design not only retains the independent modeling capability of each branch in the encoder, but also ensures sufficient information transfer between layers, effectively avoiding the degradation of features and information loss that come with traditional stacking methods. Furthermore, we design a Local-Global Feature Fusion (LGFF) module to achieve precise and efficient integration of local details and global semantic information, effectively alleviating the feature inconsistency problem and resulting in a more comprehensive feature representation. To further enhance multi-scale feature representation capabilities and suppress noise interference, we also propose a Progressive Pyramid Aggregation (PPA) module to replace traditional skip connections. Experiments on eleven public datasets demonstrate that the proposed method outperforms existing segmentation techniques, demonstrating higher segmentation accuracy and robustness. The code is available at https://github.com/xzphappy/HiPerformer.
>
---
#### [replaced 038] Towards Scalable Language-Image Pre-training for 3D Medical Imaging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21862v2](http://arxiv.org/pdf/2505.21862v2)**

> **作者:** Chenhui Zhao; Yiwei Lyu; Asadur Chowdury; Edward Harake; Akhil Kondepudi; Akshay Rao; Xinhai Hou; Honglak Lee; Todd Hollon
>
> **摘要:** The scalability of current language-image pre-training for 3D medical imaging, such as CT and MRI, is constrained by the need for radiologists to manually curate raw clinical studies. In this work, we pioneer pre-training directly on uncurated studies, which both aligns more closely with the radiologist's workflow and provides a natural path to scalability. However, the unique structure of such data presents new challenges for existing model architectures, which were originally designed for 2D slices or single 3D scans. To address this, we introduce a novel hierarchical attention mechanism inspired by the intrinsic hierarchy of radiology data: slice, scan, and study. We denote our framework as Hierarchical attention for Language-Image Pre-training (HLIP). Trained on 220K studies with 3.13 million scans for brain MRI and 240K studies with 1.44 million scans for head CT, HLIP achieves state-of-the-art performance, e.g., +10.5% balanced ACC on the proposed publicly available brain MRI benchmark Pub-Brain-5; +8.3% and +1.7% macro AUC on head CT benchmarks CQ500 and RSNA, respectively. HLIP also exhibits strong generalizability on existing 3D medical language-image pre-training benchmarks, e.g., +4.3% macro AUC on the Rad-ChestCT benchmark when pre-trained on CT-RATE. These results demonstrate that, with HLIP, directly pre-training on uncurated clinical datasets is a scalable and effective direction for language-image pre-training in 3D medical imaging. The code is available at https://github.com/Zch0414/hlip.
>
---
#### [replaced 039] Efficient Multimodal Dataset Distillation via Generative Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.15472v2](http://arxiv.org/pdf/2509.15472v2)**

> **作者:** Zhenghao Zhao; Haoxuan Wang; Junyi Wu; Yuzhang Shang; Gaowen Liu; Yan Yan
>
> **摘要:** Dataset distillation aims to synthesize a small dataset from a large dataset, enabling the model trained on it to perform well on the original dataset. With the blooming of large language models and multimodal large language models, the importance of multimodal datasets, particularly image-text datasets, has grown significantly. However, existing multimodal dataset distillation methods are constrained by the Matching Training Trajectories algorithm, which significantly increases the computing resource requirement, and takes days to process the distillation. In this work, we introduce EDGE, a generative distillation method for efficient multimodal dataset distillation. Specifically, we identify two key challenges of distilling multimodal datasets with generative models: 1) The lack of correlation between generated images and captions. 2) The lack of diversity among generated samples. To address the aforementioned issues, we propose a novel generative model training workflow with a bi-directional contrastive loss and a diversity loss. Furthermore, we propose a caption synthesis strategy to further improve text-to-image retrieval performance by introducing more text information. Our method is evaluated on Flickr30K, COCO, and CC3M datasets, demonstrating superior performance and efficiency compared to existing approaches. Notably, our method achieves results 18x faster than the state-of-the-art method.
>
---
#### [replaced 040] DVD-Quant: Data-free Video Diffusion Transformers Quantization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18663v2](http://arxiv.org/pdf/2505.18663v2)**

> **作者:** Zhiteng Li; Hanxuan Li; Junyi Wu; Kai Liu; Haotong Qin; Linghe Kong; Guihai Chen; Yulun Zhang; Xiaokang Yang
>
> **备注:** Code and models will be available at https://github.com/lhxcs/DVD-Quant
>
> **摘要:** Diffusion Transformers (DiTs) have emerged as the state-of-the-art architecture for video generation, yet their computational and memory demands hinder practical deployment. While post-training quantization (PTQ) presents a promising approach to accelerate Video DiT models, existing methods suffer from two critical limitations: (1) dependence on computation-heavy and inflexible calibration procedures, and (2) considerable performance deterioration after quantization. To address these challenges, we propose DVD-Quant, a novel Data-free quantization framework for Video DiTs. Our approach integrates three key innovations: (1) Bounded-init Grid Refinement (BGR) and (2) Auto-scaling Rotated Quantization (ARQ) for calibration data-free quantization error reduction, as well as (3) $\delta$-Guided Bit Switching ($\delta$-GBS) for adaptive bit-width allocation. Extensive experiments across multiple video generation benchmarks demonstrate that DVD-Quant achieves an approximately 2$\times$ speedup over full-precision baselines on advanced DiT models while maintaining visual fidelity. Notably, DVD-Quant is the first to enable W4A4 PTQ for Video DiTs without compromising video quality. Code and models will be available at https://github.com/lhxcs/DVD-Quant.
>
---
#### [replaced 041] FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20295v2](http://arxiv.org/pdf/2509.20295v2)**

> **作者:** Xichen Xu; Yanshu Wang; Jinbao Wang; Xiaoning Lei; Guoyang Xie; Guannan Jiang; Zhichao Lu
>
> **摘要:** Industrial anomaly segmentation relies heavily on pixel-level annotations, yet real-world anomalies are often scarce, diverse, and costly to label. Segmentation-oriented industrial anomaly synthesis (SIAS) has emerged as a promising alternative; however, existing methods struggle to balance sampling efficiency and generation quality. Moreover, most approaches treat all spatial regions uniformly, overlooking the distinct statistical differences between anomaly and background areas. This uniform treatment hinders the synthesis of controllable, structure-specific anomalies tailored for segmentation tasks. In this paper, we propose FAST, a foreground-aware diffusion framework featuring two novel modules: the Anomaly-Informed Accelerated Sampling (AIAS) and the Foreground-Aware Reconstruction Module (FARM). AIAS is a training-free sampling algorithm specifically designed for segmentation-oriented industrial anomaly synthesis, which accelerates the reverse process through coarse-to-fine aggregation and enables the synthesis of state-of-the-art segmentation-oriented anomalies in as few as 10 steps. Meanwhile, FARM adaptively adjusts the anomaly-aware noise within the masked foreground regions at each sampling step, preserving localized anomaly signals throughout the denoising trajectory. Extensive experiments on multiple industrial benchmarks demonstrate that FAST consistently outperforms existing anomaly synthesis methods in downstream segmentation tasks. We release the code at: https://github.com/Chhro123/fast-foreground-aware-anomaly-synthesis.
>
---
#### [replaced 042] Single-weight Model Editing for Post-hoc Spurious Correlation Neutralization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.14182v2](http://arxiv.org/pdf/2501.14182v2)**

> **作者:** Shahin Hakemi; Naveed Akhtar; Ghulam Mubashar Hassan; Ajmal Mian
>
> **摘要:** Neural network training tends to exploit the simplest features as shortcuts to greedily minimize training loss. However, some of these features might be spuriously correlated with the target labels, leading to incorrect predictions by the model. Several methods have been proposed to address this issue. Focusing on suppressing the spurious correlations with model training, they not only incur additional training cost, but also have limited practical utility as the model misbehavior due to spurious relations is usually discovered after its deployment. It is also often overlooked that spuriousness is a subjective notion. Hence, the precise questions that must be investigated are; to what degree a feature is spurious, and how we can proportionally distract the model's attention from it for reliable prediction. To this end, we propose a method that enables post-hoc neutralization of spurious feature impact, controllable to an arbitrary degree. We conceptualize spurious features as fictitious sub-classes within the original classes, which can be eliminated by a class removal scheme. We then propose a unique precise class removal technique that makes a single-weight modification, which entails negligible performance compromise for the remaining classes. We perform extensive experiments, demonstrating that by editing just a single weight in a post-hoc manner, our method achieves highly competitive, or better performance against the state-of-the-art methods.
>
---
#### [replaced 043] GLip: A Global-Local Integrated Progressive Framework for Robust Visual Speech Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.16031v2](http://arxiv.org/pdf/2509.16031v2)**

> **作者:** Tianyue Wang; Shuang Yang; Shiguang Shan; Xilin Chen
>
> **摘要:** Visual speech recognition (VSR), also known as lip reading, is the task of recognizing speech from silent video. Despite significant advancements in VSR over recent decades, most existing methods pay limited attention to real-world visual challenges such as illumination variations, occlusions, blurring, and pose changes. To address these challenges, we propose GLip, a Global-Local Integrated Progressive framework designed for robust VSR. GLip is built upon two key insights: (i) learning an initial coarse alignment between visual features across varying conditions and corresponding speech content facilitates the subsequent learning of precise visual-to-speech mappings in challenging environments; (ii) under adverse conditions, certain local regions (e.g., non-occluded areas) often exhibit more discriminative cues for lip reading than global features. To this end, GLip introduces a dual-path feature extraction architecture that integrates both global and local features within a two-stage progressive learning framework. In the first stage, the model learns to align both global and local visual features with corresponding acoustic speech units using easily accessible audio-visual data, establishing a coarse yet semantically robust foundation. In the second stage, we introduce a Contextual Enhancement Module (CEM) to dynamically integrate local features with relevant global context across both spatial and temporal dimensions, refining the coarse representations into precise visual-speech mappings. Our framework uniquely exploits discriminative local regions through a progressive learning strategy, demonstrating enhanced robustness against various visual challenges and consistently outperforming existing methods on the LRS2 and LRS3 benchmarks. We further validate its effectiveness on a newly introduced challenging Mandarin dataset.
>
---
#### [replaced 044] Excavating in the Wild: The GOOSE-Ex Dataset for Semantic Segmentation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.18788v2](http://arxiv.org/pdf/2409.18788v2)**

> **作者:** Raphael Hagmanns; Peter Mortimer; Miguel Granero; Thorsten Luettel; Janko Petereit
>
> **备注:** Accepted for publication at 2025 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** The successful deployment of deep learning-based techniques for autonomous systems is highly dependent on the data availability for the respective system in its deployment environment. Especially for unstructured outdoor environments, very few datasets exist for even fewer robotic platforms and scenarios. In an earlier work, we presented the German Outdoor and Offroad Dataset (GOOSE) framework along with 10000 multimodal frames from an offroad vehicle to enhance the perception capabilities in unstructured environments. In this work, we address the generalizability of the GOOSE framework. To accomplish this, we open-source the GOOSE-Ex dataset, which contains additional 5000 labeled multimodal frames from various completely different environments, recorded on a robotic excavator and a quadruped platform. We perform a comprehensive analysis of the semantic segmentation performance on different platforms and sensor modalities in unseen environments. In addition, we demonstrate how the combined datasets can be utilized for different downstream applications or competitions such as offroad navigation, object manipulation or scene completion. The dataset, its platform documentation and pre-trained state-of-the-art models for offroad perception will be made available on https://goose-dataset.de/. \
>
---
#### [replaced 045] TempFlow-GRPO: When Timing Matters for GRPO in Flow Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04324v3](http://arxiv.org/pdf/2508.04324v3)**

> **作者:** Xiaoxuan He; Siming Fu; Yuke Zhao; Wanli Li; Jian Yang; Dacheng Yin; Fengyun Rao; Bo Zhang
>
> **摘要:** Recent flow matching models for text-to-image generation have achieved remarkable quality, yet their integration with reinforcement learning for human preference alignment remains suboptimal, hindering fine-grained reward-based optimization. We observe that the key impediment to effective GRPO training of flow models is the temporal uniformity assumption in existing approaches: sparse terminal rewards with uniform credit assignment fail to capture the varying criticality of decisions across generation timesteps, resulting in inefficient exploration and suboptimal convergence. To remedy this shortcoming, we introduce \textbf{TempFlow-GRPO} (Temporal Flow GRPO), a principled GRPO framework that captures and exploits the temporal structure inherent in flow-based generation. TempFlow-GRPO introduces three key innovations: (i) a trajectory branching mechanism that provides process rewards by concentrating stochasticity at designated branching points, enabling precise credit assignment without requiring specialized intermediate reward models; (ii) a noise-aware weighting scheme that modulates policy optimization according to the intrinsic exploration potential of each timestep, prioritizing learning during high-impact early stages while ensuring stable refinement in later phases; and (iii) a seed group strategy that controls for initialization effects to isolate exploration contributions. These innovations endow the model with temporally-aware optimization that respects the underlying generative dynamics, leading to state-of-the-art performance in human preference alignment and text-to-image benchmarks.
>
---
#### [replaced 046] Calibrated Multi-Preference Optimization for Aligning Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02588v3](http://arxiv.org/pdf/2502.02588v3)**

> **作者:** Kyungmin Lee; Xiaohang Li; Qifei Wang; Junfeng He; Junjie Ke; Ming-Hsuan Yang; Irfan Essa; Jinwoo Shin; Feng Yang; Yinxiao Li
>
> **备注:** CVPR 2025, Project page: https://kyungmnlee.github.io/capo.github.io/
>
> **摘要:** Aligning text-to-image (T2I) diffusion models with preference optimization is valuable for human-annotated datasets, but the heavy cost of manual data collection limits scalability. Using reward models offers an alternative, however, current preference optimization methods fall short in exploiting the rich information, as they only consider pairwise preference distribution. Furthermore, they lack generalization to multi-preference scenarios and struggle to handle inconsistencies between rewards. To address this, we present Calibrated Preference Optimization (CaPO), a novel method to align T2I diffusion models by incorporating the general preference from multiple reward models without human annotated data. The core of our approach involves a reward calibration method to approximate the general preference by computing the expected win-rate against the samples generated by the pretrained models. Additionally, we propose a frontier-based pair selection method that effectively manages the multi-preference distribution by selecting pairs from Pareto frontiers. Finally, we use regression loss to fine-tune diffusion models to match the difference between calibrated rewards of a selected pair. Experimental results show that CaPO consistently outperforms prior methods, such as Direct Preference Optimization (DPO), in both single and multi-reward settings validated by evaluation on T2I benchmarks, including GenEval and T2I-Compbench.
>
---
#### [replaced 047] Degradation-Aware All-in-One Image Restoration via Latent Prior Encoding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.17792v2](http://arxiv.org/pdf/2509.17792v2)**

> **作者:** S M A Sharif; Abdur Rehman; Fayaz Ali Dharejo; Radu Timofte; Rizwan Ali Naqvi
>
> **摘要:** Real-world images often suffer from spatially diverse degradations such as haze, rain, snow, and low-light, significantly impacting visual quality and downstream vision tasks. Existing all-in-one restoration (AIR) approaches either depend on external text prompts or embed hand-crafted architectural priors (e.g., frequency heuristics); both impose discrete, brittle assumptions that weaken generalization to unseen or mixed degradations. To address this limitation, we propose to reframe AIR as learned latent prior inference, where degradation-aware representations are automatically inferred from the input without explicit task cues. Based on latent priors, we formulate AIR as a structured reasoning paradigm: (1) which features to route (adaptive feature selection), (2) where to restore (spatial localization), and (3) what to restore (degradation semantics). We design a lightweight decoding module that efficiently leverages these latent encoded cues for spatially-adaptive restoration. Extensive experiments across six common degradation tasks, five compound settings, and previously unseen degradations demonstrate that our method outperforms state-of-the-art (SOTA) approaches, achieving an average PSNR improvement of 1.68 dB while being three times more efficient.
>
---
#### [replaced 048] Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03318v2](http://arxiv.org/pdf/2505.03318v2)**

> **作者:** Yibin Wang; Zhimin Li; Yuhang Zang; Chunyu Wang; Qinglin Lu; Cheng Jin; Jiaqi Wang
>
> **备注:** [Accepted by NeurIPS2025] Project Page: https://codegoat24.github.io/UnifiedReward/think
>
> **摘要:** Recent advances in multimodal Reward Models (RMs) have shown significant promise in delivering reward signals to align vision models with human preferences. However, current RMs are generally restricted to providing direct responses or engaging in shallow reasoning processes with limited depth, often leading to inaccurate reward signals. We posit that incorporating explicit long chains of thought (CoT) into the reward reasoning process can significantly strengthen their reliability and robustness. Furthermore, we believe that once RMs internalize CoT reasoning, their direct response accuracy can also be improved through implicit reasoning capabilities. To this end, this paper proposes UnifiedReward-Think, the first unified multimodal CoT-based reward model, capable of multi-dimensional, step-by-step long-chain reasoning for both visual understanding and generation reward tasks. Specifically, we adopt an exploration-driven reinforcement fine-tuning approach to elicit and incentivize the model's latent complex reasoning ability: (1) We first use a small amount of image generation preference data to distill the reasoning process of GPT-4o, which is then used for the model's cold start to learn the format and structure of CoT reasoning. (2) Subsequently, by leveraging the model's prior knowledge and generalization capabilities, we prepare large-scale unified multimodal preference data to elicit the model's reasoning process across various vision tasks. During this phase, correct reasoning outputs are retained for rejection sampling to refine the model (3) while incorrect predicted samples are finally used for Group Relative Policy Optimization (GRPO) based reinforcement fine-tuning, enabling the model to explore diverse reasoning paths and optimize for correct and robust solutions. Extensive experiments across various vision reward tasks demonstrate the superiority of our model.
>
---
#### [replaced 049] ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18668v4](http://arxiv.org/pdf/2505.18668v4)**

> **作者:** Zhen Li; Duan Li; Yukai Guo; Xinyuan Guo; Bowen Li; Lanxi Xiao; Shenyu Qiao; Jiashu Chen; Zijian Wu; Hui Zhang; Xinhuan Shu; Shixia Liu
>
> **备注:** 58 pages
>
> **摘要:** Infographic charts are a powerful medium for communicating abstract data by combining visual elements (e.g., charts, images) with textual information. However, their visual and structural richness poses challenges for large vision-language models (LVLMs), which are typically trained on plain charts. To bridge this gap, we introduce ChartGalaxy, a million-scale dataset designed to advance the understanding and generation of infographic charts. The dataset is constructed through an inductive process that identifies 75 chart types, 440 chart variations, and 68 layout templates from real infographic charts and uses them to create synthetic ones programmatically. We showcase the utility of this dataset through: 1) improving infographic chart understanding via fine-tuning, 2) benchmarking code generation for infographic charts, and 3) enabling example-based infographic chart generation. By capturing the visual and structural complexity of real design, ChartGalaxy provides a useful resource for enhancing multimodal reasoning and generation in LVLMs.
>
---
#### [replaced 050] Neptune-X: Active X-to-Maritime Generation for Universal Maritime Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20745v2](http://arxiv.org/pdf/2509.20745v2)**

> **作者:** Yu Guo; Shengfeng He; Yuxu Lu; Haonan An; Yihang Tao; Huilin Zhu; Jingxian Liu; Yuguang Fang
>
> **摘要:** Maritime object detection is essential for navigation safety, surveillance, and autonomous operations, yet constrained by two key challenges: the scarcity of annotated maritime data and poor generalization across various maritime attributes (e.g., object category, viewpoint, location, and imaging environment). To address these challenges, we propose Neptune-X, a data-centric generative-selection framework that enhances training effectiveness by leveraging synthetic data generation with task-aware sample selection. From the generation perspective, we develop X-to-Maritime, a multi-modality-conditioned generative model that synthesizes diverse and realistic maritime scenes. A key component is the Bidirectional Object-Water Attention module, which captures boundary interactions between objects and their aquatic surroundings to improve visual fidelity. To further improve downstream tasking performance, we propose Attribute-correlated Active Sampling, which dynamically selects synthetic samples based on their task relevance. To support robust benchmarking, we construct the Maritime Generation Dataset, the first dataset tailored for generative maritime learning, encompassing a wide range of semantic conditions. Extensive experiments demonstrate that our approach sets a new benchmark in maritime scene synthesis, significantly improving detection accuracy, particularly in challenging and previously underrepresented settings. The code is available at https://github.com/gy65896/Neptune-X.
>
---
#### [replaced 051] SPATIALGEN: Layout-guided 3D Indoor Scene Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.14981v3](http://arxiv.org/pdf/2509.14981v3)**

> **作者:** Chuan Fang; Heng Li; Yixun Liang; Jia Zheng; Yongsen Mao; Yuan Liu; Rui Tang; Zihan Zhou; Ping Tan
>
> **备注:** 3D scene generation; diffusion model; Scene reconstruction and understanding
>
> **摘要:** Creating high-fidelity 3D models of indoor environments is essential for applications in design, virtual reality, and robotics. However, manual 3D modeling remains time-consuming and labor-intensive. While recent advances in generative AI have enabled automated scene synthesis, existing methods often face challenges in balancing visual quality, diversity, semantic consistency, and user control. A major bottleneck is the lack of a large-scale, high-quality dataset tailored to this task. To address this gap, we introduce a comprehensive synthetic dataset, featuring 12,328 structured annotated scenes with 57,440 rooms, and 4.7M photorealistic 2D renderings. Leveraging this dataset, we present SpatialGen, a novel multi-view multi-modal diffusion model that generates realistic and semantically consistent 3D indoor scenes. Given a 3D layout and a reference image (derived from a text prompt), our model synthesizes appearance (color image), geometry (scene coordinate map), and semantic (semantic segmentation map) from arbitrary viewpoints, while preserving spatial consistency across modalities. SpatialGen consistently generates superior results to previous methods in our experiments. We are open-sourcing our data and models to empower the community and advance the field of indoor scene understanding and generation.
>
---
#### [replaced 052] PDV: Prompt Directional Vectors for Zero-shot Composed Image Retrieval
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07215v3](http://arxiv.org/pdf/2502.07215v3)**

> **作者:** Osman Tursun; Sinan Kalkan; Simon Denman; Clinton Fookes
>
> **摘要:** Zero-shot Composed Image Retrieval (ZS-CIR) enables image search using a reference image and a text prompt without requiring specialized text-image composition networks trained on large-scale paired data. However, current ZS-CIR approaches suffer from three critical limitations in their reliance on composed text embeddings: static query embedding representations, insufficient utilization of image embeddings, and suboptimal performance when fusing text and image embeddings. To address these challenges, we introduce the \textbf{Prompt Directional Vector (PDV)}, a simple yet effective training-free enhancement that captures semantic modifications induced by user prompts. PDV enables three key improvements: (1) Dynamic composed text embeddings where prompt adjustments are controllable via a scaling factor, (2) composed image embeddings through semantic transfer from text prompts to image features, and (3) weighted fusion of composed text and image embeddings that enhances retrieval by balancing visual and semantic similarity. Our approach serves as a plug-and-play enhancement for existing ZS-CIR methods with minimal computational overhead. Extensive experiments across multiple benchmarks demonstrate that PDV consistently improves retrieval performance when integrated with state-of-the-art ZS-CIR approaches, particularly for methods that generate accurate compositional embeddings. The code will be released upon publication.
>
---
#### [replaced 053] Geometry aware inference of steady state PDEs using Equivariant Neural Fields representations
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.18591v2](http://arxiv.org/pdf/2504.18591v2)**

> **作者:** Giovanni Catalani; Michael Bauerheim; Frédéric Tost; Xavier Bertrand; Joseph Morlier
>
> **备注:** NeurIPS 2025 AI for Science workshop
>
> **摘要:** Advances in neural operators have introduced discretization invariant surrogate models for PDEs on general geometries, yet many approaches struggle to encode local geometric structure and variable domains efficiently. We introduce enf2enf, a neural field approach for predicting steady-state PDEs with geometric variability. Our method encodes geometries into latent features anchored at specific spatial locations, preserving locality throughout the network. These local representations are combined with global parameters and decoded to continuous physical fields, enabling effective modeling of complex shape variations. Experiments on aerodynamic and structural benchmarks demonstrate competitive or superior performance compared to graph-based, neural operator, and recent neural field methods, with real-time inference and efficient scaling to high-resolution meshes.
>
---
#### [replaced 054] Semantic Consistent Language Gaussian Splatting for Point-Level Open-vocabulary Querying
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21767v2](http://arxiv.org/pdf/2503.21767v2)**

> **作者:** Hairong Yin; Huangying Zhan; Yi Xu; Raymond A. Yeh
>
> **摘要:** Open-vocabulary 3D scene understanding is crucial for robotics applications, such as natural language-driven manipulation, human-robot interaction, and autonomous navigation. Existing methods for querying 3D Gaussian Splatting often struggle with inconsistent 2D mask supervision and lack a robust 3D point-level retrieval mechanism. In this work, (i) we present a novel point-level querying framework that performs tracking on segmentation masks to establish a semantically consistent ground-truth for distilling the language Gaussians; (ii) we introduce a GT-anchored querying approach that first retrieves the distilled ground-truth and subsequently uses the ground-truth to query the individual Gaussians. Extensive experiments on three benchmark datasets demonstrate that the proposed method outperforms state-of-the-art performance. Our method achieves an mIoU improvement of +4.14, +20.42, and +1.7 on the LERF, 3D-OVS, and Replica datasets. These results validate our framework as a promising step toward open-vocabulary understanding in real-world robotic systems.
>
---
#### [replaced 055] Pose Prior Learner: Unsupervised Categorical Prior Learning for Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.03858v3](http://arxiv.org/pdf/2410.03858v3)**

> **作者:** Ziyu Wang; Shuangpeng Han; Mengmi Zhang
>
> **摘要:** A prior represents a set of beliefs or assumptions about a system, aiding inference and decision-making. In this paper, we introduce the challenge of unsupervised categorical prior learning in pose estimation, where AI models learn a general pose prior for an object category from images in a self-supervised manner. Although priors are effective in estimating pose, acquiring them can be difficult. We propose a novel method, named Pose Prior Learner (PPL), to learn a general pose prior for any object category. PPL uses a hierarchical memory to store compositional parts of prototypical poses, from which we distill a general pose prior. This prior improves pose estimation accuracy through template transformation and image reconstruction. PPL learns meaningful pose priors without any additional human annotations or interventions, outperforming competitive baselines on both human and animal pose estimation datasets. Notably, our experimental results reveal the effectiveness of PPL using learned prototypical poses for pose estimation on occluded images. Through iterative inference, PPL leverages the pose prior to refine estimated poses, regressing them to any prototypical poses stored in memory. Our code, model, and data will be publicly available.
>
---
#### [replaced 056] Learning Personalized Driving Styles via Reinforcement Learning from Human Feedback
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10434v2](http://arxiv.org/pdf/2503.10434v2)**

> **作者:** Derun Li; Changye Li; Yue Wang; Jianwei Ren; Xin Wen; Pengxiang Li; Leimeng Xu; Kun Zhan; Peng Jia; Xianpeng Lang; Ningyi Xu; Hang Zhao
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Generating human-like and adaptive trajectories is essential for autonomous driving in dynamic environments. While generative models have shown promise in synthesizing feasible trajectories, they often fail to capture the nuanced variability of personalized driving styles due to dataset biases and distributional shifts. To address this, we introduce TrajHF, a human feedback-driven finetuning framework for generative trajectory models, designed to align motion planning with diverse driving styles. TrajHF incorporates multi-conditional denoiser and reinforcement learning with human feedback to refine multi-modal trajectory generation beyond conventional imitation learning. This enables better alignment with human driving preferences while maintaining safety and feasibility constraints. TrajHF achieves performance comparable to the state-of-the-art on NavSim benchmark. TrajHF sets a new paradigm for personalized and adaptable trajectory generation in autonomous driving.
>
---
#### [replaced 057] Frequency-Domain Refinement with Multiscale Diffusion for Super Resolution
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2405.10014v2](http://arxiv.org/pdf/2405.10014v2)**

> **作者:** Xingjian Wang; Li Chai; Jiming Chen
>
> **摘要:** The performance of single image super-resolution depends heavily on how to generate and complement high-frequency details to low-resolution images. Recently, diffusion-based DDPM models exhibit great potential in generating high-quality details for super-resolution tasks. They tend to directly predict high-frequency information of wide bandwidth by solely utilizing the high-resolution ground truth as the target for all sampling timesteps. However, as a result, they encounter hallucination problem that they generate mismatching artifacts. To tackle this problem and achieve higher-quality super-resolution, we propose a novel Frequency Domain-guided multiscale Diffusion model (FDDiff), which decomposes the high-frequency information complementing process into finer-grained steps. In particular, a wavelet packet-based frequency degradation pyramid is developed to provide multiscale intermediate targets with increasing bandwidth. Based on these targets, FDDiff guides reverse diffusion process to progressively complement missing high-frequency details over timesteps. Moreover, a multiscale frequency refinement network is designed to predict the required high-frequency components at multiple scales within one unified network. Comprehensive evaluations on popular benchmarks are conducted, and demonstrate that FDDiff outperforms prior generative methods with higher-fidelity super-resolution results.
>
---
#### [replaced 058] Content-Aware Mamba for Learned Image Compression
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.02192v4](http://arxiv.org/pdf/2508.02192v4)**

> **作者:** Yunuo Chen; Zezheng Lyu; Bing He; Hongwei Hu; Qi Wang; Yuan Tian; Li Song; Wenjun Zhang; Guo Lu
>
> **摘要:** Recent learned image compression (LIC) leverages Mamba-style state-space models (SSMs) for global receptive fields with linear complexity. However, the standard Mamba adopts content-agnostic, predefined raster (or multi-directional) scans under strict causality. This rigidity hinders its ability to effectively eliminate redundancy between tokens that are content-correlated but spatially distant. We introduce Content-Aware Mamba (CAM), an SSM that dynamically adapts its processing to the image content. Specifically, CAM overcomes prior limitations with two novel mechanisms. First, it replaces the rigid scan with a content-adaptive token permutation strategy to prioritize interactions between content-similar tokens regardless of their location. Second, it overcomes the sequential dependency by injecting sample-specific global priors into the state-space model, which effectively mitigates the strict causality without multi-directional scans. These innovations enable CAM to better capture global redundancy while preserving computational efficiency. Our Content-Aware Mamba-based LIC model (CMIC) achieves state-of-the-art rate-distortion performance, surpassing VTM-21.0 by 15.91%, 21.34%, and 17.58% in BD-rate on the Kodak, Tecnick, and CLIC datasets, respectively. Code and checkpoints will be released later.
>
---
#### [replaced 059] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.13674v4](http://arxiv.org/pdf/2410.13674v4)**

> **作者:** Yijun Liang; Shweta Bhardwaj; Tianyi Zhou
>
> **备注:** Accepted in ICCV2025. 22 pages, including references and appendix. Code is available at http://github.com/tianyi-lab/DisCL
>
> **摘要:** Low-quality or scarce data has posed significant challenges for training deep neural networks in practice. While classical data augmentation cannot contribute very different new data, diffusion models opens up a new door to build self-evolving AI by generating high-quality and diverse synthetic data through text-guided prompts. However, text-only guidance cannot control synthetic images' proximity to the original images, resulting in out-of-distribution data detrimental to the model performance. To overcome the limitation, we study image guidance to achieve a spectrum of interpolations between synthetic and real images. With stronger image guidance, the generated images are similar to the training data but hard to learn. While with weaker image guidance, the synthetic images will be easier for model but contribute to a larger distribution gap with the original data. The generated full spectrum of data enables us to build a novel "Diffusion Curriculum (DisCL)". DisCL adjusts the image guidance level of image synthesis for each training stage: It identifies and focuses on hard samples for the model and assesses the most effective guidance level of synthetic images to improve hard data learning. We apply DisCL to two challenging tasks: long-tail (LT) classification and learning from low-quality data. It focuses on lower-guidance images of high-quality to learn prototypical features as a warm-up of learning higher-guidance images that might be weak on diversity or quality. Extensive experiments showcase a gain of 2.7% and 2.1% in OOD and ID macro-accuracy when applying DisCL to iWildCam dataset. On ImageNet-LT, DisCL improves the base model's tail-class accuracy from 4.4% to 23.64% and leads to a 4.02% improvement in all-class accuracy.
>
---
#### [replaced 060] RAAG: Ratio Aware Adaptive Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03442v2](http://arxiv.org/pdf/2508.03442v2)**

> **作者:** Shangwen Zhu; Qianyu Peng; Yuting Hu; Zhantao Yang; Han Zhang; Zhao Pu; Andy Zheng; Zhilei Shu; Ruili Feng; Fan Cheng
>
> **摘要:** Flow-based generative models have achieved remarkable progress, with classifier-free guidance (CFG) becoming the standard for high-fidelity generation. However, the conventional practice of applying a strong, fixed guidance scale throughout inference is poorly suited for the rapid, few-step sampling required by modern applications. In this work, we uncover the root cause of this conflict: a fundamental sampling instability where the earliest steps are acutely sensitive to guidance. We trace this to a significant spike in the ratio of conditional to unconditional predictions--a spike that we prove to be an inherent property of the training data distribution itself, making it a almost inevitable challenge. Applying a high, static guidance value during this volatile initial phase leads to an exponential amplification of error, degrading image quality. To resolve this, we propose a simple, theoretically grounded, adaptive guidance schedule that automatically dampens the guidance scale at early steps based on the evolving ratio. Our method is lightweight, incurs no inference overhead, and is compatible with standard frameworks. Experiments across state-of-the-art image (SD3.5, Qwen-Image) and video (WAN2.1) models show our approach enables up to 3x faster sampling while maintaining or improving quality, robustness, and semantic alignment. Our findings highlight that adapting guidance to the sampling process, rather than fixing it, is critical for unlocking the full potential of fast, flow-based models.
>
---
#### [replaced 061] Real-Time Object Detection Meets DINOv3
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20787v2](http://arxiv.org/pdf/2509.20787v2)**

> **作者:** Shihua Huang; Yongjie Hou; Longfei Liu; Xuanlong Yu; Xi Shen
>
> **备注:** Source code available at https://github.com/Intellindust-AI-Lab/DEIMv2
>
> **摘要:** Benefiting from the simplicity and effectiveness of Dense O2O and MAL, DEIM has become the mainstream training framework for real-time DETRs, significantly outperforming the YOLO series. In this work, we extend it with DINOv3 features, resulting in DEIMv2. DEIMv2 spans eight model sizes from X to Atto, covering GPU, edge, and mobile deployment. For the X, L, M, and S variants, we adopt DINOv3-pretrained or distilled backbones and introduce a Spatial Tuning Adapter (STA), which efficiently converts DINOv3's single-scale output into multi-scale features and complements strong semantics with fine-grained details to enhance detection. For ultra-lightweight models (Nano, Pico, Femto, and Atto), we employ HGNetv2 with depth and width pruning to meet strict resource budgets. Together with a simplified decoder and an upgraded Dense O2O, this unified design enables DEIMv2 to achieve a superior performance-cost trade-off across diverse scenarios, establishing new state-of-the-art results. Notably, our largest model, DEIMv2-X, achieves 57.8 AP with only 50.3 million parameters, surpassing prior X-scale models that require over 60 million parameters for just 56.5 AP. On the compact side, DEIMv2-S is the first sub-10 million model (9.71 million) to exceed the 50 AP milestone on COCO, reaching 50.9 AP. Even the ultra-lightweight DEIMv2-Pico, with just 1.5 million parameters, delivers 38.5 AP, matching YOLOv10-Nano (2.3 million) with around 50 percent fewer parameters. Our code and pre-trained models are available at https://github.com/Intellindust-AI-Lab/DEIMv2
>
---
#### [replaced 062] FERD: Fairness-Enhanced Data-Free Robustness Distillation
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20793v2](http://arxiv.org/pdf/2509.20793v2)**

> **作者:** Zhengxiao Li; Liming Lu; Xu Zheng; Siyuan Liang; Zhenghan Chen; Yongbin Zhou; Shuchao Pang
>
> **摘要:** Data-Free Robustness Distillation (DFRD) aims to transfer the robustness from the teacher to the student without accessing the training data. While existing methods focus on overall robustness, they overlook the robust fairness issues, leading to severe disparity of robustness across different categories. In this paper, we find two key problems: (1) student model distilled with equal class proportion data behaves significantly different across distinct categories; and (2) the robustness of student model is not stable across different attacks target. To bridge these gaps, we present the first Fairness-Enhanced data-free Robustness Distillation (FERD) framework to adjust the proportion and distribution of adversarial examples. For the proportion, FERD adopts a robustness-guided class reweighting strategy to synthesize more samples for the less robust categories, thereby improving robustness of them. For the distribution, FERD generates complementary data samples for advanced robustness distillation. It generates Fairness-Aware Examples (FAEs) by enforcing a uniformity constraint on feature-level predictions, which suppress the dominance of class-specific non-robust features, providing a more balanced representation across all categories. Then, FERD constructs Uniform-Target Adversarial Examples (UTAEs) from FAEs by applying a uniform target class constraint to avoid biased attack directions, which distribute the attack targets across all categories and prevents overfitting to specific vulnerable categories. Extensive experiments on three public datasets show that FERD achieves state-of-the-art worst-class robustness under all adversarial attack (e.g., the worst-class robustness under FGSM and AutoAttack are improved by 15.1\% and 6.4\% using MobileNet-V2 on CIFAR-10), demonstrating superior performance in both robustness and fairness aspects.
>
---
#### [replaced 063] NeuVAS: Neural Implicit Surfaces for Variational Shape Modeling
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13050v2](http://arxiv.org/pdf/2506.13050v2)**

> **作者:** Pengfei Wang; Qiujie Dong; Fangtian Liang; Hao Pan; Lei Yang; Congyi Zhang; Guying Lin; Caiming Zhang; Yuanfeng Zhou; Changhe Tu; Shiqing Xin; Alla Sheffer; Xin Li; Wenping Wang
>
> **摘要:** Neural implicit shape representation has drawn significant attention in recent years due to its smoothness, differentiability, and topological flexibility. However, directly modeling the shape of a neural implicit surface, especially as the zero-level set of a neural signed distance function (SDF), with sparse geometric control is still a challenging task. Sparse input shape control typically includes 3D curve networks or, more generally, 3D curve sketches, which are unstructured and cannot be connected to form a curve network, and therefore more difficult to deal with. While 3D curve networks or curve sketches provide intuitive shape control, their sparsity and varied topology pose challenges in generating high-quality surfaces to meet such curve constraints. In this paper, we propose NeuVAS, a variational approach to shape modeling using neural implicit surfaces constrained under sparse input shape control, including unstructured 3D curve sketches as well as connected 3D curve networks. Specifically, we introduce a smoothness term based on a functional of surface curvatures to minimize shape variation of the zero-level set surface of a neural SDF. We also develop a new technique to faithfully model G0 sharp feature curves as specified in the input curve sketches. Comprehensive comparisons with the state-of-the-art methods demonstrate the significant advantages of our method.
>
---
#### [replaced 064] EditVerse: Unifying Image and Video Editing and Generation with In-Context Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20360v2](http://arxiv.org/pdf/2509.20360v2)**

> **作者:** Xuan Ju; Tianyu Wang; Yuqian Zhou; He Zhang; Qing Liu; Nanxuan Zhao; Zhifei Zhang; Yijun Li; Yuanhao Cai; Shaoteng Liu; Daniil Pakhomov; Zhe Lin; Soo Ye Kim; Qiang Xu
>
> **摘要:** Recent advances in foundation models highlight a clear trend toward unification and scaling, showing emergent capabilities across diverse domains. While image generation and editing have rapidly transitioned from task-specific to unified frameworks, video generation and editing remain fragmented due to architectural limitations and data scarcity. In this work, we introduce EditVerse, a unified framework for image and video generation and editing within a single model. By representing all modalities, i.e., text, image, and video, as a unified token sequence, EditVerse leverages self-attention to achieve robust in-context learning, natural cross-modal knowledge transfer, and flexible handling of inputs and outputs with arbitrary resolutions and durations. To address the lack of video editing training data, we design a scalable data pipeline that curates 232K video editing samples and combines them with large-scale image and video datasets for joint training. Furthermore, we present EditVerseBench, the first benchmark for instruction-based video editing covering diverse tasks and resolutions. Extensive experiments and user studies demonstrate that EditVerse achieves state-of-the-art performance, surpassing existing open-source and commercial models, while exhibiting emergent editing and generation abilities across modalities.
>
---
#### [replaced 065] HiSin: A Sinogram-Aware Framework for Efficient High-Resolution Inpainting
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.08809v2](http://arxiv.org/pdf/2506.08809v2)**

> **作者:** Jiaze E; Srutarshi Banerjee; Tekin Bicer; Guannan Wang; Yanfu Zhang; Bin Ren
>
> **摘要:** High-resolution sinogram inpainting is essential for computed tomography reconstruction, as missing high-frequency projections can lead to visible artifacts and diagnostic errors. Diffusion models are well-suited for this task due to their robustness and detail-preserving capabilities, but their application to high-resolution inputs is limited by excessive memory and computational demands. To address this limitation, we propose HiSin, a novel diffusion-based framework for efficient sinogram inpainting that exploits spectral sparsity and structural heterogeneity of projection data. It progressively extracts global structure at low resolution and defers high-resolution inference to small patches, enabling memory-efficient inpainting. Considering the structural features of sinograms, we incorporate frequency-aware patch skipping and structure-adaptive step allocation to reduce redundant computation. Experimental results show that HiSin reduces peak memory usage by up to 30.81% and inference time by up to 17.58% than the state-of-the-art framework, and maintains inpainting accuracy across.
>
---
#### [replaced 066] Vivid-VR: Distilling Concepts from Text-to-Video Diffusion Transformer for Photorealistic Video Restoration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.14483v3](http://arxiv.org/pdf/2508.14483v3)**

> **作者:** Haoran Bai; Xiaoxu Chen; Canqian Yang; Zongyao He; Sibin Deng; Ying Chen
>
> **摘要:** We present Vivid-VR, a DiT-based generative video restoration method built upon an advanced T2V foundation model, where ControlNet is leveraged to control the generation process, ensuring content consistency. However, conventional fine-tuning of such controllable pipelines frequently suffers from distribution drift due to limitations in imperfect multimodal alignment, resulting in compromised texture realism and temporal coherence. To tackle this challenge, we propose a concept distillation training strategy that utilizes the pretrained T2V model to synthesize training samples with embedded textual concepts, thereby distilling its conceptual understanding to preserve texture and temporal quality. To enhance generation controllability, we redesign the control architecture with two key components: 1) a control feature projector that filters degradation artifacts from input video latents to minimize their propagation through the generation pipeline, and 2) a new ControlNet connector employing a dual-branch design. This connector synergistically combines MLP-based feature mapping with cross-attention mechanism for dynamic control feature retrieval, enabling both content preservation and adaptive control signal modulation. Extensive experiments show that Vivid-VR performs favorably against existing approaches on both synthetic and real-world benchmarks, as well as AIGC videos, achieving impressive texture realism, visual vividness, and temporal consistency. The codes and checkpoints are publicly available at https://github.com/csbhr/Vivid-VR.
>
---
#### [replaced 067] Shape-for-Motion: Precise and Consistent Video Editing with 3D Proxy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22432v2](http://arxiv.org/pdf/2506.22432v2)**

> **作者:** Yuhao Liu; Tengfei Wang; Fang Liu; Zhenwei Wang; Rynson W. H. Lau
>
> **备注:** Accepted by Siggraph Asia 2025
>
> **摘要:** Recent advances in deep generative modeling have unlocked unprecedented opportunities for video synthesis. In real-world applications, however, users often seek tools to faithfully realize their creative editing intentions with precise and consistent control. Despite the progress achieved by existing methods, ensuring fine-grained alignment with user intentions remains an open and challenging problem. In this work, we present Shape-for-Motion, a novel framework that incorporates a 3D proxy for precise and consistent video editing. Shape-for-Motion achieves this by converting the target object in the input video to a time-consistent mesh, i.e., a 3D proxy, allowing edits to be performed directly on the proxy and then inferred back to the video frames. To simplify the editing process, we design a novel Dual-Propagation Strategy that allows users to perform edits on the 3D mesh of a single frame, and the edits are then automatically propagated to the 3D meshes of the other frames. The 3D meshes for different frames are further projected onto the 2D space to produce the edited geometry and texture renderings, which serve as inputs to a decoupled video diffusion model for generating edited results. Our framework supports various precise and physically-consistent manipulations across the video frames, including pose editing, rotation, scaling, translation, texture modification, and object composition. Our approach marks a key step toward high-quality, controllable video editing workflows. Extensive experiments demonstrate the superiority and effectiveness of our approach. Project page: https://shapeformotion.github.io/
>
---
#### [replaced 068] LOGen: Toward Lidar Object Generation by Point Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.07385v3](http://arxiv.org/pdf/2412.07385v3)**

> **作者:** Ellington Kirby; Mickael Chen; Renaud Marlet; Nermin Samet
>
> **备注:** BMVC 2025
>
> **摘要:** The generation of LiDAR scans is a growing topic with diverse applications to autonomous driving. However, scan generation remains challenging, especially when compared to the rapid advancement of image and 3D object generation. We consider the task of LiDAR object generation, requiring models to produce 3D objects as viewed by a LiDAR scan. It focuses LiDAR scan generation on a key aspect of scenes, the objects, while also benefiting from advancements in 3D object generative methods. We introduce a novel diffusion-based model to produce LiDAR point clouds of dataset objects, including intensity, and with an extensive control of the generation via conditioning information. Our experiments on nuScenes and KITTI-360 show the quality of our generations measured with new 3D metrics developed to suit LiDAR objects. The code is available at https://github.com/valeoai/LOGen.
>
---
#### [replaced 069] Event2Vec: Processing Neuromorphic Events directly by Representations in Vector Space
- **分类: cs.CV; cs.NE**

- **链接: [http://arxiv.org/pdf/2504.15371v3](http://arxiv.org/pdf/2504.15371v3)**

> **作者:** Wei Fang; Priyadarshini Panda
>
> **摘要:** Neuromorphic event cameras possess superior temporal resolution, power efficiency, and dynamic range compared to traditional cameras. However, their asynchronous and sparse data format poses a significant challenge for conventional deep learning methods. Existing solutions to this incompatibility often sacrifice temporal resolution, require extensive pre-processing, and do not fully leverage GPU acceleration. Inspired by word-to-vector models, we draw an analogy between words and events to introduce event2vec, a novel representation that allows neural networks to process events directly. This approach is fully compatible with the parallel processing and self-supervised learning capabilities of Transformer architectures. We demonstrate the effectiveness of event2vec on the DVS Gesture, ASL-DVS, and DVS-Lip benchmarks. A comprehensive ablation study further analyzes our method's features and contrasts them with existing representations. The experimental results show that event2vec is remarkably parameter-efficient, has high throughput, and can achieve high accuracy even with an extremely low number of events. Beyond its performance, the most significant contribution of event2vec is a new paradigm that enables neural networks to process event streams as if they were natural language. This paradigm shift paves the way for the native integration of event cameras with large language models and multimodal models. Code, model, and training logs are provided in https://github.com/Intelligent-Computing-Lab-Panda/event2vec.
>
---
#### [replaced 070] APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation
- **分类: cs.NE; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.14270v4](http://arxiv.org/pdf/2507.14270v4)**

> **作者:** Ravin Kumar
>
> **备注:** 11 pages, 2 figures, 1 table. Includes a GitHub repository for MNIST experiments and a PyPI package for APTx Neuron implementation
>
> **摘要:** We propose the APTx Neuron, a novel, unified neural computation unit that integrates non-linear activation and linear transformation into a single trainable expression. The APTx Neuron is derived from the APTx activation function, thereby eliminating the need for separate activation layers and making the architecture both computationally efficient and elegant. The proposed neuron follows the functional form $y = \sum_{i=1}^{n} ((\alpha_i + \tanh(\beta_i x_i)) \cdot \gamma_i x_i) + \delta$, where all parameters $\alpha_i$, $\beta_i$, $\gamma_i$, and $\delta$ are trainable. We validate our APTx Neuron-based architecture on the MNIST dataset, achieving up to $96.69\%$ test accuracy within 11 epochs using approximately 332K trainable parameters. The results highlight the superior expressiveness and computational efficiency of the APTx Neuron compared to traditional neurons, pointing toward a new paradigm in unified neuron design and the architectures built upon it. Source code is available at https://github.com/mr-ravin/aptx_neuron.
>
---
#### [replaced 071] TAMMs: Temporal-Aware Multimodal Model for Satellite Image Change Understanding and Forecasting
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18862v2](http://arxiv.org/pdf/2506.18862v2)**

> **作者:** Zhongbin Guo; Yuhao Wang; Ping Jian; Chengzhi Li; Xinyue Chen; Zhen Yang; Ertai E
>
> **备注:** Submitted to The Fourteenth International Conference on Learning Representations (ICLR 2026). Our dataset can be found at https://huggingface.co/datasets/IceInPot/TAMMs
>
> **摘要:** Temporal Change Description (TCD) and Future Satellite Image Forecasting (FSIF) are critical, yet historically disjointed tasks in Satellite Image Time Series (SITS) analysis. Both are fundamentally limited by the common challenge of modeling long-range temporal dynamics. To explore how to improve the performance of methods on both tasks simultaneously by enhancing long-range temporal understanding capabilities, we introduce TAMMs, the first unified framework designed to jointly perform TCD and FSIF within a single MLLM-diffusion architecture. TAMMs introduces two key innovations: Temporal Adaptation Modules (TAM) enhance frozen MLLM's ability to comprehend long-range dynamics, and Semantic-Fused Control Injection (SFCI) mechanism translates this change understanding into fine-grained generative control. This synergistic design makes the understanding from the TCD task to directly inform and improve the consistency of the FSIF task. Extensive experiments demonstrate TAMMs significantly outperforms state-of-the-art specialist baselines on both tasks.
>
---
#### [replaced 072] Ctrl-Z Sampling: Diffusion Sampling with Controlled Random Zigzag Explorations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.20294v3](http://arxiv.org/pdf/2506.20294v3)**

> **作者:** Shunqi Mao; Wei Guo; Chaoyi Zhang; Jieting Long; Ke Xie; Weidong Cai
>
> **备注:** 32 pages, 11 figures, 10 tables
>
> **摘要:** Diffusion models have shown strong performance in conditional generation by progressively denoising Gaussian samples toward a target data distribution. This denoising process can be interpreted as a form of hill climbing in a learned representation space, where the model iteratively refines a sample toward regions of higher probability. However, this learned climbing often converges to local optima with plausible but suboptimal generations due to latent space complexity and suboptimal initialization. While prior efforts often strengthen guidance signals or introduce fixed exploration strategies to address this, they exhibit limited capacity to escape steep local maxima. In contrast, we propose Controlled Random Zigzag Sampling (Ctrl-Z Sampling), a novel sampling strategy that adaptively detects and escapes such traps through controlled exploration. In each diffusion step, we first identify potential local maxima using a reward model. Upon such detection, we inject noise and revert to a previous, noisier state to escape the current plateau. The reward model then evaluates candidate trajectories, accepting only those that offer improvement, otherwise scheming progressively deeper explorations when nearby alternatives fail. This controlled zigzag process allows dynamic alternation between forward refinement and backward exploration, enhancing both alignment and visual quality in the generated outputs. The proposed method is model-agnostic and also compatible with existing diffusion frameworks. Experimental results show that Ctrl-Z Sampling substantially improves generation quality while requiring only about 7.72 times the NFEs of the original.
>
---
#### [replaced 073] Texture or Semantics? Vision-Language Models Get Lost in Font Recognition
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23768v4](http://arxiv.org/pdf/2503.23768v4)**

> **作者:** Zhecheng Li; Guoxian Song; Yujun Cai; Zhen Xiong; Junsong Yuan; Yiwei Wang
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Modern Vision-Language Models (VLMs) exhibit remarkable visual and linguistic capabilities, achieving impressive performance in various tasks such as image recognition and object localization. However, their effectiveness in fine-grained tasks remains an open question. In everyday scenarios, individuals encountering design materials, such as magazines, typography tutorials, research papers, or branding content, may wish to identify aesthetically pleasing fonts used in the text. Given their multimodal capabilities and free accessibility, many VLMs are often considered potential tools for font recognition. This raises a fundamental question: Do VLMs truly possess the capability to recognize fonts? To investigate this, we introduce the Font Recognition Benchmark (FRB), a compact and well-structured dataset comprising 15 commonly used fonts. FRB includes two versions: (i) an easy version, where 10 sentences are rendered in different fonts, and (ii) a hard version, where each text sample consists of the names of the 15 fonts themselves, introducing a stroop effect that challenges model perception. Through extensive evaluation of various VLMs on font recognition tasks, we arrive at the following key findings: (i) Current VLMs exhibit limited font recognition capabilities, with many state-of-the-art models failing to achieve satisfactory performance and being easily affected by the stroop effect introduced by textual information. (ii) Few-shot learning and Chain-of-Thought (CoT) prompting provide minimal benefits in improving font recognition accuracy across different VLMs. (iii) Attention analysis sheds light on the inherent limitations of VLMs in capturing semantic features.
>
---
#### [replaced 074] Astraea: A Token-wise Acceleration Framework for Video Diffusion Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05096v4](http://arxiv.org/pdf/2506.05096v4)**

> **作者:** Haosong Liu; Yuge Cheng; Wenxuan Miao; Zihan Liu; Aiyue Chen; Jing Lin; Yiwu Yao; Chen Chen; Jingwen Leng; Yu Feng; Minyi Guo
>
> **摘要:** Video diffusion transformers (vDiTs) have made tremendous progress in text-to-video generation, but their high compute demands pose a major challenge for practical deployment. While studies propose acceleration methods to reduce workload at various granularities, they often rely on heuristics, limiting their applicability. We introduce Astraea, a framework that searches for near-optimal configurations for vDiT-based video generation under a performance target. At its core, Astraea proposes a lightweight token selection mechanism and a memory-efficient, GPU-friendly sparse attention strategy, enabling linear savings on execution time with minimal impact on generation quality. Meanwhile, to determine optimal token reduction for different timesteps, we further design a search framework that leverages a classic evolutionary algorithm to automatically determine the distribution of the token budget effectively. Together, Astraea achieves up to 2.4$\times$ inference speedup on a single GPU with great scalability (up to 13.2$\times$ speedup on 8 GPUs) while achieving up to over 10~dB video quality compared to the state-of-the-art methods ($<$0.5\% loss on VBench compared to baselines).
>
---
#### [replaced 075] STQE: Spatial-Temporal Attribute Quality Enhancement for G-PCC Compressed Dynamic Point Clouds
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.17522v2](http://arxiv.org/pdf/2507.17522v2)**

> **作者:** Tian Guo; Hui Yuan; Xiaolong Mao; Shiqi Jiang; Raouf Hamzaoui; Sam Kwong
>
> **摘要:** Very few studies have addressed quality enhancement for compressed dynamic point clouds. In particular, the effective exploitation of spatial-temporal correlations between point cloud frames remains largely unexplored. Addressing this gap, we propose a spatial-temporal attribute quality enhancement (STQE) network that exploits both spatial and temporal correlations to improve the visual quality of G-PCC compressed dynamic point clouds. Our contributions include a recoloring-based motion compensation module that remaps reference attribute information to the current frame geometry to achieve precise inter-frame geometric alignment, a channel-aware temporal attention module that dynamically highlights relevant regions across bidirectional reference frames, a Gaussian-guided neighborhood feature aggregation module that efficiently captures spatial dependencies between geometry and color attributes, and a joint loss function based on the Pearson correlation coefficient, designed to alleviate over-smoothing effects typical of point-wise mean squared error optimization. When applied to the latest G-PCC test model, STQE achieved improvements of 0.855 dB, 0.682 dB, and 0.828 dB in delta PSNR, with Bj{\o}ntegaard Delta rate (BD-rate) reductions of -25.2%, -31.6%, and -32.5% for the Luma, Cb, and Cr components, respectively.
>
---
#### [replaced 076] Mamba-Driven Topology Fusion for Monocular 3D Human Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20611v2](http://arxiv.org/pdf/2505.20611v2)**

> **作者:** Zenghao Zheng; Lianping Yang; Jinshan Pan; Hegui Zhu
>
> **摘要:** Transformer-based methods for 3D human pose estimation face significant computational challenges due to the quadratic growth of self-attention mechanism complexity with sequence length. Recently, the Mamba model has substantially reduced computational overhead and demonstrated outstanding performance in modeling long sequences by leveraging state space model (SSM). However, the ability of SSM to process sequential data is not suitable for 3D joint sequences with topological structures, and the causal convolution structure in Mamba also lacks insight into local joint relationships. To address these issues, we propose the Mamba-Driven Topology Fusion framework in this paper. Specifically, the proposed Bone Aware Module infers the direction and length of bone vectors in the spherical coordinate system, providing effective topological guidance for the Mamba model in processing joint sequences. Furthermore, we enhance the convolutional structure within the Mamba model by integrating forward and backward graph convolutional network, enabling it to better capture local joint dependencies. Finally, we design a Spatiotemporal Refinement Module to model both temporal and spatial relationships within the sequence. Through the incorporation of skeletal topology, our approach effectively alleviates Mamba's limitations in capturing human structural relationships. We conduct extensive experiments on the Human3.6M and MPI-INF-3DHP datasets for testing and comparison, and the results show that the proposed method greatly reduces computational cost while achieving higher accuracy. Ablation studies further demonstrate the effectiveness of each proposed module. The code and models will be released.
>
---
#### [replaced 077] Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05271v5](http://arxiv.org/pdf/2412.05271v5)**

> **作者:** Zhe Chen; Weiyun Wang; Yue Cao; Yangzhou Liu; Zhangwei Gao; Erfei Cui; Jinguo Zhu; Shenglong Ye; Hao Tian; Zhaoyang Liu; Lixin Gu; Xuehui Wang; Qingyun Li; Yiming Ren; Zixuan Chen; Jiapeng Luo; Jiahao Wang; Tan Jiang; Bo Wang; Conghui He; Botian Shi; Xingcheng Zhang; Han Lv; Yi Wang; Wenqi Shao; Pei Chu; Zhongying Tu; Tong He; Zhiyong Wu; Huipeng Deng; Jiaye Ge; Kai Chen; Kaipeng Zhang; Limin Wang; Min Dou; Lewei Lu; Xizhou Zhu; Tong Lu; Dahua Lin; Yu Qiao; Jifeng Dai; Wenhai Wang
>
> **备注:** Technical Report
>
> **摘要:** We introduce InternVL 2.5, an advanced multimodal large language model (MLLM) series that builds upon InternVL 2.0, maintaining its core model architecture while introducing significant enhancements in training and testing strategies as well as data quality. In this work, we delve into the relationship between model scaling and performance, systematically exploring the performance trends in vision encoders, language models, dataset sizes, and test-time configurations. Through extensive evaluations on a wide range of benchmarks, including multi-discipline reasoning, document understanding, multi-image / video understanding, real-world comprehension, multimodal hallucination detection, visual grounding, multilingual capabilities, and pure language processing, InternVL 2.5 exhibits competitive performance, rivaling leading commercial models such as GPT-4o and Claude-3.5-Sonnet. Notably, our model is the first open-source MLLMs to surpass 70% on the MMMU benchmark, achieving a 3.7-point improvement through Chain-of-Thought (CoT) reasoning and showcasing strong potential for test-time scaling. We hope this model contributes to the open-source community by setting new standards for developing and applying multimodal AI systems. HuggingFace demo see https://huggingface.co/spaces/OpenGVLab/InternVL
>
---
#### [replaced 078] Self-Guidance: Boosting Flow and Diffusion Generation on Their Own
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05827v5](http://arxiv.org/pdf/2412.05827v5)**

> **作者:** Tiancheng Li; Weijian Luo; Zhiyang Chen; Liyuan Ma; Guo-Jun Qi
>
> **备注:** 16 pages, 13 figures
>
> **摘要:** Proper guidance strategies are essential to achieve high-quality generation results without retraining diffusion and flow-based text-to-image models. Existing guidance either requires specific training or strong inductive biases of diffusion model networks, which potentially limits their ability and application scope. Motivated by the observation that artifact outliers can be detected by a significant decline in the density from a noisier to a cleaner noise level, we propose Self-Guidance (SG), which can significantly improve the quality of the generated image by suppressing the generation of low-quality samples. The biggest difference from existing guidance is that SG only relies on the sampling score function of the original diffusion or flow model at different noise levels, with no need for any tricky and expensive guidance-specific training. This makes SG highly flexible to be used in a plug-and-play manner by any diffusion or flow models. We also introduce an efficient variant of SG, named SG-prev, which reuses the output from the immediately previous diffusion step to avoid additional forward passes of the diffusion network.We conduct extensive experiments on text-to-image and text-to-video generation with different architectures, including UNet and transformer models. With open-sourced diffusion models such as Stable Diffusion 3.5 and FLUX, SG exceeds existing algorithms on multiple metrics, including both FID and Human Preference Score. SG-prev also achieves strong results over both the baseline and the SG, with 50 percent more efficiency. Moreover, we find that SG and SG-prev both have a surprisingly positive effect on the generation of physiologically correct human body structures such as hands, faces, and arms, showing their ability to eliminate human body artifacts with minimal efforts. We have released our code at https://github.com/maple-research-lab/Self-Guidance.
>
---
#### [replaced 079] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07392v2](http://arxiv.org/pdf/2503.07392v2)**

> **作者:** Ouxiang Li; Yuan Wang; Xinting Hu; Houcheng Jiang; Tao Liang; Yanbin Hao; Guojun Ma; Fuli Feng
>
> **摘要:** Erasing concepts from large-scale text-to-image (T2I) diffusion models has become increasingly crucial due to the growing concerns over copyright infringement, offensive content, and privacy violations. In scalable applications, fine-tuning-based methods are time-consuming to precisely erase multiple target concepts, while real-time editing-based methods often degrade the generation quality of non-target concepts due to conflicting optimization objectives. To address this dilemma, we introduce SPEED, an efficient concept erasure approach that directly edits model parameters. SPEED searches for a null space, a model editing space where parameter updates do not affect non-target concepts, to achieve scalable and precise erasure. To facilitate accurate null space optimization, we incorporate three complementary strategies: Influence-based Prior Filtering (IPF) to selectively retain the most affected non-target concepts, Directed Prior Augmentation (DPA) to enrich the filtered retain set with semantically consistent variations, and Invariant Equality Constraints (IEC) to preserve key invariants during the T2I generation process. Extensive evaluations across multiple concept erasure tasks demonstrate that SPEED consistently outperforms existing methods in non-target preservation while achieving efficient and high-fidelity concept erasure, successfully erasing 100 concepts within only 5 seconds. Our code and models are available at: https://github.com/Ouxiang-Li/SPEED.
>
---
#### [replaced 080] Unstable Unlearning: The Hidden Risk of Concept Resurgence in Diffusion Models
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.08074v3](http://arxiv.org/pdf/2410.08074v3)**

> **作者:** Vinith M. Suriyakumar; Rohan Alur; Ayush Sekhari; Manish Raghavan; Ashia C. Wilson
>
> **备注:** 30 pages, 20 figures
>
> **摘要:** Text-to-image diffusion models rely on massive, web-scale datasets. Training them from scratch is computationally expensive, and as a result, developers often prefer to make incremental updates to existing models. These updates often compose fine-tuning steps (to learn new concepts or improve model performance) with "unlearning" steps (to "forget" existing concepts, such as copyrighted works or explicit content). In this work, we demonstrate a critical and previously unknown vulnerability that arises in this paradigm: even under benign, non-adversarial conditions, fine-tuning a text-to-image diffusion model on seemingly unrelated images can cause it to "relearn" concepts that were previously "unlearned." We comprehensively investigate the causes and scope of this phenomenon, which we term concept resurgence, by performing a series of experiments which compose "concept unlearning" with subsequent fine-tuning of Stable Diffusion v1.4 and Stable Diffusion v2.1. Our findings underscore the fragility of composing incremental model updates, and raise serious new concerns about current approaches to ensuring the safety and alignment of text-to-image diffusion models.
>
---
#### [replaced 081] LiteGS: A High-performance Framework to Train 3DGS in Subminutes via System and Algorithm Codesign
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.01199v3](http://arxiv.org/pdf/2503.01199v3)**

> **作者:** Kaimin Liao; Hua Wang; Zhi Chen; Luchao Wang; Yaohua Tang
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as promising alternative in 3D representation. However, it still suffers from high training cost. This paper introduces LiteGS, a high performance framework that systematically optimizes the 3DGS training pipeline from multiple aspects. At the low-level computation layer, we design a ``warp-based raster'' associated with two hardware-aware optimizations to significantly reduce gradient reduction overhead. At the mid-level data management layer, we introduce dynamic spatial sorting based on Morton coding to enable a performant ``Cluster-Cull-Compact'' pipeline and improve data locality, therefore reducing cache misses. At the top-level algorithm layer, we establish a new robust densification criterion based on the variance of the opacity gradient, paired with a more stable opacity control mechanism, to achieve more precise parameter growth. Experimental results demonstrate that LiteGS accelerates the original 3DGS training by up to 13.4x with comparable or superior quality and surpasses the current SOTA in lightweight models by up to 1.4x speedup. For high-quality reconstruction tasks, LiteGS sets a new accuracy record and decreases the training time by an order of magnitude.
>
---
#### [replaced 082] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.17513v2](http://arxiv.org/pdf/2509.17513v2)**

> **作者:** Zihan Zheng; Zhenlong Wu; Houqiang Zhong; Yuan Tian; Ning Cao; Lan Xu; Jiangchao Yao; Xiaoyun Zhang; Qiang Hu; Wenjun Zhang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Achieving seamless viewing of high-fidelity volumetric video, comparable to 2D video experiences, remains an open challenge. Existing volumetric video compression methods either lack the flexibility to adjust quality and bitrate within a single model for efficient streaming across diverse networks and devices, or struggle with real-time decoding and rendering on lightweight mobile platforms. To address these challenges, we introduce 4DGCPro, a novel hierarchical 4D Gaussian compression framework that facilitates real-time mobile decoding and high-quality rendering via progressive volumetric video streaming in a single bitstream. Specifically, we propose a perceptually-weighted and compression-friendly hierarchical 4D Gaussian representation with motion-aware adaptive grouping to reduce temporal redundancy, preserve coherence, and enable scalable multi-level detail streaming. Furthermore, we present an end-to-end entropy-optimized training scheme, which incorporates layer-wise rate-distortion (RD) supervision and attribute-specific entropy modeling for efficient bitstream generation. Extensive experiments show that 4DGCPro enables flexible quality and multiple bitrate within a single model, achieving real-time decoding and rendering on mobile devices while outperforming existing methods in RD performance across multiple datasets. Project Page: https://mediax-sjtu.github.io/4DGCPro
>
---
#### [replaced 083] DriveAction: A Benchmark for Exploring Human-like Driving Decisions in VLA Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.05667v2](http://arxiv.org/pdf/2506.05667v2)**

> **作者:** Yuhan Hao; Zhengning Li; Lei Sun; Weilong Wang; Naixin Yi; Sheng Song; Caihong Qin; Mofan Zhou; Yifei Zhan; Xianpeng Lang
>
> **备注:** Benchmark: https://huggingface.co/datasets/LiAuto-DriveAction/drive-action
>
> **摘要:** Vision-Language-Action (VLA) models have advanced autonomous driving, but existing benchmarks still lack scenario diversity, reliable action-level annotation, and evaluation protocols aligned with human preferences. To address these limitations, we introduce DriveAction, the first action-driven benchmark specifically designed for VLA models, comprising 16,185 QA pairs generated from 2,610 driving scenarios. DriveAction leverages real-world driving data proactively collected by drivers of autonomous vehicles to ensure broad and representative scenario coverage, offers high-level discrete action labels collected directly from drivers' actual driving operations, and implements an action-rooted tree-structured evaluation framework that explicitly links vision, language, and action tasks, supporting both comprehensive and task-specific assessment. Our experiments demonstrate that state-of-the-art vision-language models (VLMs) require both vision and language guidance for accurate action prediction: on average, accuracy drops by 3.3% without vision input, by 4.1% without language input, and by 8.0% without either. Our evaluation supports precise identification of model bottlenecks with robust and consistent results, thus providing new insights and a rigorous foundation for advancing human-like decisions in autonomous driving.
>
---
#### [replaced 084] Unforgettable Lessons from Forgettable Images: Intra-Class Memorability Matters in Computer Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20761v5](http://arxiv.org/pdf/2412.20761v5)**

> **作者:** Jie Jing; Yongjian Huang; Serena J. -W. Wang; Shuangpeng Han; Lucia Schiatti; Yen-Ling Kuo; Qing Lin; Mengmi Zhang
>
> **摘要:** We introduce intra-class memorability, where certain images within the same class are more memorable than others despite shared category characteristics. To investigate what features make one object instance more memorable than others, we design and conduct human behavior experiments, where participants are shown a series of images, and they must identify when the current image matches the image presented a few steps back in the sequence. To quantify memorability, we propose the Intra-Class Memorability score (ICMscore), a novel metric that incorporates the temporal intervals between repeated image presentations into its calculation. Furthermore, we curate the Intra-Class Memorability Dataset (ICMD), comprising over 5,000 images across ten object classes with their ICMscores derived from 2,000 participants' responses. Subsequently, we demonstrate the usefulness of ICMD by training AI models on this dataset for various downstream tasks: memorability prediction, image recognition, continual learning, and memorability-controlled image editing. Surprisingly, high-ICMscore images impair AI performance in image recognition and continual learning tasks, while low-ICMscore images improve outcomes in these tasks. Additionally, we fine-tune a state-of-the-art image diffusion model on ICMD image pairs with and without masked semantic objects. The diffusion model can successfully manipulate image elements to enhance or reduce memorability. Our contributions open new pathways in understanding intra-class memorability by scrutinizing fine-grained visual features behind the most and least memorable images and laying the groundwork for real-world applications in computer vision. We will release all code, data, and models publicly.
>
---
#### [replaced 085] DOTA: Distributional Test-Time Adaptation of Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2409.19375v3](http://arxiv.org/pdf/2409.19375v3)**

> **作者:** Zongbo Han; Jialong Yang; Guangyu Wang; Junfan Li; Qianli Xu; Mike Zheng Shou; Changqing Zhang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Vision-language foundation models (VLMs), such as CLIP, exhibit remarkable performance across a wide range of tasks. However, deploying these models can be unreliable when significant distribution gaps exist between training and test data, while fine-tuning for diverse scenarios is often costly. Cache-based test-time adapters offer an efficient alternative by storing representative test samples to guide subsequent classifications. Yet, these methods typically employ naive cache management with limited capacity, leading to severe catastrophic forgetting when samples are inevitably dropped during updates. In this paper, we propose DOTA (DistributiOnal Test-time Adaptation), a simple yet effective method addressing this limitation. Crucially, instead of merely memorizing individual test samples, DOTA continuously estimates the underlying distribution of the test data stream. Test-time posterior probabilities are then computed using these dynamically estimated distributions via Bayes' theorem for adaptation. This distribution-centric approach enables the model to continually learn and adapt to the deployment environment. Extensive experiments validate that DOTA significantly mitigates forgetting and achieves state-of-the-art performance compared to existing methods.
>
---
#### [replaced 086] GeoDANO: Geometric VLM with Domain Agnostic Vision Encoder
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11360v2](http://arxiv.org/pdf/2502.11360v2)**

> **作者:** Seunghyuk Cho; Zhenyue Qin; Yang Liu; Youngbin Choi; Seungbeom Lee; Dongwoo Kim
>
> **备注:** Accepted to EMNLP-Findings 2025
>
> **摘要:** We introduce GeoDANO, a geometric vision-language model (VLM) with a domain-agnostic vision encoder, for solving plane geometry problems. Although VLMs have been employed for solving geometry problems, their ability to recognize geometric features remains insufficiently analyzed. To address this gap, we propose a benchmark that evaluates the recognition of visual geometric features, including primitives such as dots and lines, and relations such as orthogonality. Our preliminary study shows that vision encoders often used in general-purpose VLMs, e.g., OpenCLIP, fail to detect these features and struggle to generalize across domains. To overcome the limitation, we develop GeoCLIP, a CLIP-based model trained on synthetic geometric diagram--caption pairs. Benchmark results show that GeoCLIP outperforms existing vision encoders in recognizing geometric features. We then propose our VLM, GeoDANO, which augments GeoCLIP with a domain adaptation strategy for unseen diagram styles. GeoDANO outperforms specialized methods for plane geometry problems and GPT-4o on MathVerse. The implementation is available at https://github.com/ml-postech/GeoDANO.
>
---
#### [replaced 087] Differential-Integral Neural Operator for Long-Term Turbulence Forecasting
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.21196v2](http://arxiv.org/pdf/2509.21196v2)**

> **作者:** Hao Wu; Yuan Gao; Fan Xu; Fan Zhang; Qingsong Wen; Kun Wang; Xiaomeng Huang; Xian Wu
>
> **摘要:** Accurately forecasting the long-term evolution of turbulence represents a grand challenge in scientific computing and is crucial for applications ranging from climate modeling to aerospace engineering. Existing deep learning methods, particularly neural operators, often fail in long-term autoregressive predictions, suffering from catastrophic error accumulation and a loss of physical fidelity. This failure stems from their inability to simultaneously capture the distinct mathematical structures that govern turbulent dynamics: local, dissipative effects and global, non-local interactions. In this paper, we propose the {\textbf{\underline{D}}}ifferential-{\textbf{\underline{I}}}ntegral {\textbf{\underline{N}}}eural {\textbf{\underline{O}}}perator (\method{}), a novel framework designed from a first-principles approach of operator decomposition. \method{} explicitly models the turbulent evolution through parallel branches that learn distinct physical operators: a local differential operator, realized by a constrained convolutional network that provably converges to a derivative, and a global integral operator, captured by a Transformer architecture that learns a data-driven global kernel. This physics-based decomposition endows \method{} with exceptional stability and robustness. Through extensive experiments on the challenging 2D Kolmogorov flow benchmark, we demonstrate that \method{} significantly outperforms state-of-the-art models in long-term forecasting. It successfully suppresses error accumulation over hundreds of timesteps, maintains high fidelity in both the vorticity fields and energy spectra, and establishes a new benchmark for physically consistent, long-range turbulence forecast.
>
---
#### [replaced 088] Draw-In-Mind: Rebalancing Designer-Painter Roles in Unified Multimodal Models Benefits Image Editing
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.01986v2](http://arxiv.org/pdf/2509.01986v2)**

> **作者:** Ziyun Zeng; Junhao Zhang; Wei Li; Mike Zheng Shou
>
> **备注:** Tech Report
>
> **摘要:** In recent years, integrating multimodal understanding and generation into a single unified model has emerged as a promising paradigm. While this approach achieves strong results in text-to-image (T2I) generation, it still struggles with precise image editing. We attribute this limitation to an imbalanced division of responsibilities. The understanding module primarily functions as a translator that encodes user instructions into semantic conditions, while the generation module must simultaneously act as designer and painter, inferring the original layout, identifying the target editing region, and rendering the new content. This imbalance is counterintuitive because the understanding module is typically trained with several times more data on complex reasoning tasks than the generation module. To address this issue, we introduce Draw-In-Mind (DIM), a dataset comprising two complementary subsets: (i) DIM-T2I, containing 14M long-context image-text pairs to enhance complex instruction comprehension; and (ii) DIM-Edit, consisting of 233K chain-of-thought imaginations generated by GPT-4o, serving as explicit design blueprints for image edits. We connect a frozen Qwen2.5-VL-3B with a trainable SANA1.5-1.6B via a lightweight two-layer MLP, and train it on the proposed DIM dataset, resulting in DIM-4.6B-T2I/Edit. Despite its modest parameter scale, DIM-4.6B-Edit achieves SOTA or competitive performance on the ImgEdit and GEdit-Bench benchmarks, outperforming much larger models such as UniWorld-V1 and Step1X-Edit. These findings demonstrate that explicitly assigning the design responsibility to the understanding module provides significant benefits for image editing. Our dataset and models are available at https://github.com/showlab/DIM.
>
---
#### [replaced 089] DriveAgent-R1: Advancing VLM-based Autonomous Driving with Active Perception and Hybrid Thinking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20879v2](http://arxiv.org/pdf/2507.20879v2)**

> **作者:** Weicheng Zheng; Xiaofei Mao; Nanfei Ye; Pengxiang Li; Kun Zhan; Xianpeng Lang; Hang Zhao
>
> **摘要:** The advent of Vision-Language Models (VLMs) has significantly advanced end-to-end autonomous driving, demonstrating powerful reasoning abilities for high-level behavior planning tasks. However, existing methods are often constrained by a passive perception paradigm, relying solely on text-based reasoning. This passivity restricts the model's capacity to actively seek crucial visual evidence when faced with uncertainty. To address this, we introduce DriveAgent-R1, the first autonomous driving agent capable of active perception for planning. In complex scenarios, DriveAgent-R1 proactively invokes tools to perform visual reasoning, firmly grounding its decisions in visual evidence, thereby enhancing both interpretability and reliability. Furthermore, we propose a hybrid thinking framework, inspired by human driver cognitive patterns, allowing the agent to adaptively switch between efficient text-only reasoning and robust tool-augmented visual reasoning based on scene complexity. This capability is cultivated through a three-stage progressive training strategy, featuring a core Cascaded Reinforcement Learning (Cascaded RL) phase. Extensive experiments on the Drive-Internal dataset, which is rich in long-tail scenarios, and the public nuScenes dataset show that, with only 3B parameters, DriveAgent-R1 achieves competitive performance comparable to top closed model systems such as GPT-5 and to human driving proficiency while remaining deployment-friendly, offering a proven path toward building more intelligent autonomous driving systems.
>
---
#### [replaced 090] Diffence: Fencing Membership Privacy With Diffusion Models
- **分类: cs.CR; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2312.04692v3](http://arxiv.org/pdf/2312.04692v3)**

> **作者:** Yuefeng Peng; Ali Naseh; Amir Houmansadr
>
> **备注:** NDSS 2025
>
> **摘要:** Deep learning models, while achieving remarkable performances, are vulnerable to membership inference attacks (MIAs). Although various defenses have been proposed, there is still substantial room for improvement in the privacy-utility trade-off. In this work, we introduce a novel defense framework against MIAs by leveraging generative models. The key intuition of our defense is to remove the differences between member and non-member inputs, which is exploited by MIAs, by re-generating input samples before feeding them to the target model. Therefore, our defense, called DIFFENCE, works pre inference, which is unlike prior defenses that are either training-time or post-inference time. A unique feature of DIFFENCE is that it works on input samples only, without modifying the training or inference phase of the target model. Therefore, it can be cascaded with other defense mechanisms as we demonstrate through experiments. DIFFENCE is designed to preserve the model's prediction labels for each sample, thereby not affecting accuracy. Furthermore, we have empirically demonstrated it does not reduce the usefulness of confidence vectors. Through extensive experimentation, we show that DIFFENCE can serve as a robust plug-n-play defense mechanism, enhancing membership privacy without compromising model utility. For instance, DIFFENCE reduces MIA accuracy against an undefended model by 15.8\% and attack AUC by 14.0\% on average across three datasets, all without impacting model utility. By integrating DIFFENCE with prior defenses, we can achieve new state-of-the-art performances in the privacy-utility trade-off. For example, when combined with the state-of-the-art SELENA defense it reduces attack accuracy by 9.3\%, and attack AUC by 10.0\%. DIFFENCE achieves this by imposing a negligible computation overhead, adding only 57ms to the inference time per sample processed on average.
>
---
#### [replaced 091] Mobi-$π$: Mobilizing Your Robot Learning Policy
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23692v2](http://arxiv.org/pdf/2505.23692v2)**

> **作者:** Jingyun Yang; Isabella Huang; Brandon Vu; Max Bajracharya; Rika Antonova; Jeannette Bohg
>
> **备注:** CoRL 2025. Project website: https://mobipi.github.io/
>
> **摘要:** Learned visuomotor policies are capable of performing increasingly complex manipulation tasks. However, most of these policies are trained on data collected from limited robot positions and camera viewpoints. This leads to poor generalization to novel robot positions, which limits the use of these policies on mobile platforms, especially for precise tasks like pressing buttons or turning faucets. In this work, we formulate the policy mobilization problem: find a mobile robot base pose in a novel environment that is in distribution with respect to a manipulation policy trained on a limited set of camera viewpoints. Compared to retraining the policy itself to be more robust to unseen robot base pose initializations, policy mobilization decouples navigation from manipulation and thus does not require additional demonstrations. Crucially, this problem formulation complements existing efforts to improve manipulation policy robustness to novel viewpoints and remains compatible with them. We propose a novel approach for policy mobilization that bridges navigation and manipulation by optimizing the robot's base pose to align with an in-distribution base pose for a learned policy. Our approach utilizes 3D Gaussian Splatting for novel view synthesis, a score function to evaluate pose suitability, and sampling-based optimization to identify optimal robot poses. To understand policy mobilization in more depth, we also introduce the Mobi-$\pi$ framework, which includes: (1) metrics that quantify the difficulty of mobilizing a given policy, (2) a suite of simulated mobile manipulation tasks based on RoboCasa to evaluate policy mobilization, and (3) visualization tools for analysis. In both our developed simulation task suite and the real world, we show that our approach outperforms baselines, demonstrating its effectiveness for policy mobilization.
>
---
#### [replaced 092] SeamCrafter: Enhancing Mesh Seam Generation for Artist UV Unwrapping via Reinforcement Learning
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20725v2](http://arxiv.org/pdf/2509.20725v2)**

> **作者:** Duoteng Xu; Yuguang Chen; Jing Li; Xinhai Liu; Xueqi Ma; Zhuo Chen; Dongyu Zhang; Chunchao Guo
>
> **摘要:** Mesh seams play a pivotal role in partitioning 3D surfaces for UV parametrization and texture mapping. Poorly placed seams often result in severe UV distortion or excessive fragmentation, thereby hindering texture synthesis and disrupting artist workflows. Existing methods frequently trade one failure mode for another-producing either high distortion or many scattered islands. To address this, we introduce SeamCrafter, an autoregressive GPT-style seam generator conditioned on point cloud inputs. SeamCrafter employs a dual-branch point-cloud encoder that disentangles and captures complementary topological and geometric cues during pretraining. To further enhance seam quality, we fine-tune the model using Direct Preference Optimization (DPO) on a preference dataset derived from a novel seam-evaluation framework. This framework assesses seams primarily by UV distortion and fragmentation, and provides pairwise preference labels to guide optimization. Extensive experiments demonstrate that SeamCrafter produces seams with substantially lower distortion and fragmentation than prior approaches, while preserving topological consistency and visual fidelity.
>
---
#### [replaced 093] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.12976v2](http://arxiv.org/pdf/2501.12976v2)**

> **作者:** Jiahao Wang; Ning Kang; Lewei Yao; Mengzhao Chen; Chengyue Wu; Songyang Zhang; Shuchen Xue; Yong Liu; Taiqiang Wu; Xihui Liu; Kaipeng Zhang; Shifeng Zhang; Wenqi Shao; Zhenguo Li; Ping Luo
>
> **备注:** 20 pages, 14 figures
>
> **摘要:** In this paper, we investigate how to convert a pre-trained Diffusion Transformer (DiT) into a linear DiT, as its simplicity, parallelism, and efficiency for image generation. Through detailed exploration, we offer a suite of ready-to-use solutions, ranging from linear attention design to optimization strategies. Our core contributions include 5 practical guidelines: 1) Applying depth-wise convolution within simple linear attention is sufficient for image generation. 2) Using fewer heads in linear attention provides a free-lunch performance boost without increasing latency. 3) Inheriting weights from a fully converged, pre-trained DiT. 4) Loading all parameters except those related to linear attention. 5) Hybrid knowledge distillation: using a pre-trained teacher DiT to help the training of the student linear DiT, supervising not only the predicted noise but also the variance of the reverse diffusion process. These guidelines lead to our proposed \underline{L}inear D\underline{i}ffusion \underline{T}ransformer (LiT), which serves as a safe and efficient alternative baseline for DiT with pure linear attention. In class-conditional 256$\times$256 and 512$\times$512 ImageNet generation, LiT can be quickly adapted from DiT using only $20\%$ and $33\%$ of DiT's training steps, respectively, while achieving comparable performance. LiT also rivals methods based on Mamba or Gated Linear Attention. Moreover, the same guidelines generalize to text-to-image generation: LiT can be swiftly converted from PixArt-$\Sigma$ to generate high-quality images, maintaining comparable GenEval scores.
>
---
#### [replaced 094] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13019v2](http://arxiv.org/pdf/2507.13019v2)**

> **作者:** Liuyi Wang; Xinyuan Xia; Hui Zhao; Hanqing Wang; Tai Wang; Yilun Chen; Chengju Liu; Qijun Chen; Jiangmiao Pang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent Vision-and-Language Navigation (VLN) advancements are promising, but their idealized assumptions about robot movement and control fail to reflect physically embodied deployment challenges. To bridge this gap, we introduce VLN-PE, a physically realistic VLN platform supporting humanoid, quadruped, and wheeled robots. For the first time, we systematically evaluate several ego-centric VLN methods in physical robotic settings across different technical pipelines, including classification models for single-step discrete action prediction, a diffusion model for dense waypoint prediction, and a train-free, map-based large language model (LLM) integrated with path planning. Our results reveal significant performance degradation due to limited robot observation space, environmental lighting variations, and physical challenges like collisions and falls. This also exposes locomotion constraints for legged robots in complex environments. VLN-PE is highly extensible, allowing seamless integration of new scenes beyond MP3D, thereby enabling more comprehensive VLN evaluation. Despite the weak generalization of current models in physical deployment, VLN-PE provides a new pathway for improving cross-embodiment's overall adaptability. We hope our findings and tools inspire the community to rethink VLN limitations and advance robust, practical VLN models. The code is available at https://crystalsixone.github.io/vln_pe.github.io/.
>
---
#### [replaced 095] $A^2R^2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20890v2](http://arxiv.org/pdf/2507.20890v2)**

> **作者:** Zhecheng Li; Guoxian Song; Yiwei Wang; Zhen Xiong; Junsong Yuan; Yujun Cai
>
> **摘要:** Img2LaTeX is a practically important task that involves translating mathematical expressions and structured visual content from images into LaTeX code. In recent years, vision-language models (VLMs) have achieved remarkable progress across a range of visual understanding tasks, largely due to their strong generalization capabilities. However, despite initial efforts to apply VLMs to the Img2LaTeX task, their performance remains suboptimal. Empirical evidence shows that VLMs can be challenged by fine-grained visual elements, such as subscripts and superscripts in mathematical expressions, which results in inaccurate LaTeX generation. To address this challenge, we propose $A^2R^2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement, a framework that effectively integrates attention localization and iterative refinement within a visual reasoning framework, enabling VLMs to perform self-correction and progressively improve LaTeX generation quality. For effective evaluation, we introduce a new dataset, Img2LaTex-Hard-1K, consisting of 1,100 carefully curated and challenging examples designed to rigorously evaluate the capabilities of VLMs within this task domain. Extensive experimental results demonstrate that: (1) $A^2R^2$ significantly improves model performance across various evaluation metrics spanning both textual and visual levels; (2) Increasing the number of inference rounds yields notable performance gains, underscoring the potential of $A^2R^2$ in test-time scaling scenarios; (3) Ablation studies and further evaluations confirm the effectiveness of our approach and the synergy of its core components during inference.
>
---
#### [replaced 096] Octic Vision Transformers: Quicker ViTs Through Equivariance
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15441v3](http://arxiv.org/pdf/2505.15441v3)**

> **作者:** David Nordström; Johan Edstedt; Fredrik Kahl; Georg Bökman
>
> **摘要:** Why are state-of-the-art Vision Transformers (ViTs) not designed to exploit natural geometric symmetries such as 90-degree rotations and reflections? In this paper, we argue that there is no fundamental reason, and what has been missing is an efficient implementation. To this end, we introduce Octic Vision Transformers (octic ViTs) which rely on octic group equivariance to capture these symmetries. In contrast to prior equivariant models that increase computational cost, our octic linear layers achieve 5.33x reductions in FLOPs and up to 8x reductions in memory compared to ordinary linear layers. In full octic ViT blocks the computational reductions approach the reductions in the linear layers with increased embedding dimension. We study two new families of ViTs, built from octic blocks, that are either fully octic equivariant or break equivariance in the last part of the network. Training octic ViTs supervised (DeiT-III) and unsupervised (DINOv2) on ImageNet-1K, we find that they match baseline accuracy while at the same time providing substantial efficiency gains.
>
---
#### [replaced 097] OS-W2S: An Automatic Labeling Engine for Language-Guided Open-Set Aerial Object Detection
- **分类: cs.CV; cs.DB**

- **链接: [http://arxiv.org/pdf/2505.03334v2](http://arxiv.org/pdf/2505.03334v2)**

> **作者:** Guoting Wei; Yu Liu; Xia Yuan; Xizhe Xue; Linlin Guo; Yifan Yang; Chunxia Zhao; Zongwen Bai; Haokui Zhang; Rong Xiao
>
> **摘要:** In recent years, language-guided open-set aerial object detection has gained significant attention due to its better alignment with real-world application needs. However, due to limited datasets, most existing language-guided methods primarily focus on vocabulary-level descriptions, which fail to meet the demands of fine-grained open-world detection. To address this limitation, we propose constructing a large-scale language-guided open-set aerial detection dataset, encompassing three levels of language guidance: from words to phrases, and ultimately to sentences. Centered around an open-source large vision-language model and integrating image-operation-based preprocessing with BERT-based postprocessing, we present the OS-W2S Label Engine, an automatic annotation pipeline capable of handling diverse scene annotations for aerial images. Using this label engine, we expand existing aerial detection datasets with rich textual annotations and construct a novel benchmark dataset, called MI-OAD, addressing the limitations of current remote sensing grounding data and enabling effective language-guided open-set aerial detection. Specifically, MI-OAD contains 163,023 images and 2 million image-caption pairs, approximately 40 times larger than comparable datasets. To demonstrate the effectiveness and quality of MI-OAD, we evaluate three representative tasks. On language-guided open-set aerial detection, training on MI-OAD lifts Grounding DINO by +31.1 AP$_{50}$ and +34.7 Recall@10 with sentence-level inputs under zero-shot transfer. Moreover, using MI-OAD for pre-training yields state-of-the-art performance on multiple existing open-vocabulary aerial detection and remote sensing visual grounding benchmarks, validating both the effectiveness of the dataset and the high quality of its OS-W2S annotations. More details are available at https://github.com/GT-Wei/MI-OAD.
>
---
#### [replaced 098] A Fully Automatic Framework for Intracranial Pressure Grading: Integrating Keyframe Identification, ONSD Measurement and Clinical Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09368v2](http://arxiv.org/pdf/2509.09368v2)**

> **作者:** Pengxu Wen; Tingting Yu; Ziwei Nie; Cheng Jiang; Zhenyu Yin; Mingyang He; Bo Liao; Xiaoping Yang
>
> **摘要:** Intracranial pressure (ICP) elevation poses severe threats to cerebral function, thus necessitating monitoring for timely intervention. While lumbar puncture is the gold standard for ICP measurement, its invasiveness and associated risks drive the need for non-invasive alternatives. Optic nerve sheath diameter (ONSD) has emerged as a promising biomarker, as elevated ICP directly correlates with increased ONSD. However, current clinical practices for ONSD measurement suffer from inconsistency in manual operation, subjectivity in optimal view selection, and variability in thresholding, limiting their reliability. To address these challenges, we introduce a fully automatic two-stage framework for ICP grading, integrating keyframe identification, ONSD measurement and clinical data. Specifically, the fundus ultrasound video processing stage performs frame-level anatomical segmentation, rule-based keyframe identification guided by an international consensus statement, and precise ONSD measurement. The intracranial pressure grading stage then fuses ONSD metrics with clinical features to enable the prediction of ICP grades, thereby demonstrating an innovative blend of interpretable ultrasound analysis and multi-source data integration for objective clinical evaluation. Experimental results demonstrate that our method achieves a validation accuracy of $0.845 \pm 0.071$ (with standard deviation from five-fold cross-validation) and an independent test accuracy of 0.786, significantly outperforming conventional threshold-based method ($0.637 \pm 0.111$ validation accuracy, $0.429$ test accuracy). Through effectively reducing operator variability and integrating multi-source information, our framework establishes a reliable non-invasive approach for clinical ICP evaluation, holding promise for improving patient management in acute neurological conditions.
>
---
#### [replaced 099] Small Dents, Big Impact: A Dataset and Deep Learning Approach for Vehicle Dent Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.15431v2](http://arxiv.org/pdf/2508.15431v2)**

> **作者:** Danish Zia Baig; Mohsin Kamal; Zahid Ullah
>
> **摘要:** Conventional car damage inspection techniques are labor-intensive, manual, and frequently overlook tiny surface imperfections like microscopic dents. Machine learning provides an innovative solution to the increasing demand for quicker and more precise inspection methods. The paper uses the YOLOv8 object recognition framework to provide a deep learning-based solution for automatically detecting microscopic surface flaws, notably tiny dents, on car exteriors. Traditional automotive damage inspection procedures are manual, time-consuming, and frequently unreliable at detecting tiny flaws. To solve this, a bespoke dataset containing annotated photos of car surfaces under various lighting circumstances, angles, and textures was created. To improve robustness, the YOLOv8m model and its customized variants, YOLOv8m-t4 and YOLOv8m-t42, were trained employing real-time data augmentation approaches. Experimental results show that the technique has excellent detection accuracy and low inference latency, making it suited for real-time applications such as automated insurance evaluations and automobile inspections. Evaluation parameters such as mean Average Precision (mAP), precision, recall, and F1-score verified the model's efficacy. With a precision of 0.86, recall of 0.84, and F1-score of 0.85, the YOLOv8m-t42 model outperformed the YOLOv8m-t4 model (precision: 0.81, recall: 0.79, F1-score: 0.80) in identifying microscopic surface defects. With a little reduced mAP@0.5:0.95 of 0.20, the mAP@0.5 for YOLOv8m-t42 stabilized at 0.60. Furthermore, YOLOv8m-t42's PR curve area was 0.88, suggesting more consistent performance than YOLOv8m-t4 (0.82). YOLOv8m-t42 has greater accuracy and is more appropriate for practical dent detection applications, even though its convergence is slower.
>
---
#### [replaced 100] Deeper Diffusion Models Amplify Bias
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17560v2](http://arxiv.org/pdf/2505.17560v2)**

> **作者:** Shahin Hakemi; Naveed Akhtar; Ghulam Mubashar Hassan; Ajmal Mian
>
> **摘要:** Despite the remarkable performance of generative Diffusion Models (DMs), their internal working is still not well understood, which is potentially problematic. This paper focuses on exploring the important notion of bias-variance tradeoff in diffusion models. Providing a systematic foundation for this exploration, it establishes that at one extreme, the diffusion models may amplify the inherent bias in the training data, and on the other, they may compromise the presumed privacy of the training samples. Our exploration aligns with the memorization-generalization understanding of the generative models, but it also expands further along this spectrum beyond "generalization", revealing the risk of bias amplification in deeper models. Our claims are validated both theoretically and empirically.
>
---
#### [replaced 101] Re-Densification Meets Cross-Scale Propagation: Real-Time Neural Compression of LiDAR Point Clouds
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.20466v2](http://arxiv.org/pdf/2508.20466v2)**

> **作者:** Pengpeng Yu; Haoran Li; Runqing Jiang; Jing Wang; Liang Lin; Yulan Guo
>
> **摘要:** LiDAR point clouds are fundamental to various applications, yet high-precision scans incur substantial storage and transmission overhead. Existing methods typically convert unordered points into hierarchical octree or voxel structures for dense-to-sparse predictive coding. However, the extreme sparsity of geometric details hinders efficient context modeling, thereby limiting their compression performance and speed. To address this challenge, we propose to generate compact features for efficient predictive coding. Our framework comprises two lightweight modules. First, the Geometry Re-Densification Module re-densifies encoded sparse geometry, extracts features at denser scale, and then re-sparsifies the features for predictive coding. This module avoids costly computation on highly sparse details while maintaining a lightweight prediction head. Second, the Cross-scale Feature Propagation Module leverages occupancy cues from multiple resolution levels to guide hierarchical feature propagation. This design facilitates information sharing across scales, thereby reducing redundant feature extraction and providing enriched features for the Geometry Re-Densification Module. By integrating these two modules, our method yields a compact feature representation that provides efficient context modeling and accelerates the coding process. Experiments on the KITTI dataset demonstrate state-of-the-art compression ratios and real-time performance, achieving 26 FPS for encoding/decoding at 12-bit quantization. Code is available at https://github.com/pengpeng-yu/FastPCC.
>
---
#### [replaced 102] LEO-VL: Efficient Scene Representation for Scalable 3D Vision-Language Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09935v2](http://arxiv.org/pdf/2506.09935v2)**

> **作者:** Jiangyong Huang; Xiaojian Ma; Xiongkun Linghu; Yue Fan; Junchao He; Wenxin Tan; Qing Li; Song-Chun Zhu; Yixin Chen; Baoxiong Jia; Siyuan Huang
>
> **备注:** Project page: https://leo-vl.github.io
>
> **摘要:** Developing vision-language models (VLMs) capable of understanding 3D scenes has been a longstanding goal in the 3D-VL community. Despite recent progress, 3D VLMs still fall short of their 2D counterparts in capability and robustness. A key bottleneck is that current scene representations struggle to balance performance and efficiency: competitive performance comes at the cost of heavy token overhead, which in turn hampers the scalability of 3D-VL learning. To address this, we propose the condensed feature grid (CFG), an efficient scene representation featuring significantly reduced token overhead and strong perception capability. Building on CFG, we introduce LEO-VL, a 3D VLM trained on 700k 3D-VL data spanning four real-world indoor domains and five tasks such as captioning and dialogue. To enhance the robustness of 3D VLM, we further propose SceneDPO for post-training, which involves contrasts across answers and scenes. LEO-VL achieves state-of-the-art performance on various 3D QA benchmarks, including SQA3D, MSQA, and Beacon3D. Our extensive experiments highlight the efficiency of our representation, the benefit of task and scene diversity, consistent scaling effects, and the advantages of SceneDPO compared to SFT and GRPO. We hope our findings advance the efficiency, scalability, and robustness of future 3D VLMs.
>
---
#### [replaced 103] video-SALMONN 2: Caption-Enhanced Audio-Visual Large Language Models
- **分类: cs.CV; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.15220v3](http://arxiv.org/pdf/2506.15220v3)**

> **作者:** Changli Tang; Yixuan Li; Yudong Yang; Jimin Zhuang; Guangzhi Sun; Wei Li; Zejun Ma; Chao Zhang
>
> **摘要:** We present video-SALMONN 2, a family of audio-visual large language models that set new state-of-the-art (SOTA) results in video description and question answering (QA). Our core contribution is multi-round direct preference optimisation (MrDPO), paired with a caption-quality objective that jointly rewards completeness and factual accuracy. Unlike standard DPO with a fixed reference policy, MrDPO periodically refreshes the reference by bootstrapping from a newly re-initialised lightweight adapter trained on the latest preferences, avoiding reference staleness and enabling continual improvement. This strategy produces captions that are consistently more detailed and accurate than those from proprietary systems such as GPT-4o and Gemini-1.5 Pro. We further distil these gains by using our model to generate a high-quality video-caption corpus for supervised fine-tuning of new models, transferring benefits beyond captioning to strong performance on complex video-QA tasks. Across widely used audio-visual and visual-only understanding benchmarks (including Video-MME, WorldSense, AVUT, Video-Holmes, DailyOmni, MLVU, and LVBench), our 3B and 7B models achieve SOTA results at comparable scales, while the 72B model surpasses all other open-source systems. Our source code, models, and data are released at \href{https://github.com/bytedance/video-SALMONN-2}{https://github.com/bytedance/video-SALMONN-2}.
>
---
#### [replaced 104] Can Diffusion Models Disentangle? A Theoretical Perspective
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.00220v2](http://arxiv.org/pdf/2504.00220v2)**

> **作者:** Liming Wang; Muhammad Jehanzeb Mirza; Yishu Gong; Yuan Gong; Jiaqi Zhang; Brian H. Tracey; Katerina Placek; Marco Vilela; James R. Glass
>
> **摘要:** This paper presents a novel theoretical framework for understanding how diffusion models can learn disentangled representations. Within this framework, we establish identifiability conditions for general disentangled latent variable models, analyze training dynamics, and derive sample complexity bounds for disentangled latent subspace models. To validate our theory, we conduct disentanglement experiments across diverse tasks and modalities, including subspace recovery in latent subspace Gaussian mixture models, image colorization, image denoising, and voice conversion for speech classification. Additionally, our experiments show that training strategies inspired by our theory, such as style guidance regularization, consistently enhance disentanglement performance.
>
---
#### [replaced 105] UIP2P: Unsupervised Instruction-based Image Editing via Edit Reversibility Constraint
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.15216v2](http://arxiv.org/pdf/2412.15216v2)**

> **作者:** Enis Simsar; Alessio Tonioni; Yongqin Xian; Thomas Hofmann; Federico Tombari
>
> **备注:** Accepted to ICCV'25. Project page: https://uip2p.github.io/
>
> **摘要:** We propose an unsupervised instruction-based image editing approach that removes the need for ground-truth edited images during training. Existing methods rely on supervised learning with triplets of input images, ground-truth edited images, and edit instructions. These triplets are typically generated either by existing editing methods, introducing biases, or through human annotations, which are costly and limit generalization. Our approach addresses these challenges by introducing a novel editing mechanism called Edit Reversibility Constraint (ERC), which applies forward and reverse edits in one training step and enforces alignment in image, text, and attention spaces. This allows us to bypass the need for ground-truth edited images and unlock training for the first time on datasets comprising either real image-caption pairs or image-caption-instruction triplets. We empirically show that our approach performs better across a broader range of edits with high-fidelity and precision. By eliminating the need for pre-existing datasets of triplets, reducing biases associated with current methods, and proposing ERC, our work represents a significant advancement in unblocking scaling of instruction-based image editing.
>
---
#### [replaced 106] VidBridge-R1: Bridging QA and Captioning for RL-based Video Understanding Models with Intermediate Proxy Tasks
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.09079v2](http://arxiv.org/pdf/2506.09079v2)**

> **作者:** Xinlong Chen; Yuanxing Zhang; Yushuo Guan; Weihong Lin; Zekun Wang; Bohan Zeng; Yang Shi; Sihan Yang; Qiang Liu; Pengfei Wan; Liang Wang; Tieniu Tan
>
> **摘要:** The "Reason-Then-Respond" paradigm, enhanced by Reinforcement Learning, has shown great promise in advancing Multimodal Large Language Models. However, its application to the video domain has led to specialized models that excel at either question answering (QA) or captioning tasks, but struggle to master both. Naively combining reward signals from these tasks results in mutual performance degradation, which we attribute to a conflict between their opposing task natures. To address this challenge, we propose a novel training framework built upon two intermediate proxy tasks: DarkEventInfer, which presents videos with masked event segments, requiring models to infer the obscured content based on contextual video cues; and MixVidQA, which presents interleaved video sequences composed of two distinct clips, challenging models to isolate and reason about one while disregarding the other. These proxy tasks compel the model to simultaneously develop both holistic, divergent understanding and precise, convergent reasoning capabilities. Embodying this framework, we present VidBridge-R1, the first versatile video reasoning model that effectively bridges the paradigm conflict. Extensive experiments show that VidBridge-R1 achieves significant performance gains on both QA and captioning within one model, demonstrating the efficacy of our approach in fostering more generalizable and powerful video understanding models.
>
---
#### [replaced 107] Pose-free 3D Gaussian splatting via shape-ray estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22978v2](http://arxiv.org/pdf/2505.22978v2)**

> **作者:** Youngju Na; Taeyeon Kim; Jumin Lee; Kyu Beom Han; Woo Jae Kim; Sung-eui Yoon
>
> **备注:** ICIP 2025 (Best Student Paper Award)
>
> **摘要:** While generalizable 3D Gaussian splatting enables efficient, high-quality rendering of unseen scenes, it heavily depends on precise camera poses for accurate geometry. In real-world scenarios, obtaining accurate poses is challenging, leading to noisy pose estimates and geometric misalignments. To address this, we introduce SHARE, a pose-free, feed-forward Gaussian splatting framework that overcomes these ambiguities by joint shape and camera rays estimation. Instead of relying on explicit 3D transformations, SHARE builds a pose-aware canonical volume representation that seamlessly integrates multi-view information, reducing misalignment caused by inaccurate pose estimates. Additionally, anchor-aligned Gaussian prediction enhances scene reconstruction by refining local geometry around coarse anchors, allowing for more precise Gaussian placement. Extensive experiments on diverse real-world datasets show that our method achieves robust performance in pose-free generalizable Gaussian splatting.
>
---
#### [replaced 108] PhyMAGIC: Physical Motion-Aware Generative Inference with Confidence-guided LLM
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16456v2](http://arxiv.org/pdf/2505.16456v2)**

> **作者:** Siwei Meng; Yawei Luo; Ping Liu
>
> **摘要:** Recent advances in 3D content generation have amplified demand for dynamic models that are both visually realistic and physically consistent. However, state-of-the-art video diffusion models frequently produce implausible results such as momentum violations and object interpenetrations. Existing physics-aware approaches often rely on task-specific fine-tuning or supervised data, which limits their scalability and applicability. To address the challenge, we present PhyMAGIC, a training-free framework that generates physically consistent motion from a single image. PhyMAGIC integrates a pre-trained image-to-video diffusion model, confidence-guided reasoning via LLMs, and a differentiable physics simulator to produce 3D assets ready for downstream physical simulation without fine-tuning or manual supervision. By iteratively refining motion prompts using LLM-derived confidence scores and leveraging simulation feedback, PhyMAGIC steers generation toward physically consistent dynamics. Comprehensive experiments demonstrate that PhyMAGIC outperforms state-of-the-art video generators and physics-aware baselines, enhancing physical property inference and motion-text alignment while maintaining visual fidelity.
>
---
#### [replaced 109] CARE: Confidence-aware Ratio Estimation for Medical Biomarkers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19585v2](http://arxiv.org/pdf/2505.19585v2)**

> **作者:** Jiameng Li; Teodora Popordanoska; Aleksei Tiulpin; Sebastian G. Gruber; Frederik Maes; Matthew B. Blaschko
>
> **备注:** 9 pages
>
> **摘要:** Ratio-based biomarkers -- such as the proportion of necrotic tissue within a tumor -- are widely used in clinical practice to support diagnosis, prognosis, and treatment planning. These biomarkers are typically estimated from soft segmentation outputs by computing region-wise ratios. Despite the high-stakes nature of clinical decision making, existing methods provide only point estimates, offering no measure of uncertainty. In this work, we propose a unified confidence-aware framework for estimating ratio-based biomarkers. Our uncertainty analysis stems from two observations: i) the probability ratio estimator inherently admits a statistical confidence interval regarding local randomness (bias and variance), ii) the segmentation network is not perfectly calibrated. We conduct a systematic analysis of error propagation in the segmentation-to-biomarker pipeline and identify model miscalibration as the dominant source of uncertainty. We leverage tunable parameters to control the confidence level of the derived bounds, allowing adaptation towards clinical practice. Extensive experiments show that our method produces statistically sound confidence intervals, with tunable confidence levels, enabling more trustworthy application of predictive biomarkers in clinical workflows.
>
---
